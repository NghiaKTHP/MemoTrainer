"""
Convert LabelMe instance-segmentation dataset → semantic segmentation masks.

Classes are discovered automatically by scanning all JSON files.
  - background is always id 0
  - foreground classes are assigned ids 1, 2, 3, … in alphabetical order
  - the final mapping is printed and saved to <dst>/classes.json

Source layout:
    Dataset/
        <name>.bmp  (or .jpg/.png)
        <name>.json  (LabelMe format, polygon shapes)

Output layout:
    ConvertedDataset/
        classes.json          {"background":0, "Laser":1, ...}
        train/
            images/   *.png
            masks/    *.png   (uint8, values = class index)
            colormap/ *.png   (RGB visualisation)
        val/
            images/
            masks/
            colormap/

Usage examples:
    python convert_dataset.py
    python convert_dataset.py --src Dataset --dst ConvertedDataset --val-ratio 0.2
    python convert_dataset.py --ignore Point
"""

import argparse
import colorsys
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


# ── color generation ──────────────────────────────────────────────────────────

def _id_to_color(class_id: int) -> tuple[int, int, int]:
    """Map an integer class id to a distinct RGB color (id 0 → black)."""
    if class_id == 0:
        return (0, 0, 0)
    hue = ((class_id - 1) * 0.618033988749895) % 1.0   # golden-ratio spacing
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


# ── class discovery ───────────────────────────────────────────────────────────

def build_label2id(json_files: list[Path], ignore: set[str]) -> dict[str, int]:
    """
    Scan all JSON files, collect every unique label name (excluding ignored
    ones), sort them alphabetically, and assign ids starting from 1.
    background is always 0.
    """
    labels: set[str] = set()
    for jpath in json_files:
        with open(jpath, encoding="utf-8") as f:
            ann = json.load(f)
        for shape in ann.get("shapes", []):
            lbl = shape["label"]
            if lbl not in ignore:
                labels.add(lbl)

    label2id = {"background": 0}
    for idx, name in enumerate(sorted(labels), start=1):
        label2id[name] = idx
    return label2id


# ── polygon fill ──────────────────────────────────────────────────────────────

def _fill_polygon(mask: np.ndarray, points: list, value: int) -> None:
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    pts = np.round(pts).astype(np.int32)
    cv2.fillPoly(mask, [pts], color=value)


# ── single-file conversion ────────────────────────────────────────────────────

def convert_one(json_path: Path, label2id: dict[str, int],
                ignore: set[str],
                priority: dict[str, int]) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (image, mask, n_kept_shapes). n_kept_shapes == 0 → background sample."""
    with open(json_path, encoding="utf-8") as f:
        ann = json.load(f)

    h, w = ann["imageHeight"], ann["imageWidth"]
    mask = np.zeros((h, w), dtype=np.uint8)

    shapes = [s for s in ann.get("shapes", [])
              if s["label"] not in ignore and s["label"] in label2id]
    # Lower priority drawn first, higher priority drawn on top.
    # Classes absent from --priority get rank -1 (drawn first).
    shapes.sort(key=lambda s: priority.get(s["label"], -1))

    for shape in shapes:
        _fill_polygon(mask, shape["points"], label2id[shape["label"]])

    for ext in (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            img_path = candidate
            break
    else:
        raise FileNotFoundError(f"No image found for {json_path.name}")

    image = np.array(Image.open(img_path).convert("RGB"))
    return image, mask, len(shapes)


# ── colorize ──────────────────────────────────────────────────────────────────

def _colorize(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id in np.unique(mask):
        color[mask == class_id] = _id_to_color(int(class_id))
    return color


# ── main ──────────────────────────────────────────────────────────────────────

def run(
    src: str | Path,
    dst: str | Path,
    ignore: list[str] | None = None,
    priority: list[str] | None = None,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """Convert a LabelMe dataset → semantic-seg dataset. Callable from GUI/exe."""
    src    = Path(src)
    dst    = Path(dst)
    ignore = set(ignore or [])
    priority = {name: rank for rank, name in enumerate(priority or ["Laser", "Metal", "Point"])}
    random.seed(seed)

    json_files = sorted(src.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {src}")
    print(f"Found {len(json_files)} samples in {src}")

    # ── discover classes ──────────────────────────────────────────────────────
    label2id = build_label2id(json_files, ignore)
    num_classes = len(label2id)

    print(f"\nClasses ({num_classes}):")
    for name, cid in sorted(label2id.items(), key=lambda x: x[1]):
        r, g, b = _id_to_color(cid)
        print(f"  {cid:2d}  {name:<20s}  color=({r:3d},{g:3d},{b:3d})")
    if ignore:
        print(f"Ignored (→ background): {', '.join(sorted(ignore))}")

    # ── output dirs ───────────────────────────────────────────────────────────
    for split in ("train", "val"):
        for sub in ("images", "masks", "colormap"):
            (dst / split / sub).mkdir(parents=True, exist_ok=True)

    # save class mapping next to the dataset so nothing is ambiguous later
    with open(dst / "classes.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2, ensure_ascii=False)
    print(f"\nClass mapping saved → {dst / 'classes.json'}")

    # ── train / val split ─────────────────────────────────────────────────────
    indices = list(range(len(json_files)))
    random.shuffle(indices)
    n_val   = max(1, int(len(indices) * val_ratio)) if val_ratio > 0 else 0
    val_idx = set(indices[:n_val])

    # ── convert ───────────────────────────────────────────────────────────────
    counts = {"train": 0, "val": 0}
    background_files: list[str] = []   # json exists but no kept polygons → background sample
    errors = []

    for i, jpath in enumerate(tqdm(json_files, desc="Converting")):
        split    = "val" if i in val_idx else "train"
        out_name = jpath.stem + ".png"

        try:
            image, mask, n_shapes = convert_one(jpath, label2id, ignore, priority)
        except Exception as e:
            errors.append((jpath.name, str(e)))
            continue

        Image.fromarray(image).save(dst / split / "images"   / out_name)
        Image.fromarray(mask ).save(dst / split / "masks"    / out_name)
        Image.fromarray(_colorize(mask)).save(dst / split / "colormap" / out_name)
        counts[split] += 1
        if n_shapes == 0:
            background_files.append(jpath.name)

    print(f"\nDone.")
    print(f"  train : {counts['train']} samples")
    print(f"  val   : {counts['val']} samples")
    if background_files:
        print(f"  background (json with no labels): {len(background_files)}")
        for name in background_files:
            print(f"    {name}")
    print(f"  output: {dst.resolve()}")

    if errors:
        print(f"\n[WARN] {len(errors)} file(s) failed:")
        for name, msg in errors:
            print(f"  {name}: {msg}")

    return {
        "train": counts["train"],
        "val": counts["val"],
        "background": len(background_files),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert LabelMe → semantic segmentation dataset (dynamic classes)"
    )
    parser.add_argument("--src", default=r"D:\Nghia\TrainDataset\MetalSheet\PointLaserMetal")
    parser.add_argument("--dst", default=r"D:\Nghia\TrainDataset\MetalSheet\PointLaserMetal_Dino")
    parser.add_argument("--ignore",    nargs="*", default=[])
    parser.add_argument("--priority",  nargs="*", default=["Laser", "Metal", "Point"])
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()
    run(args.src, args.dst, args.ignore, args.priority, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
