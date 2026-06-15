"""
Convert LabelMe instance-segmentation dataset → YOLO instance-seg format for RFDETR.

Source layout (LabelMe / AnyLabeling):
    Dataset/
        <name>.bmp  (or .jpg / .png)
        <name>.json  (polygon shapes, one shape = one instance)

Output layout (YOLO instance-seg):
    ConvertedDataset/
        data.yaml
        train/
            images/   *.png
            labels/   *.txt   (one line per instance:
                               class_id x1 y1 x2 y2 ... xn yn  — normalized)
        valid/
            images/
            labels/

Usage:
    python Anylabeling2RFDETR.py
    python Anylabeling2RFDETR.py --src Dataset --dst Out --val-ratio 0.15
    python Anylabeling2RFDETR.py --ignore Background
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


# ── class discovery ───────────────────────────────────────────────────────────

def build_label2id(json_files: list[Path], ignore: set[str]) -> dict[str, int]:
    labels: set[str] = set()
    for jpath in json_files:
        with open(jpath, encoding="utf-8") as f:
            ann = json.load(f)
        for shape in ann.get("shapes", []):
            lbl = shape["label"]
            if lbl not in ignore:
                labels.add(lbl)
    return {name: idx for idx, name in enumerate(sorted(labels))}


# ── single-file conversion ────────────────────────────────────────────────────

def convert_one(
    json_path: Path,
    label2id: dict[str, int],
    ignore: set[str],
) -> tuple[np.ndarray, list[str]]:
    with open(json_path, encoding="utf-8") as f:
        ann = json.load(f)

    h, w = ann["imageHeight"], ann["imageWidth"]
    lines: list[str] = []

    for shape in ann.get("shapes", []):
        label = shape["label"]
        if label in ignore or label not in label2id:
            continue
        if shape.get("shape_type") != "polygon":
            continue

        class_id = label2id[label]
        pts = np.array(shape["points"], dtype=np.float32)
        norm = pts / np.array([w, h], dtype=np.float32)
        coords = " ".join(f"{v:.6f}" for v in norm.flatten())
        lines.append(f"{class_id} {coords}")

    for ext in (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            img_path = candidate
            break
    else:
        raise FileNotFoundError(f"No image found for {json_path.name}")

    image = np.array(Image.open(img_path).convert("RGB"))
    return image, lines


# ── data.yaml ─────────────────────────────────────────────────────────────────

def write_yaml(dst: Path, label2id: dict[str, int]) -> None:
    names = [name for name, _ in sorted(label2id.items(), key=lambda x: x[1])]
    lines = [
        f"path: {dst.resolve()}",
        f"train: train/images",
        f"val: valid/images",
        f"nc: {len(names)}",
        f"names: {names}",
    ]
    (dst / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────────────

def run(
    src: str | Path,
    dst: str | Path,
    ignore: list[str] | None = None,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """Convert a LabelMe dataset → YOLO instance-seg dataset. Callable from GUI/exe."""
    src    = Path(src)
    dst    = Path(dst)
    ignore = set(ignore or [])
    random.seed(seed)

    json_files = sorted(src.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {src}")
    print(f"Found {len(json_files)} samples in {src}")

    label2id = build_label2id(json_files, ignore)
    if not label2id:
        raise ValueError("No classes found after applying --ignore filter.")

    print(f"\nClasses ({len(label2id)}):")
    for name, cid in sorted(label2id.items(), key=lambda x: x[1]):
        print(f"  {cid}  {name}")
    if ignore:
        print(f"Ignored: {', '.join(sorted(ignore))}")

    for split in ("train", "valid"):
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)

    write_yaml(dst, label2id)
    print(f"\ndata.yaml → {dst / 'data.yaml'}")

    indices = list(range(len(json_files)))
    random.shuffle(indices)
    n_val   = max(1, int(len(indices) * val_ratio)) if val_ratio > 0 else 0
    val_idx = set(indices[:n_val])

    counts = {"train": 0, "valid": 0}
    skipped = 0
    background_files: list[str] = []
    errors: list[tuple[str, str]] = []

    for i, jpath in enumerate(tqdm(json_files, desc="Converting")):
        split    = "valid" if i in val_idx else "train"
        out_stem = jpath.stem

        try:
            image, label_lines = convert_one(jpath, label2id, ignore)
        except Exception as e:
            errors.append((jpath.name, str(e)))
            continue

        Image.fromarray(image).save(dst / split / "images" / f"{out_stem}.png")
        label_content = "\n".join(label_lines) + "\n" if label_lines else ""
        (dst / split / "labels" / f"{out_stem}.txt").write_text(
            label_content, encoding="utf-8"
        )
        if not label_lines:
            skipped += 1
            background_files.append(jpath.name)
        counts[split] += 1

    print(f"\nDone.")
    print(f"  train : {counts['train']} samples")
    print(f"  valid : {counts['valid']} samples")
    if skipped:
        print(f"  background (no annotations): {skipped}")
        for name in background_files:
            print(f"    {name}")
    print(f"  output: {dst.resolve()}")

    if errors:
        print(f"\n[WARN] {len(errors)} file(s) failed:")
        for name, msg in errors:
            print(f"  {name}: {msg}")

    return {"train": counts["train"], "valid": counts["valid"],
            "skipped": skipped, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LabelMe → YOLO instance-seg dataset for RFDETR"
    )
    parser.add_argument("--src", default=r"D:\Nghia\TrainDataset\MetalSheet\Point-InstanceSeg")
    parser.add_argument("--dst", default=r"D:\Nghia\TrainDataset\MetalSheet\Point-InstanceSeg_RFDETR")
    parser.add_argument("--ignore",    nargs="*", default=[])
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()
    run(args.src, args.dst, args.ignore, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
