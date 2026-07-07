"""
Convert raw classification dataset -> memolib classify format.

Raw layout:
    <raw_root>/<class_name>/*.{jpg,png,bmp,...}

Output layout (memolib PPLCNet / EfficientNet):
    <out_root>/train/<class_name>/*
    <out_root>/val/<class_name>/*

Usage:
    python Raw2PPLCNet.py --src "D:/.../RawData" --dst "D:/.../Classify_MemoLib" \
        --samples 500 --val-ratio 0.2 --mode copy --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: Path):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]


def transfer(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "symlink":
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"Unknown mode: {mode}")


def convert(src_root: Path, dst_root: Path, samples: int, val_ratio: float,
            mode: str, seed: int, exclude: set[str]):
    rng = random.Random(seed)
    classes = sorted([d for d in src_root.iterdir()
                      if d.is_dir() and d.name not in exclude])
    if not classes:
        raise RuntimeError(f"No class subfolders found in {src_root}")

    print(f"Found {len(classes)} classes: {[c.name for c in classes]}")
    print(f"Output: {dst_root}\n")

    total_train = total_val = 0
    for cls_dir in classes:
        imgs = list_images(cls_dir)
        rng.shuffle(imgs)

        take = len(imgs) if samples <= 0 else min(samples, len(imgs))
        picked = imgs[:take]

        n_val = int(round(take * val_ratio))
        n_val = max(1, n_val) if take >= 2 and val_ratio > 0 else n_val
        n_val = min(n_val, take - 1) if take >= 2 else n_val

        val_set = set(picked[:n_val])
        train_set = picked[n_val:]

        for p in train_set:
            transfer(p, dst_root / "train" / cls_dir.name / p.name, mode)
        for p in val_set:
            transfer(p, dst_root / "val" / cls_dir.name / p.name, mode)

        total_train += len(train_set)
        total_val += len(val_set)
        print(f"  [{cls_dir.name:20s}] available={len(imgs):5d}  "
              f"picked={take:5d}  train={len(train_set):5d}  val={len(val_set):5d}")

    print(f"\nDone. Total train={total_train}, val={total_val}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=r"D:\Nghia\Python-Workspace\Mask2Former\Dataset\Classify\RawData", help="Raw dataset root (contains class subfolders)")
    ap.add_argument("--dst", default=r"D:\Nghia\Python-Workspace\Mask2Former\Dataset\Classify\TrainData", help="Output dataset root")
    ap.add_argument("--samples", type=int, default=100,
                    help="Max images per class (<=0 means use all)")
    ap.add_argument("--val-ratio", type=float, default=0.15,
                    help="Fraction of picked images sent to val (0.0-1.0)")
    ap.add_argument("--mode", choices=["copy", "move", "symlink"], default="copy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude", nargs="*", default=["Ignore"],
                    help="Class folder names to skip")
    ap.add_argument("--clean", action="store_true",
                    help="Delete dst before converting")
    return ap.parse_args()


def main():
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)

    if not src.is_dir():
        raise SystemExit(f"src not found: {src}")
    if not 0.0 <= args.val_ratio < 1.0:
        raise SystemExit("--val-ratio must be in [0.0, 1.0)")

    if args.clean and dst.exists():
        print(f"Cleaning {dst} ...")
        shutil.rmtree(dst)

    convert(src, dst, args.samples, args.val_ratio,
            args.mode, args.seed, set(args.exclude))


if __name__ == "__main__":
    main()
