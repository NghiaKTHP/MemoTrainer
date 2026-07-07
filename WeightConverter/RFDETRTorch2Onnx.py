"""
Convert RFDETR Lightning checkpoint (.ckpt) -> ONNX.

Load a PyTorch-Lightning .ckpt produced by RFDETRModel training, copy weights
into a fresh rfdetr model (with matching arch / num_classes / resolution),
then call rfdetr's built-in `.export()` to emit `inference_model.onnx` next
to the checkpoint.
"""

import sys
from pathlib import Path

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ── Allow importing memolib ───────────────────────────────────────────────
MEMOLIB_ROOT = r"D:\Nghia\Python-Workspace\MemoLibV2"
if MEMOLIB_ROOT not in sys.path:
    sys.path.insert(0, MEMOLIB_ROOT)

from rfdetr import (
    RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge,
    RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge,
)


# ── Config ────────────────────────────────────────────────────────────────
CKPT_PATH    = r"D:\Nghia\Python-Workspace\MemoTrainer\TrainResult\MetalSheetC-Clip_20260610_163835\last.ckpt"
ARCH         = "RFDETRNano"   # RFDETRNano | RFDETRSmall | RFDETRMedium | RFDETRLarge
TASK         = "detect"       # detect | segment
NUM_CLASSES  = 2
IMAGE_SIZE   = 384            # phải chia hết cho 32
OUTPUT_DIR   = None           # None -> cùng folder với .ckpt


_DET_MAP = {
    "RFDETRNano":   RFDETRNano,
    "RFDETRSmall":  RFDETRSmall,
    "RFDETRMedium": RFDETRMedium,
    "RFDETRLarge":  RFDETRLarge,
}
_SEG_MAP = {
    "RFDETRNano":   RFDETRSegNano,
    "RFDETRSmall":  RFDETRSegSmall,
    "RFDETRMedium": RFDETRSegMedium,
    "RFDETRLarge":  RFDETRSegLarge,
}


def _strip_prefix(state_dict: dict, prefix: str = "model.") -> dict:
    out = {}
    for k, v in state_dict.items():
        out[k[len(prefix):] if k.startswith(prefix) else k] = v
    return out


def convert(ckpt_path: str, arch: str, task: str, num_classes: int,
            image_size: int, output_dir: str | None = None) -> str:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if image_size % 32 != 0:
        new_size = ((image_size + 31) // 32) * 32
        print(f"[WARN] image_size={image_size} không chia hết cho 32 -> {new_size}")
        image_size = new_size

    cls_map = _SEG_MAP if task == "segment" else _DET_MAP
    if arch not in cls_map:
        raise ValueError(f"Unknown arch '{arch}'. Expected {list(cls_map)}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Lightning prefixes inner LWDETR keys with "model."
    raw_state = _strip_prefix(state_dict, prefix="model.")

    print(f"Building {arch} (task={task}, num_classes={num_classes}, resolution={image_size})")
    model = cls_map[arch](
        pretrain_weights=None,
        num_classes=num_classes,
        resolution=image_size,
    )

    inner = model.model.model  # RFDETR -> ModelContext -> LWDETR
    missing, unexpected = inner.load_state_dict(raw_state, strict=False)
    if missing:
        print(f"[WARN] {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"[WARN] {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")
    if not missing and not unexpected:
        print("Weights loaded (strict match).")

    out_dir = Path(output_dir) if output_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting ONNX -> {out_dir}")
    model.export(
        output_dir=str(out_dir),
        shape=(image_size, image_size),
        batch_size=1,
        verbose=False,
    )

    final = out_dir / f"{ckpt_path.stem}.onnx"
    candidates = [p for p in out_dir.glob("*.onnx") if p != final]
    if not candidates:
        raise FileNotFoundError(f"ONNX file not produced in {out_dir}")
    produced = max(candidates, key=lambda p: p.stat().st_mtime)
    if final.exists():
        final.unlink()
    produced.rename(final)

    print(f"Done: {final}")
    return str(final)


def main():
    convert(CKPT_PATH, ARCH, TASK, NUM_CLASSES, IMAGE_SIZE, OUTPUT_DIR)


if __name__ == "__main__":
    main()
