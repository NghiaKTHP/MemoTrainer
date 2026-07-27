"""
Convert DinoUperNet .pth -> .onnx voi DYNAMIC BATCH.

- Quet WEIGHTS_DIR tim tat ca file .pth
- Voi moi file: load weight qua memolib DinoUperNet.LoadWeight, ap dung model
  surgery (AdaptiveAvgPool2d -> AvgPool2d) roi torch.onnx.export voi
  dynamic_axes cho batch (H/W van fix vi PPM da bi surgery voi kernel tinh).
- Simplify bang onnxsim neu co.

Note: chi batch (dim 0) la dynamic. H/W phai bang IMAGE_SIZE khi run inference.
"""

import math
import os
import sys
from os import path
from pathlib import Path

import torch
import torch.nn as nn


from MemoLib.Model.DinoUperNet.DinoUperNet import DinoUperNet, _ONNXWrapper
from MemoLib.Model.BaseModel.eSegmentationModel import eDinoUperNetModel


# -- Config --
WEIGHTS_DIR  = r"D:\Nghia\Python-Workspace\Mask2Former\AtrongSol\Weights"

ARCHITECTURE = eDinoUperNetModel.DINO_S
NUM_CLASSES  = 2
IMAGE_SIZE   = 518     # phai chia het cho 168 de output_size == input_size
WEIGHT_GLOB  = "*.pth"

# Dynamic batch export config (H/W FIXED, chi batch dynamic)
DYNAMIC_BATCH = True
OPSET         = 20


def _log(level: str, msg: str):
    print(f"[{level}] {msg}")


def _export_one(m: DinoUperNet, pth_path: str, image_size: int, dynamic_batch: bool):
    """Load 1 weight + export ONNX voi dynamic batch axis."""
    patch_size  = 14
    ppm_divisor = patch_size * 2 * math.lcm(1, 2, 3, 6)  # 168

    if image_size % patch_size != 0:
        image_size = round(image_size / patch_size) * patch_size
        _log("Info", f"[WARN] image_size adjusted to {image_size}")

    internal_size = (image_size // ppm_divisor) * ppm_divisor
    if internal_size == 0:
        internal_size = ppm_divisor
    needs_resize = internal_size != image_size
    if needs_resize:
        _log("Info", f"[INFO] ONNX internal resize {image_size}->{internal_size}")

    temp_model = m.LoadWeight(pth_path)
    try:
        onnx_path = path.splitext(pth_path)[0] + ".onnx"
        wrapper   = _ONNXWrapper(temp_model, resize_to=internal_size if needs_resize else 0)
        dummy     = torch.randn(1, 3, image_size, image_size, device=m.Device)

        # Model surgery: AdaptiveAvgPool2d -> static AvgPool2d.
        ppm_h = internal_size // (patch_size * 2)
        for mod_name, mod in list(wrapper.named_modules()):
            if isinstance(mod, nn.AdaptiveAvgPool2d):
                out_sz = mod.output_size
                scale  = out_sz if isinstance(out_sz, int) else out_sz[0]
                k      = ppm_h // scale
                parts  = mod_name.rsplit(".", 1)
                parent = wrapper
                for attr in (parts[0].split(".") if len(parts) > 1 else []):
                    parent = getattr(parent, attr)
                setattr(parent, parts[-1], nn.AvgPool2d(k, k))

        dyn_axes = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None

        with torch.no_grad():
            torch.onnx.export(
                wrapper, dummy, onnx_path,
                opset_version=OPSET,
                input_names=["input"], output_names=["output"],
                dynamic_axes=dyn_axes,
                do_constant_folding=True, export_params=True,
            )
        _log("Info", f"Exported ONNX: {onnx_path} (dynamic_batch={dynamic_batch})")

        try:
            import onnxsim, onnx as ox
            simp, ok = onnxsim.simplify(ox.load(onnx_path), check_n=3, perform_optimization=True)
            if ok:
                backup = onnx_path.replace(".onnx", "_original.onnx")
                os.replace(onnx_path, backup)
                ox.save(simp, onnx_path)
                _log("Info", "ONNX simplified OK")
        except ImportError:
            pass
        except Exception as e:
            _log("Warning", f"ONNX simplification: {e}")

        return onnx_path
    finally:
        if temp_model is not None and hasattr(temp_model, "cpu"):
            temp_model.cpu()
        del temp_model
        torch.cuda.empty_cache()


def main():
    weights_dir = Path(WEIGHTS_DIR)
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"Weights dir not found: {weights_dir}")

    pths = sorted(weights_dir.glob(WEIGHT_GLOB))
    if not pths:
        print(f"No {WEIGHT_GLOB} in {weights_dir}")
        return

    m = DinoUperNet()
    m.cfg.Architecture = ARCHITECTURE
    m.cfg.NumClasses   = NUM_CLASSES
    m.cfg.ImageSize    = IMAGE_SIZE
    m.ClassesNumber    = NUM_CLASSES
    m.callbacks        = _log

    print(f"Found {len(pths)} weight file(s) | arch={ARCHITECTURE.name} | "
          f"num_classes={NUM_CLASSES} | image_size={IMAGE_SIZE} | "
          f"dynamic_batch={DYNAMIC_BATCH}")

    n_ok = n_fail = 0
    for pth in pths:
        print(f"\n>> Exporting: {pth}")
        try:
            _export_one(m, str(pth), IMAGE_SIZE, DYNAMIC_BATCH)
            n_ok += 1
        except Exception as ex:
            n_fail += 1
            print(f"[FAIL] {pth.name}: {ex}")

    print(f"\nDone. ok={n_ok} | fail={n_fail} | dir={weights_dir}")


if __name__ == "__main__":
    main()
