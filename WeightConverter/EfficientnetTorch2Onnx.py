"""
Convert EfficientNet .pth -> .onnx voi DYNAMIC BATCH.

- Quet WEIGHTS_DIR tim tat ca file .pth
- Voi moi file: tu dong detect architecture tu checkpoint, load weight,
  roi torch.onnx.export voi dynamic_axes cho batch (H/W fix).
- Simplify bang onnxsim neu co.

Note: chi batch (dim 0) la dynamic. H/W phai bang IMAGE_SIZE khi run inference.
"""

import os
import re
import sys
from os import path
from pathlib import Path

import torch

from MemoLib.Model.Efficientnet import EfficientNet
from MemoLib.Model.BaseModel.eClassificationModel import eEfficientNetModel


# -- Config --
WEIGHTS_DIR   = r"D:\Nghia\Python-Workspace\MemoTrainer\TrainResult"

# Fallback neu auto-detect that bai (e.g. V2 variants)
ARCHITECTURE  = eEfficientNetModel.B0
NUM_CLASSES   = 2
IMAGE_SIZE    = 224
WEIGHT_GLOB   = "*.pth"

# Dynamic batch export config (H/W FIXED, chi batch dynamic)
DYNAMIC_BATCH = True
OPSET         = 17


# -- Architecture auto-detection from state_dict --
_IN_FEATS_MAP = {
    1408: eEfficientNetModel.B2,
    1536: eEfficientNetModel.B3,
    1792: eEfficientNetModel.B4,
    2048: eEfficientNetModel.B5,
    2304: eEfficientNetModel.B6,
    2560: eEfficientNetModel.B7,
}


def _detect_arch(state_dict: dict):
    """Return (arch, num_classes) detected from state_dict, or (None, num_classes) if unknown."""
    clf_w = state_dict.get("classifier.1.weight")
    if clf_w is None:
        return None, None

    num_classes = int(clf_w.shape[0])
    in_features = int(clf_w.shape[1])

    if in_features in _IN_FEATS_MAP:
        return _IN_FEATS_MAP[in_features], num_classes

    if in_features == 1280:
        # B0/B1 and V2S/V2M/V2L all use 1280.
        # B-series stage-1 uses MBConv (has SE layer -> block.1.fc1.weight).
        # V2-series stage-1 uses FusedMBConv (no SE -> no fc1 key in stage 1).
        is_b_series = "features.1.0.block.1.fc1.weight" in state_dict
        if not is_b_series:
            # V2 variant — user must set ARCHITECTURE manually as fallback
            return None, num_classes

        # Distinguish B0 (1 block in stage 1, max index=0) vs B1 (2 blocks, max index=1)
        stage1_max = max(
            (int(re.match(r"features\.1\.(\d+)\.", k).group(1))
             for k in state_dict if re.match(r"features\.1\.(\d+)\.", k)),
            default=0,
        )
        arch = eEfficientNetModel.B0 if stage1_max == 0 else eEfficientNetModel.B1
        return arch, num_classes

    return None, num_classes


def _log(level: str, msg: str):
    print(f"[{level}] {msg}")


def _export_one(m: EfficientNet, pth_path: str, image_size: int, dynamic_batch: bool):
    """Load 1 weight + export ONNX voi dynamic batch axis."""
    raw_sd = torch.load(pth_path, map_location=m.Device, weights_only=True)
    arch, num_classes = _detect_arch(raw_sd)

    if arch is not None:
        if arch != m.cfg.Architecture or (num_classes is not None and num_classes != m.ClassesNumber):
            _log("Info", f"Auto-detected: arch={arch.name}, num_classes={num_classes} "
                         f"(config had {m.cfg.Architecture.name}/{m.ClassesNumber})")
        m.cfg.Architecture = arch
        m.ClassesNumber    = num_classes
    else:
        _log("Warning", f"Cannot auto-detect arch from '{path.basename(pth_path)}' — "
                        f"using configured {m.cfg.Architecture.name}/{m.ClassesNumber}. "
                        f"For V2 variants set ARCHITECTURE manually.")
        if num_classes is not None:
            m.ClassesNumber = num_classes

    temp_model = m.LoadWeight(pth_path)
    try:
        onnx_path = path.splitext(pth_path)[0] + ".onnx"
        dummy     = torch.randn(1, 3, image_size, image_size, device=m.Device)
        dyn_axes  = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None

        with torch.no_grad():
            torch.onnx.export(
                temp_model, dummy, onnx_path,
                opset_version       = OPSET,
                input_names         = ["input"],
                output_names        = ["output"],
                dynamic_axes        = dyn_axes,
                do_constant_folding = True,
                export_params       = True,
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

    pths = sorted(weights_dir.rglob(WEIGHT_GLOB))
    if not pths:
        print(f"No {WEIGHT_GLOB} in {weights_dir}")
        return

    m = EfficientNet()
    m.cfg.Architecture    = ARCHITECTURE
    m.cfg.IsUsePretrained = False
    m.ClassesNumber       = NUM_CLASSES
    m.callbacks           = _log

    print(f"Found {len(pths)} weight file(s) | image_size={IMAGE_SIZE} | dynamic_batch={DYNAMIC_BATCH}")

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
