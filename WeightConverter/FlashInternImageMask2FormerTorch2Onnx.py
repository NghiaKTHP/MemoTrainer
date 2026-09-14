"""
Convert FlashInternImage-Mask2Former .pth -> .onnx voi STATIC BATCH.

- Quet WEIGHTS_DIR tim tat ca file .pth.
- Voi moi file: instantiate model, load weight, export ONNX qua wrapper.Export()
  (wrapper tu apply DCNv4 -> grid_sample patch va bake static batch).
- Simplify bang onnxsim neu co.

Notes:
- **STATIC batch** (khac DinoV3ConvNextMask2Former). Ly do (per DCNv4
  upstream export_onnx.py): Mask2Former head bake `batch = len(img_metas)`
  vao trace → dynamic batch se collapse ve [1,3,H,W]. Muon batch != 1
  chi cach export nhieu ONNX voi cac batch khac nhau.
- Batch lay tu `cfg.OnnxDynamicBatchMax` (mac dinh 1). Set = N truoc khi
  chay de bake batch=N vao ONNX.
- IMAGE_SIZE phai chia het cho 32 (backbone total stride).
- Dung dynamo=False + opset 17 vi DCNv4 grid_sample patch co dynamic
  reshape → dynamo exporter fail.
"""

import os
import sys
from os import path
from pathlib import Path

import torch

from MemoLib.Model.FlashInternImageMask2Former.FlashInternImageMask2Former import (
    FlashInternImageMask2Former,
)
from MemoLib.Model.FlashInternImageMask2Former.FlashInternImageMask2FormerConfig import (
    TrainingConfig,
)
from MemoLib.Model.BaseModel.eSegmentationModel import eFlashInternImageMask2FormerModel
from MemoLib.Model.BaseModel.eModelBase import eModelExportType


# -- Config --
WEIGHTS_DIR  = r"E:\TempData\LG_FPCB\VisionMaster"

ARCHITECTURE  = eFlashInternImageMask2FormerModel.FLASH_T
NUM_CLASSES   = 3
IMAGE_SIZE    = 640            # phai chia het cho 32
WEIGHT_GLOB   = "*.pth"

# Static batch baked vao ONNX. 1 cho realtime; 2/4/8 cho throughput. Neu can
# nhieu batch khac nhau, chay script nhieu lan voi ONNX_STATIC_BATCH khac.
ONNX_STATIC_BATCH = 1
OPSET             = 17

# Decode head params (giu default cua config, doi neu training dung khac)
NUM_QUERIES                   = 100
TRANSFORMER_DECODER_LAYERS    = 9
PIXEL_DECODER_ENCODER_LAYERS  = 6
DECODE_FEAT_CHANNELS          = 256
NUM_HEADS                     = 8
NUM_FEATURE_LEVELS            = 3


def _log(level: str, msg: str):
    print(f"[{level}] {msg}")


def _export_one(m: FlashInternImageMask2Former, pth_path: str, image_size: int):
    """Load 1 weight + export ONNX voi static batch tu cfg.OnnxDynamicBatchMax."""
    _PATCH = 32
    if image_size % _PATCH != 0:
        image_size = ((image_size + _PATCH - 1) // _PATCH) * _PATCH
        _log("Info", f"[WARN] image_size adjusted up to {image_size} (multiple of {_PATCH})")

    try:
        # Export() handles: patch DCNv4 -> grid_sample, rebuild CPU, wrap,
        # torch.onnx.export (dynamo=False), + onnx.checker validation.
        onnx_path = m.Export(pth_path,
                             exportType=eModelExportType.ONNX,
                             opset=OPSET)
        _log("Info", f"Exported ONNX: {onnx_path} (static batch={m.cfg.OnnxDynamicBatchMax})")

        # Optional: onnxsim simplify (usually just re-runs constant folding).
        try:
            import onnxsim, onnx as ox
            simp, ok = onnxsim.simplify(ox.load(onnx_path), check_n=3,
                                        perform_optimization=True)
            if ok:
                backup = onnx_path.replace(".onnx", "_original.onnx")
                os.replace(onnx_path, backup)
                ox.save(simp, onnx_path)
                _log("Info", "ONNX simplified OK")
        except ImportError:
            pass
        except Exception as e:
            _log("Warning", f"ONNX simplification: {e}")

        # Sanity check voi ONNXRuntime CPU — chay 1 dummy forward.
        try:
            import onnxruntime as ort
            import numpy as np
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            B = int(m.cfg.OnnxDynamicBatchMax)
            dummy = np.random.randn(B, 3, image_size, image_size).astype(np.float32)
            out = sess.run(["output"], {"input": dummy})[0]
            _log("Info", f"ORT sanity: out shape {out.shape}, dtype {out.dtype}, "
                          f"class-hist {np.unique(out, return_counts=True)}")
        except ImportError:
            pass
        except Exception as e:
            _log("Warning", f"ORT sanity skipped: {e}")

        return onnx_path
    finally:
        # Wrapper.Export tu restore prev_model + gc; nothing to clean here.
        torch.cuda.empty_cache()


def main():
    weights_dir = Path(WEIGHTS_DIR)
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"Weights dir not found: {weights_dir}")

    pths = sorted(weights_dir.glob(WEIGHT_GLOB))
    if not pths:
        print(f"No {WEIGHT_GLOB} in {weights_dir}")
        return

    cfg = TrainingConfig(
        Architecture=ARCHITECTURE,
        ImageSize=IMAGE_SIZE,
        NumQueries=NUM_QUERIES,
        TransformerDecoderLayers=TRANSFORMER_DECODER_LAYERS,
        PixelDecoderEncoderLayers=PIXEL_DECODER_ENCODER_LAYERS,
        DecodeFeatChannels=DECODE_FEAT_CHANNELS,
        NumHeads=NUM_HEADS,
        NumFeatureLevels=NUM_FEATURE_LEVELS,
        OnnxDynamicBatchMax=ONNX_STATIC_BATCH,
    )
    m = FlashInternImageMask2Former(cfg, num_classes=NUM_CLASSES)
    m.callbacks = _log

    print(f"Found {len(pths)} weight file(s) | arch={ARCHITECTURE.name} | "
          f"num_classes={NUM_CLASSES} | image_size={IMAGE_SIZE} | "
          f"static_batch={ONNX_STATIC_BATCH}")

    n_ok = n_fail = 0
    for pth in pths:
        print(f"\n>> Exporting: {pth}")
        try:
            _export_one(m, str(pth), IMAGE_SIZE)
            n_ok += 1
        except Exception as ex:
            n_fail += 1
            import traceback
            traceback.print_exc()
            print(f"[FAIL] {pth.name}: {ex}")

    print(f"\nDone. ok={n_ok} | fail={n_fail} | dir={weights_dir}")


if __name__ == "__main__":
    main()
