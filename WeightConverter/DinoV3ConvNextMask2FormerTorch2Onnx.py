"""
Convert DinoV3ConvNextMask2Former .pth -> .onnx voi DYNAMIC BATCH.

- Quet WEIGHTS_DIR tim tat ca file .pth.
- Voi moi file: instantiate model, load weight, wrap in _ONNXWrapper, export
  ONNX voi dynamic batch axis. Dummy batch = 2 de tracer track batch la
  dynamic dim (batch=1 se bake thanh static).
- Simplify bang onnxsim neu co.

Note:
- Chi batch (dim 0) la dynamic. H/W FIX bang IMAGE_SIZE (do M2F pixel decoder
  co `int(H)`/`int(W)` bake vao trace).
- IMAGE_SIZE phai chia het cho 32 (ConvNeXt total stride).
- Do model kha nang trace bug: dung dynamo=False + opset 17.
"""

import os
import sys
from os import path
from pathlib import Path

import torch

from MemoLib.Model.DinoV3ConvNextMask2Former.DinoV3ConvNextMask2Former import (
    DinoV3ConvNextMask2Former, _ONNXWrapper,
)
from MemoLib.Model.BaseModel.eSegmentationModel import eDinoV3ConvNextMask2FormerModel


# -- Config --
WEIGHTS_DIR  = r"E:\TempData\LG_FPCB\VisionMaster"

ARCHITECTURE = eDinoV3ConvNextMask2FormerModel.CONVNEXT_SMALL
NUM_CLASSES  = 3
IMAGE_SIZE   = 640     # phai chia het cho 32
WEIGHT_GLOB  = "*.pth"

# Dynamic batch export config (H/W FIXED, chi batch dynamic)
DYNAMIC_BATCH = True
OPSET         = 17

# Mask2Former head params (giu default cua config, doi neu training dung khac)
FEAT_CHANNELS                 = 256
NUM_QUERIES                   = 100
NUM_TRANSFORMER_FEAT_LEVELS   = 3
PIXEL_DECODER_ENCODER_LAYERS  = 6
TRANSFORMER_DECODER_LAYERS    = 9
NUM_HEADS                     = 8


def _log(level: str, msg: str):
    print(f"[{level}] {msg}")


def _export_one(m: DinoV3ConvNextMask2Former, pth_path: str,
                image_size: int, dynamic_batch: bool):
    """Load 1 weight + export ONNX voi dynamic batch axis."""
    _PATCH = 32
    if image_size % _PATCH != 0:
        image_size = ((image_size + _PATCH - 1) // _PATCH) * _PATCH
        _log("Info", f"[WARN] image_size adjusted up to {image_size} (multiple of {_PATCH})")

    temp_model = m.LoadWeight(pth_path)
    try:
        onnx_path = path.splitext(pth_path)[0] + ".onnx"
        wrapper   = _ONNXWrapper(temp_model).to(m.Device).eval()

        # Dummy batch = 2 khi dynamic → tracer track dim0 la variable.
        # Neu batch=1, graph se bake static batch=1 -> TRT bao "Static model
        # does not take explicit shapes" khi truyen --min/opt/maxShapes.
        dummy_batch = 2 if dynamic_batch else 1
        dummy = torch.randn(dummy_batch, 3, image_size, image_size, device=m.Device)

        dyn_axes = ({"input": {0: "batch"}, "output": {0: "batch"}}
                    if dynamic_batch else None)

        with torch.no_grad():
            # dynamo=False vi M2F pixel decoder co nhieu `int(shape)`
            # data-dependent -> dynamo exporter fail. Legacy tracer OK.
            torch.onnx.export(
                wrapper, dummy, onnx_path,
                opset_version=OPSET,
                input_names=["input"], output_names=["output"],
                dynamic_axes=dyn_axes,
                do_constant_folding=True, export_params=True,
                dynamo=False,
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

        # Sanity check voi ONNXRuntime CPU
        try:
            import onnxruntime as ort
            import numpy as np
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            with torch.no_grad():
                ref = wrapper(dummy).cpu().numpy()
            out = sess.run(["output"], {"input": dummy.cpu().numpy()})[0]
            mismatch = 100.0 * (ref.astype(np.int64) != out.astype(np.int64)).mean()
            _log("Info", f"ORT sanity: {mismatch:.3f}% pixels differ")
        except ImportError:
            pass
        except Exception as e:
            _log("Warning", f"ORT sanity skipped: {e}")

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

    m = DinoV3ConvNextMask2Former()
    m.cfg.Architecture                = ARCHITECTURE
    m.cfg.ImageSize                   = IMAGE_SIZE
    m.cfg.FeatChannels                = FEAT_CHANNELS
    m.cfg.NumQueries                  = NUM_QUERIES
    m.cfg.NumTransformerFeatLevels    = NUM_TRANSFORMER_FEAT_LEVELS
    m.cfg.PixelDecoderEncoderLayers   = PIXEL_DECODER_ENCODER_LAYERS
    m.cfg.TransformerDecoderLayers    = TRANSFORMER_DECODER_LAYERS
    m.cfg.NumHeads                    = NUM_HEADS
    m.ClassesNumber                   = NUM_CLASSES
    m.callbacks                       = _log

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
            import traceback
            traceback.print_exc()
            print(f"[FAIL] {pth.name}: {ex}")

    print(f"\nDone. ok={n_ok} | fail={n_fail} | dir={weights_dir}")


if __name__ == "__main__":
    main()
