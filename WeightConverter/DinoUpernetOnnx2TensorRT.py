"""
Convert DinoUperNet .onnx -> .trt (TensorRT engine) voi DYNAMIC BATCH profile.

- Quet WEIGHTS_DIR tim tat ca file .onnx (bo qua *_original.onnx).
- Voi moi file: build TensorRT engine voi optimization profile
  min=[1,3,H,W], opt=[OPT_BATCH,3,H,W], max=[MAX_BATCH,3,H,W].
- Chi hoat dong khi .onnx da export voi dynamic_axes cho batch (dim 0).
- H/W FIX bang IMAGE_SIZE (PPM AvgPool2d co kernel tinh).

Yeu cau: pip install tensorrt (matched voi CUDA runtime hien dung).
"""

import sys
from pathlib import Path

import tensorrt as trt


# -- Config --
WEIGHTS_DIR  = r"D:\Nghia\Python-Workspace\Mask2Former\AtrongSol\Weights"
ONNX_GLOB    = "*.onnx"

IMAGE_SIZE   = 518     # phai match voi luc export ONNX

# Batch profile (batch=0-axis dynamic).
MIN_BATCH    = 1
OPT_BATCH    = 10       # nen chinh bang PatchBatchSize thuong dung
MAX_BATCH    = 10

INPUT_NAME   = "input"

# FP16 giup nhanh + tiet kiem VRAM. Doi thanh False neu can accuracy tuyet doi.
USE_FP16     = True

# Workspace 4GB (chinh theo VRAM san co).
WORKSPACE_GB = 4


TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def _log(level: str, msg: str):
    print(f"[{level}] {msg}")


def _build_engine(onnx_path: str, trt_path: str,
                  image_size: int,
                  min_b: int, opt_b: int, max_b: int,
                  use_fp16: bool, workspace_gb: int) -> bool:
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser  = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                _log("Error", f"ONNX parse error #{i}: {parser.get_error(i)}")
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    if use_fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            _log("Info", "FP16 enabled")
        else:
            _log("Warning", "FP16 requested nhung platform khong support -> FP32")

    profile = builder.create_optimization_profile()
    profile.set_shape(
        INPUT_NAME,
        min=(min_b, 3, image_size, image_size),
        opt=(opt_b, 3, image_size, image_size),
        max=(max_b, 3, image_size, image_size),
    )
    config.add_optimization_profile(profile)

    _log("Info", f"Building engine: min={min_b} opt={opt_b} max={max_b} @ {image_size}x{image_size}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        _log("Error", "build_serialized_network returned None")
        return False

    with open(trt_path, "wb") as f:
        f.write(bytes(serialized))
    _log("Info", f"Wrote engine: {trt_path} ({serialized.nbytes / (1 << 20):.1f} MB)")
    return True


def main():
    weights_dir = Path(WEIGHTS_DIR)
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"Weights dir not found: {weights_dir}")

    onnxs = [p for p in sorted(weights_dir.glob(ONNX_GLOB))
             if not p.stem.endswith("_original")]
    if not onnxs:
        print(f"No {ONNX_GLOB} in {weights_dir}")
        return

    print(f"TensorRT {trt.__version__} | Found {len(onnxs)} onnx file(s) | "
          f"batch profile [{MIN_BATCH}/{OPT_BATCH}/{MAX_BATCH}] @ {IMAGE_SIZE}px | "
          f"fp16={USE_FP16}")

    n_ok = n_fail = 0
    for onnx_path in onnxs:
        trt_path = onnx_path.with_suffix(".trt")
        print(f"\n>> Building: {onnx_path.name} -> {trt_path.name}")
        try:
            ok = _build_engine(
                str(onnx_path), str(trt_path),
                IMAGE_SIZE, MIN_BATCH, OPT_BATCH, MAX_BATCH,
                USE_FP16, WORKSPACE_GB,
            )
            if ok:
                n_ok += 1
            else:
                n_fail += 1
        except Exception as ex:
            n_fail += 1
            print(f"[FAIL] {onnx_path.name}: {ex}")

    print(f"\nDone. ok={n_ok} | fail={n_fail} | dir={weights_dir}")


if __name__ == "__main__":
    main()
