"""
Convert DinoUperNet ONNX -> OpenVINO IR (.xml + .bin).

Dùng openvino.convert_model trên file .onnx full-pipeline đã export
(không dùng memolib Export(OpenVINO) vì nó chỉ export decode_head — partial).
"""

import sys
from pathlib import Path

import openvino as ov


# ── Config ────────────────────────────────────────────────────────────────
WEIGHTS_DIR = r"D:\Nghia\Python-Workspace\MemoTrainer\TrainResult\TestAThuan_20260616_235927\Weights"

# Skip các file _original.onnx (chưa simplify) — chỉ convert bản đã onnxsim
SKIP_SUFFIXES = ("_original",)

# Precision lưu .bin: "FP32" hoặc "FP16" (FP16 nhỏ ~½, tốc độ tốt hơn trên GPU/NPU)
COMPRESS_TO_FP16 = False

OVERWRITE = True


# ── Main ──────────────────────────────────────────────────────────────────
def convert_one(onnx_path: Path) -> Path:
    xml_path = onnx_path.with_suffix(".xml")
    if xml_path.exists() and not OVERWRITE:
        print(f"  SKIP (exists): {xml_path.name}")
        return xml_path

    print(f"  -> {onnx_path.name}")
    model = ov.convert_model(str(onnx_path))
    ov.save_model(model, str(xml_path), compress_to_fp16=COMPRESS_TO_FP16)
    print(f"     saved: {xml_path.name} + {xml_path.with_suffix('.bin').name}")
    return xml_path


def main():
    weights_dir = Path(WEIGHTS_DIR)
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"Weights dir not found: {weights_dir}")

    onnx_files = [
        p for p in sorted(weights_dir.glob("*.onnx"))
        if not any(p.stem.endswith(suf) for suf in SKIP_SUFFIXES)
    ]
    if not onnx_files:
        print(f"No .onnx (non-_original) in {weights_dir}")
        return

    print(f"OpenVINO: {ov.__version__}")
    print(f"Found {len(onnx_files)} ONNX file(s) | FP16={COMPRESS_TO_FP16}")

    n_ok = n_fail = 0
    for p in onnx_files:
        try:
            convert_one(p)
            n_ok += 1
        except Exception as ex:
            n_fail += 1
            print(f"  [FAIL] {p.name}: {ex}")

    print(f"\nDone. ok={n_ok} | fail={n_fail} | dir={weights_dir}")


if __name__ == "__main__":
    main()
