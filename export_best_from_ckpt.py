"""Standalone: export the BEST checkpoint of an existing training run to
inference_model_best.{onnx,xml,bin} without retraining.

Usage:
    venv\\Scripts\\python.exe export_best_from_ckpt.py <TrainResult_run_folder>

Example:
    ... export_best_from_ckpt.py TrainResult\\MetalSheetC-Clip_20260701_082659
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Nghia\Python-Workspace\MemoLibV2")


def parse_training_log(log_path: str) -> dict:
    """Read Architecture / Task / Resolution / Classes from training.log."""
    info = {}
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if ":" not in line:
                continue
            k, _, v = line.partition(":")
            info[k.strip()] = v.strip()
    return info


def export_best(run_dir: str):
    run_dir = os.path.abspath(run_dir)
    log_path = os.path.join(run_dir, "training.log")
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"training.log not found in {run_dir}")

    meta = parse_training_log(log_path)
    task_type = meta["Task"]          # e.g. "segment"
    arch_name = meta["Architecture"]  # e.g. "RFDETRSmall"
    resolution = int(meta["Resolution"])
    num_classes = int(meta.get("Classes", "0"))

    print(f"Run dir     : {run_dir}")
    print(f"Task        : {task_type}")
    print(f"Architecture: {arch_name}")
    print(f"Resolution  : {resolution}")
    print(f"Classes     : {num_classes}")

    # Locate best checkpoint (parent of `checkpoints/` in rfdetr layout)
    best_pth = None
    for name in ("checkpoint_best_total.pth", "checkpoint_best_regular.pth"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            best_pth = p
            break
    if best_pth is None:
        raise FileNotFoundError(f"No checkpoint_best_*.pth in {run_dir}")
    print(f"Best ckpt   : {best_pth}")

    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Build rfdetr model class map
    from rfdetr import (
        RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge,
        RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge,
    )
    det_map = {
        "RFDETRNano": RFDETRNano, "RFDETRSmall": RFDETRSmall,
        "RFDETRMedium": RFDETRMedium, "RFDETRLarge": RFDETRLarge,
    }
    seg_map = {
        "RFDETRNano": RFDETRSegNano, "RFDETRSmall": RFDETRSegSmall,
        "RFDETRMedium": RFDETRSegMedium, "RFDETRLarge": RFDETRSegLarge,
    }
    class_map = seg_map if task_type == "segment" else det_map
    model_cls = class_map[arch_name]

    print(f"Instantiating {model_cls.__name__} with pretrain_weights={os.path.basename(best_pth)} ...")
    model = model_cls(pretrain_weights=best_pth)

    # Align resolution (mirrors what RFDETRModel does before export)
    mc = model.model_config
    derived_pe = mc.resolution // mc.patch_size
    if mc.positional_encoding_size == derived_pe:
        mc.positional_encoding_size = resolution // mc.patch_size
    mc.resolution = resolution

    # ── Export ONNX ──
    # torch.onnx (torchscript path) can't lower `aten::_upsample_bicubic2d_aa`
    # which dinov2 backbone uses in interpolate_pos_encoding. Force
    # antialias=False during export by patching F.interpolate.
    import torch.nn.functional as F
    _orig_interpolate = F.interpolate

    def _no_aa_interpolate(*args, **kwargs):
        if kwargs.get("antialias"):
            kwargs["antialias"] = False
        return _orig_interpolate(*args, **kwargs)

    F.interpolate = _no_aa_interpolate
    try:
        print("Exporting ONNX ...")
        model.export(
            output_dir=checkpoints_dir,
            shape=(resolution, resolution),
            batch_size=1,
            opset_version=17,
            verbose=False,
        )
    finally:
        F.interpolate = _orig_interpolate

    # rfdetr names file rfdetr-<size>.onnx → rename to inference_model_best.onnx
    from glob import glob
    produced = [c for c in glob(os.path.join(checkpoints_dir, "*.onnx"))
                if os.path.basename(c) not in ("inference_model.onnx",
                                                "inference_model_last.onnx",
                                                "inference_model_best.onnx")]
    if not produced:
        # rfdetr may have written inference_model.onnx (unlikely but handle it)
        candidate = os.path.join(checkpoints_dir, "inference_model.onnx")
        if os.path.exists(candidate):
            produced = [candidate]
    if not produced:
        raise FileNotFoundError(f"ONNX export produced no file in {checkpoints_dir}")
    src_onnx = max(produced, key=os.path.getmtime)
    onnx_best = os.path.join(checkpoints_dir, "inference_model_best.onnx")
    os.replace(src_onnx, onnx_best)
    print(f"  -> {onnx_best}")

    # ── Export OpenVINO ──
    print("Exporting OpenVINO ...")
    try:
        import openvino as ov
    except ImportError:
        print("  openvino not installed, skipping.")
        return

    import logging
    for lg_name in ("torch.export", "torch.export.pt2_archive._package"):
        logging.getLogger(lg_name).setLevel(logging.ERROR)

    ov_model = ov.convert_model(onnx_best)
    xml_best = os.path.join(checkpoints_dir, "inference_model_best.xml")
    ov.save_model(ov_model, xml_best)
    print(f"  -> {xml_best}")
    print(f"  -> {os.path.join(checkpoints_dir, 'inference_model_best.bin')}")

    print("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    export_best(sys.argv[1])
