"""
RFDETR inference script (detection + segmentation).

Loads a trained RFDETR checkpoint (.pth) and runs prediction on an image,
a folder of images, or a video. Saves annotated outputs and optionally
dumps raw predictions to JSON.

Usage:
    python InferRFDETR.py --weights run/checkpoints/checkpoint_best_total.pth ^
                         --source D:\\images ^
                         --out    D:\\images_out ^
                         --arch   RFDETRNano ^
                         --task   detect ^
                         --conf   0.5

Architectures: RFDETRNano | RFDETRSmall | RFDETRMedium | RFDETRLarge
Tasks:         detect | segment
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from rfdetr import (
    RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge,
    RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge,
)


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

_IMG_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
_VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def load_model(weights: str, arch: str, task: str, resolution: int | None):
    cls_map = _SEG_MAP if task == "segment" else _DET_MAP
    if arch not in cls_map:
        raise ValueError(f"Unknown architecture '{arch}'. Expected one of {list(cls_map)}")

    kwargs = {"pretrain_weights": weights}
    if resolution is not None:
        kwargs["resolution"] = resolution
    model = cls_map[arch](**kwargs)
    return model


def _color(idx: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(idx * 9973 + 1)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


def _draw(img_bgr: np.ndarray, detections, class_names: list[str] | None) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]

    boxes      = getattr(detections, "xyxy",       None)
    scores     = getattr(detections, "confidence", None)
    class_ids  = getattr(detections, "class_id",   None)
    masks      = getattr(detections, "mask",       None)

    if boxes is None:
        return out

    n = len(boxes)
    for i in range(n):
        cid   = int(class_ids[i]) if class_ids is not None else -1
        score = float(scores[i])  if scores    is not None else 0.0
        color = _color(cid if cid >= 0 else i)

        if masks is not None and i < len(masks):
            m = masks[i]
            if m.shape[:2] != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            overlay = out.copy()
            overlay[m] = color
            out = cv2.addWeighted(overlay, 0.5, out, 0.5, 0)

        x1, y1, x2, y2 = [int(v) for v in boxes[i]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        name  = class_names[cid] if (class_names and 0 <= cid < len(class_names)) else f"id{cid}"
        label = f"{name} {score:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - bl - 2), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, label, (x1 + 1, y1 - bl - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _detections_to_dict(detections, class_names: list[str] | None) -> list[dict]:
    boxes     = getattr(detections, "xyxy",       None)
    scores    = getattr(detections, "confidence", None)
    class_ids = getattr(detections, "class_id",   None)
    if boxes is None:
        return []
    out = []
    for i in range(len(boxes)):
        cid = int(class_ids[i]) if class_ids is not None else -1
        item = {
            "bbox":  [float(v) for v in boxes[i]],
            "score": float(scores[i]) if scores is not None else 0.0,
            "class_id":   cid,
            "class_name": class_names[cid] if (class_names and 0 <= cid < len(class_names)) else None,
        }
        out.append(item)
    return out


def _collect_inputs(source: Path) -> tuple[list[Path], list[Path]]:
    if source.is_file():
        ext = source.suffix.lower()
        if ext in _IMG_EXTS:
            return [source], []
        if ext in _VID_EXTS:
            return [], [source]
        raise ValueError(f"Unsupported file extension: {ext}")
    if source.is_dir():
        imgs = sorted([p for p in source.rglob("*") if p.suffix.lower() in _IMG_EXTS])
        vids = sorted([p for p in source.rglob("*") if p.suffix.lower() in _VID_EXTS])
        return imgs, vids
    raise FileNotFoundError(f"Source not found: {source}")


def infer_image(model, img_path: Path, out_dir: Path, conf: float,
                class_names: list[str] | None, dump_json: bool) -> dict:
    pil = Image.open(img_path).convert("RGB")
    detections = model.predict(pil, threshold=conf)

    img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    drawn = _draw(img_bgr, detections, class_names)

    out_img = out_dir / f"{img_path.stem}.png"
    cv2.imwrite(str(out_img), drawn)

    result = {"image": str(img_path), "output": str(out_img),
              "detections": _detections_to_dict(detections, class_names)}
    if dump_json:
        (out_dir / f"{img_path.stem}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
    return result


def infer_video(model, vid_path: Path, out_dir: Path, conf: float,
                class_names: list[str] | None) -> None:
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {vid_path}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_dir / f"{vid_path.stem}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pil  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            dets = model.predict(pil, threshold=conf)
            writer.write(_draw(frame, dets, class_names))
    finally:
        cap.release()
        writer.release()


def run(weights: str, source: str, out: str, arch: str = "RFDETRNano",
        task: str = "detect", conf: float = 0.5, resolution: int | None = None,
        class_names: list[str] | None = None, dump_json: bool = False) -> dict:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(weights, arch, task, resolution)

    if class_names is None:
        class_names = getattr(getattr(model, "model", None), "class_names", None)

    images, videos = _collect_inputs(Path(source))
    print(f"Found {len(images)} image(s), {len(videos)} video(s) in {source}")

    summary: list[dict] = []
    for p in images:
        summary.append(infer_image(model, p, out_dir, conf, class_names, dump_json))
    for p in videos:
        infer_video(model, p, out_dir, conf, class_names)

    if dump_json and summary:
        (out_dir / "predictions.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Done. Output → {out_dir.resolve()}")
    return {"images": len(images), "videos": len(videos), "out": str(out_dir.resolve())}


def main() -> None:
    p = argparse.ArgumentParser(description="RFDETR inference")
    p.add_argument("--weights",    required=True, help="Path to .pth checkpoint")
    p.add_argument("--source",     required=True, help="Image / folder / video path")
    p.add_argument("--out",        default="InferOutput")
    p.add_argument("--arch",       default="RFDETRNano",
                   choices=list(_DET_MAP.keys()))
    p.add_argument("--task",       default="detect", choices=["detect", "segment"])
    p.add_argument("--conf",       type=float, default=0.5)
    p.add_argument("--resolution", type=int, default=None,
                   help="Override input resolution (must match training)")
    p.add_argument("--classes",    nargs="*", default=None,
                   help="Optional class name list (overrides checkpoint names)")
    p.add_argument("--json",       action="store_true",
                   help="Dump per-image + summary JSON")
    args = p.parse_args()
    run(args.weights, args.source, args.out, args.arch, args.task,
        args.conf, args.resolution, args.classes, args.json)


if __name__ == "__main__":
    main()
