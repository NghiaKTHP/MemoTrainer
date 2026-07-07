"""
EfficientNet inference script (image classification).

Loads a trained EfficientNet checkpoint (.pt state_dict) and runs prediction
on an image, a folder of images, or a video. Saves annotated outputs and
optionally dumps raw predictions to JSON.

Usage:
    python InferEfficientNet.py --weights TrainResult\Run_xxx\Weights\best.pt ^
                                --classes TrainResult\Run_xxx\classes.txt ^
                                --source D:\\images ^
                                --out    D:\\images_out ^
                                --arch   B0 ^
                                --image-size 224 ^
                                --topk   3

Architectures: B0 | B1 | B2 | B3 | B4 | B5 | B6 | B7 | V2S | V2M | V2L | V2XL
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms


_TV_ARCH_MAP = {
    "B0":  torchvision.models.efficientnet_b0,
    "B1":  torchvision.models.efficientnet_b1,
    "B2":  torchvision.models.efficientnet_b2,
    "B3":  torchvision.models.efficientnet_b3,
    "B4":  torchvision.models.efficientnet_b4,
    "B5":  torchvision.models.efficientnet_b5,
    "B6":  torchvision.models.efficientnet_b6,
    "B7":  torchvision.models.efficientnet_b7,
    "V2S": torchvision.models.efficientnet_v2_s,
    "V2M": torchvision.models.efficientnet_v2_m,
    "V2L": torchvision.models.efficientnet_v2_l,
}
_TIMM_ARCH_MAP = {"V2XL": "tf_efficientnetv2_xl"}

_IMG_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
_VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def _build_model(arch: str, num_classes: int, dropout: float) -> nn.Module:
    if arch in _TV_ARCH_MAP:
        model = _TV_ARCH_MAP[arch](weights=None)
        in_feat = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_feat, num_classes),
        )
        return model

    if arch in _TIMM_ARCH_MAP:
        try:
            import timm
        except ImportError as ex:
            raise RuntimeError("V2XL requires timm. Run: pip install timm") from ex
        backbone = timm.create_model(_TIMM_ARCH_MAP[arch], pretrained=False, num_classes=0)

        class _Wrapper(nn.Module):
            def __init__(self, bb, in_feat, n, p):
                super().__init__()
                self._backbone = bb
                self.features  = bb.blocks
                self.classifier = nn.Sequential(nn.Dropout(p=p), nn.Linear(in_feat, n))
            def forward(self, x):
                x = self._backbone.forward_features(x)
                x = self._backbone.global_pool(x)
                return self.classifier(x)

        return _Wrapper(backbone, backbone.num_features, num_classes, dropout)

    raise ValueError(f"Unknown architecture '{arch}'. Expected one of "
                     f"{list(_TV_ARCH_MAP) + list(_TIMM_ARCH_MAP)}")


def load_model(weights: str, arch: str, num_classes: int, dropout: float,
               device: torch.device) -> nn.Module:
    model = _build_model(arch, num_classes, dropout)
    state = torch.load(weights, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def _build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _color(idx: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(idx * 9973 + 1)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


def _predict_pil(model: nn.Module, pil: Image.Image, tfm, device: torch.device,
                 num_classes: int) -> np.ndarray:
    tensor = tfm(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        if num_classes == 1:
            probs = torch.sigmoid(logits).squeeze(0)
            probs = torch.cat([1.0 - probs, probs], dim=0)
        else:
            probs = torch.softmax(logits, dim=1).squeeze(0)
    return probs.detach().cpu().numpy()


def _draw(img_bgr: np.ndarray, probs: np.ndarray,
          class_names: list[str] | None, topk: int) -> np.ndarray:
    out = img_bgr.copy()
    k = min(topk, len(probs))
    order = np.argsort(-probs)[:k]

    top_cid = int(order[0])
    color   = _color(top_cid)
    name    = class_names[top_cid] if (class_names and top_cid < len(class_names)) else f"id{top_cid}"
    header  = f"{name} {probs[top_cid]:.2f}"

    (tw, th), bl = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(out, (0, 0), (tw + 10, th + bl + 8), color, -1)
    cv2.putText(out, header, (5, th + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    y = th + bl + 22
    for cid in order[1:]:
        cid = int(cid)
        nm  = class_names[cid] if (class_names and cid < len(class_names)) else f"id{cid}"
        line = f"{nm} {probs[cid]:.2f}"
        (lw, lh), lb = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (0, y - lh - lb), (lw + 8, y + 2), (30, 30, 30), -1)
        cv2.putText(out, line, (4, y - lb),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += lh + lb + 6
    return out


def _prediction_to_dict(probs: np.ndarray, class_names: list[str] | None,
                        topk: int) -> list[dict]:
    k = min(topk, len(probs))
    order = np.argsort(-probs)[:k]
    return [{
        "class_id":   int(c),
        "class_name": class_names[int(c)] if (class_names and int(c) < len(class_names)) else None,
        "score":      float(probs[int(c)]),
    } for c in order]


def _collect_inputs(source: Path) -> tuple[list[Path], list[Path]]:
    if source.is_file():
        ext = source.suffix.lower()
        if ext in _IMG_EXTS: return [source], []
        if ext in _VID_EXTS: return [], [source]
        raise ValueError(f"Unsupported file extension: {ext}")
    if source.is_dir():
        imgs = sorted([p for p in source.rglob("*") if p.suffix.lower() in _IMG_EXTS])
        vids = sorted([p for p in source.rglob("*") if p.suffix.lower() in _VID_EXTS])
        return imgs, vids
    raise FileNotFoundError(f"Source not found: {source}")


def infer_image(model, img_path: Path, out_dir: Path, tfm, device, num_classes: int,
                class_names: list[str] | None, topk: int, dump_json: bool) -> dict:
    pil     = Image.open(img_path).convert("RGB")
    probs   = _predict_pil(model, pil, tfm, device, num_classes)
    img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    drawn   = _draw(img_bgr, probs, class_names, topk)

    out_img = out_dir / f"{img_path.stem}.png"
    cv2.imwrite(str(out_img), drawn)

    result = {"image": str(img_path), "output": str(out_img),
              "predictions": _prediction_to_dict(probs, class_names, topk)}
    if dump_json:
        (out_dir / f"{img_path.stem}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
    return result


def infer_video(model, vid_path: Path, out_dir: Path, tfm, device, num_classes: int,
                class_names: list[str] | None, topk: int) -> None:
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {vid_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_dir / f"{vid_path.stem}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pil   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            probs = _predict_pil(model, pil, tfm, device, num_classes)
            writer.write(_draw(frame, probs, class_names, topk))
    finally:
        cap.release()
        writer.release()


def _load_class_names(classes_file: str | None) -> list[str] | None:
    if not classes_file:
        return None
    p = Path(classes_file)
    if not p.exists():
        raise FileNotFoundError(f"classes file not found: {p}")
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def run(weights: str, source: str, out: str, arch: str = "B0",
        image_size: int = 224, dropout: float = 0.0,
        classes_file: str | None = None,
        class_names: list[str] | None = None,
        topk: int = 3, dump_json: bool = False,
        device: str | None = None) -> dict:
    out_dir = Path(out); out_dir.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = _load_class_names(classes_file)
    if not class_names:
        raise ValueError("Cần cung cấp class names qua --classes hoặc --class-names "
                         "để biết num_classes.")

    num_classes = 1 if len(class_names) == 2 and False else len(class_names)
    # Binary EfficientNet trong memolib dùng 1 logit khi ClassesNumber==1, nhưng
    # classes.txt vẫn liệt kê 2 class → giữ num_classes = len(class_names).
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(weights, arch, num_classes, dropout, dev)
    tfm   = _build_transform(image_size)

    images, videos = _collect_inputs(Path(source))
    print(f"Found {len(images)} image(s), {len(videos)} video(s) in {source} | device={dev}")

    summary: list[dict] = []
    for p in images:
        summary.append(infer_image(model, p, out_dir, tfm, dev, num_classes,
                                   class_names, topk, dump_json))
    for p in videos:
        infer_video(model, p, out_dir, tfm, dev, num_classes, class_names, topk)

    if dump_json and summary:
        (out_dir / "predictions.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Done. Output → {out_dir.resolve()}")
    return {"images": len(images), "videos": len(videos), "out": str(out_dir.resolve())}


def main() -> None:
    p = argparse.ArgumentParser(description="EfficientNet inference")
    p.add_argument("--weights",     required=True, help="Path to .pt state_dict")
    p.add_argument("--source",      required=True, help="Image / folder / video path")
    p.add_argument("--out",         default="InferOutput")
    p.add_argument("--arch",        default="B0",
                   choices=list(_TV_ARCH_MAP.keys()) + list(_TIMM_ARCH_MAP.keys()))
    p.add_argument("--image-size",  type=int, default=224,
                   help="Input resolution (must match training)")
    p.add_argument("--dropout",     type=float, default=0.0,
                   help="Classifier dropout (must match training)")
    p.add_argument("--classes",     default=None,
                   help="Path to classes.txt (one class per line)")
    p.add_argument("--class-names", nargs="*", default=None,
                   help="Inline class name list (overrides --classes)")
    p.add_argument("--topk",        type=int, default=3)
    p.add_argument("--json",        action="store_true",
                   help="Dump per-image + summary JSON")
    p.add_argument("--device",      default=None, help="cuda | cpu (auto if omitted)")
    args = p.parse_args()

    run(args.weights, args.source, args.out, args.arch,
        args.image_size, args.dropout, args.classes, args.class_names,
        args.topk, args.json, args.device)


if __name__ == "__main__":
    main()
