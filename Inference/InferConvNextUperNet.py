"""
DinoV3ConvNextUperNet inference script (semantic segmentation).

Loads a trained DinoV3ConvNextUperNet checkpoint (.pth) via memolib and runs
prediction on an image, folder of images, or video. Saves annotated overlays
and optionally raw class-index mask PNGs + JSON summary.

Usage:
    python InferConvNextUperNet.py --weights TrainResult\Run_xxx\Weights\best.pth ^
                                   --classes TrainResult\Run_xxx\classes.txt ^
                                   --source  D:\\images ^
                                   --out     D:\\images_out ^
                                   --arch    CONVNEXT_SMALL ^
                                   --image-size 512

Architectures: CONVNEXT_TINY | CONVNEXT_SMALL | CONVNEXT_BASE | CONVNEXT_LARGE
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from MemoLib.Model.DinoV3ConvNextUperNet.DinoV3ConvNextUperNet import DinoV3ConvNextUperNet
from MemoLib.Model.BaseModel.eSegmentationModel import eDinoV3ConvNextUperNetModel


_ARCH_MAP = {
    "CONVNEXT_TINY":  eDinoV3ConvNextUperNetModel.CONVNEXT_TINY,
    "CONVNEXT_SMALL": eDinoV3ConvNextUperNetModel.CONVNEXT_SMALL,
    "CONVNEXT_BASE":  eDinoV3ConvNextUperNetModel.CONVNEXT_BASE,
    "CONVNEXT_LARGE": eDinoV3ConvNextUperNetModel.CONVNEXT_LARGE,
}

_IMG_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
_VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_PATCH_STRIDE  = 32


def load_model(weights: str, arch: str, num_classes: int, image_size: int,
               decoder_channels: int, device: str | None) -> DinoV3ConvNextUperNet:
    if arch not in _ARCH_MAP:
        raise ValueError(f"Unknown architecture '{arch}'. Expected one of {list(_ARCH_MAP)}")

    inst = DinoV3ConvNextUperNet()
    inst.cfg.Architecture     = _ARCH_MAP[arch]
    inst.cfg.ImageSize        = image_size
    inst.cfg.DecoderChannels  = decoder_channels
    inst.ClassesNumber        = num_classes
    if device:
        inst.Device = torch.device(device)

    inst.model = inst.LoadWeight(weights)
    inst.model.eval()
    return inst


def _color(idx: int) -> tuple[int, int, int]:
    if idx == 0:
        return (0, 0, 0)
    rng = np.random.default_rng(idx * 9973 + 1)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


def _build_palette(num_classes: int) -> np.ndarray:
    return np.array([_color(i) for i in range(num_classes)], dtype=np.uint8)


def _round_up(x: int, stride: int) -> int:
    return ((x + stride - 1) // stride) * stride


def _preprocess(img_bgr: np.ndarray, image_size: int) -> torch.Tensor:
    size = _round_up(image_size, _PATCH_STRIDE)
    rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb  = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    arr  = rgb.astype(np.float32) / 255.0
    arr  = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor


@torch.no_grad()
def _predict_mask(model_inst: DinoV3ConvNextUperNet, img_bgr: np.ndarray,
                  image_size: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    tensor = _preprocess(img_bgr, image_size).to(model_inst.Device)
    logits = model_inst.model(tensor)
    logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    return logits.argmax(dim=1).squeeze(0).to(torch.uint8).cpu().numpy()


def _draw(img_bgr: np.ndarray, mask: np.ndarray, palette: np.ndarray,
          class_names: list[str] | None, alpha: float) -> np.ndarray:
    color_mask = palette[mask]                         # HxWx3 RGB
    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
    fg = mask > 0
    out = img_bgr.copy()
    if fg.any():
        blended = cv2.addWeighted(img_bgr, 1.0 - alpha, color_mask, alpha, 0)
        out[fg] = blended[fg]

    classes_present = sorted(int(c) for c in np.unique(mask) if c > 0)
    if not classes_present:
        return out

    y = 8
    for cid in classes_present:
        name = class_names[cid] if (class_names and cid < len(class_names)) else f"id{cid}"
        line = f"{name} ({cid})"
        (tw, th), bl = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        color_bgr = tuple(int(c) for c in cv2.cvtColor(
            palette[cid].reshape(1, 1, 3), cv2.COLOR_RGB2BGR).ravel())
        cv2.rectangle(out, (0, y), (tw + 24, y + th + bl + 4), color_bgr, -1)
        cv2.rectangle(out, (4, y + 3), (16, y + th + bl - 1), color_bgr, -1)
        cv2.putText(out, line, (20, y + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        y += th + bl + 8
    return out


def _mask_stats(mask: np.ndarray, class_names: list[str] | None) -> list[dict]:
    ids, counts = np.unique(mask, return_counts=True)
    total = int(mask.size)
    stats = []
    for cid, cnt in zip(ids.tolist(), counts.tolist()):
        cid = int(cid)
        stats.append({
            "class_id":   cid,
            "class_name": class_names[cid] if (class_names and cid < len(class_names)) else None,
            "pixels":     int(cnt),
            "ratio":      float(cnt) / total,
        })
    stats.sort(key=lambda x: -x["pixels"])
    return stats


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


def infer_image(model_inst, img_path: Path, out_dir: Path, image_size: int,
                palette: np.ndarray, class_names: list[str] | None,
                alpha: float, save_mask: bool, dump_json: bool) -> dict:
    pil     = Image.open(img_path).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    mask  = _predict_mask(model_inst, img_bgr, image_size)
    drawn = _draw(img_bgr, mask, palette, class_names, alpha)

    out_img = out_dir / f"{img_path.stem}.png"
    cv2.imwrite(str(out_img), drawn)

    result = {"image": str(img_path), "output": str(out_img),
              "classes": _mask_stats(mask, class_names)}

    if save_mask:
        mask_path = out_dir / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(mask_path), mask)
        result["mask"] = str(mask_path)

    if dump_json:
        (out_dir / f"{img_path.stem}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
    return result


def infer_video(model_inst, vid_path: Path, out_dir: Path, image_size: int,
                palette: np.ndarray, class_names: list[str] | None,
                alpha: float) -> None:
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
            mask = _predict_mask(model_inst, frame, image_size)
            writer.write(_draw(frame, mask, palette, class_names, alpha))
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


def run(weights: str, source: str, out: str, arch: str = "CONVNEXT_SMALL",
        image_size: int = 512,
        decoder_channels: int = 512,
        classes_file: str | None = None,
        class_names: list[str] | None = None,
        num_classes: int | None = None,
        alpha: float = 0.5,
        save_mask: bool = False,
        dump_json: bool = False,
        device: str | None = None) -> dict:
    out_dir = Path(out); out_dir.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = _load_class_names(classes_file)

    if num_classes is None:
        if not class_names:
            raise ValueError("Cần --classes / --class-names hoặc --num-classes "
                             "để biết số class của model.")
        num_classes = len(class_names)

    palette = _build_palette(num_classes)

    model_inst = load_model(weights, arch, num_classes, image_size,
                            decoder_channels, device)
    print(f"Loaded {arch} | ImageSize={image_size} | Classes={num_classes} "
          f"| DecoderChannels={decoder_channels} | Device={model_inst.Device}")

    images, videos = _collect_inputs(Path(source))
    print(f"Found {len(images)} image(s), {len(videos)} video(s) in {source}")

    summary: list[dict] = []
    for p in images:
        summary.append(infer_image(model_inst, p, out_dir, image_size, palette,
                                   class_names, alpha, save_mask, dump_json))
    for p in videos:
        infer_video(model_inst, p, out_dir, image_size, palette, class_names, alpha)

    if dump_json and summary:
        (out_dir / "predictions.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Done. Output -> {out_dir.resolve()}")
    return {"images": len(images), "videos": len(videos), "out": str(out_dir.resolve())}


def main() -> None:
    p = argparse.ArgumentParser(description="DinoV3ConvNextUperNet inference")
    p.add_argument("--weights",     default=r"D:\Nghia\Python-Workspace\MemoTrainer\TrainResult\EleDinoHoleSegment_20260819_131502\Weights\best.pth", help="Path to .pth checkpoint")
    p.add_argument("--source",      default=r"E:\TempData\PhoneCaseDino\TestAKiet\defect", help="Image / folder / video path")
    p.add_argument("--out",         default=r"E:\TempData\PhoneCaseDino\TestAKiet\Temp")
    p.add_argument("--arch",        default="CONVNEXT_TINY",
                   choices=list(_ARCH_MAP.keys()))
    p.add_argument("--image-size",  type=int, default=192,
                   help="Input resolution (must match training; will be rounded "
                        "up to multiple of 32 by the model)")
    p.add_argument("--decoder-channels", type=int, default=128,
                   help="UperNet decoder channels (must match training)")
    p.add_argument("--classes",     default=None,
                   help="Path to classes.txt (one class per line)")
    p.add_argument("--class-names", nargs="*", default=["Background", "Cir"],
                   help="Inline class name list (overrides --classes)")
    p.add_argument("--num-classes", type=int, default=2,
                   help="Override number of classes (fallback if no class names given)")
    p.add_argument("--alpha",       type=float, default=0.5,
                   help="Mask overlay opacity [0..1]")
    p.add_argument("--save-mask",   action="store_true",
                   help="Also save raw class-index PNG mask")
    p.add_argument("--json",        action="store_true",
                   help="Dump per-image + summary JSON")
    p.add_argument("--device",      default=None, help="cuda | cpu (auto if omitted)")
    args = p.parse_args()

    run(args.weights, args.source, args.out, args.arch, args.image_size,
        args.decoder_channels, args.classes, args.class_names, args.num_classes,
        args.alpha, args.save_mask, args.json, args.device)


if __name__ == "__main__":
    main()
