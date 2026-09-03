"""
DinoV3PMT inference script (semantic segmentation, mask-classification head).

Loads a trained DinoV3PMT checkpoint (.pth) via memolib and runs prediction on
an image, folder of images, or video. Saves annotated overlays and optionally
raw class-index mask PNGs + JSON summary.

PMT outputs (mask_logits, class_logits) from a mask-classification decoder;
we combine them into per-pixel logits via to_per_pixel_logits_semantic before
argmax.

Usage:
    python InferDinoV3PMT.py --weights TrainResult\Run_xxx\Weights\best.pth ^
                             --classes TrainResult\Run_xxx\classes.txt ^
                             --source  D:\\images ^
                             --out     D:\\images_out ^
                             --arch    VIT_BASE ^
                             --image-size 512

Architectures: VIT_SMALL | VIT_SMALL_PLUS | VIT_BASE | VIT_LARGE | VIT_HUGE_PLUS
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from MemoLib.Model.DinoV3PMT.DinoV3PMT import DinoV3PMT
from MemoLib.Model.BaseModel.eSegmentationModel import eDinoV3PMTModel


_ARCH_MAP = {
    "VIT_SMALL":      eDinoV3PMTModel.VIT_SMALL,
    "VIT_SMALL_PLUS": eDinoV3PMTModel.VIT_SMALL_PLUS,
    "VIT_BASE":       eDinoV3PMTModel.VIT_BASE,
    "VIT_LARGE":      eDinoV3PMTModel.VIT_LARGE,
    "VIT_HUGE_PLUS":  eDinoV3PMTModel.VIT_HUGE_PLUS,
}

_IMG_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
_VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

def load_model(weights: str, arch: str, num_classes: int, image_size: int,
               num_queries: int, num_decoder_blocks: int,
               decoder_hidden_dim: int | None,
               masked_attn_enabled: bool,
               lateral_projection: str, residual_projection: bool,
               device: str | None) -> DinoV3PMT:
    if arch not in _ARCH_MAP:
        raise ValueError(f"Unknown architecture '{arch}'. Expected one of {list(_ARCH_MAP)}")

    inst = DinoV3PMT()
    inst.cfg.Architecture        = _ARCH_MAP[arch]
    inst.cfg.ImageSize           = image_size
    inst.cfg.NumQueries          = num_queries
    inst.cfg.NumDecoderBlocks    = num_decoder_blocks
    inst.cfg.DecoderHiddenDim    = decoder_hidden_dim
    inst.cfg.MaskedAttnEnabled   = masked_attn_enabled
    inst.cfg.LateralProjection   = lateral_projection
    inst.cfg.ResidualProjection  = residual_projection
    inst.ClassesNumber           = num_classes
    if device:
        inst.Device = torch.device(device)

    # LoadWeight tự đọc sidecar .json (nếu có) và override toàn bộ ModelParams
    # để khớp checkpoint (image_size, num_classes, num_queries, ...).
    # LoadWeight cũng tự .to(Device), không cần gọi lại.
    inst.LoadWeight(weights)
    inst.model.eval()
    return inst


def _color(idx: int) -> tuple[int, int, int]:
    if idx == 0:
        return (0, 0, 0)
    rng = np.random.default_rng(idx * 9973 + 1)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


def _build_palette(num_classes: int) -> np.ndarray:
    return np.array([_color(i) for i in range(num_classes)], dtype=np.uint8)


@torch.no_grad()
def _predict_mask(model_inst: DinoV3PMT, img_rgb: np.ndarray) -> np.ndarray:
    # Dùng Predict() built-in để khớp 100% preprocessing training:
    # F.interpolate về ImageSize (cfg đã được sidecar khôi phục), ImageNet
    # normalize, forward, to_per_pixel_logits_semantic(last_layer), argmax,
    # rồi upsample về HxW gốc.
    return model_inst.Predict(img_rgb)


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


def infer_image(model_inst, img_path: Path, out_dir: Path,
                palette: np.ndarray, class_names: list[str] | None,
                alpha: float, save_mask: bool, dump_json: bool) -> dict:
    pil     = Image.open(img_path).convert("RGB")
    img_rgb = np.array(pil)                                    # RGB cho Predict()
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)         # BGR cho cv2 draw/write

    mask  = _predict_mask(model_inst, img_rgb)
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


def infer_video(model_inst, vid_path: Path, out_dir: Path,
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
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mask = _predict_mask(model_inst, frame_rgb)
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


def run(weights: str, source: str, out: str, arch: str = "VIT_BASE",
        image_size: int = 512,
        num_queries: int = 100,
        num_decoder_blocks: int = 6,
        decoder_hidden_dim: int | None = None,
        masked_attn_enabled: bool = True,
        lateral_projection: str = "mlp",
        residual_projection: bool = True,
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
                            num_queries, num_decoder_blocks, decoder_hidden_dim,
                            masked_attn_enabled, lateral_projection,
                            residual_projection, device)
    print(f"Loaded {arch} | ImageSize={image_size} | Classes={num_classes} "
          f"| Queries={num_queries} | DecoderBlocks={num_decoder_blocks} "
          f"| HiddenDim={decoder_hidden_dim} | MaskedAttn={masked_attn_enabled} "
          f"| Lateral={lateral_projection} | Residual={residual_projection} "
          f"| Device={model_inst.Device}")

    images, videos = _collect_inputs(Path(source))
    print(f"Found {len(images)} image(s), {len(videos)} video(s) in {source}")

    summary: list[dict] = []
    for p in images:
        summary.append(infer_image(model_inst, p, out_dir, palette,
                                   class_names, alpha, save_mask, dump_json))
    for p in videos:
        infer_video(model_inst, p, out_dir, palette, class_names, alpha)

    if dump_json and summary:
        (out_dir / "predictions.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Done. Output -> {out_dir.resolve()}")
    return {"images": len(images), "videos": len(videos), "out": str(out_dir.resolve())}


def main() -> None:
    p = argparse.ArgumentParser(description="DinoV3PMT inference")
    p.add_argument("--weights",     default=r"D:\Nghia\Python-Workspace\MemoTrainer\TrainResult\LG_FPCBV3_20260831_083243\Weights\last.pth",
                   help="Path to .pth checkpoint")
    p.add_argument("--source",      default=r"E:\TempData\LG_FPCB\temp",
                   help="Image / folder / video path")
    p.add_argument("--out",         default=r"E:\TempData\LG_FPCB\temp_pmt")
    p.add_argument("--arch",        default="VIT_SMALL",
                   choices=list(_ARCH_MAP.keys()))
    p.add_argument("--image-size",  type=int, default=640,
                   help="Input resolution (must match training; rounded up "
                        "to multiple of 16 for the ViT patch grid)")
    p.add_argument("--num-queries", type=int, default=100,
                   help="PMT decoder queries (must match training)")
    p.add_argument("--num-decoder-blocks", type=int, default=6,
                   help="Number of PMT decoder blocks (must match training)")
    p.add_argument("--decoder-hidden-dim", type=int, default=None,
                   help="Decoder MLP hidden dim; None → 4×embed_dim "
                        "(must match training)")
    p.add_argument("--no-masked-attn", action="store_true",
                   help="Disable PMT masked attention (must match training)")
    p.add_argument("--lateral-projection", default="mlp",
                   choices=["mlp", "linear", "none"],
                   help="Lateral projection on interaction features "
                        "(must match training)")
    p.add_argument("--no-residual-projection", action="store_true",
                   help="Disable residual projection on lateral features "
                        "(must match training)")
    p.add_argument("--classes",     default=None,
                   help="Path to classes.txt (one class per line)")
    p.add_argument("--class-names", nargs="*", default=["Background", "Copper", "Tin"],
                   help="Inline class name list (overrides --classes)")
    p.add_argument("--num-classes", type=int, default=3,
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
        args.num_queries, args.num_decoder_blocks, args.decoder_hidden_dim,
        not args.no_masked_attn, args.lateral_projection,
        not args.no_residual_projection,
        args.classes, args.class_names, args.num_classes,
        args.alpha, args.save_mask, args.json, args.device)


if __name__ == "__main__":
    main()
