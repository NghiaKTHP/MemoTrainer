"""
ConvNeXt fine-tuned backbone + segment head feature visualization.

Tương tự [ConvNextFeatureInference.py](ConvNextFeatureInference.py) nhưng
script này load checkpoint đã train xong (backbone + UperNet head), rồi:
  - Trích 4 feature map hierarchical của backbone (H/4, H/8, H/16, H/32)
  - Chạy nốt UperNet head để lấy segmentation mask dự đoán

Mục đích: so sánh feature map của backbone trước/sau khi fine-tune
(unfreeze backbone) để kiểm chứng thay đổi.

Usage (module):
    from Inference.ConvNextFinetunedFeatureInference import (
        ConvNextFinetunedFeatureInference,
    )
    inf = ConvNextFinetunedFeatureInference(
        weights="best.pth", arch="CONVNEXT_BASE",
        num_classes=3, image_size=640,
        class_names=["Background", "Tin", "Copper"],
    )
    inf.visualize("input.jpg", out_path="feat_finetuned.png", show=True)

Usage (CLI):
    python ConvNextFinetunedFeatureInference.py --weights best.pth ^
        --image input.jpg --arch CONVNEXT_BASE --image-size 640 ^
        --num-classes 3 --out feat_finetuned.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
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

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_PATCH_STRIDE  = 32


def _round_up(x: int, stride: int) -> int:
    return ((x + stride - 1) // stride) * stride


def _preprocess(img_bgr: np.ndarray, image_size: int) -> torch.Tensor:
    size = _round_up(image_size, _PATCH_STRIDE)
    rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb  = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    arr  = rgb.astype(np.float32) / 255.0
    arr  = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def _build_model(weights: str, arch: str, num_classes: int, image_size: int,
                 decoder_channels: int, device: str | None) -> DinoV3ConvNextUperNet:
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


class ConvNextFinetunedFeatureInference:
    STAGE_NAMES = ("stage1 (H/4)", "stage2 (H/8)",
                   "stage3 (H/16)", "stage4 (H/32)")

    def __init__(self,
                 weights: str,
                 arch: str = "CONVNEXT_BASE",
                 num_classes: int = 2,
                 image_size: int = 640,
                 decoder_channels: int = 512,
                 class_names: list[str] | None = None,
                 device: str | None = None):
        if arch not in _ARCH_MAP:
            raise ValueError(f"Unknown arch '{arch}'. "
                             f"Expected one of {list(_ARCH_MAP)}")

        self.image_size  = image_size
        self.num_classes = num_classes
        self.class_names = class_names
        self.palette     = _build_palette(num_classes)

        self.model_inst = _build_model(
            weights=weights,
            arch=arch,
            num_classes=num_classes,
            image_size=image_size,
            decoder_channels=decoder_channels,
            device=device,
        )
        self.model    = self.model_inst.model
        self.backbone = self.model.backbone
        self.device   = self.model_inst.Device
        self.model.eval()

    def _read_bgr(self, image) -> np.ndarray:
        if isinstance(image, (str, Path)):
            pil = Image.open(image).convert("RGB")
            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return image
        raise TypeError(f"Unsupported image type: {type(image)}")

    @torch.no_grad()
    def extract(self, image) -> tuple[np.ndarray, list[torch.Tensor], np.ndarray]:
        """Return (img_bgr, [feat_stage1..4], mask_HxW_uint8).

        Chạy backbone thủ công 4 stage, rồi đưa qua UperNet head để lấy
        segmentation mask ở kích thước ảnh gốc.
        """
        img_bgr = self._read_bgr(image)
        H, W    = img_bgr.shape[:2]
        tensor  = _preprocess(img_bgr, self.image_size).to(self.device)

        feats: list[torch.Tensor] = []
        x = tensor
        for i in range(4):
            x = self.backbone.downsample_layers[i](x)
            x = self.backbone.stages[i](x)
            feats.append(x)

        coarse = self.model.decode_head(tuple(feats))
        logits = F.interpolate(coarse, size=(H, W),
                               mode="bilinear", align_corners=False)

        if getattr(self.model, "use_pointrend", False):
            idx, coords  = self.model.pointrend.get_uncertain_points_infer(logits)
            refined      = self.model.pointrend.refine(feats[0], coarse, coords)
            B, C, Hf, Wf = logits.shape
            flat         = logits.reshape(B, C, Hf * Wf)
            idx_exp      = idx.unsqueeze(1).expand(-1, C, -1)
            flat         = flat.scatter(2, idx_exp, refined)
            logits       = flat.reshape(B, C, Hf, Wf)

        mask = logits.argmax(dim=1).squeeze(0).to(torch.uint8).cpu().numpy()
        return img_bgr, feats, mask

    @staticmethod
    def _feature_heatmap(feat: torch.Tensor) -> np.ndarray:
        hm = feat[0].abs().mean(dim=0).float().cpu().numpy()
        vmin, vmax = float(hm.min()), float(hm.max())
        if vmax - vmin > 1e-9:
            hm = (hm - vmin) / (vmax - vmin)
        else:
            hm = np.zeros_like(hm)
        return hm

    def _overlay_mask(self, rgb: np.ndarray, mask: np.ndarray,
                      alpha: float) -> np.ndarray:
        color_mask = self.palette[mask]           # HxWx3 RGB
        fg = mask > 0
        out = rgb.copy()
        if fg.any():
            blended = cv2.addWeighted(rgb, 1.0 - alpha, color_mask, alpha, 0)
            out[fg] = blended[fg]
        return out

    def visualize(self,
                  image,
                  out_path: str | Path | None = None,
                  cmap: str = "jet",
                  feat_overlay: bool = False,
                  feat_alpha: float = 0.5,
                  mask_alpha: float = 0.5,
                  show: bool = False,
                  title: str | None = None) -> plt.Figure:
        """Vẽ figure 1 hàng: Input | stage1..4 | Seg overlay."""
        img_bgr, feats, mask = self.extract(image)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        n = 1 + len(feats) + 1
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.2))

        axes[0].imshow(rgb)
        axes[0].set_title(f"Input\n{W}x{H}")
        axes[0].axis("off")

        cmap_obj = plt.get_cmap(cmap)
        for ax, feat, name in zip(axes[1:1 + len(feats)], feats, self.STAGE_NAMES):
            hm = self._feature_heatmap(feat)
            _, C, fH, fW = feat.shape
            title_ax = f"{name}\nC={C}  {fW}x{fH}"

            if feat_overlay:
                hm_up = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)
                heat_rgb = (cmap_obj(hm_up)[..., :3] * 255).astype(np.uint8)
                blended = cv2.addWeighted(rgb, 1.0 - feat_alpha,
                                          heat_rgb, feat_alpha, 0)
                ax.imshow(blended)
            else:
                ax.imshow(hm, cmap=cmap)

            ax.set_title(title_ax)
            ax.axis("off")

        seg_ax = axes[-1]
        seg_ax.imshow(self._overlay_mask(rgb, mask, mask_alpha))

        present = sorted(int(c) for c in np.unique(mask))
        if self.class_names:
            legend = ", ".join(
                self.class_names[c] if c < len(self.class_names) else f"id{c}"
                for c in present
            )
        else:
            legend = ", ".join(f"id{c}" for c in present)
        seg_ax.set_title(f"Seg pred\n[{legend}]")
        seg_ax.axis("off")

        if title:
            fig.suptitle(title)
        fig.tight_layout()

        if out_path:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        return fig


def _iter_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in _IMG_EXTS else []
    if source.is_dir():
        return sorted(p for p in source.rglob("*")
                      if p.suffix.lower() in _IMG_EXTS)
    raise FileNotFoundError(f"Source not found: {source}")


def _load_class_names(path_str: str | None) -> list[str] | None:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"classes file not found: {p}")
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fine-tuned ConvNeXt backbone + seg head feature viz"
    )
    ap.add_argument("--weights",
                    default=r"D:\Nghia\Python-Workspace\MemoTrainer\TrainResult\LGFPCB_V2_20260828_141700\Weights\best.pth",
                    help="Path to fine-tuned .pth checkpoint")
    ap.add_argument("--image",
                    default=r"E:\TempData\LG_FPCB\Nghia\DataLabel\28_8",
                    help="Image file hoặc folder chứa ảnh")
    ap.add_argument("--out",
                    default=r"E:\TempData\LG_FPCB\temp_finetuned",
                    help="Figure output file (.png) hoặc folder nếu --image là folder")
    ap.add_argument("--arch", default="CONVNEXT_SMALL",
                    choices=list(_ARCH_MAP.keys()))
    ap.add_argument("--image-size",        type=int, default=640)
    ap.add_argument("--decoder-channels",  type=int, default=512)
    ap.add_argument("--num-classes",       type=int, default=3)
    ap.add_argument("--classes",           default=None,
                    help="Path tới classes.txt (một class/dòng)")
    ap.add_argument("--class-names",       nargs="*",
                    default=["Background", "Tin", "Copper"],
                    help="Danh sách tên class inline (override --classes)")
    ap.add_argument("--cmap",       default="jet")
    ap.add_argument("--feat-overlay", action="store_true",
                    help="Blend feature heatmap lên ảnh gốc")
    ap.add_argument("--feat-alpha", type=float, default=0.5)
    ap.add_argument("--mask-alpha", type=float, default=0.5,
                    help="Opacity seg overlay")
    ap.add_argument("--show",    action="store_true")
    ap.add_argument("--device",  default=None)
    args = ap.parse_args()
    
    class_names = args.class_names or _load_class_names(args.classes)
    num_classes = args.num_classes or (len(class_names) if class_names else None)
    if not num_classes:
        raise ValueError("Cần --num-classes hoặc --class-names / --classes")

    inf = ConvNextFinetunedFeatureInference(
        weights=args.weights,
        arch=args.arch,
        num_classes=num_classes,
        image_size=args.image_size,
        decoder_channels=args.decoder_channels,
        class_names=class_names,
        device=args.device,
    )

    src = Path(args.image)
    images = _iter_images(src)
    if not images:
        raise FileNotFoundError(f"No image found at: {src}")

    out = Path(args.out)
    multi = len(images) > 1 or src.is_dir()
    if multi:
        out.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        out_file = (out / f"{img_path.stem}_feat.png") if multi else out
        fig = inf.visualize(
            img_path,
            out_path=out_file,
            cmap=args.cmap,
            feat_overlay=args.feat_overlay,
            feat_alpha=args.feat_alpha,
            mask_alpha=args.mask_alpha,
            show=args.show,
            title=img_path.name,
        )
        plt.close(fig)
        print(f"{img_path.name} -> {out_file}")


if __name__ == "__main__":
    main()
