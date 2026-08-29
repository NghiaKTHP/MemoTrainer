"""
ConvNeXt backbone feature visualization.

Load một checkpoint đã train của DinoV3ConvNextUperNet, chỉ chạy phần
ConvNeXt backbone để trích 4 feature map hierarchical (H/4, H/8, H/16,
H/32) rồi vẽ figure gồm ảnh gốc + heatmap trung bình kênh cho mỗi stage.

Usage (module):
    from Inference.ConvNextFeatureInference import ConvNextFeatureInference
    inf = ConvNextFeatureInference(weights="best.pth", arch="CONVNEXT_BASE",
                                   num_classes=3, image_size=640)
    inf.visualize("input.jpg", out_path="feat.png", show=True)

Usage (CLI):
    python ConvNextFeatureInference.py --weights best.pth --image input.jpg ^
                                       --arch CONVNEXT_BASE --image-size 640 ^
                                       --num-classes 3 --out feat.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
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


class ConvNextFeatureInference:
    STAGE_NAMES = ("stage1 (H/4)", "stage2 (H/8)",
                   "stage3 (H/16)", "stage4 (H/32)")

    def __init__(self,
                 weights: str,
                 arch: str = "CONVNEXT_BASE",
                 num_classes: int = 2,
                 image_size: int = 640,
                 decoder_channels: int = 512,
                 device: str | None = None):
        if arch not in _ARCH_MAP:
            raise ValueError(f"Unknown arch '{arch}'. "
                             f"Expected one of {list(_ARCH_MAP)}")

        self.image_size = image_size
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
    def extract_features(self, image) -> tuple[np.ndarray, list[torch.Tensor]]:
        """Trả về (img_bgr, [feat_stage1..4]) với feat NCHW trên self.device."""
        img_bgr = self._read_bgr(image)
        tensor  = _preprocess(img_bgr, self.image_size).to(self.device)

        feats: list[torch.Tensor] = []
        x = tensor
        for i in range(4):
            x = self.backbone.downsample_layers[i](x)
            x = self.backbone.stages[i](x)
            feats.append(x)
        return img_bgr, feats

    @staticmethod
    def _feature_heatmap(feat: torch.Tensor) -> np.ndarray:
        """(1,C,H,W) -> (H,W) heatmap [0..1] theo |activation| trung bình kênh."""
        hm = feat[0].abs().mean(dim=0).float().cpu().numpy()
        vmin, vmax = float(hm.min()), float(hm.max())
        if vmax - vmin > 1e-9:
            hm = (hm - vmin) / (vmax - vmin)
        else:
            hm = np.zeros_like(hm)
        return hm

    def visualize(self,
                  image,
                  out_path: str | Path | None = None,
                  cmap: str = "jet",
                  overlay: bool = False,
                  overlay_alpha: float = 0.5,
                  show: bool = False,
                  title: str | None = None) -> plt.Figure:
        """Tạo figure gồm ảnh gốc + 4 heatmap.

        overlay=True: heatmap được resize về kích thước ảnh gốc và blend
        với ảnh gốc (dễ so vị trí feature với ảnh).
        """
        img_bgr, feats = self.extract_features(image)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        n = 1 + len(feats)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.2))

        axes[0].imshow(rgb)
        axes[0].set_title(f"Input\n{W}x{H}")
        axes[0].axis("off")

        cmap_obj = plt.get_cmap(cmap)
        for ax, feat, name in zip(axes[1:], feats, self.STAGE_NAMES):
            hm = self._feature_heatmap(feat)
            _, C, fH, fW = feat.shape
            title_ax = f"{name}\nC={C}  {fW}x{fH}"

            if overlay:
                hm_up = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)
                heat_rgb = (cmap_obj(hm_up)[..., :3] * 255).astype(np.uint8)
                blended = cv2.addWeighted(rgb, 1.0 - overlay_alpha,
                                          heat_rgb, overlay_alpha, 0)
                ax.imshow(blended)
            else:
                ax.imshow(hm, cmap=cmap)

            ax.set_title(title_ax)
            ax.axis("off")

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


def main() -> None:
    ap = argparse.ArgumentParser(description="ConvNeXt backbone feature viz")
    ap.add_argument("--weights", default=r"Weights\MetaWeights\dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",
                    help="Path to .pth checkpoint")
    ap.add_argument("--image",   default=r"E:\TempData\LG_FPCB\Nghia\DataLabel\28_8",
                    help="Image file hoặc folder chứa ảnh")
    ap.add_argument("--out",     default=r"E:\TempData\LG_FPCB\temp",
                    help="Figure output file (.png) hoặc folder nếu --image là folder")
    ap.add_argument("--arch",    default="CONVNEXT_BASE",
                    choices=list(_ARCH_MAP.keys()))
    ap.add_argument("--image-size",        type=int, default=640)
    ap.add_argument("--decoder-channels",  type=int, default=512)
    ap.add_argument("--num-classes",       type=int, default=3)
    ap.add_argument("--cmap",    default="jet")
    ap.add_argument("--overlay", action="store_true",
                    help="Blend heatmap lên ảnh gốc (resize về H,W)")
    ap.add_argument("--alpha",   type=float, default=0.5,
                    help="Overlay opacity nếu --overlay")
    ap.add_argument("--show",    action="store_true")
    ap.add_argument("--device",  default=None)
    args = ap.parse_args()

    inf = ConvNextFeatureInference(
        weights=args.weights,
        arch=args.arch,
        num_classes=args.num_classes,
        image_size=args.image_size,
        decoder_channels=args.decoder_channels,
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
            overlay=args.overlay,
            overlay_alpha=args.alpha,
            show=args.show,
            title=img_path.name,
        )
        plt.close(fig)
        print(f"{img_path.name} -> {out_file}")


if __name__ == "__main__":
    main()
