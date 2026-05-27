# MemoTrainer

A PyQt6 desktop GUI for training computer vision models powered by [MemoLib](https://pypi.org/project/memolib/).

## Features

- **Multi-task support** — Classification, Detection, Segmentation
- **Multiple architectures** — YOLO, RF-DETR, PPLCNet, EfficientNet, DinoUperNet
- **Named config management** — Save, load, and delete named configs per model
- **Job queue** — Queue multiple training jobs and run them sequentially
- **Live log panel** — Real-time training output with auto-save to dated log files (`Logs/YYYY/MM/DD/`)
- **Training history** — Browse past training results in `TrainResult/`

## Supported Models

| Task | Models |
|------|--------|
| Classification | PPLCNet, YOLO-Cls, EfficientNet |
| Detection | YOLO-Det, RF-DETR Det |
| Segmentation | YOLO-Seg, RF-DETR Seg, DinoUperNet |

## Requirements

- Python 3.10+
- Windows (tested on Windows 10)
- NVIDIA GPU with CUDA 12.9+
- NVIDIA Driver ≥ 570.x

## Installation

```bash
git clone https://github.com/NghiaKTHP/MemoTrainer.git
cd MemoTrainer

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

1. Select a saved config from the dropdown (auto-loads model and parameters), or configure manually via the sidebar
2. Set your dataset path and training parameters
3. Click **Start Training** or **+ Queue** to add to the job queue
4. Monitor live output in the log panel

## Dataset Format

### Classification — YOLO / EfficientNet / PPLCNet

```
dataset/
├── train/
│   ├── cat/
│   └── dog/
└── val/
    ├── cat/
    └── dog/
```

### Detection — YOLO

```
dataset/
├── train/
│   ├── images/
│   └── labels/       # <class> <x_c> <y_c> <w> <h>  (normalized 0-1)
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

### Segmentation — YOLO

Same as detection, but label files use polygon points:
```
# <class> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

### Detection — RF-DETR

**COCO format** (auto-detected, recommended):
```
dataset/
├── train/
│   ├── img001.jpg
│   └── _annotations.coco.json
└── val/
    ├── img002.jpg
    └── _annotations.coco.json
```

**YOLO format** — same structure as YOLO detection above.

### Semantic Segmentation — DinoUperNet

```
dataset/
├── train/
│   ├── images/
│   └── masks/        # single-channel PNG, pixel value = class index
└── val/
    ├── images/
    └── masks/
```

## Saving Configs

Configs are stored in `Configs/<ModelName>/` as YAML files. Use the **Save** button in the toolbar to name and save the current configuration for later reuse.

## License

MIT
