# MemoTrainer v1.0.0 Release Notes

## Overview
MemoTrainer is a desktop-based computer vision training and inference tool with a user-friendly PyQt6 interface. The first version supports multiple vision tasks and a comprehensive set of pre-trained models, making it suitable for researchers, developers, and ML engineers working with image classification, object detection, and semantic segmentation.

## ✨ Key Features

### Supported Computer Vision Tasks
- **Image Classification** - Classify images into predefined categories
- **Object Detection** - Detect and localize objects within images
- **Semantic Segmentation** - Pixel-level image segmentation

### Supported Models
- **YOLO** - Fast and accurate object detection
- **RF-DETR** - Vision Transformer-based detection model
- **EfficientNet** - Lightweight classification backbone
- **PPLCNet** - High-efficiency classification model
- **DinoUpernet** - Advanced segmentation with transformer architecture

### Core Capabilities
- Easy model selection and configuration
- Real-time inference on images/video
- Batch processing support
- Model training and fine-tuning
- Results visualization and export
- Cross-platform compatibility

## 🛠️ System Requirements

### Minimum
- **OS**: Windows 10 or later
- **RAM**: 8GB (16GB+ recommended for training)
- **Storage**: 5GB available space
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

### Recommended
- **OS**: Windows 10/11 (21H2+)
- **RAM**: 16GB or more
- **GPU**: NVIDIA RTX 2080 Ti or better
- **Storage**: SSD with 20GB+ free space

## 📦 Installation

### Option 1: Download Executable (Recommended)
1. Download all `.part` files from the latest release
2. Place them in the same folder
3. Run `merge.bat` to reconstruct the full zip file
4. Extract `MemoTrainer-windows-x64.zip`
5. Run `MemoTrainer.exe`

### Option 2: From Source
```bash
git clone https://github.com/yourusername/MemoTrainer.git
cd MemoTrainer
pip install -r requirements.txt
python main.py
```

## 🚀 Quick Start

1. **Launch Application**
   - Double-click `MemoTrainer.exe`
   - Or run: `python main.py`

2. **Select a Model**
   - Choose from available models in the UI
   - Configure model parameters

3. **Load Data**
   - Select image(s) or video file
   - Configure input settings

4. **Run Inference**
   - Click "Run" to process
   - View results in real-time

5. **Export Results**
   - Save predictions in desired format
   - Export visualizations

## 📋 Model Details

### Classification Models
- **EfficientNet**: Lightweight, efficient classification
- **PPLCNet**: Mobile-friendly high-performance classifier

### Detection Models
- **YOLO**: Real-time object detection, multiple class support
- **RF-DETR**: High-accuracy detection with transformer architecture

### Segmentation Models
- **DinoUpernet**: Fine-grained segmentation with vision foundation model

## ⚙️ Configuration

Models can be configured through:
- GUI configuration panels
- Configuration files (YAML format)
- Command-line arguments (for Python users)

## 🐛 Known Limitations

- GPU acceleration requires NVIDIA drivers (CUDA 11.8+)
- Large image resolution (>4K) may require additional RAM
- Some models require minimum input image size (e.g., 224x224)
- Currently supports batch processing up to model memory limits

## 📝 Usage Examples

### Classification
```
1. Select "Classification" task
2. Choose EfficientNet or PPLCNet model
3. Load image(s)
4. Run inference
5. View confidence scores
```

### Detection
```
1. Select "Detection" task
2. Choose YOLO or RF-DETR model
3. Load image(s)
4. Configure confidence threshold
5. View detected bounding boxes
```

### Segmentation
```
1. Select "Segmentation" task
2. Choose DinoUpernet model
3. Load image(s)
4. Run inference
5. Export segmentation masks
```

## 📂 Project Structure

```
MemoTrainer/
├── main.py                 # Application entry point
├── build.py               # Build script for exe
├── merge.bat              # Script to merge split files
├── requirements.txt       # Python dependencies
├── MemoLib/               # Computer vision models library
│   ├── Model/
│   │   ├── YOLO/
│   │   ├── RFDETR/
│   │   ├── EfficientNet/
│   │   ├── PPLCNet/
│   │   └── DinoUperNet/
│   └── ...
└── README.md
```

## 🔄 Update Notes

This is the initial release (v1.0.0). Future updates will include:
- Additional model support
- Enhanced UI/UX
- Performance optimizations
- Multi-GPU support
- Video processing improvements

## 🤝 Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Contact: vominhkietx201@gmail.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

MIT License allows you to:
- ✅ Use commercially
- ✅ Modify the code
- ✅ Distribute copies
- ✅ Use privately

With the condition that you include a copy of the license and copyright notice.

## 🙏 Acknowledgments

- Computer vision models from leading research communities
- OpenVINO toolkit for model optimization
- PyQt6 for the GUI framework

---

**Release Date**: May 2026  
**Version**: 1.0.0  
**Status**: Stable
