import multiprocessing
import os
import sys
from pathlib import Path

# PyInstaller windowed mode has sys.stdout/stderr = None → breaks logging.StreamHandler
# and any library that calls .write() on stderr. Redirect to devnull early, before any
# other import (torch, logging, etc.) caches a reference to the None stream.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8", buffering=1)
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8", buffering=1)

# SSL cert bundle for frozen exe: PyInstaller khong luon bundle OS cert store dung cach,
# torch.hub / urllib download weight qua HTTPS se fail tren may khac.
# Tat cert verification — chap nhan duoc vi chi tai public model weights.
import ssl as _ssl
_ssl._create_default_https_context = _ssl._create_unverified_context

# Prevent Ultralytics from auto-installing packages via subprocess (spawns visible cmd.exe on Windows)
os.environ["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"] = "1"

_WORKSPACE = Path(__file__).parent
os.chdir(_WORKSPACE)  # ensure TrainResult/ is relative to project root

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication
from app.main_window import MainWindow


def _force_light_palette(app: QApplication):
    """Ep light theme tren toan app — tranh truong hop he dieu hanh dat dark mode
    khien text trang tren nen trang (khong doc duoc) o cac field khong override
    color trong stylesheet."""
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window,          QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.WindowText,      QColor("#111111"))
    p.setColor(QPalette.ColorRole.Base,            QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor("#f5f5f5"))
    p.setColor(QPalette.ColorRole.Text,            QColor("#111111"))
    p.setColor(QPalette.ColorRole.Button,          QColor("#fafafa"))
    p.setColor(QPalette.ColorRole.ButtonText,      QColor("#111111"))
    p.setColor(QPalette.ColorRole.ToolTipBase,     QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.ToolTipText,     QColor("#111111"))
    p.setColor(QPalette.ColorRole.PlaceholderText, QColor("#aaaaaa"))
    p.setColor(QPalette.ColorRole.Highlight,       QColor("#1a6fb5"))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.BrightText,      QColor("#111111"))
    app.setPalette(p)

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MemoLib Trainer")
    app.setStyle("Fusion")
    _force_light_palette(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()