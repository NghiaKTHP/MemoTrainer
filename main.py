import multiprocessing
import os
import sys
from pathlib import Path

# Prevent Ultralytics from auto-installing packages via subprocess (spawns visible cmd.exe on Windows)
os.environ["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"] = "1"

_WORKSPACE = Path(__file__).parent
os.chdir(_WORKSPACE)  # ensure TrainResult/ is relative to project root

from PyQt6.QtWidgets import QApplication
from app.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MemoLib Trainer")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
