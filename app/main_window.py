import importlib

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                              QPushButton, QTabWidget, QVBoxLayout, QWidget)

from .constants import MODEL_INFO
from .convert_tab import ConvertTab
from .history_tab import HistoryTab
from .queue_bar import QueueBar
from .train_tab import TrainTab
from .training_manager import TrainingManager


class _Preloader(QThread):
    """Import all heavy model/config modules in background so first-click is instant."""
    done = pyqtSignal()

    def run(self):
        for info in MODEL_INFO.values():
            for key in ("config_module", "arch_enum_module"):
                try:
                    importlib.import_module(info[key])
                except Exception:
                    pass
        self.done.emit()

_WIN_STYLE = """
QMainWindow, QWidget#root { background: #fff; }
QTabWidget::pane { border: none; }
QTabWidget::tab-bar { alignment: left; }
QTabBar::tab {
    font-size: 12px; padding: 8px 18px;
    border: none; border-bottom: 2px solid transparent;
    color: #888; background: #fafafa;
}
QTabBar::tab:hover { color: #333; }
QTabBar::tab:selected {
    color: #111; border-bottom: 2px solid #111;
    font-weight: 600; background: #fafafa;
}
QWidget#titlebar {
    background: #fafafa;
    border-bottom: 1px solid #e0e0e0;
}
QLabel#appTitle { font-size: 13px; font-weight: 500; }
QLabel#appSub { font-size: 11px; color: #aaa; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MemoLib Trainer")
        self.resize(1200, 750)
        self.setMinimumSize(900, 600)
        self.setStyleSheet(_WIN_STYLE)

        self._manager = TrainingManager(self)

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Title bar
        titlebar = QWidget()
        titlebar.setObjectName("titlebar")
        titlebar.setFixedHeight(38)
        tb = QHBoxLayout(titlebar)
        tb.setContentsMargins(14, 0, 14, 0)
        tb.setSpacing(10)

        tb.addWidget(QLabel("🧠"))

        app_title = QLabel("MemoLib Trainer")
        app_title.setObjectName("appTitle")
        tb.addWidget(app_title)

        sub = QLabel("venv  •  D:\\Nghia\\Python-Workspace\\MemoLibV2")
        sub.setObjectName("appSub")
        tb.addWidget(sub)
        tb.addStretch()

        self._status_lbl = QLabel("Loading models…")
        self._status_lbl.setStyleSheet("font-size: 11px; color: #aaa;")
        tb.addWidget(self._status_lbl)

        layout.addWidget(titlebar)

        # Tabs
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        layout.addWidget(self._tabs, 1)

        self._train_tab = TrainTab(self._manager)
        self._history_tab = HistoryTab(manager=self._manager)
        self._convert_tab = ConvertTab()

        self._tabs.addTab(self._train_tab, "▶  Train")
        self._tabs.addTab(self._history_tab, "⏱  History")
        self._tabs.addTab(self._convert_tab, "⚡  Convert")

        # Refresh history when switching to it
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # Queue bar (footer)
        self._queue_bar = QueueBar(self._manager)
        layout.addWidget(self._queue_bar)

        # Preload all model config modules in background
        self._preloader = _Preloader()
        self._preloader.done.connect(
            lambda: self._status_lbl.setText("Ready"))
        self._preloader.start()

    def _on_tab_changed(self, index: int):
        if self._tabs.widget(index) is self._history_tab:
            self._history_tab.refresh()

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "Confirm Exit",
            "Are you sure you want to exit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            event.ignore()
            return
        self._manager.stop_current()
        self._train_tab._log.close_log_file()
        super().closeEvent(event)
