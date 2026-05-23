import os
import subprocess
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                              QMessageBox, QPushButton, QTableWidget,
                              QTableWidgetItem, QVBoxLayout, QWidget)

from .constants import TRAIN_RESULT_DIR

_TAB_STYLE = """
QWidget#historyTab { background: #fff; }
QTableWidget {
    border: none; font-size: 12px;
    gridline-color: #f0f0f0; background: #fff;
}
QTableWidget::item { padding: 4px 8px; }
QTableWidget::item:selected { background: #f5f5f5; color: #111; }
QHeaderView::section {
    font-size: 11px; font-weight: 600; color: #aaa;
    text-transform: uppercase; letter-spacing: 0.5px;
    padding: 6px 8px; border: none;
    border-bottom: 1px solid #e0e0e0; background: #fafafa;
}
QPushButton#toolBtn {
    font-size: 11px; padding: 3px 10px;
    border: 1px solid #ddd; border-radius: 5px;
    background: transparent; color: #555;
}
QPushButton#toolBtn:hover { background: #f0f0f0; color: #111; }
QPushButton#actionBtn {
    font-size: 11px; padding: 2px 8px;
    border: 1px solid #ddd; border-radius: 4px;
    background: transparent; color: #555;
}
QPushButton#actionBtn:hover { background: #f0f0f0; }
QPushButton#dangerBtn {
    font-size: 11px; padding: 2px 8px;
    border: 1px solid #ddd; border-radius: 4px;
    background: transparent; color: #555;
}
QPushButton#dangerBtn:hover { background: #fef2f2; color: #b91c1c; border-color: #fca5a5; }
QLineEdit#filterEdit {
    font-size: 12px; padding: 3px 8px;
    border: 1px solid #ddd; border-radius: 4px; width: 130px;
}
"""

_COL_NAME    = 0
_COL_CREATED = 1
_COL_CKPT    = 2
_COL_STATUS  = 3
_COL_ACTS    = 4


def _fmt_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _find_checkpoint(folder: Path) -> tuple[str, float]:
    for name in ("last.pt", "last.pth"):
        candidates = list(folder.rglob(name))
        if candidates:
            p = max(candidates, key=lambda x: x.stat().st_mtime)
            return name, p.stat().st_mtime
    return "", 0.0


class HistoryTab(QWidget):
    def __init__(self, manager=None, parent=None):
        super().__init__(parent)
        self.setObjectName("historyTab")
        self.setStyleSheet(_TAB_STYLE)
        self._manager = manager  # TrainingManager reference to detect running jobs

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QWidget()
        toolbar.setStyleSheet("background: #fafafa; border-bottom: 1px solid #e0e0e0;")
        tb = QHBoxLayout(toolbar)
        tb.setContentsMargins(10, 6, 10, 6)
        tb.setSpacing(8)

        path_lbl = QLabel("TrainResult/")
        path_lbl.setStyleSheet("font-size: 12px; font-weight: 500;")
        tb.addWidget(path_lbl, 1)

        refresh_btn = QPushButton("↻  Refresh")
        refresh_btn.setObjectName("toolBtn")
        refresh_btn.clicked.connect(self.refresh)
        tb.addWidget(refresh_btn)

        self._filter_edit = QLineEdit()
        self._filter_edit.setObjectName("filterEdit")
        self._filter_edit.setPlaceholderText("Filter…")
        self._filter_edit.textChanged.connect(self._apply_filter)
        tb.addWidget(self._filter_edit)

        layout.addWidget(toolbar)

        # Table
        self._table = QTableWidget(0, 5)
        self._table.setObjectName("histTable")
        self._table.setHorizontalHeaderLabels(
            ["Process name", "Created", "Last checkpoint", "Status", "Actions"]
        )
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(4, 160)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(False)
        layout.addWidget(self._table, 1)

    def _running_folder_names(self) -> set[str]:
        """Return folder names currently being written by a running training job."""
        if not self._manager:
            return set()
        names: set[str] = set()
        for job in self._manager.jobs:
            from .training_manager import JobStatus
            if job.status == JobStatus.RUNNING:
                m = job.model_instance
                if m and getattr(m, "ProcessName", ""):
                    names.add(m.ProcessName)
        return names

    def refresh(self):
        self._table.setRowCount(0)

        root = Path(TRAIN_RESULT_DIR)
        if not root.exists():
            return

        running_names = self._running_folder_names()

        entries = sorted(
            [d for d in root.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_ctime,
            reverse=True,
        )

        for folder in entries:
            try:
                created_ts = folder.stat().st_ctime
                ckpt_name, ckpt_ts = _find_checkpoint(folder)

                if folder.name in running_names:
                    status_key, status_text = "running", "Running"
                elif ckpt_name:
                    status_key, status_text = "done", "Done"
                else:
                    status_key, status_text = "no_ckpt", "⚠ No checkpoint"

                self._add_table_row({
                    "name": folder.name,
                    "path": folder,
                    "created": created_ts,
                    "ckpt_name": ckpt_name,
                    "ckpt_ts": ckpt_ts,
                    "status_key": status_key,
                    "status_text": status_text,
                })
            except Exception:
                pass

    def _apply_filter(self, text: str):
        text = text.lower()
        for row_idx in range(self._table.rowCount()):
            item = self._table.item(row_idx, _COL_NAME)
            if item:
                self._table.setRowHidden(row_idx, bool(text) and text not in item.text().lower())

    def _add_table_row(self, data: dict):
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setRowHeight(row, 36)

        # Name
        self._table.setItem(row, _COL_NAME, QTableWidgetItem(f"  {data['name']}"))

        # Created
        created_item = QTableWidgetItem(_fmt_time(data["created"]) if data["created"] else "—")
        created_item.setForeground(Qt.GlobalColor.gray)
        self._table.setItem(row, _COL_CREATED, created_item)

        # Last checkpoint
        if data["ckpt_name"] and data["ckpt_ts"]:
            ckpt_str = f"{_fmt_time(data['ckpt_ts'])}  ({data['ckpt_name']})"
        else:
            ckpt_str = "—  (no checkpoint)"
        ckpt_item = QTableWidgetItem(ckpt_str)
        ckpt_item.setForeground(Qt.GlobalColor.gray)
        self._table.setItem(row, _COL_CKPT, ckpt_item)

        # Status
        status_item = QTableWidgetItem(f"  {data['status_text']}")
        if data["status_key"] == "done":
            status_item.setForeground(Qt.GlobalColor.darkGreen)
        elif data["status_key"] == "running":
            status_item.setForeground(Qt.GlobalColor.blue)
        else:
            status_item.setForeground(Qt.GlobalColor.gray)
        self._table.setItem(row, _COL_STATUS, status_item)

        # Actions
        acts = QWidget()
        acts_row = QHBoxLayout(acts)
        acts_row.setContentsMargins(4, 2, 4, 2)
        acts_row.setSpacing(4)

        open_btn = QPushButton("Open")
        open_btn.setObjectName("actionBtn")
        open_btn.clicked.connect(lambda _, p=data["path"]: self._open_folder(p))

        rm_btn = QPushButton("Remove")
        rm_btn.setObjectName("dangerBtn")
        rm_btn.clicked.connect(lambda _, p=data["path"]: self._remove_folder(p))

        acts_row.addWidget(open_btn)
        acts_row.addWidget(rm_btn)
        acts_row.addStretch()
        self._table.setCellWidget(row, _COL_ACTS, acts)

    @staticmethod
    def _open_folder(path: Path):
        try:
            os.startfile(str(path))
        except AttributeError:
            subprocess.Popen(["explorer", str(path)])

    def _remove_folder(self, path: Path):
        reply = QMessageBox.question(
            self,
            "Confirm Remove",
            f"Delete folder and all contents?\n\n{path.name}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        import shutil
        try:
            shutil.rmtree(str(path))
            QMessageBox.information(self, "Deleted", f"Deleted successfully:\n{path.name}")
        except Exception as e:
            QMessageBox.warning(self, "Delete Failed", f"Could not delete:\n{e}")
            return

        self.refresh()
