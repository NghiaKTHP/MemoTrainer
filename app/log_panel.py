import re
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (QApplication, QCheckBox, QFileDialog,
                              QHBoxLayout, QLabel, QPushButton, QTextEdit,
                              QVBoxLayout, QWidget)

_LOG_COLORS = {
    "info":    "#1a6fb5",
    "warning": "#b45309",
    "warn":    "#b45309",
    "error":   "#b91c1c",
    "success": "#2d7a1a",
    "default": "#555555",
}

_MAX_DISPLAY = 200  # lines kept in the text widget

_PANEL_STYLE = """
QWidget#logPanel { background: #fafafa; border-left: 1px solid #e0e0e0; }
QTextEdit#logEdit {
    background: #fafafa; border: none;
    font-family: Consolas, "Courier New", monospace; font-size: 11px;
    color: #555;
}
QPushButton#logBtn {
    font-size: 11px; padding: 3px 8px;
    border: 1px solid #ddd; border-radius: 4px;
    background: transparent; color: #666;
}
QPushButton#logBtn:hover { background: #f0f0f0; color: #111; }
QLabel#metricChip {
    font-size: 10px; padding: 2px 7px; border-radius: 4px;
    background: #f0f0f0; border: 1px solid #e0e0e0; color: #555;
}
QLabel#headerLabel { font-size: 12px; font-weight: 500; }
QLabel#lineCount { font-size: 10px; color: #aaa; }
"""


def _today_log_path() -> Path:
    now = datetime.now()
    p = Path("Logs") / f"{now.year}" / f"{now.month:02d}" / f"{now.day:02d}" / "ProgramLog.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class LogPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("logPanel")
        self.setMinimumWidth(260)
        self.setStyleSheet(_PANEL_STYLE)

        self._auto_scroll = True
        self._metrics: dict[str, str] = {}
        self._display_count = 0   # lines currently in the text widget
        self._total_count = 0     # total lines written this session

        # Open log file (append mode, line-buffered so each line is flushed immediately)
        self._log_path = _today_log_path()
        self._log_file = open(self._log_path, "a", encoding="utf-8", buffering=1)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log_file.write(f"\n{'=' * 60}\nSession: {now_str}\n{'=' * 60}\n")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setStyleSheet("background: #fafafa; border-bottom: 1px solid #e0e0e0;")
        h_row = QHBoxLayout(header)
        h_row.setContentsMargins(8, 6, 8, 6)
        h_row.setSpacing(6)

        title = QLabel("Training Log")
        title.setObjectName("headerLabel")
        h_row.addWidget(title)

        self._line_lbl = QLabel("")
        self._line_lbl.setObjectName("lineCount")
        h_row.addWidget(self._line_lbl, 1)

        self._scroll_check = QCheckBox("Auto")
        self._scroll_check.setChecked(True)
        self._scroll_check.setStyleSheet("font-size: 11px; color: #666;")
        self._scroll_check.toggled.connect(lambda v: setattr(self, "_auto_scroll", v))
        h_row.addWidget(self._scroll_check)

        for text, slot in [("Clear",   self._clear_log),
                            ("Copy",   self._copy_log),
                            ("Open",   self._open_log_folder)]:
            btn = QPushButton(text)
            btn.setObjectName("logBtn")
            btn.clicked.connect(slot)
            h_row.addWidget(btn)

        layout.addWidget(header)

        # Log text area
        self._log_edit = QTextEdit()
        self._log_edit.setObjectName("logEdit")
        self._log_edit.setReadOnly(True)
        self._log_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self._log_edit, 1)

        # Metrics bar
        self._metrics_bar = QWidget()
        self._metrics_bar.setStyleSheet("border-top: 1px solid #e0e0e0; background: #fafafa;")
        self._metrics_row = QHBoxLayout(self._metrics_bar)
        self._metrics_row.setContentsMargins(8, 4, 8, 4)
        self._metrics_row.setSpacing(4)
        self._metrics_row.addStretch()
        layout.addWidget(self._metrics_bar)

    def close_log_file(self):
        """Call this when the app closes so the file handle is released cleanly."""
        try:
            self._log_file.close()
        except Exception:
            pass

    # ── Public slots ─────────────────────────────────────────────────

    @pyqtSlot(str, str)
    def append_log(self, level: str, message: str):
        # Metric updates: parse key=value pairs, update chips, skip text log
        if level == "Metric":
            for m in re.finditer(r'(\S+?)=([\d.]+)', message):
                self.update_metric(m.group(1), m.group(2))
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        color = _LOG_COLORS.get(level.lower(), _LOG_COLORS["default"])
        line = f"[{timestamp}] {message}"

        # Write to file immediately (line-buffered, no RAM accumulation)
        self._log_file.write(line + "\n")
        self._total_count += 1

        # Append to display widget
        cursor = self._log_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.setCharFormat(fmt)
        cursor.insertText(line + "\n")
        self._display_count += 1

        # Trim oldest line from display when over the limit
        if self._display_count > _MAX_DISPLAY:
            doc = self._log_edit.document()
            trim = QTextCursor(doc.begin())
            trim.select(QTextCursor.SelectionType.BlockUnderCursor)
            trim.movePosition(QTextCursor.MoveOperation.NextCharacter,
                              QTextCursor.MoveMode.KeepAnchor)
            trim.removeSelectedText()
            self._display_count -= 1

        if self._total_count > _MAX_DISPLAY:
            self._line_lbl.setText(f"last {_MAX_DISPLAY} / {self._total_count} lines")

        if self._auto_scroll:
            self._log_edit.verticalScrollBar().setValue(
                self._log_edit.verticalScrollBar().maximum()
            )

    @pyqtSlot(str)
    def append_stdout(self, line: str):
        self.append_log("default", line)

    def update_metric(self, key: str, value: str):
        self._metrics[key] = value
        self._rebuild_metrics_bar()

    def _rebuild_metrics_bar(self):
        while self._metrics_row.count() > 1:
            item = self._metrics_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for key, val in self._metrics.items():
            chip = QLabel(f"{key} <b>{val}</b>")
            chip.setObjectName("metricChip")
            chip.setTextFormat(Qt.TextFormat.RichText)
            self._metrics_row.insertWidget(self._metrics_row.count() - 1, chip)

    # ── Button actions ───────────────────────────────────────────────

    def _clear_log(self):
        """Clear the display only — file on disk is not affected."""
        self._log_edit.clear()
        self._display_count = 0
        self._line_lbl.setText("")
        self._metrics.clear()
        self._rebuild_metrics_bar()

    def _copy_log(self):
        QApplication.clipboard().setText(self._log_edit.toPlainText())

    def _open_log_folder(self):
        import os
        try:
            os.startfile(str(self._log_path.parent))
        except AttributeError:
            import subprocess
            subprocess.Popen(["explorer", str(self._log_path.parent)])
