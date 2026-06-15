"""Tab — Convert AnyLabeling/LabelMe datasets → DinoUperNet / RFDETR formats."""

import sys
import traceback
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QScrollArea, QSplitter, QTextEdit, QVBoxLayout, QWidget,
)

# Static imports so PyInstaller picks them up.
from DatasetConverter import Anylabeling2DinoUpernet as _dino_conv
from DatasetConverter import Anylabeling2RFDETR as _rfdetr_conv


# ── Dataset-converter registry ────────────────────────────────────────────────
# field types: "dir" | "int" | "float" | "list"
#   "list" → space-separated strings → list[str]

DATASET_CONVERTERS: dict[str, dict] = {
    "AnyLabeling → DinoUperNet": {
        "icon":   "🟦",
        "desc":   "LabelMe polygons → semantic segmentation masks (images + masks + colormap).",
        "runner": lambda p: _dino_conv.run(
            src=p["src"], dst=p["dst"],
            ignore=p["ignore"], priority=p["priority"],
            val_ratio=p["val_ratio"], seed=p["seed"],
        ),
        "fields": [
            ("src",       "dir",   "",                       "Source folder"),
            ("dst",       "dir",   "",                       "Output folder"),
            ("ignore",    "list",  "",                       "Ignore classes (space-sep)"),
            ("priority",  "list",  "Laser Metal Point",      "Priority order (space-sep)"),
            ("val_ratio", "float", 0.1,                      "Val ratio"),
            ("seed",      "int",   42,                       "Seed"),
        ],
    },
    "AnyLabeling → RFDETR": {
        "icon":   "🟨",
        "desc":   "LabelMe polygons → YOLO instance-seg format (data.yaml + train/valid).",
        "runner": lambda p: _rfdetr_conv.run(
            src=p["src"], dst=p["dst"],
            ignore=p["ignore"],
            val_ratio=p["val_ratio"], seed=p["seed"],
        ),
        "fields": [
            ("src",       "dir",   "",   "Source folder"),
            ("dst",       "dir",   "",   "Output folder"),
            ("ignore",    "list",  "",   "Ignore classes (space-sep)"),
            ("val_ratio", "float", 0.15, "Val ratio"),
            ("seed",      "int",   42,   "Seed"),
        ],
    },
}


_LOG_COLORS = {
    "info":    "#1a6fb5",
    "warning": "#b45309",
    "error":   "#b91c1c",
    "success": "#2d7a1a",
    "default": "#555555",
}


# ── Sidebar ───────────────────────────────────────────────────────────────────

_SIDEBAR_STYLE = """
QWidget#dsSidebar { background: #fafafa; border-right: 1px solid #e0e0e0; }
QLabel#sectionLabel {
    font-size: 10px; font-weight: 600; color: #aaa;
    padding: 8px 10px 3px 10px; letter-spacing: 0.5px;
}
QPushButton#converterBtn {
    text-align: left;
    padding: 6px 10px 6px 14px;
    border: none;
    border-left: 2px solid transparent;
    background: transparent;
    font-size: 12px;
    color: #666;
}
QPushButton#converterBtn:hover { background: #f0f0f0; color: #111; }
QPushButton#converterBtn[active="true"] {
    background: #f0f0f0;
    color: #111;
    border-left: 2px solid #111;
    font-weight: 600;
}
"""


class _DsSidebar(QWidget):
    selection_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dsSidebar")
        self.setFixedWidth(220)
        self.setStyleSheet(_SIDEBAR_STYLE)

        self._current: str = list(DATASET_CONVERTERS)[0]
        self._btns: dict[str, QPushButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        lbl = QLabel("DATASET CONVERTER")
        lbl.setObjectName("sectionLabel")
        layout.addWidget(lbl)

        for key, info in DATASET_CONVERTERS.items():
            icon = info.get("icon", "◦")
            btn = QPushButton(f"{icon}  {key}")
            btn.setObjectName("converterBtn")
            btn.clicked.connect(lambda _, k=key: self._select(k))
            layout.addWidget(btn)
            self._btns[key] = btn

        layout.addStretch()
        self._refresh_styles()

    def _select(self, key: str):
        self._current = key
        self._refresh_styles()
        self.selection_changed.emit(key)

    def _refresh_styles(self):
        for k, btn in self._btns.items():
            btn.setProperty("active", "true" if k == self._current else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    @property
    def current(self) -> str:
        return self._current


# ── Form ──────────────────────────────────────────────────────────────────────

_FORM_STYLE = """
QWidget#dsForm { background: #fff; }
QLabel#fieldLabel { font-size: 12px; color: #555; }
QLabel#descLabel  { font-size: 11px; color: #888; padding: 0 0 6px 0; }
QLineEdit {
    font-size: 12px; padding: 3px 6px;
    border: 1px solid #ddd; border-radius: 4px;
    background: #fff; color: #111;
}
QLineEdit:focus { border-color: #888; }
QPushButton#browseBtn {
    font-size: 11px; padding: 3px 8px;
    border: 1px solid #ddd; border-radius: 4px;
    background: transparent; color: #555; min-width: 28px;
}
QPushButton#browseBtn:hover { background: #f0f0f0; color: #111; }
QPushButton#startBtn {
    font-size: 12px; padding: 6px 14px;
    border: 1px solid #111; border-radius: 6px;
    background: #111; color: #fff; font-weight: 500;
}
QPushButton#startBtn:hover { background: #333; }
QPushButton#startBtn:disabled { background: #ccc; border-color: #ccc; color: #888; }
"""


class _DsForm(QWidget):
    convert_requested = pyqtSignal(str, dict)  # (key, typed params)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dsForm")
        self.setStyleSheet(_FORM_STYLE)

        self._current_key = ""
        self._field_widgets: dict[str, QLineEdit] = {}
        self._field_meta: dict[str, tuple] = {}  # name -> (ftype, default)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        toolbar = QWidget()
        toolbar.setStyleSheet("background: #fafafa; border-bottom: 1px solid #e0e0e0;")
        tb = QHBoxLayout(toolbar)
        tb.setContentsMargins(10, 6, 10, 6)
        self._title_lbl = QLabel("Select a converter")
        self._title_lbl.setStyleSheet("font-size: 12px; font-weight: 500;")
        tb.addWidget(self._title_lbl)
        tb.addStretch()
        outer.addWidget(toolbar)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setStyleSheet("QScrollArea { border: none; }")
        outer.addWidget(self._scroll, 1)

        action_bar = QWidget()
        action_bar.setStyleSheet("border-top: 1px solid #e0e0e0;")
        ab = QHBoxLayout(action_bar)
        ab.setContentsMargins(10, 8, 10, 8)
        ab.setSpacing(10)

        self._convert_btn = QPushButton("⚡  Convert")
        self._convert_btn.setObjectName("startBtn")
        self._convert_btn.clicked.connect(self._on_convert)
        ab.addWidget(self._convert_btn, 1)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("font-size: 11px; color: #888;")
        ab.addWidget(self._status_lbl)

        outer.addWidget(action_bar)

    def load_converter(self, key: str):
        self._current_key = key
        self._field_widgets.clear()
        self._field_meta.clear()
        self._title_lbl.setText(f"  {key}")
        self._status_lbl.setText("")

        info = DATASET_CONVERTERS[key]

        content = QWidget()
        content.setStyleSheet("background: #fff;")
        vl = QVBoxLayout(content)
        vl.setContentsMargins(12, 8, 12, 8)
        vl.setSpacing(2)

        desc = QLabel(info.get("desc", ""))
        desc.setObjectName("descLabel")
        desc.setWordWrap(True)
        vl.addWidget(desc)

        for name, ftype, default, label in info["fields"]:
            self._field_meta[name] = (ftype, default)
            vl.addWidget(self._make_row(name, ftype, default, label))

        vl.addStretch()
        self._scroll.setWidget(content)

    def _make_row(self, name: str, ftype: str, default, label: str) -> QWidget:
        row_w = QWidget()
        row_l = QHBoxLayout(row_w)
        row_l.setContentsMargins(0, 2, 0, 2)
        row_l.setSpacing(6)

        lbl = QLabel(label)
        lbl.setObjectName("fieldLabel")
        lbl.setFixedWidth(168)
        row_l.addWidget(lbl)

        edit = QLineEdit(str(default))
        self._field_widgets[name] = edit
        row_l.addWidget(edit, 1)

        if ftype == "dir":
            btn = QPushButton("…")
            btn.setObjectName("browseBtn")
            btn.clicked.connect(lambda _, w=edit: self._browse_dir(w))
            row_l.addWidget(btn)

        return row_w

    def _browse_dir(self, line_edit: QLineEdit):
        start = line_edit.text().strip() or ""
        path = QFileDialog.getExistingDirectory(self, "Select folder", start)
        if path:
            line_edit.setText(path)

    def collect_params(self) -> dict:
        """Return name → typed Python value."""
        params: dict = {}
        for name, widget in self._field_widgets.items():
            ftype, default = self._field_meta[name]
            text = widget.text().strip()
            if ftype == "list":
                params[name] = text.split() if text else []
            elif ftype == "int":
                try: params[name] = int(text)
                except ValueError: params[name] = default
            elif ftype == "float":
                try: params[name] = float(text)
                except ValueError: params[name] = default
            else:
                params[name] = text
        return params

    def set_running(self, running: bool):
        self._convert_btn.setEnabled(not running)
        self._convert_btn.setText("⏹  Running…" if running else "⚡  Convert")

    def set_status(self, text: str, color: str = "#888"):
        self._status_lbl.setText(text)
        self._status_lbl.setStyleSheet(f"font-size: 11px; color: {color};")

    def _on_convert(self):
        if not self._current_key:
            return
        self.convert_requested.emit(self._current_key, self.collect_params())


# ── Log panel ─────────────────────────────────────────────────────────────────

_LOG_STYLE = """
QWidget#dsLog { background: #fafafa; border-left: 1px solid #e0e0e0; }
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
"""


class _DsLog(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dsLog")
        self.setMinimumWidth(260)
        self.setStyleSheet(_LOG_STYLE)

        self._line_buf = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QWidget()
        header.setStyleSheet("background: #fafafa; border-bottom: 1px solid #e0e0e0;")
        h = QHBoxLayout(header)
        h.setContentsMargins(8, 6, 8, 6)
        h.setSpacing(6)

        title = QLabel("Output")
        title.setStyleSheet("font-size: 12px; font-weight: 500;")
        h.addWidget(title)
        h.addStretch()

        for text, slot in [("Copy", self._copy), ("Clear", self.clear)]:
            btn = QPushButton(text)
            btn.setObjectName("logBtn")
            btn.clicked.connect(slot)
            h.addWidget(btn)

        layout.addWidget(header)

        self._edit = QTextEdit()
        self._edit.setObjectName("logEdit")
        self._edit.setReadOnly(True)
        self._edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self._edit, 1)

    def append(self, message: str, level: str = "default"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._insert(f"[{timestamp}] {message}\n", _LOG_COLORS.get(level, _LOG_COLORS["default"]))

    def feed_chunk(self, text: str):
        self._line_buf += text
        while "\n" in self._line_buf or "\r" in self._line_buf:
            nl = self._line_buf.find("\n")
            cr = self._line_buf.find("\r")
            if nl == -1: idx, sep = cr, "\r"
            elif cr == -1: idx, sep = nl, "\n"
            else:
                idx = min(nl, cr); sep = self._line_buf[idx]
            line = self._line_buf[:idx]
            self._line_buf = self._line_buf[idx+1:]
            if sep == "\r":
                self._replace_last(line, self._line_color(line))
            else:
                self._insert(line + "\n", self._line_color(line))

    def flush_buf(self):
        if self._line_buf:
            self._insert(self._line_buf + "\n", self._line_color(self._line_buf))
            self._line_buf = ""

    def clear(self):
        self._edit.clear()
        self._line_buf = ""

    @staticmethod
    def _line_color(line: str) -> str:
        lower = line.lower()
        if any(k in lower for k in ("error", "traceback", "exception")):
            return _LOG_COLORS["error"]
        if any(k in lower for k in ("warn", "[warn]")):
            return _LOG_COLORS["warning"]
        if "done" in lower:
            return _LOG_COLORS["success"]
        return _LOG_COLORS["default"]

    def _insert(self, text: str, color: str):
        cursor = self._edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.setCharFormat(fmt)
        cursor.insertText(text)
        self._edit.setTextCursor(cursor)
        self._edit.verticalScrollBar().setValue(
            self._edit.verticalScrollBar().maximum()
        )

    def _replace_last(self, text: str, color: str):
        cursor = self._edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.select(QTextCursor.SelectionType.LineUnderCursor)
        cursor.removeSelectedText()
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.setCharFormat(fmt)
        cursor.insertText(text)
        self._edit.setTextCursor(cursor)
        self._edit.verticalScrollBar().setValue(
            self._edit.verticalScrollBar().maximum()
        )

    def _copy(self):
        QApplication.clipboard().setText(self._edit.toPlainText())


# ── Worker ────────────────────────────────────────────────────────────────────

class _DsWorker(QThread):
    log_chunk = pyqtSignal(str)
    done      = pyqtSignal(bool)

    def __init__(self, key: str, params: dict, parent=None):
        super().__init__(parent)
        self._key    = key
        self._params = params

    def run(self):
        runner = DATASET_CONVERTERS[self._key]["runner"]

        class _Tee:
            def __init__(self, emit):
                self._emit = emit
            def write(self, text: str):
                if text:
                    self._emit(text)
            def flush(self): pass

        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = _Tee(self.log_chunk.emit)

        try:
            runner(self._params)
            self.done.emit(True)
        except Exception:
            sys.stdout.write("\n" + traceback.format_exc())
            self.done.emit(False)
        finally:
            sys.stdout, sys.stderr = old_out, old_err


# ── Tab ───────────────────────────────────────────────────────────────────────

class DatasetConvertTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: _DsWorker | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._sidebar = _DsSidebar()
        layout.addWidget(self._sidebar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet("QSplitter::handle { background: #e0e0e0; }")

        self._form = _DsForm()
        self._log  = _DsLog()
        self._log.setMinimumWidth(0)

        splitter.addWidget(self._form)
        splitter.addWidget(self._log)
        splitter.setSizes([1, 1])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter, 1)

        self._sidebar.selection_changed.connect(self._form.load_converter)
        self._form.convert_requested.connect(self._on_convert)

        self._form.load_converter(self._sidebar.current)

    def _on_convert(self, key: str, params: dict):
        if self._worker and self._worker.isRunning():
            return

        if not params.get("src") or not params.get("dst"):
            self._log.append("Source and Output folders are required.", "error")
            self._form.set_status("Missing folders", "#b91c1c")
            return

        self._form.set_running(True)
        self._form.set_status("")
        self._log.append(f"Starting  {key}", "info")
        self._log.append(f"  params: {params}", "info")

        self._worker = _DsWorker(key, params, self)
        self._worker.log_chunk.connect(self._log.feed_chunk)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _on_done(self, success: bool):
        self._log.flush_buf()
        self._form.set_running(False)
        if success:
            self._log.append("Conversion completed.", "success")
            self._form.set_status("Done", "#2d7a1a")
        else:
            self._log.append("Conversion failed — see output above.", "error")
            self._form.set_status("Failed", "#b91c1c")
