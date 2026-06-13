"""Tab 3 — Export .pth → .trt  (torch_tensorrt dynamo)."""

import importlib
import sys
import traceback
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QScrollArea, QSplitter, QTextEdit, QVBoxLayout, QWidget,
)

# ── Converter registry ─────────────────────────────────────────────────────────
# fields: (var_name, type, default, label, choices_or_None)
# types:  "file" | "save" | "int" | "float" | "bool" | "enum"

CONVERTERS: dict[str, dict] = {
    "DinoUperNet → TRT": {
        "module":   "MemoLib.Converter.dinoUpernetTorhc2TRT",
        "arch_mod": "MemoLib.Model.BaseModel.eSegmentationModel",
        "arch_cls": "eDinoUperNetModel",
        "icon":     "✂",
        "fields": [
            ("WEIGHT_PATH",  "file",  "",       "Weight (.pth)",    None),
            ("ARCHITECTURE", "enum",  "DINO_S", "Architecture",     ["DINO_S", "DINO_B", "DINO_L"]),
            ("IMG_SIZE",     "int",   728,      "Image size (px)",  None),
            ("NUM_CLASSES",  "int",   2,        "Num classes",      None),
            ("DECODER_CH",   "int",   512,      "Decoder channels", None),
            ("DROPOUT",      "float", 0.1,      "Dropout",          None),
            ("BATCH",        "int",   5,        "Batch size",       None),
            ("FP16",         "bool",  True,     "FP16",             None),
        ],
    },
    "Dinomaly → TRT": {
        "module":   "MemoLib.Converter.dinomalyToTRT",
        "arch_mod": "MemoLib.Model.BaseModel.eAnomalyModel",
        "arch_cls": "eDinomalyModel",
        "icon":     "🔍",
        "fields": [
            ("WEIGHT_PATH",  "file",  "",            "Weight (.pth)",    None),
            ("ARCHITECTURE", "enum",  "Dinomaly_S",  "Architecture",     ["Dinomaly_S", "Dinomaly_B", "Dinomaly_L"]),
            ("IMG_SIZE",     "int",   728,           "Image size (px)",  None),
            ("LC",           "int",   2,             "Loose Constraint", None),
            ("DROPOUT",      "float", 0.4,           "Dropout",          None),
            ("LINEAR_ATTN",  "bool",  True,          "Linear Attention", None),
            ("CTX_RECENTER", "bool",  True,          "Context Recenter", None),
            ("BATCH",        "int",   5,             "Batch size",       None),
            ("FP16",         "bool",  True,          "FP16",             None),
        ],
    },
}


def _derive_out_path(weight_path: str) -> str:
    """<dir>/<stem>.pth → <dir>/<stem>.trt next to the input weight."""
    p = Path(weight_path)
    return str(p.with_suffix(".trt"))

_LOG_COLORS = {
    "info":    "#1a6fb5",
    "warning": "#b45309",
    "error":   "#b91c1c",
    "success": "#2d7a1a",
    "default": "#555555",
}


# ── Sidebar ────────────────────────────────────────────────────────────────────

_SIDEBAR_STYLE = """
QWidget#convertSidebar { background: #fafafa; border-right: 1px solid #e0e0e0; }
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


class _ConvertSidebar(QWidget):
    selection_changed = pyqtSignal(str)  # converter key

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("convertSidebar")
        self.setFixedWidth(200)
        self.setStyleSheet(_SIDEBAR_STYLE)

        self._current: str = list(CONVERTERS)[0]
        self._btns: dict[str, QPushButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        lbl = QLabel("CONVERTER")
        lbl.setObjectName("sectionLabel")
        layout.addWidget(lbl)

        for key, info in CONVERTERS.items():
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


# ── Form panel ─────────────────────────────────────────────────────────────────

_FORM_STYLE = """
QWidget#convertForm { background: #fff; }
QLabel#fieldLabel { font-size: 12px; color: #555; }
QLineEdit, QComboBox {
    font-size: 12px; padding: 3px 6px;
    border: 1px solid #ddd; border-radius: 4px;
    background: #fff; color: #111;
}
QLineEdit:focus, QComboBox:focus { border-color: #888; }
QComboBox QAbstractItemView {
    background: #fff; color: #111;
    selection-background-color: #1a6fb5; selection-color: #fff;
}
QCheckBox { color: #111; }
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


class _ConvertForm(QWidget):
    convert_requested = pyqtSignal(str, dict)  # (key, params)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("convertForm")
        self.setStyleSheet(_FORM_STYLE)

        self._current_key = ""
        self._field_widgets: dict[str, QWidget] = {}
        self._field_meta: dict[str, tuple] = {}  # var -> (type, default)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Toolbar
        toolbar = QWidget()
        toolbar.setStyleSheet("background: #fafafa; border-bottom: 1px solid #e0e0e0;")
        tb = QHBoxLayout(toolbar)
        tb.setContentsMargins(10, 6, 10, 6)
        self._title_lbl = QLabel("Select a converter")
        self._title_lbl.setStyleSheet("font-size: 12px; font-weight: 500;")
        tb.addWidget(self._title_lbl)
        tb.addStretch()
        outer.addWidget(toolbar)

        # Scrollable form
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setStyleSheet("QScrollArea { border: none; }")
        outer.addWidget(self._scroll, 1)

        # Action bar
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

        content = QWidget()
        content.setStyleSheet("background: #fff;")
        vl = QVBoxLayout(content)
        vl.setContentsMargins(12, 8, 12, 8)
        vl.setSpacing(2)

        for var, ftype, default, label, choices in CONVERTERS[key]["fields"]:
            self._field_meta[var] = (ftype, default)
            vl.addWidget(self._make_row(var, ftype, default, label, choices))

        vl.addStretch()
        self._scroll.setWidget(content)

    def _make_row(self, var: str, ftype: str, default, label: str, choices) -> QWidget:
        row_w = QWidget()
        row_l = QHBoxLayout(row_w)
        row_l.setContentsMargins(0, 2, 0, 2)
        row_l.setSpacing(6)

        lbl = QLabel(label)
        lbl.setObjectName("fieldLabel")
        lbl.setFixedWidth(148)
        row_l.addWidget(lbl)

        widget = self._make_input(ftype, default, choices)
        self._field_widgets[var] = widget
        row_l.addWidget(widget, 1)

        if ftype in ("file", "save"):
            btn = QPushButton("…")
            btn.setObjectName("browseBtn")
            btn.clicked.connect(lambda _, w=widget, t=ftype: self._browse(w, t))
            row_l.addWidget(btn)

        return row_w

    def _make_input(self, ftype: str, default, choices) -> QWidget:
        if ftype in ("file", "save", "int", "float"):
            return QLineEdit(str(default))
        if ftype == "bool":
            w = QCheckBox()
            w.setChecked(bool(default))
            return w
        if ftype == "enum":
            w = QComboBox()
            if choices:
                w.addItems(choices)
                if default in choices:
                    w.setCurrentIndex(choices.index(default))
            return w
        return QLineEdit(str(default))

    def _browse(self, line_edit: QLineEdit, ftype: str):
        if ftype == "file":
            path, _ = QFileDialog.getOpenFileName(
                self, "Select weight file", "", "PyTorch weights (*.pth);;All (*.*)"
            )
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save TRT engine", "", "TensorRT engine (*.trt);;All (*.*)"
            )
        if path:
            line_edit.setText(path)

    def collect_params(self) -> dict:
        params: dict = {}
        for var, widget in self._field_widgets.items():
            ftype, default = self._field_meta[var]
            if isinstance(widget, QLineEdit):
                text = widget.text().strip()
                if ftype == "int":
                    try:
                        params[var] = int(text)
                    except ValueError:
                        params[var] = default
                elif ftype == "float":
                    try:
                        params[var] = float(text)
                    except ValueError:
                        params[var] = default
                else:
                    params[var] = text
            elif isinstance(widget, QCheckBox):
                params[var] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                params[var] = widget.currentText()
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


# ── Log panel ──────────────────────────────────────────────────────────────────

_LOG_STYLE = """
QWidget#convertLog { background: #fafafa; border-left: 1px solid #e0e0e0; }
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


class _ConvertLog(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("convertLog")
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

    # ── Public API ─────────────────────────────────────────────────────

    def append(self, message: str, level: str = "default"):
        """Append a timestamped status line."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._insert(f"[{timestamp}] {message}\n", _LOG_COLORS.get(level, _LOG_COLORS["default"]))

    def feed_chunk(self, text: str):
        """Feed raw stdout chunk; buffer and emit complete lines."""
        self._line_buf += text
        while "\n" in self._line_buf:
            line, self._line_buf = self._line_buf.split("\n", 1)
            self._insert(line + "\n", self._line_color(line))

    def flush_buf(self):
        if self._line_buf:
            self._insert(self._line_buf + "\n", self._line_color(self._line_buf))
            self._line_buf = ""

    def clear(self):
        self._edit.clear()
        self._line_buf = ""

    # ── Internals ──────────────────────────────────────────────────────

    @staticmethod
    def _line_color(line: str) -> str:
        lower = line.lower()
        if any(k in lower for k in ("error", "traceback", "exception", "runtimeerror")):
            return _LOG_COLORS["error"]
        if any(k in lower for k in ("warn", "nan!")):
            return _LOG_COLORS["warning"]
        if "saved ->" in lower:
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

    def _copy(self):
        QApplication.clipboard().setText(self._edit.toPlainText())


# ── Background worker ──────────────────────────────────────────────────────────

class _ConvertWorker(QThread):
    log_chunk = pyqtSignal(str)
    done      = pyqtSignal(bool)

    def __init__(self, key: str, params: dict, parent=None):
        super().__init__(parent)
        self._key    = key
        self._params = params

    def run(self):
        info = CONVERTERS[self._key]

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
            mod = importlib.import_module(info["module"])
            arch_enum = getattr(
                importlib.import_module(info["arch_mod"]), info["arch_cls"]
            )

            weight = self._params.get("WEIGHT_PATH", "").strip()
            if not weight:
                raise ValueError("WEIGHT_PATH is empty — please select a .pth file.")

            out_path = _derive_out_path(weight)
            setattr(mod, "OUT_TRT_PATH", out_path)
            print(f"  Output will be saved to: {out_path}")

            for key, value in self._params.items():
                if key == "ARCHITECTURE":
                    value = arch_enum[value]
                setattr(mod, key, value)

            mod.convert()
            self.done.emit(True)

        except Exception:
            sys.stdout.write("\n" + traceback.format_exc())
            self.done.emit(False)

        finally:
            sys.stdout, sys.stderr = old_out, old_err


# ── Tab ────────────────────────────────────────────────────────────────────────

class ConvertTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: _ConvertWorker | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._sidebar = _ConvertSidebar()
        layout.addWidget(self._sidebar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet("QSplitter::handle { background: #e0e0e0; }")

        self._form = _ConvertForm()
        self._log  = _ConvertLog()
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

        self._form.set_running(True)
        self._form.set_status("")
        self._log.append(f"Starting  {key}", "info")

        self._worker = _ConvertWorker(key, params, self)
        self._worker.log_chunk.connect(self._log.feed_chunk)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _on_done(self, success: bool):
        self._log.flush_buf()
        self._form.set_running(False)
        if success:
            self._log.append("Conversion completed successfully.", "success")
            self._form.set_status("Done", "#2d7a1a")
        else:
            self._log.append("Conversion failed — see output above.", "error")
            self._form.set_status("Failed", "#b91c1c")
