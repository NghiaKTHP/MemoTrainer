import ast
import dataclasses
import importlib
import re
from enum import Enum
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFrame,
                              QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                              QMessageBox, QPushButton, QScrollArea,
                              QSizePolicy, QToolButton, QVBoxLayout, QWidget)

from .config_io import (apply_dict_to_config, config_to_dict,
                        load_config_file, save_config)
from .constants import MODEL_INFO

_CONFIGS_DIR = Path("Configs")

_PANEL_STYLE = """
QWidget#configPanel { background: #fff; }
QLabel#groupTitle {
    font-size: 10px; font-weight: 600; color: #aaa;
    letter-spacing: 0.5px; padding: 4px 0 2px 0;
}
QLabel#fieldLabel { font-size: 12px; color: #555; }
QLabel#changedBadge {
    font-size: 10px; padding: 1px 5px; border-radius: 3px;
    background: #fff8e1; color: #b45309; font-weight: 500;
}
QLineEdit, QComboBox {
    font-size: 12px; padding: 3px 6px;
    border: 1px solid #ddd; border-radius: 4px;
    background: #fff; color: #111;
}
QLineEdit:focus, QComboBox:focus { border-color: #888; }
QLineEdit:disabled, QComboBox:disabled { background: #f5f5f5; color: #aaa; }
QComboBox QAbstractItemView {
    background: #fff; color: #111;
    selection-background-color: #1a6fb5; selection-color: #fff;
}
QCheckBox { color: #111; }
QPushButton#toolBtn {
    font-size: 11px; padding: 4px 10px;
    border: 1px solid #ddd; border-radius: 6px;
    background: transparent; color: #555;
}
QPushButton#toolBtn:hover { background: #f0f0f0; color: #111; }
QPushButton#toolBtn:disabled { color: #bbb; border-color: #eee; }
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
QPushButton#queueBtn {
    font-size: 12px; padding: 6px 12px;
    border: 1px solid #ddd; border-radius: 6px;
    background: transparent; color: #555;
}
QPushButton#queueBtn:hover { background: #f0f0f0; }
QFrame#groupFrame { border-top: 1px solid #f0f0f0; }
"""


def _browse_type(field_name: str) -> str | None:
    """Return 'folder', 'file', or None based on field name."""
    lower = field_name.lower()
    folder_keywords = ("datasetpath", "dataset_path", "datadir", "data_dir")
    if any(kw in lower for kw in folder_keywords):
        return "folder"
    if lower.endswith("dir") or field_name == "Project":
        return "folder"
    if lower.endswith("path") or lower.endswith("file"):
        return "file"
    return None


class _FieldRow(QWidget):
    value_changed = pyqtSignal()

    def __init__(self, name: str, value: Any, default_value: Any,
                 arch_enum_cls=None, fixed: bool = False,
                 browse: str | None = None, parent=None):
        super().__init__(parent)
        self.name = name
        self._default = default_value
        self._arch_enum_cls = arch_enum_cls
        self._browse_type = browse

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(6)

        lbl = QLabel(name)
        lbl.setObjectName("fieldLabel")
        lbl.setFixedWidth(155)
        lbl.setToolTip(name)
        row.addWidget(lbl)

        self._input = self._make_input(value)
        if fixed:
            self._input.setEnabled(False)
        row.addWidget(self._input, 1)

        if browse and isinstance(self._input, QLineEdit):
            btn = QPushButton("…")
            btn.setObjectName("browseBtn")
            btn.setToolTip("Browse folder" if browse == "folder" else "Browse file")
            btn.clicked.connect(self._browse)
            row.addWidget(btn)

        self._badge = QLabel("changed")
        self._badge.setObjectName("changedBadge")
        self._badge.setVisible(False)
        row.addWidget(self._badge)

    def _make_input(self, value: Any) -> QWidget:
        if isinstance(value, bool):
            w = QCheckBox()
            w.setChecked(value)
            w.toggled.connect(self._on_changed)
            return w

        if isinstance(value, Enum):
            enum_cls = (self._arch_enum_cls
                        if self.name == "Architecture" and self._arch_enum_cls
                        else type(value))
            w = QComboBox()
            for member in enum_cls:
                w.addItem(member.name, member)
            self._set_combo_to(w, value, enum_cls)
            w.currentIndexChanged.connect(self._on_changed)
            return w

        if isinstance(value, (tuple, list)):
            w = QLineEdit(repr(value))
            w.textChanged.connect(self._on_changed)
            return w

        w = QLineEdit("" if value is None else str(value))
        w.textChanged.connect(self._on_changed)
        return w

    @staticmethod
    def _set_combo_to(combo: QComboBox, value, enum_cls):
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                return
        if isinstance(value, Enum):
            idx = combo.findText(value.name)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def _browse(self):
        current = self._input.text().strip()
        if self._browse_type == "folder":
            path = QFileDialog.getExistingDirectory(
                self, "Select Folder", current or "."
            )
            if path:
                self._input.setText(path)
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select File", current or ".",
                "Model weights (*.pt *.pth *.bin *.weights *.onnx);;"
                "Text files (*.txt *.yaml *.yml *.json);;All files (*)"
            )
            if path:
                self._input.setText(path)

    def _on_changed(self):
        changed = (self.get_value() != self._default)
        self._badge.setVisible(changed)
        self.value_changed.emit()

    def get_value(self) -> Any:
        w = self._input
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, QComboBox):
            return w.currentData()
        if isinstance(w, QLineEdit):
            text = w.text().strip()
            dv = self._default
            if dv is None:
                return text if text else None
            if isinstance(dv, bool):
                return text.lower() in ("true", "1", "yes")
            if isinstance(dv, int):
                try:
                    return int(text)
                except ValueError:
                    return text
            if isinstance(dv, float):
                try:
                    return float(text)
                except ValueError:
                    return text
            if isinstance(dv, (tuple, list)):
                try:
                    parsed = ast.literal_eval(text)
                    return type(dv)(parsed)
                except Exception:
                    return text
            return text
        return None

    def set_value(self, value: Any):
        w = self._input
        if isinstance(w, QCheckBox):
            w.setChecked(bool(value))
        elif isinstance(w, QComboBox):
            if isinstance(value, Enum):
                self._set_combo_to(w, value, type(value))
            elif isinstance(value, str):
                idx = w.findText(value)
                if idx >= 0:
                    w.setCurrentIndex(idx)
        elif isinstance(w, QLineEdit):
            if value is None:
                w.setText("")
            elif isinstance(value, (tuple, list)):
                w.setText(repr(value))
            else:
                w.setText(str(value))


class _CollapsibleGroup(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(0)

        header = QWidget()
        h_row = QHBoxLayout(header)
        h_row.setContentsMargins(0, 0, 0, 4)
        h_row.setSpacing(4)

        self._toggle = QToolButton()
        self._toggle.setArrowType(Qt.ArrowType.DownArrow)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.setStyleSheet("QToolButton { border: none; }")
        self._toggle.toggled.connect(self._on_toggle)

        lbl = QLabel(title.upper())
        lbl.setObjectName("groupTitle")

        h_row.addWidget(self._toggle)
        h_row.addWidget(lbl)
        h_row.addStretch()
        layout.addWidget(header)

        sep = QFrame()
        sep.setObjectName("groupFrame")
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(8, 2, 0, 4)
        self._body_layout.setSpacing(0)
        layout.addWidget(self._body)

    def _on_toggle(self, checked: bool):
        self._body.setVisible(checked)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

    def add_row(self, row: QWidget):
        self._body_layout.addWidget(row)


class ConfigPanel(QWidget):
    config_changed = pyqtSignal()
    start_requested = pyqtSignal(dict, str)   # (config_dict, model_name)
    queue_requested = pyqtSignal(dict, str)   # (config_dict, model_name)
    model_switch_requested = pyqtSignal(str, str)  # (task, model_name)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("configPanel")
        self.setStyleSheet(_PANEL_STYLE)

        self._model_name: str = ""
        self._cfg_instance = None
        self._arch_enum_cls = None
        self._default_cfg: dict = {}
        self._field_rows: dict[str, _FieldRow] = {}
        self._pending_load_data: dict | None = None
        self._syncing = False  # guard against recursion in _propagate_change

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Toolbar
        toolbar = QWidget()
        toolbar.setStyleSheet("background: #fafafa; border-bottom: 1px solid #e0e0e0;")
        tb_row = QHBoxLayout(toolbar)
        tb_row.setContentsMargins(10, 6, 10, 6)
        tb_row.setSpacing(6)

        self._title_lbl = QLabel("Select a model")
        self._title_lbl.setStyleSheet("font-size: 12px; font-weight: 500;")
        tb_row.addWidget(self._title_lbl)

        self._config_combo = QComboBox()
        self._config_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._config_combo.setMinimumWidth(120)
        self._config_combo.currentIndexChanged.connect(self._on_combo_changed)
        tb_row.addWidget(self._config_combo, 1)

        self._save_btn = QPushButton("Save")
        self._save_btn.setObjectName("toolBtn")
        self._save_btn.clicked.connect(self._save_config)
        tb_row.addWidget(self._save_btn)

        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setObjectName("toolBtn")
        self._delete_btn.setEnabled(False)
        self._delete_btn.clicked.connect(self._delete_config)
        tb_row.addWidget(self._delete_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.setObjectName("toolBtn")
        reset_btn.clicked.connect(self._reset_config)
        tb_row.addWidget(reset_btn)

        outer.addWidget(toolbar)

        # Scroll area for config fields
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        self._scroll_content = QWidget()
        self._scroll_layout = QVBoxLayout(self._scroll_content)
        self._scroll_layout.setContentsMargins(12, 8, 12, 8)
        self._scroll_layout.setSpacing(2)
        self._scroll_layout.addStretch()

        scroll.setWidget(self._scroll_content)
        outer.addWidget(scroll, 1)

        # Action bar
        action_bar = QWidget()
        action_bar.setStyleSheet("border-top: 1px solid #e0e0e0;")
        ab_row = QHBoxLayout(action_bar)
        ab_row.setContentsMargins(10, 8, 10, 8)
        ab_row.setSpacing(8)

        self._start_btn = QPushButton("▶  Start Training")
        self._start_btn.setObjectName("startBtn")
        self._start_btn.clicked.connect(self._on_start)
        ab_row.addWidget(self._start_btn, 1)

        self._queue_btn = QPushButton("+ Queue")
        self._queue_btn.setObjectName("queueBtn")
        self._queue_btn.clicked.connect(self._on_queue)
        ab_row.addWidget(self._queue_btn)

        outer.addWidget(action_bar)

    # ── Public API ──────────────────────────────────────────────────

    def load_model(self, model_name: str):
        self._model_name = model_name
        info = MODEL_INFO[model_name]

        try:
            cfg_mod = importlib.import_module(info["config_module"])
            cfg_cls = getattr(cfg_mod, info["config_class"])
            arch_enum_cls = None
            if info.get("arch_enum_module") and info.get("arch_enum_class"):
                arch_mod = importlib.import_module(info["arch_enum_module"])
                arch_enum_cls = getattr(arch_mod, info["arch_enum_class"], None)
        except Exception as e:
            self._title_lbl.setText(f"Error loading config: {e}")
            return

        self._arch_enum_cls = arch_enum_cls
        self._cfg_instance = cfg_cls()

        for k, v in info.get("fixed_fields", {}).items():
            try:
                setattr(self._cfg_instance, k, v)
            except Exception:
                pass

        if arch_enum_cls is not None:
            try:
                default_arch_name = info.get("default_arch")
                if default_arch_name:
                    first_arch = arch_enum_cls[default_arch_name]
                else:
                    first_arch = list(arch_enum_cls)[0]
                setattr(self._cfg_instance, "Architecture", first_arch)
            except Exception:
                pass

        self._default_cfg = config_to_dict(self._cfg_instance)
        self._title_lbl.setText(f"  {model_name}")
        self._rebuild_form()
        self._refresh_config_list()

        if self._pending_load_data is not None:
            data, self._pending_load_data = self._pending_load_data, None
            self._apply_config_data(data)

    def get_config_dict(self) -> dict:
        result = {}
        for name, row in self._field_rows.items():
            val = row.get_value()
            if isinstance(val, Enum):
                result[name] = val.name
            else:
                result[name] = val
        return result

    def set_training_active(self, active: bool):
        self._start_btn.setEnabled(not active)
        self._start_btn.setText("⏹ Running..." if active else "▶  Start Training")

    # ── Config list management ──────────────────────────────────────

    def _refresh_config_list(self):
        self._config_combo.blockSignals(True)
        self._config_combo.clear()
        self._config_combo.addItem("— select config —", userData=None)

        if _CONFIGS_DIR.exists():
            for model_dir in sorted(_CONFIGS_DIR.iterdir()):
                if not model_dir.is_dir():
                    continue
                for p in sorted(model_dir.glob("*.yaml")):
                    self._config_combo.addItem(f"{model_dir.name}  /  {p.stem}", userData=p)

        self._config_combo.setCurrentIndex(0)
        self._delete_btn.setEnabled(False)
        self._config_combo.blockSignals(False)

    def _on_combo_changed(self, index: int):
        path: Path | None = self._config_combo.currentData()
        self._delete_btn.setEnabled(path is not None)
        if path is None:
            return
        self._load_from_file(path)

    def _load_from_file(self, path: Path):
        try:
            data = load_config_file(str(path))
            model_hint = data.pop("_model", None)

            if model_hint and model_hint != self._model_name:
                from .constants import TASK_MODELS
                task = next((t for t, ms in TASK_MODELS.items() if model_hint in ms), None)
                if task:
                    self._pending_load_data = data
                    self.model_switch_requested.emit(task, model_hint)
                    return

            self._apply_config_data(data)
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))

    # ── Form building ───────────────────────────────────────────────

    def _apply_config_data(self, data: dict):
        apply_dict_to_config(self._cfg_instance, data, self._arch_enum_cls)
        for fname, row in self._field_rows.items():
            if hasattr(self._cfg_instance, fname):
                row.set_value(getattr(self._cfg_instance, fname))

    def _propagate_change(self, fname: str, row: _FieldRow):
        """Apply the changed field to cfg_instance, then sync any other UI rows
        that the dataclass's __setattr__ may have updated as a side effect
        (e.g. Dinomaly: Architecture change → Backbone auto-resolves)."""
        if self._syncing or self._cfg_instance is None:
            return
        if not hasattr(self._cfg_instance, fname):
            return

        before = {f: getattr(self._cfg_instance, f, None) for f in self._field_rows}

        try:
            setattr(self._cfg_instance, fname, row.get_value())
        except Exception:
            return

        self._syncing = True
        try:
            for f, r in self._field_rows.items():
                if f == fname:
                    continue
                current = getattr(self._cfg_instance, f, None)
                if current != before[f]:
                    r.set_value(current)
        finally:
            self._syncing = False

    def _rebuild_form(self):
        while self._scroll_layout.count():
            item = self._scroll_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._field_rows.clear()

        if not self._cfg_instance:
            self._scroll_layout.addStretch()
            return

        fixed_keys = set(MODEL_INFO[self._model_name].get("fixed_fields", {}).keys())
        category_map = getattr(self._cfg_instance, "_category_map", {})

        for group_name, field_names in category_map.items():
            group_widget = _CollapsibleGroup(group_name)
            for fname in field_names:
                if not hasattr(self._cfg_instance, fname):
                    continue
                val = getattr(self._cfg_instance, fname)
                default_val = self._default_cfg.get(fname)
                if isinstance(val, Enum) and isinstance(default_val, str):
                    default_val = val

                row = _FieldRow(
                    name=fname,
                    value=val,
                    default_value=default_val,
                    arch_enum_cls=self._arch_enum_cls if fname == "Architecture" else None,
                    fixed=(fname in fixed_keys),
                    browse=_browse_type(fname),
                )
                row.value_changed.connect(self.config_changed)
                row.value_changed.connect(
                    lambda f=fname, r=row: self._propagate_change(f, r))
                group_widget.add_row(row)
                self._field_rows[fname] = row
            self._scroll_layout.addWidget(group_widget)

        self._scroll_layout.addStretch()

    # ── Toolbar actions ─────────────────────────────────────────────

    def _save_config(self):
        if not self._model_name:
            return
        name, ok = QInputDialog.getText(
            self, "Save Config", "Config name:",
            text=self._model_name
        )
        if not ok or not name.strip():
            return
        safe_name = re.sub(r'[<>:"/\\|?*]', "_", name.strip())

        configs_dir = _CONFIGS_DIR / self._model_name
        configs_dir.mkdir(parents=True, exist_ok=True)
        path = configs_dir / f"{safe_name}.yaml"

        try:
            apply_dict_to_config(self._cfg_instance, self.get_config_dict(), self._arch_enum_cls)
            save_config(self._cfg_instance, str(path), extra={"_model": self._model_name})
        except Exception as e:
            QMessageBox.warning(self, "Save Error", str(e))
            return

        self._refresh_config_list()
        display = f"{self._model_name}  /  {safe_name}"
        idx = self._config_combo.findText(display)
        if idx >= 0:
            self._config_combo.blockSignals(True)
            self._config_combo.setCurrentIndex(idx)
            self._config_combo.blockSignals(False)
            self._delete_btn.setEnabled(True)

    def _delete_config(self):
        path: Path | None = self._config_combo.currentData()
        if path is None:
            return
        reply = QMessageBox.question(
            self, "Delete Config",
            f"Delete config '{path.stem}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            path.unlink(missing_ok=True)
        except Exception as e:
            QMessageBox.warning(self, "Delete Error", str(e))
            return
        self._refresh_config_list()

    def _reset_config(self):
        if not self._model_name:
            return
        self.load_model(self._model_name)

    # ── Training actions ────────────────────────────────────────────

    def _on_start(self):
        if not self._model_name:
            return
        config = self.get_config_dict()
        if "DatasetPath" in config and not config["DatasetPath"]:
            QMessageBox.warning(
                self, "Missing Configuration",
                "DatasetPath is empty.\nPlease enter the path to your dataset directory."
            )
            return
        self.start_requested.emit(config, self._model_name)

    def _on_queue(self):
        if not self._model_name:
            return
        config = self.get_config_dict()
        if "DatasetPath" in config and not config["DatasetPath"]:
            QMessageBox.warning(
                self, "Missing Configuration",
                "DatasetPath is empty.\nPlease enter the path to your dataset directory."
            )
            return
        self.queue_requested.emit(config, self._model_name)
