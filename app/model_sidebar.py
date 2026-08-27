from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (QButtonGroup, QFrame, QLabel, QPushButton,
                              QScrollArea, QVBoxLayout, QWidget)

from .constants import TASK_MODELS

_TASK_ICONS = {
    "Classify":  "🏷",
    "Detect":    "🎯",
    "Segment":   "✂",
    "Anomaly":   "🔍",
    "Recognize": "🔤",
}

_SIDEBAR_STYLE = """
QWidget#sidebar { background: #fafafa; border-right: 1px solid #e0e0e0; }
QPushButton#taskBtn {
    text-align: left;
    padding: 7px 10px 7px 12px;
    border: none;
    border-left: 2px solid transparent;
    background: transparent;
    font-size: 12px;
    color: #666;
}
QPushButton#taskBtn:hover { background: #f0f0f0; color: #111; }
QPushButton#taskBtn[active="true"] {
    background: #f0f0f0;
    color: #111;
    border-left: 2px solid #111;
    font-weight: 600;
}
QPushButton#modelBtn {
    text-align: left;
    padding: 6px 10px 6px 26px;
    border: none;
    background: transparent;
    font-size: 12px;
    color: #666;
}
QPushButton#modelBtn:hover { background: #f0f0f0; color: #111; }
QPushButton#modelBtn[active="true"] {
    background: #f0f0f0;
    color: #111;
    font-weight: 600;
}
QLabel#sectionLabel {
    font-size: 10px;
    font-weight: 600;
    color: #aaa;
    padding: 8px 10px 3px 10px;
    letter-spacing: 0.5px;
}
"""


class ModelSidebar(QWidget):
    selection_changed = pyqtSignal(str, str)  # (task, model)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(200)
        self.setStyleSheet(_SIDEBAR_STYLE)

        self._current_task = list(TASK_MODELS.keys())[0]
        self._current_model = TASK_MODELS[self._current_task][0]

        self._task_buttons: dict[str, QPushButton] = {}
        self._model_buttons: dict[str, QPushButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Task section
        task_lbl = QLabel("TASK")
        task_lbl.setObjectName("sectionLabel")
        layout.addWidget(task_lbl)

        for task in TASK_MODELS:
            icon = _TASK_ICONS.get(task, "•")
            btn = QPushButton(f"{icon}  {task}")
            btn.setObjectName("taskBtn")
            btn.setCheckable(False)
            btn.clicked.connect(lambda _, t=task: self._select_task(t))
            layout.addWidget(btn)
            self._task_buttons[task] = btn

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(divider)

        # Model section
        model_lbl = QLabel("MODEL")
        model_lbl.setObjectName("sectionLabel")
        layout.addWidget(model_lbl)

        self._model_container = QWidget()
        self._model_layout = QVBoxLayout(self._model_container)
        self._model_layout.setContentsMargins(0, 0, 0, 0)
        self._model_layout.setSpacing(0)
        layout.addWidget(self._model_container)

        layout.addStretch()

        # Initialize with first task
        self._select_task(self._current_task, emit=False)

    def _select_task(self, task: str, emit: bool = True):
        self._current_task = task

        # Update task button styles
        for t, btn in self._task_buttons.items():
            btn.setProperty("active", "true" if t == task else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        # Rebuild model list
        self._build_model_list(TASK_MODELS[task])

        if emit:
            self.selection_changed.emit(self._current_task, self._current_model)

    def _build_model_list(self, models: list[str]):
        # Clear existing
        while self._model_layout.count():
            item = self._model_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._model_buttons.clear()

        self._current_model = models[0]
        for i, model in enumerate(models):
            btn = QPushButton(f"  ◦  {model}")
            btn.setObjectName("modelBtn")
            btn.clicked.connect(lambda _, m=model: self._select_model(m))
            self._model_layout.addWidget(btn)
            self._model_buttons[model] = btn

        self._update_model_styles()

    def _select_model(self, model: str):
        self._current_model = model
        self._update_model_styles()
        self.selection_changed.emit(self._current_task, self._current_model)

    def _update_model_styles(self):
        for m, btn in self._model_buttons.items():
            btn.setProperty("active", "true" if m == self._current_model else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def select(self, task: str, model: str):
        """Programmatically switch task and model, emitting selection_changed."""
        if task not in self._task_buttons:
            return
        if task != self._current_task:
            self._select_task(task, emit=False)
        if model in self._model_buttons:
            self._select_model(model)

    @property
    def current_task(self) -> str:
        return self._current_task

    @property
    def current_model(self) -> str:
        return self._current_model
