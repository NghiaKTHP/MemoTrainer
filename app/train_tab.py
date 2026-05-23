from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QSplitter, QWidget

from .config_panel import ConfigPanel
from .log_panel import LogPanel
from .model_sidebar import ModelSidebar
from .training_manager import JobStatus, TrainingManager


class TrainTab(QWidget):
    def __init__(self, manager: TrainingManager, parent=None):
        super().__init__(parent)
        self._manager = manager
        self._current_model = ""

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._sidebar = ModelSidebar()
        layout.addWidget(self._sidebar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet("QSplitter::handle { background: #e0e0e0; }")

        self._config = ConfigPanel()
        self._log = LogPanel()
        self._log.setMinimumWidth(0)

        splitter.addWidget(self._config)
        splitter.addWidget(self._log)
        splitter.setSizes([1, 1])  # equal split, resolved at paint time
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter, 1)

        # Wire sidebar → config
        self._sidebar.selection_changed.connect(self._on_selection_changed)

        # Wire config → training manager
        self._config.start_requested.connect(self._on_start)
        self._config.queue_requested.connect(self._on_queue)
        self._config.model_switch_requested.connect(self._sidebar.select)

        # Wire manager → log + config
        manager.log_received.connect(self._log.append_log)
        manager.stdout_received.connect(self._log.append_stdout)
        manager.job_started.connect(self._on_job_started)
        manager.job_finished.connect(self._on_job_finished)

        # Load initial model
        first_task = list(self._sidebar._task_buttons.keys())[0]
        first_model = self._sidebar.current_model
        self._on_selection_changed(first_task, first_model)

    def _on_selection_changed(self, task: str, model: str):
        self._current_model = model
        self._config.load_model(model)

    def _on_start(self, config_dict: dict, model_name: str):
        process_name = config_dict.get("ProcessName", "").strip()
        label = process_name if process_name else model_name
        self._manager.add_job(
            model_name=model_name,
            config_dict=config_dict,
            label=label,
            start_immediately=True,
        )

    def _on_queue(self, config_dict: dict, model_name: str):
        process_name = config_dict.get("ProcessName", "").strip()
        label = process_name if process_name else model_name
        self._manager.add_job(
            model_name=model_name,
            config_dict=config_dict,
            label=label,
            start_immediately=False,
        )

    def _on_job_started(self, job_id: str):
        self._config.set_training_active(True)

    def _on_job_finished(self, job_id: str, success: bool):
        if not self._manager._worker_running:
            self._config.set_training_active(False)
        status = "Done" if success else "Failed"
        self._log.append_log("success" if success else "error",
                             f"[Queue] Job {job_id} {status}.")
