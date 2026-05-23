from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QScrollArea,
                              QSizePolicy, QVBoxLayout, QWidget)

from .training_manager import JobStatus, TrainJob, TrainingManager

_STATUS_STYLE = {
    JobStatus.RUNNING: "background:#dbeafe; color:#1e40af;",
    JobStatus.PENDING: "background:#f0f0f0; color:#666;",
    JobStatus.DONE:    "background:#dcfce7; color:#166534;",
    JobStatus.FAILED:  "background:#fee2e2; color:#991b1b;",
}

_BAR_STYLE = """
QWidget#queueBar { background: #fafafa; border-top: 1px solid #e0e0e0; }
QPushButton#qbBtn {
    font-size: 11px; padding: 3px 9px;
    border: 1px solid #ddd; border-radius: 5px;
    background: transparent; color: #555;
}
QPushButton#qbBtn:hover { background: #f0f0f0; color: #111; }
QPushButton#qbBtn[active="true"] {
    background: #111; color: #fff; border-color: #111;
}
QLabel#qbTitle { font-size: 11px; font-weight: 500; color: #555; }
QLabel#jobName { font-size: 11px; font-weight: 500; color: #111; }
QLabel#jobStatus { font-size: 10px; padding: 1px 5px; border-radius: 4px; }
QPushButton#removeBtn {
    font-size: 10px; padding: 1px 4px;
    border: none; background: transparent; color: #aaa;
}
QPushButton#removeBtn:hover { color: #b91c1c; }
"""


class _JobCard(QWidget):
    remove_requested = pyqtSignal(str)  # job_id

    def __init__(self, job: TrainJob, parent=None):
        super().__init__(parent)
        self.job_id = job.job_id
        self.setStyleSheet(
            "QWidget { border: 1px solid #e0e0e0; border-radius: 6px; "
            "background: #fff; padding: 2px 4px; }"
        )

        row = QHBoxLayout(self)
        row.setContentsMargins(6, 3, 4, 3)
        row.setSpacing(5)

        name_lbl = QLabel(job.label)
        name_lbl.setObjectName("jobName")
        row.addWidget(name_lbl)

        self._status_lbl = QLabel(job.status)
        self._status_lbl.setObjectName("jobStatus")
        self._set_status(job.status)
        row.addWidget(self._status_lbl)

        self._rm_btn = QPushButton("✕")
        self._rm_btn.setObjectName("removeBtn")
        self._rm_btn.setFixedSize(16, 16)
        self._rm_btn.clicked.connect(lambda: self.remove_requested.emit(self.job_id))
        row.addWidget(self._rm_btn)

        self._update_removable(job.status)

    def _set_status(self, status: str):
        self._status_lbl.setText(status)
        style = _STATUS_STYLE.get(status, "background:#f0f0f0; color:#666;")
        self._status_lbl.setStyleSheet(f"QLabel {{ {style} border-radius: 4px; padding: 1px 5px; }}")

    def _update_removable(self, status: str):
        self._rm_btn.setVisible(status == JobStatus.PENDING)

    def update_status(self, status: str):
        self._set_status(status)
        self._update_removable(status)


class QueueBar(QWidget):
    def __init__(self, manager: TrainingManager, parent=None):
        super().__init__(parent)
        self.setObjectName("queueBar")
        self.setStyleSheet(_BAR_STYLE)
        self.setFixedHeight(80)

        self._manager = manager
        self._cards: dict[str, _JobCard] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(4)

        # Top row: title + buttons
        top_row = QHBoxLayout()
        top_row.setSpacing(6)

        title = QLabel("Training Queue")
        title.setObjectName("qbTitle")
        top_row.addWidget(title)
        top_row.addStretch()

        self._pause_btn = QPushButton("⏸ Pause")
        self._pause_btn.setObjectName("qbBtn")
        self._pause_btn.clicked.connect(self._toggle_pause)

        clear_btn = QPushButton("✕ Clear Pending")
        clear_btn.setObjectName("qbBtn")
        clear_btn.clicked.connect(self._manager.clear_pending)

        top_row.addWidget(self._pause_btn)
        top_row.addWidget(clear_btn)
        layout.addLayout(top_row)

        # Job card scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.frameShape().NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedHeight(36)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self._card_container = QWidget()
        self._card_container.setStyleSheet("background: transparent;")
        self._card_row = QHBoxLayout(self._card_container)
        self._card_row.setContentsMargins(0, 0, 0, 0)
        self._card_row.setSpacing(6)
        self._card_row.addStretch()
        scroll.setWidget(self._card_container)
        layout.addWidget(scroll)

        # Connect manager signals
        manager.job_added.connect(self._on_job_added)
        manager.job_started.connect(self._on_job_started)
        manager.job_finished.connect(self._on_job_finished)
        manager.queue_changed.connect(self._on_queue_changed)

    def _on_job_added(self, job: TrainJob):
        card = _JobCard(job)
        card.remove_requested.connect(self._manager.remove_job)
        self._cards[job.job_id] = card
        # Insert before stretch
        self._card_row.insertWidget(self._card_row.count() - 1, card)

    def _on_job_started(self, job_id: str):
        if job_id in self._cards:
            self._cards[job_id].update_status(JobStatus.RUNNING)

    def _on_job_finished(self, job_id: str, success: bool):
        if job_id in self._cards:
            self._cards[job_id].update_status(
                JobStatus.DONE if success else JobStatus.FAILED
            )

    def _on_queue_changed(self):
        # Sync cards with current jobs
        current_ids = {j.job_id for j in self._manager.jobs}
        stale = [jid for jid in self._cards if jid not in current_ids]
        for jid in stale:
            card = self._cards.pop(jid)
            self._card_row.removeWidget(card)
            card.deleteLater()

    def _toggle_pause(self):
        if self._manager.is_paused:
            self._manager.resume_queue()
            self._pause_btn.setText("⏸ Pause")
            self._pause_btn.setProperty("active", "false")
        else:
            self._manager.pause_queue()
            self._pause_btn.setText("▶ Resume")
            self._pause_btn.setProperty("active", "true")
        self._pause_btn.style().unpolish(self._pause_btn)
        self._pause_btn.style().polish(self._pause_btn)
