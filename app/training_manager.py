import gc
import importlib
import re
import sys
import threading
from enum import Enum
from typing import Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from .constants import MODEL_INFO

_ANSI_RE = re.compile(r'\x1b(?:\[[0-9;]*[A-Za-z]|\].*?(?:\x07|\x1b\\))')


class _StdoutCapture:
    def __init__(self, callback):
        self._cb = callback
        self._lock = threading.Lock()
        self._buf = ""

    def write(self, text: str):
        with self._lock:
            self._buf += text
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                # tqdm TTY mode: multiple \r overwrites accumulate before \n —
                # keep only the final overwrite (what the terminal would display)
                if "\r" in line:
                    line = line.rsplit("\r", 1)[1]
                clean = _ANSI_RE.sub("", line).strip()
                if clean:
                    self._cb(clean)

    def flush(self):
        pass

    def isatty(self):
        # Return True so tqdm/rich uses \r overwrite mode instead of printing
        # a new line per batch update — we collapse \r-separated updates above.
        return True


class JobStatus:
    PENDING = "Pending"
    RUNNING = "Running"
    DONE = "Done"
    FAILED = "Failed"


class TrainJob:
    _counter = 0

    def __init__(self, model_name: str, config_dict: dict, label: str = ""):
        TrainJob._counter += 1
        self.job_id = str(TrainJob._counter).zfill(4)
        self.model_name = model_name
        self.config_dict = config_dict
        self.label = label or f"{model_name}_{self.job_id}"
        self.status = JobStatus.PENDING
        self.model_instance = None


class TrainingWorker(QThread):
    log_signal = pyqtSignal(str, str)       # (level, message)
    stdout_signal = pyqtSignal(str)         # raw stdout line
    finished_signal = pyqtSignal(str, bool) # (job_id, success)

    def __init__(self, job: TrainJob):
        super().__init__()
        self.job = job
        self._stop_requested = False

    def run(self):
        original_stdout = sys.stdout
        capture = _StdoutCapture(lambda line: self.stdout_signal.emit(line))
        sys.stdout = capture

        try:
            info = MODEL_INFO[self.job.model_name]

            cfg_mod = importlib.import_module(info["config_module"])
            cfg_cls = getattr(cfg_mod, info["config_class"])

            model_mod = importlib.import_module(info["model_module"])
            model_cls = getattr(model_mod, info["model_class"])

            arch_enum_cls = None
            if info.get("arch_enum_module") and info.get("arch_enum_class"):
                arch_mod = importlib.import_module(info["arch_enum_module"])
                arch_enum_cls = getattr(arch_mod, info["arch_enum_class"], None)

            cfg = cfg_cls()

            # Apply fixed fields
            for k, v in info.get("fixed_fields", {}).items():
                try:
                    setattr(cfg, k, v)
                except Exception:
                    pass

            # Log key config values for diagnosis
            _dp = self.job.config_dict.get("DatasetPath", "<missing>")
            self.log_signal.emit("Info", f"[Worker] DatasetPath = '{_dp}'")

            # Apply user config
            for key, val in self.job.config_dict.items():
                if not hasattr(cfg, key) or key.startswith("_"):
                    continue
                current = getattr(cfg, key)
                if isinstance(current, Enum) or key == "Architecture":
                    enum_cls = arch_enum_cls if key == "Architecture" else type(current)
                    if isinstance(val, str):
                        try:
                            val = enum_cls[val]
                        except (KeyError, TypeError):
                            pass
                    elif isinstance(val, int):
                        try:
                            val = enum_cls(val)
                        except (ValueError, TypeError):
                            pass
                elif isinstance(current, tuple) and isinstance(val, list):
                    val = tuple(val)
                try:
                    setattr(cfg, key, val)
                except Exception:
                    pass

            model = model_cls()
            model.cfg = cfg
            self.job.model_instance = model

            def callback(level: str, message: str):
                self.log_signal.emit(level, message)

            model.Train(callbacks=callback)

            # Poll until training finishes (StopTrainingEvent may not fire on error)
            while getattr(model, "IsTraining", False) and not self._stop_requested:
                self.msleep(300)

            success = True

        except Exception as e:
            self.log_signal.emit("Error", f"Job setup failed: {e}")
            success = False
        finally:
            sys.stdout = original_stdout

        # Clear GPU cache between jobs
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

        self.log_signal.emit("Info", "[Queue] GPU cache cleared. Ready for next job.")
        self.finished_signal.emit(self.job.job_id, success)

    def request_stop(self):
        self._stop_requested = True
        m = self.job.model_instance
        if m and getattr(m, "IsTraining", False):
            try:
                m.IsStopTraining = True
            except Exception:
                pass


class TrainingManager(QObject):
    job_added      = pyqtSignal(object)       # TrainJob
    job_started    = pyqtSignal(str)          # job_id
    job_finished   = pyqtSignal(str, bool)    # job_id, success
    log_received   = pyqtSignal(str, str)     # level, message
    stdout_received = pyqtSignal(str)         # raw stdout line
    queue_changed  = pyqtSignal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._queue: list[TrainJob] = []
        self._worker: Optional[TrainingWorker] = None
        self._paused = False

    # ── Public API ──────────────────────────────────────────────────

    def add_job(self, model_name: str, config_dict: dict,
                label: str = "", start_immediately: bool = False) -> TrainJob:
        job = TrainJob(model_name, config_dict, label)
        self._queue.append(job)
        self.job_added.emit(job)
        self.queue_changed.emit()
        if start_immediately or (not self._worker_running and not self._paused):
            self._try_start_next()
        return job

    def stop_current(self):
        if self._worker:
            self._worker.request_stop()

    def pause_queue(self):
        self._paused = True

    def resume_queue(self):
        self._paused = False
        self._try_start_next()

    def remove_job(self, job_id: str):
        self._queue = [j for j in self._queue
                       if not (j.job_id == job_id and j.status == JobStatus.PENDING)]
        self.queue_changed.emit()

    def clear_pending(self):
        self._queue = [j for j in self._queue if j.status != JobStatus.PENDING]
        self.queue_changed.emit()

    @property
    def jobs(self) -> list[TrainJob]:
        return list(self._queue)

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def _worker_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    # ── Internal ────────────────────────────────────────────────────

    def _try_start_next(self):
        if self._worker_running or self._paused:
            return
        pending = [j for j in self._queue if j.status == JobStatus.PENDING]
        if not pending:
            return

        job = pending[0]
        job.status = JobStatus.RUNNING
        self.queue_changed.emit()
        self.job_started.emit(job.job_id)
        self.log_received.emit("Info", f"[Queue] Starting: {job.label}")

        worker = TrainingWorker(job)
        worker.log_signal.connect(self.log_received)
        worker.stdout_signal.connect(self.stdout_received)
        worker.finished_signal.connect(self._on_finished)
        self._worker = worker
        worker.start()

    def _on_finished(self, job_id: str, success: bool):
        for job in self._queue:
            if job.job_id == job_id:
                job.status = JobStatus.DONE if success else JobStatus.FAILED
                break
        self._worker = None
        self.job_finished.emit(job_id, success)
        self.queue_changed.emit()
        if not self._paused:
            self._try_start_next()
