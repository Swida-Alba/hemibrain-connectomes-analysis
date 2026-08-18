"""Background worker for downloading a dataset's full skeleton set.

Mirrors ``DatasetPuller`` (Settings tab full dataset pull): runs
``morphology.download_all_skeletons`` in a worker thread so the UI stays
responsive; the Settings tab polls :attr:`SkeletonPuller.state` with a
``ui.timer`` and renders progress/ETA.

Resume-safe: already-cached skeletons are skipped, so interrupted runs
(crashes, network errors, manual cancel) simply continue where they stopped.
"""

import threading
import time
from typing import Dict, Optional


class SkeletonPuller:
    """One-shot background skeleton downloader (one download at a time)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state: Dict = {
            "running": False,
            "dataset": None,
            "current": 0,
            "total": 0,
            "info": "",
            "done": False,
            "cancelled": False,
            "error": None,
            "summary": None,
            "started_at": None,
            "fetch_started_at": None,
        }

    @property
    def state(self) -> Dict:
        """Snapshot of the current pull state (thread-safe)."""
        with self._lock:
            return dict(self._state)

    @property
    def running(self) -> bool:
        with self._lock:
            return self._state["running"]

    def start(self, dataset: str, max_workers: Optional[int] = None,
              raw: bool = True, mode: Optional[str] = None) -> bool:
        """Start a background skeleton download. Returns False when one is
        already running.

        ``raw`` and ``mode`` are retained for callers from the earlier Find
        Similar UI. Pulls always use the shared raw compressed-SWC
        representation; visualization ``fast`` is a render-time
        simplification mode, not a pull mode.
        """
        selected_mode = "raw"
        with self._lock:
            if self._state["running"]:
                return False
            self._state = {
                "running": True,
                "dataset": dataset,
                "mode": selected_mode,
                "current": 0,
                "total": 0,
                "info": "Reading the neuron index...",
                "done": False,
                "cancelled": False,
                "error": None,
                "summary": None,
                "started_at": time.time(),
                "fetch_started_at": None,
            }
            self._cancel_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(dataset, max_workers, selected_mode),
            daemon=True,
            name=f"skeleton-pull-{dataset}",
        )
        self._thread.start()
        return True

    def cancel(self) -> None:
        """Request a stop (resume-safe)."""
        self._cancel_event.set()

    def _progress(self, current: int, total: int, info: str) -> None:
        with self._lock:
            self._state["current"] = current
            self._state["total"] = total
            self._state["info"] = info
            # ETA anchor: the first progress callback marks the start of the
            # actual fetching (after index reading / cache bookkeeping).
            if self._state["fetch_started_at"] is None and total > 0:
                self._state["fetch_started_at"] = time.time()

    def _run(self, dataset: str, max_workers: Optional[int], mode: str) -> None:
        try:
            import sys
            sys.path.insert(0, str(
                __import__("pathlib").Path(__file__).resolve().parents[1] / "src"
            ))
            from morphology import download_all_skeletons

            summary = download_all_skeletons(
                dataset,
                max_workers=max_workers or 8,
                progress_callback=self._progress,
                cancel_event=self._cancel_event,
                verbose=False,
                mode=mode,
            )
            with self._lock:
                self._state["summary"] = summary
                self._state["cancelled"] = bool(summary.get("cancelled"))
                self._state["info"] = "Cancelled." if summary.get("cancelled") else "Finished."
                self._state["done"] = True
        except Exception as exc:  # network errors, token problems, ...
            with self._lock:
                self._state["error"] = f"{type(exc).__name__}: {exc}"
                self._state["info"] = "Failed."
                self._state["done"] = True
        finally:
            with self._lock:
                self._state["running"] = False
