"""Background worker for pulling a full dataset into the local cache.

Runs ``FindNeuronConnection.build_connection_cache()`` in a worker thread so
the UI stays responsive; the Settings tab polls :attr:`DatasetPuller.state`
with a ``ui.timer`` and renders progress.

Supported workflows:
- **Full pull** - fetches every neuron's downstream connections and builds
  ``cache/<dataset>/connections.parquet`` + ``neuron_index.parquet``.
- **Resume / finish** - already-cached neurons are skipped and interrupted
  runs (crashes, network errors, manual cancel) are consolidated from their
  checkpoint batch files, so re-running simply continues where it stopped.
- **Force rebuild** - clears the existing cache first (use when the cache is
  broken and needs to be rebuilt completely).
- **Cancel** - stops after the current batch; fetched batches are consolidated
  first so the next run resumes cleanly.
"""

import threading
import time
from typing import Dict, Optional


class DatasetPuller:
    """One-shot background dataset puller (one pull at a time)."""

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

    def start(
        self,
        dataset: str,
        force_rebuild: bool = False,
        batch_size: int = 100,
        max_workers: Optional[int] = None,
    ) -> bool:
        """Start a background pull. Returns False when one is already running."""
        with self._lock:
            if self._state["running"]:
                return False
            self._state = {
                "running": True,
                "dataset": dataset,
                "current": 0,
                "total": 0,
                "info": "Connecting to dataset...",
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
            args=(dataset, force_rebuild, batch_size, max_workers),
            daemon=True,
            name=f"dataset-pull-{dataset}",
        )
        self._thread.start()
        return True

    def cancel(self) -> None:
        """Request a stop after the current batch (resume-safe)."""
        self._cancel_event.set()

    def _progress(self, current: int, total: int, info: str) -> None:
        with self._lock:
            self._state["current"] = current
            self._state["total"] = total
            self._state["info"] = info
            # ETA anchor: the first progress callback marks the start of the
            # actual fetching (after client init / cache bookkeeping).
            if self._state["fetch_started_at"] is None and total > 0:
                self._state["fetch_started_at"] = time.time()

    def _run(
        self,
        dataset: str,
        force_rebuild: bool,
        batch_size: int,
        max_workers: Optional[int],
    ) -> None:
        try:
            from coana import FindNeuronConnection

            fc = FindNeuronConnection(
                dataset=dataset,
                use_cache=True,
                cache_only=False,
                verbose=False,
            )
            summary = fc.build_connection_cache(
                batch_size=batch_size,
                force_rebuild=force_rebuild,
                quiet=True,
                progress_callback=self._progress,
                cancel_event=self._cancel_event,
                max_workers=max_workers,
            )
            with self._lock:
                self._state["summary"] = summary
                self._state["cancelled"] = bool(summary.get("cancelled"))
                self._state["info"] = "Finished."
                self._state["done"] = True
        except Exception as exc:  # network errors, token problems, ...
            with self._lock:
                self._state["error"] = f"{type(exc).__name__}: {exc}"
                self._state["info"] = "Failed."
                self._state["done"] = True
        finally:
            with self._lock:
                self._state["running"] = False
