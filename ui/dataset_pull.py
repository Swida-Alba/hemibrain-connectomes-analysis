"""Background worker for pulling a full dataset into the local cache.

Runs ``FindNeuronConnection.build_connection_cache()`` in a worker thread so
the UI stays responsive; the Settings tab polls :attr:`DatasetPuller.state`
with a ``ui.timer`` and renders progress.

Supported workflows:
- **Full pull** - fetches every neuron's downstream connections and builds
  ``cache/<dataset>/connections.parquet`` +
  ``neuron_indexes/<dataset>/neuron_index.parquet``.
- **Resume / finish** - already-cached neurons are skipped and interrupted
  runs (crashes, network errors, manual cancel) are consolidated from their
  checkpoint batch files, so re-running simply continues where it stopped.
- **Force rebuild** - clears the connection cache first (use when the cache is
  broken and needs to be rebuilt completely); the neuron index survives with
  reset progress flags.
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
            "operation": "full_dataset",
            "phase": "idle",
            "current": 0,
            "total": 0,
            "info": "",
            "done": False,
            "cancelled": False,
            "cancel_requested": False,
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
        operation: str = "full_dataset",
    ) -> bool:
        """Start a background pull.

        ``operation`` is a UI label (``full_dataset`` or ``connections``);
        both operations intentionally use the same resumable connection-cache
        builder so they share the same parquet cache and batching behavior.
        Returns False when one is already running.
        """
        with self._lock:
            if self._state["running"]:
                return False
            self._state = {
                "running": True,
                "dataset": dataset,
                "operation": operation,
                "phase": "prepare",
                "current": 0,
                "total": 0,
                "info": "Connecting to dataset...",
                "done": False,
                "cancelled": False,
                "cancel_requested": False,
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
        # The wind-down (in-flight batch, batch consolidation) can take a
        # while; expose the request so the UI can show a persistent
        # 'Cancelling...' hint until ``running`` flips to False.
        with self._lock:
            self._state["cancel_requested"] = True

    def _progress(self, current: int, total: int, info: str) -> None:
        with self._lock:
            self._state["current"] = current
            self._state["total"] = total
            self._state["info"] = info
            # ETA anchor: the first progress callback marks the start of the
            # actual fetching (after client init / cache bookkeeping).
            if self._state["fetch_started_at"] is None and total > 0:
                self._state["fetch_started_at"] = time.time()

    def _status(self, msg: str) -> None:
        """Live status strings (server reconnect / retry messages) shown in
        the Settings tab's status label while a batch retries."""
        with self._lock:
            self._state["info"] = msg

    def _phase(self, name: str, msg: str) -> None:
        """Report a named pipeline phase (``prepare`` / ``fetch``) so the UI
        can show what the pull is doing before the first progress callback
        (client connect, dataset download, index build) — these phases have
        no neuron totals yet and used to look like a hang at 0/0."""
        with self._lock:
            self._state["phase"] = name
            self._state["info"] = msg

    def _run(
        self,
        dataset: str,
        force_rebuild: bool,
        batch_size: int,
        max_workers: Optional[int],
    ) -> None:
        try:
            from coana import FindNeuronConnection

            # Phase 1/2 — client init + local data preparation.  A first pull
            # may download the full neuron table here (statvis.pull_dataset),
            # which can take several minutes with no neuron totals yet.
            self._phase(
                "prepare",
                "Connecting to the dataset server and preparing local data "
                "(a first pull may download the full neuron table — this can "
                "take several minutes; press Cancel to stop after it).",
            )
            fc = FindNeuronConnection(
                dataset=dataset,
                use_cache=True,
                cache_only=False,
                verbose=False,
            )
            if self._cancel_event.is_set():
                # Cancelled during preparation (e.g. while the full neuron
                # table was being downloaded): stop before the fetch loop.
                with self._lock:
                    self._state["summary"] = {
                        "total_neurons": 0,
                        "already_cached": 0,
                        "newly_cached": 0,
                        "failed_neurons": [],
                        "total_connections": 0,
                        "elapsed_time": 0.0,
                        "cancelled": True,
                    }
                    self._state["cancelled"] = True
                    self._state["info"] = "Cancelled during preparation."
                    self._state["done"] = True
                return

            # Phase 2/2 — batched connection fetch; per-batch progress
            # callbacks now drive the determinate bar.
            self._phase(
                "fetch",
                "Fetching connections in batches...",
            )
            summary = fc.build_connection_cache(
                batch_size=batch_size,
                force_rebuild=force_rebuild,
                quiet=True,
                progress_callback=self._progress,
                cancel_event=self._cancel_event,
                max_workers=max_workers,
                status_callback=self._status,
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
