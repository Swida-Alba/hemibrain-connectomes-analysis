"""Background worker for pulling dataset metadata and connections.

Runs in a worker thread so the UI stays responsive; the Settings tab polls
:attr:`DatasetPuller.state` with a ``ui.timer`` and renders progress.

Supported workflows:
- **Dataset metadata** (``operation='full_dataset'``) - downloads/verifies the
  full neuron table and ROI table (``datasets/<ds>/<ds>_allneurons_neuron_df.csv``
  + ``_roi_count_df.parquet``) and builds the materialized neuron index
  (``neuron_indexes/<ds>/neuron_index.parquet``).  No connections are fetched.
- **Connections** (``operation='connections'``) - fetches every neuron's
  downstream connections and builds ``cache/<ds>/connections.parquet``.
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

        ``operation`` selects what is pulled:
        - ``full_dataset``: dataset metadata only (neuron table + ROI table +
          materialized neuron index; no connections are fetched).
        - ``connections``: the complete resumable connection cache.
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
        operation = self._state.get("operation", "full_dataset")
        if operation == "connections":
            self._run_connections(dataset, force_rebuild, batch_size, max_workers)
        else:
            self._run_metadata(dataset)

    def _run_metadata(self, dataset: str) -> None:
        """Pull only the dataset metadata: neuron table, ROI table, index.

        ``FindNeuronConnection`` already ensures the metadata during init;
        the explicit calls below keep the button meaningful on its own and
        idempotent (already-present files are never re-downloaded).  No
        connection data is fetched or modified.
        """
        try:
            from coana import FindNeuronConnection

            self._phase(
                "prepare",
                "Checking local dataset files (a first pull downloads the "
                "full neuron table and ROI table if missing)...",
            )
            fc = FindNeuronConnection(
                dataset=dataset,
                use_cache=True,
                cache_only=False,
                verbose=False,
            )
            fc._ensure_complete_dataset()
            fc._ensure_neuron_index_from_metadata()
            if self._cancel_event.is_set():
                with self._lock:
                    self._state["summary"] = {
                        "operation": "full_dataset",
                        "cancelled": True,
                    }
                    self._state["cancelled"] = True
                    self._state["info"] = "Cancelled during metadata preparation."
                    self._state["done"] = True
                return
            neuron_count = 0
            index_rows = 0
            try:
                neuron_count = len(fc._get_all_dataset_bodyids() or [])
            except Exception:
                pass
            try:
                index_rows = len(fc._load_neuron_index())
            except Exception:
                pass
            with self._lock:
                self._state["summary"] = {
                    "operation": "full_dataset",
                    "neuron_count": neuron_count,
                    "index_rows": index_rows,
                    "elapsed_time": time.time() - self._state.get("started_at", time.time()),
                }
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

    def _run_connections(
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
