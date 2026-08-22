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

    def start(self, dataset: str, max_workers: Optional[int] = None,
              raw: bool = True, mode: Optional[str] = None,
              simplification: Optional[int] = None,
              batch_size: Optional[int] = None) -> bool:
        """Start a background skeleton download. Returns False when one is
        already running.

        ``raw`` and ``mode`` are retained for callers from the earlier Find
        Similar UI. Pulls always use the shared raw compressed-SWC
        representation (``.swc.zst``); ``simplification`` selects the percent
        of nodes removed when writing the cache (0-90; ``None`` keeps the
        backend default of 90). ``batch_size`` bounds the per-call NeuPrint
        request size (``None`` keeps the backend default of 64).  Progress is
        reported per completed skeleton regardless of the batch size; larger
        batches only reduce the repeated per-batch metadata queries.
        Visualization ``fast`` is a render-time simplification mode, not a
        pull mode.
        """
        selected_mode = "raw"
        if simplification is not None:
            simplification = int(simplification)
        if batch_size is not None:
            batch_size = int(batch_size)
        with self._lock:
            if self._state["running"]:
                return False
            self._state = {
                "running": True,
                "dataset": dataset,
                "mode": selected_mode,
                "simplification": simplification,
                "batch_size": batch_size,
                "current": 0,
                "total": 0,
                "info": "Reading the neuron index...",
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
            args=(dataset, max_workers, selected_mode, simplification,
                  batch_size),
            daemon=True,
            name=f"skeleton-pull-{dataset}",
        )
        self._thread.start()
        return True

    def cancel(self) -> None:
        """Request a stop (resume-safe)."""
        self._cancel_event.set()
        # The wind-down (in-flight navis batch, persist phase) can take a
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
            # actual fetching (after index reading / cache bookkeeping).
            if self._state["fetch_started_at"] is None and total > 0:
                self._state["fetch_started_at"] = time.time()

    def _run(self, dataset: str, max_workers: Optional[int], mode: str,
             simplification: Optional[int], batch_size: Optional[int]) -> None:
        try:
            import sys
            sys.path.insert(0, str(
                __import__("pathlib").Path(__file__).resolve().parents[1] / "src"
            ))
            from coana import FindNeuronConnection
            from statvis import DatasetPullCancelled
            from morphology import download_all_skeletons

            # The skeleton download reads the bodyId list from the local
            # neuron table, so the dataset metadata must be present first.
            # Ensure it (idempotent; a first pull downloads the neuron table
            # + ROI table and builds the materialized index).
            def _prepare_download(current: int, total: int) -> None:
                # A first pull may download the full neuron table here; stream
                # it into the state so the UI shows a determinate bar instead
                # of an indeterminate 'Preparing...' during the download.
                self._progress(
                    current, total,
                    "Downloading the full neuron table and ROI table...",
                )

            with self._lock:
                self._state["info"] = (
                    "Ensuring dataset metadata (neuron table, ROI table, "
                    "index)..."
                )
            fc = FindNeuronConnection(
                dataset=dataset,
                use_cache=True,
                cache_only=False,
                verbose=False,
                progress_callback=_prepare_download,
                cancel_event=self._cancel_event,
            )
            fc._ensure_complete_dataset()
            fc._ensure_neuron_index_from_metadata()
            if self._cancel_event.is_set():
                # Cancelled while the metadata was being prepared: stop
                # before any skeleton fetch.
                with self._lock:
                    self._state["summary"] = {
                        "total": 0, "fetched": 0, "skipped_existing": 0,
                        "errors": 0, "cancelled": True,
                    }
                    self._state["cancelled"] = True
                    self._state["info"] = "Cancelled during metadata preparation."
                    self._state["done"] = True
                return

            pull_kwargs = dict(
                max_workers=max_workers or 8,
                progress_callback=self._progress,
                cancel_event=self._cancel_event,
                verbose=False,
                mode=mode,
            )
            if simplification is not None:
                pull_kwargs["simplification"] = int(simplification)
            if batch_size is not None:
                pull_kwargs["batch_size"] = int(batch_size)
            summary = download_all_skeletons(dataset, **pull_kwargs)
            with self._lock:
                self._state["summary"] = summary
                self._state["cancelled"] = bool(summary.get("cancelled"))
                self._state["info"] = "Cancelled." if summary.get("cancelled") else "Finished."
                self._state["done"] = True
        except DatasetPullCancelled:
            # A first-time neuron-table download was cancelled during the
            # metadata preparation; mark the pull as cancelled, not failed.
            with self._lock:
                self._state["summary"] = {
                    "total": 0, "fetched": 0, "skipped_existing": 0,
                    "errors": 0, "cancelled": True,
                }
                self._state["cancelled"] = True
                self._state["info"] = "Cancelled during metadata preparation."
                self._state["done"] = True
        except Exception as exc:  # network errors, token problems, ...
            with self._lock:
                self._state["error"] = f"{type(exc).__name__}: {exc}"
                self._state["info"] = "Failed."
                self._state["done"] = True
        finally:
            with self._lock:
                self._state["running"] = False
