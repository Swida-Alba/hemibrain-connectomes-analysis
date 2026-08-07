#!/usr/bin/env python
"""
Tests for the "Pull Full Dataset" operation:

- ``ui.dataset_pull.DatasetPuller``: background thread lifecycle, progress
  reporting, cancellation, error handling, and single-pull-at-a-time guard.
- ``coana.FindNeuronConnection.build_connection_cache``: cooperative
  ``cancel_event`` stops after the current batch and consolidates fetched
  batches (resume-safe), and quiet mode still consolidates batch files.
"""

import sys
import threading
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ui.dataset_pull import DatasetPuller  # noqa: E402


# ---------------------------------------------------------------------------
# DatasetPuller
# ---------------------------------------------------------------------------

class _FakeBuild:
    """Simulates build_connection_cache: emits progress, honours cancel_event."""

    def __init__(self, neurons: int = 50, fail: bool = False, delay: float = 0.0):
        self.neurons = neurons
        self.fail = fail
        self.delay = delay
        self.called_with = None
        self.cancel_event = None

    def __call__(self, **kwargs):
        self.called_with = kwargs
        self.cancel_event = kwargs.get("cancel_event")
        progress_callback = kwargs.get("progress_callback")
        total = self.neurons
        for i in range(0, total, 10):
            if self.cancel_event is not None and self.cancel_event.is_set():
                return {
                    "total_neurons": total,
                    "newly_cached": i,
                    "already_cached": 0,
                    "failed_neurons": [],
                    "total_connections": i * 3,
                    "elapsed_time": 1.0,
                    "cancelled": True,
                }
            if progress_callback:
                progress_callback(i, total, f"Batch {i // 10 + 1}")
            if self.delay:
                time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("connection reset")
        return {
            "total_neurons": total,
            "newly_cached": total,
            "already_cached": 0,
            "failed_neurons": [],
            "total_connections": total * 3,
            "elapsed_time": 2.0,
            "cancelled": False,
        }


@pytest.fixture
def fake_fnc(monkeypatch):
    """Monkeypatch coana.FindNeuronConnection so DatasetPuller uses _FakeBuild."""
    import coana

    holder = {"build": None}

    class FakeFNC:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def build_connection_cache(self, **kwargs):
            assert holder["build"] is not None
            return holder["build"](**kwargs)

    monkeypatch.setattr(coana, "FindNeuronConnection", FakeFNC)
    return holder


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class TestDatasetPuller:
    def test_full_pull_reports_progress_and_summary(self, fake_fnc):
        fake_fnc["build"] = _FakeBuild(neurons=50)
        puller = DatasetPuller()
        assert puller.start("hemibrain:v1.2.1", batch_size=10) is True
        assert _wait_until(lambda: puller.state["done"])
        st = puller.state
        assert st["running"] is False
        assert st["error"] is None
        assert st["cancelled"] is False
        assert st["summary"]["newly_cached"] == 50
        # progress was reported
        build = fake_fnc["build"]
        assert build.called_with["quiet"] is True
        assert build.called_with["force_rebuild"] is False
        assert build.called_with["batch_size"] == 10
        # progress callback reached the state
        assert st["total"] == 50

    def test_force_rebuild_passed_through(self, fake_fnc):
        fake_fnc["build"] = _FakeBuild()
        puller = DatasetPuller()
        puller.start("male-cns:v0.9", force_rebuild=True, batch_size=25)
        assert _wait_until(lambda: puller.state["done"])
        assert fake_fnc["build"].called_with["force_rebuild"] is True
        assert fake_fnc["build"].called_with["batch_size"] == 25

    def test_max_workers_passed_through(self, fake_fnc):
        fake_fnc["build"] = _FakeBuild()
        puller = DatasetPuller()
        puller.start("manc:v1.2.1", batch_size=10, max_workers=6)
        assert _wait_until(lambda: puller.state["done"])
        assert fake_fnc["build"].called_with["max_workers"] == 6

    def test_eta_timestamps_recorded(self, fake_fnc):
        fake_fnc["build"] = _FakeBuild(neurons=50, delay=0.02)
        puller = DatasetPuller()
        puller.start("hemibrain:v1.2.1", batch_size=10)
        # while running: fetch_started_at is set after the first progress tick
        assert _wait_until(lambda: puller.state.get("fetch_started_at") is not None)
        st = puller.state
        assert st["started_at"] is not None
        assert st["fetch_started_at"] >= st["started_at"]
        assert st["total"] == 50
        assert _wait_until(lambda: puller.state["done"])

    def test_cancel_stops_early_and_is_resume_safe(self, fake_fnc):
        fake_fnc["build"] = _FakeBuild(neurons=100, delay=0.05)
        build = fake_fnc["build"]
        puller = DatasetPuller()
        puller.start("optic-lobe:v1.1", batch_size=10)
        assert _wait_until(lambda: puller.state["total"] == 100)
        puller.cancel()
        assert _wait_until(lambda: puller.state["done"])
        st = puller.state
        assert st["cancelled"] is True
        assert st["summary"]["newly_cached"] < 100  # stopped before finishing
        # the cancel event was handed to the builder (resume checkpoint)
        assert build.cancel_event is not None and build.cancel_event.is_set()

    def test_error_is_captured(self, fake_fnc):
        fake_fnc["build"] = _FakeBuild(fail=True)
        puller = DatasetPuller()
        puller.start("banc:v888")
        assert _wait_until(lambda: puller.state["done"])
        st = puller.state
        assert st["running"] is False
        assert "RuntimeError" in st["error"]

    def test_second_start_while_running_is_rejected(self, fake_fnc):
        fake_fnc["build"] = _FakeBuild(neurons=200, delay=0.02)
        puller = DatasetPuller()
        assert puller.start("hemibrain:v1.2.1") is True
        assert puller.start("manc:v1.2.1") is False  # already running
        assert _wait_until(lambda: puller.state["done"])
        # after completion a new pull is allowed again
        assert puller.start("manc:v1.2.1") is True
        assert _wait_until(lambda: puller.state["done"])
        assert puller.state["dataset"] == "manc:v1.2.1"


# ---------------------------------------------------------------------------
# build_connection_cache cancel_event + quiet consolidation (real code)
# ---------------------------------------------------------------------------

class TestBuildConnectionCacheCancel:
    def test_parallel_fetch_uses_multiple_workers(self, tmp_path, monkeypatch):
        """max_workers>1 fetches batches concurrently (bounded in-flight) while
        producing the same consolidated cache as a sequential run."""
        import coana
        import neuprint
        import threading

        class FakeClient:
            def __init__(self, *a, **k):
                pass

        monkeypatch.setattr(neuprint, "Client", FakeClient)

        fc = coana.FindNeuronConnection(
            dataset="fake:v3",
            use_cache=True,
            cache_only=False,
            verbose=False,
            script_path=str(tmp_path),
            cache_folder=str(tmp_path / "cache" / "fake_v3"),
        )

        import time
        import pandas as pd

        active = 0
        max_active = 0
        active_lock = threading.Lock()

        def fake_fetch(upstream_bodyIds, downstream_bodyIds=None):
            nonlocal active, max_active
            with active_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.1)
            try:
                return pd.DataFrame({
                    "bodyId_pre": upstream_bodyIds,
                    "bodyId_post": [str(int(b) + 1000) for b in upstream_bodyIds],
                    "weight": [5] * len(upstream_bodyIds),
                    "roi": ["fake"] * len(upstream_bodyIds),
                })
            finally:
                with active_lock:
                    active -= 1

        monkeypatch.setattr(fc, "_fetch_connections_bulk", fake_fetch)

        summary = fc.build_connection_cache(
            neuron_bodyIds=["1", "2", "3", "4", "5", "6"],
            batch_size=2,
            quiet=True,
            max_workers=3,
        )
        assert summary["newly_cached"] == 6
        assert summary["failed_neurons"] == []
        assert max_active >= 2, f"fetches did not overlap: max_active={max_active}"
        conn_path = tmp_path / "cache" / "fake_v3" / "connections.parquet"
        assert conn_path.exists()

    def test_parallel_cancel_stops_submitting(self, tmp_path, monkeypatch):
        """Cancelling a parallel build stops new submissions; the summary is
        marked cancelled and fetched batches are consolidated."""
        import coana
        import neuprint
        import threading

        class FakeClient:
            def __init__(self, *a, **k):
                pass

        monkeypatch.setattr(neuprint, "Client", FakeClient)

        fc = coana.FindNeuronConnection(
            dataset="fake:v4",
            use_cache=True,
            cache_only=False,
            verbose=False,
            script_path=str(tmp_path),
            cache_folder=str(tmp_path / "cache" / "fake_v4"),
        )

        import time
        import pandas as pd

        def fake_fetch(upstream_bodyIds, downstream_bodyIds=None):
            time.sleep(0.1)
            return pd.DataFrame({
                "bodyId_pre": upstream_bodyIds,
                "bodyId_post": [str(int(b) + 1000) for b in upstream_bodyIds],
                "weight": [5] * len(upstream_bodyIds),
                "roi": ["fake"] * len(upstream_bodyIds),
            })

        monkeypatch.setattr(fc, "_fetch_connections_bulk", fake_fetch)

        cancel_event = threading.Event()
        # Cancel immediately: nothing (or at most the in-flight first batch)
        # gets processed, and the summary must say cancelled.
        cancel_event.set()
        summary = fc.build_connection_cache(
            neuron_bodyIds=["1", "2", "3", "4", "5", "6"],
            batch_size=2,
            quiet=True,
            cancel_event=cancel_event,
            max_workers=3,
        )
        assert summary["cancelled"] is True
        assert summary["newly_cached"] == 0

    def test_cancel_stops_and_consolidates(self, tmp_path, monkeypatch):
        """A real build_connection_cache run with cancel_event set stops after
        the current batch, marks the summary as cancelled, and still produces
        the consolidated connections.parquet for the fetched batches."""
        import coana
        import neuprint

        class FakeClient:
            def __init__(self, *a, **k):
                pass

        # __post_init__ uses a local `from neuprint import Client`, so patch
        # the source module (coana.Client is shadowed by that local import).
        monkeypatch.setattr(neuprint, "Client", FakeClient)

        fc = coana.FindNeuronConnection(
            dataset="fake:v1",
            use_cache=True,
            cache_only=False,
            verbose=False,
            script_path=str(tmp_path),
            cache_folder=str(tmp_path / "cache" / "fake_v1"),
        )

        fetched = []

        def fake_fetch(upstream_bodyIds, downstream_bodyIds=None):
            fetched.extend(upstream_bodyIds)
            # The real _fetch_connections_bulk returns a pandas frame
            import pandas as pd

            return pd.DataFrame({
                "bodyId_pre": upstream_bodyIds,
                "bodyId_post": [str(int(b) + 1000) for b in upstream_bodyIds],
                "weight": [5] * len(upstream_bodyIds),
                "roi": ["fake"] * len(upstream_bodyIds),
            })

        monkeypatch.setattr(fc, "_fetch_connections_bulk", fake_fetch)
        # Mark the first batch fetched, then cancel: the second batch must
        # never be fetched.
        cancel_event = threading.Event()
        first_batch_done = threading.Event()

        real_fetch = fake_fetch

        def cancel_after_first_batch(upstream_bodyIds, downstream_bodyIds=None):
            result = real_fetch(upstream_bodyIds, downstream_bodyIds)
            first_batch_done.set()
            cancel_event.set()  # cancel as soon as the first batch returns
            return result

        monkeypatch.setattr(fc, "_fetch_connections_bulk", cancel_after_first_batch)

        summary = fc.build_connection_cache(
            neuron_bodyIds=["1", "2", "3", "4"],
            batch_size=2,
            quiet=True,
            cancel_event=cancel_event,
        )
        assert summary["cancelled"] is True
        # Only the first batch (2 neurons) was fetched
        assert sorted(fetched) == ["1", "2"]
        assert summary["newly_cached"] == 2
        # The consolidated parquet exists (resume checkpoint)
        conn_path = tmp_path / "cache" / "fake_v1" / "connections.parquet"
        assert conn_path.exists()

    def test_quiet_mode_still_consolidates(self, tmp_path, monkeypatch):
        """quiet=True must still produce the final connections.parquet (the UI
        calls with quiet=True and relies on the consolidated output)."""
        import coana
        import neuprint

        class FakeClient:
            def __init__(self, *a, **k):
                pass

        # __post_init__ uses a local `from neuprint import Client`, so patch
        # the source module (coana.Client is shadowed by that local import).
        monkeypatch.setattr(neuprint, "Client", FakeClient)

        fc = coana.FindNeuronConnection(
            dataset="fake:v2",
            use_cache=True,
            cache_only=False,
            verbose=False,
            script_path=str(tmp_path),
            cache_folder=str(tmp_path / "cache" / "fake_v2"),
        )

        def fake_fetch(upstream_bodyIds, downstream_bodyIds=None):
            # The real _fetch_connections_bulk returns a pandas frame
            import pandas as pd

            return pd.DataFrame({
                "bodyId_pre": upstream_bodyIds,
                "bodyId_post": [str(int(b) + 1000) for b in upstream_bodyIds],
                "weight": [5] * len(upstream_bodyIds),
                "roi": ["fake"] * len(upstream_bodyIds),
            })

        monkeypatch.setattr(fc, "_fetch_connections_bulk", fake_fetch)

        summary = fc.build_connection_cache(
            neuron_bodyIds=["1", "2"],
            batch_size=1,
            quiet=True,
        )
        assert summary["cancelled"] is False
        assert summary["newly_cached"] == 2
        conn_path = tmp_path / "cache" / "fake_v2" / "connections.parquet"
        assert conn_path.exists()
