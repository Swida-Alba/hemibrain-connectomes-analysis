#!/usr/bin/env python
"""Tests for the full-skeleton download operation:

- ``ui.skeleton_pull.SkeletonPuller``: background thread lifecycle, progress
  reporting, cancellation, error handling, and one-at-a-time guard
  (mirrors the DatasetPuller tests for the Settings dataset pull).
"""

import sys
import threading
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ui.skeleton_pull import SkeletonPuller  # noqa: E402


class _FakeDownload:
    """Simulates morphology.download_all_skeletons: emits progress, honours
    cancel_event."""

    def __init__(self, total: int = 50, fail: bool = False, delay: float = 0.0):
        self.total = total
        self.fail = fail
        self.delay = delay
        self.called_with = None
        self.cancel_event = None

    def __call__(self, dataset, **kwargs):
        self.called_with = kwargs
        self.cancel_event = kwargs.get("cancel_event")
        progress_callback = kwargs.get("progress_callback")
        for i in range(0, self.total, 10):
            if self.cancel_event is not None and self.cancel_event.is_set():
                return {"total": self.total, "fetched": i,
                        "skipped_existing": 0, "cancelled": True, "errors": 0}
            if progress_callback:
                progress_callback(i, self.total, f"Batch {i // 10 + 1}")
            if self.delay:
                time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("connection reset")
        return {"total": self.total, "fetched": self.total,
                "skipped_existing": 0, "cancelled": False, "errors": 0}


@pytest.fixture
def fake_download(monkeypatch):
    import morphology
    holder = {"impl": _FakeDownload()}
    monkeypatch.setattr(morphology, "download_all_skeletons", holder["impl"])
    return holder


def _wait_until(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_download_all_skeletons_passes_cancel_event_to_batch(tmp_path, monkeypatch):
    """The skeleton pull must hand its cancel event to the batched fetch so
    Cancel stops at the next batch boundary; it used to be noticed only after
    the whole batch phase completed."""
    import threading
    import morphology

    folder = "hemibrain_v1_2_1"
    ds_dir = tmp_path / "datasets" / folder
    ds_dir.mkdir(parents=True)
    (ds_dir / f"{folder}_allneurons_neuron_df.csv").write_text(
        "bodyId\n1\n2\n3\n", encoding="utf-8"
    )

    captured = {}

    def fake_batch(dataset, body_ids, **kwargs):
        captured["body_ids"] = list(body_ids)
        captured["cancel_event"] = kwargs.get("cancel_event")
        return {}

    monkeypatch.setattr(morphology, "fetch_skeletons_on_demand_batch", fake_batch)

    cancel = threading.Event()
    summary = morphology.download_all_skeletons(
        "hemibrain:v1.2.1",
        project_root=str(tmp_path),
        max_workers=2,
        cancel_event=cancel,
        verbose=False,
    )
    assert captured.get("cancel_event") is cancel
    assert captured["body_ids"] == [1, 2, 3]
    assert summary["total"] == 3
    assert summary["skipped_existing"] == 0


def test_batch_fetch_honours_pre_set_cancel(tmp_path, monkeypatch):
    """A cancel_event set before the fetch starts skips the network phase
    entirely (no new batch is submitted)."""
    import threading
    import neuprint
    import morphology

    class FakeClient:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(neuprint, "Client", FakeClient)
    calls = {"n": 0}

    def fake_fetch_skeletons(*a, **k):
        calls["n"] += 1
        return []

    monkeypatch.setattr(
        "navis.interfaces.neuprint.fetch_skeletons", fake_fetch_skeletons
    )
    cancel = threading.Event()
    cancel.set()

    result = morphology.fetch_skeletons_on_demand_batch(
        "np-fake:v1", [1, 2, 3], project_root=str(tmp_path),
        persist=False, cancel_event=cancel,
    )
    assert result == {}
    assert calls["n"] == 0


class TestSkeletonPuller:
    def test_lifecycle_and_progress(self, fake_download):
        # Keep the worker alive long enough to assert the one-at-a-time guard;
        # the zero-delay fake can otherwise finish before start() returns.
        fake_download["impl"].delay = 0.01
        puller = SkeletonPuller()
        assert puller.start("np:v1")
        assert puller.running
        assert not puller.start("np:v1")  # one download at a time
        assert _wait_until(lambda: puller.state["done"])
        st = puller.state
        assert st["dataset"] == "np:v1"
        assert st["current"] == 40  # last progress batch emitted
        assert st["total"] == 50
        assert st["summary"]["fetched"] == 50
        assert st["error"] is None
        assert not st["running"]
        # progress callback was wired with the cancel event
        assert fake_download["impl"].called_with["cancel_event"] is not None
        assert fake_download["impl"].called_with["max_workers"] == 8
        assert fake_download["impl"].called_with["mode"] == "raw"

    def test_cancel(self, fake_download):
        fake_download["impl"].delay = 0.05
        puller = SkeletonPuller()
        puller.start("np:v1")
        assert _wait_until(lambda: puller.state["total"] > 0)
        puller.cancel()
        assert _wait_until(lambda: puller.state["done"])
        assert puller.state["cancelled"] is True

    def test_error_reported(self, fake_download):
        fake_download["impl"].fail = True
        puller = SkeletonPuller()
        puller.start("np:v1")
        assert _wait_until(lambda: puller.state["done"])
        st = puller.state
        assert "RuntimeError" in st["error"]
        assert st["info"] == "Failed."

    def test_start_after_finish_allowed(self, fake_download):
        puller = SkeletonPuller()
        assert puller.start("np:v1")
        assert _wait_until(lambda: puller.state["done"])
        assert puller.start("np:v1")  # finished -> can start again
