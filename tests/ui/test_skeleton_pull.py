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


class TestSkeletonPuller:
    def test_lifecycle_and_progress(self, fake_download):
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
