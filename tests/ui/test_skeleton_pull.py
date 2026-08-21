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
    import coana

    holder = {"impl": _FakeDownload()}
    monkeypatch.setattr(morphology, "download_all_skeletons", holder["impl"])

    # The puller ensures the dataset metadata before downloading skeletons;
    # keep the real FNC (and its server access) out of the unit tests.
    class FakeFNC:
        def __init__(self, *args, **kwargs):
            pass

        def _ensure_complete_dataset(self):
            pass

        def _ensure_neuron_index_from_metadata(self):
            pass

    monkeypatch.setattr(coana, "FindNeuronConnection", FakeFNC)
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


def test_batch_fetch_cache_membership_uses_one_scan_not_per_id_lookups(
        tmp_path, monkeypatch):
    """The cache-first loop must decide membership from one directory scan:
    a full-dataset pull with an almost-empty cache used to call load_skeleton
    (and its per-id rglob) for EVERY missing id - ~30 minutes of cache scans
    before the first batch completed."""
    import threading
    import neuprint
    import morphology

    class FakeClient:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(neuprint, "Client", FakeClient)

    # Two cached files, one flat and one nested (bulk folder).
    cache_dir = tmp_path / "skeletons" / "raw_skeletons"
    cache_dir.mkdir(parents=True)
    (cache_dir / "1.swc.zst").write_bytes(b"x")
    nested = cache_dir / "bulk"
    nested.mkdir()
    (nested / "2.swc.gz").write_bytes(b"x")

    loaded = []

    class FakeRawCache:
        raw_only = True

        def _discover_skeleton_files(self):
            return [str(cache_dir / "1.swc.zst"), str(nested / "2.swc.gz")]

        def load_skeleton(self, bid, simplification=None):
            loaded.append(bid)
            return object()  # a loaded neuron

    fetch_batches = []

    def fake_batch_fetch(batch_ids, **kwargs):
        fetch_batches.append(list(batch_ids))
        return []

    # The NeuPrint batch seam is the vendored per-neuron progress fetcher.
    monkeypatch.setattr(
        morphology, "_fetch_neuprint_batch_with_progress", fake_batch_fetch
    )

    result = morphology.fetch_skeletons_on_demand_batch(
        "hemibrain:v1.2.1", [1, 2, 3], project_root=str(tmp_path),
        persist=False, raw_cache=FakeRawCache(),
    )
    # Cached ids are returned directly from the cache; the missing id went
    # to the (empty) fetch list.
    assert set(result) == {1, 2}, result
    # Only the ids that are actually cached go through load_skeleton.
    assert loaded == [1, 2], loaded
    # The missing id went straight to the fetch list.
    assert fetch_batches == [[3]], fetch_batches


def test_find_skeleton_file_skips_rglob_on_flat_cache(tmp_path, monkeypatch):
    """find_skeleton_file must not rglob the cache tree per cache miss when
    the directory is flat (bulk pulls used to rescan the whole tree for every
    missing id)."""
    import pathlib
    import morphology

    cache = morphology.find_similar_raw_cache(
        "hemibrain:v1.2.1", project_root=str(tmp_path), verbose=False
    )
    skeleton_dir = cache.skeleton_dir
    skeleton_dir.mkdir(parents=True)
    (skeleton_dir / "5.swc.zst").write_bytes(b"x")
    (skeleton_dir / "6.swc.gz").write_bytes(b"x")

    original_rglob = pathlib.Path.rglob
    calls = {"n": 0}

    def counting_rglob(self, pattern):
        calls["n"] += 1
        return original_rglob(self, pattern)

    monkeypatch.setattr(pathlib.Path, "rglob", counting_rglob)

    # Flat cache: a missing id must not rescan the tree per id.
    assert cache.find_skeleton_file(999) is None
    assert calls["n"] == 0, calls
    # Cached flat files still resolve through the direct path.
    assert cache.find_skeleton_file(5).name == "5.swc.zst"
    assert calls["n"] == 0, calls

    # Nested bulk folders still resolve via rglob once a subdir appears.
    nested = skeleton_dir / "bulk"
    nested.mkdir()
    (nested / "7.swc.zst").write_bytes(b"x")
    assert cache.find_skeleton_file(7).name == "7.swc.zst"
    assert calls["n"] >= 1, calls


class TestSkeletonPuller:
    def test_metadata_ensured_before_download(self, monkeypatch):
        """The skeleton pull must ensure the dataset metadata (neuron table,
        ROI table, index) before downloading any skeleton."""
        import coana
        import morphology

        order = []

        class FakeFNC:
            def __init__(self, *args, **kwargs):
                order.append("init")

            def _ensure_complete_dataset(self):
                order.append("metadata")

            def _ensure_neuron_index_from_metadata(self):
                order.append("index")

        def fake_download_all(dataset, **kwargs):
            order.append("download")
            return {"total": 0, "fetched": 0, "skipped_existing": 0,
                    "cancelled": False, "errors": 0}

        monkeypatch.setattr(coana, "FindNeuronConnection", FakeFNC)
        monkeypatch.setattr(morphology, "download_all_skeletons", fake_download_all)
        puller = SkeletonPuller()
        assert puller.start("np:v1") is True
        assert _wait_until(lambda: puller.state["done"])
        assert puller.state["error"] is None
        assert order == ["init", "metadata", "index", "download"]

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

    def test_simplification_forwarded_to_download(self, fake_download):
        """The pull forwards the Settings 'Cache Simplification' selector and
        the skeleton batch size to download_all_skeletons; omitting them
        keeps the backend defaults."""
        puller = SkeletonPuller()
        assert puller.start("np:v1", simplification=50, batch_size=10)
        assert _wait_until(lambda: puller.state["done"])
        assert fake_download["impl"].called_with["simplification"] == 50
        assert fake_download["impl"].called_with["batch_size"] == 10
        assert puller.state["simplification"] == 50
        assert puller.state["batch_size"] == 10

        puller = SkeletonPuller()
        assert puller.start("np:v1")
        assert _wait_until(lambda: puller.state["done"])
        assert "simplification" not in fake_download["impl"].called_with
        assert "batch_size" not in fake_download["impl"].called_with
        assert puller.state["simplification"] is None
        assert puller.state["batch_size"] is None

    def test_cancel(self, fake_download):
        fake_download["impl"].delay = 0.05
        puller = SkeletonPuller()
        puller.start("np:v1")
        assert _wait_until(lambda: puller.state["total"] > 0)
        puller.cancel()
        assert _wait_until(lambda: puller.state["done"])
        assert puller.state["cancelled"] is True

    def test_cancel_marks_cancel_requested_for_ui_hint(self, fake_download):
        """cancel() exposes the request in the state so the Settings tab can
        show a persistent 'Cancelling...' hint during the wind-down (in-flight
        navis batch + persist phase)."""
        fake_download["impl"].delay = 0.05
        puller = SkeletonPuller()
        puller.start("np:v1")
        assert _wait_until(lambda: puller.state["total"] > 0)
        assert puller.state["cancel_requested"] is False
        puller.cancel()
        assert puller.state["cancel_requested"] is True
        assert _wait_until(lambda: puller.state["done"])
        # a fresh pull resets the flag
        assert puller.start("np:v1") is True
        assert puller.state["cancel_requested"] is False
        assert _wait_until(lambda: puller.state["done"])

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
