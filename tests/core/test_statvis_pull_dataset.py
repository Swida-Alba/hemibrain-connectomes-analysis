#!/usr/bin/env python
"""
Regression tests for the chunked statvis.pull_dataset download.

The old implementation fetched the ENTIRE dataset with a single
``fetch_neurons(None)`` call: no timeout, no retry, no progress feedback —
a large dataset (male-cns ~180k neurons) stalled for minutes and looked
like a hang. The new implementation lists bodyIds with one light query and
then fetches neuron info in chunks under api_call_with_retry (timeout + 5
reconnect attempts) with a live progress bar.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import statvis  # noqa: E402


def _make_chunk_df(ids):
    return pd.DataFrame({
        'bodyId': [int(i) for i in ids],
        'type': ['T'] * len(ids),
        'instance': ['I'] * len(ids),
    })


class _FakeClient:
    """Client stub: fetch_custom returns the bodyId list, fetch_neurons is
    injected into pull_dataset via fetch_fn. all_rois is read by the
    NeuronCriteria factory during construction."""

    def __init__(self, n_ids=4500):
        self.n_ids = n_ids
        self.all_rois = []

    def fetch_custom(self, query):
        assert 'bodyId' in query
        return pd.DataFrame({'bodyId': list(range(self.n_ids))})


class TestPullDatasetChunked:
    def _patch_retry(self, monkeypatch):
        """Replace api_call_with_retry with a deterministic INLINE version.

        The real implementation runs each attempt in a worker thread
        (ThreadPoolExecutor + shutdown(wait=False)); leaked workers interact
        badly with stubbed neuprint functions in the test environment. The
        inline version keeps the retry/on_retry semantics (timeout skipped)
        and is fully deterministic."""
        import src.utils.api_utils as au

        def inline_retry(func, timeout=60, max_retries=5, retry_delay=2.0,
                         description="API call", on_retry=None, verbose=True):
            from src.utils.api_utils import APIRetryExhaustedError
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func()
                except Exception as e:
                    last_exc = e
                    if on_retry is not None:
                        on_retry(attempt, e)
            # mirror the real implementation: the final failure is wrapped in
            # APIRetryExhaustedError so callers catch it identically
            raise APIRetryExhaustedError(
                f"{description} failed after {max_retries} attempts. "
                f"Last error: {type(last_exc).__name__}: {last_exc}"
            ) from last_exc

        monkeypatch.setattr(au, 'api_call_with_retry', inline_retry)
        return inline_retry

    def _fake_fetch(self, calls, flaky_chunk_sizes=()):
        """fetch_fn stub injected directly into pull_dataset."""
        def fake_fetch(criteria, client=None):
            ids = criteria.bodyId if isinstance(criteria.bodyId, list) else [criteria.bodyId]
            calls.append(len(ids))
            if len(ids) in flaky_chunk_sizes:
                raise RuntimeError('server timeout')
            return _make_chunk_df(ids), _make_chunk_df(ids)

        return fake_fetch

    def test_downloads_in_chunks_with_progress_and_writes_tables(self, tmp_path, monkeypatch):
        """4500 neurons are fetched in chunks of 2000 (3 calls), a progress
        bar tracks the total, and the neuron CSV + ROI parquet are written."""
        import time as _time
        monkeypatch.setattr(_time, 'sleep', lambda s: None)

        class _FakeTqdm:
            instances = []

            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs
                self.updates = []
                _FakeTqdm.instances.append(self)

            def update(self, n):
                self.updates.append(n)

            def close(self):
                pass

        # statvis binds `from tqdm import tqdm` at module level, so patch
        # statvis.tqdm (not tqdm.tqdm)
        monkeypatch.setattr(statvis, 'tqdm', _FakeTqdm)
        self._patch_retry(monkeypatch)
        calls = []
        out = tmp_path / 'ds'
        # A stale CSV from an older pull must be replaced by the parquet.
        (tmp_path / 'ds_roi_count_df.csv').write_text('bodyId,roi,pre,post\n')
        statvis.pull_dataset(
            'fake:v1', save_path=str(out),
            client=_FakeClient(n_ids=4500),
            fetch_fn=self._fake_fetch(calls),
        )
        assert calls == [2000, 2000, 500]  # chunked, never one giant call
        assert len(_FakeTqdm.instances) == 1
        bar = _FakeTqdm.instances[0]
        assert bar.kwargs['total'] == 4500
        assert bar.kwargs['desc'] == 'Downloading neuron list'
        # NeuronBridge and dataset-pull output share stdout so the UI runner
        # cannot reorder a message between a bar's clear and redraw.
        assert bar.kwargs['file'] is sys.stdout
        assert bar.updates == [2000, 2000, 500]
        assert (tmp_path / 'ds_neuron_df.csv').exists()
        assert (tmp_path / 'ds_roi_count_df.parquet').exists()
        assert not (tmp_path / 'ds_roi_count_df.csv').exists()
        roi = pd.read_parquet(tmp_path / 'ds_roi_count_df.parquet')
        assert list(roi.columns) == ['bodyId', 'type', 'instance']
        # Per-neuron ROI columns are dropped before saving: the data lives
        # long-form in ds_roi_count_df.parquet.
        saved = pd.read_csv(tmp_path / 'ds_neuron_df.csv', index_col=0)
        assert not {'roiInfo', 'inputRois', 'outputRois'} & set(saved.columns)
        # Metadata sidecar is written next to the tables.
        meta = json.load(open(tmp_path / 'ds_metadata.json'))
        assert meta['neuron_counts']['total'] == 4500
        assert meta['dataset'] == 'fake:v1'

    def test_progress_callback_reports_cumulative_counts(self, tmp_path, monkeypatch):
        """progress_callback receives (current, total) after each chunk, so a
        UI caller (the Settings-tab metadata pull) can drive a determinate bar
        as the long download progresses."""
        import time as _time
        monkeypatch.setattr(_time, 'sleep', lambda s: None)
        monkeypatch.setattr(statvis, 'tqdm', lambda *a, **k: type('T', (), {
            'update': lambda self, n: None, 'close': lambda self: None})())
        self._patch_retry(monkeypatch)
        calls = []
        progress = []

        def fake_fetch(criteria, client=None):
            ids = criteria.bodyId if isinstance(criteria.bodyId, list) else [criteria.bodyId]
            calls.append(len(ids))
            return _make_chunk_df(ids), _make_chunk_df(ids)

        out = tmp_path / 'ds_pcb'
        statvis.pull_dataset(
            'fake:v1', save_path=str(out),
            client=_FakeClient(n_ids=4500),
            fetch_fn=fake_fetch,
            progress_callback=lambda c, t: progress.append((c, t)),
        )
        assert calls == [2000, 2000, 500]
        # cumulative current / total after each of the three chunks
        assert progress == [(2000, 4500), (4000, 4500), (4500, 4500)]

    def test_cancel_mid_download_raises_without_writing(self, tmp_path, monkeypatch):
        """cancel_event set between chunks stops the download, raises
        DatasetPullCancelled, and leaves no partial tables behind (a cancelled
        dataset must not look 'ready' for cache enrichment)."""
        import time as _time
        import threading
        monkeypatch.setattr(_time, 'sleep', lambda s: None)
        monkeypatch.setattr(statvis, 'tqdm', lambda *a, **k: type('T', (), {
            'update': lambda self, n: None, 'close': lambda self: None})())
        self._patch_retry(monkeypatch)

        calls = []
        cancel = threading.Event()

        def fake_fetch(criteria, client=None):
            ids = criteria.bodyId if isinstance(criteria.bodyId, list) else [criteria.bodyId]
            calls.append(len(ids))
            # cancel after the first chunk lands, before the next iteration
            cancel.set()
            return _make_chunk_df(ids), _make_chunk_df(ids)

        out = tmp_path / 'ds_cancel'
        with pytest.raises(statvis.DatasetPullCancelled):
            statvis.pull_dataset(
                'fake:v1', save_path=str(out),
                client=_FakeClient(n_ids=4500),
                fetch_fn=fake_fetch,
                cancel_event=cancel,
            )
        assert calls == [2000]  # stopped after the first chunk
        assert not (tmp_path / 'ds_cancel_neuron_df.csv').exists()
        assert not (tmp_path / 'ds_cancel_roi_count_df.parquet').exists()

    def test_cancel_already_set_raises_immediately(self, tmp_path, monkeypatch):
        """A cancel_event already set fails fast before any network query."""
        import threading
        monkeypatch.setattr(statvis, 'tqdm', lambda *a, **k: type('T', (), {
            'update': lambda self, n: None, 'close': lambda self: None})())
        self._patch_retry(monkeypatch)

        cancel = threading.Event()
        cancel.set()  # pressed before the pull started
        with pytest.raises(statvis.DatasetPullCancelled):
            statvis.pull_dataset(
                'fake:v1', save_path=str(tmp_path / 'ds_c'),
                client=_FakeClient(n_ids=100),
                cancel_event=cancel,
            )

    def test_parallel_fetch_overlaps_chunks_and_writes_same_output(self, tmp_path, monkeypatch):
        """max_workers>1 overlaps the chunk fetches (bounded in-flight) yet still
        writes the same neuron CSV + ROI parquet (a long dataset pull is a chain
        of server requests, so overlapping them cuts wall time)."""
        import time as _time
        import threading
        monkeypatch.setattr(_time, 'sleep', lambda s: None)
        monkeypatch.setattr(statvis, 'tqdm', lambda *a, **k: type('T', (), {
            'update': lambda self, n: None, 'close': lambda self: None})())
        self._patch_retry(monkeypatch)

        active = 0
        max_active = 0
        lock = threading.Lock()

        def fake_fetch(criteria, client=None):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            # A real blocking wait (not time.sleep, which the test stubs)
            # forces the chunk fetches to overlap in the worker pool.
            threading.Event().wait(0.02)
            ids = criteria.bodyId if isinstance(criteria.bodyId, list) else [criteria.bodyId]
            result = _make_chunk_df(ids), _make_chunk_df(ids)
            with lock:
                active -= 1
            return result

        out = tmp_path / 'ds_par'
        statvis.pull_dataset(
            'fake:v1', save_path=str(out),
            client=_FakeClient(n_ids=4500),
            fetch_fn=fake_fetch,
            max_workers=3,
        )
        # at least two chunks were in flight at once
        assert max_active >= 2
        assert (tmp_path / 'ds_par_neuron_df.csv').exists()
        assert (tmp_path / 'ds_par_roi_count_df.parquet').exists()
        assert len(pd.read_csv(tmp_path / 'ds_par_neuron_df.csv', index_col=0)) == 4500
        roi = pd.read_parquet(tmp_path / 'ds_par_roi_count_df.parquet')
        assert len(roi) == 4500
        assert list(roi.columns) == ['bodyId', 'type', 'instance']

    def test_metadata_sidecar_computed_from_pulled_frames(self, tmp_path, monkeypatch):
        """pull_dataset writes <dataset>_metadata.json computed from the
        pulled neuron/roi frames (counts, synapse totals, per-ROI neurons)."""
        import time as _time
        monkeypatch.setattr(_time, 'sleep', lambda s: None)
        monkeypatch.setattr(statvis, 'tqdm', lambda *a, **k: type('T', (), {
            'update': lambda self, n: None, 'close': lambda self: None})())
        self._patch_retry(monkeypatch)

        def fake_fetch(criteria, client=None):
            ids = criteria.bodyId if isinstance(criteria.bodyId, list) else [criteria.bodyId]
            ndf = pd.DataFrame({
                'bodyId': ids,
                'type': ['T' if i % 2 == 0 else None for i in ids],
                'pre': [10] * len(ids),
                'post': [20] * len(ids),
            })
            rows = [
                {'bodyId': b, 'roi': r, 'pre': 1, 'post': 2,
                 'downstream': 0, 'upstream': 2}
                for b in ids for r in ('AL(L)', 'AL(R)')
            ]
            return ndf, pd.DataFrame(rows)

        out = tmp_path / 'ds_meta'
        statvis.pull_dataset(
            'fake:v1', save_path=str(out),
            client=_FakeClient(n_ids=600),
            fetch_fn=fake_fetch,
        )
        meta = json.load(open(tmp_path / 'ds_meta_metadata.json'))
        assert meta['source'] == 'neuprint'
        assert meta['neuron_counts']['total'] == 600
        assert meta['neuron_counts']['typed'] == 300
        assert meta['neuron_counts']['untyped'] == 300
        assert meta['synapse_counts'] == {
            'total_presynaptic': 6000, 'total_postsynaptic': 12000,
            'total': 18000}
        # No client primary_rois on the fake -> roi list falls back to the
        # long-form roi-count table.
        assert meta['roi_coverage']['roi_list'] == ['AL(L)', 'AL(R)']
        assert meta['roi_coverage']['neuron_counts_per_roi'] == {
            'AL(L)': 600, 'AL(R)': 600}

    def test_drop_roi_cols_false_keeps_roi_columns(self, tmp_path, monkeypatch):
        """drop_roi_cols=False keeps roiInfo/inputRois/outputRois in the
        saved neuron CSV; the default (True) drops them."""
        import time as _time
        monkeypatch.setattr(_time, 'sleep', lambda s: None)
        monkeypatch.setattr(statvis, 'tqdm', lambda *a, **k: type('T', (), {
            'update': lambda self, n: None, 'close': lambda self: None})())
        self._patch_retry(monkeypatch)

        def fake_fetch(criteria, client=None):
            ids = criteria.bodyId if isinstance(criteria.bodyId, list) else [criteria.bodyId]
            ndf = _make_chunk_df(ids)
            ndf['roiInfo'] = '{}'
            ndf['inputRois'] = '[]'
            ndf['outputRois'] = '[]'
            return ndf, _make_chunk_df(ids)

        out = tmp_path / 'ds_keep'
        statvis.pull_dataset(
            'fake:v1', save_path=str(out),
            client=_FakeClient(n_ids=500),
            fetch_fn=fake_fetch,
            drop_roi_cols=False,
        )
        saved = pd.read_csv(tmp_path / 'ds_keep_neuron_df.csv', index_col=0)
        assert {'roiInfo', 'inputRois', 'outputRois'} <= set(saved.columns)

    def test_failed_chunks_retried_then_skipped(self, tmp_path, monkeypatch):
        """A batch that keeps failing after 5 retries is reported and SKIPPED;
        the download still completes with the remaining batches."""
        import time as _time
        monkeypatch.setattr(_time, 'sleep', lambda s: None)
        self._patch_retry(monkeypatch)
        calls = []

        def fake_fetch(criteria, client=None):
            ids = criteria.bodyId if isinstance(criteria.bodyId, list) else [criteria.bodyId]
            calls.append(len(ids))
            # the FIRST 2000-neuron chunk always fails (all 5 attempts)
            if len(ids) == 2000 and calls.count(2000) <= 5:
                raise RuntimeError('server timeout')
            return _make_chunk_df(ids), _make_chunk_df(ids)

        out = tmp_path / 'ds2'
        statvis.pull_dataset(
            'fake:v1', save_path=str(out),
            client=_FakeClient(n_ids=4500),
            fetch_fn=fake_fetch,
        )
        # 5 attempts on the first chunk + second chunk (2000) + third (500)
        assert calls == [2000, 2000, 2000, 2000, 2000, 2000, 500]
        df = pd.read_csv(tmp_path / 'ds2_neuron_df.csv', index_col=0)
        assert len(df) == 2500  # only the surviving chunks

    def test_all_batches_failing_raises(self, tmp_path, monkeypatch):
        """When every chunk fails after retries, pull_dataset raises instead
        of writing empty files."""
        import time as _time
        monkeypatch.setattr(_time, 'sleep', lambda s: None)
        self._patch_retry(monkeypatch)

        def failing_fetch(criteria, client=None):
            raise RuntimeError('server down')

        with pytest.raises(RuntimeError, match='every batch failed'):
            statvis.pull_dataset(
                'fake:v1', save_path=str(tmp_path / 'ds3'),
                client=_FakeClient(n_ids=2000),
                fetch_fn=failing_fetch,
            )
        assert not (tmp_path / 'ds3_neuron_df.csv').exists()
