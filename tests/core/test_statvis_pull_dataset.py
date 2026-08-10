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

    def test_downloads_in_chunks_with_progress_and_writes_csvs(self, tmp_path, monkeypatch):
        """4500 neurons are fetched in chunks of 2000 (3 calls), a progress
        bar tracks the total, and both CSVs are written."""
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
        assert bar.updates == [2000, 2000, 500]
        assert (tmp_path / 'ds_neuron_df.csv').exists()
        assert (tmp_path / 'ds_roi_count_df.csv').exists()

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
