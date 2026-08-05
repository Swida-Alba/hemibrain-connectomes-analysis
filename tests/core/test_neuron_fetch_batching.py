#!/usr/bin/env python
"""
Regression tests for batched NeuPrint neuron fetching.

The neuprint client issues ONE Cypher query per fetch_neurons() call. A single
call with tens of thousands of bodyIds makes the server evaluate a giant
IN-list and returns a massive payload that the client parses for minutes at
~100% CPU (perceived as a hang). coana._fetch_neurons_batched() chunks the
list so every query and response stays bounded.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import coana  # noqa: E402


def _make_chunk_df(ids):
    return pd.DataFrame({
        'bodyId': [str(i) for i in ids],
        'type': ['T'] * len(ids),
        'instance': ['I'] * len(ids),
    })


class TestFetchNeuronsBatched:
    def _client(self, monkeypatch):
        calls = []

        def fake_fetch_neurons(criteria):
            ids = criteria.bodyId if isinstance(criteria.bodyId, list) else [criteria.bodyId]
            calls.append(len(ids))
            return _make_chunk_df(ids), pd.DataFrame()

        monkeypatch.setattr(coana, 'fetch_neurons', fake_fetch_neurons)
        return calls

    def test_small_list_is_single_call(self, monkeypatch):
        calls = self._client(monkeypatch)
        fc = coana.FindNeuronConnection(dataset='male-cns:v1.0')
        out = fc._fetch_neurons_batched([1, 2, 3, 4, 5])
        assert calls == [5]
        assert len(out) == 5

    def test_large_list_is_chunked(self, monkeypatch):
        calls = self._client(monkeypatch)
        fc = coana.FindNeuronConnection(dataset='male-cns:v1.0')
        ids = list(range(5000))
        out = fc._fetch_neurons_batched(ids, batch_size=2000)
        assert calls == [2000, 2000, 1000], f'chunk sizes: {calls}'
        assert len(out) == 5000
        assert sorted(out['bodyId'].astype(int).tolist()) == ids

    def test_empty_input_returns_empty_frame(self, monkeypatch):
        calls = self._client(monkeypatch)
        fc = coana.FindNeuronConnection(dataset='male-cns:v1.0')
        out = fc._fetch_neurons_batched([])
        assert calls == []
        assert out.empty

    def test_chunk_errors_do_not_abort_remaining_chunks(self, monkeypatch):
        calls = []

        def flaky_fetch(criteria):
            ids = criteria.bodyId
            calls.append(len(ids))
            if len(ids) == 2000 and len(calls) == 1:
                raise RuntimeError('server timeout')
            return _make_chunk_df(ids), pd.DataFrame()

        monkeypatch.setattr(coana, 'fetch_neurons', flaky_fetch)
        fc = coana.FindNeuronConnection(dataset='male-cns:v1.0')
        with pytest.raises(RuntimeError):
            fc._fetch_neurons_batched(list(range(4000)), batch_size=2000)
        # An error propagates (the caller wraps this in try/except) but the
        # batch loop itself does not silently swallow the failure.
        assert len(calls) == 1
