"""Regression tests for online-only FAFB runs with caching disabled."""

import sys
from pathlib import Path

import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _stub_fafb_finder():
    from coana import FindNeuronConnection

    finder = object.__new__(FindNeuronConnection)
    finder.use_cache = False
    finder.client_type = "flywire"
    finder.dataset = "flywire_FAFB_v783"
    finder.cache_folder = ""
    finder.script_path = str(PROJECT_ROOT)
    finder.verbose_mode = "silent"
    finder.force_API_fetching = False
    finder._conn_df_cache = None
    finder._get_neuron_index_path = lambda: ""
    finder._get_connection_db_path = lambda: "connections.parquet"
    finder._vprint = lambda *args, **kwargs: None
    finder.build_connection_cache = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("no-cache FAFB metric lookup attempted to build a cache")
    )
    return finder


class _FakeCaveFetcher:
    def fetch_connections(self, body_ids, direction="both", show_progress=True):
        assert direction == "post"
        return pd.DataFrame({
            "pre_pt_root_id": ["A", "B", "C", "D"],
            "post_pt_root_id": ["101", "101", "102", "103"],
            "weight": [4, 2, 5, 1],
        })

    def fetch_neurons_by_types(self, types, show_progress=True):
        assert types == ["T"]
        return pd.DataFrame({
            "bodyId": ["101", "102"],
            "type": ["T", "T"],
            "instance": ["T_L", "T_R"],
            "post": [0, 0],
        })

    def fetch_neuron_info(self, body_ids, show_progress=True):
        return pd.DataFrame({
            "bodyId": ["101", "101"],
            "type": ["broad_class", ""],
            "instance": ["", "PPL101"],
            "tag": ["", "PPL101"],
        })


def test_fafb_no_cache_uses_online_body_and_type_denominators():
    finder = _stub_fafb_finder()
    finder._get_cave_fetcher = lambda: _FakeCaveFetcher()

    by_body = finder._fetch_total_incoming_weight(["101", "102"], min_weight=3)
    by_type = finder._fetch_total_incoming_weight_by_type(["T"], min_weight=3)

    assert dict(zip(by_body["bodyId_post"], by_body["total_incoming_weight"])) == {
        "101": 4,
        "102": 5,
    }
    assert dict(zip(by_type["type_post"], by_type["total_incoming_weight"])) == {
        "T": 9,
    }


def test_fafb_no_cache_routes_connections_to_api():
    finder = _stub_fafb_finder()
    calls = []

    def fake_cave_fetch(*args, **kwargs):
        calls.append((args, kwargs))
        return pd.DataFrame({
            "bodyId_pre": ["101"],
            "bodyId_post": ["202"],
            "weight": [4],
            "roi": ["WholeBrain"],
        })

    finder._fetch_connections_with_cave_api = fake_cave_fetch
    result = finder._fetch_connections_with_cache(
        upstream_bodyIds=["101"],
        downstream_bodyIds=None,
        min_weight=3,
        min_conn_ratio=0.0,
        min_traversal_prob=0.0,
    )

    assert len(result) == 1
    assert calls and calls[0][0][0] == ["101"]


def test_fafb_no_cache_never_reads_connection_cache():
    finder = _stub_fafb_finder()
    finder._conn_df_cache = pl.DataFrame({
        "bodyId_pre": ["cached"],
        "bodyId_post": ["cached"],
        "weight": [99],
    })
    finder._get_connection_db_path = lambda: (_ for _ in ()).throw(
        AssertionError("online-only mode inspected the connection-cache path")
    )

    result = finder._load_connection_db()

    assert result.is_empty()


def test_fafb_online_neuron_metadata_is_one_row_per_body_id():
    finder = _stub_fafb_finder()
    finder.client_flywire = None
    finder._get_cave_fetcher = lambda: _FakeCaveFetcher()

    result = finder._fetch_flywire_neurons_online(
        ["101"], columns=["bodyId", "type", "instance"]
    )

    assert len(result) == 1
    assert result.iloc[0].to_dict() == {
        "bodyId": "101",
        "type": "PPL101",
        "instance": "PPL101",
    }


def test_polars_enrichment_casts_empty_global_denominator_numeric():
    from statvis import EnrichConnectionTablePolars

    conn = pd.DataFrame({
        "bodyId_pre": ["A"],
        "bodyId_post": ["P1"],
        "type_pre": ["AType"],
        "type_post": ["T"],
        "weight": [3],
    })
    target = pl.DataFrame({
        "bodyId": ["P1"],
        "type": ["T"],
        "post": [10],
    })
    empty_body = pd.DataFrame(
        columns=["bodyId_post", "total_incoming_weight"]
    )
    empty_type = pd.DataFrame(
        columns=["type_post", "total_incoming_weight"]
    )

    conn_enriched, conn_type, _ = EnrichConnectionTablePolars(
        conn,
        target_neurons_df=target,
        global_incoming_body_weights=empty_body,
        global_incoming_weights=empty_type,
    )

    assert conn_enriched["connection_ratio"].dtype == pl.Float64
    assert conn_enriched["connection_ratio"].to_list() == [1.0]
    assert conn_type["connection_ratio"].to_list() == [1.0]


def _extrusion_neuron():
    import navis

    return navis.TreeNeuron(pd.DataFrame({
        "node_id": [0], "parent_id": [-1],
        "x": [0.0], "y": [0.0], "z": [0.0],
        "radius": [1.0], "type": ["root"],
    }))


def test_flag_extrusions_use_cache_false_skips_parquet(monkeypatch, tmp_path):
    """use_cache=False runs extrusion detection in-memory only and never
    touches extrusion_check_results.parquet (strict use_cache policy)."""
    from fafb_utils import flag_extrusions

    monkeypatch.setattr(
        "fafb_utils.load_extrusion_check_cache",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("use_cache=False read the extrusion parquet")),
    )
    monkeypatch.setattr(
        "fafb_utils.save_extrusion_check_cache",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("use_cache=False wrote the extrusion parquet")),
    )
    monkeypatch.setattr(
        "fafb_utils.detect_extrusion",
        lambda neuron, simplification=0.95: True,
    )

    flagged = flag_extrusions(
        str(tmp_path), "flywire_FAFB_v783", {"101": _extrusion_neuron()},
        use_cache=False, n_workers=0)

    assert flagged == [101]


def test_flag_extrusions_use_cache_true_serves_known_results(
        monkeypatch, tmp_path):
    """Cached extrusion results are served without re-running detection."""
    from fafb_utils import flag_extrusions

    monkeypatch.setattr(
        "fafb_utils.load_extrusion_check_cache",
        lambda *a, **k: {"101": True},
    )
    monkeypatch.setattr(
        "fafb_utils.detect_extrusion",
        lambda neuron, simplification=0.95: (_ for _ in ()).throw(
            AssertionError("cached result was re-detected")),
    )

    flagged = flag_extrusions(
        str(tmp_path), "flywire_FAFB_v783", {"101": _extrusion_neuron()},
        use_cache=True, n_workers=0)

    assert flagged == [101]
