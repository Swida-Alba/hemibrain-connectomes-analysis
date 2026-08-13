"""Cross-surface conformance tests for the shared neuron lookup backend.

The same synthetic metadata is exercised through the parquet sidecar, the
dataframe fallback used by analysis tools, the UI pool matcher, and the
available-neurons page.  These tests intentionally include identities that
appear in a later taxonomy column so a type hit cannot accidentally expand a
pathfinding query into unrelated neurons.
"""

from pathlib import Path

import pandas as pd
import polars as pl


def _metadata() -> pd.DataFrame:
    return pd.DataFrame({
        "bodyId": ["100", "101", "102", "103", "104", "105", "106"],
        "type": [
            "aMe17a", "aMe17e", "MeVPaMe1", "MeVPaMe2", "Other", "", "",
        ],
        "instance": [
            "aMe17a_L", "aMe17e_L", "MeVPaMe1_R", "MeVPaMe2_R",
            "Other_L", "MTe07_L", "Other_R",
        ],
        "flywireType": [
            "aMe17a1", "", "MTe07", "aMe19a", "", "MTe07", "MTe12",
        ],
        "hemibrainType": ["", "aMe17a", "", "", "", "", ""],
        "mancType": ["", "", "", "", "", "", "MTe12"],
        "class": ["central", "central", "visual", "visual", "other", "visual", "visual"],
        # This must never become a search column.
        "notes": ["aMe17a", "aMe17e", "MTe07", "MeVPa", "MTe", "MTe", "MTe"],
    })


def _write_cache(tmp_path: Path, frame: pd.DataFrame):
    from src.neuron_index_builder import build_search_cache_frame, search_cache_path

    dataset = "conformance:v1.0"
    folder = dataset.replace(":", "_").replace(".", "_")
    cache_dir = tmp_path / "cache" / folder
    cache_dir.mkdir(parents=True)
    index_path = cache_dir / "neuron_index.parquet"
    polars_frame = pl.from_pandas(frame)
    polars_frame.write_parquet(index_path)
    build_search_cache_frame(polars_frame).write_parquet(
        search_cache_path(index_path)
    )
    return dataset, index_path


def _pool_from_frame(frame: pd.DataFrame):
    pools = {}
    for column in (
        "bodyId", "type", "instance", "flywireType", "hemibrainType", "mancType",
        "class",
    ):
        entries = []
        for value in frame[column].tolist():
            text = "" if pd.isna(value) else str(value).strip()
            if text:
                entries.append((text, column))
        # Keep the first hint/order just as the production pool builder does.
        pools[column] = list(dict.fromkeys(entries))
    return pools


def test_strict_analysis_surfaces_agree_on_priority_and_case(tmp_path):
    from src.neuron_search import (
        get_cached_neuron_search,
        resolve_dataframe_query,
        resolve_neuron_query,
    )
    from src.statvis import _process_single_neuron

    frame = _metadata()
    dataset, _index_path = _write_cache(tmp_path, frame)
    cache = get_cached_neuron_search(dataset, cache_root=tmp_path / "cache")
    assert cache is not None

    # Bare input is exact/case-sensitive.  Explicit starts-with syntax is
    # also case-sensitive and stops at the first owning priority column.
    queries = [
        "aMe17a", "aMe17a.*", "MeVPa", "MeVPa.*", "MTe07", "MTe.*",
        ".*MTe.*", "20", "20.*", "ame17a.*",
    ]
    for query in queries:
        expected, expected_info = resolve_dataframe_query(frame, query)
        cached, cached_info = resolve_neuron_query(cache, query)
        fallback, fallback_info = _process_single_neuron(
            query,
            frame,
            frame["bodyId"].tolist(),
            verbose=False,
        )
        assert {str(value) for value in cached} == {
            str(value) for value in expected
        }, query
        assert {str(value) for value in fallback} == {
            str(value) for value in expected
        }, query
        assert cached_info["matched_column"] == expected_info["matched_column"]
        assert fallback_info["matched_column"] == expected_info["matched_column"]

    # An exact type identity owns the query; the same spelling in a later
    # hemibrainType field does not pull in the independent aMe17e row.
    ids, info = resolve_dataframe_query(frame, "aMe17a")
    assert ids == ["100"]
    assert info["matched_column"] == "type"
    ids, info = resolve_dataframe_query(frame, "aMe17a.*")
    assert ids == ["100"]
    assert info["matched_column"] == "type"

    # A later-column exact match is still supported when no earlier field owns
    # the name, and arbitrary notes remain outside the lookup boundary.
    ids, info = resolve_dataframe_query(frame, "MTe07")
    assert ids == ["102", "105"]
    assert info["matched_column"] == "flywireType"
    ids, info = resolve_dataframe_query(frame, "MTe")
    assert ids == []
    assert info["matched_column"] is None


def test_interactive_ui_pool_and_viewer_share_the_same_stages(tmp_path):
    from src.neuron_search import match_search_pools, search_plan
    from ui.neuron_index import CachedNeuronIndex, query_neuron_index
    from ui.type_suggestions import match_suggestions

    frame = _metadata()
    pools = _pool_from_frame(frame)

    # The compatibility UI wrapper is the same implementation as the core
    # pool matcher, not a second matching algorithm.
    for query in ("a", "aMe", "MTe", "ame", "20"):
        assert match_suggestions(query, pools, limit=None) == match_search_pools(
            query, pools, limit=None
        )

    assert search_plan("aMe", pools.keys()) == search_plan(
        "aMe", pools.keys(), "auto"
    )
    assert match_suggestions("aMe", pools, limit=None) == [
        ("aMe17a", "type"), ("aMe17e", "type"),
    ]

    # The full viewer intentionally returns all strict-prefix columns and
    # then substring-only rows; its first page must still be the same
    # canonical priority order rather than lexical order.
    polars_frame = pl.from_pandas(frame.drop(columns=["notes"]))
    index = CachedNeuronIndex(
        dataset="conformance:v1.0",
        path=tmp_path / "neuron_index.parquet",
        frame=polars_frame,
        columns=tuple(polars_frame.columns),
    )
    page = query_neuron_index(index, search="aMe", page_size=100)
    assert page.total >= 4
    assert [row["match_column_key"] for row in page.rows[:2]] == ["type", "type"]
    assert any(
        "flywireType" in (row.get("secondary_match_column_keys") or [])
        for row in page.rows
    )
    assert all(row["bodyId"] != "106" for row in page.rows[:4])


def test_scope_and_numeric_guards_are_identical_across_surfaces():
    from src.neuron_search import resolve_dataframe_query
    from ui.type_suggestions import match_suggestions

    frame = _metadata()
    pools = _pool_from_frame(frame)

    # Numeric queries never spill into names that contain the same digits.
    ids, info = resolve_dataframe_query(frame, "17")
    assert ids == []
    assert info["matched_column"] is None
    assert match_suggestions("17", pools, limit=None) == []

    # Explicit scopes are honored by both the strict resolver and the
    # interactive pool matcher.
    ids, info = resolve_dataframe_query(frame, "MTe07", search_columns="type")
    assert ids == []
    assert info["matched_column"] is None
    assert match_suggestions("MTe", pools, "type", limit=None) == []


def test_stale_cache_falls_back_without_changing_lookup_semantics(tmp_path):
    from src.neuron_search import (
        get_cached_neuron_search,
        resolve_cached_or_dataframe_query,
    )

    frame = _metadata()
    dataset, _index_path = _write_cache(tmp_path, frame)
    cache = get_cached_neuron_search(dataset, cache_root=tmp_path / "cache")
    assert cache is not None

    changed = pd.concat([
        frame,
        pd.DataFrame({
            "bodyId": ["107"], "type": ["NewType"], "instance": ["NewType_L"],
            "flywireType": [""], "hemibrainType": [""], "mancType": [""],
            "class": ["new"], "notes": [""],
        }),
    ], ignore_index=True)
    ids, info = resolve_cached_or_dataframe_query(cache, changed, "NewType")
    assert ids == ["107"]
    assert info["cache"] is False


def test_filter_mode_patterns_are_literal_and_end_anchored():
    from src.neuron_search import resolve_dataframe_query
    from ui.components.common import apply_filter_mode

    frame = pd.DataFrame({
        "bodyId": ["1", "2", "3", "4"],
        "type": ["Cell_R", "Cell_L", "xCell_R", "Cell.R"],
        "instance": ["Cell_R", "Cell_L", "xCell_R", "Cell.R"],
    })
    suffix_query = apply_filter_mode(["_R"], "endswith")
    assert suffix_query == [".*_R$"]
    ids, _ = resolve_dataframe_query(frame, suffix_query[0])
    assert ids == ["1", "3"]

    literal_query = apply_filter_mode(["Cell.R"], "startswith")
    ids, _ = resolve_dataframe_query(frame, literal_query[0])
    assert ids == ["4"]


def test_neuronbridge_finder_uses_the_shared_dataframe_resolver(monkeypatch):
    from src.neuronbridge_finder import NeuronBridgeFinder

    frame = _metadata()
    finder = object.__new__(NeuronBridgeFinder)
    finder._load_neuron_df_for_dataset = lambda _folder: frame

    # The method receives the same explicit query forms as the UI filter
    # mode conversion, and must return the same body IDs as the core resolver.
    result = finder._find_bodyIds_by_query("MTe.*", "conformance:v1.0")
    # The instance column owns this prefix before flywireType.
    assert [row["bodyId"] for row in result] == ["105"]
    assert all(row["dataset_folder"] == "conformance_v1_0" for row in result)

    result = finder._find_bodyIds_by_query("ame17a.*", "conformance:v1.0")
    assert result == []

    result = finder._find_bodyIds_by_query(
        ["aMe17a", "MTe07"], "conformance:v1.0"
    )
    assert {(row["bodyId"], row["dataset_folder"]) for row in result} == {
        ("100", "conformance_v1_0"),
        ("102", "conformance_v1_0"),
        ("105", "conformance_v1_0"),
    }
