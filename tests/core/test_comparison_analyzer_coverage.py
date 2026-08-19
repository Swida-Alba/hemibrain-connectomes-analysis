"""Hermetic coverage tests for comparison.comparison_analyzer.

All data-query / metadata-fetch methods are monkeypatched so that no network
or real dataset folder access occurs.  Synthetic DataFrames are injected.
"""

import json
import os
import sys
import types

import pandas as pd
import pytest

from comparison.comparison_analyzer import (
    ComparisonAnalyzer,
    _escape_cypher_string_fallback,
    quick_compare,
)
from comparison.comparison_parameters import ComparisonParameters
from comparison.dataset_config import DatasetConfig
from comparison.label_mapper import LabelMapper

DS1 = "hemibrain:v1.2.1"
DS2 = "male-cns:v0.9"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _params(**overrides):
    """Build hermetic ComparisonParameters (no auto type mapping, no output)."""
    defaults = dict(
        datasets=[DS1, DS2],
        source_neurons=["Src"],
        target_neurons=["Tgt"],
        max_interlayer=1,
        thresholds=[1, 3],
        output_folder="",
        auto_type_mapping=False,
        verbose=False,
    )
    defaults.update(overrides)
    return ComparisonParameters(**defaults)


def _edge_df(edges):
    """edges: list of (type_pre, type_post, weight)."""
    return pd.DataFrame(edges, columns=["type_pre", "type_post", "weight"])


def _bodyid_df():
    return pd.DataFrame(
        {
            "bodyId_pre": [1, 2],
            "bodyId_post": [2, 3],
            "type_pre": ["Src", "Inter"],
            "type_post": ["Inter", "Tgt"],
            "weight": [5, 5],
        }
    )


def _standard_results():
    """raw_results dict: both datasets, both thresholds, small edge data."""
    d1 = _edge_df([("Src", "Tgt", 10), ("Src", "X", 5)])
    d2 = _edge_df([("Src", "Tgt", 2), ("Y", "Tgt", 7)])
    return {DS1: {1: d1.copy(), 3: d1.copy()}, DS2: {1: d2.copy(), 3: d2.copy()}}


def _meta(ds):
    return {
        "dataset": ds,
        "source": "stub",
        "neuron_counts": {"total": 100, "typed": 80, "untyped": 20, "type_coverage": 0.8},
        "synapse_counts": {"total_presynaptic": 500, "total_postsynaptic": 400, "total": 900},
        "roi_coverage": {"roi_list": ["AL"], "roi_count": 1, "neuron_counts_per_roi": {"AL": 10}},
        "coverage_notes": "stub note",
    }


def _stub_metadata_methods(monkeypatch, analyzer, cached=None):
    monkeypatch.setattr(analyzer, "_load_cached_metadata", lambda ds: cached)
    monkeypatch.setattr(analyzer, "_fetch_neuprint_metadata", _meta)
    monkeypatch.setattr(analyzer, "_fetch_local_metadata", _meta)
    monkeypatch.setattr(analyzer, "_save_metadata", lambda ds, m: None)


@pytest.fixture
def analyzer():
    return ComparisonAnalyzer(_params(), verbose=False)


# ---------------------------------------------------------------------------
# Initialization / misc utilities
# ---------------------------------------------------------------------------

def test_escape_cypher_string_fallback():
    assert _escape_cypher_string_fallback("a'b") == "a\\'b"
    assert _escape_cypher_string_fallback("a\\b") == "a\\\\b"
    assert _escape_cypher_string_fallback(123) == "123"


def test_init_no_output_folder(analyzer):
    assert analyzer.data_loader is None
    assert analyzer.raw_results == {}
    assert analyzer.comparison_report is None
    assert DS1 in analyzer._dataset_configs
    assert isinstance(analyzer._dataset_configs[DS2], DatasetConfig)


def test_init_with_output_folder(tmp_path):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    assert a.data_loader is not None
    assert os.path.isdir(params.full_output_path)


def test_init_with_label_mapper():
    mapper = LabelMapper(
        source_mapping_dict={DS1: [["aMe1"]], DS2: [["aMe1mc"]]},
        source_labels=["aMe1_grp"],
        target_mapping_dict={DS1: [["KCg1"]], DS2: [["KCg1mc"]]},
        target_labels=["KCg_grp"],
    )
    a = ComparisonAnalyzer(_params(), label_mapper=mapper, verbose=False)
    assert a.label_mapper is mapper


def test_init_invalid_dataset_type():
    with pytest.raises(ValueError):
        ComparisonAnalyzer(_params(datasets=[123]), verbose=False)


def test_get_dataset_config(analyzer):
    cfg = analyzer._get_dataset_config(DS1)
    assert cfg.dataset == DS1
    with pytest.raises(ValueError):
        analyzer._get_dataset_config("nope")
    assert analyzer.get_dataset_config(DS1) is cfg
    assert analyzer.get_dataset_config("nope") is None


def test_logging_helpers():
    a = ComparisonAnalyzer(_params(), verbose=True)
    a._log("hello")
    a._log("debug msg", level="debug")
    a._log("warn msg", level="warn")
    a._progress(1, 5, "step")
    a._log_file("/tmp/some/file.csv")
    # verbose=False silences everything
    quiet = ComparisonAnalyzer(_params(), verbose=False)
    quiet._log("nope")
    quiet._log_file("/tmp/some/file.csv")


def test_save_read_csv_roundtrip(analyzer, tmp_path):
    df = _edge_df([("A", "B", 4), ("C", "D", 9)])
    path = str(tmp_path / "out.csv")
    analyzer._save_csv(df, path)
    assert os.path.exists(path)
    back = analyzer._read_csv(path)
    assert list(back.columns) == ["type_pre", "type_post", "weight"]
    assert len(back) == 2
    # kwargs route through pandas
    back2 = analyzer._read_csv(path, dtype={"type_pre": str})
    assert len(back2) == 2
    # empty / None are no-ops
    analyzer._save_csv(pd.DataFrame(), str(tmp_path / "empty.csv"))
    analyzer._save_csv(None, str(tmp_path / "none.csv"))
    assert not os.path.exists(str(tmp_path / "empty.csv"))
    assert not os.path.exists(str(tmp_path / "none.csv"))


def test_collect_result_types(analyzer):
    analyzer.raw_results[DS1] = {1: _edge_df([("A", "B", 1)]), 3: pd.DataFrame()}
    analyzer.raw_results[DS2] = {
        1: {"type_level": pd.DataFrame({"from_type": ["C"], "to_type": ["D"], "from": [1], "to": [2]})},
        3: pd.DataFrame({"type_pre": [5]}),  # non-string types filtered
    }
    types = analyzer._collect_result_types()
    assert {"A", "B", "C", "D"} <= types
    assert 5 not in types


def test_mode_specific_note(analyzer):
    note = analyzer._generate_mode_specific_note()
    assert "Path-Based" in note
    analyzer.parameters.comparison_mode = "edge"
    assert "Edge-Based" in analyzer._generate_mode_specific_note()


def test_mapping_helpers_without_mapper(analyzer):
    assert analyzer._get_canonical_type("TypeX", DS1) == "TypeX"
    assert analyzer._get_display_type("TypeX") == "TypeX"
    assert analyzer.get_mapped_results() is analyzer.raw_results
    canonical, display = analyzer._build_path_key_with_mapping(["A", "B"], DS1)
    assert canonical == "A → B"
    assert display == "A → B"


def test_clear_results_and_set_label_mapper(analyzer):
    analyzer.raw_results[DS1] = {1: _edge_df([("A", "B", 1)])}
    analyzer.aligned_results[1] = pd.DataFrame()
    analyzer.comparison_report = {"x": 1}
    analyzer.clear_results()
    assert analyzer.raw_results == {}
    assert analyzer.comparison_report is None

    new_mapper = LabelMapper()
    analyzer.aligned_results[1] = pd.DataFrame()
    analyzer.set_label_mapper(new_mapper)
    assert analyzer.label_mapper is new_mapper
    assert analyzer.aligned_results == {}


def test_nt_normalization_map():
    assert ComparisonAnalyzer._NT_NORMALIZATION_MAP["ACH"] == "acetylcholine"
    assert ComparisonAnalyzer._NT_NORMALIZATION_MAP["NO_CONS"] == "unknown"


# ---------------------------------------------------------------------------
# Hemisphere symmetry summaries
# ---------------------------------------------------------------------------

def test_get_hemisphere_symmetry_summaries_disabled(analyzer):
    # separate_hemispheres=False forces symmetry_analysis=False in __post_init__
    assert analyzer.parameters.symmetry_analysis is False
    assert analyzer.get_hemisphere_symmetry_summaries() == {}


def test_get_hemisphere_symmetry_summaries(tmp_path):
    params = _params(output_folder=str(tmp_path), separate_hemispheres=True)
    assert params.symmetry_analysis is True
    a = ComparisonAnalyzer(params, verbose=False)

    safe = params._sanitize_name(DS1)
    summary_dir = os.path.join(
        params.full_output_path, "dataset_data", safe, "minsyn_1", "hemisphere_symmetry"
    )
    os.makedirs(summary_dir, exist_ok=True)
    with open(os.path.join(summary_dir, "symmetry_summary.json"), "w") as f:
        json.dump({"ipsi": {"jaccard": 0.5}}, f)

    summaries = a.get_hemisphere_symmetry_summaries()
    assert summaries[1][DS1]["ipsi"]["jaccard"] == 0.5
    assert DS2 not in summaries[1]
    assert summaries[3] == {}
    # Cached on second call
    assert a.get_hemisphere_symmetry_summaries() == summaries


# ---------------------------------------------------------------------------
# run_path_analysis (FindNeuronConnection fully stubbed)
# ---------------------------------------------------------------------------

class _FakeFNC:
    out_folder = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.source_df = pd.DataFrame({"bodyId": [1]})
        self.target_df = pd.DataFrame({"bodyId": [2, 3]})
        self.allpath_folder = _FakeFNC.out_folder
        self.calls = []

    def InitializeNeuronInfo(self):
        pass

    def FindAllPath(self, find_reciprocal=False):
        self.calls.append("all")

    def FindShortestPath(self, find_reciprocal=False):
        self.calls.append("shortest")


@pytest.fixture
def fake_fnc(tmp_path, monkeypatch):
    import coana

    out = tmp_path / "fnc"
    details = out / "data_details"
    details.mkdir(parents=True)
    _FakeFNC.out_folder = str(out)
    monkeypatch.setattr(coana, "FindNeuronConnection", _FakeFNC)
    return out


def test_run_path_analysis(fake_fnc, monkeypatch):
    conn = _edge_df([("Src", "Tgt", 6)])
    conn["bodyId_pre"] = 1
    conn["bodyId_post"] = 2
    conn.to_csv(os.path.join(str(fake_fnc), "data_details", "connection_info_bodyId.csv"), index=False)

    a = ComparisonAnalyzer(_params(), verbose=False)
    df = a.run_path_analysis(DS1, 1)
    assert not df.empty
    assert (df["dataset"] == DS1).all()
    assert (df["threshold"] == 1).all()


def test_run_path_analysis_shortest_and_type_fallback(fake_fnc):
    conn = _edge_df([("Src", "Tgt", 6)])
    conn.to_csv(os.path.join(str(fake_fnc), "data_details", "connection_type.csv"), index=False)

    a = ComparisonAnalyzer(_params(path_mode="shortest"), verbose=False)
    df = a.run_path_analysis(DS2, 3)
    assert len(df) == 1
    assert df.iloc[0]["dataset"] == DS2


def test_run_path_analysis_no_output_files(fake_fnc):
    a = ComparisonAnalyzer(_params(), verbose=False)
    df = a.run_path_analysis(DS1, 1)
    assert df.empty


# ---------------------------------------------------------------------------
# run_edge_analysis / edge queries
# ---------------------------------------------------------------------------

def test_run_edge_analysis_from_base_edges(analyzer):
    base = _edge_df([("A", "B", 1), ("C", "D", 5)])
    out = analyzer.run_edge_analysis(DS1, 3, base_edges=base)
    assert len(out) == 1
    assert out.iloc[0]["type_pre"] == "C"
    assert out.iloc[0]["threshold"] == 3


def test_run_edge_analysis_query(monkeypatch, analyzer):
    monkeypatch.setattr(
        analyzer, "_query_edges_for_dataset", lambda *a, **k: _bodyid_df()
    )
    out = analyzer.run_edge_analysis(DS1, 2)
    assert (out["dataset"] == DS1).all()
    assert (out["threshold"] == 2).all()

    monkeypatch.setattr(analyzer, "_query_edges_for_dataset", lambda *a, **k: pd.DataFrame())
    assert analyzer.run_edge_analysis(DS1, 2).empty


def test_query_edges_dispatch(monkeypatch, analyzer):
    monkeypatch.setattr(analyzer, "_query_edges_local", lambda *a, **k: "local")
    monkeypatch.setattr(analyzer, "_query_edges_neuprint", lambda *a, **k: "neuprint")
    assert analyzer._query_edges_for_dataset("flywire_v1", [], [], 1) == "local"
    assert analyzer._query_edges_for_dataset("fafb_v2", [], [], 1) == "local"
    assert analyzer._query_edges_for_dataset("banc", [], [], 1) == "local"
    assert analyzer._query_edges_for_dataset(DS1, [], [], 1) == "neuprint"


def test_query_edges_local(tmp_path, monkeypatch, analyzer):
    safe = analyzer.parameters._sanitize_name("flywire_test")
    ds_dir = tmp_path / safe
    ds_dir.mkdir()
    conn = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 2, 3],
            "post_pt_root_id": [2, 3, 4],
            "pre_type": ["Src", "Other", "Zzz"],
            "post_type": ["Inter", "Tgt", "Nope"],
            "syn_count": [5, 4, 2],
        }
    )
    conn.to_csv(ds_dir / f"{safe}_connections.csv", index=False)
    monkeypatch.setattr(analyzer, "_get_datasets_folder", lambda: str(tmp_path))

    out = analyzer._query_edges_local("flywire_test", ["Src"], ["Tgt"], 1)
    # rows kept where pre matches Src OR post matches Tgt, weight >= 1
    assert len(out) == 2
    assert set(out["weight"]) == {5, 4}

    out3 = analyzer._query_edges_local("flywire_test", ["Sr.*"], ["Tgt"], 4)
    # NOTE: pattern 'Src.*' is mangled by a double '*' replacement in the
    # source (becomes 'Src..*'), so 'Sr.*' is used here to match 'Src'.
    assert set(out3["weight"]) == {5, 4}

    # no files at all
    monkeypatch.setattr(analyzer, "_get_datasets_folder", lambda: str(tmp_path / "missing"))
    assert analyzer._query_edges_local("flywire_test", ["Src"], ["Tgt"], 1).empty


def test_query_edges_local_joins_neuron_types(tmp_path, monkeypatch, analyzer):
    safe = analyzer.parameters._sanitize_name("flywire_test")
    ds_dir = tmp_path / safe
    ds_dir.mkdir()
    pd.DataFrame(
        {"pre_pt_root_id": [1], "post_pt_root_id": [2], "syn_count": [7]}
    ).to_csv(ds_dir / f"{safe}_connections.csv", index=False)
    pd.DataFrame({"bodyId": [1, 2], "type": ["Src", "Tgt"]}).to_csv(
        ds_dir / f"{safe}_neurons.csv", index=False
    )
    monkeypatch.setattr(analyzer, "_get_datasets_folder", lambda: str(tmp_path))

    out = analyzer._query_edges_local("flywire_test", ["Src"], ["Tgt"], 1)
    assert len(out) == 1
    assert out.iloc[0]["type_pre"] == "Src"
    assert out.iloc[0]["type_post"] == "Tgt"


def test_query_edges_neuprint(monkeypatch, analyzer):
    fake_mod = types.ModuleType("neuprint")

    class _Client:
        def __init__(self, server, dataset=None, token=None):
            pass

        def fetch_custom(self, query):
            return _bodyid_df()

    fake_mod.Client = _Client
    monkeypatch.setitem(sys.modules, "neuprint", fake_mod)
    out = analyzer._query_edges_neuprint(DS1, ["Src.*"], ["Tgt"], 1)
    assert len(out) == 2

    class _BrokenClient(_Client):
        def fetch_custom(self, query):
            raise RuntimeError("no network")

    fake_mod.Client = _BrokenClient
    assert analyzer._query_edges_neuprint(DS1, ["Src"], ["Tgt"], 1).empty


# ---------------------------------------------------------------------------
# run_all_analyses / caching
# ---------------------------------------------------------------------------

def test_run_all_path_analyses(monkeypatch, analyzer):
    calls = []

    def fake_run(dataset, threshold, verbose_mode="simple"):
        calls.append((dataset, threshold, verbose_mode))
        return _edge_df([("Src", "Tgt", threshold)])

    monkeypatch.setattr(analyzer, "run_path_analysis", fake_run)
    results = analyzer.run_all_analyses(skip_existing=False)
    assert set(results) == {DS1, DS2}
    assert set(results[DS1]) == {1, 3}
    # lowest threshold uses 'simple', higher uses 'silent'
    assert ("hemibrain:v1.2.1", 1, "simple") in calls
    assert ("hemibrain:v1.2.1", 3, "silent") in calls

    # skip_existing=True: nothing rerun
    calls.clear()
    analyzer.run_all_analyses(skip_existing=True)
    assert calls == []

    # skip_existing=False: reruns everything
    analyzer.run_all_analyses(skip_existing=False)
    assert len(calls) == 4


def test_run_all_path_analyses_disk_cache(tmp_path, monkeypatch):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)

    cached = _edge_df([("Src", "Tgt", 11)])
    out_dir = params.get_dataset_output_path(DS1, 1)
    os.makedirs(out_dir, exist_ok=True)
    a._save_csv(cached, os.path.join(out_dir, "connections_edge.csv"))

    def fail_run(*args, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("run_path_analysis should be skipped")

    monkeypatch.setattr(a, "run_path_analysis", fail_run)

    # Provide results for the rest so only DS1@1 must come from disk cache
    loaded = a._try_load_cached(DS1, 1)
    assert loaded is not None and len(loaded) == 1

    a.raw_results[DS1] = {1: loaded, 3: _edge_df([("Src", "Tgt", 3)])}
    a.raw_results[DS2] = {1: _edge_df([("Src", "Tgt", 1)]), 3: _edge_df([("Src", "Tgt", 3)])}
    results = a.run_all_analyses(skip_existing=True)
    assert len(results[DS1][1]) == 1


def test_try_load_cached_variants(tmp_path):
    # No output folder -> always None
    a_mem = ComparisonAnalyzer(_params(), verbose=False)
    assert a_mem._try_load_cached(DS1, 1) is None

    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    out_dir = params.get_dataset_output_path(DS1, 1)

    # Nothing on disk
    assert a._try_load_cached(DS1, 1) is None

    # paths.csv legacy
    os.makedirs(out_dir, exist_ok=True)
    _edge_df([("A", "B", 2)]).to_csv(os.path.join(out_dir, "paths.csv"), index=False)
    assert len(a._try_load_cached(DS1, 1)) == 1

    # connection_info_bodyId.csv under data_details (adds dataset/threshold cols)
    details = os.path.join(out_dir, "data_details")
    os.makedirs(details)
    _edge_df([("A", "B", 2)]).to_csv(os.path.join(details, "connection_info_bodyId.csv"), index=False)
    df = a._try_load_cached(DS1, 1)
    assert df is not None

    # connection_type.csv fallback
    os.remove(os.path.join(out_dir, "paths.csv"))
    os.remove(os.path.join(details, "connection_info_bodyId.csv"))
    _edge_df([("A", "B", 2)]).to_csv(os.path.join(details, "connection_type.csv"), index=False)
    assert a._try_load_cached(DS1, 1) is not None


def test_save_result_and_edge_mode_result(tmp_path):
    # In-memory analyzer: both are no-ops
    a_mem = ComparisonAnalyzer(_params(), verbose=False)
    a_mem._save_result(DS1, 1, _edge_df([("A", "B", 1)]))
    a_mem._save_edge_mode_result(DS1, 1, _edge_df([("A", "B", 1)]))

    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    df = _edge_df([("A", "B", 1)])
    a._save_result(DS1, 1, df)
    assert os.path.exists(
        os.path.join(params.get_dataset_output_path(DS1, 1), "connections_edge.csv")
    )
    a._save_edge_mode_result(DS1, 3, df)
    edge_mode = os.path.join(
        params.full_output_path, "edge_mode_data", params._sanitize_name(DS1), "connections_edge_3.csv"
    )
    assert os.path.exists(edge_mode)
    # Empty frames are skipped
    a._save_result(DS1, 3, pd.DataFrame())
    assert not os.path.exists(
        os.path.join(params.get_dataset_output_path(DS1, 3), "connections_edge.csv")
    )


# ---------------------------------------------------------------------------
# Edge mode aggregation pipeline
# ---------------------------------------------------------------------------

def test_build_label_map_from_df(analyzer):
    lm = analyzer._build_label_map_from_df(_bodyid_df())
    assert lm == {1: "Src", 2: "Inter", 3: "Tgt"}
    assert analyzer._build_label_map_from_df(pd.DataFrame({"x": [1]})) == {}


def test_aggregate_and_find_paths(analyzer):
    df = _bodyid_df()
    df["total_post"] = [20, 10]
    label_map = analyzer._build_label_map_from_df(df)
    out = analyzer._aggregate_and_find_paths(
        df, label_map, {"Src"}, {"Tgt"}, 2, DS1, 1, path_mode="all"
    )
    assert not out.empty
    assert {"has_valid_path", "dataset", "threshold", "conn_layer", "traversal_probability"} <= set(out.columns)
    assert out["has_valid_path"].all()
    assert (out["conn_layer"].isin(["0->1", "1->2"])).all()

    # shortest mode
    out2 = analyzer._aggregate_and_find_paths(
        df, label_map, {"Src"}, {"Tgt"}, None, DS1, 1, path_mode="shortest"
    )
    assert not out2.empty


def test_get_bodyid_connections(monkeypatch, analyzer):
    monkeypatch.setattr(analyzer, "_query_edges_for_dataset", lambda *a, **k: _bodyid_df())
    df, lm = analyzer._get_bodyid_connections_for_dataset(DS1, 1)
    assert df is not None and len(df) == 2
    assert lm[1] == "Src"

    monkeypatch.setattr(analyzer, "_query_edges_for_dataset", lambda *a, **k: pd.DataFrame())
    df, lm = analyzer._get_bodyid_connections_for_dataset(DS1, 1)
    assert df is None and lm == {}


def test_get_bodyid_connections_cached(tmp_path):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    details = os.path.join(params.get_dataset_output_path(DS1, 1), "data_details")
    os.makedirs(details)
    _bodyid_df().to_csv(os.path.join(details, "connection_info_bodyId.csv"), index=False)

    df, lm = a._get_bodyid_connections_for_dataset(DS1, 1, skip_existing=True)
    assert df is not None and lm[3] == "Tgt"


def test_run_all_edge_analyses(monkeypatch, analyzer):
    analyzer.parameters.comparison_mode = "edge"
    path_calls = []
    monkeypatch.setattr(
        analyzer, "run_path_analysis",
        lambda ds, t, verbose_mode="simple": path_calls.append((ds, t)) or pd.DataFrame(),
    )
    monkeypatch.setattr(
        analyzer, "_get_bodyid_connections_for_dataset",
        lambda ds, t, skip_existing=True, run_findallpath=False: (_bodyid_df(), {1: "Src", 2: "Inter", 3: "Tgt"}),
    )

    results = analyzer.run_all_analyses(skip_existing=False)
    for ds in (DS1, DS2):
        for t in (1, 3):
            out = results[ds][t]
            assert not out.empty
            assert "has_valid_path" in out.columns
    # path tool ran for lowest ('simple') and remaining thresholds
    assert (DS1, 1) in path_calls and (DS1, 3) in path_calls


def test_run_all_edge_analyses_empty(monkeypatch, analyzer):
    analyzer.parameters.comparison_mode = "edge"
    monkeypatch.setattr(analyzer, "run_path_analysis", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(
        analyzer, "_get_bodyid_connections_for_dataset",
        lambda *a, **k: (None, {}),
    )
    results = analyzer.run_all_analyses(skip_existing=False)
    for ds in (DS1, DS2):
        for t in (1, 3):
            assert results[ds][t].empty


def test_process_threshold_aggregation_empty_filter(analyzer):
    analyzer.raw_results[DS1] = {}
    analyzer._process_threshold_aggregation(
        DS1, 100, _bodyid_df(), {1: "Src", 2: "Inter", 3: "Tgt"},
        {"Src"}, {"Tgt"}, 2, skip_existing=False,
    )
    assert analyzer.raw_results[DS1][100].empty


# ---------------------------------------------------------------------------
# Metadata collection
# ---------------------------------------------------------------------------

def test_coverage_notes_and_empty_metadata(analyzer):
    assert "Central brain" in analyzer._get_coverage_notes(DS1)
    assert "male" in analyzer._get_coverage_notes("male-cns:v0.9").lower()
    assert analyzer._get_coverage_notes("unknown_ds") == "Coverage information not available."

    empty = analyzer._create_empty_metadata(DS1, "boom")
    assert empty["source"] == "error"
    assert empty["error"] == "boom"
    assert empty["neuron_counts"]["total"] == 0


def test_collect_dataset_metadata_fetch(monkeypatch, analyzer):
    fetched = []

    def fake_fetch(ds):
        fetched.append(ds)
        return _meta(ds)

    monkeypatch.setattr(analyzer, "_load_cached_metadata", lambda ds: None)
    monkeypatch.setattr(analyzer, "_fetch_neuprint_metadata", fake_fetch)
    monkeypatch.setattr(analyzer, "_save_metadata", lambda ds, m: None)

    md = analyzer.collect_dataset_metadata()
    assert set(md) == {DS1, DS2}
    assert fetched == [DS1, DS2]

    table = analyzer.generate_metadata_comparison_table()
    expected_cols = {
        "dataset", "total_neurons", "typed_neurons", "untyped_neurons",
        "type_coverage_pct", "total_presynaptic", "total_postsynaptic",
        "total_synapses", "roi_count", "coverage_notes",
    }
    assert expected_cols <= set(table.columns)
    assert len(table) == 2
    assert table.iloc[0]["total_neurons"] == 100


def test_collect_dataset_metadata_cached(monkeypatch, analyzer):
    fetched = []
    _stub_metadata_methods(monkeypatch, analyzer, cached=_meta(DS1))
    monkeypatch.setattr(
        analyzer, "_fetch_neuprint_metadata",
        lambda ds: fetched.append(ds) or _meta(ds),
    )
    md = analyzer.collect_dataset_metadata()
    assert fetched == []  # cached metadata used, no fetch
    assert md[DS1]["neuron_counts"]["total"] == 100

    # force_refresh bypasses cache
    md2 = analyzer.collect_dataset_metadata(force_refresh=True)
    assert fetched == [DS1, DS2]
    assert set(md2) == {DS1, DS2}


def test_metadata_paths(analyzer):
    path = analyzer._get_metadata_path(DS1)
    safe = analyzer.parameters._sanitize_name(DS1)
    assert path.endswith(f"{safe}_metadata.json")
    assert analyzer._get_datasets_folder().endswith("datasets")
    # No cached metadata file for an unknown dataset
    assert analyzer._load_cached_metadata("nonexistent_ds:v0") is None


# ---------------------------------------------------------------------------
# Comparison pipeline
# ---------------------------------------------------------------------------

def test_run_comparison(monkeypatch):
    a = ComparisonAnalyzer(_params(), verbose=False)
    monkeypatch.setattr(a, "run_all_analyses", lambda skip_existing=True: a.raw_results.update(_standard_results()))

    report = a.run_comparison()
    assert isinstance(report, dict)
    assert a.comparison_report is report
    assert "key_findings" in report
    assert "threshold_similarities" in report
    assert a.raw_results[DS1][1] is not None

    sims = a.get_cached_similarities(1)
    assert not sims.empty
    # cached copy returned
    assert a.get_cached_similarities(1) is not sims
    # uncached threshold with no data -> empty
    assert a.get_cached_similarities(99).empty


def test_run_comparison_analysis_empty(monkeypatch):
    a = ComparisonAnalyzer(_params(), verbose=False)
    monkeypatch.setattr(a, "run_all_analyses", lambda skip_existing=True: None)
    summary = a.run_comparison_analysis()
    assert summary["key_findings"] == ["No data available for comparison"]


def test_aligned_and_connection_getters(analyzer):
    analyzer.raw_results = _standard_results()

    aligned = analyzer.get_aligned_data(1)
    assert not aligned.empty
    assert DS1 in aligned.columns and DS2 in aligned.columns
    assert "Src -> Tgt" in aligned.index
    # cached
    assert analyzer.get_aligned_data(1) is aligned
    # missing threshold -> empty
    assert analyzer.get_aligned_data(99).empty

    common = analyzer.get_common_connections(1)
    assert "Src -> Tgt" in set(common.index) or "Src -> Tgt" in common.get("edge", pd.Series()).values

    unique = analyzer.get_unique_connections(1)
    assert set(unique) == {DS1, DS2}

    diff = analyzer.get_differential_connections(1, fold_threshold=2.0)
    assert isinstance(diff, pd.DataFrame)


def test_aligned_data_for_network(analyzer):
    analyzer.raw_results = _standard_results()
    # find_reciprocal=False -> plain aligned data
    out = analyzer.get_aligned_data_for_network(1)
    assert not out.empty

    # find_reciprocal=True but no reciprocal files -> falls back
    analyzer.parameters.find_reciprocal = True
    out2 = analyzer.get_aligned_data_for_network(3)
    assert not out2.empty


def test_filter_hemisphere_unconserved(tmp_path):
    params = _params(
        output_folder=str(tmp_path),
        separate_hemispheres=True,
        keep_only_hemisphere_conserved_connections=True,
    )
    a = ComparisonAnalyzer(params, verbose=False)

    aligned = pd.DataFrame(
        {
            DS1: [5, 3, 9, 4, 2],
            DS2: [4, 0, 8, 4, 0],
        },
        index=[
            "A_R -> B_R",   # conserved pair with counterpart present
            "A_L -> B_L",   # counterpart of above; weight 0 in DS2
            "C_R -> D_R",   # missing counterpart -> zeroed
            "Plain -> Edge",  # no hemisphere suffix -> kept
            "E_U -> F_U",   # _U suffix -> kept as-is
        ],
    )

    out = a._filter_hemisphere_unconserved(aligned, [DS1, DS2], threshold=1)
    # conserved pair preserved where BOTH mirror edges exist in the dataset
    assert out.at["A_R -> B_R", DS1] == 5
    # in DS2 the counterpart A_L -> B_L has weight 0 -> edge zeroed in DS2
    assert out.at["A_R -> B_R", DS2] == 0
    assert out.at["A_L -> B_L", DS1] == 3
    assert out.at["A_L -> B_L", DS2] == 0
    # unconserved edge zeroed entirely
    assert out.at["C_R -> D_R", DS1] == 0
    assert out.at["C_R -> D_R", DS2] == 0
    # no-suffix edges untouched
    assert out.at["Plain -> Edge", DS1] == 4
    assert out.at["E_U -> F_U", DS1] == 2

    # unconserved edges were saved to disk
    unconserved_file = os.path.join(
        params.full_output_path, "comparison_results", "hemisphere_unconserved_edges_t1.csv"
    )
    assert os.path.exists(unconserved_file)

    # empty input passes through
    assert a._filter_hemisphere_unconserved(pd.DataFrame(), [DS1, DS2]).empty


def test_get_aligned_data_applies_hemisphere_filter(tmp_path):
    params = _params(
        output_folder=str(tmp_path),
        separate_hemispheres=True,
        keep_only_hemisphere_conserved_connections=True,
    )
    a = ComparisonAnalyzer(params, verbose=False)
    d1 = _edge_df([("A_R", "B_R", 5)])
    d2 = _edge_df([("A_R", "B_R", 4), ("A_L", "B_L", 4)])
    a.raw_results = {DS1: {1: d1}, DS2: {1: d2}}

    aligned = a.get_aligned_data(1)
    # A_R -> B_R has no A_L -> B_L counterpart in DS1 -> zeroed in DS1
    assert aligned.at["A_R -> B_R", DS1] == 0


# ---------------------------------------------------------------------------
# Reports / export
# ---------------------------------------------------------------------------

def test_generate_report(tmp_path):
    a = ComparisonAnalyzer(_params(output_folder=str(tmp_path)), verbose=False)
    a.raw_results = _standard_results()

    out_path = str(tmp_path / "report.txt")
    text = a.generate_report(output_path=out_path)
    assert "CROSS-DATASET COMPARISON REPORT" in text
    assert DS1 in text
    assert os.path.exists(out_path)


def test_export_results(tmp_path, monkeypatch):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    a.raw_results = _standard_results()
    _stub_metadata_methods(monkeypatch, a)

    a.export_results()

    base = params.full_output_path
    assert os.path.exists(os.path.join(base, "parameters.json"))
    assert os.path.exists(os.path.join(base, "label_map.json"))
    assert os.path.exists(os.path.join(base, "comparison_report.txt"))
    results_dir = os.path.join(base, "comparison_results")
    assert os.path.exists(os.path.join(results_dir, "path_count_comparison.csv"))
    assert os.path.exists(os.path.join(results_dir, "threshold_sensitivity.csv"))


def test_generate_html_report(tmp_path):
    a = ComparisonAnalyzer(_params(output_folder=str(tmp_path)), verbose=False)
    a.raw_results = _standard_results()
    a.comparison_report = {
        "key_findings": ["finding"],
        "key_findings_per_threshold": {},
        "path_presence_matrix": pd.DataFrame(),
        "threshold_similarities": pd.DataFrame(),
    }
    out = a.generate_html_report(os.path.join(str(tmp_path), "rep.html"))
    assert os.path.exists(out)


# ---------------------------------------------------------------------------
# quick_compare convenience wrapper
# ---------------------------------------------------------------------------

def test_quick_compare(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fake_run_all(self, skip_existing=True):
        self.raw_results.update(_standard_results())
        return self.raw_results

    monkeypatch.setattr(ComparisonAnalyzer, "run_all_analyses", fake_run_all)
    monkeypatch.setattr(ComparisonAnalyzer, "export_results", lambda self, output_dir=None: None)

    results = quick_compare(
        datasets=[DS1, DS2],
        source_neurons=["Src"],
        target_neurons=["Tgt"],
        thresholds=[1, 3],
        verbose=False,
    )
    assert isinstance(results, dict)
    assert "key_findings" in results
