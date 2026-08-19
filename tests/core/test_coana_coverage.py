#!/usr/bin/env python
"""Coverage-focused hermetic tests for src/coana.py.

All tests run fully offline: connection tables are tiny synthetic
DataFrames injected directly or served by monkeypatched fetchers. File
I/O goes to pytest's tmp_path only. Existing tests already cover the
FindAllPath graph cache and path-edge/layer matching; this file targets
the remaining uncovered helpers:

  - connection-table-to-matrix pandas-2.x fallback
  - layer-table helpers (pandas branches), cache clearers
  - hemisphere normalization / suffix / filtering helpers
  - ratio & probability derivation
  - export normalizers and CSV/Excel matrix writers
  - query/export metadata helpers (all_neurons token, group labels,
    readable names, secret sanitization)
  - hemisphere conservation filtering + symmetry analysis
  - path-derivation helpers, warning notes, neuron enrollment
  - FindDirectConnections end-to-end with fake fetchers
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))

import coana  # noqa: E402


# =============================================================================
# fixture helpers
# =============================================================================

def make_fc(**attrs):
    """Build an uninitialized FindNeuronConnection with silent logging."""
    fc = object.__new__(coana.FindNeuronConnection)
    logs = []
    fc._vprint = lambda msg="", level="full", end="\n", flush=False: logs.append(str(msg))
    fc.verbose_mode = "silent"
    fc.progress_events = False
    fc._warn_notes = []
    for key, value in attrs.items():
        setattr(fc, key, value)
    return fc, logs


def conn_df(rows, extra=None):
    """rows: list of (pre, post, weight[, type_pre, type_post])"""
    data = {
        "bodyId_pre": [str(r[0]) for r in rows],
        "bodyId_post": [str(r[1]) for r in rows],
        "weight": [r[2] for r in rows],
    }
    if rows and len(rows[0]) > 3:
        data["type_pre"] = [r[3] for r in rows]
        data["type_post"] = [r[4] for r in rows]
    if extra:
        data.update(extra)
    return pd.DataFrame(data)


# =============================================================================
# connection_table_to_matrix fallback + layer helpers + cache clearers
# =============================================================================

class TestPatchedConnectionTableToMatrix:
    def test_pandas2_fallback_pivot(self, monkeypatch):
        rows = [("1", "2", 5), ("1", "2", 3), ("2", "3", 7)]
        df = conn_df(rows)

        def raiser(conn_df, group_cols="bodyId", weight_col="weight",
                   sort_by=None, make_square=False):
            raise TypeError("pivot() got an unexpected keyword argument")

        monkeypatch.setattr(coana, "_original_connection_table_to_matrix", raiser)
        mat = coana._patched_connection_table_to_matrix(df)
        assert mat.loc["1", "2"] == 8  # summed duplicate edge
        assert mat.loc["2", "3"] == 7

    def test_pandas2_fallback_make_square(self, monkeypatch):
        df = conn_df([("1", "2", 5)])

        def raiser(*a, **k):
            raise TypeError("nested pivot() failure")

        monkeypatch.setattr(coana, "_original_connection_table_to_matrix", raiser)
        mat = coana._patched_connection_table_to_matrix(
            df, group_cols="bodyId", make_square=True)
        assert list(mat.index) == ["1", "2"]
        assert list(mat.columns) == ["1", "2"]
        assert mat.loc["2", "1"] == 0

    def test_non_pivot_typeerror_propagates(self, monkeypatch):
        def raiser(*a, **k):
            raise TypeError("something else entirely")

        monkeypatch.setattr(coana, "_original_connection_table_to_matrix", raiser)
        with pytest.raises(TypeError, match="something else"):
            coana._patched_connection_table_to_matrix(conn_df([("1", "2", 1)]))


class TestLayerAndCacheHelpers:
    def test_layer_table_edge_pairs_pandas_branches(self):
        fc, _ = make_fc()
        assert coana._layer_table_edge_pairs(None) == set()
        empty = pd.DataFrame(columns=["bodyId_pre", "bodyId_post"])
        assert coana._layer_table_edge_pairs(empty) == set()
        pairs = coana._layer_table_edge_pairs(conn_df([(1, 2, 3), (3, 4, 1)]))
        assert pairs == {("1", "2"), ("3", "4")}

    def test_clear_findallpath_cache_dataset_normalized(self):
        cache = coana._FINDALLPATH_GRAPH_CACHE
        cache.clear()
        try:
            cache["hemibrain_v1_2_1_a"] = {"x": 1}
            cache["male_cns_v1_0_b"] = {"x": 2}
            coana.clear_findallpath_cache("hemibrain:v1.2.1")
            assert "hemibrain_v1_2_1_a" not in cache
            assert "male_cns_v1_0_b" in cache
            coana.clear_findallpath_cache()
            assert cache == {}
        finally:
            cache.clear()

    def test_clear_fnc_cache(self):
        cache = coana._FNC_CACHE
        cache["ds_a"] = {"conn_df": None}
        cache["ds_b"] = {"conn_df": None}
        try:
            coana.clear_fnc_cache("ds_a")
            assert "ds_a" not in cache and "ds_b" in cache
            coana.clear_fnc_cache()
            assert cache == {}
        finally:
            cache.clear()


# =============================================================================
# hemisphere helpers
# =============================================================================

class TestHemisphereHelpers:
    def test_normalize_hemisphere_value(self):
        fc, _ = make_fc()
        assert fc._normalize_hemisphere_value("right") == "R"
        assert fc._normalize_hemisphere_value(" L ") == "L"
        assert fc._normalize_hemisphere_value("lhs") == "L"
        assert fc._normalize_hemisphere_value(None) == "U"
        assert fc._normalize_hemisphere_value(float("nan")) == "U"
        assert fc._normalize_hemisphere_value("middle") == "U"

    def test_find_hemisphere_column(self):
        FNC = coana.FindNeuronConnection
        assert FNC._find_hemisphere_column(pd.DataFrame({"Soma side": ["R"]})) == "Soma side"
        assert FNC._find_hemisphere_column(pd.DataFrame({"somaSide": ["R"]})) == "somaSide"
        assert FNC._find_hemisphere_column(pd.DataFrame({"hemisphere": ["R"], "somaSide": ["L"]})) == "hemisphere"
        assert FNC._find_hemisphere_column(pd.DataFrame({"rootSide": ["R"]})) == "rootSide"
        assert FNC._find_hemisphere_column(pd.DataFrame({"type": ["x"]})) is None

    def test_ensure_hemisphere_columns(self):
        fc, _ = make_fc()
        # from Soma side column
        df = fc._ensure_hemisphere_columns(pd.DataFrame({"Soma side": ["R", "left"]}))
        assert list(df["hemisphere_code"]) == ["R", "L"]
        # from instance suffixes
        df = fc._ensure_hemisphere_columns(pd.DataFrame({"instance": ["a_R", "b_L", "c"]}))
        assert list(df["hemisphere_code"]) == ["R", "L", "U"]
        # no source at all
        df = fc._ensure_hemisphere_columns(pd.DataFrame({"type": ["x"]}))
        assert list(df["hemisphere_code"]) == ["U"]
        # empty passthrough
        assert fc._ensure_hemisphere_columns(pd.DataFrame()).empty

    def test_append_hemi_suffix(self):
        fc, _ = make_fc()
        assert fc._append_hemi_suffix("aMe12", "R") == "aMe12_R"
        assert fc._append_hemi_suffix("aMe12_L", "R") == "aMe12_L"
        assert fc._append_hemi_suffix(None, "U") == "Unknown_U"

    def test_hemi_code_series_priority(self):
        fc, _ = make_fc()
        df = pd.DataFrame({
            "hemisphere_code_pre": ["R", None, None],
            "hemisphere_pre": [None, "left", None],
            "instance_pre": ["x_L", "y_R", "z_L"],
        })
        codes = fc._hemi_code_series(df, "pre")
        # code column wins verbatim, then hemisphere normalization, then
        # instance suffix for the unhandled row
        assert list(codes) == ["R", "L", "L"]

    def test_hemi_code_series_default_side(self):
        fc, _ = make_fc()
        df = pd.DataFrame({"instance": ["n_R", "n_L"]})
        assert list(fc._hemi_code_series(df, "")) == ["R", "L"]

    def test_append_hemi_suffix_series(self):
        fc, _ = make_fc()
        labels = pd.Series(["A", "B_R", None])
        codes = pd.Series(["L", "R", "U"])
        out = fc._append_hemi_suffix_series(labels, codes)
        assert list(out) == ["A_L", "B_R", "Unknown_U"]

    def test_apply_hemisphere_suffix_to_neuron_df(self):
        fc, _ = make_fc(separate_hemispheres=True, hemisphere_filter="both")
        df = pd.DataFrame({
            "type": ["T1", "T2", "T3"],
            "custom_group": ["g", "g", "h"],
            "hemisphere": ["left", "right", ""],
        })
        out = fc._apply_hemisphere_suffix_to_neuron_df(df)
        assert list(out["type"]) == ["T1_L", "T2_R", "T3_U"]
        assert list(out["custom_group"]) == ["g_L", "g_R", "h_U"]

    def test_apply_hemisphere_filter_to_neuron_df(self):
        fc, _ = make_fc(separate_hemispheres=False, hemisphere_filter="left")
        df = pd.DataFrame({
            "type": ["T1", "T2", "T3"],
            "hemisphere": ["left", "right", ""],
        })
        out = fc._apply_hemisphere_suffix_to_neuron_df(df)
        # left filter keeps L and U, drops R
        assert list(out["type"]) == ["T1", "T3"]

    def test_ensure_ratio_prob_columns_pandas(self):
        fc, _ = make_fc()
        df = pd.DataFrame({
            "bodyId_pre": ["1", "2"],
            "bodyId_post": ["X", "X"],
            "weight": [10, 30],
        })
        out = fc._ensure_ratio_prob_columns(df, "bodyId_pre", "bodyId_post")
        assert list(out["connection_ratio"]) == [0.25, 0.75]
        assert out["traversal_probability"].iloc[0] == pytest.approx(0.25 / 0.3)
        assert out["traversal_probability"].iloc[1] == 1.0  # clipped

    def test_ensure_ratio_prob_columns_keeps_existing(self):
        fc, _ = make_fc()
        df = pd.DataFrame({
            "bodyId_pre": ["1"], "bodyId_post": ["X"], "weight": [10],
            "connection_ratio": [0.4], "traversal_probability": [0.9],
        })
        out = fc._ensure_ratio_prob_columns(df, "bodyId_pre", "bodyId_post")
        assert out["connection_ratio"].iloc[0] == 0.4

    def test_ensure_ratio_prob_columns_polars(self):
        fc, _ = make_fc()
        df = pl.DataFrame({
            "bodyId_pre": ["1", "2"],
            "bodyId_post": ["X", "X"],
            "weight": [10, 30],
        })
        out = fc._ensure_ratio_prob_columns(df, "bodyId_pre", "bodyId_post")
        ratios = sorted(out["connection_ratio"].to_list())
        assert ratios == [0.25, 0.75]

    def test_ensure_ratio_prob_columns_edge_cases(self):
        fc, _ = make_fc()
        assert fc._ensure_ratio_prob_columns(None, "a", "b") is None
        assert fc._ensure_ratio_prob_columns(pd.DataFrame(), "a", "b").empty
        no_weight = pd.DataFrame({"bodyId_pre": ["1"]})
        assert fc._ensure_ratio_prob_columns(no_weight, "a", "b") is no_weight

    def test_apply_hemisphere_suffix_to_conn_df(self):
        fc, _ = make_fc(separate_hemispheres=True, hemisphere_filter="both")
        df = conn_df(
            [("1", "2", 5, "A", "B")],
            extra={
                "custom_group_pre": ["g1"],
                "custom_group_post": ["g2"],
                "hemisphere_pre": ["left"],
                "hemisphere_post": ["right"],
            },
        )
        out = fc._apply_hemisphere_suffix_to_conn_df(df)
        assert out["type_pre"].iloc[0] == "A_L"
        assert out["type_post"].iloc[0] == "B_R"
        assert out["custom_group_pre"].iloc[0] == "g1_L"
        assert out["custom_group_post"].iloc[0] == "g2_R"

    def test_apply_hemisphere_filter_to_conn_df(self):
        fc, _ = make_fc(separate_hemispheres=False, hemisphere_filter="right")
        df = conn_df(
            [("1", "2", 5), ("3", "4", 6)],
            extra={
                "hemisphere_pre": ["right", "left"],
                "hemisphere_post": ["right", "right"],
            },
        )
        out = fc._apply_hemisphere_suffix_to_conn_df(df)
        assert list(out["bodyId_pre"]) == ["1"]

    def test_query_has_hemisphere_suffix(self):
        fc, _ = make_fc()
        assert fc._query_has_hemisphere_suffix("aMe12_L") is True
        assert fc._query_has_hemisphere_suffix("aMe12") is False
        assert fc._query_has_hemisphere_suffix(None) is False
        assert fc._query_has_hemisphere_suffix(["x", ["y_R"]]) is True
        assert fc._query_has_hemisphere_suffix(["x", "y"]) is False
        assert fc._query_has_hemisphere_suffix({"endswith": "_R"}) is True
        assert fc._query_has_hemisphere_suffix({"endswith": ["_X"]}) is False
        assert fc._query_has_hemisphere_suffix({"type": {"regex": "a_L"}}) is True
        assert fc._query_has_hemisphere_suffix({"type": "plain"}) is False


# =============================================================================
# export normalizers + writers
# =============================================================================

class TestExportNormalizers:
    def test_normalize_weight_list_for_export(self):
        norm = coana.FindNeuronConnection._normalize_weight_list_for_export
        out = norm([1.0, 2.5, "x", np.inf])
        assert out[0] == 1 and isinstance(out[0], int)
        assert out[1] == 2.5
        assert out[2] == "x"
        assert norm("scalar") == "scalar"

    def test_normalize_export_count_columns_pandas(self):
        FNC = coana.FindNeuronConnection
        df = pd.DataFrame({
            "weight": [5.0, 3.0],
            "total_weight": [8.0, 3.0],
            "fraction": [0.5, 0.25],          # not a count column
            "weights": [[1.0, 2.0], [3.5]],
        })
        out = FNC._normalize_export_count_columns_pandas(df)
        assert str(out["weight"].dtype) == "Int64"
        assert str(out["total_weight"].dtype) == "Int64"
        assert out["fraction"].iloc[0] == 0.5  # untouched
        assert out["weights"].iloc[0] == [1, 2]
        assert out["weights"].iloc[1] == [3.5]

    def test_normalize_export_count_columns_pandas_nonintegral(self):
        FNC = coana.FindNeuronConnection
        df = pd.DataFrame({"weight": [5.5]})
        out = FNC._normalize_export_count_columns_pandas(df)
        # non-integral values keep the original float column
        assert out["weight"].iloc[0] == 5.5

    def test_normalize_export_count_columns_polars(self):
        FNC = coana.FindNeuronConnection
        df = pl.DataFrame({
            "weight": [5.0, 3.0],
            "weights": [[1.0, 2.0], [3.0]],
        })
        out = FNC._normalize_export_count_columns_polars(df)
        assert out.schema["weight"] == pl.Int64
        assert out["weights"].to_list() == [[1, 2], [3]]
        # no matching columns -> returned unchanged
        same = pl.DataFrame({"x": [1]})
        assert FNC._normalize_export_count_columns_polars(same) is same

    def test_save_df_to_csv_polars_variants(self, tmp_path):
        fc, _ = make_fc()
        # pandas non-empty
        df = pd.DataFrame({"weight": [5.0], "type_pre": ["A"]})
        p1 = tmp_path / "pandas.csv"
        fc._save_df_to_csv_polars(df, str(p1))
        text = p1.read_text(encoding="utf-8")
        assert "weight" in text and "5" in text
        # pandas empty -> header only
        p2 = tmp_path / "empty_pd.csv"
        fc._save_df_to_csv_polars(pd.DataFrame(columns=["a", "b"]), str(p2))
        assert p2.read_text(encoding="utf-8").strip() == "a,b"
        # polars empty -> header only
        p3 = tmp_path / "empty_pl.csv"
        fc._save_df_to_csv_polars(pl.DataFrame({"a": [], "b": []}), str(p3))
        assert p3.read_text(encoding="utf-8").strip() == "a,b"
        # polars non-empty
        p4 = tmp_path / "polars.csv"
        fc._save_df_to_csv_polars(pl.DataFrame({"weight": [7]}), str(p4))
        assert "7" in p4.read_text(encoding="utf-8")
        # index=True keeps the index column
        p5 = tmp_path / "indexed.csv"
        mat = pd.DataFrame({"c1": [1]}, index=["r1"])
        fc._save_df_to_csv_polars(mat, str(p5), index=True)
        assert "r1" in p5.read_text(encoding="utf-8")
        # None is a no-op
        fc._save_df_to_csv_polars(None, str(tmp_path / "none.csv"))
        assert not (tmp_path / "none.csv").exists()

    def test_save_matrices_to_excel(self, tmp_path):
        fc, _ = make_fc()
        df = conn_df(
            [("1", "2", 5, "A", "B"), ("2", "3", 7, "B", "C")],
            extra={
                "connection_ratio": [0.5, 0.25],
                "traversal_probability": [0.9, 0.4],
                "nt_type": ["ACh", "GABA"],
            },
        )
        xlsx = tmp_path / "mat.xlsx"
        with pd.ExcelWriter(xlsx, mode="w", engine="xlsxwriter") as writer:
            fc._save_matrices_to_excel(df, writer, level="bodyId")
            fc._save_matrices_to_excel(pl.from_pandas(df), writer, level="type")
            fc._save_matrices_to_excel(pd.DataFrame(), writer)  # no-op
        sheets = pd.ExcelFile(xlsx).sheet_names
        for name in ("conn_mat_bodyId_weight", "conn_mat_bodyId_ratio",
                     "conn_mat_bodyId_prob", "conn_mat_bodyId_nt",
                     "conn_mat_type_weight", "conn_mat_type_ratio",
                     "conn_mat_type_prob", "conn_mat_type_nt"):
            assert name in sheets

    def test_save_matrices_to_csv(self, tmp_path):
        fc, _ = make_fc()
        folder = tmp_path / "mats"
        folder.mkdir()
        df = conn_df(
            [("A", "B", 5, "A", "B")],
            extra={
                "connection_ratio": [0.5],
                "traversal_probability": [0.9],
                "nt_type": ["ACh"],
            },
        )
        fc._save_matrices_to_csv(df, str(folder), level="type")
        assert (folder / "conn_mat_type_weight.csv").exists()
        assert (folder / "conn_mat_type_ratio.csv").exists()
        assert (folder / "conn_mat_type_prob.csv").exists()
        assert (folder / "conn_mat_type_nt.csv").exists()
        # empty frame writes nothing
        fc._save_matrices_to_csv(pd.DataFrame(), str(folder), level="type")


# =============================================================================
# query / export metadata helpers
# =============================================================================

class TestQueryAndExportMetadata:
    def test_query_uses_all_neurons(self):
        FNC = coana.FindNeuronConnection
        assert FNC._query_uses_all_neurons("all_neurons") is True
        assert FNC._query_uses_all_neurons("  ALL_NEURONS ") is True
        assert FNC._query_uses_all_neurons(["x", ["all_neurons"]]) is True
        assert FNC._query_uses_all_neurons({"contains": "all_neurons"}) is False
        assert FNC._query_uses_all_neurons("other") is False
        assert FNC._query_uses_all_neurons(None) is False
        assert FNC._query_uses_all_neurons(123) is False

    def test_apply_all_neurons_query(self):
        fc, logs = make_fc(dataset="test:v1")
        assert fc._apply_all_neurons_query("all_neurons", "source") == []
        assert logs  # detection notice printed
        assert fc._apply_all_neurons_query(["Mi1"], "target") == ["Mi1"]

    def test_expand_group_labels(self):
        class Mapper:
            is_empty = False

            def get_neurons_for_label(self, label, dataset, role):
                if label == "grp":
                    return [1, "2"]
                return None

        fc, _ = make_fc(label_mapper=Mapper(), dataset="test:v1")
        out = fc._expand_group_labels(["grp", "plain", "grp"], "source")
        assert out == ["1", "2", "plain"]
        # no mapper -> passthrough
        fc2, _ = make_fc(label_mapper=None, dataset="test:v1")
        assert fc2._expand_group_labels(["x"], "source") == ["x"]
        # dict query -> passthrough
        assert fc._expand_group_labels({"type": "x"}, "source") == {"type": "x"}

    def test_readable_query_name(self):
        name = coana.FindNeuronConnection._readable_query_name
        assert name({"type": "x"}, "fallback") == "fallback"
        assert name(["aMe12", "PPL1"]) == "aMe12_etc"
        assert name(["aMe12"]) == "aMe12"
        assert name("solo") == "solo"
        assert name([["a", "b"]]) == "a_etc"
        assert name([], "fb") == "fb"
        assert name([".*"], "fb") == "fb"

    def test_is_sensitive_export_key(self):
        sens = coana.FindNeuronConnection._is_sensitive_export_key
        assert sens("token") is True
        assert sens("API-Key") is True
        assert sens("neuprint token") is True
        assert sens("my_secret") is True
        assert sens("password") is True
        assert sens("dataset") is False
        assert sens(None) is False

    def test_sanitize_export_value(self):
        sanitize = coana.FindNeuronConnection._sanitize_export_value
        payload = {
            "dataset": "x",
            "token": "abc",
            "nested": {"api_key": "k", "keep": 1},
            "list": [{"password": "p"}, "ok"],
        }
        out = sanitize(payload)
        assert out == {
            "dataset": "x",
            "nested": {"keep": 1},
            "list": [{}, "ok"],
        }
        assert "token" in payload  # original never mutated

    def test_run_export_attributes(self, tmp_path):
        fc, _ = make_fc(
            dataset="test:v1",
            min_synapse_num=5,
            sourceNeurons=["aMe12"],
            targetNeurons=["PPL1"],
            source_df=pd.DataFrame({"bodyId": [1]}),
            target_df=pd.DataFrame({"bodyId": [2, 3]}),
            label_mapper=None,
            secret_token="hidden",
        )
        fc._requested_source_neurons = ["aMe12"]
        attrs = fc._run_export_attributes(path_mode="shortest")
        assert attrs["dataset"] == "test:v1"
        assert attrs["path_mode"] == "shortest"
        assert attrs["requested_source_neurons"] == ["aMe12"]
        assert attrs["resolved_source_bodyIds"] == ["1"]
        assert attrs["resolved_target_bodyIds"] == ["2", "3"]
        assert "secret_token" not in attrs
        assert "_warn_notes" not in attrs

    def test_custom_group_payload_and_parameters(self):
        class Mapper:
            is_empty = True

        fc, _ = make_fc(label_mapper=Mapper(), dataset="d")
        assert fc._custom_group_export_payload() is None

        class Mapper2:
            is_empty = False

            def get_all_std_labels(self, role):
                return ["grp"] if role == "source" else []

            def get_neurons_for_label(self, label, dataset, role):
                return [10, 20]

        fc2, _ = make_fc(
            label_mapper=Mapper2(), dataset="d",
            custom_mapping_file="/tmp/map.json",
            parameter_dict={},
            sourceNeurons=["grp"], targetNeurons=[],
            source_df=pd.DataFrame({"bodyId": [10, 20]}),
            target_df=pd.DataFrame({"bodyId": []}),
        )
        payload = fc2._custom_group_export_payload()
        assert payload["source_groups"] == [
            {"label": "grp", "members": ["10", "20"], "member_count": 2}]
        fc2._add_custom_group_parameters()
        assert "custom mapping file" in fc2.parameter_dict
        assert fc2.parameter_dict["resolved source bodyIds"] == "['10', '20']"


# =============================================================================
# hemisphere conservation + symmetry analysis
# =============================================================================

class TestHemisphereConservation:
    def test_is_symmetric_dataset(self):
        fc, _ = make_fc(dataset="male-cns:v1.0")
        assert fc._is_symmetric_dataset() is True
        fc.dataset = "hemibrain:v1.2.1"
        assert fc._is_symmetric_dataset() is False

    def test_extract_hemi_from_label(self):
        fc, _ = make_fc()
        assert fc._extract_hemi_from_label("aMe12_L") == ("aMe12", "L")
        assert fc._extract_hemi_from_label("aMe12_R (x)") == ("aMe12", "R")
        assert fc._extract_hemi_from_label("plain") == ("plain", None)

    def test_filter_hemisphere_unconserved_edges_pandas(self):
        fc, _ = make_fc()
        df = pd.DataFrame({
            "type_pre": ["A_L", "A_R", "A_L", "D", "F_U"],
            "type_post": ["B_L", "B_R", "C_R", "E", "G_L"],
            "weight": [10, 12, 5, 7, 3],
        })
        kept, removed = fc._filter_hemisphere_unconserved_edges(df)
        assert set(zip(kept["type_pre"], kept["type_post"])) == {
            ("A_L", "B_L"), ("A_R", "B_R"), ("D", "E"), ("F_U", "G_L")}
        assert set(zip(removed["type_pre"], removed["type_post"])) == {("A_L", "C_R")}

    def test_filter_hemisphere_unconserved_edges_polars_and_empty(self):
        fc, _ = make_fc()
        df = pl.DataFrame({
            "type_pre": ["A_L", "A_R"],
            "type_post": ["B_L", "B_R"],
            "weight": [1, 2],
        })
        kept, removed = fc._filter_hemisphere_unconserved_edges(df)
        assert isinstance(kept, pl.DataFrame)
        assert removed is None and len(kept) == 2
        empty_pd = pd.DataFrame(columns=["type_pre", "type_post", "weight"])
        out, none = fc._filter_hemisphere_unconserved_edges(empty_pd)
        assert out is empty_pd and none is None

    def test_count_hemisphere_from_df(self):
        fc, _ = make_fc()
        assert fc._count_hemisphere_from_df(None) == {"L": 0, "R": 0, "U": 0}
        df = pd.DataFrame({"hemisphere_code": ["L", "r", "x"]})
        assert fc._count_hemisphere_from_df(df) == {"L": 1, "R": 1, "U": 1}
        df = pd.DataFrame({"Soma side": ["right", "left", None]})
        assert fc._count_hemisphere_from_df(df) == {"L": 1, "R": 1, "U": 1}
        df = pd.DataFrame({"type": ["A_L", "B_R", "C"]})
        assert fc._count_hemisphere_from_df(df) == {"L": 1, "R": 1, "U": 1}

    def test_run_hemisphere_symmetry_analysis(self, tmp_path):
        fc, _ = make_fc(
            symmetry_analysis=True,
            dataset="male-cns:v1.0",
            allpath_folder=str(tmp_path),
            min_synapse_num=1,
            source_df=pd.DataFrame({"type": ["A_L", "A_R"],
                                    "hemisphere_code": ["L", "R"]}),
            target_df=pd.DataFrame({"type": ["B_L"],
                                    "hemisphere_code": ["L"]}),
        )
        conn_types = pd.DataFrame({
            "type_pre": ["A_L", "A_R", "A_L", "D"],
            "type_post": ["B_L", "B_R", "C_R", "E"],
            "weight": [10, 20, 5, 7],
        })
        paths = pd.DataFrame({
            "path_block": ["A_L -> B_L", "A_R -> B_R", "X_L -> Y_L",
                            "A_L -> C_R"],
        })
        fc._run_hemisphere_symmetry_analysis(conn_types, paths_df=paths)

        sym_dir = tmp_path / "hemisphere_symmetry"
        import json
        summary = json.loads((sym_dir / "symmetry_summary.json").read_text())
        assert summary["ipsi"]["conserved"] == 1
        assert summary["ipsi"]["union"] == 1
        assert summary["ipsi"]["jaccard"] == pytest.approx(1.0)
        assert summary["contra"]["union"] == 1
        assert summary["contra"]["conserved"] == 0
        assert summary["paths"]["total_L"] == 2
        assert summary["paths"]["total_R"] == 1
        assert summary["paths"]["conserved"] == 1
        assert summary["paths"]["unconserved_L_only"] == 1
        assert summary["hemisphere_counts"]["source"] == {"L": 1, "R": 1, "U": 0}

        for name in ("symmetry_ipsi.csv", "symmetry_contra.csv",
                     "conserved_edges.csv", "unconserved_edges.csv",
                     "pairwise_strength.csv", "type_counts_by_role.csv",
                     "conserved_paths.csv", "unconserved_paths.csv"):
            assert (sym_dir / name).exists(), name
        ipsi = pd.read_csv(sym_dir / "symmetry_ipsi.csv")
        row = ipsi.iloc[0]
        assert row["weight_L"] == 10 and row["weight_R"] == 20
        assert row["ratio"] == pytest.approx(0.5)

    def test_run_hemisphere_symmetry_analysis_skips(self, tmp_path):
        base = dict(
            symmetry_analysis=True, dataset="male-cns:v1.0",
            allpath_folder=str(tmp_path), min_synapse_num=1,
            source_df=pd.DataFrame(), target_df=pd.DataFrame(),
        )
        # disabled flag
        fc, _ = make_fc(**{**base, "symmetry_analysis": False})
        fc._run_hemisphere_symmetry_analysis(pd.DataFrame(
            {"type_pre": ["A_L"], "type_post": ["B_L"], "weight": [1]}))
        assert not (tmp_path / "hemisphere_symmetry").exists()
        # non-symmetric dataset
        fc, _ = make_fc(**{**base, "dataset": "hemibrain:v1.2.1"})
        fc._run_hemisphere_symmetry_analysis(pd.DataFrame(
            {"type_pre": ["A_L"], "type_post": ["B_L"], "weight": [1]}))
        assert not (tmp_path / "hemisphere_symmetry").exists()
        # empty frame / missing columns
        fc, _ = make_fc(**base)
        fc._run_hemisphere_symmetry_analysis(pd.DataFrame())
        fc._run_hemisphere_symmetry_analysis(pd.DataFrame({"type_pre": ["A"]}))
        assert not (tmp_path / "hemisphere_symmetry").exists()


# =============================================================================
# path helpers, warning notes, enrollment
# =============================================================================

class TestPathHelpers:
    def test_derive_type_paths_from_bodyid_paths(self):
        fc, _ = make_fc()
        label = {"1": "S", "2": "M", "3": "T", "4": "T"}
        kept_edges = {("S", "M"), ("M", "T")}
        paths = [["1", "2", "3"],      # valid
                 ["1", "2", "4"],      # duplicate type sequence
                 ["2", "3"],           # starts outside source set
                 ["1", "3"]]           # missing type edge S->T
        out = fc._derive_type_paths_from_bodyid_paths(
            paths, label.get, kept_edges, ["S"], ["T"])
        assert out == [["S", "M", "T"]]

    def test_keep_shortest_bodyid_paths(self):
        keep = coana.FindNeuronConnection._keep_shortest_bodyid_paths
        paths = [
            ["a", "b", "t"],   # pair (a,t) length 2
            ["a", "x", "y", "t"],  # longer alternative for (a,t)
            ["b", "t"],        # pair (b,t) length 1
            [],                # dropped
        ]
        out = keep(paths)
        assert ["a", "b", "t"] in out
        assert ["b", "t"] in out
        assert ["a", "x", "y", "t"] not in out
        assert len(out) == 2

    def test_build_bodyid_type_map(self):
        fc, _ = make_fc(
            all_connections_filtered=[conn_df([("1", "2", 5, "TA", "TB")])],
            source_df=pd.DataFrame({"bodyId": ["1", "3"], "type": ["TA", ""]}),
            target_df=pd.DataFrame({"bodyId": ["2"], "type": ["TB"]}),
        )
        type_map = fc._build_bodyid_type_map()
        assert type_map == {"1": "TA", "2": "TB"}

    def test_extract_nodes_from_path_graph(self):
        fc, _ = make_fc()
        assert fc._extract_nodes_from_path_graph(None) == []
        assert fc._extract_nodes_from_path_graph(pd.DataFrame()) == []
        nodes = fc._extract_nodes_from_path_graph(conn_df([("1", "2", 3)]))
        assert set(nodes) == {"1", "2"}
        nodes_pl = fc._extract_nodes_from_path_graph(
            pl.DataFrame({"bodyId_pre": ["1"], "bodyId_post": ["2"]}))
        assert set(nodes_pl) == {"1", "2"}

    def test_reset_temp_columns(self):
        fc, _ = make_fc(
            source_df=pd.DataFrame({"bodyId": ["1"], "isInPath": [True]}),
            target_df=pd.DataFrame({"bodyId": ["2"], "Checked": [True],
                                    "Layer": [1]}),
        )
        fc._reset_temp_columns()
        assert "isInPath" not in fc.source_df.columns
        assert "Checked" not in fc.target_df.columns
        assert fc._edgeN_limit_reached is False
        assert fc._shortest_target_hop_limits == {}

    def test_record_search_priority_warnings(self):
        fc, _ = make_fc()
        fc._record_search_priority_warnings("source", [
            {"matched_column": "type", "search_term": "Mi1", "match_count": 3},
            {"matched_column": "hemiline", "search_term": "Hb", "match_count": 12},
            {},
        ])
        assert len(fc._warn_notes) == 1
        assert "hemiline" in fc._warn_notes[0]

    def test_fetch_direct_connections_for_nodes(self):
        fc, _ = make_fc(min_synapse_num=1, min_ratio=0.0,
                        min_traversal_probability=0.0)
        table = conn_df([("1", "2", 5), ("1", "9", 5)])
        fc._fetch_connections_with_cache = (
            lambda upstream_bodyIds=None, downstream_bodyIds=None,
            min_weight=None, min_conn_ratio=None, min_traversal_prob=None:
            table.copy()
        )
        assert fc._fetch_direct_connections_for_nodes([]).empty
        out = fc._fetch_direct_connections_for_nodes(["1", "2"])
        assert list(out["bodyId_post"]) == ["2"]

    def test_write_user_warning_notes(self, tmp_path):
        fc, _ = make_fc(
            min_ratio=0.1,
            min_traversal_probability=0.05,
            keyword_in_path_to_remove=["GABA"],
            edgeN_limit=100,
            _edgeN_limit_reached=True,
            max_interlayer=5,
            _depth_cap_reached=True,
            _shortest_backward_active=True,
            _shortest_scope_limited=True,
            separate_hemispheres=True,
            hemisphere_filter="left",
            keep_only_hemisphere_conserved_connections=True,
            symmetry_analysis=True,
            find_reciprocal=True,
            skip_bodyId=True,
            pathN_to_show=3,
            cache_only=True,
        )
        folder = tmp_path / "run"
        folder.mkdir()
        fc._write_user_warning_notes(str(folder))
        text = (folder / "user_warning_notes.txt").read_text(encoding="utf-8")
        for token in ("min_ratio=0.1", "keyword_in_path_to_remove",
                      "edge limit", "depth", "hemisphere", "symmetry",
                      "reciprocal", "skip_bodyId", "pathN_to_show",
                      "cache_only"):
            assert token in text

    def test_write_user_warning_notes_empty(self, tmp_path):
        fc, _ = make_fc(min_ratio=0.0, min_traversal_probability=0.0,
                        hemisphere_filter="both")
        fc._write_user_warning_notes(str(tmp_path))
        assert not (tmp_path / "user_warning_notes.txt").exists()

    def test_save_path_neuron_enrollment(self, tmp_path):
        fc, _ = make_fc(
            source_df=pd.DataFrame({"bodyId": ["1"], "type": ["S"]}),
            target_df=pd.DataFrame({"bodyId": ["2"], "type": ["T"]}),
        )
        src_path, tgt_path = fc._save_path_neuron_enrollment(str(tmp_path))
        src = pd.read_csv(src_path)
        tgt = pd.read_csv(tgt_path)
        assert "isInPath" in src.columns
        assert {"Checked", "Layer"}.issubset(tgt.columns)

    def test_record_viz_edge_trim(self):
        fc, _ = make_fc(_edgeN_limit_reached=False)
        from types import SimpleNamespace
        fc._record_viz_edge_trim(SimpleNamespace(edge_limit_trimmed=True))
        assert fc._edgeN_limit_reached is True
        fc._record_viz_edge_trim(SimpleNamespace(edge_limit_trimmed=False))
        assert fc._edgeN_limit_reached is True

    def test_is_empty_df(self):
        empty = coana.FindNeuronConnection._is_empty_df
        assert empty(None) is True
        assert empty(pd.DataFrame()) is True
        assert empty(pl.DataFrame()) is True
        assert empty(pd.DataFrame({"a": [1]})) is False


# =============================================================================
# FindDirectConnections offline pipeline
# =============================================================================

def _make_direct_fc(monkeypatch, tmp_path, fetch_rows, *, types,
                    keep_conserved=False, separate=False, target_ids=None):
    """Wire a FindNeuronConnection for an offline FindDirectConnections run.

    fetch_rows: list of (pre, post, weight); types: {bodyId: type}
    """
    def fake_fetch_path_connections(upstream_bodyIds=None,
                                    downstream_bodyIds=None, **kwargs):
        rows = [(str(u), str(v), w) for (u, v, w) in fetch_rows]
        if not rows:
            return pd.DataFrame(columns=["bodyId_pre", "bodyId_post",
                                         "weight", "type_pre", "type_post"])
        return pd.DataFrame({
            "bodyId_pre": [r[0] for r in rows],
            "bodyId_post": [r[1] for r in rows],
            "weight": [r[2] for r in rows],
            "type_pre": [types[r[0]] for r in rows],
            "type_post": [types[r[1]] for r in rows],
        })

    def fake_enrich(conn_df, traversal_probability_threshold=0, dataset=None,
                    script_path=None, aggregate_method="product",
                    label_mapper=None, global_incoming_weights=None,
                    separate_hemispheres=False,
                    global_incoming_body_weights=None, **kwargs):
        conn_e = conn_df.copy()
        conn_e["connection_ratio"] = 0.5
        conn_e["traversal_probability"] = 0.5
        conn_t = (conn_e.groupby(["type_pre", "type_post"], as_index=False)
                  ["weight"].sum())
        conn_t["connection_ratio"] = 0.5
        conn_t["traversal_probability"] = 0.5
        return conn_e, conn_t, None

    monkeypatch.setattr(coana.sv, "EnrichConnectionTable", fake_enrich)

    source_ids = sorted({str(u) for (u, _, _) in fetch_rows})[:2] or ["S"]
    if target_ids is None:
        target_ids = sorted({str(v) for (_, v, _) in fetch_rows})[:2] or ["T"]
    fc, logs = make_fc(
        saveas="",
        output_dir=str(tmp_path),
        source_fname="src",
        target_fname="tgt",
        dataset="test:v1",
        max_interlayer=0,
        min_synapse_num=1,
        min_ratio=0.0,
        min_traversal_probability=0.0,
        parameter_dict={"k": "v"},
        parameter_df=pd.DataFrame({"p": [1]}),
        source_df=pd.DataFrame({"bodyId": source_ids,
                                "type": [types.get(i, "T" + str(i)) for i in source_ids]}),
        target_df=pd.DataFrame({"bodyId": target_ids,
                                "type": [types.get(i, "T" + str(i)) for i in target_ids]}),
        aggregate_method="product",
        label_mapper=None,
        separate_hemispheres=separate,
        keep_only_hemisphere_conserved_connections=keep_conserved,
        output_format="csv",
        largeTargetSet=False,
        script_path=str(PROJECT_ROOT),
    )
    fc._fetch_path_connections = fake_fetch_path_connections
    fc._fetch_total_incoming_weight_by_type = lambda *a, **k: None
    fc._fetch_total_incoming_weight = lambda *a, **k: None
    fc.VisualizeDirectConnections_simple = lambda: None
    return fc, logs


class TestFindDirectConnections:
    def test_full_run_csv_outputs(self, monkeypatch, tmp_path):
        types = {"S": "TS", "A": "TA", "X": "TX"}
        fc, _ = _make_direct_fc(
            monkeypatch, tmp_path,
            [("S", "A", 10), ("S", "X", 50)],  # X outside target set
            types=types, target_ids=["A"])
        result = fc.FindDirectConnections()
        assert result == 0

        folder = Path(fc.direct_folder)
        assert folder.name.startswith("finddirect_")
        params = (folder / "parameters.txt").read_text()
        assert "k: v" in params
        # enrollment files at run root
        assert (folder / "source_neurons.csv").exists()
        assert (folder / "target_neurons.csv").exists()
        details = folder / "data_details"
        base = "src_to_tgt_info_snp1"
        for suffix in ("_parameters.csv", "_source_info.csv",
                       "_target_info.csv", "_source_in_connection.csv",
                       "_target_in_connection.csv",
                       "_connection_groupby_type.csv",
                       "_connectionMatrix_type.csv", "_connMat_type_full.csv",
                       "_transmissionMat_type.csv",
                       "_connectionRatioMat_type.csv",
                       "_ratioMat_type_full.csv"):
            assert (details / (base + suffix)).exists(), suffix
        body_csv = folder / "src_to_tgt_bodyId_connections_snp1.csv"
        assert body_csv.exists()
        body = pd.read_csv(body_csv)
        # X is not a target: only S->A survives
        assert set(zip(body["bodyId_pre"].astype(str),
                       body["bodyId_post"].astype(str))) == {("S", "A")}
        assert (folder / "src_to_tgt_connectionMatrix_bodyId.csv").exists()
        # target enrollment marks A checked at layer 1
        tgt = pd.read_csv(folder / "target_neurons.csv")
        assert tgt.loc[tgt["bodyId"].astype(str) == "A", "Layer"].iloc[0] == 1

    def test_empty_result_early_return(self, monkeypatch, tmp_path):
        fc, _ = _make_direct_fc(monkeypatch, tmp_path, [], types={"S": "TS"})
        result = fc.FindDirectConnections()
        assert result is None  # early return path
        folder = Path(fc.direct_folder)
        assert (folder / "source_neurons.csv").exists()
        assert not (folder / "data_details").exists()

    def test_hemisphere_conserved_filtering(self, monkeypatch, tmp_path):
        # TS_L->TA_L mirrors TS_R->TA_R (conserved pair); TS_L->TB_L has no
        # TS_R->TB_R counterpart (unconserved -> exported).
        types = {"S1": "TS_L", "S2": "TS_R", "A": "TA_L", "B": "TA_R",
                 "C": "TB_L"}
        rows = [("S1", "A", 10), ("S2", "B", 9), ("S1", "C", 8)]
        fc, _ = _make_direct_fc(
            monkeypatch, tmp_path, rows, types=types,
            keep_conserved=True, separate=True)
        fc.source_df = pd.DataFrame({"bodyId": ["S1", "S2"],
                                     "type": ["TS_L", "TS_R"]})
        fc.target_df = pd.DataFrame({"bodyId": ["A", "B", "C"],
                                     "type": ["TA_L", "TA_R", "TB_L"]})
        fc.FindDirectConnections()
        details = Path(fc.direct_folder) / "data_details"
        type_csv = details / "src_to_tgt_info_snp1_connection_groupby_type.csv"
        kept = pd.read_csv(type_csv)
        unconserved = details / "hemisphere_unconserved_edges.csv"
        assert unconserved.exists()
        assert len(kept) == 2
        assert len(pd.read_csv(unconserved)) == 1
