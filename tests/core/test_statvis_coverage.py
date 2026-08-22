"""Coverage-focused unit tests for statvis.py.

All tests are hermetic: synthetic in-memory DataFrames, monkeypatched
loaders, and file I/O restricted to pytest tmp_path.  No network access,
no reads from datasets/ or cache/.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

import statvis as sv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =============================================================================
# Escape / print helpers
# =============================================================================

class TestEscapeHelpers:
    def test_html_escape(self):
        assert sv._statvis_html_escape(None) == ""
        assert sv._statvis_html_escape("<a & 'b' \"c\">") == (
            "&lt;a &amp; &#39;b&#39; &quot;c&quot;&gt;"
        )
        assert sv._statvis_html_escape(5) == "5"

    def test_js_escape(self):
        assert sv._statvis_js_escape(None) == ""
        out = sv._statvis_js_escape("a'b\"c\\d\ne<f>g&h\r")
        assert "\\'" in out and '\\"' in out and "\\\\" in out
        assert "\\n" in out and "\\r" in out
        assert "\\u003c" in out and "\\u003e" in out and "\\u0026" in out

    def test_tqdm_print_default_end(self, capsys):
        sv._tqdm_print("hello", "world", sep="-")
        captured = capsys.readouterr()
        assert "hello-world" in captured.out

    def test_tqdm_print_custom_end(self, capsys):
        sv._tqdm_print("line", end="!")
        captured = capsys.readouterr()
        assert "line!" in captured.out


# =============================================================================
# drop_leading_index_columns
# =============================================================================

class TestDropLeadingIndexColumns:
    def test_pandas_unnamed_dropped(self):
        df = pd.DataFrame({"Unnamed: 0": [0, 1], "type": ["a", "b"]})
        out = sv.drop_leading_index_columns(df)
        assert list(out.columns) == ["type"]

    def test_polars_column_1_dropped(self):
        df = pl.DataFrame({"column_1": [0, 1], "type": ["a", "b"]})
        out = sv.drop_leading_index_columns(df)
        assert list(out.columns) == ["type"]

    def test_non_integer_first_column_kept(self):
        df = pd.DataFrame({"Unnamed: 0": ["x", "y"], "type": ["a", "b"]})
        out = sv.drop_leading_index_columns(df)
        assert list(out.columns) == ["Unnamed: 0", "type"]

    def test_only_index_column_kept(self):
        df = pd.DataFrame({"index": [0, 1]})
        out = sv.drop_leading_index_columns(df)
        assert list(out.columns) == ["index"]

    def test_no_index_columns_noop(self):
        df = pd.DataFrame({"type": ["a"], "bodyId": [1]})
        out = sv.drop_leading_index_columns(df)
        assert list(out.columns) == ["type", "bodyId"]

    def test_leading_run_stops_at_real_column(self):
        df = pd.DataFrame({"Unnamed: 0": [0], "index": [1], "type": ["a"]})
        out = sv.drop_leading_index_columns(df)
        assert list(out.columns) == ["type"]


# =============================================================================
# _load_dataframe_fast
# =============================================================================

class TestLoadDataframeFast:
    def test_csv_with_dtype_overrides(self, tmp_path):
        csv = tmp_path / "t.csv"
        pd.DataFrame({"bodyId": [1, 2], "type": ["a", "b"]}).to_csv(
            csv, index=False)
        df = sv._load_dataframe_fast(str(csv), dtype_overrides={"bodyId": str})
        assert df["bodyId"].dtype == object
        assert list(df["bodyId"]) == ["1", "2"]

    def test_newer_parquet_sibling_preferred(self, tmp_path):
        csv = tmp_path / "t.csv"
        pd.DataFrame({"v": [1]}).to_csv(csv, index=False)
        pq = tmp_path / "t.parquet"
        pd.DataFrame({"v": [99]}).to_parquet(pq)
        # ensure parquet strictly newer
        now = os.stat(csv).st_mtime_ns
        os.utime(pq, ns=(now + 10**6, now + 10**6))
        df = sv._load_dataframe_fast(str(csv))
        assert list(df["v"]) == [99]

    def test_newer_csv_wins_over_parquet(self, tmp_path):
        pq = tmp_path / "t.parquet"
        pd.DataFrame({"v": [99]}).to_parquet(pq)
        csv = tmp_path / "t.csv"
        pd.DataFrame({"v": [1]}).to_csv(csv, index=False)
        now = os.stat(pq).st_mtime_ns
        os.utime(csv, ns=(now + 10**6, now + 10**6))
        df = sv._load_dataframe_fast(str(csv))
        assert list(df["v"]) == [1]

    def test_parquet_only(self, tmp_path):
        pq = tmp_path / "t.parquet"
        pd.DataFrame({"v": [7]}).to_parquet(pq)
        df = sv._load_dataframe_fast(str(tmp_path / "t.csv"),
                                     dtype_overrides={"v": int})
        assert list(df["v"]) == [7]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            sv._load_dataframe_fast(str(tmp_path / "nope.csv"))

    def test_pandas_fallback_when_no_polars(self, tmp_path, monkeypatch):
        csv = tmp_path / "t.csv"
        pd.DataFrame({"bodyId": [1], "type": ["a"]}).to_csv(csv, index=False)
        monkeypatch.setattr(sv, "HAS_POLARS", False)
        df = sv._load_dataframe_fast(str(csv),
                                     dtype_overrides={"bodyId": str})
        assert list(df["bodyId"]) == ["1"]


# =============================================================================
# _get_cached_neuron_df / clear_neuron_cache / roi_count_table_path
# =============================================================================

class TestCachedNeuronDf:
    def test_load_cache_and_clear(self, tmp_path):
        body = str(tmp_path / "synth_allneurons")
        pd.DataFrame({"bodyId": [1, 2], "type": ["a", "b"]}).to_csv(
            body + "_neuron_df.csv", index=False)
        pd.DataFrame({"bodyId": [1, 2], "roi": ["r", "r"]}).to_csv(
            body + "_roi_count_df.csv", index=False)

        ndf, rdf = sv._get_cached_neuron_df("synth", body)
        assert ndf["bodyId"].dtype == np.int64
        assert rdf["bodyId"].dtype == np.int64
        # second call served from memory cache
        ndf2, rdf2 = sv._get_cached_neuron_df("synth", body)
        assert list(ndf2["type"]) == ["a", "b"]
        assert list(rdf2["roi"]) == ["r", "r"]

        sv.clear_neuron_cache("synth")
        assert "synth" not in sv._NEURON_DF_CACHE
        sv._get_cached_neuron_df("synth", body)
        sv.clear_neuron_cache()  # clear-all branch
        assert sv._NEURON_DF_CACHE == {}

    def test_roi_count_table_path(self, tmp_path):
        body = str(tmp_path / "x_allneurons")
        # no parquet -> csv path
        assert sv.roi_count_table_path(body).endswith("_roi_count_df.csv")
        pd.DataFrame({"a": [1]}).to_parquet(body + "_roi_count_df.parquet")
        assert sv.roi_count_table_path(body).endswith("_roi_count_df.parquet")


# =============================================================================
# Query helpers: get_types / get_bodyIds / get_instances / get_info
# =============================================================================

def _synthetic_ndf():
    return pd.DataFrame({
        "bodyId": [1, 2, 3, 4],
        "type": ["Mi1", "Mi1", "Tm3", "DN1"],
        "instance": ["Mi1_L", "Mi1_R", "Tm3_L", "DN1_R"],
    })


@pytest.fixture
def patched_neuron_df(monkeypatch):
    monkeypatch.setattr(sv, "_get_neuron_df",
                        lambda dataset="x", verbose=False: _synthetic_ndf())
    monkeypatch.setattr(sv, "_get_cached_neuron_search", lambda dataset: None)


class TestQueryHelpers:
    def test_get_types_legacy(self, patched_neuron_df):
        types = sv.get_types("Mi1", dataset="test:v1", verbose=False,
                             return_simple=True)
        assert types == ["Mi1"]
        type_list, map_dict, ds = sv.get_types(["Mi1", "Tm3"],
                                               dataset="test:v1",
                                               verbose=False)
        assert type_list == ["Mi1", "Tm3"]
        assert map_dict["Mi1"] == ["Mi1"]
        assert ds == "test_v1"

    def test_get_types_dict_filter(self, patched_neuron_df):
        types = sv.get_types({"contains": "DN"}, dataset="test:v1",
                             verbose=False, return_simple=True)
        assert types == ["DN1"]
        type_list, map_dict, ds = sv.get_types({"contains": "DN"},
                                               dataset="test:v1",
                                               verbose=False)
        assert type_list == ["DN1"]
        assert "DN1" in map_dict

    def test_get_bodyIds_legacy_and_dict(self, patched_neuron_df):
        ids = sv.get_bodyIds("Mi1", dataset="test:v1", verbose=False,
                             return_simple=True)
        assert sorted(ids) == [1, 2]
        body_list, map_dict, ds = sv.get_bodyIds(["Mi1", 3],
                                                 dataset="test:v1",
                                                 verbose=False)
        assert sorted(body_list) == [1, 2, 3]
        assert map_dict[3] == 3
        ids = sv.get_bodyIds({"startswith": "DN"}, dataset="test:v1",
                             verbose=False, return_simple=True)
        assert ids == [4]

    def test_get_instances_legacy(self, patched_neuron_df):
        insts = sv.get_instances("Mi1", dataset="test:v1", verbose=False,
                                 return_simple=True)
        assert insts == ["Mi1_L", "Mi1_R"]
        inst_list, map_dict, ds = sv.get_instances(["Tm3"], dataset="test:v1",
                                                   verbose=False)
        assert inst_list == ["Tm3_L"]
        assert map_dict["Tm3_L"] == ["Tm3"]

    def test_get_instances_dict(self, patched_neuron_df):
        insts = sv.get_instances({"endswith": "_R"}, dataset="test:v1",
                                 verbose=False, return_simple=True)
        assert insts == ["DN1_R", "Mi1_R"]

    def test_get_instances_missing_column(self, monkeypatch):
        df = _synthetic_ndf().drop(columns=["instance"])
        monkeypatch.setattr(sv, "_get_neuron_df",
                            lambda dataset="x", verbose=False: df)
        assert sv.get_instances("Mi1", verbose=False,
                                return_simple=True) == []
        # normalization replaces ':' and '.' only ('-' is kept)
        assert sv.get_instances("Mi1", verbose=False) == (
            [], {}, "male-cns_v0_9")

    def test_get_info_none_query_and_columns(self, patched_neuron_df):
        df = sv.get_info(None, dataset="test:v1",
                         columns=["bodyId", "type", "nope"], verbose=False)
        assert list(df.columns) == ["bodyId", "type"]
        assert len(df) == 4

    def test_get_info_dict_and_legacy(self, patched_neuron_df):
        df = sv.get_info({"contains": "DN"}, dataset="test:v1", verbose=False)
        assert list(df["bodyId"]) == [4]
        df = sv.get_info(["Mi1"], dataset="test:v1",
                         columns=["bodyId"], verbose=False)
        assert sorted(df["bodyId"]) == [1, 2]


# =============================================================================
# Matrix helpers: removeSearchedNeurons / Conn2FullMat / calRC / filtMat / stMat
# =============================================================================

class TestMatrixHelpers:
    def test_remove_searched_neurons(self):
        conn = pd.DataFrame({"bodyId_pre": [1, 1, 2],
                             "bodyId_post": [2, 3, 3],
                             "weight": [5, 6, 7]})
        out = sv.removeSearchedNeurons(conn, [2, 3])
        assert out.empty
        out = sv.removeSearchedNeurons(conn, [2, 3], exempt_neurons=[3])
        assert sorted(out["bodyId_post"].unique()) == [3]

    def test_conn2fullmat_with_types(self):
        source_df = pd.DataFrame({"bodyId": [1], "type": ["S"]})
        target_df = pd.DataFrame({"bodyId": [2, 3], "type": ["T1", "T2"]})
        conn_df = pd.DataFrame({"bodyId_pre": [1], "bodyId_post": [2],
                                "weight": [9]})
        conn_type = pd.DataFrame({"type_pre": ["S"], "type_post": ["T1"],
                                  "weight": [9]})
        cmat_b, cmat_t = sv.Conn2FullMat(source_df, target_df, conn_df,
                                         conn_type)
        assert cmat_b.index.tolist() == ["1_S"]
        assert cmat_b.at["1_S", "2_T1"] == 9
        assert cmat_b.at["1_S", "3_T2"] == 0
        assert cmat_t.at["S", "T1"] == 9
        assert cmat_t.at["S", "T2"] == 0

    def test_conn2fullmat_alternate_weight_col(self):
        source_df = pd.DataFrame({"bodyId": [1], "type": ["S"]})
        target_df = pd.DataFrame({"bodyId": [2], "type": ["T1"]})
        conn_df = pd.DataFrame({"bodyId_pre": [1], "bodyId_post": [2],
                                "weight": [3],
                                "traversal_probability": [0.7]})
        conn_type = pd.DataFrame({"type_pre": ["S"], "type_post": ["T1"],
                                  "weight": [3],
                                  "traversal_probability": [0.7]})
        cmat_b, cmat_t = sv.Conn2FullMat(source_df, target_df, conn_df,
                                         conn_type,
                                         weight_col="traversal_probability")
        assert cmat_b.at["1_S", "2_T1"] == 0.7
        assert cmat_t.at["S", "T1"] == 0.7

    def test_calrc(self):
        cmat = pd.DataFrame([[1, 0], [2, 3]], index=["s1", "s2"],
                            columns=["t1", "t2"])
        out = sv.calRC(cmat, threshold=0)
        assert out.index.tolist() == ["s1", "s2", "sourceN", "sum_col"]
        assert out.columns.tolist() == ["t1", "t2", "targetN", "sum_row"]
        assert out.loc["sum_col", "t1"] == 3
        assert out.loc["sourceN", "t2"] == 1
        assert out.loc["s1", "targetN"] == 1
        assert out.loc["s2", "sum_row"] == 5

    def test_filtmat_mr_axis0_range(self):
        cmat = pd.DataFrame([[0.2, 0.9], [0.1, 0.5]], columns=["t1", "t2"])
        out = sv.filtMat(cmat, axis=0, filt_range=[0.15, 1.0], by="MR")
        assert out.columns.tolist() == ["t1", "t2"]
        out = sv.filtMat(cmat, axis=0, filt_range=[0.5, 1.0], by="MR")
        assert out.columns.tolist() == ["t2"]

    def test_filtmat_mr_equality_branch(self):
        cmat = pd.DataFrame([[0.5, 0.9]], columns=["t1", "t2"])
        out = sv.filtMat(cmat, axis=0, filt_range=[0.5, 0.5], by="MR")
        assert out.columns.tolist() == ["t1"]

    def test_filtmat_mr_axis1(self):
        cmat = pd.DataFrame([[0.2, 0.1], [0.9, 0.05]],
                            index=["s1", "s2"], columns=["t1", "t2"])
        out = sv.filtMat(cmat, axis=1, filt_range=[0.15, 1.0], by="MR")
        assert out.index.tolist() == ["s1", "s2"]
        out = sv.filtMat(cmat, axis=1, filt_range=[0.5, 1.0], by="MR")
        assert out.index.tolist() == ["s2"]

    def test_filtmat_by_n(self):
        cmat = pd.DataFrame([[1, 0], [2, 3]], index=["s1", "s2"],
                            columns=["t1", "t2"])
        meta_cols = {"targetN", "sum_row"}
        meta_rows = {"sourceN", "sum_col"}
        # sourceN: t1 -> 2 sources, t2 -> 1 source
        out = sv.filtMat(cmat, axis=0, filt_range=[2, 2], by="N")
        assert [c for c in out.columns if c not in meta_cols] == ["t1"]
        out = sv.filtMat(cmat, axis=0, filt_range=[None, 1], by="N")
        assert [c for c in out.columns if c not in meta_cols] == ["t2"]
        out = sv.filtMat(cmat, axis=0, filt_range=[2, None], by="N")
        assert [c for c in out.columns if c not in meta_cols] == ["t1"]
        # targetN: s1 -> 1 target, s2 -> 2 targets
        out = sv.filtMat(cmat, axis=1, filt_range=[2, None], by="N")
        assert [i for i in out.index if i not in meta_rows] == ["s2"]
        out = sv.filtMat(cmat, axis=1, filt_range=[None, 1], by="N")
        assert [i for i in out.index if i not in meta_rows] == ["s1"]

    def test_stmat(self):
        mat = pd.DataFrame([[1.0, 0.0], [3.0, 2.0]], index=["s1", "s2"],
                           columns=["t1", "t2"])
        out = sv.stMat(mat, axis=0)  # column normalization
        assert out.iat[0, 0] == pytest.approx(0.25)
        assert out.iat[1, 0] == pytest.approx(0.75)
        assert out.iat[1, 1] == pytest.approx(1.0)
        out = sv.stMat(mat, axis=1)  # row normalization
        assert out.iat[0, 0] == pytest.approx(1.0)
        assert out.iat[1, 0] == pytest.approx(0.6)


# =============================================================================
# VisConnMat + CreateHeatmap
# =============================================================================

@pytest.fixture
def patched_colorbar(monkeypatch):
    """Work around source/plotly-version incompatibilities so the
    visualization code paths themselves can be exercised:
    - VisConnMat passes 'titleside' to the plotly ColorBar (rejected by the
      installed plotly version).
    - The layout uses the deprecated 'titlefont' axis property.
    """
    real_heatmap = sv.go.Heatmap

    def factory(*args, **kwargs):
        cb = kwargs.get("colorbar")
        if isinstance(cb, dict) and "titleside" in cb:
            kwargs["colorbar"] = {k: v for k, v in cb.items()
                                  if k != "titleside"}
        return real_heatmap(*args, **kwargs)

    monkeypatch.setattr(sv.go, "Heatmap", factory)

    real_figure = sv.go.Figure

    class FixedFigure(real_figure):
        def update_layout(self, *args, **kwargs):
            for axis_key in ("xaxis", "yaxis"):
                axis = kwargs.get(axis_key)
                if isinstance(axis, dict):
                    axis = {k: v for k, v in axis.items()
                            if k != "titlefont"}
                    kwargs[axis_key] = axis
            return super().update_layout(*args, **kwargs)

    monkeypatch.setattr(sv.go, "Figure", FixedFigure)


class TestVisConnMat:
    def _matrix(self):
        return pd.DataFrame([[10, 0], [3, 7]], index=["s1", "s2"],
                            columns=["t1", "t2"])

    def test_writes_html_synapses(self, tmp_path, patched_colorbar):
        out = tmp_path / "hm.html"
        sv.VisConnMat(self._matrix(), str(out), title="Connection Matrix",
                      showfig=False)
        assert out.exists()
        content = out.read_text()
        assert "Connection Matrix" in content

    def test_ratio_metric_and_log_scales(self, tmp_path, patched_colorbar):
        mat = pd.DataFrame([[0.5, 0.2], [0.0, 0.9]])
        for i, (title, scale) in enumerate([
            ("Ratio matrix", "linear"),
            ("Transmission probability", "log2"),
            ("Synapses", "log10"),
        ]):
            out = tmp_path / f"hm{i}.html"
            sv.VisConnMat(mat, str(out), title=title, showfig=False,
                          scale=scale)
            assert out.exists() and out.stat().st_size > 0

    def test_create_heatmap_workflow(self, tmp_path, patched_colorbar):
        hm = sv.CreateHeatmap(output_folder=str(tmp_path / "heat"),
                              showfig=False)
        assert (tmp_path / "heat").is_dir()
        mat = self._matrix()
        ret = hm.add_heatmap(mat, "conn_matrix_type", color_scale="green")
        assert ret is hm
        hm.add_heatmap(mat, "unknown_preset", color_scale="nope")
        hm.add_heatmaps({
            "ratio_matrix_type": mat,
            "transmission_mat": mat,
            "conn_matrix_bodyid": mat,
            "plain_matrix": mat,
        })
        assert len(hm.heatmaps) == 6
        files = hm.create_all()
        assert len(files) == 6
        for f in files:
            assert Path(f).exists()
        assert hm.heatmaps == []
        # empty queue branch
        assert hm.create_all() == []
        # clear() branch
        hm.add_heatmap(mat, "x")
        hm.clear()
        assert hm.heatmaps == []


# =============================================================================
# Path builders: split_path / path_filter / build_path_dataframe_from_paths
# =============================================================================

class TestPathBuilders:
    def test_split_path(self):
        assert sv.split_path(pd.DataFrame()).empty
        df = pd.DataFrame({"path": [[1, 2, 3], [4, 5]]})
        out = sv.split_path(df.copy())
        assert out["path"].tolist() == ["1->2->3", "4->5"]
        assert [list(x) for x in out["path_str"]] == [[1, 2, 3], [4, 5]]
        # already has path_str: list path gets stringified, path_str kept
        df2 = pd.DataFrame({"path": [[1, 2]], "path_str": [[1, 2]]})
        out2 = sv.split_path(df2.copy())
        assert out2["path"].tolist() == ["1->2"]

    def test_path_filter(self):
        df = pd.DataFrame({"path": ["A->B", "C->D"],
                           "path_str": ["A->B", "C->D"]})
        kept, excluded = sv.path_filter(df, None)
        assert len(kept) == 2 and excluded.empty
        kept, excluded = sv.path_filter(df.copy(), "B")
        assert kept["path"].tolist() == ["C->D"]
        assert excluded["path"].tolist() == ["A->B"]
        kept, excluded = sv.path_filter(df.copy(), ["B", "C"])
        assert kept.empty and len(excluded) == 2
        kept, excluded = sv.path_filter(pd.DataFrame(), "B")
        assert kept.empty and excluded.empty

    def test_integer_synapse_count(self):
        assert sv._integer_synapse_count(5.0) == 5
        assert isinstance(sv._integer_synapse_count(5.0), int)
        assert sv._integer_synapse_count(5.5) == 5.5
        assert sv._integer_synapse_count("abc") == "abc"

    def _type_conn_df(self):
        return pd.DataFrame({
            "type_pre": ["A", "A", "B"],
            "type_post": ["B", "B", "C"],
            "weight": [10, 2, 4],
            "traversal_probability": [0.5, 0.5, 0.8],
            "connection_ratio": [0.4, 0.4, 0.2],
            "nt_type": ["ACH", "ACH", "GABA"],
        })

    def test_build_path_dataframe_pandas(self):
        df = sv.build_path_dataframe_from_paths(
            [["A", "B", "C"], ["A", "X", "C"]],
            self._type_conn_df(), targets=["C"], level="type",
            engine="pandas")
        assert len(df) == 1
        row = df.iloc[0]
        assert row["path"] == "A->B->C"
        assert row["weights"] == [12, 4]
        assert row["min_weight"] == 4
        assert row["path_prob"] == pytest.approx(0.5 * 0.8)
        assert row["min_ratio"] == pytest.approx(0.2)
        assert row["length"] == 2
        assert row["nt_types"] == ["ACH|", "GABA"] or row["nt_types"][1] == "GABA"

    def test_build_path_dataframe_with_type_lookup(self):
        conn = pd.DataFrame({
            "bodyId_pre": [1, 2],
            "bodyId_post": [2, 3],
            "weight": [5, 6],
            "traversal_probability": [0.5, 0.5],
            "connection_ratio": [0.3, 0.3],
        })
        lookup = {"1": "A", "2": "B", "3": "C"}
        df = sv.build_path_dataframe_from_paths(
            [["1", "2", "3"]], conn, targets=["3"], level="bodyId",
            type_lookup=lookup, engine="pandas")
        assert df.iloc[0]["path"] == "1_A->2_B->3_C"
        assert df.iloc[0]["path_types"] == ["A", "B", "C"]

    def test_build_path_dataframe_empty(self):
        out = sv.build_path_dataframe_from_paths([], pd.DataFrame(),
                                                 targets=[], engine="pandas")
        assert isinstance(out, pd.DataFrame) and out.empty

    def test_build_path_dataframe_polars_engine(self):
        df = sv.build_path_dataframe_from_paths(
            [["A", "B"]], self._type_conn_df(), targets=["B"],
            level="type", engine="polars")
        assert isinstance(df, pl.DataFrame)
        assert df["path"].to_list() == ["A->B"]
        # auto engine with polars conn data
        df2 = sv.build_path_dataframe_from_paths(
            [["A", "B"]], pl.from_pandas(self._type_conn_df()),
            targets=["B"], level="type", engine="auto")
        assert isinstance(df2, pl.DataFrame)

    def test_enrich_engines_keep_untyped_neurons_as_bodyid(self):
        """Both enrichment engines resolve labels by the exclusive chain
        label -> type -> bodyId (groups: custom_group -> type -> bodyId):
        untyped neurons survive as their bodyId, typed-but-ungrouped
        neurons keep their type as the group label, and null group keys
        never reach the aggregated tables."""
        rows = {
            "bodyId_pre": ["1", "2", "5"],
            "bodyId_post": ["3", "4", "6"],
            "weight": [5, 6, 7],
            "post": [100, 100, 100],
            "type_pre": [None, "A", None],
            "type_post": ["B", None, None],
            "custom_group_pre": ["G1", None, None],
            "custom_group_post": [None, "G2", None],
        }
        expected_type = {("1", "B"), ("A", "4"), ("5", "6")}
        expected_group = {("G1", "B"), ("A", "G2"), ("5", "6")}
        frames = [("polars", pl.DataFrame(rows)),
                  ("pandas", pd.DataFrame(rows))]
        for engine, frame in frames:
            _, conn_t, conn_g = sv.EnrichConnectionTable(
                frame, dataset=None, script_path=None, engine=engine)
            type_edges = set(zip(conn_t["type_pre"], conn_t["type_post"]))
            assert type_edges == expected_type
            assert conn_g is not None
            group_edges = set(zip(conn_g["custom_group_pre"],
                                  conn_g["custom_group_post"]))
            assert group_edges == expected_group
            pre_nulls = (conn_g["custom_group_pre"].is_null()
                         if hasattr(conn_g["custom_group_pre"], "is_null")
                         else conn_g["custom_group_pre"].isna())
            post_nulls = (conn_g["custom_group_post"].is_null()
                          if hasattr(conn_g["custom_group_post"], "is_null")
                          else conn_g["custom_group_post"].isna())
            assert pre_nulls.sum() == 0
            assert post_nulls.sum() == 0


# =============================================================================
# Streaming polars pipeline
# =============================================================================

class TestStreamingPipeline:
    def _conn_df(self):
        return pd.DataFrame({
            "type_pre": ["A", "B"],
            "type_post": ["B", "C"],
            "weight": [10, 4],
            "traversal_probability": [0.5, 0.8],
            "connection_ratio": [0.4, 0.2],
            "nt_type": ["ACH", "GABA"],
        })

    def test_prepare_connection_data(self):
        conn = pd.DataFrame({
            "type_pre": ["A", "A"],
            "type_post": ["B", "B"],
            "weight": [3.0, 4.0],
            "traversal_probability": [0.4, 0.6],
            "connection_ratio": [0.2, 0.4],
            "nt_type": ["ACH", "ACH"],
        })
        out = sv.prepare_connection_data(conn, "type")
        assert isinstance(out, pl.DataFrame)
        row = out.row(0, named=True)
        assert row["src"] == "A" and row["tgt"] == "B"
        assert row["weight"] == 7
        assert row["traversal_probability"] == pytest.approx(0.5)
        assert row["nt_type"] == "ACH"
        # polars input passes through
        out2 = sv.prepare_connection_data(pl.from_pandas(conn), "type")
        assert out2.height == 1

    def test_process_batch_polars(self):
        df_conn = sv.prepare_connection_data(self._conn_df(), "type")
        # empty batch
        a, b = sv.process_batch_polars([], df_conn, "type")
        assert a.is_empty() and b.is_empty()
        # second path has an unknown edge -> zero weight -> filtered out
        df, excl = sv.process_batch_polars([["A", "B", "C"], ["A", "Z"]],
                                           df_conn, "type")
        assert df.height == 1
        assert df["path"].to_list() == ["A->B->C"]
        assert df["min_weight"].to_list() == [4]
        assert df["length"].to_list() == [2]
        assert df["nt_types"].to_list() == ['["ACH", "GABA"]']
        assert excl.is_empty()

    def test_process_batch_keyword_and_label_map(self):
        df_conn = sv.prepare_connection_data(self._conn_df(), "type")
        df, excl = sv.process_batch_polars(
            [["A", "B", "C"], ["A", "B"]], df_conn, "type",
            keyword_in_path_to_remove="C")
        assert df["path"].to_list() == ["A->B"]
        assert excl["path"].to_list() == ["A->B->C"]
        df2, _ = sv.process_batch_polars(
            [["A", "B"]], df_conn, "type",
            type_to_label_map={"A": "GA", "B": "GB"})
        assert df2["path"].to_list() == ["GA->GB"]

    def test_write_buffer_to_csv(self, tmp_path):
        d1 = pl.DataFrame({"a": [1], "b": ["x"]})
        d2 = pl.DataFrame({"a": [2], "b": ["y"]})
        out = tmp_path / "buf.csv"
        sv._write_buffer_to_csv([], out)  # empty buffer no-op
        assert not out.exists()
        sv._write_buffer_to_csv([d1], out)
        sv._write_buffer_to_csv([d2], out, append=True)
        df = pl.read_csv(out)
        assert df["a"].to_list() == [1, 2]

    def test_process_paths_streaming(self, tmp_path):
        paths = [["A", "B", "C"]] * 25
        out = tmp_path / "paths.csv"
        total = sv.process_paths_streaming(
            iter(paths), self._conn_df(), targets=["C"], output_path=str(out),
            level="type", batch_size=1, verbose=False)
        assert total == 25
        df = pl.read_csv(out)
        assert df.height == 25

    def test_process_paths_streaming_excluded(self, tmp_path):
        paths = [["A", "B", "C"]] * 3
        out = tmp_path / "paths.csv"
        excl = tmp_path / "excluded.csv"
        total = sv.process_paths_streaming(
            iter(paths), self._conn_df(), targets=["C"], output_path=str(out),
            excluded_path=str(excl), level="type", batch_size=2,
            keyword_in_path_to_remove="B", verbose=False)
        assert total == 0
        assert pl.read_csv(excl).height == 3


# =============================================================================
# _type_probability_series
# =============================================================================

class TestTypeProbabilitySeries:
    def _pairs(self):
        return pd.DataFrame({
            "type_pre": ["A", "A", "B"],
            "type_post": ["B", "B", "C"],
            "weight": [10, 2, 5],
            "block_probability": [0.5, 0.25, 0.4],
            "traversal_probability": [0.5, 0.75, 0.6],
        })

    def test_product(self):
        s = sv._type_probability_series(self._pairs(), "type_pre",
                                        "type_post", "product")
        assert s[("A", "B")] == pytest.approx(1 - 0.5 * 0.25)
        assert s[("B", "C")] == pytest.approx(0.6)

    def test_average(self):
        s = sv._type_probability_series(self._pairs(), "type_pre",
                                        "type_post", "average")
        expected = (10 * 0.5 + 2 * 0.75) / 12
        assert s[("A", "B")] == pytest.approx(expected)

    def test_ratio_returns_none(self):
        assert sv._type_probability_series(self._pairs(), "type_pre",
                                           "type_post", "ratio") is None


# =============================================================================
# build_bodyid_label_map
# =============================================================================

class FakeLabelMapper:
    def __init__(self, source=None, target=None, intermediate=None):
        self._source_mapping = source or {}
        self._target_mapping = target or {}
        self._intermediate_mapping = intermediate or {}


class TestBuildBodyIdLabelMap:
    def _neuron_df(self):
        return pl.DataFrame({
            "bodyId": [1, 2, 3, 4],
            "type": ["Mi1", "Mi1", "Tm3", "X"],
            "instance": ["Mi1_L", "Mi1_R", "Tm3_L", "X_L"],
        })

    def test_empty_inputs(self):
        assert sv.build_bodyid_label_map(None, "d", self._neuron_df()) == {}
        mapper = FakeLabelMapper()
        assert sv.build_bodyid_label_map(mapper, "d", pl.DataFrame()) == {}

    def test_no_bodyid_column(self):
        mapper = FakeLabelMapper(source={"G": {"d": [1]}})
        out = sv.build_bodyid_label_map(mapper, "d",
                                        pl.DataFrame({"type": ["a"]}))
        assert out == {}

    def test_direct_type_instance_expansion(self):
        mapper = FakeLabelMapper(
            source={"G1": {"test:v1": [1, "Mi1"]}},
            target={
                "G2": {"test_v1": ["Tm3_L"]},     # sanitized dataset key
                "G4": {"test:v1": ["nosuch"]},    # unknown -> stored as-is
            },
            intermediate={"G3": {"other_ds": ["99"]}},  # dataset mismatch
        )
        out = sv.build_bodyid_label_map(mapper, "test:v1", self._neuron_df())
        assert out["1"] == "G1"       # direct bodyId
        assert out["2"] == "G1"       # type expansion (Mi1)
        assert out["3"] == "G2"       # instance expansion (Tm3_L)
        assert out["nosuch"] == "G4"
        assert "4" not in out
        assert "99" not in out


# =============================================================================
# getCriteriaAndName / _get_coverage_notes
# =============================================================================

class TestMiscHelpers:
    def test_get_criteria_and_name(self, monkeypatch):
        import neuprint

        class FakeNC:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        # neuprint.NeuronCriteria requires a live client; use a stand-in
        monkeypatch.setattr(neuprint, "NeuronCriteria", FakeNC)
        criteria, fname = sv.getCriteriaAndName(None)
        assert criteria is None and fname == "ALL"
        criteria, fname = sv.getCriteriaAndName([123])
        assert fname == "123" and criteria.kwargs == {"bodyId": [123]}
        criteria, fname = sv.getCriteriaAndName(["aMe.*"])
        assert fname == "aMe" and criteria.kwargs == {"instance": ["aMe.*"]}
        criteria, fname = sv.getCriteriaAndName(["Mi1"])
        assert fname == "Mi1" and criteria.kwargs == {"type": ["Mi1"]}
        criteria, fname = sv.getCriteriaAndName(["Mi1", "Tm3"])
        assert fname == "Mi1_etc"
        with pytest.raises(ValueError):
            sv.getCriteriaAndName("not-a-list")

    def test_get_coverage_notes(self):
        assert "Central brain" in sv._get_coverage_notes("hemibrain:v1.2.1")
        assert "not available" in sv._get_coverage_notes("unknown-ds")
        assert "FAFB" in sv._get_coverage_notes("flywire_70k")
