"""Coverage tests for comparison.visualizations (ComparisonVisualizer).

Hermetic: synthetic pandas data only, matplotlib Agg backend, no network.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")  # must be set before pyplot is imported anywhere

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from comparison.visualizations import ComparisonVisualizer

DATASETS = ["ds_one", "ds_two"]
THRESHOLDS = [1, 5]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def viz():
    return ComparisonVisualizer(verbose=False)


def _edge_df(n=4, weight_shift=0):
    """Result DataFrame per (dataset, threshold) with type columns + weight."""
    return pd.DataFrame(
        {
            "type_pre": [f"P{i}" for i in range(n)],
            "type_post": [f"Q{i}" for i in range(n)],
            "weight": [float(i + 1 + weight_shift) for i in range(n)],
            "has_valid_path": [True, False, True, True][:n],
        }
    )


@pytest.fixture
def results():
    return {
        "ds_one": {1: _edge_df(4, 0), 5: _edge_df(2, 1)},
        "ds_two": {1: _edge_df(4, 2), 5: _edge_df(3, 0)},
    }


@pytest.fixture
def aligned():
    idx = ["A -> B", "B -> C", "A -> C", "C -> D"]
    return pd.DataFrame(
        {"ds_one": [10.0, 0.0, 5.0, 3.0], "ds_two": [8.0, 4.0, 0.0, 3.0]},
        index=idx,
    )


@pytest.fixture
def similarities():
    return pd.DataFrame(
        [
            {
                "dataset_1": "ds_one",
                "dataset_2": "ds_two",
                "threshold": 1,
                "jaccard_similarity": 0.6,
                "svd_similarity": 0.7,
                "cosine_similarity": 0.8,
                "edge_rank_correlation": 0.5,
                "spearman_rank_correlation": 0.4,
                "weighted_jaccard": 0.55,
                "kernel_similarity": 0.65,
            }
        ]
    )


@pytest.fixture
def path_data():
    idx = ["A -> B", "A -> C", "B -> D"]
    return pd.DataFrame(
        {"ds_one": [9.0, 0.0, 6.0], "ds_two": [7.0, 5.0, 0.0]}, index=idx
    )


@pytest.fixture
def prob_data():
    idx = ["A -> B", "A -> C"]
    return pd.DataFrame(
        {"ds_one": [0.5, 0.01], "ds_two": [0.3, 0.2]}, index=idx
    )


@pytest.fixture
def ratio_data():
    idx = ["A -> B", "A -> C"]
    return pd.DataFrame(
        {"ds_one": [0.9, 0.1], "ds_two": [0.8, 0.0]}, index=idx
    )


# ---------------------------------------------------------------------------
# Path count plots
# ---------------------------------------------------------------------------

def test_plot_path_counts(viz, results):
    fig = viz.plot_path_counts(results, THRESHOLDS, nickname_map={"ds_one": "D1"})
    assert fig is not None
    assert len(fig.axes) >= 1


def test_plot_path_counts_missing_threshold(viz, results):
    # threshold 9 absent from results -> zero counts handled
    fig = viz.plot_path_counts(results, [1, 9])
    assert fig is not None


def test_plot_path_counts_stacked(viz, results):
    fig = viz.plot_path_counts_stacked(results, THRESHOLDS)
    assert fig is not None


# ---------------------------------------------------------------------------
# Edge heatmaps
# ---------------------------------------------------------------------------

def test_plot_edge_weight_heatmap(viz, aligned):
    fig = viz.plot_edge_weight_heatmap(
        aligned, DATASETS, max_edges=3, nickname_map={"ds_one": "D1"}
    )
    assert fig is not None


def test_plot_edge_weight_heatmap_no_columns(viz, aligned):
    fig = viz.plot_edge_weight_heatmap(aligned, ["missing_ds"])
    assert fig is not None  # fallback "No data available" figure


def test_plot_edge_weight_heatmap_all_thresholds(viz, results, aligned):
    fig = viz.plot_edge_weight_heatmap_all_thresholds(
        results, THRESHOLDS, align_func=lambda t: aligned, max_edges=3
    )
    assert fig is not None


def test_plot_edge_weight_heatmap_all_thresholds_empty(viz, results):
    empty = pd.DataFrame()
    fig = viz.plot_edge_weight_heatmap_all_thresholds(
        results, THRESHOLDS, align_func=lambda t: empty
    )
    assert fig is not None


def test_plot_edge_weight_heatmap_all_thresholds_raising(viz, results):
    def bad(t):
        raise RuntimeError("boom")

    fig = viz.plot_edge_weight_heatmap_all_thresholds(
        results, THRESHOLDS, align_func=bad
    )
    assert fig is not None


# ---------------------------------------------------------------------------
# Similarity matrices
# ---------------------------------------------------------------------------

def test_plot_similarity_matrix(viz, similarities):
    fig = viz.plot_similarity_matrix(similarities, metric="jaccard_similarity")
    assert fig is not None


def test_plot_dual_similarity_matrices(viz):
    sims = pd.DataFrame(
        [
            {
                "dataset_1": "ds_one",
                "dataset_2": "ds_two",
                "jaccard_similarity": 0.5,
                "svd_similarity": np.nan,  # exercises NaN branch
            }
        ]
    )
    fig = viz.plot_dual_similarity_matrices(sims)
    assert fig is not None
    assert len(fig.axes) >= 2  # two heatmaps (+ colorbar axes)


@pytest.mark.parametrize(
    "metric", ["jaccard", "weighted_jaccard", "svd", "edge_rank", "kernel"]
)
def test_plot_similarity_per_threshold(viz, similarities, metric):
    sims_t1 = similarities.copy()
    sims_t5 = pd.DataFrame()  # empty entry exercises fallback text
    fig = viz.plot_similarity_per_threshold(
        {1: sims_t1, 5: sims_t5}, metric=metric
    )
    assert fig is not None


def test_plot_similarity_per_threshold_none_entry(viz):
    fig = viz.plot_similarity_per_threshold({1: None})
    assert fig is not None


def test_plot_similarity_per_threshold_empty(viz):
    fig = viz.plot_similarity_per_threshold({})
    assert fig is not None


# ---------------------------------------------------------------------------
# Overlap / venn
# ---------------------------------------------------------------------------

def test_plot_edge_overlap(viz, aligned):
    fig = viz.plot_edge_overlap(aligned, DATASETS, threshold=1)
    assert fig is not None


def test_plot_edge_overlap_single_dataset(viz, aligned):
    fig = viz.plot_edge_overlap(aligned, ["ds_one"])
    assert fig is not None  # fallback figure


def test_plot_venn2_style(viz, aligned):
    fig = viz.plot_venn2_style(aligned, DATASETS, threshold=1)
    assert fig is not None


def test_plot_venn2_style_wrong_count(viz, aligned):
    with pytest.raises(ValueError):
        viz.plot_venn2_style(aligned, ["ds_one"])


def test_plot_venn2_style_missing_column(viz, aligned):
    fig = viz.plot_venn2_style(aligned, ["ds_one", "nope"])
    assert fig is not None  # fallback figure


# ---------------------------------------------------------------------------
# Scatter / distributions
# ---------------------------------------------------------------------------

def test_plot_weight_scatter(viz, aligned):
    fig = viz.plot_weight_scatter(aligned, "ds_one", "ds_two")
    assert fig is not None


def test_plot_weight_scatter_log(viz, aligned):
    fig = viz.plot_weight_scatter(aligned, "ds_one", "ds_two", log_scale=True)
    assert fig is not None


def test_plot_fold_change_distribution(viz):
    diff = pd.DataFrame({"max_fold_change": [0.5, 1.2, 2.4, 5.0]})
    fig = viz.plot_fold_change_distribution(diff)
    assert fig is not None


def test_plot_fold_change_distribution_missing_column(viz):
    fig = viz.plot_fold_change_distribution(pd.DataFrame({"x": [1, 2]}))
    assert fig is not None


def test_plot_similarity_vs_threshold(viz):
    df = pd.DataFrame(
        {
            "threshold": [1, 5, 1, 5],
            "dataset_1": ["a", "a", "b", "b"],
            "dataset_2": ["c", "c", "d", "d"],
            "jaccard_similarity": [0.4, 0.6, 0.2, 0.8],
        }
    )
    fig = viz.plot_similarity_vs_threshold(df)
    assert fig is not None


def test_plot_similarity_vs_threshold_empty(viz):
    fig = viz.plot_similarity_vs_threshold(pd.DataFrame())
    assert fig is not None


# ---------------------------------------------------------------------------
# Utility / saving
# ---------------------------------------------------------------------------

def test_save_figure(viz, tmp_path):
    fig = viz.plot_path_counts(
        {"d": {1: _edge_df(2)}}, [1]
    )
    out = tmp_path / "fig.png"
    viz.save_figure(fig, str(out), dpi=50)
    assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Composite subplots
# ---------------------------------------------------------------------------

def test_plot_threshold_comparison_subplots(viz, results):
    fig = viz.plot_threshold_comparison_subplots(results, THRESHOLDS)
    assert fig is not None


def test_plot_threshold_comparison_subplots_single_threshold(viz, results):
    fig = viz.plot_threshold_comparison_subplots(results, [1])
    assert fig is not None


def test_plot_similarity_matrix_per_threshold(viz, results, aligned, similarities):
    # Provide cached similarities including the advanced metric columns
    sims = similarities.copy()
    fig = viz.plot_similarity_matrix_per_threshold(
        results,
        THRESHOLDS,
        align_func=lambda t: aligned,
        similarity_func=lambda t: sims,
        show_progress=False,
    )
    assert fig is not None


def test_plot_similarity_matrix_per_threshold_recompute(viz, results, aligned):
    # No similarity_func -> metrics are recomputed from aligned data
    fig = viz.plot_similarity_matrix_per_threshold(
        results,
        [1],
        align_func=lambda t: aligned,
        show_progress=False,
    )
    assert fig is not None


def test_plot_similarity_matrix_per_threshold_empty(viz, results):
    fig = viz.plot_similarity_matrix_per_threshold(
        results, THRESHOLDS, align_func=lambda t: pd.DataFrame(),
        show_progress=False,
    )
    assert fig is not None


# ---------------------------------------------------------------------------
# Path-level heatmaps
# ---------------------------------------------------------------------------

def test_plot_path_heatmap(viz, path_data):
    fig = viz.plot_path_heatmap(
        path_data, DATASETS, nickname_map={"ds_one": "D1"}
    )
    assert fig is not None


def test_plot_path_heatmap_no_columns(viz, path_data):
    fig = viz.plot_path_heatmap(path_data, ["nope"])
    assert fig is not None


def test_plot_path_heatmap_all_thresholds(viz, path_data):
    fig = viz.plot_path_heatmap_all_thresholds(
        lambda t: path_data, THRESHOLDS, DATASETS
    )
    assert fig is not None


def test_plot_path_heatmap_all_thresholds_empty(viz):
    fig = viz.plot_path_heatmap_all_thresholds(
        lambda t: pd.DataFrame(), THRESHOLDS, DATASETS
    )
    assert fig is not None


def test_plot_path_heatmap_all_thresholds_raising(viz):
    def bad(t):
        raise RuntimeError("no data")

    fig = viz.plot_path_heatmap_all_thresholds(bad, THRESHOLDS, DATASETS)
    assert fig is not None


def test_plot_ratio_heatmap(viz, ratio_data):
    fig = viz.plot_ratio_heatmap(ratio_data, DATASETS)
    assert fig is not None


def test_plot_ratio_heatmap_no_columns(viz, ratio_data):
    fig = viz.plot_ratio_heatmap(ratio_data, ["nope"])
    assert fig is not None


def test_plot_traversal_probability_heatmap(viz, prob_data):
    fig = viz.plot_traversal_probability_heatmap(prob_data, DATASETS)
    assert fig is not None


def test_plot_traversal_probability_heatmap_no_columns(viz, prob_data):
    fig = viz.plot_traversal_probability_heatmap(prob_data, ["nope"])
    assert fig is not None


def test_plot_ratio_heatmap_all_thresholds(viz, ratio_data):
    fig = viz.plot_ratio_heatmap_all_thresholds(
        lambda t: ratio_data, THRESHOLDS, DATASETS
    )
    assert fig is not None


def test_plot_ratio_heatmap_all_thresholds_none(viz):
    fig = viz.plot_ratio_heatmap_all_thresholds(
        lambda t: None, THRESHOLDS, DATASETS
    )
    assert fig is not None


def test_plot_traversal_probability_heatmap_all_thresholds(viz, prob_data):
    fig = viz.plot_traversal_probability_heatmap_all_thresholds(
        lambda t: prob_data, THRESHOLDS, DATASETS
    )
    assert fig is not None


def test_plot_traversal_probability_heatmap_all_thresholds_none(viz):
    fig = viz.plot_traversal_probability_heatmap_all_thresholds(
        lambda t: None, THRESHOLDS, DATASETS
    )
    assert fig is not None


# ---------------------------------------------------------------------------
# Conservation / trend plots
# ---------------------------------------------------------------------------

def test_plot_conservation_across_thresholds(viz, results, aligned):
    fig, df = viz.plot_conservation_across_thresholds(
        results, THRESHOLDS, align_func=lambda t: aligned
    )
    assert fig is not None
    assert isinstance(df, pd.DataFrame) and not df.empty
    assert "edge_count" in df.columns


def test_plot_conservation_across_thresholds_presence_matrix(viz, results, aligned):
    matrix = pd.DataFrame(
        {
            "ds_one_t1": ["True", "False"],
            "ds_one_t5": ["True", "True"],
            "ds_two_t1": [True, False],
            "ds_two_t5": [1, 0],
        }
    )
    fig, df = viz.plot_conservation_across_thresholds(
        results,
        THRESHOLDS,
        align_func=lambda t: aligned,
        path_presence_matrix=matrix,
    )
    assert fig is not None
    assert not df.empty


def test_plot_jaccard_similarity_trend(viz, aligned):
    fig, df = viz.plot_jaccard_similarity_trend(
        lambda t: aligned, THRESHOLDS, DATASETS
    )
    assert fig is not None
    assert not df.empty and "jaccard" in df.columns


def test_plot_edge_rank_correlation_trend(viz, aligned):
    fig, df = viz.plot_edge_rank_correlation_trend(
        lambda t: aligned, THRESHOLDS, DATASETS
    )
    assert fig is not None
    assert not df.empty


def test_plot_path_rank_correlation_trend(viz, path_data):
    fig, df = viz.plot_path_rank_correlation_trend(
        lambda t: path_data, THRESHOLDS, DATASETS
    )
    assert fig is not None
    assert not df.empty


def test_plot_path_rank_correlation_trend_empty(viz):
    fig, df = viz.plot_path_rank_correlation_trend(
        lambda t: pd.DataFrame(), THRESHOLDS, DATASETS
    )
    assert fig is not None
    assert df.empty


def test_plot_cosine_similarity_trend(viz, aligned):
    fig, df = viz.plot_cosine_similarity_trend(
        lambda t: aligned, THRESHOLDS, DATASETS
    )
    assert fig is not None
    assert not df.empty


def test_plot_conservation_across_thresholds_plotly(viz, results, aligned):
    payload = viz.plot_conservation_across_thresholds_plotly(
        results, THRESHOLDS, align_func=lambda t: aligned
    )
    assert isinstance(payload, str)
    parsed = json.loads(payload)
    assert "data" in parsed and len(parsed["data"]) > 0


# ---------------------------------------------------------------------------
# save_all_plots orchestration
# ---------------------------------------------------------------------------

def test_save_all_plots(viz, results, aligned, similarities, path_data,
                        ratio_data, prob_data, tmp_path):
    out_dir = str(tmp_path / "plots")
    sims = similarities.copy()
    presence = pd.DataFrame(
        {
            "ds_one_t1": ["True"],
            "ds_one_t5": ["True"],
            "ds_two_t1": ["True"],
            "ds_two_t5": ["False"],
        }
    )
    viz.save_all_plots(
        results,
        aligned,
        sims,
        out_dir,
        THRESHOLDS,
        align_func=lambda t: aligned,
        similarity_func=lambda t: sims,
        path_data_func=lambda t: path_data,
        ratio_data_func=lambda t: ratio_data,
        prob_data_func=lambda t: prob_data,
        nickname_map={"ds_one": "D1", "ds_two": "D2"},
        path_presence_matrix=presence,
        silent=True,
    )
    files = os.listdir(out_dir)
    assert "path_counts.png" in files
    assert any(f.endswith(".png") for f in files)
    vis_data = os.path.join(out_dir, "visualization_data")
    assert os.path.isdir(vis_data)
    assert os.path.exists(os.path.join(vis_data, "path_counts.csv"))
    # subfolders for ratio / probability heatmaps
    assert os.path.isdir(os.path.join(out_dir, "by_ratio"))
    assert os.path.isdir(os.path.join(out_dir, "by_probability"))


def test_generate_vis_summary_pdf(viz, tmp_path):
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    # Create two tiny PNGs
    for name in ("a.png", "b.png"):
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([0, 1], [0, 1])
        fig.savefig(str(plots_dir / name), dpi=40)
        plt.close(fig)
    viz._generate_vis_summary_pdf(str(plots_dir))
    assert (tmp_path / "vis_summary.pdf").exists()


def test_generate_vis_summary_pdf_no_pngs(viz, tmp_path):
    plots_dir = tmp_path / "empty"
    plots_dir.mkdir()
    viz._generate_vis_summary_pdf(str(plots_dir))  # early return, no crash
    assert not (tmp_path / "vis_summary.pdf").exists()


# ============================================================================
# interactive_heatmap module
# ============================================================================

from comparison.interactive_heatmap import generate_interactive_heatmap


def _metric_df(n=4, with_nan=False):
    vals = np.linspace(0.1, 1.0, n * n).reshape(n, n)
    if with_nan:
        vals[0, 1] = np.nan
    labels = [f"T{i}" for i in range(n)]
    return pd.DataFrame(vals, index=labels, columns=labels)


def test_interactive_heatmap_basic(tmp_path):
    matrices = {"jaccard": _metric_df(with_nan=True), "cosine": _metric_df()}
    out = tmp_path / "heatmap.html"
    generate_interactive_heatmap(
        matrices, str(out), title="Test Heatmap", showfig=False, verbose=True
    )
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "Test Heatmap" in html
    assert "Jaccard" in html  # display name of metric option
    assert "clusteringAvailable" in html
    assert "cdn.plot.ly" in html


def test_interactive_heatmap_empty_dict(tmp_path):
    with pytest.raises(ValueError):
        generate_interactive_heatmap({}, str(tmp_path / "x.html"), showfig=False)


def test_interactive_heatmap_single_cell(tmp_path):
    matrices = {"m": pd.DataFrame([[0.5]], index=["a"], columns=["b"])}
    out = tmp_path / "single.html"
    generate_interactive_heatmap(matrices, str(out), showfig=False, verbose=False)
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "metricsData" in html
    assert "let showLabels = true" in html  # small matrix: labels shown


def test_interactive_heatmap_large_branch(tmp_path):
    # >100 rows triggers the is_large branch (labels hidden by default)
    n = 101
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        rng.random((n, 3)),
        index=[f"r{i}" for i in range(n)],
        columns=["a", "b", "c"],
    )
    out = tmp_path / "large.html"
    generate_interactive_heatmap({"m": df}, str(out), showfig=False, verbose=False)
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "Show Labels" in html  # large matrices start with labels hidden
