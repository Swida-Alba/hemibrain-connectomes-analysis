"""Coverage tests for comparison.profile_visualizations (ProfileVisualizer).

Hermetic: synthetic ConnectivityProfile objects and DataFrames only.
matplotlib Agg backend is enforced by the module itself.
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from comparison.connectivity_profiler import ConnectivityProfile
from comparison.profile_visualizations import ProfileVisualizer


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _profile(dataset: str, neuron_type: str, up: dict, down: dict) -> ConnectivityProfile:
    up_ranks = {k: i + 1 for i, k in enumerate(
        sorted(up, key=up.get, reverse=True))}
    down_ranks = {k: i + 1 for i, k in enumerate(
        sorted(down, key=down.get, reverse=True))}
    return ConnectivityProfile(
        neuron_id=neuron_type,
        dataset=dataset,
        neuron_type=neuron_type,
        upstream_partners=dict(up),
        downstream_partners=dict(down),
        upstream_ranks=up_ranks,
        downstream_ranks=down_ranks,
        total_upstream_weight=float(sum(up.values())),
        total_downstream_weight=float(sum(down.values())),
        actual_upstream_count=len(up),
        actual_downstream_count=len(down),
        unique_types_upstream=len(up),
        unique_types_downstream=len(down),
    )


@pytest.fixture
def profiles_two_datasets():
    """{dataset: profile} for one neuron type across two datasets."""
    return {
        "dsA": _profile("dsA", "aMe12",
                        {"LNV5": 10.0, "pdf": 8.0, "DN1a": 3.0},
                        {"sLNv": 9.0, "LNd": 4.0}),
        "dsB": _profile("dsB", "aMe12",
                        {"LNV5": 12.0, "pdf": 5.0, "DN1b": 2.0},
                        {"sLNv": 7.0, "LNd": 6.0, "DN3": 1.0}),
    }


@pytest.fixture
def profiles_by_dataset():
    """{dataset: {type: profile}} with >=2 types per dataset."""
    return {
        "dsA": {
            "aMe12": _profile("dsA", "aMe12",
                              {"LNV5": 10.0, "pdf": 8.0, "DN1a": 3.0},
                              {"sLNv": 9.0, "LNd": 4.0}),
            "DN1a": _profile("dsA", "DN1a",
                             {"aMe12": 5.0, "LNv": 2.0},
                             {"aMe12": 6.0, "DNa": 1.0}),
        },
        "dsB": {
            "aMe12": _profile("dsB", "aMe12",
                              {"LNV5": 12.0, "pdf": 5.0},
                              {"sLNv": 7.0, "LNd": 6.0}),
            "DN1a": _profile("dsB", "DN1a",
                             {"aMe12": 4.0, "LNv": 3.0},
                             {"aMe12": 5.0, "DNa": 2.0}),
        },
    }


@pytest.fixture
def similarity_matrix():
    return pd.DataFrame(
        [[1.0, 0.7, 0.4], [0.7, 1.0, 0.5], [0.4, 0.5, 1.0]],
        index=["dsA", "dsB", "dsC"],
        columns=["dsA vs dsB", "dsA vs dsC", "dsB vs dsC"],
    )


@pytest.fixture
def verification_df():
    return pd.DataFrame(
        [
            {"neuron_type": "aMe12", "role": "source", "confidence": "High",
             "avg_rank_corr": 0.9, "avg_combined_score": 0.85,
             "min_score": 0.8, "max_score": 0.95},
            {"neuron_type": "aMe12", "role": "target", "confidence": "Medium",
             "avg_rank_corr": 0.6, "avg_combined_score": 0.6,
             "min_score": 0.5, "max_score": 0.7},
            {"neuron_type": "DN1a", "role": "intermediate", "confidence": "Low",
             "avg_rank_corr": 0.35, "avg_combined_score": 0.4,
             "min_score": 0.2, "max_score": 0.5},
            {"neuron_type": "LNv", "role": "source/target", "confidence": "Error",
             "avg_rank_corr": np.nan, "avg_combined_score": np.nan,
             "min_score": np.nan, "max_score": np.nan},
        ]
    )


@pytest.fixture
def summary_df(verification_df):
    df = verification_df.drop_duplicates(subset="neuron_type").copy()
    df["datasets_found"] = 2
    df["total_datasets"] = 2
    # one value per remaining row (aMe12, DN1a, LNv)
    df["avg_jaccard"] = [0.4, 0.1, np.nan][: len(df)]
    df["avg_jaccard_both"] = df["avg_jaccard"]
    df["avg_jaccard_upstream"] = df["avg_jaccard"] + 0.05
    df["avg_jaccard_downstream"] = df["avg_jaccard"] - 0.05
    df["avg_rank_corr_upstream"] = df["avg_rank_corr"]
    df["avg_rank_corr_downstream"] = df["avg_rank_corr"] - 0.1
    return df


# ---------------------------------------------------------------------------
# plot_profile_comparison
# ---------------------------------------------------------------------------

def test_plot_profile_comparison_both(profiles_two_datasets, tmp_path):
    out = str(tmp_path / "profile_both.png")
    fig = ProfileVisualizer.plot_profile_comparison(
        profiles_two_datasets, "aMe12", direction="both", output_path=out
    )
    assert fig is not None
    assert os.path.exists(out)


@pytest.mark.parametrize("direction", ["upstream", "downstream"])
def test_plot_profile_comparison_single_direction(profiles_two_datasets, direction):
    fig = ProfileVisualizer.plot_profile_comparison(
        profiles_two_datasets, "aMe12", direction=direction
    )
    assert fig is not None


def test_plot_profile_comparison_empty_profiles():
    empty = {"dsA": _profile("dsA", "x", {}, {})}
    fig = ProfileVisualizer.plot_profile_comparison(
        empty, "x", direction="upstream"
    )
    assert fig is not None  # fallback text figure


def test_plot_profile_comparison_empty_direction_both():
    empty = {"dsA": _profile("dsA", "x", {}, {})}
    fig = ProfileVisualizer.plot_profile_comparison(empty, "x", direction="both")
    assert fig is not None


def test_plot_profile_comparison_many_partners():
    # > 15 partners exercises truncation branch
    up = {f"P{i}": float(i + 1) for i in range(20)}
    down = {f"Q{i}": float(i + 1) for i in range(20)}
    profs = {"dsA": _profile("dsA", "big", up, down)}
    fig = ProfileVisualizer.plot_profile_comparison(profs, "big", direction="upstream")
    assert fig is not None


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------

def test_plot_similarity_heatmap(similarity_matrix, tmp_path):
    out = str(tmp_path / "sim.png")
    fig = ProfileVisualizer.plot_similarity_heatmap(
        similarity_matrix, output_path=out
    )
    assert fig is not None
    assert os.path.exists(out)


def test_plot_similarity_heatmap_empty():
    fig = ProfileVisualizer.plot_similarity_heatmap(pd.DataFrame())
    assert fig is not None


def test_plot_multi_metric_heatmaps(similarity_matrix, tmp_path):
    matrices = {
        "combined": similarity_matrix,
        "jaccard": similarity_matrix,
        "cosine": similarity_matrix,
        "rank": similarity_matrix - 0.5,  # may contain negatives
    }
    figs = ProfileVisualizer.plot_multi_metric_heatmaps(
        matrices, output_dir=str(tmp_path)
    )
    assert set(figs.keys()) == set(matrices.keys())
    for m in matrices:
        assert (tmp_path / f"similarity_heatmap_{m}.png").exists()


def test_plot_directional_heatmaps(similarity_matrix, tmp_path):
    directional = {
        "upstream": similarity_matrix,
        "downstream": similarity_matrix,
        "both": similarity_matrix,
    }
    figs = ProfileVisualizer.plot_directional_heatmaps(
        directional, metric_name="combined", output_dir=str(tmp_path)
    )
    assert set(figs.keys()) == set(directional.keys())
    for d in directional:
        assert (tmp_path / f"similarity_heatmap_{d}.png").exists()


# ---------------------------------------------------------------------------
# Verification summary / role plots
# ---------------------------------------------------------------------------

def test_plot_verification_summary(verification_df, tmp_path):
    out = str(tmp_path / "summary.png")
    fig = ProfileVisualizer.plot_verification_summary(
        verification_df, output_path=out
    )
    assert fig is not None
    assert os.path.exists(out)


def test_plot_verification_summary_empty():
    fig = ProfileVisualizer.plot_verification_summary(pd.DataFrame())
    assert fig is not None


def test_plot_verification_summary_no_minmax(verification_df):
    df = verification_df.drop(columns=["min_score", "max_score"])
    fig = ProfileVisualizer.plot_verification_summary(df)
    assert fig is not None


def test_plot_role_comparison(verification_df, tmp_path):
    out = str(tmp_path / "roles.png")
    fig = ProfileVisualizer.plot_role_comparison(
        verification_df, output_path=out
    )
    assert fig is not None
    assert os.path.exists(out)


def test_plot_role_comparison_empty():
    fig = ProfileVisualizer.plot_role_comparison(pd.DataFrame())
    assert fig is not None


def test_plot_role_comparison_missing_role_column():
    fig = ProfileVisualizer.plot_role_comparison(
        pd.DataFrame({"neuron_type": ["x"], "avg_rank_corr": [0.5]})
    )
    assert fig is not None


# ---------------------------------------------------------------------------
# Inter-type heatmaps
# ---------------------------------------------------------------------------

def test_generate_inter_type_similarity_heatmap(tmp_path):
    matrix = pd.DataFrame(
        [[1.0, 0.6, 0.2, 0.1],
         [0.6, 1.0, 0.5, 0.3],
         [0.2, 0.5, 1.0, 0.4],
         [0.1, 0.3, 0.4, 1.0]],
        index=["t1", "t2", "t3", "t4"],
        columns=["t1", "t2", "t3", "t4"],
    )
    out = str(tmp_path / "inter.html")
    path = ProfileVisualizer.generate_inter_type_similarity_heatmap(
        matrix, out, title="Inter-Type", cluster=True
    )
    assert path == out
    content = open(out).read()
    assert "Inter-Type" in content
    assert "Plotly.newPlot" in content


def test_generate_inter_type_similarity_heatmap_no_cluster(tmp_path):
    matrix = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]], index=["a", "b"], columns=["a", "b"]
    )
    out = str(tmp_path / "inter2.html")
    path = ProfileVisualizer.generate_inter_type_similarity_heatmap(
        matrix, out, cluster=False
    )
    assert os.path.exists(path)


def test_generate_inter_type_similarity_heatmap_empty(tmp_path):
    result = ProfileVisualizer.generate_inter_type_similarity_heatmap(
        pd.DataFrame(), str(tmp_path / "none.html")
    )
    assert result is None


def test_generate_all_inter_type_heatmaps(profiles_by_dataset, tmp_path):
    files = ProfileVisualizer.generate_all_inter_type_heatmaps(
        profiles_by_dataset, str(tmp_path), metric="rank", direction="both"
    )
    assert files
    for path in files.values():
        assert os.path.exists(path)


def test_generate_plotly_heatmap_div(similarity_matrix):
    div = ProfileVisualizer._generate_plotly_heatmap_div(
        similarity_matrix, title="Test Heatmap"
    )
    assert isinstance(div, str)
    assert "plotly" in div.lower()


# ---------------------------------------------------------------------------
# HTML report + save_all_visualizations
# ---------------------------------------------------------------------------

def test_generate_html_report(summary_df, similarity_matrix, tmp_path):
    metric_matrices = {
        "jaccard": similarity_matrix,
        "cosine": similarity_matrix,
    }
    out = str(tmp_path / "report.html")
    path = ProfileVisualizer.generate_html_report(
        {"summary": summary_df},
        similarity_matrix=similarity_matrix,
        metric_matrices=metric_matrices,
        output_path=out,
        title="Test Verification Report",
        profile_comparison_url="profiles.html",
        main_report_url="main.html",
        inter_type_heatmap_files={"all_types": "/tmp/x/inter.html",
                                  "csv": "/tmp/x/data.csv"},
    )
    assert path == out
    html = open(out).read()
    assert "Test Verification Report" in html
    assert "Verification Summary" in html
    assert "Jaccard Similarity Matrix" in html
    assert "Cosine Similarity Matrix" in html
    assert "Dataset Pair Similarity Summary" in html
    assert "Inter-Type Comparison Heatmaps" in html


def test_generate_html_report_minimal(tmp_path):
    out = str(tmp_path / "empty_report.html")
    path = ProfileVisualizer.generate_html_report(
        {"summary": pd.DataFrame()}, output_path=out
    )
    assert os.path.exists(path)
    html = open(path).read()
    assert "<html" in html


def test_save_all_visualizations(tmp_path, verification_df, summary_df,
                                 similarity_matrix, profiles_two_datasets,
                                 profiles_by_dataset):
    out_dir = tmp_path / "visualizations"
    metric_matrices = {"jaccard": similarity_matrix, "rank": similarity_matrix}
    directional_matrices = {"upstream": similarity_matrix,
                            "downstream": similarity_matrix}
    saved = ProfileVisualizer.save_all_visualizations(
        {"summary": summary_df},
        similarity_matrix=similarity_matrix,
        profiles={"aMe12": profiles_two_datasets},
        output_dir=str(out_dir),
        metric_matrices=metric_matrices,
        directional_matrices=directional_matrices,
        profiles_by_dataset=profiles_by_dataset,
        generate_inter_type_heatmaps=True,
    )
    assert saved
    assert "html_report" in saved
    assert os.path.exists(saved["html_report"])
    assert os.path.exists(saved["summary"])
    assert os.path.exists(saved["role_comparison"])
    assert os.path.exists(saved["heatmap_combined"])
    # metric / directional heatmaps written into subfolders
    assert (out_dir / "metric_heatmaps").is_dir()
    assert (out_dir / "directional_heatmaps").is_dir()
    assert (out_dir / "profile_charts").is_dir()


def test_save_all_visualizations_minimal(tmp_path):
    out_dir = tmp_path / "visualizations"
    saved = ProfileVisualizer.save_all_visualizations(
        {"summary": pd.DataFrame()},
        output_dir=str(out_dir),
        generate_inter_type_heatmaps=False,
    )
    assert "html_report" in saved
    assert os.path.exists(saved["html_report"])
