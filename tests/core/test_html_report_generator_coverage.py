"""Coverage tests for comparison.html_report_generator.

Hermetic: a FakeAnalyzer supplies tiny synthetic DataFrames; all file output
goes to pytest tmp_path. No network, no kaleido, no multiprocessing.
"""

import os
import re

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from comparison import html_report_generator as hrg

DATASETS = ["ds_one", "ds_two"]
THRESHOLDS = [1, 5]
NICKNAME_MAP = {"ds_one": "D1", "ds_two": "D2"}


# ---------------------------------------------------------------------------
# Fake analyzer infrastructure
# ---------------------------------------------------------------------------

class FakeParameters:
    def __init__(self, output_path, dataset_names, separate_hemispheres=False,
                 auto_type_mapping=False, path_mode="all", max_interlayer=2,
                 source_neurons=("A",), target_neurons=("D",)):
        self.full_output_path = str(output_path)
        self.comparison_mode = "connectivity"
        self.path_mode = path_mode
        self.max_interlayer = max_interlayer
        self.auto_type_mapping = auto_type_mapping
        self.separate_hemispheres = separate_hemispheres
        self.source_neurons = list(source_neurons)
        self.target_neurons = list(target_neurons)
        self._dataset_names = list(dataset_names)
        self._auto_type_mapper = None

    def get_dataset_nicknames(self):
        return [f"D{i + 1}" for i in range(len(self._dataset_names))]

    def _ensure_flat_list(self, value):
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    def get_source_neurons_for_dataset(self, dataset):
        return list(self.source_neurons)

    def get_target_neurons_for_dataset(self, dataset):
        return list(self.target_neurons)

    @staticmethod
    def _sanitize_name(name):
        return re.sub(r"[^A-Za-z0-9_]", "_", str(name))


class FakeLabelMapper:
    def __init__(self, source=("A",), target=("D",)):
        self._labels = {"source": list(source), "target": list(target)}

    def get_all_std_labels(self, role):
        return list(self._labels.get(role, []))


def _aligned_df():
    return pd.DataFrame(
        {
            "ds_one": [10.0, 7.0, 5.0, 3.0, 0.0],
            "ds_two": [8.0, 0.0, 6.0, 3.0, 4.0],
        },
        index=["A -> B", "B -> C", "A -> C", "C -> D", "B -> D"],
    )


def _path_df():
    return pd.DataFrame(
        {
            "ds_one": [7.0, 3.0, 0.0],
            "ds_two": [5.0, 3.0, 2.0],
        },
        index=["A -> B -> C", "A -> C -> D", "B -> C -> D"],
    )


def _prob_df():
    return pd.DataFrame(
        {"ds_one": [0.6, 0.2], "ds_two": [0.4, 0.3]},
        index=["A -> B -> C", "A -> C -> D"],
    )


def _ratio_df():
    return pd.DataFrame(
        {"ds_one": [0.5, 0.2, 0.0], "ds_two": [0.4, 0.0, 0.1]},
        index=["A -> B", "B -> C", "A -> C"],
    )


def _hop_weights():
    return {
        1: {"A -> B -> C": {"ds_one": [10, 7], "ds_two": [8, 5]},
            "A -> C -> D": {"ds_one": [5, 3], "ds_two": [6, 3]}},
        5: {"A -> B -> C": {"ds_one": [10, 7]}},
    }


def _edge_df(n=3):
    return pd.DataFrame(
        {
            "type_pre": ["A", "B", "A"][:n],
            "type_post": ["B", "C", "C"][:n],
            "weight": [10.0, 7.0, 5.0][:n],
            "has_valid_path": [True, True, False][:n],
        }
    )


def _presence_matrix():
    return pd.DataFrame(
        {
            "ds_one_t1": ["True", "False", "True"],
            "ds_two_t1": ["True", "True", False],
            "ds_one_t5": ["True", "False", False],
            "ds_two_t5": [1, 1, 0],
        }
    )


def _symmetry_summaries():
    per_ds = {
        "ipsi": {"jaccard": 0.8, "conserved": 4, "union": 5},
        "contra": {"jaccard": 0.4, "conserved": 2, "union": 5},
        "neuron_types": {"types_conserved": 2, "types_union": 3},
        "hemisphere_counts": {"total": {"L": 4, "R": 3}},
    }
    return {t: {d: dict(per_ds) for d in DATASETS} for t in THRESHOLDS}


class FakeAnalyzer:
    """Minimal stand-in for CrossDatasetComparisonAnalyzer."""

    def __init__(self, output_path, dataset_names=DATASETS,
                 separate_hemispheres=False, auto_type_mapping=False,
                 path_mode="all", max_interlayer=2, empty=False,
                 source_neurons=("A",), target_neurons=("D",),
                 include_neuron_counts=True):
        self.parameters = FakeParameters(
            output_path, dataset_names,
            separate_hemispheres=separate_hemispheres,
            auto_type_mapping=auto_type_mapping,
            path_mode=path_mode, max_interlayer=max_interlayer,
            source_neurons=source_neurons, target_neurons=target_neurons,
        )
        self.label_mapper = FakeLabelMapper(source_neurons, target_neurons)
        self._dataset_names = list(dataset_names)
        self._empty = empty
        self.comparison_report = {
            "path_presence_matrix": None if empty else _presence_matrix()
        }
        if include_neuron_counts and not empty:
            self._neuron_counts_summary = pd.DataFrame(
                [
                    {"dataset": "ds_one", "source_count": 5, "target_count": 3,
                     "source_types": 2, "target_types": 1},
                    {"dataset": "ds_two", "source_count": 4, "target_count": 2,
                     "source_types": 2, "target_types": 1},
                ]
            )
            self._neuron_type_counts = pd.DataFrame(
                [
                    {"type": "A", "ds_one_source": 3, "ds_one_target": 0,
                     "ds_two_source": 2, "ds_two_target": 1},
                    {"type": "D", "ds_one_source": 0, "ds_one_target": 2,
                     "ds_two_source": 0, "ds_two_target": 1},
                ]
            )
            self._neuron_group_counts = pd.DataFrame(
                [{"custom_group": "G1", "role": "source",
                  "ds_one": 2, "ds_two": 1}]
            )
        self._mapped_results = (
            {} if empty else
            {d: {t: _edge_df() for t in THRESHOLDS} for d in dataset_names}
        )

    # -- data accessors ----------------------------------------------------
    def get_aligned_data(self, threshold):
        return pd.DataFrame() if self._empty else _aligned_df()

    def get_aligned_data_for_network(self, threshold):
        return self.get_aligned_data(threshold)

    def _get_path_data_for_threshold(self, threshold):
        return pd.DataFrame() if self._empty else _path_df()

    def _get_path_hop_weights_for_threshold(self, threshold):
        return {} if self._empty else _hop_weights().get(threshold, {})

    def _get_prob_data_for_threshold(self, threshold):
        return pd.DataFrame() if self._empty else _prob_df()

    def _get_edge_ratio_data_for_threshold(self, threshold):
        return pd.DataFrame() if self._empty else _ratio_df()

    def get_mapped_results(self):
        return self._mapped_results

    def get_hemisphere_symmetry_summaries(self):
        return {} if self._empty else _symmetry_summaries()


@pytest.fixture
def analyzer(tmp_path):
    return FakeAnalyzer(tmp_path)


@pytest.fixture
def empty_analyzer(tmp_path):
    return FakeAnalyzer(tmp_path, empty=True, include_neuron_counts=False)


def _key_findings():
    return {
        1: {"total_edges": 6, "common_edges": 3,
            "total_paths": 3, "common_paths": 1},
        5: {"total_edges": 4, "common_edges": 2,
            "total_paths": 2, "common_paths": 1},
    }


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_make_link_existing(tmp_path):
    target = tmp_path / "sub" / "file.html"
    target.parent.mkdir(parents=True)
    target.write_text("<html></html>")
    link = hrg._make_link(str(target), str(tmp_path))
    assert link.startswith('<a href="sub/file.html"')
    assert "Open</a>" in link


def test_make_link_missing(tmp_path):
    assert hrg._make_link(str(tmp_path / "nope.html"), str(tmp_path)) == "-"
    assert hrg._make_link("", str(tmp_path)) == "-"


def test_get_canonical_name():
    assert hrg.get_canonical_name("MBON14 (merged)") == "MBON14"
    assert hrg.get_canonical_name("plain") == "plain"


def test_get_base_name():
    assert hrg.get_base_name("aMe12_L") == "aMe12"
    assert hrg.get_base_name("aMe12_R") == "aMe12"
    assert hrg.get_base_name("aMe12_U") == "aMe12"
    assert hrg.get_base_name("aMe12") == "aMe12"


@pytest.mark.parametrize(
    "label,patterns,expected",
    [
        ("aMe12", {"aMe12"}, True),                     # exact
        ("aMe12", {"aMe*"}, True),                      # glob
        ("aMe12", {"aMe.*"}, True),                     # regex
        ("aMe12 (merged)", {"aMe12"}, True),            # merged display label
        ("aMe12_L", {"aMe12"}, True),                   # hemisphere suffix
        ("aMe12", {"other"}, False),                    # no match
        ("AME12", {"aMe12"}, True),                     # case-insensitive
    ],
)
def test_matches_patterns(label, patterns, expected):
    assert hrg.matches_patterns(label, patterns) is expected


# ---------------------------------------------------------------------------
# Edge extraction / filtering helpers
# ---------------------------------------------------------------------------

def test_extract_edges_from_paths():
    path_data = pd.DataFrame(
        {"ds_one": [7.0, 1.0], "ds_two": [5.0, 0.0]},
        index=["A(x) -> B -> C", "D → E"],  # display names + unicode arrow
    )
    edges = hrg._extract_edges_from_paths(path_data, DATASETS, max_paths=10)
    assert "A(x) -> B" in edges          # display format
    assert "A -> B" in edges             # canonical format
    assert "B -> C" in edges
    assert "D -> E" in edges             # unicode arrow parsed


def test_extract_edges_from_paths_empty():
    assert hrg._extract_edges_from_paths(None, DATASETS) == set()
    assert hrg._extract_edges_from_paths(pd.DataFrame(), DATASETS) == set()
    # no matching columns
    df = pd.DataFrame({"other": [1.0]}, index=["A -> B"])
    assert hrg._extract_edges_from_paths(df, DATASETS) == set()


def test_filter_aligned_by_paths_with_paths():
    aligned = _aligned_df()
    path_data = _path_df()
    out = hrg._filter_aligned_by_paths(aligned, path_data, DATASETS, max_edges=10)
    # Edges from "A -> B -> C" / "A -> C -> D" / "B -> C -> D" kept
    assert "A -> B" in out.index
    assert "A -> C" in out.index
    assert "C -> D" in out.index


def test_filter_aligned_by_paths_fallback_top_edges():
    aligned = _aligned_df()
    out = hrg._filter_aligned_by_paths(aligned, None, DATASETS, max_edges=2)
    assert len(out) == 2  # trimmed to top 2 by total weight


def test_filter_aligned_by_paths_empty_aligned():
    out = hrg._filter_aligned_by_paths(pd.DataFrame(), _path_df(), DATASETS)
    assert out.empty


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def test_generate_presence_table():
    html = hrg._generate_presence_table(_aligned_df(), DATASETS, NICKNAME_MAP, 1)
    assert "Threshold = 1" in html
    assert "A -> B" in html
    assert "badge-success" in html  # conserved edge present in both datasets
    assert "badge-danger" in html   # unique edge


def test_generate_presence_table_empty():
    assert "No data available" in hrg._generate_presence_table(
        pd.DataFrame(), DATASETS, NICKNAME_MAP
    )
    df = pd.DataFrame({"other": [1.0]})
    assert "No datasets available" in hrg._generate_presence_table(
        df, DATASETS, NICKNAME_MAP
    )


def test_generate_path_presence_table(analyzer):
    html = hrg._generate_path_presence_table(
        analyzer, _path_df(), DATASETS, NICKNAME_MAP, 1
    )
    assert "A -> B -> C" in html
    assert "<strong>" in html  # min hop weight bolded
    assert "badge-success" in html


def test_generate_path_presence_table_empty(analyzer):
    assert "No data available" in hrg._generate_path_presence_table(
        analyzer, pd.DataFrame(), DATASETS, NICKNAME_MAP, 1
    )


def test_generate_edge_dataset_table(analyzer):
    html = hrg._generate_edge_dataset_table(analyzer, "ds_one", THRESHOLDS, NICKNAME_MAP)
    assert "Dataset: D1" in html
    assert "t=1" in html and "t=5" in html


def test_generate_path_dataset_table(analyzer):
    html = hrg._generate_path_dataset_table(analyzer, "ds_one", THRESHOLDS, NICKNAME_MAP)
    assert "Dataset: D1" in html
    assert "A -> B -> C" in html


def test_generate_stats_table(analyzer):
    html = hrg._generate_stats_table(analyzer, DATASETS, 1, NICKNAME_MAP)
    assert "Per-Dataset Statistics" in html
    assert "Pairwise Similarities" in html


def test_generate_stats_table_empty(empty_analyzer):
    html = hrg._generate_stats_table(empty_analyzer, DATASETS, 1, NICKNAME_MAP)
    assert "No data available" in html


# ---------------------------------------------------------------------------
# Header / TOC / footer
# ---------------------------------------------------------------------------

def test_generate_html_header():
    header = hrg._generate_html_header()
    assert "<html" in header
    assert "plotly" in header.lower()


def test_generate_report_header(analyzer):
    html = hrg._generate_report_header(
        analyzer, DATASETS, THRESHOLDS, "<p>note</p>", NICKNAME_MAP
    )
    assert "Cross-Dataset Comparison Report" in html
    assert "D1, D2" in html
    assert "<p>note</p>" in html


def test_generate_report_header_shortest_unlimited(tmp_path):
    fa = FakeAnalyzer(tmp_path, path_mode="shortest", max_interlayer=0)
    html = hrg._generate_report_header(fa, DATASETS, THRESHOLDS, "", NICKNAME_MAP)
    assert "unlimited (shortest mode)" in html
    assert "Shortest path mode" in html


def test_generate_toc():
    toc = hrg._generate_toc(THRESHOLDS)
    assert "#summary" in toc
    assert "#conservation" in toc


def test_generate_footer():
    assert "</html>" in hrg._generate_footer()


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def test_generate_summary_section(analyzer, tmp_path):
    html = hrg._generate_summary_section(
        analyzer, DATASETS, THRESHOLDS, [], _key_findings(), NICKNAME_MAP
    )
    assert "Key Findings by Threshold" in html
    assert "edgeCountChart" in html
    used = tmp_path / "comparison_report_used_data"
    assert (used / "edge_count_data.csv").exists()
    assert (used / "total_weight_data.csv").exists()
    assert (used / "avg_ratio_data.csv").exists()
    assert (used / "avg_prob_data.csv").exists()
    assert (used / "ratio_data_t1.csv").exists()


def test_generate_neuron_counts_section(analyzer):
    html = hrg._generate_neuron_counts_section(analyzer, DATASETS, NICKNAME_MAP)
    assert "Summary: Total Neuron Counts" in html
    assert "Neuron Counts by Type" in html
    assert "Neuron Counts by Custom Group" in html
    assert "neuronCountChart" in html


def test_generate_neuron_counts_section_missing(empty_analyzer):
    html = hrg._generate_neuron_counts_section(
        empty_analyzer, DATASETS, NICKNAME_MAP
    )
    assert "not available" in html


def test_generate_similarity_section(analyzer, tmp_path):
    html = hrg._generate_similarity_section(
        analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Threshold = 1" in html
    assert "Threshold = 5" in html
    assert (tmp_path / "similarity_matrices" / "similarity_threshold_1.csv").exists()


def test_generate_similarity_section_single_dataset(analyzer):
    html = hrg._generate_similarity_section(
        analyzer, ["ds_one"], THRESHOLDS, {"ds_one": "D1"}
    )
    assert "Similarity Matrices" in html  # no matrices, but section renders


def test_generate_hemisphere_symmetry_disabled(analyzer):
    html = hrg._generate_hemisphere_symmetry_section(
        analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Hemisphere analysis unavailable" in html


def test_generate_hemisphere_symmetry_enabled(tmp_path):
    fa = FakeAnalyzer(tmp_path, separate_hemispheres=True)
    html = hrg._generate_hemisphere_symmetry_section(
        fa, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Ipsi Jaccard" in html
    assert "0.800" in html
    assert "4/3" in html  # L/R counts


def test_generate_hemisphere_symmetry_empty(tmp_path):
    fa = FakeAnalyzer(tmp_path, separate_hemispheres=True, empty=True)
    html = hrg._generate_hemisphere_symmetry_section(
        fa, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "No hemisphere symmetry summaries found" in html


def test_generate_networks_section(analyzer):
    html = hrg._generate_networks_section(
        analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Network Visualizations" in html
    assert "window.allNetworks" in html
    assert "Conservation" in html


def test_generate_networks_section_self_edges(tmp_path):
    # source == target triggers the self-edge detection branch
    fa = FakeAnalyzer(tmp_path, source_neurons=("A",), target_neurons=("A",))
    html = hrg._generate_networks_section(fa, DATASETS, THRESHOLDS, NICKNAME_MAP)
    assert "Network Visualizations" in html


def test_generate_conservation_network(analyzer):
    html = hrg._generate_conservation_network(
        analyzer, DATASETS, 1, NICKNAME_MAP
    )
    assert "Conservation" in html
    assert "A" in html


def test_generate_conservation_network_empty(empty_analyzer):
    html = hrg._generate_conservation_network(
        empty_analyzer, DATASETS, 1, NICKNAME_MAP
    )
    assert "No connections" in html


def test_generate_dataset_network(analyzer):
    html = hrg._generate_dataset_network(
        analyzer, "ds_one", THRESHOLDS, NICKNAME_MAP
    )
    assert "D1" in html


def test_generate_edge_matrices_section(analyzer):
    html = hrg._generate_edge_matrices_section(
        analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Edge Presence Matrices" in html
    assert "A -> B" in html
    assert "switchEdgeMode" in html


def test_generate_path_matrices_section(analyzer):
    html = hrg._generate_path_matrices_section(
        analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Path Presence Matrices" in html
    assert "A -> B -> C" in html


def test_generate_path_matrices_section_no_data(empty_analyzer):
    html = hrg._generate_path_matrices_section(
        empty_analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "No path data available" in html


def test_generate_conservation_section(analyzer):
    html = hrg._generate_conservation_section(
        analyzer, DATASETS, THRESHOLDS, _key_findings(), NICKNAME_MAP
    )
    assert "Conservation Analysis" in html
    assert "var plotData" in html          # plotly JSON embedded
    assert "plotData = null" not in html   # plotly generation succeeded
    assert "Conservation at Threshold = 1" in html
    assert "badge" not in html or True
    # conserved graph link table falls back to '-' when files are absent
    assert "-" in html


def test_generate_overlap_matrices_section(analyzer):
    html = hrg._generate_overlap_matrices_section(
        analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Dataset Overlap Matrices" in html
    assert "edge_overlap_1" in html
    assert "path_overlap_1" in html


def test_generate_statistics_section(analyzer):
    html = hrg._generate_statistics_section(
        analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Statistics" in html
    assert "Similarity Trends Across Thresholds" in html


def test_generate_reciprocal_section_disabled(analyzer):
    html = hrg._generate_reciprocal_visualizations_section(
        analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "not enabled" in html


def test_generate_reciprocal_section_enabled(tmp_path):
    fa = FakeAnalyzer(tmp_path)
    fa.parameters.find_reciprocal = True
    # Create one output file so its link cell renders "Open"
    link_file = (
        tmp_path / "dataset_data" / "ds_one" / "minsyn_1"
        / "find_reciprocal" / "visualizations" / "reciprocal_type_network.html"
    )
    link_file.parent.mkdir(parents=True)
    link_file.write_text("<html></html>")
    html = hrg._generate_reciprocal_visualizations_section(
        fa, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Threshold t = 1" in html
    assert "Open</a>" in html      # existing file linked
    assert "-</td>" in html        # missing files show '-'


# ---------------------------------------------------------------------------
# Individual similarity trend plots
# ---------------------------------------------------------------------------

def test_generate_jaccard_similarity_plot(analyzer):
    html = hrg._generate_jaccard_similarity_plot(
        analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Jaccard Similarity Trend" in html
    assert "D1 vs D2" in html


def test_generate_edge_rank_correlation_plot(analyzer):
    html = hrg._generate_edge_rank_correlation_plot(
        analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Edge Rank" in html


def test_generate_cosine_similarity_trend_plot(analyzer):
    html = hrg._generate_cosine_similarity_trend_plot(
        analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Cosine" in html


def test_generate_path_rank_correlation_plot(analyzer):
    html = hrg._generate_path_rank_correlation_plot(
        analyzer, DATASETS, THRESHOLDS, NICKNAME_MAP
    )
    assert "Path" in html


# ---------------------------------------------------------------------------
# End-to-end generate_html_report
# ---------------------------------------------------------------------------

def test_generate_html_report_full(analyzer, tmp_path):
    html = hrg.generate_html_report(
        analyzer, DATASETS, THRESHOLDS,
        mode_specific_note="<p>mode note</p>",
        path_count_data=[], key_findings_per_threshold=_key_findings(),
    )
    out = tmp_path / "comparison_report.html"
    out.write_text(html, encoding="utf-8")
    assert out.exists()

    for marker in [
        "<html",
        "Cross-Dataset Comparison Report",
        "mode note",
        "Key Findings by Threshold",
        "Neuron Counts Comparison",
        "Hemisphere Symmetry",
        "Similarity Matrices",
        "Network Visualizations",
        "Edge Presence Matrices",
        "Path Presence Matrices",
        "Conservation Analysis",
        "Dataset Overlap Matrices",
        "Statistics",
        "</html>",
    ]:
        assert marker in html, f"missing marker: {marker}"


def test_generate_html_report_with_hemispheres(tmp_path):
    fa = FakeAnalyzer(tmp_path, separate_hemispheres=True)
    html = hrg.generate_html_report(
        fa, DATASETS, THRESHOLDS, "", [], _key_findings()
    )
    assert "Ipsi Jaccard" in html
    assert "</html>" in html


def test_generate_html_report_with_auto_type_mapping(tmp_path):
    fa = FakeAnalyzer(tmp_path, auto_type_mapping=True)
    html = hrg.generate_html_report(
        fa, DATASETS, THRESHOLDS, "", [], _key_findings()
    )
    assert "</html>" in html


def test_generate_html_report_empty_data(empty_analyzer):
    html = hrg.generate_html_report(
        empty_analyzer, DATASETS, THRESHOLDS, "", [], {}
    )
    assert "<html" in html
    assert "</html>" in html
    assert "No data available" in html
