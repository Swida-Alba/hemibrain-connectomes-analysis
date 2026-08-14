"""Multi-dataset profiling: per-dataset extraction with name mapping,
inter-dataset comparisons of the same queried neuron (homolog backend),
and the overall report.html that embeds all heatmaps.

Also covers the CrossDatasetTypeMapper switch to the male-cns v1.0 neuron
info as the canonical name-mapping source (was v0.9).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison.connectivity_profiler import ConnectivityProfile  # noqa: E402
from comparison.cross_dataset_type_mapper import CrossDatasetTypeMapper  # noqa: E402
from comparison.profile_comparator import (  # noqa: E402
    ConnectivityProfileComparer, ProfileComparator,
)

DS_A = "hemibrain:v1.2.1"
DS_B = "male-cns:v0.9"  # normalizes to the male-cns v1.0 canonical


class FakeMapper:
    """Minimal stand-in for CrossDatasetTypeMapper."""

    _loaded = True

    def __init__(self, mapping):
        # canonical name -> {dataset: local name}
        self.mapping = mapping

    def resolve_type_across_datasets(self, name, datasets, source_dataset=None):
        for canon, per_ds in self.mapping.items():
            if name in per_ds.values() or name == canon:
                return {ds: per_ds.get(ds, name) for ds in datasets}
        return {ds: name for ds in datasets}

    def get_canonical_type(self, name, source_dataset=None):
        for canon, per_ds in self.mapping.items():
            if name == canon or name in per_ds.values():
                return canon
        return name

    def standardize_partner_types(self, partners, source_dataset):
        out = {}
        for t, w in partners.items():
            canon = self.get_canonical_type(t, source_dataset)
            out[canon] = out.get(canon, 0.0) + w
        return out


def _profile(bid, ds, upstream=None, downstream=None):
    return ConnectivityProfile(
        neuron_id=bid,
        dataset=ds,
        upstream_partners=upstream or {},
        downstream_partners=downstream or {},
        upstream_ranks={k: i + 1 for i, k in enumerate((upstream or {}))},
        downstream_ranks={k: i + 1 for i, k in enumerate((downstream or {}))},
        total_upstream_weight=float(sum((upstream or {}).values())),
        total_downstream_weight=float(sum((downstream or {}).values())),
    )


def _make_multi_comparer(**kwargs):
    params = dict(query=["aMe12"], datasets=[DS_A, DS_B], verbose=False)
    params.update(kwargs)
    comparer = ConnectivityProfileComparer(**params)
    # no network in tests
    comparer.profiler.get_bodyids_for_type = lambda t, ds: {"aMe12": [1]}.get(t, [])
    comparer.profiler.get_types_for_bodyids = lambda bids, ds: {}
    comparer.profiler.list_types = lambda p, ds: []
    return comparer


# ---------------------------------------------------------------------------
# CrossDatasetTypeMapper canonical source = male-cns v1.0
# ---------------------------------------------------------------------------

def test_type_mapper_default_path_uses_male_cns_v1_0():
    mapper = CrossDatasetTypeMapper(verbose=False)
    assert mapper._neuron_df_path.endswith(
        "male-cns_v1_0/male-cns_v1_0_allneurons_neuron_df.csv"
    )
    assert mapper._normalize_dataset_name("male-cns:v0.9") == "male-cns:v1.0"
    assert mapper._normalize_dataset_name("male_cns_v1_0") == "male-cns:v1.0"


def test_type_mapper_loads_v1_0_schema(tmp_path):
    csv = tmp_path / "neurons.csv"
    csv.write_text(
        "bodyId,type,flywireType,hemibrainType,mancType\n"
        "1,aMe12,MTe07,aMe12,\n"
        "2,aMe12,MTe07,aMe12,\n"
        "3,aMe10,MTe10,aMe10,\n",
        encoding="utf-8",
    )
    mapper = CrossDatasetTypeMapper(neuron_df_path=str(csv), verbose=False)
    assert mapper.load() is True
    # canonical keys are male-cns v1.0
    assert "male-cns:v1.0" in mapper._type_mappings
    assert mapper.get_mapped_type("aMe12", "male-cns:v1.0", "flywire_FAFB_v783") == "MTe07"
    assert mapper.get_mapped_type("MTe07", "flywire_FAFB_v783", "male-cns:v1.0") == "aMe12"
    assert mapper.get_canonical_type("MTe07", "flywire_FAFB_v783") == "aMe12"


def test_type_mapper_maps_mevplo2_to_mte07():
    """The auto name mapping must resolve the user's test pair:
    MeVPLo2 (male-cns v1.0) <-> MTe07 (flywire v783)."""
    real = CrossDatasetTypeMapper(verbose=False)
    # use the REAL male-cns v1.0 neuron info if present (local datasets)
    if real._neuron_df_path and Path(real._neuron_df_path).exists():
        assert real.load() is True
        assert real.get_mapped_type("MTe07", "flywire_FAFB_v783", "male-cns:v1.0") == "MeVPLo2"
        assert real.get_mapped_type("MeVPLo2", "male-cns:v1.0", "flywire_FAFB_v783") == "MTe07"
        assert real.get_canonical_type("MTe07", "flywire_FAFB_v783") == "MeVPLo2"
    else:
        pytest.skip("male-cns v1.0 neuron info not available locally")


def test_cross_dataset_auto_mapping_without_custom_labelmapper():
    """Cross-dataset comparison with NO custom LabelMapper: the auto type
    mapping (male-cns v1.0 neuron info) resolves the per-dataset neuron
    names — MTe07 (flywire) becomes MeVPLo2 in male-cns, aMe12 stays."""
    from comparison import ComparisonParameters

    p = ComparisonParameters(
        datasets=["flywire_FAFB_v783", "male-cns:v1.0"],
        source_neurons=["MTe07", "aMe12"],
        target_neurons=["MeVPLo2", "aMe12"],
        auto_type_mapping=True,
        verbose=False,
    )
    if p._auto_type_mapper is None or not p._auto_type_mapper._loaded:
        pytest.skip("male-cns v1.0 neuron info not available locally")
    # flywire keeps its own names
    src_fw = p.get_source_neurons_for_dataset("flywire_FAFB_v783")
    assert "MTe07" in src_fw
    # male-cns receives the mapped canonical name
    src_mcns = p.get_source_neurons_for_dataset("male-cns:v1.0")
    assert "MeVPLo2" in src_mcns
    assert "aMe12" in src_mcns
    # target side resolves the same way
    tgt_mcns = p.get_target_neurons_for_dataset("male-cns:v1.0")
    assert "MeVPLo2" in tgt_mcns


# ---------------------------------------------------------------------------
# Multi-dataset init + per-dataset name mapping
# ---------------------------------------------------------------------------

def test_multi_dataset_init():
    comparer = _make_multi_comparer()
    assert comparer.is_multi_dataset is True
    assert comparer.datasets == [DS_A, DS_B]
    assert comparer.dataset == DS_A
    assert comparer.skip_bodyId_level is True


def test_single_dataset_list_equivalent_to_dataset():
    # the UI always passes `datasets`; a one-element list = single-dataset mode
    comparer = _make_multi_comparer(datasets=[DS_A])
    assert comparer.is_multi_dataset is False
    assert comparer.dataset == DS_A
    assert comparer.datasets == [DS_A]


def test_single_dataset_report_indexes_all_metrics(tmp_path):
    comparer = _make_multi_comparer(datasets=[DS_A])
    output = tmp_path / "profiling"
    output.mkdir()
    matrix = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["a", "b"], columns=["a", "b"])
    metrics = {
        name: matrix
        for name in (
            "jaccard", "weighted_jaccard", "cosine", "rank_corr",
            "rank_corr_union", "combined",
        )
    }
    report = comparer._generate_single_dataset_report(
        output,
        {"combined": metrics},
        {},
        {},
    )
    text = report.read_text(encoding="utf-8")
    assert report.exists()
    for metric in metrics:
        assert metric in text
    assert "type_level/results/type_similarity_jaccard_combined.csv" in text


def test_report_redraws_plotly_and_links_vispath_editor(tmp_path):
    pytest.importorskip("plotly")
    comparer = _make_multi_comparer(datasets=[DS_A])
    output = tmp_path / "profiling"
    viz = output / "type_level" / "visualization"
    viz.mkdir(parents=True)
    (viz / "heatmap_type_combined_jaccard.html").write_text(
        "<!doctype html><title>VisPath editor</title>", encoding="utf-8"
    )
    matrix = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]],
        index=["aMe12", "aMe10"],
        columns=["aMe12", "aMe10"],
    )

    report = comparer._generate_single_dataset_report(
        output,
        {"combined": {"jaccard": matrix}},
        {},
        {},
    )
    text = report.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in text
    assert "<iframe" not in text
    assert "Open VisPath heatmap for editing" in text
    assert "type_level/visualization/heatmap_type_combined_jaccard.html" in text


def test_aggregate_inter_dataset_matrices_uses_neurons_by_dataset_pairs():
    comparer = _make_multi_comparer(
        datasets=[DS_A, DS_B, "flywire_FAFB_v783"],
    )
    labels = [DS_A, DS_B, "flywire_FAFB_v783"]
    first = pd.DataFrame(
        [[np.nan, 0.2, 0.3], [0.2, np.nan, 0.4], [0.3, 0.4, np.nan]],
        index=labels,
        columns=labels,
    )
    second = first + 0.1
    second.iloc[0, 0] = np.nan
    second.iloc[1, 1] = np.nan
    second.iloc[2, 2] = np.nan
    inter = {
        "aMe12": {"combined": {"jaccard": first}},
        "aMe10": {"combined": {"jaccard": second}},
    }

    aggregate = comparer._aggregate_inter_dataset_matrices(inter)
    result = aggregate["combined"]["jaccard"]
    assert result.index.tolist() == ["aMe12", "aMe10"]
    assert result.columns.tolist() == [
        f"{DS_A} vs {DS_B}",
        f"{DS_A} vs flywire_FAFB_v783",
        f"{DS_B} vs flywire_FAFB_v783",
    ]
    assert result.shape == (2, 3)
    assert result.loc["aMe12", f"{DS_A} vs {DS_B}"] == pytest.approx(0.2)
    assert result.loc["aMe10", f"{DS_B} vs flywire_FAFB_v783"] == pytest.approx(0.5)


def test_heatmap_generation_handles_nonfinite_values_and_nested_output(tmp_path):
    """Multi-dataset heatmaps must not fail on empty-profile NaNs."""
    comparer = _make_multi_comparer(generate_heatmaps=True, show_figures=False)
    matrix = pd.DataFrame(
        [[1.0, np.nan], [np.inf, 0.5]],
        index=["a", "b"],
        columns=["a", "b"],
    )
    saved = {"heatmaps_generated": []}
    viz_dir = tmp_path / "nested" / "visualization"

    comparer._generate_heatmaps_vispath(
        {"combined": {"jaccard": matrix}},
        viz_dir,
        saved,
        prefix="inter",
    )

    assert saved["heatmaps_generated"]
    assert (viz_dir / "heatmap_inter_combined_jaccard.html").exists()


def test_mapped_query_uses_type_mapper():
    comparer = _make_multi_comparer()
    comparer._type_mapper = FakeMapper({"aMe12": {DS_A: "aMe12", DS_B: "aMe12"}})
    # aMe12 is the canonical name in both datasets -> unchanged
    assert comparer._mapped_query_for(DS_A) == ["aMe12"]
    # an alternate name is mapped into each dataset's naming
    comparer2 = _make_multi_comparer(query=["MTe07"])
    comparer2._type_mapper = FakeMapper(
        {"aMe12": {DS_A: "aMe12", "flywire_FAFB_v783": "MTe07", DS_B: "aMe12"}}
    )
    assert comparer2._mapped_query_for(DS_A) == ["aMe12"]
    assert comparer2._mapped_query_for(DS_B) == ["aMe12"]


def test_extract_profiles_for_dataset_maps_names(monkeypatch):
    comparer = _make_multi_comparer()
    comparer._type_mapper = FakeMapper({"aMe12": {DS_A: "aMe12", DS_B: "aMe12"}})
    calls = []

    def fake_bodyids(t, ds):
        calls.append((t, ds))
        return {"aMe12": [1, 2]}.get(t, [])

    comparer.profiler.get_bodyids_for_type = fake_bodyids

    def fake_batch(bodyids, ds, **kw):
        return {b: _profile(b, ds, upstream={"X": 5.0}, downstream={"Y": 3.0})
                for b in bodyids if isinstance(b, int)}

    comparer.profiler.get_profiles_batch = fake_batch
    type_profiles, _ = comparer._extract_profiles_for_dataset(DS_B)
    assert set(type_profiles.keys()) == {"aMe12"}
    # the query was mapped into the dataset's naming before resolution
    assert ("aMe12", DS_B) in calls


# ---------------------------------------------------------------------------
# Inter-dataset anchors + matrices (homolog backend algorithm)
# ---------------------------------------------------------------------------

def test_build_anchor_profiles_type_names():
    comparer = _make_multi_comparer()
    comparer._type_mapper = FakeMapper({"aMe12": {DS_A: "aMe12", DS_B: "aMe12"}})
    profiles = {
        DS_A: {"aMe12": _profile("p", DS_A, upstream={"X": 5.0})},
        DS_B: {"aMe12": _profile("p", DS_B, upstream={"X": 5.0})},
    }
    anchors = comparer._build_anchor_profiles(profiles)
    assert list(anchors.keys()) == ["aMe12"]
    assert set(anchors["aMe12"].keys()) == {DS_A, DS_B}


def test_build_anchor_profiles_skips_single_dataset():
    comparer = _make_multi_comparer()
    comparer._type_mapper = FakeMapper({"aMe12": {DS_A: "aMe12", DS_B: "aMe12"}})
    profiles = {
        DS_A: {"aMe12": _profile("p", DS_A)},
        DS_B: {},  # aMe12 absent here
    }
    assert comparer._build_anchor_profiles(profiles) == {}


def test_build_anchor_profiles_custom_groups():
    comparer = _make_multi_comparer(query=[["grp1", ["aMe12", "aMe12_R"]]],
                                    aggregation_level="custom")
    profiles = {
        DS_A: {"grp1": _profile("p", DS_A)},
        DS_B: {"grp1": _profile("p", DS_B)},
    }
    anchors = comparer._build_anchor_profiles(profiles)
    assert list(anchors.keys()) == ["grp1"]
    assert anchors["grp1"][DS_A][0] == "grp1"


def test_compute_inter_dataset_matrices_standardizes_partners():
    comparer = _make_multi_comparer()
    mapper = FakeMapper({"X": {DS_A: "X", DS_B: "MTe07"}})
    comparer._type_mapper = mapper
    # same neuron, partners named differently per dataset but canonically equal
    profiles = {
        DS_A: {"aMe12": _profile("p", DS_A, upstream={"X": 10.0})},
        DS_B: {"aMe12": _profile("p", DS_B, upstream={"MTe07": 10.0})},
    }
    anchors = comparer._build_anchor_profiles(profiles)
    matrices = comparer._compute_inter_dataset_matrices(anchors)
    m = matrices["aMe12"]["combined"]["jaccard"]
    assert m.shape == (2, 2)
    assert m.index.tolist() == [DS_A, DS_B]
    # off-diagonal: standardized types overlap fully -> jaccard 1.0
    assert m.loc[DS_A, DS_B] == pytest.approx(1.0)
    # diagonal is not part of inter-dataset comparison
    assert np.isnan(m.loc[DS_A, DS_A])


def test_homolog_and_profiling_metrics_identical_for_same_pair():
    """The homolog backend (batch_compare_cross_dataset) and the profiling
    inter-dataset backend (_compute_similarity_from_types on standardized
    expanded types) produce IDENTICAL metrics for the same two profiles —
    verified numerically on real data (aMe12/hemibrain vs MeVPLo2/male-cns)."""
    mapper = FakeMapper({"X": {DS_A: "X", DS_B: "MTe07"}})
    pa = _profile("a", DS_A, upstream={"X": 10.0, "Y": 5.0}, downstream={"Z": 3.0})
    pb = _profile("b", DS_B, upstream={"MTe07": 10.0, "W": 2.0}, downstream={"Z": 3.0})

    # homolog path: per-candidate scoring
    row = ProfileComparator.batch_compare_cross_dataset(
        pa, {2: pb}, {2: 1}, "both", type_mapper=mapper
    )[0]

    # profiling path: standardized expanded types + shared scorer
    ta = ProfileComparator._get_expanded_types_standardized(pa, "both", mapper)
    tb = ProfileComparator._get_expanded_types_standardized(pb, "both", mapper)
    scores = ConnectivityProfileComparer._compute_similarity_from_types(None, ta, tb)

    assert row["combined"] == pytest.approx(scores["combined"], abs=1e-12)
    assert row["jaccard"] == pytest.approx(scores["jaccard"], abs=1e-12)
    assert row["weighted_jaccard"] == pytest.approx(scores["weighted_jaccard"], abs=1e-12)
    assert row["cosine"] == pytest.approx(scores["cosine"], abs=1e-12)
    # both paths yield the same rank (or NaN when < 3 shared types)
    both_nan = np.isnan(row["rank"]) and np.isnan(scores["rank"])
    assert both_nan or row["rank"] == pytest.approx(scores["rank"], abs=1e-12)
    # rank_union (union-based rank) matches too — the profiling matrices'
    # rank_corr_union cell stores the SAME raw value
    assert row["rank_union"] == pytest.approx(scores["rank_union"], abs=1e-12)


# ---------------------------------------------------------------------------
# aMe12 / MeVPLo2(MTe07): intra- and inter-dataset scenarios for BOTH tools
# ---------------------------------------------------------------------------

MC = "male-cns:v1.0"
FW = "flywire_FAFB_v783"


def _mevplo2_mapper():
    """Male-cns v1.0 canonical mapping: aMe12 is aMe12 everywhere;
    MeVPLo2 (male-cns) is MTe07 in flywire v783."""
    return FakeMapper({
        "aMe12": {MC: "aMe12", FW: "aMe12"},
        "MeVPLo2": {MC: "MeVPLo2", FW: "MTe07"},
    })


def test_intra_dataset_aMe12_vs_MeVPLo2_six_metrics():
    """Intra-dataset (within male-cns v1.0): the aMe12 × MeVPLo2 cell of the
    type-level matrix carries the same six metrics as the homolog scorer for
    the same two profiles."""
    comparer = _make_multi_comparer(datasets=[MC], query=["aMe12", "MeVPLo2"])
    pa = _profile("pa", MC, upstream={"X": 10.0, "Y": 5.0}, downstream={"Z": 3.0})
    pm = _profile("pm", MC, upstream={"X": 10.0, "W": 2.0}, downstream={"Z": 3.0})
    profiles = {"aMe12": pa, "MeVPLo2": pm}
    matrices = comparer._compute_similarity_matrices(profiles)
    cell = matrices["combined"]["jaccard"].loc["aMe12", "MeVPLo2"]
    # same pair through the homolog scorer (intra-dataset: no mapper)
    scores = ProfileComparator.combined_score(pa, pm, direction="both")
    assert cell == pytest.approx(scores["jaccard"], abs=1e-12)
    assert matrices["combined"]["weighted_jaccard"].loc["aMe12", "MeVPLo2"] == \
        pytest.approx(scores["weighted_jaccard"], abs=1e-12)
    assert matrices["combined"]["cosine"].loc["aMe12", "MeVPLo2"] == \
        pytest.approx(scores["cosine"], abs=1e-12)
    assert matrices["combined"]["combined"].loc["aMe12", "MeVPLo2"] == \
        pytest.approx(scores["combined"], abs=1e-12)
    # rank_corr_union is the RAW union rank (sign meaningful, 0 = neutral)
    assert matrices["combined"]["rank_corr_union"].loc["aMe12", "MeVPLo2"] == \
        pytest.approx(scores["rank_union"], abs=1e-12)
    # every metric matrix is exported
    for m in ("jaccard", "weighted_jaccard", "cosine", "rank_corr",
              "rank_corr_union", "combined"):
        assert m in matrices["combined"]


def test_inter_dataset_mevplo2_anchor_matches_homolog_scorer():
    """Inter-dataset: query ['aMe12', 'MeVPLo2'] over male-cns v1.0 + flywire
    v783 — the MeVPLo2 anchor spans MeVPLo2(male-cns) × MTe07(flywire) via the
    auto name mapping, and its cell equals the homolog scorer for the pair."""
    mapper = _mevplo2_mapper()
    comparer = _make_multi_comparer(datasets=[MC, FW], query=["aMe12", "MeVPLo2"])
    comparer._type_mapper = mapper

    p_ma = _profile("p_ma", MC, upstream={"X": 10.0, "Y": 5.0}, downstream={"Z": 3.0})
    p_mm = _profile("p_mm", MC, upstream={"X": 10.0, "W": 2.0}, downstream={"Z": 3.0})
    p_fa = _profile("p_fa", FW, upstream={"X": 9.0, "Y": 4.0}, downstream={"Z": 3.0})
    p_ft = _profile("p_ft", FW, upstream={"X": 10.0, "W": 2.0}, downstream={"Z": 3.0})
    profiles_by_dataset = {
        MC: {"aMe12": p_ma, "MeVPLo2": p_mm},
        FW: {"aMe12": p_fa, "MTe07": p_ft},
    }

    anchors = comparer._build_anchor_profiles(profiles_by_dataset)
    # both anchors span the two datasets (MeVPLo2 <-> MTe07 via the mapper)
    assert set(anchors.keys()) == {"aMe12", "MeVPLo2"}
    assert set(anchors["MeVPLo2"].keys()) == {MC, FW}
    assert anchors["MeVPLo2"][FW][0] == "MTe07"

    matrices = comparer._compute_inter_dataset_matrices(anchors)
    m = matrices["MeVPLo2"]["combined"]
    cell_j = m["jaccard"].loc[MC, FW]
    cell_wj = m["weighted_jaccard"].loc[MC, FW]
    cell_cos = m["cosine"].loc[MC, FW]
    cell_comb = m["combined"].loc[MC, FW]
    cell_ru = m["rank_corr_union"].loc[MC, FW]

    # the homolog scorer on the SAME profiles (flywire name standardized)
    row = ProfileComparator.batch_compare_cross_dataset(
        p_mm, {999: p_ft}, {999: 1}, "both", type_mapper=mapper
    )[0]
    assert cell_j == pytest.approx(row["jaccard"], abs=1e-12)
    assert cell_wj == pytest.approx(row["weighted_jaccard"], abs=1e-12)
    assert cell_cos == pytest.approx(row["cosine"], abs=1e-12)
    assert cell_comb == pytest.approx(row["combined"], abs=1e-12)
    assert cell_ru == pytest.approx(row["rank_union"], abs=1e-12)
    assert m["rank_corr"].loc[MC, FW] == pytest.approx(row["rank"], abs=1e-12)


# ---------------------------------------------------------------------------
# run(): output folder + overall report.html
# ---------------------------------------------------------------------------

def test_run_multi_dataset_output_structure(tmp_path, monkeypatch):
    comparer = _make_multi_comparer()
    comparer.output_dir = str(tmp_path)
    fake_mapper = FakeMapper({"aMe12": {DS_A: "aMe12", DS_B: "aMe12"}})
    comparer._type_mapper = fake_mapper

    # run() loads the mapper through the module-level get_type_mapper()
    import comparison.profile_comparator as pc_mod
    monkeypatch.setattr(pc_mod, "get_type_mapper", lambda *a, **k: fake_mapper)

    pa = _profile("p", DS_A, upstream={"X": 10.0})
    pb = _profile("p", DS_B, upstream={"MTe07": 10.0})

    def fake_extract(ds):
        profile = pa if ds == DS_A else pb
        return ({"aMe12": profile, "x": profile}, {})

    monkeypatch.setattr(comparer, "_extract_profiles_for_dataset", fake_extract)
    monkeypatch.setattr(comparer, "_compute_similarity_matrices",
                        lambda profiles: {"combined": {"jaccard": pd.DataFrame(
                            [[1.0, 0.5], [0.5, 1.0]],
                            index=["aMe12", "x"], columns=["aMe12", "x"])}})
    # keep the real save path but stub heatmap generation (no vispath needed)
    monkeypatch.setattr(comparer, "_generate_heatmaps_vispath",
                        lambda matrices, viz_dir, saved, prefix: None)

    result = comparer.run()
    assert result["is_multi_dataset"] is True
    assert result["report_path"]

    out = Path(result["output_path"])
    assert (out / "report.html").exists()
    assert (out / "parameters.json").exists()
    assert (out / "README.txt").exists()
    # reorganized intra / inter / profiles folders
    safe_a = DS_A.replace(":", "_").replace(".", "_")
    safe_b = DS_B.replace(":", "_").replace(".", "_")
    assert (out / "intra_dataset" / safe_a).exists()
    assert (out / "intra_dataset" / safe_a / "visualization").exists()
    assert (out / "cross_dataset" / "mapping_summary.csv").exists()
    assert (out / "cross_dataset" / "all_types" / "results").exists()
    assert (out / "cross_dataset" / "all_types" / "results" /
            "similarity_combined_jaccard.csv").exists()
    assert (out / "cross_dataset" / "per_neuron" / "aMe12" / "results").exists()
    assert (out / "cross_dataset" / "per_neuron" / "aMe12" / "visualization").exists()
    assert (out / "profiles" / safe_b / "aggregated").exists()

    # the report indexes every matrix (and embeds heatmaps when generated)
    report = (out / "report.html").read_text(encoding="utf-8")
    assert "Intra-dataset" in report
    assert "Inter-dataset" in report
    assert "cross_dataset/all_types/results/similarity_combined_jaccard.csv" in report
    assert "Dataset pair" in report
    assert "<iframe" not in report
    assert "similarity_combined_jaccard.csv" in report
    assert "mapping_summary" not in report  # summary is a CSV, not embedded
    params = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    assert params["datasets"] == [DS_A, DS_B]
    assert params["aggregation_level"] == "type"
