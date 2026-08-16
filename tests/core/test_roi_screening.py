"""Tests for src/roi_screening.py — ROI-distribution candidate screening."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import roi_screening as rois  # noqa: E402


PRIMARY = ["A(L)", "A(R)", "M"]   # two hemis + one midline ROI


def write_roi_dataset(tmp_path, dataset, counts, neuron_df=None,
                      metadata_rois=None):
    """Write a minimal local dataset (ROI table, neuron table, sidecar).

    ``counts``: {bodyId: {"post": {roi: n}, "pre": {roi: n}}}. The neuron
    table's pre/post totals are the ROI sums, so the primary list is a real
    partition. ``metadata_rois`` overrides the sidecar's roi_list (to test
    hierarchical/invalid lists).
    """
    folder = rois._dataset_folder(dataset)
    base = tmp_path / "datasets" / folder
    base.mkdir(parents=True, exist_ok=True)
    rows = []
    neuron_rows = []
    for bid, blocks in counts.items():
        pre_map = blocks.get("pre", {})
        post_map = blocks.get("post", {})
        for roi in set(pre_map) | set(post_map):
            rows.append({"bodyId": bid, "roi": roi,
                         "pre": pre_map.get(roi, 0),
                         "post": post_map.get(roi, 0)})
        neuron_rows.append({
            "bodyId": bid,
            "type": blocks.get("type", ""),
            "instance": blocks.get("instance", ""),
            "pre": sum(pre_map.values()),
            "post": sum(post_map.values()),
        })
    pd.DataFrame(rows).to_csv(base / f"{folder}_allneurons_roi_count_df.csv",
                              index=False)
    if neuron_df is not None:
        neuron_rows = neuron_df
    pd.DataFrame(neuron_rows).to_csv(
        base / f"{folder}_allneurons_neuron_df.csv", index=False)
    rois_list = PRIMARY if metadata_rois is None else metadata_rois
    meta = {"dataset": dataset, "source": "neuprint",
            "roi_coverage": {"roi_list": rois_list, "roi_count": len(rois_list),
                             "neuron_counts_per_roi": {}}}
    (base / f"{folder}_metadata.json").write_text(json.dumps(meta))
    return base


def twin_fixture(tmp_path, dataset="np:v1"):
    """A mirrored twin pair, a same-hemisphere decoy and a zero neuron."""
    return write_roi_dataset(tmp_path, dataset, {
        # query (right hemisphere) and its contralateral twin
        1: {"post": {"A(R)": 10, "M": 5}, "pre": {"A(R)": 8, "M": 4},
            "type": "T", "instance": "T_R"},
        2: {"post": {"A(L)": 10, "M": 5}, "pre": {"A(L)": 8, "M": 4},
            "type": "T2", "instance": "T2_L"},
        # same hemisphere as the query but very different midline emphasis
        3: {"post": {"A(R)": 10, "M": 50}, "pre": {"A(R)": 8, "M": 40},
            "type": "D", "instance": "D_R"},
        # no synapses at all: never a candidate
        4: {"post": {}, "pre": {}, "type": "Z", "instance": "Z"},
    })


def hierarchical_fixture(tmp_path, dataset="np:v1"):
    """ROI table whose rows also carry a parent ROI ('A') double-counting
    both hemispheres, and a sidecar listing parent + children."""
    write_roi_dataset(
        tmp_path, dataset,
        {
            1: {"post": {"A(L)": 10, "A(R)": 2, "A": 12},
                "pre": {"A(R)": 8, "A": 8}, "type": "T"},
            2: {"post": {"A(L)": 10, "A(R)": 2, "A": 12},
                "pre": {"A(L)": 8, "A": 8}, "type": "T2"},
        },
        neuron_df=[
            # true totals: the parent ROI re-counts the same synapses
            {"bodyId": 1, "type": "T", "instance": "T_R", "pre": 8, "post": 12},
            {"bodyId": 2, "type": "T2", "instance": "T2_L", "pre": 8, "post": 12},
        ],
        metadata_rois=["A", "A(L)", "A(R)", "M"],
    )


class TestMirrorPermutation:
    def test_l_r_swap_and_midline_fixed(self):
        perm = rois.mirror_permutation(["A(L)", "M", "A(R)"])
        assert perm.tolist() == [2, 1, 0]

    def test_missing_mirror_is_dead_column(self):
        perm = rois.mirror_permutation(["A(L)", "M"])
        assert perm.tolist() == [-1, 1]


class TestValidation:
    def test_primary_list_passes(self, tmp_path):
        twin_fixture(tmp_path)
        assert rois.validate_primary_rois(PRIMARY, "np:v1", str(tmp_path))

    def test_hierarchical_list_fails(self, tmp_path):
        hierarchical_fixture(tmp_path)
        assert not rois.validate_primary_rois(
            ["A", "A(L)", "A(R)", "M"], "np:v1", str(tmp_path))

    def test_load_primary_rois_reads_sidecar(self, tmp_path):
        twin_fixture(tmp_path)
        assert rois.load_primary_rois("np:v1", str(tmp_path)) == PRIMARY

    def test_load_primary_rois_missing(self, tmp_path):
        assert rois.load_primary_rois("np:v1", str(tmp_path)) is None


class TestRoiProfileStore:
    def test_twin_ranks_first_with_mirroring(self, tmp_path):
        """Regression: without the cross-orientation score the contralateral
        twin is near-orthogonal (hemisphere-suffixed ROIs) and ranks low."""
        twin_fixture(tmp_path)
        store = rois.RoiProfileStore("np:v1", project_root=str(tmp_path))
        store.ensure()
        res = store.screen([1])
        assert int(res.iloc[0]["bodyId"]) == 2
        assert res.iloc[0]["roi_similarity"] == pytest.approx(1.0, abs=1e-5)

    def test_no_mirroring_degrades_twin(self, tmp_path):
        twin_fixture(tmp_path)
        store = rois.RoiProfileStore("np:v1", project_root=str(tmp_path))
        store.ensure()
        store._ensure_normalized()
        # same-orientation only: the twin (other hemisphere) scores ~0
        same = 0.5 * (store._post_n @ store._post_n[0]
                      + store._pre_n @ store._pre_n[0])
        twin_same = float(same[1])
        # mirrored orientation via the permuted query (the store no longer
        # materializes mirrored candidate matrices)
        mirror = 0.5 * (
            store._post_n @ store._mirrored_query(store._post_n[0])
            + store._pre_n @ store._mirrored_query(store._pre_n[0])
        )
        assert mirror[1] > 0.99
        assert twin_same < 0.2

    def test_screen_excludes_query_and_zero_neurons(self, tmp_path):
        twin_fixture(tmp_path)
        store = rois.RoiProfileStore("np:v1", project_root=str(tmp_path))
        store.ensure()
        res = store.screen([1])
        ids = set(res["bodyId"].tolist())
        assert 1 not in ids            # query excluded
        assert 4 not in ids            # zero-vector neuron excluded
        assert ids == {2, 3}
        assert res["roi_similarity"].is_monotonic_decreasing

    def test_multi_member_query_uses_mean_distribution(self, tmp_path):
        write_roi_dataset(tmp_path, "np:v1", {
            1: {"post": {"A(R)": 10}, "pre": {"A(R)": 10}, "type": "T"},
            2: {"post": {"A(L)": 10}, "pre": {"A(L)": 10}, "type": "T"},
            3: {"post": {"A(R)": 10, "A(L)": 10}, "pre": {"A(R)": 10, "A(L)": 10},
                "type": "BIL"},
            4: {"post": {"M": 5}, "pre": {"M": 5}, "type": "MID"},
        })
        store = rois.RoiProfileStore("np:v1", project_root=str(tmp_path))
        store.ensure()
        # mean of the T members = the bilateral neuron's distribution
        res = store.screen([1, 2])
        assert int(res.iloc[0]["bodyId"]) == 3

    def test_missing_roi_table_raises(self, tmp_path):
        store = rois.RoiProfileStore("np:v1", project_root=str(tmp_path))
        with pytest.raises(rois.RoiScreeningUnavailable, match="ROI-count"):
            store.ensure()

    def test_hierarchical_sidecar_rejected_without_backfill(self, tmp_path,
                                                            monkeypatch):
        hierarchical_fixture(tmp_path)
        monkeypatch.setattr(rois, "backfill_dataset_metadata",
                            lambda *a, **k: None)
        store = rois.RoiProfileStore("np:v1", project_root=str(tmp_path))
        with pytest.raises(rois.RoiScreeningUnavailable, match="partition"):
            store.ensure()

    def test_missing_sidecar_backfills_via_preparation(self, tmp_path,
                                                       monkeypatch):
        twin_fixture(tmp_path)
        sidecar = (tmp_path / "datasets" / "np_v1" / "np_v1_metadata.json")
        sidecar.unlink()

        def fake_backfill(dataset, project_root=None, log=None):
            meta = {"dataset": dataset,
                    "roi_coverage": {"roi_list": PRIMARY}}
            sidecar.write_text(json.dumps(meta))
            return meta

        monkeypatch.setattr(rois, "backfill_dataset_metadata", fake_backfill)
        store = rois.RoiProfileStore("np:v1", project_root=str(tmp_path))
        store.ensure()
        assert store.rois == PRIMARY
        assert int(store.screen([1]).iloc[0]["bodyId"]) == 2

    def test_cache_roundtrip_and_invalidation(self, tmp_path):
        twin_fixture(tmp_path)
        store = rois.RoiProfileStore("np:v1", project_root=str(tmp_path))
        store.ensure()
        npz = (tmp_path / "cache" / "np_v1" / "morphology"
               / "roi_profiles.npz")
        assert npz.exists()

        reloaded = rois.RoiProfileStore("np:v1", project_root=str(tmp_path))
        assert reloaded.load()
        assert (reloaded.bodyIds == store.bodyIds).all()
        assert reloaded.screen([1]).iloc[0]["bodyId"] == 2

        # changing the ROI table invalidates the fingerprint -> rebuild
        folder = tmp_path / "datasets" / "np_v1"
        csv = folder / "np_v1_allneurons_roi_count_df.csv"
        csv.write_text(csv.read_text() + "5,A(R),1,1\n")
        stale = rois.RoiProfileStore("np:v1", project_root=str(tmp_path))
        assert not stale.load()
        stale.ensure()
        assert 5 in stale.bodyIds.tolist()
