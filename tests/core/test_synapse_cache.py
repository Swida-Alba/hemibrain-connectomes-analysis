"""Shared connector/site synapse-cache contracts."""

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _frame():
    return pd.DataFrame({
        "bodyId_pre": ["1"],
        "bodyId_post": ["2"],
        "x_pre": [1.0], "y_pre": [2.0], "z_pre": [3.0],
        "x_post": [4.0], "y_post": [5.0], "z_post": [6.0],
    })


def test_query_cache_roundtrip_records_exact_provenance(tmp_path):
    from synapse_cache import SynapseCache

    cache = SynapseCache("hemibrain:v1.2.1", str(tmp_path))
    spec = {
        "source_kind": "neuprint",
        "source_criteria": {"type": "aMe"},
        "target_criteria": {"type": "dn1"},
        "source_ids": ["2", "1"],
        "target_ids": ["2"],
        "min_total_weight": 1,
    }
    cache.save_query(spec, _frame(), source="neuprint")

    loaded = cache.load_query(spec)
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, _frame())

    meta_path = cache.query_meta_path(spec)
    metadata = json.loads(meta_path.read_text())
    assert metadata["dataset"] == "hemibrain:v1.2.1"
    assert metadata["query_key"] == cache.query_key(spec)
    assert metadata["row_count"] == 1

    changed = dict(spec, min_total_weight=2)
    assert cache.load_query(changed) is None


def test_pair_cache_indexes_empty_without_writing_empty_pair_file(tmp_path):
    from synapse_cache import SynapseCache

    cache = SynapseCache("hemibrain:v1.2.1", str(tmp_path))
    cache.save_pairs(None, [(101, 202)])

    loaded, missing = cache.load_pairs([101], [202])
    assert loaded is None
    assert missing == []
    assert not cache.pair_path(101, 202).exists()

    cache.save_pairs(_frame(), [(1, 2)])
    loaded, missing = cache.load_pairs([1], [2])
    assert missing == []
    assert loaded is not None and len(loaded) == 1


def test_broad_visualization_query_does_not_duplicate_pair_files(tmp_path):
    from visualize_skeleton import VisualizeSkeleton

    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.dataset = "hemibrain:v1.2.1"
    visualizer.script_path = str(tmp_path)
    visualizer.cache_synapses = True
    visualizer.min_synapse_num = 1
    visualizer.synapse_criteria = None
    visualizer._vprint = lambda *args, **kwargs: None

    frame = pd.concat([
        _frame(),
        pd.DataFrame({
            "bodyId_pre": ["9"], "bodyId_post": ["8"],
            "x_pre": [10.0], "y_pre": [20.0], "z_pre": [30.0],
            "x_post": [40.0], "y_post": [50.0], "z_post": [60.0],
        }),
    ], ignore_index=True)
    fetched = visualizer._get_or_fetch_synapse_query(
        source_criteria={"type": "aMe"},
        target_criteria={"type": "dn1"},
        source_ids={"1"}, target_ids={"2"},
        fetcher=lambda: frame.copy(),
    )
    visualizer._save_cached_synapses(
        fetched, attempted_pairs=[("1", "2")], persist_pairs=False)

    synapse_root = (
        tmp_path / "cache" / "hemibrain_v1_2_1" / "synapses"
    )
    assert list((synapse_root / "queries").glob("*.parquet"))
    assert not list(synapse_root.glob("*.parquet"))
    assert len(fetched) == 1
    assert fetched.iloc[0]["bodyId_pre"] == "1"
