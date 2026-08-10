#!/usr/bin/env python
"""
Regression tests for the DROCAT pathfinding stack.

Covers:
  - FindAllPath graph-cache key correctness (filters must be part of the key)
  - FindAllPath graph-cache bound (LRU-style eviction)
  - Path-edge to layer-table matching (reciprocal/recurrent edges must not be
    dropped when the edge's fetch layer differs from its path position)
  - All five FastGraph pathfinding algorithms against NetworkX-compatible
    all_simple_paths on seeded random directed graphs
"""

import sys
from pathlib import Path

import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))

from coana import (  # noqa: E402
    _FINDALLPATH_CACHE_MAX,
    _FINDALLPATH_GRAPH_CACHE,
    _findallpath_cache_key,
    _findallpath_cache_put,
    _match_path_edges_to_layers,
)
from vispath_pkg.fast_graph_core import FastGraph  # noqa: E402


# =============================================================================
# FindAllPath graph cache
# =============================================================================

def _base_cache_kwargs(**overrides):
    kwargs = {
        "dataset_safe": "hemibrain_v1_2_1",
        "source_ID": ["aMe12", "aMe10"],
        "target_ID": ["PPL101", "PPL103"],
        "max_interlayer": 2,
        "separate_hemispheres": False,
        "filter_by": "bodyId",
        "min_ratio": 0.0,
        "min_traversal_probability": 0.0,
        "exclude_intra_type_connections": False,
    }
    kwargs.update(overrides)
    return kwargs


class TestFindAllPathCacheKey:
    def test_key_stable_and_order_insensitive(self):
        key1 = _findallpath_cache_key(**_base_cache_kwargs())
        key2 = _findallpath_cache_key(
            **_base_cache_kwargs(
                source_ID=["aMe10", "aMe12"],
                target_ID=["PPL103", "PPL101"],
            )
        )
        assert key1 == key2

    def test_key_differs_when_connection_filters_change(self):
        base = _base_cache_kwargs()
        base_key = _findallpath_cache_key(**base)

        for override in (
            {"filter_by": "type"},
            {"min_ratio": 0.5},
            {"min_traversal_probability": 0.01},
            {"exclude_intra_type_connections": True},
            {"separate_hemispheres": True},
            {"max_interlayer": 3},
        ):
            assert _findallpath_cache_key(**_base_cache_kwargs(**override)) != base_key

    def test_threshold_is_not_part_of_the_key(self):
        # min_synapse_num is deliberately excluded: the cached graph is
        # filtered up to the requested threshold at reuse time.
        import inspect

        params = inspect.signature(_findallpath_cache_key).parameters
        assert "min_synapse_num" not in params

    def test_different_sources_or_targets_change_key(self):
        base_key = _findallpath_cache_key(**_base_cache_kwargs())
        assert (
            _findallpath_cache_key(**_base_cache_kwargs(source_ID=["other"]))
            != base_key
        )
        assert (
            _findallpath_cache_key(**_base_cache_kwargs(target_ID=["other"]))
            != base_key
        )


class TestFindAllPathCacheBound:
    def test_cache_stays_bounded_and_evicts_oldest(self):
        _FINDALLPATH_GRAPH_CACHE.clear()
        try:
            for i in range(_FINDALLPATH_CACHE_MAX + 3):
                _findallpath_cache_put(f"key_{i}", {"threshold": 1})
            assert len(_FINDALLPATH_GRAPH_CACHE) == _FINDALLPATH_CACHE_MAX
            # Oldest three entries must have been evicted.
            for i in range(3):
                assert f"key_{i}" not in _FINDALLPATH_GRAPH_CACHE
            # Newest entries survive.
            for i in range(3, _FINDALLPATH_CACHE_MAX + 3):
                assert f"key_{i}" in _FINDALLPATH_GRAPH_CACHE
        finally:
            _FINDALLPATH_GRAPH_CACHE.clear()

    def test_hit_refreshes_recency(self):
        _FINDALLPATH_GRAPH_CACHE.clear()
        try:
            for i in range(_FINDALLPATH_CACHE_MAX):
                _findallpath_cache_put(f"key_{i}", {"threshold": 1})
            # Touch the oldest entry, then add enough new entries to evict.
            entry = _FINDALLPATH_GRAPH_CACHE.pop("key_0", None)
            assert entry is not None
            _FINDALLPATH_GRAPH_CACHE["key_0"] = entry
            for i in range(_FINDALLPATH_CACHE_MAX, _FINDALLPATH_CACHE_MAX + 2):
                _findallpath_cache_put(f"key_{i}", {"threshold": 1})
            assert "key_0" in _FINDALLPATH_GRAPH_CACHE
            assert "key_1" not in _FINDALLPATH_GRAPH_CACHE
        finally:
            _FINDALLPATH_GRAPH_CACHE.clear()


# =============================================================================
# Path-edge -> layer-table matching
# =============================================================================


class TestMatchPathEdgesToLayers:
    def test_reciprocal_edge_kept_when_fetch_layer_differs_from_path_index(self):
        # Sources: A, B.  A->X, X->B, B->Y, Y->T.
        # Path A->X->B->Y->T uses edge B->Y at path index 2, but B was
        # discovered at layer 1, so the B->Y row lives in the layer-1 table.
        # Index-based matching would drop it; table-based matching must keep it.
        layers = [
            pl.DataFrame(
                {
                    "bodyId_pre": ["A", "B"],
                    "bodyId_post": ["X", "Y"],
                    "weight": [1, 1],
                }
            ),
            pl.DataFrame(
                {
                    "bodyId_pre": ["X", "Y"],
                    "bodyId_post": ["B", "T"],
                    "weight": [1, 1],
                }
            ),
        ]
        edges_in_paths = {("A", "X"), ("X", "B"), ("B", "Y"), ("Y", "T")}
        valid_by_layer, matched = _match_path_edges_to_layers(edges_in_paths, layers)

        assert ("B", "Y") in valid_by_layer[0]
        assert ("X", "B") in valid_by_layer[1]
        assert matched == edges_in_paths

    def test_all_layer_occurrences_are_kept(self):
        # The same edge appears in two layer tables; both occurrences should
        # survive the match ("keeping ALL layer-specific occurrences").
        layers = [
            pl.DataFrame(
                {
                    "bodyId_pre": ["A"],
                    "bodyId_post": ["B"],
                    "weight": [3],
                }
            ),
            pl.DataFrame(
                {
                    "bodyId_pre": ["A"],
                    "bodyId_post": ["B"],
                    "weight": [2],
                }
            ),
        ]
        edges_in_paths = {("A", "B")}
        valid_by_layer, matched = _match_path_edges_to_layers(edges_in_paths, layers)
        assert valid_by_layer[0] == {("A", "B")}
        assert valid_by_layer[1] == {("A", "B")}
        assert matched == {("A", "B")}

    def test_empty_and_missing_tables_are_handled(self):
        layers = [None, pl.DataFrame(), pl.DataFrame({"bodyId_pre": ["A"], "bodyId_post": ["B"]})]
        edges_in_paths = {("A", "B")}
        valid_by_layer, matched = _match_path_edges_to_layers(edges_in_paths, layers)
        assert valid_by_layer == [set(), set(), {("A", "B")}]
        assert matched == {("A", "B")}


# =============================================================================
# FastGraph pathfinding algorithms vs all_simple_paths
# =============================================================================

ALGORITHMS = [
    "bidirectional_bfs",
    "memoized_dfs",
    "backward_dp",
    "meet_in_middle",
    "dfs_backtracking",
]


def _run_algorithm(G, algo, sources, targets, cutoff):
    if algo == "bidirectional_bfs":
        return list(G.find_paths_bidirectional_bfs(sources, targets, cutoff))
    if algo == "memoized_dfs":
        return list(G.find_paths_memoized_dfs(sources, targets, cutoff))
    if algo == "backward_dp":
        return list(G.find_paths_backward_dp(sources, targets, cutoff))
    if algo == "meet_in_middle":
        return list(G.find_paths_meet_in_the_middle(sources, targets, cutoff))
    if algo == "dfs_backtracking":
        return list(G.find_paths_dfs_backtracking(sources, targets, cutoff))
    raise ValueError(algo)


def _expected_paths(G, sources, targets, cutoff):
    expected = set()
    for s in sources:
        for t in targets:
            for p in G.all_simple_paths(s, t, cutoff):
                # DROCAT intentionally excludes zero-length [source] paths.
                if len(p) >= 2:
                    expected.add(tuple(p))
    return expected


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_algorithms_match_all_simple_paths_on_random_graphs(algo):
    import random

    rng = random.Random(20260803 + len(algo))
    for trial in range(60):
        n_nodes = rng.randint(2, 9)
        nodes = [f"N{i}" for i in range(n_nodes)]
        edges = []
        for u in nodes:
            for v in nodes:
                if rng.random() < 0.3:
                    edges.append((u, v, rng.randint(1, 5)))

        G = FastGraph()
        for u, v, w in edges:
            G.add_edge(u, v, w)

        sources = [n for n in nodes if rng.random() < 0.5] or [nodes[0]]
        targets = [n for n in nodes if rng.random() < 0.5] or [nodes[-1]]
        cutoff = rng.randint(1, 4)

        expected = _expected_paths(G, sources, targets, cutoff)
        actual = _run_algorithm(G, algo, sources, targets, cutoff)

        # Every returned path must be simple and within the cutoff.
        for p in actual:
            assert len(p) == len(set(p)), f"non-simple path {p} in {algo}"
            assert 1 <= len(p) - 1 <= cutoff, f"path length out of range: {p}"

        actual_set = {tuple(p) for p in actual}
        assert actual_set == expected, (
            f"{algo} mismatch on trial {trial}: "
            f"missing={sorted(expected - actual_set)[:5]} "
            f"extra={sorted(actual_set - expected)[:5]}"
        )


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_algorithms_handle_edge_cases(algo):
    # Empty graph
    G = FastGraph()
    assert _run_algorithm(G, algo, ["A"], ["B"], 3) == []

    # Single edge
    G = FastGraph()
    G.add_edge("A", "B", 2)
    paths = {tuple(p) for p in _run_algorithm(G, algo, ["A"], ["B"], 2)}
    assert paths == {("A", "B")}

    # Direct cycle: A->B, B->A, B->C
    G = FastGraph()
    G.add_edge("A", "B", 1)
    G.add_edge("B", "A", 1)
    G.add_edge("B", "C", 1)
    paths = {tuple(p) for p in _run_algorithm(G, algo, ["A"], ["C"], 3)}
    assert paths == {("A", "B", "C")}

    # Cutoff 0: no positive-length paths
    G = FastGraph()
    G.add_edge("A", "B", 1)
    assert _run_algorithm(G, algo, ["A"], ["B"], 0) == []

    # Multiple sources and targets
    G = FastGraph()
    for u, v in [("A", "X"), ("B", "X"), ("X", "Y"), ("Y", "T1"), ("Y", "T2")]:
        G.add_edge(u, v, 1)
    paths = {tuple(p) for p in _run_algorithm(G, algo, ["A", "B"], ["T1", "T2"], 3)}
    assert paths == {
        ("A", "X", "Y", "T1"),
        ("A", "X", "Y", "T2"),
        ("B", "X", "Y", "T1"),
        ("B", "X", "Y", "T2"),
    }


# ---------------------------------------------------------------------------
# MemoizedDFS: per-source cap
# ---------------------------------------------------------------------------

def _fanout_graph():
    """A -> X1..X6 -> Y -> T: 6 paths of length 3 from A, 6 from B."""
    G = FastGraph()
    for u in ["A", "B"]:
        for i in range(6):
            G.add_edge(u, f"X{i}", 1)
            G.add_edge(f"X{i}", "Y", 1)
    G.add_edge("Y", "T", 1)
    return G


def test_trim_to_strongest_keeps_top_weight_edges_only():
    """trim_to_strongest keeps the strongest edges by weight across the
    whole graph (pan-graph limit), not per node."""
    G = FastGraph()
    G.add_edge("A", "B", 1)
    G.add_edge("B", "C", 9)
    G.add_edge("A", "C", 5)
    G.add_edge("C", "D", 2)
    G.add_node("Z", node_type="source")
    assert G.number_of_edges() == 4

    removed, threshold = G.trim_to_strongest(2)
    assert removed == 2
    assert threshold == 5  # applied cutoff = min weight among kept edges
    assert G.number_of_edges() == 2
    # the two strongest edges survive (9 and 5)
    assert set(G.edges()) == {("B", "C"), ("A", "C")}
    # nodes left without any edge are dropped; Z was already isolated
    assert "Z" not in G
    assert "D" not in G  # only had the weak C->D edge
    assert "A" in G and "B" in G and "C" in G


def test_trim_to_strongest_noop_for_unlimited_or_huge_limit():
    """None / 0 / negative / >= edge count all leave the graph unchanged."""
    G = FastGraph()
    G.add_edge("A", "B", 3)
    G.add_edge("B", "C", 1)
    for limit in (None, 0, -5, 2, 100):
        assert G.trim_to_strongest(limit) == (0, None)
        assert set(G.edges()) == {("A", "B"), ("B", "C")}


def test_trim_to_strongest_preserves_attrs_and_radj_invalidation():
    """Kept edges keep their edge attrs; the reverse adjacency cache is
    invalidated so predecessor lookups reflect the trimmed graph."""
    G = FastGraph()
    G.add_edge("A", "B", 10, color="red")
    G.add_edge("B", "C", 1, color="blue")
    assert list(G.predecessors("B")) == ["A"]  # build the radj cache

    removed, threshold = G.trim_to_strongest(1)
    assert (removed, threshold) == (1, 10)
    assert set(G.edges()) == {("A", "B")}
    assert G.edge_attrs[("A", "B")]["color"] == "red"
    assert list(G.predecessors("B")) == ["A"]
    assert list(G.predecessors("C")) == []  # C's edge was trimmed


def test_trim_to_strongest_reserves_source_target_edges():
    """Edges incident to reserve_nodes (outgoing AND incoming) survive the
    trim regardless of weight; the rest is filled with the strongest."""
    G = FastGraph()
    G.add_edge("S", "A", 1)    # weak source edge — must survive
    G.add_edge("S", "B", 2)    # weak source edge — must survive
    G.add_edge("A", "T", 1)    # weak target edge — must survive
    G.add_edge("B", "T", 3)    # weak target edge — must survive
    G.add_edge("A", "M", 100)  # strong intermediate
    G.add_edge("M", "B", 90)   # strong intermediate
    G.add_edge("X", "Y", 80)   # strong but isolated from S/T — may be cut

    removed, threshold = G.trim_to_strongest(4, reserve_nodes=["S", "T"])
    # 4 reserved source/target edges stay; capacity 0 left for others
    assert removed == 3
    kept = set(G.edges())
    assert {("S", "A"), ("S", "B"), ("A", "T"), ("B", "T")} <= kept
    assert ("X", "Y") not in kept
    assert threshold == 1  # a reserved weak edge defines the threshold


def test_trim_to_strongest_reservation_wins_over_limit():
    """When reserved edges alone exceed the quota, all of them are kept."""
    G = FastGraph()
    G.add_edge("S", "A", 5)
    G.add_edge("S", "B", 4)
    G.add_edge("S", "C", 3)
    G.add_edge("A", "T", 2)
    removed, threshold = G.trim_to_strongest(2, reserve_nodes=["S", "T"])
    assert removed == 0
    assert threshold == 2
    assert G.number_of_edges() == 4  # reservation wins


def test_trimmed_graph_still_finds_paths():
    """Pathfinding on the trimmed graph works and only uses kept edges."""
    G = FastGraph()
    G.add_edge("S", "X", 1)   # weak
    G.add_edge("X", "T", 1)   # weak
    G.add_edge("S", "Y", 50)  # strong
    G.add_edge("Y", "T", 60)  # strong
    G.trim_to_strongest(2)
    paths = list(G.find_paths_memoized_dfs(["S"], ["T"], 3))
    assert paths == [["S", "Y", "T"]]  # the weak route was dropped


def test_trim_reserved_edges_keep_paths_alive():
    """Reserving source/target edges keeps weak-only routes findable."""
    G = FastGraph()
    G.add_edge("S", "X", 1)   # weak source edge — reserved
    G.add_edge("X", "T", 1)   # weak target edge — reserved
    G.add_edge("S", "Y", 50)  # strong
    G.add_edge("Y", "T", 60)  # strong
    G.trim_to_strongest(2, reserve_nodes=["S", "T"])
    paths = list(G.find_paths_memoized_dfs(["S"], ["T"], 3))
    assert paths == [["S", "X", "T"], ["S", "Y", "T"]]  # both routes


# ---------------------------------------------------------------------------
# FindAllPath: early graph visualization + deep-layer warning
# ---------------------------------------------------------------------------

def test_early_graph_visualization_feeds_built_graph_as_edge_list(monkeypatch, tmp_path):
    """The early visualization consumes the built FastGraph DIRECTLY (edge
    list with source/target/weight) before any path reconstruction, and
    writes into network_early/ inside the run folder."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import coana

    fc = object.__new__(coana.FindNeuronConnection)
    fc.allpath_folder = str(tmp_path)
    fc.verbose_mode = "full"
    fc.showfig = False
    fc.edgeN_limit = 500
    fc.output_format = "xlsx"
    fc.network_layout = "hierarchical"
    fc.source_color = fc.intermediate_color = fc.target_color = fc.link_color = None
    fc._vprint = lambda *a, **k: None

    G = FastGraph()
    G.add_edge("A", "B", 3)
    G.add_edge("B", "C", 5)

    captured = {}

    class FakeVisualizePath:
        def __init__(self, **kw):
            captured.update(kw)

        def visualize(self):
            captured["visualized"] = True

    monkeypatch.setattr(coana, "VisualizePath", FakeVisualizePath)
    fc._visualize_graph_before_reconstruct(G)

    assert captured["visualized"] is True
    df = captured["path_file"]
    assert list(df.columns) == ["source", "target", "weight"]
    assert len(df) == 2
    assert (tmp_path / "network_early").is_dir()


def test_find_all_path_optimization_fields_exist():
    """The pathfinding optimization knobs are dataclass fields with the
    documented defaults (pan-graph edge limits on, early viz off)."""
    import dataclasses
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import coana
    fields = {f.name: f.default for f in dataclasses.fields(coana.FindNeuronConnection)}
    assert fields.get("graph_edge_limit_bodyid") == 5000
    assert fields.get("graph_edge_limit_groups") == 1000
    assert fields.get("visualize_before_reconstruct") is False
    # the old per-source path cap is gone
    assert "max_paths_per_source" not in fields


def test_apply_graph_edge_limit_trims_warns_and_honors_zero():
    """The pan-graph edge limit keeps the strongest edges (source/target
    reserved), prints a NOTICEABLE warning with the applied weight
    threshold, and does nothing when the limit is 0/None."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import coana

    warnings = []
    fc = object.__new__(coana.FindNeuronConnection)
    fc._vprint = lambda msg, level='full': warnings.append(msg)
    fc._warn_notes = []

    G = FastGraph()
    G.add_edge("S", "A", 1)    # weak source edge — reserved
    G.add_edge("A", "T", 2)    # weak target edge — reserved
    G.add_edge("S", "B", 100)  # strong source edge — reserved
    G.add_edge("B", "T", 200)  # strong target edge — reserved
    G.add_edge("A", "M", 50)   # strong intermediate — may be cut
    G.add_edge("M", "B", 60)   # strong intermediate — may be cut
    G.add_edge("X", "Y", 80)   # strong, isolated — fills spare capacity

    removed = fc._apply_graph_edge_limit(G, 5, "bodyId", reserve_nodes=["S", "T"])
    assert removed == 2  # the two intermediate edges were cut
    kept = set(G.edges())
    assert {("S", "A"), ("A", "T"), ("S", "B"), ("B", "T")} <= kept  # reserved
    assert ("X", "Y") in kept  # strongest non-reserved fills the capacity
    assert len(warnings) == 1
    msg = warnings[0]
    assert "⚠️" in msg and "bodyId graph edge limit" in msg
    assert "strongest edges" in msg
    assert "applied threshold: weight >= 1 synapses" in msg  # explicit trim end
    assert "always reserved" in msg
    assert "COMPLETE graph network" in msg and "remove the edge limit" in msg
    # the trim is recorded for user_warning_notes.txt
    assert len(fc._warn_notes) == 1 and "[graph edge limit]" in fc._warn_notes[0]

    # limit 0 / None / large-enough -> untouched, no warning
    G2 = FastGraph()
    G2.add_edge("X", "Y", 1)
    warnings.clear()
    assert fc._apply_graph_edge_limit(G2, 0, "type") == 0
    assert fc._apply_graph_edge_limit(G2, None, "type") == 0
    assert fc._apply_graph_edge_limit(G2, 5, "type") == 0
    assert set(G2.edges()) == {("X", "Y")}
    assert warnings == []


def test_write_user_warning_notes_lists_tilting_operations(tmp_path):
    """user_warning_notes.txt is written at the run folder root when the
    run applied output-tilting operations (trims, thresholds, filters,
    hemisphere/symmetry/reciprocal, output/visualization caps)."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import coana

    fc = object.__new__(coana.FindNeuronConnection)
    fc._vprint = lambda *a, **k: None
    fc._warn_notes = ["- [graph edge limit] bodyId graph trimmed: kept the top 5,000 edges (weight >= 20 synapses); removed 1,234 of 6,234 edges."]
    fc.edgeN_limit = 500
    fc.min_synapse_num = 3
    fc.min_ratio = 0.05
    fc.min_traversal_probability = 0.1
    fc.keyword_in_path_to_remove = ["None"]
    fc.max_interlayer = 4
    fc.separate_hemispheres = False
    fc.hemisphere_filter = "both"
    fc.keep_only_hemisphere_conserved_connections = False
    fc.symmetry_analysis = True
    fc.find_reciprocal = False
    fc.skip_bodyId = False
    fc.pathN_to_show = 200
    fc.cache_only = False

    fc._write_user_warning_notes(str(tmp_path))
    path = tmp_path / "user_warning_notes.txt"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "user warning notes" in text
    assert "[graph edge limit]" in text and "weight >= 20 synapses" in text
    assert "[threshold] min_synapse_num=3" in text
    assert "[threshold] min_ratio=0.05" in text
    assert "[threshold] min_traversal_probability=0.1" in text
    assert "[depth] max_interlayer=4" in text
    assert "[symmetry] symmetry_analysis=True" in text
    assert "[visualization] pathN_to_show=200" in text
    assert "[edge limit per neuron] edgeN_limit=500" in text
    # inactive operations are NOT listed
    assert "hemisphere" not in text and "reciprocal" not in text
    assert "skip_bodyId" not in text and "cache_only" not in text


def test_write_user_warning_notes_skipped_when_nothing_applies(tmp_path):
    """No file is written when the run applied no tilting operations."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import coana

    fc = object.__new__(coana.FindNeuronConnection)
    fc._vprint = lambda *a, **k: None
    fc._warn_notes = []
    fc.edgeN_limit = 0
    fc.min_synapse_num = 1
    fc.min_ratio = 0
    fc.min_traversal_probability = 0
    fc.keyword_in_path_to_remove = ["None"]
    fc.max_interlayer = 2
    fc.separate_hemispheres = False
    fc.hemisphere_filter = "both"
    fc.keep_only_hemisphere_conserved_connections = False
    fc.symmetry_analysis = False
    fc.find_reciprocal = False
    fc.skip_bodyId = False
    fc.pathN_to_show = -1
    fc.cache_only = False

    fc._write_user_warning_notes(str(tmp_path))
    assert not (tmp_path / "user_warning_notes.txt").exists()
