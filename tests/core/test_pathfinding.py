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

import os
import sys
from pathlib import Path

import polars as pl
import pandas as pd
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


def test_trim_to_strongest_reserves_source_outgoing_and_target_incoming():
    """Source-outgoing and target-incoming edges survive the trim regardless
    of weight (when within the reservation cap); the limit applies to the
    NON-reserved edges only (reserved edges do NOT count toward it)."""
    G = FastGraph()
    G.add_edge("S", "A", 1)    # weak source-outgoing — must survive
    G.add_edge("S", "B", 2)    # weak source-outgoing — must survive
    G.add_edge("A", "T", 1)    # weak target-incoming — must survive
    G.add_edge("B", "T", 3)    # weak target-incoming — must survive
    G.add_edge("A", "M", 100)  # strong intermediate
    G.add_edge("M", "B", 90)   # strong intermediate
    G.add_edge("X", "Y", 80)   # strong intermediate
    G.add_edge("M2", "M3", 70)  # strong intermediate
    G.add_edge("M3", "M4", 60)  # strong intermediate
    # source-INCOMING edges are NOT reserved (they cannot be part of a
    # source->target path)
    G.add_edge("Z", "S", 5)

    removed, threshold = G.trim_to_strongest(
        5, reserve_sources=["S"], reserve_targets=["T"]
    )
    # reserved (4) never count: the 5 strongest non-reserved are kept
    assert removed == 1  # only Z->S (weight 5) dropped
    kept = set(G.edges())
    assert {("S", "A"), ("S", "B"), ("A", "T"), ("B", "T"),
            ("A", "M"), ("M", "B"), ("X", "Y"), ("M2", "M3"), ("M3", "M4")} == kept
    assert ("Z", "S") not in kept  # incoming-to-source is NOT reserved
    assert threshold == 60  # min weight among kept NON-reserved edges


def test_trim_reservation_bounded_when_candidates_exceed_limit():
    """The auto-reservation is capped at the limit: when the source/target
    nodes are so many that their incident edges would swallow the graph
    (degenerate classification), only the strongest `keep` source/target
    edges are reserved and the rest rejoin the ordinary pool — the trim
    always produces a bounded graph (<= 2 x keep)."""
    G = FastGraph()
    G.add_edge("S", "A", 5)
    G.add_edge("S", "B", 4)
    G.add_edge("S", "C", 3)
    G.add_edge("A", "T", 2)
    G.add_edge("M1", "M2", 1)
    G.add_edge("M2", "M3", 2)
    G.add_edge("M3", "M4", 3)
    removed, threshold = G.trim_to_strongest(
        2, reserve_sources=["S"], reserve_targets=["T"]
    )
    # reserved capped to the strongest 2 (S->A 5, S->B 4); the leftover
    # (S->C 3, A->T 2) rejoins the pool and competes as ordinary edges
    kept = set(G.edges())
    assert kept == {("S", "A"), ("S", "B"), ("S", "C"), ("M3", "M4")}
    assert removed == 3
    assert threshold == 3  # min weight among kept non-reserved
    assert G.number_of_edges() == 4  # bounded: <= 2 x keep


def test_trim_to_strongest_noop_when_reserved_cover_quota():
    """When every edge is reserved (or non-reserved count <= limit) there is
    nothing to trim: (0, None) and the graph is untouched."""
    G = FastGraph()
    G.add_edge("S", "A", 5)
    G.add_edge("A", "T", 2)
    assert G.trim_to_strongest(
        2, reserve_sources=["S"], reserve_targets=["T"]
    ) == (0, None)
    assert set(G.edges()) == {("S", "A"), ("A", "T")}


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
    """Reserving source-outgoing/target-incoming edges keeps weak-only routes
    findable even with a tiny limit (the capped-out leftovers rejoin the
    ordinary pool, so the weak hops still compete on weight)."""
    G = FastGraph()
    G.add_edge("S", "X", 1)   # weak source edge — reserved first
    G.add_edge("X", "T", 1)   # weak target edge — reserved first
    G.add_edge("S", "Y", 50)  # strong
    G.add_edge("Y", "T", 60)  # strong
    G.trim_to_strongest(2, reserve_sources=["S"], reserve_targets=["T"])
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

        def visualize(self, **kwargs):
            captured["visualized"] = True
            captured["viz_kwargs"] = kwargs

    monkeypatch.setattr(coana, "VisualizePath", FakeVisualizePath)
    fc._visualize_graph_before_reconstruct(G)

    assert captured["visualized"] is True
    # early preview is NETWORK-ONLY: the plain edge list has no path
    # metrics, so the heatmap/Sankey are left to the final path-based call
    assert captured["viz_kwargs"] == {
        "plot_heatmap": False, "plot_Sankey": False, "plot_network": True,
    }, captured["viz_kwargs"]
    df = captured["path_file"]
    assert list(df.columns) == ["source", "target", "weight"]
    assert len(df) == 2
    assert (tmp_path / "network_early").is_dir()


def test_early_viz_type_level_and_bodyid_conditional(monkeypatch, tmp_path):
    """network_early is aggregated to the TYPE level (bodyId -> type,
    weights summed); a bodyId-level early network (network_early_bodyId/) is
    only added when skip_bodyId is False."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import coana

    calls = []

    class FakeVisualizePath:
        def __init__(self, **kw):
            calls.append(kw)

        def visualize(self, **kwargs):
            calls[-1]["visualized"] = True
            calls[-1]["viz_kwargs"] = kwargs

    monkeypatch.setattr(coana, "VisualizePath", FakeVisualizePath)

    def make_fc():
        fc = object.__new__(coana.FindNeuronConnection)
        fc.allpath_folder = str(tmp_path)
        fc.verbose_mode = "full"
        fc.showfig = False
        fc.edgeN_limit = 500
        fc.output_format = "xlsx"
        fc.network_layout = "hierarchical"
        fc.source_color = fc.intermediate_color = fc.target_color = fc.link_color = None
        fc._vprint = lambda *a, **k: None
        fc.all_connections_filtered = [pl.DataFrame({
            "bodyId_pre": ["1", "2", "1"],
            "bodyId_post": ["2", "3", "3"],
            "type_pre": ["A", "B", "A"],
            "type_post": ["B", "C", "C"],
            "weight": [3, 5, 7],
        })]
        return fc

    G = FastGraph()
    G.add_edge("1", "2", 3)
    G.add_edge("2", "3", 5)
    G.add_edge("1", "3", 7)

    # default (skip_bodyId=False): type-level + bodyId-level early networks
    fc = make_fc()
    fc._visualize_graph_before_reconstruct(G)
    assert len(calls) == 2
    # both early calls are network-only (no duplicated heatmap/Sankey)
    assert all(c["viz_kwargs"] == {"plot_heatmap": False, "plot_Sankey": False,
                                   "plot_network": True} for c in calls)
    type_df = calls[0]["path_file"]
    assert list(type_df.columns) == ["source", "target", "weight"]
    assert set(zip(type_df["source"], type_df["target"])) == \
        {("A", "B"), ("B", "C"), ("A", "C")}
    # weights summed per type pair (A->B 3 + A->C 7 = 10)
    assert type_df[type_df["source"] == "A"]["weight"].sum() == 10
    body_df = calls[1]["path_file"]
    assert set(zip(body_df["source"], body_df["target"])) == \
        {("1", "2"), ("2", "3"), ("1", "3")}
    assert (tmp_path / "network_early").is_dir()
    assert (tmp_path / "network_early_bodyId").is_dir()

    # skip_bodyId=True: only the type-level early network
    calls.clear()
    fc2 = make_fc()
    fc2.skip_bodyId = True
    fc2._visualize_graph_before_reconstruct(G)
    assert len(calls) == 1
    assert (tmp_path / "network_early").is_dir()


def test_derive_type_paths_from_bodyid_paths():
    """Type-level paths are DERIVED from the discovered bodyId paths
    (aggregate node types + verify hops) instead of a second pathfinding on
    a type-level graph: sequences are deduplicated, repeated-type routes
    (A->B->A) are preserved, and hops removed by the type-level edge limit
    (or endpoints outside the queried source/target type sets) drop the
    path — so no phantom type path can appear."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import coana

    fc = object.__new__(coana.FindNeuronConnection)
    labels = {
        "a1": "A", "b1": "B", "c1": "C", "d1": "D",
        "a2": "A", "b2": "B", "a3": "A", "x1": "X", "y1": "Y",
    }
    all_paths = [
        ["a1", "b1", "c1"],   # A->B->C (real)
        ["a1", "b1", "c1"],   # duplicate -> deduplicated
        ["a1", "b1", "d1"],   # A->B->D (real)
        ["a2", "b2", "a3"],   # A->B->A (repeated type on a real path)
        ["x1", "y1"],          # X->Y: X is not a queried source type
        ["a1", "b1"],          # A->B: B is not a queried target type
    ]
    # (X, Y) was removed by the type-level edge limit
    kept = {("A", "B"), ("B", "C"), ("B", "D"), ("B", "A")}
    out = fc._derive_type_paths_from_bodyid_paths(
        all_paths, labels.__getitem__, kept,
        source_types=["A"], target_types=["C", "D", "A"],
    )
    got = sorted(tuple(p) for p in out)
    assert got == [("A", "B", "A"), ("A", "B", "C"), ("A", "B", "D")], got


def test_derive_type_paths_verbose_shows_single_line_progress(capsys):
    """verbose=True wraps the bodyId-path iteration with the single-line
    progress display (LineProgress, \r-refreshed — no newline spam), and the
    derived result is unchanged."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import coana

    fc = object.__new__(coana.FindNeuronConnection)
    labels = {"a1": "A", "b1": "B", "c1": "C", "a2": "A", "b2": "B"}
    all_paths = [["a1", "b1", "c1"], ["a2", "b2"]]
    kept = {("A", "B"), ("B", "C")}
    out = fc._derive_type_paths_from_bodyid_paths(
        all_paths, labels.__getitem__, kept,
        source_types=["A"], target_types=["C", "B"], verbose=True)
    got = sorted(tuple(p) for p in out)
    assert got == [("A", "B"), ("A", "B", "C")], got
    out_str = capsys.readouterr().out
    assert "Deriving type-level paths" in out_str
    assert "\n" not in out_str  # \r-refreshed single line only


def test_relocate_viz_outputs_organizes_visualization_folder(tmp_path):
    """Phase-4 visualization artifacts are organized: the htmls move to
    visualization/ with Network_/Sankey_/Heatmap_ prefixes, and the
    vispath-exported data plus an explicitly named companion DataFrame move
    to visualization/visualization_data/."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import coana

    fc = object.__new__(coana.FindNeuronConnection)
    run = tmp_path / "findallpath_X_to_Y_L2w3r0_20260101_000000"
    run.mkdir()
    fc.allpath_folder = str(run)
    fc._vprint = lambda *a, **k: None
    base = run.name
    for suffix in ("_network.html", "_Sankey.html", "_heatmap.html"):
        (run / (base + suffix)).write_text("<html></html>")
    (run / (base + "_data_connections.csv")).write_text("a\n")
    (run / (base + "_data_original_paths.csv")).write_text("b\n")
    (run / (base + "_data.xlsx")).write_text("x")

    inp = pd.DataFrame({"path": ["A->B"], "length": [1]})
    fc._relocate_viz_outputs(
        input_df=inp,
        input_filename="type_paths_visualized.csv",
    )

    viz = run / "visualization"
    # prefix + run name; the redundant type suffix is dropped
    assert (viz / f"Network_{base}.html").exists()
    assert (viz / f"Sankey_{base}.html").exists()
    assert (viz / f"Heatmap_{base}.html").exists()
    assert not (viz / f"Network_{base}_network.html").exists()
    # originals moved out of the run root
    assert not (run / (base + "_network.html")).exists()
    d = viz / "visualization_data"
    assert (d / (base + "_data_connections.csv")).exists()
    assert (d / (base + "_data.xlsx")).exists()
    assert (d / "type_paths_visualized.csv").exists()
    assert not (d / "type_paths_input.csv").exists()
    saved = pd.read_csv(d / "type_paths_visualized.csv")
    assert list(saved.columns) == ["path", "length"]


def test_trim_bodyid_edges_applies_limit_only_for_deep_searches():
    """The pan-graph bodyId edge limit applies ONLY when max_interlayer >= 3
    (deep searches); shallow searches keep the COMPLETE graph — no trim
    warning is emitted and every row survives."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import coana

    layers = [pl.DataFrame({
        "bodyId_pre": ["S", "S", "A", "B", "S", "C"],
        "bodyId_post": ["A", "B", "T", "T", "C", "T"],
        "weight": [1, 3, 4, 2, 100, 90],
    })]
    for L, expect_trim in ((2, False), (3, True)):
        fc = object.__new__(coana.FindNeuronConnection)
        fc.max_interlayer = L
        fc.graph_edge_limit_bodyid = 2
        fc._vprint = lambda *a, **k: None
        fc._warn_notes = []
        out = fc._trim_bodyid_edges(layers, ["S"], ["T"])
        rows = out.height if hasattr(out, "height") else len(out)
        if expect_trim:
            # limit=2: 2 reserved (S->C, A->T) + 2 strongest non-reserved
            # (C->T, S->B) survive; the weak S->A / B->T rows are cut
            assert rows == 4, rows
            assert fc._warn_notes            # trim warning recorded
        else:
            assert rows == 6, rows          # complete graph kept
            assert fc._warn_notes == []     # no trim warning


def test_find_all_path_optimization_fields_exist():
    """The pathfinding optimization knobs are dataclass fields with the
    documented defaults (pan-graph edge limits on, early viz off)."""
    import dataclasses
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import coana
    fields = {f.name: f.default for f in dataclasses.fields(coana.FindNeuronConnection)}
    assert fields.get("graph_edge_limit_bodyid") is None
    assert fields.get("graph_edge_limit_groups") == 5000
    assert fields.get("visualize_before_reconstruct") is False
    # the old per-source path cap is gone
    assert "max_paths_per_source" not in fields


def test_trim_edges_with_path_integrity_trims_warns_and_honors_zero():
    """The table-level pan-graph edge limit keeps the strongest USABLE
    non-reserved rows (source-outgoing/target-incoming reserved first, not
    counted), drops rows that cannot lie on any source->target path, warns
    with the applied threshold, and does nothing when the limit is 0/None."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import coana

    warnings = []
    fc = object.__new__(coana.FindNeuronConnection)
    fc._vprint = lambda msg, level='full': warnings.append(msg)
    fc._warn_notes = []

    conn = pd.DataFrame({
        "bodyId_pre": ["S", "A", "S", "B", "A", "M", "X"],
        "bodyId_post": ["A", "T", "B", "T", "M", "B", "Y"],
        "weight": [1, 2, 100, 200, 50, 60, 80],
    })

    trimmed, removed, threshold = fc._trim_edges_with_path_integrity(
        conn, 2, "bodyId", sources=["S"], targets=["T"],
        pre_col="bodyId_pre", post_col="bodyId_post",
    )
    # reservation capped at the limit (2): B->T (200) and S->B (100) are
    # reserved; the 2 strongest viable non-reserved (M->B 60, A->M 50) are
    # kept; X->Y is dropped by the reachability filter (X can never reach a
    # target) and the weak source/target rows rejoin the pool
    kept = set(zip(trimmed["bodyId_pre"], trimmed["bodyId_post"]))
    assert kept == {("S", "B"), ("B", "T"), ("M", "B"), ("A", "M")}
    assert removed == 3
    assert threshold == 50  # min weight among kept NON-reserved rows
    assert len(warnings) == 1
    msg = warnings[0]
    assert "⚠️" in msg and "bodyId graph edge limit" in msg
    assert "strongest usable non-reserved edges" in msg
    assert "applied threshold: weight >= 50 synapses" in msg
    assert "do NOT count toward it" in msg
    assert "COMPLETE graph network" in msg and "remove the edge limit" in msg
    # the trim is recorded for user_warning_notes.txt
    assert len(fc._warn_notes) == 1 and "[graph edge limit]" in fc._warn_notes[0]
    assert "not counted toward it" in fc._warn_notes[0]

    # limit 0 / None -> untouched, no warning
    warnings.clear()
    out, removed0, thr0 = fc._trim_edges_with_path_integrity(
        conn, 0, "type", sources=["S"], targets=["T"],
        pre_col="bodyId_pre", post_col="bodyId_post",
    )
    assert removed0 == 0 and thr0 is None and len(out) == len(conn)
    assert warnings == []


def test_trim_edges_fill_loop_inflates_budget_for_dead_ends():
    """When the strongest edges create dead ends (their continuation is
    trimmed away), the adaptive fill loop inflates the budget until the
    usable edge count reaches the limit — the budget is never wasted on
    pruned edges, and weak last hops that complete a strong path survive."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import coana

    fc = object.__new__(coana.FindNeuronConnection)
    fc._vprint = lambda *a, **k: None
    fc._warn_notes = []

    # The strong A->B->C chain feeds T only through the weak C->T (weight
    # 1), which loses the reservation cap (top-2 reserved = S->A, S->X).
    # With limit=2 the first pass keeps A->B, B->C (usable=0: nothing
    # reaches T) -> the fill loop inflates the budget to 4 and rescues
    # S->Y / X->T so the graph becomes usable.
    conn = pd.DataFrame({
        "bodyId_pre": ["S", "S", "S", "A", "B", "X", "Y", "C"],
        "bodyId_post": ["A", "X", "Y", "B", "C", "T", "T", "T"],
        "weight": [100, 70, 65, 99, 98, 60, 55, 1],
    })

    trimmed, removed, threshold = fc._trim_edges_with_path_integrity(
        conn, 2, "bodyId", sources=["S"], targets=["T"],
        pre_col="bodyId_pre", post_col="bodyId_post",
    )
    kept = set(zip(trimmed["bodyId_pre"], trimmed["bodyId_post"]))
    # budget inflated 2 -> 4: the strong chain S->A, A->B, B->C stays dead
    # (C->T trimmed), but S->Y and X->T (60) are rescued and usable=2>=2
    assert kept == {("S", "A"), ("S", "X"), ("A", "B"), ("B", "C"),
                    ("S", "Y"), ("X", "T")}
    assert removed == 2  # Y->T (55) and C->T (1) dropped
    assert threshold == 60  # min weight among the inflated non-reserved slice
    assert len(trimmed) == 6  # every kept row is usable after dead-end pruning


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
    # the limits were actually reached during the run
    fc._edgeN_limit_reached = True
    fc._min_synapse_excluded = True
    fc._depth_cap_reached = True

    fc._write_user_warning_notes(str(tmp_path))
    path = tmp_path / "user_warning_notes.txt"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "user warning notes" in text
    assert "[graph edge limit]" in text and "weight >= 20 synapses" in text
    # Synapse-count cutoff is an execution parameter, not a warning-note
    # threshold; ratio and traversal-probability thresholds remain explicit.
    assert "[threshold] min_synapse_num" not in text
    assert "[threshold] min_ratio=0.05" in text
    assert "[threshold] min_traversal_probability=0.1" in text
    assert "[depth] max_interlayer=4" in text
    assert "[symmetry] symmetry_analysis=True" in text
    assert "[visualization] pathN_to_show=200" in text
    assert "[edge limit per neuron] edgeN_limit=500" in text
    # inactive operations are NOT listed
    assert "hemisphere" not in text and "reciprocal" not in text
    assert "skip_bodyId" not in text and "cache_only" not in text


def test_late_priority_neuron_match_is_written_to_warning_notes(tmp_path):
    """Names resolved in taxonomy columns remain visible in run caveats."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import coana

    fc = object.__new__(coana.FindNeuronConnection)
    fc._vprint = lambda *a, **k: None
    fc._warn_notes = []
    fc._record_search_priority_warnings(
        "source",
        [{
            "search_term": "MTe07",
            "matched_column": "flywireType",
            "match_count": 2,
        }],
    )
    fc._write_user_warning_notes(str(tmp_path))
    text = (tmp_path / "user_warning_notes.txt").read_text(encoding="utf-8")
    assert '[search priority] source query "MTe07" resolved via "flywireType"' in text


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
    fc._edgeN_limit_reached = False
    fc._min_synapse_excluded = False
    fc._depth_cap_reached = False

    fc._write_user_warning_notes(str(tmp_path))
    assert not (tmp_path / "user_warning_notes.txt").exists()


def test_write_user_warning_notes_omits_limits_not_reached(tmp_path):
    """A configured limit that was never reached must NOT produce its
    warning note: the edgeN_limit / min_synapse_num / max_interlayer notes
    are listed only when the run actually hit the limit (edges trimmed,
    connections dropped, depth cap reached)."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import coana

    fc = object.__new__(coana.FindNeuronConnection)
    fc._vprint = lambda *a, **k: None
    # A real graph-edge trim note (reached) keeps the file alive.
    fc._warn_notes = ["- [graph edge limit] bodyId graph trimmed: kept the top 5,000 edges (weight >= 20 synapses); removed 1,234 of 6,234 edges."]
    # Limits configured but NOT reached during the run.
    fc.edgeN_limit = 500
    fc.min_synapse_num = 3
    fc.min_ratio = 0.05
    fc.min_traversal_probability = 0.1
    fc.keyword_in_path_to_remove = ["None"]
    fc.max_interlayer = 8
    fc.separate_hemispheres = False
    fc.hemisphere_filter = "both"
    fc.keep_only_hemisphere_conserved_connections = False
    fc.symmetry_analysis = True
    fc.find_reciprocal = False
    fc.skip_bodyId = False
    fc.pathN_to_show = 200
    fc.cache_only = False
    fc._edgeN_limit_reached = False
    fc._min_synapse_excluded = False
    fc._depth_cap_reached = False

    fc._write_user_warning_notes(str(tmp_path))
    path = tmp_path / "user_warning_notes.txt"
    assert path.exists()          # the graph-edge-limit note is still there
    text = path.read_text(encoding="utf-8")
    assert "[graph edge limit]" in text
    # the three config-derived notes are gated on actual limit hits
    assert "[edge limit per neuron]" not in text
    assert "[threshold] min_synapse_num" not in text
    assert "[depth] max_interlayer" not in text


def test_edgeN_limit_note_gated_on_actual_visualization_trim(tmp_path):
    """The '[edge limit per neuron]' note appears only after a visualization
    actually trimmed edges (vp.edge_limit_trimmed), never on config alone."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from types import SimpleNamespace
    import coana

    fc = object.__new__(coana.FindNeuronConnection)
    fc._vprint = lambda *a, **k: None
    fc._warn_notes = []
    fc.edgeN_limit = 500
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

    # No trim happened: no note (and no file) despite the configured limit.
    fc._edgeN_limit_reached = False
    fc._write_user_warning_notes(str(tmp_path))
    assert not (tmp_path / "user_warning_notes.txt").exists()

    # A visualization that trimmed edges flips the per-run flag -> note.
    fc._record_viz_edge_trim(SimpleNamespace(edge_limit_trimmed=True))
    fc._write_user_warning_notes(str(tmp_path))
    text = (tmp_path / "user_warning_notes.txt").read_text(encoding="utf-8")
    assert "[edge limit per neuron] edgeN_limit=500" in text

    # A visualization that did NOT trim must not flip the flag.
    fc2 = object.__new__(coana.FindNeuronConnection)
    fc2.edgeN_limit = 500
    fc2._record_viz_edge_trim(SimpleNamespace(edge_limit_trimmed=False))
    assert getattr(fc2, "_edgeN_limit_reached", False) is False

# =============================================================================
# FastGraph.find_paths_shortest (FindShortestPath enumeration core)
# =============================================================================

def _shortest_reference(G, sources, targets, cutoff=None):
    """Reference implementation: filter the memoized-DFS all-path output to
    the per-(source, target)-pair minimum length (all ties kept).
    Zero-hop paths are excluded, matching the DROCAT convention used by
    ``_expected_paths``."""
    max_len = cutoff if cutoff is not None else max(G.number_of_nodes(), 1)
    best = {}
    paths_by_pair = {}
    for p in G.find_paths_memoized_dfs(sources, targets, max_len):
        if len(p) < 2:
            continue
        key = (p[0], p[-1])
        paths_by_pair.setdefault(key, []).append(tuple(p))
        best[key] = min(best.get(key, len(p) - 1), len(p) - 1)
    expected = set()
    for key, paths in paths_by_pair.items():
        expected.update(p for p in paths if len(p) - 1 == best[key])
    return expected


class TestFindPathsShortest:
    def test_diamond_yields_all_tied_shortest_paths(self):
        G = FastGraph()
        for u, v in [("S", "A"), ("S", "B"), ("A", "T"), ("B", "T")]:
            G.add_edge(u, v, 1)
        paths = {tuple(p) for p in G.find_paths_shortest(["S"], ["T"])}
        assert paths == {("S", "A", "T"), ("S", "B", "T")}

    def test_longer_routes_are_excluded(self):
        G = FastGraph()
        for u, v in [("S", "T"), ("S", "A"), ("A", "B"), ("B", "T")]:
            G.add_edge(u, v, 1)
        paths = {tuple(p) for p in G.find_paths_shortest(["S"], ["T"])}
        assert paths == {("S", "T")}

    def test_dead_end_branches_ignored(self):
        G = FastGraph()
        for u, v in [("S", "A"), ("A", "T"), ("S", "D"), ("D", "E")]:
            G.add_edge(u, v, 1)
        paths = {tuple(p) for p in G.find_paths_shortest(["S"], ["T"])}
        assert paths == {("S", "A", "T")}

    def test_cutoff_respected_and_none_is_unlimited(self):
        G = FastGraph()
        for u, v in [("S", "A"), ("A", "B"), ("B", "T")]:
            G.add_edge(u, v, 1)
        assert list(G.find_paths_shortest(["S"], ["T"], cutoff=2)) == []
        assert {tuple(p) for p in G.find_paths_shortest(["S"], ["T"], cutoff=3)} \
            == {("S", "A", "B", "T")}
        assert {tuple(p) for p in G.find_paths_shortest(["S"], ["T"], cutoff=None)} \
            == {("S", "A", "B", "T")}

    def test_unreachable_pairs_yield_nothing(self):
        G = FastGraph()
        G.add_edge("S", "A", 1)
        G.add_edge("A", "T", 1)
        G.add_node("X")  # isolated source
        paths = {tuple(p) for p in G.find_paths_shortest(["S", "X"], ["T"])}
        assert paths == {("S", "A", "T")}

    def test_cycles_and_reciprocal_edges_do_not_derail(self):
        G = FastGraph()
        for u, v in [("S", "A"), ("A", "B"), ("B", "A"), ("B", "T"), ("A", "S")]:
            G.add_edge(u, v, 1)
        paths = {tuple(p) for p in G.find_paths_shortest(["S"], ["T"])}
        assert paths == {("S", "A", "B", "T")}

    def test_source_is_target_still_reaches_other_targets(self):
        # Consistent with the sibling algorithms: zero-hop paths are not
        # yielded, but a source that is also a target still reaches the
        # OTHER targets (nearest-target gradients would miss these).
        G = FastGraph()
        G.add_edge("A", "C", 1)
        G.add_edge("C", "B", 1)
        paths = {tuple(p) for p in G.find_paths_shortest(["A"], ["A", "B"])}
        assert paths == {("A", "C", "B")}

    def test_empty_graph_and_unknown_targets(self):
        G = FastGraph()
        assert list(G.find_paths_shortest(["A"], ["B"])) == []
        G.add_edge("A", "B", 1)
        assert list(G.find_paths_shortest(["A"], ["ZZZ"])) == []
        assert list(G.find_paths_shortest([], ["B"])) == []

    def test_long_chain_no_recursion_error(self):
        """The iterative shortest-DAG walk must handle paths far beyond the
        Python recursion limit (unlimited depth is a supported mode)."""
        G = FastGraph()
        n = 2000
        for i in range(n):
            G.add_edge(f"N{i}", f"N{i + 1}", 1)
        paths = {tuple(p) for p in G.find_paths_shortest(["N0"], [f"N{n}"])}
        assert paths == {tuple(f"N{i}" for i in range(n + 1))}

    def test_matches_reference_on_random_graphs(self):
        import random

        rng = random.Random(20260811)
        for trial in range(60):
            n_nodes = rng.randint(2, 9)
            nodes = [f"N{i}" for i in range(n_nodes)]
            G = FastGraph()
            for u in nodes:
                for v in nodes:
                    if rng.random() < 0.3:
                        G.add_edge(u, v, rng.randint(1, 5))

            sources = [n for n in nodes if rng.random() < 0.5] or [nodes[0]]
            targets = [n for n in nodes if rng.random() < 0.5] or [nodes[-1]]
            cutoff = rng.choice([None, 1, 2, 3, 4])

            expected = _shortest_reference(G, sources, targets, cutoff)
            actual = {tuple(p) for p in
                      G.find_paths_shortest(sources, targets, cutoff)}
            assert actual == expected, (
                f"shortest mismatch on trial {trial} (cutoff={cutoff}): "
                f"missing={sorted(expected - actual)[:5]} "
                f"extra={sorted(actual - expected)[:5]}"
            )
            # Every yielded path must be simple and minimal for its pair.
            for p in actual:
                assert len(p) == len(set(p)), f"non-simple path {p}"


class TestShortestEarlyStopEquivalence:
    """Discovery stopping at the target's first layer must yield identical
    shortest paths as a full-depth search (BFS discovery order)."""

    def test_truncating_at_target_discovery_keeps_shortest_paths(self):
        edges = [("S", "A"), ("A", "T"), ("S", "B"), ("B", "C"),
                 ("C", "D"), ("D", "T")]
        full = FastGraph()
        for u, v in edges:
            full.add_edge(u, v, 1)
        # Truncated graph = the layers fetched before/including target
        # discovery (T first appears in layer 2: S->A->T).
        truncated = FastGraph()
        for u, v in [("S", "A"), ("S", "B"), ("A", "T"), ("B", "C")]:
            truncated.add_edge(u, v, 1)
        assert {tuple(p) for p in full.find_paths_shortest(["S"], ["T"])} \
            == {tuple(p) for p in truncated.find_paths_shortest(["S"], ["T"])} \
            == {("S", "A", "T")}


class TestFindShortestPathCacheKey:
    def test_shortest_key_omits_depth(self):
        # max_interlayer=None (shortest mode) drops the depth component:
        # the fetched depth is a result, not a query parameter.
        k = _findallpath_cache_key(**_base_cache_kwargs(max_interlayer=None))
        assert k != _findallpath_cache_key(**_base_cache_kwargs(max_interlayer=2))
        assert k != _findallpath_cache_key(**_base_cache_kwargs(max_interlayer=5))
        # Filters still differentiate keys in shortest mode.
        assert k != _findallpath_cache_key(
            **_base_cache_kwargs(max_interlayer=None, min_ratio=0.5))


# =============================================================================
# FindShortestPath pipeline: early stop, depth-aware cache, extension,
# bodyId edge-limit gating (drives _find_paths_core with stubbed fetching)
# =============================================================================

_PIPELINE_TYPES = {"S": "TS", "A": "TA", "B": "TB", "C": "TC", "T": "TT",
                   "Z": "TZ", "D": "TD", "T2": "TT"}


def _make_pipeline_fc(monkeypatch, tmp_path, edges, max_interlayer,
                      graph_edge_limit=0, target_ids=("T",), min_synapse=1):
    """Build a FindNeuronConnection wired for an offline _find_paths_core run.

    Returns (fc, fetch_calls, logs). Connection fetching is served from the
    static ``edges`` list (bodyId_pre, bodyId_post, weight), filtered by the
    requested min_weight exactly like the real fetch; every upstream fetch is
    recorded in ``fetch_calls``.
    """
    import coana

    edge_set = [(str(u), str(v), w) for (u, v, w) in edges]

    fetch_calls = []

    def fake_fetch(self, upstream_bodyIds, downstream_bodyIds=None,
                   min_weight=None, min_conn_ratio=None,
                   min_traversal_prob=None, **kwargs):
        ups = {str(u) for u in upstream_bodyIds}
        fetch_calls.append(sorted(ups))
        weight_floor = int(min_weight) if min_weight else 1
        rows = [(u, v, w) for (u, v, w) in edge_set
                if u in ups and w >= weight_floor]
        if min_weight and min_weight > 1 \
                and len(rows) < len([e for e in edge_set if e[0] in ups]):
            # mimic the real fetch: exclusions flip the per-run flag
            self._min_synapse_excluded = True
        if not rows:
            return pd.DataFrame(columns=["bodyId_pre", "bodyId_post",
                                         "weight", "type_pre", "type_post"])
        return pd.DataFrame({
            "bodyId_pre": [r[0] for r in rows],
            "bodyId_post": [r[1] for r in rows],
            "weight": [r[2] for r in rows],
            "type_pre": [_PIPELINE_TYPES[r[0]] for r in rows],
            "type_post": [_PIPELINE_TYPES[r[1]] for r in rows],
        })

    def fake_neurons(self, bodyIds, columns=None, **kwargs):
        ids = [str(b) for b in bodyIds]
        return pd.DataFrame({
            "bodyId": ids,
            "type": [_PIPELINE_TYPES.get(b, b) for b in ids],
            "post": [100] * len(ids),
        })

    def fake_enrich(conn, traversal_probability_threshold=None, dataset=None,
                    script_path=None, target_neurons_df=None, label_mapper=None,
                    global_incoming_weights=None, separate_hemispheres=False,
                    global_incoming_body_weights=None, aggregate_method=None,
                    engine=None, **kwargs):
        conn_e = conn.with_columns([
            pl.lit(0.5).alias("traversal_probability"),
            pl.lit(0.5).alias("connection_ratio"),
        ])
        conn_t = (conn_e.group_by(["type_pre", "type_post"])
                  .agg(pl.col("weight").sum())
                  .with_columns([
                      pl.lit(0.5).alias("traversal_probability"),
                      pl.lit(0.5).alias("connection_ratio"),
                  ]))
        return conn_e, conn_t, None

    class FakeVisualizePath:
        def __init__(self, *args, **kwargs):
            self.G_network = None

        def visualize(self, **kwargs):
            pass

    monkeypatch.setattr(coana.FindNeuronConnection,
                        "_fetch_connections_with_cache", fake_fetch)
    monkeypatch.setattr(coana.FindNeuronConnection,
                        "_fetch_neurons_local_or_api", fake_neurons)
    monkeypatch.setattr(coana.FindNeuronConnection,
                        "_fetch_total_incoming_weight_by_type",
                        lambda self, *a, **k: None)
    monkeypatch.setattr(coana.FindNeuronConnection,
                        "_fetch_total_incoming_weight",
                        lambda self, *a, **k: None)
    monkeypatch.setattr(coana.sv, "EnrichConnectionTable", fake_enrich)
    monkeypatch.setattr(coana, "VisualizePath", FakeVisualizePath)

    fc = object.__new__(coana.FindNeuronConnection)
    logs = []
    fc._vprint = lambda msg="", level="full", end="\n", flush=False: logs.append(str(msg))
    fc.source_df = pd.DataFrame({"bodyId": ["S"], "type": ["TS"]})
    fc.target_df = pd.DataFrame({"bodyId": list(target_ids),
                                 "type": [_PIPELINE_TYPES[t] for t in target_ids]})
    fc.saveas = ""
    fc.output_dir = str(tmp_path)
    fc.source_fname = "src"
    fc.target_fname = "tgt"
    fc.dataset = "test:v1"
    fc._dataset_safe = "test_v1"
    fc.max_interlayer = max_interlayer
    fc.min_synapse_num = min_synapse
    fc.min_ratio = 0.0
    fc.min_traversal_probability = 0.0
    fc.separate_hemispheres = False
    fc.hemisphere_filter = "both"
    fc.filter_by = "bodyId"
    fc.exclude_intra_type_connections = False
    fc.parameter_dict = {}
    fc.parameter_df = pd.DataFrame()
    fc.find_reciprocal = False
    fc.visualize_before_reconstruct = False
    fc.pathfinding = "MemoizedDFS"
    fc.graph_edge_limit_bodyid = graph_edge_limit
    fc.edgeN_limit = 0
    fc.network_layout = "hierarchical"
    fc.showfig = False
    fc.output_format = "csv"
    fc.skip_bodyId = True
    fc.pathN_to_show = 0
    fc.keyword_in_path_to_remove = ["None"]
    fc.label_mapper = None
    fc.symmetry_analysis = False
    fc.keep_only_hemisphere_conserved_connections = False
    fc.script_path = str(PROJECT_ROOT)
    fc.use_cache = True
    fc.cache_only = False
    fc.verbose_mode = "silent"
    fc._warn_notes = []
    return fc, fetch_calls, logs


_CHAIN_EDGES = [("S", "A", 5), ("A", "B", 5), ("B", "C", 5), ("C", "T", 5),
                ("T", "Z", 5)]


class TestFindShortestPathPipeline:
    def setup_method(self):
        _FINDALLPATH_GRAPH_CACHE.clear()

    def teardown_method(self):
        _FINDALLPATH_GRAPH_CACHE.clear()

    def test_shortest_early_stops_at_target_discovery(self, monkeypatch, tmp_path):
        fc, fetch_calls, logs = _make_pipeline_fc(
            monkeypatch, tmp_path, _CHAIN_EDGES, max_interlayer=99)
        fc.FindShortestPath()

        # Early stop: layers S, A, B, C fetched; T's outgoing (T->Z) never.
        assert fetch_calls == [["S"], ["A"], ["B"], ["C"]]
        assert any("stopping discovery early" in m for m in logs)
        # Depth-aware cache records the fetched depth + completeness.
        entry = next(iter(_FINDALLPATH_GRAPH_CACHE.values()))
        assert entry["depth"] == 4 and entry["complete"] is True
        # High bound (semi-unlimited) -> exact L{bound} folder name.
        assert os.path.basename(fc.allpath_folder).startswith("findshortestpath_")
        assert "_L99" in os.path.basename(fc.allpath_folder)
        # The one shortest path (TS->TA->TB->TC->TT) was saved.
        path_csv = os.path.join(fc.allpath_folder, "src_to_tgt_allpaths_type.csv")
        assert os.path.exists(path_csv)
        df = pl.read_csv(path_csv)
        assert len(df) == 1
        assert "TS->TA->TB->TC->TT" in str(df.row(0))

    def test_shortest_cache_extension_resumes_from_cached_depth(
            self, monkeypatch, tmp_path):
        # Run 1: capped at 1 intermediate layer -> discovery stops at the
        # depth cap WITHOUT finding the target (depth=2, incomplete).
        fc1, fetch_calls1, _ = _make_pipeline_fc(
            monkeypatch, tmp_path, _CHAIN_EDGES, max_interlayer=1)
        fc1.FindShortestPath()
        assert fetch_calls1 == [["S"], ["A"]]
        entry = next(iter(_FINDALLPATH_GRAPH_CACHE.values()))
        assert entry["depth"] == 2 and entry["complete"] is False

        # Run 2: high bound (semi-unlimited) on the same query -> the shallow
        # cache must be EXTENDED (resume at layer 2), not reused or rebuilt.
        fc2, fetch_calls2, logs2 = _make_pipeline_fc(
            monkeypatch, tmp_path, _CHAIN_EDGES, max_interlayer=99)
        fc2.FindShortestPath()
        assert any("Extending cached graph" in m for m in logs2)
        assert fetch_calls2 == [["B"], ["C"]]  # resumed, no re-fetch of S/A
        entry = next(iter(_FINDALLPATH_GRAPH_CACHE.values()))
        assert entry["depth"] == 4 and entry["complete"] is True
        path_csv = os.path.join(fc2.allpath_folder, "src_to_tgt_allpaths_type.csv")
        df = pl.read_csv(path_csv)
        assert len(df) == 1  # the full chain path survived the extension

    def test_shortest_edge_limit_off_by_default_and_opt_in_warns(
            self, monkeypatch, tmp_path):
        # Default (limit=0): no trim, no shortest-trim warning note.
        fc, _, _ = _make_pipeline_fc(
            monkeypatch, tmp_path, _CHAIN_EDGES, max_interlayer=99,
            graph_edge_limit=0)
        fc.FindShortestPath()
        assert not any("shortest mode + graph edge limit" in n
                       for n in fc._warn_notes)

        _FINDALLPATH_GRAPH_CACHE.clear()
        # Opt-in limit: trimming applies and the distance-inflation note is
        # recorded (chain graph survives the integrity-preserving trim).
        fc2, _, _ = _make_pipeline_fc(
            monkeypatch, tmp_path, _CHAIN_EDGES, max_interlayer=99,
            graph_edge_limit=2)
        fc2.FindShortestPath()
        assert any("shortest mode + graph edge limit" in n
                   for n in fc2._warn_notes)
        path_csv = os.path.join(fc2.allpath_folder, "src_to_tgt_allpaths_type.csv")
        assert len(pl.read_csv(path_csv)) == 1

    def test_shortest_cap_boundary_warns_to_raise_the_bound(
            self, monkeypatch, tmp_path):
        """A pair whose shortest path reaches the Max Layers cap triggers an
        always-visible warning suggesting a higher bound (0 = unlimited)."""
        # Chain S->A->B->C->T = 4 hops; cap 3 intermediate layers -> the
        # found path (4 edges) hits the bound.
        fc, _, logs = _make_pipeline_fc(
            monkeypatch, tmp_path, _CHAIN_EDGES, max_interlayer=3)
        fc.FindShortestPath()
        assert any("reach the Max Layers bound" in m for m in logs)
        assert any("Increase Max Layers" in m for m in logs)

        _FINDALLPATH_GRAPH_CACHE.clear()
        # With an adequate bound the warning must NOT fire.
        fc2, _, logs2 = _make_pipeline_fc(
            monkeypatch, tmp_path, _CHAIN_EDGES, max_interlayer=4)
        fc2.FindShortestPath()
        assert not any("reach the Max Layers bound" in m for m in logs2)

    def test_shortest_mode_unset_limit_defaults_to_no_trim(
            self, monkeypatch, tmp_path):
        """path_mode='shortest' with graph_edge_limit_bodyid unset (None)
        behaves like 0 — no trimming, no warning note."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        import coana
        import polars as pl

        layers = [pl.DataFrame({
            "bodyId_pre": ["S", "S", "A", "B"],
            "bodyId_post": ["A", "B", "T", "T"],
            "weight": [1, 3, 4, 2],
        })]
        for path_mode in ("shortest", "all"):
            fc = object.__new__(coana.FindNeuronConnection)
            fc.max_interlayer = 3
            fc.graph_edge_limit_bodyid = None  # unset -> per-mode default
            fc._vprint = lambda *a, **k: None
            fc._warn_notes = []
            out = fc._trim_bodyid_edges(layers, ["S"], ["T"], path_mode=path_mode)
            rows = out.height if hasattr(out, "height") else len(out)
            # 4 rows, nothing trimmed in either mode at this size
            assert rows == 4, (path_mode, rows)
            if path_mode == "shortest":
                # the whole point: unset limit in shortest mode must not
                # even enter the trim path (no warning note)
                assert not fc._warn_notes, fc._warn_notes

    def test_normalized_keyword_filter_sentinel_is_noop(self):
        """The 'None' sentinel (field default / UI convention) must never
        reach path_filter as a literal keyword — paths containing 'None'
        must not be dropped when the user left the filter empty."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        import coana

        fc = object.__new__(coana.FindNeuronConnection)
        for raw in (None, "None", ["None"], [], ["DN1p"]):
            fc.keyword_in_path_to_remove = raw
            out = fc._normalized_keyword_filter()
            if raw == ["DN1p"]:
                assert out == ["DN1p"]
            else:
                assert out is None, raw

    def test_shortest_cache_rebuilds_when_threshold_rises(
            self, monkeypatch, tmp_path):
        """A shortest cache early-stopped at threshold t must NOT be reused
        (filter-up) for a higher threshold: dropping the edges that triggered
        the early stop would hide the deeper layers the higher threshold
        needs. The run must rebuild discovery from scratch."""
        edges = [("S", "A", 5), ("A", "T", 5),          # t5 route (2 hops)
                 ("S", "B", 10), ("B", "C", 10), ("C", "T", 10)]  # t10 route (3 hops)

        # Run 1 @ threshold 5: early-stops at depth 2 with S->A->T.
        fc1, fetch_calls1, _ = _make_pipeline_fc(
            monkeypatch, tmp_path, edges, max_interlayer=99, min_synapse=5)
        fc1.FindShortestPath()
        # Layer 1 discovers both A (t5 route) and B (S->B has weight 10);
        # the early stop fires after the layer-2 fetch finds T via A.
        assert fetch_calls1 == [["S"], ["A", "B"]]
        entry = next(iter(_FINDALLPATH_GRAPH_CACHE.values()))
        assert entry["depth"] == 2 and entry["complete"] is True
        csv1 = os.path.join(fc1.allpath_folder, "src_to_tgt_allpaths_type.csv")
        df1 = pl.read_csv(csv1)
        assert len(df1) == 1 and "TS->TA->TT" in str(df1.row(0))

        # Run 2 @ threshold 10 in the SAME process: the t5 cache must be
        # rebuilt (not filtered up), discovering the deeper 3-hop route.
        fc2, fetch_calls2, logs2 = _make_pipeline_fc(
            monkeypatch, tmp_path, edges, max_interlayer=99, min_synapse=10)
        fc2.FindShortestPath()
        assert any("rebuilding" in m for m in logs2)
        assert fetch_calls2[0] == ["S"]          # fresh discovery from layer 0
        assert ["A"] not in fetch_calls2         # t5-only branch never fetched
        csv2 = os.path.join(fc2.allpath_folder, "src_to_tgt_allpaths_type.csv")
        df2 = pl.read_csv(csv2)
        assert len(df2) == 1
        assert "TS->TB->TC->TT" in str(df2.row(0))

    def test_shortest_type_paths_filtered_to_pair_minimum(
            self, monkeypatch, tmp_path):
        """Two target instances of the SAME type at different distances:
        bodyId shortest paths are per-instance, but the type-level output
        keeps only the per-(source type, target type) minimum length."""
        edges = [("S", "A", 10), ("A", "T", 10),              # TS->TT in 2 hops
                 ("S", "B", 10), ("B", "C", 10), ("C", "T2", 10)]  # TT via T2 in 3 hops
        fc, _, logs = _make_pipeline_fc(
            monkeypatch, tmp_path, edges, max_interlayer=99,
            target_ids=("T", "T2"))
        fc.FindShortestPath()
        assert any("per-type-pair minimum length" in m for m in logs)
        path_csv = os.path.join(fc.allpath_folder, "src_to_tgt_allpaths_type.csv")
        df = pl.read_csv(path_csv)
        assert len(df) == 1
        assert "TS->TA->TT" in str(df.row(0))

    def test_find_all_path_mode_unchanged_by_refactor(self, monkeypatch, tmp_path):
        # 'all' mode keeps full-depth discovery (no early stop) and the
        # findallpath_ folder prefix with the depth in the suffix.
        fc, fetch_calls, _ = _make_pipeline_fc(
            monkeypatch, tmp_path, _CHAIN_EDGES, max_interlayer=3)
        fc.FindAllPath()
        # bound = 4 layer tables: S, A, B, C (T's layer would be the 5th).
        assert fetch_calls == [["S"], ["A"], ["B"], ["C"]]
        base = os.path.basename(fc.allpath_folder)
        assert base.startswith("findallpath_") and "_L3" in base
        path_csv = os.path.join(fc.allpath_folder, "src_to_tgt_allpaths_type.csv")
        assert len(pl.read_csv(path_csv)) == 1


class TestLimitReachedFlags:
    """The per-run flags gating the config-derived warning notes are set only
    when the limit actually bit: the depth cap ended discovery with a live
    frontier, and min_synapse_num dropped connections during fetching."""

    def setup_method(self):
        _FINDALLPATH_GRAPH_CACHE.clear()

    def teardown_method(self):
        _FINDALLPATH_GRAPH_CACHE.clear()

    def test_depth_cap_flag_false_when_bound_never_hit(self, monkeypatch, tmp_path):
        # Shortest run, semi-unlimited bound: discovery stops early once all
        # targets are found — the bound never bit.
        fc, _, _ = _make_pipeline_fc(
            monkeypatch, tmp_path, _CHAIN_EDGES, max_interlayer=99)
        fc.FindShortestPath()
        assert fc._depth_cap_reached is False

        _FINDALLPATH_GRAPH_CACHE.clear()
        # All mode, the chain dries up exactly AT the cap boundary (the last
        # fetch discovers no new neurons): no deeper paths can exist.
        fc2, _, _ = _make_pipeline_fc(
            monkeypatch, tmp_path, _CHAIN_EDGES, max_interlayer=5)
        fc2.FindAllPath()
        assert fc2._depth_cap_reached is False

    def test_depth_cap_flag_true_with_live_frontier_and_note_written(
            self, monkeypatch, tmp_path):
        # max_interlayer=4 on the 5-hop chain: all-mode discovery hits the cap
        # having just discovered Z (frontier still alive) — deeper paths may
        # exist but were never searched.
        fc, _, logs = _make_pipeline_fc(
            monkeypatch, tmp_path, _CHAIN_EDGES, max_interlayer=4)
        fc.FindAllPath()
        assert fc._depth_cap_reached is True
        assert any("Depth cap reached" in m for m in logs)
        # cached discovery agrees: stopped at the depth cap (incomplete)
        entry = next(iter(_FINDALLPATH_GRAPH_CACHE.values()))
        assert entry["complete"] is False
        # the [depth] note lands in user_warning_notes.txt
        path = os.path.join(fc.allpath_folder, "user_warning_notes.txt")
        text = open(path, encoding="utf-8").read()
        assert "[depth] max_interlayer=4" in text

    def test_min_synapse_flag_follows_actual_exclusions(self, monkeypatch, tmp_path):
        # Every fetched edge is >= threshold: nothing excluded, no flag.
        edges = [("S", "A", 2), ("A", "T", 8),
                 ("S", "B", 8), ("B", "T", 8)]
        fc, _, _ = _make_pipeline_fc(
            monkeypatch, tmp_path, edges, max_interlayer=99, min_synapse=2)
        fc.FindShortestPath()
        assert fc._min_synapse_excluded is False

        _FINDALLPATH_GRAPH_CACHE.clear()
        # Threshold 3 drops the weak S->A edge: exclusions flip the flag.
        fc2, _, _ = _make_pipeline_fc(
            monkeypatch, tmp_path, edges, max_interlayer=99, min_synapse=3)
        fc2.FindShortestPath()
        assert fc2._min_synapse_excluded is True
