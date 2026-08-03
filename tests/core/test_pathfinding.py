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
