"""Tests for the shared in-memory node simplification module.

Covers `skeleton_simplification.simplify_skeleton_nodes`:
- FAFB fast tube target (25% node retention) with the topology-preservation
  floor (root / branch point / tip always retained)
- NeuPrint 50% and FAFB 90% line-mode defaults
- input immutability (the raw SWC source is never mutated)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import navis  # noqa: E402
from skeleton_simplification import (  # noqa: E402
    FAFB_FAST_NODE_RETENTION,
    FAFB_LINE_NODE_REDUCTION,
    MIN_RETAINED_NODES,
    NEUPRINT_LINE_NODE_REDUCTION,
    simplify_skeleton_nodes,
)


def make_chain(n_nodes=2000):
    """Straight chain: one root, one tip, everything else slab."""
    types = (["root"] + ["slab"] * max(0, n_nodes - 2) + ["end"])
    nodes = pd.DataFrame({
        "node_id": np.arange(n_nodes, dtype=np.int64),
        "parent_id": np.array([-1] + list(range(n_nodes - 1)),
                               dtype=np.int64),
        "x": np.arange(n_nodes, dtype=float) * 1000.0,
        "y": np.zeros(n_nodes),
        "z": np.zeros(n_nodes),
        "radius": np.ones(n_nodes),
        "type": types[:n_nodes],
    })
    nrn = navis.TreeNeuron(nodes)
    # disable the constructor's soma auto-detection (it flags most nodes of
    # a plain chain, turning them into downsample fix points)
    nrn.soma = None
    return nrn


def make_y_neuron():
    """Trunk 0..300 with a side branch 301..400 off node 300."""
    n_nodes = 401
    parent = [-1] + list(range(n_nodes - 1))
    parent[301] = 300  # side branch starts at trunk node 300
    types = ["slab"] * n_nodes
    types[0] = "root"
    types[300] = "branch"
    types[400] = "end"    # side-branch tip
    types[-1] = "end"     # trunk tip
    nodes = pd.DataFrame({
        "node_id": np.arange(n_nodes, dtype=np.int64),
        "parent_id": np.array(parent, dtype=np.int64),
        "x": np.arange(n_nodes, dtype=float) * 1000.0,
        "y": np.where(np.arange(n_nodes) > 300, 500.0, 0.0),
        "z": np.zeros(n_nodes),
        "radius": np.ones(n_nodes),
        "type": types,
    })
    nrn = navis.TreeNeuron(nodes)
    nrn.soma = None
    return nrn


class TestFastNodeRetention:
    def test_fafb_fast_retains_25_percent_of_nodes(self):
        neuron = make_chain(2000)
        reduction = 1.0 - FAFB_FAST_NODE_RETENTION
        simplified, stats = simplify_skeleton_nodes(neuron, reduction)

        assert FAFB_FAST_NODE_RETENTION == pytest.approx(0.25)
        assert stats["raw_nodes"] == 2000
        assert stats["target_nodes"] == 500
        # Achieved count is within the module's 5% tolerance of the
        # 25%-retention target.
        assert abs(stats["achieved_nodes"] - 500) <= 25
        assert stats["achieved_nodes"] < stats["raw_nodes"]
        assert stats["factor"] is not None and stats["factor"] > 1

    def test_topology_floor_keeps_root_branch_and_tips(self):
        neuron = make_y_neuron()
        simplified, stats = simplify_skeleton_nodes(neuron, 0.90)

        kept = set(simplified.nodes.node_id.values)
        # Root, the branch point, and the terminal node all survive.
        assert 0 in kept
        assert 300 in kept
        assert 400 in kept
        assert neuron.nodes.node_id.max() in kept
        assert stats["achieved_nodes"] <= stats["target_nodes"] + 10

    def test_reports_authoritative_achieved_count(self):
        """The achieved count is reported honestly even when the target
        cannot be reached exactly (navis' quantized factor behavior)."""
        neuron = make_chain(2000)
        simplified, stats = simplify_skeleton_nodes(neuron, 0.90)
        assert stats["target_nodes"] == 200
        assert stats["achieved_nodes"] <= 205
        assert stats["achieved_nodes"] == simplified.n_nodes


class TestLineModeDefaults:
    def test_neuprint_line_default_is_50_percent_reduction(self):
        assert NEUPRINT_LINE_NODE_REDUCTION == pytest.approx(0.50)
        neuron = make_chain(2000)
        simplified, stats = simplify_skeleton_nodes(
            neuron, NEUPRINT_LINE_NODE_REDUCTION)
        assert stats["target_nodes"] == 1000
        # navis can never keep more than ~1/3 of a plain chain (its
        # minimum effective stride is 3), so the module returns the
        # closest achievable count instead of stalling at a suboptimal
        # factor.
        assert stats["achieved_nodes"] <= stats["target_nodes"]
        assert stats["achieved_nodes"] >= 2000 / 3 - 5
        assert stats["factor"] is not None

    def test_fafb_line_default_is_90_percent_reduction(self):
        assert FAFB_LINE_NODE_REDUCTION == pytest.approx(0.90)
        neuron = make_chain(2000)
        simplified, stats = simplify_skeleton_nodes(
            neuron, FAFB_LINE_NODE_REDUCTION)
        assert stats["target_nodes"] == 200
        assert abs(stats["achieved_nodes"] - 200) <= 10


class TestInputSafety:
    def test_input_neuron_is_never_mutated(self):
        neuron = make_chain(2000)
        original = neuron.nodes.copy()
        original_n_nodes = neuron.n_nodes

        for reduction in (0.75, 0.90, 0.50):
            simplified, _ = simplify_skeleton_nodes(neuron, reduction)
            assert simplified is not neuron
            assert neuron.n_nodes == original_n_nodes
            pd.testing.assert_frame_equal(neuron.nodes, original)

    def test_small_neurons_are_returned_as_is(self):
        neuron = make_chain(50)
        simplified, stats = simplify_skeleton_nodes(neuron, 0.90)

        assert stats["raw_nodes"] == 50
        assert stats["achieved_nodes"] == 50
        assert stats["factor"] is None
        assert simplified is not neuron  # still a copy, but untouched
        assert simplified.n_nodes == neuron.n_nodes
        assert neuron.n_nodes == 50

    def test_floor_respects_min_retained_nodes(self):
        neuron = make_chain(150)
        simplified, stats = simplify_skeleton_nodes(neuron, 0.90)
        # 150 * 0.10 = 15 < MIN_RETAINED_NODES -> target is the floor.
        assert stats["target_nodes"] == MIN_RETAINED_NODES == 100
        assert stats["achieved_nodes"] <= 150

    def test_non_treeneuron_input_raises(self):
        vertices = np.array([
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        ])
        faces = np.array([(0, 1, 2)], dtype=np.int64)
        mesh = navis.MeshNeuron({"vertices": vertices, "faces": faces})
        with pytest.raises(TypeError):
            simplify_skeleton_nodes(mesh, 0.5)

    def test_extra_preserved_nodes_survive(self):
        neuron = make_chain(2000)
        preserve = [37, 1234]
        simplified, _ = simplify_skeleton_nodes(
            neuron, 0.90, preserve_nodes=preserve)
        kept = set(simplified.nodes.node_id.values)
        assert 37 in kept
        assert 1234 in kept
