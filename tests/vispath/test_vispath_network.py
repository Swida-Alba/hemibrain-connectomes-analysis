#!/usr/bin/env python
"""
Regression tests for the vispath network canvas.

Covers:
  - Structure of the generated network HTML: the operation-history
    dropdown, every mutating user operation recorded via pushHistory
    (hide node / hide edge / drag / filter / toggles / layout import),
    complete state snapshots (deep copies of data/positions plus the
    visibility toggles, edge filter and view), and the order-independent
    dead-end fixpoint over the current graph (as defined by the
    hide-edges filter).
  - Dead-end detection semantics executed in Node with headless
    Cytoscape, using the REAL functions extracted from the generated
    HTML: propagation to fixpoint, filter-defined current graph,
    hidden/self-loop edge exclusion, idempotence.
  - Undo/redo history semantics executed in Node with headless
    Cytoscape + DOM stubs, using the REAL functions extracted from the
    generated HTML: deep-copied snapshots, data/position/class restore,
    filter and toggle-flag restore, jump-to-history, redo clearing,
    history bound.

The Node harnesses read the generated HTML artifact (written by
vispath.py in this repository) and only execute code extracted from
that trusted, locally-produced file.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))

from vispath_pkg.fast_graph_core import FastGraph  # noqa: E402
from vispath_pkg.vispath import VisualizePath  # noqa: E402

HERE = Path(__file__).parent


# =============================================================================
# Fixtures / helpers
# =============================================================================

def _build_network_html(output_path):
    """Generate a network HTML with the same small graph used by the
    Node harnesses: S->A->B->T source-target chain, a dead-end branch
    S->X->Y and an isolated start D->T."""
    df = pd.DataFrame(
        {
            "path_block": ["S>A>B>T", "S>X>Y", "D>T"],
            "weights": [[10, 20, 30], [5, 8], [3]],
        }
    )
    vp = VisualizePath(
        path_file=df,
        output_folder=str(output_path.parent),
        showfig=False,
        verbose=False,
        network_layout="dagre",
    )
    G = FastGraph()
    for u, v, w in [("S", "A", 10), ("A", "B", 20), ("B", "T", 30),
                    ("S", "X", 5), ("X", "Y", 8), ("D", "T", 3)]:
        G.add_edge(u, v, w)
        G.node_attrs.setdefault(u, {})["node_type"] = "intermediate"
        G.node_attrs.setdefault(v, {})["node_type"] = "intermediate"
    G.node_attrs["S"]["node_type"] = "source"
    G.node_attrs["T"]["node_type"] = "target"
    vp._plot_cytoscape_network(G, output_path=str(output_path), layout="dagre", open_browser=False)
    return output_path


@pytest.fixture(scope="module")
def network_html(tmp_path_factory):
    out = tmp_path_factory.mktemp("vispath_html") / "network_test.html"
    return _build_network_html(out)


@pytest.fixture(scope="module")
def node_cache(tmp_path_factory):
    """Keep npm's test dependency cache inside pytest's disposable tree."""
    return tmp_path_factory.mktemp("vispath_node")


def _script_text(network_html):
    import re
    html = network_html.read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert scripts, "no inline scripts found in generated HTML"
    return "\n".join(scripts)


def _ensure_node_with_cytoscape(node_cache):
    """Return the node executable, installing headless Cytoscape into a
    pytest-owned temporary directory once. Skips when node/npm or network
    access is missing."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    cy_path = node_cache / "node_modules" / "cytoscape"
    if cy_path.exists():
        return node
    npm = shutil.which("npm")
    if not npm:
        pytest.skip("npm not available for installing cytoscape")
    node_cache.mkdir(parents=True, exist_ok=True)
    npm_env = os.environ.copy()
    npm_env["npm_config_cache"] = str(node_cache / ".npm-cache")
    res = subprocess.run(
        ["npm", "install", "cytoscape@3.28.1", "--no-audit", "--no-fund", "--prefix", str(node_cache)],
        capture_output=True, text=True, timeout=600, env=npm_env,
    )
    if res.returncode != 0 or not cy_path.exists():
        pytest.skip(f"could not install cytoscape for Node tests: {res.stderr[-300:]}")
    return node


def _run_node_harness(node, harness_name, network_html, node_cache):
    harness = HERE / harness_name
    res = subprocess.run(
        [node, str(harness), str(node_cache), str(network_html)],
        capture_output=True, text=True, timeout=300,
    )
    return res


# =============================================================================
# Structural checks on the generated HTML
# =============================================================================

class TestGeneratedHtmlStructure:
    def test_history_dropdown_present(self, network_html):
        html = network_html.read_text(encoding="utf-8")
        js = _script_text(network_html)
        # dropdown markup lives in the HTML body; its logic in the script
        assert 'id="historyList"' in html
        assert "jumpToHistory(this.selectedIndex)" in html
        assert "▶ Current state" in html
        assert "function jumpToHistory" in js
        assert "function updateHistoryList" in js

    def test_all_mutating_operations_recorded(self, network_html):
        js = _script_text(network_html)
        # right-click hide node / hide edge (both context-menu paths)
        assert "pushHistory('Hide node')" in js
        assert "pushHistory('Hide edge')" in js
        # drag relocation is committed to history on dragfree; the pre-drag
        # stash MUST be on 'grab' — Cytoscape.js has no node-level
        # 'dragstart' event (only a core pan gesture), so wiring to
        # dragstart silently records nothing and undo cannot restore moves
        assert "pushStateHistory('Move nodes', pendingDragState)" in js
        assert "function registerDragHistory" in js
        assert "cy.on('grab', 'node'" in js
        assert "cy.on('dragstart', 'node'" not in js
        # edge filter changes are recorded per distinct value
        assert "pushHistory('Edge filter')" in js
        # layout import and label-position toggle are recorded
        assert "pushHistory('Import layout')" in js
        assert "pushHistory('Toggle label position')" in js
        # geometry editing (precise size/position) and alignment are recorded
        assert "pushHistory('Resize element')" in js
        assert "pushHistory('Align nodes')" in js
        # the three visibility toggles are recorded
        for label in ("Toggle self-loops", "Toggle orphans", "Toggle dead-ends"):
            assert f"pushHistory('{label}')" in js

    def test_geometry_editor_present(self, network_html):
        """Precise size/position editing and alignment helpers: numeric
        inputs in the Selected Element(s) panel, node-vs-edge row groups,
        align buttons, and the apply/align functions."""
        html = network_html.read_text(encoding="utf-8")
        js = _script_text(network_html)
        # geometry inputs: X/Y/size for nodes, width for edges
        for elem_id in ('selGeomX', 'selGeomY', 'selGeomSize', 'selGeomWidth',
                        'geomNodeGroup', 'geomEdgeGroup',
                        'alignHBtn', 'alignVBtn'):
            assert f'id="{elem_id}"' in html, f'missing element {elem_id}'
        assert 'onclick="applySelectedGeometry()"' in html
        assert 'onclick="alignSelectedNodes(\'h\')"' in html
        assert 'onclick="alignSelectedNodes(\'v\')"' in html
        assert "function applySelectedGeometry" in js
        assert "function alignSelectedNodes" in js
        assert "function syncSelectedGeometryInputs" in js
        assert "function updateAlignButtons" in js
        # selection sync hooks: tap fills the inputs, dragfree refreshes
        # them after a manual drag, clearSelection resets the rows
        assert "syncSelectedGeometryInputs(element)" in js
        assert "syncSelectedGeometryInputs(evt.target)" in js
        assert "syncSelectedGeometryInputs(null)" in js
        # align-button enabled state follows any selection change (nodes
        # AND edges), and the size/position modifiers + confirm button are
        # hidden while nothing is selected
        assert "cy.on('select unselect', 'node, edge'" in js
        assert 'id="applyGeometryBtn"' in html
        assert "applyGeom.style.display = anySelected ? 'block' : 'none'" in js
        # manual edge widths are marked so they are recognizable as custom
        assert "e.data('customSize', true)" in js

    def test_selection_events_resync_geometry_controls(self, network_html):
        """A select event must re-show and repopulate geometry controls.

        Cytoscape can deliver the tap callback before it updates the
        element's selected state.  The selection listener therefore needs to
        re-sync after selection, while the final unselect must clear the rows.
        """
        js = _script_text(network_html)
        start = js.index("cy.on('select unselect', 'node, edge'")
        end = js.index("        });", start) + len("        });")
        selection_handler = js[start:end]
        assert "const selected = cy.$(':selected')" in selection_handler
        assert "syncSelectedGeometryInputs(null)" in selection_handler
        assert "syncSelectedGeometryInputs(primary)" in selection_handler

    def test_snapshots_are_complete_deep_copies(self, network_html):
        js = _script_text(network_html)
        # data()/position() return live references in Cytoscape; snapshots
        # must deep-copy them or later mutations corrupt earlier entries.
        assert "JSON.parse(JSON.stringify(n.data()))" in js
        assert "JSON.parse(JSON.stringify(e.data()))" in js
        assert "position: { x: n.position().x, y: n.position().y }" in js
        # snapshot carries the visibility toggles, filter and view so undo
        # restores the full UI state
        for field in ("deadEndsHidden", "orphansHidden", "selfLoopsHidden",
                      "hemisphereMirrorEnabled", "filterValue", "labelPosition",
                      "zoom", "pan"):
            assert field in js
        assert "syncToggleButtons()" in js
        # per-element style overrides (color/size bypasses) are captured
        # from _private.style — ele.json() does NOT expose bypasses in
        # Cytoscape 3.28.1 — and re-applied on restore, so individual
        # edits round-trip through undo/redo
        assert "function captureStyleBypass" in js
        assert "el._private && el._private.style" in js
        # computed (non-bypass) entries such as the default :active overlay
        # must never be snapshotted, or undo turns the transient drag
        # shading into a permanent bypass
        assert "if (!v || v.bypass !== true) return;" in js
        assert "style: captureStyleBypass(n)" in js
        assert "style: captureStyleBypass(e)" in js
        assert "if (n.style) cy.getElementById(n.data.id).style(n.style)" in js
        assert "if (e.style) cy.getElementById(e.data.id).style(e.style)" in js

    def test_selection_highlight_visible(self, network_html):
        """Selection feedback must be clearly visible: the default highlight
        color is a saturated orange (the old light-yellow #FFFFE0 was nearly
        invisible on the white canvas), selected nodes get a thick border
        plus an overlay halo, selected edges a thicker line."""
        js = _script_text(network_html)
        assert "#FFFFE0" not in js
        node_sel = js.index("selector: 'node:selected'")
        nblock = js[node_sel:node_sel + 800]
        assert "'border-width': '4px'" in nblock
        assert "'overlay-color': '#FF9800'" in nblock
        assert "'overlay-opacity': 0.25" in nblock
        edge_sel = js.index("selector: 'edge:selected'")
        eblock = js[edge_sel:edge_sel + 500]
        assert "'line-color': '#FF9800'" in eblock
        assert "3, 16" in eblock

    def test_dead_end_fixpoint_is_order_independent(self, network_html):
        js = _script_text(network_html)
        # additions are collected per pass and applied in batch, so the
        # result does not depend on node iteration order
        assert "const newlyDead = []" in js
        assert "newlyDead.forEach(id => deadEndSet.add(id))" in js
        # edges of the current graph are consistently defined
        assert "function isEdgeInCurrentGraph" in js
        assert "function reapplyOrphanHiding" in js
        # dead ends are re-detected whenever the graph changes
        assert "reapplyDeadEndHiding();" in js

    def test_dead_end_redetection_hooks(self, network_html):
        js = _script_text(network_html)
        # every operation that changes the visible graph must re-detect
        assert js.count("reapplyDeadEndHiding();") >= 5  # filter, hide node, self-loops, orphans, import

    def test_edge_anchors_follow_actual_node_sizes(self, network_html):
        """Individually resized nodes (geometry editor) must re-anchor their
        edges to their ACTUAL size, not the global node-size slider."""
        js = _script_text(network_html)
        # refreshEdgeStyles derives the anchor distances from each endpoint
        # node's computed width, with the global slider only as fallback
        assert "const sourceNodeSize = edge.source().numericStyle('width')" in js
        assert "const targetNodeSize = edge.target().numericStyle('width')" in js
        assert "let sourceDistance = sourceNodeSize / 2;" in js
        assert "let targetDistance = targetNodeSize / 2;" in js
        # the geometry editor refreshes edge styles after resizing
        assert "refreshEdgeStyles(false);  // keep endpoints/offsets attached to resized nodes" in js

    def test_reciprocal_detection_excludes_edge_itself(self, network_html):
        """A one-way edge must never be treated as its own parallel: the
        reciprocal check counts edges per direction, so only a DIFFERENT
        edge in the reverse direction triggers the offset branch. Otherwise
        every edge takes the reciprocal-offset branch and its arrows miss
        the node centers."""
        js = _script_text(network_html)
        # per-direction counts, not a set of all edges
        assert "const visibleEdgeCounts = new Map();" in js
        assert "const key = e.source().id() + '→' + e.target().id();" in js
        assert "visibleEdgeCounts.set(key, (visibleEdgeCounts.get(key) || 0) + 1);" in js
        # parallel check looks up the REVERSE direction only
        assert "const hasVisibleParallel = (visibleEdgeCounts.get(target + '→' + source) || 0) > 0;" in js

    def test_deadend_hidden_classes_have_display_none_style(self, network_html):
        """Regression: dead-end classes were assigned and counted, but no
        stylesheet rule hid them, so Hide Dead Ends reported counts without
        actually hiding anything."""
        js = _script_text(network_html)
        node_block = ("selector: 'node.deadend-hidden'" in js
                      and "'display': 'none'" in js)
        edge_block = ("selector: 'edge.deadend-hidden'" in js
                      and "'display': 'none'" in js)
        assert node_block, 'missing node.deadend-hidden { display: none } rule'
        assert edge_block, 'missing edge.deadend-hidden { display: none } rule'

    def test_layout_and_orphan_helpers_ignore_hidden_elements(self, network_html):
        """Layout algorithms must ignore hidden nodes/edges: the mirror
        placeholder builder and the layout/fit entry points filter through
        the class-based isVisibleElement, and orphans are recognized as dead
        ends (isOrphanNode, incl. self-loop-only nodes)."""
        js = _script_text(network_html)
        assert "function isVisibleElement" in js
        assert "function isOrphanNode" in js
        # mirror placeholders / positioning / fit / layout all use it
        assert js.count("filter(isVisibleElement)") >= 5
        # orphans are dead ends (isDeadEndNodeIn returns true on 0 in/out)
        assert "orphan: no connections in the current graph" in js
        # self-loops never count toward orphan connectivity
        assert "e.source().id() !== e.target().id()" in js


# =============================================================================
# Node-based logic tests (real functions extracted from the generated HTML)
# =============================================================================

class TestDeadEndLogicNode:
    def test_all_dead_end_scenarios(self, network_html, node_cache):
        node = _ensure_node_with_cytoscape(node_cache)
        res = _run_node_harness(node, "deadend_harness.js", network_html, node_cache)
        assert res.returncode == 0, (
            f"dead-end harness failed:\n{res.stdout}\n{res.stderr}"
        )
        assert "ALL DEAD-END TESTS PASSED" in res.stdout


class TestHistoryLogicNode:
    def test_all_history_scenarios(self, network_html, node_cache):
        node = _ensure_node_with_cytoscape(node_cache)
        res = _run_node_harness(node, "history_harness.js", network_html, node_cache)
        assert res.returncode == 0, (
            f"history harness failed:\n{res.stdout}\n{res.stderr}"
        )
        assert "ALL HISTORY TESTS PASSED" in res.stdout


# =============================================================================
# Network trim for plotting: source/target reservation + threshold warning
# =============================================================================

class TestNetworkTrimForPlot:
    """Plot trimming keeps complete strong paths and avoids dangling edges."""

    def _make_vp(self, messages=None):
        vp = object.__new__(VisualizePath)
        vp._vprint = lambda msg: messages.append(msg)
        vp.path_df = None
        return vp

    def test_no_trim_when_within_limit(self):
        messages = []
        vp = self._make_vp(messages)
        vp.edgeN_limit = 500
        G = FastGraph()
        G.add_edge("S", "A", 3)
        G.add_edge("A", "T", 5)
        vp.G_network = G
        assert vp._trim_network_for_plot() is G  # unchanged
        assert messages == []

    def test_fallback_trim_keeps_source_target_corridor_and_reports_threshold(self):
        messages = []
        vp = self._make_vp(messages)
        vp.edgeN_limit = 2
        G = FastGraph()
        # Boundary edges are weak, but are part of a complete S -> T corridor.
        G.add_edge("S", "A", 1)
        G.add_edge("B", "T", 2)
        G.node_attrs["S"] = {"node_type": "source"}
        G.node_attrs["T"] = {"node_type": "target"}
        G.node_attrs["A"] = {"node_type": "intermediate"}
        G.node_attrs["B"] = {"node_type": "intermediate"}
        # Strong intermediate edges complete the corridor.  X -> Y is a
        # disconnected decoy and must not survive merely because it is strong.
        G.add_edge("A", "M", 100)
        G.add_edge("M", "B", 90)
        G.add_edge("X", "Y", 80)
        vp.G_network = G

        G_plot = vp._trim_network_for_plot()
        kept = set(G_plot.edges())
        # Endpoint edges survive only because the full corridor survives.
        assert {("S", "A"), ("B", "T")} <= kept
        # The complete corridor is retained; the disconnected decoy is cut.
        assert len(kept) == 4
        assert ("X", "Y") not in kept  # weakest non-reserved cut
        # The warning carries the weakest edge actually retained.
        assert any("applied threshold: weight >= 1" in m for m in messages), messages

    def test_trim_reservation_capped_for_degenerate_source_target_classification(self):
        """Regression for the network_early preview: with an edge-list input
        every node is classified as source/target, so the raw reservation
        would swallow the whole graph and the limit would do nothing. The
        auto-reservation is capped at edgeN_limit and the output stays
        bounded (<= 2 x edgeN_limit)."""
        messages = []
        vp = self._make_vp(messages)
        vp.edgeN_limit = 3
        G = FastGraph()
        # every node is a source or a target (degenerate classification)
        G.add_edge("A", "B", 1)
        G.add_edge("B", "C", 2)
        G.add_edge("C", "D", 3)
        G.add_edge("D", "E", 4)
        G.add_edge("E", "F", 5)
        G.add_edge("F", "G", 6)
        G.add_edge("G", "H", 7)
        G.add_edge("H", "A", 8)
        for n in ["A", "B", "C", "D", "E", "F", "G", "H"]:
            G.node_attrs[n] = {"node_type": "source"}

        vp.G_network = G
        G_plot = vp._trim_network_for_plot()
        # No source-to-target corridor can be inferred from this degenerate
        # classification, so the ordinary edge-list limit applies.
        assert G_plot.number_of_edges() <= vp.edgeN_limit
        assert not any("source/target reservation" in m for m in messages), messages

    def test_path_based_trim_keeps_complete_paths_and_reports_threshold(self):
        messages = []
        vp = self._make_vp(messages)
        vp.edgeN_limit = 4
        G = FastGraph()
        G.add_edge("S", "A", 1)
        G.add_edge("B", "T", 2)
        G.node_attrs["S"] = {"node_type": "source"}
        G.node_attrs["T"] = {"node_type": "target"}
        G.node_attrs["A"] = {"node_type": "intermediate"}
        G.node_attrs["B"] = {"node_type": "intermediate"}
        G.add_edge("A", "M", 100)
        G.add_edge("M", "B", 90)
        G.add_edge("X", "Y", 80)
        vp.G_network = G
        # path_df: one complete path and one path whose edges are not all in
        # the graph.  The valid path must be selected as a unit.
        vp.path_df = pd.DataFrame(
            {
                "path_block": ["S->A->M->B->T", "S->A->X->Y->B->T"],
                "weights": [[1, 100, 90, 2], [1, 50, 60, 2]],
            }
        )

        G_plot = vp._trim_network_for_plot()
        kept = set(G_plot.edges())
        assert {("S", "A"), ("A", "M"), ("M", "B"), ("B", "T")} <= kept
        assert ("X", "Y") not in kept
        assert any("applied threshold" in m for m in messages), messages

    def test_path_trim_does_not_keep_a_target_tail_without_its_full_path(self):
        messages = []
        vp = self._make_vp(messages)
        vp.edgeN_limit = 3
        vp.path_df = pd.DataFrame(
            {
                "path_block": [
                    "S->Mi1->T",          # weak, independent target tail
                    "S->A->B->T",          # strong interior, weak endpoints
                ],
                "weights": [[100, 1], [1, 100, 2]],
                "path_prob": [0.01, 0.9],
            }
        )
        G = FastGraph()
        for u, v, weight in [
            ("S", "Mi1", 100), ("Mi1", "T", 1),
            ("S", "A", 1), ("A", "B", 100), ("B", "T", 2),
        ]:
            G.add_edge(u, v, weight)
        G.node_attrs["S"] = {"node_type": "source"}
        G.node_attrs["T"] = {"node_type": "target"}
        vp.G_network = G

        selected = vp._select_edges_for_plot()
        kept_edges, _boundary_capped, relaxed, selected_paths, _threshold = selected
        assert relaxed is False
        assert selected_paths == [1]
        assert set(kept_edges) == {("S", "A"), ("A", "B"), ("B", "T")}
        assert ("Mi1", "T") not in kept_edges
        visualized = vp.visualized_paths_for_export()
        assert list(visualized["path_block"]) == ["S->A->B->T"]


# =============================================================================
# save_data: the connMatrix exports can be skipped (FindAllPath keeps the
# canonical type-level matrices in data_details/conn_mat_type_*.csv)
# =============================================================================

class TestSaveDataMatrices:
    """save_data_matrices=False skips the connMatrix exports but still
    writes the connections/original_paths files."""

    def _make_vp(self, tmp_path, save_data_matrices):
        vp = object.__new__(VisualizePath)
        vp.conn_df = pd.DataFrame({
            "source": ["A", "A", "B"],
            "target": ["B", "C", "C"],
            "weight": [3, 5, 7],
            "ratio": [0.5, 0.4, 0.3],
            "probability": [0.9, 0.8, 0.7],
        })
        vp.path_df = pd.DataFrame({"path": ["A->B->C"], "length": [2], "path_prob": [0.72]})
        vp.output_folder = str(tmp_path)
        vp.base_filename = "run"
        vp.output_format = "csv"
        vp.save_data_matrices = save_data_matrices
        vp._vprint = lambda *a, **k: None
        return vp

    def test_skip_matrices_keeps_connections_and_paths(self, tmp_path):
        vp = self._make_vp(tmp_path, save_data_matrices=False)
        files = vp.save_data()
        names = [os.path.basename(f) for f in files]
        assert "run_data_connections.csv" in names
        assert "run_data_original_paths.csv" in names
        assert not any("connMatrix" in n for n in names), names

    def test_default_still_writes_matrices(self, tmp_path):
        vp = self._make_vp(tmp_path, save_data_matrices=True)
        files = vp.save_data()
        names = [os.path.basename(f) for f in files]
        assert "run_data_connMatrix_weight.csv" in names
        assert "run_data_connMatrix_ratio.csv" in names
        assert "run_data_connMatrix_prob.csv" in names


def test_empty_network_does_not_repeat_timestamp_in_folder_name(tmp_path):
    """A timestamped Net-Viz run folder contributes only one file timestamp."""
    run_folder = tmp_path / "plot-network_empty_network_20260814_170906"
    visualizer = VisualizePath(
        path_file=None,
        output_folder=str(run_folder),
        generate_empty_network=True,
        showfig=False,
        verbose=False,
    )

    output_path = Path(visualizer.generate_empty_network_html())

    assert output_path.name == "plot-network_empty_network_20260814_170906_network.html"
    assert re.findall(r"\d{8}_\d{6}", output_path.stem) == ["20260814_170906"]


def test_visualize_network_opens_generated_html_once(tmp_path, monkeypatch):
    """The network convenience method must not open the same HTML twice."""
    import webbrowser

    opened = []
    monkeypatch.setattr(webbrowser, "open", opened.append)

    visualizer = VisualizePath(
        path_file=pd.DataFrame({
            "path_block": ["S>T"],
            "weights": [[3]],
        }),
        output_folder=str(tmp_path),
        showfig=True,
        verbose=False,
    )
    graph = FastGraph()
    graph.add_edge("S", "T", 3)
    graph.node_attrs["S"]["node_type"] = "source"
    graph.node_attrs["T"]["node_type"] = "target"
    visualizer.conn_df = pd.DataFrame({
        "source": ["S"],
        "target": ["T"],
        "weight": [3],
    })
    visualizer.G_network = graph

    output_path = Path(visualizer.visualize_network())

    assert opened == [f"file://{output_path.resolve()}"]


# =============================================================================
# Visualization Edge Limit: the heatmap must consume the SAME complete-path /
# corridor edge set as the network.
# =============================================================================

class TestVisualizationEdgeLimitConsistency:
    """The shared selector gives the heatmap exactly the network edge set."""

    def _make_vp(self, messages=None):
        vp = object.__new__(VisualizePath)
        vp._vprint = lambda msg: messages.append(msg)
        vp.path_df = None
        return vp

    def _graph_and_conn(self):
        G = FastGraph()
        G.add_edge("S", "A", 1)     # weak source boundary
        G.add_edge("A", "T", 2)     # weak target boundary
        G.add_edge("A", "M", 100)   # strong intermediate
        G.add_edge("M", "B", 90)    # strong intermediate
        G.add_edge("B", "C", 80)    # weakest intermediate — cut
        G.node_attrs["S"] = {"node_type": "source"}
        G.node_attrs["T"] = {"node_type": "target"}
        conn_df = pd.DataFrame({
            "source": ["S", "A", "A", "M", "B"],
            "target": ["A", "T", "M", "B", "C"],
            "weight": [1, 2, 100, 90, 80],
        })
        return G, conn_df

    def test_heatmap_filter_uses_same_edge_set_as_network(self):
        messages = []
        vp = self._make_vp(messages)
        vp.edgeN_limit = 2
        vp.G_network, conn_df = self._graph_and_conn()

        filtered = vp._filter_conn_df_for_plot(conn_df)
        kept = set(zip(filtered["source"], filtered["target"]))
        # the weak boundary edges survive as part of the inferred corridor
        assert {("S", "A"), ("A", "T")} <= kept, kept
        # the weakest intermediate edge is cut in the heatmap too
        assert ("B", "C") not in kept, kept
        # identical to the edge set the network draws
        G_plot = vp._trim_network_for_plot()
        assert set(G_plot.edges()) == kept

    def test_selector_keeps_weak_source_target_edges_only_on_corridor(self):
        messages = []
        vp = self._make_vp(messages)
        vp.edgeN_limit = 2
        vp.G_network, _ = self._graph_and_conn()
        selected = vp._select_edges_for_plot()
        assert selected is not None
        kept_edges, boundary_capped, relaxed, selected_paths, threshold = selected
        assert {("S", "A"), ("A", "T")} <= set(kept_edges)
        assert ("B", "C") not in set(kept_edges)
        assert boundary_capped is False
        assert relaxed is False
        assert selected_paths is None          # weight-based fallback branch
        assert threshold == 1                   # weakest edge in the corridor
