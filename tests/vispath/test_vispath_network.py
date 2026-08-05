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
NODE_CACHE = Path.home() / ".cache" / "vispath-test-node"


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


def _script_text(network_html):
    import re
    html = network_html.read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert scripts, "no inline scripts found in generated HTML"
    return "\n".join(scripts)


def _ensure_node_with_cytoscape():
    """Return the node executable, installing headless Cytoscape into a
    cache dir once. Skips when node/npm or network access is missing."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    cy_path = NODE_CACHE / "node_modules" / "cytoscape"
    if cy_path.exists():
        return node
    npm = shutil.which("npm")
    if not npm:
        pytest.skip("npm not available for installing cytoscape")
    NODE_CACHE.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        ["npm", "install", "cytoscape@3.28.1", "--no-audit", "--no-fund", "--prefix", str(NODE_CACHE)],
        capture_output=True, text=True, timeout=600,
    )
    if res.returncode != 0 or not cy_path.exists():
        pytest.skip(f"could not install cytoscape for Node tests: {res.stderr[-300:]}")
    return node


def _run_node_harness(node, harness_name, network_html):
    harness = HERE / harness_name
    res = subprocess.run(
        [node, str(harness), str(NODE_CACHE), str(network_html)],
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
        # drag relocation is committed to history on dragfree
        assert "pushStateHistory('Move nodes', pendingDragState)" in js
        # edge filter changes are recorded per distinct value
        assert "pushHistory('Edge filter')" in js
        # layout import and label-position toggle are recorded
        assert "pushHistory('Import layout')" in js
        assert "pushHistory('Toggle label position')" in js
        # the three visibility toggles are recorded
        for label in ("Toggle self-loops", "Toggle orphans", "Toggle dead-ends"):
            assert f"pushHistory('{label}')" in js

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


# =============================================================================
# Node-based logic tests (real functions extracted from the generated HTML)
# =============================================================================

class TestDeadEndLogicNode:
    def test_all_dead_end_scenarios(self, network_html):
        node = _ensure_node_with_cytoscape()
        res = _run_node_harness(node, "deadend_harness.js", network_html)
        assert res.returncode == 0, (
            f"dead-end harness failed:\n{res.stdout}\n{res.stderr}"
        )
        assert "ALL DEAD-END TESTS PASSED" in res.stdout


class TestHistoryLogicNode:
    def test_all_history_scenarios(self, network_html):
        node = _ensure_node_with_cytoscape()
        res = _run_node_harness(node, "history_harness.js", network_html)
        assert res.returncode == 0, (
            f"history harness failed:\n{res.stdout}\n{res.stderr}"
        )
        assert "ALL HISTORY TESTS PASSED" in res.stdout
