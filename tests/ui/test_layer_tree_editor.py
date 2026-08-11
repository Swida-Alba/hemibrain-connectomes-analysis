"""Tests for the drag-and-drop layer tree editor (Skeleton tab).

Covers the handle's data model (layers of neurons), reordering/moving
semantics used by the drag-and-drop drop handlers, the default three seeded
layers, the collapsed custom-name editing, and the serialization contract
consumed by VisualizeSkeleton (nested neuron_layers + partial
custom_layer_names).
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from nicegui import Client
from nicegui.page import page

from ui.components.layer_tree_editor import LayerTreeHandle, layer_tree_editor


@pytest.fixture()
def handle():
    return LayerTreeHandle()


class TestStructure:
    def test_add_layer_and_neurons(self, handle):
        handle.add_layer()
        assert handle.add_neuron(0, "aMe12") is True
        assert handle.add_neuron(0, "  aMe10  ") is True
        assert handle.add_neuron(0, "   ") is False
        assert handle.layers[0]["neurons"] == ["aMe12", "aMe10"]

    def test_remove_layer(self, handle):
        handle.add_layer(["a"])
        handle.add_layer(["b"])
        handle.remove_layer(0)
        handle.remove_layer(5)  # out of range: no-op
        assert handle.get_neuron_layers() == ["b"]

    def test_move_layer_buttons(self, handle):
        handle.add_layer(["a"])
        handle.add_layer(["b"])
        handle.add_layer(["c"])
        handle.move_layer(0, 1)
        assert handle.get_neuron_layers() == ["b", "a", "c"]
        handle.move_layer(2, 1)  # out of range: no-op
        assert handle.get_neuron_layers() == ["b", "a", "c"]

    def test_move_layer_to_insertion_semantics(self, handle):
        for n in ("a", "b", "c", "d"):
            handle.add_layer([n])
        # Drop row 0 after row 2 (insertion index 3 among 4 rows).
        handle.move_layer_to(0, 3)
        assert handle.get_neuron_layers() == ["b", "c", "a", "d"]
        # Drop row 3 before row 0.
        handle.move_layer_to(3, 0)
        assert handle.get_neuron_layers() == ["d", "b", "c", "a"]


class TestNeurons:
    def test_move_neuron_across_layers(self, handle):
        handle.add_layer(["a", "b"])
        handle.add_layer(["c"])
        handle.move_neuron(0, 1, 1, 1)
        assert handle.layers[0]["neurons"] == ["a"]
        assert handle.layers[1]["neurons"] == ["c", "b"]

    def test_move_neuron_within_layer_forward(self, handle):
        handle.add_layer(["a", "b", "c"])
        handle.move_neuron(0, 0, 0, 2)
        assert handle.layers[0]["neurons"] == ["b", "a", "c"]

    def test_move_neuron_invalid_indices(self, handle):
        handle.add_layer(["a"])
        handle.move_neuron(0, 5, 0, 0)  # bad source: no-op
        handle.move_neuron(3, 0, 0, 0)  # bad layer: no-op
        assert handle.layers[0]["neurons"] == ["a"]

    def test_remove_neuron(self, handle):
        handle.add_layer(["a", "b"])
        handle.remove_neuron(0, 0)
        handle.remove_neuron(0, 7)  # out of range: no-op
        assert handle.layers[0]["neurons"] == ["b"]


class TestSerialization:
    def test_nested_list_shape(self, handle):
        handle.add_layer(["a", "b"])
        handle.add_layer(["c"])
        # Multi-neuron layer stays a list; single-neuron layer a plain value
        # (same shape the backend's layer_map_csv parser produces).
        assert handle.get_neuron_layers() == [["a", "b"], "c"]

    def test_empty_layers_skipped(self, handle):
        handle.add_layer([])
        handle.add_layer(["a"])
        handle.add_layer()
        assert handle.get_neuron_layers() == ["a"]

    def test_custom_names_partial_spec(self, handle):
        handle.add_layer(["a"], name="Input")
        handle.add_layer(["b"])
        # Unnamed layers keep '' so the backend auto-names them.
        assert handle.get_custom_layer_names() == ["Input", ""]

    def test_custom_names_empty_when_unnamed(self, handle):
        handle.add_layer(["a"])
        handle.add_layer(["b"], name="  ")
        assert handle.get_custom_layer_names() == []

    def test_names_align_with_kept_layers(self, handle):
        handle.add_layer([], name="Ghost")     # dropped: no neurons
        handle.add_layer(["a"], name="Real")
        assert handle.get_neuron_layers() == ["a"]
        assert handle.get_custom_layer_names() == ["Real"]


class TestNameEditing:
    """The per-layer custom name input is collapsed by default; an edit
    button on the row opens it. The open state survives re-renders."""

    def _build(self):
        client = Client(page(f"/layer-tree-name-{id(object())}"))
        with client:
            handle = layer_tree_editor()
        return client, handle

    def test_name_input_collapsed_by_default(self):
        client, handle = self._build()
        # Only the per-layer neuron add inputs exist; no name inputs.
        inputs = [el for el in client.elements.values()
                  if type(el).__name__ == "Input"]
        assert len(inputs) == 3
        assert all(not handle.layers[i].get("name_open", False)
                   for i in range(3))

    def test_toggle_opens_and_closes_name_input(self):
        client, handle = self._build()
        handle.toggle_name_editor(0)
        assert handle.layers[0].get("name_open") is True
        inputs = [el for el in client.elements.values()
                  if type(el).__name__ == "Input"]
        assert len(inputs) == 4  # 3 add inputs + the layer-1 name input
        handle.toggle_name_editor(0)
        assert handle.layers[0].get("name_open") is False
        inputs = [el for el in client.elements.values()
                  if type(el).__name__ == "Input"]
        assert len(inputs) == 3

    def test_open_state_survives_rerender(self):
        client, handle = self._build()
        handle.toggle_name_editor(1)
        handle.add_neuron(0, "aMe12")  # triggers a full re-render
        assert handle.layers[1].get("name_open") is True
        inputs = [el for el in client.elements.values()
                  if type(el).__name__ == "Input"]
        assert len(inputs) == 4

    def test_set_name_shows_caption_when_closed(self):
        client, handle = self._build()
        handle.toggle_name_editor(0)
        handle.set_name(0, "Input")
        handle.toggle_name_editor(0)
        texts = [el.text for el in client.elements.values()
                 if getattr(el, "text", "")]
        assert "Input" in texts
        assert handle.get_custom_layer_names() == []  # empty layers skipped


class TestDropHandlers:
    """Simulate the drop events the JS handlers emit ({payload, to})."""

    class _Event:
        def __init__(self, args):
            self.args = args

    def test_layer_drop_reorders(self, handle):
        for n in ("a", "b", "c"):
            handle.add_layer([n])
        handle._on_layer_drop(self._Event(
            {"payload": {"kind": "layer", "index": 0}, "to": 2}
        ))
        assert handle.get_neuron_layers() == ["b", "a", "c"]

    def test_layer_drop_ignores_neuron_payload(self, handle):
        handle.add_layer(["a"])
        handle.add_layer(["b"])
        handle._on_layer_drop(self._Event(
            {"payload": {"kind": "neuron", "layer": 0, "index": 0}, "to": 1}
        ))
        assert handle.get_neuron_layers() == ["a", "b"]

    def test_neuron_drop_moves(self, handle):
        handle.add_layer(["a", "b"])
        handle.add_layer(["c"])
        handler = handle._make_neuron_drop(1)
        handler(self._Event(
            {"payload": {"kind": "neuron", "layer": 0, "index": 0}, "to": 0}
        ))
        assert handle.layers[0]["neurons"] == ["b"]
        assert handle.layers[1]["neurons"] == ["a", "c"]


class TestTabIntegration:
    def test_editor_seeds_three_empty_layers(self):
        client = Client(page(f"/layer-tree-{id(object())}"))
        with client:
            handle = layer_tree_editor()
        assert len(handle.layers) == 3
        # Empty layers are skipped by serialization (run guard still fires).
        assert handle.get_neuron_layers() == []
        # Three rows are rendered on the board.
        rows = [
            el for el in client.elements.values()
            if "drocat-layer-row" in getattr(el, "_classes", [])
        ]
        assert len(rows) == 3
        # Chain support is gone; the Add Layer button remains.
        texts = [el.text for el in client.elements.values()
                 if getattr(el, "text", "")]
        assert "Add Layer" in texts
        assert not any("Chain" in t for t in texts)

    def test_optional_marker_on_layers_beyond_first(self):
        """Layers 2+ are scaffolding: a gray '(optional)' label sits beside
        their index badge; the first layer never gets one."""
        client = Client(page(f"/layer-tree-optional-{id(object())}"))
        with client:
            handle = layer_tree_editor()
        optional = [
            el.text for el in client.elements.values()
            if getattr(el, "text", "") == "(optional)"
        ]
        assert len(optional) == 2  # layers 2 and 3, not layer 1
        handle.add_layer()
        optional = [
            el.text for el in client.elements.values()
            if getattr(el, "text", "") == "(optional)"
        ]
        assert len(optional) == 3

    def test_skeleton_tab_renders_layer_tree(self):
        client = Client(page(f"/layer-tree-{id(object())}"))
        with client:
            layer_tree_editor()
        ids = [
            (getattr(el, "_props", None) or {}).get("id")
            for el in client.elements.values()
        ]
        assert "card-skeleton-layers" in ids

    def test_component_builds_outside_client(self):
        """Handle methods work without a NiceGUI slot (render is a no-op)."""
        handle = LayerTreeHandle()
        handle.add_layer(["x"])  # render() must not raise without a board
        assert handle.get_neuron_layers() == ["x"]

    def test_js_handlers_have_balanced_quotes(self):
        """Regression: a selector quoted with single quotes inside the
        single-quoted drop-handler JS produced a client-side SyntaxError that
        silently unmounted the whole layer card in the live app. Every
        js_handler attached by the editor must keep balanced quoting."""
        client = Client(page(f"/layer-tree-js-{id(object())}"))
        with client:
            handle = layer_tree_editor()
            handle.add_layer(["a"])
        handlers = [
            listener.js_handler
            for el in client.elements.values()
            for listener in getattr(el, "_event_listeners", {}).values()
            if getattr(listener, "js_handler", None)
        ]
        assert handlers, "editor must attach drag/drop js handlers"
        for js in handlers:
            assert js.count("'") % 2 == 0, f"unbalanced quotes in handler: {js[:120]}"
            assert "'[data-kind='" not in js
