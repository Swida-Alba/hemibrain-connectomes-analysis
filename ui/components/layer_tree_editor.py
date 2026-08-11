"""Drag-and-drop layer tree editor for the Skeleton tab.

Mirrors ``VisualizeSkeleton``'s nested-list layer model directly: an ordered
stack of layers where each layer holds one or more neurons (type / instance /
bodyId / regex). The tree starts with three empty layers (the 2nd and 3rd can
stay empty until needed). Layer rows can be dragged to reorder; neuron chips
can be dragged between layers (or reordered within one). A layer's custom
name is edited through a collapsed per-row input — it becomes the layer's
``custom_layer_names`` entry (unnamed layers use ``''`` so the backend
auto-names them).

The handle serializes straight to the backend contract:
``get_neuron_layers()`` -> nested list (single-neuron layers as plain values,
exactly like ``layer_map_csv`` parsing does) and ``get_custom_layer_names()``
-> partial-spec list (``[]`` when nothing is named).
"""
import json
from typing import List, Optional

from nicegui import ui

_DRAG_OVER_JS = (
    "(event) => { event.preventDefault(); event.dataTransfer.dropEffect = 'move'; }"
)
_DRAG_START_JS = (
    "(event) => { event.stopPropagation(); "
    "event.dataTransfer.setData('text/plain', "
    "event.currentTarget.dataset.payload); "
    "event.dataTransfer.effectAllowed = 'move'; }"
)


def _drop_js(selector: str, axis: str = "y") -> str:
    """Drop handler computing the insertion index along *axis* among the
    container's draggable children matching *selector*; emits {payload, to}."""
    coord = "clientX" if axis == "x" else "clientY"
    pos = "left" if axis == "x" else "top"
    size = "width" if axis == "x" else "height"
    return (
        "(event) => { event.preventDefault(); event.stopPropagation(); "
        "let payload = null; "
        "try { payload = JSON.parse(event.dataTransfer.getData('text/plain')); } "
        "catch (e) { return; } "
        "if (!payload) return; "
        "const rect = event.currentTarget.getBoundingClientRect(); "
        f"const {axis} = event.{coord} - rect.{pos}; "
        "let to = 0; "
        f"for (const child of event.currentTarget.querySelectorAll('{selector}')) {{ "
        f"  const c = child.getBoundingClientRect(); "
        f"  if ({axis} > c.{pos} + c.{size} / 2 - rect.{pos}) to++; "
        "} "
        "emit({payload, to}); }"
    )


class LayerTreeHandle:
    """State + actions of the layer tree; usable from tests without JS."""

    def __init__(self):
        self.layers: List[dict] = []  # {"name": str, "neurons": List[str]}
        self.board: Optional[ui.column] = None
        self.status_label: Optional[ui.label] = None

    # ------------------------------------------------------------ structure
    def add_layer(self, neurons: Optional[List[str]] = None, name: str = "") -> None:
        self.layers.append({"name": (name or "").strip(), "neurons": list(neurons or [])})
        self.render()

    def remove_layer(self, index: int) -> None:
        if 0 <= index < len(self.layers):
            del self.layers[index]
            self.render()

    def move_layer(self, index: int, delta: int) -> None:
        target = index + delta
        if 0 <= index < len(self.layers) and 0 <= target < len(self.layers):
            self.layers.insert(target, self.layers.pop(index))
            self.render()

    def move_layer_to(self, from_idx: int, to_idx: int) -> None:
        if not (0 <= from_idx < len(self.layers)):
            return
        to_idx = max(0, min(to_idx, len(self.layers)))
        item = self.layers.pop(from_idx)
        self.layers.insert(to_idx - 1 if from_idx < to_idx else to_idx, item)
        self.render()

    # -------------------------------------------------------------- neurons
    def add_neuron(self, layer: int, text: str) -> bool:
        value = (text or "").strip()
        if not value or not (0 <= layer < len(self.layers)):
            return False
        self.layers[layer]["neurons"].append(value)
        self.render()
        return True

    def remove_neuron(self, layer: int, index: int) -> None:
        if 0 <= layer < len(self.layers) and 0 <= index < len(self.layers[layer]["neurons"]):
            del self.layers[layer]["neurons"][index]
            self.render()

    def move_neuron(self, from_layer: int, from_idx: int, to_layer: int, to_idx: int) -> None:
        if not (0 <= from_layer < len(self.layers)) or not (0 <= to_layer < len(self.layers)):
            return
        source = self.layers[from_layer]["neurons"]
        if not (0 <= from_idx < len(source)):
            return
        item = source.pop(from_idx)
        if from_layer == to_layer and from_idx < to_idx:
            to_idx -= 1
        target = self.layers[to_layer]["neurons"]
        target.insert(max(0, min(to_idx, len(target))), item)
        self.render()

    def set_name(self, layer: int, name: str) -> None:
        if 0 <= layer < len(self.layers):
            self.layers[layer]["name"] = (name or "").strip()
        self._update_status()

    def toggle_name_editor(self, layer: int) -> None:
        """Show/hide the collapsed per-layer name input."""
        if 0 <= layer < len(self.layers):
            self.layers[layer]["name_open"] = not self.layers[layer].get("name_open", False)
            self.render()

    # --------------------------------------------------------- serialization
    def _kept_layers(self) -> List[dict]:
        return [
            layer for layer in self.layers
            if any(str(n).strip() for n in layer["neurons"])
        ]

    def get_neuron_layers(self) -> List:
        """Nested list for the backend: single-neuron layers as plain values
        (same shape the layer_map_csv parser produces)."""
        out = []
        for layer in self._kept_layers():
            neurons = [str(n).strip() for n in layer["neurons"] if str(n).strip()]
            out.append(neurons[0] if len(neurons) == 1 else neurons)
        return out

    def get_custom_layer_names(self) -> List[str]:
        """Partial-spec names ('' keeps auto-naming); [] when nothing named."""
        kept = self._kept_layers()
        if not any(layer["name"] for layer in kept):
            return []
        return [layer["name"] for layer in kept]

    # -------------------------------------------------------------- status
    def _update_status(self) -> None:
        if self.status_label is None:
            return
        kept = self._kept_layers()
        total = sum(len(l["neurons"]) for l in kept)
        self.status_label.text = (
            f"{len(kept)} layers · {total} neurons"
        )

    # ------------------------------------------------------------- rendering
    def render(self) -> None:
        if self.board is None:
            return
        self.board.clear()
        with self.board:
            for i, layer in enumerate(self.layers):
                self._render_layer(i, layer)
        self._update_status()

    def _render_layer(self, i: int, layer: dict) -> None:
        with ui.element("div").classes("drocat-layer-row") as row:
            row._props["draggable"] = "true"
            row._props["data-kind"] = "layer"
            row._props["data-payload"] = json.dumps({"kind": "layer", "index": i})
            row.on("dragstart", None, js_handler=_DRAG_START_JS)

            with ui.row().classes("items-center gap-1 w-full"):
                ui.icon("drag_indicator").classes("drocat-layer-grip")
                ui.badge(str(i + 1), color="grey-6").props("dense")
                # Layers beyond the first are scaffolding: the 2nd/3rd default
                # rows may stay empty, so mark them optional next to the index.
                if i >= 1:
                    ui.label("(optional)").classes("text-caption drocat-muted")
                # Custom names are collapsed by default: a muted caption shows
                # the set name; the edit button opens the input inline.
                ui.label(layer["name"]).classes("text-caption drocat-muted grow")
                ui.button(icon="edit_note").props("flat dense round").tooltip(
                    "Layer name (optional)"
                ).on_click(lambda _e, idx=i: self.toggle_name_editor(idx))
                ui.button(icon="arrow_upward").props("flat dense round").tooltip(
                    "Move layer up"
                ).on_click(lambda _e, idx=i: self.move_layer(idx, -1))
                ui.button(icon="arrow_downward").props("flat dense round").tooltip(
                    "Move layer down"
                ).on_click(lambda _e, idx=i: self.move_layer(idx, 1))
                ui.button(icon="delete_outline").props("flat dense round").tooltip(
                    "Delete layer"
                ).on_click(lambda _e, idx=i: self.remove_layer(idx))

            # The custom name input only exists while the row is in edit mode.
            if layer.get("name_open", False):
                ui.input(
                    placeholder=f"Layer {i + 1} name (optional)",
                    value=layer["name"],
                ).props("outlined dense").classes("w-full").on_value_change(
                    lambda event, idx=i: self.set_name(idx, event.value)
                )

            # Neuron drop target row (chips + add input).
            with ui.element("div").classes("drocat-layer-neurons") as neuron_row:
                neuron_row.on("dragover", None, js_handler=_DRAG_OVER_JS)
                neuron_row.on(
                    "drop", self._make_neuron_drop(i), js_handler=_drop_js(
                        '[data-kind="neuron"]', "x"
                    )
                )
                for j, neuron in enumerate(layer["neurons"]):
                    self._render_chip(i, j, neuron)
                ui.input(placeholder="add neuron ⏎").props(
                    "outlined dense"
                ).classes("drocat-layer-add").on(
                    "keydown.enter", lambda event, idx=i: self._commit_add(event, idx)
                ).on(
                    "blur", lambda event, idx=i: self._commit_add(event, idx)
                )

    def _commit_add(self, event, layer: int) -> None:
        sender = event.sender
        if self.add_neuron(layer, sender.value or ""):
            sender.value = ""

    def _render_chip(self, i: int, j: int, neuron: str) -> None:
        chip = ui.element("div").classes("drocat-layer-chip")
        chip._props["draggable"] = "true"
        chip._props["data-kind"] = "neuron"
        chip._props["data-payload"] = json.dumps(
            {"kind": "neuron", "layer": i, "index": j}
        )
        chip.on("dragstart", None, js_handler=_DRAG_START_JS)
        chip.tooltip("Drag to another layer to move")
        with chip:
            ui.label(neuron)
            ui.button(icon="close").props("flat dense round").classes(
                "drocat-layer-chip-x"
            ).on_click(lambda _e, a=i, b=j: self.remove_neuron(a, b))

    # ------------------------------------------------------------ DnD drops
    def _on_layer_drop(self, event) -> None:
        args = event.args or {}
        payload = args.get("payload") or {}
        if payload.get("kind") != "layer":
            return
        self.move_layer_to(int(payload.get("index", -1)), int(args.get("to", 0)))

    def _make_neuron_drop(self, to_layer: int):
        def handler(event) -> None:
            args = event.args or {}
            payload = args.get("payload") or {}
            if payload.get("kind") != "neuron":
                return
            self.move_neuron(
                int(payload.get("layer", -1)),
                int(payload.get("index", -1)),
                to_layer,
                int(args.get("to", 0)),
            )
        return handler


def layer_tree_editor(card_id: str = "card-skeleton-layers") -> LayerTreeHandle:
    """Build the layer tree card and return its handle.

    The tree starts with three empty layers so the structure is visible
    immediately; the 2nd and 3rd can stay empty until needed.
    """
    handle = LayerTreeHandle()
    # Seed the default rows before the board exists (render is a no-op), then
    # render once at the end so all three layers show up.
    handle.add_layer()
    handle.add_layer()
    handle.add_layer()

    with ui.card().classes("w-full drocat-card").props(f'id="{card_id}"'):
        with ui.row().classes("items-center gap-2"):
            ui.icon("format_list_numbered").classes("text-primary")
            ui.label("Layer Structure").classes("drocat-section-title")
        ui.label(
            "Each row is one layer; drag rows to reorder layers and drag "
            "neuron chips between rows to regroup them. A layer may hold "
            "several neurons. Optional layer names feed the legend; unnamed "
            "layers are auto-named."
        ).classes("text-caption drocat-muted")

        with ui.row().classes("items-end gap-2 w-full"):
            ui.button("Add Layer", icon="add").props("dense").on_click(
                lambda: handle.add_layer()
            )

        handle.board = ui.column().classes("w-full gap-1 drocat-layer-board")
        handle.board.on("dragover", None, js_handler=_DRAG_OVER_JS)
        handle.board.on("drop", handle._on_layer_drop, js_handler=_drop_js(
            '[data-kind="layer"]', "y"
        ))
        handle.status_label = ui.label("0 layers · 0 neurons").classes(
            "text-caption drocat-muted"
        )
    handle.render()
    return handle
