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
from typing import Callable, List, Optional

from nicegui import ui

from ..type_suggestions import get_dataset_pools, match_suggestions
from .common import neuron_list_input

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

    def __init__(self, dataset_provider: Optional[Callable[[], object]] = None):
        self.layers: List[dict] = []  # {"name": str, "neurons": List[str]}
        self.board: Optional[ui.column] = None
        self.status_label: Optional[ui.label] = None
        self._dataset_provider = dataset_provider or (lambda: "")
        self._layer_inputs: List[ui.element] = []

    def _dataset_value(self) -> str:
        """Resolve the current Skeleton dataset for suggestions/viewer links."""
        try:
            value = self._dataset_provider()
        except Exception:
            return ""
        if isinstance(value, (list, tuple, set)):
            value = next(iter(value), "")
        return str(value or "").strip()

    def _suggest_neurons(self, text: str):
        dataset = self._dataset_value()
        if not dataset:
            return []
        return match_suggestions(text, get_dataset_pools(dataset), "auto")

    def _sync_layer_input(self, index: int, widget: ui.element) -> None:
        """Keep the nested-list model synchronized with the shared input."""
        if not (0 <= index < len(self.layers)):
            return
        try:
            values = widget.get_value()[1]
        except Exception:
            return
        self.layers[index]["neurons"] = [
            str(value).strip() for value in (values or []) if str(value).strip()
        ]
        self._update_status()

    def _sync_input_state(self) -> None:
        """Copy live chip values before a structural re-render or export."""
        for index, widget in enumerate(self._layer_inputs):
            self._sync_layer_input(index, widget)

    # ------------------------------------------------------------ structure
    def add_layer(self, neurons: Optional[List[str]] = None, name: str = "") -> None:
        self._sync_input_state()
        self.layers.append({"name": (name or "").strip(), "neurons": list(neurons or [])})
        self.render(sync_inputs=False)

    def remove_layer(self, index: int) -> None:
        self._sync_input_state()
        if 0 <= index < len(self.layers):
            del self.layers[index]
            self.render(sync_inputs=False)

    def move_layer(self, index: int, delta: int) -> None:
        self._sync_input_state()
        target = index + delta
        if 0 <= index < len(self.layers) and 0 <= target < len(self.layers):
            self.layers.insert(target, self.layers.pop(index))
            self.render(sync_inputs=False)

    def move_layer_to(self, from_idx: int, to_idx: int) -> None:
        self._sync_input_state()
        if not (0 <= from_idx < len(self.layers)):
            return
        to_idx = max(0, min(to_idx, len(self.layers)))
        item = self.layers.pop(from_idx)
        self.layers.insert(to_idx - 1 if from_idx < to_idx else to_idx, item)
        self.render(sync_inputs=False)

    # -------------------------------------------------------------- neurons
    def add_neuron(self, layer: int, text: str) -> bool:
        self._sync_input_state()
        value = (text or "").strip()
        if not value or not (0 <= layer < len(self.layers)):
            return False
        self.layers[layer]["neurons"].append(value)
        self.render(sync_inputs=False)
        return True

    def remove_neuron(self, layer: int, index: int) -> None:
        self._sync_input_state()
        if 0 <= layer < len(self.layers) and 0 <= index < len(self.layers[layer]["neurons"]):
            del self.layers[layer]["neurons"][index]
            self.render(sync_inputs=False)

    def move_neuron(self, from_layer: int, from_idx: int, to_layer: int, to_idx: int) -> None:
        self._sync_input_state()
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
        self.render(sync_inputs=False)

    def set_name(self, layer: int, name: str) -> None:
        if 0 <= layer < len(self.layers):
            self.layers[layer]["name"] = (name or "").strip()
        self._update_status()

    def toggle_name_editor(self, layer: int) -> None:
        """Show/hide the collapsed per-layer name input."""
        self._sync_input_state()
        if 0 <= layer < len(self.layers):
            self.layers[layer]["name_open"] = not self.layers[layer].get("name_open", False)
            self.render(sync_inputs=False)

    # --------------------------------------------------------- serialization
    def _kept_layers(self) -> List[dict]:
        return [
            layer for layer in self.layers
            if any(str(n).strip() for n in layer["neurons"])
        ]

    def get_neuron_layers(self) -> List:
        """Nested list for the backend: single-neuron layers as plain values
        (same shape the layer_map_csv parser produces)."""
        self._sync_input_state()
        out = []
        for layer in self._kept_layers():
            neurons = [str(n).strip() for n in layer["neurons"] if str(n).strip()]
            out.append(neurons[0] if len(neurons) == 1 else neurons)
        return out

    def get_custom_layer_names(self) -> List[str]:
        """Partial-spec names ('' keeps auto-naming); [] when nothing named."""
        self._sync_input_state()
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
    def render(self, sync_inputs: bool = True) -> None:
        if self.board is None:
            return
        if sync_inputs:
            self._sync_input_state()
        self.board.clear()
        self._layer_inputs = []
        with self.board:
            for i, layer in enumerate(self.layers):
                self._render_layer(i, layer)
        self._update_status()

    def _render_layer(self, i: int, layer: dict) -> None:
        with ui.element("div").classes("w-full drocat-layer-row") as row:
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
                layer_input = neuron_list_input(
                    label="Layer neurons",
                    placeholder="Add neuron (type, instance, or bodyId)",
                    hint=(
                        "Type a neuron name, instance, bodyId, or pattern. "
                        "A blank field opens recent query history; suggestions "
                        "come from the selected dataset."
                    ),
                    unit_label="neuron",
                    show_filter=False,
                    show_upload=False,
                    initial=layer["neurons"],
                    suggestions=self._suggest_neurons,
                    available_neurons=lambda: self._dataset_value(),
                )
                layer_input.classes("w-full drocat-layer-neuron-input")
                layer_input.chip_input.on_value_change(
                    lambda _event, idx=i, widget=layer_input:
                    self._sync_layer_input(idx, widget)
                )
                self._enable_chip_drag(layer_input.chip_input, i)
                self._layer_inputs.append(layer_input)

    @staticmethod
    def _enable_chip_drag(chip_input: ui.element, layer: int) -> None:
        """Make shared QSelect chips participate in the layer drop protocol."""
        chip_input.on(
            "mouseover",
            None,
            js_handler=(
                '(event) => { const chip = event.target.closest?.(".q-chip"); '
                'if (chip) { chip.setAttribute("draggable", "true"); '
                'chip.setAttribute("data-kind", "neuron"); } }'
            ),
        )
        chip_input.on(
            "dragstart",
            None,
            js_handler=(
                f'(event) => {{ const chip = event.target.closest?.(".q-chip"); '
                f'if (!chip || !event.dataTransfer) return; '
                f'event.stopPropagation(); '
                f'const content = chip.querySelector?.(".q-chip__content") || chip; '
                f'const value = (content.textContent || "").trim(); '
                f'if (!value) return; '
                f'const chips = [...event.currentTarget.querySelectorAll(".q-chip")]; '
                f'const index = chips.indexOf(chip); '
                f'event.dataTransfer.setData("text/plain", JSON.stringify({{'
                f'kind: "neuron", layer: {layer}, index}})); '
                f'event.dataTransfer.effectAllowed = "move"; }}'
            ),
        )

    def _commit_add(self, event, layer: int) -> None:
        # add_neuron re-renders the board and REPLACES this input with a fresh
        # empty one — the sender must not be touched afterwards (writing to a
        # deleted element warns and can break the event loop).
        self.add_neuron(layer, event.sender.value or "")

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


def layer_tree_editor(
    card_id: str = "card-skeleton-layers",
    dataset_provider: Optional[Callable[[], object]] = None,
) -> LayerTreeHandle:
    """Build the layer tree card and return its handle.

    The tree starts with three empty layers so the structure is visible
    immediately; the 2nd and 3rd can stay empty until needed.
    """
    handle = LayerTreeHandle(dataset_provider=dataset_provider)
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
