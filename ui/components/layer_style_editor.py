"""Interactive layer-style editor for the Skeleton tab (advanced table).

The editor keeps its rows in a disk-backed draft (see ``ui.layer_style_store``):
every change is auto-saved after a short debounce, so an accidental UI/port
shutdown never loses edits. A draft stays "dirty" (pending export) until the
user explicitly exports the CSV.

Each row is one neuron with an optional per-neuron color and a synapse /
pre-synaptic / post-synaptic color, exactly matching the Skeleton backend's
``layer_map_csv`` contract (``layer`` + ``neuron`` + optional color columns).
The table doubles as the in-page equivalent of uploading such a CSV: rows can
be loaded from an uploaded CSV and exported back to CSV.
"""
from datetime import datetime
import tempfile
from typing import Callable, List, Optional

from nicegui import ui

from .. import layer_style_store
from ..components.common import neuron_list_input
from ..type_suggestions import dataset_suggestions

AUTOSAVE_DELAY = 0.6  # seconds between last edit and disk flush


def _notify(message: str, type: str = "info") -> None:
    """ui.notify that tolerates being called without an active UI slot
    (background tasks, direct calls from unit tests)."""
    try:
        ui.notify(message, type=type)
    except RuntimeError:
        pass


def _columns_for_mode(mode: str) -> list:
    """Table column definitions for a synapse mode.

    'synapse' shows a single ``synapse_color`` column; 'pre-post sites' shows
    the ``pre_synaptic_color`` / ``post_synaptic_color`` columns instead.
    """
    def column(name: str, label: str, *, sortable: bool = False) -> dict:
        return {
            "name": name,
            "label": label,
            "field": name,
            "align": "left",
            "sortable": sortable,
            "classes": f"drocat-{name.replace('_', '-')}-column",
            "headerClasses": f"drocat-{name.replace('_', '-')}-column",
        }

    cols = [
        column("layer", "Layer", sortable=True),
        column("neuron", "Neuron", sortable=True),
        column("color", "Color"),
    ]
    if mode == "pre-post sites":
        cols += [
            column("pre_synaptic_color", "Pre-synaptic color"),
            column("post_synaptic_color", "Post-synaptic color"),
        ]
    else:
        cols += [
            column("synapse_color", "Synapse color"),
        ]
    return cols


_TABLE_HEADER_SLOT = r"""
<q-tr :props="props" class="drocat-edge-header-row">
  <q-th auto-width class="drocat-edge-select-cell">
    <q-checkbox
      v-model="props.selected"
      :indeterminate="props.selected === null"
      dense
    />
  </q-th>
  <q-th
    v-for="col in props.cols"
    :key="col.name"
    :props="props"
    class="drocat-edge-header-cell"
    :class="[col.headerClasses || '', { 'drocat-edge-divider': col.name !== 'color' }]"
  >
    {{ col.label }}
  </q-th>
</q-tr>
"""


# Generic body slot: it iterates over the active columns (``props.cols``) so the
# cells follow the table's column set. The layer cell is a selection box backed
# by ``props.row.layer_opts``; every other colour column renders a picker button
# that opens the single-color picker.
_BODY_SLOT = r"""
<q-tr
  :props="props"
  :class="props.rowIndex % 2 === 0 ? 'drocat-edge-row-even' : 'drocat-edge-row-odd'"
>
  <q-td auto-width class="drocat-edge-select-cell">
    <q-checkbox v-model="props.selected" dense />
  </q-td>
  <q-td v-for="col in props.cols" :key="col.name" :props="props"
    :class="['drocat-edge-cell', col.classes || '', col.name !== 'color' ? 'drocat-edge-divider' : '']">
    <template v-if="col.name === 'layer'">
      <q-select v-model="props.row.layer" :options="props.row.layer_opts"
        dense borderless hide-bottom-space
        @update:model-value="$parent.$emit('layer-cell-change', { id: props.row.id, field: 'layer', value: $event })" />
    </template>
    <template v-else-if="col.name === 'neuron'">
      <q-input v-model="props.row.neuron" dense borderless hide-bottom-space placeholder="Neuron"
        @update:model-value="$parent.$emit('layer-cell-change', { id: props.row.id, field: 'neuron', value: $event })"
        @blur="$parent.$emit('layer-cell-commit', { id: props.row.id, field: 'neuron', value: props.row.neuron })"
        @keydown.enter="$parent.$emit('layer-cell-commit', { id: props.row.id, field: 'neuron', value: props.row.neuron })"
        @keydown.tab="$parent.$emit('layer-cell-commit', { id: props.row.id, field: 'neuron', value: props.row.neuron })" />
    </template>
    <template v-else>
      <div class="row items-center no-wrap gap-1 drocat-color-cell">
        <q-btn
          :icon="props.row[col.name] ? null : 'palette'"
          flat dense round size="xs"
          :class="['drocat-color-cell-picker', { 'drocat-color-cell-picker-set': !!props.row[col.name] }]"
          @click.stop="$parent.$emit('layer-color-pick', { id: props.row.id, field: col.name })"
          title="Pick color"
        >
          <span
            v-if="props.row[col.name]"
            class="drocat-color-cell-preview"
            :style="{ backgroundColor: props.row[col.name] }"
          />
        </q-btn>
        <q-input v-model="props.row[col.name]" dense borderless hide-bottom-space
          placeholder="(auto)"
          @update:model-value="$parent.$emit('layer-cell-change', { id: props.row.id, field: col.name, value: $event })"
          @blur="$parent.$emit('layer-cell-commit', { id: props.row.id, field: col.name, value: props.row[col.name] })"
          @keydown.enter="$parent.$emit('layer-cell-commit', { id: props.row.id, field: col.name, value: props.row[col.name] })"
          @keydown.tab="$parent.$emit('layer-cell-commit', { id: props.row.id, field: col.name, value: props.row[col.name] })" />
      </div>
    </template>
  </q-td>
</q-tr>
"""


class LayerStyleEditorHandle:
    """State + actions of one editor card; usable from tests without JS."""

    def __init__(
        self,
        export_dir_provider: Optional[Callable[[], str]] = None,
        dataset_provider: Optional[Callable[[], object]] = None,
        search_columns_provider: Optional[Callable[[], str]] = None,
    ):
        self.rows: List[dict] = []
        self.current_name: str = ""
        self.export_dir_provider = export_dir_provider
        self._dataset_provider = dataset_provider or (lambda: "")
        self._search_columns_provider = search_columns_provider or (lambda: "auto")
        # NiceGUI elements, assigned while the card is built.
        self.name_input: Optional[ui.input] = None
        self.table: Optional[ui.table] = None
        self.table_container: Optional[ui.element] = None
        self.status_label: Optional[ui.label] = None
        self.validation_panel: Optional[ui.element] = None
        self.validation_label: Optional[ui.label] = None
        self.available_neurons_link: Optional[ui.element] = None
        self.edit_inputs: dict = {}
        self._neuron_input = None
        self._selected_ids: List[int] = []
        self._timer = None
        self.expansion: Optional[ui.expansion] = None
        self._transient_csv_path: Optional[str] = None
        self.synapse_mode: str = "synapse"
        self._color_fields: Optional[list] = None
        self._color_control_groups: dict = {}
        self._add_color_pickers: dict = {}
        self._layer_add_select: Optional[ui.select] = None
        self._pick_popup: Optional[object] = None
        self._pending_pick: Optional[dict] = None
        self._available_batch_layer: Optional[int] = None
        self._available_query_values: List[str] = []
        self._add_input_row: Optional[ui.element] = None

    def _dataset_value(self) -> str:
        """Resolve the current Skeleton dataset for viewer/suggestions."""
        try:
            value = self._dataset_provider()
        except Exception:
            return ""
        if isinstance(value, (list, tuple, set)):
            value = next(iter(value), "")
        return str(value or "").strip()

    def _suggest_neurons(self, text: str):
        """Return the shared dataset-aware suggestion list for the add field."""
        dataset = self._dataset_value()
        if not dataset:
            return []
        try:
            scope = self._search_columns_provider()
        except Exception:
            scope = "auto"
        return dataset_suggestions(
            text,
            dataset,
            str(scope or "auto"),
            limit=None,
        )

    # ------------------------------------------------------------------ rows
    def _row_dicts(self) -> List[dict]:
        # Attach the available layer numbers so each row's layer cell can render
        # a selection box (options computed from the whole table).
        layer_opts = [str(x) for x in layer_style_store.available_layers(self.rows)]
        return [{**row, "id": i, "layer_opts": layer_opts} for i, row in enumerate(self.rows)]

    def refresh_table(self, *, preserve_selection: bool = False) -> None:
        """Refresh the table while keeping Python and QTable selection in sync."""
        row_dicts = self._row_dicts()
        valid_ids = {
            int(row["id"])
            for row in row_dicts
            if isinstance(row.get("id"), int)
        }
        if preserve_selection:
            self._selected_ids = [
                idx for idx in self._selected_ids if idx in valid_ids
            ]
        else:
            self._selected_ids = []

        # Keep the Add-Row layer selection box options in sync with the table.
        if self._layer_add_select is not None:
            self._layer_add_select.options = [
                str(x) for x in layer_style_store.available_layers(self.rows)
            ]
            # ``ui.select.options`` is a mutable ChoiceElement model.  Unlike
            # the table, it does not push an options mutation until update(),
            # which previously left the pending-row dropdown showing its
            # initial one-item list after layer 1 was added.
            self._layer_add_select.update()

        if self.table is not None:
            self.table.rows = row_dicts
            self.table.selected = [
                row for row in row_dicts if row["id"] in self._selected_ids
            ]
            self.table.update()

    def set_rows(self, rows: List[dict], name: Optional[str] = None) -> None:
        self.rows = layer_style_store.normalize_rows(rows)
        self._available_batch_layer = None
        if name is not None:
            self.current_name = name
            if self.name_input is not None:
                self.name_input.value = name
        self.refresh_table()
        self._update_validation()

    def set_synapse_mode(self, mode: str) -> None:
        """Switch to a synapse mode (updates table columns + CSV format)."""
        self.synapse_mode = mode if mode in layer_style_store.MODE_COLUMNS else "synapse"
        if self.table is not None:
            self.table.columns = _columns_for_mode(self.synapse_mode)
            self.table.update()
        self.refresh_table()
        self._sync_add_form_visibility()

    def render(self) -> None:
        """Create the table once; the columns are updated in place on mode change."""
        if self.table_container is None:
            return
        self._color_fields = (
            [("pre_synaptic_color", "Pre-synaptic color"), ("post_synaptic_color", "Post-synaptic color")]
            if self.synapse_mode == "pre-post sites"
            else [("synapse_color", "Synapse color")]
        )
        self.table_container.clear()
        with self.table_container:
            self.table = ui.table(
                columns=_columns_for_mode(self.synapse_mode),
                rows=[],
                row_key="id",
                selection="multiple",
                on_select=self.on_select,
            ).classes("w-full drocat-edge-table").props("dense flat bordered")
            self.table.add_slot("header", _TABLE_HEADER_SLOT)
            self.table.add_slot("body", _BODY_SLOT)
            self.table.on("layer-cell-change", self.on_inline_edit)
            self.table.on("layer-cell-commit", self.on_inline_commit)
            self.table.on("layer-color-pick", self.on_color_pick)
        self.refresh_table()

    def _sync_add_form_visibility(self) -> None:
        """Show each mode-specific colour input together with its picker."""
        visible_fields = {"color"}
        if self.synapse_mode == "pre-post sites":
            visible_fields.update({"pre_synaptic_color", "post_synaptic_color"})
        else:
            visible_fields.add("synapse_color")
        if self._add_input_row is not None:
            self._add_input_row.classes(
                add=(
                    "drocat-layer-add-input-row-pre-post"
                    if self.synapse_mode == "pre-post sites"
                    else "drocat-layer-add-input-row-synapse"
                ),
                remove=(
                    "drocat-layer-add-input-row-synapse"
                    if self.synapse_mode == "pre-post sites"
                    else "drocat-layer-add-input-row-pre-post"
                ),
            )
        for key, element in self.edit_inputs.items():
            if key not in layer_style_store.COLOR_FIELDS:
                continue
            visible = key in visible_fields
            # Hide the complete wrapper so no picker icon remains when its
            # associated input is not part of the active mode.
            group = self._color_control_groups.get(key)
            if group is not None:
                group.set_visibility(visible)
            element.set_visibility(visible)

    def _update_validation(self) -> list:
        """Render the persistent in-page validation panel and return errors."""
        errors = layer_style_store.validate_rows(self.rows)
        if self.validation_panel is None or self.validation_label is None:
            return errors
        if errors:
            self.validation_label.text = "Layer editor errors:\n" + "\n".join(
                f"• {error}" for error in errors
            )
            self.validation_panel.set_visibility(True)
        else:
            self.validation_label.text = ""
            self.validation_panel.set_visibility(False)
        self.validation_label.update()
        return errors

    def load_csv_path(self, path: str) -> bool:
        """Load rows from a CSV file into the table; returns success."""
        try:
            from pathlib import Path
            text = Path(path).read_text(encoding="utf-8")
        except OSError:
            return False
        return self.load_csv_text(text)

    def load_csv_text(self, text: str) -> bool:
        """Load rows from CSV *text* into the table; returns success."""
        try:
            self.set_rows(layer_style_store.load_rows_from_csv_text(text))
        except Exception:
            return False
        if self.status_label is not None:
            self.status_label.text = f"Loaded {len(self.rows)} rows from CSV"
        return True

    # ------------------------------------------------------ available neurons
    def begin_available_batch(self) -> None:
        """Start a fresh viewer-selection batch."""
        self._available_batch_layer = None
        self._available_query_values = []

    def available_query_values(self) -> List[str]:
        """Return the current selection mirrored by the available-neuron viewer."""
        return list(self._available_query_values)

    def apply_available_neurons(self, values) -> int:
        """Append selected viewer entries as one row per entry.

        The viewer reports the complete current selection after every change.
        Existing table values are therefore used as the duplicate guard, while
        the batch layer is allocated once and reused for all new entries in
        the same viewer session.
        """
        cleaned = []
        seen = set()
        for value in values or []:
            text = str(value or "").strip()
            if text and text not in seen:
                cleaned.append(text)
                seen.add(text)
        if not cleaned:
            self.begin_available_batch()
            return 0

        # The viewer sends its complete selection after every toggle. Keep that
        # selection separate from the table rows so the viewer can mirror it in
        # its own boxed preview while the table continues to append only new
        # entries.
        self._available_query_values = list(cleaned)

        if self._update_validation():
            _notify(
                "Fix the layer editor errors before applying available neurons",
                type="warning",
            )
            return 0

        if self._available_batch_layer is None:
            self._available_batch_layer = layer_style_store.next_layer_number(self.rows)

        existing = {
            str(row.get("neuron", "") or "").strip()
            for row in self.rows
            if str(row.get("neuron", "") or "").strip()
        }
        added_ids = []
        for value in cleaned:
            if value in existing:
                continue
            self.rows.append(
                layer_style_store.normalize_rows([{
                    "layer": str(self._available_batch_layer),
                    "neuron": value,
                }])[0]
            )
            existing.add(value)
            added_ids.append(len(self.rows) - 1)

        if not added_ids:
            return 0

        self._selected_ids = added_ids
        self.refresh_table(preserve_selection=True)
        self._sync_edit_inputs()
        self._update_validation()
        self._update_status(
            f"Added {len(added_ids)} neuron{'s' if len(added_ids) != 1 else ''} "
            f"to layer {self._available_batch_layer}"
        )
        self.schedule_autosave()
        return len(added_ids)

    # ------------------------------------------------------------- selection
    def on_select(self, event) -> None:
        self._selected_ids = [row.get("id") for row in getattr(event, "selection", []) or []]
        if self.table is not None:
            row_dicts = self._row_dicts()
            self.table.selected = [
                row for row in row_dicts if row["id"] in self._selected_ids
            ]
        self._sync_edit_inputs()

    def on_inline_edit(self, event) -> None:
        """Persist one value changed in a table cell.

        The body slot keeps the live QInput/QSelect model on the client and
        sends only this small payload to Python. Do not replace the table rows
        or call ``table.update()`` here: doing so would recreate the active
        input and move the cursor after every keystroke.
        """
        args = getattr(event, "args", event)
        if not isinstance(args, dict):
            return
        try:
            row_id = int(args.get("id"))
        except (TypeError, ValueError):
            return
        field = args.get("field")
        if field not in layer_style_store.LAYER_STYLE_COLUMNS:
            return
        if not 0 <= row_id < len(self.rows):
            return

        value = str(args.get("value") or "").strip()
        self.rows[row_id][field] = value
        self._update_validation()
        if field == "layer":
            # Layer options depend on the set of used layers. A layer select
            # has no text cursor to preserve, so refresh its options after the
            # selection changes; neuron/color inputs deliberately skip this
            # refresh to keep their active cursor and selection intact.
            self.refresh_table(preserve_selection=True)
        self.schedule_autosave()

    def on_inline_commit(self, event) -> None:
        """Synchronize the compact Add-Row controls after focus leaves a cell."""
        self.on_inline_edit(event)
        self._sync_edit_inputs()

    def on_color_pick(self, event) -> None:
        """Open the single-color picker popup for a color cell.

        The table body slot emits ``layer-color-pick`` with {id, field} when a
        cell's swatch is clicked; this opens the shared popup seeded with the
        current cell value and applies the committed color back to that row.
        """
        args = getattr(event, "args", event)
        if not isinstance(args, dict):
            return
        try:
            row_id = int(args.get("id"))
        except (TypeError, ValueError):
            return
        field = args.get("field")
        if field not in layer_style_store.LAYER_STYLE_COLUMNS:
            return
        if not 0 <= row_id < len(self.rows):
            return
        self._pending_pick = {"row_id": row_id, "field": field}
        if self._pick_popup is None:
            return
        initial = self.rows[row_id].get(field) or "#145cff"
        self._pick_popup.open(initial)

    def _sync_add_color_picker(self, field: str, value=None) -> None:
        """Show either the palette glyph or the current pending-row color.

        The picker is part of the input's append slot, so updating the button
        in place keeps the form width stable while still making a configured
        color immediately recognizable.  A color is intentionally kept as
        the browser's CSS value here (hex, rgb(a), named colors, etc. are all
        accepted by the backend and by CSS).
        """
        button = self._add_color_pickers.get(field)
        if button is None:
            return
        if value is None:
            element = self.edit_inputs.get(field)
            value = getattr(element, "value", "") if element is not None else ""
        color = str(value or "").strip()
        if color:
            # A square swatch replaces the round palette button once a value
            # exists.  Keep the fixed button dimensions so the input never
            # changes width when a color is selected.
            button.icon = None
            button.props(remove="round")
            safe_color = color.replace(";", "").replace("{", "").replace("}", "")
            button.style(
                replace=(
                    f"background-color:{safe_color}; border-radius:4px;"
                )
            )
            button.classes(add="drocat-layer-add-color-picker-set")
        else:
            button.icon = "palette"
            button.props("round")
            button.style(replace="")
            button.classes(remove="drocat-layer-add-color-picker-set")

    def _on_add_color_change(self, field: str, event=None) -> None:
        """Refresh a pending-row swatch after direct text editing."""
        value = getattr(event, "value", None) if event is not None else None
        if value is None:
            element = self.edit_inputs.get(field)
            value = getattr(element, "value", "") if element is not None else ""
        self._sync_add_color_picker(field, value)

    def _apply_picked_color(self, value: str) -> None:
        """Apply a committed color from the popup to the pending target.

        The pending target is either a table cell (``{"row_id", "field"}``)
        or an Add-Row form input (``{"field"}`` only).
        """
        pending = self._pending_pick
        self._pending_pick = None
        if not pending:
            return
        field = pending.get("field")
        if "row_id" in pending:
            row_id = pending["row_id"]
            if 0 <= row_id < len(self.rows):
                self.rows[row_id][field] = value
                self.refresh_table(preserve_selection=True)
                self.schedule_autosave()
        elif field in self.edit_inputs:
            self.edit_inputs[field].value = value
            self._sync_add_color_picker(field, value)

    def _open_picker_for_add(self, field: str) -> None:
        """Open the popup to pick a color for an Add-Row form input."""
        if field not in self.edit_inputs or self._pick_popup is None:
            return
        self._pending_pick = {"field": field}
        self._pick_popup.open(self.edit_inputs[field].value or "#145cff")

    def _sync_edit_inputs(self) -> None:
        row = self.rows[self._selected_ids[0]] if (
            self._selected_ids and 0 <= self._selected_ids[0] < len(self.rows)
        ) else {col: "" for col in layer_style_store.LAYER_STYLE_COLUMNS}
        for key, element in self.edit_inputs.items():
            value = row.get(key, "")
            if key == "neuron" and self._neuron_input is not None:
                self._neuron_input.chip_input.set_value([value] if value else [])
            else:
                element.value = value
                if key in layer_style_store.COLOR_FIELDS:
                    self._sync_add_color_picker(key, value)

    # --------------------------------------------------------------- editing
    def _current_edit_values(self) -> dict:
        values = {}
        for key, element in self.edit_inputs.items():
            if key == "neuron":
                # Keep the wrapper in ``edit_inputs`` so callers that used the
                # former single-value editor can still assign ``.value`` in
                # tests/integrations. In the live chip editor the value lives
                # on its child QSelect.
                raw = getattr(element, "value", None)
                if not raw and self._neuron_input is not None:
                    raw = self._neuron_input.chip_input.value
                if isinstance(raw, (list, tuple)):
                    raw = raw[0] if raw else ""
            else:
                raw = element.value
            values[key] = str(raw or "").strip()
        return values

    def add_row(self) -> None:
        """Append the row currently entered in the editor controls."""
        values = self._current_edit_values()
        self.rows.append(layer_style_store.normalize_rows([values])[0])
        self._selected_ids = [len(self.rows) - 1]
        self.refresh_table(preserve_selection=True)
        self._sync_edit_inputs()
        self._update_validation()
        self.schedule_autosave()

    def delete_selected(self) -> None:
        if not self._selected_ids:
            _notify("Select rows to delete", type="warning")
            return
        for idx in sorted(set(self._selected_ids), reverse=True):
            if 0 <= idx < len(self.rows):
                del self.rows[idx]
        self._selected_ids = []
        self.refresh_table()
        self._sync_edit_inputs()
        self._update_validation()
        self.schedule_autosave()

    # ------------------------------------------------------------- auto-save
    def schedule_autosave(self) -> None:
        """Debounce edits, then flush to disk. Outside a live NiceGUI slot
        (e.g. unit tests) the flush happens immediately."""
        self._update_status("Editing… (auto-save pending)")
        try:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = ui.timer(AUTOSAVE_DELAY, self.flush_autosave, once=True)
        except Exception:
            self.flush_autosave()

    def flush_autosave(self) -> Optional[str]:
        """Write the current rows to the draft store; returns the CSV path."""
        self._timer = None
        name = str(self.name_input.value or "").strip() if self.name_input else self.current_name
        if not name:
            self._update_status("Enter a draft name to enable auto-save")
            return None
        if self.current_name and name != self.current_name:
            layer_style_store.delete_draft(self.current_name)
        slug = layer_style_store.save_draft(name, self.rows, dirty=True)
        if slug is None:
            self._update_status("Auto-save failed (invalid name or disk error)")
            return None
        self.current_name = name
        self._update_status(f"Auto-saved {datetime.now().strftime('%H:%M:%S')} · pending export")
        return layer_style_store.draft_csv_path(name)

    # ---------------------------------------------------------------- export
    def export_csv(self) -> Optional[str]:
        """Download the current layer style and optionally copy it to the output dir."""
        name = self._draft_name()
        csv_path = self.flush_autosave() if name else None
        csv_text = layer_style_store.rows_to_csv_for_mode(self.rows, self.synapse_mode)
        errors = self._update_validation()
        if errors:
            _notify("CSV downloaded, but has validation errors: " + errors[0], type="warning")

        slug = layer_style_store.sanitize_name(name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{slug}_layers" if slug else "layers"
        filename = f"{filename_base}_{timestamp}.csv"
        export_dir = self.export_dir_provider() if self.export_dir_provider else None
        target = None
        if export_dir:
            from pathlib import Path
            try:
                Path(export_dir).mkdir(parents=True, exist_ok=True)
                target = Path(export_dir) / filename
                target.write_text(csv_text, encoding="utf-8")
            except OSError as ex:
                _notify(f"Could not save a local copy: {ex}", type="warning")

        self._download_csv(csv_text, filename)
        if name and csv_path is not None:
            layer_style_store.mark_exported(self.current_name)
            self._update_status("Downloaded · no unsaved changes")
        else:
            self._update_status(f"Downloaded {filename}")
        _notify("Layer style CSV downloaded", type="positive")
        return str(target) if target else (csv_path or filename)

    # ------------------------------------------------------------- reporting
    def _update_status(self, text: str) -> None:
        if self.status_label is not None:
            self.status_label.text = text

    def _draft_name(self) -> str:
        return (
            str(self.name_input.value or "").strip()
            if self.name_input is not None
            else self.current_name
        )

    @property
    def transient_csv_path(self) -> Optional[str]:
        """Path of the unnamed run file, if one is currently staged."""
        return self._transient_csv_path

    def _download_csv(self, csv_text: str, filename: str) -> None:
        """Trigger a browser download without requiring a filesystem dialog."""
        payload = csv_text.encode("utf-8")
        try:
            if self.table is not None:
                self.table.client.download(payload, filename, "text/csv")
            else:
                ui.download(payload, filename, media_type="text/csv")
        except RuntimeError:
            # Direct handle calls in tests or scripts may not have an active
            # NiceGUI request context; the optional local copy still remains.
            pass

    def _write_transient_csv(self) -> Optional[str]:
        self.cleanup_transient_csv()
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                prefix="drocat_layer_style_",
                suffix=".csv",
                delete=False,
            ) as handle:
                handle.write(layer_style_store.rows_to_csv_for_mode(self.rows, self.synapse_mode))
                self._transient_csv_path = handle.name
        except OSError as ex:
            self._update_status(f"Could not prepare layer list: {ex}")
            return None
        self._update_status("Ready to run · draft name is optional")
        return self._transient_csv_path

    def cleanup_transient_csv(self) -> Optional[str]:
        """Remove the temporary run CSV, returning its former path."""
        path = self._transient_csv_path
        self._transient_csv_path = None
        if path:
            try:
                from pathlib import Path
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
        return path

    def runnable_csv_path(self) -> Optional[str]:
        """Return a VisualizeSkeleton-ready CSV; a draft name is optional for runs."""
        errors = self._update_validation()
        if errors:
            self._update_status("Fix the layer editor errors before running")
            return None
        complete = layer_style_store.complete_rows(self.rows)
        if not complete:
            return None
        return self.flush_autosave() if self._draft_name() else self._write_transient_csv()


def layer_style_editor(
    export_dir_provider: Optional[Callable[[], str]] = None,
    dataset_provider: Optional[Callable[[], object]] = None,
    search_columns_provider: Optional[Callable[[], str]] = None,
    card_id: str = "card-skeleton-layer-style-editor",
    on_expand: Optional[Callable[[], None]] = None,
) -> LayerStyleEditorHandle:
    """Build the collapsed editor panel and return its handle."""
    handle = LayerStyleEditorHandle(
        export_dir_provider,
        dataset_provider=dataset_provider,
        search_columns_provider=search_columns_provider,
    )

    def on_panel_change(event) -> None:
        if getattr(event, "value", False) and on_expand:
            on_expand()

    with ui.expansion(
        "Advanced Layer Editor",
        icon="edit_note",
        value=False,
        on_value_change=on_panel_change,
    ).classes("w-full drocat-edge-editor").props(f'id="{card_id}"') as panel:
        handle.expansion = panel
        ui.label(
            "Edit the Skeleton layers and per-neuron colors directly (the in-page "
            "equivalent of the custom-layer CSV). Each row is one neuron; layer "
            "numbers may start at 0 or 1 but must remain continuous. Every change is auto-saved to disk, so "
            "edits survive an app/port shutdown."
        ).classes("text-caption drocat-muted")

        with ui.row().classes("w-full items-end gap-2 flex-wrap"):
            handle.name_input = ui.input(
                "Draft Name", placeholder="my_custom_layers",
            ).props('outlined dense').classes("grow min-w-[240px]")

        # The table is rebuilt per synapse mode (columns + body slot) inside this
        # container, and the layer cell is a selection box backed by the row's
        # ``layer_opts``.
        handle.table_container = ui.column().classes("w-full")
        handle.render()

        with ui.card().classes("w-full drocat-layer-validation").props(
            f'id="{card_id}-validation"'
        ) as validation_panel:
            with ui.row().classes("items-start gap-2"):
                ui.icon("error", color="negative").classes("mt-1")
                handle.validation_label = ui.label().classes(
                    "text-caption text-negative drocat-layer-validation-label"
                )
        handle.validation_panel = validation_panel
        validation_panel.set_visibility(False)

        # Shared single-color picker popup used by the colour cells' swatches.
        from .color_picker_popup import color_picker_popup

        handle._pick_popup = color_picker_popup(card_id=f"{card_id}-picker")
        handle._pick_popup.on_submit(handle._apply_picked_color)

        from .neuron_index_viewer import create_neuron_index_viewer_link

        handle.available_neurons_link = create_neuron_index_viewer_link(
            handle._dataset_value,
            label="See available neurons",
            on_open=handle.begin_available_batch,
            query_values_getter=handle.available_query_values,
            query_selection=handle.apply_available_neurons,
            query_label="Advanced layer selection",
        )

        color_keys = {"color", "synapse_color", "pre_synaptic_color", "post_synaptic_color"}
        # Keep the editor fields on one consistent baseline.  The action
        # buttons intentionally live in their own row so the mode-specific
        # Pre/Post fields never get pushed below the first input row.
        with ui.column().classes("w-full gap-2 drocat-layer-add-form"):
            with ui.row().classes(
                "w-full items-stretch gap-2 drocat-layer-add-input-row"
            ) as add_input_row:
                handle._add_input_row = add_input_row
                for key, label in (
                    ("layer", "Layer"), ("neuron", "Neuron"), ("color", "Color"),
                    ("synapse_color", "Synapse"), ("pre_synaptic_color", "Pre"),
                    ("post_synaptic_color", "Post"),
                ):
                    if key == "layer":
                        # Layer is a selection box starting at 1; its options are the
                        # currently available layers (refreshed on every table change).
                        layer_select = ui.select(
                            [str(x) for x in layer_style_store.available_layers(handle.rows)],
                            value="1",
                            label="Layer",
                        ).props("outlined dense").classes(
                            "drocat-layer-add-field drocat-layer-add-control "
                            "drocat-layer-add-layer"
                        )
                        handle.edit_inputs[key] = layer_select
                        handle._layer_add_select = layer_select
                    elif key == "neuron":
                        handle._neuron_input = neuron_list_input(
                            label="Neuron",
                            placeholder="Type or select a neuron",
                            hint=(
                                "Type a neuron type, instance, or bodyId. Dataset-aware "
                                "suggestions appear while typing; Enter or Tab commits it."
                            ),
                            unit_label="neuron",
                            show_filter=False,
                            show_upload=False,
                            show_count=False,
                            show_clear=False,
                            max_items=1,
                            suggestions=handle._suggest_neurons,
                        ).classes(
                            "drocat-layer-add-field drocat-layer-add-control "
                            "drocat-layer-add-neuron-input"
                        )
                        handle._neuron_input.chip_input.classes(
                            "drocat-layer-add-field-control"
                        )
                        handle.edit_inputs[key] = handle._neuron_input
                    elif key in color_keys:
                        # Keep the input and picker in one visibility-managed group;
                        # mode switches must never leave an orphan picker icon. The
                        # picker lives in the q-input append slot so it occupies the
                        # field rather than adding another horizontal column.
                        with ui.row().classes(
                            "items-center gap-1 no-wrap drocat-layer-color-control "
                            "drocat-layer-add-field drocat-layer-add-control "
                            "drocat-layer-add-color-control"
                        ) as color_group:
                            color_input = ui.input(label).props(
                                "outlined dense"
                            ).classes("drocat-layer-add-color-input")
                            with color_input.add_slot("append"):
                                picker_button = ui.button(icon="palette").props(
                                    "flat dense round size=xs"
                                ).classes("drocat-layer-add-color-picker").tooltip(
                                    "Pick this colour (alpha + Bokeh palette)"
                                ).on_click(
                                    lambda _e, k=key: handle._open_picker_for_add(k)
                                )
                            handle._add_color_pickers[key] = picker_button
                            color_input.on_value_change(
                                lambda event, k=key: handle._on_add_color_change(k, event)
                            )
                            handle.edit_inputs[key] = color_input
                            handle._color_control_groups[key] = color_group
                    else:
                        handle.edit_inputs[key] = ui.input(label).props(
                            "outlined dense"
                        ).classes("drocat-layer-add-field")

            # Initial visibility of the mode-specific colour inputs.
            handle._sync_add_form_visibility()
            with ui.row().classes(
                "w-full items-center gap-2 drocat-layer-add-actions"
            ):
                ui.button("Add Row", icon="add").props("dense").on_click(handle.add_row)
                ui.button(
                    "Delete Selected", icon="delete"
                ).props("dense outline").on_click(handle.delete_selected)

        with ui.row().classes("w-full items-center gap-3"):
            handle.status_label = ui.label("Empty draft").classes(
                "text-caption drocat-muted grow"
            )
            ui.button("Export CSV", icon="file_download").props("outline dense").on_click(
                handle.export_csv
            )
            with ui.button(icon="upload_file").props("outline dense").classes(
                "drocat-upload-trigger"
            ).tooltip("Load a custom-layer CSV into the table"):
                with ui.menu() as upload_menu:
                    ui.label("Load a custom-layer CSV").classes(
                        "text-caption drocat-muted px-3 pt-2"
                    )
                    ui.label(
                        "Columns: layer, neuron, color, synapse_color, "
                        "pre_synaptic_color, post_synaptic_color"
                    ).classes("text-caption drocat-muted px-3 pb-1")
                    ui.upload(
                        label="Choose CSV",
                        auto_upload=True,
                        on_upload=lambda e: _handle_csv_upload(handle, e),
                    ).props('accept=".csv" flat dense').classes("w-72")
                    upload_menu.update()

    return handle


async def _handle_csv_upload(handle: LayerStyleEditorHandle, event) -> None:
    """Load an uploaded CSV into the editor table."""
    from ..components.common import read_upload_event
    try:
        _filename, data = await read_upload_event(event)
        text = data.decode("utf-8")
    except Exception as ex:
        _notify(f"CSV upload failed: {ex}", type="negative")
        return
    if not handle.load_csv_text(text):
        _notify("Could not parse the uploaded CSV", type="negative")
