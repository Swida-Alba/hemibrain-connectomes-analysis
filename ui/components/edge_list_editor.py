"""Interactive edge-list editor for the Net-Viz tab.

The editor keeps its rows in a disk-backed draft (see ``ui.edge_list_store``):
every change is auto-saved after a short debounce, so an accidental UI/port
shutdown never loses edits. A draft stays "dirty" (pending export) until the
user explicitly exports the CSV.
"""
from datetime import datetime
import tempfile
from typing import Callable, List, Optional

from nicegui import ui

from .. import edge_list_store

AUTOSAVE_DELAY = 0.6  # seconds between last edit and disk flush


def _notify(message: str, type: str = "info") -> None:
    """ui.notify that tolerates being called without an active UI slot
    (background tasks, direct calls from unit tests)."""
    try:
        ui.notify(message, type=type)
    except RuntimeError:
        pass


_TABLE_COLUMNS = [
    {
        "name": "source",
        "label": "Source",
        "field": "source",
        "align": "left",
        "sortable": True,
    },
    {
        "name": "target",
        "label": "Target",
        "field": "target",
        "align": "left",
        "sortable": True,
    },
    {
        "name": "weight",
        "label": "Weight",
        "field": "weight",
        "align": "right",
        "sortable": True,
    },
    {
        "name": "color",
        "label": "Color (optional)",
        "field": "color",
        "align": "left",
    },
]


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
    :class="{ 'drocat-edge-divider': col.name !== 'color' }"
  >
    {{ col.label }}
  </q-th>
</q-tr>
"""


_TABLE_BODY_SLOT = r"""
<q-tr
  :props="props"
  :class="props.rowIndex % 2 === 0 ? 'drocat-edge-row-even' : 'drocat-edge-row-odd'"
>
  <q-td auto-width class="drocat-edge-select-cell">
    <q-checkbox v-model="props.selected" dense />
  </q-td>
  <q-td key="source" :props="props" class="drocat-edge-cell drocat-edge-divider">
    <q-input
      v-model="props.row.source"
      dense
      borderless
      hide-bottom-space
      placeholder="Source"
      @update:model-value="$parent.$emit('edge-cell-change', { id: props.row.id, field: 'source', value: $event })"
    />
  </q-td>
  <q-td key="target" :props="props" class="drocat-edge-cell drocat-edge-divider">
    <q-input
      v-model="props.row.target"
      dense
      borderless
      hide-bottom-space
      placeholder="Target"
      @update:model-value="$parent.$emit('edge-cell-change', { id: props.row.id, field: 'target', value: $event })"
    />
  </q-td>
  <q-td key="weight" :props="props" class="drocat-edge-cell drocat-edge-divider">
    <q-input
      v-model="props.row.weight"
      dense
      borderless
      hide-bottom-space
      inputmode="decimal"
      input-class="text-right"
      placeholder="Weight"
      @update:model-value="$parent.$emit('edge-cell-change', { id: props.row.id, field: 'weight', value: $event })"
    />
  </q-td>
  <q-td key="color" :props="props" class="drocat-edge-cell">
    <q-input
      v-model="props.row.color"
      dense
      borderless
      hide-bottom-space
      placeholder="Optional color"
      @update:model-value="$parent.$emit('edge-cell-change', { id: props.row.id, field: 'color', value: $event })"
    />
  </q-td>
</q-tr>
"""


class EdgeListEditorHandle:
    """State + actions of one editor card; usable from tests without JS."""

    def __init__(self, export_dir_provider: Optional[Callable[[], str]] = None):
        self.rows: List[dict] = []
        self.current_name: str = ""
        self.export_dir_provider = export_dir_provider
        # NiceGUI elements, assigned while the card is built.
        self.name_input: Optional[ui.input] = None
        self.table: Optional[ui.table] = None
        self.status_label: Optional[ui.label] = None
        self.edit_inputs: dict = {}
        self._selected_ids: List[int] = []
        self._timer = None
        self.expansion: Optional[ui.expansion] = None
        self._transient_csv_path: Optional[str] = None

    # ------------------------------------------------------------------ rows
    def _row_dicts(self) -> List[dict]:
        return [{**row, "id": i} for i, row in enumerate(self.rows)]

    def refresh_table(self, *, preserve_selection: bool = False) -> None:
        """Refresh the table while keeping Python and QTable selection in sync.

        QTable clears its visual selection when ``rows`` is replaced.  The
        editor used to leave ``_selected_ids`` untouched, so the edit panel
        could appear deselected while an action still edited a stale row (or
        edited a different row after a delete). Selection is now cleared on
        data loads and explicitly restored only for edits that should keep the
        active row selected.
        """
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

        if self.table is not None:
            self.table.rows = row_dicts
            self.table.selected = [
                row for row in row_dicts if row["id"] in self._selected_ids
            ]
            self.table.update()

    def set_rows(self, rows: List[dict], name: Optional[str] = None) -> None:
        self.rows = edge_list_store.normalize_rows(rows)
        if name is not None:
            self.current_name = name
            if self.name_input is not None:
                self.name_input.value = name
        self.refresh_table()

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

        The QTable body slot updates its local row immediately and emits only
        the small change payload to Python. Keeping the table row in place
        avoids rebuilding the table on every keypress, which would otherwise
        steal focus from the active inline input.
        """
        args = getattr(event, "args", event)
        if not isinstance(args, dict):
            return
        try:
            row_id = int(args.get("id"))
        except (TypeError, ValueError):
            return
        field = args.get("field")
        if field not in edge_list_store.EDGE_COLUMNS:
            return
        if not 0 <= row_id < len(self.rows):
            return

        value = str(args.get("value") or "").strip()
        self.rows[row_id][field] = value
        if self._selected_ids and self._selected_ids[0] == row_id:
            self._sync_edit_inputs()
        self.schedule_autosave()

    def _sync_edit_inputs(self) -> None:
        row = self.rows[self._selected_ids[0]] if (
            self._selected_ids and 0 <= self._selected_ids[0] < len(self.rows)
        ) else {"source": "", "target": "", "weight": "", "color": ""}
        for key, element in self.edit_inputs.items():
            element.value = row.get(key, "")

    # --------------------------------------------------------------- editing
    def _current_edit_values(self) -> dict:
        return {
            key: str(element.value or "").strip()
            for key, element in self.edit_inputs.items()
        }

    def add_edge(self) -> None:
        """Append the edge currently entered in the editor controls."""
        values = self._current_edit_values()
        self.rows.append(edge_list_store.normalize_rows([values])[0])
        self._selected_ids = [len(self.rows) - 1]
        self.refresh_table(preserve_selection=True)
        self._sync_edit_inputs()
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
        # Rename semantics: leaving the current draft name saves under the new
        # name and removes the stale draft file.
        if self.current_name and name != self.current_name:
            edge_list_store.delete_draft(self.current_name)
        slug = edge_list_store.save_draft(name, self.rows, dirty=True)
        if slug is None:
            self._update_status("Auto-save failed (invalid name or disk error)")
            return None
        self.current_name = name
        self._update_status(f"Auto-saved {datetime.now().strftime('%H:%M:%S')} · pending export")
        return edge_list_store.draft_csv_path(name)

    # ---------------------------------------------------------------- export
    def export_csv(self) -> Optional[str]:
        """Download the current edge list and optionally copy it to the output dir."""
        name = self._draft_name()
        csv_path = self.flush_autosave() if name else None
        csv_text = edge_list_store.rows_to_csv(self.rows)
        errors = edge_list_store.validate_rows(self.rows)
        if errors:
            _notify("CSV downloaded, but has validation errors: " + errors[0], type="warning")

        slug = edge_list_store.sanitize_name(name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{slug}_edge_list" if slug else "edge_list"
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
            edge_list_store.mark_exported(self.current_name)
            self._update_status("Downloaded · no unsaved changes")
        else:
            self._update_status(f"Downloaded {filename}")
        _notify("Edge list CSV downloaded", type="positive")
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
                prefix="drocat_edge_list_",
                suffix=".csv",
                delete=False,
            ) as handle:
                handle.write(edge_list_store.rows_to_csv(self.rows))
                self._transient_csv_path = handle.name
        except OSError as ex:
            self._update_status(f"Could not prepare edge list: {ex}")
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

    def runnable_path_file(self) -> Optional[str]:
        """Return a PlotPath-ready CSV; a draft name is optional for runs."""
        complete = edge_list_store.complete_rows(self.rows)
        if not complete:
            return None
        return self.flush_autosave() if self._draft_name() else self._write_transient_csv()


def edge_list_editor(
    export_dir_provider: Optional[Callable[[], str]] = None,
    card_id: str = "card-net-viz-edge-editor",
    on_expand: Optional[Callable[[], None]] = None,
) -> EdgeListEditorHandle:
    """Build the collapsed editor panel and return its handle."""
    handle = EdgeListEditorHandle(export_dir_provider)

    def on_panel_change(event) -> None:
        if getattr(event, "value", False) and on_expand:
            on_expand()

    with ui.expansion(
        "Edge List Editor",
        icon="edit_note",
        value=False,
        on_value_change=on_panel_change,
    ).classes("w-full drocat-edge-editor").props(f'id="{card_id}"') as panel:
        handle.expansion = panel
        ui.label(
            "Build or edit an edge list (source → target with weight). Every "
            "change is auto-saved to disk, so edits survive an app/port "
            "shutdown; the draft stays marked as unsaved until you export it."
        ).classes("text-caption drocat-muted")

        with ui.row().classes("w-full items-end gap-2 flex-wrap"):
            handle.name_input = ui.input(
                "Draft Name", placeholder="my_custom_network",
            ).props('outlined dense').classes("grow min-w-[240px]")

        handle.table = ui.table(
            columns=_TABLE_COLUMNS,
            rows=[],
            row_key="id",
            selection="multiple",
            on_select=handle.on_select,
        ).classes("w-full drocat-edge-table").props("dense flat bordered")
        handle.table.add_slot("header", _TABLE_HEADER_SLOT)
        handle.table.add_slot("body", _TABLE_BODY_SLOT)
        handle.table.on("edge-cell-change", handle.on_inline_edit)

        with ui.row().classes("w-full items-end gap-2 flex-wrap"):
            for key, label in (("source", "Source"), ("target", "Target"),
                               ("weight", "Weight"), ("color", "Color")):
                handle.edit_inputs[key] = ui.input(label).props(
                    "outlined dense"
                ).classes("w-32" if key != "weight" else "w-24")
            ui.button("Add Edge", icon="add").props("dense").on_click(handle.add_edge)
            ui.button("Delete Selected", icon="delete").props("dense outline").on_click(
                handle.delete_selected
            )

        with ui.row().classes("w-full items-center gap-3"):
            handle.status_label = ui.label("Empty draft").classes(
                "text-caption drocat-muted grow"
            )
            ui.button("Export CSV", icon="file_download").props("outline dense").on_click(
                handle.export_csv
            )

    return handle
