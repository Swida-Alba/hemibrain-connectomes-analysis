"""Interactive edge-list editor card for the Net-Viz tab.

The editor keeps its rows in a disk-backed draft (see ``ui.edge_list_store``):
every change is auto-saved after a short debounce, so an accidental UI/port
shutdown never loses edits. A draft stays "dirty" (pending export) until the
user explicitly exports the CSV; on the next start the Net-Viz tab shows a
recovery reminder for every dirty draft.
"""
from datetime import datetime
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
    {"name": "source", "label": "Source", "field": "source", "align": "left", "sortable": True},
    {"name": "target", "label": "Target", "field": "target", "align": "left", "sortable": True},
    {"name": "weight", "label": "Weight", "field": "weight", "align": "right", "sortable": True},
    {"name": "color", "label": "Color (optional)", "field": "color", "align": "left"},
]


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

    # ------------------------------------------------------------------ rows
    def _row_dicts(self) -> List[dict]:
        return [{**row, "id": i} for i, row in enumerate(self.rows)]

    def refresh_table(self) -> None:
        if self.table is not None:
            self.table.rows = self._row_dicts()
            self.table.selected = []
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
        self._sync_edit_inputs()

    def _sync_edit_inputs(self) -> None:
        row = self.rows[self._selected_ids[0]] if (
            self._selected_ids and 0 <= self._selected_ids[0] < len(self.rows)
        ) else {"source": "", "target": "", "weight": "", "color": ""}
        for key, element in self.edit_inputs.items():
            element.value = row.get(key, "")

    # --------------------------------------------------------------- editing
    def add_edge(self) -> None:
        self.rows.append({"source": "", "target": "", "weight": "", "color": ""})
        self.refresh_table()
        self._selected_ids = [len(self.rows) - 1]
        self._sync_edit_inputs()
        self.schedule_autosave()

    def apply_edit(self) -> None:
        """Write the edit-panel inputs into the first selected row."""
        if not self._selected_ids or not (0 <= self._selected_ids[0] < len(self.rows)):
            _notify("Select a row in the table first", type="warning")
            return
        row = self.rows[self._selected_ids[0]]
        for key, element in self.edit_inputs.items():
            row[key] = str(element.value or "").strip()
        self.refresh_table()
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
        # Rename semantics: leaving the loaded draft's name saves under the
        # new name and removes the stale draft.
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
        """Flush, copy the CSV into the run output dir, clear the dirty flag."""
        csv_path = self.flush_autosave()
        if csv_path is None:
            return None
        errors = edge_list_store.validate_rows(self.rows)
        if errors:
            _notify("Draft saved, but has validation errors: " + errors[0], type="warning")
        export_dir = self.export_dir_provider() if self.export_dir_provider else None
        target = None
        if export_dir:
            from pathlib import Path
            try:
                Path(export_dir).mkdir(parents=True, exist_ok=True)
                target = Path(export_dir) / f"{edge_list_store.sanitize_name(self.current_name)}_edge_list.csv"
                target.write_text(Path(csv_path).read_text(encoding="utf-8"), encoding="utf-8")
            except OSError as ex:
                _notify(f"Export failed: {ex}", type="negative")
                return None
        edge_list_store.mark_exported(self.current_name)
        self._update_status("Exported · no unsaved changes")
        _notify("Edge list exported; draft marked as saved", type="positive")
        return str(target) if target else csv_path

    # ------------------------------------------------------------- reporting
    def _update_status(self, text: str) -> None:
        if self.status_label is not None:
            self.status_label.text = text

    def runnable_path_file(self) -> Optional[str]:
        """Flush and return a PlotPath-ready CSV, or None when not runnable."""
        complete = edge_list_store.complete_rows(self.rows)
        if not complete:
            return None
        return self.flush_autosave()

    def load_draft(self, name: str) -> bool:
        """Load an existing draft into the editor."""
        rows = edge_list_store.load_draft(name)
        if rows is None:
            _notify(f"Draft '{name}' not found", type="negative")
            return False
        self.set_rows(rows, name=name)
        meta = edge_list_store.get_meta(name) or {}
        state = "pending export" if meta.get("dirty") else "exported"
        self._update_status(f"Loaded '{name}' ({len(rows)} rows) · {state}")
        return True

    def delete_current_draft(self) -> bool:
        if not self.current_name:
            return False
        ok = edge_list_store.delete_draft(self.current_name)
        if ok:
            self.set_rows([], name="")
            self._update_status("Draft deleted")
        return ok


def edge_list_editor(
    export_dir_provider: Optional[Callable[[], str]] = None,
    card_id: str = "card-net-viz-edge-editor",
) -> EdgeListEditorHandle:
    """Build the editor card and return its handle."""
    handle = EdgeListEditorHandle(export_dir_provider)

    with ui.card().classes("w-full drocat-card").props(f'id="{card_id}"'):
        with ui.row().classes("items-center gap-2"):
            ui.icon("edit_note").classes("text-primary")
            ui.label("Edge List Editor").classes("drocat-section-title")
        ui.label(
            "Build or edit an edge list (source → target with weight). Every "
            "change is auto-saved to disk, so edits survive an app/port "
            "shutdown; the draft stays marked as unsaved until you export it."
        ).classes("text-caption drocat-muted")

        with ui.row().classes("w-full items-end gap-2"):
            handle.name_input = ui.input(
                "Draft Name", placeholder="my_custom_network",
            ).props('outlined dense').classes("grow")
            with ui.select(
                options=[], label="Load Draft", with_input=True,
            ).props('outlined dense clearable').classes("w-56") as draft_select:
                pass
            ui.button("Load", icon="folder_open").props("outline dense").on_click(
                lambda: draft_select.value and handle.load_draft(draft_select.value)
            )
            ui.button("Delete Draft", icon="delete_outline").props("outline dense").on_click(
                handle.delete_current_draft
            )

        handle.table = ui.table(
            columns=_TABLE_COLUMNS,
            rows=[],
            row_key="id",
            selection="multiple",
            on_select=handle.on_select,
        ).classes("w-full").props("dense flat bordered")

        with ui.row().classes("items-end gap-2"):
            for key, label in (("source", "Source"), ("target", "Target"),
                               ("weight", "Weight"), ("color", "Color")):
                handle.edit_inputs[key] = ui.input(label).props(
                    "outlined dense"
                ).classes("w-32" if key != "weight" else "w-24")
            ui.button("Apply to Selected", icon="check").props("dense").on_click(handle.apply_edit)
            ui.button("Add Edge", icon="add").props("dense").on_click(handle.add_edge)
            ui.button("Delete Selected", icon="remove").props("dense outline").on_click(
                handle.delete_selected
            )

        with ui.row().classes("w-full items-center gap-3"):
            handle.status_label = ui.label("Empty draft").classes(
                "text-caption drocat-muted grow"
            )
            ui.button("Export CSV", icon="file_download").props("outline dense").on_click(
                handle.export_csv
            )

    # Populate the draft selector and keep the handle reachable.
    draft_select.options = [meta["name"] for meta in edge_list_store.list_drafts()]
    draft_select.update()
    handle.draft_select = draft_select
    return handle


def draft_recovery_banner(on_recover: Callable[[str], None]) -> bool:
    """Show a reminder card for dirty (edited, not exported) drafts.

    Returns True when a reminder was rendered. ``on_recover(name)`` is called
    when the user clicks a draft's recovery button.
    """
    pending = edge_list_store.pending_drafts()
    if not pending:
        return False
    with ui.card().classes("w-full drocat-card").props('id="card-edge-draft-recovery"'):
        with ui.row().classes("items-center gap-2"):
            ui.icon("history_toggle").classes("text-warning")
            ui.label("Unsaved edge-list edits recovered").classes("drocat-section-title")
        names = ", ".join(f"'{meta['name']}'" for meta in pending)
        ui.label(
            f"The previous session ended before these drafts were exported: {names}. "
            "They were auto-saved to disk and are fully recoverable."
        ).classes("text-caption drocat-muted")
        with ui.row().classes("gap-2"):
            for meta in pending:
                name = meta["name"]
                ui.button(
                    f"Recover '{name}'", icon="restore"
                ).props("dense outline color=warning").on_click(
                    lambda _event, n=name: on_recover(n)
                )
    return True
