"""Mapping grid editor: a table grid for one LabelMapper side.

Rows are custom groups (a name column plus one column per dataset); cells
hold comma-separated neuron identifiers. The grid edits the LabelMapper JSON
schema directly:

    {
        "custom_label": ["grp1", "grp2"],
        "hemibrain:v1.2.1": [["aMe12", "aMe12_R"], ["aMe12_L"]]
    }
"""
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from nicegui import ui

from .. import group_history, mapping_store
from ..config import PROJECT_ROOT
from .custom_grouper import LiteCustomGrouper

_LABEL_KEYS = ("custom_label", "std_label")

NONE_MAPPING = "(No custom mapping)"

# Transient per-run exports of the inline grouper (canonical LabelMapper
# format — directly loadable by scripts); newest per tab wins.
_INLINE_DIR = PROJECT_ROOT / "cache" / "user_mappings" / "_inline"


def inline_mapping_path(tab_key: str) -> str:
    """Timestamped export path for one tab's inline groups."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return str(_INLINE_DIR / f"{tab_key}_{stamp}.json")


# Local documentation served by the app (ui/app.py registers /docs).
# The guide's "UI workflow" section explains the preset workflow; it is an
# HTML page (like the other ui_guides) so it renders in the browser instead
# of downloading as a markdown file.
MAPPING_GUIDE_URL = "docs/ui_guides/label_mapper.html"


def mapping_selector(
    label: str = "Custom Type Mapping",
    hint: str = "Reuse a saved custom type mapping (manage them in the Settings tab).",
    show_instructions: bool = True,
) -> ui.select:
    """Dropdown of saved mapping presets; defaults to the active one.

    Options refresh when the dropdown is opened, so presets created after
    the tab was built still appear. When *show_instructions* is true a small
    link to the LabelMapper guide (custom-grouping instructions) is rendered
    right below the dropdown.
    """
    options = [NONE_MAPPING] + mapping_store.list_mappings()
    active = mapping_store.get_active_mapping() or NONE_MAPPING
    sel = ui.select(
        options=options,
        value=active if active in options else NONE_MAPPING,
        label=label,
    ).classes("w-full drocat-select").tooltip(hint)

    def _refresh_options():
        current = sel.value
        available = [NONE_MAPPING] + mapping_store.list_mappings()
        if current not in available:
            sel.value = NONE_MAPPING
        sel.options = available

    sel.on("click", _refresh_options)

    if show_instructions:
        with ui.row().classes("items-center w-full"):
            ui.link(
                "📖 Custom grouping instructions",
                MAPPING_GUIDE_URL,
                new_tab=True,
            ).classes("text-caption text-primary")
    return sel


def selected_mapping_file_path(selection: Optional[str]) -> Optional[str]:
    """Export path for the chosen preset, or None when no mapping is chosen."""
    if not selection or selection == NONE_MAPPING:
        return None
    return mapping_store.mapping_file_path(selection)


def custom_grouping_block(
    label: str = "Custom Grouping",
    hint: str = (
        "Open the grouping panel: reuse a saved custom type mapping (manage "
        "them in the Settings tab) or define groups inline for this query."
    ),
    datasets_provider: Optional[Callable[[], List[str]]] = None,
    require_names: bool = False,
    tab_key: str = "tab",
    watch_elements: Optional[List] = None,
    query_inputs: Optional[Dict[str, object]] = None,
) -> Tuple[ui.button, ui.dialog, Callable[[], Tuple[Optional[str], bool]]]:
    """Custom Grouping button + popup grouping panel.

    Returns ``(open_button, grouper_dialog, resolve_mapping_path)``:

    - The control is a BUTTON: clicking it opens the panel directly (same
      popup design as the "See available neurons" viewer). The panel hosts
      an optional saved-preset selector plus the inline group board; the
      button label mirrors the current state (none / preset / inline).
    - ``resolve_mapping_path()`` -> ``(path, ok)``. ``ok=False`` means the
      run must be aborted (errors were already notified). A selected preset
      wins; otherwise a non-empty inline board is validated, exported to
      the canonical JSON (script-loadable), recorded in the group history,
      and its path returned. No preset + empty board: ``(None, True)``.
    """
    grouper = LiteCustomGrouper(tab_key=tab_key, require_names=require_names,
                                query_inputs=query_inputs)

    open_button = ui.button(icon="group_work").props(
        "outline no-caps").classes("w-full").style(
        "justify-content: flex-start; min-height: 40px")
    open_button.tooltip(hint)

    dialog = ui.dialog()
    # Persistent: the panel must not close on outside-click / ESC, because the
    # suggestion menus portal to the body (outside the dialog DOM) and clicking
    # them would otherwise dismiss the panel mid-edit. It closes only via X.
    dialog.props("persistent")
    with dialog:
        with ui.card().classes("w-[min(98vw,1400px)] max-w-none"):
            with ui.row().classes("w-full items-center justify-between gap-3"):
                ui.label("Custom Grouping").classes("text-h6")
                ui.button(icon="close", on_click=dialog.close).props(
                    "flat round dense")
            with ui.row().classes("w-full items-center gap-3"):
                # The former "preset" control is now a HISTORY loader: it lists
                # previously-used custom groups (recorded automatically when a
                # group is pushed to a query / exported). Selecting one loads it
                # onto the board; nothing is named or saved manually.
                history_select = ui.select(
                    options=[],
                    value=None,
                    label="Load a saved group (history)",
                ).classes("drocat-select").style("min-width: 320px")
                ui.link(
                    "📖 Custom grouping instructions",
                    MAPPING_GUIDE_URL,
                    new_tab=True,
                ).classes("text-caption text-primary")
            with ui.column().classes("w-full gap-1") as grouper_container:
                pass
    grouper.create(
        grouper_container,
        datasets_provider or (lambda: []),
        watch_elements=watch_elements,
    )
    # Test/DOM hooks.
    dialog.inline_grouper = grouper
    dialog.history_select = history_select

    def _refresh_history() -> None:
        # set_options() (not a bare .options assignment) so the client QSelect
        # actually receives the refreshed history list.
        history_select.set_options(group_history.list_recent(), value=None)

    def _on_history_select(event) -> None:
        lab = event.value
        if not lab:
            return
        rec = group_history.get_label(lab)
        if rec:
            grouper.handle.upsert_row(lab, rec.get("members") or {})
            grouper.resync()
            _update_button_label()
        # Reset so the same entry can be re-selected later.
        history_select.value = None

    history_select.on_value_change(_on_history_select)

    def _group_count() -> int:
        return sum(
            1 for row in grouper.handle.rows
            if any(vals for vals in (row.get("cells") or {}).values()))

    def _update_button_label() -> None:
        n = _group_count()
        open_button.text = (
            f"{label} · inline · {n} group(s)" if n else f"{label} · none")

    def _open_panel() -> None:
        dialog.open()

    def _on_dialog_toggle(event) -> None:
        if event.value:  # opening
            _refresh_history()
            # First-open convenience: seed a single empty row for editing, but
            # only when the board is empty (never on top of existing groups).
            if not grouper.handle.rows:
                grouper.handle.add_row()
            grouper.resync()
        _update_button_label()

    open_button.on_click(_open_panel)
    dialog.on_value_change(_on_dialog_toggle)
    _update_button_label()

    def resolve_mapping_path() -> Tuple[Optional[str], bool]:
        if grouper.is_empty():
            return None, True
        errors = grouper.validate()
        if errors:
            for err in errors:
                ui.notify(err, type="negative")
            return None, False
        path = grouper.export_to(inline_mapping_path(tab_key))
        # History stores user intent at export time (label-keyed,
        # cell-granularity upsert; auto-named groups are skipped).
        group_history.record(grouper.history_payload(), origin="inline")
        return path, True

    return open_button, dialog, resolve_mapping_path



class MappingGridEditor:
    """Editable grid for one mapping side (source / target / intermediate)."""

    def __init__(self, side: str):
        self.side = side
        self._container: Optional[ui.column] = None
        self._datasets: List[str] = []
        self._groups: List[str] = []
        self._cells: Dict[str, Dict[int, str]] = {}  # dataset -> {row: "a, b"}
        self._available: List[str] = []

    # -- data ---------------------------------------------------------------

    def set_data(self, data: Optional[dict]) -> None:
        """Load a LabelMapper side dict into the grid (replaces current)."""
        self._datasets = []
        self._groups = []
        self._cells = {}
        if not data:
            self._rerender()
            return
        labels = data.get("custom_label") or data.get("std_label") or []
        self._groups = [str(g) for g in labels]
        for ds, groups in data.items():
            if ds in _LABEL_KEYS:
                continue
            self._datasets.append(ds)
            self._cells[ds] = {
                i: ", ".join(str(x) for x in group)
                for i, group in enumerate(groups or [])
            }
        for i in range(len(self._groups)):
            for ds in self._datasets:
                self._cells.setdefault(ds, {}).setdefault(i, "")
        self._rerender()

    def get_data(self) -> dict:
        """Build the LabelMapper side dict from the grid state."""
        data = {"custom_label": [g.strip() for g in self._groups]}
        for ds in self._datasets:
            data[ds] = [
                [x.strip() for x in self._cells[ds].get(i, "").split(",") if x.strip()]
                for i in range(len(self._groups))
            ]
        return data

    def is_empty(self) -> bool:
        return not self._groups or all(
            not self._cells.get(ds, {}).get(i, "").strip()
            for i in range(len(self._groups))
            for ds in self._datasets
        )

    # -- structure ----------------------------------------------------------

    def _add_group(self) -> None:
        idx = len(self._groups)
        self._groups.append(f"Group_{idx + 1}")
        for ds in self._datasets:
            self._cells.setdefault(ds, {})[idx] = ""
        self._rerender()

    def _remove_group(self, idx: int) -> None:
        if 0 <= idx < len(self._groups):
            self._groups.pop(idx)
            for ds in self._datasets:
                self._cells[ds].pop(idx, None)
        self._rerender()

    def _add_dataset(self, ds: Optional[str]) -> None:
        if not ds or ds in self._datasets:
            return
        self._datasets.append(ds)
        self._cells[ds] = {i: "" for i in range(len(self._groups))}
        self._available = [d for d in self._available if d != ds]
        self._rerender()

    # -- rendering ----------------------------------------------------------

    def create(self, container, available_datasets: List[str]) -> None:
        """Attach the grid to *container*; *available_datasets* are offered
        as new dataset columns."""
        self._container = container
        self._available = [d for d in available_datasets if d not in self._datasets]
        self._rerender()

    def _rerender(self) -> None:
        if self._container is None:
            return
        self._container.clear()
        with self._container:
            ui.label(
                f"{self.side.replace('_mapping', '').title()} mapping - "
                "one row per custom group; cells hold comma-separated neuron types"
            ).classes("text-caption drocat-muted")
            with ui.row().classes("items-center gap-2 w-full"):
                ui.label("Group Name").classes("text-caption font-bold drocat-muted").style("width: 170px")
                for ds in self._datasets:
                    ui.label(ds).classes("text-caption font-bold drocat-muted").style("width: 210px")
                ui.label("").style("width: 36px")
            for i, name in enumerate(self._groups):
                with ui.row().classes("items-center gap-2 w-full"):
                    name_input = ui.input(value=name).classes("drocat-input").style("width: 170px")
                    name_input.on_value_change(lambda e, idx=i: self._set_group_name(idx, e.value))
                    for ds in self._datasets:
                        cell = ui.input(value=self._cells[ds].get(i, "")).classes("drocat-input").style("width: 210px")
                        cell.on_value_change(lambda e, d=ds, idx=i: self._cells[d].__setitem__(idx, e.value))
                    ui.button(icon="delete", on_click=lambda idx=i: self._remove_group(idx)).props(
                        "flat dense round"
                    ).tooltip("Remove group")
            with ui.row().classes("items-center gap-2"):
                ui.button("Add Group", icon="add", on_click=self._add_group).props("flat dense")
                if self._available:
                    ds_select = ui.select(options=self._available, label="Add dataset column").classes(
                        "drocat-select"
                    ).style("width: 220px")
                    ui.button("Add Column", icon="playlist_add", on_click=lambda: self._add_dataset(ds_select.value)).props(
                        "flat dense"
                    )

    def _set_group_name(self, idx: int, value: str) -> None:
        if 0 <= idx < len(self._groups):
            self._groups[idx] = value or f"Group_{idx + 1}"
