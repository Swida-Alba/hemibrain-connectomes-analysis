"""Mapping grid editor: a table grid for one LabelMapper side.

Rows are custom groups (a name column plus one column per dataset); cells
hold comma-separated neuron identifiers. The grid edits the LabelMapper JSON
schema directly:

    {
        "custom_label": ["grp1", "grp2"],
        "hemibrain:v1.2.1": [["aMe12", "aMe12_R"], ["aMe12_L"]]
    }
"""
from typing import Dict, List, Optional

from nicegui import ui

from .. import mapping_store

_LABEL_KEYS = ("custom_label", "std_label")

NONE_MAPPING = "(No custom mapping)"

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
