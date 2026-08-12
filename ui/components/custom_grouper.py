"""Lite custom grouper: query-local, layer-manager-style group editor.

A light sibling of the Skeleton tab's layer manager: rows are GROUPS
(optional label + one neuron cell per dataset column) instead of path
layers. It plugs into every tab that offers Custom Grouping and exports the
LabelMapper ``overall_mapping_json`` schema — the single canonical format
that the backend reads directly and the UI reloads (format uniformity):

    {
        "format": "drocat_custom_groups/v1",
        "groups_meta": {"updated_at": "...", "origin": "inline"},
        "source_mapping": {
            "custom_label": ["aMe", "clock"],
            "male-cns:v0.9": [["aMe12", "aMe10"], ["DN1p"]],
            "hemibrain:v1.2.1": [["aMe12"], []]
        },
        "target_mapping": { "...same content..." }
    }

Rules baked in here:

- One ``custom_label`` entry governs EVERY dataset column; export always
  materializes every dataset from the provider as a key (empty groups
  ``[]`` included), so ``ComparisonAnalyzer.validate_datasets`` passes on
  cross-dataset runs.
- Source and target roles carry identical content (asymmetric per-role
  grouping stays the Settings preset editor's job); no intermediate role.
- Blank labels auto-name to ``Group_N`` at export (mirrors LabelMapper's
  own ``{role}_grp{i}`` auto-naming); cross-dataset tabs make names
  compulsory via ``require_names``.
- ``LiteGroupHandle`` is JS-free so harness tests can drive it directly.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from nicegui import ui

from .. import group_history, mapping_store
from ..type_suggestions import get_dataset_pools, match_suggestions
from .common import neuron_list_input

FORMAT_TAG = "drocat_custom_groups/v1"
_LABEL_KEYS = ("custom_label", "std_label")
_ROLE_PRECEDENCE = ("source_mapping", "target_mapping", "intermediate_mapping")


# ---------------------------------------------------------------------------
# Serializer (shared by export, reload and history)
# ---------------------------------------------------------------------------

def auto_label(index: int) -> str:
    """The export name of an unnamed group row (1-based)."""
    return f"Group_{index}"


def to_canonical_dict(rows: List[dict], datasets: List[str],
                      origin: str = "inline") -> dict:
    """Build the canonical LabelMapper overall-JSON payload.

    Every dataset in *datasets* becomes a key of every role side with one
    group list per label (``[]`` when the row has no members there), keeping
    the grid rectangular as ``_load_from_json`` requires.
    """
    labels = []
    for i, row in enumerate(rows):
        name = str(row.get("name") or "").strip()
        labels.append(name or auto_label(i + 1))

    def build_side() -> dict:
        side: dict = {"custom_label": labels}
        for ds in datasets:
            side[str(ds)] = [
                [str(x).strip()
                 for x in ((row.get("cells") or {}).get(ds) or [])
                 if str(x).strip()]
                for row in rows
            ]
        return side

    return {
        "format": FORMAT_TAG,
        "groups_meta": {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "origin": origin,
        },
        "source_mapping": build_side(),
        "target_mapping": build_side(),
    }


def from_canonical_dict(data: dict) -> List[dict]:
    """Parse ANY canonical/legacy mapping payload back into board rows.

    Side precedence source -> target -> intermediate; accepts both
    ``custom_label`` and legacy ``std_label``; ignores ``format`` /
    ``groups_meta`` and any unknown keys. Raises ``ValueError`` when no
    usable side exists.
    """
    if not isinstance(data, dict):
        raise ValueError("Mapping file must contain a JSON object")
    for side_key in _ROLE_PRECEDENCE:
        side = data.get(side_key)
        if not isinstance(side, dict):
            continue
        label_key = next((k for k in _LABEL_KEYS if k in side), None)
        if not label_key:
            continue
        labels = [str(x) for x in side.get(label_key) or []]
        ds_keys = [k for k in side if k != label_key]
        rows = []
        for i, lab in enumerate(labels):
            cells: Dict[str, List[str]] = {}
            for ds in ds_keys:
                groups = side.get(ds)
                vals = groups[i] if isinstance(groups, list) and i < len(groups) else []
                if isinstance(vals, (str, int, float)):
                    vals = [vals]
                cells[ds] = [str(v) for v in vals if str(v).strip()]
            rows.append({"name": lab, "cells": cells})
        return rows
    raise ValueError(
        "No mapping side with 'custom_label'/'std_label' found in file")


# ---------------------------------------------------------------------------
# State handle (JS-free, test-driven)
# ---------------------------------------------------------------------------

class LiteGroupHandle:
    """Ordered group rows: ``{"name": str, "cells": {dataset: [neurons]}}``."""

    def __init__(self):
        self.rows: List[dict] = []

    def add_row(self, name: str = "",
                cells: Optional[Dict[str, List[str]]] = None) -> int:
        self.rows.append({
            "name": (name or "").strip(),
            "cells": {str(ds): [str(v) for v in vals]
                      for ds, vals in (cells or {}).items()},
        })
        return len(self.rows) - 1

    def upsert_row(self, name: str,
                   cells: Dict[str, List[str]]) -> int:
        """Replace the row carrying *name* (cover semantics), else append."""
        name = (name or "").strip()
        for i, row in enumerate(self.rows):
            if row["name"] == name:
                self.rows[i] = {
                    "name": name,
                    "cells": {str(ds): [str(v) for v in vals]
                              for ds, vals in (cells or {}).items()},
                }
                return i
        return self.add_row(name, cells)

    def remove_row(self, index: int) -> None:
        if 0 <= index < len(self.rows):
            del self.rows[index]

    def move_row(self, index: int, delta: int) -> None:
        target = index + delta
        if 0 <= index < len(self.rows) and 0 <= target < len(self.rows):
            self.rows.insert(target, self.rows.pop(index))

    def replace_rows(self, rows: List[dict]) -> None:
        self.rows = [
            {"name": str(r.get("name") or "").strip(),
             "cells": {str(ds): [str(v) for v in vals]
                       for ds, vals in (r.get("cells") or {}).items()}}
            for r in rows
        ]

    def is_empty(self) -> bool:
        return not any(
            any(vals for vals in (row.get("cells") or {}).values())
            for row in self.rows
        )


# ---------------------------------------------------------------------------
# UI component
# ---------------------------------------------------------------------------

class LiteCustomGrouper:
    """The inline group board; one instance per tab."""

    def __init__(self, tab_key: str, require_names: bool = False):
        self.tab_key = tab_key
        self.require_names = require_names
        self.handle = LiteGroupHandle()
        self._datasets_provider: Callable[[], List[str]] = lambda: []
        self._container: Optional[ui.column] = None
        self._name_inputs: List[ui.input] = []
        self._cell_widgets: List[Dict[str, ui.element]] = []

    # ------------------------------------------------------------ lifecycle
    def create(self, container: ui.column,
               datasets_provider: Callable[[], List[str]],
               watch_elements: Optional[List] = None) -> "LiteCustomGrouper":
        self._container = container
        self._datasets_provider = datasets_provider
        for el in (watch_elements or []):
            el.on_value_change(lambda _e: self.resync())
        self.resync()
        return self

    def datasets(self) -> List[str]:
        return [str(d) for d in (self._datasets_provider() or []) if d]

    def resync(self) -> None:
        """Re-render the board (dataset columns and cell widgets follow)."""
        if self._container is None:
            return
        self._collect_rows()  # preserve typed state before rebuilding
        self._render()

    # ------------------------------------------------------------- rendering
    def _cell_suggest(self, ds: str):
        def _suggest(text: str):
            return match_suggestions(text, get_dataset_pools(ds), "auto")
        return _suggest

    def _render(self) -> None:
        container = self._container
        container.clear()
        self._name_inputs = []
        self._cell_widgets = []
        datasets = self.datasets()
        with container:
            ui.label(
                "One row per group label; one neuron box per dataset. The "
                "same label governs every dataset column — empty boxes "
                "export as empty groups."
            ).classes("text-caption drocat-muted")
            with ui.row().classes("items-center gap-2 w-full flex-wrap"):
                ui.button("Add Group", icon="add",
                          on_click=self._add_group).props("flat dense color=primary")
                self._history_button()
                self._load_button()
                self._save_preset_button()
            with ui.row().classes("items-center gap-2 w-full flex-wrap"):
                ui.label("Group Name").classes(
                    "text-caption font-bold drocat-muted").style("width: 170px")
                for ds in datasets:
                    ui.label(ds).classes(
                        "text-caption font-bold drocat-muted").style("min-width: 220px")
            for i, row in enumerate(self.handle.rows):
                self._render_row(i, row, datasets)
            if not self.handle.rows:
                ui.label("No groups yet — click 'Add Group' or pull one from "
                         "history.").classes("drocat-empty")

    def _render_row(self, i: int, row: dict, datasets: List[str]) -> None:
        widgets: Dict[str, ui.element] = {}
        with ui.row().classes("items-start gap-2 w-full flex-wrap"):
            name_input = ui.input(
                value=row["name"],
                placeholder=auto_label(i + 1)
                + (" (compulsory)" if self.require_names else " (optional)"),
            ).classes("drocat-input").style("width: 170px")
            self._name_inputs.append(name_input)
            for ds in datasets:
                with ui.element("div").style("min-width: 220px"):
                    widget = neuron_list_input(
                        label=ds,
                        unit_label="member",
                        show_filter=False,
                        show_upload=False,
                        initial=row["cells"].get(ds, []),
                        suggestions=self._cell_suggest(ds),
                    )
                widgets[ds] = widget
            with ui.column().classes("gap-0"):
                ui.button(icon="delete",
                          on_click=lambda _e, idx=i: self._remove_group(idx)
                          ).props("flat dense round").tooltip("Remove group")
                ui.button(icon="arrow_upward",
                          on_click=lambda _e, idx=i: self._move_group(idx, -1)
                          ).props("flat dense round").tooltip("Move up")
                ui.button(icon="arrow_downward",
                          on_click=lambda _e, idx=i: self._move_group(idx, 1)
                          ).props("flat dense round").tooltip("Move down")
        self._cell_widgets.append(widgets)

    # --------------------------------------------------------------- actions
    def _add_group(self) -> None:
        self._collect_rows()
        self.handle.add_row()
        self._render()

    def _remove_group(self, index: int) -> None:
        self._collect_rows()
        self.handle.remove_row(index)
        self._render()

    def _move_group(self, index: int, delta: int) -> None:
        self._collect_rows()
        self.handle.move_row(index, delta)
        self._render()

    def _history_button(self) -> None:
        with ui.button("Add from history", icon="history").props("flat dense"):
            with ui.menu() as menu:
                def _fill():
                    menu.clear()
                    with menu:
                        recent = group_history.list_recent()
                        if not recent:
                            ui.label("No group history yet — labels are "
                                     "recorded after each inline run.").classes(
                                "text-caption drocat-muted px-3 py-2")
                            return
                        for label in recent:
                            rec = group_history.get_label(label) or {}
                            n = sum(len(v) for v in rec.get("members", {}).values())
                            with ui.item(on_click=lambda _e, lab=label:
                                         self._add_from_history(lab)).props("dense"):
                                with ui.item_section():
                                    ui.label(label)
                                with ui.item_section().props("side"):
                                    ui.label(f"{n} members").classes(
                                        "text-caption drocat-muted")
                menu.on("show", _fill)

    def _add_from_history(self, label: str) -> None:
        rec = group_history.get_label(label)
        if not rec:
            return
        self._collect_rows()
        self.handle.upsert_row(label, rec.get("members") or {})
        self._render()
        ui.notify(f"Group '{label}' loaded from history", type="positive")

    def _load_button(self) -> None:
        with ui.button("Load…", icon="upload_file").props("flat dense"):
            with ui.menu():
                ui.label("Load groups from a mapping JSON (canonical or "
                         "legacy preset format)").classes(
                    "text-caption drocat-muted px-3 pt-2")
                ui.upload(label="Choose .json", auto_upload=True,
                          on_upload=self._handle_load).props(
                    'accept=".json" flat dense').classes("w-72 px-3 pb-2")

    async def _handle_load(self, e) -> None:
        try:
            raw = e.content.read()
            data = json.loads(raw.decode("utf-8-sig", errors="replace"))
            rows = from_canonical_dict(data)
            if not rows:
                raise ValueError("file defines no groups")
            self._collect_rows()
            self.handle.replace_rows(rows)
            self._render()
            ui.notify(f"Loaded {len(rows)} group(s)", type="positive")
        except Exception as exc:
            ui.notify(f"Could not load mapping file: {exc}", type="negative")

    def _save_preset_button(self) -> None:
        with ui.button("Save as preset…", icon="bookmark_add").props("flat dense"):
            with ui.menu() as menu:
                ui.label("Save the current groups as a named preset "
                         "(managed in Settings)").classes(
                    "text-caption drocat-muted px-3 pt-2")
                name_input = ui.input(label="Preset name").classes(
                    "w-72 drocat-input px-3")

                def _save():
                    errors = self.validate()
                    if errors:
                        for err in errors:
                            ui.notify(err, type="negative")
                        return
                    name = (name_input.value or "").strip()
                    if not name:
                        ui.notify("Preset name is required", type="negative")
                        return
                    payload = self.to_canonical(origin="preset")
                    mapping_data = {
                        "source_mapping": payload["source_mapping"],
                        "target_mapping": payload["target_mapping"],
                    }
                    verrs = mapping_store.validate_mapping(mapping_data)
                    if verrs:
                        for err in verrs:
                            ui.notify(err, type="negative")
                        return
                    if mapping_store.save_mapping(name, mapping_data):
                        ui.notify(f"Preset '{name}' saved", type="positive")
                        name_input.value = ""
                        menu.close()
                    else:
                        ui.notify("Saving the preset failed", type="negative")

                ui.button("Save", icon="save", on_click=_save).props(
                    "flat dense color=primary").classes("px-3 pb-2")

    # ---------------------------------------------------------------- state
    def _collect_rows(self) -> List[dict]:
        """Sync the handle from live widgets; returns the collected rows."""
        datasets = self.datasets()
        rows = []
        for i, row in enumerate(self.handle.rows):
            name = (self._name_inputs[i].value
                    if i < len(self._name_inputs) else row.get("name")) or ""
            widgets = self._cell_widgets[i] if i < len(self._cell_widgets) else {}
            cells: Dict[str, List[str]] = {}
            for ds in datasets:
                widget = widgets.get(ds)
                if widget is not None:
                    cells[ds] = [str(v) for v in widget.get_value()[1]]
                else:
                    cells[ds] = list((row.get("cells") or {}).get(ds) or [])
            # Keep members of datasets that are not currently rendered
            # (e.g. pulled from history) so they survive column changes.
            for ds, vals in (row.get("cells") or {}).items():
                if ds not in cells:
                    cells[ds] = list(vals)
            rows.append({"name": str(name).strip(), "cells": cells})
        self.handle.replace_rows(rows)
        return rows

    # --------------------------------------------------------------- export
    def validate(self) -> List[str]:
        """Collect state and report blocking errors (empty when valid)."""
        rows = self._collect_rows()
        errors = []
        if not rows or self.handle.is_empty():
            errors.append("Define at least one non-empty custom group")
            return errors
        labels = []
        for i, row in enumerate(rows):
            label = row["name"] or auto_label(i + 1)
            labels.append(label)
            if self.require_names and not row["name"]:
                errors.append(
                    f"Group {i + 1}: the label name is compulsory for "
                    "cross-dataset comparisons")
        dupes = sorted({l for l in labels if labels.count(l) > 1})
        if dupes:
            errors.append("Duplicate group labels: " + ", ".join(dupes))
        return errors

    def to_canonical(self, origin: str = "inline") -> dict:
        rows = self._collect_rows()
        return to_canonical_dict(rows, self.datasets(), origin=origin)

    def export_to(self, path) -> str:
        """Validate, write the canonical JSON and prune stale same-tab files."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Prune earlier inline exports of this tab (latest wins).
        for stale in path.parent.glob(f"{self.tab_key}_*.json"):
            try:
                stale.unlink()
            except OSError:
                pass
        payload = self.to_canonical()
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        return str(path)

    def history_payload(self) -> List[tuple]:
        """(original_label, cells) pairs; blank labels stay blank so the
        history store can skip auto-named groups."""
        return [(row["name"], row["cells"]) for row in self._collect_rows()]

    def is_empty(self) -> bool:
        self._collect_rows()
        return self.handle.is_empty()
