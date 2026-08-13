"""NiceGUI viewer for a locally cached neuron index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List

from nicegui import ui

from ..config import PROJECT_ROOT
from ..neuron_index import (
    load_cached_neuron_index,
    neuron_index_path,
    query_neuron_index,
)


# A one-character query can legitimately produce thousands of deduplicated
# names. Sending every match row to Quasar at once overwhelms the websocket
# even though the underlying server-side query is bounded and responsive.
# Keep the full membership maps for exact selection, but render a fixed page
# of match details; the panel pager still exposes every matched name.
MATCH_GROUP_PAGE_SIZE = 50


def _dataset_values(value) -> List[str]:
    """Normalize a single- or multi-dataset getter result."""
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]
    result = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _column_label(column: str) -> str:
    if column == "bodyId":
        return "Body ID"
    return column.replace("_", " ").strip().title()


def _relative_source(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _query_preview_values(getter: Callable[[], object] | None) -> List[str]:
    """Read the current query values for the viewer's compact preview."""
    if getter is None:
        return []
    try:
        value = getter()
    except Exception:
        return []
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]
    result = []
    seen = set()
    for item in values:
        text = str(item or "").strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _render_missing_cache(content, dataset: str, path: Path) -> None:
    with content:
        ui.icon("database", color="orange").classes("text-4xl")
        ui.label("This dataset is not cached locally yet.").classes("text-subtitle1 font-bold")
        ui.label(
            f"The cached neuron index was not found at {_relative_source(path)}. "
            "The viewer does not open or stream the original dataset file."
        ).classes("text-body2")
        ui.label(
            "To make the index available, either run the selected analysis once "
            "(where the workflow supports first-run cache creation), or pull the "
            "full dataset from the Settings tab."
        ).classes("text-body2")
        ui.label("Recommended UI flow").classes("font-bold text-primary mt-2")
        ui.label(
            "Open Settings → Dataset Cache, choose "
            f"{dataset}, then click Pull Full Dataset. The pull is resumable."
        ).classes("text-body2")
        ui.link(
            "Open dataset-cache instructions",
            "docs/ui_guides/settings.html",
            new_tab=True,
        ).classes("drocat-doc-link")
        ui.label("Command-line alternative (from the project root)").classes(
            "font-bold text-primary mt-2"
        )
        ui.code(
            f"python src/build_connection_cache.py {dataset}",
            language="bash",
        ).classes("w-full")
        ui.label(
            "NeuPrint datasets need a configured token. FlyWire datasets must be "
            "prepared locally first; follow the matching preparation guide in Settings."
        ).classes("text-caption drocat-muted")


def _render_index(
    content,
    dataset: str,
    *,
    query_values_getter: Callable[[], object] | None = None,
    query_selection: Callable[[List[str]], object] | None = None,
    query_resolution: Callable[[List[str]], object] | None = None,
    query_remove: Callable[[str], object] | None = None,
    add_to_query: Callable[[List[str]], object] | None = None,
    query_label: str = "Current query",
) -> None:
    """Render the current dataset's index or its cache-missing state."""
    content.clear()
    path = neuron_index_path(dataset)
    try:
        index = load_cached_neuron_index(dataset)
    except FileNotFoundError:
        _render_missing_cache(content, dataset, path)
        return
    except Exception as exc:
        with content:
            ui.icon("error", color="red").classes("text-4xl")
            ui.label("The cached neuron index could not be opened.").classes(
                "text-subtitle1 font-bold"
            )
            ui.label(str(exc)).classes("text-body2 drocat-err")
            ui.label(
                "Use Settings → Dataset Cache → Force rebuild, then open this viewer again."
            ).classes("text-caption drocat-muted")
        return

    columns = list(index.columns)
    if not columns:
        with content:
            ui.label("The cached neuron index is empty.").classes("text-body2 drocat-warn")
        return

    with content:
        with ui.row().classes("items-center gap-2 flex-wrap"):
            ui.badge(f"{index.frame.height:,} indexed rows", color="primary").props("outline")
            ui.label(f"Source: {_relative_source(index.path)}").classes(
                "text-caption drocat-muted"
            )
            if index.enriched:
                ui.label(
                    "Blank type/instance values are filled from local prepared metadata."
                ).classes("text-caption drocat-muted")

        if query_values_getter is not None:
            with ui.element("section").classes("drocat-neuron-query-preview"):
                with ui.row().classes("w-full items-center justify-between gap-2"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("playlist_add_check", color="primary").classes("text-lg")
                        ui.label(f"Current query · {query_label}").classes(
                            "text-subtitle2 font-bold"
                        )
                    ui.badge("mirrors input", color="primary").props("outline")
                query_preview = ui.row().classes(
                    "w-full items-center gap-1 flex-wrap mt-1"
                )
                query_preview_empty = ui.label(
                    "No values in the query yet. Select a match or body row to add one."
                ).classes("text-caption drocat-muted mt-1")

                def refresh_query_preview() -> None:
                    values = _query_preview_values(query_values_getter)
                    query_preview.clear()
                    query_preview_empty.set_visibility(not values)
                    with query_preview:
                        for value in values:
                            with ui.element("div").classes(
                                "drocat-neuron-query-chip-wrap"
                            ):
                                # Keep the value label's historic class so
                                # the preview remains easy to inspect and
                                # compatible with existing UI tests.
                                ui.label(value).classes("drocat-neuron-query-chip")
                                if query_remove is not None:
                                    ui.button(
                                        icon="close",
                                        on_click=lambda v=value: remove_query_value(v),
                                    ).props("flat round dense").classes(
                                        "drocat-neuron-query-chip-remove"
                                    ).tooltip("Remove from this query")

                refresh_query_preview()
        ui.label(
            "Search returns all matches across bodyId, type, instance, and useful "
            "type/taxonomy fields: strict case-sensitive prefixes come first, "
            "followed by case-insensitive substring matches. Choose a target "
            "column and match mode to apply that rule directly to this search "
            "box; leave it unset for the global search. Numeric input is verified "
            "against bodyId. Match details also keeps a secondary matched name "
            "when the same row matches in another field. Select a matched name "
            "to select every body sharing it, or select individual body rows to "
            "add their body IDs."
        ).classes("text-caption drocat-muted")

        with ui.row().classes(
            "w-full items-end gap-2 flex-wrap drocat-neuron-search-toolbar"
        ):
            search_input = ui.input(
                "Search identities & taxonomy",
                placeholder="e.g. aMe12 or 5813",
            ).props("outlined clearable input-debounce=180").classes(
                "flex-grow drocat-input drocat-neuron-search-field"
            )
            filter_options = {"__none__": "No column filter"}
            filter_options.update({column: _column_label(column) for column in columns})
            target_column = ui.select(
                options=filter_options,
                value="__none__",
                label="Target column",
            ).props("outlined").classes(
                "drocat-select drocat-neuron-search-field"
            ).style("min-width: 150px")
            filter_operator = ui.select(
                options={
                    "contains": "Contains",
                    "prefix": "Starts with",
                    "suffix": "Ends with",
                    "exact": "Exact",
                    "regex": "Regex",
                },
                value="contains",
                label="Match mode",
            ).props("outlined").classes(
                "drocat-select drocat-neuron-search-field"
            ).style("min-width: 140px")
            sort_options = {"__match_value__": "Matched value (default)"}
            sort_options.update({column: _column_label(column) for column in columns})
            sort_column = ui.select(
                options=sort_options,
                value="__match_value__",
                label="Sort by",
            ).props("outlined").classes(
                "drocat-select drocat-neuron-search-field"
            ).style("min-width: 150px")
            direction = ui.select(
                options={"asc": "Ascending", "desc": "Descending"},
                value="asc",
                label="Order",
            ).props("outlined").classes(
                "drocat-select drocat-neuron-search-field"
            ).style("min-width: 140px")
            page_size = ui.select(
                options={25: "25 / page", 50: "50 / page", 100: "100 / page", 200: "200 / page"},
                value=50,
                label="Rows",
            ).props("outlined").classes(
                "drocat-select drocat-neuron-search-field"
            ).style("min-width: 120px")

        filter_operator.set_enabled(False)

        initial = query_neuron_index(index, page_size=50)
        match_columns = [
            {
                "name": "match_column",
                "label": "Matched by",
                "field": "match_column",
                "align": "left",
                "classes": "drocat-neuron-match-by",
                "headerClasses": "drocat-neuron-match-by",
                "style": "width: 135px; min-width: 135px",
                "headerStyle": "width: 135px; min-width: 135px",
                "sortable": False,
            },
            {
                "name": "match_value",
                "label": "Matched value",
                "field": "match_value",
                "align": "left",
                "classes": "drocat-neuron-match-value",
                "headerClasses": "drocat-neuron-match-value",
                "style": "width: 190px; min-width: 190px",
                "headerStyle": "width: 190px; min-width: 190px",
                "sortable": False,
            },
            {
                "name": "body_count",
                "label": "Rows",
                "field": "body_count",
                "align": "right",
                "classes": "drocat-neuron-match-count",
                "headerClasses": "drocat-neuron-match-count",
                "style": "width: 64px; min-width: 64px",
                "headerStyle": "width: 64px; min-width: 64px",
                "sortable": False,
            },
        ]
        table_columns = [
            *[
                {
                    "name": column,
                    "label": _column_label(column),
                    "field": column,
                    # Full-index sorting is controlled above; enabling Quasar's
                    # client-side header sort here would sort only the current page.
                    "sortable": False,
                }
                for column in columns
            ],
        ]
        # The match panel is intentionally compact: one row per deduplicated
        # matched value. It is aligned with the metadata panel at the top, but
        # does not reserve blank rows for every repeated bodyId.
        selected_match_values: set[str] = set()
        selected_match_order: List[str] = []
        selected_match_members: dict[str, set[str]] = {}
        selected_match_body_ids: dict[str, tuple[str, ...]] = {}
        selected_body_ids: dict[str, str] = {}
        current_rows = list(initial.rows)
        match_groups_all = list(initial.match_groups)
        current_groups = match_groups_all[:MATCH_GROUP_PAGE_SIZE]
        match_state = {"page": 1}
        current_group_body_ids = {
            str(key): tuple(values)
            for key, values in initial.match_group_body_ids.items()
        }
        match_group_related = {
            str(key): tuple(values)
            for key, values in initial.match_group_related.items()
        }
        match_group_primary = {
            str(key): tuple(values)
            for key, values in initial.match_group_primary.items()
        }
        group_members = {
            str(key): set(values)
            for key, values in initial.match_group_members.items()
        }
        match_table = None
        table = None
        selection_status = None
        match_status = None
        match_page_position = None
        match_previous_button = None
        match_next_button = None
        query_callback = query_selection or add_to_query

        def effective_body_keys() -> set[str]:
            keys = set(selected_body_ids)
            for value in selected_match_values:
                keys.update(selected_match_members.get(value, set()))
                keys.update(group_members.get(value, set()))
            return keys

        def related_match_values(value: str) -> List[str]:
            """Return one direct primary/secondary selection bundle.

            The backend intentionally does not return transitive connected
            components here.  Two independent primary names may share a
            taxonomy spelling on different rows; walking a graph would make
            selecting one primary unexpectedly select the other one too.
            """
            value = str(value or "").strip()
            if not value:
                return []
            return [
                linked
                for linked in match_group_related.get(value, (value,))
                if str(linked or "").strip()
            ]

        def remember_match(value: str, *, expand: bool = True) -> None:
            """Persist a selected name and its verified membership.

            Match rows are replaced whenever the search changes, so the
            selection cannot depend on the current page or current search
            result. The exact body-ID snapshot is retained for execution even
            when the selected name is no longer visible in the new search.
            """
            value = str(value or "").strip()
            if not value:
                return
            linked_values = related_match_values(value) if expand else [value]
            for linked_value in linked_values:
                if linked_value not in selected_match_values:
                    selected_match_values.add(linked_value)
                selected_match_members[linked_value] = set(
                    group_members.get(
                        linked_value,
                        selected_match_members.get(linked_value, set()),
                    )
                )
                selected_match_body_ids[linked_value] = tuple(
                    current_group_body_ids.get(
                        linked_value,
                        selected_match_body_ids.get(linked_value, ()),
                    )
                )
                for primary_value in match_group_primary.get(
                    linked_value, (linked_value,)
                ):
                    if (
                        primary_value
                        and primary_value not in selected_match_order
                    ):
                        selected_match_order.append(primary_value)

        def forget_match(value: str) -> None:
            value = str(value or "").strip()
            linked_values = set(related_match_values(value))
            if not linked_values:
                linked_values = {value}
            for linked_value in linked_values:
                selected_match_values.discard(linked_value)
                selected_match_members.pop(linked_value, None)
                selected_match_body_ids.pop(linked_value, None)
            selected_match_order[:] = [
                value for value in selected_match_order
                if value not in linked_values
            ]

        def row_body_id(row) -> str:
            """Return a verified, query-safe body ID for an individual row."""
            value = str(row.get("bodyId", "") or "").strip()
            if not value:
                return ""
            integer, dot, fraction = value.partition(".")
            if dot and integer.isdigit() and fraction and set(fraction) == {"0"}:
                return integer
            return value

        def selected_query_values() -> List[str]:
            values: List[str] = []
            seen: set[str] = set()
            # Keep the user's selection order even when a later search no
            # longer displays an earlier selected match group.
            for value in selected_match_order:
                if value in selected_match_values and value not in seen:
                    values.append(value)
                    seen.add(value)
            for row in current_rows:
                key = str(row.get("__neuron_key", "") or "")
                value = str(selected_body_ids.get(key, "") or "").strip()
                if value and value not in seen:
                    values.append(value)
                    seen.add(value)
            for value in selected_body_ids.values():
                value = str(value or "").strip()
                if value and value not in seen:
                    values.append(value)
                    seen.add(value)
            return values

        def selected_query_body_ids() -> List[str]:
            """Resolve selections to exact body IDs for query execution.

            The compact match panel displays human-readable names, but a name
            can occur in multiple metadata columns. Passing the resolved IDs
            separately prevents the eventual query from re-running a priority
            string lookup and selecting a different column by accident.
            """
            values: List[str] = []
            seen: set[str] = set()
            for value in selected_match_order:
                if value not in selected_match_values:
                    continue
                for body_id in selected_match_body_ids.get(value, ()):
                    body_id = str(body_id or "").strip()
                    if body_id and body_id not in seen:
                        values.append(body_id)
                        seen.add(body_id)
            for row in current_rows:
                key = str(row.get("__neuron_key", "") or "")
                if key not in selected_body_ids:
                    continue
                body_id = row_body_id(row)
                if body_id and body_id not in seen:
                    values.append(body_id)
                    seen.add(body_id)
            for body_id in selected_body_ids.values():
                body_id = str(body_id or "").strip()
                if body_id and body_id not in seen:
                    values.append(body_id)
                    seen.add(body_id)
            return values

        def sync_query_selection() -> None:
            if query_callback is not None:
                query_callback(selected_query_values())
            if query_resolution is not None:
                query_resolution(selected_query_body_ids())
            if query_values_getter is not None:
                refresh_query_preview()

        def update_selection_status() -> None:
            if selection_status is not None:
                selected_count = len(selected_match_values) + len(selected_body_ids)
                selection_status.text = f"{selected_count} selected"
                selection_status.update()

        def refresh_table_selection() -> None:
            if match_table is not None:
                match_table.selected = [
                    row for row in current_groups
                    if row.get("match_role") != "secondary"
                    and str(row.get("match_value", "") or "") in selected_match_values
                ]
                match_table.update()
            if table is not None:
                active = effective_body_keys()
                table.selected = [
                    row for row in current_rows
                    if str(row.get("__neuron_key", "") or "") in active
                ]
                table.update()
            update_selection_status()

        def handle_match_selection(event) -> None:
            visible_rows = [
                row for row in current_groups
                if row.get("match_role") != "secondary"
            ]
            visible_values = {
                str(row.get("match_value", "") or "").strip()
                for row in visible_rows
            }
            selected_rows = [
                row for row in list(getattr(event, "selection", []) or [])
                if row.get("match_role") != "secondary"
            ]
            raw_selected_values = {
                str(row.get("match_value", "") or "").strip()
                for row in selected_rows
            }
            selected_values: set[str] = set()
            for value in raw_selected_values:
                selected_values.update(related_match_values(value))
            # The table emits only the current result rows. Update those rows
            # while preserving selections made in an earlier search.
            for value in visible_values - selected_values:
                forget_match(value)
            for row in selected_rows:
                remember_match(str(row.get("match_value", "") or "").strip())
            # A matched-name selection includes every body in that group,
            # including rows on later pages. The full table mirrors the
            # current page of that selection when it is visible.
            sync_query_selection()
            # Selection and clicking a matched value use the same focus path:
            # resolve the first exact member, compute its sorted data page,
            # then scroll the actual metadata row into view.
            focus_key = ""
            for row in selected_rows:
                value = str(row.get("match_value", "") or "").strip()
                members = tuple(group_members.get(value, ()))
                if members:
                    focus_key = members[0]
                    break
            if focus_key:
                refresh(focus_key=focus_key)
            else:
                refresh_table_selection()

        def handle_match_toggle(event) -> None:
            """Apply one checkbox toggle without trusting stale QTable state.

            The match table is re-rendered after every query and selection
            change.  Binding the slot checkbox directly with ``v-model`` can
            therefore make a second click report the previous selection back
            to the server.  The slot emits the requested next state instead;
            this handler updates the persistent selection sets first and then
            refreshes the table from those sets.
            """
            args = getattr(event, "args", None)
            if not isinstance(args, dict):
                return
            row = args.get("row")
            if not isinstance(row, dict):
                return
            if row.get("match_role") == "secondary":
                return
            value = str(row.get("match_value", "") or "").strip()
            if not value:
                return
            if bool(args.get("selected")):
                remember_match(value)
            else:
                forget_match(value)
            sync_query_selection()
            if bool(args.get("selected")):
                members = tuple(group_members.get(value, ()))
                if members:
                    refresh(focus_key=members[0])
                    return
            refresh_table_selection()

        def handle_body_selection(event) -> None:
            visible = {
                str(row.get("__neuron_key", "") or ""): row
                for row in current_rows
            }
            selected_keys = {
                str(row.get("__neuron_key", "") or "")
                for row in list(getattr(event, "selection", []) or [])
            }
            # Unchecking one body row breaks a whole-name selection. The
            # remaining checked rows become individual selections.
            for value in list(selected_match_values):
                visible_group = (
                    selected_match_members.get(value, set())
                    | group_members.get(value, set())
                ) & set(visible)
                if visible_group and not visible_group.issubset(selected_keys):
                    forget_match(value)

            active_group_keys = set()
            for value in selected_match_values:
                active_group_keys.update(group_members.get(value, set()))
            for key, row in visible.items():
                if key not in selected_keys:
                    selected_body_ids.pop(key, None)
                elif key not in active_group_keys:
                    value = row_body_id(row)
                    if value:
                        selected_body_ids[key] = value
            sync_query_selection()
            refresh_table_selection()

        def remove_query_value(value: str) -> None:
            """Remove one value from the mirrored query and selection state."""
            value = str(value or "").strip()
            if not value:
                return
            forget_match(value)
            for key, body_id in list(selected_body_ids.items()):
                if str(body_id or "").strip() == value:
                    selected_body_ids.pop(key, None)
            sync_query_selection()
            if query_remove is not None:
                query_remove(value)
            if query_values_getter is not None:
                refresh_query_preview()
            refresh_table_selection()

        with ui.element("div").classes("w-full drocat-neuron-results-layout"):
            with ui.element("section").classes("drocat-neuron-match-panel"):
                with ui.row().classes("w-full items-center justify-between gap-2"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("manage_search", color="primary").classes("text-lg")
                        ui.label("Match details").classes("text-subtitle2 font-bold")
                    with ui.row().classes("items-center gap-2"):
                        if query_callback is not None:
                            selection_status = ui.label("0 selected").classes(
                                "text-caption drocat-muted"
                            )
                match_status = ui.label("No matched names").classes(
                    "text-caption drocat-muted"
                )
                match_table = ui.table(
                    rows=current_groups,
                    columns=match_columns,
                    row_key="__match_group_key",
                    selection="multiple",
                    on_select=handle_match_selection,
                    pagination=None,
                ).classes("w-full drocat-neuron-match-table")
                match_table.add_slot(
                    "body",
                    r"""
                    <q-tr
                      :props="props"
                      :class="{
                        'drocat-neuron-match-secondary-row': props.row.match_role === 'secondary',
                      }"
                    >
                      <q-td auto-width class="drocat-neuron-match-select-cell">
                      <q-icon
                        v-if="props.row.match_role === 'secondary'"
                        name="arrow_right_alt"
                        class="drocat-neuron-match-secondary-arrow"
                        size="18px"
                      />
                      <q-checkbox
                        v-else
                        :model-value="props.selected"
                        dense
                        @click.stop="$parent.$emit('match-selection-toggle', { row: props.row, selected: !props.selected })"
                      />
                      </q-td>
                      <q-td key="match_column" :props="props">
                        <div class="drocat-neuron-match-source">
                          {{ props.row.match_column }}
                        </div>
                      </q-td>
                      <q-td key="match_value" :props="props">
                        <div class="drocat-neuron-match-value-line">
                          <q-btn
                            flat dense no-caps
                            class="drocat-neuron-match-jump"
                            :label="props.row.match_value"
                            @click.stop="$parent.$emit('match-value-click', props.row)"
                          />
                        </div>
                        <div
                          v-if="props.row.first_body_id"
                          class="drocat-neuron-match-first"
                        >first bodyId: {{ props.row.first_body_id }}</div>
                      </q-td>
                      <q-td key="body_count" :props="props" class="text-right">
                        {{ props.row.body_count }}
                      </q-td>
                    </q-tr>
                    """,
                )
                with ui.row().classes(
                    "w-full items-center justify-between gap-2 flex-wrap"
                ):
                    match_page_position = ui.label("Page 1 of 1").classes(
                        "text-caption drocat-muted"
                    )
                    with ui.row().classes("items-center gap-1"):
                        match_previous_button = ui.button(
                            "Previous matches", icon="chevron_left"
                        ).props("flat dense")
                        match_next_button = ui.button(
                            "Next matches", icon="chevron_right"
                        ).props("flat dense")

            with ui.element("section").classes("drocat-neuron-full-panel"):
                with ui.row().classes("w-full items-center gap-2"):
                    ui.icon("table_view", color="primary").classes("text-lg")
                    ui.label("Full neuron metadata").classes("text-subtitle2 font-bold")
                ui.label(
                    "Scroll horizontally to inspect every retained metadata field."
                ).classes("text-caption drocat-muted")
                with ui.element("div").classes("w-full drocat-data-viewer-scroll"):
                    table = ui.table(
                        rows=initial.rows,
                        columns=table_columns,
                        row_key="__neuron_key",
                        selection="multiple",
                        on_select=handle_body_selection,
                        pagination=None,
                    ).classes("w-full drocat-data-viewer-table")
                    # Keep the table's default selection checkbox at the left
                    # while using a body slot for the row-specific highlight.
                    table.add_slot(
                        "body",
                        r"""
                        <q-tr
                          :data-neuron-key="props.row.__neuron_key"
                          :class="{
                            'drocat-neuron-selected-row': props.selected,
                          }"
                        >
                          <q-td auto-width>
                            <q-checkbox v-model="props.selected" dense />
                          </q-td>
                          <q-td
                            v-for="col in props.cols"
                            :key="col.name"
                            :props="props"
                            :class="{
                              'drocat-neuron-hit-cell': (
                                props.row.match_column_keys || [props.row.match_column_key]
                              ).includes(col.name),
                              'drocat-neuron-secondary-hit-cell': (
                                props.row.secondary_match_column_keys || []
                              ).includes(col.name),
                            }"
                            :data-match-column="(
                              props.row.match_column_keys || [props.row.match_column_key]
                            ).includes(col.name) ? col.name : null"
                            :data-match-role="(
                              props.row.secondary_match_column_keys || []
                            ).includes(col.name) ? 'secondary' : null"
                          >
                            <span
                              v-if="props.row.__highlighted_cells && props.row.__highlighted_cells[col.name]"
                              v-html="props.row.__highlighted_cells[col.name]"
                            ></span>
                            <span v-else>{{ props.row[col.field] }}</span>
                          </q-td>
                        </q-tr>
                        """,
                    )

        with ui.row().classes("w-full items-center justify-between gap-3 flex-wrap"):
            page_info = ui.label("").classes("text-caption drocat-muted")
            no_results = ui.label("No rows match the current search/filter.").classes(
                "text-caption drocat-warn"
            )
            page_position = ui.label("").classes("text-caption drocat-muted")
            previous_button = ui.button("Previous page", icon="chevron_left").props(
                "flat dense"
            )
            next_button = ui.button("Next page", icon="chevron_right").props(
                "flat dense"
            )

        state = {"page": initial.page, "page_size": 50}

    def scroll_to_table_row(focus_key: str) -> None:
        """Scroll the focused metadata row into the visible table viewport."""
        encoded_key = json.dumps(str(focus_key or ""))
        ui.run_javascript(
            f"""
            setTimeout(() => {{
                const key = {encoded_key};
                const root = document.querySelector('.drocat-neuron-full-panel .drocat-data-viewer-scroll');
                if (!root) return;
                const row = Array.from(root.querySelectorAll('tr[data-neuron-key]'))
                    .find(candidate => candidate.dataset.neuronKey === key);
                if (row) {{
                    row.scrollIntoView({{ behavior: 'smooth', block: 'center', inline: 'nearest' }});
                    row.classList.remove('drocat-neuron-focus-flash');
                    void row.offsetWidth;
                    row.classList.add('drocat-neuron-focus-flash');
                    window.setTimeout(() => row.classList.remove('drocat-neuron-focus-flash'), 1400);
                }}
            }}, 80);
            """
        )

    def render_match_page() -> None:
        """Render one bounded match-detail page without re-running the query."""
        total = len(match_groups_all)
        pages = max(1, (total + MATCH_GROUP_PAGE_SIZE - 1) // MATCH_GROUP_PAGE_SIZE)
        match_state["page"] = max(1, min(match_state["page"], pages))
        start = (match_state["page"] - 1) * MATCH_GROUP_PAGE_SIZE
        end = min(total, start + MATCH_GROUP_PAGE_SIZE)
        current_groups[:] = match_groups_all[start:end]
        match_table.update_rows(current_groups)
        if total:
            match_status.text = (
                f"Showing {start + 1:,}–{end:,} of {total:,} matched names"
            )
        else:
            match_status.text = "No matched names"
        match_status.update()
        match_page_position.text = (
            f"Page {match_state['page']:,} of {pages:,}"
        )
        match_page_position.update()
        match_previous_button.set_enabled(match_state["page"] > 1)
        match_next_button.set_enabled(match_state["page"] < pages)

    def refresh(_event=None, *, reset_page: bool = False, focus_key: str | None = None):
        if reset_page:
            state["page"] = 1
            match_state["page"] = 1
        try:
            current_page_size = int(page_size.value or 50)
        except (TypeError, ValueError):
            current_page_size = 50
        state["page_size"] = current_page_size
        requested_sort = sort_column.value
        if requested_sort == "__match_value__":
            # Ascending matched-value order is the implicit default. Preserve
            # an explicitly chosen descending direction for that same sort.
            requested_sort = (
                "__match_value__" if direction.value == "desc" else None
            )
        def run_query(requested_page: int, requested_focus_key: str | None = None):
            return query_neuron_index(
                index,
                search=search_input.value or "",
                search_column=target_column.value,
                search_operator=filter_operator.value,
                sort_by=requested_sort,
                descending=direction.value == "desc",
                page=requested_page,
                page_size=current_page_size,
                focus_key=requested_focus_key,
            )

        result = run_query(state["page"], focus_key)
        if focus_key and result.focus_page and result.focus_page != result.page:
            # The first pass computes the sorted position; the second fetches
            # only the page containing that position.
            result = run_query(result.focus_page)
        state["page"] = result.page
        current_rows[:] = list(result.rows)
        match_groups_all[:] = list(result.match_groups)
        current_group_body_ids.clear()
        current_group_body_ids.update({
            str(key): tuple(values)
            for key, values in result.match_group_body_ids.items()
        })
        match_group_related.clear()
        match_group_related.update({
            str(key): tuple(values)
            for key, values in result.match_group_related.items()
        })
        match_group_primary.clear()
        match_group_primary.update({
            str(key): tuple(values)
            for key, values in result.match_group_primary.items()
        })
        group_members.clear()
        group_members.update({
            str(key): set(values)
            for key, values in result.match_group_members.items()
        })
        render_match_page()
        table.update_rows(current_rows)
        if focus_key:
            scroll_to_table_row(focus_key)
        # Restore selections after replacing the rows. On a different page,
        # only matching visible rows are checked; the underlying selection
        # sets still retain entries selected on other pages.
        refresh_table_selection()
        page_position.text = f"Page {result.page:,} of {result.pages:,}"
        page_position.update()
        previous_button.set_enabled(result.page > 1)
        next_button.set_enabled(result.page < result.pages)
        if result.total:
            start = (result.page - 1) * result.page_size + 1
            end = min(result.total, result.page * result.page_size)
            page_info.text = f"Showing {start:,}–{end:,} of {result.total:,} matching rows"
        else:
            page_info.text = "0 matching rows"
        page_info.update()
        no_results.set_visibility(result.total == 0)

    def reset_and_refresh(_event=None):
        refresh(reset_page=True)

    def handle_match_value_click(event):
        row = getattr(event, "args", None)
        if not isinstance(row, dict):
            return
        value = str(row.get("match_value", "") or "").strip()
        member_keys = tuple(group_members.get(value, ()))
        if member_keys:
            refresh(focus_key=member_keys[0])

    search_input.on_value_change(reset_and_refresh)
    def handle_filter_column_change(event):
        has_target = target_column.value not in {None, "", "__none__"}
        filter_operator.set_enabled(has_target)
        reset_and_refresh(event)

    target_column.on_value_change(handle_filter_column_change)
    filter_operator.on_value_change(reset_and_refresh)
    sort_column.on_value_change(reset_and_refresh)
    direction.on_value_change(reset_and_refresh)
    page_size.on_value_change(reset_and_refresh)

    def change_page(delta):
        state["page"] = max(1, state["page"] + delta)
        refresh()

    def change_match_page(delta):
        match_state["page"] = max(1, match_state["page"] + delta)
        render_match_page()
        refresh_table_selection()

    previous_button.on_click(lambda: change_page(-1))
    next_button.on_click(lambda: change_page(1))
    match_previous_button.on_click(lambda: change_match_page(-1))
    match_next_button.on_click(lambda: change_match_page(1))
    match_table.on("match-value-click", handle_match_value_click)
    match_table.on("match-selection-toggle", handle_match_toggle)
    refresh()


def create_neuron_index_viewer_link(
    dataset_getter: Callable[[], object],
    *,
    label: str = "See available neurons",
    query_values_getter: Callable[[], object] | None = None,
    query_selection: Callable[[List[str]], object] | None = None,
    query_resolution: Callable[[List[str]], object] | None = None,
    query_remove: Callable[[str], object] | None = None,
    add_to_query: Callable[[List[str]], object] | None = None,
    query_label: str = "Current query",
):
    """Create a link-like control that opens the cached-index viewer.

    ``dataset_getter`` is evaluated at click time, so changing the dataset in
    a tool tab immediately changes the viewer target.  A multi-dataset getter
    (the Cross-Dataset tab) gets a dataset selector inside the dialog.
    When query callbacks are supplied, the match panel supports multi-select
    and synchronizes selected matched values with the owning query input.
    ``query_remove`` makes the mirrored query preview editable by removing
    one value at a time.
    """
    dialog = ui.dialog()
    with dialog:
        with ui.card().classes("w-[min(98vw,1800px)] max-w-none"):
            with ui.row().classes("w-full items-center justify-between gap-3"):
                title = ui.label("Available neurons").classes("text-h6")
                ui.button(icon="close", on_click=dialog.close).props("flat round dense")
            dataset_picker_slot = ui.row().classes("w-full items-center")
            content = ui.column().classes("w-full gap-2")

    def open_viewer():
        try:
            datasets = _dataset_values(dataset_getter())
        except Exception as exc:
            datasets = []
            error = str(exc)
        else:
            error = ""

        dataset_picker_slot.clear()
        content.clear()
        if error:
            title.text = "Available neurons"
            with content:
                ui.label(f"Could not determine the selected dataset: {error}").classes(
                    "text-body2 drocat-err"
                )
        elif not datasets:
            title.text = "Available neurons"
            with content:
                ui.label("Choose a dataset before opening the neuron index.").classes(
                    "text-body2 drocat-warn"
                )
        else:
            if len(datasets) > 1:
                with dataset_picker_slot:
                    picker = ui.select(
                        options=datasets,
                        value=datasets[0],
                        label="Dataset to view",
                    ).classes("drocat-select").style("min-width: 280px")
                picker.on_value_change(lambda event: _open_dataset(str(event.value)))
            _open_dataset(datasets[0])
        dialog.open()

    def _open_dataset(dataset: str):
        title.text = f"Available neurons · {dataset}"
        title.update()
        _render_index(
            content,
            dataset,
            query_values_getter=query_values_getter,
            query_selection=query_selection,
            query_resolution=query_resolution,
            query_remove=query_remove,
            add_to_query=add_to_query,
            query_label=query_label,
        )

    link = ui.button(label, icon="table_view", on_click=open_viewer).props(
        "flat dense no-caps"
    ).classes("drocat-inline-link")
    # Expose the dialog for component-level tests and for callers that want
    # to close it after changing tabs.
    link.neuron_index_dialog = dialog
    return link
