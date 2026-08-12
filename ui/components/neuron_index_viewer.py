"""NiceGUI viewer for a locally cached neuron index."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List

from nicegui import ui

from ..config import PROJECT_ROOT
from ..neuron_index import (
    load_cached_neuron_index,
    neuron_index_path,
    query_neuron_index,
)


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


def _render_index(content, dataset: str) -> None:
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

        ui.label(
            "Search and column filters use case-insensitive substring matching. "
            "Search results prioritize bodyId, type, instance, then other columns. "
            "The Matched by column shows the highest-priority match; bodyId "
            "matches show their corresponding instance when available. The "
            "matched value is pinned beside it and the original cell is highlighted."
        ).classes("text-caption drocat-muted")

        with ui.row().classes("w-full items-end gap-2 flex-wrap"):
            search_input = ui.input(
                "Search all columns",
                placeholder="e.g. aMe12 or 5813",
            ).props("clearable input-debounce=180").classes("flex-grow drocat-input")
            filter_options = {"__all__": "All columns"}
            filter_options.update({column: _column_label(column) for column in columns})
            filter_column = ui.select(
                options=filter_options,
                value="__all__",
                label="Filter column",
            ).classes("drocat-select").style("min-width: 150px")
            filter_input = ui.input(
                "Column filter",
                placeholder="contains...",
            ).props("clearable input-debounce=180").classes("drocat-input").style(
                "min-width: 170px"
            )
            sort_column = ui.select(
                options={column: _column_label(column) for column in columns},
                value="bodyId" if "bodyId" in columns else columns[0],
                label="Sort by",
            ).classes("drocat-select").style("min-width: 150px")
            direction = ui.select(
                options={"asc": "Ascending", "desc": "Descending"},
                value="asc",
                label="Order",
            ).classes("drocat-select").style("min-width: 140px")
            page_size = ui.select(
                options={25: "25 / page", 50: "50 / page", 100: "100 / page", 200: "200 / page"},
                value=50,
                label="Rows",
            ).classes("drocat-select").style("min-width: 120px")

        initial = query_neuron_index(index, page_size=50)
        table_columns = [
            {
                "name": "match_column",
                "label": "Matched by",
                "field": "match_column",
                "align": "left",
                "classes": "text-grey-6 text-caption drocat-neuron-match-by",
                "headerClasses": "text-grey-6 drocat-neuron-match-by",
                "style": "width: 150px; min-width: 150px",
                "headerStyle": "width: 150px; min-width: 150px",
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
        with ui.element("div").classes("w-full drocat-data-viewer-scroll"):
            table = ui.table(
                rows=initial.rows,
                columns=table_columns,
                row_key="bodyId" if "bodyId" in columns else columns[0],
                pagination=None,
            ).classes("w-full drocat-data-viewer-table")
            # Render the table body explicitly so the match metadata remains
            # pinned at the left while the actual matched source cell receives
            # a row-specific highlight.  The pinned matched value is a stable
            # floating representation when the source column is horizontally
            # outside the current viewport.
            table.add_slot(
                "body",
                r"""
                <q-tr :props="props">
                  <q-td
                    v-for="col in props.cols"
                    :key="col.name"
                    :props="props"
                    :class="{
                      'drocat-neuron-match-by': col.name === 'match_column',
                      'drocat-neuron-match-value': col.name === 'match_value',
                      'drocat-neuron-hit-cell': col.name === props.row.match_column_key,
                    }"
                    :data-match-column="col.name === props.row.match_column_key ? col.name : null"
                  >
                    {{ col.name === 'match_column'
                        ? props.row.match_column
                        : col.name === 'match_value'
                          ? props.row.match_value
                          : props.row[col.field] }}
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

    def refresh(_event=None, *, reset_page: bool = False):
        if reset_page:
            state["page"] = 1
        try:
            current_page_size = int(page_size.value or 50)
        except (TypeError, ValueError):
            current_page_size = 50
        state["page_size"] = current_page_size
        result = query_neuron_index(
            index,
            search=search_input.value or "",
            filter_column=filter_column.value,
            filter_text=filter_input.value or "",
            sort_by=sort_column.value,
            descending=direction.value == "desc",
            page=state["page"],
            page_size=current_page_size,
        )
        state["page"] = result.page
        table.update_rows(result.rows)
        table.update()
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

    search_input.on_value_change(reset_and_refresh)
    filter_column.on_value_change(reset_and_refresh)
    filter_input.on_value_change(reset_and_refresh)
    sort_column.on_value_change(reset_and_refresh)
    direction.on_value_change(reset_and_refresh)
    page_size.on_value_change(reset_and_refresh)

    def change_page(delta):
        state["page"] = max(1, state["page"] + delta)
        refresh()

    previous_button.on_click(lambda: change_page(-1))
    next_button.on_click(lambda: change_page(1))
    refresh()


def create_neuron_index_viewer_link(
    dataset_getter: Callable[[], object],
    *,
    label: str = "See available neurons",
):
    """Create a link-like control that opens the cached-index viewer.

    ``dataset_getter`` is evaluated at click time, so changing the dataset in
    a tool tab immediately changes the viewer target.  A multi-dataset getter
    (the Cross-Dataset tab) gets a dataset selector inside the dialog.
    """
    dialog = ui.dialog()
    with dialog:
        with ui.card().classes("w-[min(96vw,1400px)] max-w-none"):
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
        _render_index(content, dataset)

    link = ui.button(label, icon="table_view", on_click=open_viewer).props(
        "flat dense no-caps"
    ).classes("drocat-inline-link")
    # Expose the dialog for component-level tests and for callers that want
    # to close it after changing tabs.
    link.neuron_index_dialog = dialog
    return link
