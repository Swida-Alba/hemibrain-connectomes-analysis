"""Shared UI components for DROCAT.

Design language follows the Photo Selector "gallery" reference:
light canvas, white surfaces, cobalt accent, rounded corners and
a focus-panel + contact-sheet workspace layout.
"""

import asyncio
import os
import re

from nicegui import ui
from typing import List, Optional, Callable, Tuple
from pathlib import Path
import inspect
import json
import platform
import subprocess

from .. import group_history
from ..config import PROJECT_ROOT, get_default_output_dir, set_default_output_dir


# Every dir_input instance (all output-directory fields stay in sync).
_OUTPUT_DIR_INPUTS = []


# =============================================================================
# Page layout helper (focus panel + results / contact sheet)
# =============================================================================

def tool_page(
    title: str,
    subtitle: str = "",
    icon: str = "science",
    tag: Optional[str] = None,
    tag_color: str = "purple",
    doc: Optional[str] = None,
) -> tuple:
    """
    Create the two-column workspace used by every tool tab.

    Returns (form_column, results_column). The form is the "focus panel"
    where the user configures a run; the results column is the "contact
    sheet" showing status, log and output files.
    """
    with ui.column().classes("w-full drocat-page gap-0") as page_col:
        with ui.row().classes("w-full items-center gap-3 drocat-page-head"):
            with ui.element("div").classes("drocat-page-mark"):
                ui.icon(icon).classes("text-white")
            with ui.column().classes("gap-1"):
                with ui.row().classes("items-center gap-2"):
                    ui.label(title).classes("drocat-page-title")
                    if tag:
                        ui.badge(tag, color=tag_color).classes("drocat-tag-badge")
                    if doc:
                        ui.link(
                            "Instructions",
                            f"docs/ui_guides/{doc.replace('.md', '.html')}",
                        ).classes("drocat-doc-link")
                if subtitle:
                    ui.label(subtitle).classes("drocat-page-sub")
        with ui.row().classes("w-full drocat-workspace items-start"):
            form_col = ui.column().classes("drocat-form gap-3")
            results_col = ui.column().classes("drocat-results gap-3")
    # Let full-width blocks (e.g. the inline custom grouper) mount themselves
    # below the two-column workspace instead of squeezing into the form column.
    form_col._drocat_page = page_col
    results_col._drocat_page = page_col
    return form_col, results_col


def section_header(title: str, icon: str = "settings"):
    """Create a section header with icon."""
    with ui.row().classes("items-center gap-2 drocat-section-head"):
        with ui.element("div").classes("drocat-section-icon"):
            ui.icon(icon).classes("text-primary")
        ui.label(title).classes("drocat-section-title")


def param_grid(columns: int = 2):
    """Create a responsive grid container for parameters."""
    return ui.grid(columns=columns).classes("w-full drocat-param-grid")


# =============================================================================
# Dataset Selector
# =============================================================================

def _dataset_label_parts(ds: str, service) -> List[str]:
    """Build the option label parts with source + local status tags."""
    src_tag = "[FW]" if ds.startswith("flywire_") else "[NP]"
    status_tag = ""
    info = service._cache.get(ds)
    if info and info.local_prepared:
        status_tag = "✓ local"
    elif info and info.local_cache:
        status_tag = "◐ cached"
    elif info and info.available:
        status_tag = "☁ server"
    else:
        if service._check_local_prepared(ds):
            status_tag = "✓ local"
        elif service._check_local_cache(ds):
            status_tag = "◐ cached"
        else:
            status_tag = ""
    return [ds, src_tag] + ([status_tag] if status_tag else [])


def dataset_selector(
    label: str = "Dataset",
    default: Optional[str] = None,
    datasets: Optional[List[str]] = None,
    on_change: Optional[Callable] = None,
    hint: str = "NeuPrint: fetched from server with token. FlyWire: uses converted local files; CAVE token is only needed for CAVE API features.",
    allow_custom: bool = False,
    show_local_status: bool = True,
) -> ui.select:
    """Create a dataset dropdown selector with local status labels."""
    from ..dataset_service import get_dataset_service

    service = get_dataset_service()
    options = datasets if datasets is not None else service.get_all_datasets()

    if show_local_status:
        labeled_options = {
            ds: "  ".join(_dataset_label_parts(ds, service)) for ds in options
        }
        sel_options = labeled_options
    else:
        sel_options = options

    default_val = default if default is not None else (options[0] if options else None)
    sel = ui.select(
        options=sel_options,
        value=default_val,
        label=label,
    ).classes("w-full drocat-select").tooltip(hint)
    if allow_custom:
        sel.props('use-input new-value-mode="add-unique"')
    if on_change:
        sel.on_value_change(on_change)
    return sel


def dataset_multi_selector(
    label: str = "Datasets",
    default: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    hint: str = "Select 2+ datasets to compare. Shows [NP]=NeuPrint, [FW]=FlyWire, ✓ local / ☁ server status.",
    show_local_status: bool = True,
) -> ui.select:
    """Create a multi-select dataset dropdown with local status labels."""
    from ..dataset_service import get_dataset_service

    service = get_dataset_service()
    options = datasets if datasets is not None else service.get_all_datasets()

    if show_local_status:
        sel_options = {
            ds: "  ".join(_dataset_label_parts(ds, service)) for ds in options
        }
    else:
        sel_options = options

    default_val = default or (options[:2] if len(options) >= 2 else options)
    sel = ui.select(
        options=sel_options,
        value=default_val,
        label=label,
        multiple=True,
    ).classes("w-full drocat-select").props('use-chips use-input').tooltip(hint)
    return sel


# =============================================================================
# Advanced Neuron Input (with filter mode)
# =============================================================================

def advanced_neuron_input(
    label: str = "Neurons",
    placeholder: str = "e.g., aMe12, aMe10, DN1p",
    hint: str = "Enter neuron types, bodyIds, or patterns. Use filter mode for pattern matching.",
) -> ui.element:
    """Create an advanced neuron input with filter mode selector."""
    with ui.column().classes("w-full gap-0") as container:
        with ui.row().classes("w-full items-end gap-2"):
            textarea = ui.textarea(
                label=label,
                placeholder=placeholder,
            ).props("autogrow").classes("flex-grow drocat-input").tooltip(hint)
            filter_mode = ui.select(
                options={
                    "exact": "Exact",
                    "startswith": "Starts with",
                    "contains": "Contains",
                    "endswith": "Ends with",
                    "regex": "Regex",
                },
                value="exact",
                label="Filter",
            ).classes("w-32 drocat-select").props("dense outlined").tooltip(
                "Exact: match type/bodyId exactly\n"
                "Starts with: e.g. 'DN' → DN1p, DN2\n"
                "Contains: e.g. 'PN' → adPN, lPN\n"
                "Ends with: e.g. '_R' → right hemisphere\n"
                "Regex: e.g. 'KC.*' → all KC types"
            )

    container.get_value = lambda: (filter_mode.value, textarea.value)
    container.textarea = textarea
    container.filter_mode = filter_mode
    return container


def neuron_input(
    label: str = "Neurons",
    placeholder: str = "e.g., aMe12, aMe10, DN1p",
    hint: str = "Comma or newline separated. Supports types, bodyIds, regex patterns like KC.*",
) -> ui.textarea:
    """Create a simple neuron query input textarea."""
    return ui.textarea(
        label=label,
        placeholder=placeholder,
    ).props("autogrow").classes("w-full drocat-input").tooltip(hint)


async def read_upload_event(event) -> tuple[str, bytes]:
    """Read a NiceGUI 3 upload event, with compatibility for older events."""
    upload = getattr(event, "file", None)
    if upload is not None:
        return upload.name, await upload.read()

    name = getattr(event, "name", "upload")
    content = getattr(event, "content", None)
    if content is None:
        raise ValueError("Upload event contains no file")
    data = content.read()
    if inspect.isawaitable(data):
        data = await data
    return name, data


def parse_neuron_upload(filename: str, content: bytes) -> List:
    """Parse the first column of a CSV/TSV/Excel neuron-list upload."""
    import csv
    import io
    from numbers import Integral, Real

    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".tsv", ".xlsx", ".xls"}:
        raise ValueError("Use a CSV, TSV, XLSX, or XLS neuron-list file")
    if suffix in {".xlsx", ".xls"}:
        import pandas as pd

        frame = pd.read_excel(io.BytesIO(content), header=None)
        values = frame.iloc[:, 0].dropna().tolist()
    else:
        text = content.decode("utf-8-sig", errors="replace")
        delimiter = "\t" if suffix == ".tsv" else ","
        values = [
            row[0]
            for row in csv.reader(io.StringIO(text), delimiter=delimiter)
            if row
        ]

    loaded = [
        (value, str(value).strip().strip("\"'"))
        for value in values
        if str(value).strip().strip("\"'")
    ]
    if loaded and loaded[0][1].lower() in {
        "type", "neuron", "name", "bodyid", "id", "body_id",
    }:
        loaded = loaded[1:]
    parsed = []
    for raw_value, value in loaded:
        if isinstance(raw_value, Integral):
            parsed.append(int(raw_value))
            continue
        if isinstance(raw_value, Real) and float(raw_value).is_integer():
            parsed.append(int(raw_value))
            continue
        try:
            parsed.append(int(value))
        except ValueError:
            parsed.append(value)
    return parsed


def _normalize_neuron_value(item):
    """Normalize body IDs to integers while preserving neuron type strings."""
    value = str(item)
    return int(value) if value.isdigit() else value


def neuron_list_input(
    label: str = "Neurons",
    placeholder: str = "e.g., aMe12, aMe10, DN1p (or upload CSV/TSV/Excel)",
    hint: str = (
        "Type a name and press Enter or leave the field to add it as a chip. "
        "Commas and spaces are kept as part of the name. "
        "Use the playlist button to paste lists, or upload a CSV/TSV/Excel "
        "file (first column)."
    ),
    unit_label: str = "neuron",
    show_filter: bool = True,
    show_upload: bool = True,
    max_items: Optional[int] = None,
    initial: Optional[List] = None,
    suggestions: Optional[Callable[[str], List[Tuple[str, str]]]] = None,
    available_neurons: Optional[Callable[[], object]] = None,
    suggestion_min_chars: int = 1,
    suggestion_limit: int = 50,
) -> ui.element:
    """
    Create a chip-based list input for neurons.

    - Type a name (type, bodyId or pattern) and press Enter or leave the field
      to add it as a chip. The whole typed text becomes ONE chip — commas and
      spaces inside names are preserved, never treated as separators.
    - Paste a list via the playlist button (one value per line or
      comma-separated) or upload a CSV/TSV/Excel file (first column).
    - ``initial`` seeds the chip list with pre-existing values.
    - ``max_items`` caps the list (used for single-input tabs); additional
      values are rejected once the cap is reached.
    - A live count badge and a Clear button keep the list manageable. The
      badge names items with ``unit_label`` ("neuron" by default; pass e.g.
      "threshold" for non-neuron chip inputs so the counter reads
      "3 thresholds" instead of "3 neurons").
    - ``suggestions``: optional provider ``typed_text -> [(value, hint)]``
      powering the auto-suggest dropdown (dataset type/instance/bodyId names
      with the searched column as a gray hint). The provider is prefiltered
      from the first character and suggestions are shown from the first
      character by default (``suggestion_min_chars=1``). A blank focused field
      opens the persistent query history (last 10 + most frequent); history
      is not mixed into a nonblank dataset search. At most
      ``suggestion_limit`` entries are shown. With a provider, the native
      QSelect popup is replaced by the custom suggestion menu.
    - ``available_neurons``: optional zero-argument dataset getter. When
      supplied, a ``See available neurons`` link opens the rendered,
      searchable cached neuron-index viewer for the current dataset.
      The viewer's mirrored query chips can remove values through the same
      input when it supplies a ``query_remove`` callback.

    Returns container with .get_value() -> (filter_mode, neuron_list).
    """
    def _unit(count: int) -> str:
        """Pluralized unit label for count displays."""
        return f"{unit_label}{'s' if count != 1 else ''}"

    uploaded_neurons: List = []

    async def handle_upload(e):
        """Parse the first column of an uploaded CSV/TSV/Excel neuron list."""
        try:
            filename, raw = await read_upload_event(e)
            loaded = parse_neuron_upload(filename, raw)
            # Mutate in place so container.uploaded_neurons remains current.
            uploaded_neurons[:] = loaded
            upload_label.text = (
                f"✓ {len(uploaded_neurons)} {_unit(len(uploaded_neurons))} loaded from {filename}"
            )
            upload_label.classes(replace="text-caption drocat-ok")
        except Exception as exc:
            uploaded_neurons.clear()
            upload_label.text = f"Error: {exc}"
            upload_label.classes(replace="text-caption drocat-err")
        upload_label.set_visibility(True)
        update_status()
        upload_menu.close()

    with ui.column().classes("w-full gap-1") as container:
        with ui.row().classes("w-full items-end gap-2"):
            # Chip-based list input. Quasar normally commits a new value only
            # on Enter; the input event below preserves the editor text so it
            # can also be committed when the user moves focus elsewhere.
            # Seed values are kept in the options list: NiceGUI's QSelect only
            # renders chips whose values exist in its options (model-value is
            # filtered against them), so committed values must be added there.
            initial_values = [_normalize_neuron_value(item) for item in (initial or [])]
            # Keep a private anchor around each QSelect. The custom QMenu is
            # rendered into this wrapper below, rather than into the shared
            # input row; otherwise two neuron inputs on the same page would
            # anchor their menus to the same row and one popup could obscure
            # the other input during a focus change.
            with ui.element("div").classes("relative flex-grow") as chip_input_anchor:
                chip_input = ui.select(
                    options=list(initial_values),
                    value=list(initial_values),
                    label=label,
                    multiple=True,
                ).props(
                    'use-chips use-input new-value-mode="add-unique" '
                    'input-debounce="0"'
                ).classes("w-full drocat-select drocat-chip-input").tooltip(hint)

            filter_mode = None
            if show_filter:
                filter_mode = ui.select(
                    options={
                        "exact": "Exact",
                        "startswith": "Starts with",
                        "contains": "Contains",
                        "endswith": "Ends with",
                        "regex": "Regex",
                    },
                    value="exact",
                    label="Match by",
                ).classes("w-32 drocat-select").props("dense outlined").tooltip(
                    "How the query matches the Search Columns: exact, prefix "
                    "(starts with), substring (contains), suffix (ends with) "
                    "or regex pattern"
                )

            if show_upload:
                # Paste a whole list (comma / newline separated)
                with ui.button(icon="playlist_add").props("flat dense round").classes(
                    "drocat-upload-trigger"
                ).tooltip("Paste a list of neurons (comma or newline separated)"):
                    with ui.menu() as paste_menu:
                        ui.label("Paste neuron list").classes(
                            "text-caption drocat-muted px-3 pt-2"
                        )
                        ui.label(
                            "One per line or comma-separated — e.g. aMe12, aMe10"
                        ).classes("text-caption drocat-muted px-3 pb-1")
                        paste_area = ui.textarea(
                            placeholder="aMe12, aMe10\nPPL101, PPL103"
                        ).props("autogrow").classes("w-80 drocat-input")

                        def add_pasted():
                            items = parse_neuron_list(paste_area.value)
                            if not items:
                                return
                            current = list(chip_input.value or [])
                            existing = {str(c) for c in current}
                            for item in items:
                                if str(item) not in existing:
                                    current.append(item)
                                    existing.add(str(item))
                            if max_items is not None:
                                current = current[:max_items]
                            sync_options(current)
                            _suppress_history_popup["value"] = True
                            chip_input.value = current
                            update_status()
                            paste_area.value = ""
                            paste_menu.close()

                        ui.button(
                            "Add to list", icon="add", on_click=add_pasted
                        ).props("flat dense color=primary")

                # Compact upload: hidden inside a dropdown attached to the input row
                with ui.button(icon="upload_file").props("flat dense round").classes(
                    "drocat-upload-trigger"
                ).tooltip("Upload CSV/TSV/Excel (first column = neurons)"):
                    with ui.menu() as upload_menu:
                        ui.label("Load neuron list from file").classes(
                            "text-caption drocat-muted px-3 pt-2"
                        )
                        ui.label("CSV / TSV / XLSX / XLS · first column is read").classes(
                            "text-caption drocat-muted px-3 pb-1"
                        )
                        ui.upload(
                            label="Choose neuron file",
                            on_upload=handle_upload,
                            auto_upload=True,
                        ).props('accept=".csv,.xlsx,.xls,.tsv" flat dense').classes("w-72")
                        ui.link(
                            "File format instructions",
                            "docs/ui_guides/input_formats.html",
                        ).classes("drocat-doc-link px-3 pb-2")

        def sync_viewer_selection(values):
            """Synchronize viewer-owned chips without replacing user values.

            The viewer sends its complete current selection on every checkbox
            change. Values that were already in the input (or uploaded) are
            never owned by the viewer, so deselecting a viewer group cannot
            remove an unrelated pre-existing query chip.
            """
            current = list(chip_input.value or [])
            previous_viewer_values = set(viewer_owned_values)
            base = [
                value for value in current
                if str(value) not in previous_viewer_values
            ]
            base_keys = {str(value) for value in base}
            uploaded_keys = {str(value) for value in uploaded_neurons}
            merged = list(base)
            owned = set()
            selected_display = set()
            for item in values or []:
                value = _normalize_neuron_value(item)
                key = str(value)
                if key in base_keys or key in uploaded_keys:
                    # The viewer may select a value that the user already
                    # entered. Keep the existing chip, but still mark it as
                    # viewer-selected so execution uses the verified body IDs
                    # supplied by the index rather than resolving the name
                    # through a possibly colliding metadata column.
                    selected_display.add(key)
                    continue
                if max_items is not None and len(merged) >= max_items:
                    break
                if key in {str(existing) for existing in merged}:
                    continue
                merged.append(value)
                owned.add(key)
                selected_display.add(key)
            viewer_owned_values.clear()
            viewer_owned_values.update(owned)
            viewer_selected_values.clear()
            viewer_selected_values.update(selected_display)
            sync_options(merged)
            _suppress_history_popup["value"] = True
            chip_input.run_method("updateInputValue", "")
            chip_input.set_value(merged)
            pending_input["value"] = ""
            update_status()
            return len(owned)

        def sync_viewer_body_selection(values):
            """Keep exact body-ID resolution for viewer-owned display chips."""
            viewer_owned_body_ids[:] = [
                _normalize_neuron_value(item)
                for item in (values or [])
                if str(item or "").strip()
            ]

        def remove_viewer_query_value(value):
            """Remove a value from the input when its viewer chip is closed."""
            target = str(value or "").strip()
            if not target:
                return
            uploaded_neurons[:] = [
                item for item in uploaded_neurons
                if str(item or "").strip() != target
            ]
            current = list(chip_input.value or [])
            remaining = [
                item for item in current
                if str(item or "").strip() != target
            ]
            viewer_owned_values.discard(target)
            viewer_selected_values.discard(target)
            viewer_owned_body_ids[:] = [
                item for item in viewer_owned_body_ids
                if str(item or "").strip() != target
            ]
            if remaining != current:
                _suppress_history_popup["value"] = True
                chip_input.run_method("updateInputValue", "")
                sync_options(remaining)
                chip_input.set_value(remaining)
            update_status()

        # Status row: live count + upload status + clear
        with ui.row().classes("w-full items-center gap-2"):
            count_badge = ui.badge(f"0 {_unit(0)}", color="grey-6").props("outline")
            upload_label = ui.label("").classes("text-caption drocat-muted")
            upload_label.set_visibility(False)
            clear_button = ui.button(
                "Clear",
                icon="clear_all",
            ).props("flat dense").classes("drocat-clear-btn")
            if available_neurons is not None:
                # Import lazily to keep the common input component independent
                # from the optional viewer's Polars-backed data layer.
                from .neuron_index_viewer import create_neuron_index_viewer_link

                create_neuron_index_viewer_link(
                    available_neurons,
                    query_values_getter=lambda: [
                        *uploaded_neurons,
                        *(chip_input.value or []),
                    ],
                    query_selection=sync_viewer_selection,
                    query_resolution=sync_viewer_body_selection,
                    query_remove=remove_viewer_query_value,
                    query_label=label,
                )

    pending_input = {"value": ""}
    viewer_owned_values = set()
    viewer_selected_values = set()
    viewer_owned_body_ids = []
    # Bulk list changes (paste / clear) must not pop the Recent list open.
    _suppress_history_popup = {"value": False}

    def normalize_neuron(item):
        return _normalize_neuron_value(item)

    def sync_options(values):
        """Keep committed values in the option list so they render as chips.

        NiceGUI's QSelect derives the rendered model-value from its options
        (``Select._value_to_model_value`` skips values not in the options
        list), so any value committed programmatically — blur, paste, seed —
        must be added to ``options`` or the chip stays invisible.
        """
        for value in values:
            if value not in chip_input.options:
                chip_input.options.append(value)

    def update_status():
        combined = [normalize_neuron(item) for item in uploaded_neurons]
        combined.extend(normalize_neuron(item) for item in (chip_input.value or []))
        count = len(dict.fromkeys(combined))
        count_badge.text = f"{count} {_unit(count)}"
        count_badge.props(f"color={'primary' if count else 'grey-6'}")

    def clear_all():
        uploaded_neurons.clear()
        viewer_owned_values.clear()
        viewer_selected_values.clear()
        viewer_owned_body_ids.clear()
        _suppress_history_popup["value"] = True
        chip_input.set_value([])
        upload_label.text = ""
        upload_label.set_visibility(False)
        update_status()

    def remember_user_input(event):
        """Track the native editor input — user typing AND deletions.

        The native ``input`` event fires only for user edits (Quasar resets the
        editor programmatically, which never emits it), so an empty value here
        means the user cleared the field and the pending text must be dropped.
        """
        pending_input["value"] = str(getattr(event, "args", "") or "")

    def remember_quasar_input(event):
        """Track Quasar's ``input-value``, ignoring its blur-reset emission.

        Quasar emits ``input-value ''`` when it clears the editor after focus
        loss; that must not wipe the text typed just before the blur handler
        commits it. Non-empty values always replace the pending text.
        """
        value = str(getattr(event, "args", "") or "")
        if value.strip():
            pending_input["value"] = value

    def commit_pending_text(event=None):
        """Commit editor text when focus leaves the chip selector.

        The whole editor text becomes a single chip: commas and spaces are
        legal inside names (e.g. driver line names, 'A -> B -> C' layers) and
        are never treated as separators. Enter and focus loss are the only
        commit triggers.
        """
        # While the suggestion menu is open the blur comes from clicking a
        # suggestion — the click commits the picked value, not the typed text.
        if suggestions is not None and suggest_menu.value:
            return
        args = getattr(event, "args", None) if event is not None else None
        text = str(args or "") or pending_input["value"]
        pending_input["value"] = ""
        text = text.strip()
        if not text:
            return
        current = list(chip_input.value or [])
        # Single-item inputs reject any additional value once at capacity.
        if max_items is not None and len(current) >= max_items:
            update_status()
            return
        value = _normalize_neuron_value(text)
        if value not in current:
            merged = current + [value]
            if max_items is not None:
                merged = merged[:max_items]
            sync_options(merged)
            chip_input.set_value(merged)
        update_status()

    def handle_value_change(_event):
        # Enter, chip removal, paste, and uploads all update the model value.
        # In each case there is no longer an uncommitted editor value.
        pending_input["value"] = ""
        current = list(chip_input.value or [])
        # Enforce the item cap even for values Quasar added natively (Enter).
        if max_items is not None and len(current) > max_items:
            current = current[:max_items]
        sync_options(current)
        if current != list(chip_input.value or []):
            chip_input.set_value(current)
            return
        update_status()

    # ------------------------------------------------------------------
    # Auto-suggest + query history. Only active when a provider is wired
    # (the pathfinding tabs pass ``suggestions``): a custom menu replaces
    # the native QSelect popup (suppressed via popup-content-class) so
    # entries can render a solid name with a gray column hint and history
    # sections. History is read from ui/history_store (persisted per user).
    # ------------------------------------------------------------------
    suggest_menu = None
    if suggestions is not None:
        # The native QSelect popup would open empty on focus/typing; hide it
        # (CSS rule in ui/app.py) and drive the custom menu instead.
        chip_input.props('hide-dropdown-icon '
                         'popup-content-class="drocat-native-popup-hidden"')
        # Re-enter the per-input wrapper so QMenu uses this field as its
        # anchor. NiceGUI/Quasar portals the menu to the body at runtime, but
        # its position and outside-focus lifecycle still come from the parent.
        with chip_input_anchor:
            with ui.menu() as suggest_menu:
                pass
        suggest_menu.classes("drocat-suggest-menu")
        # Anchor the popup to THIS input's wrapper explicitly. Parent-component
        # anchoring alone can detach (menu renders at the page origin) when the
        # input is rebuilt inside nested containers (e.g. the inline grouper's
        # cells); a selector target resolves the anchor element directly.
        suggest_menu.props(
            f'target="#{chip_input_anchor.html_id}" '
            'anchor="bottom start" self="top start" fit '
            'max-height=240px'
        )
        # The menu must NEVER take focus from the editor: QMenu focuses itself
        # on open by default, which blurs the QSelect, clears the typed text
        # and swallows further keystrokes. no-focus keeps the editor focused
        # while the menu overlays. (No explicit target: NiceGUI 3.15 renders
        # no DOM ids, so the menu anchors to the per-input wrapper instead of
        # relying on a selector target.)
        # ``no-focus`` keeps the popup from taking the editor focus when it
        # opens. ``no-refocus`` is equally important when a second QSelect is
        # clicked: closing the first popup must not put focus back on its old
        # editor after the new field has already been selected. The target
        # wrapper normally gives QMenu its own parent-click toggle; disable
        # that toggle because it can run after the QSelect focus event and
        # close a history menu just opened by the server.
        suggest_menu.props('no-focus no-refocus no-parent-event')
        suggest_menu.style("overflow-y: auto;")

        # Server-side closes (1-char gating, rebuilds, picks) must NOT commit
        # the pending editor text; only genuine client-side closes (outside
        # click / ESC) commit it, like a plain blur.
        _close_guard = {"value": False}
        # Whether the pointer is inside the menu: a blur caused by clicking a
        # suggestion must not hide the menu before the click lands.
        _pointer_in_menu = {"value": False}
        # Whether the editor currently has focus (the Recent list only
        # reopens after a finished input while the field is still in use).
        _focused = {"value": False}
        # QSelect forwards editor changes through ``input-value`` while the
        # native input also emits ``input``. Both are wired below because
        # either event can be skipped by a browser/Quasar transition; this
        # guard prevents the same keystroke from rebuilding the menu twice.
        _last_suggest_text = {"value": None}
        # Each input owns its own candidate history. Appended characters can
        # filter this list locally; a full backend match is needed only after
        # the continuation no longer has candidates.
        _candidate_state = {"text": None, "entries": []}

        def _suggestions_enabled() -> bool:
            """Settings toggle: the whole feature can be switched off live."""
            from ..config import get_auto_suggest_enabled
            return get_auto_suggest_enabled()

        def _reset_candidate_state():
            _candidate_state["text"] = None
            _candidate_state["entries"] = []

        def _get_suggestions(text: str) -> List[Tuple[str, str]]:
            """Reuse a case-sensitive continuation candidate list when it
            still narrows; otherwise ask the backend for a fresh staged set.
            """
            query = str(text).strip()
            if not query:
                _reset_candidate_state()
                return []

            previous_text = _candidate_state["text"]
            previous_entries = _candidate_state["entries"]
            if previous_text == query:
                return list(previous_entries)

            if previous_text and query.startswith(previous_text):
                from ..type_suggestions import filter_candidate_entries

                narrowed = filter_candidate_entries(query, previous_entries)
                if narrowed:
                    _candidate_state["text"] = query
                    _candidate_state["entries"] = narrowed
                    return narrowed

            entries = list(suggestions(query) or [])
            _candidate_state["text"] = query
            _candidate_state["entries"] = entries
            return entries

        def _refresh_menu():
            """Show freshly rebuilt content even when the menu is already
            open: clearing an open q-menu empties it (Quasar hides an empty
            menu) and open() on an open menu is a no-op — so close it
            (guarded, no commit) and reopen it. The menu's no-focus and
            no-refocus props keep the active QSelect editor in control during
            this close/open cycle without issuing a delayed focus command that
            could steal focus from the other input during a rapid handoff."""
            if suggest_menu.value:
                _close_suggest()
            suggest_menu.open()

        def _close_suggest():
            _close_guard["value"] = True
            suggest_menu.close()
            _close_guard["value"] = False

        def _commit_suggestion(value):
            """Commit a picked suggestion/history value as a chip."""
            current = list(chip_input.value or [])
            if max_items is not None and len(current) >= max_items:
                return
            if value not in current:
                merged = current + [value]
                if max_items is not None:
                    merged = merged[:max_items]
                # Quasar's new-value-mode re-adds the leftover editor text as
                # a chip when the model changes externally; wipe it first so
                # only the picked value lands in the list.
                chip_input.run_method("updateInputValue", "")
                sync_options(merged)
                chip_input.set_value(merged)
            pending_input["value"] = ""
            _reset_candidate_state()
            update_status()
            # The menu state is managed by the finished-input handler: an
            # empty editor falls back to the Recent list.

        def _show_suggestions(entries):
            # Always discard the previous query before handling the new one.
            # This matters when a narrower query has no candidates: the menu
            # is hidden, but its old rows must not survive into a later reopen.
            suggest_menu.clear()
            if not entries:
                _close_suggest()
                return
            with suggest_menu:
                for value, hint in entries[:suggestion_limit]:
                    with ui.item().props("dense").on_click(
                            lambda v=value: _commit_suggestion(v)):
                        with ui.row().classes("items-center gap-2 no-wrap"):
                            ui.label(str(value)).classes("text-body2")
                            if hint:
                                ui.label(str(hint)).classes(
                                    "text-caption text-grey-6")
            _refresh_menu()

        def _show_history(query: str = ""):
            from ..history_store import (
                frequent as _frequent,
                prune_orphaned_custom as _prune_orphaned_custom,
                recent as _recent,
            )

            valid_custom_labels = group_history.valid_labels()
            _prune_orphaned_custom(valid_custom_labels)

            def matches_query(value: str) -> bool:
                # History filtering follows the same strict prefix behavior
                # as the first suggestion stage. It is intentionally kept
                # to history only while the editor has fewer than the
                # minimum number of characters for dataset suggestions.
                return not query or str(value).startswith(query)

            recents = [v for v in _recent() if matches_query(v)]
            freqs = [
                v for v in _frequent()
                if v not in recents and matches_query(v)
            ]
            if not recents and not freqs:
                _close_suggest()
                return
            suggest_menu.clear()
            with suggest_menu:
                if recents:
                    ui.item("Recent").props("dense disabled").classes(
                        "text-caption drocat-muted")
                    for v in recents:
                        _history_item(v, v in valid_custom_labels)
                if freqs:
                    ui.item("Frequent").props("dense disabled").classes(
                        "text-caption drocat-muted")
                    for v in freqs:
                        _history_item(v, v in valid_custom_labels)
            _refresh_menu()

        def _history_item(value, is_custom=False):
            with ui.item().props("dense").on_click(
                    lambda v=value: _commit_suggestion(v)):
                with ui.row().classes("items-center gap-2 no-wrap w-full"):
                    ui.label(str(value)).classes("text-body2")
                    if is_custom:
                        ui.badge("custom", color="grey-6").props(
                            "outline dense")

        def _on_suggest_input(event):
            # The editor text changed: refresh the list immediately (the
            # settings toggle switches the whole feature off at runtime).
            if not _suggestions_enabled():
                _reset_candidate_state()
                _close_suggest()
                return
            text = str(getattr(event, "args", "") or "")
            if text == _last_suggest_text["value"]:
                return
            _last_suggest_text["value"] = text
            if len(text.strip()) < suggestion_min_chars:
                # This branch is configurable for callers that intentionally
                # want a longer minimum. The default is one character, so a
                # blank focused editor is the only default path that renders
                # query history instead of dataset suggestions.
                if text.strip():
                    _get_suggestions(text.strip())
                else:
                    # A manual clear starts a new query. Do not let the next
                    # query reuse candidates from the text that was erased.
                    _reset_candidate_state()
                if _focused["value"]:
                    _show_history(text.strip())
                else:
                    _close_suggest()
                return
            _show_suggestions(_get_suggestions(text.strip()))

        def _on_suggest_input_value(event):
            """Handle Quasar's component event only while this field owns
            focus.

            The event can be delivered after a QSelect-to-QSelect focus
            transition. In that case the shared ``document.activeElement``
            value belongs to the other field; ignoring the event prevents it
            from resurrecting this field's menu.
            """
            if not _focused["value"]:
                _reset_candidate_state()
                _close_suggest()
                return
            _on_suggest_input(event)

        def _on_suggest_focus(_event):
            _close_other_suggestion_menus()
            _focused["value"] = True
            if not _suggestions_enabled():
                return
            # The menu is already showing suggestions (e.g. after a pick) —
            # do not flip it back to the history list.
            if suggest_menu.value:
                return
            # No editor text yet -> offer the persistent query history.
            if not pending_input["value"]:
                _reset_candidate_state()
                _show_history()

        def _on_suggest_blur(event):
            # Quasar can emit a component-level blur while it is moving focus
            # from the QSelect shell to its internal search input. That is not
            # a real focus change, and closing here causes the history menu to
            # flash on the first click. The client-side blur listener waits
            # one tick and reports whether the labelled editor still owns
            # focus; an actual outside click reports ``still_inside=False``.
            blur_args = getattr(event, "args", None)
            if isinstance(blur_args, dict) and blur_args.get("still_inside"):
                return
            if not _suggestions_enabled():
                # The setting can be changed from the Settings tab while a
                # menu is open. Treat the next focus change as a real blur
                # even in that disabled state so no stale popup survives.
                _focused["value"] = False
                _reset_candidate_state()
                _close_suggest()
                return
            # Focus moved INTO the menu: a pick is about to happen; keep the
            # field "focused" so the finished-input handler offers Recent.
            if _pointer_in_menu["value"]:
                return
            _focused["value"] = False
            _reset_candidate_state()
            # Focus left the field: hide the list automatically, then commit
            # the pending text like a plain blur.
            _close_suggest()
            commit_pending_text()

        def _deactivate_for_focus_change():
            """Hide this menu when another neuron input receives focus."""
            _focused["value"] = False
            _pointer_in_menu["value"] = False
            _reset_candidate_state()
            _close_suggest()
            # If Quasar skipped the old field's blur event, preserve the
            # normal finished-input behavior and commit its editor text now.
            commit_pending_text()

        def _close_other_suggestion_menus():
            """Make this input the only active suggestion-menu owner."""
            client = suggest_menu.client
            # Store the registry on the client itself: it is naturally scoped
            # to one page connection and disappears with that client, while
            # a module-level registry could retain page closures.
            registrations = getattr(client, "_drocat_suggestion_menus", [])
            alive = []
            for menu, deactivate in registrations:
                if getattr(menu, "_deleted", False):
                    continue
                alive.append((menu, deactivate))
                if menu is not suggest_menu:
                    deactivate()
            client._drocat_suggestion_menus = alive

        def _on_menu_hide(_event):
            _pointer_in_menu["value"] = False
            # The menu closed without a suggestion pick and NOT via a
            # server-side rebuild/close: the typed text is still pending —
            # commit it like a plain blur. Suggestion commits clear
            # pending_input first, so they no-op.
            if (not suggest_menu.value and pending_input["value"]
                    and not _close_guard["value"]):
                commit_pending_text()

        def _finished_input(_event):
            # A chip was added or removed (Enter / pick / x): the input
            # finished. With an empty editor and the field still in use,
            # offer the Recent list again; otherwise hide the menu.
            _reset_candidate_state()
            if not _suggestions_enabled():
                _close_suggest()
            elif _suppress_history_popup["value"]:
                _suppress_history_popup["value"] = False
                _close_suggest()
            elif not pending_input["value"] and _focused["value"]:
                _show_history()
            else:
                _close_suggest()

        registry = getattr(suggest_menu.client, "_drocat_suggestion_menus", [])
        registry.append((suggest_menu, _deactivate_for_focus_change))
        suggest_menu.client._drocat_suggestion_menus = registry

        chip_input.on("input", _on_suggest_input,
                      js_handler="(event) => emit(event?.target?.value ?? '')")
        # This is Quasar's canonical QSelect editor event. Keep it as a
        # second input source so updates continue even when the native DOM
        # input event is swallowed during popup/selection transitions.
        chip_input.on(
            "input-value",
            _on_suggest_input_value,
            # Quasar can emit this Vue event with the value from the previous
            # render while the native editor already contains the next
            # character. Read the focused editor at dispatch time so a stale
            # payload cannot reopen the previous query's rows.
            js_handler="() => emit(document.activeElement?.value ?? '')",
        )
        chip_input.on("focus", _on_suggest_focus)
        chip_input.on(
            "blur",
            _on_suggest_blur,
            js_handler=(
                "(event) => {"
                f"const label = {json.dumps(label)};"
                "setTimeout(() => {"
                "const active = document.activeElement;"
                "const stillInside = active?.getAttribute?.('aria-label') === label;"
                "emit({still_inside: Boolean(stillInside)});"
                "}, 0);"
                "}"
            ),
        )
        # The menu itself is not focusable, but keep its pointer guard so a
        # suggestion click is not treated as an outside blur before its item
        # handler commits the selected value.
        suggest_menu.on("mousedown",
                        lambda _e: _pointer_in_menu.__setitem__("value", True),
                        js_handler="(event) => emit(0)")
        suggest_menu.on_value_change(_on_menu_hide)

    # Capture the editor text while the user types. The native ``input`` event
    # (trusted typing sets the DOM value, and the js_handler ships the text) is
    # user-only, so it also tracks deletions; Quasar's ``input-value`` Vue event
    # mirrors the text but re-emits an empty value on blur-reset, which must be
    # ignored so the typed text survives until the commit handler runs.
    chip_input.on(
        "input",
        remember_user_input,
        js_handler="(event) => emit(event?.target?.value ?? '')",
    )
    chip_input.on(
        "input-value",
        remember_quasar_input,
        js_handler="() => emit(document.activeElement?.value ?? '')",
    )
    chip_input.on_value_change(handle_value_change)
    # Registered AFTER handle_value_change so the pending text is already
    # cleared when the "finished input" decision runs.
    if suggestions is not None:
        chip_input.on_value_change(_finished_input)
    # Commit the remembered editor text when the field loses focus, so a value
    # is added as a chip without requiring Enter. ``blur`` is a Quasar field
    # event and fires reliably when focus moves elsewhere.
    chip_input.on("blur", commit_pending_text)
    clear_button.on_click(clear_all)
    update_status()

    def get_value():
        display_values = [*uploaded_neurons, *(chip_input.value or [])]
        combined = []
        # Keep the query surface readable.  A viewer selection may have an
        # exact body-ID snapshot in ``viewer_owned_body_ids`` for UI display
        # and safeguards, but execution must receive the selected name here.
        # The shared resolver converts that name to body IDs later, after it
        # has applied bodyId -> type -> instance -> metadata priority.  This
        # prevents type queries from leaking body IDs into logs, parameters,
        # attributes, and auto-named output folders.
        combined.extend(normalize_neuron(item) for item in display_values)
        combined = list(dict.fromkeys(combined))
        if max_items is not None:
            combined = combined[:max_items]
        mode = filter_mode.value if filter_mode is not None else "exact"
        return (mode, combined)

    container.get_value = get_value
    container.chip_input = chip_input
    container.filter_mode = filter_mode
    container.uploaded_neurons = uploaded_neurons
    container.suggest_menu = suggest_menu

    def add_values(values) -> List:
        """Merge *values* into the chip list (programmatic append)."""
        current = list(chip_input.value or [])
        existing = {str(c) for c in current}
        for v in values:
            v = _normalize_neuron_value(v)
            if str(v) not in existing:
                current.append(v)
                existing.add(str(v))
        if max_items is not None:
            current = current[:max_items]
        _suppress_history_popup["value"] = True
        sync_options(current)
        chip_input.set_value(current)
        update_status()
        return current

    container.add_values = add_values
    return container


# =============================================================================
# Standard Form Inputs
# =============================================================================

def number_input(
    label: str,
    value: float = 0,
    min_val: float = 0,
    max_val: float = 1000,
    step: float = 1,
    hint: str = "",
) -> ui.number:
    """Create a numeric input field with tooltip."""
    inp = ui.number(
        label=label,
        value=value,
        min=min_val,
        max=max_val,
        step=step,
    ).classes("w-full drocat-input")
    if hint:
        inp.tooltip(hint)
    return inp


def select_input(
    label: str,
    options: List[str],
    default: Optional[str] = None,
    hint: str = "",
    help_doc: Optional[str] = None,
) -> ui.select:
    """Create a dropdown select input with tooltip.

    help_doc: name of a docs/ui_guides/*.html file; a small "guide" link is
    rendered under the select so the option list can be explained in depth.
    """
    with ui.column().classes("gap-0 w-full"):
        sel = ui.select(
            options=options,
            value=default or options[0],
            label=label,
        ).classes("w-full drocat-select")
        if hint:
            sel.tooltip(hint)
        if help_doc:
            ui.link(
                "Algorithm guide",
                f"docs/ui_guides/{help_doc}",
                new_tab=True,
            ).classes("text-caption drocat-doc-link")
    return sel


def checkbox_input(
    label: str,
    value: bool = True,
    hint: str = "",
) -> ui.checkbox:
    """Create a checkbox input with tooltip."""
    cb = ui.checkbox(label, value=value)
    if hint:
        cb.tooltip(hint)
    return cb


def _list_subdirs(path: str) -> List[str]:
    """Sorted subdirectory names under *path* ('' when not a directory)."""
    try:
        return sorted(
            (e for e in os.listdir(path) if os.path.isdir(os.path.join(path, e))),
            key=str.lower,
        )
    except OSError:
        return []


async def dir_browser_dialog(title: str = "Select Directory",
                             initial: str = "",
                             default_output: str = "") -> Optional[str]:
    """Open an in-browser directory picker (server-side folder listing).

    Replaces the old tkinter dialog, which blocked the whole web server
    until dismissed and could hang (no visible window) — freezing the app.
    This picker is fully in-browser and non-blocking: navigate subfolders,
    jump to a typed path, or select the currently shown folder. Returns the
    selected path or None when cancelled.
    """
    state = {"path": ""}
    result = {"path": None}
    done = asyncio.Event()

    def _norm(p: str) -> str:
        p = os.path.expanduser((p or "").strip())
        if not p:
            return default_output or str(Path.home())
        return os.path.abspath(p)

    def _render(p: str):
        p = _norm(p)
        state["path"] = p
        path_input.value = p
        status.text = ""
        dir_list.clear()
        parent = os.path.dirname(p)
        if parent != p:
            with dir_list:
                ui.button(f"⬆️  {parent}", on_click=lambda pp=parent: _render(pp)) \
                    .props("flat align=left dense").classes("w-full justify-start")
        subs = _list_subdirs(p)
        if not subs and parent == p:
            status.text = "(empty folder)"
        for entry in subs:
            full = os.path.join(p, entry)
            with dir_list:
                ui.button(f"📁  {entry}", on_click=lambda f=full: _render(f)) \
                    .props("flat align=left dense").classes("w-full justify-start")

    def _go():
        p = _norm(path_input.value)
        if os.path.isdir(p):
            _render(p)
        else:
            status.text = f"Not a directory: {p}"

    def _select():
        result["path"] = state["path"]
        dialog.close()
        done.set()

    def _cancel():
        dialog.close()
        done.set()

    with ui.dialog() as dialog, ui.card().classes("w-[600px] max-w-[92vw] drocat-card"):
        ui.label(title).classes("text-h6")
        path_input = ui.input("Path", value="").classes("w-full drocat-input")
        dir_list = ui.column().classes("w-full max-h-72 overflow-auto")
        status = ui.label("").classes("text-caption drocat-muted")
        with ui.row().classes("w-full gap-2 justify-end"):
            ui.button("Go", on_click=_go).props("outline dense")
            ui.button("Select This Folder", on_click=_select).props("color=primary dense")
            ui.button("Cancel", on_click=_cancel).props("flat dense")

    _render(initial)
    dialog.open()
    await done.wait()
    return result["path"]


def dir_input(
    label: str = "Output Directory",
    default: Optional[str] = None,
    hint: str = "Where results will be saved. Click folder icon to browse. Changes are saved permanently as the default.",
) -> ui.input:
    """Create a directory input with an in-browser browse button.

    Changing the value (typed + blur, or picked with the browse button)
    persists it permanently as the default output directory and synchronizes
    every other output-directory field in the UI.
    """
    default_path = default or get_default_output_dir()

    inp = ui.input(
        label=label,
        value=default_path,
    ).classes("w-full drocat-input drocat-output-dir").tooltip(hint)

    # All output-directory fields stay in sync: changing one persists the
    # value and updates the others (nicegui propagates the value to the DOM).
    _OUTPUT_DIR_INPUTS.append(inp)

    def _persist_output_dir():
        saved, effective = set_default_output_dir(inp.value or "", create=False)
        if saved and effective:
            sync_output_dir_fields(inp, effective)

    with inp.add_slot("append"):
        async def browse(*args):
            selected = await dir_browser_dialog(
                title=f"Select {label}",
                initial=inp.value or str(PROJECT_ROOT),
                default_output=inp.value or get_default_output_dir(),
            )
            if selected:
                inp.value = selected
                _persist_output_dir()
        ui.button(icon="folder_open", on_click=browse).props("flat dense").tooltip("Browse")

    inp.on("blur", _persist_output_dir)

    return inp


def sync_output_dir_fields(source, value: str) -> None:
    """Update every output-directory field except *source* to *value*.

    Backend values are set directly; the client DOM is updated explicitly
    because Quasar inputs can keep stale native values when the value prop
    changes (especially in inactive tab panels). The input event keeps the
    client model in sync with the backend.
    """
    for other in _OUTPUT_DIR_INPUTS:
        if other is not source and getattr(other, "value", None) != value:
            try:
                other.value = value
            except Exception:
                # Lazy tab panels keep their elements unmounted until first
                # activation: nicegui updates the server-side props but then
                # raises while building the change event (no client yet), so
                # the value is already correct when the panel mounts.
                pass
    try:
        ui.run_javascript(
            f"""
            document.querySelectorAll('.drocat-output-dir input').forEach(inp => {{
                if (inp.value !== {json.dumps(value)}) {{
                    inp.value = {json.dumps(value)};
                    inp.dispatchEvent(new Event('input', {{ bubbles: true }}));
                }}
            }});
            """
        )
    except Exception:
        # No client connected (e.g. unit tests): backend values are enough.
        pass


# =============================================================================
# Chip List Input
# =============================================================================

def chip_list_input(
    label: str = "Items",
    placeholder: str = "Type and press Enter to add",
    hint: str = "Enter items one by one. Press Enter or comma to add a chip. Click X to remove.",
    initial: Optional[List[str]] = None,
) -> ui.select:
    """Create a chip-based input where each entry becomes a removable chip."""
    sel = ui.select(
        options=initial or [],
        value=initial or [],
        label=label,
        multiple=True,
    ).classes("w-full drocat-select drocat-chip-input").props('use-chips use-input new-value-mode="add-unique"').tooltip(hint)
    sel.props('input-debounce="0"')
    return sel


def multi_select_input(
    label: str,
    options: List[str],
    default: Optional[List[str]] = None,
    hint: str = "",
    with_search: bool = True,
) -> ui.select:
    """Create a multi-select dropdown from predefined options."""
    sel = ui.select(
        options=options,
        value=default or [],
        label=label,
        multiple=True,
    ).classes("w-full drocat-select")
    if with_search:
        sel.props("use-input use-chips")
    if hint:
        sel.tooltip(hint)
    return sel


def combo_input(
    label: str,
    options: List[str],
    default: Optional[str] = None,
    hint: str = "",
) -> ui.select:
    """Create a combo box: dropdown with predefined options + free text."""
    sel = ui.select(
        options=options,
        value=default or (options[0] if options else ""),
        label=label,
    ).classes("w-full drocat-select").props('use-input new-value-mode="add-unique" fill-input hide-selected')
    if hint:
        sel.tooltip(hint)
    return sel


# =============================================================================
# Utility Functions
# =============================================================================

def parse_neuron_list(text: str) -> List:
    """Parse neuron input text into a list of neuron names."""
    if not text or not text.strip():
        return []
    neurons = []
    for part in text.replace("\n", ",").split(","):
        part = part.strip()
        if part:
            try:
                neurons.append(int(part))
            except ValueError:
                neurons.append(part)
    return neurons


def apply_filter_mode(neurons: List, mode: str) -> List:
    """Convert an input mode into the backend's query representation.

    Exact (including a missing/unknown mode) deliberately stays a bare value;
    :func:`statvis._process_single_neuron` treats bare values as strict,
    case-sensitive matches.  The other UI modes are made explicit as regex
    patterns so the backend cannot silently reinterpret an exact query as a
    prefix or substring search.
    """
    normalized = str(mode or "exact").strip().lower()
    normalized = normalized.replace(" ", "")
    aliases = {
        "startwith": "startswith",
        "startswith": "startswith",
        "endwith": "endswith",
        "endswith": "endswith",
        "contains": "contains",
        "regex": "regex",
        "exact": "exact",
    }
    normalized = aliases.get(normalized, "exact")
    if normalized == "exact":
        return list(neurons or [])
    result = []
    for n in neurons:
        # Starts/contains/ends-with are literal text modes.  Escape user
        # punctuation so their backend representation cannot accidentally
        # become a different regex query; the explicit Regex mode remains
        # available for callers who intentionally need regular expressions.
        s = re.escape(str(n))
        if normalized == "startswith":
            result.append(f"{s}.*")
        elif normalized == "contains":
            result.append(f".*{s}.*")
        elif normalized == "endswith":
            result.append(f".*{s}$")
        else:
            result.append(s)
    return result


def open_folder(path: str):
    """Open a folder in the system file manager (shared impl in ui.runner)."""
    from ..runner import open_folder as _open_folder
    _open_folder(path)


# =============================================================================
# Dataset Status Card (lazy - only fetches on button click)
# =============================================================================

def dataset_status_card() -> ui.card:
    """
    Create a card showing dataset availability status.
    Does NOT auto-fetch on page load. Click Refresh to fetch server metadata
    and re-check local converted tables.
    """
    from ..dataset_service import get_dataset_service
    import threading

    service = get_dataset_service()
    state = {"results": None, "error": None, "done": False, "running": False}

    with ui.card().classes("w-full drocat-card") as card:
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Dataset Availability").classes("drocat-card-title")
            refresh_btn = ui.button("Refresh", icon="refresh", color="primary").props("flat dense").tooltip(
                "Check local converted tables and server availability.\n"
                "NeuPrint server status requires a valid token; FlyWire uses local files."
            )

        ui.separator()
        status_container = ui.column().classes("w-full gap-1")

        with status_container:
            ui.label(
                "Click Refresh to query NeuPrint/Codex; local dataset folders are shown immediately."
            ).classes("text-caption drocat-muted")

        def render_results(results):
            status_container.clear()
            with status_container:
                if not results:
                    ui.label(
                        "No datasets found. NeuPrint status needs a token; "
                        "FlyWire status needs the converted local tables."
                    ).classes("text-caption drocat-warn")
                    return

                def count_badge(info, color):
                    text = f"{info.neuron_count:,}" if info.neuron_count else "n/a"
                    ui.badge(text, color=color).props("outline")

                for name, info in results.items():
                    is_flywire = name.startswith("flywire_")
                    src_badge_text = "FlyWire" if is_flywire else "NeuPrint"
                    src_badge_color = "purple" if is_flywire else "blue"

                    with ui.row().classes("items-center gap-2 w-full drocat-status-row"):
                        if info.local_prepared:
                            ui.icon("check_circle", color="green")
                            ui.label(info.display_name or name).classes("font-medium flex-grow")
                            ui.badge(src_badge_text, color=src_badge_color).props("outline")
                            ui.badge("local", color="green").props("outline")
                            count_badge(info, "green")
                        elif info.available:
                            ui.icon("cloud_done", color="blue")
                            ui.label(info.display_name or name).classes("font-medium flex-grow")
                            ui.badge(src_badge_text, color=src_badge_color).props("outline")
                            ui.badge("server", color="blue").props("outline")
                            count_badge(info, "blue")
                        elif info.local_cache:
                            ui.icon("cached", color="orange")
                            ui.label(info.display_name or name).classes("font-medium flex-grow")
                            ui.badge(src_badge_text, color=src_badge_color).props("outline")
                            ui.badge("cached", color="orange").props("outline")
                            count_badge(info, "orange")
                        else:
                            ui.icon("cloud_off", color="grey")
                            ui.label(info.display_name or name).classes("font-medium flex-grow drocat-muted")
                            ui.badge(src_badge_text, color=src_badge_color).props("outline")
                            ui.badge("not ready", color="grey").props("outline")
                            count_badge(info, "grey")

        # Local status is useful even when the user has not configured a
        # NeuPrint token or is working offline.  Avoid any network call here.
        local_results = {
            info.name: info
            for info in service.get_local_datasets()
            if info.local_prepared or info.local_cache
        }
        if local_results:
            render_results(local_results)

        def render_error(msg):
            status_container.clear()
            with status_container:
                ui.label(f"Error: {msg}").classes("text-caption drocat-err")

        def do_refresh():
            try:
                results = service.refresh_availability()
                state["results"] = results
                state["error"] = None
            except Exception as e:
                state["results"] = None
                state["error"] = str(e)
            finally:
                state["done"] = True
                state["running"] = False

        def on_refresh_click():
            if state["running"]:
                return
            state.update(running=True, done=False, results=None, error=None)
            refresh_btn.disable()
            status_container.clear()
            with status_container:
                ui.spinner("dots", size="sm")
                ui.label(
                    "Checking local files and NeuPrint/Codex availability... (may take 10-30s)"
                ).classes("text-caption drocat-muted")
            thread = threading.Thread(target=do_refresh, daemon=True)
            thread.start()

        refresh_btn.on_click(on_refresh_click)

        def poll_results():
            if not state["done"]:
                return
            state["done"] = False
            if state["results"] is not None:
                render_results(state["results"])
            elif state["error"] is not None:
                render_error(state["error"])
            refresh_btn.enable()

        ui.timer(1.0, poll_results)

    return card
