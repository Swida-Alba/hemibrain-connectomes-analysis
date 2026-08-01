"""
Shared UI components for DROCAT.

Design language follows the Photo Selector "gallery" reference:
light canvas, white surfaces, cobalt accent, rounded corners and
a focus-panel + contact-sheet workspace layout.
"""

from nicegui import ui
from typing import List, Optional, Callable, Dict
from pathlib import Path
import platform
import subprocess

from ..config import DATASETS, PROJECT_ROOT, get_default_output_dir
from ..runner import pick_directory


# =============================================================================
# Page layout helper (focus panel + results / contact sheet)
# =============================================================================

def tool_page(
    title: str,
    subtitle: str = "",
    icon: str = "science",
    tag: Optional[str] = None,
) -> tuple:
    """
    Create the two-column workspace used by every tool tab.

    Returns (form_column, results_column). The form is the "focus panel"
    where the user configures a run; the results column is the "contact
    sheet" showing status, log and output files.
    """
    with ui.column().classes("w-full drocat-page gap-0"):
        with ui.row().classes("w-full items-center gap-3 drocat-page-head"):
            with ui.element("div").classes("drocat-page-mark"):
                ui.icon(icon).classes("text-white")
            with ui.column().classes("gap-1"):
                with ui.row().classes("items-center gap-2"):
                    ui.label(title).classes("drocat-page-title")
                    if tag:
                        ui.badge(tag, color="purple").classes("drocat-tag-badge")
                if subtitle:
                    ui.label(subtitle).classes("drocat-page-sub")
        with ui.row().classes("w-full drocat-workspace items-start"):
            form_col = ui.column().classes("drocat-form gap-3")
            results_col = ui.column().classes("drocat-results gap-3")
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
    hint: str = "NeuPrint: fetched from server with token. FlyWire: requires CAVE token + local files.",
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


def neuron_list_input(
    label: str = "Neurons",
    placeholder: str = "e.g., aMe12, aMe10, DN1p (or upload CSV/XLSX)",
    hint: str = "Type neurons or upload a CSV/XLSX file (first column). Supports filter mode.",
) -> ui.element:
    """
    Create a neuron input with file upload support (CSV/XLSX).
    Returns container with .get_value() -> (filter_mode, neuron_list).
    """
    uploaded_neurons: List[str] = []

    def handle_upload(e):
        """Parse the first column of an uploaded CSV/XLSX as a neuron list."""
        nonlocal uploaded_neurons
        try:
            content = e.content.read().decode("utf-8", errors="replace")
            lines = [
                l.strip().split(",")[0].strip().strip("\"'")
                for l in content.splitlines()
                if l.strip()
            ]
            if lines and lines[0].lower() in (
                "type", "neuron", "name", "bodyid", "id", "body_id",
            ):
                lines = lines[1:]
            uploaded_neurons = [l for l in lines if l]
            upload_label.text = f"✓ {len(uploaded_neurons)} neurons loaded from {e.name}"
            upload_label.classes(replace="text-caption drocat-ok")
        except Exception:
            try:
                import pandas as pd
                import io
                e.content.seek(0)
                df = pd.read_excel(io.BytesIO(e.content.read()), header=None)
                uploaded_neurons = [str(v) for v in df.iloc[:, 0].dropna().tolist()]
                upload_label.text = f"✓ {len(uploaded_neurons)} neurons loaded from {e.name}"
                upload_label.classes(replace="text-caption drocat-ok")
            except Exception as ex2:
                uploaded_neurons = []
                upload_label.text = f"Error: {ex2}"
                upload_label.classes(replace="text-caption drocat-err")
        upload_label.set_visibility(True)
        upload_menu.close()

    with ui.column().classes("w-full gap-1") as container:
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
            ).classes("w-28 drocat-select").props("dense outlined").tooltip(
                "Exact: match exactly\nStarts with: prefix match\nContains: substring\nEnds with: suffix\nRegex: pattern"
            )

            # Compact upload: hidden inside a dropdown attached to the input row
            with ui.button(icon="upload_file").props("flat dense round").classes(
                "drocat-upload-trigger"
            ).tooltip("Upload CSV/XLSX (first column = neurons)"):
                with ui.menu() as upload_menu:
                    ui.label("Load neuron list from file").classes(
                        "text-caption drocat-muted px-3 pt-2"
                    )
                    ui.label("CSV / XLSX · first column is read as the neuron list").classes(
                        "text-caption drocat-muted px-3 pb-1"
                    )
                    ui.upload(
                        label="Choose CSV / XLSX",
                        on_upload=handle_upload,
                        auto_upload=True,
                    ).props('accept=".csv,.xlsx,.xls,.tsv" flat dense').classes("w-72")

        upload_label = ui.label("").classes("text-caption drocat-muted")
        upload_label.set_visibility(False)

    def get_value():
        typed = parse_neuron_list(textarea.value)
        return (filter_mode.value, list(uploaded_neurons) + typed)

    container.get_value = get_value
    container.textarea = textarea
    container.filter_mode = filter_mode
    container.uploaded_neurons = uploaded_neurons
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
) -> ui.select:
    """Create a dropdown select input with tooltip."""
    sel = ui.select(
        options=options,
        value=default or options[0],
        label=label,
    ).classes("w-full drocat-select")
    if hint:
        sel.tooltip(hint)
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


def dir_input(
    label: str = "Output Directory",
    default: Optional[str] = None,
    hint: str = "Where results will be saved. Click folder icon to browse.",
) -> ui.input:
    """Create a directory input with a native browse button."""
    default_path = default or get_default_output_dir()

    inp = ui.input(
        label=label,
        value=default_path,
    ).classes("w-full drocat-input").tooltip(hint)

    with inp.add_slot("append"):
        def browse(*args):
            selected = pick_directory(
                title=f"Select {label}",
                initial=inp.value or str(PROJECT_ROOT),
            )
            if selected:
                inp.value = selected
        ui.button(icon="folder_open", on_click=browse).props("flat dense").tooltip("Browse")

    return inp


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
    ).classes("w-full drocat-select").props('use-chips use-input new-value-mode="add-unique"').tooltip(hint)
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
    """Convert neuron list + filter mode into regex patterns for DROCAT scripts."""
    if mode == "exact" or not mode:
        return neurons
    result = []
    for n in neurons:
        s = str(n)
        if mode == "startswith":
            result.append(f"{s}.*")
        elif mode == "contains":
            result.append(f".*{s}.*")
        elif mode == "endswith":
            result.append(f".*{s}")
        else:
            result.append(s)
    return result


def open_folder(path: str):
    """Open a folder in the system file manager."""
    path = Path(path)
    if not path.exists():
        return
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["open", str(path)])
        elif system == "Windows":
            subprocess.run(["explorer", str(path)])
        else:
            subprocess.run(["xdg-open", str(path)])
    except Exception:
        pass


# =============================================================================
# Dataset Status Card (lazy - only fetches on button click)
# =============================================================================

def dataset_status_card() -> ui.card:
    """
    Create a card showing dataset availability status.
    Does NOT auto-fetch on page load. Click Refresh to fetch from server.
    """
    from ..dataset_service import get_dataset_service
    import threading

    service = get_dataset_service()
    state = {"results": None, "error": None, "done": False, "running": False}

    with ui.card().classes("w-full drocat-card") as card:
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Dataset Availability").classes("drocat-card-title")
            refresh_btn = ui.button("Refresh", icon="refresh", color="primary").props("flat dense").tooltip(
                "Check which datasets are available from the NeuPrint server.\n"
                "Requires a valid NeuPrint token (set in API Tokens below)."
            )

        ui.separator()
        status_container = ui.column().classes("w-full gap-1")

        with status_container:
            ui.label("Click 'Refresh' to check dataset availability from server.").classes("text-caption drocat-muted")

        def render_results(results):
            status_container.clear()
            with status_container:
                if not results:
                    ui.label("No datasets found. Check your NeuPrint token in API Tokens section.").classes("text-caption drocat-warn")
                    return
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
                            if info.neuron_count:
                                ui.badge(f"{info.neuron_count:,}", color="green").props("outline")
                        elif info.available:
                            ui.icon("cloud_done", color="blue")
                            ui.label(info.display_name or name).classes("font-medium flex-grow")
                            ui.badge(src_badge_text, color=src_badge_color).props("outline")
                            ui.badge("server", color="blue").props("outline")
                            if info.neuron_count:
                                ui.badge(f"{info.neuron_count:,}", color="blue").props("outline")
                        elif info.local_cache:
                            ui.icon("cached", color="orange")
                            ui.label(info.display_name or name).classes("font-medium flex-grow")
                            ui.badge(src_badge_text, color=src_badge_color).props("outline")
                            ui.badge("cached", color="orange").props("outline")
                        else:
                            ui.icon("cloud_off", color="grey")
                            ui.label(info.display_name or name).classes("font-medium flex-grow drocat-muted")
                            ui.badge(src_badge_text, color=src_badge_color).props("outline")
                            ui.badge("not ready", color="grey").props("outline")

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
                ui.label("Fetching from NeuPrint server... (may take 10-30s)").classes("text-caption drocat-muted")
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
