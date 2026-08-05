"""
Output Panel Component

The results "contact sheet": status pill, run/cancel actions, progress,
live log console and output files rendered as selectable cards.
"""

from nicegui import ui
from typing import List, Optional, Dict
from pathlib import Path
from collections import defaultdict
import re

from ..runner import open_folder, open_file


# Label of a tqdm-style progress bar, e.g. "Building target profiles:" from
# "Building target profiles:  45%|████▍       | 1328/2972 [00:05<00:06, ...]".
# Used to refresh the same bar in place instead of appending a new line.
_PROGRESS_NAME_RE = re.compile(r"^\s*([^%]*?)\s*\d+%\|")


def _progress_bar_name(message: str) -> str:
    """Extract the progress-bar label from a tqdm-style line ('' if none)."""
    match = _PROGRESS_NAME_RE.match(message)
    return match.group(1).rstrip() if match else ""


# File type categories for organizing output
FILE_CATEGORIES = {
    "Visualizations (HTML)": [".html", ".htm"],
    "Data Tables (CSV)": [".csv", ".tsv"],
    "Data Tables (Excel)": [".xlsx", ".xls"],
    "Images": [".png", ".jpg", ".jpeg", ".svg", ".pdf"],
    "Configuration": [".json", ".txt", ".yaml", ".yml"],
    "Other": [],
}

_STATUS_COLORS = {
    "idle": "grey-5",
    "running": "blue",
    "success": "green",
    "failed": "red",
    "completed": "green",
}


def _categorize_file(filename: str) -> str:
    """Determine the category for a file based on extension."""
    ext = Path(filename).suffix.lower()
    for category, extensions in FILE_CATEGORIES.items():
        if ext in extensions:
            return category
    return "Other"


class OutputPanel:
    """Reusable panel for displaying logs and output files."""

    def __init__(self, title: str = "Output"):
        self.title = title
        self._dom_id = f"drocat-results-{id(self)}"
        self.log_area: Optional[ui.log] = None
        self.files_container = None
        self.status_label: Optional[ui.badge] = None
        self.progress_bar = None
        self.progress_row = None
        self.run_button: Optional[ui.button] = None
        self.cancel_button: Optional[ui.button] = None
        self._files: List[dict] = []
        self._last_is_progress = False
        self._last_progress_name: Optional[str] = None
        # Streaming: polls the run's output folder while the run is active so
        # files appear as soon as they are written. Category expansions are
        # tracked so in-place refreshes keep the user's open/closed state.
        self._poll_timer = None
        self._file_categories: Dict[str, ui.expansion] = {}

    def create(
        self,
        run_label: str = "Run",
        run_icon: str = "play_arrow",
    ):
        """Create the output panel UI (run/cancel actions + status + log + files)."""
        with ui.card().classes("w-full drocat-card drocat-results-card gap-0").props(
            f'id="{self._dom_id}"'
        ):
            # Header: title + status pill
            with ui.row().classes("w-full items-center justify-between drocat-results-head"):
                with ui.row().classes("items-center gap-2"):
                    with ui.element("div").classes("drocat-results-mark"):
                        ui.icon("receipt_long").classes("text-white")
                    ui.label(self.title).classes("drocat-card-title")
                self.status_label = ui.badge("Idle", color="grey-5").props("outline")

            # Action bar: Run / Cancel
            with ui.row().classes("w-full items-center gap-2 drocat-action-bar"):
                self.run_button = ui.button(
                    run_label,
                    icon=run_icon,
                    color="primary",
                ).classes("drocat-run-btn")
                self.cancel_button = ui.button(
                    "Cancel",
                    icon="stop",
                    color="negative",
                ).classes("drocat-cancel-btn")
                self.cancel_button.disable()

            # Progress row (visible while running)
            self.progress_row = ui.row().classes("w-full items-center gap-2 drocat-progress-row")
            with self.progress_row:
                self.progress_bar = ui.linear_progress(value=0, show_value=False).classes("w-full")
                self.progress_row.set_visibility(False)

            # Log console
            ui.label("Execution Log").classes("drocat-mini-label")
            # Keep the log as a plain, light console (pre-redesign styling that
            # streams reliably); only the max height is capped.
            with ui.element("div").classes("w-full").style(
                "resize: vertical; overflow: hidden; min-height: 100px; max-height: 600px;"
            ):
                self.log_area = ui.log(max_lines=500).classes(
                    "w-full font-mono text-xs"
                ).style("height: 200px;")

            ui.separator()
            ui.label("Output Files").classes("drocat-mini-label")
            self.files_container = ui.column().classes("w-full gap-2")
            with self.files_container:
                ui.label("No output files yet.").classes("drocat-empty")

    def log(self, message: str, level: str = "stdout"):
        """Add a log message to the panel."""
        if not self.log_area:
            return
        try:
            # Drop trailing whitespace (tqdm clears lines with spaces) and
            # whitespace-only residue so the log stays tidy.
            message = message.rstrip()
            if not message:
                return
            # Progress lines (tqdm-style \r updates) refresh the previous line
            # of the SAME progress bar in place, so long-running functions
            # show a live-updating status instead of flooding the log. A new
            # bar (different label) starts a fresh line.
            if level == "progress":
                children = self.log_area.default_slot.children
                name = _progress_bar_name(message)
                if (
                    self._last_is_progress
                    and children
                    and name == self._last_progress_name
                ):
                    last = children[-1]
                    if hasattr(last, "set_text"):
                        last.set_text(message)
                        last.update()
                        return
                self._last_is_progress = True
                self._last_progress_name = name
                self.log_area.push(message)
                self.log_area.update()
                return
            self._last_is_progress = False
            self._last_progress_name = None
            prefix_map = {
                "stdout": "",
                "stderr": "[WARN] ",
                "error": "[ERROR] ",
                "success": "[OK] ",
                "system": "[SYS] ",
            }
            self.log_area.push(prefix_map.get(level, "") + message)
            # Force an immediate flush to the browser
            self.log_area.update()
        except Exception:
            # The page may have been closed or navigated away while the run
            # was active; silently drop further log lines instead of crashing
            # the run handler.
            pass

    def set_status(self, status: str, color: str = "grey"):
        """Update the status pill."""
        if self.status_label:
            self.status_label.text = status
            color = _STATUS_COLORS.get(status.lower(), color)
            self.status_label.props(f"color={color}")

    def set_running(self, running: bool):
        """Update UI for running state."""
        if running:
            self.set_status("Running", "blue")
            if self.run_button:
                self.run_button.disable()
            if self.cancel_button:
                self.cancel_button.enable()
            if self.progress_row:
                self.progress_row.set_visibility(True)
            if self.progress_bar:
                self.progress_bar.props("indeterminate")
            # Make sure the results panel (with the log) is visible
            ui.run_javascript(
                f"const card = document.getElementById('{self._dom_id}');"
                "if (card) card.scrollIntoView({behavior:'smooth', block:'nearest'});"
            )
        else:
            self._stop_file_streaming()
            if self.run_button:
                self.run_button.enable()
            if self.cancel_button:
                self.cancel_button.disable()
            if self.progress_row:
                self.progress_row.set_visibility(False)
            if self.progress_bar:
                self.progress_bar.props(":indeterminate='false'")
                self.progress_bar.value = 1.0 if self._files else 0.0

    def _stop_file_streaming(self):
        """Stop the output-folder polling timer (run finished or cancelled)."""
        if self._poll_timer is not None:
            self._poll_timer.cancel()
            self._poll_timer = None

    def _poll_output_files(self, runner, output_dir: str):
        """Refresh the files panel with files created so far (streaming).

        Polls the current run's output folder while the subprocess is active;
        newly written files show up within one poll interval instead of only
        after the run completes. The panel refresh is skipped until the run
        folder is known, so files from older runs are never shown.
        """
        try:
            if not runner.is_running:
                return
            run_folder = runner._resolve_scan_dir(output_dir)
            if not run_folder:
                return
            files = runner._scan_output_files(run_folder)
            if files:
                self.show_files(files, run_folder)
        except Exception:
            # Page may have been closed or navigated away mid-run; stop
            # polling instead of crashing the run handler.
            self._stop_file_streaming()

    async def run(
        self,
        runner,
        tool_name: str,
        constructor_params: dict,
        method_name: str,
        method_params: Optional[dict] = None,
        output_dir: Optional[str] = None,
    ) -> dict:
        """
        Run a tool through the UI runner and always surface errors in the log.

        While the run is active, the output folder is polled so files appear
        in the panel as soon as they are created (streaming).

        Any exception is written into the execution log (instead of silently
        failing the handler and leaving an empty log with a stuck Run button).
        """
        try:
            # Stream output files during the run: poll every 1.5s.
            self._stop_file_streaming()
            if output_dir:
                self._poll_timer = ui.timer(
                    1.5, lambda: self._poll_output_files(runner, output_dir)
                )
            return await runner.run(
                tool_name,
                constructor_params,
                method_name,
                method_params=method_params,
                log_callback=self.log,
                output_dir=output_dir,
            )
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.log(
                f"[DROCAT] Unexpected UI error: {type(exc).__name__}: {exc}", "error"
            )
            self.log(traceback.format_exc().rstrip(), "error")
            return {"returncode": -1, "files": [], "duration": 0, "cancelled": False}
        finally:
            self._stop_file_streaming()

    def show_files(self, files: List[dict], output_dir: Optional[str] = None):
        """Display output files organized by category as a card grid.

        The panel may be refreshed repeatedly while a run is active
        (streaming); the rebuild preserves which category expansions the user
        has open, so newly created files appear without collapsing anything.
        """
        self._files = files

        if not self.files_container:
            return

        # Remember which category expansions are open so an in-place refresh
        # during streaming does not collapse them.
        expanded = {
            category: expansion.value
            for category, expansion in self._file_categories.items()
        }

        self.files_container.clear()
        self._file_categories = {}

        with self.files_container:
            if not files:
                ui.label("No output files generated.").classes("drocat-empty")
                return

            if output_dir:
                with ui.row().classes("items-center gap-2"):
                    ui.button(
                        "Open Output Folder",
                        icon="folder_open",
                        on_click=lambda: open_folder(output_dir),
                    ).props("flat dense color=primary")
                    ui.label(str(Path(output_dir))).classes("text-caption drocat-muted drocat-truncate")

            grouped = defaultdict(list)
            for f in files:
                grouped[_categorize_file(f["name"])].append(f)

            for category in FILE_CATEGORIES.keys():
                if category not in grouped:
                    continue
                cat_files = grouped[category]
                with ui.expansion(
                    f"{category}  ({len(cat_files)})", icon="folder"
                ).classes("w-full drocat-expansion") as expansion:
                    self._file_categories[category] = expansion
                    with ui.element("div").classes("drocat-file-list"):
                        for f in cat_files:
                            with ui.row().classes(
                                "drocat-file-row items-center gap-2"
                            ).on("click", lambda path=f["path"]: open_file(path)):
                                ui.icon("insert_drive_file").classes("drocat-file-icon")
                                ui.label(f["name"]).classes("drocat-file-name flex-grow")
                                ui.label(self._format_size(f.get("size", 0))).classes(
                                    "text-caption drocat-muted drocat-file-size"
                                )
                                ui.button(
                                    icon="open_in_new",
                                    on_click=lambda path=f["path"]: open_file(path),
                                ).props("flat dense round").classes("drocat-file-open")
                if expanded.get(category):
                    expansion.value = True

    def _format_size(self, size: int) -> str:
        """Format file size in human-readable format."""
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        else:
            return f"{size / (1024 * 1024):.1f} MB"

    def clear(self):
        """Clear the log and files."""
        self._last_is_progress = False
        self._last_progress_name = None
        self._stop_file_streaming()
        self._file_categories = {}
        if self.log_area:
            self.log_area.clear()
        self._files = []
        if self.files_container:
            self.files_container.clear()
            with self.files_container:
                ui.label("No output files yet.").classes("drocat-empty")
