#!/usr/bin/env python
"""
DROCAT UI - Integrated Web Interface for Connectome Analysis

Launch with: python ui/app.py
Access at: http://127.0.0.1:8080

Design language follows the Photo Selector "gallery" reference:
light canvas, white surfaces, cobalt accent, segmented navigation,
focus-panel + contact-sheet workspace.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nicegui import ui, app

# Serve the local documentation so relative links (docs/ui_guides/...)
# open the local instruction files instead of GitHub.
app.add_static_files("/docs", PROJECT_ROOT / "docs")

from ui.config import APP_TITLE, APP_VERSION, APP_PORT, APP_HOST
from ui.tabs import (
    create_find_path_tab,
    create_find_direct_tab,
    create_connectivity_profiling_tab,
    create_find_homologs_tab,
    create_inter_dataset_tab,
    create_nb_find_lines_tab,
    create_nb_find_neuron_tab,
    create_nb_colabel_tab,
    create_visualization_tab,
    create_settings_tab,
)


DROCAT_CSS = """
:root {
    --drocat-canvas: #f7f8fa;
    --drocat-surface: #ffffff;
    --drocat-soft: #f1f3f6;
    --drocat-navy: #0b1f3a;
    --drocat-muted: #667085;
    --drocat-faint: #98a2b3;
    --drocat-line: #e2e7ee;
    --drocat-line-strong: #cfd7e3;
    --drocat-cobalt: #145cff;
    --drocat-cobalt-soft: #eaf0ff;
    --drocat-ok: #16a34a;
    --drocat-warn: #d97706;
    --drocat-err: #dc2626;
    --drocat-radius-sm: 10px;
    --drocat-radius-md: 16px;
    --drocat-radius-lg: 24px;
    --drocat-shadow: 0 16px 40px rgba(11, 31, 58, .08);
}

html, body {
    background:
        radial-gradient(circle at 16% 0%, rgba(20, 92, 255, .04), transparent 32%),
        var(--drocat-canvas) !important;
    color: var(--drocat-navy);
    font-family: Inter, "Avenir Next", "Segoe UI", Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
}

/* ---------- Header ---------- */
.drocat-header {
    background: rgba(255, 255, 255, .86) !important;
    backdrop-filter: blur(14px);
    border-bottom: 1px solid var(--drocat-line);
    box-shadow: none !important;
    min-height: 76px !important;
    padding: 0 28px !important;
}
.drocat-brand-mark {
    width: 42px; height: 42px; border-radius: 13px;
    display: grid; place-items: center;
    background: var(--drocat-cobalt);
    color: #fff;
    box-shadow: 0 8px 18px rgba(20, 92, 255, .24);
}
.drocat-brand-title { font-size: 19px; font-weight: 700; letter-spacing: -.02em; color: var(--drocat-navy); }
.drocat-brand-sub { font-size: 12.5px; color: var(--drocat-muted); }
.drocat-version-pill {
    border: 1px solid var(--drocat-line-strong); border-radius: 999px;
    padding: 4px 10px; font-size: 11px; font-weight: 650; color: var(--drocat-muted);
    background: var(--drocat-surface);
}
.drocat-header-link {
    color: var(--drocat-muted) !important; font-size: 12.5px; font-weight: 600;
    text-decoration: none !important;
}
.drocat-header-link:hover { color: var(--drocat-cobalt) !important; }
.drocat-doc-link {
    color: var(--drocat-cobalt) !important;
    font-size: 11px;
    font-weight: 650;
    text-decoration: none !important;
    white-space: nowrap;
}
.drocat-doc-link:hover { text-decoration: underline !important; }

/* ---------- Segmented navigation ---------- */
.drocat-tabs {
    display: flex; gap: 4px; padding: 5px;
    background: var(--drocat-soft);
    border: 1px solid var(--drocat-line);
    border-radius: 14px;
    box-shadow: inset 0 1px 3px rgba(11, 31, 58, .04);
    overflow-x: auto;
    scrollbar-width: none;
}
.drocat-tabs .q-tab {
    min-height: 42px !important; border-radius: 10px;
    color: var(--drocat-navy); font-size: 13px; font-weight: 600;
    transition: background .16s ease, color .16s ease, box-shadow .16s ease;
    flex: 0 0 auto;
}
.drocat-tabs .q-tab:hover { background: #e8edf6; }
.drocat-tabs .q-tab--active {
    color: var(--drocat-cobalt) !important;
    background: var(--drocat-surface) !important;
    box-shadow: 0 2px 6px rgba(11, 31, 58, .10);
}
.drocat-tabs .q-tab__icon { font-size: 19px; }
.drocat-tabs .q-tab__label { font-size: 13px; font-weight: 600; }
.drocat-tabs .drocat-tab-group {
    display: flex;
    align-items: center;
    padding: 0 8px 0 14px;
    margin: 8px 2px;
    border-left: 1px solid var(--drocat-line-strong);
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--drocat-faint);
    user-select: none;
    white-space: nowrap;
}
.drocat-tabs .drocat-tab-group:first-child { border-left: 0; padding-left: 6px; }
.drocat-tabs .drocat-tab-group-nb { color: #7c3aed; }
.drocat-nb-tab .q-tab__label::after {
    content: "NB";
    margin-left: 6px;
    padding: 1px 5px;
    border-radius: 999px;
    background: #7c3aed;
    color: #fff;
    font-size: 9px;
    font-weight: 800;
    letter-spacing: .06em;
}
.drocat-tabs .q-tab--active.drocat-nb-tab {
    color: #7c3aed !important;
}
.drocat-tabs .drocat-nb-tab:hover { background: #f5f0ff !important; }

/* Panel tag badge (NeuronBridge etc.) */
.drocat-tag-badge {
    font-size: 10px !important;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
}

/* ---------- Page workspace ---------- */
.drocat-page { max-width: 1500px; margin: 0 auto; padding: 24px 28px 40px; }
.drocat-page-head { padding-bottom: 18px; border-bottom: 1px solid var(--drocat-line); margin-bottom: 20px; }
.drocat-page-mark {
    width: 46px; height: 46px; border-radius: 14px; display: grid; place-items: center;
    background: linear-gradient(135deg, var(--drocat-cobalt), #0d3bb8);
    box-shadow: 0 8px 20px rgba(20, 92, 255, .22);
}
.drocat-page-mark .q-icon { font-size: 24px; }
.drocat-page-title { font-size: 22px; font-weight: 700; letter-spacing: -.02em; color: var(--drocat-navy); }
.drocat-page-sub { font-size: 13px; color: var(--drocat-muted); }

.drocat-workspace {
    display: grid !important;
    grid-template-columns: minmax(0, 1.45fr) minmax(430px, 1fr);
    gap: 24px;
    align-items: start;
}
.drocat-form { min-width: 0; }
.drocat-results { min-width: 0; }
.drocat-results-card { position: sticky; top: 20px; }

@media (max-width: 1280px) {
    .drocat-workspace { grid-template-columns: 1fr; }
    .drocat-results-card { position: static; }
}

/* ---------- Cards ---------- */
.q-card.drocat-card {
    background: var(--drocat-surface) !important;
    border: 1px solid var(--drocat-line) !important;
    border-radius: var(--drocat-radius-md) !important;
    box-shadow: 0 2px 10px rgba(11, 31, 58, .05) !important;
    padding: 18px 20px;
}
.drocat-card-title { font-size: 15px; font-weight: 680; letter-spacing: -.01em; color: var(--drocat-navy); }
.drocat-section-head { margin: 2px 0 10px; }
.drocat-section-icon {
    width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center;
    background: var(--drocat-cobalt-soft);
}
.drocat-section-icon .q-icon { font-size: 17px; }
.drocat-section-title { font-size: 14px; font-weight: 650; color: var(--drocat-navy); }

/* ---------- Form controls ---------- */
.q-field--outlined .q-field__control {
    background: var(--drocat-surface) !important;
    border-radius: var(--drocat-radius-sm) !important;
}
.q-field--outlined .q-field__control::before { border-color: var(--drocat-line-strong) !important; }
.q-field--outlined .q-field__control:hover::before { border-color: var(--drocat-cobalt) !important; }
.q-field--focused .q-field__control::before { border-color: var(--drocat-cobalt) !important; }
.q-field__label { color: var(--drocat-muted) !important; }
.q-field--focused .q-field__label { color: var(--drocat-cobalt) !important; }
.q-field__native, .q-field__input { color: var(--drocat-navy) !important; }
.q-field--dark .q-field__native { color: #fff !important; }
.drocat-param-grid { gap: 14px 16px !important; }

/* ---------- Buttons ---------- */
.q-btn { border-radius: var(--drocat-radius-sm); font-weight: 650; }
.q-btn--flat { color: var(--drocat-muted); }
.q-btn--flat:hover { background: var(--drocat-soft) !important; }
.q-btn--unelevated.bg-primary { background: var(--drocat-cobalt) !important; box-shadow: 0 6px 14px rgba(20, 92, 255, .20); }
.q-btn--unelevated.bg-negative { background: #fff !important; color: var(--drocat-err) !important; border: 1px solid #fecaca; }
.drocat-run-btn { min-width: 150px; height: 44px !important; }
.drocat-cancel-btn { height: 44px !important; }
.drocat-upload-trigger {
    width: 34px !important;
    height: 34px !important;
    margin-bottom: 4px;
    border: 1px solid var(--drocat-line-strong) !important;
    background: var(--drocat-surface) !important;
    color: var(--drocat-muted) !important;
}
.drocat-upload-trigger:hover { border-color: var(--drocat-cobalt) !important; color: var(--drocat-cobalt) !important; }
.drocat-upload-trigger .q-icon { font-size: 17px; }
.q-menu .q-uploader { border: 1px solid var(--drocat-line) !important; border-radius: 12px !important; }
.drocat-clear-btn { color: var(--drocat-faint); font-size: 12px; }
.drocat-clear-btn:hover { color: var(--drocat-err); }

/* ---------- Palette picker (color previews) ---------- */
.drocat-palette-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 8px;
    overflow-y: auto;
    padding: 2px;
}
.drocat-palette-card {
    border: 1px solid var(--drocat-line);
    border-radius: 10px;
    padding: 6px;
    cursor: pointer;
    background: var(--drocat-surface);
    transition: border-color .15s ease, box-shadow .15s ease;
}
.drocat-palette-card:hover { border-color: var(--drocat-line-strong); }
.drocat-palette-card.selected {
    border-color: var(--drocat-cobalt);
    box-shadow: 0 0 0 2px var(--drocat-cobalt-soft);
}
.drocat-palette-swatches {
    overflow: hidden;
    border-radius: 6px;
    border: 1px solid rgba(11, 31, 58, .08);
}
.drocat-palette-name {
    font-size: 10.5px;
    font-weight: 650;
    color: var(--drocat-navy);
    margin-top: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.drocat-palette-preview {
    min-height: 26px;
    padding: 4px 8px;
    background: var(--drocat-soft);
    border-radius: 10px;
}
.drocat-palette-expansion {
    border: 1px solid var(--drocat-line);
    border-radius: 12px;
    overflow: hidden;
}
.drocat-swatch {
    padding: 3px;
    border-radius: 50%;
    border: 2px solid transparent;
    cursor: pointer;
    transition: border-color .15s ease, transform .15s ease;
}
.drocat-swatch:hover { transform: scale(1.08); }
.drocat-swatch.selected { border-color: var(--drocat-cobalt); }
.drocat-palette-strip {
    border-radius: 6px;
    border: 1px solid rgba(11, 31, 58, .10);
    min-height: 18px;
    cursor: pointer;
}
.drocat-custom-color-row {
    padding: 4px 8px;
    border: 1px solid var(--drocat-line);
    border-radius: 10px;
    background: var(--drocat-surface);
}
.drocat-custom-color-row:hover { background: var(--drocat-soft); }

/* ---------- Results panel ---------- */
.drocat-results-head { padding-bottom: 12px; }
.drocat-results-mark {
    width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center;
    background: var(--drocat-navy);
}
.drocat-results-mark .q-icon { font-size: 16px; }
.drocat-action-bar { padding: 12px 0 10px; border-top: 1px solid var(--drocat-line); }
.drocat-progress-row { padding-bottom: 4px; }
.drocat-mini-label { font-size: 11px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--drocat-faint); margin: 8px 0 4px; }
.drocat-empty { color: var(--drocat-faint); font-style: italic; font-size: 12.5px; }
.drocat-muted { color: var(--drocat-muted); }
.drocat-ok { color: var(--drocat-ok); }
.drocat-err { color: var(--drocat-err); }
.drocat-warn { color: var(--drocat-warn); }
.drocat-truncate { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 380px; }

/* ---------- Output file list (file-manager style) ---------- */
.drocat-file-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 4px 0;
}
.drocat-file-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 10px;
    border: 1px solid var(--drocat-line);
    border-radius: 10px;
    background: var(--drocat-surface);
    cursor: pointer;
    min-width: 0;
    transition: border-color .15s ease, background .15s ease;
}
.drocat-file-row:hover {
    border-color: var(--drocat-cobalt);
    background: var(--drocat-soft);
}
.drocat-file-icon { color: var(--drocat-cobalt); font-size: 18px; flex: none; }
.drocat-file-name {
    font-size: 12.5px; font-weight: 600; color: var(--drocat-navy);
    word-break: break-all; line-height: 1.3; min-width: 0;
}
.drocat-file-size { flex: none; font-variant-numeric: tabular-nums; }
.drocat-file-open { color: var(--drocat-muted); flex: none; }
.drocat-file-row:hover .drocat-file-open { color: var(--drocat-cobalt); }
.drocat-expansion { border: 1px solid var(--drocat-line); border-radius: 12px; margin-bottom: 8px; overflow: hidden; }
.drocat-expansion .q-expansion-item__container { background: var(--drocat-surface); }

/* ---------- Status rows ---------- */
.drocat-status-row { padding: 6px 8px; border-radius: 10px; }
.drocat-status-row:hover { background: var(--drocat-soft); }

/* ---------- Footer ---------- */
.drocat-footer {
    background: rgba(255, 255, 255, .86) !important;
    backdrop-filter: blur(10px);
    border-top: 1px solid var(--drocat-line);
    min-height: 44px !important;
}
.drocat-footer .q-item__label { color: var(--drocat-faint); font-size: 12px; }

/* ---------- Misc ---------- */
.q-separator { background: var(--drocat-line) !important; }
.q-badge { border-radius: 999px; font-weight: 650; }
.q-expansion-item__toggle-icon { color: var(--drocat-muted); }
.q-chip { border-radius: 8px; background: var(--drocat-soft) !important; color: var(--drocat-navy) !important; }
.q-spinner { color: var(--drocat-cobalt) !important; }
.nicegui-upload { border: 1px dashed var(--drocat-line-strong) !important; border-radius: 10px !important; }
"""

@ui.page("/")
def main_page():
    """Main application page with light Photo-Selector-inspired layout."""

    # Global theme
    ui.add_head_html(f"<style>{DROCAT_CSS}</style>")
    ui.add_head_html(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">'
    )

    # Header
    with ui.header(elevated=False).classes("drocat-header items-center justify-between"):
        with ui.row().classes("items-center gap-3"):
            with ui.element("div").classes("drocat-brand-mark"):
                ui.icon("hub")
            with ui.column().classes("gap-0"):
                ui.label("DROCAT").classes("drocat-brand-title")
                ui.label("Drosophila Connectome Analysis Toolkit").classes("drocat-brand-sub")
        with ui.row().classes("items-center gap-4"):
            ui.label(f"v{APP_VERSION}").classes("drocat-version-pill")
            ui.link("Documentation", "docs/ui_guides/README.html").classes("drocat-header-link")

    # Main content
    with ui.column().classes("w-full drocat-page gap-3"):
        # Segmented navigation
        with ui.tabs().classes("drocat-tabs w-full") as tabs:
            ui.label("Connectome").classes("drocat-tab-group")
            tab_pathfinding = ui.tab("Find Path", icon="route")
            tab_direct = ui.tab("Direct", icon="arrow_forward")
            tab_viz = ui.tab("Visualization", icon="view_in_ar")
            tab_comparison = ui.tab("Cross-Dataset", icon="sync_alt")
            tab_homologs = ui.tab("Homologs", icon="compare")
            tab_profiling = ui.tab("Profiling", icon="analytics")
            ui.label("NeuronBridge").classes("drocat-tab-group drocat-tab-group-nb")
            tab_find_lines = ui.tab("Find Lines", icon="biotech").classes("drocat-nb-tab")
            tab_find_neuron = ui.tab("Find Neurons", icon="search").classes("drocat-nb-tab")
            tab_colabel = ui.tab("Co-Labeling", icon="layers").classes("drocat-nb-tab")
            ui.label("System").classes("drocat-tab-group")
            tab_settings = ui.tab("Settings", icon="settings")

        with ui.tab_panels(tabs, value=tab_pathfinding).classes("w-full bg-transparent"):
            with ui.tab_panel(tab_pathfinding).classes("p-0"):
                create_find_path_tab()
            with ui.tab_panel(tab_direct).classes("p-0"):
                create_find_direct_tab()
            with ui.tab_panel(tab_viz).classes("p-0"):
                create_visualization_tab()
            with ui.tab_panel(tab_comparison).classes("p-0"):
                create_inter_dataset_tab()
            with ui.tab_panel(tab_find_lines).classes("p-0"):
                create_nb_find_lines_tab()
            with ui.tab_panel(tab_find_neuron).classes("p-0"):
                create_nb_find_neuron_tab()
            with ui.tab_panel(tab_colabel).classes("p-0"):
                create_nb_colabel_tab()
            with ui.tab_panel(tab_homologs).classes("p-0"):
                create_find_homologs_tab()
            with ui.tab_panel(tab_profiling).classes("p-0"):
                create_connectivity_profiling_tab()
            with ui.tab_panel(tab_settings).classes("p-0"):
                create_settings_tab()

    # Footer
    with ui.footer().classes("drocat-footer justify-center"):
        ui.label(
            f"DROCAT v{APP_VERSION}  ·  Local execution only  ·  Output files saved to your machine"
        ).classes("text-caption")


def main():
    """Entry point for the DROCAT UI application."""
    ui.run(
        title=APP_TITLE,
        host=APP_HOST,
        port=APP_PORT,
        reload=False,
        show=True,
        favicon="🧠",
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
