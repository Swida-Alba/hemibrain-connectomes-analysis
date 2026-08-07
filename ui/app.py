#!/usr/bin/env python
"""
DROCAT UI - Integrated Web Interface for Connectome Analysis

Launch with: python ui/app.py
Access at: http://127.0.0.1:8080

Design language follows the Photo Selector "gallery" reference:
light canvas, white surfaces, cobalt accent, segmented navigation,
focus-panel + contact-sheet workspace.
"""

import os
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
    create_flylight_tab,
    create_skeleton_tab,
    create_network_tab,
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
    line-height: 1.45;
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
    display: flex; gap: 2px; padding: 4px;
    background: var(--drocat-soft);
    border: 1px solid var(--drocat-line);
    border-radius: 14px;
    box-shadow: inset 0 1px 3px rgba(11, 31, 58, .04);
    overflow: hidden;
    scrollbar-width: none;
}
.drocat-tabs .q-tabs__content {
    width: 100%;
    min-width: 0;
    gap: 2px;
    overflow: hidden !important;
}
.drocat-tabs .q-tab {
    min-width: 0 !important;
    min-height: 40px !important;
    flex: 1 1 0 !important;
    padding: 0 5px !important;
    border-radius: 9px;
    color: var(--drocat-navy); font-size: 13px; font-weight: 600;
    transition: background .16s ease, color .16s ease, box-shadow .16s ease;
}
.drocat-tabs .q-tab:hover { background: #e8edf6; }
.drocat-tabs .q-tab--active {
    color: var(--drocat-cobalt) !important;
    background: var(--drocat-surface) !important;
    box-shadow: 0 2px 6px rgba(11, 31, 58, .10);
}
.drocat-tabs .q-tab__icon { font-size: 17px; }
.drocat-tabs .q-tab__label {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 11px;
    font-weight: 650;
    letter-spacing: -.015em;
}
/* Group tints: Connectome (blue) vs NeuronBridge (purple) segments */
.drocat-tabs .drocat-connectome-tab { background: var(--drocat-cobalt-soft); }
.drocat-tabs .drocat-connectome-tab:hover { background: #dfe9ff !important; }
.drocat-tabs .drocat-nb-tab { background: #f6f1ff; }
.drocat-tabs .q-tab--active.drocat-nb-tab {
    color: #7c3aed !important;
}
.drocat-tabs .drocat-nb-tab:hover { background: #efe6ff !important; }
/* FlyLight imagery download: its own light-green tint (not NB purple) */
.drocat-tabs .drocat-flylight-tab { background: #edf9f0; }
.drocat-tabs .q-tab--active.drocat-flylight-tab {
    color: #15803d !important;
}
.drocat-tabs .drocat-flylight-tab:hover { background: #dff4e6 !important; }
/* The NB badge sits NEXT to the tab icon (no overlap) and uses the same
   font as the tab names, so it reads as part of the label typography. */
.drocat-nb-tab .q-tab__icon {
    display: inline-flex;
    align-items: center;
    gap: 3px;
}
.drocat-nb-tab .q-tab__icon::after {
    content: "NB";
    padding: 1px 5px;
    border-radius: 999px;
    background: #7c3aed;
    color: #fff;
    /* The icon element uses the Material Icons font; the badge must use the
       app font (same as tab names) instead of inheriting it. */
    font-family: Inter, "Avenir Next", "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 11px;
    font-weight: 650;
    line-height: 1.2;
    letter-spacing: -.015em;
    white-space: nowrap;
}
.drocat-tabs .q-tabs__arrow { display: none !important; }

@media (max-width: 1100px) {
    .drocat-tabs { gap: 1px; padding: 3px; }
    .drocat-tabs .q-tabs__content { gap: 1px; }
    .drocat-tabs .q-tab { padding: 0 3px !important; min-height: 36px !important; }
    .drocat-tabs .q-tab__icon { font-size: 15px; }
    .drocat-tabs .q-tab__label { font-size: 9px; letter-spacing: -.03em; }
    .drocat-nb-tab .q-tab__icon::after { font-size: 9px; padding: 0 4px; }
}

/* Panel tag badge (NeuronBridge etc.) */
.drocat-tag-badge {
    font-size: 10px !important;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
}

/* ---------- Page workspace ---------- */
.drocat-shell { width: 100%; max-width: 1500px; margin: 0 auto; padding: 20px 28px 44px; }
.drocat-page { width: 100%; padding: 20px 0 0; }
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
.drocat-results-card { position: sticky; top: 96px; }

@media (max-width: 1280px) {
    .drocat-workspace { grid-template-columns: 1fr; }
    .drocat-results-card { position: static; }
}

@media (max-width: 700px) {
    .drocat-header { min-height: 64px !important; padding: 0 16px !important; }
    .drocat-brand-mark { width: 36px; height: 36px; border-radius: 11px; }
    .drocat-brand-sub, .drocat-header-link { display: none; }
    .drocat-shell { padding: 14px 14px 32px; }
    .drocat-page { padding-top: 16px; }
    .drocat-page-head { align-items: flex-start; padding-bottom: 14px; margin-bottom: 16px; }
    .drocat-page-mark { width: 40px; height: 40px; border-radius: 12px; }
    .drocat-page-title { font-size: 19px; }
    .q-card.drocat-card { padding: 15px 14px; border-radius: 14px !important; }
    .drocat-param-grid { grid-template-columns: minmax(0, 1fr) !important; }
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
/* Segmented toggles: white pill container whose inner segments nest
   concentrically (inner radius = outer radius - padding - border), so the
   selected segment's corners snap to the container's corners with a uniform
   ring and no corner gap. */
.q-btn-group {
    background: var(--drocat-surface);
    border: 1px solid var(--drocat-line);
    border-radius: 999px;
    padding: 4px;
    box-shadow: 0 2px 8px rgba(11, 31, 58, .06);
}
.q-btn-group .q-btn { border-radius: 999px; }
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
    with ui.column().classes("w-full drocat-shell gap-3"):
        # Segmented navigation. Groups are distinguished by tint instead of
        # labels: Connectome tabs get the blue segment, NeuronBridge tabs the
        # purple segment (their tabs already carry the NB badge).
        with ui.tabs().classes("drocat-tabs w-full") as tabs:
            tab_pathfinding = ui.tab("Find Path", icon="route").classes("drocat-connectome-tab")
            tab_direct = ui.tab("Direct", icon="arrow_forward").classes("drocat-connectome-tab")
            tab_skeleton = ui.tab("3D Skeleton", icon="view_in_ar").classes("drocat-connectome-tab")
            tab_network = ui.tab("Network", icon="account_tree").classes("drocat-connectome-tab")
            tab_comparison = ui.tab("Cross-Dataset", icon="sync_alt").classes("drocat-connectome-tab")
            tab_homologs = ui.tab("Homologs", icon="compare").classes("drocat-connectome-tab")
            tab_profiling = ui.tab("Profiling", icon="analytics").classes("drocat-connectome-tab")
            tab_find_lines = ui.tab("Find Lines", icon="biotech").classes("drocat-nb-tab")
            tab_find_neuron = ui.tab("Find Neurons", icon="search").classes("drocat-nb-tab")
            tab_colabel = ui.tab("Co-Labeling", icon="layers").classes("drocat-nb-tab")
            tab_flylight = ui.tab("FlyLight", icon="download").classes("drocat-flylight-tab")
            tab_settings = ui.tab("Settings", icon="settings")

        with ui.tab_panels(tabs, value=tab_pathfinding).classes("w-full bg-transparent"):
            with ui.tab_panel(tab_pathfinding).classes("p-0"):
                create_find_path_tab()
            with ui.tab_panel(tab_direct).classes("p-0"):
                create_find_direct_tab()
            with ui.tab_panel(tab_skeleton).classes("p-0"):
                create_skeleton_tab()
            with ui.tab_panel(tab_network).classes("p-0"):
                create_network_tab()
            with ui.tab_panel(tab_comparison).classes("p-0"):
                create_inter_dataset_tab()
            with ui.tab_panel(tab_find_lines).classes("p-0"):
                create_nb_find_lines_tab()
            with ui.tab_panel(tab_find_neuron).classes("p-0"):
                create_nb_find_neuron_tab()
            with ui.tab_panel(tab_colabel).classes("p-0"):
                create_nb_colabel_tab()
            with ui.tab_panel(tab_flylight).classes("p-0"):
                create_flylight_tab()
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
    host = os.environ.get("DROCAT_UI_HOST", APP_HOST)
    port = int(os.environ.get("DROCAT_UI_PORT", APP_PORT))
    show = os.environ.get("DROCAT_UI_SHOW", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    ui.run(
        title=APP_TITLE,
        host=host,
        port=port,
        reload=False,
        show=show,
        favicon="🧠",
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
