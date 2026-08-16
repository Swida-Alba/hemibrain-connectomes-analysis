#!/usr/bin/env python
"""
DROCAT UI - Integrated Web Interface for Connectome Analysis

Launch with: python ui/app.py
Access at: http://127.0.0.1:8080

Design language follows the Photo Selector "gallery" reference:
light canvas, white surfaces, cobalt accent, segmented navigation,
focus-panel + contact-sheet workspace.
"""

import logging
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


class _TimerTeardownNoiseFilter(logging.Filter):
    """Hide the harmless NiceGUI 3.15 timer-teardown race from the console.

    When a browser tab is closed or reloaded, NiceGUI deletes the client's
    elements and then wakes tasks still waiting in ``client.connected()``;
    a per-page ``ui.timer`` can enter its loop one last time against an
    already-deleted parent slot and raise
    ``RuntimeError: The parent slot of Timer(id=...) has been deleted.``
    The page is already gone at that point, so the default exception
    handler's traceback is pure log noise - drop exactly those records.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (
            "The parent slot of Timer" in message
            and "has been deleted" in message
        )


def _silence_timer_teardown_noise() -> None:
    logger = logging.getLogger("nicegui")
    if not any(isinstance(f, _TimerTeardownNoiseFilter) for f in logger.filters):
        logger.addFilter(_TimerTeardownNoiseFilter())


_silence_timer_teardown_noise()


def _migrate_legacy_neuron_indexes() -> None:
    """One-time upgrade: move legacy cache/ neuron indexes to neuron_indexes/.

    The index is a persistent "system file" now (auto-suggestions and the
    available-neurons viewer read it from ``neuron_indexes/``), so existing
    installations keep their pull state without re-downloading metadata.
    Runs once at startup; every failure is non-fatal.
    """
    try:
        from src.neuron_index_builder import migrate_legacy_neuron_indexes
        migrated = migrate_legacy_neuron_indexes(
            PROJECT_ROOT / "cache",
            PROJECT_ROOT / "neuron_indexes",
        )
        if migrated:
            print(
                "Migrated legacy neuron indexes to neuron_indexes/: "
                + ", ".join(migrated),
                flush=True,
            )
    except Exception:
        pass


_migrate_legacy_neuron_indexes()


def _saved_dark_mode() -> bool | None:
    """Return the persisted theme preference from a plain cookie.

    Values: ``True`` = dark, ``False`` = light, ``None`` = follow the
    system preference. The cookie stores 'dark', 'light' or 'auto' (older
    versions stored '1'/'0') so a reload restores the chosen mode without
    a flash. Falls back to system mode when no preference is saved yet or
    no request context exists (script mode, UI tests).
    """
    try:
        request = ui.context.client.request
    except Exception:
        return None
    saved = request.cookies.get("drocat_dark")
    if saved in ("1", "dark"):
        return True
    if saved in ("0", "light"):
        return False
    # 'auto', missing, or unrecognized -> follow the OS preference.
    return None


from ui.tabs import (
    create_find_path_tab,
    create_find_shortest_tab,
    create_network_tab,
    create_inter_dataset_tab,
    create_skeleton_tab,
    create_net_viz_tab,
    create_find_homologs_tab,
    create_find_similar_tab,
    create_connectivity_profiling_tab,
    create_nb_find_lines_tab,
    create_nb_find_neuron_tab,
    create_nb_colabel_tab,
    create_flylight_tab,
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
    /* Tab-group tints, shared by the group headers and their tabs */
    --drocat-tint-connection: #eaf0ff;
    --drocat-tint-visualization: #e2f6f5;
    --drocat-tint-similarity: #fff3e0;
    --drocat-tint-nb: #f6f1ff;
    --drocat-tint-flylight: #edf9f0;
    --drocat-tint-settings: #eef1f5;
    /* Glass and solid surfaces whose light values are baked into the CSS */
    --drocat-glass: rgba(255, 255, 255, .86);
    --drocat-panel-glass: rgba(255, 255, 255, .72);
    --drocat-table-glass: rgba(255, 255, 255, .78);
    --drocat-toolbar-soft: rgba(247, 249, 252, .92);
    --drocat-toolbar-grad: linear-gradient(135deg, rgba(234, 240, 255, .78), rgba(255, 255, 255, .96));
    --drocat-query-preview-grad: linear-gradient(135deg, rgba(232, 241, 255, .78), rgba(255, 255, 255, .92));
    --drocat-labelmapper-grad: linear-gradient(180deg, rgba(247, 249, 252, .84), #fff);
    --drocat-tab-hover: rgba(255, 255, 255, .55);
    --drocat-cell: #ffffff;
    --drocat-row-even: #f7fbff;
    --drocat-row-hover: #eaf3ff;
    --drocat-selected: #dcecff;
    --drocat-selected-hover: #cfe3ff;
    --drocat-badge-bg: #fff8d6;
    --drocat-badge-fg: #946200;
    --drocat-badge-soft-bg: #fffbe8;
    --drocat-badge-soft-fg: #a07a2c;
    --drocat-mark-bg: #fff1b8;
    --drocat-err-border: #fecaca;
}

/* ---------- Dark mode ----------
   NiceGUI/Quasar dark mode toggles the body--dark class on <body>. These
   overrides flip every drocat token; the html:has() arm keeps the page
   canvas dark behind the body as well. */
body.body--dark,
html:has(> body.body--dark) {
    --drocat-canvas: #0c1322;
    --drocat-surface: #131c2e;
    --drocat-soft: #1d2b45;
    --drocat-navy: #e7eef8;
    --drocat-muted: #9aa9c0;
    --drocat-faint: #6b7a93;
    --drocat-line: #25334d;
    --drocat-line-strong: #374a6b;
    --drocat-cobalt: #5b8cff;
    --drocat-cobalt-soft: #1a2c52;
    --drocat-ok: #22c55e;
    --drocat-warn: #f59e0b;
    --drocat-err: #f87171;
    --drocat-shadow: 0 16px 40px rgba(0, 0, 0, .5);
    --drocat-tint-connection: #18233c;
    --drocat-tint-visualization: #132a2c;
    --drocat-tint-similarity: #2c2313;
    --drocat-tint-nb: #251a3d;
    --drocat-tint-flylight: #152a1d;
    --drocat-tint-settings: #1c2637;
    --drocat-glass: rgba(19, 28, 46, .86);
    --drocat-panel-glass: rgba(19, 28, 46, .72);
    --drocat-table-glass: rgba(19, 28, 46, .78);
    --drocat-toolbar-soft: rgba(23, 33, 52, .92);
    --drocat-toolbar-grad: linear-gradient(135deg, rgba(26, 44, 82, .72), rgba(19, 28, 46, .96));
    --drocat-query-preview-grad: linear-gradient(135deg, rgba(26, 44, 82, .6), rgba(19, 28, 46, .92));
    --drocat-labelmapper-grad: linear-gradient(180deg, rgba(28, 42, 63, .84), #131c2e);
    --drocat-tab-hover: rgba(255, 255, 255, .07);
    --drocat-cell: #131c2e;
    --drocat-row-even: #17233a;
    --drocat-row-hover: #1e3050;
    --drocat-selected: #1d3a5f;
    --drocat-selected-hover: #25486f;
    --drocat-badge-bg: #3d3113;
    --drocat-badge-fg: #e5b74d;
    --drocat-badge-soft-bg: #332b16;
    --drocat-badge-soft-fg: #c9a44d;
    --drocat-mark-bg: #3d3612;
    --drocat-err-border: #5c2c2c;
}

/* A slightly stronger canvas glow keeps the dark theme from going flat. */
body.body--dark {
    background:
        radial-gradient(circle at 16% 0%, rgba(91, 140, 255, .08), transparent 32%),
        var(--drocat-canvas) !important;
}

/* Active group-tab accents need brighter hues on the dark tints. */
body.body--dark .drocat-tint-visualization .drocat-group-tab.drocat-active { color: #2dd4bf !important; }
body.body--dark .drocat-tint-similarity .drocat-group-tab.drocat-active { color: #f59e0b !important; }
body.body--dark .drocat-tint-nb .drocat-group-tab.drocat-active { color: #a78bfa !important; }
body.body--dark .drocat-tint-flylight .drocat-group-tab.drocat-active { color: #4ade80 !important; }
body.body--dark .drocat-tint-settings .drocat-group-tab.drocat-active { color: #94a3b8 !important; }

/* The results mark keeps its white icon on a dark navy chip in both themes. */
body.body--dark .drocat-results-mark { background: #1d2a42; }

/* Theme picker in the header (System / Light / Dark). The button shows a
   persistent sun | moon pair in every mode; the menu highlights the mode. */
.drocat-dark-toggle {
    color: var(--drocat-muted) !important;
    font-size: 20px;
}
.drocat-dark-toggle:hover { color: var(--drocat-cobalt) !important; }
.drocat-theme-icon-pair {
    display: flex;
    align-items: center;
    gap: 5px;
    line-height: 1;
}
.drocat-theme-icon-pair .q-icon {
    /* NiceGUI renders q-icon at 1.715em (the button was 20px -> 34.3px);
       pin the icons to exactly 2/3 of that previous size. */
    font-size: 22.9px;
}
.drocat-theme-sep {
    width: 2.5px;
    height: 14px;
    border-radius: 999px;
    background: var(--drocat-line-strong);
}
.drocat-theme-menu { min-width: 152px; }
.drocat-theme-item { color: var(--drocat-muted); border-radius: 8px; }
.drocat-theme-item .drocat-theme-icon { color: var(--drocat-muted); }
.drocat-theme-item .drocat-theme-check { color: transparent; }
.drocat-theme-item.drocat-theme-item-active,
.drocat-theme-item.drocat-theme-item-active .drocat-theme-icon,
.drocat-theme-item.drocat-theme-item-active .drocat-theme-check {
    color: var(--drocat-cobalt) !important;
}

/* Token reminder box (Settings tab) follows the theme. */
.drocat-token-reminder {
    border: 1px solid #e6a23c;
    background: #fdf6ec;
    border-radius: 8px;
    padding: 10px 12px;
}
body.body--dark .drocat-token-reminder {
    border-color: #8a6a1f;
    background: #2f2514;
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
    background: var(--drocat-glass) !important;
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
.drocat-inline-link {
    color: var(--drocat-cobalt) !important;
    font-size: 12px;
    font-weight: 650;
    text-decoration: none !important;
}
.drocat-inline-link:hover { text-decoration: underline !important; }
.drocat-neuron-search-toolbar {
    margin-top: 8px;
    padding: 10px 12px 8px;
    border: 1px solid var(--drocat-line-strong);
    border-radius: 14px;
    background: var(--drocat-toolbar-grad);
    box-shadow: 0 4px 14px rgba(11, 31, 58, .06);
}
.drocat-neuron-search-toolbar .drocat-neuron-search-field {
    background: var(--drocat-table-glass);
    border-radius: 10px;
}
.drocat-neuron-search-toolbar .q-field--outlined .q-field__control {
    border-radius: 10px;
}
.drocat-neuron-search-toolbar .q-field--outlined .q-field__control:before {
    border-color: var(--drocat-line-strong);
}
.drocat-neuron-search-toolbar .q-field--outlined.q-field--focused .q-field__control:after {
    border-color: var(--drocat-cobalt);
}
.drocat-neuron-search-toolbar .q-field__label {
    color: var(--drocat-muted);
}
.drocat-neuron-results-layout {
    display: grid;
    grid-template-columns: minmax(480px, 520px) minmax(0, 1fr);
    gap: 12px;
    align-items: stretch;
    min-height: 0;
}
.drocat-neuron-match-panel,
.drocat-neuron-full-panel {
    min-width: 0;
    min-height: 0;
    border: 1px solid var(--drocat-line);
    border-radius: 12px;
    padding: 12px;
    background: var(--drocat-surface);
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
.drocat-neuron-match-panel {
    background: linear-gradient(180deg, var(--drocat-cobalt-soft), var(--drocat-surface));
    border-color: var(--drocat-line-strong);
    box-shadow: 0 3px 12px rgba(11, 31, 58, .06);
}
.drocat-neuron-query-preview {
    margin: 10px 0 2px;
    padding: 10px 12px;
    border: 1px solid var(--drocat-line-strong);
    border-radius: 10px;
    background: var(--drocat-query-preview-grad);
}
.drocat-neuron-query-preview-list {
    max-height: 118px;
    overflow-y: auto;
    overflow-x: hidden;
    padding: 2px 2px 3px;
    scrollbar-width: thin;
}
.drocat-neuron-query-preview-list.drocat-neuron-query-preview-expanded {
    max-height: none;
    overflow-y: visible;
}
.drocat-query-preview-expand-btn {
    min-height: 24px;
    padding: 0 6px;
    color: var(--drocat-cobalt);
    font-size: 11px;
}
.drocat-query-preview-expand-btn:hover {
    background: rgba(69, 126, 191, .10);
}
.drocat-neuron-query-chip {
    display: inline-flex;
    align-items: center;
    min-height: 26px;
    padding: 3px 9px;
    border: 1px solid rgba(69, 126, 191, .38);
    border-radius: 999px;
    background: var(--drocat-cell);
    color: var(--drocat-navy);
    font-size: 12px;
    font-weight: 650;
    line-height: 1.2;
}
.drocat-neuron-query-chip-wrap {
    display: inline-flex;
    align-items: center;
    gap: 2px;
    min-height: 26px;
    padding: 3px 5px 3px 9px;
    border: 1px solid rgba(69, 126, 191, .38);
    border-radius: 999px;
    background: var(--drocat-cell);
    color: var(--drocat-navy);
}
.drocat-neuron-query-chip-wrap .drocat-neuron-query-chip {
    min-height: auto;
    padding: 0;
    border: 0;
    background: transparent;
}
.drocat-neuron-query-chip-remove {
    min-height: 20px;
    min-width: 20px;
    color: var(--drocat-muted);
}
.drocat-neuron-query-chip-remove:hover {
    color: var(--drocat-navy);
    background: rgba(69, 126, 191, .10);
}
.drocat-chip-input-shell .drocat-chip-input .q-field__control {
    align-content: flex-start;
    scrollbar-width: thin;
}
.drocat-fixed-neuron-input {
    width: 100%;
    min-width: 0;
}
.drocat-fixed-neuron-input .drocat-neuron-input-row {
    flex-wrap: nowrap;
    min-width: 0;
}
.drocat-fixed-neuron-input .drocat-chip-input-shell {
    flex: 1 1 0%;
    width: 0;
    min-width: 0;
}
.drocat-fixed-neuron-input .drocat-chip-input-shell .q-field {
    width: 100%;
    min-width: 0;
}
.drocat-fixed-neuron-input .drocat-neuron-match-filter {
    flex: 0 0 128px;
    width: 128px !important;
    min-width: 128px;
}
.drocat-fixed-neuron-input .drocat-upload-trigger {
    flex: 0 0 34px;
    width: 34px !important;
    min-width: 34px;
}
@media (max-width: 760px) {
    .drocat-fixed-neuron-input .drocat-neuron-input-row {
        flex-wrap: wrap;
    }
    .drocat-fixed-neuron-input .drocat-chip-input-shell {
        flex-basis: 100%;
        width: 100%;
    }
    .drocat-fixed-neuron-input .drocat-neuron-match-filter {
        flex: 1 1 128px;
    }
}
.drocat-chip-input-shell.drocat-chip-list-collapsed
    .drocat-chip-input .q-field__control {
    max-height: 142px;
    overflow-y: auto;
    overflow-x: hidden;
}
.drocat-chip-input-shell.drocat-chip-list-expanded
    .drocat-chip-input .q-field__control {
    max-height: none;
    overflow-y: visible;
    overflow-x: hidden;
}
.drocat-chip-expand-btn {
    min-height: 24px;
    padding: 0 6px;
    color: var(--drocat-cobalt);
    font-size: 11px;
}
.drocat-chip-expand-btn:hover {
    background: rgba(69, 126, 191, .10);
}
.drocat-neuron-match-jump {
    min-height: 26px;
    padding: 0 4px;
    color: var(--drocat-navy);
    font-weight: 650;
}
.drocat-neuron-match-jump:hover {
    color: var(--drocat-cobalt);
    background: rgba(69, 126, 191, .10);
}
.drocat-neuron-full-panel {
    background: var(--drocat-panel-glass);
}
.drocat-neuron-viewer-card {
    max-height: 94vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    padding: 12px 16px 14px;
}
.drocat-neuron-dialog-header {
    min-height: 30px;
    margin-bottom: 2px;
}
.drocat-neuron-dialog-title {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.drocat-neuron-header-meta {
    min-width: 0;
    overflow: hidden;
}
.drocat-neuron-header-meta .q-badge {
    min-height: 22px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 700;
}
.drocat-neuron-header-meta .drocat-neuron-source,
.drocat-neuron-header-meta .drocat-neuron-enriched {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.drocat-neuron-intro-row {
    display: grid;
    grid-template-columns: minmax(320px, .9fr) minmax(0, 1.6fr);
    align-items: stretch;
    gap: 10px;
    margin: 2px 0 4px;
}
.drocat-neuron-intro-row .drocat-neuron-query-preview {
    min-width: 0;
    margin: 0;
    padding: 7px 10px;
}
.drocat-neuron-search-help {
    align-self: center;
    min-width: 0;
    margin: 0;
    line-height: 1.35;
}
.drocat-neuron-viewer-content {
    min-height: 0;
    overflow: auto;
    padding-top: 0;
    padding-bottom: 4px;
}
.drocat-neuron-panel-toolbar {
    flex: 0 0 auto;
    min-height: 34px;
    margin-top: 6px;
    padding: 4px 6px;
    border: 1px solid var(--drocat-line);
    border-radius: 8px;
    background: var(--drocat-toolbar-soft);
}
.drocat-neuron-match-table .q-table__middle {
    max-height: min(58vh, 680px);
    min-height: 0;
    flex: 1 1 auto;
    overflow-x: hidden;
    overflow-y: auto;
    margin-top: 8px;
    border: 1px solid var(--drocat-line);
    border-radius: 8px;
    background: var(--drocat-table-glass);
}
.drocat-neuron-match-table table {
    width: 100%;
    table-layout: fixed;
}
.drocat-neuron-match-table th {
    position: sticky;
    top: 0;
    z-index: 2;
    background: var(--drocat-cobalt-soft) !important;
    color: var(--drocat-navy) !important;
    white-space: nowrap;
}
.drocat-neuron-match-table td,
.drocat-neuron-match-table th {
    height: 40px;
    padding: 7px 9px;
}
.drocat-neuron-match-table td.drocat-neuron-match-by {
    color: var(--drocat-muted);
    font-size: 12px;
    font-weight: 650;
    white-space: nowrap;
}
.drocat-neuron-match-table td.drocat-neuron-match-value {
    color: var(--drocat-navy);
    font-weight: 700;
    overflow: visible;
    white-space: normal;
    overflow-wrap: anywhere;
}
.drocat-neuron-match-table td.drocat-neuron-match-count {
    color: var(--drocat-muted);
    font-variant-numeric: tabular-nums;
    text-align: right;
    white-space: nowrap;
}
.drocat-neuron-match-select-cell {
    width: 36px;
    min-width: 36px;
    padding-left: 7px !important;
    padding-right: 3px !important;
    text-align: center;
}
.drocat-neuron-match-secondary-arrow {
    color: var(--drocat-faint);
    opacity: .9;
}
.drocat-neuron-match-source {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    white-space: nowrap;
}
.drocat-neuron-match-value-line {
    display: flex;
    align-items: center;
    gap: 5px;
    min-width: 0;
}
.drocat-neuron-match-secondary-label {
    display: inline-flex;
    align-items: center;
    min-height: 18px;
    padding: 1px 5px;
    border: 1px solid rgba(234, 179, 8, .68);
    border-radius: 999px;
    background: var(--drocat-badge-bg);
    color: var(--drocat-badge-fg);
    font-size: 10px;
    font-weight: 750;
    letter-spacing: .02em;
    line-height: 1.2;
    text-transform: uppercase;
    white-space: nowrap;
}
.drocat-neuron-match-first {
    margin: 1px 0 0 4px;
    color: var(--drocat-muted);
    font-size: 10px;
    font-weight: 500;
    line-height: 1.2;
    white-space: nowrap;
}
.drocat-neuron-match-table tbody tr.drocat-neuron-match-secondary-row > td {
    height: 28px;
    padding-top: 3px;
    padding-bottom: 3px;
    color: var(--drocat-faint);
    font-size: 11px;
    font-weight: 500;
    line-height: 1.15;
}
.drocat-neuron-match-table tbody tr.drocat-neuron-match-secondary-row
    .drocat-neuron-match-source {
    color: var(--drocat-faint);
    font-size: 11px;
    font-weight: 500;
}
.drocat-neuron-match-table tbody tr.drocat-neuron-match-secondary-row
    .drocat-neuron-match-value-line,
.drocat-neuron-match-table tbody tr.drocat-neuron-match-secondary-row
    .drocat-neuron-match-first {
    padding-left: 12px;
}
.drocat-neuron-match-table tbody tr.drocat-neuron-match-secondary-row
    .drocat-neuron-match-jump {
    min-height: 21px;
    padding: 0 3px;
    color: var(--drocat-muted);
    font-size: 11px;
    font-weight: 550;
}
.drocat-neuron-match-table tbody tr.drocat-neuron-match-secondary-row
    .drocat-neuron-match-first {
    margin-top: 0;
    color: var(--drocat-faint);
    font-size: 9px;
}
.drocat-neuron-match-table tbody tr.drocat-neuron-match-secondary-row
    .drocat-neuron-match-secondary-label {
    min-height: 15px;
    padding: 0 4px;
    margin-right: 2px;
    border-color: rgba(234, 179, 8, .42);
    background: var(--drocat-badge-soft-bg);
    color: var(--drocat-badge-soft-fg);
    font-size: 9px;
    font-weight: 650;
}
.drocat-neuron-match-panel .q-table__control {
    color: var(--drocat-cobalt);
}
.drocat-data-viewer-scroll {
    position: relative;
    max-width: 100%;
    overflow: auto;
    border: 1px solid var(--drocat-line);
    border-radius: 10px;
}
.drocat-data-viewer-table .q-table__middle {
    max-height: min(58vh, 680px);
    min-height: 0;
    flex: 1 1 auto;
    overflow: auto;
}
.drocat-data-viewer-table .q-table__middle thead tr:first-child th {
    position: sticky;
    top: 0;
    z-index: 8;
    background: var(--drocat-cobalt-soft) !important;
    color: var(--drocat-navy) !important;
    box-shadow: 0 1px 0 var(--drocat-line-strong), 0 3px 8px rgba(11, 31, 58, .08);
}
.drocat-data-viewer-table th { white-space: nowrap; }
.drocat-data-viewer-table td,
.drocat-data-viewer-table th {
    height: 40px;
    padding: 7px 9px;
}
.drocat-data-viewer-table tbody tr > td {
    /* A QTable selection can otherwise leave its default gray row fill in
       place while the viewer updates its persistent selection sets. The
       viewer owns the two intentional states below: theme surface for
       ordinary cells and blue for selected rows. */
    background: var(--drocat-cell) !important;
    color: var(--drocat-navy) !important;
}
.drocat-data-viewer-table.q-table__container,
.drocat-data-viewer-table .q-table__middle,
.drocat-data-viewer-table .q-table,
.drocat-data-viewer-table tbody,
.drocat-data-viewer-table tbody tr:not(.drocat-neuron-selected-row) {
    background: var(--drocat-cell) !important;
}
.drocat-data-viewer-table tbody tr.q-tr--selected:not(.drocat-neuron-selected-row) > td,
.drocat-data-viewer-table tbody tr[aria-selected="true"]:not(.drocat-neuron-selected-row) > td {
    background: var(--drocat-cell) !important;
}
.drocat-data-viewer-table tbody tr:hover > td {
    background: var(--drocat-cell) !important;
}
.drocat-data-viewer-table tr.drocat-neuron-selected-row > td {
    /* Direct body-ID selections and matched-group selections share the same
       Quasar row-selection state. Keep that state visibly distinct from the
       yellow cell used for the query match. */
    background: var(--drocat-selected) !important;
    color: var(--drocat-navy) !important;
}
.drocat-data-viewer-table tr.drocat-neuron-selected-row:hover > td {
    background: var(--drocat-selected-hover) !important;
}
.drocat-data-viewer-table tr.drocat-neuron-selected-row > td:first-child {
    box-shadow: inset 3px 0 0 var(--drocat-cobalt);
}
.drocat-data-viewer-table td.drocat-neuron-hit-cell {
    /* The matched source cell follows the scroll until it would leave the
       viewport, then floats at the nearest horizontal edge. This keeps the
       actual highlighted cell visible in both scroll directions without
       changing the full table's column layout. */
    position: sticky;
    left: 0;
    right: 0;
    z-index: 4;
    /* Keep the cell's normal row background; only the nested mark below is
       yellow. The outline still identifies every matched cell. */
    background: var(--drocat-cell) !important;
    color: var(--drocat-navy) !important;
    font-weight: 600;
    box-shadow: inset 0 0 0 2px #eab308;
}
.drocat-data-viewer-table td.drocat-neuron-secondary-hit-cell {
    position: static;
    z-index: 1;
    background: var(--drocat-cell) !important;
    color: var(--drocat-navy) !important;
    font-weight: 600;
    box-shadow: inset 0 0 0 2px #eab308;
}
.drocat-data-viewer-table tr.drocat-neuron-selected-row > td.drocat-neuron-hit-cell,
.drocat-data-viewer-table tr.drocat-neuron-selected-row > td.drocat-neuron-secondary-hit-cell {
    background: var(--drocat-selected) !important;
    color: var(--drocat-navy) !important;
}
.drocat-data-viewer-table mark.drocat-neuron-match-text {
    padding: 0 .08em;
    border-radius: 3px;
    background: var(--drocat-mark-bg);
    color: var(--drocat-navy);
    font-weight: 750;
}
@keyframes drocat-neuron-focus-breathe {
    0%, 100% { opacity: 0; }
    50% { opacity: 1; }
}
.drocat-data-viewer-table tr.drocat-neuron-focus-flash > td {
    position: relative;
}
.drocat-data-viewer-table tr.drocat-neuron-focus-flash > td::after {
    content: "";
    position: absolute;
    inset: 0;
    z-index: 3;
    pointer-events: none;
    background: rgba(148, 163, 184, .20);
    animation: drocat-neuron-focus-breathe 1.5s ease-in-out 1;
}
.drocat-data-viewer-table tr.drocat-neuron-focus-flash > td > * {
    position: relative;
    z-index: 4;
}
@media (max-width: 1100px) {
    .drocat-neuron-intro-row {
        grid-template-columns: 1fr;
    }
    .drocat-neuron-results-layout {
        grid-template-columns: 1fr;
    }
    .drocat-neuron-match-panel {
        position: static;
    }
}
@media (max-width: 760px) {
    .drocat-neuron-viewer-card {
        padding: 8px 10px 10px;
    }
    .drocat-neuron-header-meta {
        flex-basis: 100%;
    }
    .drocat-neuron-dialog-header {
        align-items: flex-start;
    }
}

/* ---------- Grouped navigation: layered cards ---------- */
/* Every tab group is its own tinted card: the group header sits on top of
   its tab segments INSIDE the same card, so header and tabs are always
   aligned and each group reads as an independent block. Settings is a
   standalone card (no header), separated from the function groups. */
.drocat-nav {
    display: flex;
    gap: 8px;
    align-items: stretch;
}
.drocat-group-card {
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 4px;
    border: 1px solid var(--drocat-line);
    border-radius: 14px;
    box-shadow: inset 0 1px 3px rgba(11, 31, 58, .04);
}
.drocat-group-card.drocat-tint-connection { background: var(--drocat-tint-connection); }
.drocat-group-card.drocat-tint-visualization { background: var(--drocat-tint-visualization); }
.drocat-group-card.drocat-tint-similarity { background: var(--drocat-tint-similarity); }
.drocat-group-card.drocat-tint-nb { background: var(--drocat-tint-nb); }
.drocat-group-card.drocat-tint-flylight { background: var(--drocat-tint-flylight); }
.drocat-group-card.drocat-tint-settings { background: var(--drocat-tint-settings); }
/* Settings stands alone on the right, apart from the function groups. */
.drocat-settings-card { margin-left: auto; flex: 0 0 auto; }
.drocat-group-head {
    min-width: 0;
    text-align: center;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--drocat-navy);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    padding: 2px 4px 0;
}
/* The single NB badge lives on the NeuronBridge group header. */
.drocat-group-head.drocat-head-nb::after {
    content: "NB";
    margin-left: 6px;
    padding: 1px 6px;
    border-radius: 999px;
    background: #7c3aed;
    color: #fff;
    /* The header is plain text; the badge keeps the app font explicitly. */
    font-family: Inter, "Avenir Next", "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .02em;
}
.drocat-group-tabs { display: flex; gap: 2px; flex: 1; }
.drocat-group-tab {
    flex: 1 1 0;
    min-width: 0;
    min-height: 54px;
    padding: 6px 4px;
    border-radius: 9px;
    transition: background .16s ease, color .16s ease, box-shadow .16s ease;
}
.drocat-group-tab.q-btn--flat { color: var(--drocat-navy); }
/* Two-row segments: icon on top, name below. */
.drocat-group-tab .q-btn__content { flex-direction: column; gap: 2px; }
.drocat-group-tab .q-icon { font-size: 18px; }
.drocat-group-tab .q-btn__label,
.drocat-group-tab .q-btn__content .block {
    min-width: 0;
    font-size: 11px;
    font-weight: 650;
    letter-spacing: -.015em;
    /* Multi-word names render one word per line with tight leading. */
    white-space: pre-line;
    line-height: 1.15;
}
.drocat-group-tab:hover { background: var(--drocat-tab-hover); }
.drocat-group-tab.drocat-active {
    background: var(--drocat-surface) !important;
    box-shadow: 0 2px 6px rgba(11, 31, 58, .10);
}
/* The active segment takes its group's accent color. */
.drocat-tint-connection .drocat-group-tab.drocat-active { color: var(--drocat-cobalt) !important; }
.drocat-tint-visualization .drocat-group-tab.drocat-active { color: #0e7490 !important; }
.drocat-tint-similarity .drocat-group-tab.drocat-active { color: #b45309 !important; }
.drocat-tint-nb .drocat-group-tab.drocat-active { color: #7c3aed !important; }
.drocat-tint-flylight .drocat-group-tab.drocat-active { color: #15803d !important; }
.drocat-tint-settings .drocat-group-tab.drocat-active { color: #475467 !important; }

@media (max-width: 1100px) {
    /* Cards no longer fit side by side: let the nav strip scroll instead
       of squeezing the segments until their labels collide. */
    .drocat-nav { overflow-x: auto; scrollbar-width: none; }
    .drocat-group-card { min-width: 160px; }
    .drocat-group-head { font-size: 10px; letter-spacing: .05em; }
    .drocat-group-head.drocat-head-nb::after { font-size: 8px; padding: 0 5px; margin-left: 4px; }
    .drocat-group-tab { min-height: 46px; padding: 4px 3px; }
    .drocat-group-tab .q-icon { font-size: 15px; }
    .drocat-group-tab .q-btn__label { font-size: 9px; letter-spacing: -.03em; }
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
.drocat-page-progress {
    padding: 10px 0 12px !important;
    border-color: var(--drocat-line-strong) !important;
    background: linear-gradient(135deg, var(--drocat-cobalt-soft), var(--drocat-surface)) !important;
}
.drocat-page-progress-compact {
    margin: 0 0 12px;
    border-bottom: 1px solid var(--drocat-line);
    background: transparent !important;
}
.drocat-page-progress-head { min-height: 30px; }
.drocat-page-progress-mark {
    width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center;
    background: var(--drocat-cobalt);
}
.drocat-page-progress-mark .q-icon { font-size: 17px; }
.drocat-page-progress-summary { min-height: 20px; }
.drocat-progress-percent {
    color: var(--drocat-navy); font-weight: 750; font-variant-numeric: tabular-nums;
}
.drocat-progress-bar {
    height: 12px !important;
    min-height: 12px !important;
    border-radius: 999px;
}

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
    .drocat-nav { overflow-x: auto; scrollbar-width: none; }
    .drocat-group-card { min-width: 170px; }
    .drocat-group-head { font-size: 9px; letter-spacing: .03em; }
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

/* ---------- Net-Viz edge-list editor ---------- */
.drocat-edge-editor {
    overflow: hidden;
    border: 1px solid var(--drocat-line) !important;
    border-radius: var(--drocat-radius-md) !important;
    background: var(--drocat-surface) !important;
    box-shadow: 0 2px 10px rgba(11, 31, 58, .05) !important;
}
.drocat-edge-editor .q-expansion-item__container,
.drocat-edge-editor .q-expansion-item__content {
    background: var(--drocat-surface) !important;
}
.drocat-edge-editor .q-item {
    min-height: 52px;
    padding: 8px 18px;
    color: var(--drocat-navy);
    font-weight: 650;
}
.drocat-edge-editor .q-expansion-item__content {
    padding: 4px 18px 18px;
}
.drocat-edge-table {
    overflow: hidden;
    border-color: var(--drocat-line-strong) !important;
    border-radius: 10px;
}
.drocat-edge-table .q-table__middle {
    overflow-x: auto;
}
.drocat-edge-table .q-table thead tr {
    background: var(--drocat-cobalt-soft);
}
.drocat-edge-table .q-table th {
    height: 44px;
    color: var(--drocat-navy);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .02em;
    text-transform: none;
}
.drocat-edge-table .q-table td {
    height: 48px;
    color: var(--drocat-navy);
    font-size: 13px;
}
.drocat-edge-header-cell,
.drocat-edge-cell {
    border-right: 1px solid var(--drocat-line) !important;
}
.drocat-edge-header-cell.drocat-edge-divider,
.drocat-edge-cell.drocat-edge-divider {
    border-right-color: var(--drocat-line-strong) !important;
}
.drocat-edge-select-cell {
    width: 52px;
    min-width: 52px;
    border-right: 1px solid var(--drocat-line-strong) !important;
    text-align: center;
}
.drocat-edge-cell {
    min-width: 142px;
    padding: 4px 10px !important;
}
.drocat-edge-cell:nth-child(4) {
    min-width: 105px;
}
.drocat-edge-cell:nth-child(5) {
    min-width: 180px;
}
.drocat-edge-table .q-field--borderless .q-field__control {
    min-height: 34px;
    padding: 0 4px;
}
.drocat-edge-table .q-field--borderless .q-field__native,
.drocat-edge-table .q-field--borderless .q-field__input {
    color: var(--drocat-navy) !important;
    font-size: 13px;
}
.drocat-edge-row-even > td {
    background: var(--drocat-row-even) !important;
}
.drocat-edge-row-odd > td {
    background: var(--drocat-surface) !important;
}
.drocat-edge-row-even:hover > td,
.drocat-edge-row-odd:hover > td {
    background: var(--drocat-row-hover) !important;
}
.drocat-empty-canvas-btn {
    min-height: 40px;
    white-space: nowrap;
}

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
.q-tooltip {
    font-size: 14px !important;
    line-height: 1.35 !important;
}
.drocat-param-grid { gap: 14px 16px !important; }

/* Compact selects that sit beside a related checkbox.  The wrapper keeps
   the field from claiming the entire flex row while still allowing the row
   to wrap cleanly on narrow screens. */
.drocat-inline-select {
    flex: 0 1 170px;
    width: 170px;
    min-width: 150px;
    max-width: 190px;
}
.drocat-inline-select .q-field { width: 100%; }

/* ---------- LabelMapper / custom-group editors ---------- */
.drocat-settings-mapping-controls {
    align-items: center;
}
.drocat-settings-mapping-controls > .q-field,
.drocat-settings-mapping-controls > .q-btn {
    flex: 1 1 0;
    min-width: 280px;
    width: auto !important;
}
.drocat-settings-mapping-button {
    min-height: 48px !important;
    font-size: 16px !important;
}
.drocat-labelmapper-board {
    min-width: 0;
}
.drocat-labelmapper-group {
    min-width: 0;
    position: relative;
    display: grid !important;
    grid-template-columns: minmax(220px, 360px) minmax(0, 1fr);
    grid-template-rows: auto auto;
    column-gap: 24px;
    row-gap: 12px;
    align-items: start;
    padding: 12px 14px 14px;
    border: 1px solid var(--drocat-line-strong);
    border-radius: 14px;
    background: var(--drocat-labelmapper-grad);
    box-shadow: 0 2px 8px rgba(11, 31, 58, .04);
}
.drocat-labelmapper-group-name {
    grid-column: 1;
    grid-row: 1;
    width: 100%;
    min-width: 0;
    z-index: 1;
}
.drocat-labelmapper-group-members {
    grid-column: 2;
    grid-row: 1;
    min-width: 0;
    width: 100%;
    padding-top: 28px;
}
.drocat-labelmapper-row-actions {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
    align-items: center;
    width: 100%;
    max-width: 100%;
    gap: 16px !important;
    min-height: 64px;
    padding: 0;
    margin-bottom: 0;
}
.drocat-labelmapper-query-actions {
    grid-column: 2;
    justify-self: center;
    max-width: 100%;
}
.drocat-labelmapper-group-controls {
    grid-column: 3;
    justify-self: end;
    position: static;
}
.drocat-labelmapper-query-action {
    min-height: 40px;
    padding: 0 16px;
}
.drocat-labelmapper-datasets {
    grid-column: 1 / -1;
    grid-row: 2;
    min-width: 0;
}
.drocat-labelmapper-dataset-row {
    display: grid !important;
    grid-template-columns: minmax(220px, 360px) minmax(0, 1fr);
    column-gap: 24px;
    align-items: start;
    min-width: 0;
    padding: 8px 0 10px;
    border-bottom: 1px solid var(--drocat-line-strong);
}
.drocat-labelmapper-dataset-row:last-child {
    padding-bottom: 0;
    border-bottom: 0;
}
.drocat-labelmapper-dataset-label {
    min-width: 0;
    padding-top: 20px;
    padding-left: 32px;
    color: var(--drocat-navy);
    font-size: 16px;
    line-height: 1.25;
    font-weight: 700;
    overflow-wrap: anywhere;
}
.drocat-labelmapper-dataset-input {
    min-width: 0;
    width: 100%;
}
.drocat-labelmapper-dataset-input .drocat-chip-input-shell,
.drocat-labelmapper-dataset-input .drocat-chip-input {
    min-width: 0;
}
.drocat-labelmapper-dataset-input .drocat-neuron-viewer-card {
    /* Keep the viewer link compact in each mapping row. */
    margin-top: 1px;
}
@media (max-width: 760px) {
    .drocat-settings-mapping-controls > .q-field,
    .drocat-settings-mapping-controls > .q-btn {
        min-width: 100%;
    }
    .drocat-labelmapper-group {
        display: block !important;
        padding: 10px;
    }
    .drocat-labelmapper-group-name {
        width: 100%;
        min-width: 0;
    }
    .drocat-labelmapper-group-members {
        width: 100%;
        margin-top: 14px;
        padding-top: 0;
    }
    .drocat-labelmapper-row-actions {
        grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
        width: 100%;
        max-width: none;
    }
    .drocat-labelmapper-group-controls {
        position: static;
    }
    .drocat-labelmapper-dataset-row {
        grid-template-columns: minmax(140px, 38%) minmax(0, 1fr);
        column-gap: 12px;
    }
    .drocat-labelmapper-dataset-label {
        width: 100%;
        padding-top: 10px;
        padding-left: 16px;
    }
}

/* ---------- Buttons ---------- */
.q-btn { border-radius: var(--drocat-radius-sm); font-weight: 650; }
.q-btn--flat { color: var(--drocat-muted); }
.q-btn--flat:hover { background: var(--drocat-soft) !important; }
.q-btn--unelevated.bg-primary { background: var(--drocat-cobalt) !important; box-shadow: 0 6px 14px rgba(20, 92, 255, .20); }
.q-btn--unelevated.bg-negative { background: var(--drocat-surface) !important; color: var(--drocat-err) !important; border: 1px solid var(--drocat-err-border); }
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

/* Output-directory actions live inside the input append slot.  Give each
   icon a fixed box so the path text never shifts or sits underneath a
   button when the field is resized. */
.drocat-output-dir .q-field__append {
    flex-wrap: nowrap;
    gap: 3px;
    padding-left: 8px;
}
.drocat-output-dir .drocat-dir-icon-btn,
.drocat-output-dir .drocat-dir-reset-btn {
    flex: 0 0 32px;
    width: 32px !important;
    min-width: 32px !important;
    height: 32px !important;
    min-height: 32px !important;
    padding: 0 !important;
    margin: 0;
    align-self: center;
}
.drocat-output-dir .drocat-dir-icon-btn .q-icon,
.drocat-output-dir .drocat-dir-reset-btn .q-icon { font-size: 20px; }

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
    border: 1px solid var(--drocat-line);
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
    border: 1px solid var(--drocat-line);
    min-height: 18px;
    cursor: pointer;
}
.drocat-palette-pick-swatch {
    cursor: pointer;
    transition: box-shadow .15s ease;
}
.drocat-palette-pick-swatch:hover {
    box-shadow: inset 0 0 0 2px var(--drocat-line-strong);
}
.drocat-palette-pick-swatch.selected {
    box-shadow: inset 0 0 0 2px var(--drocat-cobalt);
}
.drocat-select-palette-strip {
    height: 12px;
    border-radius: 4px;
    border: 1px solid var(--drocat-line);
    /* Inside a dropdown option row: fill the width right of the name. */
    flex: 1 1 55%;
    margin-left: 12px;
    min-width: 24px;
}
.drocat-custom-color-row {
    padding: 4px 8px;
    border: 1px solid var(--drocat-line);
    border-radius: 10px;
    background: var(--drocat-surface);
}
.drocat-custom-color-row:hover { background: var(--drocat-soft); }

/* ---------- Layer tree editor (Skeleton tab) ---------- */
.drocat-layer-board { min-height: 8px; }
.drocat-layer-row {
    width: 100%;
    max-width: 100%;
    box-sizing: border-box;
    border: 1px solid var(--drocat-line);
    border-radius: 12px;
    background: var(--drocat-surface);
    padding: 8px 10px;
    cursor: grab;
}
.drocat-layer-row:hover { border-color: var(--drocat-line-strong); }
.drocat-layer-grip { color: var(--drocat-faint); cursor: grab; }
.drocat-layer-neurons {
    width: 100%;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    padding: 6px 2px 2px;
    border-top: 1px dashed var(--drocat-line);
    min-height: 34px;
}
.drocat-layer-chip {
    display: inline-flex;
    align-items: center;
    gap: 2px;
    padding: 3px 4px 3px 10px;
    border: 1px solid var(--drocat-line-strong);
    border-radius: 8px;
    background: var(--drocat-soft);
    color: var(--drocat-navy);
    font-size: 12px;
    font-weight: 600;
    cursor: grab;
}
.drocat-layer-chip:hover { border-color: var(--drocat-cobalt); }
.drocat-layer-chip .q-btn { width: 22px; height: 22px; }
.drocat-layer-neuron-input {
    width: 100%;
    min-width: 0;
}
.drocat-layer-neuron-input .drocat-chip-input-shell,
.drocat-layer-neuron-input .drocat-chip-input,
.drocat-layer-neuron-input .q-field {
    width: 100%;
    min-width: 0;
}
.drocat-layer-neuron-input .q-chip { cursor: grab; }

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
    background: var(--drocat-glass) !important;
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
/* Auto-suggest inputs (pathfinding tabs) replace the native QSelect popup
   with a custom suggestion/history menu; hide the empty native one. */
.drocat-native-popup-hidden { display: none !important; }
/* Arrow-key highlight of the active row in the suggestion/history dropdown. */
.drocat-suggest-menu .q-item.drocat-suggest-active {
    background: var(--drocat-row-hover) !important;
}
/* Dataset provenance tags on history rows. */
.drocat-suggest-menu .drocat-history-dataset-badge {
    font-size: 10px;
    letter-spacing: .02em;
}
"""

@ui.page("/")
def main_page():
    """Main application page with light/dark Photo-Selector-inspired layout."""

    # Global theme
    ui.add_head_html(f"<style>{DROCAT_CSS}</style>")
    # Quasar's anchor helper logs a console error when a q-tooltip's static
    # target id (#cNN) no longer exists. NiceGUI 3.15 binds .tooltip() via
    # an id string, and the first render of each tab panel recreates some
    # inputs with fresh ids, leaving the original tooltip orphaned (it
    # simply never shows — no functional impact). Silence only that exact
    # error; everything else still reaches the console.
    ui.add_head_html(
        "<script>"
        "(function(){"
        "const _err=console.error;"
        "console.error=function(...a){"
        "if(typeof a[0]==='string'&&a[0].startsWith('Anchor: target '))return;"
        "_err.apply(console,a);"
        "};"
        "})();"
        "</script>"
    )

    # Theme: NiceGUI's dark_mode element drives Quasar's body--dark class
    # (all components, popups and menus re-theme with it). The value None
    # follows the OS preference live (Quasar auto mode listens to
    # prefers-color-scheme on macOS and Windows); True/False force
    # dark/light. The persisted preference is restored here so the first
    # paint already uses it.
    dark = ui.dark_mode(value=_saved_dark_mode())

    THEME_OPTIONS = [
        ("System", "brightness_auto", None),
        ("Light", "light_mode", False),
        ("Dark", "dark_mode", True),
    ]

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
            theme_button = ui.button().props("flat").classes(
                "drocat-dark-toggle"
            ).tooltip("Theme")
            with theme_button:
                with ui.element("div").classes("drocat-theme-icon-pair"):
                    ui.icon("light_mode").classes("drocat-theme-sun")
                    ui.element("div").classes("drocat-theme-sep")
                    ui.icon("dark_mode").classes("drocat-theme-moon")
                theme_menu = ui.menu().classes("drocat-theme-menu")
            theme_items = {}
            with theme_menu:
                for label, icon, mode in THEME_OPTIONS:
                    item = ui.item().props("clickable").classes(
                        "drocat-theme-item"
                    )
                    with item:
                        ui.icon(icon).classes("drocat-theme-icon")
                        ui.label(label)
                        ui.space()
                        ui.icon("check").classes("drocat-theme-check")
                    theme_items[mode] = item
                    item.on_click(lambda _event=None, m=mode: _apply_theme(m, persist=True))

            def _apply_theme(mode: bool | None, *, persist: bool) -> None:
                dark.value = mode
                for item_mode, item in theme_items.items():
                    if item_mode is mode:
                        item.classes(add="drocat-theme-item-active")
                    else:
                        item.classes(remove="drocat-theme-item-active")
                # Persist only on user clicks; the initial restore already
                # came from the cookie (writing during build would need the
                # client loop, which UI tests do not run).
                if persist:
                    ui.run_javascript(
                        "document.cookie = 'drocat_dark="
                        + ("dark" if mode is True else "light" if mode is False else "auto")
                        + "; max-age=31536000; path=/; SameSite=Lax'"
                    )

            # Highlight the active choice without touching the browser.
            _apply_theme(dark.value, persist=False)

    # Main content
    with ui.column().classes("w-full drocat-shell gap-3"):
        # Grouped navigation - layered cards. Every group is its own tinted
        # card holding its header on top of its tab segments (no partition,
        # always aligned): Connection blue, Visualization teal, Similarity
        # amber, NeuronBridge purple + NB badge, FlyLight green. Settings is
        # a standalone slate card (no header), separated from the groups.
        NAV_GROUPS = [
            ("Connection", "connection", 4, [
                ("Complete Paths", "route"), ("Shortest Paths", "alt_route"),
                ("Network", "schema"), ("Cross-Dataset", "sync_alt"),
            ]),
            ("Visualization", "visualization", 2, [
                ("Skeleton", "view_in_ar"), ("Net-Viz", "account_tree"),
            ]),
            ("Similarity", "similarity", 3, [
                ("Homologs", "compare"), ("Similar", "science"),
                ("Profiling", "analytics"),
            ]),
            ("NeuronBridge", "nb", 3, [
                ("Find Lines", "biotech"), ("Find Neurons", "search"),
                ("Co-Labeling", "layers"),
            ]),
            ("FlyLight", "flylight", 1, [("Downloader", "download")]),
        ]
        tab_buttons = {}

        def group_tab(label: str, icon: str):
            # Newlines + white-space: pre-line stack multi-word names one
            # word per line (Find Lines -> Find / Lines) with tight leading.
            button = ui.button(label.replace(" ", "\n"), icon=icon).props(
                "flat dense no-caps"
            ).classes("drocat-group-tab")
            button.on_click(lambda _event, name=label: nav_panels.set_value(name))
            tab_buttons[label] = button
            return button

        with ui.element("div").classes("drocat-nav w-full"):
            for group_name, tint, size, tabs in NAV_GROUPS:
                with ui.element("div").classes(
                    f"drocat-group-card drocat-tint-{tint}"
                ).style(f"flex: {size} 1 0"):
                    ui.label(group_name).classes(
                        f"drocat-group-head drocat-head-{tint}"
                    )
                    with ui.element("div").classes("drocat-group-tabs"):
                        for label, icon in tabs:
                            group_tab(label, icon)
            # Standalone Settings card (no header), apart from the groups.
            with ui.element("div").classes(
                "drocat-group-card drocat-tint-settings drocat-settings-card"
            ):
                with ui.element("div").classes("drocat-group-tabs"):
                    group_tab("Settings", "settings")

        with ui.tab_panels(value="Complete Paths").classes("w-full bg-transparent") as nav_panels:
            # Connection
            with ui.tab_panel("Complete Paths").classes("p-0"):
                create_find_path_tab()
            with ui.tab_panel("Shortest Paths").classes("p-0"):
                create_find_shortest_tab()
            with ui.tab_panel("Network").classes("p-0"):
                create_network_tab()
            with ui.tab_panel("Cross-Dataset").classes("p-0"):
                create_inter_dataset_tab()
            # Visualization
            with ui.tab_panel("Skeleton").classes("p-0"):
                create_skeleton_tab()
            with ui.tab_panel("Net-Viz").classes("p-0"):
                create_net_viz_tab()
            # Similarity
            with ui.tab_panel("Homologs").classes("p-0"):
                create_find_homologs_tab()
            with ui.tab_panel("Similar").classes("p-0"):
                create_find_similar_tab()
            with ui.tab_panel("Profiling").classes("p-0"):
                create_connectivity_profiling_tab()
            # NeuronBridge
            with ui.tab_panel("Find Lines").classes("p-0"):
                create_nb_find_lines_tab()
            with ui.tab_panel("Find Neurons").classes("p-0"):
                create_nb_find_neuron_tab()
            with ui.tab_panel("Co-Labeling").classes("p-0"):
                create_nb_colabel_tab()
            # FlyLight
            with ui.tab_panel("Downloader").classes("p-0"):
                create_flylight_tab()
            # Settings
            with ui.tab_panel("Settings").classes("p-0"):
                create_settings_tab()

        def sync_active_tab(value):
            for label, button in tab_buttons.items():
                if label == value:
                    button.classes(add="drocat-active")
                else:
                    button.classes(remove="drocat-active")

        nav_panels.on_value_change(lambda event: sync_active_tab(event.value))
        sync_active_tab(nav_panels.value)

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

    # NiceGUI prints its ready line during the server lifespan startup.  This
    # handler runs immediately afterward, so the launcher output includes a
    # usable browser fallback directly below that line on macOS, Windows, and
    # Linux when automatic browser opening is unavailable or disabled.
    app.on_startup(
        lambda: print(
            f"Tip: If the browser did not open automatically, copy and open "
            f"http://127.0.0.1:{port} in your browser.",
            flush=True,
        )
    )
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
