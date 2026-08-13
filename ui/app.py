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
    background: linear-gradient(135deg, rgba(234, 240, 255, .78), rgba(255, 255, 255, .96));
    box-shadow: 0 4px 14px rgba(11, 31, 58, .06);
}
.drocat-neuron-search-toolbar .drocat-neuron-search-field {
    background: rgba(255, 255, 255, .96);
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
}
.drocat-neuron-match-panel,
.drocat-neuron-full-panel {
    min-width: 0;
    border: 1px solid var(--drocat-line);
    border-radius: 12px;
    padding: 12px;
    background: var(--drocat-surface);
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
    background: linear-gradient(135deg, rgba(232, 241, 255, .78), rgba(255, 255, 255, .92));
}
.drocat-neuron-query-chip {
    display: inline-flex;
    align-items: center;
    min-height: 26px;
    padding: 3px 9px;
    border: 1px solid rgba(69, 126, 191, .38);
    border-radius: 999px;
    background: #fff;
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
    background: #fff;
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
    background: rgba(255, 255, 255, .72);
}
.drocat-neuron-match-table .q-table__middle {
    max-height: 52vh;
    overflow-x: hidden;
    overflow-y: auto;
    margin-top: 8px;
    border: 1px solid var(--drocat-line);
    border-radius: 8px;
    background: rgba(255, 255, 255, .78);
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
    background: #fff8d6;
    color: #946200;
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
    background: #fffbe8;
    color: #a07a2c;
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
    max-height: 52vh;
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
       viewer owns the two intentional states below: white for ordinary
       cells and blue for selected rows. */
    background: #fff !important;
    color: var(--drocat-navy) !important;
}
.drocat-data-viewer-table.q-table__container,
.drocat-data-viewer-table .q-table__middle,
.drocat-data-viewer-table .q-table,
.drocat-data-viewer-table tbody,
.drocat-data-viewer-table tbody tr:not(.drocat-neuron-selected-row) {
    background: #fff !important;
}
.drocat-data-viewer-table tbody tr.q-tr--selected:not(.drocat-neuron-selected-row) > td,
.drocat-data-viewer-table tbody tr[aria-selected="true"]:not(.drocat-neuron-selected-row) > td {
    background: #fff !important;
}
.drocat-data-viewer-table tbody tr:hover > td {
    background: #fff !important;
}
.drocat-data-viewer-table tr.drocat-neuron-selected-row > td {
    /* Direct body-ID selections and matched-group selections share the same
       Quasar row-selection state. Keep that state visibly distinct from the
       yellow cell used for the query match. */
    background: #dcecff !important;
    color: var(--drocat-navy) !important;
}
.drocat-data-viewer-table tr.drocat-neuron-selected-row:hover > td {
    background: #cfe3ff !important;
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
    background: #fff !important;
    color: var(--drocat-navy) !important;
    font-weight: 600;
    box-shadow: inset 0 0 0 2px #eab308;
}
.drocat-data-viewer-table td.drocat-neuron-secondary-hit-cell {
    position: static;
    z-index: 1;
    background: #fff !important;
    color: var(--drocat-navy) !important;
    font-weight: 600;
    box-shadow: inset 0 0 0 2px #eab308;
}
.drocat-data-viewer-table tr.drocat-neuron-selected-row > td.drocat-neuron-hit-cell,
.drocat-data-viewer-table tr.drocat-neuron-selected-row > td.drocat-neuron-secondary-hit-cell {
    background: #dcecff !important;
    color: var(--drocat-navy) !important;
}
.drocat-data-viewer-table mark.drocat-neuron-match-text {
    padding: 0 .08em;
    border-radius: 3px;
    background: #fff1b8;
    color: var(--drocat-navy);
    font-weight: 750;
}
@keyframes drocat-neuron-focus-breathe {
    0%, 100% { outline-color: transparent; outline-width: 0; }
    38%, 62% { outline-color: rgba(100, 110, 125, .72); outline-width: 4px; }
}
.drocat-data-viewer-table tr.drocat-neuron-focus-flash > td {
    outline-style: solid;
    outline-offset: -2px;
    animation: drocat-neuron-focus-breathe 1.35s ease-in-out 1;
}
@media (max-width: 1100px) {
    .drocat-neuron-results-layout {
        grid-template-columns: 1fr;
    }
    .drocat-neuron-match-panel {
        position: static;
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
.drocat-group-tab:hover { background: rgba(255, 255, 255, .55); }
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

/* ---------- Layer tree editor (Skeleton tab) ---------- */
.drocat-layer-board { min-height: 8px; }
.drocat-layer-row {
    border: 1px solid var(--drocat-line);
    border-radius: 12px;
    background: var(--drocat-surface);
    padding: 8px 10px;
    cursor: grab;
}
.drocat-layer-row:hover { border-color: var(--drocat-line-strong); }
.drocat-layer-grip { color: var(--drocat-faint); cursor: grab; }
.drocat-layer-neurons {
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
.drocat-layer-add { max-width: 180px; }

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
/* Auto-suggest inputs (pathfinding tabs) replace the native QSelect popup
   with a custom suggestion/history menu; hide the empty native one. */
.drocat-native-popup-hidden { display: none !important; }
"""

@ui.page("/")
def main_page():
    """Main application page with light Photo-Selector-inspired layout."""

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
        # Grouped navigation - layered cards. Every group is its own tinted
        # card holding its header on top of its tab segments (no partition,
        # always aligned): Connection blue, Visualization teal, Similarity
        # amber, NeuronBridge purple + NB badge, FlyLight green. Settings is
        # a standalone slate card (no header), separated from the groups.
        NAV_GROUPS = [
            ("Connection", "connection", 4, [
                ("Path", "route"), ("Shortest", "alt_route"),
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

        with ui.tab_panels(value="Path").classes("w-full bg-transparent") as nav_panels:
            # Connection
            with ui.tab_panel("Path").classes("p-0"):
                create_find_path_tab()
            with ui.tab_panel("Shortest").classes("p-0"):
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
