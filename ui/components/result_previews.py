"""Compact previews of the three homolog result views.

Every homolog run (Find Homologs tab and Similarity > Connectivity, which
share the same backend) writes three result views:

  - bodyid_results.csv       bodyId-level homologs
  - type_summary.csv         type-mean aggregated FROM the bodyId-level scores
  - type_level_results.csv   true type-level ranking (pooled all-adjacency
                             type profiles scored against every typed target
                             type)

`add_result_previews` renders a top-N preview of each view under the output
panel so all three are always visible after a run.
"""
from pathlib import Path

import pandas as pd
from nicegui import ui

RESULT_VIEWS = [
    ("bodyid_results.csv", "BodyId-level homologs",
     "Per-bodyId matches ranked by similarity."),
    ("type_summary.csv", "Type-mean (from bodyId level)",
     "Mean of the bodyId-level scores per (source, target) type — a "
     "bodyId-level aggregation view, not a type-level profile comparison."),
    ("type_level_results.csv", "Type-level homologs (pooled profiles)",
     "True type-level ranking: pooled all-adjacency type profiles scored "
     "against every typed target type."),
]

PREVIEW_ROWS = 12


def add_result_previews(output_folder, container=None) -> None:
    """Render a top-N preview of each result view under the output panel."""
    folder = Path(output_folder) if output_folder else None
    target = container
    if target is None:
        target = ui.column().classes("w-full")
    with target:
        for fname, title, desc in RESULT_VIEWS:
            path = folder / "results" / fname if folder else None
            exists = path is not None and path.exists()
            with ui.expansion(f"{title} — {fname}", caption=desc).classes(
                    "w-full").props("header-class='text-caption'"):
                if not exists:
                    ui.label("Not available for this run.").classes(
                        "text-caption drocat-muted")
                    continue
                try:
                    df = pd.read_csv(path).head(PREVIEW_ROWS)
                except Exception as e:
                    ui.label(f"Could not read {fname}: {e}").classes(
                        "text-caption drocat-muted")
                    continue
                if df.empty:
                    ui.label("No rows.").classes("text-caption drocat-muted")
                    continue
                columns = [{"name": c, "label": c, "field": c, "align": "left"}
                           for c in df.columns]
                rows = df.where(pd.notna(df), None).to_dict("records")
                ui.table(columns=columns, rows=rows, row_key=None) \
                    .classes("w-full").props("dense flat binary-state-sorting")
                ui.label(f"Top {len(df)} of the run's results — open the file "
                         f"for the full list.").classes(
                    "text-caption drocat-muted")
