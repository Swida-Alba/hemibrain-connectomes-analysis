"""FindSimilar Tab - Morphological and connection-profile similarity search."""

import re
import time

from nicegui import ui

from ..config import DEFAULTS, PROJECT_ROOT, SRC_DIR, SIMILARITY_METRICS
from ..components.common import (
    dataset_selector, neuron_list_input, number_input, select_input,
    checkbox_input, dir_input, section_header, param_grid, tool_page,
    apply_filter_mode,
)
from ..components.output_panel import OutputPanel
from ..components.skeleton_visualization_settings import skeleton_visualization_settings
from ..runner import ScriptRunner
from ..skeleton_pull import SkeletonPuller
from ..type_suggestions import dataset_suggestions
from ..dataset_service import is_banc_dataset

MORPH_METHODS = ["vector", "nblast"]
MORPH_METRICS = ["cosine", "pearson"]
MORPH_LEVELS = ["auto", "bodyid", "type"]


def create_find_similar_tab():
    runner = ScriptRunner()
    output_panel = OutputPanel("Similarity Output")
    skeleton_puller = SkeletonPuller()
    dataset = None
    source_dataset = None

    def _morph_suggest(text):
        dataset_name = dataset.value if dataset is not None else ""
        return dataset_suggestions(text, dataset_name, limit=None)

    def _profile_suggest(text):
        dataset_name = source_dataset.value if source_dataset is not None else ""
        return dataset_suggestions(text, dataset_name, limit=None)

    form_col, results_col = tool_page(
        "Similar Neurons",
        "Find morphologically or connectivity-profile similar neurons.",
        icon="science",
        doc="find_similar.md",
    )

    with form_col:
        mode_value = {"value": "Morphological similarity"}
        with ui.row().classes(
            "w-full items-center justify-between gap-8 px-2"
        ):
            morph_mode_button = ui.button(
                "Morphological similarity",
            ).props("outline no-caps").classes("w-5/12")
            connectivity_mode_button = ui.button(
                "Connectivity similarity",
            ).props("outline no-caps").classes("w-5/12")
            for button in (morph_mode_button, connectivity_mode_button):
                button.style(
                    "min-height: 3.5rem; font-size: 1.1rem; "
                    "font-weight: 700;"
                )

        # ================= Morphological similarity panel =================
        with ui.column().classes("w-full gap-1") as morph_panel:
            with ui.card().classes("w-full drocat-card").props('id="card-findsimilar-morphology-dataset"'):
                section_header("Dataset", "storage")
                dataset = dataset_selector(
                    disable_banc=True,
                    hint="Dataset to search for similar neurons in.",
                )
                morph_output_dir = dir_input(scope="find_similar_morphology")
                morph_dataset_warning = ui.label(
                    "⚠️ BANC morphological similarity is unavailable because "
                    "FlyWire does not provide BANC skeletons. Select a non-BANC dataset."
                ).classes("text-caption text-amber-8").set_visibility(False)

            with ui.card().classes("w-full drocat-card").props('id="card-findsimilar-morphology-neurons"'):
                section_header("Query", "search")
                query_input = neuron_list_input(
                    label="Query Neuron(s)",
                    placeholder="Type or upload CSV/TSV/Excel (e.g., aMe12, 1005174948)",
                    hint="Neuron types, bodyIds, or patterns. Multiple queries "
                         "are searched independently and saved as separate runs.",
                    suggestions=_morph_suggest,
                    available_neurons=lambda: dataset.value
                    if dataset is not None else "",
                ).classes("drocat-fixed-neuron-input")

            with ui.card().classes("w-full drocat-card"):
                section_header("Similarity Parameters", "tune")
                with param_grid(3):
                    level = select_input(
                        "Level", MORPH_LEVELS, DEFAULTS["morph_level"],
                        hint="'auto' (recommended): a type query returns "
                             "type-to-type results, a bodyId query returns "
                             "bodyId-to-bodyId results. 'bodyid': rank "
                             "individual neurons. 'type': aggregate candidates "
                             "by neuron type.",
                    )
                    method = select_input(
                        "Method", MORPH_METHODS, DEFAULTS["morph_method"],
                        hint="'vector': fast morphometrics + persistence vectors "
                             "(recommended). 'nblast': canonical NBLAST (slower; "
                             "runs on vector-prefiltered candidates).",
                    )
                    metric = select_input(
                        "Metric", MORPH_METRICS, DEFAULTS["morph_metric"],
                        hint="Similarity on standardized vectors: cosine or Pearson.",
                    )
                with param_grid(3):
                    morph_top_n = number_input(
                        "Top N Results", DEFAULTS["morph_top_n"], 5, 200,
                        hint="Number of ranked results to return.",
                    )
                    nblast_prefilter = number_input(
                        "NBLAST Prefilter", DEFAULTS["nblast_prefilter"], 10, 5000,
                        hint="Vector-prefiltered candidate count before NBLAST "
                             "scoring (higher = slower but more thorough).",
                    )
                    n_per_type = number_input(
                        "BodyIds / Type", DEFAULTS["n_per_type"], 1, 50,
                        hint="Per-type bodyId cap for NBLAST type-level means.",
                    )
                with param_grid(2):
                    candidate_source = select_input(
                        "Candidate Source", ["auto", "cache", "profile"],
                        DEFAULTS["candidate_source"],
                        hint="'auto': NeuPrint datasets use the connectivity-"
                             "expanded search (candidates from the connection "
                             "cache, top-N×3 similar types expanded to all "
                             "their members, then morphology); FlyWire uses "
                             "the full-morphology vector-cache search. "
                             "'profile': always connectivity-expanded. "
                             "'cache': full-morphology over the local skeleton "
                             "population (download skeletons first).",
                    )
                    candidate_expansion = number_input(
                        "Candidate Expansion (×)", DEFAULTS["morph_candidate_expansion"], 1, 20,
                        hint="Connectivity-expanded search: keep the top-N × "
                             "expansion connectivity-similar TYPES from the "
                             "connection cache, then expand to ALL their member "
                             "bodyIds. Skeletons are fetched transiently "
                             "(never written to the cache); cached skeletons "
                             "are reused.",
                    )
                roi_filter = select_input(
                    "ROI Filter", ["All ROIs"], "All ROIs",
                    hint="Restrict candidate discovery to synapse rows in "
                         "the selected ROIs (only shown when the dataset's "
                         "connection cache carries ROI data). 'All ROIs' = "
                         "no restriction.",
                )
                with ui.row().classes("w-full items-center gap-4"):
                    visualize = checkbox_input(
                        "Visualize Top Results",
                        DEFAULTS["morph_visualize_top_n"] > 0,
                        hint="Render optional 3D skeletons for the highest-ranked results.",
                    )
                    visualization_settings = skeleton_visualization_settings(
                        default_top_n=DEFAULTS["morph_visualize_top_n"],
                        top_n_label="Visualize Top N Types / Neurons",
                        top_n_hint=(
                            "Number of top results to render. The grouping choice "
                            "controls whether types or individual bodyIds are shown."
                        ),
                        default_visualize_by=DEFAULTS["morph_visualize_by"],
                        dataset_provider=lambda: dataset.value,
                        dataset_watchers=[dataset],
                    )

            with ui.card().classes("w-full drocat-card"):
                section_header("Skeleton Vector Cache", "memory")
                coverage_label = ui.label("Coverage: checking...").classes(
                    "text-caption drocat-muted"
                )
                with ui.row().classes("items-center gap-2"):
                    build_button = ui.button(
                        "Build Vector Cache", icon="auto_awesome"
                    ).props("color=secondary outline").on_click(
                        lambda: build_cache()
                    )
                    ui.label(
                        "One-time build from cached skeletons (auto-triggered on "
                        "first query); incremental afterwards."
                    ).classes("text-caption drocat-muted")

                # --- full-morphology skeleton download (like Settings pull) ---
                download_label = ui.label("").classes("text-caption drocat-muted")
                download_bar = ui.linear_progress(value=0, show_value=False).classes(
                    "w-full"
                ).set_visibility(False)
                with ui.row().classes("items-center gap-2"):
                    download_button = ui.button(
                        "Download All Skeletons", icon="download"
                    ).props("color=secondary outline")
                    cancel_download_button = ui.button(
                        "Cancel", icon="stop"
                    ).props("flat dense").set_visibility(False)
                    ui.label(
                        "Full-morphology mode: fetch every missing skeleton to "
                        "the local cache (resumable, persists for reuse). "
                        "FlyWire datasets already have bulk skeletons locally."
                    ).classes("text-caption drocat-muted")
                with ui.row().classes("items-center gap-2"):
                    cache_fetched = checkbox_input(
                        "Cache Fetched Skeletons", False,
                        hint="Save skeletons fetched transiently during a search "
                             "(profile-first candidate fetches / NBLAST) to the "
                             "local skeleton cache so later runs reuse them. "
                             "Off by default: fetched skeletons stay in memory "
                             "only (their VECTORS are always cached regardless).",
                    )
                    ui.label(
                        "For a persistent full-morphology pool, prefer 'Download All Skeletons'."
                    ).classes("text-caption drocat-muted")

            def refresh_roi_options():
                # ROI data availability differs per dataset (male-cns has 114
                # ROIs; hemibrain's connection cache has none).
                try:
                    from pathlib import Path
                    import polars as pl
                    conn_path = (Path(PROJECT_ROOT) / "cache"
                                 / dataset.value.replace(":", "_").replace(".", "_")
                                 / "connections.parquet")
                    rois = ["All ROIs"]
                    if conn_path.exists():
                        conn = pl.read_parquet(conn_path)
                        if "roi" in conn.columns:
                            vals = (conn["roi"].drop_nulls()
                                    .filter(pl.col("roi") != "")
                                    .unique().sort().to_list())
                            if vals:
                                rois = ["All ROIs"] + [str(v) for v in vals]
                    roi_filter.options = rois
                    if roi_filter.value not in rois:
                        roi_filter.value = "All ROIs"
                    roi_filter.set_visibility(len(rois) > 1)
                except Exception:
                    roi_filter.set_visibility(False)

            def refresh_download_ui():
                st = skeleton_puller.state
                download_button.set_visibility(not st["running"])
                cancel_download_button.set_visibility(st["running"])
                download_bar.set_visibility(st["running"] and st["total"] > 0)
                if st["running"] and st["total"] > 0:
                    download_bar.value = st["current"] / max(1, st["total"])
                    eta = ""
                    if st["fetch_started_at"] and st["current"] > 0:
                        elapsed = max(0.001, time.time() - st["fetch_started_at"])
                        rate = st["current"] / elapsed
                        remaining = (st["total"] - st["current"]) / rate
                        eta = f" · ETA {int(remaining // 60)}m {int(remaining % 60):02d}s"
                    download_label.text = f"{st['info']}{eta}"
                elif st["done"]:
                    s = st.get("summary") or {}
                    if st["error"]:
                        download_label.text = f"Download failed: {st['error']}"
                        ui.notify(f"Skeleton download failed: {st['error']}", type="negative")
                    else:
                        download_label.text = (
                            f"Download {st['info']} {s.get('fetched', 0)}/"
                            f"{s.get('total', 0)} fetched, {s.get('errors', 0)} errors"
                        )
                        if s.get("fetched", 0) > 0:
                            ui.notify(
                                f"{s.get('fetched')} skeletons downloaded "
                                f"({s.get('skipped_existing', 0)} already cached)",
                                type="positive",
                            )
                            refresh_coverage()
                else:
                    download_label.text = ""

            def start_download():
                if is_banc_dataset(dataset.value):
                    morph_dataset_warning.set_visibility(True)
                    ui.notify(
                        "BANC skeleton downloads are unavailable; select a non-BANC dataset.",
                        type="warning",
                    )
                    return
                ok = skeleton_puller.start(dataset.value)
                if not ok:
                    ui.notify("A skeleton download is already running", type="warning")
                refresh_download_ui()

            def stop_download():
                skeleton_puller.cancel()
                download_label.text = "Cancelling after the current batch..."

            download_button.on_click(start_download)
            cancel_download_button.on_click(stop_download)

            def refresh_coverage():
                # Lightweight: no navis/statvis import at page build (they are
                # heavy and would slow the first page response). The vector
                # cache is a plain parquet file, countable with polars.
                try:
                    from pathlib import Path
                    folder = (Path(PROJECT_ROOT) / "cache"
                              / dataset.value.replace(":", "_").replace(".", "_"))
                    # recursive: FlyWire bulk caches live in nested folders
                    n_skel = len(list((folder / "skeletons").rglob("*.pkl")))
                    n_vec = 0
                    vec_file = folder / "morphology" / "skeleton_vectors.parquet"
                    if vec_file.exists():
                        import polars as pl
                        n_vec = pl.read_parquet(vec_file).height
                    coverage_label.text = (
                        f"Dataset skeletons: {n_skel}  ·  vectorized: {n_vec}"
                    )
                except Exception:
                    coverage_label.text = "Coverage unavailable."

            async def build_cache():
                if is_banc_dataset(dataset.value):
                    morph_dataset_warning.set_visibility(True)
                    ui.notify(
                        "BANC morphological similarity is unavailable; select a non-BANC dataset.",
                        type="warning",
                    )
                    return
                build_button.disable()
                ui.notify(
                    "Building skeleton vector cache (this can take a few minutes)...",
                    type="info",
                )
                try:
                    import asyncio
                    import sys
                    sys.path.insert(0, str(SRC_DIR))
                    from morphology import SkeletonVectorCache

                    def _run():
                        cache = SkeletonVectorCache(
                            dataset.value, n_workers=8, verbose=False
                        )
                        return cache.build(fetch_missing=0)

                    stats = await asyncio.to_thread(_run)
                    ui.notify(
                        f"Vector cache ready: {stats['rows']} rows "
                        f"({stats['new']} new)"
                    )
                    refresh_coverage()
                except Exception as ex:
                    ui.notify(f"Cache build failed: {ex}", type="negative")
                finally:
                    build_button.enable()

        # ============ Connectivity similarity panel ============
        with ui.column().classes("w-full gap-1") as profile_panel:
            with ui.card().classes("w-full drocat-card").props('id="card-findsimilar-profile-dataset"'):
                section_header("Dataset", "storage")
                source_dataset = dataset_selector(
                    label="Dataset",
                    hint="Dataset used for both the query and candidate search. "
                         "Connectivity similarity is intra-dataset only.",
                )
                profile_output_dir = dir_input(scope="find_similar_profiling")

            with ui.card().classes("w-full drocat-card").props('id="card-findsimilar-profile-neurons"'):
                section_header("Query", "search")
                source_input = neuron_list_input(
                    label="Query Neuron (type or bodyId)",
                    show_filter=False,
                    show_upload=True,
                    hint="Enter one or more neuron types or bodyIds. Each input is searched independently.",
                    suggestions=_profile_suggest,
                    available_neurons=lambda: source_dataset.value
                    if source_dataset is not None else "",
                ).classes("drocat-fixed-neuron-input")

            with ui.card().classes("w-full drocat-card"):
                section_header("Search Parameters", "tune")
                with param_grid(3):
                    profile_top_n = number_input(
                        "Top N Candidates", DEFAULTS["top_n"], 5, 100,
                        hint="Number of top candidates to return.",
                    )
                    min_shared_partners = number_input(
                        "Min Shared Partners", 2, 1, 10,
                        hint="Minimum shared partners for a candidate (adjacency "
                             "expansion). Lower = looser discovery (1 = any shared "
                             "partner makes a candidate).",
                    )
                    candidate_prune = number_input(
                        "Candidate Prune %", 5, 5, 100,
                        hint="Keep the top N% of cosine-positive candidates after "
                             "vector pre-filtering. 100 = keep all (loosest search).",
                    )
                with param_grid(2):
                    similarity_metric = select_input(
                        "Similarity Metric", SIMILARITY_METRICS,
                        DEFAULTS["similarity_metric"],
                        hint="Metric for comparing connectivity profiles.",
                    )
                    top_k = number_input(
                        "Top K Partners", DEFAULTS["top_k"], 5, 50,
                        hint="Top K partners per direction for profile construction.",
                    )
                with ui.row().classes("gap-4"):
                    use_fast = checkbox_input(
                        "Fast Search", True,
                        hint="Adjacency-expansion discovery (recommended).",
                    )
                    vector_prefilter = checkbox_input(
                        "Vector Pre-filtering", True,
                        hint="Cosine pre-filter of candidates for speed.",
                    )
                    expand_2hop = checkbox_input(
                        "2-Hop Expansion", True,
                        hint="Include untyped partners via 2-hop typed partners.",
                    )
                    use_cache = checkbox_input(
                        "Use Cache", True,
                        hint="Cache profiles and connections locally.",
                    )
                saveas = ui.input(
                    label="Save Folder Name (optional)",
                    placeholder="e.g., aMe12_similar",
                ).classes("w-full drocat-input").tooltip(
                    "Custom output folder name. Leave empty for the auto name."
                )
                full_cache = checkbox_input(
                    "Pre-build Full Dataset Cache", False,
                    hint="Fetch connections for EVERY uncached neuron before "
                         "searching. Very slow on first use (can take hours).",
                )
                with ui.row().classes("w-full items-center gap-4"):
                    profile_visualize = checkbox_input(
                        "Visualize Top Candidates",
                        True,
                        hint="Generate a separate 3D visualization list for the "
                             "highest-ranked connectivity-similar candidates.",
                    )
                    profile_visualization_settings = skeleton_visualization_settings(
                        default_top_n=5,
                        top_n_label="Visualize Top N Candidates",
                        top_n_hint=(
                            "Number of top connectivity-similar candidates to "
                            "render. This list is independent of morphological "
                            "similarity results."
                        ),
                        default_visualize_by="type",
                        dataset_provider=lambda: [source_dataset.value],
                        dataset_watchers=[source_dataset],
                    )

        def sync_mode():
            is_morph = mode_value["value"] == "Morphological similarity"
            morph_panel.set_visibility(is_morph)
            profile_panel.set_visibility(not is_morph)
            morph_mode_button.props(
                "color=primary" if is_morph else "color=grey-7"
            )
            connectivity_mode_button.props(
                "color=grey-7" if is_morph else "color=primary"
            )

        def set_mode(value: str):
            mode_value["value"] = value
            sync_mode()

        def on_dataset_change(_e=None):
            morph_dataset_warning.set_visibility(is_banc_dataset(dataset.value))
            refresh_coverage()
            refresh_roi_options()

        morph_mode_button.on_click(
            lambda _event: set_mode("Morphological similarity")
        )
        connectivity_mode_button.on_click(
            lambda _event: set_mode("Connectivity similarity")
        )
        dataset.on_value_change(on_dataset_change)
        sync_mode()
        on_dataset_change()
        ui.timer(0.5, refresh_download_ui)

    with results_col:
        output_panel.create(run_label="Run Similarity Search", run_icon="play_arrow")

    def _unique_queries(values):
        """Return the entered queries in order, without duplicate chips."""
        queries = []
        seen = set()
        for value in values or []:
            text = str(value).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            queries.append(value)
        return queries

    def _saveas_for_query(base_value, index, query, total):
        """Keep custom-named multi-query runs separate on disk."""
        base = str(base_value or "").strip()
        if not base or total == 1:
            return base
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(query)).strip("_")
        safe = safe[:60] or f"query_{index + 1}"
        return f"{base}_{index + 1}_{safe}"

    def _collect_files(results):
        """Merge per-query output files while preserving their paths."""
        files = {}
        for result in results:
            for file_info in result.get("files", []):
                path = file_info.get("path")
                if path:
                    files[path] = file_info
        return list(files.values())

    async def run_morphological():
        if is_banc_dataset(dataset.value):
            morph_dataset_warning.set_visibility(True)
            ui.notify(
                "BANC morphological similarity is unavailable; select a non-BANC dataset.",
                type="warning",
            )
            return
        mode, neurons = query_input.get_value()
        queries = _unique_queries(apply_filter_mode(neurons, mode))
        if not queries:
            ui.notify("Please enter at least one query neuron", type="warning")
            return
        if len(queries) > 50:
            ui.notify("Please limit the query to 50 neurons", type="warning")
            return

        output_panel.clear()
        output_panel.set_running(True)

        visualization_values = visualization_settings.values()
        base_params = {
            "dataset": dataset.value,
            "level": level.value,
            "method": method.value,
            "metric": metric.value,
            "top_n": int(morph_top_n.value),
            "nblast_prefilter": int(nblast_prefilter.value),
            "n_per_type": int(n_per_type.value),
            "candidate_source": candidate_source.value,
            "candidate_expansion": int(candidate_expansion.value),
            "roi_filter": None if roi_filter.value == "All ROIs" else [roi_filter.value],
            "visualize_top_n": (
                visualization_values["visualize_top_n"] if visualize.value else 0
            ),
            "visualize_by": visualization_values["visualize_by"],
            "visualization_settings": visualization_values,
            "output_dir": morph_output_dir.value,
            "saveas": "",
            "verbose": True,
            "n_workers": 8,
            "use_cache": True,
            "cache_fetched_skeletons": cache_fetched.value,
        }
        results = []
        last_output_folder = None
        try:
            for index, query in enumerate(queries):
                if len(queries) > 1:
                    output_panel.log(
                        f"--- Morphology query {index + 1}/{len(queries)}: {query} ---",
                        "system",
                    )
                constructor_params = dict(base_params)
                constructor_params["query"] = query
                result = await output_panel.run(
                    runner, "find_similar_morphology", constructor_params,
                    "find_similar", output_dir=morph_output_dir.value,
                )
                results.append(result)
                last_output_folder = result.get("output_folder") or last_output_folder
                if result.get("cancelled"):
                    break

            cancelled = any(result.get("cancelled") for result in results)
            succeeded = bool(results) and all(
                result.get("returncode") == 0 for result in results
            )
            if cancelled:
                output_panel.set_status("Cancelled", "red")
            else:
                output_panel.set_status(
                    "Completed" if succeeded else "Failed",
                    "green" if succeeded else "red",
                )
            files = _collect_files(results)
            if files:
                output_panel.show_files(
                    files,
                    morph_output_dir.value if len(queries) > 1
                    else last_output_folder or morph_output_dir.value,
                )
        finally:
            output_panel.set_running(False)

    async def run_profile():
        source_vals = source_input.get_value()[1]
        sources = _unique_queries(source_vals)
        if not sources:
            ui.notify("Please enter at least one query neuron", type="warning")
            return

        output_panel.clear()
        output_panel.set_running(True)

        visualization_values = profile_visualization_settings.values()
        base_params = {
            "source_dataset": source_dataset.value,
            "target_dataset": source_dataset.value,
            "output_dir": profile_output_dir.value,
            "top_n": int(profile_top_n.value),
            "top_k": int(top_k.value),
            "min_shared_partners": int(min_shared_partners.value),
            "vector_prune_fraction": float(candidate_prune.value) / 100.0,
            "similarity_metric": similarity_metric.value,
            "vector_prefiltering": vector_prefilter.value,
            "include_untyped_partners": expand_2hop.value,
            "use_cache": use_cache.value,
            "saveas": saveas.value.strip() or "",
            "min_synapse_threshold": 3,
            "ensure_cache_complete": full_cache.value,
            "morphological_enrichment": True,
            "output_folder_prefix": "similar-connectivity",
            "visualize_skeleton": profile_visualize.value,
            "visualize_top_n": (
                visualization_values["visualize_top_n"]
                if profile_visualize.value else 0
            ),
            "visualization_settings": visualization_values,
            "verbose": True,
        }
        method_name = "find_homologs_fast" if use_fast.value else "find_novel_homologs"
        results = []
        last_output_folder = None
        try:
            for index, source in enumerate(sources):
                if len(sources) > 1:
                    output_panel.log(
                        f"--- Connectivity query {index + 1}/{len(sources)}: {source} ---",
                        "system",
                    )
                constructor_params = dict(base_params)
                constructor_params.update({
                    "source": source,
                    "saveas": _saveas_for_query(saveas.value, index, source, len(sources)),
                })
                result = await output_panel.run(
                    runner, "find_similar_profile", constructor_params,
                    method_name, output_dir=profile_output_dir.value,
                )
                results.append(result)
                last_output_folder = result.get("output_folder") or last_output_folder
                if result.get("cancelled"):
                    break

            cancelled = any(result.get("cancelled") for result in results)
            succeeded = bool(results) and all(
                result.get("returncode") == 0 for result in results
            )
            if cancelled:
                output_panel.set_status("Cancelled", "red")
            else:
                output_panel.set_status(
                    "Completed" if succeeded else "Failed",
                    "green" if succeeded else "red",
                )
            files = _collect_files(results)
            if files:
                output_panel.show_files(
                    files,
                    profile_output_dir.value if len(sources) > 1
                    else last_output_folder or profile_output_dir.value,
                )
        finally:
            output_panel.set_running(False)

    async def run_similar():
        if mode_value["value"] == "Morphological similarity":
            await run_morphological()
        else:
            await run_profile()

    output_panel.run_button.on_click(run_similar)
    output_panel.cancel_button.on_click(runner.cancel)
