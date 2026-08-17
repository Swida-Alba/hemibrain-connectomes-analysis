"""Settings Tab - Token configuration, dataset status, and app settings."""

from nicegui import run, ui

from ..config import (
    DATASETS,
    DEFAULT_OUTPUT_DIR,
    APP_DOCS_BRANCH,
    APP_DOCS_URL,
    APP_GITHUB_URL,
    APP_VERSION,
    PROJECT_ROOT,
    clear_tab_output_overrides,
    get_auto_suggest_enabled,
    get_default_output_dir,
    set_auto_suggest_enabled,
    set_default_output_dir,
)
from ..components.common import (
    dataset_multi_selector,
    dataset_status_card,
    dir_input,
    refresh_dataset_selector_statuses,
    section_header,
    sync_output_dir_fields,
)
from ..components.custom_grouper import to_canonical_dict
from ..components.mapping_editor import custom_grouping_block
from ..dataset_service import get_dataset_service
from .. import mapping_store


def create_settings_tab():
    with ui.column().classes("w-full drocat-page gap-3"):
        with ui.row().classes("w-full items-center gap-3 drocat-page-head"):
            with ui.element("div").classes("drocat-page-mark"):
                ui.icon("settings").classes("text-white")
            with ui.column().classes("gap-1"):
                with ui.row().classes("items-center gap-2"):
                    ui.label("Settings").classes("drocat-page-title")
                    ui.link(
                        "Instructions",
                        "docs/ui_guides/settings.html",
                    ).classes("drocat-doc-link")
                ui.label("Configure API tokens, view dataset status, and manage preferences.").classes("drocat-page-sub")

        # Skeleton downloads always populate the shared raw .swc.gz cache.
        from ..dataset_pull import DatasetPuller
        from ..skeleton_pull import SkeletonPuller

        puller = DatasetPuller()
        skeleton_puller = SkeletonPuller()

        # Dataset Status. The card already owns its title; adding a second
        # section header here made the Settings page look like it contained
        # two different availability controls.
        dataset_status_card()

        with ui.card().classes("w-full drocat-card"):
            section_header("Dataset Cache", "download")
            ui.label(
                "Pull a full dataset to local and build the indexed connection cache "
                "(cache/<dataset>/connections.parquet + neuron_indexes/<dataset>/"
                "neuron_index.parquet). Re-running "
                "resumes interrupted builds from their checkpoint; 'Force rebuild' clears "
                "a broken cache first and rebuilds it completely. 'Pull Complete Connections' "
                "is an explicit connection-cache action for the selected dataset; it uses "
                "the same resumable, batched builder and shared connections.parquet cache."
            ).classes("text-caption drocat-muted")

            def _format_eta(st) -> str:
                """Human-readable remaining-time estimate from fetch progress."""
                fetch_started = st.get("fetch_started_at")
                current, total = st.get("current", 0), st.get("total", 0)
                if not fetch_started or current <= 0 or total <= 0 or current >= total:
                    return "ETA --"
                import time as _time
                elapsed = max(_time.time() - fetch_started, 1e-3)
                rate = current / elapsed  # neurons per second
                remaining = (total - current) / rate
                m, s = divmod(int(remaining), 60)
                h, m = divmod(m, 60)
                if h:
                    return f"ETA ~{h}h {m:02d}m"
                return f"ETA ~{m}m {s:02d}s"

            with ui.row().classes("items-center gap-3 w-full").style("flex-wrap: wrap"):
                ds_select = ui.select(
                    options=DATASETS, value=DATASETS[0], label="Dataset"
                ).props("outlined").classes("drocat-select").style("min-width: 260px")
                force_rebuild = ui.checkbox(
                    "Force rebuild (clear broken cache first)"
                ).tooltip(
                    "Deletes the existing cache/<dataset>/connections.parquet "
                    "before fetching, rebuilding the connection cache from scratch. "
                    "The neuron index in neuron_indexes/ survives with reset progress "
                    "flags. Unchecked, the "
                    "pull only fetches the neurons missing from the cache (resume)."
                )
                batch_input = ui.number(
                    label="Batch size", value=100, min=10, max=5000, step=10
                ).classes("drocat-input").style("width: 150px").tooltip(
                    "Neurons fetched per batch (default 100)."
                )
                parallel_input = ui.number(
                    label="Parallel workers", value=4, min=1, max=32, step=1
                ).classes("drocat-input").style("width: 150px").tooltip(
                    "Batches fetched concurrently (1 = sequential). Only raise if the "
                    "NeuPrint/FlyLight server tolerates parallel requests."
                )

            with ui.row().classes("items-center gap-2").style("flex-wrap: wrap"):
                run_btn = ui.button("Pull Full Dataset", icon="download", color="primary")
                connection_run_btn = ui.button(
                    "Pull Complete Connections", icon="hub", color="secondary"
                ).props("outline")
                skeleton_run_btn = ui.button(
                    "Download All Skeletons", icon="download", color="secondary"
                ).props("outline").tooltip(
                    "Download missing raw skeletons as reusable .swc.gz files "
                    "using the Dataset and Parallel workers selected above."
                )
                cancel_btn = ui.button("Cancel", icon="stop", color="negative").props("outline")
                cancel_btn.set_enabled(False)

            progress = ui.linear_progress(value=0).props("instant-feedback").classes("w-full")
            status_label = ui.label("Idle").classes("text-caption drocat-muted")
            result_label = ui.label("").classes("text-caption")
            pull_done_synced = {"value": False}

            def refresh_pull_state():
                st = puller.state
                skeleton_running = skeleton_puller.running
                if (
                    not st["running"]
                    and run_btn.enabled
                    and connection_run_btn.enabled
                    and not skeleton_running
                ):
                    return
                run_btn.set_enabled(not st["running"] and not skeleton_running)
                connection_run_btn.set_enabled(not st["running"] and not skeleton_running)
                cancel_btn.set_enabled(st["running"] or skeleton_running)
                ds_select.set_enabled(not st["running"] and not skeleton_running)
                force_rebuild.set_enabled(not st["running"] and not skeleton_running)
                batch_input.set_enabled(not st["running"] and not skeleton_running)
                parallel_input.set_enabled(not st["running"] and not skeleton_running)
                if st["running"]:
                    pull_done_synced["value"] = False
                    total = st["total"] or 1
                    frac = min(st["current"] / total, 1.0)
                    progress.set_value(frac)
                    operation_label = (
                        "complete connections"
                        if st.get("operation") == "connections"
                        else "full dataset"
                    )
                    status_label.text = (
                        f"Pulling {operation_label} for {st['dataset']}: {st['info']} "
                        f"({st['current']:,}/{st['total']:,} neurons, "
                        f"{frac * 100:.2f}%) | {_format_eta(st)}"
                    )
                    return
                if not st["done"]:
                    return
                if not pull_done_synced["value"]:
                    # A pull creates the local cache files directly. Refresh
                    # selector labels immediately so they do not wait for a
                    # full page rebuild (or a separate availability refresh).
                    refresh_dataset_selector_statuses()
                    pull_done_synced["value"] = True
                if st["error"]:
                    status_label.text = "Failed"
                    result_label.text = f"❌ {st['error']}"
                else:
                    s = st["summary"] or {}
                    operation_label = (
                        "Complete connection pull"
                        if st.get("operation") == "connections"
                        else "Full dataset pull"
                    )
                    head = (
                        "⏹ Cancelled - fetched batches consolidated; re-run to resume."
                        if st["cancelled"]
                        else f"✅ {operation_label} complete."
                    )
                    result_label.text = (
                        f"{head} Target: {s.get('total_neurons', 0):,} | "
                        f"newly cached: {s.get('newly_cached', 0):,} | "
                        f"already cached: {s.get('already_cached', 0):,} | "
                        f"failed: {len(s.get('failed_neurons', [])):,} | "
                        f"connections: {s.get('total_connections', 0):,} | "
                        f"{s.get('elapsed_time', 0):.1f}s"
                    )
                    status_label.text = "Idle"

            def _start_pull(operation: str):
                ok = puller.start(
                    str(ds_select.value),
                    force_rebuild=force_rebuild.value,
                    batch_size=int(batch_input.value or 100),
                    max_workers=int(parallel_input.value or 1),
                    operation=operation,
                )
                if ok:
                    run_btn.set_enabled(False)
                    connection_run_btn.set_enabled(False)
                    skeleton_run_btn.set_enabled(False)
                    cancel_btn.set_enabled(True)
                    result_label.text = ""
                    progress.set_value(0)
                    pull_done_synced["value"] = False
                else:
                    ui.notify("A dataset pull is already running", type="warning")

            def start_pull():
                _start_pull("full_dataset")

            def start_connection_pull():
                _start_pull("connections")

            def stop_pull():
                if puller.running:
                    puller.cancel()
                    status_label.text = "Cancelling after the current batch..."
                if skeleton_puller.running:
                    skeleton_puller.cancel()
                    skeleton_status.text = "Cancelling after the current batch..."

            run_btn.on_click(start_pull)
            connection_run_btn.on_click(start_connection_pull)
            cancel_btn.on_click(stop_pull)
            ui.timer(0.5, refresh_pull_state)

            skeleton_progress = ui.linear_progress(value=0).props(
                "instant-feedback"
            ).classes("w-full")
            skeleton_status = ui.label("Idle").classes("text-caption drocat-muted")
            skeleton_result = ui.label("").classes("text-caption")
            skeleton_done_synced = {"value": False}

            def refresh_skeleton_pull_state():
                st = skeleton_puller.state
                running = bool(st["running"])
                dataset_running = puller.running
                skeleton_run_btn.set_enabled(not running and not dataset_running)
                cancel_btn.set_enabled(running or dataset_running)
                # The dataset selector is shared with both dataset-cache pulls.
                ds_select.set_enabled(not running and not puller.running)
                if running:
                    skeleton_done_synced["value"] = False
                    total = st["total"] or 1
                    frac = min(st["current"] / total, 1.0)
                    skeleton_progress.set_value(frac)
                    skeleton_status.text = (
                        f"Pulling {st['dataset']}: "
                        f"{st['info']} ({st['current']:,}/{st['total']:,}) | "
                        f"{_format_eta(st)}"
                    )
                    return
                if not st["done"]:
                    return
                if not skeleton_done_synced["value"]:
                    refresh_dataset_selector_statuses()
                    skeleton_done_synced["value"] = True
                if st["error"]:
                    skeleton_status.text = "Failed"
                    skeleton_result.text = f"❌ {st['error']}"
                else:
                    summary = st.get("summary") or {}
                    head = "⏹ Cancelled." if st["cancelled"] else "✅ Skeleton pull complete."
                    skeleton_result.text = (
                        f"{head} "
                        f"fetched: {summary.get('fetched', 0):,}; "
                        f"already available: {summary.get('skipped_existing', 0):,}; "
                        f"errors: {summary.get('errors', 0):,}"
                    )
                    skeleton_status.text = "Idle"

            def start_skeleton_pull():
                if puller.running:
                    ui.notify("Finish the full dataset pull before downloading skeletons", type="warning")
                    return
                ok = skeleton_puller.start(
                    str(ds_select.value),
                    max_workers=int(parallel_input.value or 1),
                    mode="raw",
                )
                if ok:
                    run_btn.set_enabled(False)
                    connection_run_btn.set_enabled(False)
                    skeleton_run_btn.set_enabled(False)
                    cancel_btn.set_enabled(True)
                    skeleton_result.text = ""
                    skeleton_progress.set_value(0)
                    skeleton_done_synced["value"] = False
                else:
                    ui.notify("A skeleton pull is already running", type="warning")

            skeleton_run_btn.on_click(start_skeleton_pull)
            ui.timer(0.5, refresh_skeleton_pull_state)

        # Tokens
        with ui.card().classes("w-full drocat-card"):
            section_header("API Tokens", "key")

            existing_tokens = _load_tokens()
            token_state = {
                "neuprint": existing_tokens.get("neuprint", ""),
                "cave": existing_tokens.get("cave", ""),
            }

            # Reminder when tokens are missing: the NeuPrint token is
            # required for NeuPrint datasets; the CAVE token is optional and
            # only needed for FlyWire FAFB online fetching. Refreshed
            # whenever the saved tokens change.
            token_reminder = ui.element("div").props('id="drocat-token-reminder"').classes(
                "w-full drocat-token-reminder"
            )
            with token_reminder:
                token_reminder_text = ui.label("").classes("text-sm drocat-warn")

            def _refresh_token_reminder():
                missing = []
                if not token_state.get("neuprint"):
                    missing.append("neuprint")
                if not token_state.get("cave"):
                    missing.append("cave")
                if not missing:
                    token_reminder.set_visibility(False)
                    return
                token_reminder.set_visibility(True)
                if missing == ["neuprint"]:
                    token_reminder_text.text = (
                        "⚠️ NeuPrint token not configured - it is required for NeuPrint datasets. "
                        "Set it below or in token_info_local.txt."
                    )
                elif missing == ["cave"]:
                    token_reminder_text.text = (
                        "ℹ️ CAVE token not configured - optional; it is only needed for "
                        "FlyWire FAFB online fetching."
                    )
                else:
                    token_reminder_text.text = (
                        "⚠️ No API tokens configured. The NeuPrint token is required for NeuPrint "
                        "datasets; the CAVE token is optional (only needed for FlyWire FAFB online "
                        "fetching). Set them below or in token_info_local.txt."
                    )
                token_reminder_text.update()

            _refresh_token_reminder()

            with ui.column().classes("w-full gap-1"):
                with ui.row().classes("items-center gap-2"):
                    ui.label("NeuPrint Token (Required for all NeuPrint datasets)").classes("text-caption font-bold")
                    neuprint_status = ui.label(_token_status(token_state["neuprint"])).classes("text-caption drocat-muted")
                ui.html("Get it from <a href='https://neuprint.janelia.org/account' target='_blank' style='color:#145cff'>neuprint.janelia.org/account</a>").classes("text-caption drocat-muted")

            neuprint_token = ui.input(
                label="NeuPrint Token",
                value="",
                placeholder="Leave blank to keep the saved token",
                password=True,
                password_toggle_button=True,
            ).classes("w-full")

            ui.separator()

            with ui.column().classes("w-full gap-1"):
                with ui.row().classes("items-center gap-2"):
                    ui.label("CAVE Token (for FlyWire CAVE API features)").classes("text-caption font-bold drocat-warn")
                    cave_status = ui.label(_token_status(token_state["cave"])).classes("text-caption drocat-muted")
                ui.html("Get it from <a href='https://codex.flywire.ai/auth_token' target='_blank' style='color:#145cff'>codex.flywire.ai/auth_token</a>").classes("text-caption drocat-muted")
                ui.label("Local converted FlyWire tables work without this token. A CAVE token is needed only when a workflow fetches data or skeletons through the CAVE API; it never replaces the required local files.").classes("text-caption drocat-warn")

            cave_token = ui.input(
                label="CAVE Token (for FlyWire)",
                value="",
                placeholder="Leave blank to keep the saved token",
                password=True,
                password_toggle_button=True,
            ).classes("w-full")

            with ui.row().classes("items-center gap-2 flex-wrap"):
                save_btn = ui.button("Save Tokens", icon="save", color="primary")
                test_btn = ui.button("Test NeuPrint", icon="wifi", color="secondary")
                clear_blank = ui.checkbox(
                    "Clear a saved value when its field is blank",
                    value=False,
                ).classes("text-caption")
            ui.label(
                "Saved token values stay on the server and are never pre-filled in the browser. "
                "Enter a new value only when you want to replace the current one."
            ).classes("text-caption drocat-muted")

        # Output
        with ui.card().classes("w-full drocat-card"):
            section_header("Output Settings", "folder")
            default_dir = dir_input(
                label="Default Output Directory",
                default=get_default_output_dir(),
                global_default=True,
            )
            ui.label(
                "This directory is inherited by tool tabs until a tab-specific "
                "override is saved. Reset clears every tab override. "
                f"Project root: {PROJECT_ROOT}"
            ).classes("text-caption drocat-muted")
            with ui.row():
                save_default_btn = ui.button("Save Default Directory", icon="save", color="primary")
                reset_default_btn = ui.button("Reset", icon="restart_alt", color="secondary")

            def save_default_dir():
                raw_value = (default_dir.value or "").strip()
                if not raw_value:
                    ui.notify("Choose an output directory first", type="warning")
                    return
                saved, effective = set_default_output_dir(raw_value, create=True)
                if not saved or not effective:
                    ui.notify("Cannot save output directory (check the path)", type="negative")
                    return
                default_dir.value = effective
                sync_output_dir_fields(default_dir, effective)
                ui.notify("Default output directory saved", type="positive")

            def reset_default_dir():
                set_default_output_dir("")
                clear_tab_output_overrides()
                default_dir.value = str(DEFAULT_OUTPUT_DIR)
                sync_output_dir_fields(
                    default_dir,
                    str(DEFAULT_OUTPUT_DIR),
                    force=True,
                )
                ui.notify("Default output directory reset", type="positive")

            save_default_btn.on_click(save_default_dir)
            reset_default_btn.on_click(reset_default_dir)

        # App Settings
        with ui.card().classes("w-full drocat-card"):
            section_header("App Settings", "tune")
            ui.label(
                "Auto-suggestion shows dataset type/instance/bodyId names while "
                "typing in the pathfinding neuron inputs, plus a Recent/Frequent "
                "query history when the field is empty. The change applies "
                "immediately to every input."
            ).classes("text-caption drocat-muted")
            auto_suggest_cb = ui.checkbox(
                "Input Auto-Suggestion",
                value=get_auto_suggest_enabled(),
            ).tooltip(
                "Type-ahead suggestions (dataset type names + query history) in "
                "the pathfinding neuron inputs. Disabling restores plain chip "
                "inputs."
            )

            def _toggle_auto_suggest(e):
                enabled = bool(e.value)
                if set_auto_suggest_enabled(enabled):
                    ui.notify(
                        "Input auto-suggestion " + ("enabled" if enabled else "disabled"),
                        type="positive" if enabled else "warning",
                    )
                else:
                    ui.notify("Failed to save the setting", type="negative")

            auto_suggest_cb.on_value_change(_toggle_auto_suggest)

        # Custom type mappings (LabelMapper presets, reusable across runs).
        # Use the same Custom Mapping panel as the tool tabs so Settings and
        # query-local mapping edits share the outlined member fields, aligned
        # dataset rows, and available-neuron viewer.
        with ui.card().classes("w-full drocat-card"):
            section_header("Custom Type Mappings", "hub")
            ui.label(
                "Define reusable neuron groups across datasets. The shared Custom Mapping "
                "panel keeps the same aligned member inputs and LabelMapper JSON format "
                "used by the analysis tabs. Saving updates the reusable Custom Mapping "
                "preset in cache/user_mappings.json and exports it for runs."
            ).classes("text-caption drocat-muted")

            mapping_row_action_renderers = []

            mapping_dataset_select = None

            def _render_mapping_dataset_selector():
                nonlocal mapping_dataset_select
                mapping_dataset_select = dataset_multi_selector(
                    label="Target datasets",
                    default=[],
                    hint=(
                        "Select the datasets this reusable mapping should apply to. "
                        "The editor renders and saves every selected dataset column for "
                        "each group, including empty cells."
                    ),
                )
                return mapping_dataset_select

            def _mapping_datasets():
                values = mapping_dataset_select.value if mapping_dataset_select else []
                values = values or []
                if not isinstance(values, (list, tuple, set)):
                    values = [values]
                return [str(dataset) for dataset in values if dataset]

            mapping_button, mapping_dialog, _resolve_mapping = custom_grouping_block(
                label="Custom Mapping",
                hint="Open the shared LabelMapper group editor with aligned dataset rows "
                     "and available-neuron lookup.",
                datasets_provider=_mapping_datasets,
                tab_key="settings_mapping",
                panel_title="Custom Mapping",
                dataset_selector_renderer=_render_mapping_dataset_selector,
                row_action_renderers=mapping_row_action_renderers,
            )
            mapping_button.classes("drocat-settings-mapping-button")
            grouper = mapping_dialog.inline_grouper

            def _save_custom_mapping():
                # Settings intentionally has one reusable mapping surface now;
                # the old name/description/preset controls are removed from
                # this card. Save to a stable preset name and make it active.
                name = "Custom Mapping"
                if not grouper.datasets():
                    ui.notify(
                        "Select at least one target dataset first",
                        type="warning",
                    )
                    return
                if grouper.is_empty():
                    ui.notify("Add at least one group with neurons first", type="warning")
                    return
                errors = grouper.validate()
                if errors:
                    for err in errors:
                        ui.notify(err, type="negative")
                    return
                payload = to_canonical_dict(
                    grouper._active_rows(),
                    grouper.datasets(),
                    origin="settings",
                )
                sides = {
                    "source_mapping": payload["source_mapping"],
                    "target_mapping": payload["target_mapping"],
                }
                if not mapping_store.save_mapping(name, sides):
                    ui.notify("Failed to save mapping", type="negative")
                    return
                mapping_store.set_active_mapping(name)
                ui.notify("Custom Mapping saved and set active", type="positive")

            def _render_save_mapping_action():
                ui.button(
                    "Save Mapping",
                    icon="save",
                    on_click=_save_custom_mapping,
                ).props("outline no-caps").classes(
                    "drocat-labelmapper-query-action"
                ).tooltip(
                    "Save this mapping using the selected target datasets"
                )

            mapping_row_action_renderers.append(_render_save_mapping_action)
            # The first render happens before the callbacks above are defined;
            # refresh once so an already-open editor receives the new action.
            grouper.resync()

        # Dataset Preparation Guide
        with ui.card().classes("w-full drocat-card"):
            section_header("Dataset Preparation Guide", "menu_book")

            with ui.expansion("NeuPrint Datasets (hemibrain, male-cns, optic-lobe, manc)", icon="cloud").classes("w-full"):
                ui.html("""
                <div style="color:#0b1f3a" class="text-sm">
                    <p><b>NeuPrint datasets are fetched automatically from the server.</b> No manual download required.</p>
                    <ol class="list-decimal ml-4 mt-2">
                        <li>Get your NeuPrint token from <a href="https://neuprint.janelia.org/account" target="_blank" style="color:#145cff">neuprint.janelia.org/account</a></li>
                        <li>Enter the token in the API Tokens section above</li>
                        <li>Click "Save Tokens"</li>
                        <li>Click "Test NeuPrint" to verify the connection</li>
                        <li>Datasets will appear in all tool tabs automatically</li>
                    </ol>
                    <p class="mt-2" style="color:#667085">Available NeuPrint datasets:</p>
                    <ul class="list-disc ml-4" style="color:#667085">
                        <li>hemibrain:v1.2.1 - Adult fly brain (central)</li>
                        <li>male-cns:v1.0 - Full male CNS (latest)</li>
                        <li>male-cns:v0.9 - Full male CNS</li>
                        <li>optic-lobe:v1.1 - Optic lobe detailed</li>
                        <li>manc:v1.2.1 / manc:v1.0 - Male VNC</li>
                    </ul>
                </div>
                """)

            with ui.expansion("FlyWire FAFB v783 · strict local preparation", icon="download").classes("w-full"):
                ui.html("""
                <div style="color:#0b1f3a" class="text-sm">
                    <p style="color:#b45309"><b>Follow the converter layout exactly.</b> Download the raw Codex files; do not rename them to a generated <code>*_allneurons_*</code> filename and do not place raw files in the dataset root.</p>

                    <p class="mt-3 font-bold" style="color:#145cff">1. Create the input folder</p>
                    <p>From the project root, create <code>datasets/flywire_FAFB_v783/downloads/</code>. Keep the original <code>.csv.gz</code> filenames in this folder.</p>

                    <p class="mt-3 font-bold" style="color:#145cff">2. Put these raw files in <code>downloads/</code></p>
                    <p><b>Required for local pathfinding and connection analysis:</b></p>
                    <ul class="list-disc ml-4">
                        <li><code>classification.csv.gz</code></li>
                        <li><code>connections_princeton_no_threshold.csv.gz</code> (preferred), or <code>connections_princeton.csv.gz</code> / <code>connections.csv.gz</code> as the supported fallback</li>
                    </ul>
                    <p><b>Recommended metadata enrichment</b> (the converter can continue without these, but neuron labels and metadata will be incomplete):</p>
                    <ul class="list-disc ml-4">
                        <li><code>names.csv.gz</code>, <code>coordinates.csv.gz</code>, <code>neurons.csv.gz</code></li>
                        <li><code>cell_stats.csv.gz</code>, <code>consolidated_cell_types.csv.gz</code></li>
                    </ul>
                    <p><b>Optional visualization inputs:</b> <code>fafb_v783_princeton_synapse_table.csv.gz</code> for a local synapse table and <code>sk_lod1_783_healed.zip</code> for local skeletons. The converter discovers matching synapse/skeleton filenames and moves the skeleton ZIP to the dataset root.</p>

                    <p class="mt-3 font-bold" style="color:#145cff">3. Convert the files</p>
                    <pre style="white-space:pre-wrap"><code>python src/FAFB_file_converter.py</code></pre>
                    <p>Alternatively, select <code>flywire_FAFB_v783</code> in a tool and run it; the first run invokes the same local preparation automatically.</p>

                    <p class="mt-3 font-bold" style="color:#145cff">4. Verify before running analysis</p>
                    <p>The dataset root should contain generated files named <code>flywire_FAFB_v783_allneurons_neuron_df.parquet</code> (and CSV) and <code>flywire_FAFB_v783_merged_connections.parquet</code>. Click <b>Refresh</b> above and look for <b>✓ local</b>.</p>
                    <p style="color:#b45309"><b>A CAVE token is not a substitute for these local tables.</b> It is only needed for CAVE API fetching or skeleton fallback; local converted tables and a local skeleton ZIP can be used without it.</p>
                </div>
                """)

            with ui.expansion("FlyWire BANC v888 / v626 · strict local preparation", icon="download").classes("w-full"):
                ui.html("""
                <div style="color:#0b1f3a" class="text-sm">
                    <p style="color:#b45309"><b>BANC is local-file only in this toolkit.</b> A CAVE token does not enable BANC API fetching; use the matching local dataset folder and raw files below.</p>

                    <p class="mt-3 font-bold" style="color:#145cff">1. Choose one exact dataset identifier</p>
                    <p>Use either <code>flywire_BANC_v888</code> or <code>flywire_BANC_v626</code>. Never mix files from one version into the other version's folder.</p>

                    <p class="mt-3 font-bold" style="color:#145cff">2. Create the input folder and copy the raw files</p>
                    <p>For the selected identifier, create <code>datasets/&lt;dataset&gt;/downloads/</code> and keep these exact Codex filenames:</p>
                    <ul class="list-disc ml-4">
                        <li><code>neurons.csv.gz</code></li>
                        <li><code>connections_princeton.csv.gz</code></li>
                    </ul>
                    <p>Do not save a manually renamed <code>*_allneurons_neuron_df.csv</code> in the root; that is a generated output, not an input.</p>

                    <p class="mt-3 font-bold" style="color:#145cff">3. Convert the files</p>
                    <pre style="white-space:pre-wrap"><code>python src/BANC_file_converter.py                 # v626 default
python -c "import sys; sys.path.insert(0, 'src'); from BANC_file_converter import ensure_banc_data; d='flywire_BANC_v888'; ensure_banc_data(d, 'datasets/' + d)"</code></pre>
                    <p>Alternatively, select the matching BANC identifier in a tool and run it; preparation is invoked automatically.</p>

                    <p class="mt-3 font-bold" style="color:#145cff">4. Verify before running analysis</p>
                    <p>The selected dataset root should contain <code>&lt;dataset&gt;_allneurons_neuron_df.parquet</code> (and CSV) and <code>&lt;dataset&gt;_merged_connections.parquet</code>. Click <b>Refresh</b> above and look for <b>✓ local</b>.</p>
                    <p style="color:#b45309"><b>BANC skeleton visualization and <code>force_API_fetching</code> are unsupported.</b> Pathfinding, network visualization, and tabular analysis use the converted local files.</p>
                </div>
                """)

        # About
        with ui.card().classes("w-full drocat-card"):
            section_header("About", "info")
            ui.label("DROCAT - Drosophila Connectome Analysis Toolkit").classes("text-subtitle1 font-bold")
            ui.label(f"Version {APP_VERSION} · branch {APP_DOCS_BRANCH}").classes("text-caption drocat-muted")
            with ui.row().classes("mt-2"):
                ui.link("GitHub", APP_GITHUB_URL, new_tab=True).classes("text-primary")
                ui.link("Docs", APP_DOCS_URL, new_tab=True).classes("text-primary")

    def _update_token_status():
        neuprint_status.text = _token_status(token_state["neuprint"])
        cave_status.text = _token_status(token_state["cave"])
        _refresh_token_reminder()

    def save_tokens():
        local_file = PROJECT_ROOT / "token_info_local.txt"
        entered_neuprint = (neuprint_token.value or "").strip()
        entered_cave = (cave_token.value or "").strip()
        saved_tokens = dict(token_state)

        if entered_neuprint:
            saved_tokens["neuprint"] = entered_neuprint
        elif clear_blank.value:
            saved_tokens.pop("neuprint", None)
        if entered_cave:
            saved_tokens["cave"] = entered_cave
        elif clear_blank.value:
            saved_tokens.pop("cave", None)

        content = (
            "# DROCAT Token Configuration\n"
            f"NEUPRINT_TOKEN={saved_tokens.get('neuprint', '')!r}\n"
            f"CAVE_TOKEN={saved_tokens.get('cave', '')!r}\n"
        )
        try:
            local_file.write_text(content, encoding="utf-8")
            local_file.chmod(0o600)
            token_state.clear()
            token_state.update(saved_tokens)
            # Clear the client-side fields after saving so a browser DOM
            # snapshot can never retain a secret value.
            neuprint_token.value = ""
            cave_token.value = ""
            clear_blank.value = False
            _update_token_status()

            ui.notify("Tokens saved to token_info_local.txt", type="positive")
            from ..dataset_service import get_dataset_service
            service = get_dataset_service()
            service._token = saved_tokens.get("neuprint") or None
            service._cave_token = saved_tokens.get("cave") or None
            service._cache.clear()
            service._available_neuprint = None
            service._server_datasets = {}
            service._last_fetch_time = 0
        except Exception as e:
            ui.notify(f"Failed to save: {e}", type="negative")

    async def test_connection():
        # The input is intentionally blank while a token is configured.  Use
        # the server-side value for testing unless the user entered a new one.
        token = (neuprint_token.value or "").strip() or token_state.get("neuprint", "")
        if not token:
            ui.notify("Enter a NeuPrint token first", type="warning")
            return
        ui.notify("Testing NeuPrint connection...", type="info")
        test_btn.disable()
        try:
            from neuprint import Client

            def probe_neuprint():
                client = Client(
                    "neuprint.janelia.org",
                    "hemibrain:v1.2.1",
                    token,
                )
                result = client.fetch_custom(
                    "MATCH (n:Neuron) RETURN count(n) as count LIMIT 1"
                )
                return result["count"].iloc[0] if not result.empty else 0

            count = await run.io_bound(probe_neuprint)
            ui.notify(f"Connected! {count:,} neurons in hemibrain.", type="positive")
        except Exception as e:
            ui.notify(f"Connection failed: {str(e)[:80]}", type="negative")
        finally:
            test_btn.enable()

    save_btn.on_click(save_tokens)
    test_btn.on_click(test_connection)


def _load_tokens() -> dict:
    """Load valid tokens with local values overriding the template.

    Parse the template first and the local file second, key by key.  The old
    implementation stopped after seeing *any* token in the first file, so a
    local CAVE-only file could hide a valid NeuPrint token from the template.
    """
    tokens = {}
    for filename in ["token_info.txt", "token_info_local.txt"]:
        is_local = filename == "token_info_local.txt"
        token_path = PROJECT_ROOT / filename
        if token_path.exists():
            try:
                for line in token_path.read_text().split("\n"):
                    line = line.strip()
                    if line.startswith("NEUPRINT_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip("'\"")
                        if is_local:
                            # A blank local value is an explicit clear and
                            # must not fall back to a template secret.
                            tokens["neuprint"] = token if token and not token.startswith("YOUR_") else ""
                        elif "neuprint" not in tokens and token and not token.startswith("YOUR_"):
                            tokens["neuprint"] = token
                    elif line.startswith("CAVE_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip("'\"")
                        if is_local:
                            tokens["cave"] = token if token and not token.startswith("YOUR_") else ""
                        elif "cave" not in tokens and token and not token.startswith("YOUR_"):
                            tokens["cave"] = token
            except Exception:
                pass
    return tokens


def _token_status(token: str) -> str:
    """Return a non-sensitive status label for a configured token."""
    return "configured (kept hidden)" if token else "not configured"
