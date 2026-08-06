"""Settings Tab - Token configuration, dataset status, and app settings."""

from nicegui import run, ui

from ..config import (
    DATASETS,
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
    get_default_output_dir,
    set_default_output_dir,
)
from ..components.common import section_header, dataset_status_card, dir_input, sync_output_dir_fields
from ..components.mapping_editor import MappingGridEditor
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

        # Dataset Status.  The card already owns its title; adding a second
        # section header here made the Settings page look like it contained
        # two different availability controls.
        dataset_status_card()

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
            token_reminder = ui.element("div").props('id="drocat-token-reminder"').classes("w-full").style(
                "border: 1px solid #e6a23c; background: #fdf6ec; border-radius: 8px; padding: 10px 12px;"
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
            default_dir = dir_input(label="Default Output Directory", default=get_default_output_dir())
            ui.label(
                "This directory is pre-filled in every tool tab. "
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
                default_dir.value = str(DEFAULT_OUTPUT_DIR)
                sync_output_dir_fields(default_dir, str(DEFAULT_OUTPUT_DIR))
                ui.notify("Default output directory reset", type="positive")

            save_default_btn.on_click(save_default_dir)
            reset_default_btn.on_click(reset_default_dir)

        # Custom type mappings (LabelMapper presets, reusable across runs)
        with ui.card().classes("w-full drocat-card"):
            section_header("Custom Type Mappings", "hub")
            ui.label(
                "Define custom neuron groups across datasets for cross-dataset comparison and "
                "pathfinding (LabelMapper format). Presets are saved permanently in "
                "cache/user_mappings.json and exported to cache/user_mappings/ for runs."
            ).classes("text-caption drocat-muted")

            preset_select = ui.select(
                options=mapping_store.list_mappings(),
                value=mapping_store.get_active_mapping(),
                label="Saved mappings",
            ).classes("w-full drocat-select")
            with ui.row().classes("items-center gap-2 w-full"):
                name_input = ui.input(
                    label="Mapping name", placeholder="e.g. aMe12 orthologs"
                ).classes("drocat-input").style("width: 300px")
                desc_input = ui.input(label="Description (optional)").classes("drocat-input")

            side_tabs = ui.tabs().classes("w-full")
            with side_tabs:
                ui.tab("Source")
                ui.tab("Target")
                ui.tab("Intermediate")
            with ui.tab_panels(side_tabs, value="Source").classes("w-full bg-transparent"):
                with ui.tab_panel("Source").classes("p-0"):
                    source_container = ui.column().classes("w-full gap-2")
                with ui.tab_panel("Target").classes("p-0"):
                    target_container = ui.column().classes("w-full gap-2")
                with ui.tab_panel("Intermediate").classes("p-0"):
                    inter_container = ui.column().classes("w-full gap-2")

            editors = {
                "source_mapping": MappingGridEditor("source_mapping"),
                "target_mapping": MappingGridEditor("target_mapping"),
                "intermediate_mapping": MappingGridEditor("intermediate_mapping"),
            }
            editors["source_mapping"].create(source_container, list(DATASETS))
            editors["target_mapping"].create(target_container, list(DATASETS))
            editors["intermediate_mapping"].create(inter_container, list(DATASETS))

            def _refresh_preset_select(value=None):
                preset_select.options = mapping_store.list_mappings()
                preset_select.value = value

            def _load_preset(name):
                preset = mapping_store.get_mapping(name)
                if not preset:
                    return
                name_input.value = preset["name"]
                desc_input.value = preset.get("description", "")
                for side in mapping_store.MAPPING_SIDES:
                    editors[side].set_data(preset.get(side) or {})

            def _new_preset():
                name_input.value = ""
                desc_input.value = ""
                for side in mapping_store.MAPPING_SIDES:
                    editors[side].set_data({})
                preset_select.value = None

            def _save_preset():
                name = (name_input.value or "").strip()
                if not name:
                    ui.notify("Enter a mapping name first", type="warning")
                    return
                sides = {}
                for side in mapping_store.MAPPING_SIDES:
                    if not editors[side].is_empty():
                        sides[side] = editors[side].get_data()
                if not sides:
                    ui.notify("Add at least one group with neurons first", type="warning")
                    return
                errors = mapping_store.validate_mapping(sides)
                if errors:
                    for err in errors:
                        ui.notify(err, type="negative")
                    return
                if not mapping_store.save_mapping(name, sides, desc_input.value or ""):
                    ui.notify("Failed to save mapping", type="negative")
                    return
                _refresh_preset_select(name)
                ui.notify(f"Mapping '{name}' saved", type="positive")

            def _rename_preset():
                old = preset_select.value
                new = (name_input.value or "").strip()
                if not old or not new:
                    ui.notify("Select a preset and enter the new name", type="warning")
                    return
                if not mapping_store.rename_mapping(old, new):
                    ui.notify("Rename failed (name taken or missing)", type="negative")
                    return
                _refresh_preset_select(new)
                name_input.value = new
                ui.notify(f"Renamed to '{new}'", type="positive")

            def _delete_preset():
                name = preset_select.value
                if not name:
                    ui.notify("Select a preset to delete", type="warning")
                    return
                if not mapping_store.delete_mapping(name):
                    ui.notify("Delete failed", type="negative")
                    return
                _refresh_preset_select(None)
                _new_preset()
                ui.notify(f"Deleted '{name}'", type="positive")

            def _set_active():
                name = preset_select.value
                if not name:
                    ui.notify("Select a preset to set active", type="warning")
                    return
                mapping_store.set_active_mapping(name)
                ui.notify(f"'{name}' is now the active mapping", type="positive")

            preset_select.on_value_change(lambda e: _load_preset(e.value) if e.value else None)
            ui.button("Save Mapping", icon="save", color="primary").on_click(_save_preset)
            ui.button("New", icon="add").on_click(_new_preset)
            ui.button("Rename", icon="edit").on_click(_rename_preset)
            ui.button("Delete", icon="delete", color="negative").on_click(_delete_preset)
            ui.button("Set Active", icon="star").on_click(_set_active)

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
                    <p>The dataset root should contain generated files named <code>flywire_FAFB_v783_allneurons_neuron_df.parquet</code> (and CSV) and <code>flywire_FAFB_v783_merged_connections.parquet</code> (and CSV). Click <b>Refresh</b> above and look for <b>✓ local</b>.</p>
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
                    <p>The selected dataset root should contain <code>&lt;dataset&gt;_allneurons_neuron_df.parquet</code> (and CSV) and <code>&lt;dataset&gt;_merged_connections.parquet</code> (and CSV). Click <b>Refresh</b> above and look for <b>✓ local</b>.</p>
                    <p style="color:#b45309"><b>BANC skeleton visualization and <code>force_API_fetching</code> are unsupported.</b> Pathfinding, network visualization, and tabular analysis use the converted local files.</p>
                </div>
                """)

        # About
        with ui.card().classes("w-full drocat-card"):
            section_header("About", "info")
            ui.label("DROCAT - Drosophila Connectome Analysis Toolkit").classes("text-subtitle1 font-bold")
            ui.label("Version 4.5.0").classes("text-caption drocat-muted")
            with ui.row().classes("mt-2"):
                ui.link("GitHub", "https://github.com/Swida-Alba/hemibrain-connectomes-analysis", new_tab=True).classes("text-primary")
                ui.link("Docs", "https://github.com/Swida-Alba/hemibrain-connectomes-analysis/blob/main/README.md", new_tab=True).classes("text-primary")

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
