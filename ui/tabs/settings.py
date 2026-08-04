"""Settings Tab - Token configuration, dataset status, and app settings."""

from nicegui import run, ui
from pathlib import Path

from ..config import (
    TOKEN_FILE, DEFAULT_OUTPUT_DIR, PROJECT_ROOT,
    get_default_output_dir, save_local_config,
)
from ..components.common import section_header, dataset_status_card, dir_input


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

        # Dataset Status
        section_header("Dataset Availability", "storage")
        dataset_status_card()

        # Tokens
        with ui.card().classes("w-full drocat-card"):
            section_header("API Tokens", "key")

            with ui.column().classes("w-full gap-1"):
                ui.label("NeuPrint Token (Required for all NeuPrint datasets)").classes("text-caption font-bold")
                ui.html("Get it from <a href='https://neuprint.janelia.org/account' target='_blank' style='color:#145cff'>neuprint.janelia.org/account</a>").classes("text-caption drocat-muted")

            existing_tokens = _load_tokens()
            neuprint_token = ui.input(
                label="NeuPrint Token",
                value=existing_tokens.get("neuprint", ""),
                password=True,
                password_toggle_button=True,
            ).classes("w-full")

            ui.separator()

            with ui.column().classes("w-full gap-1"):
                ui.label("CAVE Token (for FlyWire CAVE API features)").classes("text-caption font-bold drocat-warn")
                ui.html("Get it from <a href='https://codex.flywire.ai/auth_token' target='_blank' style='color:#145cff'>codex.flywire.ai/auth_token</a>").classes("text-caption drocat-muted")
                ui.label("Local converted FlyWire tables work without this token. A CAVE token is needed only when a workflow fetches data or skeletons through the CAVE API; it never replaces the required local files.").classes("text-caption drocat-warn")

            cave_token = ui.input(
                label="CAVE Token (for FlyWire)",
                value=existing_tokens.get("cave", ""),
                password=True,
                password_toggle_button=True,
            ).classes("w-full")

            with ui.row():
                save_btn = ui.button("Save Tokens", icon="save", color="primary")
                test_btn = ui.button("Test NeuPrint", icon="wifi", color="secondary")

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
                saved = save_local_config({"default_output_dir": default_dir.value.strip()})
                if saved:
                    ui.notify("Default output directory saved", type="positive")
                else:
                    ui.notify("Failed to save default output directory", type="negative")

            def reset_default_dir():
                default_dir.value = str(DEFAULT_OUTPUT_DIR)
                save_local_config({"default_output_dir": str(DEFAULT_OUTPUT_DIR)})
                ui.notify("Default output directory reset", type="positive")

            save_default_btn.on_click(save_default_dir)
            reset_default_btn.on_click(reset_default_dir)

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

    def save_tokens():
        local_file = PROJECT_ROOT / "token_info_local.txt"
        neuprint_value = (neuprint_token.value or "").strip()
        cave_value = (cave_token.value or "").strip()
        content = (
            "# DROCAT Token Configuration\n"
            f"NEUPRINT_TOKEN={neuprint_value!r}\n"
            f"CAVE_TOKEN={cave_value!r}\n"
        )
        try:
            local_file.write_text(content, encoding="utf-8")
            local_file.chmod(0o600)
            ui.notify("Tokens saved to token_info_local.txt", type="positive")
            from ..dataset_service import get_dataset_service
            service = get_dataset_service()
            service._token = neuprint_value or None
            service._cave_token = cave_value or None
            service._cache.clear()
        except Exception as e:
            ui.notify(f"Failed to save: {e}", type="negative")

    async def test_connection():
        token = (neuprint_token.value or "").strip()
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
    tokens = {}
    for filename in ["token_info_local.txt", "token_info.txt"]:
        token_path = PROJECT_ROOT / filename
        if token_path.exists():
            try:
                for line in token_path.read_text().split("\n"):
                    line = line.strip()
                    if line.startswith("NEUPRINT_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip("'\"")
                        if token and not token.startswith("YOUR_"):
                            tokens["neuprint"] = token
                    elif line.startswith("CAVE_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip("'\"")
                        if token and not token.startswith("YOUR_"):
                            tokens["cave"] = token
                if tokens:
                    break
            except Exception:
                pass
    return tokens
