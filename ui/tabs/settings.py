"""Settings Tab - Token configuration, dataset status, and app settings."""

from nicegui import ui
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
        with ui.card().classes("w-full"):
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
                ui.label("CAVE Token (Required for FlyWire FAFB / BANC datasets)").classes("text-caption font-bold drocat-warn")
                ui.html("Get it from <a href='https://codex.flywire.ai/auth_token' target='_blank' style='color:#145cff'>codex.flywire.ai/auth_token</a>").classes("text-caption drocat-muted")
                ui.label("Note: FlyWire datasets require BOTH a CAVE token AND manually downloaded local data files.").classes("text-caption drocat-warn")

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
        with ui.card().classes("w-full"):
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
        with ui.card().classes("w-full"):
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

            with ui.expansion("FlyWire FAFB v783 (Female Brain) - Manual Download Required", icon="download").classes("w-full"):
                ui.html("""
                <div style="color:#0b1f3a" class="text-sm">
                    <p style="color:#d97706"><b>IMPORTANT: FlyWire FAFB requires manual data download AND a CAVE token.</b></p>
                    <p>The NeuPrint token alone is NOT sufficient for FlyWire datasets.</p>

                    <p class="mt-3 font-bold" style="color:#145cff">Step 1: Get CAVE Token</p>
                    <ol class="list-decimal ml-4">
                        <li>Visit <a href="https://codex.flywire.ai/auth_token" target="_blank" style="color:#145cff">codex.flywire.ai/auth_token</a></li>
                        <li>Log in and copy your auth token</li>
                        <li>Enter it in the CAVE Token field above</li>
                        <li>Click "Save Tokens"</li>
                    </ol>

                    <p class="mt-3 font-bold" style="color:#145cff">Step 2: Download Neuron Data (Manual)</p>
                    <ol class="list-decimal ml-4">
                        <li>Visit <a href="https://codex.flywire.ai/api/download?dataset=fafb" target="_blank" style="color:#145cff">codex.flywire.ai/api/download?dataset=fafb</a></li>
                        <li>Download the neuron table CSV file</li>
                        <li>Create folder: <code>datasets/flywire_FAFB_v783/</code></li>
                        <li>Save the file as: <code>datasets/flywire_FAFB_v783/flywire_FAFB_v783_allneurons_neuron_df.csv</code></li>
                        <li>Optionally download ROI count data as: <code>flywire_FAFB_v783_allneurons_roi_count_df.csv</code></li>
                    </ol>

                    <p class="mt-3" style="color:#d97706">Without local data files, FlyWire FAFB queries will fail or timeout.</p>
                </div>
                """)

            with ui.expansion("FlyWire BANC v888 / v626 (Male VNC) - Manual Download Required", icon="download").classes("w-full"):
                ui.html("""
                <div style="color:#0b1f3a" class="text-sm">
                    <p style="color:#d97706"><b>IMPORTANT: FlyWire BANC also requires manual download AND a CAVE token.</b></p>

                    <p class="mt-3 font-bold" style="color:#145cff">Step 1: Get CAVE Token (same as FAFB)</p>
                    <p>If you already set up the CAVE token for FAFB, the same token works for BANC.</p>

                    <p class="mt-3 font-bold" style="color:#145cff">Step 2: Download Neuron Data (Manual)</p>
                    <ol class="list-decimal ml-4">
                        <li>Visit <a href="https://codex.flywire.ai/api/download?dataset=banc" target="_blank" style="color:#145cff">codex.flywire.ai/api/download?dataset=banc</a></li>
                        <li>Download the neuron table CSV file</li>
                        <li>Create folder: <code>datasets/flywire_BANC_v888/</code> (or <code>datasets/flywire_BANC_v626/</code>)</li>
                        <li>Save as: <code>flywire_BANC_v888_allneurons_neuron_df.csv</code></li>
                    </ol>
                </div>
                """)

        # About
        with ui.card().classes("w-full"):
            section_header("About", "info")
            ui.label("DROCAT - Drosophila Connectome Analysis Toolkit").classes("text-subtitle1 font-bold")
            ui.label("Version 4.5.0").classes("text-caption drocat-muted")
            with ui.row().classes("mt-2"):
                ui.link("GitHub", "https://github.com/Swida-Alba/hemibrain-connectomes-analysis", new_tab=True).classes("text-primary")
                ui.link("Docs", "https://github.com/Swida-Alba/hemibrain-connectomes-analysis/blob/main/README.md", new_tab=True).classes("text-primary")

    def save_tokens():
        local_file = PROJECT_ROOT / "token_info_local.txt"
        content = f"""# DROCAT Token Configuration
NEUPRINT_TOKEN='{neuprint_token.value}'
CAVE_TOKEN='{cave_token.value}'
"""
        try:
            local_file.write_text(content)
            ui.notify("Tokens saved to token_info_local.txt", type="positive")
            from ..dataset_service import get_dataset_service
            service = get_dataset_service()
            service._token = neuprint_token.value or None
            service._cave_token = cave_token.value or None
            service._cache.clear()
        except Exception as e:
            ui.notify(f"Failed to save: {e}", type="negative")

    def test_connection():
        if not neuprint_token.value:
            ui.notify("Enter a NeuPrint token first", type="warning")
            return
        ui.notify("Testing NeuPrint connection...", type="info")
        try:
            from neuprint import Client
            client = Client("neuprint.janelia.org", "hemibrain:v1.2.1", neuprint_token.value)
            result = client.fetch_custom("MATCH (n:Neuron) RETURN count(n) as count LIMIT 1")
            count = result["count"].iloc[0] if not result.empty else 0
            ui.notify(f"Connected! {count:,} neurons in hemibrain.", type="positive")
        except Exception as e:
            ui.notify(f"Connection failed: {str(e)[:80]}", type="negative")

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
