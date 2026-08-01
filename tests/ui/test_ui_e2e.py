#!/usr/bin/env python
"""
Comprehensive End-to-End Tests for DROCAT UI (v4.5.0 - rewritten)
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest


# =============================================================================
# Test Configuration Module
# =============================================================================
class TestConfig:
    def test_config_imports(self):
        from ui import config
        assert config is not None

    def test_config_datasets(self):
        from ui.config import DATASETS, NEUPRINT_DATASETS, FLYWIRE_DATASETS
        assert len(DATASETS) >= 5
        assert "male-cns:v0.9" in NEUPRINT_DATASETS
        assert "flywire_FAFB_v783" in FLYWIRE_DATASETS

    def test_config_defaults(self):
        from ui.config import DEFAULTS
        assert DEFAULTS["min_synapse_num"] == 3
        assert DEFAULTS["max_interlayer"] == 2

    def test_config_paths(self):
        from ui.config import PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR
        assert PROJECT_ROOT.exists()
        assert SRC_DIR.exists()

    def test_config_app_settings(self):
        from ui.config import APP_TITLE, APP_VERSION, APP_PORT, APP_HOST
        assert APP_VERSION == "4.5.0"
        assert APP_PORT == 8080


# =============================================================================
# Test Runner Engine
# =============================================================================
class TestRunner:
    def test_runner_imports(self):
        from ui.runner import ScriptRunner, TOOL_REGISTRY
        assert len(TOOL_REGISTRY) >= 9

    def test_script_runner_instantiation(self):
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        assert sr.is_running is False

    def test_tool_registry_has_all_tools(self):
        from ui.runner import TOOL_REGISTRY
        expected = {"find_path", "find_direct", "connectivity_profiling", "find_homologs",
                     "inter_dataset", "nb_find_lines", "nb_find_neuron", "nb_colabel",
                     "plot3d_skeleton", "plot_path"}
        assert expected.issubset(TOOL_REGISTRY.keys())
        for name in expected:
            assert "label" in TOOL_REGISTRY[name], f"missing UI label for {name}"

    def test_streaming_logs_emit_lines_and_progress_immediately(self):
        """Partial lines and \\r progress must reach the log before the run ends."""
        import asyncio
        import sys
        from ui.runner import ScriptRunner

        async def run() -> list:
            runner = ScriptRunner()
            code = (
                "import sys, time\n"
                "print('start-line', flush=True)\n"
                "for i in range(3):\n"
                "    sys.stdout.write(f'\\rprogress {i}')\n"
                "    sys.stdout.flush()\n"
                "    time.sleep(0.02)\n"
                "print()\n"
                "print('end-line', flush=True)\n"
            )
            runner.process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-u",
                "-c",
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            events = []
            await runner._stream_output(lambda msg, level: events.append((level, msg)))
            await runner.process.wait()
            return events

        events = asyncio.run(run())
        assert ("stdout", "start-line") in events
        assert ("stdout", "end-line") in events
        progress_events = [(lvl, msg) for lvl, msg in events if lvl == "progress"]
        assert any("progress" in msg for _, msg in progress_events), progress_events
        first_progress = next(i for i, (lvl, _) in enumerate(events) if lvl == "progress")
        assert first_progress < events.index(("stdout", "end-line"))

    def test_generate_find_path_script(self):
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        script = sr._generate_script("find_path", {"dataset": "male-cns:v0.9"}, "find_all_path", None)
        assert "from coana import FindNeuronConnection" in script
        assert "FindAllPath" in script
        assert "male-cns:v0.9" in script

    def test_generate_find_direct_script(self):
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        script = sr._generate_script("find_direct", {"dataset": "hemibrain:v1.2.1"}, "find_direct", None)
        assert "from coana import" in script
        assert "FindDirectConnections" in script

    def test_generate_inter_dataset_script(self):
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        script = sr._generate_inter_dataset_script({"datasets": ["a", "b"]}, None)
        assert "ComparisonParameters" in script
        assert "ComparisonAnalyzer" in script

    def test_generate_neuronbridge_script(self):
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        script = sr._generate_neuronbridge_script("nb_find_lines", {"verbose": True}, {"queries": ["aMe12"]})
        assert "NeuronBridgeFinder" in script
        assert "find_lines" in script

    def test_generated_scripts_include_vispath(self):
        """The PlotPath tool must resolve vispath_pkg in a fresh subprocess."""
        from ui.runner import ScriptRunner, VISPATH_DIR
        assert VISPATH_DIR.exists(), "vispath-subproject/src missing"
        sr = ScriptRunner()
        script = sr._generate_script("plot_path", {"path_file": "x.csv"}, "plot", None)
        assert str(VISPATH_DIR) in script
        assert "vispath_pkg" in script
        assert "vp.visualize()" in script

    def test_generate_plot3d_script_includes_optional_exports(self):
        """Plot3D optional profile/video calls must be appended after plot_neurons."""
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        script = sr._generate_script(
            "plot3d_skeleton",
            {"dataset": "male-cns:v0.9", "neuron_layers": ["aMe12"]},
            "plot",
            {
                "export_individual_profiles": True,
                "pdf_images_per_page": (3, 2),
                "views": ["front"],
                "summary_format": ["pdf"],
                "export_video": True,
                "fps": 30,
                "degree_per_frame": 1.0,
                "rotate": "horizontal",
                "export_gif": True,
                "gif_scale": 0.2,
            },
        )
        assert "vs.plot_neurons()" in script
        assert "vs.plot_individuals(" in script
        assert "vs.export_video(" in script
        # Optional export keys must not leak into the VisualizeSkeleton constructor
        assert "export_individual_profiles" not in script.split("VisualizeSkeleton(")[1].split(")")[0]

    def test_homologs_ui_params_match_signature(self):
        """Regression: the UI used to send expand_untyped_2hop, which does not exist."""
        import inspect
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from comparison.profile_comparator import HomologFinder
        params = inspect.signature(HomologFinder.__init__).parameters
        assert "include_untyped_partners" in params
        assert "expand_untyped_2hop" not in params

    def test_scan_output_files_empty(self):
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        assert sr._scan_output_files("/nonexistent") == []

    def test_scan_output_files_with_files(self):
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.csv").write_text("a,b\n1,2")
            files = sr._scan_output_files(tmpdir)
            assert len(files) >= 1

    def test_extract_output_folder_picks_current_run(self):
        """The results panel must link to THIS run's folder, not older ones."""
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            old = Path(tmpdir) / "aMe12_to_aMe10"
            new = Path(tmpdir) / "findpath_MCNS_aMe12_to_aMe10_L2w3r0p0_20260801_120000"
            old.mkdir()
            new.mkdir()
            sr._run_logs = [
                ("stdout", f"data will be saved in: {old}"),
                ("stdout", f"  📁 Created output folder: {new}"),
            ]
            assert sr._extract_output_folder(tmpdir) == str(new)

        sr2 = ScriptRunner()
        sr2._run_logs = [("stdout", "nothing here")]
        assert sr2._extract_output_folder("/tmp") is None

    def test_generate_color_palette_small_n(self):
        """Bokeh categorical palettes have minimum sizes; small n must still work."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from utils.color_utils import generate_color_palette
        assert len(generate_color_palette(1, "category20")) == 1
        assert len(generate_color_palette(2, "category10")) == 2
        assert len(generate_color_palette(3, "cool")) == 3

    def test_pick_directory_exists(self):
        from ui.runner import pick_directory, pick_file
        assert callable(pick_directory)
        assert callable(pick_file)


# =============================================================================
# Test Dataset Service
# =============================================================================
class TestDatasetService:
    def test_service_imports(self):
        from ui.dataset_service import get_dataset_service, DatasetInfo, folder_to_dataset, dataset_to_folder
        assert callable(get_dataset_service)

    def test_folder_to_dataset_conversion(self):
        from ui.dataset_service import folder_to_dataset
        assert folder_to_dataset("hemibrain_v1_2_1") == "hemibrain:v1.2.1"
        assert folder_to_dataset("male-cns_v0_9") == "male-cns:v0.9"
        assert folder_to_dataset("flywire_FAFB_v783") == "flywire_FAFB_v783"
        assert folder_to_dataset("optic-lobe_v1_1") == "optic-lobe:v1.1"

    def test_dataset_to_folder_conversion(self):
        from ui.dataset_service import dataset_to_folder
        assert dataset_to_folder("hemibrain:v1.2.1") == "hemibrain_v1_2_1"
        assert dataset_to_folder("male-cns:v0.9") == "male-cns_v0_9"
        assert dataset_to_folder("flywire_FAFB_v783") == "flywire_FAFB_v783"

    def test_service_get_token(self):
        from ui.dataset_service import get_dataset_service
        service = get_dataset_service()
        # Token may or may not be present
        token = service.get_token()
        assert token is None or isinstance(token, str)

    def test_dataset_info_dataclass(self):
        from ui.dataset_service import DatasetInfo
        info = DatasetInfo(name="test", source="neuprint")
        assert info.name == "test"
        assert info.available is False
        assert info.neuron_count == 0


# =============================================================================
# Test Tab Modules
# =============================================================================
class TestTabs:
    def test_all_tab_functions_exist(self):
        from ui.tabs import (
            create_find_path_tab, create_find_direct_tab, create_connectivity_profiling_tab,
            create_find_homologs_tab, create_inter_dataset_tab, create_nb_find_lines_tab,
            create_nb_find_neuron_tab, create_nb_colabel_tab, create_visualization_tab,
            create_settings_tab,
        )
        assert all(callable(f) for f in [
            create_find_path_tab, create_find_direct_tab, create_connectivity_profiling_tab,
            create_find_homologs_tab, create_inter_dataset_tab, create_nb_find_lines_tab,
            create_nb_find_neuron_tab, create_nb_colabel_tab, create_visualization_tab,
            create_settings_tab,
        ])


# =============================================================================
# Test Components
# =============================================================================
class TestComponents:
    def test_common_imports(self):
        from ui.components.common import (
            dataset_selector, neuron_input, number_input, select_input,
            checkbox_input, dir_input, parse_neuron_list, section_header,
            param_grid, open_folder, dataset_status_card,
        )
        assert all(callable(f) for f in [
            dataset_selector, neuron_input, number_input, select_input,
            checkbox_input, dir_input, parse_neuron_list, section_header,
            param_grid, open_folder, dataset_status_card,
        ])

    def test_parse_neuron_list(self):
        from ui.components.common import parse_neuron_list
        assert parse_neuron_list("aMe12, aMe10") == ["aMe12", "aMe10"]
        assert parse_neuron_list("aMe12\naMe10") == ["aMe12", "aMe10"]
        assert 720575940610453042 in parse_neuron_list("720575940610453042, aMe12")
        assert parse_neuron_list("") == []
        assert parse_neuron_list(None) == []

    def test_output_panel(self):
        from ui.components.output_panel import OutputPanel
        panel = OutputPanel("Test")
        assert panel.title == "Test"
        assert panel._files == []
        assert panel._format_size(500) == "500 B"
        assert panel._format_size(1536) == "1.5 KB"
        assert panel._format_size(1048576) == "1.0 MB"


# =============================================================================
# Test App Module
# =============================================================================
class TestApp:
    def test_app_imports(self):
        from ui.app import main, main_page
        assert callable(main)
        assert callable(main_page)


# =============================================================================
# Test Installer Scripts
# =============================================================================
class TestInstallerScripts:
    def test_all_scripts_exist(self):
        for f in ["install.sh", "install.bat", "install.ps1", "run_ui.sh", "run_ui.bat"]:
            assert (PROJECT_ROOT / f).exists(), f"Missing: {f}"

    def test_shell_scripts_executable(self):
        assert os.access(PROJECT_ROOT / "install.sh", os.X_OK)
        assert os.access(PROJECT_ROOT / "run_ui.sh", os.X_OK)


# =============================================================================
# Test pyproject.toml
# =============================================================================
class TestPyproject:
    def test_version(self):
        import tomllib
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        assert data["project"]["version"] == "4.5.0"
        assert "ui" in data["project"]["optional-dependencies"]
        assert "drocat-ui" in data["project"]["scripts"]


# =============================================================================
# Integration: HTTP Server
# =============================================================================
class TestHTTPServer:
    def test_server_starts_and_responds(self):
        import subprocess, time, urllib.request
        proc = subprocess.Popen(
            [sys.executable, str(PROJECT_ROOT / "ui" / "app.py")],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
        )
        try:
            time.sleep(4)
            response = urllib.request.urlopen("http://127.0.0.1:8080", timeout=5)
            html = response.read().decode()
            assert response.status == 200
            assert "DROCAT" in html
            assert "drocat-cobalt" in html  # Light Photo-Selector theme
        finally:
            proc.terminate()
            proc.wait(timeout=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
