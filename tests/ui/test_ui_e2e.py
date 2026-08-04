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
        assert "find_reciprocal=fc.find_reciprocal" in script
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

        colabel_script = sr._generate_script(
            "nb_colabel",
            {"verbose": True},
            "colabel",
            {"lines": ["A", "B"], "datasets_to_visualize": "male-cns:v1.0"},
        )
        assert "datasets_to_visualize" in colabel_script
        assert "male-cns:v1.0" in colabel_script

    def test_generated_scripts_include_vispath(self):
        """The PlotPath tool must resolve vispath_pkg in a fresh subprocess."""
        from ui.runner import ScriptRunner, VISPATH_DIR
        assert VISPATH_DIR.exists(), "vispath-subproject/src missing"
        sr = ScriptRunner()
        script = sr._generate_script("plot_path", {"path_file": "x.csv"}, "plot", None)
        assert str(VISPATH_DIR) in script
        assert "vispath_pkg" in script
        assert "vp.visualize()" in script

    def test_generated_script_supports_empty_network_canvas(self):
        """The network panel can run PlotPath without a path input file."""
        from ui.runner import ScriptRunner

        script = ScriptRunner()._generate_script(
            "plot_path",
            {
                "path_file": None,
                "output_folder": "/tmp/drocat-empty-network",
                "generate_empty_network": True,
            },
            "plot",
            None,
        )
        assert "path_file=None" in script
        assert "generate_empty_network=True" in script
        assert "vp.visualize()" in script

    def test_empty_network_opens_a_new_browser_tab(self, tmp_path, monkeypatch):
        """A visible empty canvas opens as a fresh browser tab after export."""
        import webbrowser

        sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))
        from vispath_pkg import VisualizePath

        opened = []
        monkeypatch.setattr(webbrowser, "open_new_tab", opened.append)
        visualizer = VisualizePath(
            path_file=None,
            output_folder=str(tmp_path),
            generate_empty_network=True,
            showfig=True,
            verbose=False,
        )
        output_path = Path(visualizer.generate_empty_network_html())

        assert output_path.exists()
        assert opened == [f"file://{output_path.resolve()}"]

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

            sibling = Path(f"{tmpdir}-outside")
            sibling.mkdir()
            try:
                sr._run_logs = [("stdout", f"Created output folder: {sibling}")]
                assert sr._extract_output_folder(tmpdir) is None
            finally:
                sibling.rmdir()

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

    def test_palette_catalog_and_sampling(self):
        """The bokeh catalog must include categorical/sequential/diverging
        palettes and sample them evenly for any requested count."""
        from ui.components.palette_picker import (
            assign_palette_colors,
            get_palette_catalog,
            move_color,
            normalize_palette_range,
            palette_slice,
            sample_palette,
        )
        catalog = get_palette_catalog()
        names = [name for name, _ in catalog]
        assert "Category20" in names
        assert "Blues" in names
        assert "Spectral" in names
        assert len(catalog) >= 40
        colors = dict(catalog)["Category20"]
        assert len(sample_palette(colors, 1)) == 1
        assert len(sample_palette(colors, 5)) == 5
        assert len(sample_palette(colors, 50)) == 50
        assert assign_palette_colors(["red", "green", "blue"], 2) == [
            "red", "green",
        ]
        assert assign_palette_colors(["blue", "red", "green"], 5) == [
            "blue", "red", "green", "blue", "red",
        ]
        half = palette_slice(colors, 0, 50)
        assert 0 < len(half) < len(colors)
        assert palette_slice(colors, 100, 100) == colors[-1:]
        assert normalize_palette_range(80, 20) == (80, 81)
        assert normalize_palette_range(-5, 200) == (0, 100)
        assert move_color(["a", "b", "c"], 1, -1) == ["b", "a", "c"]
        assert move_color(["a", "b", "c"], 0, -1) == ["a", "b", "c"]

    def test_roi_mesh_traces_have_independent_legend_entries(self):
        """Each resolved ROI stays separately toggleable in the Plotly legend."""
        import plotly.graph_objects as go

        from visualize_skeleton import VisualizeSkeleton, _configure_roi_mesh_traces

        left_traces = (go.Mesh3d(), go.Scatter3d())
        right_traces = (go.Mesh3d(),)
        _configure_roi_mesh_traces(left_traces, "LH(L)")
        _configure_roi_mesh_traces(right_traces, "LH(R)")

        assert left_traces[0].legendgroup == "roi_mesh:LH(L)"
        assert right_traces[0].legendgroup == "roi_mesh:LH(R)"
        assert left_traces[0].legendgroup != right_traces[0].legendgroup
        assert [trace.showlegend for trace in left_traces] == [True, False]
        assert right_traces[0].showlegend is True
        assert left_traces[0].name == "brain region [LH(L)]"
        assert right_traces[0].name == "brain region [LH(R)]"

        visualizer = object.__new__(VisualizeSkeleton)
        expanded_rois, expanded_colors = visualizer._expand_roi_names_with_colors(
            ["LH", "EB"],
            ["#112233", "#abcdef"],
            available_rois=["LH(L)", "LH(R)", "EB"],
        )
        assert expanded_rois == ["LH(L)", "LH(R)", "EB"]
        assert expanded_colors == ["#112233", "#112233", "#abcdef"]

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

    def test_flywire_identifier_does_not_match_neuprint_banc(self):
        from ui.dataset_service import is_flywire_dataset
        assert is_flywire_dataset("flywire_BANC_v626") is True
        assert is_flywire_dataset("flywire_FAFB_v783") is True
        assert is_flywire_dataset("banc:v888") is False

    def test_service_get_token(self):
        from ui.dataset_service import get_dataset_service
        service = get_dataset_service()
        # Token may or may not be present
        token = service.get_token()
        assert token is None or isinstance(token, str)

    def test_flywire_prepared_requires_neurons_and_connections(self, tmp_path):
        from ui.dataset_service import DatasetService

        service = DatasetService()
        service._datasets_dir = tmp_path / "datasets"
        dataset_path = service._datasets_dir / "flywire_FAFB_v783"
        dataset_path.mkdir(parents=True)

        (dataset_path / "flywire_FAFB_v783_allneurons_neuron_df.parquet").touch()
        assert service._check_local_prepared("flywire_FAFB_v783") is False

        (dataset_path / "flywire_FAFB_v783_merged_connections.parquet").touch()
        assert service._check_local_prepared("flywire_FAFB_v783") is True

        neuprint_path = service._datasets_dir / "hemibrain_v1_2_1"
        neuprint_path.mkdir()
        (neuprint_path / "hemibrain_v1_2_1_neuron_df.parquet").touch()
        assert service._check_local_prepared("hemibrain:v1.2.1") is True

    def test_settings_guide_matches_converter_layout(self):
        guide = (PROJECT_ROOT / "docs" / "ui_guides" / "settings.html").read_text()
        embedded_guide = (PROJECT_ROOT / "ui" / "tabs" / "settings.py").read_text()
        assert "datasets/&lt;dataset&gt;/downloads/" in guide
        assert "classification.csv.gz" in guide
        assert "connections_princeton_no_threshold.csv.gz" in guide
        assert "neurons.csv.gz" in guide
        assert "connections_princeton.csv.gz" in guide
        assert "flywire_FAFB_v783_allneurons_neuron_df.parquet" in guide
        assert "flywire_FAFB_v783/flywire_FAFB_v783_allneurons_neuron_df.csv" not in guide
        assert "datasets/flywire_FAFB_v783/downloads/" in embedded_guide
        assert "connections_princeton_no_threshold.csv.gz" in embedded_guide
        assert "flywire_BANC_v888" in embedded_guide

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
            create_nb_find_neuron_tab, create_nb_colabel_tab, create_skeleton_tab,
            create_network_tab, create_visualization_tab,
            create_settings_tab,
        )
        assert all(callable(f) for f in [
            create_find_path_tab, create_find_direct_tab, create_connectivity_profiling_tab,
            create_find_homologs_tab, create_inter_dataset_tab, create_nb_find_lines_tab,
            create_nb_find_neuron_tab, create_nb_colabel_tab, create_skeleton_tab,
            create_network_tab, create_visualization_tab,
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

    def test_parse_neuron_upload_text_and_excel(self):
        import asyncio
        import io
        from types import SimpleNamespace
        import pandas as pd
        from ui.components.common import parse_neuron_upload, read_upload_event

        class NiceGUIFile:
            name = "neurons.csv"

            async def read(self):
                return b"type\naMe12\n"

        filename, content = asyncio.run(
            read_upload_event(SimpleNamespace(file=NiceGUIFile()))
        )
        assert (filename, content) == ("neurons.csv", b"type\naMe12\n")

        csv_bytes = b"type,notes\naMe12,first\naMe10,second\n"
        assert parse_neuron_upload("neurons.csv", csv_bytes) == ["aMe12", "aMe10"]
        tsv_bytes = b"bodyId\tlabel\n720575940610453042\tcell\n"
        assert parse_neuron_upload("neurons.tsv", tsv_bytes) == [720575940610453042]

        workbook = io.BytesIO()
        pd.DataFrame({"type": ["PPL101", "DN1p"]}).to_excel(workbook, index=False)
        assert parse_neuron_upload("neurons.xlsx", workbook.getvalue()) == ["PPL101", "DN1p"]

        numeric_workbook = io.BytesIO()
        pd.DataFrame({"bodyId": [42, None]}).to_excel(numeric_workbook, index=False)
        assert parse_neuron_upload("body_ids.xlsx", numeric_workbook.getvalue()) == [42]

    def test_output_panel(self):
        from ui.components.output_panel import OutputPanel
        panel = OutputPanel("Test")
        assert panel.title == "Test"
        assert panel._dom_id.startswith("drocat-results-")
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

    def test_guide_html_files_and_links_are_direct(self):
        """Instruction links must point to real local HTML guides with valid
        internal links."""
        import re as _re
        from nicegui.page import page
        from nicegui import Client
        import ui.app as app_module

        p2 = page('/smoke-guide')
        client2 = Client(p2)
        with client2:
            app_module.main_page()
        hrefs = [
            str((getattr(e, '_props', None) or {}).get('href', ''))
            for e in client2.elements.values()
        ]
        guide_links = [h for h in hrefs if h.startswith('docs/ui_guides/')]
        assert 'docs/ui_guides/find_path.md'.replace('.md', '.html') in guide_links
        assert 'docs/ui_guides/input_formats.html' in guide_links
        assert 'docs/ui_guides/README.html' in guide_links

        # Every linked HTML guide must exist and its internal links must resolve
        guides_dir = PROJECT_ROOT / 'docs' / 'ui_guides'
        html_files = sorted(guides_dir.glob('*.html'))
        assert len(html_files) >= 12
        for html_file in html_files:
            text = html_file.read_text(encoding='utf-8')
            assert text.lstrip().startswith('<!doctype html>')
            assert '<title>' in text
            assert 'guide.css' in text
            for target in _re.findall(r'href="([^"]+\.html)"', text):
                if target == 'guide.css' or target.startswith(('http:', 'https:')):
                    continue
                assert (guides_dir / target).exists(), f'{html_file.name} -> {target}'


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
        try:
            import tomllib
        except ModuleNotFoundError:  # Python 3.10
            import tomli as tomllib
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        assert data["project"]["version"] == "4.5.0"
        assert "ui" in data["project"]["optional-dependencies"]
        assert "coana" in data["tool"]["setuptools"]["py-modules"]
        assert "neuronbridge_client" in data["tool"]["setuptools"]["py-modules"]


# =============================================================================
# Integration: HTTP Server
# =============================================================================
class TestHTTPServer:
    def test_server_starts_and_responds(self):
        import socket
        import subprocess
        import time
        import urllib.error
        import urllib.request

        with socket.socket() as port_probe:
            port_probe.bind(("127.0.0.1", 0))
            port = port_probe.getsockname()[1]

        child_env = os.environ.copy()
        # NiceGUI enables its own screen-test mode when this pytest variable is
        # inherited, which requires unrelated test-runner configuration.
        child_env.pop("PYTEST_CURRENT_TEST", None)
        child_env.pop("NICEGUI_SCREEN_TEST_PORT", None)
        child_env["DROCAT_UI_PORT"] = str(port)
        child_env["DROCAT_UI_SHOW"] = "0"
        proc = subprocess.Popen(
            [sys.executable, str(PROJECT_ROOT / "ui" / "app.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
            env=child_env,
            text=True,
        )
        try:
            url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 15
            while True:
                if proc.poll() is not None:
                    stdout, stderr = proc.communicate()
                    pytest.fail(
                        "UI server exited before becoming ready.\n"
                        f"stdout:\n{stdout}\nstderr:\n{stderr}"
                    )
                try:
                    with urllib.request.urlopen(url, timeout=1) as response:
                        html = response.read().decode()
                        assert response.status == 200
                        assert "DROCAT" in html
                        assert "drocat-cobalt" in html
                        break
                except urllib.error.URLError as error:
                    if time.monotonic() >= deadline:
                        pytest.fail(f"UI server did not become ready: {error}")
                    time.sleep(0.2)
        finally:
            if proc.poll() is None:
                proc.terminate()
            proc.communicate(timeout=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
