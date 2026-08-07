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
                # Raw ANSI colors and tqdm-style bars must be sanitized into
                # clean, single progress lines (no per-refresh log spam).
                "print('\\x1b[33mANSI-warning\\x1b[0m', flush=True)\n"
                "sys.stdout.write('Loading data:   0%|          | 0/4 [00:00<?, ?it/s]\\n')\n"
                "sys.stdout.write('Loading data: 100%|██████████| 4/4 [00:00<00:00,  5.22s/it]\\n')\n"
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
        # ANSI escapes are stripped, and the tqdm bars arrive as progress.
        assert ("stdout", "ANSI-warning") in events, events
        assert ("progress", "Loading data: 100%|██████████| 4/4 [00:00<00:00,  5.22s/it]") in events
        assert ("progress", "Loading data:   0%|          | 0/4 [00:00<?, ?it/s]") in events
        # No whitespace-only or empty log lines leak into the stream.
        assert all(msg.strip() for _lvl, msg in events), events

    def test_output_splitter_cleans_tqdm_refresh_stream(self):
        """\\r-refreshed tqdm bars become progress segments, spaces are dropped."""
        from ui.runner import _OutputSplitter

        splitter = _OutputSplitter()
        parts = list(splitter.feed(
            "Loading data:   0%|          | 0/4 [00:00<?, ?it/s]   \r"
            "Loading data:  25%|██▌       | 1/4 [00:09<00:28,  9.64s/it]  \r"
            "\n"
        ))
        assert parts == [
            ("Loading data:   0%|          | 0/4 [00:00<?, ?it/s]", True),
            ("Loading data:  25%|██▌       | 1/4 [00:09<00:28,  9.64s/it]", True),
        ]

    def test_output_splitter_classifies_newline_tqdm_bars(self):
        """Non-TTY tqdm writes one bar per line; each must be classified progress."""
        from ui.runner import _OutputSplitter

        splitter = _OutputSplitter()
        parts = list(splitter.feed(
            "Building target profiles:   0%|          | 0/2972 [00:00<?, ?it/s]\n"
            "Building target profiles:  50%|█████     | 1488/2972 [00:06<00:05, 265.91it/s]\n"
            "[HomologFinder] Built 2972 target profiles\n"
        ))
        assert parts == [
            ("Building target profiles:   0%|          | 0/2972 [00:00<?, ?it/s]", True),
            ("Building target profiles:  50%|█████     | 1488/2972 [00:06<00:05, 265.91it/s]", True),
            ("[HomologFinder] Built 2972 target profiles", False),
        ]

    def test_output_splitter_strips_ansi_and_drops_blank_lines(self):
        """ANSI codes are removed; whitespace-only lines are dropped."""
        from ui.runner import _OutputSplitter

        splitter = _OutputSplitter()
        parts = list(splitter.feed(
            "\x1b[33m⚠️  Warning: 19 layers but only 10 colors.\x1b[0m\n"
            "   \n"
            "\n"
            "saving figure to \x1b[34m/out/plot.html\x1b[0m...Done (HTML saved)\n"
        ))
        assert parts == [
            ("⚠️  Warning: 19 layers but only 10 colors.", False),
            ("saving figure to /out/plot.html...Done (HTML saved)", False),
        ]

    def test_output_splitter_cursor_up_refreshes_progress(self):
        """Cursor-up escapes (\\x1b[A) act as progress refresh separators."""
        from ui.runner import _OutputSplitter

        splitter = _OutputSplitter()
        parts = list(splitter.feed(
            "Fetching:   0%|          | 0/6 [00:00<?, ?it/s]\x1b[A\n"
            "Fetching:  17%|█▋        | 1/6 [00:00<00:03,  1.32it/s]\x1b[A\n"
        ))
        assert parts == [
            ("Fetching:   0%|          | 0/6 [00:00<?, ?it/s]", True),
            ("Fetching:  17%|█▋        | 1/6 [00:00<00:03,  1.32it/s]", True),
        ]

    def test_output_splitter_flushes_partial_line(self):
        """A trailing partial line is flushed at EOF."""
        from ui.runner import _OutputSplitter

        splitter = _OutputSplitter()
        assert list(splitter.feed("partial line")) == []
        assert list(splitter.flush()) == [("partial line", False)]

    def test_output_splitter_preserves_crlf_plain_lines(self):
        """CRLF line endings must not be mistaken for progress refreshes."""
        from ui.runner import _OutputSplitter

        splitter = _OutputSplitter()
        parts = list(splitter.feed("[HomologFinder] Loaded cache\r\n"))
        assert parts == [("[HomologFinder] Loaded cache", False)]

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
        """A visible empty canvas opens as a fresh browser tab after export.

        Both webbrowser entry points are patched: the same-window ``open`` must
        never fire (the canvas opens exactly once, in a new tab), otherwise the
        test would pop a real browser window.
        """
        import webbrowser

        sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))
        from vispath_pkg import VisualizePath

        opened = []
        monkeypatch.setattr(webbrowser, "open", opened.append)
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
        # Exactly one open: a fresh tab (not the same-window open + tab).
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

    def test_homologs_empty_saveas_uses_auto_folder(self, tmp_path):
        """UI sends saveas='' when blank; results must land in a per-run
        findhomologs_ folder instead of being dumped into output_dir."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        import pandas as pd
        from comparison.profile_comparator import HomologFinder

        finder = HomologFinder(
            source="aMe12",
            source_dataset="male-cns:v1.0",
            target_dataset="male-cns:v1.0",
            output_dir=str(tmp_path),
            saveas="",  # The UI passes an empty string when the field is blank
            verbose=False,
        )
        results = pd.DataFrame({
            "source_bodyId": [1, 2],
            "target_bodyId": [3, 4],
            "rank_union": [0.5, 0.4],
        })
        finder._save_homolog_results_internal(
            results_df=results,
            query="aMe12",
            source_dataset="male-cns:v1.0",
            target_dataset="male-cns:v1.0",
            output_dir=str(tmp_path),
            saveas="",
            direction="both",
            include_partner_details=False,
            top_n_details=5,
            params={"query": "aMe12"},
        )
        folders = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(folders) == 1, [p.name for p in tmp_path.iterdir()]
        folder = folders[0]
        assert folder.name.startswith("findhomologs_MCNS_to_MCNS_aMe12_"), folder.name
        assert (folder / "README.txt").exists()
        assert (folder / "results" / "homolog_results.csv").exists()
        # Nothing is dumped into the output root itself
        assert not (tmp_path / "README.txt").exists()
        assert not (tmp_path / "results").exists()

    def test_homologs_custom_saveas_respected(self, tmp_path):
        """A non-empty saveas must still be used as the folder name."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        import pandas as pd
        from comparison.profile_comparator import HomologFinder

        finder = HomologFinder(
            source="aMe12",
            source_dataset="male-cns:v1.0",
            target_dataset="male-cns:v1.0",
            output_dir=str(tmp_path),
            saveas="my_custom_run",
            verbose=False,
        )
        results = pd.DataFrame({"source_bodyId": [1], "target_bodyId": [3], "rank_union": [0.5]})
        finder._save_homolog_results_internal(
            results_df=results,
            query="aMe12",
            source_dataset="male-cns:v1.0",
            target_dataset="male-cns:v1.0",
            output_dir=str(tmp_path),
            saveas="my_custom_run",
            direction="both",
            include_partner_details=False,
            top_n_details=5,
            params={"query": "aMe12"},
        )
        assert (tmp_path / "my_custom_run" / "README.txt").exists()

    def test_interdataset_output_name_prefix(self):
        """Inter-dataset runs use the interdataset_ folder prefix."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from comparison.comparison_parameters import ComparisonParameters

        params = ComparisonParameters(
            datasets=["male-cns:v0.9", "hemibrain:v1.2.1"],
            source_neurons=["aMe12"],
            target_neurons=["PPL101"],
            output_folder="/tmp/drocat-test",
        )
        assert params.output_name.startswith("interdataset_aMe12_to_PPL101_"), params.output_name
        assert "comp_" not in params.output_name
        # Empty-string saveas must fall back to the auto name
        params.saveas = ""
        assert params.output_name.startswith("interdataset_aMe12_to_PPL101_"), params.output_name

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

    def test_scan_output_files_returns_all_files_no_cap(self):
        """The panel mirrors the run folder - no 50-file cap may hide files
        (hundreds of images are written by the NB find-lines workflow)."""
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(60):
                (Path(tmpdir) / f"file_{i:03d}.csv").write_text("a,b\n1,2")
            (Path(tmpdir) / "sub" / "nested.txt").parent.mkdir()
            (Path(tmpdir) / "sub" / "nested.txt").write_text("x")
            files = sr._scan_output_files(tmpdir)
            assert len(files) == 61  # every file, nested included

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

    def test_resolve_scan_dir_failed_run_returns_none(self):
        """A failed run that created no folder must NOT scan the shared root
        (which would surface files from previous runs, e.g. BANC files while
        running male-cns)."""
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "banc_previous_output.csv").write_text("a,b\n1,2")
            sr._run_logs = [
                ("stderr", "[ERROR] MTe07 not found as source in any columns"),
            ]
            assert sr._resolve_scan_dir(tmpdir) is None

    def test_resolve_scan_dir_uses_announced_run_folder(self):
        """The backend-announced per-run folder wins over the storage root."""
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            run = Path(tmpdir) / "findpath_MCNS_aMe12_to_aMe10_L2w3r0p0_20260801_120000"
            run.mkdir()
            sr._run_logs = [("stdout", f"  📁 Created output folder: {run}")]
            assert sr._resolve_scan_dir(tmpdir) == str(run)

    def test_resolve_scan_dir_accepts_direct_run_folder(self):
        """plot_path passes its pre-created plotpath_ folder directly."""
        from ui.runner import ScriptRunner
        sr = ScriptRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            run = Path(tmpdir) / "plotpath_my_paths_20260801_120000"
            run.mkdir()
            sr._run_logs = [("stdout", "plotting network...")]
            assert sr._resolve_scan_dir(str(run)) == str(run)

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

    def test_settings_token_loader_merges_template_and_local_values(self, tmp_path, monkeypatch):
        """A local value overrides only its key, without hiding the other token."""
        from ui.tabs import settings as settings_module

        (tmp_path / "token_info.txt").write_text(
            "NEUPRINT_TOKEN='template-neuprint'\nCAVE_TOKEN='template-cave'\n",
            encoding="utf-8",
        )
        (tmp_path / "token_info_local.txt").write_text(
            "CAVE_TOKEN='local-cave'\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)

        assert settings_module._load_tokens() == {
            "neuprint": "template-neuprint",
            "cave": "local-cave",
        }

    def test_settings_dataset_cache_card(self):
        """The Settings tab exposes the 'Pull Full Dataset' operation: a
        dataset selector, a force-rebuild option, run/cancel buttons, and a
        progress element - all wired to a DatasetPuller."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.settings import create_settings_tab

        client = Client(page("/settings-dataset-cache"))
        with client:
            create_settings_tab()

        labels = {}
        for el in client.elements.values():
            props = getattr(el, "_props", {})
            label = props.get("label")
            text = getattr(el, "text", None)
            if label:
                labels[label] = el
            if text == "Dataset Cache":
                labels["card_title"] = el

        assert "card_title" in labels
        assert "Dataset" in labels  # dataset selector
        assert "Batch size" in labels
        assert "Parallel workers" in labels
        assert any(
            getattr(el, "text", None) == "Pull Full Dataset"
            for el in client.elements.values()
        )
        assert any(
            getattr(el, "text", None) == "Cancel"
            for el in client.elements.values()
        )
        assert any(
            getattr(el, "text", None) == "Force rebuild (clear broken cache first)"
            for el in client.elements.values()
        )
        # progress element + status label
        assert any(
            getattr(el, "text", None) == "Idle"
            for el in client.elements.values()
        )

    def test_settings_token_status_is_non_sensitive(self):
        from ui.tabs.settings import _token_status

        assert _token_status("secret-token") == "configured (kept hidden)"
        assert "secret-token" not in _token_status("secret-token")
        assert _token_status("") == "not configured"

    def test_custom_grouping_instruction_link_present(self):
        """Every custom-grouping UI surface links to the LabelMapper guide:
        the Settings 'Custom Type Mappings' card and the mapping selector in
        the tool tabs (Cross-Dataset / Find Path / Find Direct)."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.mapping_editor import MAPPING_GUIDE_URL, mapping_selector

        def guide_links(client):
            return [
                el for el in client.elements.values()
                if getattr(el, "_props", {}).get("href") == MAPPING_GUIDE_URL
            ]

        # Settings card
        client = Client(page("/guide-link-settings"))
        with client:
            from ui.tabs.settings import create_settings_tab
            create_settings_tab()
        links = guide_links(client)
        assert len(links) == 1
        assert links[0]._props.get("target") == "_blank"  # opens in a new tab

        # Selector component (used by Cross-Dataset / Find Path / Find Direct)
        client = Client(page("/guide-link-selector"))
        with client:
            mapping_selector()
        links = guide_links(client)
        assert len(links) == 1
        assert links[0]._props.get("target") == "_blank"

        # Opt-out keeps the layout clean for callers that add their own link
        client = Client(page("/guide-link-selector-off"))
        with client:
            mapping_selector(show_instructions=False)
        assert guide_links(client) == []

    def test_settings_tab_reminds_when_tokens_missing(self, tmp_path, monkeypatch):
        """The Settings tab raises a visible reminder when tokens are missing:
        NeuPrint is required, CAVE is explicitly optional (only FlyWire FAFB
        online fetching), and the banner disappears once both are set."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs import settings as settings_module

        built = []

        def build(neuprint: str, cave: str):
            (tmp_path / "token_info_local.txt").write_text(
                f"NEUPRINT_TOKEN='{neuprint}'\nCAVE_TOKEN='{cave}'\n",
                encoding="utf-8",
            )
            monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)
            client = Client(page(f"/settings-reminder-{len(built)}"))
            with client:
                settings_module.create_settings_tab()
            built.append(client)
            reminder = next(
                el for el in client.elements.values()
                if getattr(el, "_props", {}).get("id") == "drocat-token-reminder"
            )
            return reminder, reminder.default_slot.children[0]

        # both missing -> prominent reminder
        reminder, text = build("", "")
        assert reminder.visible is True
        assert "No API tokens configured" in text.text
        assert "required for NeuPrint datasets" in text.text
        assert "only needed for FlyWire FAFB online fetching" in text.text

        # only CAVE missing -> soft reminder marking it optional
        reminder, text = build("real-neuprint-token", "")
        assert reminder.visible is True
        assert "CAVE token not configured - optional" in text.text
        assert "FlyWire FAFB online fetching" in text.text

        # only NeuPrint missing -> required-token reminder
        reminder, text = build("", "real-cave-token")
        assert reminder.visible is True
        assert "NeuPrint token not configured - it is required" in text.text

        # both set -> no reminder
        reminder, text = build("real-neuprint-token", "real-cave-token")
        assert reminder.visible is False

    def test_local_dataset_listing_requires_complete_flywire_conversion(self, tmp_path):
        from ui.dataset_service import DatasetService

        service = DatasetService()
        service._datasets_dir = tmp_path / "datasets"
        dataset_path = service._datasets_dir / "flywire_FAFB_v783"
        dataset_path.mkdir(parents=True)
        (dataset_path / "flywire_FAFB_v783_metadata.json").write_text(
            '{"neuron_counts": {"total": 10}}',
            encoding="utf-8",
        )
        (dataset_path / "flywire_FAFB_v783_allneurons_neuron_df.parquet").touch()

        listed = service.get_local_datasets()
        assert listed[0].local_prepared is False
        assert listed[0].available is False

        (dataset_path / "flywire_FAFB_v783_merged_connections.parquet").touch()
        listed = service.get_local_datasets()
        assert listed[0].local_prepared is True
        assert listed[0].available is True

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

    def test_neuron_list_input_commits_pending_text_on_blur(self):
        """Leaving a neuron chip field commits text without requiring Enter."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        client = Client(page("/neuron-input-blur-test"))
        with client:
            container = neuron_list_input(label="Source Neurons")

        listeners = {
            listener.type: listener
            for listener in container.chip_input._event_listeners.values()
        }
        assert listeners["input"].js_handler == "(event) => emit(event?.target?.value ?? '')"
        assert "blur" in listeners

        # Simulate NiceGUI's native input and blur events. The browser's
        # QSelect editor is empty after blur, but the selected chip remains.
        container.chip_input._handle_event({
            "listener_id": listeners["input"].id,
            "args": "PPL1",
        })
        assert container.chip_input.value == []
        container.chip_input._handle_event({
            "listener_id": listeners["blur"].id,
            "args": None,
        })
        assert container.get_value() == ("exact", ["PPL1"])
        # Committed values must live in the option list, otherwise the chip
        # never renders in the browser (model-value is filtered by options).
        assert container.chip_input.options == ["PPL1"]

    def test_neuron_list_input_clearing_text_drops_pending_value(self):
        """Deleting the editor text must not commit the previously typed text."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        client = Client(page("/neuron-input-clear-test"))
        with client:
            container = neuron_list_input(label="Source Neurons")

        listeners = {
            listener.type: listener
            for listener in container.chip_input._event_listeners.values()
        }

        # Type a value, then clear the field with Backspace (native input '').
        container.chip_input._handle_event({
            "listener_id": listeners["input"].id,
            "args": "PPL1",
        })
        container.chip_input._handle_event({
            "listener_id": listeners["input"].id,
            "args": "",
        })
        # Quasar's blur-reset emits an empty input-value; it must be ignored.
        container.chip_input._handle_event({
            "listener_id": listeners["inputValue"].id,
            "args": "",
        })
        container.chip_input._handle_event({
            "listener_id": listeners["blur"].id,
            "args": None,
        })
        # Nothing was committed: the field was cleared before blur.
        assert container.get_value() == ("exact", [])

    def test_neuron_list_input_max_items_rejects_second_chip(self):
        """A chip input with max_items=1 keeps only the first value."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        client = Client(page("/neuron-input-max-items-test"))
        with client:
            container = neuron_list_input(label="Source Neuron", max_items=1)

        listeners = {
            listener.type: listener
            for listener in container.chip_input._event_listeners.values()
        }

        # The first value is committed on blur.
        container.chip_input._handle_event({
            "listener_id": listeners["input"].id,
            "args": "PPL1",
        })
        container.chip_input._handle_event({
            "listener_id": listeners["blur"].id,
            "args": None,
        })
        assert container.get_value() == ("exact", ["PPL1"])

        # A second value is rejected once the input is at capacity.
        container.chip_input._handle_event({
            "listener_id": listeners["input"].id,
            "args": "PPL2",
        })
        container.chip_input._handle_event({
            "listener_id": listeners["blur"].id,
            "args": None,
        })
        assert container.get_value() == ("exact", ["PPL1"])

        # Quasar-native additions (Enter) are truncated to the cap as well.
        container.chip_input.value = ["PPL1", "PPL2"]
        container.chip_input._handle_event({
            "listener_id": listeners["update:modelValue"].id,
            "args": ["PPL1", "PPL2"],
        })
        assert container.get_value() == ("exact", ["PPL1"])

    def test_neuron_list_input_initial_values_survive_enter(self):
        """Seeded chips survive get_value(); Enter keeps the whole text as ONE chip."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        client = Client(page("/neuron-input-initial-test"))
        with client:
            container = neuron_list_input(
                label="Synapse Thresholds",
                initial=[3, 5, 10],
                show_filter=False,
                show_upload=False,
            )
        assert container.get_value() == ("exact", [3, 5, 10])
        assert container.chip_input.options == [3, 5, 10]

        listeners = {
            listener.type: listener
            for listener in container.chip_input._event_listeners.values()
        }

        # Quasar's add-unique adds the whole editor text as a single chip on
        # Enter; commas are legal inside names and must NOT be split.
        container.chip_input.value = [3, 5, 10, "3, 7"]
        container.chip_input._handle_event({
            "listener_id": listeners["update:modelValue"].id,
            "args": [3, 5, 10, "3, 7"],
        })
        assert container.get_value() == ("exact", [3, 5, 10, "3, 7"])
        assert container.chip_input.options == [3, 5, 10, "3, 7"]

    def test_neuron_list_input_commas_and_spaces_are_not_separators(self):
        """Blur commits the WHOLE editor text as one chip (',' and ' ' preserved)."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        client = Client(page("/neuron-input-no-split-test"))
        with client:
            container = neuron_list_input(label="Source Neurons")

        listeners = {
            listener.type: listener
            for listener in container.chip_input._event_listeners.values()
        }

        # Comma-containing names (e.g. driver lines) stay one chip.
        container.chip_input._handle_event({
            "listener_id": listeners["input"].id,
            "args": "PPL1, PPL2",
        })
        container.chip_input._handle_event({
            "listener_id": listeners["blur"].id,
            "args": None,
        })
        assert container.get_value() == ("exact", ["PPL1, PPL2"])

        # Space-containing names (e.g. 'A -> B -> C' layers) stay one chip.
        container.chip_input._handle_event({
            "listener_id": listeners["input"].id,
            "args": "A -> B -> C",
        })
        container.chip_input._handle_event({
            "listener_id": listeners["blur"].id,
            "args": None,
        })
        assert container.get_value() == ("exact", ["PPL1, PPL2", "A -> B -> C"])

        # Surrounding whitespace is trimmed; digit-only text still normalizes.
        container.chip_input._handle_event({
            "listener_id": listeners["input"].id,
            "args": "  720575940610453042  ",
        })
        container.chip_input._handle_event({
            "listener_id": listeners["blur"].id,
            "args": None,
        })
        assert container.get_value() == ("exact", ["PPL1, PPL2", "A -> B -> C", 720575940610453042])

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

    def test_output_panel_progress_refreshes_same_bar_in_place(self):
        """Progress lines refresh their own bar; new bars start fresh lines."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.output_panel import OutputPanel

        client = Client(page("/output-panel-progress-test"))
        with client:
            panel = OutputPanel("Test")
            panel.create()

        bar_0 = "Building target profiles:   0%|          | 0/2972 [00:00<?, ?it/s]"
        bar_50 = "Building target profiles:  50%|█████     | 1488/2972 [00:06<00:05, 265.91it/s]"
        bar_other = "Loading data:   0%|          | 0/4 [00:00<?, ?it/s]"

        panel.log(bar_0, "progress")
        panel.log(bar_50, "progress")
        children = panel.log_area.default_slot.children
        assert len(children) == 1
        assert children[0].text == bar_50

        # A different bar starts a new line instead of overwriting the first.
        panel.log(bar_other, "progress")
        assert len(children) == 2
        assert children[1].text == bar_other

        # A normal line ends the refresh chain; the next progress line is new.
        panel.log("[HomologFinder] Built 2972 target profiles", "stdout")
        panel.log("Loading data:  50%|█████     | 2/4 [00:00<00:00,  9.64s/it]", "progress")
        assert len(children) == 4
        assert children[3].text.startswith("Loading data:  50%")

        # Trailing whitespace (tqdm line-clearing) is trimmed; blank lines drop.
        panel.log("   ", "stdout")
        panel.log("done   ", "stdout")
        assert len(children) == 5
        assert children[4].text == "done"

    def test_output_dir_fields_sync_and_persist(self, tmp_path, monkeypatch):
        """Output-directory fields share one persisted default: setting one
        field updates every other field, and the value survives as the new
        default (permanent, not just the current run)."""
        import ui.config as cfg
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import dir_input, sync_output_dir_fields

        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")

        client = Client(page("/dir-sync-test"))
        with client:
            field_a = dir_input()
            field_b = dir_input()

        # The blur/persist handler runs the same helper the test calls now.
        target = tmp_path / "permanent_outputs"
        saved, effective = cfg.set_default_output_dir(str(target), create=False)
        assert saved is True
        # sync what the handler would sync: every other field follows
        sync_output_dir_fields(field_a, effective)
        assert field_b.value == effective
        assert cfg.get_default_output_dir() == effective

        # A second dir_input built later picks up the persisted default.
        client2 = Client(page("/dir-sync-test-2"))
        with client2:
            field_c = dir_input()
        assert field_c.value == effective

    def test_output_panel_log_is_pointer_resizable(self):
        """The execution-log window must be drag-resizable: the wrapper owns
        the CSS resize handle with a definite height range, and the inner log
        fills the wrapper (h-full) plus scrolls, so dragging actually resizes
        the visible console instead of leaving a dead handle."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.output_panel import OutputPanel

        client = Client(page("/output-panel-resize-test"))
        with client:
            panel = OutputPanel("Test")
            panel.create()

        wrapper_style = panel.log_wrapper._style
        assert wrapper_style.get("resize") == "vertical"
        assert wrapper_style.get("overflow") == "hidden"  # required for resize
        assert wrapper_style.get("height") == "200px"     # definite start height
        assert wrapper_style.get("min-height") == "100px"
        assert wrapper_style.get("max-height") == "600px"

        # The log tracks the wrapper so every drag changes the console size.
        assert "h-full" in panel.log_area._classes
        log_style = panel.log_area._style
        assert log_style.get("overflow-y") == "auto"

        # Streaming still works after the change (sanity for the console).
        panel.log("resizable log line", "stdout")
        assert panel.log_area.default_slot.children[0].text == "resizable log line"

    def _collect_panel_texts(self, container):
        """Recursively collect all label texts inside a UI container."""
        texts = []

        def walk(element):
            for child in element.default_slot.children:
                if hasattr(child, "text"):
                    texts.append(child.text)
                walk(child)

        walk(container)
        return texts

    def test_output_panel_streams_new_files_during_run(self, tmp_path):
        """Files created while a run is active appear in the panel immediately."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.output_panel import OutputPanel
        from ui.runner import ScriptRunner

        run_folder = tmp_path / "findpath_MCNS_aMe12_to_aMe10_L2w3r0p0_20260801_120000"
        run_folder.mkdir()
        (run_folder / "connections.csv").write_text("a,b\n1,2")

        client = Client(page("/output-panel-stream-test"))
        with client:
            panel = OutputPanel("Test")
            panel.create()
            runner = ScriptRunner()
            runner.is_running = True
            runner._run_logs = [("stdout", f"  📁 Created output folder: {run_folder}")]
            panel._poll_output_files(runner, str(tmp_path))

        texts = self._collect_panel_texts(panel.files_container)
        assert any("connections.csv" in text for text in texts), texts
        # the folder-structure display shows the run folder, not type categories
        assert not any("Data Tables (CSV)" in text for text in texts), texts

    def test_output_panel_streaming_skips_unknown_run_folder(self, tmp_path):
        """Before the run folder is known, polling must not show stale files."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.output_panel import OutputPanel
        from ui.runner import ScriptRunner

        (tmp_path / "old_run_connections.csv").write_text("a,b\n1,2")

        client = Client(page("/output-panel-stream-skip-test"))
        with client:
            panel = OutputPanel("Test")
            panel.create()
            runner = ScriptRunner()
            runner.is_running = True
            runner._run_logs = [("stdout", "some unrelated log line")]
            panel._poll_output_files(runner, str(tmp_path))

        texts = self._collect_panel_texts(panel.files_container)
        assert not any("old_run_connections.csv" in text for text in texts), texts

    def test_output_panel_show_files_mirrors_folder_structure(self, tmp_path):
        """Output files are grouped by their folder in the run directory
        (root files first, subfolders as nested expansions) - not by file
        type."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.output_panel import OutputPanel

        client = Client(page("/output-panel-tree-test"))
        with client:
            panel = OutputPanel("Test")
            panel.create()

        def file_entry(rel):
            return {"name": Path(rel).name, "path": str(tmp_path / rel), "size": 10,
                    "modified": "2026-08-01T00:00:00"}

        panel.show_files([
            file_entry("summary.csv"),
            file_entry("data_details/params.json"),
            file_entry("data_details/layer_1.csv"),
            file_entry("images/a.png"),
            file_entry("images/nested/b.png"),
        ], str(tmp_path))

        texts = self._collect_panel_texts(panel.files_container)
        # root file is listed directly
        assert any("summary.csv" in text for text in texts), texts
        # subfolders appear as expansions with their file counts
        assert any("data_details  (2)" in text for text in texts), texts
        assert any("images  (2)" in text for text in texts), texts
        # no type-category headers
        assert not any("Data Tables (CSV)" in text for text in texts), texts
        assert not any("Images" in text for text in texts), texts

    def test_output_panel_show_files_preserves_expanded_state(self, tmp_path):
        """Streaming refreshes keep the user's expanded folders open."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.output_panel import OutputPanel

        client = Client(page("/output-panel-expand-test"))
        with client:
            panel = OutputPanel("Test")
            panel.create()

        def file_entry(rel):
            return {"name": Path(rel).name, "path": str(tmp_path / rel), "size": 10,
                    "modified": "2026-08-01T00:00:00"}

        panel.show_files([file_entry("data_details/a.csv")], str(tmp_path))
        assert panel._file_expansions
        for expansion in panel._file_expansions.values():
            expansion.value = True  # user expands a folder

        # A streaming refresh with a new file arrives.
        panel.show_files([
            file_entry("data_details/a.csv"),
            file_entry("data_details/b.csv"),
        ], str(tmp_path))

        assert panel._file_expansions, "folders must be rebuilt"
        assert all(expansion.value for expansion in panel._file_expansions.values()), \
            "expanded folders must stay open after a streaming refresh"
        texts = self._collect_panel_texts(panel.files_container)
        assert any("b.csv" in text for text in texts), texts


# =============================================================================
# Test App Module
# =============================================================================
class TestApp:
    def test_app_imports(self):
        from ui.app import main, main_page
        assert callable(main)
        assert callable(main_page)

    def test_tab_toolbar_is_compact_and_non_scrolling(self):
        from ui.app import DROCAT_CSS

        assert ".drocat-tabs .q-tabs__content" in DROCAT_CSS
        assert "overflow: hidden" in DROCAT_CSS
        assert "flex: 1 1 0 !important" in DROCAT_CSS
        assert ".drocat-tabs .q-tabs__arrow { display: none !important; }" in DROCAT_CSS

    def test_settings_never_prefills_saved_tokens_in_browser_dom(self, tmp_path, monkeypatch):
        """Saved secrets remain server-side instead of entering the client DOM."""
        from nicegui import Client
        from nicegui.page import page
        import ui.tabs.settings as settings_module

        secret_neuprint = "neuprint-secret-for-dom-test"
        secret_cave = "cave-secret-for-dom-test"
        (tmp_path / "token_info_local.txt").write_text(
            f"NEUPRINT_TOKEN='{secret_neuprint}'\nCAVE_TOKEN='{secret_cave}'\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)

        client = Client(page("/settings-dom-test"))
        with client:
            settings_module.create_settings_tab()

        input_props = [
            getattr(element, "_props", {})
            for element in client.elements.values()
            if getattr(element, "_props", {}).get("label")
            in {"NeuPrint Token", "CAVE Token (for FlyWire)"}
        ]
        assert len(input_props) == 2
        assert all(props.get("value", "") == "" for props in input_props)
        assert secret_neuprint not in repr(client.elements)
        assert secret_cave not in repr(client.elements)

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
        for f in ["run_DROCAT.command", "run_DROCAT.bat",
                  "archive/install/install.sh", "archive/install/install.bat",
                  "archive/install/install.ps1"]:
            assert (PROJECT_ROOT / f).exists(), f"Missing: {f}"

    def test_shell_scripts_executable(self):
        assert os.access(PROJECT_ROOT / "run_DROCAT.command", os.X_OK)
        assert os.access(PROJECT_ROOT / "archive/install/install.sh", os.X_OK)


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
