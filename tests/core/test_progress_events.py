#!/usr/bin/env python
"""Tests for the [DROCAT][progress] step-event emitters.

Every executable backend pipeline reports its phases through the shared
protocol consumed by the web UI (``ui/components/output_panel.py``):

    [DROCAT][progress] <step>/<total> <label>

These tests pin the emitter contract per backend: the exact line format,
the verbose gating, and the plot3d generated-script totals for the
optional export phases.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))


def _events(capsys):
    return [ln.strip() for ln in capsys.readouterr().out.splitlines()
            if "[DROCAT][progress]" in ln]


class TestEmissionContract:
    """The shared protocol line format and per-backend gating."""

    def test_coana_progress_event_format_and_gating(self, capsys):
        import coana
        fc = object.__new__(coana.FindNeuronConnection)
        fc.verbose_mode = "full"
        fc._in_progress_bar = False
        # The protocol is opt-in: nested FindNeuronConnection runs inside
        # other pipelines default to progress_events=False and stay silent.
        fc.progress_events = False
        fc._progress(2, 5, "Fetching mutual direct connections")
        assert _events(capsys) == []

        fc.progress_events = True
        fc._progress(2, 5, "Fetching mutual direct connections")
        assert _events(capsys) == [
            "[DROCAT][progress] 2/5 Fetching mutual direct connections"]

        # Silent runs emit no control lines.
        fc.verbose_mode = "silent"
        fc._progress(3, 5, "Enriching network edges")
        assert _events(capsys) == []

    def test_comparer_progress_event_gating(self, capsys):
        from comparison.profile_comparator import ConnectivityProfileComparer
        comparer = object.__new__(ConnectivityProfileComparer)
        comparer.verbose = True
        comparer._progress(2, 4, "Extracting and aggregating connectivity profiles")
        assert _events(capsys) == [
            "[DROCAT][progress] 2/4 Extracting and aggregating connectivity profiles"]

        comparer.verbose = False
        comparer._progress(3, 4, "Computing similarity matrices")
        assert _events(capsys) == []

    def test_analyzer_progress_event_gating(self, capsys):
        from comparison.comparison_analyzer import ComparisonAnalyzer
        analyzer = object.__new__(ComparisonAnalyzer)
        analyzer.verbose = True
        analyzer._progress(3, 5, "Computing cross-dataset metrics")
        assert _events(capsys) == [
            "[DROCAT][progress] 3/5 Computing cross-dataset metrics"]

        analyzer.verbose = False
        analyzer._progress(4, 5, "Exporting result tables")
        assert _events(capsys) == []

    def test_neuronbridge_progress_event_gating(self, capsys):
        from neuronbridge_finder import NeuronBridgeFinder
        finder = object.__new__(NeuronBridgeFinder)
        finder.verbose = True
        finder._batch_mode = False
        finder._progress(2, 4, "Fetching matching driver-line records")
        assert _events(capsys) == [
            "[DROCAT][progress] 2/4 Fetching matching driver-line records"]

        finder.verbose = False
        finder._progress(3, 4, "Aggregating and ranking line matches")
        assert _events(capsys) == []

    def test_flylight_progress_event_gating(self, capsys):
        from flylight_downloader import FlyLightDownloader
        downloader = object.__new__(FlyLightDownloader)
        # The UI tab runs with verbose='pbar'; events must still print.
        downloader.verbose = "pbar"
        downloader._progress(3, 4, "Downloading selected images")
        assert _events(capsys) == [
            "[DROCAT][progress] 3/4 Downloading selected images"]

        downloader.verbose = False
        downloader._progress(4, 4, "Generating summaries")
        assert _events(capsys) == []

        # The string 'False' disables output just like the bool False.
        downloader.verbose = "False"
        downloader._progress(4, 4, "Generating summaries")
        assert _events(capsys) == []

    def test_visualize_skeleton_progress_event_gating(self, capsys):
        from visualize_skeleton import VisualizeSkeleton
        vs = object.__new__(VisualizeSkeleton)
        vs.verbose = True
        vs._progress(1, 4, "Loading skeletons")
        assert _events(capsys) == [
            "[DROCAT][progress] 1/4 Loading skeletons"]

        vs.verbose = False
        vs._progress(2, 4, "Loading synapses and meshes")
        assert _events(capsys) == []

    def test_vispath_progress_event_gating(self, capsys):
        from vispath_pkg.vispath import VisualizePath
        vp = object.__new__(VisualizePath)
        vp.verbose = True
        # Nested pipeline calls omit progress_total and must stay silent so
        # the outer tool's protocol owns the bar.
        vp.progress_total = None
        vp._progress(2, 4, "Building the pathway graph")
        assert _events(capsys) == []

        # The standalone Net-Viz run opts in with its 4-step total.
        vp.progress_total = 4
        vp._progress(2, 4, "Building the pathway graph")
        assert _events(capsys) == [
            "[DROCAT][progress] 2/4 Building the pathway graph"]

        vp.verbose = False
        vp._progress(4, 4, "Saving visualization data and files")
        assert _events(capsys) == []


class TestPlot3dScriptProgress:
    """The generated 3D-skeleton script reports the optional export phases."""

    def _generate(self, method_params):
        from ui.runner import ScriptRunner
        return ScriptRunner()._generate_plot3d_script(
            {"dataset": "test:v1"}, method_params)

    def test_base_three_phase_run(self):
        script = self._generate({
            "export_individual_profiles": False,
            "export_video": False,
        })
        assert "progress_total=3" in script
        assert "[DROCAT][progress] 4/" not in script
        assert "vs.plot_individuals(" not in script
        assert "vs.export_video(" not in script

    def test_individual_export_adds_fourth_step(self):
        script = self._generate({
            "export_individual_profiles": True,
            "export_video": False,
            "pdf_images_per_page": (3, 2),
            "views": ["front"],
            "summary_format": ["pdf"],
        })
        assert "progress_total=4" in script
        assert ("[DROCAT][progress] 4/4 Exporting individual profiles"
                in script)
        assert "vs.plot_individuals(" in script
        assert "vs.export_video(" not in script

    def test_video_export_uses_correct_step(self):
        script = self._generate({
            "export_individual_profiles": False,
            "export_video": True,
            "fps": 30,
            "degree_per_frame": 1.0,
            "rotate": "horizontal",
            "export_gif": True,
            "gif_scale": 0.2,
        })
        assert "progress_total=4" in script
        assert "[DROCAT][progress] 4/4 Exporting rotating video" in script

    def test_both_exports_extend_to_five_steps(self):
        script = self._generate({
            "export_individual_profiles": True,
            "export_video": True,
        })
        assert "progress_total=5" in script
        assert "[DROCAT][progress] 4/5 Exporting individual profiles" in script
        assert "[DROCAT][progress] 5/5 Exporting rotating video" in script
