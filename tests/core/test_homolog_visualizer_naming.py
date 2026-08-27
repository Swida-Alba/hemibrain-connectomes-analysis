"""Legend/naming regressions for homolog result visualizations.

Covers:
- ``_homolog_visualizer_kwargs``: a call site's explicit ``legend_mode``
  must win over the app-wide user preference (which defaults to ``type``),
  otherwise the promised ``{type}_{bodyId}`` layer legends silently degrade.
- ``_resolve_plotly_trace_identities``: navis renders somas as unnamed
  ``Mesh3d`` companions next to each skeleton trace; they must resolve to
  their owning neuron instead of positionally stealing another bodyId.
- ``_dedupe_profile_name``: distinct legend keys that sanitize to the same
  stem export with numeric suffixes instead of overwriting each other.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("plotly")
import plotly.graph_objects as go  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison.profile_comparator import HomologFinder  # noqa: E402
from visualize_skeleton import VisualizeSkeleton  # noqa: E402


class _FakeVol:
    """Stand-in for a navis neuron as handed to navis.plot3d."""

    def __init__(self, id_, name):
        self.id = id_
        self.name = name


def _navis_style_traces(neuron_names):
    """Traces shaped like navis output: per neuron a named skeleton plus an
    optional unnamed soma mesh (names ending in ``*soma`` add one)."""
    traces = []
    for name in neuron_names:
        traces.append(go.Scatter3d(x=[0], y=[0], z=[0], mode="lines", name=name))
        if name.endswith("*soma"):
            traces.append(go.Mesh3d())
    return traces


# ---------------------------------------------------------------------------
# _homolog_visualizer_kwargs: legend_mode precedence
# ---------------------------------------------------------------------------

def test_call_site_legend_mode_beats_user_setting(tmp_path):
    finder = HomologFinder(output_dir=str(tmp_path), verbose=False)
    finder.visualization_settings = {"legend_mode": "type"}

    options = finder._homolog_visualizer_kwargs({"legend_mode": "layer"})

    assert options["legend_mode"] == "layer"


def test_user_legend_mode_passes_through_without_call_site_default(tmp_path):
    finder = HomologFinder(output_dir=str(tmp_path), verbose=False)
    finder.visualization_settings = {"legend_mode": "single"}

    options = finder._homolog_visualizer_kwargs({})

    # No call-site requirement -> user preference still applies.
    assert options["legend_mode"] == "single"


def test_other_settings_still_override_defaults(tmp_path):
    finder = HomologFinder(output_dir=str(tmp_path), verbose=False)
    finder.visualization_settings = {"neuron_alpha": 0.5}

    options = finder._homolog_visualizer_kwargs({"neuron_alpha": 0.2})

    assert options["neuron_alpha"] == 0.5


# ---------------------------------------------------------------------------
# _resolve_plotly_trace_identities: unnamed soma companions
# ---------------------------------------------------------------------------

def test_soma_companion_follows_adjacent_skeleton_label():
    """Even when the skeleton label is NOT one of the plotted identities
    (renamed/dataset-specific labels), its unnamed soma mesh must share
    that label instead of positionally stealing another bodyId."""
    vols = [_FakeVol(101, "aMe12"), _FakeVol(202, "dn1")]
    traces = _navis_style_traces(["aMe12*soma", "dn1*soma"])

    resolved = VisualizeSkeleton._resolve_plotly_trace_identities(vols, traces)

    assert resolved[0] == ("aMe12*soma", 0)
    assert resolved[1] == resolved[0], "soma must inherit its own skeleton"
    assert resolved[2] == ("dn1*soma", 2)
    assert resolved[3] == resolved[2]


def _navis_pair(label):
    return [
        go.Scatter3d(x=[0], y=[0], z=[0], mode="lines", name=label),
        go.Mesh3d(),
    ]


def test_named_companion_resolves_to_canonical_neuron_position():
    """Production shape: navis trace labels equal the neuron names, so the
    named traces map back onto their neuron positions and somas follow."""
    vols = [_FakeVol(101, "aMe12"), _FakeVol(202, "dn1")]
    traces = _navis_pair("aMe12") + _navis_pair("dn1")

    resolved = VisualizeSkeleton._resolve_plotly_trace_identities(vols, traces)

    assert resolved[0] == ("aMe12", 0)
    assert resolved[2] == ("dn1", 1)
    assert all(resolved[idx * 2 + 1] == resolved[idx * 2]
               for idx in range(2)), "somas inherit their own skeleton"


def test_only_some_neurons_have_somas_stays_aligned():
    vols = [_FakeVol(101, "aMe12"), _FakeVol(202, "dn1")]
    traces = (
        [go.Scatter3d(x=[0], y=[0], z=[0], mode="lines", name="aMe12"),
         go.Mesh3d(),
         go.Scatter3d(x=[0], y=[0], z=[0], mode="lines", name="dn1")]
    )

    resolved = VisualizeSkeleton._resolve_plotly_trace_identities(vols, traces)

    # Trace list is [skel_a, soma_a, skel_b]; every identity stays with its
    # own neuron even though trace indices drift ahead of neuron positions.
    assert resolved == [("aMe12", 0), ("aMe12", 0), ("dn1", 1)]


def test_named_only_batch_maps_one_to_one():
    vols = [_FakeVol(7, "Mi1"), _FakeVol(8, "Tm3")]
    traces = _navis_style_traces(["Mi1", "Tm3"])

    resolved = VisualizeSkeleton._resolve_plotly_trace_identities(vols, traces)

    assert resolved[0] == ("Mi1", 0)
    assert resolved[1] == ("Tm3", 1)


def test_no_leading_anchor_orphan_meshes_fall_back_positional():
    vols = [_FakeVol(11, None), _FakeVol(22, None)]
    traces = [go.Mesh3d(), go.Mesh3d()]

    resolved = VisualizeSkeleton._resolve_plotly_trace_identities(vols, traces)

    assert [name for name, _ in resolved] == ["11", "22"]
    assert [idx for _, idx in resolved] == [0, 1]


# ---------------------------------------------------------------------------
# _dedupe_profile_name: exported profile filenames stay unique
# ---------------------------------------------------------------------------

def test_dedupe_profile_name_suffixes_collisions():
    taken = set()

    first = VisualizeSkeleton._dedupe_profile_name("aMe12_101", taken)
    second = VisualizeSkeleton._dedupe_profile_name("aMe12_101", taken)

    assert first == "aMe12_101"
    assert second == "aMe12_101_2"


def test_dedupe_profile_name_handles_empty_stem():
    taken = set()

    assert VisualizeSkeleton._dedupe_profile_name("", taken) == "individual"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
