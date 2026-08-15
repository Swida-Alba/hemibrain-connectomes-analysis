"""The 3D skeleton visualization shares the pathfinding neuron-search backend.

Covers `VisualizeSkeleton.search_columns`:
- the field exists with the same 'auto' default as pathfinding
- the value is forwarded to statvis.getNeurons (the same resolver used by
  FindNeuronConnection.InitializeNeuronInfo), so restricting the search to
  type / instance / bodyId works identically in the 3D tab
- invalid scopes are rejected at construction
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import visualize_skeleton as vs_mod  # noqa: E402
from visualize_skeleton import VisualizeSkeleton  # noqa: E402


class FakeClient:
    """Minimal neuprint client stub: construction must not hit the network."""

    def __init__(self, *args, **kwargs):
        self.dataset = kwargs.get("dataset", "hemibrain:v1.2.1")

    def fetch_version(self):
        return None


def test_search_columns_defaults_to_auto():
    assert VisualizeSkeleton.__dataclass_fields__["search_columns"].default == "auto"


def test_hemisphere_defaults_to_both():
    assert VisualizeSkeleton.__dataclass_fields__["hemisphere"].default == "both"


def test_neuron_colors_default_follows_background(tmp_path, monkeypatch):
    """Default neuron palette: bokeh Category10 on white, Set3 on black.
    Explicit neuron_colors are never overridden."""
    def fake_get_neurons(requiredNeurons, dataset="", custom_group_names=None,
                         client=None, verbose=True, search_columns="auto"):
        return pd.DataFrame(), pd.DataFrame(), "auto_name", None

    monkeypatch.setattr(vs_mod.sv, "getNeurons", fake_get_neurons)

    vs_white = VisualizeSkeleton(
        dataset="hemibrain:v1.2.1",
        neuron_layers=["aMe12"],
        client=FakeClient(dataset="hemibrain:v1.2.1"),
        verbose=False,
        output_dir=str(tmp_path),
        include_timestamp=False,
    )
    assert vs_white.neuron_colors[0] == "rgba(31, 119, 180, 0.2)"  # Category10 blue

    vs_black = VisualizeSkeleton(
        dataset="hemibrain:v1.2.1",
        neuron_layers=["aMe12"],
        background_color="black",
        client=FakeClient(dataset="hemibrain:v1.2.1"),
        verbose=False,
        output_dir=str(tmp_path),
        include_timestamp=False,
    )
    assert vs_black.neuron_colors[0] == "rgba(141, 211, 199, 0.2)"  # Set3 teal

    vs_custom = VisualizeSkeleton(
        dataset="hemibrain:v1.2.1",
        neuron_layers=["aMe12"],
        neuron_colors=["red", "blue"],
        background_color="black",
        client=FakeClient(dataset="hemibrain:v1.2.1"),
        verbose=False,
        output_dir=str(tmp_path),
        include_timestamp=False,
    )
    assert vs_custom.neuron_colors[0].startswith("rgba(255, 0, 0")  # user's red kept


def test_mixed_color_alpha_uses_global_fallback_per_layer(tmp_path, monkeypatch):
    """Explicit alpha overrides only its color; alpha-less entries inherit the global."""
    def fake_get_neurons(requiredNeurons, dataset="", custom_group_names=None,
                         client=None, verbose=True, search_columns="auto"):
        return pd.DataFrame(), pd.DataFrame(), "auto_name", None

    monkeypatch.setattr(vs_mod.sv, "getNeurons", fake_get_neurons)

    visualizer = VisualizeSkeleton(
        dataset="hemibrain:v1.2.1",
        neuron_layers=["source", "target"],
        neuron_colors=["#ff0000", "rgba(0, 255, 0, 0.25)"],
        synapse_colors=["#0000ff", "rgba(255, 255, 0, 0.75)"],
        mesh_color=["#111111", "rgba(100, 100, 100, 0.05)"],
        neuron_alpha=0.4,
        synapse_alpha=0.6,
        mesh_alpha=0.2,
        client=FakeClient(dataset="hemibrain:v1.2.1"),
        verbose=False,
        output_dir=str(tmp_path),
        include_timestamp=False,
    )

    assert visualizer.neuron_colors == (
        "rgba(255, 0, 0, 0.4)",
        "rgba(0, 255, 0, 0.25)",
    )
    assert visualizer.synapse_colors == ("rgba(0, 0, 255, 0.6)",)
    assert visualizer.mesh_color == [
        "rgba(17, 17, 17, 0.2)",
        "rgba(100, 100, 100, 0.05)",
    ]


def test_hemisphere_filter_keeps_left_and_unclassified():
    """'left' keeps L + U neurons; explicit right-hemisphere neurons drop."""
    vs = object.__new__(VisualizeSkeleton)
    vs.hemisphere = "left"
    ndf = pd.DataFrame({
        "bodyId": [1, 2, 3, 4, 5],
        "type": ["A", "B", "C", "D", "E"],
        "instance": ["A_R", "B_L", "C", "D_R", "E"],
    })
    out, _ = vs._filter_neuron_df_by_hemisphere(ndf)
    assert sorted(out["bodyId"]) == [2, 3, 5]


def test_hemisphere_filter_keeps_right_and_unclassified():
    vs = object.__new__(VisualizeSkeleton)
    vs.hemisphere = "right"
    ndf = pd.DataFrame({
        "bodyId": [1, 2, 3, 4, 5],
        "type": ["A", "B", "C", "D", "E"],
        "instance": ["A_R", "B_L", "C", "D_R", "E"],
    })
    out, _ = vs._filter_neuron_df_by_hemisphere(ndf)
    assert sorted(out["bodyId"]) == [1, 3, 4, 5]


def test_hemisphere_filter_prefers_soma_side_column():
    """'Soma side' (or 'hemisphere') wins over the instance suffix."""
    vs = object.__new__(VisualizeSkeleton)
    vs.hemisphere = "left"
    ndf = pd.DataFrame({
        "bodyId": [1, 2],
        "type": ["A", "B"],
        "instance": ["A_R", "B"],
        "Soma side": ["L", "R"],
    })
    out, _ = vs._filter_neuron_df_by_hemisphere(ndf)
    # neuron 1: Soma side L (kept despite _R instance); neuron 2: R (dropped)
    assert sorted(out["bodyId"]) == [1]


def test_hemisphere_filter_detects_camelcase_soma_side():
    """male-cns tables name the column 'somaSide' (camelCase) with 'M'
    (midline) values - midline must be treated as unclassified and kept."""
    vs = object.__new__(VisualizeSkeleton)
    vs.hemisphere = "left"
    ndf = pd.DataFrame({
        "bodyId": [1, 2, 3],
        "type": ["A", "B", "C"],
        "instance": ["", "", ""],
        "somaSide": ["R", "L", "M"],
    })
    out, _ = vs._filter_neuron_df_by_hemisphere(ndf)
    assert sorted(out["bodyId"]) == [2, 3]


def test_hemisphere_both_keeps_everything():
    vs = object.__new__(VisualizeSkeleton)
    vs.hemisphere = "both"
    ndf = pd.DataFrame({
        "bodyId": [1, 2, 3],
        "type": ["A", "B", "C"],
        "instance": ["A_R", "B_L", "C"],
    })
    out, _ = vs._filter_neuron_df_by_hemisphere(ndf)
    assert len(out) == 3


def test_hemisphere_applied_after_getNeurons(tmp_path, monkeypatch):
    """__post_init__ filters every layer's neuron frame by hemisphere."""
    calls = {"search_columns": None}

    def fake_get_neurons(requiredNeurons, dataset="", custom_group_names=None,
                         client=None, verbose=True, search_columns="auto"):
        calls["search_columns"] = search_columns
        ndf = pd.DataFrame({
            "bodyId": [1, 2, 3],
            "type": ["A", "B", "C"],
            "instance": ["A_R", "B_L", "C"],
        })
        return ndf, pd.DataFrame(), "auto_name", None

    monkeypatch.setattr(vs_mod.sv, "getNeurons", fake_get_neurons)

    vs = VisualizeSkeleton(
        dataset="hemibrain:v1.2.1",
        neuron_layers=["KCg-m.*"],
        search_columns="bodyId",
        hemisphere="left",
        client=FakeClient(dataset="hemibrain:v1.2.1"),
        verbose=False,
        output_dir=str(tmp_path),
        include_timestamp=False,
    )
    assert calls["search_columns"] == "bodyId"
    assert sorted(vs.neuron_dfs[0]["bodyId"]) == [2, 3]  # B_L + C(U), A_R dropped


def test_invalid_hemisphere_rejected(tmp_path):
    with pytest.raises(ValueError):
        VisualizeSkeleton(
            dataset="hemibrain:v1.2.1",
            neuron_layers=["aMe12"],
            hemisphere="nope",
            client=FakeClient(dataset="hemibrain:v1.2.1"),
            verbose=False,
            output_dir=str(tmp_path),
        )


def test_search_columns_forwarded_to_getNeurons(tmp_path, monkeypatch):
    calls = {"layers": [], "dataset": None, "search_columns": None}

    def fake_get_neurons(requiredNeurons, dataset="", custom_group_names=None,
                         client=None, verbose=True, search_columns="auto"):
        # getNeurons is invoked once per layer with that layer's items.
        calls["layers"].append(list(requiredNeurons))
        calls["dataset"] = dataset
        calls["search_columns"] = search_columns
        return pd.DataFrame(), pd.DataFrame(), "auto_name", None

    monkeypatch.setattr(vs_mod.sv, "getNeurons", fake_get_neurons)

    VisualizeSkeleton(
        dataset="hemibrain:v1.2.1",
        neuron_layers=["KCg-m.*", "MBON01"],
        search_columns="bodyId",
        client=FakeClient(dataset="hemibrain:v1.2.1"),
        verbose=False,
        output_dir=str(tmp_path),
        include_timestamp=False,
    )

    assert calls["layers"] == [["KCg-m.*"], ["MBON01"]]
    assert calls["dataset"] == "hemibrain:v1.2.1"
    assert calls["search_columns"] == "bodyId"


def test_auto_scope_is_default_passthrough(tmp_path, monkeypatch):
    calls = {}

    def fake_get_neurons(requiredNeurons, dataset="", custom_group_names=None,
                         client=None, verbose=True, search_columns="auto"):
        calls["search_columns"] = search_columns
        return pd.DataFrame(), pd.DataFrame(), "auto_name", None

    monkeypatch.setattr(vs_mod.sv, "getNeurons", fake_get_neurons)

    VisualizeSkeleton(
        dataset="hemibrain:v1.2.1",
        neuron_layers=["aMe12"],
        client=FakeClient(dataset="hemibrain:v1.2.1"),
        verbose=False,
        output_dir=str(tmp_path),
        include_timestamp=False,
    )
    assert calls["search_columns"] == "auto"


def test_invalid_search_columns_rejected(tmp_path):
    with pytest.raises(ValueError):
        VisualizeSkeleton(
            dataset="hemibrain:v1.2.1",
            neuron_layers=["aMe12"],
            search_columns="nope",
            client=FakeClient(dataset="hemibrain:v1.2.1"),
            verbose=False,
            output_dir=str(tmp_path),
        )
