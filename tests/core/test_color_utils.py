"""Color-format and alpha-inheritance regression tests."""

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.color_utils import (  # noqa: E402
    color_has_explicit_alpha,
    extract_rgba_tuple,
    standardize_color,
    standardize_color_list,
)
from visualize_skeleton import VisualizeSkeleton  # noqa: E402


@pytest.mark.parametrize(
    ("value", "expected_rgb", "expected_alpha"),
    [
        ("red", (255, 0, 0), 0.35),
        ("#0f08", (0, 255, 0), 8 / 15),
        ("#ff000080", (255, 0, 0), 128 / 255),
        ((255, 127, 0), (255, 127, 0), 0.35),
        ((1.0, 0.5, 0.0), (255, 128, 0), 0.35),
        ("rgb(255, 127, 0)", (255, 127, 0), 0.35),
        ("rgb(1.0, 0.5, 0.0)", (255, 128, 0), 0.35),
        ("rgba(255, 0, 0, 0.25)", (255, 0, 0), 0.25),
        ("rgb(255 0 0 / 50%)", (255, 0, 0), 0.5),
        ("rgb(100% 0% 50% / 25%)", (255, 0, 128), 0.25),
        ("rgb(255, 0, 0, 128)", (255, 0, 0), 128 / 255),
        ("hsl(120, 100%, 50%)", (0, 255, 0), 0.35),
        ("hsla(240 100% 50% / 25%)", (0, 0, 255), 0.25),
        ("rebeccapurple", (102, 51, 153), 0.35),
    ],
)
def test_standardize_color_accepts_common_formats(value, expected_rgb, expected_alpha):
    rgba = extract_rgba_tuple(value, default_alpha=0.35)
    assert rgba[:3] == expected_rgb
    assert rgba[3] == pytest.approx(expected_alpha)
    assert standardize_color(value, default_alpha=0.35).startswith("rgba(")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("red", False),
        ("#ff0000", False),
        ("#ff000080", True),
        ("rgba (255, 0, 0, 0.5)", True),
        ("rgb(255 0 0 / 50%)", True),
        ("rgb(255, 0, 0, 50%)", True),
        ((255, 0, 0), False),
        ((255, 0, 0, 128), True),
        ((255, 0, 0, "50%"), True),
    ],
)
def test_explicit_alpha_detection_distinguishes_inherited_alpha(value, expected):
    assert color_has_explicit_alpha(value) is expected


def test_mixed_color_list_keeps_global_alpha_for_colors_without_alpha():
    colors = ["#ff0000", "rgba(0, 255, 0, 0.25)", (0.0, 0.0, 1.0)]
    standardized = standardize_color_list(colors, default_alpha=0.4)
    assert [extract_rgba_tuple(color)[3] for color in standardized] == pytest.approx(
        [0.4, 0.25, 0.4]
    )


def test_single_rgb_tuple_is_one_color_in_color_list_helper():
    assert standardize_color_list((255, 0, 0), default_alpha=0.2) == [
        "rgba(255, 0, 0, 0.2)"
    ]


def test_rgba_tuple_percentage_alpha_is_decoded_as_one_color():
    assert extract_rgba_tuple([255, 0, 0, "50%"], default_alpha=0.2) == pytest.approx(
        (255, 0, 0, 0.5)
    )


def test_visualize_skeleton_helpers_preserve_mixed_alpha_and_global_fallback():
    visualizer = object.__new__(VisualizeSkeleton)
    visualizer._vprint = lambda *args, **kwargs: None

    colors = visualizer._standardize_color_input(
        ["#ff0000", "rgba(0, 255, 0, 0.25)", (0.0, 0.0, 1.0)],
        default_alpha=0.4,
    )
    assert [extract_rgba_tuple(color)[3] for color in colors] == pytest.approx(
        [0.4, 0.25, 0.4]
    )
    assert visualizer._standardize_color_input(
        [255, 0, 0], default_alpha=0.4
    ) == ["rgba(255, 0, 0, 0.4)"]
    assert visualizer._standardize_color_input(
        [255, 0, 0, "50%"], default_alpha=0.4
    ) == ["rgba(255, 0, 0, 0.5)"]

    # Expansion must retain each supplied layer's alpha instead of replacing
    # all colors with an average opacity.
    expanded = visualizer._interpolate_colors(
        ["rgba(255, 0, 0, 0.2)", "rgba(0, 0, 255, 0.8)"],
        4,
    )
    assert [extract_rgba_tuple(color)[3] for color in expanded[:2]] == pytest.approx(
        [0.2, 0.8]
    )
    assert [extract_rgba_tuple(color)[3] for color in expanded[2:]] == pytest.approx(
        [0.5, 0.5]
    )


def test_continuous_backend_sampling_spans_selected_range():
    visualizer = object.__new__(VisualizeSkeleton)
    selected = [f"color-{index}" for index in range(101)]

    sampled = visualizer._expand_color_sequence(
        selected,
        3,
        continuous=True,
        warn=False,
    )

    assert sampled == (selected[0], selected[50], selected[-1])


def test_continuous_palette_payload_is_unwrapped_by_backend():
    visualizer = object.__new__(VisualizeSkeleton)
    payload = {
        "colors": [f"color-{index}" for index in range(101)],
        "continuous": True,
    }

    colors, continuous = visualizer._unwrap_palette_value(payload)
    assert continuous is True
    assert colors[0] == "color-0"
    assert colors[-1] == "color-100"


def test_mesh_color_helper_accepts_byte_alpha_and_mixed_formats():
    visualizer = object.__new__(VisualizeSkeleton)
    assert visualizer._get_opaque_color("rgba(1, 2, 3, 0.2)") == "rgba(1, 2, 3, 1.0)"
    rgba = visualizer._standardize_mesh_color_input([255, 0, 0, 128], default_alpha=0.2)
    assert extract_rgba_tuple(rgba)[3] == pytest.approx(128 / 255)

    mixed = visualizer._standardize_mesh_color_input(
        ["#00ff00", "rgba(0, 0, 255, 0.3)"], default_alpha=0.2
    )
    assert [extract_rgba_tuple(color)[3] for color in mixed] == pytest.approx(
        [0.2, 0.3]
    )

    assert visualizer._standardize_mesh_color_input(
        [255, 0, 0, "50%"], default_alpha=0.2
    ) == "rgba(255, 0, 0, 0.5)"


def test_backend_render_and_export_decoders_use_canonical_rgba():
    visualizer = object.__new__(VisualizeSkeleton)

    import plotly.graph_objects as go

    trace = go.Mesh3d(x=[0], y=[0], z=[0], i=[0], j=[0], k=[0])
    visualizer._apply_plotly_trace_color(trace, "hsl(0.5turn 100% 50% / 25%)")
    assert trace.color == "#00ffff"
    assert trace.opacity == pytest.approx(0.25)

    class K3DObject:
        color = 0
        opacity = 1.0

    obj = K3DObject()
    visualizer._apply_k3d_object_color(obj, "rgb(100% 0% 50% / 50%)")
    assert obj.color == (255 << 16) + (0 << 8) + 128
    assert obj.opacity == pytest.approx(0.5)

    assert visualizer._to_rgba("#00ff00", alpha=128).tolist() == [0, 255, 0, 128]
    assert visualizer._to_rgba("hsl(0.5turn 100% 50%)", alpha="50%").tolist() == [0, 255, 255, 128]
    visualizer.background_color = "hsl(240 100% 50% / 25%)"
    assert visualizer._is_dark_background() is True
    assert visualizer._is_dark_background("rgb(100% 100% 100% / 50%)") is False
