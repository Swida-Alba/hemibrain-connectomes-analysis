"""Unit tests for coordinate-derived ROI segmentation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from roi_sensitive_segmentation import (  # noqa: E402
    OUTSIDE_ROI,
    classify_points_by_rois,
    segment_skeleton,
    segment_synapses,
)


def cube(center=(0, 0, 0), extents=(2, 2, 2)):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def test_synapses_are_classified_by_coordinates_not_source_roi():
    synapses = pd.DataFrame(
        {
            "bodyId": [1, 1, 1],
            "type": ["pre", "post", "pre"],
            "x": [0.0, 3.0, 20.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
            "roi": ["wrong-source-label", None, "not-used"],
        }
    )
    result = segment_synapses(
        synapses,
        {"center": cube(), "right": cube(center=(3, 0, 0))},
        overlap="first",
    )

    assert result["derived_roi"].tolist() == ["center", "right", OUTSIDE_ROI]
    assert result["roi"].tolist() == ["wrong-source-label", None, "not-used"]
    assert result["inside"].tolist() == [True, True, False]


def test_outside_synapses_can_be_snapped_to_nearest_roi_surface():
    synapses = pd.DataFrame(
        {
            "x": [4.0, 20.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
        }
    )
    result = segment_synapses(
        synapses,
        {"left": cube(), "right": cube(center=(10, 0, 0))},
        overlap="first",
        snap_outside=True,
        max_snap_distance=3.0,
    )

    assert result["derived_roi"].tolist() == [OUTSIDE_ROI, OUTSIDE_ROI]
    assert result["nearest_roi"].tolist() == ["left", "right"]
    assert np.allclose(result["nearest_roi_distance"], [3.0, 9.0], atol=1e-6)
    assert result["snapped_roi"].tolist() == ["left", OUTSIDE_ROI]
    assert result["was_snapped"].tolist() == [True, False]


def test_overlapping_synapse_rois_can_be_preserved_or_rejected():
    synapses = pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0]})
    meshes = {"first": cube(), "second": cube()}

    all_result = segment_synapses(synapses, meshes, overlap="all")
    assert all_result["derived_roi"].tolist() == ["first", "second"]

    with pytest.raises(ValueError, match="overlapping"):
        classify_points_by_rois(synapses, meshes, overlap="error")


def test_skeleton_nodes_and_crossing_segments_are_segmented():
    skeleton = pd.DataFrame(
        {
            "rowId": [0, 1, 2],
            "parentId": [-1, 0, 1],
            "x": [-2.0, 0.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
        }
    )
    result = segment_skeleton(
        skeleton,
        {"center": cube()},
        segment_samples=100,
        overlap="first",
    )

    node_labels = result.nodes.sort_values("node_id")["derived_roi"].tolist()
    assert node_labels == [OUTSIDE_ROI, "center", OUTSIDE_ROI]

    center_segments = result.segments[result.segments["derived_roi"] == "center"]
    assert len(center_segments) == 2
    assert np.allclose(center_segments["fraction_inside"], 0.5, atol=0.02)
    assert np.allclose(center_segments["length_inside"], 1.0, atol=0.05)
    assert {"parent_x", "parent_y", "parent_z", "x", "y", "z"}.issubset(
        result.segments.columns
    )
    center_samples = result.samples[result.samples["derived_roi"] == "center"]
    assert len(center_samples) == 100
    assert (center_samples["start_x"] >= -1.0).all()
    assert (center_samples["end_x"] <= 1.0).all()


def test_segment_samples_must_be_positive():
    skeleton = pd.DataFrame(
        {"node_id": [0], "parent_id": [-1], "x": [0.0], "y": [0.0], "z": [0.0]}
    )
    with pytest.raises(ValueError, match="segment_samples"):
        segment_skeleton(skeleton, {"center": cube()}, segment_samples=0)
