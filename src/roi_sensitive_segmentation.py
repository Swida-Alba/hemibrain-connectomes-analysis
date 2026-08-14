"""Geometry-derived ROI assignment for synapses and neuron skeletons.

The functions in this module deliberately do not consume NeuPrint's ``roi``
labels.  They classify coordinates against caller-provided ROI meshes so the
coordinate-derived result can be compared with source metadata or used as a
future input to ROI-sensitive profiling.

This is an exploratory implementation.  ROI meshes must be in the same
coordinate space as the points, and the skeleton segment occupancy is sampled
at configurable midpoints rather than calculated by exact mesh clipping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


OUTSIDE_ROI = "__outside__"
DEFAULT_COORDINATE_COLUMNS = ("x", "y", "z")
_OPEN3D_SCENE_CACHE: dict[int, Any] = {}


@dataclass(frozen=True)
class SkeletonSegmentation:
    """Node and edge-level ROI assignments for a skeleton."""

    nodes: pd.DataFrame
    segments: pd.DataFrame
    # Equal-length sampled subsegments are retained separately so a
    # visualization can color only the portion of an edge assigned to an ROI.
    samples: pd.DataFrame = field(default_factory=pd.DataFrame)


def _mesh_items(roi_meshes: Mapping[str, Any] | Sequence[tuple[str, Any]]):
    if isinstance(roi_meshes, Mapping):
        items = list(roi_meshes.items())
    else:
        items = list(roi_meshes)
    if not items:
        raise ValueError("roi_meshes must contain at least one ROI mesh")
    names = [str(name) for name, _ in items]
    if len(set(names)) != len(names):
        raise ValueError("ROI mesh names must be unique")
    return [(name, _coerce_mesh(mesh)) for name, mesh in items]


def _coerce_mesh(mesh: Any):
    """Return a mesh object exposing ``contains`` and ``bounds``."""
    if isinstance(mesh, (str, Path)):
        import trimesh

        mesh = trimesh.load_mesh(str(mesh), force="mesh", process=False)

    # navis.Volume and trimesh.Trimesh both expose vertices/faces.  Converting
    # through trimesh avoids relying on navis-specific point-in-volume code.
    if hasattr(mesh, "trimesh"):
        mesh = mesh.trimesh
    if not hasattr(mesh, "contains"):
        raise TypeError("Each ROI mesh must expose a trimesh-compatible contains() method")
    return mesh


def _points_array(points: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
                  coordinate_columns: Sequence[str] = DEFAULT_COORDINATE_COLUMNS):
    if isinstance(points, pd.DataFrame):
        missing = [column for column in coordinate_columns if column not in points.columns]
        if missing:
            raise ValueError(f"Missing coordinate columns: {missing}")
        values = points.loc[:, list(coordinate_columns)].to_numpy(dtype=float)
    else:
        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("points must be an (n, 3) array or a DataFrame with x/y/z columns")

    if not np.isfinite(values).all():
        raise ValueError("point coordinates must be finite")
    return values


def _open3d_scene(mesh: Any):
    """Return a cached Open3D raycasting scene for ``mesh`` when available."""
    try:
        import open3d
    except ImportError:
        return None, None

    cache_key = id(mesh)
    scene = _OPEN3D_SCENE_CACHE.get(cache_key)
    if scene is not None:
        return open3d, scene

    try:
        legacy = open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=float)),
            triangles=open3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
        )
        scene = open3d.t.geometry.RaycastingScene()
        scene.add_triangles(open3d.t.geometry.TriangleMesh.from_legacy(legacy))
        _OPEN3D_SCENE_CACHE[cache_key] = scene
        return open3d, scene
    except Exception:
        return open3d, None


def _contains(mesh: Any, points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)

    # Ray/triangle intersection allocates arrays proportional to both the
    # number of query points and the number of mesh faces.  Restricting calls
    # to the mesh bounding box and chunking them keeps large raw male-CNS ROI
    # meshes from exhausting memory during a skeleton-wide classification.
    candidate = np.ones(len(points), dtype=bool)
    try:
        bounds = np.asarray(mesh.bounds, dtype=float)
        if bounds.shape == (2, 3):
            candidate = (
                (points >= bounds[0]).all(axis=1)
                & (points <= bounds[1]).all(axis=1)
            )
    except Exception:
        pass

    result = np.zeros(len(points), dtype=bool)
    candidate_indices = np.flatnonzero(candidate)
    # Open3D's BVH-backed occupancy query is faster and less memory intensive
    # for the large raw meshes used by the pilot.  Keep trimesh as a fallback
    # for installations without the optional visualization stack.
    open3d, scene = _open3d_scene(mesh)

    chunk_size = 50_000 if scene is not None else 2_000
    try:
        for start in range(0, len(candidate_indices), chunk_size):
            indices = candidate_indices[start:start + chunk_size]
            query_points = points[indices]
            if scene is not None:
                tensor = open3d.core.Tensor(
                    query_points.astype(np.float32, copy=False),
                    dtype=open3d.core.Dtype.Float32,
                )
                occupancy = scene.compute_occupancy(tensor).numpy()
                result[indices] = np.asarray(occupancy, dtype=bool)
            else:
                result[indices] = np.asarray(mesh.contains(query_points), dtype=bool)
    except Exception as exc:
        raise RuntimeError(
            "ROI point-in-mesh classification failed. Check that the mesh is "
            "valid, sufficiently closed, and in the same coordinate space as the points."
        ) from exc
    return result


def _surface_distance(mesh: Any, points: np.ndarray) -> np.ndarray:
    """Return unsigned distance from each point to the closest mesh surface."""
    if len(points) == 0:
        return np.zeros(0, dtype=float)

    open3d, scene = _open3d_scene(mesh)
    if scene is not None:
        try:
            distances = []
            chunk_size = 50_000
            for start in range(0, len(points), chunk_size):
                query_points = points[start:start + chunk_size]
                tensor = open3d.core.Tensor(
                    query_points.astype(np.float32, copy=False),
                    dtype=open3d.core.Dtype.Float32,
                )
                distances.append(scene.compute_distance(tensor).numpy())
            return np.concatenate(distances).astype(float, copy=False)
        except Exception as exc:
            raise RuntimeError("ROI nearest-surface distance calculation failed") from exc

    try:
        import trimesh

        _ = trimesh  # Keep the dependency check explicit for the fallback below.
        from trimesh.proximity import closest_point

        _, distances, _ = closest_point(mesh, points)
        return np.asarray(distances, dtype=float)
    except Exception as exc:
        raise RuntimeError(
            "ROI nearest-surface distance calculation failed. Install Open3D or "
            "the trimesh proximity dependencies and check the mesh geometry."
        ) from exc


def _nearest_roi_assignments(
    points: np.ndarray,
    items: Sequence[tuple[str, Any]],
) -> pd.DataFrame:
    """Find the nearest ROI surface for every point."""
    if len(points) == 0:
        return pd.DataFrame(columns=["roi", "distance"])

    distances = np.column_stack([
        _surface_distance(mesh, points) for _, mesh in items
    ])
    nearest_indices = np.argmin(distances, axis=1)
    return pd.DataFrame({
        "roi": [items[index][0] for index in nearest_indices],
        "distance": distances[np.arange(len(points)), nearest_indices],
    })


def containment_backend() -> str:
    """Return the point-in-mesh backend available to the current process."""
    try:
        import open3d  # noqa: F401

        return "open3d_raycasting_scene"
    except ImportError:
        return "trimesh_ray_triangle"


def classify_points_by_rois(
    points: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    roi_meshes: Mapping[str, Any] | Sequence[tuple[str, Any]],
    *,
    coordinate_columns: Sequence[str] = DEFAULT_COORDINATE_COLUMNS,
    overlap: str = "all",
    outside_label: str | None = OUTSIDE_ROI,
) -> pd.DataFrame:
    """Classify points against ROI meshes.

    Parameters
    ----------
    points:
        DataFrame containing the coordinate columns or an ``(n, 3)`` array.
    roi_meshes:
        Ordered mapping or sequence of ``(roi_name, mesh)`` pairs.  The order
        is the priority order when ``overlap='first'``.
    overlap:
        ``'all'`` returns one row for every containing ROI, ``'first'`` keeps
        the first containing ROI, and ``'error'`` rejects points in multiple
        meshes.
    outside_label:
        Label emitted for points in no mesh.  Pass ``None`` to omit them.

    Returns
    -------
    pandas.DataFrame
        At least ``point_index``, ``roi``, and ``inside`` columns.  Overlaps
        therefore intentionally produce multiple rows for one point.
    """
    if overlap not in {"all", "first", "error"}:
        raise ValueError("overlap must be 'all', 'first', or 'error'")

    values = _points_array(points, coordinate_columns)
    items = _mesh_items(roi_meshes)
    containing = np.column_stack([_contains(mesh, values) for _, mesh in items])
    match_counts = containing.sum(axis=1)

    if overlap == "error" and np.any(match_counts > 1):
        indices = np.flatnonzero(match_counts > 1).tolist()
        raise ValueError(f"Points lie in overlapping ROI meshes: indices {indices[:10]}")

    rows: list[dict[str, Any]] = []
    for point_index in range(len(values)):
        matched_indices = np.flatnonzero(containing[point_index]).tolist()
        if overlap == "first" and matched_indices:
            matched_indices = matched_indices[:1]
        if not matched_indices:
            if outside_label is not None:
                rows.append({
                    "point_index": point_index,
                    "roi": outside_label,
                    "inside": False,
                })
            continue
        for mesh_index in matched_indices:
            rows.append({
                "point_index": point_index,
                "roi": items[mesh_index][0],
                "inside": True,
            })

    return pd.DataFrame(rows, columns=["point_index", "roi", "inside"])


def segment_synapses(
    synapses: pd.DataFrame,
    roi_meshes: Mapping[str, Any] | Sequence[tuple[str, Any]],
    *,
    coordinate_columns: Sequence[str] = DEFAULT_COORDINATE_COLUMNS,
    overlap: str = "all",
    outside_label: str | None = OUTSIDE_ROI,
    snap_outside: bool = False,
    max_snap_distance: float | None = None,
) -> pd.DataFrame:
    """Return synapses annotated with geometry-derived and optional snapped ROIs.

    ``derived_roi`` is always the direct point-in-mesh result.  When
    ``snap_outside`` is enabled, points whose direct result is ``outside_label``
    receive the nearest ROI surface in ``snapped_roi``.  The original result is
    retained, and ``was_snapped``, ``nearest_roi`` and ``nearest_roi_distance``
    make the fallback explicit.  ``max_snap_distance`` can prevent implausibly
    distant assignments; it is measured in the mesh coordinate units.
    """
    if not isinstance(synapses, pd.DataFrame):
        raise TypeError("synapses must be a pandas DataFrame")
    if max_snap_distance is not None:
        if not np.isfinite(max_snap_distance) or max_snap_distance < 0:
            raise ValueError("max_snap_distance must be a finite non-negative number")
        if not snap_outside:
            raise ValueError("max_snap_distance requires snap_outside=True")

    assignments = classify_points_by_rois(
        synapses,
        roi_meshes,
        coordinate_columns=coordinate_columns,
        overlap=overlap,
        outside_label=outside_label,
    )
    if assignments.empty:
        result = synapses.iloc[0:0].copy()
        result["point_index"] = pd.Series(dtype="int64")
        result["derived_roi"] = pd.Series(dtype="object")
        result["inside"] = pd.Series(dtype="bool")
        result["nearest_roi"] = pd.Series(dtype="object")
        result["nearest_roi_distance"] = pd.Series(dtype="float64")
        result["snapped_roi"] = pd.Series(dtype="object")
        result["was_snapped"] = pd.Series(dtype="bool")
        return result

    result = synapses.iloc[assignments["point_index"].to_numpy()].copy()
    result.insert(0, "point_index", assignments["point_index"].to_numpy())
    result["derived_roi"] = assignments["roi"].to_numpy()
    result["inside"] = assignments["inside"].to_numpy(dtype=bool)
    result["nearest_roi"] = pd.Series([None] * len(result), dtype="object")
    result["nearest_roi_distance"] = np.nan
    result["snapped_roi"] = result["derived_roi"].astype(object)
    result["was_snapped"] = False

    if snap_outside:
        items = _mesh_items(roi_meshes)
        outside_rows = result.index[~result["inside"].astype(bool)]
        if len(outside_rows):
            point_values = result.loc[outside_rows, list(coordinate_columns)].to_numpy(dtype=float)
            nearest = _nearest_roi_assignments(point_values, items)
            result.loc[outside_rows, "nearest_roi"] = nearest["roi"].to_numpy()
            result.loc[outside_rows, "nearest_roi_distance"] = nearest["distance"].to_numpy()
            eligible = np.ones(len(outside_rows), dtype=bool)
            if max_snap_distance is not None:
                eligible = nearest["distance"].to_numpy(dtype=float) <= max_snap_distance
            eligible_rows = outside_rows.to_numpy()[eligible]
            result.loc[eligible_rows, "snapped_roi"] = nearest.loc[eligible, "roi"].to_numpy()
            result.loc[eligible_rows, "was_snapped"] = True
    result.reset_index(drop=True, inplace=True)
    return result


def _find_column(frame: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"Could not find {label} column; tried {list(candidates)}")


def _skeleton_nodes(skeleton: Any) -> pd.DataFrame:
    if isinstance(skeleton, pd.DataFrame):
        nodes = skeleton.copy()
    elif hasattr(skeleton, "nodes"):
        nodes = skeleton.nodes.copy()
    else:
        raise TypeError("skeleton must be a DataFrame or a navis.TreeNeuron-like object")

    node_column = _find_column(nodes, ("node_id", "rowId", "nodeId", "id"), "node ID")
    parent_column = _find_column(
        nodes,
        ("parent_id", "parentId", "parent", "link"),
        "parent ID",
    )
    x_column = _find_column(nodes, ("x",), "x")
    y_column = _find_column(nodes, ("y",), "y")
    z_column = _find_column(nodes, ("z",), "z")

    normalized = nodes.copy()
    normalized["node_id"] = normalized[node_column].astype("int64")
    normalized["parent_id"] = pd.to_numeric(normalized[parent_column], errors="coerce")
    normalized["x"] = normalized[x_column].astype(float)
    normalized["y"] = normalized[y_column].astype(float)
    normalized["z"] = normalized[z_column].astype(float)
    if normalized[["x", "y", "z"]].isna().any().any():
        raise ValueError("skeleton coordinates must be finite")
    return normalized


def _edge_table(nodes: pd.DataFrame) -> pd.DataFrame:
    by_id = nodes.set_index("node_id")
    edges = []
    for row in nodes.itertuples(index=False):
        if pd.isna(row.parent_id):
            continue
        parent_id = int(row.parent_id)
        if parent_id < 0 or parent_id not in by_id.index or parent_id == int(row.node_id):
            continue
        parent = by_id.loc[parent_id]
        child = np.asarray([row.x, row.y, row.z], dtype=float)
        parent_xyz = np.asarray([parent.x, parent.y, parent.z], dtype=float)
        vector = child - parent_xyz
        length = float(np.linalg.norm(vector))
        edges.append({
            "parent_id": parent_id,
            "node_id": int(row.node_id),
            "parent_x": parent_xyz[0],
            "parent_y": parent_xyz[1],
            "parent_z": parent_xyz[2],
            "x": child[0],
            "y": child[1],
            "z": child[2],
            "length": length,
        })
    return pd.DataFrame(edges)


def segment_skeleton(
    skeleton: Any,
    roi_meshes: Mapping[str, Any] | Sequence[tuple[str, Any]],
    *,
    segment_samples: int = 11,
    overlap: str = "all",
    outside_label: str | None = OUTSIDE_ROI,
) -> SkeletonSegmentation:
    """Segment skeleton nodes and estimate parent-child edge occupancy.

    ``segment_samples`` is the number of equal-length midpoint samples per
    edge.  A value of 11 is appropriate for preliminary inspection; exact
    boundary clipping should replace this approximation for production use.
    """
    if segment_samples < 1:
        raise ValueError("segment_samples must be at least 1")
    if overlap not in {"all", "first", "error"}:
        raise ValueError("overlap must be 'all', 'first', or 'error'")

    nodes = _skeleton_nodes(skeleton)
    items = _mesh_items(roi_meshes)

    node_assignments = classify_points_by_rois(
        nodes[["x", "y", "z"]],
        items,
        overlap=overlap,
        outside_label=outside_label,
    )
    node_result = nodes.iloc[node_assignments["point_index"].to_numpy()].copy()
    node_result.insert(0, "point_index", node_assignments["point_index"].to_numpy())
    node_result["derived_roi"] = node_assignments["roi"].to_numpy()
    node_result["inside"] = node_assignments["inside"].to_numpy(dtype=bool)
    node_result.reset_index(drop=True, inplace=True)

    edges = _edge_table(nodes)
    segment_columns = [
        "segment_index", "parent_id", "node_id", "derived_roi", "inside",
        "length", "fraction_inside", "length_inside",
        "parent_x", "parent_y", "parent_z", "x", "y", "z",
    ]
    sample_columns = [
        "segment_index", "parent_id", "node_id", "sample_index",
        "derived_roi", "inside", "length", "length_inside",
        "start_x", "start_y", "start_z", "end_x", "end_y", "end_z",
    ]
    if edges.empty:
        return SkeletonSegmentation(
            node_result,
            pd.DataFrame(columns=segment_columns),
            pd.DataFrame(columns=sample_columns),
        )

    midpoint_t = (np.arange(segment_samples, dtype=float) + 0.5) / segment_samples
    starts = edges[["parent_x", "parent_y", "parent_z"]].to_numpy(float)
    vectors = edges[["x", "y", "z"]].to_numpy(float) - starts
    midpoint_points = (
        starts[:, None, :] + midpoint_t[None, :, None] * vectors[:, None, :]
    ).reshape(-1, 3)
    containing = np.column_stack([_contains(mesh, midpoint_points) for _, mesh in items])
    containing = containing.reshape(len(edges), segment_samples, len(items))
    match_counts = containing.sum(axis=2)

    if overlap == "error" and np.any(match_counts > 1):
        indices = np.flatnonzero(match_counts > 1).tolist()
        raise ValueError(f"Skeleton segments cross overlapping ROI meshes: samples {indices[:10]}")

    rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for edge_index, edge in edges.iterrows():
        if overlap == "first":
            labels = np.full(segment_samples, -1, dtype=int)
            has_match = match_counts[edge_index] > 0
            if np.any(has_match):
                labels[has_match] = np.argmax(containing[edge_index, has_match], axis=1)
            per_roi = {
                roi: float(np.mean(labels == roi_index))
                for roi_index, (roi, _) in enumerate(items)
                if np.any(labels == roi_index)
            }
            outside_fraction = float(np.mean(labels < 0))
        else:
            per_roi = {
                roi: float(np.mean(containing[edge_index, :, roi_index]))
                for roi_index, (roi, _) in enumerate(items)
                if np.any(containing[edge_index, :, roi_index])
            }
            outside_fraction = float(np.mean(match_counts[edge_index] == 0))

        if outside_label is not None and outside_fraction > 0:
            per_roi[outside_label] = outside_fraction

        edge_length = float(edge["length"])
        for roi, fraction in per_roi.items():
            if fraction <= 0:
                continue
            rows.append({
                "segment_index": int(edge_index),
                "parent_id": int(edge["parent_id"]),
                "node_id": int(edge["node_id"]),
                "derived_roi": roi,
                "inside": roi != outside_label,
                "length": edge_length,
                "fraction_inside": fraction,
                "length_inside": edge_length * fraction,
                "parent_x": float(edge["parent_x"]),
                "parent_y": float(edge["parent_y"]),
                "parent_z": float(edge["parent_z"]),
                "x": float(edge["x"]),
                "y": float(edge["y"]),
                "z": float(edge["z"]),
            })

        # Retain the individual sampled subsegments for faithful plotting.
        # The midpoint label is applied to its surrounding equal-length bin;
        # this is the same approximation used by fraction_inside.
        for sample_index in range(segment_samples):
            if overlap == "first":
                roi_indices = (
                    [] if labels[sample_index] < 0 else [int(labels[sample_index])]
                )
            else:
                roi_indices = np.flatnonzero(
                    containing[edge_index, sample_index]
                ).tolist()
            if not roi_indices and outside_label is not None:
                sample_rois = [(outside_label, False)]
            else:
                sample_rois = [(items[index][0], True) for index in roi_indices]

            start_t = sample_index / segment_samples
            end_t = (sample_index + 1) / segment_samples
            start_xyz = np.asarray(
                [edge["parent_x"], edge["parent_y"], edge["parent_z"]],
                dtype=float,
            ) + start_t * np.asarray(
                [edge["x"] - edge["parent_x"], edge["y"] - edge["parent_y"], edge["z"] - edge["parent_z"]],
                dtype=float,
            )
            end_xyz = np.asarray(
                [edge["parent_x"], edge["parent_y"], edge["parent_z"]],
                dtype=float,
            ) + end_t * np.asarray(
                [edge["x"] - edge["parent_x"], edge["y"] - edge["parent_y"], edge["z"] - edge["parent_z"]],
                dtype=float,
            )
            for roi, inside in sample_rois:
                sample_rows.append({
                    "segment_index": int(edge_index),
                    "parent_id": int(edge["parent_id"]),
                    "node_id": int(edge["node_id"]),
                    "sample_index": int(sample_index),
                    "derived_roi": roi,
                    "inside": bool(inside),
                    "length": edge_length / segment_samples,
                    "length_inside": edge_length / segment_samples if inside else 0.0,
                    "start_x": start_xyz[0],
                    "start_y": start_xyz[1],
                    "start_z": start_xyz[2],
                    "end_x": end_xyz[0],
                    "end_y": end_xyz[1],
                    "end_z": end_xyz[2],
                })

    segment_result = pd.DataFrame(rows, columns=segment_columns)
    sample_result = pd.DataFrame(sample_rows, columns=sample_columns)
    return SkeletonSegmentation(node_result, segment_result, sample_result)


__all__ = [
    "OUTSIDE_ROI",
    "SkeletonSegmentation",
    "classify_points_by_rois",
    "containment_backend",
    "segment_synapses",
    "segment_skeleton",
]
