"""Fetch and visualize the male-CNS aMe12 ROI-segmentation pilot.

The generated files live under ``preliminary_ROI_sensitive_segmentation/``
and are intentionally ignored except for the checked-in plan.md.  The script
uses raw NeuPrint API responses for skeletons, synapses, and ROI meshes; it
does not read the local connection or ROI-mesh caches.  Successful ROI meshes
are serialized as unsimplified navis Volume JSON artifacts.

Run from the repository root:

    python scripts/preliminary_roi_sensitive_segmentation.py

Use ``--reuse`` to skip API downloads when the raw artifacts already exist.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from roi_sensitive_segmentation import (  # noqa: E402
    OUTSIDE_ROI,
    containment_backend,
    segment_skeleton,
    segment_synapses,
)
from utils.token_manager import token_manager  # noqa: E402


DATASET = "male-cns:v1.0"
SERVER = "https://neuprint.janelia.org"
QUERY_TYPE = "aMe12"
OUTPUT_ROOT = PROJECT_ROOT / "preliminary_ROI_sensitive_segmentation"
RAW_ROOT = OUTPUT_ROOT / "raw"
SKELETON_ROOT = RAW_ROOT / "skeletons"
MESH_ROOT = RAW_ROOT / "roi_meshes"
MESH_JSON_ROOT = RAW_ROOT / "roi_meshes_json"
SEGMENTED_ROOT = OUTPUT_ROOT / "segmented"
VIS_ROOT = OUTPUT_ROOT / "visualization"


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def _ensure_dirs():
    for path in (
        RAW_ROOT,
        SKELETON_ROOT,
        MESH_ROOT,
        MESH_JSON_ROOT,
        SEGMENTED_ROOT,
        VIS_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _load_client():
    from neuprint import Client

    token = token_manager.get_token("NEUPRINT_TOKEN")
    if not token:
        raise RuntimeError("NEUPRINT_TOKEN is not configured")
    client = Client(SERVER, dataset=DATASET, token=token)
    return client


def _fetch_raw(client, *, reuse: bool = False):
    from neuprint import NeuronCriteria, SynapseCriteria, fetch_neurons, fetch_synapses

    metadata_path = RAW_ROOT / "aMe12_neurons.parquet"
    roi_counts_path = RAW_ROOT / "aMe12_roi_counts.parquet"
    synapses_path = RAW_ROOT / "aMe12_synapses_raw.parquet"

    if reuse and metadata_path.exists() and synapses_path.exists():
        metadata = pd.read_parquet(metadata_path)
        roi_counts = pd.read_parquet(roi_counts_path) if roi_counts_path.exists() else pd.DataFrame()
        synapses = pd.read_parquet(synapses_path)
    else:
        metadata, roi_counts = fetch_neurons(
            NeuronCriteria(type=QUERY_TYPE),
            client=client,
        )
        metadata.to_parquet(metadata_path, index=False)
        roi_counts.to_parquet(roi_counts_path, index=False)
        body_ids = metadata["bodyId"].astype("int64").tolist()
        synapses = fetch_synapses(
            NeuronCriteria(bodyId=body_ids),
            SynapseCriteria(primary_only=True),
            batch_size=2,
            client=client,
        )
        synapses.to_parquet(synapses_path, index=False)
        synapses.to_csv(RAW_ROOT / "aMe12_synapses_raw.csv", index=False)

    body_ids = metadata["bodyId"].astype("int64").tolist()
    skeleton_records = []
    from neuprint import skeleton_swc_to_df

    for body_id in body_ids:
        swc_path = SKELETON_ROOT / f"{body_id}.swc"
        node_path = SKELETON_ROOT / f"{body_id}.parquet"
        if reuse and swc_path.exists() and node_path.exists():
            nodes = pd.read_parquet(node_path)
            swc_text = swc_path.read_text(encoding="utf-8")
        else:
            swc_text = client.fetch_skeleton(
                int(body_id),
                heal=False,
                format="swc",
            )
            swc_path.write_text(swc_text, encoding="utf-8")
            nodes = skeleton_swc_to_df(swc_text)
            nodes.insert(0, "bodyId", int(body_id))
            nodes.to_parquet(node_path, index=False)
        skeleton_records.append(
            {
                "bodyId": int(body_id),
                "nodes": int(len(nodes)),
                "swc_sha256": _sha256(swc_path),
                "swc_path": str(swc_path.relative_to(OUTPUT_ROOT)),
                "node_path": str(node_path.relative_to(OUTPUT_ROOT)),
            }
        )

    # Candidate meshes come from the complete primary-ROI catalogue, not from
    # the ROIs already assigned to aMe12 synapses.  The latter is retained in
    # the manifest only for comparison.
    synapse_api_roi_names = sorted(str(roi) for roi in synapses["roi"].dropna().unique())
    roi_names = sorted(str(roi) for roi in client.primary_rois)
    roi_records = []
    import trimesh
    import navis

    for roi_index, roi in enumerate(roi_names):
        json_path = MESH_JSON_ROOT / f"{roi_index:03d}_{_safe_name(roi)}.json"
        record = {"roi": roi, "path": str(json_path.relative_to(OUTPUT_ROOT))}
        try:
            api_response_bytes = None
            if reuse and json_path.exists():
                volume = navis.Volume.from_json(str(json_path), name=roi, units="nm")
                mesh = volume
            else:
                api_response = client.fetch_roi_mesh(roi)
                api_response_bytes = len(api_response)
                mesh = trimesh.load_mesh(
                    io.BytesIO(api_response),
                    file_type="obj",
                    process=False,
                )
                # JSON contains the complete API mesh vertices/faces without
                # simplification and is directly readable by navis and the
                # visualizer's ROI loader.
                volume = navis.Volume(mesh, name=roi, units="nm")
                volume.to_json(str(json_path))
            record.update(
                {
                    "success": True,
                    "json_bytes": int(json_path.stat().st_size),
                    "api_response_bytes": api_response_bytes,
                    "vertices": int(len(mesh.vertices)),
                    "faces": int(len(mesh.faces)),
                    "watertight": bool(mesh.is_watertight),
                    "winding_consistent": bool(mesh.is_winding_consistent),
                    "sha256": _sha256(json_path),
                    "serialization": "navis.Volume.to_json",
                }
            )
        except Exception as exc:
            record.update({"success": False, "error": f"{type(exc).__name__}: {exc}"})
            if json_path.exists() and not reuse:
                json_path.unlink()
        roi_records.append(record)

    # The template is a local flybrains reference mesh in native male-CNS
    # template coordinates.  API-fetched ROI geometries are serialized
    # separately and remain untouched by the template export.
    template_path = RAW_ROOT / "template_JRCFIB2022M.ply"
    template_record = {"path": str(template_path.relative_to(OUTPUT_ROOT))}
    try:
        import flybrains

        template_mesh = flybrains.JRCFIB2022M.mesh
        if not (reuse and template_path.exists()):
            template_mesh.export(str(template_path), file_type="ply")
        template_record.update(
            {
                "success": True,
                "vertices": int(len(template_mesh.vertices)),
                "faces": int(len(template_mesh.faces)),
                "sha256": _sha256(template_path),
                "source": "flybrains.JRCFIB2022M.mesh",
            }
        )
    except Exception as exc:
        template_record.update({"success": False, "error": f"{type(exc).__name__}: {exc}"})

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET,
        "server": SERVER,
        "query_type": QUERY_TYPE,
        "body_ids": body_ids,
        "skeleton_fetch": {"format": "swc", "heal": False, "simplification": None},
        "synapse_fetch": {"primary_only": True, "coordinates": ["x", "y", "z"]},
        "coordinate_space_raw": "JRCFIB2022Mraw",
        "coordinate_space_template": "JRCFIB2022M",
        "synapse_rows": int(len(synapses)),
        "synapse_types": synapses["type"].value_counts(dropna=False).to_dict(),
        "roi_mesh_candidate_source": "client.primary_rois",
        "roi_mesh_candidate_count": len(roi_names),
        "synapse_api_roi_names": synapse_api_roi_names,
        "synapse_api_roi_null_rows": int(synapses["roi"].isna().sum()),
        "skeletons": skeleton_records,
        "roi_meshes": roi_records,
        "template_mesh": template_record,
        "notes": [
            "ROI mesh fetches use Client.fetch_roi_mesh directly; local ROI caches are not read.",
            "Some API ROI labels may not have downloadable meshes; those failures are retained here.",
            "Geometry-derived segmentation ignores the source roi column.",
        ],
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return metadata, synapses, manifest


def _load_meshes(manifest):
    import trimesh
    import navis

    meshes = {}
    for record in manifest["roi_meshes"]:
        if not record.get("success"):
            continue
        path = OUTPUT_ROOT / record["path"]
        if path.suffix.lower() == ".json":
            # Volume.from_json returns a navis.Volume directly; unlike a
            # freshly constructed Volume it does not expose ``.trimesh`` in
            # all supported navis versions.  The segmentation layer can use
            # its vertices/faces/contains interface without conversion.
            meshes[record["roi"]] = navis.Volume.from_json(
                str(path), name=record["roi"], units="nm"
            )
        else:
            meshes[record["roi"]] = trimesh.load_mesh(str(path), force="mesh", process=False)
    if not meshes:
        raise RuntimeError("No ROI meshes were fetched successfully")
    return meshes


def _segment_saved_inputs(
    metadata,
    synapses,
    manifest,
    *,
    segment_samples: int = 3,
    snap_outside: bool = False,
    max_snap_distance: float | None = None,
):
    meshes = _load_meshes(manifest)
    segmented_synapses = segment_synapses(
        synapses,
        meshes,
        overlap="first",
        outside_label=OUTSIDE_ROI,
        snap_outside=snap_outside,
        max_snap_distance=max_snap_distance,
    )
    segmented_synapses.to_parquet(SEGMENTED_ROOT / "aMe12_synapses_by_geometry.parquet", index=False)
    segmented_synapses.to_csv(SEGMENTED_ROOT / "aMe12_synapses_by_geometry.csv", index=False)
    if snap_outside:
        segmented_synapses.to_parquet(
            SEGMENTED_ROOT / "aMe12_synapses_by_geometry_snapped.parquet",
            index=False,
        )
        segmented_synapses.to_csv(
            SEGMENTED_ROOT / "aMe12_synapses_by_geometry_snapped.csv",
            index=False,
        )

    skeleton_records = []
    skeleton_results = {}
    skeleton_passed_rois = set()
    for body_id in metadata["bodyId"].astype("int64"):
        skeleton_path = SKELETON_ROOT / f"{body_id}.parquet"
        nodes = pd.read_parquet(skeleton_path)
        result = segment_skeleton(nodes, meshes, segment_samples=segment_samples, overlap="first")
        skeleton_results[int(body_id)] = result
        node_path = SEGMENTED_ROOT / f"{body_id}_skeleton_nodes_by_geometry.parquet"
        segment_path = SEGMENTED_ROOT / f"{body_id}_skeleton_segments_by_geometry.parquet"
        sample_path = SEGMENTED_ROOT / f"{body_id}_skeleton_sample_segments_by_geometry.parquet"
        result.nodes.to_parquet(node_path, index=False)
        result.segments.to_parquet(segment_path, index=False)
        result.samples.to_parquet(sample_path, index=False)
        skeleton_passed_rois.update(
            str(roi)
            for roi in result.nodes.loc[result.nodes["inside"], "derived_roi"].unique()
            if str(roi) != OUTSIDE_ROI
        )
        skeleton_passed_rois.update(
            str(roi)
            for roi in result.segments.loc[result.segments["inside"], "derived_roi"].unique()
            if str(roi) != OUTSIDE_ROI
        )
        skeleton_records.append(
            {
                "bodyId": int(body_id),
                "nodes": int(len(result.nodes)),
                "segments": int(len(result.segments)),
                "sample_segments": int(len(result.samples)),
                "node_roi_counts": result.nodes["derived_roi"].value_counts().to_dict(),
                "segment_length_inside": result.segments.groupby("derived_roi")["length_inside"].sum().to_dict(),
            }
        )

    source_roi = synapses["roi"].fillna("__api_none__").astype(str)
    derived_roi = segmented_synapses["derived_roi"].astype(str)
    comparison = pd.crosstab(source_roi, derived_roi)
    comparison.to_csv(SEGMENTED_ROOT / "synapse_api_vs_geometry_roi_counts.csv")

    snapped_roi = segmented_synapses["snapped_roi"].astype(str)
    manifest["segmentation"] = {
        "mesh_count": len(meshes),
        "containment_backend": containment_backend(),
        "mesh_names": sorted(meshes),
        "skeleton_passed_rois": sorted(skeleton_passed_rois),
        "synapse_passed_rois": sorted(
            str(roi)
            for roi in segmented_synapses.loc[
                segmented_synapses["inside"], "derived_roi"
            ].unique()
            if str(roi) != OUTSIDE_ROI
        ),
        "synapse_overlap": "first",
        "skeleton_overlap": "first",
        "skeleton_segment_samples": segment_samples,
        "synapse_geometry_roi_counts": segmented_synapses["derived_roi"].value_counts().to_dict(),
        "synapse_geometry_outside_rows": int((derived_roi == OUTSIDE_ROI).sum()),
        "synapse_snap": {
            "enabled": snap_outside,
            "source_label": OUTSIDE_ROI,
            "max_distance": max_snap_distance,
            "snapped_rows": int(segmented_synapses["was_snapped"].sum()),
            "remaining_outside_rows": int((snapped_roi == OUTSIDE_ROI).sum()),
            "snapped_roi_counts": segmented_synapses.loc[
                segmented_synapses["was_snapped"], "snapped_roi"
            ].value_counts().to_dict(),
        },
        "skeletons": skeleton_records,
        "api_vs_geometry_table": str(
            (SEGMENTED_ROOT / "synapse_api_vs_geometry_roi_counts.csv").relative_to(OUTPUT_ROOT)
        ),
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    passed_meshes = {
        roi: mesh for roi, mesh in meshes.items() if roi in skeleton_passed_rois
    }
    return segmented_synapses, meshes, passed_meshes, skeleton_results


ROI_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
    "#b3b3b3", "#1b9e77", "#d95f02", "#7570b3",
]


def _roi_color_map(rois):
    names = sorted({str(roi) for roi in rois if str(roi) != OUTSIDE_ROI})
    colors = {
        roi: ROI_PALETTE[index % len(ROI_PALETTE)]
        for index, roi in enumerate(names)
    }
    colors[OUTSIDE_ROI] = "#444444"
    return colors


def _add_raw_roi_traces(vs, meshes, *, source_space: str, target_space: str, roi_colors):
    import plotly.graph_objects as go
    import navis

    for roi, mesh in sorted(meshes.items()):
        volume = mesh if isinstance(mesh, navis.Volume) else navis.Volume(mesh, name=roi, units="nm")
        if source_space != target_space:
            volume = navis.xform_brain(volume, source=source_space, target=target_space)
        vertices = np.asarray(volume.vertices)
        faces = np.asarray(volume.faces)
        vs.fig_3d.add_trace(
            go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                name=f"ROI mesh [{roi}]",
                legendgroup=f"roi:{roi}",
                showlegend=True,
                color=roi_colors.get(str(roi), "#777777"),
                opacity=0.10,
                hovertemplate=f"<b>ROI mesh [{roi}]</b><extra></extra>",
            )
        )


def _add_segmented_skeleton_traces(
    vs,
    skeleton_results,
    *,
    source_space: str,
    target_space: str,
    roi_colors,
):
    """Overlay raw skeleton edges colored by their geometry-derived ROI."""
    import plotly.graph_objects as go
    import navis

    shown_rois = set()
    for body_id, result in sorted(skeleton_results.items()):
        segments = result.samples
        if segments.empty:
            continue
        for roi, group in segments.groupby("derived_roi", sort=True):
            roi = str(roi)
            parent = group[["start_x", "start_y", "start_z"]].rename(
                columns={"start_x": "x", "start_y": "y", "start_z": "z"}
            )
            child = group[["end_x", "end_y", "end_z"]].rename(
                columns={"end_x": "x", "end_y": "y", "end_z": "z"}
            )
            endpoints = pd.concat([parent, child], ignore_index=True)
            if source_space != target_space:
                endpoints = navis.xform_brain(
                    endpoints,
                    source=source_space,
                    target=target_space,
                )

            x_values = []
            y_values = []
            z_values = []
            for index in range(0, len(endpoints), 2):
                x_values.extend([endpoints.iloc[index]["x"], endpoints.iloc[index + 1]["x"], None])
                y_values.extend([endpoints.iloc[index]["y"], endpoints.iloc[index + 1]["y"], None])
                z_values.extend([endpoints.iloc[index]["z"], endpoints.iloc[index + 1]["z"], None])

            vs.fig_3d.add_trace(
                go.Scatter3d(
                    x=x_values,
                    y=y_values,
                    z=z_values,
                    mode="lines",
                    name=f"skeleton ROI [{roi}]",
                    legendgroup=f"skeleton-roi:{roi}",
                    showlegend=roi not in shown_rois,
                    line=dict(color=roi_colors.get(roi, "#777777"), width=4.5),
                    hovertemplate=(
                        f"<b>skeleton ROI [{roi}]</b><br>bodyId={body_id}"
                        "<extra></extra>"
                    ),
                )
            )
            shown_rois.add(roi)


def _add_segmented_synapse_traces(
    vs,
    segmented_synapses,
    *,
    source_space: str,
    target_space: str,
    roi_column: str,
    roi_colors,
    highlight_snapped: bool = False,
):
    import plotly.graph_objects as go
    import navis

    if roi_column not in segmented_synapses.columns:
        raise ValueError(f"Synapse table does not contain ROI column {roi_column!r}")
    group_columns = [roi_column, "type"]
    if highlight_snapped and "was_snapped" in segmented_synapses.columns:
        group_columns.append("was_snapped")

    for group_key, group in segmented_synapses.groupby(group_columns, sort=True):
        if len(group_columns) == 2:
            roi, synapse_type = group_key
            was_snapped = False
        else:
            roi, synapse_type, was_snapped = group_key
        roi = str(roi)
        coords = group[["x", "y", "z"]].copy()
        if source_space != target_space:
            coords = navis.xform_brain(coords, source=source_space, target=target_space)
        base_symbol = "circle" if str(synapse_type) == "pre" else "diamond"
        symbol = f"{base_symbol}-open" if bool(was_snapped) else base_symbol
        customdata = None
        snap_hover = ""
        if bool(was_snapped) and "nearest_roi_distance" in group.columns:
            customdata = group[["nearest_roi_distance"]].to_numpy()
            snap_hover = "<br>snap distance=%{customdata[0]:.1f}"
        label = "snapped synapses" if bool(was_snapped) else "synapses"
        vs.fig_3d.add_trace(
            go.Scatter3d(
                x=coords["x"], y=coords["y"], z=coords["z"],
                mode="markers",
                name=f"{label} [{roi}] ({synapse_type})",
                legendgroup=f"synapses:{roi}",
                marker=dict(
                    size=4.0 if bool(was_snapped) else 3.5,
                    color=roi_colors.get(roi, "#444444"),
                    symbol=symbol,
                    opacity=0.90,
                    line=(
                        dict(color=roi_colors.get(roi, "#444444"), width=1.2)
                        if bool(was_snapped) else None
                    ),
                ),
                customdata=customdata,
                hovertemplate=(
                    f"<b>{roi}</b><br>type={synapse_type}<br>"
                    f"{snap_hover}x=%{{x:.0f}}<br>y=%{{y:.0f}}<br>z=%{{z:.0f}}<extra></extra>"
                ),
            )
        )


def _visualize(
    client,
    metadata,
    segmented_synapses,
    meshes,
    skeleton_results,
    *,
    saveas: str,
    synapse_roi_column: str,
    title: str,
    highlight_snapped: bool = False,
):
    from visualize_skeleton import VisualizeSkeleton

    body_ids = metadata["bodyId"].astype("int64").tolist()
    vs = VisualizeSkeleton(
        dataset=DATASET,
        client=client,
        neuron_layers=[body_ids],
        skeleton_mode="line",
        skeleton_mesh_simplification=0.0,
        cache_neurons=False,
        cache_synapses=False,
        skip_synapse=True,
        show_connectors=False,
        mesh_roi=[],
        brain_mesh="template",
        brain_mesh_color="rgba(170, 205, 220, 0.08)",
        show_fig=False,
        export_views=False,
        include_timestamp=False,
        output_dir=str(VIS_ROOT),
        saveas=saveas,
        verbose="simple",
    )

    # This invokes the production visualizer for raw skeletons and its native
    # male-CNS template mesh.  ROI traces below are added from the serialized
    # API meshes so local/simplified ROI caches cannot be selected accidentally.
    vs.plot_skeleton()
    vs.plot_mesh()
    skeleton_rois = {
        str(roi)
        for result in skeleton_results.values()
        for roi in result.segments["derived_roi"].unique()
    }
    roi_colors = _roi_color_map(
        set(meshes) | set(segmented_synapses[synapse_roi_column].unique()) | skeleton_rois
    )
    _add_raw_roi_traces(
        vs,
        meshes,
        source_space="JRCFIB2022Mraw",
        target_space="JRCFIB2022M",
        roi_colors=roi_colors,
    )
    _add_segmented_skeleton_traces(
        vs,
        skeleton_results,
        source_space="JRCFIB2022Mraw",
        target_space="JRCFIB2022M",
        roi_colors=roi_colors,
    )
    _add_segmented_synapse_traces(
        vs,
        segmented_synapses,
        source_space="JRCFIB2022Mraw",
        target_space="JRCFIB2022M",
        roi_column=synapse_roi_column,
        roi_colors=roi_colors,
        highlight_snapped=highlight_snapped,
    )
    vs.fig_3d.update_layout(
        title=title,
        legend=dict(itemsizing="constant"),
    )
    vs.save_figure()
    return Path(vs.save_folder)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse existing raw artifacts instead of refetching them",
    )
    parser.add_argument(
        "--segment-samples",
        type=int,
        default=3,
        help="Midpoint samples per raw skeleton edge for the live check (default: 3)",
    )
    parser.add_argument(
        "--snap-outside",
        action="store_true",
        help="Assign outside synapses to their nearest ROI surface and create a snapped figure",
    )
    parser.add_argument(
        "--max-snap-distance",
        type=float,
        default=None,
        help="Optional maximum snap distance in raw mesh coordinate units",
    )
    args = parser.parse_args(argv)
    _ensure_dirs()
    client = _load_client()
    metadata, synapses, manifest = _fetch_raw(client, reuse=args.reuse)
    segmented_synapses, meshes, passed_meshes, skeleton_results = _segment_saved_inputs(
        metadata,
        synapses,
        manifest,
        segment_samples=args.segment_samples,
        snap_outside=args.snap_outside,
        max_snap_distance=args.max_snap_distance,
    )
    # Plot the ROIs the raw aMe12 skeleton actually traverses.  If the
    # sampled centerline does not intersect a successful mesh, retain all
    # successful meshes in the figure so the run still exposes the geometry.
    meshes_to_plot = passed_meshes or meshes
    figure_folder = _visualize(
        client,
        metadata,
        segmented_synapses,
        meshes_to_plot,
        skeleton_results,
        saveas="aMe12_ROI_sensitive_geometry",
        synapse_roi_column="derived_roi",
        title=(
            "male-cns:v1.0 aMe12: raw skeletons colored by geometry-derived ROI, "
            "directly segmented synapses, API ROI meshes"
        ),
    )
    snapped_figure_folder = None
    snapped_meshes = None
    if args.snap_outside:
        snapped_roi_names = set(
            segmented_synapses.loc[segmented_synapses["was_snapped"], "snapped_roi"].astype(str)
        )
        snapped_meshes = dict(meshes_to_plot)
        snapped_meshes.update({
            roi: meshes[roi]
            for roi in snapped_roi_names
            if roi in meshes
        })
        snapped_figure_folder = _visualize(
            client,
            metadata,
            segmented_synapses,
            snapped_meshes,
            skeleton_results,
            saveas="aMe12_ROI_sensitive_geometry_snapped",
            synapse_roi_column="snapped_roi",
            title=(
                "male-cns:v1.0 aMe12: skeleton ROI colors, nearest-ROI-snapped synapses, "
                "API ROI meshes"
            ),
            highlight_snapped=True,
        )
    print(json.dumps({
        "output_root": str(OUTPUT_ROOT),
        "body_ids": metadata["bodyId"].astype("int64").tolist(),
        "synapses": int(len(synapses)),
        "successful_roi_meshes": sorted(meshes),
        "skeleton_passed_rois": sorted(passed_meshes),
        "visualized_roi_meshes": sorted(meshes_to_plot),
        "snap_outside": args.snap_outside,
        "snapped_synapses": int(segmented_synapses["was_snapped"].sum()),
        "remaining_outside_after_snap": int(
            (segmented_synapses["snapped_roi"] == OUTSIDE_ROI).sum()
        ),
        "geometry_roi_counts": segmented_synapses["derived_roi"].value_counts().to_dict(),
        "snapped_roi_counts": segmented_synapses.loc[
            segmented_synapses["was_snapped"], "snapped_roi"
        ].value_counts().to_dict(),
        "figure_folder": str(figure_folder),
        "snapped_figure_folder": str(snapped_figure_folder) if snapped_figure_folder else None,
        "snapped_visualized_roi_meshes": sorted(snapped_meshes) if snapped_meshes else [],
    }, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
