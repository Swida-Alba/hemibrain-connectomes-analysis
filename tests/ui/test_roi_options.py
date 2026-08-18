"""Tests for dataset-backed skeleton ROI option lists."""

import json
from pathlib import Path

from ui.roi_options import load_roi_catalog, roi_options_for_mode


def _write_metadata(root: Path, dataset: str, metadata: dict) -> Path:
    folder = dataset.replace(":", "_").replace(".", "_")
    directory = root / "datasets" / folder
    directory.mkdir(parents=True)
    path = directory / f"{folder}_metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return directory


def test_primary_options_use_metadata_and_can_expand_bilateral_names(tmp_path):
    dataset = "test:v1.0"
    _write_metadata(
        tmp_path,
        dataset,
        {
            "primaryRois": ["A(L)", "A(R)", "M", "NotPrimary"],
        },
    )

    assert roi_options_for_mode(
        dataset, project_root=tmp_path
    ) == ["A", "M"]
    assert roi_options_for_mode(
        dataset, include_lr=True, project_root=tmp_path
    ) == ["A(L)", "A(R)", "M"]


def test_primary_subprimary_and_all_modes_use_available_inventory(tmp_path):
    dataset = "test:v1.0"
    _write_metadata(
        tmp_path,
        dataset,
        {"roi_coverage": {"roi_list": ["A(L)", "A(R)", "M"]}},
    )
    cache_dir = tmp_path / "cache" / "test_v1_0"
    cache_dir.mkdir(parents=True)
    (cache_dir / "available_rois.json").write_text(
        json.dumps(["A(L)", "A(R)", "A-sub(L)", "container", "M"]),
        encoding="utf-8",
    )

    assert roi_options_for_mode(
        dataset,
        include_subprimary=True,
        project_root=tmp_path,
    ) == ["A", "M", "A-sub", "container"]
    assert roi_options_for_mode(
        dataset, "all available", project_root=tmp_path
    ) == ["A(L)", "A(R)", "A-sub(L)", "container", "M"]


def test_roi_count_table_fills_all_mode_when_cache_is_missing(tmp_path):
    import polars as pl

    dataset = "test:v1.0"
    directory = _write_metadata(
        tmp_path,
        dataset,
        {"roi_coverage": {"roi_list": ["A(L)", "A(R)"]}},
    )
    pl.DataFrame({"roi": ["A(L)", "A(R)", "A-sub(L)", "NotPrimary"]}).write_parquet(
        directory / "test_v1_0_allneurons_roi_count_df.parquet"
    )

    catalog = load_roi_catalog(dataset, project_root=tmp_path)
    assert catalog["primary"] == ["A(L)", "A(R)"]
    assert catalog["available"] == ["A(L)", "A(R)", "A-sub(L)"]


def test_skeleton_tab_starts_with_no_meshes_and_exposes_roi_controls():
    from nicegui import Client
    from nicegui.page import page
    from ui.tabs.visualization import create_skeleton_tab

    client = Client(page("/skeleton-roi-controls"))
    with client:
        create_skeleton_tab()

    meshes = next(
        element for element in client.elements.values()
        if getattr(element, "_props", {}).get("label") == "Mesh ROIs"
    )
    labels = [
        getattr(element, "text", "")
        for element in client.elements.values()
        if getattr(element, "text", "")
    ]

    assert "ROI Selection Mode" not in labels
    assert "Include L/R variants" in labels
    assert "Include sub-primary ROIs" in labels
    assert meshes.value == []
    assert "outlined" in meshes._props
