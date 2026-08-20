"""Guards for FlyWire skeleton-backed workflows."""

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.flywire_readiness import (  # noqa: E402
    FlyWireSkeletonAccessError,
    flywire_skeleton_readiness,
    require_flywire_skeleton_access,
)
from morphology import MorphologyComparer  # noqa: E402
from visualize_skeleton import VisualizeSkeleton  # noqa: E402


def test_banc_is_always_blocked_and_does_not_expose_token(tmp_path):
    log = []

    with pytest.raises(FlyWireSkeletonAccessError, match="BANC skeletons"):
        require_flywire_skeleton_access(
            "flywire_BANC_v626", project_root=tmp_path, log=log.append
        )

    text = "\n".join(log)
    assert "BLOCKED" in text
    assert "CAVE token does not enable BANC" in text
    assert "BANC remains available" in text


def test_fafb_without_local_source_or_cave_token_is_blocked_with_instructions(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("CAVE_TOKEN", raising=False)
    log = []

    with pytest.raises(FlyWireSkeletonAccessError, match="no local FAFB"):
        require_flywire_skeleton_access(
            "flywire_FAFB_v783", project_root=tmp_path, log=log.append
        )

    text = "\n".join(log)
    assert "sk_lod1_783_healed.zip" in text
    assert "python src/FAFB_file_converter.py" in text
    assert "CAVE_TOKEN" in text


def test_fafb_local_source_is_ready_without_cave_token(tmp_path, monkeypatch):
    monkeypatch.delenv("CAVE_TOKEN", raising=False)
    skeleton_zip = (
        tmp_path / "datasets" / "flywire_FAFB_v783" / "sk_lod1_783_healed.zip"
    )
    skeleton_zip.parent.mkdir(parents=True)
    skeleton_zip.write_bytes(b"placeholder")

    status = flywire_skeleton_readiness(
        "flywire_FAFB_v783", project_root=tmp_path
    )
    assert status["local_skeletons"] is True
    assert status["cave_token"] is False
    assert status["ready"] is True

    log = []
    require_flywire_skeleton_access(
        "flywire_FAFB_v783", project_root=tmp_path, log=log.append
    )
    assert "CAVE fallback is disabled" in "\n".join(log)


def test_fafb_cave_token_from_config_json_enables_fallback(tmp_path, monkeypatch):
    """A CAVE token in the project config.json counts as configured."""
    monkeypatch.delenv("CAVE_TOKEN", raising=False)
    (tmp_path / "config.json").write_text(
        '{"tokens": {"cave": "cfg-cave-token"}}\n', encoding="utf-8"
    )

    status = flywire_skeleton_readiness(
        "flywire_FAFB_v783", project_root=tmp_path
    )
    assert status["cave_token"] is True
    assert status["ready"] is True

    log = []
    require_flywire_skeleton_access(
        "flywire_FAFB_v783", project_root=tmp_path, log=log.append
    )
    assert "CAVE_TOKEN is configured" in "\n".join(log)
    assert "cfg-cave-token" not in "\n".join(log)


def test_fafb_local_source_keeps_cave_fallback_enabled_with_token(tmp_path, monkeypatch):
    monkeypatch.setenv("CAVE_TOKEN", "configured-token")
    skeleton_zip = (
        tmp_path / "datasets" / "flywire_FAFB_v783" / "sk_lod1_783_healed.zip"
    )
    skeleton_zip.parent.mkdir(parents=True)
    skeleton_zip.write_bytes(b"placeholder")

    log = []
    require_flywire_skeleton_access(
        "flywire_FAFB_v783", project_root=tmp_path, log=log.append
    )
    text = "\n".join(log)
    assert "CAVE API fallback is configured" in text
    assert "will be attempted" in text


def test_fafb_cave_token_is_an_online_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("CAVE_TOKEN", "secret-token-value")
    status = flywire_skeleton_readiness(
        "flywire_FAFB_v783", project_root=tmp_path
    )
    assert status["local_skeletons"] is False
    assert status["cave_token"] is True
    assert status["ready"] is True

    log = []
    require_flywire_skeleton_access(
        "flywire_FAFB_v783", project_root=tmp_path, log=log.append
    )
    text = "\n".join(log)
    assert "CAVE_TOKEN is configured" in text
    assert "secret-token-value" not in text


def test_morphology_query_guard_runs_at_script_entry(tmp_path, capsys):
    comparer = MorphologyComparer(
        query=1,
        dataset="flywire_BANC_v626",
        project_root=str(tmp_path),
        verbose=True,
    )

    with pytest.raises(FlyWireSkeletonAccessError):
        comparer.find_similar()

    assert "[DROCAT][dataset-guard] BLOCKED" in capsys.readouterr().out


def test_skeleton_visualizer_guard_runs_before_banc_preparation(tmp_path, capsys):
    with pytest.raises(FlyWireSkeletonAccessError):
        VisualizeSkeleton(
            dataset="flywire_BANC_v626",
            neuron_layers=["1"],
            script_path=str(tmp_path),
            verbose=True,
        )

    text = capsys.readouterr().out
    assert "BANC skeletons are not available" in text


def test_fafb_plot_guard_runs_before_skeleton_query(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CAVE_TOKEN", raising=False)
    import FAFB_file_converter

    monkeypatch.setattr(
        FAFB_file_converter,
        "ensure_flywire_data",
        lambda dataset, dataset_dir: True,
    )
    visualizer = VisualizeSkeleton(
        dataset="flywire_FAFB_v783",
        neuron_layers=["1"],
        script_path=str(tmp_path),
        verbose=True,
    )

    with pytest.raises(FlyWireSkeletonAccessError, match="no local FAFB"):
        visualizer.plot_skeleton()

    text = capsys.readouterr().out
    assert "sk_lod1_783_healed.zip" in text
    assert "CAVE_TOKEN" in text
