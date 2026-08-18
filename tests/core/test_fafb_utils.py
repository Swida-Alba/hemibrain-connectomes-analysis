"""Regression tests for shared FAFB morphology helpers."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_flag_extrusions_accepts_string_body_id_keys(tmp_path, monkeypatch):
    import fafb_utils

    monkeypatch.setattr(
        fafb_utils,
        "detect_extrusion",
        lambda neuron, simplification=0.95: neuron == "flag",
    )

    flagged = fafb_utils.flag_extrusions(
        str(tmp_path),
        "flywire_FAFB_v783",
        {"720575940640836952": "ok", 720575940625459464: "flag"},
        n_workers=1,
    )

    assert flagged == [720575940625459464]


def _branched_extrusion_neuron():
    import navis
    import pandas as pd

    # The branch at node 3 begins with a 99-unit jump while the normal tree
    # edges are one unit long.
    return navis.TreeNeuron(pd.DataFrame({
        "node_id": [0, 1, 2, 3, 4],
        "parent_id": [-1, 0, 1, 1, 3],
        "x": [0.0, 1.0, 2.0, 100.0, 101.0],
        "y": [0.0] * 5,
        "z": [0.0] * 5,
        "radius": [1.0] * 5,
    }))


def test_repair_extruded_skeleton_prunes_only_bad_subtree():
    from fafb_utils import diagnose_extrusion_nodes, repair_extruded_skeleton

    neuron = _branched_extrusion_neuron()
    diagnosis = diagnose_extrusion_nodes(neuron)
    assert diagnosis["candidate_child_ids"] == [3]
    assert diagnosis["candidate_parent_ids"] == [1]

    repaired, stats = repair_extruded_skeleton(neuron)

    assert stats["repaired"] is True
    assert stats["removed_node_ids"] == [3, 4]
    assert set(repaired.nodes["node_id"]) == {0, 1, 2}
    assert set(neuron.nodes["node_id"]) == {0, 1, 2, 3, 4}


def test_extrusion_cache_merges_rows_and_keeps_flagged_retryable(
        tmp_path, monkeypatch):
    """Detection is cached once, while a flagged row remains available for
    the API/local repair retry path on later runs."""
    import fafb_utils

    root = str(tmp_path)
    folder = "flywire_FAFB_v783"
    fafb_utils.save_extrusion_check_cache(root, folder, {101: True})
    fafb_utils.save_extrusion_check_cache(root, folder, {202: False})

    assert fafb_utils.load_extrusion_check_cache(root, folder) == {
        "101": True,
        "202": False,
    }
    statuses = fafb_utils.load_extrusion_repair_status(root, folder)
    assert statuses["101"] == fafb_utils.EXTRUSION_REPAIR_PENDING
    assert statuses["202"] == fafb_utils.EXTRUSION_REPAIR_CLEAN

    fafb_utils.set_extrusion_repair_status(
        root, folder, {"101": fafb_utils.EXTRUSION_REPAIR_LOCAL_FALLBACK}
    )
    assert fafb_utils.load_extrusion_repair_status(root, folder)["101"] \
        == fafb_utils.EXTRUSION_REPAIR_LOCAL_FALLBACK

    monkeypatch.setattr(
        fafb_utils, "detect_extrusion",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("cached neuron was checked again")),
    )
    assert fafb_utils.flag_extrusions(
        root, folder, {101: object()}, n_workers=0
    ) == [101]
