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
