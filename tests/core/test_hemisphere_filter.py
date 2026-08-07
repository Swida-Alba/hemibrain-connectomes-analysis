"""Hemisphere filtering for pathfinding (FindNeuronConnection).

Covers the `hemisphere_filter` parameter ('both' / 'left' / 'right'):
- validation and alias normalization
- neuron-level filtering: 'left' keeps L + U, 'right' keeps R + U, 'both'
  keeps everything; unclassified ('U', no explicit hemisphere notation)
  neurons are ALWAYS included
- connection-level filtering: an edge is kept only when BOTH endpoints
  belong to the selected hemisphere (unclassified endpoints always kept)
- hemisphere suffixes (_L/_R/_U) still applied when separate_hemispheres
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coana import FindNeuronConnection  # noqa: E402


def make_fc(separate=False, hemi_filter="both"):
    """Network-free instance for the pure DataFrame helpers."""
    fc = object.__new__(FindNeuronConnection)
    fc.separate_hemispheres = separate
    fc.hemisphere_filter = hemi_filter
    return fc


def make_neuron_df():
    return pd.DataFrame({
        "bodyId": [1, 2, 3, 4, 5],
        "type": ["A", "B", "C", "D", "E"],
        "instance": ["A_R", "B_L", "C", "D_R", "E"],
    })


def make_conn_df():
    return pd.DataFrame({
        "bodyId_pre": [1, 1, 2, 2, 3],
        "bodyId_post": [10, 11, 12, 13, 14],
        "type_pre": ["A", "A", "B", "B", "C"],
        "type_post": ["X", "Y", "Z", "W", "V"],
        "instance_pre": ["A_R", "A_R", "B_L", "B_L", "C"],
        "instance_post": ["X_L", "Y_R", "Z", "W_R", "V"],
        "weight": [1, 2, 3, 4, 5],
    })


# ---------------------------------------------------------------------------
# Parameter validation / normalization
# ---------------------------------------------------------------------------

class _FakeNeuprintClient:
    def __init__(self, *args, **kwargs):
        pass


def test_hemisphere_filter_aliases_normalized(monkeypatch, tmp_path):
    import neuprint
    monkeypatch.setattr(neuprint, "Client", _FakeNeuprintClient)

    fc = FindNeuronConnection(
        dataset="fake:v1", sourceNeurons=[], targetNeurons=[],
        use_cache=True, cache_only=False, verbose=False,
        script_path=str(tmp_path), cache_folder=str(tmp_path / "cache"),
        hemisphere_filter="l",
    )
    assert fc.hemisphere_filter == "left"
    fc2 = FindNeuronConnection(
        dataset="fake:v1", sourceNeurons=[], targetNeurons=[],
        use_cache=True, cache_only=False, verbose=False,
        script_path=str(tmp_path), cache_folder=str(tmp_path / "cache"),
        hemisphere_filter="RIGHT HEMISPHERE",
    )
    assert fc2.hemisphere_filter == "right"


def test_hemisphere_filter_invalid_rejected(monkeypatch, tmp_path):
    import neuprint
    monkeypatch.setattr(neuprint, "Client", _FakeNeuprintClient)

    with pytest.raises(ValueError):
        FindNeuronConnection(
            dataset="fake:v1", sourceNeurons=[], targetNeurons=[],
            use_cache=True, cache_only=False, verbose=False,
            script_path=str(tmp_path), cache_folder=str(tmp_path / "cache"),
            hemisphere_filter="nope",
        )


# ---------------------------------------------------------------------------
# Neuron-level filtering (_apply_hemisphere_suffix_to_neuron_df)
# ---------------------------------------------------------------------------

def test_left_filter_keeps_left_and_unclassified_neurons():
    fc = make_fc(separate=True, hemi_filter="left")
    out = fc._apply_hemisphere_suffix_to_neuron_df(make_neuron_df())
    # A_R and D_R are right-hemisphere -> dropped; B_L, C (U), E (U) kept.
    assert sorted(out["bodyId"]) == [2, 3, 5]
    by_id = out.set_index("bodyId")
    assert by_id.loc[2, "type"] == "B_L"
    assert by_id.loc[3, "type"] == "C_U"
    assert by_id.loc[5, "type"] == "E_U"


def test_right_filter_keeps_right_and_unclassified_neurons():
    fc = make_fc(separate=True, hemi_filter="right")
    out = fc._apply_hemisphere_suffix_to_neuron_df(make_neuron_df())
    assert sorted(out["bodyId"]) == [1, 3, 4, 5]
    by_id = out.set_index("bodyId")
    assert by_id.loc[1, "type"] == "A_R"
    assert by_id.loc[4, "type"] == "D_R"


def test_both_filter_keeps_everything_with_suffixes():
    fc = make_fc(separate=True, hemi_filter="both")
    out = fc._apply_hemisphere_suffix_to_neuron_df(make_neuron_df())
    assert len(out) == 5
    assert out["type"].tolist() == ["A_R", "B_L", "C_U", "D_R", "E_U"]


def test_soma_side_camelcase_column_detected_and_midline_included():
    """male-cns tables name the column 'somaSide' (camelCase) and use 'M'
    for midline neurons - midline must be treated as unclassified and kept
    in every option (like 'U')."""
    fc = make_fc(separate=True, hemi_filter="left")
    df = pd.DataFrame({
        "bodyId": [1, 2, 3],
        "type": ["A", "B", "C"],
        "instance": ["", "", ""],
        "somaSide": ["R", "L", "M"],
    })
    out = fc._apply_hemisphere_suffix_to_neuron_df(df)
    # R dropped; L and M (midline -> U) kept.
    assert sorted(out["bodyId"]) == [2, 3]
    by_id = out.set_index("bodyId")
    assert by_id.loc[2, "type"] == "B_L"
    assert by_id.loc[3, "type"] == "C_U"


def test_find_hemisphere_column_variants():
    fc = make_fc()
    df = pd.DataFrame({
        "bodyId": [1],
        "Soma side": ["L"],
    })
    assert fc._find_hemisphere_column(df) == "Soma side"
    df2 = pd.DataFrame({"bodyId": [1], "somaSide": ["L"]})
    assert fc._find_hemisphere_column(df2) == "somaSide"
    df3 = pd.DataFrame({"bodyId": [1], "rootSide": ["L"]})
    assert fc._find_hemisphere_column(df3) == "rootSide"
    df4 = pd.DataFrame({"bodyId": [1]})
    assert fc._find_hemisphere_column(df4) is None


def test_left_filter_works_without_separate_suffixes():
    fc = make_fc(separate=False, hemi_filter="left")
    out = fc._apply_hemisphere_suffix_to_neuron_df(make_neuron_df())
    assert sorted(out["bodyId"]) == [2, 3, 5]
    assert out.set_index("bodyId").loc[2, "type"] == "B"  # no suffix


# ---------------------------------------------------------------------------
# Connection-level filtering (_apply_hemisphere_suffix_to_conn_df)
# ---------------------------------------------------------------------------

def test_conn_left_filter_keeps_edges_with_both_endpoints_left_or_unknown():
    fc = make_fc(separate=True, hemi_filter="left")
    out = fc._apply_hemisphere_suffix_to_conn_df(make_conn_df())
    # 1(R)->10(L) and 1(R)->11(R) dropped (pre is right); 2(L)->13(R) dropped
    # (post is right); 2(L)->12(U) and 3(U)->14(U) kept.
    assert sorted(out["bodyId_pre"]) == [2, 3]
    by_pre = out.set_index("bodyId_pre")
    assert by_pre.loc[2, "type_pre"] == "B_L"
    assert by_pre.loc[2, "type_post"] == "Z_U"
    assert by_pre.loc[3, "type_pre"] == "C_U"
    assert by_pre.loc[3, "type_post"] == "V_U"


def test_conn_right_filter_keeps_edges_with_both_endpoints_right_or_unknown():
    fc = make_fc(separate=True, hemi_filter="right")
    out = fc._apply_hemisphere_suffix_to_conn_df(make_conn_df())
    assert sorted(out["bodyId_pre"]) == [1, 3]
    by_pre = out.set_index("bodyId_pre")
    assert by_pre.loc[1, "type_pre"] == "A_R"
    assert by_pre.loc[1, "type_post"] == "Y_R"
    assert by_pre.loc[3, "type_pre"] == "C_U"


def test_conn_filter_works_without_separate_suffixes():
    fc = make_fc(separate=False, hemi_filter="right")
    out = fc._apply_hemisphere_suffix_to_conn_df(make_conn_df())
    assert sorted(out["bodyId_pre"]) == [1, 3]
    assert "type_pre" not in out.columns or out.set_index("bodyId_pre").loc[1, "type_pre"] == "A"


def test_conn_no_filter_keeps_all_rows():
    fc = make_fc(separate=True, hemi_filter="both")
    out = fc._apply_hemisphere_suffix_to_conn_df(make_conn_df())
    assert len(out) == 5
