"""Search-column scoping for neuron name resolution.

Covers statvis._process_single_neuron's `search_columns` parameter:
- 'auto' must match EXACT names in cross-dataset type columns such as
  flywireType (previously only regex patterns reached those columns, so
  source='MTe07' on male-cns v1.0 silently found nothing while 'MTe.*'
  worked)
- 'type' / 'instance' / 'bodyId' must restrict the search to one column
- the bodyId -> type -> instance -> others priority must be preserved
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from statvis import _process_single_neuron  # noqa: E402


def make_df():
    """Miniature neuron table mirroring male-cns v1.0: CNS types in `type`,
    FlyWire names only in `flywireType` (the reported real-world case)."""
    return pd.DataFrame({
        "bodyId": [1, 2, 3, 4, 5],
        "type": ["DNp01", "MBON01", "DNp02", "MBON02", None],
        "instance": ["DNp01_R", "MBON01_R", "DNp02_R", "MBON02_R", "X07_R"],
        "flywireType": ["MTe07", "MTe12", "MTe44", "MTe27", "MTe07"],
        "hemibrainType": ["", "", "HSN", "", ""],
    })


DF = make_df()
BIDS = DF["bodyId"].tolist()


def resolve(term, scope="auto"):
    """Run _process_single_neuron and return (bodyIds, search_info)."""
    return _process_single_neuron(term, DF, BIDS, verbose=False, search_columns=scope)


def test_exact_match_finds_flywireType_in_auto():
    """Exact 'MTe07' must be found in the flywireType column under 'auto'
    (this was the reported bug: only regex reached other columns)."""
    body_ids, info = resolve("MTe07")
    assert sorted(body_ids) == [1, 5]
    assert info["matched_column"] == "flywireType"
    assert info["match_count"] == 2


def test_regex_match_finds_flywireType_in_auto():
    """'MTe.*' (the previously working form) still matches flywireType."""
    body_ids, info = resolve("MTe.*")
    assert sorted(body_ids) == [1, 2, 3, 4, 5]
    assert info["matched_column"] == "flywireType"


def test_plain_queries_are_exact_and_explicit_prefixes_expand_the_family():
    """A bare pathfinding token is exact; prefix mode is explicit."""
    frame = pd.DataFrame({
        "bodyId": [11, 12, 13, 14],
        "type": ["MeVPaMe1", "MeVPaMe2", "Other", "Other"],
        "instance": ["MeVPaMe1_L", "MeVPaMe2_R", "Other_L", "Other_R"],
        "flywireType": ["", "", "MeVPaFly", "MeVPaTax"],
    })

    body_ids, info = _process_single_neuron(
        "MeVPa", frame, frame["bodyId"].tolist(), verbose=False
    )
    assert body_ids == []
    assert info["matched_column"] is None

    body_ids, info = _process_single_neuron(
        "MeVPa.*", frame, frame["bodyId"].tolist(), verbose=False
    )
    # The type column owns this prefix. Names in later columns are secondary
    # viewer evidence, not additional pathfinding identities.
    assert body_ids == [11, 12]
    assert info["matched_column"] == "type"

    # An explicitly prefixed query remains case-sensitive.
    body_ids, info = _process_single_neuron(
        "evpa.*", frame, frame["bodyId"].tolist(), verbose=False
    )
    assert body_ids == []
    assert info["matched_column"] is None


def test_startswith_union_does_not_search_arbitrary_metadata():
    """Auto name resolution uses only the useful type/taxonomy projection."""
    noisy = DF.copy()
    noisy["notes"] = ["MTe-noise", "", "", "", ""]
    body_ids, info = _process_single_neuron(
        "MTe-noise", noisy, noisy["bodyId"].tolist(), verbose=False
    )
    assert body_ids == []
    assert info["matched_column"] is None


def test_numeric_queries_are_guarded_to_bodyid_including_regex_patterns():
    """A numeric bodyId miss must not resolve a type with the same digits."""
    numeric = pd.DataFrame({
        "bodyId": [123, 124, 999],
        "type": ["123Type", "Other", "9999Type"],
        "instance": ["123_R", "Other_R", "9999_R"],
        "flywireType": ["FW123", "FW124", "FW999"],
    })
    bids = numeric["bodyId"].tolist()

    body_ids, info = _process_single_neuron(
        "123.*", numeric, bids, verbose=False, search_columns="auto"
    )
    assert body_ids == [123]
    assert info["matched_column"] == "bodyId"

    body_ids, info = _process_single_neuron(
        "9999", numeric, bids, verbose=False, search_columns="auto"
    )
    assert body_ids == []
    assert info["matched_column"] is None


def test_exact_match_still_prefers_type_in_auto():
    """A name present in `type` must resolve there, not in other columns."""
    body_ids, info = resolve("MBON01")
    assert body_ids == [2]
    assert info["matched_column"] == "type"


def test_exact_match_prefers_instance_over_other_columns():
    body_ids, info = resolve("DNp01_R")
    assert body_ids == [1]
    assert info["matched_column"] == "instance"


def test_numeric_bodyId_still_exact_in_auto():
    body_ids, info = resolve(3)
    assert body_ids == [3]
    assert info["matched_column"] == "bodyId"


def test_type_scope_restricts_to_type_column():
    body_ids, info = resolve("DNp01", scope="type")
    assert body_ids == [1]
    assert info["matched_column"] == "type"
    # MTe07 lives only in flywireType -> must NOT be found with scope='type'
    body_ids, info = resolve("MTe07", scope="type")
    assert body_ids == []
    assert info["matched_column"] is None


def test_instance_scope_restricts_to_instance_column():
    body_ids, info = resolve("MBON01_R", scope="instance")
    assert body_ids == [2]
    assert info["matched_column"] == "instance"
    body_ids, _ = resolve("MTe07", scope="instance")
    assert body_ids == []


def test_bodyid_scope_restricts_to_bodyId_column():
    body_ids, info = resolve(4, scope="bodyId")
    assert body_ids == [4]
    assert info["matched_column"] == "bodyId"
    body_ids, _ = resolve("MTe07", scope="bodyId")
    assert body_ids == []


def test_regex_respects_scope():
    body_ids, _ = resolve("MTe.*", scope="type")
    assert body_ids == []
    body_ids, _ = resolve("MTe.*", scope="flywireType_invalid")  # falls back to auto
    assert sorted(body_ids) == [1, 2, 3, 4, 5]


def test_invalid_scope_falls_back_to_auto():
    body_ids, info = resolve("MTe07", scope="bogus")
    assert sorted(body_ids) == [1, 5]
    assert info["matched_column"] == "flywireType"


def test_not_found_returns_empty():
    body_ids, info = resolve("ZZZ", scope="auto")
    assert body_ids == []
    assert info["matched_column"] is None
    assert info["match_count"] == 0


def test_input_mode_reaches_backend_as_exact_or_explicit_prefix():
    """The UI mode conversion must not silently broaden an Exact query."""
    from ui.components.common import apply_filter_mode

    exact_query = apply_filter_mode(["MeVPa"], "exact")
    prefix_query = apply_filter_mode(["MeVPa"], "startswith")

    exact_ids = []
    for token in exact_query:
        ids, _ = _process_single_neuron(
            token, DF.assign(type=["MeVPaMe1", "MeVPaMe2", "Other", "Other", None]),
            BIDS, verbose=False,
        )
        exact_ids.extend(ids)
    prefix_ids = []
    prefix_frame = DF.assign(
        type=["MeVPaMe1", "MeVPaMe2", "Other", "Other", None]
    )
    for token in prefix_query:
        ids, _ = _process_single_neuron(
            token, prefix_frame, BIDS, verbose=False,
        )
        prefix_ids.extend(ids)

    assert exact_query == ["MeVPa"]
    assert exact_ids == []
    assert set(prefix_ids) == {1, 2}


def test_exact_and_prefix_queries_stop_before_later_type_columns():
    """A type identity must not expand into a matching taxonomy spelling."""
    frame = pd.DataFrame({
        "bodyId": [101, 102, 103],
        "type": ["aMe17a", "aMe17e", "Other"],
        "instance": ["aMe17a_L", "aMe17e_L", "Other_L"],
        "flywireType": ["", "", ""],
        "hemibrainType": ["", "aMe17a", "aMe17a"],
    })
    body_ids = frame["bodyId"].tolist()

    exact, exact_info = _process_single_neuron(
        "aMe17a", frame, body_ids, verbose=False
    )
    prefix, prefix_info = _process_single_neuron(
        "aMe17a.*", frame, body_ids, verbose=False
    )

    assert exact == [101]
    assert prefix == [101]
    assert exact_info["matched_column"] == "type"
    assert prefix_info["matched_column"] == "type"
