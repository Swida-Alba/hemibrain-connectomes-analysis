"""Tests for NeuronFilter / parse_neuron_query: legacy list/int/string and
dict-based operator filtering over neuron DataFrames."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.neuron_filter import NeuronFilter, parse_neuron_query


@pytest.fixture
def neurons():
    return pd.DataFrame({
        "type": ["aMe12", "aMe12", "DN1p", "Mi1"],
        "instance": ["aMe12_R", "aMe12_L", "DN1p_R", "Mi1"],
        "bodyId": [101, 102, 103, 104],
    })


class TestParsing:
    def test_none_matches_all(self, neurons):
        f = parse_neuron_query(None)
        assert f.match_all
        assert len(f.apply(neurons)) == 4

    def test_int_is_bodyid(self, neurons):
        f = parse_neuron_query(101)
        assert len(f.apply(neurons)) == 1
        assert f.get_bodyIds(neurons) == [101]

    def test_exact_string(self, neurons):
        f = parse_neuron_query("Mi1")
        assert len(f.apply(neurons)) == 1

    def test_regex_string(self, neurons):
        f = parse_neuron_query("DN.*")
        assert len(f.apply(neurons)) == 1

    def test_list_of_exact_names(self, neurons):
        f = parse_neuron_query(["aMe12", "Mi1"])
        assert len(f.apply(neurons)) == 3

    def test_mixed_exact_and_regex_list_is_anded(self, neurons):
        # Pinned contract (test_audit_fixes): mixed exact+regex legacy lists
        # are AND-ed across operator groups, so nothing can match both.
        f = parse_neuron_query(["aMe.*", "Mi1"])
        assert len(f.apply(neurons)) == 0


class TestDictOperators:
    def test_contains(self, neurons):
        f = parse_neuron_query({"contains": "aMe"})
        assert len(f.apply(neurons)) == 2

    def test_startswith_list_or(self, neurons):
        f = parse_neuron_query({"startswith": ["aMe", "Mi"]})
        assert len(f.apply(neurons)) == 3

    def test_endswith(self, neurons):
        f = parse_neuron_query({"endswith": "_R"})
        assert len(f.apply(neurons)) == 2  # aMe12_R, DN1p_R

    def test_exact(self, neurons):
        f = parse_neuron_query({"exact": "DN1p"})
        assert len(f.apply(neurons)) == 1

    def test_not_contains_or_semantics_across_columns(self, neurons):
        # Pinned contract (see test_audit_fixes equivalence tests): a row
        # matches not_contains when ANY searchable column lacks the pattern
        # (every row has a bodyId string without 'aMe' here).
        f = parse_neuron_query({"not_contains": "aMe"})
        assert len(f.apply(neurons)) == 4

    def test_regex(self, neurons):
        f = parse_neuron_query({"regex": "^Mi"})
        assert len(f.apply(neurons)) == 1

    def test_multi_operator_is_and(self, neurons):
        f = parse_neuron_query({"contains": "aMe", "endswith": "_R"})
        assert len(f.apply(neurons)) == 1  # only aMe12_R

    def test_empty_frame(self):
        f = parse_neuron_query({"contains": "x"})
        empty = pd.DataFrame(columns=["type", "bodyId"])
        assert f.apply(empty).empty


class TestBodyIds:
    def test_get_bodyids_matches_apply(self, neurons):
        f = parse_neuron_query({"contains": "aMe"})
        assert sorted(f.get_bodyIds(neurons)) == [101, 102]
