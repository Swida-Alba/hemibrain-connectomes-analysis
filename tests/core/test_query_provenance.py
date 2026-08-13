"""Regression tests for readable query provenance after backend resolution."""

import pandas as pd


def _connection_stub():
    from src.coana import FindNeuronConnection

    connection = FindNeuronConnection.__new__(FindNeuronConnection)
    connection.dataset = "male-cns:v1.0"
    connection.label_mapper = None
    connection.custom_mapping_file = None
    connection.sourceNeurons = ["100"]
    connection.targetNeurons = ["200"]
    connection._requested_source_neurons = ["aMe17a"]
    connection._requested_target_neurons = ["PPL101"]
    connection.source_df = pd.DataFrame({"bodyId": ["100"]})
    connection.target_df = pd.DataFrame({"bodyId": ["200"]})
    connection.parameter_dict = {}
    return connection


def test_readable_query_name_prefers_original_tokens():
    from src.coana import FindNeuronConnection

    assert FindNeuronConnection._readable_query_name(["aMe17a"]) == "aMe17a"
    assert FindNeuronConnection._readable_query_name(
        ["aMe17a", "aMe17e"]
    ) == "aMe17a_etc"
    assert FindNeuronConnection._readable_query_name([100]) == "100"


def test_run_exports_keep_queries_separate_from_resolved_body_ids():
    connection = _connection_stub()

    attributes = connection._run_export_attributes(path_mode="all")
    assert attributes["requested_source_neurons"] == ["aMe17a"]
    assert attributes["requested_target_neurons"] == ["PPL101"]
    assert attributes["resolved_source_neurons"] == ["100"]
    assert attributes["resolved_target_neurons"] == ["200"]
    assert attributes["resolved_source_bodyIds"] == ["100"]
    assert attributes["resolved_target_bodyIds"] == ["200"]

    connection._add_custom_group_parameters()
    assert connection.parameter_dict["requested source neurons"] == "['aMe17a']"
    assert connection.parameter_dict["requested target neurons"] == "['PPL101']"
    assert connection.parameter_dict["resolved source bodyIds"] == "['100']"
    assert connection.parameter_dict["resolved target bodyIds"] == "['200']"


def test_run_exports_drop_authentication_values_recursively():
    connection = _connection_stub()
    connection.token = "neuprint-secret"
    connection.neuprint_token = "another-secret"
    connection.runtime_options = {
        "safe_option": "kept",
        "cave_token": "cave-secret",
        "nested": {"authorization": "Bearer secret", "safe": 1},
    }

    attributes = connection._run_export_attributes(path_mode="all")

    assert "token" not in attributes
    assert "neuprint_token" not in attributes
    assert attributes["runtime_options"] == {
        "safe_option": "kept",
        "nested": {"safe": 1},
    }
