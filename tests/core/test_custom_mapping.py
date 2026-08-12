"""Tests for the custom type-mapping pipeline: file-based LabelMapper
loading through ComparisonParameters (overlay mode) and
FindNeuronConnection.custom_mapping_file, plus the store-export -> mapper ->
EnrichConnectionTable grouping path."""
import json
import sys
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture
def mapping_file(tmp_path):
    mapping = {
        "source_mapping": {
            "custom_label": ["grpA", "grpB"],
            "hemibrain:v1.2.1": [["aMe12", "aMe12_R"], ["aMe12_L"]],
            "male-cns:v0.9": [["aMe12"], ["aMe12-like"]],
        },
        "target_mapping": {
            "custom_label": ["tg1"],
            "hemibrain:v1.2.1": [["MBON01"]],
        },
    }
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    return str(path)


class TestComparisonParametersOverlay:
    """File-based mappings act as an OVERLAY: explicit source/target neuron
    queries stay, and the mapper only renames matching neurons."""

    def test_explicit_queries_kept_with_mapping(self, mapping_file):
        from comparison.comparison_parameters import ComparisonParameters
        p = ComparisonParameters(
            datasets=["hemibrain:v1.2.1", "male-cns:v0.9"],
            source_neurons=["aMe12"], target_neurons=["PPL101"],
            thresholds=[3], output_folder="/tmp/out",
            overall_mapping_json=mapping_file,
        )
        assert p._mapping_overlay is True
        assert p.source_neurons == ["aMe12"]
        assert p.target_neurons == ["PPL101"]
        mapper = p.overall_label_mapper
        assert mapper.get_label("hemibrain:v1.2.1", "aMe12") == "grpA"
        assert mapper.get_label("male-cns:v0.9", "aMe12-like") == "grpB"
        assert mapper.get_label("male-cns:v0.9", "ZZZ") == "ZZZ"  # identity

    def test_explicit_mapper_and_file_rejected(self, mapping_file):
        from comparison.comparison_parameters import ComparisonParameters
        with pytest.raises(ValueError, match="cannot be combined"):
            ComparisonParameters(
                datasets=["hemibrain:v1.2.1"], source_neurons=["x"],
                target_neurons=["y"], thresholds=[1], output_folder="/tmp/o",
                overall_mapping_json=mapping_file, overall_label_mapper=object(),
            )

    def test_malformed_file_rejected(self, tmp_path):
        from comparison.comparison_parameters import ComparisonParameters
        bad = tmp_path / "bad.json"
        bad.write_text('{"foo": 1}', encoding="utf-8")
        with pytest.raises(ValueError, match="source_mapping"):
            ComparisonParameters(
                datasets=["hemibrain:v1.2.1"], source_neurons=["x"],
                target_neurons=["y"], thresholds=[1], output_folder="/tmp/o",
                overall_mapping_json=str(bad),
            )

    def test_no_mapping_params_keeps_plain_behavior(self):
        from comparison.comparison_parameters import ComparisonParameters
        p = ComparisonParameters(
            datasets=["hemibrain:v1.2.1"], source_neurons=["aMe12"],
            target_neurons=["PPL101"], thresholds=[3], output_folder="/tmp/out",
        )
        assert p._mapping_overlay is False
        assert p.overall_label_mapper is None


class TestCoanaCustomMappingFile:
    def test_mapper_loaded_from_file(self, mapping_file, monkeypatch):
        import coana
        monkeypatch.setattr(coana, "Client", lambda *a, **k: object())
        fc = coana.FindNeuronConnection(
            dataset="hemibrain:v1.2.1", sourceNeurons=["MBON01"], targetNeurons=[],
            use_cache=True, custom_mapping_file=mapping_file,
        )
        assert fc.label_mapper is not None
        assert fc.label_mapper.get_label("hemibrain:v1.2.1", "aMe12") == "grpA"
        assert fc.label_mapper.get_label("hemibrain:v1.2.1", "unmapped") == "unmapped"

    def test_explicit_mapper_takes_precedence(self, mapping_file, monkeypatch):
        import coana
        monkeypatch.setattr(coana, "Client", lambda *a, **k: object())
        from comparison.label_mapper import LabelMapper
        direct = LabelMapper(source_mapping_dict={"hemibrain:v1.2.1": [["X"]]},
                             source_labels=["DIRECT"])
        fc = coana.FindNeuronConnection(
            dataset="hemibrain:v1.2.1", sourceNeurons=["MBON01"], targetNeurons=[],
            use_cache=True, custom_mapping_file=mapping_file, label_mapper=direct,
        )
        assert fc.label_mapper is direct


class TestStoreExportToEnrichment:
    """The full UI path: store export file -> LabelMapper -> custom grouping
    in EnrichConnectionTable (both engines agree)."""

    def test_mapping_groups_apply_in_enrichment(self, tmp_path, monkeypatch):
        import ui.mapping_store as store
        store_dir = tmp_path / "user_mappings"
        monkeypatch.setattr(store, "_store_dir", store_dir)
        monkeypatch.setattr(store, "_store_file", store_dir / "user_mappings.json")

        from statvis import EnrichConnectionTable as pd_enrich
        from statvis import EnrichConnectionTablePolars
        from comparison.label_mapper import LabelMapper

        mapping = {
            "source_mapping": {
                "custom_label": ["SRC"],
                "ds:map": [["A", "B"]],
            },
            "target_mapping": {
                "custom_label": ["TGT"],
                "ds:map": [["X"]],
            },
        }
        assert store.save_mapping("my preset", mapping, "desc")
        export = store.mapping_file_path("my preset")
        assert export is not None

        d = tmp_path / "datasets" / "ds_map"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"bodyId": [1, 2, 10, 11], "type": ["A", "B", "X", "X"],
                      "post": [100, 90, 300, 280]}).to_csv(
            d / "ds_map_allneurons_neuron_df.csv", index=False)

        lm = LabelMapper(overall_mapping_json=export)
        conn = pd.DataFrame({"bodyId_pre": [1, 2, 3], "bodyId_post": [10, 11, 12],
                             "type_pre": ["A", "B", "C"], "type_post": ["X", "X", "Y"],
                             "weight": [5, 4, 3], "connection_ratio": [5 / 30, 4 / 30, 3 / 30]})
        _, ct_pd, _ = pd_enrich(conn.copy(), traversal_probability_threshold=0,
                                dataset="ds:map", script_path=str(tmp_path), label_mapper=lm)
        _, ct_pl, _ = EnrichConnectionTablePolars(pl.from_pandas(conn), traversal_probability_threshold=0,
                                                  dataset="ds:map", script_path=str(tmp_path), label_mapper=lm)
        pairs_pd = sorted(zip(ct_pd["type_pre"], ct_pd["type_post"]))
        pairs_pl = sorted(zip(ct_pl["type_pre"].to_list(), ct_pl["type_post"].to_list()))
        # mapped A/B -> SRC, X -> TGT; unmapped C stays C, Y stays Y
        assert pairs_pd == pairs_pl == [("C", "Y"), ("SRC", "TGT")]


class TestCanonicalInlineFormat:
    """The lite grouper's canonical export (LabelMapper overall-JSON plus the
    additive ``format``/``groups_meta`` keys) must load exactly like a named
    preset file — format uniformity between UI and scripts."""

    def _canonical_file(self, tmp_path, labels, per_ds):
        mapping = {
            "format": "drocat_custom_groups/v1",
            "groups_meta": {"updated_at": "2026-08-12T00:00:00", "origin": "inline"},
            "source_mapping": {"custom_label": labels, **per_ds},
            "target_mapping": {"custom_label": labels, **per_ds},
        }
        path = tmp_path / "inline.json"
        path.write_text(json.dumps(mapping), encoding="utf-8")
        return str(path)

    def test_additive_meta_keys_do_not_break_loading(self, tmp_path):
        from comparison.label_mapper import LabelMapper
        path = self._canonical_file(
            tmp_path, ["grpA"], {"hemibrain:v1.2.1": [["aMe12", "aMe12_R"]]})
        lm = LabelMapper(overall_mapping_json=path)
        assert sorted(lm.get_neurons_for_label("grpA", "hemibrain:v1.2.1", "source")) \
            == ["aMe12", "aMe12_R"]
        # both roles identical -> target resolves the same
        assert sorted(lm.get_neurons_for_label("grpA", "hemibrain:v1.2.1", "target")) \
            == ["aMe12", "aMe12_R"]

    def test_auto_named_inline_equals_named_preset(self, tmp_path):
        from comparison.label_mapper import LabelMapper
        # auto-named single group vs an explicitly named preset: the mapper
        # only cares about label->members, so membership is identical
        inline = self._canonical_file(
            tmp_path, ["Group_1"], {"male-cns:v0.9": [["aMe12"]]})
        preset = tmp_path / "preset.json"
        preset.write_text(json.dumps({
            "source_mapping": {"custom_label": ["Group_1"],
                               "male-cns:v0.9": [["aMe12"]]},
            "target_mapping": {"custom_label": ["Group_1"],
                               "male-cns:v0.9": [["aMe12"]]},
        }), encoding="utf-8")
        lm_inline = LabelMapper(overall_mapping_json=inline)
        lm_preset = LabelMapper(overall_mapping_json=str(preset))
        for role in ("source", "target"):
            assert (lm_inline.get_neurons_for_label("Group_1", "male-cns:v0.9", role)
                    == lm_preset.get_neurons_for_label("Group_1", "male-cns:v0.9", role)
                    == ["aMe12"])

    def test_empty_group_for_missing_dataset_passes_validation(self, tmp_path):
        """Cross-dataset runs require every dataset present in a role; an
        empty group [] satisfies that without members."""
        from comparison.label_mapper import LabelMapper
        path = self._canonical_file(tmp_path, ["grpA"], {
            "male-cns:v0.9": [["aMe12"]],
            "hemibrain:v1.2.1": [[]],
        })
        lm = LabelMapper(overall_mapping_json=path)
        lm.validate_datasets(["male-cns:v0.9", "hemibrain:v1.2.1"], role="both")
        assert lm.get_neurons_for_label("grpA", "hemibrain:v1.2.1", "source") == []

    def test_fnc_loads_canonical_file(self, tmp_path, monkeypatch):
        from coana import FindNeuronConnection
        path = self._canonical_file(
            tmp_path, ["grpA"], {"male-cns:v0.9": [["aMe12"]]})
        fc = FindNeuronConnection.__new__(FindNeuronConnection)
        fc.custom_mapping_file = path
        fc.label_mapper = None
        # mirror the __post_init__ snippet that materializes the mapper
        from comparison.label_mapper import LabelMapper
        if fc.custom_mapping_file and fc.label_mapper is None:
            fc.label_mapper = LabelMapper(overall_mapping_json=fc.custom_mapping_file)
        assert fc.label_mapper is not None and not fc.label_mapper.is_empty
        assert fc.label_mapper.get_neurons_for_label("grpA", "male-cns:v0.9", "source") == ["aMe12"]

    def test_group_label_query_expands_to_members(self, tmp_path):
        """A query token equal to a custom label expands to its members so a
        pushed group label resolves as a pathfinding query."""
        from coana import FindNeuronConnection
        path = self._canonical_file(
            tmp_path, ["aMe"], {"male-cns:v0.9": [["aMe12", "aMe10"]]})
        fc = FindNeuronConnection.__new__(FindNeuronConnection)
        fc.custom_mapping_file = path
        fc.label_mapper = None
        from comparison.label_mapper import LabelMapper
        fc.label_mapper = LabelMapper(overall_mapping_json=path)
        fc.dataset = "male-cns:v0.9"
        expanded = fc._expand_group_labels(["aMe"], "source")
        assert expanded == ["aMe12", "aMe10"]
        # Non-label tokens pass through untouched.
        assert fc._expand_group_labels(["aMe", "PPL101"], "source") == \
            ["aMe12", "aMe10", "PPL101"]
        # Without a mapper the query is unchanged.
        fc.label_mapper = None
        assert fc._expand_group_labels(["aMe"], "source") == ["aMe"]
