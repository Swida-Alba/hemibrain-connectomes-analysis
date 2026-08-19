"""Coverage tests for comparison.label_mapper.LabelMapper.

Hermetic: all mapping files are written into pytest tmp_path. No access to
real mappings/ or datasets/ folders.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison.label_mapper import LabelMapper

DS1 = "hemibrain:v1.2.1"
DS2 = "male-cns:v0.9"


def _dict_mapper():
    return LabelMapper(
        source_mapping_dict={DS1: [["aMe12", "aMe12_R"], ["aMe12_L"]], DS2: [["aMe12"], []]},
        source_labels=["aMe12_grp1", "aMe12_grp2"],
        target_mapping_dict={DS1: [["PN1"]], DS2: [["PN1", "PN2"]]},
        target_labels=["pn_grp"],
        intermediate_mapping_dict={DS1: [["IN1"]], DS2: [["IN2"]]},
        intermediate_labels=["in_grp"],
    )


def test_empty_mapper():
    mapper = LabelMapper()
    assert mapper.is_empty
    assert not mapper.has_mapping()
    assert "LabelMapper" in repr(mapper)


def test_dict_mapper_basics():
    mapper = _dict_mapper()
    assert not mapper.is_empty
    assert mapper.has_mapping()
    assert mapper.get_std_label(DS1, "aMe12", "source") == "aMe12_grp1"
    assert mapper.get_std_label(DS1, "aMe12_L", "source") == "aMe12_grp2"
    # hemisphere suffix fallback: aMe12_R not directly mapped in DS2 but base aMe12 is
    assert mapper.get_std_label(DS2, "aMe12_R", "source") == "aMe12_grp1_R"
    # unmapped -> auto-generated
    assert mapper.get_std_label(DS1, "Unknown1", "source") == "Unknown1"
    assert LabelMapper.auto_generate_label(123) == "123"
    # target role
    assert mapper.get_std_label(DS2, "PN2", "target") == "pn_grp"
    # intermediate role
    assert mapper.get_std_label(DS1, "IN1", "intermediate") == "in_grp"
    # sanitized dataset fallback: mapping stored under sanitized key,
    # queried with the original dataset identifier
    san_mapper = LabelMapper(source_mapping_dict={"hemibrain_v1_2_1": [["aMe12"]]}, source_labels=["g"])
    assert san_mapper.get_std_label("hemibrain:v1.2.1", "aMe12", "source") == "g"
    assert san_mapper.get_label("hemibrain:v1.2.1", "aMe12") == "g"
    assert san_mapper.get_neurons_for_label("g", "hemibrain:v1.2.1", "source") == ["aMe12"]
    assert san_mapper.get_all_neurons_for_dataset("hemibrain:v1.2.1", "source") == ["aMe12"]


def test_get_label_priority_and_get_mapped_label():
    mapper = _dict_mapper()
    # get_label checks source first
    assert mapper.get_label(DS1, "aMe12") == "aMe12_grp1"
    assert mapper.get_label(DS1, "PN1") == "pn_grp"
    assert mapper.get_label(DS1, "IN1") == "in_grp"
    assert mapper.get_label(DS1, "nope") == "nope"

    assert mapper.get_mapped_label("aMe12_grp1", DS1) == ["aMe12", "aMe12_R"]
    assert mapper.get_mapped_label("aMe12_grp2", DS2) is None or mapper.get_mapped_label("aMe12_grp2", DS2) == []
    assert mapper.get_mapped_label("pn_grp", DS2) == ["PN1", "PN2"]
    assert mapper.get_mapped_label("in_grp", DS1) == "IN1"
    assert mapper.get_mapped_label("missing", DS1) is None


def test_validate_datasets():
    mapper = _dict_mapper()
    mapper.validate_datasets([DS1, DS2], role="both")
    mapper.validate_datasets([DS1], role="source")
    with pytest.raises(ValueError):
        mapper.validate_datasets([DS1, "unknown-ds"], role="source")
    with pytest.raises(ValueError):
        mapper.validate_datasets(["unknown-ds"], role="target")


def test_merge():
    a = LabelMapper(source_mapping_dict={DS1: [["aMe12"]]}, source_labels=["grp"])
    b = LabelMapper(source_mapping_dict={DS1: [["aMe12", "aMe13"]], DS2: [["aMe12"]]}, source_labels=["grp"])
    a.merge(b)
    assert a.get_neurons_for_label("grp", DS1, "source") == ["aMe12", "aMe13"]
    assert a.get_neurons_for_label("grp", DS2, "source") == ["aMe12"]


def test_queries():
    mapper = _dict_mapper()
    assert mapper.get_all_std_labels("source") == ["aMe12_grp1", "aMe12_grp2"] or set(
        mapper.get_all_std_labels("source")
    ) == {"aMe12_grp1", "aMe12_grp2"}
    assert set(mapper.get_datasets("source")) == {DS1, DS2}
    assert set(mapper.get_all_neurons_for_dataset(DS1, "source")) == {"aMe12", "aMe12_R", "aMe12_L"}
    assert mapper.get_all_neurons_for_dataset("nope", "source") == []
    assert mapper.get_neurons_for_label("aMe12_grp1", DS1, "source") == ["aMe12", "aMe12_R"]
    assert mapper.get_neurons_for_label("missing", DS1, "source") == []
    # hemi suffix fallback on label
    assert mapper.get_neurons_for_label("aMe12_grp1_R", DS1, "source") == ["aMe12", "aMe12_R"]
    # _split_hemi_suffix
    assert LabelMapper._split_hemi_suffix("X_R") == ("X", "_R")
    assert LabelMapper._split_hemi_suffix("X_U") == ("X", "_U")
    assert LabelMapper._split_hemi_suffix("X") == ("X", "")
    assert LabelMapper._split_hemi_suffix(5) == (5, "")


def test_apply_to_dataframe():
    mapper = _dict_mapper()
    df = pd.DataFrame({
        "type_pre": ["aMe12", None],
        "type_post": ["PN1", "PN2"],
        "weight": [5, 2],
    })
    out = mapper.apply_to_dataframe(df, DS1)
    assert out.loc[0, "std_label_pre"] == "aMe12_grp1"
    assert out.loc[1, "std_label_pre"] == ""
    assert out.loc[0, "std_label_post"] == "pn_grp"
    # empty df unchanged
    assert mapper.apply_to_dataframe(pd.DataFrame(), DS1).empty
    # bodyId-based df
    df_body = pd.DataFrame({"bodyId_pre": [1001], "bodyId_post": [2002], "weight": [1]})
    out_body = mapper.apply_to_dataframe(df_body, DS1)
    assert out_body.loc[0, "std_label_pre"] == "1001"


def test_mapping_summary_and_export_parameters():
    mapper = _dict_mapper()
    summary = mapper.get_mapping_summary()
    assert not summary.empty
    assert set(summary["role"]) == {"source", "target", "intermediate"}
    params = mapper.export_to_parameters()
    assert "resolved_source_mapping" in params
    assert params["resolved_source_mapping"]["aMe12_grp1"][DS1] == ["aMe12", "aMe12_R"]


def test_to_dict_export_json_roundtrip(tmp_path):
    mapper = _dict_mapper()
    d = mapper.to_dict()
    assert "source_mapping" in d and "target_mapping" in d and "intermediate_mapping" in d

    path = tmp_path / "mapping.json"
    mapper.export_to_json(str(path))
    assert path.exists()

    reloaded = LabelMapper(overall_mapping_json=str(path))
    assert reloaded.get_std_label(DS1, "aMe12", "source") == "aMe12_grp1"
    assert reloaded.get_std_label(DS2, "PN2", "target") == "pn_grp"
    assert reloaded.get_std_label(DS1, "IN1", "intermediate") == "in_grp"


def test_load_from_json_missing_file():
    with pytest.raises(FileNotFoundError):
        LabelMapper(source_mapping_file="/nonexistent/nowhere.json")


def test_load_from_json_errors(tmp_path):
    # no mapping keys at all
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"foo": 1}))
    with pytest.raises(ValueError):
        LabelMapper(source_mapping_file=str(bad))

    # missing label key
    bad2 = tmp_path / "bad2.json"
    bad2.write_text(json.dumps({"source_mapping": {"ds1": [["a"]]}}))
    with pytest.raises(ValueError):
        LabelMapper(source_mapping_file=str(bad2))

    # group count mismatch
    bad3 = tmp_path / "bad3.json"
    bad3.write_text(json.dumps({
        "source_mapping": {"custom_label": ["l1", "l2"], "ds1": [["a"]]}
    }))
    with pytest.raises(ValueError):
        LabelMapper(source_mapping_file=str(bad3))

    # unified file legitimately missing target role -> ok
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({
        "source_mapping": {"custom_label": ["l1"], "ds1": [["a"]]}
    }))
    mapper = LabelMapper(overall_mapping_json=str(ok))
    assert mapper.get_std_label("ds1", "a", "source") == "l1"

    # non-list group value gets wrapped
    scalar = tmp_path / "scalar.json"
    scalar.write_text(json.dumps({
        "source_mapping": {"custom_label": ["l1"], "ds1": ["a"]}
    }))
    mapper2 = LabelMapper(source_mapping_file=str(scalar))
    assert mapper2.get_std_label("ds1", "a", "source") == "l1"


def test_load_from_csv_expanded_format(tmp_path):
    csv = tmp_path / "source.csv"
    csv.write_text(
        "custom_label,std_pattern,flywire_FAFB,notes\n"
        "grpA,A.*,100,first\n"
        "grpA,A.*_R,101,second\n"
        ",B.*,200,auto label from pattern\n"
        ",,\n"
    )
    mapper = LabelMapper(source_mapping_file=str(csv))
    assert mapper.get_std_label("flywire_FAFB", 100, "source") == "grpA"
    assert mapper.get_std_label("flywire_FAFB", 101, "source") == "grpA"
    assert mapper.get_std_label("flywire_FAFB", 200, "source") == "B.*"
    assert not mapper.is_empty


def test_load_from_csv_legacy_format(tmp_path):
    csv = tmp_path / "legacy.csv"
    csv.write_text(
        "std_label,flywire_FAFB,other_ds\n"
        "grp1,10;abc,30\n"
        ",40;\n"
    )
    mapper = LabelMapper(target_mapping_file=str(csv))
    assert mapper.get_std_label("flywire_FAFB", 10, "target") == "grp1"
    assert mapper.get_std_label("flywire_FAFB", "abc", "target") == "grp1"
    assert mapper.get_std_label("other_ds", 30, "target") == "grp1"
    assert mapper.get_std_label("other_ds", 40, "target") == "40"


def test_load_from_csv_errors(tmp_path):
    missing = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError):
        LabelMapper(source_mapping_file=str(missing))

    bad = tmp_path / "bad.csv"
    bad.write_text("col_a,col_b\n1,2\n")
    with pytest.raises(ValueError):
        LabelMapper(source_mapping_file=str(bad))


def test_verify_csv_format(tmp_path):
    good = tmp_path / "good.csv"
    good.write_text("custom_label,flywire_FAFB\ngrpA,100\ngrpA,101\n,200\n")
    res = LabelMapper.verify_csv_format(str(good))
    assert res["valid"]
    assert res["summary"]["rows"] == 3
    assert res["warnings"]  # empty label warning + duplicate label warning

    res_missing = LabelMapper.verify_csv_format(str(tmp_path / "nope.csv"))
    assert not res_missing["valid"]

    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n")
    res_bad = LabelMapper.verify_csv_format(str(bad))
    assert not res_bad["valid"]

    no_datasets = tmp_path / "nodata.csv"
    no_datasets.write_text("custom_label,notes\ngrpA,x\n")
    res_nd = LabelMapper.verify_csv_format(str(no_datasets))
    assert not res_nd["valid"]


def test_verify_json_format(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({
        "source_mapping": {"custom_label": ["a"], "ds1": [["1"]]},
        "target_mapping": {"std_label": ["b"], "ds1": [["2"]]},
    }))
    res = LabelMapper.verify_json_format(str(good))
    assert res["valid"]

    res_missing = LabelMapper.verify_json_format(str(tmp_path / "nope.json"))
    assert not res_missing["valid"]

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json")
    assert not LabelMapper.verify_json_format(str(bad_json))["valid"]

    no_sections = tmp_path / "empty.json"
    no_sections.write_text("{}")
    assert not LabelMapper.verify_json_format(str(no_sections))["valid"]

    bad_section = tmp_path / "badsec.json"
    bad_section.write_text(json.dumps({"source_mapping": {"ds1": [["a"]]}}))
    assert not LabelMapper.verify_json_format(str(bad_section))["valid"]

    bad_labels_type = tmp_path / "badtype.json"
    bad_labels_type.write_text(json.dumps({"source_mapping": {"custom_label": "notalist", "ds1": [["a"]]}}))
    assert not LabelMapper.verify_json_format(str(bad_labels_type))["valid"]

    bad_ds_type = tmp_path / "baddstype.json"
    bad_ds_type.write_text(json.dumps({"source_mapping": {"custom_label": ["a"], "ds1": "notalist"}}))
    assert not LabelMapper.verify_json_format(str(bad_ds_type))["valid"]

    mismatch = tmp_path / "mismatch.json"
    mismatch.write_text(json.dumps({"source_mapping": {"custom_label": ["a", "b"], "ds1": [["1"]]}}))
    assert not LabelMapper.verify_json_format(str(mismatch))["valid"]


def test_verify_mapping_file_dispatch(tmp_path):
    good_json = tmp_path / "m.json"
    good_json.write_text(json.dumps({"source_mapping": {"custom_label": ["a"], "ds1": [["1"]]}}))
    assert LabelMapper.verify_mapping_file(str(good_json))["valid"]

    good_csv = tmp_path / "m.csv"
    good_csv.write_text("custom_label,ds1\na,1\n")
    assert LabelMapper.verify_mapping_file(str(good_csv))["valid"]

    unsupported = LabelMapper.verify_mapping_file(str(tmp_path / "m.txt"))
    assert not unsupported["valid"]
