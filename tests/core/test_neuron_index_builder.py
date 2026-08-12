"""Tests for the local projection of pulled neuron metadata."""

import polars as pl


def test_metadata_projection_keeps_all_metadata_columns_and_source_order(tmp_path):
    from src.neuron_index_builder import read_metadata_projection

    source = tmp_path / "sample_allneurons_neuron_df.csv"
    pl.DataFrame(
        {
            "bodyId": [12345678901234567890, 42],
            "type": ["aMe12", "APL"],
            "instance": ["aMe12_L", "APL_1"],
            "post": [10, 3],
            "size": [100, 200],
            "confidence": [0.9, 0.8],
            "somaSide": ["L", "R"],
            "roiInfo": ["{\"AL\": 1}", "{\"MB\": 2}"],
            "inputRois": ["[AL]", "[MB]"],
            "notes": ["large note", "another note"],
            "matchingNotes": ["match one", "match two"],
            "synonyms": ["syn one", "syn two"],
            "last_fetched": ["old", "old"],
        }
    ).write_csv(source)

    projection = read_metadata_projection(source)

    assert projection.columns == [
        "bodyId", "type", "instance", "post", "size", "confidence",
        "somaSide", "roiInfo", "inputRois", "notes", "matchingNotes",
        "synonyms",
    ]
    assert projection["bodyId"].to_list() == ["12345678901234567890", "42"]
    assert projection["type"].to_list() == ["aMe12", "APL"]
    assert projection["size"].to_list() == [100, 200]
    assert projection["confidence"].to_list() == [0.9, 0.8]
    assert "roiInfo" in projection.columns
    assert "inputRois" in projection.columns
    assert "notes" in projection.columns
    assert "matchingNotes" in projection.columns
    assert "synonyms" in projection.columns
    assert "last_fetched" not in projection.columns


def test_metadata_projection_normalizes_flywire_body_id_alias(tmp_path):
    from src.neuron_index_builder import read_metadata_projection

    source = tmp_path / "flywire_neuron_df.csv"
    source.write_text(
        "root_id,type,instance,flywireType,super_class,size_nm\n"
        "720575940000000001,aMe12,aMe12_L,Me,central,123\n"
        "720575940000000002,APL,APL_1,APL,visual,456\n",
        encoding="utf-8",
    )

    projection = read_metadata_projection(source)

    assert projection.columns == [
        "bodyId", "type", "instance", "flywireType", "super_class",
        "size_nm",
    ]
    assert projection["bodyId"].to_list() == [
        "720575940000000001", "720575940000000002",
    ]
    assert projection["size_nm"].to_list() == [123, 456]


def test_metadata_path_prefers_the_pulled_csv_over_an_old_projection(tmp_path):
    from src.neuron_index_builder import metadata_path

    dataset = "fixture:v1.0"
    folder = tmp_path / "fixture_v1_0"
    folder.mkdir()
    parquet = folder / "fixture_v1_0_allneurons_neuron_df.parquet"
    csv = folder / "fixture_v1_0_allneurons_neuron_df.csv"
    pl.DataFrame({"bodyId": [1], "type": ["old"]}).write_parquet(parquet)
    pl.DataFrame({"bodyId": [1], "type": ["pulled"]}).write_csv(csv)

    assert metadata_path(dataset, tmp_path) == csv


def test_priority_columns_put_cross_dataset_taxonomy_after_instance():
    from src.neuron_index_builder import ordered_projection_columns

    assert ordered_projection_columns([
        "bodyId", "type", "instance", "notes", "class", "mancType",
        "flywireType", "hemibrainType", "super_class", "cell_type", "post",
    ]) == [
        "bodyId", "type", "instance", "flywireType", "hemibrainType",
        "mancType", "cell_type", "class", "super_class", "notes", "post",
    ]


def test_priority_columns_preserve_non_priority_source_order():
    from src.neuron_index_builder import ordered_projection_columns

    assert ordered_projection_columns([
        "bodyId", "instance", "type", "pre", "score", "hemibrainType",
        "comment", "class", "post", "connection_count",
    ]) == [
        "bodyId", "type", "instance", "hemibrainType", "class", "pre",
        "score", "comment", "post", "connection_count",
    ]


def test_priority_columns_leave_measurements_in_source_order():
    from src.neuron_index_builder import ordered_projection_columns

    assert ordered_projection_columns([
        "bodyId", "type", "instance", "size",
        "celltypePredictedNtConfidence", "celltypePredictedNt",
        "celltypeTotalNtPredictions", "receptorType", "notes",
    ]) == [
        "bodyId", "type", "instance", "celltypePredictedNt", "receptorType",
        "size", "celltypePredictedNtConfidence", "celltypeTotalNtPredictions",
        "notes",
    ]
