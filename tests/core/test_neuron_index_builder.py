"""Tests for the compact projection of pulled neuron metadata."""

import polars as pl


def test_metadata_projection_keeps_scalar_fields_and_drops_blob_columns(tmp_path):
    from src.neuron_index_builder import read_metadata_projection

    source = tmp_path / "sample_allneurons_neuron_df.csv"
    pl.DataFrame(
        {
            "bodyId": [12345678901234567890, 42],
            "type": ["aMe12", "APL"],
            "instance": ["aMe12_L", "APL_1"],
            "post": [10, 3],
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
        "bodyId", "type", "instance", "post", "somaSide", "notes",
        "matchingNotes", "synonyms",
    ]
    assert projection["bodyId"].to_list() == ["12345678901234567890", "42"]
    assert projection["type"].to_list() == ["aMe12", "APL"]
    assert "roiInfo" not in projection.columns
    assert "inputRois" not in projection.columns
    assert "notes" in projection.columns
    assert "matchingNotes" in projection.columns
    assert "synonyms" in projection.columns
    assert "last_fetched" not in projection.columns


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
