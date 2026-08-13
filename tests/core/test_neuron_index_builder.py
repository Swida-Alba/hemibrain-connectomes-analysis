"""Tests for the local projection of pulled neuron metadata."""

import polars as pl


def test_metadata_projection_keeps_metadata_except_large_roi_fields(tmp_path):
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
            "outputRois": ["[AL]", "[MB]"],
            "notes": ["large note", "another note"],
            "matchingNotes": ["match one", "match two"],
            "synonyms": ["syn one", "syn two"],
            "last_fetched": ["old", "old"],
        }
    ).write_csv(source)

    projection = read_metadata_projection(source)

    assert projection.columns == [
        "bodyId", "type", "instance", "post", "size", "confidence",
        "somaSide", "notes", "matchingNotes",
        "synonyms",
    ]
    assert projection["bodyId"].to_list() == ["12345678901234567890", "42"]
    assert projection["type"].to_list() == ["aMe12", "APL"]
    assert projection["size"].to_list() == [100, 200]
    assert projection["confidence"].to_list() == [0.9, 0.8]
    assert "roiInfo" not in projection.columns
    assert "inputRois" not in projection.columns
    assert "outputRois" not in projection.columns
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


def test_search_cache_is_distinct_and_presorted_by_viewer_priority():
    from src.neuron_index_builder import build_search_cache_frame

    frame = pl.DataFrame(
        {
            "bodyId": ["2", "1", "3"],
            "type": ["MeVPaMe2", "aMe01", "aMe01"],
            "instance": ["MeVPaMe2_L", "aMe01_L", ""],
            "flywireType": ["aMe19a", "", "aMe01"],
            "post": [1, 2, 3],
        }
    )

    cache = build_search_cache_frame(frame)

    assert cache.columns == [
        "search_column", "search_priority", "search_value",
        "search_value_folded", "__neuron_rows",
    ]
    columns_by_priority = (
        cache.group_by("search_column")
        .agg(pl.col("search_priority").first())
        .sort("search_priority")["search_column"]
        .to_list()
    )
    assert columns_by_priority == [
        "bodyId", "type", "instance", "flywireType",
    ]
    type_a_me = cache.filter(
        (pl.col("search_column") == "type")
        & (pl.col("search_value") == "aMe01")
    )
    assert type_a_me.height == 1
    assert type_a_me["__neuron_rows"].to_list() == [[1, 2]]
    ordered = cache.select(["search_priority", "search_value"]).to_dicts()
    assert ordered == sorted(
        ordered,
        key=lambda row: (row["search_priority"], row["search_value"]),
    )


def test_search_cache_compatibility_rejects_stale_priority_columns():
    from src.neuron_index_builder import (
        build_search_cache_frame,
        is_search_cache_compatible,
        viewer_search_columns,
    )

    frame = pl.DataFrame({
        "bodyId": ["1"],
        "type": ["aMe17a"],
        "instance": ["aMe17a_L"],
        "flywireType": ["aMe17a1"],
        "locationType": ["aMe17a-central"],
    })
    current_columns = viewer_search_columns(frame.columns)
    current = build_search_cache_frame(frame)
    stale = build_search_cache_frame(frame, current_columns[:-1])

    assert is_search_cache_compatible(current, frame.columns)
    assert not is_search_cache_compatible(stale, frame.columns)
