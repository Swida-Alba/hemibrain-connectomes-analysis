"""Coverage for the parquet-backed neuron resolver used by analysis code."""

import pandas as pd
import polars as pl


def _write_search_cache(tmp_path):
    from src.neuron_index_builder import build_search_cache_frame, search_cache_path

    dataset = "cache-test:v1.0"
    folder = dataset.replace(":", "_").replace(".", "_")
    cache_dir = tmp_path / "cache" / folder
    cache_dir.mkdir(parents=True)
    index_path = cache_dir / "neuron_index.parquet"
    frame = pl.DataFrame(
        {
        "bodyId": ["100", "200", "300", "400", "500", "600"],
        "type": ["MTe01a", "Other", "MeVPaMe2", "", "aMe17a", "aMe17e"],
        "instance": ["MTe01a_L", "Other_L", "MeVPaMe2_R", "MTe_R", "aMe17a_L", "aMe17e_L"],
        "flywireType": ["MTe07", "MTe12", "aMe19a", "MTe27", "aMe17a1", ""],
        "hemibrainType": ["", "", "", "", "", "aMe17a"],
        }
    )
    frame.write_parquet(index_path)
    build_search_cache_frame(frame).write_parquet(search_cache_path(index_path))
    return dataset, frame


def test_cached_resolver_preserves_priority_and_explicit_prefix_short_circuit(tmp_path):
    from src.neuron_search import get_cached_neuron_search, resolve_neuron_query
    from src.statvis import _process_single_neuron

    dataset, frame = _write_search_cache(tmp_path)
    cache = get_cached_neuron_search(dataset, cache_root=tmp_path / "cache")
    assert cache is not None

    # Bare names are strict and priority ordered.
    ids, info = resolve_neuron_query(cache, "MTe01a")
    assert ids == ["100"]
    assert info["matched_column"] == "type"

    # The cached resolver applies the same first-column rule as the
    # dataframe fallback: the type-owned aMe17a identity does not expand to
    # the later hemibrainType value on the independent aMe17e row.
    ids, info = resolve_neuron_query(cache, "aMe17a")
    assert ids == ["500"]
    assert info["matched_column"] == "type"
    ids, info = resolve_neuron_query(cache, "aMe17a.*")
    assert ids == ["500"]
    assert info["matched_column"] == "type"

    # A literal prefix stops at the first priority column. Later-column names
    # remain available to the viewer as secondary evidence, but must not
    # expand a pathfinding query after a type match exists.
    ids, info = resolve_neuron_query(cache, "MTe.*")
    assert ids == ["100"]
    assert info["matched_column"] == "type"

    # The plain token is not silently promoted to a prefix query.
    ids, info = resolve_neuron_query(cache, "MeVPa")
    assert ids == []
    assert info["matched_column"] is None

    # Numeric automatic searches are identity-only and prefix-restricted.
    ids, info = resolve_neuron_query(cache, "20")
    assert ids == []
    assert info["matched_column"] is None
    ids, info = resolve_neuron_query(cache, "20.*")
    assert ids == ["200"]
    assert info["matched_column"] == "bodyId"

    # A type scope may intentionally search a numeric-looking type name.
    ids, info = resolve_neuron_query(cache, "200", search_columns="type")
    assert ids == []
    assert info["matched_column"] is None

    # Compare the cache resolver with the authoritative dataframe matcher for
    # the legacy pathfinding forms. Ordering may differ because the sidecar is
    # priority/value sorted, but the resolved neuron set must be identical.
    pandas_frame = frame.to_pandas()
    body_ids = pandas_frame["bodyId"].tolist()
    for query in ("MTe01a", "MTe.*", "MeVPa", "MeVPa.*", ".*aMe.*", "20", "20.*"):
        cached_ids, _ = resolve_neuron_query(cache, query)
        dataframe_ids, _ = _process_single_neuron(
            query,
            pandas_frame,
            body_ids,
            verbose=False,
        )
        assert set(cached_ids) == {str(value) for value in dataframe_ids}


def test_get_neurons_uses_cache_resolver_before_dataframe_scan(tmp_path, monkeypatch):
    import src.statvis as statvis
    from src.neuron_search import get_cached_neuron_search

    dataset, frame = _write_search_cache(tmp_path)
    cache = get_cached_neuron_search(dataset, cache_root=tmp_path / "cache")
    assert cache is not None
    pandas_frame = frame.to_pandas()
    roi_frame = pd.DataFrame({"bodyId": pandas_frame["bodyId"]})

    monkeypatch.setattr(
        statvis,
        "_ensure_local_dataset_files",
        lambda *args, **kwargs: ("cache-test_v1_0", "unused-prefix"),
    )
    monkeypatch.setattr(
        statvis,
        "_get_cached_neuron_df",
        lambda *args, **kwargs: (pandas_frame.copy(), roi_frame.copy()),
    )
    monkeypatch.setattr(statvis, "_get_cached_neuron_search", lambda _dataset: cache)

    def fail_dataframe_match(*args, **kwargs):
        raise AssertionError("cached query unexpectedly fell back to dataframe scan")

    monkeypatch.setattr(statvis, "_process_single_neuron", fail_dataframe_match)

    neurons, _, _, _ = statvis.getNeurons(
        ["MTe.*"],
        dataset=dataset,
        verbose=False,
    )
    assert neurons["bodyId"].tolist() == ["100"]
