"""Tests for the cached neuron-index viewer data layer and UI shell."""

from types import SimpleNamespace

import polars as pl
import pytest


def _write_index(tmp_path, dataset="test:v1.0"):
    folder = dataset.replace(":", "_").replace(".", "_")
    cache_dir = tmp_path / "cache" / folder
    cache_dir.mkdir(parents=True)
    index_path = cache_dir / "neuron_index.parquet"
    pl.DataFrame(
        {
            "bodyId": ["100", "200", "300", "400"],
            "type": ["", "aMe10", "APL", "aMe12"],
            "instance": ["", "aMe10_R", "APL_1", "aMe12_L"],
            "post": [2, 4, 6, 8],
            "downstream_complete": [True, True, False, True],
        }
    ).write_parquet(index_path)
    return dataset, folder, index_path


def _write_priority_index(tmp_path, dataset="priority:v1.0"):
    folder = dataset.replace(":", "_").replace(".", "_")
    cache_dir = tmp_path / "cache" / folder
    cache_dir.mkdir(parents=True)
    index_path = cache_dir / "neuron_index.parquet"
    pl.DataFrame(
        {
            "bodyId": ["aMe-body", "100", "200", "300", "400"],
            "type": ["", "aMe-type", "", "", ""],
            "instance": ["hint", "", "aMe-instance", "", ""],
            "flywireType": ["", "", "", "aMe-other", ""],
            "hemilineage": ["", "", "", "", "aMe-ignored"],
            "post": [1, 2, 3, 4, 5],
        }
    ).write_parquet(index_path)
    return dataset


@pytest.fixture
def isolated_index_root(tmp_path, monkeypatch):
    import ui.neuron_index as neuron_index

    monkeypatch.setattr(neuron_index, "PROJECT_ROOT", tmp_path)
    neuron_index.clear_neuron_index_cache()
    yield tmp_path
    neuron_index.clear_neuron_index_cache()


class TestNeuronIndexData:
    def test_viewer_requires_cached_index_even_when_metadata_table_exists(
        self, isolated_index_root
    ):
        from ui.neuron_index import load_cached_neuron_index

        dataset = "test:v1.0"
        dataset_dir = isolated_index_root / "datasets" / "test_v1_0"
        dataset_dir.mkdir(parents=True)
        pl.DataFrame(
            {"bodyId": ["1"], "type": ["APL"], "instance": ["APL_1"]}
        ).write_parquet(dataset_dir / "test_v1_0_allneurons_neuron_df.parquet")

        with pytest.raises(FileNotFoundError):
            load_cached_neuron_index(dataset)

    def test_load_keeps_cached_rows_and_fills_blank_identifiers(
        self, isolated_index_root
    ):
        from ui.neuron_index import load_cached_neuron_index

        dataset, folder, index_path = _write_index(isolated_index_root)
        dataset_dir = isolated_index_root / "datasets" / folder
        dataset_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "bodyId": ["100", "200", "999"],
                "type": ["aMe1", "aMe10", "not_in_cache"],
                "instance": ["aMe1_L", "aMe10_R", "not_in_cache_1"],
            }
        ).write_parquet(dataset_dir / f"{folder}_allneurons_neuron_df.parquet")

        index = load_cached_neuron_index(dataset)
        assert index.path == index_path
        assert index.enriched is True
        assert index.frame.height == 4
        assert index.frame.filter(pl.col("bodyId") == "100")["type"].item() == "aMe1"
        assert index.frame.filter(pl.col("bodyId") == "100")["instance"].item() == "aMe1_L"
        assert index.frame.filter(pl.col("bodyId") == "999").height == 0

    def test_query_filters_sorts_full_index_before_paging(self, isolated_index_root):
        from ui.neuron_index import load_cached_neuron_index, query_neuron_index

        dataset, _, _ = _write_index(isolated_index_root)
        index = load_cached_neuron_index(dataset, enrich=False)

        result = query_neuron_index(
            index,
            search="ame",
            sort_by="type",
            page=2,
            page_size=1,
        )
        assert result.total == 2
        assert result.pages == 2
        assert result.page == 2
        assert result.rows[0]["type"] == "aMe12"

        filtered = query_neuron_index(
            index,
            filter_column="bodyId",
            filter_text="30",
            sort_by="bodyId",
        )
        assert filtered.total == 1
        assert filtered.rows[0]["bodyId"] == "300"

        column_filtered = query_neuron_index(
            index,
            filter_column="type",
            filter_text="APL",
        )
        assert column_filtered.rows[0]["match_column"] == "type"

        body_sorted = query_neuron_index(index, sort_by="bodyId", page_size=4)
        assert [row["bodyId"] for row in body_sorted.rows] == ["100", "200", "300", "400"]

    def test_search_results_default_to_matched_value_ascending(
        self, isolated_index_root
    ):
        from ui.neuron_index import load_cached_neuron_index, query_neuron_index

        dataset = _write_priority_index(isolated_index_root)
        index = load_cached_neuron_index(dataset, enrich=False)
        result = query_neuron_index(index, search="ame", page_size=10)

        assert [row["bodyId"] for row in result.rows] == [
            "aMe-body", "100", "200", "300",
        ]
        assert [row["match_value"] for row in result.rows] == [
            "aMe-body", "aMe-type", "aMe-instance", "aMe-other",
        ]
        assert [row["match_column"] for row in result.rows] == [
            "hint", "type", "instance", "flywireType",
        ]
        assert [row["match_column_key"] for row in result.rows] == [
            "bodyId", "type", "instance", "flywireType",
        ]
        assert result.total == 4

    def test_explicit_sort_overrides_matched_value_default(
        self, isolated_index_root
    ):
        from ui.neuron_index import load_cached_neuron_index, query_neuron_index

        dataset = _write_priority_index(isolated_index_root)
        index = load_cached_neuron_index(dataset, enrich=False)
        result = query_neuron_index(
            index, search="ame", sort_by="bodyId", page_size=10
        )

        assert [row["bodyId"] for row in result.rows] == [
            "100", "200", "300", "aMe-body",
        ]

    def test_matched_value_sort_groups_priority_then_sorts_each_group(
        self, isolated_index_root
    ):
        from ui.neuron_index import load_cached_neuron_index, query_neuron_index

        dataset = "grouped:v1.0"
        folder = dataset.replace(":", "_").replace(".", "_")
        cache_dir = isolated_index_root / "cache" / folder
        cache_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "bodyId": ["aMe-body", "100", "200", "300", "301"],
                "type": ["", "aMe-z", "", "aMe-a", ""],
                "instance": ["", "", "aMe-z-instance", "", "aMe-a-instance"],
                "flywireType": ["", "", "", "", ""],
            }
        ).write_parquet(cache_dir / "neuron_index.parquet")

        index = load_cached_neuron_index(dataset, enrich=False)
        result = query_neuron_index(index, search="ame", page_size=10)

        assert [row["match_column_key"] for row in result.rows] == [
            "bodyId", "type", "type", "instance", "instance",
        ]
        assert [row["match_value"] for row in result.rows] == [
            "aMe-body", "aMe-a", "aMe-z", "aMe-a-instance", "aMe-z-instance",
        ]

    def test_load_refreshes_when_progress_sidecar_changes(self, isolated_index_root):
        from ui.neuron_index import (
            load_cached_neuron_index,
            neuron_index_state_path,
        )

        dataset, folder, _ = _write_index(isolated_index_root)
        first = load_cached_neuron_index(dataset, enrich=False)
        state_path = neuron_index_state_path(
            dataset, isolated_index_root / "cache"
        )
        pl.DataFrame(
            {
                "bodyId": ["100"],
                "downstream_complete": [False],
                "last_fetched": ["2026-08-12T16:00:00"],
                "connection_count": [17],
            }
        ).write_parquet(state_path)

        second = load_cached_neuron_index(dataset, enrich=False)
        assert second is not first
        row = second.frame.filter(pl.col("bodyId") == "100").row(0, named=True)
        assert row["downstream_complete"] is False
        assert row["connection_count"] == 17


class TestNeuronIndexViewer:
    def _click(self, link):
        listener = next(
            listener
            for listener in link._event_listeners.values()
            if listener.type == "click"
        )
        listener.handler(SimpleNamespace())

    def test_link_opens_rendered_cached_index(self, isolated_index_root, monkeypatch):
        from nicegui import Client
        from nicegui.page import page
        import ui.components.neuron_index_viewer as viewer
        from ui.components.neuron_index_viewer import create_neuron_index_viewer_link

        dataset, _, _ = _write_index(isolated_index_root)
        monkeypatch.setattr(viewer, "PROJECT_ROOT", isolated_index_root)

        client = Client(page("/neuron-index-viewer-cached"))
        with client:
            link = create_neuron_index_viewer_link(lambda: dataset)
        self._click(link)

        texts = [el.text for el in client.elements.values() if getattr(el, "text", "")]
        assert "See available neurons" in texts
        assert any("Available neurons · test:v1.0" in text for text in texts)
        labels = [
            getattr(el, "_props", {}).get("label") for el in client.elements.values()
        ]
        assert "Search identities & taxonomy" in labels
        assert "Filter column" in labels
        assert "Sort by" in labels
        tables = [el for el in client.elements.values() if type(el).__name__ == "Table"]
        assert len(tables) == 2
        match_table = next(
            table for table in tables
            if table._props["columns"][0]["name"] == "match_column"
        )
        full_table = next(
            table for table in tables
            if table._props["columns"][0]["name"] == "bodyId"
        )
        assert match_table._props["columns"][0]["label"] == "Matched by"
        assert match_table._props["columns"][1]["name"] == "match_value"
        assert match_table._props["columns"][1]["label"] == "Matched value"
        assert full_table._props["columns"][0]["name"] == "bodyId"
        assert "match_column" not in {
            column["name"] for column in full_table._props["columns"]
        }
        assert "body" in full_table.slots
        assert "drocat-neuron-hit-cell" in full_table.slots["body"].template

    def test_match_panel_supports_multi_select_and_appends_to_query(
        self, isolated_index_root, monkeypatch
    ):
        from nicegui import Client
        from nicegui.page import page
        import ui.components.neuron_index_viewer as viewer
        from ui.components.neuron_index_viewer import create_neuron_index_viewer_link

        dataset, _, _ = _write_index(isolated_index_root)
        monkeypatch.setattr(viewer, "PROJECT_ROOT", isolated_index_root)
        current_query = ["existing"]
        added_batches = []

        def add_to_query(values):
            added_batches.append(list(values))
            for value in values:
                if value not in current_query:
                    current_query.append(value)
            return len(values)

        client = Client(page("/neuron-index-viewer-selection"))
        with client:
            link = create_neuron_index_viewer_link(
                lambda: dataset,
                query_values_getter=lambda: current_query,
                add_to_query=add_to_query,
                query_label="Source Neurons",
            )
        self._click(link)

        tables = [el for el in client.elements.values() if type(el).__name__ == "Table"]
        match_table = next(
            table for table in tables
            if table._props["columns"][0]["name"] == "match_column"
        )
        assert match_table._props["selection"] == "multiple"

        add_button = next(
            el for el in client.elements.values()
            if getattr(el, "text", "") == "Add selected to query"
        )
        selected = [
            {"bodyId": "100", "match_value": "aMe10"},
            {"bodyId": "200", "match_value": "aMe12"},
        ]
        match_table._selection_handlers[0](SimpleNamespace(selection=selected))
        self._click(add_button)

        expected = [row["match_value"] for row in selected]
        assert added_batches == [expected]
        assert current_query == ["existing", *expected]
        assert any(
            getattr(el, "text", "") == "Current query · Source Neurons"
            for el in client.elements.values()
        )
        preview_text = [
            getattr(el, "text", "")
            for el in client.elements.values()
            if "drocat-neuron-query-chip" in getattr(el, "_classes", set())
        ]
        assert preview_text == current_query

    def test_link_explains_how_to_build_a_missing_cache(
        self, isolated_index_root, monkeypatch
    ):
        from nicegui import Client
        from nicegui.page import page
        import ui.components.neuron_index_viewer as viewer
        from ui.components.neuron_index_viewer import create_neuron_index_viewer_link

        dataset = "missing:v2.0"
        monkeypatch.setattr(viewer, "PROJECT_ROOT", isolated_index_root)

        client = Client(page("/neuron-index-viewer-missing"))
        with client:
            link = create_neuron_index_viewer_link(lambda: dataset)
        self._click(link)

        texts = [el.text for el in client.elements.values() if getattr(el, "text", "")]
        joined = "\n".join(texts)
        assert "not cached locally" in joined
        assert "Settings → Dataset Cache" in joined
        assert any(
            getattr(el, "_props", {}).get("content")
            == "python src/build_connection_cache.py missing:v2.0"
            for el in client.elements.values()
        )
        assert "The viewer does not open or stream the original dataset file." in joined
