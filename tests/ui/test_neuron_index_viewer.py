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

        body_sorted = query_neuron_index(index, sort_by="bodyId", page_size=4)
        assert [row["bodyId"] for row in body_sorted.rows] == ["100", "200", "300", "400"]

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
        assert "Search all columns" in labels
        assert "Filter column" in labels
        assert "Sort by" in labels
        assert any(type(el).__name__ == "Table" for el in client.elements.values())

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
