"""Tests for the persistent neuron-query history (ui/history_store.py).

History is a convenience layer: it must never raise, must tolerate a missing
or corrupt file, and must order recency / frequency exactly as the
auto-suggest dropdown expects.
"""

import json

import pytest

import ui.history_store as history_store


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point the store at a temporary history file."""
    monkeypatch.setattr(
        history_store, "_HISTORY_PATH", tmp_path / "neuron_history.json"
    )
    return history_store


class TestRecord:
    def test_record_and_recent_order(self, store):
        store.record(["aMe12", "aMe10"], now="2026-08-11T10:00:00")
        store.record(["aMe12"], now="2026-08-11T10:05:00")
        assert store.recent() == ["aMe12", "aMe10"]

    def test_frequent_sorts_by_count_then_recency(self, store):
        store.record(["aMe12"], now="2026-08-11T10:00:00")
        store.record(["aMe10"], now="2026-08-11T10:01:00")
        store.record(["aMe12"], now="2026-08-11T10:02:00")
        assert store.frequent() == ["aMe12", "aMe10"]
        # Equal counts: the more recently used value wins.
        store.record(["aMe10"], now="2026-08-11T10:03:00")
        assert store.frequent() == ["aMe10", "aMe12"]

    def test_recent_and_frequent_limits(self, store):
        for i in range(12):
            store.record([f"N{i}"], now=f"2026-08-11T10:{i:02d}:00")
        assert store.recent() == [f"N{i}" for i in range(11, 1, -1)]
        assert len(store.frequent()) == 5

    def test_duplicates_in_one_call_count_once(self, store):
        store.record(["aMe12", "aMe12"], now="2026-08-11T10:00:00")
        assert store.frequent() == ["aMe12"]
        assert store.recent() == ["aMe12"]

    def test_non_string_values_normalized(self, store):
        store.record([5813012345], now="2026-08-11T10:00:00")
        assert store.recent() == ["5813012345"]

    def test_empty_and_blank_values_ignored(self, store):
        store.record([])
        assert not store._HISTORY_PATH.exists()
        store.record(["", "  ", None])
        assert store.recent() == []
        assert store.frequent() == []

    def test_persists_to_json_file(self, store):
        store.record(["aMe12"], now="2026-08-11T10:00:00")
        raw = json.loads(store._HISTORY_PATH.read_text(encoding="utf-8"))
        assert raw["values"]["aMe12"] == {"count": 1, "last_used": "2026-08-11T10:00:00"}

    def test_clear(self, store):
        store.record(["aMe12"], now="2026-08-11T10:00:00")
        store.clear()
        assert store.recent() == []
        assert store.frequent() == []

    def test_remove_deletes_value_from_all_history_views(self, store):
        store.record(["aMe12", "aMe10"], now="2026-08-11T10:00:00")
        assert store.remove("aMe12") is True
        assert store.recent() == ["aMe10"]
        assert store.frequent() == ["aMe10"]
        assert store.remove("aMe12") is False

    def test_custom_provenance_prunes_only_orphaned_custom_values(self, store):
        store.record(["test", "aMe12"], now="2026-08-11T10:00:00",
                     custom_values=["test"])
        raw = json.loads(store._HISTORY_PATH.read_text(encoding="utf-8"))
        assert raw["values"]["test"]["kind"] == "custom"
        assert "test" in store.prune_orphaned_custom([])
        assert store.recent() == ["aMe12"]

    def test_dataset_scope_filters_query_history_and_preserves_multi_dataset_use(
        self, store
    ):
        store.record(["only_a"], now="2026-08-11T10:00:00",
                     datasets=["dataset-a"])
        store.record(["only_b"], now="2026-08-11T10:01:00",
                     datasets=["dataset-b"])
        store.record(["both"], now="2026-08-11T10:02:00",
                     datasets=["dataset-a", "dataset-b"])

        assert store.recent(datasets=["dataset-a"]) == ["both", "only_a"]
        assert store.recent(datasets=["dataset-b"]) == ["both", "only_b"]

        # A later run adds a dataset rather than replacing prior provenance.
        store.record(["only_a"], now="2026-08-11T10:03:00",
                     datasets=["dataset-b"])
        assert store.recent(datasets=["dataset-b"])[0] == "only_a"

    def test_custom_group_and_saved_map_follow_their_member_datasets(
        self, store, tmp_path, monkeypatch
    ):
        from ui import group_history, mapping_store

        monkeypatch.setattr(
            group_history, "HISTORY_PATH", tmp_path / "group_history.json"
        )
        monkeypatch.setattr(mapping_store, "_store_dir", tmp_path / "mappings")
        monkeypatch.setattr(
            mapping_store, "_store_file", tmp_path / "mappings.json"
        )

        group_history.record([
            ("group_a", {"dataset-a": ["aMe12"], "dataset-b": []})
        ])
        store.record(["group_a"], now="2026-08-11T10:00:00",
                     custom_values=["group_a"], datasets=["dataset-a"])
        mapping_store.save_mapping(
            "map_a",
            {
                "source_mapping": {
                    "custom_label": ["mapped"],
                    "dataset-a": [["aMe12"]],
                    "dataset-b": [[]],
                }
            },
        )
        store.record(["map_a"], now="2026-08-11T10:01:00",
                     datasets=["dataset-a"])

        assert "group_a" in store.recent(datasets=["dataset-a"])
        assert "group_a" not in store.recent(datasets=["dataset-b"])
        assert "map_a" in store.recent(datasets=["dataset-a"])
        assert "map_a" not in store.recent(datasets=["dataset-b"])

    def test_legacy_unknown_values_are_not_assumed_to_be_invalid(self, store):
        store.record(["ordinary"], now="2026-08-11T10:00:00")
        assert store.prune_orphaned_custom([]) == []
        assert store.recent() == ["ordinary"]


class TestResilience:
    def test_missing_file_yields_empty(self, store):
        assert store.recent() == []
        assert store.frequent() == []

    def test_corrupt_file_swallowed(self, store):
        store._HISTORY_PATH.write_text("{not json", encoding="utf-8")
        assert store.recent() == []
        assert store.frequent() == []
        # A subsequent record still works (and repairs the file).
        store.record(["aMe12"], now="2026-08-11T10:00:00")
        assert store.recent() == ["aMe12"]

    def test_wrong_shape_swallowed(self, store):
        store._HISTORY_PATH.write_text(json.dumps(["aMe12"]), encoding="utf-8")
        assert store.recent() == []

    def test_read_only_history_dir_never_raises(self, tmp_path, monkeypatch):
        """A failing write must not break a pathfinding run."""
        monkeypatch.setattr(
            history_store, "_HISTORY_PATH", tmp_path / "missing_dir" / "history.json"
        )
        history_store.record(["aMe12"])  # parent dir does not exist -> OSError
        assert history_store.recent() == []
