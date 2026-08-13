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
