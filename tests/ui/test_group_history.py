"""Unit tests for the custom-group label history registry.

Covers the recording contract of the lite custom grouper:
- cell-granularity upsert keyed by label (board is the complete statement
  for the dataset columns it displays; hidden datasets keep their members);
- a redefined cell COVERS the previous value;
- auto-generated labels (blank -> Group_N) are never recorded;
- identical content only refreshes recency;
- recent ordering + cap; atomic file integrity.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import ui.group_history as gh  # noqa: E402


@pytest.fixture
def history_path(tmp_path, monkeypatch):
    """Redirect the registry file into tmp for full isolation."""
    path = tmp_path / "group_history.json"
    monkeypatch.setattr(gh, "HISTORY_PATH", path)
    return path


class TestRecord:
    def test_records_named_labels_with_members(self, history_path):
        n = gh.record([("aMe", {"male-cns:v0.9": ["aMe12", "aMe10"]})])
        assert n == 1
        rec = gh.get_label("aMe")
        assert rec["members"] == {"male-cns:v0.9": ["aMe12", "aMe10"]}
        assert rec["origin"] == "inline"
        assert history_path.exists()

    def test_blank_and_auto_labels_are_never_recorded(self, history_path):
        n = gh.record([
            ("", {"d": ["x"]}),
            ("Group_1", {"d": ["y"]}),
            ("Group_12", {"d": ["z"]}),
            ("real_label", {"d": ["w"]}),
        ])
        assert n == 1
        assert gh.list_recent() == ["real_label"]
        assert gh.get_label("Group_1") is None

    def test_redefined_cell_covers_previous_value(self, history_path):
        gh.record([("aMe", {"male-cns:v0.9": ["aMe12", "aMe10"]})])
        gh.record([("aMe", {"male-cns:v0.9": ["aMe9"]})])
        assert gh.get_label("aMe")["members"]["male-cns:v0.9"] == ["aMe9"]

    def test_empty_cell_covers_too(self, history_path):
        """The board is the complete statement for its dataset columns."""
        gh.record([("aMe", {"male-cns:v0.9": ["aMe12"]})])
        gh.record([("aMe", {"male-cns:v0.9": []})])
        assert gh.get_label("aMe")["members"]["male-cns:v0.9"] == []

    def test_recent_shows_only_groups_with_reusable_members(self, history_path):
        gh.record([("empty", {"male-cns:v0.9": []}),
                   ("valid", {"male-cns:v0.9": ["aMe12"]})])
        assert gh.list_recent() == ["valid"]
        assert set(gh.valid_labels()) == {"valid"}

    def test_untouched_datasets_keep_their_members(self, history_path):
        gh.record([("aMe", {"male-cns:v0.9": ["aMe12"]})])
        # A later single-dataset run for another dataset must not wipe the
        # male-cns members (label accumulates across datasets over time).
        gh.record([("aMe", {"hemibrain:v1.2.1": ["aMe12_R"]})])
        members = gh.get_label("aMe")["members"]
        assert members["male-cns:v0.9"] == ["aMe12"]
        assert members["hemibrain:v1.2.1"] == ["aMe12_R"]

    def test_member_values_are_cleaned(self, history_path):
        gh.record([("aMe", {"d": [" aMe12 ", "", "  "]})])
        assert gh.get_label("aMe")["members"]["d"] == ["aMe12"]


class TestRecency:
    def test_recent_order_follows_usage(self, history_path):
        gh.record([("a", {"d": ["1"]})])
        gh.record([("b", {"d": ["2"]})])
        gh.record([("a", {"d": ["1x"]})])
        assert gh.list_recent() == ["a", "b"]

    def test_identical_content_only_bumps_recency(self, history_path):
        gh.record([("a", {"d": ["1"]})])
        first = gh.get_label("a")["updated_at"]
        gh.record([("b", {"d": ["2"]})])
        gh.record([("a", {"d": ["1"]})])  # identical members
        assert gh.list_recent()[0] == "a"
        assert gh.get_label("a")["members"] == {"d": ["1"]}
        assert gh.get_label("a")["updated_at"] >= first

    def test_recent_cap(self, history_path):
        gh.record([(f"lab{i}", {"d": [str(i)]}) for i in range(gh.RECENT_CAP + 5)])
        recent = gh.list_recent()
        assert len(recent) == gh.RECENT_CAP
        assert recent[0] == f"lab{gh.RECENT_CAP + 4}"


class TestRegistryOps:
    def test_remove_label(self, history_path):
        gh.record([("a", {"d": ["1"]}), ("b", {"d": ["2"]})])
        assert gh.remove_label("a") is True
        assert gh.get_label("a") is None
        assert "a" not in gh.list_recent()
        assert gh.remove_label("a") is False

    def test_clear(self, history_path):
        gh.record([("a", {"d": ["1"]})])
        assert gh.clear() is True
        assert gh.list_recent() == []
        assert gh.all_labels() == {}

    def test_corrupt_file_falls_back_to_empty(self, history_path):
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text("{not json", encoding="utf-8")
        assert gh.list_recent() == []
        # recording still works afterwards (recreates a valid file)
        gh.record([("a", {"d": ["1"]})])
        assert gh.get_label("a") is not None

    def test_no_tmp_file_lingers(self, history_path):
        gh.record([("a", {"d": ["1"]})])
        leftovers = list(history_path.parent.glob("*.tmp*"))
        assert leftovers == []
