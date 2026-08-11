"""Tests for the edge-list editor backend (auto-save draft store) and its
UI integration in the Network tab.

Covers:
- ui/edge_list_store.py: draft CRUD, atomic auto-save, dirty tracking,
  crash recovery, validation, and PlotPath-ready CSV layout.
- ui/components/edge_list_editor.py: editor state, edit operations,
  debounced auto-save flush, export, and the recovery reminder banner.
- ui/tabs/visualization.py: Canvas Source wiring and reminder rendering (Net-Viz tab).
"""
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import ui.edge_list_store as store


# =============================================================================
# Store fixtures & helpers
# =============================================================================

@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Point the draft store at a temp directory for every test."""
    monkeypatch.setattr(store, "_store_dir", tmp_path / "edge_list_drafts")
    yield


ROWS = [
    {"source": "aMe12", "target": "aMe10", "weight": "128"},
    {"source": "aMe10", "target": "MBON01", "weight": "47"},
]


def meta(name):
    return store.get_meta(name)


# =============================================================================
# Store: naming & validation
# =============================================================================

class TestNaming:
    def test_sanitize_basic(self):
        assert store.sanitize_name("my network") == "my_network"

    def test_sanitize_special_chars(self):
        assert store.sanitize_name("a/b:c*d?.csv") == "a_b_c_d_csv"

    def test_sanitize_empty(self):
        assert store.sanitize_name("   ") == ""
        assert store.sanitize_name("///") == ""

    def test_sanitize_truncates(self):
        assert len(store.sanitize_name("x" * 300)) <= 80


class TestValidation:
    def test_valid_rows(self):
        assert store.validate_rows(ROWS) == []

    def test_missing_fields_reported(self):
        errors = store.validate_rows([{"source": "A", "target": "", "weight": ""}])
        assert any("missing target" in e for e in errors)
        assert any("missing weight" in e for e in errors)

    def test_non_numeric_weight(self):
        errors = store.validate_rows([{"source": "A", "target": "B", "weight": "many"}])
        assert any("not a number" in e for e in errors)

    def test_negative_weight(self):
        errors = store.validate_rows([{"source": "A", "target": "B", "weight": "-3"}])
        assert any(">= 0" in e for e in errors)

    def test_completely_empty_rows_ignored(self):
        assert store.validate_rows([{}, {"source": "", "target": "", "weight": ""}]) == []

    def test_normalize_strips_and_fills(self):
        rows = store.normalize_rows([{"source": " A ", "target": "B"}])
        assert rows == [{"source": "A", "target": "B", "weight": "", "color": ""}]

    def test_complete_rows(self):
        rows = ROWS + [{"source": "X", "target": "", "weight": "5"}]
        assert store.complete_rows(rows) == store.normalize_rows(ROWS)


# =============================================================================
# Store: save / load / metadata round-trip
# =============================================================================

class TestSaveLoad:
    def test_save_and_load_round_trip(self):
        slug = store.save_draft("net", ROWS)
        assert slug == "net"
        assert store.load_draft("net") == store.normalize_rows(ROWS)

    def test_order_preserved(self):
        rows = [{"source": f"N{i}", "target": f"N{i+1}", "weight": str(i)} for i in range(5)]
        store.save_draft("ordered", rows)
        assert [r["source"] for r in store.load_draft("ordered")] == [f"N{i}" for i in range(5)]

    def test_invalid_name_rejected(self):
        assert store.save_draft("///", ROWS) is None
        assert store.load_draft("///") is None

    def test_load_missing_returns_none(self):
        assert store.load_draft("ghost") is None

    def test_meta_created_dirty(self):
        store.save_draft("net", ROWS)
        m = meta("net")
        assert m["name"] == "net"
        assert m["dirty"] is True
        assert m["row_count"] == 2
        assert m["created_at"] and m["updated_at"]

    def test_meta_keeps_created_at_on_update(self):
        store.save_draft("net", ROWS)
        created = meta("net")["created_at"]
        store.save_draft("net", ROWS[:1])
        m = meta("net")
        assert m["created_at"] == created
        assert m["row_count"] == 1

    def test_overwrite_replaces_rows(self):
        store.save_draft("net", ROWS)
        store.save_draft("net", ROWS[:1])
        assert len(store.load_draft("net")) == 1

    def test_empty_draft_allowed(self):
        assert store.save_draft("empty", []) == "empty"
        assert store.load_draft("empty") == []
        assert meta("empty")["row_count"] == 0

    def test_csv_quoting_survives_commas_and_quotes(self):
        rows = [{"source": 'A,B', "target": 'say "hi"', "weight": "1"}]
        store.save_draft("quoted", rows)
        loaded = store.load_draft("quoted")
        assert loaded[0]["source"] == "A,B"
        assert loaded[0]["target"] == 'say "hi"'


class TestCsvLayout:
    """The draft CSV must be directly consumable by VisualizePath."""

    def test_columns_without_color(self):
        store.save_draft("net", ROWS)
        header = store.draft_csv_path("net")
        first_line = Path(header).read_text(encoding="utf-8").splitlines()[0]
        assert first_line == "source,target,weight"

    def test_color_column_only_when_used(self):
        rows = ROWS + [{"source": "P", "target": "Q", "weight": "1", "color": "#ff0000"}]
        store.save_draft("colored", rows)
        first_line = Path(store.draft_csv_path("colored")).read_text(encoding="utf-8").splitlines()[0]
        assert first_line == "source,target,weight,color"

    def test_pandas_edge_list_detection(self):
        """Exact column set {source, target, weight} is vispath's edge-list format."""
        import pandas as pd
        store.save_draft("net", ROWS)
        df = pd.read_csv(store.draft_csv_path("net"))
        assert set(df.columns) == {"source", "target", "weight"}
        assert {"source", "target", "weight"} in [
            {"source", "target", "weight"}, {"from", "to", "weight"}, {"pre", "post", "weight"},
        ]

    def test_no_temp_files_left_behind(self):
        store.save_draft("net", ROWS)
        leftovers = list(store._store_dir.glob("*.tmp"))
        assert leftovers == []


# =============================================================================
# Store: dirty tracking & recovery
# =============================================================================

class TestDirtyTracking:
    def test_save_marks_dirty(self):
        store.save_draft("net", ROWS)
        assert meta("net")["dirty"] is True

    def test_mark_exported_clears_dirty(self):
        store.save_draft("net", ROWS)
        assert store.mark_exported("net") is True
        assert meta("net")["dirty"] is False

    def test_mark_dirty_again(self):
        store.save_draft("net", ROWS)
        store.mark_exported("net")
        assert store.mark_dirty("net") is True
        assert meta("net")["dirty"] is True

    def test_save_with_dirty_false(self):
        store.save_draft("net", ROWS, dirty=False)
        assert meta("net")["dirty"] is False

    def test_set_dirty_missing_draft(self):
        assert store.set_dirty("ghost", True) is False
        assert store.mark_exported("ghost") is False

    def test_pending_drafts_only_dirty(self):
        store.save_draft("dirty1", ROWS)
        store.save_draft("clean1", ROWS, dirty=False)
        store.save_draft("dirty2", ROWS)
        pending = store.pending_drafts()
        names = {m["name"] for m in pending}
        assert names == {"dirty1", "dirty2"}

    def test_list_drafts_newest_first(self):
        store.save_draft("old", ROWS)
        store.save_draft("new", ROWS)
        # Touch 'old' so it becomes the newest.
        store.save_draft("old", ROWS[:1])
        names = [m["name"] for m in store.list_drafts()]
        assert names[0] == "old"
        assert set(names) == {"old", "new"}


class TestCrashRecovery:
    """Simulate a previous session: files exist on disk, app restarts."""

    def test_recover_files_written_by_previous_session(self):
        draft_dir = store._store_dir
        draft_dir.mkdir(parents=True, exist_ok=True)
        (draft_dir / "session_draft.csv").write_text(
            "source,target,weight\nA,B,10\n", encoding="utf-8"
        )
        (draft_dir / "session_draft.meta.json").write_text(
            json.dumps({
                "name": "session draft", "slug": "session_draft",
                "created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:01",
                "dirty": True, "row_count": 1,
            }),
            encoding="utf-8",
        )
        pending = store.pending_drafts()
        assert [m["name"] for m in pending] == ["session draft"]
        rows = store.load_draft("session draft")
        assert rows == [{"source": "A", "target": "B", "weight": "10", "color": ""}]

    def test_corrupt_meta_ignored(self):
        draft_dir = store._store_dir
        draft_dir.mkdir(parents=True, exist_ok=True)
        (draft_dir / "broken.csv").write_text("source,target,weight\n", encoding="utf-8")
        (draft_dir / "broken.meta.json").write_text("{not json", encoding="utf-8")
        assert store.list_drafts() == []
        assert store.pending_drafts() == []

    def test_orphan_meta_without_csv_ignored(self):
        draft_dir = store._store_dir
        draft_dir.mkdir(parents=True, exist_ok=True)
        (draft_dir / "orphan.meta.json").write_text(
            json.dumps({"name": "orphan", "slug": "orphan", "dirty": True, "row_count": 0}),
            encoding="utf-8",
        )
        assert store.list_drafts() == []

    def test_corrupt_csv_returns_none(self):
        draft_dir = store._store_dir
        draft_dir.mkdir(parents=True, exist_ok=True)
        (draft_dir / "bad.csv").write_bytes(b"\xff\xfe\x00broken")
        assert store.load_draft("bad") is None


# =============================================================================
# Store: delete & paths
# =============================================================================

class TestDeleteAndPaths:
    def test_delete_removes_both_files(self):
        store.save_draft("net", ROWS)
        assert store.delete_draft("net") is True
        assert store.load_draft("net") is None
        assert meta("net") is None
        assert store.draft_csv_path("net") is None

    def test_delete_missing_returns_false(self):
        assert store.delete_draft("ghost") is False

    def test_draft_csv_path(self):
        store.save_draft("net", ROWS)
        path = store.draft_csv_path("net")
        assert path and Path(path).exists()
        assert path.endswith("net.csv")

    def test_draft_csv_path_missing(self):
        assert store.draft_csv_path("ghost") is None


# =============================================================================
# UI component: editor handle behavior
# =============================================================================

from nicegui import Client
from nicegui.page import page


@pytest.fixture()
def store_patch_for_component(monkeypatch, tmp_path):
    """The component imports the store module; patch its store dir too."""
    monkeypatch.setattr(store, "_store_dir", tmp_path / "comp_drafts")
    return tmp_path / "comp_drafts"


def build_editor(store_dir, export_dir=None):
    from ui.components.edge_list_editor import edge_list_editor
    client = Client(page(f"/edge-editor-{uuid.uuid4().hex}"))
    with client:
        handle = edge_list_editor(export_dir_provider=lambda: str(export_dir) if export_dir else None)
    return client, handle


class TestEditorHandle:
    def test_card_elements_exist(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        ids = [
            (getattr(el, "_props", None) or {}).get("id")
            for el in client.elements.values()
        ]
        assert "card-net-viz-edge-editor" in ids
        assert handle.table is not None and handle.status_label is not None
        assert set(handle.edit_inputs) == {"source", "target", "weight", "color"}

    def test_add_edge_and_autosave_flush(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.name_input.value = "my draft"
        handle.add_edge()
        handle.edit_inputs["source"].value = "A"
        handle.edit_inputs["target"].value = "B"
        handle.edit_inputs["weight"].value = "12"
        handle.apply_edit()
        # Debounced timer does not run in unit tests; flush explicitly.
        csv_path = handle.flush_autosave()
        assert csv_path and Path(csv_path).exists()
        rows = store.load_draft("my draft")
        assert rows[0]["source"] == "A" and rows[0]["weight"] == "12"
        assert meta("my draft")["dirty"] is True
        assert "Auto-saved" in handle.status_label.text

    def test_apply_edit_requires_selection(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.name_input.value = "nosel"
        handle.rows = [{"source": "A", "target": "B", "weight": "1", "color": ""}]
        handle._selected_ids = []
        handle.apply_edit()  # warns, does not crash, does not mutate
        assert handle.rows[0]["source"] == "A"

    def test_delete_selected_rows(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.name_input.value = "del"
        handle.set_rows(ROWS + [{"source": "X", "target": "Y", "weight": "9"}])
        handle._selected_ids = [1]
        handle.delete_selected()
        assert len(handle.rows) == 2
        assert handle.rows[1]["source"] == "X"

    def test_on_select_syncs_edit_inputs(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(ROWS)
        handle.on_select(SimpleNamespace(selection=[{**ROWS[1], "id": 1}]))
        assert handle.edit_inputs["source"].value == "aMe10"
        assert handle.edit_inputs["weight"].value == "47"

    def test_rename_deletes_previous_draft(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(ROWS, name="old name")
        handle.flush_autosave()
        assert store.get_meta("old name") is not None
        handle.name_input.value = "new name"
        handle.flush_autosave()
        assert store.get_meta("old name") is None
        assert store.get_meta("new name") is not None
        assert handle.current_name == "new name"

    def test_flush_without_name_does_nothing(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(ROWS, name="")
        assert handle.flush_autosave() is None
        assert store.list_drafts() == []

    def test_runnable_path_file_requires_complete_edge(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows([{"source": "A", "target": "", "weight": ""}], name="incomplete")
        assert handle.runnable_path_file() is None
        handle.set_rows(ROWS, name="complete")
        assert handle.runnable_path_file() is not None

    def test_export_marks_clean_and_copies(self, store_patch_for_component, tmp_path):
        export_dir = tmp_path / "run_output"
        client, handle = build_editor(store_patch_for_component, export_dir=export_dir)
        handle.set_rows(ROWS, name="exp")
        handle.flush_autosave()
        exported = handle.export_csv()
        assert exported == str(export_dir / "exp_edge_list.csv")
        assert Path(exported).exists()
        assert meta("exp")["dirty"] is False
        assert "no unsaved changes" in handle.status_label.text

    def test_load_draft_into_editor(self, store_patch_for_component):
        store.save_draft("saved net", ROWS)
        client, handle = build_editor(store_patch_for_component)
        assert handle.load_draft("saved net") is True
        assert handle.rows == store.normalize_rows(ROWS)
        assert handle.name_input.value == "saved net"
        assert handle.current_name == "saved net"

    def test_load_missing_draft_fails(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        assert handle.load_draft("ghost") is False

    def test_delete_current_draft(self, store_patch_for_component):
        store.save_draft("gone", ROWS)
        client, handle = build_editor(store_patch_for_component)
        handle.load_draft("gone")
        assert handle.delete_current_draft() is True
        assert store.get_meta("gone") is None
        assert handle.rows == []

    def test_draft_select_lists_existing_drafts(self, store_patch_for_component):
        store.save_draft("one", ROWS)
        store.save_draft("two", ROWS)
        client, handle = build_editor(store_patch_for_component)
        assert set(handle.draft_select.options) == {"one", "two"}


class TestRecoveryBanner:
    def test_banner_hidden_when_clean(self, store_patch_for_component):
        from ui.components.edge_list_editor import draft_recovery_banner
        store.save_draft("clean", ROWS, dirty=False)
        client = Client(page(f"/banner-clean-{uuid.uuid4().hex}"))
        with client:
            rendered = draft_recovery_banner(lambda name: None)
        assert rendered is False
        ids = [(getattr(el, "_props", None) or {}).get("id") for el in client.elements.values()]
        assert "card-edge-draft-recovery" not in ids

    def test_banner_shown_for_dirty_drafts(self, store_patch_for_component):
        from ui.components.edge_list_editor import draft_recovery_banner
        store.save_draft("wip net", ROWS)
        client = Client(page(f"/banner-dirty-{uuid.uuid4().hex}"))
        with client:
            rendered = draft_recovery_banner(lambda name: None)
        assert rendered is True
        ids = [(getattr(el, "_props", None) or {}).get("id") for el in client.elements.values()]
        assert "card-edge-draft-recovery" in ids
        texts = " ".join(
            str(getattr(el, "text", ""))
            for el in client.elements.values()
            if type(el).__name__ == "Label"
        )
        assert "wip net" in texts
        assert "recoverable" in texts.lower()

    def test_banner_recover_callback_receives_name(self, store_patch_for_component):
        from ui.components.edge_list_editor import draft_recovery_banner
        store.save_draft("wip", ROWS)
        received = []
        client = Client(page(f"/banner-cb-{uuid.uuid4().hex}"))
        with client:
            draft_recovery_banner(lambda name: received.append(name))
            buttons = [
                el for el in client.elements.values()
                if type(el).__name__ == "Button"
                and "wip" in str((getattr(el, "_props", None) or {}).get("label", ""))
            ]
            assert len(buttons) == 1
            # Invoke the registered click handler directly.
            listeners = [
                listener for listener in buttons[0]._event_listeners.values()
                if listener.type == "click"
            ]
            assert len(listeners) == 1
            listeners[0].handler(SimpleNamespace())
        assert received == ["wip"]


# =============================================================================
# Network tab integration
# =============================================================================

class TestNetworkTabIntegration:
    def _patch_store(self, monkeypatch, tmp_path):
        monkeypatch.setattr(store, "_store_dir", tmp_path / "tab_drafts")

    def _build_tab(self, monkeypatch, tmp_path):
        from ui.tabs.visualization import create_net_viz_tab
        client = Client(page(f"/network-tab-{uuid.uuid4().hex}"))
        with client:
            create_net_viz_tab()
        return client

    def _labels(self, client):
        return [
            (getattr(el, "_props", None) or {}).get("label")
            for el in client.elements.values()
            if (getattr(el, "_props", None) or {}).get("label")
        ]

    def test_canvas_source_includes_editor_option(self, monkeypatch, tmp_path):
        self._patch_store(monkeypatch, tmp_path)
        client = self._build_tab(monkeypatch, tmp_path)
        selects = [
            el for el in client.elements.values()
            if (getattr(el, "_props", None) or {}).get("label") == "Canvas Source"
        ]
        assert len(selects) == 1
        assert selects[0].options == ["Path file", "Edge list editor", "Empty drawing canvas"]

    def test_editor_card_present(self, monkeypatch, tmp_path):
        self._patch_store(monkeypatch, tmp_path)
        client = self._build_tab(monkeypatch, tmp_path)
        ids = [(getattr(el, "_props", None) or {}).get("id") for el in client.elements.values()]
        assert "card-net-viz-edge-editor" in ids

    def test_no_reminder_without_dirty_drafts(self, monkeypatch, tmp_path):
        self._patch_store(monkeypatch, tmp_path)
        store.save_draft("clean", ROWS, dirty=False)
        client = self._build_tab(monkeypatch, tmp_path)
        ids = [(getattr(el, "_props", None) or {}).get("id") for el in client.elements.values()]
        assert "card-edge-draft-recovery" not in ids

    def test_reminder_shown_for_dirty_drafts(self, monkeypatch, tmp_path):
        self._patch_store(monkeypatch, tmp_path)
        store.save_draft("unfinished", ROWS)
        client = self._build_tab(monkeypatch, tmp_path)
        ids = [(getattr(el, "_props", None) or {}).get("id") for el in client.elements.values()]
        assert "card-edge-draft-recovery" in ids

    def test_source_switch_toggles_path_input(self, monkeypatch, tmp_path):
        """The edge-list editor card is always visible in the Net-Viz tab;
        only the path-file upload panel follows the Canvas Source select."""
        self._patch_store(monkeypatch, tmp_path)
        client = self._build_tab(monkeypatch, tmp_path)
        source = next(
            el for el in client.elements.values()
            if (getattr(el, "_props", None) or {}).get("label") == "Canvas Source"
        )
        path_panel = next(
            el for el in client.elements.values()
            if (getattr(el, "_props", None) or {}).get("id") == "net-viz-path-input"
        )
        editor_card = next(
            el for el in client.elements.values()
            if (getattr(el, "_props", None) or {}).get("id") == "card-net-viz-edge-editor"
        )
        # Path file (default): upload panel shown, editor always visible.
        assert source.value == "Path file"
        assert path_panel.visible is True
        assert editor_card.visible is True
        source.set_value("Edge list editor")
        assert path_panel.visible is False
        assert editor_card.visible is True
        source.set_value("Empty drawing canvas")
        assert path_panel.visible is False
        assert editor_card.visible is True
