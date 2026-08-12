"""Tests for the lite custom grouper (component, serializer, gate wiring).

Covers the refined plan contract:
- canonical LabelMapper overall-JSON as the single format (format
  uniformity): the UI exports it, reloads it, and the real LabelMapper
  reads it directly — including validate_datasets for cross-dataset runs;
- rectangular export: every selected dataset becomes a key of every role
  side (empty groups [] included), blank labels auto-name to Group_N;
- reload tolerance: legacy std_label, side precedence, unknown keys;
- selector gate: inline sentinel expands the board, other choices collapse;
- require_names (Cross-Dataset) blocks blank labels; other tabs auto-name;
- run resolution: empty inline board blocks, valid board exports + records
  the group history, presets resolve to their exported file.
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ui.components import custom_grouper as cg  # noqa: E402
from ui.components import mapping_editor as me  # noqa: E402
import ui.group_history as gh  # noqa: E402

DS_A = "male-cns:v0.9"
DS_B = "hemibrain:v1.2.1"


# =============================================================================
# Serializer (pure)
# =============================================================================
class TestCanonicalSerializer:
    def test_auto_names_blank_labels(self):
        rows = [{"name": "", "cells": {DS_A: ["aMe12"]}},
                {"name": "clock", "cells": {DS_A: ["DN1p"]}}]
        data = cg.to_canonical_dict(rows, [DS_A])
        assert data["source_mapping"]["custom_label"] == ["Group_1", "clock"]

    def test_every_dataset_materialized_with_rectangular_groups(self):
        rows = [{"name": "aMe", "cells": {DS_A: ["aMe12"]}}]
        data = cg.to_canonical_dict(rows, [DS_A, DS_B])
        for side in ("source_mapping", "target_mapping"):
            assert data[side][DS_A] == [["aMe12"]]
            # DS_B has no members but MUST exist with one (empty) group per
            # label so LabelMapper.validate_datasets passes.
            assert data[side][DS_B] == [[]]

    def test_format_and_meta_tags(self):
        data = cg.to_canonical_dict(
            [{"name": "g", "cells": {DS_A: ["x"]}}], [DS_A], origin="inline")
        assert data["format"] == cg.FORMAT_TAG
        assert data["groups_meta"]["origin"] == "inline"
        assert data["groups_meta"]["updated_at"]

    def test_source_and_target_roles_identical(self):
        rows = [{"name": "aMe", "cells": {DS_A: ["aMe12", "aMe10"]}}]
        data = cg.to_canonical_dict(rows, [DS_A])
        assert data["source_mapping"] == data["target_mapping"]
        assert "intermediate_mapping" not in data

    def test_roundtrip_identity(self):
        rows = [{"name": "aMe", "cells": {DS_A: ["aMe12", "aMe10"], DS_B: ["aMe12"]}},
                {"name": "", "cells": {DS_A: [], DS_B: ["DN1p"]}}]
        data = cg.to_canonical_dict(rows, [DS_A, DS_B])
        back = cg.from_canonical_dict(data)
        assert [r["name"] for r in back] == ["aMe", "Group_2"]
        assert back[0]["cells"][DS_A] == ["aMe12", "aMe10"]
        assert back[0]["cells"][DS_B] == ["aMe12"]
        assert back[1]["cells"][DS_A] == []
        assert back[1]["cells"][DS_B] == ["DN1p"]

    def test_reload_legacy_std_label_and_side_precedence(self):
        legacy = {
            "source_mapping": {"std_label": ["grp"], DS_A: [["aMe12"]]},
        }
        rows = cg.from_canonical_dict(legacy)
        assert rows == [{"name": "grp", "cells": {DS_A: ["aMe12"]}}]
        # source wins when several sides exist
        both = {
            "source_mapping": {"custom_label": ["s"], DS_A: [["a"]]},
            "target_mapping": {"custom_label": ["t"], DS_A: [["b"]]},
        }
        assert cg.from_canonical_dict(both)[0]["name"] == "s"

    def test_reload_rejects_unusable_payload(self):
        with pytest.raises(ValueError):
            cg.from_canonical_dict({"nothing": 1})
        with pytest.raises(ValueError):
            cg.from_canonical_dict({"source_mapping": {DS_A: [["x"]]}})


# =============================================================================
# Format uniformity: the exported file loads through the REAL LabelMapper
# =============================================================================
class TestLabelMapperEquivalence:
    def test_exported_file_loads_directly(self, tmp_path):
        rows = [{"name": "aMe", "cells": {DS_A: ["aMe12", "aMe10"], DS_B: ["aMe12"]}},
                {"name": "clock", "cells": {DS_A: ["DN1p"], DS_B: []}}]
        path = tmp_path / "inline.json"
        path.write_text(json.dumps(cg.to_canonical_dict(rows, [DS_A, DS_B])),
                        encoding="utf-8")

        from comparison.label_mapper import LabelMapper
        mapper = LabelMapper(overall_mapping_json=str(path))
        for role in ("source", "target"):
            assert sorted(mapper.get_neurons_for_label("aMe", DS_A, role)) == \
                ["aMe10", "aMe12"]
            assert mapper.get_neurons_for_label("aMe", DS_B, role) == ["aMe12"]
            assert mapper.get_neurons_for_label("clock", DS_A, role) == ["DN1p"]
            assert mapper.get_neurons_for_label("clock", DS_B, role) == []
        # Cross-dataset guard: both datasets present in both roles.
        mapper.validate_datasets([DS_A, DS_B], role="both")

    def test_auto_named_file_equivalent_to_named_file(self, tmp_path):
        """Blank labels auto-name; structurally identical to named presets."""
        rows = [{"name": "", "cells": {DS_A: ["aMe12"]}}]
        auto = cg.to_canonical_dict(rows, [DS_A])
        named = cg.to_canonical_dict(
            [{"name": "Group_1", "cells": {DS_A: ["aMe12"]}}], [DS_A])
        assert auto["source_mapping"] == named["source_mapping"]
        from comparison.label_mapper import LabelMapper
        m_auto = LabelMapper(overall_mapping_json=None,
                             source_mapping_dict={DS_A: [["aMe12"]]},
                             source_labels=["Group_1"])
        path = tmp_path / "auto.json"
        path.write_text(json.dumps(auto), encoding="utf-8")
        m_file = LabelMapper(overall_mapping_json=str(path))
        assert (m_file.get_neurons_for_label("Group_1", DS_A, "source") ==
                m_auto.get_neurons_for_label("Group_1", DS_A, "source") ==
                ["aMe12"])


# =============================================================================
# Handle (JS-free state)
# =============================================================================
class TestLiteGroupHandle:
    def test_add_remove_move(self):
        h = cg.LiteGroupHandle()
        h.add_row("a", {DS_A: ["1"]})
        h.add_row("b", {DS_A: ["2"]})
        h.add_row("c", {DS_A: ["3"]})
        h.move_row(0, 1)
        assert [r["name"] for r in h.rows] == ["b", "a", "c"]
        h.remove_row(1)
        assert [r["name"] for r in h.rows] == ["b", "c"]

    def test_upsert_covers_same_label(self):
        h = cg.LiteGroupHandle()
        h.add_row("aMe", {DS_A: ["aMe12"]})
        idx = h.upsert_row("aMe", {DS_A: ["aMe9"], DS_B: ["aMe12_R"]})
        assert idx == 0 and len(h.rows) == 1
        assert h.rows[0]["cells"][DS_A] == ["aMe9"]
        assert h.rows[0]["cells"][DS_B] == ["aMe12_R"]

    def test_is_empty(self):
        h = cg.LiteGroupHandle()
        assert h.is_empty()
        h.add_row("x", {DS_A: []})
        assert h.is_empty()
        h.add_row("y", {DS_A: ["1"]})
        assert not h.is_empty()


# =============================================================================
# UI wiring (structural harness tests)
# =============================================================================
@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Redirect history file and inline export dir into tmp."""
    monkeypatch.setattr(gh, "HISTORY_PATH", tmp_path / "group_history.json")
    inline_dir = tmp_path / "_inline"
    monkeypatch.setattr(me, "_INLINE_DIR", inline_dir)
    return tmp_path


class TestSelectorGate:
    def test_button_opens_panel_and_label_mirrors_state(self, isolated_store):
        from nicegui import Client
        from nicegui.page import page
        client = Client(page("/grouper-gate"))
        with client:
            button, dialog, _resolve = me.custom_grouping_block(
                tab_key="gate", datasets_provider=lambda: [DS_A])
        assert dialog.value is False
        assert "none" in button.text
        # The panel hosts the preset selector and the board.
        assert dialog.preset_select.value == me.NONE_MAPPING
        assert dialog.inline_grouper is not None
        with client:
            dialog.open()
        assert dialog.value is True

    def test_button_label_tracks_inline_groups(self, isolated_store):
        from nicegui import Client
        from nicegui.page import page
        client = Client(page("/grouper-gate2"))
        with client:
            button, dialog, _resolve = me.custom_grouping_block(
                tab_key="gate2", datasets_provider=lambda: [DS_A])
        grouper = dialog.inline_grouper
        grouper.handle.add_row("aMe", {DS_A: ["aMe12"]})
        with client:
            dialog.open()
            dialog.close()  # closing refreshes the button label
        assert "inline · 1 group(s)" in button.text


class TestRunResolution:
    def _build(self, page_path, require_names=False):
        from nicegui import Client
        from nicegui.page import page
        client = Client(page(page_path))
        with client:
            selector, card, resolve = me.custom_grouping_block(
                tab_key=f"res_{page_path.strip('/').replace('-', '_')}",
                datasets_provider=lambda: [DS_A, DS_B],
                require_names=require_names)
        return client, selector, card, resolve

    def test_none_selection_resolves_no_mapping(self, isolated_store):
        client, selector, _card, resolve = self._build("/res-none")
        with client:
            path, ok = resolve()
        assert ok is True and path is None

    def test_empty_board_resolves_to_no_mapping(self, isolated_store):
        """No preset + empty board = simply no mapping (no run blocking)."""
        client, selector, _card, resolve = self._build("/res-empty")
        with client:
            path, ok = resolve()
        assert ok is True and path is None

    def test_inline_valid_board_exports_and_records_history(self, isolated_store):
        client, selector, card, resolve = self._build("/res-valid")
        grouper = card.inline_grouper
        grouper.handle.add_row("aMe", {DS_A: ["aMe12"], DS_B: ["aMe12"]})
        grouper.handle.add_row("", {DS_A: ["DN1p"]})  # auto-named
        grouper.resync()
        with client:
            path, ok = resolve()
        assert ok is True and path and Path(path).exists()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["source_mapping"]["custom_label"] == ["aMe", "Group_2"]
        assert data["source_mapping"][DS_B][1] == []  # padded empty group
        # History: named label recorded, auto label skipped.
        assert gh.list_recent() == ["aMe"]
        assert gh.get_label("aMe")["members"][DS_A] == ["aMe12"]
        # Recent same-tab exports are KEPT (an in-flight run may still read
        # them); only files older than 24 h are swept on export.
        with client:
            path2, ok2 = resolve()
        assert ok2 and path2 != path
        exports = set(Path(path).parent.glob("*.json"))
        assert {Path(path), Path(path2)} <= exports
        import os, time
        old = Path(path)
        past = time.time() - 25 * 3600
        os.utime(old, (past, past))
        with client:
            path3, ok3 = resolve()
        assert ok3 and not old.exists(), "stale export must be swept"
        assert Path(path2).exists() and Path(path3).exists()

    def test_preset_wins_over_inline_board(self, isolated_store, monkeypatch):
        from ui import mapping_store as ms
        store_dir = isolated_store / "store"
        monkeypatch.setattr(ms, "_store_dir", store_dir)
        monkeypatch.setattr(ms, "_store_file", store_dir / "user_mappings.json")
        ms.save_mapping("ortho", {
            "source_mapping": {"custom_label": ["g"], DS_A: [["aMe12"]]},
            "target_mapping": {"custom_label": ["g"], DS_A: [["aMe12"]]},
        })
        client, _button, dialog, resolve = self._build("/res-preset")
        grouper = dialog.inline_grouper
        grouper.handle.add_row("inlineG", {DS_A: ["x1"]})
        dialog.preset_select.value = "ortho"
        with client:
            path, ok = resolve()
        assert ok is True
        assert path == ms.mapping_file_path("ortho")
        # No inline export happened while a preset is selected.
        assert list((isolated_store / "_inline").glob("res_preset_*.json")) == []

    def test_require_names_blocks_blank_labels(self, isolated_store):
        client, selector, card, resolve = self._build("/res-names", require_names=True)
        grouper = card.inline_grouper
        grouper.handle.add_row("", {DS_A: ["aMe12"]})
        grouper.resync()
        with client:
            path, ok = resolve()
        assert ok is False and path is None
        # Naming the group makes it pass (the name input is the source of
        # truth; resync collects from the widgets).
        grouper._name_inputs[0].value = "aMe"
        with client:
            path, ok = resolve()
        assert ok is True and path

    def test_dataset_columns_follow_watched_selector(self, isolated_store):
        from nicegui import Client
        from nicegui.page import page
        from nicegui import ui as _ui
        client = Client(page("/grouper-sync"))
        with client:
            ds_sel = _ui.select(options=[DS_A, DS_B], value=DS_A)
            _selector, card, _resolve = me.custom_grouping_block(
                tab_key="sync",
                datasets_provider=lambda: [ds_sel.value],
                watch_elements=[ds_sel])
        grouper = card.inline_grouper
        grouper.handle.add_row("aMe", {DS_A: ["aMe12"]})
        grouper.resync()
        assert list(grouper._cell_widgets[0].keys()) == [DS_A]
        ds_sel.value = DS_B  # watch fires -> resync re-renders columns
        assert grouper.datasets() == [DS_B]
        assert list(grouper._cell_widgets[0].keys()) == [DS_B]
        # Members of the hidden dataset survive the column change.
        assert grouper.handle.rows[0]["cells"][DS_A] == ["aMe12"]


class TestReloadIntoBoard:
    def test_load_replaces_rows_from_canonical_payload(self, isolated_store):
        from nicegui import Client
        from nicegui.page import page
        client = Client(page("/grouper-reload"))
        with client:
            _selector, card, _resolve = me.custom_grouping_block(
                tab_key="reload", datasets_provider=lambda: [DS_A, DS_B])
        grouper = card.inline_grouper
        payload = cg.to_canonical_dict(
            [{"name": "aMe", "cells": {DS_A: ["aMe12"], DS_B: ["aMe12"]}}],
            [DS_A, DS_B])
        grouper.handle.replace_rows(cg.from_canonical_dict(payload))
        grouper.resync()
        assert grouper.handle.rows[0]["name"] == "aMe"
        assert grouper.handle.rows[0]["cells"][DS_B] == ["aMe12"]
        # Export from the reloaded board reproduces the payload's sides.
        out = grouper.to_canonical()
        assert out["source_mapping"] == payload["source_mapping"]


# =============================================================================
# Workflow: pushing group members into query inputs
# =============================================================================
class TestQueryPushWorkflow:
    def test_push_to_query_adds_union_of_members(self, isolated_store):
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        client = Client(page("/grouper-push"))
        with client:
            source = neuron_list_input(label="Source Neurons",
                                       show_filter=False, show_upload=False)
            target = neuron_list_input(label="Target Neurons",
                                       show_filter=False, show_upload=False)
            _selector, card, _resolve = me.custom_grouping_block(
                tab_key="push",
                datasets_provider=lambda: [DS_A, DS_B],
                query_inputs={"source": source, "target": target})
        grouper = card.inline_grouper
        grouper.handle.add_row("aMe", {DS_A: ["aMe12", "aMe10"],
                                        DS_B: ["aMe12", "aMe9"]})
        grouper.resync()
        with client:
            pushed = grouper.push_to_query("source", 0)
        # Union across dataset cells, deduplicated, order preserved.
        assert pushed == ["aMe12", "aMe10", "aMe9"]
        assert source.get_value()[1] == ["aMe12", "aMe10", "aMe9"]
        assert target.get_value()[1] == []

    def test_push_twice_does_not_duplicate(self, isolated_store):
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        client = Client(page("/grouper-push2"))
        with client:
            source = neuron_list_input(label="Source Neurons",
                                       show_filter=False, show_upload=False)
            _selector, card, _resolve = me.custom_grouping_block(
                tab_key="push2",
                datasets_provider=lambda: [DS_A],
                query_inputs={"source": source})
        grouper = card.inline_grouper
        grouper.handle.add_row("aMe", {DS_A: ["aMe12"]})
        grouper.resync()
        with client:
            grouper.push_to_query("source", 0)
            grouper.push_to_query("source", 0)
        assert source.get_value()[1] == ["aMe12"]

    def test_push_empty_group_warns_and_adds_nothing(self, isolated_store):
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        client = Client(page("/grouper-push3"))
        with client:
            source = neuron_list_input(label="Source Neurons",
                                       show_filter=False, show_upload=False)
            _selector, card, _resolve = me.custom_grouping_block(
                tab_key="push3",
                datasets_provider=lambda: [DS_A],
                query_inputs={"source": source})
        grouper = card.inline_grouper
        grouper.handle.add_row("empty", {DS_A: []})
        grouper.resync()
        with client:
            assert grouper.push_to_query("source", 0) == []
        assert source.get_value()[1] == []


class TestSuggestionAnchoring:
    def test_suggest_menu_targets_its_input_wrapper(self, isolated_store):
        """The custom suggestion menu must carry an explicit target selector
        for its own input wrapper (robust anchoring inside nested rebuilds)."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        client = Client(page("/grouper-anchor"))
        with client:
            box = neuron_list_input(
                label="Source Neurons", show_filter=False, show_upload=False,
                suggestions=lambda text: [("aMe12", "type")])
        menu = box.suggest_menu
        anchor = box.chip_input.parent_slot.parent
        assert menu._props.get("target") == f"#{anchor.html_id}"

    def test_grouper_cell_menus_are_anchored(self, isolated_store):
        from nicegui import Client
        from nicegui.page import page

        client = Client(page("/grouper-anchor2"))
        with client:
            _selector, card, _resolve = me.custom_grouping_block(
                tab_key="anchor2", datasets_provider=lambda: [DS_A])
        grouper = card.inline_grouper
        grouper.handle.add_row("aMe", {DS_A: ["aMe12"]})
        grouper.resync()
        cell = grouper._cell_widgets[0][DS_A]
        menu = cell.suggest_menu
        anchor = cell.chip_input.parent_slot.parent
        assert menu._props.get("target") == f"#{anchor.html_id}"
