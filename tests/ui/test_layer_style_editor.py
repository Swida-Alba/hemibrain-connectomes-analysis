"""Tests for the layer-style editor store + component (Skeleton advanced table).

Covers:
- ui/layer_style_store.py: draft CRUD, CSV layout, validation, and upload
  parsing (including unquoted CSS color functions).
- ui/components/layer_style_editor.py: editor state, add/delete/inline edit,
  auto-save flush, CSV export/upload.
- ui/tabs/visualization.py: Skeleton Layer Editor dropdown wiring.
"""
import re
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import ui.layer_style_store as store
from ui.layer_style_store import LAYER_STYLE_COLUMNS


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Point the draft store at a temp directory for every test."""
    monkeypatch.setattr(store, "_store_dir", tmp_path / "layer_style_drafts")
    yield


ROWS = [
    {"layer": "0", "neuron": "aMe12", "color": "#ff0000"},
    {"layer": "0", "neuron": "aMe13", "color": "rgb(0,255,0)"},
    {"layer": "1", "neuron": "dn1", "color": "rgba(0,0,255,0.5)"},
]


def meta(name):
    return store.get_meta(name)


class TestStoreValidation:
    def test_normalize_fills_all_columns(self):
        assert store.normalize_rows([{"layer": "0", "neuron": " aMe12 "}]) == [{
            "layer": "0", "neuron": "aMe12", "color": "",
            "synapse_color": "", "pre_synaptic_color": "", "post_synaptic_color": "",
        }]

    def test_valid_rows(self):
        assert store.validate_rows(ROWS) == []

    def test_missing_layer_or_neuron_reported(self):
        errors = store.validate_rows([{"layer": "", "neuron": "aMe12"}])
        assert any("missing layer" in e for e in errors)
        errors = store.validate_rows([{"layer": "0", "neuron": ""}])
        assert any("missing neuron" in e for e in errors)

    def test_non_numeric_layer_reported(self):
        errors = store.validate_rows([{"layer": "x", "neuron": "aMe12"}])
        assert any("not a number" in e for e in errors)

    def test_complete_rows(self):
        # All rows carry layer + neuron, so all are complete (normalized).
        assert store.complete_rows(ROWS) == store.normalize_rows(ROWS)
        # A row missing a neuron is dropped.
        partial = ROWS + [{"layer": "2", "neuron": ""}]
        assert len(store.complete_rows(partial)) == len(ROWS)

    def test_empty_rows_ignored(self):
        assert store.validate_rows([{}, {"layer": "", "neuron": ""}]) == []

    def test_layers_must_be_continuous(self):
        errors = store.validate_rows([
            {"layer": "1", "neuron": "a"},
            {"layer": "3", "neuron": "b"},
        ])
        assert any("missing layer(s) 2" in error for error in errors)
        assert store.validate_rows([
            {"layer": "1", "neuron": "a"},
            {"layer": "2", "neuron": "b"},
        ]) == []


class TestStoreSaveLoad:
    def test_round_trip(self):
        slug = store.save_draft("layers", ROWS)
        assert slug == "layers"
        assert store.load_draft("layers") == store.normalize_rows(ROWS)

    def test_csv_header_has_all_columns(self):
        store.save_draft("layers", ROWS)
        header = Path(store.draft_csv_path("layers")).read_text(encoding="utf-8").splitlines()[0]
        assert header == ",".join(LAYER_STYLE_COLUMNS)

    def test_meta_dirty(self):
        store.save_draft("layers", ROWS)
        m = meta("layers")
        assert m["dirty"] is True
        assert m["row_count"] == 3

    def test_quoted_color_round_trips(self):
        store.save_draft("q", [{"layer": "0", "neuron": "n", "color": "rgba(1,2,3,0.5)"}])
        loaded = store.load_draft("q")
        assert loaded[0]["color"] == "rgba(1,2,3,0.5)"

    def test_load_rows_from_csv_text_unquoted_color(self):
        text = "layer,neuron,color\n0,aMe12,rgba(74,144,226,0.3)\n"
        rows = store.load_rows_from_csv_text(text)
        assert rows[0]["color"] == "rgba(74,144,226,0.3)"

    def test_delete_draft(self):
        store.save_draft("layers", ROWS)
        assert store.delete_draft("layers") is True
        assert store.load_draft("layers") is None


class TestModeColumnsAndLayers:
    def test_available_layers_contiguous_from_1(self):
        assert store.available_layers([]) == [1]
        assert store.available_layers([{"layer": "1"}]) == [1, 2]
        assert store.available_layers([{"layer": "1"}, {"layer": "2"}]) == [1, 2, 3]
        # If no row uses layer 1, only 1 is offered (never a higher number).
        assert store.available_layers([{"layer": "2"}]) == [1]

    def test_next_layer_number_starts_at_one_then_uses_maximum(self):
        assert store.next_layer_number([]) == 1
        assert store.next_layer_number([{"layer": "1", "neuron": "a"}]) == 2
        assert store.next_layer_number([
            {"layer": "1", "neuron": "a"},
            {"layer": "3", "neuron": "b"},
        ]) == 4

    def test_mode_columns(self):
        assert store.mode_columns("synapse") == ("layer", "neuron", "color", "synapse_color")
        assert store.mode_columns("pre-post sites") == (
            "layer", "neuron", "color", "pre_synaptic_color", "post_synaptic_color"
        )

    def test_rows_to_csv_for_mode(self):
        syn = store.rows_to_csv_for_mode(ROWS, "synapse")
        assert syn.splitlines()[0] == "layer,neuron,color,synapse_color"
        pp = store.rows_to_csv_for_mode(ROWS, "pre-post sites")
        assert pp.splitlines()[0] == "layer,neuron,color,pre_synaptic_color,post_synaptic_color"


# =============================================================================
# UI component: editor handle behavior
# =============================================================================

from nicegui import Client
from nicegui.page import page


@pytest.fixture()
def store_patch_for_component(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "_store_dir", tmp_path / "comp_drafts")
    return tmp_path / "comp_drafts"


def build_editor(store_dir, export_dir=None):
    from ui.components.layer_style_editor import layer_style_editor
    client = Client(page(f"/layer-style-editor-{uuid.uuid4().hex}"))
    with client:
        handle = layer_style_editor(
            export_dir_provider=lambda: str(export_dir) if export_dir else None
        )
    return client, handle


class TestEditorHandle:
    def test_card_elements_exist(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        ids = [
            (getattr(el, "_props", None) or {}).get("id")
            for el in client.elements.values()
        ]
        assert "card-skeleton-layer-style-editor" in ids
        assert handle.table is not None and handle.status_label is not None
        assert set(handle.edit_inputs) == set(LAYER_STYLE_COLUMNS)

    def test_add_neuron_control_omits_counter_and_clear_action(
        self, store_patch_for_component
    ):
        client, _handle = build_editor(store_patch_for_component)
        assert not [
            element for element in client.elements.values()
            if getattr(element, "text", "") == "Clear"
        ]
        assert not [
            element for element in client.elements.values()
            if type(element).__name__ == "Badge"
            and "neuron" in str(getattr(element, "text", "")).lower()
        ]

    def test_add_row_uses_current_editor_values_and_autosaves(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.name_input.value = "my layers"
        for key, value in {
            "layer": "0", "neuron": "aMe12", "color": "#123456",
            "synapse_color": "rgba(1,2,3,0.3)", "pre_synaptic_color": "",
            "post_synaptic_color": "",
        }.items():
            handle.edit_inputs[key].value = value
        handle.add_row()
        csv_path = handle.flush_autosave()
        assert csv_path and Path(csv_path).exists()
        rows = store.load_draft("my layers")
        assert rows[0]["neuron"] == "aMe12"
        assert rows[0]["color"] == "#123456"
        assert meta("my layers")["dirty"] is True

    def test_inline_edit_updates_row(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(ROWS)
        handle.on_inline_edit(
            SimpleNamespace(args={"id": 1, "field": "neuron", "value": " aMe13 "})
        )
        assert handle.rows[1]["neuron"] == "aMe13"

    def test_inline_text_edit_does_not_rebuild_table(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(ROWS)
        table_updates = []
        handle.table.update = lambda: table_updates.append(True)
        handle.on_inline_edit(
            SimpleNamespace(args={"id": 1, "field": "neuron", "value": "aMe13x"})
        )
        assert handle.rows[1]["neuron"] == "aMe13x"
        assert table_updates == []

    def test_available_neurons_append_one_row_per_entry_to_one_batch_layer(
        self, store_patch_for_component
    ):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows([{"layer": "1", "neuron": "existing"}])
        handle.begin_available_batch()
        assert handle.apply_available_neurons(["n1", "n2"]) == 2
        assert handle.available_query_values() == ["n1", "n2"]
        assert [(row["layer"], row["neuron"]) for row in handle.rows] == [
            ("1", "existing"), ("2", "n1"), ("2", "n2")
        ]
        # The viewer reports its complete selection after every toggle; only
        # the new entry is appended, and it stays on the same batch layer.
        assert handle.apply_available_neurons(["n1", "n2", "n3"]) == 1
        assert handle.rows[-1]["layer"] == "2"
        handle.begin_available_batch()
        assert handle.available_query_values() == []
        assert handle.apply_available_neurons(["n4"]) == 1
        assert handle.rows[-1]["layer"] == "3"

    def test_empty_available_neuron_table_starts_batch_at_layer_one(
        self, store_patch_for_component
    ):
        client, handle = build_editor(store_patch_for_component)
        assert handle.apply_available_neurons(["n1", "n2"]) == 2
        assert {row["layer"] for row in handle.rows} == {"1"}

    def test_validation_panel_reports_and_clears_layer_gap(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows([
            {"layer": "1", "neuron": "a"},
            {"layer": "3", "neuron": "b"},
        ])
        assert handle.validation_panel.visible is True
        assert "missing layer(s) 2" in handle.validation_label.text
        handle.on_inline_edit(
            SimpleNamespace(args={"id": 1, "field": "layer", "value": "2"})
        )
        assert handle.validation_panel.visible is False

    def test_mode_switch_hides_picker_wrapper_with_input(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_synapse_mode("pre-post sites")
        assert handle._color_control_groups["synapse_color"].visible is False
        assert handle._color_control_groups["pre_synaptic_color"].visible is True
        handle.set_synapse_mode("synapse")
        assert handle._color_control_groups["synapse_color"].visible is True
        assert handle._color_control_groups["pre_synaptic_color"].visible is False

    def test_advanced_neuron_input_uses_dataset_suggestions(
        self, store_patch_for_component, monkeypatch
    ):
        from ui.components import layer_style_editor as editor_module

        calls = []
        monkeypatch.setattr(
            editor_module,
            "dataset_suggestions",
            lambda text, dataset, columns, limit=None: calls.append(
                (text, dataset, columns, limit)
            ) or [("aMe12", "type")],
        )
        client = Client(page(f"/layer-style-suggest-{uuid.uuid4().hex}"))
        with client:
            handle = editor_module.layer_style_editor(
                dataset_provider=lambda: "male-cns:v1.0",
                search_columns_provider=lambda: "type",
            )
        assert handle._suggest_neurons("aMe") == [("aMe12", "type")]
        assert calls == [("aMe", "male-cns:v1.0", "type", None)]

    def test_delete_selected_rows(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.name_input.value = "del"
        handle.set_rows(ROWS + [{"layer": "2", "neuron": "X"}])
        handle._selected_ids = [1]
        handle.delete_selected()
        assert len(handle.rows) == 3
        assert handle.rows[1]["neuron"] == "dn1"

    def test_add_row_color_inputs_have_picker_buttons(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        palette_buttons = [
            el for el in client.elements.values()
            if (getattr(el, "_props", None) or {}).get("icon") == "palette"
        ]
        # One picker button per colour field (Color, Synapse, Pre, Post).
        assert len(palette_buttons) == 4

    def test_pending_color_picker_becomes_a_fixed_square_preview(
        self, store_patch_for_component
    ):
        client, handle = build_editor(store_patch_for_component)
        button = handle._add_color_pickers["color"]
        assert button.icon == "palette"
        assert "round" in button._props

        handle._open_picker_for_add("color")
        handle._apply_picked_color("#2ca02c")

        assert button.icon is None
        assert "round" not in button._props
        assert button.style["background-color"] == "#2ca02c"
        assert "drocat-layer-add-color-picker-set" in button.classes

    def test_apply_picked_color_to_form_input(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle._open_picker_for_add("color")
        assert handle._pending_pick == {"field": "color"}
        handle._apply_picked_color("rgba(1, 2, 3, 0.4)")
        assert handle.edit_inputs["color"].value == "rgba(1, 2, 3, 0.4)"

    def test_set_synapse_mode_changes_columns(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        assert [c["name"] for c in handle.table.columns] == [
            "layer", "neuron", "color", "synapse_color"
        ]
        handle.set_synapse_mode("pre-post sites")
        assert [c["name"] for c in handle.table.columns] == [
            "layer", "neuron", "color", "pre_synaptic_color", "post_synaptic_color"
        ]
        # Synapse colour input hidden in pre-post mode and vice versa.
        assert handle.edit_inputs["synapse_color"].visible is False
        assert handle.edit_inputs["pre_synaptic_color"].visible is True
        handle.set_synapse_mode("synapse")
        assert handle.edit_inputs["synapse_color"].visible is True
        assert handle.edit_inputs["pre_synaptic_color"].visible is False

    def test_add_row_layer_is_selection_box(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        layer_el = handle.edit_inputs["layer"]
        assert layer_el is handle._layer_add_select
        assert handle._layer_add_select.options == ["1"]
        handle.set_rows([{"layer": "1", "neuron": "a"}], name="x")
        assert handle._layer_add_select.options == ["1", "2"]
        assert [item["label"] for item in handle._layer_add_select._props["options"]] == [
            "1", "2"
        ]

    def test_table_row_dicts_carry_layer_opts(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(
            [{"layer": "1", "neuron": "a"}, {"layer": "2", "neuron": "b"}], name="x"
        )
        row_dicts = handle._row_dicts()
        assert row_dicts[0]["layer_opts"] == ["1", "2", "3"]
        assert row_dicts[1]["layer_opts"] == ["1", "2", "3"]

    def test_load_csv_text_into_table(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        ok = handle.load_csv_text("layer,neuron,color\n0,aMe12,rgba(74,144,226,0.3)\n")
        assert ok is True
        assert len(handle.rows) == 1
        assert handle.rows[0]["color"] == "rgba(74,144,226,0.3)"
        assert "Loaded 1 rows" in handle.status_label.text

    def test_runnable_csv_requires_layer_and_neuron(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows([{"layer": "0", "neuron": ""}], name="incomplete")
        assert handle.runnable_csv_path() is None
        handle.set_rows(ROWS, name="complete")
        assert handle.runnable_csv_path() is not None

    def test_runnable_csv_without_name_uses_transient(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(ROWS, name="")
        path = handle.runnable_csv_path()
        try:
            assert path == handle.transient_csv_path
            assert path and Path(path).exists()
            content = Path(path).read_text(encoding="utf-8")
            # The transient CSV uses the synapse-mode columns (default 'synapse').
            assert content.startswith(",".join(store.mode_columns("synapse")) + "\n")
            assert store.list_drafts() == []
        finally:
            handle.cleanup_transient_csv()
        assert not Path(path).exists()

    def test_export_downloads_without_draft_name(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(ROWS, name="")
        downloads = []
        handle.table.client.download = lambda src, filename, media_type: downloads.append(
            (src, filename, media_type)
        )
        exported = handle.export_csv()
        assert re.fullmatch(r"layers_\d{8}_\d{6}\.csv", exported)
        assert len(downloads) == 1
        content, filename, media_type = downloads[0]
        assert content.decode("utf-8").startswith(",".join(store.mode_columns("synapse")) + "\n")
        assert media_type == "text/csv"
        assert store.list_drafts() == []


# =============================================================================
# Skeleton tab integration
# =============================================================================

class TestSkeletonTabIntegration:
    def _patch_store(self, monkeypatch, tmp_path):
        monkeypatch.setattr(store, "_store_dir", tmp_path / "tab_drafts")

    def _build_tab(self, monkeypatch, tmp_path):
        from ui.tabs.visualization import create_skeleton_tab
        client = Client(page(f"/skeleton-tab-{uuid.uuid4().hex}"))
        with client:
            create_skeleton_tab()
        return client

    def _by_label(self, client, label):
        matches = [
            el for el in client.elements.values()
            if (getattr(el, "_props", None) or {}).get("label") == label
        ]
        assert len(matches) == 1, f"label {label!r}: {len(matches)}"
        return matches[0]

    def _by_id(self, client, card_id):
        return next(
            el for el in client.elements.values()
            if (getattr(el, "_props", None) or {}).get("id") == card_id
        )

    def _mode_button(self, client, text):
        matches = [el for el in client.elements.values() if getattr(el, "text", None) == text]
        assert len(matches) == 1, f"button {text!r}: {len(matches)}"
        return matches[0]

    @staticmethod
    def _click_button(button):
        for listener in (getattr(button, "_event_listeners", None) or {}).values():
            if getattr(listener, "type", None) == "click":
                listener.handler(SimpleNamespace())

    def test_layer_editor_buttons_exist(self, monkeypatch, tmp_path):
        self._patch_store(monkeypatch, tmp_path)
        client = self._build_tab(monkeypatch, tmp_path)
        for name in ("Standard", "Advanced", "File upload"):
            self._mode_button(client, name)

    def test_layer_editor_toggles_panels(self, monkeypatch, tmp_path):
        self._patch_store(monkeypatch, tmp_path)
        client = self._build_tab(monkeypatch, tmp_path)
        standard = self._by_id(client, "card-skeleton-layers")
        advanced = self._by_id(client, "card-skeleton-layer-style")
        file_upload = self._by_id(client, "card-skeleton-layer-upload")
        assert standard.visible is True
        assert advanced.visible is False
        assert file_upload.visible is False
        self._click_button(self._mode_button(client, "Advanced"))
        assert standard.visible is False
        assert advanced.visible is True
        assert file_upload.visible is False
        self._click_button(self._mode_button(client, "File upload"))
        assert standard.visible is False
        assert advanced.visible is False
        assert file_upload.visible is True
        self._click_button(self._mode_button(client, "Standard"))
        assert standard.visible is True
        assert advanced.visible is False
        assert file_upload.visible is False

    def test_color_editor_panels_only_in_standard_mode(self, monkeypatch, tmp_path):
        """The three palette-editor containers are visible only in Standard mode."""
        self._patch_store(monkeypatch, tmp_path)
        client = self._build_tab(monkeypatch, tmp_path)
        for palette_id in (
            "card-skeleton-neuron-palette",
            "card-skeleton-synapse-palette",
            "card-skeleton-roi-palette",
        ):
            palette = self._by_id(client, palette_id)
            assert palette.visible is True
            self._click_button(self._mode_button(client, "Advanced"))
            assert palette.visible is False
            self._click_button(self._mode_button(client, "File upload"))
            assert palette.visible is False
            self._click_button(self._mode_button(client, "Standard"))
            assert palette.visible is True

    def test_synapse_view_mode_options(self, monkeypatch, tmp_path):
        self._patch_store(monkeypatch, tmp_path)
        client = self._build_tab(monkeypatch, tmp_path)
        view = self._by_label(client, "Synapse Mode")
        assert view.options == ["synapse", "pre-post sites", "skip"]
        assert view.value == "synapse"
        shape = self._by_label(client, "Synapse Shape")
        assert "cone" in shape.options and "pre_post" not in shape.options
        pre_post_shape = self._by_label(client, "Pre/post shape")
        assert len(pre_post_shape.options) == 2
        # The threshold is renamed from "Min Synapse Count"; the old label is gone.
        self._by_label(client, "Synapse Threshold")
        labels = [(getattr(el, "_props", None) or {}).get("label") for el in client.elements.values()]
        assert "Min Synapse Count" not in labels

    def test_shape_defaults_follow_skeleton_mode(self, monkeypatch, tmp_path):
        self._patch_store(monkeypatch, tmp_path)
        client = self._build_tab(monkeypatch, tmp_path)
        mode = self._by_label(client, "Skeleton Mode")
        synapse_shape = self._by_label(client, "Synapse Shape")
        pre_post_shape = self._by_label(client, "Pre/post shape")

        assert mode.value == "tube"
        assert synapse_shape.value == "cone"
        assert pre_post_shape.value == "solid (spheres + cones)"

        mode.set_value("line")
        assert synapse_shape.value == "scatter"
        assert pre_post_shape.value == "scatter (circles + squares)"

        mode.set_value("tube")
        assert synapse_shape.value == "cone"
        assert pre_post_shape.value == "solid (spheres + cones)"

    def test_synapse_view_mode_toggles_shapes_and_warning(self, monkeypatch, tmp_path):
        self._patch_store(monkeypatch, tmp_path)
        client = self._build_tab(monkeypatch, tmp_path)
        view = self._by_label(client, "Synapse Mode")
        shape = self._by_label(client, "Synapse Shape")
        pre_post_shape = self._by_label(client, "Pre/post shape")
        warning = self._by_id(client, "card-skeleton-pre-post-warning")
        # Default: synapse mode -> shape visible, pre/post shape + warning hidden.
        assert shape.visible is True
        assert pre_post_shape.visible is False
        assert warning.visible is False
        view.set_value("pre-post sites")
        assert shape.visible is False
        assert pre_post_shape.visible is True
        assert warning.visible is True
        view.set_value("skip")
        assert shape.visible is False
        assert pre_post_shape.visible is False
        assert warning.visible is False
