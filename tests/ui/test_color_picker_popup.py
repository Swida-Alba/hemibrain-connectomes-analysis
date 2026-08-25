"""Tests for the reusable single-color picker popup (alpha + Bokeh palette)."""

import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

from nicegui import Client
from nicegui.page import page

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ui.components.color_picker_popup import color_picker_popup


def build_popup(initial="#145cff"):
    client = Client(page(f"/color-picker-{uuid.uuid4().hex}"))
    with client:
        handle = color_picker_popup(value=initial)
    return client, handle


class TestColorPickerPopup:
    def test_value_is_raw_when_alpha_is_one(self):
        _client, handle = build_popup()
        handle._current = "#145cff"
        handle.alpha.value = 1.0
        assert handle.get_value() == "#145cff"

    def test_value_embeds_alpha_when_below_one(self):
        _client, handle = build_popup()
        handle._current = "#145cff"
        handle.alpha.value = 0.5
        # Alpha is opt-in so the default remains compatible with global
        # visualization opacity.
        assert handle.get_value() == "#145cff"
        handle.apply_alpha.value = True
        assert handle.get_value() == "rgba(20, 92, 255, 0.5)"

    def test_open_seeds_from_initial_rgba(self):
        _client, handle = build_popup()
        handle.open("rgba(10, 20, 30, 0.4)")
        assert handle._current == "#0a141e"
        assert handle.alpha.value == 0.4
        assert handle.alpha_slider.value == 0.4
        assert handle.apply_alpha.value is True

    def test_apply_alpha_defaults_to_unchecked(self):
        client, handle = build_popup()
        assert handle.apply_alpha.value is False
        assert "Override alpha" in {
            str(getattr(element, "text", ""))
            for element in client.elements.values()
        }
        handle.alpha.value = 0.4
        assert handle.get_value() == "#145cff"

    def test_alpha_uses_a_plain_input_and_a_005_step_slider(self):
        _client, handle = build_popup()
        assert handle.alpha._props["type"] == "text"
        assert handle.alpha_slider._props["min"] == 0
        assert handle.alpha_slider._props["max"] == 1
        assert handle.alpha_slider._props["step"] == 0.05

        handle._on_alpha_change(SimpleNamespace(value="0.13"))
        assert handle.alpha.value == 0.15
        assert handle.alpha_slider.value == 0.15

        handle._on_alpha_slider_change(SimpleNamespace(value=0.8))
        assert handle.alpha.value == 0.8
        handle.apply_alpha.value = True
        assert handle.get_value() == "rgba(20, 92, 255, 0.8)"

    def test_on_submit_commit_calls_callback(self):
        _client, handle = build_popup()
        received = []
        handle.on_submit(lambda v: received.append(v))
        handle._current = "#22c55e"
        handle.alpha.value = 0.75
        handle.apply_alpha.value = True
        handle._commit()
        assert received == ["rgba(34, 197, 94, 0.75)"]

    def test_outside_dismiss_commits_once(self):
        _client, handle = build_popup()
        received = []
        handle.on_submit(received.append)
        handle.open("#145cff")
        handle._current = "#22c55e"

        # QDialog's hide event is the server-side equivalent of clicking
        # outside the dismissible picker card.
        handle._on_dialog_hide()
        handle._on_dialog_hide()
        assert received == ["#22c55e"]

    def test_cancel_dismiss_does_not_commit(self):
        _client, handle = build_popup()
        received = []
        handle.on_submit(received.append)
        handle.open("#145cff")
        handle._current = "#22c55e"
        handle._cancel()
        handle._on_dialog_hide()
        assert received == []

    def test_set_from_swatch_updates_current_and_picker(self):
        _client, handle = build_popup()
        handle._set_from_swatch("#ff0000")
        assert handle._current == "#ff0000"
        # The inline q-color grid is present and carries the model-value.
        assert handle.q_color is not None
        assert handle.q_color._props.get("model-value") == "#ff0000"

    def test_palette_selector_populates_swatches(self):
        _client, handle = build_popup()
        # The default palette (Category10) should populate a swatch grid.
        assert handle._swatch_row is not None
        slot = handle._swatch_row.default_slot
        # Each swatch is a wrapper div plus an inner colour div, so at least
        # 10 children for a 10-colour palette.
        assert len(slot.children) >= 10

    def test_palette_selector_embeds_horizontal_strip_previews(self):
        _client, handle = build_popup()
        select = handle._palette_select
        options = select._props.get("options") or []

        assert options and all(
            isinstance(option.get("label"), str)
            and "colors" in option
            and "strip" in option
            for option in options
        )
        assert "option" in select.slots
        template = select.slots["option"].template
        assert "drocat-select-palette-strip" in template
        assert "props.opt.strip" in template
        category10 = next(option for option in options if option["label"] == "Category10")
        assert category10["colors"]

    def test_swatch_picker_wires_popup(self):
        """The swatch picker exposes the popup and applies its committed color."""
        from ui.components.palette_picker import color_swatch_picker

        client = Client(page(f"/swatch-{uuid.uuid4().hex}"))
        with client:
            picker = color_swatch_picker("Test", value="auto")
        popup = picker.pick_popup
        assert popup is not None
        popup._current = "#ff0000"
        popup.alpha.value = 0.5
        popup.apply_alpha.value = True
        popup._commit()
        assert picker.get_value() == "rgba(255, 0, 0, 0.5)"
