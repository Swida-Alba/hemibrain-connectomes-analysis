"""Header theme picker tests: light / dark / system-following modes.

The header button shows a persistent sun | moon icon pair (separated by a
thick vertical line) and opens a three-item menu (System, Light, Dark)
driving NiceGUI's ``ui.dark_mode()`` element. ``None`` selects Quasar auto
mode, which follows the OS ``prefers-color-scheme`` setting live
(macOS/Windows).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from nicegui import Client
from nicegui.page import page

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import ui.app as app_module  # noqa: E402


def _build_page():
    client = Client(page("/theme-picker-test"))
    with client:
        app_module.main_page()
    return client


def _of_class(client, class_name):
    return [
        element for element in client.elements.values()
        if class_name in getattr(element, "_classes", set())
    ]


def _dark(client):
    return next(element for element in client.elements.values()
                if type(element).__name__ == "DarkMode")


def _theme_button(client):
    return _of_class(client, "drocat-dark-toggle")[0]


def _theme_items(client):
    # Created in THEME_OPTIONS order: System, Light, Dark.
    return _of_class(client, "drocat-theme-item")


def _theme_icons(client):
    """(sun, moon) icon elements inside the header button's pair."""
    sun = _of_class(client, "drocat-theme-sun")
    moon = _of_class(client, "drocat-theme-moon")
    assert len(sun) == 1, "expected exactly one sun icon"
    assert len(moon) == 1, "expected exactly one moon icon"
    return sun[0], moon[0]


def _click(item):
    listeners = [listener for listener in item._event_listeners.values()
                 if listener.type == "click"]
    assert listeners, "theme item has no click listener"
    listeners[-1].handler(None)


def test_theme_menu_offers_system_light_dark():
    """The header picker lists all three theme choices."""
    client = _build_page()
    items = _theme_items(client)
    assert len(items) == 3
    labels = {
        element.text
        for element in client.elements.values()
        if getattr(element, "text", None) in {"System", "Light", "Dark"}
    }
    assert labels == {"System", "Light", "Dark"}


def test_button_shows_sun_moon_pair_in_every_mode(monkeypatch):
    """The header button keeps its sun | moon pair across all three modes."""
    scripts = []
    monkeypatch.setattr(app_module.ui, "run_javascript", scripts.append)
    client = _build_page()
    dark = _dark(client)
    items = _theme_items(client)
    sun, moon = _theme_icons(client)
    assert sun._props.get("name") == "light_mode"
    assert moon._props.get("name") == "dark_mode"

    for item, expected in [(items[0], None), (items[2], True), (items[1], False)]:
        _click(item)
        assert dark.value is expected
        sun, moon = _theme_icons(client)
        assert sun._props.get("name") == "light_mode"
        assert moon._props.get("name") == "dark_mode"


def test_default_theme_is_system_with_active_highlight():
    """Without a saved preference the UI follows the OS, System highlighted."""
    client = _build_page()
    items = _theme_items(client)
    assert _dark(client).value is None
    assert "drocat-theme-item-active" in items[0]._classes
    assert "drocat-theme-item-active" not in items[1]._classes
    assert "drocat-theme-item-active" not in items[2]._classes


def test_selecting_system_follows_os_preference(monkeypatch):
    """System sets the dark_mode element to auto (None) and persists it."""
    scripts = []
    monkeypatch.setattr(app_module.ui, "run_javascript", scripts.append)
    client = _build_page()
    dark = _dark(client)
    items = _theme_items(client)

    _click(items[0])

    assert dark.value is None
    assert "drocat-theme-item-active" in items[0]._classes
    assert "drocat-theme-item-active" not in items[1]._classes
    assert scripts and "drocat_dark=auto" in scripts[-1]


def test_selecting_dark_and_light_update_element_and_cookie(monkeypatch):
    """Dark/Light flip the element value, highlight, and saved cookie."""
    scripts = []
    monkeypatch.setattr(app_module.ui, "run_javascript", scripts.append)
    client = _build_page()
    dark = _dark(client)
    items = _theme_items(client)

    _click(items[2])
    assert dark.value is True
    assert "drocat-theme-item-active" in items[2]._classes
    assert scripts and "drocat_dark=dark" in scripts[-1]

    _click(items[1])
    assert dark.value is False
    assert "drocat-theme-item-active" in items[1]._classes
    assert scripts and "drocat_dark=light" in scripts[-1]


def test_saved_dark_mode_reads_cookie_values(monkeypatch):
    """The cookie supports 'dark'/'light'/'auto' plus legacy '1'/'0'."""

    class _Request:
        def __init__(self, cookies):
            self.cookies = cookies

    class _Client:
        def __init__(self, cookies):
            self.request = _Request(cookies)

    for cookies, expected in [
        ({"drocat_dark": "dark"}, True),
        ({"drocat_dark": "1"}, True),   # legacy value
        ({"drocat_dark": "light"}, False),
        ({"drocat_dark": "0"}, False),  # legacy value
        ({"drocat_dark": "auto"}, None),
        ({}, None),                     # no cookie -> system
    ]:
        monkeypatch.setattr(app_module.ui, "context",
                            SimpleNamespace(client=_Client(cookies)))
        assert app_module._saved_dark_mode() is expected


def test_saved_dark_mode_defaults_to_system_without_request_context():
    """Outside a request context (script mode, UI tests) default to system."""
    assert app_module._saved_dark_mode() is None


def test_theme_css_contains_picker_rules():
    """The theme menu relies on the shared DROCAT stylesheet rules."""
    assert ".drocat-theme-menu" in app_module.DROCAT_CSS
    assert ".drocat-theme-item-active" in app_module.DROCAT_CSS
    assert ".drocat-theme-check" in app_module.DROCAT_CSS
    assert ".drocat-theme-icon-pair" in app_module.DROCAT_CSS
    assert ".drocat-theme-sep" in app_module.DROCAT_CSS
    # The separator is a thick vertical line, not a text glyph.
    assert "width: 2.5px;" in app_module.DROCAT_CSS
    assert "border-radius: 999px;" in app_module.DROCAT_CSS
    # Icons render at exactly 2/3 of the previous 34.3px size.
    assert ".drocat-theme-icon-pair .q-icon" in app_module.DROCAT_CSS
    assert "font-size: 22.9px;" in app_module.DROCAT_CSS
