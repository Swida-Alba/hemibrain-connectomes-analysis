"""Focused wiring checks for the connectivity profiling tab."""

from ui.tabs import connectivity_profiling as profiling_tab


def test_profiling_output_dir_prefers_selected_tab_value(monkeypatch):
    fallback_calls = []

    def fallback(scope):
        fallback_calls.append(scope)
        return "/tmp/inherited-profiling"

    monkeypatch.setattr(profiling_tab, "get_tab_output_dir", fallback)

    assert profiling_tab._resolve_profiling_output_dir(" /tmp/selected ") == "/tmp/selected"
    assert fallback_calls == []
    assert profiling_tab._resolve_profiling_output_dir("") == "/tmp/inherited-profiling"
    assert profiling_tab._resolve_profiling_output_dir("   ") == "/tmp/inherited-profiling"
    assert fallback_calls == ["connectivity_profiling", "connectivity_profiling"]
