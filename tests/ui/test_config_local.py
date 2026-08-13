"""Tests for the local UI configuration (ui/config.py): the user-editable
config JSON round-trip and the default output directory resolution."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ui.config as cfg


class TestLocalConfig:
    def test_load_missing_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE", tmp_path / "nope.json")
        assert cfg.load_local_config() == {}

    def test_save_then_load_roundtrip(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        assert cfg.save_local_config({"default_output_dir": "/tmp/out"}) is True
        assert cfg.load_local_config() == {"default_output_dir": "/tmp/out"}

    def test_load_invalid_json_returns_empty(self, monkeypatch, tmp_path):
        f = tmp_path / "local_config.json"
        f.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE", f)
        assert cfg.load_local_config() == {}

    def test_get_default_output_dir_without_override(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE", tmp_path / "nope.json")
        assert cfg.get_default_output_dir() == str(cfg.DEFAULT_OUTPUT_DIR)

    def test_get_default_output_dir_with_override(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        cfg.save_local_config({"default_output_dir": "/custom/out"})
        assert cfg.get_default_output_dir() == "/custom/out"

    def test_relative_override_is_rejected(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        cfg.save_local_config({"default_output_dir": "relative/path"})
        assert cfg.get_default_output_dir() == str(cfg.DEFAULT_OUTPUT_DIR)


class TestSetDefaultOutputDir:
    """Permanent persistence of the UI output directory (set_default_output_dir)."""

    def test_absolute_path_persists(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        target = tmp_path / "my_outputs"
        saved, effective = cfg.set_default_output_dir(str(target), create=False)
        assert saved is True
        assert effective == str(target)
        assert cfg.get_default_output_dir() == str(target)

    def test_relative_path_resolves_against_project_root(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        saved, effective = cfg.set_default_output_dir("rel/outputs", create=False)
        assert saved is True
        assert effective == str((cfg.PROJECT_ROOT / "rel/outputs").resolve())
        assert cfg.get_default_output_dir() == effective

    def test_empty_value_clears_override(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        cfg.set_default_output_dir(str(tmp_path / "out"), create=False)
        saved, effective = cfg.set_default_output_dir("")
        assert saved is True
        assert effective == str(cfg.DEFAULT_OUTPUT_DIR)
        assert cfg.get_default_output_dir() == str(cfg.DEFAULT_OUTPUT_DIR)

    def test_create_flag_makes_directory(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        target = tmp_path / "created" / "nested"
        saved, effective = cfg.set_default_output_dir(str(target), create=True)
        assert saved is True
        assert target.is_dir()
        assert effective == str(target)

    def test_no_create_does_not_make_directory(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        target = tmp_path / "not_created"
        cfg.set_default_output_dir(str(target), create=False)
        assert not target.exists()


class TestTabOutputDirs:
    """Tab overrides stay independent from the Settings default."""

    def test_tab_override_does_not_change_global_default(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        default = tmp_path / "default"
        tab_path = tmp_path / "pathfinding"
        cfg.set_default_output_dir(str(default), create=False)

        saved, effective = cfg.set_tab_output_dir("find_path", str(tab_path))
        assert saved is True
        assert effective == str(tab_path)
        assert cfg.get_tab_output_dir("find_path") == str(tab_path)
        assert cfg.get_tab_output_dir("find_shortest") == str(default)
        assert cfg.get_default_output_dir() == str(default)
        assert cfg.has_tab_output_override("find_path") is True

    def test_empty_tab_value_restores_inheritance(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        default = tmp_path / "default"
        cfg.set_default_output_dir(str(default), create=False)
        cfg.set_tab_output_dir("network", str(tmp_path / "network"))

        saved, effective = cfg.set_tab_output_dir("network", "")
        assert saved is True
        assert effective == str(default)
        assert cfg.has_tab_output_override("network") is False
        assert cfg.get_tab_output_dir("network") == str(default)

    def test_reset_clears_all_tab_overrides_and_preserves_default(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        default = tmp_path / "default"
        cfg.set_default_output_dir(str(default), create=False)
        cfg.set_tab_output_dir("find_path", str(tmp_path / "one"))
        cfg.set_tab_output_dir("network", str(tmp_path / "two"))

        assert cfg.clear_tab_output_overrides() is True
        assert cfg.has_tab_output_override("find_path") is False
        assert cfg.has_tab_output_override("network") is False
        assert cfg.get_default_output_dir() == str(default)
        assert cfg.load_local_config() == {"default_output_dir": str(default)}

    def test_relative_tab_path_resolves_against_project_root(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        saved, effective = cfg.set_tab_output_dir("flylight", "relative/out")
        expected = str((cfg.PROJECT_ROOT / "relative/out").resolve())
        assert saved is True
        assert effective == expected
        assert cfg.get_tab_output_dir("flylight") == expected


class TestAutoSuggestSetting:
    """The input auto-suggestion toggle persists in the local config."""

    def test_defaults_to_enabled(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        assert cfg.get_auto_suggest_enabled() is True

    def test_toggle_roundtrip(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        assert cfg.set_auto_suggest_enabled(False) is True
        assert cfg.get_auto_suggest_enabled() is False
        assert cfg.set_auto_suggest_enabled(True) is True
        assert cfg.get_auto_suggest_enabled() is True

    def test_other_keys_survive_toggle(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        cfg.save_local_config({"default_output_dir": "/custom/out"})
        cfg.set_auto_suggest_enabled(False)
        assert cfg.get_default_output_dir() == "/custom/out"
