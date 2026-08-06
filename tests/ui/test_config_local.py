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
