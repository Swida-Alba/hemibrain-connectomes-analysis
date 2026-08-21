"""Tests for TokenManager: config.json loading, precedence, env fallback, and
token type auto-detection."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.token_manager import TokenManager


class TestTokenManager:
    def test_loads_from_config_json(self, tmp_path, monkeypatch):
        (tmp_path / "config.json").write_text(
            '{"tokens": {"neuprint": "cfg-np", "cave": "cfg-cave"}}\n',
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        manager = TokenManager()
        assert manager.tokens.get("NEUPRINT_TOKEN") == "cfg-np"
        assert manager.tokens.get("CAVE_TOKEN") == "cfg-cave"

    def test_direct_input_wins_over_config(self):
        manager = TokenManager()
        token = manager.get_token("NEUPRINT_TOKEN", direct_input="direct-tok")
        assert token == "direct-tok"

    def test_placeholder_token_ignored(self, monkeypatch):
        manager = TokenManager()
        manager.tokens = {"NEUPRINT_TOKEN": "YOUR_NEUPRINT_TOKEN_HERE"}
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
        assert manager.get_token("NEUPRINT_TOKEN") is None

    def test_env_fallback(self, monkeypatch):
        manager = TokenManager()
        manager.tokens = {}
        monkeypatch.setenv("NEUPRINT_TOKEN", "env-tok")
        assert manager.get_token("NEUPRINT_TOKEN") == "env-tok"
        monkeypatch.delenv("NEUPRINT_TOKEN")

    def test_env_fallback_reads_canonical_application_credentials(
            self, monkeypatch):
        """NEUPRINT_APPLICATION_CREDENTIALS (the variable neuprint-python
        itself reads) is honored even when NEUPRINT_TOKEN is unset."""
        manager = TokenManager()
        manager.tokens = {}
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        monkeypatch.setenv("NEUPRINT_APPLICATION_CREDENTIALS", "canonical-tok")
        try:
            assert manager.get_token("NEUPRINT_TOKEN") == "canonical-tok"
            assert manager.get_neuprint_token() == "canonical-tok"
        finally:
            monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS")

    def test_config_json_wins_over_config_local(self, tmp_path, monkeypatch):
        """config.json wins per key; config_local.json only fills empties."""
        (tmp_path / "config.json").write_text(
            '{"tokens": {"neuprint": "cfg-np", "cave": "cfg-cave"}}\n',
            encoding="utf-8",
        )
        (tmp_path / "config_local.json").write_text(
            '{"tokens": {"neuprint": "local-np"}}\n', encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        manager = TokenManager()
        assert manager.tokens.get("NEUPRINT_TOKEN") == "cfg-np"
        assert manager.tokens.get("CAVE_TOKEN") == "cfg-cave"
    
    def test_config_local_fills_empty_config_json_entry(self, tmp_path, monkeypatch):
        """An empty config.json entry falls back to config_local.json."""
        (tmp_path / "config.json").write_text(
            '{"tokens": {"neuprint": "", "cave": "cfg-cave"}}\n',
            encoding="utf-8",
        )
        (tmp_path / "config_local.json").write_text(
            '{"tokens": {"neuprint": "local-np"}}\n', encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        manager = TokenManager()
        assert manager.tokens.get("NEUPRINT_TOKEN") == "local-np"
        assert manager.tokens.get("CAVE_TOKEN") == "cfg-cave"

    def test_config_json_empty_values_fall_back_to_env(self, tmp_path, monkeypatch):
        (tmp_path / "config.json").write_text(
            '{"tokens": {"neuprint": "", "cave": ""}}\n', encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CAVE_TOKEN", "env-cave")
        try:
            manager = TokenManager(project_root=str(tmp_path))
            assert manager.tokens.get("NEUPRINT_TOKEN") is None
            assert manager.get_token("CAVE_TOKEN") == "env-cave"
        finally:
            monkeypatch.delenv("CAVE_TOKEN")

    def test_legacy_token_files_are_not_read(self, tmp_path, monkeypatch):
        """token_info files are deprecated; only config.json is read."""
        (tmp_path / "token_info_local.txt").write_text(
            "NEUPRINT_TOKEN='legacy-np'\nCAVE_TOKEN='legacy-cave'\n", encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        manager = TokenManager(project_root=str(tmp_path))
        assert manager.tokens.get("NEUPRINT_TOKEN") is None
        assert manager.tokens.get("CAVE_TOKEN") is None

    def test_detect_token_type(self):
        tm = TokenManager()
        long_jwt = "x" * 120 + ".y" * 10
        assert tm.detect_token_type(long_jwt) == "neuprint"
        assert tm.detect_token_type("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4") == "cave"
        assert tm.detect_token_type("") == "unknown"
        assert tm.detect_token_type("not-a-token!") == "unknown"

    def test_require_both_tokens_raises_when_missing(self, monkeypatch):
        manager = TokenManager()
        manager.tokens = {}
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("CAVE_TOKEN", raising=False)
        with pytest.raises(ValueError, match="NEUPRINT_TOKEN"):
            manager.require_both_tokens()

    def test_direct_input_detects_neuprint(self, monkeypatch):
        """A long JWT-like direct input detects as neuprint; CAVE may be absent."""
        manager = TokenManager()
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("CAVE_TOKEN", raising=False)
        result = manager.get_auto_token(direct_input="x" * 150 + ".y" * 10)
        assert result["neuprint"]
        assert result["detected_type"] == "neuprint"

    def test_get_auto_token_no_direct_input(self, monkeypatch):
        manager = TokenManager()
        manager.tokens = {}
        monkeypatch.setenv("NEUPRINT_TOKEN", "env-np")
        monkeypatch.setenv("CAVE_TOKEN", "env-cave")
        try:
            result = manager.get_auto_token()
        finally:
            monkeypatch.delenv("NEUPRINT_TOKEN")
            monkeypatch.delenv("CAVE_TOKEN")
        assert result["neuprint"] == "env-np"
        assert result["cave"] == "env-cave"
