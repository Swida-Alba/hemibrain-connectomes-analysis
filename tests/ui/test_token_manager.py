"""Tests for TokenManager: file loading, precedence, env fallback, and
token type auto-detection."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.token_manager import TokenManager


class TestTokenManager:
    def test_loads_from_repo_files(self):
        manager = TokenManager()
        assert "NEUPRINT_TOKEN" in manager.tokens
        assert "CAVE_TOKEN" in manager.tokens

    def test_direct_input_wins_over_files(self):
        manager = TokenManager()
        token = manager.get_token("NEUPRINT_TOKEN", direct_input="direct-tok")
        assert token == "direct-tok"

    def test_placeholder_token_ignored(self, monkeypatch):
        manager = TokenManager()
        manager.tokens = {"NEUPRINT_TOKEN": "YOUR_NEUPRINT_TOKEN_HERE"}
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        assert manager.get_token("NEUPRINT_TOKEN") is None

    def test_env_fallback(self, monkeypatch):
        manager = TokenManager()
        manager.tokens = {}
        monkeypatch.setenv("NEUPRINT_TOKEN", "env-tok")
        assert manager.get_token("NEUPRINT_TOKEN") == "env-tok"
        monkeypatch.delenv("NEUPRINT_TOKEN")

    def test_quoted_values_parsed(self, tmp_path, monkeypatch):
        (tmp_path / "token_info_local.txt").write_text(
            "NEUPRINT_TOKEN='quoted-tok'\n", encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        manager = TokenManager()
        assert manager.tokens.get("NEUPRINT_TOKEN") == "quoted-tok"

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
        monkeypatch.delenv("CAVE_TOKEN", raising=False)
        with pytest.raises(ValueError, match="NEUPRINT_TOKEN"):
            manager.require_both_tokens()

    def test_require_both_tokens_accepts_direct_input(self):
        manager = TokenManager()
        result = manager.require_both_tokens(
            direct_input="x" * 150 + ".y" * 10
        )
        # a long JWT-like direct input detects as neuprint; CAVE may be absent
        assert result["neuprint"]

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
