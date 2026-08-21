"""Shared pytest configuration.

Resolves the NeuPrint token through the standard DROCAT fallback chain
(an explicitly exported NEUPRINT_APPLICATION_CREDENTIALS env var wins,
then config.json, then the gitignored config_local.json) and exposes it
as the env var neuprint-python itself reads.  Tests that create neuprint
clients therefore behave exactly like the runtime without requiring the
shell to export the token.
"""
import os
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))


@pytest.fixture(scope="session", autouse=True)
def _neuprint_token_from_config_chain():
    """Seed NEUPRINT_APPLICATION_CREDENTIALS from config files, if absent.

    Tests that need to simulate a tokenless environment still can:
    ``monkeypatch.delenv('NEUPRINT_APPLICATION_CREDENTIALS')`` removes the
    value for that test and pytest restores it afterwards.
    """
    if os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS"):
        return
    try:
        from utils.token_manager import token_manager
        token = token_manager.get_neuprint_token()
    except Exception:
        token = None
    if token:
        os.environ["NEUPRINT_APPLICATION_CREDENTIALS"] = token
