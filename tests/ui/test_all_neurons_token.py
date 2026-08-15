"""UI helper tests for the 'all_neurons' special chip token."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestUsesAllNeuronsToken:
    def test_detects_token_in_any_case(self):
        from ui.components.common import uses_all_neurons_token

        assert uses_all_neurons_token(["all_neurons"])
        assert uses_all_neurons_token(["aMe12", "ALL_NEURONS"])
        assert uses_all_neurons_token([" all_neurons "])

    def test_ignores_ordinary_chips(self):
        from ui.components.common import uses_all_neurons_token

        assert not uses_all_neurons_token([])
        assert not uses_all_neurons_token(None)
        assert not uses_all_neurons_token(["aMe12", "PPL101"])
        assert not uses_all_neurons_token(["all_neurons_x", "not_the_token"])
