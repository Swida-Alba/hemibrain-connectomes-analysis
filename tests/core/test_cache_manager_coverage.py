"""Hermetic coverage tests for ``src/core/cache_manager.py``.

The module is an interactive CLI around the NeuPrint connection cache.  All
interactions are exercised with a fake ``FindNeuronConnection`` and scripted
``input()`` responses; every filesystem touch is sandboxed to ``tmp_path``
via ``monkeypatch.chdir``.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core import cache_manager as cm  # noqa: E402


class FakeFNC:
    """Stand-in for ``FindNeuronConnection`` with scriptable behavior."""

    registry = pd.DataFrame()
    search_results = pd.DataFrame()
    registry_error = None
    cleared = []
    last_instance = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeFNC.last_instance = self

    def print_cache_info(self):
        print(f"cache-info:{self.kwargs.get('dataset')}")

    def _load_neuron_registry(self):
        if FakeFNC.registry_error is not None:
            raise FakeFNC.registry_error
        return FakeFNC.registry

    def search_cached_neurons(self, query, field):
        return FakeFNC.search_results

    def clear_cache(self, confirm=False):
        FakeFNC.cleared.append((self.kwargs.get("dataset"), confirm))


@pytest.fixture
def fake_fnc(monkeypatch):
    monkeypatch.setattr(cm, "FindNeuronConnection", FakeFNC)
    FakeFNC.registry = pd.DataFrame()
    FakeFNC.search_results = pd.DataFrame()
    FakeFNC.registry_error = None
    FakeFNC.cleared = []
    FakeFNC.last_instance = None
    return FakeFNC


def feed_inputs(monkeypatch, values):
    iterator = iter(values)
    monkeypatch.setattr("builtins.input", lambda *args, **kwargs: next(iterator))


# ---------------------------------------------------------------------------
# print_menu / view_cache_info
# ---------------------------------------------------------------------------


def test_print_menu_lists_all_options(capsys):
    cm.print_menu()
    output = capsys.readouterr().out
    assert "NEUPRINT CACHE MANAGEMENT" in output
    for option in ("1.", "2.", "3.", "4.", "5.", "6."):
        assert option in output


def test_view_cache_info_no_cache_root(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cm.view_cache_info()
    assert "No cache found" in capsys.readouterr().out


def test_view_cache_info_no_dataset_dirs(tmp_path, monkeypatch, fake_fnc, capsys):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "neuprint_cache"
    root.mkdir()
    (root / "not_a_dir.txt").write_text("x")
    cm.view_cache_info()
    assert "No cached datasets found" in capsys.readouterr().out


def test_view_cache_info_lists_datasets(tmp_path, monkeypatch, fake_fnc, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "neuprint_cache" / "hemibrain_v1_2_1").mkdir(parents=True)
    cm.view_cache_info()
    output = capsys.readouterr().out
    assert "Found cache for 1 dataset(s)" in output
    assert "cache-info:hemibrain:v1.2.1" in output
    assert FakeFNC.last_instance.kwargs["use_cache"] is True
    assert FakeFNC.last_instance.kwargs["token"] == ""


# ---------------------------------------------------------------------------
# view_dataset_cache
# ---------------------------------------------------------------------------


def test_view_dataset_cache_preset_choice(monkeypatch, fake_fnc, capsys):
    feed_inputs(monkeypatch, ["1"])
    cm.view_dataset_cache()
    output = capsys.readouterr().out
    assert "Available datasets" in output
    assert "cache-info:hemibrain:v1.2.1" in output


def test_view_dataset_cache_manual_entry(monkeypatch, fake_fnc, capsys):
    feed_inputs(monkeypatch, ["4", "manc:v1.0"])
    cm.view_dataset_cache()
    assert "cache-info:manc:v1.0" in capsys.readouterr().out


def test_view_dataset_cache_invalid_choice(monkeypatch, fake_fnc, capsys):
    feed_inputs(monkeypatch, ["abc"])
    cm.view_dataset_cache()
    assert "Invalid selection" in capsys.readouterr().out


def test_view_dataset_cache_out_of_range_choice(monkeypatch, fake_fnc, capsys):
    feed_inputs(monkeypatch, ["99"])
    cm.view_dataset_cache()
    assert "Invalid selection" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# search_cached_neurons
# ---------------------------------------------------------------------------


def test_search_cached_neurons_empty_registry(monkeypatch, fake_fnc, capsys):
    feed_inputs(monkeypatch, ["1"])
    cm.search_cached_neurons()
    assert "No neuron registry found" in capsys.readouterr().out


def _registry():
    return pd.DataFrame(
        {"bodyId": [1, 2], "type": ["Mi1", "T4a"], "instance": ["Mi1_R", "T4a_L"]}
    )


def test_search_cached_neurons_by_type(monkeypatch, fake_fnc, capsys):
    FakeFNC.registry = _registry()
    FakeFNC.search_results = pd.DataFrame({"type": ["Mi1"]})
    feed_inputs(monkeypatch, ["1", "1", "Mi.*"])
    cm.search_cached_neurons()
    output = capsys.readouterr().out
    assert "Found 2 neurons in registry" in output
    assert "matching type pattern 'Mi.*'" in output
    assert "Mi1" in output


def test_search_cached_neurons_by_instance_empty(monkeypatch, fake_fnc, capsys):
    FakeFNC.registry = _registry()
    FakeFNC.search_results = pd.DataFrame()
    feed_inputs(monkeypatch, ["1", "2", ".*_X"])
    cm.search_cached_neurons()
    assert "matching instance pattern '.*_X'" in capsys.readouterr().out


def test_search_cached_neurons_by_bodyid_found(monkeypatch, fake_fnc, capsys):
    FakeFNC.registry = _registry()
    FakeFNC.search_results = pd.DataFrame({"bodyId": [1], "type": ["Mi1"]})
    feed_inputs(monkeypatch, ["1", "3", "1"])
    cm.search_cached_neurons()
    assert "Results for bodyId 1" in capsys.readouterr().out


def test_search_cached_neurons_by_bodyid_not_found(monkeypatch, fake_fnc, capsys):
    FakeFNC.registry = _registry()
    FakeFNC.search_results = pd.DataFrame()
    feed_inputs(monkeypatch, ["1", "3", "42"])
    cm.search_cached_neurons()
    output = capsys.readouterr().out
    assert "Results for bodyId 42" in output
    assert "No matching neuron found" in output


def test_search_cached_neurons_invalid_search_method(monkeypatch, fake_fnc, capsys):
    FakeFNC.registry = _registry()
    feed_inputs(monkeypatch, ["1", "9"])
    cm.search_cached_neurons()
    assert "Invalid search method" in capsys.readouterr().out


def test_search_cached_neurons_non_numeric_bodyid(monkeypatch, fake_fnc, capsys):
    FakeFNC.registry = _registry()
    feed_inputs(monkeypatch, ["1", "3", "not-an-int"])
    cm.search_cached_neurons()
    assert "Invalid selection" in capsys.readouterr().out


def test_search_cached_neurons_registry_error(monkeypatch, fake_fnc, capsys):
    FakeFNC.registry_error = RuntimeError("boom")
    feed_inputs(monkeypatch, ["1"])
    cm.search_cached_neurons()
    assert "Search failed: boom" in capsys.readouterr().out


def test_search_cached_neurons_invalid_dataset_choice(monkeypatch, fake_fnc, capsys):
    feed_inputs(monkeypatch, ["abc"])
    cm.search_cached_neurons()
    assert "Invalid selection" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# clear_dataset_cache / clear_all_cache
# ---------------------------------------------------------------------------


def test_clear_dataset_cache_no_root(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cm.clear_dataset_cache()
    assert "No cache found" in capsys.readouterr().out


def test_clear_dataset_cache_no_datasets(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "neuprint_cache").mkdir()
    cm.clear_dataset_cache()
    assert "No cached datasets found" in capsys.readouterr().out


def test_clear_dataset_cache_selects_dataset(tmp_path, monkeypatch, fake_fnc, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "neuprint_cache" / "hemibrain_v1_2_1").mkdir(parents=True)
    (tmp_path / "neuprint_cache" / "manc_v1_0").mkdir(parents=True)
    feed_inputs(monkeypatch, ["2"])
    cm.clear_dataset_cache()
    output = capsys.readouterr().out
    assert "hemibrain:v1.2.1" in output
    assert "manc:v1.0" in output
    assert len(FakeFNC.cleared) == 1
    assert FakeFNC.cleared[0] == ("hemibrain:v1.2.1", True)


def test_clear_dataset_cache_invalid_choice(tmp_path, monkeypatch, fake_fnc, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "neuprint_cache" / "manc_v1_0").mkdir(parents=True)
    feed_inputs(monkeypatch, ["abc"])
    cm.clear_dataset_cache()
    assert "Invalid selection" in capsys.readouterr().out
    assert FakeFNC.cleared == []


def test_clear_all_cache_no_root(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cm.clear_all_cache()
    assert "No cache found" in capsys.readouterr().out


def test_clear_all_cache_cancelled(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "neuprint_cache"
    root.mkdir()
    (root / "manc_v1_0").mkdir()
    feed_inputs(monkeypatch, ["no"])
    cm.clear_all_cache()
    assert "Operation cancelled" in capsys.readouterr().out
    assert root.exists()


def test_clear_all_cache_confirmed(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "neuprint_cache"
    (root / "manc_v1_0").mkdir(parents=True)
    feed_inputs(monkeypatch, ["yes"])
    cm.clear_all_cache()
    assert "All cache cleared" in capsys.readouterr().out
    assert not root.exists()


# ---------------------------------------------------------------------------
# main() dispatch loop
# ---------------------------------------------------------------------------


def test_main_invalid_option_then_exit(monkeypatch, fake_fnc, capsys):
    feed_inputs(monkeypatch, ["9", "", "6"])
    with pytest.raises(SystemExit) as excinfo:
        cm.main()
    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "Invalid option" in output
    assert "Goodbye" in output


def test_main_dispatches_every_action(monkeypatch, fake_fnc, capsys):
    calls = []
    monkeypatch.setattr(cm, "view_cache_info", lambda: calls.append("view_info"))
    monkeypatch.setattr(cm, "view_dataset_cache", lambda: calls.append("view_dataset"))
    monkeypatch.setattr(cm, "search_cached_neurons", lambda: calls.append("search"))
    monkeypatch.setattr(cm, "clear_dataset_cache", lambda: calls.append("clear_one"))
    monkeypatch.setattr(cm, "clear_all_cache", lambda: calls.append("clear_all"))
    feed_inputs(
        monkeypatch,
        ["1", "", "2", "", "3", "", "4", "", "5", "", "6"],
    )
    with pytest.raises(SystemExit):
        cm.main()
    assert calls == ["view_info", "view_dataset", "search", "clear_one", "clear_all"]
