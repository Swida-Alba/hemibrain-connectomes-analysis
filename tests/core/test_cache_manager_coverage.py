"""Hermetic coverage tests for src/core/cache_manager.py.

cache_manager is a menu-driven CLI around the NeuPrint cache.  Every path
lookup is relative to the CWD (``neuprint_cache/``), so each test chdirs into
``tmp_path``.  ``FindNeuronConnection`` is replaced with a recording fake and
``builtins.input`` is fed scripted answers; no network or real cache is ever
touched.
"""

import builtins
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import core.cache_manager as cache_manager  # noqa: E402


def _feed(monkeypatch, responses):
    """Script ``input()`` answers in order."""
    iterator = iter(responses)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(iterator))


@pytest.fixture
def fake_fnc(monkeypatch):
    """Replace cache_manager.FindNeuronConnection with a recording fake."""
    records = {"instances": []}

    class FakeFNC:
        def __init__(self, dataset=None, sourceNeurons=None,
                     targetNeurons=None, use_cache=True, token='', **kwargs):
            self.dataset = dataset
            self.sourceNeurons = sourceNeurons
            self.targetNeurons = targetNeurons
            self.use_cache = use_cache
            self.token = token
            self.registry = pd.DataFrame()
            self.registry_error = None
            self.search_results = pd.DataFrame()
            self.last_search = None
            self.printed_info = 0
            self.cleared = []
            records["instances"].append(self)

        def print_cache_info(self):
            self.printed_info += 1

        def _load_neuron_registry(self):
            if self.registry_error:
                raise self.registry_error
            return self.registry

        def search_cached_neurons(self, pattern, field):
            self.last_search = (pattern, field)
            return self.search_results

        def clear_cache(self, confirm=False):
            self.cleared.append(confirm)

    monkeypatch.setattr(cache_manager, "FindNeuronConnection", FakeFNC)
    records["class"] = FakeFNC
    return records


# ---------------------------------------------------------------------------
# Menu + view_cache_info
# ---------------------------------------------------------------------------

def test_print_menu_lists_all_options(capsys):
    cache_manager.print_menu()
    out = capsys.readouterr().out
    assert "NEUPRINT CACHE MANAGEMENT" in out
    for option in ("1. View cache information", "2. View cache for specific",
                   "3. Search cached neurons", "4. Clear cache for specific",
                   "5. Clear all cache", "6. Exit"):
        assert option in out


def test_view_cache_info_missing_root(tmp_path, monkeypatch, capsys, fake_fnc):
    monkeypatch.chdir(tmp_path)
    cache_manager.view_cache_info()
    assert "No cache found" in capsys.readouterr().out
    assert fake_fnc["instances"] == []


def test_view_cache_info_no_dataset_dirs(tmp_path, monkeypatch, capsys, fake_fnc):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "neuprint_cache"
    root.mkdir()
    (root / "stray_file.txt").write_text("not a dataset")
    cache_manager.view_cache_info()
    assert "No cached datasets found" in capsys.readouterr().out
    assert fake_fnc["instances"] == []


def test_view_cache_info_decodes_dataset_folders(tmp_path, monkeypatch, fake_fnc):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "neuprint_cache"
    (root / "hemibrain_v1_2_1").mkdir(parents=True)
    (root / "manc_v1_0").mkdir(parents=True)

    cache_manager.view_cache_info()

    datasets = [inst.dataset for inst in fake_fnc["instances"]]
    assert datasets == ["hemibrain:v1.2.1", "manc:v1.0"]
    for inst in fake_fnc["instances"]:
        assert inst.use_cache is True
        assert inst.printed_info == 1


# ---------------------------------------------------------------------------
# view_dataset_cache
# ---------------------------------------------------------------------------

def test_view_dataset_cache_predefined_choice(monkeypatch, capsys, fake_fnc):
    _feed(monkeypatch, ["1"])
    cache_manager.view_dataset_cache()
    assert fake_fnc["instances"][0].dataset == "hemibrain:v1.2.1"
    assert fake_fnc["instances"][0].printed_info == 1


def test_view_dataset_cache_manual_entry(monkeypatch, fake_fnc):
    _feed(monkeypatch, ["4", "custom:v9.9"])
    cache_manager.view_dataset_cache()
    assert fake_fnc["instances"][0].dataset == "custom:v9.9"


def test_view_dataset_cache_invalid_choice(monkeypatch, capsys, fake_fnc):
    _feed(monkeypatch, ["abc"])
    cache_manager.view_dataset_cache()
    assert "Invalid selection" in capsys.readouterr().out
    assert fake_fnc["instances"] == []


def test_view_dataset_cache_out_of_range(monkeypatch, capsys, fake_fnc):
    _feed(monkeypatch, ["9"])
    cache_manager.view_dataset_cache()
    assert "Invalid selection" in capsys.readouterr().out
    assert fake_fnc["instances"] == []


# ---------------------------------------------------------------------------
# search_cached_neurons
# ---------------------------------------------------------------------------

def _registry():
    return pd.DataFrame({
        "bodyId": [42, 43],
        "type": ["L3", "PPL1"],
        "instance": ["L3_R", "PPL1_L"],
    })


def test_search_cached_neurons_empty_registry(monkeypatch, capsys, fake_fnc):
    _feed(monkeypatch, ["1"])
    cache_manager.search_cached_neurons()
    assert "No neuron registry found" in capsys.readouterr().out


def test_search_cached_neurons_by_type(monkeypatch, capsys, fake_fnc):
    Base = fake_fnc["class"]

    class PopulatedFNC(Base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.registry = _registry()
            self.search_results = _registry().head(1)

    monkeypatch.setattr(cache_manager, "FindNeuronConnection", PopulatedFNC)
    _feed(monkeypatch, ["1", "1", "L3.*"])
    cache_manager.search_cached_neurons()
    out = capsys.readouterr().out
    assert "Found 2 neurons in registry" in out
    assert "Found 1 neurons matching type pattern 'L3.*'" in out
    assert "L3" in out


def test_search_cached_neurons_by_instance(monkeypatch, capsys, fake_fnc):
    Base = fake_fnc["class"]

    class PopulatedFNC(Base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.registry = _registry()
            self.search_results = pd.DataFrame()

    monkeypatch.setattr(cache_manager, "FindNeuronConnection", PopulatedFNC)
    _feed(monkeypatch, ["1", "2", ".*_R"])
    cache_manager.search_cached_neurons()
    out = capsys.readouterr().out
    assert "instance pattern '.*_R'" in out


def test_search_cached_neurons_by_instance_prints_results(monkeypatch, capsys, fake_fnc):
    Base = fake_fnc["class"]

    class PopulatedFNC(Base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.registry = _registry()
            self.search_results = _registry().head(1)

    monkeypatch.setattr(cache_manager, "FindNeuronConnection", PopulatedFNC)
    _feed(monkeypatch, ["1", "2", ".*_R"])
    cache_manager.search_cached_neurons()
    out = capsys.readouterr().out
    assert "Found 1 neurons matching instance pattern '.*_R'" in out
    assert "L3_R" in out


def test_search_cached_neurons_by_bodyid(monkeypatch, capsys, fake_fnc):
    Base = fake_fnc["class"]
    captured = {}

    class PopulatedFNC(Base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.registry = _registry()

        def search_cached_neurons(self, pattern, field):
            captured["pattern"] = pattern
            captured["field"] = field
            return _registry().head(1)

    monkeypatch.setattr(cache_manager, "FindNeuronConnection", PopulatedFNC)
    _feed(monkeypatch, ["1", "3", "42"])
    cache_manager.search_cached_neurons()
    assert captured == {"pattern": 42, "field": "bodyId"}
    assert "Results for bodyId 42" in capsys.readouterr().out


def test_search_cached_neurons_bodyid_not_found(monkeypatch, capsys, fake_fnc):
    Base = fake_fnc["class"]

    class PopulatedFNC(Base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.registry = _registry()
            self.search_results = pd.DataFrame()

    monkeypatch.setattr(cache_manager, "FindNeuronConnection", PopulatedFNC)
    _feed(monkeypatch, ["1", "3", "999"])
    cache_manager.search_cached_neurons()
    assert "No matching neuron found" in capsys.readouterr().out


def test_search_cached_neurons_invalid_method(monkeypatch, capsys, fake_fnc):
    Base = fake_fnc["class"]

    class PopulatedFNC(Base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.registry = _registry()

    monkeypatch.setattr(cache_manager, "FindNeuronConnection", PopulatedFNC)
    _feed(monkeypatch, ["1", "9"])
    cache_manager.search_cached_neurons()
    assert "Invalid search method" in capsys.readouterr().out


def test_search_cached_neurons_invalid_dataset_choice(monkeypatch, capsys, fake_fnc):
    _feed(monkeypatch, ["abc"])
    cache_manager.search_cached_neurons()
    assert "Invalid selection" in capsys.readouterr().out


def test_search_cached_neurons_manual_dataset(monkeypatch, capsys, fake_fnc):
    Base = fake_fnc["class"]

    class PopulatedFNC(Base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.registry = _registry()

    monkeypatch.setattr(cache_manager, "FindNeuronConnection", PopulatedFNC)
    _feed(monkeypatch, ["4", "custom:v2", "9"])
    cache_manager.search_cached_neurons()
    assert fake_fnc["instances"][0].dataset == "custom:v2"


def test_search_cached_neurons_registry_failure(monkeypatch, capsys, fake_fnc):
    Base = fake_fnc["class"]

    class BrokenFNC(Base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.registry_error = RuntimeError("registry exploded")

    monkeypatch.setattr(cache_manager, "FindNeuronConnection", BrokenFNC)
    _feed(monkeypatch, ["1"])
    cache_manager.search_cached_neurons()
    assert "Search failed: registry exploded" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# clear_dataset_cache
# ---------------------------------------------------------------------------

def test_clear_dataset_cache_missing_root(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cache_manager.clear_dataset_cache()
    assert "No cache found" in capsys.readouterr().out


def test_clear_dataset_cache_no_datasets(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "neuprint_cache").mkdir()
    cache_manager.clear_dataset_cache()
    assert "No cached datasets found" in capsys.readouterr().out


def test_clear_dataset_cache_clears_selection(tmp_path, monkeypatch, fake_fnc):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "neuprint_cache" / "hemibrain_v1_2_1").mkdir(parents=True)
    _feed(monkeypatch, ["1"])

    cache_manager.clear_dataset_cache()

    inst = fake_fnc["instances"][0]
    assert inst.dataset == "hemibrain:v1.2.1"
    assert inst.cleared == [True]


def test_clear_dataset_cache_invalid_choice(tmp_path, monkeypatch, capsys, fake_fnc):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "neuprint_cache" / "manc_v1_0").mkdir(parents=True)
    _feed(monkeypatch, ["zz"])
    cache_manager.clear_dataset_cache()
    assert "Invalid selection" in capsys.readouterr().out
    assert fake_fnc["instances"] == []


def test_clear_dataset_cache_out_of_range(tmp_path, monkeypatch, capsys, fake_fnc):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "neuprint_cache" / "manc_v1_0").mkdir(parents=True)
    _feed(monkeypatch, ["5"])
    cache_manager.clear_dataset_cache()
    assert "Invalid selection" in capsys.readouterr().out
    assert fake_fnc["instances"] == []


# ---------------------------------------------------------------------------
# clear_all_cache
# ---------------------------------------------------------------------------

def test_clear_all_cache_missing_root(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cache_manager.clear_all_cache()
    assert "No cache found" in capsys.readouterr().out


def test_clear_all_cache_declined(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "neuprint_cache"
    root.mkdir()
    _feed(monkeypatch, ["no"])
    cache_manager.clear_all_cache()
    assert "Operation cancelled." in capsys.readouterr().out
    assert root.exists()


def test_clear_all_cache_confirmed_removes_root(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "neuprint_cache"
    (root / "hemibrain_v1_2_1").mkdir(parents=True)
    _feed(monkeypatch, ["yes"])
    cache_manager.clear_all_cache()
    assert "All cache cleared" in capsys.readouterr().out
    assert not root.exists()


def test_clear_all_cache_removal_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "neuprint_cache").mkdir()

    def boom(path):
        raise OSError("disk locked")

    monkeypatch.setattr(shutil, "rmtree", boom)
    _feed(monkeypatch, ["yes"])
    cache_manager.clear_all_cache()
    assert "Failed to clear cache: disk locked" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# main() menu loop
# ---------------------------------------------------------------------------

def test_main_exits_on_option_six(monkeypatch, capsys):
    _feed(monkeypatch, ["6"])
    with pytest.raises(SystemExit) as excinfo:
        cache_manager.main()
    assert excinfo.value.code == 0
    assert "Goodbye!" in capsys.readouterr().out


def test_main_rejects_invalid_option_then_exits(monkeypatch, capsys):
    _feed(monkeypatch, ["9", "", "6"])
    with pytest.raises(SystemExit):
        cache_manager.main()
    assert "Invalid option" in capsys.readouterr().out


def test_main_dispatches_view_cache_info(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    # Option 1 with no cache present, then exit.
    _feed(monkeypatch, ["1", "", "6"])
    with pytest.raises(SystemExit):
        cache_manager.main()
    out = capsys.readouterr().out
    assert "No cache found" in out


def test_main_dispatches_remaining_options(tmp_path, monkeypatch, capsys, fake_fnc):
    """Options 2-5 must each reach their handler before the loop exits."""
    monkeypatch.chdir(tmp_path)
    _feed(monkeypatch, [
        "2", "1", "",   # view_dataset_cache -> predefined hemibrain choice
        "3", "1", "",   # search_cached_neurons -> empty registry notice
        "4", "",        # clear_dataset_cache -> no cache found
        "5", "",        # clear_all_cache -> no cache found
        "6",            # exit
    ])
    with pytest.raises(SystemExit) as excinfo:
        cache_manager.main()
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "No neuron registry found" in out
    assert fake_fnc["instances"][0].dataset == "hemibrain:v1.2.1"
