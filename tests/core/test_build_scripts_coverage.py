"""Hermetic coverage tests for the cache/index builder CLI scripts.

Targets:
- src/build_connection_cache.py
- src/build_connectivity_profile_cache.py
- src/build_seed_indexes.py

Everything runs against tmp_path: ``__file__`` is repointed so the derived
``cache/`` and ``neuron_indexes/`` paths land in tmp_path, FindNeuronConnection
is replaced with a recording fake, and the connectivity-profiler import is
satisfied with an in-memory stub module.  No network, no multiprocessing, no
real project directories.
"""

import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import coana  # noqa: E402  (patch target for the deferred imports)
import build_connection_cache as bcc  # noqa: E402
import build_connectivity_profile_cache as bcp  # noqa: E402
import build_seed_indexes as bsi  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fake FindNeuronConnection helpers
# ---------------------------------------------------------------------------

class _FakeProfile:
    def __init__(self, up=None, down=None):
        self.upstream_partners = up
        self.downstream_partners = down
        self.unique_types_upstream = 3
        self.unique_types_downstream = 0
        self.untyped_upstream_2hop = None
        self.untyped_downstream_2hop = {"untyped": 1}


def _install_fake_fnc(monkeypatch, method_name, result, calls):
    """Patch coana.FindNeuronConnection so deferred imports pick up the fake."""

    class FakeFNC:
        def __init__(self, dataset=None, server=None, token=None,
                     use_cache=True, verbose_mode=None, **kwargs):
            self.init_kwargs = {
                "dataset": dataset,
                "server": server,
                "token": token,
                "use_cache": use_cache,
                "verbose_mode": verbose_mode,
            }
            calls["instances"].append(self)

        def __getattr__(self, name):
            # Only the method under test should ever be invoked.
            if name == method_name:
                def method(**kwargs):
                    calls["kwargs"] = kwargs
                    return result
                return method
            raise AttributeError(name)

    monkeypatch.setattr(coana, "FindNeuronConnection", FakeFNC)


# ===========================================================================
# build_connection_cache.py
# ===========================================================================

def test_bcc_build_cache_delegates_to_fnc(monkeypatch, capsys):
    calls = {"instances": []}
    result = {
        "total_neurons": 3,
        "total_connections": 2,
        "cached_neurons": ["1", "2"],
        "failed_neurons": [],
        "elapsed_time": 1.5,
    }
    _install_fake_fnc(monkeypatch, "build_connection_cache", result, calls)

    out = bcc.build_cache(
        dataset="hemibrain:v1.2.1",
        token="tok",
        batch_size=5,
        neuron_types=["Mi1", "T4a"],
    )

    assert out is result
    inst = calls["instances"][0]
    assert inst.init_kwargs["dataset"] == "hemibrain:v1.2.1"
    assert inst.init_kwargs["token"] == "tok"
    assert inst.init_kwargs["use_cache"] is True
    assert inst.init_kwargs["verbose_mode"] == "full"
    assert calls["kwargs"] == {"neuron_types": ["Mi1", "T4a"], "batch_size": 5}
    text = capsys.readouterr().out
    assert "Neuron types: ['Mi1', 'T4a']" in text
    assert "Cache location: cache/hemibrain_v1_2_1/" in text


def test_bcc_build_cache_all_types_banner(monkeypatch, capsys):
    calls = {"instances": []}
    _install_fake_fnc(monkeypatch, "build_connection_cache", {}, calls)
    bcc.build_cache(dataset="flywire_FAFB_v783", token="tok")
    assert "Neuron types: ALL" in capsys.readouterr().out


def test_bcc_show_stats_missing_cache(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bcc, "__file__", str(tmp_path / "src" / "build_connection_cache.py"))
    bcc.show_stats("hemibrain:v1.2.1")
    out = capsys.readouterr().out
    assert "[ERROR] Connection cache not found" in out


def test_bcc_show_stats_reports_numbers(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bcc, "__file__", str(tmp_path / "src" / "build_connection_cache.py"))
    cache_dir = tmp_path / "cache" / "hemibrain_v1_2_1"
    cache_dir.mkdir(parents=True)
    pd.DataFrame({
        "bodyId_pre": ["1", "1", "2"],
        "bodyId_post": ["2", "3", "3"],
        "weight": [3, 4, 5],
    }).to_parquet(cache_dir / "connections.parquet", index=False)

    index_dir = tmp_path / "neuron_indexes" / "hemibrain_v1_2_1"
    index_dir.mkdir(parents=True)
    pd.DataFrame({
        "bodyId": ["1", "2", "3"],
        "type": ["A", "B", "C"],
        "downstream_complete": [True, True, False],
    }).to_parquet(index_dir / "neuron_index.parquet", index=False)

    bcc.show_stats("hemibrain:v1.2.1")
    out = capsys.readouterr().out
    assert "Total connections: 3" in out
    assert "Unique upstream neurons: 2" in out
    assert "Unique downstream neurons: 2" in out
    assert "Total synapse count: 12" in out
    assert "Total neurons indexed: 3" in out
    assert "Fully cached neurons: 2 (66.7%)" in out


def test_bcc_show_stats_corrupt_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bcc, "__file__", str(tmp_path / "src" / "build_connection_cache.py"))
    cache_dir = tmp_path / "cache" / "hemibrain_v1_2_1"
    cache_dir.mkdir(parents=True)
    (cache_dir / "connections.parquet").write_text("not parquet")
    index_dir = tmp_path / "neuron_indexes" / "hemibrain_v1_2_1"
    index_dir.mkdir(parents=True)
    (index_dir / "neuron_index.parquet").write_text("not parquet either")

    bcc.show_stats("hemibrain:v1.2.1")
    out = capsys.readouterr().out
    assert "[ERROR] Could not read connection cache" in out
    assert "[WARNING] Could not read neuron index" in out


def test_bcc_main_stats_mode(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bcc, "__file__", str(tmp_path / "src" / "build_connection_cache.py"))
    monkeypatch.setattr(sys, "argv", ["build_connection_cache.py", "--stats", "hemibrain:v1.2.1"])
    bcc.main()
    assert "[ERROR] Connection cache not found" in capsys.readouterr().out


def test_bcc_main_build_mode_parses_arguments(monkeypatch, capsys):
    calls = {"instances": []}
    _install_fake_fnc(monkeypatch, "build_connection_cache", {}, calls)
    monkeypatch.setattr(sys, "argv", [
        "build_connection_cache.py", "flywire_FAFB_v783",
        "--token", "tok", "--server", "https://example.invalid",
        "--batch-size", "7", "--types", "Mi1", "T4a",
    ])
    bcc.main()
    inst = calls["instances"][0]
    assert inst.init_kwargs["dataset"] == "flywire_FAFB_v783"
    assert inst.init_kwargs["token"] == "tok"
    assert inst.init_kwargs["server"] == "https://example.invalid"
    assert calls["kwargs"] == {"neuron_types": ["Mi1", "T4a"], "batch_size": 7}


# ===========================================================================
# build_connectivity_profile_cache.py
# ===========================================================================

def test_bcp_build_cache_with_progress_and_sample(monkeypatch, capsys):
    calls = {"instances": []}
    result = {"total_profiles": 1, "profiles": {"TypeA": _FakeProfile(up={"u": 1})}}
    _install_fake_fnc(
        monkeypatch, "build_connectivity_profile_cache", result, calls)

    # Exercise every progress_callback branch via the stored kwargs.
    real_method_calls = []

    def capture(**kwargs):
        real_method_calls.append(kwargs)
        callback = kwargs["progress_callback"]
        callback(0, 0, "")                       # total == 0 guard
        callback(1, 2, "X" * 40)                 # long type truncation
        callback(2, 2, None)                     # falsy type
        return result

    inst_cls = coana.FindNeuronConnection

    class CallbackFNC(inst_cls):
        def __getattr__(self, name):
            if name == "build_connectivity_profile_cache":
                return capture
            raise AttributeError(name)

    monkeypatch.setattr(coana, "FindNeuronConnection", CallbackFNC)

    out_result = bcp.build_cache(
        dataset="hemibrain:v1.2.1",
        token="tok",
        top_k=15,
        top_m=8,
        expand_2hop=False,
        max_neurons=10,
        neuron_types=["TypeA"],
        force=True,
    )

    assert out_result is result
    kwargs = real_method_calls[0]
    assert kwargs["neuron_types"] == ["TypeA"]
    assert kwargs["top_k"] == 15
    assert kwargs["top_m"] == 8
    assert kwargs["expand_2hop"] is False
    assert kwargs["max_neurons"] == 10
    assert kwargs["force_refresh"] is True
    text = capsys.readouterr().out
    assert "Sample Profile: TypeA" in text
    assert "Upstream partners: 1" in text
    assert "Downstream partners: 0" in text


def test_bcp_build_cache_env_token_and_empty_profiles(monkeypatch, capsys):
    monkeypatch.setenv("NEUPRINT_TOKEN", "envtok")
    calls = {"instances": []}
    _install_fake_fnc(
        monkeypatch, "build_connectivity_profile_cache",
        {"profiles": {}, "downstream_partners": None}, calls)

    bcp.build_cache(dataset="hemibrain:v1.2.1", token=None)

    assert calls["instances"][0].init_kwargs["token"] == "envtok"
    out = capsys.readouterr().out
    assert "[INFO] Using token from NEUPRINT_TOKEN environment variable" in out
    assert "Sample Profile" not in out


def test_bcp_build_cache_none_partners_print_zero(monkeypatch, capsys):
    calls = {"instances": []}
    result = {"profiles": {"T": _FakeProfile(up=None, down=None)}}
    _install_fake_fnc(
        monkeypatch, "build_connectivity_profile_cache", result, calls)
    bcp.build_cache(dataset="ds:v1", token="tok")
    out = capsys.readouterr().out
    assert "Upstream partners: 0" in out
    assert "Downstream partners: 0" in out


def _patch_profiler_module(monkeypatch, profiles):
    """Stub comparison.connectivity_profiler without importing the real tree."""

    class ProfilerConfig:
        def __init__(self, verbose=False):
            self.verbose = verbose

    class ConnectivityProfiler:
        def __init__(self, config):
            self.config = config

        def read_connectivity_profile_cache(self, dataset):
            if isinstance(profiles, Exception):
                raise profiles
            return profiles

    module = types.ModuleType("comparison.connectivity_profiler")
    module.ConnectivityProfiler = ConnectivityProfiler
    module.ProfilerConfig = ProfilerConfig
    monkeypatch.setitem(sys.modules, "comparison", types.ModuleType("comparison"))
    monkeypatch.setitem(sys.modules, "comparison.connectivity_profiler", module)


def test_bcp_show_stats_missing_cache(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        bcp, "__file__",
        str(tmp_path / "src" / "build_connectivity_profile_cache.py"))
    bcp.show_stats("hemibrain:v1.2.1")
    assert "[ERROR] Cache not found" in capsys.readouterr().out


def _make_profile_cache_file(tmp_path):
    monkey_dir = tmp_path / "cache" / "hemibrain_v1_2_1"
    monkey_dir.mkdir(parents=True)
    cache_path = monkey_dir / "connectivity_profiles.parquet"
    pd.DataFrame({"stub": [1]}).to_parquet(cache_path, index=False)
    return cache_path


def test_bcp_show_stats_reports_profile_stats(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        bcp, "__file__",
        str(tmp_path / "src" / "build_connectivity_profile_cache.py"))
    _make_profile_cache_file(tmp_path)
    _patch_profiler_module(monkeypatch, {
        "TypeA": _FakeProfile(up={"u1": 5, "u2": 3}, down={"d1": 2}),
        "TypeB": _FakeProfile(up={"u1": 1}, down=None),
    })

    bcp.show_stats("hemibrain:v1.2.1")
    out = capsys.readouterr().out
    assert "Total profiles: 2" in out
    assert "Upstream partners: avg=1.5, max=2" in out
    assert "Profiles with 2-hop data: 2 (100.0%)" in out
    assert "TypeA" in out and "TypeB" in out


def test_bcp_show_stats_profiler_failure_warns(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        bcp, "__file__",
        str(tmp_path / "src" / "build_connectivity_profile_cache.py"))
    _make_profile_cache_file(tmp_path)
    _patch_profiler_module(monkeypatch, RuntimeError("profile read exploded"))

    bcp.show_stats("hemibrain:v1.2.1")
    assert "[WARNING] Could not read profiles: profile read exploded" in (
        capsys.readouterr().out)


def test_bcp_main_stats_mode(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        bcp, "__file__",
        str(tmp_path / "src" / "build_connectivity_profile_cache.py"))
    monkeypatch.setattr(
        sys, "argv",
        ["build_connectivity_profile_cache.py", "--stats", "hemibrain:v1.2.1"])
    bcp.main()
    assert "[ERROR] Cache not found" in capsys.readouterr().out


def test_bcp_main_build_mode_parses_arguments(monkeypatch):
    calls = {"instances": []}
    _install_fake_fnc(
        monkeypatch, "build_connectivity_profile_cache", {"profiles": {}}, calls)
    monkeypatch.setattr(sys, "argv", [
        "build_connectivity_profile_cache.py", "ds:v1",
        "-k", "5", "-m", "2", "--no-expand-2hop",
        "--max-neurons", "3", "--types", "A", "B", "--force",
    ])
    bcp.main()
    kwargs = calls["kwargs"]
    assert kwargs["top_k"] == 5
    assert kwargs["top_m"] == 2
    assert kwargs["expand_2hop"] is False
    assert kwargs["max_neurons"] == 3
    assert kwargs["neuron_types"] == ["A", "B"]
    assert kwargs["force_refresh"] is True


# ===========================================================================
# build_seed_indexes.py
# ===========================================================================

def _write_metadata(root: Path, dataset: str, csv_text: str) -> Path:
    folder = dataset.replace(":", "_").replace(".", "_")
    dataset_dir = root / "datasets" / folder
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / f"{folder}_allneurons_neuron_df.csv"
    path.write_text(csv_text, encoding="utf-8")
    return path


_GOOD_METADATA = (
    "bodyId,type,instance,post,class,roiInfo\n"
    "102,T2a,T2a_R,9,Excitatory,[3]\n"
    "101,T1a,T1a_L,5,Inhibitory,[1]\n"
)


def test_atomic_write_parquet_roundtrip(tmp_path):
    import polars as pl

    target = tmp_path / "nested" / "index.parquet"
    frame = pl.DataFrame({"bodyId": ["1", "2"], "post": [0, 3]})
    bsi._atomic_write_parquet(frame, target)

    assert target.is_file()
    assert pl.read_parquet(target)["post"].to_list() == [0, 3]
    assert not list(tmp_path.rglob("*.tmp-*"))


def test_seed_frame_zeroes_cache_state_and_drops_roi_payloads(tmp_path):
    source = _write_metadata(tmp_path, "test:v1", _GOOD_METADATA)

    frame = bsi._seed_frame(source)

    columns = frame.columns
    assert columns[:3] == ["bodyId", "type", "instance"]
    assert columns[-3:] == ["downstream_complete", "last_fetched", "connection_count"]
    assert "roiInfo" not in columns
    assert frame["downstream_complete"].to_list() == [False, False]
    assert frame["last_fetched"].to_list() == ["", ""]
    assert frame["connection_count"].to_list() == [0, 0]
    # Source values survive the projection.
    posts = dict(zip(frame["bodyId"].to_list(), frame["post"].to_list()))
    assert posts == {"101": 5, "102": 9}


def test_build_seed_index_writes_index_and_sidecar(tmp_path, monkeypatch):
    import polars as pl

    monkeypatch.setattr(bsi, "_PROJECT_ROOT", tmp_path)
    _write_metadata(tmp_path, "test:v1", _GOOD_METADATA)
    index_dir = tmp_path / "neuron_indexes"

    entry = bsi.build_seed_index("test:v1", index_dir)

    assert entry is not None
    assert entry["dataset"] == "test:v1"
    assert entry["folder"] == "test_v1"
    assert entry["rows"] == 2
    assert entry["index_bytes"] > 0 and entry["search_bytes"] > 0

    index_path = index_dir / "test_v1" / "neuron_index.parquet"
    search_path = index_dir / "test_v1" / "neuron_index_search.parquet"
    assert index_path.is_file() and search_path.is_file()

    frame = pl.read_parquet(index_path)
    assert frame["downstream_complete"].to_list() == [False, False]
    assert frame["connection_count"].to_list() == [0, 0]

    search = pl.read_parquet(search_path)
    assert {"__neuron_rows", "search_column", "search_priority",
            "search_value", "search_value_folded"}.issubset(search.columns)
    assert "type" in search["search_column"].to_list()


def test_build_seed_index_missing_metadata(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bsi, "_PROJECT_ROOT", tmp_path)
    assert bsi.build_seed_index("ghost:v9", tmp_path / "neuron_indexes") is None
    assert "No local metadata table" in capsys.readouterr().out


def test_build_seed_index_rejects_duplicate_bodyids(tmp_path, monkeypatch):
    monkeypatch.setattr(bsi, "_PROJECT_ROOT", tmp_path)
    _write_metadata(
        tmp_path, "test:v1",
        "bodyId,type,instance,post\n1,T,T_L,1\n1,T,T_L,1\n")
    with pytest.raises(ValueError, match="duplicate bodyIds"):
        bsi.build_seed_index("test:v1", tmp_path / "neuron_indexes")


def test_read_manifest_variants(tmp_path):
    assert bsi._read_manifest(tmp_path) == {}

    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
    assert bsi._read_manifest(tmp_path) == {}

    (tmp_path / "manifest.json").write_text('{"a": 1}', encoding="utf-8")
    assert bsi._read_manifest(tmp_path) == {"a": 1}


def test_main_builds_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(bsi, "_PROJECT_ROOT", tmp_path)
    _write_metadata(tmp_path, "test:v1", _GOOD_METADATA)
    monkeypatch.setattr(sys, "argv", ["build_seed_indexes.py", "--datasets", "test:v1,ghost:v1"])

    assert bsi.main() == 0

    manifest = json.loads(
        (tmp_path / "neuron_indexes" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["index_dir"] == "neuron_indexes"
    assert list(manifest["datasets"].keys()) == ["test:v1"]
    assert manifest["datasets"]["test:v1"]["rows"] == 2
    assert (tmp_path / "neuron_indexes" / "test_v1" / "neuron_index.parquet").is_file()


def test_main_defaults_to_seed_datasets_without_metadata(tmp_path, monkeypatch, capsys):
    # No datasets/ dir at all: every seed returns None -> nothing rebuilt.
    monkeypatch.setattr(bsi, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["build_seed_indexes.py"])
    assert bsi.main() == 1
    assert "Nothing was rebuilt." in capsys.readouterr().out


def test_main_returns_one_on_build_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bsi, "_PROJECT_ROOT", tmp_path)
    _write_metadata(
        tmp_path, "test:v1",
        "bodyId,type,instance,post\n1,T,T_L,1\n1,T,T_L,1\n")
    monkeypatch.setattr(sys, "argv", ["build_seed_indexes.py", "--datasets", "test:v1"])
    assert bsi.main() == 1
    assert "Failed:" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Extra edge branches
# ---------------------------------------------------------------------------

def test_bcc_build_cache_survives_missing_token_manager(monkeypatch):
    """ImportError in the token_manager helper must fall back to the raw token."""
    calls = {"instances": []}
    _install_fake_fnc(monkeypatch, "build_connection_cache", {}, calls)
    # A sys.modules entry set to None makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "utils.token_manager", None)

    bcc.build_cache(dataset="hemibrain:v1.2.1", token="direct-tok")

    assert calls["instances"][0].init_kwargs["token"] == "direct-tok"


def test_atomic_write_parquet_cleans_temp_on_replace_failure(tmp_path, monkeypatch):
    import os
    import polars as pl

    target = tmp_path / "index.parquet"
    frame = pl.DataFrame({"bodyId": ["1"]})

    def raiser(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", raiser)
    with pytest.raises(OSError, match="disk full"):
        bsi._atomic_write_parquet(frame, target)
    # The finally block must sweep the orphaned temp file.
    assert not list(tmp_path.glob("*.tmp-*"))


def test_atomic_write_parquet_swallows_cleanup_failure(tmp_path, monkeypatch):
    import os
    import polars as pl

    target = tmp_path / "index.parquet"
    frame = pl.DataFrame({"bodyId": ["1"]})

    def replace_raiser(src, dst):
        raise OSError("disk full")

    def remove_raiser(path):
        raise OSError("cannot delete")

    monkeypatch.setattr(os, "replace", replace_raiser)
    monkeypatch.setattr(os, "remove", remove_raiser)
    # The cleanup failure is swallowed; the original error still propagates.
    with pytest.raises(OSError, match="disk full"):
        bsi._atomic_write_parquet(frame, target)


def test_seed_frame_resets_existing_progress_columns(tmp_path, monkeypatch):
    """A projection already carrying progress flags must be re-zeroed."""
    import polars as pl

    frame = pl.DataFrame({
        "bodyId": ["101"],
        "type": ["T1a"],
        "instance": ["T1a_L"],
        "post": [5],
        "downstream_complete": [True],
        "last_fetched": ["2024-01-01"],
        "connection_count": [9],
    })
    monkeypatch.setattr(bsi, "read_metadata_projection", lambda path: frame)

    result = bsi._seed_frame(tmp_path / "unused.csv")

    assert result["downstream_complete"].to_list() == [False]
    assert result["last_fetched"].to_list() == [""]
    assert result["connection_count"].to_list() == [0]
    assert result["post"].to_list() == [5]


def test_seed_frame_defaults_missing_post_column(tmp_path):
    source = _write_metadata(
        tmp_path, "test:v1", "bodyId,type,instance\n101,T1a,T1a_L\n")

    frame = bsi._seed_frame(source)

    assert frame["post"].to_list() == [0]
    assert frame["bodyId"].to_list() == ["101"]


def test_build_seed_index_rejects_unusable_bodyids(tmp_path, monkeypatch):
    import polars as pl

    monkeypatch.setattr(bsi, "_PROJECT_ROOT", tmp_path)
    _write_metadata(tmp_path, "test:v1", _GOOD_METADATA)
    empty = pl.DataFrame({"bodyId": [None]}, schema={"bodyId": pl.Utf8})
    monkeypatch.setattr(bsi, "_seed_frame", lambda source: empty)

    with pytest.raises(ValueError, match="no usable bodyIds"):
        bsi.build_seed_index("test:v1", tmp_path / "neuron_indexes")


def test_build_seed_index_rejects_incompatible_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(bsi, "_PROJECT_ROOT", tmp_path)
    _write_metadata(tmp_path, "test:v1", _GOOD_METADATA)
    monkeypatch.setattr(
        bsi, "is_search_cache_compatible", lambda frame, columns: False)

    with pytest.raises(ValueError, match="failed validation"):
        bsi.build_seed_index("test:v1", tmp_path / "neuron_indexes")
