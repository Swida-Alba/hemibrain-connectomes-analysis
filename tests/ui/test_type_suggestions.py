"""Tests for the dataset type-name suggestion pools (ui/type_suggestions.py).

Covers the matching rules (type-first, expand-on-no-type-match, bodyId ->
instance hints) and the local pool sources (cache neuron_index.parquet
first, dataset tables as fallback, mtime-keyed caching).
"""

import polars as pl
import pytest

from ui.type_suggestions import (
    _POOL_CACHE, _folder_pools, entry_hint, get_dataset_pools,
    match_suggestions, suggestion_pool,
)

# Synthetic pools mirroring a cache neuron_index (type/instance/bodyId).
POOLS = {
    "type": [("APL", "type"), ("APL2", "type"), ("aMe12", "type"),
             ("MBON01", "type")],
    "instance": [("APL_1", "instance"), ("aMe12_1", "instance")],
    "bodyId": [("5813012345", "aMe12_1"), ("5813012346", "bodyId"),
               ("5813012399", "bodyId")],
}


@pytest.fixture(autouse=True)
def _clear_pool_cache():
    """Pools are cached in a module-level dict; keep tests independent."""
    _POOL_CACHE.clear()
    yield
    _POOL_CACHE.clear()


@pytest.fixture
def local_dirs(tmp_path, monkeypatch):
    """Point the pool sources at a temporary cache/datasets pair."""
    cache = tmp_path / "cache"
    datasets = tmp_path / "datasets"
    cache.mkdir(parents=True)
    datasets.mkdir(parents=True)
    monkeypatch.setattr("ui.type_suggestions._CACHE_DIR", cache)
    monkeypatch.setattr("ui.type_suggestions._DATASETS_DIR", datasets)
    return cache, datasets


class TestMatchSuggestions:
    def test_empty_input_or_pools(self):
        assert match_suggestions("", POOLS) == []
        assert match_suggestions("   ", POOLS) == []
        assert match_suggestions("aMe", {}) == []

    def test_numeric_input_auto_scopes_to_bodyid(self):
        out = match_suggestions("58130123", POOLS)
        assert [v for v, _ in out] == ["5813012345", "5813012346", "5813012399"]
        # The hint is the instance for known bodyIds, else the column name.
        assert out[0][1] == "aMe12_1"
        assert out[1][1] == "bodyId"

    def test_string_input_type_first(self):
        assert match_suggestions("ap", POOLS) == [("APL", "type"), ("APL2", "type")]

    def test_string_input_case_insensitive_prefix(self):
        assert match_suggestions("APL", POOLS)[0] == ("APL", "type")
        assert match_suggestions("Apl2", POOLS) == [("APL2", "type")]

    def test_expand_only_when_no_type_match_and_auto(self):
        # "aMe12_" matches no type -> the range expands to instance.
        assert match_suggestions("aMe12_", POOLS) == [("aMe12_1", "instance")]
        # "aMe12" itself DOES match a type -> type stays first and alone.
        assert match_suggestions("aMe12", POOLS) == [("aMe12", "type")]

    def test_expansion_order_instance_then_bodyid_then_extra(self):
        extras = dict(POOLS)
        extras["bodyId"] = extras["bodyId"] + [("FL5813", "bodyId")]
        extras["flywireType"] = [("FLaMe12_x", "flywireType")]
        out = match_suggestions("FL", extras)
        # no type match -> instance, then bodyId, then extra type columns
        assert out == [("FL5813", "bodyId"), ("FLaMe12_x", "flywireType")]

    def test_explicit_scope_restricts_column(self):
        assert match_suggestions("5813", POOLS, "type") == []
        assert match_suggestions("mbon", POOLS, "instance") == []
        assert match_suggestions("mbon", POOLS, "bodyId") == []
        assert match_suggestions("APL", POOLS, "type") == [("APL", "type"), ("APL2", "type")]

    def test_scope_case_insensitive_and_unknown_falls_back_to_auto(self):
        assert match_suggestions("APL", POOLS, "TYPE") == [("APL", "type"), ("APL2", "type")]
        assert match_suggestions("ap", POOLS, "bogus") == [("APL", "type"), ("APL2", "type")]

    def test_limit(self):
        assert len(match_suggestions("5813", POOLS, limit=2)) == 2
        assert len(match_suggestions("ap", POOLS, limit=1)) == 1


class TestEntryHint:
    def test_type_instance_bodyid_priority(self):
        assert entry_hint("APL", POOLS) == "type"
        assert entry_hint("APL_1", POOLS) == "instance"
        assert entry_hint("5813012345", POOLS) == "bodyId"

    def test_unknown_value(self):
        assert entry_hint("nope", POOLS) is None
        assert entry_hint("", POOLS) is None


class TestPoolSources:
    def test_index_parquet_pools(self, local_dirs):
        """cache/<folder>/neuron_index.parquet builds type/instance/bodyId
        pools (bodyId hints = the corresponding instance)."""
        cache, _ = local_dirs
        folder = "test_v1"
        (cache / folder).mkdir()
        pl.DataFrame({
            "bodyId": [5813012345, 5813012346],
            "type": ["APL", "MBON01"],
            "instance": ["APL_1", "MBON01_1"],
        }).write_parquet(cache / folder / "neuron_index.parquet")

        pools = _folder_pools(folder)
        assert pools["type"] == [("APL", "type"), ("MBON01", "type")]
        assert pools["bodyId"] == [
            ("5813012345", "APL_1"), ("5813012346", "MBON01_1"),
        ]

    def test_table_fallback_pools(self, local_dirs):
        """datasets/<folder> neuron tables (no cache index) build pools
        including extra type-like columns (e.g. flywireType)."""
        cache, datasets = local_dirs
        folder = "test_v1"
        ds_dir = datasets / folder
        ds_dir.mkdir()
        (ds_dir / "test_neuron_df.csv").write_text(
            "bodyId,type,instance,flywireType\n"
            "1,APL,APL_1,FW_APL\n"
            "2,MBON01,MBON01_1,FW_MBON\n"
        )

        pools = _folder_pools(folder)
        assert pools["type"] == [("APL", "type"), ("MBON01", "type")]
        assert pools["flywireType"] == [("FW_APL", "flywireType"), ("FW_MBON", "flywireType")]
        assert pools["bodyId"] == [("1", "APL_1"), ("2", "MBON01_1")]

    def test_cache_index_preferred_over_table(self, local_dirs):
        cache, datasets = local_dirs
        folder = "test_v1"
        (cache / folder).mkdir()
        pl.DataFrame({
            "bodyId": [1], "type": ["APL"], "instance": ["APL_1"],
        }).write_parquet(cache / folder / "neuron_index.parquet")
        ds_dir = datasets / folder
        ds_dir.mkdir()
        (ds_dir / "x_neuron_df.csv").write_text(
            "bodyId,type,instance\n1,MBON01,MBON01_1\n"
        )

        assert _folder_pools(folder)["type"] == [("APL", "type")]

    def test_missing_folder_returns_empty(self, local_dirs, monkeypatch):
        assert get_dataset_pools("hemibrain:v1.2.1") == {}
        assert get_dataset_pools("") == {}

    def test_pool_cache_invalidates_on_file_change(self, local_dirs):
        """Rewriting the index file (new mtime) refreshes the cached pools."""
        cache, _ = local_dirs
        folder = "test_v1"
        (cache / folder).mkdir()
        index = cache / folder / "neuron_index.parquet"
        pl.DataFrame({
            "bodyId": [1], "type": ["APL"], "instance": ["APL_1"],
        }).write_parquet(index)
        assert _folder_pools(folder)["type"] == [("APL", "type")]

        import time
        time.sleep(0.01)  # guarantee a different mtime
        pl.DataFrame({
            "bodyId": [1], "type": ["MBON01"], "instance": ["MBON01_1"],
        }).write_parquet(index)
        assert _folder_pools(folder)["type"] == [("MBON01", "type")]


class TestSuggestionPool:
    def test_union_dedup_across_datasets(self, monkeypatch):
        a = {"type": [("APL", "type"), ("MBON01", "type")]}
        b = {"type": [("APL", "type"), ("KC", "type")],
             "instance": [("KC_1", "instance")]}
        monkeypatch.setattr(
            "ui.type_suggestions.get_dataset_pools",
            lambda ds: a if ds == "A" else b,
        )
        assert suggestion_pool(["A", "B"]) == {
            "type": [("APL", "type"), ("MBON01", "type"), ("KC", "type")],
            "instance": [("KC_1", "instance")],
        }
