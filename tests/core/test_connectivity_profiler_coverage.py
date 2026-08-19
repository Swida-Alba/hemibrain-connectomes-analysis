"""Coverage tests for comparison.connectivity_profiler.

Hermetic: no network access. All data access methods are monkeypatched and
synthetic DataFrames/dicts are injected. All file I/O goes through tmp_path.
"""

import numpy as np
import pandas as pd
import pytest

import comparison.connectivity_profiler as cp_module
from comparison.connectivity_profiler import (
    ConnectivityProfile,
    ConnectivityProfiler,
    ConnectivityStatus,
    DataNotAvailableError,
    FuzzyMatchConfig,
    ProfilerConfig,
    clear_profiler_conn_cache,
    compute_ranks,
    get_profiler_conn_cache_info,
    normalize_partner_type,
)

DS = 'flywire_FAFB_v783'


def _profile(bid, ds=DS, upstream=None, downstream=None, **kwargs):
    """Build a ConnectivityProfile from synthetic partner dicts."""
    upstream = upstream or {}
    downstream = downstream or {}
    params = dict(
        neuron_id=bid,
        dataset=ds,
        upstream_partners=upstream,
        downstream_partners=downstream,
        upstream_ranks=compute_ranks(upstream),
        downstream_ranks=compute_ranks(downstream),
        total_upstream_weight=float(sum(upstream.values())),
        total_downstream_weight=float(sum(downstream.values())),
    )
    params.update(kwargs)
    return ConnectivityProfile(**params)


@pytest.fixture
def profiler(tmp_path):
    return ConnectivityProfiler(
        [DS], config=ProfilerConfig(), cache_dir=str(tmp_path), verbose=False
    )


# ---------------------------------------------------------------------------
# ConnectivityStatus enum
# ---------------------------------------------------------------------------

def test_connectivity_status_flags():
    assert ConnectivityStatus.COMPLETE.is_valid_for_comparison()
    assert ConnectivityStatus.INCOMPLETE.is_valid_for_comparison()
    assert not ConnectivityStatus.NONE.is_valid_for_comparison()
    assert not ConnectivityStatus.ORPHAN.is_valid_for_comparison()
    assert ConnectivityStatus.RARE.requires_warning()
    assert not ConnectivityStatus.COMPLETE.requires_warning()
    assert ConnectivityStatus.COMPLETE.is_complete()
    assert not ConnectivityStatus.INCOMPLETE.is_complete()
    assert 'Complete' in ConnectivityStatus.get_description(ConnectivityStatus.COMPLETE)
    assert ConnectivityStatus.get_description(None) == 'Unknown'


# ---------------------------------------------------------------------------
# Module-level connection cache helpers
# ---------------------------------------------------------------------------

def test_module_conn_cache_info_and_clear():
    clear_profiler_conn_cache()
    assert get_profiler_conn_cache_info() == {}
    cp_module._PROFILER_CONN_CACHE['test_ds'] = {
        'conn_df': pd.DataFrame({'a': [1, 2, 3]}),
        'type_lookup': {'x': 'y'},
    }
    cp_module._PROFILER_CONN_CACHE['empty_ds'] = {'conn_df': None, 'type_lookup': {}}
    info = get_profiler_conn_cache_info()
    assert info['test_ds']['conn_df_rows'] == 3
    assert info['test_ds']['type_lookup_size'] == 1
    assert info['empty_ds']['conn_df_rows'] == 0
    clear_profiler_conn_cache()
    assert get_profiler_conn_cache_info() == {}


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = ProfilerConfig()
    assert cfg.top_k_bodyid == 5
    assert cfg.top_m_type == 0
    assert cfg.normalize_method == 'rank'
    assert cfg.use_cache is True
    assert cfg.expand_untyped_2hop is True
    assert isinstance(cfg.fuzzy_match, FuzzyMatchConfig)
    fuzzy = FuzzyMatchConfig()
    assert fuzzy.enabled is False
    assert fuzzy.custom_mappings == {}


# ---------------------------------------------------------------------------
# normalize_partner_type
# ---------------------------------------------------------------------------

def test_normalize_partner_type_disabled():
    cfg = FuzzyMatchConfig(enabled=False)
    assert normalize_partner_type('T4a_R', cfg) == 'T4a_R'
    assert normalize_partner_type('', cfg) == ''
    assert normalize_partner_type(None, cfg) == ''
    # NaN round-trips to the string 'nan' when matching is disabled
    assert normalize_partner_type(np.nan, cfg) == 'nan'


def test_normalize_partner_type_case_insensitive():
    cfg = FuzzyMatchConfig(enabled=True, case_insensitive=True)
    assert normalize_partner_type('T4A', cfg) == 't4a'


def test_normalize_partner_type_strip_lr():
    cfg = FuzzyMatchConfig(enabled=True, strip_lr_suffix=True)
    assert normalize_partner_type('T4a_R', cfg) == 'T4a'
    assert normalize_partner_type('T4a-left', cfg) == 'T4a'
    assert normalize_partner_type('T4a', cfg) == 'T4a'


def test_normalize_partner_type_strip_numeric_and_like():
    cfg = FuzzyMatchConfig(enabled=True, strip_numeric_suffix=True)
    assert normalize_partner_type('aMe12_1', cfg) == 'aMe12'
    assert normalize_partner_type('aMe12-2', cfg) == 'aMe12'
    cfg_like = FuzzyMatchConfig(enabled=True, strip_like_suffix=True)
    assert normalize_partner_type('T4a-like', cfg_like) == 'T4a'
    assert normalize_partner_type('T4a_LIKE', cfg_like) == 'T4a'


def test_normalize_partner_type_custom_mappings():
    cfg = FuzzyMatchConfig(enabled=True, custom_mappings={'T4a_R': 'T4a'})
    assert normalize_partner_type('T4a_R', cfg) == 'T4a'
    # Custom mapping applied AFTER normalization as well
    cfg2 = FuzzyMatchConfig(enabled=True, case_insensitive=True,
                            custom_mappings={'t4a': 'T4'})
    assert normalize_partner_type('T4A', cfg2) == 'T4'


# ---------------------------------------------------------------------------
# compute_ranks
# ---------------------------------------------------------------------------

def test_compute_ranks():
    assert compute_ranks({}) == {}
    assert compute_ranks({'a': 1.0}) == {'a': 1}
    ranks = compute_ranks({'a': 10.0, 'b': 10.0, 'c': 5.0, 'd': 1.0})
    assert ranks['a'] == 1
    assert ranks['b'] == 1
    assert ranks['c'] == 3
    assert ranks['d'] == 4


# ---------------------------------------------------------------------------
# ConnectivityProfile: status classification
# ---------------------------------------------------------------------------

def test_profile_status_none():
    p = _profile('n1')
    assert p.connectivity_status is ConnectivityStatus.NONE
    assert p.connectivity_status_str == 'none'
    assert p.is_weak_connectivity is True
    assert p.is_valid_for_comparison() is False


def test_profile_status_unidirectional():
    # explicit counts drive classification
    p2 = ConnectivityProfile(
        neuron_id='n2', dataset=DS,
        upstream_partners={'A': 1.0}, downstream_partners={},
        actual_upstream_count=0, actual_downstream_count=10,
        top_k_bodyid_used=5, top_m_type_target=0,
    )
    assert p2.connectivity_status is ConnectivityStatus.UNIDIRECTIONAL


def test_profile_status_rare():
    p = _profile('n3', upstream={'A': 1.0},
                 actual_upstream_count=3, actual_downstream_count=10,
                 top_k_bodyid_used=5, top_m_type_target=0)
    assert p.connectivity_status is ConnectivityStatus.RARE
    assert p.is_weak_connectivity is False  # RARE is valid (with warning)


def test_profile_status_incomplete():
    p = _profile('n4', upstream={'A': 1.0},
                 actual_upstream_count=6, actual_downstream_count=6,
                 top_k_bodyid_used=10, top_m_type_target=0)
    assert p.connectivity_status is ConnectivityStatus.INCOMPLETE


def test_profile_status_incomplete_expansion_and_sparse():
    p = _profile('n5', upstream={'A': 1.0},
                 actual_upstream_count=10, actual_downstream_count=10,
                 top_k_bodyid_used=10, top_m_type_target=5,
                 unique_types_upstream=2, unique_types_downstream=10)
    assert p.connectivity_status is ConnectivityStatus.INCOMPLETE_EXPANSION
    assert p.is_sparse is True


def test_profile_status_complete():
    p = _profile('n6', upstream={'A': 1.0},
                 actual_upstream_count=10, actual_downstream_count=10,
                 top_k_bodyid_used=10, top_m_type_target=0)
    assert p.connectivity_status is ConnectivityStatus.COMPLETE
    assert 'Status' in p.summary()


# ---------------------------------------------------------------------------
# ConnectivityProfile: serialization
# ---------------------------------------------------------------------------

def test_profile_to_dict_from_dict_roundtrip():
    p = ConnectivityProfile(
        neuron_id=123, dataset=DS, neuron_type='TypeX',
        upstream_partners={'A': 10.0}, downstream_partners={'B': 5.0},
        upstream_ranks={'A': 1}, downstream_ranks={'B': 1},
        total_upstream_weight=10.0, total_downstream_weight=5.0,
        partner_type_mapping_upstream={1: 'A'},
        partner_type_mapping_downstream={9: 'B'},
        untyped_upstream_bodyids={2: 3.0},
        untyped_downstream_bodyids={3: 1.0},
        untyped_upstream_2hop={2: {'Mi1': 4.0}},
        untyped_downstream_2hop={3: {'Tm1': 2.0}},
        untyped_upstream_2hop_ranks={2: {'Mi1': 1}},
        untyped_downstream_2hop_ranks={3: {'Tm1': 1}},
        typed_upstream_bodyids={1: 10.0},
        typed_downstream_bodyids={9: 5.0},
    )
    d = p.to_dict()
    assert d['neuron_id'] == '123'
    assert d['connectivity_status'] == p.connectivity_status_str
    p2 = ConnectivityProfile.from_dict(d)
    assert p2.partner_type_mapping_upstream == {1: 'A'}
    assert p2.untyped_upstream_bodyids == {2: 3.0}
    assert p2.untyped_upstream_2hop == {2: {'Mi1': 4.0}}
    assert p2.untyped_upstream_2hop_ranks == {2: {'Mi1': 1}}
    assert p2.typed_upstream_bodyids == {1: 10.0}
    assert p2.downstream_partners == {'B': 5.0}

    # from_dict tolerates missing optional keys
    minimal = {'neuron_id': 'x', 'dataset': DS}
    p3 = ConnectivityProfile.from_dict(minimal)
    assert p3.neuron_type is None


# ---------------------------------------------------------------------------
# ConnectivityProfile: accessors and vectors
# ---------------------------------------------------------------------------

def test_profile_accessors():
    p = _profile('acc', upstream={'A': 3.0, 'C': 1.0},
                 downstream={'B': 2.0, 'C': 4.0},
                 partner_type_mapping_upstream={1: 'A', 2: 'A', 3: 'C'})
    assert p.get_all_partner_types() == {'A', 'B', 'C'}
    assert p.get_partner_weight('A', 'upstream') == 3.0
    assert p.get_partner_weight('A', 'downstream') == 0.0
    assert p.get_partner_weight('C', 'both') == 5.0
    assert p.get_partner_bodyids('A', 'upstream') == [1, 2]
    assert p.get_partner_bodyids('A', 'downstream') == []

    props = p.get_proportions('upstream')
    assert props['A'] == pytest.approx(0.75)
    assert p.get_proportions('downstream')['C'] == pytest.approx(2.0 / 3.0)
    combined = p.get_proportions('both')
    assert sum(combined.values()) == pytest.approx(1.0)
    empty = _profile('empty')
    assert empty.get_proportions('upstream') == {}


def test_profile_weight_and_rank_vectors():
    p = _profile('vec', upstream={'A': 3.0, 'B': 4.0},
                 downstream={'A': 1.0})
    vocab = ['A', 'B', 'C']
    v = p.to_weight_vector(vocab, 'upstream', normalize=False)
    assert list(v) == [3.0, 4.0, 0.0]
    v_norm = p.to_weight_vector(vocab, 'upstream', normalize=True)
    assert np.linalg.norm(v_norm) == pytest.approx(1.0)
    v_both = p.to_weight_vector(vocab, 'both', normalize=False)
    assert list(v_both) == [4.0, 4.0, 0.0]
    # zero vector stays zero (no divide-by-zero)
    assert list(p.to_weight_vector(vocab, 'upstream', normalize=True) * 0) == [0.0, 0.0, 0.0]
    assert list(_profile('z').to_weight_vector(vocab, 'upstream', normalize=True)) == [0.0, 0.0, 0.0]

    rv = p.to_rank_vector(vocab, 'upstream')
    assert list(rv) == [2.0, 1.0, 3.0]  # default rank = K+1 = 3
    rv_d = p.to_rank_vector(vocab, 'upstream', default_rank=9)
    assert rv_d[2] == 9
    rv_both = p.to_rank_vector(vocab, 'both')
    assert rv_both.shape == (3,)


def test_profile_hybrid_rank_vector():
    p = ConnectivityProfile(
        neuron_id='h', dataset=DS,
        upstream_partners={'A': 5.0}, upstream_ranks={'A': 1},
        total_upstream_weight=5.0,
        untyped_upstream_bodyids={7: 2.0},
        untyped_upstream_2hop={7: {'X': 4.0}},
        untyped_upstream_2hop_ranks={7: {'X': 1}},
    )
    vocab = ['A', 'X', 'Z']
    cross = p.to_hybrid_rank_vector(vocab, direction='upstream', mode='cross_dataset')
    # A -> rank 1; X -> hop2 rank 1 + len(typed)=1 -> 2; Z -> default (k=2 -> 3)
    assert list(cross) == [1.0, 2.0, 3.0]
    intra = p.to_hybrid_rank_vector(vocab, direction='upstream', mode='intra_dataset')
    assert list(intra) == [1.0, 3.0, 3.0]


def test_profile_untyped_bodyid_ranks():
    p = ConnectivityProfile(
        neuron_id='u', dataset=DS,
        untyped_upstream_bodyids={10: 5.0, 20: 9.0},
    )
    assert p.get_untyped_bodyid_ranks('upstream') == {20: 1, 10: 2}
    assert p.get_untyped_bodyid_ranks('downstream') == {}


# ---------------------------------------------------------------------------
# ConnectivityProfile: static vocabulary / matrix helpers
# ---------------------------------------------------------------------------

def test_build_shared_vocabulary():
    p1 = _profile('a', upstream={'X': 1.0, 'Y': 2.0})
    p2 = _profile('b', upstream={'Y': 1.0, 'Z': 1.0}, downstream={'Y': 1.0})
    vocab = ConnectivityProfile.build_shared_vocabulary({'a': p1, 'b': p2}, 'upstream')
    assert vocab == ['X', 'Y', 'Z']
    vocab2 = ConnectivityProfile.build_shared_vocabulary(
        {'a': p1, 'b': p2}, 'upstream', min_occurrence=2)
    assert vocab2 == ['Y']
    vocab_both = ConnectivityProfile.build_shared_vocabulary({'b': p2}, 'both')
    assert 'Y' in vocab_both


def test_build_hybrid_vocabulary_includes_2hop():
    p = ConnectivityProfile(
        neuron_id='h', dataset=DS,
        upstream_partners={'A': 1.0},
        untyped_upstream_2hop={5: {'H2': 2.0}},
    )
    vocab = ConnectivityProfile.build_hybrid_vocabulary({'h': p}, 'upstream')
    assert vocab == ['A', 'H2']
    vocab_no2 = ConnectivityProfile.build_hybrid_vocabulary(
        {'h': p}, 'upstream', include_2hop=False)
    assert vocab_no2 == ['A']


def test_profiles_to_matrices():
    p1 = _profile('a', upstream={'X': 3.0, 'Y': 4.0})
    p2 = _profile('b', upstream={'Y': 1.0})
    matrix, ids, vocab = ConnectivityProfile.profiles_to_weight_matrix(
        {'a': p1, 'b': p2}, direction='upstream')
    assert matrix.shape == (2, 2)
    assert ids == ['a', 'b']
    assert vocab == ['X', 'Y']
    rmat, rids, rvocab = ConnectivityProfile.profiles_to_rank_matrix(
        {'a': p1, 'b': p2}, direction='upstream')
    assert rmat.shape == (2, 2)


def test_aggregate_bodyid_profiles():
    with pytest.raises(ValueError):
        ConnectivityProfile.aggregate_bodyid_profiles([], 'T', DS)
    p1 = _profile(1, upstream={'A': 2.0, 'B': 1.0},
                  untyped_upstream_count=1, untyped_upstream_weight_fraction=0.5)
    p2 = _profile(2, upstream={'A': 2.0, 'C': 3.0})
    agg = ConnectivityProfile.aggregate_bodyid_profiles([p1, p2], 'TypeT', DS)
    assert agg.neuron_type == 'TypeT'
    assert agg.num_neurons_aggregated == 2
    assert agg.upstream_partners['A'] == 4.0
    assert agg.upstream_ranks['A'] == 1
    assert agg.upstream_ranks['C'] == 2
    assert agg.untyped_upstream_count == 1


# ---------------------------------------------------------------------------
# ConnectivityProfiler: _process_connections
# ---------------------------------------------------------------------------

def _conn_df(rows):
    return pd.DataFrame(rows, columns=['partner_type', 'weight', 'partner_bodyId'])


def test_process_connections_empty_and_missing_column(profiler):
    res = profiler._process_connections(pd.DataFrame(), 'upstream', 5)
    partners, ranks, untyped_count, untyped_frac, total, actual, unique, mapping, k_used, ub, tb = res
    assert partners == {} and ranks == {} and k_used == 5

    df = pd.DataFrame({'weight': [1.0, 2.0]})
    res = profiler._process_connections(df, 'upstream', 3)
    assert res[0] == {} and res[8] == 3


def test_process_connections_typed(profiler):
    df = _conn_df([('A', 10.0, 1), ('B', 8.0, 2), ('A', 2.0, 3), ('C', 1.0, 4)])
    (partners, ranks, untyped_count, untyped_frac, total, actual,
     unique, mapping, k_used, ub, tb) = profiler._process_connections(df, 'upstream', 5)
    assert partners['A'] == pytest.approx(12.0)
    assert partners['B'] == pytest.approx(8.0)
    assert partners['C'] == pytest.approx(1.0)
    assert ranks['A'] == 1 and ranks['B'] == 2 and ranks['C'] == 3
    assert untyped_count == 0 and untyped_frac == 0.0
    assert total == pytest.approx(21.0)
    assert actual == 3 and unique == 3
    assert mapping[2] == 'B'
    assert tb[1] == pytest.approx(10.0)
    assert ub == {}


def test_process_connections_untyped_only(profiler):
    df = _conn_df([(None, 5.0, 1), (np.nan, 3.0, 2), ('', 2.0, 3)])
    (partners, ranks, untyped_count, untyped_frac, total, actual,
     unique, mapping, k_used, ub, tb) = profiler._process_connections(df, 'upstream', 5)
    assert partners == {}
    assert untyped_count == 3
    assert untyped_frac == pytest.approx(1.0)
    assert actual == 0 and unique == 0
    assert set(ub.keys()) == {1, 2, 3}
    assert tb == {}


def test_process_connections_include_untyped(tmp_path):
    cfg = ProfilerConfig(include_untyped_partners=True)
    profiler = ConnectivityProfiler([DS], config=cfg, cache_dir=str(tmp_path), verbose=False)
    df = _conn_df([('A', 5.0, 1), (None, 3.0, 2)])
    partners = profiler._process_connections(df, 'upstream', 5)[0]
    assert 'untyped' in partners
    assert partners['untyped'] == pytest.approx(3.0)


def test_process_connections_fuzzy(tmp_path):
    cfg = ProfilerConfig(fuzzy_match=FuzzyMatchConfig(enabled=True, case_insensitive=True))
    profiler = ConnectivityProfiler([DS], config=cfg, cache_dir=str(tmp_path), verbose=False)
    df = _conn_df([('T4a', 5.0, 1), ('t4a', 3.0, 2), ('Mi1', 1.0, 3)])
    partners, ranks, *_ = profiler._process_connections(df, 'upstream', 5)
    assert set(partners.keys()) == {'t4a', 'mi1'}
    assert partners['t4a'] == pytest.approx(8.0)


def test_process_connections_dynamic_expansion(tmp_path):
    cfg = ProfilerConfig(top_k_bodyid=2, top_m_type=4)
    profiler = ConnectivityProfiler([DS], config=cfg, cache_dir=str(tmp_path), verbose=False)
    rows = [(f'T{i}', float(10 - i), i + 1) for i in range(6)]
    (partners, ranks, uc, uf, tw, actual, unique, mapping,
     k_used, ub, tb) = profiler._process_connections(_conn_df(rows), 'upstream', 2)
    assert unique == 6
    assert k_used == 7  # stepped +5 past hit position
    assert len(partners) == 6


def test_process_connections_expansion_insufficient_types(tmp_path):
    cfg = ProfilerConfig(top_k_bodyid=2, top_m_type=10, max_expansion_factor=3)
    profiler = ConnectivityProfiler([DS], config=cfg, cache_dir=str(tmp_path), verbose=False)
    rows = [('A', 5.0, 1), ('A', 4.0, 2), ('B', 3.0, 3), ('B', 2.0, 4)]
    res = profiler._process_connections(_conn_df(rows), 'upstream', 2)
    partners, k_used = res[0], res[8]
    assert set(partners.keys()) == {'A', 'B'}
    assert k_used == 4  # min(max_k=6, n_rows=4)


def test_process_connections_no_dynamic_expansion(tmp_path):
    cfg = ProfilerConfig(dynamic_expansion=False)
    profiler = ConnectivityProfiler([DS], config=cfg, cache_dir=str(tmp_path), verbose=False)
    rows = [(f'T{i}', float(10 - i), i + 1) for i in range(8)]
    res = profiler._process_connections(_conn_df(rows), 'upstream', 3)
    assert len(res[0]) == 3 and res[8] == 3


# ---------------------------------------------------------------------------
# ConnectivityProfiler: get_profile with monkeypatched data access
# ---------------------------------------------------------------------------

def _patch_queries(monkeypatch, profiler, up_df, down_df, hop2=None):
    monkeypatch.setattr(profiler, 'ensure_data_available',
                        lambda dataset, raise_on_missing=True: True)
    monkeypatch.setattr(profiler, '_query_connections_local',
                        lambda neuron, dataset: (up_df, down_df))
    if hop2 is not None:
        monkeypatch.setattr(
            profiler, '_fetch_2hop_partners',
            lambda bids, dataset, direction: {b: hop2 for b in bids})


def test_get_profile_with_2hop_and_cache(profiler, monkeypatch):
    up = _conn_df([('A', 10.0, 1), ('B', 8.0, 2), ('C', 6.0, 3), (None, 4.0, 4)])
    down = _conn_df([('D', 5.0, 5), ('E', 3.0, 6)])
    _patch_queries(monkeypatch, profiler, up, down, hop2=({'H1': 2.0}, {'H1': 1}))

    prof = profiler.get_profile('MyType', DS)
    assert prof.neuron_type == 'MyType'
    assert set(prof.upstream_partners) == {'A', 'B', 'C'}
    assert prof.untyped_upstream_bodyids == {4: 4.0}
    assert prof.untyped_upstream_2hop == {4: {'H1': 2.0}}
    assert prof.untyped_upstream_2hop_ranks == {4: {'H1': 1}}
    assert prof.downstream_partners['D'] == pytest.approx(5.0)
    assert prof.top_k_bodyid_used == 5

    # memory cache hit on second call
    assert profiler.get_profile('MyType', DS) is prof
    # force_refresh bypasses cache
    assert profiler.get_profile('MyType', DS, force_refresh=True) is not prof


def test_get_profile_bodyid_writes_batch_file(profiler, monkeypatch, tmp_path):
    up = _conn_df([('A', 10.0, 1)])
    down = _conn_df([('B', 2.0, 2)])
    _patch_queries(monkeypatch, profiler, up, down)
    prof = profiler.get_profile(720575, DS)
    assert prof.neuron_id == 720575
    batch_dir = profiler._get_profile_batch_dir(DS)
    assert batch_dir.exists()
    assert any(batch_dir.iterdir())


def test_load_from_cache_required_top_k(profiler):
    p = _profile('cached', top_k_bodyid_used=5)
    profiler._save_to_cache(p)
    assert profiler._load_from_cache('cached', DS) is p
    assert profiler._load_from_cache('cached', DS, required_top_k=10) is None


def test_load_from_cache_missing(profiler):
    assert profiler._load_from_cache(999999, DS) is None
    assert profiler._load_cache_dataframe('nonexistent_dataset') is None


def test_save_to_cache_str_neuron_id_memory_only(profiler, tmp_path):
    p = _profile('type_level')
    profiler._save_to_cache(p)
    assert profiler._memory_cache[('type_level', DS)] is p
    batch_dir = profiler._get_profile_batch_dir(DS)
    assert (not batch_dir.exists()) or not any(batch_dir.iterdir())


def test_deferred_cache_writes_and_flush(profiler, tmp_path):
    profiler._defer_cache_writes = True
    p_int = _profile(424242)
    profiler._save_to_cache(p_int)
    assert '424242' in profiler._pending_cache_writes[DS]
    profiler.flush_pending_cache_writes(silent=True)
    assert profiler._pending_cache_writes.get(DS, {}) == {}

    # type-level profiles queued but skipped on flush
    profiler._pending_cache_writes = {DS: {'str_id': _profile('str_id')}}
    profiler.flush_pending_cache_writes(silent=False)
    assert profiler._pending_cache_writes.get(DS, {}) == {}

    # flush with nothing pending returns immediately
    profiler._pending_cache_writes = {}
    profiler.flush_pending_cache_writes()


def test_profile_row_roundtrip(profiler):
    p = _profile(555, upstream={'A': 3.0},
                 partner_type_mapping_upstream={1: 'A'},
                 typed_upstream_bodyids={1: 3.0})
    row = profiler._profile_to_row(p)
    assert isinstance(row['upstream_partners'], str)
    restored = profiler._row_to_profile(pd.Series(row))
    assert restored.upstream_partners == {'A': 3.0}
    # JSON round-trip leaves mapping keys as strings in the raw row
    assert restored.partner_type_mapping_upstream == {'1': 'A'}


# ---------------------------------------------------------------------------
# ConnectivityProfiler: data availability / clients
# ---------------------------------------------------------------------------

def test_ensure_data_available_local_missing(profiler):
    with pytest.raises(DataNotAvailableError):
        profiler.ensure_data_available('flywire_missing_dataset_v9', raise_on_missing=True)
    assert profiler.ensure_data_available('flywire_missing_dataset_v9',
                                          raise_on_missing=False) is False


def test_ensure_data_available_neuprint_paths(profiler, monkeypatch):
    # client None -> raise / False
    monkeypatch.setattr(profiler, '_get_client_for_dataset', lambda d: None)
    with pytest.raises(DataNotAvailableError):
        profiler.ensure_data_available('hemibrain:v1.2.1', raise_on_missing=True)
    assert profiler.ensure_data_available('hemibrain:v1.2.1', raise_on_missing=False) is False

    # client present -> True
    monkeypatch.setattr(profiler, '_get_client_for_dataset', lambda d: object())
    assert profiler.ensure_data_available('hemibrain:v1.2.1') is True

    # client creation error -> raise / False
    def _boom(d):
        raise RuntimeError('no credentials')
    monkeypatch.setattr(profiler, '_get_client_for_dataset', _boom)
    with pytest.raises(DataNotAvailableError):
        profiler.ensure_data_available('hemibrain:v1.2.1')
    assert profiler.ensure_data_available('hemibrain:v1.2.1', raise_on_missing=False) is False


def test_get_data_status(profiler, monkeypatch):
    status = profiler.get_data_status(['flywire_missing_dataset_v9'])
    entry = status['flywire_missing_dataset_v9']
    assert entry['available'] is False
    assert entry['type'] == 'local'
    assert entry['error']

    monkeypatch.setattr(profiler, '_get_client_for_dataset', lambda d: object())
    status2 = profiler.get_data_status(['hemibrain:v1.2.1'])
    assert status2['hemibrain:v1.2.1']['available'] is True
    assert status2['hemibrain:v1.2.1']['type'] == 'neuprint'


def test_normalize_neuprint_dataset_name(profiler):
    assert profiler._normalize_neuprint_dataset_name('hemibrain:v1.2.1') == 'hemibrain:v1.2.1'
    assert profiler._normalize_neuprint_dataset_name('hemibrain_v1_2_1') == 'hemibrain:v1.2.1'
    assert profiler._normalize_neuprint_dataset_name('male_cns_v0_9') == 'male-cns:v0.9'
    assert profiler._normalize_neuprint_dataset_name('plainname') == 'plainname'


def test_get_client_for_dataset(profiler):
    # local dataset -> None (cached)
    assert profiler._get_client_for_dataset('flywire_FAFB_v783') is None
    assert 'flywire_FAFB_v783' in profiler._clients
    # neuprint dataset with injected client
    fake = object()
    profiler.client = fake
    assert profiler._get_client_for_dataset('hemibrain:v1.2.1') is fake


def test_normalize_types_vectorized(profiler):
    cfg = FuzzyMatchConfig(enabled=True, strip_lr_suffix=True)
    out = profiler._normalize_types_vectorized(pd.Series(['T4a_R', 'T4a']), cfg)
    assert list(out) == ['T4a', 'T4a']


def test_config_hash_and_cache_paths(profiler, tmp_path):
    h1 = profiler._get_config_hash()
    h2 = profiler._get_config_hash()
    assert isinstance(h1, str) and h1 == h2
    parquet_path = profiler._get_cache_parquet_path(DS)
    assert str(parquet_path).startswith(str(tmp_path))
    assert str(parquet_path).endswith('.parquet')
    assert str(profiler._get_profile_batch_dir(DS)).startswith(str(tmp_path))


def test_clear_cache(profiler):
    profiler._save_to_cache(_profile('todelete'))
    assert ('todelete', DS) in profiler._memory_cache
    profiler.clear_cache(DS)
    assert ('todelete', DS) not in profiler._memory_cache
