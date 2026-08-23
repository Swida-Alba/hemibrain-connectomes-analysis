"""Coverage tests for comparison.profile_comparator.

Hermetic: all comparisons run on synthetic ConnectivityProfile objects built
directly in memory. No profiler network methods are exercised.
"""

import numpy as np
import pandas as pd
import pytest

from comparison.connectivity_profiler import ConnectivityProfile, compute_ranks
from comparison.profile_comparator import (
    DEFAULT_SCORE_WEIGHTS,
    ComparisonResult,
    ConnectivityProfileComparer,
    ProfileComparator,
)

DS_A = 'flywire_FAFB_v783'
DS_B = 'male-cns:v1.0'


def _profile(bid, ds=DS_A, upstream=None, downstream=None, **kwargs):
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


def _rich_profile(bid, ds=DS_A, scale=1.0):
    """A profile with 6 partners per direction (enough for rank correlation)."""
    up = {f'U{i}': float((10 - i) * scale) for i in range(1, 7)}
    down = {f'D{i}': float((8 - i) * scale) for i in range(1, 7)}
    return _profile(bid, ds=ds, upstream=up, downstream=down)


# ---------------------------------------------------------------------------
# ComparisonResult
# ---------------------------------------------------------------------------

def _result(**overrides):
    params = dict(
        profile_a_id='a', profile_b_id='b', dataset_a='d1', dataset_b='d2',
        direction='both', jaccard=0.8, cosine=0.9, rank_correlation=0.6,
        overlap_a_in_b=0.5, overlap_b_in_a=0.6,
    )
    params.update(overrides)
    return ComparisonResult(**params)


def test_comparison_result_to_dict_and_summary():
    r = _result(weak_connectivity_a=True, notes='some note')
    d = r.to_dict()
    assert d['profile_a'] == 'a'
    assert d['jaccard'] == 0.8
    assert d['weak_connectivity_warning'] is True
    assert d['notes'] == 'some note'
    r_nan = _result(rank_correlation=np.nan, rank_union=np.nan)
    d_nan = r_nan.to_dict()
    assert np.isnan(d_nan['rank_corr'])
    assert np.isnan(d_nan['rank_union'])

    s = r.summary()
    assert 'RankCorr' in s
    s_nan = r_nan.summary()
    assert 'NaN' in s_nan


# ---------------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------------

def test_jaccard_similarity():
    empty_a, empty_b = _profile('a'), _profile('b')
    assert ProfileComparator.jaccard_similarity(empty_a, empty_b) == 0.0

    pa = _profile('a', upstream={'X': 1.0, 'Y': 2.0})
    pb = _profile('b', upstream={'Y': 1.0, 'Z': 1.0})
    assert ProfileComparator.jaccard_similarity(pa, pb, 'upstream') == pytest.approx(1 / 3)
    assert ProfileComparator.jaccard_similarity(pa, pa, 'upstream') == 1.0
    # direction with no partners
    assert ProfileComparator.jaccard_similarity(pa, pb, 'downstream') == 0.0


def test_weighted_cosine_similarity():
    empty_a, empty_b = _profile('a'), _profile('b')
    assert ProfileComparator.weighted_cosine_similarity(empty_a, empty_b) == 0.0

    pa = _profile('a', upstream={'X': 2.0, 'Y': 1.0})
    pb = _profile('b', upstream={'X': 4.0, 'Y': 2.0})
    # proportional weights -> cosine 1
    assert ProfileComparator.weighted_cosine_similarity(pa, pb, 'upstream') == pytest.approx(1.0)

    # one side empty -> zero norm -> 0
    pc = _profile('c')
    assert ProfileComparator.weighted_cosine_similarity(pa, pc, 'upstream') == 0.0

    # disjoint partners -> cosine 0
    pd_ = _profile('d', upstream={'Q': 1.0})
    assert ProfileComparator.weighted_cosine_similarity(pa, pd_, 'upstream') == pytest.approx(0.0)


def test_rank_correlation_identical_and_reversed():
    pa = _rich_profile('a')
    pb = _rich_profile('b')
    assert ProfileComparator.rank_correlation(pa, pb, 'upstream') == pytest.approx(1.0)
    assert ProfileComparator.rank_correlation(pa, pb, 'both') == pytest.approx(1.0)

    reversed_down = {f'D{i}': float(i) for i in range(1, 7)}  # increasing -> reversed ranks
    pb_rev = _profile('b', upstream=pb.upstream_partners, downstream=reversed_down)
    both_corr = ProfileComparator.rank_correlation(pa, pb_rev, 'both')
    assert both_corr == pytest.approx(0.0)  # (1 + -1) / 2

    # kendall also supported
    kt = ProfileComparator.rank_correlation(pa, pb, 'upstream', method='kendall')
    assert kt == pytest.approx(1.0)


def test_rank_correlation_nan_paths():
    # fewer than 2 total partners
    pa = _profile('a', upstream={'X': 1.0})
    pb = _profile('b')
    assert np.isnan(ProfileComparator.rank_correlation(pa, pb, 'upstream'))

    # constant rank arrays (all ties) -> NaN
    tied = {f'T{i}': 5.0 for i in range(6)}
    p1 = _profile('p1', upstream=tied)
    p2 = _profile('p2', upstream=tied)
    assert np.isnan(ProfileComparator.rank_correlation(p1, p2, 'upstream'))

    # both directions NaN -> NaN for 'both'
    assert np.isnan(ProfileComparator.rank_correlation(pa, pb, 'both'))


def test_rank_correlation_expansion_and_union_mode():
    # shared < 5 triggers union expansion path
    pa = _profile('a', upstream={'A': 6.0, 'B': 5.0, 'C': 4.0, 'D': 3.0})
    pb = _profile('b', upstream={'A': 6.0, 'B': 5.0, 'E': 4.0, 'F': 3.0})
    corr = ProfileComparator.rank_correlation(pa, pb, 'upstream')
    assert isinstance(corr, float) and -1.0 <= corr <= 1.0

    # use_all_partners mode with defaults for missing partners
    corr_all = ProfileComparator.rank_correlation(pa, pb, 'upstream', use_all_partners=True)
    assert isinstance(corr_all, float)
    # union < 3 in all-partners mode -> NaN
    small_a = _profile('a', upstream={'A': 1.0})
    small_b = _profile('b', upstream={'B': 1.0})
    assert np.isnan(ProfileComparator.rank_correlation(
        small_a, small_b, 'upstream', use_all_partners=True))


def test_rank_correlation_both_falls_back_to_valid_direction():
    # only upstream has data -> 'both' returns upstream correlation
    pa_down_empty = _profile('a', upstream=_rich_profile('a').upstream_partners)
    pb_down_empty = _profile('b', upstream=_rich_profile('b').upstream_partners)
    assert ProfileComparator.rank_correlation(
        pa_down_empty, pb_down_empty, 'both') == pytest.approx(1.0)


def test_overlap_fraction():
    pa = _profile('a', upstream={'X': 1.0, 'Y': 1.0})
    pb = _profile('b', upstream={'Y': 1.0, 'Z': 1.0, 'W': 1.0})
    ov_a, ov_b = ProfileComparator.overlap_fraction(pa, pb, 'upstream')
    assert ov_a == pytest.approx(0.5)
    assert ov_b == pytest.approx(1 / 3)
    empty = _profile('e')
    assert ProfileComparator.overlap_fraction(pa, empty, 'upstream') == (0.0, 0.0)


# ---------------------------------------------------------------------------
# combined_score
# ---------------------------------------------------------------------------

def test_combined_score_identical():
    pa = _rich_profile('a')
    pb = _rich_profile('b')
    scores = ProfileComparator.combined_score(pa, pb)
    assert scores['jaccard'] == 1.0
    assert scores['weighted_jaccard'] == pytest.approx(1.0)
    assert scores['cosine'] == pytest.approx(1.0)
    assert scores['rank'] == pytest.approx(1.0)
    assert scores['rank_union'] == pytest.approx(1.0)
    assert scores['overlap_avg'] == pytest.approx(1.0)
    assert 'combined' not in scores
    assert 'rank_norm' not in scores
    assert 'rank_union_norm' not in scores


def test_combined_score_empty_and_custom_weights():
    pa, pb = _profile('a'), _profile('b')
    scores = ProfileComparator.combined_score(pa, pb)
    assert scores['jaccard'] == 0.0
    assert scores['weighted_jaccard'] == 0.0
    assert np.isnan(scores['rank'])
    assert np.isnan(scores['rank_union'])
    assert 'combined' not in scores

    weights = {'jaccard': 1.0, 'rank': 0.0}
    pa2 = _profile('a', upstream={'X': 2.0})
    pb2 = _profile('b', upstream={'X': 4.0})
    scores2 = ProfileComparator.combined_score(pa2, pb2, weights=weights)
    assert scores2['jaccard'] == pytest.approx(1.0)
    assert scores2['weighted_jaccard'] == pytest.approx(0.5)  # min(2,4)/max(2,4)


# ---------------------------------------------------------------------------
# Expanded types (2-hop)
# ---------------------------------------------------------------------------

def test_get_expanded_types_1hop_only_when_typed_present():
    p = ConnectivityProfile(
        neuron_id='x', dataset=DS_A,
        upstream_partners={'A': 10.0}, upstream_ranks={'A': 1},
        total_upstream_weight=10.0,
        untyped_upstream_bodyids={7: 2.0},
        untyped_upstream_2hop={7: {'H2': 4.0}},
    )
    types = ProfileComparator._get_expanded_types(p, 'upstream')
    # typed partner present in top-N -> 2-hop NOT added
    assert types == {'A': 10.0}


def test_get_expanded_types_2hop_when_all_untyped():
    p = ConnectivityProfile(
        neuron_id='x', dataset=DS_A,
        untyped_upstream_bodyids={7: 6.0},
        untyped_upstream_2hop={7: {'H2': 3.0, 'H3': 1.0}},
    )
    types = ProfileComparator._get_expanded_types(p, 'upstream')
    # scaled = (hop2_weight / hop2_total) * untyped_weight
    assert types['2hop:H2'] == pytest.approx(4.5)
    assert types['2hop:H3'] == pytest.approx(1.5)
    types_noprefix = ProfileComparator._get_expanded_types(p, 'upstream', prefix_2hop=False)
    assert 'H2' in types_noprefix

    # downstream path as well
    p2 = ConnectivityProfile(
        neuron_id='y', dataset=DS_A,
        untyped_downstream_bodyids={8: 2.0},
        untyped_downstream_2hop={8: {'T1': 1.0}},
    )
    types_down = ProfileComparator._get_expanded_types(p2, 'both')
    assert types_down['2hop:T1'] == pytest.approx(2.0)


def test_get_expanded_types_standardized_with_mapper():
    p = _profile('x', upstream={'LocalType': 5.0})

    class FakeMapper:
        def standardize_partner_types(self, partners, source_dataset):
            return {f'canon_{k}': v for k, v in partners.items()}

    out_none = ProfileComparator._get_expanded_types_standardized(p, 'upstream', None)
    assert out_none == {'LocalType': 5.0}
    out = ProfileComparator._get_expanded_types_standardized(p, 'upstream', FakeMapper())
    assert out == {'canon_LocalType': 5.0}


# ---------------------------------------------------------------------------
# bodyId-level metrics (intra-dataset)
# ---------------------------------------------------------------------------

def _bodyid_profile(bid, typed_up=None, untyped_up=None, typed_down=None):
    return ConnectivityProfile(
        neuron_id=bid, dataset=DS_A,
        typed_upstream_bodyids=typed_up,
        untyped_upstream_bodyids=untyped_up,
        typed_downstream_bodyids=typed_down,
    )


def test_bodyid_jaccard():
    pa = _bodyid_profile('a', typed_up={1: 5.0, 2: 3.0, 3: 1.0})
    pb = _bodyid_profile('b', typed_up={2: 3.0, 3: 1.0, 4: 1.0})
    assert ProfileComparator.bodyid_jaccard(pa, pb, 'upstream') == pytest.approx(0.5)
    empty = _bodyid_profile('e')
    assert ProfileComparator.bodyid_jaccard(empty, empty, 'upstream') == 0.0
    # untyped bodyIds also count
    pc = _bodyid_profile('c', typed_up={1: 5.0}, untyped_up={9: 2.0})
    assert ProfileComparator.bodyid_jaccard(pa, pc, 'upstream') == pytest.approx(1 / 4)


def test_bodyid_rank_correlation():
    pa = _bodyid_profile('a', typed_up={1: 6.0, 2: 5.0, 3: 4.0, 4: 3.0})
    pb = _bodyid_profile('b', typed_up={1: 6.0, 2: 5.0, 3: 4.0, 4: 3.0})
    assert ProfileComparator.bodyid_rank_correlation(pa, pb, 'upstream') == pytest.approx(1.0)

    # fewer than 3 shared -> NaN
    pc = _bodyid_profile('c', typed_up={1: 6.0, 2: 5.0})
    assert np.isnan(ProfileComparator.bodyid_rank_correlation(pa, pc, 'upstream'))

    # constant weights -> NaN
    const_a = _bodyid_profile('ca', typed_up={1: 5.0, 2: 5.0, 3: 5.0})
    const_b = _bodyid_profile('cb', typed_up={1: 2.0, 2: 2.0, 3: 2.0})
    assert np.isnan(ProfileComparator.bodyid_rank_correlation(const_a, const_b, 'upstream'))


def test_bodyid_rank_correlation_union():
    pa = _bodyid_profile('a', typed_up={1: 6.0, 2: 5.0, 3: 4.0})
    pb = _bodyid_profile('b', typed_up={1: 6.0, 2: 5.0, 4: 3.0})
    corr = ProfileComparator.bodyid_rank_correlation_union(pa, pb, 'upstream')
    assert isinstance(corr, float)

    # union < 3 -> NaN
    small_a = _bodyid_profile('a', typed_up={1: 1.0})
    small_b = _bodyid_profile('b', typed_up={2: 1.0})
    assert np.isnan(ProfileComparator.bodyid_rank_correlation_union(small_a, small_b))


# ---------------------------------------------------------------------------
# Expanded-type metrics (cross-dataset)
# ---------------------------------------------------------------------------

def test_expanded_type_metrics():
    up = {f'T{i}': float(10 - i) for i in range(1, 7)}
    pa = _profile('a', upstream=up)
    pb = _profile('b', upstream=up)
    assert ProfileComparator.expanded_type_jaccard(pa, pb, 'upstream') == 1.0
    assert ProfileComparator.expanded_type_rank_correlation(pa, pb, 'upstream') == pytest.approx(1.0)
    assert ProfileComparator.expanded_type_rank_correlation_union(pa, pb, 'upstream') == pytest.approx(1.0)

    # disjoint sets
    pc = _profile('c', upstream={'Q1': 1.0, 'Q2': 2.0})
    assert ProfileComparator.expanded_type_jaccard(pa, pc, 'upstream') == 0.0
    # shared < 3 -> NaN
    assert np.isnan(ProfileComparator.expanded_type_rank_correlation(pa, pc, 'upstream'))

    empty_a, empty_b = _profile('e1'), _profile('e2')
    assert ProfileComparator.expanded_type_jaccard(empty_a, empty_b, 'upstream') == 0.0
    assert np.isnan(ProfileComparator.expanded_type_rank_correlation_union(empty_a, empty_b))


# ---------------------------------------------------------------------------
# Combined scores: intra / cross dataset
# ---------------------------------------------------------------------------

def test_combined_score_intra_dataset_empty():
    pa = _bodyid_profile('a', typed_up={1: 5.0})
    pb = _bodyid_profile('b')
    scores = ProfileComparator.combined_score_intra_dataset(pa, pb)
    assert 'combined' not in scores
    assert np.isnan(scores['jaccard'])
    assert scores['shared_type_count'] == 0


def test_combined_score_intra_dataset_full():
    typed = {i: float(10 - i) for i in range(1, 7)}
    pa = _bodyid_profile('a', typed_up=typed)
    pb = _bodyid_profile('b', typed_up=typed)
    scores = ProfileComparator.combined_score_intra_dataset(pa, pb, direction='upstream')
    assert scores['jaccard'] == pytest.approx(1.0)
    assert scores['weighted_jaccard'] == pytest.approx(1.0)
    assert scores['cosine'] == pytest.approx(1.0)
    assert scores['rank'] == pytest.approx(1.0)
    assert scores['rank_union'] == pytest.approx(1.0)
    assert 'combined' not in scores
    assert scores['shared_type_count'] == 6
    assert scores['union_type_count'] == 6


def test_combined_score_cross_dataset_empty_and_full():
    pa = _profile('a', upstream={'X': 1.0})
    pb = _profile('b')
    scores = ProfileComparator.combined_score_cross_dataset(pa, pb)
    assert 'combined' not in scores
    assert scores['union_type_count'] == 1

    up = {f'T{i}': float(10 - i) for i in range(1, 7)}
    pa2 = _profile('a', upstream=up)
    pb2 = _profile('b', upstream=up)
    scores2 = ProfileComparator.combined_score_cross_dataset(pa2, pb2, direction='upstream')
    assert scores2['jaccard'] == pytest.approx(1.0)
    assert 'combined' not in scores2
    assert scores2['shared_type_count'] == 6


# ---------------------------------------------------------------------------
# batch_compare_cross_dataset
# ---------------------------------------------------------------------------

class FakeMapper:
    """Minimal mapper canonicalizing type names."""

    def standardize_partner_types(self, partners, source_dataset):
        return {k.replace('_local', ''): v for k, v in partners.items()}


def test_batch_compare_cross_dataset_empty_source():
    src = _profile('src')
    assert ProfileComparator.batch_compare_cross_dataset(
        src, {1: _profile(1)}, {1: 5}) == []


def test_batch_compare_cross_dataset_variants():
    up = {f'T{i}': float(10 - i) for i in range(1, 7)}
    src = _profile('src', upstream=up)
    same = _profile(1, ds=DS_B, upstream=up)
    disjoint = _profile(2, ds=DS_B, upstream={'Q': 1.0})
    empty_target = _profile(3, ds=DS_B)

    targets = {1: same, 2: disjoint, 3: empty_target}
    candidate_map = {1: 10, 2: 8, 3: 4, 99: 1}  # 99 has no profile -> skipped

    results = ProfileComparator.batch_compare_cross_dataset(src, targets, candidate_map)
    assert len(results) == 3
    by_bid = {r['target_bid']: r for r in results}

    assert by_bid[1]['jaccard'] == pytest.approx(1.0)
    assert by_bid[1]['rank'] == pytest.approx(1.0)

    # no overlap -> jaccard 0
    assert by_bid[2]['jaccard'] == 0.0

    # empty target -> NaN row
    assert 'combined' not in by_bid[3]
    assert by_bid[3]['target_type_count'] == 0

    # with type mapper standardizing names
    up_local = {f'T{i}_local': float(10 - i) for i in range(1, 7)}
    src_local = _profile('src', upstream=up_local)
    results_mapped = ProfileComparator.batch_compare_cross_dataset(
        src_local, {1: same}, {1: 5}, type_mapper=FakeMapper())
    assert results_mapped[0]['jaccard'] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compare_profiles / compare_profiles_simple / overlap details / multiple
# ---------------------------------------------------------------------------

def test_compare_profiles():
    pa = _rich_profile('a')
    pb = _rich_profile('b', ds=DS_B)
    result = ProfileComparator.compare_profiles(pa, pb)
    assert isinstance(result, ComparisonResult)
    assert result.profile_a_id == 'a'
    assert result.dataset_b == DS_B
    assert result.rank_correlation == pytest.approx(1.0)

    # score_weights alias + NaN rank note
    tied = {f'T{i}': 5.0 for i in range(6)}
    p1 = _profile('p1', upstream=tied)
    p2 = _profile('p2', upstream=tied)
    result_nan = ProfileComparator.compare_profiles(
        p1, p2, direction='upstream', score_weights=DEFAULT_SCORE_WEIGHTS)
    assert np.isnan(result_nan.rank_correlation)
    assert 'constant' in result_nan.notes.lower()


def test_compare_profiles_simple():
    pa = _rich_profile('a')
    pb = _rich_profile('b')
    d = ProfileComparator.compare_profiles_simple(pa, pb)
    assert set(d.keys()) == {'jaccard', 'cosine', 'rank_correlation', 'shared_count'}
    assert d['shared_count'] == 12  # 6 up + 6 down shared
    d_up = ProfileComparator.compare_profiles_simple(pa, pb, direction='upstream')
    assert d_up['shared_count'] == 6


def test_get_partner_overlap_details():
    # SOURCE BUG (reported, not fixed here): with NO overlapping/any partners
    # the internal sort-key assignment fails because df.apply(axis=1) on the
    # empty DataFrame cannot be assigned to the single 'sort_key' column.
    pa0 = _profile('a', upstream={'X': 3.0})
    pb0 = _profile('b')
    with pytest.raises(ValueError):
        ProfileComparator.get_partner_overlap_details(pa0, pb0, 'downstream')

    pa = _profile('a', upstream={'X': 3.0, 'Y': 2.0, 'Z': 1.0})
    pb = _profile('b', upstream={'X': 1.0, 'W': 1.0})
    df = ProfileComparator.get_partner_overlap_details(pa, pb, 'upstream')
    assert list(df.columns) == ['partner_type', 'in_a', 'in_b', 'weight_a',
                                'weight_b', 'rank_a', 'rank_b', 'status']
    assert df.iloc[0]['status'] == 'shared'  # shared rows sorted first
    assert set(df['status']) == {'shared', 'a_only', 'b_only'}


def test_compare_multiple_profiles():
    profiles = {
        'a': _rich_profile('a'),
        'b': _rich_profile('b'),
        'c': _rich_profile('c', ds=DS_B),
    }
    df = ProfileComparator.compare_multiple_profiles(profiles)
    assert len(df) == 3
    assert 'pair' in df.columns
    assert set(df['pair']) == {'a vs b', 'a vs c', 'b vs c'}


# ---------------------------------------------------------------------------
# ConnectivityProfileComparer helpers (called unbound with synthetic data)
# ---------------------------------------------------------------------------

def test_compute_similarity_from_types():
    types_a = {'X': 5.0, 'Y': 3.0, 'Z': 1.0}
    same = ConnectivityProfileComparer._compute_similarity_from_types(None, types_a, types_a)
    assert same['jaccard'] == pytest.approx(1.0)
    assert same['weighted_jaccard'] == pytest.approx(1.0)
    assert same['cosine'] == pytest.approx(1.0)
    assert same['rank'] == pytest.approx(1.0)
    assert same['rank_union'] == pytest.approx(1.0)
    assert 'combined' not in same

    empty = ConnectivityProfileComparer._compute_similarity_from_types(None, {}, {})
    assert empty['jaccard'] == 0.0
    assert empty['cosine'] == 0.0
    assert 'rank_norm' not in empty
    assert 'combined' not in empty

    partial = ConnectivityProfileComparer._compute_similarity_from_types(
        None, types_a, {'X': 5.0, 'Q': 1.0})
    assert partial['jaccard'] == pytest.approx(1 / 4)
    assert np.isnan(partial['rank'])  # only 1 shared type (< 3)
    assert 'combined' not in partial


def test_same_name_flag():
    p = _rich_profile('a')
    per_ds = {DS_A: ('TypeX', p), DS_B: ('TypeX', p)}
    assert ConnectivityProfileComparer._same_name_flag(per_ds, [DS_A, DS_B]) == 1
    per_ds_diff = {DS_A: ('TypeX', p), DS_B: ('TypeY', p)}
    assert ConnectivityProfileComparer._same_name_flag(per_ds_diff, [DS_A, DS_B]) == 0
    per_ds_missing = {DS_A: ('TypeX', p)}
    assert ConnectivityProfileComparer._same_name_flag(per_ds_missing, [DS_A, DS_B]) == 0
    per_ds_none = {DS_A: (None, p), DS_B: ('TypeX', p)}
    assert ConnectivityProfileComparer._same_name_flag(per_ds_none, [DS_A, DS_B]) == 0


def test_comparer_compute_ranks():
    assert ConnectivityProfileComparer._compute_ranks({}) == {}
    ranks = ConnectivityProfileComparer._compute_ranks({'a': 3.0, 'b': 5.0})
    assert ranks == {'b': 1, 'a': 2}


# ---------------------------------------------------------------------------
# ProfileComparator: type-level similarity matrices
# ---------------------------------------------------------------------------

def test_build_similarity_matrix_metrics():
    profiles = {
        'T1': _rich_profile('T1'),
        'T2': _rich_profile('T2'),  # identical to T1
        'T3': _profile('T3', upstream={'Z9': 1.0}),  # disjoint
    }
    m = ProfileComparator.build_similarity_matrix(profiles, metric='rank')
    assert m.shape == (3, 3)
    assert np.allclose(np.diag(m.to_numpy()), 1.0)
    assert m.loc['T1', 'T2'] == pytest.approx(1.0)
    assert m.loc['T2', 'T1'] == pytest.approx(m.loc['T1', 'T2'])  # symmetric
    # disjoint types: jaccard 0, small residual from rank expansion
    assert m.loc['T1', 'T3'] < 0.2

    for metric in ['jaccard', 'cosine', 'rank']:
        mm = ProfileComparator.build_similarity_matrix(profiles, metric=metric)
        assert mm.loc['T1', 'T2'] == pytest.approx(1.0)


def test_compute_inter_type_rank_correlation_matrix():
    profiles_by_dataset = {
        DS_A: {'T1': _rich_profile('T1'), 'T2': _profile('T2', upstream={'Z': 1.0})},
        DS_B: {'T1': _rich_profile('T1')},
    }
    out = ProfileComparator.compute_inter_type_rank_correlation_matrix(
        profiles_by_dataset, metric='rank')
    assert DS_A in out['intra_dataset']          # >= 2 types -> intra matrix
    assert DS_B not in out['intra_dataset']      # single type -> skipped
    assert out['all_types_matrix'].shape == (3, 3)
    cross = out['cross_dataset']
    assert not cross.empty
    same = cross[cross['same_type']]
    assert same.iloc[0]['similarity'] == pytest.approx(1.0)

    # empty input -> empty results
    out_empty = ProfileComparator.compute_inter_type_rank_correlation_matrix({})
    assert out_empty['cross_dataset'] is None
    assert out_empty['intra_dataset'] == {}

    # other metrics run through the same paths
    for metric in ['rank', 'jaccard', 'cosine']:
        o = ProfileComparator.compute_inter_type_rank_correlation_matrix(
            profiles_by_dataset, metric=metric)
        assert not o['cross_dataset'].empty


def test_find_similar_types_across_datasets():
    profiles_by_dataset = {
        DS_A: {'T1': _rich_profile('T1'), 'Other': _profile('O', upstream={'Z': 1.0})},
        DS_B: {'T1': _rich_profile('T1'), 'Alt': _profile('Alt', upstream={'Q': 1.0})},
    }
    df = ProfileComparator.find_similar_types_across_datasets(
        profiles_by_dataset, 'T1', metric='rank', top_n=1)
    assert not df.empty
    # best match in DS_B is the identical T1
    ds_b_rows = df[df['target_dataset'] == DS_B]
    assert ds_b_rows.iloc[0]['target_type'] == 'T1'
    assert ds_b_rows.iloc[0]['similarity'] == pytest.approx(1.0)
    # top_n=1 per dataset
    assert len(ds_b_rows) == 1

    # unknown type -> empty result frame
    df_none = ProfileComparator.find_similar_types_across_datasets(
        profiles_by_dataset, 'NoSuch')
    assert df_none.empty

    # top_n=None keeps all rows
    df_all = ProfileComparator.find_similar_types_across_datasets(
        profiles_by_dataset, 'T1', top_n=None)
    assert len(df_all) == 3  # everything except self-comparison


# ---------------------------------------------------------------------------
# HomologFinder (hermetic: profiler replaced by a stub)
# ---------------------------------------------------------------------------

class _FakeProfilerForFinder:
    """Stub profiler covering the methods HomologFinder delegates to."""

    def __init__(self, profiles=None, bodyids=None, available=None):
        self.profiles = profiles or {}
        self.bodyids = bodyids or {}
        self.available = available or {}
        self._defer_cache_writes = False
        self._pending_cache_writes = {}
        self.calls = {'consolidate': 0, 'load_cache': 0, 'cached_conn': 0}

    def get_profile(self, neuron, dataset, force_refresh=False):
        return self.profiles.get((str(neuron), dataset))

    def get_bodyids_for_type(self, neuron_type, dataset):
        return self.bodyids.get((neuron_type, dataset))

    def get_available_types(self, dataset):
        return self.available.get(dataset)

    def ensure_data_available(self, dataset, raise_on_missing=True):
        if dataset not in self.available:
            raise RuntimeError(f'no data for {dataset}')
        return True

    def get_data_status(self, datasets):
        return {ds: {'available': ds in self.available} for ds in datasets}

    def consolidate_profile_cache(self, dataset):
        self.calls['consolidate'] += 1

    def _load_cache_dataframe(self, dataset, force_reload=False):
        self.calls['load_cache'] += 1

    def _get_cached_conn_df(self, dataset):
        self.calls['cached_conn'] += 1
        return None

    def flush_pending_cache_writes(self, silent=False):
        pass


@pytest.fixture
def finder(tmp_path):
    from comparison.profile_comparator import HomologFinder
    f = HomologFinder(output_dir=str(tmp_path), verbose=False)
    return f


def test_homolog_finder_type_mapper_variants(finder, tmp_path):
    from comparison.cross_dataset_type_mapper import CrossDatasetTypeMapper

    # disabled entirely
    finder.use_auto_type_mapping = False
    assert finder._get_type_mapper_for_comparison(True) is None

    # intra-dataset -> always None
    finder.use_auto_type_mapping = True
    assert finder._get_type_mapper_for_comparison(False) is None

    # pre-seeded loaded mapper is returned
    csv = tmp_path / 'n.csv'
    csv.write_text(
        "bodyId,type,flywireType,hemibrainType,mancType\n1,aMe12,MTe07,,\n",
        encoding='utf-8')
    loaded = CrossDatasetTypeMapper(neuron_df_path=str(csv), verbose=False)
    loaded.load()
    finder._type_mapper = loaded
    assert finder._get_type_mapper_for_comparison(True) is loaded

    # unloaded mapper -> None
    unloaded = CrossDatasetTypeMapper(
        neuron_df_path=str(tmp_path / 'missing.csv'), verbose=False)
    finder._type_mapper = unloaded
    assert finder._get_type_mapper_for_comparison(True) is None


def test_homolog_finder_profile_helpers(finder):
    p = _rich_profile('T', DS_A)
    fake = _FakeProfilerForFinder(
        profiles={('T', DS_A): p},
        bodyids={('T', DS_A): [11, 12]},
        available={DS_A: ['T']},
    )
    finder.profiler = fake

    assert finder.get_profile('T', DS_A) is p
    assert finder.get_profile('X', DS_A) is None

    assert finder.get_available_types(DS_A) == ['T']
    assert finder.get_available_types(DS_A) == ['T']  # cached
    assert finder.get_bodyids_for_type('T', DS_A) == [11, 12]
    assert finder.get_bodyids_for_type('T', DS_A) == [11, 12]  # cached
    assert finder.get_bodyids_for_type('Missing', DS_A) == []

    finder._prewarm_profile_cache(DS_A)
    assert fake.calls == {'consolidate': 1, 'load_cache': 1, 'cached_conn': 1}


def test_homolog_finder_get_profiles_batch(finder):
    p1, p2 = _rich_profile('A', DS_A), _rich_profile('B', DS_A)
    fake = _FakeProfilerForFinder(profiles={('A', DS_A): p1, ('B', DS_A): p2})
    finder.profiler = fake

    out = finder.get_profiles_batch(['A', 'B', 'Missing'], DS_A,
                                    show_progress=False)
    assert out == {'A': p1, 'B': p2}
    # deferred-write mode reset afterwards
    assert fake._defer_cache_writes is False


def test_homolog_finder_data_availability(finder):
    fake = _FakeProfilerForFinder(available={DS_A: ['T']})
    finder.profiler = fake
    finder.source_dataset = DS_A
    finder.target_dataset = DS_B

    avail = finder.ensure_data_available(raise_on_missing=False)
    assert avail == {DS_A: True, DS_B: False}

    with pytest.raises(RuntimeError):
        finder.ensure_data_available(raise_on_missing=True)

    # explicit dataset list
    assert finder.ensure_data_available([DS_A]) == {DS_A: True}

    status = finder.get_data_status()
    assert status[DS_A]['available'] is True
    assert status[DS_B]['available'] is False


def test_homolog_finder_get_neuron_info(finder):
    class FakeFNC:
        def _fetch_neurons_local_or_api(self, ids, columns=None):
            return pd.DataFrame({'bodyId': ids})

        def _fetch_neurons_by_types(self, types, columns=None):
            return pd.DataFrame({'type': types})

    finder._fnc_cache[DS_A] = FakeFNC()
    df_bid = finder.get_neuron_info([1, 2], DS_A)
    assert list(df_bid['bodyId']) == [1, 2]
    df_type = finder.get_neuron_info(['T1'], DS_A)
    assert list(df_type['type']) == ['T1']

    # no FNC available -> empty frame
    assert finder.get_neuron_info([1], DS_B).empty


def test_compare_types_bodyid_core(finder):
    def bodyid_profile(bid, ds, offset=0):
        up_bids = {100 + i + offset: float(10 - i) for i in range(6)}
        down_bids = {200 + i + offset: float(8 - i) for i in range(6)}
        return ConnectivityProfile(
            neuron_id=bid, dataset=ds,
            typed_upstream_bodyids=up_bids,
            typed_downstream_bodyids=down_bids,
            upstream_ranks={}, downstream_ranks={},
            total_upstream_weight=float(sum(up_bids.values())),
            total_downstream_weight=float(sum(down_bids.values())),
        )

    pa1 = bodyid_profile(1, DS_A)
    pa2 = bodyid_profile(2, DS_A)
    pb1 = bodyid_profile(3, DS_B)  # identical bodyId sets -> perfect match
    fake = _FakeProfilerForFinder(
        profiles={('1', DS_A): pa1, ('2', DS_A): pa2, ('3', DS_B): pb1},
        bodyids={('T', DS_A): [1, 2], ('T', DS_B): [3]},
    )
    finder.profiler = fake

    bodyid_df, type_summary = ProfileComparator.compare_types_bodyid_core(
        profiler=fake, neuron_types=['T'], dataset_a=DS_A, dataset_b=DS_B,
        min_common_partners=3, verbose=False)
    assert len(bodyid_df) == 2  # 2 x 1 bodyId pairs
    assert set(bodyid_df.columns) >= {'neuron_type', 'source_bodyId',
                                      'target_bodyId', 'jaccard', 'cosine',
                                      'rank_corr', 'rank_union'}
    assert bodyid_df['jaccard'].iloc[0] == pytest.approx(1.0)
    # NOTE: source bug - compare_types_bodyid_core calls
    # profile.get_partner_types() which does not exist on
    # ConnectivityProfile, so overlap_count is always 0 and rank metrics
    # are forced to NaN whenever min_common_partners > 0.
    assert np.isnan(bodyid_df['rank_corr'].iloc[0])
    assert type_summary.iloc[0]['n_source_bodyIds'] == 2
    assert type_summary.iloc[0]['avg_jaccard'] == pytest.approx(1.0)

    # min_common_partners=0 keeps rank metrics
    bodyid_df_ok, _ = ProfileComparator.compare_types_bodyid_core(
        profiler=fake, neuron_types=['T'], dataset_a=DS_A, dataset_b=DS_B,
        min_common_partners=0, verbose=False)
    assert bodyid_df_ok['rank_corr'].notna().all()

    # type with no bodyIds anywhere -> empty results
    fake3 = _FakeProfilerForFinder()
    bodyid_df3, summary3 = ProfileComparator.compare_types_bodyid_core(
        profiler=fake3, neuron_types=['Ghost'], dataset_a=DS_A,
        dataset_b=DS_B, verbose=False)
    assert bodyid_df3.empty and summary3.empty


def test_homolog_finder_get_partner_overlap(finder):
    pa = _rich_profile('Q', DS_A)
    pb = _rich_profile('C', DS_B)
    fake = _FakeProfilerForFinder(
        profiles={('Q', DS_A): pa, ('C', DS_B): pb})
    finder.profiler = fake

    df = finder.get_partner_overlap('Q', DS_A, 'C', DS_B, direction='upstream')
    assert not df.empty

    # missing profile on either side -> empty frame
    assert finder.get_partner_overlap('Nope', DS_A, 'C', DS_B).empty
