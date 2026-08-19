"""Coverage tests for comparison.cross_dataset_verifier.

Hermetic: uses a FakeProfiler stub returning synthetic ConnectivityProfile
objects. No network, no multiprocessing (threads only, via parallel=True).
"""

import numpy as np
import pandas as pd
import pytest

from comparison.connectivity_profiler import ConnectivityProfile, compute_ranks
from comparison.cross_dataset_verifier import CrossDatasetVerifier, VerificationResult
from comparison.profile_comparator import ComparisonResult

DS_A = 'flywire_FAFB_v783'
DS_B = 'male-cns:v1.0'
DS_C = 'hemibrain:v1.2.1'


def _profile(bid, ds=DS_A, upstream=None, downstream=None, **kwargs):
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


def _rich(bid, ds, reversed_weights=False):
    up = {f'U{i}': float(10 - i if not reversed_weights else i) for i in range(1, 7)}
    down = {f'D{i}': float(8 - i if not reversed_weights else i) for i in range(1, 7)}
    return _profile(bid, ds=ds, upstream=up, downstream=down)


class FakeProfiler:
    """Stub profiler serving synthetic profiles keyed by (str(neuron), dataset)."""

    def __init__(self, profiles=None, raise_for=None, none_for=None):
        self.profiles = profiles or {}
        self.raise_for = raise_for or set()
        self.none_for = none_for or set()
        self.calls = []
        self._defer_cache_writes = False

    def get_profile(self, neuron, dataset, force_refresh=False):
        self.calls.append((neuron, dataset, force_refresh))
        key = (str(neuron), dataset)
        if key in self.raise_for:
            raise RuntimeError(f'no data for {key}')
        if key in self.none_for:
            return None
        return self.profiles.get(key)

    # Stubs for parallel pre-warm / flush hooks used by batch methods
    def _get_cached_conn_df(self, dataset):
        return None

    def _load_cache_dataframe(self, dataset, force_reload=False):
        return None

    def flush_pending_cache_writes(self, silent=False):
        pass


class FakeLabelMapper:
    def __init__(self, mapping=None, raise_exc=False):
        self.mapping = mapping or {}
        self.raise_exc = raise_exc

    def get_mapped_label(self, name, dataset):
        if self.raise_exc:
            raise RuntimeError('mapper broken')
        return self.mapping.get((name, dataset))


def _verifier(profiles, **kwargs):
    kwargs.setdefault('verbose', False)
    return CrossDatasetVerifier(FakeProfiler(profiles), **kwargs)


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------

def _comparison_result(combined=0.8):
    return ComparisonResult(
        profile_a_id='a', profile_b_id='b', dataset_a=DS_A, dataset_b=DS_B,
        direction='both', jaccard=0.9, cosine=0.9, rank_correlation=0.8,
        overlap_a_in_b=0.7, overlap_b_in_a=0.7, combined=combined, confidence='High')


def test_verification_result_to_dict_and_summary():
    vr = VerificationResult(
        neuron_type='T', datasets=[DS_A, DS_B],
        pairwise_scores=[_comparison_result()],
        avg_combined_score=0.8, min_score=0.8, max_score=0.8,
        confidence='High', weak_connectivity_datasets=[DS_B],
        verification_status='verified')
    d = vr.to_dict()
    assert d['neuron_type'] == 'T'
    assert d['num_pairs'] == 1
    assert d['avg_combined_score'] == 0.8
    assert d['weak_connectivity_datasets'] == [DS_B]
    assert len(d['pairwise_details']) == 1
    s = vr.summary()
    assert 'verified' in s and 'Weak connectivity' in s

    vr_nan = VerificationResult(
        neuron_type='T', datasets=[DS_A, DS_B], pairwise_scores=[],
        avg_combined_score=np.nan, min_score=np.nan, max_score=np.nan,
        confidence='N/A', verification_status='type_not_found')
    d_nan = vr_nan.to_dict()
    assert d_nan['avg_combined_score'] is None  # NaN -> JSON-safe None
    assert 'N/A' in vr_nan.summary()


# ---------------------------------------------------------------------------
# _get_profile / _get_mapped_type
# ---------------------------------------------------------------------------

def test_get_profile_caching():
    pa = _rich('T', DS_A)
    profiler = FakeProfiler({('T', DS_A): pa})
    verifier = CrossDatasetVerifier(profiler, verbose=False)
    assert verifier._get_profile('T', DS_A) is pa
    assert verifier._get_profile('T', DS_A) is pa  # cached
    assert len(profiler.calls) == 1
    verifier._get_profile('T', DS_A, force_refresh=True)
    assert len(profiler.calls) == 2

    none_profiler = FakeProfiler(none_for={('X', DS_A)})
    v2 = CrossDatasetVerifier(none_profiler, verbose=False)
    assert v2._get_profile('X', DS_A) is None


def test_get_mapped_type():
    verifier = _verifier({})
    assert verifier._get_mapped_type('T', DS_A) == 'T'

    verifier.label_mapper = FakeLabelMapper({('T', DS_B): 'T_alt'})
    assert verifier._get_mapped_type('T', DS_B) == 'T_alt'
    assert verifier._get_mapped_type('T', DS_A) == 'T'  # no mapping -> original

    verifier.label_mapper = FakeLabelMapper(raise_exc=True)
    assert verifier._get_mapped_type('T', DS_B) == 'T'  # exception -> fallback


# ---------------------------------------------------------------------------
# verify_type_assignment
# ---------------------------------------------------------------------------

def test_verify_insufficient_profiles():
    # profile only in one dataset, other raises -> fewer than 2 profiles
    verifier = _verifier(
        {('T', DS_A): _rich('T', DS_A)},
        comparison_mode='loose')
    verifier.profiler.raise_for = {('T', DS_B)}
    result = verifier.verify_type_assignment('T', [DS_A, DS_B])
    assert result.verification_status == 'failed'
    assert result.confidence == 'Very Low'
    assert result.avg_combined_score == 0.0
    assert result.pairwise_scores == []


def test_verify_identical_profiles_loose():
    verifier = _verifier({
        ('T', DS_A): _rich('T', DS_A),
        ('T', DS_B): _rich('T', DS_B),
    })
    result = verifier.verify_type_assignment('T', [DS_A, DS_B])
    assert result.verification_status == 'verified'
    assert result.confidence == 'Very High'
    assert result.avg_combined_score == pytest.approx(1.0)
    assert len(result.pairwise_scores) == 1


def test_verify_three_datasets_pairwise_count():
    verifier = _verifier({
        ('T', DS_A): _rich('T', DS_A),
        ('T', DS_B): _rich('T', DS_B),
        ('T', DS_C): _rich('T', DS_C),
    })
    result = verifier.verify_type_assignment('T', [DS_A, DS_B, DS_C])
    assert len(result.pairwise_scores) == 3


def test_verify_reversed_weights_fails():
    verifier = _verifier({
        ('T', DS_A): _rich('T', DS_A),
        ('T', DS_B): _rich('T', DS_B, reversed_weights=True),
    })
    result = verifier.verify_type_assignment('T', [DS_A, DS_B])
    assert result.verification_status == 'failed'
    assert result.confidence == 'Very Low'


def test_verify_empty_profiles_type_not_found():
    verifier = _verifier({
        ('T', DS_A): _profile('T', DS_A),
        ('T', DS_B): _profile('T', DS_B),
    })
    result = verifier.verify_type_assignment('T', [DS_A, DS_B])
    assert result.confidence == 'N/A'
    assert result.verification_status == 'type_not_found'
    assert np.isnan(result.avg_combined_score)
    # empty profiles are weak connectivity
    assert set(result.weak_connectivity_datasets) == {DS_A, DS_B}


def test_verify_with_label_mapper():
    profiler = FakeProfiler({
        ('T', DS_A): _rich('T', DS_A),
        ('T_alt', DS_B): _rich('T_alt', DS_B),
    })
    verifier = CrossDatasetVerifier(profiler, verbose=False,
                                    label_mapper=FakeLabelMapper({('T', DS_B): 'T_alt'}))
    result = verifier.verify_type_assignment('T', [DS_A, DS_B])
    assert result.confidence == 'Very High'
    # B-side profile was requested under the mapped name
    assert ('T_alt', DS_B, False) in profiler.calls


# ---------------------------------------------------------------------------
# _compare_profiles_strict
# ---------------------------------------------------------------------------

def test_compare_profiles_strict_sufficient_common():
    verifier = _verifier({}, comparison_mode='strict', min_common_partners=3)
    pa = _rich('a', DS_A)
    pb = _rich('b', DS_B)
    result = verifier._compare_profiles_strict(pa, pb)
    assert result.rank_correlation == pytest.approx(1.0)
    assert 'Insufficient' not in result.notes


def test_compare_profiles_strict_insufficient_common():
    verifier = _verifier({}, comparison_mode='strict', min_common_partners=3)
    pa = _profile('a', upstream={'X': 5.0}, downstream={'Y': 2.0})
    pb = _profile('b', upstream={'X': 3.0}, downstream={'Z': 1.0})
    result = verifier._compare_profiles_strict(pa, pb)
    assert np.isnan(result.rank_correlation)
    assert 'Insufficient common partners (1 < 3)' in result.notes
    expected = 0.5 * result.jaccard + 0.5 * 0.5
    assert result.combined == pytest.approx(expected)

    # also exercise strict path through verify_type_assignment
    verifier2 = _verifier({
        ('T', DS_A): pa, ('T', DS_B): pb,
    }, comparison_mode='strict')
    vresult = verifier2.verify_type_assignment('T', [DS_A, DS_B])
    assert vresult.pairwise_scores[0].notes.count('Insufficient') == 1


# ---------------------------------------------------------------------------
# find_similar_neurons / find_homologs_for_untyped
# ---------------------------------------------------------------------------

def test_find_similar_neurons():
    query = _rich('Q', DS_A)
    candidates = {
        ('U1', DS_B): _rich('U1', DS_B),           # identical -> best match
        ('U2', DS_B): _rich('U2', DS_B, reversed_weights=True),
        ('U3', DS_B): _profile('U3', DS_B),        # empty -> skipped
        ('U4', DS_B): None,                        # missing -> skipped (AttributeError caught)
    }
    candidates[('U4', DS_B)] = None
    profiles = {('Q', DS_A): query}
    profiles.update({k: v for k, v in candidates.items() if v is not None})
    verifier = _verifier(profiles)

    df = verifier.find_similar_neurons('Q', DS_A, DS_B, top_k=5)
    assert not df.empty
    assert df.iloc[0]['target_type'] == 'U1'
    assert df.iloc[0]['combined_score'] == pytest.approx(1.0)
    assert 'U3' not in df['target_type'].values  # empty profile skipped
    assert df['combined_score'].is_monotonic_decreasing

    # explicit candidate list + top_k limiting
    df2 = verifier.find_similar_neurons('Q', DS_A, DS_B,
                                        candidate_types=['U1', 'U2'], top_k=1)
    assert len(df2) == 1

    # no candidates -> empty DataFrame
    empty_query = _profile('E', DS_A)
    verifier2 = _verifier({('E', DS_A): empty_query})
    assert verifier2.find_similar_neurons('E', DS_A, DS_B).empty


def test_find_homologs_for_untyped():
    query_bid = _rich(12345, DS_A)
    verifier = _verifier({
        (str(12345), DS_A): query_bid,
        ('U1', DS_B): _rich('U1', DS_B),
        ('U2', DS_B): _rich('U2', DS_B),
    })
    df = verifier.find_homologs_for_untyped([12345], DS_A, DS_B, top_k=3)
    assert not df.empty
    assert list(df.columns)[0] == 'query_bodyid'
    assert (df['query_bodyid'] == 12345).all()

    # query profile raising -> warning logged, empty result
    profiler = FakeProfiler({}, raise_for={(str(999), DS_A)})
    v2 = CrossDatasetVerifier(profiler, verbose=False)
    assert v2.find_homologs_for_untyped([999], DS_A, DS_B).empty


# ---------------------------------------------------------------------------
# batch_verify_types
# ---------------------------------------------------------------------------

def _batch_profiles():
    return {
        ('Good', DS_A): _rich('Good', DS_A),
        ('Good', DS_B): _rich('Good', DS_B),
        ('Partial', DS_A): _rich('Partial', DS_A),
        # 'Partial' missing in DS_B -> raise
        ('Bad', DS_A): _rich('Bad', DS_A),
        ('Bad', DS_B): _rich('Bad', DS_B, reversed_weights=True),
    }


def test_batch_verify_types_sequential():
    verifier = _verifier(_batch_profiles())
    verifier.profiler.raise_for = {('Partial', DS_B)}
    df = verifier.batch_verify_types(['Good', 'Partial', 'Bad'], [DS_A, DS_B],
                                     parallel=False)
    assert len(df) == 3
    assert list(df.columns[:3]) == ['neuron_type', 'datasets_found', 'total_datasets']
    # sorted by datasets_found desc -> Good/Bad (2) before Partial (1)
    assert df.iloc[-1]['neuron_type'] == 'Partial'
    good = df[df['neuron_type'] == 'Good'].iloc[0]
    assert good['confidence'] == 'Very High'
    assert good['datasets_found'] == 2
    # directional columns included by default
    assert 'avg_rank_corr_upstream' in df.columns
    assert 'avg_jaccard_both' in df.columns


def test_batch_verify_types_parallel():
    verifier = _verifier(_batch_profiles())
    verifier.profiler.raise_for = {('Partial', DS_B)}
    df = verifier.batch_verify_types(['Good', 'Bad'], [DS_A, DS_B],
                                     parallel=True, max_workers=2,
                                     include_directional=False)
    assert len(df) == 2
    assert 'avg_rank_corr_upstream' not in df.columns
    assert verifier.profiler._defer_cache_writes is False  # reset after flush


def test_verify_single_type_error_row():
    profiler = FakeProfiler({})
    verifier = CrossDatasetVerifier(profiler, verbose=False)
    # verify_type_assignment raises internally because profiles dict values
    # trigger an exception path only if get_profile itself raises AFTER
    # partial success; simplest: monkeypatch to force the outer exception.
    def _boom(*a, **k):
        raise RuntimeError('boom')
    verifier.verify_type_assignment = _boom
    row = verifier._verify_single_type('X', [DS_A, DS_B], 'both', None, True)
    assert row['confidence'] == 'Error'
    assert row['datasets_found'] == 0
    assert 'avg_rank_corr_upstream' in row


# ---------------------------------------------------------------------------
# verify_comparison_results
# ---------------------------------------------------------------------------

def test_verify_comparison_results_roles():
    verifier = _verifier(_batch_profiles())
    out = verifier.verify_comparison_results(
        source_types=['Good', 'Bad'],
        target_types=['Bad'],           # Bad is source AND target
        intermediate_types=['Good'],    # already source -> redundant
        datasets=[DS_A, DS_B],
        parallel=False)
    assert set(out.keys()) == {'source', 'target', 'intermediate', 'summary'}
    summary = out['summary']
    roles = dict(zip(summary['neuron_type'], summary['role']))
    assert roles['Bad'] == 'source/target'
    assert roles['Good'] == 'source'
    assert 'Good' in out['source']['neuron_type'].values
    assert 'Bad' in out['target']['neuron_type'].values
    assert out['intermediate'].empty

    # empty inputs -> empty frames
    out_empty = verifier.verify_comparison_results([], [], [], [DS_A, DS_B],
                                                   parallel=False)
    assert out_empty['summary'].empty


# ---------------------------------------------------------------------------
# similarity matrices
# ---------------------------------------------------------------------------

def test_build_similarity_matrix_metrics():
    verifier = _verifier({
        ('T', DS_A): _rich('T', DS_A),
        ('T', DS_B): _rich('T', DS_B),
        ('T', DS_C): _rich('T', DS_C),
    })
    datasets = [DS_A, DS_B, DS_C]
    df = verifier.build_cross_dataset_similarity_matrix(
        ['T'], datasets, metric='combined', parallel=False)
    assert df.shape == (1, 3)
    assert np.allclose(df.loc['T'].to_numpy(), 1.0)

    for metric in ['jaccard', 'cosine', 'rank']:
        dfm = verifier.build_cross_dataset_similarity_matrix(
            ['T'], datasets, metric=metric, parallel=False)
        assert dfm.loc['T'].notna().all()

    # nicknames shorten pair labels
    dfn = verifier.build_cross_dataset_similarity_matrix(
        ['T'], datasets, parallel=False,
        dataset_nicknames={DS_A: 'FW', DS_B: 'MC', DS_C: 'HB'})
    assert 'FW vs MC' in dfn.columns

    # empty profiles in both datasets -> NaN
    verifier2 = _verifier({('E', DS_A): _profile('E', DS_A),
                           ('E', DS_B): _profile('E', DS_B)})
    dfe = verifier2.build_cross_dataset_similarity_matrix(
        ['E'], [DS_A, DS_B], parallel=False)
    assert np.isnan(dfe.loc['E'].iloc[0])

    # failing lookup -> NaN
    verifier2.profiler.raise_for = {('Boom', DS_A)}
    dfb = verifier2.build_cross_dataset_similarity_matrix(
        ['Boom'], [DS_A, DS_B], parallel=False)
    assert np.isnan(dfb.loc['Boom'].iloc[0])


def test_directional_and_multi_metric_matrices():
    verifier = _verifier({
        ('T', DS_A): _rich('T', DS_A),
        ('T', DS_B): _rich('T', DS_B),
    })
    dirs = verifier.build_directional_similarity_matrices(['T'], [DS_A, DS_B],
                                                          parallel=False)
    assert set(dirs.keys()) == {'upstream', 'downstream', 'both'}
    metrics = verifier.build_multi_metric_matrices(['T'], [DS_A, DS_B],
                                                   parallel=False)
    assert set(metrics.keys()) == {'combined', 'jaccard', 'cosine', 'rank'}


def test_similarity_matrix_parallel():
    verifier = _verifier({
        ('T1', DS_A): _rich('T1', DS_A), ('T1', DS_B): _rich('T1', DS_B),
        ('T2', DS_A): _rich('T2', DS_A), ('T2', DS_B): _rich('T2', DS_B),
    })
    df = verifier.build_cross_dataset_similarity_matrix(
        ['T1', 'T2'], [DS_A, DS_B], parallel=True, max_workers=2)
    assert df.shape == (2, 1)
    assert verifier.profiler._defer_cache_writes is False


# ---------------------------------------------------------------------------
# generate_verification_report
# ---------------------------------------------------------------------------

def test_generate_verification_report(tmp_path):
    verifier = _verifier({
        ('T', DS_A): _rich('T', DS_A),
        ('T', DS_B): _rich('T', DS_B),
    })
    out_dir = tmp_path / 'report'
    verifier.generate_verification_report(
        ['T'], [DS_A, DS_B], str(out_dir), parallel=False,
        dataset_nicknames={DS_A: 'FW', DS_B: 'MC'})
    assert (out_dir / 'verification_summary.csv').exists()
    assert (out_dir / 'verification_details.csv').exists()
    assert (out_dir / 'similarity_matrix_combined.csv').exists()
    assert (out_dir / 'directional_matrices' / 'similarity_matrix_upstream.csv').exists()
    assert (out_dir / 'metric_matrices' / 'similarity_matrix_rank.csv').exists()
    assert (out_dir / 'partner_details' / 'partner_overlap_T.csv').exists()

    summary = pd.read_csv(out_dir / 'verification_summary.csv')
    assert summary.iloc[0]['confidence'] == 'Very High'


def test_generate_verification_report_minimal(tmp_path):
    verifier = _verifier({
        ('T', DS_A): _rich('T', DS_A),
        ('T', DS_B): _rich('T', DS_B),
    })
    out_dir = tmp_path / 'report_min'
    verifier.generate_verification_report(
        ['T'], [DS_A, DS_B], str(out_dir), parallel=False,
        include_partner_details=False,
        save_directional_matrices=False, save_metric_matrices=False)
    assert (out_dir / 'verification_summary.csv').exists()
    assert not (out_dir / 'directional_matrices').exists()
    assert not (out_dir / 'metric_matrices').exists()
    assert not (out_dir / 'partner_details').exists()


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------

def test_clear_cache():
    verifier = _verifier({('T', DS_A): _rich('T', DS_A)})
    verifier._get_profile('T', DS_A)
    assert verifier._profile_cache
    verifier.clear_cache()
    assert verifier._profile_cache == {}
