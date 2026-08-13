"""The cross-dataset comparison is backended by FindAllPath: ComparisonParameters
must carry the same edge limits as the Find All Paths tab — the bodyId pan-graph
edge limit (1M, applied only for deep searches) and the Visualization Edge Limit
(2000) — and pass them through to every FindAllPath run."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison import ComparisonParameters  # noqa: E402


def test_comparison_parameters_edge_limit_defaults():
    """The cross-dataset defaults match FindAllPath: the bodyId edge limit
    is mode-resolved (None = per-mode default: 1M for FindAllPath, 0 for
    FindShortestPath) and the Visualization Edge Limit is 500."""
    p = ComparisonParameters(
        datasets=['hemibrain:v1.2.1', 'male-cns:v1.0'])
    assert p.graph_edge_limit_bodyid is None
    assert p.edgeN_limit == 500


def test_symmetry_analysis_gated_by_separate_hemispheres():
    """symmetry_analysis is forced OFF when separate_hemispheres=False
    (it would otherwise run and produce meaningless summaries) and
auto-enabled when separate_hemispheres=True."""
    p = ComparisonParameters(
        datasets=['hemibrain:v1.2.1', 'male-cns:v1.0'],
        separate_hemispheres=False, symmetry_analysis=True,
    )
    assert p.symmetry_analysis is False
    p2 = ComparisonParameters(
        datasets=['hemibrain:v1.2.1', 'male-cns:v1.0'],
        separate_hemispheres=True, symmetry_analysis=False,
    )
    assert p2.symmetry_analysis is True


def test_comparison_parameters_edge_limits_survive_dict_roundtrip():
    """Customized limits survive to_dict -> from_dict."""
    p = ComparisonParameters(
        datasets=['hemibrain:v1.2.1', 'male-cns:v1.0'],
        graph_edge_limit_bodyid=500000, edgeN_limit=1500,
    )
    d = p.to_dict()
    assert d["graph_edge_limit_bodyid"] == 500000
    assert d["edgeN_limit"] == 1500
    p2 = ComparisonParameters.from_dict(d)
    assert p2.graph_edge_limit_bodyid == 500000
    assert p2.edgeN_limit == 1500


# =============================================================================
# path_mode ('all' | 'shortest') — FindShortestPath integration
# =============================================================================

def test_path_mode_default_is_all():
    """Default stays 'all' (FindAllPath) so existing comparisons are
    byte-compatible."""
    p = ComparisonParameters(
        datasets=['hemibrain:v1.2.1', 'male-cns:v1.2.1'])
    assert p.path_mode == 'all'


def test_path_mode_invalid_value_raises():
    import pytest
    with pytest.raises(ValueError):
        ComparisonParameters(
            datasets=['hemibrain:v1.2.1', 'male-cns:v1.2.1'],
            path_mode='direct')


def test_path_mode_survives_dict_roundtrip_and_legacy_default():
    p = ComparisonParameters(
        datasets=['hemibrain:v1.2.1', 'male-cns:v1.2.1'],
        path_mode='shortest')
    d = p.to_dict()
    assert d['path_mode'] == 'shortest'
    p2 = ComparisonParameters.from_dict(d)
    assert p2.path_mode == 'shortest'
    # Legacy saved parameters without the key fall back to 'all'.
    d_legacy = p.to_dict()
    del d_legacy['path_mode']
    assert ComparisonParameters.from_dict(d_legacy).path_mode == 'all'


def test_run_path_analysis_selects_find_tool_by_path_mode(monkeypatch):
    """run_path_analysis must call FindShortestPath in shortest mode and
    FindAllPath in all mode, with identical constructor wiring otherwise."""
    import coana
    from comparison import ComparisonAnalyzer

    calls = {'init': [], 'method': None}

    class FakeFNC:
        def __init__(self, **kwargs):
            calls['init'].append(kwargs)

        def InitializeNeuronInfo(self):
            pass

        def FindAllPath(self, **kwargs):
            calls['method'] = 'FindAllPath'

        def FindShortestPath(self, **kwargs):
            calls['method'] = 'FindShortestPath'

        allpath_folder = ''

    monkeypatch.setattr(coana, 'FindNeuronConnection', FakeFNC)

    def _run(path_mode):
        calls['method'] = None
        params = ComparisonParameters(
            datasets=['hemibrain:v1.2.1'],
            source_neurons=['aMe12'], target_neurons=['PPL101'],
            thresholds=[5], max_interlayer=0, path_mode=path_mode,
            verbose=False,
        )
        analyzer = ComparisonAnalyzer(params, verbose=False)
        analyzer._get_dataset_config = lambda name: None
        analyzer.parameters.get_source_neurons_for_dataset = lambda name: ['aMe12']
        analyzer.parameters.get_target_neurons_for_dataset = lambda name: ['PPL101']
        analyzer.parameters.get_dataset_output_path = lambda name, thr: '/tmp/fake_out'
        return analyzer.run_path_analysis('hemibrain:v1.2.1', 5,
                                          verbose_mode='silent')

    _run('shortest')
    assert calls['method'] == 'FindShortestPath'
    shortest_kwargs = calls['init'][0]

    _run('all')
    assert calls['method'] == 'FindAllPath'
    all_kwargs = calls['init'][1]

    # Same constructor wiring in both modes (path mode is handled inside
    # the FNC method call, not by different constructor args). The label
    # mapper is a fresh instance per analyzer run, so compare it by state.
    lm_shortest = shortest_kwargs.pop('label_mapper')
    lm_all = all_kwargs.pop('label_mapper')
    assert shortest_kwargs == all_kwargs
    assert (lm_shortest is None) == (lm_all is None)


def test_single_dataset_pipeline_runs_each_threshold(monkeypatch):
    """One dataset still runs the complete threshold iteration pipeline."""
    import pandas as pd
    from comparison import ComparisonAnalyzer

    params = ComparisonParameters(
        datasets=['hemibrain:v1.2.1'],
        source_neurons=['aMe12'],
        target_neurons=['PPL101'],
        thresholds=[3, 7],
        output_folder="",
        auto_type_mapping=False,
        verbose=False,
    )
    analyzer = ComparisonAnalyzer(params, verbose=False)
    calls = []

    def fake_run_path_analysis(dataset, threshold, verbose_mode="simple"):
        calls.append((dataset, threshold, verbose_mode))
        return pd.DataFrame()

    monkeypatch.setattr(analyzer, "run_path_analysis", fake_run_path_analysis)
    results = analyzer.run_all_analyses(skip_existing=False)
    summary = analyzer.run_comparison_analysis()

    assert calls == [
        ("hemibrain:v1.2.1", 3, "simple"),
        ("hemibrain:v1.2.1", 7, "silent"),
    ]
    assert sorted(results["hemibrain:v1.2.1"]) == [3, 7]
    assert summary["datasets"] == ["hemibrain:v1.2.1"]
    assert summary["thresholds"] == [3, 7]


def test_aggregate_and_find_paths_shortest_flags_only_shortest_route():
    """In shortest mode an edge is valid only when it lies on a per-pair
    minimum-hop path; 'all' mode keeps every on-path edge. max_layers=None
    (unlimited) must also work."""
    import pandas as pd
    from comparison import ComparisonAnalyzer

    params = ComparisonParameters(
        datasets=['hemibrain:v1.2.1'],
        source_neurons=['TS'], target_neurons=['TT'],
        thresholds=[1], verbose=False,
    )
    analyzer = ComparisonAnalyzer(params, verbose=False)

    # One bodyId per type: S->A->T (2 hops) plus the long route S->B->C->T.
    bodyid_df = pd.DataFrame({
        'bodyId_pre': ['S', 'A', 'S', 'B', 'C'],
        'bodyId_post': ['A', 'T', 'B', 'C', 'T'],
        'weight': [10, 10, 10, 10, 10],
    })
    label_map = {'S': 'TS', 'A': 'TA', 'T': 'TT', 'B': 'TB', 'C': 'TC'}

    res_shortest = analyzer._aggregate_and_find_paths(
        bodyid_df, label_map, {'TS'}, {'TT'},
        max_layers=None, dataset_name='hemibrain:v1.2.1', threshold=1,
        path_mode='shortest')
    flags = {(r['type_pre'], r['type_post']): bool(r['has_valid_path'])
             for _, r in res_shortest.iterrows()}
    assert flags[('TS', 'TA')] is True
    assert flags[('TA', 'TT')] is True
    # The long route carries no shortest path -> its edges are invalid.
    assert flags[('TS', 'TB')] is False
    assert flags[('TB', 'TC')] is False
    assert flags[('TC', 'TT')] is False

    res_all = analyzer._aggregate_and_find_paths(
        bodyid_df, label_map, {'TS'}, {'TT'},
        max_layers=4, dataset_name='hemibrain:v1.2.1', threshold=1,
        path_mode='all')
    flags_all = {(r['type_pre'], r['type_post']): bool(r['has_valid_path'])
                 for _, r in res_all.iterrows()}
    assert all(flags_all.values()), flags_all
