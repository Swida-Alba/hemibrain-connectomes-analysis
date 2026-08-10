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
    """The cross-dataset defaults match FindAllPath: 1M bodyId edge limit and
    2000 Visualization Edge Limit."""
    p = ComparisonParameters(
        datasets=['hemibrain:v1.2.1', 'male-cns:v1.0'])
    assert p.graph_edge_limit_bodyid == 1000000
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
