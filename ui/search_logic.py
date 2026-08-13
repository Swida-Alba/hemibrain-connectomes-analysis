"""Compatibility exports for the shared neuron-search contract.

The semantic implementation lives in :mod:`src.neuron_search` so analysis
backends, the inline suggestion menu, and the available-neurons viewer import
the same stages and matching helpers.  This module keeps the historic UI
import path stable for components and third-party extensions.
"""

try:
    from src.neuron_search import (
        SearchStage,
        filter_candidate_entries,
        is_numeric_search,
        match_search_pools,
        normalize_search_text,
        ordered_search_columns,
        search_plan,
    )
except ImportError:  # pragma: no cover - supports ``src/`` on sys.path
    from neuron_search import (  # type: ignore
        SearchStage,
        filter_candidate_entries,
        is_numeric_search,
        match_search_pools,
        normalize_search_text,
        ordered_search_columns,
        search_plan,
    )


__all__ = [
    "SearchStage",
    "filter_candidate_entries",
    "is_numeric_search",
    "match_search_pools",
    "normalize_search_text",
    "ordered_search_columns",
    "search_plan",
]
