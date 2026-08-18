"""Persistent, shared driver-line history for NeuronBridge and FlyLight inputs.

This module deliberately uses a different file from :mod:`history_store` so
driver-line names cannot appear in the neuron-query history (or vice versa).
All three driver-line workflows use this same store. Line history is global:
dataset provenance is intentionally ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from . import history_store as _store


_HISTORY_PATH = Path(__file__).resolve().parent / "line_history.json"
_LIMIT_RECENT = _store._LIMIT_RECENT
_LIMIT_FREQUENT = _store._LIMIT_FREQUENT


def record(values: List[str], now: Optional[str] = None,
           custom_values: Optional[Iterable[str]] = None,
           datasets=None) -> None:
    # Keep the public signature compatible with the generic history API, but
    # never persist dataset provenance for driver lines.
    return _store.record(
        values,
        now=now,
        custom_values=custom_values,
        datasets=None,
        _history_path=_HISTORY_PATH,
    )


def datasets_of(value: str) -> List[str]:
    # Line history is deliberately not dataset-aware.
    return []


def mark_custom(values: Iterable[str]) -> None:
    return _store.mark_custom(values, _history_path=_HISTORY_PATH)


def prune_orphaned_custom(valid_values: Iterable[str]) -> List[str]:
    return _store.prune_orphaned_custom(
        valid_values, _history_path=_HISTORY_PATH
    )


def recent(limit: int = _LIMIT_RECENT, datasets=None) -> List[str]:
    return _store.recent(
        limit=limit, datasets=None, _history_path=_HISTORY_PATH
    )


def frequent(limit: int = _LIMIT_FREQUENT, datasets=None) -> List[str]:
    return _store.frequent(
        limit=limit, datasets=None, _history_path=_HISTORY_PATH
    )


def remove(value: str) -> bool:
    return _store.remove(value, _history_path=_HISTORY_PATH)


def clear() -> None:
    return _store.clear(_history_path=_HISTORY_PATH)
