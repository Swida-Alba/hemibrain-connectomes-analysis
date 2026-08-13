"""Persistent neuron-query history for the auto-suggest dropdown.

Values are recorded after the runner confirms that the submitted query found
neurons (the chips that were searched). The store keeps a per-value count and a last-used timestamp
in ``ui/neuron_history.json`` (gitignored); the dropdown shows the last 10
(recency) and the most frequent 5. All writes are atomic (tmp + replace)
and failures are swallowed — history is a convenience, never an error.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

_HISTORY_PATH = Path(__file__).resolve().parent / "neuron_history.json"
_LIMIT_RECENT = 10
_LIMIT_FREQUENT = 5
_LOCK = threading.Lock()


def _current_custom_values() -> Set[str]:
    """Return currently reusable custom-group labels, if available.

    Query history is used by several UI components, so keeping the optional
    provenance lookup here avoids requiring every run tab to know about the
    custom-group registry. The local import also keeps this store usable in
    isolation and during test bootstrap.
    """
    try:
        from . import group_history

        valid = getattr(group_history, "valid_labels", None)
        labels = valid() if callable(valid) else group_history.all_labels()
        return {str(label).strip() for label in labels if str(label).strip()}
    except Exception:  # pragma: no cover - history must never break a run
        return set()


def _load() -> Dict[str, dict]:
    try:
        if _HISTORY_PATH.exists():
            data = json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("values"), dict):
                return data["values"]
    except (OSError, ValueError):
        pass
    return {}


def _save(values: Dict[str, dict]) -> None:
    try:
        tmp = _HISTORY_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"values": values}, indent=2),
                       encoding="utf-8")
        tmp.replace(_HISTORY_PATH)
    except OSError:
        pass


def record(values: List[str], now: Optional[str] = None,
           custom_values: Optional[Iterable[str]] = None) -> None:
    """Record searched values (raw chips, pre-pattern): bump the count and
    refresh the last-used timestamp of each value.

    Values that are current custom-group labels are marked so a later removal
    can invalidate old query-history entries even when the UI was not open at
    the time of deletion. ``custom_values`` is injectable for callers and
    tests; omitted values are resolved from the group registry.
    """
    if not values:
        return
    stamp = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    custom_keys = {
        str(value).strip() for value in (
            _current_custom_values()
            if custom_values is None else custom_values
        ) if str(value).strip()
    }
    with _LOCK:
        data = _load()
        for value in values:
            value = str(value).strip()
            if not value or value.lower() in ("nan", "none", "null"):
                continue
            entry = data.get(value, {"count": 0})
            entry["count"] = int(entry.get("count", 0)) + 1
            entry["last_used"] = stamp
            if value in custom_keys:
                entry["kind"] = "custom"
            else:
                # A label can later be searched as an ordinary query after
                # its custom group has been removed; do not retain stale
                # custom provenance in that case.
                entry.pop("kind", None)
            data[value] = entry
        _save(data)


def mark_custom(values: Iterable[str]) -> None:
    """Mark already-recorded query values as custom-group entries."""
    keys = {str(value).strip() for value in values if str(value).strip()}
    if not keys:
        return
    with _LOCK:
        data = _load()
        changed = False
        for value in keys:
            entry = data.get(value)
            if isinstance(entry, dict) and entry.get("kind") != "custom":
                entry["kind"] = "custom"
                changed = True
        if changed:
            _save(data)


def prune_orphaned_custom(valid_values: Iterable[str]) -> List[str]:
    """Delete custom-marked query history whose group no longer exists.

    Unknown legacy entries are intentionally preserved because they may be
    ordinary neuron types, body IDs, or patterns. Only entries with explicit
    custom provenance are safe to classify as invalid here.
    """
    valid = {str(value).strip() for value in valid_values
             if str(value).strip()}
    with _LOCK:
        data = _load()
        stale = [
            value for value, entry in data.items()
            if isinstance(entry, dict)
            and (entry.get("kind") == "custom" or entry.get("custom") is True)
            and value not in valid
        ]
        if stale:
            for value in stale:
                del data[value]
            _save(data)
        return stale


def recent(limit: int = _LIMIT_RECENT) -> List[str]:
    """Most recently searched values, newest first."""
    with _LOCK:
        data = _load()
        ordered = sorted(data.items(),
                         key=lambda kv: str(kv[1].get("last_used", "")),
                         reverse=True)
        return [v for v, _ in ordered[:limit]]


def frequent(limit: int = _LIMIT_FREQUENT) -> List[str]:
    """Most frequently searched values (count desc, recency as tie-break)."""
    with _LOCK:
        data = _load()
        ordered = sorted(
            data.items(),
            key=lambda kv: (int(kv[1].get("count", 0)),
                            str(kv[1].get("last_used", ""))),
            reverse=True,
        )
        return [v for v, _ in ordered[:limit]]


def remove(value: str) -> bool:
    """Remove one value from query history and return whether it existed."""
    value = str(value or "").strip()
    if not value:
        return False
    with _LOCK:
        data = _load()
        if value not in data:
            return False
        del data[value]
        _save(data)
        return True


def clear() -> None:
    """Wipe the history file (used by tests and the UI clear action)."""
    with _LOCK:
        _save({})
