"""Persistent neuron-query history for the auto-suggest dropdown.

Values are recorded when a pathfinding run actually starts (the chips that
were searched). The store keeps a per-value count and a last-used timestamp
in ``ui/neuron_history.json`` (gitignored); the dropdown shows the last 10
(recency) and the most frequent 5. All writes are atomic (tmp + replace)
and failures are swallowed — history is a convenience, never an error.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

_HISTORY_PATH = Path(__file__).resolve().parent / "neuron_history.json"
_LIMIT_RECENT = 10
_LIMIT_FREQUENT = 5
_LOCK = threading.Lock()


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


def record(values: List[str], now: Optional[str] = None) -> None:
    """Record searched values (raw chips, pre-pattern): bump the count and
    refresh the last-used timestamp of each value."""
    if not values:
        return
    stamp = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _LOCK:
        data = _load()
        for value in values:
            value = str(value).strip()
            if not value or value.lower() in ("nan", "none", "null"):
                continue
            entry = data.get(value, {"count": 0})
            entry["count"] = int(entry.get("count", 0)) + 1
            entry["last_used"] = stamp
            data[value] = entry
        _save(data)


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


def clear() -> None:
    """Wipe the history file (used by tests and the UI clear action)."""
    with _LOCK:
        _save({})
