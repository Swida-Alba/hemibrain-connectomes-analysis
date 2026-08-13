"""Custom-group label history: a label-keyed registry for easy reuse.

Every run that exports inline custom groups records its labels here so the
grouper's "Add from history" menu can refill them later. Semantics:

- One record per label: ``members`` maps dataset -> neuron list.
- CELL-GRANULARITY UPSERT: a run's board is the complete statement for the
  dataset columns it displays, so every displayed cell (empty included)
  overwrites that label+dataset slot; datasets NOT on the board keep their
  stored members. This is how a label accumulates members across datasets
  over time while a redefined cell covers its previous value.
- Auto-generated labels (blank names exported as ``Group_N``) are never
  recorded — history keys on meaningful user names only.
- Identical cell contents only bump ``updated_at`` and move the label to
  the front of ``recent`` (capped).

Storage: ``cache/user_mappings/group_history.json`` (gitignored dir),
thread-locked and written atomically (tmp + replace), mirroring the
project's auto-save pattern. Independent of the named preset store:
saving a preset never prunes history and vice versa.
"""
import json
import os
import re
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .config import PROJECT_ROOT

HISTORY_PATH = PROJECT_ROOT / "cache" / "user_mappings" / "group_history.json"
RECENT_CAP = 30
# Labels matching the export auto-name pattern carry no user intent.
AUTO_LABEL_RE = re.compile(r"^Group_\d+$")

_lock = threading.Lock()


def _has_members(record: dict) -> bool:
    """Whether a stored group contains at least one reusable member."""
    members = record.get("members") if isinstance(record, dict) else None
    if not isinstance(members, dict):
        return False
    return any(
        isinstance(values, (list, tuple))
        and any(str(value).strip() for value in values)
        for values in members.values()
    )


def _load() -> dict:
    try:
        if HISTORY_PATH.exists():
            data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("labels"), dict):
                data.setdefault("recent", [])
                return data
    except (OSError, ValueError):
        pass
    return {"labels": {}, "recent": []}


def _save(data: dict) -> bool:
    """Atomic write: tmp file in the same directory, then replace."""
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = HISTORY_PATH.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(HISTORY_PATH)
        return True
    except OSError:
        return False


def record(groups: List[Tuple[str, Dict[str, List[str]]]],
           origin: str = "inline") -> int:
    """Upsert exported group rows into the history.

    *groups* is a list of ``(label, {dataset: [members]})`` pairs using the
    ORIGINAL label names as typed by the user — blank/auto-generated names
    must be passed as ``""`` so they can be skipped. Returns the number of
    labels actually recorded.
    """
    now = datetime.now().isoformat(timespec="seconds")
    recorded = 0
    with _lock:
        data = _load()
        labels: Dict[str, dict] = data["labels"]
        recent: List[str] = data["recent"]
        for label, cells in groups:
            label = (label or "").strip()
            if not label or AUTO_LABEL_RE.match(label):
                continue  # auto-named groups carry no reusable identity
            record = labels.setdefault(
                label, {"members": {}, "updated_at": now, "origin": origin})
            for ds, members in (cells or {}).items():
                cleaned = [str(m).strip() for m in members if str(m).strip()]
                record["members"][str(ds)] = cleaned
            record["updated_at"] = now
            record["origin"] = origin
            if label in recent:
                recent.remove(label)
            recent.insert(0, label)
            recorded += 1
        data["recent"] = recent[:RECENT_CAP]
        _save(data)
    return recorded


def list_recent(limit: int = RECENT_CAP) -> List[str]:
    """Labels ordered by most recently used."""
    with _lock:
        data = _load()
        return [
            label for label in data["recent"]
            if label in data["labels"] and _has_members(data["labels"][label])
        ][:limit]


def get_label(label: str) -> Optional[dict]:
    """The stored record for *label* ({members, updated_at, origin}), or None."""
    label = (label or "").strip()
    if not label:
        return None
    with _lock:
        rec = _load()["labels"].get(label)
        return dict(rec, members={k: list(v) for k, v in rec.get("members", {}).items()}) if rec else None


def all_labels() -> Dict[str, dict]:
    """A copy of the whole label registry."""
    with _lock:
        return {k: dict(v) for k, v in _load()["labels"].items()}


def valid_labels() -> Dict[str, dict]:
    """Return named groups that can actually be reused from history.

    A record with only empty dataset cells is retained in storage for
    compatibility with cell-level upsert semantics, but it is not a valid
    history entry because loading it cannot populate a query or mapping.
    """
    with _lock:
        return {
            label: dict(record)
            for label, record in _load()["labels"].items()
            if _has_members(record)
        }


def remove_label(label: str) -> bool:
    """Drop one label from the registry. Returns success."""
    label = (label or "").strip()
    with _lock:
        data = _load()
        if label not in data["labels"]:
            return False
        del data["labels"][label]
        if label in data["recent"]:
            data["recent"].remove(label)
        return _save(data)


def clear() -> bool:
    """Wipe the whole history. Returns success."""
    with _lock:
        return _save({"labels": {}, "recent": []})
