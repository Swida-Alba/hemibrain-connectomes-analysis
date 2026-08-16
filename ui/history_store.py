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


def _normalize_datasets(datasets) -> Optional[Set[str]]:
    """Normalize a dataset selector value for history filtering.

    ``None`` means the caller has no dataset context and keeps the legacy
    unfiltered behavior. An empty selector is also treated as unscoped so
    users can inspect existing history before choosing a dataset.
    """
    if datasets is None:
        return None
    if isinstance(datasets, str):
        values = [datasets]
    else:
        try:
            values = list(datasets)
        except TypeError:
            values = [datasets]
    normalized = {
        str(value).strip()
        for value in values
        if isinstance(value, str) and str(value).strip()
    }
    return normalized or None


def _nonempty_group_datasets(value: str) -> Set[str]:
    """Find datasets carrying members for a saved custom group or mapping."""
    datasets: Set[str] = set()
    try:
        from . import group_history

        record = group_history.get_label(value)
        members = record.get("members") if isinstance(record, dict) else None
        if isinstance(members, dict):
            datasets.update(
                str(dataset).strip()
                for dataset, values in members.items()
                if str(dataset).strip()
                and any(str(member).strip() for member in (values or []))
            )
    except Exception:  # pragma: no cover - history must never break a run
        pass

    # Named LabelMapper presets use the same dataset-keyed shape as group
    # history. This covers a saved mapping whose label was used as a query
    # history value by an older client.
    try:
        from . import mapping_store

        mapping = mapping_store.get_mapping(value)
        if isinstance(mapping, dict):
            for side_name in mapping_store.MAPPING_SIDES:
                side = mapping.get(side_name)
                if not isinstance(side, dict):
                    continue
                for dataset, groups in side.items():
                    if dataset in {"custom_label", "std_label"}:
                        continue
                    if isinstance(groups, list) and any(
                        isinstance(group, (list, tuple))
                        and any(str(member).strip() for member in group)
                        for group in groups
                    ):
                        datasets.add(str(dataset).strip())
    except Exception:  # pragma: no cover - optional preset store
        pass
    return datasets


def _matches_dataset_scope(value: str, entry: dict,
                           scope: Optional[Set[str]]) -> bool:
    """Whether one stored value belongs in a dataset-scoped history list."""
    if scope is None:
        return True

    recorded = _normalize_datasets(entry.get("datasets")) or set()
    if recorded and not recorded.intersection(scope):
        return False

    custom_datasets = _nonempty_group_datasets(value)
    if custom_datasets and not custom_datasets.intersection(scope):
        return False

    # Entries written before dataset provenance was introduced remain visible;
    # new records carry ``datasets`` and are filtered above. Custom entries
    # additionally use their live group/map member datasets when available.
    return True


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
           custom_values: Optional[Iterable[str]] = None,
           datasets=None) -> None:
    """Record searched values (raw chips, pre-pattern): bump the count and
    refresh the last-used timestamp of each value.

    Values that are current custom-group labels are marked so a later removal
    can invalidate old query-history entries even when the UI was not open at
    the time of deletion. ``custom_values`` is injectable for callers and
    tests; omitted values are resolved from the group registry. ``datasets``
    records the selected dataset(s) so later history menus can use the same
    scope as suggestions.
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
            dataset_values = _normalize_datasets(datasets)
            if dataset_values:
                prior_datasets = {
                    str(dataset).strip()
                    for dataset in (entry.get("datasets") or [])
                    if str(dataset).strip()
                }
                entry["datasets"] = sorted(prior_datasets | dataset_values)
            data[value] = entry
        _save(data)


def datasets_of(value: str) -> List[str]:
    """Recorded dataset names for one history value, sorted and deduplicated.

    Entries recorded before dataset provenance was introduced return ``[]``;
    the dropdown renders no dataset tags for them.
    """
    value = str(value or "").strip()
    if not value:
        return []
    with _LOCK:
        data = _load()
        entry = data.get(value)
        if not isinstance(entry, dict):
            return []
        return sorted({
            str(dataset).strip()
            for dataset in (entry.get("datasets") or [])
            if str(dataset).strip()
        })


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


def recent(limit: int = _LIMIT_RECENT, datasets=None) -> List[str]:
    """Most recently searched values, optionally scoped to datasets."""
    scope = _normalize_datasets(datasets)
    with _LOCK:
        data = _load()
        ordered = sorted(data.items(),
                         key=lambda kv: str(kv[1].get("last_used", "")),
                         reverse=True)
        return [
            value for value, entry in ordered
            if _matches_dataset_scope(value, entry, scope)
        ][:limit]


def frequent(limit: int = _LIMIT_FREQUENT, datasets=None) -> List[str]:
    """Most frequently searched values, optionally scoped to datasets."""
    scope = _normalize_datasets(datasets)
    with _LOCK:
        data = _load()
        ordered = sorted(
            data.items(),
            key=lambda kv: (int(kv[1].get("count", 0)),
                            str(kv[1].get("last_used", ""))),
            reverse=True,
        )
        return [
            value for value, entry in ordered
            if _matches_dataset_scope(value, entry, scope)
        ][:limit]


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
