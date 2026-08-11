"""Dataset type-name pools for the neuron-input auto-suggest.

Suggestion entries are ``(value, hint)`` pairs so the dropdown can render the
matched name with a gray hint of the searched column (for bodyId matches the
hint is the corresponding instance). Pools are read from LOCAL dataset files
only — ``cache/<dataset>/neuron_index.parquet`` first, falling back to the
``datasets/<dataset>`` neuron tables — and cached per (dataset, file mtime);
the server is never contacted, so a dataset that has not been pulled yet
simply yields no suggestions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .config import PROJECT_ROOT
from .dataset_service import dataset_to_folder

# (value, hint) — the hint is the searched column name, except for bodyId
# matches where it is the corresponding instance.
Entry = Tuple[str, str]

_CACHE_DIR = PROJECT_ROOT / "cache"
_DATASETS_DIR = PROJECT_ROOT / "datasets"

_POOL_CACHE: Dict[tuple, Dict[str, List[Entry]]] = {}

# Columns whose distinct values become suggestion pools (hint = column name).
_TYPE_COLUMNS = ("type", "instance")
# Extra type-like columns from the dataset tables (e.g. flywireType,
# hemibrainType, mancType) — the backend 'auto' search covers them too.
_EXTRA_TYPE_COLUMNS_PATTERN = ("type", "cell_type", "celltype")

_EMPTY_POOLS: Dict[str, List[Entry]] = {}


def _clean(values) -> List[str]:
    """Deduplicated, sorted, non-empty string values from a column."""
    seen = set()
    out = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() in ("nan", "none", "null", "<na>", "<null>"):
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return sorted(out)


def _index_pools(folder: str) -> Optional[Dict[str, List[Entry]]]:
    """Pools from cache/<folder>/neuron_index.parquet (None when absent)."""
    index = _CACHE_DIR / folder / "neuron_index.parquet"
    if not index.exists():
        return None
    try:
        import polars as pl

        frame = pl.read_parquet(index)
        cols = set(frame.columns)
        pools: Dict[str, List[Entry]] = {}
        for col in ("type", "instance"):
            if col not in cols:
                continue
            values = frame[col].drop_nulls().to_list()
            pools[col] = [(v, col) for v in _clean(values)]
        # bodyId pool: string-form ids, hint = the corresponding instance.
        if "bodyId" in cols:
            bid_col = frame["bodyId"].cast(pl.Utf8, strict=False)
            inst_col = frame["instance"].cast(pl.Utf8, strict=False) \
                if "instance" in cols else None
            rows = dict(zip(bid_col.to_list(), inst_col.to_list() if inst_col is not None
                            else [None] * len(bid_col)))
            body_entries = []
            for bid, inst in rows.items():
                if bid is None or str(bid).strip() in ("", "nan"):
                    continue
                hint = str(inst) if inst is not None and str(inst).strip() \
                    and str(inst).strip().lower() not in ("nan", "none") else "bodyId"
                body_entries.append((str(bid), hint))
            pools["bodyId"] = sorted(set(body_entries))
        return pools
    except Exception:
        return None


def _table_pools(folder: str) -> Optional[Dict[str, List[Entry]]]:
    """Pools from the datasets/<folder> neuron tables (None when absent)."""
    ds_dir = _DATASETS_DIR / folder
    if not ds_dir.exists():
        return None
    candidates = sorted(
        p for pattern in ("*_neuron_df.csv", "*_neuron_df.parquet",
                          "*_allneurons*.csv", "*_allneurons*.parquet")
        for p in ds_dir.glob(pattern)
    )
    if not candidates:
        return None
    for table in candidates:
        try:
            if table.suffix == ".parquet":
                df = __import__("pandas").read_parquet(table)
            else:
                df = __import__("pandas").read_csv(table)
            cols = set(df.columns)
            if not {"bodyId", "type"}.issubset(cols):
                continue
            pools: Dict[str, List[Entry]] = {}
            for col in _TYPE_COLUMNS:
                if col in cols:
                    pools[col] = [(v, col) for v in _clean(df[col].tolist())]
            for col in cols:
                low = col.lower()
                if col not in pools and any(k in low for k in _EXTRA_TYPE_COLUMNS_PATTERN):
                    pools[col] = [(v, col) for v in _clean(df[col].tolist())]
            if "bodyId" in cols:
                inst_series = df["instance"].astype(str) if "instance" in cols \
                    else None
                rows = dict(zip(df["bodyId"].astype(str).tolist(),
                                inst_series.tolist() if inst_series is not None
                                else [None] * len(df)))
                body_entries = []
                for bid, inst in rows.items():
                    if bid.strip() in ("", "nan"):
                        continue
                    hint = inst if inst and inst.strip() and inst.strip().lower() not in ("nan", "none") else "bodyId"
                    body_entries.append((bid, hint))
                pools["bodyId"] = sorted(set(body_entries))
            return pools
        except Exception:
            continue
    return None


def _folder_pools(folder: str) -> Dict[str, List[Entry]]:
    # Cache key covers every local source file's mtime so a rebuilt index
    # or an updated dataset table invalidates the pools.
    sources = []
    index = _CACHE_DIR / folder / "neuron_index.parquet"
    if index.exists():
        sources.append((str(index), index.stat().st_mtime_ns))
    ds_dir = _DATASETS_DIR / folder
    if ds_dir.exists():
        for pattern in ("*_neuron_df.csv", "*_neuron_df.parquet",
                        "*_allneurons*.csv", "*_allneurons*.parquet"):
            for p in ds_dir.glob(pattern):
                sources.append((str(p), p.stat().st_mtime_ns))
    key = ("pools", folder, tuple(sorted(sources)))
    if key in _POOL_CACHE:
        return _POOL_CACHE[key]
    pools = _index_pools(folder)
    if pools is None:
        pools = _table_pools(folder)
    if pools is None:
        pools = _EMPTY_POOLS
    _POOL_CACHE[key] = pools
    return pools


def get_dataset_pools(dataset: str) -> Dict[str, List[Entry]]:
    """``{column: [(value, hint), ...]}`` for one dataset; {} when nothing
    is available locally (dataset not pulled / no local tables)."""
    if not dataset:
        return _EMPTY_POOLS
    return _folder_pools(dataset_to_folder(str(dataset)))


def suggestion_pool(datasets: Sequence[str]) -> Dict[str, List[Entry]]:
    """Union of per-dataset pools (cross-dataset tab): same column keys,
    values deduplicated across datasets."""
    merged: Dict[str, List[Entry]] = {}
    for ds in datasets:
        for col, entries in get_dataset_pools(ds).items():
            seen = {v for v, _ in merged.get(col, [])}
            merged[col] = merged.get(col, []) + [
                e for e in entries if e[0] not in seen
            ]
    return merged


def match_suggestions(
    text: str,
    pools: Dict[str, List[Entry]],
    search_columns: str = "auto",
    limit: int = 50,
) -> List[Entry]:
    """Match ``text`` against the pools with the backend's column priority.

    - numeric input -> bodyId (hint = the instance)
    - string input -> type FIRST; only when no type matches and the search
      scope is 'auto', expand to instance / bodyId / extra type columns
    - explicit scope ('type' / 'instance' / 'bodyId') -> that column only
    """
    text = str(text).strip()
    if not text or not pools:
        return []
    lower = text.lower()
    scope = str(search_columns or "auto").strip().lower()
    if scope not in ("auto", "type", "instance", "bodyid"):
        scope = "auto"

    def matches(entries: List[Entry]) -> List[Entry]:
        return [e for e in entries if e[0].lower().startswith(lower)]

    is_numeric = text.replace(".", "").isdigit()

    if scope == "auto":
        if is_numeric:
            return matches(pools.get("bodyId", []))[:limit]
        out = matches(pools.get("type", []))[:limit]
        if out:
            return out
        # No type matched: expand the range to the other columns.
        out = matches(pools.get("instance", []))[:limit]
        if len(out) < limit:
            out += matches(pools.get("bodyId", []))[: limit - len(out)]
        for col in sorted(c for c in pools
                          if c not in ("type", "instance", "bodyId")):
            if len(out) >= limit:
                break
            out += matches(pools[col])[: limit - len(out)]
        return out[:limit]

    col = "bodyId" if scope == "bodyid" else scope
    return matches(pools.get(col, []))[:limit]


def entry_hint(value: str, pools: Dict[str, List[Entry]]) -> Optional[str]:
    """Column of a value in the pools (type/instance/bodyId priority), for
    labeling history entries; None when the value is not in any pool."""
    for col in ("type", "instance", "bodyId"):
        if any(v == str(value) for v, _ in pools.get(col, [])):
            return col
    return None
