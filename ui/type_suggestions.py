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


def _valid_text(value) -> Optional[str]:
    """Return a useful display/search string, or ``None`` for a null value."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() in {
        "nan", "none", "null", "<na>", "<null>",
    }:
        return None
    return text


def _body_id_entries(body_ids, instances) -> List[Entry]:
    """Build bodyId suggestions with the corresponding instance as hint."""
    body_entries = []
    for body_id, instance in zip(body_ids, instances):
        body_id_text = _valid_text(body_id)
        if body_id_text is None:
            continue
        hint = _valid_text(instance) or "bodyId"
        body_entries.append((body_id_text, hint))
    return sorted(set(body_entries))


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
        for col in _TYPE_COLUMNS:
            if col not in cols:
                continue
            values = frame[col].drop_nulls().to_list()
            pools[col] = [(v, col) for v in _clean(values)]
        # Extra type-like columns are present in some cached indexes (for
        # example flywireType/hemibrainType). Keep them in the same search
        # universe as the backend's auto-column lookup.
        for col in sorted(cols):
            if col in ("bodyId", "type", "instance"):
                continue
            low = str(col).lower()
            if any(key in low for key in _EXTRA_TYPE_COLUMNS_PATTERN):
                values = frame[col].drop_nulls().to_list()
                pools[col] = [(v, col) for v in _clean(values)]
        # bodyId pool: string-form ids, hint = the corresponding instance.
        if "bodyId" in cols:
            bid_col = frame["bodyId"].cast(pl.Utf8, strict=False)
            inst_col = frame["instance"].cast(pl.Utf8, strict=False) \
                if "instance" in cols else None
            pools["bodyId"] = _body_id_entries(
                bid_col.to_list(),
                inst_col.to_list() if inst_col is not None
                else [None] * len(bid_col),
            )
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
            for col in sorted(cols):
                low = col.lower()
                if col not in pools and any(k in low for k in _EXTRA_TYPE_COLUMNS_PATTERN):
                    pools[col] = [(v, col) for v in _clean(df[col].tolist())]
            if "bodyId" in cols:
                inst_series = df["instance"].astype(str) if "instance" in cols \
                    else None
                pools["bodyId"] = _body_id_entries(
                    df["bodyId"].tolist(),
                    inst_series.tolist() if inst_series is not None
                    else [None] * len(df),
                )
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

    Matching is deliberately staged across the whole search scope:

    1. strict, case-sensitive type prefixes;
    2. strict, case-sensitive prefixes in the remaining auto-search columns;
    3. case-insensitive substring matches, with type still preferred before
       the remaining columns.

    The stages are global rather than per-column: a type substring cannot
    hide a more precise instance/bodyId prefix. Canonical names such as
    ``aMe12`` therefore stay precise while a mistyped or mid-name query can
    still recover.

    - numeric input -> bodyId (hint = the instance)
    - string input -> type FIRST; only when no type matches and the search
      scope is 'auto', expand to instance / bodyId / extra type columns
    - explicit scope ('type' / 'instance' / 'bodyId') -> that column only
    """
    text = str(text).strip()
    if not text or not pools:
        return []
    scope = str(search_columns or "auto").strip().lower()
    if scope not in ("auto", "type", "instance", "bodyid"):
        scope = "auto"

    def prefix_matches(entries: List[Entry]) -> List[Entry]:
        """Strict prefix stage; preserve canonical case in the input."""
        return [entry for entry in entries if entry[0].startswith(text)]

    def substring_matches(entries: List[Entry]) -> List[Entry]:
        """Forgiving fallback stage, reached only after all prefixes fail."""
        folded = text.casefold()
        return [entry for entry in entries if folded in entry[0].casefold()]

    def ordered_columns() -> List[str]:
        """Search columns in the same priority used by the backend."""
        return [
            *[col for col in ("type", "instance", "bodyId") if col in pools],
            *sorted(col for col in pools
                    if col not in ("type", "instance", "bodyId")),
        ]

    def collect(columns: Sequence[str], matcher) -> List[Entry]:
        """Collect candidates by column, stopping at the display limit."""
        out: List[Entry] = []
        for column in columns:
            out.extend(matcher(pools.get(column, [])))
            if len(out) >= limit:
                return out[:limit]
        return out

    is_numeric = text.replace(".", "").isdigit()

    if scope == "auto":
        if is_numeric:
            prefix = prefix_matches(pools.get("bodyId", []))
            return (prefix or substring_matches(pools.get("bodyId", [])))[:limit]

        columns = ordered_columns()
        type_prefix = prefix_matches(pools.get("type", []))
        if type_prefix:
            return type_prefix[:limit]

        other_columns = [col for col in columns if col != "type"]
        other_prefix = collect(other_columns, prefix_matches)
        if other_prefix:
            return other_prefix

        # The last stage is the first intentionally broad stage: preserve
        # type-first ordering, but include matching values from the other
        # columns too so the gray column hint explains why each candidate is
        # present.
        type_substring = substring_matches(pools.get("type", []))
        other_substring = collect(other_columns, substring_matches)
        return (type_substring + other_substring)[:limit]

    col = "bodyId" if scope == "bodyid" else scope
    prefix = prefix_matches(pools.get(col, []))
    return (prefix or substring_matches(pools.get(col, [])))[:limit]


def entry_hint(value: str, pools: Dict[str, List[Entry]]) -> Optional[str]:
    """Column of a value in the pools (type/instance/bodyId priority), for
    labeling history entries; None when the value is not in any pool."""
    for col in ("type", "instance", "bodyId"):
        if any(v == str(value) for v, _ in pools.get(col, [])):
            return col
    return None
