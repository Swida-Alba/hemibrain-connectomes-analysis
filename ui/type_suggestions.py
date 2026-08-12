"""Dataset type-name pools for the neuron-input auto-suggest.

Suggestion entries are ``(value, hint)`` pairs so the dropdown can render the
matched name with a gray hint of the searched column (for bodyId matches the
hint is the corresponding instance). Pools are read from LOCAL dataset files
only — ``cache/<dataset>/neuron_index.parquet`` first, supplemented by the
``datasets/<dataset>`` neuron tables when the cache is incomplete — and cached
per (dataset, file mtime); the server is never contacted, so a dataset that
has not been pulled yet simply yields no suggestions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .config import PROJECT_ROOT
from .dataset_service import dataset_to_folder

try:
    from src.neuron_index_builder import (
        priority_metadata_columns,
        read_metadata_projection,
    )
except ImportError:
    from neuron_index_builder import (
        priority_metadata_columns,
        read_metadata_projection,
    )

# (value, hint) — the hint is the searched column name, except for bodyId
# matches where it is the corresponding instance.
Entry = Tuple[str, str]

_CACHE_DIR = PROJECT_ROOT / "cache"
_DATASETS_DIR = PROJECT_ROOT / "datasets"

_POOL_CACHE: Dict[tuple, Dict[str, List[Entry]]] = {}

# Columns whose distinct values become suggestion pools (hint = column name).
_TYPE_COLUMNS = ("type", "instance")
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

        schema_columns = pl.scan_parquet(index).collect_schema().names()
        selected_columns = [
            column for column in (
                "bodyId", "type", "instance",
                *priority_metadata_columns(schema_columns),
            )
            if column in schema_columns
        ]
        frame = pl.read_parquet(
            index,
            columns=list(dict.fromkeys(selected_columns)),
        )
        cols = set(frame.columns)
        pools: Dict[str, List[Entry]] = {}
        for col in _TYPE_COLUMNS:
            if col not in cols:
                continue
            values = frame[col].drop_nulls().to_list()
            pools[col] = [(v, col) for v in _clean(values)]
        # Auto-suggestion expands beyond the canonical type column only into
        # type/class taxonomy fields.  The viewer still displays every
        # retained string field and searches it when explicitly requested.
        for col in priority_metadata_columns(frame.columns):
            if col not in cols:
                continue
            values = frame[col].cast(pl.Utf8, strict=False).to_list()
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


def _index_has_search_projection(folder: str) -> bool:
    """Whether the cache index already contains the pulled metadata fields."""
    index = _CACHE_DIR / folder / "neuron_index.parquet"
    if not index.exists():
        return False
    try:
        import polars as pl

        columns = set(pl.scan_parquet(index).collect_schema().names())
        # Legacy connection indexes contain only bodyId/type/instance/post and
        # cache bookkeeping.  A generated projection has at least one extra
        # metadata field and therefore does not need a second CSV scan.
        legacy = {
            "bodyId", "type", "instance", "post", "downstream_complete",
            "last_fetched", "connection_count",
        }
        return bool(columns - legacy)
    except Exception:
        return False


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
            import polars as pl
            frame = read_metadata_projection(table)
            cols = set(frame.columns)
            if not {"bodyId", "type"}.issubset(cols):
                continue
            pools: Dict[str, List[Entry]] = {}
            for col in _TYPE_COLUMNS:
                if col in cols:
                    values = frame[col].cast(pl.Utf8, strict=False).to_list()
                    pools[col] = [(v, col) for v in _clean(values)]
            for col in priority_metadata_columns(frame.columns):
                if col in pools or col in ("bodyId", "type", "instance"):
                    continue
                values = frame[col].cast(pl.Utf8, strict=False).to_list()
                pools[col] = [(v, col) for v in _clean(values)]
            if "bodyId" in cols:
                body_ids = frame["bodyId"].cast(pl.Utf8, strict=False).to_list()
                inst_series = (
                    frame["instance"].cast(pl.Utf8, strict=False).to_list()
                    if "instance" in cols else [None] * len(body_ids)
                )
                pools["bodyId"] = _body_id_entries(
                    body_ids,
                    inst_series,
                )
            return pools
        except Exception:
            continue
    return None


def _merge_pools(
    primary: Dict[str, List[Entry]],
    fallback: Dict[str, List[Entry]],
) -> Dict[str, List[Entry]]:
    """Supplement a cache pool with values from the local neuron table.

    The cache remains first: duplicate values keep its ordering and hint.
    Missing values from the table are appended so a partially populated
    connection index cannot hide valid names. BodyId hints are upgraded when
    the cache only has the generic ``bodyId`` label but the table knows the
    corresponding instance.
    """
    merged: Dict[str, List[Entry]] = {}
    for column in dict.fromkeys((*primary.keys(), *fallback.keys())):
        primary_entries = list(primary.get(column, []))
        fallback_entries = list(fallback.get(column, []))
        fallback_by_value = {value: hint for value, hint in fallback_entries}
        entries: List[Entry] = []
        seen = set()
        for value, hint in primary_entries:
            if (column == "bodyId" and hint == "bodyId"
                    and fallback_by_value.get(value)
                    and fallback_by_value[value] != "bodyId"):
                hint = fallback_by_value[value]
            if value not in seen:
                entries.append((value, hint))
                seen.add(value)
        for value, hint in fallback_entries:
            if value not in seen:
                entries.append((value, hint))
                seen.add(value)
        if entries:
            merged[column] = entries
    return merged


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
    index_pools = _index_pools(folder)
    if index_pools is not None and _index_has_search_projection(folder):
        # The generated rich index already contains the pulled metadata; do
        # not scan a 500+ MiB CSV again just to rebuild identical value pools.
        pools = index_pools
    else:
        table_pools = _table_pools(folder)
        # A legacy connection cache may be complete for bodyIds but only
        # partial for names. Supplement it instead of letting it mask the
        # full local table; otherwise valid names such as ``aMe12`` disappear.
        if index_pools is not None and table_pools is not None:
            pools = _merge_pools(index_pools, table_pools)
        elif index_pools is not None:
            pools = index_pools
        else:
            pools = table_pools
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


def filter_candidate_entries(
    text: str,
    candidates: Sequence[Entry],
) -> List[Entry]:
    """Narrow an existing candidate list with a case-sensitive prefix.

    This is intentionally smaller than :func:`match_suggestions`: it is the
    fast path used while a user appends characters to the same query. The
    caller must fall back to the full matcher when the query is not a strict
    continuation or this list produces no matches.
    """
    query = str(text).strip()
    if not query:
        return []
    return [entry for entry in candidates if str(entry[0]).startswith(query)]


def match_suggestions(
    text: str,
    pools: Dict[str, List[Entry]],
    search_columns: str = "auto",
    limit: Optional[int] = 50,
) -> List[Entry]:
    """Match ``text`` against the pools with the backend's column priority.

    Matching is deliberately staged across the whole search scope:

    1. strict, case-sensitive type prefixes;
    2. strict, case-sensitive prefixes in the remaining auto-search columns;
    3. case-sensitive substring matches, with type still preferred before
       the remaining columns.

    The stages are global rather than per-column: a type substring cannot
    hide a more precise instance/bodyId prefix. Canonical names such as
    ``aMe12`` therefore stay precise while a mistyped or mid-name query can
    still recover without changing the user's capitalization.

    - numeric input -> bodyId (hint = the instance)
    - string input -> type FIRST; only when no type matches and the search
      scope is 'auto', expand to instance / bodyId / extra type columns
    - explicit scope ('type' / 'instance' / 'bodyId') -> that column only

    ``limit=None`` returns the complete candidate pool. The UI providers use
    this mode because the input component applies its own display limit after
    each continuation is narrowed; limiting here would make names outside
    the first page impossible to reach by typing more characters.
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
        """Substring fallback, preserving the input's exact case."""
        return [entry for entry in entries if text in entry[0]]

    def ordered_columns() -> List[str]:
        """Search columns in the same priority used by the backend."""
        priority = priority_metadata_columns(pools)
        return [
            *[col for col in ("type", "instance", "bodyId") if col in pools],
            *[
                col for col in priority
                if col not in ("type", "instance", "bodyId")
            ],
            *sorted(
                col for col in pools
                if col not in ("type", "instance", "bodyId")
                and col not in priority
            ),
        ]

    def collect(columns: Sequence[str], matcher) -> List[Entry]:
        """Collect candidates by column, stopping at the display limit."""
        out: List[Entry] = []
        for column in columns:
            out.extend(matcher(pools.get(column, [])))
            if limit is not None and len(out) >= limit:
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
