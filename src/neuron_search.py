"""Cache-backed neuron-name resolution shared by analysis backends.

The UI suggestion/viewer and the analysis tools all need the same identity
search boundary.  This module reads the compact ``neuron_index_search``
sidecar next to a cached ``neuron_index.parquet`` and returns body IDs without
scanning the wide CSV metadata table for every query.

The resolver is deliberately optional: ``None`` means that the cache is not
available or is not usable, while ``([], info)`` means that a usable cache
searched successfully and found no rows.  Callers can therefore retain their
legacy dataframe matcher as a correctness fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from .neuron_index_builder import (
        build_search_cache_frame,
        body_id_column,
        dataset_folder,
        is_search_cache_compatible,
        priority_metadata_columns,
        search_cache_path,
        viewer_search_columns,
    )
except ImportError:  # pragma: no cover - supports src/ on sys.path imports
    from neuron_index_builder import (  # type: ignore
        build_search_cache_frame,
        body_id_column,
        dataset_folder,
        is_search_cache_compatible,
        priority_metadata_columns,
        search_cache_path,
        viewer_search_columns,
    )


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SEARCH_CACHE: Dict[Tuple[Any, ...], "CachedNeuronSearch"] = {}


@dataclass(frozen=True)
class CachedNeuronSearch:
    """The narrow searchable projection and its source-row identity map."""

    dataset: str
    index_path: Path
    search_path: Optional[Path]
    search_frame: Any
    body_ids: Tuple[str, ...]

    @property
    def body_id_keys(self) -> frozenset[str]:
        return frozenset(_body_id_key(value) for value in self.body_ids)


@dataclass(frozen=True)
class SearchStage:
    """One ordered candidate stage in the shared interactive search plan."""

    mode: str  # ``prefix`` or ``substring``
    columns: Tuple[str, ...]


def normalize_search_text(value: Any) -> str:
    """Return the exact trimmed text used by every lookup surface."""
    return str(value or "").strip()


def is_numeric_search(value: Any) -> bool:
    """Whether *value* is a bodyId-shaped query.

    Decimal-looking values are accepted only when the suffix is all zeroes;
    this matches the normalization used for spreadsheet/parquet body IDs and
    prevents a numeric typo from falling through into type metadata.
    """
    if isinstance(value, Integral) and not isinstance(value, bool):
        return True
    text = normalize_search_text(value)
    return bool(re.fullmatch(r"\d+(?:\.0+)?", text))


def ordered_search_columns(columns: Iterable[str]) -> List[str]:
    """Return the canonical bodyId/type/instance/taxonomy search order."""
    return viewer_search_columns(columns)


def _scope_column(scope: str) -> str:
    value = str(scope or "auto").strip().casefold()
    return "bodyId" if value == "bodyid" else value


def search_plan(
    text: Any,
    columns: Iterable[str],
    search_columns: str = "auto",
    *,
    all_prefix_matches: bool = False,
) -> List[SearchStage]:
    """Build the shared prefix-first plan used by suggestions and the viewer.

    Inline suggestions stop at the first non-empty stage.  The full viewer
    asks for ``all_prefix_matches=True`` so it can show every strict-prefix
    result, followed by substring-only results, while preserving the same
    canonical column order.
    """
    query = normalize_search_text(text)
    if not query:
        return []

    available = ordered_search_columns(columns)
    scope = str(search_columns or "auto").strip().casefold()
    if scope not in {"auto", "type", "instance", "bodyid"}:
        scope = "auto"

    if scope != "auto":
        column = _scope_column(scope)
        if column not in available:
            return []
        return [
            SearchStage("prefix", (column,)),
            SearchStage("substring", (column,)),
        ]

    if is_numeric_search(query):
        return (
            [
                SearchStage("prefix", ("bodyId",)),
                SearchStage("substring", ("bodyId",)),
            ]
            if "bodyId" in available else []
        )

    if all_prefix_matches:
        return (
            [
                SearchStage("prefix", tuple(available)),
                SearchStage("substring", tuple(available)),
            ]
            if available else []
        )

    stages: List[SearchStage] = []
    if "type" in available:
        stages.append(SearchStage("prefix", ("type",)))
    expanded = tuple(column for column in available if column != "type")
    if expanded:
        stages.append(SearchStage("prefix", expanded))
    if available:
        stages.append(SearchStage("substring", tuple(available)))
    return stages


def filter_candidate_entries(
    text: Any,
    candidates: Sequence[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """Narrow a cached candidate pool by a case-sensitive prefix."""
    query = normalize_search_text(text)
    if not query:
        return []
    return [entry for entry in candidates if str(entry[0]).startswith(query)]


def match_search_pools(
    text: Any,
    pools: Dict[str, List[Tuple[str, str]]],
    search_columns: str = "auto",
    limit: Optional[int] = 50,
    *,
    all_prefix_matches: bool = False,
) -> List[Tuple[str, str]]:
    """Apply :func:`search_plan` to the value pools used by UI inputs.

    This lives in the same module as the dataframe/cache resolver so the
    inline menu and available-neurons viewer cannot acquire different match
    rules by importing separate UI-only implementations.
    """
    query = normalize_search_text(text)
    if not query or not pools:
        return []

    def collect(columns: Sequence[str], predicate) -> List[Tuple[str, str]]:
        matches: List[Tuple[str, str]] = []
        for column in columns:
            matches.extend(
                entry for entry in pools.get(column, []) if predicate(entry[0])
            )
            if limit is not None and len(matches) >= limit:
                return matches[:limit]
        return matches

    prefix = lambda value: str(value).startswith(query)
    folded = query.casefold()
    substring = lambda value: folded in str(value).casefold()
    for stage in search_plan(
        query,
        pools.keys(),
        search_columns,
        all_prefix_matches=all_prefix_matches,
    ):
        matches = collect(
            stage.columns,
            prefix if stage.mode == "prefix" else substring,
        )
        if matches:
            return matches[:limit] if limit is not None else matches
    return []


def _body_id_key(value: Any) -> str:
    text = str(value or "").strip()
    # Strip ALL trailing zero decimals ("100.0", "100.00", ...), matching the
    # \d+(?:\.0+)? shape accepted by the numeric-search checks.
    match = re.fullmatch(r"(\d+)\.0+", text)
    if match:
        return match.group(1)
    return text


def _is_numeric_query(value: Any) -> bool:
    if isinstance(value, Integral) and not isinstance(value, bool):
        return True
    text = str(value or "").strip()
    return bool(re.fullmatch(r"\d+(?:\.0+)?", text))


def _normalized_query(value: Any) -> str:
    text = str(value or "").strip()
    if _is_numeric_query(value) and "." in text:
        return _body_id_key(text)
    return text


def _prefix_literal(pattern: str) -> Optional[str]:
    """Return the literal prefix represented by ``name.*``/``name*``."""
    value = str(pattern or "")
    if value.endswith(".*"):
        candidate = value[:-2]
    elif value.endswith("*") and value.count("*") == 1:
        candidate = value[:-1]
    else:
        return None
    if not candidate or re.search(r"[\\[\]().+?^$|{}]", candidate):
        return None
    return candidate


def _numeric_pattern(pattern: str) -> bool:
    """Whether a wildcard query is an identity-only bodyId pattern."""
    value = str(pattern or "").strip()
    if not value:
        return False
    stripped = (
        value.replace(".*", "")
        .replace("*", "")
        .replace("^", "")
        .replace("$", "")
        .replace(".", "")
    )
    return bool(stripped) and stripped.isdigit()


def _frame_column_values(frame: Any, column: str) -> List[Any]:
    """Read one dataframe column without imposing pandas on the UI path."""
    series = frame[column]
    if hasattr(series, "to_list"):
        return series.to_list()
    if hasattr(series, "tolist"):
        return series.tolist()
    return list(series)


def _display_value(value: Any, *, body_id: bool = False) -> str:
    """Normalize one source value exactly as the parquet sidecar does."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.casefold() in {
        "nan", "none", "null", "<na>", "<null>",
    }:
        return ""
    if body_id:
        return _body_id_key(text)
    return text


def _dataframe_search_columns(frame: Any) -> List[Tuple[str, str]]:
    """Return ``(canonical_name, source_name)`` pairs in shared priority."""
    names = [str(column) for column in getattr(frame, "columns", [])]
    actual_body = body_id_column(names)
    pairs: List[Tuple[str, str]] = []
    if actual_body is not None:
        pairs.append(("bodyId", actual_body))
    for column in ("type", "instance"):
        if column in names:
            pairs.append((column, column))
    for column in priority_metadata_columns(names):
        if column not in {source for _, source in pairs}:
            pairs.append((column, column))
    return pairs


def structured_search_columns(frame: Any) -> List[str]:
    """Return columns used by legacy/operator neuron filters.

    The interactive resolver intentionally searches only the compact
    identity/type/taxonomy projection.  ``NeuronFilter`` also supports
    explicit operators such as ``contains`` and ``regex`` over arbitrary
    string metadata, so its compatibility boundary keeps those columns after
    the canonical fields.  The order is shared across pandas callers and
    recognizes FlyWire body-ID aliases without changing the source schema.
    """
    names = [str(column) for column in getattr(frame, "columns", [])]
    actual_body = body_id_column(names)
    ordered: List[str] = []

    def add(column: Optional[str]) -> None:
        if column and column in names and column not in ordered:
            ordered.append(column)

    add(actual_body)
    add("type")
    add("instance")
    for column in priority_metadata_columns(names):
        if column in {actual_body, "type", "instance"}:
            continue
        try:
            dtype = getattr(frame[column], "dtype", None)
            import pandas as pd
            if not (
                pd.api.types.is_object_dtype(dtype)
                or pd.api.types.is_string_dtype(dtype)
            ):
                continue
        except Exception:
            continue
        add(column)

    # Preserve the old operator-filter contract for ordinary string fields,
    # but do not turn numeric measurement columns into accidental text search
    # targets.  Body IDs are the one deliberate numeric exception above.
    try:
        import pandas as pd

        for column in names:
            if column in ordered:
                continue
            dtype = getattr(frame[column], "dtype", None)
            if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
                add(column)
    except Exception:
        # Non-pandas callers still receive the canonical fields.  The actual
        # dataframe resolver remains the fallback for those callers.
        pass
    return ordered


def _structured_operator_column(series: Any, operator: str, patterns: Sequence[Any]):
    """Apply one legacy NeuronFilter operator to one pandas Series."""
    import pandas as pd

    if not patterns:
        return pd.Series(False, index=series.index, dtype=bool)
    notna = series.notna()
    string_series = series.astype(str)
    string_patterns = [str(pattern) for pattern in patterns]

    if operator == "exact":
        return (series.isin(list(patterns)) | string_series.isin(string_patterns)) & notna
    if operator == "contains":
        pattern = "|".join(re.escape(value) for value in string_patterns)
        return string_series.str.contains(pattern, na=False) & notna
    if operator == "not_contains":
        pattern = "|".join(re.escape(value) for value in string_patterns)
        return notna & ~string_series.str.contains(pattern, na=False)
    if operator == "startswith":
        return string_series.str.startswith(tuple(string_patterns), na=False) & notna
    if operator == "endswith":
        return string_series.str.endswith(tuple(string_patterns), na=False) & notna
    if operator == "regex":
        result = pd.Series(False, index=series.index, dtype=bool)
        for pattern in string_patterns:
            try:
                result = result | string_series.str.contains(
                    pattern, regex=True, na=False
                )
            except re.error:
                # Preserve the public NeuronFilter compatibility behavior:
                # an invalid regex becomes an exact literal for that value.
                result = result | (string_series == pattern)
        return result & notna
    if operator == "not_regex":
        result = pd.Series(True, index=series.index, dtype=bool)
        for pattern in string_patterns:
            try:
                result = result & ~string_series.str.contains(
                    pattern, regex=True, na=False
                )
            except re.error:
                result = result & (string_series != pattern)
        return result & notna
    return pd.Series(False, index=series.index, dtype=bool)


def apply_structured_filter(
    frame: Any,
    filter_spec: Optional[Mapping[str, Sequence[Any]]],
    *,
    match_all: bool = False,
) -> Any:
    """Apply a parsed operator filter through the shared neuron backend.

    This is the execution half of :class:`src.utils.neuron_filter.NeuronFilter`.
    It deliberately keeps the historical semantics: OR across searchable
    columns for each operator, AND across operators, exact integer patterns
    restricted to the real body-ID column, and invalid regular expressions
    falling back to literal equality.  Keeping that behavior here lets
    pathfinding, visualization, similarity, and notebook helpers share one
    implementation without changing their established filter language.
    """
    import pandas as pd

    if frame is None:
        # No frame to filter; None stays None (copy() would crash).
        return frame
    if match_all or len(frame) == 0 or not filter_spec:
        return frame.copy()

    columns = structured_search_columns(frame)
    if not columns:
        return frame.copy()

    actual_body = body_id_column([str(column) for column in frame.columns])
    mask = pd.Series(True, index=frame.index, dtype=bool)
    for operator, patterns in filter_spec.items():
        if isinstance(patterns, (str, bytes)):
            values = [patterns]
        else:
            values = list(patterns or [])
        if not values:
            mask &= False
            continue
        operator_mask = pd.Series(False, index=frame.index, dtype=bool)
        integer_exact = (
            operator == "exact"
            and isinstance(values[0], Integral)
            and not isinstance(values[0], bool)
        )
        for column in columns:
            if integer_exact and column != actual_body:
                continue
            operator_mask |= _structured_operator_column(
                frame[column], operator, values
            )
        mask &= operator_mask
    return frame[mask].copy()


def normalize_search_operator(value: Any) -> str:
    """Normalize the operator names shared by the UI and core predicates."""
    aliases = {
        "starts_with": "prefix",
        "starts with": "prefix",
        "startswith": "prefix",
        "prefix": "prefix",
        "ends_with": "suffix",
        "ends with": "suffix",
        "endswith": "suffix",
        "suffix": "suffix",
        "exact": "exact",
        "equals": "exact",
        "contains": "substring",
        "substring": "substring",
        "regex": "regex",
    }
    return aliases.get(str(value or "").strip().casefold(), "substring")


def polars_display_expression(column: str):
    """Return the canonical string expression used by UI search predicates."""
    import polars as pl

    expression = pl.col(column).cast(pl.Utf8, strict=False).fill_null("")
    if column == "bodyId":
        expression = expression.str.strip_chars().str.replace(r"\.0+$", "")
    return expression


def polars_match_column_expression(frame: Any, column: str, text: Any, mode: str):
    """Build one case-sensitive/insensitive Polars match expression."""
    import polars as pl

    needle = normalize_search_text(text)
    display_value = polars_display_expression(column)
    mode = normalize_search_operator(mode)
    if mode == "prefix":
        return display_value.str.starts_with(needle)
    if mode == "suffix":
        return display_value.str.ends_with(needle)
    if mode == "exact":
        return display_value == needle
    if mode == "regex":
        try:
            re.compile(needle)
        except re.error:
            return pl.lit(False)
        return display_value.str.contains(needle, literal=False)
    return display_value.str.to_lowercase().str.contains(
        needle.casefold(), literal=True
    )


def polars_body_id_guard(frame: Any, columns: Iterable[str], text: Any):
    """Guard numeric UI queries so they can only match integer-like body IDs."""
    import polars as pl

    columns = list(columns)
    if "bodyId" not in columns or not is_numeric_search(text):
        return pl.lit(True)
    return polars_display_expression("bodyId").str.contains(r"^\d+$")


def polars_match_expression(
    frame: Any,
    columns: Iterable[str],
    text: Any,
    mode: str,
):
    """Build a shared OR-across-columns Polars predicate."""
    import polars as pl

    available = [column for column in columns if column in frame.columns]
    expressions = [
        polars_match_column_expression(frame, column, text, mode)
        for column in available
    ]
    if not expressions:
        return pl.lit(False)
    return pl.any_horizontal(expressions) & polars_body_id_guard(
        frame, available, text
    )


def _legacy_regex_pattern(pattern: str) -> str:
    """Translate the explicit filter syntax used by the UI into regex."""
    regex = pattern.replace("*", ".*") \
        if "*" in pattern and ".*" not in pattern else pattern
    if not regex.startswith(".*") and not regex.startswith("^"):
        regex = "^" + regex
    return regex


def resolve_dataframe_query(
    frame: Any,
    query: Any,
    *,
    search_columns: str = "auto",
    verbose: bool = False,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Resolve a query against an in-memory metadata frame.

    This is the correctness fallback for the parquet reader and the shared
    adapter for callers that receive a dataframe (NeuronBridge, morphology,
    notebooks). It intentionally implements the same strict contract as
    :func:`resolve_neuron_query`: bare strings are exact/case-sensitive;
    ``name.*``/``name*`` are case-sensitive prefixes; other wildcard forms
    are explicit regex; automatic numeric queries search bodyId only; and the
    first canonical column with a hit owns the query.
    """
    pairs = _dataframe_search_columns(frame)
    info: Dict[str, Any] = {
        "search_term": str(query),
        "matched_column": None,
        "match_count": 0,
        "cache": False,
    }
    query_text = _normalized_query(query)
    if not query_text or not pairs:
        return [], info

    scope = str(search_columns or "auto").strip().casefold()
    if scope not in {"auto", "type", "instance", "bodyid"}:
        scope = "auto"
    actual_by_canonical = dict(pairs)
    available = [canonical for canonical, _ in pairs]
    if scope == "bodyid":
        columns = ["bodyId"] if "bodyId" in actual_by_canonical else []
    elif scope in {"type", "instance"}:
        columns = [scope] if scope in actual_by_canonical else []
    else:
        columns = list(available)

    values_by_column = {
        canonical: [
            _display_value(value, body_id=canonical == "bodyId")
            for value in _frame_column_values(frame, actual)
        ]
        for canonical, actual in pairs
    }
    def find_rows(column: str, predicate) -> List[int]:
        return [
            index for index, value in enumerate(values_by_column.get(column, []))
            if value and predicate(value)
        ]

    def finish(rows: List[int], column: Optional[str], mode: str):
        body_column = actual_by_canonical.get("bodyId")
        raw_body_ids = (
            _frame_column_values(frame, body_column)
            if body_column is not None else []
        )
        output: List[Any] = []
        seen = set()
        for index in rows:
            if index >= len(raw_body_ids):
                continue
            value = raw_body_ids[index]
            key = _body_id_key(value)
            if not key or key in seen:
                continue
            seen.add(key)
            output.append(value)
        info["matched_column"] = column if output else None
        info["match_count"] = len(output)
        if output:
            info["match_mode"] = mode
        return output, info

    numeric = _is_numeric_query(query)
    wildcard = isinstance(query, str) and (".*" in query or "*" in query)
    prefix = _prefix_literal(query_text) if wildcard else None
    numeric_wildcard = wildcard and _numeric_pattern(query_text)

    if numeric and scope in {"auto", "bodyid"}:
        columns = ["bodyId"] if "bodyId" in actual_by_canonical else []
    elif numeric_wildcard and scope == "auto":
        columns = ["bodyId"] if "bodyId" in actual_by_canonical else []

    if prefix is not None:
        for column in columns:
            rows = find_rows(column, lambda value: value.startswith(prefix))
            if rows:
                return finish(rows, column, "prefix")
        return finish([], None, "prefix")

    if not wildcard and not numeric:
        exact_columns = (
            [column for column in columns if column != "bodyId"]
            if scope == "auto" else columns
        )
        for column in exact_columns:
            rows = find_rows(column, lambda value: value == query_text)
            if rows:
                return finish(rows, column, "exact")
        return finish([], None, "exact")

    if numeric and not wildcard:
        if scope not in {"auto", "bodyid"}:
            for column in columns:
                rows = find_rows(column, lambda value: value == query_text)
                if rows:
                    return finish(rows, column, "exact")
            return finish([], None, "exact")
        rows = find_rows(
            "bodyId",
            lambda value: bool(re.fullmatch(r"\d+", value))
            and value == query_text,
        )
        return finish(rows, "bodyId", "exact")

    try:
        compiled = re.compile(_legacy_regex_pattern(query_text))
    except re.error:
        return finish([], None, "regex")
    for column in columns:
        rows = find_rows(column, compiled.match)
        if rows:
            return finish(rows, column, "regex")
    return finish([], None, "regex")


def _scope_columns(search_frame, search_columns: str) -> List[str]:
    available = set(search_frame["search_column"].unique().to_list())
    scope = str(search_columns or "auto").strip().casefold()
    if scope == "bodyid":
        wanted = ["bodyId"]
    elif scope in {"type", "instance"}:
        wanted = [scope]
    else:
        wanted = viewer_search_columns(
            search_frame["search_column"].unique().to_list()
        )
    return [column for column in wanted if column in available]


def _cache_covers_frame(cache: CachedNeuronSearch, frame: Any) -> bool:
    """Return whether a cache can answer every canonical query for *frame*.

    Body-ID equality alone is not enough to validate a cache.  Older cache
    indexes can contain the same neurons while predating a newly pulled
    ``*Type`` or taxonomy column.  In that situation using the cache would
    return an incorrect empty result instead of consulting the authoritative
    metadata table.  Compare the canonical searchable projection before
    allowing the fast path; non-searchable metadata columns intentionally do
    not affect this check.
    """
    try:
        expected = [
            canonical for canonical, _ in _dataframe_search_columns(frame)
        ]
        cached = ordered_search_columns(
            cache.search_frame["search_column"].unique().to_list()
        )
    except Exception:
        return False
    return cached == expected


def _load_signature(path: Path) -> Tuple[Any, ...]:
    try:
        stat = path.stat()
    except OSError:
        return (str(path), None)
    return (str(path), stat.st_mtime_ns, stat.st_size)


def clear_cached_neuron_search() -> None:
    """Clear process-local sidecar readers (useful after a cache rebuild)."""
    _SEARCH_CACHE.clear()


def get_cached_neuron_search(
    dataset: str,
    *,
    index_root: Optional[Path] = None,
) -> Optional[CachedNeuronSearch]:
    """Load the cached search projection for *dataset*, if it exists.

    The index lives in the app-owned ``neuron_indexes/`` directory (a
    persistent "system files" location that survives ``cache/`` cleanups).  A
    missing sidecar is built in memory from the index.  It is not written
    here: index construction remains an explicit pipeline operation, so an
    analysis lookup never creates an unexpected file.
    """
    import polars as pl

    root = Path(index_root) if index_root is not None else _REPO_ROOT / "neuron_indexes"
    folder = root / dataset_folder(dataset)
    index_path = folder / "neuron_index.parquet"
    if not index_path.is_file():
        return None
    sidecar_path = search_cache_path(index_path)
    signature = (
        _load_signature(index_path),
        _load_signature(sidecar_path) if sidecar_path.is_file() else None,
    )
    key = signature
    cached = _SEARCH_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        body_frame = pl.read_parquet(index_path, columns=["bodyId"])
        body_ids = tuple(
            _body_id_key(value)
            for value in body_frame.get_column("bodyId").to_list()
        )
        if sidecar_path.is_file():
            search_frame = pl.read_parquet(sidecar_path)
            source_columns = pl.scan_parquet(index_path).collect_schema().names()
            if is_search_cache_compatible(search_frame, source_columns):
                search_source = sidecar_path
            else:
                # A readable but stale sidecar must not hide a newly added
                # flywireType/taxonomy column. Rebuild only the narrow search
                # frame in memory; the pull/cache pipeline will materialize it
                # on its next normal cache-maintenance pass.
                full_frame = pl.read_parquet(index_path)
                search_frame = build_search_cache_frame(full_frame)
                search_source = None
        else:
            full_frame = pl.read_parquet(index_path)
            search_frame = build_search_cache_frame(full_frame)
            search_source = None
        required = {
            "__neuron_rows",
            "search_column",
            "search_priority",
            "search_value",
        }
        if not required.issubset(set(search_frame.columns)):
            return None
    except Exception:
        return None

    result = CachedNeuronSearch(
        dataset=str(dataset),
        index_path=index_path,
        search_path=search_source,
        search_frame=search_frame,
        body_ids=body_ids,
    )
    # Discard stale signatures for this index path while retaining other
    # datasets' readers.
    for old_key in list(_SEARCH_CACHE):
        if old_key != key and old_key[0][0] == str(index_path):
            _SEARCH_CACHE.pop(old_key, None)
    _SEARCH_CACHE[key] = result
    return result


def _entries_for_values(search_frame, columns: Iterable[str], predicate):
    import polars as pl

    columns = list(columns)
    if not columns:
        return search_frame.head(0)
    values = search_frame.filter(
        pl.col("search_column").is_in(columns)
    ).filter(predicate)
    if values.is_empty():
        return values.head(0)
    return (
        values
        .explode("__neuron_rows")
        .rename({"__neuron_rows": "__neuron_row"})
        .select(
            "__neuron_row",
            "search_column",
            "search_priority",
            "search_value",
        )
    )


def _body_ids_for_entries(cache: CachedNeuronSearch, entries) -> List[str]:
    if entries is None or entries.is_empty():
        return []
    result: List[str] = []
    seen = set()
    for row in entries.get_column("__neuron_row").to_list():
        try:
            position = int(row)
            value = cache.body_ids[position]
        except (IndexError, TypeError, ValueError):
            continue
        key = _body_id_key(value)
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result


def _empty_info(query: Any) -> Dict[str, Any]:
    return {
        "search_term": str(query),
        "matched_column": None,
        "match_count": 0,
        "cache": True,
    }


def _result(
    cache: CachedNeuronSearch,
    query: Any,
    entries,
    matched_column: Optional[str],
) -> Tuple[List[str], Dict[str, Any]]:
    body_ids = _body_ids_for_entries(cache, entries)
    info = _empty_info(query)
    info["matched_column"] = matched_column if body_ids else None
    info["match_count"] = len(body_ids)
    return body_ids, info


def resolve_neuron_query(
    cache: Optional[CachedNeuronSearch],
    query: Any,
    *,
    search_columns: str = "auto",
) -> Optional[Tuple[List[str], Dict[str, Any]]]:
    """Resolve one legacy neuron query through the cached search sidecar.

    Bare values are strict/case-sensitive. Explicit ``name.*``/``name*``
    values are strict case-sensitive prefixes and stop at the first canonical
    field with a match in automatic scope. Other wildcard forms retain the
    legacy regex behavior and use the first matching canonical field. Numeric
    automatic queries are restricted to the real bodyId field.
    """
    import polars as pl

    if cache is None:
        return None
    query_text = _normalized_query(query)
    info = _empty_info(query)
    if not query_text:
        return ([], info)

    scope = str(search_columns or "auto").strip().casefold()
    if scope not in {"auto", "type", "instance", "bodyid"}:
        scope = "auto"
    all_columns = _scope_columns(cache.search_frame, scope)
    if not all_columns:
        return ([], info)

    numeric = _is_numeric_query(query)
    wildcard = isinstance(query, str) and (".*" in query or "*" in query)
    prefix = _prefix_literal(query_text) if wildcard else None
    numeric_wildcard = wildcard and _numeric_pattern(query_text)

    if numeric and scope in {"auto", "bodyid"}:
        columns = ["bodyId"] if "bodyId" in all_columns else []
    elif numeric_wildcard and scope == "auto":
        columns = ["bodyId"] if "bodyId" in all_columns else []
    elif scope == "auto":
        columns = list(all_columns)
    else:
        columns = list(all_columns)

    # Explicit prefix mode follows the analysis priority contract. The viewer
    # deliberately keeps later-column hits as secondary explanations, but a
    # pathfinding/skeleton query must not silently expand a type name into a
    # different taxonomy column. Stop at the first canonical column with any
    # match, including for ``name.*``.
    if prefix is not None:
        for column in columns:
            entries = _entries_for_values(
                cache.search_frame,
                [column],
                pl.col("search_value").str.starts_with(prefix),
            )
            if not entries.is_empty():
                return _result(cache, query, entries, column)
        return _result(cache, query, cache.search_frame.head(0), None)

    if not wildcard and not numeric:
        # A bare automatic string follows type -> instance -> taxonomy. A
        # named scope may explicitly search bodyId as well.
        exact_columns = (
            [column for column in columns if column != "bodyId"]
            if scope == "auto"
            else columns
        )
        for column in exact_columns:
            entries = _entries_for_values(
                cache.search_frame,
                [column],
                pl.col("search_value") == query_text,
            )
            if not entries.is_empty():
                return _result(cache, query, entries, column)
        return _result(cache, query, cache.search_frame.head(0), None)

    if numeric and not wildcard:
        if scope not in {"auto", "bodyid"}:
            for column in columns:
                entries = _entries_for_values(
                    cache.search_frame,
                    [column],
                    pl.col("search_value") == query_text,
                )
                if not entries.is_empty():
                    return _result(cache, query, entries, column)
            return _result(cache, query, cache.search_frame.head(0), None)
        entries = _entries_for_values(
            cache.search_frame,
            ["bodyId"],
            (pl.col("search_value") == query_text)
            & pl.col("search_value").str.contains(r"^\d+$"),
        )
        return _result(cache, query, entries, "bodyId")

    # Remaining wildcard forms are true regex-style searches.  Match the
    # first canonical field with hits, mirroring _process_single_neuron's
    # priority path. Numeric wildcard forms stay bodyId-only.
    regex = query_text
    if not regex.startswith(".*") and not regex.startswith("^"):
        regex = "^" + regex
    try:
        re.compile(regex)
    except re.error:
        return ([], info)
    for column in columns:
        entries = _entries_for_values(
            cache.search_frame,
            [column],
            pl.col("search_value").str.contains(regex, literal=False),
        )
        if not entries.is_empty():
            return _result(cache, query, entries, column)
    return _result(cache, query, cache.search_frame.head(0), None)


def resolve_cached_or_dataframe_query(
    cache: Optional[CachedNeuronSearch],
    frame: Any,
    query: Any,
    *,
    search_columns: str = "auto",
) -> Tuple[List[Any], Dict[str, Any]]:
    """Resolve through the compact cache, with a validated frame fallback.

    The cache is an optimization, never an independent source of truth.  It
    is used only when the frame and cache contain the same normalized body-ID
    set; otherwise the exact same resolver runs against the caller's frame.
    This boundary is shared by pathfinding, morphology, skeleton rendering,
    and NeuronBridge so a stale or partial cache cannot change semantics.
    """
    actual_body = body_id_column(
        [str(column) for column in getattr(frame, "columns", [])]
    )
    if cache is not None and actual_body is not None:
        source_values = _frame_column_values(frame, actual_body)
        source_keys = {
            _body_id_key(value) for value in source_values
            if _body_id_key(value)
        }
        if source_keys == set(cache.body_id_keys) and _cache_covers_frame(
            cache, frame
        ):
            cached_result = resolve_neuron_query(
                cache,
                query,
                search_columns=search_columns,
            )
            if cached_result is not None:
                cached_ids, info = cached_result
                source_by_key = {
                    _body_id_key(value): value for value in source_values
                }
                resolved = [
                    source_by_key[_body_id_key(value)]
                    for value in cached_ids
                    if _body_id_key(value) in source_by_key
                ]
                if not cached_ids or resolved:
                    info["match_count"] = len(resolved)
                    return resolved, info
    return resolve_dataframe_query(
        frame,
        query,
        search_columns=search_columns,
    )


__all__ = [
    "CachedNeuronSearch",
    "SearchStage",
    "apply_structured_filter",
    "clear_cached_neuron_search",
    "filter_candidate_entries",
    "get_cached_neuron_search",
    "is_numeric_search",
    "match_search_pools",
    "normalize_search_text",
    "normalize_search_operator",
    "ordered_search_columns",
    "polars_body_id_guard",
    "polars_display_expression",
    "polars_match_column_expression",
    "polars_match_expression",
    "resolve_dataframe_query",
    "resolve_cached_or_dataframe_query",
    "resolve_neuron_query",
    "search_plan",
    "structured_search_columns",
]
