"""Cached neuron-index loading and querying for the UI viewer.

The auto-suggestion backend and the available-neurons viewer intentionally
share the same local cache boundary: a viewer is available only when
``cache/<dataset>/neuron_index.parquet`` exists.  The viewer never serves the
raw dataset file to the browser.  The cache index is built from the
materialized projection of the prepared local neuron table; an older/partial
cache can still be enriched from that table to fill blank ``type``/``instance``
values.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import PROJECT_ROOT
from .dataset_service import dataset_to_folder

try:
    from src.neuron_index_builder import (
        metadata_columns,
        ordered_projection_columns,
        priority_metadata_columns,
        read_metadata_projection,
    )
except ImportError:
    from neuron_index_builder import (
        metadata_columns,
        ordered_projection_columns,
        priority_metadata_columns,
        read_metadata_projection,
    )


@dataclass(frozen=True)
class CachedNeuronIndex:
    """A cached neuron index plus the path used to load it."""

    dataset: str
    path: Path
    frame: Any  # polars.DataFrame; kept Any so importing the UI does not require Polars eagerly
    columns: Tuple[str, ...]
    enriched: bool = False


@dataclass(frozen=True)
class NeuronIndexPage:
    """One server-side page of a filtered/sorted neuron index."""

    rows: List[Dict[str, Any]]
    total: int
    page: int
    pages: int
    page_size: int
    sort_by: str
    descending: bool


# The largest local indexes are only a few megabytes as Parquet but are read
# by several UI clients.  Keep one process-local copy and invalidate it when
# either the cache index or its optional metadata table changes.
_INDEX_CACHE: Dict[Tuple, CachedNeuronIndex] = {}

# Search-result priority intentionally differs from the suggestion input's
# string-first expansion: the viewer has a stable, visible result order.
_MATCH_PRIORITY_COLUMNS = ("bodyId", "type", "instance")


def clear_neuron_index_cache() -> None:
    """Clear the process-local viewer cache (primarily useful for tests)."""
    _INDEX_CACHE.clear()


def neuron_index_path(dataset: str, cache_dir: Optional[Path] = None) -> Path:
    """Return the cached neuron-index path for *dataset*."""
    root = Path(cache_dir) if cache_dir is not None else PROJECT_ROOT / "cache"
    return root / dataset_to_folder(str(dataset).strip()) / "neuron_index.parquet"


def neuron_index_state_path(dataset: str, cache_dir: Optional[Path] = None) -> Path:
    """Return the optional progress sidecar for a cached index."""
    root = Path(cache_dir) if cache_dir is not None else PROJECT_ROOT / "cache"
    return root / dataset_to_folder(str(dataset).strip()) / "neuron_index_state.parquet"


def _metadata_candidates(dataset: str, datasets_dir: Optional[Path]) -> List[Path]:
    """Find generated metadata files, preferring the pulled CSV source."""
    if datasets_dir is None:
        datasets_dir = PROJECT_ROOT / "datasets"
    folder = Path(datasets_dir) / dataset_to_folder(str(dataset).strip())
    if not folder.is_dir():
        return []

    safe = folder.name
    exact = [
        folder / f"{safe}_allneurons_neuron_df.csv",
        folder / f"{safe}_neuron_df.csv",
        folder / f"{safe}_allneurons_neuron_df.parquet",
        folder / f"{safe}_neuron_df.parquet",
    ]
    discovered = sorted(
        [
            p
            for pattern in ("*_allneurons_neuron_df.csv", "*_neuron_df.csv",
                            "*_allneurons_neuron_df.parquet", "*_neuron_df.parquet")
            for p in folder.glob(pattern)
        ],
        key=lambda p: p.name,
    )
    result: List[Path] = []
    for path in exact + discovered:
        if path.is_file() and path not in result:
            result.append(path)
    return result


def _metadata_signature(dataset: str, datasets_dir: Optional[Path]) -> Optional[Tuple[str, int]]:
    candidates = _metadata_candidates(dataset, datasets_dir)
    if not candidates:
        return None
    path = candidates[0]
    try:
        return str(path), path.stat().st_mtime_ns
    except OSError:
        return None


def _is_blank(expression):
    """Return a Polars expression identifying null/empty display values."""
    import polars as pl

    return (
        expression.is_null()
        | (
            expression.cast(pl.Utf8, strict=False)
            .fill_null("")
            .str.strip_chars()
            == ""
        )
    )


def _read_metadata_table(path: Path):
    """Read the materialized projection from a generated local neuron table."""
    import polars as pl

    try:
        frame = read_metadata_projection(path)
    except Exception:
        return None
    if "bodyId" not in frame.columns:
        return None
    frame = frame.with_columns(
        pl.col("bodyId").cast(pl.Utf8, strict=False).fill_null("").alias("bodyId")
    )
    rename = {
        column: f"__metadata_{column}"
        for column in frame.columns
        if column != "bodyId"
    }
    return frame.rename(rename).unique(subset=["bodyId"], keep="first")


def _enrich_identifiers(frame, dataset: str, datasets_dir: Optional[Path]):
    """Fill blank type/instance values from local generated metadata."""
    import polars as pl

    if "bodyId" not in frame.columns:
        frame = frame.with_columns(pl.lit("").alias("bodyId"))

    display_columns = ["bodyId", "type", "instance"]
    expressions = []
    for column in display_columns:
        if column in frame.columns:
            expressions.append(
                pl.col(column).cast(pl.Utf8, strict=False).fill_null("").alias(column)
            )
        else:
            expressions.append(pl.lit("").alias(column))
    frame = frame.with_columns(expressions)

    metadata_signature = _metadata_signature(dataset, datasets_dir)
    if metadata_signature is None:
        return frame, False

    metadata_file = Path(metadata_signature[0])
    # A freshly generated rich index already contains the same projection.
    # Avoid opening the hundreds-of-MiB CSV again just to fill values it has.
    try:
        import polars as pl
        source_columns = set(metadata_columns(metadata_file))
        if (
            source_columns.issubset(set(frame.columns))
            and {"downstream_complete", "last_fetched", "connection_count"}
            .issubset(set(frame.columns))
        ):
            return frame, False
    except Exception:
        pass

    metadata = _read_metadata_table(metadata_file)
    if metadata is None:
        return frame, False

    frame = frame.join(metadata, on="bodyId", how="left")
    for metadata_column in metadata.columns:
        if not metadata_column.startswith("__metadata_"):
            continue
        column = metadata_column.removeprefix("__metadata_")
        if column not in frame.columns:
            frame = frame.with_columns(
                pl.col(metadata_column).alias(column)
            ).drop(metadata_column)
            continue
        metadata_column = f"__metadata_{column}"
        frame = frame.with_columns(
            pl.when(_is_blank(pl.col(column)))
            .then(pl.col(metadata_column).cast(pl.Utf8, strict=False).fill_null(""))
            .otherwise(pl.col(column))
            .alias(column)
        ).drop(metadata_column)
    return frame, True


def load_cached_neuron_index(
    dataset: str,
    *,
    cache_dir: Optional[Path] = None,
    datasets_dir: Optional[Path] = None,
    enrich: bool = True,
) -> CachedNeuronIndex:
    """Load a cached neuron index for display.

    Raises:
        FileNotFoundError: when the cached ``neuron_index.parquet`` is absent.
        ValueError: when the file cannot be read as a usable table.
    """
    import polars as pl

    dataset = str(dataset or "").strip()
    if not dataset:
        raise FileNotFoundError("No dataset was selected")

    path = neuron_index_path(dataset, cache_dir)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    metadata_signature = _metadata_signature(dataset, datasets_dir) if enrich else None
    state_path = neuron_index_state_path(dataset, cache_dir)
    try:
        state_signature = (
            (str(state_path), state_path.stat().st_mtime_ns)
            if state_path.is_file()
            else (str(state_path), None)
        )
        signature = (
            str(path),
            path.stat().st_mtime_ns,
            metadata_signature,
            state_signature,
            bool(enrich),
        )
    except OSError as exc:
        raise FileNotFoundError(str(path)) from exc
    if signature in _INDEX_CACHE:
        return _INDEX_CACHE[signature]

    try:
        frame = pl.read_parquet(path)
    except Exception as exc:
        raise ValueError(f"Could not read cached neuron index: {exc}") from exc

    # Overlay progress written by an in-flight connection pull.  The sidecar
    # contains only cache flags/counts and never changes the metadata columns.
    if state_path.is_file():
        try:
            state = pl.read_parquet(state_path)
            if "bodyId" in state.columns and "bodyId" in frame.columns:
                frame = frame.with_columns(
                    pl.col("bodyId").cast(pl.Utf8, strict=False).fill_null("").alias("bodyId")
                )
                state = state.with_columns(
                    pl.col("bodyId").cast(pl.Utf8, strict=False).fill_null("").alias("bodyId")
                )
                frame = frame.join(state, on="bodyId", how="left", suffix="__state")
                for column in ("downstream_complete", "last_fetched", "connection_count"):
                    state_column = f"{column}__state"
                    if state_column not in frame.columns:
                        continue
                    if column not in frame.columns:
                        frame = frame.with_columns(
                            pl.col(state_column).alias(column)
                        )
                    else:
                        frame = frame.with_columns(
                            pl.when(pl.col(state_column).is_not_null())
                            .then(pl.col(state_column))
                            .otherwise(pl.col(column))
                            .alias(column)
                        )
                    frame = frame.drop(state_column)
        except Exception:
            # A partially written sidecar must not make the viewer unusable;
            # the canonical index remains readable on its own.
            pass

    frame, enriched = _enrich_identifiers(frame, dataset, datasets_dir) if enrich else (frame, False)
    if not frame.columns:
        raise ValueError("The cached neuron index has no columns")

    # Legacy indexes may have been enriched from a source table after load;
    # apply the same order as newly generated caches before exposing columns.
    frame = frame.select(ordered_projection_columns(frame.columns))

    # Keep the schema stable and JSON-friendly for the UI.  Index values are
    # displayed as strings for bodyId/type/instance so large IDs are never
    # rounded by browser JavaScript.
    frame = frame.with_columns(
        pl.col("bodyId").cast(pl.Utf8, strict=False).fill_null("").alias("bodyId")
        if "bodyId" in frame.columns
        else pl.lit("").alias("bodyId")
    )
    for column in ("type", "instance"):
        if column in frame.columns:
            frame = frame.with_columns(
                pl.col(column).cast(pl.Utf8, strict=False).fill_null("").alias(column)
            )

    result = CachedNeuronIndex(
        dataset=dataset,
        path=path,
        frame=frame,
        columns=tuple(frame.columns),
        enriched=enriched,
    )
    _INDEX_CACHE[signature] = result
    # Remove old versions of this path so a rebuilt index does not accumulate
    # unbounded DataFrames in a long-running UI process.
    for old_key in list(_INDEX_CACHE):
        if old_key != signature and old_key[0] == str(path):
            _INDEX_CACHE.pop(old_key, None)
    return result


def _contains_expression(frame, columns: List[str], text: str):
    import polars as pl

    needle = str(text).strip().lower()
    expressions = [
        pl.col(column)
        .cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.to_lowercase()
        .str.contains(needle, literal=True)
        for column in columns
        if column in frame.columns
    ]
    if not expressions:
        return pl.lit(False)
    return pl.any_horizontal(expressions)


def _ordered_match_columns(columns: List[str]) -> List[str]:
    """Return only the viewer's identity/taxonomy search scope.

    The full cache remains available for display and explicit column filters,
    but global viewer search must not scan operational, measurement, notes, or
    other arbitrary metadata. Keeping this list small also avoids repeatedly
    decoding large non-search fields for every keystroke.
    """
    priority = priority_metadata_columns(columns)
    return [
        *[column for column in _MATCH_PRIORITY_COLUMNS if column in columns],
        *[column for column in priority if column not in _MATCH_PRIORITY_COLUMNS],
    ]


def _match_metadata(frame, columns: List[str], text: str, scope=None):
    """Build match-priority and display metadata expressions for a query.

    The first matching column in the ordered scope wins.  For bodyId matches,
    the displayed hint mirrors auto-suggestion: show the corresponding
    instance when one exists, otherwise show ``bodyId``.  The match key and
    value are kept separately so the viewer can highlight the actual source
    cell while showing the compact hint in its pinned info columns.
    """
    import polars as pl

    ordered = [
        column for column in (scope or _ordered_match_columns(columns))
        if column in frame.columns
    ]
    if not ordered:
        empty = pl.lit("")
        return pl.lit(0), empty, empty, empty

    needle = str(text).strip().lower()
    priority = pl.lit(len(ordered))
    match_column = pl.lit("")
    match_column_key = pl.lit("")
    match_value = pl.lit("")
    for rank, column in reversed(list(enumerate(ordered))):
        display_value = (
            pl.col(column)
            .cast(pl.Utf8, strict=False)
            .fill_null("")
        )
        matched = display_value.str.to_lowercase().str.contains(needle, literal=True)
        if column == "bodyId" and "instance" in frame.columns:
            instance = (
                pl.col("instance")
                .cast(pl.Utf8, strict=False)
                .fill_null("")
                .str.strip_chars()
            )
            hint = pl.when(instance != "").then(instance).otherwise(pl.lit("bodyId"))
        else:
            hint = pl.lit(column)
        priority = pl.when(matched).then(pl.lit(rank)).otherwise(priority)
        match_column = pl.when(matched).then(hint).otherwise(match_column)
        match_column_key = pl.when(matched).then(pl.lit(column)).otherwise(match_column_key)
        match_value = pl.when(matched).then(display_value).otherwise(match_value)
    return priority, match_column, match_column_key, match_value


def query_neuron_index(
    index: CachedNeuronIndex,
    *,
    search: str = "",
    filter_column: Optional[str] = None,
    filter_text: str = "",
    sort_by: Optional[str] = None,
    descending: bool = False,
    page: int = 1,
    page_size: int = 50,
) -> NeuronIndexPage:
    """Filter, sort, and page a cached index without sending all rows to JS.

    Search and column filters use case-insensitive substring matching. Global
    search uses only bodyId, type, instance, and useful type/class taxonomy
    fields, in that order. An explicit column filter may still target any
    retained metadata column. When a query is present and no explicit sort is
    supplied, rows are grouped by matched-column priority and sorted by
    matched value within each group. An explicit sort column becomes the
    primary sort and matching priority is used only as a tie-breaker.
    """
    import polars as pl

    page_size = max(1, min(int(page_size), 500))
    frame = index.frame
    columns = list(index.columns)
    filtered = frame

    search_text = str(search or "").strip()
    search_columns = _ordered_match_columns(columns)
    if search_text:
        filtered = filtered.filter(
            _contains_expression(filtered, search_columns, search_text)
        )

    filter_text = str(filter_text or "").strip()
    if filter_text:
        if not filter_column or filter_column in {"__all__", "All columns"}:
            filtered = filtered.filter(
                _contains_expression(filtered, search_columns, filter_text)
            )
        elif filter_column in filtered.columns:
            expression = (
                pl.col(filter_column)
                .cast(pl.Utf8, strict=False)
                .fill_null("")
                .str.to_lowercase()
                .str.contains(filter_text.lower(), literal=True)
            )
            filtered = filtered.filter(expression)

    # The info column describes the global query when present.  If only a
    # column filter is active, describe that filter instead; this keeps the
    # table useful even when the top search box is empty.
    match_text = search_text or filter_text
    match_scope = None
    if not search_text and filter_text and filter_column not in (None, "__all__", "All columns"):
        match_scope = [filter_column]
    if match_text:
        match_priority, match_column, match_column_key, match_value = _match_metadata(
            filtered, columns, match_text, scope=match_scope
        )
        filtered = filtered.with_columns(
            match_priority.alias("__match_priority"),
            match_column.alias("match_column"),
            match_column_key.alias("match_column_key"),
            match_value.alias("match_value"),
        )
    else:
        filtered = filtered.with_columns(
            pl.lit("").alias("match_column"),
            pl.lit("").alias("match_column_key"),
            pl.lit("").alias("match_value"),
        )

    manual_match_value_sort = sort_by == "__match_value__"
    manual_sort = manual_match_value_sort or sort_by in columns
    selected_sort = "match_value" if manual_match_value_sort else (
        sort_by if manual_sort else ("bodyId" if "bodyId" in columns else columns[0])
    )
    sort_columns = []
    sort_directions = []
    if match_text and (not manual_sort or manual_match_value_sort):
        # Keep each match-column subset together in bodyId → type → instance →
        # taxonomy order, then sort the actual matched values within that
        # subset. If the UI explicitly requests descending matched values,
        # reverse only the values inside each priority group.
        sort_columns.extend(("__match_priority", "match_value"))
        sort_directions.extend((False, bool(descending) if manual_match_value_sort else False))
    else:
        if selected_sort == "bodyId":
            # Body IDs are stored as strings in the cache to preserve large
            # values in the browser.  Sort them numerically so 10001 follows
            # 9999 instead of appearing between 10000 and 100011.
            filtered = filtered.with_columns(
                pl.col(selected_sort).cast(pl.UInt64, strict=False).alias("__body_id_sort")
            )
            sort_columns.append("__body_id_sort")
        else:
            sort_columns.append(selected_sort)
        sort_directions.append(bool(descending))
        if match_text:
            # An explicit sort column is primary; matching priority only
            # resolves rows with the same selected-column value.
            sort_columns.append("__match_priority")
            sort_directions.append(False)
    try:
        try:
            filtered = filtered.sort(
                sort_columns,
                descending=sort_directions,
                nulls_last=[True] * len(sort_columns),
            )
        except TypeError:  # compatibility with older Polars releases
            filtered = filtered.sort(sort_columns, descending=sort_directions)
    except TypeError:  # compatibility with older Polars releases
        filtered = filtered.sort(sort_columns, descending=sort_directions)

    total = int(filtered.height)
    pages = max(1, (total + page_size - 1) // page_size)
    current_page = max(1, min(int(page or 1), pages))
    output_columns = ["match_column", "match_value", "match_column_key", *columns]
    rows = (
        filtered
        .select(output_columns)
        .slice((current_page - 1) * page_size, page_size)
        .to_dicts()
    )

    def json_value(value):
        if value is None:
            return ""
        if isinstance(value, (datetime, date, time)):
            return value.isoformat()
        return value

    safe_rows = [
        {column: json_value(value) for column, value in row.items()}
        for row in rows
    ]
    return NeuronIndexPage(
        rows=safe_rows,
        total=total,
        page=current_page,
        pages=pages,
        page_size=page_size,
        sort_by=(
            "match_value"
            if match_text and (not manual_sort or manual_match_value_sort)
            else selected_sort
        ),
        descending=(
            bool(descending)
            if match_text and manual_match_value_sort
            else (False if match_text and not manual_sort else bool(descending))
        ),
    )
