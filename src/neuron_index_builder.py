"""Build the compact local neuron index used by the UI and cache layer.

The pulled ``*_allneurons_neuron_df.csv`` file remains the authoritative
dataset metadata.  This module creates a typed, columnar projection for local
search and display.  Serialized/blob-like columns are intentionally omitted:
they are not useful suggestion identifiers and can dominate the size of the
source CSV (for example ``roiInfo`` and ``inputRois``).

Cache completion state is kept by :class:`FindNeuronConnection`; this module
only owns metadata-source discovery and the list of columns that make up the
search projection so the UI and backend use the same scope.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional


# These fields are serialized collections or bookkeeping rather than useful
# viewer/search columns.  Small scalar text fields such as synonyms and
# matchingNotes remain in the projection so the viewer truly searches all
# scalar metadata columns; the suggestion layer filters those separately.
SEARCH_EXCLUDED_COLUMNS = frozenset({
    "last_fetched",
    "roiinfo",
    "inputrois",
    "outputrois",
    "unnamed: 0",
    "",
})


def dataset_folder(dataset: str) -> str:
    """Return the repository-safe folder name for a dataset identifier."""
    return str(dataset or "").strip().replace(":", "_").replace(".", "_")


def metadata_candidates(dataset: str, datasets_dir: Path) -> List[Path]:
    """Return local neuron metadata files in preferred read order."""
    folder = Path(datasets_dir) / dataset_folder(dataset)
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
            path
            for pattern in (
                "*_allneurons_neuron_df.csv",
                "*_neuron_df.csv",
                "*_allneurons_neuron_df.parquet",
                "*_neuron_df.parquet",
            )
            for path in folder.glob(pattern)
        ],
        key=lambda path: path.name,
    )
    result: List[Path] = []
    for path in (*exact, *discovered):
        if path.is_file() and path not in result:
            result.append(path)
    return result


def metadata_path(dataset: str, datasets_dir: Path) -> Optional[Path]:
    """Return the first usable local neuron metadata file, if any."""
    candidates = metadata_candidates(dataset, datasets_dir)
    return candidates[0] if candidates else None


def metadata_columns(path: Path) -> List[str]:
    """Return source columns without materializing the metadata table."""
    import polars as pl

    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return searchable_columns(pl.scan_parquet(path).collect_schema().names())
    # Header-only discovery avoids a schema-inference pass over a 500+ MiB
    # CSV.  The actual projection read below still parses the selected fields
    # once, with bodyId forced to text.
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        try:
            names = next(csv.reader(stream))
        except StopIteration:
            names = []
    return searchable_columns(names)


def searchable_columns(columns: Iterable[str]) -> List[str]:
    """Return metadata columns retained in the compact search projection."""
    result: List[str] = []
    seen = set()
    for column in columns:
        name = str(column)
        if name.casefold() in SEARCH_EXCLUDED_COLUMNS or name in seen:
            continue
        result.append(name)
        seen.add(name)
    return result


def read_metadata_projection(path: Path):
    """Read the compact metadata projection as a Polars DataFrame.

    Only columns in :func:`searchable_columns` are materialized.  The caller
    adds cache bookkeeping fields and writes the result to the cache index.
    """
    import polars as pl

    path = Path(path)
    if path.suffix.lower() == ".parquet":
        schema = pl.scan_parquet(path).collect_schema()
        columns = searchable_columns(schema.names())
        frame = pl.read_parquet(path, columns=columns)
    else:
        columns = metadata_columns(path)
        frame = pl.read_csv(
            path,
            columns=columns,
            # Body IDs can exceed signed 64-bit and JavaScript's safe integer
            # range.  Read them as text before any projection/cast so Polars
            # never turns an out-of-range value into null.
            schema_overrides={"bodyId": pl.Utf8},
            infer_schema_length=1000,
            ignore_errors=True,
            try_parse_dates=False,
        )

    if "bodyId" not in frame.columns:
        raise ValueError(f"Neuron metadata has no bodyId column: {path}")

    # Large FlyWire IDs must remain exact.  The UI also uses strings so the
    # browser never rounds a value beyond JavaScript's safe integer range.
    frame = frame.with_columns(
        pl.col("bodyId").cast(pl.Utf8, strict=False).fill_null("").alias("bodyId")
    )
    for column in ("type", "instance"):
        if column in frame.columns:
            frame = frame.with_columns(
                pl.col(column).cast(pl.Utf8, strict=False).fill_null("").alias(column)
            )
        else:
            frame = frame.with_columns(pl.lit("").alias(column))
    return frame
