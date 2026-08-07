"""
ThresholdedConnectionMap: one synapse cutoff of a connection cache as an
independent dataset.

The connection cache stores the COMPLETE graph (every edge with weight >= 1).
A query with ``min_synapse_num = t`` works on the derived graph

    D_t = { edges with weight >= t }

which is treated as an independent dataset: its totals (total incoming per
post neuron / per post type) are computed from D_t alone and cached per
cutoff, so ratios at different thresholds are never mixed.

The map owns both aggregate tables (bodyId-level and type-level) plus the
neuron-index lookup, so numerator and denominator always come from the same
D_t. Aggregates are computed lazily with vectorized Polars and cached for
the lifetime of the map (one map per cutoff; stale maps are discarded when
the underlying cache files change).
"""

import threading
from typing import Optional, Union

import pandas as pd
import polars as pl


class ThresholdedConnectionMap:
    """The connection graph at synapse cutoff ``t``, with its own aggregates."""

    def __init__(
        self,
        db_path: str,
        neuron_index_path: str,
        min_weight: int = 1,
        conn_frame: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
        source_signature: tuple = None,
    ):
        self._db_path = db_path
        self._index_path = neuron_index_path
        self._min_weight = int(min_weight)
        self._conn_frame = conn_frame
        self.source_signature = source_signature
        self._lock = threading.Lock()
        self._cache: dict = {}

    @property
    def min_weight(self) -> int:
        return self._min_weight

    def _load_neuron_index(self) -> pl.DataFrame:
        """bodyId -> type lookup from the cache's neuron index.

        Preserves the legacy dict semantics: the LAST entry wins for
        duplicate bodyIds; untyped neurons are excluded (they are grouped by
        their bodyId at type level and served by the bodyId-level table).
        """
        index = pl.read_parquet(self._index_path, columns=["bodyId", "type"])
        index = index.unique(subset=["bodyId"], keep="last")
        if "type" in index.columns:
            index = index.filter(
                pl.col("type").is_not_null() & (pl.col("type") != "")
            )
        return index

    def _thresholded_source(self) -> pl.LazyFrame:
        """The D_t edge source (lazy scan for disk, eager frame in memory)."""
        if self._conn_frame is not None:
            conn = self._conn_frame
            if isinstance(conn, pd.DataFrame):
                conn = pl.from_pandas(conn)
            lazy = conn.lazy()
        else:
            lazy = pl.scan_parquet(self._db_path)
        if self._min_weight > 1:
            lazy = lazy.filter(pl.col("weight") >= self._min_weight)
        return lazy

    def total_incoming_by_bodyid(self) -> pl.DataFrame:
        """Full-D_t aggregate ``bodyId_post -> total_incoming_weight``."""
        cached = self._cache.get("by_bodyid")
        if cached is not None:
            return cached
        with self._lock:
            cached = self._cache.get("by_bodyid")
            if cached is not None:
                return cached
            result = (
                self._thresholded_source()
                .group_by("bodyId_post")
                .agg(pl.col("weight").sum().alias("total_incoming_weight"))
                .collect()
            )
            self._cache["by_bodyid"] = result
            return result

    def total_incoming_by_type(self) -> pl.DataFrame:
        """Full-D_t aggregate ``type_post -> total_incoming_weight``."""
        cached = self._cache.get("by_type")
        if cached is not None:
            return cached
        with self._lock:
            cached = self._cache.get("by_type")
            if cached is not None:
                return cached
            result = (
                self._thresholded_source()
                .join(
                    self._load_neuron_index().lazy(),
                    left_on="bodyId_post",
                    right_on="bodyId",
                    how="inner",
                )
                .group_by("type")
                .agg(pl.col("weight").sum().alias("total_incoming_weight"))
                .rename({"type": "type_post"})
                .collect()
            )
            self._cache["by_type"] = result
            return result
