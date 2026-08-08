"""
Morphological similarity for DROCAT.

Two comparison backends are provided:

- **Vector-based** (default, fast): each neuron is reduced to a fixed vector
  of ~24 L-Measure-style morphometrics plus a 100-dim persistence vector
  (navis). Vectors are cached per dataset as a single parquet file
  (``cache/{dataset}/morphology/skeleton_vectors.parquet``) and queried with
  cosine / Pearson similarity.
- **NBLAST** (navis implementation of Costa et al. 2016): the canonical
  pairwise morphology score. Dotprops are NEVER cached (a persisted cache
  would be ~100 KB/neuron); they are rebuilt on demand for the vector-
  prefiltered candidate set of the current query.

Also provides ``enrich_homolog_results``, which attaches vector-based
morphological similarity scores to already-ranked homolog finding results
(post-search, result rows only).
"""

import json
import multiprocessing as mp
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import navis

from statvis import getNeurons

# Feature columns (morphometrics part of the vector). The full per-neuron
# vector is these 24 features + 100 persistence dimensions.
MORPHOMETRIC_FEATURES: List[str] = [
    "cable_length", "n_nodes", "n_branch", "n_leaf", "n_root", "n_primary",
    "max_branch_order", "mean_edge_length",
    "bbox_x", "bbox_y", "bbox_z", "bbox_diagonal",
    "max_path_length", "mean_path_length", "std_path_length", "tortuosity",
    "soma_radius", "strahler_max", "strahler_mean",
    "leaf_density", "branch_density", "bbox_xy_ratio", "path_ratio", "edge_cv",
]
PERSISTENCE_DIM = 100
VECTOR_DIM = len(MORPHOMETRIC_FEATURES) + PERSISTENCE_DIM

VECTOR_CACHE_VERSION = 1


# =============================================================================
# Feature extraction
# =============================================================================

def _dataset_folder(dataset: str) -> str:
    """Map a dataset name to its cache folder (hemibrain:v1.2.1 -> hemibrain_v1_2_1)."""
    return dataset.replace(":", "_").replace(".", "_")


def compute_morphometrics(neuron) -> Dict[str, float]:
    """Compute the curated L-Measure-style morphometric feature set.

    All features are computed directly from ``TreeNeuron.nodes`` (vectorized
    numpy where possible) so results are deterministic and fast; navis is only
    used for the persistence vector (see ``compute_persistence_vector``).
    """
    df = neuron.nodes
    coords = df[["x", "y", "z"]].to_numpy(dtype=float)
    parent = df["parent_id"].to_numpy(dtype=np.int64)
    node_ids = df["node_id"].to_numpy(dtype=np.int64)
    has_radius = "radius" in df.columns
    radius = df["radius"].to_numpy(dtype=float) if has_radius else np.zeros(len(df))
    n = len(df)

    # Root detection must use "parent not in the node set": SWC parent ids are
    # 0-based in navis tables, so parent_id == 0 is a VALID edge (not a root).
    is_root = ~np.isin(parent, node_ids)

    # Parent index lookup (robust to unsorted node ids).
    id_to_idx = {int(i): k for k, i in enumerate(node_ids)}
    pidx = np.array([id_to_idx.get(int(p), -1) for p in parent], dtype=np.int64)
    has_parent = ~is_root

    # Edge lengths (parent -> node).
    seg = coords[has_parent] - coords[pidx[has_parent]]
    edge_len = np.linalg.norm(seg, axis=1)
    edge_len_by_child = np.zeros(n)
    edge_len_by_child[has_parent] = edge_len
    cable = float(edge_len.sum())

    # Child counts.
    child_count = np.bincount(pidx[has_parent], minlength=n)
    n_branch = int((child_count >= 2).sum())
    n_leaf = int((child_count == 0).sum())
    n_root = int(is_root.sum())
    n_primary = int(is_root[pidx[has_parent]].sum()) if has_parent.any() else 0

    # Tree traversal: depth, path length from root, root id per node.
    children: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        if has_parent[i]:
            children[pidx[i]].append(i)
    roots = np.where(is_root)[0].tolist()
    depth = np.zeros(n)
    path_len = np.zeros(n)
    root_of = np.zeros(n, dtype=np.int64)
    stack = list(roots)
    for r in roots:
        root_of[r] = r
    while stack:
        i = stack.pop()
        for c in children[i]:
            depth[c] = depth[i] + 1
            path_len[c] = path_len[i] + edge_len_by_child[c]
            root_of[c] = root_of[i]
            stack.append(c)

    # Leaf statistics (tips).
    leaves = np.where(child_count == 0)[0]
    if leaves.size:
        straight = np.linalg.norm(
            coords[leaves] - coords[root_of[leaves]], axis=1
        )
        tortuosity = float(np.mean(
            path_len[leaves] / np.maximum(straight, 1e-9)
        ))
        mean_path = float(path_len[leaves].mean())
        std_path = float(path_len[leaves].std())
    else:
        tortuosity, mean_path, std_path = 1.0, 0.0, 0.0

    # Strahler order via reverse depth processing. A node with a single
    # child inherits the child's order; with >= 2 children it increments only
    # when the two highest child orders are equal.
    strahler = np.ones(n)
    for i in sorted(range(n), key=lambda k: -depth[k]):
        vals = [strahler[c] for c in children[i]]
        if not vals:
            strahler[i] = 1
        elif len(vals) == 1:
            strahler[i] = vals[0]
        else:
            top2 = sorted(vals, reverse=True)[:2]
            strahler[i] = top2[0] + 1 if top2[0] == top2[1] else top2[0]

    # Soma radius (0 when absent).
    soma_radius = 0.0
    try:
        soma = neuron.soma
        if soma is not None:
            soma_ids = [soma] if np.isscalar(soma) else list(soma)
            radii = [float(radius[id_to_idx[int(s)]]) for s in soma_ids if int(s) in id_to_idx]
            if radii:
                soma_radius = float(np.mean(radii))
    except Exception:
        pass

    bbox = np.ptp(coords, axis=0)
    bbox_diag = float(np.linalg.norm(bbox))
    mean_edge = float(edge_len.mean()) if edge_len.size else 0.0
    edge_std = float(edge_len.std()) if edge_len.size else 0.0

    features = {
        "cable_length": cable,
        "n_nodes": float(n),
        "n_branch": float(n_branch),
        "n_leaf": float(n_leaf),
        "n_root": float(n_root),
        "n_primary": float(n_primary),
        "max_branch_order": float(depth.max()) if n else 0.0,
        "mean_edge_length": mean_edge,
        "bbox_x": float(bbox[0]),
        "bbox_y": float(bbox[1]),
        "bbox_z": float(bbox[2]),
        "bbox_diagonal": bbox_diag,
        "max_path_length": float(path_len.max()) if n else 0.0,
        "mean_path_length": mean_path,
        "std_path_length": std_path,
        "tortuosity": tortuosity,
        "soma_radius": soma_radius,
        "strahler_max": float(strahler.max()) if n else 0.0,
        "strahler_mean": float(strahler.mean()) if n else 0.0,
        "leaf_density": n_leaf / cable if cable > 0 else 0.0,
        "branch_density": n_branch / cable if cable > 0 else 0.0,
        "bbox_xy_ratio": float(bbox[0]) / max(float(bbox[1]), 1e-9),
        "path_ratio": (float(path_len.max()) if n else 0.0) / max(bbox_diag, 1e-9),
        "edge_cv": edge_std / max(mean_edge, 1e-9),
    }
    return {k: (0.0 if not np.isfinite(v) else float(v)) for k, v in features.items()}


def compute_persistence_vector(neuron) -> np.ndarray:
    """Return the 100-dim persistence vector (navis), zeros on failure."""
    try:
        pv = navis.morpho.persistence_vectors(neuron, samples=PERSISTENCE_DIM)
        arr = np.asarray(pv[0]).ravel()  # primary root vector
        if arr.size != PERSISTENCE_DIM:
            return np.zeros(PERSISTENCE_DIM)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(float)
    except Exception:
        return np.zeros(PERSISTENCE_DIM)


# =============================================================================
# Mesh-based features (for datasets cached as MeshNeurons, e.g. FlyWire)
# =============================================================================

def compute_mesh_morphometrics(mesh) -> Dict[str, float]:
    """24 morphometrics computed directly from a navis MeshNeuron.

    Uses the same feature names as the skeleton set so both neuron types
    share one vector schema (the shape block differs: persistence for
    skeletons, a spatial histogram for meshes).
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n = len(vertices)

    # Triangle geometry.
    tri = vertices[faces]
    e0 = tri[:, 1] - tri[:, 0]
    e1 = tri[:, 2] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 1]
    edge_lens = np.concatenate([
        np.linalg.norm(e0, axis=1), np.linalg.norm(e1, axis=1),
        np.linalg.norm(e2, axis=1),
    ])
    tri_area = 0.5 * np.linalg.norm(np.cross(e0, e1), axis=1)
    area = float(tri_area.sum())

    # Bounding box.
    bbox = np.ptp(vertices, axis=0)
    bbox_diag = float(np.linalg.norm(bbox))

    # Centroid statistics.
    centroid = vertices.mean(axis=0)
    radii = np.linalg.norm(vertices - centroid, axis=1)

    features = {
        "cable_length": float(edge_lens.sum()),  # total edge length (mesh analog)
        "n_nodes": float(n),
        "n_branch": float(len(faces)),
        "n_leaf": float(len(np.unique(faces))),
        "n_root": 0.0,
        "n_primary": 0.0,
        "max_branch_order": 0.0,
        "mean_edge_length": float(edge_lens.mean()) if edge_lens.size else 0.0,
        "bbox_x": float(bbox[0]),
        "bbox_y": float(bbox[1]),
        "bbox_z": float(bbox[2]),
        "bbox_diagonal": bbox_diag,
        "max_path_length": float(radii.max()) if n else 0.0,
        "mean_path_length": float(radii.mean()) if n else 0.0,
        "std_path_length": float(radii.std()) if n else 0.0,
        "tortuosity": 1.0,
        "soma_radius": 0.0,
        "strahler_max": 0.0,
        "strahler_mean": 0.0,
        "leaf_density": area / max(bbox_diag**2, 1e-9),  # surface compactness
        "branch_density": len(faces) / max(area, 1e-9),  # face density
        "bbox_xy_ratio": float(bbox[0]) / max(float(bbox[1]), 1e-9),
        "path_ratio": float(radii.max()) / max(bbox_diag, 1e-9) if n else 0.0,
        "edge_cv": (float(edge_lens.std()) / max(float(edge_lens.mean()), 1e-9))
        if edge_lens.size else 0.0,
    }
    return {k: (0.0 if not np.isfinite(v) else float(v)) for k, v in features.items()}


def compute_spatial_histogram(mesh, bins: Tuple[int, int, int] = (5, 5, 4)) -> np.ndarray:
    """Coarse 3D shape descriptor: vertex counts in a normalized grid.

    Default 5x5x4 = 100 bins, matching the persistence-vector dimension so
    skeleton and mesh vectors share one schema.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    if not len(vertices):
        return np.zeros(PERSISTENCE_DIM)
    lo = vertices.min(axis=0)
    span = np.ptp(vertices, axis=0)
    span = np.where(span > 0, span, 1.0)
    norm = (vertices - lo) / span
    idx = np.floor(norm * np.array(bins)).astype(int)
    idx = np.clip(idx, 0, np.array(bins) - 1)
    flat = idx[:, 0] * (bins[1] * bins[2]) + idx[:, 1] * bins[2] + idx[:, 2]
    hist = np.bincount(flat, minlength=bins[0] * bins[1] * bins[2]).astype(float)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist


def vectorize_neuron(neuron) -> Tuple[Dict[str, float], np.ndarray]:
    """Return (24-feature dict, full 124-dim vector).

    Dispatches on the neuron type: TreeNeurons use skeleton morphometrics +
    a persistence vector; MeshNeurons (FlyWire bulk caches) use mesh
    morphometrics + a 100-bin spatial histogram. Both share one schema.
    """
    if hasattr(neuron, "nodes") and neuron.nodes is not None and len(neuron.nodes):
        morph = compute_morphometrics(neuron)
        shape = compute_persistence_vector(neuron)
    elif hasattr(neuron, "vertices") and neuron.vertices is not None and len(neuron.vertices):
        morph = compute_mesh_morphometrics(neuron)
        shape = compute_spatial_histogram(neuron)
    else:
        raise ValueError(
            "Unsupported neuron type for vectorization "
            f"({type(neuron).__name__}): expected a skeleton or a mesh."
        )
    vector = np.concatenate([
        np.array([morph[f] for f in MORPHOMETRIC_FEATURES], dtype=float),
        shape,
    ])
    return morph, vector


def _vectorize_one_file(path: str) -> Optional[Tuple[int, List[float], List[float]]]:
    """Module-level worker for parallel cache builds (picklable).

    Returns None for un-vectorizable pickles (corrupt or unexpected types)
    so a single bad file cannot break the whole build."""
    try:
        with open(path, "rb") as f:
            neuron = pickle.load(f)
        morph, vector = vectorize_neuron(neuron)
    except Exception:
        return None
    body_id = int(Path(path).stem)
    # Shape block (persistence / spatial histogram) = the tail after the
    # morphometric block.
    return body_id, [morph[f] for f in MORPHOMETRIC_FEATURES], vector[len(MORPHOMETRIC_FEATURES):].tolist()


def _import_visualizer():
    """Lazily import the VisualizeSkeleton class (heavy module; never loaded
    unless a run actually renders). Returns None when unavailable."""
    try:
        from visualize_skeleton import VisualizeSkeleton
        return VisualizeSkeleton
    except Exception:
        return None


def _load_neuron_type_map(dataset: str, project_root: Optional[str] = None
                          ) -> Tuple[Dict[int, str], Dict[int, str]]:
    """bodyId -> type / instance maps for a dataset.

    Uses the allneurons neuron table (fullest coverage), falling back to the
    neuron index parquet. These are the same sources ``SkeletonVectorCache``
    merges into the vector cache, so type lookups work even for datasets that
    have no vector cache yet (e.g. male-cns v1.0 with cached skeletons only).
    """
    root = Path(project_root) if project_root else Path(__file__).parent.parent
    folder = _dataset_folder(dataset)
    type_map: Dict[int, str] = {}
    instance_map: Dict[int, str] = {}

    csv_path = root / "datasets" / folder / f"{folder}_allneurons_neuron_df.csv"
    if csv_path.exists():
        try:
            tdf = pd.read_csv(csv_path, usecols=["bodyId", "type", "instance"])
            tdf["bodyId"] = tdf["bodyId"].astype(np.int64)
            type_map = dict(zip(tdf["bodyId"], tdf["type"].fillna("").astype(str)))
            instance_map = dict(zip(tdf["bodyId"], tdf["instance"].fillna("").astype(str)))
            return type_map, instance_map
        except Exception:
            pass

    index_path = root / "cache" / folder / "neuron_index.parquet"
    if index_path.exists():
        try:
            idx_df = pd.read_parquet(index_path, columns=["bodyId", "type", "instance"])
            idx_df["bodyId"] = idx_df["bodyId"].astype(np.int64)
            type_map = dict(zip(idx_df["bodyId"], idx_df["type"].fillna("").astype(str)))
            instance_map = dict(zip(idx_df["bodyId"], idx_df["instance"].fillna("").astype(str)))
        except Exception:
            pass
    return type_map, instance_map


def _find_skeleton_file(dataset: str, body_id: int,
                        project_root: Optional[str] = None) -> Optional[Path]:
    """Locate a cached skeleton/mesh pickle for a bodyId.

    Searches the dataset's skeletons directory recursively (datasets such as
    FlyWire keep bulk downloads in nested subfolders) and returns the first
    match, or None.
    """
    root = Path(project_root) if project_root else Path(__file__).parent.parent
    cache_dir = root / "cache" / _dataset_folder(dataset) / "skeletons"
    if not cache_dir.exists():
        return None
    direct = cache_dir / f"{body_id}.pkl"
    if direct.exists():
        return direct
    nested = sorted(cache_dir.rglob(f"{body_id}.pkl"))
    return nested[0] if nested else None


# =============================================================================
# Similarity
# =============================================================================

def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one vector against rows of a matrix (standardized)."""
    q = np.asarray(query, dtype=float).ravel()
    m = np.asarray(matrix, dtype=float)
    qn = np.linalg.norm(q)
    if qn == 0:
        return np.zeros(len(m))
    rows = np.linalg.norm(m, axis=1)
    scores = m @ q / (qn * np.maximum(rows, 1e-12))
    return np.where(rows == 0, 0.0, scores)


def pearson_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Pearson correlation of one vector against rows of a matrix."""
    q = np.asarray(query, dtype=float).ravel() - np.asarray(query, dtype=float).mean()
    m = np.asarray(matrix, dtype=float)
    m = m - m.mean(axis=1, keepdims=True)
    return cosine_similarity_matrix(q, m)


def similarity_matrix(query: np.ndarray, matrix: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Similarity of one vector against matrix rows for a metric."""
    if metric == "pearson":
        return pearson_similarity_matrix(query, matrix)
    return cosine_similarity_matrix(query, matrix)


# =============================================================================
# Skeleton vector cache
# =============================================================================

class SkeletonVectorCache:
    """Per-dataset cache of vectorized skeletons (parquet + meta.json)."""

    def __init__(self, dataset: str, project_root: Optional[str] = None,
                 n_workers: int = 8, verbose: bool = True):
        self.dataset = dataset
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.n_workers = max(1, int(n_workers))
        self.verbose = verbose
        folder = _dataset_folder(dataset)
        self.morph_dir = self.project_root / "cache" / folder / "morphology"
        self.skeleton_dir = self.project_root / "cache" / folder / "skeletons"
        self.parquet_path = self.morph_dir / "skeleton_vectors.parquet"
        self.meta_path = self.morph_dir / "meta.json"

    # ------------------------------------------------------------------ paths
    def cache_exists(self) -> bool:
        return self.parquet_path.exists()

    def _discover_skeleton_files(self) -> List[str]:
        """All cached skeleton/mesh pickles, including nested bulk folders
        (e.g. FlyWire's skeletons/FLYWIRE_simp95_soma80_r20/)."""
        if not self.skeleton_dir.exists():
            return []
        return sorted(str(p) for p in self.skeleton_dir.rglob("*.pkl"))

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------ meta
    def _write_meta(self, stats: Dict[str, List[float]], n_rows: int):
        meta = {
            "version": VECTOR_CACHE_VERSION,
            "dataset": self.dataset,
            "feature_columns": MORPHOMETRIC_FEATURES,
            "persistence_dim": PERSISTENCE_DIM,
            "n_rows": n_rows,
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "mean": stats["mean"],
            "std": stats["std"],
        }
        self.meta_path.write_text(json.dumps(meta, indent=2))

    def _load_meta(self) -> Optional[dict]:
        if not self.meta_path.exists():
            return None
        try:
            return json.loads(self.meta_path.read_text())
        except Exception:
            return None

    # ------------------------------------------------------------ build
    def build(self, fetch_missing: int = 0) -> Dict[str, int]:
        """Vectorize all cached skeletons (incremental) and persist.

        Reuses existing rows; optionally fetches up to ``fetch_missing``
        additional neurons (persisted to the skeleton cache first).
        """
        self.morph_dir.mkdir(parents=True, exist_ok=True)
        self.skeleton_dir.mkdir(parents=True, exist_ok=True)

        existing: Dict[int, dict] = {}
        if self.parquet_path.exists():
            try:
                df_old = pd.read_parquet(self.parquet_path)
                existing = {int(r["bodyId"]): r for r in df_old.to_dict("records")}
            except Exception:
                existing = {}

        # Candidate skeleton files not yet vectorized.
        files = self._discover_skeleton_files()
        pending = [f for f in files if int(Path(f).stem) not in existing]

        # Optional on-demand fetch to extend coverage (cap applies).
        fetched_new = 0
        if fetch_missing and fetch_missing > 0:
            index_path = self.project_root / "cache" / _dataset_folder(self.dataset) / "neuron_index.parquet"
            index: List[int] = []
            if index_path.exists():
                try:
                    idx_df = pd.read_parquet(index_path, columns=["bodyId"])
                    index = [int(b) for b in idx_df["bodyId"].tolist()]
                except Exception:
                    index = []
            if index:
                have = {int(b) for b in (list(existing) + [int(Path(f).stem) for f in files])}
                missing = [b for b in index if b not in have]
                for bid in missing[:fetch_missing]:
                    nrn = fetch_skeleton_on_demand(self.dataset, bid, project_root=str(self.project_root))
                    if nrn is not None:
                        pending.append(str(self.skeleton_dir / f"{bid}.pkl"))
                        fetched_new += 1

        rows = []
        if pending:
            started = time.time()
            self._log(
                f"[SkeletonVectorCache] Vectorizing {len(pending)} skeletons "
                f"({self.dataset})..."
            )
            if self.n_workers > 1:
                try:
                    rows = self._vectorize_parallel(pending)
                except Exception:
                    rows = [None] * len(pending)
                    for i, p in enumerate(pending):
                        rows[i] = _vectorize_one_file(p)
            else:
                rows = [_vectorize_one_file(p) for p in pending]
            elapsed = time.time() - started
            self._log(
                f"[SkeletonVectorCache] Vectorized {len(pending)} neurons in "
                f"{elapsed:.1f}s ({elapsed / max(len(pending), 1) * 1000:.1f} ms/neuron)"
            )

        # Merge with existing rows (type/instance are refreshed from the
        # neuron index below, so drop any stale copies from previous builds).
        records = []
        for bid, rec in existing.items():
            row = {k: v for k, v in rec.items() if k not in ("type", "instance")}
            row["bodyId"] = int(bid)
            records.append(row)
        for row in rows:
            if row is None:
                continue
            bid, morph_vals, pv_vals = row
            record = {"bodyId": bid}
            for name, val in zip(MORPHOMETRIC_FEATURES, morph_vals):
                record[name] = float(val)
            for i, val in enumerate(pv_vals):
                record[f"pv_{i}"] = float(val)
            records.append(record)

        if not records:
            self._log("[SkeletonVectorCache] No skeletons available to vectorize.")
            self._write_meta({"mean": [], "std": []}, 0)
            return {"rows": 0, "new": 0, "fetched": 0}

        df = pd.DataFrame(records).sort_values("bodyId").reset_index(drop=True)

        # Attach type/instance (used by type-level aggregation and result
        # reporting). The allneurons neuron table has the fullest bodyId ->
        # type/instance coverage; the neuron index is the fallback.
        type_map, instance_map = _load_neuron_type_map(self.dataset, str(self.project_root))
        if type_map:
            df["type"] = df["bodyId"].map(type_map).fillna("")
            df["instance"] = df["bodyId"].map(instance_map or {}).fillna("")
        else:
            df["type"] = ""
            df["instance"] = ""

        df.to_parquet(self.parquet_path, index=False)

        # Z-score stats over the population.
        mat = self._raw_matrix(df)
        mean = mat.mean(axis=0).tolist()
        std = mat.std(axis=0).tolist()
        std = [s if s > 0 else 1.0 for s in std]
        self._write_meta({"mean": mean, "std": std}, len(df))

        self._log(
            f"[SkeletonVectorCache] Cache ready: {len(df)} rows "
            f"({len(rows)} new, {fetched_new} fetched) -> {self.parquet_path}"
        )
        return {"rows": len(df), "new": len(rows), "fetched": fetched_new}

    def _vectorize_parallel(self, files: List[str]) -> List[Tuple[int, List[float], List[float]]]:
        ctx = mp.get_context("fork") if hasattr(mp, "get_context") and "fork" in mp.get_all_start_methods() else mp.get_context()
        with ProcessPoolExecutor(max_workers=self.n_workers, mp_context=ctx) as ex:
            return list(ex.map(_vectorize_one_file, files, chunksize=16))

    # ------------------------------------------------------------ load
    @staticmethod
    def _raw_matrix(df: pd.DataFrame) -> np.ndarray:
        cols = MORPHOMETRIC_FEATURES + [f"pv_{i}" for i in range(PERSISTENCE_DIM)]
        return df[cols].to_numpy(dtype=float)

    def load(self) -> Optional[dict]:
        """Load the cache: meta + raw df + standardized matrix + index arrays."""
        if not self.parquet_path.exists():
            return None
        df = pd.read_parquet(self.parquet_path)
        if df.empty:
            return None
        meta = self._load_meta() or {}
        raw = self._raw_matrix(df)
        mean = np.asarray(meta.get("mean") or raw.mean(axis=0), dtype=float)
        std = np.asarray(meta.get("std") or raw.std(axis=0), dtype=float)
        std = np.where(std <= 0, 1.0, std)
        X = (raw - mean) / std
        return {
            "meta": meta,
            "df": df,
            "raw": raw,
            "X": X,
            "bodyIds": df["bodyId"].astype(np.int64).to_numpy(),
            "types": df.get("type", pd.Series([""] * len(df))).fillna("").astype(str).tolist(),
            "instances": df.get("instance", pd.Series([""] * len(df))).fillna("").astype(str).tolist(),
        }

    def ensure(self, fetch_missing: int = 0) -> dict:
        """Auto-build the cache on first query (lazy)."""
        if not self.cache_exists():
            self._log(f"[SkeletonVectorCache] Building vector cache for {self.dataset}...")
            return self.build(fetch_missing=fetch_missing)
        return {"rows": len(pd.read_parquet(self.parquet_path)) if self.parquet_path.exists() else 0,
                "new": 0, "fetched": 0}

    def coverage(self) -> Dict[str, int]:
        """Skeleton and vector counts for the dataset."""
        n_skeletons = len(self._discover_skeleton_files())
        n_vectors = 0
        if self.parquet_path.exists():
            try:
                n_vectors = len(pd.read_parquet(self.parquet_path))
            except Exception:
                n_vectors = 0
        return {"skeletons": n_skeletons, "vectors": n_vectors}

    # ------------------------------------------------------------ vectors_for
    def vectors_for(self, body_ids: List[int], compute_missing: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Return standardized vectors for bodyIds.

        Rows missing from the cache are computed on the fly when a skeleton
        file exists (``compute_missing=True``); otherwise they are NaN rows.
        Never fetches from the server and never forces a full cache build.
        """
        body_ids = [int(b) for b in body_ids]
        data = self.load()
        known: Dict[int, int] = {}
        X = np.zeros((0, VECTOR_DIM))
        if data is not None:
            known = {int(b): i for i, b in enumerate(data["bodyIds"])}
            X = data["X"]

        result = np.full((len(body_ids), VECTOR_DIM), np.nan)
        for j, bid in enumerate(body_ids):
            if bid in known:
                result[j] = X[known[bid]]
                continue
            if compute_missing:
                pkl = _find_skeleton_file(
                    self.dataset, bid, project_root=str(self.project_root)
                )
                if pkl is not None:
                    try:
                        with open(pkl, "rb") as f:
                            neuron = pickle.load(f)
                        _, vec = vectorize_neuron(neuron)
                        result[j] = vec
                    except Exception:
                        result[j] = np.nan
        mask = ~np.isnan(result[:, 0])
        return result, mask


# =============================================================================
# On-demand skeleton fetching
# =============================================================================

def _fetch_neuprint_skeleton(dataset: str, body_id: int):
    """Fetch one skeleton from a NeuPrint dataset.

    ``neuprint.fetch_skeleton`` returns a skeleton DataFrame; convert it to a
    navis TreeNeuron before caching.
    """
    from neuprint import Client, fetch_skeleton, set_default_client
    try:
        from utils.token_manager import token_manager
        token = token_manager.get_token("NEUPRINT_TOKEN")
    except Exception:
        token = ""
    client = Client("neuprint.janelia.org", dataset=dataset, token=token)
    set_default_client(client)
    df = fetch_skeleton(body_id)
    if df is None or len(df) == 0:
        return None
    try:
        return navis.TreeNeuron(df)
    except Exception:
        return None


def _fetch_cave_skeleton(dataset: str, body_id: int):
    """Fetch one skeleton from a FlyWire/CAVE dataset."""
    from cave_data_fetcher import CAVEDataFetcher
    fetcher = CAVEDataFetcher(dataset=_dataset_folder(dataset), verbose=False)
    return fetcher.fetch_skeleton(body_id, use_cache=True)


def fetch_skeleton_on_demand(dataset: str, body_id: int,
                             project_root: Optional[str] = None,
                             persist: bool = True) -> Optional["navis.TreeNeuron"]:
    """Fetch a neuron skeleton if missing (reusing any cached file).

    NeuPrint datasets use ``neuprint.fetch_skeleton``; FlyWire/CAVE datasets
    use ``CAVEDataFetcher.fetch_skeleton`` (which caches itself). With
    ``persist=True`` the fetched neuron is pickled to
    ``cache/{dataset}/skeletons/{body_id}.pkl`` so later calls reuse it; with
    ``persist=False`` (transient, profile-first comparisons) the neuron is
    returned in memory only and never written to the skeleton cache.
    """
    body_id = int(body_id)
    root = Path(project_root) if project_root else Path(__file__).parent.parent
    cache_dir = root / "cache" / _dataset_folder(dataset) / "skeletons"

    existing = _find_skeleton_file(dataset, body_id, project_root=str(root))
    if existing is not None:
        try:
            with open(existing, "rb") as f:
                neuron = pickle.load(f)
            if type(neuron).__name__ in ("TreeNeuron", "MeshNeuron"):
                return neuron
        except Exception:
            pass
        # Unsupported/corrupt pickle: drop it and fetch fresh.
        existing.unlink(missing_ok=True)

    dataset_l = dataset.lower()
    if any(k in dataset_l for k in ("flywire", "fafb", "banc")):
        neuron = _fetch_cave_skeleton(dataset, body_id)
    else:
        neuron = _fetch_neuprint_skeleton(dataset, body_id)

    if neuron is not None and persist:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_dir / f"{body_id}.pkl", "wb") as f:
            pickle.dump(neuron, f)
    return neuron


# =============================================================================
# MorphologyComparer (runner-facing)
# =============================================================================

class MorphologyComparer:
    """Query-vs-all morphological similarity search for one dataset.

    The runner script constructs this with all user parameters and calls
    ``find_similar()``; module-level defaults are used when arguments are
    omitted.
    """

    def __init__(
        self,
        query: Optional[Union[str, int]] = None,
        dataset: Optional[str] = None,
        level: str = "bodyid",
        method: str = "vector",
        metric: str = "cosine",
        top_n: int = 20,
        nblast_prefilter: int = 100,
        n_per_type: int = 5,
        candidate_source: str = "auto",
        fetch_top_n: int = 20,
        fetch_missing: Optional[int] = None,
        visualize_top_n: int = 0,
        visualize_by: str = "type",
        output_dir: Optional[str] = None,
        saveas: Optional[str] = None,
        verbose: bool = True,
        n_workers: int = 8,
        use_cache: bool = True,
        project_root: Optional[str] = None,
    ):
        self.query = query
        self.dataset = dataset
        self.level = str(level).lower()
        self.method = str(method).lower()
        self.metric = str(metric).lower()
        self.top_n = int(top_n)
        self.nblast_prefilter = int(nblast_prefilter)
        self.n_per_type = int(n_per_type)
        self.candidate_source = str(candidate_source).lower()
        # Backward-compatible alias: fetch_missing -> fetch_top_n.
        self.fetch_top_n = int(fetch_top_n if fetch_missing is None else fetch_missing)
        self.visualize_top_n = int(visualize_top_n)
        self.visualize_by = str(visualize_by).lower()
        self.verbose = verbose
        self.n_workers = max(1, int(n_workers))
        self.use_cache = use_cache
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent

        if self.level not in ("bodyid", "type"):
            raise ValueError(f"Invalid level: {self.level} (bodyid|type)")
        if self.method not in ("vector", "nblast"):
            raise ValueError(f"Invalid method: {self.method} (vector|nblast)")
        if self.metric not in ("cosine", "pearson"):
            raise ValueError(f"Invalid metric: {self.metric} (cosine|pearson)")
        if self.candidate_source not in ("auto", "cache", "profile"):
            raise ValueError(
                f"Invalid candidate_source: {self.candidate_source} (auto|cache|profile)"
            )
        if self.visualize_by not in ("type", "bodyid"):
            raise ValueError(
                f"Invalid visualize_by: {self.visualize_by} (type|bodyId)"
            )

        if output_dir is None:
            self.output_dir = str(self.project_root / "outputs" / "similar")
        else:
            self.output_dir = output_dir
        self.saveas = saveas or None

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------ candidate source
    def _is_flywire(self) -> bool:
        """FlyWire/CAVE datasets cache MeshNeurons and use cache-direct search."""
        d = (self.dataset or "").lower()
        return any(k in d for k in ("flywire", "fafb", "banc"))

    def _resolved_candidate_source(self) -> str:
        if self.candidate_source != "auto":
            return self.candidate_source
        # NeuPrint datasets have sparse skeleton caches: start from connection
        # similarity, then fetch skeletons for the top-N candidates only.
        # FlyWire has bulk mesh caches, so search the vector cache directly.
        return "profile" if not self._is_flywire() else "cache"

    # ------------------------------------------------------------------ query
    def _resolve_query(self) -> pd.DataFrame:
        if self.query is None:
            raise ValueError("No query neuron specified.")
        # getNeurons returns (neuron_df, roi_count_df, auto_name, criteria).
        # Some local loaders (FlyWire) only accept list queries, so retry with
        # a list-wrapped query when the scalar form fails or finds nothing.
        attempts = [self.query]
        if not isinstance(self.query, (list, tuple)):
            attempts.append([self.query])
        df = None
        for q in attempts:
            try:
                candidate = getNeurons(q, self.dataset, verbose=False, search_columns="auto")[0]
            except Exception:
                candidate = None
            if candidate is not None and len(candidate):
                df = candidate
                break
        if df is None or len(df) == 0:
            raise ValueError(f"Query '{self.query}' not found in {self.dataset}")
        out = df[["bodyId"]].copy()
        out["bodyId"] = out["bodyId"].astype(np.int64)
        out["type"] = df.get("type", pd.Series([""] * len(df))).fillna("").astype(str)
        out["instance"] = df.get("instance", pd.Series([""] * len(df))).fillna("").astype(str)
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------ run
    def find_similar(self) -> pd.DataFrame:
        """Run the similarity search and save results. Returns the results df."""
        if self.dataset is None:
            raise ValueError("No dataset specified.")
        source = self._resolved_candidate_source()
        self._log(f"Morphological similarity: query={self.query} dataset={self.dataset} "
                  f"method={self.method} level={self.level} metric={self.metric} "
                  f"candidate_source={source}")

        # NBLAST needs skeletons; FlyWire bulk caches hold meshes only.
        if self.method == "nblast" and self._is_flywire():
            raise ValueError(
                "NBLAST requires neuron skeletons, but this FlyWire dataset "
                "cache contains meshes. Use the 'vector' method instead."
            )

        query_df = self._resolve_query()
        cache = SkeletonVectorCache(
            self.dataset, project_root=str(self.project_root),
            n_workers=self.n_workers, verbose=self.verbose,
        )

        if source == "profile":
            results = self._profile_first_search(query_df, cache)
            if results.empty:
                # Connection-profile discovery can return nothing for neurons
                # with sparse connectivity; fall back to the vector cache when
                # one exists, otherwise surface a clear error.
                data = cache.load()
                if data is not None and len(data["bodyIds"]):
                    self._log(
                        "Profile-first found no candidates; falling back to "
                        "the vector cache."
                    )
                    if self.method == "vector":
                        results = self._vector_search(query_df, data)
                    else:
                        results = self._nblast_search(query_df, data)
                else:
                    raise ValueError(
                        "Connection-profile search found candidates for "
                        f"{self.query} in {self.dataset} but none could be "
                        "scored: no cached skeletons were usable, transient "
                        "fetches returned nothing, and no vector cache exists "
                        "to fall back to. Check network/token access and the "
                        "skeleton cache, or build the vector cache first."
                    )
        else:
            # Cache-direct search (FlyWire bulk caches and explicit choice).
            cache.ensure(fetch_missing=self.fetch_top_n if self.candidate_source != "auto" else 0)
            data = cache.load()
            if data is None or len(data["bodyIds"]) == 0:
                raise ValueError(
                    f"No vectorized neurons for {self.dataset}. Build the vector "
                    "cache first (Build Vector Cache button)."
                )
            if self.method == "vector":
                results = self._vector_search(query_df, data)
            else:
                results = self._nblast_search(query_df, data)

        if results.empty:
            self._log("No similar neurons found.")
        else:
            self._save_results(results, query_df)
            self._visualize_top_results(results)
        return results

    # ---------------------------------------------------- profile-first
    def _profile_candidates(self, query_df: pd.DataFrame) -> pd.DataFrame:
        """Rank candidates by connection-profile similarity (HomologFinder).

        Returns a DataFrame with target_bodyId and profile_similarity,
        sorted descending. Runs in a temp output dir (candidate discovery
        only) which is removed afterwards.
        """
        import shutil
        import tempfile
        from comparison.profile_comparator import HomologFinder

        query_label = str(self.query)
        tmp = tempfile.mkdtemp(prefix="drocat_profile_cand_")
        try:
            finder = HomologFinder(
                source=query_label,
                source_dataset=self.dataset,
                target_dataset=self.dataset,
                output_dir=tmp,
                top_n=max(self.top_n, self.fetch_top_n),
                min_shared_partners=2,
                vector_prune_fraction=0.2,
                similarity_metric="rank_union",
                vector_prefiltering=True,
                include_untyped_partners=True,
                use_cache=self.use_cache,
                min_synapse_threshold=3,
                morphological_enrichment=False,
                ensure_cache_complete=False,
                verbose=False,
            )
            res = finder.find_homologs_fast()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        if res is None or res.empty or "target_bodyId" not in res.columns:
            return pd.DataFrame()
        out = res[["target_bodyId"]].copy()
        out["target_bodyId"] = out["target_bodyId"].astype(np.int64)
        score_col = "rank_union" if "rank_union" in res.columns else (
            "combined" if "combined" in res.columns else "similarity"
        )
        out["profile_similarity"] = pd.to_numeric(
            res.get(score_col, np.nan), errors="coerce"
        )
        # A multi-source query can return the same candidate twice; keep the
        # best profile score per bodyId.
        return (out
                .sort_values("profile_similarity", ascending=False)
                .drop_duplicates("target_bodyId")
                .reset_index(drop=True))

    def _profile_first_search(self, query_df: pd.DataFrame, cache: SkeletonVectorCache) -> pd.DataFrame:
        """Connection-similarity first, then morphology on the top-N fetched
        skeletons. Bounds the skeleton fetch to ``fetch_top_n`` neurons.

        Fetched skeletons are TRANSIENT: they are used for the current
        comparison only and never written to the skeleton cache (the vector
        cache is the persistent artifact). Already-cached skeletons are still
        reused.
        """
        self._log("Profile-first search: running connection-profile candidate discovery...")
        candidates = self._profile_candidates(query_df)
        if candidates.empty:
            self._log("Connection-profile search returned no candidates.")
            return pd.DataFrame()
        candidates = candidates.head(max(1, self.fetch_top_n))
        cand_ids = [int(b) for b in candidates["target_bodyId"]]

        def _load_missing(ids: List[int]) -> Dict[int, "navis.TreeNeuron"]:
            """Fetch skeletons missing from the cache, kept in memory only."""
            loaded: Dict[int, navis.TreeNeuron] = {}
            for bid in ids:
                if _find_skeleton_file(self.dataset, bid, project_root=str(self.project_root)) is not None:
                    continue
                nrn = fetch_skeleton_on_demand(
                    self.dataset, bid, project_root=str(self.project_root), persist=False
                )
                if nrn is not None:
                    loaded[bid] = nrn
            return loaded

        # Query skeleton is always ensured (fetched transiently if missing).
        query_ids = [int(b) for b in query_df["bodyId"].tolist()]
        query_neurons = _load_missing(query_ids)

        # Fetch skeletons for the top-N candidates (bounded, transient).
        cand_neurons = _load_missing(cand_ids)
        self._log(f"Profile-first: {len(cand_ids)} candidates, "
                  f"{len(cand_neurons)} skeletons fetched (transient)")

        # Vectors for query + candidates: cached vectors first, then any
        # in-memory fetched neurons.
        X_q, mask_q = cache.vectors_for(query_ids, compute_missing=True)
        for i, bid in enumerate(query_ids):
            if not mask_q[i] and bid in query_neurons:
                _, vec = vectorize_neuron(query_neurons[bid])
                X_q[i], mask_q[i] = vec, True
        X_c, mask_c = cache.vectors_for(cand_ids, compute_missing=True)
        for i, bid in enumerate(cand_ids):
            if not mask_c[i] and bid in cand_neurons:
                _, vec = vectorize_neuron(cand_neurons[bid])
                X_c[i], mask_c[i] = vec, True
        if not mask_q.any():
            raise ValueError("Could not vectorize the query neuron.")
        q_vec = X_q[mask_q].mean(axis=0)
        keep = mask_c
        scores = np.full(len(cand_ids), np.nan)
        if keep.any():
            scores[keep] = similarity_matrix(q_vec, X_c[keep], self.metric)

        query_type = query_df["type"].iloc[0] if len(query_df) else ""
        # Type lookup: vector cache first, then the neuron table / index (so
        # datasets without a vector cache still get typed results). The
        # intra-type similarity comes from the vector cache when present,
        # otherwise from the query members' own vectors (the query type's
        # members are exactly what a type query resolves).
        cache_data = cache.load()
        id_to_type, _ = _load_neuron_type_map(self.dataset, str(self.project_root))
        if cache_data is not None:
            id_to_type = {int(b): t for b, t in zip(cache_data["bodyIds"], cache_data["types"])}
        intra = float("nan")
        if cache_data is not None and len(cache_data["bodyIds"]):
            intra = self._intra_type_similarity(
                query_type, cache_data["bodyIds"], cache_data["types"],
                cache_data["X"], self.metric,
            )
        elif query_type and mask_q.any():
            ok = np.where(mask_q)[0]
            intra = self._intra_type_similarity(
                query_type,
                np.array([query_ids[i] for i in ok], dtype=np.int64),
                [str(query_df["type"].iloc[i]) if i < len(query_df) else ""
                 for i in ok],
                X_q[ok], self.metric,
            )

        rows = []
        for i, bid in enumerate(cand_ids):
            if not keep[i]:
                continue
            t = id_to_type.get(bid, "")
            rows.append({
                "source_bodyId": query_ids[0],
                "source_type": query_type,
                "target_bodyId": bid,
                "target_type": t,
                "target_instance": "",
                "profile_similarity": float(candidates.loc[i, "profile_similarity"])
                if np.isfinite(candidates.loc[i, "profile_similarity"]) else np.nan,
                "similarity": float(scores[i]),
                "is_same_type": t == query_type if t else False,
                "intra_type_similarity": intra,
                "method": self.method,
                "metric": self.metric,
                "candidate_source": "profile",
            })
        if not rows:
            return pd.DataFrame()

        # Type-level: aggregate candidate rows by type (the query type itself
        # is included as the intra-type reference row; injected from the
        # vector cache when no query-type candidate reached the top-N).
        if self.level == "type":
            import collections
            agg: Dict[str, List[float]] = collections.defaultdict(list)
            prof: Dict[str, List[float]] = collections.defaultdict(list)
            for r in rows:
                if r["target_type"]:
                    agg[r["target_type"]].append(r["similarity"])
                    if np.isfinite(r["profile_similarity"]):
                        prof[r["target_type"]].append(r["profile_similarity"])
            agg_rows = []
            for t, vals in agg.items():
                agg_rows.append({
                    "target_type": t,
                    "similarity": float(np.mean(vals)),
                    "n_bodyids": len(vals),
                    "profile_similarity": float(np.mean(prof[t])) if prof[t] else np.nan,
                    "is_intra_type": t == query_type,
                    "intra_type_similarity": intra if t == query_type else float("nan"),
                    "method": self.method,
                    "metric": self.metric,
                    "candidate_source": "profile",
                })
            if query_type and query_type not in agg and np.isfinite(intra):
                if cache_data is not None:
                    n_members = int(sum(1 for t in cache_data["types"] if t == query_type))
                else:
                    n_members = int(sum(1 for t in id_to_type.values() if t == query_type))
                agg_rows.append({
                    "target_type": query_type,
                    "similarity": intra,
                    "n_bodyids": n_members,
                    "profile_similarity": np.nan,
                    "is_intra_type": True,
                    "intra_type_similarity": intra,
                    "method": self.method,
                    "metric": self.metric,
                    "candidate_source": "profile",
                })
            agg_rows = sorted(agg_rows, key=lambda r: (not r["is_intra_type"], -r["similarity"], r["target_type"]))
            results = pd.DataFrame(agg_rows).head(self.top_n).reset_index(drop=True)
            results.insert(0, "rank", np.arange(1, len(results) + 1))
            return results

        rows = sorted(rows, key=lambda r: (-r["similarity"], r["target_bodyId"]))
        results = pd.DataFrame(rows).head(self.top_n).reset_index(drop=True)
        results.insert(0, "rank", np.arange(1, len(results) + 1))

        # NBLAST refinement over the fetched candidate skeletons (transient).
        if self.method == "nblast":
            neurons = dict(query_neurons)
            neurons.update(cand_neurons)
            results = self._nblast_refine(results, query_df, cache, neurons)
        return results

    def _nblast_refine(self, results: pd.DataFrame, query_df: pd.DataFrame,
                       cache: SkeletonVectorCache,
                       neurons: Optional[Dict[int, "navis.TreeNeuron"]] = None) -> pd.DataFrame:
        """Replace vector scores with NBLAST scores for the fetched candidates."""
        cand_ids = [int(b) for b in results["target_bodyId"].tolist()]
        query_dp = self._dotprops_for_ids(
            [int(b) for b in query_df["bodyId"].tolist()], neurons=neurons
        )
        if not query_dp:
            self._log("NBLAST: query dotprops unavailable; keeping vector scores.")
            return results
        cand_dp = self._dotprops_for_ids(cand_ids, neurons=neurons)
        cand_dp = {b: dp for b, dp in cand_dp.items() if dp is not None}
        if not cand_dp:
            self._log("NBLAST: no candidate dotprops; keeping vector scores.")
            return results
        n_cores = min(self.n_workers, max(1, len(cand_dp)))
        nblast_scores: Dict[int, float] = {}
        for q_bid, q_dp in query_dp.items():
            targets = list(cand_dp.keys())
            mat = navis.nblast(
                q_dp, [cand_dp[t] for t in targets], normalized=True,
                n_cores=n_cores, progress=False,
            )
            row = mat.iloc[0]
            for j, t in enumerate(targets):
                val = float(row.iloc[j])
                nblast_scores[t] = max(nblast_scores.get(t, -np.inf), val)
        results = results.copy()
        results["similarity"] = results["target_bodyId"].map(
            lambda b: nblast_scores.get(int(b), np.nan)
        )
        results = results.sort_values(
            ["similarity", "target_bodyId"], ascending=[False, True]
        ).reset_index(drop=True)
        results["rank"] = np.arange(1, len(results) + 1)
        return results

    # ------------------------------------------------------------------ vector
    def _intra_type_similarity(self, type_name: str, body_ids: np.ndarray,
                              types: List[str], X: np.ndarray,
                              metric: str) -> float:
        """Mean pairwise similarity among a type's members (vector-based).

        1.0 for a single member (trivially identical); NaN when the type has
        no members in the population."""
        if not type_name:
            return float("nan")
        idxs = [i for i, t in enumerate(types) if t == type_name]
        n = len(idxs)
        if n == 0:
            return float("nan")
        if n == 1:
            return 1.0
        sub = X[idxs]
        # Pairwise similarity among members: similarity_matrix takes a single
        # query vector against matrix rows, so evaluate row by row.
        pair = np.empty((n, n))
        for i in range(n):
            pair[i] = similarity_matrix(sub[i], sub, metric)
        total = float(pair.sum()) - n  # drop the diagonal (self = 1)
        return total / (n * (n - 1))

    def _vector_search(self, query_df: pd.DataFrame, data: dict) -> pd.DataFrame:
        body_ids = data["bodyIds"]
        types = data["types"]
        X = data["X"]

        if self.level == "type":
            # Query type vector = mean of the query's member vectors.
            q_mask = np.isin(body_ids, query_df["bodyId"].to_numpy())
            q_vec = X[q_mask].mean(axis=0) if q_mask.any() else None
            if q_vec is None:
                return pd.DataFrame()
            q_type = query_df["type"].iloc[0] if len(query_df) else ""
            scores = similarity_matrix(q_vec, X, self.metric)
            return self._aggregate_by_type(
                body_ids, types, scores, query_type=q_type,
                metric=self.metric, X=X,
            )
        else:
            rows = []
            for _, qrow in query_df.iterrows():
                q_vec = self._vector_for_body_id(int(qrow["bodyId"]), body_ids, X)
                if q_vec is None:
                    continue
                scores = similarity_matrix(q_vec, X, self.metric)
                intra = self._intra_type_similarity(
                    qrow["type"], body_ids, types, X, self.metric
                )
                for i, bid in enumerate(body_ids):
                    if int(bid) == int(qrow["bodyId"]):
                        continue
                    rows.append({
                        "source_bodyId": int(qrow["bodyId"]),
                        "source_type": qrow["type"],
                        "target_bodyId": int(bid),
                        "target_type": types[i],
                        "target_instance": data["instances"][i],
                        "similarity": float(scores[i]),
                        "is_same_type": types[i] == qrow["type"],
                        "intra_type_similarity": intra,
                        "method": self.method,
                        "metric": self.metric,
                    })
            rows = sorted(rows, key=lambda r: (-r["similarity"], r["target_bodyId"]))
            results = pd.DataFrame(rows).head(self.top_n).reset_index(drop=True)
            results.insert(0, "rank", np.arange(1, len(results) + 1))
            return results

    def _vector_for_body_id(self, bid: int, body_ids: np.ndarray, X: np.ndarray) -> Optional[np.ndarray]:
        idx = np.where(body_ids == bid)[0]
        if not len(idx):
            return None
        return X[idx[0]]

    def _aggregate_by_type(self, body_ids: np.ndarray, types: List[str], scores: np.ndarray,
                           query_type: str = "", metric: str = "cosine",
                           X: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Aggregate bodyId scores to type level (mean of member scores).

        The query type itself is included as the first row (is_intra_type=True)
        so intra-type similarity data is part of the results. ``X``/``metric``
        are used for the pairwise intra-type similarity column."""
        import collections
        agg: Dict[str, List[float]] = collections.defaultdict(list)
        for i, t in enumerate(types):
            if not t:
                continue
            agg[t].append(float(scores[i]))
        rows = []
        for t, vals in agg.items():
            intra = float("nan")
            if X is not None and t == query_type:
                intra = self._intra_type_similarity(t, body_ids, types, X, metric)
            rows.append({
                "target_type": t,
                "similarity": float(np.mean(vals)),
                "n_bodyids": len(vals),
                "is_intra_type": t == query_type,
                "intra_type_similarity": intra,
                "method": self.method,
                "metric": self.metric,
            })
        # The intra-type (query-type) row always ranks first as the reference.
        rows = sorted(rows, key=lambda r: (not r["is_intra_type"], -r["similarity"], r["target_type"]))
        results = pd.DataFrame(rows).head(self.top_n).reset_index(drop=True)
        results.insert(0, "rank", np.arange(1, len(results) + 1))
        return results

    # ------------------------------------------------------------------ nblast
    def _nblast_search(self, query_df: pd.DataFrame, data: dict) -> pd.DataFrame:
        body_ids = data["bodyIds"]
        types = data["types"]
        X = data["X"]

        # Vector prefilter: top-K candidate bodyIds by cosine, excluding the
        # query itself. Same-type members stay in (they provide the intra-type
        # reference data at the type level).
        query_ids = set(int(b) for b in query_df["bodyId"].tolist())
        candidate_mask = ~np.isin(body_ids, list(query_ids))

        scores = np.full(len(body_ids), -np.inf)
        for _, qrow in query_df.iterrows():
            q_vec = self._vector_for_body_id(int(qrow["bodyId"]), body_ids, X)
            if q_vec is None:
                continue
            s = similarity_matrix(q_vec, X, "cosine")
            scores = np.maximum(scores, s)
        prefilter_idx = np.where(candidate_mask)[0]
        prefilter_idx = prefilter_idx[np.argsort(-scores[prefilter_idx])][: self.nblast_prefilter]

        if not len(prefilter_idx):
            self._log("NBLAST: no candidates survived the vector prefilter.")
            return pd.DataFrame()

        # Build dotprops for query + candidates (in microns; NEVER cached).
        query_dp = self._dotprops_for_ids(list(query_ids))
        if not query_dp:
            raise ValueError("NBLAST: could not build dotprops for the query neuron(s).")
        cand_ids = [int(body_ids[i]) for i in prefilter_idx]
        cand_dp = self._dotprops_for_ids(cand_ids)
        cand_dp = {bid: dp for bid, dp in cand_dp.items() if dp is not None}

        n_cores = min(self.n_workers, max(1, len(cand_dp)))
        self._log(f"NBLAST: {len(query_dp)} query x {len(cand_dp)} candidates "
                  f"({self.nblast_prefilter} prefiltered), {n_cores} cores")
        nblast_scores: Dict[int, float] = {}
        for q_bid, q_dp in query_dp.items():
            targets = list(cand_dp.keys())
            mat = navis.nblast(
                q_dp, [cand_dp[t] for t in targets], normalized=True,
                n_cores=n_cores, progress=False,
            )
            row = mat.iloc[0]
            for j, t in enumerate(targets):
                val = float(row.iloc[j])
                nblast_scores[t] = max(nblast_scores.get(t, -np.inf), val)

        # Rank candidates by NBLAST score.
        ranked = sorted(nblast_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        id_to_type = {int(b): t for b, t in zip(body_ids, types)}
        id_to_inst = {int(b): i for b, i in zip(body_ids, data["instances"])}

        query_type = query_df["type"].iloc[0] if len(query_df) else ""
        intra = self._intra_type_similarity(query_type, body_ids, types, X, "cosine")

        if self.level == "type":
            import collections
            agg: Dict[str, List[float]] = collections.defaultdict(list)
            for bid, score in ranked:
                t = id_to_type.get(bid, "")
                if t:
                    agg[t].append(score)
            rows = []
            for t, vals in agg.items():
                # Cap per-type pairs to n_per_type for the mean.
                vals = sorted(vals, reverse=True)[: self.n_per_type]
                rows.append({
                    "target_type": t,
                    "similarity": float(np.mean(vals)),
                    "n_bodyids": len(vals),
                    "is_intra_type": t == query_type,
                    "intra_type_similarity": intra if t == query_type else float("nan"),
                    "method": self.method,
                    "metric": "nblast",
                })
            # All query-type members are the query itself and cannot be
            # candidates; inject the intra-type reference row from the vector
            # statistics so type queries always carry the intra data.
            if query_type and query_type not in agg and np.isfinite(intra):
                n_members = int(sum(1 for t in types if t == query_type))
                rows.append({
                    "target_type": query_type,
                    "similarity": intra,
                    "n_bodyids": n_members,
                    "is_intra_type": True,
                    "intra_type_similarity": intra,
                    "method": self.method,
                    "metric": "nblast",
                })
            rows = sorted(rows, key=lambda r: (not r["is_intra_type"], -r["similarity"], r["target_type"]))
            results = pd.DataFrame(rows).head(self.top_n).reset_index(drop=True)
            results.insert(0, "rank", np.arange(1, len(results) + 1))
            return results

        rows = []
        for bid, score in ranked[: self.top_n]:
            t = id_to_type.get(bid, "")
            rows.append({
                "source_bodyId": int(query_df["bodyId"].iloc[0]),
                "source_type": query_type,
                "target_bodyId": bid,
                "target_type": t,
                "target_instance": id_to_inst.get(bid, ""),
                "similarity": float(score),
                "is_same_type": t == query_type,
                "intra_type_similarity": intra,
                "method": self.method,
                "metric": "nblast",
            })
        results = pd.DataFrame(rows).reset_index(drop=True)
        results.insert(0, "rank", np.arange(1, len(results) + 1))
        return results

    def _dotprops_for_ids(self, body_ids: List[int],
                          neurons: Optional[Dict[int, "navis.TreeNeuron"]] = None
                          ) -> Dict[int, Optional["navis.core.dotprop.Dotprops"]]:
        """Load skeletons and build dotprops in microns (never persisted).

        ``neurons`` supplies transient in-memory skeletons (profile-first
        fetches) so they are not re-fetched; anything missing is fetched
        transiently (persist=False)."""
        out: Dict[int, Optional[navis.core.dotprop.Dotprops]] = {}
        for bid in body_ids:
            nrn = (neurons or {}).get(bid)
            if nrn is None:
                pkl = _find_skeleton_file(self.dataset, bid, project_root=str(self.project_root))
                if pkl is None:
                    nrn = fetch_skeleton_on_demand(
                        self.dataset, bid, project_root=str(self.project_root), persist=False
                    )
                else:
                    with open(pkl, "rb") as f:
                        nrn = pickle.load(f)
            if nrn is None:
                out[bid] = None
                continue
            try:
                nrn_um = nrn / 1000.0  # nanometres -> microns (NBLAST requirement)
                out[bid] = navis.make_dotprops(nrn_um, k=20)
            except Exception:
                out[bid] = None
        return out

    # ------------------------------------------------------------------ save
    def _save_results(self, results: pd.DataFrame, query_df: pd.DataFrame):
        query_label = str(self.query)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = self.saveas or f"findsimilar_{_dataset_folder(self.dataset)}_{query_label[:40]}_{timestamp}"
        run_dir = Path(self.output_dir) / name
        run_dir.mkdir(parents=True, exist_ok=True)

        results.to_csv(run_dir / "results.csv", index=False)
        params = {
            "query": str(self.query),
            "dataset": self.dataset,
            "level": self.level,
            "method": self.method,
            "metric": self.metric,
            "top_n": self.top_n,
            "nblast_prefilter": self.nblast_prefilter,
            "n_per_type": self.n_per_type,
            "fetch_top_n": self.fetch_top_n,
            "visualize_top_n": self.visualize_top_n,
            "visualize_by": self.visualize_by,
            "query_bodyIds": [int(b) for b in query_df["bodyId"].tolist()],
            "output_folder": str(run_dir),
        }
        readme = [f"DROCAT Find Similar Neurons (morphological)",
                  f"Generated: {datetime.now().isoformat(timespec='seconds')}",
                  ""]
        for k, v in params.items():
            readme.append(f"{k}: {v}")
        (run_dir / "README.txt").write_text("\n".join(readme))
        self._log(f"Results saved to: {run_dir}")
        self.output_folder = str(run_dir)

    # ------------------------------------------------------------------ viz
    def _visualize_top_results(self, results: pd.DataFrame):
        """Render the 3D skeletons of the top-N found types/bodyIds (NB-style).

        Enabled when ``visualize_top_n > 0``. With ``visualize_by='type'``
        (default) each of the top-N distinct result types becomes one layer
        containing its member bodyIds (the result rows for bodyId-level
        searches, or the vector-cache members capped at ``n_per_type`` for
        type-level searches); with ``'bodyId'`` each top result row is one
        layer. The intra-type reference row is never rendered. Output goes to
        the same run folder (plot3d_{dataset}/ subfolder); a visualization
        failure is logged but never fails the similarity search.
        """
        if self.visualize_top_n <= 0 or results is None or results.empty:
            return
        VisualizeSkeleton = _import_visualizer()
        if VisualizeSkeleton is None:
            self._log("Visualization skipped: visualize_skeleton not available.")
            return

        work = results
        if "is_intra_type" in work.columns:
            work = work[work["is_intra_type"] != True]  # noqa: E712
        if work.empty:
            self._log("Visualization skipped: no inter-type results to show.")
            return

        layers: List[List[int]] = []
        names: List[str] = []
        if self.visualize_by == "type":
            seen: set = set()
            rank = 0
            for _, row in work.iterrows():
                t = str(row.get("target_type", "") or "")
                if not t or t in seen:
                    continue
                if self.level == "type" or "target_bodyId" not in work.columns:
                    # Type-level results carry no bodyIds: resolve the type's
                    # members from the vector cache (bounded to n_per_type).
                    members = self._type_members_from_cache(t)
                else:
                    members = [int(b) for b in work.loc[
                        work["target_type"] == t, "target_bodyId"
                    ].tolist()]
                if not members:
                    continue
                seen.add(t)
                rank += 1
                layers.append(members)
                names.append(f"r{rank}_{t}_x{len(members)}")
                if rank >= self.visualize_top_n:
                    break
        else:
            for rank, (_, row) in enumerate(work.head(self.visualize_top_n).iterrows(), start=1):
                bid = int(row.get("target_bodyId"))
                t = str(row.get("target_type", "") or "")
                layers.append([bid])
                names.append(f"r{rank}_{t or f'unknown_{bid}'}_{bid}")

        if not layers:
            self._log("Visualization skipped: no renderable layers.")
            return

        run_dir = Path(getattr(self, "output_folder", "") or self.output_dir)
        try:
            vs = VisualizeSkeleton(
                dataset=self.dataset,
                output_dir=str(run_dir),
                neuron_layers=layers,
                custom_layer_names=names,
                saveas=_dataset_folder(self.dataset),
                include_timestamp=False,
                skip_synapse=True,
                skeleton_mode="tube",
                legend_mode="layer" if self.visualize_by == "type" else "single",
                brain_mesh="template",
                export_views=False,
                show_fig=False,
                cache_neurons=True,
                verbose="simple",
            )
            vs.plot_neurons()
            viz_dir = run_dir / f"plot3d_{_dataset_folder(self.dataset)}"
            self._log(f"3D visualization saved to: {viz_dir}")
        except Exception as ex:
            self._log(f"3D visualization failed (search results kept): {ex}")

    def _type_members_from_cache(self, type_name: str) -> List[int]:
        """Member bodyIds of a type from the vector cache (capped to
        n_per_type so type-level renders stay bounded); falls back to the
        neuron table / index when the dataset has no vector cache yet."""
        try:
            data = SkeletonVectorCache(
                self.dataset, project_root=str(self.project_root), verbose=False
            ).load()
            members: List[int] = []
            if data is not None:
                members = [int(b) for b, t in zip(data["bodyIds"], data["types"])
                           if t == type_name]
            if not members:
                type_map, _ = _load_neuron_type_map(
                    self.dataset, str(self.project_root)
                )
                members = [int(b) for b, t in type_map.items() if t == type_name]
            return members[: max(1, self.n_per_type)]
        except Exception:
            return []


# =============================================================================
# Homolog results enrichment (post-search, vector-based)
# =============================================================================

def enrich_homolog_results(
    results_df: pd.DataFrame,
    source_dataset: str,
    target_dataset: str,
    project_root: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Attach vector-based morphological similarity to homolog results.

    Runs only on the final ranked result rows (never during candidate search
    or ranking). Adds ``morph_cosine`` and ``morph_pearson`` columns; NaN
    where either side has no cached/available skeleton. No rows are dropped
    or re-ranked, and no server fetching happens.
    """
    if results_df is None or results_df.empty:
        return results_df
    needed = {"source_bodyId", "target_bodyId"}
    if not needed.issubset(results_df.columns):
        return results_df

    def _vectors(dataset: str, bids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        cache = SkeletonVectorCache(
            dataset, project_root=project_root, verbose=verbose
        )
        return cache.vectors_for(bids, compute_missing=True)

    src_ids = [int(b) for b in results_df["source_bodyId"].tolist()]
    tgt_ids = [int(b) for b in results_df["target_bodyId"].tolist()]
    try:
        src_X, src_ok = _vectors(source_dataset, src_ids)
        tgt_X, tgt_ok = _vectors(target_dataset, tgt_ids)
    except Exception:
        if verbose:
            print("[morphology] Enrichment skipped (vector computation failed).")
        results_df["morph_cosine"] = np.nan
        results_df["morph_pearson"] = np.nan
        return results_df

    cos = np.full(len(results_df), np.nan)
    pears = np.full(len(results_df), np.nan)
    for i in range(len(results_df)):
        if not (src_ok[i] and tgt_ok[i]):
            continue
        q = src_X[i]
        t = tgt_X[i]
        cos[i] = float(similarity_matrix(q, t.reshape(1, -1), "cosine")[0])
        pears[i] = float(similarity_matrix(q, t.reshape(1, -1), "pearson")[0])

    results_df = results_df.copy()
    results_df["morph_cosine"] = cos
    results_df["morph_pearson"] = pears
    return results_df
