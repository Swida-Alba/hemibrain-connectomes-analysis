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

try:
    from .visualization_options import default_analysis_skeleton_mesh_simplification
except ImportError:
    from visualization_options import default_analysis_skeleton_mesh_simplification

try:
    from .utils.flywire_readiness import require_flywire_skeleton_access
except ImportError:
    from utils.flywire_readiness import require_flywire_skeleton_access

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

# Step-progress totals reported to the web UI during a similarity run
# (see the [DROCAT][progress] event protocol in ui/components/output_panel.py).
PROFILE_FIRST_TOTAL_STEPS = 6   # connectivity-first (NeuPrint) pipeline
CACHE_DIRECT_TOTAL_STEPS = 4    # vector-cache-direct (FlyWire) pipeline

# Maximum number of cached skeletons sampled for population standardization
# statistics when a dataset has no vector cache (see ``population_stats``).
POPULATION_STATS_SAMPLE = 3000

# A dataset whose skeleton cache holds fewer than this many neurons cannot
# estimate stable population statistics on its own; ``population_stats``
# then borrows them from a version sibling (e.g. male-cns:v1.0 <- v0.9).
MIN_POPULATION_STATS_SKELETONS = 300

# Vectorization levels ("basis"). Every vector cache holds ONE level; the
# basis is decided by the M2.1 level sweep (raw won: simplification adds no
# discrimination). NeuPrint skeletons are cached on disk ONLY at the fixed
# 90%-simplified level (``SKELETON_CACHE_LEVEL``), raw skeletons are never
# persisted; raw vectors are persisted at fetch time instead.
VECTOR_BASIS_RAW = "raw"
VECTOR_BASIS_SIMP90 = "simp90"
SKELETON_CACHE_LEVEL = VECTOR_BASIS_SIMP90   # on-disk NeuPrint cache level
SKELETON_DOWNSAMPLE_FACTOR = 10             # navis.downsample_neuron factor
                                            # (keeps ~10% of nodes)


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


def _neuron_rep(neuron) -> str:
    """Representation of a neuron: 'skeleton' or 'mesh' ('' otherwise).

    Skeletons and meshes produce different feature semantics in the shared
    124-dim schema, so a comparison must never mix the two (nor two
    simplification levels of the same kind).
    """
    if hasattr(neuron, "nodes") and neuron.nodes is not None and len(neuron.nodes):
        return "skeleton"
    if hasattr(neuron, "vertices") and neuron.vertices is not None and len(neuron.vertices):
        return "mesh"
    return ""


def _vectorize_one_file(path: str) -> Optional[Tuple[int, List[float], List[float], str]]:
    """Module-level worker for parallel cache builds (picklable).

    Returns None for un-vectorizable pickles (corrupt or unexpected types)
    so a single bad file cannot break the whole build. The 4th element is
    the neuron representation ('skeleton' | 'mesh')."""
    try:
        with open(path, "rb") as f:
            neuron = pickle.load(f)
        morph, vector = vectorize_neuron(neuron)
        rep = _neuron_rep(neuron)
    except Exception:
        return None
    body_id = int(Path(path).stem)
    # Shape block (persistence / spatial histogram) = the tail after the
    # morphometric block.
    return (body_id, [morph[f] for f in MORPHOMETRIC_FEATURES],
            vector[len(MORPHOMETRIC_FEATURES):].tolist(), rep)


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

    index_path = root / "neuron_indexes" / folder / "neuron_index.parquet"
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


def _skeleton_folder_level(dataset: str,
                           project_root: Optional[str] = None) -> str:
    """Simplification level of a dataset's on-disk skeleton cache.

    Reads the ``skeletons/.level`` marker ("raw" | "simp90"). A missing
    marker means the cache predates the level marker and holds RAW
    skeletons, which is also the default for datasets that never simplify
    (FlyWire/BANC mesh caches).
    """
    root = Path(project_root) if project_root else Path(__file__).parent.parent
    marker = root / "cache" / _dataset_folder(dataset) / "skeletons" / ".level"
    try:
        v = marker.read_text().strip()
        if v in (VECTOR_BASIS_RAW, VECTOR_BASIS_SIMP90):
            return v
    except Exception:
        pass
    return VECTOR_BASIS_RAW


def _write_skeleton_level_marker(dataset: str,
                                 project_root: Optional[str] = None):
    """Write the ``skeletons/.level`` marker (= the simplified cache level).

    Idempotent: once a folder is marked simplified it stays so; the marker
    is never downgraded, so a partially-simplified migration cannot
    silently revert to raw.
    """
    root = Path(project_root) if project_root else Path(__file__).parent.parent
    marker = root / "cache" / _dataset_folder(dataset) / "skeletons" / ".level"
    try:
        if not marker.exists():
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(SKELETON_CACHE_LEVEL + "\n")
    except Exception:
        pass


def _downsample_for_cache(neuron) -> "navis.TreeNeuron":
    """Deterministic simplification for the on-disk NeuPrint skeleton cache.

    ``navis.downsample_neuron(factor=10)`` keeps ~10% of nodes while
    preserving root/leaves/branchpoints — the canonical "90% simplified"
    skeleton. Falls back to the original neuron when downsampling fails.
    """
    try:
        # navis' default soma detection flags every node with radius >= 1
        # (neuprint radii are in nm) as soma; a whole-neuron "soma" makes
        # every node a downsample fix point and freezes the skeleton at
        # full resolution. Treat a multi-node soma as no soma.
        soma = getattr(neuron, "soma", None)
        if soma is not None and hasattr(soma, "__len__") and len(soma) > 1:
            neuron = neuron.copy()
            neuron.soma = None
        return navis.downsample_neuron(neuron, downsampling_factor=SKELETON_DOWNSAMPLE_FACTOR)
    except Exception:
        return neuron


# Per-(dataset, root) memo for legacy vector caches without a ``rep`` column.
_REP_MEMO: Dict[Tuple[str, str], str] = {}


def _infer_dataset_rep(dataset: str, project_root: Optional[str] = None) -> str:
    """Representation of a dataset's cached skeletons ('skeleton'|'mesh').

    Legacy vector caches predate the ``rep`` column; the representation is
    inferred once per (dataset, project) from the first cached pickle.
    """
    root = Path(project_root) if project_root else Path(__file__).parent.parent
    key = (dataset, str(root))
    if key in _REP_MEMO:
        return _REP_MEMO[key]
    rep = ""
    try:
        files = SkeletonVectorCache(
            dataset, project_root=str(root), verbose=False
        )._discover_skeleton_files()
        if files:
            with open(files[0], "rb") as f:
                rep = _neuron_rep(pickle.load(f))
    except Exception:
        rep = ""
    _REP_MEMO[key] = rep
    return rep


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
    def _write_meta(self, stats: Dict[str, List[float]], n_rows: int,
                    rep: str = "", vector_basis: str = VECTOR_BASIS_RAW):
        meta = {
            "version": VECTOR_CACHE_VERSION,
            "dataset": self.dataset,
            "feature_columns": MORPHOMETRIC_FEATURES,
            "persistence_dim": PERSISTENCE_DIM,
            "n_rows": n_rows,
            "rep": rep,
            # Simplification level of the vectors in this cache ("raw" |
            # "simp90"); the cache holds ONE level and never mixes.
            "vector_basis": vector_basis,
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

        # The cache holds ONE vectorization level (its "basis"). On-disk
        # skeletons are vectorized only when their simplification level
        # matches that basis: post-cleanup NeuPrint caches hold simp90
        # files while the basis is raw, so those files are skipped (their
        # vectors come from the vector cache / raw fetches instead).
        basis = (self._load_meta() or {}).get("vector_basis") or VECTOR_BASIS_RAW
        folder_level = _skeleton_folder_level(self.dataset, str(self.project_root))

        # Candidate skeleton files not yet vectorized.
        files = self._discover_skeleton_files()
        files = [f for f in files if folder_level == basis]
        pending = [f for f in files if int(Path(f).stem) not in existing]

        # Optional on-demand fetch to extend coverage (cap applies).
        # fetch_skeleton_on_demand already vectorizes at fetch time (raw
        # basis) and persists the vector; the fetched row set is refreshed
        # below so those rows are not dropped by the merge.
        fetched_new = 0
        if fetch_missing and fetch_missing > 0:
            index_path = self.project_root / "neuron_indexes" / _dataset_folder(self.dataset) / "neuron_index.parquet"
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
                        fetched_new += 1
            if fetched_new:
                # Re-discover after the fetches: they wrote new skeleton
                # files (and, in the real pipeline, already appended the
                # raw vectors). Refresh the row set and the pending files
                # so neither the fetched vectors nor the on-disk files are
                # dropped by the merge below.
                try:
                    df_old = pd.read_parquet(self.parquet_path)
                    existing = {int(r["bodyId"]): r for r in df_old.to_dict("records")}
                except Exception:
                    pass
                files = self._discover_skeleton_files()
                files = [f for f in files if folder_level == basis]
                pending = [f for f in files if int(Path(f).stem) not in existing]

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

        # A cache must hold ONE representation (skeleton vs mesh) and one
        # simplification level: vector features differ between the two, so
        # rows of any other representation are skipped (never mixed into
        # comparisons). The majority representation of the pending set wins.
        ok_rows = [r for r in rows if r is not None]
        rep = ""
        if ok_rows:
            from collections import Counter
            rep = Counter(r[3] for r in ok_rows).most_common(1)[0][0]
            foreign = sum(1 for r in ok_rows if r[3] != rep)
            if foreign:
                self._log(
                    f"[SkeletonVectorCache] Skipping {foreign} pickles of a "
                    f"different representation ({'mesh' if rep == 'skeleton' else 'skeleton'})"
                )
            rows = [r if r is None or r[3] == rep else None for r in rows]

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
            bid, morph_vals, pv_vals, row_rep = row
            record = {"bodyId": bid, "rep": row_rep}
            for name, val in zip(MORPHOMETRIC_FEATURES, morph_vals):
                record[name] = float(val)
            for i, val in enumerate(pv_vals):
                record[f"pv_{i}"] = float(val)
            records.append(record)

        if not records:
            self._log("[SkeletonVectorCache] No skeletons available to vectorize.")
            self._write_meta({"mean": [], "std": []}, 0, rep=rep,
                             vector_basis=basis)
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
        self._write_meta({"mean": mean, "std": std}, len(df), rep=rep,
                         vector_basis=basis)

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
        """Load the cache: meta + raw df + standardized matrix + index arrays.

        ``rep`` carries each row's representation ('skeleton' | 'mesh');
        ``dataset_rep`` the cache's single representation (legacy caches
        without a ``rep`` column infer it from the first cached file).
        """
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
        reps = df.get("rep", pd.Series([""] * len(df))).fillna("").astype(str).tolist()
        if reps and reps[0]:
            from collections import Counter
            dataset_rep = Counter(reps).most_common(1)[0][0]
        else:
            dataset_rep = _infer_dataset_rep(self.dataset, self.project_root)
        return {
            "meta": meta,
            "df": df,
            "raw": raw,
            "X": X,
            "bodyIds": df["bodyId"].astype(np.int64).to_numpy(),
            "types": df.get("type", pd.Series([""] * len(df))).fillna("").astype(str).tolist(),
            "instances": df.get("instance", pd.Series([""] * len(df))).fillna("").astype(str).tolist(),
            "rep": reps,
            "dataset_rep": dataset_rep,
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

    # ------------------------------------------------------------ append
    def append_vectors(self, records: List[Tuple[int, np.ndarray, str]],
                       vector_basis: str = VECTOR_BASIS_RAW) -> int:
        """Persist freshly-computed vectors (raw feature rows) into the cache.

        Called when a vector was computed from a cached skeleton file or from
        an online-fetched skeleton that was NOT persisted: the VECTOR is
        stored so later queries reuse it without re-fetching or
        re-vectorizing, even though the original skeleton stays uncached.
        Rows are merged by bodyId (dedupe); the cache's standardization
        statistics (meta mean/std) are left untouched, so the standardized
        space stays consistent across appends. Rows of a representation
        different from the cache's are rejected (a cache holds ONE level),
        and rows whose ``vector_basis`` differs from the cache's basis are
        rejected too (a cache holds ONE simplification level). Returns the
        number of rows actually added.
        """
        if not records:
            return 0
        # Cross-process safety: UI runs execute in separate subprocesses, so
        # the read-modify-write is guarded with an advisory file lock
        # (POSIX; best-effort elsewhere).
        lock_fd = None
        lock_path = self.parquet_path.with_suffix(".parquet.lock")
        try:
            import fcntl
            lock_fd = open(lock_path, "w")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        except (ImportError, OSError):
            lock_fd = None
        try:
            type_map, instance_map = _load_neuron_type_map(
                self.dataset, str(self.project_root)
            )
            rows_new = []
            for bid, vec, rep in records:
                bid = int(bid)
                row = {"bodyId": bid, "rep": rep}
                for i, name in enumerate(MORPHOMETRIC_FEATURES):
                    row[name] = float(vec[i])
                for i in range(PERSISTENCE_DIM):
                    row[f"pv_{i}"] = float(vec[len(MORPHOMETRIC_FEATURES) + i])
                row["type"] = type_map.get(bid, "") if type_map else ""
                row["instance"] = (instance_map or {}).get(bid, "") if instance_map else ""
                rows_new.append(row)
            df_new = pd.DataFrame(rows_new)

            existing = self.load()
            if existing is not None:
                cache_rep = existing.get("dataset_rep", "")
                cache_basis = ((existing.get("meta") or {})
                               .get("vector_basis") or VECTOR_BASIS_RAW)
                if cache_basis != vector_basis:
                    # Different simplification level: never mix (a cache
                    # holds ONE basis).
                    return 0
                if cache_rep:
                    df_new = df_new[df_new["rep"] == cache_rep]
                if df_new.empty:
                    return 0
                old = existing["df"]
                keep_cols = [c for c in old.columns]
                df_new = df_new[[c for c in keep_cols if c in df_new.columns]]
                known = set(old["bodyId"].astype(np.int64))
                df_new = df_new[~df_new["bodyId"].astype(np.int64).isin(known)]
                if df_new.empty:
                    return 0
                df = pd.concat([old, df_new], ignore_index=True)
            else:
                # Creating the cache: keep it homogeneous (majority
                # representation of this batch) and record the basis.
                from collections import Counter
                canonical = (Counter(df_new["rep"].astype(str))
                             .most_common(1)[0][0] if len(df_new) else "")
                if canonical:
                    df_new = df_new[df_new["rep"].astype(str) == canonical]
                if df_new.empty:
                    return 0
                df = df_new

            self.morph_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.parquet_path, index=False)

            # The standardization stats (meta mean/std) stay untouched so the
            # standardized space of existing rows is preserved; only the row
            # count and bookkeeping are refreshed. A freshly-created cache
            # gets its own stats (like a full build).
            meta = self._load_meta() or {}
            if not meta.get("mean"):
                mat = self._raw_matrix(df)
                mean = mat.mean(axis=0).tolist()
                std = mat.std(axis=0).tolist()
                std = [s if s > 0 else 1.0 for s in std]
                meta["mean"] = mean
                meta["std"] = std
            meta["dataset"] = self.dataset
            meta["n_rows"] = len(df)
            meta["built_at"] = datetime.now().isoformat(timespec="seconds")
            if "rep" not in meta and len(df_new) and "rep" in df_new.columns:
                meta["rep"] = str(df_new["rep"].iloc[0])
            # Record the basis when creating the cache; existing caches keep
            # their own basis (enforced by the check above).
            if "vector_basis" not in meta:
                meta["vector_basis"] = vector_basis
            self.meta_path.write_text(json.dumps(meta, indent=2))
            return len(df_new)
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except Exception:
                    pass
                lock_fd.close()

    # ------------------------------------------------------------ vectors_for
    def vectors_for(self, body_ids: List[int], compute_missing: bool = True
                    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Return standardized vectors for bodyIds.

        Rows missing from the cache are computed on the fly when a skeleton
        file of the cache's representation AND simplification level exists
        (``compute_missing=True``); files of a DIFFERENT representation
        (e.g. skeletons beside bulk meshes) or of a different level than
        the cache's ``vector_basis`` are skipped so comparisons never mix
        levels. Otherwise they are NaN rows. Never fetches from the server
        and never forces a full cache build. Returns (vectors, mask, reps)
        where ``reps`` carries each row's representation ('skeleton' |
        'mesh' | '').
        """
        body_ids = [int(b) for b in body_ids]
        data = self.load()
        known: Dict[int, int] = {}
        X = np.zeros((0, VECTOR_DIM))
        dataset_rep = ""
        basis = VECTOR_BASIS_RAW
        if data is not None:
            known = {int(b): i for i, b in enumerate(data["bodyIds"])}
            X = data["X"]
            dataset_rep = data.get("dataset_rep", "")
            basis = ((data.get("meta") or {}).get("vector_basis")
                     or VECTOR_BASIS_RAW)

        result = np.full((len(body_ids), VECTOR_DIM), np.nan)
        reps = [""] * len(body_ids)
        computed: List[Tuple[int, np.ndarray, str]] = []
        for j, bid in enumerate(body_ids):
            if bid in known:
                result[j] = X[known[bid]]
                reps[j] = dataset_rep
                continue
            if compute_missing:
                # Level guard: on-disk skeletons are vectorized only when
                # their simplification level matches the cache's basis
                # (post-cleanup NeuPrint caches hold simp90 files while
                # the basis is raw -> never vectorized here).
                pkl = None
                if _skeleton_folder_level(self.dataset, str(self.project_root)) == basis:
                    pkl = _find_skeleton_file(
                        self.dataset, bid, project_root=str(self.project_root)
                    )
                if pkl is not None:
                    try:
                        with open(pkl, "rb") as f:
                            neuron = pickle.load(f)
                        row_rep = _neuron_rep(neuron)
                        if dataset_rep and row_rep != dataset_rep:
                            continue  # different representation: never mix
                        _, vec = vectorize_neuron(neuron)
                        result[j] = vec
                        reps[j] = row_rep
                        # Persist the vector: later queries reuse it without
                        # re-loading and re-vectorizing the skeleton file.
                        computed.append((bid, vec, row_rep))
                    except Exception:
                        result[j] = np.nan
        if computed:
            self.append_vectors(computed, vector_basis=basis)
        mask = ~np.isnan(result[:, 0])
        return result, mask, reps


# =============================================================================
# Population standardization statistics
# =============================================================================

def _datasets_share_population(dataset: str, other: str,
                               root: Path) -> bool:
    """True when at least 30% of the smaller neuron index is in the other.

    Guards the version-sibling statistics fallback: two versions of one
    reconstruction (e.g. male-cns v0.9/v1.0) share their neurons, so the
    older release's population stats are a valid baseline for the newer.
    """
    def _ids(ds: str):
        p = root / "neuron_indexes" / _dataset_folder(ds) / "neuron_index.parquet"
        if not p.exists():
            return None
        try:
            import polars as pl
            return set(pl.read_parquet(p, columns=["bodyId"])["bodyId"].to_list())
        except Exception:
            return None
    a, b = _ids(dataset), _ids(other)
    if not a or not b:
        return False
    smaller, bigger = (a, b) if len(a) <= len(b) else (b, a)
    return len(smaller & bigger) / len(smaller) >= 0.3


def _sibling_skeleton_dirs(dataset: str, root: Path) -> List[Path]:
    """Skeleton-cache directories of version siblings sharing the population.

    Two versions of one reconstruction (e.g. male-cns v0.9/v1.0) contain the
    same neurons, so the sibling's cached skeletons extend the sample used
    for population standardization statistics. The dataset-folder convention
    is inverted (``male-cns_v1_0`` -> ``male-cns:v1.0``) and every candidate
    is verified against ``_datasets_share_population``.
    """
    name = dataset.split(":")[0] if ":" in dataset else dataset
    if not name:
        return []
    dirs: List[Path] = []
    cache_root = root / "cache"
    if not cache_root.is_dir():
        return []
    for folder in sorted(cache_root.iterdir()):
        if not folder.is_dir() or not folder.name.startswith(name + "_"):
            continue
        parts = folder.name.split("_")
        if len(parts) < 2:
            continue
        sibling = f"{parts[0]}:{parts[1]}" + (
            f".{'.'.join(parts[2:])}" if len(parts) > 2 else ""
        )
        if sibling == dataset or not _datasets_share_population(dataset, sibling, root):
            continue
        skel = folder / "skeletons"
        if skel.is_dir():
            dirs.append(skel)
    return dirs


def population_stats(dataset: str, project_root: Optional[str] = None,
                     max_sample: int = POPULATION_STATS_SAMPLE
                     ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Population mean/std for a dataset's cached skeletons.

    Stable standardization statistics used when a dataset has no vector
    cache: pool-only statistics depend on the (connectivity-skewed) pool
    composition and distort the geometry between query and candidates. The
    stats are computed once from a bounded sample of cached skeletons and
    persisted under ``morphology/population_stats.json`` for reuse. A
    dataset with too few cached skeletons extends its sample with a version
    sibling's skeletons (same reconstruction, e.g. male-cns v1.0 <- v0.9).
    Returns (None, None) when no statistics can be estimated.
    """
    root = Path(project_root) if project_root else Path(__file__).parent.parent
    folder = _dataset_folder(dataset)
    cache_dir = root / "cache" / folder
    stats_file = cache_dir / "morphology" / "population_stats.json"
    if stats_file.exists():
        try:
            data = json.loads(stats_file.read_text())
            if (data.get("dataset") == dataset and data.get("dim") == VECTOR_DIM
                    and data.get("sample_cap") == max_sample
                    and int(data.get("n", 0)) >= MIN_POPULATION_STATS_SKELETONS):
                return (np.asarray(data["mean"], dtype=float),
                        np.asarray(data["std"], dtype=float))
        except Exception:
            pass

    vc = SkeletonVectorCache(dataset, project_root=str(root), verbose=False)
    files = vc._discover_skeleton_files()

    # Level guard: the statistics must match the vector basis (raw). Once
    # raw skeletons are replaced by the simplified cache (NeuPrint), the
    # on-disk sample can no longer be vectorized at the right level; fall
    # back to the vector cache's own raw meta stats, which were computed
    # from the same feature schema at fetch time.
    basis = (vc._load_meta() or {}).get("vector_basis") or VECTOR_BASIS_RAW
    if _skeleton_folder_level(dataset, str(root)) != basis:
        data = vc.load()
        if data is not None:
            meta = data.get("meta") or {}
            m = meta.get("mean")
            s = meta.get("std")
            if m is not None and s is not None:
                mm = np.asarray(m, dtype=float)
                ss = np.asarray(s, dtype=float)
                if mm.shape == (VECTOR_DIM,) and ss.shape == (VECTOR_DIM,):
                    return mm, ss
        return None, None

    # Too few cached skeletons for stable stats: sample from the version
    # sibling's cache instead — it contains the same neurons (shared
    # reconstruction, e.g. male-cns v1.0 <- v0.9), and the sparse local
    # cache may be morphologically skewed (e.g. one query's transient
    # fetches), which would bias the statistics. Only a LARGER sibling
    # cache is used (the sibling may itself be the sparse one).
    if len(files) < MIN_POPULATION_STATS_SKELETONS:
        sibling_files: List[str] = []
        for skel_dir in _sibling_skeleton_dirs(dataset, root):
            sf = sorted(str(p) for p in skel_dir.rglob("*.pkl"))
            if len(sf) > len(sibling_files):
                sibling_files = sf
        if len(sibling_files) > len(files):
            files = sibling_files

    if not files:
        return None, None
    if len(files) > max_sample:
        rng = np.random.default_rng(0)
        files = [files[i] for i in
                 rng.choice(len(files), max_sample, replace=False)]
    try:
        rows = vc._vectorize_parallel(files)
        if len(rows) != len(files):
            # A broken worker pool can silently drop rows; recompute
            # sequentially so the statistics stay deterministic.
            raise ValueError("parallel vectorization incomplete")
    except Exception:
        rows = [_vectorize_one_file(p) for p in files]
    # One representation per dataset: mixed skeleton/mesh samples would
    # bias the statistics (different feature semantics in one schema).
    ok_rows = [r for r in rows if r is not None]
    if ok_rows:
        from collections import Counter
        canonical = Counter(r[3] for r in ok_rows).most_common(1)[0][0]
        ok_rows = [r for r in ok_rows if r[3] == canonical]
    vecs = [np.concatenate([r[1], r[2]]) for r in ok_rows]
    if not vecs:
        return None, None
    mat = np.asarray(vecs, dtype=float)
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0)
    sd = np.where(sd <= 0, 1.0, sd)
    try:
        (cache_dir / "morphology").mkdir(parents=True, exist_ok=True)
        stats_file.write_text(json.dumps({
            "dataset": dataset,
            "dim": VECTOR_DIM,
            "n": len(vecs),
            "sample_cap": max_sample,
            "mean": mu.tolist(),
            "std": sd.tolist(),
        }))
    except Exception:
        pass
    return mu, sd


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
        nrn = navis.TreeNeuron(df)
        # navis' default soma detection flags every node with radius >= 1
        # (neuprint radii are in nm) as soma; a whole-neuron "soma" would
        # freeze the skeleton at full resolution during downsampling and
        # distort the soma_radius feature.
        soma = nrn.soma
        if soma is not None and hasattr(soma, "__len__") and len(soma) > 1:
            nrn.soma = None
        return nrn
    except Exception:
        return None


def _fetch_cave_skeleton(dataset: str, body_id: int):
    """Fetch one skeleton from a FlyWire/CAVE dataset."""
    from cave_data_fetcher import CAVEDataFetcher
    fetcher = CAVEDataFetcher(dataset=_dataset_folder(dataset), verbose=False)
    return fetcher.fetch_skeleton(body_id, use_cache=True)


def fetch_skeleton_on_demand(dataset: str, body_id: int,
                             project_root: Optional[str] = None,
                             persist: bool = True,
                             level: str = VECTOR_BASIS_RAW
                             ) -> Optional["navis.TreeNeuron"]:
    """Fetch a neuron skeleton if missing (reusing any cached file).

    NeuPrint datasets use ``neuprint.fetch_skeleton``; FlyWire/CAVE datasets
    use ``CAVEDataFetcher.fetch_skeleton`` (which caches itself). With
    ``persist=True`` the fetched neuron is simplified to the fixed cache
    level (``SKELETON_CACHE_LEVEL``, downsample factor 10) and pickled to
    ``cache/{dataset}/skeletons/{body_id}.pkl`` so later calls reuse it; with
    ``persist=False`` (transient, profile-first comparisons) the neuron is
    returned in memory only and never written to the skeleton cache.

    ``level`` selects the consumer's simplification level:

    - ``"raw"`` (default): returns the RAW skeleton. Raw is NEVER served
      from the disk cache (raw skeletons are not persisted); vectors come
      from the vector cache instead.
    - ``"simp90"``: hits the simplified cache file when present.

    Every online fetch vectorizes the RAW skeleton immediately and persists
    the vector (basis ``VECTOR_BASIS_RAW``), so later comparisons reuse it
    even though the raw skeleton is not stored on disk.
    """
    body_id = int(body_id)
    level = str(level).lower()
    if level not in (VECTOR_BASIS_RAW, VECTOR_BASIS_SIMP90):
        raise ValueError(f"Invalid level: {level} (raw|simp90)")
    root = Path(project_root) if project_root else Path(__file__).parent.parent
    cache_dir = root / "cache" / _dataset_folder(dataset) / "skeletons"

    # Cache hit only for simp90 consumers: raw skeletons are never on disk,
    # so a raw request always goes to the server (or the vector cache).
    if level == VECTOR_BASIS_SIMP90:
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

    if neuron is None:
        return None

    # Vectorize the RAW skeleton at fetch time and persist the vector
    # (basis raw): later comparisons reuse it without re-fetching or
    # re-vectorizing, even though the raw skeleton itself is not stored.
    # Best-effort: a vectorization failure must not break the fetch.
    try:
        _, vec = vectorize_neuron(neuron)
        SkeletonVectorCache(dataset, project_root=str(root), verbose=False).append_vectors(
            [(body_id, vec, _neuron_rep(neuron))], vector_basis=VECTOR_BASIS_RAW
        )
    except Exception:
        pass

    if persist:
        # Persist ONLY the simplified skeleton (raw never cached) and mark
        # the folder's level so level guards can enforce the invariant.
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_dir / f"{body_id}.pkl", "wb") as f:
            pickle.dump(_downsample_for_cache(neuron), f)
        _write_skeleton_level_marker(dataset, str(root))
        if level == VECTOR_BASIS_SIMP90:
            try:
                return _downsample_for_cache(neuron)
            except Exception:
                pass
    return neuron


def download_all_skeletons(dataset: str, project_root: Optional[str] = None,
                           max_workers: int = 8, limit: Optional[int] = None,
                           progress_callback=None, cancel_event=None,
                           verbose: bool = True) -> Dict[str, object]:
    """Download every missing skeleton of a dataset to the local cache.

    Mirrors the Settings-panel full dataset pull: iterates the neuron index
    (allneurons table, neuron index fallback), fetches only the neurons
    missing from ``cache/{dataset}/skeletons/`` (resumable — existing files
    are skipped), parallel over a thread pool, and persists each skeleton at
    the fixed simplified cache level (``persist=True``; raw is never
    persisted, raw vectors are appended at fetch time).
    ``progress_callback(current, total, info)`` and
    ``cancel_event`` (threading.Event) drive the UI; ``limit`` bounds the
    download (tests / smoke runs). Returns a summary dict.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    require_flywire_skeleton_access(
        dataset,
        project_root=project_root,
        log=print if verbose else (lambda _message: None),
    )

    root = Path(project_root) if project_root else Path(__file__).parent.parent
    folder = _dataset_folder(dataset)
    skeleton_dir = root / "cache" / folder / "skeletons"
    skeleton_dir.mkdir(parents=True, exist_ok=True)

    # Index of all bodyIds.
    index: List[int] = []
    csv_path = root / "datasets" / folder / f"{folder}_allneurons_neuron_df.csv"
    if csv_path.exists():
        try:
            import polars as pl
            index = pl.read_csv(csv_path, columns=["bodyId"])["bodyId"] \
                .cast(pl.Int64).to_list()
        except Exception:
            index = []
    if not index:
        index_path = root / "neuron_indexes" / folder / "neuron_index.parquet"
        if index_path.exists():
            try:
                index = pd.read_parquet(index_path, columns=["bodyId"])["bodyId"] \
                    .astype(np.int64).tolist()
            except Exception:
                index = []

    existing = {int(p.stem) for p in skeleton_dir.rglob("*.pkl")}
    missing = [int(b) for b in index if int(b) not in existing]
    if limit is not None:
        missing = missing[: int(limit)]
    total = len(missing)
    if total == 0:
        if verbose:
            print(f"[morphology] download_all_skeletons: "
                  f"{len(existing)} skeletons already cached; nothing to fetch.")
        return {"total": 0, "fetched": 0, "skipped_existing": len(existing),
                "cancelled": False, "errors": 0}

    if verbose:
        print(f"[morphology] download_all_skeletons: fetching {total} "
              f"skeletons ({dataset})...")
    if progress_callback:
        progress_callback(0, total, f"Fetching skeletons (0/{total})")

    fetched = 0
    errors = 0
    cancelled = False
    cancel_event = cancel_event or threading.Event()
    lock = threading.Lock()

    def _fetch_one(bid: int) -> bool:
        if cancel_event.is_set():
            return False
        try:
            nrn = fetch_skeleton_on_demand(
                dataset, bid, project_root=str(root), persist=True
            )
            ok = nrn is not None
        except Exception:
            ok = False
        with lock:
            nonlocal fetched, errors
            if ok:
                fetched += 1
            else:
                errors += 1
            if progress_callback:
                progress_callback(fetched + errors, total,
                                  f"Fetching skeletons ({fetched + errors}/{total})")
        return ok

    try:
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
            futures = [ex.submit(_fetch_one, bid) for bid in missing]
            for fut in as_completed(futures):
                if cancel_event.is_set():
                    cancelled = True
                    for f in futures:
                        f.cancel()
                    break
                fut.result()
    finally:
        if progress_callback:
            progress_callback(fetched + errors, total,
                              "Cancelled." if cancelled else "Finished.")

    summary = {
        "total": total,
        "fetched": fetched,
        "skipped_existing": len(existing),
        "cancelled": cancelled,
        "errors": errors,
    }
    if verbose:
        print(f"[morphology] download_all_skeletons: {fetched}/{total} fetched, "
              f"{errors} errors, cancelled={cancelled}")
    return summary


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
        level: str = "auto",
        method: str = "vector",
        metric: str = "cosine",
        top_n: int = 20,
        nblast_prefilter: int = 100,
        n_per_type: int = 5,
        candidate_source: str = "auto",
        fetch_top_n: int = 20,
        fetch_missing: Optional[int] = None,
        candidate_expansion: int = 3,
        min_weight: int = 3,
        min_shared_partners: int = 2,
        roi_filter: Optional[List[str]] = None,
        max_pool_per_type: int = 100,
        visualize_top_n: int = 0,
        visualize_by: str = "type",
        visualization_settings: Optional[Dict[str, object]] = None,
        output_dir: Optional[str] = None,
        saveas: Optional[str] = None,
        verbose: bool = True,
        n_workers: int = 8,
        use_cache: bool = True,
        cache_fetched_skeletons: bool = False,
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
        self.candidate_expansion = int(candidate_expansion)
        self.min_weight = int(min_weight)
        self.min_shared_partners = int(min_shared_partners)
        self.roi_filter = list(roi_filter) if roi_filter else None
        self.max_pool_per_type = int(max_pool_per_type)
        self.visualize_top_n = int(visualize_top_n)
        self.visualize_by = str(visualize_by).lower()
        self.visualization_settings = dict(visualization_settings or {})
        self.verbose = verbose
        self.n_workers = max(1, int(n_workers))
        self.use_cache = use_cache
        # Persist transiently-fetched skeletons (profile-first / NBLAST) to
        # the skeleton cache for reuse; off by default (memory-only fetches).
        self.cache_fetched_skeletons = bool(cache_fetched_skeletons)
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent

        if self.level not in ("auto", "bodyid", "type"):
            raise ValueError(f"Invalid level: {self.level} (auto|bodyid|type)")
        if self.method not in ("vector", "nblast"):
            raise ValueError(f"Invalid method: {self.method} (vector|nblast)")
        if self.metric not in ("cosine", "pearson"):
            raise ValueError(f"Invalid metric: {self.metric} (cosine|pearson)")
        if self.candidate_source not in ("auto", "cache", "profile"):
            raise ValueError(
                f"Invalid candidate_source: {self.candidate_source} (auto|cache|profile)"
            )
        if self.candidate_expansion < 1:
            raise ValueError(
                f"Invalid candidate_expansion: {self.candidate_expansion} (>= 1)"
            )
        if self.min_shared_partners < 1:
            raise ValueError(
                f"Invalid min_shared_partners: {self.min_shared_partners} (>= 1)"
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

    def _progress(self, step: int, total: int, label: str = ""):
        """Emit a structured step-progress event consumed by the web UI.

        The line is a control event (determinate bar + step label in the
        results panel), not log output, and is only emitted when verbose.
        """
        if self.verbose:
            print(f"[DROCAT][progress] {int(step)}/{int(total)} {label}".rstrip(),
                  flush=True)

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
    def _resolve_level(self, query_df: pd.DataFrame) -> str:
        """Resolve the result level for ``level='auto'``.

        A bodyId query (all-numeric input) yields bodyId-to-bodyId rows; a
        type query (type/pattern input resolving to one common type) yields
        type-to-type rows; mixed or multi-type lists fall back to bodyId."""
        if self.level != "auto":
            return self.level
        q = self.query
        if isinstance(q, (list, tuple)):
            all_digits = all(str(x).strip().isdigit() for x in q)
        else:
            all_digits = str(q).strip().isdigit()
        if all_digits:
            return "bodyid"
        types = {str(t) for t in query_df["type"].tolist() if str(t)}
        return "type" if len(types) == 1 else "bodyid"

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
        # The step scheme depends on the pipeline: connectivity-first
        # (NeuPrint) reports 6 steps, vector-cache-direct (FlyWire) 4.
        total_steps = (PROFILE_FIRST_TOTAL_STEPS if source == "profile"
                       else CACHE_DIRECT_TOTAL_STEPS)
        self._log(f"Morphological similarity: query={self.query} dataset={self.dataset} "
                  f"method={self.method} level={self.level} metric={self.metric} "
                  f"candidate_source={source}")

        # BANC has connectivity tables but no skeleton release.  FAFB may use
        # its local skeleton bundle or CAVE, but must not silently fall into a
        # failed remote fetch when neither source is configured.
        require_flywire_skeleton_access(
            self.dataset,
            project_root=self.project_root,
            log=self._log,
        )

        # NBLAST needs skeletons; FlyWire bulk caches hold meshes only.
        if self.method == "nblast" and self._is_flywire():
            raise ValueError(
                "NBLAST requires neuron skeletons, but this FlyWire dataset "
                "cache contains meshes. Use the 'vector' method instead."
            )

        self._progress(1, total_steps, "Resolving query neuron")
        query_df = self._resolve_query()
        if self.level == "auto":
            self.level = self._resolve_level(query_df)
            self._log(f"Level auto-resolved to: {self.level} "
                      f"({'type-to-type' if self.level == 'type' else 'bodyId-to-bodyId'})")
        cache = SkeletonVectorCache(
            self.dataset, project_root=str(self.project_root),
            n_workers=self.n_workers, verbose=self.verbose,
        )

        if source == "profile":
            bodyid_df, type_df = self._profile_first_search(query_df, cache)
            if bodyid_df.empty and type_df.empty:
                # Connection-profile discovery can return nothing for neurons
                # with sparse connectivity; fall back to the vector cache when
                # one exists, otherwise surface a clear error.
                data = cache.load()
                if data is not None and len(data["bodyIds"]):
                    self._log(
                        "Profile-first found no candidates; falling back to "
                        "the vector cache."
                    )
                    # The fallback switches the whole run to the 4-step
                    # cache-direct protocol; later events (including the
                    # final one) must keep that total so the bar never
                    # regresses or jumps between step schemes.
                    total_steps = CACHE_DIRECT_TOTAL_STEPS
                    self._progress(2, total_steps,
                                   "Loading vector cache (fallback)")
                    self._progress(3, CACHE_DIRECT_TOTAL_STEPS,
                                   self._scoring_step_label())
                    if self.method == "vector":
                        bodyid_df, type_df = self._vector_search(query_df, data)
                    else:
                        bodyid_df, type_df = self._nblast_search(query_df, data)
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
            # Skeletons are only ever fetched explicitly (Download All
            # Skeletons); the vector cache builds from what is local.
            self._progress(2, CACHE_DIRECT_TOTAL_STEPS, "Loading vector cache")
            cache.ensure(fetch_missing=0)
            data = cache.load()
            if data is None or len(data["bodyIds"]) == 0:
                raise ValueError(
                    f"No vectorized neurons for {self.dataset}. Build the vector "
                    "cache first (Build Vector Cache button)."
                )
            self._progress(3, CACHE_DIRECT_TOTAL_STEPS, self._scoring_step_label())
            if self.method == "vector":
                bodyid_df, type_df = self._vector_search(query_df, data)
            else:
                bodyid_df, type_df = self._nblast_search(query_df, data)

        # The returned/primary frame follows the level (type-to-type for type
        # queries, bodyId-to-bodyId otherwise); both files are always saved.
        results = type_df if self.level == "type" else bodyid_df
        if results.empty:
            self._log("No similar neurons found.")
            # Keep the query-only reference visualization useful even when a
            # search has no ranked matches.  The query and empty result tables
            # are still saved in the normal per-run folder.
            self._save_results(results, bodyid_df, type_df, query_df)
            self._visualize_top_results(results, query_df=query_df)
        else:
            self._save_results(results, bodyid_df, type_df, query_df)
            self._visualize_top_results(results, query_df=query_df)
        final_label = "Search finished (no similar neurons found)"
        if not results.empty:
            final_label = ("Saving results & visualization"
                           if self.visualize_top_n else "Saving results")
        self._progress(total_steps, total_steps, final_label)
        return results

    def _scoring_step_label(self) -> str:
        """Label for the scoring step (vector vs NBLAST refinement)."""
        if self.method == "nblast":
            return "Building dotprops & NBLAST scoring"
        return "Scoring similarity (vector)"

    # ---------------------------------------------------- profile-first
    def _connection_cache_candidates(self, query_df: pd.DataFrame,
                                     min_weight: Optional[int] = None,
                                     min_shared_partners: Optional[int] = None,
                                     roi_filter: Optional[List[str]] = None,
                                     top_k: Optional[int] = None) -> pd.DataFrame:
        """Rank candidate neurons directly from the connection cache.

        Reads ``cache/{dataset}/connections.parquet``
        (bodyId_pre, bodyId_post, weight, roi) with polars and finds neurons
        sharing upstream/downstream partners with the query:

        1. the query's partner rows (``weight >= min_weight``; optionally
           restricted to ``roi_filter`` ROIs when the dataset has ROI data —
           ``"*"`` = non-empty ROIs only),
        2. shared-partner adjacency: candidates = neurons sharing at least
           ``min_shared_partners`` upstream/downstream partner neurons with
           the query,
        3. ranked by shared-partner count (descending), target types joined
           via the neuron type map.

        Lightweight (no profile building); same adjacency semantics as the
        homolog finding, direct from the cache. Returns a DataFrame with
        target_bodyId, shared_count, profile_similarity (normalized 0-1 by
        the max shared count) and target_type.
        """
        import polars as pl

        min_weight = self.min_weight if min_weight is None else int(min_weight)
        min_shared_partners = self.min_shared_partners \
            if min_shared_partners is None else int(min_shared_partners)
        roi_filter = self.roi_filter if roi_filter is None else roi_filter

        conn_path = (Path(self.project_root) / "cache"
                     / _dataset_folder(self.dataset) / "connections.parquet")
        if not conn_path.exists():
            self._log("Connection cache missing; no candidates discoverable.")
            return pd.DataFrame()
        try:
            conn = pl.read_parquet(conn_path)
        except Exception as exc:
            self._log(f"Connection cache unreadable: {exc}")
            return pd.DataFrame()

        has_roi = "roi" in conn.columns and \
            bool(conn.filter(pl.col("roi").is_not_null() & (pl.col("roi") != "")).height)
        if has_roi and roi_filter:
            if roi_filter == ["*"] or roi_filter == "*":
                conn = conn.filter(pl.col("roi").is_not_null() & (pl.col("roi") != ""))
            else:
                conn = conn.filter(pl.col("roi").is_in(pl.Series("roi_filter", roi_filter).implode()))

        query_ids = [int(b) for b in query_df["bodyId"].tolist()]
        q = pl.Series("q", query_ids, dtype=pl.Int64)
        # implode() marks the Series unambiguously as a single membership
        # collection (polars >=1.30 deprecates same-dtype scalar semantics).
        q_coll = q.implode()
        conn = conn.with_columns([
            pl.col("bodyId_pre").cast(pl.Int64, strict=False),
            pl.col("bodyId_post").cast(pl.Int64, strict=False),
        ]).drop_nulls(["bodyId_pre", "bodyId_post"])
        conn = conn.filter(pl.col("weight") >= min_weight)

        up = conn.filter(pl.col("bodyId_post").is_in(q_coll))      # partners -> query
        down = conn.filter(pl.col("bodyId_pre").is_in(q_coll))     # query -> partners

        def _shared(partner_col: str, candidate_col: str, partner_ids) -> "pl.DataFrame":
            if len(partner_ids) == 0:
                return pl.DataFrame({candidate_col: [], "n_shared": []})
            shared = (conn
                      .filter(pl.col(partner_col).is_in(partner_ids.implode())
                              & ~pl.col(candidate_col).is_in(q_coll))
                      .group_by([candidate_col, partner_col])
                      .agg(pl.len().alias("_w"))
                      .group_by(candidate_col)
                      .agg(pl.len().alias("n_shared")))
            return shared

        up_shared = _shared("bodyId_pre", "bodyId_post", up["bodyId_pre"].unique())
        down_shared = _shared("bodyId_post", "bodyId_pre", down["bodyId_post"].unique())

        counts = None
        for part in (up_shared, down_shared):
            if part.height == 0:
                continue
            part = part.rename({part.columns[0]: "candidate"})
            counts = part if counts is None else counts.vstack(part)
        if counts is None:
            return pd.DataFrame()
        counts = (counts
                  .group_by("candidate")
                  .agg(pl.col("n_shared").sum())
                  .filter(pl.col("n_shared") >= min_shared_partners)
                  .sort("n_shared", descending=True))
        if top_k:
            counts = counts.head(top_k)

        out = pd.DataFrame({
            "target_bodyId": counts["candidate"].to_list(),
            "shared_count": counts["n_shared"].to_list(),
        })
        if out.empty:
            return out
        max_shared = float(out["shared_count"].max()) or 1.0
        out["profile_similarity"] = out["shared_count"] / max_shared
        type_map, _ = _load_neuron_type_map(self.dataset, str(self.project_root))
        out["target_type"] = out["target_bodyId"].map(
            lambda b: type_map.get(int(b), "")
        )
        return out

    def _profile_first_search(self, query_df: pd.DataFrame, cache: SkeletonVectorCache) -> pd.DataFrame:
        """Connectivity-first, then morphology on the expanded candidate pool.

        Candidate discovery reads the connection cache directly
        (``_connection_cache_candidates``); the top ``top_n * candidate_expansion``
        connectivity-similar TYPES are then expanded to ALL their member
        bodyIds (type map, safety cap ``max_pool_per_type``) — the scoring
        pool. Fetched skeletons are TRANSIENT (used for the current
        comparison only, never written to the skeleton cache); cached
        skeletons are reused. Every scored vector is standardized with ONE
        consistent set of statistics (cache meta, sample-based population
        stats, or pool stats as a last resort), and the whole comparison
        runs at ONE representation level (skeleton vs mesh — rows of any
        other representation are unscorable).
        """
        self._log("Step 2/6 — Discovering candidates: running connection-cache "
                  "candidate discovery...")
        self._progress(2, PROFILE_FIRST_TOTAL_STEPS,
                       "Discovering candidates (connection cache)")
        candidates = self._connection_cache_candidates(query_df)
        if candidates.empty:
            self._log("Connection-cache search returned no candidates.")
            return pd.DataFrame(), pd.DataFrame()

        # Rank the candidate TYPES by mean shared-partner count, keep the
        # top (top_n * candidate_expansion) types.
        import collections
        type_scores: Dict[str, List[float]] = collections.defaultdict(list)
        for _, row in candidates.iterrows():
            t = str(row.get("target_type", "") or "")
            if t:
                type_scores[t].append(float(row["shared_count"]))
        ranked_types = sorted(type_scores.items(),
                              key=lambda kv: (-np.mean(kv[1]), kv[0]))
        keep_types = [t for t, _ in ranked_types[: max(1, self.top_n * self.candidate_expansion)]]
        if not keep_types:
            self._log("Connection-cache candidates carry no types.")
            return pd.DataFrame(), pd.DataFrame()

        expansion_label = (
            f"Expanding {len(keep_types)} candidate types to the scoring pool"
        )
        self._log(f"Step 3/6 — {expansion_label}")
        self._progress(3, PROFILE_FIRST_TOTAL_STEPS, expansion_label)

        # Expand every kept type to ALL its member bodyIds (type map); the
        # union is the scoring pool. Per-type safety cap bounds huge types.
        type_map, _ = _load_neuron_type_map(self.dataset, str(self.project_root))
        pool_ids: List[int] = []
        seen_pool = set()
        for t in keep_types:
            members = [int(b) for b, tt in type_map.items() if tt == t]
            members = members[: self.max_pool_per_type]
            for m in members:
                if m not in seen_pool:
                    seen_pool.add(m)
                    pool_ids.append(m)
        self._log(f"Profile-first: {len(keep_types)} expanded types -> "
                  f"{len(pool_ids)} pool neurons "
                  f"({len(candidates)} connectivity candidates)")

        prof_by_id = {
            int(b): float(v)
            for b, v in zip(candidates["target_bodyId"], candidates["profile_similarity"])
            if np.isfinite(v)
        }

        def _load_missing(ids: List[int], rep: str = "") -> Dict[int, "navis.TreeNeuron"]:
            """Fetch skeletons missing from the cache, kept in memory only.

            Workflow order: neurons whose VECTOR is already cached need no
            skeleton at all (``cache_ids``); otherwise a cached skeleton
            file is reused — but only when its simplification level matches
            the cache's ``vector_basis`` (post-cleanup NeuPrint caches hold
            simp90 files while the basis is raw, so those are fetched raw
            transiently); only the truly missing ones are fetched online.
            ``rep`` ('skeleton'|'mesh') is the comparison's representation:
            fetches of any other representation are skipped so levels are
            never mixed within one comparison.
            """
            loaded: Dict[int, navis.TreeNeuron] = {}
            n_ids = len(ids)
            for i, bid in enumerate(ids, start=1):
                if int(bid) in cache_ids:
                    continue  # vector already cached: no skeleton needed
                if (_find_skeleton_file(self.dataset, bid, project_root=str(self.project_root)) is not None
                        and _skeleton_folder_level(self.dataset, str(self.project_root)) == cache_basis):
                    continue
                load_label = f"Step 4/6 — Loading skeletons ({i}/{n_ids})"
                self._progress(4, PROFILE_FIRST_TOTAL_STEPS,
                               load_label.replace("Step 4/6 — ", ""))
                # Progress events drive the determinate UI bar and are not
                # copied into the execution log.  Emit coarse-grained phase
                # updates there as well so a long fetch is auditable.
                checkpoint = max(1, n_ids // 10)
                if i == 1 or i == n_ids or i % checkpoint == 0:
                    self._log(load_label)
                nrn = fetch_skeleton_on_demand(
                    self.dataset, bid, project_root=str(self.project_root),
                    persist=self.cache_fetched_skeletons,
                )
                if nrn is not None:
                    if rep and _neuron_rep(nrn) != rep:
                        continue  # different representation: never comparable
                    loaded[bid] = nrn
            return loaded

        self._log("Step 4/6 — Loading & vectorizing skeletons")
        self._progress(4, PROFILE_FIRST_TOTAL_STEPS, "Loading & vectorizing skeletons")
        query_ids = [int(b) for b in query_df["bodyId"].tolist()]

        cache_data = cache.load()
        cache_ids = (set(int(b) for b in cache_data["bodyIds"])
                     if cache_data is not None else set())
        cache_rep = cache_data.get("dataset_rep", "") if cache_data is not None else ""
        cache_basis = (((cache_data.get("meta") or {}).get("vector_basis")
                        or VECTOR_BASIS_RAW) if cache_data is not None
                       else VECTOR_BASIS_RAW)

        # Query skeleton is always ensured (fetched transiently if missing;
        # neurons with a cached VECTOR need no fetch at all).
        query_neurons = _load_missing(query_ids)

        # The comparison's representation: the majority among the query
        # members (cache rows carry the cache's representation).
        from collections import Counter
        known_q = []
        for bid in query_ids:
            if int(bid) in cache_ids:
                known_q.append(cache_rep)
            elif int(bid) in query_neurons:
                known_q.append(_neuron_rep(query_neurons[int(bid)]))
        q_rep = Counter(r for r in known_q if r).most_common(1)[0][0] \
            if any(known_q) else ""
        if not q_rep:
            # No cached/fetched query member with a known representation
            # (e.g. all query rows computed from cache files): infer it from
            # the dataset's skeleton store.
            q_rep = _infer_dataset_rep(self.dataset, str(self.project_root))

        # Fetch skeletons for pool neurons missing from the cache (bounded,
        # transient; every pool neuron can trigger a fetch). Fetches of a
        # different representation than the query are skipped.
        pool_neurons = _load_missing(pool_ids, rep=q_rep)
        self._log(f"Profile-first: {len(pool_ids)} pool neurons, "
                  f"{len(pool_neurons)} skeletons fetched (transient)")

        # Vectors for query + pool: cached vectors first, then any
        # in-memory fetched neurons. ``reps`` tracks each row's
        # representation ('skeleton' | 'mesh') so the comparison never mixes
        # levels.
        X_q, mask_q, rep_q = cache.vectors_for(query_ids, compute_missing=True)
        fetched_vectors: List[Tuple[int, np.ndarray, str]] = []
        for i, bid in enumerate(query_ids):
            if not mask_q[i] and bid in query_neurons:
                _, vec = vectorize_neuron(query_neurons[bid])
                X_q[i], mask_q[i] = vec, True
                rep_q[i] = _neuron_rep(query_neurons[bid])
                fetched_vectors.append((int(bid), vec, rep_q[i]))
        X_c, mask_c, rep_c = cache.vectors_for(pool_ids, compute_missing=True)
        for i, bid in enumerate(pool_ids):
            if not mask_c[i] and bid in pool_neurons:
                _, vec = vectorize_neuron(pool_neurons[bid])
                X_c[i], mask_c[i] = vec, True
                rep_c[i] = _neuron_rep(pool_neurons[bid])
                fetched_vectors.append((int(bid), vec, rep_c[i]))

        # ALWAYS persist the computed vectors, even when the skeleton itself
        # was not cached (transient fetch): later queries reuse the vector
        # without re-fetching or re-vectorizing. Rows of a representation
        # different from the cache's are rejected inside append_vectors.
        if fetched_vectors:
            cache.append_vectors(fetched_vectors, vector_basis=cache_basis)

        # ONE representation per comparison: skeletons and meshes (or two
        # simplification levels) produce different features in the shared
        # schema, so rows of any other representation than the query's are
        # unscorable and stay out of the masks.
        from collections import Counter
        known_q = [r for r in rep_q if r]
        q_rep = Counter(known_q).most_common(1)[0][0] if known_q else ""
        if q_rep:
            rep_q = np.array(rep_q)
            rep_c = np.array(rep_c)
            mask_q = mask_q & (rep_q == q_rep)
            mask_c = mask_c & (rep_c == q_rep)
        if not mask_q.any():
            raise ValueError("Could not vectorize the query neuron.")

        # Standardize EVERY scored vector with ONE consistent set of
        # statistics so cosine is scale-fair (raw morphometrics + shape are
        # on very different scales). Cache rows are already standardized
        # with the cache's own meta stats: they are left untouched when
        # those stats are used, and restored to raw (then re-standardized)
        # when the stats come from elsewhere. Freshly-computed rows always
        # receive the chosen transform. A small vector cache carries skewed,
        # unreliable stats (it is typically built from one query's transient
        # fetches), so its meta is ignored in favour of sample-based stats,
        # which extend the sample with a version sibling's skeletons.
        meta_mu = meta_sd = None
        if cache_data is not None:
            meta = cache_data.get("meta") or {}
            m = meta.get("mean")
            s = meta.get("std")
            if m is not None and s is not None:
                mm = np.asarray(m, dtype=float)
                ss = np.asarray(s, dtype=float)
                if mm.shape == (VECTOR_DIM,) and ss.shape == (VECTOR_DIM,):
                    meta_mu, meta_sd = mm, ss

        using_cache_stats = False
        if mask_q.any() or mask_c.any():
            mu = sd = None
            if (meta_mu is not None
                    and len(cache_data["bodyIds"]) >= MIN_POPULATION_STATS_SKELETONS):
                mu, sd, using_cache_stats = meta_mu, meta_sd, True
            if mu is None:
                mu, sd = population_stats(self.dataset, str(self.project_root))
            if mu is None:
                # Last resort: pool-computed statistics.
                all_rows = np.vstack([X_q[mask_q], X_c[mask_c]])
                mu = all_rows.mean(axis=0)
                sd = all_rows.std(axis=0)
                sd = np.where(sd <= 0, 1.0, sd)

            cache_q = (np.array([int(b) in cache_ids for b in query_ids])
                       & mask_q)
            cache_c = (np.array([int(b) in cache_ids for b in pool_ids])
                       & mask_c)
            if using_cache_stats:
                # Cache rows already use these stats; transform only the
                # freshly-computed rows (raw -> standardized).
                X_q[mask_q & ~cache_q] = (X_q[mask_q & ~cache_q] - mu) / sd
                X_c[mask_c & ~cache_c] = (X_c[mask_c & ~cache_c] - mu) / sd
            else:
                # Cache rows carry the cache's own standardization: restore
                # the raw vectors first so every row gets the same transform.
                if meta_mu is not None:
                    X_q[cache_q] = X_q[cache_q] * meta_sd + meta_mu
                    X_c[cache_c] = X_c[cache_c] * meta_sd + meta_mu
                X_q[mask_q] = (X_q[mask_q] - mu) / sd
                X_c[mask_c] = (X_c[mask_c] - mu) / sd

        q_vec = X_q[mask_q].mean(axis=0)
        keep = mask_c
        scores = np.full(len(pool_ids), np.nan)
        if keep.any():
            self._log(f"Step 5/6 — Scoring similarity ({self.method}) for "
                      f"{int(keep.sum())} usable candidates")
            self._progress(5, PROFILE_FIRST_TOTAL_STEPS,
                           f"Scoring similarity ({self.method})")
            scores[keep] = similarity_matrix(q_vec, X_c[keep], self.metric)

        query_type = ""
        if len(query_df):
            raw_query_type = query_df["type"].iloc[0]
            query_type = "" if pd.isna(raw_query_type) else str(raw_query_type).strip()
        query_ids_set = set(query_ids)
        # Type lookup: the full neuron table / index map, with the vector
        # cache's labels overriding for the neurons it covers (the cache is
        # freshest there). The old replace-all behaviour silently dropped
        # every pool neuron outside the cache from the type-level results.
        id_to_type, id_to_instance = _load_neuron_type_map(
            self.dataset, str(self.project_root)
        )
        if cache_data is not None:
            id_to_type.update(
                {int(b): t for b, t in zip(cache_data["bodyIds"], cache_data["types"])}
            )
            id_to_instance.update(
                {int(b): i for b, i in zip(cache_data["bodyIds"], cache_data["instances"])}
            )
        intra = float("nan")
        # Type queries are resolved to all members of the queried type. Use
        # those unified-standardized query vectors directly, rather than a
        # possibly stale cache snapshot loaded before fetched query vectors
        # were appended.
        if self.level == "type" and query_type and mask_q.any():
            query_ok = np.where(mask_q)[0]
            intra = self._intra_type_similarity(
                query_type,
                np.asarray([query_ids[i] for i in query_ok], dtype=np.int64),
                [query_type for _ in query_ok],
                X_q[query_ok],
                self.metric,
            )
        elif using_cache_stats and cache_data is not None and len(cache_data["bodyIds"]):
            # BodyId queries still use the complete cached type population so
            # their same-type rows retain the established reference value.
            intra = self._intra_type_similarity(
                query_type, cache_data["bodyIds"], cache_data["types"],
                cache_data["X"], self.metric,
            )
        if not np.isfinite(intra) and query_type and mask_q.any():
            ok = np.where(mask_q)[0]
            intra = self._intra_type_similarity(
                query_type,
                np.array([query_ids[i] for i in ok], dtype=np.int64),
                [str(query_df["type"].iloc[i]).strip() if i < len(query_df) else ""
                 for i in ok],
                X_q[ok], self.metric,
            )

        rows: List[Dict[str, object]] = []
        if self.level == "type":
            # Compare every resolved query member to every usable candidate.
            # The old centroid row carried query_ids[0] as its source and
            # therefore made a multi-bodyId type query look like one neuron.
            candidate_indices = [i for i, ok in enumerate(keep) if ok]
            query_indices = [i for i, ok in enumerate(mask_q) if ok]
            for query_i in query_indices:
                q_bid = int(query_ids[query_i])
                raw_q_type = query_df["type"].iloc[query_i]
                q_type = "" if pd.isna(raw_q_type) else str(raw_q_type).strip()
                q_scores = (similarity_matrix(
                    X_q[query_i], X_c[candidate_indices], self.metric
                ) if candidate_indices else np.asarray([]))
                for score_i, candidate_i in enumerate(candidate_indices):
                    bid = int(pool_ids[candidate_i])
                    if bid in query_ids_set:
                        continue
                    target_type = str(id_to_type.get(bid, "") or "").strip()
                    rows.append({
                        "source_bodyId": q_bid,
                        "source_type": q_type,
                        "target_bodyId": bid,
                        "target_type": target_type,
                        "target_instance": id_to_instance.get(bid, ""),
                        "profile_similarity": prof_by_id.get(bid, np.nan),
                        "similarity": float(q_scores[score_i]),
                        "is_same_type": target_type == q_type if target_type else False,
                        "intra_type_similarity": intra,
                        "method": self.method,
                        "metric": self.metric,
                        "candidate_source": "profile",
                    })
            rows.extend(self._type_query_intra_rows(
                query_df, X_q, mask_q, intra, candidate_source="profile"
            ))
        else:
            for i, bid in enumerate(pool_ids):
                if not keep[i]:
                    continue
                bid = int(bid)
                target_type = str(id_to_type.get(bid, "") or "").strip()
                if bid in query_ids_set:
                    continue
                rows.append({
                    "source_bodyId": query_ids[0],
                    "source_type": query_type,
                    "target_bodyId": bid,
                    "target_type": target_type,
                    "target_instance": id_to_instance.get(bid, ""),
                    "profile_similarity": prof_by_id.get(bid, np.nan),
                    "similarity": float(scores[i]),
                    "is_same_type": target_type == query_type if target_type else False,
                    "intra_type_similarity": intra,
                    "method": self.method,
                    "metric": self.metric,
                    "candidate_source": "profile",
                })
        if not rows and not (self.level == "type" and np.isfinite(intra)):
            return pd.DataFrame(), pd.DataFrame()

        query_type_count = self._type_member_count(
            query_type, id_to_type, query_df=query_df
        )
        type_df = self._aggregate_type_rows(
            rows,
            query_type=query_type,
            intra=intra,
            query_type_count=query_type_count,
            candidate_source="profile",
        )

        # bodyId-level rows: top-N scored neurons.
        bodyid_df = self._bodyid_dataframe(rows, query_type=query_type)

        # NBLAST refinement over the fetched pool skeletons (transient).
        if self.method == "nblast":
            neurons = dict(query_neurons)
            neurons.update(pool_neurons)
            bodyid_df = self._nblast_refine(bodyid_df, query_df, cache, neurons)
        return bodyid_df, type_df

    def _nblast_refine(self, results: pd.DataFrame, query_df: pd.DataFrame,
                       cache: SkeletonVectorCache,
                       neurons: Optional[Dict[int, "navis.TreeNeuron"]] = None) -> pd.DataFrame:
        """Replace vector scores with NBLAST scores for the fetched candidates."""
        self._progress(5, PROFILE_FIRST_TOTAL_STEPS, "Refining scores with NBLAST")
        query_ids = {int(b) for b in query_df["bodyId"].tolist()}
        # Type-level results also contain intra-type rows whose target
        # is another query member.  They are reference pairs, not candidates
        # for refinement; keep their vector similarity below.
        cand_ids = [
            int(b) for b in results["target_bodyId"].tolist()
            if int(b) not in query_ids
        ]
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
        # In type mode, ``results`` also contains the vector-based ordered
        # intra-type pairs.  They are not in the candidate NBLAST map because
        # query members are deliberately excluded from that candidate set;
        # preserve their already-computed intra similarity instead of turning
        # those rows into NaN during refinement.
        results["similarity"] = results.apply(
            lambda row: nblast_scores.get(
                int(row["target_bodyId"]), row["similarity"]
            ) if bool(row.get("is_same_type", False))
            else nblast_scores.get(int(row["target_bodyId"]), np.nan),
            axis=1,
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

    @staticmethod
    def _type_member_count(type_name: str, id_to_type: Dict[int, str],
                           query_df: Optional[pd.DataFrame] = None) -> int:
        """Count the bodyIds belonging to a type without relying on a cache.

        A profile-first run can load ``cache_data`` before it vectorizes the
        query members.  Counting from that snapshot therefore undercounts a
        queried type (and historically produced ``n_bodyids=0``).  The neuron
        index is the authoritative population; ``query_df`` is a final
        fallback for small/test datasets whose index is unavailable.
        """
        member_ids = set()
        for bid, t in (id_to_type or {}).items():
            if str(t or "").strip() == str(type_name or "").strip():
                try:
                    member_ids.add(int(bid))
                except (TypeError, ValueError):
                    continue
        if query_df is not None and not query_df.empty:
            for _, row in query_df.iterrows():
                if str(row.get("type", "") or "").strip() != str(type_name or "").strip():
                    continue
                try:
                    member_ids.add(int(row.get("bodyId")))
                except (TypeError, ValueError):
                    continue
        return len(member_ids)

    def _type_query_intra_rows(
        self,
        query_df: pd.DataFrame,
        X: np.ndarray,
        mask: np.ndarray,
        intra: float,
        candidate_source: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """Return unique bodyId-level pairs within a type query.

        Type searches resolve to every bodyId in the queried type.  Keep the
        individual same-type comparisons in ``results.csv`` as well as the
        type-level reference row.  Emit one row per unordered pair, while
        orienting the rows so every member is represented as a source when
        there are at least three members.  The mean of these unique pairs
        matches the off-diagonal mean used by ``_intra_type_similarity``
        because the supported similarity metrics are symmetric.
        """
        if self.level != "type" or query_df is None or query_df.empty:
            return []

        records = []
        for i, (_, row) in enumerate(query_df.iterrows()):
            if i >= len(mask) or not bool(mask[i]):
                continue
            try:
                bid = int(row["bodyId"])
            except (KeyError, TypeError, ValueError):
                continue
            raw_type = row.get("type", "")
            q_type = "" if pd.isna(raw_type) else str(raw_type).strip()
            records.append((i, bid, q_type, str(row.get("instance", "") or "")))

        rows: List[Dict[str, object]] = []
        pair_directions = {}
        n_records = len(records)
        # A cyclic orientation gives every member an outgoing row without
        # duplicating an unordered pair.  For two members there is only one
        # possible row, so both members are still represented across its
        # endpoints.
        if n_records >= 3:
            for i in range(n_records):
                j = (i + 1) % n_records
                pair_directions[tuple(sorted((i, j)))] = (i, j)
        for i in range(n_records):
            for j in range(i + 1, n_records):
                pair_directions.setdefault((i, j), (i, j))

        for source_i, target_i in pair_directions.values():
            _, source_bid, source_type, _ = records[source_i]
            _, target_bid, target_type, target_instance = records[target_i]
            score = float(similarity_matrix(
                X[source_i], X[target_i].reshape(1, -1), self.metric
            )[0])
            row: Dict[str, object] = {
                "source_bodyId": source_bid,
                "source_type": source_type,
                "target_bodyId": target_bid,
                "target_type": target_type,
                "target_instance": target_instance,
                "profile_similarity": np.nan,
                "similarity": score,
                "is_same_type": True,
                "intra_type_similarity": intra,
                "method": self.method,
                "metric": self.metric,
            }
            if candidate_source is not None:
                row["candidate_source"] = candidate_source
            rows.append(row)
        return rows

    def _aggregate_type_rows(
        self,
        rows: List[Dict[str, object]],
        query_type: str,
        intra: float,
        query_type_count: int = 0,
        candidate_source: Optional[str] = None,
    ) -> pd.DataFrame:
        """Aggregate bodyId rows while preserving the queried type count.

        For inter-type rows, ``n_bodyids`` counts unique target bodyIds even
        when a type query contributes one row per query source.  The queried
        type is a reference row, so its count is the population member count,
        not the number of pairwise rows.
        """
        import collections

        grouped: Dict[str, List[Dict[str, object]]] = collections.defaultdict(list)
        for row in rows:
            target_type = str(row.get("target_type", "") or "").strip()
            if target_type:
                grouped[target_type].append(row)

        agg_rows: List[Dict[str, object]] = []
        for target_type, subrows in grouped.items():
            values = [float(row["similarity"]) for row in subrows
                      if pd.notna(row.get("similarity"))]
            if not values:
                continue
            target_ids = {
                int(row["target_bodyId"])
                for row in subrows
                if row.get("target_bodyId") is not None
            }
            profile_values = [float(row["profile_similarity"])
                              for row in subrows
                              if pd.notna(row.get("profile_similarity"))]
            is_intra = target_type == str(query_type or "").strip()
            row = {
                "target_type": target_type,
                "similarity": float(intra if is_intra and np.isfinite(intra)
                                     else np.mean(values)),
                "n_bodyids": int(query_type_count if is_intra and query_type_count
                                  else len(target_ids)),
                "profile_similarity": (
                    np.nan if is_intra else
                    (float(np.mean(profile_values)) if profile_values else np.nan)
                ),
                "is_intra_type": is_intra,
                "intra_type_similarity": intra if is_intra else float("nan"),
                "method": self.method,
                "metric": self.metric,
            }
            if candidate_source is not None:
                row["candidate_source"] = candidate_source
            agg_rows.append(row)

        if (query_type and query_type not in grouped and np.isfinite(intra)):
            row = {
                "target_type": str(query_type).strip(),
                "similarity": intra,
                "n_bodyids": int(query_type_count),
                "profile_similarity": np.nan,
                "is_intra_type": True,
                "intra_type_similarity": intra,
                "method": self.method,
                "metric": self.metric,
            }
            if candidate_source is not None:
                row["candidate_source"] = candidate_source
            agg_rows.append(row)

        agg_rows = sorted(
            agg_rows,
            key=lambda row: (
                not bool(row["is_intra_type"]),
                -float(row["similarity"]),
                str(row["target_type"]),
            ),
        )
        result = pd.DataFrame(agg_rows).head(self.top_n).reset_index(drop=True)
        result.insert(0, "rank", np.arange(1, len(result) + 1))
        return result

    def _bodyid_dataframe(
        self, rows: List[Dict[str, object]], query_type: str = ""
    ) -> pd.DataFrame:
        """Rank bodyId rows, retaining all intra-type pairs for type queries."""
        if not rows:
            return pd.DataFrame()
        ordered = sorted(
            rows,
            key=lambda row: (-float(row["similarity"]), int(row["target_bodyId"])),
        )
        if self.level == "type" and query_type:
            # Preserve every resolved same-type pair so a type query does not
            # collapse to the first source bodyId. Inter-type rows retain the
            # normal top-N limit.
            intra_rows = [
                row for row in ordered
                if bool(row.get("is_same_type"))
                and str(row.get("target_type", "") or "").strip() == query_type
            ]
            inter_rows = [
                row for row in ordered
                if not (
                    bool(row.get("is_same_type"))
                    and str(row.get("target_type", "") or "").strip() == query_type
                )
            ]
            ordered = intra_rows + inter_rows[: self.top_n]
            ordered = sorted(
                ordered,
                key=lambda row: (-float(row["similarity"]),
                                 int(row["source_bodyId"]),
                                 int(row["target_bodyId"])),
            )
        else:
            ordered = ordered[: self.top_n]
        result = pd.DataFrame(ordered).reset_index(drop=True)
        result.insert(0, "rank", np.arange(1, len(result) + 1))
        return result

    def _vector_search(self, query_df: pd.DataFrame, data: dict
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Vector similarity over the cache population.

        Returns (bodyId-level df, type-level df): results.csv always holds
        the bodyId rows and type_summary.csv the type rows, whatever the
        query kind. Type queries score every resolved query member against
        every candidate member in the bodyId export; results.csv keeps
        ordered intra-type pairs. Inter-type summary rows average those
        pairs; the cache-direct query-type reference retains its centroid
        score contract."""
        body_ids = data["bodyIds"]
        types = data["types"]
        X = data["X"]
        query_ids = [int(b) for b in query_df["bodyId"].tolist()]
        query_ids_set = set(query_ids)
        q_type = ""
        if len(query_df):
            raw_q_type = query_df["type"].iloc[0]
            q_type = "" if pd.isna(raw_q_type) else str(raw_q_type).strip()

        q_mask = np.isin(body_ids, query_ids)
        # Cache-direct type searches use every cached member of the queried
        # type; the type index below supplies the authoritative member count
        # even if a legacy query resolver returned only one member.
        intra = self._intra_type_similarity(q_type, body_ids, types, X, self.metric)
        type_map, _ = _load_neuron_type_map(
            self.dataset, str(self.project_root)
        )
        query_type_count = self._type_member_count(
            q_type, type_map, query_df=query_df
        )

        # --- bodyId rows ---
        rows = []
        if self.level == "type":
            if not q_mask.any():
                return pd.DataFrame(), pd.DataFrame()
            query_pair_X = np.full(
                (len(query_df), X.shape[1]), np.nan, dtype=float
            )
            query_pair_mask = np.zeros(len(query_df), dtype=bool)
            for query_i, qrow in enumerate(query_df.itertuples(index=False)):
                q_bid = int(qrow.bodyId)
                q_idx = np.where(body_ids == q_bid)[0]
                if not len(q_idx):
                    continue
                q_vec = X[q_idx[0]]
                query_pair_X[query_i] = q_vec
                query_pair_mask[query_i] = True
                scores = similarity_matrix(q_vec, X, self.metric)
                source_type = str(getattr(qrow, "type", q_type) or "").strip()
                for i, bid in enumerate(body_ids):
                    bid = int(bid)
                    if bid in query_ids_set:
                        continue
                    target_type = str(types[i] or "").strip()
                    rows.append({
                        "source_bodyId": q_bid,
                        "source_type": source_type,
                        "target_bodyId": bid,
                        "target_type": target_type,
                        "target_instance": data["instances"][i],
                        "similarity": float(scores[i]),
                        "is_same_type": target_type == source_type,
                        "intra_type_similarity": intra,
                        "method": self.method,
                        "metric": self.metric,
                    })
            rows.extend(self._type_query_intra_rows(
                query_df, query_pair_X, query_pair_mask, intra,
                candidate_source=None
            ))
        else:
            # Multi-query support: each query row is ranked independently
            # (self and any co-query neurons excluded from its rows).
            for _, qrow in query_df.iterrows():
                q_vec = self._vector_for_body_id(int(qrow["bodyId"]), body_ids, X)
                if q_vec is None:
                    continue
                scores = similarity_matrix(q_vec, X, self.metric)
                row_intra = self._intra_type_similarity(
                    qrow["type"], body_ids, types, X, self.metric
                )
                for i, bid in enumerate(body_ids):
                    if int(bid) in query_ids_set:
                        continue
                    rows.append({
                        "source_bodyId": int(qrow["bodyId"]),
                        "source_type": str(qrow["type"] or "").strip(),
                        "target_bodyId": int(bid),
                        "target_type": str(types[i] or "").strip(),
                        "target_instance": data["instances"][i],
                        "similarity": float(scores[i]),
                        "is_same_type": types[i] == qrow["type"],
                        "intra_type_similarity": row_intra,
                        "method": self.method,
                        "metric": self.metric,
                    })
        bodyid_df = self._bodyid_dataframe(rows, query_type=q_type)

        # --- type rows ---
        if self.level == "type":
            # Aggregate inter-type scores over every query-member/candidate
            # pair.  Keep the established centroid score for the query-type
            # reference itself; ``intra_type_similarity`` remains the
            # off-diagonal member-pair reference.
            q_vec = X[q_mask].mean(axis=0) if q_mask.any() else None
            type_df = self._aggregate_type_rows(
                rows,
                query_type=q_type,
                intra=intra,
                query_type_count=(query_type_count or int(q_mask.sum())),
            )
            if q_vec is not None and q_type and not type_df.empty:
                centroid_scores = similarity_matrix(q_vec, X, self.metric)
                query_type_mask = np.asarray(
                    [str(t or "").strip() == q_type for t in types],
                    dtype=bool,
                )
                if query_type_mask.any():
                    centroid_similarity = float(
                        np.mean(centroid_scores[query_type_mask])
                    )
                    type_df.loc[
                        type_df["target_type"] == q_type, "similarity"
                    ] = centroid_similarity
                    if query_type_count:
                        type_df.loc[
                            type_df["target_type"] == q_type, "n_bodyids"
                        ] = query_type_count
        else:
            # Keep the established centroid aggregation for bodyId queries'
            # per-type summary.
            q_vec = X[q_mask].mean(axis=0) if q_mask.any() else None
            type_df = pd.DataFrame()
            if q_vec is not None:
                scores = similarity_matrix(q_vec, X, self.metric)
                type_df = self._aggregate_by_type(
                    body_ids, types, scores, query_type=q_type,
                    metric=self.metric, X=X,
                )
        return bodyid_df, type_df

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
    def _nblast_search(self, query_df: pd.DataFrame, data: dict
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
            return pd.DataFrame(), pd.DataFrame()

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

        query_type = ""
        if len(query_df):
            raw_query_type = query_df["type"].iloc[0]
            query_type = "" if pd.isna(raw_query_type) else str(raw_query_type).strip()
        query_mask = np.isin(body_ids, list(query_ids))
        if self.level == "type" and query_mask.any():
            intra = self._intra_type_similarity(
                query_type,
                body_ids[query_mask],
                [query_type for _ in range(int(query_mask.sum()))],
                X[query_mask],
                "cosine",
            )
        else:
            intra = self._intra_type_similarity(query_type, body_ids, types, X, "cosine")

        type_map, _ = _load_neuron_type_map(
            self.dataset, str(self.project_root)
        )
        query_type_count = self._type_member_count(
            query_type, type_map, query_df=query_df
        )

        # --- bodyId rows (query members excluded) ---
        rows: List[Dict[str, object]] = []
        for bid, score in ranked:
            if bid in query_ids:
                continue
            t = str(id_to_type.get(bid, "") or "").strip()
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
        if self.level == "type" and query_mask.any():
            query_pair_X = np.full(
                (len(query_df), X.shape[1]), np.nan, dtype=float
            )
            query_pair_mask = np.zeros(len(query_df), dtype=bool)
            for query_i, qrow in enumerate(query_df.itertuples(index=False)):
                q_idx = np.where(body_ids == int(qrow.bodyId))[0]
                if len(q_idx):
                    query_pair_X[query_i] = X[q_idx[0]]
                    query_pair_mask[query_i] = True
            rows.extend(self._type_query_intra_rows(
                query_df, query_pair_X, query_pair_mask, intra,
                candidate_source=None,
            ))
        bodyid_df = self._bodyid_dataframe(rows, query_type=query_type)

        # --- type rows (per-type NBLAST means + intra reference) ---
        if self.level == "type":
            type_df = self._aggregate_type_rows(
                rows,
                query_type=query_type,
                intra=intra,
                query_type_count=(query_type_count or int(query_mask.sum())),
            )
            return bodyid_df, type_df

        import collections
        agg: Dict[str, List[float]] = collections.defaultdict(list)
        for bid, score in ranked:
            if bid in query_ids:
                continue
            t = id_to_type.get(bid, "")
            if t:
                agg[t].append(score)
        agg_rows = []
        for t, vals in agg.items():
            # Cap per-type pairs to n_per_type for the mean.
            vals = sorted(vals, reverse=True)[: self.n_per_type]
            agg_rows.append({
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
            agg_rows.append({
                "target_type": query_type,
                "similarity": intra,
                "n_bodyids": n_members,
                "is_intra_type": True,
                "intra_type_similarity": intra,
                "method": self.method,
                "metric": "nblast",
            })
        agg_rows = sorted(agg_rows, key=lambda r: (not r["is_intra_type"], -r["similarity"], r["target_type"]))
        type_df = pd.DataFrame(agg_rows).head(self.top_n).reset_index(drop=True)
        type_df.insert(0, "rank", np.arange(1, len(type_df) + 1))
        return bodyid_df, type_df

    def _dotprops_for_ids(self, body_ids: List[int],
                          neurons: Optional[Dict[int, "navis.TreeNeuron"]] = None
                          ) -> Dict[int, Optional["navis.core.dotprop.Dotprops"]]:
        """Load skeletons and build dotprops in microns.

        ``neurons`` supplies transient in-memory skeletons (profile-first
        fetches) so they are not re-fetched; anything missing is fetched
        (persisted only when ``cache_fetched_skeletons`` is enabled)."""
        out: Dict[int, Optional[navis.core.dotprop.Dotprops]] = {}
        for bid in body_ids:
            nrn = (neurons or {}).get(bid)
            if nrn is None:
                pkl = _find_skeleton_file(self.dataset, bid, project_root=str(self.project_root))
                if pkl is None:
                    nrn = fetch_skeleton_on_demand(
                        self.dataset, bid, project_root=str(self.project_root),
                        persist=self.cache_fetched_skeletons,
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
    def _save_results(self, results: pd.DataFrame, bodyid_df: pd.DataFrame,
                      type_df: pd.DataFrame, query_df: pd.DataFrame):
        """Save the run outputs: results.csv always holds the bodyId-level
        rows and type_summary.csv the type-level rows, whatever the query
        kind (mirrors the homolog finding outputs). ``results`` is the
        primary frame (type-to-type for type queries, bodyId-to-bodyId
        otherwise) recorded in the README."""
        query_label = str(self.query)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = self.saveas or f"similar-morphology_{_dataset_folder(self.dataset)}_{query_label[:40]}_{timestamp}"
        run_dir = Path(self.output_dir) / name
        run_dir.mkdir(parents=True, exist_ok=True)

        bodyid_df.to_csv(run_dir / "results.csv", index=False)
        type_df.to_csv(run_dir / "type_summary.csv", index=False)
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
            "n_bodyid_rows": len(bodyid_df),
            "n_type_rows": len(type_df),
            "primary_level": self.level,
            "output_folder": str(run_dir),
        }
        readme = [f"DROCAT Find Similar Neurons (morphological)",
                  f"Generated: {datetime.now().isoformat(timespec='seconds')}",
                  ""]
        for k, v in params.items():
            readme.append(f"{k}: {v}")
        (run_dir / "README.txt").write_text("\n".join(readme))
        # The run-folder marker line must carry the path ONLY (the UI parses
        # the folder by splitting after the marker and checking isdir).
        self._log(f"Results saved to: {run_dir}")
        self._log(f"Saved {len(bodyid_df)} bodyId rows -> results.csv, "
                  f"{len(type_df)} type rows -> type_summary.csv")
        self.output_folder = str(run_dir)

    # ------------------------------------------------------------------ viz
    def _visualize_top_results(
        self,
        results: pd.DataFrame,
        query_df: Optional[pd.DataFrame] = None,
    ):
        """Render the query plus the top-N result neurons/types.

        The query is always the first visualization layer and does not consume
        the requested top-N result count. With ``visualize_by='type'``
        (default) each of the top-N distinct result types becomes one layer
        containing its member bodyIds (the result rows for bodyId-level
        searches, or the vector-cache members capped at ``n_per_type`` for
        type-level searches); with ``'bodyId'`` each top result row is one
        layer. The intra-type reference row is never rendered. Output goes to
        the same run folder (plot-3d_{dataset}/subfolder); a visualization
        failure is logged but never fails the similarity search.

        ``query_df`` is retained for compatibility with callers from older
        releases. It supplies the query bodyIds for the reference layer and is
        also used to remove query bodyIds if they are present in a supplied
        result frame.
        """
        if self.visualize_top_n <= 0:
            return
        if results is None:
            work = pd.DataFrame()
        else:
            work = results.copy()
        if work.empty and (query_df is None or query_df.empty):
            return
        VisualizeSkeleton = _import_visualizer()
        if VisualizeSkeleton is None:
            self._log("Visualization skipped: visualize_skeleton not available.")
            return

        if "is_intra_type" in work.columns:
            work = work[work["is_intra_type"] != True]  # noqa: E712

        layers: List[List[int]] = []
        names: List[str] = []

        def _body_ids(frame: pd.DataFrame) -> List[int]:
            values = frame.get("bodyId", pd.Series(dtype=object)).tolist()
            ids: List[int] = []
            for value in values:
                try:
                    if pd.isna(value):
                        continue
                    ids.append(int(value))
                except (TypeError, ValueError):
                    continue
            return list(dict.fromkeys(ids))

        def _body_id_in(value, ids: set) -> bool:
            try:
                return not pd.isna(value) and int(value) in ids
            except (TypeError, ValueError):
                return False

        query_body_ids = set()
        if query_df is not None and not query_df.empty:
            query_body_ids = set(_body_ids(query_df))

        # Keep the query visible as a reference layer. It is intentionally
        # added before selecting result layers so the first color and legend
        # entry identify the neuron that was searched.
        if query_body_ids:
            query_label = str(self.query)
            safe_query = "".join(
                char if char.isalnum() or char in "._-" else "_"
                for char in query_label
            ).strip("_") or "neuron"
            query_members = list(dict.fromkeys(_body_ids(query_df)))
            layers.append(query_members)
            names.append(f"query_{safe_query}_x{len(query_members)}")

        # Be defensive if a caller supplies a frame that still contains the
        # query rows. Filtering before the top-N selection is what makes the
        # requested count mean "top N results", not "top N rows including the
        # query".
        if query_body_ids and "target_bodyId" in work.columns:
            work = work.loc[
                ~work["target_bodyId"].map(
                    lambda value: _body_id_in(value, query_body_ids)
                )
            ]
        if work.empty and not layers:
            self._log("Visualization skipped: no renderable query or results.")
            return

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
                    members = _body_ids(work.loc[
                        work["target_type"] == t
                    ].rename(columns={"target_bodyId": "bodyId"}))
                members = [bid for bid in members if bid not in query_body_ids]
                if not members:
                    continue
                seen.add(t)
                rank += 1
                layers.append(members)
                names.append(f"r{rank}_{t}_x{len(members)}")
                if rank >= self.visualize_top_n:
                    break
        else:
            rank = 0
            for _, row in work.iterrows():
                if rank >= self.visualize_top_n:
                    break
                try:
                    bid = int(row.get("target_bodyId"))
                except (TypeError, ValueError):
                    continue
                if bid in query_body_ids:
                    continue
                t = str(row.get("target_type", "") or "")
                rank += 1
                layers.append([bid])
                names.append(f"r{rank}_{t or f'unknown_{bid}'}_{bid}")

        if not layers:
            self._log("Visualization skipped: no renderable layers.")
            return

        self._log(
            f"3D visualization: including query + {max(0, len(layers) - 1)} "
            "result layer(s)"
        )

        run_dir = Path(getattr(self, "output_folder", "") or self.output_dir)
        try:
            viz_kwargs = {
                "dataset": self.dataset,
                "output_dir": str(run_dir),
                "neuron_layers": layers,
                "custom_layer_names": names,
                "saveas": _dataset_folder(self.dataset),
                "include_timestamp": False,
                "skip_synapse": True,
                "skeleton_mode": "tube",
                "legend_mode": "layer" if self.visualize_by == "type" else "single",
                "brain_mesh": "template",
                "export_views": False,
                "show_fig": False,
                "cache_neurons": True,
                "verbose": "simple",
            }
            # The panel contains the same keyword names as VisualizeSkeleton.
            # Ranking controls belong to MorphologyComparer, not the renderer.
            for key, value in self.visualization_settings.items():
                if key in {"visualize_top_n", "visualize_by", "use_default_simplification"}:
                    continue
                if key == "mesh_color" and value == "auto":
                    continue
                viz_kwargs[key] = value

            # The shared panel returns None when its dataset-aware default is
            # selected. Resolve that default at the analysis boundary so the
            # dedicated Skeleton tab can retain its own historical defaults.
            if viz_kwargs.get("skeleton_mesh_simplification") is None:
                viz_kwargs["skeleton_mesh_simplification"] = (
                    default_analysis_skeleton_mesh_simplification(self.dataset)
                )

            vs = VisualizeSkeleton(
                **viz_kwargs,
            )
            vs.plot_neurons()
            viz_dir = run_dir / f"plot-3d_{_dataset_folder(self.dataset)}"
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
        X, ok, _ = cache.vectors_for(bids, compute_missing=True)
        return X, ok

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
