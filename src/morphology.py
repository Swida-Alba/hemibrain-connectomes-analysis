"""
Morphological similarity for DROCAT.

Two comparison backends are provided:

- **Vector-based** (default, fast): each neuron is reduced to a fixed vector
  of ~24 L-Measure-style morphometrics plus a 100-dim persistence vector
  (navis). Vectors are cached per dataset as a single parquet file
  (``cache/{dataset}/find_similar/morphology/skeleton_vectors.parquet``) and
  queried with cosine / Pearson similarity. Raw skeleton files live in the
  shared dataset skeleton cache at
  ``cache/{dataset}/skeletons/raw_skeletons/`` as portable ``.swc.zst``
  files (zstd-19, simplification level recorded in the header; legacy
  ``.swc.gz`` remains readable). The former ``find_similar/raw_skeletons``
  directory remains a read-only migration fallback.
- **NBLAST** (navis implementation of Costa et al. 2016): the canonical
  pairwise morphology score. It uses the same raw skeleton/vector cache for
  prefiltering, while dotprops are rebuilt on demand for the vector-
  prefiltered candidate set of the current query.

Also provides ``enrich_homolog_results``, which attaches vector-based
morphological similarity scores to already-ranked homolog finding results
(post-search, result rows only).
"""

import gzip
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import navis
from tqdm import tqdm

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - installation is covered by requirements
    zstd = None

from statvis import getNeurons

try:
    from .roi_screening import (
        RoiProfileStore, RoiScreeningUnavailable, backfill_dataset_metadata,
        load_primary_rois, roi_count_table_path,
    )
except ImportError:
    from roi_screening import (
        RoiProfileStore, RoiScreeningUnavailable, backfill_dataset_metadata,
        load_primary_rois, roi_count_table_path,
    )

try:
    from .utils.flywire_readiness import (
        FlyWireSkeletonAccessError,
        flywire_manual_skeleton_instruction,
        is_fafb_dataset,
        require_flywire_skeleton_access,
    )
except ImportError:
    from utils.flywire_readiness import (
        FlyWireSkeletonAccessError,
        flywire_manual_skeleton_instruction,
        is_fafb_dataset,
        require_flywire_skeleton_access,
    )

try:
    from .flywire_ids import (
        body_id_to_api_int,
        is_flywire_dataset,
        normalize_flywire_body_id,
        normalize_flywire_id_columns,
    )
except ImportError:
    from flywire_ids import (
        body_id_to_api_int,
        is_flywire_dataset,
        normalize_flywire_body_id,
        normalize_flywire_id_columns,
    )

try:
    from .flywire_mesh_cache import (
        FLYWIRE_MESH_CACHE_SIMPLIFICATION,
        FLYWIRE_MESH_CACHE_SOMA_RADIUS,
        FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION,
        FlyWireMeshCache,
        flywire_mesh_cache_key,
        parse_soma_position,
    )
except ImportError:
    from flywire_mesh_cache import (
        FLYWIRE_MESH_CACHE_SIMPLIFICATION,
        FLYWIRE_MESH_CACHE_SOMA_RADIUS,
        FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION,
        FlyWireMeshCache,
        flywire_mesh_cache_key,
        parse_soma_position,
    )

try:
    from .visualization_options import default_analysis_skeleton_mesh_simplification
except ImportError:
    from visualization_options import default_analysis_skeleton_mesh_simplification

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
RAW_SKELETON_CACHE_VERSION = 2

# Step-progress totals reported to the web UI during a similarity run
# (see the [DROCAT][progress] event protocol in ui/components/output_panel.py).
PROFILE_FIRST_TOTAL_STEPS = 6   # candidate-screen-first (NeuPrint) pipeline
CACHE_DIRECT_TOTAL_STEPS = 4    # vector-cache-direct (FlyWire) pipeline

# Candidate sources that discover a bounded pool before morphology scoring:
# 'profile' (connection-cache shared partners), 'roi' (primary-ROI
# distribution cosine with bilateral mirroring) and 'combined' (both).
CANDIDATE_SCREEN_SOURCES = ("profile", "roi", "combined")

# Members per type sampled for NBLAST type-level means and type-level 3D
# visualizations. A scoring/rendering detail, not a candidate-list knob:
# pool size is controlled by ``candidate_cap`` alone.
TYPE_MEMBER_SAMPLE_CAP = 5

# Maximum number of cached skeletons sampled for population standardization
# statistics when a dataset has no vector cache (see ``population_stats``).
POPULATION_STATS_SAMPLE = 3000

# A dataset whose skeleton cache holds fewer than this many neurons cannot
# estimate stable population statistics on its own; ``population_stats``
# then borrows them from a version sibling (e.g. male-cns:v1.0 <- v0.9).
MIN_POPULATION_STATS_SKELETONS = 300

# Vectorization level ("basis"). Every morphology vector cache is built from
# the raw skeleton returned by the source API. Visualization simplification is
# a render-time concern and never changes the on-disk skeleton representation.
# ``VECTOR_BASIS_SIMP90`` remains as a compatibility label for older callers;
# it is not a valid persisted skeleton-cache level anymore.
VECTOR_BASIS_RAW = "raw"
VECTOR_BASIS_SIMP90 = "simp90"
SKELETON_CACHE_LEVEL = VECTOR_BASIS_SIMP90   # legacy compatibility label
SKELETON_DOWNSAMPLE_FACTOR = 10             # legacy compatibility constant

# On-disk NeuPrint skeleton simplification: percent of nodes removed, 0-90.
# 90 = the canonical "simp90" cache (keep ~10% of nodes); 0 = raw. Every
# compressed-SWC cache file records its level in a ``# DROCAT simpl: N``
# header line so later loads can re-simplify to a different target level
# (only ever coarser; detail cannot be restored).
SIMPLIFICATION_HEADER = "DROCAT simpl:"
DEFAULT_SIMPLIFICATION = 90
MAX_SIMPLIFICATION = 90

# Keep the online NeuPrint fetch policy shared by visualization and similarity
# workflows.  The request is batched at the application boundary, while
# navis/NeuPrint performs the individual SWC requests in a small worker pool.
# This avoids one client/query setup per neuron without creating an unbounded
# nested thread pool.
NEUPRINT_FETCH_BATCH_SIZE = 64
NEUPRINT_FETCH_MAX_THREADS = 3


# =============================================================================
# Feature extraction
# =============================================================================

def _dataset_folder(dataset: str) -> str:
    """Map a dataset name to its cache folder (hemibrain:v1.2.1 -> hemibrain_v1_2_1)."""
    return dataset.replace(":", "_").replace(".", "_")


def _canonical_dataset_body_id(dataset: str, body_id):
    """Keep FlyWire IDs exact while retaining NeuPrint's integer contract."""

    if is_flywire_dataset(dataset):
        return normalize_flywire_body_id(body_id)
    return int(body_id)


def _api_dataset_body_id(dataset: str, body_id) -> int:
    """Convert a dataset body ID only at a numeric third-party API boundary."""

    if is_flywire_dataset(dataset):
        return body_id_to_api_int(body_id)
    return int(body_id)


def _load_flywire_soma_positions(
        dataset: str, root: Path,
        body_ids: Optional[List[Union[int, str]]] = None,
        ) -> Dict[str, np.ndarray]:
    """Load optional FAFB soma coordinates from the local neuron table."""
    folder = _dataset_folder(dataset)
    candidates = [
        root / "datasets" / folder / f"{folder}_allneurons_neuron_df.parquet",
        root / "datasets" / folder / f"{folder}_allneurons_neuron_df.csv",
    ]
    table = next((path for path in candidates if path.exists()), None)
    if table is None:
        return {}
    try:
        if table.suffix.lower() == ".parquet":
            columns = pd.read_parquet(table).columns.tolist()
            pos_col = next(
                (name for name in (
                    "position", "soma_position", "soma_pos", "somaLocation")
                 if name in columns),
                None,
            )
            if pos_col is None:
                return {}
            frame = pd.read_parquet(table, columns=["bodyId", pos_col])
        else:
            header = pd.read_csv(table, nrows=0).columns.tolist()
            pos_col = next(
                (name for name in (
                    "position", "soma_position", "soma_pos", "somaLocation")
                 if name in header),
                None,
            )
            if pos_col is None:
                return {}
            frame = pd.read_csv(table, usecols=["bodyId", pos_col])
        requested = (
            set(normalize_flywire_body_ids(body_ids))
            if body_ids is not None else None
        )
        out = {}
        for body_id, value in zip(frame["bodyId"], frame[pos_col]):
            key = normalize_flywire_body_id(body_id)
            if requested is not None and key not in requested:
                continue
            position = parse_soma_position(value)
            if position is not None:
                out[key] = position
        return out
    except Exception:
        return {}


def _has_local_dataset_presence(dataset: str, root: Path) -> bool:
    """Whether the dataset has real local data beyond a shipped index seed.

    The bundled neuron-index seeds only support search surfaces (auto-suggest,
    the available-neurons viewer, name resolution).  Skeleton workflows must
    not treat a seed as the authoritative neuron list of a pulled dataset:
    the index fallback is therefore only used when prepared tables, cached
    connections, or cached skeletons show the dataset is genuinely local, so
    a fresh clone never triggers bulk fetches from a seed alone.
    """
    folder = _dataset_folder(dataset)
    dataset_dir = root / "datasets" / folder
    if dataset_dir.is_dir() and any(dataset_dir.glob("*_neuron_df.*")):
        return True
    cache_dir = root / "cache" / folder
    if (cache_dir / "connections.parquet").exists():
        return True
    skeletons = cache_dir / "skeletons"
    if skeletons.is_dir() and any(
            p.suffix == ".pkl"
            or str(p).endswith((".pkl.zst", ".swc.gz", ".swc.zst"))
            for p in skeletons.rglob("*")):
        return True
    raw_skeletons = cache_dir / "skeletons" / "raw_skeletons"
    if raw_skeletons.is_dir() and any(
            p.suffix == ".pkl"
            or str(p).endswith((".pkl.zst", ".swc.gz", ".swc.zst"))
            for p in raw_skeletons.rglob("*")):
        return True
    meshes = cache_dir / "meshes"
    if meshes.is_dir() and any(
            p.suffix == ".pkl" or str(p).endswith(".pkl.zst")
            for p in meshes.rglob("*")):
        return True
    legacy_raw_skeletons = cache_dir / "find_similar" / "raw_skeletons"
    if legacy_raw_skeletons.is_dir() and any(
            p.suffix == ".pkl"
            or str(p).endswith((".pkl.zst", ".swc.gz", ".swc.zst"))
            for p in legacy_raw_skeletons.rglob("*")):
        return True
    return False


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


def _skeleton_body_id(path) -> int:
    """Extract a bodyId from a cached skeleton filename.

    The cache accepts historical ``.pkl`` files, compressed mesh pickles
    (``{bodyId}.pkl.zst``), and the portable compressed-SWC forms
    (``{bodyId}.swc.zst`` / ``{bodyId}.swc.gz``). Keeping this parser in one
    place prevents ``Path.stem`` from turning ``123.swc.gz`` or
    ``123.pkl.zst`` into an invalid bodyId during parallel cache builds.
    """
    name = Path(path).name
    if name.endswith(".swc.zst"):
        name = name[:-len(".swc.zst")]
    elif name.endswith(".swc.gz"):
        name = name[:-len(".swc.gz")]
    elif name.endswith(".pkl.zst"):
        name = name[:-len(".pkl.zst")]
    elif name.endswith(".pkl"):
        name = name[:-len(".pkl")]
    return int(name)


def _load_cached_skeleton_file(path, target_simplification: Optional[int] = None):
    """Load a legacy pickle, compressed mesh pickle, or compressed SWC.

    Compressed-SWC files record their simplification level
    (``# DROCAT simpl: N``); when ``target_simplification`` is given and is
    coarser than the stored level, the neuron is re-simplified to that
    target on load.  The stored level is attached as
    ``neuron._drocat_simplification`` (headerless legacy files read as raw,
    level 0).
    """
    import io

    path = Path(path)
    stored = 0
    if str(path).endswith(".swc.zst"):
        if zstd is None:
            raise ImportError(
                "zstandard is required to read .swc.zst caches")
        with path.open("rb") as handle:
            with zstd.ZstdDecompressor().stream_reader(handle) as reader:
                content = reader.read()
        stored = _read_stored_simplification(content)
        neuron = navis.read_swc(io.StringIO(
            content.decode("utf-8", "replace")))
        try:
            neuron.id = _skeleton_body_id(path)
        except Exception:
            pass
        # SWC has no units field. Local skeleton caches use nanometers, so
        # restore that convention after a compressed-SWC round trip.
        try:
            neuron.units = "nm"
        except Exception:
            pass
    elif str(path).endswith(".swc.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            content = handle.read()
        stored = _read_stored_simplification(content)
        neuron = navis.read_swc(io.StringIO(content))
        try:
            neuron.id = _skeleton_body_id(path)
        except Exception:
            pass
        try:
            neuron.units = "nm"
        except Exception:
            pass
    elif str(path).endswith(".pkl.zst"):
        if zstd is None:
            raise ImportError(
                "zstandard is required to read FlyWire .pkl.zst caches")
        with path.open("rb") as handle:
            with zstd.ZstdDecompressor().stream_reader(handle) as reader:
                return pickle.load(reader)
    else:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    if target_simplification is not None:
        neuron = _relevel_for_target(neuron, stored, target_simplification)
    try:
        neuron._drocat_simplification = stored
    except Exception:
        pass
    return neuron


def _simplification_factor(simplification: int) -> int:
    """Downsample factor for a simplification level (percent removed, 0-90).

    90 -> factor 10 (keep ~10% of nodes), 50 -> 2, 0 -> 1 (raw).  Values
    outside 0..90 raise ``ValueError``; the factor is floored to an int
    (the canonical downsample helper floors too).
    """
    try:
        simplification = int(simplification)
    except (TypeError, ValueError):
        raise ValueError(
            f"simplification must be an integer in 0..{MAX_SIMPLIFICATION} "
            f"(percent of nodes removed); got {simplification!r}")
    if not 0 <= simplification <= MAX_SIMPLIFICATION:
        raise ValueError(
            f"simplification must be an integer in 0..{MAX_SIMPLIFICATION} "
            f"(percent of nodes removed); got {simplification!r}")
    if simplification == 0:
        return 1
    return max(1, 100 // (100 - simplification))


def _read_stored_simplification(text) -> int:
    """Parse the simplification level recorded in a compressed-SWC header.

    Header line format: ``# DROCAT simpl: 50``. Headerless files (legacy
    gzip cache, pickles) default to raw (level 0).
    """
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    for line in text.splitlines()[:8]:
        line = line.strip()
        if line.startswith("#"):
            line = line[1:].strip()
        if line.startswith(SIMPLIFICATION_HEADER):
            try:
                return int(line[len(SIMPLIFICATION_HEADER):].strip())
            except ValueError:
                return 0
    return 0


def _relevel_for_target(neuron, stored: int, target: int):
    """Re-simplify a cached neuron from its stored level to a coarser target.

    Only ever simplifies further (``target > stored``); requesting more
    detail than the file holds returns the neuron unchanged (the caller may
    fetch raw online instead).  Factor = (100-stored)/(100-target): a 50%
    file re-leveled to 90% keeps 20% of its remaining nodes ("simplify by
    80%").  Never mutates the input.
    """
    stored = max(0, int(stored))
    _simplification_factor(target)  # validate 0..90
    target = int(target)
    # MeshNeurons and anything without a node table are never re-leveled:
    # the simplification pipeline is NeuPrint/TreeNeuron-only.
    if not hasattr(neuron, "nodes"):
        return neuron
    if target <= stored or target <= 0 or stored >= MAX_SIMPLIFICATION:
        return neuron
    factor = (100 - stored) / (100 - target)
    if factor <= 1:
        return neuron
    return _downsample_for_cache(neuron, factor)


def _write_compressed_skeleton(path, neuron,
                               simplification: Optional[int] = DEFAULT_SIMPLIFICATION,
                               codec: str = "zst",
                               codec_level: int = 19) -> None:
    """Shared simplify + compress pipeline for the on-disk skeleton cache.

    Every raw-SWC cache write routes through this function so the recorded
    simplification level and the compression format stay consistent.

    - ``simplification`` int 0-90: deterministically simplifies the neuron
      to that level (percent removed) before writing; ``0`` = raw.
    - ``simplification=None``: writes the neuron as-is and records the level
      already attached to it (``neuron._drocat_simplification``; absent = 0)
      - used by lazy migrations that must not re-simplify.
    - ``codec_level``: zstd compression level (default 19).  The transient
      ``_temp_cache`` staging writer uses level 3: measured ~2.8x faster than
      level 19 (Step 0), which lets the staging stage keep up with the fetch
      rate; the standard loader decompresses any level.

    The level is recorded as the first SWC header line
    (``# DROCAT simpl: N``) and the payload is atomically written as
    ``.swc.zst`` (zstd-19) or the legacy ``.swc.gz`` (gzip-6) form.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if simplification is None:
        stored = getattr(neuron, "_drocat_simplification", 0) or 0
        neuron_out = neuron
    else:
        factor = _simplification_factor(simplification)
        stored = int(simplification)
        neuron_out = (_downsample_for_cache(neuron, factor)
                      if simplification > 0 else neuron)
    temp_swc = None
    temp_out = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tempfile.NamedTemporaryFile(
                suffix=".swc", dir=str(path.parent), delete=False) as handle:
            temp_swc = Path(handle.name)
        navis.write_swc(neuron_out, temp_swc, write_meta=True)
        payload = temp_swc.read_bytes()
        # Record the level in the header so later loads can re-level.
        payload = b"# DROCAT simpl: %d\n" % int(stored) + payload
        if codec == "zst":
            if zstd is None:
                raise ImportError(
                    "zstandard is required to write .swc.zst caches")
            blob = zstd.ZstdCompressor(
                level=codec_level, write_content_size=True).compress(payload)
        else:
            blob = gzip.compress(payload, compresslevel=6, mtime=0)
        temp_out.write_bytes(blob)
        os.replace(temp_out, path)
    finally:
        if temp_swc is not None:
            temp_swc.unlink(missing_ok=True)
        temp_out.unlink(missing_ok=True)


def _write_compressed_swc(path, neuron) -> None:
    """Legacy raw gzip SWC writer (level 0, gzip-6).

    Retained for older callers/tests; new writes go through
    :func:`_write_compressed_skeleton` (``.swc.zst``, recorded level).
    """
    _write_compressed_skeleton(path, neuron, simplification=0, codec="gz")


def _vectorize_one_file(path: str) -> Optional[Tuple[int, List[float], List[float], str]]:
    """Module-level worker for parallel cache builds (picklable).

    Returns None for un-vectorizable files (corrupt or unexpected types)
    so a single bad file cannot break the whole build. The 4th element is
    the neuron representation ('skeleton' | 'mesh')."""
    try:
        neuron = _load_cached_skeleton_file(path)
        # Per-file level guard: only raw (level 0) on-disk skeletons are
        # vectorized into the raw-basis cache; simplified files are skipped
        # (their vectors come from the vector cache / raw fetches instead).
        if getattr(neuron, "_drocat_simplification", 0) != 0:
            return None
        morph, vector = vectorize_neuron(neuron)
        rep = _neuron_rep(neuron)
    except Exception:
        return None
    body_id = _skeleton_body_id(path)
    # Shape block (persistence / spatial histogram) = the tail after the
    # morphometric block.
    return (body_id, [morph[f] for f in MORPHOMETRIC_FEATURES],
            vector[len(MORPHOMETRIC_FEATURES):].tolist(), rep)


# ---------------------------------------------------------------------------
# Healed-bundle workers (FAFB full-dataset vectorization)
# ---------------------------------------------------------------------------
# The FAFB healed bundle ({bodyId}.swc entries) is the full skeleton source
# for FAFB v783: the local pickle cache holds meshes, which is the wrong
# representation for the vector cache. Workers open the bundle (.zst first,
# ZIP fallback) once per process; the index read is the expensive part, per-
# id reads are cheap, and ZIP-served ids are lazily converted into the .zst.

_FAFB_WORKER_BUNDLE = None


def _init_fafb_zip_worker(source_path: str, zip_path: Optional[str] = None):
    """Per-worker initializer: open the healed bundle once per process.

    ``source_path`` is the .zst bundle (created lazily when absent);
    ``zip_path`` is the legacy healed ZIP used as the fallback source with
    lazy per-skeleton conversion.
    """
    global _FAFB_WORKER_BUNDLE
    from fafb_bundle import FAFBSkeletonBundle

    _FAFB_WORKER_BUNDLE = FAFBSkeletonBundle(
        source_path, zip_path=zip_path, lazy_convert=True)


def _vectorize_one_swc(body_id: int
                       ) -> Optional[Tuple[int, List[float], List[float], str]]:
    """Module-level worker: vectorize one healed-bundle skeleton."""
    global _FAFB_WORKER_BUNDLE
    import io

    try:
        content = _FAFB_WORKER_BUNDLE.get(int(body_id))
        if content is None:
            return None
        neuron = navis.read_swc(io.StringIO(content))
        neuron.units = "nm"
        morph, vector = vectorize_neuron(neuron)
    except Exception:
        return None
    return (int(body_id), [morph[f] for f in MORPHOMETRIC_FEATURES],
            vector[len(MORPHOMETRIC_FEATURES):].tolist(), "skeleton")


def _fafb_bundle(dataset: str,
                 project_root: Optional[str] = None):
    """Healed-bundle reader for FAFB v783: .zst first, ZIP fallback (lazy).

    Returns a :class:`fafb_bundle.FAFBSkeletonBundle` when either file
    exists, else None.  The ZIP fallback path lazily converts every served
    skeleton into the .zst container.
    """
    from fafb_bundle import open_bundle as _open_fafb_bundle

    root = Path(project_root) if project_root else Path(__file__).parent.parent
    folder = _dataset_folder(dataset)
    return _open_fafb_bundle(root / "datasets" / folder, lazy_convert=True)


def _bundle_tree_neuron(bundle, body_id: int):
    """TreeNeuron for a healed-bundle body id (.zst-first, lazy ZIP convert)."""
    import io

    text = bundle.get(int(body_id))
    if text is None:
        return None
    nrn = navis.read_swc(io.StringIO(text))
    nrn.units = "nm"
    nrn.id = int(body_id)
    nrn.name = str(int(body_id))
    return nrn


def _import_visualizer():
    """Lazily import the VisualizeSkeleton class (heavy module; never loaded
    unless a run actually renders). Returns None when unavailable."""
    try:
        from visualize_skeleton import VisualizeSkeleton
        return VisualizeSkeleton
    except Exception:
        return None


def _load_neuron_type_map(dataset: str, project_root: Optional[str] = None
                          ) -> Tuple[Dict[Union[int, str], str],
                                     Dict[Union[int, str], str]]:
    """bodyId -> type / instance maps for a dataset.

    Uses the allneurons neuron table (fullest coverage), falling back to the
    neuron index parquet. These are the same sources ``SkeletonVectorCache``
    merges into the vector cache, so type lookups work even for datasets that
    have no vector cache yet (e.g. male-cns v1.0 with cached skeletons only).
    """
    root = Path(project_root) if project_root else Path(__file__).parent.parent
    folder = _dataset_folder(dataset)
    flywire = is_flywire_dataset(dataset)
    type_map: Dict[Union[int, str], str] = {}
    instance_map: Dict[Union[int, str], str] = {}

    dataset_dir = root / "datasets" / folder
    table_candidates = [
        dataset_dir / f"{folder}_allneurons_neuron_df.parquet",
        dataset_dir / f"{folder}_allneurons_neuron_df.csv",
    ]
    table_path = next((path for path in table_candidates if path.exists()), None)
    if table_path is not None:
        try:
            if table_path.suffix.lower() == ".parquet":
                tdf = pd.read_parquet(
                    table_path, columns=["bodyId", "type", "instance"]
                )
            else:
                tdf = pd.read_csv(
                    table_path,
                    usecols=["bodyId", "type", "instance"],
                    dtype={"bodyId": "string"} if flywire else None,
                )
            if flywire:
                normalize_flywire_id_columns(tdf, ["bodyId"])
            else:
                tdf["bodyId"] = tdf["bodyId"].astype(np.int64)
            type_map = dict(zip(tdf["bodyId"], tdf["type"].fillna("").astype(str)))
            instance_map = dict(zip(tdf["bodyId"], tdf["instance"].fillna("").astype(str)))
            return type_map, instance_map
        except Exception:
            pass

    index_path = root / "neuron_indexes" / folder / "neuron_index.parquet"
    if index_path.exists() and _has_local_dataset_presence(dataset, root):
        try:
            idx_df = pd.read_parquet(index_path, columns=["bodyId", "type", "instance"])
            if flywire:
                normalize_flywire_id_columns(idx_df, ["bodyId"])
            else:
                idx_df["bodyId"] = idx_df["bodyId"].astype(np.int64)
            type_map = dict(zip(idx_df["bodyId"], idx_df["type"].fillna("").astype(str)))
            instance_map = dict(zip(idx_df["bodyId"], idx_df["instance"].fillna("").astype(str)))
        except Exception:
            pass
    return type_map, instance_map


def _find_skeleton_file(dataset: str, body_id: int,
                        project_root: Optional[str] = None) -> Optional[Path]:
    """Locate a cached skeleton/mesh file for a bodyId.

    Searches the dataset's skeletons directory recursively (datasets such as
    FlyWire keep bulk downloads in nested subfolders) and returns the first
    match, or None.
    """
    root = Path(project_root) if project_root else Path(__file__).parent.parent
    cache_dir = root / "cache" / _dataset_folder(dataset) / "skeletons"
    if not cache_dir.exists():
        return None
    for name in (f"{body_id}.swc.zst", f"{body_id}.pkl.zst",
                 f"{body_id}.pkl", f"{body_id}.swc.gz"):
        direct = cache_dir / name
        if direct.exists():
            return direct
    nested = sorted(
        path for path in cache_dir.rglob(f"{body_id}.pkl.zst")
        if "raw_skeletons" not in path.parts
    )
    if nested:
        return nested[0]
    nested = sorted(
        path for path in cache_dir.rglob(f"{body_id}.pkl")
        if "raw_skeletons" not in path.parts
    )
    if nested:
        return nested[0]
    nested = sorted(
        path for path in cache_dir.rglob(f"{body_id}.swc.zst")
        if "raw_skeletons" not in path.parts
    )
    if nested:
        return nested[0]
    nested = sorted(
        path for path in cache_dir.rglob(f"{body_id}.swc.gz")
        if "raw_skeletons" not in path.parts
    )
    return nested[0] if nested else None


def _fafb_skeleton_zip_path(dataset: str,
                            project_root: Optional[str] = None) -> Optional[Path]:
    """Path of the healed FAFB skeleton bundle (``{bodyId}.swc`` entries).

    The FlyWire local mesh cache cannot serve NBLAST (dotprops want
    skeletons); the official healed skeleton ZIP is the real skeleton source.
    Returns None when the dataset has no such bundle.
    """
    root = Path(project_root) if project_root else Path(__file__).parent.parent
    folder = _dataset_folder(dataset)
    base = root / "datasets" / folder
    for candidate in (
        base / "sk_lod1_783_healed.zip",
        base / "downloads" / "sk_lod1_783_healed.zip",
        base / f"{folder}_skeletons.zip",
    ):
        if candidate.is_file():
            return candidate
    return None


def _read_fafb_zip_skeleton(zfile: "zipfile.ZipFile",
                            body_id: int) -> Optional["navis.TreeNeuron"]:
    """Load one skeleton (``{bodyId}.swc``) from the healed FAFB bundle."""
    try:
        import io

        content = zfile.read(f"{int(body_id)}.swc").decode("utf-8")
        nrn = navis.read_swc(io.StringIO(content))
        nrn.units = "nm"
        nrn.id = int(body_id)
        nrn.name = str(int(body_id))
        return nrn
    except Exception:
        return None


def _skeleton_folder_level(dataset: str,
                           project_root: Optional[str] = None) -> str:
    """Read a legacy folder-level marker for compatibility only.

    New raw SWC cache paths do not use ``skeletons/.level``. Older vector
    cache/build callers may still inspect the marker, so a missing marker
    conservatively resolves to ``raw``.
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
    """Legacy marker writer retained for old integrations and tests.

    Production fetch/pull paths never call this function: raw skeletons are
    represented by compressed SWC and have no folder-level simplification
    marker.
    """
    root = Path(project_root) if project_root else Path(__file__).parent.parent
    marker = root / "cache" / _dataset_folder(dataset) / "skeletons" / ".level"
    try:
        if not marker.exists():
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(SKELETON_CACHE_LEVEL + "\n")
    except Exception:
        pass


def _downsample_for_cache(
        neuron, downsampling_factor: int = SKELETON_DOWNSAMPLE_FACTOR
        ) -> "navis.TreeNeuron":
    """Deterministic simplification for the on-disk NeuPrint skeleton cache.

    ``navis.downsample_neuron(factor=10)`` keeps ~10% of nodes while
    preserving root/leaves/branchpoints — the canonical "90% simplified"
    skeleton. ``downsampling_factor=20`` is used by the Settings bulk-pull
    option for the separate 95% simplified skeleton cache. Falls back to the
    original neuron when downsampling fails.
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
        return navis.downsample_neuron(
            neuron, downsampling_factor=max(1, int(downsampling_factor)))
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
            rep = _neuron_rep(_load_cached_skeleton_file(files[0]))
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


def pairwise_similarity_matrix(matrix: np.ndarray,
                               metric: str = "cosine") -> np.ndarray:
    """Pairwise similarity for every row of ``matrix``.

    The one-query helper is convenient for a single search, but calling it
    once per member repeats the row norms (and the matrix multiply setup) for
    every member of a type.  Normalize the whole matrix once and use one
    matrix multiplication instead.  Zero vectors remain zero-similarity,
    matching :func:`cosine_similarity_matrix`.
    """
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be a two-dimensional array")
    if values.shape[0] == 0:
        return np.zeros((0, 0), dtype=float)
    if metric == "pearson":
        values = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(values, axis=1)
    normalized = values / np.maximum(norms[:, None], 1e-12)
    scores = normalized @ normalized.T
    zero = norms == 0
    if np.any(zero):
        scores[zero, :] = 0.0
        scores[:, zero] = 0.0
    return scores


def _sorted_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Candidate frame sorted by ``_score`` descending (ties by bodyId).

    Every discovery screen hands out its candidates in this one canonical
    order so the scoring pool is unambiguously the first ``candidate_cap``
    rows, whatever the source mode."""
    if candidates is None or candidates.empty or "_score" not in candidates.columns:
        return candidates
    return candidates.sort_values(
        ["_score", "target_bodyId"], ascending=[False, True]
    ).reset_index(drop=True)


# =============================================================================
# Skeleton vector cache
# =============================================================================

# Amortized merge thresholds for the append-only vector cache (main +
# pending): pending rows are folded into the main parquet once either
# threshold is crossed, so appends stay O(batch) and the full-file
# read-modify-write happens rarely.
PENDING_MERGE_ROWS = 5000
PENDING_MERGE_APPENDS = 8


class SkeletonVectorCache:
    """Per-dataset cache of vectorized skeletons (parquet + meta.json).

    ``raw_only=True`` creates the shared raw NeuPrint skeleton cache used by
    visualization and morphology comparison. Its files live below
    ``cache/{dataset}/skeletons/raw_skeletons/``. ``representation="mesh"``
    creates the separate FlyWire/FAFB prepared-mesh cache; it never writes
    SWC and reads the former visualization mesh folder as a migration source.
    """

    def __init__(self, dataset: str, project_root: Optional[str] = None,
                 n_workers: int = 8, verbose: bool = True,
                 raw_only: bool = False, raw_format: str = "swc.zst",
                 representation: Optional[str] = None):
        self.dataset = dataset
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.n_workers = max(1, int(n_workers))
        self.verbose = verbose
        self.raw_only = bool(raw_only)
        self.representation = str(representation or "").strip().lower()
        if self.representation not in {"", "mesh", "skeleton"}:
            raise ValueError("representation must be 'mesh', 'skeleton', or None")
        if self.raw_only and self.representation == "mesh":
            raise ValueError("raw_only cache cannot use mesh representation")
        self.mesh_only = self.representation == "mesh"
        self._flywire_ids = is_flywire_dataset(dataset)
        # Raw skeleton caches are portable compressed SWC by default.  The
        # legacy non-raw cache may still be explicitly opened as pickle for
        # migration/mesh compatibility, but new raw callers converge here.
        raw_format = str(raw_format or "swc.zst").strip().lower()
        if raw_format not in {"pkl", "swc.gz", "swc.zst"}:
            raise ValueError(
                "raw_format must be 'pkl', 'swc.gz', or 'swc.zst'")
        self.raw_format = raw_format
        folder = _dataset_folder(dataset)
        if self.mesh_only:
            base = self.project_root / "cache" / folder
            self.morph_dir = base / "find_similar" / "morphology"
            mesh_key = flywire_mesh_cache_key(
                FLYWIRE_MESH_CACHE_SIMPLIFICATION,
                FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION,
                FLYWIRE_MESH_CACHE_SOMA_RADIUS,
            )
            self.skeleton_dir = base / "meshes" / mesh_key
            self.legacy_skeleton_dir = base / "skeletons" / mesh_key
        elif self.raw_only:
            base = self.project_root / "cache" / folder
            self.morph_dir = base / "find_similar" / "morphology"
            self.skeleton_dir = base / "skeletons" / "raw_skeletons"
            self.legacy_skeleton_dir = (
                base / "find_similar" / "raw_skeletons"
            )
        else:
            self.morph_dir = self.project_root / "cache" / folder / "morphology"
            self.skeleton_dir = self.project_root / "cache" / folder / "skeletons"
            self.legacy_skeleton_dir = None
        vector_name = "mesh_vectors.parquet" if self.mesh_only \
            else "skeleton_vectors.parquet"
        meta_name = "mesh_meta.json" if self.mesh_only else "meta.json"
        self.parquet_path = self.morph_dir / vector_name
        self.meta_path = self.morph_dir / meta_name
        # Append-only staging file: new rows land here (O(batch)) and are
        # folded into ``parquet_path`` by the amortized merge checkpoint.
        self.pending_path = self.morph_dir / (
            "mesh_vectors_pending.parquet" if self.mesh_only
            else "skeleton_vectors_pending.parquet"
        )
        # Raw NeuPrint SWCs are the shared morphology/render source.  Keep a
        # small provenance manifest beside them so a cleared/rebuilt cache is
        # self-describing and cannot be mistaken for a render-time mesh cache.
        self.skeleton_manifest_path = self.skeleton_dir / (
            "raw_skeleton_manifest.json" if self.raw_only
            else "skeleton_manifest.json"
        )

    def _canonical_body_id(self, body_id):
        """Canonical cache key: exact strings for FlyWire, ints for NeuPrint."""

        if self._flywire_ids:
            return normalize_flywire_body_id(body_id)
        return int(body_id)

    # ------------------------------------------------------------------ paths
    def cache_exists(self) -> bool:
        # A pending file counts as cache existence: rows appended but not yet
        # merged are real data (a crash must never make them invisible or
        # trigger a rebuild that would drop them).
        return self.parquet_path.exists() or self.pending_path.exists()

    def _load_skeleton_manifest(self) -> dict:
        """Load the optional raw-skeleton provenance manifest."""
        if not self.raw_only or not self.skeleton_manifest_path.exists():
            return {
                "cache_schema_version": RAW_SKELETON_CACHE_VERSION,
                "dataset": self.dataset,
                "representation": "skeleton",
                "source": "neuprint.fetch_skeleton",
                "coordinate_units": "nm",
                "vector_basis": VECTOR_BASIS_RAW,
                "files": {},
            }
        try:
            manifest = json.loads(self.skeleton_manifest_path.read_text())
            if (
                manifest.get("cache_schema_version")
                != RAW_SKELETON_CACHE_VERSION
                or manifest.get("dataset") != self.dataset
                or manifest.get("representation") != "skeleton"
                or not isinstance(manifest.get("files", {}), dict)
            ):
                raise ValueError("raw skeleton manifest has incompatible provenance")
            manifest.setdefault("files", {})
            return manifest
        except Exception:
            # A malformed/old manifest must not make otherwise valid SWCs
            # unreadable; the next successful write repairs the metadata.
            return {
                "cache_schema_version": RAW_SKELETON_CACHE_VERSION,
                "dataset": self.dataset,
                "representation": "skeleton",
                "source": "neuprint.fetch_skeleton",
                "coordinate_units": "nm",
                "vector_basis": VECTOR_BASIS_RAW,
                "files": {},
            }

    def _write_skeleton_manifest(self, manifest: dict) -> None:
        """Atomically persist raw-skeleton provenance metadata."""
        if not self.raw_only:
            return
        self.skeleton_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.skeleton_manifest_path.with_name(
            f".{self.skeleton_manifest_path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
            os.replace(temporary, self.skeleton_manifest_path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    # -------------------------------------------------- pending (append-only)
    @staticmethod
    def _atomic_parquet(frame, path) -> None:
        """Atomic parquet write (temp + os.replace): a crash never leaves a
        truncated main/pending file behind."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            frame.to_parquet(temporary, index=False)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass

    @staticmethod
    def _dedupe_frames(main_df: pd.DataFrame,
                       pending_df: pd.DataFrame) -> pd.DataFrame:
        """Concat main + pending and dedupe by bodyId, first occurrence wins
        (main rows win over pending rows — matches the historical "existing
        rows win" semantics)."""
        if pending_df is None or pending_df.empty:
            return (main_df.reset_index(drop=True)
                    if main_df is not None and not main_df.empty else main_df)
        if main_df is None or main_df.empty:
            combined = pending_df
        else:
            combined = pd.concat([main_df, pending_df], ignore_index=True)
        if combined.empty or "bodyId" not in combined.columns:
            return combined
        return combined.drop_duplicates(
            subset=["bodyId"], keep="first").reset_index(drop=True)

    def _clear_pending(self) -> None:
        """Remove the pending file and reset its append counter."""
        try:
            if self.pending_path.exists():
                self.pending_path.unlink()
        except OSError:
            pass
        meta = self._load_meta() or {}
        if meta.get("pending_appends"):
            meta["pending_appends"] = 0
            try:
                self.meta_path.write_text(json.dumps(meta, indent=2))
            except OSError:
                pass

    def _vector_row_count(self) -> int:
        """Rows in main + pending (informational counts)."""
        n = 0
        for path in (self.parquet_path, self.pending_path):
            if not path.exists():
                continue
            try:
                n += len(pd.read_parquet(path))
            except Exception:
                pass
        return n

    def _merge_pending(self) -> int:
        """Fold pending rows into the main parquet (deduped, first-wins) and
        clear pending.  Called by ``append_vectors`` once a threshold is
        crossed (amortized O(n)) and as the pull's final checkpoint.
        Returns the number of rows written to main (0 = nothing to merge)."""
        if not self.pending_path.exists():
            return 0
        try:
            pending_df = pd.read_parquet(self.pending_path)
        except Exception:
            pending_df = pd.DataFrame()
        if pending_df is None or pending_df.empty:
            self._clear_pending()
            return 0
        main_df = pd.DataFrame()
        if self.parquet_path.exists():
            try:
                main_df = pd.read_parquet(self.parquet_path)
            except Exception:
                main_df = pd.DataFrame()
        merged = self._dedupe_frames(main_df, pending_df)
        if merged.empty:
            self._clear_pending()
            return 0
        # Align the schema to the main file's columns (legacy mains may lack
        # newer columns), as the historical append did with keep_cols.
        if main_df is not None and not main_df.empty:
            keep = [c for c in main_df.columns]
            merged = merged[[c for c in keep if c in merged.columns]]
        merged = merged.sort_values("bodyId").reset_index(drop=True)
        self._atomic_parquet(merged, self.parquet_path)
        meta = self._load_meta() or {}
        if not meta.get("mean") and not merged.empty:
            mat = self._raw_matrix(merged)
            mean = mat.mean(axis=0).tolist()
            std = mat.std(axis=0).tolist()
            std = [s if s > 0 else 1.0 for s in std]
            meta["mean"] = mean
            meta["std"] = std
        meta["dataset"] = self.dataset
        meta["n_rows"] = len(merged)
        meta["pending_appends"] = 0
        meta["built_at"] = datetime.now().isoformat(timespec="seconds")
        if "rep" not in meta and "rep" in merged.columns and len(merged):
            meta["rep"] = str(merged["rep"].iloc[0])
        if "vector_basis" not in meta:
            meta["vector_basis"] = VECTOR_BASIS_RAW
        if self.raw_only and "raw_format" not in meta:
            meta["raw_format"] = self.raw_format
        self.meta_path.write_text(json.dumps(meta, indent=2))
        self._clear_pending()
        return len(merged)

    # -------------------------------------------------- temp staging
    def temp_cache_dir(self) -> Path:
        """Transient raw-skeleton staging directory (crash-resume).

        Files here are written as fetch batches arrive and deleted once each
        batch is fully persisted; stale files from a crash are reprocessed
        (never re-fetched) on the next run.  Lives OUTSIDE ``skeleton_dir``
        (``skeletons/_temp_cache`` vs ``skeletons/raw_skeletons``) so the
        cache discovery never treats staging files as cached skeletons.
        """
        return (self.project_root / "cache" / _dataset_folder(self.dataset)
                / "skeletons" / "_temp_cache")

    def write_temp_skeleton(self, body_id, neuron) -> Path:
        """Write a raw (level-0) skeleton into the staging dir.

        Uses the fast zstd-3 codec (measured ~2.8x faster than level 19 in
        Step 0) so the staging stage keeps up with the fetch rate even for
        the densest neurons; the standard loader decompresses any level.
        Atomic per file - a crash never leaves a partial staging entry.
        """
        path = self.temp_cache_dir() / f"{body_id}.swc.zst"
        _write_compressed_skeleton(path, neuron, simplification=0,
                                   codec_level=3)
        return path

    def delete_temp_skeletons(self, body_ids) -> None:
        """Remove staging entries for fully-persisted neurons (best-effort)."""
        temp_dir = self.temp_cache_dir()
        for body_id in body_ids:
            try:
                (temp_dir / f"{body_id}.swc.zst").unlink(missing_ok=True)
            except OSError:
                continue

    def _discover_skeleton_files(self) -> List[str]:
        """All cached skeleton/mesh files, including nested bulk folders."""
        directories = [self.skeleton_dir]
        if (self.raw_only or self.mesh_only) \
                and self.legacy_skeleton_dir is not None:
            directories.append(self.legacy_skeleton_dir)
        directories = [directory for directory in directories
                       if directory.exists()]
        if not directories:
            return []
        files = []
        for directory in directories:
            files.extend(directory.rglob("*.pkl.zst"))
            files.extend(directory.rglob("*.pkl"))
            files.extend(directory.rglob("*.swc.zst"))
            files.extend(directory.rglob("*.swc.gz"))
        if not self.raw_only:
            files = [path for path in files
                     if "raw_skeletons" not in path.parts]
        # Transient crash-resume staging files are never cached skeletons.
        files = [path for path in files if "_temp_cache" not in path.parts]
        preferred = {}
        for path in files:
            try:
                body_id = _skeleton_body_id(path)
            except (TypeError, ValueError):
                continue
            current = preferred.get(body_id)
            # Prefer the new shared path over the legacy Find Similar path,
            # then prefer the canonical compressed representation for the
            # cache type (zstd SWC for raw skeletons, Zstandard for meshes).
            if str(path).endswith(".swc.zst"):
                format_rank = 0
            elif str(path).endswith(".swc.gz"):
                format_rank = 1
            elif str(path).endswith(".pkl.zst"):
                format_rank = 2
            else:
                format_rank = 3
            path_rank = (
                0 if self.skeleton_dir in path.parents else 1,
                format_rank,
            )
            current_path = Path(current) if current is not None else None
            if current_path is not None and str(current_path).endswith(".swc.zst"):
                current_format_rank = 0
            elif current_path is not None and str(current_path).endswith(".swc.gz"):
                current_format_rank = 1
            elif current_path is not None and str(current_path).endswith(".pkl.zst"):
                current_format_rank = 2
            else:
                current_format_rank = 3
            current_rank = (
                0 if self.skeleton_dir in current_path.parents else 1,
                current_format_rank,
            ) if current_path is not None else None
            if current is None or path_rank < current_rank:
                preferred[body_id] = path
        return sorted(str(path) for path in preferred.values())

    def _cached_dir_listing(self, directory: Path):
        """Flat file names + subdirectory flag of a cache directory.

        Cached per directory mtime: bulk pulls write thousands of files, and
        a per-id rglob would rescan the whole tree on every cache miss
        (O(N^2) across a full-dataset pull).  Returns ``(names, has_subdirs)``.
        """
        cache = getattr(self, "_dir_listing_cache", None)
        if cache is None:
            cache = {}
            self._dir_listing_cache = cache
        try:
            mtime = directory.stat().st_mtime_ns
        except OSError:
            return set(), False
        entry = cache.get(directory)
        if entry is not None and entry[0] == mtime:
            return entry[1], entry[2]
        names = set()
        has_subdirs = False
        try:
            with os.scandir(directory) as iterator:
                for item in iterator:
                    if item.is_dir(follow_symlinks=False):
                        has_subdirs = True
                    else:
                        names.add(item.name)
        except OSError:
            pass
        cache[directory] = (mtime, names, has_subdirs)
        return names, has_subdirs

    def find_skeleton_file(self, body_id: Union[int, str]) -> Optional[Path]:
        """Find a skeleton belonging to this cache namespace.

        The raw cache must never fall back to the shared simp90 pickle files.
        Its canonical location is the shared dataset skeleton namespace;
        historical Find Similar raw files are accepted only as a migration
        fallback.
        """
        body_id = self._canonical_body_id(body_id)
        if self.raw_only or self.mesh_only:
            directories = [self.skeleton_dir]
            if self.legacy_skeleton_dir is not None:
                directories.append(self.legacy_skeleton_dir)
            # Prefer the new shared path. Within a mesh cache prefer the
            # canonical compressed pickle; within a raw cache prefer zstd SWC.
            if self.mesh_only:
                names = (
                    f"{body_id}.pkl.zst",
                    f"{body_id}.pkl",
                )
            else:
                names = (
                    f"{body_id}.swc.zst",
                    f"{body_id}.swc.gz",
                    f"{body_id}.pkl",
                    f"{body_id}.pkl.zst",
                )
            for directory in directories:
                if not directory.exists():
                    continue
                _, has_subdirs = self._cached_dir_listing(directory)
                for name in names:
                    direct = directory / name
                    if direct.exists():
                        return direct
                    # Nested bulk folders exist only when the directory has
                    # subdirectories; skip the per-id rglob otherwise (it
                    # rescans the whole tree on every cache miss).
                    if has_subdirs:
                        nested = sorted(directory.rglob(name))
                        if nested:
                            return nested[0]
            return None
        return _find_skeleton_file(
            self.dataset, body_id, project_root=str(self.project_root)
        )

    def load_skeleton(self, body_id: int,
                      simplification: Optional[int] = None):
        """Load one valid cached neuron from this cache namespace.

        ``simplification`` optionally requests a target level (percent of
        nodes removed); when it is coarser than the stored level a
        TreeNeuron is re-simplified on load.  ``None`` (default) returns the
        neuron at its stored level.  MeshNeurons are never re-leveled (the
        simplification pipeline is NeuPrint/TreeNeuron-only).
        """
        path = self.find_skeleton_file(body_id)
        if path is None:
            return None
        try:
            # Load at the STORED level first: a lazy migration below must
            # persist the as-stored neuron (with its recorded level), never a
            # re-leveled copy that would mismatch its header.
            neuron = _load_cached_skeleton_file(path)
            if self.mesh_only:
                allowed = ("MeshNeuron",)
            elif self.raw_only:
                allowed = ("TreeNeuron",)
            else:
                allowed = ("TreeNeuron", "MeshNeuron")
            if type(neuron).__name__ not in allowed:
                return None
            if (self.mesh_only
                    and (self.skeleton_dir not in path.parents
                         or str(path).endswith(".pkl"))):
                self.persist_skeletons({self._canonical_body_id(body_id): neuron})
            if (self.raw_only and self.raw_format in ("swc.gz", "swc.zst")
                    and (self.skeleton_dir not in path.parents
                         or not str(path).endswith(f".{self.raw_format}"))):
                # Migrate legacy Find Similar files, and any temporary raw
                # file, to the canonical compressed-SWC namespace on first
                # successful load. The old file is intentionally kept
                # recoverable as a non-destructive fallback, and the stored
                # simplification level is preserved (never re-simplified).
                self.persist_skeletons(
                    {self._canonical_body_id(body_id): neuron},
                    simplification=None)
            if simplification is not None \
                    and type(neuron).__name__ == "TreeNeuron":
                stored_level = getattr(neuron, "_drocat_simplification", 0)
                neuron = _relevel_for_target(
                    neuron, stored_level, simplification)
            return neuron
        except Exception:
            return None

    def persist_skeletons(self, neurons: Dict[Union[int, str], object],
                          simplification: Optional[int] = DEFAULT_SIMPLIFICATION
                          ) -> int:
        """Persist neurons in this cache's skeleton namespace.

        Shared raw caches store the fetched TreeNeuron through the shared
        simplify + compress pipeline: the requested ``simplification`` level
        (percent of nodes removed, 0-90; default 90) is applied and recorded
        in each ``.swc.zst`` file header.  ``simplification=None`` writes the
        neuron as-is, recording the level already attached to it (used by
        lazy migrations).  Legacy pickle files remain readable and selectable
        with ``raw_format='pkl'``. The legacy visualization cache continues
        to use its explicit downsampling path and does not call this helper.
        """
        if not neurons:
            return 0
        self.skeleton_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        manifest = self._load_skeleton_manifest() if self.raw_only else None
        manifest_changed = False
        mesh_cache = None
        if self.mesh_only:
            mesh_cache = FlyWireMeshCache(
                self.dataset,
                project_root=self.project_root,
                simplification=FLYWIRE_MESH_CACHE_SIMPLIFICATION,
                soma_simplification=FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION,
                soma_radius=FLYWIRE_MESH_CACHE_SOMA_RADIUS,
            )
        for body_id, neuron in neurons.items():
            try:
                body_id = self._canonical_body_id(body_id)
                if self.mesh_only and _neuron_rep(neuron) != "mesh":
                    continue
                if self.raw_only and _neuron_rep(neuron) != "skeleton":
                    continue
                if self.mesh_only:
                    written += mesh_cache.save({body_id: neuron})
                elif self.raw_only and self.raw_format == "swc.zst":
                    path = self.skeleton_dir / f"{body_id}.swc.zst"
                    _write_compressed_skeleton(
                        path, neuron,
                        simplification=simplification)
                    written += 1
                    stored_level = (
                        getattr(neuron, "_drocat_simplification", 0)
                        if simplification is None else simplification
                    ) or 0
                elif self.raw_only and self.raw_format == "swc.gz":
                    path = self.skeleton_dir / f"{body_id}.swc.gz"
                    _write_compressed_swc(
                        path, neuron)
                    written += 1
                    stored_level = 0
                else:
                    path = self.skeleton_dir / f"{body_id}.pkl"
                    with open(path, "wb") as handle:
                        pickle.dump(neuron, handle)
                    written += 1
                    stored_level = getattr(neuron, "_drocat_simplification", 0) or 0
                if self.raw_only and manifest is not None:
                    manifest["files"][str(body_id)] = {
                        "file": path.name,
                        "simplification": int(stored_level),
                        "representation": "skeleton",
                        "source": "neuprint.fetch_skeleton",
                        "coordinate_units": "nm",
                        "vector_basis": VECTOR_BASIS_RAW,
                        "updated_at": datetime.now().isoformat(
                            timespec="seconds"),
                    }
                    manifest_changed = True
            except Exception:
                continue
        if manifest_changed and manifest is not None:
            self._write_skeleton_manifest(manifest)
        return written

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
            "raw_format": self.raw_format if self.raw_only else None,
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
        additional neurons (persisted to the skeleton cache first). For
        FAFB v783 the full-dataset source is the healed skeleton bundle
        (``{bodyId}.swc`` entries — the local pickle cache holds meshes,
        which is the wrong representation for the vector cache).
        """
        self.morph_dir.mkdir(parents=True, exist_ok=True)
        self.skeleton_dir.mkdir(parents=True, exist_ok=True)

        # FAFB v783: the healed bundle is the full skeleton source (.zst
        # first; ZIP fallback with lazy conversion).
        use_bundle = False
        bundle_source = None
        bundle_ids: List[Union[int, str]] = []
        if not self.mesh_only and is_fafb_dataset(self.dataset):
            try:
                bundle_source = _fafb_bundle(
                    self.dataset, str(self.project_root))
            except Exception:
                bundle_source = None
            if bundle_source is not None:
                bundle_ids = sorted(
                    {
                        normalize_flywire_body_id(b)
                        if self._flywire_ids else int(b)
                        for b in bundle_source.ids()
                    }
                )
                use_bundle = True

        existing: Dict[int, dict] = {}
        if self.parquet_path.exists():
            try:
                df_old = pd.read_parquet(self.parquet_path)
                if use_bundle:
                    # The bundle produces skeleton vectors; a mesh-based
                    # (or legacy, rep-less) cache is incompatible and must
                    # be rebuilt from scratch.
                    reps = (set(df_old["rep"].fillna("").astype(str))
                            if "rep" in df_old.columns else {"legacy"})
                    if not reps <= {"skeleton", ""}:
                        self._log(
                            "[SkeletonVectorCache] Existing FAFB vector cache "
                            "is mesh-based; rebuilding it from the healed "
                            "skeleton bundle.")
                        df_old = None
                if df_old is not None:
                    existing = {
                        self._canonical_body_id(r["bodyId"]): r
                        for r in df_old.to_dict("records")
                    }
            except Exception:
                existing = {}
        # Fold any pending (appended-but-unmerged) rows into the rebuild so
        # they are never dropped; the pending file is cleared once the new
        # main parquet is written below.
        if self.pending_path.exists():
            try:
                pending_df = pd.read_parquet(self.pending_path)
                for r in pending_df.to_dict("records"):
                    existing.setdefault(
                        self._canonical_body_id(r["bodyId"]), r)
            except Exception:
                pass

        # The cache holds ONE vectorization level (its "basis"). On-disk
        # skeletons are vectorized only when their simplification level
        # matches that basis: post-cleanup NeuPrint caches hold simp90
        # files while the basis is raw, so those files are skipped (their
        # vectors come from the vector cache / raw fetches instead).
        basis = (VECTOR_BASIS_RAW if self.raw_only else
                 ((self._load_meta() or {}).get("vector_basis")
                  or VECTOR_BASIS_RAW))
        folder_level = (VECTOR_BASIS_RAW if self.raw_only else
                         _skeleton_folder_level(self.dataset,
                                                 str(self.project_root)))

        # Candidate skeletons not yet vectorized.
        if use_bundle:
            files: List[str] = []
            pending = [
                self._canonical_body_id(b)
                for b in bundle_ids
                if self._canonical_body_id(b) not in existing
            ]
        else:
            files = self._discover_skeleton_files()
            files = [f for f in files if folder_level == basis]
            pending = [
                f for f in files
                if self._canonical_body_id(_skeleton_body_id(f)) not in existing
            ]

        # Optional on-demand fetch to extend coverage (cap applies).
        # fetch_skeleton_on_demand already vectorizes at fetch time (raw
        # basis) and persists the vector; the fetched row set is refreshed
        # below so those rows are not dropped by the merge.
        fetched_new = 0
        if fetch_missing and fetch_missing > 0:
            index_path = self.project_root / "neuron_indexes" / _dataset_folder(self.dataset) / "neuron_index.parquet"
            index: List[int] = []
            if index_path.exists() and _has_local_dataset_presence(
                self.dataset, Path(self.project_root)
            ):
                try:
                    idx_df = pd.read_parquet(index_path, columns=["bodyId"])
                    index = (
                        [normalize_flywire_body_id(b)
                         for b in idx_df["bodyId"].tolist()]
                        if self._flywire_ids else
                        [int(b) for b in idx_df["bodyId"].tolist()]
                    )
                except Exception:
                    index = []
            if index:
                have = {self._canonical_body_id(b) for b in list(existing)}
                if use_bundle:
                    have |= set(bundle_ids)
                else:
                    have |= {
                        self._canonical_body_id(_skeleton_body_id(f))
                        for f in files
                    }
                missing = [b for b in index if b not in have]
                fetched_map = fetch_skeletons_on_demand_batch(
                    self.dataset,
                    missing[:fetch_missing],
                    project_root=str(self.project_root),
                    persist=True,
                    level=VECTOR_BASIS_RAW,
                    raw_cache=self if (self.raw_only or self.mesh_only) else None,
                    vector_cache=self if (self.raw_only or self.mesh_only) else None,
                )
                fetched_new = len(fetched_map)
            if fetched_new:
                # Re-discover after the fetches: they wrote new skeleton
                # files (and, in the real pipeline, already appended the
                # raw vectors). Refresh the row set and the pending files
                # so neither the fetched vectors nor the on-disk files are
                # dropped by the merge below.
                try:
                    df_old = pd.read_parquet(self.parquet_path)
                    existing = {
                        self._canonical_body_id(r["bodyId"]): r
                        for r in df_old.to_dict("records")
                    }
                except Exception:
                    pass
                if use_bundle:
                    pending = [
                        self._canonical_body_id(b)
                        for b in bundle_ids
                        if self._canonical_body_id(b) not in existing
                    ]
                else:
                    files = self._discover_skeleton_files()
                    files = [f for f in files if folder_level == basis]
                    pending = [
                        f for f in files
                        if self._canonical_body_id(_skeleton_body_id(f))
                        not in existing
                    ]

        rows = []
        if pending:
            started = time.time()
            source_label = ("healed bundle skeletons" if use_bundle
                            else "skeletons")
            self._log(
                f"[SkeletonVectorCache] Vectorizing {len(pending)} {source_label} "
                f"({self.dataset})..."
            )
            if use_bundle:
                source_path = str(bundle_source.bundle_path)
                zip_path = (str(bundle_source.zip_path)
                            if bundle_source.zip_path else None)
                if self.n_workers > 1:
                    try:
                        rows = self._vectorize_parallel_swc(
                            source_path, zip_path, pending)
                    except Exception:
                        rows = self._vectorize_swc_serial(
                            source_path, zip_path, pending)
                else:
                    rows = self._vectorize_swc_serial(
                        source_path, zip_path, pending)
            elif self.n_workers > 1:
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
            rep = ("skeleton" if self.raw_only else
                   Counter(r[3] for r in ok_rows).most_common(1)[0][0])
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
            row["bodyId"] = self._canonical_body_id(bid)
            records.append(row)
        for row in rows:
            if row is None:
                continue
            bid, morph_vals, pv_vals, row_rep = row
            record = {
                "bodyId": self._canonical_body_id(bid),
                "rep": row_rep,
            }
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

        df = pd.DataFrame(records)
        if self._flywire_ids:
            normalize_flywire_id_columns(df, ["bodyId"])
        df = df.sort_values("bodyId").reset_index(drop=True)

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

        self._atomic_parquet(df, self.parquet_path)

        # Z-score stats over the population.
        mat = self._raw_matrix(df)
        mean = mat.mean(axis=0).tolist()
        std = mat.std(axis=0).tolist()
        std = [s if s > 0 else 1.0 for s in std]
        self._write_meta({"mean": mean, "std": std}, len(df), rep=rep,
                         vector_basis=basis)
        # A full rebuild supersedes the append-only staging file.
        self._clear_pending()

        self._log(
            f"[SkeletonVectorCache] Cache ready: {len(df)} rows "
            f"({len(rows)} new, {fetched_new} fetched) -> {self.parquet_path}"
        )
        if bundle_source is not None:
            try:
                bundle_source.close()
            except Exception:
                pass
        return {"rows": len(df), "new": len(rows), "fetched": fetched_new}

    def _vectorize_parallel(self, files: List[str]) -> List[Tuple[int, List[float], List[float]]]:
        ctx = mp.get_context("fork") if hasattr(mp, "get_context") and "fork" in mp.get_all_start_methods() else mp.get_context()
        with ProcessPoolExecutor(max_workers=self.n_workers, mp_context=ctx) as ex:
            return list(ex.map(_vectorize_one_file, files, chunksize=16))

    def _vectorize_parallel_swc(self, source_path: str,
                                zip_path: Optional[str], bids: List[int]
                                ) -> List[Tuple[int, List[float], List[float]]]:
        """Vectorize healed-bundle skeletons in a worker pool; each worker
        opens the bundle (.zst first, ZIP fallback) via the initializer."""
        ctx = mp.get_context("fork") if hasattr(mp, "get_context") and "fork" in mp.get_all_start_methods() else mp.get_context()
        with ProcessPoolExecutor(max_workers=self.n_workers, mp_context=ctx,
                                 initializer=_init_fafb_zip_worker,
                                 initargs=(source_path, zip_path)) as ex:
            return list(ex.map(_vectorize_one_swc, bids, chunksize=16))

    def _vectorize_swc_serial(self, source_path: str,
                              zip_path: Optional[str], bids: List[int]
                              ) -> List[Tuple[int, List[float], List[float]]]:
        """Serial healed-bundle vectorization (single-worker or fallback)."""
        global _FAFB_WORKER_BUNDLE
        _init_fafb_zip_worker(source_path, zip_path)
        try:
            return [_vectorize_one_swc(b) for b in bids]
        finally:
            if _FAFB_WORKER_BUNDLE is not None:
                _FAFB_WORKER_BUNDLE.close()
                _FAFB_WORKER_BUNDLE = None

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
        Main + pending rows are merged with a first-wins dedupe by bodyId, so
        crash-duplicate appends are never visible to consumers.
        """
        if not self.parquet_path.exists() and not self.pending_path.exists():
            return None
        df = pd.DataFrame()
        if self.parquet_path.exists():
            try:
                df = pd.read_parquet(self.parquet_path)
            except Exception:
                df = pd.DataFrame()
        if self.pending_path.exists():
            try:
                pending_df = pd.read_parquet(self.pending_path)
                df = self._dedupe_frames(df, pending_df)
            except Exception:
                pass
        if df.empty:
            return None
        if self._flywire_ids:
            normalize_flywire_id_columns(df, ["bodyId"])
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
        body_ids = (
            df["bodyId"].astype("string").to_numpy(dtype=object)
            if self._flywire_ids else
            df["bodyId"].astype(np.int64).to_numpy()
        )
        return {
            "meta": meta,
            "df": df,
            "raw": raw,
            "X": X,
            "bodyIds": body_ids,
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
        return {"rows": self._vector_row_count(), "new": 0, "fetched": 0}

    def coverage(self) -> Dict[str, int]:
        """Skeleton and vector counts for the dataset.

        For FAFB v783 the local skeleton count is the healed bundle's entry
        count (the pickle cache holds meshes, not skeletons).
        """
        n_skeletons = len(self._discover_skeleton_files())
        if is_fafb_dataset(self.dataset):
            try:
                bundle = _fafb_bundle(self.dataset, str(self.project_root))
            except Exception:
                bundle = None
            if bundle is not None:
                try:
                    n_skeletons = bundle.count()
                finally:
                    bundle.close()
            else:
                zip_path = _fafb_skeleton_zip_path(
                    self.dataset, str(self.project_root))
                if zip_path is not None:
                    try:
                        import zipfile
                        with zipfile.ZipFile(zip_path, "r") as z:
                            n_skeletons = sum(
                                1 for n in z.namelist() if n.endswith(".swc"))
                    except Exception:
                        pass
        n_vectors = self._vector_row_count()
        return {"skeletons": n_skeletons, "vectors": n_vectors}

    # ------------------------------------------------------------ append
    def append_vectors(self, records: List[Tuple[int, np.ndarray, str]],
                       vector_basis: str = VECTOR_BASIS_RAW) -> int:
        """Persist freshly-computed vectors (raw feature rows) into the cache.

        Called when a vector was computed from a cached skeleton file or from
        an online-fetched skeleton that was NOT persisted: the VECTOR is
        stored so later queries reuse it without re-fetching or
        re-vectorizing, even though the original skeleton stays uncached.

        Append-only: the rows are written to the small ``*_pending.parquet``
        staging file (O(batch); atomic) and folded into the main parquet by
        the amortized merge checkpoint once a threshold is crossed.  Dedupe
        by bodyId is deferred to read/merge time (first occurrence wins), so
        crash-duplicate appends are never visible to consumers.  The cache's
        standardization statistics (meta mean/std) are left untouched, so the
        standardized space stays consistent across appends.  Rows of a
        representation different from the cache's are rejected (a cache
        holds ONE level), and rows whose ``vector_basis`` differs from the
        cache's basis are rejected too (a cache holds ONE simplification
        level).  Returns the number of rows appended to pending (may
        over-count by rows that dedupe against existing data at merge).
        """
        if not records:
            return 0
        # Cross-process safety: UI runs execute in separate subprocesses, so
        # the pending read-append-write is guarded with an advisory file lock
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
            seen = set()
            for bid, vec, rep in records:
                if self.raw_only and str(rep) != "skeleton":
                    continue
                bid = self._canonical_body_id(bid)
                if bid in seen:
                    continue
                seen.add(bid)
                row = {"bodyId": bid, "rep": rep}
                for i, name in enumerate(MORPHOMETRIC_FEATURES):
                    row[name] = float(vec[i])
                for i in range(PERSISTENCE_DIM):
                    row[f"pv_{i}"] = float(vec[len(MORPHOMETRIC_FEATURES) + i])
                row["type"] = type_map.get(bid, "") if type_map else ""
                row["instance"] = (instance_map or {}).get(bid, "") if instance_map else ""
                rows_new.append(row)
            if not rows_new:
                return 0
            df_new = pd.DataFrame(rows_new)

            # A cache holds ONE basis and ONE representation; the meta
            # records them once they exist (build or first append).
            meta = self._load_meta() or {}
            cache_basis = meta.get("vector_basis") or VECTOR_BASIS_RAW
            if meta.get("vector_basis") and cache_basis != vector_basis:
                # Different simplification level: never mix (a cache
                # holds ONE basis).
                return 0
            cache_rep = meta.get("rep") or ""
            if cache_rep:
                df_new = df_new[df_new["rep"] == cache_rep]
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

            # Append to the pending staging file (O(pending + batch), atomic).
            pending_df = pd.DataFrame()
            if self.pending_path.exists():
                try:
                    pending_df = pd.read_parquet(self.pending_path)
                except Exception:
                    pending_df = pd.DataFrame()
            combined = self._dedupe_frames(pending_df, df_new)
            self._atomic_parquet(combined, self.pending_path)

            meta["dataset"] = self.dataset
            meta["n_rows"] = (self._vector_row_count()
                               if self.parquet_path.exists() else len(combined))
            meta["pending_appends"] = int(meta.get("pending_appends") or 0) + 1
            meta["built_at"] = datetime.now().isoformat(timespec="seconds")
            if "rep" not in meta and len(df_new) and "rep" in df_new.columns:
                meta["rep"] = str(df_new["rep"].iloc[0])
            # Record the basis when creating the cache; existing caches keep
            # their own basis (enforced by the check above).
            if "vector_basis" not in meta:
                meta["vector_basis"] = vector_basis
            if self.raw_only and "raw_format" not in meta:
                meta["raw_format"] = self.raw_format
            self.meta_path.write_text(json.dumps(meta, indent=2))

            # Amortized merge checkpoint: fold pending into main once either
            # threshold is crossed (read-modify-write of the whole main file
            # happens rarely, never per append).
            if (len(combined) >= PENDING_MERGE_ROWS
                    or int(meta.get("pending_appends") or 0)
                    >= PENDING_MERGE_APPENDS):
                self._merge_pending()
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
        body_ids = [self._canonical_body_id(b) for b in body_ids]
        data = self.load()
        known: Dict[int, int] = {}
        X = np.zeros((0, VECTOR_DIM))
        dataset_rep = ""
        basis = VECTOR_BASIS_RAW
        if data is not None:
            known = {
                self._canonical_body_id(b): i
                for i, b in enumerate(data["bodyIds"])
            }
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
                if (self.raw_only or self.mesh_only or
                        _skeleton_folder_level(
                            self.dataset, str(self.project_root)) == basis):
                    pkl = self.find_skeleton_file(bid)
                if pkl is not None:
                    try:
                        neuron = _load_cached_skeleton_file(pkl)
                        # Per-file level guard: simplified on-disk skeletons
                        # are never vectorized into the raw-basis cache (their
                        # vectors come from the vector cache / raw fetches).
                        if getattr(neuron, "_drocat_simplification", 0) != 0:
                            continue
                        row_rep = _neuron_rep(neuron)
                        if self.raw_only and row_rep != "skeleton":
                            continue
                        if self.mesh_only and row_rep != "mesh":
                            continue
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


def find_similar_raw_cache(dataset: str,
                           project_root: Optional[str] = None,
                           n_workers: int = 8,
                           verbose: bool = True,
                           raw_format: str = "swc.zst") -> SkeletonVectorCache:
    """Return the shared raw-skeleton/vector cache for a dataset.

    The historical function name is retained for compatibility with callers,
    but raw skeletons now live outside the Find Similar directory at
    ``cache/{dataset}/skeletons/raw_skeletons/``. New persisted raw skeletons
    use zstd-compressed SWC by default; pass ``raw_format='pkl'`` or
    ``raw_format='swc.gz'`` only for legacy compatibility.
    """
    return SkeletonVectorCache(
        dataset,
        project_root=project_root,
        n_workers=n_workers,
        verbose=verbose,
        raw_only=True,
        raw_format=raw_format,
    )


def find_similar_flywire_mesh_cache(
        dataset: str,
        project_root: Optional[str] = None,
        n_workers: int = 8,
        verbose: bool = True,
        ) -> SkeletonVectorCache:
    """Return the separate prepared-mesh/vector cache for FlyWire."""
    if not is_flywire_dataset(dataset):
        raise ValueError("FlyWire mesh cache requires a FlyWire/FAFB dataset")
    return SkeletonVectorCache(
        dataset,
        project_root=project_root,
        n_workers=n_workers,
        verbose=verbose,
        representation="mesh",
    )


def find_similar_dataset_cache(
        dataset: str,
        project_root: Optional[str] = None,
        n_workers: int = 8,
        verbose: bool = True,
        ) -> SkeletonVectorCache:
    """Return the dataset-native vector/cache manager.

    NeuPrint owns the raw SWC manager; FlyWire owns the prepared mesh
    manager. This boundary prevents a CAVE MeshNeuron from being serialized
    through the NeuPrint SWC writer.
    """
    if is_flywire_dataset(dataset):
        return find_similar_flywire_mesh_cache(
            dataset, project_root=project_root, n_workers=n_workers,
            verbose=verbose)
    return find_similar_raw_cache(
        dataset, project_root=project_root, n_workers=n_workers,
        verbose=verbose)


def cache_fetched_skeleton_vectors(
        dataset: str, neurons, project_root: Optional[str] = None,
        vector_cache: Optional[SkeletonVectorCache] = None,
        progress_callback=None, progress_offset: int = 0,
        progress_total: Optional[int] = None, verbose: bool = False
        ) -> Dict[str, object]:
    """Vectorize and persist freshly fetched skeletons or FlyWire meshes.

    This is the cache transaction shared by the morphology batch fetcher and
    the visualizer's own NeuPrint batch loop.  The function returns only after
    the parquet append has completed, so a caller cannot enter layered
    rendering (or similarity scoring) while the vector cache is still being
    written.  ``progress_callback`` uses the same ``(done, total, message)``
    contract as the online skeleton fetch.

    Cache failures are reported explicitly and do not discard the fetched
    neurons: visualization remains usable even if a read-only cache directory
    is encountered. The returned ``cached`` count makes that condition
    observable to callers and tests.
    """
    if isinstance(neurons, dict):
        items = list(neurons.items())
    else:
        items = []
        for index, neuron in enumerate(list(neurons or [])):
            body_id = getattr(neuron, "id", None)
            if body_id is None:
                body_id = index
            items.append((body_id, neuron))

    total = int(progress_total if progress_total is not None
               else progress_offset + len(items))
    if progress_callback:
        progress_callback(
            progress_offset, total,
            f"Vectorizing fetched skeletons (0/{len(items)})")

    started = time.perf_counter()
    rows = []
    failures = 0
    for index, (body_id, neuron) in enumerate(items, start=1):
        try:
            rep = _neuron_rep(neuron)
            if rep not in {"skeleton", "mesh"}:
                raise TypeError(
                    f"expected TreeNeuron or MeshNeuron, got {type(neuron).__name__}")
            _, vector = vectorize_neuron(neuron)
            rows.append((
                _canonical_dataset_body_id(dataset, body_id),
                vector,
                rep,
            ))
        except Exception as exc:
            failures += 1
            if verbose:
                print(f"[morphology] vectorization skipped for {body_id}: {exc}")
        if progress_callback:
            progress_callback(
                progress_offset + index, total,
                f"Vectorizing fetched skeletons ({index}/{len(items)})")

    cached = 0
    cache_error = None
    if rows:
        if vector_cache is not None:
            target = vector_cache
        elif all(row[2] == "mesh" for row in rows):
            target = find_similar_flywire_mesh_cache(
                dataset, project_root=project_root, verbose=False)
        else:
            target = SkeletonVectorCache(
                dataset, project_root=project_root, verbose=False)
        try:
            cached = int(target.append_vectors(
                rows, vector_basis=VECTOR_BASIS_RAW))
        except Exception as exc:
            cache_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"[morphology] vector cache write failed: {cache_error}")

    elapsed = time.perf_counter() - started
    if progress_callback:
        progress_callback(
            progress_offset + len(items), total,
            f"Vector cache complete ({cached}/{len(rows)} rows; "
            f"{elapsed:.2f}s)")
    return {
        "seen": len(items),
        "vectorized": len(rows),
        "cached": cached,
        "failures": failures,
        "elapsed": elapsed,
        "cache_error": cache_error,
    }


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
                     max_sample: int = POPULATION_STATS_SAMPLE,
                     cache: Optional[SkeletonVectorCache] = None
                     ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Population mean/std for a dataset's cached skeletons.

    Stable standardization statistics used when a dataset has no vector
    cache: pool-only statistics depend on the (connectivity-skewed) pool
    composition and distort the geometry between query and candidates. The
    stats are computed once from a bounded sample of cached skeletons and
    persisted under the selected cache namespace's
    ``morphology/population_stats.json`` for reuse. The legacy visualization
    cache may extend a sparse sample with a version sibling; the raw-only
    cache never borrows simplified/shared skeletons.
    Returns (None, None) when no statistics can be estimated.
    """
    root = Path(project_root) if project_root else Path(__file__).parent.parent
    vc = cache or SkeletonVectorCache(dataset, project_root=str(root),
                                      verbose=False)
    stats_file = vc.morph_dir / "population_stats.json"
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

    files = vc._discover_skeleton_files()

    # Level guard: the statistics must match the vector basis (raw). Once
    # raw skeletons are replaced by the simplified cache (NeuPrint), the
    # on-disk sample can no longer be vectorized at the right level; fall
    # back to the vector cache's own raw meta stats, which were computed
    # from the same feature schema at fetch time.
    basis = (vc._load_meta() or {}).get("vector_basis") or VECTOR_BASIS_RAW
    if (not vc.raw_only and
            _skeleton_folder_level(dataset, str(root)) != basis):
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

    # A raw comparison may have vector rows but no persisted raw files (for
    # example after an interrupted run). Its metadata is still a valid
    # raw-vector standardization baseline once the cache is large enough;
    # never borrow the shared visualization cache or a sibling's simplified
    # files.
    if vc.raw_only and len(files) == 0:
        data = vc.load()
        if data is not None and len(data["bodyIds"]) >= MIN_POPULATION_STATS_SKELETONS:
            meta = data.get("meta") or {}
            m, s = meta.get("mean"), meta.get("std")
            if m is not None and s is not None:
                mm, ss = np.asarray(m, dtype=float), np.asarray(s, dtype=float)
                if mm.shape == (VECTOR_DIM,) and ss.shape == (VECTOR_DIM,):
                    return mm, ss

    # Too few cached skeletons for stable stats: sample from the version
    # sibling's cache instead — it contains the same neurons (shared
    # reconstruction, e.g. male-cns v1.0 <- v0.9), and the sparse local
    # cache may be morphologically skewed (e.g. one query's transient
    # fetches), which would bias the statistics. Only a LARGER sibling
    # cache is used (the sibling may itself be the sparse one).
    if len(files) < MIN_POPULATION_STATS_SKELETONS and not vc.raw_only:
        sibling_files: List[str] = []
        for skel_dir in _sibling_skeleton_dirs(dataset, root):
            sf = sorted(
                [str(p) for p in skel_dir.rglob("*.pkl")
                 if "raw_skeletons" not in p.parts]
                + [str(p) for p in skel_dir.rglob("*.pkl.zst")
                   if "raw_skeletons" not in p.parts]
                + [str(p) for p in skel_dir.rglob("*.swc.gz")
                   if "raw_skeletons" not in p.parts]
            )
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
        stats_file.parent.mkdir(parents=True, exist_ok=True)
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
        token = token_manager.get_neuprint_token()
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


def _fetch_cave_skeleton(dataset: str, body_id: int,
                         project_root: Optional[str] = None,
                         use_cache: bool = True):
    """Legacy compatibility fetch; production FlyWire uses ``_fetch_cave_mesh``."""
    from cave_data_fetcher import CAVEDataFetcher
    fetcher = CAVEDataFetcher(
        dataset=_dataset_folder(dataset), project_root=project_root,
        verbose=False,
    )
    return fetcher.fetch_skeleton(
        body_id_to_api_int(body_id), use_cache=use_cache,
        simplify_mesh=0.0, denoise_twigs=None,
    )


def _fetch_cave_mesh(dataset: str, body_id: int,
                     project_root: Optional[str] = None,
                     use_cache: bool = True,
                     soma_pos=None):
    """Fetch one prepared MeshNeuron from a FlyWire/CAVE dataset."""
    from cave_data_fetcher import CAVEDataFetcher
    fetcher = CAVEDataFetcher(
        dataset=_dataset_folder(dataset), project_root=project_root,
        verbose=False,
    )
    return fetcher.fetch_fafb_mesh(
        body_id_to_api_int(body_id),
        use_cache=use_cache,
        simplify_mesh=FLYWIRE_MESH_CACHE_SIMPLIFICATION,
        soma_simplification=FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION,
        soma_radius=FLYWIRE_MESH_CACHE_SOMA_RADIUS,
        soma_pos=soma_pos,
    )


def fetch_skeleton_on_demand(dataset: str, body_id: int,
                             project_root: Optional[str] = None,
                             persist: bool = True,
                             level: str = VECTOR_BASIS_RAW,
                             raw_cache: Optional[SkeletonVectorCache] = None,
                             vector_cache: Optional[SkeletonVectorCache] = None,
                             soma_pos=None,
                             simplification: int = DEFAULT_SIMPLIFICATION
                             ) -> Optional[object]:
    """Fetch one dataset-native neuron if missing.

    NeuPrint datasets use ``neuprint.fetch_skeleton`` and persist raw
    ``TreeNeuron`` objects through the shared simplify + compress pipeline
    (``simplification`` percent removed, default 90, recorded in the
    ``.swc.zst`` header). FlyWire/FAFB datasets use the CAVE mesh path and
    persist prepared ``MeshNeuron`` objects in the representation-specific
    mesh cache. The two paths never share a file.

    Vectorization always runs on the RAW fetched neuron and is persisted to
    the standalone vector cache BEFORE the simplified on-disk file is
    written, so the on-disk simplification level never affects vectors.

    ``persist=False`` is an online-only compatibility escape hatch for the
    selected representation; it does not read or write local morphology data.

    ``level`` is retained for compatibility with older callers, but is
    deliberately ignored after validation for NeuPrint. FlyWire's prepared
    mesh cache has its fixed 95%/80% visualization preparation level.
    """
    body_id = _canonical_dataset_body_id(dataset, body_id)
    level = str(level).lower()
    if level not in (VECTOR_BASIS_RAW, VECTOR_BASIS_SIMP90):
        raise ValueError(f"Invalid level: {level} (raw|simp90)")
    # None is never accepted here: 0 (raw) is the explicit escape.
    _simplification_factor(simplification)
    # The simplification pipeline is NeuPrint-only: FlyWire/FAFB/BANC always
    # fetch mesh representations and must never be re-leveled or simplified.
    if is_flywire_dataset(dataset):
        simplification = 0
    root = Path(project_root) if project_root else Path(__file__).parent.parent

    if is_flywire_dataset(dataset):
        mesh = _fetch_cave_mesh(
            dataset,
            body_id,
            project_root=str(root),
            use_cache=bool(persist),
            soma_pos=soma_pos,
        )
        if mesh is not None and persist:
            cache_fetched_skeleton_vectors(
                dataset,
                {body_id: mesh},
                project_root=str(root),
                vector_cache=vector_cache,
                verbose=False,
            )
        return mesh

    # Every fetch-level caller uses the same raw cache, regardless of the
    # legacy level value supplied by an older visualization caller.
    if raw_cache is None:
        try:
            raw_cache = find_similar_raw_cache(
                dataset, project_root=str(root), verbose=False,
            )
        except Exception:
            raw_cache = None

    if raw_cache is not None:
        cached = raw_cache.load_skeleton(body_id, simplification=simplification)
        if cached is not None:
            # A raw-file hit should also populate the raw vector cache when
            # a previous interrupted run left only the skeleton behind.
            cache_fetched_skeleton_vectors(
                dataset, {body_id: cached}, project_root=str(root),
                vector_cache=vector_cache or raw_cache, verbose=False,
            )
            return cached

    dataset_l = dataset.lower()
    if any(k in dataset_l for k in ("flywire", "fafb", "banc")):
        try:
            neuron = _fetch_cave_skeleton(
                dataset, body_id, project_root=str(root), use_cache=persist,
            )
        except TypeError as exc:
            # Preserve older integrations that replaced the two-argument
            # CAVE seam; production uses the project-root-aware raw path.
            if not any(name in str(exc)
                       for name in ("project_root", "use_cache")):
                raise
            neuron = _fetch_cave_skeleton(dataset, body_id)
    else:
        neuron = _fetch_neuprint_skeleton(
            dataset, _api_dataset_body_id(dataset, body_id)
        )

    if neuron is None:
        return None

    # Vectorize the RAW skeleton at fetch time and persist the vector before
    # any simplification: the vector cache is standalone and always raw-basis.
    cache_fetched_skeleton_vectors(
        dataset, {body_id: neuron}, project_root=str(root),
        vector_cache=vector_cache or raw_cache, verbose=False,
    )

    if persist and raw_cache is not None:
        raw_cache.persist_skeletons({body_id: neuron},
                                    simplification=simplification)
    return neuron


# Keep a compatibility seam for callers/tests that intentionally replace the
# singular fetcher (for example, an offline fixture). Normal production calls
# never take this branch; the optimized path below remains the default.
_SINGLE_FETCH_IMPLEMENTATION = fetch_skeleton_on_demand


def _fetch_neuprint_batch_with_progress(
        batch_ids, *, client, max_threads, on_neuron=None,
        missing_swc: str = "warn") -> list:
    """Fetch one batch of NeuPrint SWCs with a per-neuron completion hook.

    Vendored from ``navis.interfaces.neuprint.fetch_skeletons`` (navis 1.5.0)
    so every completed skeleton can be reported: the upstream wrapper only
    updates its own hidden tqdm counter, which never reaches the UI.  The
    behavior is intentionally identical to navis: one ``fetch_neurons``
    metadata query per batch (soma/instance/status), then a
    ``ThreadPoolExecutor`` with ``max_threads`` workers fetching the
    individual SWCs concurrently, ``missing_swc`` warn/skip handling, and
    the same TreeNeuron construction (``__fetch_skeleton``).

    ``on_neuron(done, batch_total)`` is called from the caller's thread for
    every completed future (successful or failed) - exactly the per-neuron
    cadence of navis' internal bar.  Returns the fetched TreeNeurons.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from navis.interfaces import neuprint as neu
    from navis.interfaces.neuprint import __fetch_skeleton

    meta, _roi_info = neu.fetch_neurons(
        neu.NeuronCriteria(bodyId=list(batch_ids)), client=client)
    if meta is None or meta.empty:
        return []
    # Make sure there is a somaLocation and somaRadius column (navis does
    # the same before the row iteration below).
    if "somaLocation" not in meta.columns:
        meta["somaLocation"] = None
    if "somaRadius" not in meta.columns:
        meta["somaRadius"] = None

    neurons = []
    done = 0
    batch_total = len(meta)
    with ThreadPoolExecutor(max_workers=max(1, int(max_threads))) as executor:
        futures = {}
        for row in meta.itertuples():
            future = executor.submit(
                __fetch_skeleton, row, client=client,
                with_synapses=False, missing_swc=missing_swc, heal=False)
            futures[future] = row.bodyId
        for future in as_completed(futures):
            done += 1
            try:
                neuron = future.result()
            except Exception as exc:  # navis prints and continues
                print(f"{futures[future]} generated an exception: {exc}")
                neuron = None
            if neuron is not None:
                neurons.append(neuron)
            if on_neuron is not None:
                on_neuron(done, batch_total)
    return neurons


def _normalize_fetched_neurons(dataset: str, neurons: Dict[Union[int, str], object],
                               flywire: bool) -> Dict[Union[int, str], object]:
    """Normalize fetched neurons at the dataset boundary.

    FlyWire stays a MeshNeuron; only NeuPrint accepts DataFrame ->
    TreeNeuron coercion.  Multi-node ``soma`` (navis' default soma detection
    flags every radius >= 1 node, so a whole-neuron "soma" would freeze
    downsampling) is cleared.  Returns the canonical-keyed mapping with the
    surviving neurons.
    """
    out: Dict[Union[int, str], object] = {}
    for fallback_id, neuron in neurons.items():
        try:
            if flywire:
                if not isinstance(neuron, navis.MeshNeuron):
                    continue
                neuron.id = _canonical_dataset_body_id(dataset, fallback_id)
            else:
                if not isinstance(neuron, navis.TreeNeuron):
                    neuron = navis.TreeNeuron(neuron)
                neuron.id = _api_dataset_body_id(dataset, fallback_id)
                soma = getattr(neuron, "soma", None)
                if soma is not None and hasattr(soma, "__len__") and len(soma) > 1:
                    neuron.soma = None
            out[_canonical_dataset_body_id(dataset, fallback_id)] = neuron
        except Exception:
            continue
    return out


def fetch_skeletons_on_demand_batch(
        dataset: str, body_ids, project_root: Optional[str] = None,
        persist: bool = True, level: str = VECTOR_BASIS_RAW,
        batch_size: int = NEUPRINT_FETCH_BATCH_SIZE,
        max_threads: int = NEUPRINT_FETCH_MAX_THREADS,
        progress_callback=None, client=None,
        raw_cache: Optional[SkeletonVectorCache] = None,
        vector_cache: Optional[SkeletonVectorCache] = None,
        simplification: int = DEFAULT_SIMPLIFICATION,
        cancel_event=None) -> Dict[int, object]:
    """Fetch a set of skeletons through one cache-aware online phase.

    NeuPrint and FlyWire use separate cache transactions. NeuPrint loads and
    writes raw ``TreeNeuron`` SWC through the shared simplify + compress
    pipeline (``simplification`` percent removed, default 90, recorded in
    the ``.swc.zst`` header); FlyWire loads and writes prepared
    ``MeshNeuron`` pickles at the fixed visualization mesh level. Neither
    path converts the other representation.

    Vectorization always runs on the RAW fetched neurons and is persisted to
    the standalone vector cache BEFORE the simplified on-disk files are
    written. Cached files are loaded with the requested target level, so a
    file stored at a lower level is re-simplified on read.

    ``progress_callback(done, total, message)`` is optional and is called
    at batch boundaries and, during the online NeuPrint fetch, once per
    completed skeleton (the per-neuron cadence of the terminal bar).
    ``cancel_event`` (threading.Event) is optional: when
    set, no new fetch batch is started and the already-fetched skeletons are
    still vectorized/persisted (resume-safe).  The returned mapping is keyed
    by integer body ID and contains only skeletons that were available or
    successfully fetched.
    """
    requested = []
    seen = set()
    if body_ids is not None:
        for body_id in body_ids:
            try:
                bid = _canonical_dataset_body_id(dataset, body_id)
            except (TypeError, ValueError):
                continue
            if bid not in seen:
                seen.add(bid)
                requested.append(bid)

    level = str(level).lower()
    if level not in (VECTOR_BASIS_RAW, VECTOR_BASIS_SIMP90):
        raise ValueError(f"Invalid level: {level} (raw|simp90)")
    # None is never accepted here: 0 (raw) is the explicit escape.
    _simplification_factor(simplification)
    # The simplification pipeline is NeuPrint-only: FlyWire/FAFB/BANC always
    # fetch mesh representations and must never be re-leveled or simplified.
    if is_flywire_dataset(dataset):
        simplification = 0
    if not requested:
        return {}

    root = Path(project_root) if project_root else Path(__file__).parent.parent
    flywire = is_flywire_dataset(dataset)
    flywire_soma_positions = (
        _load_flywire_soma_positions(dataset, root, requested)
        if flywire else {}
    )
    loaded: Dict[int, object] = {}
    missing = []
    raw_lookup_cache = raw_cache
    # The pipelined NeuPrint path (fetch loop + standalone persist worker)
    # is enabled only for the batched NeuPrint branch below.
    pipeline = False
    if flywire:
        # A FlyWire caller must never inherit a NeuPrint raw-SWC cache object
        # supplied by an older integration.
        if not getattr(raw_lookup_cache, "mesh_only", False):
            raw_lookup_cache = None
        if persist and raw_lookup_cache is None:
            try:
                raw_lookup_cache = find_similar_flywire_mesh_cache(
                    dataset, project_root=str(root), verbose=False)
            except Exception:
                raw_lookup_cache = None
        if not persist:
            raw_lookup_cache = None
        if raw_cache is None or not getattr(raw_cache, "mesh_only", False):
            raw_cache = raw_lookup_cache
    elif raw_lookup_cache is None:
        # Every NeuPrint caller gets the same first lookup into the shared
        # raw SWC cache. The same cache is also the raw-file write target.
        try:
            raw_lookup_cache = find_similar_raw_cache(
                dataset, project_root=str(root), verbose=False)
        except Exception:
            raw_lookup_cache = None
        if raw_cache is None:
            raw_cache = raw_lookup_cache

    if raw_lookup_cache is not None:
        # Cache membership is decided with ONE directory scan instead of one
        # per-id lookup: a full-dataset pull with an almost-empty cache used
        # to spend ~30 minutes re-scanning the cache directory (rglob per
        # missing id) before the first batch even completed.  Only ids that
        # ARE cached go through load_skeleton (which loads/migrates the
        # actual neuron); the rest go straight to the fetch list.  Caches
        # without a directory scanner (e.g. test fakes) keep the per-id
        # lookup semantics.
        cached_ids: set = set()
        scan_ok = False
        scanner = getattr(raw_lookup_cache, "_discover_skeleton_files", None)
        if callable(scanner):
            try:
                for path in scanner():
                    try:
                        cached_ids.add(
                            _canonical_dataset_body_id(
                                dataset, _skeleton_body_id(path)
                            )
                        )
                    except (TypeError, ValueError):
                        continue
                scan_ok = True
            except Exception:
                cached_ids = set()
        # Crash-resume staging: raw skeletons written by a previous
        # interrupted pull live in ``skeletons/_temp_cache``.  They are loaded
        # from disk (never re-fetched) and routed through the pipeline like
        # freshly-fetched batches; their temp entries are removed once the
        # final simplified files are persisted.
        temp_pending: Dict[int, object] = {}
        temp_dir_fn = getattr(raw_lookup_cache, "temp_cache_dir", None)
        if callable(temp_dir_fn):
            try:
                temp_dir = temp_dir_fn()
            except Exception:
                temp_dir = None
            if temp_dir is not None and temp_dir.is_dir():
                try:
                    for path in temp_dir.glob("*.swc.zst"):
                        try:
                            bid = _canonical_dataset_body_id(
                                dataset, _skeleton_body_id(path))
                        except (TypeError, ValueError):
                            continue
                        neuron = _load_cached_skeleton_file(str(path))
                        if neuron is not None:
                            temp_pending.setdefault(bid, neuron)
                except Exception:
                    temp_pending = {}
        for bid in requested:
            if bid in temp_pending:
                continue  # processed from temp by the pipeline below
            if scan_ok and bid not in cached_ids:
                missing.append(bid)
                continue
            neuron = raw_lookup_cache.load_skeleton(
                bid, simplification=simplification)
            if neuron is None:
                missing.append(bid)
            else:
                loaded[bid] = neuron
    else:
        # If cache construction failed, still fetch online; persistence is
        # best-effort and the in-memory result remains usable.
        missing = list(requested)
        temp_pending = {}

    if progress_callback:
        progress_callback(len(loaded), len(requested),
                          f"Neuron cache ({len(loaded)}/{len(requested)})")

    if missing or temp_pending:
        dataset_l = dataset.lower()
        fetched_by_id: Dict[int, object] = {}
        # Normalize the temp-pending neurons once; they flow through the same
        # pipeline as fetched batches (vectorize + persist + temp cleanup)
        # without a network round-trip.
        temp_normalized: Dict[int, object] = {}
        if temp_pending:
            temp_normalized = _normalize_fetched_neurons(
                dataset, temp_pending, flywire=False)
            fetched_by_id.update(temp_normalized)
        if fetch_skeleton_on_demand is not _SINGLE_FETCH_IMPLEMENTATION:
            # Preserve an explicit singular-fetch override. This is useful
            # for offline clients and keeps downstream integrations that
            # monkeypatch the old public seam working while production uses
            # the batch API below.
            for bid in missing:
                if cancel_event is not None and cancel_event.is_set():
                    break
                try:
                    neuron = fetch_skeleton_on_demand(
                        dataset, bid, project_root=str(root), persist=persist,
                        level=level, raw_cache=raw_cache,
                        vector_cache=vector_cache,
                        soma_pos=flywire_soma_positions.get(str(bid)))
                except TypeError as exc:
                    # A few older integrations exposed the original helper
                    # without the newer cache/level keywords.  The batch
                    # phase below owns persistence, so old overrides remain
                    # compatible without changing the new raw-cache path.
                    if not any(k in str(exc) for k in
                               ("persist", "level", "raw_cache", "vector_cache",
                                "soma_pos")):
                        raise
                    try:
                        neuron = fetch_skeleton_on_demand(
                            dataset, bid, project_root=str(root),
                            persist=persist,
                            soma_pos=flywire_soma_positions.get(str(bid)))
                    except TypeError as older_exc:
                        if not any(k in str(older_exc)
                                   for k in ("persist", "soma_pos")):
                            raise
                        try:
                            neuron = fetch_skeleton_on_demand(
                                dataset, bid, project_root=str(root),
                                persist=persist)
                        except TypeError as legacy_exc:
                            if "persist" not in str(legacy_exc):
                                raise
                            try:
                                neuron = fetch_skeleton_on_demand(
                                    dataset, bid,
                                    project_root=str(root))
                            except TypeError as root_exc:
                                if "project_root" not in str(root_exc):
                                    raise
                                neuron = fetch_skeleton_on_demand(
                                    dataset, bid)
                if neuron is not None:
                    fetched_by_id[
                        _canonical_dataset_body_id(dataset, bid)
                    ] = neuron
            if progress_callback:
                progress_callback(
                    len(loaded) + len(fetched_by_id), len(requested),
                    f"Fetching skeletons ({len(loaded) + len(fetched_by_id)}/"
                    f"{len(requested)})")
        elif flywire:
            # CAVE returns meshes. Keep this branch separate from NeuPrint's
            # TreeNeuron/SWC batching and never call fetch_skeletons().
            if cancel_event is None or not cancel_event.is_set():
                from cave_data_fetcher import CAVEDataFetcher
                fetcher = CAVEDataFetcher(
                    dataset=_dataset_folder(dataset),
                    project_root=str(root),
                    verbose=False,
                )
                neurons = fetcher.fetch_fafb_meshes(
                    [body_id_to_api_int(bid) for bid in missing],
                    use_cache=bool(persist),
                    simplify_mesh=FLYWIRE_MESH_CACHE_SIMPLIFICATION,
                    soma_simplification=FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION,
                    soma_radius=FLYWIRE_MESH_CACHE_SOMA_RADIUS,
                    soma_positions=flywire_soma_positions,
                )
                for neuron in neurons or []:
                    neuron_id = getattr(neuron, "id", None)
                    if neuron_id is None:
                        continue
                    try:
                        fetched_by_id[
                            _canonical_dataset_body_id(dataset, neuron_id)
                        ] = neuron
                    except (TypeError, ValueError):
                        continue
        else:
            # One NeuPrint client is shared by all bounded requests.  This is
            # the important distinction from the legacy per-neuron helper.
            from neuprint import Client, set_default_client
            from navis.interfaces import neuprint as neu

            if client is None:
                try:
                    from utils.token_manager import token_manager
                    token = token_manager.get_neuprint_token()
                except Exception:
                    token = ""
                client = Client(
                    "neuprint.janelia.org", dataset=dataset, token=token)
            try:
                set_default_client(client)
            except Exception:
                pass

            effective_batch_size = max(1, int(batch_size))
            effective_threads = max(1, int(max_threads))
            total_missing = len(missing)
            total_batches = (
                (total_missing + effective_batch_size - 1)
                // effective_batch_size
            )

            # Pipeline: raw staging + simplification/cache writing run on
            # standalone threads so the network fetch loop is never
            # interrupted by CPU/disk work.  Every completed batch is first
            # staged to ``skeletons/_temp_cache`` (raw level-0 .swc.zst, fast
            # zstd-3 codec) by the staging worker - a crash then loses at
            # most the in-flight batch, and the next run reprocesses the
            # staging files instead of re-fetching.  The persist worker
            # vectorizes the RAW neurons and appends their rows (O(batch) via
            # the main + pending design), then writes the simplified
            # .swc.zst files and removes the staging entries.  Unbounded
            # queues keep the fetch loop from ever blocking (memory is
            # bounded by ``fetched_by_id`` anyway; the staging worker is
            # ~2.3x faster than the fetch rate, so its backlog stays ~0).
            import queue as _queue
            import threading
            staging_queue: "_queue.Queue" = _queue.Queue()
            persist_queue: "_queue.Queue" = _queue.Queue()
            persist_failures = {"n": 0}
            pipeline = True

            def _staging_worker() -> None:
                """Stage raw skeletons to disk ASAP, then forward to persist.
                Existing staging files (crash leftovers being reprocessed)
                are skipped - they are already on disk."""
                while True:
                    item = staging_queue.get()
                    if item is None:
                        staging_queue.task_done()
                        break
                    try:
                        if persist and raw_cache is not None:
                            temp_dir = raw_cache.temp_cache_dir()
                            for body_id, neuron in item.items():
                                try:
                                    target = temp_dir / f"{body_id}.swc.zst"
                                    if not target.exists():
                                        raw_cache.write_temp_skeleton(
                                            body_id, neuron)
                                except Exception:
                                    continue
                    except Exception:
                        persist_failures["n"] += 1
                    finally:
                        persist_queue.put(item)
                        staging_queue.task_done()

            def _persist_worker() -> None:
                processed = 0
                while True:
                    item = persist_queue.get()
                    if item is None:
                        persist_queue.task_done()
                        break
                    try:
                        processed += len(item)
                        # Vectorize the RAW neurons first and append their
                        # rows BEFORE the simplified file is written
                        # ("cached only after vectorization"); the
                        # main + pending append makes per-batch appends
                        # O(batch).
                        rows = []
                        for body_id, neuron in item.items():
                            try:
                                rep = _neuron_rep(neuron)
                                if rep in {"skeleton", "mesh"}:
                                    _, vector = vectorize_neuron(neuron)
                                    rows.append((
                                        _canonical_dataset_body_id(
                                            dataset, body_id),
                                        vector, rep,
                                    ))
                            except Exception as exc:
                                # Mirrors cache_fetched_skeleton_vectors
                                # (which this flow called with verbose=True):
                                # glitchy fetches (empty/partial SWC) produce
                                # neurons without nodes; they are skipped from
                                # the vector cache, never from the result set.
                                print(
                                    f"[morphology] vectorization skipped "
                                    f"for {body_id}: {exc}"
                                )
                                continue
                        if (persist or not flywire) and rows \
                                and raw_cache is not None:
                            try:
                                raw_cache.append_vectors(
                                    rows, vector_basis=VECTOR_BASIS_RAW)
                            except Exception as exc:
                                print(
                                    f"[morphology] vector append failed: "
                                    f"{exc}"
                                )
                        if persist and raw_cache is not None:
                            raw_cache.persist_skeletons(
                                item, simplification=simplification)
                            # The final skeleton is on disk: staging entries
                            # are no longer needed (resume-safe).
                            raw_cache.delete_temp_skeletons(
                                list(item.keys()))
                        if progress_callback:
                            progress_callback(
                                min(len(requested), processed),
                                len(requested),
                                f"Vectorizing + caching skeletons "
                                f"({min(len(requested), processed)}/"
                                f"{len(requested)})")
                    except Exception:
                        persist_failures["n"] += 1
                    finally:
                        persist_queue.task_done()

            persist_thread = threading.Thread(
                target=_persist_worker, daemon=True,
                name=f"skeleton-persist-{_dataset_folder(dataset)}",
            )
            staging_thread = threading.Thread(
                target=_staging_worker, daemon=True,
                name=f"skeleton-staging-{_dataset_folder(dataset)}",
            )
            persist_thread.start()
            staging_thread.start()

            # Temp-pending neurons from a crashed previous run are already on
            # disk: route them through the pipeline without a network fetch
            # (the staging stage skips rewriting existing temp files).
            if temp_normalized:
                staging_queue.put(dict(temp_normalized))

            try:
                for batch_index, start in enumerate(
                        range(0, total_missing, effective_batch_size),
                        start=1):
                    # Cooperative cancel: stop submitting new batches; the
                    # in-flight one finishes and its results are persisted
                    # below (resume-safe).
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    batch_ids = missing[start:start + effective_batch_size]
                    # Report the batch BEFORE the network call: the first
                    # batch can take a minute or more, and the UI used to sit
                    # frozen on the "Neuron cache (0/N)" message the whole
                    # time.  The batch counter keeps the pull visibly alive
                    # between the after-batch updates.
                    done_so_far = min(len(requested), len(loaded) + start)
                    if progress_callback:
                        progress_callback(
                            done_so_far, len(requested),
                            f"Fetching skeletons - batch {batch_index}/"
                            f"{total_batches} ({done_so_far}/{len(requested)})")

                    # Per-neuron progress: the vendored loop mirrors navis'
                    # own parallel fetch (ThreadPoolExecutor over the batch)
                    # but reports every completed skeleton, so the UI ticks
                    # like the terminal bar instead of waiting for the batch
                    # boundary.
                    def _neuron_progress(done, _batch_total):
                        if progress_callback:
                            current = min(
                                len(requested), len(loaded) + start + done)
                            progress_callback(
                                current, len(requested),
                                f"Fetching skeletons - batch {batch_index}/"
                                f"{total_batches} ({current}/{len(requested)})")

                    try:
                        batch_neurons = _fetch_neuprint_batch_with_progress(
                            batch_ids,
                            client=client,
                            max_threads=max(
                                1, min(effective_threads, len(batch_ids))),
                            on_neuron=_neuron_progress,
                        )
                    except Exception:
                        # Fall back to the upstream wrapper if a future navis
                        # release changes the vendored internals; only the
                        # per-batch progress granularity is lost.
                        batch_df = pd.DataFrame({"bodyId": batch_ids})
                        result = neu.fetch_skeletons(
                            batch_df,
                            parallel=True,
                            max_threads=max(
                                1, min(effective_threads, len(batch_ids))),
                            missing_swc="warn",
                            client=client,
                        )
                        batch_neurons = list(result or [])
                    batch_map = {}
                    for index, neuron in enumerate(batch_neurons):
                        neuron_id = getattr(neuron, "id", None)
                        if neuron_id is None and index < len(batch_ids):
                            neuron_id = batch_ids[index]
                        try:
                            batch_map[
                                _canonical_dataset_body_id(dataset, neuron_id)
                            ] = neuron
                        except (TypeError, ValueError):
                            continue
                    # Normalize at the batch boundary, stage the raw
                    # skeletons for crash-resume, and hand the batch to the
                    # persist worker while the fetch loop continues.
                    normalized_batch = _normalize_fetched_neurons(
                        dataset, batch_map, flywire=False)
                    fetched_by_id.update(normalized_batch)
                    staging_queue.put(normalized_batch)
                    if progress_callback:
                        done = min(len(requested), len(loaded)
                                   + len(fetched_by_id))
                        progress_callback(
                            done, len(requested),
                            f"Fetching skeletons ({done}/{len(requested)})")
            finally:
                # Stop the workers only after the in-flight batch's results
                # were handed over; both workers drain their queues
                # (staging every fetched batch, persisting everything
                # fetched, incl. after a cancel) before returning.
                staging_queue.put(None)
                staging_queue.join()
                persist_queue.put(None)
                persist_queue.join()

        if pipeline:
            # Pipelined NeuPrint path: per-batch normalization, vectorization
            # (appended per batch via the main + pending design) and
            # simplification already ran in the standalone workers (joined
            # above, so every fetched batch is on disk and its staging
            # entries are gone).  The final merge checkpoint folds any
            # remaining pending rows into the main parquet, so the pull
            # leaves a clean, deduped cache.
            if raw_cache is not None:
                try:
                    raw_cache._merge_pending()
                except Exception as exc:
                    # The staged batches are already persisted; a failed
                    # final merge only delays folding pending rows (they are
                    # merged by the next append/load).
                    print(f"[morphology] final vector merge failed: {exc}")
        else:
            # Normalize the object at the dataset boundary.  FlyWire remains
            # a MeshNeuron; only NeuPrint accepts DataFrame -> TreeNeuron
            # coercion.
            for fallback_id, neuron in fetched_by_id.items():
                try:
                    if flywire:
                        if not isinstance(neuron, navis.MeshNeuron):
                            continue
                        neuron.id = _canonical_dataset_body_id(
                            dataset, fallback_id)
                    else:
                        if not isinstance(neuron, navis.TreeNeuron):
                            neuron = navis.TreeNeuron(neuron)
                        neuron.id = _api_dataset_body_id(dataset, fallback_id)
                        soma = getattr(neuron, "soma", None)
                        if soma is not None and hasattr(soma, "__len__") and len(soma) > 1:
                            neuron.soma = None
                    fetched_by_id[
                        _canonical_dataset_body_id(dataset, fallback_id)
                    ] = neuron
                except Exception:
                    continue

            # Vectorize and persist the raw vectors before the batch returns.
            # A single append keeps the vector cache consistent with the
            # fetched skeleton set and exposes the previously invisible
            # post-fetch wait.
            vector_stats = {"cache_error": None}
            if persist or not flywire:
                vector_stats = cache_fetched_skeleton_vectors(
                    dataset,
                    fetched_by_id,
                    project_root=str(root),
                    vector_cache=vector_cache or raw_cache,
                    progress_callback=progress_callback,
                    progress_offset=len(loaded),
                    progress_total=len(requested),
                    verbose=True,
                )
                if vector_stats.get("cache_error"):
                    print(
                        f"[morphology] vector cache incomplete for {dataset}: "
                        f"{vector_stats['cache_error']}"
                    )

            if persist and fetched_by_id and raw_cache is not None:
                raw_cache.persist_skeletons(fetched_by_id,
                                            simplification=simplification)
                # Temp-pending neurons (crashed previous run) are now fully
                # persisted: their staging entries are no longer needed.
                raw_cache.delete_temp_skeletons(list(fetched_by_id.keys()))

        loaded.update(fetched_by_id)

    if progress_callback:
        progress_callback(len(loaded), len(requested),
                          f"Skeletons ready ({len(loaded)}/{len(requested)})")

    return {bid: loaded[bid] for bid in requested if bid in loaded}


def download_all_skeletons(dataset: str, project_root: Optional[str] = None,
                           max_workers: int = 8, limit: Optional[int] = None,
                           progress_callback=None, cancel_event=None,
                           verbose: bool = True,
                           raw: bool = True, mode: Optional[str] = None,
                           raw_format: str = "swc.zst",
                           simplification: int = DEFAULT_SIMPLIFICATION,
                           batch_size: int = NEUPRINT_FETCH_BATCH_SIZE
                           ) -> Dict[str, object]:
    """Download every missing skeleton of a dataset to the local cache.

    Mirrors the Settings-panel full dataset pull. NeuPrint pulls persist
    ``TreeNeuron`` objects through the shared simplify + compress pipeline
    in ``skeletons/raw_skeletons/{bodyId}.swc.zst``: one simplification level
    per run (``simplification``, percent of nodes removed, 0-90, default 90)
    recorded in each file header.

    FlyWire/FAFB/BANC datasets are NOT supported: bulk skeleton downloads
    are disabled for them (their skeleton bundles are large one-time
    downloads), and a ``FlyWireSkeletonAccessError`` with the manual Codex
    download instructions is raised instead.
    Visualization ``fast`` is not a pull mode for either representation.

    Vectorization always runs on the RAW fetched neurons first and is
    persisted to the standalone vector cache, so the downloaded on-disk
    simplification level never affects analysis vectors.

    ``mode`` and ``raw`` remain compatibility parameters for older callers;
    any accepted mode is normalized to ``"raw"``. NeuPrint requests use
    bounded online batches.
    ``raw_format`` and ``simplification`` are retained for NeuPrint
    compatibility and do not alter FlyWire caches.
    ``batch_size`` bounds the per-call NeuPrint request size: every call
    repeats a metadata query first, so larger batches amortize it (the
    default 64).  Progress is reported per completed skeleton, so the batch
    size no longer affects progress granularity.
    ``progress_callback(current, total, info)`` and
    ``cancel_event`` (threading.Event) drive the UI; ``limit`` bounds the
    download (tests / smoke runs). Returns a summary dict.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _simplification_factor(simplification)  # validate 0..90 up front

    # Bulk skeleton downloads are disabled for FlyWire datasets: the bundles
    # (e.g. sk_lod1_783_healed.zip for FAFB) are large one-time downloads
    # that must be fetched manually from the FlyWire Codex and placed by the
    # converter. On-demand CAVE fetches for individual missing skeletons
    # still work during visualization.
    if is_flywire_dataset(dataset):
        raise FlyWireSkeletonAccessError(
            flywire_manual_skeleton_instruction(dataset))

    require_flywire_skeleton_access(
        dataset,
        project_root=project_root,
        log=print if verbose else (lambda _message: None),
    )

    root = Path(project_root) if project_root else Path(__file__).parent.parent
    folder = _dataset_folder(dataset)
    requested_mode = str(mode or "raw").strip().lower()
    requested_mode = {
        "raw_skeletons": "raw", "fine95": "fine",
        "fine_skeletons": "fine", "simp90": "fast",
    }.get(requested_mode, requested_mode)
    if requested_mode not in {"raw", "fine", "fast"}:
        raise ValueError("mode must be 'raw', 'fine', or 'fast' (compatibility only)")
    # Pulling is representation-only. ``fast`` and ``fine`` are visualization
    # simplification choices and must never select a different disk format.
    mode = "raw"
    flywire = is_flywire_dataset(dataset)
    if flywire:
        # The simplification pipeline is NeuPrint-only; FlyWire pulls persist
        # prepared meshes and must never be re-leveled or simplified.
        simplification = 0
    if flywire:
        raw_cache = find_similar_flywire_mesh_cache(
            dataset, project_root=str(root), n_workers=max_workers,
            verbose=verbose,
        )
    else:
        raw_cache = find_similar_raw_cache(
            dataset, project_root=str(root), n_workers=max_workers,
            verbose=verbose, raw_format=raw_format
        )
    skeleton_dir = raw_cache.skeleton_dir
    skeleton_dir.mkdir(parents=True, exist_ok=True)

    # Index of all bodyIds. Read FlyWire IDs as strings before any operation
    # that could route through pandas/Polars numeric inference.
    index: List[Union[int, str]] = []
    table_candidates = [
        root / "datasets" / folder / f"{folder}_allneurons_neuron_df.parquet",
        root / "datasets" / folder / f"{folder}_allneurons_neuron_df.csv",
    ]
    table_path = next((path for path in table_candidates if path.exists()), None)
    if table_path is not None:
        try:
            if table_path.suffix.lower() == ".parquet":
                index_frame = pd.read_parquet(table_path, columns=["bodyId"])
            else:
                index_frame = pd.read_csv(
                    table_path,
                    usecols=["bodyId"],
                    dtype={"bodyId": "string"}
                    if is_flywire_dataset(dataset) else None,
                )
            index = (
                [normalize_flywire_body_id(b)
                 for b in index_frame["bodyId"].tolist()]
                if is_flywire_dataset(dataset) else
                [int(b) for b in index_frame["bodyId"].tolist()]
            )
        except Exception:
            index = []
    if not index:
        index_path = root / "neuron_indexes" / folder / "neuron_index.parquet"
        if index_path.exists() and _has_local_dataset_presence(dataset, root):
            try:
                index_frame = pd.read_parquet(index_path, columns=["bodyId"])
                index = (
                    [normalize_flywire_body_id(b)
                     for b in index_frame["bodyId"].tolist()]
                    if is_flywire_dataset(dataset) else
                    [int(b) for b in index_frame["bodyId"].tolist()]
                )
            except Exception:
                index = []

    flywire_soma_positions = (
        _load_flywire_soma_positions(dataset, root, index)
        if flywire else {}
    )

    # Only files from the representation-specific cache count as available.
    existing_paths = [Path(path) for path in raw_cache._discover_skeleton_files()]
    existing = set()
    for path in existing_paths:
        try:
            existing.add(
                _canonical_dataset_body_id(dataset, _skeleton_body_id(path))
            )
        except (TypeError, ValueError):
            continue

    # FAFB v783 only: the healed bundle (.zst first; ZIP fallback) already
    # provides most skeletons locally — count its entries as available
    # instead of re-fetching them through the CAVE API. Only the genuinely
    # missing ids are downloaded.
    local_bundle_ids: set = set()
    if is_fafb_dataset(dataset):
        try:
            bundle = _fafb_bundle(dataset, str(root))
        except Exception:
            bundle = None
        if bundle is not None:
            try:
                local_bundle_ids = {
                    _canonical_dataset_body_id(dataset, b)
                    for b in bundle.ids()
                }
            except Exception:
                local_bundle_ids = set()
            finally:
                bundle.close()
        else:
            zip_path = _fafb_skeleton_zip_path(dataset, str(root))
            if zip_path is not None:
                try:
                    import zipfile
                    with zipfile.ZipFile(zip_path, "r") as z:
                        local_bundle_ids = {
                            _canonical_dataset_body_id(dataset, n[:-4])
                            for n in z.namelist() if n.endswith(".swc")
                        }
                except Exception:
                    local_bundle_ids = set()

    missing = [
        _canonical_dataset_body_id(dataset, b)
        for b in index
        if _canonical_dataset_body_id(dataset, b) not in existing
        and _canonical_dataset_body_id(dataset, b) not in local_bundle_ids
    ]
    if limit is not None:
        missing = missing[: int(limit)]
    total = len(missing)
    skipped_existing = len(existing) + len(local_bundle_ids)
    if total == 0:
        if verbose:
            print(f"[morphology] download_all_skeletons: "
                  f"{skipped_existing} skeletons already available locally "
                  f"({len(existing)} cached, {len(local_bundle_ids)} from the "
                  f"healed bundle); nothing to fetch.")
        return {"total": 0, "fetched": 0, "skipped_existing": skipped_existing,
                "cancelled": False, "errors": 0, "mode": mode,
                "representation": "mesh" if flywire else "skeleton",
                "simplification": int(simplification)}

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

    # NeuPrint downloads use the same aggregated, bounded batch path as the
    # visualization and similarity workflows. Keep FlyWire's legacy worker
    # path below because its local-bundle/API fallback has per-neuron
    # cancellation semantics that are independent of NeuPrint's SWC API.
    if not is_flywire_dataset(dataset):
        if cancel_event.is_set():
            cancelled = True
        else:
            def _batch_progress(done, batch_total, message):
                if progress_callback:
                    progress_callback(min(done, total), total, message)

            try:
                fetched_map = fetch_skeletons_on_demand_batch(
                    dataset,
                    missing,
                    project_root=str(root),
                    persist=True,
                    level=VECTOR_BASIS_RAW,
                    batch_size=int(batch_size),
                    max_threads=max_workers,
                    progress_callback=_batch_progress,
                    raw_cache=raw_cache,
                    vector_cache=raw_cache,
                    simplification=simplification,
                    cancel_event=cancel_event,
                )
                fetched = len(fetched_map)
                errors = total - fetched
                cancelled = cancel_event.is_set()
            except Exception as exc:
                errors = total
                if verbose:
                    print(f"[morphology] batched NeuPrint fetch failed: {exc}")
            finally:
                if progress_callback:
                    progress_callback(
                        fetched + errors, total,
                        "Cancelled." if cancelled else "Finished.")

        summary = {
            "total": total,
            "fetched": fetched,
            "skipped_existing": skipped_existing,
            "cancelled": cancelled,
            "errors": errors,
            "mode": mode,
            "representation": "skeleton",
            "simplification": int(simplification),
        }
        if verbose:
            print(f"[morphology] download_all_skeletons: {fetched}/{total} "
                  f"fetched, {errors} errors, cancelled={cancelled}, "
                  f"simplification={simplification}")
        return summary

    def _fetch_one(bid: Union[int, str]) -> bool:
        if cancel_event.is_set():
            return False
        try:
            fetch_kwargs = {
                "project_root": str(root),
                "persist": True,
                "level": VECTOR_BASIS_RAW,
            }
            if raw_cache is not None:
                fetch_kwargs.update({"raw_cache": raw_cache,
                                     "vector_cache": raw_cache})
            if flywire:
                fetch_kwargs["soma_pos"] = flywire_soma_positions.get(str(bid))
            try:
                nrn = fetch_skeleton_on_demand(dataset, bid, **fetch_kwargs)
            except TypeError as exc:
                # Keep Download All compatible with older integrations that
                # still expose the pre-namespace singular fetcher.
                if not any(k in str(exc) for k in (
                        "level", "raw_cache", "vector_cache", "soma_pos")):
                    raise
                nrn = fetch_skeleton_on_demand(
                    dataset, bid, project_root=str(root),
                    persist=True,
                )
            if nrn is not None and raw_cache is not None:
                try:
                    raw_cache.persist_skeletons({
                        _canonical_dataset_body_id(dataset, bid): nrn
                    })
                except Exception:
                    pass
            expected_rep = "mesh" if flywire else "skeleton"
            ok = nrn is not None and _neuron_rep(nrn) == expected_rep
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
        "skipped_existing": skipped_existing,
        "cancelled": cancelled,
        "errors": errors,
        "mode": mode,
        "representation": "mesh" if flywire else "skeleton",
        "simplification": int(simplification),
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
        # The two size knobs shared by EVERY candidate source mode:
        # ``candidate_cap`` bounds how many screen candidates enter the
        # morphology comparison (also the NBLAST prefilter of cache-direct
        # searches) and ``visualize_top_n`` bounds rendering only. Results
        # themselves are never truncated: every compared candidate is
        # returned and written.
        candidate_cap: int = 500,
        candidate_source: str = "auto",
        visualize_top_n: int = 0,
        visualize_by: str = "type",
        # Connection-cache screen thresholds ('profile' / 'combined' modes).
        min_weight: int = 3,
        min_shared_partners: int = 2,
        roi_filter: Optional[List[str]] = None,
        visualization_settings: Optional[Dict[str, object]] = None,
        output_dir: Optional[str] = None,
        saveas: Optional[str] = None,
        verbose: bool = True,
        n_workers: int = 8,
        use_cache: bool = True,
        # Retained as a compatibility keyword. Raw skeletons are now always
        # persisted as compressed SWC in the shared dataset skeleton cache.
        cache_fetched_skeletons: bool = True,
        project_root: Optional[str] = None,
    ):
        self.query = query
        self.dataset = dataset
        self.level = str(level).lower()
        self.method = str(method).lower()
        self.metric = str(metric).lower()
        self.candidate_cap = int(candidate_cap)
        self.candidate_source = str(candidate_source).lower()
        # Lazily-built ROI profile store for 'roi'/'combined' screens.
        self._roi_store: Optional[RoiProfileStore] = None
        self.min_weight = int(min_weight)
        self.min_shared_partners = int(min_shared_partners)
        self.roi_filter = list(roi_filter) if roi_filter else None
        self.visualize_top_n = int(visualize_top_n)
        self.visualize_by = str(visualize_by).lower()
        self.visualization_settings = dict(visualization_settings or {})
        self.verbose = verbose
        self.n_workers = max(1, int(n_workers))
        self.use_cache = use_cache
        # Keep accepting the old keyword, but raw persistence is unconditional
        # so visualization, Find Similar, and Settings pulls share one source.
        self.cache_fetched_skeletons = True
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent

        if self.level not in ("auto", "bodyid", "type"):
            raise ValueError(f"Invalid level: {self.level} (auto|bodyid|type)")
        if self.method not in ("vector", "nblast"):
            raise ValueError(f"Invalid method: {self.method} (vector|nblast)")
        if self.metric not in ("cosine", "pearson"):
            raise ValueError(f"Invalid metric: {self.metric} (cosine|pearson)")
        if self.candidate_source not in ("auto", "roi", "combined", "profile",
                                         "cache"):
            raise ValueError(
                f"Invalid candidate_source: {self.candidate_source} "
                "(auto|roi|combined|profile|cache)"
            )
        if self.candidate_cap < 1:
            raise ValueError(
                f"Invalid candidate_cap: {self.candidate_cap} (>= 1)"
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

    def _body_id(self, value):
        """Return the comparison-layer body-ID representation."""

        return _canonical_dataset_body_id(self.dataset, value)

    def _body_ids(self, values):
        """Canonicalize a body-ID iterable for this dataset."""

        return [self._body_id(value) for value in values]

    def _resolved_candidate_source(self) -> str:
        if self.candidate_source != "auto":
            return self.candidate_source
        # NeuPrint datasets screen candidates from the per-neuron primary-ROI
        # distributions: every neuron is reachable, while the connection-cache
        # screen can never find neurons without shared partners. FlyWire
        # datasets have no ROI table and search the vector cache directly.
        if self._is_flywire():
            return "cache"
        return "roi" if self._roi_screening_ready(allow_backfill=True) \
            else "profile"

    def _roi_screening_ready(self, allow_backfill: bool) -> bool:
        """Cheap local probe for the ROI screen's prerequisites.

        Checks the ROI-count table (parquet or CSV) and the metadata sidecar;
        when the sidecar is missing, ``allow_backfill`` fetches and saves it
        once through the dataset-preparation code (network). The partition
        validation of the ROI list happens later, at store build.
        """
        try:
            if not roi_count_table_path(
                    self.dataset, str(self.project_root)).exists():
                return False
            if load_primary_rois(self.dataset, str(self.project_root)):
                return True
            if allow_backfill:
                meta = backfill_dataset_metadata(
                    self.dataset, str(self.project_root), log=self._log)
                return bool(
                    meta and (meta.get("roi_coverage") or {}).get("roi_list")
                )
        except Exception:
            return False
        return False

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
        out["bodyId"] = self._body_ids(out["bodyId"].tolist())
        if not self._is_flywire():
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
        # The step scheme depends on the pipeline: candidate-screen-first
        # (NeuPrint: roi / combined / profile) reports 6 steps,
        # vector-cache-direct (FlyWire) 4.
        total_steps = (PROFILE_FIRST_TOTAL_STEPS
                       if source in CANDIDATE_SCREEN_SOURCES
                       else CACHE_DIRECT_TOTAL_STEPS)
        self.resolved_candidate_source = source
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

        # BANC raises FlyWireSkeletonAccessError in the guard above (no BANC
        # skeleton source exists); FAFB may proceed because its real skeleton
        # sources (healed ZIP / CAVE fallback) were validated there.

        self._progress(1, total_steps, "Resolving query neuron")
        query_df = self._resolve_query()
        if self.level == "auto":
            self.level = self._resolve_level(query_df)
            self._log(f"Level auto-resolved to: {self.level} "
                      f"({'type-to-type' if self.level == 'type' else 'bodyId-to-bodyId'})")
        cache = find_similar_dataset_cache(
            self.dataset, project_root=str(self.project_root),
            n_workers=self.n_workers, verbose=self.verbose,
        )

        if source in CANDIDATE_SCREEN_SOURCES:
            bodyid_df, type_df = self._profile_first_search(query_df, cache,
                                                            source)
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
            # Build from raw local sources first, then fetch any missing query
            # skeletons online. Candidate skeletons remain bounded by the
            # available raw population; this avoids silently using the shared
            # visualization simp90 cache.
            self._progress(2, CACHE_DIRECT_TOTAL_STEPS, "Loading vector cache")
            cache.ensure(fetch_missing=0)
            data = cache.load()
            query_ids = self._body_ids(query_df["bodyId"].tolist())
            cached_ids = (set(self._body_ids(data["bodyIds"]))
                          if data is not None else set())
            missing_query = [bid for bid in query_ids if bid not in cached_ids]
            if missing_query:
                self._log(
                    f"Vector cache miss: fetching {len(missing_query)} raw "
                    "query skeleton(s) online."
                )
                fetched_query = fetch_skeletons_on_demand_batch(
                    self.dataset,
                    missing_query,
                    project_root=str(self.project_root),
                    persist=self.cache_fetched_skeletons,
                    level=VECTOR_BASIS_RAW,
                    max_threads=min(NEUPRINT_FETCH_MAX_THREADS,
                                    max(1, int(self.n_workers))),
                    raw_cache=cache,
                    vector_cache=cache,
                )
                # Keep the vector-mode contract even for integrations that
                # override the batch fetch seam and return neurons without
                # appending their own vector rows.
                rows = []
                for bid, neuron in (fetched_query or {}).items():
                    try:
                        if _neuron_rep(neuron) != "skeleton":
                            continue
                        _, vec = vectorize_neuron(neuron)
                        rows.append((self._body_id(bid), vec, "skeleton"))
                    except Exception:
                        continue
                if rows:
                    cache.append_vectors(rows, vector_basis=VECTOR_BASIS_RAW)
                data = cache.load()
            if data is None or len(data["bodyIds"]) == 0:
                raise ValueError(
                    f"No raw vectorized neurons for {self.dataset}. Fetch or "
                    "build the shared raw vector cache first."
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

        query_ids = self._body_ids(query_df["bodyId"].tolist())
        id_dtype = pl.Utf8 if self._is_flywire() else pl.Int64
        q = pl.Series("q", query_ids, dtype=id_dtype)
        # implode() marks the Series unambiguously as a single membership
        # collection (polars >=1.30 deprecates same-dtype scalar semantics).
        q_coll = q.implode()
        if self._is_flywire():
            conn = conn.with_columns([
                pl.col("bodyId_pre").cast(pl.Utf8, strict=False),
                pl.col("bodyId_post").cast(pl.Utf8, strict=False),
            ])
        else:
            conn = conn.with_columns([
                pl.col("bodyId_pre").cast(pl.Int64, strict=False),
                pl.col("bodyId_post").cast(pl.Int64, strict=False),
            ])
        conn = conn.drop_nulls(["bodyId_pre", "bodyId_post"])
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
            lambda b: type_map.get(self._body_id(b), "")
        )
        return out

    def _roi_candidates(self, query_df: pd.DataFrame,
                         top_k: Optional[int] = None) -> pd.DataFrame:
        """Rank candidate neurons by primary-ROI distribution similarity.

        Uses ``RoiProfileStore`` (built once per dataset from the ROI-count
        CSV, cached under ``cache/{dataset}/morphology/roi_profiles.npz``):
        mirrored cosine of the input/output synapse distributions over the
        dataset's primary ROIs. Raises ``RoiScreeningUnavailable`` when the
        dataset lacks the ROI table or a usable primary-ROI list. Returns a
        DataFrame [target_bodyId, roi_similarity, target_type] sorted by
        similarity descending; ``top_k`` bounds it to the best K rows.
        """
        if self._roi_store is None:
            self._roi_store = RoiProfileStore(
                self.dataset, project_root=str(self.project_root),
                verbose=self.verbose, log=self._log,
            ).ensure()
        scores = self._roi_store.screen(
            [int(b) for b in query_df["bodyId"].tolist()], top_k=top_k
        )
        if scores.empty:
            return pd.DataFrame()
        type_map, _ = _load_neuron_type_map(self.dataset, str(self.project_root))
        out = pd.DataFrame({
            "target_bodyId": scores["bodyId"].astype(np.int64),
            "roi_similarity": scores["roi_similarity"].astype(float),
        })
        # Vectorized type lookup (the previous per-row lambda mapped the
        # full dataset population in Python).
        out["target_type"] = (
            pd.Series(out["target_bodyId"].values).map(type_map)
            .fillna("").astype(str)
        )
        return out

    def _discover_candidates(self, query_df: pd.DataFrame,
                             source: str) -> Tuple[pd.DataFrame, str]:
        """Rank candidate neurons for the scoring-pool selection.

        Returns the unified candidate frame and the source label recorded in
        the results. The frame carries ``target_bodyId`` / ``target_type`` /
        ``profile_similarity`` (connectivity evidence, 0-1) /
        ``roi_similarity`` (ROI-screen evidence, 0-1) and ``_score``, the
        per-candidate ranking value: shared-partner count for 'profile', ROI
        cosine for 'roi', and the mean of the max-normalized scores of both
        screens for 'combined'. The frame is ALWAYS sorted by ``_score``
        descending — the pool is simply its top ``candidate_cap`` rows.
        """
        if source == "profile":
            candidates = self._connection_cache_candidates(query_df)
            if not candidates.empty:
                candidates = candidates.copy()
                candidates["_score"] = candidates["shared_count"].astype(float)
                candidates["roi_similarity"] = np.nan
            return _sorted_candidates(candidates), "profile"

        roi = self._roi_candidates(
            query_df,
            # 'roi' mode pools exactly the top candidate_cap rows, so the
            # screen can return the partial ranking directly; 'combined'
            # needs the full list to merge with the connectivity screen.
            top_k=self.candidate_cap if source == "roi" else None,
        )   # raises when unavailable
        if source == "roi":
            roi = roi.copy()
            roi["_score"] = roi["roi_similarity"]
            roi["profile_similarity"] = np.nan
            return _sorted_candidates(roi), "roi"

        # combined: outer-merge both screens; a missing connection cache
        # degrades to the ROI screen (the reverse is not possible — the ROI
        # table is a hard requirement here).
        conn = self._connection_cache_candidates(query_df)
        if conn.empty:
            self._log("Connection cache missing; 'combined' candidate "
                      "discovery uses the ROI screen only.")
            roi = roi.copy()
            roi["_score"] = roi["roi_similarity"]
            roi["profile_similarity"] = np.nan
            return _sorted_candidates(roi), "combined"
        conn = conn.copy()
        conn["_score"] = conn["shared_count"] / max(
            1.0, float(conn["shared_count"].max()))
        roi = roi.copy()
        roi["_score"] = roi["roi_similarity"] / max(
            1e-9, float(roi["roi_similarity"].max()))
        merged = conn.merge(
            roi, on="target_bodyId", how="outer",
            suffixes=("", "_roi"),
        )
        # Both frames type their candidates from the same map; keep whichever
        # side has a non-empty label, then average the per-screen scores
        # (pandas mean skips the NaN of a screen that missed the neuron).
        merged["target_type"] = (
            merged["target_type"].replace("", np.nan)
            .fillna(merged["target_type_roi"].replace("", np.nan))
            .fillna("")
        )
        merged["_score"] = merged[["_score", "_score_roi"]].mean(axis=1)
        merged = merged.drop(columns=["target_type_roi", "_score_roi"])
        return _sorted_candidates(merged), "combined"

    def _profile_first_search(self, query_df: pd.DataFrame,
                              cache: SkeletonVectorCache,
                              source: str = "profile") -> pd.DataFrame:
        """Candidate-screen-first, then morphology on the capped pool.

        Candidate discovery ranks neurons by the selected screen
        (``_discover_candidates``: connection-cache shared partners, primary-
        ROI distribution cosine, or both) and returns them sorted; the
        scoring pool is simply the top ``candidate_cap`` candidates (query
        members are already excluded by the screens). Fetched skeletons are
        transient (used for the current comparison only), while their raw
        vectors are persisted for reuse; cached skeletons are reused. Every scored
        vector is standardized with ONE consistent set of statistics (cache
        meta, sample-based population stats, or pool stats as a last
        resort), and the whole comparison runs at ONE representation level
        (skeleton vs mesh — rows of any other representation are
        unscorable). ALL compared candidates are returned and written.
        """
        step2_label = {
            "profile": "connection cache",
            "roi": "ROI distribution screen",
            "combined": "ROI + connectivity screens",
        }.get(source, source)
        self._log(f"Step 2/6 — Discovering candidates: running {step2_label} "
                  "candidate discovery...")
        self._progress(2, PROFILE_FIRST_TOTAL_STEPS,
                       f"Discovering candidates ({step2_label})")
        try:
            candidates, source = self._discover_candidates(query_df, source)
        except RoiScreeningUnavailable as exc:
            # Auto-resolved ROI runs fall back to the connection-cache screen;
            # explicit roi/combined selections surface the preparation hint.
            if self.candidate_source != "auto":
                raise
            self._log(f"{exc}")
            self._log("Falling back to the connection-cache candidate screen.")
            candidates, source = self._discover_candidates(query_df, "profile")
        self.resolved_candidate_source = source
        if candidates.empty:
            self._log(f"Candidate discovery ({source}) returned no candidates.")
            return pd.DataFrame(), pd.DataFrame()

        # The scoring pool: the sorted candidate list truncated to the cap.
        # Untyped candidates stay in — they are comparable neurons, their
        # rows just carry an empty target_type.
        pool_ids = self._body_ids(candidates["target_bodyId"].tolist())
        pool_ids = pool_ids[: self.candidate_cap]
        n_screened = len(candidates)
        self._log(f"Step 3/6 — Selecting the scoring pool: top {len(pool_ids)} "
                  f"of {n_screened} candidates (cap {self.candidate_cap})")
        self._progress(3, PROFILE_FIRST_TOTAL_STEPS,
                       f"Selecting top {len(pool_ids)} candidates for scoring")

        prof_by_id = {
            self._body_id(b): float(v)
            for b, v in zip(candidates["target_bodyId"], candidates["profile_similarity"])
            if np.isfinite(v)
        }
        roi_by_id = {
            self._body_id(b): float(v)
            for b, v in zip(candidates["target_bodyId"], candidates["roi_similarity"])
            if np.isfinite(v)
        }

        self._log("Step 4/6 — Loading & vectorizing skeletons")
        self._progress(4, PROFILE_FIRST_TOTAL_STEPS, "Loading & vectorizing skeletons")
        query_ids = self._body_ids(query_df["bodyId"].tolist())

        cache_data = cache.load()
        cache_ids = (set(self._body_ids(cache_data["bodyIds"]))
                     if cache_data is not None else set())
        cache_rep = cache_data.get("dataset_rep", "") if cache_data is not None else ""
        cache_basis = (((cache_data.get("meta") or {}).get("vector_basis")
                        or VECTOR_BASIS_RAW) if cache_data is not None
                       else VECTOR_BASIS_RAW)

        # Determine every online miss before fetching anything.  Query and
        # candidate IDs are intentionally combined: a query can also occur in
        # the candidate pool, and the optimized path must not issue two
        # separate online phases for the same comparison.
        all_load_ids = []
        seen_load_ids = set()
        for bid in query_ids + pool_ids:
            bid = self._body_id(bid)
            if bid not in seen_load_ids:
                seen_load_ids.add(bid)
                all_load_ids.append(bid)
        missing_ids = []
        for bid in all_load_ids:
            if bid in cache_ids:
                continue  # the vector cache is sufficient for screening
            pkl = cache.find_skeleton_file(bid)
            if pkl is not None:
                continue
            missing_ids.append(bid)

        def _report_fetch(done, total, message):
            self._progress(4, PROFILE_FIRST_TOTAL_STEPS, message)
            if total and (done == 0 or done >= total
                          or done % max(1, total // 10) == 0):
                self._log(f"Step 4/6 — {message}")

        fetched_all = fetch_skeletons_on_demand_batch(
            self.dataset,
            missing_ids,
            project_root=str(self.project_root),
            persist=self.cache_fetched_skeletons,
            level=VECTOR_BASIS_RAW,
            max_threads=min(NEUPRINT_FETCH_MAX_THREADS,
                            max(1, int(self.n_workers))),
            progress_callback=_report_fetch,
            raw_cache=cache,
            vector_cache=cache,
        ) if missing_ids else {}

        query_neurons = {
            self._body_id(bid): fetched_all[self._body_id(bid)]
            for bid in query_ids if self._body_id(bid) in fetched_all
        }

        # The comparison's representation: the majority among the query
        # members (cache rows carry the cache's representation).
        from collections import Counter
        known_q = []
        for bid in query_ids:
            if self._body_id(bid) in cache_ids:
                known_q.append(cache_rep)
            elif self._body_id(bid) in query_neurons:
                known_q.append(_neuron_rep(query_neurons[self._body_id(bid)]))
        q_rep = Counter(r for r in known_q if r).most_common(1)[0][0] \
            if any(known_q) else ""
        if not q_rep:
            # No cached/fetched query member with a known representation
            # (e.g. all query rows computed from cache files): infer it from
            # the dataset's skeleton store.
            q_rep = _infer_dataset_rep(self.dataset, str(self.project_root))

        # The one combined fetch above supplies both sides of the comparison.
        # Filter only after q_rep is known so the representation guard remains
        # identical to the former query-then-pool implementation.
        pool_neurons = {
            self._body_id(bid): fetched_all[self._body_id(bid)]
            for bid in pool_ids if self._body_id(bid) in fetched_all
            and (not q_rep or _neuron_rep(fetched_all[self._body_id(bid)]) == q_rep)
        }
        self._log(f"Profile-first: {len(pool_ids)} pool neurons, "
                  f"{len(pool_neurons)} skeletons fetched (transient)")

        # Vectorize any in-memory fetches before the cache lookup.  The batch
        # fetcher normally appends these rows itself; this append is a
        # deduplicating compatibility path for singular-fetch overrides and
        # offline fixtures.  It also makes the subsequent snapshot complete.
        fetched_vectors: List[Tuple[Union[int, str], np.ndarray, str]] = []
        for bid, neuron in fetched_all.items():
            try:
                _, vec = vectorize_neuron(neuron)
                rep = _neuron_rep(neuron)
                if not q_rep or rep == q_rep:
                    fetched_vectors.append((self._body_id(bid), vec, rep))
            except Exception:
                continue
        if fetched_vectors:
            cache.append_vectors(fetched_vectors, vector_basis=cache_basis)

        # Resolve all local-file misses in one call, then reload the cache.
        # ``vectors_for`` can return raw vectors for rows it computes during
        # that call, while cached rows are standardized.  Scoring that mixed
        # result was the first-run/warm-cache inconsistency: the online batch
        # had already appended standardized rows before this function used
        # its stale ``cache_data`` snapshot.  The fresh snapshot below makes
        # every row use the same persisted representation and statistics.
        pre_X, pre_mask, pre_reps = cache.vectors_for(
            all_load_ids, compute_missing=True
        )
        cache_data = cache.load()
        cache_ids = (set(self._body_ids(cache_data["bodyIds"]))
                     if cache_data is not None else set())
        cache_rep = (cache_data.get("dataset_rep", "")
                     if cache_data is not None else "")
        cache_basis = (((cache_data.get("meta") or {}).get("vector_basis")
                        or VECTOR_BASIS_RAW) if cache_data is not None
                       else VECTOR_BASIS_RAW)

        snapshot_rows = {}
        if cache_data is not None:
            snapshot_rows = {
                self._body_id(bid): i for i, bid in enumerate(cache_data["bodyIds"])
            }
        pre_rows = {self._body_id(bid): i for i, bid in enumerate(all_load_ids)}

        def _snapshot_vectors(ids):
            vectors = np.full((len(ids), VECTOR_DIM), np.nan)
            mask = np.zeros(len(ids), dtype=bool)
            reps = [""] * len(ids)
            for i, bid in enumerate(ids):
                bid = self._body_id(bid)
                row = snapshot_rows.get(bid)
                if row is not None and cache_data is not None:
                    candidate = cache_data["X"][row]
                    if np.isfinite(candidate).all():
                        vectors[i] = candidate
                        mask[i] = True
                        reps[i] = (cache_data["rep"][row]
                                   or cache_data.get("dataset_rep", ""))
                        continue
                # If persistence was unavailable, retain the raw vector
                # computed by this call as a last-resort transient row.  It
                # is intentionally marked as non-cache data below so the
                # standardization block applies the chosen transform.
                pre_row = pre_rows.get(bid)
                if pre_row is not None and pre_mask[pre_row]:
                    vectors[i] = pre_X[pre_row]
                    mask[i] = True
                    reps[i] = pre_reps[pre_row]
            return vectors, mask, reps

        X_q, mask_q, rep_q = _snapshot_vectors(query_ids)
        X_c, mask_c, rep_c = _snapshot_vectors(pool_ids)

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
                mu, sd = population_stats(
                    self.dataset, str(self.project_root), cache=cache
                )
            if mu is None:
                # Last resort: pool-computed statistics.
                all_rows = np.vstack([X_q[mask_q], X_c[mask_c]])
                mu = all_rows.mean(axis=0)
                sd = all_rows.std(axis=0)
                sd = np.where(sd <= 0, 1.0, sd)

            cache_q = (np.array([self._body_id(b) in cache_ids for b in query_ids])
                       & mask_q)
            cache_c = (np.array([self._body_id(b) in cache_ids for b in pool_ids])
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
                {self._body_id(b): t for b, t in zip(cache_data["bodyIds"], cache_data["types"])}
            )
            id_to_instance.update(
                {self._body_id(b): i for b, i in zip(cache_data["bodyIds"], cache_data["instances"])}
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
                np.asarray(
                    [query_ids[i] for i in query_ok],
                    dtype=object if self._is_flywire() else np.int64,
                ),
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
                np.array(
                    [query_ids[i] for i in ok],
                    dtype=object if self._is_flywire() else np.int64,
                ),
                [str(query_df["type"].iloc[i]).strip() if i < len(query_df) else ""
                 for i in ok],
                X_q[ok], self.metric,
            )

        rows: List[Dict[str, object]] = []
        if self.level == "type":
            # Compare every resolved query member to every pool candidate.
            # Candidates whose vector is unavailable keep a NaN similarity
            # row so the written candidate list stays complete (they rank
            # last and are excluded from type aggregations and rendering).
            candidate_indices = [i for i, ok in enumerate(keep) if ok]
            query_indices = [i for i, ok in enumerate(mask_q) if ok]
            for query_i in query_indices:
                q_bid = self._body_id(query_ids[query_i])
                raw_q_type = query_df["type"].iloc[query_i]
                q_type = "" if pd.isna(raw_q_type) else str(raw_q_type).strip()
                q_scores = np.full(len(pool_ids), np.nan)
                if candidate_indices:
                    q_scores[candidate_indices] = similarity_matrix(
                        X_q[query_i], X_c[candidate_indices], self.metric
                    )
                for candidate_i in range(len(pool_ids)):
                    bid = self._body_id(pool_ids[candidate_i])
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
                        "roi_similarity": roi_by_id.get(bid, np.nan),
                        "similarity": float(q_scores[candidate_i]),
                        "is_same_type": target_type == q_type if target_type else False,
                        "intra_type_similarity": intra,
                        "method": self.method,
                        "metric": self.metric,
                        "candidate_source": source,
                    })
            rows.extend(self._type_query_intra_rows(
                query_df, X_q, mask_q, intra, candidate_source=source
            ))
        else:
            # Every pool candidate gets a row: comparable ones carry their
            # similarity, the rest a NaN (sorted last, never rendered).
            for i, bid in enumerate(pool_ids):
                bid = self._body_id(bid)
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
                    "roi_similarity": roi_by_id.get(bid, np.nan),
                    "similarity": float(scores[i]),
                    "is_same_type": target_type == query_type if target_type else False,
                    "intra_type_similarity": intra,
                    "method": self.method,
                    "metric": self.metric,
                    "candidate_source": source,
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
            candidate_source=source,
        )

        # bodyId-level rows: top-N scored neurons.
        bodyid_df = self._bodyid_dataframe(rows, query_type=query_type)

        # NBLAST refinement over the fetched pool skeletons (transient).
        if self.method == "nblast":
            neurons = dict(query_neurons)
            neurons.update(pool_neurons)
            bodyid_df = self._nblast_refine(bodyid_df, query_df, cache, neurons)
        return bodyid_df, type_df

    def _nblast_pairwise(self, query_dps: Dict[int, object],
                         cand_dps: Dict[int, object],
                         desc: str = "NBLAST scoring") -> Dict[int, float]:
        """Score every query-candidate pair in-process with a per-individual
        progress bar.

        Mirrors ``navis.nblast`` defaults (forward, normalized, FCWB scoring)
        but scores pair-by-pair with ``navis.nbl.NBlaster`` so the bar can
        name the current candidate. This also avoids navis' process-pool
        startup (its spawn overhead dominates small candidate pools)."""
        from navis.nbl.nblast_funcs import NBlaster

        nb = NBlaster(use_alpha=False, normalized=True, progress=False)
        q_idx = {}
        for q_bid, q_dp in query_dps.items():
            q_idx[q_bid] = nb.append(q_dp, self_hit=nb.calc_self_hit(q_dp))
        t_idx = {}
        for t_bid, t_dp in cand_dps.items():
            t_idx[t_bid] = nb.append(t_dp, self_hit=nb.calc_self_hit(t_dp))

        total = len(q_idx) * len(t_idx)
        pbar = tqdm(total=total, desc=desc, unit="pair",
                    disable=not self.verbose, leave=False, file=sys.stdout)
        nblast_scores: Dict[int, float] = {}
        try:
            for q_bid, qi in q_idx.items():
                for t_bid, ti in t_idx.items():
                    pbar.set_postfix_str(f"{q_bid} x {t_bid}")
                    try:
                        val = float(nb.single_query_target(qi, ti,
                                                           scores='forward'))
                    except Exception:
                        val = float("nan")
                    if np.isfinite(val):
                        nblast_scores[t_bid] = max(
                            nblast_scores.get(t_bid, -np.inf), val)
                    pbar.update(1)
        finally:
            pbar.close()
        return nblast_scores

    def _nblast_refine(self, results: pd.DataFrame, query_df: pd.DataFrame,
                       cache: SkeletonVectorCache,
                       neurons: Optional[Dict[int, "navis.TreeNeuron"]] = None) -> pd.DataFrame:
        """Replace vector scores with NBLAST scores for the fetched candidates."""
        self._progress(5, PROFILE_FIRST_TOTAL_STEPS, "Refining scores with NBLAST")
        query_ids = set(self._body_ids(query_df["bodyId"].tolist()))
        # Type-level results also contain intra-type rows whose target
        # is another query member.  They are reference pairs, not candidates
        # for refinement; keep their vector similarity below.
        cand_ids = [
            self._body_id(b) for b, s in zip(results["target_bodyId"],
                                             results["similarity"])
            if self._body_id(b) not in query_ids and pd.notna(s)
        ]
        query_dp = self._dotprops_for_ids(
            self._body_ids(query_df["bodyId"].tolist()), neurons=neurons
        )
        if not query_dp:
            self._log("NBLAST: query dotprops unavailable; keeping vector scores.")
            return results
        cand_dp = self._dotprops_for_ids(cand_ids, neurons=neurons)
        cand_dp = {b: dp for b, dp in cand_dp.items() if dp is not None}
        if not cand_dp:
            self._log("NBLAST: no candidate dotprops; keeping vector scores.")
            return results
        nblast_scores = self._nblast_pairwise(
            query_dp, cand_dp, desc="NBLAST scoring")
        results = results.copy()
        # In type mode, ``results`` also contains the vector-based ordered
        # intra-type pairs.  They are not in the candidate NBLAST map because
        # query members are deliberately excluded from that candidate set;
        # preserve their already-computed intra similarity instead of turning
        # those rows into NaN during refinement.
        results["similarity"] = results.apply(
            lambda row: nblast_scores.get(
                self._body_id(row["target_bodyId"]), row["similarity"]
            ) if bool(row.get("is_same_type", False))
            else nblast_scores.get(self._body_id(row["target_bodyId"]), np.nan),
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
        # Compute the whole pairwise matrix once.  The previous row-by-row
        # implementation recalculated all row norms for every member and was
        # unnecessarily slow for large types.
        pair = pairwise_similarity_matrix(sub, metric)
        total = float(pair.sum()) - n  # drop the diagonal (self = 1)
        return total / (n * (n - 1))

    def _type_member_count(self, type_name: str, id_to_type: Dict[int, str],
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
                    member_ids.add(self._body_id(bid))
                except (TypeError, ValueError):
                    continue
        if query_df is not None and not query_df.empty:
            for _, row in query_df.iterrows():
                if str(row.get("type", "") or "").strip() != str(type_name or "").strip():
                    continue
                try:
                    member_ids.add(self._body_id(row.get("bodyId")))
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
                bid = self._body_id(row["bodyId"])
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

        pair_matrix = pairwise_similarity_matrix(
            np.asarray([X[i] for i, _, _, _ in records], dtype=float),
            self.metric,
        )
        for source_i, target_i in pair_directions.values():
            _, source_bid, source_type, _ = records[source_i]
            _, target_bid, target_type, target_instance = records[target_i]
            score = float(pair_matrix[source_i, target_i])
            row: Dict[str, object] = {
                "source_bodyId": source_bid,
                "source_type": source_type,
                "target_bodyId": target_bid,
                "target_type": target_type,
                "target_instance": target_instance,
                "profile_similarity": np.nan,
                "roi_similarity": np.nan,
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
                self._body_id(row["target_bodyId"])
                for row in subrows
                if row.get("target_bodyId") is not None
                and pd.notna(row.get("similarity"))
            }
            profile_values = [float(row["profile_similarity"])
                              for row in subrows
                              if pd.notna(row.get("profile_similarity"))]
            roi_values = [float(row["roi_similarity"])
                          for row in subrows
                          if pd.notna(row.get("roi_similarity"))]
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
                "roi_similarity": (
                    np.nan if is_intra else
                    (float(np.mean(roi_values)) if roi_values else np.nan)
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
                "roi_similarity": np.nan,
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
        result = pd.DataFrame(agg_rows).reset_index(drop=True)
        result.insert(0, "rank", np.arange(1, len(result) + 1))
        return result

    def _bodyid_dataframe(
        self, rows: List[Dict[str, object]], query_type: str = ""
    ) -> pd.DataFrame:
        """Rank bodyId rows, retaining every compared candidate.

        All compared neurons are kept (no top-N truncation; the candidate cap
        already bounded the pool). Rows without a finite similarity
        (unscorable candidates) are kept but rank after every scored row. For
        type queries every resolved same-type pair is preserved so the query
        does not collapse to one source."""
        if not rows:
            return pd.DataFrame()

        def _sim_key(row: Dict[str, object]) -> float:
            sim = row.get("similarity")
            value = float(sim)
            return -value if np.isfinite(value) else float("inf")

        if self.level == "type" and query_type:
            ordered = sorted(
                rows,
                key=lambda row: (
                    _sim_key(row),
                    self._body_id(row["source_bodyId"]),
                    self._body_id(row["target_bodyId"]),
                ),
            )
        else:
            ordered = sorted(
                rows,
                key=lambda row: (_sim_key(row), self._body_id(row["target_bodyId"])),
            )
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
        query_ids = self._body_ids(query_df["bodyId"].tolist())
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
                q_bid = self._body_id(qrow.bodyId)
                q_idx = np.where(body_ids == q_bid)[0]
                if not len(q_idx):
                    continue
                q_vec = X[q_idx[0]]
                query_pair_X[query_i] = q_vec
                query_pair_mask[query_i] = True
                scores = similarity_matrix(q_vec, X, self.metric)
                source_type = str(getattr(qrow, "type", q_type) or "").strip()
                for i, bid in enumerate(body_ids):
                    bid = self._body_id(bid)
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
                q_vec = self._vector_for_body_id(
                    self._body_id(qrow["bodyId"]), body_ids, X
                )
                if q_vec is None:
                    continue
                scores = similarity_matrix(q_vec, X, self.metric)
                row_intra = self._intra_type_similarity(
                    qrow["type"], body_ids, types, X, self.metric
                )
                for i, bid in enumerate(body_ids):
                    bid = self._body_id(bid)
                    if bid in query_ids_set:
                        continue
                    rows.append({
                        "source_bodyId": self._body_id(qrow["bodyId"]),
                        "source_type": str(qrow["type"] or "").strip(),
                        "target_bodyId": bid,
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
        results = pd.DataFrame(rows).reset_index(drop=True)
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
        query_ids = set(self._body_ids(query_df["bodyId"].tolist()))
        candidate_mask = ~np.isin(body_ids, list(query_ids))

        scores = np.full(len(body_ids), -np.inf)
        for _, qrow in query_df.iterrows():
            q_vec = self._vector_for_body_id(
                self._body_id(qrow["bodyId"]), body_ids, X
            )
            if q_vec is None:
                continue
            s = similarity_matrix(q_vec, X, "cosine")
            scores = np.maximum(scores, s)
        prefilter_idx = np.where(candidate_mask)[0]
        prefilter_idx = prefilter_idx[np.argsort(-scores[prefilter_idx])][: self.candidate_cap]

        if not len(prefilter_idx):
            self._log("NBLAST: no candidates survived the vector prefilter.")
            return pd.DataFrame(), pd.DataFrame()

        # Build dotprops for query + candidates in one load/fetch phase
        # (in microns; NEVER cached). Splitting after construction prevents a
        # query/candidate overlap from triggering two online batch requests.
        cand_ids = [self._body_id(body_ids[i]) for i in prefilter_idx]
        all_dotprop_ids = []
        seen_dotprop_ids = set()
        for bid in list(query_ids) + cand_ids:
            bid = self._body_id(bid)
            if bid not in seen_dotprop_ids:
                seen_dotprop_ids.add(bid)
                all_dotprop_ids.append(bid)
        all_dp = self._dotprops_for_ids(all_dotprop_ids)
        query_dp = {bid: all_dp.get(bid) for bid in query_ids
                    if all_dp.get(bid) is not None}
        if not query_dp:
            raise ValueError("NBLAST: could not build dotprops for the query neuron(s).")
        cand_dp = {bid: all_dp.get(bid) for bid in cand_ids}
        cand_dp = {bid: dp for bid, dp in cand_dp.items() if dp is not None}

        self._log(f"NBLAST: {len(query_dp)} query x {len(cand_dp)} candidates "
                  f"({self.candidate_cap} prefiltered)")
        nblast_scores = self._nblast_pairwise(
            query_dp, cand_dp, desc="NBLAST scoring")

        # Rank candidates by NBLAST score.
        ranked = sorted(nblast_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        id_to_type = {self._body_id(b): t for b, t in zip(body_ids, types)}
        id_to_inst = {
            self._body_id(b): i for b, i in zip(body_ids, data["instances"])
        }

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
                "source_bodyId": self._body_id(query_df["bodyId"].iloc[0]),
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
                q_idx = np.where(body_ids == self._body_id(qrow.bodyId))[0]
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
            # Cap per-type pairs to the sampling cap for the mean.
            vals = sorted(vals, reverse=True)[: TYPE_MEMBER_SAMPLE_CAP]
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
        type_df = pd.DataFrame(agg_rows).reset_index(drop=True)
        type_df.insert(0, "rank", np.arange(1, len(type_df) + 1))
        return bodyid_df, type_df

    def _load_fafb_skeletons(self, body_ids: List[int]
                             ) -> Dict[int, object]:
        """Load FAFB sources following the visualization pipeline:

        1. local first: extrusion-fixed skeletons cached under
           ``cache/{dataset}/API_cache/skeletons/``,
        2. the healed skeleton bundle (``{bodyId}.swc``),
        3. extrusion test on the bundle skeletons (results cached),
        4. online fallback via the CAVE API (token-gated) for ids missing
           locally or flagged by the extrusion test. CAVE replacements are
           prepared ``MeshNeuron`` objects cached as ``.pkl.zst`` files; when
           CAVE cannot repair a flagged tree, a safe local extrusion branch
           cut is attempted in memory and never written to the raw SWC cache.
        """
        from fafb_utils import flag_extrusions, repair_extruded_skeleton

        ids = sorted({int(b) for b in body_ids})
        if not ids:
            return {}
        root = self.project_root
        folder = _dataset_folder(self.dataset)
        loaded: Dict[int, object] = {}

        # 1. Local first: the API skeleton cache holds previously fetched
        #    (extrusion-fixed) skeletons and takes priority over the bundle,
        #    exactly like the visualization pipeline.
        api_dir = root / "cache" / folder / "API_cache" / "skeletons"
        zip_ids: List[int] = []
        for bid in ids:
            nrn = None
            api_pkl = api_dir / f"{bid}.pkl"
            if api_pkl.exists():
                try:
                    with open(api_pkl, "rb") as f:
                        nrn = pickle.load(f)
                except Exception:
                    nrn = None
            if nrn is not None:
                loaded[bid] = nrn
            else:
                zip_ids.append(bid)

        # 2. The healed skeleton bundle (.zst first; ZIP fallback with lazy
        #    per-skeleton conversion).
        if zip_ids:
            try:
                bundle = _fafb_bundle(self.dataset, str(root))
            except Exception:
                bundle = None
            if bundle is not None:
                try:
                    for bid in zip_ids:
                        nrn = _bundle_tree_neuron(bundle, bid)
                        if nrn is not None:
                            loaded[bid] = nrn
                finally:
                    bundle.close()
            else:
                zip_path = _fafb_skeleton_zip_path(self.dataset, str(root))
                if zip_path is not None:
                    import zipfile
                    with zipfile.ZipFile(zip_path, "r") as z:
                        for bid in zip_ids:
                            nrn = _read_fafb_zip_skeleton(z, bid)
                            if nrn is not None:
                                loaded[bid] = nrn

        # 3. Extrusion test on the bundle-sourced skeletons (cached per
        #    neuron; unchecked ids are analyzed in a parallel batch).
        zip_loaded = {b: loaded[b] for b in zip_ids if b in loaded}
        extrusion_ids = flag_extrusions(
            str(root), folder, zip_loaded,
            verbose=self.verbose, log=self._log,
            n_workers=self.n_workers,
        )

        missing = [b for b in ids if b not in loaded]
        extrusion_ids = sorted(set(extrusion_ids))
        # 4. Online fallback (token-gated; matches the visualization
        #    pipeline's CAVE_TOKEN check). Missing neurons may reuse a
        #    prepared mesh cache. Extrusion replacements must bypass that
        #    cache so an old prepared copy cannot mask the repair.
        if missing:
            self._log(f"FAFB: {len(missing)} skeleton(s) missing locally; "
                      "trying the CAVE API fallback.")
            loaded.update(self._fafb_cave_fallback(missing))
        if extrusion_ids:
            self._log(
                f"FAFB: refreshing {len(extrusion_ids)} extrusion-affected "
                "mesh(es) from the CAVE API.")
            cave_fixed = self._fafb_cave_fallback(
                extrusion_ids, force_refresh=True)
            cave_fixed = {
                int(body_id): neuron
                for body_id, neuron in cave_fixed.items()
            }
            loaded.update(cave_fixed)
            repair_statuses = {
                int(body_id): "api_repaired" for body_id in cave_fixed
            }

            # If CAVE returned only a partial batch (or was unavailable),
            # prune a diagnosed local branch instead of silently retaining
            # the known-bad source.  Repairs stay transient and are not
            # written into the raw SWC cache.
            for body_id in extrusion_ids:
                body_id = int(body_id)
                if body_id in cave_fixed:
                    continue
                if body_id not in loaded:
                    repair_statuses[body_id] = "api_failed"
                    continue
                repaired, repair_stats = repair_extruded_skeleton(loaded[body_id])
                if repair_stats.get("repaired"):
                    loaded[body_id] = repaired
                    repair_statuses[body_id] = "local_fallback"
                    self._log(
                        f"FAFB: CAVE fetch failed for {body_id}; pruned "
                        f"{repair_stats['removed_nodes']} extrusion node(s) "
                        "locally.")
                else:
                    self._log(
                        f"FAFB: CAVE fetch failed for {body_id}; no safe "
                        "local branch cut was available.")
                    repair_statuses[body_id] = "api_failed"

            # Keep detection and repair outcomes together. A local fallback
            # remains flagged, so a subsequent run retries the CAVE request
            # without re-running the expensive extrusion detector.
            try:
                from fafb_utils import set_extrusion_repair_status

                set_extrusion_repair_status(
                    str(root), folder, repair_statuses)
            except Exception as exc:
                self._log(f"FAFB: could not save extrusion repair status: {exc}")
        return loaded

    def _fafb_cave_fallback(
            self, body_ids: List[int], force_refresh: bool = False
            ) -> Dict[int, object]:
        """Fetch FAFB meshes through CAVE (token-gated).

        CAVE fallback remains mesh-native. NBLAST callers can still use local
        healed SWC skeletons; a CAVE mesh is not silently skeletonized just
        to satisfy that separate backend. ``force_refresh=True`` is reserved
        for replacing extrusion-affected local data and bypasses the
        prepared-mesh cache read while still writing the repaired mesh cache.
        """
        from utils.flywire_readiness import flywire_skeleton_readiness

        status = flywire_skeleton_readiness(self.dataset, self.project_root)
        if not status.get("cave_token"):
            self._log("FAFB API fallback skipped: CAVE_TOKEN is not "
                      "configured; using local skeleton data only.")
            return {}
        from cave_data_fetcher import CAVEDataFetcher

        pbar = tqdm(total=len(body_ids), desc="Fetching meshes (CAVE)",
                    unit="neuron", disable=not self.verbose, leave=False,
                    file=sys.stdout)
        out: Dict[int, navis.MeshNeuron] = {}
        try:
            fetcher = CAVEDataFetcher(
                dataset=_dataset_folder(self.dataset),
                project_root=str(self.project_root),
                verbose=False,
            )
            soma_positions = _load_flywire_soma_positions(
                self.dataset, self.project_root, body_ids)
            neurons = fetcher.fetch_fafb_meshes(
                [int(b) for b in body_ids], use_cache=True,
                simplify_mesh=FLYWIRE_MESH_CACHE_SIMPLIFICATION,
                soma_simplification=FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION,
                soma_radius=FLYWIRE_MESH_CACHE_SOMA_RADIUS,
                soma_positions=soma_positions,
                force_refresh=force_refresh,
            )
            for n in neurons:
                bid = getattr(n, "id", None)
                if bid is not None:
                    out[int(bid)] = n
                pbar.update(1)
                pbar.set_postfix_str(str(bid))
        except Exception as exc:
            self._log(f"FAFB API fallback failed ({exc}); keeping local "
                      "skeleton data only.")
        finally:
            pbar.close()
        return out

    def _dotprops_for_ids(self, body_ids: List[int],
                          neurons: Optional[Dict[int, "navis.TreeNeuron"]] = None,
                          desc: str = "Building dotprops"
                          ) -> Dict[int, Optional["navis.core.dotprop.Dotprops"]]:
        """Load skeletons and build dotprops in microns.

        ``neurons`` supplies transient in-memory raw skeletons (profile-first
        fetches) so they are not re-fetched; anything missing is resolved
        through the shared raw cache and batched online fetch (raw skeletons
        are always persisted as compressed SWC). For
        FlyWire datasets the raw sources follow the healed-skeleton pipeline
        (local API cache / healed bundle -> extrusion check -> token-gated
        CAVE fallback); visualization simp90 pickles are never used."""
        out: Dict[int, Optional[navis.core.dotprop.Dotprops]] = {}
        local_neurons: Dict[int, object] = {
            int(bid): neuron for bid, neuron in (neurons or {}).items()
            if _neuron_rep(neuron) == "skeleton"
        }

        raw_cache = find_similar_raw_cache(
            self.dataset, project_root=str(self.project_root),
            n_workers=self.n_workers, verbose=False,
        )

        # Keep raw vector rows and raw skeleton persistence in the same cache
        # transaction. This makes an NBLAST-first run useful to a later
        # vector-mode or visualization run.
        def _cache_raw_neurons(mapping: Dict[int, object]) -> None:
            rows = []
            for bid, neuron in mapping.items():
                try:
                    if _neuron_rep(neuron) != "skeleton":
                        continue
                    _, vec = vectorize_neuron(neuron)
                    rows.append((int(bid), vec, "skeleton"))
                except Exception:
                    continue
            if rows:
                try:
                    raw_cache.append_vectors(rows, vector_basis=VECTOR_BASIS_RAW)
                except Exception:
                    pass
            try:
                raw_cache.persist_skeletons(mapping)
            except Exception:
                pass

        # The shared raw cache has priority for every dataset.
        for body_id in body_ids:
            bid = int(body_id)
            if bid in local_neurons:
                continue
            cached = raw_cache.load_skeleton(bid)
            if cached is not None:
                local_neurons[bid] = cached

        # FlyWire skeletons: resolve every non-transient id through the
        # FAFB pipeline once for the whole batch (the extrusion check is
        # cached, so repeated batches reuse earlier results).
        if self._is_flywire():
            fetched = self._load_fafb_skeletons(
                [int(b) for b in body_ids if int(b) not in local_neurons]
            )
            local_neurons.update(fetched)
            _cache_raw_neurons(fetched)
        else:
            # Resolve raw-cache hits first, then issue one combined NeuPrint
            # fetch for all remaining dotprops. This covers cache-direct
            # NBLAST, where no profile-first in-memory map is available.
            missing_online = []
            for body_id in body_ids:
                bid = int(body_id)
                if bid in local_neurons:
                    continue
                missing_online.append(bid)
            if missing_online:
                fetched = fetch_skeletons_on_demand_batch(
                    self.dataset,
                    missing_online,
                    project_root=str(self.project_root),
                    persist=self.cache_fetched_skeletons,
                    level=VECTOR_BASIS_RAW,
                    raw_cache=raw_cache,
                    vector_cache=raw_cache,
                )
                local_neurons.update(fetched)
                # Compatibility with singular-fetch overrides that predate
                # the raw_cache/vector_cache keywords.
                _cache_raw_neurons(fetched)

        _cache_raw_neurons({
            int(b): n for b, n in local_neurons.items()
            if int(b) in {int(x) for x in body_ids}
        })

        pbar = tqdm(body_ids, desc=desc, unit="neuron",
                    disable=not self.verbose, leave=False, file=sys.stdout)
        try:
            for bid in pbar:
                bid = int(bid)
                pbar.set_postfix_str(f"{bid}")
                nrn = local_neurons.get(bid)
                if nrn is None:
                    # The FAFB pipeline (API cache -> healed bundle ->
                    # extrusion check -> CAVE fallback) already ran for this
                    # id; there is no other source to try.
                    out[bid] = None
                    continue
                try:
                    if _neuron_rep(nrn) != "skeleton":
                        out[bid] = None
                        continue
                    nrn_um = nrn / 1000.0  # nanometres -> microns (NBLAST requirement)
                    out[bid] = navis.make_dotprops(nrn_um, k=20)
                except Exception:
                    out[bid] = None
        finally:
            pbar.close()
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
            "candidate_source": getattr(self, "resolved_candidate_source",
                                        self.candidate_source),
            "candidate_cap": self.candidate_cap,
            "note": ("results.csv/type_summary.csv contain EVERY compared "
                     "candidate (never truncated; bounded by candidate_cap); "
                     "visualize_top_n applies to visualization only"),
            "visualize_top_n": self.visualize_top_n,
            "visualize_by": self.visualize_by,
            "cache_raw_skeletons": self.cache_fetched_skeletons,
            "raw_skeleton_cache": str(
                find_similar_dataset_cache(
                    self.dataset, project_root=str(self.project_root),
                    verbose=False,
                ).skeleton_dir
            ),
            "raw_vector_cache": str(
                find_similar_dataset_cache(
                    self.dataset, project_root=str(self.project_root),
                    verbose=False,
                ).parquet_path
            ),
            "query_bodyIds": (
                self._body_ids(query_df["bodyId"].tolist())
                if self._is_flywire()
                else [int(b) for b in query_df["bodyId"].tolist()]
            ),
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
        # Unscorable candidates (NaN similarity) are never rendered.
        if "similarity" in work.columns:
            work = work[work["similarity"].notna()]

        layers: List[List[Union[int, str]]] = []
        names: List[str] = []

        def _body_ids(frame: pd.DataFrame) -> List[Union[int, str]]:
            values = frame.get("bodyId", pd.Series(dtype=object)).tolist()
            ids: List[Union[int, str]] = []
            for value in values:
                try:
                    if pd.isna(value):
                        continue
                    ids.append(self._body_id(value))
                except (TypeError, ValueError):
                    continue
            return list(dict.fromkeys(ids))

        def _body_id_in(value, ids: set) -> bool:
            try:
                return not pd.isna(value) and self._body_id(value) in ids
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
                    bid = self._body_id(row.get("target_bodyId"))
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
            pipeline = str(
                self.visualization_settings.get(
                    "neuprint_skeleton_pipeline", "fine"
                ) or "fine"
            ).strip().lower()
            viz_kwargs = {
                "dataset": self.dataset,
                "output_dir": str(run_dir),
                "neuron_layers": layers,
                "custom_layer_names": names,
                "saveas": _dataset_folder(self.dataset),
                "include_timestamp": False,
                "skip_synapse": True,
                # Analysis visualizations default to the light-weight line
                # representation. Callers can explicitly request tube/fine
                # rendering through visualization_settings.
                "skeleton_mode": self.visualization_settings.get(
                    "skeleton_mode", "line"
                ),
                "legend_mode": "layer" if self.visualize_by == "type" else "single",
                "brain_mesh": "template",
                "export_views": False,
                "show_fig": False,
                "cache_neurons": (
                    True if is_flywire_dataset(self.dataset)
                    else pipeline not in {
                        "fast", "direct", "artistic", "fine_opt1"
                    }
                ),
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

            # The shared panel returns None when its analysis default is
            # selected. Resolve that default at the analysis boundary so the
            # dedicated Skeleton tab can retain its own historical defaults.
            if viz_kwargs.get("skeleton_mesh_simplification") is None:
                viz_kwargs["skeleton_mesh_simplification"] = (
                    default_analysis_skeleton_mesh_simplification(
                        self.dataset, pipeline
                    )
                )

            vs = VisualizeSkeleton(
                **viz_kwargs,
            )
            vs.plot_neurons()
            viz_dir = run_dir / f"plot-3d_{_dataset_folder(self.dataset)}"
            self._log(f"3D visualization saved to: {viz_dir}")
        except Exception as ex:
            self._log(f"3D visualization failed (search results kept): {ex}")

    def _type_members_from_cache(self, type_name: str) -> List[Union[int, str]]:
        """Member bodyIds of a type from the vector cache (capped to the
        per-type sample size so type-level renders stay bounded); falls back
        to the neuron table / index when the dataset has no vector cache."""
        try:
            data = find_similar_dataset_cache(
                self.dataset, project_root=str(self.project_root),
                n_workers=self.n_workers, verbose=False,
            ).load()
            members: List[Union[int, str]] = []
            if data is not None:
                members = [self._body_id(b)
                           for b, t in zip(data["bodyIds"], data["types"])
                           if t == type_name]
            if not members:
                type_map, _ = _load_neuron_type_map(
                    self.dataset, str(self.project_root)
                )
                members = [self._body_id(b) for b, t in type_map.items()
                           if t == type_name]
            return members[: TYPE_MEMBER_SAMPLE_CAP]
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

    def _vectors(
        dataset: str, bids: List[Union[int, str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        cache = find_similar_dataset_cache(
            dataset, project_root=project_root, verbose=verbose
        )
        X, ok, _ = cache.vectors_for(bids, compute_missing=True)
        return X, ok

    src_ids = [
        _canonical_dataset_body_id(source_dataset, b)
        for b in results_df["source_bodyId"].tolist()
    ]
    tgt_ids = [
        _canonical_dataset_body_id(target_dataset, b)
        for b in results_df["target_bodyId"].tolist()
    ]
    # Keep the public result frame aligned with the same dataset-aware
    # representation used by the vector cache lookup.  NeuPrint retains its
    # historical integer output; FlyWire/FAFB/BANC results are strings.
    results_df = results_df.copy()
    results_df["source_bodyId"] = src_ids
    results_df["target_bodyId"] = tgt_ids
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

    results_df["morph_cosine"] = cos
    results_df["morph_pearson"] = pears
    return results_df
