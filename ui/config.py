"""
DROCAT UI Configuration
Datasets, defaults, and path settings.
"""

from pathlib import Path
import json

# Project root (parent of ui/)
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_data"
TOKEN_FILE = PROJECT_ROOT / "token_info.txt"
LOCAL_CONFIG_FILE = PROJECT_ROOT / "ui" / "local_config.json"


def load_local_config() -> dict:
    """Load the user-editable local UI configuration (output dir, etc.)."""
    try:
        if LOCAL_CONFIG_FILE.exists():
            data = json.loads(LOCAL_CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (OSError, ValueError):
        pass
    return {}


def save_local_config(config: dict) -> None:
    """Persist the local UI configuration."""
    try:
        LOCAL_CONFIG_FILE.write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return True
    except OSError:
        return False


def get_default_output_dir() -> str:
    """Resolve the effective default output directory (user override wins)."""
    override = load_local_config().get("default_output_dir")
    if override and Path(override).is_absolute():
        return override
    return str(DEFAULT_OUTPUT_DIR)


def set_default_output_dir(value: str, create: bool = True) -> tuple:
    """Persist the UI default output directory permanently (across sessions).

    Relative paths are resolved against PROJECT_ROOT; the directory is
    created when *create* is true. Empty values clear the override so the
    project default is used again.

    Returns (saved: bool, effective_path: Optional[str]).
    """
    raw = (value or "").strip()
    if not raw:
        config = load_local_config()
        config.pop("default_output_dir", None)
        saved = save_local_config(config)
        return saved, str(DEFAULT_OUTPUT_DIR) if saved else None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False, None
    config = load_local_config()
    config["default_output_dir"] = str(path)
    saved = save_local_config(config)
    return saved, str(path) if saved else None

# Available datasets (static fallback - use dataset_service for dynamic fetching)
DATASETS = [
    "male-cns:v1.0",
    "male-cns:v0.9",
    "hemibrain:v1.2.1",
    "hemibrain:v1.1",
    "optic-lobe:v1.1",
    "optic-lobe:v1.0.1",
    "manc:v1.2.3",
    "manc:v1.2.1",
    "manc:v1.0",
    "fib19:v1.0",
    "mushroombody",
    "flywire_FAFB_v783",
    "flywire_BANC_v888",
    "flywire_BANC_v626",
]

# NeuPrint server datasets (can be fetched dynamically via /api/dbmeta/datasets)
NEUPRINT_DATASETS = [
    "male-cns:v1.0",
    "male-cns:v0.9",
    "hemibrain:v1.2.1",
    "hemibrain:v1.1",
    "optic-lobe:v1.1",
    "optic-lobe:v1.0.1",
    "manc:v1.2.3",
    "manc:v1.2.1",
    "manc:v1.0",
    "fib19:v1.0",
    "mushroombody",
]

# FlyWire datasets (require converted local files; CAVE token is only needed
# when a workflow explicitly fetches data or skeletons through the CAVE API)
FLYWIRE_DATASETS = [
    "flywire_FAFB_v783",
    "flywire_BANC_v888",
    "flywire_BANC_v626",
]

# Default parameter values
DEFAULTS = {
    "min_synapse_num": 3,
    "min_ratio": 0.0,
    "min_traversal_probability": 0.0,
    "max_interlayer": 2,
    "edgeN_limit": 500,
    "filter_by": "bodyId",
    "output_format": "csv",
    "pathfinding": "MemoizedDFS",
    "network_layout": "distributed",
    "use_cache": True,
    "top_k": 15,
    "top_m": 5,
    "similarity_metric": "rank_union",
    "top_n": 30,
    "match_algorithm": "cds",
    # Find Similar Neurons (morphological mode)
    "morph_level": "auto",
    "morph_method": "vector",
    "morph_metric": "cosine",
    "morph_top_n": 20,
    "nblast_prefilter": 100,
    "n_per_type": 5,
    "candidate_source": "auto",
    "fetch_top_n": 20,
    "morph_visualize_top_n": 6,
    "morph_visualize_by": "type",
}

# Pathfinding algorithms (names match the FastGraph implementations:
# MemoizedDFS = memoized DFS forward, DFS = memoized DFS backward,
# MeetInMiddle = meet-in-the-middle)
PATHFINDING_ALGORITHMS = ["Bidirectional", "DP", "MemoizedDFS", "MeetInMiddle", "DFS"]

# Columns searched when resolving source/target neuron names
SEARCH_COLUMNS = ["auto", "type", "instance", "bodyId"]

# Filter options
FILTER_OPTIONS = ["bodyId", "type"]

# Output formats
OUTPUT_FORMATS = ["csv", "xlsx"]

# Network layouts
NETWORK_LAYOUTS = ["distributed", "circular", "shell", "spring"]

# Similarity metrics
SIMILARITY_METRICS = ["rank_union", "jaccard", "cosine", "rank_corr"]

# Match algorithms for NeuronBridge
MATCH_ALGORITHMS = ["cds", "pppm", "both"]

# Comparison modes
COMPARISON_MODES = ["path", "edge"]

# Skeleton modes
SKELETON_MODES = ["tube", "line"]

# Brain mesh options
BRAIN_MESH_OPTIONS = ["template", "whole", "none"]

# App settings
APP_TITLE = "DROCAT - Connectome Analysis Toolkit"
APP_VERSION = "4.5.0"
APP_PORT = 8080
APP_HOST = "127.0.0.1"
