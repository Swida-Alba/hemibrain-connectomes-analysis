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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT.parent / "local_data"
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
    "banc:v888",
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
    "banc:v888",
    "fib19:v1.0",
    "mushroombody",
]

# FlyWire datasets (require local files + CAVE token for API access)
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
    "pathfinding": "Bidirectional",
    "network_layout": "distributed",
    "use_cache": True,
    "top_k": 15,
    "top_m": 5,
    "similarity_metric": "rank_union",
    "top_n": 30,
    "match_algorithm": "cds",
}

# Pathfinding algorithms
PATHFINDING_ALGORITHMS = ["Bidirectional", "DP", "MemoizedDFS", "DFS"]

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
