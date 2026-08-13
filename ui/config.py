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
TAB_OUTPUT_DIRS_KEY = "tab_output_dirs"


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


def _resolve_output_path(value: str) -> str:
    """Normalize a user-entered output path without touching the filesystem."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return str(path)


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
    path = Path(_resolve_output_path(raw))
    if create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False, None
    config = load_local_config()
    config["default_output_dir"] = str(path)
    saved = save_local_config(config)
    return saved, str(path) if saved else None


def get_tab_output_override(scope: str | None) -> str | None:
    """Return a persisted tab-specific output directory, if one exists."""
    key = str(scope or "").strip()
    if not key:
        return None
    overrides = load_local_config().get(TAB_OUTPUT_DIRS_KEY, {})
    if not isinstance(overrides, dict):
        return None
    value = overrides.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    return str(path) if path.is_absolute() else None


def has_tab_output_override(scope: str | None) -> bool:
    """Whether *scope* has its own path instead of inheriting the default."""
    return get_tab_output_override(scope) is not None


def get_tab_output_dir(scope: str | None) -> str:
    """Resolve a tab path, falling back to the global output default."""
    return get_tab_output_override(scope) or get_default_output_dir()


def set_tab_output_dir(
    scope: str | None,
    value: str,
    create: bool = False,
) -> tuple:
    """Persist or clear one tab's output-directory override.

    An empty value removes the override and makes the tab inherit the global
    default again.  Tab overrides never modify ``default_output_dir``.
    """
    key = str(scope or "").strip()
    if not key:
        return False, None
    raw = (value or "").strip()
    config = load_local_config()
    overrides = config.get(TAB_OUTPUT_DIRS_KEY, {})
    if not isinstance(overrides, dict):
        overrides = {}
    if not raw:
        overrides.pop(key, None)
        if overrides:
            config[TAB_OUTPUT_DIRS_KEY] = overrides
        else:
            config.pop(TAB_OUTPUT_DIRS_KEY, None)
        saved = save_local_config(config)
        return saved, get_default_output_dir() if saved else None

    path = Path(_resolve_output_path(raw))
    if create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False, None
    overrides[key] = str(path)
    config[TAB_OUTPUT_DIRS_KEY] = overrides
    saved = save_local_config(config)
    return saved, str(path) if saved else None


def clear_tab_output_overrides() -> bool:
    """Clear every tab override so all tabs inherit the global default."""
    config = load_local_config()
    config.pop(TAB_OUTPUT_DIRS_KEY, None)
    return save_local_config(config)


def get_auto_suggest_enabled() -> bool:
    """Whether the neuron-input auto-suggest + history is enabled (Settings)."""
    return bool(load_local_config().get("auto_suggest_enabled", True))


def set_auto_suggest_enabled(enabled: bool) -> bool:
    """Persist the input auto-suggestion toggle; applies immediately."""
    config = load_local_config()
    config["auto_suggest_enabled"] = bool(enabled)
    return save_local_config(config)

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
    "similarity_metric": "cosine",  # Sort By (homolog candidates): sorting only — all metrics are always computed
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
    "morph_candidate_expansion": 3,
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
PATH_MODES = ["all", "shortest"]

# Skeleton modes
SKELETON_MODES = ["tube", "line"]

# Brain mesh options
BRAIN_MESH_OPTIONS = ["template", "whole", "none"]

# App settings
APP_TITLE = "DROCAT - Connectome Analysis Toolkit"
APP_VERSION = "4.5.0"
APP_PORT = 8080
APP_HOST = "127.0.0.1"
