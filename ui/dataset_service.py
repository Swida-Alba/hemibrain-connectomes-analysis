"""
Dataset Service for DROCAT UI
Fetches available datasets from NeuPrint server dynamically.
"""

import json
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .config import PROJECT_ROOT


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    source: str  # 'neuprint', 'flywire', 'local'
    available: bool = False
    neuron_count: int = 0
    typed_count: int = 0
    local_cache: bool = False
    local_prepared: bool = False  # Has local data files ready to use
    display_name: str = ""  # Human-readable name from Codex/server
    metadata: Dict = field(default_factory=dict)
    error: Optional[str] = None


# Dataset name normalization: folder name <-> dataset name
# e.g. 'hemibrain_v1_2_1' <-> 'hemibrain:v1.2.1'
def folder_to_dataset(folder_name: str) -> str:
    """Convert a folder name to a dataset identifier."""
    # FlyWire datasets keep their names as-is
    if "flywire" in folder_name.lower():
        return folder_name
    # NeuPrint: first _ becomes :, remaining _ become .
    # e.g. hemibrain_v1_2_1 -> hemibrain:v1.2.1
    #      male-cns_v0_9 -> male-cns:v0.9
    parts = folder_name.split("_", 1)
    if len(parts) == 2:
        prefix, version = parts
        version = version.replace("_", ".")
        return f"{prefix}:{version}"
    return folder_name


def dataset_to_folder(dataset: str) -> str:
    """Convert a dataset identifier to a folder name."""
    # FlyWire datasets keep their names as-is
    if "flywire" in dataset.lower():
        return dataset
    # NeuPrint: : becomes _, . becomes _
    return dataset.replace(":", "_").replace(".", "_")


def is_flywire_dataset(dataset: str) -> bool:
    """Return whether *dataset* is a FlyWire identifier.

    The NeuPrint server metadata lists a hidden dataset named ``banc:v888``
    that is not queryable through the API (BANC is served via FlyWire/
    Codex).  Do not classify that identifier as the local FlyWire BANC
    release merely because it contains the word ``banc``.
    """
    normalized = dataset.strip().lower()
    return normalized.startswith("flywire_") or "fafb" in normalized


class DatasetService:
    """Service for fetching and managing dataset availability."""

    # Known NeuPrint dataset candidates (fallback if /api/dbmeta/datasets fails)
    NEUPRINT_CANDIDATES = [
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

    # FlyWire/Codex datasets (from https://codex.flywire.ai/)
    # These require local files + CAVE token for API access
    FLYWIRE_DATASETS = [
        "flywire_FAFB_v783",
        "flywire_BANC_v888",
        "flywire_BANC_v626",
    ]

    # Codex display info (fetched from codex.flywire.ai rendered page)
    CODEX_DATASETS = {
        "flywire_FAFB_v783": {"display": "FAFB v783 (CB)", "desc": "Female Adult Fly Brain", "neurons": 139255},
        "flywire_BANC_v888": {"display": "BANC v888 (CNS)", "desc": "Brain and Nerve Cord", "neurons": 158262},
        "flywire_BANC_v626": {"display": "BANC v626 (CNS)", "desc": "Brain and Nerve Cord (older)", "neurons": 115151},
    }

    NEUPRINT_SERVER = "https://neuprint.janelia.org"
    CODEX_URL = "https://codex.flywire.ai/"

    def __init__(self):
        self._token: Optional[str] = None
        self._cave_token: Optional[str] = None
        self._cache: Dict[str, DatasetInfo] = {}
        self._lock = threading.Lock()
        self._datasets_dir = PROJECT_ROOT / "datasets"
        self._cache_dir = PROJECT_ROOT / "cache"
        self._available_neuprint: Optional[List[str]] = None
        self._server_datasets: Dict[str, dict] = {}  # Full server response from /api/dbmeta/datasets
        self._last_fetch_time: float = 0

    def _load_tokens(self):
        """Load tokens from token_info_local.txt or token_info.txt."""
        if self._token is not None and self._cave_token is not None:
            return

        # Parse the template first and the local file second so local values
        # override only their own key.  Stopping after the first token caused
        # a local NeuPrint-only file to hide a valid CAVE token (and vice
        # versa) from the other file.
        loaded = {}
        for filename in ["token_info.txt", "token_info_local.txt"]:
            is_local = filename == "token_info_local.txt"
            token_path = PROJECT_ROOT / filename
            if token_path.exists():
                try:
                    with open(token_path, "r") as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("NEUPRINT_TOKEN="):
                                token = line.split("=", 1)[1].strip().strip("'\"")
                                if is_local:
                                    # A blank local value explicitly clears
                                    # the template value for this key.
                                    loaded["neuprint"] = (
                                        token if token and not token.startswith("YOUR_") else None
                                    )
                                elif "neuprint" not in loaded and token and not token.startswith("YOUR_"):
                                    loaded["neuprint"] = token
                            elif line.startswith("CAVE_TOKEN="):
                                token = line.split("=", 1)[1].strip().strip("'\"")
                                if is_local:
                                    loaded["cave"] = (
                                        token if token and not token.startswith("YOUR_") else None
                                    )
                                elif "cave" not in loaded and token and not token.startswith("YOUR_"):
                                    loaded["cave"] = token
                except OSError:
                    continue

        if self._token is None:
            self._token = loaded.get("neuprint")
        if self._cave_token is None:
            self._cave_token = loaded.get("cave")

    def get_token(self) -> Optional[str]:
        """Get NeuPrint token."""
        self._load_tokens()
        return self._token

    def get_cave_token(self) -> Optional[str]:
        """Get CAVE token."""
        self._load_tokens()
        return self._cave_token

    def fetch_neuprint_datasets(self) -> List[str]:
        """
        Fetch the FULL list of available NeuPrint datasets from the server API.
        Uses /api/dbmeta/datasets endpoint with Bearer token auth.
        Falls back to probing known candidates if API call fails.
        Returns a list of available dataset names.
        """
        self._load_tokens()

        if not self._token:
            return []

        # Try the proper API endpoint first
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self._token}",
                "Content-type": "application/json",
            }
            r = requests.get(
                f"{self.NEUPRINT_SERVER}/api/dbmeta/datasets",
                headers=headers,
                timeout=15,
            )
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict):
                    # Store full server metadata, excluding datasets the
                    # server marks as hidden (e.g. banc:v888): they are
                    # listed but not queryable through the API.
                    self._server_datasets = data
                    available = sorted(
                        name for name, meta in data.items()
                        if not (isinstance(meta, dict)
                                and str(meta.get("hidden", "")).lower() == "true")
                    )
                    self._available_neuprint = available
                    self._last_fetch_time = time.time()
                    return available
        except Exception:
            pass

        # Fallback: probe known candidates individually
        available = []
        for ds in self.NEUPRINT_CANDIDATES:
            info = self._probe_neuprint_dataset(ds)
            if info.available:
                available.append(ds)

        self._available_neuprint = available
        self._last_fetch_time = time.time()
        return available

    def get_all_datasets(self) -> List[str]:
        """
        Get list of all available datasets (NeuPrint + FlyWire).
        If we have fetched from the server, includes ALL server datasets
        (including older versions like hemibrain:v1.1, fib19:v1.0, etc.).
        """
        if self._available_neuprint is not None:
            return self._available_neuprint + self.FLYWIRE_DATASETS
        return self.NEUPRINT_CANDIDATES + self.FLYWIRE_DATASETS

    def get_neuprint_datasets(self) -> List[str]:
        """Get list of available NeuPrint datasets."""
        if self._available_neuprint is not None:
            return self._available_neuprint.copy()
        return self.NEUPRINT_CANDIDATES.copy()

    def get_flywire_datasets(self) -> List[str]:
        """Get list of FlyWire datasets."""
        return self.FLYWIRE_DATASETS.copy()

    def check_dataset_availability(self, dataset: str) -> DatasetInfo:
        """
        Check if a dataset is available.
        Uses cached results if available.
        """
        with self._lock:
            if dataset in self._cache:
                return self._cache[dataset]

        self._load_tokens()

        # Determine source
        if is_flywire_dataset(dataset):
            info = self._check_flywire_dataset(dataset)
        else:
            info = self._probe_neuprint_dataset(dataset)

        # Check local cache/prepared status
        info.local_cache = self._check_local_cache(dataset)
        info.local_prepared = self._check_local_prepared(dataset)
        if info.source == "flywire":
            # FlyWire has no server-backed dataset status in this UI.  A
            # directory (or a lone neuron table) is not enough to call it
            # available; both converter outputs must be present.
            info.available = info.local_prepared
            if not info.local_prepared:
                info.error = "Local FlyWire neuron and connection tables are not both prepared."

        # Set display name
        if not info.display_name:
            if dataset in self.CODEX_DATASETS:
                info.display_name = self.CODEX_DATASETS[dataset]["display"]
            else:
                info.display_name = dataset

        # Cache result
        with self._lock:
            self._cache[dataset] = info

        return info

    def fetch_codex_datasets(self) -> Dict[str, dict]:
        """
        Fetch available FlyWire datasets from Codex (codex.flywire.ai).
        Returns dict of {dataset_name: {display, desc, neurons}}.
        Falls back to hardcoded CODEX_DATASETS if fetch fails.
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            r = requests.get(self.CODEX_URL, timeout=10)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                # Look for dataset cards in the rendered HTML
                # The page structure has dataset names like "FAFB v783 (CB)"
                text = soup.get_text()
                # Parse known patterns
                import re
                patterns = [
                    (r'FAFB\s+v(\d+)', 'flywire_FAFB_v{}', 'FAFB v{} (CB)', 'Female Adult Fly Brain'),
                    (r'BANC\s+v(\d+)', 'flywire_BANC_v{}', 'BANC v{} (CNS)', 'Brain and Nerve Cord'),
                ]
                found = {}
                for pattern, key_fmt, display_fmt, desc in patterns:
                    matches = re.findall(pattern, text)
                    for ver in matches:
                        key = key_fmt.format(ver)
                        display = display_fmt.format(ver)
                        found[key] = {"display": display, "desc": desc, "neurons": 0}
                if found:
                    # Update our CODEX_DATASETS with fetched info
                    self.CODEX_DATASETS.update(found)
                    # Also update FLYWIRE_DATASETS list
                    for k in found:
                        if k not in self.FLYWIRE_DATASETS:
                            self.FLYWIRE_DATASETS.append(k)
                    return found
        except Exception:
            pass
        return self.CODEX_DATASETS

    def _probe_neuprint_dataset(self, dataset: str) -> DatasetInfo:
        """Check NeuPrint dataset availability.
        Uses server metadata from /api/dbmeta/datasets if available (fast).
        Falls back to individual query for neuron counts (slow).
        """
        info = DatasetInfo(name=dataset, source="neuprint")

        if not self._token:
            info.error = "No NeuPrint token configured"
            return info

        # Fast path: use server metadata if we have it
        if self._server_datasets and dataset in self._server_datasets:
            info.available = True
            server_info = self._server_datasets[dataset]
            info.metadata = {
                "server": self.NEUPRINT_SERVER,
                "last_mod": server_info.get("last-mod", ""),
                "uuid": server_info.get("uuid", ""),
                "rois": server_info.get("ROIs", []),
            }
            # Set display name from server info
            info.display_name = dataset
            return info

        # Slow path: individual query (only for datasets not in server list)
        try:
            from neuprint import Client

            client = Client(self.NEUPRINT_SERVER, dataset, self._token)

            # Query neuron count
            result = client.fetch_custom(
                "MATCH (n:Neuron) RETURN count(n) as total, "
                "sum(CASE WHEN n.type IS NOT NULL AND n.type <> '' THEN 1 ELSE 0 END) as typed"
            )

            if not result.empty:
                info.available = True
                info.neuron_count = int(result["total"].iloc[0])
                info.typed_count = int(result["typed"].iloc[0])
                info.metadata = {
                    "server": self.NEUPRINT_SERVER,
                    "checked_at": datetime.now().isoformat(),
                }
            else:
                info.error = "Empty response from server"

        except Exception as e:
            info.error = str(e)

        return info

    def _check_flywire_dataset(self, dataset: str) -> DatasetInfo:
        """Check FlyWire dataset availability."""
        info = DatasetInfo(name=dataset, source="flywire")

        local_path = self._get_dataset_path(dataset)
        if local_path and local_path.exists():
            info.available = True
            info.local_cache = True

            metadata_file = self._find_metadata_file(dataset)
            if metadata_file and metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        meta = json.load(f)
                        info.neuron_count = meta.get("neuron_counts", {}).get("total", 0)
                        info.typed_count = meta.get("neuron_counts", {}).get("typed", 0)
                        info.metadata = meta
                except Exception:
                    pass
        else:
            info.error = (
                "Local FlyWire dataset is not prepared. Put the raw Codex files "
                "under datasets/<dataset>/downloads/ and run the converter."
            )

        return info

    def _check_local_cache(self, dataset: str) -> bool:
        """Check if dataset has local cache files."""
        dataset_path = self._get_dataset_path(dataset)
        if dataset_path and dataset_path.exists():
            for pattern in ["*_neuron_df.csv", "*_neuron_df.parquet", "*_allneurons*.csv"]:
                if list(dataset_path.glob(pattern)):
                    return True

        cache_path = self._cache_dir / dataset_to_folder(dataset)
        if cache_path.exists():
            if (cache_path / "connections.parquet").exists():
                return True
            if (cache_path / "neuron_index.parquet").exists():
                return True

        return False

    def _check_local_prepared(self, dataset: str) -> bool:
        """Check if dataset has local data files ready for analysis (not just cache)."""
        dataset_path = self._get_dataset_path(dataset)
        if dataset_path and dataset_path.exists():
            if not is_flywire_dataset(dataset):
                # NeuPrint can legitimately use a local neuron table while
                # connections remain server-backed; preserve that behavior.
                return any(
                    path
                    for pattern in ("*_neuron_df.csv", "*_neuron_df.parquet")
                    for path in dataset_path.glob(pattern)
                )

            # A FlyWire conversion is usable only when both generated tables
            # exist.  A neuron table by itself is not enough for pathfinding:
            # the converter also writes the merged connection table.
            neuron_ready = any(
                path
                for pattern in (
                    "*_allneurons_neuron_df.parquet",
                    "*_allneurons_neuron_df.csv",
                )
                for path in dataset_path.glob(pattern)
            )
            connections_ready = any(
                path
                for pattern in (
                    "*_merged_connections.parquet",
                    "*_merged_connections.csv",
                )
                for path in dataset_path.glob(pattern)
            )
            if neuron_ready and connections_ready:
                return True
        return False

    def _get_dataset_path(self, dataset: str) -> Optional[Path]:
        """Get the local path for a dataset."""
        safe_name = dataset_to_folder(dataset)
        return self._datasets_dir / safe_name

    def _find_metadata_file(self, dataset: str) -> Optional[Path]:
        """Find metadata file for a dataset."""
        dataset_path = self._get_dataset_path(dataset)
        if not dataset_path or not dataset_path.exists():
            return None

        safe_name = dataset_to_folder(dataset)
        metadata_file = dataset_path / f"{safe_name}_metadata.json"
        if metadata_file.exists():
            return metadata_file

        for f in dataset_path.glob("*_metadata.json"):
            return f

        return None

    def get_local_datasets(self) -> List[DatasetInfo]:
        """Get information about locally available datasets."""
        datasets = []

        if not self._datasets_dir.exists():
            return datasets

        for folder in self._datasets_dir.iterdir():
            if folder.is_dir() and not folder.name.startswith("."):
                name = folder_to_dataset(folder.name)

                info = DatasetInfo(
                    name=name,
                    source="flywire" if is_flywire_dataset(name) else "neuprint",
                    local_cache=True,
                )

                # Keep the local listing consistent with
                # check_dataset_availability().  In particular, a FlyWire
                # directory containing only the neuron table is not ready for
                # pathfinding until its merged connection table is present.
                info.local_prepared = self._check_local_prepared(name)
                info.available = info.local_prepared

                metadata_file = self._find_metadata_file(name)
                if metadata_file and metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            meta = json.load(f)
                            info.neuron_count = meta.get("neuron_counts", {}).get("total", 0)
                            info.typed_count = meta.get("neuron_counts", {}).get("typed", 0)
                            info.metadata = meta
                    except Exception:
                        pass

                datasets.append(info)

        return datasets

    def refresh_availability(self, datasets: Optional[List[str]] = None) -> Dict[str, DatasetInfo]:
        """
        Refresh availability for all or specific datasets.
        Also fetches FlyWire dataset list from Codex.
        """
        with self._lock:
            self._cache.clear()

        if datasets is None:
            # Fetch FlyWire datasets from Codex
            self.fetch_codex_datasets()
            # Fetch NeuPrint datasets from server
            neuprint_available = self.fetch_neuprint_datasets()
            datasets = neuprint_available + self.FLYWIRE_DATASETS

        results = {}
        for dataset in datasets:
            results[dataset] = self.check_dataset_availability(dataset)

        return results

    def is_cache_fresh(self, max_age_seconds: int = 300) -> bool:
        """Check if the cached availability data is still fresh."""
        if self._last_fetch_time == 0:
            return False
        return (time.time() - self._last_fetch_time) < max_age_seconds


# Global instance
_dataset_service: Optional[DatasetService] = None


def get_dataset_service() -> DatasetService:
    """Get the global DatasetService instance."""
    global _dataset_service
    if _dataset_service is None:
        _dataset_service = DatasetService()
    return _dataset_service
