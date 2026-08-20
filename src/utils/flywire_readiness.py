"""Readiness checks for FlyWire morphology and skeleton workflows.

The connectivity tables for FlyWire datasets and their morphology sources are
different resources.  In particular, BANC has no skeleton release available
through FlyWire Codex, while FAFB can use either the downloaded local skeleton
bundle or the CAVE API.  Keep those rules in one small, dependency-light
module so the UI-facing scripts and the library entry points agree.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional


class FlyWireSkeletonAccessError(RuntimeError):
    """Raised when a FlyWire morphology workflow cannot obtain skeletons."""


def is_banc_dataset(dataset: object) -> bool:
    """Return whether *dataset* identifies a BANC release."""

    return "banc" in str(dataset or "").strip().lower()


def is_fafb_dataset(dataset: object) -> bool:
    """Return whether *dataset* identifies the FlyWire FAFB release."""

    normalized = str(dataset or "").strip().lower()
    return "fafb" in normalized and not is_banc_dataset(normalized)


def dataset_folder(dataset: object) -> str:
    """Map a dataset identifier to the repository folder convention."""

    return str(dataset or "").replace(":", "_").replace(".", "_")


def _configured_cave_token(project_root: Path) -> Optional[str]:
    """Read a CAVE token without ever returning it to a caller-facing log.

    config.json (project config) is checked first; the environment is the
    fallback, matching the project's token manager.  A ``YOUR_*`` placeholder
    is treated as unconfigured.
    """

    config_value = _cave_token_from_config(project_root)
    if config_value:
        return config_value
    environment_value = os.environ.get("CAVE_TOKEN", "").strip()
    return (
        environment_value
        if environment_value and not environment_value.startswith("YOUR_")
        else None
    )


def _cave_token_from_config(project_root: Path) -> Optional[str]:
    """Read the CAVE token from the project config.json, if present."""
    import json

    config_path = project_root / "config.json"
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    section = data.get("tokens") if isinstance(data, dict) else None
    if not isinstance(section, dict):
        return None
    value = section.get("cave")
    if isinstance(value, str):
        value = value.strip()
        if value and not value.startswith("YOUR_"):
            return value
    return None


def _first_existing(paths: list[Path]) -> Optional[Path]:
    """Return the first existing file or populated directory in *paths*."""

    for path in paths:
        if path.is_file():
            return path
        if path.is_dir():
            try:
                if (next(path.rglob("*.pkl"), None) is not None
                        or next(path.rglob("*.pkl.zst"), None) is not None):
                    return path
            except OSError:
                continue
    return None


def local_fafb_skeleton_source(
    dataset: object,
    project_root: Optional[str | Path] = None,
) -> Optional[Path]:
    """Locate a usable local FAFB skeleton/geometry source.

    The converter's ZIP/parquet outputs are the preferred sources.  Existing
    API/skeleton pickle caches also count as local preparation: they allow a
    repeat run to proceed without making another CAVE request.
    """

    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    folder = dataset_folder(dataset)
    dataset_dir = root / "datasets" / folder
    cache_dir = root / "cache" / folder

    source = _first_existing([
        dataset_dir / f"{folder}_skeletons.zip",
        dataset_dir / "sk_lod1_783_healed.zip",
        dataset_dir / "downloads" / "sk_lod1_783_healed.zip",
        dataset_dir / f"{folder}_skeletons.parquet",
        cache_dir / "API_cache" / "skeletons",
        cache_dir / "skeletons",
        cache_dir / "meshes",
    ])
    if source is not None:
        return source

    # Be permissive about a converter-produced skeleton filename while still
    # requiring it to be a file in the selected dataset directory.
    try:
        for path in sorted(dataset_dir.glob("*skeleton*.zip")):
            if path.is_file():
                return path
        for path in sorted(dataset_dir.glob("*skeleton*.parquet")):
            if path.is_file():
                return path
    except OSError:
        pass
    return None


def flywire_skeleton_readiness(
    dataset: object,
    project_root: Optional[str | Path] = None,
) -> dict:
    """Return non-secret readiness information for a morphology workflow."""

    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    local_source = (
        local_fafb_skeleton_source(dataset, root)
        if is_fafb_dataset(dataset)
        else None
    )
    cave_configured = bool(_configured_cave_token(root))
    banc = is_banc_dataset(dataset)
    fafb = is_fafb_dataset(dataset)
    return {
        "dataset": str(dataset or ""),
        "is_banc": banc,
        "is_fafb": fafb,
        "local_skeletons": local_source is not None,
        "local_source": str(local_source) if local_source is not None else None,
        "cave_token": cave_configured,
        "ready": not banc and (not fafb or local_source is not None or cave_configured),
    }


def require_flywire_skeleton_access(
    dataset: object,
    project_root: Optional[str | Path] = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Validate access for a skeleton-based FlyWire workflow.

    BANC is always rejected because no BANC skeleton source is available.
    FAFB is accepted when either local skeleton data or a CAVE token is
    available.  When both are absent, the log includes the local preparation
    and token setup instructions before raising a clear exception.
    """

    status = flywire_skeleton_readiness(dataset, project_root)
    if not status["is_banc"] and not status["is_fafb"]:
        return status

    dataset_name = status["dataset"]
    if status["is_banc"]:
        message = (
            f"Skeleton workflow blocked for {dataset_name}: BANC skeletons are "
            "not available from FlyWire Codex. A CAVE token does not enable "
            "BANC skeleton access."
        )
        log(f"[DROCAT][dataset-guard] BLOCKED: {message}")
        log(
            "Use a non-BANC dataset for 3D skeleton visualization or "
            "morphological similarity. BANC remains available for supported "
            "local connectivity/path analyses."
        )
        raise FlyWireSkeletonAccessError(message)

    if status["local_skeletons"]:
        source = status["local_source"] or "local FAFB skeleton cache"
        if status["cave_token"]:
            log(
                f"[DROCAT][dataset-guard] FAFB skeleton access ready: local "
                f"source found ({source}); CAVE API fallback is configured "
                "and will be attempted for missing or extrusion-affected "
                "skeletons."
            )
        else:
            log(
                f"[DROCAT][dataset-guard] FAFB skeleton access ready: local "
                f"source found ({source}). CAVE_TOKEN is not configured, so "
                "CAVE fallback is disabled."
            )
        return status

    if status["cave_token"]:
        log(
            "[DROCAT][dataset-guard] FAFB local skeletons were not found; "
            "CAVE_TOKEN is configured, so missing skeletons will be fetched "
            "through the CAVE API."
        )
        log(
            "For repeatable/offline runs, download sk_lod1_783_healed.zip "
            "from https://codex.flywire.ai/api/download?dataset=fafb, place "
            f"it in datasets/{dataset_name}/downloads/, then run "
            "python src/FAFB_file_converter.py."
        )
        return status

    message = (
        f"Skeleton workflow blocked for {dataset_name}: no local FAFB "
        "skeleton source was found and CAVE_TOKEN is not configured."
    )
    log(f"[DROCAT][dataset-guard] BLOCKED: {message}")
    log(
        "Prepare local FAFB skeletons: download "
        "sk_lod1_783_healed.zip from "
        "https://codex.flywire.ai/api/download?dataset=fafb, place it in "
        f"datasets/{dataset_name}/downloads/, then run "
        "python src/FAFB_file_converter.py."
    )
    log(
        "Or configure CAVE_TOKEN in config.json (or the environment) "
        "using https://codex.flywire.ai/auth_token and rerun. The converted "
        "local neuron/connection tables are still required for FAFB queries."
    )
    raise FlyWireSkeletonAccessError(message)
