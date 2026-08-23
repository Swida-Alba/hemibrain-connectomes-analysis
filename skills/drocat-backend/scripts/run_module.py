#!/usr/bin/env python3
"""Run a DROCAT backend module script with repository-safe paths and environment.

Layer-2 helper for the drocat-backend skill. It resolves the repository root
robustly (by finding ``src/coana.py``), sets ``PYTHONPATH`` for ``src/``, the repo
root, and ``vispath-subproject/src``, and runs the selected script either in the
current Python or through ``conda run``.

Usage:
    python run_module.py --script archive/scripts_local/agent_Module_2026-08-23.py
    python run_module.py --conda-env drocat-4.5.0 --script my_script.py --dry-run
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` until a directory containing src/coana.py is found."""
    for parent in reversed([*start.parents, start]):
        if (parent / "src" / "coana.py").is_file():
            return parent
    raise SystemExit("Could not locate the DROCAT repository root (src/coana.py)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--script", required=True,
        help="Script path relative to the repository (or absolute).",
    )
    parser.add_argument(
        "--repo", type=Path, help="Explicit repository root (default: auto-detected).",
    )
    parser.add_argument(
        "--conda-env", help="Run through `conda run -n` in this environment.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the resolved command and working directory without executing.",
    )
    parser.add_argument(
        "script_args", nargs=argparse.REMAINDER,
        help="Optional arguments passed to the selected script after `--`.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = (args.repo.expanduser().resolve() if args.repo else find_repo_root(Path.cwd()))

    candidate = Path(args.script).expanduser()
    if not candidate.is_absolute():
        candidate = repo / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(repo)
    except ValueError as exc:
        raise SystemExit(f"Script must be inside repository: {candidate}") from exc
    if candidate.suffix.lower() != ".py":
        raise SystemExit(f"Expected a Python script: {candidate}")
    if not candidate.is_file():
        raise SystemExit(f"Script does not exist: {candidate}")

    script_args = list(args.script_args)
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]

    if args.conda_env:
        conda = shutil.which("conda")
        if conda is None:
            raise SystemExit("--conda-env requires `conda` to be available on PATH")
        command = [conda, "run", "--no-capture-output", "-n", args.conda_env,
                   "python", str(candidate), *script_args]
    else:
        command = [sys.executable, str(candidate), *script_args]

    src_paths = [
        repo / "src",
        repo / "vispath-subproject" / "src",
        repo,
    ]
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    existing_pythonpath = env.get("PYTHONPATH")
    pythonpath = os.pathsep.join(str(path) for path in src_paths)
    if existing_pythonpath:
        pythonpath = os.pathsep.join([pythonpath, existing_pythonpath])
    env["PYTHONPATH"] = pythonpath

    print(f"[DROCAT] cwd: {candidate.parent}")
    print(f"[DROCAT] command: {' '.join(map(str, command))}")
    if args.dry_run:
        return 0

    completed = subprocess.run(command, cwd=candidate.parent, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
