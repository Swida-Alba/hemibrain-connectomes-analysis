#!/usr/bin/env python3
"""Run a DROCAT example script with repository-safe paths and environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


SKILL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SKILL_ROOT.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a DROCAT v4.4.5 script from its own directory so template "
            "relative output paths resolve consistently."
        )
    )
    parser.add_argument(
        "--script",
        required=True,
        help="Script path relative to the repository (for example scripts/FindPath.py).",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=REPO_ROOT,
        help="DROCAT repository root (default: inferred from this launcher).",
    )
    parser.add_argument(
        "--conda-env",
        help="Run through `conda run` in this environment instead of the current Python.",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=sys.executable,
        help="Python executable when --conda-env is not used.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command and working directory without executing.",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Optional arguments passed to the selected script after `--`.",
    )
    return parser.parse_args()


def resolve_script(repo: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = repo / candidate
    candidate = candidate.resolve()
    repo = repo.resolve()
    try:
        candidate.relative_to(repo)
    except ValueError as exc:
        raise SystemExit(f"Script must be inside repository: {candidate}") from exc
    if candidate.suffix.lower() != ".py":
        raise SystemExit(f"Expected a Python script: {candidate}")
    if not candidate.is_file():
        raise SystemExit(f"Script does not exist: {candidate}")
    return candidate


def main() -> int:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    if not repo.is_dir():
        raise SystemExit(f"Repository directory does not exist: {repo}")
    script = resolve_script(repo, args.script)

    script_args = list(args.script_args)
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]

    if args.conda_env:
        conda = shutil.which("conda")
        if conda is None:
            raise SystemExit("--conda-env requires `conda` to be available on PATH")
        command = [conda, "run", "--no-capture-output", "-n", args.conda_env,
                   "python", str(script), *script_args]
    else:
        command = [args.python_executable, str(script), *script_args]

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

    print(f"[DROCAT] cwd: {script.parent}")
    print(f"[DROCAT] command: {' '.join(map(str, command))}")
    if args.dry_run:
        return 0

    completed = subprocess.run(command, cwd=script.parent, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
