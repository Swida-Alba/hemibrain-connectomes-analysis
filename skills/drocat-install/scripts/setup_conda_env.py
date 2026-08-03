#!/usr/bin/env python3
"""Create the DROCAT conda environment, handling env-name conflicts.

Behavior:
  - If the preferred env name (default: 'drocat') is free, it is created.
  - If an env with that name ALREADY EXISTS on this machine, this script
    warns the user (on stderr) and instead creates a fresh env whose name
    reflects the DROCAT version: '<preferred>-v<version>' (e.g.
    'drocat-v4.4.3'). If that name is taken too, a free index is appended:
    'drocat-v4.4.3-2', 'drocat-v4.4.3-3', ... The existing env is never
    modified or removed.
  - The chosen name is printed on the LAST stdout line as
    'DROCAT_ENV_NAME=<name>' so callers (agents, shell scripts) can parse it.

Usage:
    python setup_conda_env.py                        # env 'drocat', Python 3.11
    python setup_conda_env.py --name drocat --python 3.11 --version 4.4.3
    python setup_conda_env.py --conda /path/to/conda # explicit conda binary
    python setup_conda_env.py --dry-run              # report the name only

Exit codes: 0 = env ready, 1 = conda not found / creation failed.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_conda(explicit: str = None) -> str:
    """Locate the conda binary."""
    if explicit:
        if Path(explicit).exists():
            return str(Path(explicit).expanduser().resolve())
        raise SystemExit(f"ERROR: conda binary not found at: {explicit}")

    found = shutil.which("conda")
    if found:
        return found

    home = Path.home()
    candidates = [
        home / "miniconda3",
        home / "anaconda3",
        home / "miniforge3",
        home / "mambaforge",
        Path("/opt/miniconda3"),
        Path("/opt/anaconda3"),
        Path("/usr/local/miniconda3"),
        Path("/usr/local/anaconda3"),
        Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / "miniconda3",
        Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / "anaconda3",
    ]
    for base in candidates:
        for rel in ("bin/conda", "Scripts/conda.exe", "condabin/conda"):
            cand = base / rel
            if cand.exists():
                return str(cand)
    raise SystemExit(
        "ERROR: conda not found. Install Miniconda first: "
        "https://docs.conda.io/en/latest/miniconda.html"
    )


def list_env_names(conda: str) -> set:
    """Collect all existing env names (named envs + path-based basenames)."""
    proc = subprocess.run(
        [conda, "env", "list", "--json"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    names = set()
    try:
        data = json.loads(proc.stdout)
        for path in data.get("envs", []):
            base = Path(path).name
            if base:
                names.add(base)
    except (ValueError, KeyError):
        pass
    # Also parse the plain listing to catch the name column directly
    proc = subprocess.run(
        [conda, "env", "list"], capture_output=True, text=True, timeout=120
    )
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts and not parts[0].startswith("/"):
            names.add(parts[0])
    return names


def pick_env_name(preferred: str, version: str, existing: set) -> tuple:
    """Return (chosen_name, conflicted). Never collides with existing envs.

    Order: 'drocat' -> 'drocat-v<version>' -> 'drocat-v<version>-2', -3, ...
    The versioned name keeps parallel installs of different DROCAT releases
    distinguishable on the same machine.
    """
    if preferred not in existing:
        return preferred, False
    versioned = f"{preferred}-v{version}"
    if versioned not in existing:
        return versioned, True
    n = 2
    while f"{versioned}-{n}" in existing:
        n += 1
    return f"{versioned}-{n}", True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="drocat", help="Preferred env name")
    parser.add_argument("--python", default="3.11", help="Python version (3.10-3.11 supported)")
    parser.add_argument(
        "--version",
        default="4.4.3",
        help="DROCAT version used to build the fallback env name (drocat-v<version>)",
    )
    parser.add_argument("--conda", default=None, help="Explicit path to the conda binary")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report the env name that would be used; do not create it",
    )
    args = parser.parse_args()

    conda = find_conda(args.conda)
    existing = list_env_names(conda)
    chosen, conflicted = pick_env_name(args.name, args.version, existing)

    if conflicted:
        print(
            f"WARNING: a conda env named '{args.name}' already exists on this "
            f"machine. Leaving it untouched and creating a new env named "
            f"'{chosen}' instead (name reflects DROCAT v{args.version}).",
            file=sys.stderr,
        )

    if args.dry_run:
        print(f"Would use conda env: {chosen}")
    else:
        cmd = [conda, "create", "-n", chosen, f"python={args.python}", "-y"]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"ERROR: failed to create conda env '{chosen}'.",
                file=sys.stderr,
            )
            return 1

    print(f"Activate with: conda activate {chosen}")
    # Machine-parseable result - must stay the LAST line of stdout.
    print(f"DROCAT_ENV_NAME={chosen}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
