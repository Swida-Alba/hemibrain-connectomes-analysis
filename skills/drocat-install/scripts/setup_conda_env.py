#!/usr/bin/env python3
"""Create the DROCAT conda environment, handling env-name conflicts.

Conda discovery:
  - Looks for a local conda (PATH, then well-known install locations such as
    ~/miniconda3, ~/anaconda3, ~/miniforge3, /opt/*, C:\\ProgramData\\*).
  - If NO local conda is found, Miniconda is downloaded and installed
    automatically into ~/miniconda3 (use --no-install to disable; dry runs
    never install anything).

Env-name behavior:
  - The preferred env is 'drocat-4.4.5'. A caller passing the legacy
    '--name drocat' spelling is normalized to that versioned name.
  - An existing candidate is reused when it has the requested Python.
  - A candidate with the wrong Python is left untouched and the next suffix
    is checked: 'drocat-4.4.5-2', 'drocat-4.4.5-3', ...
  - The chosen name is printed on the LAST stdout line as
    'DROCAT_ENV_NAME=<name>' so callers (agents, shell scripts) can parse it.

Usage:
    python setup_conda_env.py                        # env 'drocat-4.4.5', Python 3.11
    python setup_conda_env.py --name drocat --python 3.11 --version 4.4.5
    python setup_conda_env.py --conda /path/to/conda # explicit conda binary
    python setup_conda_env.py --no-install           # fail instead of installing Miniconda
    python setup_conda_env.py --dry-run              # report the name only

Exit codes: 0 = env ready, 1 = conda not found / creation failed.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _miniconda_installer_url() -> str:
    """Pick the right Miniconda installer for this OS/architecture."""
    if sys.platform == "darwin":
        arch = "arm64" if platform.machine() in ("arm64", "aarch64") else "x86_64"
        return f"https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-{arch}.sh"
    if sys.platform.startswith("linux"):
        arch = "aarch64" if platform.machine() in ("arm64", "aarch64") else "x86_64"
        return f"https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-{arch}.sh"
    if sys.platform == "win32":
        return "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
    raise SystemExit(f"ERROR: unsupported platform for Miniconda auto-install: {sys.platform}")


def install_miniconda() -> str:
    """Download and silently install Miniconda; return the new conda binary path.

    Used as the fallback when no local conda is found, so the pipeline works
    on a fresh machine without manual steps.
    """
    import tempfile
    import urllib.request

    url = _miniconda_installer_url()
    install_dir = Path.home() / "miniconda3"
    is_windows = sys.platform == "win32"
    suffix = ".exe" if is_windows else ".sh"
    if is_windows and install_dir.exists() and not (install_dir / "Scripts/conda.exe").exists():
        # The Windows installer cannot update an arbitrary partial directory.
        install_dir = Path.home() / "miniconda3-drocat"

    print(f"WARNING: conda not found on this machine.", file=sys.stderr)
    print(f"Downloading Miniconda from: {url}", file=sys.stderr)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        installer_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(url, installer_path)
        print("Installing Miniconda (silent)...", file=sys.stderr)
        if is_windows:
            # /D= must be the LAST argument and must not be quoted
            subprocess.run(
                [str(installer_path), "/InstallationType=JustMe",
                 "/RegisterPython=0", "/S", f"/D={install_dir}"],
                check=True,
            )
        else:
            # The .sh installer REFUSES to install into an existing directory
            # (e.g. a broken/partial earlier install); -u repairs it instead.
            flags = ["-b"]
            if install_dir.exists():
                flags.append("-u")
            subprocess.run(
                ["bash", str(installer_path), *flags, "-p", str(install_dir)],
                check=True,
            )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"ERROR: Miniconda installer failed (exit {exc.returncode}).")
    finally:
        try:
            installer_path.unlink()
        except OSError:
            pass

    conda_bin = install_dir / ("Scripts/conda.exe" if is_windows else "bin/conda")
    if not conda_bin.exists():
        raise SystemExit(f"ERROR: Miniconda install finished but {conda_bin} was not found.")

    # Register conda in the user's shells so future terminals can use it
    try:
        if is_windows:
            subprocess.run([str(conda_bin), "init", "powershell"],
                           capture_output=True, timeout=120)
        else:
            for shell in ("bash", "zsh"):
                subprocess.run([str(conda_bin), "init", shell],
                               capture_output=True, timeout=120)
    except Exception:
        pass  # shell init is best-effort; the env can still be used directly

    print(f"Miniconda installed at: {install_dir}", file=sys.stderr)
    return str(conda_bin)


def find_conda(explicit: str = None, allow_install: bool = True) -> str:
    """Locate the conda binary.

    Discovery order: explicit path -> PATH -> well-known install locations.
    If nothing is found and allow_install is True, Miniconda is downloaded
    and installed automatically (pass --no-install to disable).
    """
    if explicit:
        explicit_path = Path(explicit).expanduser()
        if explicit_path.exists():
            return str(explicit_path.resolve())
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

    if allow_install:
        return install_miniconda()
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


def env_matches_python(conda: str, name: str, python_version: str) -> bool:
    """Return whether an existing named env uses the requested Python."""
    code = (
        "import sys; raise SystemExit(0 if sys.version_info[:2] == "
        f"tuple(map(int, {python_version!r}.split('.'))) else 1)"
    )
    return subprocess.run(
        [conda, "run", "-n", name, "python", "-c", code],
        capture_output=True,
        timeout=120,
    ).returncode == 0


def pick_env_name(
    conda: str, preferred: str, python_version: str, existing: set
) -> tuple:
    """Return (chosen_name, create_required) using installer-compatible order."""
    for index in range(21):
        candidate = preferred if index == 0 else f"{preferred}-{index + 1}"
        if candidate not in existing:
            return candidate, True
        if env_matches_python(conda, candidate, python_version):
            return candidate, False
        print(
            f"WARNING: leaving '{candidate}' untouched because it does not "
            f"use Python {python_version}.",
            file=sys.stderr,
        )
    raise SystemExit(f"ERROR: no usable/free environment name for {preferred}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="drocat-4.4.5", help="Preferred env name")
    parser.add_argument(
        "--python",
        default="3.11",
        choices=("3.10", "3.11"),
        help="Python version (3.10-3.11 supported)",
    )
    parser.add_argument(
        "--version",
        default="4.4.5",
        help="DROCAT version used to build the fallback env name (drocat-<version>)",
    )
    parser.add_argument("--conda", default=None, help="Explicit path to the conda binary")
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Do not auto-install Miniconda when no local conda is found",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report the env name that would be used; do not create it",
    )
    args = parser.parse_args()

    # Never auto-install Miniconda for a dry run
    allow_install = not args.no_install and not args.dry_run
    conda = find_conda(args.conda, allow_install=allow_install)
    existing = list_env_names(conda)
    version = args.version.strip()
    if version.lower().startswith("v"):
        version = version[1:]
    preferred_name = args.name.strip()
    if not version:
        parser.error("--version must not be empty")
    if not preferred_name:
        parser.error("--name must not be empty")
    preferred = f"drocat-{version}" if preferred_name == "drocat" else preferred_name
    chosen, create_required = pick_env_name(
        conda, preferred, args.python, existing
    )

    if args.dry_run:
        print(f"Would use conda env: {chosen}")
    elif create_required:
        cmd = [conda, "create", "-n", chosen, f"python={args.python}", "-y"]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"ERROR: failed to create conda env '{chosen}'.",
                file=sys.stderr,
            )
            return 1
    else:
        print(f"Reusing existing conda env: {chosen}")

    print(f"Activate with: conda activate {chosen}")
    # Machine-parseable result - must stay the LAST line of stdout.
    print(f"DROCAT_ENV_NAME={chosen}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
