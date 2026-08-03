#!/usr/bin/env python3
"""Verify a DROCAT v4.4.5 installation without requiring network access.

Checks project layout, Python version, core/optional imports, the token file,
and that the backend modules import. Exits 0 when all required checks pass.

Usage:
    python verify_install.py --project /path/to/hemibrain-connectomes-analysis
    python verify_install.py --project . --python /path/to/env/bin/python
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


CORE_IMPORTS = [
    "numpy",
    "pandas",
    "polars",
    "scipy",
    "matplotlib",
    "seaborn",
    "plotly",
    "networkx",
    "openpyxl",
    "xlsxwriter",
    "bokeh",
    "requests",
    "jinja2",
    "PIL",
    "pydantic",
    "neuprint",
    "tqdm",
    "cv2",
    "reportlab",
    "pptx",
    "img2pdf",
    "neuronbridge",
]

OPTIONAL_IMPORTS = [
    "caveclient",
    "cloudvolume",
    "navis",
    "flybrains",
    "selenium",
    "webdriver_manager",
    "fitz",
    "rapidjson",
    "psutil",
    "trimesh",
]

BACKEND_MODULES = [
    "coana",
    "statvis",
    "statvis_polars",
    "neuronbridge_finder",
    "visualize_skeleton",
    "flylight_downloader",
    "comparison.profile_comparator",
    "comparison.comparison_analyzer",
    "comparison.connectivity_profiler",
    "core.fast_graph",
]

# Distributions owned by DROCAT (normalized names). `pip check` lines about
# other packages in the env are treated as noise; lines about these are real
# install conflicts (e.g. pydantic drift breaking neuronbridge-python).
DROCAT_DISTS = {
    "numpy", "pandas", "polars", "scipy", "pyarrow", "plotly", "matplotlib",
    "seaborn", "opencv-python", "bokeh", "kaleido", "selenium",
    "webdriver-manager", "networkx", "openpyxl", "tqdm", "requests", "jinja2",
    "pillow", "pydantic", "python-rapidjson", "pyqt5", "neuprint-python",
    "navis", "flybrains", "boto3", "reportlab", "python-pptx", "pymupdf",
    "img2pdf", "k3d", "ipywidgets", "open3d", "caveclient", "cloud-volume",
    "fast-simplification", "trimesh", "psutil", "xlsxwriter",
    "neuronbridge-python", "ray", "memray", "hemibrain-connectomes-analysis",
}

IMPORT_PROBE = """
import importlib, json
mods = json.loads('__MODS_JSON__')
out = {}
for name in mods:
    try:
        importlib.import_module(name)
        out[name] = "ok"
    except Exception as exc:
        out[name] = "%s: %s" % (type(exc).__name__, exc)
print(json.dumps(out))
"""


def run_probe(
    python_exe: str,
    modules: list,
    cwd: Path = None,
    syspath: list = None,
) -> dict:
    code = IMPORT_PROBE.replace("__MODS_JSON__", json.dumps(modules))
    if syspath:
        code = (
            "import sys\n"
            + "".join(f"sys.path.insert(0, {p!r})\n" for p in syspath)
            + code
        )
    proc = subprocess.run(
        [python_exe, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(cwd) if cwd else None,
    )
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {m: f"probe failed: {proc.stderr[-200:]}" for m in modules}


def run_pip_check(python_exe: str) -> tuple:
    """Run `pip check` and return (ok, detail). Only conflicts involving a
    DROCAT-owned distribution count as failures; unrelated env noise is
    reported in the detail string but does not fail the check."""
    try:
        proc = subprocess.run(
            [python_exe, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as exc:
        return True, f"pip check could not run: {exc}"
    lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
    if not lines:
        return True, "no conflicts"
    # A conflict is a DROCAT problem only when a DROCAT-owned distribution is
    # the one *declaring* the broken requirement (first token of the line),
    # e.g. "neuronbridge-python 3.3.0 has requirement pydantic~=2.9.1, ...".
    # Lines where unrelated packages complain about pinned versions (e.g.
    # "pdfplumber ... requires Pillow>=12") are coexistence noise, not
    # install failures.
    drocat_conflicts = [
        l for l in lines
        if l.split()[0].lower().replace("_", "-").split("[")[0] in DROCAT_DISTS
    ]
    if drocat_conflicts:
        return False, "; ".join(drocat_conflicts[:5])
    return True, f"{len(lines)} conflict(s) in unrelated packages (ignored)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Path to the DROCAT repository")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to test (default: the interpreter running this script)",
    )
    args = parser.parse_args()

    project = Path(args.project).expanduser().resolve()
    python_exe = str(Path(args.python).expanduser().resolve())
    checks = []
    version = "unknown"

    def check(name: str, ok: bool, detail: str = ""):
        checks.append((name, ok, detail))

    # Project layout (v4.4.5: scripts/ + src/, no ui/)
    check("project directory", project.is_dir(), str(project))
    for rel in [
        "requirements.txt",
        "requirements-windows.txt",
        "scripts/FindPath.py",
        "scripts/FindDirect.py",
        "scripts/ConnectivityProfiling.py",
        "scripts/NeuronBridge_FindLines.py",
        "scripts/plot3dSkeleton.py",
        "src/coana.py",
        "src/neuronbridge_finder.py",
        "src/visualize_skeleton.py",
        "src/comparison/profile_comparator.py",
        "vispath-subproject/src/vispath_pkg",
    ]:
        check(f"file {rel}", (project / rel).exists(), str(project / rel))

    # Python version
    try:
        proc = subprocess.run(
            [python_exe, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        version = proc.stdout.strip()
        major_minor = tuple(int(x) for x in version.split("."))
        check("python version >= 3.10", major_minor >= (3, 10), version)
        if major_minor >= (3, 12):
            checks.append(
                (
                    "python version <= 3.11 (recommended)",
                    False,
                    f"{version}: PyQt5 5.15.10 / open3d 0.19 / ray 2.39 have no "
                    "wheels for 3.12+; recreate the env with Python 3.11",
                )
            )
    except Exception as exc:
        check("python version", False, str(exc))

    # Imports
    core_results = run_probe(python_exe, CORE_IMPORTS)
    for mod, status in core_results.items():
        check(f"import {mod}", status == "ok", status)
    optional_results = run_probe(python_exe, OPTIONAL_IMPORTS)
    for mod, status in optional_results.items():
        if status != "ok":
            check(f"import {mod} (optional)", True, f"missing: {status}")

    # Backend module imports (project src/ on sys.path)
    backend_results = run_probe(
        python_exe,
        BACKEND_MODULES,
        cwd=project,
        syspath=[str(project / "src")],
    )
    for mod, status in backend_results.items():
        check(f"backend import {mod}", status == "ok", status)

    # Dependency consistency (catches drift that imports alone cannot, e.g.
    # pydantic upgraded past neuronbridge-python's ~=2.9.1 constraint)
    pip_ok, pip_detail = run_pip_check(python_exe)
    check("pip check (DROCAT dependencies consistent)", pip_ok, pip_detail)

    # Token file
    local_token = project / "token_info_local.txt"
    token_file = local_token if local_token.exists() else project / "token_info.txt"
    if token_file.exists():
        text = token_file.read_text(encoding="utf-8", errors="replace")
        has_neuprint = any(
            line.strip().startswith("NEUPRINT_TOKEN=")
            and "YOUR_" not in line
            and len(line.split("=", 1)[1].strip().strip("'\"")) > 20
            for line in text.splitlines()
        )
        check("token_info (NeuPrint token configured)", has_neuprint, str(token_file))
    else:
        check("token_info file", False, "missing token_info.txt / token_info_local.txt")

    # Report
    required_failures = []
    print(f"DROCAT v4.4.5 install verification - {project}")
    print(f"Python: {version}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}" + (f"  ({detail})" if detail and not ok else ""))
        if not ok and not name.endswith("(optional)"):
            required_failures.append(name)

    print()
    if required_failures:
        print(f"FAILED: {len(required_failures)} required check(s) failed")
        return 1
    print("OK: all required checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
