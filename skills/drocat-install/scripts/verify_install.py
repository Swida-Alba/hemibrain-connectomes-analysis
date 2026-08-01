#!/usr/bin/env python3
"""Verify a DROCAT installation without requiring network access.

Checks project layout, Python version, core/optional imports, the token file,
and that the UI package imports. Exits 0 when all required checks pass.

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
    "nicegui",
    "tqdm",
    "cv2",
    "reportlab",
    "pptx",
    "img2pdf",
]

OPTIONAL_IMPORTS = [
    "neuronbridge",
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


def run_probe(python_exe: str, modules: list) -> dict:
    code = IMPORT_PROBE.replace("__MODS_JSON__", json.dumps(modules))
    proc = subprocess.run(
        [python_exe, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
    )
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {m: f"probe failed: {proc.stderr[-200:]}" for m in modules}


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

    # Project layout
    check("project directory", project.is_dir(), str(project))
    for rel in [
        "install.sh",
        "install.ps1",
        "requirements.txt",
        "ui/requirements.txt",
        "ui/app.py",
        "src/coana.py",
        "src/neuronbridge_finder.py",
        "src/visualize_skeleton.py",
        "vispath-subproject/src/vispath_pkg",
    ]:
        check(f"file {rel}", (project / rel).exists(), str(project / rel))

    # Python version
    version = "unknown"
    try:
        proc = subprocess.run(
            [python_exe, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        version = proc.stdout.strip()
        major_minor = tuple(int(x) for x in version.split("."))
        check("python version >= 3.9", major_minor >= (3, 9), version)
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

    # UI import (project must be on sys.path)
    ui_probe = (
        "import sys; sys.path.insert(0, '.'); "
        "import ui.app; print('ui-ok')"
    )
    try:
        proc = subprocess.run(
            [python_exe, "-c", ui_probe],
            cwd=str(project),
            capture_output=True,
            text=True,
            timeout=300,
        )
        check("UI imports (ui.app)", "ui-ok" in proc.stdout, proc.stderr[-200:])
    except Exception as exc:
        check("UI imports (ui.app)", False, str(exc))

    # Report
    required_failures = []
    print(f"DROCAT install verification - {project}")
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
