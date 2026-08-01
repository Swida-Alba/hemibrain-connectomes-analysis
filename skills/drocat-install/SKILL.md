---
name: drocat-install
description: Auto-install the DROCAT Drosophila connectome analysis toolkit (hemibrain-connectomes-analysis, v4.4.x script edition - no web UI) on macOS, Linux, or Windows. Use when the user asks to install, set up, repair, or prepare DROCAT on the v4.4.3 branch (e.g., "install DROCAT", "set up the connectome toolkit", "fix my DROCAT install", "fresh machine setup") - covers conda environment creation, dependency installation, NeuPrint/CAVE token configuration, verification, and running the standalone scripts.
---

# DROCAT Install (v4.4.3)

## Overview

Install DROCAT v4.4.3 end-to-end on a fresh machine: create the `drocat` conda environment, install pinned dependencies, configure API tokens, verify the installation, and run the standalone analysis scripts. This branch has **no web UI** - all entry points are `scripts/*.py`.

## Workflow

### 1. Locate the project

- Find the repository (folder usually named `hemibrain-connectomes-analysis*` or `DROCAT`) checked out on the `v4.4.3` branch. Ask the user if it is not obvious.
- Confirm the required files exist: `scripts/FindPath.py`, `scripts/FindDirect.py`, `src/coana.py`, `requirements.txt`, `requirements-windows.txt`, `vispath-subproject/src/vispath_pkg`.
- Expected layout (v4.4.3 has no `ui/` directory):

  ```
  scripts/          # standalone entry points (10+ scripts)
  src/              # backend modules (coana, statvis, comparison/, ...)
  vispath-subproject/src/vispath_pkg
  requirements.txt  # Linux/macOS
  requirements-windows.txt
  token_info.txt    # token template
  ```

### 2. Check prerequisites

- Python 3.9+ and conda are required (Python 3.11 is recommended).
- If conda is missing, install Miniconda from <https://docs.conda.io/miniconda.html> after user approval.
- Dependency download and token checks need network access. In a sandboxed environment, request escalation for network commands.

### 3. Install dependencies

```bash
conda create -n drocat python=3.11 -y
conda activate drocat

# Linux/macOS:
pip install -r requirements.txt
# Windows:
pip install -r requirements-windows.txt

# NeuronBridge is installed separately per the README:
pip install neuronbridge-python --no-deps

# Optional (scripts work without it via sys.path manipulation):
pip install -e .
```

- If `neuronbridge-python` fails on Windows (memray dependency), record the error, install the remaining dependencies, and continue; report that NeuronBridge scripts may be limited.
- There is no installer script or launcher on v4.4.3 - the conda commands above ARE the install.

### 4. Configure tokens

- Copy `token_info.txt` to `token_info_local.txt` (the local file is gitignored and takes precedence - `src/utils/token_manager.py` reads `token_info.txt` first, then `token_info_local.txt`).
- Ask the user for their tokens; never invent or reuse tokens without permission:
  - `NEUPRINT_TOKEN` from <https://neuprint.janelia.org/account>
  - `CAVE_TOKEN` (FlyWire only) from <https://codex.flywire.ai/auth_token>
- File format:

  ```
  NEUPRINT_TOKEN='...'
  CAVE_TOKEN='...'
  ```

- If `token_info_local.txt` already exists with non-placeholder tokens, keep them.

### 5. Verify

Run the bundled verifier with the environment Python:

```bash
conda activate drocat
python skills/drocat-install/scripts/verify_install.py --project /path/to/repo
```

Required checks: project layout, Python version, core imports (numpy, pandas, polars, neuprint, neuronbridge, ...), backend module imports (`src/coana.py`, `src/neuronbridge_finder.py`, ...), token file, vispath subproject. Fix anything reported, then re-run until it passes.

### 6. Smoke-test a script

- Edit the query parameters at the top of a script (e.g., `scripts/FindDirect.py` or `scripts/ConnectivityProfiling.py`) or run with its defaults, then:

  ```bash
  conda activate drocat
  python scripts/FindDirect.py
  ```

- First run: the dataset auto-downloads on first query (requires token + network, can take minutes). FlyWire FAFB/BANC additionally require manually downloaded data files.

## Key facts

- Conda env name: `drocat` (Python 3.11).
- No web UI on v4.4.3; entry points are `scripts/*.py`.
- Token files: `token_info.txt` (template) + `token_info_local.txt` (user secrets, takes precedence).
- `vispath-subproject/src` is added to `sys.path` by `scripts/PlotPath.py` and `src/core/fast_graph.py`; no separate pip install is needed.
- Chrome + WebDriver are only needed for PNG/video exports in `scripts/plot3dSkeleton.py`.

## Troubleshooting

See [references/project-notes.md](references/project-notes.md) for platform notes and known issues.
