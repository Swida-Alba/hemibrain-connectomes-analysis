---
name: drocat-install
description: Auto-install the DROCAT Drosophila connectome analysis toolkit (hemibrain-connectomes-analysis) on macOS, Linux, or Windows. Use when the user asks to install, set up, repair, or prepare DROCAT (e.g., "install DROCAT", "set up the connectome toolkit", "fix my DROCAT install", "fresh machine setup") - fetches the repository from GitHub, creates the conda environment, installs dependencies, configures NeuPrint/CAVE tokens, verifies the installation, and launches the web UI.
---

# DROCAT Install

## Overview

Install DROCAT end-to-end on a fresh machine: fetch the repository from GitHub, create the `drocat` conda environment, install pinned dependencies, configure API tokens, verify the installation, and launch the web UI.

## Workflow

### 1. Fetch the repository

Pull DROCAT v4.5.0 from GitHub (works in any agent that can run shell commands):

```bash
git clone --branch v4.5.0 https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis.git drocat
cd drocat
```

If a checkout already exists, fetch the branch instead:

```bash
git fetch origin v4.5.0
git checkout v4.5.0
```

- If the user already has the repository locally, reuse it (ask if the path is not obvious) and skip the clone.
- Confirm the required files exist: `install.sh` (macOS/Linux), `install.ps1` / `install.bat` (Windows), `ui/app.py`, `src/coana.py`, `requirements.txt`.

### 2. Check prerequisites

- Python 3.9+ and conda are required (Python 3.11 is recommended).
- If conda is missing, install Miniconda from <https://docs.conda.io/miniconda.html> after user approval.
- Dependency download and token checks need network access. In a sandboxed environment, request escalation for network commands.

### 3. Run the installer

- **macOS / Linux:** `bash install.sh` - creates conda env `drocat` (Python 3.11), installs `requirements.txt`, `ui/requirements.txt`, and runs `pip install -e .`.
- **Windows:** run `install.ps1` in PowerShell (`powershell -ExecutionPolicy Bypass -File install.ps1`) or `install.bat`.
- **macOS alternative:** `DROCAT.command` (double-click) creates/activates the env and launches the UI.
- The launchers are self-healing: `run_ui.sh` / `DROCAT.command` create the env if it is missing.
- If a single dependency fails (e.g., `neuronbridge-python` on Windows, PyQt5 wheels), record the error, install the remaining dependencies, and continue; report the limitation to the user.

### 4. Configure tokens

- Copy `token_info.txt` to `token_info_local.txt` (the local file is gitignored and takes precedence).
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

Required checks: Python version, core imports (numpy, pandas, polars, nicegui, neuprint, ...), backend module imports, token file, UI import. Fix anything reported, then re-run until it passes.

### 6. Launch the UI

- `./run_ui.sh` (macOS/Linux) or `run_ui.bat` (Windows), or `conda activate drocat && python ui/app.py`.
- Confirm the server responds: `curl -s http://127.0.0.1:8080/` should contain `DROCAT`.
- First run: datasets auto-download on first query (requires token + network). FlyWire FAFB/BANC additionally require manually downloaded data files (the Settings tab has the guide).

## Key facts

- Conda env name: `drocat` (Python 3.11).
- UI: `python ui/app.py` → <http://127.0.0.1:8080>.
- Token file: `token_info_local.txt` at the repo root.
- The UI runner adds `vispath-subproject/src` to `sys.path`; no separate pip install is needed for PlotPath.
- Chrome + WebDriver are only needed for PNG/video exports in the 3D skeleton tool.

## Troubleshooting

See [references/project-notes.md](references/project-notes.md) for platform notes and known issues.
