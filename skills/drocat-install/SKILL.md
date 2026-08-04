---
name: drocat-install
description: Auto-install the DROCAT Drosophila connectome analysis toolkit (hemibrain-connectomes-analysis) on macOS, Linux, or Windows. Use when the user asks to install, set up, repair, or prepare DROCAT (e.g., "install DROCAT", "set up the connectome toolkit", "fix my DROCAT install", "fresh machine setup") - fetches the repository from GitHub, creates the conda environment, installs dependencies, configures NeuPrint/CAVE tokens, verifies the installation, and launches the web UI.
---

# DROCAT Install

## Overview

Install DROCAT end-to-end on a fresh machine: fetch the repository from GitHub, create a versioned conda environment, install pinned dependencies, configure API tokens, verify the installation, and launch the web UI.

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

- Python 3.10-3.11 is supported; **Python 3.11 is recommended** (matplotlib
  3.10.0 requires >=3.10, and this release is validated through 3.11). The installers
  create a Python 3.11 environment, so the system Python version does not
  matter.
- If conda is missing, install Miniconda from <https://docs.conda.io/miniconda.html> after user approval (install.sh / install.ps1 / install.bat do this automatically).
- Export `PYTHONNOUSERSITE=1` before running pip (the installers do this
  automatically): otherwise pip treats packages in the user's
  `~/.local/lib/python3.11/site-packages` as already installed and never
  installs them into the env.
- Running `scripts/*.py` directly works from any working directory; the
  toolkit resolves its own project root, so tokens in `token_info_local.txt`
  are always loaded.
- Dependency download and token checks need network access. In a sandboxed environment, request escalation for network commands.

### 3. Run the installer

- **macOS / Linux:** `bash install.sh` - creates/reuses a Python 3.11 environment named for the release, installs both pinned requirement files, installs DROCAT in editable mode, runs `pip check`, and runs the bundled verifier.
- **Windows:** run `install.ps1` in PowerShell (`powershell -ExecutionPolicy Bypass -File install.ps1`) or `install.bat`.
- **macOS alternative:** `DROCAT.command` (double-click) creates/activates the env and launches the UI.
- The launchers are self-healing: `run_ui.sh`, `run_ui.bat`, and `DROCAT.command` invoke the one-click installer when the environment is missing or inconsistent.
- **Environment naming:** DROCAT uses a versioned env name read from `ui/config.py` (`drocat-4.5.0`). If that name already exists with Python 3.11, the installers reuse it and update dependencies in place (this is the env the launchers prefer, so re-runs actually update the env in use). If it exists with a different Python, never modify or delete it - warn the user and create the next free name instead (`drocat-4.5.0-2`, `drocat-4.5.0-3`, ...). The launchers resolve the same way (first usable env wins). Legacy unversioned `drocat` envs are left untouched.
- DROCAT v4.5.0 bundles a lightweight NeuronBridge API client. Do **not** install `neuronbridge-python`: its Pydantic 2.9 constraint conflicts with the current NiceGUI stack. The installer removes that legacy distribution when repairing an older versioned environment.
- Treat a failed dependency installation or `pip check` as an installation failure. Do not continue with a partially inconsistent environment.

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
conda activate drocat-4.5.0
python skills/drocat-install/scripts/verify_install.py --project /path/to/repo
```

Required checks: Python 3.10-3.11, project layout, every installed version against the pinned manifests, dependency metadata (`pip check`), core imports (including the bundled `neuronbridge_client`), and UI import. Token configuration is advisory during installation; use `--require-token` when validating a configured workstation.

### 6. Launch the UI

- `./run_ui.sh` (macOS/Linux) or `run_ui.bat` (Windows), or `conda activate drocat-4.5.0 && python ui/app.py`.
- Confirm the server responds: `curl -s http://127.0.0.1:8080/` should contain `DROCAT`.
- First run: datasets auto-download on first query (requires token + network). FlyWire FAFB/BANC additionally require manually downloaded data files (the Settings tab has the guide).

## Key facts

- Conda env name: `drocat-4.5.0` (Python 3.11), with `-2`, `-3`, ... used only when an existing candidate has the wrong Python version.
- UI: `python ui/app.py` → <http://127.0.0.1:8080>. Override with
  `DROCAT_UI_HOST`, `DROCAT_UI_PORT`, or `DROCAT_UI_SHOW=0` for headless use.
- Token file: `token_info_local.txt` at the repo root.
- The UI runner adds `vispath-subproject/src` to `sys.path`; no separate pip install is needed for PlotPath.
- Chrome + WebDriver are only needed for PNG/video exports in the 3D skeleton tool.

## After installation: direct script mode

If the user wants analysis without opening NiceGUI, hand off to the
[`drocat-usage` skill](../drocat-usage/SKILL.md). It uses the same
`drocat-4.5.0` environment and provides a repository-safe launcher for
`FindDirect.py`, `FindPath.py`, `PlotPath.py`, `plot3dSkeleton.py`, comparison,
NeuronBridge, FlyLight, homolog, profile, and empty-network workflows. Keep
the UI closed for unattended runs and use `showfig=False` until the output has
been validated.

## Troubleshooting

See [references/project-notes.md](references/project-notes.md) for platform notes and known issues.
