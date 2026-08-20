---
name: drocat-install
description: Auto-install the DROCAT Drosophila connectome analysis toolkit (hemibrain-connectomes-analysis) on macOS, Linux, or Windows. Use when the user asks to install, set up, repair, or prepare DROCAT (e.g., "install DROCAT", "set up the connectome toolkit", "fix my DROCAT install", "fresh machine setup") - fetches the repository from GitHub, creates the conda environment, installs dependencies, configures NeuPrint/CAVE tokens, verifies the installation, and launches the web UI.
---

# DROCAT Install

## Overview

Install DROCAT end-to-end on a fresh machine: fetch the repository from GitHub, create a versioned conda environment, install pinned dependencies, configure API tokens, verify the installation, and launch the web UI.

## Completion contract

Unless the user asks for instructions only, complete the requested installation
or repair in the current agent session. Do not stop after cloning the
repository, showing shell commands, or starting an installer: wait for the
installer and verifier to finish, confirm the final environment/UI status, and
report the exact result. Ask only for required tokens, an unclear repository
path, or an approval that the user must provide.

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
- Confirm the required files exist: `run_DROCAT.command` / `run_DROCAT.bat` (launchers), `archive/install/install.sh` (macOS/Linux), `archive/install/install.ps1` / `archive/install/install.bat` (Windows), `ui/app.py`, `src/coana.py`, `requirements.txt`.

### 2. Check prerequisites

- Python 3.10-3.11 is supported; **Python 3.11 is recommended** (matplotlib
  3.10.0 requires >=3.10, and this release is validated through 3.11). The installers
  create a Python 3.11 environment, so the system Python version does not
  matter.
- If conda is missing, install Miniconda from <https://docs.conda.io/miniconda.html> after user approval (archive/install/install.sh / install.ps1 / install.bat do this automatically).
- Export `PYTHONNOUSERSITE=1` before running pip (the installers do this
  automatically): otherwise pip treats packages in the user's
  `~/.local/lib/python3.11/site-packages` as already installed and never
  installs them into the env.
- Running `scripts/*.py` directly works from any working directory; the
  toolkit resolves its own project root, so tokens in `config.json`
  are always loaded.
- Dependency download and token checks need network access. In a sandboxed environment, request escalation for network commands.

### 3. Run the installer

- **macOS / Linux:** `bash archive/install/install.sh` - creates/reuses a Python 3.11 environment named for the release, installs both pinned requirement files, installs DROCAT in editable mode, runs `pip check`, runs the bundled verifier, and (interactive terminals only) asks for NeuPrint/CAVE tokens.
- **Windows:** run `archive/install/install.ps1` in PowerShell (`powershell -ExecutionPolicy Bypass -File archive/install/install.ps1`) or `archive/install/install.bat`.
- **macOS alternative:** `run_DROCAT.command` (double-click) creates/activates the env and launches the UI.
- The launchers are self-healing: `run_DROCAT.command` / `run_DROCAT.bat` invoke the one-click installer when the environment is missing or inconsistent.
- **Environment naming:** DROCAT uses a versioned env name read from `ui/config.py` (`drocat-4.5.0`). If that name already exists with Python 3.11, the installers reuse it and update dependencies in place (this is the env the launchers prefer, so re-runs actually update the env in use). If it exists with a different Python, never modify or delete it - warn the user and create the next free name instead (`drocat-4.5.0-2`, `drocat-4.5.0-3`, ...). The launchers resolve the same way (first usable env wins). Legacy unversioned `drocat` envs are left untouched.
- **Custom env name (config.json):** a custom conda env name can be pinned per release in the project config `config.json` (created automatically from `config.example.json` by the installers when missing):

  ```json
  { "tokens": { "neuprint": "", "cave": "" }, "envs": { "4.5.0": "my-custom-env" } }
  ```

  The `envs` entry is version-specific: it is only consulted for the CURRENT `APP_VERSION`, so upgrading DROCAT never reuses an older release's custom environment. If the custom env exists with Python 3.11 it is reused; if missing it is created; if it exists with a different Python it is left untouched and the default `drocat-<version>` auto-find (with `-2`, `-3`, ... suffixes) is used instead. The launchers (`run_DROCAT.command` / `run_DROCAT.bat`) resolve the same override first.
- **Auto-fill:** an empty `envs."4.5.0"` entry means "create/use the default versioned env automatically". After the environment is selected (auto-created or found), the installers AND the launchers write the actual env name back into `config.json`, so the entry is filled after the first run and every subsequent script resolves it directly. A custom name is always created by that name and written back unchanged.
- DROCAT v4.5.0 bundles a lightweight NeuronBridge API client. Do **not** install `neuronbridge-python`: its Pydantic 2.9 constraint conflicts with the current NiceGUI stack. The installer removes that legacy distribution when repairing an older versioned environment.
- Treat a failed dependency installation or `pip check` as an installation failure. Do not continue with a partially inconsistent environment.

### 4. Configure tokens

- Tokens live in the gitignored project config `config.json` - the ONLY token file. The legacy `token_info.txt` / `token_info_local.txt` files are deprecated and no longer read (token_info.txt now only documents the migration). The installers create `config.json` from `config.example.json` when it is missing.
- Copy `config.example.json` to `config.json` if it does not exist (the local file is gitignored).
- Ask the user for their tokens; never invent or reuse tokens without permission:
  - `NEUPRINT_TOKEN` from <https://neuprint.janelia.org/account>
  - `CAVE_TOKEN` (FlyWire only) from <https://codex.flywire.ai/auth_token>
- File format:

  ```json
  {
    "tokens": {
      "neuprint": "...",
      "cave": "..."
    },
    "envs": {
      "4.5.0": ""
    }
  }
  ```

- If `config.json` already exists with non-placeholder tokens, keep them. The UI Settings tab reads and writes the same `config.json` (preserving the `envs` section).

### 5. Verify

Run the bundled verifier with the environment Python:

```bash
conda activate drocat-4.5.0
python skills/drocat-install/scripts/verify_install.py --project /path/to/repo
```

Required checks: Python 3.10-3.11, project layout, every installed version against the pinned manifests, dependency metadata (`pip check`), core imports (including the bundled `neuronbridge_client`), and UI import. Token configuration is advisory during installation; use `--require-token` when validating a configured workstation.

### 6. Launch the UI

- `./run_DROCAT.command` (macOS/Linux) or `run_DROCAT.bat` (Windows), or `conda activate drocat-4.5.0 && python ui/app.py`.
- Confirm the server responds: `curl -s http://127.0.0.1:8080/` should contain `DROCAT`.
- First run: datasets auto-download on first query (requires token + network). FlyWire FAFB/BANC additionally require manually downloaded data files (the Settings tab has the guide).

## Key facts

- Conda env name: `drocat-4.5.0` (Python 3.11), with `-2`, `-3`, ... used only when an existing candidate has the wrong Python version. A version-specific custom name can be set in `config.json` under `envs."4.5.0"`; an empty entry is auto-filled with the env actually used after the first install/launch.
- UI: `python ui/app.py` → <http://127.0.0.1:8080>. Override with
  `DROCAT_UI_HOST`, `DROCAT_UI_PORT`, or `DROCAT_UI_SHOW=0` for headless use.
- Token file: `config.json` (gitignored) at the repo root; legacy `token_info*.txt` files are no longer read.
- The UI runner adds `vispath-subproject/src` to `sys.path`; no separate pip install is needed for PlotPath.
- Chrome + WebDriver are only needed for PNG/video exports in the 3D skeleton tool.
- Neuron indexes are persistent "system files" in `neuron_indexes/` (not in
  `cache/`): `male-cns:v1.0`, `flywire_FAFB_v783`, and `flywire_BANC_v888`
  ship committed seed indexes so auto-suggestions and the "See available
  neurons" panel work right after install; other datasets get their index on
  the first pull. Clearing `cache/` never removes the indexes. Refresh the
  bundled seeds with `python src/build_seed_indexes.py` after a dataset
  release changes.

## After installation: direct script mode

If the user wants analysis without opening NiceGUI, hand off to the
[`drocat-usage` skill](../drocat-usage/SKILL.md). It uses the same
`drocat-4.5.0` environment and provides a repository-safe launcher for
`FindDirect.py`, `FindPath.py`, `PlotPath.py`, `plot3dSkeleton.py`, comparison,
NeuronBridge, FlyLight, homolog, profile, and empty-network workflows. Keep
the UI closed for unattended runs and use `showfig=False` until the output has
been validated.

Agent handoff command (continues through completion):

> Fetch https://raw.githubusercontent.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/v4.5.0/skills/drocat-usage/SKILL.md and follow it to install and use the DROCAT v4.5.0 direct-analysis skill for this repository, then finish the requested analysis end-to-end without opening the UI.

## Troubleshooting

See [references/project-notes.md](references/project-notes.md) for platform notes and known issues.
