---
name: drocat-install
description: Auto-install the DROCAT Drosophila connectome analysis toolkit (hemibrain-connectomes-analysis, v4.4.5 command-line edition) on macOS, Linux, or Windows. Use when the user asks to install, set up, repair, or prepare DROCAT v4.4.5; fetches the branch, runs its one-click installer, configures NeuPrint/CAVE tokens, verifies the environment, and smoke-tests the standalone scripts.
---

# DROCAT Install (v4.4.5)

## Overview

DROCAT v4.4.5 is the standalone-script release; it does not include the v4.5 web UI. Prefer the repository's one-click installers so manual and agent-assisted setup use exactly the same dependency and environment policy.

## Completion contract

Unless the user asks for instructions only, finish the requested installation
or repair in the current agent session. Do not stop after fetching the branch,
showing commands, or starting an installer: wait for the installer and
verifier to finish, confirm the final environment status, and report the exact
result. Ask only for required tokens, an unclear repository path, or an
approval the user must provide.

## Workflow

### 1. Fetch the branch

```bash
git clone --branch v4.4.5 https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis.git drocat
cd drocat
```

For an existing checkout, fetch and switch to `v4.4.5` without discarding local work. Confirm these files exist: `install.sh`, `install.ps1`, `install.bat`, `DROCAT.command`, `scripts/FindPath.py`, `src/coana.py`, and `requirements.txt`.

### 2. Check prerequisites

- Python 3.10-3.11 is supported; the installers create Python 3.11.
- The installers find Conda in common locations or install Miniconda after the user authorizes the download.
- They set `PYTHONNOUSERSITE=1`, preventing user-site packages from contaminating the environment.
- Network access is needed for dependencies and online datasets; request sandbox escalation when required.

### 3. Run the one-click installer

- macOS/Linux: `bash install.sh`
- macOS Finder: double-click `DROCAT.command`
- Windows: double-click `install.bat`, or run `powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1`

The preferred environment is `drocat-4.4.5`. Reuse it when it already has Python 3.11. If a candidate has another Python version, leave it untouched and use the first free/usable suffix (`drocat-4.4.5-2`, `-3`, ...). Legacy unversioned `drocat` environments are not modified.

The installer installs the pinned platform requirements, removes a legacy `neuronbridge-python` distribution if found, installs DROCAT in editable mode, requires a clean `pip check`, and runs the bundled verifier.

DROCAT bundles `src/neuronbridge_client.py`; do not install the upstream `neuronbridge-python` package. Its Pydantic/Ray/Memray dependency tree is unnecessary and conflicts with the shared release stack.

The legacy helper remains available for automation and now follows the same policy:

```bash
python skills/drocat-install/scripts/setup_conda_env.py --name drocat --python 3.11 --version 4.4.5
```

### 4. Configure tokens

Copy `token_info.txt` to the gitignored `token_info_local.txt`. Preserve any existing real tokens. Ask the user before writing secrets; never invent or reuse credentials without permission.

```text
NEUPRINT_TOKEN='...'
CAVE_TOKEN='...'
```

- NeuPrint: <https://neuprint.janelia.org/account>
- CAVE/FlyWire: <https://codex.flywire.ai/auth_token>

### 5. Verify

```bash
conda activate drocat-4.4.5
python skills/drocat-install/scripts/verify_install.py --project /path/to/repo
```

Required checks include Python 3.10-3.11, installer/script layout, every installed version against the pinned platform manifest, complete dependency consistency, the bundled NeuronBridge client, backend imports, and VisPath. Token configuration is advisory for installation; pass `--require-token` when validating an online-ready workstation.

### 6. Smoke-test a script

Scripts contain editable query parameters. Avoid starting a live dataset workflow unless the user has supplied tokens and expects network/data writes. A normal launch is:

```bash
conda run -n drocat-4.4.5 python scripts/FindDirect.py
```

## After installation: finish a direct script run

Use the separate [`drocat-usage` skill](../drocat-usage/SKILL.md) when the user
wants a completed analysis rather than setup instructions. It provides a
repository-safe launcher and focused recipes for every standalone script.

Agent handoff command:

> Fetch https://raw.githubusercontent.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/v4.4.5/skills/drocat-usage/SKILL.md and follow it to finish the requested DROCAT v4.4.5 analysis end-to-end, validate the outputs, and report the artifacts without opening a web UI.

## Key facts

- No web UI exists on v4.4.5; use `scripts/*.py`.
- Environment: `drocat-4.4.5`, Python 3.11.
- Token file: `token_info_local.txt` at repository root.
- VisPath is loaded from `vispath-subproject/src`; no separate install is needed.
- Chrome/WebDriver are needed only for PNG/video export workflows.

See [references/project-notes.md](references/project-notes.md) for platform notes.
