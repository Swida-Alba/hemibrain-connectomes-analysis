---
name: drocat-install
description: Auto-install the DROCAT Drosophila connectome analysis toolkit (hemibrain-connectomes-analysis, v4.4.x script edition - no web UI) on macOS, Linux, or Windows. Use when the user asks to install, set up, repair, or prepare DROCAT on the v4.4.3 branch (e.g., "install DROCAT", "set up the connectome toolkit", "fix my DROCAT install", "fresh machine setup") - fetches the repository from GitHub, creates the conda environment, installs dependencies, configures NeuPrint/CAVE tokens, verifies the installation, and runs the standalone scripts.
---

# DROCAT Install (v4.4.3)

## Overview

Install DROCAT v4.4.3 end-to-end on a fresh machine: fetch the repository from GitHub, create the `drocat` conda environment, install pinned dependencies, configure API tokens, verify the installation, and run the standalone analysis scripts. This branch has **no web UI** - all entry points are `scripts/*.py`.

## Workflow

### 1. Fetch the repository

Pull DROCAT v4.4.3 from GitHub (works in any agent that can run shell commands):

```bash
git clone --branch v4.4.3 https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis.git drocat
cd drocat
```

If a checkout already exists, fetch the branch instead:

```bash
git fetch origin v4.4.3
git checkout v4.4.3
```

- If the user already has the repository locally, reuse it (ask if the path is not obvious) and skip the clone.
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

- Python 3.10-3.11 is supported; **Python 3.11 is recommended**. Do NOT use
  3.9 (matplotlib 3.10.0 requires >=3.10) or 3.12+ (no wheels for PyQt5
  5.15.10 / open3d 0.19 / ray 2.39). The env created below pins 3.11, so an
  existing system Python version does not matter.
- conda is required. The helper finds a local conda automatically (PATH and
  common install locations); if none exists it downloads and installs
  Miniconda into `~/miniconda3` on its own (tell the user this is happening;
  pass `--no-install` to opt out and install manually from
  <https://docs.conda.io/miniconda.html> instead).
- Dependency download and token checks need network access. In a sandboxed environment, request escalation for network commands.

### 3. Install dependencies

Create the conda env with the bundled helper, which handles name conflicts
(an existing env is never modified or removed):

```bash
python skills/drocat-install/scripts/setup_conda_env.py --name drocat --python 3.11 --version 4.4.3
```

- If no env named `drocat` exists, it is created.
- **If an env named `drocat` already exists**, the script warns the user and
  creates a fresh env whose name reflects the DROCAT version:
  `drocat-v4.4.3`; if that is taken too, a free index is appended
  (`drocat-v4.4.3-2`, `drocat-v4.4.3-3`, ...).
- The chosen name is printed on the last line as `DROCAT_ENV_NAME=<name>`.
  Capture it and use it in every `conda activate` / run command below; also
  tell the user explicitly which env name was used.

Then install (substitute the chosen env name):

```bash
conda activate <DROCAT_ENV_NAME>

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
- If `pip install -r requirements.txt` fails on macOS/Linux because memray
  cannot build (no compiler, unsupported platform): comment out the
  `neuronbridge-python` line in `requirements.txt`, install the rest, then
  run `pip install neuronbridge-python --no-deps`. Report that NeuronBridge
  scripts may be limited.
- After installing, run `pip check` and confirm no DROCAT-related conflicts
  are reported (the verifier in step 5 repeats this check automatically).
  Do NOT upgrade individual packages to "fix" a conflict - the pins are
  mutually constrained (especially `pydantic~=2.9.1`, required by
  neuronbridge-python).
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

Run the bundled verifier with the environment Python (substitute the chosen env name):

```bash
conda activate <DROCAT_ENV_NAME>
python skills/drocat-install/scripts/verify_install.py --project /path/to/repo
```

Required checks: project layout, Python version (3.10-3.11), core imports (numpy, pandas, polars, neuprint, neuronbridge, ...), backend module imports (`src/coana.py`, `src/neuronbridge_finder.py`, ...), `pip check` consistency of DROCAT dependencies, token file, vispath subproject. Fix anything reported, then re-run until it passes.

### 6. Smoke-test a script

- Edit the query parameters at the top of a script (e.g., `scripts/FindDirect.py` or `scripts/ConnectivityProfiling.py`) or run with its defaults, then:

  ```bash
  conda activate <DROCAT_ENV_NAME>
  python scripts/FindDirect.py
  ```

- First run: the dataset auto-downloads on first query (requires token + network, can take minutes). FlyWire FAFB/BANC additionally require manually downloaded data files.

## Key facts

- Conda env name: `drocat` (Python 3.11). If `drocat` already existed on the
  machine, the helper created `drocat-v4.4.3` (or `drocat-v4.4.3-N` if that
  was also taken) instead - always check the `DROCAT_ENV_NAME=` line from
  `setup_conda_env.py`.
- No web UI on v4.4.3; entry points are `scripts/*.py`.
- Token files: `token_info.txt` (template) + `token_info_local.txt` (user secrets, takes precedence).
- `vispath-subproject/src` is added to `sys.path` by `scripts/PlotPath.py` and `src/core/fast_graph.py`; no separate pip install is needed.
- Chrome + WebDriver are only needed for PNG/video exports in `scripts/plot3dSkeleton.py`.

## Troubleshooting

See [references/project-notes.md](references/project-notes.md) for platform notes and known issues.
