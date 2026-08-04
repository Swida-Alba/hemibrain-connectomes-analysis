# Installation Guide

DROCAT v4.5.0 supports Python 3.10-3.11 and uses Python 3.11 for its one-click installers. A versioned conda environment keeps this release isolated from older DROCAT installations.

## One-click installation (recommended)

### macOS

Double-click `DROCAT.command` in Finder. It creates or repairs the environment when needed and launches the web UI.

From Terminal, the equivalent commands are:

```bash
chmod +x install.sh run_ui.sh DROCAT.command
./install.sh
./run_ui.sh
```

### Linux

```bash
chmod +x install.sh run_ui.sh
./install.sh
./run_ui.sh
```

### Windows

Double-click `install.bat`, or run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1
```

Launch with `run_ui.bat`.

The installers:

1. Locate Conda or install Miniconda.
2. Select a Python 3.11 environment for this release.
3. Install the pinned platform and UI requirements.
4. Remove the legacy upstream `neuronbridge-python` distribution if it is present.
5. Install this checkout in editable mode, run `pip check`, and run the bundled verifier.

The UI opens at <http://127.0.0.1:8080>.

## Environment policy

The preferred environment is `drocat-4.5.0`. Re-running the installer updates it in place when it uses Python 3.11. If that name is occupied by an environment with the wrong Python version, it is left untouched and the first usable/free suffix is selected: `drocat-4.5.0-2`, `drocat-4.5.0-3`, and so on.

Launchers use the same selection order. Legacy environments such as `drocat` are not modified.

`PYTHONNOUSERSITE=1` is set by all installers and launchers so packages from a user-level Python installation cannot silently contaminate the conda environment.

## Manual installation

Set `PYTHONNOUSERSITE=1` first (`export PYTHONNOUSERSITE=1` in bash/zsh, or
`$env:PYTHONNOUSERSITE = "1"` in PowerShell), then run:

```bash
conda create -n drocat-4.5.0 python=3.11 -y
conda activate drocat-4.5.0

# macOS/Linux
python -m pip install -r requirements.txt -r ui/requirements.txt

# Windows (use this instead of requirements.txt)
python -m pip install -r requirements-windows.txt -r ui/requirements.txt

python -m pip install -e . --no-deps
python -m pip check
python skills/drocat-install/scripts/verify_install.py --project .
```

Do not install `neuronbridge-python`. DROCAT includes `src/neuronbridge_client.py`, which implements the API calls used by the toolkit with Requests and Pillow. This avoids the upstream package's incompatible Pydantic 2.9 pin and its Ray/Memray dependency tree while keeping NeuronBridge features available on every supported platform.

The web UI is repository-based because it uses the checkout's local datasets, caches, guides, and VisPath source. Launch it with the provided scripts from the checkout; the Python wheel contains the reusable analysis modules but intentionally does not install a misleading global `drocat-ui` entry point.

## Configure authentication

NeuronBridge is public and does not require a token. NeuPrint and FlyWire do.

The recommended approach is the web UI's **Settings** tab. Alternatively:

```bash
cp token_info.txt token_info_local.txt
```

Then edit the gitignored `token_info_local.txt`:

```text
NEUPRINT_TOKEN='your_actual_neuprint_token'
CAVE_TOKEN='your_actual_cave_token'
```

- Get a NeuPrint token at <https://neuprint.janelia.org/account>.
- Get a CAVE token at <https://codex.flywire.ai/auth_token>.
- FlyWire FAFB/BANC workflows also need their documented local data files.

To require a configured NeuPrint token during verification:

```bash
python skills/drocat-install/scripts/verify_install.py --project . --require-token
```

## Agent-assisted installation (Codex)

The repository includes [`skills/drocat-install/SKILL.md`](../skills/drocat-install/SKILL.md). In Codex, ask:

> Install DROCAT on this machine and verify it works.

The skill uses the same one-click installers; it additionally guides token setup and verifies that the server responds.

For direct script execution after installation, use the separate
[`drocat-usage` skill](../skills/drocat-usage/SKILL.md). It runs pathfinding,
comparison, NeuronBridge, PlotPath, and 3D scripts without starting the UI.
Beginners can follow the [agent setup guide](AGENT_SETUP.md), including the
official low-cost DeepSeek/Codex configuration.

To fetch this release first:

```bash
git clone --branch v4.5.0 https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis.git drocat
cd drocat
./install.sh
```

## Standalone VisPath

The VisPath subproject uses the same Python and scientific-package window as DROCAT:

```bash
cd vispath-subproject
python -m pip install -e .
```

The main UI loads `vispath-subproject/src` directly; a separate VisPath install is not required for normal DROCAT use.

## Troubleshooting

### Dependency conflict or Requests warning

Rerun the appropriate one-click installer. A valid environment must finish with:

```bash
python -m pip check
```

and report `No broken requirements found.` The release pins `chardet==5.2.0` to remain compatible with Requests and CloudVolume.

### Port 8080 is busy

Stop the process using the port, or launch on another one, for example:

```bash
DROCAT_UI_PORT=8081 ./run_ui.sh
```

### Static image or video export fails

Install Google Chrome for WebGL/ChromeDriver exports. The pinned Kaleido fallback is retained for platforms where it is available.

### Native folder picker is unavailable

Type the path directly in the UI. Linux systems may also need `tkinter` or `xdg-utils` for native dialogs and opening folders.
