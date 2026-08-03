# DROCAT v4.4.5 Installation

This branch is the standalone-script release (no web UI). It supports Python 3.10-3.11 and includes one-click installation for macOS, Linux, and Windows.

## One-click installation

### macOS

Double-click `DROCAT.command`, or run:

```bash
chmod +x install.sh DROCAT.command
./install.sh
```

### Linux

```bash
chmod +x install.sh
./install.sh
```

### Windows

Double-click `install.bat`, or run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1
```

The installer locates Conda (or installs Miniconda), prepares a Python 3.11 environment, installs the pinned dependencies, installs the checkout in editable mode, and requires both `pip check` and the bundled verifier to pass.

## Environment policy

The preferred name is `drocat-4.4.5`. An existing candidate with Python 3.11 is reused and repaired. A candidate with another Python version is left untouched; setup checks `drocat-4.4.5-2`, `drocat-4.4.5-3`, and so on. The legacy unversioned `drocat` environment is not modified.

Every installer sets `PYTHONNOUSERSITE=1` so user-level packages cannot contaminate the release environment.

## Manual installation

Set `PYTHONNOUSERSITE=1` first (`export PYTHONNOUSERSITE=1` in bash/zsh, or
`$env:PYTHONNOUSERSITE = "1"` in PowerShell), then run:

```bash
conda create -n drocat-4.4.5 python=3.11 -y
conda activate drocat-4.4.5

# macOS/Linux
python -m pip install -r requirements.txt

# Windows (use this instead)
python -m pip install -r requirements-windows.txt

python -m pip install -e . --no-deps
python -m pip check
python skills/drocat-install/scripts/verify_install.py --project .
```

Do not install `neuronbridge-python`. DROCAT v4.4.5 now bundles `src/neuronbridge_client.py`, which implements the NeuronBridge API subset used by the scripts with Requests and Pillow. This removes the unnecessary Ray/Memray tree and its incompatible Pydantic constraint.

## Authentication

Copy the token template to the gitignored local file:

```bash
cp token_info.txt token_info_local.txt
```

Then add the credentials needed by your workflows:

```text
NEUPRINT_TOKEN='your_actual_neuprint_token'
CAVE_TOKEN='your_actual_cave_token'
```

- NeuPrint token: <https://neuprint.janelia.org/account>
- FlyWire/CAVE token: <https://codex.flywire.ai/auth_token>
- NeuronBridge is public and needs no token.

Require a real NeuPrint token during verification with:

```bash
python skills/drocat-install/scripts/verify_install.py --project . --require-token
```

## Running workflows

Edit the parameters in a standalone script, then run it inside the selected environment:

```bash
conda run -n drocat-4.4.5 python scripts/FindDirect.py
```

The first online query may populate `datasets/` and `cache/`. FlyWire FAFB/BANC workflows also require their documented local data files.

## Agent-assisted setup

The [`drocat-install` skill](../skills/drocat-install/SKILL.md) invokes these same one-click installers, guides token configuration, and smoke-tests the script edition.

```bash
git clone --branch v4.4.5 https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis.git drocat
cd drocat
./install.sh
```

## Troubleshooting

- A valid environment reports `No broken requirements found` from `python -m pip check`.
- If Requests warns about Chardet, rerun the installer; v4.4.5 pins `chardet==5.2.0`.
- Chrome/WebDriver is needed only for WebGL PNG/video exports.
- On Linux, native file dialogs may require `tkinter`; scripts can also take explicit paths.
