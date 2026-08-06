# Installation Guide

DROCAT v4.5.0 needs Python 3.10-3.11 and a conda environment. The one-click
installers handle everything: conda discovery (installing Miniconda if
missing), a fresh `drocat-4.5.0` environment (Python 3.11), pinned
dependencies, `pip check`, and a bundled verifier.

## 1. One-click install (recommended)

| Platform | Install + Launch |
| --- | --- |
| macOS | Double-click `run_DROCAT.command` (installs on first run) |
| Linux | `./run_DROCAT.command` |
| Windows | Double-click `run_DROCAT.bat` (installs on first run) |

Standalone installers (usually only needed for repairs or scripting): `bash archive/install/install.sh` (macOS/Linux), `powershell -ExecutionPolicy Bypass -File archive/install/install.ps1` (Windows).

The UI opens at <http://127.0.0.1:8080>.

The launchers are self-healing: if the environment is missing or inconsistent
(imports fail or `pip check` reports conflicts), they run the installer
automatically before starting the UI.

## 2. Environment policy

- Preferred env: **`drocat-4.5.0`** (Python 3.11). Re-running the installer
  updates it in place.
- If that name is taken by a non-3.11 env, it is left untouched and a suffix
  is used: `drocat-4.5.0-2`, `drocat-4.5.0-3`, ... Launchers resolve the same
  way. Legacy `drocat` envs are never modified.
- All installers/launchers set `PYTHONNOUSERSITE=1`, so packages from a
  user-level Python cannot contaminate the environment.
- Do **not** install `neuronbridge-python` — DROCAT bundles its own client
  (`src/neuronbridge_client.py`) to avoid that package's incompatible
  Pydantic pin and heavy dependency tree.

## 3. Manual install

```bash
conda create -n drocat-4.5.0 python=3.11 -y
conda activate drocat-4.5.0
export PYTHONNOUSERSITE=1                      # PowerShell: $env:PYTHONNOUSERSITE="1"

pip install -r requirements.txt -r ui/requirements.txt            # Windows: requirements-windows.txt
pip install -e . --no-deps
python -m pip check
python skills/drocat-install/scripts/verify_install.py --project .
```

## 4. Authentication

NeuPrint datasets need a **NeuPrint token** (required). The **CAVE token is
optional** — it is only needed for FlyWire FAFB *online* fetching; local
converted FlyWire tables work without it. Two equivalent ways — both
read/write the same gitignored `token_info_local.txt`:

1. **UI Settings tab** — paste the tokens there after launching; the tab
   reminds you when tokens are missing.
2. **Token file** — copy the template and edit it:

```bash
cp token_info.txt token_info_local.txt        # gitignored, takes precedence
```

```text
NEUPRINT_TOKEN='your_actual_neuprint_token'
CAVE_TOKEN='your_actual_cave_token'           # optional
```

- NeuPrint token: <https://neuprint.janelia.org/account>
- CAVE token: <https://codex.flywire.ai/auth_token>
- FlyWire FAFB/BANC also need their local data files in
  `datasets/<dataset>/downloads/` — see the Settings tab guide.
- Strict verification for a configured workstation:
  `python skills/drocat-install/scripts/verify_install.py --project . --require-token`

## 5. Agent-assisted install

In any coding agent, ask:

> Install DROCAT on this machine, verify it works, and report the final environment and UI status.

The agent follows the bundled
[`drocat-install` skill](../skills/drocat-install/SKILL.md) — or, without a
local checkout, paste:

> Fetch https://raw.githubusercontent.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/v4.5.0/skills/drocat-install/SKILL.md and follow it to finish installing, verifying, and launching DROCAT on this machine.

To fetch the release first:

```bash
git clone --branch v4.5.0 https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis.git drocat
cd drocat
./run_DROCAT.command
```

For agent-driven analysis *without* the UI, use the separate
[`drocat-usage` skill](../skills/drocat-usage/SKILL.md).

## 6. Troubleshooting

| Problem | Fix |
| --- | --- |
| `pip check` fails / dependency conflict | Re-run the one-click installer; it repairs the env. The release pins `chardet==5.2.0` (Requests/CloudVolume compatibility). |
| Port 8080 busy | Running `run_DROCAT.command` / `run_DROCAT.bat` interactively now asks what to do: **[1]** start on a new port, **[2]** kill the existing DROCAT process and restart on the same port (non-DROCAT processes are never auto-killed), **[3]** cancel. Manual override: `DROCAT_UI_PORT=8081 ./run_DROCAT.command` (Windows: `set DROCAT_UI_PORT=8081 && run_DROCAT.bat`). |
| PNG/video export fails | Install Google Chrome (WebGL/ChromeDriver). Kaleido is the fallback. |
| Native folder picker missing | Type the path directly. Linux may need `tkinter` / `xdg-utils`. |

## 7. Standalone VisPath

The VisPath subproject (`vispath-subproject/`) reuses the same environment.
The UI loads it from `vispath-subproject/src` directly — no separate install
needed. To use it as a library:

```bash
cd vispath-subproject && pip install -e .
```
