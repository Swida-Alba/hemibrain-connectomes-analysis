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
- **Custom env name (version-specific):** put the env name under the matching
  release key in the project configs (`envs` section) - the gitignored
  `config_local.json` wins per key, the committed `config.json` is the
  fallback. The entry is only consulted for *that* release, so upgrading
  DROCAT never reuses an older release's custom environment. An empty value
  means default auto-find:

  ```json
  {
    "tokens": { "neuprint": "", "cave": "" },
    "envs": { "4.5.0": "my-custom-env" }
  }
  ```

  If the custom env exists with Python 3.11 it is reused; if it does not
  exist it is created; if it exists with a different Python the installers
  abort with a clear error (never silently switching envs or rewriting the
  config).
- **Auto-fill:** an empty `envs."4.5.0"` entry means "create/use the default
  versioned env automatically". The installers and launchers write the
  environment they actually used back into `config_local.json`, so after the
  first install the entry holds the concrete env name (e.g.
  `drocat-4.5.0`) and every script resolves it directly. The committed
  `config.json` is never rewritten.
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
converted FlyWire tables work without it. Tokens live in the project configs:
the gitignored `config_local.json` (override, wins per key) and the committed
clean `config.json` (fallback). `token_info.txt` / `token_info_local.txt`
were removed and are never read. Two equivalent ways:

1. **UI Settings tab** — paste the tokens there after launching; the tab
   reminds you when tokens are missing. Saving writes `config_local.json`
   and keeps its `envs` section intact (the committed `config.json` stays
   clean).
2. **Config file** — copy the template to the local override and edit it:

```bash
cp config.example.json config_local.json     # gitignored override
```

```json
{
  "tokens": {
    "neuprint": "your_actual_neuprint_token",
    "cave": "your_actual_cave_token"
  },
  "envs": {
    "4.5.0": ""
  }
}
```

An empty token value means "not configured"; the environment variables
(`NEUPRINT_TOKEN`, `CAVE_TOKEN`) remain a fallback for script mode.
`config_local.json` (and `config.json` if it is missing from a checkout) is
created automatically from `config.example.json` by the one-click installers.

- NeuPrint token: <https://neuprint.janelia.org/account>
- CAVE token: <https://codex.flywire.ai/auth_token>
- FlyWire FAFB/BANC also need their local data files in
  `datasets/<dataset>/downloads/` — see the Settings tab guide.
- Strict verification for a configured workstation:
  `python skills/drocat-install/scripts/verify_install.py --project . --require-token`

## 5. Agent-assisted install & agent setup

### 5.1 Agent-assisted install

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

### 5.2 Recommended low-cost agent: Codex + DeepSeek V4 Flash

DeepSeek documents a Codex integration through its Responses API. At the time
of this release, `deepseek-v4-flash` is the model documented as supporting
Codex; use it for routine script execution, result inspection, and small
patches. Reserve high/max reasoning or a stronger model for a difficult backend
change. Pricing and availability can change, so check the platform before
adding funds.

Official references:

- [DeepSeek Platform](https://platform.deepseek.com/)
- [Responses API guide](https://api-docs.deepseek.com/guides/responses_api/)
- [Codex integration guide](https://api-docs.deepseek.com/quick_start/agent_integrations/codex/)

#### Configure Codex

1. Install Codex CLI or launch the Codex desktop/VS Code client once so that
   `~/.codex` exists.
2. Create a DeepSeek API key on the platform. Treat it like a password and do
   not place it in this repository.
3. Run DeepSeek's official setup script:

   macOS/Linux:

   ```bash
   bash <(curl -fsSL https://cdn.deepseek.com/api-docs/codex-deepseek-setup-en.sh)
   ```

   Windows PowerShell:

   ```powershell
   irm https://cdn.deepseek.com/api-docs/codex-deepseek-setup-en.ps1 | iex
   ```

   Review remote setup scripts according to your security policy. The official
   script backs up `~/.codex/config.toml` under `~/.codex/backup-deepseek/`,
   writes the model catalog, preserves compatible project/MCP settings, and
   validates the configuration before writing it.
4. Choose `deepseek-v4-flash` and restart the client if it is not listed.
   Keep the generated configuration files at user level; do not copy a full
   `models.json` into the DROCAT checkout.

### 5.3 Install/use the direct-analysis skill through an agent

Do not manually copy the skill. Open the repository in Codex (or another
tool-enabled agent) and paste this command:

> Fetch https://raw.githubusercontent.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/v4.5.0/skills/drocat-usage/SKILL.md and follow it to install and use the DROCAT v4.5.0 direct-analysis skill for this repository, then finish the requested analysis end-to-end without opening the UI.

The agent can use the checked-in `skills/drocat-usage/SKILL.md` when local skill
discovery is enabled; the raw URL is the fallback for a new agent session.

### 5.4 First DROCAT agent request

Open the repository in the agent and paste a focused request:

```text
Use the DROCAT v4.5.0 direct-analysis skill. Run a cached FindPath analysis
from aMe12 to PPL101 in male-cns:v0.9, with max_interlayer=2, CSV output, and
save everything under local_data/agent_runs/aMe12_to_PPL101. Use showfig=False,
inspect the generated files, summarize row counts and warnings, and finish by
reporting the validated artifacts. Do not stop at a plan.
```

For a code repair:

```text
Use the DROCAT v4.5.0 direct-analysis skill. Reproduce this direct-script
failure with the smallest query, inspect only the relevant script and backend
signature, patch the call, compile it, and run the focused regression test.
Finish by reporting the patch and test result. Do not change tokens or
dependencies.
```

### 5.5 Running scripts directly

The skill's launcher keeps the script's relative paths correct and adds the
repository modules to `PYTHONPATH`:

```bash
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 \
  --script scripts/FindPath.py \
  --dry-run
```

Remove `--dry-run` to execute. The main direct entry points are:

| Goal | Script |
| --- | --- |
| Direct one-hop edges | `scripts/FindDirect.py` |
| Multi-hop pathways | `scripts/FindPath.py` |
| Path/Sankey/network HTML | `scripts/PlotPath.py` |
| 3D morphology | `scripts/plot3dSkeleton.py` |
| Cross-dataset comparison | `scripts/InterDatasetComparator.py` |
| Homolog finding | `scripts/FindHomologs.py` |
| Connectivity profiles | `scripts/ConnectivityProfiling.py` |
| EM → LM lines | `scripts/NeuronBridge_FindLines.py` |
| LM → EM neurons | `scripts/NeuronBridge_FindNeuron.py` |
| Co-labeling | `scripts/NeuronBridge_Colabel.py` |

Start with small cached queries, `showfig=False`, CSV output, and a unique
output folder. Inspect CSV schemas and HTML existence before increasing hop
counts, enabling bodyId-level work, or exporting images/video.

### 5.6 Safety checklist

- Never paste `NEUPRINT_TOKEN`, `CAVE_TOKEN`, or the contents of
  `config.json` into a prompt, patch, log, or report.
- Ask before remote queries that may download large datasets or spend API/data
  budget.
- Keep `use_cache=True`; use `cache_only=True` only when cache coverage is
  known.
- Do not silently change dataset, thresholds, pathfinding algorithm, or mesh
  transforms to make a run pass.
- Preserve the example scripts; make changes in a copied run script or an
  explicitly requested source file.
- Keep outputs under `local_data/agent_runs/` or another user-approved folder.

For the complete operation catalog and recipes, see
[`skills/drocat-usage/SKILL.md`](../skills/drocat-usage/SKILL.md).

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
