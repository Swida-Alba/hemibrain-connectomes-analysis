# Agent setup for beginners

DROCAT v4.5.0 can be used through a coding agent without opening the NiceGUI
web interface. An agent can run the prepared scripts, inspect result files,
make a focused patch, and explain what happened. This guide assumes no prior
agent experience.

## What an agent does

An agent is a program that combines your request with tools such as a terminal,
file editor, and (when enabled) web search. For DROCAT, give the agent the
repository as its project and ask it to use the repository's
[`drocat-usage`](../skills/drocat-usage/SKILL.md) skill. It should work inside
the checkout and selected output directories, ask before changing credentials
or deleting data, and show you the commands and artifacts it produced.

You do not need to paste the whole repository into a chat. The skill tells the
agent which small script and backend signature to inspect for each task.

## Install DROCAT first

Use the [installation guide](INSTALLATION.md) or ask an agent:

> Install DROCAT v4.5.0 on this machine, verify `pip check`, and leave the UI closed. I will run scripts directly.

The direct-analysis workflow expects the `drocat-4.5.0` Python 3.11
environment and a configured `token_info_local.txt` when NeuPrint or FlyWire
queries require credentials. NeuronBridge itself does not require a token.

## Recommended low-cost agent: Codex + DeepSeek V4 Flash

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

### Configure Codex

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

### Install the DROCAT direct-analysis skill

From the repository root, make the skill available to future agent sessions:

```bash
mkdir -p ~/.codex/skills/drocat-usage
cp -R skills/drocat-usage/. ~/.codex/skills/drocat-usage/
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force "$HOME\.codex\skills\drocat-usage" | Out-Null
Copy-Item -Recurse -Force "skills\drocat-usage\*" "$HOME\.codex\skills\drocat-usage\"
```

If the agent already supports repository-local skills, this copy is optional;
the checked-in `skills/drocat-usage/SKILL.md` is the source of truth.

## First DROCAT agent request

Open the repository in the agent and paste a focused request:

```text
Use the DROCAT v4.5.0 direct-analysis skill. Run a cached FindPath analysis
from aMe12 to PPL101 in male-cns:v0.9, with max_interlayer=2, CSV output, and
save everything under local_data/agent_runs/aMe12_to_PPL101. Use showfig=False,
inspect the generated files, and summarize row counts and warnings.
```

For a code repair:

```text
Use the DROCAT v4.5.0 direct-analysis skill. Reproduce this direct-script
failure with the smallest query, inspect only the relevant script and backend
signature, patch the call, compile it, and run the focused regression test.
Do not change tokens or dependencies.
```

## Running scripts directly

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

## Safety checklist

- Never paste `NEUPRINT_TOKEN`, `CAVE_TOKEN`, or `token_info_local.txt` into a
  prompt, patch, log, or report.
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
