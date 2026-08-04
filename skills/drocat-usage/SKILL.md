---
name: drocat-usage
description: Directly run, explain, debug, and safely modify DROCAT v4.5.0 connectome-analysis scripts without the NiceGUI UI. Use when an agent needs to perform pathfinding, direct-connection analysis, cross-dataset comparison, homolog or connectivity-profile analysis, NeuronBridge/FlyLight searches, PlotPath or 3D skeleton visualization, empty-network generation, output inspection, or targeted backend fixes in this repository.
---

# DROCAT Direct Analysis

Use this skill to operate the v4.5.0 backend as a script-driven research
tool. Keep the UI out of the execution path: edit a focused script or create a
small run script, execute it in the versioned environment, inspect the generated
artifacts, and report the exact parameters and output paths.

## Operating contract

1. Confirm the repository root and branch before running anything:

   ```bash
   cd /path/to/Drosophila-cross-dataset-connectome-analysis
   git branch --show-current
   git status --short
   ```

   Use branch `v4.5.0`. If the checkout is elsewhere, resolve that path first;
   do not clone a second copy over an existing project.

2. Use Python 3.11 in the `drocat-4.5.0` environment. Prefer
   `conda run -n drocat-4.5.0 ...` for a reproducible non-interactive command;
   an already activated environment is also acceptable. Set
   `PYTHONNOUSERSITE=1` so user-site packages cannot shadow the pinned
   environment.

3. Run the example scripts with their own directory as the working directory.
   Many templates use paths such as `../local_data`; running from the repository
   root changes where those paths resolve. Use the bundled launcher for this:

   ```bash
   python skills/drocat-usage/scripts/run_direct.py \
     --conda-env drocat-4.5.0 \
     --script scripts/FindPath.py
   ```

   The launcher sets `PYTHONPATH` for `src/`, the repository root, and
   `vispath-subproject/src/`, streams output, and never invokes a shell. Use
   `--dry-run` to inspect the resolved command and working directory.

   The files under `scripts/` are configuration templates, not a stable CLI;
   never assume their sample dataset, neurons, or output folder match the
   user's request. Copy a template into `scripts_local/` (or patch an
   explicitly requested source file) before a real scientific run.

4. Never print, commit, or paste `token_info_local.txt`, API keys, cookies, or
   full authentication environment variables. Read credentials through the
   repository token manager or the environment. Ask before changing tokens,
   downloading large datasets, or deleting/replacing result folders.

5. Preserve the example templates. Before a substantial edit, copy the script
   to a run-specific file (for example,
   `scripts_local/agent_FindPath_2026-08-04.py`) or create a new script in an
   explicitly named output directory. Use `apply_patch` for code changes, then
   compile the changed file and run the smallest relevant test.

6. Use absolute output paths when possible. Keep each run isolated, record the
   dataset, source/target queries, filters, cache mode, and output directory,
   and inspect the generated CSV/HTML/JSON before claiming success.

## Choose the backend operation

Load [references/tool-catalog.md](references/tool-catalog.md) when selecting a
tool or checking its constructor and method names.

| User goal | Direct entry point | Primary result |
| --- | --- | --- |
| Direct one-hop edges | `scripts/FindDirect.py` → `FindDirectConnections()` | edge tables and optional HTML/heatmap |
| Multi-hop pathways | `scripts/FindPath.py` → `FindAllPath()` | type/bodyId path tables and summaries |
| Render path results | `scripts/PlotPath.py` → `VisualizePath.visualize()` | network, Sankey, heatmap HTML |
| Create a blank editable graph | `VisualizePath(..., generate_empty_network=True)` | Cytoscape HTML canvas |
| Render neuron morphology | `scripts/plot3dSkeleton.py` → `VisualizeSkeleton.plot_neurons()` | interactive 3D HTML and optional exports |
| Compare datasets | `scripts/InterDatasetComparator.py` | comparison tables, report, conserved-path views |
| Find homologs | `scripts/FindHomologs.py` → `HomologFinder.find_homologs_fast()` | bodyId/type homolog tables |
| Compare connectivity profiles | `scripts/ConnectivityProfiling.py` | similarity matrices and heatmaps |
| Find EM→LM driver lines | `scripts/NeuronBridge_FindLines.py` | ranked line summaries and images |
| Find LM→EM neurons | `scripts/NeuronBridge_FindNeuron.py` | ranked neuron matches and optional 3D views |
| Analyze line co-labeling | `scripts/NeuronBridge_Colabel.py` | expression, similarity, specificity, report |
| Download FlyLight files | `scripts/FlyLight_fetcher.py` | images/metadata under the selected output folder |

## Standard agent workflow

### 1. Clarify the analysis

Before execution, identify:

- dataset name and whether it is NeuPrint or local FlyWire/BANC data;
- source and target neuron types, instances, bodyIds, or regex patterns;
- direct versus multi-hop analysis and maximum intermediate layers;
- minimum synapse/ratio/probability thresholds and bodyId/type filtering;
- whether the user wants CSV, Excel, HTML, images, PDF/PPTX, or only tables;
- output directory and whether cached/offline data may be used.

If a request is underspecified, make the smallest safe assumption and state it
before executing a remote query.

### 2. Inspect only the focused contract

Do not read the whole repository. Start with the selected example and the
target class/method definitions:

```bash
rg -n "class |def |if __name__|output_dir|dataset|sourceNeurons|targetNeurons" \
  scripts/FindPath.py src/coana.py
```

For a signature check, use a short import-time probe instead of loading large
files into context:

```bash
PYTHONNOUSERSITE=1 conda run -n drocat-4.5.0 python - <<'PY'
import inspect
from coana import FindNeuronConnection
print(inspect.signature(FindNeuronConnection))
print(inspect.signature(FindNeuronConnection.FindAllPath))
PY
```

Read the relevant reference file only when the selected operation needs it.

### 3. Prepare a run

For a template edit, change only the configuration block above the execution
guard. Prefer explicit values:

```python
from pathlib import Path

OUTPUT = Path("/absolute/path/to/local_data/agent_runs/aMe12_to_PPL101")
OUTPUT.mkdir(parents=True, exist_ok=True)

fc = FindNeuronConnection(
    dataset="male-cns:v0.9",
    sourceNeurons=["aMe12"],
    targetNeurons=["PPL101"],
    output_dir=str(OUTPUT),
    min_synapse_num=3,
    max_interlayer=2,
    filter_by="bodyId",
    use_cache=True,
    output_format="csv",
    showfig=False,
)
fc.InitializeNeuronInfo()
fc.FindAllPath(forward_only=True)
```

Use `showfig=False` for unattended analysis. Generate HTML separately with
PlotPath when the data is ready. Set `cache_only=True` only when the required
local cache has already been verified.

### 4. Execute and monitor

Run the script through `run_direct.py` or from its directory:

```bash
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 \
  --script scripts_local/agent_FindPath_2026-08-04.py
```

Capture the exit code and last meaningful log lines. A zero exit code is not
enough: confirm that expected files exist and contain rows/HTML content.

### 5. Inspect and report

Use targeted, read-only checks:

```bash
find /absolute/path/to/output -maxdepth 2 -type f -print | sort
python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("/absolute/path/to/output")
for path in sorted(root.rglob("*.csv"))[:5]:
    frame = pd.read_csv(path)
    print(path, frame.shape, list(frame.columns)[:8])
PY
```

Report the exact run command, parameters that matter, output folder, row/file
counts, warnings, and any follow-up visualization or validation performed.

## Direct-script safety and repair

- Validate dataset spelling against the configured dataset list before an API
  call. NeuPrint names include versions, for example `hemibrain:v1.2.1` and
  `male-cns:v0.9`; FlyWire names include `flywire_FAFB_v783` and
  `flywire_BANC_v888`.
- Keep `use_cache=True` for repeat work. For deprecated or offline datasets,
  use `cache_only=True` only after checking cache coverage.
- Prefer CSV for large intermediate tables and Excel only when the user needs
  workbook sheets.
- Limit graph size with `edgeN_limit` before generating an interactive HTML.
- For 3D renders, use `skeleton_mode="line"` and disable exports during a first
  smoke test; use tube/WebDriver/PDF/video only after the HTML is valid.
- For PlotPath, pass `path_file` from a completed FindPath result. Use
  `generate_empty_network=True` and `path_file=None` only for an empty editable
  Cytoscape canvas.
- When modifying a backend, reproduce the bug with the smallest local fixture,
  patch the narrowest function, run `python -m py_compile` and the relevant
  tests, then rerun the original command.
- Do not “fix” a failing run by silently lowering thresholds, switching datasets,
  or disabling validation. Explain the trade-off and ask if a change affects the
  scientific interpretation.

## References

- [Tool catalog and contracts](references/tool-catalog.md)
- [Workflow recipes](references/workflow-recipes.md)
- [Datasets, authentication, and output files](references/datasets-and-auth.md)
- [Beginner agent setup with DeepSeek and Codex](references/deepseek-codex.md)

Use [the install skill](../drocat-install/SKILL.md) only for installation,
environment repair, token bootstrap, or UI launch. This skill assumes the
environment is already installed and focuses on direct analysis.

## Install the skill for future agent sessions

When the agent does not automatically discover repository-local skills, copy
this directory into the user skill directory from the repository root:

```bash
mkdir -p ~/.codex/skills/drocat-usage
cp -R skills/drocat-usage/. ~/.codex/skills/drocat-usage/
```

On Windows PowerShell, use:

```powershell
New-Item -ItemType Directory -Force "$HOME\.codex\skills\drocat-usage" | Out-Null
Copy-Item -Recurse -Force "skills\drocat-usage\*" "$HOME\.codex\skills\drocat-usage\"
```

Re-run the copy after updating this branch so the agent sees the current
catalog and recipes. The skill contains no credentials and is safe to keep in
the user-level skill directory.
