# *Drosophila* Connectome Analysis Toolkit (DROCAT) v4.5.0

A Python toolkit for analyzing and visualizing connectome data from **all NeuPrint databases and FlyWire datasets**: type-based pathfinding, interactive network visualizations with NT grouping, 3D neuron morphology rendering, cross-dataset comparison, and EM↔LM driver line mapping (NeuronBridge).

> [!TIP]
> 🤖 **Agent-assisted:** ask your AI agent to run the bundled
> [`drocat-install`](skills/drocat-install/SKILL.md) skill — it fetches this repository,
> installs all dependencies, configures tokens, verifies the installation, and launches
> the web UI for you. For script analysis without the UI, use the
> [`drocat-usage`](skills/drocat-usage/SKILL.md) skill. New to agents? Start with the
> [agent setup section](docs/INSTALLATION.md#5-agent-assisted-install--agent-setup) of the installation guide.
>
> **One-line agent handoff (no local repo needed):**
>
> > Fetch https://raw.githubusercontent.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/v4.5.0/skills/drocat-usage/SKILL.md and follow it to install and use the DROCAT v4.5.0 direct-analysis skill for this repository, then finish the requested analysis end-to-end without opening the UI.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Key Features

| Category | Features |
| --- | --- |
| **🗄️ Dataset Support** | NeuPrint (hemibrain, male-cns, optic-lobe, manc) + FlyWire (FAFB, BANC), inter-dataset analysis |
| **🔬 EM↔LM Mapping** | NeuronBridge integration for GAL4/Split-GAL4 driver line discovery |
| **🎨 Visualization** | 3D skeletons, interactive networks, Sankey diagrams, heatmaps |
| **📊 Analysis** | Multi-hop pathfinding, cross-dataset comparison, hemisphere-aware analysis |
| **⚡ Performance** | 10-100x speedup with local caching, Polars acceleration |

---

## Getting Started

| Guide | Description |
| --- | --- |
| **[Quick Start](docs/QUICK_START.md)** | First-time setup and basic examples |
| **[Installation](docs/INSTALLATION.md)** | One-click, agent-assisted, and manual install + token & agent setup |
| **[Script Examples](docs/core-features/ScriptExamples_Guide.md)** | Copy-paste code for pathfinding, comparison, NeuronBridge |
| **[Troubleshooting](docs/TROUBLESHOOTING.md)** | Common issues and solutions |
| **[Documentation Hub](docs/README.md)** | Full documentation index |

## Feature Documentation

| Feature | Guide | Script |
| --- | --- | --- |
| **Basic Usage** | [Basic Usage Guide](docs/core-features/BasicUsage_Guide.md) | `FindDirect.py`, `FindPath.py` |
| **Score Calculations** | [Score Calculation Guide](docs/core-features/ScoreCalculation_Guide.md) | All pathfinding scripts |
| **EM↔LM Mapping** | [NeuronBridge Guide](docs/core-features/NeuronBridge_Guide.md) | `NeuronBridge_FindLines.py` |
| **Cross-Dataset** | [Comparison Guide](docs/core-features/CrossDatasetComparison_Guide.md) | `InterDatasetComparator.py` |
| **Homolog Finding** | [Homolog Guide](docs/core-features/HomologFinding_Guide.md) | `FindHomologs.py` |
| **3D Visualization** | [3D Skeleton Guide](docs/visualizations/3D_Skeleton_Guide.md) | `plot3dSkeleton.py` |
| **Path Visualization** | [Interaction Guide](docs/visualizations/VisualizePath_Interaction_Guide.md) | `PlotPath.py` |
| **UI Guides** | [docs/ui_guides/README.html](docs/ui_guides/README.html) | All web UI panels |
| **Output Files** | [Output Files Reference](docs/OUTPUT_FILES.md) | File formats |

---

## Installation

After cloning the repository
(`git clone --branch v4.5.0 https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis.git drocat && cd drocat`),
two options:

**Option 1 — One-click install**

| Platform | Command |
| --- | --- |
| macOS / Linux | `bash archive/install/install.sh` |
| Windows | `powershell -ExecutionPolicy Bypass -File archive/install/install.ps1` |

The installer discovers conda (installing Miniconda if missing), creates the
versioned `drocat-4.5.0` Python 3.11 environment, installs the pinned
dependencies, runs `pip check`, and verifies the installation. Alternatively,
double-click `run_DROCAT.command` (macOS) / `run_DROCAT.bat` (Windows) — it
installs on first run and launches the web UI.

**Option 2 — Agent-assisted install**

Ask your AI agent to fetch and follow the install skill:

> Fetch https://raw.githubusercontent.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/v4.5.0/skills/drocat-install/SKILL.md and follow it to finish installing, verifying, and launching DROCAT on this machine.

For script analysis without the UI, the agent should instead fetch the
direct-analysis skill:

> Fetch https://raw.githubusercontent.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/v4.5.0/skills/drocat-usage/SKILL.md and follow it to install and use the DROCAT v4.5.0 direct-analysis skill for this repository, then finish the requested analysis end-to-end without opening the UI.

📖 **[Full Installation Guide](docs/INSTALLATION.md)** — manual steps, environment policy, token setup (NeuPrint / CAVE), and the Codex + DeepSeek (`deepseek-v4-flash`) agent configuration.

---

## Launch the Web UI

| Platform | Command |
| --- | --- |
| macOS | Double-click `run_DROCAT.command`, or `./run_DROCAT.command` |
| Linux | `./run_DROCAT.command` |
| Windows | Double-click `run_DROCAT.bat`, or `run_DROCAT.bat` |
| Manual | `conda activate drocat-4.5.0 && python ui/app.py` |

The UI opens at **http://127.0.0.1:8080** — every panel links to its own instruction guide (see [docs/ui_guides/README.html](docs/ui_guides/README.html)).

---

## Supported Datasets

All NeuPrint server datasets are supported (verified against `api.neuprint.janelia.org`), plus the FlyWire/Codex datasets. NeuPrint datasets are fetched automatically; FlyWire datasets use local files (see the Settings tab).

### NeuPrint (11)

| Dataset | Description (from NeuPrint server) |
| --- | --- |
| `male-cns:v1.0` | Complete MaleCNS connectome (Janelia FlyEM + Cambridge) — latest |
| `male-cns:v0.9` | Complete MaleCNS connectome |
| `hemibrain:v1.2.1` | Adult female brain reconstruction (central complex + surrounding neuropils) |
| `hemibrain:v1.1` | Older hemibrain release |
| `optic-lobe:v1.1` | Drosophila optic lobe (right lobe, ~50k neurons, subset of MaleCNS) |
| `optic-lobe:v1.0.1` | Fly optic lobe reconstruction (~50k neurons) |
| `manc:v1.2.3` | MANC connectome — latest |
| `manc:v1.2.1` | MANC connectome |
| `manc:v1.0` | MANC connectome (original) |
| `fib19:v1.0` | Partial reconstruction of the fly medulla / lobula / lobula plate |
| `mushroombody` | Fly alpha lobe in the mushroom body (983 neurons) |

### FlyWire / Codex (3, local files required)

| Dataset | Description |
| --- | --- |
| `flywire_FAFB_v783` | Female Adult Fly Brain (FAFB v783, 139,255 neurons) |
| `flywire_BANC_v888` | Brain and Nerve Cord (BANC v888, 158,262 neurons) |
| `flywire_BANC_v626` | Brain and Nerve Cord, older (BANC v626, 115,151 neurons) |

> BANC (Brain And Nerve Cord) is served via FlyWire/Codex as `flywire_BANC_v888` (local data files). The NeuPrint server metadata also lists a hidden `banc:v888` entry, but it is not queryable through the NeuPrint API and is therefore not supported.

📖 **[FlyWire Setup Guide](docs/FLYWIRE_USAGE.md)** | **[Available ROI Meshes](docs/AVAILABLE_ROIS.md)**

---

## What's New in v4.5.0

- **Script-first analysis with coding agents** — run pathfinding, comparison, NeuronBridge, FlyLight, homolog, profile, PlotPath, and 3D skeleton scripts without the UI, via the [`drocat-usage`](skills/drocat-usage/SKILL.md) skill and its `run_direct.py` launcher.
- **Local FAFB/BANC dataset support** — local-first caching for 10-100x faster FlyWire access ([FAFB Integration](docs/FAFB_INTEGRATION.md)).
- **NT visualization & grouping** — neurotransmitter edge groups, custom groups, export/import ([Network Features](docs/visualizations/VisualizePath_Network_Features.md)).

📖 **[Full changelog](docs/README.md#recent-updates)** | **[Agent setup](docs/INSTALLATION.md#5-agent-assisted-install--agent-setup)** — including the low-cost DeepSeek (`deepseek-v4-flash`) Codex configuration

---

## Contributing

Contributions are welcome — please open an issue or PR on [GitHub](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis).

## License

MIT License — see [LICENSE](LICENSE).

## Support

- **[Documentation](docs/README.md)** — full index
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** — common issues
- **[Issues](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/issues)** — bug reports and feature requests

When reporting an issue, include: Python version and OS, the full error traceback, and minimal code to reproduce.
