# *Drosophila* Connectome Analysis Toolkit (DROCAT) v4.5.0

A comprehensive Python toolkit for analyzing and visualizing connectome data from **all NeuPrint databases and FlyWire datasets**. Features type-based pathfinding algorithms, interactive network visualizations with NT grouping, 3D neuron morphology rendering, and EM↔LM driver line mapping.

> [!TIP]
> 🤖 **Agent-assisted installation:** ask your AI agent (e.g., Codex) to run the bundled
> [`drocat-install`](skills/drocat-install/SKILL.md) skill — it installs all dependencies,
> configures tokens, verifies the installation, and launches the web UI for you.
> See [Option 3: Agent-Assisted Install](#option-3-agent-assisted-install-codex).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Key Features

| Category              | Features                                                                   |
| --------------------- | -------------------------------------------------------------------------- |
| **🗄️ Dataset Support** | Inter-dataset analysis and comprehensive dataset support                   |
| **🔬 EM↔LM Mapping**   | NeuronBridge integration for GAL4/Split-GAL4 driver line discovery         |
| **🎨 Visualization**   | 3D skeletons, interactive networks, Sankey diagrams, heatmaps              |
| **📊 Analysis**        | Multi-hop pathfinding, cross-dataset comparison, hemisphere-aware analysis |
| **⚡ Performance**     | 10-100x speedup with local caching, Polars acceleration                    |

---

## 📚 Quick Navigation

### 🚀 Getting Started

| Guide                                                           | Description                         |
| --------------------------------------------------------------- | ----------------------------------- |
| **[Quick Start](QUICK_START.md)**                               | First-time setup and basic examples |
| **[Installation](docs/INSTALLATION.md)**                        | Detailed installation instructions  |
| **[Authentication](docs/INSTALLATION.md#authentication-setup)** | **Set up NeuPrint and CAVE tokens** |
| **[Troubleshooting](docs/TROUBLESHOOTING.md)**                  | Common issues and solutions         |
| **[Agent Install (Codex)](skills/drocat-install/SKILL.md)**     | **Let Codex install, verify & launch DROCAT** |
| **[Basic Usage](#basic-usage)**                                 | Core script tutorials               |

### 📖 Feature Documentation

| Feature                | Guide                                                                       | Script                         |
| ---------------------- | --------------------------------------------------------------------------- | ------------------------------ |
| **Basic Usage**        | [Basic Usage Guide](docs/core-features/BasicUsage_Guide.md)                 | `FindDirect.py`, `FindPath.py` |
| **Score Calculations** | [Score Calculation Guide](docs/core-features/ScoreCalculation_Guide.md)     | All pathfinding scripts        |
| **EM↔LM Mapping**      | [NeuronBridge Guide](docs/core-features/NeuronBridge_Guide.md)              | `NeuronBridge_FindLines.py`    |
| **Line Analysis**      | [Workflow Guide](docs/core-features/NeuronBridge_Workflow.md)               | `NeuronBridge_Colabel.py`      |
| **FlyLight Images**    | [FlyLight Guide](docs/core-features/FlyLight_Guide.md)                      | `FlyLight_fetcher.py`          |
| **Cross-Dataset**      | [Comparison Guide](docs/core-features/CrossDatasetComparison_Guide.md)      | `InterDatasetComparator.py`    |
| **Auto Type Mapping**  | [Type Mapping Guide](docs/AUTO_TYPE_MAPPING.md)                             | Cross-dataset comparisons      |
| **Homolog Finding**    | [Homolog Guide](docs/core-features/HomologFinding_Guide.md)                 | `FindHomologs.py`              |
| **3D Visualization**   | [3D Skeleton Guide](docs/visualizations/3D_Skeleton_Guide.md)               | `plot3dSkeleton.py`            |
| **Path Visualization** | [Interaction Guide](docs/visualizations/VisualizePath_Interaction_Guide.md) | `PlotPath.py`                  |

### 📂 Full Documentation Index

- **[Documentation Hub](docs/README.md)** - Complete documentation index
- **[Core Features](docs/core-features/README.md)** - All feature guides
- **[Visualizations](docs/visualizations/README.md)** - Visualization options
- **[Output Files](docs/OUTPUT_FILES.md)** - File format reference
- **[Available ROI Meshes](docs/AVAILABLE_ROIS.md)** - ROI mesh reference for 3D visualizations

---

## 🔬 NeuronBridge Integration (NEW!)

Find GAL4/Split-GAL4 driver lines matching your EM neurons:

```python
from src.neuronbridge_finder import NeuronBridgeFinder

finder = NeuronBridgeFinder(
    verbose=True,
    separate_splitgal4=True  # Separate GAL4 from Split-GAL4
)

# Find lines for a neuron type across multiple datasets
results = finder.find_lines_batch(
    queries='aMe12',
    dataset=['hemibrain:v1.2.1', 'male-cns:v0.9'],
    output_dir='./results',
    download_img_for_top_n_lines=20
)
```

### Key Features

- **Weighted Score Ranking**: Lines are ranked by `weighted_score = avg_score × coverage_ratio`
- **Multi-Dataset Search**: Search across hemibrain, male-cns, FlyWire FAFB/BANC
- **Co-Labeling Analysis**: Analyze specificity and overlap between driver lines
- **Automatic Image Download**: Download FlyLight imagery for top candidates

📖 **[Full NeuronBridge Workflow Guide](docs/core-features/NeuronBridge_Workflow.md)**

---

## Basic Usage

### Quick Example

```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['KC.*'],       # Regex patterns supported
    targetNeurons=['MBON03'],     # Searches: bodyId → type → instance
    min_synapse_num=10,
)

fc.InitializeNeuronInfo()
fc.FindDirectConnection()  # Direct connections
# or
fc.FindAllPath()           # Multi-hop pathways
```

### Hemisphere-Aware Cross-Dataset Comparison

Enable hemisphere-aware aggregation and symmetry analysis in cross-dataset comparisons:

```python
from comparison import ComparisonParameters, ComparisonAnalyzer

params = ComparisonParameters(
    datasets=['male-cns:v0.9', 'flywire_FAFB_v783'],
    source_neurons=['aMe12'],
    target_neurons=['PPL101'],
    thresholds=[1, 3, 5],
    output_folder='/path/to/output',
    separate_hemispheres=True,  # Adds _L/_R/_U suffixes at type/group level
    keep_only_hemisphere_conserved_connections=True,  # Keep only L/R-conserved edges
    symmetry_analysis=True,     # Auto-enabled when separate_hemispheres=True
    find_reciprocal=True,       # Build reciprocal graphs and reports
)

analyzer = ComparisonAnalyzer(params, verbose=True)
analyzer.run_comparison()
analyzer.generate_report()
```

### Available Methods

| Method                   | Description                 | Use Case         |
| ------------------------ | --------------------------- | ---------------- |
| `FindDirectConnection()` | Direct synaptic connections | One-hop analysis |
| `FindAllPath()`          | Multi-hop pathways          | Circuit tracing  |
| `FetchNeuronsOnly()`     | Get neuron metadata only    | Data exploration |

📖 **[Full Basic Usage Guide](docs/core-features/BasicUsage_Guide.md)** - Detailed examples and parameters

📖 **[Score Calculation Guide](docs/core-features/ScoreCalculation_Guide.md)** - Understanding `connection_ratio`, `traversal_probability`, and path metrics

📖 **[Pathfinding Methods](docs/core-features/PathFinding_Methods.md)** - Algorithm selection for `FindAllPath`

---

## Installation

### Option 1: One-Click Install (Recommended)

**macOS** — Double-click `DROCAT.command` in Finder (or run in Terminal):
```bash
chmod +x DROCAT.command
./DROCAT.command
```

**macOS/Linux** — Run the installer script:
```bash
chmod +x install.sh
./install.sh
```

**Windows** — Double-click `install.bat` or run in PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```

These scripts automatically:
1. Install Miniconda (if not present)
2. Create a `drocat` conda environment with Python 3.11
3. Install all dependencies
4. Create launcher scripts

### Option 2: Manual Install

```bash
# Create environment
conda create -n drocat python=3.11 -y
conda activate drocat

# Install dependencies
pip install -r requirements.txt       # Linux/macOS
pip install -r ui/requirements.txt    # Web UI (NiceGUI)
pip install -e .                      # Install package in editable mode

# Windows users:
pip install -r requirements-windows.txt
pip install neuronbridge-python --no-deps
pip install -r ui/requirements.txt
pip install -e .
```

### Option 3: Agent-Assisted Install (Codex)

DROCAT ships a Codex skill that lets an AI agent install, verify, and launch the toolkit for you:

- **Skill source (repo):** [`skills/drocat-install/`](skills/drocat-install/SKILL.md)
- **Global copy (auto-discovered):** `~/.codex/skills/drocat-install/`

**Usage** — open Codex in this repository and ask:

> Install DROCAT on this machine and verify it works.

The agent will:

1. Run the OS-appropriate installer (`install.sh` / `install.ps1` / `install.bat`)
2. Create the `drocat` conda environment (Python 3.11) and install all dependencies
3. Ask you for NeuPrint / CAVE tokens and write them to `token_info_local.txt`
4. Verify the installation with [`verify_install.py`](skills/drocat-install/scripts/verify_install.py) (Python version, imports, token file, UI)
5. Launch the web UI and confirm it responds at <http://127.0.0.1:8080>

Manual verification (or to re-check an existing install):

```bash
conda activate drocat
python skills/drocat-install/scripts/verify_install.py --project .
```

To install/refresh the global skill copy after updating the repo version:

```bash
mkdir -p ~/.codex/skills/drocat-install
cp -R skills/drocat-install/. ~/.codex/skills/drocat-install/
```

### Authentication Setup

1. **NeuPrint Token** (required for NeuPrint datasets):
   - Visit [neuprint.janelia.org/account](https://neuprint.janelia.org/account)
   - Login → Copy Auth Token
   - Enter in UI Settings tab or save to `token_info_local.txt`

2. **CAVE Token** (required for FlyWire FAFB/BANC datasets):
   - Visit [codex.flywire.ai/auth_token](https://codex.flywire.ai/auth_token)
   - Copy token → Enter in UI Settings tab

📖 **[Full Installation Guide](docs/INSTALLATION.md)**

---

## 🖥️ Web UI (v4.5.0)

DROCAT includes a local web interface (light theme, photo-selector-inspired design) for all analysis tools.

### Launch the UI

**macOS** — Double-click `DROCAT.command` in Finder, or:
```bash
./DROCAT.command
```

**macOS/Linux:**
```bash
./run_ui.sh
```

**Windows:**
```
run_ui.bat
```

**Manual:**
```bash
conda activate drocat
python ui/app.py
```

The UI opens at **http://127.0.0.1:8080**

### UI Features

| Tab | Tool | Description |
|-----|------|-------------|
| **Pathfinding** | FindPath | Multi-hop path analysis between neuron groups |
| **Direct** | FindDirect | Direct synaptic connection analysis |
| **Profiling** | ConnectivityProfiling | Intra-dataset connectivity profile comparison |
| **Homologs** | FindHomologs | Cross-dataset homolog finding |
| **Cross-Dataset** | InterDatasetComparator | Compare connectivity across datasets |
| **Find Lines** | NeuronBridge_FindLines | EM→LM driver line search |
| **Find Neurons** | NeuronBridge_FindNeuron | LM→EM neuron search |
| **Co-Labeling** | NeuronBridge_Colabel | Driver line co-labeling analysis |
| **Visualization** | plot3dSkeleton/PlotPath | 3D neuron visualization |
| **Settings** | - | Token configuration and dataset status |

### Advanced Neuron Search

All neuron input fields support **filter modes**:

| Mode | Behavior | Example |
|------|----------|--------|
| Exact | Match type/bodyId exactly | `aMe12` → only aMe12 |
| Starts with | Types beginning with text | `DN` → DN1p, DN2, DNa01... |
| Contains | Types containing text | `PN` → adPN, lPN, vPN... |
| Ends with | Types ending with text | `_R` → all right-hemisphere types |
| Regex | Full regex pattern | `KC.*` → all KC types |

### Dataset Selection

- **NeuPrint datasets** (hemibrain, male-cns v1.0/v0.9, optic-lobe, manc): Fetched from server automatically
- **FlyWire FAFB v783**: Requires CAVE token + manual data download
- **FlyWire BANC v888/v626**: Requires CAVE token + manual data download

See the **Settings** tab for detailed setup instructions.

### Directory Selection

All output path fields include a browse button (📁) that opens your system's native file picker dialog.

---

## Supported Datasets

| Dataset             | Type      | Description               |
| ------------------- | --------- | ------------------------- |
| `male-cns:v1.0`     | NeuPrint  | Full male CNS (latest)    |
| `male-cns:v0.9`     | NeuPrint  | Full male CNS             |
| `hemibrain:v1.2.1`  | NeuPrint  | Adult fly brain (central) |
| `optic-lobe:v1.1`   | NeuPrint  | Optic lobe detailed       |
| `manc:v1.2.1`       | NeuPrint  | Male VNC                  |
| `flywire_FAFB_v783` | FlyWire   | Female brain (CAVE+local) |
| `flywire_BANC_v888` | FlyWire   | Male VNC latest (CAVE+local) |
| `flywire_BANC_v626` | FlyWire   | Male VNC (CAVE+local)     |

📖 **[FlyWire Setup Guide](docs/FLYWIRE_USAGE.md)**

---

## Output Examples

### NeuronBridge FindLines Output

```
findlines_aMe12_20241230/
├── line_summary.csv           # Ranked by weighted_score
├── gal4_lexa_summary.csv      # GAL4/LexA lines
├── split_gal4_summary.csv     # Split-GAL4 lines
├── all_lines.csv              # All matches
└── images/                    # Downloaded FlyLight images
```

### Pathfinding Output

```
connection_data/
├── network.html               # Interactive network
├── sankey.html                # Flow diagram
├── heatmap.html               # Connection matrix
└── paths.csv                  # Path data
```

📖 **[Output Files Reference](docs/OUTPUT_FILES.md)**

---

## Performance Features

| Feature         | Speedup | Description                      |
| --------------- | ------- | -------------------------------- |
| **Local Cache** | 10-100x | Automatic caching of API results |
| **Polars**      | 5-50x   | Fast CSV/matrix operations       |
| **Batch Mode**  | 2-10x   | Optimized batch processing       |

```python
# Enable caching
fc = FindNeuronConnection(
    use_cache=True,  # Automatic local caching
    # ...
)
```

📖 **[Cache System Guide](docs/core-features/CacheSystem_Guide.md)**

---

## What's New in v4.4.0

### 🚀 Local FAFB/BANC Dataset Support (RECOMMENDED)
- **Local-first architecture**: Store FlyWire datasets locally for 10-100x faster access
- **Mixed mode**: Seamlessly combines local cache + API fallback
- **Zero API latency**: Instant queries for cached neurons
- **Automatic caching**: Build your cache once, reuse forever

📖 **[FAFB Integration Guide](docs/FAFB_INTEGRATION.md)** | **[Cache System](docs/core-features/CacheSystem_Guide.md)**

### 🔍 Priority-Based Neuron Search
- **Smart search order**: bodyId → type → instance with automatic fallback
- **Flexible input**: Accept both int and string bodyIds: `[123456789]` or `['123456789']`
- **Regex support**: Use patterns like `['KC.*']`, `['.*PN.*']` across all columns
- **Consistent matching**: String-based comparison internally for reliability

### 🎨 NT Visualization & Grouping
- **NT edge groups**: ACH, GABA, GLUT, DA, SER, OCT - select and style by neurotransmitter
- **Custom groups**: Create and save custom element groups for batch editing
- **Hover labels**: NT type displayed in edge tooltips with color coding
- **Export/Import**: Save complete graph states including custom groups and NT settings
- **Default opacity**: 50% for edges (vs 20%), 100% for nodes (vs 50%) - better visibility

📖 **[Network Features Guide](docs/visualizations/VisualizePath_Network_Features.md)** | **[Interaction Guide](docs/visualizations/VisualizePath_Interaction_Guide.md)**

### 🔐 Authentication Improvements
- **token_info.txt recommended**: Store all API tokens in one file (NeuPrint, CAVE, NeuronBridge)
- **Automatic loading**: No need to pass tokens manually in scripts
- **Secure storage**: Keep credentials out of version control

📖 **[Authentication Setup](docs/INSTALLATION.md#authentication-setup)**

### Previous Updates (v4.3)
- Weighted Score Ranking for NeuronBridge
- HomologFinder with hierarchical ConnectivityStatus
- Polars integration for 10-100x faster operations

📖 **[Full Changelog](docs/README.md#recent-updates-december-2025---v43)**

---

## Examples

See the `examples/` folder for complete working examples:

| Example          | Description                         |
| ---------------- | ----------------------------------- |
| `basic/`         | Basic pathfinding and visualization |
| `comparison/`    | Cross-dataset comparison            |
| `visualization/` | Advanced visualization options      |

---

## Contributing

Contributions are welcome! Please read the contribution guidelines before submitting PRs.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this toolkit in your research, please cite the relevant connectome datasets and this repository.

---

## Support

- **[Documentation](docs/README.md)** - Full documentation
- **[Examples](examples/)** - Working code examples
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[Issues](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/issues)** - Bug reports and feature requests

## Report an Issue

If you encounter bugs or have feature requests:

1. **Check [Troubleshooting Guide](docs/TROUBLESHOOTING.md)** for common solutions
2. **Search [existing issues](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/issues)**
3. **Email the author:** krleng(a*t)pku.edu.cn (replace `(a*t)` with `@`)

When reporting, please include:
- Python version and OS
- Full error traceback
- Minimal code to reproduce the issue
