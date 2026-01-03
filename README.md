# Drosophila Connectome Analysis Toolkit (DROCAT) v4.3.3

A comprehensive Python toolkit for analyzing and visualizing connectome data from **all NeuPrint databases and FlyWire datasets**. Features type-based pathfinding algorithms, interactive network visualizations, 3D neuron morphology rendering, and EM↔LM driver line mapping.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Key Features

| Category | Features |
|----------|----------|
| **🗄️ Dataset Support** | Inter-dataset analysis and comprehensive dataset support |
| **🔬 EM↔LM Mapping** | NeuronBridge integration for GAL4/Split-GAL4 driver line discovery |
| **🎨 Visualization** | 3D skeletons, interactive networks, Sankey diagrams, heatmaps |
| **📊 Analysis** | Multi-hop pathfinding, cross-dataset comparison, homolog finding |
| **⚡ Performance** | 10-100x speedup with local caching, Polars acceleration |

---

## 📚 Quick Navigation

### 🚀 Getting Started

| Guide | Description |
|-------|-------------|
| **[Quick Start](QUICK_START.md)** | First-time setup and basic examples |
| **[Installation](docs/INSTALLATION.md)** | Detailed installation instructions |
| **[Basic Usage](#basic-usage)** | Core script tutorials |

### 📖 Feature Documentation

| Feature | Guide | Script |
|---------|-------|--------|
| **EM↔LM Mapping** | [NeuronBridge Guide](docs/core-features/NeuronBridge_Guide.md) | `NeuronBridge_FindLines.py` |
| **Line Analysis** | [Workflow Guide](docs/core-features/NeuronBridge_Workflow.md) | `NeuronBridge_Colabel.py` |
| **FlyLight Images** | [FlyLight Guide](docs/core-features/FlyLight_Guide.md) | `FlyLight_fetcher.py` |
| **Cross-Dataset** | [Comparison Guide](docs/core-features/CrossDatasetComparison_Guide.md) | `InterDatasetComparator.py` |
| **Homolog Finding** | [Homolog Guide](docs/core-features/HomologFinding_Guide.md) | `FindHomologs.py` |
| **3D Visualization** | [3D Skeleton Guide](docs/visualizations/3D_Skeleton_Guide.md) | `plot3dSkeleton.py` |

### 📂 Full Documentation Index

- **[Documentation Hub](docs/README.md)** - Complete documentation index
- **[Core Features](docs/core-features/README.md)** - All feature guides
- **[Visualizations](docs/visualizations/README.md)** - Visualization options
- **[Output Files](docs/OUTPUT_FILES.md)** - File format reference

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

### FindDirect.py - Direct Connections

```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(
    token='your_neuprint_token',
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['KC.*'],
    targetNeurons=['MBON03'],
    min_synapse_num=10,
)

fc.InitializeNeuronInfo()
fc.FindDirectConnection()
```

### FindPath.py - Multi-Hop Pathways

```python
fc = FindNeuronConnection(
    token='your_neuprint_token',
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['.*PN.*'],
    targetNeurons=['MBON.*'],
    max_interlayer=2,  # Up to 2 intermediate layers
    keyword_in_path_to_remove=['None', 'APL'],
)

fc.InitializeNeuronInfo()
fc.FindAllPath()
```

📖 **[Detailed Usage Guide](docs/core-features/PathFinding_Methods.md)**

---

## Installation

### Quick Install (Conda Recommended)

```bash
# Create environment
conda create -n drocat python=3.11 -y
conda activate drocat

# Install dependencies
pip install -r requirements.txt  # Linux/macOS
# or
pip install -r requirements-windows.txt  # Windows
```

### Get NeuPrint Token

1. Visit [NeuPrint](https://neuprint.janelia.org/)
2. Login → Account → Copy Auth Token

📖 **[Full Installation Guide](docs/INSTALLATION.md)**

---

## Supported Datasets

| Dataset | Type | Description |
|---------|------|-------------|
| `hemibrain:v1.2.1` | NeuPrint | Adult fly brain (central) |
| `male-cns:v0.9` | NeuPrint | Full male CNS |
| `optic-lobe:v1.1` | NeuPrint | Optic lobe detailed |
| `manc:v1.2.3` | NeuPrint | Male VNC |
| `flywire_FAFB_v783` | Local | FlyWire female brain |
| `flywire_BANC_v626` | Local | FlyWire male VNC |

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

| Feature | Speedup | Description |
|---------|---------|-------------|
| **Local Cache** | 10-100x | Automatic caching of API results |
| **Polars** | 5-50x | Fast CSV/matrix operations |
| **Batch Mode** | 2-10x | Optimized batch processing |

```python
# Enable caching
fc = FindNeuronConnection(
    use_cache=True,  # Automatic local caching
    # ...
)
```

📖 **[Cache System Guide](docs/core-features/CacheSystem_Guide.md)**

---

## What's New in v4.3

### NeuronBridge Enhancements
- **Weighted Score Ranking**: `weighted_score = avg_score × coverage_ratio`
- **Coverage Ratio**: Fraction of queried neurons labeled by each line
- **Multi-Type Query**: Find lines labeling ALL queried neuron types

### HomologFinder Improvements
- Hierarchical ConnectivityStatus (5 levels)
- Dict-based similarity_metric for weighted combinations
- Vector prefiltering for faster candidate selection

### Performance
- Polars integration for 10-100x faster operations
- Skip BodyId processing option for large-scale analyses

📖 **[Full Changelog](docs/README.md#recent-updates-december-2025---v43)**

---

## Examples

See the `examples/` folder for complete working examples:

| Example | Description |
|---------|-------------|
| `basic/` | Basic pathfinding and visualization |
| `comparison/` | Cross-dataset comparison |
| `visualization/` | Advanced visualization options |

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
- **[Issues](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/issues)** - Bug reports and feature requests
