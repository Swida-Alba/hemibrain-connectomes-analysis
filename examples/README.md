# Examples Directory

This directory contains example scripts demonstrating the key features of the Hemibrain Connectomes Analysis toolkit, organized by category.

## 📁 Directory Structure

```
examples/
├── README.md                 # This file
├── basic/                    # Basic usage examples
├── comparison/               # Cross-dataset comparison examples
├── data/                     # Sample data files (CSV, JSON)
├── performance/              # Caching and parallel processing examples
└── visualization/
    ├── input_formats/        # Data format and input examples
    └── skeleton/             # 3D skeleton visualization examples
```

---

## 📘 Basic Usage Examples (`basic/`)

### 1. Empty Network Template
**File:** `basic/Example_EmptyNetwork.py`

Generate an empty network HTML template for custom visualizations.

```bash
python examples/basic/Example_EmptyNetwork.py
```

### 2. Visualizing Selected Pathways
**File:** `basic/Example_VisualizeSelectedPaths.py`

Create publication-ready pathway visualizations using standalone VisualizePath.

```bash
python examples/basic/Example_VisualizeSelectedPaths.py
```

**Features:** Standalone usage, custom colors, network + Sankey visualizations

### 3. Exclude Intra-Type Connections
**File:** `basic/Example_ExcludeIntraType.py`

Network visualization focused on inter-type connectivity (e.g., MBON→MBON analysis).

```bash
python examples/basic/Example_ExcludeIntraType.py
```

### 4. ROI Listing Demo
**File:** `basic/Example_ROI_Listing_Demo.py`

Direct demonstration of ROI listing with environment variables and caching.

```bash
python examples/basic/Example_ROI_Listing_Demo.py
```

---

## 🔄 Comparison Examples (`comparison/`)

### 1. Inter-Dataset Comparison
**File:** `comparison/Example_InterDatasetComparison.py`

Full cross-dataset comparison workflow (hemibrain vs male-cns) with HTML reports.

```bash
python examples/comparison/Example_InterDatasetComparison.py
```

**Features:** Multi-threshold analysis, comparison metrics, HTML report generation

### 2. Homolog Finding
**File:** `comparison/Example_HomologFinding.py`

Find homologous neurons across datasets using connectivity profiles.

```bash
python examples/comparison/Example_HomologFinding.py
```

**Features:**
- Connectivity profile matching using 1-hop/2-hop hybrid approach
- Cross-dataset homolog identification
- Both bodyId-level and type-level results saved automatically
- Output files: `bodyid_results.csv` (sorted by source, rank_corr), `type_summary.csv`

---

## ⚡ Performance Examples (`performance/`)

### 1. Caching System
**File:** `performance/Example_CachingDemo.py`

Demonstrates the local caching feature for 10-100x speedup on repeated analyses.

```bash
python examples/performance/Example_CachingDemo.py
```

**Features:** Cache enable/disable, performance comparison, cache management

### 2. Parallel Processing
**File:** `performance/Example_ParallelProcessing.py`

Multi-core parallel processing for 4-14x speedup on large datasets.

```bash
python examples/performance/Example_ParallelProcessing.py
```

**Features:** Sequential vs. parallel comparison, CPU core configuration

### 3. Build Connectivity Cache
**File:** `performance/Example_BuildConnectivityCache.py`

Pre-build connectivity profile cache for efficient homolog finding.

```bash
python examples/performance/Example_BuildConnectivityCache.py
```

---

## 🎨 Visualization Examples

### Input Format Examples (`visualization/input_formats/`)

#### 1. All Input Formats Test
**File:** `visualization/input_formats/Example_AllInputFormats_Test.py`

Comprehensive test for VisualizePath input formats: connection matrix, edge-list, paths.

```bash
python examples/visualization/input_formats/Example_AllInputFormats_Test.py
```

#### 2. Simple Edge List
**File:** `visualization/input_formats/Example_SimpleEdgeList.py`

Flexible edge-list format with auto-detection of source/target/weight columns.

```bash
python examples/visualization/input_formats/Example_SimpleEdgeList.py
```

#### 3. Connection Matrix 10x12
**File:** `visualization/input_formats/Example_ConnMatrix_10x12.py`

Generate and visualize a 10x12 connection matrix.

```bash
python examples/visualization/input_formats/Example_ConnMatrix_10x12.py
```

#### 4. Connection Matrix Heatmap
**File:** `visualization/input_formats/Example_ConnectionMatrixHeatmap.py`

Basic 3x3 connection matrix heatmap visualization.

```bash
python examples/visualization/input_formats/Example_ConnectionMatrixHeatmap.py
```

#### 5. File Picker
**File:** `visualization/input_formats/Example_FilePicker.py`

Interactive file picker dialog for loading pathway data.

```bash
python examples/visualization/input_formats/Example_FilePicker.py
```

---

### 3D Skeleton Visualization (`visualization/skeleton/`)

#### 1. Simple Demo
**File:** `visualization/skeleton/Example_VisualizeSkeleton_Simple.py`

Simple demo using environment variables and ROI listing.

```bash
python examples/visualization/skeleton/Example_VisualizeSkeleton_Simple.py
```

#### 2. Quick Demo
**File:** `visualization/skeleton/Example_VisualizeSkeleton_QuickDemo.py`

Quick demo of cache directory structure and mesh discovery (no token required).

```bash
python examples/visualization/skeleton/Example_VisualizeSkeleton_QuickDemo.py
```

#### 3. Multi-Dataset Support
**File:** `visualization/skeleton/Example_VisualizeSkeleton_MultiDataset.py`

Multi-dataset support (hemibrain vs optic-lobe ROI caching).

```bash
python examples/visualization/skeleton/Example_VisualizeSkeleton_MultiDataset.py
```

**Features:** Dataset-specific ROI caching, automatic ROI discovery

#### 4. Optimizations
**File:** `visualization/skeleton/Example_VisualizeSkeleton_Optimizations.py`

Optimization features: ignore_synapses, export_video, performance improvements.

```bash
python examples/visualization/skeleton/Example_VisualizeSkeleton_Optimizations.py
```

#### 5. Comprehensive Tests
**File:** `visualization/skeleton/Example_VisualizeSkeleton_ComprehensiveTests.py`

Full test suite for ROI mesh download, caching, error handling.

```bash
python examples/visualization/skeleton/Example_VisualizeSkeleton_ComprehensiveTests.py
```

---

## 📊 Sample Data (`data/`)

| File | Description |
|------|-------------|
| `example_simple_network.csv` | Basic 3-column edge list |
| `example_bodyid_network.csv` | Network with body IDs |
| `example_neuron_network.csv` | Neuron type format example |
| `example_paths.csv` | Multi-hop pathway example |
| `example_label_mapping.csv` | Label mapping for neuron types |
| `example_label_mapping.json` | Label mapping in JSON format |
| `example_source_mapping.csv` | Source neuron mapping |
| `example_target_mapping.csv` | Target neuron mapping |
| `Example_ConnMatrix_10x12.csv` | Sample 10x12 connection matrix |
| `sample_network_data.csv` | Generic sample network data |
| `simple_network_data.csv` | Simple network data |

---

## 🎯 Usage Pattern

All example scripts follow this pattern:

```python
# 1. Import required modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from coana import FindNeuronConnection
from vispath import VisualizePath

# 2. Set up parameters
fc = FindNeuronConnection(
    token='your_token_here',
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['KC.*'],
    targetNeurons=['MBON.*'],
    use_cache=True
)

# 3. Run analysis
fc.InitializeNeuronInfo()
fc.FindAllPath()

# 4. Visualize results
vis = VisualizePath(path_file='results.xlsx')
vis.visualize()
```

> **Note:** For files in subdirectories, adjust the path insertion accordingly (e.g., `parent.parent.parent` for files two levels deep).

---

## 🔗 Related Resources

### Quick Start Guides
- [Installation Guide](../docs/INSTALLATION.md)
- [Quick Reference](../docs/QUICK_REFERENCE.md)

### Core Feature Documentation
- [Cache System Guide](../docs/CacheSystem_Guide.md)
- [Path Finding](../docs/FindAllPath_Documentation.md)
- [Visualization](../docs/VisualizeSelectedPaths_Guide.md)

### Visualization Guides
- [VisualizeSkeleton Updates](../docs/visualizations/VisualizeSkeleton_Updates_Nov2024.md)
- [VisualizePath Updates](../docs/visualizations/VisualizePath_Updates_Nov2025.md)
- [Custom Colors Guide](../docs/CUSTOM_COLORS_GUIDE.md)

---

## 💡 Tips

1. **Get a NeuPrint Token:** Visit [https://neuprint.janelia.org/account](https://neuprint.janelia.org/account)
2. **Start small:** Use limited source/target neurons for testing
3. **Enable caching:** `use_cache=True` for significant speedup
4. **Check sample data:** `data/` folder has ready-to-use examples

---

## 🆘 Troubleshooting

**Import errors?**
- Make sure you're in the project root when running examples
- Check dependencies: `pip install -r requirements.txt`
- For nested examples, adjust `sys.path.insert()` depth

**Slow performance?**
- Enable caching: `use_cache=True`
- Reduce dataset size for testing

**File picker not working?**
- Install PyQt5: `pip install PyQt5`
- See [DIALOG_PERFORMANCE_GUIDE.md](../docs/DIALOG_PERFORMANCE_GUIDE.md)

---

**Need more help?** Check the [main documentation](../README.md) or see detailed guides in [docs/](../docs/).
