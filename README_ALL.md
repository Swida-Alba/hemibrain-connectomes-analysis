# Hemibrain Connectomes Analysis - Complete Documentation

**Version 2.1 | Complete Reference Guide**

This document contains the complete documentation for the Hemibrain Connectomes Analysis toolkit in a single file for easy HTML rendering.

---

**Table of Contents**

1. [Overview & Key Features](#overview--key-features)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Core Features](#core-features)
5. [Visualization Guides](#visualization-guides)
6. [Technical Documentation](#technical-documentation)
7. [Examples & Tutorials](#examples--tutorials)

---

# Overview & Key Features

A comprehensive Python toolkit for analyzing and visualizing *Drosophila* hemibrain connectome data from NeuPrint. Find direct and indirect neural connections, visualize pathways, and perform network analysis with high-performance caching and parallel processing.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Key Features

- 🔍 **Connection Discovery**: Find direct and multi-hop paths between neuron populations
- ⚡ **High Performance**: 10-100x speedup with local caching, 4-14x with parallel processing
- 📊 **Rich Visualizations**: Interactive networks, Sankey diagrams, heatmaps, and 3D skeletons
- 🎯 **Flexible Filtering**: Multiple filtering modes (synapse count, connection ratio, traversal probability)
- 💾 **Smart Caching**: Efficient local storage with automatic complete dataset handling
- 🔧 **Modular Design**: Reorganized src/ layout for better maintainability

---

# Installation

## Quick Install (Recommended)

```bash
pip install -r requirements.txt
```

This will install all necessary dependencies including:
- Core packages: numpy, pandas, scipy
- Visualization: plotly, matplotlib, seaborn, networkx
- Data processing: openpyxl, xlrd
- **PyQt5**: For fast, responsive GUI file dialogs (recommended)
- Neuroscience: neuprint-python, navis

## Step-by-Step Guide for Beginners

### Step 1: Install Python

**Download Python 3.11.3** (or any version 3.8-3.11):
- Visit [Python Downloads](https://www.python.org/downloads/)
- Download the installer for your operating system:
  - **Windows**: Windows installer (64-bit)
  - **macOS**: macOS installer (Intel or Apple Silicon)
  - **Linux**: Usually pre-installed, or use your package manager

**Important for Windows users:**
- ✅ Check "Add Python to PATH" during installation
- ✅ Check "Install pip" (usually checked by default)

**Verify installation:**
```bash
# Open Terminal (Mac/Linux) or Command Prompt (Windows)
python --version
# Should show: Python 3.11.3 (or your installed version)

pip --version
# Should show: pip 23.x.x or similar
```

### Step 2: Download This Repository

**Option A: Download as ZIP (easier for beginners)**
1. Click the green "Code" button at the top of this page
2. Select "Download ZIP"
3. Extract the ZIP file to a location you can remember (e.g., `Documents/hemibrain-analysis`)

**Option B: Using Git (recommended for developers)**
```bash
git clone https://github.com/Swida-Alba/hemibrain-connectomes-analysis.git
cd hemibrain-connectomes-analysis
```

### Step 3: Install Dependencies

**Open Terminal/Command Prompt** in the project folder:
- **Windows**: Right-click in the folder → "Open in Terminal"
- **macOS**: Right-click in the folder → "New Terminal at Folder"
- **Linux**: Right-click in the folder → "Open in Terminal"

**Install all required packages:**
```bash
# Upgrade pip first (recommended)
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

This will install ~20 packages including:
- Data processing: numpy, pandas, scipy
- Visualization: plotly, matplotlib, networkx
- Neuroscience: neuprint-python, navis
- GUI dialogs: PyQt5 (for faster file pickers)

**Installation time:** Usually 2-5 minutes depending on your internet speed.

### Step 4: Set Up VS Code (Optional but Recommended)

**Download and install VS Code:**
- Visit [Visual Studio Code](https://code.visualstudio.com/)
- Download and install for your operating system

**Install Python extension:**
1. Open VS Code
2. Click the Extensions icon (square icon on left sidebar)
3. Search for "Python" (by Microsoft)
4. Click "Install"

**Configure VS Code settings (recommended):**
1. Press `Ctrl+,` (Windows/Linux) or `Cmd+,` (Mac) to open Settings
2. Search for "Auto Save" → Select "onFocusChange"
3. Search for "Execute In File Dir" → Check the box

**Select Python interpreter:**
1. Open any `.py` file in the project
2. Click the Python version in the bottom-right corner
3. Select your installed Python version (e.g., Python 3.11.3)

### Step 5: Get Your NeuPrint Token

**Register and get authentication token:**
1. Visit [NeuPrint](https://neuprint.janelia.org/)
2. Click "LOGIN" (top-right corner)
3. Log in with your Google account
4. Click "Account" → Copy your Auth Token

**Important:** Keep your token private! Don't share it or commit it to version control.

### Step 6: Configure Your First Script

**Edit `scripts/FindDirect.py` or `scripts/PlotPath.py`:**

1. Open the script in VS Code or any text editor
2. Find the line with `token=''`
3. Paste your NeuPrint token between the quotes:

```python
fc = FindNeuronConnection(
    token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...',  # Your actual token here
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['KC.*'],
    targetNeurons=['MBON03'],
    # ... other parameters
)
```

### Step 7: Run Your First Analysis

**In VS Code:**
1. Open `scripts/FindDirect.py`
2. Press `F5` or click the "Run" button (▶️) in the top-right
3. Check the output in the Terminal panel

**In Terminal/Command Prompt:**
```bash
# Navigate to the scripts folder
cd scripts

# Run the script
python FindDirect.py
```

**Expected output:**
- Progress messages showing data fetching
- Connection data saved to `connection_data/` folder
- HTML visualizations automatically opened in your browser

### Step 8: Explore Functionality

**Hover over function names** in VS Code to see documentation and parameter descriptions.

**Try different scripts:**
- `scripts/FindDirect.py` - Find direct connections between neuron types
- `scripts/FindPath.py` - Find multi-hop pathways (2-3 layers)
- `scripts/PlotPath.py` - Visualize pathways with custom colors
- `scripts/plot3dSkeleton.py` - Render 3D neuron morphology

**Modify parameters:**
- Change `sourceNeurons` and `targetNeurons` to analyze different connections
- Adjust `min_synapse_num` to filter weak connections
- Try different `network_layout` options: 'hierarchical', 'spring', 'circular'

## Troubleshooting

**Problem: "python: command not found"**
- Solution: Python is not in PATH. Reinstall Python with "Add to PATH" checked.

**Problem: "pip: command not found"**
- Solution: Try `python -m pip install -r requirements.txt` instead.

**Problem: "ModuleNotFoundError: No module named 'X'"**
- Solution: Run `pip install -r requirements.txt` again.

**Problem: Installation errors with PyQt5**
- Solution: Skip PyQt5 (optional), use tkinter instead:
  ```bash
  pip install -r requirements.txt --no-deps
  pip install numpy pandas scipy plotly matplotlib networkx neuprint-python navis
  ```

**Problem: "SSL Certificate Error" during pip install**
- Solution: Temporarily disable VPN/proxy or use:
  ```bash
  pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org
  ```

**Problem: Script runs but no output files**
- Solution: Check the `connection_data/` folder. File paths are printed in the terminal.

---

# Basic Usage

## FindDirect.py - Finding Direct Connections

This script is located in `scripts/FindDirect.py` and is used for finding direct connections between neuron clusters.

### Basic Configuration

```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(
    token='your_neuprint_token',
    dataset='hemibrain:v1.2.1',
    data_folder='./connectome_data',
    sourceNeurons=['KC.*'], 
    targetNeurons=['MBON03'], 
    custom_source_name='', 
    custom_target_name='',
    min_synapse_num=1,
    min_traversal_probability=0.001,
    showfig=False,
)

# Initialize and find connections
fc.InitializeNeuronInfo()
fc.FindDirectConnection()
```

### Available Datasets

```python
# All available datasets:
'fib19:v1.0'
'hemibrain:v0.9'
'hemibrain:v1.0.1'
'hemibrain:v1.1'
'hemibrain:v1.2.1'  # Default
'manc:v1.0'
```

### Specifying Neurons

Source neurons and target neurons can be specified as:
- **bodyId**: Exact neuron identifier (e.g., `123456789`)
- **type**: Neuron type name (e.g., `'KC_alpha'`, `'MBON03'`)
- **instance**: Use regular expressions (e.g., `'KC.*'`, `'.*_R'` for right hemisphere)

```python
# Example: Using regular expressions
fc = FindNeuronConnection(
    sourceNeurons=['KC.*'],           # All KC neurons
    targetNeurons=['MBON.*'],         # All MBON neurons
)

# Example: From external file
import pandas as pd
source_list = pd.read_excel('sourceNeurons.xlsx', header=None).iloc[:,0].tolist()
target_list = pd.read_csv('targetNeurons.csv', header=None).iloc[:,0].tolist()

fc = FindNeuronConnection(
    sourceNeurons=source_list,
    targetNeurons=target_list,
)
```

### Filtering Options

#### Three Complementary Filters:

1. **min_synapse_num**: Absolute synapse count
```python
fc = FindNeuronConnection(
    min_synapse_num=10,  # Keep connections with ≥10 synapses
)
```

2. **min_ratio**: Direct connection ratio (proportion of downstream inputs)
```python
fc = FindNeuronConnection(
    min_ratio=0.01,  # Keep connections that are ≥1% of downstream neuron's inputs
)
```

3. **min_traversal_probability**: Scaled probability
```python
fc = FindNeuronConnection(
    min_traversal_probability=0.001,  # Keep connections with ≥0.1% probability
)
```

**Note:** All three filters can be used together. Output includes all three metrics: `weight`, `connection_ratio`, and `traversal_probability`.

## FindPath.py - Finding Multi-Hop Pathways

Use this function to find direct and indirect connection paths between neuron clusters.

```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(
    token='your_neuprint_token',
    dataset='hemibrain:v1.2.1',
    data_folder='./connectome_data',
    sourceNeurons=['.*_.*PN.*'],     # All projection neurons
    targetNeurons=['MBON.*'],        # All mushroom body output neurons
    min_synapse_num=1,
    min_traversal_probability=0.001,
    showfig=False,
    max_interlayer=2,                # Maximum intermediate layers
    keyword_in_path_to_remove=['None'],  # Remove paths with unlabeled neurons
)

# Initialize and find all paths
fc.InitializeNeuronInfo()
fc.FindAllPath()
```

### Path Finding Parameters

**max_interlayer**: Maximum number of intermediate layers
```python
max_interlayer=1  # Source → Target (direct only)
max_interlayer=2  # Source → Intermediate → Target
max_interlayer=3  # Source → Int1 → Int2 → Target
```

**keyword_in_path_to_remove**: Filter out unwanted paths
```python
# Remove paths containing unlabeled neurons
keyword_in_path_to_remove=['None']

# Remove paths through specific neurons
keyword_in_path_to_remove=['APL', 'None']

# Remove paths through multiple neuron types
keyword_in_path_to_remove=['APL', 'None', 'ExR.*']
```

---

# Core Features

## Cache System v4.0

To avoid repeated API calls and speed up analysis, the toolkit includes a **local caching system** that stores fetched connection data.

### Enabling Cache

Simply set `use_cache=True` when creating a `FindNeuronConnection` instance:

```python
fc = FindNeuronConnection(
    token='your_token',
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['KC.*'],
    targetNeurons=['MBON.*'],
    min_synapse_num=10,
    use_cache=True  # Enable caching
)
```

### Key Features

✅ **Automatic Caching**: Connection data is automatically saved to local disk  
✅ **Smart Reuse**: Cached data is reused for different `min_synapse_num` values  
✅ **Neuron Registry**: Search cached neurons without API calls  
✅ **Database Architecture**: Efficient parquet-based storage with query indexing  
✅ **Complete Dataset**: Automatically downloads all neurons (including type=None) for cache enrichment  

### Cache Benefits

- **10-100x faster** for repeated queries
- **Offline analysis** once data is cached
- **Flexible filtering** - fetch once with `min_weight=1`, filter locally for any threshold
- **Neuron search** - find cached neurons by type/instance/bodyId
- **40-60% smaller cache** - stores only bodyId pairs, joins neuron metadata from local files
- **Complete coverage** - handles all neurons including those without types

### First Use (One-Time Setup)

When you enable caching for the first time on a dataset, the system automatically downloads a complete neuron dataset:

```
📥 Complete dataset not found, downloading ALL neurons (including type=None)...
   This is a one-time download for cache enrichment.
Pulled 53847 neurons from hemibrain:v1.2.1
✅ Complete dataset saved
```

This ensures cache can enrich all connections, even those involving untyped neurons. Takes ~15-30 seconds depending on dataset size.

### Cache Management

**View cache information:**
```python
fc.print_cache_info()
```

**Clear cache:**
```python
fc.clear_cache()
```

**Search cached neurons:**
```python
# Search by neuron type
kc_neurons = fc.search_cached_neurons('KC.*', 'type')

# Search by instance (e.g., right hemisphere)
right_neurons = fc.search_cached_neurons('.*_R$', 'instance')
```

## Parallel Processing

For large datasets with many source-target neuron pairs, pathfinding can be accelerated using parallel processing:

```python
# Enable parallel processing (recommended for >100 source-target pairs)
fc = FindNeuronConnection(
    token='your_token_here',
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['PPL1-01', 'PPL1-02', 'PPL1-03'],
    targetNeurons=['MBON14', 'MBON11', 'MBON01'],
    max_interlayer=3,
    use_parallel=True,    # Enable parallel processing
    n_jobs=-1             # Use all CPU cores (-1 = auto-detect)
)

fc.InitializeNeuronInfo()
fc.FindAllPath()  # Will automatically use parallel processing if beneficial
```

### Performance Benefits

- **4-core CPU**: 3-4x faster for large datasets
- **8-core CPU**: 6-8x faster for large datasets
- **16+ core CPU**: 10-14x faster for large datasets

### Parameters

- `use_parallel`: Enable/disable parallel processing (default: `False`)
- `n_jobs`: Number of processes to use
  - `-1`: Use all CPU cores (recommended)
  - `1`: Sequential processing
  - `N`: Use exactly N processes

### Automatic Optimization

- Datasets with <100 pairs: Uses sequential processing (overhead not worth it)
- Datasets with >100 pairs: Uses parallel processing (significant speedup)

---

# Visualization Guides

## 1. Heatmap Visualization

Interactive connection matrix heatmaps for visualizing neuron-to-neuron connectivity patterns.

### Overview

The heatmap visualization displays connectivity data as a color-coded matrix where:
- **Rows** represent source neurons (presynaptic)
- **Columns** represent target neurons (postsynaptic)
- **Cell colors** represent connection strength (synapse count, ratio, or probability)
- **Interactive controls** allow real-time exploration and customization

### Example Input Data

Your connection dataframe should contain these columns:

| Column | Description | Example Values |
|--------|-------------|----------------|
| `source` or `sourcetype` | Source neuron names/types | 'KC_alpha', 'MBON01', 'PPL1-01' |
| `target` or `targettype` | Target neuron names/types | 'MBON03', 'MBON14', 'DAN' |
| `weight` | Synapse count (required) | 10, 25, 150 |
| `ratio` | Connection ratio (optional) | 0.05, 0.15, 0.8 |
| `probability` | Traversal probability (optional) | 0.1, 0.5, 0.9 |

### Quick Start

```python
from statvis import plot_stat

# Create heatmap from connection data
plot_stat(
    conn_df=connection_dataframe,
    output_folder='./output',
    metric='weight',           # 'weight', 'ratio', or 'probability'
    custom_order=None,         # Optional: custom row/column ordering
    showfig=True               # Open in browser automatically
)
```

### Key Features

#### 1. Multiple Scales

Transform data for better visualization:
- **Linear**: Original values (default)
- **Logarithmic (log₂)**: Emphasize smaller differences
- **Logarithmic (log₁₀)**: Alternative log base
- **Square Root**: Moderate compression of large values

**Usage**: Click scale buttons at the top of the interface

#### 2. Hierarchical Clustering

Automatically group similar connection patterns:
- **Ward Linkage** (default): Creates compact, well-separated clusters
- **Average Linkage**: Balanced approach
- **Complete Linkage**: Produces tight, compact clusters
- **Single Linkage**: Can reveal hierarchical structures

**Usage**: 
1. Click "Clustered Order" button
2. Select clustering method from dropdown
3. Toggle back to "Original Order" anytime

#### 3. Custom Colorscales

Choose from 15+ built-in colorscales:
- **Sequential**: Blues, Greens, Reds, Purples, Oranges, Greys
- **Diverging**: RdBu, PiYG
- **Perceptually uniform**: Viridis, Plasma, Inferno, Magma, Cividis
- **Specialized**: Turbo, Jet

**Custom 2-Color Scale**:
1. Select "Custom" from colorscale dropdown
2. Choose minimum color (e.g., white for low values)
3. Choose maximum color (e.g., dark purple for high values)

### Example Files

- `examples/sample_network_data.csv` - Sample connection data
- `datasets/hemibrain_v1_2_1_alltypes_neuron_df.csv` - All neuron types with metadata
- Output from `scripts/FindDirect.py` - Direct connections

---

## 2. Network Visualization

Interactive network graph visualization for exploring neural pathways and connectivity structures.

### Overview

The network visualization displays connectivity as an interactive graph where:
- **Nodes** represent neurons (colored by role: source, intermediate, target)
- **Edges** represent connections (thickness = synapse count, ratio, or probability)
- **Layout algorithms** arrange nodes to reveal structure
- **Interactive controls** enable exploration, editing, and customization

### Example Input Data

Your edge list should contain these columns:

| Column | Description | Example Values | Required |
|--------|-------------|----------------|----------|
| `source`, `from`, or `*_pre` | Source neuron | 'KC_alpha', 'PN1', 'L3_R' | ✅ Yes |
| `target`, `to`, or `*_post` | Target neuron | 'MBON03', 'LN1', 'Mi1_R' | ✅ Yes |
| `weight` or `synapse_count` | Connection strength | 10, 25, 150 | ✅ Yes |
| `ratio` | Connection ratio | 0.05, 0.15, 0.8 | ❌ Optional |
| `probability` | Traversal probability | 0.1, 0.5, 0.9 | ❌ Optional |

### Quick Start

```python
from vispath import VisualizePath

# Create network from path analysis results
vp = VisualizePath(
    path_file='path_results.xlsx',    # Excel, CSV, or DataFrame
    sheet_name=None,                   # Auto-select sheet
    output_folder='./visualizations',
    network_layout='hierarchical',     # Layout algorithm
    showfig=True                       # Open in browser
)

# Generate visualizations
conn_df, G = vp.visualize()
```

### Layout Algorithms

#### Hierarchical (Dagre)
- **Best for**: Layered pathways, feedforward networks
- **Algorithm**: Sugiyama's algorithm with edge crossing minimization
- **Direction**: Top-to-bottom flow
- **Use case**: Source → Intermediate → Target pathways

#### Force-Directed (Spring)
- **Best for**: General networks, community detection
- **Algorithm**: Force-atlas with attraction/repulsion
- **Behavior**: Connected nodes attract, others repel
- **Use case**: Exploring network communities

#### Circular
- **Best for**: Small networks, showing all connections
- **Algorithm**: Evenly spaced around circle
- **Benefit**: No overlapping nodes
- **Use case**: Dense connectivity, comparison view

#### Distributed
- **Best for**: Multiple disconnected components
- **Algorithm**: Separates components, uses force-directed within
- **Benefit**: Prevents component overlap

### Interactive Features

- **Drag nodes**: Rearrange manually
- **Hide nodes**: Right-click → Hide
- **Hover**: View connection details
- **Zoom/Pan**: Mouse wheel and drag
- **Export layout**: Save custom arrangements
- **Edit mode**: Add/remove nodes and edges

### Example Files

- `examples/example_neuron_network.csv` - Multi-layer network
- `examples/example_bodyid_network.csv` - BodyId-level network
- `examples/simple_network_data.csv` - Minimal 3-column format
- Output from `scripts/FindPath.py` - Path analysis results

---

## 3. Sankey Diagram

Flow-based visualization showing the magnitude of connections between neuron layers.

### Overview

Sankey diagrams visualize neural pathways as flowing connections where:
- **Nodes** represent neurons or neuron types
- **Links** are connections with width proportional to strength
- **Flow direction** shows signal progression (left to right)
- **Colors** distinguish source, intermediate, and target populations

### Example Input Data

Your edge list should contain these columns:

| Column | Description | Example Values | Required |
|--------|-------------|----------------|----------|
| `source` or `from` | Source neuron | 'KC_alpha', 'MBON01' | ✅ Yes |
| `target` or `to` | Target neuron | 'MBON03', 'DAN_a' | ✅ Yes |
| `weight` | Connection strength | 10, 25, 150 | ✅ Yes |
| `ratio` | Connection ratio | 0.05, 0.15, 0.8 | ❌ Optional |
| `probability` | Traversal probability | 0.1, 0.5, 0.9 | ❌ Optional |

### Quick Start

```python
from vispath import VisualizePath

# Create Sankey diagram from path data
vp = VisualizePath(
    path_file='path_results.xlsx',
    output_folder='./visualizations',
    showfig=True
)

conn_df, G = vp.visualize()
```

The Sankey diagram is automatically generated as `sankey_*.html` alongside the network visualization.

### Key Features

#### 1. Connection Metrics

Toggle between different strength measures:
- **Synapse Count (weight)**: Total synapses in connection
- **Connection Ratio**: Proportion of source neuron's output
- **Traversal Probability**: Likelihood of signal propagation

**Usage**: Click metric buttons at top of interface

#### 2. Layout Modes

**Snap Layout (Default)**:
- Nodes snap to vertical layers
- Best for clearly layered pathways
- Automatic vertical positioning

**Freeform Layout**:
- Nodes can be positioned anywhere
- Best for custom arrangements
- Full manual positioning

#### 3. Node Ordering

Control vertical order within layers:
- **Automatic Ordering**: Sorts by node name (alphabetical)
- **Manual Ordering**: Drag nodes vertically
- **Custom Ordering**: Provide ordering via API

### Interactive Features

- **Drag nodes**: Rearrange vertically (snap mode) or freely (freeform mode)
- **Hover**: View connection details
- **Click links**: Highlight specific pathways
- **Toggle metrics**: Switch between weight/ratio/probability
- **Export**: Save diagram as PNG or SVG

### Example Files

- `examples/example_neuron_network.csv` - Multi-layer pathways
- `examples/simple_network_data.csv` - Basic format
- Output from `scripts/PlotPath.py` - Visualized pathways

---

## 4. 3D Skeleton Visualization

Three-dimensional rendering of neuron morphologies and spatial connectivity in the Drosophila brain.

### Overview

The 3D skeleton visualization displays:
- **Neuron morphologies** as 3D skeletons with accurate spatial coordinates
- **Brain regions** (ROIs) as semi-transparent meshes
- **Connections** between neurons with spatial context
- **Interactive exploration** with rotation, zoom, and selection

### Example Input Data

3D skeleton visualization requires **neuron bodyIds** from NeuPrint:

| Input Type | Description | Example | Source |
|------------|-------------|---------|--------|
| **bodyId** | Unique neuron identifier | 123456789, 987654321 | NeuPrint database |
| **Neuron types** | Type name (converted to bodyIds) | 'KC_alpha', 'MBON03' | Fetched via NeuPrint API |
| **Layer specification** | Multi-layer pathway | ['KC.*', 'MBON.*', 'DAN.*'] | From path analysis |

### Quick Start

```python
from navis_related import plot_navis_3d

# Visualize neurons with 3D skeletons
plot_navis_3d(
    bodyIds=[123456789, 987654321],  # List of neuron bodyIds
    dataset='hemibrain:v1.2.1',
    token='your_neuprint_token',
    show_rois=True,                   # Display brain regions
    show_connectors=True,             # Show synapse locations
    output_folder='./3d_output',
    showfig=True
)
```

### Prerequisites

**Required packages**:
```bash
pip install navis
pip install navis-flybrains
pip install plotly
```

**Data requirements**:
- Valid NeuPrint authentication token
- Neuron bodyIds from chosen dataset
- Internet connection (first time, to fetch skeleton data)

### Key Features

#### 1. Neuron Skeleton Rendering

**Display multiple neurons together**:
```python
plot_navis_3d(
    bodyIds=[
        123456789,  # Neuron 1
        987654321,  # Neuron 2
        111222333   # Neuron 3
    ],
    color_by='neuron'  # Color each neuron differently
)
```

#### 2. Brain Region (ROI) Meshes

**Show specific ROIs**:
```python
plot_navis_3d(
    bodyIds=[123456789],
    show_rois=True,
    roi_list=['MB_CA', 'MB_PED', 'SLP_R'],  # Mushroom body regions
    roi_opacity=0.2  # Semi-transparent
)
```

**Available ROIs**:
- **Mushroom Body**: MB_CA (calyx), MB_PED (pedunculus), MB_VL (vertical lobe)
- **Central Complex**: FB (fan-shaped body), EB (ellipsoid body), PB (protocerebral bridge)
- **Optic Lobes**: ME (medulla), LO (lobula), LOP (lobula plate)
- **Antennal Lobe**: AL_R, AL_L
- **Lateral Horn**: LH_R, LH_L

#### 3. Synapse Visualization

Show synaptic connections in 3D space:
- **Presynaptic sites** (outputs): Red (default)
- **Postsynaptic sites** (inputs): Blue (default)
- **Size**: Proportional to partner count
- **Location**: Exact spatial coordinates

### Getting BodyIds

**From neuron types**:
```python
from neuprint import fetch_neurons, Client

client = Client('neuprint.janelia.org', dataset='hemibrain:v1.2.1', token='your_token')
neurons_df, roi_counts = fetch_neurons(NC(type='KC_alpha'))
bodyIds = neurons_df['bodyId'].tolist()
```

**From path analysis results**:
```python
import pandas as pd

# Load path results
paths = pd.read_excel('path_results.xlsx', sheet_name='path_bodyId')

# Extract unique bodyIds
all_bodyIds = set()
for col in paths.columns:
    if 'bodyId' in col or col in ['source', 'target']:
        all_bodyIds.update(paths[col].dropna().unique())
```

### Example Files

- Sample script: `scripts/plot3dSkeleton.py`
- Type-level data: `datasets/hemibrain_v1_2_1_alltypes_neuron_df.csv`
- Path results with bodyIds: Output from `scripts/FindPath.py`

---

# Technical Documentation

## Performance Optimization

### Caching System Architecture

The caching system uses a parquet-based database architecture:

**Storage Format**:
- **Connection cache**: Parquet files with gzip compression
- **Neuron metadata**: CSV files with type/instance information
- **Index files**: Fast lookup tables for query optimization

**Cache Location**:
```
cache/
├── hemibrain_v1_2_1/
│   ├── connections/
│   │   ├── all_connections.parquet  # All cached connections
│   │   └── metadata.json             # Cache metadata
│   └── neurons/
│       ├── all_neurons.csv           # Complete neuron list
│       └── typed_neurons.csv         # Neurons with assigned types
└── optic-lobe_v1_1/
    └── ...
```

**Query Optimization**:
- Indexed by bodyId pairs for O(1) lookup
- Pre-filtered by minimum weight threshold
- Lazy loading of neuron metadata

### Parallel Processing Architecture

The parallel processing system uses multiprocessing with process pools:

**Architecture**:
```
Main Process
├── Task Queue (source-target pairs)
├── Process Pool (N workers)
│   ├── Worker 1: Process pair batch 1
│   ├── Worker 2: Process pair batch 2
│   └── Worker N: Process pair batch N
└── Result Aggregation
```

**Optimization Strategies**:
- **Batch processing**: Groups pairs to minimize overhead
- **Load balancing**: Distributes work evenly across cores
- **Memory management**: Each worker has isolated memory
- **Progress tracking**: Real-time progress updates

**When to Use**:
- ✅ >100 source-target pairs
- ✅ max_interlayer ≥ 2
- ✅ Multi-core CPU available
- ❌ Small datasets (<100 pairs)
- ❌ Single-core systems

## Data Formats

### Connection Data Format

Output CSV/Excel files contain:

| Column | Type | Description |
|--------|------|-------------|
| `source` | str | Source neuron type/name |
| `target` | str | Target neuron type/name |
| `weight` | int | Synapse count |
| `ratio` | float | Connection ratio (weight/post_inputs) |
| `probability` | float | Traversal probability |
| `source_bodyId` | int | Source neuron bodyId (optional) |
| `target_bodyId` | int | Target neuron bodyId (optional) |

### Path Data Format

Multi-hop pathway files contain:

| Column | Type | Description |
|--------|------|-------------|
| `path_id` | int | Unique path identifier |
| `source` | str | Starting neuron |
| `target` | str | Ending neuron |
| `intermediate_1` | str | First intermediate (if exists) |
| `intermediate_2` | str | Second intermediate (if exists) |
| `total_weight` | int | Sum of all connection weights |
| `path_probability` | float | Product of all connection probabilities |
| `inter_layer_num` | int | Number of intermediate layers |

### Neuron Metadata Format

Neuron information files contain:

| Column | Type | Description |
|--------|------|-------------|
| `bodyId` | int | Unique neuron identifier |
| `type` | str | Neuron type name (or None) |
| `instance` | str | Neuron instance name |
| `status` | str | Tracing status |
| `cropped` | bool | Whether neuron is cropped |
| `soma` | bool | Whether soma is present |

---

# Examples & Tutorials

## Example 1: Simple Direct Connection Analysis

```python
from coana import FindNeuronConnection

# Find direct connections from Kenyon Cells to MBONs
fc = FindNeuronConnection(
    token='your_token',
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['KC.*'],
    targetNeurons=['MBON.*'],
    min_synapse_num=5,
    showfig=True
)

fc.InitializeNeuronInfo()
fc.FindDirectConnection()
```

## Example 2: Multi-Hop Pathway Analysis

```python
from coana import FindNeuronConnection

# Find 2-layer pathways from PNs to MBONs
fc = FindNeuronConnection(
    token='your_token',
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['.*PN.*'],
    targetNeurons=['MBON.*'],
    max_interlayer=2,
    min_synapse_num=10,
    keyword_in_path_to_remove=['None', 'APL'],
    use_cache=True,
    use_parallel=True
)

fc.InitializeNeuronInfo()
fc.FindAllPath()
```

## Example 3: Filtered Path Visualization

```python
from vispath import VisualizePath
import pandas as pd

# Load all paths
all_paths = pd.read_excel('path_results.xlsx', sheet_name='path_type')

# Filter for high-quality paths
high_quality = all_paths[
    (all_paths['traversal_probability'] > 0.5) &
    (all_paths['weight'] > 20) &
    (all_paths['inter_layer_num'] <= 2)
]

# Visualize filtered paths
vp = VisualizePath(
    path_file=high_quality,
    output_folder='./high_quality_paths',
    network_layout='hierarchical',
    source_color='#3498db',
    target_color='#e74c3c',
    showfig=True
)

conn_df, G = vp.visualize()
```

## Example 4: Custom Color Visualization

```python
from vispath import VisualizePath

# Create custom color scheme
vp = VisualizePath(
    path_file='path_results.xlsx',
    output_folder='./custom_viz',
    source_color='#FF6B6B',           # Red source nodes
    intermediate_color='#FFA500',     # Orange intermediate
    target_color='#FFD700',           # Gold target nodes
    link_color='rgba(255,107,107,0.3)', # Semi-transparent red links
    network_layout='spring',
    showfig=True
)

conn_df, G = vp.visualize()
```

## Example 5: 3D Skeleton Visualization

```python
from navis_related import plot_navis_3d

# Visualize specific neurons with ROIs
plot_navis_3d(
    bodyIds=[123456789, 987654321, 111222333],
    dataset='hemibrain:v1.2.1',
    token='your_token',
    show_rois=True,
    roi_list=['MB_CA', 'MB_PED', 'LH_R'],
    roi_opacity=0.2,
    show_connectors=True,
    color_by='neuron',
    output_folder='./3d_output',
    showfig=True
)
```

## Example 6: Batch Processing with Cache

```python
from coana import FindNeuronConnection

# Process multiple neuron pairs efficiently
source_types = ['KC_alpha', 'KC_beta', 'KC_gamma']
target_types = ['MBON01', 'MBON03', 'MBON14']

for source in source_types:
    for target in target_types:
        fc = FindNeuronConnection(
            token='your_token',
            dataset='hemibrain:v1.2.1',
            sourceNeurons=[source],
            targetNeurons=[target],
            use_cache=True,  # Reuse cached connections
            showfig=False
        )
        
        fc.InitializeNeuronInfo()
        fc.FindDirectConnection()
        
        print(f"✓ Completed: {source} → {target}")
```

## Example 7: Heatmap with Custom Ordering

```python
from statvis import plot_stat
import pandas as pd

# Create connection dataframe
conn_df = pd.read_csv('connections.csv')

# Define custom ordering
custom_row_order = ['KC_alpha', 'KC_beta', 'KC_gamma']
custom_col_order = ['MBON01', 'MBON03', 'MBON14']

# Create heatmap with custom order
plot_stat(
    conn_df=conn_df,
    output_folder='./heatmap_output',
    metric='weight',
    custom_order={'rows': custom_row_order, 'cols': custom_col_order},
    showfig=True
)
```

---

# Quick Reference

## Command Cheat Sheet

```bash
# Installation
pip install -r requirements.txt

# Run basic scripts
python scripts/FindDirect.py
python scripts/FindPath.py
python scripts/PlotPath.py
python scripts/plot3dSkeleton.py

# Cache management
python ManageCache.py

# Run tests
python scripts/test_comprehensive.py
```

## Common Parameter Values

### Datasets
- `hemibrain:v1.2.1` (default, most comprehensive)
- `optic-lobe:v1.1` (visual system)
- `manc:v1.0` (ventral nerve cord)

### Neuron Specifications
- **By type**: `'KC_alpha'`, `'MBON03'`, `'PPL1-01'`
- **Regular expression**: `'KC.*'`, `'MBON.*'`, `'.*_R'`
- **By bodyId**: `123456789`, `987654321`

### Filtering Thresholds
- **min_synapse_num**: `1` (all), `5` (weak filter), `10` (moderate), `20+` (strong)
- **min_ratio**: `0.001` (all), `0.01` (1% filter), `0.05` (5% filter)
- **min_traversal_probability**: `0.001` (all), `0.01` (moderate), `0.1` (strong)

### Network Layouts
- `'hierarchical'` - Best for layered pathways
- `'spring'` - Best for general networks
- `'circular'` - Best for small networks
- `'distributed'` - Best for disconnected components

### Color Schemes (RGB Hex)
- **Blue**: `#3498db`, `#2980b9`, `#1f77b4`
- **Red**: `#e74c3c`, `#c0392b`, `#d62728`
- **Green**: `#2ecc71`, `#27ae60`, `#2ca02c`
- **Orange**: `#e67e22`, `#d35400`, `#ff7f0e`
- **Purple**: `#9b59b6`, `#8e44ad`, `#9467bd`

---

# Appendix

## File Structure

```
hemibrain-connectomes-analysis/
├── README.md                    # Main documentation
├── requirements.txt             # Python dependencies
├── setup.py                     # Installation script
├── pyproject.toml              # Project configuration
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── coana.py               # Core analysis functions
│   ├── statvis.py             # Heatmap visualization
│   ├── vispath.py             # Network/Sankey visualization
│   ├── core/                  # Core modules
│   ├── plotting/              # Plotting utilities
│   └── utils/                 # Utility functions
│
├── scripts/                   # Example scripts
│   ├── FindDirect.py          # Find direct connections
│   ├── FindPath.py            # Find multi-hop paths
│   ├── PlotPath.py            # Visualize pathways
│   └── plot3dSkeleton.py      # 3D skeleton rendering
│
├── examples/                  # Example data and notebooks
│   ├── Example_SimpleEdgeList.py
│   ├── Example_Clustering_Demo.py
│   ├── example_paths.csv
│   └── README.md
│
├── docs/                      # Documentation
│   ├── README.md              # Documentation index
│   ├── visualizations/        # Visualization guides
│   ├── core-features/         # Core feature docs
│   ├── technical/             # Technical documentation
│   └── archive/               # Historical documentation
│
├── datasets/                  # Sample datasets
│   ├── hemibrain_v1_2_1_allneurons_neuron_df.csv
│   └── hemibrain_v1_2_1_alltypes_neuron_df.csv
│
└── cache/                     # Cache directory
    ├── hemibrain_v1_2_1/
    └── optic-lobe_v1_1/
```

## License

MIT License - See LICENSE file for details

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{hemibrain_connectomes_analysis,
  title = {Hemibrain Connectomes Analysis},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/Swida-Alba/hemibrain-connectomes-analysis},
  version = {2.1}
}
```

## Contact & Support

- **GitHub Issues**: https://github.com/Swida-Alba/hemibrain-connectomes-analysis/issues
- **Documentation**: https://github.com/Swida-Alba/hemibrain-connectomes-analysis/tree/main/docs
- **NeuPrint Help**: https://neuprint.janelia.org/help

---

**End of Complete Documentation**

*Generated: November 5, 2025*  
*Version: 2.1*  
*Repository: hemibrain-connectomes-analysis*
