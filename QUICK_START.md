# Quick Start Guide

This guide provides a quick overview of how to perform common tasks using the Drosophila Connectome Analysis Toolkit (DROCAT) v4.4.0.

## Prerequisites

1. **Install the toolkit** - See [Installation Guide](docs/INSTALLATION.md)
2. **Set up authentication** - Copy `token_info.txt` to `token_info_local.txt` and add your API tokens:
   ```
   NEUPRINT_TOKEN='your_neuprint_token_here'
   CAVE_TOKEN='your_cave_token_here'
   ```
   
   **Token Requirements:**
   - `NEUPRINT_TOKEN`: **Required** for NeuPrint datasets (hemibrain, male-cns, MANC, optic-lobe)
   - `CAVE_TOKEN`: **Required** for FlyWire datasets (FAFB, BANC)
   - NeuronBridge API requires **no authentication**
   
   📖 **[Authentication Setup](docs/INSTALLATION.md#authentication-setup)**

## 1. Connection Path Finding

Find all paths between source and target neurons up to a specified number of hops (layers).

**Script:** `scripts/FindPath.py`
**Class:** `FindNeuronConnection` (in `src/coana.py`)

```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(
    # Token automatically loaded from token_info.txt
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['L2'],         # Supports bodyId, type, instance with regex
    targetNeurons=['l-LNv'],
    min_synapse_num=3,
    max_interlayer=3,
    data_folder='./output_data'
)

fc.InitializeNeuronInfo()
fc.FindPath()
```

**NEW in v4.4.0**: Priority-based search (bodyId → type → instance), accepts both int and string bodyIds!

## 2. Direct Connections

Find direct connections between a set of neurons (1-hop).

**Script:** `scripts/FindDirect.py`
**Class:** `FindNeuronConnection`

```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(
    # Token automatically loaded from token_info.txt
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['aMe12'],
    targetNeurons=['KCg-d'],
    min_synapse_num=1,
    data_folder='./output_data'
)

fc.InitializeNeuronInfo()
fc.FindDirectConnections()
```

## 2.5. Local FAFB/BANC Analysis (NEW! 10-100x faster!)

Use local dataset files for blazing-fast FlyWire analysis.

```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(
    dataset='flywire_FAFB_v783',  # Automatically uses local cache if available
    sourceNeurons=['CB0038'],      # FlyWire root IDs
    targetNeurons=['LPLC2'],
    min_synapse_num=3,
    use_cache=True,               # Enable local caching
    data_folder='./output_data'
)

fc.InitializeNeuronInfo()
fc.FindAllPath()
```

📖 **[FAFB Integration Guide](docs/FAFB_INTEGRATION.md)** - Complete setup instructions for local datasets

## 3. Network Visualization (Sankey, Heatmap)

Visualize the paths found in step 1 using Sankey diagrams, heatmaps, and network graphs.

**Script:** `scripts/PlotPath.py`
**Class:** `VisualizePath` (in `vispath-subproject/src/vispath_pkg/vispath.py`)

**NEW in v4.4.0**: NT edge groups, custom groups, improved default opacity (50% edges, 100% nodes)!

```python
from vispath_pkg import VisualizePath

# Initialize visualization with path data
vp = VisualizePath(
    path_file='./output_data/allpaths_.../path_data.xlsx', # Path to your FindPath output file
    sheet_name=0, # or 'path_bodyId' or 'path_type'
    output_folder='./visualization_output',
    source_color='#1f77b4',
    intermediate_color='#2ca02c',
    target_color='#d62728',
    network_layout='hierarchical', # 'hierarchical', 'spring', 'circular', 'distributed'
    showfig=True
)

# Generate all visualizations (Sankey, Network, Heatmap)
conn_df, G = vp.visualize()
```

📖 **[Network Features Guide](docs/visualizations/VisualizePath_Network_Features.md)** - NT grouping, custom groups, export/import

## 4. 3D Skeleton Visualization

Visualize neurons and their synapses in 3D space.

**Script:** `scripts/plot3dSkeleton.py`
**Class:** `VisualizeSkeleton` (in `src/coana.py`)

```python
from coana import VisualizeSkeleton

vs = VisualizeSkeleton(
    token='YOUR_NEUPRINT_TOKEN',
    dataset='hemibrain:v1.2.1',
    neuron_layers=['VA1d_adPN', 'LHCENT3', 'MBON01'], # Layers to plot
    min_synapse_num=10,
    skeleton_mode='tube',
    synapse_mode='cone',
    brain_mesh='whole',
    data_folder='./output_data'
)

vs.plot_neurons()
```

## 5. Cross-Dataset Comparison

Compare connectivity pathways across different datasets (e.g., Hemibrain vs. FAFB vs. Male CNS).

**Script:** `scripts/interdataset_comparison.py`
**Classes:** `ComparisonParameters`, `ComparisonAnalyzer` (in `src/comparison/`)

```python
from comparison.comparison_parameters import ComparisonParameters
from comparison.comparison_analyzer import ComparisonAnalyzer

# Define parameters
params = ComparisonParameters(
    datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
    source_neurons=['MBON14.*_R'],
    target_neurons=['KCg-d.*_R'],
    max_interlayer=2,
    thresholds=[1, 3, 5, 10],
    output_folder='./comparison_output'
)

# Run analysis
analyzer = ComparisonAnalyzer(params)
analyzer.run_comparison()
analyzer.generate_report()
```

## 6. Homolog Finding

Find potential homologs of a neuron in another dataset based on connectivity profiles.

**Script:** `scripts/FindHomologs.py`
**Class:** `HomologFinder` (in `src/comparison/profile_comparator.py`)

```python
from comparison.profile_comparator import HomologFinder

finder = HomologFinder(
    token='YOUR_NEUPRINT_TOKEN',
    source='aMe12',
    source_dataset='flywire_BANC_v626',
    target_dataset='flywire_FAFB_v783',
    output_dir='./homolog_results',
    similarity_metric='jaccard', # or 'cosine', 'rank'
    visualize_skeleton=True      # Visualize top candidates
)

# Run fast search
results = finder.find_homologs_fast()
```

For detailed information on the output files generated by these scripts, please refer to [OUTPUT_FILES.md](docs/OUTPUT_FILES.md).
