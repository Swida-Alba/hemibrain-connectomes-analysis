# NeuPrint Connectome Analysis v4.0

A comprehensive Python toolkit for analyzing and visualizing connectome data from **all NeuPrint databases**. Features type-based pathfinding algorithms, interactive network visualizations, 3D neuron morphology rendering with video export, and high-performance caching. Supports hemibrain, optic lobe, FIB, MANC, and other NeuPrint datasets.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Key Features

- 🔍 **Type-Based Pathfinding**: Comprehensive multi-hop path discovery between neuron populations with forward-only validation
- 🎬 **3D Visualization**: Interactive neuron skeleton rendering with rotating video export (navis-based)
- 🌐 **Interactive Networks**: Cytoscape.js-powered network graphs with hierarchical and force-directed layouts
- 🪞 **Contralateral Mirroring**: Automatically mirror neurons and ROIs to the contralateral hemisphere for full-brain visualization
- 📊 **Rich Visualizations**: Sankey diagrams, heatmaps with clustering, and connection matrices
- 🗄️ **Universal Dataset Support**: Works with all NeuPrint datasets (hemibrain, optic-lobe, FIB, MANC, etc.) and **FlyWire FAFB/BANC** datasets.
- ⚡ **High Performance**: 10-100x speedup with local caching, 4-14x with parallel processing
- 🎯 **Flexible Filtering**: Multiple filtering modes (synapse count, connection ratio, traversal probability)
- 💾 **Smart Caching**: Efficient local storage with automatic complete dataset handling
- 🔧 **Modular Design**: Reorganized src/ layout for better maintainability

---

## 📚 Documentation

### 🚀 Getting Started
- **[Installation Guide](#installation-for-users-who-can-prepare-the-python-environments-by-themselves)** - Setup and dependencies
- **[FlyWire FAFB Integration](docs/FAFB_INTEGRATION.md)** - Guide for setting up and using local FAFB datasets (Data Prep & Download)
- **[FlyWire BANC Integration](docs/BANC_INTEGRATION.md)** - Guide for setting up and using local BANC datasets (Data Prep & Download)
- **[FlyWire Usage](docs/FLYWIRE_USAGE.md)** - Guide for using FlyWire/FAFB/BANC datasets (Local File based)
- **[Basic Usage](#basic-functions)** - FindDirect.py and FindPath.py tutorials
- **[Quick Start After Reorganization](docs/QUICK_START_AFTER_REORGANIZATION.md)** - Get started with v3.0 structure
- **[Performance Optimization](#performance-optimization)** - Caching and parallel processing

### 📖 Core Documentation
For comprehensive documentation, see **[docs/README.md](docs/README.md)** - your central navigation hub for all documentation.

### 🔑 Core Features
Detailed documentation in **[docs/core-features/](docs/core-features/)** including:
- **✨ NEW: [Cross-Dataset Comparison Guide](docs/core-features/CrossDatasetComparison_Guide.md)** - Compare connectivity across hemibrain, male-cns, FlyWire, and more
- **Cache System** - 10-100x faster queries with intelligent local storage
- **Path Finding** - Graph-based algorithms for multi-hop connections
- **Parallel Processing** - 4-14x speedup with multi-core execution
- **Filtering** - Multiple filtering modes and criteria
- See **[Core Features Overview](docs/core-features/README.md)** for complete list

### 📊 Visualizations
Comprehensive guides in **[docs/visualizations/](docs/visualizations/)** including:
- **[Heatmap Guide](docs/visualizations/Heatmap_Guide.md)** - Connection matrices with hierarchical clustering
- **[Network Guide](docs/visualizations/Network_Guide.md)** - Interactive Cytoscape.js networks
- **[Sankey Guide](docs/visualizations/Sankey_Guide.md)** - Flow-based pathway diagrams
- **[3D Skeleton Guide](docs/visualizations/3D_Skeleton_Guide.md)** - Neuron morphology rendering
- **✨ NEW: [VisualizeSkeleton Multi-Dataset Support (Nov 2024)](docs/visualizations/VisualizeSkeleton_Updates_Nov2024.md)** - Dataset-specific ROI mesh caching, automatic ROI discovery, and brain transformation confirmation
- **✨ NEW: [VisualizePath Updates (Nov 2024)](docs/visualizations/VisualizePath_Updates_Nov2025.md)** - Connection matrix input and individual visualization control
- See **[Visualizations Overview](docs/visualizations/README.md)** for comparison table

### 🔧 Technical Documentation
Developer resources in **[docs/technical/](docs/technical/)** including:
- Performance optimization and profiling
- Data format specifications
- Backend architecture details
- Debugging guides
- See **[Technical Overview](docs/technical/README.md)** for complete list

### 📜 Historical Archive
Previous updates and bug fixes in **[docs/archive/](docs/archive/)** including reorganization summaries, bug fix documentation, and deprecated guides

---

## Basic functions

### FindDirect.py

This script is located in `scripts/FindDirect.py` and is used for finding direct connections between neuron clusters.

**Usage:**
```bash
python scripts/FindDirect.py
```

**Configuration:**
```python
fc = FindNeuronConnection(
    token='',
    dataset = 'hemibrain:v1.2.1',
    data_folder=R'D:\connectome_data',
    sourceNeurons = ['KC.*'], 
    targetNeurons = ['MBON03'], 
    custom_source_name = '', 
    custom_target_name = '',
    min_synapse_num = 1,
    min_traversal_probability = 0.001,
    showfig = False,
)
```

in the main codes, we call the ```FindNeuronConnection``` class at first. In the class, you should input your own token obtained from [Neuprint Account Page](https://neuprint.janelia.org/account).

```python
fc = FindNeuronConnection(
    ... # other parameters
    token = 'Your Auth Token',
    ... # other parameters
)
```

And you can specify the ```dataset``` to use (default is ```"hemibrain:v1.2.1"```).

```python
fc = FindNeuronConnection(
    ... # other parameters
    dataset = 'hemibrain:v1.2.1',
    ... # other parameters
)

# All available datasets are listed below:
'''
'fib19:v1.0', 
'hemibrain:v0.9', 
'hemibrain:v1.0.1', 
'hemibrain:v1.1', 
'hemibrain:v1.2.1', 
'manc:v1.0'
'''
```

If ```data_folder``` was not specified, all fetched data will be saved in the "connection_data" folder in the current directory. We highly recommand you to specify the ```data_folder``` to save all data in a specific directory. Then each time you run the codes, the data will be saved in a new folder with auto-generated name in the specified ```data_folder``` directory.

```python
fc = FindNeuronConnection(
    ... # other parameters
    data_folder = R'D:\connectome_data',
    ... # other parameters
)

```

Alternatively, you can specify the save_folder to save the current data in a specific directory for this time only.

```python
fc = FindNeuronConnection(
    ... # other parameters
    save_folder = R'D:\connectome_data\current_data',
    ... # other parameters
)
```

Source neurons and target neurons should be specified as ```bodyId```, ```type```, or ```instance``` (use regular expression to search for instances matching the regular expression). See details in the docstrings by hanging your cursor over the parameter name.

```python
fc = FindNeuronConnection(
    ... # other parameters
    sourceNeurons = ['KC.*'],
    targetNeurons = ['MBON03'],
    # sourceNeurons = pd.read_excel('sourceNeurons.xlsx', header=None).iloc[:,0].tolist(),
    # targetNeurons = pd.read_excel('targetNeurons.xlsx', header=None).iloc[:,0].tolist(),
    ... # other parameters
)


# sourceNeurons and targetNeurons can also be read from other files, e.g. xlsx, csv, txt, etc.
# when reading xlsx and csv files, you can use:
import pandas as pd
neuron_list_1 = pd.read_excel('sourceNeurons.xlsx', header=None).iloc[:,0].tolist()
neuron_list_2 = pd.read_csv('sourceNeurons.csv', header=None).iloc[:,0].tolist()
# to read the first column of the file as a list of bodyIds, types, or instances, without the header.

#### Handling Symmetric Datasets (Left/Right Separation)

For datasets like `male-cns:v0.9` that contain symmetric structures but are NeuPrint-only (no separate left/right instances by default), you can use nested lists with regex patterns to explicitly separate left and right hemisphere neurons.

```python
# Example: Separating Left and Right neurons for symmetric analysis
# Use nested lists to group neurons by hemisphere
neurons_network = [
    ['aMe12.*_R'],   # Right hemisphere aMe12
    ['aMe12.*_L'],   # Left hemisphere aMe12
    ['KCg-s1.*_R'],  # Right hemisphere KCg-s1
    ['KCg-s1.*_L']   # Left hemisphere KCg-s1
]

fc = FindNeuronConnection(
    ... # other parameters
    dataset = 'male-cns:v0.9',
    sourceNeurons = neurons_network,
    targetNeurons = neurons_network,
    ... # other parameters
)
```
```

If your source or target neurons are too many items in a list, you can specify a custom name for them.

```python
fc = FindNeuronConnection(
    ... # other parameters
    custom_source_name = 'all_KCs',
    custom_target_name = 'my_MBON',
    ... # other parameters
)
```

In the ```min_synapse_num``` parameter, you can specify the minimum number of synapses between each pair of the connected neurons.

```python
fc = FindNeuronConnection(
    ... # other parameters
    min_synapse_num = 10,
    ... # other parameters
)
```

In the ```min_traversal_probability``` parameter, you can specify the minimum traversal probability for each connection (edge-level filter). This probability is calculated by the number of synapses between each pair of connected neurons divided by the (30% total number) of input synapses of the downstream neuron. $p = min(1, w_{ij} / (W_j \times 0.3))$, where $w_{ij}$ is the number of synapses between neuron $i$ and $j$, and $W_j$ is the total number of input synapses of neuron $j$.

**Note:** This filter is applied to **each individual connection** (like `min_synapse_num`), not to entire paths. Connections with probability below this threshold are excluded from the network before pathfinding begins.

```python
fc = FindNeuronConnection(
    ... # other parameters
    min_traversal_probability = 1e-3, # 0.001
    ... # other parameters
)
```

In the ```min_ratio``` parameter, you can specify the minimum connection ratio (direct proportion without scaling) for filtering connections. Connection ratio is calculated as `weight / post`, representing the raw fraction of downstream neuron's inputs from the upstream neuron.

```python
fc = FindNeuronConnection(
    ... # other parameters
    min_ratio = 0.01,  # Keep connections that are ≥1% of downstream neuron's inputs
    ... # other parameters
)
```

**Three complementary filters:**
- `min_synapse_num`: Absolute synapse count (e.g., ≥10 synapses)
- `min_ratio`: Direct connection ratio (e.g., ≥1% of inputs)
- `min_traversal_probability`: Scaled probability (e.g., ≥0.1 probability)

All output data includes: `weight`, `connection_ratio`, and `traversal_probability` columns.

#### Excluding Intra-Type Connections

You can exclude connections within the same neuron type using the `exclude_intra_type_connections` parameter. This is particularly useful when analyzing cross-type connectivity patterns or building network visualizations that focus on inter-type communication.

```python
fc = FindNeuronConnection(
    ... # other parameters
    sourceNeurons=['MBON.*'],
    targetNeurons=['MBON.*'],
    exclude_intra_type_connections=True,  # Exclude MBON→MBON connections
    ... # other parameters
)
```

**Use cases:**
- Analyzing cross-type connectivity patterns while excluding self-connections
- Building cleaner network visualizations focused on inter-type pathways
- Studying how different neuron types interact without intra-type noise

See **[Example: Network Visualization with Excluded Intra-Type Connections](examples/Example_ExcludeIntraType.py)** for a complete demonstration.

If you want to show the figure of the connection matrix, set ```showfig``` to ```True```, otherwise set it to False.

```python
fc = FindNeuronConnection(
    ... # other parameters
    showfig = False, # default is True
    ... # other parameters
)
```

after specified necessary parameters, you can run the codes to find the direct connections between the source neurons and the target neurons.

To find the direct connections, we call the ```FindNeuronConnection.InitializeNeuronInfo()``` method to initialize before running the ```FindNeuronConnection.FindDirectConnection()``` method.

```python
fc.InitializeNeuronInfo()
fc.FindDirectConnection()
```

In the ```FindNeuronConnection.FindDirectConnection()``` method, you can specify the ```full_data``` parameter to ```True``` (defaulty ```False```) to do clustering and other analysis on the connection data.

```python
fc.FindDirectConnection(full_data=True) # defaultly, full_data is False
```

### FindPath.py

Use this function to find direct and indirect connection paths between the neuron clusters.

```python
fc = FindNeuronConnection(
    token='',
    dataset = 'hemibrain:v1.2.1',
    data_folder=R'D:\connectome_data',
    sourceNeurons = ['.*_.*PN.*'], 
    targetNeurons = ['MBON.*'], 
    custom_source_name = '', 
    custom_target_name = '',
    min_synapse_num = 1,
    min_traversal_probability = 0.001,
    showfig = False,
    max_interlayer=2,
    keyword_in_path_to_remove=['None'],
)
```

Comparing with the FindDirect.py, the call of FindNeuronConnection in FindPath.py uses two more parameters: ```max_interlayer``` and ```keyword_in_path_to_remove```.

the max_interlayer parameter is used to specify the maximum number of layers between the source neurons and the target neurons to search for the indirect connection paths.

The ```keyword_in_path_to_remove``` parameter is used to specify the keywords in the indirect connection paths to remove. For example, if you want to remove the paths that contain the keyword 'None', you can set the ```keyword_in_path_to_remove``` parameter to ['None'] (most neurons in the dataset are not labels, which has a type of "None" in the found paths). If you want to remove paths containing the APL neuron and the "None" neurons, you can set the ```keyword_in_path_to_remove``` parameter to ['APL','None'].

```python
fc = FindNeuronConnection(
    ... # other parameters
    max_interlayer=2, # default is 2
    keyword_in_path_to_remove=['APL', 'None'],
    ... # other parameters
)
```

### Special Mode: FetchNeuronsOnly (max_interlayer=-1)

Setting `max_interlayer=-1` activates **FetchNeuronsOnly mode** - this fetches only the source and target neurons without computing any connections. This is useful for:

- Getting neuron metadata for cross-dataset comparison preparation
- Extracting neuron lists for downstream processing
- Pre-fetching neuron data for visualization or skeleton rendering

```python
# FetchNeuronsOnly mode - no connection computation
fc = FindNeuronConnection(
    token='your_token',
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['aMe12'],
    targetNeurons=['PPL101'],
    max_interlayer=-1,  # Fetch neurons only, no connections
)

# Access fetched neurons
source_df = fc.sourceNeurons_df  # DataFrame with source neuron metadata
target_df = fc.targetNeurons_df  # DataFrame with target neuron metadata

# Save neuron info to files
fc.SaveNeuronInfo(
    output_dir='/path/to/output',
    format='both'  # 'csv', 'parquet', or 'both'
)
# Creates: source_neurons.csv, source_neurons.parquet, target_neurons.csv, target_neurons.parquet
```

This is particularly useful for cross-dataset comparison workflows where you need to identify neurons first before running comprehensive path analysis.

Optionally, there are more parameters you can specify in the ```FindNeuronConnection``` call.

## Cache System v3.0 (NEW)

To avoid repeated API calls and speed up analysis, FindNetwork now includes a **local caching system** that stores fetched connection data.

### Enabling Cache

Simply set `use_cache=True` when creating a `FindNeuronConnection` instance:

```python
fc = FindNeuronConnection(
    token='your_token',
    dataset='optic-lobe:v1.1',
    sourceNeurons=['L3_R'],
    targetNeurons=['l-LNv_R'],
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
Pulled 53847 neurons from optic-lobe:v1.1
✅ Complete dataset saved
```

This ensures cache can enrich all connections, even those involving untyped neurons. Takes ~15-30 seconds depending on dataset size.

### Cache Management

View cache information:
```python
fc.print_cache_info()
```

Clear cache:
```python
fc.clear_cache()
```

Interactive management:
```bash
python ManageCache.py
```

Search cached neurons:
```python
# Search by neuron type
l3_neurons = fc.search_cached_neurons('L3.*', 'type')

# Search by instance (e.g., right hemisphere)
right_neurons = fc.search_cached_neurons('.*_R$', 'instance')
```

For detailed information, see:
- **[Cache System Guide](docs/CacheSystem_Guide.md)** - Comprehensive caching guide
- **[Cache System v3.0 Architecture](docs/CacheSystem_v3_DatabaseArchitecture.md)** - Database architecture and technical details
- **[Cache System Quick Start](docs/CacheSystem_QuickStart.md)** - Quick reference guide
- **[Storage Optimization](docs/CacheSystem_StorageOptimization.md)** - Why we only store bodyId pairs
- **[Complete Dataset](docs/CacheSystem_CompleteDataset.md)** - Handling neurons with type=None

---

Use this function to plot the 3D skeleton of the neuron clusters at different layers, you can also input a single layer to plot the skeleton of all the neurons in that layer.

We use ```statvis.LogInHemibrain()``` to provide your token and specify the dataset.

```python
import statvis as sv
sv.LogInHemibrain(token='', dataset='hemibrain:v1.2.1')
```

Then, by calling the ```VisualizeSkeleton``` class, we can initialize the parameters for plotting the 3D skeleton.

```python
from coana import VisualizeSkeleton as vs
vs = VisualizeSkeleton(
    data_folder = R'',
    neuron_layers = ['VA1d_adPN', 'LHCENT3', 'MBON01'], # or in the format: 'VA1d_adPN -> LHCENT3 -> MBON01'
    custom_layer_names = ['VA1d PN', 'my LHN', 'MBON_1'],
    neuron_alpha = 0.2,
    saveas = None,
    min_synapse_num = 1,
    synapse_size = 3,
    synapse_alpha = 0.6,
    mesh_roi = ['LH(R)','AL(R)','EB','gL(R)'],
    brain_mesh = 'template',  # 'none', 'template' (EM resolution), or 'whole' (standard brain)
    skeleton_mode = 'tube',
    synapse_mode = 'scatter',
    legend_mode = 'merge',
    use_size_slider = True,
    show_fig = True,
)
```

After that, we use the ```plot_neuron()``` method to plot the 3D skeleton. and export the 3-D skeleton to a video with rotating objects by using the ```export_video()``` method. See details in the docstrings of the methods.

```python
vs.plot_neurons()
vs.export_video(fps=30,rotate_plane='xy',synapse_size=2,scale=2,)
```

parameters of the ```export_video()``` determines the properties of the video.

### Multi-Dataset Brain Mesh Support

**NEW in v3.1**: VisualizeSkeleton now supports all NeuPrint datasets with automatic brain/VNC mesh rendering:

**Supported Datasets:**
- **hemibrain:v1.2.1** / **optic-lobe:v1.1**: Adult brain (supports whole-brain visualization)
- **manc:v1.2.3**: Male VNC (native VNC template)
- **male-cns:v0.9**: Full CNS (brain + VNC combined)

**Brain Mesh Options:**

```python
# Option 1: No brain mesh (fastest, clearest for single neurons)
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['KC.*'],
    brain_mesh='none'  # No background mesh
)

# Option 2: Template mesh (native EM resolution, 0.5-2 seconds)
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['KC.*'],
    brain_mesh='template'  # JRCFIB2018F for hemibrain, MANC for manc, JRCFIB2022M for male-cns
)

# Option 3: Whole brain/VNC mesh (standard resolution, requires one-time download)
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['KC.*'],
    brain_mesh='whole'  # JRC2018F whole brain, automatically downloads transforms
)
```

**Template Mappings:**
- **hemibrain/optic-lobe**:
  - Source: `JRCFIB2018Fraw` (NeuPrint raw coordinates)
  - Template: `JRCFIB2018F` (calibrated EM mesh)
  - Whole: `JRC2018F` (standard confocal brain)
  
- **manc**:
  - Source: `MANCraw` (NeuPrint raw)
  - Native: `MANC` (VNC template, no transforms needed)
  
- **male-cns**:
  - Source: `JRCFIB2022Mraw` (NeuPrint raw)
  - Native: `JRCFIB2022M` (full CNS template, no transforms needed)

**Transform Downloads:**

When using `brain_mesh='whole'` for hemibrain/optic-lobe datasets, the system will automatically prompt you to download brain transforms (~10GB total, one-time):

```
⚠ Brain transforms not found for hemibrain:v1.2.1
  Transform path: JRCFIB2018Fraw → JRCFIB2018F → JRCFIB2018Fum → JRC2018F
  File needed: JRC2018F_JRCFIB2018F.h5 (~1.3 GB)
  Total download: ~10 GB (8 files bundled together)
  Download time: ~1-2 hours
  Storage location: ~/flybrain-data

Download all transforms now? [y/N]:
```

**Why download 10GB when you only need 1.3GB?**

The flybrains package bundles all JRC brain transforms together:
- **JRC2018F_JRCFIB2018F.h5 (~1.29 GB)** - What you actually need for hemibrain/optic-lobe
- 7 other transform files (~8.7 GB) - Enable other datasets and cross-registration

Selective download is not supported by the flybrains library. However, downloading all transforms:
- ✅ Enables cross-dataset functionality (e.g., FAFB, male-cns)
- ✅ One-time setup shared across all projects
- ✅ Future-proof for working with multiple datasets

Transforms are stored in `~/flybrain-data` (managed by flybrains package) and shared across all NeuPrint projects.

**Documentation:**
- Full Guide: [docs/visualizations/VisualizeSkeleton_Updates_Nov2024.md](docs/visualizations/VisualizeSkeleton_Updates_Nov2024.md)
- Brain Template Fix: [docs/bugfixes/Brain_Template_Fix_Nov2024.md](docs/bugfixes/Brain_Template_Fix_Nov2024.md)

## Performance Optimization

### Parallel Processing for Pathfinding

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

**Performance Benefits:**
- **4-core CPU**: 3-4x faster for large datasets
- **8-core CPU**: 6-8x faster for large datasets
- **16+ core CPU**: 10-14x faster for large datasets

**Parameters:**
- `use_parallel`: Enable/disable parallel processing (default: `False`)
- `n_jobs`: Number of processes to use
  - `-1`: Use all CPU cores (recommended)
  - `1`: Sequential processing
  - `N`: Use exactly N processes

**Automatic Optimization:**
- Datasets with <100 pairs: Uses sequential processing (overhead not worth it)
- Datasets with >100 pairs: Uses parallel processing (significant speedup)

**Examples:**
```python
# Example 1: Use all CPU cores (recommended for workstations)
fc = FindNeuronConnection(
    token='your_token_here',
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['PPL1-01'],
    targetNeurons=['MBON14'],
    max_interlayer=3,
    use_parallel=True,
    n_jobs=-1  # Auto-detect and use all cores
)

# Example 2: Use specific number of cores (recommended for laptops)
fc = FindNeuronConnection(
    token='your_token_here',
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['PPL1-01'],
    targetNeurons=['MBON14'],
    max_interlayer=3,
    use_parallel=True,
    n_jobs=4  # Use exactly 4 cores, leave others for system
)

# Example 3: Disable parallel processing (small datasets)
fc = FindNeuronConnection(
    token='your_token_here',
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['PPL1-01'],
    targetNeurons=['MBON14'],
    max_interlayer=3,
    use_parallel=False  # Use sequential processing
)
```

**Documentation:**
- Full details: [ParallelProcessing_Documentation.md](docs/ParallelProcessing_Documentation.md)
- Examples: [Example_ParallelProcessing.py](Example_ParallelProcessing.py)

### Caching System

The system automatically caches fetched data to avoid redundant API calls:

```python
# First run: Fetches data from NeuPrint API
fc = FindNeuronConnection(
    token='your_token_here',
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['PPL1-01'],
    targetNeurons=['MBON14'],
    use_cache=True  # Enable caching (default)
)

# Subsequent runs: Loads from local cache (much faster)
# No API calls needed for previously fetched data
```

**Cache Features:**
- **Automatic**: No manual cache management needed
- **Smart**: Only caches essential connection data (40-60% smaller than raw data)
- **Complete**: Includes all neurons (even those without assigned types)
- **Fast**: Parquet format with gzip compression for quick loading

**Cache Management:**
```python
# Clear cache for specific dataset
fc = FindNeuronConnection(
    token='your_token_here',
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['PPL1-01'],
    targetNeurons=['MBON14'],
    use_cache=False  # Bypass cache, fetch fresh data
)

# Or use ManageCache.py to manage cache manually
from ManageCache import CacheManager
manager = CacheManager()
manager.clear_cache('hemibrain_v1.2.1')
```

**Documentation:**
- Complete Guide: [CacheSystem_Guide.md](docs/CacheSystem_Guide.md)
- Architecture: [CacheSystem_v3_DatabaseArchitecture.md](docs/CacheSystem_v3_DatabaseArchitecture.md)
- Optimization: [CacheSystem_StorageOptimization.md](docs/CacheSystem_StorageOptimization.md)
- Complete dataset: [CacheSystem_CompleteDataset.md](docs/CacheSystem_CompleteDataset.md)

### VisualizePath: Standalone Network Visualization

The `VisualizePath` class provides standalone visualization without requiring NeuPrint API access. You can visualize **any network data** with simple input formats.

**Supported Input Formats:**

1. **Simple Edge-List** (Just 3 columns required):
```python
from vispath import VisualizePath
import pandas as pd

# Create simple edge data
edges = pd.DataFrame({
    'source': ['A', 'B', 'C'],
    'target': ['B', 'C', 'D'],
    'weight': [10, 20, 15]
})

# Visualize
vis = VisualizePath(path_file=edges, output_folder='./output')
vis.create_network()  # Interactive Cytoscape.js network
vis.create_sankey()   # Layered Sankey diagram
```

2. **Flexible Column Names** (Auto-detected):
```python
# Works with bodyId format
df = pd.DataFrame({
    'bodyId_pre': [123, 456],
    'bodyId_post': [456, 789],
    'weight': [25, 30]
})

# Works with neuron format
df = pd.DataFrame({
    'neuron_pre': ['PN1', 'PN2'],
    'neuron_post': ['LN1', 'LN2'],
    'synapse_count': [100, 150]
})

# Works with from/to format
df = pd.DataFrame({
    'from': ['KC_a', 'KC_b'],
    'to': ['MBON_a', 'MBON_b'],
    'weight': [50, 60]
})
```

**Recognized Column Names:**
- **Source**: `source`, `from`, `pre`, `*_pre` (e.g., `bodyId_pre`, `type_pre`)
- **Target**: `target`, `to`, `post`, `*_post` (e.g., `bodyId_post`, `type_post`)
- **Weight**: `weight`, `weights`, `synapse_count`, `count`

**Load from CSV/Excel:**
```python
# From CSV file
vis = VisualizePath(path_file='my_network.csv')
vis.create_network()

# From Excel file (auto-detects sheet)
vis = VisualizePath(path_file='my_network.xlsx')
vis.create_sankey()
```

**Customization Options:**
```python
vis = VisualizePath(
    path_file='network.csv',
    output_folder='./results',
    source_color='#3498db',           # Blue source nodes
    target_color='#e74c3c',           # Red target nodes
    intermediate_color='#2ecc71',     # Green intermediate
    link_color='rgba(100,100,100,0.4)', # Semi-transparent links
    node_size=30,
    font_size=12
)
```

**Examples:**
- Basic: [Example_SimpleEdgeList.py](Example_SimpleEdgeList.py)
- Advanced: [Example_VisualizeSelectedPaths.py](Example_VisualizeSelectedPaths.py)

**Documentation:**
- Full Guide: [VisualizeSelectedPaths_Guide.md](docs/VisualizeSelectedPaths_Guide.md)

## VisualizePath Sub-Project: Standalone Installation

**NEW in v3.0**: The `VisualizePath` visualization toolkit is now available as a standalone sub-project that can be installed **independently** without the full neuroscience analysis suite!

### Why Use the Standalone Sub-Project?

The sub-project provides a **minimal lightweight installation** focused only on network visualization:
- ✅ **Smaller footprint**: Only 6 core dependencies (vs 15 in full package)
- ✅ **No neuroscience deps**: No neuprint-python, navis, flybrains required
- ✅ **Standalone visualizations**: Works with any network data (CSV/Excel)
- ✅ **Same API**: Identical functionality to the full package for visualization features

### Standalone Installation

```bash
# Install only the visualization toolkit
cd vispath-subproject
pip install -e .

# Or directly from GitHub
pip install git+https://github.com/Swida-Alba/hemibrain-connectomes-analysis.git#subdirectory=vispath-subproject
```

**Recommended with conda:**
```bash
# Create dedicated environment for vispath
conda create -n vispath python=3.11 -y
conda activate vispath
cd vispath-subproject
pip install -e .
```

**Minimal dependencies installed:**
- numpy (<2.0.0), pandas (<2.0.0), scipy
- plotly, networkx
- openpyxl
- PyQt5 (optional, for GUI)

### Usage Comparison

```python
# Standalone installation
from vispath_pkg import VisualizePath

# Full installation
from vispath import VisualizePath

# API is identical after import!
vp = VisualizePath(path_file='network.csv', output_folder='./output')
vp.create_network()
vp.create_sankey()
```

### When to Use Standalone vs Full Package

**Use standalone if you:**
- ✅ Only need network visualization features
- ✅ Have your own network data (CSV/Excel)
- ✅ Want minimal dependencies
- ✅ Don't need NeuPrint API access or neuron morphology features

**Use full package if you:**
- ✅ Need NeuPrint database connectivity
- ✅ Want automated pathfinding algorithms
- ✅ Need 3D neuron skeleton rendering
- ✅ Want advanced heatmap/clustering features
- ✅ Need the complete analysis pipeline

**Note:** The full installation automatically includes the visualization toolkit, so you can use both approaches:
```bash
# Full installation includes everything
pip install -e .  # From root directory

# After full install, both imports work:
from vispath import VisualizePath        # Full package
from vispath_pkg import VisualizePath    # Sub-project (same functionality)
```

### Documentation

For complete sub-project documentation, see:
- **[Sub-Project Installation Guide](vispath-subproject/INSTALLATION.md)** - Setup and usage
- **[Sub-Project README](vispath-subproject/README.md)** - Quick reference
- **[Visualization Guide](docs/visualizations/README.md)** - Visualization features

---

## Cross-Dataset Comparison

**NEW in v4.0**: Compare connectivity patterns across multiple connectome datasets to identify conserved circuits, dataset-specific connections, and validate findings across independent reconstructions.

### Supported Datasets

- **NeuPrint datasets**: `hemibrain:v1.2.1`, `male-cns:v0.9`, `optic-lobe:v1.1`, etc.
- **FlyWire datasets**: `flywire_FAFB_v783`, `flywire_BANC_v626` (local parquet files)

### Quick Start Example

```python
from comparison import ComparisonParameters, ComparisonAnalyzer

# Define comparison parameters
params = ComparisonParameters(
    datasets=['hemibrain:v1.2.1', 'male-cns:v0.9', 'flywire_FAFB_v783'],
    datasets_nickname=['hemi', 'mcns', 'fafb'],  # Short names for displays
    source_neurons=['aMe12'],
    target_neurons=['PPL101'],
    max_interlayer=1,
    thresholds=[1, 5, 10],
    comparison_mode='edge',  # or 'path'
    output_folder='/path/to/output',
)

# Run comparison
analyzer = ComparisonAnalyzer(params, verbose=True)
results = analyzer.run_comparison()
```

### Comparison Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `path` | Analyzes edges on complete paths only | Functional circuit analysis |
| `edge` | Direct edge weight comparison | Identifying all possible connections |

### Output Files

The comparison generates a comprehensive output folder:

```
comparison_results_YYYYMMDD_HHMMSS/
├── comparison_report.html      # Interactive HTML report
├── comparison_results/         # CSV data files
│   ├── edge_presence_matrix.csv
│   ├── path_presence_matrix.csv
│   └── neuron_counts_by_type.csv
├── comparison_visualizations/  # PNG charts
│   ├── similarity_heatmap_jaccard.png
│   └── threshold_comparison.png
└── comparison_networks/        # Interactive network HTML
```

### Key Metrics

- **Conservation Count**: Number of datasets where an edge/path is present
- **Jaccard Similarity**: Set-based overlap of connections between dataset pairs
- **Cosine Similarity**: Weight vector similarity accounting for magnitudes
- **Rank Correlation**: Spearman correlation of partner rankings

### Intra-Dataset Analysis (Internal Network)

Analyze internal connectivity within a neuron group by setting source = target:

```python
# Analyze connections within aMe-type neurons
neuron_group = ['aMe.*']

params = ComparisonParameters(
    datasets=['male-cns:v0.9', 'hemibrain:v1.2.1'],
    source_neurons=neuron_group,
    target_neurons=neuron_group,  # Same as source
    max_interlayer=0,  # Direct connections only
    thresholds=[5, 10, 20],
    comparison_mode='edge',
    output_folder='/path/to/output',
)
```

**Documentation:** [Cross-Dataset Comparison Guide](docs/core-features/CrossDatasetComparison_Guide.md)

---

## Connectivity Profile Verification

**NEW in v4.0**: Verify neuron type assignments across datasets by comparing connectivity profiles (fingerprints). This system extracts the top upstream and downstream partners of neurons and computes similarity metrics.

### What is a Connectivity Profile?

A connectivity profile captures a neuron's **connectivity fingerprint**:
- Top K upstream partners (neurons providing input)
- Top K downstream partners (neurons receiving output)
- Normalized weights for each partner

### Quick Start Example

```python
from comparison import ConnectivityProfiler, CrossDatasetVerifier, ProfilerConfig

# Configure profiler
config = ProfilerConfig(
    top_k_bodyid=20,        # Top 20 connections per direction
    top_m_type=5,           # Ensure at least 5 unique partner types
    min_synapse_threshold=3 # Filter weak connections
)

# Create profiler and verifier
profiler = ConnectivityProfiler(
    datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
    config=config
)
verifier = CrossDatasetVerifier(profiler)

# Verify a neuron type across datasets
results = verifier.verify_type_assignment(
    'aMe12', 
    datasets=['hemibrain:v1.2.1', 'male-cns:v0.9']
)
print(results.summary())
```

### Similarity Metrics

| Metric | Description | Confidence Thresholds |
|--------|-------------|----------------------|
| **Jaccard** | Set overlap of partner types | >0.5 Very High, >0.3 High, >0.2 Medium |
| **Rank Correlation** | Spearman correlation of partner rankings | ≥0.85 Very High, 0.7-0.85 High, 0.5-0.7 Medium |
| **Cosine** | Weight vector similarity | Used for combined scoring |

### Verification Status

Based on the combined similarity score:
- **verified**: High confidence (score ≥ 0.7)
- **needs_review**: Medium confidence (score 0.5-0.7)
- **questionable**: Low confidence (score 0.3-0.5)
- **failed**: Very low confidence (score < 0.3)

### Batch Verification

Verify multiple neuron types at once:

```python
# Verify all types found in comparison results
batch_results = verifier.batch_verify(
    neuron_types=['aMe12', 'PPL101', 'KCg-d', 'MBON01'],
    datasets=['hemibrain:v1.2.1', 'male-cns:v0.9']
)

# Generate HTML report with similarity matrices
verifier.generate_html_report(
    batch_results,
    output_path='/path/to/verification_report.html'
)
```

### HTML Report Features

The verification report includes:
- **Verification Summary Table**: All types with scores and confidence badges
- **Jaccard Similarity Matrix**: Pairwise dataset comparison for each direction
- **Upstream/Downstream Breakdown**: Separate Jaccard scores for input vs output connectivity
- **Confidence Indicators**: Color-coded badges (Very High, High, Medium, Low, Very Low)

**Documentation:** See `src/comparison/` module and [Connectivity Profile Verification Guide](docs/core-features/ConnectivityProfileVerification_Guide.md).

---

## Installation: For users who can prepare the python environments by themselves

### Recommended: Using Conda Environment

**Create a new conda environment for better isolation:**

```bash
# Create environment with Python 3.11
conda create -n hemibrain python=3.11 -y

# Activate the environment
conda activate hemibrain

# Install dependencies
pip install -r requirements.txt
```

**Or install the package in editable mode:**

```bash
conda activate hemibrain
pip install -e .
```

### Quick Install (Without Conda)

```bash
pip install -r requirements.txt
```

This will install all necessary dependencies including:
- Core packages: numpy (<2.0.0), pandas (<2.0.0), scipy
- Visualization: plotly, matplotlib, seaborn, networkx
- Data processing: openpyxl, xlrd
- **PyQt5**: For fast, responsive GUI file dialogs (recommended)
- Neuroscience: neuprint-python, navis

**Important:** numpy is constrained to <2.0.0 for binary compatibility with pandas 1.x.

### Alternative Installation Methods

**Editable installation (for development):**
```bash
pip install -e .
```

**Using pyproject.toml (Modern Python):**
```bash
# Basic installation
pip install .

# With development tools
pip install -e ".[dev]"

# With all optional dependencies
pip install -e ".[all]"
```

### GUI Backend for File Dialogs

The system automatically uses the **fastest available** GUI backend:

| Backend | Speed | Installation |
|---------|-------|--------------|
| **PyQt5** | ⚡⚡⚡ Fastest (0.1s) | `pip install PyQt5` (Included in requirements.txt) |
| **PyQt6** | ⚡⚡⚡ Fastest | `pip install PyQt6` |
| **wxPython** | ⚡⚡ Fast | `pip install wxPython` |
| **tkinter** | ⚡ Slow (2-5s) | Built-in with Python |

**For best performance**, PyQt5 is included by default in `requirements.txt`. File picker and sheet selection dialogs will be **10-50x faster** than using tkinter alone.

### Verify Installation

```bash
# Quick test
python -c "import numpy, pandas, plotly, networkx, PyQt5; print('✅ All packages installed')"

# GUI test
python test_sheet_confirmation.py
```

### Platform-Specific Notes

**macOS:**
```bash
pip install -r requirements.txt
```
All dependencies install via pip. PyQt5 provides native macOS dialogs.

**Linux (Ubuntu/Debian):**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-pip python3-dev python3-tk

# Install Python packages
pip install -r requirements.txt
```

**Windows:**
```bash
pip install -r requirements.txt
```
Make sure Python was installed with PATH option enabled.

### Troubleshooting

**numpy/pandas Binary Compatibility Error:**
```bash
# If you see "numpy.dtype size changed" error
pip install 'numpy<2.0.0' --force-reinstall
```
This is already handled in requirements.txt with the constraint `numpy>=1.20.0,<2.0.0`.

**PyQt5 won't install:**
```bash
pip install --upgrade pip
pip install PyQt5
```

**Pandas version conflict:**
```bash
pip install "pandas<2"
```

**Import errors:**
```bash
pip install --upgrade --force-reinstall -r requirements.txt
```

### Complete Documentation

- **[Installation Guide](docs/INSTALLATION.md)** - Detailed installation instructions
- **[Dialog Performance Guide](docs/DIALOG_PERFORMANCE_GUIDE.md)** - GUI backend comparison and optimization
- **[Dependency Summary](docs/DEPENDENCY_SUMMARY.md)** - Overview of all dependencies

**Package Requirements:** See [requirements.txt](requirements.txt) for full list  
**Note:** pandas version should be <2.0.0 for compatibility

## Installation: Step-by-Step Guide for Beginners

This guide is for users who are new to Python or need detailed installation instructions. If you're comfortable with Python environments, see the [Quick Install](#quick-install-recommended) section above.

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
- `scripts/FindPath_flywire.py` - Find paths in FlyWire/FAFB dataset (local file based)
- `scripts/PlotPath.py` - Visualize pathways with custom colors
- `scripts/plot3dSkeleton.py` - Render 3D neuron morphology

**Modify parameters:**
- Change `sourceNeurons` and `targetNeurons` to analyze different connections
- Adjust `min_synapse_num` to filter weak connections
- Try different `network_layout` options: 'hierarchical', 'spring', 'circular'

### Troubleshooting

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

**Still having issues?**
- Check [docs/INSTALLATION.md](docs/INSTALLATION.md) for detailed troubleshooting
- Open an issue on GitHub with your error message

### Next Steps

✅ **You're all set!** Now you can:
1. Explore the [example scripts](examples/) for different use cases
2. Read the [Quick Start Guide](docs/QUICK_START_AFTER_REORGANIZATION.md)
3. Check [Core Features](docs/core-features/README.md) for advanced usage
4. Browse [Visualization Guides](docs/visualizations/README.md) for customization options

**Happy analyzing! 🧠✨**

---

## 🆕 What’s New in V4.0 (November 2025)

- **Contralateral Mirroring**: `VisualizeSkeleton` now supports mirroring neurons and ROIs to the contralateral hemisphere.
  - Set `mirror_on_contralateral=True` to enable.
  - Automatically mirrors ROIs ending in `(R)` to `(L)`.
  - Uses dataset-specific templates (`JRCFIB2018F` for hemibrain/optic-lobe, `JRCFIB2022M` for male-cns) for accurate mirroring.

### Cross-Dataset Comparison & Profile Verification
- **Cross-Dataset Comparison**: Compare connectivity patterns across multiple connectome datasets (hemibrain, male-cns, FlyWire FAFB/BANC).
  - Path-based and edge-based comparison modes
  - Conservation analysis with Jaccard, cosine, and rank correlation metrics
  - Interactive HTML reports with similarity matrices and network visualizations
  - See [Cross-Dataset Comparison Guide](docs/core-features/CrossDatasetComparison_Guide.md)

- **Connectivity Profile Verification**: Verify neuron type assignments using connectivity fingerprints.
  - Extract top upstream/downstream partners as profiles
  - Compute similarity scores (Jaccard, rank correlation, cosine)
  - Batch verification with confidence levels (Very High, High, Medium, Low)
  - Upstream/downstream breakdown for directional analysis

### Visualization Enhancements
- **Connection Matrix Support**: `VisualizePath` now accepts connection matrices (square or rectangular DataFrames) as direct input. Auto-detects format and converts to edge-list internally.
- **Individual Visualization Control**: New parameters in `visualize()` (`plot_heatmap`, `plot_Sankey`, `plot_network`) allow generating specific visualizations on demand.
- **Reciprocal Edge Offset & Mode Toggle**: Network visualization now supports parallel reciprocal edges with a user-adjustable offset slider and a toggle for straight/curved edge modes.

### HTML Report Improvements
- **Overlap Matrix**: Larger, square overlap matrices with count/proportion toggle
- **Jaccard Similarity Matrix**: Added to connectivity profile HTML reports
- **Updated Confidence Thresholds**:
  - Jaccard: >0.5 Very High, >0.3 High, >0.2 Medium, >0.1 Low
  - Rank Correlation: ≥0.85 Very High, 0.7-0.85 High, 0.5-0.7 Medium, 0.3-0.5 Low

**Documentation**: See [VisualizePath Updates Nov 2025](docs/visualizations/VisualizePath_Updates_Nov2025.md) for visualization details.

---
