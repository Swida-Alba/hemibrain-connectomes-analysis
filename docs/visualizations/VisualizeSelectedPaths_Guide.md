# Visualize Selected Paths Guide

## Overview

**`VisualizePath`** is a standalone class for creating focused visualizations of specific pathways after running pathfinding analysis. This allows you to:

- Visualize only interesting/significant paths
- Create publication-ready figures of selected connections
- Explore specific pathways in detail
- Compare different path subsets

**Key Feature:** `VisualizePath` is completely independent - you don't need to initialize `FindNeuronConnection` first!

## Example Input Data

### Required Format

Your path data should be an edge list with these columns:

| Column | Description | Example Values | Required |
|--------|-------------|----------------|----------|
| `source` or `from` | Source neuron | 'KC_alpha', 'MBON01' | ✅ Yes |
| `target` or `to` | Target neuron | 'MBON03', 'DAN_a' | ✅ Yes |
| `weight` | Connection strength | 10, 25, 150 | ✅ Yes |

### Example Input Files

**From FindAllPath results** (recommended):
- Path: `connection_data/*/allpaths_*/path_info.xlsx`
- Sheets:
  - `path_type` - Type-level connections (neuron types)
  - `path_bodyId` - BodyId-level connections (individual neurons)
- Example: [`examples/example_paths.csv`](../../archive/examples/data/example_paths.csv)

**Simple edge-list files**:
```csv
# Minimal 3-column format
source,target,weight
KC_a,MBON01,25
KC_a,MBON03,10
MBON01,DAN_a,50
```

**With additional metrics**:
```csv
source,target,weight,ratio,probability
KC_alpha,MBON03,25,0.15,0.3
KC_beta,MBON03,12,0.08,0.2
MBON03,DAN_a,50,0.4,0.7
```

**Example files provided**:
- [`examples/example_neuron_network.csv`](../../archive/examples/data/example_neuron_network.csv) - Multi-layer pathways
- [`examples/example_bodyid_network.csv`](../../archive/examples/data/example_bodyid_network.csv) - BodyId-level network
- [`examples/simple_network_data.csv`](../../archive/examples/data/simple_network_data.csv) - Minimal format
- [`examples/example_paths.csv`](../../archive/examples/data/example_paths.csv) - Path analysis results

**Filtered subset example**:
```python
import pandas as pd

# Load all paths
all_paths = pd.read_excel('path_info.xlsx', sheet_name='path_type')

# Filter for high-quality paths
high_quality = all_paths[
    (all_paths['traversal_probability'] > 0.5) &
    (all_paths['weight'] > 10)
]

# Visualize filtered subset
vp = VisualizePath(path_file=high_quality)  # Pass DataFrame directly!
conn_df, G = vp.visualize()
```

## Quick Start

```python
from vispath import VisualizePath

# Simple one-liner
vp = VisualizePath('path_type.xlsx')
conn_df, G = vp.visualize()
```

## When to Use

✅ **Use VisualizePath when:**
- You've run FindAllPath and have too many paths to visualize clearly
- You want to focus on high-probability or high-weight paths
- You need to visualize paths through specific intermediate neurons
- You want to create custom visualizations with specific path combinations
- You need multiple views of the same data with different filters
- You want standalone visualization without initializing FindNeuronConnection
- You have simple edge-list data from any source (not just NeuPrint)

❌ **Don't use when:**
- You haven't run FindAllPath yet (run that first)
- You want to see all paths (just use the default FindAllPath output)
- You're still exploring the data (use full visualizations first)

---

## Class Signature

```python
from vispath import VisualizePath

vp = VisualizePath(
    path_file,                    # Required: CSV or Excel file with paths
    sheet_name=None,              # Optional: Excel sheet name
    output_folder=None,           # Optional: Where to save visualizations
    node_color=None,              # Optional: Colors for nodes
    target_color=None,            # Optional: Color for target nodes
    link_color=None,              # Optional: Color for connections
    network_layout='hierarchical', # Optional: Network layout algorithm
    showfig=False                 # Optional: Auto-open in browser
)

# Create all visualizations
conn_df, G = vp.visualize()
```

## Alternative: Convenience Function

```python
from vispath import visualize_paths

# One-liner wrapper
conn_df, G = visualize_paths(
    path_file='path_type.xlsx',
    showfig=True
)
```

## Alternative: Through FindNeuronConnection

```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(...)

# This internally uses VisualizePath
conn_df, G = fc.VisualizeSelectedPaths('path_type.xlsx')
```

---

## Input File Format

### Required Columns

Your CSV/Excel file must contain:

1. **`path_block`** (string)
   - Format: `"NodeA -> NodeB -> NodeC -> NodeD"`
   - Nodes separated by ` -> ` (space-arrow-space)
   - Example: `"L3_R -> Mi1_R -> Tm3_R -> T4a_R"`

2. **`weights`** (list of integers)
   - Synapse counts for each hop in the path
   - Format: `[weight1, weight2, weight3]`
   - Example: `[150, 80, 45]` means:
     - L3→Mi1: 150 synapses
     - Mi1→Tm3: 80 synapses
     - Tm3→T4a: 45 synapses

### Optional Columns

3. **`connection_ratios`** (list of floats)
   - Connection ratio (weight/post) for each hop
   - Format: `[ratio1, ratio2, ratio3]`
   - Example: `[0.25, 0.18, 0.12]`

4. **`traversal_probabilities`** (list of floats)
   - Traversal probability for each hop
   - Format: `[prob1, prob2, prob3]`
   - Example: `[0.85, 0.75, 0.65]`

### Example Input File

**selected_paths.csv:**
```csv
path_block,weights,connection_ratios,traversal_probabilities
"L3_R -> Mi1_R -> Tm3_R -> T4a_R","[150, 80, 45]","[0.25, 0.18, 0.12]","[0.85, 0.75, 0.65]"
"L3_R -> Mi4_R -> Tm3_R -> T4a_R","[120, 65, 40]","[0.22, 0.16, 0.11]","[0.82, 0.70, 0.60]"
"L3_R -> Mi1_R -> TmY3_R -> T4a_R","[150, 70, 35]","[0.25, 0.15, 0.10]","[0.85, 0.68, 0.55]"
```

**Or from Excel (path_type.xlsx):**
| path_block | weights | connection_ratios | traversal_probabilities | inter_layer_num |
|------------|---------|-------------------|-------------------------|-----------------|
| L3_R -> Mi1_R -> Tm3_R -> T4a_R | [150, 80, 45] | [0.25, 0.18, 0.12] | [0.85, 0.75, 0.65] | 2 |
| L3_R -> Mi4_R -> Tm3_R -> T4a_R | [120, 65, 40] | [0.22, 0.16, 0.11] | [0.82, 0.70, 0.60] | 2 |

---

## Usage Examples

### Example 1: Basic Usage - Visualize Existing Paths

```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(
    token='your_token',
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['L3.*_R'],
    targetNeurons=['T4a.*_R']
)

# Visualize paths from FindAllPath results
fc.VisualizeSelectedPaths(
    path_file='./path_L3_to_T4a/path_type.xlsx',
    sheet_name='path_type',
    output_folder='./selected_visualization'
)
```

### Example 2: Filter High-Quality Paths

```python
import pandas as pd

# Read all paths
all_paths = pd.read_excel('./path_results/path_type.xlsx', sheet_name='path_type')

# Filter: high probability AND short paths
high_quality = all_paths[
    (all_paths['traversal_probability'] > 0.5) &
    (all_paths['inter_layer_num'] <= 2)
]

# Save filtered paths
high_quality.to_excel('high_quality_paths.xlsx', index=False)

# Visualize
fc.VisualizeSelectedPaths(
    path_file='high_quality_paths.xlsx',
    output_folder='./high_quality_viz',
    network_layout='hierarchical'
)
```

### Example 3: Paths Through Specific Neurons

```python
# Find paths through specific intermediate neuron
intermediate_type = 'Mi1'

paths_through_mi1 = all_paths[
    all_paths['path_block'].str.contains(intermediate_type)
]

print(f"Found {len(paths_through_mi1)} paths through {intermediate_type}")

# Visualize
fc.VisualizeSelectedPaths(
    path_file=paths_through_mi1,  # Can pass DataFrame directly
    output_folder=f'./paths_via_{intermediate_type}',
    showfig=True
)
```

### Example 4: Custom Colors

```python
fc.VisualizeSelectedPaths(
    path_file='selected_paths.xlsx',
    output_folder='./custom_colors',
    node_color=['#FF6B6B', '#4ECDC4'],  # [source_color, intermediate_color]
    target_color='#FFE66D',              # Bright yellow for targets
    link_color='rgba(255,107,107,0.3)', # Semi-transparent red links
    network_layout='spring',
    showfig=True
)
```

### Example 5: Compare Different Path Sets

```python
# Scenario: Compare strong vs weak connections

# Strong connections (high weight)
strong_paths = all_paths[all_paths['min_weight'] > 50]
fc.VisualizeSelectedPaths(
    path_file=strong_paths,
    output_folder='./comparison/strong_connections',
    node_color=['#2E7D32', '#66BB6A'],  # Green theme
    target_color='#1B5E20'
)

# Weak connections (low weight but present)
weak_paths = all_paths[
    (all_paths['min_weight'] >= 10) & 
    (all_paths['min_weight'] <= 30)
]
fc.VisualizeSelectedPaths(
    path_file=weak_paths,
    output_folder='./comparison/weak_connections',
    node_color=['#1565C0', '#42A5F5'],  # Blue theme
    target_color='#0D47A1'
)
```

---

## Parameters

### `path_file` (Required)

**Type:** `str` or `pandas.DataFrame`

**Description:** Path to CSV/Excel file or DataFrame containing path data

**Formats:**
- CSV file: `'./my_paths.csv'`
- Excel file: `'./my_paths.xlsx'`
- DataFrame: Can pass DataFrame directly

**Example:**
```python
# From file
fc.VisualizeSelectedPaths(path_file='paths.csv')

# From DataFrame
selected = df[df['probability'] > 0.5]
fc.VisualizeSelectedPaths(path_file=selected)
```

---

### `sheet_name` (Optional)

**Type:** `str` or `None`

**Default:** `None` (auto-detect 'path_type' or use first sheet)

**Description:** Excel sheet name to read

**Options:**
- `'path_type'` - Type-level paths
- `'path_bodyId'` - BodyId-level paths
- Custom name

**Example:**
```python
fc.VisualizeSelectedPaths(
    path_file='results.xlsx',
    sheet_name='path_bodyId'
)
```

---

### `output_folder` (Optional)

**Type:** `str` or `None`

**Default:** `None` (uses `./selected_paths` in current path folder)

**Description:** Where to save visualization files

**Example:**
```python
fc.VisualizeSelectedPaths(
    path_file='paths.csv',
    output_folder='./publication_figures/pathway_X'
)
```

---

### `node_color` (Optional)

**Type:** `list` of 2 strings or `None`

**Default:** `['#1f77b4', '#2ca02c']` (blue, green)

**Format:** `[source_color, intermediate_color]`

**Description:** Colors for source and intermediate nodes

**Examples:**
```python
# Default blues and greens
node_color=['#1f77b4', '#2ca02c']

# Warm colors
node_color=['#FF6B6B', '#FFA500']

# Grayscale
node_color=['#333333', '#888888']

# Purple theme
node_color=['#9C27B0', '#BA68C8']
```

---

### `target_color` (Optional)

**Type:** `str` or `None`

**Default:** `'#d62728'` (red)

**Description:** Color for target nodes

**Examples:**
```python
target_color='#d62728'  # Red (default)
target_color='#FFD700'  # Gold
target_color='#FF1493'  # Deep pink
target_color='#8B0000'  # Dark red
```

---

### `link_color` (Optional)

**Type:** `str` or `None`

**Default:** `'rgba(100,100,100,0.3)'` (gray, semi-transparent)

**Description:** Color for connections in Sankey diagram

**Format:** RGB or RGBA

**Examples:**
```python
link_color='rgba(100,100,100,0.3)'    # Gray, 30% opacity
link_color='rgba(255,0,0,0.5)'        # Red, 50% opacity
link_color='rgba(0,128,255,0.4)'      # Blue, 40% opacity
```

---

### `network_layout` (Optional)

**Type:** `str`

**Default:** `'hierarchical'`

**Options:**
- `'hierarchical'` - Layer-based layout (best for directed paths)
- `'spring'` - Force-directed layout (organic)
- `'circular'` - Nodes arranged in a circle
- `'distributed'` - Kamada-Kawai layout (balanced)

**Examples:**
```python
# Best for sequential pathways
network_layout='hierarchical'

# Best for complex networks
network_layout='spring'

# Best for comparing connectivity
network_layout='circular'

# Best for balanced view
network_layout='distributed'
```

---

### `showfig` (Optional)

**Type:** `bool`

**Default:** `False`

**Description:** Whether to automatically open visualizations in browser

**Example:**
```python
showfig=True   # Open in browser
showfig=False  # Just save files
```

---

## Output Files

Each visualization creates 3 files in the output folder:

### 1. `sankey_selected_paths.html`
- **Type:** Interactive Sankey diagram
- **Features:**
  - Flow-based visualization
  - Node widths = number of connections
  - Link widths = synapse weights
  - Hover for details
  - Color-coded by node type

### 2. `network_selected_paths.html`
- **Type:** Interactive network graph (Cytoscape.js)
- **Features:**
  - Fully draggable nodes
  - Hover edges for weight/ratio/probability
  - Right-click to hide nodes
  - Export as PNG
  - Zoom and pan
  - Double-click to highlight connections

### 3. `selected_paths_connections.xlsx`
- **Type:** Excel file with 2 sheets
- **Sheets:**
  - `connections`: Aggregated connection data
  - `original_paths`: Your input paths

---

## Common Workflows

### Workflow 1: Publication-Ready Figures

```python
# 1. Run full analysis
fc.FindAllPath(forward_only=True)

# 2. Identify significant paths
paths = pd.read_excel('./path_results/path_type.xlsx', sheet_name='path_type')
significant = paths[paths['traversal_probability'] > 0.8]

# 3. Create clean visualization
fc.VisualizeSelectedPaths(
    path_file=significant,
    output_folder='./publication/figure_3',
    network_layout='hierarchical',
    showfig=True
)

# 4. Open network HTML, arrange nodes, export PNG
```

### Workflow 2: Progressive Filtering

```python
# Start broad, get narrower
paths = pd.read_excel('./results/path_type.xlsx', sheet_name='path_type')

# Level 1: All direct and 1-hop paths
short_paths = paths[paths['inter_layer_num'] <= 1]
fc.VisualizeSelectedPaths(short_paths, output_folder='./filter_1_short')

# Level 2: High-weight connections
strong = short_paths[short_paths['min_weight'] > 100]
fc.VisualizeSelectedPaths(strong, output_folder='./filter_2_strong')

# Level 3: Through specific neuron type
via_mi1 = strong[strong['path_block'].str.contains('Mi1')]
fc.VisualizeSelectedPaths(via_mi1, output_folder='./filter_3_via_mi1')
```

### Workflow 3: Compare Pathways

```python
# Compare pathways to different targets
sources = ['L3.*_R']
target_a = ['T4a.*_R']
target_b = ['T4b.*_R']

# Get paths for each target (from separate FindAllPath runs)
paths_to_a = pd.read_excel('./L3_to_T4a/path_type.xlsx', sheet_name='path_type')
paths_to_b = pd.read_excel('./L3_to_T4b/path_type.xlsx', sheet_name='path_type')

# Visualize with different colors
fc.VisualizeSelectedPaths(
    paths_to_a,
    output_folder='./comparison/to_T4a',
    target_color='#FF0000'  # Red for T4a
)

fc.VisualizeSelectedPaths(
    paths_to_b,
    output_folder='./comparison/to_T4b',
    target_color='#0000FF'  # Blue for T4b
)
```

---

## Tips and Best Practices

### 🎯 **Filtering Strategies**

1. **By Path Quality:**
   ```python
   high_quality = paths[
       (paths['traversal_probability'] > 0.5) &
       (paths['min_weight'] > 30)
   ]
   ```

2. **By Path Length:**
   ```python
   direct_or_short = paths[paths['inter_layer_num'] <= 2]
   ```

3. **By Specific Neurons:**
   ```python
   through_mi1 = paths[paths['path_block'].str.contains('Mi1')]
   ```

4. **By Weight Distribution:**
   ```python
   balanced = paths[
       paths['weights'].apply(lambda w: max(w) / min(w) < 3)
   ]
   ```

### 🎨 **Color Schemes**

**Scientific publications:**
- Source: `#1f77b4` (blue)
- Intermediate: `#2ca02c` (green)
- Target: `#d62728` (red)

**Warm theme:**
- Source: `#FF6B6B`
- Intermediate: `#FFA500`
- Target: `#FFD700`

**Cool theme:**
- Source: `#4A90E2`
- Intermediate: `#50E3C2`
- Target: `#B8E986`

### 📊 **Layout Selection**

- **Hierarchical:** Best for clear layer-by-layer pathways
- **Spring:** Best for visualizing network topology
- **Circular:** Best for comparing connectivity patterns
- **Distributed:** Best for balanced, aesthetically pleasing layouts

---

## Troubleshooting

### Error: "Missing required columns"

**Cause:** Input file doesn't have `path_block` or `weights` columns

**Solution:**
```python
# Check your columns
df = pd.read_excel('paths.xlsx')
print(df.columns)

# Ensure required columns exist
required = ['path_block', 'weights']
missing = [col for col in required if col not in df.columns]
print(f"Missing: {missing}")
```

### Error: "Invalid path format"

**Cause:** `path_block` doesn't follow `"A -> B -> C"` format

**Solution:**
```python
# Check path format
print(df['path_block'].iloc[0])

# Should be: "NodeA -> NodeB -> NodeC"
# NOT: "NodeA->NodeB->NodeC" (no spaces)
# NOT: "NodeA, NodeB, NodeC" (wrong separator)
```

### Empty visualization

**Cause:** All paths filtered out or no valid connections

**Solution:**
```python
# Check path count
print(f"Number of paths: {len(df)}")

# Check if weights are valid
print(df[['path_block', 'weights']].head())
```

---

## See Also

- [FindAllPath Documentation](../core-features/FindAllPath_Documentation.md)
- [Network Visualization Guide](Network_Guide.md)
- [Example Scripts](../../archive/examples/basic/Example_VisualizeSelectedPaths.py)
