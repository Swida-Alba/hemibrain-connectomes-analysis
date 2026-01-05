# VisualizePath - Quick Reference

## Import

```python
from vispath import VisualizePath, visualize_paths
```

## Basic Usage

### Method 1: Class Instance
```python
vp = VisualizePath('path_type.xlsx')
conn_df, G = vp.visualize()
```

### Method 2: One-Liner Function
```python
conn_df, G = visualize_paths('path_type.xlsx', showfig=True)
```

### Method 3: With DataFrame
```python
import pandas as pd
df = pd.read_excel('path_type.xlsx')
filtered = df[df['traversal_probability'] > 0.5]

vp = VisualizePath(filtered)
conn_df, G = vp.visualize()
```

## Common Workflows

### Filter by Quality
```python
paths = pd.read_excel('path_type.xlsx', sheet_name='path_type')

# High probability + short paths
quality = paths[
    (paths['traversal_probability'] > 0.5) &
    (paths['inter_layer_num'] <= 2)
]

vp = VisualizePath(quality, output_folder='./high_quality')
vp.visualize()
```

### Filter by Neuron Type
```python
# Paths through Mi1
via_mi1 = paths[paths['path_block'].str.contains('Mi1')]

vp = VisualizePath(via_mi1, output_folder='./via_mi1')
vp.visualize()
```

### Custom Colors
```python
vp = VisualizePath(
    'path_type.xlsx',
    node_color=['#FF6B6B', '#FFA500'],  # Source, Intermediate
    target_color='#FFD700',              # Target
    link_color='rgba(255,107,107,0.3)', # Links
    network_layout='spring',
    showfig=True
)
vp.visualize()
```

### Compare Path Sets
```python
# Strong connections
strong = paths[paths['min_weight'] > 100]
vp1 = VisualizePath(
    strong, 
    output_folder='./strong',
    node_color=['#2E7D32', '#66BB6A']  # Green
)
vp1.visualize()

# Weak connections  
weak = paths[(paths['min_weight'] >= 10) & (paths['min_weight'] <= 30)]
vp2 = VisualizePath(
    weak,
    output_folder='./weak', 
    node_color=['#1565C0', '#42A5F5']  # Blue
)
vp2.visualize()
```

## Parameters

| Parameter        | Type             | Default                   | Description                              |
| ---------------- | ---------------- | ------------------------- | ---------------------------------------- |
| `path_file`      | str or DataFrame | **Required**              | CSV/Excel file or DataFrame              |
| `sheet_name`     | str              | `None`                    | Excel sheet ('path_type', 'path_bodyId') |
| `output_folder`  | str              | `'./selected_paths'`      | Output directory                         |
| `node_color`     | list             | `['#1f77b4', '#2ca02c']`  | [source, intermediate]                   |
| `target_color`   | str              | `'#d62728'`               | Target node color                        |
| `link_color`     | str              | `'rgba(100,100,100,0.3)'` | Connection color                         |
| `network_layout` | str              | `'hierarchical'`          | Layout algorithm                         |
| `showfig`        | bool             | `False`                   | Auto-open in browser                     |

## Layout Options

| Layout           | Best For                | Description                |
| ---------------- | ----------------------- | -------------------------- |
| `'hierarchical'` | Sequential pathways     | Layer-by-layer arrangement |
| `'spring'`       | Complex networks        | Force-directed (organic)   |
| `'circular'`     | Connectivity comparison | Circular arrangement       |
| `'distributed'`  | Balanced view           | Kamada-Kawai (aesthetic)   |

## Color Schemes

### Scientific Publication
```python
node_color=['#1f77b4', '#2ca02c']  # Blue, Green
target_color='#d62728'              # Red
```

### Warm Theme
```python
node_color=['#FF6B6B', '#FFA500']  # Red, Orange
target_color='#FFD700'              # Gold
```

### Cool Theme
```python
node_color=['#4A90E2', '#50E3C2']  # Blue, Cyan
target_color='#B8E986'              # Light Green
```

### Grayscale
```python
node_color=['#333333', '#888888']  # Dark, Medium Gray
target_color='#000000'              # Black
```

## Output Files

Each run creates 3 files in `output_folder`:

| File                              | Type      | Features                               |
| --------------------------------- | --------- | -------------------------------------- |
| `sankey_selected_paths.html`      | Sankey    | Flow diagram, hover details            |
| `network_selected_paths.html`     | Cytoscape | Drag nodes, hide, hover, export PNG    |
| `selected_paths_connections.xlsx` | Excel     | 2 sheets: connections + original paths |

## Required Data Format

### CSV/Excel Columns

**Required:**
- `path_block` (str): `"A -> B -> C -> D"` format
- `weights` (list): `[150, 80, 45]` synapse counts

**Optional:**
- `connection_ratios` (list): `[0.25, 0.18, 0.12]`
- `traversal_probabilities` (list): `[0.85, 0.75, 0.65]`
- `nt_types` (list): `['ACH', 'GABA', 'GLUT']` (NT for each connection)

### Example DataFrame
```python
df = pd.DataFrame({
    'path_block': ['L3_R -> Mi1_R -> Tm3_R -> T4a_R'],
    'weights': [[150, 80, 45]],
    'connection_ratios': [[0.25, 0.18, 0.12]],
    'traversal_probabilities': [[0.85, 0.75, 0.65]],
    'nt_types': [['ACH', 'GABA', 'GLUT']]  # NEW: NT types
})
```

## Neurotransmitter (NT) Groups

### NT Edge Groups in Network
The network visualization automatically creates groups for each neurotransmitter type:
- **ACH Edges** - Acetylcholine (orange)
- **GABA Edges** - GABAergic (green)
- **GLUT Edges** - Glutamate (red)
- **DA Edges** - Dopamine (purple)
- **SER Edges** - Serotonin (blue)
- **OCT Edges** - Octopamine (teal)

### Using NT Groups
1. Open the network HTML in a browser
2. Expand "Group Selection" panel
3. Select an NT group (e.g., "ACH Edges")
4. Adjust color/opacity
5. Click "Apply to Group"

### NT Hover Display
Hovering over edges shows NT type with color coding:
```
Connection: A → B
Weight: 150 synapses
NT: ACH (displayed in orange)
```

## Methods

### Main Method
```python
conn_df, G = vp.visualize()
```
Creates all visualizations and returns (DataFrame, NetworkX graph)

### Individual Methods
```python
vp = VisualizePath('path_type.xlsx')

# Build network only
conn_df, G = vp.build_network()

# Create specific visualizations
vp.create_sankey()
vp.create_network()
vp.save_data()
```

## Common Filters

### By Probability
```python
high_prob = paths[paths['traversal_probability'] > 0.5]
```

### By Weight
```python
strong = paths[paths['min_weight'] > 100]
```

### By Path Length
```python
short = paths[paths['inter_layer_num'] <= 2]
```

### By Intermediate Neuron
```python
via_mi1 = paths[paths['path_block'].str.contains('Mi1')]
```

### Combined
```python
filtered = paths[
    (paths['traversal_probability'] > 0.5) &
    (paths['min_weight'] > 30) &
    (paths['inter_layer_num'] <= 2)
]
```

## Errors

### "Missing required columns"
```python
# Check columns
print(df.columns)

# Ensure path_block and weights exist
required = ['path_block', 'weights']
```

### "Invalid path format"
```python
# Use space-arrow-space separator
path_block = "A -> B -> C"  # ✓ Correct
path_block = "A->B->C"      # ✗ Wrong
```

### Empty visualization
```python
# Check path count
print(f"Paths: {len(df)}")

# Verify data
print(df[['path_block', 'weights']].head())
```

## Tips

1. **Filter before visualize** - Load all paths, filter, then visualize
2. **Use consistent colors** - Define color schemes as constants
3. **Start hierarchical** - Best for understanding flow
4. **Export networks as PNG** - Right-click in browser or use Export button
5. **Hide cluttered nodes** - Right-click or press 'H' in network view

## Examples

Full examples: `Example_VisualizeSelectedPaths_Standalone.py`

## Documentation

- **Full Guide**: [VisualizeSelectedPaths_Guide.md](./VisualizeSelectedPaths_Guide.md)
- **Architecture**: [VisualizePath_Architecture.md](./VisualizePath_Architecture.md)
- **Main README**: [../README.md](../README.md)

## vs FindNeuronConnection

| Feature        | VisualizePath (Standalone)          | fc.VisualizeSelectedPaths()              |
| -------------- | ----------------------------------- | ---------------------------------------- |
| Import         | `from vispath import VisualizePath` | `from coana import FindNeuronConnection` |
| Initialization | Light (no token/dataset)            | Heavy (requires FC init)                 |
| Use Case       | Post-analysis visualization         | During analysis workflow                 |
| Performance    | Fast                                | Slightly slower                          |
| Recommendation | ✅ Use this                          | Only if FC already initialized           |

## Quick Comparison

### Old Way (≤ v2.0)
```python
# Had to initialize full FC
fc = FindNeuronConnection(token='...', dataset='...', ...)
fc.VisualizeSelectedPaths('path_type.xlsx')
```

### New Way (v2.1+)
```python
# Standalone, no FC needed
from vispath import visualize_paths
conn_df, G = visualize_paths('path_type.xlsx')
```

**Recommendation:** Use standalone `VisualizePath` for all new projects!
