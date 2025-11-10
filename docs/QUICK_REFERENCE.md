# Quick Reference - Using vispath_pkg

## Import Pattern

```python
import sys
from pathlib import Path

# Add vispath-subproject to Python path
vispath_pkg_path = Path(__file__).parent.parent / 'vispath-subproject' / 'src'
if vispath_pkg_path.exists():
    sys.path.insert(0, str(vispath_pkg_path))

# Import what you need
from vispath_pkg import VisualizePath, VisConnMatInteractive
```

## Available Functions

### 1. VisualizePath (Class)
Main class for pathway visualization.

```python
vp = VisualizePath(
    conn_df=connections_df,           # DataFrame with source, target, weight
    source_names=['TypeA'],           # Source neuron types
    target_names=['TypeB'],           # Target neuron types
    output_folder='./output',         # Where to save files
    source_color='#1f77b4',          # Source node color
    intermediate_color='rgba(44,160,44,0.2)',  # Intermediate color
    target_color='#d62728',          # Target node color
    link_color='rgba(100,100,100,0.3)'  # Edge color
)

# Generate visualization
vp.visualize()
```

### 2. visualize_paths (Function)
Quick visualization without creating class instance.

```python
from vispath_pkg import visualize_paths

visualize_paths(
    path_df=paths_df,                 # DataFrame with path information
    output_folder='./output',
    output_name='my_paths',
    # ... other parameters same as VisualizePath
)
```

### 3. VisConnMatInteractive (Function)
Create interactive heatmaps with comprehensive controls.

```python
from vispath_pkg import VisConnMatInteractive

# Simple usage (single matrix)
VisConnMatInteractive(
    cmat=connection_matrix,           # pandas DataFrame (NxN)
    filename='heatmap.html',
    title='My Connection Matrix',
    showfig=True                      # Open in browser
)

# Advanced usage (multiple metrics)
matrices_dict = {
    'weight': weight_matrix,          # Synapse counts
    'ratio': ratio_matrix,            # Connection ratios
    'probability': prob_matrix        # Traversal probabilities
}

VisConnMatInteractive(
    cmat=weight_matrix,               # Default matrix to show
    filename='heatmap_multi.html',
    title='Multi-Metric Heatmap',
    matrices_dict=matrices_dict,      # Enable metric toggle
    conn_df=connections_df,           # For enhanced hover labels
    showfig=True
)
```

### 4. parse_color_to_hex_opacity (Function)
Convert various color formats to hex + opacity.

```python
from vispath_pkg import parse_color_to_hex_opacity

hex_color, opacity = parse_color_to_hex_opacity('rgba(255, 0, 0, 0.5)')
# Returns: ('#ff0000', 0.5)
```

## Data Formats

### Connection DataFrame Format
```python
import pandas as pd

conn_df = pd.DataFrame({
    'bodyId_pre': [123, 456, 789],      # Source neuron IDs
    'type_pre': ['TypeA', 'TypeB', 'TypeC'],
    'bodyId_post': [456, 789, 123],     # Target neuron IDs
    'type_post': ['TypeB', 'TypeC', 'TypeA'],
    'weight': [10, 5, 8],               # Synapse counts
    'roi': ['MB(R)', 'MB(R)', 'AL(R)']  # Brain region
})
```

### Path DataFrame Format
```python
path_df = pd.DataFrame({
    'path_id': [1, 1, 2, 2],
    'source': ['A', 'A', 'C', 'C'],
    'layer_1': ['B', 'B', 'D', 'D'],
    'target': ['C', 'C', 'E', 'E'],
    'weight': [10, 10, 5, 5]
})
```

### Connection Matrix Format
```python
# NxN pandas DataFrame with neuron names as index and columns
import pandas as pd

matrix = pd.DataFrame(
    data=[[0, 10, 5],
          [2, 0, 8],
          [3, 6, 0]],
    index=['NeuronA', 'NeuronB', 'NeuronC'],
    columns=['NeuronA', 'NeuronB', 'NeuronC']
)
```

## Common Use Cases

### 1. Visualize Simple Network
```python
from vispath_pkg import VisualizePath

vp = VisualizePath(
    conn_df=my_connections,
    source_names=['DA1'],
    target_names=['MBON'],
    output_folder='./results'
)
vp.visualize()
```

### 2. Create Heatmap from Matrix
```python
from vispath_pkg import VisConnMatInteractive

VisConnMatInteractive(
    cmat=connection_matrix,
    filename='heatmap.html',
    title='Connection Strength Matrix',
    color_scale=[[0, 'rgb(255,255,255)'], [1, 'rgb(14,83,13)']],
    showfig=True
)
```

### 3. Multi-Metric Heatmap
```python
matrices = {
    'weight': synapse_counts,
    'ratio': connection_ratios,
    'probability': traversal_probs
}

VisConnMatInteractive(
    cmat=synapse_counts,
    filename='multi_metric.html',
    title='Multi-Metric Analysis',
    matrices_dict=matrices,
    showfig=True
)
```

### 4. Custom Colors for Pathways
```python
vp = VisualizePath(
    conn_df=my_connections,
    source_names=['Input'],
    target_names=['Output'],
    output_folder='./viz',
    source_color='#FF6B6B',           # Red
    intermediate_color='#FFA500',     # Orange
    target_color='#FFD700',           # Gold
    link_color='rgba(255,107,107,0.3)'
)
vp.visualize()
```

## Color Formats Supported

```python
# Hex colors
'#FF0000'

# RGB
'rgb(255, 0, 0)'

# RGBA (with opacity)
'rgba(255, 0, 0, 0.5)'

# Named colors (HTML/CSS)
'red', 'blue', 'green', etc.
```

## Output Files

### VisualizePath Generates:
- `{output_name}_network.html` - Interactive network graph
- `{output_name}_sankey.html` - Sankey diagram (if applicable)
- `{output_name}_data.xlsx` - Raw data export

### VisConnMatInteractive Generates:
- `{filename}.html` - Standalone interactive heatmap
  - No external dependencies
  - Plotly.js loaded from CDN
  - Settings saved to browser localStorage

## Tips

1. **Large Networks**: Use `intermediate_color` with low opacity (e.g., 0.2) for clarity
2. **Heatmaps**: Use Log₂ or Log₁₀ scale for large dynamic ranges
3. **Colors**: Use contrasting colors for source/target (e.g., blue/red)
4. **File Size**: For huge heatmaps (>500 nodes), expect larger HTML files
5. **Browser**: Chrome/Firefox recommended for best interactive performance

## Error Handling

```python
try:
    from vispath_pkg import VisualizePath
except ImportError:
    print("Make sure vispath-subproject/src is in your Python path!")
    import sys
    from pathlib import Path
    vispath_pkg_path = Path(__file__).parent.parent / 'vispath-subproject' / 'src'
    sys.path.insert(0, str(vispath_pkg_path))
    from vispath_pkg import VisualizePath
```

## Documentation

- **Full README**: `vispath-subproject/README.md`
- **Installation**: `vispath-subproject/INSTALLATION.md`
- **Examples**: `examples/` directory
- **Scripts**: `scripts/` directory

---

**Last Updated**: November 7, 2025  
**Package Version**: 1.0.0  
**Status**: ✅ Production Ready
