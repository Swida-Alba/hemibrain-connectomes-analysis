# Simple Input Format Guide

## Overview

The `VisualizePath` class now supports **simple edge-list input format** with just three columns: source, target, and weight. This makes it easy to visualize any network data without requiring the complex path_block format.

## Minimum Requirements

Only **3 columns** are needed:
1. **Source node** (starting point)
2. **Target node** (ending point)
3. **Weight** (connection strength, synapse count, etc.)

## Supported Column Names

The system automatically detects columns using flexible naming:

### Source Column
Any of these names will be recognized:
- `source`
- `from`
- `pre`
- **ANY** `[prefix]_pre` format (e.g., `bodyId_pre`, `neuron_pre`, `type_pre`, `custom_name_pre`)

**Examples:** `bodyId_pre`, `type_pre`, `neuron_pre`, `instance_pre`, `synapse_pre`, etc.

### Target Column
Any of these names will be recognized:
- `target`
- `to`
- `post`
- **ANY** `[prefix]_post` format (e.g., `bodyId_post`, `neuron_post`, `type_post`, `custom_name_post`)

**Examples:** `bodyId_post`, `type_post`, `neuron_post`, `instance_post`, `synapse_post`, etc.

### Weight Column
Any of these names will be recognized:
- `weight`
- `weights`
- `synapse_count`
- `count`

## Example Formats

### Format 1: Simple source/target/weight

**CSV file:**
```csv
source,target,weight
A,B,10
B,C,20
C,D,15
```

**Python:**
```python
from vispath import VisualizePath
import pandas as pd

df = pd.DataFrame({
    'source': ['A', 'B', 'C'],
    'target': ['B', 'C', 'D'],
    'weight': [10, 20, 15]
})

vis = VisualizePath(path_file=df, output_folder='./output')
vis.create_network()
vis.create_sankey()
```

### Format 2: BodyId format

**CSV file:**
```csv
bodyId_pre,bodyId_post,weight
123456,234567,25
234567,345678,30
345678,456789,20
```

**Python:**
```python
vis = VisualizePath(path_file='bodyid_network.csv')
vis.create_network()
```

### Format 3: Neuron format

**CSV file:**
```csv
neuron_pre,neuron_post,synapse_count
DA1_PN,LHON1,150
LHON1,MBON_a1,200
MBON_a1,DAN,100
```

**Python:**
```python
vis = VisualizePath(path_file='neuron_network.csv')
vis.create_sankey()
```

### Format 4: From/To format

**CSV file:**
```csv
from,to,weight
KC_a,MBON_a,100
KC_b,MBON_a,80
MBON_a,DAN,50
```

**Python:**
```python
vis = VisualizePath(path_file='fromto_network.csv')
vis.create_network()
```

## Usage Examples

### Example 1: Load from DataFrame

```python
from vispath import VisualizePath
import pandas as pd

# Create edge data
edges = pd.DataFrame({
    'source': ['PN1', 'PN1', 'PN2', 'LN1'],
    'target': ['LN1', 'LN2', 'LN1', 'MBON1'],
    'weight': [50, 30, 40, 25]
})

# Visualize
vis = VisualizePath(
    path_file=edges,
    output_folder='./output',
    source_color='#3498db',
    target_color='#e74c3c'
)

vis.create_network()  # Creates network_selected_paths.html
vis.create_sankey()   # Creates sankey_selected_paths.html
```

### Example 2: Load from CSV file

```python
from vispath import VisualizePath

# Load and visualize
vis = VisualizePath(path_file='my_network.csv')
vis.create_network()
vis.create_sankey()
```

### Example 3: Load from Excel file

```python
from vispath import VisualizePath

# System will auto-detect the correct sheet
vis = VisualizePath(path_file='my_network.xlsx')
vis.create_network()
```

### Example 4: Customize colors and appearance

```python
from vispath import VisualizePath

vis = VisualizePath(
    path_file='network.csv',
    output_folder='./results',
    
    # Node colors (hex or RGBA)
    source_color='#3498db',              # Blue
    target_color='#e74c3c',              # Red
    intermediate_color='#2ecc71',        # Green
    
    # Edge color and opacity
    link_color='rgba(100, 100, 100, 0.4)',  # Semi-transparent gray
    
    # Sizes
    node_size=30,
    font_size=12
)

vis.create_network()
vis.create_sankey()
```

## Automatic Format Detection

The system automatically detects whether your data is:
1. **Path-based format** (original): Contains `path_block` and `weights` columns
2. **Edge-list format** (new): Contains source/target/weight columns

You don't need to specify the format - it's detected automatically!

## Output Files

Both visualization methods create interactive HTML files:

### Network Visualization (Cytoscape.js)
- **File**: `network_selected_paths.html`
- **Features**: Interactive, draggable nodes, layered layout, zoom/pan

### Sankey Diagram (Plotly)
- **File**: `sankey_selected_paths.html`
- **Features**: Multi-layer flow diagram, interactive controls, color/opacity sliders

## Error Messages

If column names are not recognized, you'll get a helpful error message:

```
ValueError: Data must contain either:
  1. Path-based format: 'path_block' and 'weights' columns
  2. Edge-list format: source, target, and weight columns
  
  Recognized column names:
  - Source: source, from, pre, *_pre (e.g., bodyId_pre)
  - Target: target, to, post, *_post (e.g., bodyId_post)
  - Weight: weight, weights, synapse_count, count
  
  Your columns: ['col1', 'col2', 'col3']
```

## Complete Examples

See these example files:
- **[Example_SimpleEdgeList.py](Example_SimpleEdgeList.py)** - Comprehensive examples with all formats
- **[example_simple_network.csv](example_simple_network.csv)** - Simple source/target/weight
- **[example_bodyid_network.csv](example_bodyid_network.csv)** - BodyId format
- **[example_neuron_network.csv](example_neuron_network.csv)** - Neuron format

## Advanced: Original Path-Based Format

The original format is still fully supported for complex pathway analysis:

```python
df = pd.DataFrame({
    'path_block': ['A -> B -> C', 'A -> D -> C'],
    'weights': [[10, 20], [15, 25]],
    'connection_ratios': [[0.5, 0.8], [0.6, 0.7]],
    'layer': [[0, 1, 2], [0, 1, 2]]
})

vis = VisualizePath(path_file=df)
vis.create_network()
```

Both formats work seamlessly with all visualization features!

## Visualization Behavior

### Sankey Diagram Layered Layout

The Sankey diagram automatically creates a **layered layout** based on network topology:

**How it works:**
1. Analyzes the network structure from edge-list data
2. Identifies **source nodes** (only outgoing edges), **target nodes** (only incoming edges), and **intermediate nodes** (both)
3. Creates visual layers: Sources (left) → Intermediates (middle) → Targets (right)
4. Shows proper flow connections between layers

**Example:**
```python
# Input: Simple edges
df = pd.DataFrame({
    'source': ['A', 'A', 'B', 'B'],
    'target': ['B', 'C', 'D', 'D'],
    'weight': [10, 5, 8, 12]
})

# Result: Layered Sankey diagram
# Layer 0 (Source):  A
# Layer 1 (Intermediate): B, C
# Layer 2 (Target): D
#
# Visual:  A ━━━ B ━━━ D
#           ╲   ╱
#            ━C━
```

**Result:** Clear layered flow, NOT isolated connections! ✅

### Network Graph

The network graph shows:
- Color-coded nodes by type (source=blue, intermediate=green, target=red)
- Layered vertical layout (sources at top, targets at bottom)
- Interactive dragging and zooming
- Connection weights shown on edges

## Tips

1. **Column naming**: Use standard names (`source`, `target`, `weight`) for best compatibility
2. **Excel files**: The system will prompt you to select/confirm the correct sheet
3. **File picker**: A fast GUI dialog (PyQt5) will open for file selection
4. **Customization**: All color, size, and layout options work with both formats
5. **Performance**: For large networks (>1000 edges), consider filtering the data first

## Troubleshooting

**Q: My columns aren't recognized**
- Make sure column names match one of the supported variants
- Check for typos or extra spaces in column names
- See the error message for what columns were found

**Q: Can I use custom column names?**
- No, but you can rename your columns before loading:
  ```python
  df.rename(columns={'my_source': 'source', 'my_target': 'target'}, inplace=True)
  ```

**Q: Does this work with Excel files?**
- Yes! Both `.xlsx` and `.xls` formats are supported
- The system will auto-detect or ask you to select the correct sheet

**Q: Can I mix formats?**
- No, choose one format per file (either path-based or edge-list)
- But you can use different formats for different visualizations

## Related Documentation

- **[VisualizePath Guide](docs/VisualizeSelectedPaths_Guide.md)** - Complete class documentation
- **[File Support](VISPATH_FILE_SUPPORT.md)** - CSV/Excel file handling
- **[Installation Guide](INSTALLATION.md)** - Setup and requirements
