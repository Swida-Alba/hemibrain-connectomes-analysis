# Custom Colors Guide for VisualizePath

This guide explains how to customize node and edge colors in Sankey diagrams and network visualizations.

## Overview

VisualizePath now supports custom colors for both **nodes** and **edges** through:
- **Node Colors**: Custom colors for specific nodes via `node_colors` parameter
- **Edge Colors**: Custom colors for specific edges via optional `color` column in edge-list format

## Node Colors

### Supported Input Formats

The `node_colors` parameter accepts three formats:

#### 1. DataFrame
```python
import pandas as pd
from vispath import VisualizePath

# Create node color mapping
node_colors = pd.DataFrame({
    'node': ['A', 'B', 'C', 'D', 'E'],
    'color': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
})

# Create visualization
vp = VisualizePath(
    path_file='network.csv',
    node_colors=node_colors  # Pass DataFrame
)
vp.visualize()
```

#### 2. CSV/Excel File Path
```python
# Save colors to CSV
node_colors.to_csv('node_colors.csv', index=False)

# Use CSV file
vp = VisualizePath(
    path_file='network.csv',
    node_colors='node_colors.csv'  # Pass file path
)
vp.visualize()
```

#### 3. Excel Sheet Name
```python
# If colors are in a sheet within the same Excel file
vp = VisualizePath(
    path_file='network.xlsx',
    sheet_name='edges',
    node_colors='node_colors'  # Sheet name in same Excel file
)
vp.visualize()
```

### Node Color File Format

The node colors file/DataFrame must have two columns (case-insensitive):
- `node`: Node name (must match node names in your network data)
- `color`: Color in hex or rgba format

**Example CSV:**
```csv
node,color
Input1,#FF6B6B
Process1,rgba(78, 205, 196, 0.8)
Output1,#45B7D1
```

**Example Excel:**
| Node     | Color                     |
|----------|---------------------------|
| Input1   | #FF6B6B                   |
| Process1 | rgba(78, 205, 196, 0.8)  |
| Output1  | #45B7D1                   |

### Color Format Support

Both **hex** and **rgba** formats are supported:

**Hex Format:**
```python
'#FF6B6B'  # Red
'#4ECDC4'  # Teal
'#45B7D1'  # Blue
```

**RGBA Format (with opacity):**
```python
'rgba(255, 107, 107, 0.6)'  # Red with 60% opacity
'rgba(78, 205, 196, 0.8)'   # Teal with 80% opacity
'rgba(69, 183, 209, 1.0)'   # Blue with 100% opacity
```

## Edge Colors

### Edge-List Format with Color Column

When using edge-list format, you can add an optional `color` column:

```python
import pandas as pd

network = pd.DataFrame({
    'source': ['A', 'A', 'B', 'B', 'C', 'C'],
    'target': ['B', 'C', 'D', 'E', 'D', 'E'],
    'weight': [10, 15, 8, 12, 6, 9],
    'color': [
        'rgba(255, 107, 107, 0.6)',  # A->B: red
        'rgba(78, 205, 196, 0.6)',   # A->C: teal
        '#45B7D1',                    # B->D: blue (hex format)
        'rgba(255, 160, 122, 0.7)',  # B->E: orange
        '#98D8C8',                    # C->D: green
        '#FFD93D'                     # C->E: yellow
    ]
})

vp = VisualizePath(path_file=network)
vp.visualize()
```

### Edge Color Format

Edge colors support the same formats as node colors:
- **Hex**: `#FF6B6B`
- **RGBA**: `rgba(255, 107, 107, 0.6)`

## Combined Node and Edge Colors

You can use both custom node and edge colors together:

```python
# Network with edge colors
network = pd.DataFrame({
    'source': ['A', 'B', 'C'],
    'target': ['B', 'C', 'D'],
    'weight': [10, 15, 8],
    'color': ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Edge colors
})

# Custom node colors
node_colors = pd.DataFrame({
    'node': ['A', 'B', 'C', 'D'],
    'color': ['#FF1744', '#00897B', '#7B1FA2', '#FF6F00']
})

# Create visualization with both
vp = VisualizePath(
    path_file=network,
    node_colors=node_colors
)
vp.visualize()
```

## Color Precedence

### Node Colors
1. **Custom colors** (from `node_colors` parameter) - highest priority
2. **Default colors** by node type:
   - Source nodes: `source_color` (default: '#1f77b4' blue)
   - Intermediate nodes: `intermediate_color` (default: '#2ca02c' green)
   - Target nodes: `target_color` (default: '#d62728' red)

### Edge Colors
1. **Custom colors** (from `color` column) - highest priority
2. **Default color**: `link_color` (default: 'rgba(100,100,100,0.3)' gray)

## Complete Example

```python
import pandas as pd
from vispath import VisualizePath

# Create network with edge colors
network = pd.DataFrame({
    'source': ['Input1', 'Input1', 'Process1', 'Process2'],
    'target': ['Process1', 'Process2', 'Output1', 'Output1'],
    'weight': [100, 150, 80, 70],
    'color': [
        'rgba(255, 107, 107, 0.6)',
        'rgba(78, 205, 196, 0.6)',
        'rgba(69, 183, 209, 0.6)',
        'rgba(255, 193, 7, 0.6)'
    ]
})

# Custom node colors
node_colors = pd.DataFrame({
    'node': ['Input1', 'Process1', 'Process2', 'Output1'],
    'color': [
        '#FF1744',                    # Red for input
        'rgba(0, 150, 136, 0.9)',    # Teal for process 1
        '#7B1FA2',                    # Purple for process 2
        'rgba(33, 150, 243, 0.8)'    # Blue for output
    ]
})

# Create visualization
vp = VisualizePath(
    path_file=network,
    node_colors=node_colors,
    output_folder='./custom_viz',
    showfig=True
)

conn_df, G = vp.visualize()
```

## Tips and Best Practices

### 1. Color Consistency
- Use a consistent color scheme across your visualizations
- Consider using opacity (0.3-0.8) for edges to avoid visual clutter

### 2. Accessibility
- Ensure sufficient contrast between colors
- Avoid red-green combinations for colorblind accessibility
- Test with different backgrounds (the Sankey has white background)

### 3. Semantic Colors
- Use meaningful colors (e.g., red for inhibitory, blue for excitatory)
- Keep important nodes in high-contrast colors

### 4. File Organization
- Store color schemes in separate CSV files for reusability
- Use Excel sheets to keep network data and colors together

### 5. Testing Colors
```python
# Test with opacity gradient
test_network = pd.DataFrame({
    'source': ['Input'] * 5,
    'target': ['A', 'B', 'C', 'D', 'E'],
    'weight': [10] * 5,
    'color': [
        'rgba(255, 0, 0, 0.2)',
        'rgba(255, 0, 0, 0.4)',
        'rgba(255, 0, 0, 0.6)',
        'rgba(255, 0, 0, 0.8)',
        'rgba(255, 0, 0, 1.0)'
    ]
})
```

## Validation and Error Handling

### Invalid Color Format
```python
# ⚠️ Invalid color will be skipped with warning
node_colors = pd.DataFrame({
    'node': ['A', 'B'],
    'color': ['#FF6B6B', 'invalid_color']  # Second color invalid
})
# Output: "⚠️ Warning: Invalid color format for node 'B': invalid_color. Skipping."
```

### Missing Nodes
- If a node in `node_colors` doesn't exist in the network, it's ignored
- If a node in the network isn't in `node_colors`, it uses default colors

### Column Name Case
The column names are case-insensitive:
- `node` = `Node` = `NODE`
- `color` = `Color` = `COLOR`

## See Also

- `test_custom_colors.py` - Comprehensive test examples
- `SIMPLE_INPUT_FORMAT.md` - Edge-list format documentation
- `SANKEY_ENHANCEMENT_SUMMARY.md` - Sankey features
- `README.md` - Main documentation

## Version

Custom color support added: October 2025
- Node colors: DataFrame, CSV, Excel sheet
- Edge colors: Optional column in edge-list format
- Support: hex and rgba formats with opacity
