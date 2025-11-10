# Quick Reference: Custom Colors in VisualizePath

## Basic Usage

### 1. Custom Node Colors (DataFrame)
```python
import pandas as pd
from vispath import VisualizePath

# Define node colors
node_colors = pd.DataFrame({
    'node': ['A', 'B', 'C'],
    'color': ['#FF6B6B', '#4ECDC4', '#45B7D1']
})

# Create visualization
vp = VisualizePath(
    path_file='network.csv',
    node_colors=node_colors
)
vp.visualize()
```

### 2. Custom Node Colors (CSV File)
```python
# Save to CSV
node_colors.to_csv('colors.csv', index=False)

# Use file path
vp = VisualizePath(
    path_file='network.csv',
    node_colors='colors.csv'
)
vp.visualize()
```

### 3. Custom Node Colors (Excel Sheet)
```python
vp = VisualizePath(
    path_file='network.xlsx',
    sheet_name='edges',
    node_colors='node_colors'  # Sheet in same Excel file
)
vp.visualize()
```

### 4. Custom Edge Colors
```python
network = pd.DataFrame({
    'source': ['A', 'B'],
    'target': ['B', 'C'],
    'weight': [10, 15],
    'color': ['#FF6B6B', 'rgba(78, 205, 196, 0.6)']  # Optional column
})

vp = VisualizePath(path_file=network)
vp.visualize()
```

### 5. Combined Colors
```python
# Network with edge colors
network = pd.DataFrame({
    'source': ['A', 'B'],
    'target': ['B', 'C'],
    'weight': [10, 15],
    'color': ['#FF6B6B', '#4ECDC4']
})

# Custom node colors
node_colors = pd.DataFrame({
    'node': ['A', 'B', 'C'],
    'color': ['#FF1744', '#00897B', '#7B1FA2']
})

# Use both
vp = VisualizePath(
    path_file=network,
    node_colors=node_colors
)
vp.visualize()
```

## Color Formats

### Hex (Solid)
```python
'#FF6B6B'  # Red
'#4ECDC4'  # Teal
'#45B7D1'  # Blue
```

### RGBA (With Opacity)
```python
'rgba(255, 107, 107, 0.6)'  # Red, 60% opacity
'rgba(78, 205, 196, 0.8)'   # Teal, 80% opacity
'rgba(69, 183, 209, 1.0)'   # Blue, 100% opacity
```

## File Formats

### Node Colors CSV
```csv
node,color
Input1,#FF6B6B
Process1,rgba(78, 205, 196, 0.8)
Output1,#45B7D1
```

### Network with Edge Colors CSV
```csv
source,target,weight,color
A,B,10,#FF6B6B
B,C,15,rgba(78, 205, 196, 0.6)
```

## Common Patterns

### By Neuron Type
```python
node_colors = pd.DataFrame({
    'node': ['DA1', 'GABA1', 'ACh1'],
    'color': [
        'rgba(255, 0, 0, 0.8)',    # Dopaminergic: Red
        'rgba(0, 0, 255, 0.8)',    # GABAergic: Blue
        'rgba(0, 255, 0, 0.8)'     # Cholinergic: Green
    ]
})
```

### By Layer
```python
node_colors = pd.DataFrame({
    'node': ['Input1', 'Hidden1', 'Output1'],
    'color': ['#1976D2', '#FFA726', '#66BB6A']
})
```

### Gradient Effect
```python
network = pd.DataFrame({
    'source': ['A'] * 5,
    'target': ['B1', 'B2', 'B3', 'B4', 'B5'],
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

## Tips

1. **Use opacity (0.3-0.8) for edges** to reduce visual clutter
2. **Keep node colors solid (1.0)** for better visibility
3. **Test colors** on white background (Sankey default)
4. **Case doesn't matter**: 'node' = 'Node' = 'NODE'
5. **Missing nodes** use default colors (no error)
6. **Invalid colors** are skipped with warning

## Testing

```bash
# Run comprehensive tests
python test_custom_colors.py

# Results in: test_output/custom_colors/
```

## See Also

- `CUSTOM_COLORS_GUIDE.md` - Full documentation
- `test_custom_colors.py` - Working examples
- `SANKEY_LAYOUT_COLORS_SUMMARY.md` - Implementation details
