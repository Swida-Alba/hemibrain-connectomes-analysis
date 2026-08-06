# Heatmap Visualization Guide

Interactive connection matrix heatmaps for visualizing neuron-to-neuron connectivity patterns.

## Overview

The heatmap visualization displays connectivity data as a color-coded matrix where:
- **Rows** represent source neurons (presynaptic)
- **Columns** represent target neurons (postsynaptic)
- **Cell colors** represent connection strength (synapse count, ratio, or probability)
- **Interactive controls** allow real-time exploration and customization

## Example Input Data

### Required Format

Your connection dataframe should contain these columns:

| Column | Description | Example Values |
|--------|-------------|----------------|
| `source` or `sourcetype` | Source neuron names/types | 'KC_alpha', 'MBON01', 'PPL1-01' |
| `target` or `targettype` | Target neuron names/types | 'MBON03', 'MBON14', 'DAN' |
| `weight` | Synapse count (required) | 10, 25, 150 |
| `ratio` | Connection ratio (optional) | 0.05, 0.15, 0.8 |
| `probability` | Traversal probability (optional) | 0.1, 0.5, 0.9 |

### Example Input Files

**From FindAllPath results** (recommended):
- Path: `connection_data/*/direct_connections.xlsx` or `*_connections.csv`
- Example: [`examples/sample_network_data.csv`](../../archive/examples/data/sample_network_data.csv)

**From Custom Analysis**:
```python
import pandas as pd

# Example: Create simple connection matrix
conn_df = pd.DataFrame({
    'source': ['KC_a', 'KC_a', 'KC_b', 'KC_b'],
    'target': ['MBON01', 'MBON03', 'MBON01', 'MBON14'],
    'weight': [25, 10, 50, 15],
    'ratio': [0.1, 0.05, 0.2, 0.08],
    'probability': [0.3, 0.15, 0.5, 0.2]
})
```

**Sample datasets provided**:
- `datasets/hemibrain_v1_2_1_alltypes_neuron_df.csv` - All neuron types with metadata
- Output from `scripts/FindDirect.py` - Direct connections between neuron populations

## Quick Start

### Basic Usage

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

## Key Features

### 1. Multiple Scales

Transform data for better visualization:

- **Linear**: Original values (default)
- **Logarithmic (log₂)**: Emphasize smaller differences, reduce large value dominance
- **Logarithmic (log₁₀)**: Alternative log base for different dynamic range
- **Square Root**: Moderate compression of large values

**Usage**: Click scale buttons at the top of the interface

### 2. Hierarchical Clustering

Automatically group similar connection patterns:

- **Ward Linkage** (default): Creates compact, well-separated clusters
- **Average Linkage**: Balanced approach between compactness and connectivity
- **Complete Linkage**: Produces tight, compact clusters
- **Single Linkage**: Can reveal hierarchical structures, may produce elongated clusters

**Usage**: 
1. Click "Clustered Order" button
2. Select clustering method from dropdown
3. Toggle back to "Original Order" anytime

**Note**: Clustering uses Euclidean distance on connection patterns

### 3. Custom Colorscales

Choose from 15+ built-in colorscales or create custom ones:

**Built-in Colorscales**:
- Sequential: Blues, Greens, Reds, Purples, Oranges, Greys
- Diverging: RdBu (Red-Blue), PiYG (Pink-Yellow-Green)
- Perceptually uniform: Viridis, Plasma, Inferno, Magma, Cividis
- Specialized: Turbo, Jet

**Custom 2-Color Scale**:
1. Select "Custom" from colorscale dropdown
2. Choose minimum color (e.g., white for low values)
3. Choose maximum color (e.g., dark purple for high values)

**Custom 3-Color Scale**:
1. Enable "3-Color Scale" checkbox
2. Set min/mid/max colors
3. Set corresponding data values for each color point

### 4. Interactive Reordering

Drag-and-drop rows and columns to explore patterns:

1. Click "Reorder" button to open reorder panel
2. Drag items in the list to new positions
3. Changes apply instantly to the heatmap
4. Close panel when done

**Tip**: Works in both original and clustered modes

### 5. Cell Values Display

Show exact values on each cell:

1. Click "Show Values" button
2. Adjust value font size with slider (6-20px)
3. Set contrast threshold to auto-adjust text color for readability
4. Use "Reverse Contrast" if needed

**Note**: Values show original data regardless of scale transformation

### 6. Custom Color Ranges

Fine-tune the color mapping:

- **Auto Range** (default): Uses data min/max
- **Manual Range**: Set custom zmin/zmax values
  - Useful for comparing multiple heatmaps
  - Clip outliers for better mid-range visibility

### 7. Data Filtering

Hide specific values from display:

**Syntax**: Enter comma-separated rules
- Exact value: `0`
- Less than: `<5`
- Greater than: `>100`
- Range: `>=10, <=20`

**Example**: `0, <3, >1000` hides zero, values below 3, and above 1000

### 8. Transpose Matrix

Swap rows and columns:
- Click "Transpose" button
- Useful for different analysis perspectives
- All features (clustering, reordering) work in transposed mode

### 9. Export Options

**PNG Export**:
- Set export scale (1-5x) for higher resolution
- Download as `heatmap_YYYYMMDD_HHMMSS.png`

**SVG Export**:
- Vector format for publications
- Infinite scaling without quality loss
- Download as `heatmap_YYYYMMDD_HHMMSS.svg`

### 10. Settings Persistence

Your preferences are automatically saved:
- Scale selection
- Colorscale choice
- Custom colors
- Plot dimensions
- Clustering method
- Row/column order after reordering

**Manual Controls**:
- Click "💾 Save Settings" to save current state
- Click "🔄 Reset Settings" to restore defaults

## Advanced Usage

### Custom Row/Column Ordering

Provide biological ordering at creation time:

```python
# Define custom order for source neurons
custom_row_order = ['KC_alpha', 'KC_beta', 'KC_gamma', 'MBON']

# Define custom order for target neurons  
custom_col_order = ['DAN', 'MBON', 'Output']

plot_stat(
    conn_df=df,
    custom_order={
        'row': custom_row_order,
        'col': custom_col_order
    }
)
```

### Multiple Metrics

If your data includes weight, ratio, and probability:

```python
# Heatmap will include metric toggle buttons
plot_stat(
    conn_df=df,
    metric='weight'  # Initial metric to display
)
```

Switch between metrics using buttons in the interface.

### Negative Values

The heatmap supports negative connection values:
- Diverging colorscales (RdBu, PiYG) work best
- Center point (0) is automatically detected
- Useful for showing inhibitory vs excitatory connections

### Large Matrices

For matrices with >500 neurons:
- Consider using clustering to find patterns
- Use logarithmic scales to handle wide dynamic ranges
- Hide cell values for better performance
- Increase plot size for better readability

## Keyboard Shortcuts

- `H`: Hide/show heatmap
- `L`: Toggle label position
- `Ctrl+S` (in some browsers): Quick save settings

## Tips & Best Practices

### 1. Finding Patterns
- Start with clustering to reveal structure
- Try different clustering methods (Ward usually works best)
- Use log scale if a few connections dominate

### 2. Publication-Ready Figures
- Set appropriate plot dimensions (1200x1000 recommended)
- Use perceptually uniform colorscales (Viridis, Plasma)
- Export as SVG for vector graphics
- Show cell values for small matrices

### 3. Comparing Heatmaps
- Use manual color range with same zmin/zmax
- Keep same colorscale across comparisons
- Export at same dimensions

### 4. Performance
- Large matrices (>1000 neurons): disable cell values
- Use square cells lock for faster rendering
- Close reorder panel when not in use

## Technical Details

### Data Processing
- Linear data: Used directly from input
- Log data: Applied as `log(value + 1)` to handle zeros
- Sqrt data: Applied as `sqrt(abs(value)) * sign(value)` for negative values
- Missing connections shown as gray cells

### Clustering Algorithm
- Method: Hierarchical clustering via scipy
- Distance: Euclidean distance on connection vectors
- Linkage: User-selectable (Ward, Average, Complete, Single)
- Applied independently to rows and columns

### Storage
- Settings stored in browser localStorage
- Persists across sessions
- Separate storage per heatmap file

### Browser Compatibility
- Chrome/Edge: Recommended (best performance)
- Firefox: Fully supported
- Safari: Supported (some export limitations)

## Troubleshooting

**Clustering not available**: Matrix too small (<2 rows or columns) or contains NaN values

**Cell values not readable**: Adjust contrast threshold or use reverse contrast

**Colors look wrong**: Check if using appropriate colorscale for your data type (sequential vs diverging)

**Slow performance**: Disable cell values, reduce matrix size, or use smaller plot dimensions

**Export fails**: Try different browser or check file download permissions

## Related Documentation

- [Custom Colors Guide](./CUSTOM_COLORS_GUIDE.md)
- [Clustering Quick Reference](./HEATMAP_CLUSTERING_QUICKREF.md)
- [Backend Optimization](./HEATMAP_BACKEND_OPTIMIZATION.md)
