# Edge Width Scaling Guide

## Overview

This document describes the edge width scaling methods available in VisualizePath for controlling how edge weights are visualized.

## Edge Width Scaling Methods

### Sankey Diagrams

**Scaling Method:** Automatic proportional scaling by Plotly

- Plotly's Sankey implementation automatically scales link widths proportionally to the `value` parameter
- Links with higher weights appear proportionally wider
- No manual scaling configuration needed - this is handled internally by Plotly
- The width is relative to the total flow through the diagram

**Example:**
```python
vis = VisualizePath('pathways.csv')
vis.create_sankey()  # Link widths auto-scaled by Plotly
```

### Network Graphs (Cytoscape.js)

**Scaling Method:** Configurable via `edge_width_scale` parameter

Network graphs support four edge width scaling methods:

#### 1. **Logarithmic Scaling** (Default)
- **Method:** `edge_width_scale='log'`
- **Formula:** `width ∝ log(weight + 1)`
- **Best for:** Wide range of weight values (e.g., 1 to 10,000)
- **Effect:** Compresses large differences, making small edges visible

```python
vis = VisualizePath(
    'pathways.csv',
    edge_width_scale='log',  # Default
    edge_width_factor=1.0
)
```

#### 2. **Linear Scaling**
- **Method:** `edge_width_scale='linear'`
- **Formula:** `width ∝ weight`
- **Best for:** Narrow range of weight values or when exact proportions matter
- **Effect:** Edges scale directly with weight values

```python
vis = VisualizePath(
    'pathways.csv',
    edge_width_scale='linear',
    edge_width_factor=0.5  # Scale down if edges too thick
)
```

#### 3. **Square Root Scaling**
- **Method:** `edge_width_scale='sqrt'`
- **Formula:** `width ∝ √weight`
- **Best for:** Moderate compression of weight differences
- **Effect:** Less compression than log, more than linear

```python
vis = VisualizePath(
    'pathways.csv',
    edge_width_scale='sqrt',
    edge_width_factor=1.5
)
```

#### 4. **No Scaling** (Constant Width)
- **Method:** `edge_width_scale='none'`
- **Formula:** `width = constant`
- **Best for:** Focusing on topology, not weight magnitude
- **Effect:** All edges same width regardless of weight

```python
vis = VisualizePath(
    'pathways.csv',
    edge_width_scale='none',
    edge_width_factor=3.0  # Sets the constant width
)
```

## Edge Width Factor

The `edge_width_factor` parameter is a multiplier applied **after** scaling:

```python
final_width = scaled_width × edge_width_factor
```

**Use cases:**
- `edge_width_factor < 1.0`: Make edges thinner (e.g., 0.5 = half width)
- `edge_width_factor = 1.0`: Default size
- `edge_width_factor > 1.0`: Make edges thicker (e.g., 2.0 = double width)

**Example:**
```python
# Make edges 3x thicker with linear scaling
vis = VisualizePath(
    'pathways.csv',
    edge_width_scale='linear',
    edge_width_factor=3.0
)
```

## Edge Weight Aggregation Fix

### Problem (Before)

When multiple paths contained the same edge, weights were **summed**:

```python
Path 1: A -> B (weight=100)
Path 2: A -> B (weight=100)
Result: A -> B (weight=200)  # WRONG - summed
```

### Solution (After)

Weights now use **maximum** value across paths:

```python
Path 1: A -> B (weight=100)
Path 2: A -> B (weight=100)
Result: A -> B (weight=100)  # CORRECT - max
```

This is more accurate since the same edge across different paths typically represents the same biological connection with the same synapse count.

**Implementation:**
```python
# In _build_network_from_pathways()
agg_dict = {'weight': 'max'}  # Changed from 'sum'
```

## Visual Comparison

To see the differences between scaling methods, run:

```bash
python test_edge_width_scaling.py
```

This generates four network visualizations (one for each scaling method) in:
```
test_output/edge_width_scaling/
├── linear/network_selected_paths.html
├── log/network_selected_paths.html
├── sqrt/network_selected_paths.html
└── none/network_selected_paths.html
```

## Choosing a Scaling Method

| Data Characteristics | Recommended Method | Reason |
|---------------------|-------------------|--------|
| Weight range: 1-100 | `linear` | Narrow range, proportions visible |
| Weight range: 1-10,000 | `log` (default) | Wide range, prevents huge edges |
| Weight range: 10-1,000 | `sqrt` | Moderate range, balanced |
| Topology focus | `none` | Ignore weights, show structure |
| Very thick edges | Any + `factor=0.5` | Scale down with factor |
| Very thin edges | Any + `factor=2.0` | Scale up with factor |

## API Summary

```python
from vispath import VisualizePath

vis = VisualizePath(
    path_file='pathways.csv',
    
    # Edge width scaling (network graphs only)
    edge_width_scale='log',      # 'linear', 'log', 'sqrt', 'none'
    edge_width_factor=1.0,       # Multiplier (e.g., 2.0 = 2x thicker)
    
    # Other parameters...
    source_color='#1f77b4',
    showfig=True
)

# Create visualizations
vis.create_sankey()   # Plotly auto-scales
vis.create_network()  # Uses edge_width_scale
```

## Implementation Details

### Scaling Function

```python
def _calculate_edge_widths(self, weights):
    """Calculate scaled edge widths from raw weights."""
    weights = np.array(weights, dtype=float)
    weights = np.maximum(weights, 1e-6)  # Avoid zero/negative
    
    if self.edge_width_scale == 'linear':
        scaled = weights
    elif self.edge_width_scale == 'log':
        scaled = np.log(weights + 1)
    elif self.edge_width_scale == 'sqrt':
        scaled = np.sqrt(weights)
    elif self.edge_width_scale == 'none':
        scaled = np.ones_like(weights)
    
    return scaled * self.edge_width_factor
```

### Cytoscape.js Integration

```javascript
// Edge styling in network
{
    selector: 'edge',
    style: {
        'width': 'data(scaled_width)',  // Uses pre-calculated widths
        // ...
    }
}

// Selected edges (1.5x thicker)
{
    selector: 'edge:selected',
    style: {
        'width': 'calc(data(scaled_width) * 1.5)',
        // ...
    }
}

// Highlighted edges (2x thicker)
{
    selector: 'edge.highlighted',
    style: {
        'width': 'calc(data(scaled_width) * 2)',
        // ...
    }
}
```

## Related Files

- **Implementation:** `vispath.py`
  - `_calculate_edge_widths()` - Scaling calculation
  - `_build_network_from_pathways()` - Weight aggregation (max)
  - `_plot_cytoscape_network()` - Network visualization
  
- **Tests:** `test_edge_width_scaling.py`
  - Demonstrates all scaling methods
  - Verifies max aggregation fix

## Backward Compatibility

These changes are **fully backward compatible**:

- Default `edge_width_scale='log'` maintains similar appearance to previous logarithmic scaling
- Default `edge_width_factor=1.0` preserves existing visual scale
- Edge weight aggregation fix improves accuracy without breaking existing code
- Sankey diagrams unaffected (Plotly handles scaling)

Existing code will work without modifications while new parameters enable fine-grained control.
