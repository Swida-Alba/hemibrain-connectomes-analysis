# Interactive Edge Width Controls for Network Visualization

## Overview

The network visualization now includes **interactive controls** for adjusting edge widths in real-time, allowing you to explore different scaling methods and adjust the overall thickness without regenerating the visualization.

## Problem Fixed

**Original Issue:** When using logarithmic scaling (`edge_width_scale='log'`), all edges appeared to have the same width due to:
1. The scaled values having a small range (e.g., 3.4 to 6.2)
2. Cytoscape.js not properly mapping these values to visible widths

**Solution:** 
- Use Cytoscape's `mapData()` function to map scaled values to a visible width range (1-10px for normal edges)
- Add interactive controls to adjust scaling method and width factor on-the-fly

## Interactive Controls

The network visualization now includes two new controls in the color palette panel:

### 1. Edge Width Scale Dropdown

**Location:** Color Palette → Edge Width Scale

**Options:**
- **Linear:** Direct proportional scaling (width ∝ weight)
- **Logarithmic:** Log scaling (width ∝ log(weight)) - Default
- **Square Root:** Square root scaling (width ∝ √weight)
- **None (Constant):** All edges same width

**How it works:** When you select a different scaling method, the JavaScript function `updateEdgeWidths()` recalculates all edge widths using the new formula and updates the visualization instantly.

### 2. Edge Width Factor Slider

**Location:** Color Palette → Edge Width Factor

**Range:** 0.1× to 5.0× (adjustable in 0.1 increments)

**Purpose:** Multiplies the scaled widths by the factor, making all edges proportionally thicker or thinner.

**Examples:**
- `0.5×`: Half thickness (good for dense networks)
- `1.0×`: Default thickness
- `2.0×`: Double thickness (good for sparse networks)
- `5.0×`: Very thick edges (emphasize connections)

## How to Use

### Initial Setup (Python)

```python
from vispath import VisualizePath

vis = VisualizePath(
    'pathways.csv',
    edge_width_scale='log',      # Initial scaling method
    edge_width_factor=1.0,       # Initial multiplier
    showfig=True
)

vis.create_network()
```

### Interactive Adjustment (Browser)

1. **Open the network visualization** (`network_selected_paths.html`)
2. **Click the color palette icon** (🎨) in the top-right corner
3. **Scroll to the edge width controls**:
   - **Edge Width Scale:** Select scaling method from dropdown
   - **Edge Width Factor:** Drag slider to adjust thickness
4. **Changes apply instantly** - no need to click "Apply"

### Best Practices

| Your Data | Recommended Initial Settings | Interactive Adjustment |
|-----------|----------------------------|------------------------|
| Weights 1-100 | `linear`, `factor=1.0` | Try `factor=2.0` if edges too thin |
| Weights 1-10,000 | `log`, `factor=1.0` | Try `sqrt` if differences too subtle |
| Weights 10-1,000 | `sqrt`, `factor=1.5` | Adjust factor for visual preference |
| Dense network | Any method, `factor=0.5` | Reduce factor to avoid overlap |
| Sparse network | Any method, `factor=2.0` | Increase factor for visibility |

## Technical Implementation

### Cytoscape mapData() Function

The edge width is now calculated using Cytoscape's built-in `mapData()` function:

```javascript
'width': 'mapData(scaled_width, minValue, maxValue, minWidth, maxWidth)'
```

**Parameters:**
- `scaled_width`: Data attribute containing the scaled weight value
- `minValue, maxValue`: Range of scaled_width values
- `minWidth, maxWidth`: Corresponding visual width range in pixels

**Example:**
```javascript
// Normal edges: map to 1-10px range
'width': 'mapData(scaled_width, 3.4, 6.2, 1, 10)'

// Selected edges: map to 2-15px range (thicker)
'width': 'mapData(scaled_width, 3.4, 6.2, 2, 15)'

// Highlighted edges: map to 3-20px range (thickest)
'width': 'mapData(scaled_width, 3.4, 6.2, 3, 20)'
```

### JavaScript Update Function

```javascript
function updateEdgeWidths() {
    const scalingMethod = document.getElementById('edgeWidthScale').value;
    const widthFactor = parseFloat(document.getElementById('edgeWidthFactor').value);
    
    // Get all edge weights
    const edges = cy.edges();
    const weights = edges.map(e => e.data('weight'));
    
    // Calculate scaled widths based on method
    let scaledWidths = weights.map(w => {
        let scaled;
        if (scalingMethod === 'linear') {
            scaled = w;
        } else if (scalingMethod === 'log') {
            scaled = Math.log(w + 1);
        } else if (scalingMethod === 'sqrt') {
            scaled = Math.sqrt(w);
        } else { // 'none'
            scaled = 1.0;
        }
        return scaled * widthFactor;
    });
    
    // Find min/max for mapping
    const minScaled = Math.min(...scaledWidths);
    const maxScaled = Math.max(...scaledWidths);
    
    // Update each edge's scaled_width data
    edges.forEach((edge, i) => {
        edge.data('scaled_width', scaledWidths[i]);
    });
    
    // Update the stylesheet with new mapping range
    cy.style()
        .selector('edge')
        .style({
            'width': `mapData(scaled_width, ${minScaled}, ${maxScaled}, 1, 10)`
        })
        .update();
}
```

### Control HTML

```html
<div class="color-group">
    <label>Edge Width Scale:</label>
    <div class="color-input-group">
        <select id="edgeWidthScale" onchange="updateEdgeWidths()">
            <option value="linear">Linear</option>
            <option value="log" selected>Logarithmic</option>
            <option value="sqrt">Square Root</option>
            <option value="none">None (Constant)</option>
        </select>
    </div>
</div>

<div class="color-group">
    <label>Edge Width Factor:</label>
    <div class="color-input-group">
        <input type="range" id="edgeWidthFactor" 
               min="0.1" max="5" step="0.1" value="1.0" 
               oninput="updateEdgeWidthDisplay(this.value)">
        <span class="alpha-value" id="edgeWidthFactorValue">1.0×</span>
    </div>
</div>
```

## Comparison: Before vs After

### Before (Fixed Settings)

```python
# Settings defined at creation time
vis = VisualizePath('data.csv', edge_width_scale='log', edge_width_factor=1.0)
vis.create_network()

# To change settings, must recreate the entire visualization
vis2 = VisualizePath('data.csv', edge_width_scale='linear', edge_width_factor=2.0)
vis2.create_network()
```

### After (Interactive Controls)

```python
# Settings defined at creation time (optional)
vis = VisualizePath('data.csv', edge_width_scale='log', edge_width_factor=1.0)
vis.create_network()

# In browser: adjust scaling method and factor interactively
# Changes apply instantly without recreating the visualization
```

## Edge Width Ranges

Different edge types use different width ranges for visual hierarchy:

| Edge Type | Width Range | Use Case |
|-----------|-------------|----------|
| Normal edges | 1-10px | Default view |
| Selected edges | 2-15px | User clicked on edge |
| Highlighted edges | 3-20px | Double-clicked or highlighted via neighbor selection |

These ranges are dynamically updated based on the min/max of the scaled values.

## Troubleshooting

### All edges still look the same width

**Possible causes:**
1. Weight values are too similar (e.g., all between 95-105)
2. Using 'none' scaling method
3. Width factor set too low (< 0.3)

**Solutions:**
- Try 'linear' scaling to see raw proportions
- Increase width factor to 2.0 or higher
- Check actual weight values in the data

### Edges are too thick/thin

**Solutions:**
- Adjust the **Edge Width Factor** slider
- For thick edges: reduce to 0.5× or 0.3×
- For thin edges: increase to 2.0× or 3.0×

### Changes don't apply

**Solution:**
- Make sure you're using the latest version with interactive controls
- Check browser console for JavaScript errors (F12 → Console)
- The dropdown applies changes immediately, no "Apply" button needed

## Example Use Case

**Scenario:** You have edge weights ranging from 10 to 5,000 synapses.

**Step 1:** Create visualization
```python
vis = VisualizePath(
    'pathways.csv',
    edge_width_scale='log',  # Start with log for wide range
    edge_width_factor=1.0
)
vis.create_network()
```

**Step 2:** Open in browser and experiment
1. Open `network_selected_paths.html`
2. Click color palette icon (🎨)
3. Try different scaling methods:
   - **Log:** Edges visible but differences subtle
   - **Linear:** Strongest edges dominate, weak edges barely visible
   - **Square root:** Good balance between the two
4. Adjust factor slider:
   - `1.0×`: Default
   - `2.0×`: Edges more prominent
   - `0.5×`: Edges less cluttered

**Step 3:** Find optimal settings for your analysis needs

## Related Files

- **Implementation:** `vispath.py`
  - Line ~2430: `_calculate_edge_widths()` - Backend calculation
  - Line ~2975: Cytoscape edge style with `mapData()`
  - Line ~2900: HTML controls for edge width
  - Line ~3260: JavaScript `updateEdgeWidths()` function

- **Tests:** `test_edge_width_scaling.py`
  - Demonstrates different scaling methods
  - Shows weight aggregation fix (max vs sum)

- **Documentation:**
  - `EDGE_WIDTH_SCALING.md` - Original edge width scaling guide
  - `EDGE_WIDTH_INTERACTIVE_CONTROLS.md` - This document

## Summary

Interactive edge width controls provide:
- ✅ **Real-time adjustment** of edge widths without regenerating visualizations
- ✅ **Four scaling methods** to handle different weight distributions
- ✅ **Width factor slider** for fine-tuning overall thickness
- ✅ **Proper mapData()** implementation for visible width differences
- ✅ **Seamless integration** with existing color palette controls

This feature makes it easy to explore your network data visually and find the optimal edge width settings for your specific analysis needs!
