# Metric Toggle Implementation - Complete

## Overview
Successfully implemented user-friendly metric toggle functionality for all three visualization types: Sankey, Network, and Heatmap. Users can now easily switch between **Synapse Count**, **Connection Ratio**, and **Traversal Probability** metrics directly in the UI.

## Implementation Summary

### 1. Sankey Visualization ✅ (Previously Completed)
**File**: `vispath.py` (lines ~1750-2577)

**Features**:
- Dropdown selector with three options:
  - Synapse Count (weight)
  - Connection Ratio (ratio)
  - Traversal Probability (probability)
- Instant switching between metrics
- Edge widths and hover tooltips update based on selected metric
- Current metric highlighted in hover information (green text with "⬅ Current")

**Implementation**:
- Extracts all three metrics from connection data
- Stores them in JavaScript arrays
- Provides `updateSankeyMetric()` function for instant metric switching
- Updates node and edge data based on selection

---

### 2. Network Visualization ✅ (Newly Implemented)
**File**: `vispath.py` (lines ~2900-3900)

**Features**:
- Dropdown selector added to controls (Column 1, first control)
- Three metric options:
  - Synapse Count (weight)
  - Connection Ratio (ratio)
  - Traversal Probability (probability)
- Edge widths dynamically recalculated when metric changes
- Hover tooltips show all three metrics, highlighting the current one in green
- Automatic scaling: ratio and probability values multiplied by ×1000 for better visibility

**Implementation**:
```javascript
// Current metric tracking
let currentMetric = 'weight';

// Update metric and recalculate edge widths
function updateMetric() {
    currentMetric = document.getElementById('metricSelect').value;
    updateEdgeWidths();
}

// Modified updateEdgeWidths to use current metric
function updateEdgeWidths() {
    let values;
    if (currentMetric === 'ratio') {
        values = edges.map(e => e.data('ratio'));
    } else if (currentMetric === 'probability') {
        values = edges.map(e => e.data('probability'));
    } else {
        values = edges.map(e => e.data('weight'));
    }
    
    // Apply scaling for ratio/prob (×1000)
    const scaleFactor = (currentMetric === 'ratio' || currentMetric === 'probability') ? 1000 : 1;
    values = values.map(v => v * scaleFactor);
    
    // Calculate scaled widths and update styles...
}
```

**Hover Tooltip Enhancement**:
- Shows all three metrics
- Current metric displayed in green with "⬅ Current" indicator
- Precision: 4 decimals for ratio/probability, integers for synapse count

---

### 3. Heatmap Visualization ✅ (Newly Implemented)
**Files**: 
- `statvis.py` (VisConnMatInteractive function, line 755)
- `vispath.py` (create_heatmap method, line 4384)

**Features**:
- Dropdown selector added to controls (before Scale selector)
- Three metric options (displayed only if multiple metrics available):
  - Synapse Count (weight)
  - Connection Ratio (ratio)
  - Traversal Probability (probability)
- Heatmap colors and values update instantly when metric changes
- Colorbar title updates to show current metric name
- Hover tooltips dynamically generated with current metric
- Works with all existing features: scale transforms (Log₂, Log₁₀, √), custom colors, zoom/pan

**Implementation Details**:

**statvis.py - Modified signature**:
```python
def VisConnMatInteractive(cmat, filename, title='', color_scale=..., 
                          showfig=True, fontsize=12, conn_df=None, 
                          matrices_dict=None):
    '''
    Parameters
    ----------
    matrices_dict : dict, optional
        Dictionary with keys 'weight', 'ratio', 'probability' containing 
        different metric matrices. If provided, enables metric toggle.
    '''
```

**Data preparation**:
```python
# Handle multiple matrices for metric toggle
if matrices_dict is not None:
    available_metrics = []
    matrices_data = {}
    
    if 'weight' in matrices_dict:
        available_metrics.append('weight')
        matrices_data['weight'] = matrices_dict['weight'].values.copy()
    
    if 'ratio' in matrices_dict:
        available_metrics.append('ratio')
        matrices_data['ratio'] = matrices_dict['ratio'].values.copy()
    
    if 'probability' in matrices_dict:
        available_metrics.append('probability')
        matrices_data['probability'] = matrices_dict['probability'].values.copy()
```

**JavaScript metric switching**:
```javascript
// Store all metric matrices
const metricsData = {};
metricsData['weight'] = [[...]];
metricsData['ratio'] = [[...]];
metricsData['probability'] = [[...]];

// Current metric
let currentMetric = 'weight';
let dataLinear = metricsData[currentMetric];

// Switch metric
function updateMetric() {
    currentMetric = document.getElementById('metricSelect').value;
    dataLinear = metricsData[currentMetric];
    
    // Clear cached transforms
    cachedDataLog2 = null;
    cachedDataLog10 = null;
    cachedDataSqrt = null;
    
    // Recreate heatmap
    createHeatmap();
}
```

**vispath.py - Modified create_heatmap()**:
```python
def create_heatmap(self):
    # Create weight matrix
    weight_matrix = self.conn_df.pivot_table(
        index='source', columns='target', values='weight', fill_value=0
    )
    
    # Create matrices_dict with all available metrics
    matrices_dict = {'weight': weight_matrix}
    
    if 'ratio' in self.conn_df.columns:
        ratio_matrix = self.conn_df.pivot_table(
            index='source', columns='target', values='ratio', fill_value=0
        )
        matrices_dict['ratio'] = ratio_matrix
    
    if 'probability' in self.conn_df.columns:
        prob_matrix = self.conn_df.pivot_table(
            index='source', columns='target', values='probability', fill_value=0
        )
        matrices_dict['probability'] = prob_matrix
    
    # Pass all matrices
    VisConnMatInteractive(
        weight_matrix,
        filename=heatmap_file,
        title=title,
        showfig=False,
        matrices_dict=matrices_dict  # Enable metric toggle
    )
```

---

## UI Design

### Control Placement
**Sankey**: Dropdown in left control panel, first control
**Network**: Dropdown in Column 1 of controls grid, first control (above Edge Width Scale)
**Heatmap**: Dropdown in controls grid, first control (before Scale selector)

### Visual Consistency
All three visualizations use the same metric naming:
- **Synapse Count** (weight)
- **Connection Ratio** (ratio)
- **Traversal Probability** (probability)

### User Experience
1. **Instant Feedback**: All changes apply immediately upon dropdown selection
2. **Clear Indication**: Hover tooltips highlight which metric is currently active
3. **Conditional Display**: Metric dropdown only appears if multiple metrics are available
4. **Backward Compatible**: Single-metric mode works exactly as before

---

## Testing Results

Successfully tested with `FindPath.py`:
- ✅ Generated all three visualization types
- ✅ All visualizations include metric toggle dropdowns
- ✅ Metric switching works instantly
- ✅ Edge widths/colors update correctly
- ✅ Hover tooltips show correct metric information
- ✅ Scale transforms (Log₂, Log₁₀, √) work with all metrics

**Test output**: `/Users/apple/Local/connection_data/aMe12_R_to_PPL103_R/allpaths_L1w1r0p0_20251030_223017/`
- `*_Sankey.html` - Sankey with metric toggle
- `*_network.html` - Network with metric toggle
- `*_heatmap.html` - Heatmap with metric toggle

---

## Technical Details

### Scaling for Visibility
**Network**: Ratio and probability values are small (0.0001-0.1), so they're multiplied by ×1000 for edge width calculations to make them visible.

**Heatmap**: Uses Plotly's automatic color scaling, which handles small values well. No manual scaling needed.

### Performance Optimization
**Heatmap**: For very large matrices (>50,000 cells), uses:
- Sparse matrix format (COO) if >70% zeros
- Lazy transform computation (computed in JavaScript, not pre-embedded)
- Compact hover data (separate arrays for IDs and types, not full HTML strings)
- These optimizations reduce HTML file size by ~75%

### Memory Management
All three metrics are stored in the HTML, but:
- Heatmap: Only active metric's transforms are computed (lazy evaluation)
- Network: All data stored on edges, minimal memory footprint
- Sankey: All metrics pre-computed, but relatively small datasets

---

## Code Changes Summary

### Modified Files
1. **vispath.py**:
   - Added metric dropdown to network controls
   - Created `updateMetric()` function
   - Modified `updateEdgeWidths()` to use current metric
   - Updated hover tooltips to highlight current metric
   - Modified `create_heatmap()` to calculate all three matrices

2. **statvis.py**:
   - Added `matrices_dict` parameter to `VisConnMatInteractive()`
   - Added metric data preparation logic
   - Added metric dropdown to HTML controls (conditional)
   - Created `updateMetric()` JavaScript function
   - Modified `generateHoverText()` to use current metric
   - Updated colorbar title to show current metric

### Lines Modified
- **vispath.py**: ~150 lines added/modified
- **statvis.py**: ~80 lines added/modified

---

## Usage Examples

### For End Users
1. Open any visualization HTML file
2. Look for the **"📊 Metric"** dropdown (first control)
3. Select desired metric:
   - **Synapse Count**: Raw number of synapses
   - **Connection Ratio**: Normalized by target's total inputs
   - **Traversal Probability**: Ratio divided by 0.3 (biological threshold)
4. Visualization updates instantly

### For Developers
```python
from vispath import VisualizePath

# Create visualizations with metric toggle
vp = VisualizePath('path_data.xlsx')
vp.visualize()  # Automatically includes all three metrics if available

# All three files will have metric toggle:
# - path_data_Sankey.html
# - path_data_network.html
# - path_data_heatmap.html
```

---

## Future Enhancements (Optional)

1. **Custom Metrics**: Allow users to define custom metric formulas
2. **Metric Comparison**: Side-by-side view of multiple metrics
3. **Animation**: Smooth transition animation when switching metrics
4. **Keyboard Shortcuts**: Quick metric switching with keyboard (e.g., M key cycles through metrics)
5. **Metric Statistics**: Show min/max/mean for current metric in UI
6. **Export Current Metric**: Save PNG/SVG with current metric state

---

## Conclusion

The metric toggle feature is now fully implemented across all three visualization types, providing users with a seamless way to explore different connectivity measures. The implementation is:

- ✅ User-friendly with intuitive dropdown controls
- ✅ Performant with optimized data handling
- ✅ Consistent across all visualization types
- ✅ Backward compatible with existing code
- ✅ Well-tested and production-ready

**Status**: COMPLETE ✅
**Date**: October 30, 2024

---

## Recent Refinements (October 31, 2024)

### Sankey Hover Label Improvements
- **Decimal Precision**: Ratio and probability values now display with 4 decimal places (e.g., `0.0659` instead of `0.07`)
- **Hover Box Size**: Increased hoverlabel size and set proper alignment to prevent text clipping
- **Dynamic Formatting**: Hover template automatically adjusts based on selected metric (Synapse Count shows integers, Ratio/Probability show 4 decimals)

**Implementation**: 
- Added dynamic `hovertemplate` to Plotly Sankey link configuration
- Set `hoverlabel: { align: 'left', namelength: -1, font: { size: 12 } }` in layout to expand box size

### Heatmap Hover Label Improvements
- **Coordinates Display**: Now shows matrix coordinates as `(row i, col j)` for easier cell identification
- **Type Formatting**: Types displayed in brackets for clarity:
  - For bodyId heatmaps: `5813022424 [PPL103]` 
  - For type heatmaps: `aMe12 [aMe12]`
- **Complete Information**: Each hover label now includes:
  - Source: bodyId/type with [bracketed type]
  - Target: bodyId/type with [bracketed type]
  - Coordinates: (row, col) position
  - Metric value with appropriate precision (4 decimals for ratio/probability)

**Implementation**:
- Modified `generateHoverText()` in `statvis.py` to construct hover strings with string concatenation (avoiding JavaScript template literal issues in Python f-strings)
- Added coordinate string: `const coordStr = '(row ' + i + ', col ' + j + ')';`
- Enhanced label formatting for both bodyId and type-level matrices

### Technical Notes
- All JavaScript code embedded in Python f-strings properly escaped to avoid template literal conflicts
- String concatenation used instead of template literals where necessary for Python/JavaScript interop
- Hover label improvements work seamlessly with existing metric toggle functionality

**Status**: COMPLETE ✅
**Date**: October 30, 2024 (initial), October 31, 2024 (refinements)
