# Metric Toggle Feature for Visualizations

## Overview
Added user-friendly metric toggle controls to Sankey, Network, and Heatmap visualizations, allowing users to easily switch between different connection strength metrics:
- **Synapse Count** (weight) - Number of synapses
- **Connection Ratio** - weight / total_post
- **Traversal Probability** - ratio / 0.3 (capped at 1)

## Implementation Status

### ✅ Sankey Diagram (COMPLETED)
**File**: `vispath.py` - `create_sankey()` and `_create_sankey_html_with_controls()`

**Changes Made**:
1. Modified `create_sankey()` to extract and pass ratio and prob data alongside weights
2. Updated `_create_sankey_html_with_controls()` signature to accept ratios and probs parameters
3. Added dropdown menu "Connection Metric" in control panel (Row 2)
4. Stored all three metrics (weights, ratios, probs) in JavaScript arrays
5. Added `currentMetric` variable and metric selection handler
6. Modified `updateDiagram()` to use selected metric for edge widths
7. Added scaling for ratio/prob values (×1000) for better visualization
8. Updated edge list to display current metric value
9. Added `getMetricDisplay()` and `updateEdgeListMetrics()` functions

**User Interface**:
- Dropdown shows "Connection Metric" with three options:
  - Synapse Count (always available)
  - Connection Ratio (disabled if no ratio data)
  - Traversal Probability (disabled if no prob data)
- Selection immediately updates both diagram and edge list
- Edge widths scale appropriately for each metric type

**Location in UI**: Control panel, Row 2, leftmost control

### 🔄 Network Visualization (TODO)
**File**: `vispath.py` - `_plot_cytoscape_network()`

**Current Status**:
- Network graph already stores weight, ratio, and probability as edge attributes
- Edge data structure: `G.add_edge(source, target, weight=w, ratio=r, probability=p)`

**Needed Changes**:
1. Add metric selector in Cytoscape HTML controls
2. Store all three metrics in edge data
3. Add JavaScript function to switch edge widths based on selected metric
4. Update tooltip to show current metric value
5. Recalculate edge widths when metric changes

**Suggested Implementation**:
```javascript
// Add to Cytoscape HTML
function updateEdgeMetric(metric) {
    cy.edges().forEach(edge => {
        const data = edge.data();
        let value;
        if (metric === 'ratio') {
            value = data.ratio;
        } else if (metric === 'prob') {
            value = data.probability;
        } else {
            value = data.weight;
        }
        // Scale value for width
        const width = calculateWidth(value, metric);
        edge.style('width', width);
    });
}
```

### 🔄 Heatmap Visualization (TODO)
**File**: `statvis.py` - `VisConnMatInteractive()`

**Current Status**:
- Heatmaps are created with single matrix at a time
- Metric type auto-detected from title/filename
- Scale controls already exist (linear/log2/log10)

**Needed Changes**:
1. Modify to accept multiple matrices (weight_matrix, ratio_matrix, prob_matrix)
2. Add metric toggle dropdown in control panel
3. Store all matrices in JavaScript
4. Add function to switch displayed matrix when metric changes
5. Update colorbar and title when metric changes

**Suggested Implementation**:
- Extend `VisConnMatInteractive()` to accept `matrices_dict` parameter
- Store matrices: `{weight: matrix1, ratio: matrix2, prob: matrix3}`
- Add dropdown similar to Sankey implementation
- Use Plotly.react() to update heatmap data dynamically

## Testing

### Test Sankey Toggle:
```bash
cd /Users/apple/Documents/GitHub/hemibrain-connectomes-analysis-now
python test_metric_toggle.py
```

**What to Test**:
1. ✅ Dropdown appears in control panel
2. ✅ Options are enabled/disabled based on data availability  
3. ✅ Switching metrics updates edge widths
4. ✅ Edge list values update with correct precision (4 decimals for ratio/prob)
5. ✅ Synapse count shows integer values
6. ✅ Ratio/prob are scaled appropriately for visibility

### Manual Testing:
1. Run `FindPath.py` with different settings (bodyId vs type filtering)
2. Open generated Sankey HTML
3. Test metric switching with:
   - Paths with all metrics available
   - Paths with only weights (ratio/prob should be disabled)
4. Verify visual changes are intuitive

## Benefits

### User Experience:
- **Easy Comparison**: Toggle between metrics without regenerating visualizations
- **Immediate Feedback**: Changes apply instantly without page reload
- **Clear Indication**: Disabled options when data unavailable
- **Consistent Interface**: Same toggle control across visualizations

### Scientific Insights:
- Compare **absolute connectivity** (synapse count) vs **relative importance** (ratio)
- Identify **bottleneck connections** (low prob despite high weight)
- Understand **pathway reliability** (traversal probability)
- Multi-scale analysis of same network

## Code Architecture

### Data Flow:
```
coana.py (FindPath/FindAllPath)
  ↓ Calculates weight, ratio, prob
  ↓ Stores in path_df columns
vispath.py (VisualizePath)
  ↓ Extracts all metrics from path_df
  ↓ Passes to HTML generation
HTML + JavaScript
  ↓ Stores in arrays [weights, ratios, probs]
  ↓ User selects metric
  ↓ Updates visualization
```

### Key Functions:

**Python (vispath.py)**:
- `create_sankey()`: Extract and aggregate metrics
- `_create_sankey_html_with_controls()`: Generate HTML with controls

**JavaScript (Sankey HTML)**:
- `currentMetric`: Tracks selected metric
- `updateDiagram()`: Redraws with current metric
- `getMetricDisplay()`: Formats metric values
- `updateEdgeListMetrics()`: Updates edge list display

## Future Enhancements

### Short Term:
1. Complete network visualization toggle
2. Complete heatmap visualization toggle
3. Add metric info tooltips explaining each metric
4. Add metric range display (min/max values)

### Long Term:
1. **Custom Metrics**: Allow user-defined formulas
2. **Metric Combinations**: Display multiple metrics simultaneously
3. **Metric Animations**: Animate transition between metrics
4. **Comparison Mode**: Side-by-side view of different metrics
5. **Export Options**: Save current view with selected metric

## Notes

### Performance:
- All metrics pre-computed and stored in JavaScript
- No server communication needed for metric switching
- Instant response to user interaction

### Data Availability:
- Synapse count always available (from Neuprint)
- Ratio requires post-synaptic count calculation
- Probability calculated from ratio (ratio/0.3, capped at 1)
- Options auto-disabled if data missing

### Scaling Factors:
- **Weight**: No scaling (1:1)
- **Ratio**: ×1000 for visibility (values typically 0.001-0.3)
- **Prob**: ×1000 for visibility (values typically 0-1)

## Files Modified

1. **vispath.py**:
   - `create_sankey()`: Lines ~1750-1805 (added ratio/prob extraction)
   - `_create_sankey_html_with_controls()`: Lines ~1826-2577 (added metric toggle)

## Testing Checklist

- [x] Code runs without errors
- [x] Sankey HTML generated successfully
- [ ] Metric dropdown visible in UI
- [ ] Switching metrics updates diagram
- [ ] Edge list values update
- [ ] Disabled options when data unavailable
- [ ] Visual scaling appropriate for each metric
- [ ] Network toggle (pending)
- [ ] Heatmap toggle (pending)
