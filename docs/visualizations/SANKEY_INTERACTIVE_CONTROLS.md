# Sankey Interactive Controls Enhancement

## Summary

Enhanced the Sankey diagram visualization with three key improvements:
1. **Removed info-box** that was covering the left side of the diagram
2. **Added zoom controls** for better diagram inspection
3. **Added label toggle** to show/hide node names

**Date**: October 28, 2025

---

## Changes Made

### 1. Removed Info-Box ✅

**Issue**: The hover instruction label (`#info-box`) was covering the diagram from top to bottom on the left side.

**Solution**: Completely removed the info-box element and its CSS.

**Code Removed**:
```css
/* REMOVED */
#info-box {
    position: absolute;
    top: 10px;
    left: 10px;
    background: rgba(255,255,255,0.95);
    padding: 10px 15px;
    border-radius: 8px;
    /* ... */
}
```

```html
<!-- REMOVED -->
<div id="info-box">
    <strong>Interactive Sankey</strong> • 
    Adjust colors at top • 
    Click "Show/Hide" to toggle node/edge visibility
</div>
```

**Result**: Clean diagram area without any obstructing overlays.

---

### 2. Added Zoom Controls ✅

**Feature**: Zoom in/out functionality with three buttons:
- **🔍 Zoom In**: Increases diagram size by 20%
- **🔍 Zoom Out**: Decreases diagram size by 20%
- **⟲ Reset Zoom**: Returns to 100% size

**Implementation**:

**Buttons Added** (in control panel):
```html
<div class="btn-group">
    <button class="btn-secondary" onclick="zoomIn()">🔍 Zoom In</button>
    <button class="btn-secondary" onclick="zoomOut()">🔍 Zoom Out</button>
    <button class="btn-secondary" onclick="resetZoom()">⟲ Reset Zoom</button>
    <button class="btn-secondary" onclick="toggleLabels()">🏷️ Toggle Labels</button>
</div>
```

**JavaScript Functions**:
```javascript
// Zoom functionality
let zoomLevel = 1.0;
const zoomStep = 0.2;
const minZoom = 0.3;   // 30%
const maxZoom = 3.0;   // 300%

function zoomIn() {
    zoomLevel = Math.min(maxZoom, zoomLevel + zoomStep);
    applyZoom();
}

function zoomOut() {
    zoomLevel = Math.max(minZoom, zoomLevel - zoomStep);
    applyZoom();
}

function resetZoom() {
    zoomLevel = 1.0;
    applyZoom();
}

function applyZoom() {
    const container = document.querySelector('#sankey-container > div');
    if (container) {
        container.style.transform = `scale(${zoomLevel})`;
        container.style.transformOrigin = 'center center';
    }
}
```

**Features**:
- Zoom range: 30% to 300%
- 20% increment per click
- Transform origin centered for balanced zooming
- Smooth CSS transform-based zooming

---

### 3. Added Label Toggle ✅

**Feature**: Toggle button to show/hide node labels in the Sankey diagram.

**Implementation**:

**Button Added**:
```html
<button class="btn-secondary" onclick="toggleLabels()">🏷️ Toggle Labels</button>
```

**JavaScript Functions**:
```javascript
// Label toggle functionality
let labelsVisible = true;

function toggleLabels() {
    labelsVisible = !labelsVisible;
    updateDiagram();
}
```

**Updated `updateDiagram()` function**:
```javascript
const visibleNodeLabels = nodeLabels.map((label, idx) => {
    if (hiddenNodes.has(idx)) return '';
    if (!labelsVisible) return '';  // Hide labels when toggled off
    return label;
});
```

**Features**:
- Single-click toggle
- Works in combination with node visibility controls
- Labels stay hidden for hidden nodes regardless of toggle state
- Instant visual feedback

---

## Control Panel Layout

The enhanced control panel now has 4 sections (left to right):

```
┌─────────────────────────────────────────────────────────────────────┐
│ [Source Color] [Intermediate Color] [Target Color]                 │
│                                                                      │
│ [Edge Color] [Edge Opacity: ──────○───── 30%]                      │
│                                                                      │
│ [Apply] [Reset] [Show/Hide]                                         │
│                                                                      │
│ [🔍 Zoom In] [🔍 Zoom Out] [⟲ Reset Zoom] [🏷️ Toggle Labels]      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Usage Examples

### Basic Usage
```python
from vispath import VisualizePath

vp = VisualizePath('network.csv')
vp.visualize()
# Open HTML → Use zoom and label controls in browser
```

### With Custom Colors
```python
import pandas as pd

network = pd.DataFrame({
    'source': ['A', 'B', 'C'],
    'target': ['B', 'C', 'D'],
    'weight': [10, 15, 8],
    'color': ['#FF6B6B', '#4ECDC4', '#45B7D1']
})

node_colors = pd.DataFrame({
    'node': ['A', 'B', 'C', 'D'],
    'color': ['#E91E63', '#9C27B0', '#673AB7', '#4CAF50']
})

vp = VisualizePath(
    path_file=network,
    node_colors=node_colors
)
vp.visualize()
```

---

## Testing

### Test Script: `test_sankey_features.py`

Creates a comprehensive test with:
- 7 nodes (2 inputs, 3 process, 2 outputs)
- 12 edges with custom colors
- Custom node colors by layer
- All enhanced features enabled

**Run Test**:
```bash
python test_sankey_features.py
open ./test_output/sankey_features/sankey_selected_paths.html
```

**What to Test**:
1. ✅ No info-box visible anywhere on the diagram
2. 🔍 Click "Zoom In" → Diagram enlarges (up to 300%)
3. 🔍 Click "Zoom Out" → Diagram shrinks (down to 30%)
4. ⟲ Click "Reset Zoom" → Returns to 100%
5. 🏷️ Click "Toggle Labels" → Node names disappear
6. 🏷️ Click "Toggle Labels" again → Node names reappear

---

## Technical Details

### Zoom Implementation

**Method**: CSS Transform
- Uses `transform: scale()` for smooth, GPU-accelerated zooming
- `transform-origin: center center` ensures balanced scaling
- No Plotly relayout needed (pure CSS)

**Advantages**:
- Instant visual feedback
- No redrawing overhead
- Works with all Plotly interactions
- Browser-optimized performance

**Limitations**:
- Does not change actual SVG coordinates
- Very high zoom (>200%) may show pixelation
- Scroll position not automatically adjusted

### Label Toggle Implementation

**Method**: Data Update
- Modifies `label` array in Plotly data
- Uses `Plotly.react()` to update the figure
- Respects node visibility state

**Advantages**:
- Clean implementation
- Works with all other controls
- No performance overhead

---

## Files Modified

### `vispath.py`

**Lines Changed**: ~50 lines
- Removed info-box CSS (~20 lines)
- Removed info-box HTML (~5 lines)
- Added zoom controls HTML (~5 lines)
- Added zoom JavaScript (~25 lines)
- Added label toggle JavaScript (~10 lines)
- Updated `updateDiagram()` function (~5 lines)

### Files Created
- `test_sankey_features.py`: Comprehensive test (100 lines)

---

## Backward Compatibility

✅ **Fully backward compatible**
- All existing code works without changes
- No breaking changes to API
- New controls are purely additive enhancements

---

## User Experience Improvements

### Before
- ❌ Info-box covering left side of diagram
- ❌ No way to zoom for detailed inspection
- ❌ No way to hide labels for cleaner view

### After
- ✅ Clean, unobstructed diagram area
- ✅ Zoom controls for detailed inspection (30%-300%)
- ✅ Label toggle for decluttered visualization
- ✅ All controls easily accessible in top panel

---

## Common Use Cases

### 1. Large Diagrams - Use Zoom
```
Problem: Too many nodes, hard to see details
Solution: Click "Zoom In" 2-3 times to focus on specific areas
```

### 2. Presentations - Hide Labels
```
Problem: Node names clutter the visual flow
Solution: Click "Toggle Labels" to show clean pathways
         Click again when you need to reference specific nodes
```

### 3. Complex Networks - Combine Features
```
1. Hide unnecessary edges with "Show/Hide" panel
2. Toggle labels off for clean view
3. Zoom in to inspect specific connections
4. Adjust colors for better contrast
5. Reset zoom when done
```

---

## Future Enhancements (Potential)

1. **Pan functionality** while zoomed
2. **Zoom to specific node** on click
3. **Keyboard shortcuts** (e.g., +/- for zoom, L for labels)
4. **Persistent zoom level** across interactions
5. **Mouse wheel zoom** support
6. **Minimap** for navigation when zoomed

---

## Related Documentation

- `CUSTOM_COLORS_GUIDE.md` - Custom node/edge colors
- `SANKEY_LAYOUT_COLORS_SUMMARY.md` - Layout and color features
- `SANKEY_ENHANCEMENT_SUMMARY.md` - Overall Sankey features
- `test_sankey_features.py` - Test script

---

## Summary of All Sankey Features

The Sankey diagram now includes:

### Visual Customization
- ✅ Custom node colors (DataFrame, CSV, Excel)
- ✅ Custom edge colors (color column in edge-list)
- ✅ Adjustable node type colors (source/intermediate/target)
- ✅ Adjustable edge color and opacity

### Interactive Controls
- ✅ Show/Hide individual nodes and edges
- ✅ Zoom In/Out/Reset (30%-300%)
- ✅ Toggle node labels on/off
- ✅ Apply/Reset color changes

### Layout
- ✅ Full-page responsive layout
- ✅ Proper margins (40-80px)
- ✅ Layered topology-based arrangement
- ✅ No obstructing overlays

### Data Format Support
- ✅ Path-based format
- ✅ Edge-list format with flexible column naming
- ✅ Support for `*_pre` and `*_post` patterns
- ✅ Optional color columns

All features work seamlessly together for a powerful, customizable visualization tool! 🎉
