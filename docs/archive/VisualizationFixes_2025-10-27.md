# Visualization Fixes - October 27, 2025

## Summary
Fixed three critical issues in the visualization system based on user feedback:
1. Incorrect Sankey diagram structure
2. Alpha channel only affecting legend, not nodes
3. Alt/Option key unreliable for label dragging

---

## 1. Sankey Diagram Fix

### Problem
The `create_sankey()` method was passing the connection DataFrame directly to `SankeyDirect()`, but this function expects a **connection matrix** (pivot table) with:
- **Rows**: Source neurons
- **Columns**: Target neurons  
- **Values**: Connection weights

This is how `coana.py` correctly implements Sankey diagrams.

### Solution
Transform the connection DataFrame into a proper pivot table before passing to `SankeyDirect()`:

```python
# Group by source-target pairs and sum weights
conn_grouped = self.conn_df.groupby(['neuron_pre', 'neuron_post'])['weight'].sum().reset_index()

# Create pivot table: sources as rows, targets as columns
conn_matrix = conn_grouped.pivot_table(
    index='neuron_pre',
    columns='neuron_post', 
    values='weight',
    fill_value=0
)

# Pass connection matrix to SankeyDirect (like coana does)
SankeyDirect(
    conn_matrix,
    file_path=output_path,
    node_color=self.node_color,
    link_color=self.link_color,
    showfig=self.showfig,
    title='Sankey diagram of selected pathways'
)
```

### Changes Made
- **File**: `vispath.py`
- **Method**: `create_sankey()` (lines ~369-420)
- **Key Changes**:
  - Added DataFrame grouping and pivot table transformation
  - Added debug output showing matrix dimensions
  - Fixed parameter passing to match `statvis.SankeyDirect()` signature
  - Removed unused `target_color` parameter (not supported by SankeyDirect)

---

## 2. Alpha Channel Fix (Node Colors)

### Problem
The `applyColors()` function was using `.style('background-color', rgba)` to update node colors with transparency, but this didn't work because:
- Cytoscape node styles use `'background-color': 'data(color)'`
- This means colors come from the node's **data attribute**, not direct style
- The `.style()` method was being overridden by the data attribute

Additionally, the alpha channel only affected the legend, not the actual nodes.

### Solution
Use `.data('color', rgba)` instead of `.style('background-color', rgba)` to update the data attribute that controls node colors:

```javascript
// Update node colors using .data() method (since style uses 'data(color)')
cy.nodes('[node_type = "source"]').data('color', sourceRgba);
cy.nodes('[node_type = "intermediate"]').data('color', intermediateRgba);
cy.nodes('[node_type = "target"]').data('color', targetRgba);
```

### Changes Made
- **File**: `vispath.py`
- **Function**: `applyColors()` (lines ~1160-1200)
- **Key Changes**:
  - Changed from `.style('background-color', rgba)` to `.data('color', rgba)`
  - Now properly updates node colors with transparency
  - Alpha channel (0-100%) correctly controls node opacity

---

## 3. Edge Color Control Added

### Problem
Users could customize node colors and opacity, but edges (connections) had fixed colors with no control.

### Solution
Added a complete color picker + alpha slider for edges in the Style Panel.

### HTML Added
```html
<div class="color-group">
    <label>Edge Color</label>
    <div class="color-input-group">
        <input type="color" id="edgeColor" value="#888888">
        <input type="text" id="edgeColorText" value="#888888" readonly>
    </div>
    <div class="color-input-group" style="margin-top: 5px;">
        <label style="font-size: 11px; margin: 0;">Opacity:</label>
        <input type="range" id="edgeAlpha" min="0" max="100" value="100" step="1" 
               oninput="updateAlphaDisplay('edge', this.value)">
        <span class="alpha-value" id="edgeAlphaValue">100%</span>
    </div>
</div>
```

### JavaScript Updated
```javascript
function applyColors() {
    // ... existing node color code ...
    
    const edgeColor = document.getElementById('edgeColor').value;
    const edgeAlpha = document.getElementById('edgeAlpha').value;
    const edgeRgba = hexToRgba(edgeColor, edgeAlpha);
    
    // Update edge colors
    cy.edges().style('line-color', edgeRgba);
    cy.edges().style('target-arrow-color', edgeRgba);
}
```

### Changes Made
- **File**: `vispath.py`
- **Sections**: HTML (lines ~813-826), JavaScript (lines ~1160-1200)
- **Features**:
  - Edge color picker (default: #888888 gray)
  - Edge opacity slider (0-100%)
  - Color updates both edge lines and arrows
  - Syncs with hex text display

---

## 4. Label Dragging: Alt → R Key

### Problem
The Alt/Option key modifier for dragging labels independently was unreliable:
- macOS treats Option as a special modifier for international characters
- Key detection via `evt.originalEvent.altKey` was inconsistent
- Users reported it "doesn't work well"

### Solution
Changed from **Alt+Drag** to **R+Drag** (press and hold R while dragging):

```javascript
// Label dragging functionality
let isDraggingLabel = false;
let draggedNode = null;
let rKeyPressed = false;

// Track R key state
document.addEventListener('keydown', function(e) {
    if (e.key === 'r' || e.key === 'R') {
        rKeyPressed = true;
    }
});

document.addEventListener('keyup', function(e) {
    if (e.key === 'r' || e.key === 'R') {
        rKeyPressed = false;
        isDraggingLabel = false;
        draggedNode = null;
    }
});

cy.on('grab', 'node', function(evt) {
    // Check if R key is pressed
    if (rKeyPressed) {
        isDraggingLabel = true;
        draggedNode = evt.target;
        evt.preventDefault();
        evt.stopPropagation();
    }
});

cy.on('drag', 'node', function(evt) {
    if (isDraggingLabel && draggedNode && rKeyPressed) {
        // ... dragging logic ...
    }
});
```

### Changes Made
- **File**: `vispath.py`
- **Section**: JavaScript (lines ~1070-1110)
- **Key Changes**:
  - Added global `rKeyPressed` state variable
  - Added `keydown` and `keyup` event listeners for R key
  - Changed condition from `evt.originalEvent.altKey` to `rKeyPressed`
  - Updated info text: "Alt+Drag" → "R+Drag"

### Benefits
- More reliable across different operating systems
- Clearer user feedback (hold R, then drag)
- No conflicts with OS-level keyboard shortcuts
- Releases label drag when R key is released

---

## Testing Recommendations

### 1. Sankey Diagram
- Run `PlotPath.py` with existing path data
- Verify Sankey diagram shows proper flow structure
- Check that node widths represent connection counts
- Confirm link widths represent synapse weights

### 2. Alpha Channel (Nodes)
- Open network visualization
- Open Style Panel (⚙️ icon)
- Adjust opacity sliders for source/intermediate/target nodes
- Click "Apply Changes"
- **Expected**: Nodes become transparent (not just legend)
- Test values: 0% (invisible), 50% (half), 100% (opaque)

### 3. Edge Colors
- Open Style Panel
- Select edge color (e.g., red #FF0000)
- Adjust edge opacity (e.g., 50%)
- Click "Apply Changes"
- **Expected**: All edges change color and transparency

### 4. Label Dragging (R+Drag)
- Open network visualization
- Press and hold 'R' key
- While holding R, click and drag a node
- **Expected**: Label moves independently from node
- Release R key
- **Expected**: Normal node dragging resumes

---

## Files Modified
- `vispath.py` (main visualization module)
  - `create_sankey()` method
  - HTML template (color palette section)
  - JavaScript functions (applyColors, label dragging, keyboard handlers)

---

## User-Facing Changes

### Updated Instructions
The info bar now shows:
```
Press 'H' to hide nodes, 'E' to hide edges, 'L' to toggle label position | 
R+Drag to move labels | Right-click to hide | Double-click to highlight
```

### Style Panel Enhancements
The collapsible Style Panel (⚙️) now includes:
1. **Font Family** - Dropdown with 10 fonts
2. **Source Nodes** - Color picker + opacity slider
3. **Intermediate Nodes** - Color picker + opacity slider
4. **Target Nodes** - Color picker + opacity slider
5. **Edge Color** - Color picker + opacity slider (NEW!)
6. **Apply Changes** - Button to apply all style updates

---

## Technical Notes

### Why .data() Instead of .style()?
Cytoscape.js allows two ways to set node properties:
1. **`.style('property', value)`** - Direct style override
2. **`.data('key', value)`** - Update data attribute

When a style rule uses `'data(key)'` (like `'background-color': 'data(color)'`), you must update the **data attribute** for changes to take effect. Direct style overrides are ignored.

### RGBA Format
Colors are converted from hex to RGBA using:
```javascript
function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    const a = alpha / 100;  // Convert 0-100 to 0.0-1.0
    return `rgba(${r}, ${g}, ${b}, ${a})`;
}
```

Example: `#FF5733` with 50% opacity → `rgba(255, 87, 51, 0.5)`

---

## Related Documentation
- `VISPATH_IMPLEMENTATION_SUMMARY.md` - Core visualization architecture
- `AdvancedVisualizationFeatures.md` - Feature documentation
- `VISPATH_COLOR_API_UPDATE.md` - Color system design

---

## Author
**Author**: Kang-Rui Leng  
**Date**: 2025-10-27  
Based on user feedback and `coana.py` reference implementation
