# Multi-Selection Color Feature

## Overview

The network visualization now supports **applying color and opacity changes to multiple selected elements at once**. This leverages Cytoscape.js's built-in multi-selection capability to make batch customization quick and easy.

## Features

### ✅ Multi-Selection Support
- Select multiple nodes and/or edges simultaneously
- Visual feedback with yellow borders on selected elements
- Selection count displayed in the color panel

### ✅ Batch Color Application
- Change color for all selected elements at once
- Adjust opacity for all selected elements at once
- Mix of nodes and edges can be selected together

### ✅ Smart UI Updates
- Panel shows how many elements are currently selected
- Displays type breakdown (nodes vs edges)
- Shows which element's colors are being used as reference

## How to Use

### Basic Selection

**Single Selection:**
- Click on any node or edge to select it

**Multi-Selection:**
- Hold **Shift** and click on multiple elements
- Each element will show a yellow border when selected
- The color panel shows the total count

### Applying Colors

1. **Select elements:**
   - Hold Shift and click on nodes/edges you want to change
   - Example: Select all intermediate nodes

2. **Adjust color:**
   - Click the color picker in the "Selected Element(s)" section
   - Choose your desired color

3. **Adjust opacity:**
   - Use the opacity slider (0-100%)
   - Preview the value in real-time

4. **Apply:**
   - Click "Apply to Selected" button
   - All selected elements update simultaneously!

### Clear Selection

Click the "Clear Selection" button to deselect all elements and hide the controls.

## Visual Guide

```
Before:
  [Node A] [Node B] [Node C]  (all blue)
  
Action:
  1. Shift+Click on Node A
  2. Shift+Click on Node B
  3. Change color to red (#FF0000)
  4. Set opacity to 50%
  5. Click "Apply to Selected"
  
After:
  [Node A] [Node B] [Node C]
   (red)    (red)    (blue)
   50%      50%      100%
```

## UI Elements

### Color Panel - Selected Element(s) Section

**When no selection:**
```
Click on a node or edge to customize its color
Hold Shift to select multiple elements
```

**When single element selected:**
```
Node: Process_1 (intermediate)
[Color picker] [Opacity slider]
[Apply to Selected] [Clear Selection]
```

**When multiple elements selected:**
```
Multi-Selection:
2 node(s), 3 edge(s)
Colors from: Process_1

[Color picker] [Opacity slider]
[Apply to Selected] [Clear Selection]
```

**After applying:**
```
✓ Updated:
2 node(s), 3 edge(s)
```

## Technical Details

### Implementation

**JavaScript Functions:**

1. **`getSelectedElements()`**
   - Returns all currently selected elements using `cy.$(':selected')`
   - Supports both nodes and edges

2. **`getSelectionCount()`**
   - Returns object: `{ nodes: X, edges: Y, total: Z }`
   - Used for UI display

3. **`applyIndividualColor()`** (Enhanced)
   - Loops through all selected elements
   - Applies color/opacity to each
   - Marks each as customized
   - Shows update count

4. **`clearSelection()`** (Enhanced)
   - Deselects all elements using `cy.$(':selected').unselect()`
   - Resets UI state

### Selection Handler (Enhanced)

```javascript
cy.on('tap', 'node, edge', function(evt) {
    const element = evt.target;
    const selectionCount = getSelectionCount();
    
    // Update UI based on selection count
    if (selectionCount.total > 1) {
        // Show multi-selection info
    } else {
        // Show single element info
    }
});
```

### Apply Function (Enhanced)

```javascript
function applyIndividualColor() {
    const selectedElements = getSelectedElements();
    
    selectedElements.forEach(function(element) {
        if (element.isNode()) {
            element.style({
                'background-color': color,
                'opacity': opacity
            });
        } else {
            element.style({
                'line-color': color,
                'target-arrow-color': color,
                'opacity': opacity
            });
        }
    });
}
```

## Use Cases

### 1. Highlight Specific Pathway
Select all nodes in a particular pathway and make them stand out:
- Shift+Click on pathway nodes
- Set color to bright orange (#FF6B00)
- Set opacity to 100%
- Apply

### 2. Fade Background Nodes
De-emphasize nodes not in focus:
- Shift+Click on background nodes
- Keep original color
- Set opacity to 30%
- Apply

### 3. Color-Code by Function
Group nodes by functional category:
- Select all sensory neurons → Color green
- Select all motor neurons → Color red
- Select all interneurons → Color blue

### 4. Highlight Connections
Emphasize important edges:
- Shift+Click on strong connections
- Set color to bold red (#FF0000)
- Set opacity to 100%
- Make thickness stand out

### 5. Create Visual Clusters
Distinguish different subnetworks:
- Select cluster 1 → Purple (#9B59B6)
- Select cluster 2 → Orange (#E67E22)
- Select cluster 3 → Teal (#1ABC9C)

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Multi-select | Hold **Shift** + Click |
| Deselect all | Click "Clear Selection" or click on empty space |

## Tips & Tricks

### Efficient Selection
1. Start with the first element (click)
2. Hold Shift for all subsequent clicks
3. Yellow borders confirm selection

### Color Consistency
- The color picker shows the color of the last-clicked element
- This serves as your reference point
- All selected elements will receive this color when applied

### Mixed Selection
- You can select both nodes AND edges together
- Each type is colored appropriately (nodes: background, edges: line)
- The update count shows breakdown by type

### Undo Colors
To reset customized colors:
1. Select the customized elements
2. Use the "Node Colors by Type" section
3. Click "Apply to All" to reset to default colors

## Comparison: Before vs After

### Before (Single Selection Only)
```
Click Node A → Change color → Apply
Click Node B → Change color → Apply
Click Node C → Change color → Apply
Result: 3 separate actions needed
```

### After (Multi-Selection)
```
Shift+Click Node A
Shift+Click Node B
Shift+Click Node C
Change color → Apply
Result: 1 action for all 3 nodes!
```

**Time saved:** ~66% for 3 elements, more for larger selections!

## Examples

### Example 1: Color All Source Nodes Red

```python
# Create visualization
vis = VisualizePath(path_file=df)
vis.create_network()

# In browser:
# 1. Shift+Click on all blue source nodes
# 2. Color picker → #FF0000 (red)
# 3. Opacity → 80%
# 4. Apply to Selected
# Result: All source nodes are now red at 80% opacity
```

### Example 2: Fade Weak Connections

```python
# In browser:
# 1. Identify weak edges (thin lines)
# 2. Shift+Click to select multiple weak edges
# 3. Keep gray color
# 4. Opacity → 20%
# 5. Apply to Selected
# Result: Weak connections are faded, strong ones stand out
```

### Example 3: Highlight Critical Path

```python
# In browser:
# 1. Shift+Click on nodes: Input_A → Process_1 → Output_X
# 2. Color → #FFD700 (gold)
# 3. Opacity → 100%
# 4. Apply to Selected
# 5. Shift+Click on edges between these nodes
# 6. Color → #FFD700 (gold)
# 7. Opacity → 100%
# 8. Apply to Selected
# Result: Critical path highlighted in gold
```

## Browser Compatibility

Works in all modern browsers:
- ✅ Chrome/Edge (Recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Opera

## Performance

- Efficient for networks up to 1000+ nodes
- Batch updates are applied in a single pass
- No performance degradation with multi-selection

## Related Features

- **Global Color Changes:** Use "Node Colors by Type" section
- **Individual Customization:** Click single element
- **Reset Colors:** "Apply to All" resets customizations

## Testing

Run the test:
```bash
python test_multi_selection.py
```

This creates a demo network where you can test:
- Multi-selection with Shift+Click
- Batch color application
- Selection count display
- Clear selection functionality

## Summary

### Key Benefits
✅ **Faster workflow** - Update multiple elements at once  
✅ **Intuitive selection** - Standard Shift+Click behavior  
✅ **Visual feedback** - Yellow borders show selection  
✅ **Smart UI** - Shows selection count and breakdown  
✅ **Flexible** - Mix nodes and edges in one selection  

### Quick Reference
- **Select multiple:** Hold Shift + Click
- **See count:** Check color panel
- **Apply colors:** One button for all selected
- **Clear:** Click "Clear Selection"

**Now you can customize your network visualizations faster than ever!** 🎨✨
