# Node Width Slider - Quick Guide

## Overview

Added a **Node Width** slider to adjust the thickness of nodes in the Sankey diagram dynamically.

---

## New Control

### Node Width Slider

**Location**: Control panel, after "Edge Opacity"

**Range**: 1 to 50 pixels
**Default**: 5 pixels

```
Control Panel Layout:
┌─────────────────────────────────────────┐
│ Edge Settings                           │
├─────────────────────────────────────────┤
│ Edge Color:    ■                        │
│ Edge Opacity:  ──────○───── 30%        │
│ Node Width:    ──────○───── 5    ← NEW │
└─────────────────────────────────────────┘
```

---

## How It Works

**Real-time Updates**: Moving the slider instantly updates the node width - no need to click "Apply"!

**Visual Effect**:
- **Thin nodes (1-3)**: Minimalist, focus on flow connections
- **Default (5)**: Balanced visibility
- **Thick nodes (10-20)**: Emphasize node importance
- **Very thick (30-50)**: Maximum node prominence

---

## Use Cases

### 1. Dense Networks
```
Problem: Many nodes, hard to see individual nodes
Solution: Reduce node width to 2-3 for cleaner view
```

### 2. Node-Focused Analysis
```
Problem: Want to emphasize nodes over connections
Solution: Increase node width to 15-20
```

### 3. Presentations
```
Problem: Need clear visibility for projectors
Solution: Increase node width to 10-15
```

### 4. Flow-Focused View
```
Problem: Connections are more important than nodes
Solution: Reduce node width to 1-2, emphasize flow
```

---

## Examples

### Minimal Nodes (Width = 2)
```python
from vispath import VisualizePath

vp = VisualizePath('network.csv')
vp.visualize()
# In browser: Move "Node Width" slider to 2
# Result: Thin nodes, emphasis on flow patterns
```

### Prominent Nodes (Width = 20)
```python
vp = VisualizePath('network.csv')
vp.visualize()
# In browser: Move "Node Width" slider to 20
# Result: Thick nodes, node names clearly visible
```

---

## Technical Details

**Implementation**: 
- Plotly Sankey `node.thickness` parameter
- Real-time update using `Plotly.react()`
- No page reload needed

**Performance**: 
- Instant visual feedback
- No performance impact
- Works smoothly with all other controls

---

## Interaction with Other Features

### ✅ Compatible With:
- Zoom controls (zoom after adjusting width)
- Label toggle (works regardless of width)
- Custom colors (width doesn't affect colors)
- Show/Hide controls (hidden nodes stay hidden)
- Edge opacity (independent settings)

### 💡 Tips:
1. **Adjust width before zooming** for best results
2. **Reset zoom** if nodes look distorted after width change
3. **Combine with label toggle** - hide labels for thin nodes
4. **Use thick nodes** when labels are important

---

## Complete Control Panel

```
┌──────────────────────────────────────────────────────────┐
│ Node Colors                                              │
│   Source: ■  Intermediate: ■  Target: ■                 │
│                                                          │
│ Edge Settings                                            │
│   Color: ■  Opacity: ───○─── 30%  Width: ───○─── 5  ←NEW│
│                                                          │
│ Actions                                                  │
│   [Apply] [Reset] [Show/Hide]                           │
│                                                          │
│ Zoom & Display                                          │
│   [🔍 +] [🔍 -] [⟲ Reset] [🏷️ Labels]                  │
└──────────────────────────────────────────────────────────┘
```

---

## Reset Behavior

The **Reset** button now resets:
- ✅ All color selections → defaults
- ✅ Edge opacity → default (30%)
- ✅ **Node width → 5** (default)
- ✅ Node visibility → all shown
- ✅ Edge visibility → all shown

---

## FAQ

**Q: Why isn't my width change visible?**
A: Make sure you haven't zoomed too far out. Reset zoom and try again.

**Q: Can I save the node width setting?**
A: Currently width resets when you reload. Export the HTML if you need to preserve the view.

**Q: What's the best width for screenshots?**
A: Use 8-12 for clear, professional screenshots.

**Q: Does width affect performance?**
A: No, width is a visual-only setting with no performance impact.

---

## Comparison: Cytoscape vs Plotly for Sankey

### Why Not Cytoscape for Sankey?

**Cytoscape.js** (used for network visualization):
- ❌ No native Sankey support
- ❌ Can't render variable-width flow links
- ❌ No flow-based layout algorithms
- ✅ Great for node-link networks

**Plotly** (current Sankey implementation):
- ✅ Native Sankey diagram support
- ✅ Variable-width links based on flow
- ✅ Automatic layered layout
- ✅ Flow proportions visualized correctly
- ✅ Better suited for hierarchical flow data

**Recommendation**: 
- Use **Plotly** for Sankey (flow diagrams)
- Use **Cytoscape** for networks (node-link graphs)
- They serve different purposes and excel at different visualizations

### Visual Uniformity

Both visualizations now have **consistent controls**:
- Color pickers
- Opacity sliders
- Show/Hide toggles
- Zoom controls
- Label toggles

The UI is **unified** even though the underlying libraries are different!

---

## Summary

✅ **Added**: Node Width slider (1-50 pixels)  
✅ **Real-time**: Instant updates, no Apply needed  
✅ **Flexible**: Thin to very thick nodes  
✅ **Integrated**: Works with all existing controls  
✅ **Reset**: Included in Reset button  

**Result**: More control over Sankey appearance for different use cases! 🎨
