# Sankey Enhancements - Quick Reference

## What's New

### ❌ Removed
- **Info-box overlay** that covered the left side of the diagram

### ✅ Added
- **🔍 Zoom In/Out** - Scale diagram from 30% to 300%
- **⟲ Reset Zoom** - Return to 100% scale
- **🏷️ Toggle Labels** - Show/hide node names

---

## Quick Start

### Test New Features
```bash
python test_sankey_features.py
open ./test_output/sankey_features/sankey_selected_paths.html
```

### In the Browser
1. **Zoom**: Click "Zoom In" or "Zoom Out" buttons
2. **Reset**: Click "Reset Zoom" to return to 100%
3. **Labels**: Click "Toggle Labels" to hide/show node names

---

## Control Panel

```
┌──────────────────────────────────────────────────────────┐
│  Colors          Edge Settings       Actions             │
├──────────────────────────────────────────────────────────┤
│ Source: ■        Edge Color: ■      [Apply]              │
│ Inter:  ■        Opacity: ───○───   [Reset]              │
│ Target: ■                            [Show/Hide]          │
│                                                           │
│ Zoom & Display                                           │
├──────────────────────────────────────────────────────────┤
│ [🔍 Zoom In]  [🔍 Zoom Out]  [⟲ Reset]  [🏷️ Labels]    │
└──────────────────────────────────────────────────────────┘
```

---

## Zoom Behavior

| Action | Effect | Range |
|--------|--------|-------|
| Zoom In | +20% per click | Up to 300% |
| Zoom Out | -20% per click | Down to 30% |
| Reset Zoom | Return to 100% | - |

**How it works**: CSS `transform: scale()` - smooth, instant, GPU-accelerated

---

## Label Toggle Behavior

| State | Node Labels | Hidden Nodes |
|-------|-------------|--------------|
| ON (default) | Visible | No labels |
| OFF | Hidden | No labels |

**How it works**: Updates Plotly data array, works with visibility controls

---

## Complete Example

```python
import pandas as pd
from vispath import VisualizePath

# Network with edge colors
network = pd.DataFrame({
    'source': ['A', 'B', 'C'],
    'target': ['B', 'C', 'D'],
    'weight': [10, 15, 8],
    'color': ['#FF6B6B', '#4ECDC4', '#45B7D1']
})

# Custom node colors
node_colors = pd.DataFrame({
    'node': ['A', 'B', 'C', 'D'],
    'color': ['#E91E63', '#9C27B0', '#673AB7', '#4CAF50']
})

# Create visualization with all features
vp = VisualizePath(
    path_file=network,
    node_colors=node_colors,
    showfig=True
)
vp.visualize()

# In browser:
# - No info-box blocking view ✅
# - Click "Zoom In" to inspect details 🔍
# - Click "Toggle Labels" for clean view 🏷️
# - Adjust colors as needed 🎨
```

---

## All Available Controls

### Color Controls
- Source node color picker
- Intermediate node color picker
- Target node color picker
- Edge color picker
- Edge opacity slider (0-100%)

### Visibility Controls (Show/Hide panel)
- Toggle individual nodes
- Toggle individual edges
- Show All / Hide All buttons

### Zoom Controls
- Zoom In (max 300%)
- Zoom Out (min 30%)
- Reset Zoom (100%)

### Display Controls
- Toggle node labels
- Apply color changes
- Reset to defaults

---

## Tips

### For Large Diagrams
1. Toggle labels OFF for cleaner view
2. Zoom IN to inspect specific regions
3. Use Show/Hide panel to focus on paths of interest

### For Presentations
1. Adjust colors for better contrast
2. Toggle labels ON when explaining
3. Toggle labels OFF to show flow patterns
4. Use zoom for detailed discussions

### For Analysis
1. Use custom colors to highlight categories
2. Zoom in to examine connection strengths
3. Hide irrelevant paths
4. Toggle labels based on audience familiarity

---

## Keyboard Shortcuts (Future)

*Currently all controls are button-based. Future versions may include:*
- `+` / `-` for zoom
- `L` for label toggle
- `R` for reset
- Arrow keys for pan when zoomed

---

## Files

- `vispath.py` - Enhanced implementation
- `test_sankey_features.py` - Comprehensive test
- `SANKEY_INTERACTIVE_CONTROLS.md` - Full documentation
- `CUSTOM_COLORS_GUIDE.md` - Color customization guide

---

## Summary

✅ **Removed**: Info-box overlay  
✅ **Added**: Zoom controls (30%-300%)  
✅ **Added**: Label toggle  
✅ **Maintained**: All existing features  
✅ **Tested**: Complete test script provided  

**Result**: Cleaner, more interactive, more flexible Sankey diagrams! 🎉
