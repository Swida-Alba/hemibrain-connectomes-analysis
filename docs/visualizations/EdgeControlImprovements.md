# Edge Control and Label Alignment Improvements

**Date**: October 27, 2025  
**Component**: `vispath.py` - Edge hiding and hover label alignment

## Changes Made

### 1. ✅ Added Function to Hide Single Edges

**New Features:**

#### Right-Click on Edges
- Right-click (context menu click) on any edge to hide it
- Edge disappears from view immediately
- "Show All Nodes" button appears to restore hidden elements

#### Keyboard Shortcut 'E'
- Select one or more edges by clicking
- Press **'E'** to hide all selected edges
- Multiple edges can be hidden at once

**Implementation:**
```javascript
// Right-click to hide edges
cy.on('cxttap', 'edge', function(evt) {
    const edge = evt.target;
    edge.addClass('hidden');
    document.getElementById('showAllBtn').style.display = 'inline-block';
});

// Keyboard shortcut: E to hide selected edges
document.addEventListener('keydown', function(e) {
    if (e.key === 'e' || e.key === 'E') {
        const selected = cy.$('edge:selected');
        if (selected.length > 0) {
            selected.addClass('hidden');
            document.getElementById('showAllBtn').style.display = 'inline-block';
        }
    }
});
```

**Use Cases:**
- Clean up cluttered visualizations
- Focus on specific pathways by hiding irrelevant connections
- Remove weak connections from view
- Create cleaner screenshots by hiding unwanted edges

---

### 2. ✅ Aligned Edge Hover Label Text to the Left

**Problem**: Edge hover labels had centered text, making multi-line information harder to read.

**Solution**: Added left alignment and top vertical alignment for better readability.

**Added Properties:**
```javascript
'text-halign': 'left',   // Horizontal alignment: left
'text-valign': 'top',    // Vertical alignment: top
```

**Before (Centered):**
```
┌─────────────────────┐
│   Weight: 1,234     │
│    Ratio: 0.567     │
│ Probability: 0.890  │
└─────────────────────┘
```

**After (Left-Aligned):**
```
┌─────────────────────┐
│ Weight: 1,234       │
│ Ratio: 0.567        │
│ Probability: 0.890  │
└─────────────────────┘
```

**Benefits:**
- Easier to read multi-line information
- Professional appearance
- Consistent alignment like a list
- Better use of tooltip space

---

## Complete Feature Set for Hiding Elements

### Nodes
| Action | Method |
|--------|--------|
| Hide single node | Right-click on node |
| Hide selected nodes | Select nodes + press **'H'** |
| Hide node + edges | Right-click or press 'H' (edges auto-hide) |

### Edges
| Action | Method |
|--------|--------|
| Hide single edge | Right-click on edge |
| Hide selected edges | Select edges + press **'E'** |

### Restore
| Action | Method |
|--------|--------|
| Show all hidden elements | Click "Show All Nodes" button |

---

## Updated Keyboard Shortcuts

| Key | Target | Action |
|-----|--------|--------|
| **H** | Nodes | Hide selected nodes (and their edges) |
| **E** | Edges | Hide selected edges |
| **L** | Labels | Toggle label position (center ↔ outside) |

---

## Updated Control Panel Info Bar

**New Text:**
```
50 nodes, 123 connections | 
Press 'H' to hide nodes, 'E' to hide edges, 'L' to toggle label position | 
Right-click to hide | Double-click to highlight
```

**Features Documented:**
- ✅ Keyboard shortcuts (H, E, L)
- ✅ Right-click functionality
- ✅ Double-click highlight
- ✅ Node and edge counts

---

## Technical Implementation

### Edge Hover Label Style
```javascript
selector: 'edge',
style: {
    'width': 'mapData(weight, 0, max, 1, 10)',
    'line-color': '#999',
    'target-arrow-color': '#999',
    'target-arrow-shape': 'triangle',
    'curve-style': 'bezier',
    'arrow-scale': 1.5,
    'label': '',
    'font-size': '11px',
    'text-background-color': '#fff',
    'text-background-opacity': 0.9,
    'text-background-padding': '4px',
    'text-background-shape': 'roundrectangle',
    'text-border-color': '#999',
    'text-border-width': 1,
    'text-border-opacity': 0.5,
    'text-wrap': 'wrap',
    'text-max-width': '200px',
    'text-halign': 'left',      // NEW: Left align
    'text-valign': 'top',        // NEW: Top align
    'color': '#000'
}
```

---

## Usage Examples

### Example 1: Hide Weak Edges
```python
from vispath import VisualizePath

vp = VisualizePath('pathway_data.xlsx')
vp.visualize()

# In browser:
# 1. Identify edges with low weight (hover to see weight)
# 2. Right-click on weak edges to hide them
# 3. Clean visualization with only strong connections
```

### Example 2: Focus on Specific Pathway
```python
# In browser:
# 1. Select multiple unrelated edges (Ctrl+Click or Cmd+Click)
# 2. Press 'E' to hide all selected edges
# 3. Focus on the pathway of interest
# 4. Export PNG for presentation
```

### Example 3: Read Edge Information
```python
# In browser:
# 1. Hover over any edge
# 2. See tooltip with left-aligned information:
#    Weight: 1,234
#    Ratio: 0.567
#    Probability: 0.890
# 3. Easy to read and compare values
```

---

## Testing Checklist

### Test Edge Hiding

**Right-Click Method:**
- [ ] Right-click on an edge
- [ ] Edge disappears
- [ ] "Show All Nodes" button appears
- [ ] Click button to restore edge

**Keyboard Method:**
- [ ] Click to select an edge (turns gold)
- [ ] Press 'E' key
- [ ] Edge disappears
- [ ] Multiple edges can be selected and hidden together

### Test Label Alignment

- [ ] Hover over an edge with all three fields (weight, ratio, probability)
- [ ] Verify text is left-aligned in tooltip
- [ ] Verify text is top-aligned
- [ ] All three lines should align at the left edge
- [ ] No centering of individual lines

---

## Visual Examples

### Edge Tooltip Alignment

**Properly Aligned (Current):**
```
┌──────────────────────┐
│ Weight: 1,234        │ ← Left edge
│ Ratio: 0.567         │ ← Left edge
│ Probability: 0.890   │ ← Left edge
└──────────────────────┘
```

**Benefits:**
- Professional appearance
- Easy to scan values
- Consistent with standard UI patterns
- Better readability

---

## Performance Impact

- **Edge hiding**: O(1) - instant class addition
- **Label alignment**: No runtime cost (CSS only)
- **Show all**: O(E) where E = number of hidden edges
- **Memory**: Minimal (just CSS classes)

---

## Browser Compatibility

✅ All modern browsers support these features:
- Right-click events (`cxttap`)
- Keyboard events
- CSS text alignment
- Cytoscape.js class toggling

---

## Files Modified

**vispath.py:**
- Line ~697: Added `'text-halign': 'left'` to edge style
- Line ~698: Added `'text-valign': 'top'` to edge style
- Lines ~776-780: Added right-click handler for edges
- Lines ~792-800: Added keyboard shortcut 'E' for edge hiding
- Line ~637: Updated info text to document new features

---

## Summary

✅ **Hide single edges**: Right-click or press 'E' on selected edges  
✅ **Left-aligned labels**: Edge tooltips now show information in clean left-aligned format  
✅ **Better UX**: Consistent with node hiding, clear keyboard shortcuts  
✅ **Complete control**: Users can now hide both nodes and edges individually

---

**Author**: GitHub Copilot  
**Date**: October 27, 2025  
**Version**: v2.1
