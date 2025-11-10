# Network Edge Filtering Guide

## Overview

The interactive network visualization includes a powerful edge filtering system that allows you to hide edges based on their weight values. This is useful for:

- Focusing on strong/weak connections
- Simplifying complex networks
- Exploring connection patterns at different thresholds
- Temporarily hiding irrelevant edges without deleting them

## Location

Find the **"Hide Edges (weight):"** input box in the **Edge Controls** section (top-left of the visualization, below the Arrow Size slider).

## Basic Usage

### Exact Values

Hide specific edge weights by typing comma-separated values:

```
1
```
Hides all edges with weight = 1

```
1, 5, 10
```
Hides edges with weights 1, 5, or 10

```
0
```
Hides all zero-weight edges

### Comparison Operators

Use comparison operators to hide ranges of values:

| Operator | Meaning | Example | Result |
|----------|---------|---------|--------|
| `<` | Less than | `<5` | Hides edges with weight < 5 |
| `>` | Greater than | `>100` | Hides edges with weight > 100 |
| `<=` | Less than or equal | `<=10` | Hides edges with weight ≤ 10 |
| `>=` | Greater than or equal | `>=50` | Hides edges with weight ≥ 50 |
| `==` | Equal to | `==20` | Hides edges with weight = 20 |
| `!=` | Not equal to | `!=0` | Hides all non-zero edges |

### Combined Filters

Combine multiple conditions with commas (OR logic):

```
<5, >100
```
Hides edges with weight < 5 OR weight > 100

```
0, <10, >90
```
Hides edges with weight = 0 OR < 10 OR > 90

```
1, 3, >=100
```
Hides edges with weight = 1 OR = 3 OR ≥ 100

## Real-World Examples

### Example 1: Focus on Strong Connections

**Dataset:** Network with edge weights from 1 to 500

**Goal:** Hide weak connections to focus on main pathways

**Filter:**
```
<10
```

**Result:** Only edges with weight ≥ 10 remain visible, showing primary connections.

### Example 2: Remove Outliers

**Dataset:** Network with most weights 10-100, but a few extreme values

**Goal:** Focus on typical connections

**Filter:**
```
<10, >100
```

**Result:** Hides both very weak and very strong edges, showing the typical range.

### Example 3: Exclude Specific Values

**Dataset:** Network where weight = 1 represents uncertain connections

**Goal:** Show only confident connections

**Filter:**
```
1
```

**Result:** All edges with weight = 1 are hidden.

### Example 4: Complex Filtering

**Dataset:** Network with diverse edge weights

**Goal:** Show only medium-strength connections

**Filter:**
```
<=5, >=100
```

**Result:** Hides edges with weight ≤ 5 and ≥ 100, leaving only medium values visible.

## Visual Feedback

### Status Display

The status area (bottom-left) shows real-time information:

```
🔍 Edge filter: 15 shown, 4 hidden
```

When no filter is active:
```
🔍 Edge filter: All edges visible
```

### Hidden Edges

- Hidden edges are completely invisible (not just faded)
- Node positions and labels remain unchanged
- Hidden edges can be instantly restored by clearing the filter

## Advanced Features

### Decimal Values

The filter works with decimal edge weights:

```
<2.5
```
Hides edges with weight < 2.5

```
1.5, 3.7, >=10.5
```
Supports mixed decimal and integer values

### Negative Weights

If your network has negative edge weights (e.g., inhibitory connections):

```
<0
```
Hides all negative edges

```
>=-50, <0
```
Hides edges from -50 to -1

### Whitespace Tolerance

The filter ignores extra spaces:

```
< 5 , > 100
```
Works the same as `<5, >100`

### Empty Filter

Deleting all text from the input instantly shows all edges again.

## Technical Details

### How It Works

1. **Parsing:** Input is split by commas and each part is analyzed
2. **Exact Match:** Plain numbers are stored in a Set for fast lookup
3. **Expressions:** Operators are parsed into expression objects with operator and threshold
4. **Application:** Each edge's weight is checked against exact values and expressions
5. **Display:** Matching edges get CSS `display: 'none'`, others get `display: 'element'`

### Performance

- Filter updates in real-time as you type
- Efficient for networks with thousands of edges
- Uses Cytoscape's native styling system

### Edge Weight Source

The filter uses `original_weight` data field if available, falling back to `weight`. This ensures negative weights are handled correctly after any transformations.

## Integration with Export/Import

### Settings Preservation

Edge filter settings are automatically saved when you export the graph:

```json
{
  "settings": {
    "edgeFilter": {
      "inputValue": "<5, >100",
      "ignoredValues": [],
      "expressions": [
        {"operator": "<", "threshold": 5},
        {"operator": ">", "threshold": 100}
      ]
    }
  }
}
```

### Restoration on Import

When you import a graph:
1. The filter input is populated with the saved expression
2. The filter is automatically applied
3. Hidden edges remain hidden

See [NETWORK_EXPORT_IMPORT.md](NETWORK_EXPORT_IMPORT.md) for details.

## Tips & Best Practices

### 1. Start Broad, Then Refine

Begin with a simple filter like `<10`, then adjust based on what you see.

### 2. Use Status Display

Always check the "X shown, Y hidden" message to ensure your filter is working as expected.

### 3. Compare Views

Toggle between filtered and unfiltered views by saving the expression, clearing it, then pasting it back.

### 4. Combine with Other Features

- Use filtering with **Layout Export** to save clean, filtered layouts
- Combine with **Edit Mode** to modify visible edges
- Use with **Color Palette** to highlight remaining edges

### 5. Document Your Filters

When exporting results, note the filter expression in your documentation:
```
Analysis performed with edge filter: <5
Network shows only connections with weight ≥ 5
```

### 6. Test Incrementally

Try different thresholds to find the best view:
```
<5    → Still too cluttered
<10   → Better, but could be cleaner  
<20   → Perfect balance!
```

## Troubleshooting

### Filter Not Working

**Issue:** Typing in the filter box doesn't hide any edges

**Solutions:**
- Check browser console (F12) for errors
- Verify your syntax (no spaces around operators)
- Ensure edge weights are numeric
- Try a simpler filter first (e.g., just `<5`)

### Wrong Edges Hidden

**Issue:** The filter hides unexpected edges

**Solutions:**
- Double-check operator direction: `<5` hides edges LESS than 5
- Hover over edges to see their actual weights
- Remember: filters use OR logic (any match hides the edge)
- Clear filter and try one condition at a time

### Can't Clear Filter

**Issue:** Deleting text doesn't restore edges

**Solutions:**
- Ensure input box is completely empty (no spaces)
- Refresh the page if needed
- Check console for JavaScript errors

### Filter Not Saved

**Issue:** Filter resets after export/import

**Solutions:**
- Verify you're using Export format v2.0 (not v1.0)
- Check the JSON file contains `"settings"` object
- See [NETWORK_EXPORT_IMPORT.md](NETWORK_EXPORT_IMPORT.md)

## Keyboard Shortcuts

- **Enter:** Apply filter (re-applies on every keystroke anyway)
- **Escape:** Clear filter (select all text first)
- **Ctrl+A / Cmd+A:** Select all text in filter input
- **Ctrl+Z / Cmd+Z:** Undo text changes in filter input

## Console Logging

For debugging, the filter logs to the browser console:

```javascript
Parsed ignored edges: ["<5"]
  Exact values to ignore: []
  Expressions: [{"operator":"<","threshold":5}]
Applying edge filter...
  Edge filter complete: 6 shown, 3 hidden
```

Open DevTools (F12) to see these messages.

## API Reference

### updateIgnoredEdges()

Parses the filter input and updates the ignored edges set and expressions array.

**Called:** Automatically on input change (`oninput` event)

**Updates:**
- `ignoredEdges` - Set of exact values to hide
- `ignoredEdgeExpressions` - Array of `{operator, threshold}` objects

### shouldIgnoreEdge(weight)

Checks if an edge with the given weight should be hidden.

**Parameters:**
- `weight` (number) - The edge weight to check

**Returns:** `true` if edge should be hidden, `false` otherwise

**Logic:**
1. Check exact match in `ignoredEdges` Set
2. Evaluate all expressions in `ignoredEdgeExpressions`
3. Return `true` if any match found

### applyEdgeFilter()

Applies the current filter to all edges in the network.

**Called:** Automatically after parsing the filter input

**Actions:**
1. Iterates through all edges using `cy.edges()`
2. Gets edge weight from `original_weight` or `weight` data
3. Calls `shouldIgnoreEdge()` for each edge
4. Sets edge style `display: 'none'` or `display: 'element'`
5. Updates hover info with counts

## Related Documentation

- [NETWORK_EXPORT_IMPORT.md](NETWORK_EXPORT_IMPORT.md) - Saving and loading graphs with settings
- [NETWORK_INTERACTIVE_EDITING.md](NETWORK_INTERACTIVE_EDITING.md) - Edit mode and manual edge manipulation
- [EDGE_WIDTH_SCALING.md](EDGE_WIDTH_SCALING.md) - Edge width visualization methods
- [VisualizePath Guide](../README.md) - Main visualization documentation

## Version History

- **v2.0** (Nov 2025) - Added edge filtering with comparison operators and export/import integration
- **v1.0** (Oct 2025) - Initial release with basic graph export
