# VisualizePath Color API Update

## Summary of Changes

Updated the `VisualizePath` class to support **individual color parameters** (`source_color`, `intermediate_color`) instead of requiring an array (`node_color`), while maintaining full backward compatibility.

## What Changed

### 1. VisualizePath Class (`vispath.py`)

**New Parameters:**
```python
VisualizePath(
    path_file='path_type.xlsx',
    source_color='#FF6B6B',         # NEW: Individual source color
    intermediate_color='#FFA500',    # NEW: Individual intermediate color
    target_color='#FFD700',          # Enhanced
    link_color='rgba(255,0,0,0.3)',  # Enhanced
    node_color=None,                 # DEPRECATED but still supported
    ...
)
```

**Benefits:**
- ✅ More intuitive - each neuron type has its own parameter
- ✅ Easier to customize - change one color without touching others
- ✅ Clearer code - no need to remember array positions
- ✅ Backward compatible - old `node_color` still works

### 2. Backward Compatibility Logic

The class automatically handles both old and new APIs:

```python
# New API (preferred)
vp = VisualizePath(
    'path.xlsx',
    source_color='#FF6B6B',
    intermediate_color='#FFA500'
)

# Old API (still works)
vp = VisualizePath(
    'path.xlsx',
    node_color=['#FF6B6B', '#FFA500']
)

# Mixed (new takes precedence)
vp = VisualizePath(
    'path.xlsx',
    source_color='#FF6B6B',           # This is used
    node_color=['#000000', '#FFA500']  # intermediate from here is used
)
```

### 3. FindNeuronConnection Wrapper (`coana.py`)

Updated `fc.VisualizeSelectedPaths()` to support new parameters:

```python
fc.VisualizeSelectedPaths(
    path_file='path.xlsx',
    source_color='#FF6B6B',
    intermediate_color='#FFA500',
    target_color='#FFD700'
)
```

### 4. PlotPath.py Example Script

Now uses the new, cleaner API:

```python
# Define colors individually
source_color = '#1f77b4'
intermediate_color = '#2ca02c'
target_color = '#d62728'
link_color = 'rgba(100,100,100,0.3)'

# Pass directly to VisualizePath
vp = VisualizePath(
    path_file=path_file,
    source_color=source_color,
    intermediate_color=intermediate_color,
    target_color=target_color,
    link_color=link_color
)
```

## Migration Guide

### For New Code

**Use the new individual color parameters:**

```python
from vispath import VisualizePath

vp = VisualizePath(
    'path_type.xlsx',
    source_color='#FF6B6B',         # Red
    intermediate_color='#FFA500',    # Orange
    target_color='#FFD700',          # Gold
    showfig=True
)

conn_df, G = vp.visualize()
```

### For Existing Code

**No changes needed!** Old code continues to work:

```python
# This still works perfectly
vp = VisualizePath(
    'path_type.xlsx',
    node_color=['#FF6B6B', '#FFA500'],  # Still supported
    target_color='#FFD700',
    showfig=True
)
```

### Mixing Old and New

You can mix both APIs, but new parameters take precedence:

```python
vp = VisualizePath(
    'path_type.xlsx',
    source_color='#FF0000',              # This is used for source
    node_color=['#000000', '#00FF00'],   # intermediate_color from here is used
    target_color='#0000FF'
)
# Result: source=#FF0000, intermediate=#00FF00, target=#0000FF
```

## API Comparison

### Old API (Still Supported)

```python
# Had to combine colors into array
node_color = ['#1f77b4', '#2ca02c']  # [source, intermediate]

vp = VisualizePath(
    'path.xlsx',
    node_color=node_color,
    target_color='#d62728'
)
```

**Issues:**
- Need to remember array order
- Can't set just one color easily
- Less intuitive

### New API (Recommended)

```python
# Individual colors - much clearer!
vp = VisualizePath(
    'path.xlsx',
    source_color='#1f77b4',
    intermediate_color='#2ca02c',
    target_color='#d62728'
)
```

**Benefits:**
- ✅ Self-documenting code
- ✅ Easy to change individual colors
- ✅ No need to remember positions
- ✅ More flexible

## Color Parameter Details

### source_color
- **Type:** `str`
- **Default:** `'#1f77b4'` (blue)
- **Description:** Color for source neurons (starting points of paths)
- **Format:** Any CSS color (hex, rgb, rgba, named)

### intermediate_color
- **Type:** `str`
- **Default:** `'#2ca02c'` (green)
- **Description:** Color for intermediate neurons (middle nodes in paths)
- **Format:** Any CSS color (hex, rgb, rgba, named)

### target_color
- **Type:** `str`
- **Default:** `'#d62728'` (red)
- **Description:** Color for target neurons (endpoints of paths)
- **Format:** Any CSS color (hex, rgb, rgba, named)

### link_color
- **Type:** `str`
- **Default:** `'rgba(100,100,100,0.3)'` (semi-transparent gray)
- **Description:** Color for connections in Sankey diagram
- **Format:** Preferably RGBA for transparency

### node_color (Deprecated)
- **Type:** `list of 2 strings`
- **Default:** `['#1f77b4', '#2ca02c']`
- **Description:** `[source_color, intermediate_color]`
- **Status:** **DEPRECATED** - Use `source_color` and `intermediate_color` instead
- **Note:** Still supported for backward compatibility

## Examples

### Example 1: Basic Usage

```python
from vispath import VisualizePath

vp = VisualizePath(
    'path_type.xlsx',
    source_color='#1f77b4',      # Blue
    intermediate_color='#2ca02c', # Green
    target_color='#d62728'        # Red
)

vp.visualize()
```

### Example 2: Warm Color Theme

```python
vp = VisualizePath(
    'path_type.xlsx',
    source_color='#FF6B6B',      # Red
    intermediate_color='#FFA500', # Orange
    target_color='#FFD700',       # Gold
    link_color='rgba(255,107,107,0.3)'
)

vp.visualize()
```

### Example 3: Custom Mix

```python
# Mix colors from different themes
vp = VisualizePath(
    'path_type.xlsx',
    source_color='#FF6B6B',      # Red from warm theme
    intermediate_color='#50E3C2', # Cyan from cool theme
    target_color='#9C27B0',       # Purple from purple theme
    link_color='rgba(156,39,176,0.3)'
)

vp.visualize()
```

### Example 4: Through FindNeuronConnection

```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(...)

# New API works here too!
fc.VisualizeSelectedPaths(
    'path_type.xlsx',
    source_color='#FF6B6B',
    intermediate_color='#FFA500',
    target_color='#FFD700'
)
```

## Testing

All changes have been tested for:
- ✅ New API works correctly
- ✅ Old API still works (backward compatibility)
- ✅ Mixed API works as expected
- ✅ Default colors applied when none specified
- ✅ No syntax errors in any modified files

## Documentation Updates

Updated documentation in:
- `vispath.py` - Class docstring and parameter descriptions
- `coana.py` - Method docstring
- `PlotPath.py` - Example usage with new API
- This change log

## Recommendation

**For all new code:** Use the new individual color parameters (`source_color`, `intermediate_color`, `target_color`)

**For existing code:** No changes needed, but consider migrating to new API for better clarity

## Summary

The update provides a more intuitive and flexible color API while maintaining 100% backward compatibility. Users can now easily customize individual neuron type colors without dealing with array positions.
