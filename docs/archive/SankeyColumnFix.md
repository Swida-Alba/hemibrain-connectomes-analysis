# Sankey Column Names Fix - October 27, 2025

## Issue
When running `PlotPath.py`, the script crashed with:
```python
KeyError: 'neuron_pre'
```

## Root Cause
**Column name mismatch** between two methods in `vispath.py`:

### `build_network()` method creates:
```python
connections.append({
    'source': source,      # ← Uses 'source'
    'target': target,      # ← Uses 'target'
    'weight': weight,
    'ratio': ratio,
    'probability': prob
})
```

### `create_sankey()` method expected:
```python
# INCORRECT - these columns don't exist!
conn_grouped = self.conn_df.groupby(['neuron_pre', 'neuron_post'])['weight'].sum()
```

The column names `'neuron_pre'` and `'neuron_post'` are used in `coana.py` for its connection DataFrames, but `vispath.py` uses different naming conventions.

## Solution
Changed `create_sankey()` to use the correct column names:

```python
# CORRECT - matches build_network() output
conn_grouped = self.conn_df.groupby(['source', 'target'])['weight'].sum().reset_index()

# Create pivot table
conn_matrix = conn_grouped.pivot_table(
    index='source',      # ← Changed from 'neuron_pre'
    columns='target',    # ← Changed from 'neuron_post'
    values='weight',
    fill_value=0
)
```

## Testing
Successfully ran `PlotPath.py`:
```
✓ Loaded 5 pathways from data
✓ Created 8 unique connections from pathways
✓ Connection matrix: 4 sources × 4 targets
✓ Total connections: 8
✓ Sankey diagram saved successfully
✓ Network visualization saved successfully
```

## Files Modified
- `vispath.py` - Line ~396, `create_sankey()` method
  - Changed `'neuron_pre'` → `'source'`
  - Changed `'neuron_post'` → `'target'`

## Lesson Learned
When copying code patterns from one module (`coana.py`) to another (`vispath.py`), always verify:
1. ✅ Column names match between methods
2. ✅ DataFrame structure is consistent
3. ✅ Variable naming conventions are aligned

## Related Fixes
This is part of the Sankey diagram overhaul documented in:
- `docs/VisualizationFixes_2025-10-27.md` - Main fix documentation
- This fix completes the Sankey implementation

---

**Status**: ✅ RESOLVED  
**Date**: October 27, 2025  
**Impact**: Critical - Script was completely broken, now fully functional
