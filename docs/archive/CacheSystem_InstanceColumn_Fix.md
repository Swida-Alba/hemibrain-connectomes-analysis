# Fix: Cache Warning "['instance_pre'] not in index"

## Issue

When fetching connections and caching data, the following warning appeared:

```
⚠️ Warning: Failed to cache data: "['instance_pre'] not in index"
```

## Root Cause

The `_update_neuron_registry()` function was attempting to access columns that might not exist in the connection dataframe:

```python
# OLD CODE (PROBLEMATIC):
def _update_neuron_registry(self, conn_df):
    # ...
    if 'bodyId_pre' in conn_df.columns:
        upstream = conn_df[['bodyId_pre', 'type_pre', 'instance_pre']].copy()
        # ERROR: 'instance_pre' might not exist!
```

### Why This Happened

Different neuPrint API fetch methods return different columns:
- `fetch_simple_connections()`: Returns `type_pre`, `type_post` but **NOT** `instance_pre`, `instance_post`
- `fetch_adjacencies()`: May or may not include instance columns depending on dataset
- Some datasets (like optic-lobe) may not have instance information for all neurons

The code was assuming **all** columns would always be present, causing a KeyError when trying to access missing columns.

## Solution

Made the `_update_neuron_registry()` function more robust by:

1. **Checking column existence** before accessing
2. **Building column lists dynamically** based on what's available
3. **Only updating registry** when we have useful information (type or instance)

### Fixed Code

```python
def _update_neuron_registry(self, conn_df):
    '''Update neuron registry with neurons from connection data'''
    registry = self._load_neuron_registry()
    new_neurons = pd.DataFrame()
    
    # Upstream neurons - check which columns are available
    if 'bodyId_pre' in conn_df.columns:
        upstream_cols = ['bodyId_pre']
        rename_map = {'bodyId_pre': 'bodyId'}
        
        if 'type_pre' in conn_df.columns:
            upstream_cols.append('type_pre')
            rename_map['type_pre'] = 'type'
        
        if 'instance_pre' in conn_df.columns:
            upstream_cols.append('instance_pre')
            rename_map['instance_pre'] = 'instance'
        
        if len(upstream_cols) > 1:  # Only add if we have type or instance info
            upstream = conn_df[upstream_cols].copy()
            upstream = upstream.rename(columns=rename_map)
            new_neurons = pd.concat([new_neurons, upstream])
    
    # Similar logic for downstream neurons...
```

## Benefits

1. **No more warnings**: Function gracefully handles missing columns
2. **Works with all fetch methods**: Compatible with different API response formats
3. **Works with all datasets**: Handles datasets with/without instance information
4. **Better cache robustness**: Only saves information that's actually available

## Testing

The fix handles these scenarios:

### Scenario 1: Full columns (hemibrain dataset)
```python
conn_df.columns = ['bodyId_pre', 'type_pre', 'instance_pre', 'bodyId_post', 'type_post', 'instance_post', 'weight', 'roi']
# ✅ All columns extracted successfully
```

### Scenario 2: No instance columns (optic-lobe dataset, fetch_simple_connections)
```python
conn_df.columns = ['bodyId_pre', 'type_pre', 'bodyId_post', 'type_post', 'weight', 'roi']
# ✅ Extracts bodyId + type only, skips instance (no error)
```

### Scenario 3: Minimal columns (cache storage)
```python
conn_df.columns = ['bodyId_pre', 'bodyId_post', 'weight', 'roi']
# ✅ Skips registry update (no type/instance info to save)
```

## Impact

- **Before**: Cache would fail with KeyError, warning printed, data not cached
- **After**: Cache succeeds, no warning, data stored correctly

## Related Files

- `coana.py`: Lines 287-340 (fixed `_update_neuron_registry()` function)
- Affects: `_save_connections_to_cache()` which calls this function

## Version

- **Fixed**: October 25, 2025
- **Commit**: Cache warning fix for missing instance columns
- **Status**: ✅ Complete and tested

---

**Note**: The neuron registry feature is optional - it's meant to track which neurons have been seen during caching. The main cache functionality (storing bodyId pairs) works regardless of whether type/instance columns are present.
