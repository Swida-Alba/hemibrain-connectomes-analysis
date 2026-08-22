# Filter Modes Fix Summary

## Issue
Both `filter_by='bodyId'` and `filter_by='type'` were not working correctly with the specified criteria:
- `min_synapse_num=1`
- `min_ratio=0`
- `min_traversal_probability=0`

The `filter_by='type'` mode was incorrectly removing connections with null types, even when all filters were set to their minimum values.

## Root Cause
In the `_apply_type_level_filters` function, connections where either `type_pre` or `type_post` was null (NaN) were being lost during the groupby operation and subsequent filtering logic. These neurons exist in the dataset but don't have type information.

Example counts of null-type connections:
- Layer 0→1: 212 connections with null type_post
- Layer 1→2: 46,284 connections with null type_post
- Layer 2→3: 1,491,193 connections with null types

## Solution
Modified `_apply_type_level_filters` in `coana.py` to:

1. **Separate null-type connections** at the start:
   ```python
   null_type_mask = combined['type_pre'].isna() | combined['type_post'].isna()
   connections_with_null_types = combined[null_type_mask].copy()
   connections_with_types = combined[~null_type_mask].copy()
   ```

2. **Apply filters only to connections with valid types**:
   - Group and aggregate only connections with both type_pre and type_post
   - Apply weight/ratio/prob filters to these type pairs
   - Filter bodyId-level connections based on passing type pairs

3. **Preserve all null-type connections**:
   ```python
   if len(connections_with_null_types) > 0:
       combined = pd.concat([filtered_connections, connections_with_null_types], ignore_index=True)
   ```

4. **Add informative logging**:
   - Shows how many connections with null types were preserved
   - Makes it clear these connections bypass type-level filtering

## Verification

### Before Fix
- **filter_by='bodyId'**: 662 connections → 24,815 paths ✓
- **filter_by='type'**: 450 connections → fewer paths ✗ (removed 212 null-type connections)

### After Fix
- **filter_by='bodyId'**: 662 connections → 24,815 paths ✓
- **filter_by='type'**: 662 connections → 24,815 paths ✓

Both modes now:
1. Respect the same filtering criteria
2. Preserve connections with null types
3. Find identical paths
4. Produce the expected results

## Files Modified
- `/Users/apple/Documents/GitHub/drocat/coana.py`
  - `_apply_type_level_filters()` method (lines 882-970)
  
- `/Users/apple/Documents/GitHub/drocat/FindPath.py`
  - Updated to use min_synapse_num=1, min_ratio=0

## Test Results
```
Layer 0→1: 662 connections (478 neurons)
  - bodyId mode: all 662 kept
  - type mode: all 662 kept (212 null-type preserved)

Layer 1→2: 139,287 connections (29,926 neurons)  
  - bodyId mode: all 139,287 kept
  - type mode: all 139,287 kept (46,284 null-type preserved)

Final paths found: 24,815 (both modes)
```

## Key Takeaway
Type-level filtering now correctly handles neurons without type information by:
- Preserving them when filters are at minimum values
- Applying type-level aggregation only to connections between typed neurons
- Maintaining consistency with bodyId-level filtering behavior
