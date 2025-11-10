# Bug Fix: DataFrame Column Merge Conflict

## Issue
When running `FindAllPath()`, the following error occurred:
```
AttributeError: 'DataFrame' object has no attribute 'type_pre'. Did you mean: 'type_pre_x'?
```

## Root Cause

The error was caused by pandas merge behavior when both DataFrames contain the same column names:

1. **Cache enrichment** (`_enrich_connections_with_neuron_info`, line ~560):
   - Merges neuron info to add `type_pre`, `type_post`, `instance_pre`, `instance_post` columns
   
2. **Later enrichment** (`EnrichConnectionTable`, statvis.py line ~580):
   - Tried to access `type_pre` column, but it had become `type_pre_x` and `type_pre_y`

**Why?** When merging DataFrames that both have columns with the same name, pandas automatically renames them with `_x` and `_y` suffixes to avoid conflicts.

## Example of the Problem

```python
# DataFrame already has type_pre column
conn_df:
  bodyId_pre  bodyId_post  weight  type_pre
  123         456          10      "L3"

# Merge with another DataFrame that also has type_pre
neuron_info:
  bodyId  type_pre
  123     "L3"

# Result after merge: type_pre becomes type_pre_x and type_pre_y
merged:
  bodyId_pre  bodyId_post  weight  type_pre_x  type_pre_y
  123         456          10      "L3"        "L3"

# Accessing type_pre fails!
merged.type_pre  # AttributeError!
```

## Solution

### Fix 1: `_enrich_connections_with_neuron_info` (coana.py, lines 558-567)
Drop existing type/instance columns before merging to prevent `_x`/`_y` suffix creation:

```python
# Drop existing type/instance columns if they exist (to avoid _x, _y suffixes after merge)
columns_to_drop = []
for col in ['type_pre', 'instance_pre', 'type_post', 'instance_post']:
    if col in conn_df.columns:
        columns_to_drop.append(col)
if columns_to_drop:
    conn_df = conn_df.drop(columns=columns_to_drop)

# Now merge safely - no conflicts
conn_df = conn_df.merge(
    neuron_info.rename(columns={'type': 'type_pre', 'instance': 'instance_pre'}),
    ...
)
```

### Fix 2: `EnrichConnectionTable` (statvis.py, lines 578-585)
Check if columns exist before accessing them:

```python
# Handle case where type_pre/type_post columns already exist (from cache enrichment)
if 'type_pre' in conn_df.columns:
    conn_df.loc[conn_df.type_pre.isnull(),'type_pre'] = 'None'
if 'type_post' in conn_df.columns:
    conn_df.loc[conn_df.type_post.isnull(),'type_post'] = 'None'
```

## Why This Happened

The bug appeared when using Cache v4.0 because:

1. Cache v4.0 enriches connections with type/instance info when fetching
2. The enriched DataFrame gets passed to `FindAllPath()`
3. `FindAllPath()` calls `EnrichConnectionTable()` which assumes columns don't exist yet
4. Result: merge conflict and column renaming

## Prevention

**Best Practice**: Before merging DataFrames that might have overlapping columns:
1. Check if target columns already exist
2. Drop them if necessary (like in Fix 1)
3. Or check before accessing (like in Fix 2)

## Test Case

To verify the fix works:
```bash
cd /Users/apple/Documents/GitHub/hemibrain-connectomes-analysis-now
conda activate hemibrain
python FindPath_Kun.py
```

Should now complete without the `AttributeError`.

## Files Modified

1. **coana.py** (lines 558-567): Added column drop before merge
2. **statvis.py** (lines 578-585): Added column existence check

---

**Date**: October 25, 2025  
**Bug**: AttributeError with type_pre column  
**Fix**: Prevent pandas merge suffix conflicts by dropping existing columns before merge
