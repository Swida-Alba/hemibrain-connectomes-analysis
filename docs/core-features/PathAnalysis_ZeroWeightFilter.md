# Path Analysis: Zero-Weight Filter

## Problem

When analyzing paths at the **type level**, some paths appeared with **weight=0** in certain hops. This occurred even though individual bodyId-level connections existed.

### Example
```
L3 -> Mi15 -> l-LNv    weights: [0.32494868, 0]
```
- The first hop (L3 → Mi15) has weight 0.324...
- The second hop (Mi15 → l-LNv) has weight **0**

This is confusing because paths shouldn't include connections with zero weight.

## Root Cause

The issue arises from how **type-level aggregation** works in graph-based pathfinding:

### Step-by-Step Breakdown

1. **Graph Construction** (`FindAllPath` and `FindPath`)
   - NetworkX graph `G` is built from ALL bodyId-level connections
   - Graph contains edges like: `Mi15[bodyId=123] -> l-LNv[bodyId=456]` with weight=2

2. **Type-Level Aggregation** (`EnrichConnectionTable`)
   - Individual bodyId connections are aggregated to type level
   - Type-level weight = sum of all bodyId weights for that type pair
   - Example: If only one Mi15→l-LNv connection exists with weight=2, the type-level weight is 2

3. **Filter Application**
   - If `min_synapse_num=5`, the type pair `Mi15 -> l-LNv` doesn't meet the threshold
   - In `conn_types` DataFrame, this edge may have weight=0 or be filtered out

4. **Path Extraction** (`getAllPath`)
   - Paths are found through the graph (which contains all bodyId edges)
   - When extracting path metrics, the function looks up type-level weights from `conn_types`
   - If an edge exists in the graph but not in `conn_types` (or has weight=0), the fallback is:
     ```python
     weight_edge = 0
     travP_edge = 0
     ratio_edge = 0
     ```

### Why This Happens

The graph structure is built from **bodyId-level connections** (which may pass individual filters), but path **metrics** are extracted from **type-level aggregations** (which may fail aggregate filters). This mismatch creates paths with zero-weight hops.

## Solution

### Implementation

Both `FindPath()` and `FindAllPath()` now **filter out paths with any zero-weight hops** immediately after calling `getAllPath()`:

```python
# Get all paths (by type)
path_df_type,_ = sv.getAllPath(conn_data = conn_types,
                            targets = target_types,
                            traversal_probability_threshold = min_traversal_probability,
                            max_path_length = max_interlayer + 1)

# Filter out paths with any zero-weight hops
if len(path_df_type) > 0:
    before_filter = len(path_df_type)
    path_df_type = path_df_type[
        path_df_type['weights'].apply(lambda w_list: all(w > 0 for w in w_list))
    ]
    after_filter = len(path_df_type)
    if before_filter > after_filter:
        print(f'  Removed {before_filter - after_filter} paths with zero-weight hops at type level')
```

### What Gets Filtered

- **Paths with ANY hop having weight=0** at the type level
- This includes paths where:
  - Individual bodyId connections exist
  - But the aggregated type-level weight doesn't meet filter thresholds
  - Or the type pair was filtered out entirely

### Impact

- **Type-level analysis**: Only paths with ALL non-zero type-level weights are included
- **BodyId-level analysis**: No filtering needed (bodyId paths already have proper weights from individual connections)

## Expected Behavior

### Before Fix
```
path_type sheet:
L3 -> Mi15 -> l-LNv    weights: [0.32, 0]       # Invalid - has zero weight
L3 -> R8y -> l-LNv     weights: [0.51, 0]       # Invalid - has zero weight
L3 -> Tm34 -> MeVC20 -> l-LNv  weights: [0.56, 0.85, 0]  # Invalid - has zero weight
```

### After Fix
```
path_type sheet:
L3 -> Tm34 -> MeVC20 -> l-LNv  weights: [0.56, 0.85, 0.12]  # Valid - all weights > 0
L3 -> Tm37 -> MeVC20 -> l-LNv  weights: [0.11, 0.85, 0.12]  # Valid - all weights > 0
```

Only paths where **every hop** has non-zero weight at the type level are included.

## Technical Notes

### Why Not Fix the Graph Construction?

We could rebuild the graph using only type-level filtered edges, but this would:
1. Require significant refactoring of the pathfinding logic
2. Potentially miss valid paths where intermediate neurons connect through multiple weak edges
3. Complicate the relationship between bodyId and type-level analysis

### Why Post-Filter Instead?

Post-filtering is:
- **Simple**: One line of code after `getAllPath()`
- **Clear**: Explicitly removes invalid paths with zero-weight hops
- **Consistent**: Works for both `FindPath` and `FindAllPath`
- **Safe**: Doesn't affect the underlying pathfinding algorithm

## Related Files

- `coana.py`: Lines ~1202 and ~2433 (filter implementation)
- `statvis.py`: Lines ~472-622 (`getAllPath` function)
- `statvis.py`: Lines ~632-752 (`EnrichConnectionTable` function)

## Related Issues

- Type-level filtering: See `CacheSystem_v4_Complete.md` for filter behavior
- Path extraction: See `FindAllPath_Documentation.md` for pathfinding details
