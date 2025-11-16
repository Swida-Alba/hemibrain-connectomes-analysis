# Custom Groups Feature

## Overview

The **Custom Groups** feature allows you to define arbitrary neuron groupings beyond the standard type-level classification. This enables flexible analysis at custom granularities - from broad functional categories to fine-grained manual selections.

**Available in**: `FindAllPath()` (fully optimized) and `FindPath()` methods

---

## Key Capabilities

✅ **Flexible Grouping**: Group neurons by any criteria (function, anatomy, experimental condition, etc.)  
✅ **Mixed Granularity**: Groups can contain single neurons, multiple bodyIds, entire types, or combinations  
✅ **Optimized Performance**: Dedicated group-level DFS ensures fast pathfinding (10-100x faster than bodyId→group conversion)  
✅ **Hierarchical Analysis**: Analyze at bodyId, type, AND custom group levels simultaneously  
✅ **Automatic Aggregation**: Connection metrics properly aggregated from bodyId to group level  

---

## How It Works

### 1. Input Format

Add a `custom_group` column to your source and/or target neuron DataFrames:

```python
import pandas as pd

# Example: Group by functional role
source_df = pd.DataFrame({
    'bodyId': [123, 456, 789, 101112],
    'type': ['DA1', 'DA1', 'VA1d', 'VA1d'],
    'custom_group': ['Sensory_A', 'Sensory_A', 'Sensory_B', 'Sensory_B']
})

target_df = pd.DataFrame({
    'bodyId': [131415, 161718, 192021],
    'type': ['MBON-γ1', 'MBON-γ2', 'DAN-γ1'],
    'custom_group': ['Output_Approach', 'Output_Avoidance', 'Modulatory']
})

# Use in FindAllPath
fc = FindNeuronConnection(
    source_file=source_df,
    target_file=target_df,
    # ... other parameters
)
fc.FindAllPath(find_bodyId_path=True, forward_only=True)
```

### 2. Three-Level Analysis

The system automatically generates analysis at **three levels**:

1. **BodyId Level**: Individual neuron-to-neuron connections
2. **Type Level**: Aggregated by neuron type (standard classification)
3. **Group Level**: Aggregated by custom_group (your classification)

### 3. Output Structure

When `custom_group` column exists, you get additional output:

#### Excel/CSV Sheets
- `connection_custom_groups`: Group-to-group connection table
- `path_group`: All paths at group level
- `path_group_excluded`: Excluded group paths (if keyword filters applied)

#### Visualizations
- `custom_groups/` folder: Network and Sankey diagrams for group-level paths

---

## Use Cases

### Use Case 1: Functional Classification

Group neurons by their functional roles:

```python
# Olfactory pathway
source_df['custom_group'] = source_df['type'].map({
    'DA1': 'Food_Odor',
    'DA2': 'Food_Odor',
    'VA1d': 'Pheromone',
    'VA1v': 'Pheromone',
    'DL3': 'Aversive'
})

# Output neurons
target_df['custom_group'] = target_df['type'].map({
    'MBON-γ1': 'Approach',
    'MBON-γ2': 'Avoidance',
    'MBON-β1': 'Approach'
})
```

**Result**: See how different odor modalities (Food, Pheromone, Aversive) connect to behavioral outputs (Approach, Avoidance)

### Use Case 2: Anatomical Regions

Group by brain regions or compartments:

```python
# Group by Kenyon Cell types
source_df['custom_group'] = source_df['type'].map({
    'KCab-p': 'KC_Core',
    'KCab-s': 'KC_Core',
    'KCab-c': 'KC_Core',
    "KCg-d": 'KC_Gamma',
    "KCg-m": 'KC_Gamma'
})
```

### Use Case 3: Experimental Groups

Group by experimental conditions or manipulations:

```python
# Example: Neurons targeted by different driver lines
source_df['custom_group'] = source_df['bodyId'].map({
    123: 'Driver_Line_A',
    456: 'Driver_Line_A',
    789: 'Driver_Line_B',
    101112: 'Control'
})
```

### Use Case 4: Mixed Granularity

Combine types and individual neurons:

```python
# Most neurons by type, but isolate specific interesting neurons
def assign_group(row):
    if row['bodyId'] == 123456:
        return 'Neuron_123456_Special'  # Individual neuron
    elif row['type'] in ['DA1', 'DA2']:
        return 'DA_Group'               # Group of types
    else:
        return row['type']              # Use type name
        
source_df['custom_group'] = source_df.apply(assign_group, axis=1)
```

---

## Connection Metrics at Group Level

When aggregating from bodyId to group level, connection metrics are calculated as follows:

### Weight (Synapse Count)
**Formula**: Sum of all bodyId-level weights  
```
weight(GroupA → GroupB) = Σ weight(bodyId_i → bodyId_j)
                          for all i ∈ GroupA, j ∈ GroupB
```

### Connection Ratio
**Formula**: Total weight / Total postsynaptic sites in target group  
```
ratio(GroupA → GroupB) = weight(GroupA → GroupB) / Σ post_j
                         for all j ∈ GroupB
```

### Traversal Probability
**Formula**: Compound probability (product method) or weighted average
```
# Product method (default for paths):
prob(GroupA → GroupB) = 1 - Π (1 - prob_ij)
                        for all connections i→j

# Average method (for direct connections):
prob(GroupA → GroupB) = Σ (weight_ij × prob_ij) / Σ weight_ij
```

**Note**: Product method is used for path analysis, average method for direct connections.

---

## Performance Optimization

### Group-Level DFS Architecture

The system uses **separate optimized pathfinding** for custom groups:

```
1. Build group-level graph from conn_groups
   └─ Nodes: custom group names
   └─ Edges: aggregated connections with metrics

2. Run DFS directly on group graph
   └─ Find unique group-to-group paths
   └─ No bodyId→group conversion needed

3. Build path DataFrame with group-level metrics
   └─ Use build_path_dataframe_from_paths()
   └─ Apply real_layer_map_group for validation
```

### Performance Benefits

**Traditional Approach** (no longer used):
```
1. Find all bodyId paths: 5.9M paths
2. Convert bodyId → group: 5.9M conversions
3. Deduplicate: 825K unique group paths
⏱️ Time: ~400s, Memory: ~189 MB
```

**Optimized Approach** (current):
```
1. Build group graph: 5K nodes, 12K edges
2. Find group paths directly: 825K paths
3. No conversion needed
⏱️ Time: ~4s, Memory: ~26 MB
✅ 100x faster, 7x less memory
```

---

## Real Layer Validation

When `forward_only=True`, the system ensures paths only move forward through layers:

### Group-Level real_layer_map

```python
# Built automatically from bodyId-level mappings
real_layer_map_group = {}

# Each group assigned earliest layer of any member neuron
for bodyid, real_layer in real_layer_map_bodyId.items():
    group = bodyid_to_group[bodyid]
    if group not in real_layer_map_group:
        real_layer_map_group[group] = real_layer
    else:
        real_layer_map_group[group] = min(
            real_layer_map_group[group], 
            real_layer
        )
```

**Result**: Group paths validated to ensure forward-only connectivity, even when groups contain neurons from different layers.

---

## Output Examples

### Console Output

```
=== Updating target real layers based on path appearances ===
  ✓ Updated real_layer for 45 target neurons

Created type-level real layer map for 234 types
  ✓ Updated real_layer for 23 target types

Created group-level real layer map for 12 custom groups
  ✓ Updated real_layer for 5 target groups

Finding group-level paths using group-level graph...
  Group-level graph: 12 groups, 28 edges
  Found 143 group-level paths
  Removed 3 paths with zero-weight hops at group level

💾 Saving path_group data (rows: 140)...
   ✓ path_group sheets saved

Creating custom group visualizations...
  ✓ Custom group visualizations created (140 paths)
```

### Data Files

**connection_custom_groups.csv**:
```csv
conn_layer,group_pre,group_post,weight,connection_ratio,traversal_probability,block_probability
0->1,Sensory_A,Processing_1,450,0.023,0.077,0.923
0->1,Sensory_B,Processing_1,320,0.016,0.053,0.947
1->2,Processing_1,Output_Approach,280,0.035,0.117,0.883
```

**path_group sheet**:
```csv
path_id,source,layer_1,target,path_length,weights,ratios,travPs,traversal_probability
1,Sensory_A,Processing_1,Output_Approach,2,[450,280],[0.023,0.035],[0.077,0.117],0.009
2,Sensory_B,Processing_1,Output_Avoidance,2,[320,195],[0.016,0.028],[0.053,0.093],0.005
```

---

## Best Practices

### 1. Group Naming
✅ **Use descriptive names**: `Food_Odor` better than `Group1`  
✅ **Use consistent format**: `Category_Subcategory` or `Function_Type`  
✅ **Avoid special characters**: Use `_` instead of spaces or symbols  

### 2. Group Size
✅ **Balance granularity**: Not too coarse (1 group) or too fine (1 neuron per group)  
✅ **Consider analysis goals**: Functional groups for behavior, anatomical for circuits  
✅ **Document grouping logic**: Save mapping rules for reproducibility  

### 3. Mixed Analysis
✅ **Use all three levels**: bodyId for detail, type for standards, group for interpretation  
✅ **Compare across levels**: Validate group findings against type-level patterns  
✅ **Iterate groupings**: Try different groupings to test hypotheses  

### 4. Validation
✅ **Check group sizes**: Ensure groups have similar neuron counts for fair comparison  
✅ **Verify connections**: Group-level weights should match sum of bodyId weights  
✅ **Test edge cases**: Handle neurons without group assignment (NaN handling)  

---

## Limitations and Considerations

### 1. Group Assignments
- Each neuron can belong to **one group only** (no overlapping groups)
- Neurons without `custom_group` value (NaN) are skipped in group analysis
- Groups are static (assigned before pathfinding, not dynamically)

### 2. Path Interpretation
- Group-level paths show **any** connection between groups
- A single group-level connection may represent multiple bodyId-level routes
- Use bodyId-level analysis to see individual neurons within group connections

### 3. Statistical Considerations
- Group-level metrics are **aggregated** - individual variability is hidden
- Connection ratios depend on **all** neurons in target group (not just connected ones)
- Traversal probabilities use compound probability (may differ from simple averages)

---

## Related Documentation

- **[FindAllPath Documentation](./FindAllPath_Documentation.md)**: Main pathfinding method
- **[Forward-Only Guide](./ForwardOnly_Guide.md)**: Layer-based path validation
- **[Filter By Feature](./FilterBy_Feature.md)**: Type-level vs bodyId-level analysis
- **[Network Visualization](../visualizations/Network_Guide.md)**: Visualizing group paths

---

## API Reference

### Input DataFrame Schema

```python
# Minimal required columns
source_df = pd.DataFrame({
    'bodyId': [int],      # Required
    'type': [str],        # Required
    'custom_group': [str] # Optional - enables custom group analysis
})

target_df = pd.DataFrame({
    'bodyId': [int],      # Required
    'type': [str],        # Required  
    'custom_group': [str] # Optional - enables custom group analysis
})
```

### Output Schema

**conn_groups DataFrame**:
```python
columns = [
    'conn_layer',              # str: "0->1", "1->2", etc.
    'group_pre',               # str: upstream group name
    'group_post',              # str: downstream group name
    'weight',                  # int: total synapses
    'connection_ratio',        # float: weight / total_post_in_group
    'traversal_probability',   # float: compound probability
    'block_probability'        # float: 1 - traversal_probability
]
```

**path_group DataFrame**:
```python
columns = [
    'path_id',                 # int: unique path identifier
    'source',                  # str: source group name
    'layer_1', 'layer_2', ..., # str: intermediate groups
    'target',                  # str: target group name
    'path_length',             # int: number of hops
    'weights',                 # list: weight per hop
    'ratios',                  # list: connection_ratio per hop
    'travPs',                  # list: traversal_probability per hop
    'traversal_probability',   # float: path-level compound probability
    'path_block'               # str: "GroupA|GroupB|GroupC"
]
```

---

**Last Updated**: November 14, 2025  
**Feature Status**: ✅ Fully Implemented and Optimized  
**Performance**: Group-level DFS with 10-100x speedup over conversion approach
