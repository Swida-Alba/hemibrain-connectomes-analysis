# FindAllPath() Method Documentation

## Overview
The `FindAllPath()` method finds **all paths** from source neurons to target neurons within the specified `max_interlayer`, keeping paths of different lengths while filtering out connections that don't lead to any target neurons.

## Key Differences from FindPath()

### FindPath()
- Stops searching when **all** target neurons are found
- Returns only the **shortest paths** to each target neuron
- May miss longer alternative paths

### FindAllPath()
- Continues searching through **all layers** up to `max_interlayer`
- Returns **all paths** to target neurons, regardless of length
- Includes targets found at different layers (Layer 1, Layer 2, etc.)
- Filters out "dead-end" branches that don't connect to any target

## Algorithm

### Phase 1: Forward Search (Exploration)
1. Starting from source neurons, explore all downstream connections
2. Continue for all layers up to `max_interlayer`
3. Track which target neurons are found at each layer
4. Store all discovered connections in `conn_layers`

### Phase 2: Backward Filtering (Path Extraction)
1. Identify all target neurons found at any layer
2. Backtrack through layers to identify neurons that are part of paths to targets
3. Keep only connections where **both** pre- and post-synaptic neurons are in paths to targets
4. This preserves:
   - **All path lengths**: Targets found at Layer 1, Layer 2, Layer 3, etc.
   - **All alternative paths**: Multiple routes to the same target
   - **Convergent and divergent patterns**: One-to-many and many-to-one connections

### Why This Approach Works
- **Forward search**: Ensures we explore all possible connections within the layer limit
- **Backward filtering**: Removes branches that don't lead anywhere (dead ends)
- **Result**: All paths from source to target, with no dead-end branches

## Output Structure

### Excel File Sheets
1. **parameters**: Analysis parameters
2. **source_neurons**: Information about source neurons
3. **target_neurons**: Information about target neurons (with Layer column showing where found)
4. **total_weight_layer**: Total synaptic weight per layer
5. **connection_info**: All connections in paths (by bodyId)
6. **connection_type**: All connections in paths (by type)
7. **path_type**: All paths to targets (by neuron type)
8. **path_type_excluded**: Paths excluded by keyword filter
9. **path_bodyId**: All paths to targets (by bodyId) - if find_bodyId_path=True
10. **layer_X**: Neuron information for each intermediate layer

### Sankey Diagrams
- **Sankey_type_allpaths_snp{N}.html**: Shows all paths by neuron type
- **Sankey_bodyId_allpaths_snp{N}.html**: Shows all paths by bodyId

Both diagrams visualize the complete network including:
- All layers from source to furthest target
- All connections between layers
- Color-coded target neurons

## Console Output

```
Completed exploring all {N} layers (max_interlayer={M})
Target neurons found: {X} / {Y}

Filtering connections to keep only paths from source to target...
Neurons in paths to targets: {Z}

Path Network Statistics (source to target):
Total connections in paths: {A}
Total connection types in paths: {B}
Total neurons in paths: {C}
  Layer 0: {N0} neurons
  Layer 1: {N1} neurons
  Layer 2: {N2} neurons
  ...

Target neurons by layer:
  Layer 1: {T1} targets
  Layer 2: {T2} targets
  Layer 3: {T3} targets
  ...
```

## Use Cases

### When to Use FindAllPath()
1. **Comprehensive path analysis**: You want to see all possible routes, not just shortest
2. **Multi-layered targeting**: Targets are expected at different distances from source
3. **Alternative pathway discovery**: Finding backup routes or parallel pathways
4. **Network motif analysis**: Identifying patterns like feedforward loops, convergence, divergence

### When to Use FindPath() Instead
1. **Efficiency focus**: You only care about the shortest/most direct connections
2. **Large networks**: When computational resources are limited
3. **Single-layer targets**: When targets are all expected at the same distance

## Example Scenario

Suppose you have:
- Source: PPL1 dopaminergic neurons
- Target: MBON output neurons
- max_interlayer: 3

**FindPath()** would stop as soon as it finds all MBON neurons, even if some are at Layer 1. It would miss longer paths.

**FindAllPath()** would:
1. Search through all 3 layers
2. Find MBONs at Layer 1, Layer 2, and Layer 3
3. Keep all paths to each MBON
4. Remove neurons that connect to nothing downstream

Result: Complete picture of all PPL1→MBON pathways within 3 layers.

## Parameters

### Required
- Source neurons and target neurons must be initialized
- At least one target neuron must be found in the search

### Optional
- `find_bodyId_path=True`: Also generate bodyId-level paths (can be very large)
- `max_interlayer`: Maximum layers to search (default: 2)
- `min_synapse_num`: Minimum synapses for connection (default: 10)
- `min_traversal_probability`: Minimum probability threshold (default: 0.001)

## Performance Considerations

- Computation time increases with:
  - Number of layers (max_interlayer)
  - Network size (neurons and connections)
  - Number of paths to analyze
- Memory usage scales with the number of paths found
- Excel file size can be large for complex networks with many paths
