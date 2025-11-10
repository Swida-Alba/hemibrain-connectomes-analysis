# FindAllPath() Method - Updated Logic

## Overview
The `FindAllPath()` method has been completely rewritten to better handle complex neural networks with reciprocal connections and preserve the complete network structure.

## New Three-Phase Approach

### Phase 1: Fetch Complete Network Structure
**Objective**: Build the complete network up to `max_interlayer` layers without excluding any neurons.

**Key Changes**:
- **Removed** `sv.removeSearchedNeurons()` calls
- **Preserves** reciprocal connections within and across layers
- **Stores** all connections in their original form
- Each layer can contain neurons that connect back to previous layers

**Why**: This prevents disrupting the natural network structure. Reciprocal connections are common in neural circuits and should be preserved for accurate analysis.

### Phase 2: Identify Targets in Network
**Objective**: Determine which target neurons exist anywhere in the searched network.

**Process**:
1. Collect all unique neurons discovered across all layers
2. Check which target neurons are present in this set
3. Record the layer where each target first appears
4. Report target distribution by layer

**Output**: 
- `Checked` column: True if target found in network
- `Layer` column: First layer where target appears

### Phase 3: Extract Valid Paths
**Objective**: Find ALL paths from sources to targets with path length ≤ `max_interlayer`.

**Method**:
1. Build a NetworkX directed graph from all connections
2. Use `nx.all_simple_paths()` to find all simple paths (no repeated nodes)
3. Apply `cutoff=max_interlayer` to limit path length
4. Extract only neurons and edges that are part of at least one valid path

**Advantages**:
- Handles reciprocal connections correctly
- Finds truly all paths, not just layered paths
- Path length is measured by actual hops, not layer assignment
- Preserves network topology

## Comparison with Old Logic

### Old Approach
```
1. Layer-by-layer expansion
2. Exclude searched neurons at each step (removeSearchedNeurons)
3. Assume neurons fit into strict layers
4. Backtrack from targets to find paths
```

**Problems**:
- Reciprocal connections could create paths longer than expected
- Removing searched neurons disrupted network structure
- Assumed feed-forward topology

### New Approach
```
1. Fetch entire network structure (preserve all connections)
2. Identify targets present in network
3. Graph-based pathfinding with explicit length constraint
```

**Benefits**:
- Correctly handles reciprocal connections
- Preserves complete network topology
- Explicit path length control
- More accurate representation of neural circuits

## Example Scenario

### Network with Reciprocal Connections
```
Source (L0) → A (L1) → B (L2) → Target (L3)
              ↑         ↓
              └─────────┘
```

**Old logic**: 
- Might exclude B when searching L3 because it was "already searched" in L2
- Could create paths: Source → A → B → A → B → Target (length > max_interlayer)

**New logic**:
- Preserves all connections including B ↔ A
- Finds path: Source → A → B → Target (length = 3)
- Correctly identifies this as a valid path if max_interlayer ≥ 3
- Won't include Source → A → B → A → Target because it repeats node A

## Implementation Details

### Graph Construction
```python
G = nx.DiGraph()
for each connection:
    add_edge(pre, post, weight)
```

### Path Finding
```python
nx.all_simple_paths(
    G, 
    source=source, 
    target=target,
    cutoff=max_interlayer  # Maximum path length (edges)
)
```

### Path Filtering
- **Simple paths**: No repeated nodes (prevents infinite loops)
- **Length constraint**: Path length ≤ max_interlayer
- **Comprehensive**: Finds ALL valid paths, not just shortest

## Output Changes

### Console Output
Now includes three distinct phases with clear progress reporting:
```
=== PHASE 1: Fetching all network layers ===
Note: Preserving complete network structure including reciprocal connections

Layer 0->1: X downstream neurons, Y connections
Layer 1->2: X downstream neurons, Y connections
...

=== PHASE 2: Identifying targets in the network ===
Targets found in network: X / Y
Target distribution by layer:
  Layer 1: X targets
  Layer 2: Y targets
  ...

=== PHASE 3: Finding all paths from sources to targets ===
Using graph-based pathfinding to handle reciprocal connections...

Neurons in valid paths: X
Edges in valid paths: Y
```

### Data Structure
Same output files and formats as before:
- Excel file with all sheets
- Sankey diagrams
- Interactive network visualizations
- Path analysis (type and bodyId)

## Performance Considerations

**NetworkX pathfinding**:
- Efficient for moderate-sized networks (< 10,000 nodes)
- May be slower for very large networks
- Trade-off: Accuracy vs. Speed

**Memory usage**:
- Stores complete network before filtering
- May use more memory than old approach
- Benefit: More accurate results

## Use Cases

This new logic is particularly useful for:
1. **Recurrent networks**: Networks with feedback loops
2. **Multi-path circuits**: Finding all alternative routes
3. **Complex topologies**: Non-feed-forward architectures
4. **Reciprocal connections**: Bidirectional communication
5. **Network motifs**: Identifying circuit patterns

## Migration Notes

### For Existing Users
- **Same interface**: No changes to function signature
- **Same outputs**: All output files remain the same
- **Better accuracy**: More complete path discovery
- **Slower**: May take longer for large networks

### When to Use
- Use `FindPath()` for: Quick shortest-path analysis
- Use `FindAllPath()` for: Comprehensive path discovery in complex networks

## Technical Requirements

**Dependencies**:
- NetworkX 2.x or higher
- `nx.all_simple_paths()` function
- NumPy, Pandas (existing requirements)

**No changes needed** to calling code or output processing.
