# Forward-Only Mode Guide

Complete guide to the `forward_only` parameter in `FindAllPath()` - controls network querying, path validation, and visualization filtering.

---

## Table of Contents
1. [Overview](#overview)
2. [Three Functions of forward_only](#three-functions-of-forward_only)
3. [Real Layer vs Appearance Layer](#real-layer-vs-appearance-layer)
4. [Path Validation Rules](#path-validation-rules)
5. [Visualization Filtering](#visualization-filtering)
6. [Usage Examples](#usage-examples)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The `forward_only` parameter controls **three distinct aspects** of the `FindAllPath()` method:

1. **Network Querying Strategy** - How neurons are fetched from Neuprint
2. **Path Validation Logic** - Which paths are considered valid
3. **Visualization Filtering** - Which edges appear in Sankey diagrams and network graphs

**Default:** `forward_only=True` (recommended for most analyses)

```python
fc = FindNeuronConnection(
    sourceNeurons=['L3_R'],
    targetNeurons=['l-LNv_R'],
    max_interlayer=3,
    min_synapse_num=10
)

fc.InitializeNeuronInfo()
fc.FindAllPath(forward_only=True)  # Default behavior
```

---

## Three Functions of forward_only

### 1. Network Querying Strategy

Controls how neurons are queried from Neuprint during network discovery.

#### forward_only=True (Layer-by-Layer)
```python
# Layer 0: Query sources
# Layer 1: Query only layer 0 neurons → new neurons in layer 1
# Layer 2: Query only layer 1 neurons → new neurons in layer 2
# Layer 3: Query only layer 2 neurons → new neurons in layer 3
```

**Benefits:**
- ✅ Each neuron queried only once (faster)
- ✅ Less API overhead
- ✅ Still fetches ALL connections (including reciprocal/recurrent)

**Example:**
```
Querying: ['L3_R']
Layer 0→1: 892 downstream neurons, 892 new, 15,234 connections
Querying: [Mi15, Tm34, R8y, ...]  # 892 neurons from layer 0
Layer 1→2: 1,247 downstream neurons, 355 new, 23,456 connections
Querying: [MeVC20, ...]  # 355 NEW neurons from layer 1
Layer 2→3: 89 downstream neurons, 12 new, 1,234 connections
```

#### forward_only=False (Comprehensive)
```python
# Layer 0: Query sources
# Layer 1: Query ALL neurons (sources + layer 1) → new neurons in layer 1
# Layer 2: Query ALL neurons (sources + layer 1 + layer 2) → new neurons in layer 2
# Layer 3: Query ALL neurons (all discovered so far) → new neurons in layer 3
```

**Benefits:**
- ✅ More comprehensive (catches all possible connections)
- ❌ Slower (re-queries neurons multiple times)
- ❌ More API overhead

**When to use:** Rarely needed - only if you suspect layer-by-layer querying misses connections

---

### 2. Path Validation Logic

Controls which paths are considered valid based on "real layer" ordering.

#### forward_only=True (Strict)

**Rule:** Paths must have **non-decreasing real layers** (no backward connections)

**Valid:**
```
L3 → Mi15 → MeVC20 → l-LNv
 0  →   1  →    2   →   4    ✓ (all non-decreasing)

L3 → Tm34 → Tm37 → MeVC20 → l-LNv
 0  →   1  →   1  →    2   →   4    ✓ (lateral 1→1 allowed)
```

**Invalid:**
```
L3 → Mi15 → L3 → l-LNv
 0  →   1  →  0 →   4    ✗ (backward: 0 < 1)

L3 → Tm34 → MeVC20 → Tm34 → l-LNv
 0  →   1  →    2   →   1  →   4    ✗ (recurrent: 1 < 2)
```

**What's allowed:**
- ✅ Forward connections (real_layer increases)
- ✅ Lateral connections (real_layer stays same)
- ❌ Backward connections (real_layer decreases)
- ❌ Recurrent paths (revisiting same neuron)

#### forward_only=False (Permissive)

**Rule:** Paths must have **unique neurons only** (no validation by real layer)

**Valid:**
```
L3 → Mi15 → MeVC20 → l-LNv    ✓
L3 → Tm34 → Tm37 → MeVC20 → l-LNv    ✓
L3 → Mi15 → L3 → l-LNv    ✗ (repeated neuron - NetworkX prevents this)
L3 → MeVC20 → Mi15 → l-LNv    ✓ (any order allowed)
```

**What's allowed:**
- ✅ All simple paths (enforced by NetworkX)
- ✅ Any connection order (forward, backward, lateral)
- ❌ Only repeated neurons blocked (NetworkX `all_simple_paths`)

---

### 3. Visualization Filtering

Controls which edges appear in Sankey diagrams and network graphs.

#### forward_only=True (Filtered)

**Shows only:** Edges that appear in discovered valid paths

**Workflow:**
1. Extract edges from `path_type` and `path_bodyId` sheets
2. For type level: exclude type self-connections (Mi15→Mi15)
3. For bodyId level: keep all edges (including same-type connections)
4. Filter Sankey diagrams to show only these edges
5. Filter network graphs to show only these edges

**Result:** Clean, focused visualization matching the path sheets

#### forward_only=False (Complete Graph)

**Shows:** All connections in `conn_types` and `conn_inpath`

**Result:** Complete connectivity graph (may include unused connections)

**Type Self-Connections:**
- **Type level (Mi15→Mi15):** Hidden when `forward_only=True`, shown when `False`
  - Represents different individual neurons of same type
  - Usually clutters visualization without adding insight
- **BodyId level (1234→5678):** Always shown
  - Represents specific neuron-to-neuron connections
  - If in paths, should be visualized

---

## Real Layer vs Appearance Layer

### Real Layer (Discovery Order)

The layer where a neuron was **first discovered** during network search.

**Assignment:**
- Sources: layer 0
- Intermediate neurons: discovery layer (1, 2, 3, ...)
- **Targets: max_interlayer + 1** (special handling)

**Example:**
```
max_interlayer = 3

L3 (source) → real_layer = 0
Mi15 (discovered in layer 0→1) → real_layer = 1
MeVC20 (discovered in layer 1→2) → real_layer = 2
l-LNv (target) → real_layer = 4  (= max_interlayer + 1)
```

**Why targets get max_interlayer+1:**

Targets can be discovered early through direct connections:
```
L3 → l-LNv (direct)
Real layers: 0 → would be 1 if we used discovery order
```

But we also want multi-hop paths:
```
L3 → Mi15 → MeVC20 → l-LNv
 0  →   1  →    2   →   ?
```

If l-LNv has real_layer=1, the path 2→1 would be invalid (backward!).

**Solution:** All targets get real_layer = max_interlayer + 1
- Ensures ALL paths to targets are valid
- Allows both direct and multi-hop paths
- Targets can appear in any "appearance layer" in different paths

**Note:** After pathfinding completes, the system updates target real layers based on actual appearances:
```
Target l-LNv appearances:
  - Layer 1: Direct path (L3 → l-LNv)
  - Layer 2: 2-hop paths (L3 → X → l-LNv)
  - Layer 3: 3-hop paths (L3 → X → Y → l-LNv)
Final real_layer = max(3, max_interlayer+1) = 4
```

### Appearance Layer (Position in Path)

The position of a neuron **within a specific path**.

**Example:**
```
Path 1: L3 → Mi15 → MeVC20 → l-LNv
        Mi15 appears at position 1 (appearance layer 1)

Path 2: L3 → R8y → Mi15 → l-LNv
        Mi15 appears at position 2 (appearance layer 2)
```

**Key Difference:**
- Real layer is **fixed** for each neuron (discovery order)
- Appearance layer **varies** depending on the path

**In Output Files:**
- `path_type` / `path_bodyId`: Shows appearance layers (path position)
- Path validation: Uses real layers (discovery order)

---

## Path Validation Rules

### When forward_only=True

```python
def is_valid_path(path, real_layer_map):
    """Check if path has non-decreasing real layers"""
    for i in range(len(path) - 1):
        current_neuron = path[i]
        next_neuron = path[i + 1]
        
        current_layer = real_layer_map[current_neuron]
        next_layer = real_layer_map[next_neuron]
        
        if next_layer < current_layer:
            return False  # Backward connection
    
    return True  # All edges are forward or lateral
```

**Valid paths:**
```
✓ L3(0) → Mi15(1) → MeVC20(2) → l-LNv(4)      # Forward
✓ L3(0) → Tm34(1) → Tm37(1) → MeVC20(2)       # Lateral + forward
✓ L3(0) → l-LNv(4)                            # Direct to target
✓ L3(0) → Mi15(1) → l-LNv(4)                  # Skip layers
```

**Invalid paths:**
```
✗ L3(0) → Mi15(1) → L3(0)                     # Backward (0 < 1)
✗ L3(0) → Tm34(1) → MeVC20(2) → Tm34(1)       # Recurrent (1 < 2)
✗ L3(0) → MeVC20(2) → Mi15(1)                 # Backward (1 < 2)
```

### When forward_only=False

No real layer validation - all simple paths are valid (NetworkX handles uniqueness).

---

## Visualization Filtering

### Sankey Diagrams

#### forward_only=True
```python
# Extract edges from path_type
edges_in_paths = set()
for path in path_type['path_block']:
    # Parse: "L3 -> Mi15 -> MeVC20 -> l-LNv"
    steps = path.split(' -> ')
    for i in range(len(steps) - 1):
        source, target = steps[i], steps[i+1]
        if source != target:  # Exclude type self-connections
            edges_in_paths.add((i, source, target))

# Filter Sankey to show only these edges
for edge in conn_types:
    if edge not in edges_in_paths:
        continue  # Skip this edge
    # Add to Sankey diagram
```

#### forward_only=False
```python
# Show all edges in conn_types (no filtering)
for edge in conn_types:
    # Add to Sankey diagram
```

### Network Graphs

Same filtering logic as Sankey diagrams:
- `forward_only=True`: Only edges in paths
- `forward_only=False`: All edges

**Implementation:**
```python
def _create_interactive_network(self, conn_types, forward_only=True, 
                                edges_in_path_type=None):
    G_type = nx.DiGraph()
    
    for idx in conn_types.index:
        layer_label = conn_types.at[idx, 'conn_layer']
        layer_idx = int(layer_label.split('->')[0])
        source = conn_types.at[idx, 'type_pre']
        target = conn_types.at[idx, 'type_post']
        
        # Filter if forward_only=True
        if forward_only and (layer_idx, source, target) not in edges_in_path_type:
            continue
        
        G_type.add_edge(source, target, ...)
```

---

## Usage Examples

### Example 1: Standard Analysis (Recommended)

```python
fc = FindNeuronConnection(
    sourceNeurons=['L3_R'],
    targetNeurons=['l-LNv_R'],
    max_interlayer=3,
    min_synapse_num=10
)

fc.InitializeNeuronInfo()
fc.FindAllPath(forward_only=True)  # Default
```

**Result:**
- Layer-by-layer querying (fast)
- Paths validated by real layer (no backward/recurrent)
- Visualizations show only path edges (clean)

### Example 2: Complete Graph Visualization

```python
fc.FindAllPath(forward_only=False)
```

**Result:**
- Comprehensive querying (slower)
- All simple paths valid (permissive)
- Visualizations show all connections (busier)

**Use case:** Want to see the complete connectivity graph, not just paths

### Example 3: Comparing Modes

```python
# Run both modes
fc.FindAllPath(forward_only=True)   # Saves to paths_XXXXXX_True/
fc.FindAllPath(forward_only=False)  # Saves to paths_XXXXXX_False/

# Compare outputs:
# - forward_only=True: Fewer, validated paths; clean visualizations
# - forward_only=False: More paths; complete graph visualizations
```

---

## Troubleshooting

### No Paths Found (forward_only=True)

**Problem:** `FindAllPath(forward_only=True)` finds no paths, but `forward_only=False` does

**Cause:** All paths have backward/recurrent connections

**Solutions:**
```python
# 1. Increase max_interlayer (allow more layers)
fc = FindNeuronConnection(max_interlayer=5)  # Instead of 3

# 2. Lower filter thresholds (find weaker paths)
fc = FindNeuronConnection(
    min_synapse_num=5,         # Lower from 10
    min_traversal_probability=0.01  # Lower from 0.1
)

# 3. Check if targets are in the network
fc.FindAllPath(forward_only=True)
# Look for: "Targets found in network: X / Y"
```

### Too Many Paths

**Problem:** Thousands of paths found, analysis is slow

**Solutions:**
```python
# 1. Increase filter thresholds
fc = FindNeuronConnection(
    min_synapse_num=20,         # Higher from 10
    min_traversal_probability=0.5  # Higher from 0.1
)

# 2. Decrease max_interlayer
fc = FindNeuronConnection(max_interlayer=2)  # Instead of 3

# 3. Use forward_only=True (stricter validation)
fc.FindAllPath(forward_only=True)
```

### Visualization Shows No Edges (forward_only=True)

**Problem:** Network graphs and Sankey diagrams are empty

**Cause:** No valid paths found, so no edges to visualize

**Solution:** Same as "No Paths Found" above

### Type Self-Connections in Visualization

**Problem:** Seeing Mi15→Mi15 edges in type-level Sankey

**Cause:** `forward_only=False` (shows all connections)

**Solution:**
```python
# Use forward_only=True to hide type self-connections
fc.FindAllPath(forward_only=True)
```

---

## Summary

The `forward_only` parameter provides **three levels of control**:

| Aspect | forward_only=True | forward_only=False |
|--------|-------------------|-------------------|
| **Querying** | Layer-by-layer (fast) | Comprehensive (slow) |
| **Validation** | Real layer ordering | All simple paths |
| **Visualization** | Filtered (paths only) | Complete graph |
| **Use Case** | Standard analysis | Complete connectivity |

**Recommendation:** Use `forward_only=True` (default) for most analyses.

---

**See Also:**
- PathFinding_Guide.md for general pathfinding
- Visualization_Guide.md for Sankey and network graphs
- Configuration_Guide.md for filter parameters
