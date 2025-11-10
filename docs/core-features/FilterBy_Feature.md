# filter_by Parameter: BodyId vs Type Level Filtering

## Overview

The `filter_by` parameter controls the level at which connection filters (`min_synapse_num`, `min_ratio`, `min_traversal_probability`) are applied:

- **`filter_by='bodyId'`** (default): Apply filters to individual neuron connections
- **`filter_by='type'`**: Apply filters to aggregated type-to-type connections

This allows you to analyze connectivity at different granularities depending on your research question.

## When to Use Each Mode

### Use `filter_by='bodyId'` when:
- You want to analyze individual neuron-to-neuron connections
- You care about variability within neuron types
- You want to identify specific strong connections between individual neurons
- **Default behavior** - most conservative filtering

### Use `filter_by='type'` when:
- You want to analyze overall connectivity between neuron types
- Individual connection strength is less important than type-level patterns
- You want to identify type pairs with strong aggregate connectivity
- You're interested in circuit-level organization

## How It Works

### BodyId-Level Filtering (default)

```python
fc = FindNeuronConnection(
    sourceNeurons=['L3'],
    targetNeurons=['Tm3'],
    min_synapse_num=10,
    min_ratio=0.01,
    filter_by='bodyId'  # Default
)
```

**Process:**
1. Fetch all connections between L3 and Tm3 neurons
2. For each connection (bodyId_pre → bodyId_post):
   - Check: weight ≥ 10 synapses?
   - Check: weight/post ≥ 0.01 (1% of inputs)?
3. Keep only connections that pass BOTH filters

**Example:**
```
L3[123] → Tm3[456]: 15 synapses, post=1000 → ratio=0.015 (1.5%)  ✓ PASS
L3[234] → Tm3[456]: 8 synapses, post=1000 → ratio=0.008 (0.8%)   ✗ FAIL (weight too low)
L3[345] → Tm3[567]: 12 synapses, post=5000 → ratio=0.002 (0.2%)  ✗ FAIL (ratio too low)
```

### Type-Level Filtering

```python
fc = FindNeuronConnection(
    sourceNeurons=['L3'],
    targetNeurons=['Tm3'],
    min_synapse_num=10,
    min_ratio=0.01,
    filter_by='type'
)
```

**Process:**
1. Fetch all connections between L3 and Tm3 neurons
2. Group by type pairs and aggregate:
   - Sum all weights from L3 → Tm3
   - Sum all post-synaptic sites of all Tm3 neurons
3. Calculate type-level ratio: total_weight / total_post
4. If type pair passes filters, keep ALL individual connections between those types

**Example:**
```
Individual connections:
  L3[123] → Tm3[456]: 15 synapses (Tm3[456] has 1000 post)
  L3[234] → Tm3[456]: 8 synapses
  L3[345] → Tm3[567]: 12 synapses (Tm3[567] has 5000 post)
  ... (many more)

Aggregated:
  L3 → Tm3: 350 total synapses, 50000 total post → ratio=0.007 (0.7%)
  
Result: ✗ FAIL at type level
  → ALL L3→Tm3 connections are filtered out (even the strong ones)
```

## Comparison Example

### Scenario
You're analyzing connections from L3 (892 neurons) to Tm3 (hundreds of neurons):
- Some individual L3→Tm3 connections are very strong (>50 synapses, >5% ratio)
- Most individual connections are weak (<10 synapses, <1% ratio)
- Aggregate L3→Tm3 connectivity is moderate

### BodyId-Level Results
```
filter_by='bodyId', min_synapse_num=10, min_ratio=0.01

Keeps: Only strong individual connections
Example output:
  L3[123] → Tm3[456]: 52 synapses, 3.2% ratio  ✓
  L3[234] → Tm3[789]: 28 synapses, 1.8% ratio  ✓
  (45 connections total)
```

### Type-Level Results
```
filter_by='type', min_synapse_num=10, min_ratio=0.01

Checks: Aggregate L3→Tm3 connectivity
If total_weight=5000, total_post=200000 → ratio=0.025 (2.5%)  ✓ PASS

Keeps: ALL L3→Tm3 connections (even weak ones)
Example output:
  L3[123] → Tm3[456]: 52 synapses, 3.2% ratio  ✓
  L3[234] → Tm3[789]: 28 synapses, 1.8% ratio  ✓
  L3[345] → Tm3[101]: 3 synapses, 0.2% ratio   ✓ (kept because type pair passed)
  (3,456 connections total)
```

## Use Cases

### Research Question 1: "Which individual neurons have strong connections?"
**Answer:** Use `filter_by='bodyId'`

```python
fc = FindNeuronConnection(
    sourceNeurons=['PPL1'],
    targetNeurons=['MBON'],
    min_synapse_num=20,
    min_ratio=0.05,  # 5% of inputs
    filter_by='bodyId'
)
```

**Result:** List of specific PPL1→MBON neuron pairs with strong connections

### Research Question 2: "Which neuron types are connected?"
**Answer:** Use `filter_by='type'`

```python
fc = FindNeuronConnection(
    sourceNeurons=['PPL1.*'],  # All PPL1 subtypes
    targetNeurons=['MBON.*'],  # All MBON subtypes
    min_synapse_num=10,
    min_ratio=0.01,  # 1% at type level
    filter_by='type'
)
```

**Result:** All connections between PPL1 and MBON types that show aggregate connectivity >1%

### Research Question 3: "Map entire circuit at type level"
**Answer:** Use `filter_by='type'` with loose filters

```python
fc = FindNeuronConnection(
    sourceNeurons=['L3'],
    targetNeurons=['Tm3', 'Tm9', 'Mi1'],
    min_synapse_num=5,
    min_ratio=0.001,  # 0.1% at type level
    filter_by='type',
    max_interlayer=3
)
```

**Result:** Complete type-level connectivity map with weak connections included

## Output Differences

### BodyId-Level Output

```
Filtered (bodyId level): 50000 → 120 connections (weight ≥ 10, ratio ≥ 0.01)
```

**Excel file contains:**
- Only strong individual connections
- Typically fewer connections
- Shows variability within types
- Good for identifying hub neurons

### Type-Level Output

```
Filtered (type level): 45 → 12 type pairs, 50000 → 8500 connections (weight ≥ 10, type-ratio ≥ 0.01)
```

**Excel file contains:**
- All connections between passing type pairs
- Typically more connections
- Shows type-level patterns
- Good for circuit diagrams

## Parameters in Folder Name

The folder naming now includes filter_by information:

```
# BodyId-level filtering (default)
L3_to_Tm3_L2w10r0_01p0_bodyId_20251025_203045

# Type-level filtering  
L3_to_Tm3_L2w10r0_01p0_type_20251025_203045
```

This makes it clear which filtering mode was used.

## Performance Considerations

### BodyId-Level
- ✓ Faster (filters early)
- ✓ Lower memory usage
- ✓ Smaller output files
- ✗ May miss type-level patterns

### Type-Level
- ✗ Slower (must aggregate first)
- ✗ Higher memory usage
- ✗ Larger output files
- ✓ Captures type-level patterns
- ✓ Better for circuit-level analysis

## Implementation Details

### BodyId-Level (Default)
1. Fetch connections with weight ≥ 1
2. Filter by min_synapse_num
3. Calculate ratio for each connection
4. Filter by min_ratio and min_traversal_probability
5. Return filtered connections

### Type-Level
1. Fetch connections with weight ≥ 1
2. Filter by min_synapse_num
3. Enrich with type information
4. Group by (type_pre, type_post)
5. Sum weights and post-synaptic sites
6. Calculate type-level ratios
7. Filter type pairs by thresholds
8. Return ALL connections from passing type pairs

## Best Practices

### Start with BodyId-Level
```python
# First pass: identify strong connections
fc = FindNeuronConnection(
    sourceNeurons=['L3'],
    targetNeurons=['Tm3'],
    min_ratio=0.05,  # Stringent
    filter_by='bodyId'
)
```

### Then Explore Type-Level
```python
# Second pass: see full type connectivity
fc = FindNeuronConnection(
    sourceNeurons=['L3'],
    targetNeurons=['Tm3'],
    min_ratio=0.01,  # More permissive
    filter_by='type'
)
```

### Compare Results
- BodyId-level shows "elite connections"
- Type-level shows "complete type circuit"
- Difference reveals connectivity diversity within types

## Edge Cases

### What if a type has only one neuron?
- BodyId and type filtering give same result
- No aggregation occurs

### What if filter_by='type' but using bodyId list?
- Still groups by type
- Uses type information from neuprint
- Works correctly

### What about neurons without types?
- Treated as separate "None" type
- Can aggregate "None → SomeType" connections
- Enrichment uses complete dataset

## Parameter Validation

The code validates filter_by parameter:

```python
if self.filter_by not in ['bodyId', 'type']:
    raise ValueError(f"filter_by must be 'bodyId' or 'type', got '{self.filter_by}'")
```

## Example Scripts

### Example 1: Compare Both Modes

```python
from coana import FindNeuronConnection

# BodyId-level (conservative)
fc_bodyid = FindNeuronConnection(
    sourceNeurons=['L3'],
    targetNeurons=['Tm3'],
    min_synapse_num=10,
    min_ratio=0.01,
    filter_by='bodyId'
)
fc_bodyid.InitializeNeuronInfo()
fc_bodyid.FindDirectConnections()  # Save as separate folder

# Type-level (inclusive)
fc_type = FindNeuronConnection(
    sourceNeurons=['L3'],
    targetNeurons=['Tm3'],
    min_synapse_num=10,
    min_ratio=0.01,
    filter_by='type'
)
fc_type.InitializeNeuronInfo()
fc_type.FindDirectConnections()  # Save as separate folder

# Compare Excel files to see difference
```

### Example 2: Multi-Layer Type Analysis

```python
fc = FindNeuronConnection(
    sourceNeurons=['L3'],
    targetNeurons=['Tm3', 'Tm9'],
    min_synapse_num=5,
    min_ratio=0.005,  # 0.5% at type level
    filter_by='type',
    max_interlayer=2
)
fc.InitializeNeuronInfo()
fc.FindAllPath()  # Complete type-level circuit
```

---

**Date**: October 25, 2025  
**Feature**: filter_by parameter for bodyId vs type level filtering  
**Default**: 'bodyId' (backward compatible)
