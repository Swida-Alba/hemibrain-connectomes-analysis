# FindAllPath: exclude_searched_neurons True vs False

## TL;DR

**For 99% of use cases (finding source→target paths): Use `exclude_searched_neurons=True`**
- ✅ Finds ALL acyclic paths from sources to targets
- ✅ Much faster (queries each neuron only once)
- ✅ Complete for standard pathway analysis
- ✅ Ignores cycles/loops (which aren't needed for source→target paths)

## The Key Insight

When querying a neuron, we get **ALL its downstream connections**, including to neurons not yet discovered. This means:

```python
# When we query neuron A at Layer 0:
connections = fetch_connections(A, downstream=None)

# We get ALL connections FROM A:
# - A → B (will be discovered in Layer 1)
# - A → C (will be discovered in Layer 1)  
# - A → Z (will be discovered in Layer 3)
# - A → Target (direct path!)

# ALL of these are found and stored, even if Z and Target 
# haven't been discovered yet!
```

## What Each Mode Does

### `exclude_searched_neurons=True` (RECOMMENDED)

**Query Strategy:**
- Layer 0→1: Query A (sources)
- Layer 1→2: Query B, C (new neurons from Layer 1)
- Layer 2→3: Query D, E (new neurons from Layer 2)
- Each neuron queried **exactly once**

**What it finds:**
- ✅ ALL connections from each neuron (including to future-discovered neurons)
- ✅ ALL acyclic paths from sources to targets
- ✅ Direct paths (A→T)
- ✅ Multi-hop paths (A→B→C→T)
- ✅ All paths with length ≤ max_interlayer+1

**What it intentionally ignores:**
- 🔄 Cyclic paths (A→B→C→B→T)
- 🔄 Feedback loops
- 🔄 Paths that revisit neurons

**Performance:** Fast! Linear query growth with network size.

### `exclude_searched_neurons=False`

**Query Strategy:**
- Layer 0→1: Query A
- Layer 1→2: Query A, B, C (re-query A!)
- Layer 2→3: Query A, B, C, D, E (re-query everything!)
- Each neuron queried **multiple times**

**What it finds:**
- ✅ Everything that True finds
- ➕ Cyclic paths (A→B→C→B→T)
- ➕ Feedback loops
- ➕ Paths that revisit neurons

**Performance:** Slower! Quadratic query growth with network size.

## Concrete Example

```
Network:
- A (source)
- A → B
- A → C
- A → T (direct)
- B → D
- C → E
- D → T
- E → T
- T → B (feedback loop)

max_interlayer = 3
```

### With `exclude_searched_neurons=True`:

```
Layer 0→1: Query A
  Finds: A→B, A→C, A→T
  Discovers: B, C, T (all in Layer 1)
  
Layer 1→2: Query B, C, T
  Finds: B→D, C→E, T→B
  Discovers: D, E (Layer 2)
  Note: T→B is found but B already in network
  
Layer 2→3: Query D, E
  Finds: D→T, E→T
  Note: T already in network, these are alternative paths

Paths to T found:
1. A → T (1 hop) ✓
2. A → B → D → T (3 hops) ✓
3. A → C → E → T (3 hops) ✓

Cyclic path NOT found:
- A → T → B → D → T (revisits T) ✗
```

### With `exclude_searched_neurons=False`:

```
Layer 0→1: Query A
  Same as True
  
Layer 1→2: Query A, B, C, T (re-query A)
  Finds everything True found + re-confirms A's connections
  
Layer 2→3: Query A, B, C, T, D, E (re-query everything)
  Finds: Everything True found
  Plus can now construct: A→T→B→D→T (cyclic)

Paths to T found:
1. A → T (1 hop) ✓
2. A → B → D → T (3 hops) ✓
3. A → C → E → T (3 hops) ✓
4. A → T → B → D → T (4 hops, cyclic) ✓
```

## Performance Comparison

For a network discovering 100 → 1,000 → 5,000 neurons per layer:

| Metric | True | False |
|--------|------|-------|
| Layer 0→1 queries | 100 | 100 |
| Layer 1→2 queries | 1,000 | 1,100 |
| Layer 2→3 queries | 5,000 | 6,100 |
| **Total queries** | **6,100** | **7,300** |
| Speed | Fast | 20% slower |

For larger networks (10k+ neurons), the difference becomes more dramatic!

## When to Use Each Mode

### Use `exclude_searched_neurons=True` when:
- ✅ Finding pathways from sources to targets (standard use case)
- ✅ You need acyclic paths only
- ✅ You want fast performance
- ✅ Network has >1000 neurons
- ✅ You don't care about feedback loops

**This is 99% of connectomics pathway analysis!**

### Use `exclude_searched_neurons=False` when:
- 🔄 Analyzing feedback circuits specifically
- 🔄 Need to find cyclic pathways
- 🔄 Studying reciprocal loops
- 🔄 Complete network topology analysis (not just pathfinding)
- 🔄 Research specifically requires paths that revisit neurons

**This is specialized analysis only!**

## Bottom Line

For finding all paths from sources to targets (the primary use case of `FindAllPath`):

**`exclude_searched_neurons=True` gives you:**
- ✅ Complete results (all acyclic paths)
- ✅ Fast performance
- ✅ Everything you need

**There's no reason to use `False` unless you specifically need cyclic paths!**
