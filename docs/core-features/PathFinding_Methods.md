# Pathfinding Methods in FindAllPath

The `FindAllPath` function in `coana.py` leverages the optimized `FastGraph` core to support multiple advanced pathfinding algorithms. You can select the algorithm using the `pathfinding` parameter in `FindNeuronConnection` (and from the **Algorithm** selector in the Find All Paths and Cross-Dataset tabs).

The four algorithms are also evaluated theoretically and by measured time/memory in [PATHFINDING_ALGORITHM_EVALUATION.md](../technical/PATHFINDING_ALGORITHM_EVALUATION.md), with the benchmark harness at `examples/performance/benchmark_pathfinding.py`.

## Available Algorithms

### 1. Bidirectional Search (Layer Intersection)
**Parameter:** `pathfinding='Bidirectional'`
**Method:** `FastGraph.find_paths_bidirectional_bfs`

This algorithm performs a simultaneous Breadth-First Search (BFS) from both the source and target sets.

*   **Mechanism**: 
    *   Expands "layers" of nodes from sources (Forward) and targets (Backward).
    *   Finds the intersection of these layers at the midpoint (e.g., for length 4, intersects Forward Layer 2 and Backward Layer 2).
    *   Reconstructs paths by backtracking from the intersection nodes.
*   **Best Use Case**: Finding **shortest paths** or all paths of a specific length in shallow graphs.
*   **Pros**: Guarantees finding shortest paths first.
*   **Cons**: High memory usage for dense graphs as it stores full layers (measured 452 MB at 4 layers, 1.1 GB at 5).

### 2. Backward Reachability (DP)
**Parameter:** `pathfinding='DP'`
**Method:** `FastGraph.find_paths_backward_dp`

A hybrid approach combining Backward BFS for pruning and Forward DFS for path construction. **The recommended default** (fastest measured at 2–3 intermediate layers).

*   **Mechanism**:
    *   **Phase 1 (Backward Reachability)**: Computes sets R_k containing all nodes that can reach a target in exactly k steps.
    *   **Phase 2 (Guided DFS)**: Runs a forward DFS from sources, but only visits a neighbor v if v ∈ R_{k-1}.
*   **Best Use Case**: Sparse graphs or queries where many branches lead to dead ends.
*   **Pros**: **Lowest memory footprint**. Aggressively prunes dead ends before the main search.
*   **Cons**: Requires two passes over the graph; degenerates on very deep (5+) queries.

### 3. Memoized DFS (forward) — the recommended default
**Parameter:** `pathfinding='MemoizedDFS'`
**Method:** `FastGraph.find_paths_memoized_dfs` (direction='forward')

A depth-first traversal from the sources that memoizes which successors
can still reach a target at each remaining depth, so every (node, depth)
state is computed once and reconstruction walks only valid successors.

*   **Mechanism**:
    *   Recursively explores paths from sources to targets.
    *   Memoizes valid successors per (node, remaining-depth); dead-end branches are never walked.
*   **Best Use Case**: the common case — fastest measured at **every** depth on real connectomes (no reversed-graph copy; the memo stays inside the source-side cone).
*   **Pros**: Fastest measured; smallest peak allocation at 2–4 layers.
*   **Cons**: Can use significant memory for memoization tables on very large source cones.

### 4. Memoized DFS (backward)
**Parameter:** `pathfinding='DFS'`
**Method:** `FastGraph.find_paths_memoized_dfs` (direction='backward')

The same memoized DFS started from the targets on the reversed graph.

*   **Mechanism**:
    *   Recursively explores paths from targets to sources.
    *   Memoizes which nodes can reach the source within k steps.
*   **Best Use Case**: When targets are fewer than sources.
*   **Pros**: Fast at 4–5 layers with few targets (measured 0.99–1.03 s at 4 layers, fastest at 5).
*   **Cons**: Builds a reversed-graph copy (~250 MB on a 722k-edge graph) and explores the target-side cone.

### 5. Meet-in-the-middle DFS
**Parameter:** `pathfinding='MeetInMiddle'`
**Method:** `FastGraph.find_paths_meet_in_the_middle`

*   **Mechanism**: 
    *   **Forward Phase**: Performs DFS from sources for ⌊L/2⌋ steps. Stores these half-paths in a memory-efficient hash map.
    *   **Backward Phase**: Performs DFS from targets for the remaining steps.
    *   **Join**: When the backward search hits a node existing in the forward map, it stitches the paths together.
*   **Best Use Case**: Fastest at shallow depths (2–3 intermediate layers, within noise of MemoizedDFS).
*   **Pros**: Half-depth exponential in theory.
*   **Cons**: The half-path table is rebuilt for every length, the forward walk is unpruned and the pairwise stitching costs more than a fully-pruned shared memo on real connectomes (measured 2.17 s at 4 layers, 5.88 s at 5).

## Comparison Summary

| Algorithm | Parameter | Underlying Method | Best For | Time (worst) | Memory (worst) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Memoized DFS (fwd)** | `MemoizedDFS` | `find_paths_memoized_dfs` | **Default; all depths** | O(L·E + P·L) | O(L·E) |
| **Memoized DFS (bwd)** | `DFS` | `find_paths_memoized_dfs` (backward) | **Deep paths, few targets** | O(L·E + P·L) | O(L·E) |
| **Meet-in-the-middle** | `MeetInMiddle` | `find_paths_meet_in_the_middle` | Shallow; safe mid-ground | O(b^{L/2}·L + P·L) | O(b^{L/2}·L) |
| **Backward DP** | `DP` | `find_paths_backward_dp` | Robust, no reverse copy | O(L·E + P·L) | O(L·V) |
| **Bidirectional BFS** | `Bidirectional` | `find_paths_bidirectional_bfs` | Shortest Paths | O(L·E + P·L) | **O(L·(V+E))** |

*L = cutoff (max path length), E = edges, V = nodes, b = branching factor, P = number of found paths.*

## Measured Evaluation (2026-08)

Benchmarked on real hemibrain v1.2.1 queries (connections ≥ 5 synapses) —
all algorithms returned **identical path sets** in every scenario:

- **2–3 intermediate layers**: LC → MBON/PPL, 10,331 nodes, 301,443 edges, 4 sources × 5 targets (14 / 2,874 paths)
- **4–5 intermediate layers**: LPLC1 → MBON01, 24,576 nodes, 722,143 edges, 2 sources × 2 targets (35,819 paths; 5M+ paths at 5 layers, capped)

| Algorithm | 2 layers | 3 layers | 4 layers | 5 layers |
| :--- | ---: | ---: | ---: | ---: |
| **MemoizedDFS** (default) | 0.01 s / 0.1 MB | 0.06 s / 1.7 MB | **0.23 s** / 7.6 MB | 4.97 s / 651 MB |
| **DFS** (backward) | 0.02 s / 0.3 MB | 0.05 s / 1.8 MB | 0.42 s / 9.6 MB | 5.01 s / 612 MB |
| **MeetInMiddle** | **0.00 s** / 0.3 MB | **0.01 s** / 0.6 MB | 1.25 s / 4.8 MB | 6.18 s / 615 MB |
| **DP** | 0.06 s / 0.8 MB | 0.02 s / 1.5 MB | 0.44 s / 9.4 MB | 20.2 s / 651 MB |
| **Bidirectional** | 0.07 s / 18.9 MB† | 0.18 s / 59.4 MB† | 0.64 s / 194.9 MB† | **4.54 s** / 892 MB† |

*Fastest time per column in **bold**; † = highest memory in that column.
**2026-08 optimization:** FastGraph keeps a lazy reverse adjacency index
instead of copying the whole reversed graph per run — the earlier
100–450 MB footprints were mostly those copies and are gone (DP dropped
from 272 MB to 9.4 MB at 4 layers). At 5 layers every value includes
~610 MB of the 5 million collected paths.*

Key findings:

- **MemoizedDFS (forward) is fastest at 4 layers** (0.23 s) and within
  noise elsewhere; all algorithms except Bidirectional now run in
  <10 MB at 2–4 layers thanks to the lazy reverse-adjacency index.
- **MeetInMiddle is fastest at 2–3 layers** (0.00–0.01 s).
- **DFS (backward) is the best alternative for few targets**; it no longer
  pays for a reversed-graph copy (0.3–9.6 MB at 2–4 layers).
- **Bidirectional** remains the memory leader (18.9 → 892 MB) — its full
  layer trees dominate now that the other algorithms no longer copy the
  graph.
- **DP** degenerates on deep queries (20.2 s at 5 layers).

**Recommendation (2026-08)**: `MemoizedDFS` is the default (fastest
measured at every depth, no graph copy). Use `DFS` for deep paths with few
targets, `MeetInMiddle` for shallow queries, `Bidirectional` only for
shortest-first output with memory to spare.


