# Pathfinding Methods in FindAllPath

The `FindAllPath` function in `coana.py` supports multiple pathfinding algorithms to optimize performance based on the network structure and query requirements. You can select the algorithm using the `pathfinding` parameter in `FindNeuronConnection`.

## Available Algorithms

### 1. Bidirectional Search (Meet-in-the-middle)
**Parameter:** `pathfinding='Bidirectional'`

This is the **recommended** algorithm for most use cases. It simultaneously expands from the source neurons (forward) and target neurons (backward) and attempts to meet in the middle.

*   **Mechanism**: 
    *   Maintains two frontiers: one from sources, one from targets.
    *   Dynamically expands the smaller frontier to minimize the search space.
    *   Merges paths when frontiers intersect.
*   **Time Complexity**: $O(b^{d/2})$, where $b$ is the branching factor and $d$ is the path length. This is significantly faster than standard DFS ($O(b^d)$).
*   **Memory Cost**: Moderate to High. Needs to store frontiers from both directions.
*   **Pros**: 
    *   Drastically reduces search depth (e.g., for 4 hops, it only searches 2 hops from each side).
    *   Best balance of speed and memory for finding specific connections.
*   **Cons**: 
    *   Memory usage can grow if the graph is extremely dense.

### 2. Optimized Backward Search (DP)
**Parameter:** `pathfinding='DP'`

This algorithm uses Dynamic Programming to build paths backwards from the target neurons.

*   **Mechanism**:
    *   Starts with paths of length 0 (just the targets).
    *   Iteratively extends paths backwards by one hop using the reverse graph.
    *   Stores all valid path suffixes of length $L$ at each node.
    *   Finally, filters for paths that start at the specified source neurons.
*   **Time Complexity**: $O(L \cdot E)$, where $L$ is max path length and $E$ is number of edges.
*   **Memory Cost**: High. Stores all valid path suffixes for every node at every length.
*   **Pros**:
    *   Very efficient when you have a small set of targets and want to find *all* paths reaching them from a large set of sources.
    *   Avoids redundant computations for shared suffixes.
*   **Cons**:
    *   Can consume significant memory if the number of paths is large.

### 3. Memoized DFS
**Parameter:** `pathfinding='MemoizedDFS'`

A recursive Depth-First Search augmented with memoization (caching).

*   **Mechanism**:
    *   Recursively explores paths from sources.
    *   Caches the result of `(node, length_remaining)` -> `list of path suffixes`.
    *   If a node is visited again with the same remaining length, the cached result is returned immediately.
*   **Time Complexity**: Depends on the number of unique states `(node, length)`.
*   **Memory Cost**: Moderate. Stores the cache of path suffixes.
*   **Pros**:
    *   Faster than standard DFS for graphs with many overlapping paths.
*   **Cons**:
    *   Recursion depth limits in Python (though usually not an issue for typical connectome path lengths of < 10).
    *   Overhead of managing the cache.

### 4. Standard DFS
**Parameter:** `pathfinding='DFS'`

The classic Depth-First Search algorithm.

*   **Mechanism**:
    *   Recursively explores one path at a time until it hits a target or the max length.
    *   Backtracks to explore other branches.
*   **Time Complexity**: $O(b^d)$. Exponential with depth.
*   **Memory Cost**: Low. Only stores the current path stack.
*   **Pros**:
    *   Lowest memory footprint.
    *   Simple implementation.
*   **Cons**:
    *   **Extremely slow** for dense graphs or paths longer than 2-3 hops.
    *   Recomputes shared sub-paths many times.
    *   Not recommended for complex connectome analysis.

## Comparison Summary

| Algorithm | Speed | Memory | Best Use Case |
| :--- | :--- | :--- | :--- |
| **Bidirectional** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | **General purpose**, finding connections between specific groups. |
| **DP (Backward)** | ⭐⭐⭐⭐ | ⭐⭐ | Many sources to few targets. |
| **Memoized DFS** | ⭐⭐⭐ | ⭐⭐⭐ | Dense graphs with overlapping paths. |
| **Standard DFS** | ⭐ | ⭐⭐⭐⭐⭐ | Low memory environments, very short paths. |

## Parallel Processing

Note that `FindAllPath` also supports a `use_parallel=True` mode. When enabled, this overrides the `pathfinding` parameter and uses a parallelized implementation (typically based on optimized forward/backward searches distributed across CPU cores). This is recommended for large-scale analyses.
