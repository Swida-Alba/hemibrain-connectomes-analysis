# Pathfinding Methods in FindAllPath

The `FindAllPath` function in `coana.py` leverages the optimized `FastGraph` core to support multiple advanced pathfinding algorithms. You can select the algorithm using the `pathfinding` parameter in `FindNeuronConnection`.

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
*   **Cons**: High memory usage for dense graphs as it stores full layers.

### 2. Meet-in-the-middle DFS (Bidirectional Memoized DFS)
**Parameter:** `pathfinding='MemoizedDFS'`
**Method:** `FastGraph.find_paths_meet_in_the_middle`

**The Gold Standard for deep paths.** This is the most robust algorithm for finding long paths (5+ hops) in dense connectomes.

*   **Mechanism**: 
    *   **Forward Phase**: Performs DFS from sources for /2$ steps. Stores these half-paths in a memory-efficient hash map.
    *   **Backward Phase**: Performs DFS from targets for /2$ steps.
    *   **Join**: When the backward search hits a node existing in the forward map, it stitches the paths together.
*   **Best Use Case**: Deep pathfinding (Length $\ge$ 5) where standard DFS is too slow and BFS runs out of memory.
*   **Pros**: Drastically reduces memory usage compared to storing full paths.
*   **Cons**: Requires storing half-paths in memory (though much less than full paths).

### 3. Backward Reachability (DP)
**Parameter:** `pathfinding='DP'`
**Method:** `FastGraph.find_paths_backward_dp`

A hybrid approach combining Backward BFS for pruning and Forward DFS for path construction.

*   **Mechanism**:
    *   **Phase 1 (Backward Reachability)**: Computes sets $ containing all nodes that can reach a target in exactly $ steps.
    *   **Phase 2 (Guided DFS)**: Runs a forward DFS from sources, but only visits a neighbor $ if  \in R_{k-1}$.
*   **Best Use Case**: Sparse graphs or queries where many branches lead to dead ends.
*   **Pros**: **Lowest memory footprint**. Aggressively prunes dead ends before the main search.
*   **Cons**: Requires two passes over the graph.

### 4. Backward Memoized DFS
**Parameter:** `pathfinding='DFS'`
**Method:** `FastGraph.find_paths_memoized_dfs` (direction='backward')

A standard Depth-First Search starting from targets, augmented with memoization.

*   **Mechanism**:
    *   Recursively explores paths from targets to sources.
    *   Memoizes which nodes can reach the source within $k$ steps.
*   **Best Use Case**: When targets are fewer than sources.
*   **Pros**: Faster than pure backtracking due to memoization.
*   **Cons**: Can use significant memory for memoization tables.

### 5. Backward DFS with Backtracking
**Parameter:** `pathfinding='Backtracking'`
**Method:** `FastGraph.find_paths_dfs_backtracking`

A pure Depth-First Search (Iterative Deepening) starting from targets, with **no memoization**.

*   **Mechanism**:
    *   Explores paths from targets to sources using recursion and backtracking.
    *   Does not store visited states (except for current path cycle checking).
*   **Best Use Case**: **Extreme memory constraints** where even memoization tables are too large.
*   **Pros**: Absolute lowest memory overhead (excluding the reverse graph structure).
*   **Cons**: Slower than memoized methods due to re-visiting nodes.

## Comparison Summary

| Algorithm | Parameter | Underlying Method | Best For | Memory |
| :--- | :--- | :--- | :--- | :--- |
| **Meet-in-the-middle** | `MemoizedDFS` | `find_paths_meet_in_the_middle` | **Deep Paths (L $\ge$ 5)** | Low |
| **Bidirectional BFS** | `Bidirectional` | `find_paths_bidirectional_bfs` | Shortest Paths | High |
| **Backward DP** | `DP` | `find_paths_backward_dp` | Pruning Dead Ends | Low |
| **Backward Memoized** | `DFS` | `find_paths_memoized_dfs` | Standard Traversal | Medium |
| **Backtracking** | `Backtracking` | `find_paths_dfs_backtracking` | **Extreme Memory Constraints** | **Lowest** |


