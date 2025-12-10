import sys
import os
import time
import networkx as nx
from collections import defaultdict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.core.fast_graph import FastGraph

import random

def create_large_random_graph(num_nodes=100000, num_edges=3000000):
    """
    Creates a large random graph.
    """
    print(f"Generating graph with {num_nodes} nodes and {num_edges} edges...")
    G = FastGraph()
    
    # Generate edges
    # To ensure connectivity, we can create a spanning tree first or just random edges
    # For pure performance testing, random edges are fine.
    
    # Use integer IDs for speed in generation
    nodes = range(num_nodes)
    
    # Pre-generate edges to avoid overhead during add_edge loop if possible, 
    # but FastGraph.add_edge is simple.
    
    # Let's use a simple loop with random.
    # To make it faster, we can generate in batches or just loop.
    # 3M iterations is fast in Python.
    
    start_gen = time.time()
    
    # Create a few guaranteed paths from Source to Target
    source = "S"
    target = "T"
    
    # Connect Source to random nodes
    for _ in range(50):
        n = random.randint(0, num_nodes-1)
        G.add_edge(source, n, 1)
        
    # Connect random nodes to Target
    for _ in range(50):
        n = random.randint(0, num_nodes-1)
        G.add_edge(n, target, 1)
        
    # Random edges
    for _ in range(num_edges):
        u = random.randint(0, num_nodes-1)
        v = random.randint(0, num_nodes-1)
        G.add_edge(u, v, 1)
        
    print(f"Graph generation took {time.time() - start_gen:.2f}s")
    return G, source, target

def test_methods():
    # Use smaller graph for quick verification, or large for stress test
    # G = create_test_graph()
    # source = 'S'
    # target = 'T'
    
    G, source, target = create_large_random_graph(100000, 3000000)
    
    cutoff = 4
    
    print(f"Test Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Finding paths from {source} to {target} (len={cutoff})")
    print("-" * 60)
    
    methods = [
        ("Bidirectional BFS", G.find_paths_bidirectional_bfs, {}),
        ("Backward DP", G.find_paths_backward_dp, {}),
        ("Backward Memoized DFS", G.find_paths_memoized_dfs, {'direction': 'backward'}),
        ("Bidirectional Memoized DFS", G.find_paths_meet_in_the_middle, {})
    ]
    
    for name, func, kwargs in methods:
        print(f"Testing {name}...", end='', flush=True)
        start = time.time()
        # Convert to list to force execution
        paths = list(func([source], [target], cutoff, **kwargs))
        elapsed = time.time() - start
        print(f"\r{name:<30} | Found: {len(paths):<6} | Time: {elapsed:.4f}s")
        
        # Verify paths (check first few)
        if len(paths) > 0:
            p = paths[0]
            assert p[0] == source and p[-1] == target
            assert len(p) - 1 <= cutoff
            assert len(p) - 1 <= cutoff

if __name__ == "__main__":
    test_methods()
