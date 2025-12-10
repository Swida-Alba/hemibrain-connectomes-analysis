import sys
import os
import time
import psutil
import random
from collections import defaultdict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.core.fast_graph import FastGraph

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def create_large_random_graph(num_nodes=1000000, num_edges=30000000):
    """
    Creates a large random graph.
    """
    print(f"Generating graph with {num_nodes:,} nodes and {num_edges:,} edges...")
    start_mem = get_memory_usage_mb()
    start_gen = time.time()
    
    G = FastGraph()
    
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
    print("  Generating edges...", end='', flush=True)
    
    # Use a simple loop
    for i in range(num_edges):
        if i % 1000000 == 0:
            print(f".", end='', flush=True)
        u = random.randint(0, num_nodes-1)
        v = random.randint(0, num_nodes-1)
        G.add_edge(u, v, 1)
    print(" Done.")
        
    elapsed = time.time() - start_gen
    end_mem = get_memory_usage_mb()
    print(f"Graph generation took {elapsed:.2f}s")
    print(f"Graph memory usage: {end_mem - start_mem:.2f} MB (Total: {end_mem:.2f} MB)")
    
    return G, source, target

def test_methods():
    # 1M nodes, 30M edges
    G, source, target = create_large_random_graph(1000000, 30000000)
    
    cutoff = 4
    
    print(f"\nTest Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"Finding paths from {source} to {target} (len={cutoff})")
    print("-" * 90)
    print(f"{'Method':<30} | {'Found':<8} | {'Time (s)':<10} | {'Peak Mem (MB)':<15} | {'Mem Delta (MB)':<15}")
    print("-" * 90)
    
    methods = [
        ("Bidirectional BFS", G.find_paths_bidirectional_bfs, {}),
        ("Backward DP", G.find_paths_backward_dp, {}),
        ("Backward Memoized DFS", G.find_paths_memoized_dfs, {'direction': 'backward'}),
        ("Bidirectional Memoized DFS", G.find_paths_meet_in_the_middle, {}),
        ("Backtracking DFS", G.find_paths_dfs_backtracking, {})
    ]
    
    for name, func, kwargs in methods:
        # Force garbage collection before each test
        import gc
        gc.collect()
        time.sleep(1)
        
        start_mem = get_memory_usage_mb()
        start_time = time.time()
        
        try:
            # Convert to list to force execution and store all paths
            paths = list(func([source], [target], cutoff, **kwargs))
            count = len(paths)
        except Exception as e:
            print(f"{name:<30} | FAILED: {e}")
            continue
            
        # Force GC to measure true persistent memory
        gc.collect()
        
        elapsed = time.time() - start_time
        end_mem = get_memory_usage_mb()
        mem_delta = end_mem - start_mem
        
        print(f"{name:<30} | {count:<8} | {elapsed:<10.4f} | {end_mem:<15.2f} | {mem_delta:<15.2f}")
        
        # Verify paths (check first few)
        if len(paths) > 0:
            p = paths[0]
            assert p[0] == source and p[-1] == target
            assert len(p) - 1 <= cutoff

if __name__ == "__main__":
    test_methods()
