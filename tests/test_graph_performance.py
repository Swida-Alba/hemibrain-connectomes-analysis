
import sys
import os
import time
import pandas as pd
import numpy as np
import networkx as nx

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from core.fast_graph import FastGraph

def benchmark_graph_building():
    print("Generating random graph data (100k nodes, 2M edges)...")
    num_nodes = 100000
    num_edges = 2000000
    
    # Generate random edges
    sources = np.random.randint(0, num_nodes, num_edges)
    targets = np.random.randint(0, num_nodes, num_edges)
    weights = np.random.rand(num_edges)
    
    df = pd.DataFrame({
        'source': sources,
        'target': targets,
        'weight': weights
    })
    
    print(f"Data generated. DataFrame shape: {df.shape}")
    
    # Benchmark NetworkX
    print("\nBenchmarking NetworkX DiGraph building...")
    start_time = time.time()
    G_nx = nx.DiGraph()
    # Standard way to build from dataframe in NetworkX
    # Iterating rows is slow, so let's try the fastest way usually recommended: from_pandas_edgelist
    # But wait, the user's original code was iterating rows.
    # "for idx in conn_df.index: ..."
    # I should probably test both the naive iteration (as it was in the code) and the optimized way if possible,
    # but the user asked to compare "FastGraph and nx.DiGraph of graph building".
    # I will test the standard nx.from_pandas_edgelist as a fair baseline for "using NetworkX properly",
    # and maybe the loop version to show how bad it was.
    # Actually, let's stick to the loop version first as that's what we are replacing, 
    # BUT 2M edges with a python loop will take forever (seconds to minutes).
    # Let's test the optimized NetworkX loading vs FastGraph.
    
    G_nx = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr='weight', create_using=nx.DiGraph)
    nx_time = time.time() - start_time
    print(f"NetworkX (from_pandas_edgelist) time: {nx_time:.4f} seconds")
    print(f"Nodes: {G_nx.number_of_nodes()}, Edges: {G_nx.number_of_edges()}")
    
    # Benchmark FastGraph
    print("\nBenchmarking FastGraph building...")
    start_time = time.time()
    G_fast = FastGraph()
    G_fast.build_from_dataframe(df, 'source', 'target', 'weight')
    fast_time = time.time() - start_time
    print(f"FastGraph time: {fast_time:.4f} seconds")
    print(f"Nodes: {G_fast.number_of_nodes()}, Edges: {G_fast.number_of_edges()}")
    
    print(f"\nSpeedup: {nx_time / fast_time:.2f}x")

if __name__ == "__main__":
    benchmark_graph_building()
