#!/usr/bin/env python3
"""
Test script for cache performance verification.

Tests:
1. coana.py: FindNeuronConnection dict-based in-memory cache
2. connectivity_profiler.py: 3-tier cache (memory -> disk index -> disk load)

Run from project root:
    cd /path/to/project
    python dev/test_cache_performance.py
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from comparison.connectivity_profiler import ConnectivityProfiler, ProfilerConfig


def test_connectivity_profiler_cache():
    """Test ConnectivityProfiler 3-tier cache performance."""
    print("=" * 70)
    print("Testing ConnectivityProfiler Cache Performance")
    print("=" * 70)
    
    # Initialize profiler
    profiler = ConnectivityProfiler(
        datasets=['hemibrain:v1.2.1'],
        config=ProfilerConfig(top_k_bodyid=20),
        verbose=True
    )
    
    test_neurons = ['aMe12', 'aMe10', 'aMe8', 'Mi4', 'Mi9']
    
    print("\n--- Test 1: First fetch (no cache) ---")
    times_first = []
    for neuron in test_neurons:
        t1 = time.perf_counter()
        profile = profiler.get_profile(neuron, 'hemibrain:v1.2.1')
        t2 = time.perf_counter()
        elapsed = (t2 - t1) * 1000
        times_first.append(elapsed)
        print(f"  {neuron}: {elapsed:.1f} ms (up:{len(profile.upstream_partners)}, down:{len(profile.downstream_partners)})")
    
    print(f"\n  Average first fetch: {sum(times_first)/len(times_first):.1f} ms")
    
    print("\n--- Test 2: Second fetch (memory cache) ---")
    times_memory = []
    for neuron in test_neurons:
        t1 = time.perf_counter()
        profile = profiler.get_profile(neuron, 'hemibrain:v1.2.1')
        t2 = time.perf_counter()
        elapsed = (t2 - t1) * 1000
        times_memory.append(elapsed)
        print(f"  {neuron}: {elapsed:.3f} ms")
    
    print(f"\n  Average memory cache: {sum(times_memory)/len(times_memory):.3f} ms")
    
    # Clear memory cache to test disk cache
    print("\n--- Test 3: Clearing memory cache, testing disk cache ---")
    profiler._memory_cache.clear()
    
    times_disk = []
    for neuron in test_neurons:
        t1 = time.perf_counter()
        profile = profiler.get_profile(neuron, 'hemibrain:v1.2.1')
        t2 = time.perf_counter()
        elapsed = (t2 - t1) * 1000
        times_disk.append(elapsed)
        print(f"  {neuron}: {elapsed:.3f} ms")
    
    print(f"\n  Average disk cache (in-memory DataFrame): {sum(times_disk)/len(times_disk):.3f} ms")
    
    # Clear both caches to test cold disk load
    print("\n--- Test 4: Clearing all caches, testing cold disk load ---")
    profiler._memory_cache.clear()
    profiler._disk_cache_df.clear()
    profiler._disk_cache_index.clear()
    
    times_cold = []
    for neuron in test_neurons:
        t1 = time.perf_counter()
        profile = profiler.get_profile(neuron, 'hemibrain:v1.2.1')
        t2 = time.perf_counter()
        elapsed = (t2 - t1) * 1000
        times_cold.append(elapsed)
        print(f"  {neuron}: {elapsed:.3f} ms")
    
    print(f"\n  First cold load: {times_cold[0]:.1f} ms (includes disk read + index build)")
    print(f"  Subsequent (disk cache in memory): {sum(times_cold[1:])/len(times_cold[1:]):.3f} ms")
    
    # Summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"  First fetch (no cache):     {sum(times_first)/len(times_first):.1f} ms average")
    print(f"  Memory cache:               {sum(times_memory)/len(times_memory):.3f} ms average")
    print(f"  Disk cache (in-memory DF):  {sum(times_disk)/len(times_disk):.3f} ms average")
    print(f"  Cold disk load (first):     {times_cold[0]:.1f} ms")
    print(f"  Cold disk (subsequent):     {sum(times_cold[1:])/len(times_cold[1:]):.3f} ms average")
    
    speedup_memory = times_first[0] / times_memory[0] if times_memory[0] > 0 else float('inf')
    speedup_disk = times_first[0] / times_disk[0] if times_disk[0] > 0 else float('inf')
    
    print(f"\n  Speedup (memory cache vs first fetch): {speedup_memory:.0f}x")
    print(f"  Speedup (disk cache vs first fetch):   {speedup_disk:.0f}x")


def test_coana_cache():
    """Test coana.py FindNeuronConnection dict-based cache performance."""
    print("\n" + "=" * 70)
    print("Testing coana.py FindNeuronConnection Cache Performance")
    print("=" * 70)
    
    from coana import FindNeuronConnection
    
    fnc = FindNeuronConnection(dataset='hemibrain:v1.2.1', verbose_mode='simple')
    
    print("\n--- Test 1: First load (from disk) ---")
    t1 = time.perf_counter()
    conn_db = fnc._load_connection_db()
    t2 = time.perf_counter()
    first_load = (t2 - t1) * 1000
    print(f"  Connection DB: {first_load:.1f} ms ({len(conn_db):,} rows, {len(fnc._conn_index):,} unique upstream)")
    
    t1 = time.perf_counter()
    neuron_idx = fnc._load_neuron_index()
    t2 = time.perf_counter()
    first_idx_load = (t2 - t1) * 1000
    print(f"  Neuron Index:  {first_idx_load:.1f} ms ({len(neuron_idx):,} rows, {len(fnc._neuron_index_dict):,} in dict)")
    
    print("\n--- Test 2: Second load (from memory) ---")
    t1 = time.perf_counter()
    conn_db = fnc._load_connection_db()
    t2 = time.perf_counter()
    second_load = (t2 - t1) * 1000
    print(f"  Connection DB: {second_load:.4f} ms")
    
    t1 = time.perf_counter()
    neuron_idx = fnc._load_neuron_index()
    t2 = time.perf_counter()
    second_idx_load = (t2 - t1) * 1000
    print(f"  Neuron Index:  {second_idx_load:.4f} ms")
    
    print("\n--- Test 3: O(1) dict lookups ---")
    test_bodyids = list(fnc._neuron_index_dict.keys())[:10]
    
    t1 = time.perf_counter()
    for bodyId in test_bodyids * 100:  # 1000 lookups
        _ = fnc._neuron_index_dict.get(bodyId)
    t2 = time.perf_counter()
    dict_lookup_time = (t2 - t1) * 1000
    print(f"  1000 dict lookups: {dict_lookup_time:.3f} ms ({dict_lookup_time/1000*1000:.3f} µs per lookup)")
    
    t1 = time.perf_counter()
    for bodyId in test_bodyids * 100:  # 1000 lookups
        if bodyId in fnc._conn_index:
            _ = fnc._conn_index[bodyId]
    t2 = time.perf_counter()
    conn_lookup_time = (t2 - t1) * 1000
    print(f"  1000 conn lookups: {conn_lookup_time:.3f} ms ({conn_lookup_time/1000*1000:.3f} µs per lookup)")
    
    # Summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"  First load (disk):    {first_load:.1f} ms")
    print(f"  Second load (memory): {second_load:.4f} ms")
    print(f"  Speedup:              {first_load/second_load:.0f}x" if second_load > 0 else "  Speedup: instant")


if __name__ == '__main__':
    print("Cache Performance Test Suite")
    print("=" * 70)
    print()
    
    # Test coana cache first
    test_coana_cache()
    
    print("\n")
    
    # Test connectivity profiler cache
    test_connectivity_profiler_cache()
    
    print("\n" + "=" * 70)
    print("✓ All cache tests completed!")
    print("=" * 70)
