#!/usr/bin/env python3
"""
Example: Building Connectivity Profile Cache

This example demonstrates how to pre-build connectivity profiles for
efficient homolog finding. The cache stores:
- Typed partner weights and ranks
- 2-hop expansion for untyped partners
- Dynamic expansion metadata

Usage:
    python Example_BuildConnectivityCache.py [dataset] [--max-neurons N]
    python Example_BuildConnectivityCache.py --read [dataset]  # Read existing cache
    python Example_BuildConnectivityCache.py --stats [dataset]  # Show cache stats

Author: Hemibrain Analysis Team
Date: November 2024
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from comparison.connectivity_profiler import ConnectivityProfiler, ProfilerConfig


def main():
    """Build connectivity profile cache for a dataset."""
    
    # Check for special modes
    if '--read' in sys.argv:
        read_cache_demo()
        return
    
    if '--stats' in sys.argv:
        show_cache_stats()
        return
    
    # Parse arguments
    dataset = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'hemibrain_v1_2_1'
    max_neurons = None
    
    if '--max-neurons' in sys.argv:
        idx = sys.argv.index('--max-neurons')
        if idx + 1 < len(sys.argv):
            max_neurons = int(sys.argv[idx + 1])
    
    print(f"=" * 60)
    print(f"Building Connectivity Profile Cache")
    print(f"=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Max neurons: {max_neurons or 'all'}")
    print()
    
    # Create profiler with verbose output
    config = ProfilerConfig(
        top_k_bodyid=10,     # Store top 10 partners by weight
        top_m_type=5,        # Ensure at least 5 unique types via expansion
        expand_untyped_2hop=True,  # Fetch 2-hop for untyped partners
        use_cache=True,
        verbose=True
    )
    
    profiler = ConnectivityProfiler(config)
    
    # Progress callback for real-time updates
    def progress_callback(current, total, neuron_type):
        pct = 100 * current / total
        bar_len = 40
        filled = int(bar_len * current / total)
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f"\r[{bar}] {pct:5.1f}% ({current}/{total}) - {neuron_type[:30]:30s}", 
              end='', flush=True)
    
    # Build cache
    start_time = time.time()
    
    profiles = profiler.build_connectivity_profile_cache(
        dataset=dataset,
        neuron_types=None,        # All types in dataset
        top_k_bodyid=10,
        top_m_type=5,
        expand_untyped_2hop=True,
        force_refresh=False,      # Use existing cache if available
        max_neurons=max_neurons,
        progress_callback=progress_callback
    )
    
    elapsed = time.time() - start_time
    print()  # New line after progress bar
    
    # Summary
    print()
    print(f"=" * 60)
    print(f"Cache Build Complete")
    print(f"=" * 60)
    print(f"Profiles built: {len(profiles)}")
    print(f"Time elapsed: {elapsed:.1f} seconds")
    print(f"Rate: {len(profiles) / elapsed:.1f} profiles/second" if elapsed > 0 else "")
    
    # Sample profile analysis
    if profiles:
        sample_type = list(profiles.keys())[0]
        sample_profile = profiles[sample_type]
        
        print()
        print(f"Sample Profile: {sample_type}")
        print(f"-" * 40)
        print(f"  Upstream partners: {len(sample_profile.upstream_partners or {})}")
        print(f"  Downstream partners: {len(sample_profile.downstream_partners or {})}")
        print(f"  Unique types upstream: {sample_profile.unique_types_upstream}")
        print(f"  Unique types downstream: {sample_profile.unique_types_downstream}")
        print(f"  Untyped upstream count: {sample_profile.untyped_upstream_count}")
        print(f"  Untyped downstream count: {sample_profile.untyped_downstream_count}")
        
        if sample_profile.untyped_upstream_2hop:
            print(f"  2-hop upstream profiles: {len(sample_profile.untyped_upstream_2hop)}")
        if sample_profile.untyped_downstream_2hop:
            print(f"  2-hop downstream profiles: {len(sample_profile.untyped_downstream_2hop)}")
    
    # Cache location
    cache_path = Path("cache") / dataset.replace(':', '_').replace('.', '_') / 'connectivity_profiles.parquet'
    print()
    print(f"Cache location: {cache_path}")
    if cache_path.exists():
        import os
        size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print(f"Cache size: {size_mb:.2f} MB")


def read_cache_demo():
    """Demonstrate reading profiles from cache."""
    
    # Get dataset from args
    idx = sys.argv.index('--read')
    dataset = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else 'hemibrain_v1_2_1'
    
    print(f"=" * 60)
    print(f"Reading Connectivity Profile Cache")
    print(f"=" * 60)
    print(f"Dataset: {dataset}")
    print()
    
    profiler = ConnectivityProfiler(ProfilerConfig(verbose=True))
    
    # Read all profiles from cache
    start_time = time.time()
    profiles = profiler.read_connectivity_profile_cache(dataset)
    elapsed = time.time() - start_time
    
    print()
    print(f"Loaded {len(profiles)} profiles in {elapsed:.2f} seconds")
    
    if profiles:
        # Show sample
        sample_types = list(profiles.keys())[:5]
        print()
        print("Sample profiles:")
        for ntype in sample_types:
            p = profiles[ntype]
            print(f"  {ntype}: {len(p.upstream_partners or {})} up, {len(p.downstream_partners or {})} down")


def show_cache_stats():
    """Show statistics about cached profiles."""
    
    # Get dataset from args
    idx = sys.argv.index('--stats')
    dataset = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else 'hemibrain_v1_2_1'
    
    print(f"=" * 60)
    print(f"Connectivity Profile Cache Statistics")
    print(f"=" * 60)
    print(f"Dataset: {dataset}")
    print()
    
    profiler = ConnectivityProfiler(ProfilerConfig(verbose=False))
    stats = profiler.get_cache_stats(dataset)
    
    print(f"Total profiles: {stats['total_profiles']}")
    print(f"Total cache size: {stats['total_size_mb']} MB")
    print(f"Cache modified: {stats['cache_modified']}")
    
    if stats['top_k_distribution']:
        print()
        print("Top-k distribution:")
        for k, count in sorted(stats['top_k_distribution'].items()):
            print(f"  top_k={k}: {count} profiles")


def demo_homolog_finding():
    """Demonstrate homolog finding after cache is built."""
    
    print()
    print("=" * 60)
    print("Homolog Finding Demo")
    print("=" * 60)
    
    profiler = ConnectivityProfiler(ProfilerConfig(verbose=True))
    
    # Example: Find homologs for 'aMe12' in hemibrain -> FAFB
    query_type = 'aMe12'
    query_dataset = 'hemibrain_v1_2_1'
    target_dataset = 'flywire_FAFB_v783'
    
    print(f"\nFinding homologs for {query_type}...")
    print(f"  Query dataset: {query_dataset}")
    print(f"  Target dataset: {target_dataset}")
    
    # Loose matching (Jaccard similarity)
    print("\n1. Loose Matching (Jaccard Similarity):")
    print("-" * 40)
    
    try:
        results_loose = profiler.find_homologs_loose(
            query_type=query_type,
            query_dataset=query_dataset,
            target_dataset=target_dataset,
            direction='both',
            top_n=5
        )
        
        if not results_loose.empty:
            print(results_loose[['target_type', 'direction', 'jaccard_union', 
                                 'weighted_jaccard', 'common_partners']].to_string(index=False))
        else:
            print("  No matches found")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Strict matching (rank correlation + 2-hop)
    print("\n2. Strict Matching (Rank Correlation + 2-hop):")
    print("-" * 40)
    
    try:
        results_strict = profiler.find_homologs_strict(
            query_type=query_type,
            query_dataset=query_dataset,
            target_dataset=target_dataset,
            direction='both',
            top_n=5,
            min_common_partners=2
        )
        
        if not results_strict.empty:
            cols = ['target_type', 'direction', 'rank_correlation', 
                    'jaccard_typed', 'combined_score', 'common_partners']
            print(results_strict[[c for c in cols if c in results_strict.columns]].to_string(index=False))
        else:
            print("  No matches found")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == '__main__':
    main()
    
    # Optionally run homolog finding demo
    if '--demo' in sys.argv:
        demo_homolog_finding()
