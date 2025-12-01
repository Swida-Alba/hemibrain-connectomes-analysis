#!/usr/bin/env python3
"""
Build Connectivity Profile Cache

This script pre-builds connectivity profiles for all neuron types in a dataset.
Connectivity profiles are used for homolog finding and cross-dataset comparisons.

The profile cache stores for each neuron type:
- Upstream and downstream partner weights and ranks
- Unique types counts
- 2-hop expansion data for untyped partners
- Dynamic expansion metadata

Usage:
    python build_connectivity_profile_cache.py <dataset> [options]
    
    # Build cache for hemibrain
    python build_connectivity_profile_cache.py hemibrain_v1_2_1
    
    # Build cache with custom parameters
    python build_connectivity_profile_cache.py hemibrain_v1_2_1 --top-k 15 --top-m 8
    
    # Build cache using parallel processing
    python build_connectivity_profile_cache.py hemibrain_v1_2_1 --parallel --workers 8
    
    # Limit to specific number of neurons (for testing)
    python build_connectivity_profile_cache.py hemibrain_v1_2_1 --max-neurons 100
    
    # Read existing cache
    python build_connectivity_profile_cache.py --read hemibrain_v1_2_1
    
    # Show cache statistics
    python build_connectivity_profile_cache.py --stats hemibrain_v1_2_1

Options:
    --top-k N           Store top N partners by weight (default: 10)
    --top-m N           Ensure at least M unique types via expansion (default: 5)
    --expand-2hop       Enable 2-hop expansion for untyped partners (default: True)
    --no-expand-2hop    Disable 2-hop expansion
    --parallel          Use parallel processing
    --workers N         Number of parallel workers (default: CPU count)
    --max-neurons N     Limit to first N neurons (for testing)
    --force             Force rebuild even if cache exists
    --read              Read and display existing cache
    --stats             Show cache statistics
    -v, --verbose       Verbose output

Author: Hemibrain Analysis Team
Date: November 2024
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np


def get_project_paths() -> Dict[str, Path]:
    """Get standard project paths."""
    src_dir = Path(__file__).parent
    project_root = src_dir.parent
    return {
        'src': src_dir,
        'project_root': project_root,
        'datasets': project_root / 'datasets',
        'cache': project_root / 'cache',
    }


def sanitize_dataset_name(dataset: str) -> str:
    """Sanitize dataset name for filesystem use."""
    return dataset.replace(':', '_').replace('.', '_')


def get_cache_path(dataset: str, paths: Dict[str, Path]) -> Path:
    """Get the path to connectivity profiles cache."""
    safe_name = sanitize_dataset_name(dataset)
    return paths['cache'] / safe_name / 'connectivity_profiles.parquet'


def check_connection_cache(dataset: str, paths: Dict[str, Path]) -> bool:
    """Check if connection cache exists for the dataset."""
    safe_name = sanitize_dataset_name(dataset)
    dataset_path = paths['datasets'] / safe_name
    
    conn_files = [
        dataset_path / f'{safe_name}_merged_connections.parquet',
        dataset_path / f'{safe_name}_connections.parquet',
        dataset_path / 'connections.parquet',
    ]
    
    for f in conn_files:
        if f.exists():
            return True
    return False


def build_cache(
    dataset: str,
    top_k: int = 10,
    top_m: int = 5,
    expand_2hop: bool = True,
    parallel: bool = False,
    workers: Optional[int] = None,
    max_neurons: Optional[int] = None,
    force: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Build connectivity profile cache for a dataset.
    
    Uses the ConnectivityProfiler from the comparison module.
    """
    from comparison.connectivity_profiler import ConnectivityProfiler, ProfilerConfig
    
    paths = get_project_paths()
    cache_path = get_cache_path(dataset, paths)
    
    # Check if cache exists
    if cache_path.exists() and not force:
        print(f"[INFO] Cache already exists: {cache_path}")
        print(f"[INFO] Use --force to rebuild")
        return {}
    
    # Check if connection cache exists
    if not check_connection_cache(dataset, paths):
        print(f"[WARNING] Connection cache not found for {dataset}")
        print(f"[INFO] Run: python build_connection_cache.py {dataset}")
        print(f"[INFO] Continuing anyway - will try to use NeuPrint if available...")
    
    # Create profiler config
    config = ProfilerConfig(
        top_k_bodyid=top_k,
        top_m_type=top_m,
        expand_untyped_2hop=expand_2hop,
        use_cache=True,
        verbose=verbose
    )
    
    profiler = ConnectivityProfiler(config)
    
    # Progress callback
    def progress_callback(current: int, total: int, neuron_type: str):
        pct = 100 * current / total if total > 0 else 0
        bar_len = 40
        filled = int(bar_len * current / total) if total > 0 else 0
        bar = '█' * filled + '░' * (bar_len - filled)
        type_display = neuron_type[:30] if neuron_type else ''
        print(f"\r[{bar}] {pct:5.1f}% ({current}/{total}) - {type_display:30s}", 
              end='', flush=True)
    
    print(f"\n[INFO] Building connectivity profiles for {dataset}...")
    print(f"       Parameters: top_k={top_k}, top_m={top_m}, expand_2hop={expand_2hop}")
    if max_neurons:
        print(f"       Max neurons: {max_neurons}")
    print()
    
    start_time = time.time()
    
    # Build cache
    profiles = profiler.build_connectivity_profile_cache(
        dataset=dataset,
        neuron_types=None,  # All types
        top_k_bodyid=top_k,
        top_m_type=top_m,
        expand_untyped_2hop=expand_2hop,
        force_refresh=force,
        max_neurons=max_neurons,
        progress_callback=progress_callback
    )
    
    elapsed = time.time() - start_time
    print()  # New line after progress bar
    
    # Summary
    print(f"\n[OK] Built {len(profiles)} connectivity profiles in {elapsed:.1f} seconds")
    if elapsed > 0:
        print(f"     Rate: {len(profiles) / elapsed:.1f} profiles/second")
    
    # Cache location
    if cache_path.exists():
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        print(f"\n[OK] Cache saved: {cache_path}")
        print(f"     Size: {size_mb:.2f} MB")
    
    return profiles


def read_cache(dataset: str, limit: int = 10, verbose: bool = False) -> Dict[str, Any]:
    """Read and display profiles from cache."""
    from comparison.connectivity_profiler import ConnectivityProfiler, ProfilerConfig
    
    paths = get_project_paths()
    cache_path = get_cache_path(dataset, paths)
    
    if not cache_path.exists():
        print(f"[ERROR] Cache not found: {cache_path}")
        return {}
    
    profiler = ConnectivityProfiler(ProfilerConfig(verbose=verbose))
    
    print(f"\n[INFO] Reading cache from {cache_path}...")
    
    start_time = time.time()
    profiles = profiler.read_connectivity_profile_cache(dataset)
    elapsed = time.time() - start_time
    
    print(f"[OK] Loaded {len(profiles)} profiles in {elapsed:.2f} seconds")
    
    if profiles:
        # Show sample profiles
        sample_types = list(profiles.keys())[:limit]
        print(f"\nSample profiles (first {limit}):")
        print("-" * 60)
        
        for ntype in sample_types:
            p = profiles[ntype]
            up_count = len(p.upstream_partners or {})
            down_count = len(p.downstream_partners or {})
            print(f"  {ntype:30s}: {up_count:3d} upstream, {down_count:3d} downstream")
    
    return profiles


def show_stats(dataset: str) -> None:
    """Show statistics about the connectivity profile cache."""
    from comparison.connectivity_profiler import ConnectivityProfiler, ProfilerConfig
    
    paths = get_project_paths()
    cache_path = get_cache_path(dataset, paths)
    
    print(f"\n{'=' * 60}")
    print(f"Connectivity Profile Cache Statistics: {dataset}")
    print(f"{'=' * 60}")
    
    if not cache_path.exists():
        print(f"\n[ERROR] Cache not found: {cache_path}")
        print(f"        Run: python build_connectivity_profile_cache.py {dataset}")
        return
    
    # Basic file info
    print(f"\nCache file: {cache_path}")
    print(f"File size: {cache_path.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"Last modified: {time.ctime(cache_path.stat().st_mtime)}")
    
    # Load and analyze
    profiler = ConnectivityProfiler(ProfilerConfig(verbose=False))
    
    try:
        stats = profiler.get_cache_stats(dataset)
        
        print(f"\nProfile Statistics:")
        print(f"  Total profiles: {stats.get('total_profiles', 'N/A')}")
        print(f"  Cache size: {stats.get('total_size_mb', 'N/A')} MB")
        
        if stats.get('top_k_distribution'):
            print(f"\n  Top-k distribution:")
            for k, count in sorted(stats['top_k_distribution'].items()):
                print(f"    top_k={k}: {count} profiles")
    except Exception as e:
        print(f"\n[WARNING] Could not compute detailed stats: {e}")
    
    # Read profiles for detailed analysis
    try:
        profiles = profiler.read_connectivity_profile_cache(dataset)
        
        if profiles:
            # Compute additional statistics
            up_counts = []
            down_counts = []
            has_2hop = 0
            
            for p in profiles.values():
                up_counts.append(len(p.upstream_partners or {}))
                down_counts.append(len(p.downstream_partners or {}))
                if p.untyped_upstream_2hop or p.untyped_downstream_2hop:
                    has_2hop += 1
            
            print(f"\n  Partner Statistics:")
            print(f"    Upstream partners: avg={np.mean(up_counts):.1f}, max={max(up_counts)}")
            print(f"    Downstream partners: avg={np.mean(down_counts):.1f}, max={max(down_counts)}")
            print(f"    Profiles with 2-hop data: {has_2hop} ({100*has_2hop/len(profiles):.1f}%)")
            
            # Sample types
            print(f"\n  Sample types:")
            for i, ntype in enumerate(list(profiles.keys())[:5]):
                print(f"    {ntype}")
    except Exception as e:
        print(f"\n[WARNING] Could not read profiles for analysis: {e}")


def verify_cache(dataset: str) -> bool:
    """Verify cache integrity."""
    paths = get_project_paths()
    cache_path = get_cache_path(dataset, paths)
    
    print(f"\n[INFO] Verifying cache for {dataset}...")
    
    if not cache_path.exists():
        print(f"[FAIL] Cache not found: {cache_path}")
        return False
    
    # Try to read
    try:
        df = pd.read_parquet(cache_path)
    except Exception as e:
        print(f"[FAIL] Could not read cache: {e}")
        # Delete corrupt file
        cache_path.unlink(missing_ok=True)
        print(f"[INFO] Deleted corrupt cache file")
        return False
    
    # Verify structure
    if len(df) == 0:
        print("[FAIL] Cache is empty")
        return False
    
    # Check for required columns
    required = ['neuron_type']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        print(f"[FAIL] Missing columns: {missing}")
        return False
    
    print(f"[PASS] Cache verified: {len(df)} profiles")
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build connectivity profile cache for a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default parameters (set at top of file)
    python build_connectivity_profile_cache.py
    
    # Build cache for hemibrain
    python build_connectivity_profile_cache.py hemibrain_v1_2_1
    
    # Build cache with custom parameters
    python build_connectivity_profile_cache.py hemibrain_v1_2_1 --top-k 15 --top-m 8
    
    # Build cache using parallel processing
    python build_connectivity_profile_cache.py hemibrain_v1_2_1 --parallel --workers 8
    
    # Limit to first 100 neurons (for testing)
    python build_connectivity_profile_cache.py hemibrain_v1_2_1 --max-neurons 100
    
    # Read existing cache
    python build_connectivity_profile_cache.py --read hemibrain_v1_2_1
    
    # Show cache statistics  
    python build_connectivity_profile_cache.py --stats hemibrain_v1_2_1
"""
    )
    
    parser.add_argument('dataset', nargs='?', default=DEFAULT_DATASET,
                        help='Dataset name (e.g., hemibrain_v1_2_1, flywire_FAFB_v783)')
    parser.add_argument('--top-k', '-k', type=int, default=DEFAULT_TOP_K,
                        help='Store top N partners by weight (default: 10)')
    parser.add_argument('--top-m', '-m', type=int, default=DEFAULT_TOP_M,
                        help='Ensure at least M unique types via expansion (default: 5)')
    parser.add_argument('--expand-2hop', action='store_true', default=DEFAULT_EXPAND_2HOP,
                        help='Enable 2-hop expansion for untyped partners (default)')
    parser.add_argument('--no-expand-2hop', dest='expand_2hop', action='store_false',
                        help='Disable 2-hop expansion')
    parser.add_argument('--parallel', '-p', action='store_true', default=DEFAULT_PARALLEL,
                        help='Use parallel processing')
    parser.add_argument('--workers', '-w', type=int, default=DEFAULT_WORKERS,
                        help='Number of parallel workers')
    parser.add_argument('--max-neurons', '-n', type=int, default=DEFAULT_MAX_NEURONS,
                        help='Limit to first N neurons (for testing)')
    parser.add_argument('--force', '-f', action='store_true', default=DEFAULT_FORCE,
                        help='Force rebuild even if cache exists')
    parser.add_argument('--read', action='store_true',
                        help='Read and display existing cache')
    parser.add_argument('--stats', action='store_true',
                        help='Show cache statistics')
    parser.add_argument('--verify', action='store_true',
                        help='Verify cache integrity')
    parser.add_argument('--verbose', '-v', action='store_true', default=DEFAULT_VERBOSE,
                        help='Verbose output')
    
    args = parser.parse_args()
    
    dataset = args.dataset
    
    print(f"\n{'=' * 60}")
    print(f"Connectivity Profile Cache Builder")
    print(f"{'=' * 60}")
    print(f"Dataset: {dataset}")
    
    # Stats mode
    if args.stats:
        show_stats(dataset)
        return
    
    # Read mode
    if args.read:
        read_cache(dataset, verbose=args.verbose)
        return
    
    # Verify mode
    if args.verify:
        success = verify_cache(dataset)
        sys.exit(0 if success else 1)
    
    # Build mode
    profiles = build_cache(
        dataset=dataset,
        top_k=args.top_k,
        top_m=args.top_m,
        expand_2hop=args.expand_2hop,
        parallel=args.parallel,
        workers=args.workers,
        max_neurons=args.max_neurons,
        force=args.force,
        verbose=args.verbose
    )
    
    # Show sample profile
    if profiles:
        sample_type = list(profiles.keys())[0]
        sample_profile = profiles[sample_type]
        
        print(f"\nSample Profile: {sample_type}")
        print("-" * 40)
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


if __name__ == '__main__':
    main()
