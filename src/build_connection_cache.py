#!/usr/bin/env python3
"""
Build Connection Cache

This script pre-builds the connection cache for a dataset by downloading
and caching all connection data (synaptic connections between neurons).

The connection cache stores:
- Pre-synaptic and post-synaptic bodyIds
- Connection weights (synapse counts)
- Neuron type information (type_pre, type_post)

Usage:
    python build_connection_cache.py <dataset> [options]
    
    # Build cache for hemibrain
    python build_connection_cache.py hemibrain_v1_2_1 --token YOUR_TOKEN
    
    # Build cache for FAFB (local files)
    python build_connection_cache.py flywire_FAFB_v783
    
    # Show cache statistics
    python build_connection_cache.py --stats hemibrain_v1_2_1

Options:
    --token TOKEN       NeuPrint API token (required for hemibrain/optic-lobe)
    --server URL        NeuPrint server URL (default: neuprint.janelia.org)
    --force             Force rebuild even if cache exists
    --batch-size N      Batch size for API calls (default: 100)
    --stats             Show cache statistics for dataset
    --verify            Verify cache integrity after build
    -v, --verbose       Verbose output

Author: Hemibrain Analysis Team
Date: November 2024
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
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


def is_local_dataset(dataset: str) -> bool:
    """Check if dataset is local (FlyWire/FAFB/BANC) vs NeuPrint."""
    dataset_lower = dataset.lower()
    return any(x in dataset_lower for x in ['flywire', 'fafb', 'banc'])


def load_local_connections(dataset: str, paths: Dict[str, Path], verbose: bool = False) -> Optional[pd.DataFrame]:
    """
    Load connection data from local dataset files.
    
    Checks for existing connection files in the datasets folder.
    """
    safe_name = sanitize_dataset_name(dataset)
    dataset_path = paths['datasets'] / safe_name
    
    if not dataset_path.exists():
        print(f"[ERROR] Dataset folder not found: {dataset_path}")
        return None
    
    # Try to load connections file - check multiple naming conventions
    conn_files = [
        dataset_path / f'{safe_name}_merged_connections.parquet',
        dataset_path / f'{safe_name}_merged_connections.csv',
        dataset_path / f'{safe_name}_connections.parquet',
        dataset_path / f'{safe_name}_connections.csv',
        dataset_path / 'connections.parquet',
        dataset_path / 'connections.csv',
    ]
    
    for conn_file in conn_files:
        if conn_file.exists():
            try:
                if verbose:
                    print(f"[INFO] Loading connections from {conn_file.name}...")
                
                if str(conn_file).endswith('.parquet'):
                    conn_df = pd.read_parquet(conn_file)
                else:
                    conn_df = pd.read_csv(conn_file)
                
                print(f"[OK] Loaded {len(conn_df):,} connections from {conn_file.name}")
                return conn_df
            except Exception as e:
                print(f"[WARNING] Could not load {conn_file.name}: {e}")
    
    print(f"[ERROR] No connection file found in {dataset_path}")
    return None


def load_neuron_df(dataset: str, paths: Dict[str, Path], verbose: bool = False) -> Optional[pd.DataFrame]:
    """Load neuron DataFrame with type information."""
    safe_name = sanitize_dataset_name(dataset)
    dataset_path = paths['datasets'] / safe_name
    
    if not dataset_path.exists():
        return None
    
    neuron_files = [
        dataset_path / f'{safe_name}_allneurons_neuron_df.parquet',
        dataset_path / f'{safe_name}_allneurons_neuron_df.csv',
        dataset_path / f'{safe_name}_neuron_df.parquet',
        dataset_path / f'{safe_name}_neuron_df.csv',
        dataset_path / 'neuron_df.parquet',
        dataset_path / 'neuron_df.csv',
    ]
    
    for neuron_file in neuron_files:
        if neuron_file.exists():
            try:
                if verbose:
                    print(f"[INFO] Loading neuron info from {neuron_file.name}...")
                
                if str(neuron_file).endswith('.parquet'):
                    neuron_df = pd.read_parquet(neuron_file)
                else:
                    neuron_df = pd.read_csv(neuron_file)
                
                print(f"[OK] Loaded {len(neuron_df):,} neurons from {neuron_file.name}")
                return neuron_df
            except Exception as e:
                print(f"[WARNING] Could not load {neuron_file.name}: {e}")
    
    return None


def standardize_columns(conn_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize connection DataFrame column names."""
    col_mapping = {
        'pre_pt_root_id': 'bodyId_pre',
        'post_pt_root_id': 'bodyId_post',
        'pre_type': 'type_pre',
        'post_type': 'type_post',
        'syn_count': 'weight',
    }
    conn_df = conn_df.rename(columns={k: v for k, v in col_mapping.items() if k in conn_df.columns})
    
    # Ensure weight column exists
    if 'weight' not in conn_df.columns:
        if 'syn_count' in conn_df.columns:
            conn_df['weight'] = conn_df['syn_count']
        else:
            conn_df['weight'] = 1
    
    return conn_df


def enrich_with_type_info(conn_df: pd.DataFrame, neuron_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Add type information to connection DataFrame if missing.
    
    Joins type info from neuron_df based on bodyId.
    """
    if 'type_pre' in conn_df.columns and 'type_post' in conn_df.columns:
        # Check if types are actually populated
        has_pre = conn_df['type_pre'].notna().any()
        has_post = conn_df['type_post'].notna().any()
        if has_pre and has_post:
            return conn_df
    
    if neuron_df is None or neuron_df.empty:
        print("[WARNING] No neuron_df available for type enrichment")
        return conn_df
    
    if verbose:
        print("[INFO] Enriching connections with type information...")
    
    # Find bodyId and type columns
    bodyid_col = None
    type_col = None
    
    for col in ['bodyId', 'body_id', 'pt_root_id', 'root_id']:
        if col in neuron_df.columns:
            bodyid_col = col
            break
    
    for col in ['type', 'Type', 'cellType', 'cell_type']:
        if col in neuron_df.columns:
            type_col = col
            break
    
    if bodyid_col is None or type_col is None:
        print("[WARNING] Cannot find bodyId or type columns in neuron_df")
        return conn_df
    
    # Create lookup table
    type_lookup = neuron_df[[bodyid_col, type_col]].copy()
    type_lookup.columns = ['bodyId', 'type']
    type_lookup = type_lookup.drop_duplicates(subset='bodyId')
    
    # Join for pre-synaptic
    if 'bodyId_pre' in conn_df.columns:
        pre_lookup = type_lookup.rename(columns={'bodyId': 'bodyId_pre', 'type': 'type_pre'})
        conn_df = conn_df.merge(pre_lookup, on='bodyId_pre', how='left', suffixes=('', '_new'))
        if 'type_pre_new' in conn_df.columns:
            conn_df['type_pre'] = conn_df['type_pre_new'].fillna(conn_df.get('type_pre', ''))
            conn_df = conn_df.drop(columns=['type_pre_new'])
    
    # Join for post-synaptic
    if 'bodyId_post' in conn_df.columns:
        post_lookup = type_lookup.rename(columns={'bodyId': 'bodyId_post', 'type': 'type_post'})
        conn_df = conn_df.merge(post_lookup, on='bodyId_post', how='left', suffixes=('', '_new'))
        if 'type_post_new' in conn_df.columns:
            conn_df['type_post'] = conn_df['type_post_new'].fillna(conn_df.get('type_post', ''))
            conn_df = conn_df.drop(columns=['type_post_new'])
    
    # Count enriched types
    typed_pre = conn_df['type_pre'].notna().sum() if 'type_pre' in conn_df.columns else 0
    typed_post = conn_df['type_post'].notna().sum() if 'type_post' in conn_df.columns else 0
    print(f"[OK] Enriched types: {typed_pre:,} pre-synaptic, {typed_post:,} post-synaptic")
    
    return conn_df


def fetch_neuprint_connections(
    dataset: str,
    token: str,
    server: str = "neuprint.janelia.org",
    batch_size: int = 100,
    verbose: bool = False
) -> Optional[pd.DataFrame]:
    """
    Fetch all connections from NeuPrint API.
    
    This fetches connections in batches to avoid timeouts.
    """
    try:
        from neuprint import Client, set_default_client, fetch_neurons, fetch_adjacencies
    except ImportError:
        print("[ERROR] neuprint-python not installed. Run: pip install neuprint-python")
        return None
    
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, **kwargs):
            return iterable
    
    print(f"[INFO] Connecting to NeuPrint server: {server}")
    
    try:
        client = Client(server, dataset, token)
        set_default_client(client)
        print(f"[OK] Connected to {dataset}")
    except Exception as e:
        print(f"[ERROR] Failed to connect to NeuPrint: {e}")
        return None
    
    # First, get all neurons to get their bodyIds
    print("[INFO] Fetching neuron list...")
    try:
        neuron_df, _ = fetch_neurons(None)
        print(f"[OK] Found {len(neuron_df):,} neurons")
    except Exception as e:
        print(f"[ERROR] Failed to fetch neurons: {e}")
        return None
    
    # Get all bodyIds
    all_bodyIds = neuron_df['bodyId'].tolist()
    
    # Fetch connections in batches
    print(f"[INFO] Fetching connections for {len(all_bodyIds):,} neurons in batches of {batch_size}...")
    
    batches = [all_bodyIds[i:i + batch_size] for i in range(0, len(all_bodyIds), batch_size)]
    all_connections = []
    
    for batch in tqdm(batches, desc="Fetching connections"):
        try:
            _, roi_conn_df = fetch_adjacencies(sources=batch, min_total_weight=1)
            if not roi_conn_df.empty:
                all_connections.append(roi_conn_df)
        except Exception as e:
            if verbose:
                print(f"\n[WARNING] Batch failed: {e}")
    
    if not all_connections:
        print("[ERROR] No connections fetched")
        return None
    
    conn_df = pd.concat(all_connections, ignore_index=True)
    print(f"\n[OK] Fetched {len(conn_df):,} connections")
    
    # Enrich with type info
    conn_df = enrich_with_type_info(conn_df, neuron_df, verbose)
    
    return conn_df


def save_connection_cache(
    conn_df: pd.DataFrame,
    dataset: str,
    paths: Dict[str, Path],
    verbose: bool = False
) -> Path:
    """Save connection cache to parquet file."""
    safe_name = sanitize_dataset_name(dataset)
    dataset_path = paths['datasets'] / safe_name
    
    # Create directory if needed
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet
    cache_file = dataset_path / f'{safe_name}_merged_connections.parquet'
    
    # Ensure required columns
    required_cols = ['bodyId_pre', 'bodyId_post', 'weight']
    for col in required_cols:
        if col not in conn_df.columns:
            print(f"[WARNING] Missing required column: {col}")
    
    if verbose:
        print(f"[INFO] Saving cache to {cache_file}...")
    
    conn_df.to_parquet(cache_file, index=False)
    
    # Get file size
    size_mb = cache_file.stat().st_size / (1024 * 1024)
    print(f"[OK] Saved connection cache: {cache_file.name} ({size_mb:.2f} MB)")
    
    return cache_file


def show_cache_stats(dataset: str, paths: Dict[str, Path]) -> None:
    """Show statistics about the connection cache."""
    safe_name = sanitize_dataset_name(dataset)
    dataset_path = paths['datasets'] / safe_name
    
    print(f"\n{'=' * 60}")
    print(f"Connection Cache Statistics: {dataset}")
    print(f"{'=' * 60}")
    
    if not dataset_path.exists():
        print(f"[ERROR] Dataset folder not found: {dataset_path}")
        return
    
    # Find connection file
    conn_files = [
        dataset_path / f'{safe_name}_merged_connections.parquet',
        dataset_path / f'{safe_name}_connections.parquet',
        dataset_path / 'connections.parquet',
    ]
    
    conn_file = None
    for f in conn_files:
        if f.exists():
            conn_file = f
            break
    
    if conn_file is None:
        print("[ERROR] No connection cache found")
        return
    
    # Load and analyze
    try:
        conn_df = pd.read_parquet(conn_file)
    except Exception as e:
        print(f"[ERROR] Could not read cache: {e}")
        return
    
    print(f"\nCache file: {conn_file.name}")
    print(f"File size: {conn_file.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"Last modified: {time.ctime(conn_file.stat().st_mtime)}")
    
    print(f"\nConnection Statistics:")
    print(f"  Total connections: {len(conn_df):,}")
    
    if 'bodyId_pre' in conn_df.columns:
        print(f"  Unique pre-synaptic neurons: {conn_df['bodyId_pre'].nunique():,}")
    if 'bodyId_post' in conn_df.columns:
        print(f"  Unique post-synaptic neurons: {conn_df['bodyId_post'].nunique():,}")
    
    if 'weight' in conn_df.columns:
        print(f"  Total synapse count: {conn_df['weight'].sum():,}")
        print(f"  Weight range: {conn_df['weight'].min()} - {conn_df['weight'].max()}")
        print(f"  Mean weight: {conn_df['weight'].mean():.2f}")
    
    if 'type_pre' in conn_df.columns:
        typed_pre = conn_df['type_pre'].notna().sum()
        print(f"  Connections with type_pre: {typed_pre:,} ({100*typed_pre/len(conn_df):.1f}%)")
        print(f"  Unique pre-types: {conn_df['type_pre'].dropna().nunique():,}")
    
    if 'type_post' in conn_df.columns:
        typed_post = conn_df['type_post'].notna().sum()
        print(f"  Connections with type_post: {typed_post:,} ({100*typed_post/len(conn_df):.1f}%)")
        print(f"  Unique post-types: {conn_df['type_post'].dropna().nunique():,}")
    
    print(f"\nColumns: {', '.join(conn_df.columns)}")


def verify_cache(dataset: str, paths: Dict[str, Path]) -> bool:
    """Verify cache integrity."""
    safe_name = sanitize_dataset_name(dataset)
    dataset_path = paths['datasets'] / safe_name
    
    print(f"\n[INFO] Verifying cache for {dataset}...")
    
    # Find connection file
    conn_files = [
        dataset_path / f'{safe_name}_merged_connections.parquet',
        dataset_path / f'{safe_name}_connections.parquet',
    ]
    
    conn_file = None
    for f in conn_files:
        if f.exists():
            conn_file = f
            break
    
    if conn_file is None:
        print("[FAIL] No connection cache found")
        return False
    
    # Try to read
    try:
        conn_df = pd.read_parquet(conn_file)
    except Exception as e:
        print(f"[FAIL] Could not read cache: {e}")
        # Delete corrupt file
        conn_file.unlink(missing_ok=True)
        print(f"[INFO] Deleted corrupt cache file")
        return False
    
    # Verify required columns
    required = ['bodyId_pre', 'bodyId_post', 'weight']
    missing = [col for col in required if col not in conn_df.columns]
    
    if missing:
        print(f"[FAIL] Missing columns: {missing}")
        return False
    
    # Verify data
    if len(conn_df) == 0:
        print("[FAIL] Cache is empty")
        return False
    
    print(f"[PASS] Cache verified: {len(conn_df):,} connections")
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build connection cache for a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default parameters (set at top of file)
    python build_connection_cache.py
    
    # Build cache for hemibrain (requires token)
    python build_connection_cache.py hemibrain_v1_2_1 --token YOUR_TOKEN
    
    # Build cache for FAFB (uses local files)
    python build_connection_cache.py flywire_FAFB_v783
    
    # Show cache statistics
    python build_connection_cache.py --stats hemibrain_v1_2_1
    
    # Force rebuild cache
    python build_connection_cache.py hemibrain_v1_2_1 --token YOUR_TOKEN --force
"""
    )
    
    parser.add_argument('dataset', nargs='?', default=DEFAULT_DATASET,
                        help='Dataset name (e.g., hemibrain_v1_2_1, flywire_FAFB_v783)')
    parser.add_argument('--token', '-t', type=str, default=DEFAULT_TOKEN,
                        help='NeuPrint API token')
    parser.add_argument('--server', '-s', type=str, default=DEFAULT_SERVER,
                        help='NeuPrint server URL')
    parser.add_argument('--batch-size', '-b', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Batch size for API calls')
    parser.add_argument('--force', '-f', action='store_true', default=DEFAULT_FORCE,
                        help='Force rebuild even if cache exists')
    parser.add_argument('--stats', action='store_true',
                        help='Show cache statistics')
    parser.add_argument('--verify', action='store_true',
                        help='Verify cache integrity')
    parser.add_argument('--verbose', '-v', action='store_true', default=DEFAULT_VERBOSE,
                        help='Verbose output')
    
    args = parser.parse_args()
    
    paths = get_project_paths()
    
    # Stats mode
    if args.stats:
        dataset = args.dataset or 'hemibrain_v1_2_1'
        show_cache_stats(dataset, paths)
        return
    
    # Verify mode
    if args.verify:
        success = verify_cache(args.dataset, paths)
        sys.exit(0 if success else 1)
    
    dataset = args.dataset
    safe_name = sanitize_dataset_name(dataset)
    
    print(f"\n{'=' * 60}")
    print(f"Building Connection Cache")
    print(f"{'=' * 60}")
    print(f"Dataset: {dataset}")
    print(f"Output: datasets/{safe_name}/")
    print()
    
    start_time = time.time()
    
    # Check if cache already exists
    dataset_path = paths['datasets'] / safe_name
    cache_file = dataset_path / f'{safe_name}_merged_connections.parquet'
    
    if cache_file.exists() and not args.force:
        print(f"[INFO] Cache already exists: {cache_file.name}")
        print(f"[INFO] Use --force to rebuild")
        show_cache_stats(dataset, paths)
        return
    
    # Build cache based on dataset type
    if is_local_dataset(dataset):
        # Local dataset (FlyWire/FAFB/BANC)
        conn_df = load_local_connections(dataset, paths, args.verbose)
        
        if conn_df is not None:
            # Standardize columns
            conn_df = standardize_columns(conn_df)
            
            # Load neuron_df for type enrichment
            neuron_df = load_neuron_df(dataset, paths, args.verbose)
            
            # Enrich with types if needed
            if neuron_df is not None:
                conn_df = enrich_with_type_info(conn_df, neuron_df, args.verbose)
            
            # Save standardized cache
            save_connection_cache(conn_df, dataset, paths, args.verbose)
    else:
        # NeuPrint dataset
        if args.token is None:
            print("[ERROR] Token required for NeuPrint datasets")
            print("       Use --token YOUR_TOKEN or set NEUPRINT_TOKEN environment variable")
            
            # Try environment variable
            import os
            token = os.environ.get('NEUPRINT_TOKEN')
            if token:
                print("[INFO] Using token from NEUPRINT_TOKEN environment variable")
                args.token = token
            else:
                sys.exit(1)
        
        conn_df = fetch_neuprint_connections(
            dataset=dataset,
            token=args.token,
            server=args.server,
            batch_size=args.batch_size,
            verbose=args.verbose
        )
        
        if conn_df is not None:
            conn_df = standardize_columns(conn_df)
            save_connection_cache(conn_df, dataset, paths, args.verbose)
    
    elapsed = time.time() - start_time
    print(f"\n[DONE] Build completed in {elapsed:.1f} seconds")
    
    # Verify
    if args.verbose:
        verify_cache(dataset, paths)


if __name__ == '__main__':
    main()
