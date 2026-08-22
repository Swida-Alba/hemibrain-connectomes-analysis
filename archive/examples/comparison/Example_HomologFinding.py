#!/usr/bin/env python
"""
Example: Connectivity Profile-Based Homolog Finding

This example demonstrates how to use the HomologFinder module to find potential
homologs of neurons across different connectome datasets.

Key Features:
    - Module-level defaults: Set source, source_dataset, target_dataset once
    - Fast search: Adjacency expansion for efficient candidate discovery
    - 1-hop/2-hop hybrid: Uses ConnectivityProfiler for 2-hop expansion of untyped
    - Automatic saving: Both bodyId-level and type-level results always saved
    - Skeleton visualization: Optionally visualize top candidates with VisualizeSkeleton

Profile Construction Rules (consistent with ConnectivityProfiler):
    - top_k: Top K partners per direction by synapse weight (default: 15)
    - top_m: Minimum unique partner types to ensure (default: 5)
    - Dynamic expansion: If top_k yields < top_m types, expand K
    - expand_untyped_2hop: Fetch 2-hop typed partners for untyped 1-hop (default: True)

Output Files (always saved when output_dir is set):
    - bodyid_results.csv: BodyId-level comparisons (sorted by source_bodyId, rank_corr)
    - type_summary.csv: Type-level aggregated summary (avg/best/std metrics)
    - homolog_results.csv: Legacy format (sorted by rank_corr only)
    - visualizations/: Skeleton visualizations (if visualize_skeleton=True)

Finding Methods:
    - find_homologs(): Comprehensive search (builds all target profiles)
    - find_homologs_fast(): Fast search via adjacency expansion

Author: Example script for drocat
"""

import sys
from pathlib import Path

# Add repo src/ to path to force loading the in-repo comparison module (avoids picking up any older installed version)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from comparison.profile_comparator import HomologFinder


def main():
    """Run homolog finding examples."""
    print("\n" + "=" * 70)
    print("  CONNECTIVITY PROFILE-BASED HOMOLOG FINDING")
    print("  Using HomologFinder with module-level defaults")
    print("=" * 70)
    
    finder = HomologFinder(
        token='',

        source='aMe12',
        source_dataset='male-cns:v0.9',
        target_dataset='hemibrain:v1.2.1',
        # Alternative: search across different datasets
        # source='aMe12',
        # source_dataset='flywire_FAFB_v783',
        # target_dataset='male-cns:v0.9',
        
        output_dir='../../local_data/homolog_finding',
        visualize_skeleton=True,  # Enable to visualize top candidates
        visualize_top_n=5,         # Number of candidates to visualize
        verbose=True,
        similarity_metric='rank_union',
        top_n=30,
        vector_prefiltering=True,
    )
    
    # Run using defaults - no arguments needed!
    results1 = finder.find_homologs_fast()
    # results2 = finder.find_homologs()
    
    # display_results(results1, "Top matches for aMe12")
    
    if not results1.empty:
        top_match = results1.iloc[0]
        print(f"\nTop match: {top_match['target_type']} (rank_corr: {top_match['rank_corr']:.3f})")
    
    
    print("\nFolder Structure (when output_dir is set):")
    print("  {saveas}/")
    print("  ├── README.txt              # Parameters and summary")
    print("  ├── results/")
    print("  │   ├── bodyid_results.csv  # BodyId-level (sorted by source, rank_corr)")
    print("  │   ├── type_summary.csv    # Type-level aggregated")
    print("  │   └── homolog_results.csv # Legacy format")
    print("  ├── profiles/")
    print("  │   ├── query/              # Query neuron profile")
    print("  │   └── matches/            # Top match profiles")
    print("  └── overlaps/               # Partner overlap details")


if __name__ == "__main__":
    main()
