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

Author: Example script for hemibrain-connectomes-analysis
"""

import sys
from pathlib import Path

# Add repo src/ to path to force loading the in-repo comparison module (avoids picking up any older installed version)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
from comparison.profile_comparator import HomologFinder

if __name__ == "__main__":
    # dataset = 'male-cns:v0.9'
    finder = HomologFinder(
        token='',

        source='GNG458', # MTe07/MeVPLo2
        # source_dataset=dataset,
        # target_dataset=dataset,
        # source='aMe12',
        source_dataset='male-cns:v0.9',
        target_dataset='flywire_FAFB_v783',
        
        output_dir='/Users/apple/Local/connection_data/HomologFinding/',
        visualize_skeleton=True,  # Enable to visualize top candidates
        visualize_top_n=5,         # Number of candidates to visualize
        verbose=True,
        similarity_metric='jaccard',
        top_n=30,
        vector_prefiltering=True,
    )
    
    # Run using defaults - no arguments needed!
    results1 = finder.find_homologs_fast()
    # results2 = finder.direct_comparison(
    #     neurons_a='aMe12',
    #     neurons_b='aMe12',
    #     dataset_a='flywire_FAFB_v783',
    #     dataset_b='male-cns:v0.9',
    #     )
    # results2 = finder.find_homologs()
