#!/usr/bin/env python
"""
Example: Connectivity Profile-Based Homolog Finding

This example demonstrates how to use connectivity profiles to find potential
homologs of a neuron (by bodyId or type) across different connectome datasets.

The key insight is that homologous neurons (same cell type in different animals
or brain regions) tend to have similar connectivity patterns - they connect
to the same types of partners in similar proportions.

Usage:
    1. Initialize HomologFinder with configuration
    2. Call finder.find_homologs() with query neuron and target datasets
    3. Optionally get detailed partner overlap with finder.get_partner_overlap()

Features:
    - Input can be a bodyId (int) or type name (str)
    - Search across different datasets or within the same dataset
    - Find novel homologs (different names but similar connectivity)
    - Detailed comparison metrics and partner overlap
    - Single initialization, multiple queries

Author: Example script for hemibrain-connectomes-analysis
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.comparison import HomologFinder


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subsection(title: str):
    """Print a subsection header."""
    print("\n" + "-" * 50)
    print(f"  {title}")
    print("-" * 50)


def display_results(results: pd.DataFrame, title: str = "Results"):
    """Display homolog finding results."""
    if results.empty:
        print(f"\n{title}: No candidates found")
        return
    
    print(f"\n{title}:")
    print("  Rank  Type                 Similarity  Jaccard   Cosine    RankCorr  Match")
    print("  " + "-" * 75)
    
    for i, (_, row) in enumerate(results.iterrows(), 1):
        match = "✓" if row.get('is_same_type', False) else " "
        sim = row['similarity']
        sim_str = f"{sim:.3f}" if not pd.isna(sim) else "N/A  "
        
        # Get individual metrics if available
        jaccard = row.get('jaccard', None)
        cosine = row.get('cosine', None)
        rank_corr = row.get('rank_corr', None)
        
        jaccard_str = f"{jaccard:.3f}" if jaccard is not None and not pd.isna(jaccard) else "N/A  "
        cosine_str = f"{cosine:.3f}" if cosine is not None and not pd.isna(cosine) else "N/A  "
        rank_str = f"{rank_corr:.3f}" if rank_corr is not None and not pd.isna(rank_corr) else "N/A  "
        
        print(f"  {i:4d}  {row['target_type']:20s} {sim_str}     {jaccard_str}     {cosine_str}     {rank_str}     {match}")


def main():
    """Run homolog finding examples."""
    print("\n" + "=" * 70)
    print("  CONNECTIVITY PROFILE-BASED HOMOLOG FINDING")
    print("  Using HomologFinder for efficient multi-query searches")
    print("=" * 70)
    
    # =========================================================================
    # Initialize HomologFinder (no datasets required upfront)
    # =========================================================================
    print_section("Initializing HomologFinder")
    
    print("Creating finder with default settings...")
    print("Datasets are specified per query, not at initialization")
    
    # Create finder with default settings
    # - top_k: combined upstream+downstream partner count
    # - top_m: minimum unique partner types to ensure
    # - include_untyped_partners=True (default) to include untyped neurons
    finder = HomologFinder(
        top_k=5,
        top_m=3,
        min_synapse_threshold=3,
        include_untyped_partners=True,  # Include untyped neurons
        use_cache=True,
        verbose=True
    )
    
    # =========================================================================
    # Example 1: Fast Homolog Discovery (NEW - uses connection cache)
    # =========================================================================
    print_section("Example 1: Fast Homolog Discovery (Round 6)")
    print("Query: aMe12 from FAFB → fast search in FAFB")
    print("Uses pre-aggregated type dictionaries for O(1) lookups")
    
    results1 = finder.find_homologs_fast(
        query='aMe12',
        source_dataset='flywire_FAFB_v783',
        target_dataset='hemibrain:v1.2.1',
        top_n_candidates=50,
        min_shared_partners=2,
        min_weight=3,
        show_progress=True
    )
    
    display_results(results1, "Top matches (fast search)")
    
    if not results1.empty:
        top_match = results1.iloc[0]
        print(f"\nTop match: {top_match['target_type']} (similarity: {top_match['similarity']:.3f})")
        print(f"Shared partner count: {top_match['shared_partner_count']}")
    
    # =========================================================================
    # Example 2: Compare fast vs full search (commented - slow)
    # =========================================================================
    # print_section("Example 2: Compare Fast vs Full Search")
    # print("Running full search for comparison...")
    # 
    # results_full = finder.find_homologs(
    #     query='aMe12',
    #     source_dataset='flywire_FAFB_v783',
    #     target_datasets='flywire_FAFB_v783',
    #     top_n=10,
    #     metric='combined'
    # )
    # display_results(results_full, "Full search results")
    
    # =========================================================================
    # Example 3: Intra-dataset search for untyped neurons
    # =========================================================================
    # print_section("Example 3: Find Similar Typed Neurons")
    # print("Query: Mi1 → find similar typed neurons in FAFB")
    # 
    # results3 = finder.find_homologs_intra_dataset(
    #     query='Mi1',
    #     dataset='flywire_FAFB_v783',
    #     search_untyped=False,  # Search typed neurons
    #     top_n=10,
    #     min_weight=3
    # )
    # display_results(results3, "Similar typed neurons")
    
    # # =========================================================================
    # # Example 2: Multi-dataset search
    # # =========================================================================
    # print_section("Example 2: Multi-Dataset Search")
    # print("Query: Tm3 from hemibrain → search in FAFB and optic-lobe")
    
    # results2 = finder.find_homologs(
    #     query='Tm3',
    #     source_dataset='hemibrain_v1_2_1',
    #     target_datasets=['flywire_FAFB_v783', 'optic-lobe_v1_1'],
    #     top_n=5,
    #     metric='combined'
    # )
    
    # # Display by dataset
    # for dataset in ['flywire_FAFB_v783', 'optic-lobe_v1_1']:
    #     ds_results = results2[results2['target_dataset'] == dataset]
    #     display_results(ds_results, f"Top 5 in {dataset}")
    
    # # =========================================================================
    # # Example 3: Novel homolog discovery (same dataset)
    # # =========================================================================
    # print_section("Example 3: Novel Homolog Discovery (Same Dataset)")
    # print("Query: Mi1 from hemibrain → find similar types in hemibrain")
    # print("This identifies types with similar connectivity but different names")
    
    # # Use the convenience method
    # novel_results = finder.find_novel_homologs(
    #     query='Mi1',
    #     dataset='hemibrain_v1_2_1',
    #     top_n=10,
    #     min_score=0.3  # Only show matches with score > 0.3
    # )
    
    # display_results(novel_results, "Potential novel homologs")
    
    # # Highlight high-scoring matches
    # high_score = novel_results[novel_results['similarity'] > 0.5]
    # if not high_score.empty:
    #     print_subsection("High-Similarity Candidates (> 0.5)")
    #     for _, row in high_score.iterrows():
    #         print(f"\n  {row['target_type']}: similarity = {row['similarity']:.3f}")
            
    #         # Get partner overlap details
    #         overlap = finder.get_partner_overlap(
    #             query='Mi1',
    #             source_dataset='hemibrain_v1_2_1',
    #             target_type=row['target_type'],
    #             target_dataset='hemibrain_v1_2_1',
    #             direction='upstream'
    #         )
    #         if not overlap.empty:
    #             shared = overlap[overlap['status'] == 'shared']
    #             print(f"    Shared upstream partners: {len(shared)}/{len(overlap)}")
    #             if not shared.empty:
    #                 top_shared = shared.head(3)['partner_type'].tolist()
    #                 print(f"    Top shared: {', '.join(top_shared)}")
    
    # # =========================================================================
    # # Example 4: Using different metrics
    # # =========================================================================
    # print_section("Example 4: Different Similarity Metrics")
    # print("Comparing results with different metrics for Dm9")
    
    # for metric in ['combined', 'jaccard', 'rank']:
    #     results = finder.find_homologs(
    #         query='Dm9',
    #         source_dataset='hemibrain_v1_2_1',
    #         target_datasets='flywire_FAFB_v783',
    #         top_n=3,
    #         metric=metric
    #     )
    #     print(f"\nMetric: {metric}")
    #     if not results.empty:
    #         for _, row in results.iterrows():
    #             print(f"  {row['target_type']:20s} similarity: {row['similarity']:.3f}")
    
    # =========================================================================
    # Example 5: BodyId query (commented - needs real bodyId)
    # =========================================================================
    # print_section("Example 5: BodyId Query")
    # results = finder.find_homologs(
    #     query=12345678,  # Replace with real bodyId
    #     source_dataset='hemibrain_v1_2_1',
    #     target_datasets='flywire_FAFB_v783',
    #     top_n=10
    # )
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_section("Summary")
    print("\nHomologFinder provides efficient homolog discovery:")
    print("  • No upfront dataset initialization needed")
    print("  • Specify source_dataset and target_datasets per query")
    print("  • Profiles are cached for fast repeated access")
    print("\nKey methods:")
    print("  • finder.find_homologs() - Main search method")
    print("  • finder.find_novel_homologs() - Same-dataset discovery shortcut")
    print("  • finder.get_partner_overlap() - Detailed partner comparison")
    print("\nMetrics available:")
    print("  • 'combined' - Weighted average (default)")
    print("  • 'jaccard' - Partner set overlap")
    print("  • 'cosine' - Weight vector similarity")
    print("  • 'rank' - Partner ranking correlation")
    print("\nInterpretation:")
    print("  • Score > 0.7: Strong match - likely same cell type")
    print("  • Score 0.5-0.7: Moderate match - possibly related")
    print("  • Score < 0.5: Weak match - likely different types")


if __name__ == "__main__":
    main()
