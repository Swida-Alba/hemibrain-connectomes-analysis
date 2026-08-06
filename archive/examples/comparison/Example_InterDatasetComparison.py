"""
Example: Cross-Dataset Comparison Analysis

This example demonstrates the complete workflow for comparing connectivity 
across multiple datasets using the comparison module.

Features demonstrated:
- ComparisonParameters as primary entry point
- Two-dataset comparison (hemibrain vs male-cns)
- Intermediate layer analysis (max_interlayer=1)
- Automatic export of all comparison results
- Interactive HTML report with Cytoscape.js network visualization
- Automatic matplotlib visualizations
- PDF export capability
- Comparison modes: 'path' (path-based) vs 'edge' (edge-based)
- **NEW**: Direct comparison of specific neurons (direct_comparison)
- **NEW**: Connectivity profile comparison as SEPARATE step (connectivity_profile_comparison)


Comparison Modes:
  - 'path' (default): Edges are discovered through path traversal. May miss 
    strong edges if they're on paths with weak intermediate connections.
  - 'edge': Edges are evaluated independently by weight. Provides direct 
    comparison of synaptic connectivity but loses path-level circuit structure.

Direct Comparison:
  - Compares specific neurons by type name or bodyId
  - Uses ProfileComparator.direct_comparison() under the hood
  - Supports LabelMapper for cross-dataset type name resolution
  - Example: analyzer.direct_comparison('aMe12', 'aMe12')

Connectivity Profile Comparison:
  - Compares all neuron types from comparison results across datasets
  - Uses multiple similarity metrics (Jaccard, Cosine, Rank correlation)
  - Generates confidence scores for type assignments
  - Supports two comparison modes:
    * 'loose' (default): Type-aggregated profiles - faster, compares overall
      connectivity patterns by aggregating all neurons of the same type
    * 'strict': Per-bodyId profiles - more precise, computes rank correlation
      on individual neuron pairs with optional 2-hop expansion
  
  **IMPORTANT**: Both methods are now SEPARATE from export_results().
  Parameters can be configured in ComparisonParameters or passed directly.
"""

import os
import sys
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from comparison import (
    ComparisonAnalyzer,
    ComparisonParameters,
    ComparisonVisualizer,
    LabelMapper,
)


def run_comprehensive_comparison():
    """
    Comprehensive comparison example demonstrating all features.
    
    This example:
    1. Compares hemibrain and male-cns datasets
    2. Analyzes aMe12 → PPL101 connectivity with 1 intermediate layer
    3. Tests multiple thresholds (1, 3, 5, 10)
    4. Exports all comparison results
    5. Generates interactive HTML report with Cytoscape.js network
    6. Generates matplotlib visualizations
    7. Runs connectivity profile verification (optional, separate step)
    """
    
    # =========================================================================
    # Step 1: Create ComparisonParameters
    # =========================================================================
    print("\n📋 Step 1: Creating ComparisonParameters...")
    
    target_map = LabelMapper(
      target_mapping_dict={
        'flywire_FAFB_v783': [[720575940619067259,720575940613413791,720575940631973089,720575940642237344,720575940606868828,720575940609174392,720575940610478531,720575940628527095]],
        'flywire_BANC_v626': [[720575941552713626,720575941589129982,720575941645302945,720575941689244824]],
        'male-cns:v0.9': [[15832,16461,16552,17336,16634,17355,17916,18945]],
      },
      target_labels=['s-LNv'],
    )
    
    params = ComparisonParameters(
        # Token - empty means load from NEUPRINT_APPLICATION_TOKEN env var
        token='',
        # Output settings
        output_folder='../../local_data/dataset_comparison',
        saveas=None,  # Auto-generate timestamp folder
        
        # Datasets to compare
        datasets=['male-cns:v0.9', 'flywire_FAFB_v783'],
        datasets_nickname=['male-CNS', 'FAFB'],
        # Alternative: compare more datasets
        # datasets=['male-cns:v0.9', 'flywire_FAFB_v783', 'hemibrain:v1.2.1'],
        # datasets_nickname=['male-CNS', 'FAFB', 'hemibrain'],
        
        # Source neurons - example with specific types
        source_neurons=['aMe12'],
        # Or use target_map for unified naming:
        # source_neurons=target_map,
        
        # Target neurons
        target_neurons=['PPL101'],
        # Or use target_map for unified naming:
        # target_neurons=target_map,
        
        # Allow intermediate layers (source → inter → target)
        max_interlayer=2,
        
        # Multiple thresholds to analyze sensitivity
        thresholds=[3, 5, 10],
        
        # Top edges to include in analysis
        top_edges=500,
        
        # Comparison mode: 'path' (path-based filtering) or 'edge' (edge-based filtering)
        # - 'path': Discovers edges through paths; may miss strong edges on weak paths
        # - 'edge': Evaluates each edge independently by weight
        comparison_mode='path',  # Change to 'edge' to use edge-based comparison
        
        skip_bodyId=True,
        
        # Performance Settings (for parallel processing)
        # -----------------------------------------------------------------
        parallel=True,                # Enable parallel processing
        max_workers=12,               # Auto-detect optimal worker count
        pathfinding='MemoizedDFS',  # 'MemoizedDFS' (default, fastest), 'DFS' (backward), 'MeetInMiddle', 'DP', 'Bidirectional'
    )
    
    # =========================================================================
    # Step 2: Create Analyzer and Run Comparison
    # =========================================================================
    print("\n🔬 Step 2: Running comparison analysis...")
    
    analyzer = ComparisonAnalyzer(params, verbose=True)
    results = analyzer.run_comparison()
    
    # =========================================================================
    # Step 3: Generate Text Report
    # =========================================================================
    print("\n📝 Step 3: Generating text report...")
    
    analyzer.generate_report()
    
    # =========================================================================
    # Step 4: Export All Results
    # =========================================================================
    print("\n💾 Step 4: Exporting all results...")
    
    # NOTE: export_results() no longer runs connectivity profile verification.
    # Call run_connectivity_profile_verification() separately in Step 6 if needed.
    analyzer.export_results()
    
    # =========================================================================
    # Step 5: Additional Analysis with ComparisonVisualizer
    # =========================================================================
    print("\n📈 Step 5: Additional visualization analysis...")
    
    try:
        viz = ComparisonVisualizer(verbose=True)
        print('   ComparisonVisualizer ready for custom analysis')
        
        # Example: Generate a stacked path count plot (not included in standard set)
        fig_stacked = viz.plot_path_counts_stacked(
            results=analyzer.raw_results,
            thresholds=params.thresholds,
            title="Path Counts by Threshold (Stacked)"
        )
        stacked_path = os.path.join(params.full_output_path, 'comparison_visualizations', 'path_counts_stacked.png')
        fig_stacked.savefig(stacked_path)
        print(f"   Saved custom plot: {stacked_path}")
        
    except Exception as e:
        print(f"   Visualization analysis skipped: {e}")
    
    # =========================================================================
    # Step 6: Direct Comparison (NEW - specific neurons)
    # =========================================================================
    print("\n🎯 Step 6: Running direct comparison of specific neurons...")
    
    # Direct comparison allows comparing specific neurons by type name or bodyId
    try:
        # Compare specific types across all datasets
        direct_results = analyzer.direct_comparison(
            neurons_a=['aMe12'],  # Source neurons
            neurons_b=['aMe12'],  # Target neurons (same type for verification)
        )
        
        if direct_results and not direct_results['results'].empty:
            print(f"   Direct comparison: {len(direct_results['results'])} pairs")
            print(f"   Average combined score: {direct_results['summary'].get('avg_combined', 0):.3f}")
    except Exception as e:
        print(f"   Direct comparison skipped: {e}")
    
    # # =========================================================================
    # # Step 7: Connectivity Profile Comparison (OPTIONAL - batch comparison)
    # # =========================================================================
    # print("\n🔍 Step 7: Running connectivity profile comparison...")
    
    # This step compares all neuron types from the comparison results.
    # Parameters are read from ComparisonParameters by default, or can be overridden:
    try:
      comparison_results = analyzer.connectivity_profile_comparison()
      if comparison_results:
        summary = comparison_results.get('summary')
        matrix = comparison_results.get('matrix')
        heatmap = comparison_results.get('heatmap_path')
        heatmaps = comparison_results.get('heatmap_paths')
        report = comparison_results.get('report_path') # This might not be returned currently?

        import pandas as pd
        if isinstance(summary, pd.DataFrame) and not summary.empty:
          print(f"   Compared {len(summary)} neuron types across {len(params.datasets)} datasets")
        elif isinstance(summary, dict) and summary:
          print(f"   Summary (dict): {summary}")

        if isinstance(matrix, pd.DataFrame) and not matrix.empty:
          print(f"   Matrix shape: {matrix.shape[0]} types x {max(0, matrix.shape[1]-1)} pairs")
        
        if heatmap:
          print(f"   Heatmap: {heatmap}")
        elif heatmaps:
          print(f"   Heatmaps generated: {list(heatmaps.keys())}")
          
        if report:
          print(f"   HTML report: {report}")
    except Exception as e:
      print(f"   Profile comparison skipped: {e}")
    
    return analyzer


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    import time
    # time it
    
    start_time = time.time()
    try:
        analyzer = run_comprehensive_comparison()
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n⏱️ Total elapsed time: {elapsed/60:.2f} minutes")
