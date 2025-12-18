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
  Parameters can be configured in ComparisonParameters or passed directly.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

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
    
    label_map = LabelMapper(
        # target_mapping_dict={
        #     'flywire_FAFB_v783': [[720575940619067259,720575940613413791,720575940631973089,720575940642237344,720575940606868828,720575940609174392,720575940610478531,720575940628527095]],
        #     'flywire_BANC_v626': [[720575941552713626,720575941589129982,720575941645302945,720575941689244824]],
        #     'male-cns:v0.9': [[15832,16461,16552,17336,16634,17355,17916,18945]],
        # },
        # target_labels=['s-LNv'],
        
        target_mapping_dict={
            'flywire_FAFB_v783': [[720575940634984800,720575940627933336,720575940625254636,720575940619074049]],
            'flywire_BANC_v626': [[720575941671706023,720575941645496264,720575941568371246]],
            'male-cns:v0.9': [[11901,14633,12254,13531]],
        },
        target_labels=['E cells'],
        
        # target_mapping_dict={
        #   'flywire_FAFB_v783': [['l-LNv.*']],
        #   'flywire_BANC_v626': [['l-LNv.*']],
        #   'male-cns:v0.9': [['l-LNv.*']],
        # },
        # target_labels=['l-LNv'],
        
        # target_mapping_dict={
        #   'flywire_FAFB_v783': [[720575940625254636,720575940619074049]],
        #   'flywire_BANC_v626': [[720575941568371246]],
        #   'male-cns:v0.9': [[12254,13531]],
        # },
        # target_labels=['5th s-LNv'],
        
        # target_mapping_dict={
        #     'flywire_FAFB_v783': [[720575940634984800,720575940627933336]],
        #     'flywire_BANC_v626': [[720575941671706023,720575941645496264]],
        #     'male-cns:v0.9': [[11901,14633]],
        # },
        # target_labels=['ITP LNd'],
        
        source_mapping_dict={
            'flywire_FAFB_v783': ['L1','L2','L3'],
            'flywire_BANC_v626': ['L1','L2','L3'],
            'male-cns:v0.9': ['L1','L2','L3'],
        },
        source_labels=['L1','L2','L3'],
        
        intermediate_mapping_file='/Users/apple/Desktop/intermediate_map.csv',
        
    )
    
    # target_map = LabelMapper(
    #   target_mapping_dict={
    #     'flywire_FAFB_v783': [['l-LNv.*']],
    #     'flywire_BANC_v626': [['l-LNv.*']],
    #     'male-cns:v0.9': [['l-LNv.*']],
    #   },
    #   target_labels=['l-LNv'],
    # )
    
    # target_map = LabelMapper(
    #   target_mapping_dict={
    #     'flywire_FAFB_v783': [[720575940625254636,720575940619074049]],
    #     'flywire_BANC_v626': [[720575941568371246]],
    #     'male-cns:v0.9': [[12254,13531]],
    #   },
    #   target_labels=['5th s-LNv'],
    # )
    
    # target_map = LabelMapper(
    #   target_mapping_dict={
    #     'flywire_FAFB_v783': [[720575940634984800,720575940627933336]],
    #     'flywire_BANC_v626': [[720575941671706023,720575941645496264]],
    #     'male-cns:v0.9': [[11901,14633]],
    #   },
    #   target_labels=['ITP LNd'],
    # )
    
    # target_map = LabelMapper(
    #     target_mapping_dict={
    #         'flywire_FAFB_v783': [[720575940634984800,720575940627933336,720575940625254636,720575940619074049]],
    #         'flywire_BANC_v626': [[720575941671706023,720575941645496264,720575941568371246]],
    #         'male-cns:v0.9': [[11901,14633,12254,13531]],
    #     },
    #     target_labels=['E cells'],
    # )
    
    params = ComparisonParameters(
        # Token - empty means load from NEUPRINT_APPLICATION_TOKEN env var
        token='',
        # Output settings
        output_folder='/Users/apple/Local/connection_data',
        saveas=None,  # Auto-generate timestamp folder
        
        # Datasets to compare
        # datasets=['male-cns:v0.9', 'flywire_FAFB_v783', 'hemibrain:v1.2.1', 'flywire_BANC_v626'],
        # datasets_nickname=['MCNS', 'FAFB', 'HEMI', 'BANC',],
        
        datasets=['flywire_FAFB_v783', 'male-cns:v0.9'],
        datasets_nickname=['FAFB', 'MCNS'],
        
        # source_neurons=['aMe.*'],
        # source_neurons=neurons_network,
        
        # Target neurons - PPL101 dopaminergic neurons
        # target_neurons=['PPL101','PPL103'],
        # target_neurons=target_map,
        overall_label_mapper=label_map,
        
        max_interlayer=3,
        
        # Multiple thresholds to analyze sensitivity
        thresholds=[3, 5, 10, 15, 20],
        # thresholds = list(range(1,21)),
        
        # Top edges to include in analysis
        top_edges=500,
        
        # Comparison mode: 'path' (path-based filtering) or 'edge' (edge-based filtering)
        # - 'path': Discovers edges through paths; may miss strong edges on weak paths
        # - 'edge': Evaluates each edge independently by weight
        comparison_mode='path',  # Change to 'edge' to use edge-based comparison
        
        skip_bodyId=True, # Skip bodyId-level results for speed and local storage
        
        # Performance Settings (path finding algorithm)
        # -----------------------------------------------------------------
        pathfinding='MemoizedDFS', # 'MemoizedDFS', 'Bidirectional', 'DP', 'DFS'
    )
    
    analyzer = ComparisonAnalyzer(params, verbose=True)
    results = analyzer.run_comparison()
    analyzer.generate_report()
    analyzer.export_results()
    
    ### ====== optional comparison of connectivity profiles ======
    
    # comparison_results = analyzer.connectivity_profile_comparison()
    
    ### ============================================================
    
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
