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
- **NEW**: Connectivity profile verification for neuron type consistency


Comparison Modes:
  - 'path' (default): Edges are discovered through path traversal. May miss 
    strong edges if they're on paths with weak intermediate connections.
  - 'edge': Edges are evaluated independently by weight. Provides direct 
    comparison of synaptic connectivity but loses path-level circuit structure.

Connectivity Profile Verification:
  - Verifies that neurons with the same type label have similar connectivity
    patterns across datasets
  - Uses multiple similarity metrics (Jaccard, Cosine, Rank correlation)
  - Generates confidence scores for type assignments
  - Supports two comparison modes:
    * 'loose' (default): Type-aggregated profiles - faster, compares overall
      connectivity patterns by aggregating all neurons of the same type
    * 'strict': Per-bodyId profiles - more precise, computes rank correlation
      on individual neuron pairs with optional 2-hop expansion
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
    """
    # Output directory
    output_dir = '/Users/apple/Local/connection_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # Step 1: Create ComparisonParameters
    # =========================================================================
    print("\n📋 Step 1: Creating ComparisonParameters...")
    
    neurons_network = ['SMP238']
    
    params = ComparisonParameters(
        # Datasets to compare
        datasets=['flywire_BANC_v626', 'male-cns:v0.9', 'flywire_FAFB_v783', 'hemibrain:v1.2.1'],
        datasets_nickname=['BANC','male-CNS', 'FAFB','hemibrain'],
        
        # Source neurons - aMe12 medulla neurons
        source_neurons=['aMe12','aMe26'],
        # source_neurons=neurons_network,
        
        # Target neurons - PPL101 dopaminergic neurons
        target_neurons=['PPL101','PPL103'],
        # target_neurons=neurons_network,
        
        # Allow 1 intermediate layer (source → inter → target)
        max_interlayer=1,
        
        # Multiple thresholds to analyze sensitivity
        thresholds=[1,3,5,10],
        
        # Top edges to include in analysis
        top_edges=50,
        
        # Comparison mode: 'path' (path-based filtering) or 'edge' (edge-based filtering)
        # - 'path': Discovers edges through paths; may miss strong edges on weak paths
        # - 'edge': Evaluates each edge independently by weight
        comparison_mode='edge',  # Change to 'edge' to use edge-based comparison
        
        # Output settings
        output_folder=output_dir,
        saveas=None,  # Auto-generate timestamp folder
        
        # Token - empty means load from NEUPRINT_APPLICATION_TOKEN env var
        token='',
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
    
    analyzer.export_results()
    
    # =========================================================================
    # Step 5: Additional Analysis with ComparisonVisualizer
    # =========================================================================
    print("\n📈 Step 5: Additional visualization analysis...")
    
    try:
        viz = ComparisonVisualizer()
        print('Done')
        
    except Exception as e:
        print(f"   Visualization analysis skipped: {e}")
    
    # =========================================================================
    # Step 6: Connectivity Profile Verification (NEW)
    # =========================================================================
    print("\n🔬 Step 6: Running connectivity profile verification...")
    
    try:
        # Run connectivity profile verification
        # This verifies that neurons with the same type label have similar
        # connectivity patterns across all compared datasets
        #
        # Two comparison modes are available:
        # - 'loose' (default): Type-aggregated comparison (faster)
        # - 'strict': Per-bodyId comparison with rank correlation (more precise)
        verification_results = analyzer.run_connectivity_profile_verification(
            direction='both',  # Compare both upstream and downstream partners
            comparison_mode='loose',  # 'loose' (type-aggregated) or 'strict' (per-bodyId)
            # Profile extraction parameters
            top_k=10,   # Number of top partners per direction (default: 5)
            top_m=3,   # Minimum unique partner types (default: 0 = no expansion)
            # Strict mode parameters (only used when comparison_mode='strict')
            min_common_partners=3,  # Minimum shared partners for comparison
            # Note: Ranks are always used for bodyId-level comparison
            # 2-hop expansion is controlled via profiler config.expand_untyped_2hop
        )
        
        # Print summary if available
        if 'summary' in verification_results and not verification_results['summary'].empty:
            print("\n   📊 Verification Summary:")
            summary = verification_results['summary']
            
            # Count by verification status
            status_counts = summary['verification_status'].value_counts()
            for status, count in status_counts.items():
                print(f"      • {status}: {count} types")
            
            # Show top verified types
            verified = summary[summary['verification_status'] == 'verified']
            if not verified.empty:
                print(f"\n   ✅ Top verified types (high confidence):")
                for _, row in verified.head(5).iterrows():
                    print(f"      • {row['neuron_type']}: score={row['avg_combined_score']:.3f}")
            
            # Show types needing review
            needs_review = summary[summary['verification_status'].isin(['needs_review', 'questionable'])]
            if not needs_review.empty:
                print(f"\n   ⚠️ Types needing review:")
                for _, row in needs_review.head(5).iterrows():
                    print(f"      • {row['neuron_type']}: score={row['avg_combined_score']:.3f} ({row['confidence']})")
        
        print("\n   ✅ Connectivity profile verification complete!")
        print(f"   📁 Results saved to: {analyzer.parameters.full_output_path}/connectivity_profile_verification/")
        
    except Exception as e:
        print(f"   ⚠️ Connectivity profile verification skipped: {e}")
        import traceback
        traceback.print_exc()
    
    return analyzer


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    
    try:
        analyzer = run_comprehensive_comparison()
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
