"""
Example: Network Visualization with Excluded Intra-Type Connections

This example demonstrates how to use the exclude_intra_type_connections parameter
to build and visualize networks that focus on inter-type connectivity patterns.

Use Case: Analyzing connections between different MBON types while excluding 
connections within the same MBON type (e.g., MBON01→MBON01).

This is particularly useful for:
1. Understanding cross-type communication pathways
2. Creating cleaner network visualizations
3. Identifying inter-type connectivity patterns without intra-type noise
"""

import sys
from pathlib import Path
import warnings

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

warnings.filterwarnings("ignore")
from coana import FindNeuronConnection

if __name__ == '__main__':
    # Example 1: Direct connections between MBONs, excluding intra-type connections
    print("=" * 80)
    print("Example 1: MBON→MBON Direct Connections (Excluding Intra-Type)")
    print("=" * 80)
    
    fc_mbon = FindNeuronConnection(
        # Your neuprint token from https://neuprint.janelia.org/account
        token='',
        data_folder='/path/to/your/data/folder',
        dataset='hemibrain:v1.2.1',
        
        # Search for connections between all MBON types
        sourceNeurons=[['MBON.*']],
        targetNeurons=[['MBON.*']],
        
        # Custom names for clarity
        custom_source_name='All_MBONs',
        custom_target_name='All_MBONs',
        
        # Exclude connections within the same type (e.g., MBON01→MBON01)
        exclude_intra_type_connections=True,
        
        # Filter parameters
        min_synapse_num=3,
        min_traversal_probability=0.01,
        filter_by='bodyId',
        
        # Visualization settings
        showfig=True,
        network_layout='distributed',  # Better for cross-type connections
        
        # Performance
        use_cache=True,
    )
    
    # Initialize and find direct connections
    fc_mbon.InitializeNeuronInfo()
    fc_mbon.FindDirectConnections()
    
    print("\nResults saved! Check the output folder for:")
    print("  - Connection matrix showing only inter-type connections")
    print("  - Heatmap highlighting cross-type connectivity patterns")
    print("  - Network visualization focused on inter-type pathways")
    
    # Example 2: Multi-hop pathfinding between Kenyon Cells and DANs, excluding intra-type
    print("\n" + "=" * 80)
    print("Example 2: KC→DAN Pathfinding (Excluding Intra-Type at All Layers)")
    print("=" * 80)
    
    fc_kc_dan = FindNeuronConnection(
        token='',
        data_folder='/path/to/your/data/folder',
        dataset='hemibrain:v1.2.1',
        
        # Find multi-hop paths from Kenyon Cells to DANs
        sourceNeurons=[['KC.*']],
        targetNeurons=[['DAN.*']],
        
        custom_source_name='Kenyon_Cells',
        custom_target_name='DANs',
        
        # Exclude intra-type connections at all layers
        # This ensures intermediate neurons don't have self-connections
        exclude_intra_type_connections=True,
        
        # Path parameters
        max_interlayer=2,
        keyword_in_path_to_remove=['None'],
        
        # Filter parameters
        min_synapse_num=5,
        min_traversal_probability=0.01,
        filter_by='type',  # Type-level filtering
        
        # Visualization
        showfig=True,
        network_layout='layered',  # Good for hierarchical paths
        pathN_to_show=20,  # Show top 20 paths
        
        # Performance
        use_cache=True,
    )
    
    fc_kc_dan.InitializeNeuronInfo()
    fc_kc_dan.FindAllPath(forward_only=True)
    
    print("\nPathfinding complete! The exclude_intra_type_connections filter ensures:")
    print("  - No KC→KC connections in paths")
    print("  - No DAN→DAN connections in paths")
    print("  - No self-connections in intermediate layers")
    print("  - Cleaner visualization of cross-type pathways")
    
    # Example 3: Comparing with and without intra-type exclusion
    print("\n" + "=" * 80)
    print("Example 3: Comparison - With and Without Intra-Type Exclusion")
    print("=" * 80)
    
    # First run: Include all connections
    print("\n[Run 1] Including intra-type connections...")
    fc_with = FindNeuronConnection(
        token='',
        data_folder='/path/to/your/data/folder',
        dataset='hemibrain:v1.2.1',
        sourceNeurons=[['MBON01', 'MBON03', 'MBON05']],
        targetNeurons=[['MBON01', 'MBON03', 'MBON05']],
        custom_source_name='Selected_MBONs',
        custom_target_name='Selected_MBONs',
        exclude_intra_type_connections=False,  # Include intra-type
        min_synapse_num=3,
        showfig=False,
        use_cache=True,
    )
    fc_with.InitializeNeuronInfo()
    fc_with.FindDirectConnections()
    
    # Second run: Exclude intra-type connections
    print("\n[Run 2] Excluding intra-type connections...")
    fc_without = FindNeuronConnection(
        token='',
        data_folder='/path/to/your/data/folder',
        dataset='hemibrain:v1.2.1',
        sourceNeurons=[['MBON01', 'MBON03', 'MBON05']],
        targetNeurons=[['MBON01', 'MBON03', 'MBON05']],
        custom_source_name='Selected_MBONs',
        custom_target_name='Selected_MBONs',
        exclude_intra_type_connections=True,  # Exclude intra-type
        min_synapse_num=3,
        showfig=False,
        use_cache=True,
    )
    fc_without.InitializeNeuronInfo()
    fc_without.FindDirectConnections()
    
    print("\nComparison complete! Check the output folders to see:")
    print("  - Run 1: Includes all connections (MBON01→MBON01, etc.)")
    print("  - Run 2: Only cross-type connections (MBON01→MBON03, etc.)")
    print("\nThis demonstrates how excluding intra-type connections:")
    print("  ✓ Simplifies network structure")
    print("  ✓ Highlights inter-type communication")
    print("  ✓ Reduces visual clutter in network diagrams")
    print("  ✓ Focuses analysis on cross-type functional pathways")

    print("\n" + "=" * 80)
    print("Examples Complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Set exclude_intra_type_connections=True to focus on inter-type connectivity")
    print("2. This applies to FindDirect, FindPath, and FindAllPath methods")
    print("3. Particularly useful for network visualization and cross-type analysis")
    print("4. The filter is applied early, before weight/ratio/probability filters")
    print("\nFor more information, see the documentation:")
    print("  - README.md: Parameter descriptions")
    print("  - docs/core-features/: Detailed feature guides")
    print("  - examples/README.md: More example scripts")
