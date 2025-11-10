"""
Example: Using File Picker for Path Visualization

This script demonstrates the new file picker functionality in VisualizePath.
When path_file is None or the file doesn't exist, a file picker dialog will open.
"""

import sys
from pathlib import Path
import warnings

# Add vispath-subproject to Python path for local development
vispath_pkg_path = Path(__file__).parent.parent / 'vispath-subproject' / 'src'
if vispath_pkg_path.exists():
    sys.path.insert(0, str(vispath_pkg_path))

warnings.filterwarnings("ignore")

from vispath_pkg import VisualizePath

if __name__ == '__main__':
    
    print("="*70)
    print("VisualizePath - File Picker Example")
    print("="*70)
    print()
    print("This example shows three ways to load pathway data:")
    print()
    print("1. Using file picker (path_file=None)")
    print("2. Direct file path (auto-detect sheet)")
    print("3. CSV file (sheet_name ignored)")
    print()
    print("="*70)
    print()
    
    # =========================================================================
    # METHOD 1: Use file picker
    # =========================================================================
    
    # Uncomment to open file picker dialog
    """
    print("Opening file picker...")
    vp = VisualizePath(
        path_file=None,              # None = opens file picker
        sheet_name=None,              # None = auto-select or prompt for sheet
        showfig=True
    )
    conn_df, G = vp.visualize()
    print(f"✓ Loaded {len(conn_df)} connections from selected file")
    """
    
    # =========================================================================
    # METHOD 2: Direct Excel file with auto-detection
    # =========================================================================
    
    # Uncomment and modify path to your Excel file
    """
    vp = VisualizePath(
        path_file='your_file.xlsx',
        sheet_name=None,              # None = auto-detect or prompt
        source_color='#1f77b4',
        intermediate_color='#2ca02c',
        target_color='#d62728',
        link_color='rgba(100,100,100,0.3)',
        showfig=True
    )
    conn_df, G = vp.visualize()
    """
    
    # =========================================================================
    # METHOD 3: CSV file (no sheet selection needed)
    # =========================================================================
    
    # Example: Create and load a CSV file
    import pandas as pd
    
    # Create sample data
    sample_paths = pd.DataFrame({
        'path_block': [
            'SourceNeuron -> IntermediateA -> TargetNeuron',
            'SourceNeuron -> IntermediateB -> TargetNeuron',
            'SourceNeuron -> IntermediateA -> IntermediateB -> TargetNeuron'
        ],
        'weights': [
            [100, 50],
            [80, 60],
            [90, 45, 55]
        ],
        'connection_ratios': [
            [0.5, 0.3],
            [0.4, 0.35],
            [0.45, 0.25, 0.32]
        ],
        'traversal_probabilities': [
            [0.8, 0.6],
            [0.7, 0.65],
            [0.75, 0.55, 0.62]
        ]
    })
    
    # Save to CSV
    csv_path = './example_paths.csv'
    sample_paths.to_csv(csv_path, index=False)
    print(f"Created example CSV: {csv_path}")
    
    # Load and visualize
    vp = VisualizePath(
        path_file=csv_path,
        # sheet_name is ignored for CSV files
        source_color='#FF6B6B',           # Red theme
        intermediate_color='#FFA500',     # Orange
        target_color='#FFD700',           # Gold
        link_color='rgba(255,107,107,0.3)',
        output_folder='./csv_visualization',
        showfig=True
    )
    
    conn_df, G = vp.visualize()
    
    print()
    print("="*70)
    print("Visualization Complete!")
    print("="*70)
    print(f"Input: {csv_path}")
    print(f"Connections: {len(conn_df)}")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print()
    print("Output files:")
    print("  • sankey_selected_paths.html - Interactive Sankey diagram")
    print("  • network_selected_paths.html - Interactive network graph")
    print("  • selected_paths_connections.xlsx - Connection data")
    print("="*70)
