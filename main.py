import sys
from pathlib import Path
import warnings

# Add vispath-subproject to Python path for local development
vispath_pkg_path = Path(__file__).parent / 'src'
if vispath_pkg_path.exists():
    sys.path.insert(0, str(vispath_pkg_path))

warnings.filterwarnings("ignore")

from vispath_pkg import VisualizePath

if __name__ == '__main__':
    
    # Create visualizations with custom colors
    vp = VisualizePath(
        path_file='',
        sheet_name=0,             # or 'path_bodyId' for bodyId-level paths (None = auto-select)
        output_folder=None,                 # None = auto-creates 'selected_paths' folder
        source_color='#4A90E2',          # Custom source node color
        intermediate_color='#50E3C2',  # Custom intermediate node color
        target_color='#B8E986',          # Custom target node color
        link_color='rgba(74,144,226,0.3)',              # Custom link color
        network_layout='hierarchical',      # 'hierarchical', 'spring', 'circular', 'distributed'
        edge_width_scale='sqrt',            # Edge width scaling method
        showfig=True,                        # Open visualizations in browser
        max_edge_width=30,
        generate_empty_network=False, # Set to True to generate empty network for manual editing, ignore path_file
    )
    
    # Generate all visualizations
    conn_df, G = vp.visualize()
    
    print(f"\n✓ Visualization complete!")
    print(f"✓ Created {len(conn_df) if conn_df is not None else 0} connections")
    print(f"✓ Network has {G.number_of_nodes() if G is not None else 0} nodes and {G.number_of_edges() if G is not None else 0} edges")

    print("\n" + "="*80)
    print("Output files created:")
    print("  • sankey_selected_paths.html - Flow-based diagram")
    print("  • network_selected_paths.html - Interactive network (drag, hide, hover)")
    print("  • selected_paths_connections.xlsx - Connection data")
    print("="*80)
