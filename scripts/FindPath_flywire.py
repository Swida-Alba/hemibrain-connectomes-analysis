import sys
from pathlib import Path
import warnings
import pandas as pd

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

warnings.filterwarnings("ignore")
from coana import FindNeuronConnection, VisualizeSkeleton

if __name__ == '__main__':
    print("Initializing FlyWire Pathfinding...")
    dataset = 'flywire_FAFB_v783' # Use local FAFB dataset name
    # dataset = 'flywire_BANC_v626'
    
    fc = FindNeuronConnection(
        token='', 
        data_folder='/Users/apple/Local/connection_data/_flywire_test',
        dataset=dataset, 
        sourceNeurons=['aMe12'], 
        targetNeurons=['PPL101','PPL103'],
        custom_source_name='', 
        custom_target_name='',
        min_synapse_num=5, # Adjust as needed for FlyWire (synapse counts might be different)
        min_ratio=0.0,
        min_traversal_probability=0.0,
        filter_by='bodyId',
        showfig=True,
        max_interlayer=2,
        keyword_in_path_to_remove=[], # Empty list to keep all paths (FlyWire has many None types)
        network_layout='distributed',
        use_cache=True, 
        use_parallel=True,
        n_jobs=-1,
        edgeN_limit=50,
        output_format='csv',
    )

    print("Initializing Neuron Info...")
    fc.InitializeNeuronInfo()
    
    
    print("Finding Paths...")
    # fc.FindDirectConnections()
    # fc.FindPath()
    fc.FindAllPath(forward_only=True)
    print("Done.")
    
    # Skeleton Visualization
    print("\nVisualizing Skeletons...")
    # Visualize source and target neurons
    vs = VisualizeSkeleton(
        dataset=dataset,
        data_folder='/Users/apple/Local/connection_data/_flywire_test',
        neuron_layers=['aMe12', 'KCg-s1',['PPL101', 'PPL103']], # Test with source and target neurons
        # custom_layer_names=[],
        mesh_roi=['ME(R)', 'EB', 'AL(R)', 'ME(L)', 'AL(L)'], # Add ROIs to test loading and mirroring
        
        skip_synapse=False, # Synapses might be heavy for FAFB
        min_synapse_num=5,
        synapse_mode='cone',
        synapse_alpha=0.8,
        synapse_size='real',
        
        neuron_alpha=0.2,
        skeleton_mode='tube',
        merge_neurons=True,
        use_size_slider=False,
        skeleton_mesh_simplification=0.9,
        roi_mesh_simplification=0.95,
        show_fig=True,
        brain_mesh='whole', # No brain mesh for FAFB yet
    )
    vs.plot_neurons()
    # vs.export_video(fps=30, scale=1)
