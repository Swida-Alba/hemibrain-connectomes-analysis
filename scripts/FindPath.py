import sys
from pathlib import Path
import warnings
import pandas as pd
import time

# Add project root and src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

warnings.filterwarnings("ignore")
from src.coana import FindNeuronConnection

if __name__ == '__main__':
    fc = FindNeuronConnection(
        # Token automatically loaded from token_info.txt (recommended) or set token='' here
        output_dir='../local_data/connection_data',
        # dataset='male-cns:v0.9', 
        # dataset='hemibrain:v1.2.1',
        # dataset='optic-lobe:v1.1',
        dataset='flywire_FAFB_v783',
        sourceNeurons=['CB0038'],  # pd.read_excel('sourceNeurons.xlsx', header=None).iloc[:,0].tolist()
        targetNeurons=['LPLC2'],
        custom_source_name='Fdg', # you can specify a custom name for the source neurons, especially when you are using a list of many types of neurons or a list of neurons read from a file
        custom_target_name='',  # you can specify a custom name for the target neurons
        custom_source_group_names=[],
        custom_target_group_names=[],
        min_synapse_num=3,
        min_ratio=0.0,
        min_traversal_probability=0,
        filter_by='bodyId',  # 'bodyId' or 'type' level filtering
        showfig=False,
        max_interlayer=4,
        keyword_in_path_to_remove=['None'],
        network_layout='distributed',
        use_cache=True,  # Enable caching for faster subsequent runs
        edgeN_limit=500,
        output_format='csv',  # 'xlsx' (default) or 'csv'
        pathfinding='Bidirectional',  # 'Bidirectional' (fastest), 'DP' (backward), 'MemoizedDFS' (depends on repeats), 'DFS'
        skip_bodyId=True,
    )

    start_time = time.time()
    fc.InitializeNeuronInfo()
    # fc.FindPath()
    fc.FindAllPath(forward_only=True)
    end_time = time.time()
    print(f"Pathfinding completed in {end_time - start_time:.2f} seconds")