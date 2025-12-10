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
        # please provide your own neuprint token, which can be found at https://neuprint.janelia.org/account
        token='',
        output_dir='/Users/apple/Local/connection_data',
        # saveas='_name_test1',
        dataset='male-cns:v0.9', 
        # dataset='hemibrain:v1.2.1',
        # dataset='optic-lobe:v1.1',
        # dataset='flywire_FAFB_v783',
        sourceNeurons=['L2'],  # pd.read_excel('sourceNeurons.xlsx', header=None).iloc[:,0].tolist()
        # targetNeurons=['DN.*'],  # pd.read_excel('targetNeurons.xlsx', header=None).iloc[:,0].tolist()
        targetNeurons=['DN.*'],
        # sourceNeurons = pd.read_excel('sourceNeurons.xlsx', header=None).iloc[:,0].tolist(),
        # targetNeurons = pd.read_excel('targetNeurons.xlsx', header=None).iloc[:,0].tolist(),
        custom_source_name='', # you can specify a custom name for the source neurons, especially when you are using a list of many types of neurons or a list of neurons read from a file
        custom_target_name='',  # you can specify a custom name for the target neurons
        custom_source_group_names=[],
        custom_target_group_names=[],
        min_synapse_num=20,
        min_ratio=0.0,
        min_traversal_probability=0,
        filter_by='bodyId',  # 'bodyId' or 'type' level filtering
        showfig=False,
        max_interlayer=5,
        keyword_in_path_to_remove=['None'],
        network_layout='distributed',
        use_cache=True,  # Enable caching for faster subsequent runs
        edgeN_limit=500,
        output_format='csv',  # 'xlsx' (default) or 'csv'
        pathfinding='MemoizedDFS',  # 'Bidirectional' (fastest), 'DP' (backward), 'MemoizedDFS' (depends on repeats), 'DFS'
        skip_bodyId=False,
    )

    start_time = time.time()
    fc.InitializeNeuronInfo()
    # fc.FindPath()
    fc.FindAllPath(forward_only=True)
    end_time = time.time()
    print(f"Pathfinding completed in {end_time - start_time:.2f} seconds")