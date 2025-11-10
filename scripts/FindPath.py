import sys
from pathlib import Path
import warnings
import pandas as pd

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

warnings.filterwarnings("ignore")
from coana import FindNeuronConnection

if __name__ == '__main__':
    fc = FindNeuronConnection(
        # please provide your own neuprint token, which can be found at https://neuprint.janelia.org/account
        token='',
        data_folder='/Users/apple/Local/connection_data',
        dataset='hemibrain:v1.2.1', 
        sourceNeurons=['aMe.*_R'],  # pd.read_excel('sourceNeurons.xlsx', header=None).iloc[:,0].tolist()
        targetNeurons=['PPL103.*_R'],  # pd.read_excel('targetNeurons.xlsx', header=None).iloc[:,0].tolist()
        # sourceNeurons = pd.read_excel('sourceNeurons.xlsx', header=None).iloc[:,0].tolist(),
        # targetNeurons = pd.read_excel('targetNeurons.xlsx', header=None).iloc[:,0].tolist(),
        # custom_source_name = 'VTaMe', # you can specify a custom name for the source neurons, especially when you are using a list of many types of neurons or a list of neurons read from a file
        # custom_target_name='',  # you can specify a custom name for the target neurons
        min_synapse_num=1,
        min_ratio=0,
        min_traversal_probability=0,
        filter_by='bodyId',  # 'bodyId' or 'type' level filtering
        showfig=False,
        max_interlayer=2,
        keyword_in_path_to_remove=['None'],
        network_layout='distributed',
        use_cache=True,  # Enable caching for faster subsequent runs
        use_parallel=True,  # Enable parallel processing for pathfinding (4-14x faster for large datasets)
        n_jobs=-1,  # Use all CPU cores (-1 = auto-detect, or specify number like 4)
        pathN_to_show=10,
    )

    fc.InitializeNeuronInfo()
    # fc.FindPath()
    fc.FindAllPath(forward_only=True)
