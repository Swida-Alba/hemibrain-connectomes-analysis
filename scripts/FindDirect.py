import sys
from pathlib import Path
import warnings
import pandas as pd

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

warnings.filterwarnings("ignore")
from coana import FindNeuronConnection

if __name__ == '__main__':
    # neurons_network = [['aMe5','aMe9','aMe10','aMe12', 'aMe26'], 'KCg-d', 'KCg-m', 'KCab-p', 'PPL101', 'PPL103', 'APL', 'DPM']
    # neurons_network = [['aMe5','aMe9','aMe10','aMe12', 'aMe26'], 'KCg-d', 'KCg-m', 'KCab-p', 'PPL101', 'PPL103']
    # neurons_network = ['aMe12','aMe26', 'KCg-d', 'KCg-m', 'KCab-p', 'PPL101', 'PPL103', 'APL', 'DPM']
    # neurons_network = ['aMe.*']  # Empty list means all neurons in the dataset
    # neurons_network = [['aMe12.*_R'], ['aMe12.*_L'], ['KCg-s1.*_R'], ['KCg-s1.*_L']]
    neurons_network = ['aMe.*']
    
    fc = FindNeuronConnection(
        # please provide your own neuprint token, which can be found at https://neuprint.janelia.org/account
        token='',
        data_folder='/Users/apple/Local/connection_data',
        # dataset='flywire_FAFB_v783',  # Combined dataset and version in one parameter
        # dataset = 'male-cns:v0.9',
        dataset='hemibrain:v1.2.1',
        sourceNeurons=neurons_network,  # [] = all neurons; or list of types/instances like ['L3.*_R']
        targetNeurons=neurons_network,  # pd.read_excel('targetNeurons.xlsx', header=None).iloc[:,0].tolist()
        exclude_intra_type_connections=False,
        # sourceNeurons=[],  # [] = all neurons; or list of types/instances like ['L3.*_R']
        # targetNeurons=['s-LNv', 'l-LNv', 'LNd'],
        # sourceNeurons = pd.read_excel('sourceNeurons.xlsx', header=None).iloc[:,0].tolist(),
        # targetNeurons = pd.read_excel('targetNeurons.xlsx', header=None).iloc[:,0].tolist(),
        # custom_source_name='LM',  # Custom name for source neurons (useful when using [] or reading from file)
        # custom_target_name='LM_Network',  # Custom name for target neurons
        min_synapse_num=1,
        min_ratio=0.0,  # Minimum connection ratio (weight/post) - direct ratio without 0.3 scaling
        min_traversal_probability=0.0,  # Minimum traversal probability (ratio/0.3, capped at 1.0)
        filter_by='bodyId',  # 'bodyId' or 'type' level filtering
        showfig=False,
        use_cache=True,  # Enable caching for faster subsequent runs
        network_layout='distributed',
        output_format='csv',  # 'xlsx' (default) or 'csv'
        edgeN_limit=50,
    )

    fc.InitializeNeuronInfo()
    fc.FindDirectConnections()
