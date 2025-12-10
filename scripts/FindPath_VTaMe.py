import sys
from pathlib import Path
import warnings
import pandas as pd

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

warnings.filterwarnings("ignore")
from coana import FindNeuronConnection

if __name__ == '__main__':
    df_VTaMe = pd.read_excel('/Users/apple/Local/connection_data/VT037867_aMe.xlsx', header=0, index_col=0)
    vt_neurons = df_VTaMe.bodyId.tolist()

    fc = FindNeuronConnection(
        # please provide your own neuprint token, which can be found at https://neuprint.janelia.org/account
        token='',
        output_dir='/Users/apple/Local/connection_data',
        dataset='hemibrain:v1.2.1', 
        sourceNeurons=['PPL103'],  # pd.read_excel('sourceNeurons.xlsx', header=None).iloc[:,0].tolist()
        targetNeurons=vt_neurons,  # pd.read_excel('targetNeurons.xlsx', header=None).iloc[:,0].tolist()
        custom_source_name='',  # you can specify a custom name for the source neurons, especially when you are using a list of many types of neurons or a list of neurons read from a file
        custom_target_name='VTaMe',  # you can specify a custom name for the target neurons
        min_synapse_num=1,
        min_traversal_probability=1e-6,
        showfig=False,
        max_interlayer=2,
        keyword_in_path_to_remove=['APL', 'None'],
        simple_fetch=False,
    )

    fc.InitializeNeuronInfo()
    fc.FindAllPath()
