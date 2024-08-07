import warnings
import pandas as pd
warnings.filterwarnings("ignore")
from coana import FindNeuronConnection

fc = FindNeuronConnection(
    # please provide your own neuprint token, which can be found at https://neuprint.janelia.org/
    token='',
    data_folder='', # the folder to save the data
    dataset = 'hemibrain:v1.2.1', 
    sourceNeurons = ['MBON.*'], # pd.read_excel('sourceNeurons.xlsx', header=None).iloc[:,0].tolist()
    targetNeurons = ['aMe.*'], # pd.read_excel('targetNeurons.xlsx', header=None).iloc[:,0].tolist()
    # sourceNeurons = pd.read_excel('sourceNeurons.xlsx', header=None).iloc[:,0].tolist(),
    # targetNeurons = pd.read_excel('targetNeurons.xlsx', header=None).iloc[:,0].tolist(),
    custom_source_name = '', # you can specify a custom name for the source neurons, especially when you are using a list of many types of neurons or a list of neurons read from a file
    custom_target_name = '', # you can specify a custom name for the target neurons
    min_synapse_num = 1,
    min_traversal_probability = 0.001,
    showfig = False,
)

fc.InitializeNeuronInfo()
fc.FindDirectConnections(full_data=False)

