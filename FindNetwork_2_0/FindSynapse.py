import warnings
import pandas as pd
warnings.filterwarnings("ignore")
from coana import FindNeuronConnection

fc = FindNeuronConnection(dataset = 'hemibrain:v1.2.1', showfig = False,)

required_neurons = ['MBON05']

fc.ROImat(
    requiredNeurons = required_neurons,
    folder_name = None, 
    site='pre', # 'pre' or 'post'
    break_threshod=2000,
)

fc.ROImat(
    requiredNeurons = required_neurons,
    folder_name = None, 
    site='post', # 'pre' or 'post'
    break_threshod=2000,
)

# fc.SynapseDistribution(
#     requiredNeurons = required_neurons, # pd.read_excel('requiredNeurons.xlsx', header=None).iloc[:,0].tolist()
#     site = 'pre',
#     snp_rois = ['SMP(R)'],
#     info_df = pd.DataFrame(),
# )