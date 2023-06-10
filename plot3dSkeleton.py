import statvis as sv
import bokeh.palettes as bp
from coana import VisualizeSkeleton
a = ["a'L(L)", "a'L(R)", 'AB(L)', 'AB(R)', 'AL(L)_', 'AL(R)', 'alphaL(L)', 'alphaL(R)', 'AME(R)', 'AOTU(R)', 'ATL(L)', 'ATL(R)', 'AVLP(R)', "b'L(L)", "b'L(R)", 'bL(L)', 'bL(R)', 'BU(L)', 'BU(R)', 'CA(L)', 'CA(R)', 'CAN(R)', 'CRE(L)', 'CRE(R)', 'EB', 'EPA(L)', 'EPA(R)', 'FB', 'FLA(R)', 'gL(L)', 'gL(R)', 'GNG', 'GOR(L)', 'GOR(R)', 'IB', 'ICL(L)', 'ICL(R)', 'IPS(R)', 'LAL(L)', 'LAL(R)', 'LH(R)', 'LO(R)', 'LOP(R)', 'ME(R)', 'NO', 'PB', 'PED(R)', 'PLP(R)', 'PRW', 'PVLP(R)', 'SAD', 'SCL(L)', 'SCL(R)', 'SIP(L)', 'SIP(R)', 'SLP(R)', 'SMP(L)', 'SMP(R)', 'SPS(L)', 'SPS(R)', 'VES(L)', 'VES(R)', 'WED(R)']
for i in a:
    print(i, '\\n')
# please provide your own neuprint token, which can be found at https://neuprint.janelia.org/account
server_client, dataset = sv.LogInHemibrain(token='')
vs = VisualizeSkeleton(
    neuron_layers = ['ORN_VC5', 'VP2+VC5_l2PN', 'SAD087', 'DNp26'], # or 'ORN_VC5 -> VP2+VC5_l2PN -> SAD087 -> DNp26'
    custom_layer_names = [],
    neuron_alpha = 0.2,
    saveas = None,
    min_synapse_num = 1,
    synapse_size = 3, 
    synapse_alpha = 0.6,
    mesh_roi = ['LH(R)','AL(R)','EB'],
    skeleton_mode = 'tube',
    synapse_mode = 'scatter',
    legend_mode = 'merge',
    use_size_slider = True,
)
vs.plot_neurons()
