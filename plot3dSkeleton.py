import statvis as sv
import bokeh.palettes as bp
from coana import VisualizeSkeleton

# please provide your own neuprint token, which can be found at https://neuprint.janelia.org/account
server_client, dataset = sv.LogInHemibrain(token='')
layers = []

vs = VisualizeSkeleton(
    neuron_layers = layers,
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