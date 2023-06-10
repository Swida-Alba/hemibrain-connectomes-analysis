import statvis as sv
import bokeh.palettes as bp
from coana import VisualizeSkeleton

# please provide your own neuprint token, which can be found at https://neuprint.janelia.org/account
server_client, dataset = sv.LogInHemibrain(token='')
layers = ['aMe12','SMP238','ExR1','ER5']

vs = VisualizeSkeleton(
    neuron_layers = layers,
    custom_layer_names = [],
    neuron_alpha=0.1,
    saveas=None,
    min_synapse_num = 1,
    synapse_size = 3, 
    synapse_alpha = 0.6,
    synapse_mode = 'sphere',
    mesh_roi = ['LH(R)','AL(R)','EB'],
    use_size_slider = True,
    legend_mode = 'normal',
)
vs.plot_neurons()
