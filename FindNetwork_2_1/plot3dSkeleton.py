import statvis as sv
import bokeh.palettes as bp
from coana import VisualizeSkeleton


server_client, dataset = sv.LogInHemibrain()
layers = ['aMe12','SMP238','ExR1.*_R','ER5.*_R']

vs = VisualizeSkeleton(
    neuron_layers=layers,
    min_synapse_num = 10,
    synapse_size = 100, 
    synapse_mode='sphere',
    mesh_roi = ['LH(R)','AL(R)','EB'],
    use_size_slider = False,
)
vs.plot_neurons()
