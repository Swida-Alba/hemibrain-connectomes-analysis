import statvis as sv
import bokeh.palettes as bp
from coana import VisualizeSkeleton

# please provide your own neuprint token, which can be found at https://neuprint.janelia.org/account
server_client, dataset = sv.LogInHemibrain(token='')
vs = VisualizeSkeleton(
    neuron_layers = ['VA1d_adPN','LHCENT3','MBON01'], # or in the format: 'VA1d_adPN -> LHCENT3 -> MBON01'
    custom_layer_names = [],
    neuron_alpha = 0.2,
    saveas = None,
    min_synapse_num = 1,
    synapse_size = 2,
    synapse_alpha = 0.6,
    mesh_roi = ['LH(R)','AL(R)','EB'],
    skeleton_mode = 'tube',
    synapse_mode = 'scatter',
    legend_mode = 'merge',
    use_size_slider = True,
    show_fig = True,
    brain_mesh = 'whole',
)

vs.plot_neurons()
vs.export_video(fps=30)
