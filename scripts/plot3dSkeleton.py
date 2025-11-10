import sys
from pathlib import Path
import statvis as sv
import bokeh.palettes as bp

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from coana import VisualizeSkeleton

# please provide your own neuprint token, which can be found at https://neuprint.janelia.org/account
dataset = 'optic-lobe:v1.1'
server_client, dataset = sv.LogInHemibrain(token='',dataset = dataset)
vs = VisualizeSkeleton(
    dataset = dataset,
    data_folder='/Users/apple/Local/connection_data',
    neuron_layers = ['LNd'], # or in the format: 'VA1d_adPN -> LHCENT3 -> MBON01'
    custom_layer_names = [],
    neuron_alpha = 0.2,
    saveas = None,
    min_synapse_num = 1,
    synapse_size = 2,
    synapse_alpha = 0.6,
    # mesh_roi = ['EB','ME(R)', 'AME(R)'],
    skeleton_mode = 'tube',
    synapse_mode = 'scatter',
    legend_mode = 'normal',
    use_size_slider = True,
    show_fig = True,
    brain_mesh = 'none',
)

vs.plot_neurons()
# vs.export_video(fps=30)
