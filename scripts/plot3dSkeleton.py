import sys
from pathlib import Path
import bokeh.palettes as bp

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from coana import VisualizeSkeleton
import statvis as sv

# please provide your own neuprint token, which can be found at https://neuprint.janelia.org/account
# dataset = 'hemibrain:v1.2.1'
# Note: optic-lobe:v1.1 is in JRCFIB2018Fraw coordinates (same as hemibrain).
# When brain_mesh='whole', it transforms to JRC2018F.
# If alignment looks off, try brain_mesh='template' to view in native JRCFIB2018F space.
dataset = 'optic-lobe:v1.1'
server_client, dataset = sv.LogInHemibrain(token='',dataset = dataset)
vs = VisualizeSkeleton(
    dataset = dataset,
    data_folder='/Users/apple/Local/connection_data',
    neuron_layers = ['aMe12','aMe26'], # or in the format: 'VA1d_adPN -> LHCENT3 -> MBON01'
    custom_layer_names = [],
    ignore_synapses=True,
    neuron_alpha = 0.2,
    min_synapse_num = 1,
    synapse_size = 2,
    synapse_alpha = 0.6,
    mesh_roi = ['ME(R)'],
    mesh_color= (100,100,100,0.2),
    mirror_on_contralateral=True, # Mirror neurons and ROIs (e.g. ME(R) -> ME(L)) to contralateral side
    skeleton_mode = 'tube',
    synapse_mode = 'scatter',
    legend_mode = 'merge',
    merge_neurons=True,
    mesh_simplification=0.5,
    use_size_slider = False,
    show_fig = True,
    brain_mesh = 'whole',
    cache_neurons=True,
    cache_synapses=True,
    backend='k3d',
)

vs.plot_neurons()
# Use lower scale (1) for faster rendering of large scenes
vs.export_video(fps=30, scale=1)
