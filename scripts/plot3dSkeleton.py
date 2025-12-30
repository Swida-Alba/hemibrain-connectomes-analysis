import sys
from pathlib import Path
import bokeh.palettes as bp

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from coana import VisualizeSkeleton

# please provide your own neuprint token, which can be found at https://neuprint.janelia.org/account
# dataset = 'hemibrain:v1.2.1'
# Note: optic-lobe:v1.1 is in JRCFIB2018Fraw coordinates (same as hemibrain).
# When brain_mesh='whole', it transforms to JRC2018F.
# If alignment looks off, try brain_mesh='template' to view in native JRCFIB2018F space.

import pandas as pd

# df = pd.read_csv('/Users/apple/Local/connection_data/_info/MCNS_v0_9_MeVP_neurons.csv', header=0, index_col=0)
# df = pd.read_csv('/Users/apple/Local/connection_data/_info/FAFB_v783_MTe_neurons.csv', header=0, index_col=0)

# df = pd.read_csv('/Users/apple/Local/connection_data/_info/FAFB_v783_aMe_neurons.csv', header=0, index_col=0)
# df = pd.read_csv('/Users/apple/Local/connection_data/_info/MCNS_v0_9_aMe_neurons.csv', header=0, index_col=0)

# df = pd.read_csv('/Users/apple/Local/connection_data/_info/FAFB_v783_MeMe_neurons.csv', header=0, index_col=0)
df = pd.read_csv('/Users/apple/Local/connection_data/_info/MCNS_v0_9_MeVC_neurons.csv', header=0, index_col=0)

target_types = df['type'].unique().tolist()

vs = VisualizeSkeleton(
    # dataset = 'flywire_FAFB_v783', 
    dataset = 'male-cns:v0.9',
    # dataset = 'hemibrain:v1.2.1',
    token = '',
    output_dir='/Users/apple/Local/connection_data/plot_3d',
    # neuron_layers = ['aMe12', 'MTe07'], # or in the format: 'VA1d_adPN -> LHCENT3 -> MBON01'
    # neuron_layers = [[38776,31872], 12295, 521721], # body IDs of some example neurons, 2 aMe5, 1 aMe20, 1 aMe22, for male-cns:v0.9
    neuron_layers=target_types,
    # custom_layer_names = [],
    # layer_map_csv='/Users/apple/Local/connection_data/_info/FAFB_v783_clock_neurons_map_by_type.csv',
    skip_synapse = True,
    neuron_alpha = 0.2,
    min_synapse_num = 3,
    synapse_size = 'real',
    synapse_alpha = 0.6,
    # mesh_roi = ['EB'],
    # mesh_color=(1,1,1,0),
    skeleton_mode = 'tube',
    synapse_mode = 'cone',
    merge_neurons=True,
    skeleton_mesh_simplification=0.9,
    roi_mesh_simplification=0.95,
    
    show_fig=True,
    brain_mesh='template',
    vnc_mesh=True,
    cache_neurons=True,
    cache_synapses=True,
)

vs.plot_neurons()

# Export PDF with individual neuron profiles, per-legend plots. 
# set VisualizeSkeleton.merge_neurons=False to plot single neurons.
# plot_individuals() should be called AFTER plot_neurons() for correct figure references.

vs.plot_individuals(
    pdf_images_per_page=(3, 2),  # (columns, rows)
    views='front',
    scale=3,
)

# Export rotating video: degree_per_frame controls rotation speed
# 1.0 = 360 frames (12 sec at 30fps), 2.0 = 180 frames (6 sec)
# rotate='horizontal' (default) or 'vertical'

# vs.export_video(
#     fps=30, 
#     degree_per_frame=1.0, 
#     rotate='vertical', 
#     scale=3,
# )
