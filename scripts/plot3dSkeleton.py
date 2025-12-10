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

vs = VisualizeSkeleton(
    # dataset = 'flywire_FAFB_v783', 
    dataset = 'male-cns:v0.9',
    # dataset = 'hemibrain:v1.2.1',
    token = '',
    output_dir='/Users/apple/Local/connection_data',
    neuron_layers = ['MeVPLo2','aMe12'], # or in the format: 'VA1d_adPN -> LHCENT3 -> MBON01'
    custom_layer_names = [],
    skip_synapse=False,
    neuron_alpha = 0.2,
    min_synapse_num = 3,
    synapse_size = 'real',
    use_size_slider = False,
    synapse_alpha = 0.6,
    # mesh_roi = ['EB'],
    # mesh_color=(1,1,1,0),
    skeleton_mode = 'tube',
    synapse_mode = 'cone',
    legend_mode = 'normal',
    merge_neurons=True,
    skeleton_mesh_simplification=0.9,
    roi_mesh_simplification=0.95,
    
    show_fig = True,
    brain_mesh = 'whole',
    cache_neurons=True,
    cache_synapses=True,
)

vs.plot_neurons()
# Use lower scale (1) for faster rendering of large scenes
# vs.export_video(fps=30, scale=1)
