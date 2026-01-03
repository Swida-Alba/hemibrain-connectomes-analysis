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
    output_dir='../local_data/plot_3d',
    neuron_layers = ['aMe12', 'MeVPLo2'], # or in the format: 'VA1d_adPN -> LHCENT3 -> MBON01'
    custom_layer_names = ['commissural_1', 'commissural_2'],  # Optional custom names for layers
    skip_synapse = True,
    neuron_alpha = 0.2,
    min_synapse_num = 3,
    synapse_size = 'real',
    synapse_alpha = 0.6,
    skeleton_mode = 'tube',
    synapse_mode = 'cone',
    merge_neurons=True,
    skeleton_mesh_simplification=0.9,
    roi_mesh_simplification=0.95,
    
    show_fig=True,
    
    mesh_roi = ['EB', 'LH'],
    # mesh_color=(1,1,1,0),
    brain_mesh='template',
    vnc_mesh=False, #
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

vs.export_video(
    fps=30, 
    degree_per_frame=1.0, 
    rotate='vertical', 
    scale=3,
    export_gif=True,      # Enable GIF export (default: True)
    gif_scale=0.2,        # GIF resolution scale (default: 0.2)
)
