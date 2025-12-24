import os
import sys
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List
import logging
import warnings
from contextlib import contextmanager

# Suppress FutureWarning from neuprint about Series.__getitem__
warnings.filterwarnings("ignore", category=FutureWarning, module="neuprint")

import numpy as np
import pandas as pd
import cv2
import navis
import navis.interfaces.neuprint as neu
from neuprint import Client, fetch_synapse_connections, SynapseCriteria, fetch_meta
import plotly.graph_objects as go
import bokeh.palettes

# Local imports
try:
    import statvis as sv
    import FAFB_file_converter
    import BANC_file_converter
except ImportError:
    # Fallback for when running from different context
    from . import statvis as sv
    from . import FAFB_file_converter
    from . import BANC_file_converter

@dataclass
class VisualizeSkeleton:
    '''3-D visualize skeleton with synapses and brain roi meshes'''
    
    backend: str = 'plotly'
    '''
    visualization backend: 'plotly' (default) or 'k3d'
    'plotly': interactive HTML with plotly (good for small/medium scenes)
    'k3d': WebGL-based, faster for large scenes, supports binary export
    '''

    dataset: str = 'hemibrain:v1.2.1'
    '''dataset to use, default is hemibrain:v1.2.1'''

    client_type: str = 'neuprint'
    '''client type: 'neuprint' (default) or 'flywire' '''

    client_flywire: object = None
    '''flywire client instance'''

    client: object = None
    '''neuprint client instance (optional, to reuse existing client)'''

    server: str = 'https://neuprint.janelia.org'
    '''the neuprint server to visit'''
    
    token: str = None
    '''neuprint auth token'''

    version: int | None = None
    '''Materialization version for FlyWire (e.g. 783). If None, uses default/latest.'''

    source_path: str = os.path.dirname(os.path.abspath(__file__))
    '''absolute path to the src/ directory where coana.py is located'''
    
    script_path: str = os.path.dirname(source_path)
    '''absolute path to the project root directory (parent of src/)'''

    data_folder: str = os.path.join(os.path.expanduser('~'), 'connectome_analysis')
    '''
    folder to save all data (subfolders auto-generated based on neuron_layers)
    Default: ~/connectome_analysis/
    '''

    output_dir: str = None
    '''Directory to save output. If None, uses data_folder.'''

    verbose: bool | str = 'full'
    '''
    Verbosity level:
    'full' or True: Print all messages
    'simple': Print only essential progress
    False: Silent
    '''
    
    save_folder: str = ''
    '''
    folder to save the current data (auto-generated from neuron_layers)
    # initialized in __post_init__
    # You can set the "saveas" parameter to customize the folder name'''

    neuron_layers: str | list = ''
    '''
    layers of neurons to plot, can be:
        list of neuron layers: e.g. ['L1', 'L2', 'L3']; or \n
        str of neuron layers separated by '->': \n
        e.g. 'L1->L2->L3'. All type, instance (in regular expression), and bodyId are compatible.\n
    when use list, each layer can be neuron bodyIds, types, instances in regular expressions, or a list of them\n
    e.g. [['L1_0','L1_1'], ['L2_0','L2_1'], ['L3_0','L3_1']]
    '''

    custom_layer_names: list = field(default_factory=list)

    min_synapse_num: int = 10
    '''minimum number of synapses to fetch and plot'''

    saveas: str = None
    '''filename to save the plot, if an absolute path is given, ignore data_folder'''

    neuron_colors: tuple = bokeh.palettes.Category10[10]
    '''
    colors of neuron layers to plot \n
    list of colors, each item for each layer, i.e., item i for layer i, and item i can be a list of colors for each neuron in layer i, or a single color for all neurons in layer i \n
    if you want to assign different colors to different neurons in the same layer, the color list should be the same length as the number of neurons in the layer. \n
    color format: 'red', '#ff0000', (255,0,0), or a dict mapping bodyId to color, {bodyId: color}. \n
    See https://navis.readthedocs.io/en/latest/source/tutorials/generated/navis.plot3d.html#navis.plot3d for more details.
    '''

    neuron_alpha: float = 0.3
    '''alpha of neuron, only works when the radius of neuron exists (show_skeleton_radius=True)'''

    synapse_colors: tuple = bokeh.palettes.Category10[10]
    '''colors of synapse layers to plot'''

    synapse_size: int | str = 1
    '''
    size of synapse\n
    when synapse_mode='scatter', 1 to 10 is recommended\n
    when synapse_mode='sphere', 100 is recommended\n
    can be 'real' to use the real distance between pre- and post-synaptic sites (only for sphere/cone/tetrahedron)\n
    '''

    synapse_criteria: SynapseCriteria = None
    '''criteria to filter synapses'''

    synapse_mode: str = 'scatter'
    '''
    mode to plot synapses, 'scatter', 'sphere', 'cone', or 'tetrahedron'\n
    'scatter': plot synapses as scatter points, relative size to the view\n
    'sphere': plot synapses as spheres, absolute size in the figure \n
    'cone': plot synapses as cones pointing from pre to post\n
    'tetrahedron': plot synapses as tetrahedrons pointing from pre to post\n
    '''
    
    synapse_alpha: float = 0.6
    '''alpha of synapse, only works when synapse_mode='sphere' '''

    mesh_roi: list = field(default_factory=list)
    '''
    meshes of brain ROIs to plot\n
    defaultly use ['LH(R)', 'AL(R)', 'EB'] to mark the position of the brain\n
    if you want to show the whole brain or hemibrain, see brain_mesh parameter. \n
    hide all meshes by setting mesh_roi = None \n
    Available meshes: \n
    a'L(L) \n
    a'L(R) \n   
    AB(L) \n    
    AB(R) \n    
    AL(L)_ \n   
    AL(R) \n    
    alphaL(L) \n
    alphaL(R) \n
    AME(R) \n   
    AOTU(R) \n  
    ATL(L) \n   
    ATL(R) \n   
    AVLP(R) \n  
    b'L(L) \n   
    b'L(R) \n   
    bL(L) \n    
    bL(R) \n    
    BU(L) \n    
    BU(R) \n    
    CA(L) \n    
    CA(R) \n    
    CAN(R) \n   
    CRE(L) \n   
    CRE(R) \n   
    EB \n       
    EPA(L) \n   
    EPA(R) \n
    FB \n
    FLA(R) \n
    gL(L) \n
    gL(R) \n
    GNG \n
    GOR(L) \n
    GOR(R) \n
    IB \n
    ICL(L) \n
    ICL(R) \n
    IPS(R) \n
    LAL(L) \n
    LAL(R) \n
    LH(R) \n
    LO(R) \n
    LOP(R) \n
    ME(R) \n
    NO \n
    PB \n
    PED(R) \n
    PLP(R) \n
    PRW \n
    PVLP(R) \n
    SAD \n
    SCL(L) \n
    SCL(R) \n
    SIP(L) \n
    SIP(R) \n
    SLP(R) \n
    SMP(L) \n
    SMP(R) \n
    SPS(L) \n
    SPS(R) \n
    VES(L) \n
    VES(R) \n
    WED(R) \n
    '''

    mesh_color: tuple | list = (100, 100, 100, 0.1)
    '''
    color of brain meshes, single color or list of colors matching the length of mesh_roi
    single color: tuple including an alpha channel: (R, G, B, alpha)
    multiple colors: list of tuples, each tuple including an alpha channel: [(R1, G1, B1, alpha1), (R2, G2, B2, alpha2), ...]
    '''

    merge_neurons: bool = True
    '''
    Whether to merge all neurons of the same type (layer) into a single 3D object.
    True: Merge neurons into one mesh (tube mode) or trace (line mode).
          Significantly reduces file size and rendering overhead for large populations.
    False: Plot each neuron individually (default).
    '''

    mirror_on_contralateral: bool = False
    '''
    Whether to mirror neurons and ROIs to the contralateral hemisphere.
    True: Mirror neurons and ROIs (e.g. 'ME(R)' -> 'ME(L)') to the other side.
          Useful for visualizing the full brain structure from hemibrain data.
    False: Only show the original data (default).
    '''

    skeleton_mesh_simplification: float = 0.9
    '''
    Mesh simplification factor for neuron skeletons (0.0 to 1.0).
    Only applies when skeleton_mode='tube'.
    0.0: No simplification (keep all faces).
    0.8: Remove 80% of faces (keep 20%).
    Higher values reduce file size but may lose detail.
    Recommended: 0.5 - 0.9 for large populations.
    '''

    roi_mesh_simplification: float = 0.95
    '''
    Mesh simplification factor for ROI meshes (0.0 to 1.0).
    0.0: No simplification (keep all faces).
    0.9: Remove 90% of faces (keep 10%).
    Higher values reduce file size but may lose detail.
    Recommended: 0.9 - 0.99 for large ROI meshes.
    '''

    show_soma: bool = True
    '''whether to show soma'''

    show_fig: bool = True
    '''whether to show the figure'''

    skeleton_mode: str = 'tube'
    '''
    whether to plot the radius of skeleton or only skeleton lines\n
    'tube': plot the radius of skeleton\n
    'line': only plot skeleton lines\n
    when 'line', the file size will be significantly smaller and the rendering will be faster
    '''

    show_connectors: bool = False
    '''whether to fetch and plot the connectors, all pre- and post-synaptic sites of the neurons, for single layer of neurons'''

    skip_synapse: bool = False
    '''
    whether to skip synapse fetching and plotting between layers
    True: skip all synapse operations (faster initialization, smaller file size)
    False: fetch and plot synapses between layers (default behavior)
    Note: This only affects inter-layer synapses, not show_connectors (neuron connectors)
    '''

    use_size_slider: bool = False
    '''
    whether to use size slider to adjust the size of synapses\n
    only works when synapse_mode='scatter'
    '''

    legend_mode: str = 'normal'
    '''
    'normal': show legend for individual neurons, requires `merge_neurons=False`\n
    'merge': merge all neurons in the same layer and show legend for each layer\n
    '''
    
    transforms_dir: str = '~/flybrain-data'
    '''
    Directory for brain transform files (used by flybrains package)\n
    Default: ~/flybrain-data (flybrains default location)\n
    To use a custom location:\n
    1. Set this attribute to your preferred path\n
    2. Ensure the flybrains package uses this path\n
    Note: Changing this requires setting the FLYBRAINS_DATA environment variable\n
    before importing flybrains, or manually moving existing transform files.\n
    '''
    
    cache_neurons: bool = False
    '''
    Whether to cache fetched neuron skeletons to disk\n
    True: Save fetched skeletons as individual {bodyId}.pkl files to cache/{dataset}/skeletons/\n
    False: Fetch from NeuPrint every time (default)\n
    Cache location: cache/{dataset}/skeletons/{bodyId}.pkl\n
    Individual files allow better reuse across different neuron layer selections.\n
    '''
    
    cache_synapses: bool = False
    '''
    Whether to cache fetched synapse data to disk\n
    True: Use synapse table from datasets/{dataset}/*_synapse_table.parquet if available\n
    False: Fetch from NeuPrint every time (default)\n
    For FlyWire/FAFB: Always uses datasets/{dataset}/flywire_FAFB_v783_synapse_table.parquet\n
    '''
    
    brain_mesh: str = 'none'
    ''' 
    Brain/VNC mesh visualization options (dataset-specific):\n
    - 'none': Only plot meshes specified in mesh_roi parameter\n
    - 'template': Plot the dataset's native template mesh (EM resolution)\n
      • hemibrain → JRCFIB2018F (hemibrain only)\n
      • optic-lobe → JRCFIB2018F (optic lobe region)\n
      • manc → MANC (male adult nerve cord VNC)\n
      • male-cns → JRCFIB2022M (full male CNS: brain + VNC)\n
    - 'whole': Plot standard whole-brain/VNC envelope mesh\n
      • hemibrain/optic-lobe → JRC2018F (requires transforms)\n
      • manc → MANC VNC envelope (no transform needed)\n
      • male-cns → JRCFIB2022M CNS envelope (no transform needed)\n
    - 'hemi': HEMIBRAIN ONLY - Plot hemisphere mesh (left or right)\n
      • Only works with hemibrain:v1.2.1 dataset\n
      • VNC datasets (manc, male-cns) do not support hemisphere mode\n
      • manc → MANC template (native VNC)\n
      • male-cns → JRCFIB2022M (native full CNS)\n
    Note: Some transforms require download (~500MB, one-time)\n
    See https://github.com/navis-org/navis-flybrains
    '''
    
    brain_mesh_color: str = 'rgba(200, 230, 240, 0.1)'
    ''' 
    Color of the brain/VNC mesh, works with brain_mesh = 'template' or 'whole'\n
    Format: 'rgba(r, g, b, a)' where a=transparency (0=transparent, 1=opaque)\n
    Example: 'rgba(200, 230, 240, 0.1)' for light blue semi-transparent\n
    See https://plotly.com/python/discrete-color/
    '''

    def list_available_rois(self, refresh=False, fetch_online=True):
        """List all available ROIs for the current dataset.
        
        Parameters
        ----------
        refresh : bool
            If True, force refresh from NeuPrint API. If False, use cached data if available.
        fetch_online : bool
            If True, attempt to fetch from NeuPrint online database. If False, only use local cache.
        
        Returns
        -------
        list
            Sorted list of available ROI names.
        
        Examples
        --------
        >>> vs = VisualizeSkeleton(dataset='hemibrain:v1.2.1', neuron_layers=['EB'])
        >>> available_rois = vs.list_available_rois()
        >>> print(f"Found {len(available_rois)} available ROIs")
        >>> print(available_rois[:10])  # Show first 10 ROIs
        
        >>> # Force refresh from online database
        >>> fresh_rois = vs.list_available_rois(refresh=True, fetch_online=True)
        """
        self._vprint(f'\\n' + '='*70)
        self._vprint(f'Available ROIs for {self.dataset}')
        self._vprint('='*70)
        
        rois = self._get_available_rois(use_cache=not refresh, fetch_online=fetch_online)
        
        if rois:
            self._vprint(f'\\n📊 Total: {len(rois)} ROIs')
            self._vprint(f'\\n🔹 First 30 ROIs:')
            for i in range(0, min(30, len(rois)), 5):
                self._vprint('  ' + ', '.join(rois[i:i+5]))
            if len(rois) > 30:
                self._vprint(f'  ... and {len(rois) - 30} more')
            self._vprint(f'\\n💡 Use these ROI names in the mesh_roi parameter')
            self._vprint('='*70)
        else:
            self._vprint('⚠️  No ROIs found')
            self._vprint('='*70)
        
        return rois
    
    def _vprint(self, msg, level='simple', **kwargs):
        """
        Print message based on verbosity level.
        level: 'simple' (default) or 'full'
        """
        if not self.verbose:
            return
        
        # If verbose is 'simple', only print 'simple' messages
        if self.verbose == 'simple' and level == 'full':
            return
            
        # If verbose is 'full', print everything
        print(msg, **kwargs)

    @contextmanager
    def _suppress_output(self):
        """Suppress stdout and stderr if verbose is not full."""
        if self.verbose == 'full':
            yield
        else:
            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = devnull
                sys.stderr = devnull
                try:  
                    yield
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr

    def __post_init__(self):
        # Normalize verbose parameter
        if self.verbose is True:
            self.verbose = 'full'
        elif self.verbose is False:
            self.verbose = False
        
        # Silence navis and other libraries if verbose is not full
        if self.verbose != 'full':
            logging.getLogger('navis').setLevel(logging.ERROR)
            logging.getLogger('trimesh').setLevel(logging.ERROR)
        
        # Initialize output_dir if not set
        if self.output_dir is None:
            self.output_dir = self.data_folder

        # Initialize list to store meshes for export
        self.exportable_meshes = []
        
        # Auto-detect client_type from dataset if not explicitly set to flywire
        if self.client_type == 'neuprint' and ('flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower()):
            self.client_type = 'flywire'
            self._vprint(f"Auto-detected client_type='flywire' from dataset '{self.dataset}'", level='full')

        # Force disable caching for FlyWire/FAFB
        if self.client_type == 'flywire' or 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower():
            if self.cache_neurons:
                self._vprint(" Disabling neuron skeleton caching for FlyWire/FAFB (files too large)", level='full')
                self.cache_neurons = False
            if self.cache_synapses:
                self._vprint(" Disabling synapse caching for FlyWire/FAFB (files too large)", level='full')
                self.cache_synapses = False

        # Auto-detect version from dataset if not provided
        if self.client_type == 'flywire' and self.version is None:
            import re
            # Look for v783 or version 783
            match = re.search(r'v(\d+)', self.dataset)
            if match:
                self.version = int(match.group(1))
                self._vprint(f"Auto-detected version={self.version} from dataset '{self.dataset}'", level='full')

        # Initialize client if needed
        if self.client_type == 'neuprint':
            import neuprint
            
            # Use provided client if available
            if self.client is not None:
                self._vprint(f'Using provided client for {self.dataset}', level='full')
            else:
                # Check if global client exists
                client_exists = False
                try:
                    if neuprint.default_client() is not None:
                        client_exists = True
                except RuntimeError:
                    pass

                if not client_exists:
                    if self.token:
                        self.client = Client(self.server, dataset=self.dataset, token=self.token)
                        self.client.fetch_version()
                        self._vprint(f'Client initialized for {self.dataset}', level='full')
                    elif os.environ.get('NEUPRINT_APPLICATION_CREDENTIALS'):
                        # Auto-detect from env
                        self.client = Client(self.server, dataset=self.dataset)
                        self.client.fetch_version()
                        self._vprint(f'Client initialized from env for {self.dataset}', level='full')
                    else:
                        # Only warn if we are not using local cache/files exclusively
                        # But we don't know that yet.
                        pass
        
        # Initialize FlyWire client if needed
        if self.client_type == 'flywire' and self.client_flywire is None:
            # FlyWire API fetching removed
            pass

        # Check FlyWire visualization files
        if 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower():
            # Ensure data is prepared using the converter
            dataset_dir = os.path.join(self.script_path, 'datasets', self.dataset)
            
            # Use the converter module to ensure data is ready
            if 'BANC' in self.dataset:
                success = BANC_file_converter.ensure_banc_data(self.dataset, dataset_dir)
            else:
                success = FAFB_file_converter.ensure_flywire_data(self.dataset, dataset_dir)

            if not success:
                print("\\n\033[31mCRITICAL ERROR: FlyWire/BANC data preparation failed.\033[0m")
                print("Please follow the instructions above to download the required files.")
                sys.exit(1)
            
            try:
                import fafb_utils
                # Check for skeleton zip
                if not os.path.exists(dataset_dir):
                    dataset_dir = os.path.join(self.script_path, 'datasets', 'flywire_FAFB_v783')
                
                if os.path.exists(dataset_dir):
                    sk_zip = fafb_utils.get_fafb_skeleton_zip(dataset_dir)
                    if not sk_zip:
                        print(f'\033[31mWarning: FlyWire skeleton zip not found in {dataset_dir}\033[0m')
                        if 'BANC' in self.dataset:
                            print('Skeleton visualization not available for BANC, because the skeleton data not available in flywire codex')
                        else:
                            print(f'Please download sk_lod1_783_healed.zip from https://codex.flywire.ai/api/download?dataset=fafb')
                        print(f'Visualization might fail or be incomplete.')
                        sys.exit(0)
            except ImportError:
                pass

        if self.synapse_mode not in ['scatter', 'sphere', 'cone', 'tetrahedron']:
            raise ValueError('synapse_mode can only be "scatter", "sphere", "cone", or "tetrahedron"')
        if self.legend_mode not in ['normal', 'merge']:
            raise ValueError('legend_mode can only be "normal" or "merge"')
        if self.skeleton_mode not in ['line','tube']:
            raise ValueError('skeleton_mode can only be "line" or "tube"')
        if self.brain_mesh not in ['none', 'whole', 'template']:
            raise ValueError('brain_mesh must be "none", "template", or "whole"')
        if self.backend not in ['plotly', 'k3d']:
            raise ValueError('backend must be "plotly" or "k3d"')
        
        # Check brain transforms early if brain_mesh='whole' is requested
        # Only some datasets require transforms
        if self.brain_mesh == 'whole':
            needs_transform = self._dataset_needs_transform()
            if needs_transform and not self._check_and_download_transforms():
                self.brain_mesh = 'none'
                self._vprint('⚠️  brain_mesh reset to "none" due to missing transforms', level='full')
        
        # convert neuron_layers str to list, if is str
        if type(self.neuron_layers) is str:
            self.neuron_layers = self.neuron_layers.replace(' ','').split('->')
            for i,layer in enumerate(self.neuron_layers): # convert bodyId str to int
                if layer.isnumeric():
                    self.neuron_layers[i] = int(layer)
        
        if self.synapse_mode == 'scatter' and self.synapse_size == 0:
            self.synapse_size = 2
        elif self.synapse_mode in ['sphere', 'cone', 'tetrahedron']:
            # Only check size limit if synapse_size is a number (not 'real')
            if isinstance(self.synapse_size, (int, float)) and self.synapse_size < 20 and self.brain_mesh != 'whole':
                self.synapse_size = 20
                self._vprint('\033[33mSynapse size is too small (< 20) for sphere, cone, or tetrahedron mode, automatically reset to 20\033[0m', level='full')
            if self.use_size_slider:
                self.use_size_slider = False
                self._vprint('\033[33msize slider is only available for synapse_mode="scatter", automatically reset use_size_slider to False\033[0m', level='full')
            
        if self.mesh_roi == None:
            self.mesh_roi = []
        
        if len(self.neuron_layers) <= len(self.neuron_colors): 
            self.neuron_colors = self.neuron_colors[:len(self.neuron_layers)]
            self.synapse_colors = self.synapse_colors[:len(self.neuron_layers)-1]

        # Validate brain_mesh options
        if self.brain_mesh == 'hemi':
            if 'hemibrain' not in self.dataset.lower():
                self._vprint('\033[33m⚠️  brain_mesh="hemi" only works with hemibrain:v1.2.1 dataset', level='full')
                self._vprint('   VNC datasets (manc, male-cns) do not support hemisphere mode', level='full')
                self._vprint('   Automatically switching to brain_mesh="whole"\033[0m', level='full')
                self.brain_mesh = 'whole'
        
        if self.skeleton_mode == 'line':
            self.show_skeleton_radius = False
            # neuron_alpha is now supported for line mode via opacity
        elif self.skeleton_mode == 'tube':
            self.show_skeleton_radius = True
        
        # fetch neurons and automatically generate layer names
        self.neuron_dfs = []
        self.roi_dfs = []
        self.layer_criteria = []
        self.layer_names = []
        for i in range(len(self.neuron_layers)):
            self._vprint(f'fetching neuron info of layer {i}...', level='full')
            layer_input = self.neuron_layers[i]
            if not isinstance(layer_input, list):
                layer_input = [layer_input]
            ndf, rdf, auto_name, cri = sv.getNeurons(layer_input, dataset=self.dataset, client=self.client)
            self.neuron_dfs.append(ndf)
            self.roi_dfs.append(rdf)
            self.layer_criteria.append(cri)
            self.layer_names.append(auto_name)
        self._vprint('Fetched neuron layers', level='full')

        # Generate smart layer names based on types (if not using custom names)
        if not self.custom_layer_names:
            self.layer_names = self._generate_smart_layer_names()
        else:
            self.layer_names = self.custom_layer_names
            
        if self.saveas is None:
            self.saveas = '_'.join(self.layer_names)
        
        # Create timestamped subfolder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_folder = os.path.join(self.output_dir, 'plot3d_' + self.saveas.split('.')[0] + '_' + timestamp)
        if not os.path.exists(self.save_folder): os.makedirs(self.save_folder)
        
        # Save parameters to text file with improved formatting
        param_file = os.path.join(self.save_folder, 'parameters.txt')
        with open(param_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("VisualizeSkeleton Parameters\n")
            f.write("=" * 60 + "\n\n")
            
            # Basic Info
            f.write("[Basic Info]\n")
            f.write(f"  Dataset:     {self.dataset}\n")
            f.write(f"  Timestamp:   {timestamp}\n")
            if self.version:
                f.write(f"  Version:     {self.version}\n")
            f.write(f"  Client Type: {self.client_type}\n")
            f.write("\n")
            
            # Layer Info
            f.write("[Layers]\n")
            for i, (layer, name) in enumerate(zip(self.neuron_layers, self.layer_names)):
                n_neurons = len(self.neuron_dfs[i]) if i < len(self.neuron_dfs) and self.neuron_dfs[i] is not None else 0
                f.write(f"  Layer {i}: {name} ({n_neurons} neurons)\n")
                # Show first few neuron IDs if available
                if n_neurons > 0 and 'bodyId' in self.neuron_dfs[i].columns:
                    body_ids = self.neuron_dfs[i]['bodyId'].tolist()[:5]
                    ids_str = ', '.join(str(bid) for bid in body_ids)
                    if n_neurons > 5:
                        ids_str += f", ... (+{n_neurons - 5} more)"
                    f.write(f"           IDs: {ids_str}\n")
            f.write("\n")
            
            # Visualization Settings
            f.write("[Visualization]\n")
            f.write(f"  Skeleton Mode:   {self.skeleton_mode}\n")
            f.write(f"  Backend:         {self.backend}\n")
            f.write(f"  Brain Mesh:      {self.brain_mesh}\n")
            if self.mesh_roi:
                f.write(f"  Mesh ROI:        {self.mesh_roi}\n")
            f.write("\n")
            
            # Synapse Settings
            f.write("[Synapse Settings]\n")
            f.write(f"  Synapse Mode:    {self.synapse_mode}\n")
            f.write(f"  Synapse Size:    {self.synapse_size}\n")
            f.write(f"  Min Synapse Num: {self.min_synapse_num}\n")
            f.write("\n")
            
            f.write("=" * 60 + "\n")
        
        if self.backend == 'plotly':
            self.fig_3d = go.Figure()
        elif self.backend == 'k3d':
            try:
                import k3d
                self.fig_3d = k3d.plot()
            except ImportError:
                self._vprint("⚠️  k3d not installed. Please install it with `pip install k3d`")
                self._vprint("   Falling back to plotly backend")
                self.backend = 'plotly'
                self.fig_3d = go.Figure()
        
        # save neuron dataframes to excel file
        file_path = os.path.join(self.save_folder, self.saveas+'_neuron_info.xlsx')
        for i in range(len(self.neuron_layers)):
            if i == 0:
                mode = 'w'
            else:
                mode = 'a'
            with pd.ExcelWriter(file_path,mode=mode,engine='openpyxl') as writer:
                self.neuron_dfs[i].to_excel(writer, sheet_name=f'neuron_df{i}')
                self.roi_dfs[i].to_excel(writer, sheet_name=f'roi_count_df{i}')
    
    def _get_cache_path(self, cache_type):
        """Get the cache directory for skeletons or synapses
        
        Uses project cache/ folder for organized storage:
        cache/{dataset}/skeletons/ - for individual skeleton .pkl files
        cache/{dataset}/synapses/ - for synapse cache files
        
        For datasets folder resources:
        datasets/{dataset}/*_synapse_table.parquet - synapse table
        
        Example:
        - cache/hemibrain_v1_2_1/skeletons/{bodyId}.pkl
        - datasets/flywire_FAFB_v783/flywire_FAFB_v783_synapse_table.parquet
        """
        dataset_normalized = self.dataset.replace(':', '_').replace('.', '_')
        cache_dir = os.path.join(self.script_path, 'cache', dataset_normalized, cache_type)
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    
    def _generate_smart_layer_names(self) -> List[str]:
        """Generate smart layer names based on neuron types.
        
        For each layer, generates a name in format:
        - {type} if all neurons in layer are the same type (even if multiple neurons)
        - {type}_etc if multiple neurons with different types (uses most common type)
        - {bodyId} if single untyped neuron
        - {bodyId}_etc if multiple untyped neurons with different IDs
        
        This method looks at the neuron_dfs to determine types and counts.
        
        Returns:
            List[str]: Smart layer names for each layer
        """
        smart_names = []
        
        for i, ndf in enumerate(self.neuron_dfs):
            if ndf is None or len(ndf) == 0:
                # Fallback to original auto-generated name
                smart_names.append(self.layer_names[i] if i < len(self.layer_names) else f"layer_{i}")
                continue
            
            n_neurons = len(ndf)
            
            # Get type column (different datasets may use different column names)
            type_col = None
            for col in ['type', 'cell_type', 'neuronType']:
                if col in ndf.columns:
                    type_col = col
                    break
            
            # Get types from the dataframe
            if type_col and type_col in ndf.columns:
                types = ndf[type_col].dropna().unique().tolist()
                # Filter out empty strings and None
                types = [t for t in types if t and str(t).strip()]
            else:
                types = []
            
            if types:
                # Count unique types
                n_unique_types = len(types)
                
                # Use the most common type as the representative
                if type_col in ndf.columns:
                    type_counts = ndf[type_col].value_counts()
                    primary_type = type_counts.index[0] if len(type_counts) > 0 else types[0]
                else:
                    primary_type = types[0]
                
                # Only add _etc if there are multiple different types
                if n_unique_types > 1:
                    smart_names.append(f"{primary_type}_etc")
                else:
                    # All neurons are the same type
                    smart_names.append(str(primary_type))
            else:
                # No type info - use bodyId
                body_ids = ndf['bodyId'].tolist() if 'bodyId' in ndf.columns else []
                if body_ids:
                    first_id = body_ids[0]
                    # Multiple untyped neurons with different IDs -> _etc
                    if n_neurons > 1:
                        smart_names.append(f"{first_id}_etc")
                    else:
                        smart_names.append(str(first_id))
                else:
                    # Ultimate fallback
                    smart_names.append(self.layer_names[i] if i < len(self.layer_names) else f"layer_{i}")
        
        return smart_names

    def _get_synapse_table_path(self):
        """Get path to synapse table in datasets folder.
        
        Returns the path to the synapse table parquet file.
        For FlyWire/FAFB: datasets/flywire_FAFB_v783/flywire_FAFB_v783_synapse_table.parquet
        For NeuPrint: datasets/{dataset}/{dataset}_synapse_table.parquet
        
        Returns:
            str: Path to synapse table, or None if not found
        """
        dataset_normalized = self.dataset.replace(':', '_').replace('.', '_')
        datasets_dir = os.path.join(self.script_path, 'datasets', dataset_normalized)
        
        # Look for synapse table file
        parquet_file = os.path.join(datasets_dir, f"{dataset_normalized}_synapse_table.parquet")
        
        if os.path.exists(parquet_file):
            return parquet_file
        
        # Fallback: try FAFB naming if dataset includes flywire
        if 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower():
            fafb_dir = os.path.join(self.script_path, 'datasets', 'flywire_FAFB_v783')
            fafb_file = os.path.join(fafb_dir, 'flywire_FAFB_v783_synapse_table.parquet')
            if os.path.exists(fafb_file):
                return fafb_file
        
        return None
    
    def _load_cached_neurons(self, neuron_df):
        """Load cached neuron skeletons if available.
        
        Loads individual {bodyId}.pkl files from cache/{dataset}/skeletons/
        
        Returns:
            navis.NeuronList or None if no cached neurons found
        """
        if not self.cache_neurons:
            return None
        
        cache_dir = self._get_cache_path('skeletons')
        body_ids = neuron_df['bodyId'].tolist()
        
        import pickle
        neurons = []
        loaded_ids = []
        missing_ids = []
        
        for bid in body_ids:
            cache_file = os.path.join(cache_dir, f'{bid}.pkl')
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        neuron = pickle.load(f)
                    neurons.append(neuron)
                    loaded_ids.append(bid)
                except Exception as e:
                    self._vprint(f'  ⚠ Failed to load cached skeleton {bid}: {e}')
                    missing_ids.append(bid)
            else:
                missing_ids.append(bid)
        
        if neurons:
            self._vprint(f'  ✓ Loaded {len(neurons)} neurons from cache', level='full')
            if missing_ids:
                self._vprint(f'  ℹ  {len(missing_ids)} neurons not in cache, will fetch', level='full')
            # Return loaded neurons plus info about missing ones
            return navis.NeuronList(neurons), missing_ids
        
        return None, body_ids  # All missing
    
    def _save_cached_neurons(self, neuron_df, neuron_vols):
        """Save neuron skeletons to cache as individual {bodyId}.pkl files.
        
        Saves each neuron as a separate file for better reusability.
        """
        if not self.cache_neurons:
            return
        
        cache_dir = self._get_cache_path('skeletons')
        
        import pickle
        saved_count = 0
        
        # Handle both NeuronList and list of neurons
        if hasattr(neuron_vols, '__iter__'):
            for neuron in neuron_vols:
                try:
                    # Get bodyId from neuron
                    bid = getattr(neuron, 'id', None) or getattr(neuron, 'bodyId', None)
                    if bid is None:
                        continue
                    
                    cache_file = os.path.join(cache_dir, f'{bid}.pkl')
                    
                    # Skip if already cached
                    if os.path.exists(cache_file):
                        continue
                    
                    with open(cache_file, 'wb') as f:
                        pickle.dump(neuron, f)
                    saved_count += 1
                except Exception as e:
                    self._vprint(f'  ⚠ Failed to save skeleton {bid}: {e}')
        
        if saved_count > 0:
            self._vprint(f'  💾 Saved {saved_count} new neurons to cache', level='full')
    
    def plot_skeleton(self):
        for i in range(len(self.neuron_layers)):
            self._vprint(f'fetching skeletons of layer {i}...', level='full')
            
            # Try to load from cache first
            cache_result = self._load_cached_neurons(self.neuron_dfs[i])
            
            cached_neurons = None
            missing_ids = self.neuron_dfs[i]['bodyId'].tolist()  # Default: all missing
            
            if cache_result is not None:
                cached_neurons, missing_ids = cache_result
            
            neuron_vols = None
            
            # Fetch missing neurons
            if missing_ids:
                # Special handling for FAFB local data
                if 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower():
                    try:
                        import fafb_utils
                        project_root = os.path.dirname(os.path.dirname(__file__))
                        
                        # Try to find dataset directory by name
                        data_dir = os.path.join(project_root, "datasets", self.dataset)
                        if not os.path.exists(data_dir):
                            data_dir = os.path.join(project_root, "datasets", "flywire_FAFB_v783")
                        
                        # Load from Zip
                        zip_path = fafb_utils.get_fafb_skeleton_zip(data_dir)
                        
                        if zip_path:
                            self._vprint(f"  Loading skeletons from Zip: {zip_path}...", level='full')
                            
                            import zipfile
                            import io
                            
                            neurons = []
                            with zipfile.ZipFile(zip_path, 'r') as z:
                                for bid in missing_ids:
                                    filename = f"{bid}.swc"
                                    try:
                                        # Check if file exists in zip
                                        if filename in z.namelist():
                                            with z.open(filename) as f:
                                                # Read content
                                                content = f.read().decode('utf-8')
                                                # Parse with navis
                                                n = navis.read_swc(io.StringIO(content))
                                                n.units = 'nm' # Explicitly set units for FAFB
                                                n.id = bid
                                                n.name = str(bid)
                                                neurons.append(n)
                                        else:
                                            self._vprint(f"    Warning: Skeleton {filename} not found in zip", level='full')
                                    except Exception as e:
                                        self._vprint(f"    Error reading {filename}: {e}", level='full')
                            
                            if neurons:
                                neuron_vols = navis.NeuronList(neurons)
                                self._vprint(f"  ✓ Loaded {len(neurons)} skeletons from local zip", level='full')
                    except ImportError:
                        pass
                    except Exception as e:
                        self._vprint(f"  Warning: Error loading local FAFB skeletons: {e}", level='full')

                # Fetch from API if not loaded locally
                if neuron_vols is None and missing_ids:
                    if self.client_type == 'flywire' and self.client_flywire:
                        # Filter neuron_df to only missing IDs
                        missing_df = self.neuron_dfs[i][self.neuron_dfs[i]['bodyId'].isin(missing_ids)]
                        neuron_vols = self.client_flywire.fetch_skeletons(self.layer_criteria[i], with_synapses=self.show_connectors)
                    else:
                        # Fetch from NeuPrint - filter to missing IDs only
                        missing_df = self.neuron_dfs[i][self.neuron_dfs[i]['bodyId'].isin(missing_ids)]
                        if not missing_df.empty:
                            # Pass client explicitly if available
                            kwargs = {'with_synapses': self.show_connectors}
                            if self.client:
                                kwargs['client'] = self.client
                            neuron_vols = neu.fetch_skeletons(missing_df, **kwargs)
                
                # Save newly fetched neurons to cache
                if neuron_vols is not None:
                    self._save_cached_neurons(self.neuron_dfs[i], neuron_vols)
            
            # Combine cached and newly fetched neurons
            if cached_neurons is not None and neuron_vols is not None:
                # Combine both lists
                all_neurons = list(cached_neurons) + list(neuron_vols)
                neuron_vols = navis.NeuronList(all_neurons)
            elif cached_neurons is not None:
                neuron_vols = cached_neurons
            # else neuron_vols is already set from fetch

            # Normalize to NeuronList so downstream len()/iteration works for single TreeNeuron
            if neuron_vols is not None and not isinstance(neuron_vols, (list, navis.NeuronList)):
                neuron_vols = navis.NeuronList([neuron_vols])
            
            if neuron_vols is None or len(neuron_vols) == 0:
                self._vprint(f'⚠️  Failed to fetch skeletons for layer {i}', level='full')
                continue

            if self.brain_mesh in ['whole', 'template']:
                template_info = self._get_template_info()
                self._vprint(f'Transforming skeletons of layer {i} to {template_info["mesh_name"]}...', end='', level='full')
                try:
                    # Ensure float64 coordinates to avoid dtype warnings in navis
                    if isinstance(neuron_vols, (list, navis.NeuronList)):
                        for n in neuron_vols:
                            if hasattr(n, 'nodes') and isinstance(n.nodes, pd.DataFrame):
                                for col in ['x', 'y', 'z']:
                                    if col in n.nodes.columns:
                                        n.nodes[col] = n.nodes[col].astype('float64')
                                # Print range for first neuron
                                if n == neuron_vols[0]:
                                    self._vprint(f"  Skeleton coords range (nm): X[{n.nodes.x.min():.1f}, {n.nodes.x.max():.1f}], Y[{n.nodes.y.min():.1f}, {n.nodes.y.max():.1f}], Z[{n.nodes.z.min():.1f}, {n.nodes.z.max():.1f}]", level='full')
                    elif hasattr(neuron_vols, 'nodes') and isinstance(neuron_vols.nodes, pd.DataFrame):
                         for col in ['x', 'y', 'z']:
                            if col in neuron_vols.nodes.columns:
                                neuron_vols.nodes[col] = neuron_vols.nodes[col].astype('float64')
                         self._vprint(f"  Skeleton coords range (nm): X[{neuron_vols.nodes.x.min():.1f}, {neuron_vols.nodes.x.max():.1f}], Y[{neuron_vols.nodes.y.min():.1f}, {neuron_vols.nodes.y.max():.1f}], Z[{neuron_vols.nodes.z.min():.1f}, {neuron_vols.nodes.z.max():.1f}]", level='full')

                    with self._suppress_output():
                        neuron_vols = navis.xform_brain(neuron_vols, source=template_info['source'], target=template_info['target'])
                except Exception as e:
                    self._vprint(f'\\n⚠️  Transforming skeletons failed: {e}', level='full')
                    if self._dataset_needs_transform() and not self._check_and_download_transforms():
                        self.brain_mesh = 'none'
                    else:
                        # Retry transformation after download
                        try:
                            with self._suppress_output():
                                neuron_vols = navis.xform_brain(neuron_vols, source=template_info['source'], target=template_info['target'])
                            self._vprint('✓ Transformation successful after download', level='full')
                        except Exception as retry_e:
                            self._vprint(f'⚠️  Transformation still failed: {retry_e}', level='full')
                            self._vprint('   Setting brain_mesh to "none"', level='full')
                            self.brain_mesh = 'none'
            
            # Ensure iterable after potential transforms (navis may return TreeNeuron)
            if neuron_vols is not None and not isinstance(neuron_vols, (list, navis.NeuronList)):
                neuron_vols = navis.NeuronList([neuron_vols])

            # Mirror neurons if requested
            if self.mirror_on_contralateral:
                self._vprint(f'Mirroring {len(neuron_vols)} neurons...', end='', level='full')
                try:
                    template = None
                    if self.brain_mesh == 'whole':
                        template_info = self._get_template_info()
                        template = template_info['target']
                    elif self.brain_mesh == 'template':
                         if 'hemibrain' in self.dataset or 'optic-lobe' in self.dataset:
                             template = 'JRCFIB2018F'
                         elif 'male-cns' in self.dataset:
                             template = 'JRCFIB2022M'
                    
                    if template:
                        mirrored = navis.mirror_brain(neuron_vols, template, mirror_axis='x')
                        if isinstance(neuron_vols, navis.NeuronList):
                            neuron_vols = neuron_vols + mirrored
                        else:
                            neuron_vols = navis.NeuronList([neuron_vols, mirrored])
                        self._vprint(' (mirrored) ', end='', level='full')
                    else:
                        self._vprint(' (mirror skipped: unknown template) ', end='', level='full')
                except Exception as e:
                    self._vprint(f' (mirror failed: {e})', end='', level='full')

            # Simplify individual neurons if requested (and not merging)
            # If merging is enabled, simplification is handled during the merge process
            if self.skeleton_mesh_simplification > 0 and self.skeleton_mode == 'tube' and not self.merge_neurons:
                self._vprint(f'Simplifying {len(neuron_vols)} neurons ({self.skeleton_mesh_simplification*100:.0f}%)...', end='', level='full')
                try:
                    import trimesh
                    simplified_neurons = []
                    # Ensure iterable
                    neurons_to_simplify = neuron_vols if isinstance(neuron_vols, navis.NeuronList) else [neuron_vols]
                    
                    for n in neurons_to_simplify:
                        try:
                            # Convert to mesh if needed (TreeNeuron -> MeshNeuron)
                            mesh_n = None
                            if isinstance(n, navis.TreeNeuron):
                                # Fix radii if needed
                                if hasattr(n, 'nodes') and 'radius' in n.nodes.columns:
                                    invalid_mask = (n.nodes['radius'] <= 0) | (n.nodes['radius'].isna())
                                    if invalid_mask.any():
                                        n.nodes.loc[invalid_mask, 'radius'] = 1
                                elif hasattr(n, 'nodes'):
                                    n.nodes['radius'] = 1
                                
                                # Convert
                                if hasattr(navis, 'conversion') and hasattr(navis.conversion, 'tree2meshneuron'):
                                    mesh_n = navis.conversion.tree2meshneuron(n)
                            elif isinstance(n, navis.MeshNeuron):
                                mesh_n = n
                                
                            # Simplify if we have a mesh neuron
                            if mesh_n and hasattr(mesh_n, 'trimesh'):
                                n_faces = len(mesh_n.trimesh.faces)
                                target_faces = int(n_faces * (1 - self.skeleton_mesh_simplification))
                                if target_faces < n_faces:
                                    # simplify_quadratic_decimation returns a new trimesh object
                                    mesh_n.trimesh = mesh_n.trimesh.simplify_quadratic_decimation(target_faces)
                                simplified_neurons.append(mesh_n)
                            else:
                                # Keep original if conversion failed or not applicable
                                simplified_neurons.append(n)
                        except Exception as e:
                            # print(f'Warning: Failed to simplify neuron {n.id}: {e}')
                            simplified_neurons.append(n) # Keep original if failed
                    
                    neuron_vols = navis.NeuronList(simplified_neurons)
                    self._vprint(' Done', level='full')
                except Exception as e:
                    self._vprint(f' (simplification failed: {e})', end='', level='full')

            # Merge neurons if requested (optimization)
            num_neurons = len(neuron_vols) if isinstance(neuron_vols, (list, navis.NeuronList)) else 1
            if self.merge_neurons and num_neurons > 1:
                self._vprint(f'Merging {num_neurons} neurons into single object...', end='', level='full')
                try:
                    if self.skeleton_mode == 'tube':
                        import trimesh
                        # navis.conversion is already available via 'import navis' at module level
                        
                        # Convert all neurons to meshes
                        meshes = []
                        neurons_to_merge = neuron_vols if isinstance(neuron_vols, navis.NeuronList) else [neuron_vols]
                        
                        for n in neurons_to_merge:
                            try:
                                # Fix missing radii to avoid navis warning
                                if hasattr(n, 'nodes') and 'radius' in n.nodes.columns:
                                    # Check for invalid radii (<= 0 or NaN)
                                    invalid_mask = (n.nodes['radius'] <= 0) | (n.nodes['radius'].isna())
                                    if invalid_mask.any():
                                        # Set default radius (e.g. 40 units) for visibility
                                        n.nodes.loc[invalid_mask, 'radius'] = 1
                                elif hasattr(n, 'nodes'):
                                    # If radius column missing entirely, create it
                                    n.nodes['radius'] = 1

                                # Convert to mesh (TreeNeuron -> MeshNeuron)
                                # Use navis.conversion.tree2meshneuron if available, or navis.MeshNeuron.from_tree
                                # Or simply navis.MeshNeuron(n) which might work
                                # Let's try navis.conversion.tree2meshneuron first as it's explicit
                                if hasattr(navis, 'conversion') and hasattr(navis.conversion, 'tree2meshneuron'):
                                    mesh_neuron = navis.conversion.tree2meshneuron(n)
                                else:
                                    # Fallback: try to create MeshNeuron directly or use other method
                                    # navis.MeshNeuron(n) might not work directly for TreeNeuron
                                    # Try navis.volume.from_object? No.
                                    # Try n.mesh property?
                                    # Actually, navis has a function to mesh neurons: navis.mesh_neurons (which failed before)
                                    # Let's try to use the internal method if possible.
                                    # Or use navis.TreeNeuron.to_mesh() if it exists? No.
                                    
                                    # Let's try a simpler approach:
                                    # navis.plot3d generates meshes internally.
                                    # But we want to merge them BEFORE plotting.
                                    
                                    # Try: mesh_neuron = navis.MeshNeuron(n) - this might work if n is compatible
                                    # Or: mesh_neuron = n.convert_to_mesh() - hypothetical
                                    
                                    # Let's assume navis.conversion.tree2meshneuron works as per subagent
                                    # If not, we catch exception.
                                    mesh_neuron = navis.conversion.tree2meshneuron(n)
                                
                                if hasattr(mesh_neuron, 'trimesh'):
                                    meshes.append(mesh_neuron.trimesh)
                            except Exception as e:
                                # print(f'Warning: Failed to mesh neuron {n.id}: {e}')
                                pass
                        
                        if meshes:
                            # Concatenate meshes
                            merged_mesh = trimesh.util.concatenate(meshes)
                            
                            # Simplify if requested
                            if self.skeleton_mesh_simplification > 0:
                                n_faces = len(merged_mesh.faces)
                                target_faces = int(n_faces * (1 - self.skeleton_mesh_simplification))
                                if target_faces < n_faces:
                                    try:
                                        # Try open3d simplification first (better quality)
                                        # If open3d not installed, trimesh might fail or use other method
                                        # trimesh.simplify_quadratic_decimation uses open3d or fast-simplification
                                        merged_mesh = merged_mesh.simplify_quadratic_decimation(target_faces)
                                    except Exception as e:
                                        self._vprint(f' (simplification failed: {e})', end='', level='full')
                            
                            # Convert back to navis object
                            neuron_vols = navis.MeshNeuron(merged_mesh)
                            neuron_vols.name = self.layer_names[i]
                            self._vprint(' (merged) ', end='', level='full')
                        else:
                            self._vprint(' (merge failed: no meshes generated) ', end='', level='full')
                    else:
                        # For line mode, we can merge traces later in plotting?
                        # Actually, navis.plot3d returns a figure with traces.
                        # We can merge them there.
                        self._vprint(' (will merge traces in plot) ', end='', level='full')
                except Exception as e:
                    self._vprint(f'⚠️  Merge failed: {e}, plotting individually', level='full')

            self._vprint('plotting...', end='', level='full')
            
            if self.backend == 'plotly':
                with self._suppress_output():
                    fig_layer = navis.plot3d(
                        neuron_vols,
                        backend='plotly',
                        color=self.neuron_colors[i],
                        alpha=self.neuron_alpha,
                        soma=self.show_soma if not isinstance(neuron_vols, navis.Volume) else False,
                        # fig=self.fig_3d,
                        radius=self.show_skeleton_radius,
                        connectors=self.show_connectors if not isinstance(neuron_vols, navis.Volume) else False,
                    )
                fig_traces = fig_layer.data
                
                # If merging was requested for line mode, we can optimize here by combining traces
                if self.merge_neurons and self.skeleton_mode == 'line' and len(fig_traces) > 1:
                    # Combine all scatter3d traces into one
                    x_all, y_all, z_all = [], [], []
                    for trace in fig_traces:
                        if hasattr(trace, 'x') and trace.x is not None:
                            x_all.extend(trace.x)
                            x_all.append(None) # Add break between lines
                            y_all.extend(trace.y)
                            y_all.append(None)
                            z_all.extend(trace.z)
                            z_all.append(None)
                    
                    # Create single merged trace
                    merged_trace = go.Scatter3d(
                        x=x_all, y=y_all, z=z_all,
                        mode='lines',
                        line=dict(color=self.neuron_colors[i], width=1),
                        opacity=self.neuron_alpha,
                        name=self.layer_names[i]
                    )
                    fig_traces = [merged_trace]

                for j,trace in enumerate(fig_traces):
                    # Enforce opacity for lines if not already set or if we want to override
                    if self.skeleton_mode == 'line':
                        trace.opacity = self.neuron_alpha

                    if self.legend_mode == 'merge':
                        if j == 0:
                            trace.showlegend = True
                        else:
                            trace.showlegend = False
                        trace.name = self.layer_names[i]
                        trace.hovertemplate = '<b>%{fullData.name}</b><extra></extra>'  # show full name in hover tooltip
                        trace.legendgroup = self.layer_names[i]
                        trace.hoverinfo = 'name'
                        self.fig_3d.add_trace(trace)
                    elif self.legend_mode == 'normal':
                        trace.hoverinfo = 'name'
                        trace.hovertemplate = '<b>%{fullData.name}</b><extra></extra>'
                        self.fig_3d.add_trace(trace)
                    else:
                        raise ValueError(f'legend_mode {self.legend_mode} not supported')
            
            elif self.backend == 'k3d':
                try:
                    # navis.plot3d with k3d backend returns a k3d.Plot object
                    with self._suppress_output():
                        temp_plot = navis.plot3d(
                            neuron_vols,
                            backend='k3d',
                            color=self.neuron_colors[i],
                            alpha=self.neuron_alpha,
                            soma=self.show_soma if not isinstance(neuron_vols, navis.Volume) else False,
                            radius=self.show_skeleton_radius,
                            connectors=self.show_connectors if not isinstance(neuron_vols, navis.Volume) else False,
                            inline=False
                        )
                    
                    for obj in temp_plot.objects:
                        if hasattr(obj, 'name'):
                            obj.name = self.layer_names[i]
                        self.fig_3d += obj
                except Exception as e:
                    self._vprint(f'⚠️  k3d plotting failed: {e}', level='full')

            self._vprint('Done', level='full')
        return 0
    
    def _get_synapse_cache_path(self, pre_id, post_id):
        """Get cache file path for synapses between a specific pre/post neuron pair.
        
        Cache structure: cache/{dataset}/synapses/{pre_id}_{post_id}.parquet
        
        This caches by neuron pair rather than by layer, because:
        1. The same synapse data is reusable across different queries
        2. Layer indices are arbitrary and session-specific
        3. Avoids duplicate storage of the same synaptic connections
        """
        cache_dir = self._get_cache_path('synapses')  # Note: 'synapses' not 'synapse'
        return os.path.join(cache_dir, f'{pre_id}_{post_id}.parquet')
    
    def _load_cached_synapses(self, source_ids, target_ids):
        """Load cached synapse connections for given source/target neuron pairs.
        
        For FlyWire/FAFB datasets, loads from the master synapse table at:
            datasets/{dataset}/{dataset}_synapse_table.parquet
        and filters by source_ids and target_ids.
        
        For other datasets, loads individual cache files per neuron pair from:
            cache/{dataset}/synapses/{pre_id}_{post_id}.parquet
        
        Args:
            source_ids: Set/list of source (presynaptic) body IDs
            target_ids: Set/list of target (postsynaptic) body IDs
            
        Returns:
            Tuple of (cached_df, missing_pairs) where:
            - cached_df: DataFrame of cached synapses (may be None if nothing cached)
            - missing_pairs: List of (pre_id, post_id) tuples not found in cache
        """
        if not self.cache_synapses:
            # Return all pairs as missing
            all_pairs = [(s, t) for s in source_ids for t in target_ids]
            return None, all_pairs
        
        source_ids = set(str(s) for s in source_ids)
        target_ids = set(str(t) for t in target_ids)
        
        # For FlyWire/FAFB, use the master synapse table from datasets folder
        if 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower():
            synapse_table_path = self._get_synapse_table_path()
            if os.path.exists(synapse_table_path):
                try:
                    # Load master synapse table
                    synapse_df = pd.read_parquet(synapse_table_path)
                    self._vprint(f'  ✓ Loaded synapse table from {synapse_table_path}', level='full')
                    
                    # Determine column names (may vary by dataset)
                    pre_col = 'pre_pt_root_id' if 'pre_pt_root_id' in synapse_df.columns else 'bodyId_pre'
                    post_col = 'post_pt_root_id' if 'post_pt_root_id' in synapse_df.columns else 'bodyId_post'
                    
                    # Convert to string for matching
                    synapse_df[pre_col] = synapse_df[pre_col].astype(str)
                    synapse_df[post_col] = synapse_df[post_col].astype(str)
                    
                    filtered_df = synapse_df[
                        (synapse_df[pre_col].isin(source_ids)) & 
                        (synapse_df[post_col].isin(target_ids))
                    ]
                    self._vprint(f'  ✓ Filtered to {len(filtered_df)} synapses between {len(source_ids)} sources and {len(target_ids)} targets', level='full')
                    # For FlyWire, master table has all data - no missing pairs
                    return filtered_df, []
                except Exception as e:
                    self._vprint(f'  ⚠ Failed to load synapse table: {e}', level='full')
                    all_pairs = [(s, t) for s in source_ids for t in target_ids]
                    return None, all_pairs
            else:
                self._vprint(f'  ⚠ Synapse table not found at {synapse_table_path}', level='full')
                all_pairs = [(s, t) for s in source_ids for t in target_ids]
                return None, all_pairs
        
        # For other datasets, load from individual cache files per neuron pair
        cached_dfs = []
        missing_pairs = []
        
        for pre_id in source_ids:
            for post_id in target_ids:
                cache_file = self._get_synapse_cache_path(pre_id, post_id)
                if os.path.exists(cache_file):
                    try:
                        df = pd.read_parquet(cache_file)
                        if not df.empty:
                            cached_dfs.append(df)
                    except Exception as e:
                        self._vprint(f'  ⚠ Cache load failed for {pre_id}→{post_id}: {e}', level='full')
                        missing_pairs.append((pre_id, post_id))
                else:
                    missing_pairs.append((pre_id, post_id))
        
        if cached_dfs:
            cached_df = pd.concat(cached_dfs, ignore_index=True)
            self._vprint(f'  ✓ Loaded {len(cached_df)} synapses from cache ({len(cached_dfs)} pairs cached, {len(missing_pairs)} pairs missing)', level='full')
        else:
            cached_df = None
            
        return cached_df, missing_pairs
    
    def _save_cached_synapses(self, conn_df, attempted_pairs=None):
        """Save synapse connections to cache, organized by pre/post neuron pairs.
        
        Each unique (pre_id, post_id) pair gets its own cache file at:
            cache/{dataset}/synapses/{pre_id}_{post_id}.parquet
            
        This approach ensures:
        1. Synapses are cached by their actual content (neuron pairs + positions)
        2. Same synapse data is reusable across different queries/layers
        3. Incremental caching - only fetch what's not already cached
        
        Args:
            conn_df: DataFrame containing synapse data
            attempted_pairs: Optional list of (pre_id, post_id) tuples that were queried.
                             Used to cache empty results for pairs with no synapses.
        """
        if not self.cache_synapses:
            return
            
        # Do not cache for FlyWire/FAFB - they use the master synapse table
        if 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower():
            return
        
        # Track which pairs we have saved
        saved_pairs = set()
        
        if conn_df is not None and not conn_df.empty:
            # Determine column names for pre/post body IDs
            pre_col = 'bodyId_pre' if 'bodyId_pre' in conn_df.columns else 'pre_pt_root_id'
            post_col = 'bodyId_post' if 'bodyId_post' in conn_df.columns else 'post_pt_root_id'
            
            if pre_col not in conn_df.columns or post_col not in conn_df.columns:
                self._vprint(f'  ⚠ Cannot cache synapses: missing {pre_col} or {post_col} columns', level='full')
                return
            
            # Group by pre/post pairs and save each group
            saved_count = 0
            for (pre_id, post_id), group_df in conn_df.groupby([pre_col, post_col]):
                pre_id_str = str(pre_id)
                post_id_str = str(post_id)
                cache_file = self._get_synapse_cache_path(pre_id_str, post_id_str)
                
                try:
                    group_df.to_parquet(cache_file, index=False)
                    saved_count += 1
                    saved_pairs.add((pre_id_str, post_id_str))
                except Exception as e:
                    self._vprint(f'  ⚠ Cache save failed for {pre_id}→{post_id}: {e}', level='full')
            
            self._vprint(f'  💾 Saved synapses to cache ({saved_count} neuron pairs)', level='full')

        # Handle empty results for attempted pairs
        if attempted_pairs:
            empty_saved_count = 0
            for pre_id, post_id in attempted_pairs:
                pre_id_str = str(pre_id)
                post_id_str = str(post_id)
                if (pre_id_str, post_id_str) not in saved_pairs:
                    # Save empty dataframe
                    cache_file = self._get_synapse_cache_path(pre_id_str, post_id_str)
                    try:
                        # Create empty DF
                        pd.DataFrame().to_parquet(cache_file)
                        empty_saved_count += 1
                    except Exception as e:
                        self._vprint(f'  ⚠ Cache save failed for empty {pre_id}→{post_id}: {e}', level='full')
            
            if empty_saved_count > 0:
                self._vprint(f'  💾 Cached {empty_saved_count} empty synapse pairs', level='full')
    
    def plot_synapses(self):
        if self.skip_synapse:
            self._vprint('Skipping synapse plotting as requested.', level='full')
            return

        for i in range(len(self.neuron_layers) - 1):
            source_criteria = self.layer_criteria[i]
            target_criteria = self.layer_criteria[i + 1]
            # Use a single file for all synapse layers, consistent with neuron_info.xlsx
            file_path = os.path.join(self.save_folder, self.saveas + '_synapses.xlsx')
            conn_df = None

            # --- Begin FlyWire/NeuPrint synapse loading logic ---
            if self.client_type == 'flywire':
                # Try loading from local file first
                # Find dataset folder and synapse file dynamically
                dataset_normalized = self.dataset.replace(':', '_').replace('.', '_')
                dataset_dir = os.path.join(self.script_path, 'datasets', dataset_normalized)
                
                # Explicitly look for the file generated by FAFB_file_converter
                parquet_file = os.path.join(dataset_dir, f"{dataset_normalized}_synapse_table.parquet")
                
                # Use Parquet if available
                if os.path.exists(parquet_file):
                    try:
                        self._vprint(f'  Reading synapses from {parquet_file} (Parquet)...', level='full')
                        source_ids = set(self.neuron_dfs[i]['bodyId'].astype(str))
                        target_ids = set(self.neuron_dfs[i+1]['bodyId'].astype(str))
                        
                        # Read parquet with filters (requires pyarrow)
                        import pyarrow.parquet as pq
                        schema = pq.read_schema(parquet_file)
                        pre_col = next((c for c in schema.names if c.startswith('pre_root_id')), None)
                        post_col = next((c for c in schema.names if c.startswith('post_root_id')), None)
                        
                        if pre_col and post_col:
                            # Check for coordinate columns
                            coord_cols = ['pre_x', 'pre_y', 'pre_z', 'post_x', 'post_y', 'post_z']
                            available_cols = schema.names
                            missing_coords = [c for c in coord_cols if c not in available_cols]
                            
                            if missing_coords:
                                self._vprint(f"  ⚠ Missing coordinate columns in Parquet: {missing_coords}", level='full')
                                # Try to find alternatives (e.g. x_pre vs pre_x)
                                alt_map = {
                                    'pre_x': ['x_pre', 'pre_pt_x'],
                                    'pre_y': ['y_pre', 'pre_pt_y'],
                                    'pre_z': ['z_pre', 'pre_pt_z'],
                                    'post_x': ['x_post', 'post_pt_x'],
                                    'post_y': ['y_post', 'post_pt_y'],
                                    'post_z': ['z_post', 'post_pt_z']
                                }
                                found_map = {}
                                for target, alts in alt_map.items():
                                    if target in available_cols:
                                        found_map[target] = target
                                    else:
                                        for alt in alts:
                                            if alt in available_cols:
                                                found_map[target] = alt
                                                break
                                
                                if len(found_map) == 6:
                                    self._vprint("  ✓ Found alternative coordinate columns", level='full')
                                    columns = list(found_map.values()) + [pre_col, post_col]
                                    df = pd.read_parquet(parquet_file, columns=columns)
                                    # Rename to standard
                                    inv_map = {v: k for k, v in found_map.items()}
                                    df = df.rename(columns=inv_map)
                                else:
                                    self._vprint("  ❌ Could not resolve all coordinate columns. Skipping.", level='full')
                                    conn_df = None
                            else:
                                columns = coord_cols + [pre_col, post_col]
                                df = pd.read_parquet(parquet_file, columns=columns)

                            if conn_df is None and 'df' in locals():
                                df[pre_col] = df[pre_col].astype(str)
                                df[post_col] = df[post_col].astype(str)
                                
                                mask = (df[pre_col].isin(source_ids)) & (df[post_col].isin(target_ids))
                                conn_df = df[mask].copy()
                                
                                if not conn_df.empty:
                                    rename_map = {
                                        'pre_x': 'x_pre', 'pre_y': 'y_pre', 'pre_z': 'z_pre',
                                        'post_x': 'x_post', 'post_y': 'y_post', 'post_z': 'z_post',
                                        pre_col: 'bodyId_pre',
                                        post_col: 'bodyId_post'
                                    }
                                    conn_df = conn_df.rename(columns=rename_map)
                                    
                                    # Check coordinate scale
                                    # If Z > 10000, assume nm and DO NOT scale
                                    if conn_df['z_pre'].max() > 10000:
                                        self._vprint('  ✓ Detected coordinates in nanometers (no scaling applied)', level='full')
                                    else:
                                        self._vprint('  ✓ Detected coordinates in voxels (scaling 4x4x40)', level='full')
                                        conn_df['x_pre'] = conn_df['x_pre'] * 4
                                        conn_df['y_pre'] = conn_df['y_pre'] * 4
                                        conn_df['z_pre'] = conn_df['z_pre'] * 40
                                        conn_df['x_post'] = conn_df['x_post'] * 4
                                        conn_df['y_post'] = conn_df['y_post'] * 4
                                        conn_df['z_post'] = conn_df['z_post'] * 40

                                    self._vprint(f'  ✓ Found {len(conn_df)} synapses in Parquet file', level='full')
                                else:
                                    self._vprint('  No matching synapses found in Parquet file', level='full')
                                    conn_df = None
                        else:
                            self._vprint("  ⚠️ Could not find root_id columns in Parquet schema", level='full')
                            conn_df = None
                    except Exception as e:
                        self._vprint(f'  ⚠️ Failed to read Parquet file: {e}', level='full')
                        conn_df = None
                else:
                    # Fallback or warning
                    self._vprint(f"  ℹ️  Synapse table not found: {parquet_file}", level='full')
                    self._vprint("     If you have the raw CSV, please ensure FAFB_file_converter has run successfully.", level='full')
                    conn_df = None

                
                # Fallback to client if local failed or returned nothing
                if conn_df is None and self.client_flywire:
                    self._vprint(f"\\n  ⚠️  Local synapse file not found for dataset '{self.dataset}'.", level='full')
                    if 'fafb' in self.dataset.lower():
                        self._vprint("  Please download the synapse table from: https://codex.flywire.ai/api/download?dataset=fafb", level='full')
                    self._vprint(f"  Save the file to: {dataset_dir}", level='full')
                    self._vprint("  Skipping synapse plotting for this layer.", level='full')
                    continue
            else:
                # Fetch from NeuPrint - use new caching strategy
                source_ids = set(self.neuron_dfs[i]['bodyId'].astype(str))
                target_ids = set(self.neuron_dfs[i+1]['bodyId'].astype(str))
                
                # Try to load from cache first
                cached_df, missing_pairs = self._load_cached_synapses(source_ids, target_ids)
                
                if not missing_pairs:
                    # All data cached
                    conn_df = cached_df
                elif cached_df is not None and len(missing_pairs) < len(source_ids) * len(target_ids):
                    # Partial cache - fetch missing and combine
                    self._vprint(f'  Fetching {len(missing_pairs)} missing neuron pairs from NeuPrint...')
                    fetched_df = fetch_synapse_connections(
                        source_criteria=source_criteria,
                        target_criteria=target_criteria,
                        min_total_weight=self.min_synapse_num,
                        synapse_criteria=self.synapse_criteria,
                        client=self.client,
                    )
                    if fetched_df is not None and not fetched_df.empty:
                        conn_df = pd.concat([cached_df, fetched_df], ignore_index=True)
                        # Save newly fetched data to cache
                        self._save_cached_synapses(fetched_df, attempted_pairs=missing_pairs)
                    else:
                        conn_df = cached_df
                        # Also save empty results for missing pairs
                        self._save_cached_synapses(None, attempted_pairs=missing_pairs)
                else:
                    # No cache - fetch all
                    conn_df = fetch_synapse_connections(
                        source_criteria=source_criteria,
                        target_criteria=target_criteria,
                        min_total_weight=self.min_synapse_num,
                        synapse_criteria=self.synapse_criteria,
                        client=self.client,
                    )
                    # Save to cache
                    all_pairs = [(s, t) for s in source_ids for t in target_ids]
                    self._save_cached_synapses(conn_df, attempted_pairs=all_pairs)
        
            if conn_df is None or conn_df.empty:
                self._vprint('  No synapses found.', level='full')
                continue

            # Check if file exists to determine mode (handle skipped layers)
            if os.path.exists(file_path):
                mode = 'a'
            else:
                mode = 'w'
                
            with pd.ExcelWriter(file_path, mode=mode, engine='openpyxl') as writer:
                conn_df.to_excel(writer, sheet_name=f'conn_df{i}_{i+1}')
            
            self._vprint('plotting...', end='', level='full')
            
            if self.synapse_mode == 'scatter' or self.backend == 'k3d':
                X = (conn_df['x_pre']+conn_df['x_post'])/2
                Y = (conn_df['y_pre']+conn_df['y_post'])/2
                Z = (conn_df['z_pre']+conn_df['z_post'])/2
                xyz_df = pd.DataFrame({'x':X, 'y':Y, 'z':Z})
                
                # Ensure coordinates are float to avoid dtype warnings during transform
                xyz_df = xyz_df.astype(float)

                # Attach colors to dataframe to preserve order during transform
                c_val = self.synapse_colors[i]
                is_color_array = False
                if isinstance(c_val, (list, np.ndarray)) and len(c_val) == len(xyz_df):
                     # Check if it's not just a single RGB tuple
                     if len(xyz_df) != 3 or (isinstance(c_val[0], (str, list, tuple, np.ndarray))):
                         xyz_df['__color'] = c_val
                         is_color_array = True
                
                if self.brain_mesh in ['whole', 'template']:
                    template_info = self._get_template_info()
                    self._vprint(f'Transforming synapses of layer {i} -> {i+1}...', end='', level='full')
                    with self._suppress_output():
                        xyz_df = navis.xform_brain(xyz_df, source=template_info['source'], target=template_info['target'])
                
                # Retrieve colors
                if is_color_array and '__color' in xyz_df.columns:
                    plot_colors = xyz_df['__color'].tolist()
                else:
                    plot_colors = self.synapse_colors[i]
                
                if self.backend == 'plotly':

                    # Create 3 layers for gradient effect (Outer -> Inner)
                    # Center: synapse_alpha, Surround: synapse_alpha/10
                    base_alpha = self.synapse_alpha
                    outer_alpha = base_alpha / 10.0
                    layers = 3
                    
                    for l in range(layers):
                        # Calculate size and alpha for this layer
                        # l=0 (Outer): Size=100%, Alpha=Low
                        # l=2 (Inner): Size=33%, Alpha=High
                        
                        # Size factor: 1.0 -> 0.33
                        size_factor = (layers - l) / layers 
                        current_size = self.synapse_size * size_factor
                        
                        # Alpha interpolation: outer_alpha -> base_alpha
                        if layers > 1:
                            t = l / (layers - 1)
                            current_alpha = outer_alpha + t * (base_alpha - outer_alpha)
                        else:
                            current_alpha = base_alpha
                            
                        # Only show legend for the inner-most layer (most representative color)
                        show_legend = (l == layers - 1)
                        
                        sp = go.Scatter3d(
                            x = xyz_df['x'],
                            y = xyz_df['y'],
                            z = xyz_df['z'],
                            mode = 'markers',
                            name = f'synapses {i} -> {i+1} ({len(conn_df)})',
                            hoverinfo = 'name',
                            hovertemplate = 'x: %{x}<br>y: %{y}<br>z: %{z}<br>name: %{fullData.name}<extra></extra>',
                            legendgroup = f'synapses {i} -> {i+1} ({len(conn_df)})',
                            showlegend = show_legend,
                            marker = dict(
                                size = current_size,
                                color = plot_colors,
                                symbol = 'circle',
                                opacity = current_alpha
                            ),
                        )
                        self.fig_3d.add_trace(sp)
                elif self.backend == 'k3d':
                    try:
                        import k3d
                        # import numpy as np # Removed to avoid UnboundLocalError
                        import matplotlib.colors as mcolors
                        
                        # Color conversion helper
                        def to_int_color(c):
                            color_int = 0xff0000 # Default red
                            try:
                                if isinstance(c, str):
                                    if not c.startswith('#'):
                                        c = mcolors.to_hex(c)
                                    color_int = int(c.replace('#', ''), 16)
                                elif isinstance(c, (tuple, list, np.ndarray)):
                                    if len(c) >= 3:
                                        if isinstance(c[0], float) and c[0] <= 1.0:
                                            r, g, b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
                                        else:
                                            r, g, b = int(c[0]), int(c[1]), int(c[2])
                                        color_int = (r << 16) + (g << 8) + b
                                    elif len(c) == 1: # Handle single element array
                                        return to_int_color(c[0])
                            except Exception:
                                pass
                            return color_int

                        # Determine if we have per-point colors or single color
                        c_val = plot_colors
                        colors_to_pass = None
                        
                        # Check if c_val is a list/array of colors matching the number of points
                        # Note: A single RGB tuple (r,g,b) has len 3, but we shouldn't treat it as 3 points if len(xyz_df) != 3
                        is_array_of_colors = False
                        if isinstance(c_val, (list, np.ndarray)):
                            if len(c_val) == len(xyz_df) and len(xyz_df) > 0:
                                # It matches length, but is it a list of colors or a single RGB tuple?
                                # If len(xyz_df) == 3, it's ambiguous. Assume RGB tuple if elements are numbers.
                                first_elem = c_val[0]
                                if isinstance(first_elem, (str, list, tuple, np.ndarray)):
                                    is_array_of_colors = True
                                elif len(xyz_df) != 3: # If not 3 points, it must be array of colors
                                    is_array_of_colors = True
                                # If len is 3 and elements are numbers, assume single RGB color (default behavior)

                        if is_array_of_colors:
                            # Convert each color to int
                            colors_to_pass = [to_int_color(c) for c in c_val]
                            # k3d expects uint32 array for per-point colors
                            colors_to_pass = np.array(colors_to_pass, dtype=np.uint32)
                        else:
                            # Single color
                            colors_to_pass = to_int_color(c_val)

                        pts = k3d.points(
                            positions=xyz_df[['x', 'y', 'z']].values.astype(np.float32),
                            point_size=float(self.synapse_size) if self.synapse_mode == 'scatter' else float(self.synapse_size)/10.0,
                            color=colors_to_pass,
                            opacity=self.synapse_alpha,
                            name=f'synapses {i} -> {i+1} ({len(conn_df)})'
                        )
                        self.fig_3d += pts
                    except Exception as e:
                        self._vprint(f'⚠️  k3d synapse plotting failed: {e}', level='full')
            
            elif self.synapse_mode in ['sphere', 'cone', 'tetrahedron'] and self.backend == 'plotly':
                pre_coords = conn_df[['x_pre', 'y_pre', 'z_pre']].rename(columns={'x_pre':'x', 'y_pre':'y', 'z_pre':'z'})
                post_coords = conn_df[['x_post', 'y_post', 'z_post']].rename(columns={'x_post':'x', 'y_post':'y', 'z_post':'z'})
                
                if self.brain_mesh in ['whole', 'template']:
                    template_info = self._get_template_info()
                    self._vprint(f'Transforming synapses of layer {i} -> {i+1}...', end='', level='full')
                    pre_coords = navis.xform_brain(pre_coords, source=template_info['source'], target=template_info['target'])
                    post_coords = navis.xform_brain(post_coords, source=template_info['source'], target=template_info['target'])
                
                # Calculate sizes if 'real'
                current_size = self.synapse_size
                if self.synapse_size == 'real':
                     # Calculate Euclidean distance
                     diff = pre_coords[['x', 'y', 'z']].values - post_coords[['x', 'y', 'z']].values
                     dists = np.linalg.norm(diff, axis=1)
                     current_size = dists
                
                mesh = sv.build_synapse_mesh(
                    pre_coords, 
                    post_coords, 
                    mode=self.synapse_mode, 
                    size=current_size, 
                    color=self.synapse_colors[i], 
                    opacity=self.synapse_alpha,
                    name=f'synapses {i} -> {i+1} ({len(conn_df)})'
                )
                mesh.hoverinfo = 'name'
                mesh.legendgroup = f'synapses {i} -> {i+1} ({len(conn_df)})'
                mesh.hovertemplate = '<b>%{fullData.name}</b><extra></extra>'
                mesh.showlegend = False
                self.fig_3d.add_trace(mesh)

                # Add dummy scatter trace for legend
                dummy_legend = go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers',
                    name=f'synapses {i} -> {i+1} ({len(conn_df)})',
                    legendgroup=f'synapses {i} -> {i+1} ({len(conn_df)})',
                    showlegend=True,
                    marker=dict(
                        size=10,
                        color=self.synapse_colors[i],
                        symbol='circle'
                    )
                )
                self.fig_3d.add_trace(dummy_legend)
            self._vprint('Done', level='full')
        return 0
    
    def _get_dataset_mesh_dir(self):
        """Get dataset-specific mesh directory path.
        
        Uses cache/ folder for ROI meshes:
        - hemibrain:v1.2.1 -> cache/hemibrain_v1_2_1/meshes/
        - optic-lobe:v1.1 -> cache/optic-lobe_v1_1/meshes/
        
        References:
        - navis mesh handling: https://navis.readthedocs.io/en/latest/source/api.html#navis.Volume
        - mesh compression: use navis.Volume.to_json() with compression for storage optimization
        """
        dataset_normalized = self.dataset.replace(':', '_').replace('.', '_')
        cache_mesh_dir = os.path.join(self.script_path, 'cache', dataset_normalized, 'meshes')
        os.makedirs(cache_mesh_dir, exist_ok=True)
        return cache_mesh_dir
    
    def _get_available_rois(self, use_cache=True, fetch_online=True):
        """Query NeuPrint database for available ROIs in the current dataset.
        
        Caches results locally to avoid repeated API calls. Returns a list of ROI names
        that are available in the NeuPrint database for the current dataset.
        
        Parameters
        ----------
        use_cache : bool
            If True, use cached ROI list if available. If False, force refresh from API.
        fetch_online : bool
            If True, attempt to fetch from NeuPrint online. If False, only use local cache/meshes.
        
        Returns
        -------
        list
            List of available ROI names for the current dataset.
        
        References:
        - NeuPrint ROI documentation: https://neuprint.janelia.org/
        - navis neuprint interface: https://navis-org.github.io/navis/reference/navis/interfaces/neuprint/
        - neuprint-python API: https://github.com/connectome-neuprint/neuprint-python
        """
        # Cache file path in organized cache/ structure
        dataset_normalized = self.dataset.replace(':', '_').replace('.', '_')
        cache_dir = os.path.join(self.script_path, 'cache', dataset_normalized)
        cache_file = os.path.join(cache_dir, 'available_rois.json')
        
        # Try to load from cache first
        if use_cache and os.path.exists(cache_file):
            try:
                import json
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    self._vprint(f'✓ Loaded {len(cached_data)} available ROIs from cache', level='full')
                    return cached_data
            except Exception as e:
                self._vprint(f'⚠️ Failed to load ROI cache: {e}, fetching from API...', level='full')
        
        # Fetch from NeuPrint API
        if fetch_online:
            # Special handling for FlyWire/FAFB: Do not use API, use local primary_rois or hemibrain cache
            if 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower():
                self._vprint('ℹ️  FlyWire/FAFB dataset detected: Skipping online API fetch for ROIs.', level='full')
                self._vprint('   Scanning local ROI meshes...', level='full')
                
                found_rois = set()
                
                # Scan primary_rois
                primary_dir = os.path.join(self.script_path, 'navis_roi_meshes_json', 'primary_rois')
                if os.path.exists(primary_dir):
                    for f in os.listdir(primary_dir):
                        if f.endswith('.json'):
                            found_rois.add(f[:-5])
                            
                # Scan hemibrain cache
                hb_cache = os.path.join(self.script_path, 'cache', 'hemibrain_v1_2_1', 'meshes')
                if os.path.exists(hb_cache):
                    for f in os.listdir(hb_cache):
                        if f.endswith('.json'):
                            found_rois.add(f[:-5])
                            
                roi_list = sorted(list(found_rois))
                self._vprint(f'✓ Found {len(roi_list)} available ROIs from local storage', level='full')
                
                # Cache the results
                if roi_list:
                    try:
                        import json
                        os.makedirs(cache_dir, exist_ok=True)
                        with open(cache_file, 'w') as f:
                            json.dump(roi_list, f, indent=2)
                    except Exception as e:
                        self._vprint(f'⚠️ Failed to cache ROI list: {e}', level='full')
                        
                return roi_list

            try:
                self._vprint('📥 Fetching available ROIs from NeuPrint online database...', level='full')
                
                # Initialize neuprint client using environment variable or global client
                from neuprint import Client, fetch_meta
                
                client = self.client
                
                if client is None:
                    # Try to get token from environment variable first
                    token = os.environ.get('NEUPRINT_APPLICATION_CREDENTIALS')
                    
                    if token:
                        # Determine server URL based on dataset
                        if 'optic' in self.dataset.lower():
                            server = 'https://neuprint-optic-lobe.janelia.org'
                            dataset_name = self.dataset.split(':')[0]  # 'optic-lobe'
                        else:
                            server = 'https://neuprint.janelia.org'
                            dataset_name = 'hemibrain:v1.2.1'  # default
                        
                        try:
                            client = Client(server, dataset=dataset_name, token=token)
                        except Exception as e:
                            self._vprint(f'   Warning: Failed to create client with token: {e}', level='full')
                            self._vprint(f'   Attempting to use default/global client...', level='full')
                            client = None
                
                # Fetch metadata (will use client if provided, otherwise global)
                meta = fetch_meta(client=client)
                
                roi_list = []
                # Extract ROI list from meta info
                if 'roiInfo' in meta:
                    roi_list = list(meta['roiInfo'].keys())
                    self._vprint(f'   Found {len(roi_list)} ROIs from roiInfo', level='full')
                elif 'primaryRois' in meta:
                    roi_list = list(meta['primaryRois'])
                    self._vprint(f'   Found {len(roi_list)} primary ROIs', level='full')
                else:
                    self._vprint(f'   Warning: No roiInfo/primaryRois in metadata, falling back to local cache', level='full')
                
                roi_list = sorted(roi_list)
                
                # Cache the results (create directory only when needed)
                if roi_list:
                    try:
                        import json
                        os.makedirs(cache_dir, exist_ok=True)
                        with open(cache_file, 'w') as f:
                            json.dump(roi_list, f, indent=2)
                        self._vprint(f'✓ Cached {len(roi_list)} available ROIs to {cache_file}', level='full')
                    except Exception as e:
                        self._vprint(f'⚠️ Failed to cache ROI list: {e}', level='full')
                
                return roi_list
                
            except Exception as e:
                self._vprint(f'⚠️ Failed to fetch available ROIs from NeuPrint: {e}', level='full')
                self._vprint(f'   Tip: Set NEUPRINT_APPLICATION_CREDENTIALS environment variable', level='full')
                self._vprint(f'   Using ROIs from local mesh directory instead.', level='full')
        
        # Fallback: list available meshes from local directory
        mesh_dir = self._get_dataset_mesh_dir()
        if os.path.exists(mesh_dir):
            roi_list = [f.replace('.json', '') for f in os.listdir(mesh_dir) if f.endswith('.json')]
            roi_list = sorted(roi_list)
            self._vprint(f'✓ Found {len(roi_list)} ROIs in local cache: {mesh_dir}', level='full')
            
            # Cache the results from local scan
            if roi_list:
                try:
                    import json
                    os.makedirs(cache_dir, exist_ok=True)
                    with open(cache_file, 'w') as f:
                        json.dump(roi_list, f, indent=2)
                    self._vprint(f'✓ Cached {len(roi_list)} available ROIs to {cache_file}', level='full')
                except Exception as e:
                    self._vprint(f'⚠️ Failed to cache ROI list: {e}', level='full')
            
            return roi_list
        else:
            self._vprint(f'⚠️ No ROI data available (online fetch failed and no local cache)', level='full')
            return []
    
    def _dataset_needs_transform(self):
        """Check if current dataset needs transforms for 'whole' brain mesh.
        
        Returns
        -------
        bool
            True if transforms are required, False if native template is sufficient
        """
        dataset_lower = self.dataset.lower()
        # Hemibrain needs transform to JRC2018F
        # Optic-lobe and Male CNS need transform to JRCFIB2022M
        # MANC needs transform to MANC
        if 'hemibrain' in dataset_lower:
            return True
        if 'optic' in dataset_lower or 'male-cns' in dataset_lower or 'malecns' in dataset_lower:
            return True
        if 'manc' in dataset_lower:
            return True
        if 'flywire' in dataset_lower or 'fafb' in dataset_lower:
            return True
        return False
    
    def _get_template_info(self):
        """Get template brain/VNC information for current dataset.
        
        Handles transform paths for all NeuPrint datasets:
        - Brain datasets: hemibrain, optic-lobe
        - VNC datasets: manc (various versions)
        - Brain+VNC datasets: male-cns
        
        Returns
        -------
        dict
            Dictionary with 'source', 'target', 'template_obj', and 'mesh_name' keys
            
        Notes
        -----
        Transform paths by dataset:
        - hemibrain: JRCFIB2018Fraw → JRCFIB2018F → JRCFIB2018Fum → JRC2018F
        - optic-lobe: JRCFIB2018Fraw → JRCFIB2018F → JRCFIB2018Fum → JRC2018F (same as hemibrain)
        - manc: MANCraw → MANC (VNC only, no brain transform)
        - male-cns: JRCFIB2022Mraw → JRCFIB2022M (brain + VNC)
        
        Note: optic-lobe uses the same coordinate system as hemibrain because it's
        a focused reconstruction of the optic lobe region within the hemibrain volume.
        """
        dataset_lower = self.dataset.lower()
        import flybrains
        
        # Brain datasets
        if 'hemibrain' in dataset_lower:
            return {
                'source': 'JRCFIB2018Fraw',
                'target': 'JRC2018F' if self.brain_mesh == 'whole' else 'JRCFIB2018F',
                'template_obj': flybrains.JRC2018F if self.brain_mesh == 'whole' else flybrains.JRCFIB2018F,
                'mesh_name': 'JRC2018F (whole brain)' if self.brain_mesh == 'whole' else 'JRCFIB2018F (hemibrain)'
            }
        elif 'optic' in dataset_lower:
            # Optic-lobe dataset is part of the Male CNS (JRCFIB2022M) volume
            # It is NOT part of the hemibrain (JRCFIB2018F) volume
            # Stored in JRCFIB2022Mraw coordinates
            return {
                'source': 'JRCFIB2022Mraw',
                'target': 'JRCFIB2022M',  # Male CNS template
                'template_obj': flybrains.JRCFIB2022M,
                'mesh_name': 'JRCFIB2022M (Male CNS)'
            }
        
        # VNC datasets
        elif 'manc' in dataset_lower:
            # MANC (Male Adult Nerve Cord) - VNC only
            # For VNC: 'whole' and 'template' both show VNC envelope
            # 'hemi' is not supported (VNC doesn't have hemispheres like brain)
            return {
                'source': 'MANCraw',
                'target': 'MANC',  # VNC template (no brain transform needed)
                'template_obj': flybrains.MANC,
                'mesh_name': 'MANC (VNC envelope)'
            }
        
        # Brain + VNC datasets
        elif 'male-cns' in dataset_lower or 'malecns' in dataset_lower:
            # Male CNS (JRCFIB2022M) - Brain + VNC
            # 'whole' shows full CNS envelope (brain + VNC)
            # 'hemi' is not supported (use brain_mesh to get brain/vnc separately)
            return {
                'source': 'JRCFIB2022Mraw',
                'target': 'JRCFIB2022M',
                'template_obj': flybrains.JRCFIB2022M,
                'mesh_name': 'JRCFIB2022M (male CNS: brain + VNC)'
            }
        
        # FlyWire / FAFB datasets
        elif 'flywire' in dataset_lower or 'fafb' in dataset_lower:
            # FlyWire is in FAFB14 space (approx)
            return {
                'source': 'FAFB',
                'target': 'JRC2018F',
                'template_obj': flybrains.JRC2018F,
                'mesh_name': 'JRC2018F (whole brain)'
            }
        
        # Fallback to hemibrain for unknown datasets
        else:
            self._vprint(f'⚠️  Unknown dataset "{self.dataset}", defaulting to hemibrain template')
            return {
                'source': 'JRCFIB2018Fraw',
                'target': 'JRCFIB2018F',
                'template_obj': flybrains.JRCFIB2018F,
                'mesh_name': 'JRCFIB2018F (hemibrain)'
            }
    
    def _check_and_download_transforms(self):
        """Check if flybrains transforms exist locally, prompt user before downloading.
        
        Brain transforms are large files (multiple files, ~10GB total uncompressed). 
        This method checks if the required transforms exist locally before attempting 
        to download them, and prompts the user for confirmation.
        
        Transforms are stored in the default flybrains data directory:
        ~/flybrain-data/
        
        Returns
        -------
        bool
            True if transforms are available (already exist or successfully downloaded),
            False otherwise.
        
        References:
        - flybrains package: https://github.com/navis-org/navis-flybrains
        - JRC2018F brain template: https://www.janelia.org/open-science/jrc-2018-brain-templates
        """
        if not self.verbose:
            return False

        try:
            import flybrains
            
            # Get the transform directory from attribute or use default
            transforms_dir = os.path.expanduser(self.transforms_dir)
            
            # Set environment variable if custom path is specified
            if self.transforms_dir != '~/flybrain-data':
                os.environ['FLYBRAINS_DATA'] = transforms_dir
                self._vprint(f'Using custom transform directory: {transforms_dir}', level='full')
            
            # Get dataset-specific template info
            template_info = self._get_template_info()
            source = template_info['source']
            target = template_info['target']
            
            # ANSI color codes
            YELLOW = '\033[93m'
            RESET = '\033[0m'
            
            # Check if the transformation path exists by attempting to find bridging path
            try:
                path = navis.transforms.registry.find_bridging_path(source, target)
                self._vprint(f'✓ Brain transforms already available', level='full')
                self._vprint(f'  Location: {YELLOW}{transforms_dir}{RESET}', level='full')
                self._vprint(f'  Transform path: {" -> ".join([str(p) for p in path])}', level='full')
                return True
            except (ValueError, KeyError):
                # Transform path not found, need to download
                pass
            
            # ANSI color codes
            YELLOW = '\033[93m'
            RESET = '\033[0m'
            
            # Prompt user for download confirmation
            self._vprint('\\n' + '='*70)
            self._vprint('⚠️  Brain Transformation Required')
            self._vprint('='*70)
            self._vprint(f'To use brain_mesh="whole" for {self.dataset}, you need brain transforms.')
            self._vprint(f'Transform path needed: {source} → JRCFIB2018F → JRCFIB2018Fum → {target}')
            self._vprint('')
            self._vprint('⚠️  IMPORTANT: flybrains downloads ALL JRC transforms as a bundle:')
            self._vprint('   • JRC2018F_JRCFIB2018F.h5   (~1.29 GB)  ← YOU NEED THIS for hemibrain/optic-lobe')
            self._vprint('   • JRC2018F_FAFB.h5          (~580 MB)   (enables FAFB dataset support)')
            self._vprint('   • JRC2018F_JFRC2013.h5      (~1.39 GB)  (enables JFRC2013 template)')
            self._vprint('   • JRC2018F_FCWB.h5          (~1.29 GB)  (enables FCWB template)')
            self._vprint('   • JRC2018U_JRC2018F.h5      (~717 MB)   (enables unisex template)')
            self._vprint('   • JRC2018U_JRC2018M.h5      (~1.10 GB)  (enables male template)')
            self._vprint('   • JRC2018F_JFRC2010.h5      (~1.65 GB)  (enables legacy template)')
            self._vprint('   • JRCFIB2022M_JRC2018M.h5   (~2.12 GB)  (enables male CNS registration)')
            self._vprint('')
            self._vprint('   Total download: ~10 GB (but only ~1.3 GB used for your dataset)')
            self._vprint('   Download time: ~1-2 hours (cannot download individual files)')
            self._vprint('   Why all files? The flybrains package bundles all transforms together.')
            self._vprint('')
            self._vprint('The transforms will be cached in:')
            self._vprint(f'  {YELLOW}{transforms_dir}/{RESET}')
            
            # Save transform path info to file
            info_file = os.path.join(self.output_dir, 'brain_transforms_info.txt')
            os.makedirs(self.output_dir, exist_ok=True)
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write('Brain Transforms Information\\n')
                f.write('='*70 + '\\n\\n')
                f.write(f'Dataset: {self.dataset}\\n')
                f.write(f'Transform path: {source} → JRCFIB2018F → JRCFIB2018Fum → {target}\\n\\n')
                f.write('Storage Location:\\n')
                f.write(f'  {transforms_dir}/\\n\\n')
                f.write('Transform Files (8 files, ~10 GB total):\\n')
                f.write('  • JRC2018F_JRCFIB2018F.h5   (~1.29 GB)\\n')
                f.write('  • JRC2018F_FAFB.h5          (~580 MB)\\n')
                f.write('  • JRC2018F_JFRC2013.h5      (~1.39 GB)\\n')
                f.write('  • JRC2018F_FCWB.h5          (~1.29 GB)\\n')
                f.write('  • JRC2018U_JRC2018F.h5      (~717 MB)\\n')
                f.write('  • JRC2018U_JRC2018M.h5      (~1.10 GB)\\n')
                f.write('  • JRC2018F_JFRC2010.h5      (~1.65 GB)\\n')
                f.write('  • JRCFIB2022M_JRC2018M.h5   (~2.12 GB)\\n\\n')
                f.write('To change the storage location:\\n')
                f.write('  1. Set transforms_dir attribute when creating VisualizeSkeleton\\n')
                f.write('  2. Set FLYBRAINS_DATA environment variable before importing flybrains\\n')
                f.write('  3. Or manually move files to the new location\\n\\n')
                f.write('More information:\\n')
                f.write('  https://github.com/navis-org/navis-flybrains\\n')
            self._vprint(f'\\n📄 Transform info saved to: {info_file}')
            self._vprint('')
            self._vprint('💡 Note: The flybrains.download_jrc_transforms() function downloads')
            self._vprint('   ALL 8 files as a bundle with no selective download option.')
            self._vprint('   This is by design in the flybrains library to provide complete')
            self._vprint('   cross-dataset registration capabilities.')
            self._vprint('')
            self._vprint('For more information, see:')
            self._vprint('  https://github.com/navis-org/navis-flybrains')
            self._vprint('='*70)
            
            response = input('Download all transforms now? [y/N]: ').strip().lower()
            
            if response in ['y', 'yes']:
                self._vprint('\\n📥 Downloading brain transforms...')
                self._vprint('This may take several minutes depending on your connection.')
                flybrains.download_jrc_transforms()
                
                # Re-register transforms after download
                self._vprint('📝 Registering downloaded transforms...')
                flybrains.register_transforms()
                
                # Verify the transform path is now available
                try:
                    path = navis.transforms.registry.find_bridging_path(source, target)
                    self._vprint(f'✓ Transforms downloaded and registered successfully!')
                    self._vprint(f'  Location: {YELLOW}{transforms_dir}{RESET}')
                    self._vprint(f'  Transform path: {" -> ".join([str(p) for p in path])}')
                    
                    # Update the saved info file with success status
                    info_file = os.path.join(self.output_dir, 'brain_transforms_info.txt')
                    with open(info_file, 'a', encoding='utf-8') as f:
                        f.write(f'\\nDownload Status: SUCCESS\\n')
                        f.write(f'Downloaded at: {pd.Timestamp.now()}\\n')
                    return True
                except (ValueError, KeyError) as e:
                    self._vprint(f'⚠️  Transforms downloaded but bridging path not found: {e}')
                    self._vprint(f'   This may indicate the transforms do not include {source} → {target}')
                    return False
            else:
                self._vprint('\\n⚠️  Download cancelled. Setting brain_mesh to "none".')
                return False
                
        except ImportError:
            self._vprint('\\n⚠️  flybrains package not installed.')
            self._vprint('   Install it with: pip install navis[flybrains]')
            self._vprint('   Setting brain_mesh to "none".')
            return False
        except Exception as e:
            self._vprint(f'\\n⚠️  Error checking brain transforms: {e}')
            self._vprint('   Setting brain_mesh to "none".')
            return False
    
    def plot_mesh(self):
        """Plot ROI meshes and brain meshes.
        
        Loads ROI meshes from dataset-specific cache directories, with fallback to
        primary_rois/ for backward compatibility. Supports brain mesh visualization
        with automatic transform handling.
        
        Dataset-specific mesh caching:
        - hemibrain:v1.2.1 -> navis_roi_meshes_json/hemibrain_v1_2_1/
        - optic-lobe:v1.1 -> navis_roi_meshes_json/optic-lobe_v1_1/
        - Fallback: navis_roi_meshes_json/primary_rois/
        
        Brain mesh options (dataset-aware):
        - 'none': Only plot ROI meshes specified in mesh_roi parameter
        - 'template': Plot native EM template mesh (JRCFIB2018F, MANC, or JRCFIB2022M)
        - 'whole': Plot standard template mesh (may require transforms for some datasets)
        
        Behavior with mesh_roi=[]:
        - When mesh_roi is an empty list [], no ROI meshes are plotted
        - But brain_mesh='whole' or 'template' will still plot the brain mesh
        - This allows showing neurons with only the whole brain outline
        
        References:
        - navis Volume API: https://navis.readthedocs.io/en/latest/source/api.html#navis.Volume
        - flybrains templates: https://github.com/navis-org/navis-flybrains
        - mesh optimization: use Volume.simplify() to reduce mesh complexity for faster rendering
        """
        # Skip if mesh_roi is None (explicitly disabled)
        # Note: Empty list [] means "no ROI meshes but maybe brain mesh"
        if self.mesh_roi is None:
            return
        
        # Check if we have any work to do (ROI meshes or brain mesh)
        has_roi_meshes = len(self.mesh_roi) > 0
        has_brain_mesh = self.brain_mesh in ['template', 'whole']
        
        if not has_roi_meshes and not has_brain_mesh:
            return
        
        # Ensure available_rois.json exists (generate if missing)
        # This checks cache first, and if missing, fetches from API or scans local meshes
        self._get_available_rois(use_cache=True, fetch_online=True)
        
        # Get dataset-specific mesh directory
        mesh_dir = self._get_dataset_mesh_dir()
        self._vprint(f'Using mesh directory: {mesh_dir}', level='full')
        
        roiunits = []
        roi_names = []
        roi_colors = []
        
        # Use mesh_roi list directly (no auto-mirroring suffix expansion)
        final_mesh_roi = self.mesh_roi
        
        # Handle colors
        final_mesh_colors = []
        for i, roi in enumerate(final_mesh_roi):
            if isinstance(self.mesh_color, list):
                if i < len(self.mesh_color):
                    color = self.mesh_color[i]
                else:
                    color = (100, 100, 100, 0.2)
            else:
                color = self.mesh_color
            final_mesh_colors.append(color)
        
        for i, roi in enumerate(final_mesh_roi):
            color = final_mesh_colors[i]
            source_info = "Dataset Cache"
            roi_source_space = None # Track the coordinate space of the ROI
            
            # Determine if this is FlyWire/FAFB
            is_flywire = 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower()

            # Try dataset-specific directory first
            mesh_file = os.path.join(mesh_dir, roi + '.json')
            
            # Special handling for FlyWire/FAFB
            if is_flywire:
                if not os.path.exists(mesh_file):
                    self._vprint(f'📥 ROI mesh "{roi}" not found locally, attempting to download...', level='full')
                    mesh_found = False
                    
                    # 1. Try male-cns:v0.9 (NeuPrint)
                    try:
                        import navis.interfaces.neuprint as neu
                        from neuprint import Client
                        
                        token = os.environ.get('NEUPRINT_APPLICATION_CREDENTIALS') or self.token
                        if token:
                            try:
                                self._vprint(f'   Checking male-cns:v0.9...', level='full')
                                mc_client = Client('https://neuprint.janelia.org', dataset='male-cns:v0.9', token=token)
                                mesh = neu.fetch_roi(roi, client=mc_client)
                                if mesh:
                                    os.makedirs(mesh_dir, exist_ok=True)
                                    mesh.to_json(mesh_file)
                                    self._vprint(f'   ✓ Found in male-cns:v0.9', level='full')
                                    source_info = "male-cns:v0.9 (Downloaded)"
                                    roi_source_space = 'JRCFIB2022Mraw' # Use raw coordinates for male-cns ROIs
                                    mesh_found = True
                            except Exception as e:
                                # print(f'   (male-cns check failed: {e})')
                                pass
                    except ImportError:
                        pass
                    
                    # 2. Try generic navis fetch (fallback) - REMOVED as it causes errors if navis doesn't have fetch_roi
                    # if not mesh_found:
                    #     try:
                    #         print(f'   Attempting generic navis fetch...')
                    #         # This tries to use whatever client is default or configured in navis
                    #         # Usually fetches from Hemibrain if no dataset specified, or checks available clients
                    #         mesh = navis.fetch_roi(roi)
                    #         if mesh:
                    #             os.makedirs(mesh_dir, exist_ok=True)
                    #             mesh.to_json(mesh_file)
                    #             print(f'   ✓ Found via navis.fetch_roi')
                    #             source_info = "navis.fetch_roi"
                    #             roi_source_space = 'JRCFIB2018F' # Default for Hemibrain ROIs
                    #             mesh_found = True
                    #     except Exception as e:
                    #         print(f'   Warning: Failed to fetch "{roi}" via navis: {e}')

            # Standard logic for non-FlyWire or if file exists
            # Fallback to primary_rois if not found (only for non-FlyWire or if we want to support it)
            if not os.path.exists(mesh_file) and not is_flywire:
                mesh_file_fallback = os.path.join(self.script_path, 'navis_roi_meshes_json', 'primary_rois', roi + '.json')
                if os.path.exists(mesh_file_fallback):
                    mesh_file = mesh_file_fallback
                    source_info = "Primary ROIs (Local)"
                    roi_source_space = 'JRCFIB2018F'
                else:
                    # Try to download from NeuPrint (Hemibrain/Optic Lobe/Male CNS)
                    self._vprint(f'📥 ROI mesh "{roi}" not found locally, attempting to download from NeuPrint...', level='full')
                    source_info = "NeuPrint (Downloaded)"
                    try:
                        import navis.interfaces.neuprint as neu
                        from neuprint import Client
                        
                        client = self.client
                        
                        if client is None:
                            token = os.environ.get('NEUPRINT_APPLICATION_CREDENTIALS') or self.token
                            
                            if token:
                                if 'optic' in self.dataset.lower():
                                    server = 'https://neuprint-optic-lobe.janelia.org'
                                    dataset_name = self.dataset.split(':')[0]
                                    roi_source_space = 'JRCFIB2022Mraw' # Optic lobe
                                elif 'male-cns' in self.dataset.lower() or 'malecns' in self.dataset.lower():
                                    server = 'https://neuprint.janelia.org'
                                    dataset_name = 'male-cns:v0.9' # Default for male-cns
                                    roi_source_space = 'JRCFIB2022Mraw' # Male CNS raw
                                else:
                                    server = 'https://neuprint.janelia.org'
                                    dataset_name = 'hemibrain:v1.2.1'
                                    roi_source_space = 'JRCFIB2018F'
                                
                                try:
                                    client = Client(server, dataset=dataset_name, token=token)
                                except Exception as e:
                                    self._vprint(f'   Warning: Failed to create client: {e}', level='full')
                        
                        mesh = neu.fetch_roi(roi, client=client)
                        os.makedirs(mesh_dir, exist_ok=True)
                        mesh.to_json(mesh_file)
                        self._vprint(f'✓ Downloaded and cached "{roi}" mesh to {mesh_file}', level='full')
                        
                        # Transform if needed (Hemibrain specific)
                        if self.brain_mesh in ['whole', 'template']:
                            template_info = self._get_template_info()
                            self._vprint(f'Transforming brain region {roi}...', end='', level='full')
                            with self._suppress_output():
                                mesh = navis.xform_brain(mesh, source=template_info['source'], target=template_info['target'])
                            # Note: We don't save the transformed mesh back to cache here to keep cache pure?
                            # Actually previous code didn't save transformed.
                    except Exception as e:
                        self._vprint(f'⚠️  Failed to download "{roi}" mesh: {e}', level='full')
            
            # Load and plot
            if os.path.exists(mesh_file):
                try:
                    mesh = navis.Volume.from_json(mesh_file)
                    self._vprint(f'✓ Loaded "{roi}" from {source_info}', level='full')
                    
                    # Transform if needed
                    if self.brain_mesh in ['whole', 'template']:
                        template_info = self._get_template_info()
                        target = template_info['target']
                        
                        # Determine source for transform
                        if is_flywire:
                            # For FlyWire, use the source space of the ROI, not the dataset source (FAFB14)
                            if roi_source_space:
                                source = roi_source_space
                            else:
                                # If loading from cache (roi_source_space is None), assume it's from male-cns (JRCFIB2022Mraw)
                                # This fixes the issue where cached meshes were wrongly assumed to be in JRCFIB2018F
                                source = 'JRCFIB2022Mraw'
                        else:
                            source = template_info['source']
                            
                        self._vprint(f'Transforming brain region {roi} ({source} -> {target})...', end='', level='full')
                        try:
                            with self._suppress_output():
                                mesh = navis.xform_brain(mesh, source=source, target=target)
                            self._vprint(' Done', level='full')
                        except Exception as e:
                            self._vprint(f' Failed: {e}', level='full')
                    
                    # Simplify mesh if requested
                    if self.roi_mesh_simplification > 0:
                        try:
                            import trimesh
                            # Access underlying trimesh object
                            tm = None
                            if hasattr(mesh, 'trimesh'):
                                tm = mesh.trimesh
                            elif hasattr(mesh, 'mesh'):
                                tm = mesh.mesh
                            elif hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                                # Fallback: create trimesh from vertices/faces
                                # Note: navis.Volume properties might be numpy arrays
                                tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
                            
                            if tm:
                                n_faces = len(tm.faces)
                                target_faces = int(n_faces * (1 - self.roi_mesh_simplification))
                                if target_faces < n_faces:
                                    # simplify_quadratic_decimation returns a new trimesh object
                                    new_tm = tm.simplify_quadratic_decimation(target_faces)
                                    
                                    # Re-instantiate Volume to ensure it's clean and updated
                                    # Preserving attributes
                                    old_name = getattr(mesh, 'name', roi)
                                    old_id = getattr(mesh, 'id', None)
                                    
                                    mesh = navis.Volume(new_tm, name=old_name, id=old_id)
                                    
                                    self._vprint(f' (simplified {self.roi_mesh_simplification*100:.0f}%: {n_faces}->{len(new_tm.faces)} faces)', end='', level='full')
                                else:
                                    self._vprint(f' (simplification skipped: target {target_faces} >= {n_faces} faces)', end='', level='full')
                            else:
                                # Debug: print available attributes to help diagnose
                                attrs = [a for a in dir(mesh) if not a.startswith('_')]
                                self._vprint(f' (simplification skipped: could not extract mesh from {type(mesh)}. Available attrs: {attrs[:10]}...)', end='', level='full')
                        except Exception as e:
                            self._vprint(f' (simplification failed: {e})', end='', level='full')

                    # Collect for export
                    try:
                        tm = None
                        if hasattr(mesh, 'trimesh'):
                            tm = mesh.trimesh
                        elif hasattr(mesh, 'mesh'):
                            tm = mesh.mesh
                        elif hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                            import trimesh
                            tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
                        
                        if tm:
                            # Copy and apply color
                            tm = tm.copy()
                            rgba = self._to_rgba(color)
                            tm.visual.face_colors = rgba
                            self.exportable_meshes.append(tm)
                        else:
                            # self._vprint(f' (export skip: no mesh in {type(mesh)})', end='')
                            pass
                    except Exception as e:
                        self._vprint(f' (export collection failed: {e})', end='', level='full')

                    roiunits.append(mesh)
                    roi_names.append(roi)
                    roi_colors.append(color)

                    # Mirror logic: SKIP if FlyWire
                    if not is_flywire:
                        contralateral_roi = roi.replace('(R)', '(L)')
                        should_mirror = (
                            self.mirror_on_contralateral and 
                            roi.endswith('(R)') and 
                            contralateral_roi not in final_mesh_roi
                        )
                        
                        if should_mirror:
                            try:
                                template = None
                                if self.brain_mesh == 'whole':
                                    template_info = self._get_template_info()
                                    template = template_info['target']
                                elif self.brain_mesh == 'template':
                                    if 'hemibrain' in self.dataset or 'optic-lobe' in self.dataset:
                                        template = 'JRCFIB2018F'
                                    elif 'male-cns' in self.dataset:
                                        template = 'JRCFIB2022M'
                                
                                if template:
                                    mirrored_mesh = navis.mirror_brain(mesh, template, mirror_axis='x')
                                    roiunits.append(mirrored_mesh)
                                    roi_names.append(contralateral_roi)
                                    roi_colors.append(color)
                            except Exception as e:
                                self._vprint(f' (mirror failed: {e})', end='', level='full')

                except Exception as e:
                    self._vprint(f'⚠️  Failed to load mesh {roi}: {e}', level='full')
            else:
                if not is_flywire: # Only warn if we expected to find it (FlyWire might just fail silently if not found)
                     self._vprint(f'⚠️  ROI mesh "{roi}" not found.', level='full')
        
        # Plot ROI meshes if any were loaded
        if roiunits:
            self._vprint('plotting mesh of brain regions...', level='full')
            for roi_i in range(len(roiunits)):
                roiunits[roi_i].color = roi_colors[roi_i]
                
                if self.backend == 'plotly':
                    with self._suppress_output():
                        fig_mesh = navis.plot3d(roiunits[roi_i],backend='plotly')
                    mesh_traces = fig_mesh.data
                    for ti, trace in enumerate(mesh_traces):
                        if self.legend_mode == 'merge':
                            if ti == 0:
                                trace.showlegend = True
                            else:
                                trace.showlegend = False
                            trace.legendgroup = 'roi_mesh'
                        elif self.legend_mode == 'normal':
                            trace.showlegend = True
                            trace.legendgroup = roi_names[roi_i]
                        trace.hovertemplate = '<b>%{fullData.name}</b><extra></extra>'  # show full name in hover tooltip
                        trace.hoverinfo = 'name'
                        trace.name = 'brain regions [' + roi_names[roi_i] + '...]'
                    self.fig_3d.add_traces(mesh_traces)
                elif self.backend == 'k3d':
                    try:
                        with self._suppress_output():
                            temp_plot = navis.plot3d(roiunits[roi_i], backend='k3d', inline=False)
                        for obj in temp_plot.objects:
                            obj.name = f'brain regions [{roi_names[roi_i]}...]'
                            self.fig_3d += obj
                    except Exception as e:
                        self._vprint(f'⚠️  k3d mesh plotting failed: {e}', level='full')
        elif has_roi_meshes:
            # Only warn if user specified ROI meshes but none loaded
            self._vprint('⚠️  No valid ROI meshes loaded', level='full')

        # Plot brain mesh (whole brain or template) regardless of ROI mesh status
        if self.brain_mesh in ['template', 'whole']:
            template_info = self._get_template_info()
            mesh_display_name = template_info['mesh_name']
            
            self._vprint(f'Plotting {mesh_display_name} mesh...', level='full')
            try:
                brain_template = template_info['template_obj']
                
                if self.backend == 'plotly':
                    with self._suppress_output():
                        fig_brain = navis.plot3d(brain_template, backend='plotly')
                    brain_traces = fig_brain.data
                    for trace in brain_traces:
                        trace.showlegend = True
                        trace.name = mesh_display_name
                        trace.hoverinfo = 'none'
                        trace.color = self.brain_mesh_color
                    self.fig_3d.add_traces(brain_traces)
                elif self.backend == 'k3d':
                    with self._suppress_output():
                        temp_plot = navis.plot3d(brain_template, backend='k3d', inline=False)
                    for obj in temp_plot.objects:
                        obj.name = mesh_display_name
                        self.fig_3d += obj
                        
                self._vprint(f'✓ {mesh_display_name} mesh loaded successfully', level='full')
            except Exception as e:
                self._vprint(f'⚠️  Failed to load {mesh_display_name} mesh: {e}', level='full')
                if self._dataset_needs_transform() and not self._check_and_download_transforms():
                    self._vprint('   Skipping brain/VNC mesh visualization', level='full')
                else:
                    # Retry after download
                    try:
                        brain_template = template_info['template_obj']
                        if self.backend == 'plotly':
                            with self._suppress_output():
                                fig_brain = navis.plot3d(brain_template, backend='plotly')
                            brain_traces = fig_brain.data
                            for trace in brain_traces:
                                trace.showlegend = True
                                trace.name = mesh_display_name
                                trace.hoverinfo = 'none'
                                trace.color = self.brain_mesh_color
                            self.fig_3d.add_traces(brain_traces)
                        elif self.backend == 'k3d':
                            with self._suppress_output():
                                temp_plot = navis.plot3d(brain_template, backend='k3d', inline=False)
                            for obj in temp_plot.objects:
                                obj.name = mesh_display_name
                                self.fig_3d += obj
                        self._vprint(f'✓ {mesh_display_name} mesh loaded successfully after download', level='full')
                    except Exception as retry_e:
                        self._vprint(f'⚠️  Still failed to load {mesh_display_name} mesh: {retry_e}', level='full')
                        self._vprint('   Skipping brain/VNC mesh visualization', level='full')
        self._vprint('Done', level='full')
        return 0
    
    def save_figure(self):
        if self.backend == 'plotly':
            # add sliders
            if self.use_size_slider:
                sliders = [
                    dict(
                        active=self.synapse_size,
                        currentvalue={"prefix": "Synapse Size: "},
                        pad={"t": 50},
                        steps=[
                            dict(
                                label=str(size),
                                method="update",
                                args=[{"marker": {"size": size}}]
                            )
                            for size in list(range(0,11))
                        ],
                    ),
                ]
            else:
                sliders = []
            
            # set layout
            # Always use frontal view camera regardless of brain_mesh setting
            # This ensures consistent viewing angle for all visualizations
            # Standard fly brain orientation: X: Left-Right, Y: Dorsal-Ventral, Z: Anterior-Posterior
            # Frontal view: Look from Anterior (negative Z direction)
            scene_camera_parameters = dict(
                up=dict(x=0, y=-1, z=0),  # Y is up (inverted in some templates)
                eye=dict(x=0, y=0, z=-2.0),  # Look from front
                # center=dict(x=0, y=0, z=0), # Let Plotly auto-center
            )
            
            self.fig_3d.update_layout(
                colorway = self.synapse_colors,
                sliders=sliders,
                scene=dict(
                    dragmode='orbit',
                    xaxis={'visible':False}, 
                    yaxis={'visible':False},
                    zaxis={'visible':False},
                    # Use 'data' aspectmode to ensure equal axis scaling
                    # This prevents distortion when no meshes are plotted
                    aspectmode='data',
                ),
                scene_camera=scene_camera_parameters,
            )

            # save figure
            self.fig_path = os.path.join(self.save_folder,self.saveas)
            
            # Ensure save folder exists
            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder, exist_ok=True)
            
            self._vprint(f'saving figure to \033[34m{self.fig_path}.html\033[0m...', end='')
            
            # Optimization: use 'cdn' for smaller file size (loads plotly.js from CDN)
            # This reduces HTML file size significantly compared to 'directory' or including full plotly.js
            # Fix: Set auto_open=False to prevent hanging, handle opening manually
            # Reverted 'cdn' to default (embed) as user reported issues with subsequent PNG export
            self.fig_3d.write_html(
                self.fig_path+'.html',
                auto_open=False, 
                # include_plotlyjs='cdn',  # Reverted to default to avoid potential issues
                config={'displayModeBar': False}  # Remove toolbar to reduce overhead
            )
            
            if self.show_fig:
                try:
                    import webbrowser
                    webbrowser.open('file://' + os.path.abspath(self.fig_path+'.html'))
                except Exception as e:
                    self._vprint(f'\\n⚠️  Failed to open browser: {e}')
            
            self._vprint('Done (HTML saved)')
            
            # Optimize PNG export: only save if needed, use lower scale for speed
            try:
                self._vprint('   Exporting static PNG (may take a moment)...', end='', flush=True, level='full')
                
                # Update layout for static export to remove UI elements
                # We re-apply the camera parameters to ensure they are used for the static render
                self.fig_3d.update_layout(
                    margin=dict(l=0, r=0, b=0, t=0),
                    sliders=[],      # Remove sliders
                    updatemenus=[],   # Remove any buttons
                    scene_camera=scene_camera_parameters # Ensure camera is locked
                )
                
                # Use standard write_image which handles kaleido internally
                # We use a standard resolution (1200x900) to ensure consistent output
                self.fig_3d.write_image(self.fig_path+'.png', width=1200, height=900, scale=2)
                
                # Verify file
                if os.path.exists(self.fig_path+'.png'):
                    size = os.path.getsize(self.fig_path+'.png')
                    self._vprint(f' Done ({size/1024:.1f} KB)', level='full')
                    if size < 15 * 1024: # < 15KB is suspicious for a 3D plot
                        self._vprint('   ⚠️  Warning: Exported PNG seems blank/empty.', level='full')
                        self._vprint('       This is a known issue with Kaleido and 3D plots on some systems.')
                        self._vprint('       Please rely on the HTML file for visualization.')
                else:
                    self._vprint(' Done (File not found)')
            except Exception as e:
                self._vprint(f'\\n   ⚠️  PNG export failed: {e}. Continuing without PNG...')
            
        elif self.backend == 'k3d':
            self.fig_path = os.path.join(self.save_folder,self.saveas)
            self._vprint(f'saving figure to \033[34m{self.fig_path}.html\033[0m...', end='')
            
            try:
                from ipywidgets.embed import embed_minimal_html
                embed_minimal_html(
                    self.fig_path+'.html', 
                    views=[self.fig_3d], 
                    title=self.saveas
                )
                self._vprint('Done')
                
                if self.show_fig:
                    self._vprint('Note: k3d plots cannot be automatically opened from script. Please open the HTML file manually.')
                    
            except ImportError:
                self._vprint('\\n⚠️  ipywidgets not installed. Cannot save k3d plot to HTML.')
                self._vprint('   Please install it with `pip install ipywidgets`')
            except Exception as e:
                self._vprint(f'\\n⚠️  Failed to save k3d plot: {e}')
    
    def plot_neurons(self):
        self.plot_skeleton()
        self.plot_synapses()
        self.plot_mesh()
        self.save_figure()
    
    def _to_rgba(self, color, alpha=None):
        # Convert color to uint8 RGBA for trimesh.
        import matplotlib.colors as mcolors
        import numpy as np
        
        # Convert to RGBA float (0-1)
        try:
            # If alpha is provided, override the alpha channel of the color
            if alpha is not None:
                c = mcolors.to_rgba(color, alpha=alpha)
            else:
                c = mcolors.to_rgba(color)
        except:
            c = (0.5, 0.5, 0.5, 1.0) # Default gray
            
        # Convert to uint8 (0-255)
        return (np.array(c) * 255).astype(np.uint8)

    def export_3d_model(self, filename=None, format='glb'):
        """Export the 3D scene to a model file (GLB, OBJ, STL).
        
        Exports the current scene (neurons, synapses, ROIs) to a 3D model file.
        Useful for importing into Blender, Unity, or other 3D software.
        
        Parameters
        ----------
        filename : str, optional
            Output filename. If None, uses self.saveas.
        format : str, default 'glb'
            Output format: 'glb' (binary glTF, recommended), 'obj', 'stl', 'ply'.
        
        Returns
        -------
        str
            Path to the saved file.
        """
        if not self.exportable_meshes and not hasattr(self, 'fig_3d'):
            self._vprint('⚠️  No meshes to export. Run plot_neurons() first.')
            return None
            
        if filename is None:
            filename = self.saveas
            
        # Ensure extension
        if not filename.lower().endswith(f'.{format}'):
            filename += f'.{format}'
            
        filepath = os.path.join(self.save_folder, filename)
        
        self._vprint(f'Exporting 3D model to {filepath}...', level='full')
        
        try:
            import trimesh
            
            # Collect all meshes
            meshes = []
            
            # 1. ROI meshes (already collected in self.exportable_meshes during plot_mesh)
            if hasattr(self, 'exportable_meshes'):
                meshes.extend(self.exportable_meshes)
                
            if not meshes:
                self._vprint('⚠️  No meshes found to export.')
                return None
                
            # Combine scene
            scene = trimesh.Scene(meshes)
            
            # Export
            scene.export(filepath)
            self._vprint(f'✓ Saved {filepath}')
            return filepath
            
        except ImportError:
            self._vprint('⚠️  trimesh not installed. Cannot export 3D model.')
            self._vprint('   pip install trimesh')
            return None
        except Exception as e:
            self._vprint(f'⚠️  Export failed: {e}')
            return None

    def export_video(self, fps=30, rotate_plane=None, view_direction=None, view_distance=None, synapse_size=1, 
                    html_file=None, use_existing_images=False, parallel_workers=None, **kwargs):
        '''
        Export the rotating 3-D object to a video with optimization for speed.
        
        Parameters
        ----------
        fps : int, default 30
            Frames per second, also determines rotation step size (30 degrees per second).
        rotate_plane : str, optional
            Plane to rotate: 'xy', 'xz', or 'yz'. Auto-detected based on brain_mesh.
        view_direction : tuple, optional
            Camera direction: (1, 1), (1, -1), (-1, 1), or (-1, -1). Auto-detected.
        view_distance : float, optional
            Relative camera distance from center. Auto-detected based on brain_mesh.
        synapse_size : int, default 1
            Size of synapse markers in the video.
        html_file : str, optional
            Path to existing HTML file from plot_neurons() to load figure data.
            If provided, skips plot_neurons() and loads from file (much faster).
            Example: 'path/to/existing_plot.html'
        use_existing_images : bool, default False
            If True, skip image rendering and use existing images in pics_*fps_*plane folder.
            Useful for regenerating video with different settings from cached images.
        parallel_workers : int, optional
            Number of parallel workers for frame rendering. Each worker renders frames
            independently using ProcessPoolExecutor, which can significantly speed up
            rendering on multi-core systems. Default: None (sequential rendering).
            Recommended: Set to number of CPU cores (e.g., 8-12 for modern CPUs).
            Note: Parallel rendering may use more memory.
        **kwargs : dict
            Additional arguments for plotly write_image().
            - 'scale': Resolution multiplier (default 2 for balance of quality/speed)
            - 'width', 'height': Specific dimensions in pixels
            - Lower scale = faster rendering, smaller file
        
        Returns
        -------
        int
            0 on success
        
        Examples
        --------
        # Standard usage after plot_neurons()
        vs.plot_neurons()
        vs.export_video(fps=30)
        
        # Fast re-export from existing HTML (no re-plotting needed)
        vs.export_video(fps=30, html_file='connection_data/my_plot/my_plot.html')
        
        # Use cached images to regenerate video quickly
        vs.export_video(fps=30, use_existing_images=True)
        
        # High quality but slower
        vs.export_video(fps=30, scale=4)
        
        # Fast preview
        vs.export_video(fps=15, scale=1, width=800, height=600)
        
        # Parallel rendering for speed (uses multiple CPU cores)
        vs.export_video(fps=30, scale=2, parallel_workers=8)
        '''
        # Set default parameters - always use frontal view defaults for consistency
        # rotate_plane='xz' rotates around the vertical (Y) axis for frontal view
        if rotate_plane is None:
            rotate_plane = 'xz'
        if view_direction is None:
            view_direction = (1, -1)
        if view_distance is None:
            view_distance = 2.2
        
        # Set default scale if not specified
        if kwargs.get('scale') is None and kwargs.get('width') is None and kwargs.get('height') is None:
            kwargs['scale'] = 2
        
        step = 30 / fps
        
        # Load figure from existing HTML file if provided (OPTIMIZATION)
        if html_file is not None:
            self._vprint(f'📂 Loading figure from existing HTML: {html_file}')
            if not os.path.exists(html_file):
                raise FileNotFoundError(f'HTML file not found: {html_file}')
            
            # Read and parse the HTML file to extract figure data
            import plotly.io as pio
            try:
                fig_loaded = pio.read_html(html_file)
                fig_traces = fig_loaded.data
                self._vprint(f'✓ Loaded {len(fig_traces)} traces from HTML file')
            except Exception as e:
                raise RuntimeError(f'Failed to load figure from HTML: {e}')
        else:
            # Use current figure
            if not hasattr(self, 'fig_path') or not os.path.exists(self.fig_path+'.html'):
                raise RuntimeError(
                    'No figure found. Either run plot_neurons() first or provide html_file parameter.'
                )
            html_size = os.path.getsize(self.fig_path+'.html') / 1024 / 1024 # in MB
            if html_size > 100:
                self._vprint(f'⚠️  Figure is large ({html_size:.1f} MB). Rendering may be slow.')
                self._vprint(f'   Consider using lower scale or smaller dimensions in kwargs.')
            fig_traces = self.fig_3d.data
        # Configure figure for video export
        for trace in fig_traces:
            trace.showlegend = False
            if hasattr(trace,'marker'):
                trace.marker.size = synapse_size
        
        fig_layout = go.Layout(
            margin=dict(l=1, r=1, b=1, t=1, pad=0),
        )
        fig_new = go.Figure(data=fig_traces, layout=fig_layout)
        
        # Set camera parameters - always use frontal view for consistency
        scene_camera_parameters = dict(
            up=dict(x=0, y=-1, z=0),
            eye=dict(x=0, y=0, z=-view_distance),
        )
        
        fig_new.update_layout(
            sliders=[],  # Remove sliders for cleaner video
            scene=dict(
                dragmode='orbit',
                xaxis={'visible':False}, 
                yaxis={'visible':False},
                zaxis={'visible':False},
            ),
            scene_camera=scene_camera_parameters,
        )
        
        # Set up image folder
        pic_folder = os.path.join(self.save_folder, f'pics_{fps}fps_{rotate_plane}')
        
        # Calculate rotation steps
        if step > 0:
            steps_to_write = np.linspace(0, 360, int(360/step), endpoint=False)
        elif step < 0:
            steps_to_write = np.linspace(360, 0, int(360/step), endpoint=False)
        
        # OPTIMIZATION: Skip image rendering if use_existing_images=True
        if use_existing_images and os.path.exists(pic_folder):
            existing_images = [f for f in os.listdir(pic_folder) if f.endswith('.jpeg')]
            if len(existing_images) == len(steps_to_write):
                self._vprint(f'✓ Using {len(existing_images)} existing images from {pic_folder}')
                self._vprint(f'  Skipping image rendering (use_existing_images=True)')
            else:
                self._vprint(f'⚠️  Found {len(existing_images)} images but need {len(steps_to_write)}')
                self._vprint(f'  Re-rendering images...')
                use_existing_images = False
        else:
            use_existing_images = False
        
        # Render images if needed
        if not use_existing_images:
            if os.path.exists(pic_folder):
                shutil.rmtree(pic_folder)
            os.makedirs(pic_folder)
            
            self._vprint(f'🎬 Rendering {len(steps_to_write)} frames at {fps} fps...')
            self._vprint(f'   Resolution: scale={kwargs.get("scale", "auto")}', end='')
            if 'width' in kwargs and 'height' in kwargs:
                self._vprint(f', size={kwargs["width"]}x{kwargs["height"]}')
            else:
                self._vprint()
            
            # Ensure dimensions are set to avoid blank images if not provided
            if 'width' not in kwargs: kwargs['width'] = 1200
            if 'height' not in kwargs: kwargs['height'] = 900
            
            t0 = time.time()
            
            # Note: parallel_workers is accepted but not used
            # Parallel rendering with ProcessPoolExecutor doesn't work reliably with Plotly/Kaleido
            # on macOS due to the "spawn" multiprocessing method re-executing the entire script.
            # Sequential rendering is used instead for reliability.
            if parallel_workers is not None and parallel_workers > 1:
                self._vprint(f'   ⚠️  parallel_workers={parallel_workers} ignored (not supported with Plotly/Kaleido)')
                self._vprint(f'   Using sequential rendering instead...')
            
            # Sequential rendering (reliable approach)
            self._vprint(f'   Rendering frames... (First frame may take longer to initialize Kaleido)')
            
            for i, deg in enumerate(steps_to_write):
                rad_i = np.deg2rad(deg)
                x = view_distance * np.sin(rad_i) * view_direction[0]
                y = view_distance * np.cos(rad_i) * view_direction[1]
                
                if rotate_plane == 'xy':
                    fig_new.update_layout(scene_camera=dict(eye=dict(x=x, y=y, z=0)))
                elif rotate_plane == 'yz':
                    fig_new.update_layout(scene_camera=dict(eye=dict(x=0, y=x, z=y)))
                elif rotate_plane == 'xz':
                    fig_new.update_layout(scene_camera=dict(eye=dict(x=x, y=0, z=y)))
                
                fig_path = os.path.join(pic_folder, f'deg_{deg:.1f}.jpeg')
                
                try:
                    fig_new.write_image(fig_path, **kwargs)
                except Exception as e:
                    self._vprint(f'\\n⚠️  Frame {i+1} failed: {e}')
                    if i == 0:
                        self._vprint('   Try reducing "scale" (e.g. scale=1) or using "width"/"height" parameters.')
                        return 1
                
                elapsed = time.time() - t0
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (len(steps_to_write) - i - 1)
                self._vprint(f'\\r  Frame {i+1}/{len(steps_to_write)} | '
                      f'Elapsed: {elapsed:.1f}s | '
                      f'ETA: {remaining:.1f}s | '
                      f'{avg_time:.2f}s/frame', end='    ')
            
            self._vprint('\\n✓ Image rendering complete')
        # Generate videos from images
        self._vprint(f'\\nGenerating videos...')
        imglist = os.listdir(pic_folder)
        img_eg = cv2.imread(os.path.join(pic_folder, imglist[0]))
        height, width, layers = img_eg.shape
        
        self._vprint(f'   Video resolution: {width}x{height}')

        # Forward video - OPTIMIZED with faster codec
        video_dir = os.path.join(self.save_folder, f'{self.saveas}_video_forward.mp4')
        # Use H.264 codec for better compression and compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec (faster than mp4v)
        out = cv2.VideoWriter(video_dir, fourcc, fps, frameSize=(width, height))
        
        t0 = time.time()
        for i, deg in enumerate(steps_to_write):
            img = cv2.imread(os.path.join(pic_folder, f'deg_{deg:.1f}.jpeg'))
            out.write(img)
            if (i + 1) % 10 == 0 or i == len(steps_to_write) - 1:
                self._vprint(f'\\r  Forward video: {i+1}/{len(steps_to_write)} frames', end='  ')
        out.release()
        t1 = time.time()
        self._vprint(f'\\n\u2713 Forward video: {video_dir} ({t1-t0:.1f}s)')
        
        # Backward video
        video_dir = os.path.join(self.save_folder, f'{self.saveas}_video_backward.mp4')
        out = cv2.VideoWriter(video_dir, fourcc, fps, frameSize=(width, height))
        
        t0 = time.time()
        for i, deg in enumerate(steps_to_write[::-1]):
            img = cv2.imread(os.path.join(pic_folder, f'deg_{deg:.1f}.jpeg'))
            out.write(img)
            if (i + 1) % 10 == 0 or i == len(steps_to_write) - 1:
                self._vprint(f'\\r  Backward video: {i+1}/{len(steps_to_write)} frames', end='  ')
        out.release()
        t1 = time.time()
        self._vprint(f'\\n\u2713 Backward video: {video_dir} ({t1-t0:.1f}s)')
        
        self._vprint(f'\\n\u2705 Video export complete!')
        self._vprint(f'   Image cache: {pic_folder}')
        self._vprint(f'   Tip: Use use_existing_images=True to skip re-rendering next time')
        return 0
