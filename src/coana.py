# connectome analysis module -- coana
import os
import sys
import json
import shutil
import time
import logging
from dataclasses import dataclass, field

import cv2
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import navis
import navis.interfaces.neuprint as neu
import networkx as nx
import numpy as np
import pandas as pd
import flybrains
import plotly.graph_objects as go
import seaborn as sns
from tqdm import tqdm
from neuprint import *
from neuprint.utils import connection_table_to_matrix

# Add vispath-subproject to path for VisualizePath import
vispath_src = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vispath-subproject', 'src')
if vispath_src not in sys.path:
    sys.path.insert(0, vispath_src)
from vispath_pkg import VisualizePath

# Monkey-patch for pandas 2.x compatibility
import neuprint.utils as neuprint_utils
_original_connection_table_to_matrix = connection_table_to_matrix

def _patched_connection_table_to_matrix(conn_df, group_cols='bodyId', weight_col='weight', sort_by=None, make_square=False):
    """Wrapper for connection_table_to_matrix with pandas 2.x compatibility"""
    # Call original but catch pivot() errors and retry with keyword args
    try:
        return _original_connection_table_to_matrix(conn_df, group_cols=group_cols, weight_col=weight_col, sort_by=sort_by, make_square=make_square)
    except TypeError as e:
        if 'pivot()' in str(e):
            # Manual implementation for pandas 2.x
            import neuprint.utils
            # Get the source from the function
            col_pre = f'{group_cols}_pre'
            col_post = f'{group_cols}_post'
            agg_weights_df = conn_df.groupby([col_pre, col_post], as_index=False)[weight_col].sum()
            # Use keyword arguments for pandas 2.x
            matrix = agg_weights_df.pivot(index=col_pre, columns=col_post, values=weight_col).fillna(0)
            if sort_by:
                # Sort logic from original function
                pass
            if make_square:
                all_ids = sorted(set(matrix.index) | set(matrix.columns))
                matrix = matrix.reindex(index=all_ids, columns=all_ids, fill_value=0)
            return matrix
        raise

neuprint_utils.connection_table_to_matrix = _patched_connection_table_to_matrix
connection_table_to_matrix = _patched_connection_table_to_matrix

sns.set()
from copy import copy
from datetime import datetime
from types import SimpleNamespace

import bokeh.palettes
import img2pdf

import statvis as sv
import FAFB_file_converter
import BANC_file_converter

# Ignore the navis warning
logging.getLogger('navis').setLevel(logging.WARNING)

# ============================================================================
# Module-level cache for sharing connection data across FindNeuronConnection instances
# This avoids repeated disk reads when comparison module creates multiple instances
# Structure: {dataset: {'conn_df': DataFrame, 'conn_index': dict, 'neuron_index': DataFrame, 'neuron_dict': dict}}
# ============================================================================
_FNC_CACHE = {}


def clear_fnc_cache(dataset: str = None):
    """
    Clear the module-level FindNeuronConnection cache.
    
    Args:
        dataset: Specific dataset to clear (e.g., 'hemibrain_v1_2_1'). If None, clears all.
    """
    global _FNC_CACHE
    if dataset is None:
        _FNC_CACHE.clear()
    elif dataset in _FNC_CACHE:
        del _FNC_CACHE[dataset]


@dataclass
class FindNeuronConnection:
    '''
    Through the neuprint-python API, visit the hemibrain database for connectome data analysis:\n
    https://github.com/connectome-neuprint/neuprint-python \n
    https://connectome-neuprint.github.io/neuprint-python/docs \n
    see also the following links for more information:\n
    https://github.com/connectome-neuprint/neuPrintExplorer \n
    https://neuprint.janelia.org \n
    '''

    def _reset_temp_columns(self):
        '''Reset temporary columns in source_df and target_df to allow sequential calls'''
        if hasattr(self, 'target_df'):
            cols_to_drop = [col for col in ['Checked', 'Layer'] if col in self.target_df.columns]
            if cols_to_drop:
                self.target_df = self.target_df.drop(columns=cols_to_drop)
        
        if hasattr(self, 'source_df'):
            cols_to_drop = [col for col in ['isInPath'] if col in self.source_df.columns]
            if cols_to_drop:
                self.source_df = self.source_df.drop(columns=cols_to_drop)

    def _vprint(self, message: str, level: str = 'full', end: str = '\n', flush: bool = False):
        '''Print message based on verbose_mode setting.
        
        Parameters:
        -----------
        message : str
            Message to print
        level : str
            'full': Only print if verbose_mode is 'full'
            'simple': Only print if verbose_mode is 'simple' or 'progress'
            'progress': Only print if verbose_mode is 'progress' (inline progress)
            'both': Print for 'full' and 'simple' but not 'silent'
            'always': Always print regardless of verbose_mode (even in silent)
        end : str
            End character for print (default: newline)
        flush : bool
            Whether to flush output immediately
            
        verbose_mode values:
            'full': Show all output (default)
            'simple': Show phase indicators and completion messages
            'progress': Show inline progress (overwriting single line)
            'silent': Suppress all output
        '''
        if self.verbose_mode == 'silent':
            if level == 'always':
                print(message, end=end, flush=flush)
            return
            
        if level == 'always':
            print(message, end=end, flush=flush)
        elif level == 'both':
            if self.verbose_mode in ('full', 'simple', 'progress'):
                print(message, end=end, flush=flush)
        elif level == 'full' and self.verbose_mode == 'full':
            print(message, end=end, flush=flush)
        elif level == 'simple' and self.verbose_mode in ('simple', 'progress'):
            print(message, end=end, flush=flush)
        elif level == 'progress' and self.verbose_mode == 'progress':
            # For progress mode, print with carriage return to overwrite
            print(f'\r{message}', end='', flush=True)

    def _save_matrices_to_excel(self, df, writer, level='bodyId'):
        """Generate and save connection matrices to Excel"""
        if df.empty:
            return

        # Determine columns
        if level == 'bodyId':
            index_col = 'bodyId_pre'
            columns_col = 'bodyId_post'
        else:
            index_col = 'type_pre'
            columns_col = 'type_post'
            
        # 1. Weight Matrix
        try:
            mat_weight = df.pivot(index=index_col, columns=columns_col, values='weight').fillna(0)
            mat_weight.to_excel(writer, sheet_name=f'conn_mat_{level}_weight')
        except Exception as e:
            print(f"Warning: Could not create weight matrix: {e}")

        # 2. Ratio Matrix
        if 'connection_ratio' in df.columns:
            try:
                mat_ratio = df.pivot(index=index_col, columns=columns_col, values='connection_ratio').fillna(0)
                mat_ratio.to_excel(writer, sheet_name=f'conn_mat_{level}_ratio')
            except Exception as e:
                print(f"Warning: Could not create ratio matrix: {e}")

        # 3. Probability Matrix
        if 'traversal_probability' in df.columns:
            try:
                mat_prob = df.pivot(index=index_col, columns=columns_col, values='traversal_probability').fillna(0)
                mat_prob.to_excel(writer, sheet_name=f'conn_mat_{level}_prob')
            except Exception as e:
                print(f"Warning: Could not create probability matrix: {e}")

        # 4. NT Type Matrix
        if 'nt_type' in df.columns:
            try:
                # For strings, fillna with empty string
                mat_nt = df.pivot(index=index_col, columns=columns_col, values='nt_type').fillna('')
                mat_nt.to_excel(writer, sheet_name=f'conn_mat_{level}_nt')
            except Exception as e:
                print(f"Warning: Could not create nt_type matrix: {e}")

    def _save_matrices_to_csv(self, df, folder, level='bodyId'):
        """Generate and save connection matrices to CSV"""
        if df.empty:
            return

        # Determine columns
        if level == 'bodyId':
            index_col = 'bodyId_pre'
            columns_col = 'bodyId_post'
        else:
            index_col = 'type_pre'
            columns_col = 'type_post'
            
        # 1. Weight Matrix
        try:
            mat_weight = df.pivot(index=index_col, columns=columns_col, values='weight').fillna(0)
            mat_weight.to_csv(os.path.join(folder, f'conn_mat_{level}_weight.csv'))
        except Exception as e:
            print(f"Warning: Could not create weight matrix: {e}")

        # 2. Ratio Matrix
        if 'connection_ratio' in df.columns:
            try:
                mat_ratio = df.pivot(index=index_col, columns=columns_col, values='connection_ratio').fillna(0)
                mat_ratio.to_csv(os.path.join(folder, f'conn_mat_{level}_ratio.csv'))
            except Exception as e:
                print(f"Warning: Could not create ratio matrix: {e}")

        # 3. Probability Matrix
        if 'traversal_probability' in df.columns:
            try:
                mat_prob = df.pivot(index=index_col, columns=columns_col, values='traversal_probability').fillna(0)
                mat_prob.to_csv(os.path.join(folder, f'conn_mat_{level}_prob.csv'))
            except Exception as e:
                print(f"Warning: Could not create probability matrix: {e}")

        # 4. NT Type Matrix
        if 'nt_type' in df.columns:
            try:
                mat_nt = df.pivot(index=index_col, columns=columns_col, values='nt_type').fillna('')
                mat_nt.to_csv(os.path.join(folder, f'conn_mat_{level}_nt.csv'))
            except Exception as e:
                print(f"Warning: Could not create nt_type matrix: {e}")

    def _prepare_flywire_data(self):
        '''
        Check and prepare FlyWire data from downloaded archives.
        Uses FAFB_file_converter or BANC_file_converter to ensure data validity and conversion.
        '''
        if self.client_type != 'flywire':
            return

        dataset_safe = self.dataset.replace(':', '_').replace('.', '_')
        dataset_dir = os.path.join(self.script_path, 'datasets', dataset_safe)
        
        # Use the converter module to ensure data is ready
        if 'BANC' in self.dataset:
            success = BANC_file_converter.ensure_banc_data(self.dataset, dataset_dir)
        else:
            success = FAFB_file_converter.ensure_flywire_data(self.dataset, dataset_dir)
            
        if not success:
            print("\n\033[31mCRITICAL ERROR: FlyWire/BANC data preparation failed.\033[0m")
            print("Please follow the instructions above to download the required files.")
            sys.exit(1)

    source_path: str = os.path.dirname(os.path.abspath(__file__))
    '''absolute path to the src/ directory where coana.py is located'''
    
    script_path: str = os.path.dirname(source_path)
    '''absolute path to the project root directory (parent of src/)'''
    
    data_folder: str = os.path.join(script_path, 'connection_data')
    '''folder to save all data'''
    
    save_folder: str = '' # initialized in InitializeNeuronInfo()
    '''folder to save the current data'''
    
    server: str = 'https://neuprint.janelia.org'
    '''the neuprint server to visit, see https://neuprint.janelia.org for more information'''
    
    dataset: str = 'hemibrain:v1.2.1'
    '''
    the hemibrain dataset to visit, see https://neuprint.janelia.org for more information
    All available datasets are listed below:
    'fib19:v1.0', 
    'hemibrain:v0.9', 
    'hemibrain:v1.0.1', 
    'hemibrain:v1.1', 
    'hemibrain:v1.2.1', 
    'manc:v1.0'
    '''
    
    token: str = ''
    '''
    provide your own user token for accessing the hemibrain dataset\n
    visit https://neuprint.janelia.org to get your own Auth Token, you can find it in your account information
    '''
    
    client_type: str = 'neuprint'
    '''client type: 'neuprint' (default) or 'flywire' '''

    client_hemibrain: Client | None = None
    '''neuprint client'''

    client_flywire: object | None = None
    '''flywire client adapter (deprecated)'''

    version: int | None = None
    '''Materialization version for FlyWire (e.g. 783). If None, uses default/latest.'''
    
    sourceNeurons: list = field(default_factory=list)
    '''
    Source neurons to find connection. All neurons in the list will be treated as a single source neuron group.\n
    Can be a list of neuron types or a list of neuron bodyIds, but must be a list even if only one item is in the list.\n
    All items in the list should be in the same category, that is, all types or all bodyIds.\n
    To search for all neurons, use None as input.\n
    To search for all neurons having a given type, use empty list [] as input.\n
    e.g. ['MBON01', MBON02', 'MBON03'] # neuron types\n
    e.g. ['MBON.*'] or ['MBON.*_R'] or ['.*_.*PN.*'] ... # all neurons whose type matches the regular expression\n
    e.g. [12345, 23456, 34567] # neuron bodyIds\n
    e.g. None # all neurons in the dataset\n
    e.g. [ ] or list() # all neurons having a given type\n
    All types of neurons can be found in the corresponding hemibrain dataset.\n
    see https://neuprint.janelia.org for more information.\n
    '''
    
    targetNeurons: list = field(default_factory=list)
    '''
    target neurons to find connection\n
    same as sourceNeurons
    '''
    
    largeTargetSet: bool = False
    '''if the target neuron set contains more than 16383 neurons (largeTargetSet will be set True), write excel transposed'''
    
    min_synapse_num: int = 1
    '''minimum number of synapses to be considered as connection'''
    
    min_ratio: float = 0.0
    '''
    minimum connection ratio (weight/post) to be considered as connection\n
    connection ratio is calculated as w_ij / W_j\n
    where w_ij is the number of synapses from neuron i to neuron j and W_j is the total number of post-synaptic sites of neuron j\n
    This is the direct ratio without the 0.3 scaling factor used in traversal_probability
    '''
    
    min_traversal_probability: float = 0.0
    '''
    minimum traversal probability to be considered as connection\n
    traversal probability is calculated as \n
    max{1, w_ij / (W_j*0.3)}\n
    where w_ij is the number of synapses from neuron i to neuron j and W_j is the total number of post-synaptic sites of neuron j
    '''
    
    filter_by: str = 'bodyId'
    '''
    Level at which to apply min_synapse_num, min_ratio, and min_traversal_probability filters\n
    - 'bodyId': Filter at individual neuron (bodyId) level (default)\n
    - 'type': Filter at aggregated type-to-type level after grouping connections by type\n
    When 'type' is used, connections between neurons of the same type are merged first,\n
    then filters are applied to the aggregated weights
    '''
    
    exclude_intra_type_connections: bool = False
    '''
    whether to exclude connections within the same neuron type (type_pre == type_post)\n
    when True, removes all connections where source and target neurons have the same type\n
    when False (default), keeps all connections including intra-type connections\n
    applies to all connection searches (FindDirect, FindPath, FindAllPath)\n
    This feature is particularly useful when analyzing cross-type connectivity patterns\n
    while excluding self-connections within the same neuron type.\n
    It's also useful when building networks and illustrating connections of given neurons,\n
    helping to focus on inter-type communication pathways.
    '''
    
    max_interlayer: int = 1
    '''
    Maximum number of interlayers to be considered in connection.
    Values:
      -1: Fetch source/target neurons only (no connections). Use FetchNeuronsOnly().
       0: Direct connections only. Use FindDirectConnections().
       1, 2, ...: Include interlayer connections. Use FindAllPath() or FindPath().
    '''
    
    run_date: str = datetime.now().strftime('%Y%m%d_%H%M%S')
    '''date and time when the script is run'''
    
    source_fname: str = ''
    '''auto-generated file name for source neurons'''
    
    source_criteria: NeuronCriteria | None = None
    '''auto-generated neuron criteria for source neurons'''
    
    target_criteria: NeuronCriteria | None = None
    '''auto-generated neuron criteria for target neurons'''
    
    target_fname: str = ''
    '''auto-generated file name for target neurons'''
    
    custom_source_name: str = ''
    '''custom name for source neurons, used in plot and file name'''
    
    custom_target_name: str = ''
    '''custom name for target neurons, used in plot and file name'''
    
    custom_source_group_names: list = field(default_factory=list)
    '''custom names for source neuron groups when using nested lists. If empty, auto-generated names will be used.'''
    
    custom_target_group_names: list = field(default_factory=list)
    '''custom names for target neuron groups when using nested lists. If empty, auto-generated names will be used.'''
    
    folder_prefix: str = ''
    '''prefix for the auto-generated save folder name'''
    
    saveas: str = ''
    '''
    custom folder name or absolute path for output. 
    If relative, it's created inside data_folder. 
    If absolute, it overrides data_folder.
    '''

    parameter_dict = dict()
    '''dictionary to store all specified parameters'''
    
    parameter_df = pd.DataFrame()
    '''dataframe to store all specified parameters, converted from parameter_dict'''
    
    showfig: bool = False
    '''whether to show the figures'''
    
    link_color: str = 'rgba(100,150,240,0.2)'
    '''link color for Sankey diagram'''
    
    node_color: str = 'rgba(60,100,200,0.5)'
    '''node color for Sankey diagram'''
    
    target_color: str = 'rgba(120,40,70,0.7)'
    '''target node color for Sankey diagram, only works when interlayers exist'''
    
    default_mesh_rois = ['LH(R)','AL(R)','EB']
    '''default mesh rois to be plotted'''
    
    keyword_in_path_to_remove: str | list[str] = 'None'
    '''path blocks including these keywords will be removed. Can be a single keyword string or a list of keywords.'''
    
    network_layout: str = 'layered'
    '''
    layout algorithm for interactive network visualization\n
    'layered': multipartite layout - arranges nodes in distinct layers (good for strictly hierarchical networks)\n
    'distributed': spring layout - distributes nodes for better clarity (good for networks with cross-layer connections)\n
    '''
    
    simple_fetch: bool = True
    '''
    when True, use neuprint.fetch_simple_connections() to fetch connections, for small sets of neurons and fast speed\n
    when False, use neuprint.fetch_adjacencies(), for large sets of neurons but slower
    '''
    
    kwargs_fetch: dict = field(default_factory=dict)
    '''
    kwargs to be passed to neuprint.fetch_simple_connections() or neuprint.fetch_adjacencies() \n
    upstream_criteria, downstream_criteria, min_weight of fetch_simple_connections() should NOT be included here \n
    sources, targets, min_total_weight of fetch_adjacencies() should NOT be included here \n
    they should be specified in sourceNeurons, targetNeurons, min_synapse_num \n
    see more in: \n
    https://connectome-neuprint.github.io/neuprint-python/docs/queries.html#neuprint.queries.fetch_simple_connections \n 
    and \n
    https://connectome-neuprint.github.io/neuprint-python/docs/queries.html#neuprint.queries.fetch_adjacencies \n
    '''
    
    output_format: str = 'csv'
    '''
    output data format: 'xlsx' (default) or 'csv'\n
    'xlsx': save all data in Excel files\n
    'csv': save all data in CSV files in a subfolder named 'output_data'
    '''
    
    use_cache: bool = True
    '''
    when True, save fetched connection data to local cache and check cache before fetching from API\n
    when False, always fetch from API (slower but ensures latest data)\n
    Cache is stored in: cache/{dataset}/connections/ (in project root)\n
    '''
    
    use_parallel: bool = False
    '''
    whether to use parallel processing for pathfinding (PHASE 3)\n
    when True, uses multiprocessing to speed up path searches on multi-core systems\n
    recommended for large datasets (>10000 source-target pairs)\n
    set to False if you encounter issues or prefer sequential processing
    '''
    
    n_jobs: int = -1
    '''
    number of parallel processes for pathfinding\n
    -1: use all available CPU cores\n
    1: sequential processing (same as use_parallel=False)\n
    n > 1: use n parallel processes\n
    only used when use_parallel=True
    '''
    
    cache_folder: str = ''
    '''folder to store cached data, automatically set based on dataset'''
    
    edgeN_limit: int = 500
    '''
    number of strongest edges to show in network visualization\n
    -1: show all edges\n
    n > 0: show only the top n edges ranked by weight (default: 1000)\n
    applies to VisualizePath visualizations\n
    helps focus on most significant connections in large networks and prevents browser crashes\n
    '''
    
    pathN_to_show: int = -1
    '''
    [DEPRECATED] Use edgeN_limit instead.\n
    number of strongest paths to show in network visualization\n
    -1: show all paths (default)\n
    n > 0: show only the top n paths ranked by traversal_probability (product of edge probabilities)\n
    applies to both FindPath and FindAllPath visualizations\n
    helps focus on most significant pathways in large networks\n
    Note: paths are already sorted by traversal_probability in the path_type/path_bodyId DataFrames
    '''
    
    verbose_mode: str = 'full'
    '''
    Controls the verbosity of output during FindAllPath execution.\n
    'full': Show all detailed output (default) - layer-by-layer info, statistics, etc.\n
    'simple': Show simplified progress output with phase markers and progress bars only.\n
    The simple mode shows:
      - Phase 1: layer 0->1: processing...Done
      - Phase 2: identifying targets...Done
      - Phase 3: pathfinding[parallel/sequential]...Done
      - Phase 4: creating visualizations...Done
      - ¡COMPLETED! banner
    '''
    
    def __post_init__(self):
        self._vprint('Initializing...', level='full')
        
        # Auto-detect client_type from dataset if not explicitly set to flywire
        if self.client_type == 'neuprint' and ('flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower()):
            self.client_type = 'flywire'
            self._vprint(f"Auto-detected client_type='flywire' from dataset '{self.dataset}'", level='full')

        # Auto-detect version from dataset if not provided
        if self.client_type == 'flywire' and self.version is None:
            import re
            # Look for v783 or version 783
            match = re.search(r'v(\d+)', self.dataset)
            if match:
                self.version = int(match.group(1))
                self._vprint(f"Auto-detected version={self.version} from dataset '{self.dataset}'", level='full')
        
        # Prepare FlyWire data if needed
        if self.client_type == 'flywire':
            self._prepare_flywire_data()
        
        # Initialize NeuPrint client if needed
        if self.client_type == 'neuprint' and self.client_hemibrain is None:
            from neuprint import Client, set_default_client, default_client
            # Only login if not already done (default_client() raises RuntimeError if not set)
            try:
                client = default_client()
            except RuntimeError:
                client = None
            
            if client is None:
                self._vprint(f"Initializing NeuPrint client for dataset: {self.dataset}", level='full')
                self.client_hemibrain = Client(self.server, self.dataset, self.token)
                set_default_client(self.client_hemibrain)
            else:
                self.client_hemibrain = client

        # Validate filter_by parameter
        if self.filter_by not in ['bodyId', 'type']:
            raise ValueError(f"filter_by must be 'bodyId' or 'type', got '{self.filter_by}'")
        
        # Initialize cache folder and in-memory cache structures
        # Try to use module-level shared cache first (avoids repeated disk reads)
        dataset_safe = self.dataset.replace(':', '_').replace('.', '_')
        self._dataset_safe = dataset_safe
        
        # Check module-level cache first
        global _FNC_CACHE
        if dataset_safe in _FNC_CACHE:
            cache = _FNC_CACHE[dataset_safe]
            self._conn_df_cache = cache.get('conn_df')
            self._conn_index = cache.get('conn_index')
            self._neuron_index_cache = cache.get('neuron_index')
            self._neuron_index_dict = cache.get('neuron_dict')
            self._vprint(f'Using shared module cache for {dataset_safe}', level='full')
        else:
            # Initialize empty caches (will be populated on first load)
            self._conn_df_cache = None  # DataFrame cache for connections
            self._conn_index = None  # Dict: bodyId_pre → list of row indices
            self._neuron_index_cache = None  # DataFrame cache for neuron index
            self._neuron_index_dict = None  # Dict: bodyId → row data dict
        
        if self.use_cache:
            self.cache_folder = os.path.join(self.script_path, 'cache', dataset_safe)
            os.makedirs(self.cache_folder, exist_ok=True)
            self._vprint(f'Cache enabled: {self.cache_folder}', level='full')
            # Ensure complete dataset with ALL neurons exists (including type=None)
            self._ensure_complete_dataset()
        if self.exclude_intra_type_connections:
            self._vprint('⚠️  Intra-type connections will be excluded (type_pre == type_post)', level='full')
        if self.sourceNeurons is None or self.targetNeurons is None:
            self._vprint('\033[33mIt is not recommended to search for all neurons in the dataset.\n Using [] or list() to search for all neurons having a given type, instead.\033[0m', level='full')
        elif self.targetNeurons is None:
            self.largeTargetSet = True
    
    def _ensure_complete_dataset(self):
        '''
        Ensure complete local dataset exists (including neurons with type=None).
        This is needed for cache enrichment since cached connections may reference
        neurons without types.
        '''
        if self.client_type == 'flywire':
            self._vprint("   Skipping complete dataset download for FlyWire (too large). Cache enrichment will rely on on-demand fetching.", level='full')
            return

        # Create datasets folder if it doesn't exist
        datasets_folder = os.path.join(self.script_path, 'datasets')
        if not os.path.exists(datasets_folder):
            os.makedirs(datasets_folder)
            self._vprint(f'Created datasets folder: {datasets_folder}', level='full')
        
        dataset_safe = self.dataset.replace(':', '_').replace('.', '_')
        dataset_dir = os.path.join(datasets_folder, dataset_safe)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            self._vprint(f'Created dataset folder: {dataset_dir}', level='full')

        dataset_path = os.path.join(
            dataset_dir, 
            f"{dataset_safe}_allneurons"
        )
        
        neuron_csv = dataset_path + '_neuron_df.csv'
        roi_csv = dataset_path + '_roi_count_df.csv'
        
        if not os.path.exists(neuron_csv) or not os.path.exists(roi_csv):
            self._vprint(f'\n📥 Complete dataset not found, downloading ALL neurons (including type=None)...', level='full')
            self._vprint(f'   This is a one-time download for cache enrichment.', level='full')
            # Login to neuprint only if needed
            from neuprint import Client, set_default_client, default_client
            # Only login if not already done (default_client() returns None if not set)
            if self.client_hemibrain is None and default_client() is None:
                self.client_hemibrain = Client(self.server, self.dataset, self.token)
                set_default_client(self.client_hemibrain)
            try:
                # Pull complete dataset with omitNoneType=False
                sv.pull_dataset(self.dataset, save_path=dataset_path, omitNoneType=False)
                self._vprint(f'✅ Complete dataset saved to: {dataset_path}_*.csv', level='full')
            except Exception as e:
                self._vprint(f'⚠️ Warning: Failed to download complete dataset: {e}', level='full')
                self._vprint(f'   Cache enrichment may fail for neurons without types.', level='full')
    
    # ============================================================================
    # Core Database Access
    # ============================================================================
    
    def _get_connection_db_path(self):
        '''Get path to unified connection database'''
        return os.path.join(self.cache_folder, 'connections.parquet')
    
    def _get_neuron_index_path(self):
        '''Get path to neuron index (tracks cached neurons)'''
        return os.path.join(self.cache_folder, 'neuron_index.parquet')
    
    def _load_connection_db(self, force_reload=False):
        '''
        Load unified connection database with in-memory caching and O(1) index.
        
        On first load, reads parquet from disk and builds a dict index for fast lookups.
        Subsequent calls return the cached DataFrame without disk I/O.
        
        Schema: bodyId_pre, bodyId_post, weight, roi (optional), cached_date
        
        Parameters:
        -----------
        force_reload : bool
            If True, reload from disk even if cached in memory
        
        Returns:
        --------
        pd.DataFrame : Connection database
        '''
        # Return cached DataFrame if available
        if self._conn_df_cache is not None and not force_reload:
            return self._conn_df_cache
        
        db_path = self._get_connection_db_path()
        
        # Special handling for FlyWire: Import from CSV if cache missing
        if not os.path.exists(db_path) and self.client_type == 'flywire':
            self._vprint(f'  ⏳ FlyWire cache missing. Importing from local CSV...', level='full')
            
            csv_path = None
            dataset_safe = self.dataset.replace(':', '_').replace('.', '_')
            dataset_dir = os.path.join(self.script_path, 'datasets', dataset_safe)
            
            import glob
            merged_candidates = glob.glob(os.path.join(dataset_dir, "*_merged_connections.csv"))
            if merged_candidates:
                csv_path = merged_candidates[0]
            
            if csv_path and os.path.exists(csv_path):
                try:
                    self._vprint(f'  ⏳ Reading {csv_path} (this may take a while)...', level='full')
                    df = pd.read_csv(csv_path, dtype={'pre_root_id': str, 'post_root_id': str, 'bodyId_pre': str, 'bodyId_post': str})
                    
                    column_map = {
                        'pre_root_id': 'bodyId_pre',
                        'post_root_id': 'bodyId_post',
                        'syn_count': 'weight',
                        'neuropil': 'roi',
                        'pre': 'bodyId_pre',
                        'post': 'bodyId_post',
                        'synapses': 'weight'
                    }
                    df = df.rename(columns=column_map)
                    
                    if 'weight' not in df.columns:
                        df['weight'] = 1
                    if 'roi' not in df.columns:
                        df['roi'] = 'None'
                    if 'cached_date' not in df.columns:
                        df['cached_date'] = datetime.now().strftime("%Y-%m-%d")
                        
                    cols_to_keep = ['bodyId_pre', 'bodyId_post', 'weight', 'roi', 'nt_type', 'cached_date']
                    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
                    df = df[cols_to_keep]
                    
                    df['bodyId_pre'] = df['bodyId_pre'].astype(str)
                    df['bodyId_post'] = df['bodyId_post'].astype(str)
                    
                    self._vprint(f'  ✓ Imported {len(df):,} connections from CSV', level='full')
                    
                    self._vprint(f'  💾 Saving to cache for faster future access...', level='full')
                    df.to_parquet(db_path, index=False, compression='gzip')
                    
                    # Cache in memory and build index
                    self._conn_df_cache = df
                    self._build_conn_index()
                    return df
                except Exception as e:
                    self._vprint(f'  ⚠️ Error importing FlyWire CSV: {e}', level='full')
        
        if os.path.exists(db_path):
            try:
                file_size_mb = os.path.getsize(db_path) / (1024 * 1024)
                self._vprint(f'  ⏳ Loading connection database ({file_size_mb:.1f} MB)...', level='full')
                df = pd.read_parquet(db_path)
                
                if 'bodyId_pre' in df.columns:
                    df['bodyId_pre'] = df['bodyId_pre'].astype(str)
                if 'bodyId_post' in df.columns:
                    df['bodyId_post'] = df['bodyId_post'].astype(str)
                    
                self._vprint(f'  ✓ Loaded {len(df):,} cached connections', level='full')
                
                # Cache in memory and build index
                self._conn_df_cache = df
                self._build_conn_index()
                return df
            except Exception as e:
                self._vprint(f'  ⚠️ Warning: Failed to load connection database: {e}', level='full')
                self._conn_df_cache = pd.DataFrame(columns=['bodyId_pre', 'bodyId_post', 'weight', 'roi', 'cached_date'])
                self._conn_index = {}
                return self._conn_df_cache
        
        self._conn_df_cache = pd.DataFrame(columns=['bodyId_pre', 'bodyId_post', 'weight', 'roi', 'cached_date'])
        self._conn_index = {}
        return self._conn_df_cache
    
    def _build_conn_index(self):
        '''
        Build dict index for O(1) connection lookups by bodyId_pre.
        Called after loading connection database from disk.
        Also updates the module-level shared cache.
        '''
        if self._conn_df_cache is None or self._conn_df_cache.empty:
            self._conn_index = {}
            return
        
        self._vprint(f'  ⏳ Building connection index for fast lookups...', level='full')
        self._conn_index = {}
        
        # Group by bodyId_pre and store row indices
        for idx, bodyId_pre in enumerate(self._conn_df_cache['bodyId_pre'].values):
            if bodyId_pre not in self._conn_index:
                self._conn_index[bodyId_pre] = []
            self._conn_index[bodyId_pre].append(idx)
        
        self._vprint(f'  ✓ Index built: {len(self._conn_index):,} unique upstream neurons', level='full')
        
        # Update module-level shared cache for other instances
        global _FNC_CACHE
        if hasattr(self, '_dataset_safe'):
            if self._dataset_safe not in _FNC_CACHE:
                _FNC_CACHE[self._dataset_safe] = {}
            _FNC_CACHE[self._dataset_safe]['conn_df'] = self._conn_df_cache
            _FNC_CACHE[self._dataset_safe]['conn_index'] = self._conn_index
    
    def _save_connection_db(self, conn_db):
        '''
        Save unified connection database with compression.
        Also updates the in-memory cache and rebuilds the index.
        '''
        db_path = self._get_connection_db_path()
        try:
            conn_db.to_parquet(db_path, index=False, compression='gzip')
            self._vprint(f'  ✓ Database saved successfully', level='full')
            
            # Update in-memory cache
            self._conn_df_cache = conn_db
            self._build_conn_index()
        except Exception as e:
            self._vprint(f'  ⚠️ Warning: Failed to save connection database: {e}', level='full')
    
    def _load_neuron_index(self, force_reload=False):
        '''
        Load neuron index with in-memory caching and O(1) dict lookup.
        
        On first load, reads parquet from disk and builds a dict for fast lookups.
        Subsequent calls return the cached DataFrame without disk I/O.
        
        Schema: bodyId, type, instance, post, downstream_complete, last_fetched, connection_count
        
        Parameters:
        -----------
        force_reload : bool
            If True, reload from disk even if cached in memory
        
        Returns:
        --------
        pd.DataFrame : Neuron index
        '''
        # Return cached DataFrame if available
        if self._neuron_index_cache is not None and not force_reload:
            return self._neuron_index_cache
        
        index_path = self._get_neuron_index_path()
        
        # Special handling for FlyWire: Import from enriched CSV if cache missing
        if not os.path.exists(index_path) and self.client_type == 'flywire':
            self._vprint(f'  ⏳ FlyWire index missing. Importing from enriched CSV...', level='full')
            dataset_safe = self.dataset.replace(':', '_').replace('.', '_')
            csv_path = os.path.join(self.script_path, 'datasets', dataset_safe, f"{dataset_safe}_allneurons_neuron_df.csv")
            
            if os.path.exists(csv_path):
                try:
                    self._vprint(f'  ⏳ Reading {csv_path}...', level='full')
                    df = pd.read_csv(csv_path, dtype={'bodyId': str})
                    
                    if 'instance' not in df.columns:
                        df['instance'] = df['name'] if 'name' in df.columns else ''
                    if 'post' not in df.columns:
                        df['post'] = 0
                    
                    df['downstream_complete'] = True
                    df['last_fetched'] = datetime.now().strftime("%Y-%m-%d")
                    df['connection_count'] = df['post']
                    
                    cols_to_keep = ['bodyId', 'type', 'instance', 'post', 'downstream_complete', 'last_fetched', 'connection_count']
                    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
                    df = df[cols_to_keep]
                    
                    self._vprint(f'  ✓ Imported {len(df):,} neurons from CSV', level='full')
                    
                    self._vprint(f'  💾 Saving to cache...', level='full')
                    df.to_parquet(index_path, index=False, compression='gzip')
                    
                    # Cache in memory and build dict
                    self._neuron_index_cache = df
                    self._build_neuron_index_dict()
                    return df
                except Exception as e:
                    self._vprint(f'  ⚠️ Error importing FlyWire Index: {e}', level='full')

        if os.path.exists(index_path):
            try:
                file_size_mb = os.path.getsize(index_path) / (1024 * 1024)
                if file_size_mb > 1:
                    self._vprint(f'  ⏳ Loading neuron index ({file_size_mb:.1f} MB)...', level='full')
                df = pd.read_parquet(index_path)
                
                if 'bodyId' in df.columns:
                    df['bodyId'] = df['bodyId'].astype(str)
                    
                if file_size_mb > 1:
                    self._vprint(f'  ✓ Loaded index for {len(df):,} neurons', level='full')
                
                # Cache in memory and build dict
                self._neuron_index_cache = df
                self._build_neuron_index_dict()
                return df
            except Exception as e:
                self._vprint(f'  ⚠️ Warning: Failed to load neuron index: {e}', level='full')
                self._neuron_index_cache = pd.DataFrame(columns=[
                    'bodyId', 'type', 'instance', 'post', 'downstream_complete', 
                    'last_fetched', 'connection_count'
                ])
                self._neuron_index_dict = {}
                return self._neuron_index_cache
        
        self._neuron_index_cache = pd.DataFrame(columns=[
            'bodyId', 'type', 'instance', 'post', 'downstream_complete',
            'last_fetched', 'connection_count'
        ])
        self._neuron_index_dict = {}
        return self._neuron_index_cache
    
    def _build_neuron_index_dict(self):
        '''
        Build dict for O(1) neuron index lookups by bodyId.
        Called after loading neuron index from disk.
        Also updates the module-level shared cache.
        '''
        if self._neuron_index_cache is None or self._neuron_index_cache.empty:
            self._neuron_index_dict = {}
            return
        
        self._vprint(f'  ⏳ Building neuron index dict for fast lookups...', level='full')
        self._neuron_index_dict = {}
        
        # Build dict: bodyId → {downstream_complete: bool, ...}
        for idx, row in self._neuron_index_cache.iterrows():
            bodyId = str(row['bodyId'])
            self._neuron_index_dict[bodyId] = {
                'downstream_complete': row.get('downstream_complete', False),
                'type': row.get('type', ''),
                'instance': row.get('instance', ''),
                'post': row.get('post', 0),
                'last_fetched': row.get('last_fetched', ''),
                'connection_count': row.get('connection_count', 0),
                'row_idx': idx  # Store row index for DataFrame updates
            }
        
        self._vprint(f'  ✓ Neuron index dict built: {len(self._neuron_index_dict):,} neurons', level='full')
        
        # Update module-level shared cache for other instances
        global _FNC_CACHE
        if hasattr(self, '_dataset_safe'):
            if self._dataset_safe not in _FNC_CACHE:
                _FNC_CACHE[self._dataset_safe] = {}
            _FNC_CACHE[self._dataset_safe]['neuron_index'] = self._neuron_index_cache
            _FNC_CACHE[self._dataset_safe]['neuron_dict'] = self._neuron_index_dict
    
    def _save_neuron_index(self, index_df):
        '''
        Save neuron index with compression.
        Also updates the in-memory cache and rebuilds the dict.
        '''
        index_path = self._get_neuron_index_path()
        try:
            index_df.to_parquet(index_path, index=False, compression='gzip')
            self._vprint(f'  ✓ Neuron index saved successfully', level='full')
            
            # Update in-memory cache
            self._neuron_index_cache = index_df
            self._build_neuron_index_dict()
        except Exception as e:
            self._vprint(f'  ⚠️ Warning: Failed to save neuron index: {e}', level='full')
    
    # ============================================================================
    # Query Resolution Logic
    # ============================================================================
    
    def _query_connection_db(self, upstream_bodyIds, downstream_bodyIds=None):
        '''
        Query unified connection database for specific connections using O(1) dict lookups.
        Returns (cached_df, uncached_upstream_ids)
        
        Parameters:
        -----------
        upstream_bodyIds : list
            List of upstream neuron bodyIds to query
        downstream_bodyIds : list or None
            List of downstream neuron bodyIds (None = all downstream)
        
        Returns:
        --------
        tuple: (cached_connections_df, list_of_uncached_upstream_ids, list_of_partially_cached_ids)
        '''
        if not self.use_cache:
            return pd.DataFrame(), upstream_bodyIds, []
        
        self._vprint(f'  ⏳ Querying cache for {len(upstream_bodyIds):,} neurons...', level='full')
        
        # Load caches (uses in-memory if already loaded)
        conn_db = self._load_connection_db()
        neuron_index = self._load_neuron_index()
        
        if conn_db.empty:
            return pd.DataFrame(), upstream_bodyIds, []
        
        # Separate cached vs uncached neurons using O(1) dict lookups
        cached_upstream = []
        uncached_upstream = []
        partially_cached = []
        
        for bodyId in upstream_bodyIds:
            bodyId = str(bodyId)
            
            # O(1) dict lookup instead of O(n) DataFrame scan
            neuron_data = self._neuron_index_dict.get(bodyId)
            
            if neuron_data is not None:
                is_complete = neuron_data.get('downstream_complete', False)
                
                if downstream_bodyIds is None:
                    if is_complete:
                        cached_upstream.append(bodyId)
                    else:
                        uncached_upstream.append(bodyId)
                else:
                    if is_complete:
                        cached_upstream.append(bodyId)
                    else:
                        uncached_upstream.append(bodyId)
            else:
                uncached_upstream.append(bodyId)
        
        # Retrieve cached connections using O(1) dict index
        all_cached = cached_upstream + partially_cached  # partially_cached will be empty (no recovery)
        if len(all_cached) > 0:
            self._vprint(f'  ⏳ Retrieving {len(all_cached):,} neurons from cache...', level='full')
            
            # Use dict index for O(1) lookups instead of DataFrame filter
            row_indices = []
            for bodyId in all_cached:
                if bodyId in self._conn_index:
                    row_indices.extend(self._conn_index[bodyId])
            
            if row_indices:
                cached_conn = conn_db.iloc[row_indices].copy()
            else:
                cached_conn = pd.DataFrame()
            
            # Filter by downstream if specified
            if downstream_bodyIds is not None and not cached_conn.empty:
                downstream_set = set(str(b) for b in downstream_bodyIds)
                cached_conn = cached_conn[cached_conn['bodyId_post'].isin(downstream_set)].copy()
            
            # Note: Neurons with 0 connections are valid! Don't refetch them.
            # The neuron_index already tracks which neurons are complete via downstream_complete flag.
            # Only refetch if they're marked incomplete (which is already handled above in the loop).
            
            # Return both cached connections and list of partially cached neurons for later marking
            return cached_conn, uncached_upstream, partially_cached
        
        return pd.DataFrame(), uncached_upstream, []
    
    def _try_recover_neuron_metadata(self, bodyId, conn_db, neuron_index):
        '''
        Attempt to recover neuron metadata from local dataset and add to neuron index.
        Called during crash recovery when connections exist but neuron not in index.
        
        Parameters:
        -----------
        bodyId : int
            Neuron bodyId to recover
        conn_db : pd.DataFrame
            Connection database (used to count existing connections)
        neuron_index : pd.DataFrame
            Current neuron index (will be updated and saved)
        '''
        # Try to get metadata from local dataset
        dataset_safe = self.dataset.replace(':', '_').replace('.', '_')
        dataset_path = os.path.join(
            self.script_path,
            'datasets',
            dataset_safe,
            f"{dataset_safe}_allneurons_neuron_df.csv"
        )
        
        if os.path.exists(dataset_path):
            try:
                # Check if it's FAFB to decide on index_col (FAFB utils saves without index)
                is_fafb = 'fafb' in self.dataset.lower() or 'flywire' in self.dataset.lower()
                
                if is_fafb:
                    ndf_complete = pd.read_csv(dataset_path, header=0, index_col=None, dtype={'bodyId': str}, low_memory=False)
                else:
                    ndf_complete = pd.read_csv(dataset_path, header=0, index_col=0, low_memory=False)
                
                # Ensure bodyId is string
                if 'bodyId' in ndf_complete.columns:
                    ndf_complete['bodyId'] = ndf_complete['bodyId'].astype(str)
                    
                neuron_row = ndf_complete[ndf_complete['bodyId'] == str(bodyId)]
                
                if not neuron_row.empty:
                    # Found neuron metadata - add to index as incomplete
                    neuron_type = neuron_row.iloc[0]['type'] if 'type' in neuron_row.columns else ''
                    neuron_instance = neuron_row.iloc[0]['instance'] if 'instance' in neuron_row.columns else ''
                    neuron_post = neuron_row.iloc[0]['post'] if 'post' in neuron_row.columns else 0
                    
                    # Count connections from database
                    conn_count = len(conn_db[conn_db['bodyId_pre'] == bodyId])
                    
                    # Add to neuron index (but not marked as complete yet)
                    new_entry = pd.DataFrame([{
                        'bodyId': bodyId,
                        'type': neuron_type,
                        'instance': neuron_instance,
                        'post': neuron_post,
                        'downstream_complete': False,  # Not complete yet - needs enrichment validation
                        'last_fetched': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'connection_count': conn_count
                    }])
                    
                    # Update and save neuron_index immediately
                    updated_index = pd.concat([neuron_index, new_entry], ignore_index=True)
                    self._save_neuron_index(updated_index)
                    
                    return True  # Successfully recovered
            except Exception as e:
                # If recovery fails, just skip - will be marked as complete after enrichment
                pass
        
        return False  # Recovery failed or not possible
    
    def _update_connection_db(self, new_connections, upstream_bodyIds, downstream_bodyIds=None):
        '''
        Add new connections to unified database without duplicates.
        Updates neuron index to mark neurons as fully cached (if querying all downstream).
        
        Parameters:
        -----------
        new_connections : pd.DataFrame
            New connections to add (must have bodyId_pre, bodyId_post, weight, optionally roi)
        upstream_bodyIds : list
            List of upstream neurons that were queried
        downstream_bodyIds : list or None
            If None, marks neurons as downstream_complete. If list, doesn't mark as complete.
        '''
        # Even if no new connections, still update the neuron index to mark as complete
        if new_connections.empty:
            # Update neuron index to mark neurons as fetched (even if they have 0 connections)
            self._update_neuron_index_after_fetch(new_connections, upstream_bodyIds, downstream_bodyIds)
            return
        
        # Load existing database
        conn_db = self._load_connection_db()
        
        # Prepare new connections
        new_conn = new_connections[['bodyId_pre', 'bodyId_post', 'weight']].copy()
        
        # Ensure bodyIds are strings
        new_conn['bodyId_pre'] = new_conn['bodyId_pre'].astype(str)
        new_conn['bodyId_post'] = new_conn['bodyId_post'].astype(str)
        
        if 'roi' in new_connections.columns:
            new_conn['roi'] = new_connections['roi']
        else:
            new_conn['roi'] = ''
        
        new_conn['cached_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Merge with existing, removing duplicates (keep existing entries)
        if not conn_db.empty:
            self._vprint(f'  ⏳ Merging {len(new_conn):,} connections with existing database...', level='full')
            # Remove any new connections that already exist (based on bodyId_pre, bodyId_post, roi)
            merge_cols = ['bodyId_pre', 'bodyId_post', 'roi']
            combined = pd.concat([conn_db, new_conn])
            combined = combined.drop_duplicates(subset=merge_cols, keep='first')
        else:
            combined = new_conn
        
        # Save updated database
        self._vprint(f'  ⏳ Saving connection database ({len(combined):,} connections)...', level='full')
        self._save_connection_db(combined)
        
        new_count = len(combined) - len(conn_db)
        if new_count > 0:
            self._vprint(f'  💾 Added {new_count} new connections to database (total: {len(combined):,})', level='full')
        else:
            self._vprint(f'  📂 All connections already in database ({len(conn_db):,} total)', level='full')
        
        # Update neuron index
        self._update_neuron_index_after_fetch(new_conn, upstream_bodyIds, downstream_bodyIds)
    
    def _save_connections_only(self, new_connections, upstream_bodyIds):
        '''
        Save connections to database without updating neuron index.
        Used when we want to delay marking neurons as cached until after enrichment succeeds.
        
        Parameters:
        -----------
        new_connections : pd.DataFrame
            New connections to add (must have bodyId_pre, bodyId_post, weight, optionally roi)
        upstream_bodyIds : list
            List of upstream neurons that were queried (not marked as cached yet)
        '''
        if new_connections.empty:
            self._vprint(f'  📂 No connections found for {len(upstream_bodyIds)} neurons', level='full')
            return
        
        # Load existing database
        conn_db = self._load_connection_db()
        
        # Prepare new connections
        new_conn = new_connections[['bodyId_pre', 'bodyId_post', 'weight']].copy()
        
        # Ensure bodyIds are strings
        new_conn['bodyId_pre'] = new_conn['bodyId_pre'].astype(str)
        new_conn['bodyId_post'] = new_conn['bodyId_post'].astype(str)
        
        if 'roi' in new_connections.columns:
            new_conn['roi'] = new_connections['roi']
        else:
            new_conn['roi'] = ''
        
        new_conn['cached_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Merge with existing, removing duplicates (keep existing entries)
        if not conn_db.empty:
            self._vprint(f'  ⏳ Merging {len(new_conn):,} connections with existing database...', level='full')
            merge_cols = ['bodyId_pre', 'bodyId_post', 'roi']
            combined = pd.concat([conn_db, new_conn])
            combined = combined.drop_duplicates(subset=merge_cols, keep='first')
        else:
            combined = new_conn
        
        # Save updated database
        self._vprint(f'  ⏳ Saving connection database ({len(combined):,} connections)...', level='full')
        self._save_connection_db(combined)
        
        new_count = len(combined) - len(conn_db)
        if new_count > 0:
            self._vprint(f'  💾 Added {new_count} new connections to database (total: {len(combined):,})', level='full')
        else:
            self._vprint(f'  📂 All connections already in database ({len(conn_db):,} total)', level='full')
    
    def _mark_neurons_as_cached(self, upstream_bodyIds, connections, downstream_bodyIds=None):
        '''
        Mark neurons as cached in neuron index after successful enrichment.
        This is called AFTER enrichment to ensure data integrity.
        Neurons with empty/None type are valid and will be marked as complete.
        Neurons with 0 connections are valid and will be marked as complete.
        
        Parameters:
        -----------
        upstream_bodyIds : list
            List of upstream neurons to mark as cached
        connections : pd.DataFrame
            Successfully fetched and enriched connections (may be empty for neurons with 0 connections)
        downstream_bodyIds : list or None
            If None, marks neurons as downstream_complete. If list, doesn't mark as complete.
        '''
        # If connections is empty, all neurons have 0 connections - that's valid, mark them all
        if connections.empty:
            self._update_neuron_index_after_fetch(connections, upstream_bodyIds, downstream_bodyIds)
            return
        
        # Validate that connections are properly enriched before marking
        required_cols = ['bodyId_pre', 'bodyId_post', 'weight', 'type_pre', 'instance_pre']
        missing_cols = [col for col in required_cols if col not in connections.columns]
        if missing_cols:
            self._vprint(f'  ⚠️  Warning: Connections missing required columns {missing_cols}, skipping cache update', level='full')
            return
        
        # Note: Neurons with None or empty type/instance are VALID
        # The dataset legitimately has neurons without type assignments
        # We should NOT treat them as incomplete and refuse to cache them
        
        # All neurons can be marked as complete - no validation needed for type/instance
        self._update_neuron_index_after_fetch(connections, upstream_bodyIds, downstream_bodyIds)
    
    def _update_neuron_index_after_fetch(self, connections, upstream_bodyIds, downstream_bodyIds=None):
        '''
        Update neuron index after fetching connections.
        Only marks neurons as downstream_complete if we fetched ALL downstream (downstream_bodyIds=None).
        '''
        neuron_index = self._load_neuron_index()
        
        # Get neuron info from complete dataset
        dataset_safe = self.dataset.replace(':', '_').replace('.', '_')
        dataset_path = os.path.join(
            self.script_path,
            'datasets',
            dataset_safe,
            f"{dataset_safe}_allneurons_neuron_df.csv"
        )
        
        self._vprint(f'  ⏳ Loading neuron metadata for {len(upstream_bodyIds):,} neurons...', level='full')
        if os.path.exists(dataset_path):
            # Check if it's FAFB to decide on index_col (FAFB utils saves without index)
            is_fafb = 'fafb' in self.dataset.lower() or 'flywire' in self.dataset.lower()
            
            if is_fafb:
                ndf_complete = pd.read_csv(dataset_path, header=0, index_col=None, dtype={'bodyId': str}, low_memory=False)
            else:
                ndf_complete = pd.read_csv(dataset_path, header=0, index_col=0, low_memory=False)

            # Ensure bodyId is string in local dataset
            if 'bodyId' in ndf_complete.columns:
                ndf_complete['bodyId'] = ndf_complete['bodyId'].astype(str)
            
            neuron_info = ndf_complete[ndf_complete['bodyId'].isin(upstream_bodyIds)][['bodyId', 'type', 'instance', 'post']].copy()
        else:
            # Fallback: fetch from API
            try:
                ndf, _ = fetch_neurons(NeuronCriteria(bodyId=upstream_bodyIds))
                neuron_info = ndf[['bodyId', 'type', 'instance', 'post']].copy()
            except:
                neuron_info = pd.DataFrame(columns=['bodyId', 'type', 'instance', 'post'])
        
        # Count connections per neuron
        self._vprint(f'  ⏳ Counting connections per neuron...', level='full')
        if not connections.empty:
            conn_counts = connections.groupby('bodyId_pre').size().reset_index(name='connection_count')
        else:
            conn_counts = pd.DataFrame(columns=['bodyId_pre', 'connection_count'])
        
        # Only mark as downstream_complete if we fetched ALL downstream
        mark_complete = (downstream_bodyIds is None)
        
        for bodyId in tqdm(upstream_bodyIds, desc='  ⏳ Updating neuron index', leave=False):
            # Ensure bodyId is string for comparison
            bodyId = str(bodyId)
            
            neuron_row = neuron_info[neuron_info['bodyId'] == bodyId]
            if not neuron_row.empty:
                neuron_type = neuron_row.iloc[0]['type'] if 'type' in neuron_row.columns else ''
                neuron_instance = neuron_row.iloc[0]['instance'] if 'instance' in neuron_row.columns else ''
                neuron_post = neuron_row.iloc[0]['post'] if 'post' in neuron_row.columns else 0
            else:
                neuron_type = ''
                neuron_instance = ''
                neuron_post = 0
            
            conn_count = conn_counts[conn_counts['bodyId_pre'] == bodyId]['connection_count'].iloc[0] if bodyId in conn_counts['bodyId_pre'].values else 0
            
            # Check if bodyId exists in index (ensure string comparison)
            if bodyId in neuron_index['bodyId'].values:
                # Update existing entry
                if mark_complete:
                    neuron_index.loc[neuron_index['bodyId'] == bodyId, 'downstream_complete'] = True
                neuron_index.loc[neuron_index['bodyId'] == bodyId, 'last_fetched'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                neuron_index.loc[neuron_index['bodyId'] == bodyId, 'connection_count'] = conn_count
                neuron_index.loc[neuron_index['bodyId'] == bodyId, 'type'] = neuron_type
                neuron_index.loc[neuron_index['bodyId'] == bodyId, 'instance'] = neuron_instance
                neuron_index.loc[neuron_index['bodyId'] == bodyId, 'instance'] = neuron_instance
                neuron_index.loc[neuron_index['bodyId'] == bodyId, 'post'] = neuron_post
            else:
                # Add new entry
                new_entry = pd.DataFrame([{
                    'bodyId': bodyId,
                    'type': neuron_type,
                    'instance': neuron_instance,
                    'post': neuron_post,
                    'downstream_complete': mark_complete,
                    'last_fetched': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'connection_count': conn_count
                }])
                neuron_index = pd.concat([neuron_index, new_entry], ignore_index=True)
        
        self._vprint(f'  ⏳ Saving neuron index ({len(neuron_index):,} total neurons)...', level='full')
        self._save_neuron_index(neuron_index)
        
        if mark_complete:
            completed_count = len([b for b in upstream_bodyIds if b in neuron_index[neuron_index['downstream_complete'] == True]['bodyId'].values])
            self._vprint(f'  📝 Updated neuron index: {completed_count} neurons marked as complete', level='full')
    
    # ============================================================================
    # Enrichment with Type/Instance
    # ============================================================================
    
    def _enrich_connections_with_neuron_info(self, conn_df):
        '''
        Enrich connection dataframe with type and instance from complete local dataset.
        Also adds custom_group columns if source/target dataframes have them.
        '''
        if conn_df.empty:
            return conn_df
        
        self._vprint(f'  ⏳ Enriching {len(conn_df):,} connections with neuron info...', level='full')
        # Get unique bodyIds that need enrichment
        all_bodyids = list(set(conn_df['bodyId_pre'].tolist() + conn_df['bodyId_post'].tolist()))
        
        # Load from complete dataset (includes type=None neurons)
        dataset_safe = self.dataset.replace(':', '_').replace('.', '_')
        dataset_path = os.path.join(
            self.script_path,
            'datasets',
            dataset_safe,
            f"{dataset_safe}_allneurons_neuron_df.csv"
        )
        
        # Check for dataset in subfolder (common for FlyWire/FAFB)
        if not os.path.exists(dataset_path):
            # Fallback for legacy or different naming
            subfolder_path = os.path.join(
                self.script_path,
                'datasets',
                self.dataset,
                f"{self.dataset}_allneurons_neuron_df.csv"
            )
            if os.path.exists(subfolder_path):
                dataset_path = subfolder_path
        
        if not os.path.exists(dataset_path):
            # Fallback: fetch from API
            self._vprint(f'  ⚠️ Warning: Complete dataset not found, fetching from API...', level='full')
            neuron_df = self._fetch_neurons_local_or_api(all_bodyids, columns=['bodyId', 'type', 'instance'])
        else:
            # Load complete dataset from CSV
            # Check if it's FAFB to decide on index_col (FAFB utils saves without index)
            is_fafb = 'fafb' in self.dataset.lower() or 'flywire' in self.dataset.lower()
            
            if is_fafb:
                ndf_complete = pd.read_csv(dataset_path, header=0, index_col=None, dtype={'bodyId': str}, low_memory=False)
            else:
                ndf_complete = pd.read_csv(dataset_path, header=0, index_col=0, low_memory=False)
            
            # Ensure bodyId is string for FAFB
            if is_fafb and 'bodyId' in ndf_complete.columns:
                ndf_complete['bodyId'] = ndf_complete['bodyId'].astype(str)

            # Filter to only neurons we need
            neuron_df = ndf_complete[ndf_complete['bodyId'].isin(all_bodyids)].copy()
            
            # Check for missing neurons and fetch from API if needed
            found_bodyids = set(neuron_df['bodyId'].unique())
            missing_bodyids = set(all_bodyids) - found_bodyids
            
            if missing_bodyids:
                self._vprint(f'  ℹ️  {len(missing_bodyids)} neurons not in local dataset, fetching from API...', level='full')
                missing_neuron_df = self._fetch_neurons_local_or_api(
                    list(missing_bodyids), 
                    columns=['bodyId', 'type', 'instance']
                )
                if not missing_neuron_df.empty:
                    neuron_df = pd.concat([neuron_df, missing_neuron_df], ignore_index=True)
        
        neuron_info = neuron_df[['bodyId', 'type', 'instance']].copy()
        # Ensure bodyId is string for merging
        neuron_info['bodyId'] = neuron_info['bodyId'].astype(str)
        
        # Add custom_group from source_df and target_df if available
        if hasattr(self, 'source_df') and 'custom_group' in self.source_df.columns:
            source_custom = self.source_df[['bodyId', 'custom_group']].rename(
                columns={'custom_group': 'custom_group_pre'}
            )
            source_custom['bodyId'] = source_custom['bodyId'].astype(str)
            neuron_info = neuron_info.merge(source_custom, on='bodyId', how='left')
        
        if hasattr(self, 'target_df') and 'custom_group' in self.target_df.columns:
            target_custom = self.target_df[['bodyId', 'custom_group']].rename(
                columns={'custom_group': 'custom_group_post'}
            )
            target_custom['bodyId'] = target_custom['bodyId'].astype(str)
            neuron_info = neuron_info.merge(target_custom, on='bodyId', how='left')
        
        # Drop existing type/instance/custom_group columns if they exist (to avoid _x, _y suffixes after merge)
        columns_to_drop = []
        for col in ['type_pre', 'instance_pre', 'type_post', 'instance_post', 
                    'custom_group_pre', 'custom_group_post']:
            if col in conn_df.columns:
                columns_to_drop.append(col)
        if columns_to_drop:
            conn_df = conn_df.drop(columns=columns_to_drop)
        
        # Prepare columns to merge
        merge_cols = {'type': 'type_pre', 'instance': 'instance_pre'}
        if 'custom_group_pre' in neuron_info.columns:
            merge_cols['custom_group_pre'] = 'custom_group_pre'
        if 'custom_group_post' in neuron_info.columns:
            merge_cols['custom_group_post'] = 'custom_group_post'
        
        # Join type and instance for pre-synaptic neurons
        merge_info_pre = neuron_info.rename(columns={'type': 'type_pre', 'instance': 'instance_pre'})
        if 'custom_group_pre' in merge_info_pre.columns:
            merge_info_pre = merge_info_pre[['bodyId', 'type_pre', 'instance_pre', 'custom_group_pre']]
        else:
            merge_info_pre = merge_info_pre[['bodyId', 'type_pre', 'instance_pre']]
        
        # Ensure bodyId columns are strings for merging to avoid warnings
        conn_df['bodyId_pre'] = conn_df['bodyId_pre'].astype(str)
        merge_info_pre['bodyId'] = merge_info_pre['bodyId'].astype(str)

        conn_df = conn_df.merge(
            merge_info_pre,
            left_on='bodyId_pre',
            right_on='bodyId',
            how='left'
        ).drop(columns=['bodyId'])
        
        # Join type and instance for post-synaptic neurons  
        merge_info_post = neuron_info.rename(columns={'type': 'type_post', 'instance': 'instance_post'})
        if 'custom_group_post' in merge_info_post.columns:
            merge_info_post = merge_info_post[['bodyId', 'type_post', 'instance_post', 'custom_group_post']]
        else:
            merge_info_post = merge_info_post[['bodyId', 'type_post', 'instance_post']]
        
        # Ensure bodyId columns are strings for merging to avoid warnings
        conn_df['bodyId_post'] = conn_df['bodyId_post'].astype(str)
        merge_info_post['bodyId'] = merge_info_post['bodyId'].astype(str)

        conn_df = conn_df.merge(
            merge_info_post,
            left_on='bodyId_post',
            right_on='bodyId',
            how='left'
        ).drop(columns=['bodyId'])
        
        self._vprint(f'  ✓ Enrichment complete', level='full')
        return conn_df
    
    def _fetch_neurons_local_or_api(self, bodyIds, columns=None):
        '''
        Fetch neuron information from cache, local dataset, or API (in that order).
        
        Parameters:
        -----------
        bodyIds : list
            List of neuron bodyIds to fetch
        columns : list or None
            Specific columns to return (None = all columns)
        
        Returns:
        --------
        pd.DataFrame : Neuron information dataframe
        '''
        if not bodyIds:
            return pd.DataFrame()
        
        # 1. Try to load from neuron index cache first (fastest)
        neuron_index = self._load_neuron_index()
        if not neuron_index.empty:
            # Filter to requested bodyIds
            cached_neurons = neuron_index[neuron_index['bodyId'].isin(bodyIds)].copy()
            
            if len(cached_neurons) > 0:
                # Check if all requested columns are available in cache
                if columns:
                    available_cols = set(columns) & set(cached_neurons.columns)
                    missing_cols = set(columns) - available_cols
                    
                    if not missing_cols and len(cached_neurons) == len(bodyIds):
                        # Perfect cache hit - all neurons and columns found!
                        return cached_neurons[columns].copy()
                    elif len(cached_neurons) == len(bodyIds) and available_cols:
                        # All neurons found but missing some columns - need to fetch from dataset/API
                        pass  # Fall through to dataset/API fetch
                    elif available_cols:
                        # Partial hit - some neurons cached, some not
                        cached_bodyIds = set(cached_neurons['bodyId'])
                        uncached_bodyIds = [bid for bid in bodyIds if bid not in cached_bodyIds]
                        
                        if uncached_bodyIds:
                            # Fetch missing neurons from dataset/API
                            uncached_df = self._fetch_from_dataset_or_api(uncached_bodyIds, columns)
                            # Combine cached and uncached data
                            cached_subset = cached_neurons[[c for c in columns if c in available_cols]].copy()
                            result = pd.concat([cached_subset, uncached_df], ignore_index=True)
                            return result
                else:
                    # No specific columns requested - return all available from cache
                    if len(cached_neurons) == len(bodyIds):
                        return cached_neurons.copy()
        
        # 2. Cache miss - fetch from dataset or API
        return self._fetch_from_dataset_or_api(bodyIds, columns)
    
    def _fetch_from_dataset_or_api(self, bodyIds, columns=None):
        '''
        Helper function to fetch neurons from local dataset or API.
        
        Parameters:
        -----------
        bodyIds : list
            List of neuron bodyIds to fetch
        columns : list or None
            Specific columns to return
        
        Returns:
        --------
        pd.DataFrame : Neuron information dataframe
        '''
        # Try local dataset first
        dataset_safe = self.dataset.replace(':', '_').replace('.', '_')
        dataset_path = os.path.join(
            self.script_path,
            'datasets',
            dataset_safe,
            f"{dataset_safe}_allneurons_neuron_df.csv"
        )
        
        # Check for FAFB specific path if generic path doesn't exist
        if not os.path.exists(dataset_path) and ('fafb' in self.dataset.lower() or 'flywire' in self.dataset.lower()):
            fafb_path = os.path.join(self.script_path, 'datasets', 'flywire_FAFB_v783', 'flywire_v783_allneurons_neuron_df.csv')
            if os.path.exists(fafb_path):
                dataset_path = fafb_path
        
        if os.path.exists(dataset_path):
            # Fast: Load from local CSV
            # Check if it's FAFB to decide on index_col (FAFB utils saves without index)
            is_fafb = 'fafb' in self.dataset.lower() or 'flywire' in self.dataset.lower()
            
            if is_fafb:
                ndf_complete = pd.read_csv(dataset_path, header=0, index_col=None, dtype={'bodyId': str}, low_memory=False)
            else:
                ndf_complete = pd.read_csv(dataset_path, header=0, index_col=0, low_memory=False)
            
            # Ensure bodyId is string for FAFB
            if is_fafb:
                ndf_complete['bodyId'] = ndf_complete['bodyId'].astype(str)
                bodyIds = [str(b) for b in bodyIds]
            
            neuron_df = ndf_complete[ndf_complete['bodyId'].isin(bodyIds)].copy()
            if columns:
                # Ensure columns exist
                for col in columns:
                    if col not in neuron_df.columns:
                        if col == 'post':
                            neuron_df[col] = 1000 # Default post count
                        else:
                            neuron_df[col] = ''
                neuron_df = neuron_df[columns].copy()
            return neuron_df
        else:
            # Check if we should enforce local-only for FAFB/FlyWire
            if 'fafb' in self.dataset.lower() or 'flywire' in self.dataset.lower():
                 self._vprint(f"\n  ⚠️  Local neuron data not found for dataset '{self.dataset}'.", level='full')
                 self._vprint("  Please download the neuron table from: https://codex.flywire.ai/api/download?dataset=fafb", level='full')
                 self._vprint(f"  Save the file to: {dataset_path}", level='full') 
                 self._vprint("  Skipping API fetch to avoid timeouts/limits.", level='full')
                 return pd.DataFrame(columns=columns if columns else [])

            # Slow: API call
            if self.client_type == 'flywire':
                if self.client_flywire:
                    # Use FlyWire adapter
                    criteria = SimpleNamespace(bodyId=bodyIds)
                    neuron_df, _ = self.client_flywire.fetch_neurons(criteria)
                    if columns:
                        # Ensure columns exist
                        for col in columns:
                            if col not in neuron_df.columns:
                                if col == 'post':
                                    neuron_df[col] = 1000 # Default post count
                                else:
                                    neuron_df[col] = ''
                        neuron_df = neuron_df[columns].copy()
                    return neuron_df
                else:
                    return pd.DataFrame(columns=columns if columns else [])

            # Ensure client is logged in (NeuPrint)
            if self.client_hemibrain is None:
                from neuprint import Client, set_default_client
                self.client_hemibrain = Client(self.server, self.dataset, self.token)
                set_default_client(self.client_hemibrain)
            
            neuron_df, _ = fetch_neurons(NeuronCriteria(bodyId=bodyIds))
            if columns:
                neuron_df = neuron_df[columns].copy()
            return neuron_df
    
    def _fetch_neurons_by_types(self, types, columns=None):
        '''
        Fetch ALL neurons of given types from local dataset if available, otherwise use API.
        
        Parameters:
        -----------
        types : list
            List of neuron types to fetch
        columns : list or None
            Specific columns to return (None = all columns)
        
        Returns:
        --------
        pd.DataFrame : Neuron information dataframe
        '''
        # Try local dataset first
        dataset_safe = self.dataset.replace(':', '_').replace('.', '_')
        dataset_path = os.path.join(
            self.script_path,
            'datasets',
            dataset_safe,
            f"{dataset_safe}_allneurons_neuron_df.csv"
        )
        
        # Check for FAFB specific path if generic path doesn't exist
        if not os.path.exists(dataset_path) and ('fafb' in self.dataset.lower() or 'flywire' in self.dataset.lower()):
            fafb_path = os.path.join(self.script_path, 'datasets', 'flywire_FAFB_v783', 'flywire_v783_allneurons_neuron_df.csv')
            if os.path.exists(fafb_path):
                dataset_path = fafb_path
        
        if os.path.exists(dataset_path):
            # Fast: Load from local CSV
            # Check if it's FAFB to decide on index_col (FAFB utils saves without index)
            is_fafb = 'fafb' in self.dataset.lower() or 'flywire' in self.dataset.lower()
            
            if is_fafb:
                ndf_complete = pd.read_csv(dataset_path, header=0, index_col=None, dtype={'bodyId': str}, low_memory=False)
            else:
                ndf_complete = pd.read_csv(dataset_path, header=0, index_col=0, low_memory=False)
            
            # Ensure bodyId is string for FAFB
            if is_fafb:
                ndf_complete['bodyId'] = ndf_complete['bodyId'].astype(str)
            
            neuron_df = ndf_complete[ndf_complete['type'].isin(types)].copy()
            if columns:
                # Ensure columns exist
                for col in columns:
                    if col not in neuron_df.columns:
                        if col == 'post':
                            neuron_df[col] = 1000 # Default post count
                        else:
                            neuron_df[col] = ''
                neuron_df = neuron_df[columns].copy()
            return neuron_df
        else:
            # Check if we should enforce local-only for FAFB/FlyWire
            if 'fafb' in self.dataset.lower() or 'flywire' in self.dataset.lower():
                 self._vprint(f"\n  ⚠️  Local neuron data not found for dataset '{self.dataset}'.", level='full')
                 self._vprint("  Please download the neuron table from: https://codex.flywire.ai/api/download?dataset=fafb", level='full')
                 self._vprint(f"  Save the file to: {dataset_path}", level='full') 
                 self._vprint("  Skipping API fetch to avoid timeouts/limits.", level='full')
                 return pd.DataFrame(columns=columns if columns else [])

            # Slow: API call (ensure client is logged in)
            if self.client_type == 'flywire':
                if self.client_flywire:
                    # Fetch neurons by type using FlyWire adapter
                    all_neurons = []
                    for neuron_type in types:
                        # Assuming adapter has fetch_neurons that accepts criteria object or dict
                        # We construct a simple object or dict
                        criteria = SimpleNamespace(type=neuron_type)
                        neuron_df, _ = self.client_flywire.fetch_neurons(criteria)
                        all_neurons.append(neuron_df)
                else:
                    return pd.DataFrame(columns=columns if columns else [])
            else:
                if self.client_hemibrain is None:
                    from neuprint import Client, set_default_client
                    self.client_hemibrain = Client(self.server, self.dataset, self.token)
                    set_default_client(self.client_hemibrain)
                
                # Fetch neurons by type
                all_neurons = []
                for neuron_type in types:
                    neuron_df, _ = fetch_neurons(NeuronCriteria(type=neuron_type))
                    all_neurons.append(neuron_df)
            
            if all_neurons:
                neuron_df = pd.concat(all_neurons, ignore_index=True)
                if columns:
                    # Ensure columns exist
                    for col in columns:
                        if col not in neuron_df.columns:
                            if col == 'post':
                                neuron_df[col] = 1000 # Default post count
                            else:
                                neuron_df[col] = ''
                    neuron_df = neuron_df[columns].copy()
                return neuron_df
            else:
                return pd.DataFrame(columns=columns if columns else [])
    
    # ============================================================================
    # Main Fetch Method (replaces old _fetch_connections_with_cache)
    # ============================================================================
    
    def _fetch_connections_with_cache(self, upstream_bodyIds, downstream_bodyIds=None, 
                                      min_weight=None, min_traversal_prob=None, min_conn_ratio=None):
        '''
        Fetch connections with v4.0 pair-level caching.
        Queries unified database first, only fetches missing neurons from API.
        
        Parameters:
        -----------
        upstream_bodyIds : list
            List of upstream neuron bodyIds
        downstream_bodyIds : list or None
            List of downstream neuron bodyIds (None = all downstream)
        min_weight : int or None
            Minimum synapse count for filtering (uses self.min_synapse_num if None)
        min_traversal_prob : float or None
            Minimum traversal probability for edge filtering (uses self.min_traversal_probability if None)
        min_conn_ratio : float or None
            Minimum connection ratio (weight/post) for edge filtering (uses self.min_ratio if None)
        
        Returns:
        --------
        pd.DataFrame : Connection table filtered by min_weight, min_traversal_prob, and min_conn_ratio
        '''
        if min_weight is None:
            min_weight = self.min_synapse_num
        if min_traversal_prob is None:
            min_traversal_prob = self.min_traversal_probability
        if min_conn_ratio is None:
            min_conn_ratio = self.min_ratio
        
        # Step 1: Query database for cached connections
        cached_conn, uncached_upstream, partially_cached = self._query_connection_db(upstream_bodyIds, downstream_bodyIds)
        
        if not cached_conn.empty:
            self._vprint(f'  📂 Found {len(set(upstream_bodyIds) - set(uncached_upstream))}/{len(upstream_bodyIds)} neurons in cache', level='full')
            self._vprint(f'     Retrieved {len(cached_conn):,} connections from database', level='full')
        
        # Step 2: Fetch uncached neurons from API if needed
        api_conn = pd.DataFrame()
        if len(uncached_upstream) > 0:
            self._vprint(f'  🌐 Fetching {len(uncached_upstream)} uncached neurons from API (weight ≥ 1)...', level='full')
            
            fetched_locally = False
            # Special handling for FAFB/FlyWire local data
            if 'fafb' in self.dataset.lower() or 'flywire' in self.dataset.lower():
                try:
                    import fafb_utils
                    project_root = os.path.dirname(os.path.dirname(__file__))
                    
                    # Try to find dataset directory by name
                    data_dir = os.path.join(project_root, "datasets", self.dataset)
                    if not os.path.exists(data_dir):
                        # Fallback to default FAFB directory
                        data_dir = os.path.join(project_root, "datasets", "flywire_FAFB_v783")
                    
                    if os.path.exists(data_dir):
                        # Only try local if the directory exists
                        _, conn_file = fafb_utils.prepare_fafb_data(data_dir)
                        
                        # Load connections
                        # Optimization: Load once if possible, but here we load on demand
                        # Use string for IDs
                        full_conn = pd.read_csv(conn_file, dtype={'pre_root_id': str, 'post_root_id': str})
                        full_conn = full_conn.rename(columns={
                            'pre_root_id': 'bodyId_pre',
                            'post_root_id': 'bodyId_post',
                            'syn_count': 'weight'
                        })
                        
                        # Filter by upstream
                        upstream_strs = [str(x) for x in uncached_upstream]
                        api_conn = full_conn[full_conn['bodyId_pre'].isin(upstream_strs)].copy()
                        
                        # Filter by downstream if provided
                        if downstream_bodyIds is not None:
                            downstream_strs = [str(x) for x in downstream_bodyIds]
                            api_conn = api_conn[api_conn['bodyId_post'].isin(downstream_strs)].copy()
                            
                        # Add dummy ROI column if missing
                        if 'roi' not in api_conn.columns:
                            api_conn['roi'] = 'WholeBrain'
                            
                        fetched_locally = True
                        self._vprint(f"  ✓ Loaded {len(api_conn)} connections from local FAFB data", level='full')
                except ImportError:
                    pass
                except Exception as e:
                    self._vprint(f"  ⚠️ Error loading local FAFB data: {e}", level='full')

            if not fetched_locally:
                # Check if we should enforce local-only for FAFB/FlyWire
                if 'fafb' in self.dataset.lower() or 'flywire' in self.dataset.lower():
                     self._vprint(f"\n  ⚠️  Local connection data not found for dataset '{self.dataset}'.", level='full')
                     self._vprint("  Please download the synapse table from: https://codex.flywire.ai/api/download?dataset=fafb", level='full')
                     self._vprint(f"  Save the file to: datasets/{self.dataset.replace(':', '_')}", level='full') 
                     self._vprint("  Skipping API fetch to avoid timeouts/limits.", level='full')
                     return pd.DataFrame()

                if self.client_type == 'flywire':
                    if self.client_flywire:
                        # Use FlyWire adapter
                        # Note: FlyWire adapter mimics fetch_adjacencies behavior
                        neuron_df, api_conn = self.client_flywire.fetch_adjacencies(
                            sources=uncached_upstream,
                            targets=downstream_bodyIds
                        )
                        # api_conn should have bodyId_pre, bodyId_post, weight, roi
                    else:
                        self._vprint("Error: FlyWire client not initialized", level='full')
                else:
                    # Login to neuprint only if needed
                    from neuprint import Client, set_default_client, default_client, NeuronCriteria
                    try:
                        from tqdm import tqdm
                    except ImportError:
                        # Fallback if tqdm not installed
                        def tqdm(iterable, **kwargs): return iterable
                    
                    # Ensure bodyIds are integers for NeuPrint
                    # NeuPrint client requires bodyIds to be integers, not strings or floats
                    # This fixes AssertionError: bodyId should be an integer or list of integers
                    if uncached_upstream:
                        try:
                            uncached_upstream = [int(x) for x in uncached_upstream]
                        except (ValueError, TypeError):
                            # If conversion fails (e.g. non-numeric IDs), keep as is and let NeuPrint handle/fail
                            pass
                            
                    if downstream_bodyIds:
                        try:
                            downstream_bodyIds = [int(x) for x in downstream_bodyIds]
                        except (ValueError, TypeError):
                            pass

                    # Only login if not already done (default_client() returns None if not set)
                    if self.client_hemibrain is None and default_client() is None:
                        self.client_hemibrain = Client(self.server, self.dataset, self.token)
                        set_default_client(self.client_hemibrain)
                    
                    # Batch processing
                    batch_size = 100
                    all_api_conn = []
                    
                    # Create batches
                    batches = [uncached_upstream[i:i + batch_size] for i in range(0, len(uncached_upstream), batch_size)]
                    
                    if len(batches) > 1:
                        self._vprint(f'     Processing {len(batches)} batches (size={batch_size})...', level='full')
                    
                    # Use tqdm only if multiple batches or large single batch
                    iterator = tqdm(batches, desc="Fetching batches", unit="batch") if len(batches) > 1 else batches
                    
                    for batch in iterator:
                        try:
                            if self.simple_fetch:
                                from neuprint import fetch_simple_connections
                                upstream_criteria = NeuronCriteria(bodyId=batch)
                                downstream_criteria = NeuronCriteria(bodyId=downstream_bodyIds) if downstream_bodyIds is not None else None
                                batch_conn = fetch_simple_connections(
                                    upstream_criteria=upstream_criteria,
                                    downstream_criteria=downstream_criteria,
                                    min_weight=1,  # Always fetch with min_weight=1
                                    **self.kwargs_fetch
                                )
                                if not batch_conn.empty:
                                    all_api_conn.append(batch_conn)
                            else:
                                from neuprint import fetch_adjacencies
                                import statvis as sv
                                neuron_df, roi_conn_df = fetch_adjacencies(
                                    sources=batch,
                                    targets=downstream_bodyIds,
                                    min_total_weight=1,  # Always fetch with min_weight=1
                                    **self.kwargs_fetch
                                )
                                batch_conn = sv.merge_conn_roi(neuron_df, roi_conn_df)
                                if not batch_conn.empty:
                                    all_api_conn.append(batch_conn)
                        except Exception as e:
                            self._vprint(f"     ⚠️ Error fetching batch: {e}", level='full')
                            
                    if all_api_conn:
                        api_conn = pd.concat(all_api_conn, ignore_index=True)
                    else:
                        api_conn = pd.DataFrame()
            
            # Save connections to database (but don't mark neurons as cached yet)
            self._save_connections_only(api_conn, uncached_upstream)
        
        # Step 3: Combine cached and API results
        if cached_conn.empty and api_conn.empty:
            # Return empty DataFrame with correct columns to avoid KeyErrors downstream
            return pd.DataFrame(columns=['bodyId_pre', 'bodyId_post', 'weight', 'roi', 'type_pre', 'type_post', 'instance_pre', 'instance_post'])
        
        # Combine results
        combined = pd.concat([cached_conn, api_conn], ignore_index=True) if not cached_conn.empty and not api_conn.empty else (cached_conn if not cached_conn.empty else api_conn)
        
        total_before_filter = len(combined)
        
        # Step 4: Apply filters based on filter_by level
        # Enrich with type and instance info (needed for both filtering modes)
        combined = self._enrich_connections_with_neuron_info(combined)
        
        # NOW mark neurons as cached (after successful enrichment)
        # Mark newly fetched neurons (partially_cached will be empty)
        neurons_to_mark = list(set(uncached_upstream + partially_cached))
        if len(neurons_to_mark) > 0:
            self._vprint(f'  ⏳ Preparing to mark {len(neurons_to_mark):,} neurons as cached...', level='full')
            # Get the connections for these neurons from the combined dataframe
            neurons_conn = combined[combined['bodyId_pre'].isin(neurons_to_mark)]
            
            # Debug: Check if some neurons have no connections
            neurons_with_conns = set(neurons_conn['bodyId_pre'].unique())
            neurons_without_conns = set(neurons_to_mark) - neurons_with_conns
            if neurons_without_conns:
                self._vprint(f'  ℹ️  Note: {len(neurons_without_conns)} neurons have 0 connections (will still be marked as complete)', level='full')
            
            self._mark_neurons_as_cached(neurons_to_mark, neurons_conn, downstream_bodyIds)
            self._vprint(f'  ✓ Cache update complete - {len(neurons_to_mark)} neurons marked as fetched', level='full')
        
        # Exclude intra-type connections if requested (before applying other filters)
        if self.exclude_intra_type_connections and len(combined) > 0:
            before_count = len(combined)
            # Remove connections where type_pre == type_post
            combined = combined[combined['type_pre'] != combined['type_post']].copy()
            after_count = len(combined)
            if before_count > after_count:
                self._vprint(f'  ⚠️  Excluded {before_count - after_count:,} intra-type connections (type_pre == type_post)', level='full')
        
        # Apply filters at the specified level
        self._vprint(f'  ⏳ Applying filters to {len(combined):,} connections...', level='full')
        if self.filter_by == 'type':
            # Type-level filtering: aggregate first, then filter
            # Weight filter applied at type level (sum of all weights per type pair)
            combined = self._apply_type_level_filters(combined, min_weight, min_conn_ratio, min_traversal_prob, total_before_filter)
        else:
            # BodyId-level filtering: filter individual connections by weight first
            if min_weight > 1:
                combined = combined[combined['weight'] >= min_weight].copy()
            
            # Then apply ratio/prob filters if specified
            if (min_traversal_prob > 0 or min_conn_ratio > 0) and len(combined) > 0:
                combined = self._apply_bodyid_level_filters(combined, min_conn_ratio, min_traversal_prob, total_before_filter, min_weight)
            else:
                # No ratio filters, just print weight filter summary
                if min_weight > 1:
                    self._vprint(f'     Filtered: {total_before_filter} → {len(combined)} connections (weight ≥ {min_weight})', level='full')
                self._vprint(f'     Enriched with neuron info from complete local dataset', level='full')
        
        return combined
    
    def _apply_bodyid_level_filters(self, combined, min_conn_ratio, min_traversal_prob, total_before_filter, min_weight):
        """Apply filters at individual bodyId level (default behavior)"""
        # Get post-synaptic counts (use local dataset if available)
        post_bodyIds = combined['bodyId_post'].unique().tolist()
        post_df = self._fetch_neurons_local_or_api(post_bodyIds, columns=['bodyId', 'post'])
        post_info = post_df[['bodyId', 'post']].copy()
        post_info.columns = ['bodyId_post', 'post']
        
        # Merge and calculate both ratios
        combined = combined.merge(post_info, how='left', on='bodyId_post')
        combined['connection_ratio'] = combined['weight'] / combined['post']
        combined['traversal_probability'] = combined['connection_ratio'] / 0.3
        combined.loc[combined['traversal_probability'] > 1, 'traversal_probability'] = 1
        
        # Filter by connection ratio
        if min_conn_ratio > 0:
            combined = combined[combined['connection_ratio'] >= min_conn_ratio].copy()
        
        # Filter by traversal probability
        if min_traversal_prob > 0:
            combined = combined[combined['traversal_probability'] >= min_traversal_prob].copy()
        
        # Drop temporary columns - KEEP ratio/prob for downstream use
        combined = combined.drop(columns=['post'])
        
        # Print filter summary
        filter_msg = []
        if min_weight > 1:
            filter_msg.append(f'weight ≥ {min_weight}')
        if min_conn_ratio > 0:
            filter_msg.append(f'ratio ≥ {min_conn_ratio}')
        if min_traversal_prob > 0:
            filter_msg.append(f'prob ≥ {min_traversal_prob}')
        
        self._vprint(f'     Filtered (bodyId level): {total_before_filter} → {len(combined)} connections ({", ".join(filter_msg)})', level='full')
        self._vprint(f'     Enriched with neuron info from complete local dataset', level='full')
        
        return combined
    
    def _apply_type_level_filters(self, combined, min_weight, min_conn_ratio, min_traversal_prob, total_before_filter):
        """
        Apply filters at aggregated type-to-type level.
        For type-level filtering: aggregate first, then apply weight/ratio/prob filters to type pairs.
        """
        # Separate connections with null types (preserve them always)
        null_type_mask = combined['type_pre'].isna() | combined['type_post'].isna()
        connections_with_null_types = combined[null_type_mask].copy()
        connections_with_types = combined[~null_type_mask].copy()
        
        # Get post-synaptic counts (use local dataset if available) if not already present
        if 'post' not in connections_with_types.columns and len(connections_with_types) > 0:
            post_bodyIds = connections_with_types['bodyId_post'].unique().tolist()
            post_df = self._fetch_neurons_local_or_api(post_bodyIds, columns=['bodyId', 'post'])
            post_info = post_df[['bodyId', 'post']].copy()
            post_info.columns = ['bodyId_post', 'post']
            connections_with_types = connections_with_types.merge(post_info, how='left', on='bodyId_post')
        
        # Group by type pairs and aggregate (only for connections with valid types)
        if len(connections_with_types) > 0:
            type_grouped = connections_with_types.groupby(['type_pre', 'type_post'], as_index=False).agg({
                'weight': 'sum',  # Sum of all synapses for this type pair
            })
        else:
            type_grouped = pd.DataFrame(columns=['type_pre', 'type_post', 'weight'])
        
        # Apply weight filter at type level (total synapses per type pair)
        type_grouped_before_weight = len(type_grouped)
        if min_weight > 1:
            type_grouped = type_grouped[type_grouped['weight'] >= min_weight].copy()
        
        # Calculate total post-synaptic sites for each type (ALL neurons of that type, not just those in connections)
        # Get all unique types that appear in connections
        all_types = combined['type_post'].unique().tolist()
        # Fetch ALL neurons of these types from dataset
        all_neurons_df = self._fetch_neurons_by_types(all_types, columns=['bodyId', 'type', 'post'])
        # Calculate total post per type
        type_post_totals = all_neurons_df.groupby('type')['post'].sum().reset_index()
        type_post_totals.columns = ['type_post', 'total_post']
        
        # Merge with grouped data (each type pair gets the total_post for its target type)
        type_grouped = type_grouped.merge(type_post_totals, on='type_post', how='left')
        
        # Calculate ratios at type level
        type_grouped['connection_ratio'] = type_grouped.apply(
            lambda row: row['weight'] / row['total_post'] if pd.notnull(row['total_post']) and row['total_post'] > 0 else 0.0,
            axis=1
        )
        type_grouped['traversal_probability'] = type_grouped['connection_ratio'] / 0.3
        type_grouped.loc[type_grouped['traversal_probability'] > 1, 'traversal_probability'] = 1
        
        # Apply ratio/prob filters at type level
        if len(type_grouped) > 0:
            filtered_type_pairs = type_grouped.copy()
            if min_conn_ratio > 0:
                filtered_type_pairs = filtered_type_pairs[filtered_type_pairs['connection_ratio'] >= min_conn_ratio].copy()
            if min_traversal_prob > 0:
                filtered_type_pairs = filtered_type_pairs[filtered_type_pairs['traversal_probability'] >= min_traversal_prob].copy()
            
            # Keep ALL bodyId connections that belong to passing type pairs
            passing_type_pairs = set(zip(filtered_type_pairs['type_pre'], filtered_type_pairs['type_post']))
            filtered_connections = connections_with_types[connections_with_types.apply(lambda row: (row['type_pre'], row['type_post']) in passing_type_pairs, axis=1)].copy()
            
            # Drop temporary 'post' column if it exists
            if 'post' in filtered_connections.columns:
                filtered_connections = filtered_connections.drop(columns=['post'])
        else:
            filtered_type_pairs = type_grouped
            filtered_connections = connections_with_types
        
        # Recombine with connections that have null types (always keep these)
        if len(connections_with_null_types) > 0:
            combined = pd.concat([filtered_connections, connections_with_null_types], ignore_index=True)
        else:
            combined = filtered_connections
        
        # Print filter summary
        filter_msg = []
        if min_weight > 1:
            filter_msg.append(f'type-weight ≥ {min_weight}')
        if min_conn_ratio > 0:
            filter_msg.append(f'type-ratio ≥ {min_conn_ratio}')
        if min_traversal_prob > 0:
            filter_msg.append(f'type-prob ≥ {min_traversal_prob}')
        
        type_pairs_after = len(filtered_type_pairs)
        null_conn_count = len(connections_with_null_types)
        self._vprint(f'     Filtered (type level): {type_grouped_before_weight} → {type_pairs_after} type pairs, {total_before_filter} → {len(combined)} connections ({", ".join(filter_msg)})', level='full')
        if null_conn_count > 0:
            self._vprint(f'     Note: {null_conn_count} connections with null types preserved (not filtered)', level='full')
        self._vprint(f'     Note: All 3 filters applied at type level (weight=sum, ratio=sum(weight)/sum(post))', level='full')
        self._vprint(f'     Enriched with neuron info from complete local dataset', level='full')
        
        return combined
    
    # ============================================================================
    # Cache Building Methods
    # ============================================================================
    
    def build_connection_cache(
        self,
        neuron_types: list = None,
        neuron_bodyIds: list = None,
        batch_size: int = 100,
        progress_callback: callable = None
    ) -> dict:
        """
        Pre-build connection cache for specified neurons or all neurons in dataset.
        
        This method efficiently pre-fetches and caches connections for neurons,
        enabling faster subsequent queries. Useful for building offline caches
        or preparing for batch analysis.
        
        Parameters:
        -----------
        neuron_types : list, optional
            List of neuron types to cache. If None and neuron_bodyIds is None,
            caches all neurons in the dataset.
        neuron_bodyIds : list, optional
            List of specific bodyIds to cache. Takes precedence over neuron_types.
        batch_size : int
            Number of neurons to fetch per batch (default: 100)
        progress_callback : callable, optional
            Callback function(current, total, neuron_info) for progress updates
        
        Returns:
        --------
        dict : Summary with keys:
            - 'total_neurons': Number of neurons processed
            - 'total_connections': Total connections cached
            - 'cached_neurons': List of successfully cached neuron bodyIds
            - 'failed_neurons': List of neurons that failed to cache
            - 'elapsed_time': Time taken in seconds
        
        Example:
        --------
        >>> fnc = FindNeuronConnection(dataset='hemibrain:v1.2.1', ...)
        >>> result = fnc.build_connection_cache(neuron_types=['aMe12', 'Mi1'])
        >>> print(f"Cached {result['total_connections']} connections")
        """
        import time
        start_time = time.time()
        
        print("=" * 60)
        print("Building Connection Cache")
        print("=" * 60)
        
        if not self.use_cache:
            print("⚠️  Cache is disabled. Enable with use_cache=True")
            return {'total_neurons': 0, 'total_connections': 0, 
                    'cached_neurons': [], 'failed_neurons': [], 'elapsed_time': 0}
        
        # Get bodyIds to cache
        if neuron_bodyIds is not None:
            bodyIds_to_cache = [str(x) for x in neuron_bodyIds]
            print(f"Caching connections for {len(bodyIds_to_cache)} specified bodyIds...")
        elif neuron_types is not None:
            # Fetch bodyIds for the given types
            print(f"Fetching bodyIds for {len(neuron_types)} neuron types...")
            bodyIds_to_cache = []
            for ntype in neuron_types:
                try:
                    neurons_df = self._fetch_neurons_local_or_api(
                        [ntype], 
                        columns=['bodyId', 'type'],
                        search_by='type'
                    )
                    if not neurons_df.empty:
                        bodyIds_to_cache.extend([str(x) for x in neurons_df['bodyId'].tolist()])
                except Exception as e:
                    print(f"  ⚠️ Failed to get bodyIds for type {ntype}: {e}")
            bodyIds_to_cache = list(set(bodyIds_to_cache))
            print(f"Found {len(bodyIds_to_cache)} unique bodyIds")
        else:
            # Cache all neurons in dataset
            print("Caching all neurons in dataset...")
            dataset_safe = self.dataset.replace(':', '_').replace('.', '_')
            dataset_path = os.path.join(
                self.script_path, 'datasets', dataset_safe,
                f"{dataset_safe}_allneurons_neuron_df.csv"
            )
            
            if os.path.exists(dataset_path):
                is_fafb = 'fafb' in self.dataset.lower() or 'flywire' in self.dataset.lower()
                if is_fafb:
                    ndf = pd.read_csv(dataset_path, dtype={'bodyId': str}, low_memory=False)
                else:
                    ndf = pd.read_csv(dataset_path, index_col=0, low_memory=False)
                    ndf['bodyId'] = ndf['bodyId'].astype(str)
                bodyIds_to_cache = ndf['bodyId'].unique().tolist()
                print(f"Found {len(bodyIds_to_cache)} neurons in dataset")
            else:
                print(f"⚠️ Dataset file not found: {dataset_path}")
                return {'total_neurons': 0, 'total_connections': 0,
                        'cached_neurons': [], 'failed_neurons': [], 'elapsed_time': 0}
        
        # Check which neurons are already cached
        neuron_index = self._load_neuron_index()
        if not neuron_index.empty:
            already_cached = neuron_index[
                neuron_index['downstream_complete'] == True
            ]['bodyId'].astype(str).tolist()
            uncached = [x for x in bodyIds_to_cache if x not in already_cached]
            print(f"Already cached: {len(already_cached)}, need to cache: {len(uncached)}")
        else:
            uncached = bodyIds_to_cache
            print(f"No existing cache, need to cache: {len(uncached)}")
        
        if not uncached:
            elapsed = time.time() - start_time
            print("✅ All neurons already cached!")
            return {'total_neurons': len(bodyIds_to_cache), 'total_connections': 0,
                    'cached_neurons': bodyIds_to_cache, 'failed_neurons': [], 
                    'elapsed_time': elapsed}
        
        # Process in batches
        total = len(uncached)
        cached_neurons = []
        failed_neurons = []
        total_connections = 0
        
        for i in range(0, total, batch_size):
            batch = uncached[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size
            
            # Progress callback
            if progress_callback:
                progress_callback(i, total, f"Batch {batch_num}/{total_batches}")
            
            print(f"\n📥 Batch {batch_num}/{total_batches}: Processing {len(batch)} neurons...")
            
            try:
                # Fetch connections for this batch (this will cache them)
                connections = self._fetch_connections_with_cache(
                    upstream_bodyIds=batch,
                    downstream_bodyIds=None,  # All downstream
                    min_weight=1,  # Cache all connections
                    min_traversal_prob=0,
                    min_conn_ratio=0
                )
                
                if not connections.empty:
                    total_connections += len(connections)
                    cached_neurons.extend(batch)
                    print(f"  ✅ Cached {len(connections)} connections")
                else:
                    # Even 0 connections is valid
                    cached_neurons.extend(batch)
                    print(f"  ✅ Cached (0 connections)")
                    
            except Exception as e:
                print(f"  ❌ Error caching batch: {e}")
                failed_neurons.extend(batch)
        
        elapsed = time.time() - start_time
        
        # Summary
        print("\n" + "=" * 60)
        print("Cache Build Complete")
        print("=" * 60)
        print(f"Total neurons processed: {len(cached_neurons) + len(failed_neurons)}")
        print(f"Successfully cached: {len(cached_neurons)}")
        print(f"Failed: {len(failed_neurons)}")
        print(f"Total connections cached: {total_connections:,}")
        print(f"Time elapsed: {elapsed:.1f} seconds")
        
        if failed_neurons:
            print(f"\nFailed neurons: {failed_neurons[:10]}{'...' if len(failed_neurons) > 10 else ''}")
        
        return {
            'total_neurons': len(cached_neurons) + len(failed_neurons),
            'total_connections': total_connections,
            'cached_neurons': cached_neurons,
            'failed_neurons': failed_neurons,
            'elapsed_time': elapsed
        }

    def InitializeNeuronInfo(self):
        # Ensure neuprint Client is set before any statvis/neuprint API call
        if self.client_type != 'flywire':
            from neuprint import Client, set_default_client
            try:
                from neuprint import default_client
                _ = default_client()
            except RuntimeError:
                self.client_hemibrain = Client(self.server, self.dataset, self.token)
                set_default_client(self.client_hemibrain)
        ''' initialize neuron info '''
        ''' initialize neuron info '''
        print('Fetching source and target neurons...')
        
        # Determine client to pass
        active_client = self.client_flywire if self.client_type == 'flywire' else self.client_hemibrain
        
        # Optimization: when max_interlayer=-1 and source==target, fetch only once
        self._source_target_identical = (self.max_interlayer == -1 and self.sourceNeurons == self.targetNeurons)
        
        if self._source_target_identical:
            print('\033[36mOptimization: source==target with max_interlayer=-1, fetching only one set\033[0m')
            self.source_df, _, source_fname_auto, self.source_criteria = sv.getNeurons(
                self.sourceNeurons, 
                dataset=self.dataset,
                custom_group_names=self.custom_source_group_names if self.custom_source_group_names else None,
                client=active_client
            )
            # Reuse source data for target
            self.target_df = self.source_df
            target_fname_auto = source_fname_auto
            self.target_criteria = self.source_criteria
        else:
            self.source_df, _, source_fname_auto, self.source_criteria = sv.getNeurons(
                self.sourceNeurons, 
                dataset=self.dataset,
                custom_group_names=self.custom_source_group_names if self.custom_source_group_names else None,
                client=active_client
            )
            self.target_df, _, target_fname_auto, self.target_criteria = sv.getNeurons(
                self.targetNeurons, 
                dataset=self.dataset,
                custom_group_names=self.custom_target_group_names if self.custom_target_group_names else None,
                client=active_client
            )
        
        if self.max_interlayer > 2 or len(self.source_df) > 200:
            self.simple_fetch = False
            print('\033[33mLarge data detected!!! simple_fetch is set to False, using fetch_adjacencies()\033[0m')

        if len(self.target_df) > 16383: # 16383 is the maximum number of excel sheet rows
            self.largeTargetSet = True
        
        if self.custom_source_name:
            self.source_fname = self.custom_source_name
        else:
            self.source_fname = source_fname_auto
        
        if self.custom_target_name:
            self.target_fname = self.custom_target_name
        else:
            self.target_fname = target_fname_auto
        
        print('Processing:',self.source_fname,'to',self.target_fname)
        print(f'Source neurons ({self.source_fname}) in processing: {len(self.source_df)}')
        print(f'Target neurons ({self.target_fname}) in processing: {len(self.target_df)}')
        
        if self.saveas:
            if os.path.isabs(self.saveas):
                self.save_folder = self.saveas
            else:
                self.save_folder = os.path.join(self.data_folder, self.saveas)
        elif not self.save_folder: # if save_folder is not specified, save in data_folder, with auto-generated name
            # Create base folder with just source_to_target (no parameters)
            folder_name = self.source_fname + '_to_' + self.target_fname
            if self.folder_prefix:
                folder_name = f"{self.folder_prefix}_{folder_name}"
            self.save_folder = os.path.join(self.data_folder, folder_name)
        elif not os.path.isabs(self.save_folder): # if save_folder is not absolute path, save in data_folder with specified relative path and name
            self.save_folder = os.path.join(self.data_folder, self.save_folder)
        if not os.path.exists(self.save_folder): os.makedirs(self.save_folder)
        print(f'data will be saved in: {self.save_folder}\n')
        
        # Prepare parameter dictionary (will be saved in method-specific subfolders)
        self.parameter_dict = {
            'source neurons': str(self.sourceNeurons),
            'source name': self.source_fname,
            'target neurons': str(self.targetNeurons),
            'target name': self.target_fname,
            'min synapse number': str(self.min_synapse_num),
            'min connection ratio': str(self.min_ratio),
            'min traversal probability': str(self.min_traversal_probability),
            'filter by': self.filter_by,
            'exclude intra-type connections': str(self.exclude_intra_type_connections),
            'max interlayer': str(self.max_interlayer),
            'keyword in path to remove': self.keyword_in_path_to_remove,
            'server': self.server,
            'dataset': self.dataset,
            'run date': self.run_date,
        }
        self.parameter_dict.update(self.kwargs_fetch)
        
        # Create parameter DataFrame (for use in methods)
        self.parameter_df = pd.DataFrame.from_dict(self.parameter_dict, orient='index', columns=['value'])
        self.parameter_df.reset_index(inplace=True)
        self.parameter_df.columns = ['parameter','value']
        
        # If max_interlayer == -1, only fetch neurons without connections
        if self.max_interlayer == -1:
            print('\033[36mmax_interlayer=-1: Neurons fetched (no connections will be queried)\033[0m')
            print('Use FetchNeuronsOnly() for connectivity profile analysis.')
    
    def FetchNeuronsOnly(self) -> tuple:
        '''
        Fetch source and target neurons only, without any connection data.
        
        This method is optimized for connectivity profile analysis where only
        neuron information is needed, not the actual connections between them.
        
        When sourceNeurons == targetNeurons (strict equality) with max_interlayer=-1,
        only one fetch is performed and the same DataFrame is returned for both.
        
        Returns:
            tuple: (source_df, target_df) as pandas DataFrames
            
        Example:
            >>> fnc = FindNeuronConnection()
            >>> fnc.sourceNeurons = ['aMe12', 'aMe10']
            >>> fnc.targetNeurons = ['PPL101', 'KC']
            >>> fnc.max_interlayer = -1  # Signal: neurons only
            >>> fnc.InitializeNeuronInfo()
            >>> source_df, target_df = fnc.FetchNeuronsOnly()
        '''
        if not hasattr(self, 'source_df') or not hasattr(self, 'target_df'):
            raise RuntimeError("Call InitializeNeuronInfo() first")
        
        print(f'\n=== FetchNeuronsOnly ===')
        print(f'Source neurons: {len(self.source_df)} ({self.source_fname})')
        if hasattr(self, '_source_target_identical') and self._source_target_identical:
            print(f'Target neurons: same as source (optimized)')
        else:
            print(f'Target neurons: {len(self.target_df)} ({self.target_fname})')
        print(f'No connections fetched (max_interlayer={self.max_interlayer})')
        
        return self.source_df.copy(), self.target_df.copy()
    
    def GetNeuronTypes(self, role: str = 'both') -> list:
        '''
        Get unique neuron types from source and/or target neurons.
        
        Args:
            role: 'source', 'target', or 'both' (default)
            
        Returns:
            list: Unique neuron type names
            
        Example:
            >>> fnc.InitializeNeuronInfo()
            >>> types = fnc.GetNeuronTypes('source')
            >>> print(types)  # ['aMe12', 'aMe10']
        '''
        if not hasattr(self, 'source_df') or not hasattr(self, 'target_df'):
            raise RuntimeError("Call InitializeNeuronInfo() first")
        
        types = []
        if role in ['source', 'both']:
            if 'type' in self.source_df.columns:
                types.extend(self.source_df['type'].dropna().unique().tolist())
        if role in ['target', 'both']:
            if 'type' in self.target_df.columns:
                types.extend(self.target_df['type'].dropna().unique().tolist())
        
        return list(set(types))
    
    def GetNeuronBodyIds(self, role: str = 'both') -> list:
        '''
        Get all bodyIds from source and/or target neurons.
        
        Args:
            role: 'source', 'target', or 'both' (default)
            
        Returns:
            list: bodyIds as integers
        '''
        if not hasattr(self, 'source_df') or not hasattr(self, 'target_df'):
            raise RuntimeError("Call InitializeNeuronInfo() first")
        
        bodyids = []
        if role in ['source', 'both']:
            if 'bodyId' in self.source_df.columns:
                bodyids.extend(self.source_df['bodyId'].tolist())
        if role in ['target', 'both']:
            if 'bodyId' in self.target_df.columns:
                bodyids.extend(self.target_df['bodyId'].tolist())
        
        return list(set(bodyids))

    def SaveNeuronInfo(self, output_dir: str = None, filename_prefix: str = None) -> str:
        '''
        Save source and target neuron information to CSV files.
        
        This method is particularly useful when max_interlayer=-1 (neurons-only mode),
        as no connection methods are called that would normally save neuron info.
        
        Args:
            output_dir: Directory to save files (default: self.save_folder)
            filename_prefix: Prefix for output files (default: source_fname_to_target_fname)
            
        Returns:
            str: Path to the output directory
            
        Example:
            >>> fnc = FindNeuronConnection()
            >>> fnc.sourceNeurons = ['aMe12', 'aMe10']
            >>> fnc.targetNeurons = ['PPL101', 'KC']
            >>> fnc.max_interlayer = -1
            >>> fnc.InitializeNeuronInfo()
            >>> fnc.SaveNeuronInfo()  # Saves source_neurons.csv, target_neurons.csv
        '''
        if not hasattr(self, 'source_df') or not hasattr(self, 'target_df'):
            raise RuntimeError("Call InitializeNeuronInfo() first")
        
        # Determine output directory
        if output_dir is None:
            output_dir = self.save_folder
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Determine filename prefix
        if filename_prefix is None:
            filename_prefix = f"{self.source_fname}_to_{self.target_fname}"
        
        # Save source neurons
        source_path = os.path.join(output_dir, f'{filename_prefix}_source_neurons.csv')
        self.source_df.to_csv(source_path, index=False)
        
        # Save target neurons
        target_path = os.path.join(output_dir, f'{filename_prefix}_target_neurons.csv')
        if hasattr(self, '_source_target_identical') and self._source_target_identical:
            # When source==target, just copy the reference
            self.target_df.to_csv(target_path, index=False)
            print(f'Target neurons: same as source (saved separately)')
        else:
            self.target_df.to_csv(target_path, index=False)
        
        # Save parameters
        params_path = os.path.join(output_dir, f'{filename_prefix}_parameters.csv')
        if hasattr(self, 'parameter_df'):
            self.parameter_df.to_csv(params_path, index=False)
        
        print(f'\n=== SaveNeuronInfo ===')
        print(f'Output directory: {output_dir}')
        print(f'Source neurons saved: {source_path}')
        print(f'Target neurons saved: {target_path}')
        print(f'Parameters saved: {params_path}')
        print(f'Source: {len(self.source_df)} neurons')
        print(f'Target: {len(self.target_df)} neurons')
        
        return output_dir

    def PrintROIHierarchy(self):
        '''print the ROI hierarchy, with primary ROIs marked with *'''
        # Show the ROI hierarchy, with primary ROIs marked with '*'
        print('*: Primary ROI')
        print(fetch_roi_hierarchy(False, mark_primary=True, format='text'))
            
    def FindDirectConnections(self):
        '''
        find direct connections between source and target neurons
        '''
        # Reset status columns if they exist
        self._reset_temp_columns()

        # Create direct folder with parameters and timestamp (match FindAllPath/FindPath naming)
        def _format_decimal_for_folder(value):
            """Format decimal numbers for folder-safe string (replace . with _ and negative sign with 'neg')"""
            if isinstance(value, (int, float)):
                s = str(value)
                return s.replace('.', '_').replace('-', 'neg')
            return str(value)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        param_suffix = (
            f"L{self.max_interlayer}"
            f"w{self.min_synapse_num}"
            f"r{_format_decimal_for_folder(self.min_ratio)}"
            f"p{_format_decimal_for_folder(self.min_traversal_probability)}"
            f"_{timestamp}"
        )
        
        if self.saveas:
            # If saveas is set, use save_folder directly (it was set to saveas in InitializeNeuronInfo)
            self.direct_folder = self.save_folder
        else:
            # Otherwise create subfolder with parameters
            self.direct_folder = os.path.join(self.save_folder, f'direct_{param_suffix}')
            
        if not os.path.exists(self.direct_folder): os.makedirs(self.direct_folder)
        
        # Initialize parameter.txt file
        self.parameter_txt = os.path.join(self.direct_folder, 'parameters.txt')
        with open(self.parameter_txt, 'w') as f:
            for key, value in self.parameter_dict.items():
                f.write(f'{key}: {value}\n')
            f.write('\n')
        # fetch connection table with caching
        print('Fetching direct connections:')
        source_bodyIds = self.source_df['bodyId'].tolist()
        target_bodyIds = self.target_df['bodyId'].tolist()
        
        # Optimization: Always fetch all downstream connections
        # This ensures neurons are marked as 'downstream_complete' in the cache
        # and avoids potential issues with API-side target filtering (especially for FlyWire)
        print('  (Fetching all downstream connections for robust caching)')
        self.conn_df = self._fetch_connections_with_cache(
            upstream_bodyIds=source_bodyIds,
            downstream_bodyIds=None,  # Fetch ALL downstream
            min_weight=self.min_synapse_num,
            min_conn_ratio=self.min_ratio,
            min_traversal_prob=self.min_traversal_probability
        )
        
        # Filter to only keep connections within the target set
        if not self.conn_df.empty:
            # Ensure bodyId_post is string for comparison
            self.conn_df['bodyId_post'] = self.conn_df['bodyId_post'].astype(str)
            target_bodyIds = [str(x) for x in target_bodyIds]
            self.conn_df = self.conn_df[self.conn_df['bodyId_post'].isin(target_bodyIds)].copy()
        if self.conn_df.empty:
            print('\033[33mNo direct connections found.\033[0m\n')
            return
        
        # enrich connection information (recalculate metrics for display)
        # Type-level prob = 1 - product(bodyId-level block_prob)
        # Don't pass target_neurons_df - let EnrichConnectionTable use neurons from connections
        # This uses sum(post) of neurons that actually received connections as denominator
        self.conn_df, self.conn_type, self.conn_group = sv.EnrichConnectionTable(
            self.conn_df, 
            traversal_probability_threshold=0,
            dataset=self.dataset,
            script_path=self.script_path,
            aggregate_method='product'  # Type-level prob = 1 - product(bodyId-level block_prob)
        )
        # fill empty values
        self.conn_df = self.conn_df.fillna("")
        self.source_df = self.source_df.fillna("")
        self.target_df = self.target_df.fillna("")
        print(f'Found connected neuron pairs: {len(self.conn_df)}')
        print(f'Total synapses between {self.source_fname} and {self.target_fname}: {self.conn_df.weight.sum()}')
        # convert connection table to matrix
        self.conn_matrix_bodyId: pd.DataFrame = connection_table_to_matrix(self.conn_df, group_cols='bodyId', sort_by='type')
        self.conn_matrix_bodyId.index = self.conn_matrix_bodyId.index.astype(str)
        self.conn_matrix_bodyId.columns = self.conn_matrix_bodyId.columns.astype(str)
        self.conn_matrix_type: pd.DataFrame = connection_table_to_matrix(self.conn_df, group_cols='type', sort_by='type')
        self.conn_matrix_type.index = self.conn_matrix_type.index.astype(str)
        self.conn_matrix_type.columns = self.conn_matrix_type.columns.astype(str)
        self.cmat_full_bodyId,self.cmat_full_type = sv.Conn2FullMat(self.source_df,self.target_df,self.conn_df,self.conn_type)
        self.transitionMat_bodyId,self.transitionMat_type = sv.Conn2FullMat(self.source_df,self.target_df,self.conn_df,self.conn_type,weight_col='traversal_probability')
        # Create ratio-based matrices (both square and full rectangular)
        self.conn_matrix_ratio_bodyId: pd.DataFrame = connection_table_to_matrix(self.conn_df, group_cols='bodyId', sort_by='type', weight_col='connection_ratio')
        self.conn_matrix_ratio_bodyId.index = self.conn_matrix_ratio_bodyId.index.astype(str)
        self.conn_matrix_ratio_bodyId.columns = self.conn_matrix_ratio_bodyId.columns.astype(str)
        # IMPORTANT: Use conn_type (not conn_df) for type-level ratio matrix
        # conn_type has corrected ratios that sum to 1.0 for each target type
        self.conn_matrix_ratio_type: pd.DataFrame = connection_table_to_matrix(self.conn_type, group_cols='type', sort_by='type', weight_col='connection_ratio')
        self.conn_matrix_ratio_type.index = self.conn_matrix_ratio_type.index.astype(str)
        self.conn_matrix_ratio_type.columns = self.conn_matrix_ratio_type.columns.astype(str)
        # Create full rectangular ratio matrices (source rows × target cols)
        self.ratioMat_full_bodyId,self.ratioMat_full_type = sv.Conn2FullMat(self.source_df,self.target_df,self.conn_df,self.conn_type,weight_col='connection_ratio')
        
        # Create custom group matrices if custom grouping was used
        if self.conn_group is not None:
            # Create connection matrices for custom groups
            self.conn_matrix_group: pd.DataFrame = self.conn_group.pivot_table(
                index='group_pre', columns='group_post', values='weight', fill_value=0
            )
            self.conn_matrix_group.index = self.conn_matrix_group.index.astype(str)
            self.conn_matrix_group.columns = self.conn_matrix_group.columns.astype(str)
            
            self.conn_matrix_ratio_group: pd.DataFrame = self.conn_group.pivot_table(
                index='group_pre', columns='group_post', values='connection_ratio', fill_value=0
            )
            self.conn_matrix_ratio_group.index = self.conn_matrix_ratio_group.index.astype(str)
            self.conn_matrix_ratio_group.columns = self.conn_matrix_ratio_group.columns.astype(str)
        
        # Ensure string comparison for bodyIds
        self.conn_df['bodyId_pre'] = self.conn_df['bodyId_pre'].astype(str)
        self.conn_df['bodyId_post'] = self.conn_df['bodyId_post'].astype(str)
        
        self.source_in_conn: pd.DataFrame = self.source_df[self.source_df['bodyId'].astype(str).isin(self.conn_df['bodyId_pre'].unique())]
        self.source_in_conn = self.source_in_conn.reset_index(drop=True)
        self.target_in_conn: pd.DataFrame = self.target_df[self.target_df['bodyId'].astype(str).isin(self.conn_df['bodyId_post'].unique())]
        self.target_in_conn = self.target_in_conn.reset_index(drop=True)
        print(f'{len(self.source_in_conn)} / {len(self.source_df)} source neurons involved in connections')
        print(f'{len(self.target_in_conn)} / {len(self.target_df)} target neurons involved in connections')
        with open(self.parameter_txt, 'a') as f:
            f.write(f'{len(self.source_in_conn)} / {len(self.source_df)} source {self.source_fname} neurons involved in connections\n')
            f.write(f'{len(self.target_in_conn)} / {len(self.target_df)} target {self.target_fname} neurons involved in connections\n')
            f.write('\n')
        
        # Save main file with type-level and custom group data
        if self.output_format == 'csv':
            print(f'Saving type-level connection info to CSV files...')
            
            # Create data_details subfolder
            details_folder = os.path.join(self.direct_folder, 'data_details')
            os.makedirs(details_folder, exist_ok=True)
            
            base_name = os.path.join(details_folder, self.source_fname+'_to_'+self.target_fname+'_info_snp'+str(self.min_synapse_num))
            
            self.parameter_df.to_csv(base_name + '_parameters.csv')
            self.source_df.to_csv(base_name + '_source_info.csv')
            self.target_df.to_csv(base_name + '_target_info.csv')
            self.source_in_conn.to_csv(base_name + '_source_in_connection.csv')
            self.target_in_conn.to_csv(base_name + '_target_in_connection.csv')
            self.conn_type.to_csv(base_name + '_connection_groupby_type.csv')
            
            # Add custom group sheets if custom grouping was used
            if self.conn_group is not None:
                self.conn_group.to_csv(base_name + '_connection_groupby_custom.csv')
                if not self.largeTargetSet:
                    self.conn_matrix_group.to_csv(base_name + '_connectionMatrix_group.csv')
                    self.conn_matrix_ratio_group.to_csv(base_name + '_connectionRatioMat_group.csv')
                else:
                    self.conn_matrix_group.transpose().to_csv(base_name + '_connectionMatrix_group.csv')
                    self.conn_matrix_ratio_group.transpose().to_csv(base_name + '_connectionRatioMat_group.csv')
            
            # Type-level matrices
            if not self.largeTargetSet:
                self.conn_matrix_type.to_csv(base_name + '_connectionMatrix_type.csv')
                self.cmat_full_type.to_csv(base_name + '_connMat_type_full.csv')
                self.transitionMat_type.to_csv(base_name + '_transmissionMat_type.csv')
                self.conn_matrix_ratio_type.to_csv(base_name + '_connectionRatioMat_type.csv')
                self.ratioMat_full_type.to_csv(base_name + '_ratioMat_type_full.csv')
            else:
                self.conn_matrix_type.transpose().to_csv(base_name + '_connectionMatrix_type.csv')
                self.cmat_full_type.transpose().to_csv(base_name + '_connMat_type_full.csv')
                self.transitionMat_type.transpose().to_csv(base_name + '_transmissionMat_type.csv')
                self.conn_matrix_ratio_type.transpose().to_csv(base_name + '_connectionRatioMat_type.csv')
        else:
            output_excel_name = os.path.join(self.direct_folder,self.source_fname+'_to_'+self.target_fname+'_info_snp'+str(self.min_synapse_num)+'.xlsx')
            print(f'Saving type-level connection info to excel file...')
            with pd.ExcelWriter(output_excel_name, mode='w', engine='xlsxwriter') as dataWriter:
                self.parameter_df.to_excel(dataWriter,sheet_name='parameters')
                worksheet = dataWriter.sheets['parameters']
                worksheet.set_column('A:A', 30, dataWriter.book.add_format({'bold': True, 'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
                worksheet.set_column('B:B', 30, dataWriter.book.add_format({'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
                
                self.source_df.to_excel(dataWriter,sheet_name='source_info')
                self.target_df.to_excel(dataWriter,sheet_name='target_info')
                self.source_in_conn.to_excel(dataWriter,sheet_name='source_in_connection')
                self.target_in_conn.to_excel(dataWriter,sheet_name='target_in_connection')
                self.conn_type.to_excel(dataWriter,sheet_name='connection_groupby_type')
                
                # Add custom group sheets if custom grouping was used
                if self.conn_group is not None:
                    self.conn_group.to_excel(dataWriter,sheet_name='connection_groupby_custom')
                    if not self.largeTargetSet:
                        self.conn_matrix_group.to_excel(dataWriter,sheet_name='connectionMatrix_group')
                        self.conn_matrix_ratio_group.to_excel(dataWriter,sheet_name='connectionRatioMat_group')
                    else:
                        self.conn_matrix_group.transpose().to_excel(dataWriter,sheet_name='connectionMatrix_group')
                        self.conn_matrix_ratio_group.transpose().to_excel(dataWriter,sheet_name='connectionRatioMat_group')
                
                # Type-level matrices
                if not self.largeTargetSet:
                    self.conn_matrix_type.to_excel(dataWriter,sheet_name='connectionMatrix_type')
                    self.cmat_full_type.to_excel(dataWriter,sheet_name='connMat_type_full')
                    self.transitionMat_type.to_excel(dataWriter,sheet_name='transmissionMat_type')
                    self.conn_matrix_ratio_type.to_excel(dataWriter,sheet_name='connectionRatioMat_type')
                    self.ratioMat_full_type.to_excel(dataWriter,sheet_name='ratioMat_type_full')
                else:
                    self.conn_matrix_type.transpose().to_excel(dataWriter,sheet_name='connectionMatrix_type')
                    self.cmat_full_type.transpose().to_excel(dataWriter,sheet_name='connMat_type_full')
                    self.transitionMat_type.transpose().to_excel(dataWriter,sheet_name='transmissionMat_type')
                    self.conn_matrix_ratio_type.transpose().to_excel(dataWriter,sheet_name='connectionRatioMat_type')
        
        # Save bodyId-level data (use CSV for large data)
        print(f'Saving bodyId-level data (rows: {len(self.conn_df):,})...')
        
        EXCEL_ROW_LIMIT = 1_048_576
        use_csv = (len(self.conn_df) >= EXCEL_ROW_LIMIT * 0.9) or (self.output_format == 'csv')
        
        if use_csv:
            if len(self.conn_df) >= EXCEL_ROW_LIMIT * 0.9:
                print(f'  ⚠️  Data too large for Excel ({len(self.conn_df):,} rows), saving as CSV')
            else:
                print(f'  Saving as CSV (requested format)')
            
            # Save parameters
            if self.output_format == 'csv':
                # Create data_details subfolder
                details_folder = os.path.join(self.direct_folder, 'data_details')
                os.makedirs(details_folder, exist_ok=True)
                
                output_params_csv = os.path.join(details_folder, self.source_fname+'_to_'+self.target_fname+'_bodyId_parameters_snp'+str(self.min_synapse_num)+'.csv')
                self.parameter_df.to_csv(output_params_csv)
            else:
                output_params_excel = os.path.join(self.direct_folder,self.source_fname+'_to_'+self.target_fname+'_bodyId_parameters_snp'+str(self.min_synapse_num)+'.xlsx')
                with pd.ExcelWriter(output_params_excel, mode='w', engine='xlsxwriter') as dataWriter:
                    self.parameter_df.to_excel(dataWriter,sheet_name='parameters')
                    worksheet = dataWriter.sheets['parameters']
                    worksheet.set_column('A:A', 30, dataWriter.book.add_format({'bold': True, 'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
                    worksheet.set_column('B:B', 30, dataWriter.book.add_format({'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
            
            # Save bodyId connection data as CSV
            output_bodyid_csv = os.path.join(self.direct_folder,self.source_fname+'_to_'+self.target_fname+'_bodyId_connections_snp'+str(self.min_synapse_num)+'.csv')
            self.conn_df.to_csv(output_bodyid_csv, index=False)
            print(f'  ✓ Saved to: {output_bodyid_csv}')
            
            # Save matrices as separate CSVs
            if not self.largeTargetSet:
                self.conn_matrix_bodyId.to_csv(os.path.join(self.direct_folder,self.source_fname+'_to_'+self.target_fname+'_connectionMatrix_bodyId.csv'))
                self.transitionMat_bodyId.to_csv(os.path.join(self.direct_folder,self.source_fname+'_to_'+self.target_fname+'_transmissionMat_bodyId.csv'))
            else:
                self.conn_matrix_bodyId.transpose().to_csv(os.path.join(self.direct_folder,self.source_fname+'_to_'+self.target_fname+'_connectionMatrix_bodyId.csv'))
                self.transitionMat_bodyId.transpose().to_csv(os.path.join(self.direct_folder,self.source_fname+'_to_'+self.target_fname+'_transmissionMat_bodyId.csv'))
        else:
            # Data fits in Excel
            output_bodyid_excel = os.path.join(self.direct_folder,self.source_fname+'_to_'+self.target_fname+'_bodyId_data_snp'+str(self.min_synapse_num)+'.xlsx')
            with pd.ExcelWriter(output_bodyid_excel, mode='w', engine='xlsxwriter') as dataWriter:
                self.parameter_df.to_excel(dataWriter,sheet_name='parameters')
                worksheet = dataWriter.sheets['parameters']
                worksheet.set_column('A:A', 30, dataWriter.book.add_format({'bold': True, 'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
                worksheet.set_column('B:B', 30, dataWriter.book.add_format({'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
                
                self.conn_df.to_excel(dataWriter,sheet_name='connection_info_bodyId')
                
                if not self.largeTargetSet:
                    self.conn_matrix_bodyId.to_excel(dataWriter,sheet_name='connectionMatrix_bodyId')
                    self.cmat_full_bodyId.to_excel(dataWriter,sheet_name='connMat_bodyId_full')
                    self.transitionMat_bodyId.to_excel(dataWriter,sheet_name='transmissionMat_bodyId')
                    self.conn_matrix_ratio_bodyId.to_excel(dataWriter,sheet_name='connectionRatioMat_bodyId')
                    self.ratioMat_full_bodyId.to_excel(dataWriter,sheet_name='ratioMat_bodyId_full')
                else:
                    self.conn_matrix_bodyId.transpose().to_excel(dataWriter,sheet_name='connectionMatrix_bodyId')
                    self.cmat_full_bodyId.transpose().to_excel(dataWriter,sheet_name='connMat_bodyId_full')
                    self.transitionMat_bodyId.transpose().to_excel(dataWriter,sheet_name='transmissionMat_bodyId')
                    self.conn_matrix_ratio_bodyId.transpose().to_excel(dataWriter,sheet_name='connectionRatioMat_bodyId')
                    self.ratioMat_full_bodyId.transpose().to_excel(dataWriter,sheet_name='ratioMat_bodyId_full')
            print(f'  ✓ Saved to: {output_bodyid_excel}')
        
        print('Done\n')
        self.VisualizeDirectConnections_simple()
        return 0
        
    def VisualizeDirectConnections_simple(self):
        # Visualize connection matrix in heatmap using CreateHeatmap class
        print('Visualizing connection matrix in heatmap...')
        # print('  (Legacy heatmap generation disabled)')
        
        # # Optionally filter out empty rows/columns
        # if filter_zeros:
        #     # Filter matrices to remove empty rows/columns
        #     cmat_bodyId = self.cmat_full_bodyId.loc[
        #         self.cmat_full_bodyId.sum(axis=1) > 0,
        #         self.cmat_full_bodyId.sum(axis=0) > 0
        #     ]
        #     cmat_type = self.cmat_full_type.loc[
        #         self.cmat_full_type.sum(axis=1) > 0,
        #         self.cmat_full_type.sum(axis=0) > 0
        #     ]
        #     transitionMat_bodyId = self.transitionMat_bodyId.loc[
        #         self.transitionMat_bodyId.sum(axis=1) > 0,
        #         self.transitionMat_bodyId.sum(axis=0) > 0
        #     ]
        #     transitionMat_type = self.transitionMat_type.loc[
        #         self.transitionMat_type.sum(axis=1) > 0,
        #         self.transitionMat_type.sum(axis=0) > 0
        #     ]
        #     ratioMat_bodyId = self.ratioMat_full_bodyId.loc[
        #         self.ratioMat_full_bodyId.sum(axis=1) > 0,
        #         self.ratioMat_full_bodyId.sum(axis=0) > 0
        #     ]
        #     ratioMat_type = self.ratioMat_full_type.loc[
        #         self.ratioMat_full_type.sum(axis=1) > 0,
        #         self.ratioMat_full_type.sum(axis=0) > 0
        #     ]
        #     print(f'  Filtered matrices: bodyId ({self.cmat_full_bodyId.shape} → {cmat_bodyId.shape}), type ({self.cmat_full_type.shape} → {cmat_type.shape})')
        # else:
        #     # Use full matrices
        #     cmat_bodyId = self.cmat_full_bodyId
        #     cmat_type = self.cmat_full_type
        #     transitionMat_bodyId = self.transitionMat_bodyId
        #     transitionMat_type = self.transitionMat_type
        #     ratioMat_bodyId = self.ratioMat_full_bodyId
        #     ratioMat_type = self.ratioMat_full_type
        
        # # Create heatmap generator instance
        # heatmap_gen = sv.CreateHeatmap(
        #     output_folder=self.direct_folder,
        #     showfig=self.showfig
        # )
        
        # # Add connection matrix heatmaps (use filtered or full matrices based on parameter)
        # # Use interactive mode for bodyId heatmaps (allows user to switch scales)
        # heatmap_gen.add_heatmap(
        #     matrix=cmat_bodyId,
        #     name=f'heatmap_connMatrix_bodyId_snp{self.min_synapse_num}',
        #     title=f'heatmap of connection matrix: {self.source_fname} to {self.target_fname}<br>based on bodyId',
        #     color_scale='green',
        #     interactive=True,  # Enable interactive controls
        #     conn_df=self.conn_df  # Pass connection data for enhanced hover info
        # )
        
        # # Type heatmaps - enable interactive UI for user control
        # heatmap_gen.add_heatmap(
        #     matrix=cmat_type,
        #     name=f'heatmap_connMatrix_type_snp{self.min_synapse_num}',
        #     title=f'heatmap of connection matrix: {self.source_fname} to {self.target_fname}<br>based on type',
        #     color_scale='purple',
        #     interactive=True  # Enable interactive controls
        # )
        
        # # Add transmission matrix heatmaps
        # heatmap_gen.add_heatmap(
        #     matrix=transitionMat_bodyId,
        #     name=f'heatmap_transmissionMat_bodyId_snp{self.min_synapse_num}',
        #     title=f'heatmap of full transmission matrix: {self.source_fname} to {self.target_fname}<br>based on bodyId',
        #     color_scale='green',
        #     interactive=True,  # Enable interactive controls
        #     conn_df=self.conn_df  # Pass connection data for enhanced hover info
        # )
        
        # heatmap_gen.add_heatmap(
        #     matrix=transitionMat_type,
        #     name=f'heatmap_transmissionMat_type_snp{self.min_synapse_num}',
        #     title=f'heatmap of full transmission matrix: {self.source_fname} to {self.target_fname}<br>based on type',
        #     color_scale='purple',
        #     interactive=True  # Enable interactive controls
        # )
        
        # # Add ratio matrix heatmaps (use filtered or full matrices)
        # heatmap_gen.add_heatmap(
        #     matrix=ratioMat_bodyId,
        #     name=f'heatmap_ratioMat_bodyId_snp{self.min_synapse_num}',
        #     title=f'heatmap of connection ratio matrix: {self.source_fname} to {self.target_fname}<br>based on bodyId',
        #     color_scale='orange',
        #     interactive=True,  # Enable interactive controls
        #     conn_df=self.conn_df  # Pass connection data for enhanced hover info
        # )
        
        # heatmap_gen.add_heatmap(
        #     matrix=ratioMat_type,
        #     name=f'heatmap_ratioMat_type_snp{self.min_synapse_num}',
        #     title=f'heatmap of connection ratio matrix: {self.source_fname} to {self.target_fname}<br>based on type',
        #     color_scale='orange',
        #     interactive=True  # Enable interactive controls
        # )
        
        # # Generate all heatmaps
        # heatmap_gen.create_all()
        # # Visualize by sankey diagram and network graph, only for neuron type
        # print('Visualizing by Sankey diagram and network graph...')
        # # sankey_name = 'sankey_type_snp'+str(self.min_synapse_num)+'.html'
        # # sv.SankeyDirect(self.conn_matrix_type,file_path=os.path.join(self.direct_folder,sankey_name),showfig=self.showfig,node_color=self.node_color,link_color=self.link_color)
        # # Create ratio-based Sankey diagram
        # # sankey_ratio_name = 'sankey_type_ratio_snp'+str(self.min_synapse_num)+'.html'
        # # sv.SankeyDirect(self.conn_matrix_ratio_type,file_path=os.path.join(self.direct_folder,sankey_ratio_name),showfig=self.showfig,node_color=self.node_color,link_color=self.link_color)
        # print('Done\n')
        
        # VisualizePath network visualization for direct connections
        print('Creating VisualizePath network visualization...')
        try:
            
            # Convert direct connections to path format
            # Each connection is a single-hop path: source -> target
            if len(self.conn_type) > 0:
                path_data = []
                for idx in self.conn_type.index:
                    source = self.conn_type.at[idx, 'type_pre']
                    target = self.conn_type.at[idx, 'type_post']
                    weight = self.conn_type.at[idx, 'weight']
                    ratio = self.conn_type.at[idx, 'connection_ratio'] if 'connection_ratio' in self.conn_type.columns else 0.0
                    prob = self.conn_type.at[idx, 'traversal_probability'] if 'traversal_probability' in self.conn_type.columns else 0.0
                    
                    # Create a single-hop path
                    path_data.append({
                        'path_block': f'{source} -> {target}',
                        'weights': [weight],
                        'connection_ratios': [ratio],
                        'traversal_probabilities': [prob]
                    })
                
                # Create DataFrame from path data
                import pandas as pd
                path_df = pd.DataFrame(path_data)
                
                # Create VisualizePath visualization with path data
                # This creates: (1) Heatmap, (2) Sankey diagram, (3) Network graph
                vp = VisualizePath(
                    path_file=path_df,
                    output_folder=self.direct_folder,
                    source_color=self.source_color if hasattr(self, 'source_color') else '#1f77b4',
                    intermediate_color=self.intermediate_color if hasattr(self, 'intermediate_color') else '#2ca02c',
                    target_color=self.target_color if hasattr(self, 'target_color') else '#d62728',
                    link_color=self.link_color if hasattr(self, 'link_color') else 'rgba(100,100,100,0.3)',
                    network_layout=self.network_layout if hasattr(self, 'network_layout') else 'hierarchical',
                    showfig=self.showfig,
                    edgeN_limit=self.edgeN_limit,
                    verbose=(self.verbose_mode == 'full')
                )
                vp.visualize()
                self._vprint('  ✓ Created complete VisualizePath visualization:')
                self._vprint('    - Interactive heatmap (type-level connections)')
                self._vprint('    - Sankey diagram (flow visualization)')
                self._vprint('    - Network graph (interactive topology)')
                
            else:
                self._vprint('  No connections to visualize')
            
            # Create VisualizePath visualization for bodyId-level connections
            if len(self.conn_df) > 0:
                self._vprint('\nCreating VisualizePath visualization for bodyId-level connections...')
                bodyId_path_data = []
                for idx in self.conn_df.index:
                    # Add type suffix to bodyIds
                    source_type = str(self.conn_df.at[idx, 'type_pre'])
                    target_type = str(self.conn_df.at[idx, 'type_post'])
                    source = f"{self.conn_df.at[idx, 'bodyId_pre']}_{source_type}"
                    target = f"{self.conn_df.at[idx, 'bodyId_post']}_{target_type}"
                    
                    weight = self.conn_df.at[idx, 'weight']
                    ratio = self.conn_df.at[idx, 'connection_ratio'] if 'connection_ratio' in self.conn_df.columns else 0.0
                    prob = self.conn_df.at[idx, 'traversal_probability'] if 'traversal_probability' in self.conn_df.columns else 0.0
                    
                    # Create a single-hop path
                    bodyId_path_data.append({
                        'path_block': f'{source} -> {target}',
                        'weights': [weight],
                        'connection_ratios': [ratio],
                        'traversal_probabilities': [prob]
                    })
                
                # Create DataFrame from path data
                bodyId_path_df = pd.DataFrame(bodyId_path_data)
                
                # Create VisualizePath visualization for bodyId
                vp_bodyId = VisualizePath(
                    path_file=bodyId_path_df,
                    output_folder=os.path.join(self.direct_folder, 'bodyId_visualization'),
                    source_color=self.source_color if hasattr(self, 'source_color') else '#1f77b4',
                    intermediate_color=self.intermediate_color if hasattr(self, 'intermediate_color') else '#2ca02c',
                    target_color=self.target_color if hasattr(self, 'target_color') else '#d62728',
                    link_color=self.link_color if hasattr(self, 'link_color') else 'rgba(100,100,100,0.3)',
                    network_layout=self.network_layout if hasattr(self, 'network_layout') else 'hierarchical',
                    showfig=self.showfig,
                    edgeN_limit=self.edgeN_limit,
                    output_format=self.output_format,
                    verbose=(self.verbose_mode == 'full')
                )
                vp_bodyId.visualize()
                self._vprint('  ✓ Created VisualizePath visualization for bodyId-level connections:')
                self._vprint('    - Interactive heatmap (bodyId-level connections)')
                self._vprint('    - Sankey diagram (bodyId flow visualization)')
                self._vprint('    - Network graph (bodyId topology)')
                
            # Create visualization for custom groups if available
            if self.conn_group is not None and len(self.conn_group) > 0:
                self._vprint('\nCreating VisualizePath visualization for custom groups...')
                group_path_data = []
                for idx in self.conn_group.index:
                    source = self.conn_group.at[idx, 'group_pre']
                    target = self.conn_group.at[idx, 'group_post']
                    weight = self.conn_group.at[idx, 'weight']
                    ratio = self.conn_group.at[idx, 'connection_ratio'] if 'connection_ratio' in self.conn_group.columns else 0.0
                    prob = self.conn_group.at[idx, 'traversal_probability'] if 'traversal_probability' in self.conn_group.columns else 0.0
                    
                    # Create a single-hop path
                    group_path_data.append({
                        'path_block': f'{source} -> {target}',
                        'weights': [weight],
                        'connection_ratios': [ratio],
                        'traversal_probabilities': [prob]
                    })
                
                # Create DataFrame from path data
                group_path_df = pd.DataFrame(group_path_data)
                
                # Create VisualizePath visualization for custom groups
                vp_group = VisualizePath(
                    path_file=group_path_df,
                    output_folder=os.path.join(self.direct_folder, 'custom_groups'),
                    source_color=self.source_color if hasattr(self, 'source_color') else '#1f77b4',
                    intermediate_color=self.intermediate_color if hasattr(self, 'intermediate_color') else '#2ca02c',
                    target_color=self.target_color if hasattr(self, 'target_color') else '#d62728',
                    link_color=self.link_color if hasattr(self, 'link_color') else 'rgba(100,100,100,0.3)',
                    network_layout=self.network_layout if hasattr(self, 'network_layout') else 'hierarchical',
                    showfig=self.showfig,
                    edgeN_limit=self.edgeN_limit,
                    verbose=(self.verbose_mode == 'full')
                )
                vp_group.visualize()
                self._vprint('  ✓ Created VisualizePath visualization for custom groups:')
                self._vprint('    - Interactive heatmap (custom group connections)')
                self._vprint('    - Sankey diagram (group flow visualization)')
                self._vprint('    - Network graph (group topology)')
                
        except Exception as e:
            import traceback
            self._vprint(f'  Warning: VisualizePath visualization failed: {e}')
            self._vprint(traceback.format_exc())
        self._vprint('Done\n')
    
    def FindPath(self, find_bodyId_path=True):
        '''Find path between source and target neurons, adapted from FindInterClusterConnection.ipynb'''
        # Reset status columns if they exist (to allow sequential calls)
        self._reset_temp_columns()

        # Initialize output folder (base folder without parameters)
        base_folder = self.save_folder
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        
        # Create path folder with parameters and timestamp
        def format_decimal(value):
            """Convert decimal to folder-safe string (replace . with _)"""
            if isinstance(value, (int, float)):
                str_val = str(value)
                return str_val.replace('.', '_').replace('-', 'neg')
            return str(value)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        param_suffix = (
            f"L{self.max_interlayer}"
            f"w{self.min_synapse_num}"
            f"r{format_decimal(self.min_ratio)}"
            f"p{format_decimal(self.min_traversal_probability)}"
            f"_{timestamp}"
        )
        
        if self.saveas:
            # If saveas is set, use save_folder directly
            self.path_folder = self.save_folder
        else:
            # Otherwise create subfolder with parameters
            self.path_folder = os.path.join(base_folder, f'paths_{param_suffix}')
            
        if not os.path.exists(self.path_folder):
            os.makedirs(self.path_folder)
        targetNum = len(self.target_df)
        self.target_df.insert(loc=0,column='Checked',value=False)
        
        # Ensure bodyIds are strings for consistent processing (handles int64 vs str mismatch)
        self.source_df['bodyId'] = self.source_df['bodyId'].astype(str)
        self.target_df['bodyId'] = self.target_df['bodyId'].astype(str)
        
        source_ID = self.source_df['bodyId'].unique() # convert to np.ndarray
        target_ID = self.target_df['bodyId'].unique()
        target_type = self.target_df['type'].unique()
        currLayer = 0
        targetNum_checked = 0
        Flag = True
        conn_layers = []
        searchedNeurons = source_ID
        # searching for target neurons
        while Flag and currLayer <= self.max_interlayer:
            print(f'Layer {currLayer}->{currLayer+1}:')
            conn_df = self._fetch_connections_with_cache(
                upstream_bodyIds=source_ID.tolist(),
                downstream_bodyIds=None,
                min_weight=self.min_synapse_num,
                min_conn_ratio=self.min_ratio,
                min_traversal_prob=self.min_traversal_probability
            )
            
            # Ensure connection dataframe has string bodyIds
            if not conn_df.empty:
                conn_df['bodyId_pre'] = conn_df['bodyId_pre'].astype(str)
                conn_df['bodyId_post'] = conn_df['bodyId_post'].astype(str)
            
            conn_df = sv.removeSearchedNeurons(conn_df,searchedNeurons, exempt_neurons=target_ID)
            conn_layers.append(conn_df)
            post_ID = conn_df['bodyId_post'].unique()
            searchedNeurons = np.concatenate((searchedNeurons,post_ID),axis=0)
            print('fetched connections between L%d and L%d %d neurons    connection found: %d pairs'%(currLayer,currLayer+1,len(post_ID),len(conn_df)))
            ind = self.target_df['bodyId'].isin(post_ID)
            self.target_df.loc[ind,'Checked'] = True
            self.target_df.loc[ind,'Layer'] = currLayer + 1
            targetNum_checked = len(self.target_df[self.target_df['Checked'] == True])
            print('Total targets checked: %d / %d neurons'%(targetNum_checked,targetNum))
            if targetNum_checked == targetNum:
                Flag = False
            source_ID = post_ID
            currLayer += 1
            if len(post_ID) == 0:
                print('!!!NO NEURONS FOUND IN NEXT LAYER!!!')
                break
        if Flag: print('\nNOT All Target Neurons Traced')
        else: print('\nAll Target Neurons Traced')
        
        # searching layers
        conn_inpath = pd.DataFrame()
        conn_types = pd.DataFrame()
        post_ID = target_ID
        neuron_layers = [target_ID]
        weight_layers = {} # dict
        
        for i in reversed(range(len(conn_layers))): # searching for connection path from target neurons to source neurons
            conn: pd.DataFrame = conn_layers[i]
            conn_df = conn[conn['bodyId_post'].isin(post_ID)] # remove neurons not in the connection path
            if len(conn_df) == 0: continue # if not found target neurons in the last x searched layers, skip these layers (when max_interlayer is too large)
            
            # Get all neurons involved in this layer's connections (for accurate ratio calculation)
            bodyIds_in_layer = np.unique(np.concatenate([conn_df['bodyId_pre'].unique(), conn_df['bodyId_post'].unique()]))
            neurons_in_layer_df = self._fetch_neurons_local_or_api(bodyIds_in_layer.tolist(), columns=['bodyId', 'type', 'post'])
            
            conn_df, conn_type, conn_group = sv.EnrichConnectionTable(
                conn_df, 
                traversal_probability_threshold=0,
                dataset=self.dataset,
                script_path=self.script_path,
                target_neurons_df=neurons_in_layer_df
            )
            conn_df.insert(loc=0,column='conn_layer',value=str(i)+'->'+str(i+1))
            conn_type.insert(loc=0,column='conn_layer',value=str(i)+'->'+str(i+1))
            if conn_group is not None:
                conn_group.insert(loc=0,column='conn_layer',value=str(i)+'->'+str(i+1))
            conn_inpath = pd.concat([conn_inpath,conn_df])
            conn_types = pd.concat([conn_types,conn_type])
            
            post_ID = conn_df['bodyId_pre'].unique()
            neuron_layers.append(post_ID)
            post_ID = np.concatenate((post_ID,target_ID)) # post ID for next cycle. include target_ID because all target neurons may not be at the last layer
            post_ID = np.unique(post_ID)
            weight_layers.update({str(i)+'->'+str(i+1): conn_df['weight'].sum()})
            
        neuron_layers.reverse()
        if not conn_inpath.empty:
            conn_inpath = conn_inpath.sort_values(by=['conn_layer','traversal_probability','weight'],ascending=[True,False,False])
            conn_inpath = conn_inpath.reset_index(drop=True)
            conn_types = conn_types.sort_values(by=['conn_layer','traversal_probability','weight'],ascending=[True,False,False])
            conn_types = conn_types.reset_index(drop=True)
        else:
            print("Warning: No paths found connecting source to target.")

        totalweight_df = pd.DataFrame(weight_layers.items(),columns=['conn_layer','weight'])
        totalweight_df = totalweight_df.sort_values(by='conn_layer',ascending=True)

        self.source_df.insert(loc=0,column='isInPath',value=False)
        if not conn_inpath.empty:
            source_inpath = conn_inpath.loc[conn_inpath.conn_layer=='0->1','bodyId_pre'].unique()
            self.source_df.loc[self.source_df.bodyId.isin(source_inpath),'isInPath'] = True
        
        # Save main file with type-level data
        print('Saving type-level path info...')
        if self.output_format == 'csv':
            # Create data_details subfolder
            csv_folder = os.path.join(self.path_folder, 'data_details')
            os.makedirs(csv_folder, exist_ok=True)
            print(f'  💾 Saving data as CSV files to: {csv_folder}')
            self.parameter_df.to_csv(os.path.join(csv_folder, 'parameters.csv'), index=False)
            self.source_df.to_csv(os.path.join(csv_folder, 'source_neurons.csv'))
            self.target_df.to_csv(os.path.join(csv_folder, 'target_neurons.csv'))
            totalweight_df.to_csv(os.path.join(csv_folder, 'total_weight_layer.csv'))
            conn_types.to_csv(os.path.join(csv_folder, 'connection_type.csv'))
            self._save_matrices_to_csv(conn_types, csv_folder, level='type')
        else:
            output_excel_name = os.path.join(self.path_folder,self.source_fname+'_to_'+self.target_fname+'_path_info.xlsx')
            with pd.ExcelWriter(output_excel_name,mode='w',engine='xlsxwriter') as writer:
                self.parameter_df.to_excel(writer,sheet_name='parameters',index=False)
                worksheet = writer.sheets['parameters']
                worksheet.set_column('A:A', 30, writer.book.add_format({'bold': True, 'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
                worksheet.set_column('B:B', 30, writer.book.add_format({'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
                
                self.source_df.to_excel(writer,sheet_name='source_neurons')
                self.target_df.to_excel(writer,sheet_name='target_neurons')
                totalweight_df.to_excel(writer,sheet_name='total_weight_layer')
                conn_types.to_excel(writer,sheet_name='connection_type')
                self._save_matrices_to_excel(conn_types, writer, level='type')
        
        # Save bodyId-level data (use CSV if too large or if output_format='csv')
        print(f'Saving bodyId-level path data (rows: {len(conn_inpath):,})...')
        
        EXCEL_ROW_LIMIT = 1_048_576
        use_csv = (self.output_format == 'csv') or (len(conn_inpath) >= EXCEL_ROW_LIMIT * 0.9)
        
        if use_csv:
            if self.output_format == 'csv':
                print(f'  💾 Saving bodyId data as CSV (output_format="csv")')
            else:
                print(f'  ⚠️  Data too large for Excel ({len(conn_inpath):,} rows), saving as CSV')
            
            # Use data_details folder
            bodyid_folder = os.path.join(self.path_folder, 'data_details')
            os.makedirs(bodyid_folder, exist_ok=True)
            
            # Save parameters (if not already saved)
            if not os.path.exists(os.path.join(bodyid_folder, 'parameters.csv')):
                self.parameter_df.to_csv(os.path.join(bodyid_folder, 'parameters.csv'), index=False)
            
            # Save bodyId connection data as CSV
            output_bodyid_csv = os.path.join(bodyid_folder, 'connection_info_bodyId.csv')
            conn_inpath.to_csv(output_bodyid_csv, index=False)
            self._save_matrices_to_csv(conn_inpath, bodyid_folder, level='bodyId')
            print(f'  ✓ Saved to: {bodyid_folder}/')
        else:
            # Data fits in Excel
            output_bodyid_excel = os.path.join(self.path_folder,self.source_fname+'_to_'+self.target_fname+'_path_bodyId_data.xlsx')
            with pd.ExcelWriter(output_bodyid_excel,mode='w',engine='xlsxwriter') as writer:
                self.parameter_df.to_excel(writer,sheet_name='parameters',index=False)
                worksheet = writer.sheets['parameters']
                worksheet.set_column('A:A', 30, writer.book.add_format({'bold': True, 'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
                worksheet.set_column('B:B', 30, writer.book.add_format({'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
                
                conn_inpath.to_excel(writer,sheet_name='connection_info_bodyId')
                self._save_matrices_to_excel(conn_inpath, writer, level='bodyId')
            print(f'  ✓ Saved to: {output_bodyid_excel}')
        
        # get connection path (by type) - OPTIMIZED: Use direct graph pathfinding
        path_df_type = pd.DataFrame()
        print('Analyzing path info by type:')
        print('Building type-level graph and finding paths...')
        
        # Build type-level graph from conn_types
        G_type = nx.DiGraph()
        for idx in conn_types.index:
            row = conn_types.loc[idx]
            type_pre = row['type_pre']
            type_post = row['type_post']
            weight = row['weight']
            # Ensure scalar values (not Series)
            if isinstance(type_pre, pd.Series):
                type_pre = type_pre.iloc[0]
            if isinstance(type_post, pd.Series):
                type_post = type_post.iloc[0]
            if isinstance(weight, pd.Series):
                weight = weight.iloc[0]
                
            if G_type.has_edge(type_pre, type_post):
                G_type[type_pre][type_post]['weight'] += weight
            else:
                G_type.add_edge(type_pre, type_post, weight=weight)
        
        self._vprint(f'  Type-level graph: {G_type.number_of_nodes()} types, {G_type.number_of_edges()} edges', level='full')
        
        # Get source and target types (filter out NaN/None values)
        source_types = [t for t in self.source_df['type'].unique().tolist() 
                        if t is not None and (not isinstance(t, float) or not pd.isna(t))]
        target_types = [t for t in self.target_df.loc[self.target_df.Checked, 'type'].unique().tolist()
                        if t is not None and (not isinstance(t, float) or not pd.isna(t))]
        
        # Find paths using DFS on type graph
        type_paths = []
        for source_type in source_types:
            if source_type not in G_type:
                continue
            for target_type in target_types:
                if target_type not in G_type:
                    continue
                if nx.has_path(G_type, source_type, target_type):
                    # Find all simple paths with length <= max_interlayer + 1
                    for path in nx.all_simple_paths(G_type, source_type, target_type, cutoff=self.max_interlayer + 1):
                        type_paths.append(path)
        
        self._vprint(f'  Found {len(type_paths):,} type-level paths', level='full')
        
        # Build DataFrame from type paths (no real_layer_map needed - layer-by-layer ensures forward-only)
        path_df_type = sv.build_path_dataframe_from_paths(
            paths=type_paths,
            conn_data=conn_types,
            targets=target_types,
            real_layer_map=None,
            level='type'
        )
        
        # Filter out paths with any zero-weight hops
        # This happens when bodyId-level connections exist but type-level aggregation results in 0 weight
        if len(path_df_type) > 0:
            before_filter = len(path_df_type)
            path_df_type = path_df_type[
                path_df_type['weights'].apply(lambda w_list: all(w > 0 for w in w_list))
            ]
            after_filter = len(path_df_type)
            if before_filter > after_filter:
                self._vprint(f'  Removed {before_filter - after_filter} paths with zero-weight hops at type level', level='full')
        
        path_df_type = sv.split_path(path_df_type)
        path_df_type, path_df_type_excluded = sv.path_filter(path_df_type,self.keyword_in_path_to_remove)
        
        # Save configuration files to path folder
        self._vprint('\nSaving configuration files...', level='full')
        all_attributes_dict = {
            'source_fname': self.source_fname,
            'target_fname': self.target_fname,
            'max_interlayer': self.max_interlayer,
            'min_synapse_num': self.min_synapse_num,
            'min_ratio': self.min_ratio,
            'min_traversal_probability': self.min_traversal_probability,
            'keyword_in_path_to_remove': self.keyword_in_path_to_remove,
            'node_color': self.node_color,
            'target_color': self.target_color,
            'link_color': self.link_color,
            'showfig': self.showfig,
            'timestamp': timestamp
        }
        
        # Save as JSON
        with open(os.path.join(self.path_folder, 'all_attributes.json'), 'w') as f:
            json.dump(all_attributes_dict, f, indent=4)
        
        # Save as readable text
        with open(os.path.join(self.path_folder, 'parameters.txt'), 'w') as f:
            f.write(f"Analysis Parameters for FindPath\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Source neurons: {self.source_fname}\n")
            f.write(f"Target neurons: {self.target_fname}\n")
            f.write(f"Maximum interlayer: {self.max_interlayer}\n")
            f.write(f"Minimum synapse number: {self.min_synapse_num}\n")
            f.write(f"Minimum connection ratio: {self.min_ratio}\n")
            f.write(f"Minimum traversal probability: {self.min_traversal_probability}\n")
            f.write(f"Keywords to remove: {self.keyword_in_path_to_remove}\n")
            f.write(f"Timestamp: {timestamp}\n")
        
        # Display target statistics with found/total format
        print('\n' + '='*70)
        print('TARGET NEURON SUMMARY')
        print('='*70)
        
        # Get targets found in each layer
        targets_by_layer = {}
        all_found_targets = set()
        for layer_idx in range(1, self.max_interlayer + 1):
            layer_targets = self.target_df[self.target_df['Layer'] == layer_idx]['type'].unique()
            if len(layer_targets) > 0:
                targets_by_layer[layer_idx] = set(layer_targets)
                all_found_targets.update(layer_targets)
        
        total_target_types = len(self.target_df['type'].unique())
        total_found = len(all_found_targets)
        
        print(f'\nTotal target types: {total_found}/{total_target_types}')
        
        if targets_by_layer:
            print('\nTargets found by layer:')
            for layer_idx in sorted(targets_by_layer.keys()):
                layer_targets = sorted(list(targets_by_layer[layer_idx]))
                print(f'  Layer {layer_idx}: {len(layer_targets)} types')
                print(f'    {", ".join(layer_targets)}')
            
            # Check for targets appearing in multiple layers
            all_layers = list(targets_by_layer.values())
            if len(all_layers) > 1:
                for i in range(len(all_layers)):
                    for j in range(i+1, len(all_layers)):
                        overlap = all_layers[i] & all_layers[j]
                        if overlap:
                            layer_i = list(targets_by_layer.keys())[i]
                            layer_j = list(targets_by_layer.keys())[j]
                            print(f'\n  Note: {len(overlap)} target(s) found in both Layer {layer_i} and Layer {layer_j}:')
                            print(f'    {", ".join(sorted(list(overlap)))}')
        
        print('='*70 + '\n')
        
        print('💾 Saving path_type data...')
        if self.output_format == 'csv':
             # Save path_type.csv in the parent folder (self.path_folder)
             path_df_type.to_csv(os.path.join(self.path_folder, f'{self.source_fname}_to_{self.target_fname}_path_type.csv'), index=False)
             
             # Save excluded paths in data_details
             csv_folder = os.path.join(self.path_folder, 'data_details')
             os.makedirs(csv_folder, exist_ok=True)
             path_df_type_excluded.to_csv(os.path.join(csv_folder, 'path_type_excluded.csv'), index=False)
             print('   ✓ path_type CSVs saved')
        else:
            with pd.ExcelWriter(output_excel_name, mode='a', engine='openpyxl') as writer:
                path_df_type.to_excel(writer,sheet_name='path_type')
                path_df_type_excluded.to_excel(writer,sheet_name='path_type_excluded')
            print('   ✓ path_type sheets saved')
        
        # get connection path (by bodyId) - OPTIMIZED: Use direct graph pathfinding
        if find_bodyId_path:
            path_df_bodyId = pd.DataFrame()
            print('Analyzing path info by bodyId:')
            print('Building bodyId-level graph and finding paths...')
            
            # Build bodyId-level graph from conn_inpath
            G_bodyId = nx.DiGraph()
            for idx in conn_inpath.index:
                row = conn_inpath.loc[idx]
                bodyId_pre = row['bodyId_pre']
                bodyId_post = row['bodyId_post']
                weight = row['weight']
                # Ensure scalar values (not Series)
                if isinstance(bodyId_pre, pd.Series):
                    bodyId_pre = bodyId_pre.iloc[0]
                if isinstance(bodyId_post, pd.Series):
                    bodyId_post = bodyId_post.iloc[0]
                if isinstance(weight, pd.Series):
                    weight = weight.iloc[0]
                    
                if G_bodyId.has_edge(bodyId_pre, bodyId_post):
                    G_bodyId[bodyId_pre][bodyId_post]['weight'] += weight
                else:
                    G_bodyId.add_edge(bodyId_pre, bodyId_post, weight=weight)
            
            print(f'  BodyId-level graph: {G_bodyId.number_of_nodes()} neurons, {G_bodyId.number_of_edges()} edges')
            
            # Get source and target bodyIds
            source_bodyIds = self.source_df['bodyId'].unique().tolist()
            target_bodyIds = self.target_df.loc[self.target_df.Checked, 'bodyId'].tolist()
            
            # Find paths using DFS on bodyId graph
            bodyId_paths = []
            for source_id in source_bodyIds:
                if source_id not in G_bodyId:
                    continue
                for target_id in target_bodyIds:
                    if nx.has_path(G_bodyId, source_id, target_id):
                        # Find all simple paths with length <= max_interlayer + 1
                        for path in nx.all_simple_paths(G_bodyId, source_id, target_id, cutoff=self.max_interlayer + 1):
                            bodyId_paths.append(path)
            
            print(f'  Found {len(bodyId_paths):,} bodyId-level paths')
            
            # Create type lookup from connection data
            type_lookup = {}
            if 'type_pre' in conn_inpath.columns:
                for _, row in conn_inpath[['bodyId_pre', 'type_pre']].drop_duplicates().iterrows():
                    type_lookup[str(row['bodyId_pre'])] = row['type_pre']
            if 'type_post' in conn_inpath.columns:
                for _, row in conn_inpath[['bodyId_post', 'type_post']].drop_duplicates().iterrows():
                    type_lookup[str(row['bodyId_post'])] = row['type_post']
            
            # Also add source and target info
            for _, row in self.source_df.iterrows():
                type_lookup[str(row['bodyId'])] = row['type']
            for _, row in self.target_df.iterrows():
                type_lookup[str(row['bodyId'])] = row['type']

            # Build DataFrame from bodyId paths (no real_layer_map needed - layer-by-layer ensures forward-only)
            path_df_bodyId = sv.build_path_dataframe_from_paths(
                paths=bodyId_paths,
                conn_data=conn_inpath,
                targets=target_bodyIds,
                real_layer_map=None,
                level='bodyId',
                type_lookup=type_lookup
            )
            
            # Save path_bodyId to the bodyId data file
            print(f'💾 Saving path_bodyId data (rows: {len(path_df_bodyId):,})...')
            if use_csv:
                # Save as CSV if connection data was saved as CSV
                # Save in parent folder with unified naming
                output_path_csv = os.path.join(self.path_folder, self.source_fname+'_to_'+self.target_fname+'_path_bodyId.csv')
                path_df_bodyId.to_csv(output_path_csv, index=False)
                print(f'   ✓ Saved to: {output_path_csv}')
            else:
                # Add to the bodyId Excel file if it was created
                if len(path_df_bodyId) < EXCEL_ROW_LIMIT:
                    with pd.ExcelWriter(output_bodyid_excel, mode='a', engine='openpyxl') as writer:
                        path_df_bodyId.to_excel(writer,sheet_name='path_bodyId')
                    print(f'   ✓ Added path_bodyId sheet to: {output_bodyid_excel}')
                else:
                    print(f'   ⚠️  path_bodyId too large ({len(path_df_bodyId):,} rows), saving as separate CSV')
                    # Save in parent folder with unified naming
                    output_path_csv = os.path.join(self.path_folder, self.source_fname+'_to_'+self.target_fname+'_path_bodyId.csv')
                    path_df_bodyId.to_csv(output_path_csv, index=False)
                    print(f'   ✓ Saved to: {output_path_csv}')
        
        # save interlayer info to excel
        print('💾 Saving interlayer neuron info to Excel...')
        
        # Try to load complete neuron dataset for faster lookup
        dataset_clean = self.dataset.replace(':', '_').replace('.', '_')
        dataset_path = os.path.join(
            self.script_path,
            'datasets',
            f"{dataset_clean}_allneurons_neuron_df.csv"
        )
        
        # Check for subdirectory structure (common for FlyWire/FAFB)
        if not os.path.exists(dataset_path):
            # Try exact match in subdirectory
            dataset_path_subdir = os.path.join(
                self.script_path,
                'datasets',
                dataset_clean,
                f"{dataset_clean}_allneurons_neuron_df.csv"
            )
            if os.path.exists(dataset_path_subdir):
                dataset_path = dataset_path_subdir
            else:
                # Try to find ANY file ending in _allneurons_neuron_df.csv in the subdirectory
                subdir_path = os.path.join(self.script_path, 'datasets', dataset_clean)
                if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                    import glob
                    candidates = glob.glob(os.path.join(subdir_path, "*_allneurons_neuron_df.csv"))
                    if candidates:
                        dataset_path = candidates[0]
                        self._vprint(f"   Found dataset file via glob: {os.path.basename(dataset_path)}", level='full')

        use_local_dataset = os.path.exists(dataset_path)
        if use_local_dataset:
            self._vprint(f'   Using local dataset: {os.path.basename(dataset_path)}', level='full')
            if 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower():
                ndf_complete = pd.read_csv(dataset_path, header=0, index_col=None, dtype={'bodyId': str}, low_memory=False)
            else:
                ndf_complete = pd.read_csv(dataset_path, header=0, index_col=0, low_memory=False)
        else:
            if 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower():
                self._vprint(f'   ⚠️  Local dataset not found for FlyWire/FAFB. Skipping interlayer info fetch (NeuPrint API not supported for this dataset).', level='full')
                ndf_complete = pd.DataFrame()
            else:
                self._vprint(f'   Local dataset not found, will use API calls', level='full')
                # Ensure client is logged in before API calls
                if self.client_hemibrain is None:
                    from neuprint import Client, set_default_client
                    self.client_hemibrain = Client(self.server, self.dataset, self.token)
                    set_default_client(self.client_hemibrain)
        
        interlayers = []
        num_layers = len(neuron_layers[1:])
        for layer_idx, neurons in enumerate(neuron_layers[1:], 1):
            # Filter to only neurons that are actually in connections
            layer_label = f'{layer_idx-1}->{layer_idx}'
            neurons_in_conn = set(
                conn_inpath[conn_inpath['conn_layer'] == layer_label]['bodyId_post'].unique()
            )
            # Also include neurons from next layer if they appear as bodyId_pre
            next_layer_label = f'{layer_idx}->{layer_idx+1}'
            if next_layer_label in conn_inpath['conn_layer'].values:
                neurons_in_conn.update(
                    conn_inpath[conn_inpath['conn_layer'] == next_layer_label]['bodyId_pre'].unique()
                )
            
            # Only fetch neurons that are actually in connections
            neurons_to_fetch = list(set(neurons) & neurons_in_conn)
            print(f'   Fetching layer {layer_idx}/{num_layers} info ({len(neurons_to_fetch)}/{len(neurons)} neurons in connections)...', end='', flush=True)
            
            if len(neurons_to_fetch) == 0:
                # No neurons in this layer are in connections, create empty dataframe
                n_df = pd.DataFrame()
            elif use_local_dataset:
                # Fast: lookup from local CSV
                # Ensure string matching for FlyWire bodyIds
                neurons_to_fetch_str = [str(x) for x in neurons_to_fetch]
                ndf_complete['bodyId'] = ndf_complete['bodyId'].astype(str)
                n_df = ndf_complete[ndf_complete['bodyId'].isin(neurons_to_fetch_str)].copy()
            else:
                # Slow: API call to neuprint (client already logged in above)
                n_df,_ = fetch_neurons(NeuronCriteria(bodyId=neurons_to_fetch))
            
            # Slim down to essential columns only: bodyId, type, instance
            # This significantly reduces file size for large datasets
            essential_cols = ['bodyId', 'type', 'instance']
            available_cols = [c for c in essential_cols if c in n_df.columns]
            if available_cols and len(n_df) > 0:
                n_df = n_df[available_cols].copy()
            
            interlayers.append(n_df)
            print(' ✓')
        
        print('   Writing interlayer sheets to bodyId file...', end='', flush=True)
        if use_csv:
            # Save each layer as CSV in bodyId subfolder
            for i in range(len(interlayers)):
                layer_csv = os.path.join(bodyid_folder, f'layer_{i+1}.csv')
                interlayers[i].to_csv(layer_csv, index=False)
        else:
            # Save to bodyId Excel file
            with pd.ExcelWriter(output_bodyid_excel, mode='a', engine='openpyxl') as writer:
                for i in range(len(interlayers)):
                    interlayers[i].to_excel(writer, sheet_name='layer_'+str(i+1), index=False)
        print(' ✓')
        print('   ✓ Interlayer sheets saved to bodyId file')
        print('Done\n')
        
        # ============================================================================
        # OLD VISUALIZATION CODE - REPLACED BY VisualizePath (see below)
        # ============================================================================
        # Build Sankey diagrams from path data (not from conn_types)
        # This ensures only paths TO TARGETS are shown (no non-target terminals)
        # BLOCKED: Old Sankey/heatmap code replaced by VisualizePath for better consistency
        # See VisualizePath calls below for current visualization approach
        # ============================================================================
        
        # ============================================================================
        # VISUALIZATION: Using VisualizePath only
        # ============================================================================
        
        # VisualizePath network visualization
        print('\nCreating interactive network visualizations...')
        try:
            
            # Create network from path_type if it exists
            if len(path_df_type) > 0:
                paths_to_visualize = path_df_type.copy()
                print(f'  Processing all {len(path_df_type)} paths for visualization')
                
                # Ensure path_block column exists (required by VisualizePath)
                if 'path_block' not in paths_to_visualize.columns:
                    if 'path' in paths_to_visualize.columns:
                        # path is the string representation (A->B)
                        paths_to_visualize['path_block'] = paths_to_visualize['path']
                    elif 'path_str' in paths_to_visualize.columns:
                        # path_str is the list representation
                        paths_to_visualize['path_block'] = paths_to_visualize['path_str'].apply(
                            lambda x: '->'.join(map(str, x)) if isinstance(x, list) else str(x)
                        )
                
                # Ensure column names match what VisualizePath expects
                if 'ratios' in paths_to_visualize.columns and 'connection_ratios' not in paths_to_visualize.columns:
                    paths_to_visualize['connection_ratios'] = paths_to_visualize['ratios']
                if 'probabilities' in paths_to_visualize.columns and 'traversal_probabilities' not in paths_to_visualize.columns:
                    paths_to_visualize['traversal_probabilities'] = paths_to_visualize['probabilities']

                vp = VisualizePath(
                    path_file=paths_to_visualize,
                    output_folder=self.path_folder,
                    source_color=self.source_color if hasattr(self, 'source_color') else '#1f77b4',
                    intermediate_color=self.intermediate_color if hasattr(self, 'intermediate_color') else '#2ca02c',
                    target_color=self.target_color if hasattr(self, 'target_color') else '#d62728',
                    link_color=self.link_color if hasattr(self, 'link_color') else 'rgba(100,100,100,0.3)',
                    network_layout=self.network_layout if hasattr(self, 'network_layout') else 'hierarchical',
                    showfig=self.showfig,
                    edgeN_limit=self.edgeN_limit,
                    output_format=self.output_format,
                    verbose=(self.verbose_mode == 'full')
                )
                vp.visualize()
                self._vprint('  Created network_selected_paths.html and sankey_selected_paths.html')
            else:
                self._vprint('  No paths found to visualize')
            
            # Create network from path_bodyId if it exists and requested
            if find_bodyId_path and len(path_df_bodyId) > 0:
                self._vprint('\nCreating bodyId-level network visualizations...')
                # Filter paths if pathN_to_show is specified
                if self.pathN_to_show > 0 and len(path_df_bodyId) > self.pathN_to_show:
                    paths_to_visualize_bodyId = path_df_bodyId.head(self.pathN_to_show).copy()
                    self._vprint(f'  Showing top {self.pathN_to_show} bodyId paths (by traversal_probability) out of {len(path_df_bodyId)} total paths')
                else:
                    paths_to_visualize_bodyId = path_df_bodyId.copy()
                    self._vprint(f'  Showing all {len(path_df_bodyId)} bodyId paths')
                
                # Ensure path_block column exists
                if 'path_block' not in paths_to_visualize_bodyId.columns:
                    if 'path' in paths_to_visualize_bodyId.columns:
                        paths_to_visualize_bodyId['path_block'] = paths_to_visualize_bodyId['path']
                    elif 'path_str' in paths_to_visualize_bodyId.columns:
                        paths_to_visualize_bodyId['path_block'] = paths_to_visualize_bodyId['path_str'].apply(
                            lambda x: '->'.join(map(str, x)) if isinstance(x, list) else str(x)
                        )
                
                # Ensure column names match what VisualizePath expects
                if 'ratios' in paths_to_visualize_bodyId.columns and 'connection_ratios' not in paths_to_visualize_bodyId.columns:
                    paths_to_visualize_bodyId['connection_ratios'] = paths_to_visualize_bodyId['ratios']
                if 'probabilities' in paths_to_visualize_bodyId.columns and 'traversal_probabilities' not in paths_to_visualize_bodyId.columns:
                    paths_to_visualize_bodyId['traversal_probabilities'] = paths_to_visualize_bodyId['probabilities']
                
                # Add types to bodyIds in path_block for better visualization
                if 'type_lookup' in locals():
                    def add_types_to_path(path_str):
                        if not isinstance(path_str, str): return str(path_str)
                        nodes = path_str.split('->')
                        new_nodes = []
                        for node in nodes:
                            node = node.strip()
                            if node in type_lookup:
                                new_nodes.append(f"{node}_{type_lookup[node]}")
                            else:
                                new_nodes.append(node)
                        return '->'.join(new_nodes)
                    
                    paths_to_visualize_bodyId['path_block'] = paths_to_visualize_bodyId['path_block'].apply(add_types_to_path)

                vp_bodyId = VisualizePath(
                    path_file=paths_to_visualize_bodyId,
                    output_folder=os.path.join(self.path_folder, 'bodyId_visualization'),
                    source_color=self.source_color if hasattr(self, 'source_color') else '#1f77b4',
                    intermediate_color=self.intermediate_color if hasattr(self, 'intermediate_color') else '#2ca02c',
                    target_color=self.target_color if hasattr(self, 'target_color') else '#d62728',
                    link_color=self.link_color if hasattr(self, 'link_color') else 'rgba(100,100,100,0.3)',
                    network_layout=self.network_layout if hasattr(self, 'network_layout') else 'hierarchical',
                    showfig=self.showfig,
                    edgeN_limit=self.edgeN_limit,
                    output_format=self.output_format,
                    verbose=(self.verbose_mode == 'full')
                )
                vp_bodyId.visualize()
                self._vprint('  Created bodyId-level visualizations in bodyId_visualization subfolder')

        except Exception as e:
            self._vprint(f'  Warning: VisualizePath visualization failed: {e}')
            import traceback
            traceback.print_exc()
        

        
        self._vprint('Done\n')
    
    def _create_interactive_network_for_path(self, conn_types, conn_inpath, neuron_layers, target_type, target_ID, output_folder):
        '''Create interactive network visualizations for FindPath method'''
        
        # Network by type
        self._vprint('Building interactive network by type...', level='full')
        G_type = nx.DiGraph()
        
        # Add nodes with layer information
        for layer_idx, layer in enumerate(neuron_layers):
            if layer_idx == 0:
                # Source neurons
                source_types = conn_types[conn_types['conn_layer'] == '0->1']['type_pre'].unique()
                for node_type in source_types:
                    G_type.add_node(node_type, layer=layer_idx, node_type='source')
            else:
                # Get types in this layer
                layer_conn = conn_types[conn_types['conn_layer'] == f'{layer_idx-1}->{layer_idx}']
                for node_type in layer_conn['type_post'].unique():
                    is_target = node_type in target_type
                    G_type.add_node(node_type, layer=layer_idx, 
                                   node_type='target' if is_target else 'intermediate')
        
        # Add edges
        for idx in conn_types.index:
            source = conn_types.at[idx, 'type_pre']
            target = conn_types.at[idx, 'type_post']
            weight = conn_types.at[idx, 'weight']
            prob = conn_types.at[idx, 'traversal_probability']
            G_type.add_edge(source, target, weight=weight, probability=prob)
        
        # Create layout based on network_layout parameter
        print(f'Using "{self.network_layout}" layout...')
        pos_type = self._get_network_layout(G_type)
        
        # Create plotly figure for type network
        self._plot_interactive_network(
            G_type, pos_type, 
            title=f'Interactive Network: {self.source_fname} to {self.target_fname} (by type)',
            filename=os.path.join(output_folder, f'Network_type_path.html'),
            color_by='node_type'
        )
        
        # Network by bodyId (only if network is not too large)
        if len(conn_inpath) < 5000:  # Limit for performance
            print('Building interactive network by bodyId...')
            G_bodyId = nx.DiGraph()
            
            # Add nodes with layer information
            for layer_idx, layer in enumerate(neuron_layers):
                for bodyId in layer:
                    is_target = bodyId in target_ID
                    is_source = layer_idx == 0
                    if is_target:
                        node_cat = 'target'
                    elif is_source:
                        node_cat = 'source'
                    else:
                        node_cat = 'intermediate'
                    G_bodyId.add_node(bodyId, layer=layer_idx, node_type=node_cat)
            
            # Add edges
            for idx in conn_inpath.index:
                source = conn_inpath.at[idx, 'bodyId_pre']
                target = conn_inpath.at[idx, 'bodyId_post']
                weight = conn_inpath.at[idx, 'weight']
                prob = conn_inpath.at[idx, 'traversal_probability']
                G_bodyId.add_edge(source, target, weight=weight, probability=prob)
            
            # Create layout based on network_layout parameter
            pos_bodyId = self._get_network_layout(G_bodyId)
            
            # Fetch neuron info for labels (use local dataset if available)
            all_bodyIds = list(G_bodyId.nodes())
            node_info_df = self._fetch_neurons_local_or_api(all_bodyIds, columns=['bodyId', 'type'])
            node_labels = {}
            for idx in node_info_df.index:
                bodyId = node_info_df.at[idx, 'bodyId']
                neuron_type = node_info_df.at[idx, 'type'] if node_info_df.at[idx, 'type'] else 'None'
                node_labels[bodyId] = f"{neuron_type}_{bodyId}"
            
            # Create plotly figure for bodyId network
            self._plot_interactive_network(
                G_bodyId, pos_bodyId,
                title=f'Interactive Network: {self.source_fname} to {self.target_fname} (by bodyId)',
                filename=os.path.join(output_folder, f'Network_bodyId_path.html'),
                color_by='node_type',
                node_labels=node_labels
            )
        else:
            print(f'Skipping bodyId network (too large: {len(conn_inpath)} connections)')
    
    @staticmethod
    def _find_paths_dfs_optimized(args):
        '''
        Helper function for parallel pathfinding using optimized DFS with backtracking.
        
        This function explores all paths from a set of source neurons to all target neurons
        in a single DFS traversal per source. This avoids redundant edge exploration when
        paths share common segments (e.g., A→B→C→T and A→B→D→T both explore A→B only once).
        
        Parameters:
        -----------
        args : tuple
            (sources, targets_set, G_edges, cutoff, layer_neurons_list)
            - sources: list of source neuron IDs to explore from
            - targets_set: set of target neuron IDs to find
            - G_edges: list of (u, v, weight) tuples representing graph edges
            - cutoff: maximum path length (number of edges)
            - layer_neurons_list: list of sets for layer membership
        
        Returns:
        --------
        tuple: (neurons_set, edges_set, edges_with_layer_set, path_count, pairs_with_paths, total_pairs_checked, paths_found)
               paths_found is list of paths (each path is list of node IDs)
        '''
        import networkx as nx
        
        sources, targets_set, G_edges, cutoff, layer_neurons_list = args
        
        # Reconstruct graph from edges (graphs can't be pickled easily)
        G = nx.DiGraph()
        for u, v, weight in G_edges:
            G.add_edge(u, v, weight=weight)
        
        # Convert layer_neurons_list back to list of sets
        layer_neurons = [set(layer) for layer in layer_neurons_list]
        
        # Accumulate results across all sources
        neurons_in_paths = set()
        edges_in_paths = set()
        edges_in_paths_with_layer = set()
        path_count = 0
        pairs_with_paths_dict = {}  # Track (source, target) pairs that have paths
        paths_found = []  # Store actual paths
        
        def dfs_find_all_paths(current, target_set, path, visited):
            '''
            DFS with backtracking to find all paths from current node to any target.
            
            Parameters:
            -----------
            current : node
                Current node in the traversal
            target_set : set
                Set of target nodes to find
            path : list
                Current path being explored
            visited : set
                Nodes in current path (to prevent cycles)
            '''
            nonlocal path_count, neurons_in_paths, edges_in_paths, edges_in_paths_with_layer, paths_found
            
            # Check if current node is a target
            if current in target_set:
                # Found a complete path to a target
                path_count += 1
                neurons_in_paths.update(path)
                
                # Record this source-target pair
                source_node = path[0]
                pairs_with_paths_dict[(source_node, current)] = True
                
                # Store the complete path
                paths_found.append(list(path))
                
                # Add edges from this path
                for i in range(len(path) - 1):
                    pre_node = path[i]
                    post_node = path[i+1]
                    edges_in_paths.add((pre_node, post_node))
                    
                    # Edge layer is determined by position in path (path starts at layer 0)
                    # i=0 means layer 0->1, i=1 means layer 1->2, etc.
                    edge_layer = i
                    edges_in_paths_with_layer.add((edge_layer, pre_node, post_node))
                
                # Continue searching for more paths through this target
                # (in case this target is also an intermediate node to other targets)
            
            # Stop if we've reached maximum depth
            if len(path) - 1 >= cutoff:
                return
            
            # Explore neighbors
            if current in G:
                for neighbor in G.neighbors(current):
                    # Skip if already in current path (prevent cycles)
                    if neighbor not in visited:
                        # Add neighbor to path and continue DFS
                        path.append(neighbor)
                        visited.add(neighbor)
                        
                        dfs_find_all_paths(neighbor, target_set, path, visited)
                        
                        # Backtrack: remove neighbor from path
                        path.pop()
                        visited.remove(neighbor)
        
        # Explore from each source neuron
        for source in sources:
            if source in G:  # Make sure source exists in graph
                initial_path = [source]
                initial_visited = {source}
                dfs_find_all_paths(source, targets_set, initial_path, initial_visited)
        
        pairs_with_paths = len(pairs_with_paths_dict)
        total_pairs_checked = len(sources) * len(targets_set)
        
        return (neurons_in_paths, edges_in_paths, edges_in_paths_with_layer, 
                path_count, pairs_with_paths, total_pairs_checked, paths_found)
    
    def FindAllPath(self, find_bodyId_path=True, forward_only=True, exclude_searched_neurons=None):
        '''
        Find all paths between source and target neurons within max_interlayer.
        
        Parameters:
        -----------
        find_bodyId_path : bool, default=True
            Whether to find paths at bodyId level
        forward_only : bool, default=True
            If True: Query each neuron only once per layer (RECOMMENDED - more efficient)
            If False: Re-query all discovered neurons at each layer (slower but comprehensive)
            
            IMPORTANT: Both modes fetch ALL connections including recurrent/reciprocal ones.
            The difference is query efficiency, NOT the connections found:
            - True: Queries each neuron once → faster, less redundant
            - False: Re-queries neurons → slower, but ensures no connections missed due to filtering
            
            For most use cases, True is recommended (4-14x faster with same results).
        exclude_searched_neurons : bool, deprecated
            Deprecated parameter name. Use forward_only instead.
            If provided, it will override forward_only for backward compatibility.
        
        Logic:
        1. Fetch connections layer by layer, discovering network structure
        2. Identify which target neurons exist in the searched network
        3. Find all paths from sources to targets with path length ≤ max_interlayer
        '''
        # Reset status columns if they exist (to allow sequential calls)
        self._reset_temp_columns()
        
        # Check if source or target dataframes are empty
        if self.source_df.empty:
            self._vprint("Error: Source neuron DataFrame is empty. Cannot find paths.", level='always')
            return
        if self.target_df.empty:
            self._vprint("Error: Target neuron DataFrame is empty. Cannot find paths.", level='always')
            return
        
        # Handle deprecated parameter
        if exclude_searched_neurons is not None:
            forward_only = exclude_searched_neurons
            self._vprint('⚠️  Warning: exclude_searched_neurons is deprecated. Use forward_only instead.', level='always')
            self._vprint(f'   Setting forward_only={forward_only}', level='always')
        
        # Helper function to format decimal numbers for folder names
        def format_decimal(val):
            """Format decimal number for folder name, replacing '.' with '_'"""
            if val == int(val):
                return str(int(val))
            else:
                formatted = f"{val:.6f}".rstrip('0').rstrip('.')
                return formatted.replace('.', '_')
        
        # Create allpaths folder with parameter suffix
        import datetime
        param_suffix = f"_L{self.max_interlayer}w{self.min_synapse_num}"
        param_suffix += f"r{format_decimal(self.min_ratio)}"
        param_suffix += f"p{format_decimal(self.min_traversal_probability)}"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        param_suffix += f"_{timestamp}"
        
        if self.saveas:
            # If saveas is set, use save_folder directly
            self.allpath_folder = self.save_folder
        else:
            # Otherwise create subfolder with parameters
            self.allpath_folder = os.path.join(self.save_folder, f'allpaths{param_suffix}')
            
        if not os.path.exists(self.allpath_folder): 
            os.makedirs(self.allpath_folder, exist_ok=True)
            self._vprint(f'  📁 Created output folder: {self.allpath_folder}', level='full')
        
        # Save all attributes and parameters to the allpaths folder
        with open(os.path.join(self.allpath_folder, 'all_attributes.json'), 'w') as f:
            json.dump(self.__dict__, f, indent=4, default=lambda o: '<not serializable>')
        
        with open(os.path.join(self.allpath_folder, 'parameters.txt'), 'w') as f:
            f.write(f'Parameters for processing {self.source_fname} to {self.target_fname}:\n')
            for key, value in self.parameter_dict.items():
                keylen = len(key)
                f.write(f'{key}:{" "*(30-keylen)}{value}\n')
            f.write('\n')
        
        # Ensure bodyIds are strings for consistent processing
        self.source_df['bodyId'] = self.source_df['bodyId'].astype(str)
        self.target_df['bodyId'] = self.target_df['bodyId'].astype(str)
        
        source_ID = self.source_df['bodyId'].unique()
        target_ID = self.target_df['bodyId'].unique()
        target_type = self.target_df['type'].unique()
        
        # PHASE 1: Fetch all connections in the network up to max_interlayer layers
        if self.verbose_mode == 'simple':
            self._vprint(f'\nPhase 1:', level='simple')
        elif self.verbose_mode == 'full':
            self._vprint(f'\n=== PHASE 1: Fetching all network layers (0 to {self.max_interlayer + 1}) ===', level='full')
            if forward_only:
                self._vprint('Mode: Layer-by-layer querying (query each neuron once - RECOMMENDED)', level='full')
                self._vprint('Note: Still fetches ALL connections including recurrent/reciprocal ones', level='full')
            else:
                self._vprint('Mode: Comprehensive re-querying (re-query all neurons at each layer)', level='full')
                self._vprint('Note: Slower but ensures no connections missed due to filtering', level='full')
            self._vprint('', level='full')
        
        all_neurons_in_network = set(source_ID)
        layer_neurons = [set(source_ID)]  # Layer 0: sources
        all_connections = []
        
        for layer_idx in range(self.max_interlayer + 1):
            # Determine which neurons to fetch based on mode
            if forward_only:
                # Only fetch from current layer's neurons (faster, each neuron queried once)
                neurons_to_fetch = list(layer_neurons[layer_idx])
            else:
                # Fetch from ALL neurons discovered so far (slower, comprehensive)
                neurons_to_fetch = list(all_neurons_in_network)
            
            if len(neurons_to_fetch) == 0:
                self._vprint(f'Layer {layer_idx} is empty, stopping.', level='full')
                break
            
            # Fetch connections (fetch with weight≥1, filter by all criteria together later)
            if self.verbose_mode == 'simple':
                self._vprint(f'layer {layer_idx}->{layer_idx+1}: processing...', level='simple', end='', flush=True)
            elif self.verbose_mode == 'full':
                self._vprint(f'Layer {layer_idx}->{layer_idx+1}:', level='full')
            conn_df = self._fetch_connections_with_cache(
                upstream_bodyIds=neurons_to_fetch,
                downstream_bodyIds=None,
                min_weight=self.min_synapse_num,
                min_conn_ratio=self.min_ratio,
                min_traversal_prob=self.min_traversal_probability
            )
            
            # Ensure connection dataframe has string bodyIds
            if not conn_df.empty:
                conn_df['bodyId_pre'] = conn_df['bodyId_pre'].astype(str)
                conn_df['bodyId_post'] = conn_df['bodyId_post'].astype(str)
            
            # Store connections for pathfinding
            conn_df.insert(loc=0, column='conn_layer', value=f'{layer_idx}->{layer_idx+1}')
            all_connections.append(conn_df)
            
            # Collect all downstream neurons for next layer
            post_neurons = set(conn_df['bodyId_post'].unique())
            
            # Calculate newly discovered neurons
            next_layer = post_neurons - all_neurons_in_network
            all_neurons_in_network.update(next_layer)
            
            # Add this layer to layer_neurons for target identification
            # (even if we won't fetch from it in the next iteration)
            layer_neurons.append(next_layer)
            
            if self.verbose_mode == 'simple':
                self._vprint('Done', level='simple')
            elif self.verbose_mode == 'full':
                if forward_only:
                    self._vprint(f'Layer {layer_idx}->{layer_idx+1}: {len(post_neurons)} downstream neurons, {len(next_layer)} new, {len(conn_df)} connections', level='full')
                else:
                    self._vprint(f'Layer {layer_idx}->{layer_idx+1}: {len(post_neurons)} total downstream, {len(next_layer)} new neurons, {len(conn_df)} connections', level='full')
        
        self._vprint(f'\nTotal neurons in network: {len(all_neurons_in_network)}', level='full')
        self._vprint(f'Total layers fetched: {len(layer_neurons)}', level='full')
        
        # PHASE 2: Identify which targets exist in the searched network
        if self.verbose_mode == 'simple':
            self._vprint(f'Phase 2:', level='simple')
            self._vprint(f'identifying targets...', level='simple', end='', flush=True)
        elif self.verbose_mode == 'full':
            self._vprint(f'\n=== PHASE 2: Identifying targets in the network ===', level='full')
        
        self.target_df.insert(loc=0, column='Checked', value=False)
        self.target_df.insert(loc=1, column='Layer', value=-1)
        
        # Check which targets are in the network
        targets_found = []
        for idx in self.target_df.index:
            target_bodyId = self.target_df.at[idx, 'bodyId']
            if target_bodyId in all_neurons_in_network:
                self.target_df.at[idx, 'Checked'] = True
                targets_found.append(target_bodyId)
                # Find which layer this target first appears in
                for layer_idx, layer_set in enumerate(layer_neurons):
                    if target_bodyId in layer_set:
                        self.target_df.at[idx, 'Layer'] = layer_idx
                        break
        
        targetNum = len(self.target_df)
        targetNum_checked = len(targets_found)
        
        if self.verbose_mode == 'simple':
            self._vprint('Done', level='simple')
        elif self.verbose_mode == 'full':
            self._vprint(f'Targets found in network: {targetNum_checked} / {targetNum}', level='full')
        
        if targetNum_checked == 0:
            self._vprint('\033[33mNo target neurons found in the searched network. Cannot construct paths.\033[0m', level='always')
            return
        
        # Print target distribution by layer (same target can appear in multiple layers)
        if self.verbose_mode == 'full':
            print('\nTarget distribution by layer:')
            total_target_occurrences = 0
            for layer_idx in sorted(self.target_df[self.target_df['Checked']]['Layer'].unique()):
                targets_in_layer = self.target_df[
                    (self.target_df['Layer'] == layer_idx) & (self.target_df['Checked'])
                ]
                count = len(targets_in_layer)
                total_target_occurrences += count
                
                # Show target identifiers (bodyId or type depending on filter_by)
                if self.filter_by == 'bodyId':
                    target_list = targets_in_layer['bodyId'].tolist()
                else:
                    # Check if type column exists and has valid values
                    if 'type' in targets_in_layer.columns and targets_in_layer['type'].notna().any():
                        target_list = targets_in_layer['type'].tolist()
                    else:
                        target_list = targets_in_layer['bodyId'].tolist()
                
                # Display targets (limit to first 10 per layer for readability)
                if count <= 10:
                    print(f'  Layer {layer_idx}: {count} targets - {target_list}')
                else:
                    print(f'  Layer {layer_idx}: {count} targets - {target_list[:10]} ... (+{count-10} more)')
            
            # Show if targets appear in multiple layers
            if total_target_occurrences > targetNum_checked:
                print(f'\nNote: {total_target_occurrences} total target occurrences across layers ({targetNum_checked} unique targets)')
                print(f'      Some targets appear in multiple layers')
        
        # PHASE 3: Extract all paths from sources to targets (path length ≤ max_interlayer)
        if self.verbose_mode == 'simple':
            self._vprint(f'Phase 3:', level='simple')
        elif self.verbose_mode == 'full':
            self._vprint(f'\n=== PHASE 3: Finding all paths from sources to targets ===', level='full')
            self._vprint('Using graph-based pathfinding to handle reciprocal connections...', level='full')
        
        # Create INITIAL real layer mapping (neuron ID -> discovery layer)
        # Targets will be updated later based on their actual appearance in paths
        real_layer_map_bodyId = {}
        for layer_idx, layer_set in enumerate(layer_neurons):
            for neuron_id in layer_set:
                # Use earliest layer if neuron appears in multiple layers
                if neuron_id not in real_layer_map_bodyId:
                    real_layer_map_bodyId[neuron_id] = layer_idx
        
        self._vprint(f'Created initial real layer map for {len(real_layer_map_bodyId)} neurons', level='full')
        self._vprint(f'  Note: Target real layers will be updated after pathfinding completes', level='full')
        
        # Build a directed graph from all connections
        self._vprint('Building connection graph...', level='full', end=' ')
        G = nx.DiGraph()
        for conn_df in all_connections:
            for idx in conn_df.index:
                pre = conn_df.at[idx, 'bodyId_pre']
                post = conn_df.at[idx, 'bodyId_post']
                weight = conn_df.at[idx, 'weight']
                # Add edge (can have multiple edges between same nodes in original data)
                if G.has_edge(pre, post):
                    G[pre][post]['weight'] += weight
                else:
                    G.add_edge(pre, post, weight=weight)
        self._vprint(f'Done! ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)', level='full')
        
        # Pruning: Remove nodes that cannot reach any target
        if G.number_of_nodes() > 0 and len(targets_found) > 0:
            self._vprint('Pruning graph to remove dead ends...', level='full', end=' ')
            # Only start BFS from targets that are actually in the graph
            valid_targets = [t for t in targets_found if t in G]
            
            if valid_targets:
                # Use BFS to find all ancestors (nodes that can reach targets)
                # This is much faster than checking descendants for every node
                nodes_that_can_reach_targets = set(valid_targets)
                
                # nx.ancestors returns all nodes having a path to target
                # For multiple targets, we can do a single BFS on the reversed graph
                R = G.reverse(copy=False)
                
                # Perform BFS from all targets simultaneously
                # This finds all nodes that can reach ANY target
                reachable = set()
                # Initialize queue with targets
                queue = list(valid_targets)
                visited = set(valid_targets)
                
                while queue:
                    node = queue.pop(0)
                    reachable.add(node)
                    
                    for neighbor in R.neighbors(node):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                nodes_that_can_reach_targets = reachable
                
                # Intersect with nodes reachable from sources
                # Since G is built layer-by-layer from sources, most nodes are reachable.
                # But let's be safe and precise.
                # Actually, we can just restrict G to nodes_that_can_reach_targets
                # because any node NOT in this set is a dead end w.r.t targets.
                
                original_node_count = G.number_of_nodes()
                G = G.subgraph(nodes_that_can_reach_targets).copy()
                self._vprint(f'Done! ({original_node_count} -> {G.number_of_nodes()} nodes)', level='full')
            else:
                self._vprint('Warning: No targets found in graph (should have been caught earlier).', level='full')
        
        # Find all neurons that are on ANY path from any source to any target
        # with path length ≤ max_interlayer
        neurons_in_paths = set()
        edges_in_paths = set()  # Stores (pre, post) pairs
        edges_in_paths_with_layer = set()  # Stores (layer_idx, pre, post) to track layer-specific edges
        
        self._vprint(f'\nSearching paths: {len(source_ID)} sources × {len(targets_found)} targets = {len(source_ID) * len(targets_found)} pairs', level='full')
        self._vprint(f'Maximum path length: {self.max_interlayer + 1} edges', level='full')
        self._vprint(f'Using optimized DFS algorithm (explores shared path segments only once)', level='full')
        
        # Decide whether to use parallel processing
        total_pairs = len(source_ID) * len(targets_found)
        use_parallel = self.use_parallel and len(source_ID) > 4  # Parallelize if >4 sources
        
        if use_parallel:
            import multiprocessing as mp
            import os as os_module
            
            # Determine number of processes
            if self.n_jobs == -1:
                n_processes = mp.cpu_count()
            elif self.n_jobs == 1:
                use_parallel = False  # Fall back to sequential
                n_processes = 1
            else:
                n_processes = min(self.n_jobs, mp.cpu_count())
            
            if use_parallel:
                if self.verbose_mode == 'simple':
                    self._vprint(f'pathfinding[parallel]...', level='simple', end='', flush=True)
                elif self.verbose_mode == 'full':
                    self._vprint(f'Using parallel processing with {n_processes} processes...', level='full')
                
                # Prepare graph edges for pickling
                G_edges = [(u, v, data['weight']) for u, v, data in G.edges(data=True)]
                
                # Prepare layer neurons as list of lists (sets can't be pickled easily)
                layer_neurons_list = [list(layer) for layer in layer_neurons]
                
                # Convert targets to set for efficient lookup
                targets_set = set(targets_found)
                
                # Split sources into chunks for parallel processing (not pairs!)
                # Each chunk will explore all targets from its source neurons
                sources_list = list(source_ID)
                
                # Distribute sources across processes
                # Use fewer sources per chunk for better load balancing
                if len(sources_list) > 20:
                    target_chunk_size = max(1, len(sources_list) // (n_processes * 4))
                else:
                    target_chunk_size = max(1, len(sources_list) // n_processes)
                
                chunk_size = target_chunk_size
                source_chunks = [sources_list[i:i + chunk_size] for i in range(0, len(sources_list), chunk_size)]
                
                self._vprint(f'Split into {len(source_chunks)} chunks (~{chunk_size} sources per chunk)', level='full')
                self._vprint(f'Each chunk will explore paths to all {len(targets_set)} targets', level='full')
                
                # More realistic time estimate based on graph complexity
                # With DFS optimization, each source is explored once (not once per target)
                # Factors affecting speed:
                # - Graph size (nodes and edges)
                # - Path length (cutoff)
                # - Graph density (average degree)
                
                # Base estimate on graph complexity
                avg_degree = G.number_of_edges() / G.number_of_nodes() if G.number_of_nodes() > 0 else 1
                path_complexity = self.max_interlayer + 1  # Maximum path length
                
                # Time estimation based on empirical measurements from hemibrain connectome
                # Calibrated with: 142K nodes, avg_degree=36.8, depth=3 → 12 sec/source with 12 workers
                # Observed: 0.0069 sources/sec/process for very dense graphs
                
                # Base speed estimates (sources/sec per process) - calibrated to real performance
                if avg_degree < 3:
                    base_speed = 50   # Very sparse: trivial pathfinding
                elif avg_degree < 8:
                    base_speed = 10   # Sparse: fast pathfinding
                elif avg_degree < 15:
                    base_speed = 2    # Medium: moderate complexity
                elif avg_degree < 25:
                    base_speed = 0.3  # Dense: significant path explosion
                elif avg_degree < 40:
                    base_speed = 0.05 # Very dense: severe path explosion
                else:
                    base_speed = 0.01 # Extremely dense: exponential explosion
                
                # Depth penalty - each layer multiplies search space
                # Empirically: depth=3, degree=37 → penalty ~5x from depth=2
                if path_complexity <= 2:
                    depth_factor = 1.0
                elif path_complexity == 3:
                    depth_factor = (avg_degree / 10) ** 1.2  # Calibrated: 36.8/10^1.2 = 5.1
                else:
                    depth_factor = (avg_degree / 10) ** (path_complexity - 2)
                
                adjusted_speed = base_speed / max(1, depth_factor)
                
                # Large graph overhead (memory, cache misses)
                if G.number_of_nodes() > 100000:
                    # Calibrated: 142K nodes → 1.4x penalty
                    size_factor = 1 + ((G.number_of_nodes() - 100000) / 100000) * 0.4
                    adjusted_speed = adjusted_speed / size_factor
                
                # Ensure minimum speed (avoid infinity)
                adjusted_speed = max(0.0001, adjusted_speed)
                
                # Total with parallelization
                total_estimated_speed = adjusted_speed * n_processes
                estimated_time = len(sources_list) / total_estimated_speed if total_estimated_speed > 0 else 0
                
                # Overhead: startup (excluded), load imbalance, synchronization
                estimated_time *= 1.3
                
                if estimated_time < 10:
                    time_str = f"~{estimated_time:.0f} seconds"
                elif estimated_time < 120:
                    time_str = f"~{estimated_time/60:.1f} minutes"
                else:
                    time_str = f"~{estimated_time/60:.0f} minutes"
                
                self._vprint(f'Estimated time: {time_str} (graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, avg degree: {avg_degree:.1f})', level='full')
                self._vprint(f'Processing...\n', level='full')
                
                # Prepare arguments for each process
                args_list = [
                    (chunk, targets_set, G_edges, self.max_interlayer + 1, layer_neurons_list)
                    for chunk in source_chunks
                ]
                
                # Progress tracking
                import time
                start_time = time.time()
                last_update = start_time
                update_interval = 0.5  # Update every 0.5 seconds for better feedback
                
                # Run parallel pathfinding with progress tracking
                path_count = 0
                pairs_with_paths = 0
                chunks_completed = 0
                sources_processed = 0
                all_paths = []  # Collect paths from all workers
                
                # For simple observed-speed ETA calculation
                first_chunk_time = None  # Track when first chunk completes (exclude startup overhead)
                productive_start_time = None  # Start time for actual work (after first chunk)
                
                self._vprint(f'⏳ Starting {n_processes} worker processes...', level='full')
                self._vprint(f'   (First update will appear when a chunk completes)', level='full')
                self._vprint('', level='full')
                
                with mp.Pool(processes=n_processes) as pool:
                    # Use imap_unordered for progress tracking (returns results as they complete)
                    for neurons_set, edges_set, edges_layer_set, p_count, p_with_paths, chunk_size_actual, paths_chunk in pool.imap_unordered(
                        self._find_paths_dfs_optimized, args_list
                    ):
                        # Update totals
                        neurons_in_paths.update(neurons_set)
                        edges_in_paths.update(edges_set)
                        edges_in_paths_with_layer.update(edges_layer_set)
                        path_count += p_count
                        pairs_with_paths += p_with_paths
                        all_paths.extend(paths_chunk)  # Collect paths from this worker
                        chunks_completed += 1
                        sources_processed += len(source_chunks[chunks_completed - 1])  # Actual sources in this chunk
                        
                        # Progress update - show every chunk for better feedback
                        current_time = time.time()
                        
                        # Record first chunk completion time to exclude startup overhead
                        if first_chunk_time is None:
                            first_chunk_time = current_time
                            startup_overhead = first_chunk_time - start_time
                            # Start productive time tracking AFTER first chunk
                            productive_start_time = first_chunk_time
                            self._vprint(f'   ⚡ Workers initialized in {startup_overhead:.1f}s, starting main processing...\n', level='full')
                        
                        # Calculate current speed using ONLY productive time (excludes startup)
                        productive_elapsed = current_time - productive_start_time if productive_start_time else 0.1
                        current_speed = sources_processed / productive_elapsed if productive_elapsed > 0 else 0
                        
                        progress_pct = (sources_processed / len(sources_list)) * 100
                        remaining_sources = len(sources_list) - sources_processed
                        
                        # Simple reliable ETA: remaining / observed_speed
                        # Wait for minimum samples before showing ETA
                        min_samples_for_eta = max(3, int(len(sources_list) * 0.05))  # At least 3 sources or 5%
                        
                        if sources_processed >= min_samples_for_eta and current_speed > 0:
                            eta_seconds = remaining_sources / current_speed
                            
                            # Format ETA in HH:mm:ss
                            hours = int(eta_seconds // 3600)
                            minutes = int((eta_seconds % 3600) // 60)
                            seconds = int(eta_seconds % 60)
                            eta_str = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
                        else:
                            eta_str = 'calculating...'
                        
                        # Update more frequently - show every chunk or every 0.5 seconds
                        should_update = (current_time - last_update >= update_interval or 
                                       chunks_completed == 1 or  # Always show first chunk
                                       chunks_completed % 5 == 0 or  # Show every 5 chunks
                                       chunks_completed == len(source_chunks))  # Always show completion
                        
                        if should_update and self.verbose_mode == 'full':
                            # Use \033[K to clear to end of line (removes residual characters)
                            self._vprint(f'\r   Progress: {sources_processed}/{len(sources_list)} sources ({progress_pct:.1f}%) | ETA: {eta_str}\033[K', level='full', end='', flush=True)
                            last_update = current_time
                
                # Final newline
                if self.verbose_mode == 'full':
                    self._vprint('', level='full')
                
                elapsed = time.time() - start_time
                if self.verbose_mode == 'simple':
                    self._vprint('Done', level='simple')
                    self._vprint('building paths...', level='simple', end='', flush=True)
                elif self.verbose_mode == 'full':
                    self._vprint(f'\n✅ Parallel pathfinding complete in {elapsed:.1f}s!', level='full')
                    self._vprint(f'   Average: {len(sources_list)/elapsed:.1f} sources/s (explored {len(targets_set)} targets per source)', level='full')
                    self._vprint(f'   Processed by {n_processes} workers across {len(source_chunks)} chunks', level='full')
                    self._vprint(f'   📦 Collected {len(all_paths):,} paths in memory (~{len(all_paths) * 50 / 1024 / 1024:.1f} MB)', level='full')
        
        if not use_parallel:
            if self.verbose_mode == 'simple':
                self._vprint(f'pathfinding[sequential]...', level='simple', end='', flush=True)
            elif self.verbose_mode == 'full':
                self._vprint('Using sequential processing (optimized DFS)...', level='full')
                self._vprint('This may take a while for large datasets...\n', level='full')
            
            path_count = 0
            sources_processed = 0
            pairs_with_paths_dict = {}
            all_paths = []  # Initialize empty list for sequential mode (not collected in sequential)
            
            # Progress tracking for simple observed-speed ETA
            import time
            start_time = time.time()
            last_update = start_time
            update_interval = 2.0  # Update every 2 seconds
            
            targets_set = set(targets_found)
            
            def dfs_find_all_paths(current, target_set, path, visited):
                '''DFS with backtracking to find all paths from current node to any target.'''
                nonlocal path_count, neurons_in_paths, edges_in_paths, edges_in_paths_with_layer
                
                # Check if current node is a target
                if current in target_set:
                    # Found a complete path to a target
                    path_count += 1
                    neurons_in_paths.update(path)
                    
                    # Record this source-target pair
                    source_node = path[0]
                    pairs_with_paths_dict[(source_node, current)] = True
                    
                    # Store the complete path
                    all_paths.append(list(path))
                    
                    # Add edges from this path
                    for i in range(len(path) - 1):
                        pre_node = path[i]
                        post_node = path[i+1]
                        edges_in_paths.add((pre_node, post_node))
                        
                        # Edge layer is determined by position in path (path starts at layer 0)
                        # i=0 means layer 0->1, i=1 means layer 1->2, etc.
                        edge_layer = i
                        edges_in_paths_with_layer.add((edge_layer, pre_node, post_node))
                
                # Stop if we've reached maximum depth
                if len(path) - 1 >= self.max_interlayer + 1:
                    return
                
                # Explore neighbors
                if current in G:
                    for neighbor in G.neighbors(current):
                        # Skip if already in current path (prevent cycles)
                        if neighbor not in visited:
                            # Add neighbor to path and continue DFS
                            path.append(neighbor)
                            visited.add(neighbor)
                            
                            dfs_find_all_paths(neighbor, target_set, path, visited)
                            
                            # Backtrack: remove neighbor from path
                            path.pop()
                            visited.remove(neighbor)
            
            # Explore from each source neuron
            for source_idx, source in enumerate(source_ID):
                sources_processed += 1
                
                if source in G:  # Make sure source exists in graph
                    initial_path = [source]
                    initial_visited = {source}
                    dfs_find_all_paths(source, targets_set, initial_path, initial_visited)
                
                # Progress update every 2 seconds with dynamic ETA
                current_time = time.time()
                if current_time - last_update >= update_interval:
                    elapsed = current_time - start_time
                    
                    # Simple reliable ETA: remaining / observed_speed
                    current_speed = sources_processed / elapsed if elapsed > 0 else 0
                    
                    # Wait for minimum samples before showing ETA
                    min_samples_for_eta = max(3, int(len(source_ID) * 0.05))  # At least 3 sources or 5%
                    
                    progress_pct = (sources_processed / len(source_ID)) * 100
                    remaining_sources = len(source_ID) - sources_processed
                    
                    if sources_processed >= min_samples_for_eta and current_speed > 0:
                        eta_seconds = remaining_sources / current_speed
                        
                        # Format ETA in HH:mm:ss
                        hours = int(eta_seconds // 3600)
                        minutes = int((eta_seconds % 3600) // 60)
                        seconds = int(eta_seconds % 60)
                        eta_str = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
                    else:
                        eta_str = 'calculating...'
                    
                    pairs_with_paths = len(pairs_with_paths_dict)
                    
                    if self.verbose_mode == 'full':
                        # Use \033[K to clear to end of line
                        self._vprint(f'\r   Progress: {sources_processed}/{len(source_ID)} sources ({progress_pct:.1f}%) | ETA: {eta_str}\033[K', level='full', end='', flush=True)
                    last_update = current_time
            
            pairs_with_paths = len(pairs_with_paths_dict)
            
            # Final update
            elapsed = time.time() - start_time
            if self.verbose_mode == 'simple':
                self._vprint('Done', level='simple')
                self._vprint('building paths...', level='simple', end='', flush=True)
            elif self.verbose_mode == 'full':
                # Use \033[K to clear to end of line
                self._vprint(f'\r   Progress: {sources_processed}/{len(source_ID)} sources (100.0%) | Completed in {elapsed:.1f}s\033[K', level='full')
        
        self._vprint(f'\n✅ Pathfinding complete!', level='full')
        self._vprint(f'   Total paths found: {path_count:,}', level='full')
        self._vprint(f'   Neurons in valid paths: {len(neurons_in_paths):,}', level='full')
        self._vprint(f'   Unique edges in valid paths: {len(edges_in_paths):,}', level='full')
        self._vprint(f'   Layer-specific edges in valid paths: {len(edges_in_paths_with_layer):,}', level='full')
        
        # Now extract connections, keeping ALL layer-specific occurrences
        # This means if neuron A→B exists in both Layer 0→1 and Layer 2→3, both are kept
        # Initialize with expected columns to handle empty case gracefully
        conn_inpath = pd.DataFrame(columns=['conn_layer', 'bodyId_pre', 'bodyId_post', 'weight', 'type_pre', 'type_post', 'traversal_probability', 'connection_ratio'])
        conn_types = pd.DataFrame(columns=['conn_layer', 'type_pre', 'type_post', 'weight', 'traversal_probability', 'connection_ratio'])
        conn_groups = pd.DataFrame()  # For custom group aggregations
        weight_layers = {}
        
        for layer_idx, conn_df in enumerate(all_connections):
            # Skip empty connection DataFrames
            if conn_df.empty:
                continue
                
            # Get the actual layer index from the conn_layer label
            layer_label = conn_df['conn_layer'].iloc[0]
            actual_layer_idx = int(layer_label.split('->')[0])
            
            # Filter to keep only edges that are in valid paths for THIS specific layer
            conn_filtered = conn_df[
                conn_df.apply(
                    lambda row: (actual_layer_idx, row['bodyId_pre'], row['bodyId_post']) in edges_in_paths_with_layer,
                    axis=1
                )
            ]
            
            if len(conn_filtered) == 0:
                continue
            
            # Remove conn_layer temporarily (will add back after enrichment)
            conn_filtered_no_layer = conn_filtered.drop(columns=['conn_layer'])
            
            # Get all neurons involved in this layer's connections (for accurate ratio calculation)
            bodyIds_in_layer = np.unique(np.concatenate([conn_filtered_no_layer['bodyId_pre'].unique(), conn_filtered_no_layer['bodyId_post'].unique()]))
            neurons_in_layer_df = self._fetch_neurons_local_or_api(bodyIds_in_layer.tolist(), columns=['bodyId', 'type', 'post'])
            
            # Enrich with traversal probability (use local dataset if available)
            conn_enriched, conn_type, conn_group = sv.EnrichConnectionTable(
                conn_filtered_no_layer,
                dataset=self.dataset, 
                script_path=self.script_path,
                target_neurons_df=neurons_in_layer_df
            )
            
            # Add conn_layer column AFTER enrichment
            conn_enriched.insert(loc=0, column='conn_layer', value=layer_label)
            conn_type.insert(loc=0, column='conn_layer', value=layer_label)
            if conn_group is not None:
                conn_group.insert(loc=0, column='conn_layer', value=layer_label)
            
            conn_inpath = pd.concat([conn_inpath, conn_enriched])
            conn_types = pd.concat([conn_types, conn_type])
            if conn_group is not None:
                conn_groups = pd.concat([conn_groups, conn_group])
            
            weight_layers[layer_label] = conn_enriched['weight'].sum()
            
            self._vprint(f'Layer {layer_label}: {len(conn_filtered)} connections kept', level='full')
        
        # Build neuron_layers structure for visualization (based on actual path data)
        # Group neurons by their earliest appearance layer in valid paths
        neuron_layers = []
        for layer_idx in range(len(all_connections) + 1):
            layer_label_in = f'{layer_idx-1}->{layer_idx}' if layer_idx > 0 else None
            layer_label_out = f'{layer_idx}->{layer_idx+1}'
            
            neurons_in_layer = set()
            
            if layer_idx == 0:
                # Layer 0: source neurons that are in paths
                neurons_in_layer = set(source_ID) & neurons_in_paths
            else:
                # Neurons that appear as targets in this layer's incoming connections
                if len(conn_inpath) > 0 and layer_label_in in conn_inpath['conn_layer'].values:
                    incoming = conn_inpath[conn_inpath['conn_layer'] == layer_label_in]
                    neurons_in_layer = set(incoming['bodyId_post'].unique())
            
            if len(neurons_in_layer) > 0:
                neuron_layers.append(np.array(list(neurons_in_layer)))
            elif layer_idx == 0:
                # Always include layer 0 even if empty
                neuron_layers.append(np.array([]))
        
        # Ensure we have at least source layer
        if len(neuron_layers) == 0:
            neuron_layers = [np.array(list(set(source_ID) & neurons_in_paths))]
        
        # Update target real layers based on their actual appearance in paths
        # Targets should have real_layer = their earliest appearance layer
        # This is assigned AFTER pathfinding completes to avoid interfering with the search
        self._vprint('\n=== Updating target real layers based on path appearances ===', level='full')
        target_appearance_layers = {}  # Track all layers each target appears in
        
        # Iterate over all neurons with a progress indicator to aid long runs
        total_neurons_iter = sum(len(l) for l in neuron_layers)
        try:
            progress_iter = ((layer_idx, neuron_id) for layer_idx, layer in enumerate(neuron_layers) for neuron_id in layer)
            for layer_idx, neuron_id in tqdm(progress_iter, total=total_neurons_iter, desc='Updating target real layers', unit='neurons'):
                if neuron_id in targets_found:
                    if neuron_id not in target_appearance_layers:
                        target_appearance_layers[neuron_id] = []
                    target_appearance_layers[neuron_id].append(layer_idx)
        except Exception:
            # Fallback to simple loop if tqdm fails for any reason
            for layer_idx, layer in enumerate(neuron_layers):
                for neuron_id in layer:
                    if neuron_id in targets_found:
                        if neuron_id not in target_appearance_layers:
                            target_appearance_layers[neuron_id] = []
                        target_appearance_layers[neuron_id].append(layer_idx)
        
        # Update real_layer_map for targets to their earliest appearance
        for target_id, appearance_layers in target_appearance_layers.items():
            earliest_layer = min(appearance_layers)
            # Assign target real_layer as earliest appearance
            # This is done after pathfinding to avoid backward connection issues during search
            real_layer_map_bodyId[target_id] = earliest_layer
        
        # Print summary only
        if len(target_appearance_layers) > 0:
            self._vprint(f'  ✓ Updated real_layer for {len(target_appearance_layers)} target neurons', level='full')
        else:
            self._vprint('  ⚠ No targets found in paths', level='full')
        
        # Sort the combined connection data (only if non-empty)
        if not conn_inpath.empty:
            conn_inpath = conn_inpath.sort_values(by=['conn_layer','traversal_probability','weight'],ascending=[True,False,False])
            conn_inpath = conn_inpath.reset_index(drop=True)
        if not conn_types.empty:
            conn_types = conn_types.sort_values(by=['conn_layer','traversal_probability','weight'],ascending=[True,False,False])
            conn_types = conn_types.reset_index(drop=True)

        totalweight_df = pd.DataFrame(weight_layers.items(),columns=['conn_layer','weight'])
        if not totalweight_df.empty:
            totalweight_df = totalweight_df.sort_values(by='conn_layer',ascending=True)
        
        # Create type-level real layer map from bodyId-level real layers
        # For type-level analysis, use the earliest layer any neuron of that type appears
        # Targets already have their real layers updated based on actual path appearances
        real_layer_map_type = {}
        target_types_set = set(self.target_df.loc[self.target_df.Checked, 'type'].unique())
        target_type_appearances = {}  # Track appearance layers for target types
        
        if len(conn_inpath) > 0:
            has_types = 'type_pre' in conn_inpath.columns and 'type_post' in conn_inpath.columns
            
            for idx in conn_inpath.index:
                bodyId_pre = conn_inpath.at[idx, 'bodyId_pre']
                bodyId_post = conn_inpath.at[idx, 'bodyId_post']
                
                if has_types:
                    type_pre = conn_inpath.at[idx, 'type_pre']
                    type_post = conn_inpath.at[idx, 'type_post']
                    
                    # Map each type to earliest real layer of any neuron of that type
                    # For targets, use their updated real_layer from bodyId map
                    if bodyId_pre in real_layer_map_bodyId:
                        layer_pre = real_layer_map_bodyId[bodyId_pre]
                        if type_pre not in real_layer_map_type or layer_pre < real_layer_map_type[type_pre]:
                            real_layer_map_type[type_pre] = layer_pre
                        
                        # Track target type appearances
                        if type_pre in target_types_set and bodyId_pre in target_appearance_layers:
                            if type_pre not in target_type_appearances:
                                target_type_appearances[type_pre] = set()
                            target_type_appearances[type_pre].update(target_appearance_layers[bodyId_pre])
                    
                    if bodyId_post in real_layer_map_bodyId:
                        layer_post = real_layer_map_bodyId[bodyId_post]
                        if type_post not in real_layer_map_type or layer_post < real_layer_map_type[type_post]:
                            real_layer_map_type[type_post] = layer_post
                        
                        # Track target type appearances
                        if type_post in target_types_set and bodyId_post in target_appearance_layers:
                            if type_post not in target_type_appearances:
                                target_type_appearances[type_post] = set()
                            target_type_appearances[type_post].update(target_appearance_layers[bodyId_post])
        
        self._vprint(f'\nCreated type-level real layer map for {len(real_layer_map_type)} types', level='full')
        
        # Print target type appearance summary
        if target_type_appearances:
            self._vprint(f'  ✓ Updated real_layer for {len(target_type_appearances)} target types', level='full')
        
        # Create group-level real layer map if custom groups exist
        real_layer_map_group = {}
        if conn_groups is not None and not conn_groups.empty and 'custom_group' in self.source_df.columns:
            target_groups_set = set(self.target_df.loc[self.target_df.Checked, 'custom_group'].unique())
            target_group_appearances = {}  # Track appearance layers for target groups
            
            # Build bodyId to custom_group mapping from source and target dataframes
            bodyid_to_group = {}
            for df in [self.source_df, self.target_df]:
                if 'custom_group' in df.columns:
                    for idx in df.index:
                        bodyid = df.at[idx, 'bodyId']
                        group = df.at[idx, 'custom_group']
                        if pd.notna(group):
                            bodyid_to_group[bodyid] = group
            
            # Map each group to earliest real layer of any neuron in that group
            for bodyid, real_layer in real_layer_map_bodyId.items():
                if bodyid in bodyid_to_group:
                    group = bodyid_to_group[bodyid]
                    if group not in real_layer_map_group or real_layer < real_layer_map_group[group]:
                        real_layer_map_group[group] = real_layer
                    
                    # Track target group appearances
                    if group in target_groups_set and bodyid in target_appearance_layers:
                        if group not in target_group_appearances:
                            target_group_appearances[group] = set()
                        target_group_appearances[group].update(target_appearance_layers[bodyid])
            
            # Ensure all groups in conn_groups have layer assignments
            # Use type-level real_layer_map to assign layers to groups
            if 'type_pre' in conn_inpath.columns and 'custom_group_pre' in conn_inpath.columns:
                # Build type to group mapping from conn_inpath
                type_to_group = {}
                for idx in conn_inpath.index:
                    group_pre = conn_inpath.at[idx, 'custom_group_pre']
                    type_pre = conn_inpath.at[idx, 'type_pre']
                    if pd.notna(group_pre) and group_pre not in real_layer_map_group:
                        # Use type's layer for this group if type has layer
                        if type_pre in real_layer_map_type:
                            if group_pre not in type_to_group:
                                type_to_group[group_pre] = []
                            type_to_group[group_pre].append(real_layer_map_type[type_pre])
                    
                    group_post = conn_inpath.at[idx, 'custom_group_post']
                    type_post = conn_inpath.at[idx, 'type_post']
                    if pd.notna(group_post) and group_post not in real_layer_map_group:
                        if type_post in real_layer_map_type:
                            if group_post not in type_to_group:
                                type_to_group[group_post] = []
                            type_to_group[group_post].append(real_layer_map_type[type_post])
                
                # Assign minimum layer to each group
                for group, layers in type_to_group.items():
                    if layers:
                        real_layer_map_group[group] = min(layers)
            
            print(f'\nCreated group-level real layer map for {len(real_layer_map_group)} custom groups')
            if target_group_appearances:
                print(f'  ✓ Updated real_layer for {len(target_group_appearances)} target groups')

        # Mark which source neurons are in paths to targets
        if len(conn_inpath) > 0:
            source_inpath = conn_inpath.loc[conn_inpath.conn_layer=='0->1','bodyId_pre'].unique()
            if 'isInPath' in self.source_df.columns:
                self.source_df['isInPath'] = False
            else:
                self.source_df.insert(loc=0,column='isInPath',value=False)
            self.source_df.loc[self.source_df.bodyId.isin(source_inpath),'isInPath'] = True
        
        # Print statistics about paths
        self._vprint(f'\nPath Network Statistics (source to target):', level='full')
        self._vprint(f'Total connections in paths: {len(conn_inpath)}', level='full')
        self._vprint(f'Total connection types in paths: {len(conn_types)}', level='full')
        total_neurons = sum(len(layer) for layer in neuron_layers)
        self._vprint(f'Total neurons in paths: {total_neurons}', level='full')
        for i, layer in enumerate(neuron_layers):
            self._vprint(f'  Layer {i}: {len(layer)} neurons', level='full')
        
        # Print target distribution and which targets were found in each layer
        self._vprint('\nTarget neurons by layer:', level='full')
        all_found_targets = set()
        total_checked_targets = len(self.target_df[self.target_df['Checked']])
        
        for layer_idx in sorted(self.target_df[self.target_df['Checked']]['Layer'].unique()):
            targets_in_layer = self.target_df[
                (self.target_df['Layer'] == layer_idx) & (self.target_df['Checked'])
            ]
            
            # Check which targets from this layer are actually in paths
            if self.filter_by == 'bodyId':
                found_in_layer = targets_in_layer[
                    targets_in_layer['bodyId'].isin(conn_inpath['bodyId_post'].unique())
                ]
                all_found_targets.update(found_in_layer['bodyId'].tolist())
                self._vprint(f'  Layer {layer_idx}: {len(found_in_layer)}/{len(targets_in_layer)} targets found', level='full')
                if len(found_in_layer) > 0 and len(found_in_layer) <= 20:
                    self._vprint(f'    Found: {found_in_layer["bodyId"].tolist()}', level='full')
            else:  # filter_by == 'type'
                if 'type' in targets_in_layer.columns and 'type_post' in conn_types.columns:
                    found_in_layer = targets_in_layer[
                        targets_in_layer['type'].isin(conn_types['type_post'].unique())
                    ]
                    all_found_targets.update(found_in_layer['type'].tolist())
                    self._vprint(f'  Layer {layer_idx}: {len(found_in_layer)}/{len(targets_in_layer)} targets found', level='full')
                    if len(found_in_layer) > 0 and len(found_in_layer) <= 20:
                        self._vprint(f'    Found: {found_in_layer["type"].tolist()}', level='full')
                else:
                    self._vprint(f'  Layer {layer_idx}: (Type info missing) targets found', level='full')
        
        self._vprint(f'\nTotal found targets: {len(all_found_targets)}/{total_checked_targets}', level='full')
        
        # Ensure output directory exists before saving
        if not os.path.exists(self.allpath_folder):
            os.makedirs(self.allpath_folder, exist_ok=True)
            self._vprint(f'  📁 Recreated output folder: {self.allpath_folder}', level='full')
        
        # Handle the case where no paths were found
        if conn_inpath.empty:
            self._vprint('\n⚠️  No paths found - saving minimal output data', level='full')
            
            # Create data_details folder
            csv_folder = os.path.join(self.allpath_folder, 'data_details')
            os.makedirs(csv_folder, exist_ok=True)
            
            # Save parameters and source/target info even without paths
            self.parameter_df.to_csv(os.path.join(csv_folder, 'parameters.csv'), index=False)
            self.source_df.to_csv(os.path.join(csv_folder, 'source_neurons.csv'))
            self.target_df.to_csv(os.path.join(csv_folder, 'target_neurons.csv'))
            
            # Create empty connection files
            empty_conn = pd.DataFrame(columns=['bodyId_pre', 'bodyId_post', 'weight', 'type_pre', 'type_post'])
            empty_conn.to_csv(os.path.join(csv_folder, 'connection_info_bodyId.csv'), index=False)
            empty_conn.to_csv(os.path.join(csv_folder, 'connection_type.csv'), index=False)
            
            self._vprint(f'  ✓ Saved to: {csv_folder}/', level='full')
            self._vprint('  ✓ Saved connection data', level='full')
            return
        
        # Update types for source and target neurons in conn_inpath using self.source_df and self.target_df
        # This ensures that even if enrichment failed (e.g. FAFB), we at least have types for start/end of paths
        
        # Create mapping from bodyId to type
        body_to_type = {}
        if 'bodyId' in self.source_df.columns and 'type' in self.source_df.columns:
            for idx, row in self.source_df.iterrows():
                body_to_type[str(row['bodyId'])] = row['type']
        
        if 'bodyId' in self.target_df.columns and 'type' in self.target_df.columns:
            for idx, row in self.target_df.iterrows():
                body_to_type[str(row['bodyId'])] = row['type']
                
        # Apply mapping to conn_inpath
        # conn_inpath has bodyId_pre, bodyId_post, type_pre, type_post
        if body_to_type:
            self._vprint(f'  Updating types for {len(body_to_type)} source/target neurons in connection table...', level='full')
            # Update type_pre
            conn_inpath['type_pre'] = conn_inpath.apply(
                lambda row: body_to_type.get(str(row['bodyId_pre']), row['type_pre']), axis=1
            )
            
            # Update type_post
            conn_inpath['type_post'] = conn_inpath.apply(
                lambda row: body_to_type.get(str(row['bodyId_post']), row['type_post']), axis=1
            )
            
            # Note: We do NOT re-aggregate conn_types here to preserve layer information.
            # Instead, we will generate a global type aggregation for the matrix below.

        # Generate global type-level aggregation for matrix generation (avoids duplicates from layers)
        self._vprint('  Generating global type-level matrix...', level='full')
        # Use conn_inpath (which has all edges). Deduplicate by bodyId pair to avoid double counting physical edges.
        conn_inpath_global = conn_inpath.drop_duplicates(subset=['bodyId_pre', 'bodyId_post'])
        
        # Fetch all neurons involved for accurate post counts
        all_bodyIds = np.unique(np.concatenate([conn_inpath_global['bodyId_pre'].unique(), conn_inpath_global['bodyId_post'].unique()]))
        all_neurons_df = self._fetch_neurons_local_or_api(all_bodyIds.tolist(), columns=['bodyId', 'type', 'post'])
        
        _, conn_types_global, _ = sv.EnrichConnectionTable(
            conn_inpath_global, 
            traversal_probability_threshold=self.min_traversal_probability,
            dataset=self.dataset,
            script_path=self.script_path,
            target_neurons_df=all_neurons_df,
            aggregate_method='product'
        )

        # Save main data (type-level aggregations)
        self._vprint('\nSaving connection data...', level='full')
        
        # Determine if using CSV or Excel based on output_format or data size
        EXCEL_ROW_LIMIT = 1_048_576
        use_csv = (self.output_format == 'csv') or (len(conn_types) >= EXCEL_ROW_LIMIT * 0.9)
        
        if use_csv:
            if self.output_format == 'csv':
                self._vprint(f'  💾 Saving data as CSV files (output_format="csv")', level='full')
            else:
                self._vprint(f'  ⚠️  Data too large for Excel ({len(conn_types):,} rows), saving as CSV', level='full')
            
            # Create data_details folder
            csv_folder = os.path.join(self.allpath_folder, 'data_details')
            os.makedirs(csv_folder, exist_ok=True)
            self._vprint(f'  💾 Saving data as CSV files to: {csv_folder}', level='full')
            self.parameter_df.to_csv(os.path.join(csv_folder, 'parameters.csv'), index=False)
            self.source_df.to_csv(os.path.join(csv_folder, 'source_neurons.csv'))
            self.target_df.to_csv(os.path.join(csv_folder, 'target_neurons.csv'))
            totalweight_df.to_csv(os.path.join(csv_folder, 'total_weight_layer.csv'))
            conn_types.to_csv(os.path.join(csv_folder, 'connection_type.csv'))
            if conn_groups is not None and not conn_groups.empty:
                conn_groups.to_csv(os.path.join(csv_folder, 'connection_custom_groups.csv'))
            
            # Save matrices (use global aggregation)
            self._save_matrices_to_csv(conn_types_global, csv_folder, level='type')
        else:
            output_excel_name = os.path.join(self.allpath_folder, self.source_fname + '_to_' + self.target_fname + '_allpaths_info.xlsx')
            self._vprint(f'  💾 Saving type-level data to: {output_excel_name}', level='full')
            with pd.ExcelWriter(output_excel_name, mode='w', engine='xlsxwriter') as writer:
                self.parameter_df.to_excel(writer,sheet_name='parameters',index=False)
                worksheet = writer.sheets['parameters']
                worksheet.set_column('A:A', 30, writer.book.add_format({'bold': True, 'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
                worksheet.set_column('B:B', 30, writer.book.add_format({'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
                
                self.source_df.to_excel(writer,sheet_name='source_neurons')
                self.target_df.to_excel(writer,sheet_name='target_neurons')
                totalweight_df.to_excel(writer,sheet_name='total_weight_layer')
                conn_types.to_excel(writer,sheet_name='connection_type')
                
                # Add custom group sheet if custom grouping was used
                if conn_groups is not None and not conn_groups.empty:
                    conn_groups.to_excel(writer,sheet_name='connection_custom_groups')
                
                # Save matrices (use global aggregation)
                self._save_matrices_to_excel(conn_types_global, writer, level='type')
        
        # Save bodyId-level data
        self._vprint(f'Saving bodyId-level allpaths data (rows: {len(conn_inpath):,})...', level='full')
        
        # Recalculate use_csv for bodyId data
        use_csv = (self.output_format == 'csv') or (len(conn_inpath) >= EXCEL_ROW_LIMIT * 0.9)
        
        if use_csv:
            if self.output_format == 'csv':
                self._vprint(f'  💾 Saving bodyId data as CSV (output_format="csv")', level='full')
            else:
                self._vprint(f'  ⚠️  Data too large for Excel ({len(conn_inpath):,} rows), saving as CSV', level='full')
            
            # Use data_details folder (same as type-level data)
            bodyid_folder = os.path.join(self.allpath_folder, 'data_details')
            os.makedirs(bodyid_folder, exist_ok=True)
            
            # Save bodyId connection data as CSV (parameters.csv already saved with type-level data)
            output_bodyid_csv = os.path.join(bodyid_folder, 'connection_info_bodyId.csv')
            conn_inpath.to_csv(output_bodyid_csv, index=False)
            self._save_matrices_to_csv(conn_inpath_global, bodyid_folder, level='bodyId')
            self._vprint(f'  ✓ Saved to: {bodyid_folder}/', level='full')
        else:
            # Data fits in Excel
            output_bodyid_excel = os.path.join(self.allpath_folder, self.source_fname + '_to_' + self.target_fname + '_allpaths_bodyId_data.xlsx')
            with pd.ExcelWriter(output_bodyid_excel, mode='w', engine='xlsxwriter') as writer:
                self.parameter_df.to_excel(writer,sheet_name='parameters',index=False)
                worksheet = writer.sheets['parameters']
                worksheet.set_column('A:A', 30, writer.book.add_format({'bold': True, 'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
                worksheet.set_column('B:B', 30, writer.book.add_format({'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
                
                # Save bodyId-level connection info
                conn_inpath.to_excel(writer,sheet_name='connection_info_bodyId')
                self._save_matrices_to_excel(conn_inpath_global, writer, level='bodyId')
            self._vprint(f'  ✓ Saved to: {output_bodyid_excel}', level='full')
        
        self._vprint(f'  ✓ Saved connection data', level='full')
        
        # Build path DataFrames directly from collected paths (OPTIMIZED - no re-pathfinding!)
        self._vprint('\n=== Building path DataFrames from collected paths ===', level='full')
        if use_parallel:
            self._vprint(f'Processing {len(all_paths):,} paths found during parallel DFS...', level='full')
            self._vprint('Note: Path structure already found by DFS, now extracting connection metrics (weights, probabilities, ratios)...', level='full')
        else:
            self._vprint(f'Found {path_count:,} paths during sequential DFS', level='full')
            self._vprint('Note: Now building type/group level summaries...', level='full')
        
        # Type-level paths - Use separate DFS on type-level graph (much faster!)
        self._vprint('\nFinding type-level paths using type-level graph...', level='full')
        
        # Build type-level graph from conn_types
        G_type = nx.DiGraph()
        for idx in conn_types.index:
            type_pre = conn_types.at[idx, 'type_pre']
            type_post = conn_types.at[idx, 'type_post']
            weight = conn_types.at[idx, 'weight']
            if G_type.has_edge(type_pre, type_post):
                G_type[type_pre][type_post]['weight'] += weight
            else:
                G_type.add_edge(type_pre, type_post, weight=weight)
        
        self._vprint(f'  Type-level graph: {G_type.number_of_nodes()} types, {G_type.number_of_edges()} edges', level='full')
        
        # Get source and target types (filter out NaN/None values)
        source_types = [t for t in self.source_df['type'].unique().tolist() 
                        if t is not None and (not isinstance(t, float) or not pd.isna(t))]
        target_types = [t for t in self.target_df.loc[self.target_df.Checked, 'type'].unique().tolist()
                        if t is not None and (not isinstance(t, float) or not pd.isna(t))]
        
        # Find paths using DFS on type graph
        type_paths = []
        for source_type in source_types:
            if source_type not in G_type:
                continue
            for target_type in target_types:
                if target_type not in G_type:
                    continue
                if nx.has_path(G_type, source_type, target_type):
                    # Find all simple paths with length <= max_interlayer + 1
                    for path in nx.all_simple_paths(G_type, source_type, target_type, cutoff=self.max_interlayer + 1):
                        type_paths.append(path)
        
        self._vprint(f'  Found {len(type_paths):,} type-level paths', level='full')
        
        # Build DataFrame from type paths
        path_df_type = sv.build_path_dataframe_from_paths(
            paths=type_paths,
            conn_data=conn_types,
            targets=target_types,
            real_layer_map=real_layer_map_type if forward_only else None,
            level='type'
        )
        
        # Group-level paths - Use separate DFS on group-level graph (if custom groups exist)
        path_df_group = pd.DataFrame()
        path_df_group_excluded = pd.DataFrame()
        
        if conn_groups is not None and not conn_groups.empty and 'custom_group' in self.source_df.columns:
            self._vprint('\nFinding group-level paths using group-level graph...', level='full')
            
            # Build group-level graph from conn_groups
            G_group = nx.DiGraph()
            for idx in conn_groups.index:
                row = conn_groups.loc[idx]
                group_pre = row['group_pre']
                group_post = row['group_post']
                weight = row['weight']
                # Ensure scalar values (not Series)
                if isinstance(group_pre, pd.Series):
                    group_pre = group_pre.iloc[0]
                if isinstance(group_post, pd.Series):
                    group_post = group_post.iloc[0]
                if isinstance(weight, pd.Series):
                    weight = weight.iloc[0]
                    
                if G_group.has_edge(group_pre, group_post):
                    G_group[group_pre][group_post]['weight'] += weight
                else:
                    G_group.add_edge(group_pre, group_post, weight=weight)
            
            self._vprint(f'  Group-level graph: {G_group.number_of_nodes()} groups, {G_group.number_of_edges()} edges', level='full')
            
            # Get source and target groups
            source_groups = self.source_df['custom_group'].unique().tolist()
            target_groups = self.target_df.loc[self.target_df.Checked, 'custom_group'].unique().tolist()
            
            # Find paths using DFS on group graph
            group_paths = []
            for source_group in source_groups:
                if pd.isna(source_group) or source_group not in G_group:
                    continue
                for target_group in target_groups:
                    if pd.isna(target_group) or target_group not in G_group:
                        continue
                    if nx.has_path(G_group, source_group, target_group):
                        # Find all simple paths with length <= max_interlayer + 1
                        for path in nx.all_simple_paths(G_group, source_group, target_group, cutoff=self.max_interlayer + 1):
                            group_paths.append(path)
            
            self._vprint(f'  Found {len(group_paths):,} group-level paths', level='full')
            
            # Debug: Check if all groups in paths have layer assignments
            if forward_only and len(group_paths) > 0:
                all_groups_in_paths = set()
                for path in group_paths:
                    all_groups_in_paths.update(path)
                missing_groups = [g for g in all_groups_in_paths if g not in real_layer_map_group]
                if missing_groups:
                    self._vprint(f'  ⚠ Warning: {len(missing_groups)} groups in paths missing from real_layer_map_group', level='full')
                    self._vprint(f'    First few missing: {missing_groups[:5]}', level='full')
            
            # Build DataFrame from group paths
            # Rename columns to match expected format (type_pre/type_post)
            conn_groups_for_paths = conn_groups.rename(columns={'group_pre': 'type_pre', 'group_post': 'type_post'})
            
            path_df_group = sv.build_path_dataframe_from_paths(
                paths=group_paths,
                conn_data=conn_groups_for_paths,
                targets=target_groups,
                real_layer_map=real_layer_map_group if forward_only else None,
                level='type'  # Use 'type' level since groups are treated like types
            )
            
            # Filter out paths with any zero-weight hops
            if len(path_df_group) > 0:
                before_filter = len(path_df_group)
                path_df_group = path_df_group[
                    path_df_group['weights'].apply(lambda w_list: all(w > 0 for w in w_list))
                ]
                after_filter = len(path_df_group)
                if before_filter > after_filter:
                    self._vprint(f'  Removed {before_filter - after_filter} paths with zero-weight hops at group level', level='full')
            
            path_df_group = sv.split_path(path_df_group)
            path_df_group, path_df_group_excluded = sv.path_filter(path_df_group, self.keyword_in_path_to_remove)
        
        # Filter out paths with any zero-weight hops
        # This happens when bodyId-level connections exist but type-level aggregation results in 0 weight
        if len(path_df_type) > 0:
            before_filter = len(path_df_type)
            path_df_type = path_df_type[
                path_df_type['weights'].apply(lambda w_list: all(w > 0 for w in w_list))
            ]
            after_filter = len(path_df_type)
            if before_filter > after_filter:
                self._vprint(f'  Removed {before_filter - after_filter} paths with zero-weight hops at type level', level='full')
        
        path_df_type = sv.split_path(path_df_type)
        path_df_type, path_df_type_excluded = sv.path_filter(path_df_type,self.keyword_in_path_to_remove)
        
        EXCEL_ROW_LIMIT = 1_048_576
        
        # Save group-level paths if they exist
        if len(path_df_group) > 0:
            # Create custom group visualizations if available
            self._vprint('\nCreating custom group visualizations...', level='full')
            group_paths_to_viz = path_df_group.head(self.pathN_to_show) if self.pathN_to_show > 0 else path_df_group.copy()
            
            # Ensure column names match what VisualizePath expects
            if 'ratios' in group_paths_to_viz.columns and 'connection_ratios' not in group_paths_to_viz.columns:
                group_paths_to_viz['connection_ratios'] = group_paths_to_viz['ratios']
            if 'probabilities' in group_paths_to_viz.columns and 'traversal_probabilities' not in group_paths_to_viz.columns:
                group_paths_to_viz['traversal_probabilities'] = group_paths_to_viz['probabilities']
            
            vp_group = VisualizePath(path_file=group_paths_to_viz, output_folder=os.path.join(self.allpath_folder, 'custom_groups'), verbose=(self.verbose_mode == 'full'))
            self._vprint(f'💾 Saving path_group data (rows: {len(path_df_group):,})...', level='full')
            # Check if we should save as CSV (matches type-level data format OR group data too large)
            save_group_as_csv = use_csv or (len(path_df_group) >= EXCEL_ROW_LIMIT * 0.9)
            
            if save_group_as_csv:
                # Save as CSV
                if len(path_df_group) >= EXCEL_ROW_LIMIT * 0.9:
                    self._vprint(f'   ⚠️  Group path data too large for Excel ({len(path_df_group):,} rows), saving as CSV', level='full')
                output_path_group_csv = os.path.join(self.allpath_folder, self.source_fname+'_to_'+self.target_fname+'_allpaths_group.csv')
                path_df_group.to_csv(output_path_group_csv, index=False)
                if len(path_df_group_excluded) > 0:
                    # Save excluded paths to data_details folder
                    details_folder = os.path.join(self.allpath_folder, 'data_details')
                    output_path_group_excluded_csv = os.path.join(details_folder, self.source_fname+'_to_'+self.target_fname+'_allpaths_group_excluded.csv')
                    path_df_group_excluded.to_csv(output_path_group_excluded_csv, index=False)
                self._vprint(f'   ✓ Saved to: {self.allpath_folder}/', level='full')
            else:
                # Add to Excel file (type-level was saved to Excel, so output_excel_name exists)
                output_excel_name = os.path.join(self.allpath_folder, self.source_fname + '_to_' + self.target_fname + '_allpaths_info.xlsx')
                with pd.ExcelWriter(output_excel_name, mode='a', engine='openpyxl') as writer:
                    path_df_group.to_excel(writer,sheet_name='path_group')
                    if len(path_df_group_excluded) > 0:
                        path_df_group_excluded.to_excel(writer,sheet_name='path_group_excluded')
                self._vprint('   ✓ path_group sheets saved', level='full')
        
        self._vprint(f'💾 Saving path_type data (rows: {len(path_df_type):,})...', level='full')
        # Check if we should save as CSV (matches type-level data format OR path data too large)
        save_type_as_csv = use_csv or (len(path_df_type) >= EXCEL_ROW_LIMIT * 0.9)
        
        if save_type_as_csv:
            # Save as CSV
            if len(path_df_type) >= EXCEL_ROW_LIMIT * 0.9:
                self._vprint(f'   ⚠️  Path data too large for Excel ({len(path_df_type):,} rows), saving as CSV', level='full')
            output_path_type_csv = os.path.join(self.allpath_folder, self.source_fname+'_to_'+self.target_fname+'_allpaths_type.csv')
            path_df_type.to_csv(output_path_type_csv, index=False)
            if len(path_df_type_excluded) > 0:
                # Save excluded paths to data_details folder
                details_folder = os.path.join(self.allpath_folder, 'data_details')
                output_path_type_excluded_csv = os.path.join(details_folder, self.source_fname+'_to_'+self.target_fname+'_allpaths_type_excluded.csv')
                path_df_type_excluded.to_csv(output_path_type_excluded_csv, index=False)
            self._vprint(f'   ✓ Saved to: {self.allpath_folder}/', level='full')
        else:
            # Add to Excel file (type-level was saved to Excel, so output_excel_name exists)
            output_excel_name = os.path.join(self.allpath_folder, self.source_fname + '_to_' + self.target_fname + '_allpaths_info.xlsx')
            with pd.ExcelWriter(output_excel_name, mode='a', engine='openpyxl') as writer:
                path_df_type.to_excel(writer,sheet_name='path_type')
                path_df_type_excluded.to_excel(writer,sheet_name='path_type_excluded')
            self._vprint('   ✓ path_type sheets saved', level='full')
        
        # BodyId-level paths
        if find_bodyId_path:
            self._vprint('\nBuilding bodyId-level paths with real_layer validation...', level='full')
            
            # Create type lookup from connection data
            type_lookup = {}
            if 'type_pre' in conn_inpath.columns:
                for _, row in conn_inpath[['bodyId_pre', 'type_pre']].drop_duplicates().iterrows():
                    type_lookup[row['bodyId_pre']] = row['type_pre']
            if 'type_post' in conn_inpath.columns:
                for _, row in conn_inpath[['bodyId_post', 'type_post']].drop_duplicates().iterrows():
                    type_lookup[row['bodyId_post']] = row['type_post']
            
            # Also add source and target info
            for _, row in self.source_df.iterrows():
                type_lookup[row['bodyId']] = row['type']
            for _, row in self.target_df.iterrows():
                type_lookup[row['bodyId']] = row['type']

            path_df_bodyId = sv.build_path_dataframe_from_paths(
                paths=all_paths,
                conn_data=conn_inpath,
                targets=self.target_df.loc[self.target_df.Checked,'bodyId'].tolist(),
                real_layer_map=real_layer_map_bodyId if forward_only else None,
                level='bodyId',
                type_lookup=type_lookup
            )
            
            # Save path_bodyId to the bodyId data file
            self._vprint(f'💾 Saving path_bodyId data (rows: {len(path_df_bodyId):,})...', level='full')
            if use_csv:
                # Save as CSV if connection data was saved as CSV
                output_path_csv = os.path.join(self.allpath_folder,self.source_fname+'_to_'+self.target_fname+'_allpaths_bodyId_paths.csv')
                path_df_bodyId.to_csv(output_path_csv, index=False)
                self._vprint(f'   ✓ Saved to: {output_path_csv}', level='full')
            else:
                # Add to the bodyId Excel file if it was created
                if len(path_df_bodyId) < EXCEL_ROW_LIMIT:
                    with pd.ExcelWriter(output_bodyid_excel, mode='a', engine='openpyxl') as writer:
                        path_df_bodyId.to_excel(writer,sheet_name='path_bodyId')
                    self._vprint(f'   ✓ Added path_bodyId sheet to: {output_bodyid_excel}', level='full')
                else:
                    self._vprint(f'   ⚠️  path_bodyId too large ({len(path_df_bodyId):,} rows), saving as separate CSV', level='full')
                    output_path_csv = os.path.join(self.allpath_folder,self.source_fname+'_to_'+self.target_fname+'_allpaths_bodyId_paths.csv')
                    path_df_bodyId.to_csv(output_path_csv, index=False)
                    self._vprint(f'   ✓ Saved to: {output_path_csv}', level='full')
        
        # save interlayer info to excel
        self._vprint('💾 Saving interlayer neuron info to Excel...', level='full')
        
        interlayers = []
        
        # Try to load complete neuron dataset for faster lookup
        dataset_clean = self.dataset.replace(':', '_').replace('.', '_')
        dataset_path = os.path.join(
            self.script_path,
            'datasets',
            f"{dataset_clean}_allneurons_neuron_df.csv"
        )
        
        # Check for subdirectory structure (common for FlyWire/FAFB)
        if not os.path.exists(dataset_path):
            # Try exact match in subdirectory
            dataset_path_subdir = os.path.join(
                self.script_path,
                'datasets',
                dataset_clean,
                f"{dataset_clean}_allneurons_neuron_df.csv"
            )
            if os.path.exists(dataset_path_subdir):
                dataset_path = dataset_path_subdir
            else:
                # Try to find ANY file ending in _allneurons_neuron_df.csv in the subdirectory
                subdir_path = os.path.join(self.script_path, 'datasets', dataset_clean)
                if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                    import glob
                    candidates = glob.glob(os.path.join(subdir_path, "*_allneurons_neuron_df.csv"))
                    if candidates:
                        dataset_path = candidates[0]
                        print(f"   Found dataset file via glob: {os.path.basename(dataset_path)}")
        
        use_local_dataset = os.path.exists(dataset_path)
        ndf_complete = None
        
        if use_local_dataset:
            self._vprint(f'   Using local dataset: {os.path.basename(dataset_path)}', level='full')
            if 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower():
                ndf_complete = pd.read_csv(dataset_path, header=0, index_col=None, dtype={'bodyId': str}, low_memory=False)
            else:
                ndf_complete = pd.read_csv(dataset_path, header=0, index_col=0, low_memory=False)
        else:
            if 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower():
                self._vprint(f'   ⚠️  Local dataset not found for FlyWire/FAFB. Skipping interlayer info fetch.', level='full')
                ndf_complete = pd.DataFrame()
            else:
                self._vprint(f'   Local dataset not found, will use API calls', level='full')
                # Ensure client is logged in before API calls
                if self.client_type == 'neuprint':
                    if self.client_hemibrain is None:
                        from neuprint import Client, set_default_client
                        self.client_hemibrain = Client(self.server, self.dataset, self.token)
                        set_default_client(self.client_hemibrain)
        
        # Fetch info for each layer
        from neuprint import NeuronCriteria as NC
        
        for i, neurons in enumerate(layer_neurons[1:], 1):
            neuron_list = list(neurons)
            if not neuron_list:
                interlayers.append(pd.DataFrame())
                continue
                
            if ndf_complete is not None and not ndf_complete.empty:
                # Use local dataset
                # Ensure string matching
                neuron_list_str = [str(x) for x in neuron_list]
                ndf_complete['bodyId'] = ndf_complete['bodyId'].astype(str)
                n_df = ndf_complete[ndf_complete['bodyId'].isin(neuron_list_str)].copy()
            else:
                # Use API
                if self.client_type == 'neuprint':
                    try:
                        n_df, _ = fetch_neurons(NC(bodyId=neuron_list))
                    except Exception as e:
                        print(f"Warning: Failed to fetch neurons for layer {i}: {e}")
                        n_df = pd.DataFrame()
                else:
                    n_df = pd.DataFrame()
            
            # Slim down to essential columns only: bodyId, type, instance
            # This significantly reduces file size for large datasets
            essential_cols = ['bodyId', 'type', 'instance']
            available_cols = [c for c in essential_cols if c in n_df.columns]
            if available_cols and len(n_df) > 0:
                n_df = n_df[available_cols].copy()
            
            interlayers.append(n_df)
            
        self._vprint(' ✓', level='full')
        
        self._vprint('   Writing interlayer sheets to bodyId file...', level='full', end='', flush=True)
        if use_csv:
            # Save each layer as CSV in bodyId subfolder
            for i in range(len(interlayers)):
                layer_csv = os.path.join(bodyid_folder, f'layer_{i+1}.csv')
                interlayers[i].to_csv(layer_csv, index=False)
        else:
            # Save to bodyId Excel file
            with pd.ExcelWriter(output_bodyid_excel, mode='a', engine='openpyxl') as writer:
                for i in range(len(interlayers)):
                    interlayers[i].to_excel(writer, sheet_name='layer_'+str(i+1), index=False)
        self._vprint(' ✓', level='full')
        self._vprint('   ✓ Interlayer sheets saved to bodyId file', level='full')
        self._vprint('Done\n', level='full')
        
        # ============================================================================
        # VISUALIZATION: Using VisualizePath only (PHASE 4)
        # ============================================================================
        
        # VisualizePath network visualization
        if self.verbose_mode == 'simple':
            self._vprint('Done', level='simple')  # End of "building paths..."
            self._vprint(f'Phase 4:', level='simple')
            self._vprint('creating type-level visualizations...', level='simple', end='', flush=True)
        else:
            self._vprint('\nCreating interactive network visualizations...', level='full')
        try:
            
            # Create network from path_type if it exists
            if len(path_df_type) > 0:
                # Filter paths if pathN_to_show is specified
                if self.pathN_to_show > 0 and len(path_df_type) > self.pathN_to_show:
                    # Calculate path strength (product of traversal probabilities)
                    # Paths are already sorted by traversal_probability in sv.getAllPath()
                    # Just take the first N paths
                    paths_to_visualize = path_df_type.head(self.pathN_to_show).copy()
                    if self.verbose_mode == 'full':
                        print(f'  Showing top {self.pathN_to_show} paths (by traversal_probability) out of {len(path_df_type)} total paths')
                else:
                    paths_to_visualize = path_df_type.copy()
                    if self.verbose_mode == 'full':
                        print(f'  Showing all {len(path_df_type)} paths')
                
                # Ensure path_block column exists (required by VisualizePath)
                if 'path_block' not in paths_to_visualize.columns:
                    if 'path' in paths_to_visualize.columns:
                        # path is the string representation (A->B)
                        paths_to_visualize['path_block'] = paths_to_visualize['path']
                    elif 'path_str' in paths_to_visualize.columns:
                        # path_str is the list representation
                        paths_to_visualize['path_block'] = paths_to_visualize['path_str'].apply(
                            lambda x: '->'.join(map(str, x)) if isinstance(x, list) else str(x)
                        )
                
                # Ensure column names match what VisualizePath expects
                if 'ratios' in paths_to_visualize.columns and 'connection_ratios' not in paths_to_visualize.columns:
                    paths_to_visualize['connection_ratios'] = paths_to_visualize['ratios']
                if 'probabilities' in paths_to_visualize.columns and 'traversal_probabilities' not in paths_to_visualize.columns:
                    paths_to_visualize['traversal_probabilities'] = paths_to_visualize['probabilities']

                vp = VisualizePath(
                    path_file=paths_to_visualize,
                    output_folder=self.allpath_folder,
                    source_color=self.source_color if hasattr(self, 'source_color') else '#1f77b4',
                    intermediate_color=self.intermediate_color if hasattr(self, 'intermediate_color') else '#2ca02c',
                    target_color=self.target_color if hasattr(self, 'target_color') else '#d62728',
                    link_color=self.link_color if hasattr(self, 'link_color') else 'rgba(100,100,100,0.3)',
                    network_layout=self.network_layout if hasattr(self, 'network_layout') else 'hierarchical',
                    showfig=self.showfig,
                    edgeN_limit=self.edgeN_limit,
                    output_format=self.output_format,
                    verbose=(self.verbose_mode == 'full')
                )
                vp.visualize()
                if self.verbose_mode == 'simple':
                    self._vprint('Done', level='simple')
                else:
                    self._vprint('  Created network_selected_paths.html and sankey_selected_paths.html', level='full')
            else:
                if self.verbose_mode == 'full':
                    self._vprint('  No paths found to visualize', level='full')
            
            # Create network from path_bodyId if it exists and requested
            if find_bodyId_path and len(path_df_bodyId) > 0:
                if self.verbose_mode == 'simple':
                    self._vprint('creating bodyId-level visualizations...', level='simple', end='', flush=True)
                else:
                    self._vprint('\nCreating bodyId-level network visualizations...', level='full')
                # Filter paths if pathN_to_show is specified
                if self.pathN_to_show > 0 and len(path_df_bodyId) > self.pathN_to_show:
                    paths_to_visualize_bodyId = path_df_bodyId.head(self.pathN_to_show).copy()
                    if self.verbose_mode == 'full':
                        self._vprint(f'  Showing top {self.pathN_to_show} bodyId paths (by traversal_probability) out of {len(path_df_bodyId)} total paths', level='full')
                else:
                    paths_to_visualize_bodyId = path_df_bodyId.copy()
                    if self.verbose_mode == 'full':
                        self._vprint(f'  Showing all {len(path_df_bodyId)} bodyId paths', level='full')
                
                # Ensure path_block column exists and format with types if available
                # We want format: bodyId_type -> bodyId_type -> ...
                
                def format_path_with_types(path_list):
                    if not isinstance(path_list, list):
                        # Try to parse if string
                        if isinstance(path_list, str) and '->' in path_list:
                            path_list = path_list.split('->')
                        else:
                            # Single node or other format
                            path_list = [path_list]
                    
                    formatted_nodes = []
                    for node in path_list:
                        node_str = str(node).strip()
                        # type_lookup should be available from the earlier block if find_bodyId_path is True
                        node_type = type_lookup.get(node_str) if 'type_lookup' in locals() else None
                        
                        if not node_type and 'type_lookup' in locals():
                            # Try int if key is int
                            try:
                                node_type = type_lookup.get(int(node_str))
                            except:
                                pass
                        
                        if node_type:
                            formatted_nodes.append(f"{node_str}_{node_type}")
                        else:
                            formatted_nodes.append(node_str)
                    
                    return '->'.join(formatted_nodes)

                if 'path_str' in paths_to_visualize_bodyId.columns:
                    paths_to_visualize_bodyId['path_block'] = paths_to_visualize_bodyId['path_str'].apply(format_path_with_types)
                elif 'path' in paths_to_visualize_bodyId.columns:
                     paths_to_visualize_bodyId['path_block'] = paths_to_visualize_bodyId['path'].apply(format_path_with_types)

                vp_bodyId = VisualizePath(
                    path_file=paths_to_visualize_bodyId,
                    output_folder=os.path.join(self.allpath_folder, 'bodyId_visualization'),
                    source_color=self.source_color if hasattr(self, 'source_color') else '#1f77b4',
                    intermediate_color=self.intermediate_color if hasattr(self, 'intermediate_color') else '#2ca02c',
                    target_color=self.target_color if hasattr(self, 'target_color') else '#d62728',
                    link_color=self.link_color if hasattr(self, 'link_color') else 'rgba(100,100,100,0.3)',
                    network_layout=self.network_layout if hasattr(self, 'network_layout') else 'hierarchical',
                    showfig=self.showfig,
                    edgeN_limit=self.edgeN_limit,
                    output_format=self.output_format,
                    verbose=(self.verbose_mode == 'full')
                )
                vp_bodyId.visualize()
                if self.verbose_mode == 'simple':
                    self._vprint('Done', level='simple')
                else:
                    self._vprint('  Created bodyId-level visualizations in bodyId_visualization subfolder', level='full')
                
            # Create custom group visualizations if available
            if len(path_df_group) > 0:
                if self.verbose_mode == 'full':
                    self._vprint('\nCreating custom group visualizations...', level='full')
                group_paths_to_viz = path_df_group.head(self.pathN_to_show) if self.pathN_to_show > 0 else path_df_group
                vp_group = VisualizePath(path_file=group_paths_to_viz, output_folder=os.path.join(self.allpath_folder, 'custom_groups'),
                                        source_color=self.source_color if hasattr(self, 'source_color') else '#1f77b4', showfig=self.showfig,
                                        edgeN_limit=self.edgeN_limit,
                                        output_format=self.output_format,
                                        verbose=(self.verbose_mode == 'full'))
                vp_group.visualize()
                if self.verbose_mode == 'full':
                    self._vprint(f'  ✓ Custom group visualizations created ({len(group_paths_to_viz)} paths)', level='full')
                    
        except Exception as e:
            self._vprint(f'  Warning: VisualizePath visualization failed: {e}', level='full')
            import traceback
            traceback.print_exc()
        
        # Heatmap generation removed - use VisualizePath.visualize() for heatmaps instead
        
        if self.verbose_mode == 'simple':
            self._vprint('\n===========', level='simple')
            self._vprint('¡COMPLETED!', level='simple')
            self._vprint('===========\n', level='simple')
        else:
            self._vprint('Done\n', level='full')
    
    def _get_network_layout(self, G):
        '''Get network layout based on network_layout parameter'''
        if self.network_layout == 'layered':
            # Multipartite layout - nodes arranged in layers
            pos = nx.multipartite_layout(G, subset_key='layer', align='horizontal')
        elif self.network_layout == 'distributed':
            # Spring layout with layer-based initial positions for better distribution
            # Start with multipartite layout as seed
            initial_pos = nx.multipartite_layout(G, subset_key='layer', align='horizontal')
            # Apply spring layout for better clarity
            pos = nx.spring_layout(G, pos=initial_pos, k=1.5, iterations=50, seed=42)
        else:
            raise ValueError(f"network_layout must be 'layered' or 'distributed', got '{self.network_layout}'")
        return pos
    
    def _create_interactive_network(self, conn_types, conn_inpath, neuron_layers, target_type, target_ID,
                                   forward_only=True, edges_in_path_type=None, edges_in_path_bodyId=None):
        '''Create interactive network visualizations using NetworkX and Plotly
        
        Parameters:
        -----------
        forward_only : bool
            If True, only show edges that appear in valid paths (filtered visualization)
            If False, show all edges in conn_types/conn_inpath (complete graph)
        edges_in_path_type : set
            Set of (layer_idx, type_pre, type_post) tuples from path_type
        edges_in_path_bodyId : set
            Set of (layer_idx, bodyId_pre, bodyId_post) tuples from path_bodyId
        '''
        
        if edges_in_path_type is None:
            edges_in_path_type = set()
        if edges_in_path_bodyId is None:
            edges_in_path_bodyId = set()
        
        # Network by type
        self._vprint('Building interactive network by type...', level='full')
        G_type = nx.DiGraph()
        
        # Build a mapping from bodyId to type for layer assignment
        bodyId_to_type = {}
        if len(conn_inpath) > 0:
            for idx in conn_inpath.index:
                pre_id = conn_inpath.at[idx, 'bodyId_pre']
                post_id = conn_inpath.at[idx, 'bodyId_post']
                pre_type = conn_inpath.at[idx, 'type_pre']
                post_type = conn_inpath.at[idx, 'type_post']
                bodyId_to_type[pre_id] = pre_type
                bodyId_to_type[post_id] = post_type
        
        # Create type to layer mapping (type can appear in multiple layers, use earliest)
        type_to_layer = {}
        for layer_idx, layer in enumerate(neuron_layers):
            for bodyId in layer:
                if bodyId in bodyId_to_type:
                    neuron_type = bodyId_to_type[bodyId]
                    # Use earliest layer appearance
                    if neuron_type not in type_to_layer:
                        type_to_layer[neuron_type] = layer_idx
        
        # First, add edges to determine which nodes are actually involved
        nodes_in_edges = set()
        for idx in conn_types.index:
            layer_label = conn_types.at[idx, 'conn_layer']
            layer_idx = int(layer_label.split('->')[0])
            source = conn_types.at[idx, 'type_pre']
            target = conn_types.at[idx, 'type_post']
            weight = conn_types.at[idx, 'weight']
            prob = conn_types.at[idx, 'traversal_probability']
            ratio = conn_types.at[idx, 'connection_ratio'] if 'connection_ratio' in conn_types.columns else 0
            
            # If forward_only=True, only consider edges that are in path_type
            if forward_only and (layer_idx, source, target) not in edges_in_path_type:
                continue
            
            # Track nodes that appear in edges
            nodes_in_edges.add(source)
            nodes_in_edges.add(target)
            G_type.add_edge(source, target, weight=weight, probability=prob, ratio=ratio)
        
        # Now add only nodes that are involved in edges (have connections)
        for neuron_type in nodes_in_edges:
            if neuron_type in type_to_layer:
                layer_idx = type_to_layer[neuron_type]
                is_target = neuron_type in target_type
                is_source = layer_idx == 0
                if is_target:
                    node_cat = 'target'
                elif is_source:
                    node_cat = 'source'
                else:
                    node_cat = 'intermediate'
                G_type.add_node(neuron_type, layer=layer_idx, node_type=node_cat)
        
        # Create layout based on network_layout parameter
        print(f'Using "{self.network_layout}" layout...')
        pos_type = self._get_network_layout(G_type)
        
        # Create Cytoscape interactive network
        print('Creating interactive network...')
        self._plot_cytoscape_network(
            G_type, pos_type,
            title=f'Interactive Network: {self.source_fname} to {self.target_fname} (by type)',
            filename=os.path.join(self.allpath_folder, f'Network_type_allpaths_snp{self.min_synapse_num}.html'),
            node_labels=None
        )
        
        # Network by bodyId (only if network is not too large)
        if len(conn_inpath) < 5000:  # Limit for performance
            print('Building interactive network by bodyId...')
            G_bodyId = nx.DiGraph()
            
            # First, add edges to determine which nodes are actually involved
            nodes_in_edges_bodyId = set()
            for idx in conn_inpath.index:
                layer_label = conn_inpath.at[idx, 'conn_layer']
                layer_idx = int(layer_label.split('->')[0])
                source = conn_inpath.at[idx, 'bodyId_pre']
                target = conn_inpath.at[idx, 'bodyId_post']
                weight = conn_inpath.at[idx, 'weight']
                prob = conn_inpath.at[idx, 'traversal_probability']
                ratio = conn_inpath.at[idx, 'connection_ratio'] if 'connection_ratio' in conn_inpath.columns else 0
                
                # If forward_only=True, only add edges that are in path_bodyId
                if forward_only and (layer_idx, source, target) not in edges_in_path_bodyId:
                    continue
                
                # Track nodes that appear in edges
                nodes_in_edges_bodyId.add(source)
                nodes_in_edges_bodyId.add(target)
                G_bodyId.add_edge(source, target, weight=weight, probability=prob, ratio=ratio)
            
            # Now add only nodes that are involved in edges (have connections)
            for layer_idx, layer in enumerate(neuron_layers):
                for bodyId in layer:
                    if bodyId not in nodes_in_edges_bodyId:
                        continue  # Skip nodes not involved in any edge
                    
                    is_target = bodyId in target_ID
                    is_source = layer_idx == 0
                    if is_target:
                        node_cat = 'target'
                    elif is_source:
                        node_cat = 'source'
                    else:
                        node_cat = 'intermediate'
                    G_bodyId.add_node(bodyId, layer=layer_idx, node_type=node_cat)
            
            # Create layout based on network_layout parameter
            pos_bodyId = self._get_network_layout(G_bodyId)
            
            # Fetch neuron info for labels (use local dataset if available)
            all_bodyIds = list(G_bodyId.nodes())
            node_info_df = self._fetch_neurons_local_or_api(all_bodyIds, columns=['bodyId', 'type'])
            node_labels = {}
            for idx in node_info_df.index:
                bodyId = node_info_df.at[idx, 'bodyId']
                neuron_type = node_info_df.at[idx, 'type'] if node_info_df.at[idx, 'type'] else 'None'
                node_labels[bodyId] = f"{neuron_type}_{bodyId}"
            
            # Create Cytoscape interactive network
            print('Creating interactive network...')
            self._plot_cytoscape_network(
                G_bodyId, pos_bodyId,
                title=f'Interactive Network: {self.source_fname} to {self.target_fname} (by bodyId)',
                filename=os.path.join(self.allpath_folder, f'Network_bodyId_allpaths_snp{self.min_synapse_num}.html'),
                node_labels=node_labels
            )
        else:
            print(f'Skipping bodyId network (too large: {len(conn_inpath)} connections)')
    
    def _plot_interactive_network(self, G, pos, title, filename, color_by='node_type', node_labels=None):
        '''Helper function to create interactive network plot using Plotly'''
        
        # Define color scheme
        color_map = {
            'source': 'rgba(60,100,200,0.8)',      # Blue for source
            'target': 'rgba(120,40,70,0.8)',       # Red for target
            'intermediate': 'rgba(100,200,100,0.6)' # Green for intermediate
        }
        
        # Create edge traces with arrows
        edge_traces = []
        edge_annotations = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            source_node = edge[0]
            target_node = edge[1]
            weight = G.edges[edge].get('weight', 1)
            prob = G.edges[edge].get('probability', 0)
            
            # Edge width based on weight (log scale for better visualization)
            edge_width = max(0.5, min(5, np.log10(weight + 1) * 2))
            
            # Create hover text with direction and weight (improved formatting)
            hover_text = f'<b>{source_node} → {target_node}</b><br>' \
                        f'<b>Weight: {weight:,}</b> synapses<br>' \
                        f'Probability: {prob:.3f}'
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=edge_width, color='rgba(150,150,150,0.5)'),
                hoverinfo='text',
                hovertext=hover_text,
                hoverlabel=dict(
                    bgcolor='white',
                    font_size=12,
                    font_family='Arial'
                ),
                showlegend=False
            )
            edge_traces.append(edge_trace)
            
            # Add arrow annotation at the end of each edge
            # Calculate arrow position (80% along the edge to avoid overlap with target node)
            arrow_x = x0 + 0.8 * (x1 - x0)
            arrow_y = y0 + 0.8 * (y1 - y0)
            
            annotation = dict(
                x=arrow_x,
                y=arrow_y,
                ax=x0 + 0.6 * (x1 - x0),
                ay=y0 + 0.6 * (y1 - y0),
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=max(1, edge_width * 0.5),
                arrowcolor='rgba(100,100,100,0.5)',
            )
            edge_annotations.append(annotation)
        
        # Create node trace with draggable markers
        node_x = []
        node_y = []
        node_color = []
        node_text = []
        node_size = []
        node_ids = []  # Store node IDs for reference
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_ids.append(str(node))
            
            # Get node attributes
            node_type = G.nodes[node].get('node_type', 'intermediate')
            layer = G.nodes[node].get('layer', 0)
            
            # Color by category
            node_color.append(color_map.get(node_type, 'gray'))
            
            # Node size based on degree
            degree = G.degree(node)
            node_size.append(max(10, min(30, degree * 3)))
            
            # Create hover text
            if node_labels and node in node_labels:
                label = node_labels[node]
            else:
                label = str(node)
            
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            hover_text = f'<b>{label}</b><br>' \
                        f'Layer: {layer}<br>' \
                        f'Type: {node_type}<br>' \
                        f'In-degree: {in_degree}<br>' \
                        f'Out-degree: {out_degree}<br>' \
                        f'<i>Drag to reposition</i>'
            node_text.append(hover_text)
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=1, color='white')
            ),
            text=node_text,
            textposition='top center',
            textfont=dict(size=8, color='rgba(0,0,0,0)'),  # Hidden text labels by default
            hoverinfo='text',
            showlegend=False,
            customdata=node_ids,  # Store node IDs for interaction
            name='nodes'
        )
        
        # Create legend traces (invisible scatter plots for legend)
        legend_traces = []
        for node_type, color in color_map.items():
            legend_trace = go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=node_type.capitalize(),
                showlegend=True
            )
            legend_traces.append(legend_trace)
        
        # Combine all traces
        fig = go.Figure(data=edge_traces + [node_trace] + legend_traces)
        
        # Update layout with interactive features
        fig.update_layout(
            title=dict(
                text=f'{title}<br><sub>Scroll to zoom • Double-click to reset view</sub>',
                font=dict(size=16)
            ),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=60),
            xaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                fixedrange=False  # Allow zooming and panning
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                fixedrange=False,  # Allow zooming and panning
                scaleanchor='x',
                scaleratio=1
            ),
            plot_bgcolor='white',
            width=1200,
            height=800,
            annotations=edge_annotations,  # Add arrow annotations
            dragmode='select'  # Use select mode for better node interaction
        )
        
        # Configuration for better interactivity
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': filename.replace('.html', ''),
                'height': 800,
                'width': 1200,
                'scale': 2
            },
            'scrollZoom': True
        }
        
        # Save figure with configuration
        fig.write_html(filename, auto_open=self.showfig, config=config, include_plotlyjs='cdn')
        
        # Add custom JavaScript for node dragging
        self._add_drag_functionality(filename, node_ids)
        
        print(f'Saved interactive network to {filename}')
        print('  → Interactive features: Pan/zoom with mouse, edges show direction and weight on hover')
    
    def _plot_cytoscape_network(self, G, pos, title, filename, node_labels=None):
        '''Create interactive network using Cytoscape.js for better dragging and interaction'''
        
        # Define color scheme
        color_map = {
            'source': '#3C64C8',      # Blue for source
            'target': '#782846',       # Red for target
            'intermediate': '#64C864'  # Green for intermediate
        }
        
        # Prepare nodes data
        nodes_data = []
        for node in G.nodes():
            node_type = G.nodes[node].get('node_type', 'intermediate')
            layer = G.nodes[node].get('layer', 0)
            x, y = pos[node]
            
            # Get label
            if node_labels and node in node_labels:
                label = node_labels[node]
            else:
                label = str(node)
            
            # Node size based on degree
            degree = G.degree(node)
            node_size = max(20, min(60, degree * 5))
            
            nodes_data.append({
                'data': {
                    'id': str(node),
                    'label': label,
                    'node_type': node_type,
                    'layer': layer,
                    'degree': degree
                },
                'position': {'x': x * 500, 'y': y * 500},  # Scale positions
                'classes': node_type
            })
        
        # Prepare edges data
        edges_data = []
        for edge in G.edges():
            source, target = edge
            weight = G.edges[edge].get('weight', 1)
            prob = G.edges[edge].get('probability', 0)
            ratio = G.edges[edge].get('ratio', 0)
            
            # Edge width based on weight (log scale)
            edge_width = max(1, min(10, np.log10(weight + 1) * 3))
            
            edges_data.append({
                'data': {
                    'id': f'{source}-{target}',
                    'source': str(source),
                    'target': str(target),
                    'weight': int(weight),
                    'probability': float(prob),
                    'ratio': float(ratio),
                    'edge_width': edge_width
                }
            })
        
        # Create HTML with Cytoscape.js
        html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }}
        #cy {{
            width: 100%;
            height: 100vh;
            display: block;
            background-color: white;
        }}
        #title {{
            position: absolute;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1000;
            background: rgba(255,255,255,0.95);
            padding: 15px 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }}
        #controls {{
            position: absolute;
            top: 80px;
            right: 10px;
            z-index: 1000;
            background: rgba(255,255,255,0.95);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            width: 200px;
        }}
        .control-btn {{
            width: 100%;
            padding: 8px;
            margin: 5px 0;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
        }}
        .control-btn:hover {{
            background: #45a049;
        }}
        #legend {{
            position: absolute;
            top: 80px;
            left: 10px;
            z-index: 1000;
            background: rgba(255,255,255,0.95);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .legend-item {{
            margin: 8px 0;
            display: flex;
            align-items: center;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
            border: 2px solid #333;
        }}
        #info {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            z-index: 1000;
            background: rgba(255,255,255,0.95);
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            font-size: 12px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div id="title">{title}</div>
    <div id="legend">
        <div style="font-weight: bold; margin-bottom: 10px;">Node Types</div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: {color_map['source']};"></div>
            <span>Source</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: {color_map['intermediate']};"></div>
            <span>Intermediate</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: {color_map['target']};"></div>
            <span>Target</span>
        </div>
    </div>
    <div id="controls">
        <div style="font-weight: bold; margin-bottom: 10px;">Controls</div>
        <button class="control-btn" onclick="cy.fit()">Fit to Screen</button>
        <button class="control-btn" onclick="cy.center()">Center View</button>
        <button class="control-btn" onclick="resetLayout()">Reset Layout</button>
        <button class="control-btn" onclick="toggleLabels()">Toggle Labels</button>
        <button class="control-btn" onclick="showAllNodes()" style="background: #2196F3;">Show All Nodes</button>
        <button class="control-btn" onclick="exportPNG()">Export PNG</button>
    </div>
    <div id="info">
        💡 <b>Drag nodes</b> to reposition • <b>Scroll</b> to zoom • <b>Drag background</b> to pan<br>
        <b>Click node + H</b> to hide • <b>Click</b> nodes/edges for details • <b>Double-click</b> to highlight
    </div>
    <div id="cy"></div>

    <script>
        let showLabels = true;
        
        const cy = cytoscape({{
            container: document.getElementById('cy'),
            
            elements: {{
                nodes: {nodes_data},
                edges: {edges_data}
            }},
            
            style: [
                {{
                    selector: 'node',
                    style: {{
                        'background-color': 'data(node_type)',
                        'width': 'mapData(degree, 0, 20, 20, 60)',
                        'height': 'mapData(degree, 0, 20, 20, 60)',
                        'label': 'data(label)',
                        'font-size': '10px',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'color': '#000',
                        'text-outline-color': '#fff',
                        'text-outline-width': 2,
                        'border-width': 2,
                        'border-color': '#333',
                        'cursor': 'grab'
                    }}
                }},
                {{
                    selector: 'node.source',
                    style: {{
                        'background-color': '{color_map['source']}'
                    }}
                }},
                {{
                    selector: 'node.intermediate',
                    style: {{
                        'background-color': '{color_map['intermediate']}'
                    }}
                }},
                {{
                    selector: 'node.target',
                    style: {{
                        'background-color': '{color_map['target']}'
                    }}
                }},
                {{
                    selector: 'node:selected',
                    style: {{
                        'border-width': 4,
                        'border-color': '#FFA500',
                        'background-color': '#FFD700'
                    }}
                }},
                {{
                    selector: 'edge',
                    style: {{
                        'width': 'data(edge_width)',
                        'line-color': '#999',
                        'target-arrow-color': '#999',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'arrow-scale': 1.5,
                        'opacity': 0.6
                    }}
                }},
                {{
                    selector: 'edge:selected',
                    style: {{
                        'line-color': '#FFA500',
                        'target-arrow-color': '#FFA500',
                        'width': 'calc(data(edge_width) * 1.5)',
                        'opacity': 1
                    }}
                }},
                {{
                    selector: '.highlighted',
                    style: {{
                        'background-color': '#FFD700',
                        'line-color': '#FFD700',
                        'target-arrow-color': '#FFD700',
                        'opacity': 1
                    }}
                }},
                {{
                    selector: '.hidden',
                    style: {{
                        'display': 'none'
                    }}
                }}
            ],
            
            layout: {{
                name: 'preset'
            }},
            
            minZoom: 0.1,
            maxZoom: 5,
            wheelSensitivity: 0.2
        }});
        
        // Make nodes draggable
        cy.nodes().grabify();
        
        // Show tooltips on hover
        cy.on('mouseover', 'node', function(evt) {{
            const node = evt.target;
            const data = node.data();
            const info = document.getElementById('info');
            info.innerHTML = `
                <b>Node:</b> ${{data.label}}<br>
                <b>Type:</b> ${{data.node_type}}<br>
                <b>Layer:</b> ${{data.layer}}<br>
                <b>Degree:</b> ${{data.degree}} (connections)
            `;
        }});
        
        cy.on('mouseover', 'edge', function(evt) {{
            const edge = evt.target;
            const data = edge.data();
            const info = document.getElementById('info');
            info.innerHTML = `
                <b>Connection:</b> ${{edge.source().data('label')}} → ${{edge.target().data('label')}}<br>
                <b>Weight:</b> ${{data.weight.toLocaleString()}} synapses<br>
                <b>Probability:</b> ${{data.probability.toFixed(4)}}<br>
                <b>Ratio:</b> ${{data.ratio.toFixed(4)}}
            `;
        }});
        
        cy.on('mouseout', 'node, edge', function() {{
            const hiddenCount = cy.nodes('.hidden').length;
            if (hiddenCount > 0) {{
                document.getElementById('info').innerHTML = `
                    💡 <b>${{hiddenCount}}</b> node(s) hidden • <b>Right-click</b> or press <b>H</b> on selected node to hide<br>
                    Click <b>Show All Nodes</b> button to restore hidden nodes
                `;
            }} else {{
                document.getElementById('info').innerHTML = `
                    💡 <b>Drag nodes</b> to reposition • <b>Scroll</b> to zoom • <b>Drag background</b> to pan<br>
                    <b>Click node + H</b> to hide • <b>Click</b> nodes/edges for details • <b>Double-click</b> to highlight
                `;
            }}
        }});
        
        // Double-click to highlight connected nodes
        cy.on('dblclick', 'node', function(evt) {{
            const node = evt.target;
            cy.elements().removeClass('highlighted');
            node.addClass('highlighted');
            node.neighborhood().addClass('highlighted');
        }});
        
        // Click background to clear highlights
        cy.on('tap', function(evt) {{
            if (evt.target === cy) {{
                cy.elements().removeClass('highlighted');
            }}
        }});
        
        // Keyboard shortcut: 'H' to hide selected nodes
        let selectedNode = null;
        cy.on('select', 'node', function(evt) {{
            selectedNode = evt.target;
        }});
        
        cy.on('unselect', 'node', function(evt) {{
            selectedNode = null;
        }});
        
        document.addEventListener('keydown', function(evt) {{
            if (evt.key === 'h' || evt.key === 'H') {{
                if (selectedNode && !selectedNode.hasClass('hidden')) {{
                    hideNode(selectedNode);
                }}
            }}
        }});
        
        // Context menu for right-click hide
        cy.on('cxttap', 'node', function(evt) {{
            const node = evt.target;
            if (!node.hasClass('hidden')) {{
                hideNode(node);
            }}
        }});
        
        // Function to hide a node and its connected edges
        function hideNode(node) {{
            // Hide the node
            node.addClass('hidden');
            
            // Hide connected edges
            node.connectedEdges().addClass('hidden');
            
            // Update info
            const hiddenCount = cy.nodes('.hidden').length;
            document.getElementById('info').innerHTML = `
                💡 <b>${{hiddenCount}}</b> node(s) hidden • <b>Right-click</b> or press <b>H</b> on selected node to hide<br>
                Click <b>Show All Nodes</b> button to restore hidden nodes
            `;
            
            // Deselect the node
            node.unselect();
        }}
        
        // Function to show all hidden nodes
        function showAllNodes() {{
            cy.elements('.hidden').removeClass('hidden');
            document.getElementById('info').innerHTML = `
                💡 <b>Drag nodes</b> to reposition • <b>Scroll</b> to zoom • <b>Drag background</b> to pan<br>
                <b>Click node + H</b> to hide • <b>Click</b> nodes/edges for details • <b>Double-click</b> to highlight
            `;
        }}
        
        // Control functions
        function resetLayout() {{
            cy.nodes().positions(function(node) {{
                return node.data('originalPos') || node.position();
            }});
            cy.fit();
        }}
        
        function toggleLabels() {{
            showLabels = !showLabels;
            if (showLabels) {{
                cy.style().selector('node').style({{'label': 'data(label)'}}).update();
            }} else {{
                cy.style().selector('node').style({{'label': ''}}).update();
            }}
        }}
        
        function exportPNG() {{
            const png = cy.png({{
                output: 'blob',
                bg: 'white',
                full: true,
                scale: 3
            }});
            const url = URL.createObjectURL(png);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'network.png';
            link.click();
            URL.revokeObjectURL(url);
        }}
        
        // Initial fit
        cy.fit();
    </script>
</body>
</html>'''
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f'✓ Saved interactive network to {filename}')
        print('  → Drag nodes to reposition • Hover edges to see weight/ratio/probability • Double-click to highlight')
    
    def _add_drag_functionality(self, filename, node_ids):
        '''Add JavaScript for draggable nodes to the HTML file'''
        
        # Read the generated HTML
        with open(filename, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # JavaScript code for dragging nodes
        drag_script = '''
<script>
// Add drag functionality to network nodes
(function() {
    // Wait for Plotly to be ready
    const checkPlotly = setInterval(function() {
        const plotDiv = document.querySelector('.plotly-graph-div');
        if (plotDiv && window.Plotly) {
            clearInterval(checkPlotly);
            initDragNodes(plotDiv);
        }
    }, 100);
    
    function initDragNodes(gd) {
        let isDragging = false;
        let dragNodeIndex = null;
        let dragMode = true;  // Start with drag mode ON by default
        let originalNodeX = [];
        let originalNodeY = [];
        
        // Add button to toggle drag mode
        const toggleButton = document.createElement('button');
        toggleButton.innerHTML = '🎯 Node Drag Mode (ON)';
        toggleButton.style.cssText = 'position: absolute; top: 10px; right: 10px; z-index: 1000; ' +
                                     'padding: 10px 16px; background: #f44336; color: white; ' +
                                     'border: none; border-radius: 6px; cursor: pointer; ' +
                                     'font-size: 13px; font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.2);';
        toggleButton.onclick = function() {
            dragMode = !dragMode;
            toggleButton.innerHTML = dragMode ? 
                '🎯 Node Drag Mode (ON)' : '🔄 Node Drag Mode (OFF)';
            toggleButton.style.background = dragMode ? '#f44336' : '#4CAF50';
            if (!dragMode) {
                gd.style.cursor = 'default';
            }
        };
        gd.parentElement.style.position = 'relative';
        gd.parentElement.appendChild(toggleButton);
        
        // Store original positions
        const nodeTrace = gd.data.find(trace => trace.name === 'nodes');
        if (nodeTrace) {
            originalNodeX = [...nodeTrace.x];
            originalNodeY = [...nodeTrace.y];
        }
        
        // Mouse event handlers
        gd.on('plotly_click', function(data) {
            if (!dragMode) return;
            
            // Find if clicked on a node
            for (let i = 0; i < data.points.length; i++) {
                const point = data.points[i];
                if (point.data.name === 'nodes') {
                    dragNodeIndex = point.pointIndex;
                    isDragging = true;
                    gd.style.cursor = 'grabbing';
                    // Store current position
                    originalNodeX[dragNodeIndex] = nodeTrace.x[dragNodeIndex];
                    originalNodeY[dragNodeIndex] = nodeTrace.y[dragNodeIndex];
                    break;
                }
            }
        });
        
        gd.addEventListener('mousemove', function(evt) {
            if (!isDragging || !dragMode || dragNodeIndex === null) return;
            
            evt.preventDefault();
            evt.stopPropagation();
            
            // Get mouse position relative to plot
            const xaxis = gd._fullLayout.xaxis;
            const yaxis = gd._fullLayout.yaxis;
            
            // Get the plot area
            const plotBbox = gd.getBoundingClientRect();
            const l = gd._fullLayout.margin.l;
            const t = gd._fullLayout.margin.t;
            
            // Convert pixel coordinates to data coordinates
            const xPixel = evt.clientX - plotBbox.left - l;
            const yPixel = evt.clientY - plotBbox.top - t;
            
            const xData = xaxis.p2c(xPixel);
            const yData = yaxis.p2c(yPixel);
            
            // Update node position in the data arrays directly
            const nodeTrace = gd.data.find(trace => trace.name === 'nodes');
            if (nodeTrace && dragNodeIndex < nodeTrace.x.length) {
                const oldX = nodeTrace.x[dragNodeIndex];
                const oldY = nodeTrace.y[dragNodeIndex];
                
                // Update node position
                nodeTrace.x[dragNodeIndex] = xData;
                nodeTrace.y[dragNodeIndex] = yData;
                
                // Update connected edges
                updateConnectedEdges(gd, dragNodeIndex, oldX, oldY, xData, yData);
                
                // Redraw
                Plotly.redraw(gd);
            }
        });
        
        gd.addEventListener('mouseup', function() {
            if (isDragging && dragMode) {
                isDragging = false;
                dragNodeIndex = null;
                gd.style.cursor = 'grab';
            }
        });
        
        gd.addEventListener('mouseleave', function() {
            if (isDragging) {
                isDragging = false;
                dragNodeIndex = null;
                gd.style.cursor = dragMode ? 'grab' : 'default';
            }
        });
        
        // Set initial cursor
        if (dragMode) {
            gd.style.cursor = 'grab';
        }
        
        function updateConnectedEdges(gd, nodeIndex, oldX, oldY, newNodeX, newNodeY) {
            // Get node trace
            const nodeTrace = gd.data.find(trace => trace.name === 'nodes');
            if (!nodeTrace) return;
            
            // Update all edge traces
            gd.data.forEach((trace, traceIdx) => {
                if (trace.mode === 'lines' && trace.x && trace.x.length >= 3) {
                    // Edge traces have format [x0, x1, null, x0, x1, null, ...]
                    // Check each edge (every 3 elements)
                    for (let i = 0; i < trace.x.length; i += 3) {
                        const x0 = trace.x[i];
                        const y0 = trace.y[i];
                        const x1 = trace.x[i + 1];
                        const y1 = trace.y[i + 1];
                        
                        // Check if source node matches (within tolerance)
                        if (x0 !== null && Math.abs(x0 - oldX) < 0.01 && 
                            Math.abs(y0 - oldY) < 0.01) {
                            trace.x[i] = newNodeX;
                            trace.y[i] = newNodeY;
                            
                            // Update arrow annotation if it exists
                            if (gd.layout.annotations && gd.layout.annotations[Math.floor(i/3)]) {
                                const ann = gd.layout.annotations[Math.floor(i/3)];
                                // Recalculate arrow position
                                ann.ax = newNodeX + 0.6 * (trace.x[i+1] - newNodeX);
                                ann.ay = newNodeY + 0.6 * (trace.y[i+1] - newNodeY);
                                ann.x = newNodeX + 0.8 * (trace.x[i+1] - newNodeX);
                                ann.y = newNodeY + 0.8 * (trace.y[i+1] - newNodeY);
                            }
                        }
                        
                        // Check if target node matches
                        if (x1 !== null && Math.abs(x1 - oldX) < 0.01 && 
                            Math.abs(y1 - oldY) < 0.01) {
                            trace.x[i + 1] = newNodeX;
                            trace.y[i + 1] = newNodeY;
                            
                            // Update arrow annotation if it exists
                            if (gd.layout.annotations && gd.layout.annotations[Math.floor(i/3)]) {
                                const ann = gd.layout.annotations[Math.floor(i/3)];
                                // Recalculate arrow position
                                ann.ax = trace.x[i] + 0.6 * (newNodeX - trace.x[i]);
                                ann.ay = trace.y[i] + 0.6 * (newNodeY - trace.y[i]);
                                ann.x = trace.x[i] + 0.8 * (newNodeX - trace.x[i]);
                                ann.y = trace.y[i] + 0.8 * (newNodeY - trace.y[i]);
                            }
                        }
                    }
                }
            });
        }
    }
})();
</script>
'''
        
        # Insert the script before the closing </body> tag
        html_content = html_content.replace('</body>', f'{drag_script}</body>')
        
        # Write back to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def ROImat(self, requiredNeurons: list = None, folder_name: str = None, site: str = 'post', break_threshod: int = 1e3, roi_list = None, roi_name = None, roi_kw: list | None = None, roi_kw_exclude: list | None = None):
        """ get the distribution matrix of ROI by the given site of neurons.
        
        Only the R hemisphere is considered.
        
        Args:
            requiredNeurons (list, optional): _description_. Defaults to self.sourceNeurons.
            folder_name (str, optional): _description_. Defaults to self.source_fname.
            break_threshod (int, optional): _description_. Defaults to 1e3. synapse number of one neuron, if synapse number is greater than the break_threshod, it will be breaked in the axis.
            roi_list (list, optional): _description_. Defaults to None. if None, all the ROIs will be considered.
            roi_name (list, optional): _description_. Defaults to None. if None, roi_name will be the same as roi_list except stripping the "(R)" suffix, that is, "AL(R)" -> "AL", "AL(L)" -> "AL(L)", "EB" -> "EB".
            roi_kw (str, optional): _description_. roi_keyword, Defaults to None. if None, all the ROIs will be considered, else only ROIs containing the keyword will be considered, such as 'AL-'.
        """
        
        if requiredNeurons == None:
            requiredNeurons = self.sourceNeurons
        # required_criteria, auto_name = sv.getCriteriaAndName(requiredNeurons)
        neuron_df, roi_count_df, auto_name, required_criteria = sv.getNeurons(requiredNeurons, dataset=self.dataset)
        if folder_name is None or folder_name == '':
            folder_name = auto_name
        print(f'Generating ROI distribution matrix of {folder_name} {site} synaptic sites...')
        # neuron_df,roi_count_df = fetch_neurons(required_criteria) # Fetch neuron info from hemibrain server.
        neuron_df: pd.DataFrame = neuron_df
        neuron_df.sort_values(by='type',inplace=True) # The order of neuron_df will be the order in the distribution matrix.
        rpath = os.path.join(self.data_folder, '_'.join(['roi_distribution',folder_name,site]))
        if not os.path.exists(rpath): os.makedirs(rpath)
        
        if roi_list is None:
            roi_list = roi_count_df.roi.unique().tolist()
        roi_list.sort()
        
        roi_list_new = [] # keep only ROIs containing the keyword in roi_kw
        if roi_kw is not None and roi_kw != '' and roi_kw != []:
            if type(roi_kw) == str:
                roi_kw = [roi_kw]
            for roi in roi_list:
                for kw in roi_kw:
                    if kw in roi:
                        roi_list_new.append(roi)
                        break
        roi_list = roi_list_new
        
        roi_list_new = [] # exclude ROIs containing the keyword in roi_kw_exclude
        if roi_kw_exclude is not None and roi_kw_exclude != '' and roi_kw_exclude != []:
            if type(roi_kw_exclude) == str:
                roi_kw_exclude = [roi_kw_exclude]
            for roi in roi_list:
                for kw in roi_kw_exclude:
                    if kw in roi: break
                else:
                    roi_list_new.append(roi)
        roi_list = roi_list_new
        
        if roi_name is None:
            roi_name = [] # custom name corresponding to "roi" property
            for roi in roi_list:
                if '(R)' in roi:
                    name = roi[:-3]
                else:
                    name = roi
                roi_name.append(name)
        
        if len(roi_list) != len(roi_name):
            raise ValueError('roi_list and roi_name must have the same length.')
        
        # generate a template for the roi matrix
        distMat = pd.DataFrame(
            data = np.zeros([len(roi_list),len(neuron_df)],dtype=int),
            index = roi_list,
            columns = neuron_df[['bodyId','type']]
        ) 
        distMat.columns = pd.MultiIndex.from_tuples(distMat.columns) # Column names include both "type" and "bodyId"
        for col in distMat.columns:
            bodyId = col[0]
            temp_df = roi_count_df[roi_count_df.bodyId == bodyId]
            for ind in distMat.index:
                series_snp = temp_df.loc[temp_df.roi == ind, site]
                if not series_snp.empty:
                    snpN = series_snp.iat[0]
                    distMat.at[ind,col] = snpN
        distMat_new = distMat.copy(deep=True)
        columns_name = [] 
        for ind in neuron_df.index: # set the column name to the format, (type)_(bodyId)
            name_i = neuron_df.at[ind,'type'] + '_' + str(neuron_df.at[ind,'bodyId'])
            columns_name.append(name_i)
        distMat_new.index = roi_name
        distMat_new.columns = columns_name
        
        # group by type
        distMat_type = distMat.copy()
        distMat_type.columns = neuron_df.type
        distMat_type.index = roi_name
        distMat_type = distMat_type.groupby(distMat_type.columns, axis=1).sum()

        # group breaked data by type
        distMat_break = distMat_new.copy()
        distMat_break[distMat_break > break_threshod] = break_threshod # traverse plane and break z-axis

        distMat_type_break = distMat_type.copy()
        distMat_type_break = distMat_type_break.groupby(distMat_type_break.columns, axis=1).sum()
        distMat_type_break[distMat_type_break > break_threshod] = break_threshod

        
        print('Saving ROI distribution matrix...')
        file = os.path.join(rpath,'mat_ROI.xlsx')
        with pd.ExcelWriter(file) as w:
            neuron_df.to_excel(w,sheet_name='PN_R_info')
            neuron_df.to_excel(w,sheet_name='neuron_df')
            roi_count_df.to_excel(w,sheet_name='roi_count_df')
            distMat_type.to_excel(w,sheet_name='roi_mat_type')
            distMat_new.to_excel(w,sheet_name='roi_mat')
            distMat_break.to_excel(w,sheet_name='roi_mat_break')
            distMat_type_break.to_excel(w,sheet_name='roi_mat_type_break')
            distMat.to_excel(w,sheet_name='roi_mat_multilevelCol')
        # visualize roi distribution matrix by the VisConnMat function
        print('Visualizing ROI distribution matrix...')
        sv.VisConnMat(distMat_type,os.path.join(rpath,'roi_type_heatmap.html'),fontsize=8,title=f'roi matrix of {folder_name} ({site}), grouped by type',showfig=self.showfig)
        sv.VisConnMat(distMat_new,os.path.join(rpath,'roi_heatmap.html'),fontsize=8,title=f'roi matrix of {folder_name} ({site})',showfig=self.showfig)
        sv.VisConnMat(distMat_type_break,os.path.join(rpath,'roi_type_break.html'),fontsize=8,title=f'roi matrix of {folder_name} ({site}), grouped by type, breaks data > {break_threshod}',showfig=self.showfig)
        sv.VisConnMat(distMat_break,os.path.join(rpath,'roi_mat_break.html'),fontsize=8,title=f'roi matrix of {folder_name} ({site}), breaks data > {break_threshod}',showfig=self.showfig)
        print('Done\n')
    
    def SynapseDistribution(self, requiredNeurons=None, folder_name=None, site='post', snp_rois=None, categories=['type'], info_df = pd.DataFrame()):
        """get and synapse distribution, adapted from PlotSynapses.ipynb
        Args:
            requiredNeurons (_type_, optional): _description_. Defaults to None.
            folder_name (_type_, optional): _description_. Defaults to None.
            site (str, optional): _description_. 'pre' or 'post' synaptic site, Defaults to 'post'.
            snp_rois (_type_, optional): _description_. Defaults to None (auto-generated roi list), if given, use the given roi list.
            visualization_threshod (_type_, optional): _description_. Defaults to 1e2. synaptic number threshold for auto-generated roi list
            categories (list, optional): _description_. Defaults to ['type']. other options can be used if info_df is given.
            info_df (pd.DataFrame, optional): _description_. Defaults to pd.DataFrame(). neuron info dataframe, including given categories of classified neurons.
        """        
        
        para_dict = {
            'neurons': str(requiredNeurons),
            'name': str(folder_name),
            'site': site,
            'snp_rois': snp_rois,
            'dataset': self.dataset,
            'server': self.server,
            'run date': self.run_date,
        }
        if requiredNeurons == None:
            requiredNeurons = self.sourceNeurons
            para_dict.update({'neurons': str(requiredNeurons)})
        required_criteria, auto_name = sv.getCriteriaAndName(requiredNeurons)
        if folder_name == None or folder_name == '':
            folder_name = auto_name
            para_dict.update({'name': folder_name})
        rpath = os.path.join(self.data_folder, '_'.join(['synapse_distribution',folder_name,site]))
        if not os.path.exists(rpath): os.makedirs(rpath)
        
        neuron_info_path = os.path.join(rpath,'neuron_info_'+folder_name+'.xlsx')
        if not os.path.isfile(neuron_info_path):
            print('fetching neurons...')
            noi_df, roic_df = fetch_neurons(required_criteria) # neurons of interest, roi_count
            with pd.ExcelWriter(neuron_info_path) as w:
                noi_df.to_excel(w,sheet_name='neuron_df')
                roic_df.to_excel(w,sheet_name='roi_count')
            print('fetched %d neurons'%(len(noi_df)))
        else:
            print('neuron_df existed')
            noi_df = pd.read_excel(neuron_info_path,sheet_name='neuron_df',index_col=0,header=0)
            roic_df = pd.read_excel(neuron_info_path,sheet_name='roi_count',index_col=0,header=0)
            print('read %d neurons'%(len(noi_df)))
        
        if snp_rois is None:
            snp_rois = roic_df.groupby(by=['roi']).sum()
            snp_rois.reset_index(inplace=True)
            snp_rois = snp_rois.sort_values(by=[site],ascending=False).iloc[:4,:]
            snp_rois = snp_rois['roi'].tolist()
            para_dict.update({'snp_rois': snp_rois})
        para_df = pd.DataFrame.from_dict(para_dict, orient='index', columns=['value'])
        snp_file_path = os.path.join(rpath,'synapse_info_' + folder_name + '.xlsx')
        sv.getSynapses(snp_file_path,noi_df) # get synapse info and write to excel file, #snp_file_path
        roi_str = '_'.join(snp_rois)
        summary_path = os.path.join(rpath,'summary_' + folder_name + '_' + roi_str + '_' + site + '.xlsx')
        snp_summary_df = sv.sumSnpInfo(noi_df,para_df,summary_path,snp_file_path,site=site,snp_rois=snp_rois,info_df=info_df)
        
        # plot synapse distribution in each roi in the #para_dict['snp_rois]
        site_info = str(para_dict['site'])
        save_path = os.path.join(rpath,site_info)
        print("current path to save data: ", save_path)
        if not os.path.exists(save_path): os.makedirs(save_path)
        for roi in para_dict['snp_rois']:
            print()
            summary_path = os.path.join(rpath,'_'.join(['summary',folder_name,roi,para_dict['site']]) + '.xlsx')
            print("current summary path: ", summary_path)
            snp_summary_df = sv.sumSnpInfo(noi_df,para_df,summary_path,snp_file_path,snp_rois=roi,site=para_dict['site'],info_df=info_df)
            pic_names = ['_'.join([folder_name,site_info,roi,suf]) for suf in categories]
            show_mesh_rois = self.default_mesh_rois + [roi]
            show_mesh_rois = sorted(list(set(show_mesh_rois)))
            for i,cla in enumerate(categories):
                sv.Vis3S(data_df=snp_summary_df,
                    save_path=os.path.join(save_path,pic_names[i]),
                    title=pic_names[i],
                    classby=cla,
                    toPlot='synapse_distribution',
                    mesh_roi=show_mesh_rois,
                    site=para_dict['site'],
                    snp_rois=roi,
                    )
        sv.ConcatenateIMG2PDF(save_path)
        
        # plot soma locations
        save_path = os.path.join(rpath,'soma_location')
        print("current path to save data: ", save_path)
        if not os.path.exists(save_path): os.makedirs(save_path)
        pic_names = [folder_name+'_soma_'+suf for suf in categories]
        if para_dict['snp_rois'] != None:
            show_mesh_rois = self.default_mesh_rois + para_dict['snp_rois']
        else:
            show_mesh_rois = self.default_mesh_rois
        show_mesh_rois = sorted(list(set(show_mesh_rois)))
        show_mesh_rois = self.default_mesh_rois
        for i,cla in enumerate(categories):
            sv.Vis3S(data_df=snp_summary_df,
                save_path=os.path.join(save_path,pic_names[i]),
                title=pic_names[i],
                classby=cla,
                toPlot='soma',
                mesh_roi=show_mesh_rois,
                **para_dict)
        sv.ConcatenateIMG2PDF(save_path)
        
        # plot synapse locations
        site_info = str(para_dict['site'])
        save_path = os.path.join(rpath,site_info+'_synpases')
        print("current path to save data: ", save_path)
        if not os.path.exists(save_path): os.makedirs(save_path)
        pic_names = [folder_name+'_snp_'+site_info+'_'+suf for suf in categories]
        if para_dict['snp_rois'] != None:
            show_mesh_rois = self.default_mesh_rois + para_dict['snp_rois']
        else:
            show_mesh_rois = self.default_mesh_rois
        show_mesh_rois = sorted(list(set(show_mesh_rois)))
        show_mesh_rois = self.default_mesh_rois
        for i,cla in enumerate(categories):
            sv.Vis3S(data_df=snp_summary_df,
                save_path=os.path.join(save_path,pic_names[i]),
                synapse_file_path=snp_file_path,
                title=pic_names[i],
                classby=cla,
                toPlot='synapse',
                mesh_roi = show_mesh_rois,
                site='pre',
                confidence=0,
                snp_rois=None,
                dpi=600,
                )
        sv.ConcatenateIMG2PDF(save_path)
    
    def VisualizeSelectedPaths(self, 
                              path_file,
                              sheet_name=None,
                              output_folder=None,
                              source_color=None,
                              intermediate_color=None,
                              target_color=None,
                              link_color=None,
                              node_color=None,  # For backward compatibility
                              network_layout='hierarchical',
                              showfig=False):
        '''
        Visualize selected paths from CSV/Excel file using Sankey diagram and interactive network.
        
        This is a convenience wrapper that uses the VisualizePath class for visualization.
        The VisualizePath class can also be used independently without initializing FindNeuronConnection.
        
        Parameters:
        -----------
        path_file : str or pd.DataFrame
            Path to CSV or Excel file, or DataFrame containing path data.
            Required columns:
            - 'path_block': Path in format 'A -> B -> C -> D'
            - 'weights': List of synapse numbers for each hop, e.g., [10, 20, 15]
            - 'connection_ratios': List of ratios for each hop (optional)
            - 'traversal_probabilities': List of probabilities for each hop (optional)
            
        sheet_name : str, optional
            Sheet name if reading from Excel file. Options:
            - 'path_type': For type-level paths (default if exists)
            - 'path_bodyId': For bodyId-level paths
            - Custom sheet name
            If None, will auto-detect 'path_type' or 'path_bodyId'
            
        output_folder : str, optional
            Folder to save visualizations. If None, uses './selected_paths'
            
        source_color : str, optional
            Color for source nodes. Defaults to self.source_color if available.
            
        intermediate_color : str, optional
            Color for intermediate nodes. Defaults to self.intermediate_color if available.
            
        target_color : str, optional
            Color for target nodes. Defaults to self.target_color or '#d62728'
            
        link_color : str, optional
            Color for connections. Defaults to self.link_color or 'rgba(100,100,100,0.3)'
            
        node_color : list, optional
            [DEPRECATED] Colors for nodes [source_color, intermediate_color].
            Use source_color and intermediate_color instead.
            Kept for backward compatibility.
            
        network_layout : str, optional
            Layout algorithm for network: 'hierarchical', 'spring', 'circular', 'distributed'
            Default: 'hierarchical'
            
        showfig : bool, optional
            Whether to open visualizations in browser. Default: False
            
        Returns:
        --------
        tuple: (conn_df, G_network)
            - conn_df: DataFrame with connection information
            - G_network: NetworkX graph object
            
        Example:
        --------
        >>> fc = FindNeuronConnection(...)
        >>> # After running FindAllPath, select interesting paths and save to Excel
        >>> # Then visualize them:
        >>> conn_df, G = fc.VisualizeSelectedPaths(
        ...     path_file='selected_paths.xlsx',
        ...     sheet_name='path_type',
        ...     output_folder='./selected_visualization'
        ... )
        
        Notes:
        ------
        For more control and standalone usage, you can use VisualizePath directly:
        >>> from vispath_pkg import VisualizePath
        >>> vp = VisualizePath('path_type.xlsx')
        >>> conn_df, G = vp.visualize()
        '''
        
        # Set default colors from class attributes if available
        # Support both new API (source_color, intermediate_color) and old API (node_color)
        if source_color is None and hasattr(self, 'source_color'):
            source_color = self.source_color
        if intermediate_color is None and hasattr(self, 'intermediate_color'):
            intermediate_color = self.intermediate_color
        if target_color is None and hasattr(self, 'target_color'):
            target_color = self.target_color
        if link_color is None and hasattr(self, 'link_color'):
            link_color = self.link_color
        
        # Backward compatibility: if node_color not provided but class has it
        if node_color is None and hasattr(self, 'node_color'):
            node_color = self.node_color
        
        # Create VisualizePath instance and run visualization
        vp = VisualizePath(
            path_file=path_file,
            sheet_name=sheet_name,
            output_folder=output_folder,
            source_color=source_color,
            intermediate_color=intermediate_color,
            target_color=target_color,
            link_color=link_color,
            node_color=node_color,  # Pass for backward compatibility
            network_layout=network_layout,
            showfig=showfig,
            edgeN_limit=self.edgeN_limit if hasattr(self, 'edgeN_limit') else 500,
            verbose=(self.verbose_mode == 'full') if hasattr(self, 'verbose_mode') else True
        )
        
        return vp.visualize()

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

    data_folder: str = os.path.join(script_path, 'connection_data')
    '''folder to save all data'''
    
    save_folder: str = ''
    '''
    folder to save the current data
    # initialized in __post_init__, not customizable
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

    neuron_alpha: float = 0.2
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
    'normal': show legend for individual neurons\n
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
        print(f'\n' + '='*70)
        print(f'Available ROIs for {self.dataset}')
        print('='*70)
        
        rois = self._get_available_rois(use_cache=not refresh, fetch_online=fetch_online)
        
        if rois:
            print(f'\n📊 Total: {len(rois)} ROIs')
            print(f'\n🔹 First 30 ROIs:')
            for i in range(0, min(30, len(rois)), 5):
                print('  ', ', '.join(rois[i:i+5]))
            if len(rois) > 30:
                print(f'  ... and {len(rois) - 30} more')
            print(f'\n💡 Use these ROI names in the mesh_roi parameter')
            print('='*70)
        else:
            print('⚠️  No ROIs found')
            print('='*70)
        
        return rois
    
    def __post_init__(self):
        # Initialize list to store meshes for export
        self.exportable_meshes = []
        
        # Auto-detect client_type from dataset if not explicitly set to flywire
        if self.client_type == 'neuprint' and ('flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower()):
            self.client_type = 'flywire'
            print(f"Auto-detected client_type='flywire' from dataset '{self.dataset}'")

        # Force disable caching for FlyWire/FAFB
        if self.client_type == 'flywire' or 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower():
            if self.cache_neurons:
                print(" Disabling neuron skeleton caching for FlyWire/FAFB (files too large)")
                self.cache_neurons = False
            if self.cache_synapses:
                print(" Disabling synapse caching for FlyWire/FAFB (files too large)")
                self.cache_synapses = False

        # Auto-detect version from dataset if not provided
        if self.client_type == 'flywire' and self.version is None:
            import re
            # Look for v783 or version 783
            match = re.search(r'v(\d+)', self.dataset)
            if match:
                self.version = int(match.group(1))
                print(f"Auto-detected version={self.version} from dataset '{self.dataset}'")

        # Initialize client if needed
        if self.client_type == 'neuprint':
            import neuprint
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
                    print(f'Client initialized for {self.dataset}')
                elif os.environ.get('NEUPRINT_APPLICATION_CREDENTIALS'):
                    # Auto-detect from env
                    self.client = Client(self.server, dataset=self.dataset)
                    self.client.fetch_version()
                    print(f'Client initialized from env for {self.dataset}')
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
                print("\n\033[31mCRITICAL ERROR: FlyWire/BANC data preparation failed.\033[0m")
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
                print('⚠️  brain_mesh reset to "none" due to missing transforms')
        
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
                print('\033[33mSynapse size is too small (< 20) for sphere, cone, or tetrahedron mode, automatically reset to 20\033[0m')
            if self.use_size_slider:
                self.use_size_slider = False
                print('\033[33msize slider is only available for synapse_mode="scatter", automatically reset use_size_slider to False\033[0m')
            
        if self.mesh_roi == None:
            self.mesh_roi = []
        
        if len(self.neuron_layers) <= len(self.neuron_colors): 
            self.neuron_colors = self.neuron_colors[:len(self.neuron_layers)]
            self.synapse_colors = self.synapse_colors[:len(self.neuron_layers)-1]

        # Validate brain_mesh options
        if self.brain_mesh == 'hemi':
            if 'hemibrain' not in self.dataset.lower():
                print('\033[33m⚠️  brain_mesh="hemi" only works with hemibrain:v1.2.1 dataset')
                print('   VNC datasets (manc, male-cns) do not support hemisphere mode')
                print('   Automatically switching to brain_mesh="whole"\033[0m')
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
            print(f'fetching neuron info of layer {i}...')
            layer_input = self.neuron_layers[i]
            if not isinstance(layer_input, list):
                layer_input = [layer_input]
            ndf, rdf, auto_name, cri = sv.getNeurons(layer_input, dataset=self.dataset)
            self.neuron_dfs.append(ndf)
            self.roi_dfs.append(rdf)
            self.layer_criteria.append(cri)
            self.layer_names.append(auto_name)
        print('Fetched neuron layers')

        
        if self.custom_layer_names:
            self.layer_names = self.custom_layer_names
        if self.saveas is None:
            self.saveas = '_'.join(self.layer_names)
        
        # Create timestamped subfolder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_folder = os.path.join(self.data_folder, 'plot3d_' + self.saveas.split('.')[0] + '_' + timestamp)
        if not os.path.exists(self.save_folder): os.makedirs(self.save_folder)
        
        # Save parameters to text file
        param_file = os.path.join(self.save_folder, 'parameters.txt')
        with open(param_file, 'w') as f:
            f.write(f"Dataset: {self.dataset}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Neuron Layers: {self.neuron_layers}\n")
            f.write(f"Layer Names: {self.layer_names}\n")
            f.write(f"Min Synapse Num: {self.min_synapse_num}\n")
            f.write(f"Synapse Mode: {self.synapse_mode}\n")
            f.write(f"Synapse Size: {self.synapse_size}\n")
            f.write(f"Skeleton Mode: {self.skeleton_mode}\n")
            f.write(f"Brain Mesh: {self.brain_mesh}\n")
            f.write(f"Mesh ROI: {self.mesh_roi}\n")
            f.write(f"Backend: {self.backend}\n")
            f.write(f"Client Type: {self.client_type}\n")
            if self.version:
                f.write(f"Version: {self.version}\n")
        
        if self.backend == 'plotly':
            self.fig_3d = go.Figure()
        elif self.backend == 'k3d':
            try:
                import k3d
                self.fig_3d = k3d.plot()
            except ImportError:
                print("⚠️  k3d not installed. Please install it with `pip install k3d`")
                print("   Falling back to plotly backend")
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
                    print(f'  ⚠ Failed to load cached skeleton {bid}: {e}')
                    missing_ids.append(bid)
            else:
                missing_ids.append(bid)
        
        if neurons:
            print(f'  ✓ Loaded {len(neurons)} neurons from cache')
            if missing_ids:
                print(f'  ℹ  {len(missing_ids)} neurons not in cache, will fetch')
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
                    print(f'  ⚠ Failed to save skeleton {bid}: {e}')
        
        if saved_count > 0:
            print(f'  💾 Saved {saved_count} new neurons to cache')
    
    def plot_skeleton(self):
        for i in range(len(self.neuron_layers)):
            print(f'fetching skeletons of layer {i}...')
            
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
                            print(f"  Loading skeletons from Zip: {zip_path}...")
                            
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
                                            print(f"    Warning: Skeleton {filename} not found in zip")
                                    except Exception as e:
                                        print(f"    Error reading {filename}: {e}")
                            
                            if neurons:
                                neuron_vols = navis.NeuronList(neurons)
                                print(f"  ✓ Loaded {len(neurons)} skeletons from local zip")
                    except ImportError:
                        pass
                    except Exception as e:
                        print(f"  Warning: Error loading local FAFB skeletons: {e}")

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
                            neuron_vols = neu.fetch_skeletons(missing_df, with_synapses=self.show_connectors)
                
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
            
            if neuron_vols is None or len(neuron_vols) == 0:
                print(f'⚠️  Failed to fetch skeletons for layer {i}')
                continue

            if self.brain_mesh in ['whole', 'template']:
                template_info = self._get_template_info()
                print(f'Transforming skeletons of layer {i} to {template_info["mesh_name"]}...', end='')
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
                                    print(f"  Skeleton coords range (nm): X[{n.nodes.x.min():.1f}, {n.nodes.x.max():.1f}], Y[{n.nodes.y.min():.1f}, {n.nodes.y.max():.1f}], Z[{n.nodes.z.min():.1f}, {n.nodes.z.max():.1f}]")
                    elif hasattr(neuron_vols, 'nodes') and isinstance(neuron_vols.nodes, pd.DataFrame):
                         for col in ['x', 'y', 'z']:
                            if col in neuron_vols.nodes.columns:
                                neuron_vols.nodes[col] = neuron_vols.nodes[col].astype('float64')
                         print(f"  Skeleton coords range (nm): X[{neuron_vols.nodes.x.min():.1f}, {neuron_vols.nodes.x.max():.1f}], Y[{neuron_vols.nodes.y.min():.1f}, {neuron_vols.nodes.y.max():.1f}], Z[{neuron_vols.nodes.z.min():.1f}, {neuron_vols.nodes.z.max():.1f}]")

                    neuron_vols = navis.xform_brain(neuron_vols, source=template_info['source'], target=template_info['target'])
                except Exception as e:
                    print(f'\n⚠️  Transforming skeletons failed: {e}')
                    if self._dataset_needs_transform() and not self._check_and_download_transforms():
                        self.brain_mesh = 'none'
                    else:
                        # Retry transformation after download
                        try:
                            neuron_vols = navis.xform_brain(neuron_vols, source=template_info['source'], target=template_info['target'])
                            print('✓ Transformation successful after download')
                        except Exception as retry_e:
                            print(f'⚠️  Transformation still failed: {retry_e}')
                            print('   Setting brain_mesh to "none"')
                            self.brain_mesh = 'none'
            
            # Mirror neurons if requested
            if self.mirror_on_contralateral:
                print(f'Mirroring {len(neuron_vols)} neurons...', end='')
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
                        print(' (mirrored) ', end='')
                    else:
                        print(' (mirror skipped: unknown template) ', end='')
                except Exception as e:
                    print(f' (mirror failed: {e})', end='')

            # Simplify individual neurons if requested (and not merging)
            # If merging is enabled, simplification is handled during the merge process
            if self.skeleton_mesh_simplification > 0 and self.skeleton_mode == 'tube' and not self.merge_neurons:
                print(f'Simplifying {len(neuron_vols)} neurons ({self.skeleton_mesh_simplification*100:.0f}%)...', end='')
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
                    print(' Done')
                except Exception as e:
                    print(f' (simplification failed: {e})', end='')

            # Merge neurons if requested (optimization)
            num_neurons = len(neuron_vols) if isinstance(neuron_vols, (list, navis.NeuronList)) else 1
            if self.merge_neurons and num_neurons > 1:
                print(f'Merging {num_neurons} neurons into single object...', end='')
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
                                        print(f' (simplification failed: {e})', end='')
                            
                            # Convert back to navis object
                            neuron_vols = navis.MeshNeuron(merged_mesh)
                            neuron_vols.name = self.layer_names[i]
                            print(' (merged) ', end='')
                        else:
                            print(' (merge failed: no meshes generated) ', end='')
                    else:
                        # For line mode, we can merge traces later in plotting?
                        # Actually, navis.plot3d returns a figure with traces.
                        # We can merge them there.
                        print(' (will merge traces in plot) ', end='')
                except Exception as e:
                    print(f'⚠️  Merge failed: {e}, plotting individually')

            print('plotting...', end='')
            
            if self.backend == 'plotly':
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
                    print(f'⚠️  k3d plotting failed: {e}')

            print('Done')
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
                    print(f'  ✓ Loaded synapse table from {synapse_table_path}')
                    
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
                    print(f'  ✓ Filtered to {len(filtered_df)} synapses between {len(source_ids)} sources and {len(target_ids)} targets')
                    # For FlyWire, master table has all data - no missing pairs
                    return filtered_df, []
                except Exception as e:
                    print(f'  ⚠ Failed to load synapse table: {e}')
                    all_pairs = [(s, t) for s in source_ids for t in target_ids]
                    return None, all_pairs
            else:
                print(f'  ⚠ Synapse table not found at {synapse_table_path}')
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
                        print(f'  ⚠ Cache load failed for {pre_id}→{post_id}: {e}')
                        missing_pairs.append((pre_id, post_id))
                else:
                    missing_pairs.append((pre_id, post_id))
        
        if cached_dfs:
            cached_df = pd.concat(cached_dfs, ignore_index=True)
            print(f'  ✓ Loaded {len(cached_df)} synapses from cache ({len(cached_dfs)} pairs cached, {len(missing_pairs)} pairs missing)')
        else:
            cached_df = None
            
        return cached_df, missing_pairs
    
    def _save_cached_synapses(self, conn_df):
        """Save synapse connections to cache, organized by pre/post neuron pairs.
        
        Each unique (pre_id, post_id) pair gets its own cache file at:
            cache/{dataset}/synapses/{pre_id}_{post_id}.parquet
            
        This approach ensures:
        1. Synapses are cached by their actual content (neuron pairs + positions)
        2. Same synapse data is reusable across different queries/layers
        3. Incremental caching - only fetch what's not already cached
        """
        if not self.cache_synapses:
            return
            
        # Do not cache for FlyWire/FAFB - they use the master synapse table
        if 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower():
            return
        
        if conn_df is None or conn_df.empty:
            return
            
        # Determine column names for pre/post body IDs
        pre_col = 'bodyId_pre' if 'bodyId_pre' in conn_df.columns else 'pre_pt_root_id'
        post_col = 'bodyId_post' if 'bodyId_post' in conn_df.columns else 'post_pt_root_id'
        
        if pre_col not in conn_df.columns or post_col not in conn_df.columns:
            print(f'  ⚠ Cannot cache synapses: missing {pre_col} or {post_col} columns')
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
            except Exception as e:
                print(f'  ⚠ Cache save failed for {pre_id}→{post_id}: {e}')
        
        print(f'  💾 Saved synapses to cache ({saved_count} neuron pairs)')
    
    def plot_synapses(self):
        if self.skip_synapse:
            print('Skipping synapse plotting as requested.')
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
                        print(f'  Reading synapses from {parquet_file} (Parquet)...')
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
                                print(f"  ⚠️ Missing coordinate columns in Parquet: {missing_coords}")
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
                                    print("  ✓ Found alternative coordinate columns")
                                    columns = list(found_map.values()) + [pre_col, post_col]
                                    df = pd.read_parquet(parquet_file, columns=columns)
                                    # Rename to standard
                                    inv_map = {v: k for k, v in found_map.items()}
                                    df = df.rename(columns=inv_map)
                                else:
                                    print("  ❌ Could not resolve all coordinate columns. Skipping.")
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
                                        print('  ✓ Detected coordinates in nanometers (no scaling applied)')
                                    else:
                                        print('  ✓ Detected coordinates in voxels (scaling 4x4x40)')
                                        conn_df['x_pre'] = conn_df['x_pre'] * 4
                                        conn_df['y_pre'] = conn_df['y_pre'] * 4
                                        conn_df['z_pre'] = conn_df['z_pre'] * 40
                                        conn_df['x_post'] = conn_df['x_post'] * 4
                                        conn_df['y_post'] = conn_df['y_post'] * 4
                                        conn_df['z_post'] = conn_df['z_post'] * 40

                                    print(f'  ✓ Found {len(conn_df)} synapses in Parquet file')
                                else:
                                    print('  No matching synapses found in Parquet file')
                                    conn_df = None
                        else:
                            print("  ⚠️ Could not find root_id columns in Parquet schema")
                            conn_df = None
                    except Exception as e:
                        print(f'  ⚠️ Failed to read Parquet file: {e}')
                        conn_df = None
                else:
                    # Fallback or warning
                    print(f"  ℹ️  Synapse table not found: {parquet_file}")
                    print("     If you have the raw CSV, please ensure FAFB_file_converter has run successfully.")
                    conn_df = None

                
                # Fallback to client if local failed or returned nothing
                if conn_df is None and self.client_flywire:
                    print(f"\n  ⚠️  Local synapse file not found for dataset '{self.dataset}'.")
                    if 'fafb' in self.dataset.lower():
                        print("  Please download the synapse table from: https://codex.flywire.ai/api/download?dataset=fafb")
                    print(f"  Save the file to: {dataset_dir}")
                    print("  Skipping synapse plotting for this layer.")
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
                    print(f'  Fetching {len(missing_pairs)} missing neuron pairs from NeuPrint...')
                    fetched_df = fetch_synapse_connections(
                        source_criteria=source_criteria,
                        target_criteria=target_criteria,
                        min_total_weight=self.min_synapse_num,
                        synapse_criteria=self.synapse_criteria,
                    )
                    if fetched_df is not None and not fetched_df.empty:
                        conn_df = pd.concat([cached_df, fetched_df], ignore_index=True)
                        # Save newly fetched data to cache
                        self._save_cached_synapses(fetched_df)
                    else:
                        conn_df = cached_df
                else:
                    # No cache - fetch all
                    conn_df = fetch_synapse_connections(
                        source_criteria=source_criteria,
                        target_criteria=target_criteria,
                        min_total_weight=self.min_synapse_num,
                        synapse_criteria=self.synapse_criteria,
                    )
                    # Save to cache
                    if conn_df is not None and not conn_df.empty:
                        self._save_cached_synapses(conn_df)
        
            if conn_df is None or conn_df.empty:
                print('  No synapses found.')
                continue

            # Check if file exists to determine mode (handle skipped layers)
            if os.path.exists(file_path):
                mode = 'a'
            else:
                mode = 'w'
                
            with pd.ExcelWriter(file_path, mode=mode, engine='openpyxl') as writer:
                conn_df.to_excel(writer, sheet_name=f'conn_df{i}_{i+1}')
            
            print('plotting...', end='')
            
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
                    print(f'Transforming synapses of layer {i} -> {i+1}...', end='')
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
                        print(f'⚠️  k3d synapse plotting failed: {e}')
            
            elif self.synapse_mode in ['sphere', 'cone', 'tetrahedron'] and self.backend == 'plotly':
                pre_coords = conn_df[['x_pre', 'y_pre', 'z_pre']].rename(columns={'x_pre':'x', 'y_pre':'y', 'z_pre':'z'})
                post_coords = conn_df[['x_post', 'y_post', 'z_post']].rename(columns={'x_post':'x', 'y_post':'y', 'z_post':'z'})
                
                if self.brain_mesh in ['whole', 'template']:
                    template_info = self._get_template_info()
                    print(f'Transforming synapses of layer {i} -> {i+1}...', end='')
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
            print('Done')
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
                    print(f'✓ Loaded {len(cached_data)} available ROIs from cache')
                    return cached_data
            except Exception as e:
                print(f'⚠️ Failed to load ROI cache: {e}, fetching from API...')
        
        # Fetch from NeuPrint API
        if fetch_online:
            # Special handling for FlyWire/FAFB: Do not use API, use local primary_rois or hemibrain cache
            if 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower():
                print('ℹ️  FlyWire/FAFB dataset detected: Skipping online API fetch for ROIs.')
                print('   Scanning local ROI meshes...')
                
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
                print(f'✓ Found {len(roi_list)} available ROIs from local storage')
                
                # Cache the results
                if roi_list:
                    try:
                        import json
                        os.makedirs(cache_dir, exist_ok=True)
                        with open(cache_file, 'w') as f:
                            json.dump(roi_list, f, indent=2)
                    except Exception as e:
                        print(f'⚠️ Failed to cache ROI list: {e}')
                        
                return roi_list

            try:
                print('📥 Fetching available ROIs from NeuPrint online database...')
                
                # Initialize neuprint client using environment variable or global client
                from neuprint import Client, fetch_meta
                
                # Try to get token from environment variable first
                token = os.environ.get('NEUPRINT_APPLICATION_CREDENTIALS')
                client = None
                
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
                        print(f'   Warning: Failed to create client with token: {e}')
                        print(f'   Attempting to use default/global client...')
                        client = None
                
                # Fetch metadata (will use client if provided, otherwise global)
                meta = fetch_meta(client=client)
                
                roi_list = []
                # Extract ROI list from meta info
                if 'roiInfo' in meta:
                    roi_list = list(meta['roiInfo'].keys())
                    print(f'   Found {len(roi_list)} ROIs from roiInfo')
                elif 'primaryRois' in meta:
                    roi_list = list(meta['primaryRois'])
                    print(f'   Found {len(roi_list)} primary ROIs')
                else:
                    print(f'   Warning: No roiInfo/primaryRois in metadata, falling back to local cache')
                
                roi_list = sorted(roi_list)
                
                # Cache the results (create directory only when needed)
                if roi_list:
                    try:
                        import json
                        os.makedirs(cache_dir, exist_ok=True)
                        with open(cache_file, 'w') as f:
                            json.dump(roi_list, f, indent=2)
                        print(f'✓ Cached {len(roi_list)} available ROIs to {cache_file}')
                    except Exception as e:
                        print(f'⚠️ Failed to cache ROI list: {e}')
                
                return roi_list
                
            except Exception as e:
                print(f'⚠️ Failed to fetch available ROIs from NeuPrint: {e}')
                print(f'   Tip: Set NEUPRINT_APPLICATION_CREDENTIALS environment variable')
                print(f'   Using ROIs from local mesh directory instead.')
        
        # Fallback: list available meshes from local directory
        mesh_dir = self._get_dataset_mesh_dir()
        if os.path.exists(mesh_dir):
            roi_list = [f.replace('.json', '') for f in os.listdir(mesh_dir) if f.endswith('.json')]
            roi_list = sorted(roi_list)
            print(f'✓ Found {len(roi_list)} ROIs in local cache: {mesh_dir}')
            
            # Cache the results from local scan
            if roi_list:
                try:
                    import json
                    os.makedirs(cache_dir, exist_ok=True)
                    with open(cache_file, 'w') as f:
                        json.dump(roi_list, f, indent=2)
                    print(f'✓ Cached {len(roi_list)} available ROIs to {cache_file}')
                except Exception as e:
                    print(f'⚠️ Failed to cache ROI list: {e}')
            
            return roi_list
        else:
            print(f'⚠️ No ROI data available (online fetch failed and no local cache)')
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
            print(f'⚠️  Unknown dataset "{self.dataset}", defaulting to hemibrain template')
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
        try:
            import flybrains
            
            # Get the transform directory from attribute or use default
            transforms_dir = os.path.expanduser(self.transforms_dir)
            
            # Set environment variable if custom path is specified
            if self.transforms_dir != '~/flybrain-data':
                os.environ['FLYBRAINS_DATA'] = transforms_dir
                print(f'Using custom transform directory: {transforms_dir}')
            
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
                print(f'✓ Brain transforms already available')
                print(f'  Location: {YELLOW}{transforms_dir}{RESET}')
                print(f'  Transform path: {" -> ".join([str(p) for p in path])}')
                return True
            except (ValueError, KeyError):
                # Transform path not found, need to download
                pass
            
            # ANSI color codes
            YELLOW = '\033[93m'
            RESET = '\033[0m'
            
            # Prompt user for download confirmation
            print('\n' + '='*70)
            print('⚠️  Brain Transformation Required')
            print('='*70)
            print(f'To use brain_mesh="whole" for {self.dataset}, you need brain transforms.')
            print(f'Transform path needed: {source} → JRCFIB2018F → JRCFIB2018Fum → {target}')
            print('')
            print('⚠️  IMPORTANT: flybrains downloads ALL JRC transforms as a bundle:')
            print('   • JRC2018F_JRCFIB2018F.h5   (~1.29 GB)  ← YOU NEED THIS for hemibrain/optic-lobe')
            print('   • JRC2018F_FAFB.h5          (~580 MB)   (enables FAFB dataset support)')
            print('   • JRC2018F_JFRC2013.h5      (~1.39 GB)  (enables JFRC2013 template)')
            print('   • JRC2018F_FCWB.h5          (~1.29 GB)  (enables FCWB template)')
            print('   • JRC2018U_JRC2018F.h5      (~717 MB)   (enables unisex template)')
            print('   • JRC2018U_JRC2018M.h5      (~1.10 GB)  (enables male template)')
            print('   • JRC2018F_JFRC2010.h5      (~1.65 GB)  (enables legacy template)')
            print('   • JRCFIB2022M_JRC2018M.h5   (~2.12 GB)  (enables male CNS registration)')
            print('')
            print('   Total download: ~10 GB (but only ~1.3 GB used for your dataset)')
            print('   Download time: ~1-2 hours (cannot download individual files)')
            print('   Why all files? The flybrains package bundles all transforms together.')
            print('')
            print('The transforms will be cached in:')
            print(f'  {YELLOW}{transforms_dir}/{RESET}')
            
            # Save transform path info to file
            info_file = os.path.join(self.data_folder, 'brain_transforms_info.txt')
            os.makedirs(self.data_folder, exist_ok=True)
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write('Brain Transforms Information\n')
                f.write('='*70 + '\n\n')
                f.write(f'Dataset: {self.dataset}\n')
                f.write(f'Transform path: {source} → JRCFIB2018F → JRCFIB2018Fum → {target}\n\n')
                f.write('Storage Location:\n')
                f.write(f'  {transforms_dir}/\n\n')
                f.write('Transform Files (8 files, ~10 GB total):\n')
                f.write('  • JRC2018F_JRCFIB2018F.h5   (~1.29 GB)\n')
                f.write('  • JRC2018F_FAFB.h5          (~580 MB)\n')
                f.write('  • JRC2018F_JFRC2013.h5      (~1.39 GB)\n')
                f.write('  • JRC2018F_FCWB.h5          (~1.29 GB)\n')
                f.write('  • JRC2018U_JRC2018F.h5      (~717 MB)\n')
                f.write('  • JRC2018U_JRC2018M.h5      (~1.10 GB)\n')
                f.write('  • JRC2018F_JFRC2010.h5      (~1.65 GB)\n')
                f.write('  • JRCFIB2022M_JRC2018M.h5   (~2.12 GB)\n\n')
                f.write('To change the storage location:\n')
                f.write('  1. Set transforms_dir attribute when creating VisualizeSkeleton\n')
                f.write('  2. Set FLYBRAINS_DATA environment variable before importing flybrains\n')
                f.write('  3. Or manually move files to the new location\n\n')
                f.write('More information:\n')
                f.write('  https://github.com/navis-org/navis-flybrains\n')
            print(f'\n📄 Transform info saved to: {info_file}')
            print('')
            print('💡 Note: The flybrains.download_jrc_transforms() function downloads')
            print('   ALL 8 files as a bundle with no selective download option.')
            print('   This is by design in the flybrains library to provide complete')
            print('   cross-dataset registration capabilities.')
            print('')
            print('For more information, see:')
            print('  https://github.com/navis-org/navis-flybrains')
            print('='*70)
            
            response = input('Download all transforms now? [y/N]: ').strip().lower()
            
            if response in ['y', 'yes']:
                print('\n📥 Downloading brain transforms...')
                print('This may take several minutes depending on your connection.')
                flybrains.download_jrc_transforms()
                
                # Re-register transforms after download
                print('📝 Registering downloaded transforms...')
                flybrains.register_transforms()
                
                # Verify the transform path is now available
                try:
                    path = navis.transforms.registry.find_bridging_path(source, target)
                    print(f'✓ Transforms downloaded and registered successfully!')
                    print(f'  Location: {YELLOW}{transforms_dir}{RESET}')
                    print(f'  Transform path: {" -> ".join([str(p) for p in path])}')
                    
                    # Update the saved info file with success status
                    info_file = os.path.join(self.data_folder, 'brain_transforms_info.txt')
                    with open(info_file, 'a', encoding='utf-8') as f:
                        f.write(f'\nDownload Status: SUCCESS\n')
                        f.write(f'Downloaded at: {pd.Timestamp.now()}\n')
                    return True
                except (ValueError, KeyError) as e:
                    print(f'⚠️  Transforms downloaded but bridging path not found: {e}')
                    print(f'   This may indicate the transforms do not include {source} → {target}')
                    return False
            else:
                print('\n⚠️  Download cancelled. Setting brain_mesh to "none".')
                return False
                
        except ImportError:
            print('\n⚠️  flybrains package not installed.')
            print('   Install it with: pip install navis[flybrains]')
            print('   Setting brain_mesh to "none".')
            return False
        except Exception as e:
            print(f'\n⚠️  Error checking brain transforms: {e}')
            print('   Setting brain_mesh to "none".')
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
        
        References:
        - navis Volume API: https://navis.readthedocs.io/en/latest/source/api.html#navis.Volume
        - flybrains templates: https://github.com/navis-org/navis-flybrains
        - mesh optimization: use Volume.simplify() to reduce mesh complexity for faster rendering
        """
        if self.mesh_roi is None:
            return
        
        # Ensure available_rois.json exists (generate if missing)
        # This checks cache first, and if missing, fetches from API or scans local meshes
        self._get_available_rois(use_cache=True, fetch_online=True)
        
        # Get dataset-specific mesh directory
        mesh_dir = self._get_dataset_mesh_dir()
        print(f'Using mesh directory: {mesh_dir}')
        
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
                    print(f'📥 ROI mesh "{roi}" not found locally, attempting to download...')
                    mesh_found = False
                    
                    # 1. Try male-cns:v0.9 (NeuPrint)
                    try:
                        import navis.interfaces.neuprint as neu
                        from neuprint import Client
                        
                        token = os.environ.get('NEUPRINT_APPLICATION_CREDENTIALS') or self.token
                        if token:
                            try:
                                print(f'   Checking male-cns:v0.9...')
                                mc_client = Client('https://neuprint.janelia.org', dataset='male-cns:v0.9', token=token)
                                mesh = neu.fetch_roi(roi, client=mc_client)
                                if mesh:
                                    os.makedirs(mesh_dir, exist_ok=True)
                                    mesh.to_json(mesh_file)
                                    print(f'   ✓ Found in male-cns:v0.9')
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
                    print(f'📥 ROI mesh "{roi}" not found locally, attempting to download from NeuPrint...')
                    source_info = "NeuPrint (Downloaded)"
                    try:
                        import navis.interfaces.neuprint as neu
                        from neuprint import Client
                        
                        token = os.environ.get('NEUPRINT_APPLICATION_CREDENTIALS') or self.token
                        client = None
                        
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
                                print(f'   Warning: Failed to create client: {e}')
                        
                        mesh = neu.fetch_roi(roi, client=client)
                        os.makedirs(mesh_dir, exist_ok=True)
                        mesh.to_json(mesh_file)
                        print(f'✓ Downloaded and cached "{roi}" mesh to {mesh_file}')
                        
                        # Transform if needed (Hemibrain specific)
                        if self.brain_mesh in ['whole', 'template']:
                            template_info = self._get_template_info()
                            print(f'Transforming brain region {roi}...', end='')
                            mesh = navis.xform_brain(mesh, source=template_info['source'], target=template_info['target'])
                            # Note: We don't save the transformed mesh back to cache here to keep cache pure?
                            # Actually previous code didn't save transformed.
                    except Exception as e:
                        print(f'⚠️  Failed to download "{roi}" mesh: {e}')
            
            # Load and plot
            if os.path.exists(mesh_file):
                try:
                    mesh = navis.Volume.from_json(mesh_file)
                    print(f'✓ Loaded "{roi}" from {source_info}')
                    
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
                            
                        print(f'Transforming brain region {roi} ({source} -> {target})...', end='')
                        try:
                            mesh = navis.xform_brain(mesh, source=source, target=target)
                            print(' Done')
                        except Exception as e:
                            print(f' Failed: {e}')
                    
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
                                    
                                    print(f' (simplified {self.roi_mesh_simplification*100:.0f}%: {n_faces}->{len(new_tm.faces)} faces)', end='')
                                else:
                                    print(f' (simplification skipped: target {target_faces} >= {n_faces} faces)', end='')
                            else:
                                # Debug: print available attributes to help diagnose
                                attrs = [a for a in dir(mesh) if not a.startswith('_')]
                                print(f' (simplification skipped: could not extract mesh from {type(mesh)}. Available attrs: {attrs[:10]}...)', end='')
                        except Exception as e:
                            print(f' (simplification failed: {e})', end='')

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
                            # print(f' (export skip: no mesh in {type(mesh)})', end='')
                            pass
                    except Exception as e:
                        print(f' (export collection failed: {e})', end='')

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
                                print(f' (mirror failed: {e})', end='')

                except Exception as e:
                    print(f'⚠️  Failed to load mesh {roi}: {e}')
            else:
                if not is_flywire: # Only warn if we expected to find it (FlyWire might just fail silently if not found)
                     print(f'⚠️  ROI mesh "{roi}" not found.')
        
        if not roiunits:
            print('⚠️  No valid ROI meshes loaded')
            return
        
        print('plotting mesh of brain regions...')
        for roi_i in range(len(roiunits)):
            roiunits[roi_i].color = roi_colors[roi_i]
            
            if self.backend == 'plotly':
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
                    temp_plot = navis.plot3d(roiunits[roi_i], backend='k3d', inline=False)
                    for obj in temp_plot.objects:
                        obj.name = f'brain regions [{roi_names[roi_i]}...]'
                        self.fig_3d += obj
                except Exception as e:
                    print(f'⚠️  k3d mesh plotting failed: {e}')

        if self.brain_mesh in ['template', 'whole']:
            template_info = self._get_template_info()
            mesh_display_name = template_info['mesh_name']
            
            print(f'Plotting {mesh_display_name} mesh...')
            try:
                brain_template = template_info['template_obj']
                
                if self.backend == 'plotly':
                    fig_brain = navis.plot3d(brain_template, backend='plotly')
                    brain_traces = fig_brain.data
                    for trace in brain_traces:
                        trace.showlegend = True
                        trace.name = mesh_display_name
                        trace.hoverinfo = 'none'
                        trace.color = self.brain_mesh_color
                    self.fig_3d.add_traces(brain_traces)
                elif self.backend == 'k3d':
                    temp_plot = navis.plot3d(brain_template, backend='k3d', inline=False)
                    for obj in temp_plot.objects:
                        obj.name = mesh_display_name
                        self.fig_3d += obj
                        
                print(f'✓ {mesh_display_name} mesh loaded successfully')
            except Exception as e:
                print(f'⚠️  Failed to load {mesh_display_name} mesh: {e}')
                if self._dataset_needs_transform() and not self._check_and_download_transforms():
                    print('   Skipping brain/VNC mesh visualization')
                else:
                    # Retry after download
                    try:
                        brain_template = template_info['template_obj']
                        if self.backend == 'plotly':
                            fig_brain = navis.plot3d(brain_template, backend='plotly')
                            brain_traces = fig_brain.data
                            for trace in brain_traces:
                                trace.showlegend = True
                                trace.name = mesh_display_name
                                trace.hoverinfo = 'none'
                                trace.color = self.brain_mesh_color
                            self.fig_3d.add_traces(brain_traces)
                        elif self.backend == 'k3d':
                            temp_plot = navis.plot3d(brain_template, backend='k3d', inline=False)
                            for obj in temp_plot.objects:
                                obj.name = mesh_display_name
                                self.fig_3d += obj
                        print(f'✓ {mesh_display_name} mesh loaded successfully after download')
                    except Exception as retry_e:
                        print(f'⚠️  Still failed to load {mesh_display_name} mesh: {retry_e}')
                        print('   Skipping brain/VNC mesh visualization')
        print('Done')
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
            if self.brain_mesh == 'hemi' or self.brain_mesh == 'none':
                scene_camera_parameters = dict(
                    up=dict(x=0, y=0, z=-1),
                    eye=dict(x=0, y=1.8, z=0),  # Increased from 1.4 to 1.8 to fit more objects
                    # center=dict(x=0, y=0, z=0), # Let Plotly auto-center
                )
            elif self.brain_mesh == 'whole':
                # Adjust for frontal view
                # Assuming standard fly brain orientation (X: LR, Y: DV, Z: AP)
                # Frontal view: Look from Anterior (Z) or Posterior
                scene_camera_parameters = dict(
                    up=dict(x=0, y=-1, z=0), # Y is up (inverted in some templates)
                    eye=dict(x=0, y=0, z=-2.0), # Look from front/back
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
                ),
                scene_camera=scene_camera_parameters,
            )

            # save figure
            self.fig_path = os.path.join(self.save_folder,self.saveas)
            
            # Ensure save folder exists
            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder, exist_ok=True)
            
            print(f'saving figure to \033[34m{self.fig_path}.html\033[0m...', end='')
            
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
                    print(f'\n⚠️  Failed to open browser: {e}')
            
            print('Done (HTML saved)')
            
            # Optimize PNG export: only save if needed, use lower scale for speed
            try:
                print('   Exporting static PNG (may take a moment)...', end='', flush=True)
                
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
                    print(f' Done ({size/1024:.1f} KB)')
                    if size < 15 * 1024: # < 15KB is suspicious for a 3D plot
                        print('   ⚠️  Warning: Exported PNG seems blank/empty.')
                        print('       This is a known issue with Kaleido and 3D plots on some systems.')
                        print('       Please rely on the HTML file for visualization.')
                else:
                    print(' Done (File not found)')
            except Exception as e:
                print(f'\n   ⚠️  PNG export failed: {e}. Continuing without PNG...')
            
        elif self.backend == 'k3d':
            self.fig_path = os.path.join(self.save_folder,self.saveas)
            print(f'saving figure to \033[34m{self.fig_path}.html\033[0m...', end='')
            
            try:
                from ipywidgets.embed import embed_minimal_html
                embed_minimal_html(
                    self.fig_path+'.html', 
                    views=[self.fig_3d], 
                    title=self.saveas
                )
                print('Done')
                
                if self.show_fig:
                    print('Note: k3d plots cannot be automatically opened from script. Please open the HTML file manually.')
                    
            except ImportError:
                print('\n⚠️  ipywidgets not installed. Cannot save k3d plot to HTML.')
                print('   Please install it with `pip install ipywidgets`')
            except Exception as e:
                print(f'\n⚠️  Failed to save k3d plot: {e}')
    
    def plot_neurons(self):
        self.plot_skeleton()
        self.plot_synapses()
        self.plot_mesh()
        self.save_figure()
    
    def _to_rgba(self, color, alpha=None):
        """Convert color to uint8 RGBA for trimesh."""
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

    def export_3d_model(self, filename=None, format='obj'):
        """
        Export the built 3D structure (neurons + ROIs) to a 3D model file.
        
        Parameters
        ----------
        filename : str, optional
            Output filename. If None, uses self.saveas + '.' + format.
        format : str, default 'obj'
            Export format supported by trimesh (e.g., 'obj', 'stl', 'ply', 'glb').
            Note: 'glb' or 'ply' are recommended for preserving color and transparency.
            'obj' supports color via .mtl files but transparency support varies by viewer.
            'stl' does NOT support color or transparency.
        """
        if not self.exportable_meshes:
            print('⚠️  No meshes available for export. Ensure skeleton_mode="tube" and ROIs are loaded.')
            return

        if filename is None:
            filename = os.path.join(self.save_folder, f'{self.saveas}.{format}')
        
        print(f'Exporting 3D model to {filename}...')
        try:
            import trimesh
            # Concatenate all meshes
            combined_mesh = trimesh.util.concatenate(self.exportable_meshes)
            
            # Export
            combined_mesh.export(filename)
            print(f'✓ 3D model exported successfully ({len(combined_mesh.faces)} faces)')
            
            if format == 'obj':
                print('  Note: OBJ export includes a .mtl file for materials. Keep them together.')
            elif format == 'stl':
                print('  Warning: STL format does not support color or transparency.')
        except ImportError:
            print('⚠️  trimesh not installed. Cannot export 3D model.')
        except Exception as e:
            print(f'⚠️  3D model export failed: {e}')
        
    def export_video(self, fps=30, rotate_plane=None, view_direction=None, view_distance=None, synapse_size=1, 
                    html_file=None, use_existing_images=False, **kwargs):
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
        '''
        # Set default parameters based on brain_mesh
        if rotate_plane is None:
            if self.brain_mesh == 'hemi' or self.brain_mesh == 'none':
                rotate_plane = 'xy'
            elif self.brain_mesh == 'whole':
                rotate_plane = 'xz'
        if view_direction is None:
            if self.brain_mesh == 'hemi' or self.brain_mesh == 'none':
                view_direction = (1, 1)
            elif self.brain_mesh == 'whole':
                view_direction = (1, -1)
        if view_distance is None:
            if self.brain_mesh == 'hemi' or self.brain_mesh == 'none':
                view_distance = 1.8
            elif self.brain_mesh == 'whole':
                view_distance = 2.2
        
        # Set default scale if not specified
        if kwargs.get('scale') is None and kwargs.get('width') is None and kwargs.get('height') is None:
            kwargs['scale'] = 2
        
        step = 30 / fps
        
        # Load figure from existing HTML file if provided (OPTIMIZATION)
        if html_file is not None:
            print(f'📂 Loading figure from existing HTML: {html_file}')
            if not os.path.exists(html_file):
                raise FileNotFoundError(f'HTML file not found: {html_file}')
            
            # Read and parse the HTML file to extract figure data
            import plotly.io as pio
            try:
                fig_loaded = pio.read_html(html_file)
                fig_traces = fig_loaded.data
                print(f'✓ Loaded {len(fig_traces)} traces from HTML file')
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
                print(f'⚠️  Figure is large ({html_size:.1f} MB). Rendering may be slow.')
                print(f'   Consider using lower scale or smaller dimensions in kwargs.')
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
        
        # Set camera parameters
        if self.brain_mesh == 'hemi' or self.brain_mesh == 'none':
            scene_camera_parameters = dict(
                up=dict(x=0, y=0, z=-1),
                eye=dict(x=0, y=view_distance, z=0),
            )
        elif self.brain_mesh == 'whole':
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
                print(f'✓ Using {len(existing_images)} existing images from {pic_folder}')
                print(f'  Skipping image rendering (use_existing_images=True)')
            else:
                print(f'⚠️  Found {len(existing_images)} images but need {len(steps_to_write)}')
                print(f'  Re-rendering images...')
                use_existing_images = False
        else:
            use_existing_images = False
        
        # Render images if needed
        if not use_existing_images:
            if os.path.exists(pic_folder):
                shutil.rmtree(pic_folder)
            os.makedirs(pic_folder)
            
            print(f'🎬 Rendering {len(steps_to_write)} frames at {fps} fps...')
            print(f'   Resolution: scale={kwargs.get("scale", "auto")}', end='')
            if 'width' in kwargs and 'height' in kwargs:
                print(f', size={kwargs["width"]}x{kwargs["height"]}')
            else:
                print()
            
            # Try to initialize Kaleido scope for faster rendering
            # scope = None
            # try:
            #     from kaleido.scopes.plotly import PlotlyScope
            #     scope = PlotlyScope() # Use bundled plotlyjs for better stability
            #     print('   ✓ Using Kaleido scope for optimized rendering')
            # except ImportError:
            #     print('   ℹ️  Kaleido not found or failed to initialize. Using standard write_image (slower).')

            t0 = time.time()
            print(f'   Starting render loop... (First frame may take longer to initialize engine)')
            
            # Ensure dimensions are set to avoid blank images if not provided
            if 'width' not in kwargs: kwargs['width'] = 1200
            if 'height' not in kwargs: kwargs['height'] = 900

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
                
                # Write image
                try:
                    # Use standard write_image which is more reliable than Scope in some envs
                    fig_new.write_image(fig_path, **kwargs)
                except Exception as e:
                    print(f'\n⚠️  Frame {i+1} failed: {e}')
                    # If first frame fails, it's likely a system/memory issue
                    if i == 0:
                        print('   Try reducing "scale" (e.g. scale=1) or using "width"/"height" parameters.')
                        return 1
                
                ti = time.time()
                elapsed = ti - t0
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (len(steps_to_write) - i - 1)
                print(f'\r  Frame {i+1}/{len(steps_to_write)} | '
                      f'Elapsed: {elapsed:.1f}s | '
                      f'Remaining: {remaining:.1f}s | '
                      f'Speed: {avg_time:.2f}s/frame', end='    ')
            print('\n✓ Image rendering complete')
        # Generate videos from images
        print(f'\nGenerating videos...')
        imglist = os.listdir(pic_folder)
        img_eg = cv2.imread(os.path.join(pic_folder, imglist[0]))
        height, width, layers = img_eg.shape
        
        print(f'   Video resolution: {width}x{height}')

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
                print(f'\r  Forward video: {i+1}/{len(steps_to_write)} frames', end='  ')
        out.release()
        t1 = time.time()
        print(f'\n\u2713 Forward video: {video_dir} ({t1-t0:.1f}s)')
        
        # Backward video
        video_dir = os.path.join(self.save_folder, f'{self.saveas}_video_backward.mp4')
        out = cv2.VideoWriter(video_dir, fourcc, fps, frameSize=(width, height))
        
        t0 = time.time()
        for i, deg in enumerate(steps_to_write[::-1]):
            img = cv2.imread(os.path.join(pic_folder, f'deg_{deg:.1f}.jpeg'))
            out.write(img)
            if (i + 1) % 10 == 0 or i == len(steps_to_write) - 1:
                print(f'\r  Backward video: {i+1}/{len(steps_to_write)} frames', end='  ')
        out.release()
        t1 = time.time()
        print(f'\n\u2713 Backward video: {video_dir} ({t1-t0:.1f}s)')
        
        print(f'\n\u2705 Video export complete!')
        print(f'   Image cache: {pic_folder}')
        print(f'   Tip: Use use_existing_images=True to skip re-rendering next time')
        return 0
