# connectome analysis module -- coana
import os
import sys
import json
import shutil
import time
import gc
import logging
from dataclasses import dataclass, field

import cv2
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import navis
import navis.interfaces.neuprint as neu
# import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
import flybrains
# import plotly.graph_objects as go
import seaborn as sns
from tqdm import tqdm
from neuprint import *
from neuprint.utils import connection_table_to_matrix
try:
    import src.statvis_polars as svp
    from src.statvis_polars import EnrichConnectionTablePolars
except ImportError:
    import statvis_polars as svp
    from statvis_polars import EnrichConnectionTablePolars

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

try:
    from .visualize_skeleton import VisualizeSkeleton
except ImportError:
    try:
        from src.visualize_skeleton import VisualizeSkeleton
    except ImportError:
        from visualize_skeleton import VisualizeSkeleton

# Ignore the navis warning
logging.getLogger('navis').setLevel(logging.WARNING)

# ============================================================================
# Module-level cache for sharing connection data across FindNeuronConnection instances
# This avoids repeated disk reads when comparison module creates multiple instances
# Structure: {dataset: {'conn_df': DataFrame, 'conn_index': dict, 'neuron_index': DataFrame, 'neuron_dict': dict}}
# ============================================================================
_FNC_CACHE = {}

# ============================================================================
# Module-level cache for FindAllPath graph data (bodyId-level)
# Used by comparison module to skip heavy graph building when running same query at different thresholds
# Structure: {cache_key: {'threshold': int, 'graph': FastGraph, 'all_connections': list[DataFrame], 
#             'layer_neurons': list[set], 'targets_found': list, 'source_ID': list, 'target_ID': list}}
# cache_key = f"{dataset}_{source_hash}_{target_hash}_{max_interlayer}"
# ============================================================================
_FINDALLPATH_GRAPH_CACHE = {}


def clear_findallpath_cache(dataset: str = None):
    """
    Clear the module-level FindAllPath graph cache.
    
    Args:
        dataset: Specific dataset to clear. If None, clears all.
    """
    global _FINDALLPATH_GRAPH_CACHE
    if dataset is None:
        _FINDALLPATH_GRAPH_CACHE.clear()
    else:
        # Clear entries that start with the dataset name
        keys_to_delete = [k for k in _FINDALLPATH_GRAPH_CACHE if k.startswith(dataset)]
        for k in keys_to_delete:
            del _FINDALLPATH_GRAPH_CACHE[k]


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


from core.fast_graph import FastGraph


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
        # Use tqdm.write when inside a progress bar to avoid disrupting the bar
        def _do_print(msg, end=end, flush=flush):
            if getattr(self, '_in_progress_bar', False):
                from tqdm import tqdm
                # tqdm.write doesn't support end/flush params the same way
                if end == '\n':
                    tqdm.write(msg)
                else:
                    # For non-newline endings, still use print but it may disrupt bar
                    print(msg, end=end, flush=flush)
            else:
                print(msg, end=end, flush=flush)
        
        if self.verbose_mode == 'silent':
            if level == 'always':
                _do_print(message)
            return
            
        if level == 'always':
            _do_print(message)
        elif level == 'both':
            if self.verbose_mode in ('full', 'simple', 'progress'):
                _do_print(message)
        elif level == 'full' and self.verbose_mode == 'full':
            _do_print(message)
        elif level == 'simple' and self.verbose_mode in ('simple', 'progress'):
            _do_print(message)
        elif level == 'progress' and self.verbose_mode == 'progress':
            # For progress mode, print with carriage return to overwrite
            print(f'\r{message}', end='', flush=True)

    def _save_matrices_to_excel(self, df, writer, level='bodyId'):
        """Generate and save connection matrices to Excel"""
        # Convert Polars to Pandas if needed
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

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

    def _save_df_to_csv_polars(self, df, path, index=False):
        """Save DataFrame to CSV using Polars for speed.
        
        Uses UTF-8 encoding for cross-platform compatibility (Windows/macOS/Linux).
        """
        import polars as pl
        if df is None:
            return

        is_polars = isinstance(df, pl.DataFrame)
        
        if is_polars:
            if df.is_empty():
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(','.join(df.columns) + '\n')
                return
            
            try:
                # Polars doesn't have index, so ignore index param
                df.write_csv(path)
            except Exception as e:
                print(f"Error saving Polars DF: {e}")
        else:
            if df.empty:
                # Create empty file if dataframe is empty, to match pandas behavior
                with open(path, 'w', encoding='utf-8') as f:
                    if df is not None:
                        f.write(','.join(df.columns) + '\n')
                return
                
            try:
                import polars as pl
                # If index is True, reset index to make it a column
                if index:
                    df_to_save = df.reset_index()
                else:
                    df_to_save = df
                    
                pl_df = pl.from_pandas(df_to_save)
                pl_df.write_csv(path)
            except Exception as e:
                # Fallback to Pandas if Polars fails (e.g. object types)
                try:
                    df.to_csv(path, index=index, encoding='utf-8')
                except Exception as e2:
                    print(f"  Error saving CSV (Polars: {e}, Pandas: {e2})", flush=True)

    def _read_csv(self, filepath: str, **kwargs) -> 'pd.DataFrame':
        """Read CSV with polars (faster) and convert to pandas.
        
        Uses polars for faster reads when available, falls back to pandas.
        Ensures cross-platform compatibility with UTF-8 encoding.
        
        Args:
            filepath: Path to CSV file
            **kwargs: Additional arguments passed to pandas read_csv
            
        Returns:
            pandas DataFrame
        """
        import pandas as pd
        try:
            import polars as pl
            # Check for pandas-specific args that polars doesn't support
            pandas_only_args = {'index_col', 'low_memory', 'dtype'}
            if any(k in kwargs for k in pandas_only_args):
                # Use pandas directly for complex reads
                return pd.read_csv(filepath, encoding='utf-8', **kwargs)
            # Use polars for simple reads
            return pl.read_csv(filepath, infer_schema_length=10000).to_pandas()
        except ImportError:
            return pd.read_csv(filepath, encoding='utf-8', **kwargs)
        except Exception:
            # Fallback for polars issues
            return pd.read_csv(filepath, encoding='utf-8', **kwargs)

    def _save_matrices_to_csv(self, df, folder, level='bodyId'):
        """Generate and save connection matrices to CSV using Polars for speed"""
        import polars as pl
        
        is_polars = isinstance(df, pl.DataFrame)
        if is_polars:
            if df.is_empty(): return
        else:
            if df.empty: return

        # Determine columns
        if level == 'bodyId':
            index_col = 'bodyId_pre'
            columns_col = 'bodyId_post'
        else:
            index_col = 'type_pre'
            columns_col = 'type_post'
            
        if is_polars:
            pl_df = df
        else:
            try:
                pl_df = pl.from_pandas(df)
            except Exception as e:
                print(f"  Error converting to Polars: {e}", flush=True)
                return
            
        # 1. Weight Matrix
        if level != 'bodyId':
            try:
                # Use sum aggregation for weights to handle duplicates (e.g. same connection in multiple layers)
                mat_weight = pl_df.pivot(values='weight', index=index_col, columns=columns_col, aggregate_function='sum').fill_null(0)
                mat_weight.write_csv(os.path.join(folder, f'conn_mat_{level}_weight.csv'))
            except Exception as e:
                print(f" Failed: {e}", flush=True)

        # 2. Ratio Matrix
        if level != 'bodyId' and 'connection_ratio' in df.columns:
            try:
                # Use max for ratios to show the strongest connection ratio found
                mat_ratio = pl_df.pivot(values='connection_ratio', index=index_col, columns=columns_col, aggregate_function='max').fill_null(0)
                mat_ratio.write_csv(os.path.join(folder, f'conn_mat_{level}_ratio.csv'))
            except Exception as e:
                print(f" Failed: {e}", flush=True)

        # 3. Probability Matrix
        if level != 'bodyId' and 'traversal_probability' in df.columns:
            try:
                # Use max for probabilities
                mat_prob = pl_df.pivot(values='traversal_probability', index=index_col, columns=columns_col, aggregate_function='max').fill_null(0)
                mat_prob.write_csv(os.path.join(folder, f'conn_mat_{level}_prob.csv'))
            except Exception as e:
                print(f" Failed: {e}", flush=True)

        # 4. NT Type Matrix
        if 'nt_type' in df.columns:
            try:
                # Use first for strings
                mat_nt = pl_df.pivot(values='nt_type', index=index_col, columns=columns_col, aggregate_function='first')
                mat_nt.write_csv(os.path.join(folder, f'conn_mat_{level}_nt.csv'))
            except Exception as e:
                print(f" Failed: {e}", flush=True)

    def _prepare_flywire_data(self):
        '''
        Check and prepare FlyWire data from downloaded archives.
        Uses FAFB_file_converter or BANC_file_converter to ensure data validity and conversion.
        
        If cache already exists with complete data, source files are not required.
        '''
        if self.client_type != 'flywire':
            return

        dataset_safe = self.dataset.replace(':', '_').replace('.', '_')
        dataset_dir = os.path.join(self.script_path, 'datasets', dataset_safe)
        cache_dir = os.path.join(self.script_path, 'cache', dataset_safe)
        
        # Check if cache already exists and is complete
        # If so, we don't need the source files
        cache_conn_path = os.path.join(cache_dir, 'connections.parquet')
        cache_index_path = os.path.join(cache_dir, 'neuron_index.parquet')
        
        if os.path.exists(cache_conn_path) and os.path.exists(cache_index_path):
            try:
                # Quick check - just verify files are readable parquet
                import pyarrow.parquet as pq
                pq.ParquetFile(cache_conn_path)
                pq.ParquetFile(cache_index_path)
                self._vprint(f"Using existing cache for {self.dataset} (source files not required)", level='simple')
                return  # Cache is valid, no need for source files
            except Exception as e:
                self._vprint(f"Cache exists but invalid, will rebuild: {e}", level='simple')
        
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
    
    output_dir: str = os.path.join(os.path.expanduser('~'), 'connectome_analysis')
    '''
    folder to save all data (subfolders auto-generated based on query)
    Default: ~/connectome_analysis/
    '''
    
    save_folder: str = '' # initialized in InitializeNeuronInfo()
    '''folder to save the current data (auto-generated from source/target names)'''
    
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
    
    skip_bodyId: bool = False
    '''
    If True, skip saving bodyId-level data, visualizations, and calculations in FindAllPath.
    This significantly reduces processing time and disk usage when only type-level analysis is needed.
    '''

    max_interlayer: int = 1
    '''
    Maximum number of interlayers to be considered in connection.
    Values:
      -1: Fetch source/target neurons only (no connections). Use FetchNeuronsOnly().
       0: Direct connections only. Use FindDirectConnections().
       1, 2, ...: Include interlayer connections. Use FindAllPath() or FindPath().
    '''
    
    pathfinding: str = 'MemoizedDFS'
    '''
    Pathfinding algorithm to use in FindAllPath:
    - 'MemoizedDFS': Meet-in-the-middle DFS - optimized for deep paths (L>=5) (default)
    - 'Bidirectional': Bidirectional BFS - optimized for shortest paths
    - 'DP': Backward Reachability (DP) - optimized for pruning dead ends (lowest memory)
    - 'DFS': Backward Memoized DFS - standard traversal
    - 'Backtracking': Backward DFS with backtracking - no memoization (lowest memory, slower)
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
    
    label_mapper: object | None = None
    '''
    Optional LabelMapper object for standardizing neuron types across datasets.
    If provided, it will be used to overwrite 'type' columns in source/target DataFrames
    and connection DataFrames with standardized labels.
    '''
    
    def __post_init__(self):
        # Flag to use tqdm.write instead of print when inside progress bar
        self._in_progress_bar = False
        
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
            # Check if existing default client is for the SAME dataset
            # Different datasets require different clients (they connect to different neuprint servers)
            try:
                existing_client = default_client()
            except RuntimeError:
                existing_client = None
            
            # Check if existing client matches our dataset
            need_new_client = True
            if existing_client is not None:
                # Compare dataset names (NeuPrint client stores dataset in .dataset attribute)
                try:
                    existing_dataset = existing_client.dataset
                    if existing_dataset == self.dataset:
                        # Same dataset - reuse existing client
                        self.client_hemibrain = existing_client
                        need_new_client = False
                        self._vprint(f"Reusing existing NeuPrint client for dataset: {self.dataset}", level='full')
                    else:
                        self._vprint(f"Existing client is for '{existing_dataset}', need new client for '{self.dataset}'", level='full')
                except AttributeError:
                    # Client doesn't have dataset attribute, create new one
                    pass
            
            if need_new_client:
                self._vprint(f"Initializing NeuPrint client for dataset: {self.dataset}", level='full')
                
                # Use TokenManager
                try:
                    from .utils.token_manager import token_manager
                    self.token = token_manager.get_token('NEUPRINT_TOKEN', self.token)
                except ImportError:
                    # Fallback if import fails (e.g. running script directly)
                    try:
                        from src.utils.token_manager import token_manager
                        self.token = token_manager.get_token('NEUPRINT_TOKEN', self.token)
                    except ImportError:
                        pass

                self.client_hemibrain = Client(self.server, self.dataset, self.token)
                set_default_client(self.client_hemibrain)

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
    
    def _ensure_neuprint_client(self):
        '''
        Ensure NeuPrint client exists for THIS dataset.
        
        Important: The global default_client() may be for a DIFFERENT dataset
        (e.g., when processing multiple datasets in comparison mode).
        This method checks if the existing client matches our dataset and 
        creates a new one if needed.
        '''
        if self.client_type != 'neuprint':
            return  # Not using NeuPrint
        
        if self.client_hemibrain is not None:
            # Already have a client - verify it's for the right dataset
            try:
                if self.client_hemibrain.dataset == self.dataset:
                    return  # Correct client already set
            except AttributeError:
                pass  # Can't verify, proceed to create new one
        
        from neuprint import Client, set_default_client, default_client
        
        # Check if existing default client is for the SAME dataset
        try:
            existing_client = default_client()
        except RuntimeError:
            existing_client = None
        
        if existing_client is not None:
            try:
                if existing_client.dataset == self.dataset:
                    self.client_hemibrain = existing_client
                    return  # Reuse existing client
            except AttributeError:
                pass  # Can't verify, create new one
        
        # Need a new client for this dataset
        self._vprint(f"Creating NeuPrint client for dataset: {self.dataset}", level='full')
        self.client_hemibrain = Client(self.server, self.dataset, self.token)
        set_default_client(self.client_hemibrain)
    
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
            self._vprint(f'\n📥 Complete dataset not found, downloading ALL neurons (including type=None)...', level='always')
            self._vprint(f'   This is a one-time download for cache enrichment.', level='always')
            # Ensure we have a valid client for THIS dataset (not a different one from global default)
            self._ensure_neuprint_client()
            try:
                # Pull complete dataset with omitNoneType=False
                sv.pull_dataset(self.dataset, save_path=dataset_path, omitNoneType=False)
                self._vprint(f'✅ Complete dataset saved to: {dataset_path}_*.csv', level='always')
            except Exception as e:
                self._vprint(f'⚠️ Warning: Failed to download complete dataset: {e}', level='always')
                self._vprint(f'   Cache enrichment may fail for neurons without types.', level='always')
    
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
        pl.DataFrame : Connection database (Polars)
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
                    # Use Polars to read CSV
                    # Don't restrict dtypes on read - this can cause columns to be dropped
                    # Use infer_schema_length to properly detect column types from more rows
                    df = pl.read_csv(
                        csv_path, 
                        infer_schema_length=10000
                    )
                    
                    column_map = {
                        'pre_root_id': 'bodyId_pre',
                        'post_root_id': 'bodyId_post',
                        'syn_count': 'weight',
                        'neuropil': 'roi',
                        'pre': 'bodyId_pre',
                        'post': 'bodyId_post',
                        'synapses': 'weight'
                    }
                    # Rename columns if they exist
                    existing_cols = df.columns
                    rename_dict = {k: v for k, v in column_map.items() if k in existing_cols and v not in existing_cols}
                    if rename_dict:
                        df = df.rename(rename_dict)
                    
                    if 'weight' not in df.columns:
                        df = df.with_columns(pl.lit(1).alias('weight'))
                    if 'roi' not in df.columns:
                        df = df.with_columns(pl.lit('None').alias('roi'))
                    if 'cached_date' not in df.columns:
                        df = df.with_columns(pl.lit(datetime.now().strftime("%Y-%m-%d")).alias('cached_date'))
                        
                    cols_to_keep = ['bodyId_pre', 'bodyId_post', 'weight', 'roi', 'nt_type', 'cached_date']
                    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
                    df = df.select(cols_to_keep)
                    
                    df = df.with_columns([
                        pl.col('bodyId_pre').cast(pl.Utf8),
                        pl.col('bodyId_post').cast(pl.Utf8)
                    ])
                    
                    self._vprint(f'  ✓ Imported {len(df):,} connections from CSV', level='full')
                    
                    self._vprint(f'  💾 Saving to cache for faster future access...', level='full')
                    df.write_parquet(db_path, compression='gzip')
                    
                    # Cache in memory and build index
                    self._conn_df_cache = df
                    self._build_conn_index()
                    return df
                except Exception as e:
                    self._vprint(f'  ⚠️ Error importing FlyWire CSV: {e}', level='full')
        
        if os.path.exists(db_path):
            try:
                file_size_mb = os.path.getsize(db_path) / (1024 * 1024)
                self._vprint(f'  ⏳ Loading connection database ({file_size_mb:.1f} MB)...', level='always')
                
                # Check for batch files that haven't been consolidated
                cache_dir = os.path.dirname(db_path)
                batch_dir = os.path.join(cache_dir, '_batch_files')
                batch_files = []
                if os.path.exists(batch_dir):
                    batch_files = sorted([
                        os.path.join(batch_dir, f) 
                        for f in os.listdir(batch_dir) 
                        if f.startswith('batch_') and f.endswith('.parquet')
                    ])
                
                # Use Polars for memory-efficient loading
                self._vprint(f'  ⏳ Using Polars to load {len(batch_files)} batch files + main cache...', level='always')
                
                # Load all files with Polars
                all_files = [db_path] + batch_files
                
                # Common columns to avoid schema mismatch
                # We scan the first file to get schema, assuming consistency
                lf_schema = pl.scan_parquet(db_path).collect_schema()
                common_cols = ['bodyId_pre', 'bodyId_post', 'weight', 'roi', 'cached_date']
                available_cols = [c for c in common_cols if c in lf_schema.names()]
                
                # Scan and concat lazily with column selection, then collect
                lazy_frames = []
                for f in all_files:
                    lf = pl.scan_parquet(f)
                    lazy_frames.append(lf.select(available_cols))
                
                df = pl.concat(lazy_frames, how='diagonal_relaxed').collect()
                
                # Ensure string types
                df = df.with_columns([
                    pl.col('bodyId_pre').cast(pl.Utf8),
                    pl.col('bodyId_post').cast(pl.Utf8)
                ])
                    
                self._vprint(f'  ✓ Loaded {len(df):,} cached connections', level='always')
                
                # Cache in memory and build index
                self._conn_df_cache = df
                self._build_conn_index()
                return df
            except Exception as e:
                self._vprint(f'  ⚠️ Warning: Failed to load connection database: {e}', level='full')
                self._conn_df_cache = pl.DataFrame(schema={'bodyId_pre': pl.Utf8, 'bodyId_post': pl.Utf8, 'weight': pl.Int64, 'roi': pl.Utf8, 'cached_date': pl.Utf8})
                self._conn_index = {}
                return self._conn_df_cache
        
        # No cache exists - return empty DataFrame
        self._vprint(f'  ℹ️ No connection cache found. Starting fresh.', level='full')
        self._conn_df_cache = pl.DataFrame(schema={'bodyId_pre': pl.Utf8, 'bodyId_post': pl.Utf8, 'weight': pl.Int64, 'roi': pl.Utf8, 'cached_date': pl.Utf8})
        self._conn_index = {}
        return self._conn_df_cache

    def _build_conn_index(self):
        '''
        Build dict indexes for O(1) connection lookups by bodyId_pre and bodyId_post.
        Called after loading connection database from disk.
        Also updates the module-level shared cache.
        '''
        if self._conn_df_cache is None or self._conn_df_cache.is_empty():
            self._conn_index = {}
            self._conn_index_post = {}
            return

        # self._vprint(f'  ⏳ Building connection indexes for fast lookups...', level='always')
        self._conn_index = {}
        self._conn_index_post = {}

        n_rows = len(self._conn_df_cache)
        # Try Polars for faster index building (2-3x faster for large datasets)
        try:
            import polars as pl
            
            # If _conn_df_cache is already Polars, use it directly
            if isinstance(self._conn_df_cache, pl.DataFrame):
                df_pl = self._conn_df_cache.with_row_index('idx')
            else:
                # Fallback if somehow it's Pandas (shouldn't happen with new load)
                df_pl = pl.DataFrame({
                    'bodyId_pre': self._conn_df_cache['bodyId_pre'].values,
                    'bodyId_post': self._conn_df_cache['bodyId_post'].values,
                    'idx': range(n_rows)
                })
            
            # Group by pre and collect indices using iter_rows for efficiency
            pre_result = df_pl.group_by('bodyId_pre').agg(pl.col('idx'))
            self._conn_index = {row[0]: row[1] for row in pre_result.iter_rows()}
            
            # Group by post and collect indices
            post_result = df_pl.group_by('bodyId_post').agg(pl.col('idx'))
            self._conn_index_post = {row[0]: row[1] for row in post_result.iter_rows()}
            
            # del df_pl, pre_result, post_result
            
        except ImportError:
            # Fallback to optimized Python with defaultdict
            from collections import defaultdict
            self._conn_index = defaultdict(list)
            self._conn_index_post = defaultdict(list)
            # Assuming Polars DF, convert to numpy/list for iteration if Polars not available (unlikely)
            pass

        self._vprint(f'  ✓ Index built: {len(self._conn_index):,} upstream, {len(self._conn_index_post):,} downstream neurons', level='always')
        
        # Update module-level shared cache for other instances
        global _FNC_CACHE
        if hasattr(self, '_dataset_safe'):
            if self._dataset_safe not in _FNC_CACHE:
                _FNC_CACHE[self._dataset_safe] = {}
            _FNC_CACHE[self._dataset_safe]['conn_df'] = self._conn_df_cache
            _FNC_CACHE[self._dataset_safe]['conn_index'] = self._conn_index
            _FNC_CACHE[self._dataset_safe]['conn_index_post'] = self._conn_index_post
    
    def _save_connection_db(self, conn_db):
        '''
        Save unified connection database with compression.
        Also updates the in-memory cache and rebuilds the index.
        Uses Polars for efficient writing.
        '''
        db_path = self._get_connection_db_path()
        try:
            import polars as pl
            # Ensure conn_db is Polars DataFrame
            if not isinstance(conn_db, pl.DataFrame):
                conn_db = pl.from_pandas(conn_db)
                
            conn_db.write_parquet(db_path, compression='gzip')
            self._vprint(f'  ✓ Database saved successfully', level='full')
            
            # Update in-memory cache
            self._conn_df_cache = conn_db
            self._build_conn_index()
        except Exception as e:
            self._vprint(f'  ⚠️ Warning: Failed to save connection database: {e}', level='full')
    
    def _append_connections_to_cache(self, connections, neurons_fetched, mark_complete_if_empty=False):
        """
        MEMORY-EFFICIENT: Append connections to cache using batch files.
        
        Strategy:
        - Write each batch to a separate parquet file in a batch directory
        - Files are named: batch_XXXXXX.parquet
        - Final merge happens only at the end via _consolidate_batch_files()
        - Never load the full existing cache into memory during fetching
        
        Parameters:
        -----------
        connections : pd.DataFrame
            New connections to append (must have bodyId_pre, bodyId_post, weight)
        neurons_fetched : list
            List of neurons that were fetched (to mark as cached)
        mark_complete_if_empty : bool
            If True, mark neurons as complete even when connections.empty.
            Default False: prevents marking neurons complete when API might have failed.
        """
        import os
        
        if connections.empty:
            # FIXED: Only mark as complete if explicitly requested
            # This prevents falsely marking neurons as complete when API call failed/timed out
            if mark_complete_if_empty:
                self._update_neuron_index_batch(neurons_fetched)
            return
        
        # Use a batch directory for temporary batch files
        cache_dir = os.path.dirname(self._get_connection_db_path())
        batch_dir = os.path.join(cache_dir, '_batch_files')
        os.makedirs(batch_dir, exist_ok=True)
        
        # Find next batch number
        existing_batches = [f for f in os.listdir(batch_dir) if f.startswith('batch_') and f.endswith('.parquet')]
        batch_num = len(existing_batches)
        batch_path = os.path.join(batch_dir, f'batch_{batch_num:06d}.parquet')
        
        # Prepare connections
        conn = connections[['bodyId_pre', 'bodyId_post', 'weight']].copy()
        conn['bodyId_pre'] = conn['bodyId_pre'].astype(str)
        conn['bodyId_post'] = conn['bodyId_post'].astype(str)
        
        if 'roi' in connections.columns:
            conn['roi'] = connections['roi']
        else:
            conn['roi'] = ''
        
        conn['cached_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Write this batch to its own file - NO loading of existing data
        conn.to_parquet(batch_path, index=False, compression='gzip')
        
        # Calculate connection counts per neuron
        conn_counts = connections.groupby('bodyId_pre').size().to_dict()
        # Ensure all neurons_fetched have a count (0 if not in connections)
        for n in neurons_fetched:
            n_str = str(n)
            if n_str not in conn_counts:
                conn_counts[n_str] = 0
        
        # Update neuron index with actual connection counts
        self._update_neuron_index_batch(neurons_fetched, connection_counts=conn_counts)
    
    def _consolidate_batch_files(self, deduplicate=True):
        """
        Merge all batch files into the main connections.parquet file.
        Called after all batches are fetched, or periodically if needed.
        
        Parameters:
        -----------
        deduplicate : bool
            If True, remove duplicates during merge
            
        Returns:
        --------
        int : Number of connections after consolidation
        """
        import os
        import gc
        
        cache_dir = os.path.dirname(self._get_connection_db_path())
        batch_dir = os.path.join(cache_dir, '_batch_files')
        db_path = self._get_connection_db_path()
        
        if not os.path.exists(batch_dir):
            return 0
        
        batch_files = sorted([
            os.path.join(batch_dir, f) 
            for f in os.listdir(batch_dir) 
            if f.startswith('batch_') and f.endswith('.parquet')
        ])
        
        if not batch_files:
            return 0
        
        print(f"  Consolidating {len(batch_files)} batch files...")
        
        # Use Polars for memory-efficient consolidation
        try:
            import polars as pl
            print(f"  Using Polars for memory-efficient consolidation...")
            
            # Common columns we need (ignore extras like conn_roiInfo)
            common_cols = ['bodyId_pre', 'bodyId_post', 'weight', 'roi', 'cached_date']
            
            # Collect all parquet files to merge
            all_files = batch_files.copy()
            if os.path.exists(db_path):
                all_files.insert(0, db_path)
            
            # Use lazy evaluation to minimize memory usage
            # Select only common columns to avoid schema mismatch
            lazy_frames = []
            for f in all_files:
                lf = pl.scan_parquet(f)
                # Get available columns and select only the ones we need
                available_cols = [c for c in common_cols if c in lf.collect_schema().names()]
                lazy_frames.append(lf.select(available_cols))
            
            combined = pl.concat(lazy_frames, how='diagonal_relaxed')
            
            # Deduplicate if requested (using lazy API)
            if deduplicate:
                print(f"  Deduplicating...")
                merge_cols = ['bodyId_pre', 'bodyId_post', 'roi']
                # Only use columns that exist
                merge_cols = [c for c in merge_cols if c in combined.collect_schema().names()]
                if merge_cols:
                    combined = combined.unique(subset=merge_cols, keep='last')
            
            # Write to temp file, then replace original
            tmp_path = db_path + '.tmp'
            print(f"  Writing consolidated cache...")
            combined.collect().write_parquet(tmp_path, compression='gzip')
            
            # Get count before deleting
            total_count = pl.scan_parquet(tmp_path).select(pl.len()).collect().item()
            
            # Replace original with consolidated
            if os.path.exists(db_path):
                os.remove(db_path)
            os.rename(tmp_path, db_path)
            
            # Clean up batch files
            import shutil
            shutil.rmtree(batch_dir)
            
            print(f"  ✓ Consolidated to {total_count:,} connections")
            return total_count
            
        except ImportError:
            # Polars not available - just skip consolidation and let loading handle it
            print(f"  ⚡ Polars not installed - skipping consolidation")
            print(f"     {len(batch_files)} batch files will be loaded on demand")
            print(f"     Install polars for better memory efficiency: pip install polars")
            
            # Just count the connections without loading into memory
            total_count = 0
            if os.path.exists(db_path):
                import pyarrow.parquet as pq
                total_count += pq.read_metadata(db_path).num_rows
            
            for bf in batch_files:
                import pyarrow.parquet as pq
                total_count += pq.read_metadata(bf).num_rows
            
            print(f"  ✓ Total connections available: {total_count:,}")
            return total_count
    
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
                    df = self._read_csv(csv_path, dtype={'bodyId': str})
                    
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
        # Use vectorized access for better performance
        df = self._neuron_index_cache
        bodyids = df['bodyId'].astype(str).values
        downstream_complete = df['downstream_complete'].values if 'downstream_complete' in df.columns else [False] * len(df)
        types = df['type'].values if 'type' in df.columns else [''] * len(df)
        instances = df['instance'].values if 'instance' in df.columns else [''] * len(df)
        posts = df['post'].values if 'post' in df.columns else [0] * len(df)
        last_fetched = df['last_fetched'].values if 'last_fetched' in df.columns else [''] * len(df)
        connection_counts = df['connection_count'].values if 'connection_count' in df.columns else [0] * len(df)
        
        for idx in range(len(bodyids)):
            self._neuron_index_dict[bodyids[idx]] = {
                'downstream_complete': downstream_complete[idx] if downstream_complete[idx] is not None else False,
                'type': types[idx] if types[idx] is not None else '',
                'instance': instances[idx] if instances[idx] is not None else '',
                'post': posts[idx] if posts[idx] is not None else 0,
                'last_fetched': last_fetched[idx] if last_fetched[idx] is not None else '',
                'connection_count': connection_counts[idx] if connection_counts[idx] is not None else 0,
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
        Uses Polars for performance.
        '''
        import polars as pl
        if not self.use_cache:
            return pl.DataFrame(), upstream_bodyIds, []
        
        self._vprint(f'  ⏳ Querying cache for {len(upstream_bodyIds):,} neurons...', level='full')
        
        # Load caches (uses in-memory if already loaded)
        conn_db = self._load_connection_db()
        neuron_index = self._load_neuron_index()
        
        # Handle None or empty cache gracefully
        if conn_db is None or (hasattr(conn_db, 'is_empty') and conn_db.is_empty()) or len(conn_db) == 0:
            return pl.DataFrame(), upstream_bodyIds, []
        
        # Build a set of neurons that actually have connections in the cache
        # This provides a stricter validation than just trusting neuron_index
        if isinstance(conn_db, pl.DataFrame):
             neurons_with_connections = set(conn_db['bodyId_pre'].cast(pl.Utf8).unique().to_list())
        else:
             # Fallback if somehow Pandas
             neurons_with_connections = set(conn_db['bodyId_pre'].astype(str).unique())
        
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
                
                # STRICTER VALIDATION: Even if marked complete, verify it has connections OR
                # is explicitly marked with connection_count (to handle legitimate 0-connection neurons)
                conn_count = neuron_data.get('connection_count', -1)
                
                # Trust the cache if:
                # 1. Marked complete AND has connections in cache, OR
                # 2. Marked complete AND explicitly has connection_count >= 0 (including 0)
                if is_complete:
                    has_connections = bodyId in neurons_with_connections
                    has_valid_count = conn_count >= 0
                    
                    if has_connections or has_valid_count:
                        cached_upstream.append(bodyId)
                    else:
                        # Marked complete but no connections and no valid count - needs refetch
                        uncached_upstream.append(bodyId)
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
                # Polars slicing
                cached_conn = conn_db[row_indices]
            else:
                cached_conn = pl.DataFrame()
            
            # Filter by downstream if specified
            if downstream_bodyIds is not None and not cached_conn.is_empty():
                downstream_set = set(str(b) for b in downstream_bodyIds)
                # Polars filter
                cached_conn = cached_conn.filter(pl.col('bodyId_post').cast(pl.Utf8).is_in(downstream_set))
            
            # Return both cached connections and list of partially cached neurons for later marking
            return cached_conn, uncached_upstream, partially_cached
        
        return pl.DataFrame(), uncached_upstream, []
    
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
                    ndf_complete = self._read_csv(dataset_path, header=0, index_col=None, dtype={'bodyId': str}, low_memory=False)
                else:
                    ndf_complete = self._read_csv(dataset_path, header=0, index_col=0, low_memory=False)
                
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
        
        # Load existing database (returns Polars DataFrame)
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
        
        # Convert to Polars for consistency with conn_db
        new_conn_pl = pl.from_pandas(new_conn)
        
        # Merge with existing, removing duplicates (keep existing entries)
        # conn_db is a Polars DataFrame, use .is_empty() not .empty
        if not conn_db.is_empty():
            self._vprint(f'  ⏳ Merging {len(new_conn_pl):,} connections with existing database...', level='full')
            # Remove any new connections that already exist (based on bodyId_pre, bodyId_post, roi)
            merge_cols = ['bodyId_pre', 'bodyId_post', 'roi']
            combined = pl.concat([conn_db, new_conn_pl], how='diagonal_relaxed')
            combined = combined.unique(subset=merge_cols, keep='first')
        else:
            combined = new_conn_pl
        
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
        
        # Load existing database (returns Polars DataFrame)
        conn_db = self._load_connection_db()
        
        # Prepare new connections as Polars DataFrame
        new_conn = new_connections[['bodyId_pre', 'bodyId_post', 'weight']].copy()
        
        # Ensure bodyIds are strings
        new_conn['bodyId_pre'] = new_conn['bodyId_pre'].astype(str)
        new_conn['bodyId_post'] = new_conn['bodyId_post'].astype(str)
        
        if 'roi' in new_connections.columns:
            new_conn['roi'] = new_connections['roi']
        else:
            new_conn['roi'] = ''
        
        new_conn['cached_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert to Polars for consistency with conn_db
        new_conn_pl = pl.from_pandas(new_conn)
        
        # Merge with existing, removing duplicates (keep existing entries)
        # conn_db is a Polars DataFrame, use .is_empty() not .empty
        if not conn_db.is_empty():
            self._vprint(f'  ⏳ Merging {len(new_conn_pl):,} connections with existing database...', level='full')
            merge_cols = ['bodyId_pre', 'bodyId_post', 'roi']
            combined = pl.concat([conn_db, new_conn_pl], how='diagonal_relaxed')
            combined = combined.unique(subset=merge_cols, keep='first')
        else:
            combined = new_conn_pl
        
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
                ndf_complete = self._read_csv(dataset_path, header=0, index_col=None, dtype={'bodyId': str}, low_memory=False)
            else:
                ndf_complete = self._read_csv(dataset_path, header=0, index_col=0, low_memory=False)

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
                    'downstream_complete': bool(mark_complete),  # Explicit bool for consistent dtype
                    'last_fetched': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'connection_count': conn_count
                }])
                neuron_index = pd.concat([neuron_index, new_entry], ignore_index=True)
                # Ensure consistent bool dtype after concat to avoid FutureWarning
                neuron_index['downstream_complete'] = neuron_index['downstream_complete'].astype(bool)
        
        self._vprint(f'  ⏳ Saving neuron index ({len(neuron_index):,} total neurons)...', level='full')
        self._save_neuron_index(neuron_index)
        
        if mark_complete:
            # Explicitly cast to bool to avoid FutureWarning about object-dtype columns
            downstream_complete = neuron_index['downstream_complete'].astype(bool)
            completed_count = len([b for b in upstream_bodyIds if b in neuron_index[downstream_complete]['bodyId'].values])
            self._vprint(f'  📝 Updated neuron index: {completed_count} neurons marked as complete', level='full')
    
    def _update_neuron_index_batch(self, bodyids, connection_counts=None):
        '''
        Efficiently update neuron index for a batch of neurons.
        Marks them as downstream_complete=True.
        Used by build_connection_cache after consolidation.
        
        Parameters:
        -----------
        bodyids : list
            List of bodyIds to mark as complete
        connection_counts : dict, optional
            Dict mapping bodyId (str) -> connection count. If provided, updates
            connection_count for each neuron. If None, sets connection_count=0.
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
        
        # Try parquet first (faster)
        parquet_path = dataset_path.replace('.csv', '.parquet')
        
        bodyids_str = [str(x) for x in bodyids]
        bodyids_set = set(bodyids_str)
        
        if os.path.exists(parquet_path):
            ndf_complete = pd.read_parquet(parquet_path)
            if 'bodyId' in ndf_complete.columns:
                ndf_complete['bodyId'] = ndf_complete['bodyId'].astype(str)
        elif os.path.exists(dataset_path):
            is_fafb = 'fafb' in self.dataset.lower() or 'flywire' in self.dataset.lower()
            if is_fafb:
                ndf_complete = self._read_csv(dataset_path, header=0, index_col=None, dtype={'bodyId': str}, low_memory=False)
            else:
                ndf_complete = self._read_csv(dataset_path, header=0, index_col=0, low_memory=False)
            if 'bodyId' in ndf_complete.columns:
                ndf_complete['bodyId'] = ndf_complete['bodyId'].astype(str)
        else:
            ndf_complete = pd.DataFrame(columns=['bodyId', 'type', 'instance', 'post'])
        
        # Filter to only the bodyIds we need
        if not ndf_complete.empty and 'bodyId' in ndf_complete.columns:
            neuron_info = ndf_complete[ndf_complete['bodyId'].isin(bodyids_set)].copy()
        else:
            neuron_info = pd.DataFrame(columns=['bodyId', 'type', 'instance', 'post'])
        
        # Create a dict for fast lookup using vectorized access
        neuron_info_dict = {}
        if not neuron_info.empty:
            bodyid_col = neuron_info['bodyId'].astype(str).values
            type_col = neuron_info['type'].values if 'type' in neuron_info.columns else [''] * len(neuron_info)
            instance_col = neuron_info['instance'].values if 'instance' in neuron_info.columns else [''] * len(neuron_info)
            post_col = neuron_info['post'].values if 'post' in neuron_info.columns else [0] * len(neuron_info)
            
            for i in range(len(bodyid_col)):
                neuron_info_dict[bodyid_col[i]] = {
                    'type': type_col[i] if type_col[i] is not None else '',
                    'instance': instance_col[i] if instance_col[i] is not None else '',
                    'post': post_col[i] if post_col[i] is not None else 0
                }
        
        # Check existing in index
        existing_set = set(neuron_index['bodyId'].astype(str).values) if not neuron_index.empty else set()
        
        # Update existing entries
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for bid in bodyids_str:
            if bid in existing_set:
                neuron_index.loc[neuron_index['bodyId'].astype(str) == bid, 'downstream_complete'] = True
                neuron_index.loc[neuron_index['bodyId'].astype(str) == bid, 'last_fetched'] = now
                # Update connection count if provided
                if connection_counts is not None:
                    count = connection_counts.get(bid, 0)
                    neuron_index.loc[neuron_index['bodyId'].astype(str) == bid, 'connection_count'] = count
        
        # Add new entries in bulk
        new_entries = []
        for bid in bodyids_str:
            if bid not in existing_set:
                info = neuron_info_dict.get(bid, {'type': '', 'instance': '', 'post': 0})
                # Get connection count from dict if provided, else 0
                count = connection_counts.get(bid, 0) if connection_counts else 0
                new_entries.append({
                    'bodyId': bid,
                    'type': info['type'],
                    'instance': info['instance'],
                    'post': info['post'],
                    'downstream_complete': True,
                    'last_fetched': now,
                    'connection_count': count
                })
        
        if new_entries:
            new_df = pd.DataFrame(new_entries)
            neuron_index = pd.concat([neuron_index, new_df], ignore_index=True)
            neuron_index['downstream_complete'] = neuron_index['downstream_complete'].astype(bool)
        
        self._save_neuron_index(neuron_index)
    
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
                ndf_complete = self._read_csv(dataset_path, header=0, index_col=None, dtype={'bodyId': str}, low_memory=False)
            else:
                ndf_complete = self._read_csv(dataset_path, header=0, index_col=0, low_memory=False)
            
            # Ensure bodyId is string for all datasets (not just FAFB)
            # This is critical for matching with cached connections which use string bodyIds
            if 'bodyId' in ndf_complete.columns:
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
                ndf_complete = self._read_csv(dataset_path, header=0, index_col=None, dtype={'bodyId': str}, low_memory=False)
            else:
                ndf_complete = self._read_csv(dataset_path, header=0, index_col=0, low_memory=False)
            
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

            # Ensure client is logged in (NeuPrint) for the CORRECT dataset
            self._ensure_neuprint_client()
            
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
                ndf_complete = self._read_csv(dataset_path, header=0, index_col=None, dtype={'bodyId': str}, low_memory=False)
            else:
                ndf_complete = self._read_csv(dataset_path, header=0, index_col=0, low_memory=False)
            
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
                # Ensure we have a valid client for THIS dataset
                self._ensure_neuprint_client()
                
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
        
        # Convert Polars to Pandas for compatibility with rest of pipeline
        try:
            import polars as pl
            if isinstance(cached_conn, pl.DataFrame):
                cached_conn = cached_conn.to_pandas()
        except ImportError:
            pass
        
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
                        full_conn = self._read_csv(conn_file, dtype={'pre_root_id': str, 'post_root_id': str})
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

                    # Ensure we have a valid client for THIS dataset (not a different one from global default)
                    self._ensure_neuprint_client()
                    
                    # Batch processing with timeout and retry
                    batch_size = 1000
                    all_api_conn = []
                    
                    # Import API utilities for timeout/retry
                    try:
                        from src.utils.api_utils import api_call_with_retry, APITimeoutError, APIRetryExhaustedError
                    except ImportError:
                        # Fallback: define inline if utils not available
                        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
                        class APITimeoutError(Exception): pass
                        class APIRetryExhaustedError(Exception): pass
                        def api_call_with_retry(func, timeout=60, max_retries=3, retry_delay=2.0, description="API call", on_retry=None, verbose=True):
                            import time
                            last_exc = None
                            for attempt in range(1, max_retries + 1):
                                try:
                                    with ThreadPoolExecutor(max_workers=1) as executor:
                                        future = executor.submit(func)
                                        return future.result(timeout=timeout)
                                except FuturesTimeoutError:
                                    last_exc = APITimeoutError(f"{description} timed out after {timeout}s (attempt {attempt}/{max_retries})")
                                    if attempt < max_retries:
                                        time.sleep(retry_delay * (2 ** (attempt - 1)))
                                except Exception as e:
                                    last_exc = e
                                    if attempt < max_retries:
                                        time.sleep(retry_delay * (2 ** (attempt - 1)))
                            raise last_exc or Exception("Unknown error")
                    
                    # Create batches
                    batches = [uncached_upstream[i:i + batch_size] for i in range(0, len(uncached_upstream), batch_size)]
                    
                    if len(batches) > 1:
                        self._vprint(f'     Processing {len(batches)} batches (size={batch_size})...', level='full')
                    
                    # Use tqdm only if multiple batches or large single batch
                    iterator = tqdm(batches, desc="Fetching batches", unit="batch") if len(batches) > 1 else batches
                    
                    failed_batches = []
                    for batch_idx, batch in enumerate(iterator):
                        def fetch_batch(b=batch):
                            """Inner function for timeout wrapping."""
                            if self.simple_fetch:
                                from neuprint import fetch_simple_connections
                                upstream_criteria = NeuronCriteria(bodyId=b)
                                downstream_criteria = NeuronCriteria(bodyId=downstream_bodyIds) if downstream_bodyIds is not None else None
                                return fetch_simple_connections(
                                    upstream_criteria=upstream_criteria,
                                    downstream_criteria=downstream_criteria,
                                    min_weight=1,
                                    **self.kwargs_fetch
                                )
                            else:
                                from neuprint import fetch_adjacencies
                                neuron_df, roi_conn_df = fetch_adjacencies(
                                    sources=b,
                                    targets=downstream_bodyIds,
                                    min_total_weight=1,
                                    **self.kwargs_fetch
                                )
                                # roi_conn_df already has bodyId_pre, bodyId_post, roi, weight
                                return roi_conn_df
                        
                        try:
                            # Use timeout and retry for each batch
                            batch_conn = api_call_with_retry(
                                fetch_batch,
                                timeout=120.0,  # 2 minutes per batch
                                max_retries=3,
                                retry_delay=5.0,
                                description=f"Batch {batch_idx+1}/{len(batches)}",
                                verbose=True
                            )
                            if batch_conn is not None and not batch_conn.empty:
                                all_api_conn.append(batch_conn)
                        except (APITimeoutError, APIRetryExhaustedError) as e:
                            self._vprint(f"     ⚠️ Batch {batch_idx+1} failed after retries: {e}", level='full')
                            failed_batches.append(batch_idx + 1)
                        except Exception as e:
                            self._vprint(f"     ⚠️ Error fetching batch {batch_idx+1}: {e}", level='full')
                            failed_batches.append(batch_idx + 1)
                    
                    if failed_batches:
                        self._vprint(f"     ⚠️ {len(failed_batches)} batches failed: {failed_batches}", level='full')
                            
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
        
        # Apply label mapping if available (AFTER caching, so cache keeps original types)
        if self.label_mapper and not combined.empty:
            self._vprint(f'  🏷️  Applying label mapping to {len(combined):,} connections...', level='full')
            # Use apply_to_dataframe from LabelMapper
            # It adds std_label_pre and std_label_post
            combined = self.label_mapper.apply_to_dataframe(combined, self.dataset)
            
            # Overwrite type_pre with std_label_pre
            if 'std_label_pre' in combined.columns:
                mask = combined['std_label_pre'] != ''
                combined.loc[mask, 'type_pre'] = combined.loc[mask, 'std_label_pre']
                combined = combined.drop(columns=['std_label_pre'])
                
            # Overwrite type_post with std_label_post
            if 'std_label_post' in combined.columns:
                mask = combined['std_label_post'] != ''
                combined.loc[mask, 'type_post'] = combined.loc[mask, 'std_label_post']
                combined = combined.drop(columns=['std_label_post'])
        
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
    
    def warm_up_cache(self, quiet: bool = False) -> dict:
        """
        Load cache into memory and build indexes for fast O(1) lookups.
        
        This method is called automatically on first query, but can be called
        explicitly for faster initial queries. It loads:
        1. Connection database (connections.parquet) -> _conn_df_cache
        2. Connection index (bodyId_pre -> row indices) -> _conn_index  
        3. Neuron index (neuron_index.parquet) -> _neuron_index_cache
        4. Neuron dict (bodyId -> metadata) -> _neuron_index_dict
        
        Cache Hierarchy:
        ---------------
        Level 0: datasets/{dataset}/*_neuron_df.parquet - Authoritative neuron info
        Level 1: cache/{dataset}/neuron_index.parquet - Which neurons are cached
        Level 2: cache/{dataset}/connections.parquet - Connection data
        Level 3: Connectivity profiles (built by ConnectivityProfiler)
        
        Parameters:
        -----------
        quiet : bool
            If True, suppress progress messages
        
        Returns:
        --------
        dict : Cache status with keys:
            - 'connections_loaded': Number of connections in cache
            - 'neurons_indexed': Number of neurons in index
            - 'index_ready': Whether O(1) lookup indexes are built
            - 'elapsed_time': Time taken in seconds
        """
        import time
        start_time = time.time()
        
        if not quiet:
            print(f"Warming up cache for {self.dataset}...")
        
        # Load connection database (triggers index building)
        conn_db = self._load_connection_db(force_reload=False)
        connections_loaded = len(conn_db) if conn_db is not None and not conn_db.is_empty() else 0
        
        # Load neuron index (triggers dict building)
        neuron_index = self._load_neuron_index(force_reload=False)
        neurons_indexed = len(neuron_index) if neuron_index is not None and not neuron_index.empty else 0
        
        # Verify indexes are built
        index_ready = (
            self._conn_index is not None and len(self._conn_index) > 0 and
            self._neuron_index_dict is not None and len(self._neuron_index_dict) > 0
        )
        
        elapsed = time.time() - start_time
        
        if not quiet:
            print(f"  Connections: {connections_loaded:,}")
            print(f"  Neurons indexed: {neurons_indexed:,}")
            print(f"  O(1) index ready: {index_ready}")
            print(f"  Time: {elapsed:.2f}s")
        
        return {
            'connections_loaded': connections_loaded,
            'neurons_indexed': neurons_indexed,
            'index_ready': index_ready,
            'elapsed_time': elapsed
        }
    
    def get_cache_status(self) -> dict:
        """
        Get comprehensive cache status for this dataset.
        
        Returns information about all cache levels:
        - Level 0: datasets/{dataset}/ neuron_df files (authoritative neuron list)
        - Level 1: cache/{dataset}/neuron_index.parquet (which neurons are cached)
        - Level 2: cache/{dataset}/connections.parquet (connection data)
        
        Returns:
        --------
        dict : Cache status with keys:
            - 'dataset': Dataset identifier
            - 'neuron_df_exists': Whether authoritative neuron list exists
            - 'neuron_df_count': Number of neurons in neuron_df (or 0)
            - 'neuron_index_exists': Whether neuron index cache exists
            - 'neurons_indexed': Number of neurons in index
            - 'neurons_complete': Number marked as downstream_complete
            - 'connection_cache_exists': Whether connection cache exists
            - 'connections_cached': Number of connections
            - 'unique_upstream': Number of unique upstream neurons in cache
            - 'index_ready': Whether O(1) lookup indexes are built in memory
            - 'completeness': Ratio of cached vs expected neurons (0.0 to 1.0)
        """
        import os
        
        dataset_safe = self.dataset.replace(':', '_').replace('.', '_')
        
        # Check Level 0: datasets/ neuron_df
        neuron_df_path_parquet = os.path.join(
            self.script_path, 'datasets', dataset_safe,
            f"{dataset_safe}_allneurons_neuron_df.parquet"
        )
        neuron_df_path_csv = os.path.join(
            self.script_path, 'datasets', dataset_safe,
            f"{dataset_safe}_allneurons_neuron_df.csv"
        )
        neuron_df_exists = os.path.exists(neuron_df_path_parquet) or os.path.exists(neuron_df_path_csv)
        neuron_df_count = len(self._get_all_dataset_bodyids()) if neuron_df_exists else 0
        
        # Check Level 1: neuron_index
        index_path = self._get_neuron_index_path()
        neuron_index_exists = os.path.exists(index_path)
        neurons_indexed = 0
        neurons_complete = 0
        if neuron_index_exists:
            neuron_index = self._load_neuron_index()
            neurons_indexed = len(neuron_index)
            if 'downstream_complete' in neuron_index.columns:
                neurons_complete = neuron_index['downstream_complete'].astype(bool).sum()
        
        # Check Level 2: connections
        conn_path = self._get_connection_db_path()
        connection_cache_exists = os.path.exists(conn_path)
        connections_cached = 0
        unique_upstream = 0
        if connection_cache_exists:
            conn_db = self._load_connection_db()
            connections_cached = len(conn_db) if conn_db is not None else 0
            if conn_db is not None and 'bodyId_pre' in conn_db.columns:
                unique_upstream = conn_db['bodyId_pre'].nunique()
        
        # Check in-memory indexes
        index_ready = (
            self._conn_index is not None and len(self._conn_index) > 0 and
            self._neuron_index_dict is not None and len(self._neuron_index_dict) > 0
        )
        
        # Calculate completeness
        completeness = neurons_complete / neuron_df_count if neuron_df_count > 0 else 0.0
        
        return {
            'dataset': self.dataset,
            'neuron_df_exists': neuron_df_exists,
            'neuron_df_count': neuron_df_count,
            'neuron_index_exists': neuron_index_exists,
            'neurons_indexed': neurons_indexed,
            'neurons_complete': neurons_complete,
            'connection_cache_exists': connection_cache_exists,
            'connections_cached': connections_cached,
            'unique_upstream': unique_upstream,
            'index_ready': index_ready,
            'completeness': completeness
        }
    
    def build_connection_cache(
        self,
        neuron_types: list = None,
        neuron_bodyIds: list = None,
        batch_size: int = 100,
        force_rebuild: bool = False,
        quiet: bool = False,
        progress_callback: callable = None
    ) -> dict:
        """
        Build connection cache incrementally for specified or all neurons.
        
        MEMORY-EFFICIENT WORKFLOW:
        --------------------------
        1. Divide all neurons into batches (each neuron as upstream/source)
        2. For each batch: fetch ALL downstream connections (target=None)
        3. Append directly to cache file (no in-memory accumulation)
        4. Deduplicate only at the end if needed
        
        This works because fetching all neurons' downstream connections captures
        every edge in the graph - if A→B exists, we get it when fetching A's downstream.
        
        Cache Hierarchy:
        ---------------
        Level 0: datasets/{dataset}/*_neuron_df.parquet - Authoritative neuron list
        Level 1: cache/{dataset}/neuron_index.parquet - Tracks cached neurons
        Level 2: cache/{dataset}/connections.parquet - Actual connection data
        
        Parameters:
        -----------
        neuron_types : list, optional
            List of neuron types to cache. If None and neuron_bodyIds is None,
            caches all neurons in the dataset.
        neuron_bodyIds : list, optional
            List of specific bodyIds to cache. Takes precedence over neuron_types.
        batch_size : int
            Number of neurons to fetch per batch (default: 100)
        force_rebuild : bool
            If True, delete existing cache and rebuild from scratch (default: False)
        quiet : bool
            If True, suppress progress messages (default: False)
        progress_callback : callable, optional
            Callback function(current, total, neuron_info) for progress updates
        
        Returns:
        --------
        dict : Summary with keys:
            - 'total_neurons': Total neurons in target set
            - 'already_cached': Number of neurons already in cache
            - 'newly_cached': Number of neurons cached in this call
            - 'failed_neurons': List of neurons that failed to cache
            - 'total_connections': Total connections in cache after build
            - 'elapsed_time': Time taken in seconds
        """
        import time
        import os
        import gc
        start_time = time.time()
        
        def _print(msg):
            if not quiet:
                print(msg)
        
        _print("=" * 60)
        _print("Building Connection Cache")
        _print("=" * 60)
        _print(f"Dataset: {self.dataset}")
        
        if not self.use_cache:
            _print("Warning: Cache is disabled. Enable with use_cache=True")
            return {'total_neurons': 0, 'already_cached': 0, 'newly_cached': 0,
                    'failed_neurons': [], 'total_connections': 0, 'elapsed_time': 0}
        
        # Handle force_rebuild - clear cache first
        if force_rebuild:
            _print("Force rebuild - clearing existing cache...")
            conn_path = self._get_connection_db_path()
            index_path = self._get_neuron_index_path()
            batch_dir = os.path.join(os.path.dirname(conn_path), '_batch_files')
            if os.path.exists(conn_path):
                os.remove(conn_path)
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(batch_dir):
                import shutil
                shutil.rmtree(batch_dir)
            # Clear in-memory caches
            self._conn_df_cache = None
            self._conn_index = {}
            self._neuron_index_cache = None
            self._neuron_index_dict = {}
        else:
            # Check for pending batch files from interrupted previous run
            conn_path = self._get_connection_db_path()
            batch_dir = os.path.join(os.path.dirname(conn_path), '_batch_files')
            if os.path.exists(batch_dir):
                batch_files = [f for f in os.listdir(batch_dir) if f.startswith('batch_') and f.endswith('.parquet')]
                if batch_files:
                    _print(f"\n⚡ Found {len(batch_files)} pending batch files from interrupted run")
                    _print(f"   Consolidating to resume from checkpoint...")
                    self._consolidate_batch_files(deduplicate=True)
                    # Clear caches to reload updated index
                    self._neuron_index_cache = None
                    self._neuron_index_dict = {}
        
        # Get target bodyIds from dataset
        target_bodyIds = None
        
        if neuron_bodyIds is not None:
            target_bodyIds = [str(x) for x in neuron_bodyIds]
            _print(f"Target: {len(target_bodyIds)} specified bodyIds")
        elif neuron_types is not None:
            _print(f"Fetching bodyIds for {len(neuron_types)} neuron types...")
            target_bodyIds = []
            for ntype in neuron_types:
                try:
                    # Get bodyIds for this type from the dataset's neuron_df
                    all_bodyids = self._get_all_dataset_bodyids()
                    if all_bodyids:
                        # Load neuron_df and filter by type
                        dataset_safe = self.dataset.replace(':', '_').replace('.', '_')
                        parquet_path = os.path.join(
                            self.script_path, 'datasets', dataset_safe,
                            f"{dataset_safe}_allneurons_neuron_df.parquet"
                        )
                        csv_path = os.path.join(
                            self.script_path, 'datasets', dataset_safe,
                            f"{dataset_safe}_allneurons_neuron_df.csv"
                        )
                        
                        ndf = None
                        if os.path.exists(parquet_path):
                            ndf = pd.read_parquet(parquet_path)
                        elif os.path.exists(csv_path):
                            ndf = self._read_csv(csv_path, index_col=0, low_memory=False)
                        
                        if ndf is not None and 'type' in ndf.columns:
                            type_neurons = ndf[ndf['type'] == ntype]
                            if not type_neurons.empty and 'bodyId' in type_neurons.columns:
                                target_bodyIds.extend([str(x) for x in type_neurons['bodyId'].tolist()])
                except Exception as e:
                    _print(f"  Warning: Failed to get bodyIds for type {ntype}: {e}")
            target_bodyIds = list(set(target_bodyIds))
            _print(f"Found {len(target_bodyIds)} unique bodyIds")
        else:
            # Cache all neurons in dataset
            _print("Target: all neurons in dataset")
            target_bodyIds = self._get_all_dataset_bodyids()
            if target_bodyIds:
                _print(f"Found {len(target_bodyIds)} neurons in dataset")
            else:
                _print("Warning: Could not determine target neurons from datasets/")
                _print("   Ensure neuron_df file exists in datasets/{dataset}/")
                return {'total_neurons': 0, 'already_cached': 0, 'newly_cached': 0,
                        'failed_neurons': [], 'total_connections': 0, 'elapsed_time': 0}
        
        # Check which neurons are already cached using neuron_index
        # This uses O(1) dict lookup after warm-up
        neuron_index = self._load_neuron_index()
        already_cached_set = set()
        
        if not neuron_index.empty:
            # Use O(1) dict lookup
            for bodyId in target_bodyIds:
                bodyId_str = str(bodyId)
                if bodyId_str in self._neuron_index_dict:
                    if self._neuron_index_dict[bodyId_str].get('downstream_complete', False):
                        already_cached_set.add(bodyId_str)
        
        uncached = [x for x in target_bodyIds if str(x) not in already_cached_set]
        already_cached_count = len(already_cached_set)
        
        _print(f"\nCache Status:")
        _print(f"  Already cached: {already_cached_count:,}")
        _print(f"  Need to fetch: {len(uncached):,}")
        
        if not uncached:
            elapsed = time.time() - start_time
            _print("All target neurons already cached!")
            return {
                'total_neurons': len(target_bodyIds),
                'already_cached': already_cached_count,
                'newly_cached': 0,
                'failed_neurons': [],
                'total_connections': self._count_cached_connections(),
                'elapsed_time': elapsed
            }
        
        # Process in batches with progress bar
        total = len(uncached)
        newly_cached = []
        failed_neurons = []
        batch_connections = 0
        total_batches = (total + batch_size - 1) // batch_size
        
        _print(f"\nFetching connections for {total:,} neurons...")
        _print(f"  Strategy: Fetch each batch's downstream, append to cache immediately")
        _print(f"  Memory: No accumulation - each batch saved directly to disk")
        
        # Use tqdm progress bar
        from tqdm import tqdm
        
        # Set flag so _vprint uses tqdm.write instead of print
        self._in_progress_bar = True
        
        # Get cache paths
        db_path = self._get_connection_db_path()
        
        # Ensure cache directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        batch_iter = range(0, total, batch_size)
        if not quiet:
            batch_iter = tqdm(
                batch_iter,
                total=total_batches,
                desc="Building cache",
                unit="batch"
            )
        
        try:
            import psutil
            process = psutil.Process(os.getpid())
        except ImportError:
            process = None

        try:
            for i in batch_iter:
                batch = uncached[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                # Progress callback
                if progress_callback:
                    progress_callback(i, total, f"Batch {batch_num}/{total_batches}")
                
                try:
                    # Fetch connections for this batch (upstream=batch, downstream=None for ALL)
                    connections = self._fetch_connections_bulk(
                        upstream_bodyIds=batch,
                        downstream_bodyIds=None
                    )
                    
                    if connections is not None and not connections.empty:
                        batch_connections += len(connections)
                        
                        # MEMORY-EFFICIENT: Save this batch directly to cache
                        # No accumulation in memory
                        self._append_connections_to_cache(connections, batch)
                        
                        # Mark neurons as fetched
                        newly_cached.extend(batch)
                    else:
                        # Empty connections returned - these neurons genuinely have 0 downstream
                        # FIXED: Still mark as cached so we don't refetch, but with connection_count=0
                        self._update_neuron_index_batch(batch)
                        newly_cached.extend(batch)
                    
                    # Force GC every batch
                    gc.collect()
                    
                    # Update progress bar postfix
                    if not quiet and hasattr(batch_iter, 'set_postfix_str'):
                        mem_usage = f"{process.memory_info().rss / 1024 / 1024:.0f}MB" if process else "?"
                        batch_iter.set_postfix_str(f'neurons={len(newly_cached):,}, conns={batch_connections:,} Mem:{mem_usage}')
                        
                except Exception as e:
                    failed_neurons.extend(batch)
                    if not quiet:
                        # Log the actual error for debugging
                        _print(f"\n  ⚠️ Batch {batch_num} error: {type(e).__name__}: {e}")
                        if hasattr(batch_iter, 'set_postfix_str'):
                            batch_iter.set_postfix_str(f'neurons={len(newly_cached):,}, failed={len(failed_neurons)}')
            
            # Consolidate batch files into main cache file
            # This is where merging happens, but only once at the end
            if newly_cached and not quiet:
                _print(f"\n  ✓ All batches fetched. Consolidating batch files...")
                self._consolidate_batch_files(deduplicate=True)
                
        finally:
            # Reset progress bar flag
            self._in_progress_bar = False
            # Clear any bulk cache to free memory
            if hasattr(self, '_bulk_conn_cache'):
                self._bulk_conn_cache = None
                gc.collect()
        
        elapsed = time.time() - start_time
        
        # Get final cache stats (without loading full cache into memory)
        total_connections = self._count_cached_connections()
        
        # Summary
        _print("\n" + "=" * 60)
        _print("Cache Build Complete")
        _print("=" * 60)
        _print(f"Target neurons: {len(target_bodyIds):,}")
        _print(f"Already cached: {already_cached_count:,}")
        _print(f"Newly cached: {len(newly_cached):,}")
        if failed_neurons:
            _print(f"Failed: {len(failed_neurons):,}")
        _print(f"Total connections in cache: {total_connections:,}")
        _print(f"Time elapsed: {elapsed:.1f} seconds")
        
        if failed_neurons and not quiet:
            print(f"\nFailed neurons (first 10): {failed_neurons[:10]}{'...' if len(failed_neurons) > 10 else ''}")
        
        return {
            'total_neurons': len(target_bodyIds),
            'already_cached': already_cached_count,
            'newly_cached': len(newly_cached),
            'failed_neurons': failed_neurons,
            'total_connections': total_connections,
            'elapsed_time': elapsed
        }
    
    def _fetch_connections_bulk(self, upstream_bodyIds, downstream_bodyIds=None):
        """
        Fetch connections from local data without caching overhead.
        Used by build_connection_cache for faster bulk fetching.
        
        Returns raw connections DataFrame without filtering or enrichment.
        """
        if not upstream_bodyIds:
            return pd.DataFrame()
        
        # For FlyWire/FAFB: use local CSV data
        if 'fafb' in self.dataset.lower() or 'flywire' in self.dataset.lower():
            try:
                import fafb_utils
                project_root = os.path.dirname(os.path.dirname(__file__))
                data_dir = os.path.join(project_root, "datasets", self.dataset)
                if not os.path.exists(data_dir):
                    data_dir = os.path.join(project_root, "datasets", "flywire_FAFB_v783")
                
                if os.path.exists(data_dir):
                    # Suppress fafb_utils print statements
                    import io
                    import sys
                    old_stdout = sys.stdout
                    sys.stdout = io.StringIO()
                    try:
                        _, conn_file = fafb_utils.prepare_fafb_data(data_dir)
                    finally:
                        sys.stdout = old_stdout
                    
                    # Load and filter - use cached full_conn if available
                    if not hasattr(self, '_bulk_conn_cache') or self._bulk_conn_cache is None:
                        self._bulk_conn_cache = self._read_csv(
                            conn_file, 
                            dtype={'pre_root_id': str, 'post_root_id': str}
                        )
                        self._bulk_conn_cache = self._bulk_conn_cache.rename(columns={
                            'pre_root_id': 'bodyId_pre',
                            'post_root_id': 'bodyId_post',
                            'syn_count': 'weight'
                        })
                    
                    upstream_strs = set(str(x) for x in upstream_bodyIds)
                    result = self._bulk_conn_cache[
                        self._bulk_conn_cache['bodyId_pre'].isin(upstream_strs)
                    ].copy()
                    
                    if downstream_bodyIds is not None:
                        downstream_strs = set(str(x) for x in downstream_bodyIds)
                        result = result[result['bodyId_post'].isin(downstream_strs)]
                    
                    if 'roi' not in result.columns:
                        result['roi'] = 'WholeBrain'
                    
                    return result
            except Exception as e:
                # Re-raise to let caller handle/log the error properly
                raise RuntimeError(f"Bulk fetch error for FlyWire/FAFB: {type(e).__name__}: {e}") from e
        
        # For NeuPrint: Direct API call without caching overhead
        # This is used by build_connection_cache which handles caching separately
        try:
            self._ensure_neuprint_client()
            
            from neuprint import fetch_adjacencies, NeuronCriteria
            import statvis as sv
            
            # Ensure bodyIds are integers
            upstream_ints = [int(x) for x in upstream_bodyIds]
            downstream_ints = [int(x) for x in downstream_bodyIds] if downstream_bodyIds else None
            
            if self.simple_fetch:
                from neuprint import fetch_simple_connections
                upstream_criteria = NeuronCriteria(bodyId=upstream_ints)
                downstream_criteria = NeuronCriteria(bodyId=downstream_ints) if downstream_ints else None
                result = fetch_simple_connections(
                    upstream_criteria=upstream_criteria,
                    downstream_criteria=downstream_criteria,
                    min_weight=1,
                    **self.kwargs_fetch
                )
            else:
                neuron_df, roi_conn_df = fetch_adjacencies(
                    sources=upstream_ints,
                    targets=downstream_ints,
                    min_total_weight=1,
                    **self.kwargs_fetch
                )
                # roi_conn_df already has bodyId_pre, bodyId_post, roi, weight
                result = roi_conn_df
            
            return result if result is not None else pd.DataFrame()
            
        except Exception as e:
            # Re-raise to let caller handle/log the error properly
            raise RuntimeError(f"NeuPrint bulk fetch error: {type(e).__name__}: {e}") from e
    
    def _bulk_save_connections(self, connection_list, neurons_fetched):
        """
        Save accumulated connections to cache in bulk.
        Much faster than saving after each batch.
        
        Parameters:
        -----------
        connection_list : list of DataFrames
            List of connection DataFrames to save
        neurons_fetched : list
            List of neurons that were fetched
        """
        if not connection_list:
            return
        
        # Combine all connections
        all_connections = pd.concat(connection_list, ignore_index=True)
        
        # Ensure required columns
        all_connections['bodyId_pre'] = all_connections['bodyId_pre'].astype(str)
        all_connections['bodyId_post'] = all_connections['bodyId_post'].astype(str)
        if 'roi' not in all_connections.columns:
            all_connections['roi'] = ''
        all_connections['cached_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Load existing and merge (conn_db is Polars DataFrame)
        conn_db = self._load_connection_db()
        
        # Convert all_connections to Polars
        all_conn_pl = pl.from_pandas(all_connections)
        
        if not conn_db.is_empty():
            merge_cols = ['bodyId_pre', 'bodyId_post', 'roi']
            combined = pl.concat([conn_db, all_conn_pl], how='diagonal_relaxed')
            combined = combined.unique(subset=merge_cols, keep='first')
        else:
            combined = all_conn_pl
        
        # Save connection database (without rebuilding index - we'll do that at the end)
        db_path = self._get_connection_db_path()
        combined.write_parquet(db_path, compression='gzip')
        self._conn_df_cache = combined
        
        # Update neuron index for all fetched neurons
        neurons_str = [str(x) for x in neurons_fetched]
        neuron_index = self._load_neuron_index()
        
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Count connections per neuron
        conn_counts = all_connections.groupby('bodyId_pre').size().to_dict()
        
        # Update or add entries
        updates = []
        for bodyId in neurons_str:
            count = conn_counts.get(bodyId, 0)
            updates.append({
                'bodyId': bodyId,
                'downstream_complete': True,
                'last_fetched': now,
                'connection_count': count
            })
        
        if updates:
            updates_df = pd.DataFrame(updates)
            if not neuron_index.empty:
                # Merge updates
                neuron_index = neuron_index[~neuron_index['bodyId'].isin(neurons_str)]
                neuron_index = pd.concat([neuron_index, updates_df], ignore_index=True)
            else:
                neuron_index = updates_df
            
            # Save neuron index
            self._save_neuron_index(neuron_index)

    def _get_all_dataset_bodyids(self) -> list:
        """Get all bodyIds from dataset's neuron_df file."""
        dataset_safe = self.dataset.replace(':', '_').replace('.', '_')
        
        # Try parquet first, then CSV
        parquet_path = os.path.join(
            self.script_path, 'datasets', dataset_safe,
            f"{dataset_safe}_allneurons_neuron_df.parquet"
        )
        csv_path = os.path.join(
            self.script_path, 'datasets', dataset_safe,
            f"{dataset_safe}_allneurons_neuron_df.csv"
        )
        
        ndf = None
        if os.path.exists(parquet_path):
            try:
                ndf = pd.read_parquet(parquet_path)
            except Exception:
                pass
        
        if ndf is None and os.path.exists(csv_path):
            try:
                is_fafb = 'fafb' in self.dataset.lower() or 'flywire' in self.dataset.lower()
                if is_fafb:
                    ndf = self._read_csv(csv_path, dtype={'bodyId': str}, low_memory=False)
                else:
                    ndf = self._read_csv(csv_path, index_col=0, low_memory=False)
            except Exception:
                pass
        
        if ndf is not None and 'bodyId' in ndf.columns:
            return [str(x) for x in ndf['bodyId'].unique().tolist()]
        
        return []
    
    def _check_cache_completeness(self, expected_bodyIds: list) -> dict:
        """
        Check cache completeness against expected bodyIds.
        
        Returns dict with:
        - expected: Number of expected neurons
        - cached: Number of neurons in cache
        - missing: Number of missing neurons
        - ratio: Completeness ratio (0.0 to 1.0)
        - cached_bodyids: List of cached bodyIds
        - missing_bodyids: List of missing bodyIds
        """
        expected_set = set(str(x) for x in expected_bodyIds)
        
        # Check connection database for cached neurons
        conn_db = self._load_connection_db()
        cached_set = set()
        
        if conn_db is not None and not conn_db.is_empty():
            if 'bodyId_pre' in conn_db.columns:
                cached_set.update(conn_db['bodyId_pre'].cast(pl.Utf8).unique().to_list())
        
        # Also check neuron_index for neurons with 0 connections
        neuron_index = self._load_neuron_index()
        if neuron_index is not None and not neuron_index.empty:
            if 'downstream_complete' in neuron_index.columns:
                complete_mask = neuron_index['downstream_complete'].astype(bool)
                indexed_bodyids = neuron_index[complete_mask]['bodyId'].astype(str).tolist()
                cached_set.update(indexed_bodyids)
        
        # Calculate completeness
        cached_in_expected = cached_set.intersection(expected_set)
        missing_set = expected_set - cached_set
        
        ratio = len(cached_in_expected) / len(expected_set) if expected_set else 1.0
        
        return {
            'expected': len(expected_set),
            'cached': len(cached_in_expected),
            'missing': len(missing_set),
            'ratio': ratio,
            'cached_bodyids': list(cached_in_expected),
            'missing_bodyids': list(missing_set)
        }
    
    def validate_and_repair_cache(self, quiet: bool = False) -> dict:
        """
        Validate cache integrity and repair inconsistencies.
        
        This function:
        1. Checks if neurons marked 'downstream_complete' actually have connections
        2. Cross-references neuron_index with actual connections.parquet
        3. Marks neurons that were incorrectly flagged as complete as uncached
        4. Enriches neuron_index with type/instance from neuron_df
        
        Returns:
        --------
        dict : Summary with keys:
            - 'total_indexed': Total neurons in neuron_index
            - 'total_with_connections': Neurons that have connections in cache
            - 'falsely_complete': Neurons marked complete but no connections
            - 'repaired': Number of entries repaired
            - 'types_updated': Number of type/instance values updated
        """
        import polars as pl
        
        def _print(msg):
            if not quiet:
                print(msg)
        
        _print("=" * 60)
        _print("Validating and Repairing Connection Cache")
        _print("=" * 60)
        _print(f"Dataset: {self.dataset}")
        
        # Get paths
        index_path = self._get_neuron_index_path()
        conn_path = self._get_connection_db_path()
        
        if not os.path.exists(index_path):
            _print("No neuron_index found. Nothing to repair.")
            return {'total_indexed': 0, 'total_with_connections': 0, 
                    'falsely_complete': 0, 'repaired': 0, 'types_updated': 0}
        
        # Load neuron_index
        ni = pl.read_parquet(index_path)
        total_indexed = len(ni)
        _print(f"Neurons in index: {total_indexed:,}")
        
        # Get neurons that actually have connections
        neurons_with_conns = set()
        if os.path.exists(conn_path):
            conns = pl.read_parquet(conn_path)
            neurons_with_conns = set(conns['bodyId_pre'].unique().to_list())
            _print(f"Neurons with downstream connections: {len(neurons_with_conns):,}")
        
        # Find neurons marked complete but no connections
        complete_mask = ni['downstream_complete'] == True
        complete_ids = set(ni.filter(complete_mask)['bodyId'].to_list())
        falsely_complete = complete_ids - neurons_with_conns
        
        _print(f"Neurons marked complete: {len(complete_ids):,}")
        _print(f"Falsely marked complete (no connections): {len(falsely_complete):,}")
        
        if len(falsely_complete) == 0:
            _print("✓ Cache integrity OK - no repairs needed")
        else:
            _print(f"\n⚠️ Found {len(falsely_complete):,} neurons incorrectly marked as complete")
            _print("   Resetting their downstream_complete flag to False...")
            
            # Convert to pandas for update (polars is read-only)
            ni_pd = ni.to_pandas()
            ni_pd.loc[ni_pd['bodyId'].isin(falsely_complete), 'downstream_complete'] = False
            ni_pd.loc[ni_pd['bodyId'].isin(falsely_complete), 'connection_count'] = -1  # Mark as needing fetch
            
            # Save updated index
            ni_pd.to_parquet(index_path, index=False)
            _print(f"   ✓ Repaired {len(falsely_complete):,} entries")
        
        # Enrich with type/instance from neuron_df
        dataset_safe = self.dataset.replace(':', '_').replace('.', '_')
        ndf_path = os.path.join(
            self.script_path, 'datasets', dataset_safe,
            f"{dataset_safe}_allneurons_neuron_df.csv"
        )
        parquet_ndf_path = ndf_path.replace('.csv', '.parquet')
        
        types_updated = 0
        if os.path.exists(parquet_ndf_path) or os.path.exists(ndf_path):
            _print("\nEnriching neuron_index with type/instance from neuron_df...")
            
            # Load neuron_df
            if os.path.exists(parquet_ndf_path):
                ndf = pl.read_parquet(parquet_ndf_path)
            else:
                ndf = pl.read_csv(ndf_path)
            
            # Ensure bodyId is string
            if 'bodyId' in ndf.columns:
                ndf = ndf.with_columns(pl.col('bodyId').cast(pl.Utf8))
            
            # Load current index again (might have been updated)
            ni = pl.read_parquet(index_path)
            
            # Find neurons with empty type
            empty_type_mask = (pl.col('type').is_null()) | (pl.col('type') == '')
            empty_type_ids = ni.filter(empty_type_mask)['bodyId'].to_list()
            
            if empty_type_ids and 'bodyId' in ndf.columns and 'type' in ndf.columns:
                # Get type/instance info from neuron_df
                ndf_lookup = ndf.filter(pl.col('bodyId').is_in(empty_type_ids))
                
                if len(ndf_lookup) > 0:
                    # Create lookup dict
                    lookup_dict = {}
                    for row in ndf_lookup.iter_rows(named=True):
                        bid = str(row.get('bodyId', ''))
                        lookup_dict[bid] = {
                            'type': row.get('type', ''),
                            'instance': row.get('instance', ''),
                            'post': row.get('post', 0)
                        }
                    
                    # Update in pandas
                    ni_pd = ni.to_pandas()
                    for bid, info in lookup_dict.items():
                        mask = ni_pd['bodyId'] == bid
                        if mask.any():
                            if info['type']:
                                ni_pd.loc[mask, 'type'] = info['type']
                            if info.get('instance'):
                                ni_pd.loc[mask, 'instance'] = info['instance']
                            if info.get('post'):
                                ni_pd.loc[mask, 'post'] = info['post']
                            types_updated += 1
                    
                    # Save
                    ni_pd.to_parquet(index_path, index=False)
                    _print(f"   ✓ Updated {types_updated:,} type/instance values")
        
        # Clear caches so next load picks up repairs
        self._neuron_index_cache = None
        self._neuron_index_dict = {}
        
        _print("\n" + "=" * 60)
        _print("Cache Validation Complete")
        _print("=" * 60)
        
        return {
            'total_indexed': total_indexed,
            'total_with_connections': len(neurons_with_conns),
            'falsely_complete': len(falsely_complete),
            'repaired': len(falsely_complete),
            'types_updated': types_updated
        }
    
    def _count_cached_connections(self) -> int:
        """Count total connections in cache."""
        # Return in-memory count if available
        if self._conn_df_cache is not None and not self._conn_df_cache.empty:
            return len(self._conn_df_cache)
            
        # Optimization: If using parquet, try to read metadata only to avoid loading full file
        db_path = self._get_connection_db_path()
        if os.path.exists(db_path):
            try:
                # Try pyarrow first
                import pyarrow.parquet as pq
                metadata = pq.read_metadata(db_path)
                return metadata.num_rows
            except ImportError:
                pass
            except Exception:
                pass
                
        # Fallback to loading full DB (legacy behavior)
        conn_db = self._load_connection_db()
        if conn_db is not None and not conn_db.is_empty():
            return len(conn_db)
        return 0

    def build_connectivity_profile_cache(
        self,
        neuron_types: list = None,
        top_k: int = 10,
        top_m: int = 5,
        expand_2hop: bool = True,
        max_neurons: int = None,
        force_refresh: bool = False,
        progress_callback: callable = None
    ) -> dict:
        """
        Build connectivity profile cache for neuron types using ConnectivityProfiler.
        
        Connectivity profiles are used for homolog finding and cross-dataset 
        comparisons. This delegates to the ConnectivityProfiler.
        
        Parameters:
        -----------
        neuron_types : list, optional
            List of neuron types to cache. If None, caches all types in dataset.
        top_k : int
            Store top N partners by weight (default: 10)
        top_m : int  
            Ensure at least M unique types via expansion (default: 5)
        expand_2hop : bool
            Enable 2-hop expansion for untyped partners (default: True)
        max_neurons : int, optional
            Limit to first N neurons (for testing)
        force_refresh : bool
            Force rebuild even if profiles exist in cache
        progress_callback : callable, optional
            Callback function(current, total, type_name) for progress updates
        
        Returns:
        --------
        dict : Summary with keys:
            - 'total_profiles': Number of profiles built
            - 'profiles': Dict mapping neuron_type to ConnectivityProfile
            - 'failed_types': List of types that failed
            - 'elapsed_time': Time taken in seconds
        
        Example:
        --------
        >>> fnc = FindNeuronConnection(dataset='hemibrain:v1.2.1', ...)
        >>> result = fnc.build_connectivity_profile_cache(top_k=10, top_m=5)
        >>> print(f"Built {result['total_profiles']} profiles")
        """
        import time
        start_time = time.time()
        
        print("=" * 60)
        print("Building Connectivity Profile Cache")
        print("=" * 60)
        print(f"Dataset: {self.dataset}")
        print(f"Parameters: top_k={top_k}, top_m={top_m}, expand_2hop={expand_2hop}")
        if neuron_types:
            print(f"Neuron types: {len(neuron_types)} specified")
        else:
            print("Neuron types: ALL")
        if max_neurons:
            print(f"Max neurons: {max_neurons}")
        print()
        
        try:
            from comparison.connectivity_profiler import ConnectivityProfiler, ProfilerConfig
        except ImportError:
            print("❌ Could not import ConnectivityProfiler")
            print("   Make sure comparison module is available")
            return {'total_profiles': 0, 'profiles': {}, 'failed_types': [], 
                    'elapsed_time': 0}
        
        # Create profiler config
        config = ProfilerConfig(
            top_k_bodyid=top_k,
            top_m_type=top_m,
            expand_untyped_2hop=expand_2hop,
            use_cache=True,
            verbose=self.verbose_mode != 'none'
        )
        
        profiler = ConnectivityProfiler(config)
        
        # Build profiles
        profiles = profiler.build_connectivity_profile_cache(
            dataset=self.dataset,
            neuron_types=neuron_types,
            top_k_bodyid=top_k,
            top_m_type=top_m,
            expand_untyped_2hop=expand_2hop,
            force_refresh=force_refresh,
            max_neurons=max_neurons,
            progress_callback=progress_callback
        )
        
        elapsed = time.time() - start_time
        
        # Extract failed types (compare requested vs returned)
        failed_types = []
        if neuron_types:
            returned_types = set(profiles.keys())
            failed_types = [t for t in neuron_types if t not in returned_types]
        
        # Summary
        print()
        print("=" * 60)
        print("Connectivity Profile Cache Complete")
        print("=" * 60)
        print(f"Total profiles built: {len(profiles)}")
        if failed_types:
            print(f"Failed types: {len(failed_types)}")
        print(f"Elapsed time: {elapsed:.1f} seconds")
        
        return {
            'total_profiles': len(profiles),
            'profiles': profiles,
            'failed_types': failed_types,
            'elapsed_time': elapsed
        }

    def InitializeNeuronInfo(self):
        # Ensure neuprint Client is set for the CORRECT dataset
        if self.client_type != 'flywire':
            self._ensure_neuprint_client()
        ''' initialize neuron info '''
        self._vprint('Fetching source and target neurons...', level='simple')
        
        # Determine client to pass
        active_client = self.client_flywire if self.client_type == 'flywire' else self.client_hemibrain
        
        # Optimization: when max_interlayer=-1 and source==target, fetch only once
        self._source_target_identical = (self.max_interlayer == -1 and self.sourceNeurons == self.targetNeurons)
        
        # Determine verbose for getNeurons based on verbose_mode
        neurons_verbose = (self.verbose_mode != 'silent')
        
        if self._source_target_identical:
            self._vprint('\033[36mOptimization: source==target with max_interlayer=-1, fetching only one set\033[0m', level='simple')
            self.source_df, _, source_fname_auto, self.source_criteria = sv.getNeurons(
                self.sourceNeurons, 
                dataset=self.dataset,
                custom_group_names=self.custom_source_group_names if self.custom_source_group_names else None,
                client=active_client,
                verbose=neurons_verbose
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
                client=active_client,
                verbose=neurons_verbose
            )
            self.target_df, _, target_fname_auto, self.target_criteria = sv.getNeurons(
                self.targetNeurons, 
                dataset=self.dataset,
                custom_group_names=self.custom_target_group_names if self.custom_target_group_names else None,
                client=active_client,
                verbose=neurons_verbose
            )
        
        # Apply label mapping if available
        if self.label_mapper and not self.label_mapper.is_empty:
            self._vprint(f'\033[36mApplying label mapping to source/target neurons...\033[0m', level='simple')
            # Apply to source_df
            if not self.source_df.empty and 'type' in self.source_df.columns:
                # Create a copy to avoid SettingWithCopyWarning
                self.source_df = self.source_df.copy()
                # Map types to standardized labels
                # We use 'source' role for source neurons
                self.source_df['std_label'] = self.source_df.apply(
                    lambda row: self.label_mapper.get_std_label(
                        self.dataset, 
                        row['type'] if pd.notna(row['type']) else row['bodyId'], 
                        'source'
                    ), axis=1
                )
                # Overwrite type with std_label where available
                mask = self.source_df['std_label'] != ''
                self.source_df.loc[mask, 'type'] = self.source_df.loc[mask, 'std_label']
                # Drop temporary column
                self.source_df = self.source_df.drop(columns=['std_label'])
                
            # Apply to target_df
            if not self.target_df.empty and 'type' in self.target_df.columns:
                self.target_df = self.target_df.copy()
                # We use 'target' role for target neurons
                self.target_df['std_label'] = self.target_df.apply(
                    lambda row: self.label_mapper.get_std_label(
                        self.dataset, 
                        row['type'] if pd.notna(row['type']) else row['bodyId'], 
                        'target'
                    ), axis=1
                )
                mask = self.target_df['std_label'] != ''
                self.target_df.loc[mask, 'type'] = self.target_df.loc[mask, 'std_label']
                self.target_df = self.target_df.drop(columns=['std_label'])
        
        if self.max_interlayer > 2 or len(self.source_df) > 200:
            self.simple_fetch = False
            self._vprint('\033[33mLarge data detected!!! simple_fetch is set to False, using fetch_adjacencies()\033[0m', level='simple')

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
        
        self._vprint(f'Processing: {self.source_fname} to {self.target_fname}', level='simple')
        self._vprint(f'Source neurons ({self.source_fname}) in processing: {len(self.source_df)}', level='simple')
        self._vprint(f'Target neurons ({self.target_fname}) in processing: {len(self.target_df)}', level='simple')
        
        if self.saveas:
            if os.path.isabs(self.saveas):
                self.save_folder = self.saveas
            else:
                self.save_folder = os.path.join(self.output_dir, self.saveas)
        elif not self.save_folder: # if save_folder is not specified, save in data_folder, with auto-generated name
            # Create base folder with just source_to_target (no parameters)
            folder_name = self.source_fname + '_to_' + self.target_fname
            if self.folder_prefix:
                folder_name = f"{self.folder_prefix}_{folder_name}"
            self.save_folder = os.path.join(self.output_dir, folder_name)
        elif not os.path.isabs(self.save_folder): # if save_folder is not absolute path, save in data_folder with specified relative path and name
            self.save_folder = os.path.join(self.output_dir, self.save_folder)
        if not os.path.exists(self.save_folder): os.makedirs(self.save_folder)
        self._vprint(f'data will be saved in: {self.save_folder}\n', level='simple')
        
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
            self._vprint('\033[36mmax_interlayer=-1: Neurons fetched (no connections will be queried)\033[0m', level='simple')
            self._vprint('Use FetchNeuronsOnly() for connectivity profile analysis.', level='simple')
    
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
        self._save_df_to_csv_polars(self.source_df, source_path)
        
        # Save target neurons
        target_path = os.path.join(output_dir, f'{filename_prefix}_target_neurons.csv')
        if hasattr(self, '_source_target_identical') and self._source_target_identical:
            # When source==target, just copy the reference
            self._save_df_to_csv_polars(self.target_df, target_path)
            print(f'Target neurons: same as source (saved separately)')
        else:
            self._save_df_to_csv_polars(self.target_df, target_path)
        
        # Save parameters
        params_path = os.path.join(output_dir, f'{filename_prefix}_parameters.csv')
        if hasattr(self, 'parameter_df'):
            self._save_df_to_csv_polars(self.parameter_df, params_path)
        
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
            aggregate_method='product',  # Type-level prob = 1 - product(bodyId-level block_prob)
            label_mapper=self.label_mapper
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
            
            self._save_df_to_csv_polars(self.parameter_df, base_name + '_parameters.csv')
            self._save_df_to_csv_polars(self.source_df, base_name + '_source_info.csv')
            self._save_df_to_csv_polars(self.target_df, base_name + '_target_info.csv')
            self._save_df_to_csv_polars(self.source_in_conn, base_name + '_source_in_connection.csv')
            self._save_df_to_csv_polars(self.target_in_conn, base_name + '_target_in_connection.csv')
            self._save_df_to_csv_polars(self.conn_type, base_name + '_connection_groupby_type.csv')
            
            # Add custom group sheets if custom grouping was used
            if self.conn_group is not None:
                self._save_df_to_csv_polars(self.conn_group, base_name + '_connection_groupby_custom.csv')
                if not self.largeTargetSet:
                    self._save_df_to_csv_polars(self.conn_matrix_group, base_name + '_connectionMatrix_group.csv', index=True)
                    self._save_df_to_csv_polars(self.conn_matrix_ratio_group, base_name + '_connectionRatioMat_group.csv', index=True)
                else:
                    self._save_df_to_csv_polars(self.conn_matrix_group.transpose(), base_name + '_connectionMatrix_group.csv', index=True)
                    self._save_df_to_csv_polars(self.conn_matrix_ratio_group.transpose(), base_name + '_connectionRatioMat_group.csv', index=True)
            
            # Type-level matrices
            if not self.largeTargetSet:
                self._save_df_to_csv_polars(self.conn_matrix_type, base_name + '_connectionMatrix_type.csv', index=True)
                self._save_df_to_csv_polars(self.cmat_full_type, base_name + '_connMat_type_full.csv', index=True)
                self._save_df_to_csv_polars(self.transitionMat_type, base_name + '_transmissionMat_type.csv', index=True)
                self._save_df_to_csv_polars(self.conn_matrix_ratio_type, base_name + '_connectionRatioMat_type.csv', index=True)
                self._save_df_to_csv_polars(self.ratioMat_full_type, base_name + '_ratioMat_type_full.csv', index=True)
            else:
                self._save_df_to_csv_polars(self.conn_matrix_type.transpose(), base_name + '_connectionMatrix_type.csv', index=True)
                self._save_df_to_csv_polars(self.cmat_full_type.transpose(), base_name + '_connMat_type_full.csv', index=True)
                self._save_df_to_csv_polars(self.transitionMat_type.transpose(), base_name + '_transmissionMat_type.csv', index=True)
                self._save_df_to_csv_polars(self.conn_matrix_ratio_type.transpose(), base_name + '_connectionRatioMat_type.csv', index=True)
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
                self._save_df_to_csv_polars(self.parameter_df, output_params_csv)
            else:
                output_params_excel = os.path.join(self.direct_folder,self.source_fname+'_to_'+self.target_fname+'_bodyId_parameters_snp'+str(self.min_synapse_num)+'.xlsx')
                with pd.ExcelWriter(output_params_excel, mode='w', engine='xlsxwriter') as dataWriter:
                    self.parameter_df.to_excel(dataWriter,sheet_name='parameters')
                    worksheet = dataWriter.sheets['parameters']
                    worksheet.set_column('A:A', 30, dataWriter.book.add_format({'bold': True, 'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
                    worksheet.set_column('B:B', 30, dataWriter.book.add_format({'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
            
            # Save bodyId connection data as CSV
            output_bodyid_csv = os.path.join(self.direct_folder,self.source_fname+'_to_'+self.target_fname+'_bodyId_connections_snp'+str(self.min_synapse_num)+'.csv')
            self._save_df_to_csv_polars(self.conn_df, output_bodyid_csv)
            print(f'  ✓ Saved to: {output_bodyid_csv}')
            
            # Save matrices as separate CSVs
            if not self.largeTargetSet:
                self._save_df_to_csv_polars(self.conn_matrix_bodyId, os.path.join(self.direct_folder,self.source_fname+'_to_'+self.target_fname+'_connectionMatrix_bodyId.csv'), index=True)
                self._save_df_to_csv_polars(self.transitionMat_bodyId, os.path.join(self.direct_folder,self.source_fname+'_to_'+self.target_fname+'_transmissionMat_bodyId.csv'), index=True)
            else:
                self._save_df_to_csv_polars(self.conn_matrix_bodyId.transpose(), os.path.join(self.direct_folder,self.source_fname+'_to_'+self.target_fname+'_connectionMatrix_bodyId.csv'), index=True)
                self._save_df_to_csv_polars(self.transitionMat_bodyId.transpose(), os.path.join(self.direct_folder,self.source_fname+'_to_'+self.target_fname+'_transmissionMat_bodyId.csv'), index=True)
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
        
        # Use FastGraph for pathfinding
        print('\nUsing FastGraph for pathfinding...')
        
        # Build graph from conn_layers
        G = FastGraph()
        for conn in conn_layers:
            G.build_from_dataframe(conn, 'bodyId_pre', 'bodyId_post', 'weight')
        
        sources = list(self.source_df['bodyId'].unique())
        # Targets found in the network (Checked=True)
        targets = list(self.target_df[self.target_df['Checked'] == True]['bodyId'].unique())
        cutoff = self.max_interlayer + 1
        
        paths_found = []
        # Use memoized DFS to find all paths
        for path in G.find_paths_memoized_dfs(sources, targets, cutoff, verbose=True):
            paths_found.append(path)
            
        path_count = len(paths_found)
        pairs_with_paths = len(set((p[0], p[-1]) for p in paths_found))
        print(f'Found {path_count} paths between {pairs_with_paths} source-target pairs.')
        
        # Process paths to extract neurons and edges
        neurons_in_paths = set()
        edges_in_paths = set()
        edges_in_paths_with_layer = set()
        
        for path in paths_found:
            neurons_in_paths.update(path)
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edges_in_paths.add((u, v))
                # Determine layer index
                # Since conn_layers are sequential (L0->L1, L1->L2...), 
                # edge at index i in path corresponds to layer i
                edges_in_paths_with_layer.add((i, u, v))
        
        # Reconstruct conn_inpath and conn_types
        conn_inpath = pd.DataFrame()
        conn_types = pd.DataFrame()
        weight_layers = {}
        
        # Filter conn_layers based on edges_in_paths_with_layer
        for i in range(len(conn_layers)):
            conn = conn_layers[i]
            
            # Filter rows where (i, bodyId_pre, bodyId_post) is in edges_in_paths_with_layer
            # Create a set of (pre, post) for this layer for fast lookup
            valid_edges_in_layer = set()
            for layer_idx, u, v in edges_in_paths_with_layer:
                if layer_idx == i:
                    valid_edges_in_layer.add((u, v))
            
            if not valid_edges_in_layer:
                continue
                
            # Filter dataframe
            # Vectorized filtering using MultiIndex or map
            # Create a temporary index for filtering
            conn_idx = pd.MultiIndex.from_frame(conn[['bodyId_pre', 'bodyId_post']])
            mask = conn_idx.isin(valid_edges_in_layer)
            conn_df = conn[mask].copy()
            
            if len(conn_df) == 0: continue
            
            # Get all neurons involved in this layer's connections (for accurate ratio calculation)
            bodyIds_in_layer = np.unique(np.concatenate([conn_df['bodyId_pre'].unique(), conn_df['bodyId_post'].unique()]))
            neurons_in_layer_df = self._fetch_neurons_local_or_api(bodyIds_in_layer.tolist(), columns=['bodyId', 'type', 'post'])
            
            conn_df, conn_type, conn_group = sv.EnrichConnectionTable(
                conn_df, 
                traversal_probability_threshold=0,
                dataset=self.dataset,
                script_path=self.script_path,
                target_neurons_df=neurons_in_layer_df,
                label_mapper=self.label_mapper
            )
            conn_df.insert(loc=0,column='conn_layer',value=str(i)+'->'+str(i+1))
            conn_type.insert(loc=0,column='conn_layer',value=str(i)+'->'+str(i+1))
            if conn_group is not None:
                conn_group.insert(loc=0,column='conn_layer',value=str(i)+'->'+str(i+1))
            conn_inpath = pd.concat([conn_inpath,conn_df])
            conn_types = pd.concat([conn_types,conn_type])
            
            weight_layers.update({str(i)+'->'+str(i+1): conn_df['weight'].sum()})
            
        # Reconstruct neuron_layers for visualization
        neuron_layers = []
        if not conn_inpath.empty:
            # Get all unique layer indices from conn_inpath
            # conn_layer format is "i->i+1"
            layers = sorted(conn_inpath['conn_layer'].unique(), key=lambda x: int(x.split('->')[0]))
            
            if layers:
                first_layer = layers[0]
                neuron_layers.append(conn_inpath[conn_inpath['conn_layer'] == first_layer]['bodyId_pre'].unique())
                
                for layer in layers:
                    neuron_layers.append(conn_inpath[conn_inpath['conn_layer'] == layer]['bodyId_post'].unique())
        else:
             neuron_layers = [self.source_df['bodyId'].unique()]
            
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
            self._vprint(f'  💾 Saving data as CSV files to: {csv_folder}', level='simple')
            self._save_df_to_csv_polars(self.parameter_df, os.path.join(csv_folder, 'parameters.csv'))
            self._save_df_to_csv_polars(self.source_df, os.path.join(csv_folder, 'source_neurons.csv'))
            self._save_df_to_csv_polars(self.target_df, os.path.join(csv_folder, 'target_neurons.csv'))
            self._save_df_to_csv_polars(totalweight_df, os.path.join(csv_folder, 'total_weight_layer.csv'))
            self._save_df_to_csv_polars(conn_types, os.path.join(csv_folder, 'connection_type.csv'))
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
                self._save_df_to_csv_polars(self.parameter_df, os.path.join(bodyid_folder, 'parameters.csv'))
            
            # Save bodyId connection data as CSV
            output_bodyid_csv = os.path.join(bodyid_folder, 'connection_info_bodyId.csv')
            self._save_df_to_csv_polars(conn_inpath, output_bodyid_csv)
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
        G_type = FastGraph()
        G_type.build_from_dataframe(conn_types, 'type_pre', 'type_post', 'weight')
        
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
                # Find all simple paths with length <= max_interlayer + 1
                for path in G_type.all_simple_paths(source_type, target_type, cutoff=self.max_interlayer + 1):
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
             self._save_df_to_csv_polars(path_df_type, os.path.join(self.path_folder, f'{self.source_fname}_to_{self.target_fname}_path_type.csv'))
             
             # Save excluded paths in data_details
             csv_folder = os.path.join(self.path_folder, 'data_details')
             os.makedirs(csv_folder, exist_ok=True)
             self._save_df_to_csv_polars(path_df_type_excluded, os.path.join(csv_folder, 'path_type_excluded.csv'))
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
            G_bodyId = FastGraph()
            G_bodyId.build_from_dataframe(conn_inpath, 'bodyId_pre', 'bodyId_post', 'weight')
            
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
                    if target_id not in G_bodyId:
                        continue
                    # Find all simple paths with length <= max_interlayer + 1
                    for path in G_bodyId.all_simple_paths(source_id, target_id, cutoff=self.max_interlayer + 1):
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
                self._save_df_to_csv_polars(path_df_bodyId, output_path_csv)
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
                    self._save_df_to_csv_polars(path_df_bodyId, output_path_csv)
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
                ndf_complete = self._read_csv(dataset_path, header=0, index_col=None, dtype={'bodyId': str}, low_memory=False)
            else:
                ndf_complete = self._read_csv(dataset_path, header=0, index_col=0, low_memory=False)
        else:
            if 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower():
                self._vprint(f'   ⚠️  Local dataset not found for FlyWire/FAFB. Skipping interlayer info fetch (NeuPrint API not supported for this dataset).', level='full')
                ndf_complete = pd.DataFrame()
            else:
                self._vprint(f'   Local dataset not found, will use API calls', level='full')
                # Ensure client is logged in for the CORRECT dataset
                self._ensure_neuprint_client()
        
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
                self._save_df_to_csv_polars(interlayers[i], layer_csv)
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
    
    def FindAllPath(self, find_bodyId_path=True, forward_only=True, exclude_searched_neurons=None, 
                    use_graph_cache=True):
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
        use_graph_cache : bool, default=True
            If True, caches the bodyId-level graph at the lowest threshold seen and reuses
            it for higher thresholds by filtering edges. This significantly speeds up 
            comparison runs that test multiple thresholds on the same dataset.
            
            Cache reuse rules:
            - If cached_threshold <= current_threshold: Reuse cached graph, filter edges
            - If cached_threshold > current_threshold: Rebuild from scratch (need more edges)
        exclude_searched_neurons : bool, deprecated
            Deprecated parameter name. Use forward_only instead.
            If provided, it will override forward_only for backward compatibility.
        
        Logic:
        1. Fetch connections layer by layer, discovering network structure
        2. Identify which target neurons exist in the searched network
        3. Find all paths from sources to targets with path length ≤ max_interlayer
        '''
        import polars as pl
        
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
        # Filter out internal/private attributes (starting with '_') and large cached data
        public_attrs = {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_') and k not in ('source_df', 'target_df', 'client_hemibrain', 'client_flywire')
        }
        with open(os.path.join(self.allpath_folder, 'all_attributes.json'), 'w') as f:
            json.dump(public_attrs, f, indent=4, default=lambda o: '<not serializable>')
        
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
        
        # ============================================================================
        # GRAPH CACHE LOGIC: Check if we can reuse cached graph from lower threshold
        # ============================================================================
        global _FINDALLPATH_GRAPH_CACHE
        
        # Generate cache key based on query parameters (not threshold)
        # Threshold is handled separately - we can filter a lower-threshold graph
        source_hash = hash(tuple(sorted(str(s) for s in source_ID)))
        target_hash = hash(tuple(sorted(str(t) for t in target_ID)))
        cache_key = f"{self._dataset_safe}_{source_hash}_{target_hash}_{self.max_interlayer}"
        
        cached_data = _FINDALLPATH_GRAPH_CACHE.get(cache_key) if use_graph_cache else None
        use_cached_graph = False
        
        if cached_data is not None:
            cached_threshold = cached_data.get('threshold', float('inf'))
            # Can reuse if cached threshold <= current threshold (more edges in cache)
            if cached_threshold <= self.min_synapse_num:
                use_cached_graph = True
                self._vprint(f'\n⚡ Reusing cached graph from threshold={cached_threshold} (current={self.min_synapse_num})', level='simple')
            else:
                # Cached threshold is higher - need to rebuild with lower threshold
                self._vprint(f'\n📊 Cache exists at threshold={cached_threshold}, but need threshold={self.min_synapse_num} - rebuilding', level='full')
        
        if use_cached_graph:
            # ===== FAST PATH: Reuse cached graph and filter by threshold =====
            all_connections = cached_data['all_connections']
            layer_neurons = cached_data['layer_neurons']
            all_neurons_in_network = cached_data['all_neurons_in_network']
            # Note: targets_found will be recomputed in Phase 2 based on filtered graph
            
            # Filter connections by current threshold
            if self.min_synapse_num > cached_threshold:
                filtered_connections = []
                for conn_pl in all_connections:
                    if not conn_pl.is_empty():
                        filtered = conn_pl.filter(pl.col('weight') >= self.min_synapse_num)
                        filtered_connections.append(filtered)
                    else:
                        filtered_connections.append(conn_pl)
                all_connections_filtered = filtered_connections
                self._vprint(f'  Filtered connections by weight >= {self.min_synapse_num}', level='full')
            else:
                all_connections_filtered = all_connections
            
            # Skip to Phase 2 - target identification still needed for this threshold
            # because some targets may become unreachable after filtering
        else:
            # ===== STANDARD PATH: Fetch connections and build graph =====
            all_connections_filtered = None  # Will be set in Phase 1
        
        # PHASE 1: Fetch all connections in the network up to max_interlayer layers
        if not use_cached_graph:
            if self.verbose_mode == 'simple':
                self._vprint(f'\nPhase 1: Fetching all network layers...', level='simple')
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
                
                # Convert to Polars for faster processing
                if not conn_df.empty:
                    # Ensure string types
                    conn_df['bodyId_pre'] = conn_df['bodyId_pre'].astype(str)
                    conn_df['bodyId_post'] = conn_df['bodyId_post'].astype(str)
                    
                    conn_pl = pl.from_pandas(conn_df)
                    
                    # Add conn_layer column
                    conn_pl = conn_pl.with_columns(pl.lit(f'{layer_idx}->{layer_idx+1}').alias('conn_layer'))
                    
                    all_connections.append(conn_pl)
                    
                    # Collect all downstream neurons for next layer
                    post_neurons = set(conn_pl['bodyId_post'].unique().to_list())
                else:
                    all_connections.append(pl.DataFrame())
                    post_neurons = set()
                
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
            
            # Cache the graph data for future runs at higher thresholds
            if use_graph_cache:
                _FINDALLPATH_GRAPH_CACHE[cache_key] = {
                    'threshold': self.min_synapse_num,
                    'all_connections': all_connections,
                    'layer_neurons': layer_neurons,
                    'all_neurons_in_network': all_neurons_in_network,
                }
                self._vprint(f'  💾 Cached graph at threshold={self.min_synapse_num} for future reuse', level='full')
            
            # Use the freshly fetched connections
            all_connections_filtered = all_connections
        else:
            # Using cached data - all_connections_filtered was already set above
            self._vprint(f'Phase 1: Skipped (using cached graph)', level='simple')
            self._vprint(f'  Cached neurons in network: {len(all_neurons_in_network)}', level='full')
            self._vprint(f'  Cached layers: {len(layer_neurons)}', level='full')
        
        # PHASE 2: Identify which targets exist in the searched network
        if self.verbose_mode == 'simple':
            self._vprint(f'Phase 2: Identifying Targets...', level='simple')
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
            self._vprint(f'Phase 3: Building Graph and Finding Paths...', level='simple')
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
        G = FastGraph()
        for conn_df in all_connections_filtered:
            G.build_from_dataframe(conn_df, 'bodyId_pre', 'bodyId_post', 'weight')
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
        # self._vprint(f'Using optimized DFS algorithm (explores shared path segments only once)', level='full')
        
        # Select pathfinding algorithm
        algo = self.pathfinding
        valid_algos = ['DP', 'Bidirectional', 'DFS', 'MemoizedDFS', 'Backtracking']
        if algo not in valid_algos:
            self._vprint(f'Warning: Unknown pathfinding algorithm "{algo}", defaulting to "DP"', level='always')
            algo = 'DP'
        
        path_count = 0
        all_paths = []  # Initialize list to store all found paths
        pairs_with_paths_dict = {}
        
        import time
        start_time = time.time()
        
        path_gen = None
        
        if algo == 'Bidirectional':
            if self.verbose_mode == 'simple':
                self._vprint(f'Finding path [bidirectional]...', level='simple')
            elif self.verbose_mode == 'full':
                self._vprint('Using bidirectional BFS (layer intersection)...', level='full')
            
            path_gen = G.find_paths_bidirectional_bfs(source_ID, targets_found, self.max_interlayer + 1, verbose=(self.verbose_mode in ['simple', 'full']))
            
        elif algo == 'MemoizedDFS':
            if self.verbose_mode == 'simple':
                self._vprint(f'Finding path [memoized DFS]...', level='simple')
            elif self.verbose_mode == 'full':
                self._vprint('Using Bidirectional DFS (Meet-in-the-middle)...', level='full')
                self._vprint('   ⚡ Optimized for memory: storing L/2 paths', level='full')
            
            path_gen = G.find_paths_meet_in_the_middle(source_ID, targets_found, self.max_interlayer + 1, verbose=(self.verbose_mode in ['simple', 'full']))
            
        elif algo == 'DFS':
            if self.verbose_mode == 'simple':
                self._vprint(f'Finding path [standard DFS]...', level='simple')
            elif self.verbose_mode == 'full':
                self._vprint('Using standard DFS pathfinding (recursive)...', level='full')
            
            # Use Backward Memoized DFS as a proxy for standard DFS behavior (finding all paths)
            path_gen = G.find_paths_memoized_dfs(source_ID, targets_found, self.max_interlayer + 1, direction='backward', verbose=(self.verbose_mode in ['simple', 'full']))

        elif algo == 'Backtracking':
            if self.verbose_mode == 'simple':
                self._vprint(f'Finding path [backtracking]...', level='simple')
            elif self.verbose_mode == 'full':
                self._vprint('Using backward DFS with backtracking (no memoization)...', level='full')
            
            path_gen = G.find_paths_dfs_backtracking(source_ID, targets_found, self.max_interlayer + 1, verbose=(self.verbose_mode in ['simple', 'full']))

        else: # algo == 'DP'
            if self.verbose_mode == 'simple':
                self._vprint(f'Finding path [optimized DP]...', level='simple')
            elif self.verbose_mode == 'full':
                self._vprint('Using optimized backward search (DP)...', level='full')
            
            path_gen = G.find_paths_backward_dp(source_ID, targets_found, self.max_interlayer + 1, verbose=(self.verbose_mode in ['simple', 'full']))

        # Common collection logic
        if path_gen:
            if self.verbose_mode in ['simple', 'full']:
                    path_iter = tqdm(path_gen, desc="Processing paths", leave=False, unit="path")
            else:
                    path_iter = path_gen

            for p in path_iter:
                path_count += 1
                all_paths.append(p)  # Collect path
                s = p[0]
                t = p[-1]
                pairs_with_paths_dict[(s, t)] = True
                neurons_in_paths.update(p)
                for i in range(len(p) - 1):
                    edges_in_paths.add((p[i], p[i+1]))
                    edges_in_paths_with_layer.add((i, p[i], p[i+1]))
            
            pairs_with_paths = len(pairs_with_paths_dict)
            
            elapsed = time.time() - start_time
            if self.verbose_mode == 'simple':
                self._vprint('Done', level='simple')
                self._vprint('building paths...', level='simple', end='', flush=True)
            elif self.verbose_mode == 'full':
                self._vprint(f'   Pathfinding completed in {elapsed:.1f}s', level='full')
        
        self._vprint(f'\n✅ Pathfinding complete!', level='full')
        self._vprint(f'   Total paths found: {path_count:,}', level='full')
        self._vprint(f'   Neurons in valid paths: {len(neurons_in_paths):,}', level='full')
        self._vprint(f'   Unique edges in valid paths: {len(edges_in_paths):,}', level='full')
        self._vprint(f'   Layer-specific edges in valid paths: {len(edges_in_paths_with_layer):,}', level='full')
        
        # Now extract connections, keeping ALL layer-specific occurrences
        # This means if neuron A→B exists in both Layer 0→1 and Layer 2→3, both are kept
        # Initialize lists for accumulation (more efficient than repeated concat)
        conn_inpath_list = []
        conn_types_list = []
        conn_groups_list = []
        weight_layers = {}
        
        iterator = all_connections
        if self.verbose_mode in ['simple', 'full']:
            iterator = tqdm(all_connections, desc="Building paths", unit="layer", leave=True)
            
        for layer_idx, conn_df in enumerate(iterator):
            # Skip empty connection DataFrames
            if conn_df.is_empty():
                continue
                
            # Get the actual layer index from the conn_layer label
            layer_label = conn_df['conn_layer'][0]
            actual_layer_idx = int(layer_label.split('->')[0])
            
            # Filter to keep only edges that are in valid paths for THIS specific layer
            # Create a set of valid (pre, post) for this layer
            valid_pairs = { (u, v) for (l, u, v) in edges_in_paths_with_layer if l == actual_layer_idx }
            
            if not valid_pairs:
                continue
                
            # Create a DataFrame for filtering
            valid_pairs_df = pl.DataFrame(list(valid_pairs), schema=['bodyId_pre', 'bodyId_post'], orient='row')
            # Ensure types match
            valid_pairs_df = valid_pairs_df.with_columns([
                pl.col('bodyId_pre').cast(pl.Utf8),
                pl.col('bodyId_post').cast(pl.Utf8)
            ])
            
            # Filter conn_df (inner join is efficient for filtering)
            conn_filtered = conn_df.join(valid_pairs_df, on=['bodyId_pre', 'bodyId_post'], how='inner')
            
            if conn_filtered.is_empty():
                continue
            
            # Remove conn_layer temporarily (will add back after enrichment)
            conn_filtered_no_layer = conn_filtered.drop('conn_layer')
            
            # Get all neurons involved in this layer's connections (for accurate ratio calculation)
            bodyIds_in_layer = pl.concat([conn_filtered_no_layer['bodyId_pre'], conn_filtered_no_layer['bodyId_post']]).unique()
            
            # _fetch_neurons_local_or_api likely returns Pandas, convert to Polars
            neurons_in_layer_df_pd = self._fetch_neurons_local_or_api(bodyIds_in_layer.to_list(), columns=['bodyId', 'type', 'post'])
            neurons_in_layer_df = pl.from_pandas(neurons_in_layer_df_pd)
            
            # Enrich with traversal probability (use local dataset if available)
            conn_enriched, conn_type, conn_group = EnrichConnectionTablePolars(
                conn_filtered_no_layer,
                dataset=self.dataset, 
                script_path=self.script_path,
                target_neurons_df=neurons_in_layer_df,
                label_mapper=self.label_mapper
            )
            
            # Add conn_layer column AFTER enrichment
            conn_enriched = conn_enriched.with_columns(pl.lit(layer_label).alias('conn_layer'))
            conn_type = conn_type.with_columns(pl.lit(layer_label).alias('conn_layer'))
            if conn_group is not None:
                conn_group = conn_group.with_columns(pl.lit(layer_label).alias('conn_layer'))
            
            if not conn_enriched.is_empty():
                conn_inpath_list.append(conn_enriched)
            
            if not conn_type.is_empty():
                conn_types_list.append(conn_type)
                
            if conn_group is not None and not conn_group.is_empty():
                conn_groups_list.append(conn_group)
            
            weight_layers[layer_label] = conn_enriched['weight'].sum()
            
            self._vprint(f'Layer {layer_label}: {len(conn_filtered)} connections kept', level='full')
        
        # Concatenate all results at once (avoids FutureWarning about empty/NA entries)
        if conn_inpath_list:
            conn_inpath = pl.concat(conn_inpath_list, how='diagonal_relaxed')
        else:
            conn_inpath = pl.DataFrame(schema={
                'conn_layer': pl.Utf8, 'bodyId_pre': pl.Utf8, 'bodyId_post': pl.Utf8, 
                'weight': pl.Int64, 'type_pre': pl.Utf8, 'type_post': pl.Utf8, 
                'traversal_probability': pl.Float64, 'connection_ratio': pl.Float64
            })

        if conn_types_list:
            conn_types = pl.concat(conn_types_list, how='diagonal_relaxed')
        else:
            conn_types = pl.DataFrame(schema={
                'conn_layer': pl.Utf8, 'type_pre': pl.Utf8, 'type_post': pl.Utf8, 
                'weight': pl.Int64, 'traversal_probability': pl.Float64, 'connection_ratio': pl.Float64
            })

        if conn_groups_list:
            conn_groups = pl.concat(conn_groups_list, how='diagonal_relaxed')
        else:
            conn_groups = pl.DataFrame()
        
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
                if len(conn_inpath) > 0 and layer_label_in in conn_inpath['conn_layer'].unique().to_list():
                    incoming = conn_inpath.filter(pl.col('conn_layer') == layer_label_in)
                    neurons_in_layer = set(incoming['bodyId_post'].unique().to_list())
            
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
            # Only show progress bar in non-silent modes
            if self.verbose_mode != 'silent':
                progress_iter = tqdm(progress_iter, total=total_neurons_iter, desc='Updating target real layers', unit='neurons')
            for layer_idx, neuron_id in progress_iter:
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
        if not conn_inpath.is_empty():
            conn_inpath = conn_inpath.sort(['conn_layer','traversal_probability','weight'], descending=[False,True,True])
        if not conn_types.is_empty():
            conn_types = conn_types.sort(['conn_layer','traversal_probability','weight'], descending=[False,True,True])

        totalweight_df = pl.DataFrame(list(weight_layers.items()), schema={'conn_layer': pl.Utf8, 'weight': pl.Float64}, orient="row")
        if not totalweight_df.is_empty():
            totalweight_df = totalweight_df.sort('conn_layer')
        
        # Create type-level real layer map from bodyId-level real layers
        # For type-level analysis, use the earliest layer any neuron of that type appears
        # Targets already have their real layers updated based on actual path appearances
        real_layer_map_type = {}
        
        # Handle target_df (Pandas or Polars)
        if isinstance(self.target_df, pd.DataFrame):
             target_types_set = set(self.target_df.loc[self.target_df.Checked, 'type'].unique())
        else:
             target_types_set = set(self.target_df.filter(pl.col('Checked'))['type'].unique().to_list())
             
        target_type_appearances = {}  # Track appearance layers for target types
        
        if not conn_inpath.is_empty():
            # Create mapping from bodyId to type
            # Extract unique bodyId -> type from conn_inpath
            pre_map = conn_inpath.select(['bodyId_pre', 'type_pre']).rename({'bodyId_pre': 'bodyId', 'type_pre': 'type'})
            post_map = conn_inpath.select(['bodyId_post', 'type_post']).rename({'bodyId_post': 'bodyId', 'type_post': 'type'})
            body_type_map = pl.concat([pre_map, post_map]).unique()
            
            # Create DataFrame from real_layer_map_bodyId
            # Ensure keys are strings
            real_layer_df = pl.DataFrame({
                'bodyId': [str(k) for k in real_layer_map_bodyId.keys()],
                'real_layer': list(real_layer_map_bodyId.values())
            })
            
            # Join
            # Ensure bodyId in body_type_map is string (it should be from previous steps)
            type_layers = body_type_map.join(real_layer_df, on='bodyId', how='inner')
            
            # Group by type and find min layer
            min_layers = type_layers.group_by('type').agg(pl.col('real_layer').min())
            
            real_layer_map_type = dict(zip(min_layers['type'].to_list(), min_layers['real_layer'].to_list()))
            
            # Handle target type appearances
            body_to_type_dict = dict(zip(body_type_map['bodyId'].to_list(), body_type_map['type'].to_list()))
            
            for bodyId, layers in target_appearance_layers.items():
                bodyId_str = str(bodyId)
                if bodyId_str in body_to_type_dict:
                    type_val = body_to_type_dict[bodyId_str]
                    if type_val in target_types_set:
                        if type_val not in target_type_appearances:
                            target_type_appearances[type_val] = set()
                        target_type_appearances[type_val].update(layers)
        
        self._vprint(f'\nCreated type-level real layer map for {len(real_layer_map_type)} types', level='full')
        
        # Print target type appearance summary
        if target_type_appearances:
            self._vprint(f'  ✓ Updated real_layer for {len(target_type_appearances)} target types', level='full')
        
        # Create group-level real layer map if custom groups exist
        real_layer_map_group = {}
        if conn_groups is not None and not conn_groups.is_empty() and 'custom_group' in self.source_df.columns:
            if isinstance(self.target_df, pd.DataFrame):
                 target_groups_set = set(self.target_df.loc[self.target_df.Checked, 'custom_group'].unique())
            else:
                 target_groups_set = set(self.target_df.filter(pl.col('Checked'))['custom_group'].unique().to_list())
            
            target_group_appearances = {}
            
            if not conn_inpath.is_empty() and 'custom_group_pre' in conn_inpath.columns:
                # Create mapping from bodyId to group from conn_inpath
                pre_map = conn_inpath.select(['bodyId_pre', 'custom_group_pre']).rename({'bodyId_pre': 'bodyId', 'custom_group_pre': 'group'})
                post_map = conn_inpath.select(['bodyId_post', 'custom_group_post']).rename({'bodyId_post': 'bodyId', 'custom_group_post': 'group'})
                body_group_map = pl.concat([pre_map, post_map]).unique()
                
                # Join with real_layer_df
                group_layers = body_group_map.join(real_layer_df, on='bodyId', how='inner')
                
                # Group by group and find min layer
                min_layers = group_layers.group_by('group').agg(pl.col('real_layer').min())
                
                real_layer_map_group = dict(zip(min_layers['group'].to_list(), min_layers['real_layer'].to_list()))
                
                # Handle target group appearances
                body_to_group_dict = dict(zip(body_group_map['bodyId'].to_list(), body_group_map['group'].to_list()))
                
                for bodyId, layers in target_appearance_layers.items():
                    bodyId_str = str(bodyId)
                    if bodyId_str in body_to_group_dict:
                        group_val = body_to_group_dict[bodyId_str]
                        if group_val in target_groups_set:
                            if group_val not in target_group_appearances:
                                target_group_appearances[group_val] = set()
                            target_group_appearances[group_val].update(layers)
            
            print(f'\nCreated group-level real layer map for {len(real_layer_map_group)} custom groups')
            if target_group_appearances:
                print(f'  ✓ Updated real_layer for {len(target_group_appearances)} target groups')

        # Mark which source neurons are in paths to targets
        if len(conn_inpath) > 0:
            # Polars syntax
            source_inpath = conn_inpath.filter(pl.col('conn_layer') == '0->1')['bodyId_pre'].unique().to_list()
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
        if conn_inpath.is_empty():
            self._vprint('\n⚠️  No paths found - saving minimal output data', level='full')
            
            # Create data_details folder
            csv_folder = os.path.join(self.allpath_folder, 'data_details')
            os.makedirs(csv_folder, exist_ok=True)
            
            # Save parameters and source/target info even without paths
            self._save_df_to_csv_polars(self.parameter_df, os.path.join(csv_folder, 'parameters.csv'))
            self._save_df_to_csv_polars(self.source_df, os.path.join(csv_folder, 'source_neurons.csv'))
            self._save_df_to_csv_polars(self.target_df, os.path.join(csv_folder, 'target_neurons.csv'))
            
            # Save discovered type-level edges even without valid paths
            # conn_types contains type-level aggregated edges (correctly aggregated at this threshold)
            if not conn_types.is_empty():
                self._save_df_to_csv_polars(conn_types, os.path.join(csv_folder, 'connection_type.csv'))
                self._vprint(f'  ✓ Saved {len(conn_types)} type-level edges to connection_type.csv (no valid paths)', level='full')
            else:
                # Create empty connection file
                empty_conn = pl.DataFrame(schema={'type_pre': pl.Utf8, 'type_post': pl.Utf8, 'weight': pl.Int64, 
                                                  'conn_layer': pl.Utf8, 'traversal_probability': pl.Float64, 'connection_ratio': pl.Float64})
                self._save_df_to_csv_polars(empty_conn, os.path.join(csv_folder, 'connection_type.csv'))
            
            self._vprint(f'  ✓ Saved to: {csv_folder}/', level='full')
            return
        
        # Update types for source and target neurons in conn_inpath using self.source_df and self.target_df
        # This ensures that even if enrichment failed (e.g. FAFB), we at least have types for start/end of paths
        
        # Create mapping DataFrame
        source_map = pl.from_pandas(self.source_df[['bodyId', 'type']]) if isinstance(self.source_df, pd.DataFrame) else self.source_df.select(['bodyId', 'type'])
        target_map = pl.from_pandas(self.target_df[['bodyId', 'type']]) if isinstance(self.target_df, pd.DataFrame) else self.target_df.select(['bodyId', 'type'])
        
        # Ensure bodyId is string
        source_map = source_map.with_columns(pl.col('bodyId').cast(pl.Utf8))
        target_map = target_map.with_columns(pl.col('bodyId').cast(pl.Utf8))
        
        type_map_df = pl.concat([source_map, target_map]).unique()
        
        if not type_map_df.is_empty():
            self._vprint(f'  Updating types for {len(type_map_df)} source/target neurons in connection table...', level='full')
            
            # Update type_pre
            conn_inpath = conn_inpath.join(type_map_df.rename({'bodyId': 'bodyId_pre', 'type': 'type_new'}), on='bodyId_pre', how='left')
            conn_inpath = conn_inpath.with_columns(pl.col('type_new').fill_null(pl.col('type_pre')).alias('type_pre')).drop('type_new')
            
            # Update type_post
            conn_inpath = conn_inpath.join(type_map_df.rename({'bodyId': 'bodyId_post', 'type': 'type_new'}), on='bodyId_post', how='left')
            conn_inpath = conn_inpath.with_columns(pl.col('type_new').fill_null(pl.col('type_post')).alias('type_post')).drop('type_new')

        # Regenerate conn_types and conn_groups from updated conn_inpath to ensure types are correct
        # This fixes the issue where types might be missing in the initial pass but recovered via source/target mapping
        if not conn_inpath.is_empty():
            self._vprint('  Regenerating type-level connections from updated bodyId data...', level='full')
            conn_types_list_new = []
            conn_groups_list_new = []
            
            # Get unique layers
            layers = conn_inpath['conn_layer'].unique().to_list()
            
            for layer in layers:
                # Filter for this layer
                layer_conn = conn_inpath.filter(pl.col('conn_layer') == layer)
                
                # Get neurons for this layer for accurate ratio calculation
                bodyIds_in_layer = pl.concat([layer_conn['bodyId_pre'], layer_conn['bodyId_post']]).unique()
                
                neurons_in_layer_df_pd = self._fetch_neurons_local_or_api(bodyIds_in_layer.to_list(), columns=['bodyId', 'type', 'post'])
                neurons_in_layer_df = pl.from_pandas(neurons_in_layer_df_pd)
                
                # Enrich
                _, layer_conn_type, layer_conn_group = EnrichConnectionTablePolars(
                    layer_conn.drop('conn_layer'), 
                    dataset=self.dataset,
                    script_path=self.script_path,
                    target_neurons_df=neurons_in_layer_df,
                    label_mapper=self.label_mapper
                )
                
                # Add conn_layer back
                if not layer_conn_type.is_empty():
                    layer_conn_type = layer_conn_type.with_columns(pl.lit(layer).alias('conn_layer'))
                    conn_types_list_new.append(layer_conn_type)
                
                if layer_conn_group is not None and not layer_conn_group.is_empty():
                    layer_conn_group = layer_conn_group.with_columns(pl.lit(layer).alias('conn_layer'))
                    conn_groups_list_new.append(layer_conn_group)
            
            if conn_types_list_new:
                conn_types = pl.concat(conn_types_list_new)
                conn_types = conn_types.sort(['conn_layer','traversal_probability','weight'], descending=[False,True,True])
            
            if conn_groups_list_new:
                conn_groups = pl.concat(conn_groups_list_new)
            else:
                conn_groups = pl.DataFrame()

        # Generate global type-level aggregation for matrix generation (avoids duplicates from layers)
        self._vprint('  Generating global type-level matrix...', level='full')
        # Use conn_inpath (which has all edges). Deduplicate by bodyId pair to avoid double counting physical edges.
        conn_inpath_global = conn_inpath.unique(subset=['bodyId_pre', 'bodyId_post'])
        
        # Fetch all neurons involved for accurate post counts
        all_bodyIds = pl.concat([conn_inpath_global['bodyId_pre'], conn_inpath_global['bodyId_post']]).unique()
        
        # Use tqdm for fetching if large
        all_neurons_df = None
        if len(all_bodyIds) > 5000 and self.verbose_mode in ['simple', 'full']:
            # Split into chunks to show progress
            chunk_size = 5000
            all_bodyIds_list = all_bodyIds.to_list()
            chunks = [all_bodyIds_list[i:i + chunk_size] for i in range(0, len(all_bodyIds_list), chunk_size)]
            
            all_neurons_list = []
            for chunk in chunks:
                chunk_df = self._fetch_neurons_local_or_api(chunk, columns=['bodyId', 'type', 'post'])
                all_neurons_list.append(pl.from_pandas(chunk_df))
            
            if all_neurons_list:
                all_neurons_df = pl.concat(all_neurons_list)
            else:
                all_neurons_df = pl.DataFrame()
        else:
            all_neurons_df_pd = self._fetch_neurons_local_or_api(all_bodyIds.to_list(), columns=['bodyId', 'type', 'post'])
            all_neurons_df = pl.from_pandas(all_neurons_df_pd)
        
        _, conn_types_global, _ = EnrichConnectionTablePolars(
            conn_inpath_global, 
            traversal_probability_threshold=self.min_traversal_probability,
            dataset=self.dataset,
            script_path=self.script_path,
            target_neurons_df=all_neurons_df,
            aggregate_method='product',
            label_mapper=self.label_mapper
        )
        
        # print("  Enrichment returned. Proceeding to save...", flush=True)

        # Save main data (type-level aggregations)
        # Force print this message so user knows we are moving to save phase
        # print('\nSaving connection data...', flush=True)
        
        # Determine if using CSV or Excel based on output_format or data size
        EXCEL_ROW_LIMIT = 1_048_576
        use_csv = (self.output_format == 'csv') or (len(conn_types) >= EXCEL_ROW_LIMIT * 0.9)
        
        # print(f"  Format check: output_format='{self.output_format}', rows={len(conn_types):,}, use_csv={use_csv}", flush=True)
        
        if use_csv:
            if self.output_format == 'csv':
                self._vprint(f'  💾 Saving data as CSV files (output_format="csv")', level='full', flush=True)
            else:
                self._vprint(f'  ⚠️  Data too large for Excel ({len(conn_types):,} rows), saving as CSV', level='simple', flush=True)
            
            # Create data_details folder
            csv_folder = os.path.join(self.allpath_folder, 'data_details')
            os.makedirs(csv_folder, exist_ok=True)
            self._vprint(f'  💾 Saving data as CSV files to: {csv_folder}', level='simple', flush=True)
            
            # print("    - parameters.csv", flush=True)
            self._save_df_to_csv_polars(self.parameter_df, os.path.join(csv_folder, 'parameters.csv'))
            
            # print("    - source_neurons.csv", flush=True)
            # Use reset_index() to preserve index as in pandas to_csv(index=True)
            self._save_df_to_csv_polars(self.source_df, os.path.join(csv_folder, 'source_neurons.csv'), index=True)
            
            # print("    - target_neurons.csv", flush=True)
            self._save_df_to_csv_polars(self.target_df, os.path.join(csv_folder, 'target_neurons.csv'), index=True)
            
            # print("    - total_weight_layer.csv", flush=True)
            self._save_df_to_csv_polars(totalweight_df, os.path.join(csv_folder, 'total_weight_layer.csv'), index=True)
            
            # print("    - connection_type.csv", flush=True)
            self._save_df_to_csv_polars(conn_types, os.path.join(csv_folder, 'connection_type.csv'), index=True)
            
            if conn_groups is not None and not conn_groups.is_empty():
                # print("    - connection_custom_groups.csv", flush=True)
                self._save_df_to_csv_polars(conn_groups, os.path.join(csv_folder, 'connection_custom_groups.csv'), index=True)
            
            # Save matrices (use global aggregation)
            self._save_matrices_to_csv(conn_types_global, csv_folder, level='type')
        else:
            output_excel_name = os.path.join(self.allpath_folder, self.source_fname + '_to_' + self.target_fname + '_allpaths_info.xlsx')
            print(f'  💾 Saving type-level data to: {output_excel_name}', flush=True)
            print(f'  ⏳ Writing Excel file (this may take a while)...', flush=True)
            with pd.ExcelWriter(output_excel_name, mode='w', engine='xlsxwriter') as writer:
                self.parameter_df.to_excel(writer,sheet_name='parameters',index=False)
                worksheet = writer.sheets['parameters']
                worksheet.set_column('A:A', 30, writer.book.add_format({'bold': True, 'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
                worksheet.set_column('B:B', 30, writer.book.add_format({'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
                
                self.source_df.to_excel(writer,sheet_name='source_neurons')
                self.target_df.to_excel(writer,sheet_name='target_neurons')
                totalweight_df.to_excel(writer,sheet_name='total_weight_layer')
                
                if isinstance(conn_types, pl.DataFrame):
                    conn_types.to_pandas().to_excel(writer,sheet_name='connection_type')
                else:
                    conn_types.to_excel(writer,sheet_name='connection_type')
                
                # Add custom group sheet if custom grouping was used
                is_groups_empty = conn_groups.is_empty() if isinstance(conn_groups, pl.DataFrame) else conn_groups.empty
                if conn_groups is not None and not is_groups_empty:
                    if isinstance(conn_groups, pl.DataFrame):
                        conn_groups.to_pandas().to_excel(writer,sheet_name='connection_custom_groups')
                    else:
                        conn_groups.to_excel(writer,sheet_name='connection_custom_groups')
                
                # Save matrices (use global aggregation)
                self._save_matrices_to_excel(conn_types_global, writer, level='type')
        
        # Save bodyId-level data
        if not self.skip_bodyId:
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
                # print(f"    - connection_info_bodyId.csv", flush=True)
                self._save_df_to_csv_polars(conn_inpath, output_bodyid_csv)
                
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
                    if isinstance(conn_inpath, pl.DataFrame):
                        conn_inpath.to_pandas().to_excel(writer,sheet_name='connection_info_bodyId')
                    else:
                        conn_inpath.to_excel(writer,sheet_name='connection_info_bodyId')
                    self._save_matrices_to_excel(conn_inpath_global, writer, level='bodyId')
                self._vprint(f'  ✓ Saved to: {output_bodyid_excel}', level='full')
        else:
            self._vprint('Skipping bodyId-level data saving (skip_bodyId=True)', level='full')
        
        self._vprint(f'  ✓ Saved connection data', level='full')
        
        # Release memory for bodyId-level data
        # Only delete if we won't need it for path enrichment later
        if not (find_bodyId_path and not self.skip_bodyId):
            self._vprint('Releasing bodyId-level memory...', level='full')
            del conn_inpath
            del conn_inpath_global
            del edges_in_paths
            del edges_in_paths_with_layer
            del neurons_in_paths
            gc.collect()
        
        # Build path DataFrames directly from collected paths (OPTIMIZED - no re-pathfinding!)
        self._vprint('\n=== Building path DataFrames from collected paths ===', level='full')
        self._vprint(f'Found {path_count:,} paths during sequential DFS', level='full')
        self._vprint('Note: Now building type/group level summaries...', level='full')
        
        # Type-level paths - Use separate DFS on type-level graph (much faster!)
        self._vprint('\nFinding type-level paths using type-level graph...', level='full')
        
        # Build type-level graph from conn_types
        # NOTE: conn_types now uses std_label values in type_pre/type_post when label_mapper is provided
        # (thanks to EnrichConnectionTablePolars implementing the 6-step approach)
        G_type = FastGraph()
        G_type.build_from_dataframe(conn_types, 'type_pre', 'type_post', 'weight')
        
        self._vprint(f'  Type-level graph: {G_type.number_of_nodes()} types, {G_type.number_of_edges()} edges', level='full')
        
        # Build bodyId → std_label map for source/target identification
        # This is needed because conn_types uses std_labels but source_df/target_df have bodyIds
        bodyid_to_label = {}
        if self.label_mapper:
            # Use the same mapping function that EnrichConnectionTablePolars uses
            ndf_path = None
            if self.dataset and self.script_path:
                dataset_clean = self.dataset.replace(':', '_').replace('.', '_')
                ndf_path = os.path.join(
                    self.script_path, 'datasets', dataset_clean,
                    f"{dataset_clean}_allneurons_neuron_df.csv"
                )
                if not os.path.exists(ndf_path):
                    ndf_path = os.path.join(
                        self.script_path, 'datasets',
                        f"{dataset_clean}_allneurons_neuron_df.csv"
                    )
            
            if ndf_path and os.path.exists(ndf_path):
                ndf_complete = pl.read_csv(ndf_path, infer_schema_length=10000)
                if 'bodyId' in ndf_complete.columns:
                    ndf_complete = ndf_complete.with_columns(pl.col('bodyId').cast(pl.Utf8))
                bodyid_to_label = svp.build_bodyid_label_map(self.label_mapper, self.dataset, ndf_complete)
        
        # Get source and target labels (mapped or original types)
        # When label_mapper is provided, conn_types uses std_labels, so we need to match
        # For untyped neurons, use bodyId as fallback to handle data quality gracefully
        source_labels = set()
        for idx, row in self.source_df.iterrows():
            b = str(row['bodyId']) if 'bodyId' in row else ''
            t = row['type'] if 'type' in row else None
            
            # Use std_label if available, else fall back to type, else fall back to bodyId
            if b and b in bodyid_to_label:
                label = bodyid_to_label[b]
            elif t is not None and (not isinstance(t, float) or not pd.isna(t)) and str(t).strip() != '':
                label = str(t)
            elif b:
                # Use bodyId as fallback for untyped neurons
                label = b
            else:
                continue
            source_labels.add(label)
        
        source_types = list(source_labels)

        target_labels = set()
        target_rows = self.target_df.loc[self.target_df.Checked]
        for idx, row in target_rows.iterrows():
            b = str(row['bodyId']) if 'bodyId' in row else ''
            t = row['type'] if 'type' in row else None
            
            # Use std_label if available, else fall back to type, else fall back to bodyId
            if b and b in bodyid_to_label:
                label = bodyid_to_label[b]
            elif t is not None and (not isinstance(t, float) or not pd.isna(t)) and str(t).strip() != '':
                label = str(t)
            elif b:
                # Use bodyId as fallback for untyped neurons
                label = b
            else:
                continue
            target_labels.add(label)
        
        target_types = list(target_labels)
        
        # No need for type_to_label_map anymore - conn_types already uses std_labels
        # and source/target types are now properly mapped to std_labels
        
        # Find paths using DFS on type graph
        # type_paths = [] # Removed to save memory
        
        # Use optimized pathfinding for type graph as well
        # Filter sources/targets that are in the graph
        valid_source_types = [s for s in source_types if s in G_type]
        valid_target_types = [t for t in target_types if t in G_type]
        
        # Convert conn_types to Pandas if it's Polars (statvis expects Pandas)
        conn_types_pd = conn_types
        try:
            import polars as pl
            if isinstance(conn_types, pl.DataFrame):
                conn_types_pd = conn_types.to_pandas()
        except ImportError:
            pass

        # Prepare output paths for streaming
        output_path_type_csv = os.path.join(self.allpath_folder, self.source_fname+'_to_'+self.target_fname+'_allpaths_type.csv')
        details_folder = os.path.join(self.allpath_folder, 'data_details')
        os.makedirs(details_folder, exist_ok=True)
        output_path_type_excluded_csv = os.path.join(details_folder, self.source_fname+'_to_'+self.target_fname+'_allpaths_type_excluded.csv')
        
        total_type_paths = 0
        
        if valid_source_types and valid_target_types:
            # Use Meet-in-the-middle for type graph too
            path_gen = G_type.find_paths_meet_in_the_middle(
                valid_source_types, 
                valid_target_types, 
                cutoff=self.max_interlayer + 1,
                verbose=(self.verbose_mode in ['simple', 'full'])
            )
            
            # Stream directly to CSV to avoid OOM
            # NOTE: conn_types already uses std_labels (from EnrichConnectionTablePolars)
            # so no additional type_to_label_map transformation is needed
            self._vprint(f'  Streaming type-level paths to CSV (Polars)...', level='full')
            total_type_paths = svp.process_paths_streaming(
                path_gen,
                conn_types_pd,
                target_types,
                output_path_type_csv,
                excluded_path=output_path_type_excluded_csv,
                real_layer_map=real_layer_map_type if forward_only else None,
                level='type',
                keyword_in_path_to_remove=self.keyword_in_path_to_remove,
                verbose=(self.verbose_mode != 'silent')
            )
            
        self._vprint(f'  Found and saved {total_type_paths:,} type-level paths', level='full')

        # Sort the output file if paths were found
        if total_type_paths > 0 and os.path.exists(output_path_type_csv):
            self._vprint(f'  Sorting type-level paths file...', level='full')
            try:
                # Read back, sort, and save using Polars
                df_paths = pl.read_csv(output_path_type_csv)
                
                sort_cols = []
                descending = []
                
                # Check for length column
                if 'length' in df_paths.columns:
                    sort_cols.append('length')
                    descending.append(False)
                elif 'path_length' in df_paths.columns:
                    sort_cols.append('path_length')
                    descending.append(False)
                    
                # Check for probability column
                if 'path_prob' in df_paths.columns:
                    sort_cols.append('path_prob')
                    descending.append(True)
                elif 'path_probability' in df_paths.columns:
                    sort_cols.append('path_probability')
                    descending.append(True)
                
                if sort_cols:
                    df_paths = df_paths.sort(sort_cols, descending=descending)
                    df_paths.write_csv(output_path_type_csv)
                    self._vprint(f'  ✓ Sorted {os.path.basename(output_path_type_csv)}', level='full')
            except Exception as e:
                self._vprint(f'  ⚠️ Warning: Failed to sort type-level paths file: {e}', level='full')
        
        # Set path_df_type to empty as we've already saved it
        # This prevents the later code from trying to save it again or use it in memory
        path_df_type = pd.DataFrame()
        type_paths_saved_streaming = True
        
        # If paths were found, reload them for visualization (HTML generation)
        # We reload even if showfig=False because VisualizePath generates HTML files
        if total_type_paths > 0:
            try:
                nrows = self.pathN_to_show if self.pathN_to_show > 0 else None
                self._vprint(f'  Reloading top {nrows if nrows else "all"} paths for visualization...', level='full')
                path_df_type = self._read_csv(output_path_type_csv, nrows=nrows)
                
                # Convert stringified lists back to lists if needed (though visualization might handle strings)
                # But VisualizePath expects lists for 'weights', 'probabilities', 'ratios'
                import ast
                for col in ['weights', 'probabilities', 'ratios']:
                    if col in path_df_type.columns:
                        path_df_type[col] = path_df_type[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                        
            except Exception as e:
                self._vprint(f'  Warning: Failed to reload paths for visualization: {e}', level='full')
        
        # Build DataFrame from type paths (SKIPPED - already done via streaming)
        # path_df_type = sv.build_path_dataframe_from_paths(...)
        
        # Group-level paths - Use separate DFS on group-level graph (if custom groups exist)
        path_df_group = pd.DataFrame()
        path_df_group_excluded = pd.DataFrame()
        
        if conn_groups is not None and not conn_groups.is_empty() and 'custom_group' in self.source_df.columns:
            self._vprint('\nFinding group-level paths using group-level graph...', level='full')
            
            # Build group-level graph from conn_groups
            G_group = FastGraph()
            G_group.build_from_dataframe(conn_groups, 'group_pre', 'group_post', 'weight')
            
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
                    # Find all simple paths with length <= max_interlayer + 1
                    for path in G_group.all_simple_paths(source_group, target_group, cutoff=self.max_interlayer + 1):
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
            
            # Sort path_df_group
            if not path_df_group.empty:
                sort_cols = []
                ascending = []
                if 'length' in path_df_group.columns:
                    sort_cols.append('length')
                    ascending.append(True)
                elif 'path_length' in path_df_group.columns:
                    sort_cols.append('path_length')
                    ascending.append(True)
                if 'path_prob' in path_df_group.columns:
                    sort_cols.append('path_prob')
                    ascending.append(False)
                elif 'path_probability' in path_df_group.columns:
                    sort_cols.append('path_probability')
                    ascending.append(False)
                if sort_cols:
                    path_df_group = path_df_group.sort_values(by=sort_cols, ascending=ascending)
        
        # Filter out paths with any zero-weight hops
        # This happens when bodyId-level connections exist but type-level aggregation results in 0 weight
        # Note: If streaming was used (type_paths_saved_streaming=True), this filtering was already done during streaming
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
                self._save_df_to_csv_polars(path_df_group, output_path_group_csv)
                if len(path_df_group_excluded) > 0:
                    # Save excluded paths to data_details folder
                    details_folder = os.path.join(self.allpath_folder, 'data_details')
                    output_path_group_excluded_csv = os.path.join(details_folder, self.source_fname+'_to_'+self.target_fname+'_allpaths_group_excluded.csv')
                    self._save_df_to_csv_polars(path_df_group_excluded, output_path_group_excluded_csv)
                self._vprint(f'   ✓ Saved to: {self.allpath_folder}/', level='full')
            else:
                # Add to Excel file (type-level was saved to Excel, so output_excel_name exists)
                output_excel_name = os.path.join(self.allpath_folder, self.source_fname + '_to_' + self.target_fname + '_allpaths_info.xlsx')
                with pd.ExcelWriter(output_excel_name, mode='a', engine='openpyxl') as writer:
                    path_df_group.to_excel(writer,sheet_name='path_group')
                    if len(path_df_group_excluded) > 0:
                        path_df_group_excluded.to_excel(writer,sheet_name='path_group_excluded')
                self._vprint('   ✓ path_group sheets saved', level='full')
        
        # If we streamed type paths, we skip the standard saving block unless path_df_type was populated (e.g. fallback)
        if 'type_paths_saved_streaming' in locals() and type_paths_saved_streaming:
            self._vprint(f'  ✓ Type-level paths already saved via streaming', level='full')
        else:
            # Sort path_df_type before saving
            if not path_df_type.empty:
                sort_cols = []
                ascending = []
                
                if 'length' in path_df_type.columns:
                    sort_cols.append('length')
                    ascending.append(True)
                elif 'path_length' in path_df_type.columns:
                    sort_cols.append('path_length')
                    ascending.append(True)
                    
                if 'path_prob' in path_df_type.columns:
                    sort_cols.append('path_prob')
                    ascending.append(False)
                elif 'path_probability' in path_df_type.columns:
                    sort_cols.append('path_probability')
                    ascending.append(False)
                
                if sort_cols:
                    path_df_type = path_df_type.sort_values(by=sort_cols, ascending=ascending)

            self._vprint(f'💾 Saving path_type data (rows: {len(path_df_type):,})...', level='full')
            # Check if we should save as CSV (matches type-level data format OR path data too large)
            save_type_as_csv = use_csv or (len(path_df_type) >= EXCEL_ROW_LIMIT * 0.9)
            
            if save_type_as_csv:
                # Save as CSV
                if len(path_df_type) >= EXCEL_ROW_LIMIT * 0.9:
                    self._vprint(f'   ⚠️  Path data too large for Excel ({len(path_df_type):,} rows), saving as CSV', level='full')
                output_path_type_csv = os.path.join(self.allpath_folder, self.source_fname+'_to_'+self.target_fname+'_allpaths_type.csv')
                self._save_df_to_csv_polars(path_df_type, output_path_type_csv)
                if len(path_df_type_excluded) > 0:
                    # Save excluded paths to data_details folder
                    details_folder = os.path.join(self.allpath_folder, 'data_details')
                    output_path_type_excluded_csv = os.path.join(details_folder, self.source_fname+'_to_'+self.target_fname+'_allpaths_type_excluded.csv')
                    self._save_df_to_csv_polars(path_df_type_excluded, output_path_type_excluded_csv)
                self._vprint(f'   ✓ Saved to: {self.allpath_folder}/', level='full')
            else:
                # Add to Excel file (type-level was saved to Excel, so output_excel_name exists)
                output_excel_name = os.path.join(self.allpath_folder, self.source_fname + '_to_' + self.target_fname + '_allpaths_info.xlsx')
                with pd.ExcelWriter(output_excel_name, mode='a', engine='openpyxl') as writer:
                    path_df_type.to_excel(writer,sheet_name='path_type')
                    path_df_type_excluded.to_excel(writer,sheet_name='path_type_excluded')
                self._vprint('   ✓ path_type sheets saved', level='full')
        
        # BodyId-level paths
        path_df_bodyId = pd.DataFrame()
        if find_bodyId_path and not self.skip_bodyId and 'conn_inpath' in locals() and 'all_paths' in locals():
            self._vprint('\nEnriching bodyId-level paths with connection metrics...', level='full')
            
            # Create type lookup from connection data
            type_lookup = {}
            if 'type_pre' in conn_inpath.columns:
                if isinstance(conn_inpath, pl.DataFrame):
                    unique_pre = conn_inpath.select(['bodyId_pre', 'type_pre']).unique()
                    for row in unique_pre.iter_rows(named=True):
                        type_lookup[row['bodyId_pre']] = row['type_pre']
                else:
                    for _, row in conn_inpath[['bodyId_pre', 'type_pre']].drop_duplicates().iterrows():
                        type_lookup[row['bodyId_pre']] = row['type_pre']
            
            if 'type_post' in conn_inpath.columns:
                if isinstance(conn_inpath, pl.DataFrame):
                    unique_post = conn_inpath.select(['bodyId_post', 'type_post']).unique()
                    for row in unique_post.iter_rows(named=True):
                        type_lookup[row['bodyId_post']] = row['type_post']
                else:
                    for _, row in conn_inpath[['bodyId_post', 'type_post']].drop_duplicates().iterrows():
                        type_lookup[row['bodyId_post']] = row['type_post']
            
            # Also add source and target info
            for _, row in self.source_df.iterrows():
                type_lookup[row['bodyId']] = row['type']
            for _, row in self.target_df.iterrows():
                type_lookup[row['bodyId']] = row['type']

            if isinstance(conn_inpath, pl.DataFrame):
                path_df_bodyId = svp.build_path_dataframe_from_paths(
                    paths=all_paths,
                    conn_data=conn_inpath,
                    targets=self.target_df.loc[self.target_df.Checked,'bodyId'].tolist(),
                    real_layer_map=real_layer_map_bodyId if forward_only else None,
                    level='bodyId',
                    type_lookup=type_lookup
                )
                is_polars = True
            else:
                path_df_bodyId = sv.build_path_dataframe_from_paths(
                    paths=all_paths,
                    conn_data=conn_inpath,
                    targets=self.target_df.loc[self.target_df.Checked,'bodyId'].tolist(),
                    real_layer_map=real_layer_map_bodyId if forward_only else None,
                    level='bodyId',
                    type_lookup=type_lookup
                )
                is_polars = False
            
            # Sort path_df_bodyId - handle both Polars and pandas
            is_empty = path_df_bodyId.is_empty() if is_polars else path_df_bodyId.empty
            if not is_empty:
                sort_cols = []
                ascending = []
                cols = path_df_bodyId.columns
                if 'length' in cols:
                    sort_cols.append('length')
                    ascending.append(True)
                elif 'path_length' in cols:
                    sort_cols.append('path_length')
                    ascending.append(True)
                if 'path_prob' in cols:
                    sort_cols.append('path_prob')
                    ascending.append(False)
                elif 'path_probability' in cols:
                    sort_cols.append('path_probability')
                    ascending.append(False)
                if sort_cols:
                    if is_polars:
                        # Polars sorting
                        path_df_bodyId = path_df_bodyId.sort(
                            by=sort_cols, 
                            descending=[not asc for asc in ascending]
                        )
                    else:
                        path_df_bodyId = path_df_bodyId.sort_values(by=sort_cols, ascending=ascending)

            # Save path_bodyId to the bodyId data file
            self._vprint(f'💾 Saving path_bodyId data (rows: {len(path_df_bodyId):,})...', level='full')
            if use_csv:
                # Save as CSV if connection data was saved as CSV
                output_path_csv = os.path.join(self.allpath_folder,self.source_fname+'_to_'+self.target_fname+'_allpaths_bodyId_paths.csv')
                self._save_df_to_csv_polars(path_df_bodyId, output_path_csv)
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
                    self._save_df_to_csv_polars(path_df_bodyId, output_path_csv)
                    self._vprint(f'   ✓ Saved to: {output_path_csv}', level='full')
        elif self.skip_bodyId:
            self._vprint('Skipping bodyId-level path enrichment (skip_bodyId=True)', level='full')
        
        # save interlayer info to excel
        if not self.skip_bodyId:
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
                    ndf_complete = self._read_csv(dataset_path, header=0, index_col=None, dtype={'bodyId': str}, low_memory=False)
                else:
                    ndf_complete = self._read_csv(dataset_path, header=0, index_col=0, low_memory=False)
            else:
                if 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower():
                    self._vprint(f'   ⚠️  Local dataset not found for FlyWire/FAFB. Skipping interlayer info fetch.', level='full')
                    ndf_complete = pd.DataFrame()
                else:
                    self._vprint(f'   Local dataset not found, will use API calls', level='full')
                    # Ensure client is logged in for the CORRECT dataset
                    self._ensure_neuprint_client()
            
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
                    self._save_df_to_csv_polars(interlayers[i], layer_csv)
            else:
                # Save to bodyId Excel file
                with pd.ExcelWriter(output_bodyid_excel, mode='a', engine='openpyxl') as writer:
                    for i in range(len(interlayers)):
                        interlayers[i].to_excel(writer, sheet_name='layer_'+str(i+1), index=False)
            self._vprint(' ✓', level='full')
            self._vprint('   ✓ Interlayer sheets saved to bodyId file', level='full')
        else:
            self._vprint('Skipping interlayer info saving (skip_bodyId=True)', level='full')
        
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
                
                # Convert Polars to pandas if necessary for visualization
                if is_polars:
                    path_df_bodyId_pd = path_df_bodyId.to_pandas()
                else:
                    path_df_bodyId_pd = path_df_bodyId
                    
                # Filter paths if pathN_to_show is specified
                if self.pathN_to_show > 0 and len(path_df_bodyId_pd) > self.pathN_to_show:
                    paths_to_visualize_bodyId = path_df_bodyId_pd.head(self.pathN_to_show).copy()
                    if self.verbose_mode == 'full':
                        self._vprint(f'  Showing top {self.pathN_to_show} bodyId paths (by traversal_probability) out of {len(path_df_bodyId_pd)} total paths', level='full')
                else:
                    paths_to_visualize_bodyId = path_df_bodyId_pd.copy()
                    if self.verbose_mode == 'full':
                        self._vprint(f'  Showing all {len(path_df_bodyId_pd)} bodyId paths', level='full')
                
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
    
    def VisualizeSelectedPaths(
        self, 
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
