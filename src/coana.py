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

# Ignore the navis warning
logging.getLogger('navis').setLevel(logging.WARNING)

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
    
    client_hemibrain: Client | None = None
    '''neuprint client'''
    
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
    
    min_synapse_num: int = 10
    '''minimum number of synapses to be considered as connection'''
    
    min_ratio: float = 0.0
    '''
    minimum connection ratio (weight/post) to be considered as connection\n
    connection ratio is calculated as w_ij / W_j\n
    where w_ij is the number of synapses from neuron i to neuron j and W_j is the total number of post-synaptic sites of neuron j\n
    This is the direct ratio without the 0.3 scaling factor used in traversal_probability
    '''
    
    min_traversal_probability: float = 0.001
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
    
    max_interlayer: int = 2
    '''maximum number of interlayers to be considered in connection'''
    
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
    
    parameter_dict = dict()
    '''dictionary to store all specified parameters'''
    
    parameter_df = pd.DataFrame()
    '''dataframe to store all specified parameters, converted from parameter_dict'''
    
    showfig: bool = True
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
    
    pathN_to_show: int = -1
    '''
    number of strongest paths to show in network visualization\n
    -1: show all paths (default)\n
    n > 0: show only the top n paths ranked by traversal_probability (product of edge probabilities)\n
    applies to both FindPath and FindAllPath visualizations\n
    helps focus on most significant pathways in large networks\n
    Note: paths are already sorted by traversal_probability in the path_type/path_bodyId DataFrames
    '''
    
    def __post_init__(self):
        print('Initializing...')
        # Validate filter_by parameter
        if self.filter_by not in ['bodyId', 'type']:
            raise ValueError(f"filter_by must be 'bodyId' or 'type', got '{self.filter_by}'")
        # Initialize cache folder
        if self.use_cache:
            dataset_safe = self.dataset.replace(':', '_').replace('.', '_')
            self.cache_folder = os.path.join(self.script_path, 'cache', dataset_safe)
            os.makedirs(self.cache_folder, exist_ok=True)
            print(f'Cache enabled: {self.cache_folder}')
            # Ensure complete dataset with ALL neurons exists (including type=None)
            self._ensure_complete_dataset()
        if self.sourceNeurons is None or self.targetNeurons is None:
            print('\033[33mIt is not recommended to search for all neurons in the dataset.\n Using [] or list() to search for all neurons having a given type, instead.\033[0m')
        elif self.targetNeurons is None:
            self.largeTargetSet = True
    
    def _ensure_complete_dataset(self):
        '''
        Ensure complete local dataset exists (including neurons with type=None).
        This is needed for cache enrichment since cached connections may reference
        neurons without types.
        '''
        # Create datasets folder if it doesn't exist
        datasets_folder = os.path.join(self.script_path, 'datasets')
        if not os.path.exists(datasets_folder):
            os.makedirs(datasets_folder)
            print(f'Created datasets folder: {datasets_folder}')
        
        dataset_path = os.path.join(
            datasets_folder, 
            f"{self.dataset.replace(':', '_').replace('.', '_')}_allneurons"
        )
        
        neuron_csv = dataset_path + '_neuron_df.csv'
        roi_csv = dataset_path + '_roi_count_df.csv'
        
        if not os.path.exists(neuron_csv) or not os.path.exists(roi_csv):
            print(f'\n📥 Complete dataset not found, downloading ALL neurons (including type=None)...')
            print(f'   This is a one-time download for cache enrichment.')
            # Login to neuprint only if needed
            from neuprint import Client, set_default_client, default_client
            # Only login if not already done (default_client() returns None if not set)
            if self.client_hemibrain is None and default_client() is None:
                self.client_hemibrain = Client(self.server, self.dataset, self.token)
                set_default_client(self.client_hemibrain)
            try:
                # Pull complete dataset with omitNoneType=False
                sv.pull_dataset(self.dataset, save_path=dataset_path, omitNoneType=False)
                print(f'✅ Complete dataset saved to: {dataset_path}_*.csv')
            except Exception as e:
                print(f'⚠️ Warning: Failed to download complete dataset: {e}')
                print(f'   Cache enrichment may fail for neurons without types.')
    
    # ============================================================================
    # Core Database Access
    # ============================================================================
    
    def _get_connection_db_path(self):
        '''Get path to unified connection database'''
        return os.path.join(self.cache_folder, 'connections.parquet')
    
    def _get_neuron_index_path(self):
        '''Get path to neuron index (tracks cached neurons)'''
        return os.path.join(self.cache_folder, 'neuron_index.parquet')
    
    def _load_connection_db(self):
        '''
        Load unified connection database.
        Schema: bodyId_pre, bodyId_post, weight, roi (optional), cached_date
        '''
        db_path = self._get_connection_db_path()
        if os.path.exists(db_path):
            try:
                df = pd.read_parquet(db_path)
                return df
            except Exception as e:
                print(f'  ⚠️ Warning: Failed to load connection database: {e}')
                return pd.DataFrame(columns=['bodyId_pre', 'bodyId_post', 'weight', 'roi', 'cached_date'])
        return pd.DataFrame(columns=['bodyId_pre', 'bodyId_post', 'weight', 'roi', 'cached_date'])
    
    def _save_connection_db(self, conn_db):
        '''Save unified connection database with compression'''
        db_path = self._get_connection_db_path()
        try:
            conn_db.to_parquet(db_path, index=False, compression='gzip')
        except Exception as e:
            print(f'  ⚠️ Warning: Failed to save connection database: {e}')
    
    def _load_neuron_index(self):
        '''
        Load neuron index - tracks which neurons are fully cached.
        Schema: bodyId, type, instance, post, downstream_complete, last_fetched, connection_count
        '''
        index_path = self._get_neuron_index_path()
        if os.path.exists(index_path):
            try:
                return pd.read_parquet(index_path)
            except Exception as e:
                print(f'  ⚠️ Warning: Failed to load neuron index: {e}')
                return pd.DataFrame(columns=[
                    'bodyId', 'type', 'instance', 'post', 'downstream_complete', 
                    'last_fetched', 'connection_count'
                ])
        return pd.DataFrame(columns=[
            'bodyId', 'type', 'instance', 'post', 'downstream_complete',
            'last_fetched', 'connection_count'
        ])
    
    def _save_neuron_index(self, index_df):
        '''Save neuron index with compression'''
        index_path = self._get_neuron_index_path()
        try:
            index_df.to_parquet(index_path, index=False, compression='gzip')
        except Exception as e:
            print(f'  ⚠️ Warning: Failed to save neuron index: {e}')
    
    # ============================================================================
    # Query Resolution Logic
    # ============================================================================
    
    def _query_connection_db(self, upstream_bodyIds, downstream_bodyIds=None):
        '''
        Query unified connection database for specific connections.
        Returns (cached_df, uncached_upstream_ids)
        
        Parameters:
        -----------
        upstream_bodyIds : list
            List of upstream neuron bodyIds to query
        downstream_bodyIds : list or None
            List of downstream neuron bodyIds (None = all downstream)
        
        Returns:
        --------
        tuple: (cached_connections_df, list_of_uncached_upstream_ids)
        '''
        if not self.use_cache:
            return pd.DataFrame(), upstream_bodyIds
        
        conn_db = self._load_connection_db()
        neuron_index = self._load_neuron_index()
        
        if conn_db.empty:
            # No cache yet
            return pd.DataFrame(), upstream_bodyIds
        
        # Separate cached vs uncached neurons
        cached_upstream = []
        uncached_upstream = []
        
        for bodyId in upstream_bodyIds:
            if bodyId in neuron_index['bodyId'].values:
                row = neuron_index[neuron_index['bodyId'] == bodyId].iloc[0]
                
                if downstream_bodyIds is None:
                    # Need all downstream - check if fully cached
                    if row['downstream_complete']:
                        cached_upstream.append(bodyId)
                    else:
                        uncached_upstream.append(bodyId)
                else:
                    # Specific targets - for now, treat as uncached if not fully complete
                    # TODO: Could optimize by checking if specific pairs exist
                    if row['downstream_complete']:
                        cached_upstream.append(bodyId)
                    else:
                        uncached_upstream.append(bodyId)
            else:
                # Not in index = not cached
                uncached_upstream.append(bodyId)
        
        # Retrieve cached connections
        if len(cached_upstream) > 0:
            cached_conn = conn_db[conn_db['bodyId_pre'].isin(cached_upstream)].copy()
            
            # Filter by downstream if specified
            if downstream_bodyIds is not None:
                cached_conn = cached_conn[cached_conn['bodyId_post'].isin(downstream_bodyIds)].copy()
            
            return cached_conn, uncached_upstream
        
        return pd.DataFrame(), uncached_upstream
    
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
        if 'roi' in new_connections.columns:
            new_conn['roi'] = new_connections['roi']
        else:
            new_conn['roi'] = ''
        
        new_conn['cached_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Merge with existing, removing duplicates (keep existing entries)
        if not conn_db.empty:
            # Remove any new connections that already exist (based on bodyId_pre, bodyId_post, roi)
            merge_cols = ['bodyId_pre', 'bodyId_post', 'roi']
            combined = pd.concat([conn_db, new_conn])
            combined = combined.drop_duplicates(subset=merge_cols, keep='first')
        else:
            combined = new_conn
        
        # Save updated database
        self._save_connection_db(combined)
        
        new_count = len(combined) - len(conn_db)
        if new_count > 0:
            print(f'  💾 Added {new_count} new connections to database (total: {len(combined):,})')
        else:
            print(f'  📂 All connections already in database ({len(conn_db):,} total)')
        
        # Update neuron index
        self._update_neuron_index_after_fetch(new_conn, upstream_bodyIds, downstream_bodyIds)
    
    def _update_neuron_index_after_fetch(self, connections, upstream_bodyIds, downstream_bodyIds=None):
        '''
        Update neuron index after fetching connections.
        Only marks neurons as downstream_complete if we fetched ALL downstream (downstream_bodyIds=None).
        '''
        neuron_index = self._load_neuron_index()
        
        # Get neuron info from complete dataset
        dataset_path = os.path.join(
            self.script_path,
            'datasets',
            f"{self.dataset.replace(':', '_').replace('.', '_')}_allneurons_neuron_df.csv"
        )
        
        if os.path.exists(dataset_path):
            ndf_complete = pd.read_csv(dataset_path, header=0, index_col=0)
            neuron_info = ndf_complete[ndf_complete['bodyId'].isin(upstream_bodyIds)][['bodyId', 'type', 'instance', 'post']].copy()
        else:
            # Fallback: fetch from API
            try:
                ndf, _ = fetch_neurons(NeuronCriteria(bodyId=upstream_bodyIds))
                neuron_info = ndf[['bodyId', 'type', 'instance', 'post']].copy()
            except:
                neuron_info = pd.DataFrame(columns=['bodyId', 'type', 'instance', 'post'])
        
        # Count connections per neuron
        if not connections.empty:
            conn_counts = connections.groupby('bodyId_pre').size().reset_index(name='connection_count')
        else:
            conn_counts = pd.DataFrame(columns=['bodyId_pre', 'connection_count'])
        
        # Only mark as downstream_complete if we fetched ALL downstream
        mark_complete = (downstream_bodyIds is None)
        
        for bodyId in upstream_bodyIds:
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
            
            if bodyId in neuron_index['bodyId'].values:
                # Update existing entry
                if mark_complete:
                    neuron_index.loc[neuron_index['bodyId'] == bodyId, 'downstream_complete'] = True
                neuron_index.loc[neuron_index['bodyId'] == bodyId, 'last_fetched'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                neuron_index.loc[neuron_index['bodyId'] == bodyId, 'connection_count'] = conn_count
                neuron_index.loc[neuron_index['bodyId'] == bodyId, 'type'] = neuron_type
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
        
        self._save_neuron_index(neuron_index)
        
        if mark_complete:
            completed_count = len([b for b in upstream_bodyIds if b in neuron_index[neuron_index['downstream_complete'] == True]['bodyId'].values])
            print(f'  📝 Updated neuron index: {completed_count} neurons marked as complete')
    
    # ============================================================================
    # Enrichment with Type/Instance
    # ============================================================================
    
    def _enrich_connections_with_neuron_info(self, conn_df):
        '''
        Enrich connection dataframe with type and instance from complete local dataset.
        '''
        if conn_df.empty:
            return conn_df
        
        # Get unique bodyIds that need enrichment
        all_bodyids = list(set(conn_df['bodyId_pre'].tolist() + conn_df['bodyId_post'].tolist()))
        
        # Load from complete dataset (includes type=None neurons)
        dataset_path = os.path.join(
            self.script_path,
            'datasets',
            f"{self.dataset.replace(':', '_').replace('.', '_')}_allneurons_neuron_df.csv"
        )
        
        if not os.path.exists(dataset_path):
            # Fallback: try to use standard dataset (may miss type=None neurons)
            print(f'  ⚠️ Warning: Complete dataset not found, using standard dataset')
            print(f'     Some neurons without types may be missing.')
            try:
                import statvis as sv
                neuron_df, _, _, _ = sv.getNeurons(all_bodyids, dataset=self.dataset)
            except:
                neuron_df = pd.DataFrame(columns=['bodyId', 'type', 'instance'])
        else:
            # Load complete dataset from CSV
            ndf_complete = pd.read_csv(dataset_path, header=0, index_col=0)
            # Filter to only neurons we need
            neuron_df = ndf_complete[ndf_complete['bodyId'].isin(all_bodyids)].copy()
        
        neuron_info = neuron_df[['bodyId', 'type', 'instance']].copy()
        
        # Drop existing type/instance columns if they exist (to avoid _x, _y suffixes after merge)
        columns_to_drop = []
        for col in ['type_pre', 'instance_pre', 'type_post', 'instance_post']:
            if col in conn_df.columns:
                columns_to_drop.append(col)
        if columns_to_drop:
            conn_df = conn_df.drop(columns=columns_to_drop)
        
        # Join type and instance for pre-synaptic neurons
        conn_df = conn_df.merge(
            neuron_info.rename(columns={'type': 'type_pre', 'instance': 'instance_pre'}),
            left_on='bodyId_pre',
            right_on='bodyId',
            how='left'
        ).drop(columns=['bodyId'])
        
        # Join type and instance for post-synaptic neurons
        conn_df = conn_df.merge(
            neuron_info.rename(columns={'type': 'type_post', 'instance': 'instance_post'}),
            left_on='bodyId_post',
            right_on='bodyId',
            how='left'
        ).drop(columns=['bodyId'])
        
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
        dataset_path = os.path.join(
            self.script_path,
            'datasets',
            f"{self.dataset.replace(':', '_').replace('.', '_')}_allneurons_neuron_df.csv"
        )
        
        if os.path.exists(dataset_path):
            # Fast: Load from local CSV
            ndf_complete = pd.read_csv(dataset_path, header=0, index_col=0)
            neuron_df = ndf_complete[ndf_complete['bodyId'].isin(bodyIds)].copy()
            if columns:
                neuron_df = neuron_df[columns].copy()
            return neuron_df
        else:
            # Slow: API call (ensure client is logged in)
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
        dataset_path = os.path.join(
            self.script_path,
            'datasets',
            f"{self.dataset.replace(':', '_').replace('.', '_')}_allneurons_neuron_df.csv"
        )
        
        if os.path.exists(dataset_path):
            # Fast: Load from local CSV
            ndf_complete = pd.read_csv(dataset_path, header=0, index_col=0)
            neuron_df = ndf_complete[ndf_complete['type'].isin(types)].copy()
            if columns:
                neuron_df = neuron_df[columns].copy()
            return neuron_df
        else:
            # Slow: API call (ensure client is logged in)
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
        cached_conn, uncached_upstream = self._query_connection_db(upstream_bodyIds, downstream_bodyIds)
        
        if not cached_conn.empty:
            print(f'  📂 Found {len(set(upstream_bodyIds) - set(uncached_upstream))}/{len(upstream_bodyIds)} neurons in cache')
            print(f'     Retrieved {len(cached_conn):,} connections from database')
        
        # Step 2: Fetch uncached neurons from API if needed
        api_conn = pd.DataFrame()
        if len(uncached_upstream) > 0:
            print(f'  🌐 Fetching {len(uncached_upstream)} uncached neurons from API (weight ≥ 1)...')
            # Login to neuprint only if needed
            from neuprint import Client, set_default_client, default_client
            # Only login if not already done (default_client() returns None if not set)
            if self.client_hemibrain is None and default_client() is None:
                self.client_hemibrain = Client(self.server, self.dataset, self.token)
                set_default_client(self.client_hemibrain)
            if self.simple_fetch:
                from neuprint import fetch_simple_connections
                upstream_criteria = NeuronCriteria(bodyId=uncached_upstream)
                downstream_criteria = NeuronCriteria(bodyId=downstream_bodyIds) if downstream_bodyIds is not None else None
                api_conn = fetch_simple_connections(
                    upstream_criteria=upstream_criteria,
                    downstream_criteria=downstream_criteria,
                    min_weight=1,  # Always fetch with min_weight=1
                    **self.kwargs_fetch
                )
            else:
                from neuprint import fetch_adjacencies
                import statvis as sv
                neuron_df, roi_conn_df = fetch_adjacencies(
                    sources=uncached_upstream,
                    targets=downstream_bodyIds,
                    min_total_weight=1,  # Always fetch with min_weight=1
                    **self.kwargs_fetch
                )
                api_conn = sv.merge_conn_roi(neuron_df, roi_conn_df)
            # Always update database, even if empty (marks neurons as cached)
            self._update_connection_db(api_conn, uncached_upstream, downstream_bodyIds)
        
        # Step 3: Combine cached and API results
        if cached_conn.empty and api_conn.empty:
            return pd.DataFrame()
        
        # Combine results
        combined = pd.concat([cached_conn, api_conn], ignore_index=True) if not cached_conn.empty and not api_conn.empty else (cached_conn if not cached_conn.empty else api_conn)
        
        total_before_filter = len(combined)
        
        # Step 4: Apply filters based on filter_by level
        # Enrich with type and instance info (needed for both filtering modes)
        combined = self._enrich_connections_with_neuron_info(combined)
        
        # Apply filters at the specified level
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
                    print(f'     Filtered: {total_before_filter} → {len(combined)} connections (weight ≥ {min_weight})')
                print(f'     Enriched with neuron info from complete local dataset')
        
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
        
        # Drop temporary columns
        combined = combined.drop(columns=['post', 'connection_ratio', 'traversal_probability'])
        
        # Print filter summary
        filter_msg = []
        if min_weight > 1:
            filter_msg.append(f'weight ≥ {min_weight}')
        if min_conn_ratio > 0:
            filter_msg.append(f'ratio ≥ {min_conn_ratio}')
        if min_traversal_prob > 0:
            filter_msg.append(f'prob ≥ {min_traversal_prob}')
        
        print(f'     Filtered (bodyId level): {total_before_filter} → {len(combined)} connections ({", ".join(filter_msg)})')
        print(f'     Enriched with neuron info from complete local dataset')
        
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
        print(f'     Filtered (type level): {type_grouped_before_weight} → {type_pairs_after} type pairs, {total_before_filter} → {len(combined)} connections ({", ".join(filter_msg)})')
        if null_conn_count > 0:
            print(f'     Note: {null_conn_count} connections with null types preserved (not filtered)')
        print(f'     Note: All 3 filters applied at type level (weight=sum, ratio=sum(weight)/sum(post))')
        print(f'     Enriched with neuron info from complete local dataset')
        
        return combined
    

    def InitializeNeuronInfo(self):
        # Ensure neuprint Client is set before any statvis/neuprint API call
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
        self.source_df, _, source_fname_auto, self.source_criteria = sv.getNeurons(self.sourceNeurons, dataset=self.dataset)
        self.target_df, _, target_fname_auto, self.target_criteria = sv.getNeurons(self.targetNeurons, dataset=self.dataset)
        
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
        
        if not self.save_folder: # if save_folder is not specified, save in data_folder, with auto-generated name
            # Create base folder with just source_to_target (no parameters)
            self.save_folder = os.path.join(self.data_folder, self.source_fname + '_to_' + self.target_fname)
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
    
    def PrintROIHierarchy(self):
        '''print the ROI hierarchy, with primary ROIs marked with *'''
        # Show the ROI hierarchy, with primary ROIs marked with '*'
        print('*: Primary ROI')
        print(fetch_roi_hierarchy(False, mark_primary=True, format='text'))
            
    def FindDirectConnections(self, full_data=False, heatmap_scale='linear', filter_zeros_in_heatmap=True):
        '''
        find direct connections between source and target neurons
        
        Parameters
        ----------
        full_data : bool, optional
            Whether to save the full connection table. If False, only visualize in heatmap.
            If True, run clustering and save all matrices. (default: False)
        heatmap_scale : str, optional
            Scale for heatmap color mapping: 'linear', 'log2', or 'log10' (default: 'linear')
            Applies to bodyId-level heatmaps. Log scale is useful for large dynamic ranges.
        filter_zeros_in_heatmap : bool, optional
            If True, remove empty rows/columns from heatmaps (neurons with no connections).
            If False, keep all neurons in heatmaps. (default: True)
        '''
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
        
        # Optimization: if source and target sets are the same, fetch all downstream once
        # This allows neurons to be marked as downstream_complete and cached properly
        if set(source_bodyIds) == set(target_bodyIds):
            print('  (Source and target are the same - fetching all downstream for better caching)')
            self.conn_df = self._fetch_connections_with_cache(
                upstream_bodyIds=source_bodyIds,
                downstream_bodyIds=None,  # Fetch ALL downstream
                min_weight=self.min_synapse_num,
                min_conn_ratio=self.min_ratio,
                min_traversal_prob=self.min_traversal_probability
            )
            # Filter to only keep connections within the target set
            self.conn_df = self.conn_df[self.conn_df['bodyId_post'].isin(target_bodyIds)].copy()
        else:
            # Different source/target sets - fetch specific targets
            self.conn_df = self._fetch_connections_with_cache(
                upstream_bodyIds=source_bodyIds,
                downstream_bodyIds=target_bodyIds,
                min_weight=self.min_synapse_num,
                min_conn_ratio=self.min_ratio,
                min_traversal_prob=self.min_traversal_probability
            )
        if self.conn_df.empty:
            print('\033[33mNo direct connections found.\033[0m\n')
            return
        
        # enrich connection information (recalculate metrics for display)
        # Type-level prob = 1 - product(bodyId-level block_prob)
        # Don't pass target_neurons_df - let EnrichConnectionTable use neurons from connections
        # This uses sum(post) of neurons that actually received connections as denominator
        self.conn_df, self.conn_type = sv.EnrichConnectionTable(
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
        # 
        self.source_in_conn: pd.DataFrame = self.source_df[self.source_df['bodyId'].isin(self.conn_df['bodyId_pre'].unique())]
        self.source_in_conn = self.source_in_conn.reset_index(drop=True)
        self.target_in_conn: pd.DataFrame = self.target_df[self.target_df['bodyId'].isin(self.conn_df['bodyId_post'].unique())]
        self.target_in_conn = self.target_in_conn.reset_index(drop=True)
        print(f'{len(self.source_in_conn)} / {len(self.source_df)} source neurons involved in connections')
        print(f'{len(self.target_in_conn)} / {len(self.target_df)} target neurons involved in connections')
        with open(self.parameter_txt, 'a') as f:
            f.write(f'{len(self.source_in_conn)} / {len(self.source_df)} source {self.source_fname} neurons involved in connections\n')
            f.write(f'{len(self.target_in_conn)} / {len(self.target_df)} target {self.target_fname} neurons involved in connections\n')
            f.write('\n')
        
        output_excel_name = os.path.join(self.direct_folder,self.source_fname+'_to_'+self.target_fname+'_info_snp'+str(self.min_synapse_num)+'.xlsx')
        print(f'Saving connection info to excel file...')
        with pd.ExcelWriter(output_excel_name, mode='w', engine='xlsxwriter') as dataWriter:
            self.parameter_df.to_excel(dataWriter,sheet_name='parameters')
            worksheet = dataWriter.sheets['parameters']
            worksheet.set_column('A:A', 30, dataWriter.book.add_format({'bold': True, 'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
            worksheet.set_column('B:B', 30, dataWriter.book.add_format({'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
            
            self.source_df.to_excel(dataWriter,sheet_name='source_info')
            self.target_df.to_excel(dataWriter,sheet_name='target_info')
            self.source_in_conn.to_excel(dataWriter,sheet_name='source_in_connection')
            self.target_in_conn.to_excel(dataWriter,sheet_name='target_in_connection')
            self.conn_df.to_excel(dataWriter,sheet_name='connection_info')
            self.conn_type.to_excel(dataWriter,sheet_name='connection_groupby_type')
            if not self.largeTargetSet:
                self.conn_matrix_bodyId.to_excel(dataWriter,sheet_name='connectionMatrix_bodyId')
                self.conn_matrix_type.to_excel(dataWriter,sheet_name='connectionMatrix_type')
                self.cmat_full_bodyId.to_excel(dataWriter,sheet_name='connMat_bodyId_full')
                self.cmat_full_type.to_excel(dataWriter,sheet_name='connMat_type_full')
                self.transitionMat_bodyId.to_excel(dataWriter,sheet_name='transmissionMat_bodyId')
                self.transitionMat_type.to_excel(dataWriter,sheet_name='transmissionMat_type')
                self.conn_matrix_ratio_bodyId.to_excel(dataWriter,sheet_name='connectionRatioMat_bodyId')
                self.conn_matrix_ratio_type.to_excel(dataWriter,sheet_name='connectionRatioMat_type')
                self.ratioMat_full_bodyId.to_excel(dataWriter,sheet_name='ratioMat_bodyId_full')
                self.ratioMat_full_type.to_excel(dataWriter,sheet_name='ratioMat_type_full')
            else:
                self.conn_matrix_bodyId.transpose().to_excel(dataWriter,sheet_name='connectionMatrix_bodyId')
                self.conn_matrix_bodyId.transpose().to_excel(dataWriter,sheet_name='connectionMatrix_bodyId')
                self.conn_matrix_type.transpose().to_excel(dataWriter,sheet_name='connectionMatrix_type')
                self.cmat_full_bodyId.transpose().to_excel(dataWriter,sheet_name='connMat_bodyId_full')
                self.cmat_full_type.transpose().to_excel(dataWriter,sheet_name='connMat_type_full')
                self.transitionMat_bodyId.transpose().to_excel(dataWriter,sheet_name='transmissionMat_bodyId')
                self.transitionMat_type.transpose().to_excel(dataWriter,sheet_name='transmissionMat_type')
                self.conn_matrix_ratio_bodyId.transpose().to_excel(dataWriter,sheet_name='connectionRatioMat_bodyId')
                self.conn_matrix_ratio_type.transpose().to_excel(dataWriter,sheet_name='connectionRatioMat_type')
                self.ratioMat_full_bodyId.transpose().to_excel(dataWriter,sheet_name='ratioMat_bodyId_full')
                self.ratioMat_full_type.transpose().to_excel(dataWriter,sheet_name='ratioMat_type_full')
        print('Done\n')
        self.VisualizeDirectConnections_simple(heatmap_scale=heatmap_scale, filter_zeros=filter_zeros_in_heatmap)
        if full_data:
            self.VisualizeDirectConnections_complex()
        return 0
        
    def VisualizeDirectConnections_simple(self, heatmap_scale='linear', filter_zeros=True):
        # Visualize connection matrix in heatmap using CreateHeatmap class
        print('Visualizing connection matrix in heatmap...')
        
        # Optionally filter out empty rows/columns
        if filter_zeros:
            # Filter matrices to remove empty rows/columns
            cmat_bodyId = self.cmat_full_bodyId.loc[
                self.cmat_full_bodyId.sum(axis=1) > 0,
                self.cmat_full_bodyId.sum(axis=0) > 0
            ]
            cmat_type = self.cmat_full_type.loc[
                self.cmat_full_type.sum(axis=1) > 0,
                self.cmat_full_type.sum(axis=0) > 0
            ]
            transitionMat_bodyId = self.transitionMat_bodyId.loc[
                self.transitionMat_bodyId.sum(axis=1) > 0,
                self.transitionMat_bodyId.sum(axis=0) > 0
            ]
            transitionMat_type = self.transitionMat_type.loc[
                self.transitionMat_type.sum(axis=1) > 0,
                self.transitionMat_type.sum(axis=0) > 0
            ]
            ratioMat_bodyId = self.ratioMat_full_bodyId.loc[
                self.ratioMat_full_bodyId.sum(axis=1) > 0,
                self.ratioMat_full_bodyId.sum(axis=0) > 0
            ]
            ratioMat_type = self.ratioMat_full_type.loc[
                self.ratioMat_full_type.sum(axis=1) > 0,
                self.ratioMat_full_type.sum(axis=0) > 0
            ]
            print(f'  Filtered matrices: bodyId ({self.cmat_full_bodyId.shape} → {cmat_bodyId.shape}), type ({self.cmat_full_type.shape} → {cmat_type.shape})')
        else:
            # Use full matrices
            cmat_bodyId = self.cmat_full_bodyId
            cmat_type = self.cmat_full_type
            transitionMat_bodyId = self.transitionMat_bodyId
            transitionMat_type = self.transitionMat_type
            ratioMat_bodyId = self.ratioMat_full_bodyId
            ratioMat_type = self.ratioMat_full_type
        
        # Create heatmap generator instance
        heatmap_gen = sv.CreateHeatmap(
            output_folder=self.direct_folder,
            showfig=self.showfig
        )
        
        # Add connection matrix heatmaps (use filtered or full matrices based on parameter)
        # Use interactive mode for bodyId heatmaps (allows user to switch scales)
        heatmap_gen.add_heatmap(
            matrix=cmat_bodyId,
            name=f'heatmap_connMatrix_bodyId_snp{self.min_synapse_num}',
            title=f'heatmap of connection matrix: {self.source_fname} to {self.target_fname}<br>based on bodyId',
            color_scale='green',
            interactive=True,  # Enable interactive controls
            conn_df=self.conn_df  # Pass connection data for enhanced hover info
        )
        
        # Type heatmaps - enable interactive UI for user control
        heatmap_gen.add_heatmap(
            matrix=cmat_type,
            name=f'heatmap_connMatrix_type_snp{self.min_synapse_num}',
            title=f'heatmap of connection matrix: {self.source_fname} to {self.target_fname}<br>based on type',
            color_scale='purple',
            interactive=True  # Enable interactive controls
        )
        
        # Add transmission matrix heatmaps
        heatmap_gen.add_heatmap(
            matrix=transitionMat_bodyId,
            name=f'heatmap_transmissionMat_bodyId_snp{self.min_synapse_num}',
            title=f'heatmap of full transmission matrix: {self.source_fname} to {self.target_fname}<br>based on bodyId',
            color_scale='green',
            interactive=True,  # Enable interactive controls
            conn_df=self.conn_df  # Pass connection data for enhanced hover info
        )
        
        heatmap_gen.add_heatmap(
            matrix=transitionMat_type,
            name=f'heatmap_transmissionMat_type_snp{self.min_synapse_num}',
            title=f'heatmap of full transmission matrix: {self.source_fname} to {self.target_fname}<br>based on type',
            color_scale='purple',
            interactive=True  # Enable interactive controls
        )
        
        # Add ratio matrix heatmaps (use filtered or full matrices)
        heatmap_gen.add_heatmap(
            matrix=ratioMat_bodyId,
            name=f'heatmap_ratioMat_bodyId_snp{self.min_synapse_num}',
            title=f'heatmap of connection ratio matrix: {self.source_fname} to {self.target_fname}<br>based on bodyId',
            color_scale='orange',
            interactive=True,  # Enable interactive controls
            conn_df=self.conn_df  # Pass connection data for enhanced hover info
        )
        
        heatmap_gen.add_heatmap(
            matrix=ratioMat_type,
            name=f'heatmap_ratioMat_type_snp{self.min_synapse_num}',
            title=f'heatmap of connection ratio matrix: {self.source_fname} to {self.target_fname}<br>based on type',
            color_scale='orange',
            interactive=True  # Enable interactive controls
        )
        
        # Generate all heatmaps
        heatmap_gen.create_all()
        # Visualize by sankey diagram and network graph, only for neuron type
        print('Visualizing by Sankey diagram and network graph...')
        sankey_name = 'sankey_type_snp'+str(self.min_synapse_num)+'.html'
        sv.SankeyDirect(self.conn_matrix_type,file_path=os.path.join(self.direct_folder,sankey_name),showfig=self.showfig,node_color=self.node_color,link_color=self.link_color)
        # Create ratio-based Sankey diagram
        sankey_ratio_name = 'sankey_type_ratio_snp'+str(self.min_synapse_num)+'.html'
        sv.SankeyDirect(self.conn_matrix_ratio_type,file_path=os.path.join(self.direct_folder,sankey_ratio_name),showfig=self.showfig,node_color=self.node_color,link_color=self.link_color)
        print('Done\n')
        
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
                    showfig=self.showfig
                )
                vp.visualize()
                print('  ✓ Created complete VisualizePath visualization:')
                print('    - Interactive heatmap (type-level connections)')
                print('    - Sankey diagram (flow visualization)')
                print('    - Network graph (interactive topology)')
                
            else:
                print('  No connections to visualize')
        except Exception as e:
            import traceback
            print(f'  Warning: VisualizePath visualization failed: {e}')
            print(traceback.format_exc())
        print('Done\n')
    
    
    def VisualizeDirectConnections_complex(self):
        '''plot connection distribution, clustering, normalized cluster, 2-D sorting by maximums'''
        
        # Visualize connection distribution
        print('plotting connection distribution...')
        save_path = os.path.join(self.direct_folder,'connection distribution')
        if not os.path.exists(save_path): os.makedirs(save_path)
        sv.VisConnDist(self.conn_matrix_type,save_path,suffix='type',showfig=self.showfig)
        sv.VisConnDist(self.conn_matrix_bodyId,save_path,suffix='bodyId',showfig=self.showfig)
        print('Done')
        
        ## clustering
        save_path = os.path.join(self.direct_folder,'clustering')
        save_format = '.svg'
        if not os.path.exists(save_path): os.makedirs(save_path)
        # clustering by type
        print('clustering by type...')
        _,matt_n = sv.ClusterMap(self.conn_matrix_type,cmap='Blues',filename=os.path.join(save_path,'cluster_type_snp'+str(self.min_synapse_num)+save_format),showfig=self.showfig)
        _,matt_col = sv.ClusterMap(self.conn_matrix_type,zs=0,filename=os.path.join(save_path,'cluster_type_normCol_snp'+str(self.min_synapse_num)+save_format),showfig=self.showfig) # normalize vertically
        _,matt_row = sv.ClusterMap(self.conn_matrix_type,zs=1,filename=os.path.join(save_path,'cluster_type_normRow_snp'+str(self.min_synapse_num)+save_format),showfig=self.showfig) # normalize horizontally
        print('Done')
        # # clustering by bodyId
        # print('clustering by bodyId...')
        # _,matb_n = sv.ClusterMap(self.conn_matrix_bodyId,cmap='Blues',scale_ratio=9,filename=os.path.join(save_path,'cluster_bodyId_snp'+str(self.min_synapse_num)+save_format),showfig=self.showfig)
        # _,matb_col = sv.ClusterMap(self.conn_matrix_bodyId,zs=0,scale_ratio=9,filename=os.path.join(save_path,'cluster_bodyId_normCol_snp'+str(self.min_synapse_num)+save_format),showfig=self.showfig)
        # _,matb_row = sv.ClusterMap(self.conn_matrix_bodyId,zs=1,scale_ratio=9,filename=os.path.join(save_path,'cluster_bodyId_normRow_snp'+str(self.min_synapse_num)+save_format),showfig=self.showfig)
        # print('Done')
        
        # save clustered matrix
        print('saving clustered matrix...')
        with pd.ExcelWriter(os.path.join(save_path,'clustered_mat.xlsx')) as dataWriter:
            matt_n.to_excel(dataWriter,sheet_name='cluster_type')
            matt_col.to_excel(dataWriter,sheet_name='cluster_type_normCol')
            # matt_row.to_excel(dataWriter,sheet_name='cluster_type_normRow')
            # matb_n.to_excel(dataWriter,sheet_name='cluster_bodyId')
            # matb_col.to_excel(dataWriter,sheet_name='cluster_bodyId_normCol')
            # matb_row.to_excel(dataWriter,sheet_name='cluster_bodyId_normRow')
        print('Done\n')
        
        ## 2-D sorting by maximums
        print('2-D sorting by maximums...')
        save_path = os.path.join(self.direct_folder,'Expansion or Convergence')
        if not os.path.exists(save_path): os.makedirs(save_path)
        
        sourceMR_ranges = [[0.7,1],[0,0.7]]
        sourceN_ranges = [[1,1],[2,np.Inf]]
        targetMR_ranges = [[0.7,1],[0,0.7]]
        targetN_ranges = [[1,1],[2,np.Inf]]
        # for source neurons
        print('sorting source neurons...')
        for rr in sourceMR_ranges:
            sv.sortMatByMax(self.conn_matrix_type,save_path,title='source max ratio range (type): '+str(rr),suffix='type',by='sourceMR',filt_range=rr,clusterFlag=False,showfig=False)
            sv.sortMatByMax(self.conn_matrix_bodyId,save_path,title='source max ratio range (bodyId): '+str(rr),suffix='bodyId',by='sourceMR',filt_range=rr)
        for rr in sourceN_ranges:
            sv.sortMatByMax(self.conn_matrix_type,save_path,title='source neuron number range (type): '+str(rr),suffix='type',by='sourceN',filt_range=rr)
            sv.sortMatByMax(self.conn_matrix_bodyId,save_path,title='source neuron number range(bodyId): '+str(rr),suffix='bodyId',by='sourceN',filt_range=rr)
        print('Done')
        # # for target neurons
        print('sorting target neurons...')
        for rr in targetMR_ranges:
            sv.sortMatByMax(self.conn_matrix_type,save_path,title='target max ratio range (type): '+str(rr),suffix='type',by='targetMR',filt_range=rr)
            sv.sortMatByMax(self.conn_matrix_bodyId,save_path,title='target max ratio range(bodyId): '+str(rr),suffix='bodyId',by='targetMR',filt_range=rr)
        for rr in targetN_ranges:
            sv.sortMatByMax(self.conn_matrix_type,save_path,title='target neuron number range (type): '+str(rr),suffix='type',by='targetN',filt_range=rr)
            sv.sortMatByMax(self.conn_matrix_bodyId,save_path,title='target neuron number range(bodyId): '+str(rr),suffix='bodyId',by='targetN',filt_range=rr)
        print('Done\n')
    
    def FindPath(self, find_bodyId_path=True):
        '''Find path between source and target neurons, adapted from FindInterClusterConnection.ipynb'''
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
        self.path_folder = os.path.join(base_folder, f'paths_{param_suffix}')
        if not os.path.exists(self.path_folder):
            os.makedirs(self.path_folder)
        targetNum = len(self.target_df)
        self.target_df.insert(loc=0,column='Checked',value=False)
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
            conn_df = sv.removeSearchedNeurons(conn_df,searchedNeurons)
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
            
            conn_df, conn_type = sv.EnrichConnectionTable(
                conn_df, 
                traversal_probability_threshold=0,
                dataset=self.dataset,
                script_path=self.script_path,
                target_neurons_df=neurons_in_layer_df
            )
            conn_df.insert(loc=0,column='conn_layer',value=str(i)+'->'+str(i+1))
            conn_type.insert(loc=0,column='conn_layer',value=str(i)+'->'+str(i+1))
            conn_inpath = pd.concat([conn_inpath,conn_df])
            conn_types = pd.concat([conn_types,conn_type])
            
            post_ID = conn_df['bodyId_pre'].unique()
            neuron_layers.append(post_ID)
            post_ID = np.concatenate((post_ID,target_ID)) # post ID for next cycle. include target_ID because all target neurons may not be at the last layer
            post_ID = np.unique(post_ID)
            weight_layers.update({str(i)+'->'+str(i+1): conn_df['weight'].sum()})
            
        neuron_layers.reverse()
        conn_inpath = conn_inpath.sort_values(by=['conn_layer','traversal_probability','weight'],ascending=[True,False,False])
        conn_inpath = conn_inpath.reset_index(drop=True)
        conn_types = conn_types.sort_values(by=['conn_layer','traversal_probability','weight'],ascending=[True,False,False])
        conn_types = conn_types.reset_index(drop=True)

        totalweight_df = pd.DataFrame(weight_layers.items(),columns=['conn_layer','weight'])
        totalweight_df = totalweight_df.sort_values(by='conn_layer',ascending=True)

        source_inpath = conn_inpath.loc[conn_inpath.conn_layer=='0->1','bodyId_pre'].unique()
        self.source_df.insert(loc=0,column='isInPath',value=False)
        self.source_df.loc[self.source_df.bodyId.isin(source_inpath),'isInPath'] = True
        
        # saving data
        output_excel_name = os.path.join(self.path_folder,self.source_fname+'_to_'+self.target_fname+'_path_info.xlsx')
        with pd.ExcelWriter(output_excel_name,mode='w',engine='xlsxwriter') as writer:
            self.parameter_df.to_excel(writer,sheet_name='parameters',index=False)
            worksheet = writer.sheets['parameters']
            worksheet.set_column('A:A', 30, writer.book.add_format({'bold': True, 'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
            worksheet.set_column('B:B', 30, writer.book.add_format({'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
            
            self.source_df.to_excel(writer,sheet_name='source_neurons')
            self.target_df.to_excel(writer,sheet_name='target_neurons')
            totalweight_df.to_excel(writer,sheet_name='total_weight_layer')
            conn_inpath.to_excel(writer,sheet_name='connection_info')
            conn_types.to_excel(writer,sheet_name='connection_type')
        
        # get connection path (by type)
        path_df_type = pd.DataFrame()
        print('Analyzing path info by type:')
        # Note: FindPath uses layer-by-layer discovery which already ensures forward-only paths
        # No need for real_layer_map validation here
        path_df_type,_ = sv.getAllPath(conn_data = conn_types,
                                    targets = self.target_df.loc[self.target_df.Checked,'type'].unique().tolist(),
                                    traversal_probability_threshold = self.min_traversal_probability,
                                    max_path_length = self.max_interlayer + 1,
                                    real_layer_map = None)
        
        # Filter out paths with any zero-weight hops
        # This happens when bodyId-level connections exist but type-level aggregation results in 0 weight
        if len(path_df_type) > 0:
            before_filter = len(path_df_type)
            path_df_type = path_df_type[
                path_df_type['weights'].apply(lambda w_list: all(w > 0 for w in w_list))
            ]
            after_filter = len(path_df_type)
            if before_filter > after_filter:
                print(f'  Removed {before_filter - after_filter} paths with zero-weight hops at type level')
        
        path_df_type = sv.split_path(path_df_type)
        path_df_type, path_df_type_excluded = sv.path_filter(path_df_type,self.keyword_in_path_to_remove)
        
        # Save configuration files to path folder
        print('\nSaving configuration files...')
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
        
        print('💾 Saving path_type data to Excel...')
        with pd.ExcelWriter(output_excel_name, mode='a', engine='openpyxl') as writer:
            path_df_type.to_excel(writer,sheet_name='path_type')
            path_df_type_excluded.to_excel(writer,sheet_name='path_type_excluded')
        print('   ✓ path_type sheets saved')
        
        # get connection path (by bodyId)
        if find_bodyId_path:
            path_df_bodyId = pd.DataFrame()
            print('Analyzing path info by bodyId:')
            # Note: FindPath uses layer-by-layer discovery which already ensures forward-only paths
            path_df_bodyId,_ = sv.getAllPath(conn_data = conn_inpath,
                                        targets = self.target_df.loc[self.target_df.Checked,'bodyId'].tolist(),
                                        traversal_probability_threshold = self.min_traversal_probability,
                                        max_path_length = self.max_interlayer + 1,
                                        real_layer_map = None)
            if len(path_df_bodyId) > 1048575:
                path_df_bodyId = path_df_bodyId.iloc[:1048575,:]
                print('\033[33mWarning: Excel has a limit of 1048576 rows, only the first 1048575 rows are saved.\033[0m')
            print('💾 Saving path_bodyId data to Excel...')
            with pd.ExcelWriter(output_excel_name, mode='a', engine='openpyxl') as writer:
                path_df_bodyId.to_excel(writer,sheet_name='path_bodyId')
            print('   ✓ path_bodyId sheet saved')
        
        # save interlayer info to excel
        print('💾 Saving interlayer neuron info to Excel...')
        
        # Try to load complete neuron dataset for faster lookup
        dataset_path = os.path.join(
            self.script_path,
            'datasets',
            f"{self.dataset.replace(':', '_').replace('.', '_')}_allneurons_neuron_df.csv"
        )
        use_local_dataset = os.path.exists(dataset_path)
        if use_local_dataset:
            print(f'   Using local dataset: {os.path.basename(dataset_path)}')
            ndf_complete = pd.read_csv(dataset_path, header=0, index_col=0)
        else:
            print(f'   Local dataset not found, will use API calls')
            # Ensure client is logged in before API calls
            if self.client_hemibrain is None:
                from neuprint import Client, set_default_client
                self.client_hemibrain = Client(self.server, self.dataset, self.token)
                set_default_client(self.client_hemibrain)
        
        interlayers = []
        num_layers = len(neuron_layers[1:])
        for layer_idx, neurons in enumerate(neuron_layers[1:], 1):
            print(f'   Fetching layer {layer_idx}/{num_layers} info ({len(neurons)} neurons)...', end='', flush=True)
            
            if use_local_dataset:
                # Fast: lookup from local CSV
                n_df = ndf_complete[ndf_complete['bodyId'].isin(neurons)].copy()
            else:
                # Slow: API call to neuprint (client already logged in above)
                n_df,_ = fetch_neurons(NeuronCriteria(bodyId=neurons))
            
            interlayers.append(n_df)
            print(' ✓')
        
        print('   Writing to Excel...', end='', flush=True)
        with pd.ExcelWriter(output_excel_name, mode='a', engine='openpyxl') as writer:
            for i in range(len(interlayers)):
                interlayers[i].to_excel(writer,sheet_name='layer_'+str(i+1))
        print(' ✓')
        print('   ✓ Interlayer sheets saved')
        print('Done\n')
        
        # Build Sankey diagrams from path data (not from conn_types)
        # This ensures only paths TO TARGETS are shown (no non-target terminals)
        print('Creating Sankey diagrams from path data...')
        
        def parse_path_to_edges(path_block):
            """Parse 'A -> B -> C' into list of (layer_idx, source, target)"""
            nodes = [n.strip() for n in path_block.split('->')]
            return [(i, nodes[i], nodes[i+1]) for i in range(len(nodes) - 1)]

        def weighted_average(values, weights):
            weights = np.asarray(weights)
            total = weights.sum()
            if total <= 0:
                return 0.0
            values = np.asarray(values)
            return float(np.dot(values, weights) / total)

        # Pre-compute metrics from connection info to ensure accurate weights
        type_edge_metrics = {}
        body_edge_metrics = {}

        if len(conn_types) > 0:
            # Aggregate conn_types by (conn_layer, type_pre, type_post) to handle duplicate edges
            # Sum weights, compute weighted average for ratios, and use product for traversal probabilities
            grouped = conn_types.groupby(['conn_layer', 'type_pre', 'type_post'])
            for (layer_label, type_pre, type_post), df_edge in grouped:
                layer_idx = int(layer_label.split('->')[0])
                weight_sum = float(df_edge['weight'].sum())
                
                # For connection_ratio: weighted average by weight
                ratio_avg = weighted_average(df_edge['connection_ratio'], df_edge['weight'])
                
                # For traversal_probability: product of block probabilities
                if 'block_probability' in df_edge.columns:
                    block_prod = df_edge['block_probability'].prod()
                    prob_agg = 1 - block_prod
                else:
                    prob_avg = weighted_average(df_edge['traversal_probability'], df_edge['weight'])
                    prob_agg = prob_avg
                
                type_edge_metrics[(layer_idx, type_pre, type_post)] = {
                    'weight': weight_sum,
                    'ratio': max(0.0, min(1.0, ratio_avg)),  # Clamp to [0, 1]
                    'prob': max(0.0, min(1.0, prob_agg))  # Clamp to [0, 1]
                }

        if len(conn_inpath) > 0:
            body_group = conn_inpath.groupby(['conn_layer', 'bodyId_pre', 'bodyId_post'])
            for (layer_label, body_pre, body_post), df_edge in body_group:
                layer_idx = int(layer_label.split('->')[0])
                weight_sum = float(df_edge['weight'].sum())
                ratio_avg = weighted_average(df_edge['connection_ratio'], df_edge['weight'])
                prob_avg = weighted_average(df_edge['traversal_probability'], df_edge['weight'])
                body_edge_metrics[(layer_idx, int(body_pre), int(body_post))] = {
                    'weight': weight_sum,
                    'ratio': max(0.0, min(1.0, ratio_avg)),  # Clamp to [0, 1]
                    'prob': max(0.0, min(1.0, prob_avg))  # Clamp to [0, 1]
                }

        # Build Sankey diagram from connection_type sheet (conn_types)
        # Show ALL connections in the network, not just those in specific paths
        if len(conn_types) > 0:
            # Extract all edges directly from conn_types DataFrame
            edge_agg = {}
            for idx, row in conn_types.iterrows():
                layer_label = row['conn_layer']
                layer_idx = int(layer_label.split('->')[0])
                source = row['type_pre']
                target = row['type_post']
                edge_key = (layer_idx, source, target)
                
                # Read values directly without modification for debugging
                weight_val = float(row['weight'])
                ratio_val = float(row['connection_ratio'])
                prob_val = float(row['traversal_probability'])
                
                # Check for unexpected values
                if ratio_val > 1.0 or ratio_val < 0.0:
                    print(f'\033[33mWarning: connection_ratio out of range [0,1]: {ratio_val} for {source}->{target}\033[0m')
                if prob_val > 1.0 or prob_val < 0.0:
                    print(f'\033[33mWarning: traversal_probability out of range [0,1]: {prob_val} for {source}->{target}\033[0m')
                
                edge_agg[edge_key] = {
                    'weight': weight_val,
                    'ratio': ratio_val,  # Use raw value without clamping
                    'prob': prob_val     # Use raw value without clamping
                }
            
            if len(edge_agg) == 0:
                print('\033[33mWarning: No connections found in connection_type sheet for Sankey diagrams.\033[0m')
            else:
                
                # Build node list and track all layers each type appears in
                all_types_by_layer = {}
                type_all_layers = {}  # Track all layers for each neuron type
                for (layer_idx, source, target) in edge_agg.keys():
                    all_types_by_layer.setdefault(layer_idx, set()).add(source)
                    all_types_by_layer.setdefault(layer_idx + 1, set()).add(target)
                    
                    # Track all layers for each type
                    type_all_layers.setdefault(source, set()).add(layer_idx)
                    type_all_layers.setdefault(target, set()).add(layer_idx + 1)

                node_type = []
                node_layers = []  # Track primary layer for positioning
                node_labels = []  # Labels with all layers
                for layer_idx in sorted(all_types_by_layer.keys()):
                    layer_nodes = sorted(all_types_by_layer[layer_idx])
                    for node in layer_nodes:
                        node_type.append(node)
                        node_layers.append(layer_idx)
                        # Create label showing all layers
                        all_layers = sorted(type_all_layers[node])
                        if len(all_layers) == 1:
                            node_labels.append(f"{node} (L{all_layers[0]})")
                        else:
                            layers_str = ','.join(map(str, all_layers))
                            node_labels.append(f"{node} (L{layers_str})")

                node_to_idx = {node: idx for idx, node in enumerate(node_type)}
                node_type_color = [self.node_color] * len(node_type)
                
                # Create custom hover text for nodes
                node_hover_text = []
                for idx, node in enumerate(node_type):
                    all_layers = sorted(type_all_layers[node])
                    layers_display = ', '.join(map(str, all_layers))
                    if node in target_type:
                        node_type_color[idx] = self.target_color
                        node_hover_text.append(f"{node}<br>Layers: {layers_display}<br>(Target)")
                    else:
                        node_hover_text.append(f"{node}<br>Layers: {layers_display}")

                source_indices = []
                target_indices = []
                weights_for_links = []
                ratios_for_links = []
                probs_for_links = []

                for (layer_idx, source, target), metrics in edge_agg.items():
                    source_indices.append(node_to_idx[source])
                    target_indices.append(node_to_idx[target])
                    weights_for_links.append(metrics['weight'])
                    ratios_for_links.append(metrics['ratio'])
                    probs_for_links.append(metrics['prob'])

                # Debug: Print value ranges
                print(f"\nSankey value ranges:")
                print(f"  Weights: min={min(weights_for_links):.1f}, max={max(weights_for_links):.1f}")
                print(f"  Ratios: min={min(ratios_for_links):.4f}, max={max(ratios_for_links):.4f}")
                print(f"  Probs: min={min(probs_for_links):.4f}, max={max(probs_for_links):.4f}")

                fig_type_weight = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=5,
                        thickness=5,
                        line=dict(color="black", width=0),
                        label=node_labels,
                        color=node_type_color,
                        customdata=node_hover_text,
                        hovertemplate='%{customdata}<extra></extra>'
                    ),
                    link=dict(
                        source=source_indices,
                        target=target_indices,
                        value=weights_for_links,
                        color=self.link_color,
                        customdata=weights_for_links,
                        hovertemplate='%{customdata:.1f} synapses<extra></extra>'
                    )
                )])
                fig_type_weight.update_layout(
                    title_text='Sankey diagram of connections to targets<br>based on neuron type (by synapse count)',
                    font_size=12
                )
                fig_type_weight.write_html(os.path.join(self.path_folder, 'Sankey_type_path_snp.html'), auto_open=self.showfig, include_plotlyjs='cdn')

                # Create custom hover text for ratio values
                ratio_hover_text = [f"{node_type[source_indices[i]]} → {node_type[target_indices[i]]}<br>Ratio: {ratios_for_links[i]:.4f}" 
                                   for i in range(len(source_indices))]

                fig_type_ratio = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=5,
                        thickness=5,
                        line=dict(color="black", width=0),
                        label=node_labels,
                        color=node_type_color,
                        customdata=node_hover_text,
                        hovertemplate='%{customdata}<extra></extra>'
                    ),
                    link=dict(
                        source=source_indices,
                        target=target_indices,
                        value=ratios_for_links,
                        color=self.link_color,
                        customdata=ratios_for_links,
                        hovertemplate='%{customdata:.4f}<extra></extra>'
                    )
                )])
                fig_type_ratio.update_layout(
                    title_text='Sankey diagram of connections to targets<br>based on neuron type (by connection ratio)',
                    font_size=12
                )
                fig_type_ratio.write_html(os.path.join(self.path_folder, 'Sankey_type_path_ratio.html'), auto_open=self.showfig, include_plotlyjs='cdn')

                # Create custom hover text for probability values
                prob_hover_text = [f"{node_type[source_indices[i]]} → {node_type[target_indices[i]]}<br>Prob: {probs_for_links[i]:.4f}" 
                                  for i in range(len(source_indices))]

                fig_type_prob = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=5,
                        thickness=5,
                        line=dict(color="black", width=0),
                        label=node_labels,
                        color=node_type_color,
                        customdata=node_hover_text,
                        hovertemplate='%{customdata}<extra></extra>'
                    ),
                    link=dict(
                        source=source_indices,
                        target=target_indices,
                        value=probs_for_links,
                        color=self.link_color,
                        customdata=probs_for_links,
                        hovertemplate='%{customdata:.4f}<extra></extra>'
                    )
                )])
                fig_type_prob.update_layout(
                    title_text='Sankey diagram of connections to targets<br>based on neuron type (by traversal probability)',
                    font_size=12
                )
                fig_type_prob.write_html(os.path.join(self.path_folder, 'Sankey_type_path_prob.html'), auto_open=self.showfig, include_plotlyjs='cdn')

                print(f'Created 3 type-level Sankey diagrams with {len(node_type)} nodes and {len(weights_for_links)} edges')

        # Build bodyId-level Sankey from connection_info sheet (conn_inpath)
        # Show ALL connections in the network, not just those in specific paths
        if find_bodyId_path and len(conn_inpath) > 0:
            # Extract all edges directly from conn_inpath DataFrame
            edge_weight_bodyId = {}
            edge_ratio_bodyId = {}
            edge_prob_bodyId = {}
            
            for idx, row in conn_inpath.iterrows():
                layer_label = row['conn_layer']
                layer_idx = int(layer_label.split('->')[0])
                source_id = int(row['bodyId_pre'])
                target_id = int(row['bodyId_post'])
                edge_key = (layer_idx, source_id, target_id)
                
                # Aggregate if same edge appears multiple times (shouldn't happen but be safe)
                if edge_key in edge_weight_bodyId:
                    edge_weight_bodyId[edge_key] += float(row['weight'])
                    # For ratio and prob, use weighted average
                    edge_ratio_bodyId[edge_key] = max(edge_ratio_bodyId[edge_key], float(row['connection_ratio']))
                    edge_prob_bodyId[edge_key] = max(edge_prob_bodyId[edge_key], float(row['traversal_probability']))
                else:
                    edge_weight_bodyId[edge_key] = float(row['weight'])
                    edge_ratio_bodyId[edge_key] = max(0.0, min(1.0, float(row['connection_ratio'])))
                    edge_prob_bodyId[edge_key] = max(0.0, min(1.0, float(row['traversal_probability'])))
            
            if len(edge_weight_bodyId) == 0:
                print('\033[33mWarning: No connections found in connection_info sheet for bodyId Sankey diagrams.\033[0m')
            else:
                # Build node list by layer
                all_bodyIds_by_layer = {}
                for (layer_idx, source, target) in edge_weight_bodyId.keys():
                    all_bodyIds_by_layer.setdefault(layer_idx, set()).add(source)
                    all_bodyIds_by_layer.setdefault(layer_idx + 1, set()).add(target)

                node_bodyId = []
                for layer_idx in sorted(all_bodyIds_by_layer.keys()):
                    node_bodyId.extend(sorted(all_bodyIds_by_layer[layer_idx]))

                # Fetch neuron types for labels (use local dataset if available)
                node_df = self._fetch_neurons_local_or_api(node_bodyId, columns=['bodyId', 'type'])
                for ind in node_df.index:
                    if node_df.at[ind, 'type'] is None:
                        node_df.at[ind, 'type'] = 'None'
                bodyId_to_type = dict(zip(node_df['bodyId'], node_df['type']))
                node_bodyId_labels = [f"{bodyId_to_type.get(bid, 'Unknown')}_{bid}" for bid in node_bodyId]

                node_to_idx_bodyId = {node: idx for idx, node in enumerate(node_bodyId)}
                node_bodyId_color = [self.node_color] * len(node_bodyId)
                for idx, bodyId in enumerate(node_bodyId):
                    if bodyId in target_ID:
                        node_bodyId_color[idx] = self.target_color

                source_indices_bodyId = []
                target_indices_bodyId = []
                weights_bodyId = []

                for (layer_idx, source, target), weight in edge_weight_bodyId.items():
                    source_indices_bodyId.append(node_to_idx_bodyId[source])
                    target_indices_bodyId.append(node_to_idx_bodyId[target])
                    weights_bodyId.append(weight)

                fig_bodyId = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=1,
                        thickness=5,
                        line=dict(color="black", width=0),
                        label=node_bodyId_labels,
                        color=node_bodyId_color
                    ),
                    link=dict(
                        source=source_indices_bodyId,
                        target=target_indices_bodyId,
                        value=weights_bodyId,
                        color=self.link_color
                    )
                )])
                fig_bodyId.update_layout(
                    title_text='Sankey diagram of connections to targets<br>based on neuron bodyId',
                    font_size=6
                )
                fig_bodyId.write_html(os.path.join(self.path_folder, 'Sankey_bodyId_path.html'), auto_open=self.showfig, include_plotlyjs='cdn')

                print(f'Created bodyId-level Sankey diagram with {len(node_bodyId)} nodes and {len(weights_bodyId)} edges')

        
        # VisualizePath network visualization
        print('\nCreating interactive network visualizations...')
        try:
            
            # Create network from path_type if it exists
            if len(path_df_type) > 0:
                # Filter paths if pathN_to_show is specified
                paths_to_visualize = path_df_type
                if self.pathN_to_show > 0 and len(path_df_type) > self.pathN_to_show:
                    # Calculate path strength (product of traversal probabilities)
                    # Paths are already sorted by traversal_probability in sv.getAllPath()
                    # Just take the first N paths
                    paths_to_visualize = path_df_type.head(self.pathN_to_show).copy()
                    print(f'  Showing top {self.pathN_to_show} paths (by traversal_probability) out of {len(path_df_type)} total paths')
                else:
                    print(f'  Showing all {len(path_df_type)} paths')
                
                vp = VisualizePath(
                    path_file=paths_to_visualize,
                    output_folder=self.path_folder,
                    source_color=self.source_color if hasattr(self, 'source_color') else '#1f77b4',
                    intermediate_color=self.intermediate_color if hasattr(self, 'intermediate_color') else '#2ca02c',
                    target_color=self.target_color if hasattr(self, 'target_color') else '#d62728',
                    link_color=self.link_color if hasattr(self, 'link_color') else 'rgba(100,100,100,0.3)',
                    network_layout=self.network_layout if hasattr(self, 'network_layout') else 'hierarchical',
                    showfig=self.showfig
                )
                vp.visualize()
                print('  Created network_selected_paths.html and sankey_selected_paths.html')
            else:
                print('  No paths found to visualize')
        except Exception as e:
            print(f'  Warning: VisualizePath visualization failed: {e}')
            import traceback
            traceback.print_exc()
        
        # Create type-level heatmap visualization
        print('Creating type-level connection heatmap...')
        try:
            if len(conn_types) > 0:
                # Build connection matrix from conn_types
                # Group by type_pre and type_post, summing weights
                conn_matrix_data = conn_types.groupby(['type_pre', 'type_post'])['weight'].sum().reset_index()
                
                # Create matrix
                conn_matrix_type = conn_matrix_data.pivot(
                    index='type_pre', 
                    columns='type_post', 
                    values='weight'
                ).fillna(0)
                
                # Use CreateHeatmap class
                heatmap_gen = sv.CreateHeatmap(
                    output_folder=self.path_folder,
                    showfig=self.showfig
                )
                heatmap_gen.add_heatmap(
                    matrix=conn_matrix_type,
                    name='heatmap_path_type',
                    title=f'Connection Heatmap: {self.source_fname} to {self.target_fname}<br>Type-level connections in paths',
                    color_scale='purple',
                    interactive=True
                )
                heatmap_gen.create_all()
                print('  Created heatmap_path_type.html')
            else:
                print('  No connections to visualize in heatmap')
        except Exception as e:
            print(f'  Warning: Heatmap visualization failed: {e}')
            import traceback
            traceback.print_exc()
        
        print('Done\n')
    
    def _create_interactive_network_for_path(self, conn_types, conn_inpath, neuron_layers, target_type, target_ID, output_folder):
        '''Create interactive network visualizations for FindPath method'''
        
        # Network by type
        print('Building interactive network by type...')
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
        tuple: (neurons_set, edges_set, edges_with_layer_set, path_count, pairs_with_paths)
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
            nonlocal path_count, neurons_in_paths, edges_in_paths, edges_in_paths_with_layer
            
            # Check if current node is a target
            if current in target_set:
                # Found a complete path to a target
                path_count += 1
                neurons_in_paths.update(path)
                
                # Record this source-target pair
                source_node = path[0]
                pairs_with_paths_dict[(source_node, current)] = True
                
                # Add edges from this path
                for i in range(len(path) - 1):
                    pre_node = path[i]
                    post_node = path[i+1]
                    edges_in_paths.add((pre_node, post_node))
                    
                    # Determine layer(s) for this edge
                    for layer_idx, layer_set in enumerate(layer_neurons):
                        if pre_node in layer_set:
                            edges_in_paths_with_layer.add((layer_idx, pre_node, post_node))
                
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
                path_count, pairs_with_paths, total_pairs_checked)
    
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
        # Handle deprecated parameter
        if exclude_searched_neurons is not None:
            forward_only = exclude_searched_neurons
            print('⚠️  Warning: exclude_searched_neurons is deprecated. Use forward_only instead.')
            print(f'   Setting forward_only={forward_only}')
        
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
        
        self.allpath_folder = os.path.join(self.save_folder, f'allpaths{param_suffix}')
        if not os.path.exists(self.allpath_folder): os.makedirs(self.allpath_folder)
        
        # Save all attributes and parameters to the allpaths folder
        with open(os.path.join(self.allpath_folder, 'all_attributes.json'), 'w') as f:
            json.dump(self.__dict__, f, indent=4, default=lambda o: '<not serializable>')
        
        with open(os.path.join(self.allpath_folder, 'parameters.txt'), 'w') as f:
            f.write(f'Parameters for processing {self.source_fname} to {self.target_fname}:\n')
            for key, value in self.parameter_dict.items():
                keylen = len(key)
                f.write(f'{key}:{" "*(30-keylen)}{value}\n')
            f.write('\n')
        
        source_ID = self.source_df['bodyId'].unique()
        target_ID = self.target_df['bodyId'].unique()
        target_type = self.target_df['type'].unique()
        
        # PHASE 1: Fetch all connections in the network up to max_interlayer layers
        print(f'\n=== PHASE 1: Fetching all network layers (0 to {self.max_interlayer + 1}) ===')
        if forward_only:
            print('Mode: Layer-by-layer querying (query each neuron once - RECOMMENDED)')
            print('Note: Still fetches ALL connections including recurrent/reciprocal ones')
        else:
            print('Mode: Comprehensive re-querying (re-query all neurons at each layer)')
            print('Note: Slower but ensures no connections missed due to filtering')
        print()
        
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
                print(f'Layer {layer_idx} is empty, stopping.')
                break
            
            # Fetch connections (fetch with weight≥1, filter by all criteria together later)
            print(f'Layer {layer_idx}->{layer_idx+1}:')
            conn_df = self._fetch_connections_with_cache(
                upstream_bodyIds=neurons_to_fetch,
                downstream_bodyIds=None,
                min_weight=self.min_synapse_num,
                min_conn_ratio=self.min_ratio,
                min_traversal_prob=self.min_traversal_probability
            )
            
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
            
            if forward_only:
                print(f'Layer {layer_idx}->{layer_idx+1}: {len(post_neurons)} downstream neurons, {len(next_layer)} new, {len(conn_df)} connections')
            else:
                print(f'Layer {layer_idx}->{layer_idx+1}: {len(post_neurons)} total downstream, {len(next_layer)} new neurons, {len(conn_df)} connections')
        
        print(f'\nTotal neurons in network: {len(all_neurons_in_network)}')
        print(f'Total layers fetched: {len(layer_neurons)}')
        
        # PHASE 2: Identify which targets exist in the searched network
        print(f'\n=== PHASE 2: Identifying targets in the network ===')
        
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
        print(f'Targets found in network: {targetNum_checked} / {targetNum}')
        
        if targetNum_checked == 0:
            print('\033[33mNo target neurons found in the searched network. Cannot construct paths.\033[0m')
            return
        
        # Print target distribution by layer (same target can appear in multiple layers)
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
                target_list = targets_in_layer['type'].tolist()
            
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
        print(f'\n=== PHASE 3: Finding all paths from sources to targets ===')
        print('Using graph-based pathfinding to handle reciprocal connections...')
        
        # Create INITIAL real layer mapping (neuron ID -> discovery layer)
        # Targets will be updated later based on their actual appearance in paths
        real_layer_map_bodyId = {}
        for layer_idx, layer_set in enumerate(layer_neurons):
            for neuron_id in layer_set:
                # Use earliest layer if neuron appears in multiple layers
                if neuron_id not in real_layer_map_bodyId:
                    real_layer_map_bodyId[neuron_id] = layer_idx
        
        print(f'Created initial real layer map for {len(real_layer_map_bodyId)} neurons')
        print(f'  Note: Target real layers will be updated after pathfinding completes')
        
        # Build a directed graph from all connections
        print('Building connection graph...', end=' ')
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
        print(f'Done! ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)')
        
        # Find all neurons that are on ANY path from any source to any target
        # with path length ≤ max_interlayer
        neurons_in_paths = set()
        edges_in_paths = set()  # Stores (pre, post) pairs
        edges_in_paths_with_layer = set()  # Stores (layer_idx, pre, post) to track layer-specific edges
        
        print(f'\nSearching paths: {len(source_ID)} sources × {len(targets_found)} targets = {len(source_ID) * len(targets_found)} pairs')
        print(f'Maximum path length: {self.max_interlayer + 1} edges')
        print(f'Using optimized DFS algorithm (explores shared path segments only once)')
        
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
                print(f'Using parallel processing with {n_processes} processes...')
                
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
                
                print(f'Split into {len(source_chunks)} chunks (~{chunk_size} sources per chunk)')
                print(f'Each chunk will explore paths to all {len(targets_set)} targets')
                
                # More realistic time estimate based on graph complexity
                # With DFS optimization, each source is explored once (not once per target)
                # Factors affecting speed:
                # - Graph size (nodes and edges)
                # - Path length (cutoff)
                # - Graph density (average degree)
                
                # Base estimate on graph complexity
                avg_degree = G.number_of_edges() / G.number_of_nodes() if G.number_of_nodes() > 0 else 1
                path_complexity = self.max_interlayer + 1  # Maximum path length
                
                # Empirical formula - DFS is generally faster than pair-wise NetworkX
                # - Small graphs (<10k nodes, degree<10): ~100 sources/sec per process
                # - Medium graphs (10k-100k nodes, degree 10-100): ~20 sources/sec per process
                # - Large graphs (>100k nodes, degree>100): ~5 sources/sec per process
                
                if G.number_of_nodes() < 10000 and avg_degree < 10:
                    base_speed = 100  # sources/sec per process
                elif G.number_of_nodes() < 100000 and avg_degree < 100:
                    base_speed = 20
                else:
                    base_speed = 5
                
                # Adjust for path length (longer paths = exponentially slower)
                # Each additional layer approximately doubles the search space
                complexity_factor = 2 ** (path_complexity - 2)  # Normalized to length 2
                adjusted_speed = base_speed / max(1, complexity_factor * 0.5)
                
                # Adjust for number of targets (more targets = slightly longer per source)
                # But much less impact than in pair-wise approach since we explore tree once
                target_factor = 1 + (len(targets_set) / 1000) * 0.1  # Small penalty for many targets
                adjusted_speed = adjusted_speed / target_factor
                
                # Total estimated speed with parallel processing
                total_estimated_speed = adjusted_speed * n_processes
                estimated_time = len(sources_list) / total_estimated_speed if total_estimated_speed > 0 else 0
                
                # Add some buffer (actual time is often 20-50% longer due to overhead)
                estimated_time *= 1.3
                
                if estimated_time < 10:
                    time_str = f"~{estimated_time:.0f} seconds"
                elif estimated_time < 120:
                    time_str = f"~{estimated_time/60:.1f} minutes"
                else:
                    time_str = f"~{estimated_time/60:.0f} minutes"
                
                print(f'Estimated time: {time_str} (graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, avg degree: {avg_degree:.1f})')
                print(f'Processing...\n')
                
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
                
                # For dynamic ETA calculation using exponentially weighted moving average (EWMA)
                ewma_speed = None  # Exponentially weighted moving average of speed
                alpha = 0.3  # Smoothing factor (0.2-0.4 is typical, lower = smoother but slower to adapt)
                first_chunk_time = None  # Track when first chunk completes (exclude startup overhead)
                productive_start_time = None  # Start time for actual work (after first chunk)
                
                print(f'⏳ Starting {n_processes} worker processes...')
                print(f'   (First update will appear when a chunk completes)')
                print()
                
                with mp.Pool(processes=n_processes) as pool:
                    # Use imap_unordered for progress tracking (returns results as they complete)
                    for neurons_set, edges_set, edges_layer_set, p_count, p_with_paths, chunk_size_actual in pool.imap_unordered(
                        self._find_paths_dfs_optimized, args_list
                    ):
                        # Update totals
                        neurons_in_paths.update(neurons_set)
                        edges_in_paths.update(edges_set)
                        edges_in_paths_with_layer.update(edges_layer_set)
                        path_count += p_count
                        pairs_with_paths += p_with_paths
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
                            print(f'   ⚡ Workers initialized in {startup_overhead:.1f}s, starting main processing...\n')
                        
                        # Calculate current speed using ONLY productive time (excludes startup)
                        productive_elapsed = current_time - productive_start_time if productive_start_time else 0.1
                        current_speed = sources_processed / productive_elapsed if productive_elapsed > 0 else 0
                        
                        # Update EWMA speed (adapts to changing processing rates)
                        if ewma_speed is None:
                            # Initialize with first measurement
                            ewma_speed = current_speed
                        else:
                            # EWMA formula: new_avg = alpha * current + (1 - alpha) * old_avg
                            # This gives more weight to recent speeds while smoothing out noise
                            ewma_speed = alpha * current_speed + (1 - alpha) * ewma_speed
                        
                        progress_pct = (sources_processed / len(sources_list)) * 100
                        remaining_sources = len(sources_list) - sources_processed
                        eta_seconds = remaining_sources / ewma_speed if ewma_speed > 0 else 0
                        
                        # Format ETA in HH:mm:ss
                        hours = int(eta_seconds // 3600)
                        minutes = int((eta_seconds % 3600) // 60)
                        seconds = int(eta_seconds % 60)
                        eta_str = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
                        
                        # Update more frequently - show every chunk or every 0.5 seconds
                        should_update = (current_time - last_update >= update_interval or 
                                       chunks_completed == 1 or  # Always show first chunk
                                       chunks_completed % 5 == 0 or  # Show every 5 chunks
                                       chunks_completed == len(source_chunks))  # Always show completion
                        
                        if should_update:
                            # Use \033[K to clear to end of line (removes residual characters)
                            print(f'\r   Progress: {sources_processed}/{len(sources_list)} sources ({progress_pct:.1f}%) | ETA: {eta_str}\033[K', end='', flush=True)
                            last_update = current_time
                
                # Final newline
                print()
                
                elapsed = time.time() - start_time
                print(f'\n✅ Parallel pathfinding complete in {elapsed:.1f}s!')
                print(f'   Average: {len(sources_list)/elapsed:.1f} sources/s (explored {len(targets_set)} targets per source)')
                print(f'   Processed by {n_processes} workers across {len(source_chunks)} chunks')
        
        if not use_parallel:
            print('Using sequential processing (optimized DFS)...')
            print('This may take a while for large datasets...\n')
            
            path_count = 0
            sources_processed = 0
            pairs_with_paths_dict = {}
            
            # Progress tracking with EWMA for dynamic ETA
            import time
            start_time = time.time()
            last_update = start_time
            update_interval = 2.0  # Update every 2 seconds
            
            ewma_speed = None  # Exponentially weighted moving average of speed
            alpha = 0.3  # Smoothing factor (same as parallel mode)
            
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
                    
                    # Add edges from this path
                    for i in range(len(path) - 1):
                        pre_node = path[i]
                        post_node = path[i+1]
                        edges_in_paths.add((pre_node, post_node))
                        
                        # Determine layer(s) for this edge
                        for layer_idx, layer_set in enumerate(layer_neurons):
                            if pre_node in layer_set:
                                edges_in_paths_with_layer.add((layer_idx, pre_node, post_node))
                
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
                    
                    # Calculate current speed
                    current_speed = sources_processed / elapsed if elapsed > 0 else 0
                    
                    # Update EWMA speed (same algorithm as parallel mode)
                    if ewma_speed is None:
                        ewma_speed = current_speed
                    else:
                        ewma_speed = alpha * current_speed + (1 - alpha) * ewma_speed
                    
                    progress_pct = (sources_processed / len(source_ID)) * 100
                    remaining_sources = len(source_ID) - sources_processed
                    eta_seconds = remaining_sources / ewma_speed if ewma_speed > 0 else 0
                    
                    # Format ETA in HH:mm:ss
                    hours = int(eta_seconds // 3600)
                    minutes = int((eta_seconds % 3600) // 60)
                    seconds = int(eta_seconds % 60)
                    eta_str = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
                    
                    pairs_with_paths = len(pairs_with_paths_dict)
                    
                    # Use \033[K to clear to end of line
                    print(f'\r   Progress: {sources_processed}/{len(source_ID)} sources ({progress_pct:.1f}%) | ETA: {eta_str}\033[K', end='', flush=True)
                    last_update = current_time
            
            pairs_with_paths = len(pairs_with_paths_dict)
            
            # Final update
            elapsed = time.time() - start_time
            # Use \033[K to clear to end of line
            print(f'\r   Progress: {sources_processed}/{len(source_ID)} sources (100.0%) | Completed in {elapsed:.1f}s\033[K')
        
        print(f'\n✅ Pathfinding complete!')
        print(f'   Total paths found: {path_count:,}')
        print(f'   Neurons in valid paths: {len(neurons_in_paths):,}')
        print(f'   Unique edges in valid paths: {len(edges_in_paths):,}')
        print(f'   Layer-specific edges in valid paths: {len(edges_in_paths_with_layer):,}')
        
        # Now extract connections, keeping ALL layer-specific occurrences
        # This means if neuron A→B exists in both Layer 0→1 and Layer 2→3, both are kept
        conn_inpath = pd.DataFrame()
        conn_types = pd.DataFrame()
        weight_layers = {}
        
        for layer_idx, conn_df in enumerate(all_connections):
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
            conn_enriched, conn_type = sv.EnrichConnectionTable(
                conn_filtered_no_layer, 
                dataset=self.dataset, 
                script_path=self.script_path,
                target_neurons_df=neurons_in_layer_df
            )
            
            # Add conn_layer column AFTER enrichment
            conn_enriched.insert(loc=0, column='conn_layer', value=layer_label)
            conn_type.insert(loc=0, column='conn_layer', value=layer_label)
            
            conn_inpath = pd.concat([conn_inpath, conn_enriched])
            conn_types = pd.concat([conn_types, conn_type])
            
            weight_layers[layer_label] = conn_enriched['weight'].sum()
            
            print(f'Layer {layer_label}: {len(conn_filtered)} connections kept')
        
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
        # Targets should have real_layer = their latest appearance layer to allow all paths
        print('\n=== Updating target real layers based on path appearances ===')
        target_appearance_layers = {}  # Track all layers each target appears in
        
        for layer_idx, layer in enumerate(neuron_layers):
            for neuron_id in layer:
                if neuron_id in targets_found:
                    if neuron_id not in target_appearance_layers:
                        target_appearance_layers[neuron_id] = []
                    target_appearance_layers[neuron_id].append(layer_idx)
        
        # Update real_layer_map for targets to their latest appearance
        for target_id, appearance_layers in target_appearance_layers.items():
            latest_layer = max(appearance_layers)
            # Assign target real_layer as max(latest_appearance, max_interlayer+1)
            # This ensures all intermediate → target connections are valid
            real_layer_map_bodyId[target_id] = max(latest_layer, self.max_interlayer + 1)
        
        # Print target appearance information
        print(f'\nTarget neurons appearance in paths:')
        for target_id in sorted(target_appearance_layers.keys()):
            appearance_layers = target_appearance_layers[target_id]
            real_layer = real_layer_map_bodyId[target_id]
            layers_str = ', '.join(map(str, sorted(appearance_layers)))
            if len(appearance_layers) == 1:
                print(f'  Target {target_id}: appears in layer {appearance_layers[0]}, real_layer = {real_layer}')
            else:
                print(f'  Target {target_id}: appears in layers [{layers_str}], real_layer = {real_layer}')
        
        if len(target_appearance_layers) == 0:
            print('  No targets found in paths')
        
        # Sort the combined connection data
        conn_inpath = conn_inpath.sort_values(by=['conn_layer','traversal_probability','weight'],ascending=[True,False,False])
        conn_inpath = conn_inpath.reset_index(drop=True)
        conn_types = conn_types.sort_values(by=['conn_layer','traversal_probability','weight'],ascending=[True,False,False])
        conn_types = conn_types.reset_index(drop=True)

        totalweight_df = pd.DataFrame(weight_layers.items(),columns=['conn_layer','weight'])
        totalweight_df = totalweight_df.sort_values(by='conn_layer',ascending=True)
        
        # Create type-level real layer map from bodyId-level real layers
        # For type-level analysis, use the earliest layer any neuron of that type appears
        # Targets already have their real layers updated based on actual path appearances
        real_layer_map_type = {}
        target_types_set = set(self.target_df.loc[self.target_df.Checked, 'type'].unique())
        target_type_appearances = {}  # Track appearance layers for target types
        
        if len(conn_inpath) > 0:
            for idx in conn_inpath.index:
                bodyId_pre = conn_inpath.at[idx, 'bodyId_pre']
                bodyId_post = conn_inpath.at[idx, 'bodyId_post']
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
        
        print(f'\nCreated type-level real layer map for {len(real_layer_map_type)} types')
        
        # Print target type appearance information
        if target_type_appearances:
            print(f'\nTarget types appearance in paths:')
            for target_type in sorted(target_type_appearances.keys()):
                appearance_layers = sorted(list(target_type_appearances[target_type]))
                real_layer = real_layer_map_type.get(target_type, -1)
                layers_str = ', '.join(map(str, appearance_layers))
                if len(appearance_layers) == 1:
                    print(f'  Type {target_type}: appears in layer {appearance_layers[0]}, real_layer = {real_layer}')
                else:
                    print(f'  Type {target_type}: appears in layers [{layers_str}], real_layer = {real_layer}')

        # Mark which source neurons are in paths to targets
        if len(conn_inpath) > 0:
            source_inpath = conn_inpath.loc[conn_inpath.conn_layer=='0->1','bodyId_pre'].unique()
            self.source_df.insert(loc=0,column='isInPath',value=False)
            self.source_df.loc[self.source_df.bodyId.isin(source_inpath),'isInPath'] = True
        
        # Print statistics about paths
        print(f'\nPath Network Statistics (source to target):')
        print(f'Total connections in paths: {len(conn_inpath)}')
        print(f'Total connection types in paths: {len(conn_types)}')
        total_neurons = sum(len(layer) for layer in neuron_layers)
        print(f'Total neurons in paths: {total_neurons}')
        for i, layer in enumerate(neuron_layers):
            print(f'  Layer {i}: {len(layer)} neurons')
        
        # Print target distribution and which targets were found in each layer
        print('\nTarget neurons by layer:')
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
                print(f'  Layer {layer_idx}: {len(found_in_layer)}/{len(targets_in_layer)} targets found')
                if len(found_in_layer) > 0 and len(found_in_layer) <= 20:
                    print(f'    Found: {found_in_layer["bodyId"].tolist()}')
            else:  # filter_by == 'type'
                found_in_layer = targets_in_layer[
                    targets_in_layer['type'].isin(conn_types['type_post'].unique())
                ]
                all_found_targets.update(found_in_layer['type'].tolist())
                print(f'  Layer {layer_idx}: {len(found_in_layer)}/{len(targets_in_layer)} targets found')
                if len(found_in_layer) > 0 and len(found_in_layer) <= 20:
                    print(f'    Found: {found_in_layer["type"].tolist()}')
        
        print(f'\nTotal found targets: {len(all_found_targets)}/{total_checked_targets}')
        
        # saving data
        output_excel_name = os.path.join(self.allpath_folder, self.source_fname + '_to_' + self.target_fname + '_allpaths_info.xlsx')
        with pd.ExcelWriter(output_excel_name, mode='w', engine='xlsxwriter') as writer:
            self.parameter_df.to_excel(writer,sheet_name='parameters',index=False)
            worksheet = writer.sheets['parameters']
            worksheet.set_column('A:A', 30, writer.book.add_format({'bold': True, 'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
            worksheet.set_column('B:B', 30, writer.book.add_format({'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
            
            self.source_df.to_excel(writer,sheet_name='source_neurons')
            self.target_df.to_excel(writer,sheet_name='target_neurons')
            totalweight_df.to_excel(writer,sheet_name='total_weight_layer')
            conn_inpath.to_excel(writer,sheet_name='connection_info')
            conn_types.to_excel(writer,sheet_name='connection_type')
        
        # Get all paths (by type) - this includes paths of all lengths
        path_df_type = pd.DataFrame()
        print('\nAnalyzing all paths by type (all lengths):')
        print('Applying real layer validation: excluding backward and recurrent paths...')
        path_df_type,_ = sv.getAllPath(conn_data = conn_types,
                                    targets = self.target_df.loc[self.target_df.Checked,'type'].unique().tolist(),
                                    traversal_probability_threshold = self.min_traversal_probability,
                                    max_path_length = self.max_interlayer + 1,
                                    real_layer_map = real_layer_map_type if forward_only else None)
        
        # Filter out paths with any zero-weight hops
        # This happens when bodyId-level connections exist but type-level aggregation results in 0 weight
        if len(path_df_type) > 0:
            before_filter = len(path_df_type)
            path_df_type = path_df_type[
                path_df_type['weights'].apply(lambda w_list: all(w > 0 for w in w_list))
            ]
            after_filter = len(path_df_type)
            if before_filter > after_filter:
                print(f'  Removed {before_filter - after_filter} paths with zero-weight hops at type level')
        
        path_df_type = sv.split_path(path_df_type)
        path_df_type, path_df_type_excluded = sv.path_filter(path_df_type,self.keyword_in_path_to_remove)
        
        print('💾 Saving path_type data to Excel...')
        with pd.ExcelWriter(output_excel_name, mode='a', engine='openpyxl') as writer:
            path_df_type.to_excel(writer,sheet_name='path_type')
            path_df_type_excluded.to_excel(writer,sheet_name='path_type_excluded')
        print('   ✓ path_type sheets saved')
        
        # Get all paths (by bodyId) - this includes paths of all lengths
        if find_bodyId_path:
            path_df_bodyId = pd.DataFrame()
            print('Analyzing all paths by bodyId (all lengths):')
            print('Applying real layer validation: excluding backward and recurrent paths...')
            path_df_bodyId,_ = sv.getAllPath(conn_data = conn_inpath,
                                        targets = self.target_df.loc[self.target_df.Checked,'bodyId'].tolist(),
                                        traversal_probability_threshold = self.min_traversal_probability,
                                        max_path_length = self.max_interlayer + 1,
                                        real_layer_map = real_layer_map_bodyId if forward_only else None)
            if len(path_df_bodyId) > 1048575:
                path_df_bodyId = path_df_bodyId.iloc[:1048575,:]
                print('\033[33mWarning: Excel has a limit of 1048576 rows, only the first 1048575 rows are saved.\033[0m')
            print('💾 Saving path_bodyId data to Excel...')
            with pd.ExcelWriter(output_excel_name, mode='a', engine='openpyxl') as writer:
                path_df_bodyId.to_excel(writer,sheet_name='path_bodyId')
            print('   ✓ path_bodyId sheet saved')
        
        # save interlayer info to excel
        print('💾 Saving interlayer neuron info to Excel...')
        
        # Try to load complete neuron dataset for faster lookup
        dataset_path = os.path.join(
            self.script_path,
            'datasets',
            f"{self.dataset.replace(':', '_').replace('.', '_')}_allneurons_neuron_df.csv"
        )
        use_local_dataset = os.path.exists(dataset_path)
        if use_local_dataset:
            print(f'   Using local dataset: {os.path.basename(dataset_path)}')
            ndf_complete = pd.read_csv(dataset_path, header=0, index_col=0)
        else:
            print(f'   Local dataset not found, will use API calls')
            # Ensure client is logged in before API calls
            if self.client_hemibrain is None:
                from neuprint import Client, set_default_client
                self.client_hemibrain = Client(self.server, self.dataset, self.token)
                set_default_client(self.client_hemibrain)
        
        interlayers = []
        num_layers = len(neuron_layers[1:])
        for layer_idx, neurons in enumerate(neuron_layers[1:], 1):
            print(f'   Fetching layer {layer_idx}/{num_layers} info ({len(neurons)} neurons)...', end='', flush=True)
            
            if use_local_dataset:
                # Fast: lookup from local CSV
                n_df = ndf_complete[ndf_complete['bodyId'].isin(neurons)].copy()
            else:
                # Slow: API call to neuprint (client already logged in above)
                n_df,_ = fetch_neurons(NeuronCriteria(bodyId=neurons))
            
            interlayers.append(n_df)
            print(' ✓')
        
        print('   Writing to Excel...', end='', flush=True)
        with pd.ExcelWriter(output_excel_name, mode='a', engine='openpyxl') as writer:
            for i in range(len(interlayers)):
                interlayers[i].to_excel(writer,sheet_name='layer_'+str(i+1))
        print(' ✓')
        print('   ✓ Interlayer sheets saved')
        print('Done\n')
        
        # Build Sankey diagrams from path data (ensures only paths to targets are shown)
        print('Building Sankey diagrams from path data...')
        
        # Helper function to parse path_block and extract edges with their positions
        def parse_path_to_edges(path_block):
            """Parse 'A -> B -> C' into list of (A, B), (B, C) with layer info"""
            nodes = [n.strip() for n in path_block.split('->')]
            edges = []
            for i in range(len(nodes) - 1):
                edges.append((i, nodes[i], nodes[i+1]))  # (layer_idx, source, target)
            return edges
        
        # If forward_only=True, extract edges from path_type to filter visualizations
        edges_in_path_type = set()
        if forward_only and len(path_df_type) > 0:
            print('Extracting edges from path_type for filtered visualization...')
            for idx in path_df_type.index:
                path_block = path_df_type.at[idx, 'path_block']
                edges = parse_path_to_edges(path_block)
                for layer_idx, source, target in edges:
                    # Skip self-connections at type level for cleaner visualization
                    if source != target:
                        edges_in_path_type.add((layer_idx, source, target))
            print(f'  Extracted {len(edges_in_path_type)} unique edges from paths (excluding type self-connections)')
        
        # Build Sankey diagram from connection_type sheet
        # forward_only=True: Show only edges in path_type (filtered)
        # forward_only=False: Show ALL connections in conn_types (complete graph)
        if len(conn_types) > 0:
            # Extract all edges directly from conn_types DataFrame
            edge_weight_type = {}
            edge_ratio_type = {}
            edge_prob_type = {}
            for idx, row in conn_types.iterrows():
                layer_label = row['conn_layer']
                layer_idx = int(layer_label.split('->')[0])
                source = row['type_pre']
                target = row['type_post']
                edge_key = (layer_idx, source, target)
                
                # If forward_only=True, only include edges that are in path_type
                # This filters visualization to show only connections in valid paths
                if forward_only and edge_key not in edges_in_path_type:
                    continue
                
                # Read values directly without modification for debugging
                weight_val = float(row['weight'])
                ratio_val = float(row['connection_ratio'])
                prob_val = float(row['traversal_probability'])
                
                # Check for unexpected values
                if ratio_val > 1.0 or ratio_val < 0.0:
                    print(f'\033[33mWarning: connection_ratio out of range [0,1]: {ratio_val} for {source}->{target}\033[0m')
                if prob_val > 1.0 or prob_val < 0.0:
                    print(f'\033[33mWarning: traversal_probability out of range [0,1]: {prob_val} for {source}->{target}\033[0m')
                
                edge_weight_type[edge_key] = weight_val
                edge_ratio_type[edge_key] = ratio_val  # Use raw value without clamping
                edge_prob_type[edge_key] = prob_val    # Use raw value without clamping
            
            if forward_only:
                print(f'Filtered to {len(edge_weight_type)} edges for visualization (forward_only=True)')
            
            if len(edge_weight_type) == 0:
                print('\033[33mWarning: No connections found in connection_type sheet for Sankey diagrams.\033[0m')
            else:
                # Build node list and track all layers each type appears in
                all_types_by_layer = {}
                type_all_layers = {}  # Track all layers for each neuron type
                for (layer_idx, source, target) in edge_weight_type.keys():
                    if layer_idx not in all_types_by_layer:
                        all_types_by_layer[layer_idx] = set()
                    all_types_by_layer[layer_idx].add(source)
                    if layer_idx + 1 not in all_types_by_layer:
                        all_types_by_layer[layer_idx + 1] = set()
                    all_types_by_layer[layer_idx + 1].add(target)
                    
                    # Track all layers for each type
                    type_all_layers.setdefault(source, set()).add(layer_idx)
                    type_all_layers.setdefault(target, set()).add(layer_idx + 1)
                
                # Create ordered node list with labels showing all layers
                node_type = []
                node_type_layers = []  # Primary layer for positioning
                node_labels = []  # Labels with all layers
                for layer_idx in sorted(all_types_by_layer.keys()):
                    layer_types = sorted(list(all_types_by_layer[layer_idx]))
                    for node in layer_types:
                        node_type.append(node)
                        node_type_layers.append(layer_idx)
                        # Create label showing all layers
                        all_layers = sorted(type_all_layers[node])
                        if len(all_layers) == 1:
                            node_labels.append(f"{node} (L{all_layers[0]})")
                        else:
                            layers_str = ','.join(map(str, all_layers))
                            node_labels.append(f"{node} (L{layers_str})")
                
                # Create node index mapping
                node_to_idx = {node: idx for idx, node in enumerate(node_type)}
                
                # Color nodes and create hover text (mark targets)
                node_type_color = [self.node_color] * len(node_type)
                node_hover_text = []
                for idx, node in enumerate(node_type):
                    all_layers = sorted(type_all_layers[node])
                    layers_display = ', '.join(map(str, all_layers))
                    if node in target_type:
                        node_type_color[idx] = self.target_color
                        node_hover_text.append(f"{node}<br>Layers: {layers_display}<br>(Target)")
                    else:
                        node_hover_text.append(f"{node}<br>Layers: {layers_display}")
                
                # Build link data for all three visualizations
                source_indices = []
                target_indices = []
                weights_for_links = []
                ratios_for_links = []
                probs_for_links = []
                
                for (layer_idx, source, target), weight in edge_weight_type.items():
                    source_indices.append(node_to_idx[source])
                    target_indices.append(node_to_idx[target])
                    weights_for_links.append(weight)
                    ratios_for_links.append(edge_ratio_type[(layer_idx, source, target)])
                    probs_for_links.append(edge_prob_type[(layer_idx, source, target)])
                
                # Debug: Print value ranges
                print(f"\nSankey value ranges:")
                print(f"  Weights: min={min(weights_for_links):.1f}, max={max(weights_for_links):.1f}")
                print(f"  Ratios: min={min(ratios_for_links):.4f}, max={max(ratios_for_links):.4f}")
                print(f"  Probs: min={min(probs_for_links):.4f}, max={max(probs_for_links):.4f}")
                
                # Visualization 1: Weight-based (synapse count)
                fig_type_weight = go.Figure(data=[go.Sankey(
                    node = dict(
                        pad = 5,
                        thickness = 5,
                        line = dict(color = "black", width = 0),
                        label = node_labels,
                        color = node_type_color,
                        customdata = node_hover_text,
                        hovertemplate = '%{customdata}<extra></extra>'
                    ),
                    link = dict(
                        source = source_indices,
                        target = target_indices,
                        value = weights_for_links,
                        color = self.link_color,
                        customdata = weights_for_links,
                        hovertemplate = '%{customdata:.1f} synapses<extra></extra>'
                    )
                )])
                fig_type_weight.update_layout(
                    title_text='Sankey diagram of all connections to targets<br>based on neuron type (by synapse count)',
                    font_size=12
                )
                fig_type_weight.write_html(os.path.join(self.allpath_folder,'Sankey_type_allpaths_snp.html'), auto_open=self.showfig, include_plotlyjs='cdn')
                
                # Visualization 2: Connection Ratio-based
                fig_type_ratio = go.Figure(data=[go.Sankey(
                    node = dict(
                        pad = 5,
                        thickness = 5,
                        line = dict(color = "black", width = 0),
                        label = node_labels,
                        color = node_type_color,
                        customdata = node_hover_text,
                        hovertemplate = '%{customdata}<extra></extra>'
                    ),
                    link = dict(
                        source = source_indices,
                        target = target_indices,
                        value = ratios_for_links,
                        color = self.link_color,
                        customdata = ratios_for_links,
                        hovertemplate = '%{customdata:.4f}<extra></extra>'
                    )
                )])
                fig_type_ratio.update_layout(
                    title_text='Sankey diagram of all connections to targets<br>based on neuron type (by connection ratio)',
                    font_size=12
                )
                fig_type_ratio.write_html(os.path.join(self.allpath_folder,'Sankey_type_allpaths_ratio.html'), auto_open=self.showfig, include_plotlyjs='cdn')
                
                # Visualization 3: Traversal Probability-based
                fig_type_prob = go.Figure(data=[go.Sankey(
                    node = dict(
                        pad = 5,
                        thickness = 5,
                        line = dict(color = "black", width = 0),
                        label = node_labels,
                        color = node_type_color,
                        customdata = node_hover_text,
                        hovertemplate = '%{customdata}<extra></extra>'
                    ),
                    link = dict(
                        source = source_indices,
                        target = target_indices,
                        value = probs_for_links,
                        color = self.link_color,
                        customdata = probs_for_links,
                        hovertemplate = '%{customdata:.4f}<extra></extra>'
                    )
                )])
                fig_type_prob.update_layout(
                    title_text='Sankey diagram of all connections to targets<br>based on neuron type (by traversal probability)',
                    font_size=12
                )
                fig_type_prob.write_html(os.path.join(self.allpath_folder,'Sankey_type_allpaths_prob.html'), auto_open=self.showfig, include_plotlyjs='cdn')
                
                print(f'Created 3 type-level Sankey diagrams with {len(node_type)} nodes and {len(weights_for_links)} edges')
        
        # Build bodyId-level Sankey from connection_info sheet (conn_inpath)
        # forward_only=True: Show only edges in path_bodyId (filtered)
        # forward_only=False: Show ALL connections in conn_inpath (complete graph)
        if find_bodyId_path and len(conn_inpath) > 0:
            # If forward_only=True, extract edges from path_bodyId for filtering
            edges_in_path_bodyId = set()
            if forward_only and 'path_df_bodyId' in locals() and len(path_df_bodyId) > 0:
                print('\nExtracting edges from path_bodyId for filtered visualization...')
                for idx in path_df_bodyId.index:
                    path_block = path_df_bodyId.at[idx, 'path_block']
                    edges = parse_path_to_edges(path_block)
                    for layer_idx, source_str, target_str in edges:
                        # Convert to int (path_block has strings)
                        edges_in_path_bodyId.add((layer_idx, int(source_str), int(target_str)))
                print(f'  Extracted {len(edges_in_path_bodyId)} unique edges from bodyId paths')
            
            # Extract all edges directly from conn_inpath DataFrame
            edge_weight_bodyId = {}
            edge_ratio_bodyId = {}
            edge_prob_bodyId = {}
            
            for idx, row in conn_inpath.iterrows():
                layer_label = row['conn_layer']
                layer_idx = int(layer_label.split('->')[0])
                source_id = int(row['bodyId_pre'])
                target_id = int(row['bodyId_post'])
                edge_key = (layer_idx, source_id, target_id)
                
                # If forward_only=True, only include edges that are in path_bodyId
                if forward_only and edge_key not in edges_in_path_bodyId:
                    continue
                
                # Aggregate if same edge appears multiple times (shouldn't happen but be safe)
                if edge_key in edge_weight_bodyId:
                    edge_weight_bodyId[edge_key] += float(row['weight'])
                    edge_ratio_bodyId[edge_key] = max(edge_ratio_bodyId[edge_key], float(row['connection_ratio']))
                    edge_prob_bodyId[edge_key] = max(edge_prob_bodyId[edge_key], float(row['traversal_probability']))
                else:
                    edge_weight_bodyId[edge_key] = float(row['weight'])
                    edge_ratio_bodyId[edge_key] = max(0.0, min(1.0, float(row['connection_ratio'])))
                    edge_prob_bodyId[edge_key] = max(0.0, min(1.0, float(row['traversal_probability'])))
            
            if forward_only:
                print(f'Filtered to {len(edge_weight_bodyId)} bodyId edges for visualization (forward_only=True)')
            
            if len(edge_weight_bodyId) == 0:
                print('\033[33mWarning: No connections found in connection_info sheet for bodyId Sankey diagrams.\033[0m')
            else:
                # Build node list by layer
                all_bodyIds_by_layer = {}
                for (layer_idx, source, target) in edge_weight_bodyId.keys():
                    if layer_idx not in all_bodyIds_by_layer:
                        all_bodyIds_by_layer[layer_idx] = set()
                    all_bodyIds_by_layer[layer_idx].add(source)
                    if layer_idx + 1 not in all_bodyIds_by_layer:
                        all_bodyIds_by_layer[layer_idx + 1] = set()
                    all_bodyIds_by_layer[layer_idx + 1].add(target)
                
                # Create ordered node list
                node_bodyId = []
                for layer_idx in sorted(all_bodyIds_by_layer.keys()):
                    layer_bodyIds = sorted(list(all_bodyIds_by_layer[layer_idx]))
                    node_bodyId.extend(layer_bodyIds)
                
                # Fetch neuron info for labels (use local dataset if available)
                node_df = self._fetch_neurons_local_or_api(node_bodyId, columns=['bodyId', 'type'])
                for ind in node_df.index:
                    if node_df.at[ind, 'type'] == None:
                        node_df.at[ind, 'type'] = 'None'
                
                bodyId_to_type = dict(zip(node_df['bodyId'], node_df['type']))
                node_bodyId_labels = [f"{bodyId_to_type.get(bid, 'Unknown')}_{bid}" for bid in node_bodyId]
                
                # Create node index mapping
                node_to_idx_bodyId = {node: idx for idx, node in enumerate(node_bodyId)}
                
                # Color nodes
                node_bodyId_color = [self.node_color] * len(node_bodyId)
                for idx, bodyId in enumerate(node_bodyId):
                    if bodyId in target_ID:
                        node_bodyId_color[idx] = self.target_color
                
                # Build links
                source_indices_bodyId = []
                target_indices_bodyId = []
                weights_bodyId = []
                
                for (layer_idx, source, target), weight in edge_weight_bodyId.items():
                    source_indices_bodyId.append(node_to_idx_bodyId[source])
                    target_indices_bodyId.append(node_to_idx_bodyId[target])
                    weights_bodyId.append(weight)
                
                # Create bodyId Sankey
                fig_bodyId = go.Figure(data=[go.Sankey(
                    node = dict(
                        pad = 1,
                        thickness = 5,
                        line = dict(color = "black", width = 0),
                        label = node_bodyId_labels,
                        color = node_bodyId_color
                    ),
                    link = dict(
                        source = source_indices_bodyId,
                        target = target_indices_bodyId,
                        value = weights_bodyId,
                        color = self.link_color
                    )
                )])
                fig_bodyId.update_layout(
                    title_text='Sankey diagram of all connections to targets<br>based on neuron bodyId',
                    font_size=6
                )
                fig_bodyId.write_html(os.path.join(self.allpath_folder,'Sankey_bodyId_allpaths.html'), auto_open=self.showfig, include_plotlyjs='cdn')
                
                print(f'Created bodyId-level Sankey diagram with {len(node_bodyId)} nodes and {len(weights_bodyId)} edges')
        
        # VisualizePath network visualization
        print('\nCreating interactive network visualizations...')
        try:
            
            # Create network from path_type if it exists
            if len(path_df_type) > 0:
                # Filter paths if pathN_to_show is specified
                paths_to_visualize = path_df_type
                if self.pathN_to_show > 0 and len(path_df_type) > self.pathN_to_show:
                    # Calculate path strength (product of traversal probabilities)
                    # Paths are already sorted by traversal_probability in sv.getAllPath()
                    # Just take the first N paths
                    paths_to_visualize = path_df_type.head(self.pathN_to_show).copy()
                    print(f'  Showing top {self.pathN_to_show} paths (by traversal_probability) out of {len(path_df_type)} total paths')
                else:
                    print(f'  Showing all {len(path_df_type)} paths')
                
                vp = VisualizePath(
                    path_file=paths_to_visualize,
                    output_folder=self.allpath_folder,
                    source_color=self.source_color if hasattr(self, 'source_color') else '#1f77b4',
                    intermediate_color=self.intermediate_color if hasattr(self, 'intermediate_color') else '#2ca02c',
                    target_color=self.target_color if hasattr(self, 'target_color') else '#d62728',
                    link_color=self.link_color if hasattr(self, 'link_color') else 'rgba(100,100,100,0.3)',
                    network_layout=self.network_layout if hasattr(self, 'network_layout') else 'hierarchical',
                    showfig=self.showfig
                )
                vp.visualize()
                print('  Created network_selected_paths.html and sankey_selected_paths.html')
            else:
                print('  No paths found to visualize')
        except Exception as e:
            print(f'  Warning: VisualizePath visualization failed: {e}')
            import traceback
            traceback.print_exc()
        
        # Create type-level heatmap visualization
        print('Creating type-level connection heatmap...')
        try:
            if len(conn_types) > 0:
                # Build connection matrix from conn_types
                # Group by type_pre and type_post, summing weights
                conn_matrix_data = conn_types.groupby(['type_pre', 'type_post'])['weight'].sum().reset_index()
                
                # Create matrix
                conn_matrix_type = conn_matrix_data.pivot(
                    index='type_pre', 
                    columns='type_post', 
                    values='weight'
                ).fillna(0)
                
                # Use CreateHeatmap class
                heatmap_gen = sv.CreateHeatmap(
                    output_folder=self.allpath_folder,
                    showfig=self.showfig
                )
                heatmap_gen.add_heatmap(
                    matrix=conn_matrix_type,
                    name='heatmap_allpaths_type',
                    title=f'Connection Heatmap: {self.source_fname} to {self.target_fname}<br>Type-level connections in all paths',
                    color_scale='purple',
                    interactive=True
                )
                heatmap_gen.create_all()
                print('  Created heatmap_allpaths_type.html')
            else:
                print('  No connections to visualize in heatmap')
        except Exception as e:
            print(f'  Warning: Heatmap visualization failed: {e}')
            import traceback
            traceback.print_exc()
        
        print('Done\n')
    
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
        print('Building interactive network by type...')
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
            showfig=showfig
        )
        
        return vp.visualize()

@dataclass
class VisualizeSkeleton:
    '''3-D visualize skeleton with synapses and brain roi meshes'''
    
    dataset: str = 'hemibrain:v1.2.1'
    '''dataset to use, default is hemibrain:v1.2.1'''

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

    synapse_size: int = 0
    '''
    size of synapse\n
    when synapse_mode='scatter', 1 to 10 is recommended\n
    when synapse_mode='sphere', 100 is recommended\n
    '''

    synapse_criteria: SynapseCriteria = None
    '''criteria to filter synapses'''

    synapse_mode: str = 'scatter'
    '''
    mode to plot synapses, 'scatter' or 'sphere' \n
    'scatter': plot synapses as scatter points, relative size to the view\n
    'sphere': plot synapses as spheres, absolute size in the figure \n
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

    mesh_color: tuple | list = (100, 100, 100, 0.2)
    '''
    color of brain meshes, single color or list of colors matching the length of mesh_roi
    single color: tuple including an alpha channel: (R, G, B, alpha)
    multiple colors: list of tuples, each tuple including an alpha channel: [(R1, G1, B1, alpha1), (R2, G2, B2, alpha2), ...]
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

    use_size_slider: bool = True
    '''
    whether to use size slider to adjust the size of synapses\n
    only works when synapse_mode='scatter'
    '''

    legend_mode: str = 'normal'
    '''
    'normal': show legend for individual neurons\n
    'merge': merge all neurons in the same layer and show legend for each layer\n
    '''
    
    brain_mesh: bool = 'none'
    ''' 
    brain_mesh = 'none', only plot the meshes in mesh_roi \n
    brain_mesh = 'whole', plot the whole brain mesh. Run flybrains.download_jrc_transforms() to download the transformations, see https://github.com/navis-org/navis-flybrains \n
    brain_mesh = 'hemi', plot the hemibrain mesh \n
    change the color of hemibrain mesh by brain_mesh_color parameter \n
    '''
    
    brain_mesh_color: str = 'rgba(200, 230, 240, 0.1)'
    ''' 
    color of the hemibrain mesh, only works when brain_mesh = 'whole' or 'hemi' \n
    e.g. 'rgba(200, 230, 240, 0.1)' \n
    see more at https://plotly.com/python/discrete-color/ \n
    '''

    def __post_init__(self):
        if self.synapse_mode not in ['scatter', 'sphere']:
            raise ValueError('synapse_mode can only be "scatter" or "sphere"')
        if self.legend_mode not in ['normal', 'merge']:
            raise ValueError('legend_mode can only be "normal" or "merge"')
        if self.skeleton_mode not in ['line','tube']:
            raise ValueError('skeleton_mode can only be "line" or "tube"')
        if self.brain_mesh not in ['none', 'whole', 'hemi']:
            raise ValueError('brain_mesh can only be "none", "whole" or "hemi"')
        
        # convert neuron_layers str to list, if is str
        if type(self.neuron_layers) is str:
            self.neuron_layers = self.neuron_layers.replace(' ','').split('->')
            for i,layer in enumerate(self.neuron_layers): # convert bodyId str to int
                if layer.isnumeric():
                    self.neuron_layers[i] = int(layer)
        
        if self.synapse_mode == 'scatter' and self.synapse_size == 0:
            self.synapse_size = 2
        elif self.synapse_mode == 'sphere':
            if self.synapse_size < 100 and self.brain_mesh != 'whole':
                self.synapse_size = 100
                print('\033[33mSynapse size is too small (< 100) for sphere mode, automatically reset to 100\033[0m')
            if self.use_size_slider:
                self.use_size_slider = False
                print('\033[33msize slider is not available for synapse_mode="sphere", automatically reset use_size_slider to False\033[0m')
            
        if self.mesh_roi == None:
            self.mesh_roi = []
        
        if len(self.neuron_layers) <= len(self.neuron_colors): 
            self.neuron_colors = self.neuron_colors[:len(self.neuron_layers)]
            self.synapse_colors = self.synapse_colors[:len(self.neuron_layers)-1]

        if self.skeleton_mode == 'line':
            self.show_skeleton_radius = False
            if self.neuron_alpha < 1:
                self.neuron_alpha = 1
                print('\033[33mneuron_alpha is not available for skeleton_mode="line", automatically reset to 1\033[0m')
        elif self.skeleton_mode == 'tube':
            self.show_skeleton_radius = True
        
        # fetch neurons and automatically generate layer names
        self.neuron_dfs = []
        self.roi_dfs = []
        self.layer_criteria = []
        self.layer_names = []
        for i in range(len(self.neuron_layers)):
            print(f'fetching neuron info of layer {i}...')
            ndf, rdf, auto_name, cri = sv.getNeurons(self.neuron_layers[i], dataset=self.dataset)
            self.neuron_dfs.append(ndf)
            self.roi_dfs.append(rdf)
            self.layer_criteria.append(cri)
            self.layer_names.append(auto_name)
        print('Fetched neuron layers')

        
        if self.custom_layer_names:
            self.layer_names = self.custom_layer_names
        if self.saveas is None:
            self.saveas = '_'.join(self.layer_names)
        self.save_folder = os.path.join(self.data_folder, 'plot3d_' + self.saveas.split('.')[0])
        if not os.path.exists(self.save_folder): os.makedirs(self.save_folder)
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
    
    def plot_skeleton(self):
        for i in range(len(self.neuron_layers)):
            print(f'fetching skeletons of layer {i}...')
            neuron_vols = neu.fetch_skeletons(self.neuron_dfs[i],with_synapses=self.show_connectors)
            if self.brain_mesh == 'whole':
                print(f'Transforming skeletons of layer {i}...', end='')
                try:
                    neuron_vols = navis.xform_brain(neuron_vols, source='JRCFIB2018Fraw', target='JRC2018F')
                except:
                    print('\033[33mTransforming skeletons failed. Please install transformations at first. Run flybrains.download_jrc_transforms() to download the transformations. See https://github.com/navis-org/navis-flybrains for more details\n\nbrain_mesh is automatically reset to "none"\n\033[0m')
                    self.brain_mesh = 'none'
            print('plotting...', end='')
            fig_layer = navis.plot3d(
                neuron_vols,
                backend='plotly',
                color=self.neuron_colors[i],
                alpha=self.neuron_alpha,
                soma=self.show_soma,
                # fig=self.fig_3d,
                radius=self.show_skeleton_radius,
                connectors=self.show_connectors,
            )
            fig_traces = fig_layer.data

            for j,trace in enumerate(fig_traces):
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

            print('Done')
        return 0
    
    def plot_synapses(self):
        file_path = os.path.join(self.save_folder, self.saveas+'_connection_info.xlsx')
        for i in range(len(self.neuron_layers)-1):
            source_criteria = self.layer_criteria[i]
            target_criteria = self.layer_criteria[i+1]
            print(f'\rfetching synapses of layer {i} -> layer {i+1}...')
            conn_df = fetch_synapse_connections(
                source_criteria=source_criteria,
                target_criteria=target_criteria,
                min_total_weight=self.min_synapse_num,
                synapse_criteria=self.synapse_criteria,
            )
            if i == 0:
                mode = 'w'
            else:
                mode = 'a'
            with pd.ExcelWriter(file_path, mode=mode, engine='openpyxl') as writer:
                conn_df.to_excel(writer, sheet_name=f'conn_df{i}_{i+1}')
            
            print('plotting...', end='')
            X = (conn_df['x_pre']+conn_df['x_post'])/2
            Y = (conn_df['y_pre']+conn_df['y_post'])/2
            Z = (conn_df['z_pre']+conn_df['z_post'])/2
            xyz_df = pd.DataFrame({'x':X, 'y':Y, 'z':Z})
            if self.brain_mesh == 'whole':
                print(f'Transforming synapses of layer {i} -> {i+1}...', end='')
                xyz_df = navis.xform_brain(xyz_df, source='JRCFIB2018Fraw', target='JRC2018F')
            if self.synapse_mode == 'scatter':
                sp = go.Scatter3d(
                    x = xyz_df['x'],
                    y = xyz_df['y'],
                    z = xyz_df['z'],
                    mode = 'markers',
                    name = f'synapses {i} -> {i+1} ({len(conn_df)})',
                    hoverinfo = 'name',
                    hovertemplate = 'x: %{x}<br>y: %{y}<br>z: %{z}<br>name: %{fullData.name}<extra></extra>',
                    legendgroup = f'synapses {i} -> {i+1} ({len(conn_df)})',
                    marker = dict(
                        size = self.synapse_size,
                        color = self.synapse_colors[i],
                        symbol = 'circle',
                    ),
                )
                self.fig_3d.add_trace(sp)
            elif self.synapse_mode == 'sphere':
                for ind in range(len(xyz_df)):
                    x = xyz_df['x'][ind]
                    y = xyz_df['y'][ind]
                    z = xyz_df['z'][ind]
                    sp = sv.build_sphere(x,y,z,r=self.synapse_size,color_scale=[self.synapse_colors[i]]*2,opacity=self.synapse_alpha)
                    sp.name = f'synapses {i} -> {i+1} ({len(conn_df)})'
                    sp.hoverinfo = 'name'
                    sp.legendgroup = f'synapses {i} -> {i+1} ({len(conn_df)})'
                    sp.hovertemplate = '<b>%{fullData.name}</b><extra></extra>'
                    if ind == 0: sp.showlegend = True
                    self.fig_3d.add_trace(sp)
            print('Done')
        return 0
    
    def plot_mesh(self):
        if self.mesh_roi is None:
            return
        roiunits = []
        for roi in self.mesh_roi:
            mesh_file = os.path.join(self.script_path, 'navis_roi_meshes_json','primary_rois',roi+'.json')
            if os.path.exists(mesh_file):
                mesh = navis.Volume.from_json(mesh_file)
                if self.brain_mesh == 'whole':
                    print(f'Transforming brain region {roi}...', end='')
                    mesh = navis.xform_brain(mesh, source='JRCFIB2018Fraw', target='JRC2018F')
                roiunits.append(mesh)
            else:
                print(f'mesh file {roi}.json not found!')
        print('plotting mesh of brain regions...')
        for roi_i in range(len(roiunits)):
            if type(self.mesh_color) == list:
                roiunits[roi_i].color = self.mesh_color[roi_i]
            else:
                roiunits[roi_i].color = self.mesh_color
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
                    trace.legendgroup = self.mesh_roi[roi_i]
                trace.hovertemplate = '<b>%{fullData.name}</b><extra></extra>'  # show full name in hover tooltip
                trace.hoverinfo = 'name'
                trace.name = 'brain regions [' + self.mesh_roi[roi_i] + '...]'
            self.fig_3d.add_traces(mesh_traces)
        if self.brain_mesh == 'hemi':
            print('plotting hemibrain mesh...')
            brain_meshes = flybrains.JRCFIB2018Fraw
            fig_hemi = navis.plot3d(brain_meshes,backend='plotly')
            hemi_traces = fig_hemi.data
            for trace in hemi_traces:
                trace.showlegend = True
                trace.name = 'hemibrain'
                trace.hoverinfo = 'none'
                trace.color = self.brain_mesh_color
            self.fig_3d.add_traces(hemi_traces)
        if self.brain_mesh == 'whole':
            print('plotting whole brain mesh...')
            brain_meshes = flybrains.JRC2018F
            fig_whole = navis.plot3d(brain_meshes,backend='plotly')
            whole_traces = fig_whole.data
            for trace in whole_traces:
                trace.showlegend = True
                trace.name = 'whole brain'
                trace.hoverinfo = 'none'
                trace.color = self.brain_mesh_color
            self.fig_3d.add_traces(whole_traces)
        print('Done')
        return 0
    
    def save_figure(self):
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
                eye=dict(x=0, y=1.4, z=0),
                center=dict(x=0, y=0, z=0),
            )
        elif self.brain_mesh == 'whole':
            scene_camera_parameters = dict(
                up=dict(x=0, y=-1, z=0),
                eye=dict(x=0, y=0, z=-1.0),
                center=dict(x=0, y=0, z=0),
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
        print(f'saving figure to \033[34m{self.fig_path}.html\033[0m...', end='')
        self.fig_3d.write_html(self.fig_path+'.html',auto_open=self.show_fig, include_plotlyjs='cdn')
        self.fig_3d.write_image(self.fig_path+'.png',scale=3)
        print('Done')
    
    def plot_neurons(self):
        self.plot_skeleton()
        self.plot_synapses()
        self.plot_mesh()
        self.save_figure()
        
    def export_video(self, fps=30, rotate_plane=None, view_direction = None, view_distance=None, synapse_size=1,**kwargs):
        '''
        export the rotating 3-D object to a video. rendering is slow, it helps to visualize complex objects and the video file is more portable and versatile.
        
        when file is too large, exporting may fail. try to reduce the resolution by setting "scale" in kwargs, or set "width" and "height" to specific values.
        
        fps: default 30
            frames per second, also determines the step size of rotation, 30 degrees per second.
        rotate_plane: default 'xy' for hemibrain, 'xz' for transformed whole brain mesh
            the plane to rotate the object. can be 'xy', 'xz', 'yz'.
        view_direction: default (1, 1) or (1, -1) depending on the brain mesh
            the direction of the camera. can be (1, 1), (1, -1), (-1, 1), (-1, -1).
        view_distance: default 1.6 or 2.2 depending on the brain mesh
            the relative distance between the camera and the center of the object.
        synapse_size: default 1,
            the size of the synapse markers.
        **kwargs: other arguments for plotly.offline.plot. see https://plotly.github.io/plotly.py-docs/generated/plotly.io.write_image.html
            In the kwargs, you can use "scale" to set the resolution of the video (e.g. scale=2 doubles the resolution), or set "width" and "height" to specific values.
            recommended values for scale: 2
        '''
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
        
        if kwargs.get('scale') is None and kwargs.get('width') is None and kwargs.get('height') is None:
            kwargs['scale'] = 2
        kwargs.update(kwargs)
        step = 30 / fps
        html_size = os.path.getsize(self.fig_path+'.html') / 1024 / 1024 # in MB
        if html_size > 100:
            print(f'\033[33mFigure is large. If rendering hangs, try to reduce the resolution by setting "scale", or "width" and "height" in kwargs to smaller values.\033[0m')
        # set layout
        fig_traces = self.fig_3d.data
        for trace in fig_traces:
            trace.showlegend = False
            if hasattr(trace,'marker'):
                trace.marker.size = synapse_size
        fig_layout = go.Layout(
            margin=dict(
                l=1,
                r=1,
                b=1,
                t=1,
                pad=0,
            ),
        )
        fig_new = go.Figure(data=fig_traces, layout=fig_layout)
        
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
            sliders=[], # remove sliders
            scene=dict(
                dragmode='orbit',
                xaxis={'visible':False}, 
                yaxis={'visible':False},
                zaxis={'visible':False},
            ),
            scene_camera=scene_camera_parameters,
        )
        
        pic_folder = os.path.join(self.save_folder,f'pics_{fps}fps_{rotate_plane}')
        if os.path.exists(pic_folder):
            shutil.rmtree(pic_folder)
        os.makedirs(pic_folder)
        if step > 0:
            steps_to_write = np.linspace(0,360,int(360/step),endpoint=False)
        elif step < 0:
            steps_to_write = np.linspace(360,0,int(360/step),endpoint=False)
        t0 = time.time()
        for i,deg in enumerate(steps_to_write):
            rad_i = np.deg2rad(deg)
            x = view_distance * np.sin(rad_i) * view_direction[0]
            y = view_distance * np.cos(rad_i) * view_direction[1]
            if rotate_plane == 'xy':
                fig_new.update_layout(scene_camera=dict(eye=dict(x=x, y=y, z=0)))
            elif rotate_plane == 'yz':
                fig_new.update_layout(scene_camera=dict(eye=dict(x=0, y=x, z=y)))
            elif rotate_plane == 'xz':
                fig_new.update_layout(scene_camera=dict(eye=dict(x=x, y=0, z=y)))
            fig_path = os.path.join(pic_folder,f'deg_{deg:.1f}.jpeg')
            fig_new.write_image(fig_path,**kwargs)
            cv2.waitKey(2000)
            ti = time.time()
            print(f'\rExporting image: {i+1}/{len(steps_to_write)}...Elapsed {ti-t0:.2f}s. Remaining {(ti-t0)/(i+1)*(len(steps_to_write)-i-1):.2f}s',end='    ')
        print('\nDone')
        imglist = os.listdir(pic_folder)
        img_eg = cv2.imread(os.path.join(pic_folder,imglist[0]))
        height, width, layers = img_eg.shape

        # forward video
        video_dir = os.path.join(self.save_folder,f'{self.saveas}_video_forward.mp4')
        out = cv2.VideoWriter(
            video_dir, cv2.VideoWriter_fourcc(*'mp4v'), fps, frameSize=(width,height))
        for i,deg in enumerate(steps_to_write):
            img = cv2.imread(os.path.join(pic_folder,f'deg_{deg:.1f}.jpeg'))
            out.write(img)
            print(f'\rwriting forward video: {i+1}/{len(steps_to_write)}...',end='  ')
        out.release()
        print('Done')
        # backward video
        video_dir = os.path.join(self.save_folder,f'{self.saveas}_video_backward.mp4')
        out = cv2.VideoWriter(
            video_dir, cv2.VideoWriter_fourcc(*'mp4v'), fps, frameSize=(width,height))
        for i,deg in enumerate(steps_to_write[::-1]):
            img = cv2.imread(os.path.join(pic_folder,f'deg_{deg:.1f}.jpeg'))
            out.write(img)
            print(f'\rwriting backward video: {i+1}/{len(steps_to_write)}...',end='  ')
        out.release()
        print('Done')
        return 0
