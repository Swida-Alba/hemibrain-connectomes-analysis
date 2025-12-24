import os
import json
from copy import copy
from types import SimpleNamespace
import warnings

# Suppress FutureWarning from neuprint about Series.__getitem__
warnings.filterwarnings("ignore", category=FutureWarning, module="neuprint")

import bokeh.palettes
import img2pdf
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import navis
import navis.interfaces.neuprint as neu
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly
import seaborn as sns
from neuprint import *
from tqdm import tqdm

# FlyWire client support removed

# ============================================================================
# In-Memory Cache for Neuron DataFrames
# ============================================================================
# Avoids repeated CSV reads when getNeurons() is called multiple times
# Structure: {dataset: {'neuron_df': DataFrame, 'roi_df': DataFrame}}
_NEURON_DF_CACHE = {}


def _get_cached_neuron_df(dataset: str, dataset_path_body: str):
    """
    Load neuron DataFrame from cache or disk.
    
    First call loads from disk and caches in memory.
    Subsequent calls return the cached DataFrame instantly.
    
    Prefers parquet format if available (faster ~10x), falls back to CSV.
    
    Args:
        dataset: Dataset identifier (normalized, e.g., 'hemibrain_v1_2_1')
        dataset_path_body: Path prefix for data files (without suffix)
    
    Returns:
        Tuple of (neuron_df, roi_count_df)
    """
    global _NEURON_DF_CACHE
    
    if dataset in _NEURON_DF_CACHE:
        # Return cached DataFrames
        cached = _NEURON_DF_CACHE[dataset]
        return cached['neuron_df'].copy(), cached['roi_df'].copy()
    
    # Load from disk - prefer parquet over CSV (faster ~10x)
    neuron_parquet = dataset_path_body + '_neuron_df.parquet'
    roi_parquet = dataset_path_body + '_roi_count_df.parquet'
    neuron_csv = dataset_path_body + '_neuron_df.csv'
    roi_csv = dataset_path_body + '_roi_count_df.csv'
    
    # Load neuron_df (prefer parquet)
    if os.path.exists(neuron_parquet):
        ndf = pd.read_parquet(neuron_parquet)
    else:
        ndf = pd.read_csv(neuron_csv, header=0, index_col=0, low_memory=False)
    
    # Load roi_count_df (prefer parquet)
    if os.path.exists(roi_parquet):
        rdf = pd.read_parquet(roi_parquet)
    else:
        rdf = pd.read_csv(roi_csv, header=0, index_col=0, low_memory=False)
    
    # Cache in memory
    _NEURON_DF_CACHE[dataset] = {
        'neuron_df': ndf,
        'roi_df': rdf
    }
    
    return ndf.copy(), rdf.copy()


def clear_neuron_cache(dataset: str = None):
    """
    Clear the neuron DataFrame cache.
    
    Args:
        dataset: Specific dataset to clear. If None, clears all.
    """
    global _NEURON_DF_CACHE
    
    if dataset is None:
        _NEURON_DF_CACHE.clear()
    elif dataset in _NEURON_DF_CACHE:
        del _NEURON_DF_CACHE[dataset]


class CreateHeatmap:
    """
    A class for creating and managing heatmap visualizations of connection matrices.
    
    This class provides a clean interface for generating multiple heatmaps with
    consistent styling and automatic color scale selection based on matrix type.
    
    Attributes
    ----------
    output_folder : str
        Directory where heatmap HTML files will be saved
    showfig : bool
        Whether to automatically open heatmaps in browser (default: False)
    default_fontsize : int
        Default font size for heatmap labels (default: 12)
    
    Examples
    --------
    >>> # Create heatmap generator
    >>> hm = CreateHeatmap(output_folder='./my_heatmaps', showfig=False)
    >>> 
    >>> # Add single heatmap
    >>> hm.add_heatmap(
    ...     matrix=conn_matrix_type,
    ...     name='connection_matrix_type',
    ...     title='Connection Matrix by Type',
    ...     color_scale='green'  # or custom [[0, 'white'], [1, 'green']]
    ... )
    >>> 
    >>> # Add multiple heatmaps at once
    >>> hm.add_heatmaps({
    ...     'conn_matrix': conn_matrix,
    ...     'ratio_matrix': ratio_matrix
    ... }, base_title='My Analysis')
    >>> 
    >>> # Generate all heatmaps
    >>> created_files = hm.create_all()
    """
    
    # Predefined color scales for different matrix types
    COLOR_SCALES = {
        'green': [[0, 'rgb(255,255,255)'], [1, 'rgb(14,83,13)']],
        'purple': [[0, 'rgb(255,255,255)'], [1, 'rgb(104,55,164)']],
        'orange': [[0, 'rgb(255,255,255)'], [1, 'rgb(204,102,0)']],
        'blue': [[0, 'rgb(255,255,255)'], [1, 'rgb(31,119,180)']],
        'red': [[0, 'rgb(255,255,255)'], [1, 'rgb(214,39,40)']],
    }
    
    def __init__(self, output_folder, showfig=False, default_fontsize=12):
        """
        Initialize CreateHeatmap instance.
        
        Parameters
        ----------
        output_folder : str
            Directory to save heatmap HTML files
        showfig : bool, optional
            Whether to auto-open heatmaps in browser (default: False)
        default_fontsize : int, optional
            Default font size for labels (default: 12)
        """
        self.output_folder = output_folder
        self.showfig = showfig
        self.default_fontsize = default_fontsize
        self.heatmaps = []
        
        # Create output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    
    def add_heatmap(self, matrix, name, title=None, color_scale='purple', fontsize=None, scale='linear', interactive=False, conn_df=None):
        """
        Add a single heatmap to the generation queue.
        
        Parameters
        ----------
        matrix : pd.DataFrame
            Connection matrix to visualize
        name : str
            Base filename (without .html extension)
        title : str, optional
            Title to display on heatmap. If None, uses name.
        color_scale : str or list, optional
            Either a preset name ('green', 'purple', 'orange', 'blue', 'red')
            or a custom Plotly color scale [[0, 'color1'], [1, 'color2']]
        fontsize : int, optional
            Font size for this heatmap. If None, uses default_fontsize.
        scale : str, optional
            Scale for color mapping: 'linear', 'log2', or 'log10' (default: 'linear')
            Only used if interactive=False
        interactive : bool, optional
            If True, creates interactive heatmap with scale controls (default: False)
        conn_df : pd.DataFrame, optional
            Connection dataframe with type information for enhanced hover labels
        
        Returns
        -------
        self
            Returns self for method chaining
        """
        if title is None:
            title = name.replace('_', ' ').title()
        
        if fontsize is None:
            fontsize = self.default_fontsize
        
        # Resolve color scale
        if isinstance(color_scale, str):
            if color_scale in self.COLOR_SCALES:
                color_scale = self.COLOR_SCALES[color_scale]
            else:
                # Default to purple if unknown preset
                color_scale = self.COLOR_SCALES['purple']
        
        self.heatmaps.append({
            'matrix': matrix,
            'name': name,
            'title': title,
            'color_scale': color_scale,
            'fontsize': fontsize,
            'scale': scale,
            'interactive': interactive,
            'conn_df': conn_df
        })
        
        return self  # Allow method chaining
    
    def add_heatmaps(self, matrices_dict, titles_dict=None, color_scales_dict=None, fontsize=None, scale='linear', interactive=False):
        """
        Add multiple heatmaps at once from dictionaries.
        
        Parameters
        ----------
        matrices_dict : dict
            Dictionary of {name: matrix_dataframe}
        titles_dict : dict, optional
            Dictionary of {name: title_string}. If None, auto-generates titles.
        color_scales_dict : dict, optional
            Dictionary of {name: color_scale}. If None, auto-detects based on name.
        fontsize : int, optional
            Font size for all heatmaps. If None, uses default_fontsize.
        scale : str, optional
            Scale for color mapping: 'linear', 'log2', or 'log10' (default: 'linear')
        interactive : bool, optional
            If True, creates interactive heatmaps with scale controls (default: False)
        
        Returns
        -------
        self
            Returns self for method chaining
        
        Examples
        --------
        >>> matrices = {
        ...     'conn_matrix_type': conn_mat,
        ...     'ratio_matrix_type': ratio_mat
        ... }
        >>> hm.add_heatmaps(matrices)
        """
        if titles_dict is None:
            titles_dict = {}
        if color_scales_dict is None:
            color_scales_dict = {}
        
        for name, matrix in matrices_dict.items():
            # Get or generate title
            title = titles_dict.get(name, name.replace('_', ' ').title())
            
            # Get or auto-detect color scale
            if name in color_scales_dict:
                color_scale = color_scales_dict[name]
            else:
                # Auto-detect based on name
                if 'ratio' in name.lower():
                    color_scale = 'orange'
                elif 'transmission' in name.lower() or 'prob' in name.lower():
                    color_scale = 'purple'
                elif 'bodyid' in name.lower():
                    color_scale = 'green'
                else:
                    color_scale = 'purple'
            
            self.add_heatmap(matrix, name, title, color_scale, fontsize, scale, interactive)
        
        return self
    
    def create_all(self):
        """
        Generate all queued heatmaps.
        
        Returns
        -------
        list
            List of created file paths
        """
        if not self.heatmaps:
            print('No heatmaps to create.')
            return []
        
        print(f'Creating {len(self.heatmaps)} heatmap(s)...')
        created_files = []
        
        for hm in self.heatmaps:
            filename = os.path.join(self.output_folder, f"{hm['name']}.html")
            
            # Use interactive version if requested
            if hm.get('interactive', False):
                VisConnMatInteractive(
                    hm['matrix'],
                    filename=filename,
                    title=hm['title'],
                    color_scale=hm['color_scale'],
                    showfig=self.showfig,
                    fontsize=hm['fontsize'],
                    conn_df=hm.get('conn_df')
                )
            else:
                VisConnMat(
                    hm['matrix'],
                    filename=filename,
                    title=hm['title'],
                    color_scale=hm['color_scale'],
                    showfig=self.showfig,
                    fontsize=hm['fontsize'],
                    scale=hm['scale']
            )
            
            created_files.append(filename)
            print(f"  Created: {hm['name']}.html")
        
        print('Done\n')
        self.heatmaps = []  # Clear queue after creation
        return created_files
    
    def clear(self):
        """Clear the heatmap queue without creating them."""
        self.heatmaps = []
        return self


def LogInHemibrain(token,dataset='hemibrain:v1.2.1'): # log in to hemibrain dataset
    '''
    Log in to hemibrain dataset;
    Please provide your own token, which can be obtained from https://neuprint.janelia.org/account
    '''
    client = Client(
        'neuprint.janelia.org',
        dataset = dataset,
        token = token,
    )
    print("Logged in \ndataset: " + dataset)
    return client, dataset

def getCriteriaAndName(requiredNeurons):
    from neuprint import NeuronCriteria as NC
    if requiredNeurons == None:
        criteria = None
        fname = 'ALL'
    elif type(requiredNeurons) != list:
        raise ValueError('requiredNeurons must be a list or None')
    elif type(requiredNeurons[0]) == int: # bodyId
        criteria = NC(bodyId=requiredNeurons)
        fname = str(requiredNeurons[0])
    elif requiredNeurons[0].find('.*') != -1: # instance
        criteria = NC(instance=requiredNeurons)
        fname = requiredNeurons[0].replace('.*','')
    else: # type
        criteria = NC(type=requiredNeurons)
        fname = requiredNeurons[0]
    if requiredNeurons != None and len(requiredNeurons) > 1:
        fname += '_etc'
    return criteria, fname

def pull_dataset(dataset, save_path=None, omitNoneType=False, client=None):
    # requires login to hemibrain dataset
    if save_path is None:
        # Go up from src/ to project root, then into datasets/
        dataset_normalized = dataset.replace(':','_').replace('.','_')
        project_root = os.path.dirname(os.path.dirname(__file__))
        dataset_dir = os.path.join(project_root, "datasets", dataset_normalized)
        
        # Use new structure if directory exists, otherwise fallback (or create new structure)
        if os.path.exists(dataset_dir):
            save_path = os.path.join(dataset_dir, f"{dataset_normalized}_allneurons")
        else:
            # Create new structure by default
            os.makedirs(dataset_dir, exist_ok=True)
            save_path = os.path.join(dataset_dir, f"{dataset_normalized}_allneurons")
            
    neuron_df, roi_count_df = fetch_neurons(None, client=client)
    if omitNoneType:
        # delete rows with type is empty
        neuron_df = neuron_df[neuron_df['type'].notna()]
    print(f'Pulled {len(neuron_df)} neurons from {dataset}')
    print('Writing to',save_path, end='...')
    # write to csv
    neuron_df.to_csv(save_path + '_neuron_df.csv',index=True)
    roi_count_df.to_csv(save_path + '_roi_count_df.csv',index=True)
    print('Done!')

def getNeurons(requiredNeurons, dataset='hemibrain:v1.2.1', custom_group_names=None, client=None, verbose=True):
    '''get neurons locally from a given dataset
    
    Parameters
    ----------
    requiredNeurons : list or None
        List of neuron identifiers (types, instances, bodyIds).
        Supports nested lists for custom grouping: e.g., ['A', 'B', ['C', 'D']]
        Nested lists will be merged into a single custom group.
    dataset : str
        Dataset name
    custom_group_names : list, optional
        Custom names for groups when using nested lists
    client : object, optional
        Client object (NeuPrint or FlyWire) for direct fetching if local dataset missing
    verbose : bool, optional
        Whether to print progress messages (default: True)
        
    Returns
    -------
    neuron_df : pd.DataFrame
        DataFrame of neurons with 'custom_group' column for nested list groups
    roi_count_df : pd.DataFrame
        ROI count DataFrame
    auto_name : str
        Auto-generated name for the neuron set
    criteria : NeuronCriteria
        Neuprint criteria object
    '''
    from neuprint import NeuronCriteria as NC
    
    # Special handling for FlyWire/FAFB/BANC
    if 'flywire' in dataset.lower() or 'fafb' in dataset.lower() or 'banc' in dataset.lower():
        # Try to use local data first
        try:
            import fafb_utils
            # Go up from src/ to project root, then into datasets/
            project_root = os.path.dirname(os.path.dirname(__file__))
            
            # Try to find dataset directory by name
            data_dir = os.path.join(project_root, "datasets", dataset)
            if not os.path.exists(data_dir):
                # Fallback to default FAFB directory
                data_dir = os.path.join(project_root, "datasets", "flywire_FAFB_v783")
            
            if os.path.exists(data_dir):
                # Use dataset name in message instead of hardcoded "FAFB"
                dataset_short = dataset.split('_')[1] if '_' in dataset else dataset
                if verbose:
                    print(f"Checking local {dataset_short} data in {data_dir}...")
                neuron_file, _ = fafb_utils.prepare_fafb_data(data_dir)
                
                # Load the full neuron DataFrame
                if verbose:
                    print(f"Loading neurons from {neuron_file}...")
                full_neuron_df = pd.read_csv(neuron_file, dtype={'bodyId': str})
                
                if requiredNeurons is None:
                    return full_neuron_df, pd.DataFrame(), 'ALL_FAFB', None
                
                # Filter based on requiredNeurons
                selected_dfs = []
                
                # Handle nested lists (custom groups)
                # Structure: [item1, item2, [group_item1, group_item2]]
                
                # Flatten for simple filtering first
                flat_list = []
                custom_groups = {} # Map bodyId/type -> group_name
                
                group_idx = 0
                for i, item in enumerate(requiredNeurons):
                    if isinstance(item, list):
                        # It's a custom group
                        group_name = custom_group_names[group_idx] if custom_group_names and group_idx < len(custom_group_names) else f"Group_{group_idx+1}"
                        group_idx += 1
                        for subitem in item:
                            flat_list.append(subitem)
                            custom_groups[str(subitem)] = group_name
                    else:
                        flat_list.append(item)
                
                # Filter logic
                # Check if items are bodyIds (digits) or types (strings)
                bodyIds = [str(x) for x in flat_list if str(x).isdigit()]
                types = [str(x) for x in flat_list if not str(x).isdigit()]
                
                filtered_df = pd.DataFrame()
                
                if bodyIds:
                    df_by_id = full_neuron_df[full_neuron_df['bodyId'].isin(bodyIds)].copy()
                    selected_dfs.append(df_by_id)
                    
                if types:
                    # Regex matching for types
                    for t in types:
                        # Escape regex special characters if it's a literal type
                        # But allow regex if intended (e.g. "MBON.*")
                        # For now, assume simple contains or exact match if no regex chars
                        if any(c in t for c in ['.', '*', '+', '?', '^', '$', '(', ')', '[', ']', '{', '}', '|', '\\']):
                             df_by_type = full_neuron_df[full_neuron_df['type'].str.match(t, na=False)].copy()
                        else:
                             df_by_type = full_neuron_df[full_neuron_df['type'] == t].copy()
                        selected_dfs.append(df_by_type)
                
                if selected_dfs:
                    filtered_df = pd.concat(selected_dfs).drop_duplicates(subset=['bodyId'])
                    
                    # Apply custom groups
                    if custom_groups:
                        filtered_df['custom_group'] = filtered_df.apply(
                            lambda row: custom_groups.get(str(row['bodyId'])) or custom_groups.get(str(row['type'])), axis=1
                        )
                    
                    # Generate auto_name based on requiredNeurons
                    if len(requiredNeurons) == 1:
                        if isinstance(requiredNeurons[0], list):
                             # Single group
                             items_str = [str(x).replace('.*', '') for x in requiredNeurons[0]]
                             auto_name = items_str[0] + '_etc' if len(items_str) > 1 else items_str[0]
                        else:
                             auto_name = str(requiredNeurons[0]).replace('.*', '')
                    elif len(requiredNeurons) > 1:
                        first_item = requiredNeurons[0]
                        if isinstance(first_item, list):
                             items_str = [str(x).replace('.*', '') for x in first_item]
                             first_name = items_str[0] + '_etc' if len(items_str) > 1 else items_str[0]
                        else:
                             first_name = str(first_item).replace('.*', '')
                        auto_name = first_name + '_etc'
                    else:
                        auto_name = "fafb_selection"

                    return filtered_df, pd.DataFrame(), auto_name, None
                else:
                    print("Warning: No neurons found matching criteria in local FAFB data.")
                    return pd.DataFrame(), pd.DataFrame(), "empty", None
                    
        except ImportError:
            print("Warning: fafb_utils not found.")
        except Exception as e:
            print(f"Warning: Error loading local FAFB data: {e}.")

        print("Warning: FlyWire API fetching has been removed. Please ensure local data is available.")
        return pd.DataFrame(), pd.DataFrame(), "error", None

    if requiredNeurons == None:
        criteria = None
        auto_name = 'ALL'
        neuron_df, roi_count_df = fetch_neurons(criteria, client=client)
        return neuron_df, roi_count_df, auto_name, criteria
    if type(requiredNeurons) != list:
        requiredNeurons = [requiredNeurons]
    
    # Go up from src/ to project root, then into datasets/
    dataset_normalized = dataset.replace(':','_').replace('.','_')
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    # Check if dataset is in a subdirectory (new structure)
    dataset_dir = os.path.join(project_root, "datasets", dataset_normalized)
    if os.path.exists(dataset_dir):
        dataset_path_body = os.path.join(dataset_dir, f"{dataset_normalized}_allneurons")
    else:
        # Fallback to old structure (flat in datasets/)
        dataset_path_body = os.path.join(project_root, "datasets", f"{dataset_normalized}_allneurons")

    if not os.path.exists(dataset_path_body + '_neuron_df.csv') or not os.path.exists(dataset_path_body + '_roi_count_df.csv'):
        print(f'\033[33mcsv files of dataset "{dataset}" not found, downloading...\033[0m')
        # If using new structure, ensure directory exists before downloading
        if os.path.exists(dataset_dir):
             pull_dataset(dataset, save_path=dataset_path_body, omitNoneType=False, client=client)
        else:
             # If directory doesn't exist, pull_dataset might create files in root or fail if it expects dir
             # Let's assume pull_dataset handles path creation or we should create it
             os.makedirs(dataset_dir, exist_ok=True)
             dataset_path_body = os.path.join(dataset_dir, f"{dataset_normalized}_allneurons")
             pull_dataset(dataset, save_path=dataset_path_body, omitNoneType=False, client=client)
        # After download, clear any stale cache for this dataset
        clear_neuron_cache(dataset_normalized)

    # Use in-memory cache for neuron DataFrames (avoids repeated CSV reads)
    ndf_alltypes, rdf_alltypes = _get_cached_neuron_df(dataset_normalized, dataset_path_body)
    bodyId_alltypes = ndf_alltypes['bodyId'].tolist()
    
    if len(requiredNeurons) == 0:
        neuron_df = ndf_alltypes
        roi_count_df = rdf_alltypes
        auto_name = 'allneurons'
        bodyId_list = neuron_df['bodyId'].tolist()
    else:
        # Check if we have nested lists for custom grouping
        has_nested = any(isinstance(item, list) for item in requiredNeurons)
        
        if has_nested:
            # Process with custom grouping
            bodyId_list = []
            group_names = []
            group_custom_idx = 0
            
            for i, requiredNeuron in enumerate(requiredNeurons):
                if isinstance(requiredNeuron, list):
                    # Nested list - create custom group
                    group_bodyIds = []
                    group_items = []
                    
                    for item in requiredNeuron:
                        group_items.append(str(item).replace('.*', ''))
                        item_bodyIds = _process_single_neuron(item, ndf_alltypes, bodyId_alltypes, verbose=verbose)
                        group_bodyIds.extend(item_bodyIds)
                    
                    # Generate group name
                    if custom_group_names and group_custom_idx < len(custom_group_names):
                        group_name = custom_group_names[group_custom_idx]
                    else:
                        # Auto-generate name from first item
                        if len(group_items) == 1:
                            group_name = group_items[0]
                        else:
                            group_name = group_items[0] + '_etc'
                    
                    group_names.append(group_name)
                    bodyId_list.extend(group_bodyIds)
                    group_custom_idx += 1
                    
                    print(f'Custom group "{group_name}": {len(group_bodyIds)} neurons from {len(requiredNeuron)} items')
                else:
                    # Regular item
                    item_bodyIds = _process_single_neuron(requiredNeuron, ndf_alltypes, bodyId_alltypes, verbose=verbose)
                    bodyId_list.extend(item_bodyIds)
                    group_names.append(str(requiredNeuron).replace('.*', ''))
            
            # Create auto_name from group names
            if len(group_names) == 1:
                auto_name = group_names[0]
            elif len(group_names) == 2:
                auto_name = '_'.join(group_names)
            else:
                auto_name = group_names[0] + '_etc'
            
            # Build neuron_df with custom_group column
            neuron_df = ndf_alltypes[ndf_alltypes['bodyId'].isin(bodyId_list)].copy()
            roi_count_df = rdf_alltypes[rdf_alltypes['bodyId'].isin(bodyId_list)]
            
            # Add custom_group column by matching original type or creating merged type
            neuron_df['custom_group'] = neuron_df['type']  # Default to original type
            
            # Reassign custom groups for nested list items
            group_custom_idx = 0
            for i, requiredNeuron in enumerate(requiredNeurons):
                if isinstance(requiredNeuron, list):
                    # Get bodyIds for this custom group
                    group_bodyIds = []
                    for item in requiredNeuron:
                        item_bodyIds = _process_single_neuron(item, ndf_alltypes, bodyId_alltypes, verbose=verbose)
                        group_bodyIds.extend(item_bodyIds)
                    
                    # Assign custom group name
                    if custom_group_names and group_custom_idx < len(custom_group_names):
                        group_name = custom_group_names[group_custom_idx]
                    else:
                        items_str = [str(item).replace('.*', '') for item in requiredNeuron]
                        group_name = items_str[0] + '_etc' if len(items_str) > 1 else items_str[0]
                    
                    neuron_df.loc[neuron_df['bodyId'].isin(group_bodyIds), 'custom_group'] = group_name
                    group_custom_idx += 1
            
        else:
            # Original logic for flat list
            bodyId_list = []
            for i, requiredNeuron in enumerate(requiredNeurons):
                if i == 0: 
                    auto_name = str(requiredNeuron).replace('.*','')
                elif i == 1:
                    auto_name += '_etc'
                
                item_bodyIds = _process_single_neuron(requiredNeuron, ndf_alltypes, bodyId_alltypes, verbose=verbose)
                bodyId_list.extend(item_bodyIds)
            
            neuron_df = ndf_alltypes[ndf_alltypes['bodyId'].isin(bodyId_list)]
            roi_count_df = rdf_alltypes[rdf_alltypes['bodyId'].isin(bodyId_list)]
    
    criteria = NC(bodyId=bodyId_list)
    return neuron_df, roi_count_df, auto_name, criteria

def _process_single_neuron(requiredNeuron, ndf_alltypes, bodyId_alltypes, verbose=True):
    '''Helper function to process a single neuron identifier and return bodyIds
    
    Parameters
    ----------
    verbose : bool
        Whether to print progress messages
    '''
    bodyId_list = []
    
    # Check if it's a numeric bodyId (int, np.int64, or numeric string)
    is_bodyid = False
    bodyid_value = None
    
    if isinstance(requiredNeuron, (int, np.integer)):
        # Native int or numpy integer types (np.int64, etc.)
        is_bodyid = True
        bodyid_value = int(requiredNeuron)
    elif isinstance(requiredNeuron, str) and requiredNeuron.isdigit():
        # String that looks like a number (e.g., "535898")
        is_bodyid = True
        bodyid_value = int(requiredNeuron)
    
    if is_bodyid:
        # bodyId lookup
        if bodyid_value in bodyId_alltypes:
            bodyId_list.append(bodyid_value)
        else:
            if verbose:
                print(f'\033[33mbodyId {bodyid_value} not found, please check your input (skipped)\033[0m')
    elif isinstance(requiredNeuron, str) and requiredNeuron.find('.*') != -1:
        # regex of instance
        find_df = ndf_alltypes[ndf_alltypes['instance'].str.match(requiredNeuron, na=False)]
        if len(find_df) > 0:
            bodyId_list = find_df['bodyId'].tolist()
            if verbose:
                print(f'Found {len(find_df)} neurons of instance "{requiredNeuron}"')
        else:
            if verbose:
                print(f'\033[33minstance "{requiredNeuron}" not found, please check your input (skipped)\033[0m')
    else:
        # type
        find_df = ndf_alltypes[ndf_alltypes['type']==requiredNeuron]
        if len(find_df) > 0:
            bodyId_list = find_df['bodyId'].tolist()
            if verbose:
                print(f'Found {len(find_df)} neurons of type "{requiredNeuron}"')
        else:
            if verbose:
                print(f'\033[33mtype "{requiredNeuron}" not found, please check your input (skipped)\033[0m')
    
    return bodyId_list

def removeSearchedNeurons(conn_df,searchedNeurons,exempt_neurons=None):
    '''remove neurons on searched layers, except those in exempt_neurons'''
    neurons_post = conn_df['bodyId_post'].unique()
    
    # Identify neurons to remove: those in searchedNeurons
    to_remove = np.intersect1d(neurons_post, searchedNeurons, assume_unique=True)
    
    # If exempt_neurons provided, keep them even if they are in searchedNeurons
    if exempt_neurons is not None and len(exempt_neurons) > 0:
        # Remove exempt neurons from the to_remove list
        to_remove = np.setdiff1d(to_remove, exempt_neurons, assume_unique=True)
        
    df = conn_df[~conn_df['bodyId_post'].isin(to_remove)]
    return df

def Conn2FullMat(source_df,target_df,conn_df,conn_type,weight_col='weight'): 
    '''convert connection table (conn_df) to a full connection matrix (keep zero connections)'''
    # Append type to bodyId for row/column names if available
    if 'type' in source_df.columns:
        sbodyId = [f"{row.bodyId}_{row.type}" for _, row in source_df.iterrows()]
    else:
        sbodyId = source_df.bodyId.tolist()
        
    if 'type' in target_df.columns:
        tbodyId = [f"{row.bodyId}_{row.type}" for _, row in target_df.iterrows()]
    else:
        tbodyId = target_df.bodyId.tolist()
    stype = source_df.type.unique().tolist()
    ttype = target_df.type.unique().tolist()
    sbodyId.sort()
    tbodyId.sort()
    stype.sort()
    ttype.sort()
    # Convert bodyId to strings to ensure consistent treatment as labels, not numbers
    sbodyId = [str(x) for x in sbodyId]
    tbodyId = [str(x) for x in tbodyId]
    cmat_bodyId = pd.DataFrame(data=np.zeros([len(sbodyId),len(tbodyId)],dtype=int),index=sbodyId,columns=tbodyId)
    cmat_type = pd.DataFrame(data=np.zeros([len(stype),len(ttype)],dtype=int),index=stype,columns=ttype)
    # Create mappings for bodyId lookup
    source_map = {}
    if 'type' in source_df.columns:
        for _, row in source_df.iterrows():
            source_map[str(row.bodyId)] = f"{row.bodyId}_{row.type}"
    else:
        for _, row in source_df.iterrows():
            source_map[str(row.bodyId)] = str(row.bodyId)
            
    target_map = {}
    if 'type' in target_df.columns:
        for _, row in target_df.iterrows():
            target_map[str(row.bodyId)] = f"{row.bodyId}_{row.type}"
    else:
        for _, row in target_df.iterrows():
            target_map[str(row.bodyId)] = str(row.bodyId)

    for i in conn_df.index:
        raw_pre = str(conn_df.at[i,'bodyId_pre'])
        raw_post = str(conn_df.at[i,'bodyId_post'])
        
        bpre = source_map.get(raw_pre, raw_pre)
        bpost = target_map.get(raw_post, raw_post)
        
        if bpre in cmat_bodyId.index and bpost in cmat_bodyId.columns:
            bweight = conn_df.at[i,weight_col]
            cmat_bodyId.at[bpre,bpost] = bweight
    for i in conn_type.index:
        tpre  = conn_type.at[i,'type_pre']
        tpost = conn_type.at[i,'type_post']
        tweight = conn_type.at[i,weight_col]
        cmat_type.at[tpre,tpost] = tweight
    return cmat_bodyId,cmat_type

def calRC(cmat,threshold=0):
    '''calculate row and column sums of a connection matrix'''
    n_row,n_col = cmat.shape
    sourceN = [0]*n_col 
    targetN = [0]*n_row
    sum_col = [0]*n_col
    sum_row = [0]*n_row
    for i in range(n_row):
        for j in range(n_col):
            val = cmat.iat[i,j]
            sum_row[i] += val
            sum_col[j] += val
            if val > threshold:
                targetN[i] += 1
                sourceN[j] += 1
    cmat_new = pd.DataFrame(np.insert(cmat.values, len(cmat.index), values=sourceN, axis=0))
    cmat_new = pd.DataFrame(np.insert(cmat_new.values, len(cmat_new.index), values=sum_col, axis=0))
    cmat_new.columns = cmat.columns
    cmat_new.index = list(cmat.index) + ['sourceN','sum_col']
    cmat_new.insert(loc=len(cmat.columns),column='targetN',value=targetN+[0,0])
    cmat_new.insert(loc=len(cmat.columns)+1,column='sum_row',value=sum_row+[0,sum(sum_row)])
    return cmat_new

def filtMat(cmat,axis=0,filt_range=[0,1],by='MR'): 
    '''identify columns whose maximums are in the range'''
    if by == 'MR': # maximum ratio
        nval = cmat.shape # nval = (n_row, n_col)
        criterion = [1]*nval[1-axis]
        maxVal = cmat.max(axis=axis)
        if filt_range[0] != filt_range[1]:
            for j in range(nval[1-axis]):
                if maxVal[j] <= filt_range[0] or maxVal[j] > filt_range[1]: # left open, right closed interval
                    criterion[j] = 0
        else:
            for j in range(nval[1-axis]):
                if maxVal[j] != filt_range[0]: # left open, right closed interval
                    criterion[j] = 0
        if axis == 0:
            cmat_new = pd.DataFrame(np.insert(cmat.values, len(cmat.index), values=criterion, axis=0))
            cmat_new.index = list(cmat.index) + ['sourceCriterion']
            cmat_new.columns = cmat.columns
            cmat_new = cmat_new.loc[:,cmat_new.loc['sourceCriterion'] == 1]
            cmat_new = cmat_new.iloc[:-1,:]
        elif axis == 1:
            cmat_new = cmat.copy()
            cmat_new.insert(loc=len(cmat.columns), column='targetCriterion', value=criterion)
            cmat_new = cmat_new.loc[cmat_new['targetCriterion'] == 1,:]
            cmat_new = cmat_new.iloc[:,:-1]
    elif by == 'N': # synapse number
        cmat_t = calRC(cmat) # new connection matrix
        if axis == 0:
            if filt_range[0] != None and filt_range[1] != None:
                cmat_t = cmat_t.loc[:,cmat_t.loc['sourceN'] >= filt_range[0]]
                cmat_t = cmat_t.loc[:,cmat_t.loc['sourceN'] <= filt_range[1]]
            elif filt_range[0] == None:
                cmat_t = cmat_t.loc[:,cmat_t.loc['sourceN'] <= filt_range[1]]
            elif filt_range[1] == None:
                cmat_t = cmat_t.loc[:,cmat_t.loc['sourceN'] >= filt_range[0]]
            cmat_t = cmat_t.iloc[:-2,:]
        elif axis == 1:
            if filt_range[0] != None and filt_range[1] != None:
                cmat_t = cmat_t.loc[cmat_t['targetN'] >= filt_range[0],:]
                cmat_t = cmat_t.loc[cmat_t['targetN'] <= filt_range[1],:]
            elif filt_range[0] == None:
                cmat_t = cmat_t.loc[cmat_t['targetN'] <= filt_range[1],:]
            elif filt_range[1] == None:
                cmat_t = cmat_t.loc[cmat_t['targetN'] >= filt_range[0],:]
            cmat_t = cmat_t.iloc[:,:-2]
        cmat_new = cmat_t
    return cmat_new

def stMat(mat,axis=0):
    '''standardize matrix by row or column'''
    matt = calRC(mat)
    rowN,colN = matt.shape
    if axis == 0: # standardize by column
        for i in range(rowN-2):
            for j in range(colN-2):
                matt.iat[i,j] /= matt.iat[-1,j]
    elif axis == 1: # standardize by row
        for i in range(rowN-2):
            for j in range(colN-2):
                matt.iat[i,j] /= matt.iat[i,-1]
    return matt.iloc[:-2,:-2]

def VisConnMat(cmat,filename,title='',color_scale=[[0, 'rgb(255,255,255)'], [1, 'rgb(104,55,164)']],showfig=True,fontsize=12,scale='linear'): 
    '''visualize connection matrix with enhanced labels and hover information
    
    Parameters
    ----------
    cmat : pd.DataFrame
        Connection matrix to visualize
    filename : str
        Output HTML filename
    title : str, optional
        Title for the heatmap
    color_scale : list, optional
        Plotly color scale
    showfig : bool, optional
        Whether to open in browser
    fontsize : int, optional
        Font size for labels
    scale : str, optional
        Scale for color mapping: 'linear', 'log2', or 'log10' (default: 'linear')
    '''
    
    # Determine the metric type from the title or filename
    metric_type = 'synapses'  # default
    if 'ratio' in title.lower() or 'ratio' in filename.lower():
        metric_type = 'ratio'
    elif 'transmission' in title.lower() or 'probability' in title.lower():
        metric_type = 'probability'
    
    # Check if this is a large matrix (optimization for performance)
    is_large = cmat.shape[0] > 100 or cmat.shape[1] > 100
    is_very_large = cmat.shape[0] > 500 or cmat.shape[1] > 500
    
    # Calculate sparsity for additional optimization decisions
    sparsity = (cmat.values == 0).sum() / cmat.size
    is_sparse = sparsity > 0.7  # More than 70% zeros
    
    # Deep optimization: For extremely large matrices, consider showing only non-zero entries
    # This creates a scatter plot instead of full heatmap for massive size reduction
    # DISABLED FOR NOW - needs more work to properly handle all cases
    use_scatter_mode = False  # is_very_large and is_sparse and cmat.size > 250000
    
    # if use_scatter_mode:
    #     print(f"  ⚡ Ultra-optimization: Using scatter mode for {cmat.shape[0]}×{cmat.shape[1]} sparse matrix")
    
    # Apply scale transformation to data
    z_data = cmat.values.copy()
    scale_label = ''
    
    if scale == 'log2':
        # Apply log2 transformation (add 1 to avoid log(0))
        z_data = np.log2(z_data + 1)
        scale_label = ' (log2)'
    elif scale == 'log10':
        # Apply log10 transformation (add 1 to avoid log(0))
        z_data = np.log10(z_data + 1)
        scale_label = ' (log10)'
    # else: scale == 'linear', use original values
    
    # Create custom hover text (only for smaller matrices)
    if not is_large:
        hover_text = []
        for i, row_label in enumerate(cmat.index):
            hover_row = []
            for j, col_label in enumerate(cmat.columns):
                value = cmat.iloc[i, j]
                # Format value based on metric type (show original value)
                if metric_type == 'ratio' or metric_type == 'probability':
                    value_str = f'{value:.4f}'
                else:
                    value_str = f'{int(value):,}' if value == int(value) else f'{value:,.2f}'
                
                hover_row.append(
                    f'<b>Source:</b> {row_label}<br>'
                    f'<b>Target:</b> {col_label}<br>'
                    f'<b>{metric_type.capitalize()}:</b> {value_str}'
                )
            hover_text.append(hover_row)
    else:
        # For large matrices, use simplified hover with automatic formatting
        hover_text = None

    # Determine color bar range
    zmin = z_data.min()
    zmax = z_data.max()
    if metric_type == 'synapses' and scale == 'linear':
        # For synapse counts, outliers can skew the color bar. Use 99th percentile for zmax.
        if zmax > 0:
            zmax = np.percentile(z_data, 99)
    elif metric_type in ['ratio', 'probability'] and scale == 'linear':
        zmin = 0.0
        zmax = min(zmax, 1.0) # Cap at 1.0

    # Create visualization with appropriate mode
    if use_scatter_mode:
        # Ultra-optimization: Use scatter plot showing only non-zero values
        # This reduces file size by 90%+ for sparse matrices
        non_zero_mask = z_data != 0
        rows, cols = np.where(non_zero_mask)
        values = z_data[non_zero_mask]
        
        # Create scatter plot (much more efficient for sparse data)
        fig = go.Figure(data=go.Scatter(
            x=cols,
            y=rows,
            mode='markers',
            marker=dict(
                size=8,
                color=values,
                colorscale=color_scale,
                cmin=zmin,
                cmax=zmax,
                colorbar=dict(
                    title=metric_type.capitalize() + scale_label,
                    titleside='right'
                ),
                line=dict(width=0.5, color='rgba(0,0,0,0.2)')
            ),
            hovertemplate=(
                'Row: %{y}<br>'
                'Col: %{x}<br>'
                'Value: %{marker.color:.2f}<br>'
                '<extra></extra>'
            )
        ))
        
        # Add note about visualization mode
        title = f"{title}<br><sub style='color:#666;'>Sparse matrix visualization (showing {len(values):,} non-zero connections)</sub>"
        
    else:
        # Standard heatmap mode
        heatmap_config = {
            'z': z_data,
            'colorscale': color_scale,
            'zmin': zmin,
            'zmax': zmax,
            'colorbar': dict(
                title=metric_type.capitalize() + scale_label,
                titleside='right'
            )
        }
        
        # Deep optimization: For large matrices, use indices instead of full labels
        if is_large:
            # Use numeric indices to drastically reduce JSON size
            # Labels stored separately and not embedded in every data point
            heatmap_config['x'] = list(range(len(cmat.columns)))
            heatmap_config['y'] = list(range(len(cmat.index)))
            
            # Simplified hover using indices (Plotly will auto-format)
            heatmap_config['hovertemplate'] = (
                'Row: %{y}<br>'
                'Col: %{x}<br>'
                'Value: %{z:.2f}<br>'
                '<extra></extra>'
            )
        else:
            # For small matrices, keep full labels
            heatmap_config['x'] = cmat.columns.astype(str)
            heatmap_config['y'] = cmat.index.astype(str)
            
            if hover_text is not None:
                heatmap_config['text'] = hover_text
                heatmap_config['hoverinfo'] = 'text'
            else:
                heatmap_config['hovertemplate'] = (
                    '<b>Source:</b> %{y}<br>'
                    '<b>Target:</b> %{x}<br>'
                    '<b>Value:</b> %{z:.2f}<br>'
                    '<extra></extra>'
                )
        
        fig = go.Figure(data=go.Heatmap(**heatmap_config))
    
    # Update layout with axis labels
    layout_config = {
        'title_text': title,
        'font_size': fontsize,
        'xaxis': dict(
            title='<b>Target</b>',
            side='bottom',
            titlefont=dict(size=fontsize+2, color='#333333'),
            tickangle=-45 if len(cmat.columns) > 1 else 0,  # Always rotate when multiple labels
            range=[-0.5, len(cmat.columns) - 0.5] if use_scatter_mode else None
        ),
        'yaxis': dict(
            title='<b>Source</b>',
            side='left',
            titlefont=dict(size=fontsize+2, color='#333333'),
            autorange='reversed',  # Keep the original order (top to bottom)
            range=[len(cmat.index) - 0.5, -0.5] if use_scatter_mode else None
        ),
        'hoverlabel': dict(
            bgcolor='white',
            font_size=12,
            font_family='Arial'
        ),
        'autosize': True,
        'margin': dict(l=120, r=40, b=120, t=140 if use_scatter_mode else 100, pad=4)
    }
    
    # For scatter mode, ensure proper aspect ratio
    if use_scatter_mode:
        layout_config['xaxis']['constrain'] = 'domain'
        layout_config['yaxis']['scaleanchor'] = 'x'
        layout_config['plot_bgcolor'] = 'white'
        layout_config['xaxis']['showgrid'] = True
        layout_config['yaxis']['showgrid'] = True
        layout_config['xaxis']['gridcolor'] = 'rgba(0,0,0,0.1)'
        layout_config['yaxis']['gridcolor'] = 'rgba(0,0,0,0.1)'
    
    # For large matrices, hide tick labels to reduce file size
    if is_large or use_scatter_mode:
        layout_config['xaxis']['showticklabels'] = False
        layout_config['yaxis']['showticklabels'] = False
        layout_config['xaxis']['title'] = f'<b>Target</b> ({len(cmat.columns)} neurons)'
        layout_config['yaxis']['title'] = f'<b>Source</b> ({len(cmat.index)} neurons)'
    
    fig.update_layout(**layout_config)
    
    # Write HTML with deep backend optimizations
    write_config = {
        'auto_open': showfig,
        'include_plotlyjs': 'cdn',  # Use CDN instead of embedding 3MB library
        'config': {
            'displayModeBar': True, 
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'] if is_large else []
        }
    }
    
    # Deep optimization for large matrices (but not scatter mode)
    if is_large and not use_scatter_mode:
        # Use div-only output for embedding (smaller than full HTML)
        write_config['include_mathjax'] = False  # Disable MathJax (not needed)
        write_config['div_id'] = 'heatmap'  # Enable div mode for potential embedding
        
        # Reduce decimal precision in JSON to save space
        # Plotly uses default precision which can be excessive for visualization
        # This is handled by converting figure to dict and rounding
        fig_dict = fig.to_dict()
        
        # Round z values to reduce JSON size
        if 'data' in fig_dict and len(fig_dict['data']) > 0:
            z_values = fig_dict['data'][0].get('z', [])
            if isinstance(z_values, (list, np.ndarray)):
                # For sparse matrices, more aggressive rounding
                decimals = 1 if is_sparse else 2
                
                # Convert to numpy array if needed
                z_array = np.array(z_values)
                
                # Round values
                z_rounded = np.round(z_array, decimals)
                
                # For very sparse matrices, explicitly convert zeros to reduce storage
                if is_sparse:
                    # Set very small values to exactly 0 (reduces file size via compression)
                    z_rounded[np.abs(z_rounded) < 0.01] = 0
                
                fig_dict['data'][0]['z'] = z_rounded.tolist()
        
        # Recreate figure from optimized dict
        fig = go.Figure(fig_dict)
        
        # Add compression hint in title for user awareness
        if is_sparse and not use_scatter_mode:
            sparsity_pct = int(sparsity * 100)
            fig.update_layout(
                title_text=f"{title}<br><sub style='color:#666;'>Matrix {sparsity_pct}% sparse - optimized for file size</sub>"
            )
    
    fig.write_html(filename, **write_config)


def VisConnMatInteractive(cmat, filename, title='', color_scale=[[0, 'rgb(255,255,255)'], [1, 'rgb(104,55,164)']], showfig=True, fontsize=12, conn_df=None, matrices_dict=None, verbose=True):
    '''Create interactive heatmap with comprehensive controls similar to network visualization
    
    Features:
    - Metric toggle: Switch between weight/ratio/probability (if provided)
    - Clustering toggle: Toggle between original and clustered ordering (hierarchical clustering)
    - Scale switcher: Linear / Log2 / Log10 / Sqrt
    - Colorscale selector with presets (Greens, Purples, Oranges, Blues, Reds, Viridis, etc.)
    - Font size slider
    - Export to SVG with adjustable resolution
    - Zoom/pan controls
    - Save/load layout state
    
    Parameters
    ----------
    cmat : pd.DataFrame
        Connection matrix to visualize (weight matrix if matrices_dict not provided)
    filename : str
        Output HTML filename
    title : str, optional
        Title for the heatmap
    color_scale : list, optional
        Plotly color scale (default starting point)
    showfig : bool, optional
        Whether to open in browser
    fontsize : int, optional
        Default font size for labels
    conn_df : pd.DataFrame, optional
        Connection dataframe with type information for enhanced hover labels (bodyId heatmaps only)
    matrices_dict : dict, optional
        Dictionary with keys 'weight', 'ratio', 'probability' containing different metric matrices
        If provided, enables metric toggle. Otherwise uses cmat as weight matrix only.
    verbose : bool, optional
        Whether to print progress messages (default: True)
    '''
    
    # Handle multiple matrices for metric toggle
    has_multiple_metrics = matrices_dict is not None and isinstance(matrices_dict, dict)
    
    if has_multiple_metrics:
        # Use provided matrices dictionary
        available_metrics = []
        matrices_data = {}
        
        if 'weight' in matrices_dict and matrices_dict['weight'] is not None:
            available_metrics.append('weight')
            matrices_data['weight'] = matrices_dict['weight'].values.copy()
        
        if 'ratio' in matrices_dict and matrices_dict['ratio'] is not None:
            available_metrics.append('ratio')
            matrices_data['ratio'] = matrices_dict['ratio'].values.copy()
        
        if 'probability' in matrices_dict and matrices_dict['probability'] is not None:
            available_metrics.append('probability')
            matrices_data['probability'] = matrices_dict['probability'].values.copy()
        
        # Use first available metric as default
        default_metric = available_metrics[0] if available_metrics else 'weight'
        data_linear = matrices_data.get(default_metric, cmat.values.copy())
        metric_type = default_metric
    else:
        # Single matrix mode - determine metric type from title/filename
        available_metrics = ['weight']  # Only one metric available
        matrices_data = {}
        
        metric_type = 'weight'
        if 'ratio' in title.lower() or 'ratio' in filename.lower():
            metric_type = 'ratio'
            available_metrics = ['ratio']
        elif 'transmission' in title.lower() or 'probability' in title.lower():
            metric_type = 'probability'
            available_metrics = ['probability']
        
        data_linear = cmat.values.copy()
        matrices_data[metric_type] = data_linear
    
    is_large = cmat.shape[0] > 100 or cmat.shape[1] > 100
    
    # Check sparsity for potential optimization
    zero_count = np.count_nonzero(data_linear == 0)
    sparsity_ratio = zero_count / data_linear.size
    is_sparse = sparsity_ratio > 0.5  # More than 50% zeros
    
    # Compute hierarchical clustering with multiple methods for row/column ordering
    if verbose:
        print("  Computing hierarchical clustering...")
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist
    
    # Store clustering results for all methods
    clustering_methods = ['ward', 'average', 'complete', 'single']
    clustering_results = {}
    
    try:
        for method in clustering_methods:
            method_results = {}
            
            # Cluster rows (source neurons)
            if data_linear.shape[0] > 1:
                # Use euclidean distance (required for ward, good for others)
                row_distances = pdist(data_linear, metric='euclidean')
                # Check for non-finite values
                if not np.all(np.isfinite(row_distances)):
                    raise ValueError("Non-finite distances in row clustering")
                row_linkage = linkage(row_distances, method=method)
                method_results['row_order'] = leaves_list(row_linkage).tolist()
            else:
                method_results['row_order'] = [0]
            
            # Cluster columns (target neurons)
            if data_linear.shape[1] > 1:
                col_distances = pdist(data_linear.T, metric='euclidean')
                # Check for non-finite values
                if not np.all(np.isfinite(col_distances)):
                    raise ValueError("Non-finite distances in column clustering")
                col_linkage = linkage(col_distances, method=method)
                method_results['col_order'] = leaves_list(col_linkage).tolist()
            else:
                method_results['col_order'] = [0]
            
            clustering_results[method] = method_results
        
        # Use Ward as default (best for most connectome data)
        row_order_clustered = np.array(clustering_results['ward']['row_order'])
        col_order_clustered = np.array(clustering_results['ward']['col_order'])
        
        clustering_successful = True
        print(f"  ✓ Clustering complete: {len(row_order_clustered)} rows, {len(col_order_clustered)} cols")
        print(f"  Available methods: Ward (default), Average, Complete, Single")
    except Exception as e:
        print(f"  ⚠ Clustering failed: {e}")
        print(f"  Using original order")
        row_order_clustered = np.array(range(data_linear.shape[0]))
        col_order_clustered = np.array(range(data_linear.shape[1]))
        clustering_successful = False
        clustering_results = {}

    
    # Store both original and clustered orders
    row_order_original = list(range(data_linear.shape[0]))
    col_order_original = list(range(data_linear.shape[1]))
    
    # For large matrices, reduce precision to save HTML size
    # Keep more precision for ratio/probability metrics
    if is_large:
        if metric_type in ['ratio', 'probability']:
            # Keep 4 decimal places for ratios/probabilities
            data_linear = np.round(data_linear, 4)
        else:
            # For synapse counts, round to integers (no precision loss)
            data_linear = np.round(data_linear, 0)
    
    # Deep optimization: For very large matrices, compute transforms in JavaScript
    # This saves ~75% of HTML file size by not embedding pre-computed transforms
    use_lazy_transforms = is_large and data_linear.size > 50000
    
    # Sparse matrix optimization: For matrices with >70% zeros, use COO format
    use_sparse_format = is_large and sparsity_ratio > 0.7 and data_linear.size > 50000
    sparse_data = None
    
    if use_sparse_format:
        # Convert to COO (Coordinate) format: store only non-zero values
        rows, cols = np.nonzero(data_linear)
        values = data_linear[rows, cols]
        sparse_data = {
            'rows': rows.tolist(),
            'cols': cols.tolist(),
            'values': values.tolist(),
            'shape': list(data_linear.shape)
        }
        print(f"  Using sparse format: {sparsity_ratio*100:.1f}% zeros, storing {len(values)} values instead of {data_linear.size}")
    
    if use_lazy_transforms:
        # Store only linear data; transforms computed client-side
        data_log2 = None
        data_log10 = None
        data_sqrt = None
    else:
        # Pre-compute for small matrices (faster initial display)
        # Handle negative values: sign(v) * transform(|v|)
        data_log2 = np.where(data_linear >= 0, 
                             np.log2(data_linear + 1), 
                             -np.log2(-data_linear + 1))
        data_log10 = np.where(data_linear >= 0, 
                              np.log10(data_linear + 1), 
                              -np.log10(-data_linear + 1))
        data_sqrt = np.where(data_linear >= 0, 
                            np.sqrt(data_linear), 
                            -np.sqrt(-data_linear))
        
        if is_large:
            if metric_type in ['ratio', 'probability']:
                data_log2 = np.round(data_log2, 4)
                data_log10 = np.round(data_log10, 4)
                data_sqrt = np.round(data_sqrt, 4)
            else:
                data_log2 = np.round(data_log2, 2)
                data_log10 = np.round(data_log10, 2)
                data_sqrt = np.round(data_sqrt, 2)
    
    # Create hover text with original values
    # If conn_df is provided, create type lookup for bodyId heatmaps
    type_lookup = None
    
    if conn_df is not None and 'bodyId_pre' in conn_df.columns and 'type_pre' in conn_df.columns:
        # Create lookup dictionaries for bodyId -> type
        # Convert bodyId keys to strings to match matrix index/columns
        type_lookup = {
            'pre': {str(k): v for k, v in conn_df.set_index('bodyId_pre')['type_pre'].to_dict().items()},
            'post': {str(k): v for k, v in conn_df.set_index('bodyId_post')['type_post'].to_dict().items()}
        }
    # Generate hover text with actual labels for all matrix sizes
    # No longer use compact mode - always show full information with proper labels
    hover_text = []
    for i, row_label in enumerate(cmat.index):
        hover_row = []
        for j, col_label in enumerate(cmat.columns):
            value = cmat.iloc[i, j]
            if metric_type == 'ratio' or metric_type == 'probability':
                value_str = f'{value:.4f}'
            else:
                value_str = f'{int(value):,}' if value == int(value) else f'{value:,.2f}'
            
            # Always use actual labels with type info if available
            if type_lookup:
                try:
                    # Labels are already strings, use them directly for type lookup
                    row_id = str(row_label);
                    col_id = str(col_label);
                    row_type = type_lookup['pre'].get(row_id, 'Unknown')
                    col_type = type_lookup['post'].get(col_id, 'Unknown')
                    hover_row.append(f'<b>Source:</b> {row_label} ({row_type})<br><b>Target:</b> {col_label} ({col_type})<br><b>{metric_type.capitalize()}:</b> {value_str}')
                except:
                    # Fall back to label-only display if type lookup fails
                    hover_row.append(f'<b>Source:</b> {row_label}<br><b>Target:</b> {col_label}<br><b>{metric_type.capitalize()}:</b> {value_str}')
            else:
                # No type info available - just show labels
                hover_row.append(f'<b>Source:</b> {row_label}<br><b>Target:</b> {col_label}<br><b>{metric_type.capitalize()}:</b> {value_str}')
        hover_text.append(hover_row)
    
    # Determine axis labels - ALWAYS use actual names, not numeric indices
    # Even for large matrices, show proper labels (optimization only affects hover text)
    x_labels = cmat.columns.astype(str).tolist()
    y_labels = cmat.index.astype(str).tolist();
    
    # Generate unique storage key for this heatmap
    from datetime import datetime
    output_name = os.path.splitext(os.path.basename(filename))[0]
    timestamp_hash = datetime.now().strftime('%Y%m%d%H%M%S');
    storage_key = f"heatmap_settings_{output_name}#{timestamp_hash}"
    
    # Determine default colorscale name
    default_colorscale = 'Greens'
    if 'ratio' in filename.lower():
        default_colorscale = 'Oranges'
    elif 'transmission' in filename.lower() or 'probability' in filename.lower():
        default_colorscale = 'Purples'
    
    # Create HTML with comprehensive interactive controls
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            user-select: text;
        }}
        
        .main-container {{
            max-width: 1800px;
            margin: 0 auto;
        }}
        
        .controls {{
            background: white;
            padding: 12px;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }}
        
        .controls-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 8px;
            margin-bottom: 10px;
        }}
        
        .control-section {{
            background: #f8f9fa;
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #e9ecef;
        }}
        
        .control-section h3 {{
            margin: 0 0 8px 0;
            font-size: 12px;
            font-weight: 600;
            color: #495057;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}
        
        .button-group {{
            display: flex;
            gap: 4px;
            flex-wrap: wrap;
        }}
        
        button {{
            padding: 6px 10px;
            border: 1px solid #dee2e6;
            background: white;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
            font-weight: 500;
            transition: all 0.2s;
            color: #495057;
        }}
        
        button:hover {{
            background: #f8f9fa;
            border-color: #adb5bd;
        }}
        
        button.active {{
            background: #4CAF50;
            color: white;
            border-color: #4CAF50;
        }}
        
        button.export-btn {{
            background: #2196F3;
            color: white;
            border-color: #2196F3;
        }}
        
        button.export-btn:hover {{
            background: #1976D2;
            border-color: #1976D2;
        }}
        
        button.save-btn {{
            background: #FF9800;
            color: white;
            border-color: #FF9800;
        }}
        
        button.save-btn:hover {{
            background: #F57C00;
            border-color: #F57C00;
        }}
        
        select {{
            width: 100%;
            padding: 4px 6px;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            font-size: 11px;
            background: white;
            cursor: pointer;
            color: #495057;
        }}
        
        select:focus {{
            outline: none;
            border-color: #4CAF50;
            box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.1);
        }}
        
        .slider-control {{
            margin-bottom: 6px;
        }}
        
        .slider-control label {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 3px;
            font-size: 10px;
            color: #495057;
            font-weight: 500;
        }}
        
        .slider-value {{
            color: #4CAF50;
            font-weight: 600;
        }}
        
        input[type="range"] {{
            width: 100%;
            height: 4px;
            border-radius: 2px;
            background: #dee2e6;
            outline: none;
            -webkit-appearance: none;
        }}
        
        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #4CAF50;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        input[type="range"]::-webkit-slider-thumb:hover {{
            background: #45a049;
            transform: scale(1.15);
        }}
        
        input[type="range"]::-moz-range-thumb {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #4CAF50;
            cursor: pointer;
            border: none;
        }}
        
        #heatmap-container {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        #heatmap {{
            width: 100%;
            height: 1200px;
        }}
        
        .status-message {{
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            text-align: center;
            margin-top: 8px;
        }}
        
        .status-success {{
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        
        .status-info {{
            background: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }}
        
        .status-error {{
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}
        
        .info-box {{
            background: #e7f3ff;
            border-left: 3px solid #2196F3;
            padding: 8px;
            border-radius: 3px;
            font-size: 10px;
            color: #1976D2;
            margin-top: 8px;
            line-height: 1.4;
        }}
        
        .info-box strong {{
            display: block;
            margin-bottom: 3px;
            font-size: 11px;
        }}
        
        .drag-item {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 3px;
            padding: 6px 8px;
            margin-bottom: 4px;
            cursor: move;
            user-select: none;
            display: flex;
            align-items: center;
            transition: all 0.2s;
        }}
        
        .drag-item:hover {{
            background: #f0f0f0;
            border-color: #4CAF50;
        }}
        
        .drag-item.dragging {{
            opacity: 0.5;
            background: #e3f2fd;
        }}
        
        .drag-item.drag-over {{
            border-top: 3px solid #4CAF50;
        }}
        
        .drag-handle {{
            margin-right: 6px;
            color: #999;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="main-container">
        <div class="controls">
            <div class="controls-grid">
                <!-- Metric, Ordering & Scale Combined Section -->
                {'<div class="control-section" id="metricOrderingSection">' if has_multiple_metrics else '<div class="control-section">'}
                    {'<h3>📊 Metric, Ordering & Scale</h3>' if has_multiple_metrics else '<h3>🔀 Ordering & Scale</h3>'}
                    
                    <!-- Metric Selection (if multiple metrics available) -->
                    {'<div style="margin-bottom: 8px;"><label style="font-size: 10px; display: block; margin-bottom: 2px;">Metric:</label>' if has_multiple_metrics else '<!-- Single metric mode -->'}
                    {'<select id="metricSelect" onchange="updateMetric()">' if has_multiple_metrics else ''}
                        {'<option value="weight">Synapse Count</option>' if has_multiple_metrics and 'weight' in available_metrics else ''}
                        {'<option value="ratio"' + (' selected' if metric_type == 'ratio' else '') + '>Connection Ratio</option>' if has_multiple_metrics and 'ratio' in available_metrics else ''}
                        {'<option value="probability"' + (' selected' if metric_type == 'probability' else '') + '>Traversal Probability</option>' if has_multiple_metrics and 'probability' in available_metrics else ''}
                    {'</select></div>' if has_multiple_metrics else ''}
                    
                    <!-- Clustering Toggle -->
                    <div style="margin-bottom: 8px;">
                        <label style="font-size: 10px; display: block; margin-bottom: 2px;">Ordering:</label>
                        <div class="button-group">
                            <button id="btn-original" class="active" onclick="toggleClustering('original')">Original</button>
                            <button id="btn-clustered" onclick="toggleClustering('clustered')">Clustered</button>
                        </div>
                    </div>
                    
                    <!-- Clustering Method Selection -->
                    <div id="clusteringMethodSection" style="margin-bottom: 8px; display: none;">
                        <label style="font-size: 10px; display: block; margin-bottom: 2px;">Clustering Method:</label>
                        <select id="clusteringMethodSelect" onchange="updateClusteringMethod()" style="width: 100%; font-size: 10px; padding: 4px;">
                            <option value="ward">Ward (Compact Clusters)</option>
                            <option value="average">Average (Balanced)</option>
                            <option value="complete">Complete (Tight Clusters)</option>
                            <option value="single">Single (Loose Clusters)</option>
                        </select>
                    </div>
                    
                    <!-- Scale Selection -->
                    <div>
                        <label style="font-size: 10px; display: block; margin-bottom: 2px;">Scale:</label>
                        <div class="button-group">
                            <button id="btn-linear" class="active" onclick="setScale('linear')">Linear</button>
                            <button id="btn-log2" onclick="setScale('log2')">Log₂</button>
                            <button id="btn-log10" onclick="setScale('log10')">Log₁₀</button>
                            <button id="btn-sqrt" onclick="setScale('sqrt')">√</button>
                        </div>
                    </div>
                </div>
                
                <!-- Color -->
                <div class="control-section" id="colorscaleSection">
                    <h3>🎨 Color</h3>
                    <select id="colorscaleSelect" onchange="updateColorscale()" style="margin-bottom: 8px;">
                        <option value="Greens" {'selected' if default_colorscale == 'Greens' else ''}>Greens</option>
                        <option value="Purples" {'selected' if default_colorscale == 'Purples' else ''}>Purples</option>
                        <option value="Oranges" {'selected' if default_colorscale == 'Oranges' else ''}>Oranges</option>
                        <option value="Blues" {'selected' if default_colorscale == 'Blues' else ''}>Blues</option>
                        <option value="Reds">Reds</option>
                        <option value="Viridis">Viridis</option>
                        <option value="Plasma">Plasma</option>
                        <option value="Inferno">Inferno</option>
                        <option value="Magma">Magma</option>
                        <option value="Cividis">Cividis</option>
                        <option value="Hot">Hot</option>
                        <option value="Jet">Jet</option>
                        <option value="RdBu">Red-Blue (Diverging)</option>
                        <option value="RdYlGn">Red-Yellow-Green</option>
                        <option value="Custom">Custom</option>
                    </select>
                    
                    <div id="customColorSection">
                        <div style="margin-bottom: 6px;">
                            <label style="display: block; margin-bottom: 3px; font-size: 10px;">
                                <input type="checkbox" id="use3PointScale" onchange="toggle3PointScale()"> 
                                3-Point Scale (diverging)
                            </label>
                        </div>
                        <div id="twoPointColors">
                            <div style="margin-bottom: 4px;">
                                <label style="font-size: 10px; display: block; margin-bottom: 2px;">Min (value):</label>
                                <div style="display: flex; gap: 4px;">
                                    <input type="color" id="colorMin" value="#ffffff" style="width: 40px; height: 26px; cursor: pointer;">
                                    <input type="number" id="valueMin2" placeholder="Auto" step="any" style="flex: 1; padding: 2px 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px;">
                                </div>
                            </div>
                            <div style="margin-bottom: 4px;">
                                <label style="font-size: 10px; display: block; margin-bottom: 2px;">Max (value):</label>
                                <div style="display: flex; gap: 4px;">
                                    <input type="color" id="colorMax" value="#68379c" style="width: 40px; height: 26px; cursor: pointer;">
                                    <input type="number" id="valueMax2" placeholder="Auto" step="any" style="flex: 1; padding: 2px 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px;">
                                </div>
                            </div>
                        </div>
                        <div id="threePointColors" style="display: none;">
                            <div style="margin-bottom: 4px;">
                                <label style="font-size: 10px; display: block; margin-bottom: 2px;">Min (value):</label>
                                <div style="display: flex; gap: 4px;">
                                    <input type="color" id="colorMin3" value="#0000ff" style="width: 40px; height: 26px; cursor: pointer;">
                                    <input type="number" id="valueMin3" value="0" step="any" style="flex: 1; padding: 2px 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px;">
                                </div>
                            </div>
                            <div style="margin-bottom: 4px;">
                                <label style="font-size: 10px; display: block; margin-bottom: 2px;">Mid (value):</label>
                                <div style="display: flex; gap: 4px;">
                                    <input type="color" id="colorMid3" value="#ffffff" style="width: 40px; height: 26px; cursor: pointer;">
                                    <input type="number" id="valueMid3" value="0.5" step="any" style="flex: 1; padding: 2px 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px;">
                                </div>
                            </div>
                            <div style="margin-bottom: 4px;">
                                <label style="font-size: 10px; display: block; margin-bottom: 2px;">Max (value):</label>
                                <div style="display: flex; gap: 4px;">
                                    <input type="color" id="colorMax3" value="#ff0000" style="width: 40px; height: 26px; cursor: pointer;">
                                    <input type="number" id="valueMax3" value="1" step="any" style="flex: 1; padding: 2px 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px;">
                                </div>
                            </div>
                        </div>
                        <div style="display: flex; gap: 4px; margin-top: 4px;">
                            <button onclick="applyCustomColors()" style="flex: 1; font-size: 10px;">Apply</button>
                            <button onclick="resetToAutoColors()" style="flex: 1; font-size: 10px;">Auto</button>
                        </div>
                    </div>
                </div>
                
                <!-- Font Size & Colorbar Settings -->
                <div class="control-section">
                    <h3>🎚️ Display</h3>
                    <div class="slider-control">
                        <label>
                            <span>Font Size:</span>
                            <span class="slider-value" id="fontSizeValue">{fontsize}px</span>
                        </label>
                        <input type="range" id="fontSizeSlider" min="8" max="48" value="{fontsize}" step="1" oninput="updateFontSize(this.value)">
                    </div>
                    <div style="margin-top: 8px; display: flex; gap: 4px;">
                        <button id="toggleLabelsBtn" onclick="toggleLabels()" style="flex: 1;">
                            {'🏷️ Hide Text' if not is_large else '🏷️ Show Text'}
                        </button>
                        <button id="toggleCellValuesBtn" onclick="toggleCellValues()" style="flex: 1;">
                            🔢 Hide Values
                        </button>
                    </div>
                    <div class="slider-control" style="margin-top: 8px;">
                        <label>
                            <span>Cell Value Size:</span>
                            <span class="slider-value" id="cellValueSizeValue">10px</span>
                        </label>
                        <input type="range" id="cellValueSizeSlider" min="6" max="48" value="10" step="1" oninput="updateCellValueSize(this.value)">
                    </div>
                    <div style="margin-top: 8px;">
                        <label style="font-size: 11px; display: block; margin-bottom: 4px;">Ignore Values (comma-separated):</label>
                        <input type="text" id="ignoreValuesInput" placeholder="e.g., 0, >20, <=5" style="width: 100%; padding: 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px; box-sizing: border-box;" oninput="updateIgnoredValues()">
                    </div>
                    <div style="margin-top: 8px;">
                        <label style="font-size: 11px; display: block; margin-bottom: 4px;">
                            🔍 Data Filter (hide rows/cols):
                            <button onclick="resetDataFilter()" style="padding: 2px 6px; font-size: 9px; background: #6c757d; color: white; border: none; border-radius: 3px; cursor: pointer; margin-left: 4px;" title="Reset filter">🔄 Reset</button>
                        </label>
                        <input type="text" id="dataFilterInput" placeholder="e.g., <5, <=10, >100" style="width: 100%; padding: 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px; box-sizing: border-box;" oninput="applyDataFilter()">
                        <div id="filterStatus" style="font-size: 9px; color: #666; margin-top: 2px; min-height: 14px;"></div>
                    </div>
                    <div class="slider-control" style="margin-top: 8px;">
                        <label>
                            <span>Contrast Threshold:</span>
                            <span class="slider-value" id="contrastThresholdValue">0.5000</span>
                            <button onclick="reverseContrastColors()" style="padding: 2px 6px; font-size: 10px; background: #6c757d; color: white; border: none; border-radius: 3px; cursor: pointer; margin-left: 4px;" title="Reverse black/white colors">🔄</button>
                        </label>
                        <input type="range" id="contrastThresholdSlider" min="0" max="1" value="0.5" step="0.0001" oninput="updateContrastThreshold(this.value)">
                    </div>
                </div>
                
                <!-- Plot Dimensions -->
                <div class="control-section">
                    <h3>📐 Plot Size</h3>
                    <div class="slider-control">
                        <label>
                            <span>Width:</span>
                            <span class="slider-value" id="widthValue">800px</span>
                        </label>
                        <div style="display: flex; gap: 4px; align-items: center;">
                            <input type="range" id="widthSlider" min="400" max="2400" value="800" step="20" oninput="updatePlotSize()" style="flex: 1;">
                            <input type="number" id="widthInput" value="800" min="100" step="20" style="width: 70px; padding: 2px 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px;" oninput="updatePlotSizeFromInput()">
                        </div>
                   
                                      
                    <div class="slider-control">
                        <label>
                            <span>Height:</span>
                            <span class="slider-value" id="heightValue">800px</span>
                        </label>
                        <div style="display: flex; gap: 4px; align-items: center;">
                            <input type="range" id="heightSlider" min="400" max="2400" value="800" step="20" oninput="updatePlotSize()" style="flex: 1;">
                            <input type="number" id="heightInput" value="800" min="100" step="20" style="width: 70px; padding: 2px 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px;" oninput="updatePlotSizeFromInput()">
                        </div>
                    </div>
                    <div style="display: flex; gap: 4px;">
                        <button id="squareCellsBtn" onclick="makeSquareCells()" style="flex: 1;">⬜ Square Cells</button>
                        <button onclick="resetPlotSize()" style="flex: 1;">🔄 Reset</button>
                    </div>
                    <div style="margin-top: 8px;">
                        <button id="transposeBtn" onclick="transposeMatrix()" style="width: 100%;">🔄 Swap Rows ↔ Columns</button>
                    </div>
                </div>
                
                <!-- Row/Column Ordering -->
                <div class="control-section">
                    <h3>📋 Row/Column Order</h3>
                    <button onclick="toggleOrderPanel('rows')" style="width: 100%; font-size: 10px; margin-bottom: 4px;">📑 Reorder Rows</button>
                    <button onclick="toggleOrderPanel('cols')" style="width: 100%; font-size: 10px; margin-bottom: 4px;">📑 Reorder Columns</button>
                    <button onclick="resetOrder()" style="width: 100%; font-size: 10px;">🔄 Reset to Original</button>
                    
                </div>
                
                <!-- Export & Saving -->
                <div class="control-section">
                    <h3>💾 Export & Saving</h3>
                    <div class="slider-control" style="margin-bottom: 8px;">
                        <label>
                            <span>Export Scale:</span>
                            <span class="slider-value" id="exportScaleValue">2x</span>
                        </label>
                        <input type="range" id="exportScaleSlider" min="1" max="5" value="2" step="0.5" oninput="updateExportScale(this.value)">
                    </div>
                    <div class="button-group" style="flex-direction: column; margin-bottom: 8px;">
                        <button class="export-btn" onclick="exportSVG()" style="width: 100%;">📥 Export SVG</button>
                    </div>
                    <div class="button-group">
                        <button class="save-btn" onclick="saveSettings()">💾 Save</button>
                        <button class="save-btn" onclick="loadSettings()">📂 Load</button>
                        <button onclick="resetSettings()">🔄 Reset</button>
                    </div>
                    <div id="settingsStatus"></div>
                </div>
            </div>
            
            <div class="info-box">
                <strong>💡 Tips:</strong>
                Use Log₂ or Log₁₀ scales for large dynamic ranges • 
                Adjust plot size with width/height sliders for better visualization • 
                Use export scale (1x-5x) to control SVG resolution • 
                3-point custom colors ideal for diverging data (negative → zero → positive) • 
                Hover over cells for details • 
                Zoom and pan with mouse • 
                Settings persist across sessions
            </div>
        </div>
        
        <div id="heatmap-container">
            <div id="heatmap"></div>
        </div>
    </div>
    
    <!-- Floating Reorder Panel -->
    <div id="orderPanel" style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                                 background: white; border: 2px solid #333; border-radius: 8px; padding: 16px; 
                                 box-shadow: 0 4px 20px rgba(0,0,0,0.3); z-index: 10000; min-width: 300px; max-width: 400px; 
                                 max-height: 70vh; flex-direction: column; display: none;">
        <div style="margin-bottom: 12px; border-bottom: 2px solid #ddd; padding-bottom: 8px;">
            <label id="orderPanelLabel" style="font-size: 14px; font-weight: bold; color: #333;"></label>
        </div>
        <div id="orderList" style="font-size: 12px; overflow-y: auto; flex: 1; margin-bottom: 12px;"></div>
        <button onclick="closeOrderPanel()" style="width: 100%; font-size: 12px; padding: 8px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">✓ Close</button>
    </div>
    
    <!-- Overlay backdrop for floating panel -->
    <div id="orderPanelBackdrop" style="display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; 
                                        background: rgba(0,0,0,0.5); z-index: 9999;" onclick="closeOrderPanel()"></div>
    
    <script>
        // Metric toggle support
        const availableMetrics = {json.dumps(available_metrics)};
        const hasMultipleMetrics = availableMetrics.length > 1;
        let currentMetric = '{metric_type}';
        
        // Store all metric matrices
        const metricsData = {{}};
'''
    
    # Add metric data assignments
    for metric in available_metrics:
        html_content += f"        metricsData['{metric}'] = {json.dumps(matrices_data[metric].tolist())};\n"
    
    html_content += f'''
        
        // Data for different scales
        const sparseData = {json.dumps(sparse_data) if sparse_data is not None else 'null'};
        const useSparseFormat = sparseData !== null;
        
        // Get current metric data
        let dataLinear = metricsData[currentMetric];
        
        const dataLog2 = {'null' if data_log2 is None else json.dumps(data_log2.tolist())};
        const dataLog10 = {'null' if data_log10 is None else json.dumps(data_log10.tolist())};
        const dataSqrt = {'null' if data_sqrt is None else json.dumps(data_sqrt.tolist())};
        const xLabels = {json.dumps(x_labels)};
        const yLabels = {json.dumps(y_labels)};
        const storageKey = '{storage_key}';
        const useLazyTransforms = {json.dumps(use_lazy_transforms)};
        
        // Track current row/column order (for interactive reordering)
        let currentXLabels = xLabels.slice();
        let currentYLabels = yLabels.slice();
        
        // Hover text - always use full array with proper labels (no compact mode)
        const hoverText = {json.dumps(hover_text)};
        
        // Cache for lazy-computed transforms
        let cachedDataLog2 = null;
        let cachedDataLog10 = null;
        let cachedDataSqrt = null;
        
        // Clustering data - row and column orders for all methods
        const rowOrderOriginal = {json.dumps(row_order_original)};
        const colOrderOriginal = {json.dumps(col_order_original)};
        const clusteringAvailable = {json.dumps(clustering_successful)};
        
        // All clustering method results
        const clusteringResults = {json.dumps(clustering_results)};
        
        // Default to Ward method
        const rowOrderClustered = {json.dumps(row_order_clustered.tolist())};
        const colOrderClustered = {json.dumps(col_order_clustered.tolist())};
        
        // Current settings
        let currentScale = 'linear';
        let currentColorscale = '{default_colorscale}';
        let currentFontSize = {fontsize};
        let useAutoRange = true;
        let customZmin = null;
        let customZmax = null;
        let customColorScale = null;  // Store custom color scale
        let use3PointScale = false;
        let currentWidth = 800;
        let currentHeight = 800;
        let exportScale = 2;
        let squareCellsLocked = false;  // Track if square cells are locked
        let showLabels = !{json.dumps(is_large)};  // Show labels for small matrices, hide for large
        let showCellValues = true;  // Track if cell values should be displayed in cells (default: true)
        let cellValueFontSize = 10;  // Font size for cell value annotations
        let ignoredValues = new Set();  // Set of values to ignore when displaying cell values
        let contrastThreshold = 0.5;  // Luminance threshold for contrast color (0-1, default: 0.5)
        let reverseContrast = false;  // Whether to reverse black/white contrast colors
        let useClusteredOrder = false;  // Track current ordering mode
        let currentClusteringMethod = 'ward';  // Current clustering method (ward, average, complete, single)
        let isTransposed = false;  // Track if matrix is transposed
        const metricType = '{metric_type}';
        const isLarge = {json.dumps(is_large)};
        const originalTitle = '{title}';
        
        // Data filter state
        let dataFilterActive = false;
        let dataFilterExpressions = [];
        let filteredRowIndices = [];  // Indices of rows to show after filtering
        let filteredColIndices = [];  // Indices of columns to show after filtering
        
        // Function to generate hover text dynamically when needed
        // Hover text is pre-generated in Python with proper labels
        // This function regenerates it when switching metrics (multi-metric mode)
        function generateHoverText() {{
            if (!hasMultipleMetrics) {{
                return hoverText;  // Use pre-generated hover text for single-metric mode
            }}
            
            // Generate hover text on-the-fly for multi-metric mode
            const rows = dataLinear.length;
            const cols = dataLinear[0].length;
            const result = new Array(rows);
            
            // Get metric display name
            const metricNames = {{
                'weight': 'Synapses',
                'ratio': 'Ratio',
                'probability': 'Probability'
            }};
            const currentMetricName = metricNames[currentMetric] || currentMetric;
            
            for (let i = 0; i < rows; i++) {{
                result[i] = new Array(cols);
                for (let j = 0; j < cols; j++) {{
                    const value = dataLinear[i][j];
                    let valueStr;
                    if (currentMetric === 'ratio' || currentMetric === 'probability') {{
                        valueStr = value.toFixed(4);
                    }} else {{
                        valueStr = Math.floor(value) === value ? 
                            value.toLocaleString() : 
                            value.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
                    }}

                    // Always use actual labels from yLabels and xLabels
                    const srcLabel = yLabels[i];
                    const tgtLabel = xLabels[j];
                    result[i][j] = '<b>Source:</b> ' + srcLabel + '<br><b>Target:</b> ' + tgtLabel + '<br><b>' + currentMetricName + ':</b> ' + valueStr;
                }}
            }}
            return result;
        }}
        
        function getDataForScale(scale) {{
            if (!useLazyTransforms) {{
                // Use pre-computed data for small matrices
                switch(scale) {{
                    case 'log2': return dataLog2;
                    case 'log10': return dataLog10;
                    case 'sqrt': return dataSqrt;
                    default: return dataLinear;
                }}
            }}
            
            // Lazy computation for large matrices
            switch(scale) {{
                case 'log2':
                    if (cachedDataLog2 === null) {{
                        console.log('Computing log₂ transform...');
                        cachedDataLog2 = dataLinear.map(row => row.map(v => {{
                            // Handle negative values: sign(v) * log2(|v| + 1)
                            if (v < 0) return -Math.log2(-v + 1);
                            return Math.log2(v + 1);
                        }}));
                    }}
                    return cachedDataLog2;
                case 'log10':
                    if (cachedDataLog10 === null) {{
                        console.log('Computing log₁₀ transform...');
                        cachedDataLog10 = dataLinear.map(row => row.map(v => {{
                            // Handle negative values: sign(v) * log10(|v| + 1)
                            if (v < 0) return -Math.log10(-v + 1);
                            return Math.log10(v + 1);
                        }}));
                    }}
                    return cachedDataLog10;
                case 'sqrt':
                    if (cachedDataSqrt === null) {{
                        console.log('Computing √ transform...');
                        cachedDataSqrt = dataLinear.map(row => row.map(v => {{
                            // Handle negative values: sign(v) * sqrt(|v|)
                            if (v < 0) return -Math.sqrt(-v);
                            return Math.sqrt(v);
                        }}));
                    }}
                    return cachedDataSqrt;
                default:
                    return dataLinear;
            }}
        }}
        
        function getScaleLabel(scale) {{
            switch(scale) {{
                case 'log2': return ' (log₂)';
                case 'log10': return ' (log₁₀)';
                case 'sqrt': return ' (√)';
                default: return '';
            }}
        }}
        
        function getDataRange(data) {{
            let min = Infinity;
            let max = -Infinity;
            for (let row of data) {{
                for (let val of row) {{
                    if (val < min) min = val;
                    if (val > max) max = val;
                }}
            }}
            return {{min, max}};
        }}
        
        function reorderData(data, rowOrder, colOrder) {{
            // Reorder rows and columns of the data matrix according to given orders
            const reordered = new Array(rowOrder.length);
            for (let i = 0; i < rowOrder.length; i++) {{
                reordered[i] = new Array(colOrder.length);
                for (let j = 0; j < colOrder.length; j++) {{
                    reordered[i][j] = data[rowOrder[i]][colOrder[j]];
                }}
            }}
            return reordered;
        }}
        
        function reorderLabels(labels, order) {{
            // Reorder labels according to given order
            const reordered = new Array(order.length);
            for (let i = 0; i < order.length; i++) {{
                reordered[i] = labels[order[i]];
            }}
            return reordered;
        }}
        
        function reorderHoverText(hoverText, rowOrder, colOrder) {{
            // Reorder hover text according to given orders
            if (hoverText === null) return null;
            const reordered = new Array(rowOrder.length);
            for (let i = 0; i < rowOrder.length; i++) {{
                reordered[i] = new Array(colOrder.length);
                for (let j = 0; j < colOrder.length; j++) {{
                    reordered[i][j] = hoverText[rowOrder[i]][colOrder[j]];
                }}
            }}
            return reordered;
        }}
        
        function createHeatmap() {{
            // Safety check: ensure data is available
            if (!dataLinear || dataLinear.length === 0) {{
                console.error('Cannot create heatmap: data not available');
                return;
            }}
            
            let data = getDataForScale(currentScale);
            let dataOriginal = dataLinear.map(row => row.slice()); // Keep original for cell values
            const scaleLabel = getScaleLabel(currentScale);
            
            // Determine which labels to use based on transpose state
            let displayXLabels, displayYLabels;
            let currentHoverText = generateHoverText();
            
            if (isTransposed) {{
                // When transposed: rows become columns, columns become rows
                // So we use the swapped tracking variables
                displayXLabels = currentYLabels.slice();
                displayYLabels = currentXLabels.slice();
                
                // Transpose the data matrix
                data = data[0].map((_, colIndex) => data.map(row => row[colIndex]));
                dataOriginal = dataOriginal[0].map((_, colIndex) => dataOriginal.map(row => row[colIndex]));
                
                // Transpose hover text if available
                if (currentHoverText !== null) {{
                    currentHoverText = currentHoverText[0].map((_, colIndex) => 
                        currentHoverText.map(row => row[colIndex])
                    );
                }}
                
                // Now apply reordering based on current tracked order (already transposed)
                const baseXLabels = yLabels;
                const baseYLabels = xLabels;
                
                const rowMapping = displayYLabels.map(label => baseYLabels.indexOf(label));
                const colMapping = displayXLabels.map(label => baseXLabels.indexOf(label));
                
                // Reorder transposed data
                data = rowMapping.map(rowIdx => 
                    colMapping.map(colIdx => data[rowIdx][colIdx])
                );
                dataOriginal = rowMapping.map(rowIdx => 
                    colMapping.map(colIdx => dataOriginal[rowIdx][colIdx])
                );
                
                // Reorder hover text if available
                if (currentHoverText !== null) {{
                    currentHoverText = rowMapping.map(rowIdx => 
                        colMapping.map(colIdx => currentHoverText[rowIdx][colIdx])
                    );
                }}
            }} else {{
                // Normal (non-transposed) mode
                displayXLabels = currentXLabels.slice();
                displayYLabels = currentYLabels.slice();
                
                // Apply reordering if different from base labels
                const baseXLabels = xLabels;
                const baseYLabels = yLabels;
                
                const needsRowReorder = !arraysEqual(displayYLabels, baseYLabels);
                const needsColReorder = !arraysEqual(displayXLabels, baseXLabels);
                
                if (needsRowReorder || needsColReorder) {{
                    const rowMapping = displayYLabels.map(label => baseYLabels.indexOf(label));
                    const colMapping = displayXLabels.map(label => baseXLabels.indexOf(label));
                    
                    // Reorder data matrix
                    data = rowMapping.map(rowIdx => 
                        colMapping.map(colIdx => data[rowIdx][colIdx])
                    );
                    dataOriginal = rowMapping.map(rowIdx => 
                        colMapping.map(colIdx => dataOriginal[rowIdx][colIdx])
                    );
                    
                    // Reorder hover text if available
                    if (currentHoverText !== null) {{
                        currentHoverText = rowMapping.map(rowIdx => 
                            colMapping.map(colIdx => currentHoverText[rowIdx][colIdx])
                        );
                    }}
                }}
            }}
            
            // Apply clustering reordering if enabled (after transpose and custom reordering)
            if (useClusteredOrder && clusteringAvailable) {{
                // Get clustering results for the selected method
                const selectedMethod = clusteringResults[currentClusteringMethod];
                let methodRowOrder = rowOrderClustered;
                let methodColOrder = colOrderClustered;
                
                if (selectedMethod) {{
                    methodRowOrder = selectedMethod.row_order;
                    methodColOrder = selectedMethod.col_order;
                }} else {{
                    console.warn('Clustering method not found:', currentClusteringMethod, '- using default');
                }}
                
                // When transposed, swap the cluster orders to match the transposed dimensions
                const effectiveRowOrder = isTransposed ? methodColOrder : methodRowOrder;
                const effectiveColOrder = isTransposed ? methodRowOrder : methodColOrder;
                
                data = reorderData(data, effectiveRowOrder, effectiveColOrder);
                dataOriginal = reorderData(dataOriginal, effectiveRowOrder, effectiveColOrder);
                displayXLabels = reorderLabels(displayXLabels, effectiveColOrder);
                displayYLabels = reorderLabels(displayYLabels, effectiveRowOrder);
                // Reorder hover text if available
                if (currentHoverText !== null) {{
                    currentHoverText = reorderHoverText(currentHoverText, effectiveRowOrder, effectiveColOrder);
                }}
            }}
            
            // Apply data filter if active (hide rows/columns based on their max values)
            if (dataFilterActive && filteredRowIndices.length > 0 && filteredColIndices.length > 0) {{
                // Filter data matrix
                data = filteredRowIndices.map(rowIdx => 
                    filteredColIndices.map(colIdx => data[rowIdx][colIdx])
                );
                dataOriginal = filteredRowIndices.map(rowIdx => 
                    filteredColIndices.map(colIdx => dataOriginal[rowIdx][colIdx])
                );
                
                // Filter labels
                displayXLabels = filteredColIndices.map(idx => displayXLabels[idx]);
                displayYLabels = filteredRowIndices.map(idx => displayYLabels[idx]);
                
                // Filter hover text if available
                if (currentHoverText !== null) {{
                    currentHoverText = filteredRowIndices.map(rowIdx => 
                        filteredColIndices.map(colIdx => currentHoverText[rowIdx][colIdx])
                    );
                }}
                
                console.log(`Data filter: showing ${{filteredRowIndices.length}} rows × ${{filteredColIndices.length}} cols`);
            }}
            
            const range = getDataRange(data);
            
            // Determine which colorscale to use
            let colorscaleToUse;
            
            // Check if we should use custom colorscale
            if (currentColorscale === 'Custom' && customColorScale && Array.isArray(customColorScale) && customColorScale.length > 0) {{
                // Use the custom colorscale array directly
                colorscaleToUse = customColorScale;
                console.log('✓ createHeatmap: Using CUSTOM colorscale:', {{
                    scale: customColorScale,
                    length: customColorScale.length,
                    positions: customColorScale.map(c => c[0])
                }});
            }} else {{
                // For preset colorscales, convert to array format for Plotly compatibility
                // Plotly v1.58.5 doesn't recognize all colorscale names, so we define them explicitly
                colorscaleToUse = getPlotlyColorscaleArray(currentColorscale);
                console.log('createHeatmap: Using preset colorscale:', {{
                    name: currentColorscale,
                    isArray: Array.isArray(colorscaleToUse),
                    length: Array.isArray(colorscaleToUse) ? colorscaleToUse.length : 'N/A'
                }});
            }}
            
            // Get metric display name for colorbar
            const metricDisplayNames = {{
                'weight': 'Synapses',
                'ratio': 'Ratio',
                'probability': 'Probability'
            }};
            const metricDisplayName = metricDisplayNames[currentMetric] || currentMetric;
            
            const trace = {{
                z: data,
                x: displayXLabels.map((_, i) => i),  // Use indices for positioning
                y: displayYLabels.map((_, i) => i),  // Use indices for positioning
                type: 'heatmap',
                colorscale: colorscaleToUse,
                colorbar: {{
                    title: metricDisplayName + scaleLabel,
                    titleside: 'right'
                }}
            }};
            
            // Configure text display for cell values
            console.log('createHeatmap: showCellValues =', showCellValues);
            if (showCellValues) {{
                // Show cell values: use texttemplate to display z values
                console.log('Setting texttemplate to show cell values');
                
                // Create a text array from the data for display
                const textArray = data.map(row => row.map(val => val.toString()));
                
                trace.text = textArray;  // Text array for display
                trace.texttemplate = '%{{text}}';  // Use the text array
                trace.textfont = {{
                    size: Math.max(8, Math.min(16, currentFontSize * 0.8))
                }};
                // For hover, use the detailed hover text
                trace.hovertext = currentHoverText;
                trace.hoverinfo = 'text';
                trace.hovertemplate = '%{{hovertext}}<extra></extra>';  // <extra></extra> hides "trace 0"
            }} else {{
                // Hide cell values: no texttemplate, only hover text
                console.log('NOT setting texttemplate - hiding cell values');
                trace.text = currentHoverText;  // Text for hover only
                trace.hoverinfo = 'text';  // Show hover text on hover
                trace.hovertemplate = '%{{text}}<extra></extra>';  // <extra></extra> hides "trace 0"
            }}
            console.log('trace texttemplate:', trace.texttemplate);
            console.log('trace text sample:', trace.text ? trace.text[0] : 'none');
            
            // Apply custom colorbar range
            // Priority: 1) Custom color range (for cross-heatmap comparison)
            //           2) Manual slider range (if not auto)
            //           3) Auto range (default)
            if (window.customColorRange) {{
                trace.zmin = window.customColorRange.min;
                trace.zmax = window.customColorRange.max;
                console.log('Using custom color range:', window.customColorRange);
            }} else if (!useAutoRange && customZmin !== null && customZmax !== null) {{
                trace.zmin = customZmin;
                trace.zmax = customZmax;
            }}
            
            // Store current range for slider scaling
            window.currentDataRange = range;
            
            // Update 2-point color value inputs to show current data range in auto mode
            if (!window.customColorRange) {{
                document.getElementById('valueMin2').value = formatValueDisplay(range.min);
                document.getElementById('valueMax2').value = formatValueDisplay(range.max);
            }}
            
            // Determine axis titles based on transpose state
            const xAxisLabel = isTransposed ? 'Source' : 'Target';
            const yAxisLabel = isTransposed ? 'Target' : 'Source';
            const xAxisCount = displayXLabels.length;
            const yAxisCount = displayYLabels.length;
            
            const layout = {{
                title: originalTitle,
                font: {{size: currentFontSize}},
                autosize: false,
                xaxis: {{
                    title: isLarge ? '<b>' + xAxisLabel + '</b> (' + xAxisCount + ' neurons)' : '<b>' + xAxisLabel + '</b>',
                    side: 'bottom',
                    titlefont: {{size: currentFontSize + 2, color: '#333333'}},
                    tickangle: displayXLabels.length > 1 ? -45 : 0,  // Always rotate when multiple labels
                    showticklabels: showLabels,
                    tickmode: 'array',  // Use explicit tick values
                    tickvals: displayXLabels.map((_, i) => i),  // Use indices as tick positions
                    ticktext: displayXLabels  // Use labels as tick text
                }},
                yaxis: {{
                    title: isLarge ? '<b>' + yAxisLabel + '</b> (' + yAxisCount + ' neurons)' : '<b>' + yAxisLabel + '</b>',
                    side: 'left',
                    titlefont: {{size: currentFontSize + 2, color: '#333333'}},
                    autorange: 'reversed',
                    showticklabels: showLabels,
                    tickmode: 'array',  // Use explicit tick values
                    tickvals: displayYLabels.map((_, i) => i),  // Use indices as tick positions
                    ticktext: displayYLabels  // Use labels as tick text
                }},
                hoverlabel: {{
                    bgcolor: 'white',
                    font_size: 12,
                    font_family: 'Arial'
                }},
                width: currentWidth,
                height: currentHeight,
                margin: {{l: 120, r: 40, b: 120, t: 100, pad: 4}}
            }};
            
            // Add compression hint in title for user awareness
            if (is_sparse && !use_scatter_mode) {{
                const sparsity_pct = Math.round(sparsity * 100);
                layout.title.text += `<br><sub style='color:#666;'>Matrix ${{
                    sparsity_pct
                }}% sparse - optimized for file size</sub>`;
            }}
            
            // For scatter mode, lock aspect ratio and adjust margins
            if (use_scatter_mode) {{
                layout.xaxis.constrain = 'domain';
                layout.yaxis.scaleanchor = 'x';
                layout.plot_bgcolor = 'white';
                layout.xaxis.showgrid = true;
                layout.yaxis.showgrid = true;
                layout.xaxis.gridcolor = 'rgba(0,0,0,0.1)';
                layout.yaxis.gridcolor = 'rgba(0,0,0,0.1)';
                layout.margin.t = 100;
                layout.margin.b = 120;
            }}
            
            // Create or update the heatmap using Plotly.react (handles both creation and updates)
            Plotly.react('heatmap', [trace], layout);
        }}
        
        function toggleClustering(mode) {{
            // Toggle between original and clustered ordering
            useClusteredOrder = (mode === 'clustered');
            
            // Update button states
            document.getElementById('btn-original').classList.toggle('active', mode === 'original');
            document.getElementById('btn-clustered').classList.toggle('active', mode === 'clustered');
            
            // Show/hide clustering method selector
            const methodSection = document.getElementById('clusteringMethodSection');
            if (methodSection) {{
                methodSection.style.display = (mode === 'clustered' && clusteringAvailable) ? 'block' : 'none';
            }}
            
            // If clustering is not available, show message and revert
            if (mode === 'clustered' && !clusteringAvailable) {{
                alert('Clustering is not available for this matrix. Using original order.');
                useClusteredOrder = false;
                document.getElementById('btn-original').classList.add('active');
                document.getElementById('btn-clustered').classList.remove('active');
                return;
            }}
            
            // Update data filter state (disables when clustering is active)
            applyDataFilter();
            
            // Recreate heatmap with new ordering
            createHeatmap();
        }}
        
        function updateClusteringMethod() {{
            // Get selected clustering method
            const methodSelect = document.getElementById('clusteringMethodSelect');
            currentClusteringMethod = methodSelect.value;
            
            console.log('Switching to clustering method:', currentClusteringMethod);
            
            // Update the heatmap with new clustering method
            if (useClusteredOrder) {{
                createHeatmap();
            }}
        }}
        
        function setScale(scale) {{
            currentScale = scale;
            
            // Update button states
            document.querySelectorAll('[id^="btn-"]').forEach(btn => {{
                btn.classList.remove('active');
            }});
            document.getElementById('btn-' + scale).classList.add('active');
            
            createHeatmap();
        }}
        
        function updateMetric() {{
            // Switch to the selected metric
            currentMetric = document.getElementById('metricSelect').value;
            console.log('Switching to metric:', currentMetric);
            
            // Update dataLinear with the new metric's data (always use original ordering)
            dataLinear = metricsData[currentMetric];
            
            // Clear cached transforms so they're recomputed for new metric
            cachedDataLog2 = null;
            cachedDataLog10 = null;
            cachedDataSqrt = null;
            
            // Recreate the heatmap with new metric data (clustering will be applied in createHeatmap)
            createHeatmap();
        }}
        
        function updateColorscale() {{
            currentColorscale = document.getElementById('colorscaleSelect').value;
            
            // If switching to Custom and no custom scale exists, create default
            if (currentColorscale === 'Custom' && !customColorScale) {{
                applyCustomColors();
            }}
            
            createHeatmap();
        }}
        
        function toggleCustomColorPanel() {{
            const panel = document.getElementById('customColorPanel');
            if (panel.style.display === 'none') {{
                panel.style.display = 'block';
            }} else {{
                panel.style.display = 'none';
            }}
        }}
        
        function toggle3PointScale() {{
            use3PointScale = document.getElementById('use3PointScale').checked;
            const twoPoint = document.getElementById('twoPointColors');
            const threePoint = document.getElementById('threePointColors');
            
            if (use3PointScale) {{
                twoPoint.style.display = 'none';
                threePoint.style.display = 'block';
                
                // Set default values based on current data range
                if (window.currentDataRange) {{
                    const range = window.currentDataRange;
                    const mid = (range.min + range.max) / 2;
                    document.getElementById('valueMin3').value = formatValueDisplay(range.min);
                    document.getElementById('valueMid3').value = formatValueDisplay(mid);
                    document.getElementById('valueMax3').value = formatValueDisplay(range.max);
                }}
            }} else {{
                twoPoint.style.display = 'block';
                threePoint.style.display = 'none';
            }}
        }}
        
        function rgbToPlotlyFormat(hex) {{
            // Convert hex color to RGB format for Plotly
            const r = parseInt(hex.slice(1, 3), 16);
            const g = parseInt(hex.slice(3, 5), 16);
            const b = parseInt(hex.slice(5, 7), 16);
            return `rgb(${{r}},${{g}},${{b}})`;
        }}
        
        // Helper function to compare two arrays for equality
        function arraysEqual(arr1, arr2) {{
            if (arr1.length !== arr2.length) return false;
            for (let i = 0; i < arr1.length; i++) {{
                if (arr1[i] !== arr2[i]) return false;
            }}
            return true;
        }}
        
        function formatValueDisplay(value) {{
            // Format number to remove trailing zeros and unnecessary decimal point
            // Examples: 0.000000 -> "0", 250.123456 -> "250.123456", 1.500000 -> "1.5"
            if (value === 0) return "0";
            const str = value.toFixed(6);
            // Remove trailing zeros and decimal point if not needed
            return str.replace(/\.?0+$/, '');
        }}
        
        function applyCustomColors() {{
            if (use3PointScale) {{
                // 3-point scale with custom value mapping
                const colorMin = document.getElementById('colorMin3').value;
                const colorMid = document.getElementById('colorMid3').value;
                const colorMax = document.getElementById('colorMax3').value;
                
                const valueMin = parseFloat(document.getElementById('valueMin3').value);
                const valueMid = parseFloat(document.getElementById('valueMid3').value);
                const valueMax = parseFloat(document.getElementById('valueMax3').value);
                
                // Get current data range
                const range = window.currentDataRange;
                if (!range) {{
                    alert('Please wait for data to load before applying custom colors.');
                    return;
                }}
                
                // Map custom values to [0, 1] range - allows values beyond actual data range
                const normalizeValue = (val, rangeMin, rangeMax) => {{
                    if (rangeMax === rangeMin) return 0.5;
                    return (val - rangeMin) / (rangeMax - rangeMin);
                }};
                
                // Use custom value range for normalization (allows cross-heatmap comparison)
                const customRangeMin = valueMin;
                const customRangeMax = valueMax;
                
                if (customRangeMax === customRangeMin) {{
                    alert('Custom min and max values cannot be the same.');
                    return;
                }}
                
                // Map custom value points to [0, 1] colorscale positions
                // This defines where each color appears on the scale
                const posMid = normalizeValue(valueMid, customRangeMin, customRangeMax);
                
                // Clamp mid position to [0, 1]
                const clampedPosMid = Math.max(0, Math.min(1, posMid));
                
                // Create color scale array spanning 0 to 1
                // Plotly will map data values to this scale based on customColorRange
                customColorScale = [
                    [0, rgbToPlotlyFormat(colorMin)],
                    [clampedPosMid, rgbToPlotlyFormat(colorMid)],
                    [1, rgbToPlotlyFormat(colorMax)]
                ];
                
                // Set custom range for Plotly to use
                window.customColorRange = {{min: valueMin, max: valueMax}};
                
                // Sort by position (required by Plotly)
                customColorScale.sort((a, b) => a[0] - b[0]);
                
                // Ensure positions are distinct (avoid duplicates)
                const epsilon = 0.001;
                for (let i = 1; i < customColorScale.length; i++) {{
                    if (Math.abs(customColorScale[i][0] - customColorScale[i-1][0]) < epsilon) {{
                        customColorScale[i][0] = customColorScale[i-1][0] + epsilon;
                    }}
                }}
                
                console.log('Applied 3-point scale:', {{
                    inputs: {{
                        min: {{value: valueMin, color: colorMin}},
                        mid: {{value: valueMid, color: colorMid}},
                        max: {{value: valueMax, color: colorMax}}
                    }},
                    customRange: {{min: valueMin, max: valueMax}},
                    midPosition: clampedPosMid,
                    colorScale: customColorScale
                }});
            }} else {{
                // 2-point scale with optional custom value mapping
                const colorMin = document.getElementById('colorMin').value;
                const colorMax = document.getElementById('colorMax').value;
                
                const valueMin2Input = document.getElementById('valueMin2').value;
                const valueMax2Input = document.getElementById('valueMax2').value;
                
                // Check if custom values are specified
                if (valueMin2Input !== '' && valueMax2Input !== '') {{
                    // Use custom values for cross-heatmap comparison
                    const valueMin = parseFloat(valueMin2Input);
                    const valueMax = parseFloat(valueMax2Input);
                    
                    if (valueMax === valueMin) {{
                        alert('Custom min and max values cannot be the same.');
                        return;
                    }}
                    
                    // Colorscale spans from 0 to 1 (representing valueMin to valueMax)
                    // Plotly will map data values to this scale automatically
                    customColorScale = [
                        [0, rgbToPlotlyFormat(colorMin)],
                        [1, rgbToPlotlyFormat(colorMax)]
                    ];
                    
                    // Override the data normalization by setting colorscale range
                    window.customColorRange = {{min: valueMin, max: valueMax}};
                    
                    console.log('Applied 2-point scale with custom values:', {{
                        customRange: {{min: valueMin, max: valueMax}},
                        colorScale: customColorScale
                    }});
                }} else {{
                    // Auto mode: use full data range
                    customColorScale = [
                        [0, rgbToPlotlyFormat(colorMin)],
                        [1, rgbToPlotlyFormat(colorMax)]
                    ];
                    
                    // Clear custom range
                    window.customColorRange = null;
                    
                    console.log('Applied 2-point scale (auto):', customColorScale);
                }}
            }}
            
            // Switch to Custom colorscale and update
            currentColorscale = 'Custom';
            
            // Update dropdown without triggering the onchange handler
            const selectElement = document.getElementById('colorscaleSelect');
            const oldOnchange = selectElement.onchange;
            selectElement.onchange = null;
            selectElement.value = 'Custom';
            selectElement.onchange = oldOnchange;
            
            console.log('About to create heatmap with custom scale:', {{
                currentColorscale: currentColorscale,
                customColorScale: customColorScale,
                dropdownValue: selectElement.value
            }});
            
            createHeatmap();
        }}
        
        function resetToAutoColors() {{
            // Clear custom color range
            window.customColorRange = null;
            
            // Update value input boxes to show current data range
            const range = window.currentDataRange;
            if (range) {{
                document.getElementById('valueMin2').value = formatValueDisplay(range.min);
                document.getElementById('valueMax2').value = formatValueDisplay(range.max);
            }}
            
            // Recreate heatmap with auto colors
            createHeatmap();
            
            console.log('Reset to auto color mode');
        }}
        
        function updateFontSize(size) {{
            currentFontSize = parseInt(size);
            document.getElementById('fontSizeValue').textContent = size + 'px';
            createHeatmap();
        }}
        
        function toggleLabels() {{
            showLabels = !showLabels;
            const btn = document.getElementById('toggleLabelsBtn');
            btn.textContent = showLabels ? '🏷️ Hide Text' : '🏷️ Show Text';
            
            // Update the layout to hide/show ALL text elements including colorbar
            const gd = document.getElementById('heatmap');
            
            // Update colorbar text (trace-level property)
            const traceUpdate = {{
                'colorbar.title.text': showLabels ? (metricType.charAt(0).toUpperCase() + metricType.slice(1)) : '',
                'colorbar.showticklabels': showLabels
            }};
            
            // Update layout elements
            const layoutUpdate = {{
                'title.text': showLabels ? originalTitle : '',
                'xaxis.showticklabels': showLabels,
                'yaxis.showticklabels': showLabels,
                'xaxis.title.text': showLabels ? (isLarge ? '<b>Target</b> (' + gd.data[0].x.length + ' neurons)' : '<b>Target</b>') : '',
                'yaxis.title.text': showLabels ? (isLarge ? '<b>Source</b> (' + gd.data[0].y.length + ' neurons)' : '<b>Source</b>') : '',
                'xaxis.ticks': showLabels ? 'outside' : '',
                'yaxis.ticks': showLabels ? 'outside' : '',
                // Prevent autosize from expanding the plot
                'autosize': false,
                // Keep margins fixed to prevent rescaling
                'margin.l': 120,
                'margin.r': 40,
                'margin.t': 100,
                'margin.b': 120,
                // Preserve dimensions explicitly
                'width': currentWidth,
                'height': currentHeight
            }};
            
            // Add compression hint in title for user awareness
            if (is_sparse && !use_scatter_mode) {{
                const sparsity_pct = Math.round(sparsity * 100);
                layout.title.text += `<br><sub style='color:#666;'>Matrix ${{
                    sparsity_pct
                }}% sparse - optimized for file size</sub>`;
            }}
            
            // For scatter mode, lock aspect ratio and adjust margins
            if (use_scatter_mode) {{
                layout.xaxis.constrain = 'domain';
                layout.yaxis.scaleanchor = 'x';
                layout.plot_bgcolor = 'white';
                layout.xaxis.showgrid = true;
                layout.yaxis.showgrid = true;
                layout.xaxis.gridcolor = 'rgba(0,0,0,0.1)';
                layout.yaxis.gridcolor = 'rgba(0,0,0,0.1)';
                layout.margin.t = 100;
                layout.margin.b = 120;
            }}
            
            // Create or update the heatmap using Plotly.react (handles both creation and updates)
            Plotly.react('heatmap', [trace], layout);
        }}
        
        function toggleCellValues() {{
            showCellValues = !showCellValues;
            const btn = document.getElementById('toggleCellValuesBtn');
            btn.textContent = showCellValues ? '🔢 Hide Values' : '🔢 Show Values';
            
            console.log('toggleCellValues called, showCellValues is now:', showCellValues);
            
            // Recreate heatmap to add/remove cell value annotations
            createHeatmap();
        }}
        
        function updateCellValueSize(size) {{
            cellValueFontSize = parseInt(size);
            document.getElementById('cellValueSizeValue').textContent = cellValueFontSize + 'px';
            
            // Only recreate if cell values are currently shown
            if (showCellValues) {{
                createHeatmap();
            }}
        }}
        
        function updateContrastThreshold(value) {{
            contrastThreshold = parseFloat(value);
            document.getElementById('contrastThresholdValue').textContent = contrastThreshold.toFixed(4);
            console.log('Contrast threshold updated to:', contrastThreshold);
            
            // Recreate heatmap if cell values are currently shown
            if (showCellValues) {{
                createHeatmap();
            }}
        }}
        
        function reverseContrastColors() {{
            reverseContrast = !reverseContrast;
            console.log('Contrast colors reversed:', reverseContrast);
            
            // Recreate heatmap if cell values are currently shown
            if (showCellValues) {{
                createHeatmap();
            }}
        }}
        
        function updateIgnoredValues() {{
            const input = document.getElementById('ignoreValuesInput');
            const expressions = input.value.split(',').map(v => v.trim()).filter(v => v !== '');
            
            // Clear and repopulate the ignored values array
            // Store both exact values and comparison expressions
            ignoredValues.clear();
            ignoredValues.expressions = [];  // Array to store comparison expressions
            
            expressions.forEach(expr => {{
                // Check if it's a comparison expression (>, <, >=, <=)
                const compMatch = expr.match(/^([><]=?|==|!=)\\s*(-?\\d+\\.?\\d*)$/);
                if (compMatch) {{
                    // It's a comparison expression
                    const operator = compMatch[1];
                    const threshold = parseFloat(compMatch[2]);
                    ignoredValues.expressions.push({{ operator, threshold }});
                }} else {{
                    // Try to parse as exact number
                    const num = parseFloat(expr);
                    if (!isNaN(num)) {{
                        ignoredValues.add(num);
                    }}
                }}
            }});
            
            console.log('Ignored exact values:', Array.from(ignoredValues));
            console.log('Ignored expressions:', ignoredValues.expressions);
            
            // Recreate heatmap if cell values are shown
            if (showCellValues) {{
                createHeatmap();
            }}
        }}
        
        function shouldIgnoreValue(value) {{
            // Check if value matches any exact value
            if (ignoredValues.has(value)) {{
                return true;
            }}
            
            // Check if value matches any comparison expression
            if (ignoredValues.expressions && ignoredValues.expressions.length > 0) {{
                for (const expr of ignoredValues.expressions) {{
                    let matches = false;
                    switch (expr.operator) {{
                        case '>':
                            matches = value > expr.threshold;
                            break;
                        case '<':
                            matches = value < expr.threshold;
                            break;
                        case '>=':
                            matches = value >= expr.threshold;
                            break;
                        case '<=':
                            matches = value <= expr.threshold;
                            break;
                        case '==':
                            matches = value === expr.threshold;
                            break;
                        case '!=':
                            matches = value !== expr.threshold;
                            break;
                    }}
                    if (matches) {{
                        return true;
                    }}
                }}
            }}
            
            return false;
        }}
        
        // ===== DATA FILTER FUNCTIONS =====
        // Filter entire rows/columns based on their maximum values
        
        function parseFilterExpressions(inputString) {{
            const expressions = inputString.split(',').map(v => v.trim()).filter(v => v !== '');
            const parsedExpressions = [];
            
            expressions.forEach(expr => {{
                // Check if it's a comparison expression (>, <, >=, <=, ==, !=)
                const compMatch = expr.match(/^([><]=?|==|!=)\\s*(-?\\d+\\.?\\d*)$/);
                if (compMatch) {{
                    const operator = compMatch[1];
                    const threshold = parseFloat(compMatch[2]);
                    parsedExpressions.push({{ operator, threshold }});
                }} else {{
                    // Try to parse as exact number (will hide if max == this value)
                    const num = parseFloat(expr);
                    if (!isNaN(num)) {{
                        parsedExpressions.push({{ operator: '==', threshold: num }});
                    }}
                }}
            }});
            
            return parsedExpressions;
        }}
        
        function shouldHideRowOrColumn(maxValue, expressions) {{
            if (expressions.length === 0) return false;
            
            for (const expr of expressions) {{
                let matches = false;
                switch (expr.operator) {{
                    case '>':
                        matches = maxValue > expr.threshold;
                        break;
                    case '<':
                        matches = maxValue < expr.threshold;
                        break;
                    case '>=':
                        matches = maxValue >= expr.threshold;
                        break;
                    case '<=':
                        matches = maxValue <= expr.threshold;
                        break;
                    case '==':
                        matches = maxValue === expr.threshold;
                        break;
                    case '!=':
                        matches = maxValue !== expr.threshold;
                        break;
                }}
                if (matches) {{
                    return true;  // Hide if any expression matches
                }}
            }}
            
            return false;
        }}
        
        function applyDataFilter() {{
            const input = document.getElementById('dataFilterInput');
            const statusDiv = document.getElementById('filterStatus');
            const filterValue = input.value.trim();
            
            // Disable data filtering when clustering is active
            if (useClusteredOrder && clusteringAvailable) {{
                statusDiv.textContent = '⚠️ Data filter disabled during clustering';
                statusDiv.style.color = '#ff9800';
                input.disabled = true;
                dataFilterActive = false;
                dataFilterExpressions = [];
                filteredRowIndices = [];
                filteredColIndices = [];
                return;
            }} else {{
                input.disabled = false;
            }}
            
            if (!filterValue) {{
                // No filter - show all rows/columns
                dataFilterActive = false;
                dataFilterExpressions = [];
                filteredRowIndices = [];
                filteredColIndices = [];
                statusDiv.textContent = '';
                createHeatmap();
                return;
            }}
            
            // Parse filter expressions
            dataFilterExpressions = parseFilterExpressions(filterValue);
            
            if (dataFilterExpressions.length === 0) {{
                statusDiv.textContent = '⚠️ Invalid filter format';
                statusDiv.style.color = '#d32f2f';
                return;
            }}
            
            // Get current data based on scale
            let currentData = dataLinear;
            if (currentScale === 'log2' && (useLazyTransforms ? cachedDataLog2 : dataLog2)) {{
                currentData = useLazyTransforms ? cachedDataLog2 : dataLog2;
            }} else if (currentScale === 'log10' && (useLazyTransforms ? cachedDataLog10 : dataLog10)) {{
                currentData = useLazyTransforms ? cachedDataLog10 : dataLog10;
            }} else if (currentScale === 'sqrt' && (useLazyTransforms ? cachedDataSqrt : dataSqrt)) {{
                currentData = useLazyTransforms ? cachedDataSqrt : dataSqrt;
            }}
            
            // Use original unscaled data for filtering
            const filterData = metricsData[currentMetric];
            
            const nRows = filterData.length;
            const nCols = filterData[0].length;
            
            // Calculate max value for each row and column
            const rowMaxValues = new Array(nRows).fill(-Infinity);
            const colMaxValues = new Array(nCols).fill(-Infinity);
            
            for (let i = 0; i < nRows; i++) {{
                for (let j = 0; j < nCols; j++) {{
                    const value = filterData[i][j];
                    if (value > rowMaxValues[i]) rowMaxValues[i] = value;
                    if (value > colMaxValues[j]) colMaxValues[j] = value;
                }}
            }}
            
            // Determine which rows and columns to keep
            filteredRowIndices = [];
            filteredColIndices = [];
            
            for (let i = 0; i < nRows; i++) {{
                if (!shouldHideRowOrColumn(rowMaxValues[i], dataFilterExpressions)) {{
                    filteredRowIndices.push(i);
                }}
            }}
            
            for (let j = 0; j < nCols; j++) {{
                if (!shouldHideRowOrColumn(colMaxValues[j], dataFilterExpressions)) {{
                    filteredColIndices.push(j);
                }}
            }}
            
            dataFilterActive = true;
            
            const hiddenRows = nRows - filteredRowIndices.length;
            const hiddenCols = nCols - filteredColIndices.length;
            
            if (filteredRowIndices.length === 0 || filteredColIndices.length === 0) {{
                statusDiv.textContent = '⚠️ Filter hides all data!';
                statusDiv.style.color = '#d32f2f';
                dataFilterActive = false;
                return;
            }}
            
            statusDiv.textContent = `✓ Showing ${{filteredRowIndices.length}}/${{nRows}} rows, ${{filteredColIndices.length}}/${{nCols}} cols`;
            statusDiv.style.color = '#2e7d32';
            
            console.log(`Data filter applied: hiding ${{hiddenRows}} rows and ${{hiddenCols}} cols`);
            
            createHeatmap();
        }}
        
        function resetDataFilter() {{
            document.getElementById('dataFilterInput').value = '';
            document.getElementById('filterStatus').textContent = '';
            dataFilterActive = false;
            dataFilterExpressions = [];
            filteredRowIndices = [];
            filteredColIndices = [];
            createHeatmap();
        }}
        
        // ===== END DATA FILTER FUNCTIONS =====
        
        function getPlotlyColorscaleArray(scaleName) {{
            // Return colorscale array for Plotly heatmap
            // Plotly v1.58.5 doesn't recognize all colorscale names, so we define them as arrays
            const colorscales = {{
                'Greens': [
                    [0.0, 'rgb(247,252,245)'],
                    [0.125, 'rgb(229,245,224)'],
                    [0.25, 'rgb(199,233,192)'],
                    [0.375, 'rgb(161,217,155)'],
                    [0.5, 'rgb(116,196,118)'],
                    [0.625, 'rgb(65,171,93)'],
                    [0.75, 'rgb(35,139,69)'],
                    [0.875, 'rgb(0,109,44)'],
                    [1.0, 'rgb(0,68,27)']
                ],
                'Blues': [
                    [0.0, 'rgb(247,251,255)'],
                    [0.125, 'rgb(222,235,247)'],
                    [0.25, 'rgb(198,219,239)'],
                    [0.375, 'rgb(158,202,225)'],
                    [0.5, 'rgb(107,174,214)'],
                    [0.625, 'rgb(66,146,198)'],
                    [0.75, 'rgb(33,113,181)'],
                    [0.875, 'rgb(8,81,156)'],
                    [1.0, 'rgb(8,48,107)']
                ],
                'Reds': [
                    [0.0, 'rgb(255,245,240)'],
                    [0.125, 'rgb(254,224,210)'],
                    [0.25, 'rgb(252,187,161)'],
                    [0.375, 'rgb(252,146,114)'],
                    [0.5, 'rgb(251,106,74)'],
                    [0.625, 'rgb(239,59,44)'],
                    [0.75, 'rgb(203,24,29)'],
                    [0.875, 'rgb(165,15,21)'],
                    [1.0, 'rgb(103,0,13)']
                ],
                'Purples': [
                    [0.0, 'rgb(252,251,253)'],
                    [0.125, 'rgb(239,237,245)'],
                    [0.25, 'rgb(218,218,235)'],
                    [0.375, 'rgb(188,189,220)'],
                    [0.5, 'rgb(158,154,200)'],
                    [0.625, 'rgb(128,125,186)'],
                    [0.75, 'rgb(106,81,163)'],
                    [0.875, 'rgb(84,39,143)'],
                    [1.0, 'rgb(63,0,125)']
                ],
                'Oranges': [
                    [0.0, 'rgb(255,245,235)'],
                    [0.125, 'rgb(254,230,206)'],
                    [0.25, 'rgb(253,208,162)'],
                    [0.375, 'rgb(253,174,107)'],
                    [0.5, 'rgb(253,141,60)'],
                    [0.625, 'rgb(241,105,19)'],
                    [0.75, 'rgb(217,72,1)'],
                    [0.875, 'rgb(166,54,3)'],
                    [1.0, 'rgb(127,39,4)']
                ],
                'Viridis': [
                    [0, 'rgb(68,1,84)'],
                    [0.25, 'rgb(59,82,139)'],
                    [0.5, 'rgb(33,145,140)'],
                    [0.75, 'rgb(94,201,98)'],
                    [1, 'rgb(253,231,37)']
                ],
                'Plasma': [
                    [0, 'rgb(13,8,135)'],
                    [0.25, 'rgb(126,3,168)'],
                    [0.5, 'rgb(204,71,120)'],
                    [0.75, 'rgb(248,149,64)'],
                    [1, 'rgb(240,249,33)']
                ],
                'Inferno': [
                    [0, 'rgb(0,0,4)'],
                    [0.25, 'rgb(87,16,110)'],
                    [0.5, 'rgb(188,55,84)'],
                    [0.75, 'rgb(249,142,9)'],
                    [1, 'rgb(252,255,164)']
                ],
                'Magma': [
                    [0, 'rgb(0,0,4)'],
                    [0.25, 'rgb(81,18,124)'],
                    [0.5, 'rgb(182,54,121)'],
                    [0.75, 'rgb(251,136,97)'],
                    [1, 'rgb(252,253,191)']
                ],
                'Cividis': [
                    [0, 'rgb(0,32,76)'],
                    [0.25, 'rgb(0,79,110)'],
                    [0.5, 'rgb(53,133,136)'],
                    [0.75, 'rgb(149,189,161)'],
                    [1, 'rgb(253,231,37)']
                ],
                'Hot': [
                    [0, 'rgb(0,0,0)'],
                    [0.33, 'rgb(255,0,0)'],
                    [0.66, 'rgb(255,255,0)'],
                    [1, 'rgb(255,255,255)']
                ],
                'Jet': [
                    [0, 'rgb(0,0,143)'],
                    [0.25, 'rgb(0,159,255)'],
                    [0.5, 'rgb(0,255,0)'],
                    [0.75, 'rgb(255,159,0)'],
                    [1, 'rgb(143,0,0)']
                ],
                'RdBu': [
                    [0, 'rgb(5,10,172)'],
                    [0.35, 'rgb(106,137,247)'],
                    [0.5, 'rgb(190,190,190)'],
                    [0.65, 'rgb(220,170,132)'],
                    [1, 'rgb(178,10,28)']
                ],
                'RdYlGn': [
                    [0, 'rgb(165,0,38)'],
                    [0.25, 'rgb(253,174,97)'],
                    [0.5, 'rgb(255,255,191)'],
                    [0.75, 'rgb(166,217,106)'],
                    [1, 'rgb(0,104,55)']
                ]
            }};
            
            // Return the colorscale array, or fallback to the name string
            return colorscales[scaleName] || scaleName;
        }}
        
        function getColorFromPlotlyScale(scaleName, normalized) {{
            // Map of Plotly colorscales to their RGB interpolations
            // These are approximations of Plotly's built-in scales
            const colorscales = {{
                'Greens': [
                    [0.0, 'rgb(247,252,245)'],
                    [0.125, 'rgb(229,245,224)'],
                    [0.25, 'rgb(199,233,192)'],
                    [0.375, 'rgb(161,217,155)'],
                    [0.5, 'rgb(116,196,118)'],
                    [0.625, 'rgb(65,171,93)'],
                    [0.75, 'rgb(35,139,69)'],
                    [0.875, 'rgb(0,109,44)'],
                    [1.0, 'rgb(0,68,27)']
                ],
                'Blues': [
                    [0.0, 'rgb(247,251,255)'],
                    [0.125, 'rgb(222,235,247)'],
                    [0.25, 'rgb(198,219,239)'],
                    [0.375, 'rgb(158,202,225)'],
                    [0.5, 'rgb(107,174,214)'],
                    [0.625, 'rgb(66,146,198)'],
                    [0.75, 'rgb(33,113,181)'],
                    [0.875, 'rgb(8,81,156)'],
                    [1.0, 'rgb(8,48,107)']
                ],
                'Reds': [
                    [0.0, 'rgb(255,245,240)'],
                    [0.125, 'rgb(254,224,210)'],
                    [0.25, 'rgb(252,187,161)'],
                    [0.375, 'rgb(252,146,114)'],
                    [0.5, 'rgb(251,106,74)'],
                    [0.625, 'rgb(239,59,44)'],
                    [0.75, 'rgb(203,24,29)'],
                    [0.875, 'rgb(165,15,21)'],
                    [1.0, 'rgb(103,0,13)']
                ],
                'Purples': [
                    [0.0, 'rgb(252,251,253)'],
                    [0.125, 'rgb(239,237,245)'],
                    [0.25, 'rgb(218,218,235)'],
                    [0.375, 'rgb(188,189,220)'],
                    [0.5, 'rgb(158,154,200)'],
                    [0.625, 'rgb(128,125,186)'],
                    [0.75, 'rgb(106,81,163)'],
                    [0.875, 'rgb(84,39,143)'],
                    [1.0, 'rgb(63,0,125)']
                ],
                'Oranges': [
                    [0.0, 'rgb(255,245,235)'],
                    [0.125, 'rgb(254,230,206)'],
                    [0.25, 'rgb(253,208,162)'],
                    [0.375, 'rgb(253,174,107)'],
                    [0.5, 'rgb(253,141,60)'],
                    [0.625, 'rgb(241,105,19)'],
                    [0.75, 'rgb(217,72,1)'],
                    [0.875, 'rgb(166,54,3)'],
                    [1.0, 'rgb(127,39,4)']
                ],
                'Viridis': [
                    [0, 'rgb(68,1,84)'],
                    [0.25, 'rgb(59,82,139)'],
                    [0.5, 'rgb(33,145,140)'],
                    [0.75, 'rgb(94,201,98)'],
                    [1, 'rgb(253,231,37)']
                ],
                'Plasma': [
                    [0, 'rgb(13,8,135)'],
                    [0.25, 'rgb(126,3,168)'],
                    [0.5, 'rgb(204,71,120)'],
                    [0.75, 'rgb(248,149,64)'],
                    [1, 'rgb(240,249,33)']
                ],
                'Inferno': [
                    [0, 'rgb(0,0,4)'],
                    [0.25, 'rgb(87,16,110)'],
                    [0.5, 'rgb(188,55,84)'],
                    [0.75, 'rgb(249,142,9)'],
                    [1, 'rgb(252,255,164)']
                ],
                'Magma': [
                    [0, 'rgb(0,0,4)'],
                    [0.25, 'rgb(81,18,124)'],
                    [0.5, 'rgb(182,54,121)'],
                    [0.75, 'rgb(251,136,97)'],
                    [1, 'rgb(252,253,191)']
                ],
                'Cividis': [
                    [0, 'rgb(0,32,76)'],
                    [0.25, 'rgb(0,79,110)'],
                    [0.5, 'rgb(53,133,136)'],
                    [0.75, 'rgb(149,189,161)'],
                    [1, 'rgb(253,231,37)']
                ],
                'Hot': [
                    [0, 'rgb(0,0,0)'],
                    [0.33, 'rgb(255,0,0)'],
                    [0.66, 'rgb(255,255,0)'],
                    [1, 'rgb(255,255,255)']
                ],
                'Jet': [
                    [0, 'rgb(0,0,143)'],
                    [0.25, 'rgb(0,159,255)'],
                    [0.5, 'rgb(0,255,0)'],
                    [0.75, 'rgb(255,159,0)'],
                    [1, 'rgb(143,0,0)']
                ],
                'RdBu': [
                    [0, 'rgb(5,10,172)'],
                    [0.35, 'rgb(106,137,247)'],
                    [0.5, 'rgb(190,190,190)'],
                    [0.65, 'rgb(220,170,132)'],
                    [1, 'rgb(178,10,28)']
                ],
                'RdYlGn': [
                    [0, 'rgb(165,0,38)'],
                    [0.25, 'rgb(253,174,97)'],
                    [0.5, 'rgb(255,255,191)'],
                    [0.75, 'rgb(166,217,106)'],
                    [1, 'rgb(0,104,55)']
                ]
            }};
            
            // Interpolate color from a custom colorscale array
            // colorscale format: [[0, 'color1'], [0.5, 'color2'], [1, 'color3'], ...]
            
            if (!Array.isArray(colorscale) || colorscale.length === 0) {{
                return 'rgb(128, 128, 128)';  // fallback gray
            }}
            
            // Handle edge cases
            if (normalized <= 0 || normalized <= colorscale[0][0]) {{
                return Array.isArray(colorscale[0]) && colorscale[0].length > 1 ? colorscale[0][1] : 'rgb(128, 128, 128)';
            }} else if (normalized >= 1 || normalized >= colorscale[colorscale.length - 1][0]) {{
                const last = colorscale[colorscale.length - 1];
                return Array.isArray(last) && last.length > 1 ? last[1] : 'rgb(128, 128, 128)';
            }}
            
            // Find the two color stops to interpolate between
            let lower = colorscale[0];
            let upper = colorscale[colorscale.length - 1];
            
            for (let i = 0; i < colorscale.length - 1; i++) {{
                if (normalized >= colorscale[i][0] && normalized <= colorscale[i + 1][0]) {{
                    lower = colorscale[i];
                    upper = colorscale[i + 1];
                    break;
                }}
            }}
            
            // Interpolate between the two colors
            const t = (normalized - lower[0]) / (upper[0] - lower[0]);
            const lowerRgb = hexToRgb(lower[1]);
            const upperRgb = hexToRgb(upper[1]);
            
            const r = Math.round(lowerRgb[0] + t * (upperRgb[0] - lowerRgb[0]));
            const g = Math.round(lowerRgb[1] + t * (upperRgb[1] - lowerRgb[1]));
            const b = Math.round(lowerRgb[2] + t * (upperRgb[2] - lowerRgb[2]));
            
            return `rgb(${{r}},${{g}},${{b}})`;
        }}
        
        function getContrastColor(rgb) {{
            // Calculate luminance from RGB color
            // If luminance is high (light background), use dark text; otherwise use light text
            const r = rgb[0];
            const g = rgb[1];
            const b = rgb[2];
            
            // Calculate relative luminance using the formula for sRGB
            const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            
            // Convert normalized threshold (0-1) to 0-255 range for comparison
            const threshold = contrastThreshold * 255;
            
            // Compare against the adjustable threshold
            // Normal: high luminance (light bg) → black text, low luminance (dark bg) → white text
            // Reverse: swap the logic
            if (reverseContrast) {{
                return luminance > threshold ? 'white' : 'black';
            }} else {{
                return luminance > threshold ? 'black' : 'white';
            }}
        }}
        
        function getColorForValue(value, zmin, zmax, colorscale) {{
            // Normalize value to 0-1 range
            const normalized = (value - zmin) / (zmax - zmin);
            
            // Get RGB color from the colorscale at the normalized position
            // This is a simplified version - Plotly has complex colorscale interpolation
            // For now, we'll sample the colorscale array
            if (Array.isArray(colorscale) && colorscale.length > 0) {{
                const idx = Math.floor(normalized * (colorscale.length - 1));
                const colorStop = colorscale[Math.max(0, Math.min(idx, colorscale.length - 1))];
                if (Array.isArray(colorStop) && colorStop.length > 1) {{
                    return colorStop[1];
                }}
            }}
            
            // Fallback: return a color based on normalized value
            if (normalized < 0.5) {{
                return `rgb(${{Math.round(normalized * 510)}}, ${{Math.round(normalized * 510)}}, 255)`;
            }} else {{
                return `rgb(255, ${{Math.round((1 - normalized) * 510)}}, ${{Math.round((1 - normalized) * 510)}})`;
            }}
        }}
        
        function hexToRgb(hex) {{
            // Convert hex color to RGB array
            if (hex.startsWith('#')) {{
                const result = /^#?([a-f\\d]{{2}})([a-f\\d]{{2}})([a-f\\d]{{2}})$/i.exec(hex);
                return result ? [
                    parseInt(result[1], 16),
                    parseInt(result[2], 16),
                    parseInt(result[3], 16)
                ] : [128, 128, 128];
            }} else if (hex.startsWith('rgb')) {{
                const match = hex.match(/\\d+/g);
                return match ? match.slice(0, 3).map(Number) : [128, 128, 128];
            }}
            return [128, 128, 128];
        }}
        
        function updatePlotSize() {{
            const gd = document.getElementById('heatmap');
            currentWidth = parseInt(document.getElementById('widthSlider').value);
            
            // If square cells are locked, auto-adjust height
            if (squareCellsLocked && gd.data && gd.data[0]) {{
                const numRows = gd.data[0].y.length;
                const numCols = gd.data[0].x.length;
                const margins = gd.layout.margin || {{l: 120, r: 40, b: 120, t: 100}};
                const marginHorizontal = margins.l + margins.r;
                const marginVertical = margins.t + margins.b;
                const plotAreaWidth = currentWidth - marginHorizontal;
                const plotAreaHeight = plotAreaWidth * numRows / numCols;
                currentHeight = Math.round(plotAreaHeight + marginVertical);
            }} else {{
                currentHeight = parseInt(document.getElementById('heightSlider').value);
            }}
            
            // Sync input boxes with sliders
            document.getElementById('widthInput').value = currentWidth;
            document.getElementById('heightInput').value = currentHeight;
            document.getElementById('widthValue').textContent = currentWidth + 'px';
            document.getElementById('heightValue').textContent = currentHeight + 'px';
            document.getElementById('heightSlider').value = Math.min(2400, Math.max(400, currentHeight));
            
            // Update the layout without recreating the entire plot
            Plotly.relayout(gd, {{
                width: currentWidth,
                height: currentHeight
            }});
        }}
        
        function updatePlotSizeFromInput() {{
            const gd = document.getElementById('heatmap');
            const widthInput = parseInt(document.getElementById('widthInput').value);
            
            // Update width
            currentWidth = widthInput;
            
            // If square cells are locked, auto-adjust height
            if (squareCellsLocked && gd.data && gd.data[0]) {{
                const numRows = gd.data[0].y.length;
                const numCols = gd.data[0].x.length;
                const margins = gd.layout.margin || {{l: 120, r: 40, b: 120, t: 100}};
                const marginHorizontal = margins.l + margins.r;
                const marginVertical = margins.t + margins.b;
                const plotAreaWidth = currentWidth - marginHorizontal;
                const plotAreaHeight = plotAreaWidth * numRows / numCols;
                currentHeight = Math.round(plotAreaHeight + marginVertical);
            }} else {{
                currentHeight = parseInt(document.getElementById('heightInput').value);
            }}
            
            // Update sliders (clamped to their range) and displays
            document.getElementById('widthSlider').value = Math.min(2400, Math.max(400, currentWidth));
            document.getElementById('heightSlider').value = Math.min(2400, Math.max(400, currentHeight));
            document.getElementById('widthInput').value = currentWidth;
            document.getElementById('heightInput').value = currentHeight;
            document.getElementById('widthValue').textContent = currentWidth + 'px';
            document.getElementById('heightValue').textContent = currentHeight + 'px';
            
            // Update the layout
            Plotly.relayout(gd, {{
                width: currentWidth,
                height: currentHeight
            }});
        }}
        
        function makeSquareCells() {{
            const gd = document.getElementById('heatmap');
            if (!gd.data || !gd.data[0]) return;
            
            const btn = document.getElementById('squareCellsBtn');
            squareCellsLocked = !squareCellsLocked;
            
            if (squareCellsLocked) {{
                // Lock to square cells
                const numRows = gd.data[0].y.length;
                const numCols = gd.data[0].x.length;
                
                // Get margins (l=120, r=40, b=120, t=100)
                const margins = gd.layout.margin || {{l: 120, r: 40, b: 120, t: 100}};
                const marginHorizontal = margins.l + margins.r;  // 160px
                const marginVertical = margins.t + margins.b;    // 220px
                
                // Calculate height for square cells based on current width
                const plotAreaWidth = currentWidth - marginHorizontal;
                const plotAreaHeight = plotAreaWidth * numRows / numCols;
                const targetHeight = Math.round(plotAreaHeight + marginVertical);
                
                // Update height
                currentHeight = targetHeight;
                document.getElementById('heightSlider').value = Math.min(2400, Math.max(400, targetHeight));
                document.getElementById('heightInput').value = targetHeight;
                document.getElementById('heightValue').textContent = targetHeight + 'px';
                
                // Lock aspect ratio
                Plotly.relayout(gd, {{
                    width: currentWidth,
                    height: targetHeight,
                    'xaxis.scaleanchor': 'y',
                    'xaxis.scaleratio': 1,
                    'yaxis.constrain': 'domain'
                }});
                
                btn.textContent = '🔓 Unlock Cells';
                btn.style.backgroundColor = '#28a745';
                
                console.log('Square cells LOCKED:', {{
                    numCols: numCols,
                    numRows: numRows,
                    width: currentWidth,
                    height: targetHeight,
                    cellAspectRatio: 1.0
                }});
            }} else {{
                // Unlock - remove aspect ratio constraint
                Plotly.relayout(gd, {{
                    'xaxis.scaleanchor': null,
                    'xaxis.scaleratio': null,
                    'yaxis.constrain': null
                }});
                
                btn.textContent = '⬜ Square Cells';
                btn.style.backgroundColor = '';
                
                console.log('Square cells UNLOCKED - free adjustment enabled');
            }}
        }}
        
        function transposeMatrix() {{
            isTransposed = !isTransposed;
            
            // Update button text
            const btn = document.getElementById('transposeBtn');
            btn.textContent = isTransposed ? '🔄 Restore Original' : '🔄 Swap Rows ↔ Columns';
            btn.style.backgroundColor = isTransposed ? '#17a2b8' : '';
            
            console.log('Matrix transposed:', isTransposed);
            
            // Recreate heatmap with transposed data
            createHeatmap();
        }}
        
        // Row/Column reordering functions
        function resetOrder() {{
            // Reset to original order (before any reordering operations)
            currentXLabels = xLabels.slice();
            currentYLabels = yLabels.slice();
            console.log('Reset to original order');
            closeOrderPanel();  // Close panel if open
            createHeatmap();
        }}
        
        // Drag and drop ordering
        let currentOrderType = null;  // 'rows' or 'cols'
        let draggedItem = null;
        let tempOrder = [];
        
        function toggleOrderPanel(type) {{
            currentOrderType = type;
            const panel = document.getElementById('orderPanel');
            const backdrop = document.getElementById('orderPanelBackdrop');
            const label = document.getElementById('orderPanelLabel');
            const listContainer = document.getElementById('orderList');
            
            // Get current labels based on type and transpose state
            // We need to show the ACTUAL order displayed on heatmap, including clustering
            let labels;
            if (type === 'rows') {{
                // Visual rows = Y-axis
                labels = isTransposed ? currentXLabels.slice() : currentYLabels.slice();
                
                // Apply clustering if enabled
                if (useClusteredOrder && clusteringAvailable) {{
                    const effectiveRowOrder = isTransposed ? colOrderClustered : rowOrderClusterled;
                    labels = reorderLabels(labels, effectiveRowOrder);
                }}
                label.textContent = 'Reorder Rows (Y-axis)';
            }} else {{
                // Visual columns = X-axis
                labels = isTransposed ? currentYLabels.slice() : currentXLabels.slice();
                
                // Apply clustering if enabled
                if (useClusteredOrder && clusteringAvailable) {{
                    const effectiveColOrder = isTransposed ? rowOrderClustered : colOrderClusterled;
                    labels = reorderLabels(labels, effectiveColOrder);
                }}
                label.textContent = 'Reorder Columns (X-axis)';
            }}
            
            tempOrder = labels.slice();
            console.log('toggleOrderPanel:', type, 'isTransposed:', isTransposed, 'clustered:', useClusteredOrder, 'labels:', labels);
            
            // Create draggable list
            listContainer.innerHTML = '';
            labels.forEach((item, index) => {{
                const div = document.createElement('div');
                div.className = 'drag-item';
                div.draggable = true;
                div.dataset.label = item;
                div.innerHTML = '<span class="drag-handle">☰</span>' + item;
                
                div.addEventListener('dragstart', handleDragStart);
                div.addEventListener('dragover', handleDragOver);
                div.addEventListener('drop', handleDrop);
                div.addEventListener('dragend', handleDragEnd);
                div.addEventListener('dragenter', handleDragEnter);
                div.addEventListener('dragleave', handleDragLeave);
                
                listContainer.appendChild(div);
            }});
            
            // Show panel and backdrop
            panel.style.display = 'flex';
            backdrop.style.display = 'block';
        }}
        
        function closeOrderPanel() {{
            document.getElementById('orderPanel').style.display = 'none';
            document.getElementById('orderPanelBackdrop').style.display = 'none';
            currentOrderType = null;
            draggedItem = null;
            tempOrder = [];
        }}
        
        function handleDragStart(e) {{
            draggedItem = this;
            this.classList.add('dragging');
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/html', this.innerHTML);
        }}
        
        function handleDragOver(e) {{
            if (e.preventDefault) {{
                e.preventDefault();
            }}
            e.dataTransfer.dropEffect = 'move';
            return false;
        }}
        
        function handleDragEnter(e) {{
            if (this !== draggedItem) {{
                this.classList.add('drag-over');
            }}
        }}
        
        function handleDragLeave(e) {{
            this.classList.remove('drag-over');
        }}
        
        function handleDrop(e) {{
            if (e.stopPropagation) {{
                e.stopPropagation();
            }}
            
            if (draggedItem !== this) {{
                // Reorder in DOM - insert before the target
                const draggedLabel = draggedItem.dataset.label;
                const targetLabel = this.dataset.label;
                
                const listContainer = document.getElementById('orderList');
                
                // Always insert before the target element
                // This gives consistent behavior: dropping on X puts item before X
                this.parentNode.insertBefore(draggedItem, this);
                
                // Read the new order from DOM to ensure perfect sync
                const itemsAfter = Array.from(listContainer.children);
                tempOrder = itemsAfter.map(item => item.dataset.label);
                
                console.log('Dragged', draggedLabel, 'before', targetLabel, '| New order:', tempOrder);
                
                // Apply immediately to heatmap
                applyReorderImmediate();
            }}
            
            this.classList.remove('drag-over');
            return false;
        }}
        
        function handleDragEnd(e) {{
            this.classList.remove('dragging');
            
            // Remove drag-over class from all items
            const items = document.querySelectorAll('.drag-item');
            items.forEach(item => item.classList.remove('drag-over'));
        }}
        
        function applyReorderImmediate() {{
            if (!currentOrderType || tempOrder.length === 0) return;
            
            // When user manually reorders, disable clustering to respect their choice
            if (useClusteredOrder) {{
                useClusteredOrder = false;
                const orderBtn = document.getElementById('orderBtn');
                if (orderBtn) {{
                    orderBtn.textContent = '🔀 Clustered Order';
                }}
                console.log('Disabled clustering due to manual reordering');
            }}
            
            if (currentOrderType === 'rows') {{
                if (isTransposed) {{
                    currentXLabels = tempOrder.slice();
                }} else {{
                    currentYLabels = tempOrder.slice();
                }}
                console.log('Applied immediate reorder to rows:', tempOrder);
            }} else {{
                if (isTransposed) {{
                    currentYLabels = tempOrder.slice();
                }} else {{
                    currentXLabels = tempOrder.slice();
                }}
                console.log('Applied immediate reorder to columns:', tempOrder);
            }}
            
            createHeatmap();
        }}
        
        function applyDragOrder() {{
            // Just close the panel - reordering already applied immediately
            closeOrderPanel();
        }}
        
        function resetPlotSize() {{
            currentWidth = 800;
            currentHeight = 800;
            document.getElementById('widthSlider').value = 800;
            document.getElementById('heightSlider').value = 800;
            document.getElementById('widthValue').textContent = '800px';
            document.getElementById('heightValue').textContent = '800px';
            document.getElementById('exportScaleSlider').value = 2;
            document.getElementById('exportScaleValue').textContent = '2x';
            
            // Reset custom color inputs
            document.getElementById('colorMin').value = '#ffffff';
            document.getElementById('colorMax').value = '#68379c';
            document.getElementById('colorMin3').value = '#0000ff';
            document.getElementById('colorMid3').value = '#ffffff';
            document.getElementById('colorMax3').value = '#ff0000';
            document.getElementById('use3PointScale').checked = false;
            toggle3PointScale();
            
            createHeatmap();
            showStatus('✅ Reset to defaults', 'success');
        }}
        
        function showStatus(message, type) {{
            console.log('showStatus called:', message, type);
            const statusDiv = document.getElementById('settingsStatus');
            console.log('statusDiv found:', statusDiv);
            if (!statusDiv) {{
                console.error('settingsStatus div not found!');
                return;
            }}
            statusDiv.innerHTML = '<div class="status-message status-' + type + '">' + message + '</div>';
            console.log('Status message displayed:', statusDiv.innerHTML);
            setTimeout(() => {{
                statusDiv.innerHTML = '';
            }}, 3000);
        }}
        
        // Try to load saved settings on page load
        window.addEventListener('load', () => {{
            const saved = localStorage.getItem(storageKey);
            if (saved) {{
                loadSettings(false);  // Silent load on initialization
            }} else {{
                createHeatmap();
            }}
        }});
    </script>
</body>
</html>
'''
    
    # Write HTML file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    if showfig:
        import webbrowser
        webbrowser.open('file://' + os.path.abspath(filename))


def split_path(path_df):
    """
    Convert path list to string representation.
    """
    if path_df.empty:
        return path_df
    
    if 'path' in path_df.columns:
        # If path_str already exists, preserve it (it might contain the original list)
        if 'path_str' in path_df.columns:
            # Ensure path is string if it's still a list
            if not path_df.empty and isinstance(path_df['path'].iloc[0], list):
                path_df['path'] = path_df['path'].apply(lambda x: '->'.join(map(str, x)))
            return path_df

        # Generate string representation
        path_strings = path_df['path'].apply(lambda x: '->'.join(map(str, x)) if isinstance(x, list) else str(x))
        
        # Save original list to path_str
        path_df['path_str'] = path_df['path']
        # Overwrite path with string
        path_df['path'] = path_strings
        
    return path_df

def path_filter(path_df, keyword_in_path_to_remove=None):
    """
    Filter paths containing specific keywords.
    """
    if path_df.empty or not keyword_in_path_to_remove:
        return path_df, pd.DataFrame()
        
    if isinstance(keyword_in_path_to_remove, str):
        keywords = [keyword_in_path_to_remove]
    else:
        keywords = keyword_in_path_to_remove
        
    mask = pd.Series(False, index=path_df.index)
    if 'path_str' in path_df.columns:
        for kw in keywords:
            mask |= path_df['path_str'].str.contains(kw, na=False)
    
    excluded = path_df[mask].copy()
    kept = path_df[~mask].copy()
    
    return kept, excluded

def build_path_dataframe_from_paths(paths, conn_data, targets, real_layer_map=None, level='type', type_lookup=None, edge_lookup=None):
    """
    Build a DataFrame from a list of paths, calculating weights and probabilities.
    Optimized with dictionary lookup for O(1) edge access.
    """
    if not paths:
        return pd.DataFrame()
        
    # Determine source and target columns based on level
    src_col = f'{level}_pre' if f'{level}_pre' in conn_data.columns else 'bodyId_pre'
    tgt_col = f'{level}_post' if f'{level}_post' in conn_data.columns else 'bodyId_post'
    
    if edge_lookup is None:
        # Pre-process connection data into a lookup dictionary
        # This avoids O(N) filtering inside the loop
        print(f"Optimizing path building: Pre-indexing {len(conn_data)} connections...")
        
        # Ensure we work with strings for consistent lookup
        # (The original code handled mixed types by trying both, so we normalize to string)
        conn_data_str = conn_data.copy()
        conn_data_str['src_str'] = conn_data_str[src_col].astype(str)
        conn_data_str['tgt_str'] = conn_data_str[tgt_col].astype(str)
        
        # Define aggregation functions
        agg_funcs = {
            'weight': 'sum',
        }
        if 'traversal_probability' in conn_data.columns:
            agg_funcs['traversal_probability'] = 'mean'
        if 'connection_ratio' in conn_data.columns:
            agg_funcs['connection_ratio'] = 'mean'
        if 'nt_type' in conn_data.columns:
            # Custom aggregator for nt_type to get unique values
            def unique_nt_types(x):
                types = x.dropna().unique().tolist()
                valid_types = sorted([str(t) for t in types if t and str(t).lower() != 'none' and str(t).lower() != 'nan'])
                return '|'.join(valid_types) if valid_types else 'Unknown'
            agg_funcs['nt_type'] = unique_nt_types

        # Group and aggregate
        grouped = conn_data_str.groupby(['src_str', 'tgt_str']).agg(agg_funcs)
        
        # Convert to dictionary for fast lookup
        # Key: (src_str, tgt_str), Value: dict of metrics
        edge_lookup = grouped.to_dict('index')
    
    rows = []
    
    # Use tqdm for progress bar
    path_iterator = tqdm(paths, desc=f"Enriching {level}-level paths", unit="path", 
                         disable=len(paths) < 100)
    
    for path in path_iterator:
        # Calculate path metrics
        weights = []
        probs = []
        ratios = []
        nt_types = []
        
        valid_path = True
        path_str_nodes = [str(n) for n in path]
        
        for i in range(len(path) - 1):
            u_str, v_str = path_str_nodes[i], path_str_nodes[i+1]
            
            # Fast lookup
            metrics = edge_lookup.get((u_str, v_str))
            
            if not metrics:
                # Debug print for first failure
                if len(rows) == 0 and i == 0:
                    print(f"Debug: Failed to find connection {path[i]} -> {path[i+1]} in conn_data")
                    print(f"  src_col: {src_col}, tgt_col: {tgt_col}")
                valid_path = False
                break
                
            # Get metrics
            w = metrics.get('weight', 0)
            p = metrics.get('traversal_probability', 0)
            r = metrics.get('connection_ratio', 0)
            
            weights.append(w)
            probs.append(p)
            ratios.append(r)
            
            if 'nt_type' in metrics:
                nt_types.append(metrics['nt_type'])
            
        if valid_path:
            # Format path string with types
            if type_lookup:
                path_formatted_parts = []
                for node in path:
                    node_type = 'Unknown'
                    # Handle potential string/int mismatch in lookup keys
                    t = type_lookup.get(node)
                    if t is None:
                        t = type_lookup.get(str(node))
                    if t is None and isinstance(node, str) and node.isdigit():
                         t = type_lookup.get(int(node))
                    if t is not None:
                        node_type = str(t)
                    path_formatted_parts.append(f"{node}_{node_type}")
                path_formatted = "->".join(path_formatted_parts)
            else:
                # No type lookup provided (e.g. level='type'), just join nodes
                path_formatted = "->".join(map(str, path))

            row = {
                'path_str': path,
                'path': path_formatted,
                'weights': weights,
                'probabilities': probs,
                'ratios': ratios,
                'min_weight': min(weights) if weights else 0,
                'path_prob': np.prod(probs) if probs else 0,
                'min_ratio': min(ratios) if ratios else 0,
                'length': len(path) - 1
            }
            
            if nt_types:
                row['nt_types'] = nt_types
            
            # Add path types if lookup is provided
            if type_lookup:
                path_types = []
                for node in path:
                    # Handle potential string/int mismatch in lookup keys
                    node_type = type_lookup.get(node)
                    if node_type is None:
                        node_type = type_lookup.get(str(node))
                    if node_type is None and isinstance(node, str) and node.isdigit():
                         node_type = type_lookup.get(int(node))
                    
                    path_types.append(str(node_type) if node_type is not None else 'Unknown')
                row['path_types'] = path_types
                
            rows.append(row)
            
    return pd.DataFrame(rows)

def process_paths_streaming(path_gen, conn_data, targets, output_path, 
                          excluded_path=None, real_layer_map=None, 
                          level='type', type_lookup=None, 
                          keyword_in_path_to_remove=None,
                          batch_size=100000):
    """
    Stream paths from generator, process in batches, and write to CSV.
    Returns total count of saved paths.
    """
    import gc
    import os
    
    # Pre-process connection data into a lookup dictionary once
    # Determine source and target columns based on level
    src_col = f'{level}_pre' if f'{level}_pre' in conn_data.columns else 'bodyId_pre'
    tgt_col = f'{level}_post' if f'{level}_post' in conn_data.columns else 'bodyId_post'
    
    print(f"Optimizing path building: Pre-indexing {len(conn_data)} connections...")
    
    # Ensure we work with strings for consistent lookup
    conn_data_str = conn_data.copy()
    conn_data_str['src_str'] = conn_data_str[src_col].astype(str)
    conn_data_str['tgt_str'] = conn_data_str[tgt_col].astype(str)
    
    # Define aggregation functions
    agg_funcs = {
        'weight': 'sum',
    }
    if 'traversal_probability' in conn_data.columns:
        agg_funcs['traversal_probability'] = 'mean'
    if 'connection_ratio' in conn_data.columns:
        agg_funcs['connection_ratio'] = 'mean'
    if 'nt_type' in conn_data.columns:
        def unique_nt_types(x):
            types = x.dropna().unique().tolist()
            valid_types = sorted([str(t) for t in types if t and str(t).lower() != 'none' and str(t).lower() != 'nan'])
            return '|'.join(valid_types) if valid_types else 'Unknown'
        agg_funcs['nt_type'] = unique_nt_types

    # Group and aggregate
    grouped = conn_data_str.groupby(['src_str', 'tgt_str']).agg(agg_funcs)
    
    # Convert to dictionary for fast lookup
    edge_lookup = grouped.to_dict('index')
    
    batch = []
    total_saved = 0
    total_excluded = 0
    
    # Initialize CSV files (write headers)
    first_batch = True
    
    # Use tqdm for progress bar if possible
    try:
        from tqdm import tqdm
        iterator = tqdm(path_gen, desc=f"Streaming {level}-level paths", unit="path")
    except ImportError:
        iterator = path_gen
    
    for path in iterator:
        batch.append(path)
        
        if len(batch) >= batch_size:
            # Process batch
            df = build_path_dataframe_from_paths(batch, conn_data, targets, 
                                               real_layer_map, level, type_lookup,
                                               edge_lookup=edge_lookup)
            
            # Filter zero-weight
            if not df.empty:
                df = df[df['weights'].apply(lambda w_list: all(w > 0 for w in w_list))]
            
            # Split path (convert to string)
            df = split_path(df)
            
            # Filter keywords
            df, excluded = path_filter(df, keyword_in_path_to_remove)
            
            # Write to CSV
            if not df.empty:
                mode = 'w' if first_batch else 'a'
                header = first_batch
                df.to_csv(output_path, mode=mode, header=header, index=False)
                total_saved += len(df)
                
            if excluded_path and not excluded.empty:
                mode = 'w' if first_batch else 'a'
                header = first_batch
                excluded.to_csv(excluded_path, mode=mode, header=header, index=False)
                total_excluded += len(excluded)
            
            if first_batch:
                first_batch = False
                
            batch = []
            gc.collect()
            
    # Process remaining
    if batch:
        df = build_path_dataframe_from_paths(batch, conn_data, targets, 
                                           real_layer_map, level, type_lookup,
                                           edge_lookup=edge_lookup)
        
        if not df.empty:
            df = df[df['weights'].apply(lambda w_list: all(w > 0 for w in w_list))]
        
        df = split_path(df)
        df, excluded = path_filter(df, keyword_in_path_to_remove)
        
        if not df.empty:
            mode = 'w' if first_batch else 'a'
            header = first_batch
            df.to_csv(output_path, mode=mode, header=header, index=False)
            total_saved += len(df)
            
        if excluded_path and not excluded.empty:
            mode = 'w' if first_batch else 'a'
            header = first_batch
            excluded.to_csv(excluded_path, mode=mode, header=header, index=False)
            total_excluded += len(excluded)
            
    return total_saved

def EnrichConnectionTable(conn_table, traversal_probability_threshold=0, dataset=None, script_path=None, target_neurons_df=None, aggregate_method='product', label_mapper=None):
    '''Add traversal probability, connection ratio, and layer information to the connection table
    
    Parameters
    ----------
    conn_table : DataFrame
        Connection table to enrich
    traversal_probability_threshold : float, optional
        Minimum traversal probability threshold (default: 0)
    dataset : str, optional
        Dataset name (e.g., 'optic-lobe:v1.1') for local dataset lookup
    script_path : str, optional
        Path to script directory containing 'datasets' folder
    target_neurons_df : DataFrame, optional
        Full dataframe of target neurons (with bodyId, type, post columns).
        Used to get correct type-level denominators. If not provided, only
        neurons appearing in connections will be used (less accurate).
    aggregate_method : str, optional
        Method for aggregating type-level traversal probabilities from bodyId level:
        - 'product': compound probability (product of block probs) for paths (default)
        - 'average': weighted average for direct parallel connections
    label_mapper : LabelMapper, optional
        LabelMapper object to standardize types in the local dataset for accurate ratio calculation.
    
    Returns
    -------
    conn_df : DataFrame
        Enriched connection table with bodyId-level metrics
    conn_type : DataFrame
        Type-level aggregation (always based on original type column)
    conn_group : DataFrame or None
        Custom group-level aggregation (only if custom_group columns exist)
    '''
    conn_df = conn_table.copy()
    
    # Determine grouping columns (use custom_group if available, otherwise type)
    group_pre = 'custom_group_pre' if 'custom_group_pre' in conn_df.columns else 'type_pre'
    group_post = 'custom_group_post' if 'custom_group_post' in conn_df.columns else 'type_post'
    
    # Try to use local dataset first
    use_local = False
    ndf_complete = None
    if dataset and script_path:
        dataset_clean = dataset.replace(':', '_').replace('.', '_')
        # Prioritize subdirectory structure
        dataset_path = os.path.join(
            script_path,
            'datasets',
            dataset_clean,
            f"{dataset_clean}_allneurons_neuron_df.csv"
        )
        
        # Enhanced dataset discovery logic
        if not os.path.exists(dataset_path):
            # Fallback: Try root datasets folder (legacy)
            legacy_path = os.path.join(
                script_path,
                'datasets',
                f"{dataset_clean}_allneurons_neuron_df.csv"
            )
            if os.path.exists(legacy_path):
                dataset_path = legacy_path
            else:
                # Try globbing for any *_allneurons_neuron_df.csv in subdir
                subdir_path = os.path.join(script_path, 'datasets', dataset_clean)
                if os.path.exists(subdir_path):
                    import glob
                    candidates = glob.glob(os.path.join(subdir_path, "*_allneurons_neuron_df.csv"))
                    if candidates:
                        dataset_path = candidates[0]

        if os.path.exists(dataset_path):
            use_local = True
            # Handle FlyWire/FAFB which might use string bodyIds
            if 'flywire' in dataset.lower() or 'fafb' in dataset.lower():
                ndf_complete = pd.read_csv(dataset_path, header=0, index_col=None, dtype={'bodyId': str}, low_memory=False)
            else:
                ndf_complete = pd.read_csv(dataset_path, header=0, index_col=0, low_memory=False)
            
            # Ensure bodyId column is string for comparison
            if 'bodyId' in ndf_complete.columns:
                ndf_complete['bodyId'] = ndf_complete['bodyId'].astype(str)
            
            mapped_dict = {}
            # Apply label mapping to local dataset if provided
            if label_mapper and 'type' in ndf_complete.columns:
                # Create a copy to avoid modifying the original cache
                ndf_complete = ndf_complete.copy()
                
                # Define mapping function that prioritizes bodyId mapping
                def get_mapped_label(row):
                    # 1. Try bodyId first (most specific)
                    body_id = str(row['bodyId'])
                    # Use get_label to check all roles (Source -> Target -> Intermediate)
                    mapped_body = label_mapper.get_label(dataset, body_id)
                    
                    if body_id == '720575940634984800':
                        # print(f"DEBUG: Mapping 720575940634984800 -> {mapped_body}")
                        pass
                    
                    # Check if mapped (heuristic: different from original ID)
                    if mapped_body != body_id:
                         return mapped_body
                    
                    # 2. Try type if available
                    if pd.notna(row['type']):
                        type_val = str(row['type'])
                        # Skip if type is same as bodyId (redundant)
                        if type_val != body_id:
                            mapped_type = label_mapper.get_label(dataset, type_val)
                            if mapped_type != type_val:
                                return mapped_type
                            
                    # No mapping found
                    return ''

                # Apply mapping
                ndf_complete['std_label'] = ndf_complete.apply(get_mapped_label, axis=1)
                
                # Overwrite type with std_label where available
                mask = ndf_complete['std_label'] != ''
                ndf_complete.loc[mask, 'type'] = ndf_complete.loc[mask, 'std_label']
                
                # Create a dictionary of ONLY the mapped labels
                mapped_dict = ndf_complete.loc[mask].set_index('bodyId')['std_label'].to_dict()
                
                # Also overwrite custom_group if present, to ensure grouping uses the mapped label
                if 'custom_group' in ndf_complete.columns:
                    ndf_complete.loc[mask, 'custom_group'] = ndf_complete.loc[mask, 'std_label']
                
                # print(f"DEBUG: Unique types after mapping: {ndf_complete['type'].unique()}")
                # if 'custom_group' in ndf_complete.columns:
                #      print(f"DEBUG: custom_group sample: {ndf_complete['custom_group'].unique()[:5]}")
                
            bodyIds_needed = conn_df.bodyId_post.astype(str).unique().tolist()
            df_post = ndf_complete[ndf_complete['bodyId'].isin(bodyIds_needed)][['bodyId', 'post']].copy()
    
    if not use_local:
        # Fallback to API call
        # Note: fetch_neurons is not available in this context, so we rely on target_neurons_df or existing data
        if target_neurons_df is not None:
            # Ensure consistent types for filtering (convert to string to handle int64 vs str mismatch)
            target_bodyIds = target_neurons_df['bodyId'].astype(str);
            conn_bodyIds = conn_df.bodyId_post.astype(str).unique();
            
            # Apply label mapping to target_neurons_df if provided
            if label_mapper and 'type' in target_neurons_df.columns:
                target_neurons_df = target_neurons_df.copy()
                
                # Define mapping function (same as above)
                def get_mapped_label_target(row):
                    body_id = str(row['bodyId'])
                    # Use get_label to check all roles
                    mapped_body = label_mapper.get_label(dataset, body_id)
                    if mapped_body != body_id:
                         return mapped_body
                    
                    if pd.notna(row['type']):
                        type_val = str(row['type'])
                        if type_val != body_id:
                            mapped_type = label_mapper.get_label(dataset, type_val)
                            if mapped_type != type_val:
                                return mapped_type
                    return ''

                target_neurons_df['std_label'] = target_neurons_df.apply(get_mapped_label_target, axis=1)
                mask = target_neurons_df['std_label'] != ''
                target_neurons_df.loc[mask, 'type'] = target_neurons_df.loc[mask, 'std_label']
            
            df_post = target_neurons_df[target_bodyIds.isin(conn_bodyIds)][['bodyId', 'post']].copy();
        else:
            # Last resort: assume post info might be in conn_df or we can't get it
            # If we can't get it, we can't calculate connection_ratio accurately
            print("Warning: Could not fetch neuron info for connection ratio calculation.");
            df_post = pd.DataFrame(columns=['bodyId', 'post']);
    
    post_info = df_post.copy();
    post_info.columns = ['bodyId_post','post'];
    
    # Handle case where type_pre/type_post columns already exist (from cache enrichment)
    if 'type_pre' in conn_df.columns:
        conn_df.loc[conn_df.type_pre.isnull(),'type_pre'] = 'Unknown';
    else:
        conn_df['type_pre'] = 'Unknown';
        
    if 'type_post' in conn_df.columns:
        conn_df.loc[conn_df.type_post.isnull(),'type_post'] = 'Unknown';
    else:
        conn_df['type_post'] = 'Unknown';

    # Update types from local dataset if available
    if use_local and ndf_complete is not None and 'type' in ndf_complete.columns:
        # Ensure bodyIds are strings
        conn_df['bodyId_pre'] = conn_df['bodyId_pre'].astype(str);
        conn_df['bodyId_post'] = conn_df['bodyId_post'].astype(str);
        
        # Map types if missing or empty
        type_map = ndf_complete.set_index('bodyId')['type'].to_dict();
        
        if label_mapper:
            # 1. Apply explicit mappings (overwrite types for mapped neurons)
            if mapped_dict:
                conn_df['type_pre'] = conn_df['bodyId_pre'].map(mapped_dict).fillna(conn_df['type_pre'])
                conn_df['type_post'] = conn_df['bodyId_post'].map(mapped_dict).fillna(conn_df['type_post'])
            
            # 2. Fill unknowns with type_map (original types + mapped types)
            # This preserves existing valid types in conn_df that were NOT mapped, preventing fallback to bodyIds
            mask_pre = (conn_df['type_pre'].isnull()) | (conn_df['type_pre'] == '') | (conn_df['type_pre'] == 'Unknown')
            if mask_pre.any():
                conn_df.loc[mask_pre, 'type_pre'] = conn_df.loc[mask_pre, 'bodyId_pre'].map(type_map).fillna('Unknown')
                
            mask_post = (conn_df['type_post'].isnull()) | (conn_df['type_post'] == '') | (conn_df['type_post'] == 'Unknown')
            if mask_post.any():
                conn_df.loc[mask_post, 'type_post'] = conn_df.loc[mask_post, 'bodyId_post'].map(type_map).fillna('Unknown')
        else:
            # Update type_pre
            mask_pre = (conn_df['type_pre'].isnull()) | (conn_df['type_pre'] == '') | (conn_df['type_pre'] == 'Unknown')
            if mask_pre.any():
                conn_df.loc[mask_pre, 'type_pre'] = conn_df.loc[mask_pre, 'bodyId_pre'].map(type_map).fillna('Unknown')
                
            # Update type_post
            mask_post = (conn_df['type_post'].isnull()) | (conn_df['type_post'] == '') | (conn_df['type_post'] == 'Unknown')
            if mask_post.any():
                conn_df.loc[mask_post, 'type_post'] = conn_df.loc[mask_post, 'bodyId_post'].map(type_map).fillna('Unknown')

    # Fill custom_group columns if they exist
    if 'custom_group_pre' in conn_df.columns:
        conn_df.loc[conn_df.custom_group_pre.isnull(),'custom_group_pre'] = conn_df.loc[conn_df.custom_group_pre.isnull(),'type_pre']
    if 'custom_group_post' in conn_df.columns:
        conn_df.loc[conn_df.custom_group_post.isnull(),'custom_group_post'] = conn_df.loc[conn_df.custom_group_post.isnull(),'type_post']
    
    # Ensure bodyId columns are strings for merging to avoid warnings
    conn_df['bodyId_post'] = conn_df['bodyId_post'].astype(str)
    post_info['bodyId_post'] = post_info['bodyId_post'].astype(str)

    conn_df = conn_df.merge(post_info,how='left',on='bodyId_post')
    
    # Handle potential column collision if 'post' already existed (creates post_x, post_y)
    if 'post_x' in conn_df.columns and 'post_y' in conn_df.columns:
        # Prefer the new info (post_y) if available, otherwise keep old (post_x)
        conn_df['post'] = conn_df['post_y'].fillna(conn_df['post_x'])
        conn_df = conn_df.drop(columns=['post_x', 'post_y'])
        
    # Calculate connection_ratio (handle if already exists)
    if 'connection_ratio' in conn_df.columns:
        conn_df['connection_ratio'] = conn_df.weight / conn_df.post
    else:
        conn_df.insert(loc=len(conn_df.columns),column='connection_ratio',value=conn_df.weight/conn_df.post)
        
    # Calculate traversal_probability (handle if already exists)
    if 'traversal_probability' in conn_df.columns:
        conn_df['traversal_probability'] = conn_df.connection_ratio / 0.3
    else:
        conn_df.insert(loc=3,column='traversal_probability',value=conn_df.connection_ratio/0.3)
    
    conn_df.loc[conn_df.traversal_probability > 1,'traversal_probability'] = 1
    
    # Calculate block_probability (handle if already exists)
    if 'block_probability' in conn_df.columns:
        conn_df['block_probability'] = 1 - conn_df.traversal_probability
    else:
        conn_df.insert(loc=len(conn_df.columns),column='block_probability',value= 1 - conn_df.traversal_probability)
    
    conn_df = conn_df.loc[conn_df.traversal_probability >= traversal_probability_threshold]
    
    # Aggregate connection data by neuron type/group
    # Use custom_group if available, otherwise fall back to type
    # Calculate from bodyId level to ensure accuracy (neurons in connections, not types in connections)
    # First deduplicate by bodyId pairs to avoid counting same connection multiple times
    cols_to_keep = ['bodyId_pre', 'bodyId_post', group_pre, group_post, 'weight']
    has_nt = 'nt_type' in conn_df.columns
    if has_nt:
        cols_to_keep.append('nt_type')
        
    bodyid_pairs = conn_df[cols_to_keep].drop_duplicates(subset=['bodyId_pre', 'bodyId_post'])
    
    # Add progress indicator for large aggregations
    pbar = None
    # if len(bodyid_pairs) > 50000:
    #     pbar = tqdm(total=5, desc=f"  Enriching {len(bodyid_pairs):,} connections", unit="step")
    
    if has_nt:
        # For nt_type, take the most frequent one (mode)
        weight_sum = bodyid_pairs.groupby([group_pre, group_post]).agg({
            'weight': 'sum',
            'nt_type': lambda x: x.mode().iloc[0] if not x.mode().empty else None
        }).reset_index()
    else:
        weight_sum = bodyid_pairs.groupby([group_pre, group_post])['weight'].sum().reset_index(name='weight')
    
    if pbar: pbar.update(1) # Step 1: Weight aggregation complete
    
    # Calculate total incoming weights per group_post (sum across all group_pre sources)
    total_incoming_per_type = weight_sum.groupby(group_post)['weight'].sum().reset_index(name='total_incoming_weight')

    if pbar: pbar.update(1) # Step 2: Incoming weights calculated

    # Calculate total post-synaptic sites for ALL neurons of each group_post
    # Only use target_neurons_df if we don't have a local dataset, or if we specifically need it for custom groups
    # If use_local is True and we are using standard types, we prefer the full dataset to ensure we get counts for all intermediate types
    if target_neurons_df is not None and (not use_local or group_post != 'type_post'):
        # Determine which column to use for grouping in target_neurons_df
        target_group_col = 'custom_group' if 'custom_group' in target_neurons_df.columns else 'type'
        if target_group_col in target_neurons_df.columns and 'post' in target_neurons_df.columns:
            all_post_neurons = target_neurons_df[[target_group_col, 'post']].copy()
            all_post_neurons = all_post_neurons.rename(columns={target_group_col: group_post})
            # Remove None types and empty strings
            all_post_neurons = all_post_neurons[all_post_neurons[group_post].notnull()]
            all_post_neurons = all_post_neurons[all_post_neurons[group_post] != '']
            all_post_neurons = all_post_neurons[all_post_neurons[group_post] != 'None']
            type_post_totals = all_post_neurons.groupby(group_post)['post'].sum().reset_index(name='total_post')
        else:
            # Fallback
            type_post_totals = None
    else:
        type_post_totals = None
    
    # Auto-load if not provided
    if type_post_totals is None:
        if use_local and group_post in conn_df.columns:
            # Get all groups that appear in connections
            groups_in_conn = conn_df[group_post].unique().tolist()
            
            # Apply label mapping to ndf_complete if available
            # This ensures that when we look up total post counts for a type,
            # we match the standardized labels used in conn_df
            if label_mapper and ndf_complete is not None and group_post == 'type_post':
                # Create a copy to avoid modifying the cached dataframe
                ndf_complete = ndf_complete.copy()
                
                # Map types
                ndf_complete['std_label'] = ndf_complete.apply(
                    lambda row: label_mapper.get_label(dataset, row['type'] if pd.notna(row['type']) else row['bodyId']),
                    axis=1
                )
                
                # Overwrite type
                mask = ndf_complete['std_label'] != ''
                ndf_complete.loc[mask, 'type'] = ndf_complete.loc[mask, 'std_label']
            
            # Need to match back to original neurons for dataset lookup
            if group_post == 'custom_group_post' and 'bodyId_post' in conn_df.columns:
                # Get all bodyIds that belong to each custom group from connections
                # This gives us the mapping, but we need ALL neurons in each group, not just those in connections
                group_to_bodyids = conn_df[['custom_group_post', 'bodyId_post']].drop_duplicates()
                
                # For nested groups, we need ALL neurons that belong to each group
                # Use target_neurons_df if available (should have custom_group column)
                if target_neurons_df is not None and 'custom_group' in target_neurons_df.columns and 'bodyId' in target_neurons_df.columns:
                    # Get all neurons with their groups
                    neurons_with_groups = target_neurons_df[['bodyId', 'custom_group', 'post']].copy()
                    neurons_with_groups = neurons_with_groups[neurons_with_groups['custom_group'].notnull()]
                    neurons_with_groups = neurons_with_groups[neurons_with_groups['custom_group'].isin(groups_in_conn)]
                    # Sum post by custom_group
                    type_post_totals = neurons_with_groups.groupby('custom_group')['post'].sum().reset_index()
                    type_post_totals.columns = ['custom_group_post', 'total_post']
                else:
                    # Fallback: use bodyIds from connections to find their groups, then get ALL neurons of those types
                    # Get unique bodyIds and their groups
                    all_bodyids_in_groups = conn_df[['bodyId_post', 'custom_group_post']].drop_duplicates()
                    # Fetch neuron data for these bodyIds to get their types
                    if 'type_post' in conn_df.columns:
                        # Use type information from conn_df
                        bodyid_type_group = conn_df[['bodyId_post', 'type_post', 'custom_group_post']].drop_duplicates()
                        # Get all types for each group
                        group_to_types = bodyid_type_group.groupby('custom_group_post')['type_post'].apply(lambda x: x.unique().tolist()).reset_index()
                        
                        # For each group, get ALL neurons of its constituent types
                        group_post_totals = []
                        for idx in group_to_types.index:
                            grp = group_to_types.at[idx, 'custom_group_post']
                            types = group_to_types.at[idx, 'type_post']
                            # Get all neurons of these types from local dataset
                            neurons_in_group = ndf_complete[ndf_complete['type'].isin(types)][['bodyId', 'post']].copy()
                            total_post = neurons_in_group['post'].sum()
                            group_post_totals.append({'custom_group_post': grp, 'total_post': total_post})
                        type_post_totals = pd.DataFrame(group_post_totals)
                    else:
                        # Last resort: use only neurons appearing in connections
                        type_post_totals = conn_df[['custom_group_post', 'bodyId_post', 'post']].drop_duplicates(subset=['bodyId_post']).groupby('custom_group_post')['post'].sum().reset_index(name='total_post')
            else:
                # Standard type-based grouping
                all_post_neurons = ndf_complete[ndf_complete['type'].isin(groups_in_conn)][['type', 'post']].copy()
                all_post_neurons = all_post_neurons.rename(columns={'type': group_post})
                all_post_neurons = all_post_neurons[all_post_neurons[group_post].notnull()]
                all_post_neurons = all_post_neurons[all_post_neurons[group_post] != '']
                all_post_neurons = all_post_neurons[all_post_neurons[group_post] != 'None']
                type_post_totals = all_post_neurons.groupby(group_post)['post'].sum().reset_index(name='total_post')
        else:
            # Last resort: use only neurons appearing in connections
            all_post_neurons = conn_df[[group_post, 'bodyId_post', 'post']].drop_duplicates(subset=['bodyId_post'])
            type_post_totals = all_post_neurons.groupby(group_post)['post'].sum().reset_index(name='total_post')
    
    if pbar: pbar.update(1) # Step 3: Post-synaptic totals calculated

    # Calculate group-to-group connection_ratio
    conn_type = weight_sum.merge(type_post_totals, on=group_post, how='left')
    
    # Calculate ratio using total post-synaptic sites as denominator
    conn_type['connection_ratio'] = conn_type.apply(
        lambda row: row['weight'] / row['total_post'] if pd.notnull(row['total_post']) and row['total_post'] > 0 else 0.0,
        axis=1
    )

    if pbar: pbar.update(1) # Step 4: Ratios calculated

    # Group-to-group traversal_probability aggregation
    if aggregate_method == 'product':
        # Product method: compound probability for paths
        conn_traversal = conn_df[[group_pre, group_post, 'block_probability']]
        conn_traversal = conn_traversal.groupby([group_pre, group_post]).prod().reset_index()
        conn_type = conn_type.merge(conn_traversal, how='left', on=[group_pre, group_post])
        conn_type['traversal_probability'] = 1 - conn_type['block_probability']
    else:
        # Average method: weighted average for direct connections
        conn_traversal = conn_df[[group_pre, group_post, 'weight', 'traversal_probability']]
        weighted_sum = conn_traversal.groupby([group_pre, group_post]).apply(
            lambda g: (g['weight'] * g['traversal_probability']).sum() / g['weight'].sum() if g['weight'].sum() > 0 else 0.0
        ).reset_index(name='traversal_probability')
        conn_type = conn_type.merge(weighted_sum, how='left', on=[group_pre, group_post])
        conn_type['block_probability'] = 1 - conn_type['traversal_probability']
    
    if pbar: 
        pbar.update(1) # Step 5: Traversal probabilities calculated
        pbar.close()
    
    # Fix FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated
    # Explicitly convert to numeric before filling NaNs to avoid object-dtype downcasting issues
    conn_aggregated = conn_type.copy()
    conn_aggregated['connection_ratio'] = pd.to_numeric(conn_aggregated['connection_ratio'], errors='coerce').fillna(0.0)
    conn_aggregated['traversal_probability'] = pd.to_numeric(conn_aggregated['traversal_probability'], errors='coerce').fillna(0.0)
    conn_aggregated['block_probability'] = pd.to_numeric(conn_aggregated['block_probability'], errors='coerce').fillna(1.0)
    # conn_aggregated = conn_aggregated.infer_objects() # No longer needed as we explicitly converted
    
    conn_aggregated = conn_aggregated[[group_pre, group_post, 'weight', 'connection_ratio', 'traversal_probability', 'block_probability']]
    
    # Check if we're using custom groups
    has_custom_groups = (group_pre == 'custom_group_pre' and group_post == 'custom_group_post')
    
    if has_custom_groups:
        # Compute BOTH type-based and custom group-based aggregations
        # 1. Custom group aggregation (already computed)
        conn_group = conn_aggregated.rename(columns={group_pre: 'group_pre', group_post: 'group_post'})
        
        # 2. Original type-based aggregation
        # Calculate from bodyId level for accuracy
        bodyid_pairs_type = conn_df[['bodyId_pre', 'bodyId_post', 'type_pre', 'type_post', 'weight']].drop_duplicates(subset=['bodyId_pre', 'bodyId_post'])
        
        pbar_type = None
        # if len(bodyid_pairs_type) > 50000:
        #     pbar_type = tqdm(total=4, desc=f"  Enriching {len(bodyid_pairs_type):,} type-level connections", unit="step")

        weight_sum_type = bodyid_pairs_type.groupby(['type_pre', 'type_post'])['weight'].sum().reset_index(name='weight')
        total_incoming_per_type_orig = weight_sum_type.groupby('type_post')['weight'].sum().reset_index(name='total_incoming_weight')
        
        if pbar_type: pbar_type.update(1) # Step 1: Weight aggregation

        # Calculate type-level denominators
        if target_neurons_df is not None and 'type' in target_neurons_df.columns and 'post' in target_neurons_df.columns:
            all_post_neurons_type = target_neurons_df[['type', 'post']].copy()
            all_post_neurons_type = all_post_neurons_type.rename(columns={'type': 'type_post'})
            all_post_neurons_type = all_post_neurons_type[all_post_neurons_type['type_post'].notnull()]
            all_post_neurons_type = all_post_neurons_type[all_post_neurons_type['type_post'] != '']
            all_post_neurons_type = all_post_neurons_type[all_post_neurons_type['type_post'] != 'None']
            type_post_totals_orig = all_post_neurons_type.groupby('type_post')['post'].sum().reset_index(name='total_post')
        elif use_local and 'type_post' in conn_df.columns:
            types_in_conn = conn_df['type_post'].unique().tolist()
            all_post_neurons_type = ndf_complete[ndf_complete['type'].isin(types_in_conn)][['type', 'post']].copy()
            all_post_neurons_type = all_post_neurons_type.rename(columns={'type': 'type_post'})
            all_post_neurons_type = all_post_neurons_type[all_post_neurons_type['type_post'].notnull()]
            all_post_neurons_type = all_post_neurons_type[all_post_neurons_type['type_post'] != '']
            all_post_neurons_type = all_post_neurons_type[all_post_neurons_type['type_post'] != 'None']
            type_post_totals_orig = all_post_neurons_type.groupby('type_post')['post'].sum().reset_index(name='total_post')
        else:
            all_post_neurons_type = conn_df[['type_post', 'bodyId_post', 'post']].drop_duplicates(subset=['bodyId_post'])
            type_post_totals_orig = all_post_neurons_type.groupby('type_post')['post'].sum().reset_index(name='total_post')
        
        if pbar_type: pbar_type.update(1) # Step 2: Post-synaptic totals

        conn_type = weight_sum_type.merge(type_post_totals_orig, on='type_post', how='left')
        conn_type['connection_ratio'] = conn_type.apply(
            lambda row: row['weight'] / row['total_post'] if pd.notnull(row['total_post']) and row['total_post'] > 0 else 0.0,
            axis=1
        )
        
        if pbar_type: pbar_type.update(1) # Step 3: Ratios calculated

        # Type-level traversal probability
        if aggregate_method == 'product':
            conn_traversal_type = conn_df[['type_pre', 'type_post', 'block_probability']]
            conn_traversal_type = conn_traversal_type.groupby(['type_pre', 'type_post']).prod().reset_index()
            conn_type = conn_type.merge(conn_traversal_type, how='left', on=['type_pre', 'type_post'])
            conn_type['traversal_probability'] = 1 - conn_type['block_probability']
        else:
            conn_traversal_type = conn_df[['type_pre', 'type_post', 'weight', 'traversal_probability']]
            weighted_sum_type = conn_traversal_type.groupby(['type_pre', 'type_post']).apply(
                lambda g: (g['weight'] * g['traversal_probability']).sum() / g['weight'].sum() if g['weight'].sum() > 0 else 0.0
            ).reset_index(name='traversal_probability')
            conn_type = conn_type.merge(weighted_sum_type, how='left', on=['type_pre', 'type_post'])
            conn_type['block_probability'] = 1 - conn_type['traversal_probability']
        
        if pbar_type: 
            pbar_type.update(1) # Step 4: Traversal probabilities
            pbar_type.close()
        
        conn_type = conn_type.fillna({'connection_ratio': 0.0, 'traversal_probability': 0.0, 'block_probability': 1.0}).infer_objects()
        conn_type = conn_type[['type_pre', 'type_post', 'weight', 'connection_ratio', 'traversal_probability', 'block_probability']]
        
        # if len(bodyid_pairs) > 50000:
        #     print(f"  Enrichment complete. Result shape: {conn_type.shape}. Returning results...", flush=True)
        return conn_df, conn_type, conn_group
    else:
        # No custom groups - return original type aggregation only
        conn_type = conn_aggregated.rename(columns={group_pre: 'type_pre', group_post: 'type_post'})
        # if len(bodyid_pairs) > 50000:
        #     print(f"  Enrichment complete. Result shape: {conn_type.shape}. Returning results...", flush=True)
        return conn_df, conn_type, None


def SankeyDirect(cmat, file_path, showfig=True, node_color='rgba(31, 119, 180, 0.8)', link_color='rgba(0, 0, 0, 0.2)'):
    """
    Create a Sankey diagram from a connection matrix.
    
    Parameters
    ----------
    cmat : pd.DataFrame
        Connection matrix (rows=source, cols=target)
    file_path : str
        Path to save the HTML file
    showfig : bool
        Whether to show the figure
    node_color : str
        Color for nodes
    link_color : str
        Color for links
    """
    import plotly.graph_objects as go
    
    # Get sources and targets
    sources = cmat.index.tolist()
    targets = cmat.columns.tolist()
    
    # Create node list (sources + targets)
    # Treat them as bipartite for "Direct" connection
    # Source nodes indices: 0 to len(sources)-1
    # Target nodes indices: len(sources) to len(sources)+len(targets)-1
    
    labels = sources + targets
    
    # Create links
    source_indices = []
    target_indices = []
    values = []
    
    for i, src in enumerate(sources):
        for j, tgt in enumerate(targets):
            val = cmat.at[src, tgt]
            if val > 0:
                source_indices.append(i)
                target_indices.append(len(sources) + j)
                values.append(val)

    if not values:
        print("No connections to plot in Sankey diagram.")
        return

    # Create figure
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = labels,
          color = node_color
        ),
        link = dict(
          source = source_indices,
          target = target_indices,
          value = values,
          color = link_color
      ))])

    fig.update_layout(title_text="Direct Connections Sankey Diagram", font_size=10)
    
    # Save
    fig.write_html(file_path)
    print(f"Sankey diagram saved to {file_path}")
    
    if showfig:
        fig.show()

def ConcatenateIMG2PDF(folder_path,file_format=['png','jpg'],filename='PDF_sum',include_subfolder=False):
    ''' Concatenate all images in a folder to a single PDF file.'''
    if 'jpg' in file_format:
        file_format.append('jpeg')
    elif 'jpeg' in file_format:
        file_format.append('jpg')
    file_format = list(set(file_format))
    files = os.listdir(folder_path)
    figs = []
    for f in files:
        ftype = os.path.splitext(f)[-1][1:]
        if ftype in file_format:
            figs.append(os.path.join(folder_path,f))
    figs.sort()
    if len(figs) > 0:
        with open(os.path.join(folder_path,filename + '.pdf'),'wb') as f_sum:
            f_sum.write(img2pdf.convert(figs))
        print('Concatenated {:d} pictures to PDF'.format(len(figs)))
    else:
        print('Found no pictures to concatenate.')
    
def Vis3S(data_df,**kwargs): 
    """ Visualize Soma, Skeletons, Synapses or synapse distributions
    Args:
        data_df (pandas.DataFrame): dataframe contains centroid, classification, axis lengths (Ellipse) or radius (Circle).
    """
    
    options = {
        "save_path" : '_3S',
        "title"     : 'MyTitle',
        "classby"   : 'type',
        "plane"     : 'xz',
        "alpha"     : .3,
        "dpi"       : 300,
        "toPlot"    : 'soma', # "soma" or "synapse_distribution" or "synapse" or "skeleton"
        "xlim"      : (0,50000),
        "ylim"      : (50000,0), # reversed
        "showfig"   : False, # faster than True
        "facecolor" : bokeh.palettes.Set1[9],
        "site"      : None, # None, 'pre' or 'post'
        "snp_rois"   : None,
        "show_mesh"  : True,
        "mesh_roi"   : None,
        "roi_range"  : 'primary_rois', # {"primary_rois", "all_rois"}, see more details in neuprint.
        "mesh_color"    : [0.1,0.1,0.1],
        "mesh_alpha"    : 0.1,
        "confidence"    : 0,
        "synapseRadius" : 100,
        "synpase_file_path" : None,
        "save_format": '.png',
        "dataset": 'hemibrain',
        "data_folder": None
    }
    options.update(kwargs)
    if options['snp_rois'] != None and options['mesh_roi'] == None: 
        options['mesh_roi'] = options['snp_rois']
    elif options['snp_rois'] == None and options['mesh_roi'] == None:
        options['mesh_roi'] = ['LH(R)', 'AL(R)', 'EB']
    op = SimpleNamespace(**options)
    
    # Mesh loading
    roimesh = None
    if op.show_mesh:
        roiunits = []
        # Only load meshes for hemibrain for now, or if we have FAFB meshes
        # FAFB meshes are not standard in this repo yet
        if 'hemibrain' in str(op.dataset).lower():
            for roi in op.mesh_roi:
                mesh_file = os.path.join('navis_roi_meshes_json',op.roi_range,roi+'.json')
                if os.path.exists(mesh_file):
                    mesh = navis.Volume.from_json(mesh_file)
                    roiunits.append(mesh)
                else:
                    print('mesh file %s.json not found!'%(roi))
            if roiunits:
                roimesh = navis.Volume.combine(roiunits)
    
    if op.toPlot == 'synapse':
        if op.synapse_file_path:
            snp_file = pd.ExcelFile(op.synapse_file_path)
        else:
            print("Warning: synapse_file_path not provided for synapse plot")
            return

    summary_df = data_df.copy()
    if op.toPlot == 'soma':
        if 'somaLocation' in summary_df.columns:
            print('not found soma of %d neurons'%(summary_df['somaLocation'].isnull().sum()))
            summary_df = summary_df[summary_df['somaLocation'].notnull()]
        else:
            print("Warning: somaLocation column missing")
    elif op.toPlot == 'synapse_distribution':
        if 'snpN_roi' in summary_df.columns:
            print('drop %d neurons having no more than 1 synapses in the ROI'%((summary_df['snpN_roi']<=1).sum()))
            summary_df = summary_df[summary_df['snpN_roi'] > 1]
            
    if op.classby in summary_df.columns:
        print('drop %d unclassified neurons'%(summary_df[op.classby].isnull().sum()))
        summary_df = summary_df[summary_df[op.classby].notnull()]
    summary_df = summary_df.reset_index(drop=True)
    
    classes = sorted(summary_df[op.classby].unique().tolist()) if op.classby in summary_df.columns else ['All']
    classN = len(classes)
    print('categorized by %s:'%(op.classby), classes)
    
    # Color handling
    colors = op.facecolor
    if isinstance(colors, list):
        multi_factor = int(np.ceil(classN / len(colors)))
        if multi_factor > 1: 
            print('Repeated colors were used in plot.')
            colors *= multi_factor
        colors = colors[:classN]
    
    legend_handles = [mp.Patch(color=colors[i],label=classes[i]) for i in range(len(classes))]
    
    # Subplot layout
    lower = int(np.sqrt(classN))
    upper = int(np.ceil(np.sqrt(classN)))
    if lower**2 <= classN <= lower*upper:
        rowN = lower
        colN = upper
    elif lower*upper < classN < upper**2:
        rowN = upper
        colN = upper
    else:
        rowN = upper
        colN = upper
    rowN = max(rowN,2)
    colN = max(colN,2)
    print("subplot size: rowN = %d,colN = %d"%(rowN,colN))
    
    fig, ax = plt.subplots(tight_layout=True,dpi=op.dpi,subplot_kw={'aspect': 'equal'})
    fig_sup, axes = plt.subplots(nrows=rowN,ncols=colN,sharex=True,sharey=True,dpi=op.dpi,subplot_kw={'aspect': 'equal'})
    np.vectorize(lambda axes:axes.axis('off'))(axes)
    fig_sup.suptitle(op.title+'_subplots')
    
    ellipses = []
    skeletons = []
    
    # Load skeletons if needed
    if op.toPlot == 'skeleton':
        if 'fafb' in str(op.dataset).lower() or 'flywire' in str(op.dataset).lower():
            import fafb_utils
            import zipfile
            import io
            
            if op.data_folder is None:
                 project_root = os.path.dirname(os.path.dirname(__file__))
                 op.data_folder = os.path.join(project_root, "datasets", "flywire_FAFB_v783")
            
            zip_path = fafb_utils.get_fafb_skeleton_zip(op.data_folder)
            if zip_path:
                print(f"Loading skeletons from {zip_path}...")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        all_files = set(z.namelist())
                        for ind in summary_df.index:
                            bodyid = str(summary_df.at[ind,'bodyId'])
                            filename = f"{bodyid}.swc"
                            if filename in all_files:
                                with z.open(filename) as f:
                                    content = f.read()
                                    try:
                                        n = navis.read_swc(io.BytesIO(content))
                                        n.name = bodyid
                                        # Assign color based on class
                                        cls = summary_df.at[ind, op.classby]
                                        cls_idx = classes.index(cls)
                                        n.color = colors[cls_idx]
                                        skeletons.append(n)
                                    except Exception as e:
                                        print(f"Error reading SWC for {bodyid}: {e}")
                            else:
                                # print(f"Skeleton for {bodyid} not found in zip.")
                                pass
                except Exception as e:
                    print(f"Error opening zip file: {e}")
        else:
            # Hemibrain fetch
            try:
                from neuprint import fetch_skeletons
                # Batch fetch might be better but let's do simple for now
                # Or use navis.fetch_skeletons
                # Assuming fetch_skeletons returns NeuronList
                # We need to map colors
                pass 
            except ImportError:
                pass

    for i,cla in enumerate(classes):
        df = summary_df[summary_df[op.classby] == cla]
        ax_x = i % rowN
        ax_y = int(i / rowN)
        
        # Plot mesh
        if roimesh:
            navis.plot2d(roimesh,method='2d',ax=axes[ax_x,ax_y],view=(op.plane[0],op.plane[1]),color=op.mesh_color,alpha=op.mesh_alpha)
        
        if op.toPlot == 'skeleton':
            # Filter skeletons for this class
            class_skels = [s for s in skeletons if getattr(s, 'name', '') in df['bodyId'].astype(str).values]
            if class_skels:
                navis.plot2d(class_skels, method='2d', ax=axes[ax_x,ax_y], view=(op.plane[0],op.plane[1]), color=colors[i], alpha=op.alpha)
                
        elif op.toPlot == 'soma':
            for ind in df.index:
                if 'somaLocation' in df.columns and isinstance(df.at[ind,'somaLocation'], str):
                    somaLoc_str = df.at[ind,'somaLocation'][1:-1].split(', ')
                    name_str = 'xyz'
                    somaLoc = {name_str[k]: int(somaLoc_str[k]) for k in range(3)}
                    e = mp.Circle(xy = (somaLoc[op.plane[0]], somaLoc[op.plane[1]]),
                            radius = df.at[ind,'somaRadius'] if 'somaRadius' in df.columns else 100,
                            alpha = op.alpha,
                            facecolor = colors[i],
                    )
                    ellipses.append(copy(e))
                    axes[ax_x,ax_y].add_patch(copy(e))
                    
    # Save
    if op.save_path:
        plt.savefig(op.save_path + op.save_format)
        print(f"Saved figure to {op.save_path + op.save_format}")
        
    if op.showfig:
        plt.show()

def build_synapse_mesh(pre_coords, post_coords, mode='sphere', size=100, color='red', opacity=1.0, name='synapses'):
    """
    Build a single mesh containing all synapses with specified geometry.
    
    Parameters
    ----------
    pre_coords : pd.DataFrame or np.ndarray
        (N, 3) array of pre-synaptic coordinates (x, y, z)
    post_coords : pd.DataFrame or np.ndarray
        (N, 3) array of post-synaptic coordinates (x, y, z)
    mode : str
        'sphere', 'cone', or 'tetrahedron'
    size : float or np.ndarray
        Size of the glyphs (radius or scale). Can be a single float or an array of shape (N,).
    color : str or list
        Color of the mesh
    opacity : float
        Opacity of the mesh
    name : str
        Name for the trace
        
    Returns
    -------
    go.Mesh3d
        Plotly Mesh3d trace
    """
    import numpy as np
    import plotly.graph_objects as go
    
    # Convert to numpy arrays
    if hasattr(pre_coords, 'values'):
        pre_coords = pre_coords.values
    if hasattr(post_coords, 'values'):
        post_coords = post_coords.values
        
    n_synapses = len(pre_coords)
    if n_synapses == 0:
        return go.Mesh3d()
        
    # Calculate midpoints and direction vectors
    midpoints = (pre_coords + post_coords) / 2
    vectors = post_coords - pre_coords
    
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1)
    # Handle zero length vectors (shouldn't happen but good to be safe)
    norms[norms == 0] = 1
    directions = vectors / norms[:, np.newaxis]
    
    # Define template geometry (centered at origin, aligned with Z axis)
    # Use unit size for template, scale later
    template_size = 1.0
    
    if mode == 'sphere':
        # UV Sphere with 48 faces (8 segments, 4 rings)
        # This provides a smoother approximation than the icosahedron (20 faces)
        N = 8 # segments
        M = 4 # rings (stacks)
        verts = []
        
        # North Pole
        verts.append([0, 0, template_size])
        
        # Rings
        for i in range(1, M):
            phi = np.pi * i / M
            z = template_size * np.cos(phi)
            r_ring = template_size * np.sin(phi)
            for j in range(N):
                theta = 2 * np.pi * j / N
                x = r_ring * np.cos(theta)
                y = r_ring * np.sin(theta)
                verts.append([x, y, z])
        
        # South Pole
        verts.append([0, 0, -template_size])
        
        verts_template = np.array(verts)
        
        faces = []
        # Top cap: North Pole (0) to Ring 1 (1..N)
        for j in range(N):
            idx1 = 1 + j
            idx2 = 1 + (j + 1) % N
            faces.append([0, idx1, idx2])
            
        # Middle rings
        for i in range(M - 2):
            start_curr = 1 + i * N
            start_next = 1 + (i + 1) * N
            for j in range(N):
                curr1 = start_curr + j
                curr2 = start_curr + (j + 1) % N
                next1 = start_next + j
                next2 = start_next + (j + 1) % N
                
                faces.append([curr1, next1, curr2])
                faces.append([next1, next2, curr2])
                
        # Bottom cap: Ring M-1 to South Pole (last index)
        last_idx = len(verts) - 1
        start_last_ring = 1 + (M - 2) * N
        for j in range(N):
            idx1 = start_last_ring + j
            idx2 = start_last_ring + (j + 1) % N
            faces.append([last_idx, idx2, idx1])
            
        faces_template = np.array(faces)
        
    elif mode == 'cone':
        # Cone with 48 faces (24 segments)
        # Pointing along Z.
        s = template_size
        N = 24 # segments
        
        verts = []
        # Tip
        verts.append([0, 0, s]) # Index 0
        # Base Center
        verts.append([0, 0, -s]) # Index 1
        
        # Base Ring
        for i in range(N):
            theta = 2 * np.pi * i / N
            x = s * np.cos(theta)
            y = s * np.sin(theta)
            verts.append([x, y, -s])
            
        verts_template = np.array(verts)
        
        faces = []
        # Sides: Tip (0) to Ring (2..N+1)
        for i in range(N):
            idx1 = 2 + i
            idx2 = 2 + (i + 1) % N
            faces.append([0, idx1, idx2])
            
        # Base: Base Center (1) to Ring (2..N+1)
        # Note: Order reversed for correct normal
        for i in range(N):
            idx1 = 2 + i
            idx2 = 2 + (i + 1) % N
            faces.append([1, idx2, idx1])
            
        faces_template = np.array(faces)
        
    elif mode == 'tetrahedron':
        # Tetrahedron (4 vertices, 4 faces)
        # "pre-post direction length ... 50% longer"
        # Standard tetrahedron inscribed in cube
        s = template_size
        z_scale = 1.5
        verts_template = np.array([
            [s, s, s*z_scale],
            [s, -s, -s*z_scale],
            [-s, s, -s*z_scale],
            [-s, -s, s*z_scale]
        ])
        faces_template = np.array([
            [0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]
        ])
    else:
        raise ValueError(f"Unknown mode: {mode}")
        
    n_verts_template = len(verts_template)
    n_faces_template = len(faces_template)
    
    # Construct rotation matrices to align Z axis (0,0,1) to 'directions'
    # R = [X_new, Y_new, Z_new]
    # Z_new = direction
    # X_new = cross(up, Z_new). If Z_new ~ up, use another up.
    
    z_axis = directions
    
    # Arbitrary up vector (0, 1, 0)
    up = np.array([0, 1, 0])
    
    # Check for parallel vectors
    cross_prod = np.cross(up, z_axis)
    cross_norms = np.linalg.norm(cross_prod, axis=1)
    
    # If parallel to Y, use X as up
    mask_parallel = cross_norms < 1e-6
    if np.any(mask_parallel):
        # Create a copy of up vectors
        up_vectors = np.tile(up, (n_synapses, 1))
        up_vectors[mask_parallel] = np.array([1, 0, 0])
        x_axis = np.cross(up_vectors, z_axis)
    else:
        x_axis = np.cross(up, z_axis)
        
    # Normalize X
    x_norms = np.linalg.norm(x_axis, axis=1)
    x_axis = x_axis / x_norms[:, np.newaxis]
    
    # Y = Z x X
    y_axis = np.cross(z_axis, x_axis)
    
    # Construct Rotation Matrices (N, 3, 3)
    # R[i] = [x_axis[i], y_axis[i], z_axis[i]] (columns)
    # So R[i] @ v_template should align v_template to world
    # v_world = R @ v_local
    
    # Stack axes to form R: shape (N, 3, 3)
    # Transpose to get columns: stack along last axis
    R = np.stack([x_axis, y_axis, z_axis], axis=2)
    
    # Apply rotation
    rotated_verts = np.einsum('nij,vj->nvi', R, verts_template)
    
    # Apply scaling
    if np.isscalar(size):
        rotated_verts *= size
    else:
        size_arr = np.array(size)
        if size_arr.ndim == 1 and len(size_arr) == n_synapses:
             rotated_verts *= size_arr[:, np.newaxis, np.newaxis]
        else:
             # Fallback if size is not compatible
             rotated_verts *= size
    
    # Add midpoints
    # (N, V, 3) + (N, 1, 3)
    final_verts = rotated_verts + midpoints[:, np.newaxis, :]
    
    # Flatten vertices
    all_verts = final_verts.reshape(-1, 3)
    
    # Construct faces
    # Faces indices need to be offset
    # (N, F, 3)
    offsets = np.arange(n_synapses) * n_verts_template
    all_faces = faces_template[np.newaxis, :, :] + offsets[:, np.newaxis, np.newaxis]
    all_faces = all_faces.reshape(-1, 3)
    
    # Create Mesh3d
    return go.Mesh3d(
        x=all_verts[:, 0],
        y=all_verts[:, 1],
        z=all_verts[:, 2],
        i=all_faces[:, 0],
        j=all_faces[:, 1],
        k=all_faces[:, 2],
        color=color,
        opacity=opacity,
        name=name,
        lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.1, specular=0.1),
        lightposition=dict(x=1000, y=1000, z=2000)
    )

