"""
NeuronBridge Finder Module

This module provides a wrapper around the NeuronBridge Python API for finding
correspondences between EM reconstructed neurons (bodyIds) and light microscopy 
(LM) GAL4 driver lines.

Key features:
- Convert neuron bodyIds to matching driver lines (id_to_lines)
- Convert neuron queries (bodyId/instance/type) to lines (neuron_to_lines)
- Reverse search: find neurons matching a driver line (line_to_neuron)
- Support for both CDS and PPPM matching algorithms
- Multi-dataset support (hemibrain, male-cns, FlyWire, etc.)
- Automatic dataset detection from NeuronBridge image metadata
- Caching of API results to reduce redundant calls
- CSV export/import for offline analysis

Dependencies:
- neuronbridge-python
- pandas

Example usage:
    from neuronbridge_finder import NeuronBridgeFinder
    
    nbf = NeuronBridgeFinder()
    
    # Find lines matching a bodyId
    lines = nbf.id_to_lines(636798093, top_n=10)
    
    # Find lines matching a neuron type
    results = nbf.neuron_to_lines('MBON01', top_n=5)
    
    # Find neurons matching a driver line (returns neurons from all datasets)
    neurons = nbf.line_to_neuron('LH173', top_n=10)

Author: Generated for hemibrain-connectomes-analysis project
"""

import os
import re
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None

# Try to import neuronbridge
# Apply patch to fix API compatibility issue with pydantic validation
# (NeuronBridge API added new fields 'defaultSearchLibrary' that the Python client doesn't recognize)
NEURONBRIDGE_AVAILABLE = False

def _patch_neuronbridge_models():
    """
    Patch neuronbridge pydantic models to allow extra fields.
    
    The NeuronBridge API has added new fields that the Python client doesn't
    recognize, causing pydantic validation errors. This patches the models
    to ignore unknown fields.
    """
    try:
        import neuronbridge.client as nb_client
        
        # Get the original CustomSearchConfig class
        OriginalCustomSearchConfig = nb_client.CustomSearchConfig
        
        # Create a new class that allows extra fields
        from pydantic import BaseModel
        
        class PatchedCustomSearchConfig(BaseModel):
            """Patched version that allows extra fields."""
            model_config = {'extra': 'ignore'}
            searchFolder: str = ""
            lmLibraries: list = []
            emLibraries: list = []
        
        # Replace in the module
        nb_client.CustomSearchConfig = PatchedCustomSearchConfig
        
        # Also need to patch DataStore which contains customSearch
        OriginalDataStore = nb_client.DataStore
        
        class PatchedDataStore(BaseModel):
            """Patched version that allows extra fields."""
            model_config = {'extra': 'ignore'}
            label: str = ""
            anatomicalArea: str = ""
            prefixes: dict = {}
            customSearch: PatchedCustomSearchConfig = None
            
            def __init__(self, **data):
                # Handle customSearch conversion
                if 'customSearch' in data and data['customSearch'] is not None:
                    if not isinstance(data['customSearch'], PatchedCustomSearchConfig):
                        data['customSearch'] = PatchedCustomSearchConfig(**data['customSearch'])
                super().__init__(**data)
        
        nb_client.DataStore = PatchedDataStore
        
        # Patch DataConfig
        OriginalDataConfig = nb_client.DataConfig
        
        class PatchedDataConfig(BaseModel):
            """Patched version that allows extra fields."""
            model_config = {'extra': 'ignore'}
            stores: dict = {}
            
            def __init__(self, **data):
                # Convert stores dict values to PatchedDataStore
                if 'stores' in data:
                    new_stores = {}
                    for k, v in data['stores'].items():
                        if isinstance(v, dict):
                            new_stores[k] = PatchedDataStore(**v)
                        else:
                            new_stores[k] = v
                    data['stores'] = new_stores
                super().__init__(**data)
        
        nb_client.DataConfig = PatchedDataConfig
        
        return True
    except Exception as e:
        import warnings
        warnings.warn(f"Failed to patch neuronbridge models: {e}")
        return False

try:
    # Apply patches before creating client
    _patch_neuronbridge_models()
    from neuronbridge.client import Client as NBClient
    NEURONBRIDGE_AVAILABLE = True
except ImportError:
    warnings.warn(
        "neuronbridge-python is not installed. "
        "Install it with: pip install neuronbridge-python"
    )
except Exception as e:
    warnings.warn(f"Failed to initialize neuronbridge: {e}")

# Mapping from NeuronBridge library names to our local dataset folder names
LIBRARY_TO_DATASET = {
    'FlyEM_Hemibrain_v1.2.1': 'hemibrain_v1_2_1',
    'FlyEM_Hemibrain_v1.2': 'hemibrain_v1_2_1',
    'FlyEM_Hemibrain': 'hemibrain_v1_2_1',
    'FlyEM_MANC_v1.0': 'male-cns_v0_9',
    'FlyEM_MANC': 'male-cns_v0_9',
    'FlyEM_Male_CNS_Brain_v0.9': 'male-cns_v0_9',
    'FlyEM_Male_CNS_v0.9': 'male-cns_v0_9',
    'FlyEM_Male_CNS': 'male-cns_v0_9',
    'FlyWire_FAFB': 'flywire_FAFB_v783',
    'FlyWire_FAFB_v783': 'flywire_FAFB_v783',
    'FlyWire_BANC': 'flywire_BANC_v626',
    'FlyWire_BANC_v626': 'flywire_BANC_v626',
    'FlyEM_Optic_Lobe': 'optic-lobe_v1_1',
}

# Mapping to human-readable dataset names (for display/neuprint format)
LIBRARY_TO_DATASET_NAME = {
    'FlyEM_Hemibrain_v1.2.1': 'hemibrain:v1.2.1',
    'FlyEM_Hemibrain_v1.2': 'hemibrain:v1.2.1',
    'FlyEM_Hemibrain': 'hemibrain:v1.2.1',
    'FlyEM_MANC_v1.0': 'male-cns:v0.9',
    'FlyEM_MANC': 'male-cns:v0.9',
    'FlyEM_Male_CNS_Brain_v0.9': 'male-cns:v0.9',
    'FlyEM_Male_CNS_v0.9': 'male-cns:v0.9',
    'FlyEM_Male_CNS': 'male-cns:v0.9',
    'FlyWire_FAFB': 'flywire_FAFB_v783',
    'FlyWire_FAFB_v783': 'flywire_FAFB_v783',
    'FlyWire_BANC': 'flywire_BANC_v626',
    'FlyWire_BANC_v626': 'flywire_BANC_v626',
    'FlyEM_Optic_Lobe': 'optic-lobe:v1.1',
}


# Line name prefixes for classification
# GAL4/LexA lines: typically VT (Vienna Tile), R (Rubin), GMR, etc.
GAL4_LEXA_PREFIXES = ('VT', 'R', 'GMR')
# Split-GAL4 lines: SS (Split Screen), LH (Lateral Horn), MB (Mushroom Body), etc.
SPLIT_GAL4_PREFIXES = ('SS', 'LH', 'MB', 'IS', 'OL', 'LC', 'LLPC', 'LPC', 'JRC_SS', 'BJD_SS')


@dataclass
class NeuronBridgeFinder:
    """
    A class for finding correspondences between EM neurons and LM driver lines.
    
    This class wraps the NeuronBridge API to provide convenient methods for
    mapping between EM body IDs and GAL4 driver lines across multiple datasets.
    
    Parameters
    ----------
    datasets_path : str, optional
        Path to the datasets folder containing neuron_df CSV files.
        Default: auto-detect from module location.
    use_cache : bool
        Whether to cache API results locally. Default: True
    cache_folder : str, optional
        Folder for cached results. Default: auto-detect.
    verbose : bool
        Print progress messages. Default: True
    separate_splitgal4 : bool
        If True, separate results into GAL4/LexA and Split-GAL4 categories.
        When enabled, download_top_n_img applies separately to each category.
        Default: False
    neuprint_token : str, optional
        NeuPrint API token for pulling missing datasets. If not provided, will check
        NEUPRINT_TOKEN or NEUPRINT_APPLICATION_CREDENTIALS environment variables.
        Without a token, local dataset features (type lookups, specificity) will be skipped.
        Get your token at: https://neuprint.janelia.org/account
    neuprint_server : str
        NeuPrint server URL. Default: 'https://neuprint.janelia.org'
    match_type : str
        Default match algorithm: 'cds', 'pppm', or 'both'. Default: 'cds'
    region : str
        Filter images by anatomical region: 'Brain', 'VNC', or 'All'. Default: 'All'
    
    Attributes
    ----------
    _client : NBClient
        The NeuronBridge API client.
    _neuron_dfs : dict
        Dictionary mapping dataset names to loaded neuron DataFrames.
    
    Example
    -------
    >>> nbf = NeuronBridgeFinder()
    >>> lines = nbf.id_to_lines(636798093, top_n=5)
    >>> print(lines)
    
    # Separate GAL4/LexA from Split-GAL4 lines
    >>> nbf = NeuronBridgeFinder(separate_splitgal4=True)
    >>> results = nbf.find_lines_batch('MBON01', download_top_n_img=5)  # 5 GAL4 + 5 Split-GAL4
    
    # With NeuPrint token for pulling missing datasets
    >>> nbf = NeuronBridgeFinder(neuprint_token='your_token_here')
    """
    
    datasets_path: Optional[str] = None
    use_cache: bool = True
    cache_folder: Optional[str] = None
    verbose: bool = True
    separate_splitgal4: bool = False
    neuprint_token: Optional[str] = None
    neuprint_server: str = 'https://neuprint.janelia.org'
    match_type: str = 'cds'
    region: str = 'All'
    
    # Private fields
    _client: Any = field(init=False, repr=False, default=None)
    _neuron_dfs: Dict[str, pd.DataFrame] = field(init=False, repr=False, default_factory=dict)
    _suppress_loading_msgs: bool = field(init=False, repr=False, default=False)
    
    def __post_init__(self):
        """Initialize the finder after dataclass initialization."""
        if not NEURONBRIDGE_AVAILABLE:
            raise ImportError(
                "neuronbridge-python is required. Install with: pip install neuronbridge-python"
            )
        
        # Set default datasets path
        if self.datasets_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(module_dir)
            self.datasets_path = os.path.join(project_root, 'datasets')
        
        # Set default cache folder
        if self.cache_folder is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(module_dir)
            self.cache_folder = os.path.join(project_root, 'cache', 'neuronbridge')
        
        # Create cache folder if needed
        if self.use_cache:
            os.makedirs(self.cache_folder, exist_ok=True)
        
        # Initialize client
        self._init_client()
    
    def _vprint(self, msg: str, end: str = '\n'):
        """
        Print message if verbose mode is enabled.
        
        Uses tqdm.write() if tqdm is available and progress bars might be active,
        otherwise falls back to regular print().
        
        Parameters
        ----------
        msg : str
            Message to print
        end : str
            String appended after the message (default: newline)
        """
        if self.verbose:
            # Use tqdm.write if available to avoid conflicts with progress bars
            if HAS_TQDM:
                try:
                    from tqdm import tqdm
                    tqdm.write(msg, end=end)
                except:
                    print(msg, end=end)
            else:
                print(msg, end=end)
    
    def _retry_with_backoff(
        self,
        func,
        *args,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        **kwargs
    ):
        """
        Retry a function with exponential backoff for transient network errors.
        
        Parameters
        ----------
        func : callable
            Function to retry
        max_retries : int
            Maximum number of retry attempts (default: 3)
        initial_delay : float
            Initial delay in seconds (default: 1.0)
        *args, **kwargs
            Arguments to pass to func
            
        Returns
        -------
        Any
            Result from func, or None if all retries failed
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check if it's a retryable network error
                is_retryable = any([
                    'incompleteread' in error_str,
                    'ssl' in error_str and 'eof' in error_str,
                    'connection' in error_str and 'broken' in error_str,
                    'max retries exceeded' in error_str,
                    'connection reset' in error_str,
                    'timeout' in error_str
                ])
                
                if not is_retryable or attempt >= max_retries - 1:
                    # Not retryable or last attempt - give up
                    raise
                
                # Wait with exponential backoff
                delay = initial_delay * (2 ** attempt)
                time.sleep(delay)
        
        # Should never reach here, but just in case
        if last_error:
            raise last_error
    
    def _normalize_dataset_name(self, dataset: str) -> str:
        """
        Normalize dataset name for comparison.
        
        Handles different naming conventions:
        - 'flywire_FAFB_v783' -> 'flywire-fafb'
        - 'flywire_fafb:v783' -> 'flywire-fafb'
        - 'male-cns:v0.9' -> 'male-cns'
        - 'hemibrain:v1.2.1' -> 'hemibrain'
        - 'hemibrain_v1_2_1' -> 'hemibrain'
        
        Parameters
        ----------
        dataset : str
            Dataset name in any format
            
        Returns
        -------
        str
            Normalized dataset name (lowercase, no version, hyphens for separators)
        """
        import re
        
        # Convert to lowercase
        ds = dataset.lower()
        
        # Remove version info in various formats:
        # - ':v1.2.1' or ':v783' (colon format)
        # - '_v1_2_1' or '_v783' (underscore format)
        # First handle colon-separated version
        ds = ds.split(':')[0]
        
        # Remove underscore-separated version suffix (e.g., '_v783', '_v1_2_1')
        # Match _v followed by digits and optional underscored sub-versions
        ds = re.sub(r'_v\d+(_\d+)*$', '', ds)
        
        # Normalize separators: convert underscores to hyphens
        ds = ds.replace('_', '-')
        
        return ds
    
    def _filter_images_by_region(self, images: List[Any]) -> List[Any]:
        """
        Filter images based on the anatomical region setting.
        
        Filters LM or EM images based on their anatomicalArea attribute.
        Only processes filtering when self.region is 'Brain' or 'VNC'.
        
        Parameters
        ----------
        images : list
            List of LM or EM image objects from NeuronBridge API.
            
        Returns
        -------
        list
            Filtered list of images matching the region criteria.
        """
        if self.region == 'All':
            return images
        
        filtered = []
        for img in images:
            # Get anatomicalArea attribute if it exists
            area = getattr(img, 'anatomicalArea', None)
            if area is None:
                # If no area specified, include by default
                filtered.append(img)
                continue
            
            # Normalize area name (case-insensitive)
            area_lower = area.lower()
            
            if self.region == 'Brain':
                # Include if area contains 'brain' but not 'vnc'
                if 'brain' in area_lower and 'vnc' not in area_lower:
                    filtered.append(img)
            elif self.region == 'VNC':
                # Include if area contains 'vnc'
                if 'vnc' in area_lower:
                    filtered.append(img)
        
        return filtered
    
    def _classify_line_type(self, line_name: str) -> str:
        """
        Classify a driver line as 'gal4_lexa' or 'split_gal4'.
        
        Classification rules:
        - GAL4/LexA: Lines starting with VT, R, GMR (e.g., VT037867, R10A06)
        - Split-GAL4: Lines starting with SS, LH, MB, IS, OL, LC, etc.
        
        Parameters
        ----------
        line_name : str
            The driver line name to classify.
            
        Returns
        -------
        str
            'gal4_lexa' or 'split_gal4'
        """
        if not line_name:
            return 'gal4_lexa'  # default
        
        line_upper = line_name.upper()
        
        # Check Split-GAL4 prefixes first (more specific)
        for prefix in SPLIT_GAL4_PREFIXES:
            if line_upper.startswith(prefix.upper()):
                return 'split_gal4'
        
        # Check GAL4/LexA prefixes
        for prefix in GAL4_LEXA_PREFIXES:
            if line_upper.startswith(prefix.upper()):
                return 'gal4_lexa'
        
        # Default to gal4_lexa for unknown prefixes
        return 'gal4_lexa'
    
    def _separate_lines_by_type(
        self, 
        lines_df: pd.DataFrame,
        line_column: str = 'line'
    ) -> Dict[str, pd.DataFrame]:
        """
        Separate a DataFrame of lines into GAL4/LexA and Split-GAL4 categories.
        
        Parameters
        ----------
        lines_df : pd.DataFrame
            DataFrame containing line results.
        line_column : str
            Column name containing line names. Default: 'line'
            
        Returns
        -------
        dict
            Dictionary with 'gal4_lexa' and 'split_gal4' keys, each containing
            a DataFrame of lines for that category.
        """
        if lines_df.empty or line_column not in lines_df.columns:
            return {'gal4_lexa': pd.DataFrame(), 'split_gal4': pd.DataFrame()}
        
        # Add line_type column
        lines_df = lines_df.copy()
        lines_df['line_type'] = lines_df[line_column].apply(self._classify_line_type)
        
        return {
            'gal4_lexa': lines_df[lines_df['line_type'] == 'gal4_lexa'].copy(),
            'split_gal4': lines_df[lines_df['line_type'] == 'split_gal4'].copy()
        }
    
    def _calculate_expression_entropy(self, type_counts: Dict[str, int]) -> float:
        """
        Calculate Shannon entropy of expression pattern.
        
        H = -Σ p_i × log2(p_i)
        
        Lower entropy = more specific (labels fewer types more strongly)
        Higher entropy = more promiscuous (labels many types equally)
        
        Parameters
        ----------
        type_counts : dict
            Dictionary mapping neuron types to their occurrence counts.
            
        Returns
        -------
        float
            Shannon entropy in bits. Range: 0 (perfectly specific) to log2(N) (uniform).
        """
        if not type_counts:
            return 0.0
        
        total = sum(type_counts.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in type_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _calculate_weighted_specificity(
        self, 
        neurons_df: pd.DataFrame,
        queried_types_lower: set,
        score_column: str = 'score'
    ) -> Dict[str, float]:
        """
        Calculate expression-strength-weighted specificity metrics.
        
        Weights each neuron's contribution by its NeuronBridge match score,
        so high-confidence matches count more than low-confidence ones.
        
        Parameters
        ----------
        neurons_df : pd.DataFrame
            DataFrame from line_to_neuron with 'type' and score columns.
        queried_types_lower : set
            Set of queried neuron types (lowercase).
        score_column : str
            Name of score column to use for weighting.
            
        Returns
        -------
        dict
            Dictionary with:
            - 'weighted_type_proportion': Score-weighted proportion of queried types
            - 'weighted_queried_score': Total score for queried types
            - 'weighted_total_score': Total score for all types
            - 'mean_queried_score': Mean score for queried types
        """
        if neurons_df.empty or score_column not in neurons_df.columns:
            return {
                'weighted_type_proportion': 0.0,
                'weighted_queried_score': 0.0,
                'weighted_total_score': 0.0,
                'mean_queried_score': 0.0
            }
        
        # Calculate total weighted scores
        total_score = neurons_df[score_column].sum()
        
        # Get scores for queried types
        queried_mask = neurons_df['type'].fillna('').str.lower().isin(queried_types_lower)
        queried_scores = neurons_df.loc[queried_mask, score_column]
        
        queried_total = queried_scores.sum()
        queried_mean = queried_scores.mean() if len(queried_scores) > 0 else 0.0
        
        # Weighted proportion
        weighted_proportion = queried_total / total_score if total_score > 0 else 0.0
        
        return {
            'weighted_type_proportion': weighted_proportion,
            'weighted_queried_score': queried_total,
            'weighted_total_score': total_score,
            'mean_queried_score': queried_mean
        }
    
    def _build_colabeling_matrix(
        self,
        lines: List[str],
        match_type: str = 'cds',
        top_n: int = 100
    ) -> Tuple[pd.DataFrame, Dict[str, set]]:
        """
        Build a co-labeling matrix showing how often pairs of lines label the same neurons.
        
        The matrix M[i,j] represents the Jaccard similarity between lines i and j:
        J = |A ∩ B| / |A ∪ B|
        
        Parameters
        ----------
        lines : list of str
            List of driver line names.
        match_type : str
            Match algorithm for line_to_neuron.
        top_n : int
            Number of top matches to consider per line.
            
        Returns
        -------
        tuple
            (co_labeling_matrix, line_neuron_sets)
            - co_labeling_matrix: DataFrame with Jaccard similarities
            - line_neuron_sets: Dict mapping line names to sets of (bodyId, type) tuples
        """
        self._vprint(f"\n🔗 Building co-labeling matrix for {len(lines)} lines...")
        self._vprint(f"   ⏱️  Note: Fetching neurons for each line to build similarity matrix")
        
        # Collect neuron sets for each line
        line_neuron_sets = {}
        
        if HAS_TQDM and self.verbose:
            from tqdm import tqdm as tqdm_progress
            iterator = tqdm_progress(
                lines, 
                desc="   🔍 Collecting neurons per line", 
                unit="line",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                ncols=100
            )
        else:
            iterator = lines
        
        for idx, line_name in enumerate(iterator):
            if HAS_TQDM and self.verbose:
                iterator.set_description(f"   🔍 [{idx+1}/{len(lines)}] Fetching neurons for {line_name}")
            try:
                neurons_df = self.line_to_neuron(line_name, match_type=match_type)
                if not neurons_df.empty:
                    neurons_top = neurons_df.head(top_n)
                    # Use (bodyId, type) pairs for more precise matching
                    neuron_set = set()
                    for _, row in neurons_top.iterrows():
                        body_id = row.get('bodyId', '')
                        n_type = row.get('type', '')
                        if body_id:
                            neuron_set.add((str(body_id), str(n_type).lower()))
                    line_neuron_sets[line_name] = neuron_set
                else:
                    line_neuron_sets[line_name] = set()
            except Exception as e:
                line_neuron_sets[line_name] = set()
                if self.verbose:
                    self._vprint(f"   ⚠️ Error getting neurons for {line_name}: {e}")
        
        # Build Jaccard similarity matrix
        self._vprint(f"   🔢 Computing Jaccard similarities between {len(lines)} lines...")
        n = len(lines)
        matrix = np.zeros((n, n))
        
        for i, line_i in enumerate(lines):
            set_i = line_neuron_sets.get(line_i, set())
            for j, line_j in enumerate(lines):
                if i == j:
                    matrix[i, j] = 1.0  # Self-similarity is 1
                elif j > i:  # Only compute upper triangle
                    set_j = line_neuron_sets.get(line_j, set())
                    if set_i and set_j:
                        intersection = len(set_i & set_j)
                        union = len(set_i | set_j)
                        jaccard = intersection / union if union > 0 else 0.0
                    else:
                        jaccard = 0.0
                    matrix[i, j] = jaccard
                    matrix[j, i] = jaccard  # Symmetric
        
        co_labeling_df = pd.DataFrame(matrix, index=lines, columns=lines)
        
        return co_labeling_df, line_neuron_sets
    
    def _calculate_colabeling_sparsity(self, co_labeling_matrix: pd.DataFrame, threshold: float = 0.1) -> Dict[str, float]:
        """
        Calculate sparsity metrics from co-labeling matrix.
        
        Sparsity measures how unique each line's labeling pattern is.
        High sparsity = line labels neurons that few other lines label.
        
        Parameters
        ----------
        co_labeling_matrix : pd.DataFrame
            Jaccard similarity matrix from _build_colabeling_matrix.
        threshold : float
            Similarity threshold to consider lines as "co-labeling".
            
        Returns
        -------
        dict
            Dictionary mapping line names to sparsity scores (0-1, higher = more unique).
        """
        sparsity_scores = {}
        
        for line in co_labeling_matrix.index:
            # Get similarities with other lines (exclude self)
            similarities = co_labeling_matrix.loc[line].drop(line)
            
            # Count how many lines have significant overlap
            n_colabeling = (similarities > threshold).sum()
            total_other_lines = len(similarities)
            
            # Sparsity = 1 - (proportion of co-labeling lines)
            sparsity = 1.0 - (n_colabeling / total_other_lines) if total_other_lines > 0 else 1.0
            
            # Also compute mean non-self similarity (lower = more unique)
            mean_similarity = similarities.mean() if len(similarities) > 0 else 0.0
            
            sparsity_scores[line] = {
                'colabel_sparsity': sparsity,
                'n_colabeling_lines': int(n_colabeling),
                'mean_colabel_similarity': mean_similarity
            }
        
        return sparsity_scores
    
    def visualize_colabeling_matrix(
        self,
        co_labeling_matrix: pd.DataFrame,
        output_path: str,
        title: str = "Co-Labeling Matrix (Jaccard Similarity)",
        color_scale: str = 'purple'
    ) -> str:
        """
        Visualize co-labeling matrix as a heatmap using CreateHeatmap.
        
        Parameters
        ----------
        co_labeling_matrix : pd.DataFrame
            Jaccard similarity matrix from _build_colabeling_matrix.
        output_path : str
            Directory to save the heatmap HTML file.
        title : str
            Title for the heatmap.
        color_scale : str
            Color scale preset: 'purple', 'green', 'blue', 'orange', 'red'.
            
        Returns
        -------
        str
            Path to the created heatmap file.
        """
        try:
            from .statvis import CreateHeatmap
        except ImportError:
            try:
                from statvis import CreateHeatmap
            except ImportError:
                self._vprint("   ⚠️ Could not import CreateHeatmap from statvis")
                return ""
        
        # Create output directory if needed
        os.makedirs(output_path, exist_ok=True)
        
        # Create heatmap
        hm = CreateHeatmap(output_folder=output_path, showfig=False)
        hm.add_heatmap(
            matrix=co_labeling_matrix,
            name='colabeling_matrix',
            title=title,
            color_scale=color_scale,
            interactive=True  # Enable interactive controls
        )
        
        created_files = hm.create_all()
        
        if created_files:
            self._vprint(f"   📊 Created co-labeling heatmap: {created_files[0]}")
            return created_files[0]
        
        return ""
    
    def _calculate_mutual_information(
        self,
        lines: List[str],
        queried_types: List[str],
        match_type: str = 'cds',
        top_n: int = 100
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate mutual information between driver lines and neuron types.
        
        MI(L; T) = Σ p(l,t) × log2(p(l,t) / (p(l) × p(t)))
        
        This measures how much knowing a line's expression tells you about neuron type.
        High MI = line expression is highly informative about neuron type.
        
        Parameters
        ----------
        lines : list of str
            List of driver line names to analyze.
        queried_types : list of str
            List of queried neuron types (for highlighting in visualization).
        match_type : str
            Match algorithm for line_to_neuron.
        top_n : int
            Number of top matches to consider per line.
            
        Returns
        -------
        tuple
            (mi_df, expression_matrix)
            - mi_df: DataFrame with MI values per line
            - expression_matrix: Binary expression matrix (lines × types)
        """
        self._vprint(f"\n📊 Calculating mutual information for {len(lines)} lines...")
        self._vprint(f"   ⏱️  Note: Fetching neuron types for each line (may take time)")
        
        # Normalize queried types
        queried_types_lower = set(t.lower() for t in queried_types if t)
        
        # Build expression matrix: rows = lines, cols = neuron types
        # Value = 1 if line labels that type, 0 otherwise
        all_types = set()
        line_type_sets = {}
        line_type_scores = {}  # For weighted MI
        
        if HAS_TQDM and self.verbose:
            from tqdm import tqdm as tqdm_progress
            iterator = tqdm_progress(
                lines, 
                desc="   🧬 Building expression matrix", 
                unit="line",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                ncols=100
            )
        else:
            iterator = lines
        
        for idx, line_name in enumerate(iterator):
            if HAS_TQDM and self.verbose:
                iterator.set_description(f"   🧬 [{idx+1}/{len(lines)}] Processing {line_name}")
            try:
                neurons_df = self.line_to_neuron(line_name, match_type=match_type, top_n=top_n)
                if not neurons_df.empty:
                    types = neurons_df['type'].fillna('Unknown').unique()
                    types_lower = set(t.lower() for t in types)
                    line_type_sets[line_name] = types_lower
                    all_types.update(types_lower)
                    
                    # Store scores per type for weighted MI
                    type_scores = neurons_df.groupby(neurons_df['type'].fillna('Unknown').str.lower())['score'].max().to_dict()
                    line_type_scores[line_name] = type_scores
                else:
                    line_type_sets[line_name] = set()
                    line_type_scores[line_name] = {}
            except Exception as e:
                line_type_sets[line_name] = set()
                line_type_scores[line_name] = {}
        
        if not all_types:
            self._vprint("   ⚠️ No types found for any line")
            return pd.DataFrame(), pd.DataFrame()
        
        # Create binary expression matrix
        all_types_list = sorted(all_types)
        expression_matrix = np.zeros((len(lines), len(all_types_list)))
        
        for i, line in enumerate(lines):
            for j, ntype in enumerate(all_types_list):
                if ntype in line_type_sets.get(line, set()):
                    expression_matrix[i, j] = 1
        
        expression_df = pd.DataFrame(
            expression_matrix,
            index=lines,
            columns=all_types_list
        )
        
        # Calculate MI for each line
        # MI(L; T) = H(T) - H(T|L)
        # where H(T) = -Σ p(t) log2(p(t))
        # and H(T|L=l) = -Σ p(t|l) log2(p(t|l))
        
        n_lines = len(lines)
        n_types = len(all_types_list)
        
        # Marginal probability of each type (across all lines)
        type_counts = expression_matrix.sum(axis=0)  # How many lines label each type
        p_type = type_counts / n_lines  # P(type)
        p_type = np.clip(p_type, 1e-10, 1)  # Avoid log(0)
        
        # Entropy of type distribution H(T)
        H_T = -np.sum(p_type * np.log2(p_type))
        
        mi_results = []
        
        for i, line in enumerate(lines):
            types_labeled = line_type_sets.get(line, set())
            n_labeled = len(types_labeled)
            
            if n_labeled == 0:
                mi_results.append({
                    'line': line,
                    'mutual_information': 0.0,
                    'normalized_mi': 0.0,
                    'n_types_labeled': 0,
                    'queried_type_coverage': 0.0
                })
                continue
            
            # Conditional entropy H(T|L=l)
            # P(t|L=l) = 1/n_labeled if line labels type t, 0 otherwise
            p_t_given_l = np.zeros(n_types)
            for j, ntype in enumerate(all_types_list):
                if ntype in types_labeled:
                    p_t_given_l[j] = 1.0 / n_labeled
            
            # H(T|L=l) = -Σ p(t|l) log2(p(t|l))
            p_t_given_l_nonzero = p_t_given_l[p_t_given_l > 0]
            H_T_given_L = -np.sum(p_t_given_l_nonzero * np.log2(p_t_given_l_nonzero))
            
            # Point-wise MI for this line
            # This is simplified: we compute reduction in entropy
            mi_line = H_T - H_T_given_L
            
            # Normalized MI (0-1 scale)
            max_mi = H_T  # Maximum possible MI is H(T)
            normalized_mi = mi_line / max_mi if max_mi > 0 else 0.0
            
            # Queried type coverage
            covered = len(types_labeled & queried_types_lower)
            coverage = covered / len(queried_types_lower) if queried_types_lower else 0.0
            
            mi_results.append({
                'line': line,
                'mutual_information': mi_line,
                'normalized_mi': normalized_mi,
                'n_types_labeled': n_labeled,
                'queried_type_coverage': coverage
            })
        
        mi_df = pd.DataFrame(mi_results)
        
        self._vprint(f"   ✓ MI calculated for {len(mi_df)} lines")
        self._vprint(f"   Total types in analysis: {n_types}")
        self._vprint(f"   Type entropy H(T): {H_T:.3f} bits")
        
        return mi_df, expression_df
    
    def visualize_expression_matrix(
        self,
        expression_df: pd.DataFrame,
        output_path: str,
        queried_types: Optional[List[str]] = None,
        title: str = "Line × Type Expression Matrix",
        color_scale: str = 'green'
    ) -> str:
        """
        Visualize the expression matrix (lines × types) as a heatmap.
        
        Parameters
        ----------
        expression_df : pd.DataFrame
            Binary expression matrix from _calculate_mutual_information.
        output_path : str
            Directory to save the heatmap HTML file.
        queried_types : list of str, optional
            Queried types to highlight in the matrix.
        title : str
            Title for the heatmap.
        color_scale : str
            Color scale preset.
            
        Returns
        -------
        str
            Path to the created heatmap file.
        """
        try:
            from .statvis import CreateHeatmap
        except ImportError:
            try:
                from statvis import CreateHeatmap
            except ImportError:
                self._vprint("   ⚠️ Could not import CreateHeatmap from statvis")
                return ""
        
        os.makedirs(output_path, exist_ok=True)
        
        # Optionally filter to show only queried types + a sample of others
        if queried_types and len(expression_df.columns) > 50:
            queried_lower = [t.lower() for t in queried_types]
            queried_cols = [c for c in expression_df.columns if c in queried_lower]
            other_cols = [c for c in expression_df.columns if c not in queried_lower]
            # Keep queried + top 20 other most common types
            other_counts = expression_df[other_cols].sum().nlargest(20).index.tolist()
            cols_to_show = queried_cols + other_counts
            expression_filtered = expression_df[cols_to_show]
        else:
            expression_filtered = expression_df
        
        # Create heatmap
        hm = CreateHeatmap(output_folder=output_path, showfig=False)
        hm.add_heatmap(
            matrix=expression_filtered,
            name='expression_matrix',
            title=title,
            color_scale=color_scale,
            interactive=True
        )
        
        created_files = hm.create_all()
        
        if created_files:
            self._vprint(f"   📊 Created expression matrix heatmap: {created_files[0]}")
            return created_files[0]
        
        return ""

    def _calculate_line_specificity(
        self,
        line_stats: pd.DataFrame,
        queried_types: List[str],
        match_type: str = 'cds',
        top_n: int = 100,
        max_lines: int = 100
    ) -> pd.DataFrame:
        """
        Calculate comprehensive specificity metrics for driver lines.
        
        For each line, computes:
        - rank_sum: Sum of ranks for queried neuron types in line_to_neuron results
        - type_proportion: Proportion of queried types among all labeled types (top N)
        - n_queried_types: Number of queried types labeled by this line
        - n_total_types: Total number of distinct types labeled by this line
        - selectivity: 1 / n_total_types (higher = more selective)
        - expression_entropy: Shannon entropy of type distribution (lower = more specific)
        - normalized_entropy: Entropy normalized by max possible (0-1, lower = more specific)
        - weighted_type_proportion: Type proportion weighted by match scores
        - mean_queried_score: Mean match score for queried types
        - specificity_score: Composite score combining all metrics
        
        Parameters
        ----------
        line_stats : pd.DataFrame
            DataFrame with 'line' column from find_lines_batch aggregation.
        queried_types : list of str
            List of queried neuron types (from the original query).
        match_type : str
            Match algorithm for line_to_neuron. Default: 'cds'
        top_n : int
            Number of top matches to consider for type proportion. Default: 100
        max_lines : int
            Maximum number of lines to process (top N by ranking). Default: 100
            
        Returns
        -------
        pd.DataFrame
            Updated line_stats with specificity columns added.
        """
        if line_stats.empty or 'line' not in line_stats.columns:
            return line_stats
        
        if not queried_types:
            self._vprint("  ℹ️  No neuron types to calculate specificity (query was by bodyId)")
            return line_stats
        
        # Normalize queried types for comparison
        queried_types_lower = set(t.lower() for t in queried_types if t)
        
        # Limit to top N lines to reduce API calls
        total_lines = len(line_stats)
        if max_lines and total_lines > max_lines:
            self._vprint(f"\n📊 Calculating specificity for top {max_lines} of {total_lines} lines...")
            lines_to_process = line_stats['line'].head(max_lines).tolist()
        else:
            self._vprint(f"\n📊 Calculating specificity metrics for {total_lines} lines...")
            lines_to_process = line_stats['line'].tolist()
        
        self._vprint(f"   Queried types: {list(queried_types)[:5]}{'...' if len(queried_types) > 5 else ''}")
        self._vprint(f"   ⏱️  Note: Each line requires an API call to fetch neuron matches (may take time)")
        
        # Initialize new columns (for ALL lines, not just processed ones)
        line_stats = line_stats.copy()
        line_stats['rank_sum'] = float('inf')
        line_stats['type_proportion'] = 0.0
        line_stats['n_queried_types'] = 0
        line_stats['n_total_types'] = 0
        line_stats['selectivity'] = 0.0
        # New entropy columns
        line_stats['expression_entropy'] = 0.0
        line_stats['normalized_entropy'] = 0.0
        # New weighted columns
        line_stats['weighted_type_proportion'] = 0.0
        line_stats['mean_queried_score'] = 0.0
        
        # Process each line (with progress bar if available)
        if HAS_TQDM and self.verbose:
            from tqdm import tqdm as tqdm_progress
            iterator = tqdm_progress(
                lines_to_process, 
                desc="   🔬 Analyzing specificity", 
                unit="line",
                bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                ncols=110,
                position=0
            )
        else:
            iterator = lines_to_process
        
        import time
        start_time = time.time()
        
        for idx, line_name in enumerate(iterator):
            # Update progress description with current line
            if HAS_TQDM and self.verbose:
                iterator.set_description(f"   🔬 [{idx+1}/{len(lines_to_process)}] {line_name}")
            try:
                # Check if cached before calling
                cache_key = f"{line_name}_{match_type}"
                is_cached = self._load_from_cache('line_to_neuron', cache_key) is not None
                
                if HAS_TQDM and self.verbose:
                    status = "💾cached" if is_cached else "🌐fetching"
                    iterator.set_postfix_str(status)
                
                # Get neurons labeled by this line (use cached if available)
                neurons_df = self.line_to_neuron(line_name, match_type=match_type)
                
                if neurons_df.empty:
                    continue
                
                # Get top N matches
                neurons_top = neurons_df.head(top_n)
                
                # Get all unique neuron types (across all datasets)
                all_types = neurons_top['type'].dropna().unique().tolist()
                all_types_lower = set(t.lower() for t in all_types if t)
                
                # Count how many queried types are labeled
                matched_types = queried_types_lower & all_types_lower
                n_queried = len(matched_types)
                n_total = len(all_types_lower)
                
                # Calculate rank sum for queried types
                rank_sum = 0
                for idx, row in neurons_top.iterrows():
                    neuron_type = str(row.get('type', '')).lower()
                    if neuron_type in queried_types_lower:
                        # Rank is 1-indexed position
                        rank = neurons_top.index.get_loc(idx) + 1
                        rank_sum += rank
                
                # If no matches, set rank_sum to a high value
                if rank_sum == 0 and n_queried > 0:
                    rank_sum = top_n * n_queried  # Penalty for not appearing in top N
                
                # Calculate type proportion
                type_proportion = n_queried / n_total if n_total > 0 else 0.0
                
                # Calculate selectivity (inverse of diversity)
                selectivity = 1.0 / n_total if n_total > 0 else 0.0
                
                # === NEW: Calculate Expression Entropy ===
                # Count occurrences of each type
                type_counts = neurons_top['type'].fillna('Unknown').value_counts().to_dict()
                expression_entropy = self._calculate_expression_entropy(type_counts)
                
                # Normalize entropy by max possible (log2(n_types))
                max_entropy = np.log2(n_total) if n_total > 1 else 1.0
                normalized_entropy = expression_entropy / max_entropy if max_entropy > 0 else 0.0
                
                # === NEW: Calculate Weighted Specificity ===
                weighted_metrics = self._calculate_weighted_specificity(
                    neurons_top, queried_types_lower, score_column='score'
                )
                
                # Update the row
                mask = line_stats['line'] == line_name
                line_stats.loc[mask, 'rank_sum'] = rank_sum
                line_stats.loc[mask, 'type_proportion'] = type_proportion
                line_stats.loc[mask, 'n_queried_types'] = n_queried
                line_stats.loc[mask, 'n_total_types'] = n_total
                line_stats.loc[mask, 'selectivity'] = selectivity
                line_stats.loc[mask, 'expression_entropy'] = expression_entropy
                line_stats.loc[mask, 'normalized_entropy'] = normalized_entropy
                line_stats.loc[mask, 'weighted_type_proportion'] = weighted_metrics['weighted_type_proportion']
                line_stats.loc[mask, 'mean_queried_score'] = weighted_metrics['mean_queried_score']
                
            except Exception as e:
                if self.verbose:
                    self._vprint(f"   ⚠️ Error calculating specificity for {line_name}: {e}")
        
        # Calculate composite specificity score
        # Combines: type_proportion, entropy (inverted), rank, and weighted proportion
        max_rank = line_stats['rank_sum'].replace(float('inf'), 0).max()
        
        if max_rank > 0:
            normalized_rank = line_stats['rank_sum'].replace(float('inf'), max_rank) / max_rank
        else:
            normalized_rank = 0
        
        # Specificity = high type_proportion + low entropy + low rank + high weighted_proportion
        # Formula: 0.3*type_prop + 0.2*(1-norm_entropy) + 0.2*(1-norm_rank) + 0.3*weighted_prop
        line_stats['specificity_score'] = (
            0.30 * line_stats['type_proportion'] +
            0.20 * (1.0 - line_stats['normalized_entropy']) +
            0.20 * (1.0 - normalized_rank) +
            0.30 * line_stats['weighted_type_proportion']
        )
        
        # Replace inf with max + 1 for display
        max_finite = line_stats.loc[line_stats['rank_sum'] != float('inf'), 'rank_sum'].max()
        line_stats['rank_sum'] = line_stats['rank_sum'].replace(float('inf'), max_finite + 1 if pd.notna(max_finite) else 9999)
        
        # Calculate and report timing statistics
        elapsed_time = time.time() - start_time
        lines_processed = min(len(lines_to_process), max_lines) if max_lines else len(lines_to_process)
        avg_time_per_line = elapsed_time / lines_processed if lines_processed > 0 else 0
        
        self._vprint(f"   ✓ Specificity calculated for {len(line_stats)} lines")
        self._vprint(f"   ⏱️  Processing time: {elapsed_time:.1f}s total, ~{avg_time_per_line:.2f}s per line")
        
        return line_stats

    def _init_client(self):
        """Initialize the NeuronBridge client."""
        self._vprint("🔌 Initializing NeuronBridge client...")
        try:
            self._client = NBClient()
            # Fix S3 URL version mismatch if needed
            self._fix_store_prefixes()
            self._vprint("  ✓ Client initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize NeuronBridge client: {e}")
    
    def _fix_store_prefixes(self):
        """
        Fix S3 URL version mismatch in the NeuronBridge client.
        
        The config may reference v3_8_0 but data might exist at v3_8_1.
        Try production bucket first, as it's more reliable.
        """
        if hasattr(self._client, 'config') and hasattr(self._client.config, 'stores'):
            stores = self._client.config.stores
            # stores is a dictionary of DataStore objects
            if isinstance(stores, dict):
                for store_key, store in stores.items():
                    if hasattr(store, 'prefixes') and store.prefixes:
                        # prefixes is a dict like {'CDSResults': 'url', ...}
                        if isinstance(store.prefixes, dict):
                            for key, url in store.prefixes.items():
                                if isinstance(url, str):
                                    new_url = url
                                    # Keep production bucket but fix version
                                    if 'v3_8_0' in new_url:
                                        new_url = new_url.replace('v3_8_0', 'v3_8_1')
                                    # Switch dev to prod (prod is more reliable)
                                    if 'data-dev' in new_url:
                                        new_url = new_url.replace('data-dev', 'data-prod')
                                    store.prefixes[key] = new_url
    
    def _get_dataset_from_library(self, library_name: str) -> Optional[str]:
        """
        Get local dataset folder name from NeuronBridge library name.
        
        Parameters
        ----------
        library_name : str
            NeuronBridge library name (e.g., 'FlyEM_Hemibrain_v1.2.1')
            
        Returns
        -------
        str or None
            Local dataset folder name, or None if not found.
        """
        if not library_name:
            return None
        
        # Direct match
        if library_name in LIBRARY_TO_DATASET:
            return LIBRARY_TO_DATASET[library_name]
        
        # Partial match (try prefixes)
        for lib_pattern, dataset in LIBRARY_TO_DATASET.items():
            if library_name.startswith(lib_pattern.split('_v')[0]):
                return dataset
        
        return None
    
    def _get_dataset_name_from_library(self, library_name: str) -> Optional[str]:
        """
        Get human-readable dataset name from NeuronBridge library name.
        
        Parameters
        ----------
        library_name : str
            NeuronBridge library name (e.g., 'FlyEM_Male_CNS_Brain_v0.9')
            
        Returns
        -------
        str or None
            Human-readable dataset name (e.g., 'male-cns:v0.9'), or None if not found.
        """
        if not library_name:
            return None
        
        # Direct match
        if library_name in LIBRARY_TO_DATASET_NAME:
            return LIBRARY_TO_DATASET_NAME[library_name]
        
        # Partial match (try prefixes)
        for lib_pattern, dataset_name in LIBRARY_TO_DATASET_NAME.items():
            if library_name.startswith(lib_pattern.split('_v')[0]):
                return dataset_name
        
        return None
    
    def _load_neuron_df_for_dataset(self, dataset: str) -> Optional[pd.DataFrame]:
        """
        Load neuron DataFrame for a specific dataset.
        
        Parameters
        ----------
        dataset : str
            Dataset folder name (e.g., 'hemibrain_v1_2_1')
            
        Returns
        -------
        pd.DataFrame or None
            Loaded DataFrame, or None if not found.
        """
        # Check cache
        if dataset in self._neuron_dfs:
            return self._neuron_dfs[dataset]
        
        # Try to load from datasets folder
        if self.datasets_path:
            neuron_df_path = os.path.join(
                self.datasets_path, dataset, f"{dataset}_allneurons_neuron_df.csv"
            )
            
            if os.path.exists(neuron_df_path):
                try:
                    df = pd.read_csv(neuron_df_path, dtype={'bodyId': str}, low_memory=False)
                    self._neuron_dfs[dataset] = df
                    # Only print loading message if not suppressing
                    if not self._suppress_loading_msgs:
                        self._vprint(f"  ✓ Loaded {len(df):,} neurons from {dataset}")
                    return df
                except Exception as e:
                    self._vprint(f"  ⚠️ Could not load neuron data for {dataset}: {e}")
            else:
                # Try to pull dataset using statvis.pull_dataset
                return self._pull_and_load_dataset(dataset)
        
        # Try fallback to FindNeuronConnection
        return self._fetch_neuron_df_via_fnc(dataset)
    
    def _pull_and_load_dataset(self, dataset: str) -> Optional[pd.DataFrame]:
        """
        Pull dataset from NeuPrint and load neuron DataFrame.
        
        Uses statvis.pull_dataset() to fetch data when local files don't exist.
        
        Parameters
        ----------
        dataset : str
            Dataset folder name (e.g., 'hemibrain_v1_2_1')
            
        Returns
        -------
        pd.DataFrame or None
            Loaded DataFrame, or None if failed.
        """
        try:
            # Try relative import first (when imported as package)
            try:
                from .statvis import pull_dataset
            except ImportError:
                # Fall back to absolute import (when running from scripts)
                from statvis import pull_dataset
            
            # Convert folder name to dataset format (hemibrain_v1_2_1 -> hemibrain:v1.2.1)
            dataset_name = self._folder_to_dataset_name(dataset)
            
            self._vprint(f"  ⏳ Dataset not found locally, attempting to pull {dataset_name}...")
            
            # Initialize NeuPrint client for this dataset
            try:
                from neuprint import Client, set_default_client, default_client
                
                # Check if there's an existing default client for the same dataset
                try:
                    existing = default_client()
                    if existing.dataset != dataset_name:
                        # Different dataset, need new client
                        existing = None
                except (RuntimeError, AttributeError):
                    existing = None
                
                if existing is None:
                    # Get token: 1) instance attribute, 2) env vars
                    token = self.neuprint_token
                    if not token:
                        token = os.environ.get('NEUPRINT_TOKEN', os.environ.get('NEUPRINT_APPLICATION_CREDENTIALS', ''))
                    
                    if not token:
                        self._vprint(f"")
                        self._vprint(f"  ⚠️  No NeuPrint token found - cannot pull dataset.")
                        self._vprint(f"  ╔══════════════════════════════════════════════════════════════╗")
                        self._vprint(f"  ║  To enable dataset pulling, provide your NeuPrint token:    ║")
                        self._vprint(f"  ║                                                              ║")
                        self._vprint(f"  ║  Option 1: Pass directly to NeuronBridgeFinder:             ║")
                        self._vprint(f"  ║    nbf = NeuronBridgeFinder(neuprint_token='YOUR_TOKEN')    ║")
                        self._vprint(f"  ║                                                              ║")
                        self._vprint(f"  ║  Option 2: Set environment variable:                        ║")
                        self._vprint(f"  ║    export NEUPRINT_TOKEN='YOUR_TOKEN'                       ║")
                        self._vprint(f"  ║                                                              ║")
                        self._vprint(f"  ║  Get your token at:                                         ║")
                        self._vprint(f"  ║  👉 https://neuprint.janelia.org/account                    ║")
                        self._vprint(f"  ╚══════════════════════════════════════════════════════════════╝")
                        self._vprint(f"")
                        self._vprint(f"  ℹ️  Skipping local dataset features (type lookups, specificity metrics).")
                        return None
                    
                    # Parse server URL (remove https:// if present for neuprint client)
                    server = self.neuprint_server
                    if server.startswith('https://'):
                        server = server[8:]
                    if server.startswith('http://'):
                        server = server[7:]
                    
                    client = Client(server, dataset_name, token)
                    set_default_client(client)
                    self._vprint(f"  ✓ Connected to NeuPrint ({dataset_name})")
                    
            except ImportError:
                self._vprint(f"  ⚠️ neuprint package not available. Install with: pip install neuprint-python")
                return None
            except Exception as e:
                self._vprint(f"  ⚠️ Could not connect to NeuPrint: {e}")
                return None
            
            # Create dataset directory
            dataset_dir = os.path.join(self.datasets_path, dataset)
            os.makedirs(dataset_dir, exist_ok=True)
            
            save_path = os.path.join(dataset_dir, f"{dataset}_allneurons")
            
            try:
                pull_dataset(dataset_name, save_path=save_path, omitNoneType=False)
                
                # Load the pulled data
                neuron_df_path = save_path + '_neuron_df.csv'
                if os.path.exists(neuron_df_path):
                    df = pd.read_csv(neuron_df_path, dtype={'bodyId': str}, low_memory=False)
                    self._neuron_dfs[dataset] = df
                    self._vprint(f"  ✓ Pulled and loaded {len(df):,} neurons from {dataset}")
                    return df
            except Exception as e:
                self._vprint(f"  ⚠️ Could not pull dataset {dataset_name}: {e}")
                
            return None
            
        except ImportError:
            self._vprint(f"  ⚠️ statvis module not available for pulling dataset")
            return None
        except Exception as e:
            self._vprint(f"  ⚠️ Error pulling dataset: {e}")
            return None
    
    def _fetch_neuron_df_via_fnc(self, dataset: str) -> Optional[pd.DataFrame]:
        """
        Try to fetch neuron DataFrame via FindNeuronConnection.
        
        This is a fallback when local dataset files don't exist.
        
        Parameters
        ----------
        dataset : str
            Dataset name
            
        Returns
        -------
        pd.DataFrame or None
            Fetched DataFrame, or None if failed.
        """
        try:
            from scripts.FindNeuronConnection import FindNeuronConnection
            
            self._vprint(f"  ⏳ Fetching neuron data for {dataset} via FindNeuronConnection...")
            
            fnc = FindNeuronConnection(dataset=dataset)
            
            # Check if we can get neuron index
            cache_dir = os.path.join(self.datasets_path, '..', 'cache', dataset)
            csv_path = os.path.join(cache_dir, f'{dataset}_allneurons_neuron_df.csv')
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, dtype={'bodyId': str}, low_memory=False)
                self._neuron_dfs[dataset] = df
                self._vprint(f"  ✓ Loaded {len(df):,} neurons via FindNeuronConnection cache")
                return df
            
            return None
            
        except ImportError:
            self._vprint(f"  ⚠️ FindNeuronConnection not available")
            return None
        except Exception as e:
            self._vprint(f"  ⚠️ Could not fetch via FindNeuronConnection: {e}")
            return None
    
    def _enrich_match_with_dataset_info(self, match: Dict[str, Any], em_image) -> Dict[str, Any]:
        """
        Enrich a match dictionary with dataset-specific information.
        
        Parameters
        ----------
        match : dict
            Match dictionary to enrich
        em_image : EMImage
            NeuronBridge EM image object
            
        Returns
        -------
        dict
            Enriched match dictionary
        """
        # Get library name and dataset
        library_name = getattr(em_image, 'libraryName', '')
        dataset_folder = self._get_dataset_from_library(library_name)
        dataset_name = self._get_dataset_name_from_library(library_name)
        
        # Add dataset info to match (use human-readable name, folder for internal lookup)
        match['dataset'] = dataset_name or 'unknown'
        match['dataset_folder'] = dataset_folder or 'unknown'
        match['library'] = library_name
        
        # Get neuron type and instance from EM image if available
        match['neuronType'] = getattr(em_image, 'neuronType', '') or ''
        match['neuronInstance'] = getattr(em_image, 'neuronInstance', '') or ''
        
        # Try to enrich from local dataset
        if dataset_folder:
            neuron_df = self._load_neuron_df_for_dataset(dataset_folder)
            if neuron_df is not None and not neuron_df.empty:
                body_id = str(match.get('bodyId', ''))
                neuron_row = neuron_df[neuron_df['bodyId'] == body_id]
                if not neuron_row.empty:
                    row = neuron_row.iloc[0]
                    # Use local data, fallback to NB data
                    match['type'] = row.get('type', '') or match['neuronType']
                    match['instance'] = row.get('instance', '') or match['neuronInstance']
                    match['status'] = row.get('status', '')
                else:
                    # Use NeuronBridge data
                    match['type'] = match['neuronType']
                    match['instance'] = match['neuronInstance']
                    match['status'] = ''
            else:
                match['type'] = match['neuronType']
                match['instance'] = match['neuronInstance']
                match['status'] = ''
        else:
            match['type'] = match['neuronType']
            match['instance'] = match['neuronInstance']
            match['status'] = ''
        
        return match
    
    def _get_cache_path(self, cache_type: str, identifier: str) -> str:
        """Get the cache file path for a given type and identifier."""
        safe_id = str(identifier).replace('/', '_').replace(':', '_')
        return os.path.join(self.cache_folder, f"{cache_type}_{safe_id}.csv")
    
    def _load_from_cache(self, cache_type: str, identifier: str) -> Optional[pd.DataFrame]:
        """Load cached results if available."""
        if not self.use_cache:
            return None
        
        cache_path = self._get_cache_path(cache_type, identifier)
        if os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path)
                self._vprint(f"  ⏩ Loaded from cache: {cache_path}")
                return df
            except Exception:
                return None
        return None
    
    def _save_to_cache(self, cache_type: str, identifier: str, df: pd.DataFrame):
        """Save results to cache."""
        if not self.use_cache or df.empty:
            return
        
        cache_path = self._get_cache_path(cache_type, identifier)
        try:
            df.to_csv(cache_path, index=False)
            self._vprint(f"  💾 Saved to cache: {cache_path}")
        except Exception as e:
            warnings.warn(f"Failed to save cache: {e}")
    
    def _get_em_image_for_dataset(
        self,
        body_id: int,
        expected_dataset: Optional[str] = None
    ):
        """
        Get EM image for a body ID, optionally filtering by expected dataset.
        
        NeuronBridge can return multiple EM images for the same body ID from
        different datasets (e.g., vnc:v0.5, manc:v1.2.1, male-cns:v0.9).
        This method finds the one matching the expected dataset.
        
        Parameters
        ----------
        body_id : int
            The body ID to look up.
        expected_dataset : str, optional
            Expected dataset name (e.g., 'male-cns:v0.9').
            If None, returns the first result.
            
        Returns
        -------
        EMImage or None
            The matching EM image, or None if not found.
        """
        if not self._client:
            return None
        
        try:
            # Use get_em_images (plural) to get ALL results for this body ID
            em_images = self._client.get_em_images(body_id)
            
            if not em_images:
                return None
            
            # Convert to list if needed
            if not isinstance(em_images, list):
                em_images = list(em_images)
            
            if not em_images:
                return None
            
            # If no expected dataset, return first result
            if not expected_dataset:
                return em_images[0]
            
            # Normalize expected dataset for comparison
            expected_base = expected_dataset.lower().split(':')[0].replace('_', '-')
            
            # Find the EM image matching the expected dataset
            for em_image in em_images:
                pub_name = getattr(em_image, 'publishedName', '')
                if ':' in pub_name:
                    actual_base = pub_name.lower().split(':')[0].replace('_', '-')
                    if actual_base == expected_base:
                        return em_image
            
            # If no exact match, return first result
            return em_images[0]
            
        except Exception:
            # Fallback to singular get_em_image
            try:
                return self._client.get_em_image(body_id)
            except Exception:
                return None
    
    def _validate_body_id_dataset(
        self, 
        body_id: int, 
        expected_dataset: str
    ) -> Optional[str]:
        """
        Validate that a body ID exists in the expected dataset in NeuronBridge.
        
        Parameters
        ----------
        body_id : int
            The body ID to validate.
        expected_dataset : str
            Expected dataset name (e.g., 'male-cns:v0.9').
            
        Returns
        -------
        str or None
            The actual dataset name if found matching expected, or None if not found.
        """
        if not self._client:
            return expected_dataset  # Can't validate without client
        
        try:
            # Use get_em_images (plural) to get ALL results for this body ID
            em_images = self._client.get_em_images(body_id)
            
            if not em_images:
                return None
            
            # Convert to list if needed
            if not isinstance(em_images, list):
                em_images = list(em_images)
            
            if not em_images:
                return None
            
            # Normalize expected dataset for comparison
            expected_base = expected_dataset.lower().split(':')[0].replace('_', '-')
            
            # Check if any EM image matches the expected dataset
            for em_image in em_images:
                pub_name = getattr(em_image, 'publishedName', '')
                if ':' in pub_name:
                    parts = pub_name.split(':')
                    actual_base = parts[0].lower().replace('_', '-')
                    if actual_base == expected_base:
                        # Found matching dataset
                        return f"{parts[0]}:{parts[1]}" if len(parts) >= 2 else parts[0]
            
            # No matching dataset found - return what was found
            pub_name = getattr(em_images[0], 'publishedName', '')
            if ':' in pub_name:
                parts = pub_name.split(':')
                return f"{parts[0]}:{parts[1]}" if len(parts) >= 2 else parts[0]
            return None
            
        except Exception:
            return expected_dataset  # On error, assume expected dataset
    
    def _sort_matches_by_rank(
        self, 
        matches: List[Dict[str, Any]], 
        key_field: str = 'line'
    ) -> List[Dict[str, Any]]:
        """
        Sort matches by combined rank when both CDS and PPPM results exist.
        
        For each unique line/bodyId, compute:
        - CDS rank (1-based, by descending score)
        - PPPM rank (1-based, by descending score)
        - Combined rank = CDS_rank + PPPM_rank
        
        Sort by combined rank ascending (lower is better).
        
        Parameters
        ----------
        matches : list
            List of match dictionaries with 'match_type', 'score', and key_field.
        key_field : str
            Field to use as unique identifier ('line' for EM->LM, 'bodyId' for LM->EM).
            
        Returns
        -------
        list
            Sorted list of match dictionaries with added 'cds_rank', 'pppm_rank', 
            'combined_rank' columns.
        """
        if not matches:
            return matches
        
        # Separate CDS and PPPM matches
        cds_matches = [m for m in matches if m.get('match_type') == 'cds']
        pppm_matches = [m for m in matches if m.get('match_type') == 'pppm']
        
        # Sort each by score descending and assign ranks
        cds_matches.sort(key=lambda x: x.get('score', 0), reverse=True)
        pppm_matches.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Build rank dictionaries (key -> rank)
        cds_ranks = {}
        for rank, m in enumerate(cds_matches, 1):
            key = m.get(key_field)
            if key and key not in cds_ranks:
                cds_ranks[key] = rank
        
        pppm_ranks = {}
        for rank, m in enumerate(pppm_matches, 1):
            key = m.get(key_field)
            if key and key not in pppm_ranks:
                pppm_ranks[key] = rank
        
        # Default rank for missing matches (use max+1 or a high number)
        max_cds_rank = len(cds_matches) + 1 if cds_matches else 1
        max_pppm_rank = len(pppm_matches) + 1 if pppm_matches else 1
        
        # Combine matches by key, keeping best score per match_type
        combined = {}
        for m in matches:
            key = m.get(key_field)
            if not key:
                continue
            
            if key not in combined:
                combined[key] = {
                    key_field: key,
                    'library': m.get('library', ''),
                    'image_id': m.get('image_id', ''),
                    'cds_score': 0,
                    'pppm_score': 0,
                    'cds_rank': max_cds_rank,
                    'pppm_rank': max_pppm_rank,
                }
                # Copy other fields for LM->EM matches
                for field in ['bodyId', 'dataset', 'instance', 'type', 'status', 'lm_sample']:
                    if field in m:
                        combined[key][field] = m[field]
            
            # Update scores and ranks
            if m.get('match_type') == 'cds':
                if m.get('score', 0) > combined[key]['cds_score']:
                    combined[key]['cds_score'] = m.get('score', 0)
                combined[key]['cds_rank'] = cds_ranks.get(key, max_cds_rank)
            elif m.get('match_type') == 'pppm':
                if m.get('score', 0) > combined[key]['pppm_score']:
                    combined[key]['pppm_score'] = m.get('score', 0)
                combined[key]['pppm_rank'] = pppm_ranks.get(key, max_pppm_rank)
        
        # Calculate combined rank and sort
        result = []
        for key, data in combined.items():
            data['combined_rank'] = data['cds_rank'] + data['pppm_rank']
            # Use higher score as primary score for display
            data['score'] = max(data['cds_score'], data['pppm_score'])
            data['match_type'] = 'both'
            result.append(data)
        
        # Sort by combined rank ascending
        result.sort(key=lambda x: x['combined_rank'])
        
        return result
    
    def _get_em_matches(
        self, 
        body_id: int, 
        match_type: Optional[str] = None,
        expected_dataset: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get LM matches for an EM body ID.
        
        Parameters
        ----------
        body_id : int
            The EM body ID to search for.
        match_type : str, optional
            Match algorithm: 'cds', 'pppm', or 'both'. 
            If None, uses self.match_type. Default: None
        expected_dataset : str, optional
            Expected dataset name (e.g., 'male-cns:v0.9').
            If provided, will filter EM images to match this dataset.
            
        Returns
        -------
        list
            List of match dictionaries, sorted by:
            - 'cds' or 'pppm': score descending
            - 'both': combined rank ascending (sum of CDS rank + PPPM rank)
        """
        # Use class-level match_type if not specified
        if match_type is None:
            match_type = self.match_type
            
        try:
            # Use helper to get EM image for the expected dataset
            em_image = self._get_em_image_for_dataset(body_id, expected_dataset)
            if em_image is None:
                self._vprint(f"  ⚠️ No EM image found for body ID {body_id}")
                return []
            
            # Check if em_image has required URL for fetching matches
            # Some body IDs exist but don't have valid match URLs in NeuronBridge
            em_files = getattr(em_image, 'files', None)
            if em_files is None:
                self._vprint(f"  ⚠️ No files metadata for body ID {body_id}")
                return []
            
            all_matches = []
            
            cds_failed = False
            pppm_failed = False
            
            # Get CDS matches with retry logic
            if match_type in ['cds', 'both']:
                try:
                    # Check if CDSResults URL exists
                    cds_url = getattr(em_files, 'CDSResults', None)
                    if cds_url is None:
                        cds_failed = True
                    else:
                        cds_matches = self._retry_with_backoff(
                            self._client.get_cds_matches,
                            em_image,
                            max_retries=3,
                            initial_delay=1.0
                        )
                        for match in cds_matches:
                            if hasattr(match, 'image') and hasattr(match.image, 'type'):
                                if match.image.type == 'LMImage':
                                    all_matches.append({
                                        'line': getattr(match.image, 'publishedName', ''),
                                        'library': getattr(match.image, 'libraryName', ''),
                                        'score': getattr(match, 'normalizedScore', 0),
                                        'image_id': getattr(match.image, 'id', ''),
                                        'match_type': 'cds'
                                    })
                except Exception as e:
                    cds_failed = True
                    # Suppress individual errors - will be caught by calling method
            
            # Get PPPM matches with retry logic
            if match_type in ['pppm', 'both']:
                try:
                    # Check if PPPMResults URL exists
                    pppm_url = getattr(em_files, 'PPPMResults', None)
                    if pppm_url is None:
                        pppm_failed = True
                    else:
                        pppm_matches = self._retry_with_backoff(
                            self._client.get_ppp_matches,
                            em_image,
                            max_retries=3,
                            initial_delay=1.0
                        )
                        for match in pppm_matches:
                            if hasattr(match, 'image') and hasattr(match.image, 'type'):
                                if match.image.type == 'LMImage':
                                    all_matches.append({
                                        'line': getattr(match.image, 'publishedName', ''),
                                        'library': getattr(match.image, 'libraryName', ''),
                                        'score': getattr(match, 'normalizedScore', 0),
                                        'image_id': getattr(match.image, 'id', ''),
                                        'match_type': 'pppm'
                                    })
                except Exception as e:
                    pppm_failed = True
                    # Suppress individual errors - will be caught by calling method
            
            # Only warn if both failed when match_type='both'
            if match_type == 'both' and cds_failed and pppm_failed:
                self._vprint(f"  ⚠️ Both CDS and PPPM matches failed for body ID {body_id}")
            
            # Sort results based on match_type
            if match_type == 'both' and all_matches:
                # Use rank-based sorting for combined results
                all_matches = self._sort_matches_by_rank(all_matches, key_field='line')
            else:
                # Sort by score descending for single match type
                all_matches.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            return all_matches
            
        except Exception as e:
            self._vprint(f"  ⚠️ Error getting matches for body ID {body_id}: {e}")
            return []
    
    def _get_lm_matches(
        self, 
        line_name: str, 
        match_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get EM matches for an LM line name.
        
        Parameters
        ----------
        line_name : str
            The driver line name to search for.
        match_type : str, optional
            Match algorithm: 'cds', 'pppm', or 'both'. 
            If None, uses self.match_type. Default: None
            
        Returns
        -------
        list
            List of match dictionaries with dataset info, sorted by:
            - 'cds' or 'pppm': score descending
            - 'both': combined rank ascending (sum of CDS rank + PPPM rank)
        """
        # Use class-level match_type if not specified
        if match_type is None:
            match_type = self.match_type
            
        try:
            lm_images = self._client.get_lm_images(line_name)
            if not lm_images:
                self._vprint(f"  ⚠️ No LM images found for line '{line_name}'")
                return []
            
            # Convert generator to list if needed
            if not isinstance(lm_images, list):
                lm_images = list(lm_images)
            
            # Filter images by region
            lm_images = self._filter_images_by_region(lm_images)
            if not lm_images:
                self._vprint(f"  ⚠️ No LM images for line '{line_name}' in region '{self.region}'")
                return []
            
            n_images = len(lm_images)
            # Only print verbose message if not in a progress bar context
            if not HAS_TQDM or not self.verbose:
                self._vprint(f"  Found {n_images} LM images for '{line_name}'")
            
            all_matches = []
            cds_errors = 0
            pppm_errors = 0
            
            # Create progress bar for image processing if we have multiple images and tqdm is available
            # Only show if we're not already in a progress bar context (avoid double nesting)
            show_image_progress = HAS_TQDM and self.verbose and n_images > 10
            
            if show_image_progress:
                from tqdm import tqdm as tqdm_progress
                # Suppress loading messages while progress bar is active
                self._suppress_loading_msgs = True
                image_iterator = tqdm_progress(
                    lm_images,
                    desc=f"     Processing {n_images} images",
                    unit="img",
                    leave=False,
                    bar_format='     {desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                    ncols=100,
                    position=1
                )
            else:
                image_iterator = lm_images
            
            for img_idx, lm_image in enumerate(image_iterator, 1):
                cds_failed = False
                pppm_failed = False
                
                # Get CDS matches with retry logic
                if match_type in ['cds', 'both']:
                    try:
                        cds_matches = self._retry_with_backoff(
                            self._client.get_cds_matches,
                            lm_image,
                            max_retries=3,
                            initial_delay=1.0
                        )
                        for match in cds_matches:
                            if hasattr(match, 'image') and hasattr(match.image, 'type'):
                                if match.image.type == 'EMImage':
                                    # Extract body ID from publishedName
                                    body_id = self._extract_body_id(match.image)
                                    match_dict = {
                                        'bodyId': body_id,
                                        'score': getattr(match, 'normalizedScore', 0),
                                        'image_id': getattr(match.image, 'id', ''),
                                        'lm_sample': getattr(lm_image, 'id', ''),
                                        'match_type': 'cds'
                                    }
                                    # Enrich with dataset info
                                    match_dict = self._enrich_match_with_dataset_info(match_dict, match.image)
                                    all_matches.append(match_dict)
                    except Exception as e:
                        cds_failed = True
                        cds_errors += 1
                        # Only warn if this is a 'cds'-only request and verbose
                        # Suppress individual errors to avoid spam
                
                # Get PPPM matches with retry logic
                if match_type in ['pppm', 'both']:
                    try:
                        pppm_matches = self._retry_with_backoff(
                            self._client.get_ppp_matches,
                            lm_image,
                            max_retries=3,
                            initial_delay=1.0
                        )
                        for match in pppm_matches:
                            if hasattr(match, 'image') and hasattr(match.image, 'type'):
                                if match.image.type == 'EMImage':
                                    body_id = self._extract_body_id(match.image)
                                    match_dict = {
                                        'bodyId': body_id,
                                        'score': getattr(match, 'normalizedScore', 0),
                                        'image_id': getattr(match.image, 'id', ''),
                                        'lm_sample': getattr(lm_image, 'id', ''),
                                        'match_type': 'pppm'
                                    }
                                    # Enrich with dataset info
                                    match_dict = self._enrich_match_with_dataset_info(match_dict, match.image)
                                    all_matches.append(match_dict)
                    except Exception as e:
                        pppm_failed = True
                        pppm_errors += 1
                        # Only warn if this is a 'pppm'-only request and verbose
                        # Suppress individual errors to avoid spam
                
                # Only warn if both failed when match_type='both'
                if match_type == 'both' and cds_failed and pppm_failed:
                    lm_id = getattr(lm_image, 'id', 'unknown')
                    self._vprint(f"  ⚠️ Both CDS and PPPM matches failed for LM image {lm_id}")
            
            # Restore loading messages flag after processing images
            if show_image_progress:
                self._suppress_loading_msgs = False
            
            # Report error summary if there were failures
            if cds_errors > 0 or pppm_errors > 0:
                error_parts = []
                if cds_errors > 0:
                    error_parts.append(f"CDS: {cds_errors}/{n_images}")
                if pppm_errors > 0:
                    error_parts.append(f"PPPM: {pppm_errors}/{n_images}")
                self._vprint(f"  ℹ️  Network errors (retried 3x): {', '.join(error_parts)} images failed")
            
            # Sort results based on match_type
            if match_type == 'both' and all_matches:
                # Use rank-based sorting for combined results
                unique_matches = self._sort_matches_by_rank(all_matches, key_field='bodyId')
            else:
                # Sort by score descending for single match type
                all_matches.sort(key=lambda x: x.get('score', 0), reverse=True)
                
                # Remove duplicates (same bodyId + dataset), keeping highest score
                seen = set()
                unique_matches = []
                for match in all_matches:
                    key = (match.get('bodyId'), match.get('dataset'))
                    if key not in seen:
                        seen.add(key)
                        unique_matches.append(match)
            
            return unique_matches
            
        except Exception as e:
            self._vprint(f"  ⚠️ Error getting matches for line '{line_name}': {e}")
            return []
    
    def _extract_body_id(self, em_image) -> str:
        """Extract body ID from an EM image object."""
        # Try publishedName first (format: 'hemibrain:v1.2.1:BODY_ID')
        published_name = getattr(em_image, 'publishedName', '')
        if published_name and ':' in published_name:
            parts = published_name.split(':')
            if len(parts) >= 3:
                return parts[-1]
        
        # Try id field
        image_id = getattr(em_image, 'id', '')
        if image_id:
            # Extract digits
            digits = ''.join(c for c in str(image_id) if c.isdigit())
            if digits:
                return digits
        
        return ''
    
    def _find_bodyIds_by_query(
        self, 
        query: Union[str, int, List],
        dataset: Optional[Union[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find body IDs matching a query in the neuron DataFrame(s).
        
        Parameters
        ----------
        query : str, int, or list
            - If int: treated as bodyId
            - If str: searched in instance and type columns (supports regex)
            - If list: each item processed individually
        dataset : str, list of str, or None
            Specific dataset(s) to search (e.g., 'hemibrain:v1.2.1', 'male-cns:v0.9').
            Can be a single string or a list of datasets.
            If None, searches all available datasets.
            
        Returns
        -------
        list
            List of dicts with 'bodyId' and 'dataset' keys.
        """
        # Determine which datasets to search
        datasets_to_search = []
        
        if dataset:
            # Handle list of datasets
            dataset_list = [dataset] if isinstance(dataset, str) else list(dataset)
            
            for ds in dataset_list:
                # Map display name to folder name if needed
                dataset_folder = self._dataset_name_to_folder(ds)
                if dataset_folder:
                    datasets_to_search.append(dataset_folder)
                else:
                    datasets_to_search.append(ds.replace(':', '_').replace('.', '_'))
            # Remove duplicates while preserving order
            datasets_to_search = list(dict.fromkeys(datasets_to_search))
        else:
            # Search all available datasets
            # Get unique dataset folder names
            datasets_to_search = list(set(LIBRARY_TO_DATASET.values()))
        
        if isinstance(query, list):
            all_results = []
            for q in query:
                all_results.extend(self._find_bodyIds_by_query(q, dataset))
            # Remove duplicates (same bodyId + dataset)
            seen = set()
            unique_results = []
            for r in all_results:
                key = (r['bodyId'], r['dataset'])
                if key not in seen:
                    seen.add(key)
                    unique_results.append(r)
            return unique_results
        
        if isinstance(query, int):
            # For direct body ID, we don't know the dataset - return for all with matching ID
            results = []
            for ds_folder in datasets_to_search:
                neuron_df = self._load_neuron_df_for_dataset(ds_folder)
                if neuron_df is not None and not neuron_df.empty:
                    if str(query) in neuron_df['bodyId'].astype(str).values:
                        ds_name = self._folder_to_dataset_name(ds_folder)
                        results.append({'bodyId': str(query), 'dataset': ds_name, 'dataset_folder': ds_folder})
            # If not found in any, still return it
            if not results:
                results = [{'bodyId': str(query), 'dataset': 'unknown', 'dataset_folder': ''}]
            return results
        
        query_str = str(query)
        matched_results = []
        
        # Try exact bodyId match first
        if query_str.isdigit():
            for ds_folder in datasets_to_search:
                neuron_df = self._load_neuron_df_for_dataset(ds_folder)
                if neuron_df is not None and not neuron_df.empty:
                    if query_str in neuron_df['bodyId'].astype(str).values:
                        ds_name = self._folder_to_dataset_name(ds_folder)
                        matched_results.append({
                            'bodyId': query_str, 
                            'dataset': ds_name,
                            'dataset_folder': ds_folder
                        })
            if matched_results:
                return matched_results
        
        # Search in type/instance columns across all datasets
        for ds_folder in datasets_to_search:
            neuron_df = self._load_neuron_df_for_dataset(ds_folder)
            if neuron_df is None or neuron_df.empty:
                continue
            
            ds_name = self._folder_to_dataset_name(ds_folder)
            
            # Search in 'type' column
            if 'type' in neuron_df.columns:
                try:
                    type_matches = neuron_df[
                        neuron_df['type'].astype(str).str.match(f'^{query_str}$', case=False, na=False)
                    ]
                    for body_id in type_matches['bodyId'].astype(str).tolist():
                        matched_results.append({
                            'bodyId': body_id, 
                            'dataset': ds_name,
                            'dataset_folder': ds_folder
                        })
                except re.error:
                    type_matches = neuron_df[
                        neuron_df['type'].astype(str).str.lower() == query_str.lower()
                    ]
                    for body_id in type_matches['bodyId'].astype(str).tolist():
                        matched_results.append({
                            'bodyId': body_id, 
                            'dataset': ds_name,
                            'dataset_folder': ds_folder
                        })
            
            # Search in 'instance' column
            if 'instance' in neuron_df.columns:
                try:
                    instance_matches = neuron_df[
                        neuron_df['instance'].astype(str).str.match(f'^{query_str}$', case=False, na=False)
                    ]
                    for body_id in instance_matches['bodyId'].astype(str).tolist():
                        matched_results.append({
                            'bodyId': body_id, 
                            'dataset': ds_name,
                            'dataset_folder': ds_folder
                        })
                except re.error:
                    instance_matches = neuron_df[
                        neuron_df['instance'].astype(str).str.lower() == query_str.lower()
                    ]
                    for body_id in instance_matches['bodyId'].astype(str).tolist():
                        matched_results.append({
                            'bodyId': body_id, 
                            'dataset': ds_name,
                            'dataset_folder': ds_folder
                        })
        
        # Remove duplicates (same bodyId + dataset)
        seen = set()
        unique_results = []
        for r in matched_results:
            key = (r['bodyId'], r['dataset'])
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        
        return unique_results
    
    def _dataset_name_to_folder(self, dataset_name: str) -> Optional[str]:
        """Convert display dataset name to folder name."""
        # Try direct lookup via reverse mapping
        for lib, name in LIBRARY_TO_DATASET_NAME.items():
            if name.lower() == dataset_name.lower():
                return LIBRARY_TO_DATASET.get(lib)
        # Try folder names directly
        for lib, folder in LIBRARY_TO_DATASET.items():
            if folder.lower() == dataset_name.lower().replace(':', '_').replace('.', '_'):
                return folder
        return None
    
    def _folder_to_dataset_name(self, folder: str) -> str:
        """Convert folder name to display dataset name."""
        # Reverse lookup in LIBRARY_TO_DATASET
        for lib, lib_folder in LIBRARY_TO_DATASET.items():
            if lib_folder == folder:
                return LIBRARY_TO_DATASET_NAME.get(lib, folder)
        return folder
    
    # =========================================================================
    # Public Methods
    # =========================================================================
    
    def id_to_lines(
        self, 
        body_id: int, 
        match_type: str = 'cds',
        expected_dataset: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Find driver lines matching a given EM body ID.
        
        Parameters
        ----------
        body_id : int
            The EM body ID to search for.
        match_type : str
            Match algorithm: 'cds', 'pppm', or 'both'. Default: 'cds'
        expected_dataset : str, optional
            Expected dataset name (e.g., 'male-cns:v0.9').
            If provided, will filter EM images to match this dataset.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: line, library, score, image_id, match_type
            Sorted by score descending (cds/pppm) or combined_rank ascending (both)
            
        Example
        -------
        >>> nbf = NeuronBridgeFinder()
        >>> lines = nbf.id_to_lines(636798093)
        >>> print(lines)
        """
        self._vprint(f"🔍 Searching for lines matching body ID: {body_id}")
        
        # Check cache (include dataset in key to differentiate results)
        ds_key = expected_dataset.replace(':', '_') if expected_dataset else 'any'
        cache_key = f"{body_id}_{match_type}_{ds_key}"
        cached = self._load_from_cache('id_to_lines', cache_key)
        if cached is not None:
            return cached
        
        # Fetch from API with expected dataset
        matches = self._get_em_matches(
            body_id, 
            match_type=match_type, 
            expected_dataset=expected_dataset
        )
        
        if not matches:
            self._vprint(f"  ℹ️ No matches found for body ID {body_id}")
            cols = ['line', 'library', 'score', 'image_id', 'match_type']
            if match_type == 'both':
                cols.extend(['cds_score', 'pppm_score', 'cds_rank', 'pppm_rank', 'combined_rank'])
            return pd.DataFrame(columns=cols)
        
        df = pd.DataFrame(matches)
        
        # Cache results
        self._save_to_cache('id_to_lines', cache_key, df)
        
        self._vprint(f"  ✓ Found {len(df)} matches")
        
        return df
    
    def neuron_to_lines(
        self, 
        query: Union[str, int, List],
        dataset: Optional[Union[str, List[str]]] = None,
        match_type: str = 'cds'
    ) -> Dict[str, pd.DataFrame]:
        """
        Find driver lines matching neurons specified by bodyId, instance, or type.
        
        Parameters
        ----------
        query : str, int, or list
            - If int: treated as bodyId
            - If str: searched in instance and type columns (supports regex)
            - If list: each item processed individually
        dataset : str, list of str, or None
            Specific dataset(s) to search (e.g., 'hemibrain:v1.2.1', 'male-cns:v0.9').
            Can be a single string or a list of datasets to search multiple.
            If None, searches all available datasets.
        match_type : str
            Match algorithm: 'cds', 'pppm', or 'both'. Default: 'cds'
            
        Returns
        -------
        dict
            Dictionary mapping bodyId -> DataFrame of matching lines
            All results sorted by score descending (cds/pppm) or combined_rank ascending (both)
            
        Example
        -------
        >>> nbf = NeuronBridgeFinder()
        >>> results = nbf.neuron_to_lines('MBON01')
        >>> for body_id, lines_df in results.items():
        ...     print(f"Body ID {body_id}: {len(lines_df)} matches")
        """
        self._vprint(f"🔍 Searching for lines matching query: {query}")
        
        # Find matching bodyIds (now returns list of dicts with bodyId and dataset)
        body_info_list = self._find_bodyIds_by_query(query, dataset)
        
        if not body_info_list:
            self._vprint(f"  ⚠️ No neurons found matching query: {query}")
            return {}
        
        self._vprint(f"  Found {len(body_info_list)} neurons to search")
        
        results = {}
        skipped_count = 0
        for i, body_info in enumerate(body_info_list):
            body_id = body_info['bodyId']
            expected_ds = body_info.get('dataset', 'unknown')
            
            # Validate body ID against NeuronBridge to ensure dataset match
            actual_ds = self._validate_body_id_dataset(int(body_id), expected_ds)
            
            # Check for dataset mismatch
            if actual_ds and expected_ds != 'unknown':
                # Normalize for comparison - extract base dataset name
                # Handle different naming conventions:
                # - 'flywire_FAFB_v783' (underscore + version suffix)
                # - 'flywire_fafb:v783' (colon-separated version)
                # - 'male-cns:v0.9' -> 'male-cns'
                # - 'hemibrain:v1.2.1' -> 'hemibrain'
                expected_base = self._normalize_dataset_name(expected_ds)
                actual_base = self._normalize_dataset_name(actual_ds)
                
                if expected_base != actual_base:
                    self._vprint(f"  ⏭️  Skipping {body_id}: belongs to {actual_ds}, not {expected_ds} (dataset mismatch)")
                    skipped_count += 1
                    continue
            
            self._vprint(f"  Processing {i+1}/{len(body_info_list)}: {body_id} ({actual_ds or expected_ds})")
            try:
                # Pass expected dataset to get correct EM image
                lines_df = self.id_to_lines(
                    int(body_id), 
                    match_type=match_type,
                    expected_dataset=expected_ds
                )
                # Add source dataset info (use actual dataset from NeuronBridge)
                if not lines_df.empty:
                    lines_df = lines_df.copy()
                    lines_df['source_dataset'] = actual_ds or expected_ds
                    lines_df['source_bodyId'] = body_id
                results[body_id] = lines_df
            except Exception as e:
                self._vprint(f"    ⚠️ Error processing body ID {body_id}: {e}")
                results[body_id] = pd.DataFrame()
        
        if skipped_count > 0:
            self._vprint(f"  ℹ️  Skipped {skipped_count} body IDs due to dataset mismatch")
        
        return results
    
    def line_to_neuron(
        self, 
        line_name: str,
        match_type: str = 'cds',
        top_n: int = -1
    ) -> pd.DataFrame:
        """
        Find EM neurons matching a driver line name.
        
        Parameters
        ----------
        line_name : str
            The driver line name to search for (e.g., 'LH173').
        match_type : str
            Match algorithm: 'cds', 'pppm', or 'both'. Default: 'cds'
        top_n : int
            Maximum number of matches to return. Default: -1 (all matches)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: bodyId, dataset, instance, type, status, 
            score, image_id, lm_sample, match_type, library
            Sorted by score descending (cds/pppm) or combined_rank ascending (both)
            
        Example
        -------
        >>> nbf = NeuronBridgeFinder()
        >>> neurons = nbf.line_to_neuron('LH173')
        >>> print(neurons)
        """
        # Only print search message if not in batch/progress mode
        # (check if we're being called from within a progress bar loop)
        import inspect
        frame = inspect.currentframe()
        in_progress_context = False
        if frame and frame.f_back:
            # Check if caller's local variables include tqdm iterator
            caller_locals = frame.f_back.f_locals
            in_progress_context = any('tqdm' in str(type(v)).lower() for v in caller_locals.values())
        
        if not in_progress_context:
            self._vprint(f"🔍 Searching for neurons matching line: {line_name}")
        
        # Check cache
        cache_key = f"{line_name}_{match_type}"
        cached = self._load_from_cache('line_to_neuron', cache_key)
        if cached is not None:
            # Indicate cache hit in verbose mode
            if self.verbose and not any(c in str(self._vprint.__code__) for c in ['silent', 'quiet']):
                # Only print if not suppressed by progress bar
                pass  # Progress bar will show cache status
            return cached
        
        # Fetch from API (matches are already enriched with dataset info)
        matches = self._get_lm_matches(line_name, match_type=match_type)
        
        if not matches:
            self._vprint(f"  ℹ️ No matches found for line '{line_name}'")
            cols = ['bodyId', 'dataset', 'instance', 'type', 'status', 'score', 
                    'image_id', 'lm_sample', 'match_type', 'library']
            if match_type == 'both':
                cols.extend(['cds_score', 'pppm_score', 'cds_rank', 'pppm_rank', 'combined_rank'])
            return pd.DataFrame(columns=cols)
        
        df = pd.DataFrame(matches)
        
        # Reorder columns - include dataset and rank columns if present
        base_cols = ['bodyId', 'dataset', 'instance', 'type', 'status', 'score', 
                     'image_id', 'lm_sample', 'match_type', 'library']
        rank_cols = ['cds_score', 'pppm_score', 'cds_rank', 'pppm_rank', 'combined_rank']
        cols = [c for c in base_cols if c in df.columns]
        cols.extend([c for c in rank_cols if c in df.columns])
        df = df[cols]
        
        # Apply top_n limit if specified
        if top_n > 0 and len(df) > top_n:
            df = df.head(top_n)
        
        # Cache results (cache full results, not limited)
        self._save_to_cache('line_to_neuron', cache_key, df)
        
        self._vprint(f"  ✓ Found {len(df)} matches")
        
        return df
    
    def save_results(
        self, 
        results: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
        output_path: str,
        include_timestamp: bool = True
    ) -> str:
        """
        Save results to a CSV file.
        
        Parameters
        ----------
        results : DataFrame or dict
            Results from id_to_lines, neuron_to_lines, or line_to_neuron.
        output_path : str
            Output file path.
        include_timestamp : bool
            Whether to include timestamp in filename. Default: True
            
        Returns
        -------
        str
            Path to the saved file.
        """
        if include_timestamp:
            base, ext = os.path.splitext(output_path)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"{base}_{timestamp}{ext}"
        
        if isinstance(results, dict):
            # Combine dict of DataFrames into one
            combined = []
            for body_id, df in results.items():
                if not df.empty:
                    df_copy = df.copy()
                    df_copy.insert(0, 'query_bodyId', body_id)
                    combined.append(df_copy)
            
            if combined:
                final_df = pd.concat(combined, ignore_index=True)
            else:
                final_df = pd.DataFrame()
        else:
            final_df = results
        
        final_df.to_csv(output_path, index=False)
        self._vprint(f"💾 Saved results to: {output_path}")
        
        return output_path
    
    # =========================================================================
    # Batch Processing Methods (for simplified script usage)
    # =========================================================================
    
    def find_neurons_batch(
        self,
        line_names: Union[str, List[str]],
        top_n: int = -1,
        match_type: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Find EM neurons for multiple driver lines with automatic saving.
        
        This is a convenience method that processes multiple lines, adds source
        information, and optionally saves results to files.
        
        Parameters
        ----------
        line_names : str or list
            Driver line name(s). Can be comma-separated string or list.
        top_n : int
            Maximum matches per line. Default: -1 (all matches)
        match_type : str, optional
            Match algorithm: 'cds', 'pppm', or 'both'. 
            If None, uses self.match_type. Default: None
        output_dir : str, optional
            Directory to save results. If provided, saves individual and combined CSVs.
            
        Returns
        -------
        pd.DataFrame
            Combined DataFrame with all results, including 'source_line' column.
            
        Example
        -------
        >>> nbf = NeuronBridgeFinder()
        >>> results = nbf.find_neurons_batch('LH173,VT037867', output_dir='./output')
        """
        # Use class-level match_type if not specified
        if match_type is None:
            match_type = self.match_type
        # Parse line names
        if isinstance(line_names, str):
            lines = [ln.strip() for ln in line_names.split(',') if ln.strip()]
        else:
            lines = list(line_names)
        
        if not lines:
            self._vprint("❌ No line names provided")
            return pd.DataFrame()
        
        self._vprint(f"🔍 Finding neurons for {len(lines)} line(s)")
        
        # Create output directory if needed
        output_path = None
        if output_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(output_dir, f'findneuron_{timestamp}')
            os.makedirs(output_path, exist_ok=True)
            self._vprint(f"   Output: {output_path}")
        
        # Process each line
        all_results = []
        
        # Add progress bar for multiple lines
        if HAS_TQDM and self.verbose and len(lines) > 1:
            from tqdm import tqdm as tqdm_progress
            line_iterator = tqdm_progress(
                lines,
                desc="   🧬 Finding neurons",
                unit="line",
                bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                ncols=110,
                position=0
            )
        else:
            line_iterator = lines
        
        for idx, line_name in enumerate(line_iterator):
            # Update progress description
            if HAS_TQDM and self.verbose and len(lines) > 1:
                line_iterator.set_description(f"   🧬 [{idx+1}/{len(lines)}] {line_name}")
                
                # Check cache status
                cache_key = f"{line_name}_{match_type}"
                is_cached = self._load_from_cache('line_to_neuron', cache_key) is not None
                status = "💾cached" if is_cached else "🌐fetching"
                line_iterator.set_postfix_str(status)
            
            # Show individual processing message only if not using progress bar
            if not (HAS_TQDM and self.verbose and len(lines) > 1):
                self._vprint(f"\n📋 Processing: {line_name}")
            
            try:
                neurons_df = self.line_to_neuron(
                    line_name,
                    top_n=top_n,
                    match_type=match_type
                )
                
                if neurons_df.empty:
                    if not (HAS_TQDM and self.verbose and len(lines) > 1):
                        self._vprint(f"   ⚠️ No matching neurons found")
                    continue
                
                # Add source line info
                neurons_df = neurons_df.copy()
                neurons_df['source_line'] = line_name
                all_results.append(neurons_df)
                
                # Show results only if not using progress bar
                if not (HAS_TQDM and self.verbose and len(lines) > 1):
                    self._vprint(f"   ✅ Found {len(neurons_df)} matching neurons")
                    
                    # Show dataset distribution
                    if 'dataset' in neurons_df.columns:
                        datasets = neurons_df['dataset'].value_counts()
                        for ds, count in datasets.items():
                            self._vprint(f"      {ds}: {count}")
                
                # Save individual results
                if output_path:
                    output_file = os.path.join(output_path, f'{line_name}_neurons.csv')
                    neurons_df.to_csv(output_file, index=False)
                    if not (HAS_TQDM and self.verbose and len(lines) > 1):
                        self._vprint(f"   💾 Saved: {output_file}")
                    
            except Exception as e:
                if not (HAS_TQDM and self.verbose and len(lines) > 1):
                    self._vprint(f"   ❌ Error: {e}")
        
        # Combine results
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # Show summary statistics
            self._vprint(f"\n📊 Summary:")
            self._vprint(f"   Total neurons found: {len(combined_df)}")
            self._vprint(f"   From {len(lines)} line(s), {len(all_results)} had matches")
            
            # Dataset distribution
            if 'dataset' in combined_df.columns:
                self._vprint(f"   Dataset distribution:")
                datasets = combined_df['dataset'].value_counts()
                for ds, count in datasets.items():
                    self._vprint(f"      {ds}: {count}")
            
            # Save combined results
            if output_path:
                combined_file = os.path.join(output_path, 'all_neurons.csv')
                combined_df.to_csv(combined_file, index=False)
                self._vprint(f"\n💾 Combined results: {combined_file}")
            
            return combined_df
        
        self._vprint(f"\n⚠️ No neurons found for any of the {len(lines)} line(s)")
        return pd.DataFrame()
    
    def find_lines_batch(
        self,
        queries: Union[str, int, List],
        dataset: Optional[Union[str, List[str]]] = None,
        match_type: Optional[str] = None,
        output_dir: Optional[str] = None,
        download_images: Optional[str] = None,
        download_top_n_img: Optional[int] = 10,
        image_formats: Union[str, List[str]] = ['png','jpg'],
        image_types: Union[str, List[str]] = 'mip',
        max_images_per_line: Optional[int] = 20,
        flylight_category: Optional[Union[str, List[str]]] = ['GAL4/LEXA', 'SplitGAL4'],
        organize_by_region: bool = False,
        simple_mode: bool = False,
        calculate_specificity: bool = True,
        specificity_top_n: int = 100
    ) -> pd.DataFrame:
        """
        Find driver lines for multiple EM neurons with automatic saving.
        
        This is a convenience method that processes multiple queries (bodyIds,
        types, or instances), adds source information, optionally saves results,
        and optionally downloads images.
        
        Parameters
        ----------
        queries : str, int, or list
            Neuron query(s). Can be bodyId, type, instance, or comma-separated string.
        dataset : str, list of str, or None
            Dataset(s) for type/instance lookups (e.g., 'hemibrain:v1.2.1').
            Can be a single string or a list of datasets to search multiple.
            Set to None to search ALL available datasets.
        match_type : str, optional
            Match algorithm: 'cds', 'pppm', or 'both'. 
            If None, uses self.match_type. Default: None
        output_dir : str, optional
            Directory to save results. If provided, saves individual and combined CSVs.
        download_images : str, optional
            Image download source (case-insensitive):
            - 'neuronbridge': Download CDM images from NeuronBridge
            - 'flylight': Download images from FlyLight (S3/HTTP CDN)
            - 'both': Download from both sources
            - None/False: No image download (default)
        download_top_n_img : int, optional
            Download images only for top N lines (by aggregate score/rank).
            Default: None (download for all lines)
        image_formats : str or list
            File formats to download. For neuronbridge: 'png', 'jpg'.
            For flylight: 'png', 'jpg', 'h5j', 'lsm', 'mp4', 'json', 'all'.
            Default: 'png'
        image_types : str or list
            Image types to download. For neuronbridge: 'cdm', 'mip'.
            For flylight: 'mip', 'cdm', 'aligned', 'translation', 'metadata', 'all'.
            Default: 'all' (download all available image types)
        max_images_per_line : int, optional
            Maximum images to download per line. Default: None (no limit)
        flylight_category : str or list, optional
            FlyLight collection category for flylight downloads.
            Options: 'GAL4/LEXA', 'SplitGAL4', 'MCFO', 'RawImages', 'All'.
            Default: ['GAL4/LEXA', 'SplitGAL4'] (search these collections)
        organize_by_region : bool
            If True, organize downloaded images into Brain/VNC subfolders
            based on the anatomical region in the filename. Default: False
        simple_mode : bool
            If True, apply filename filtering to reduce download volume:
            - Split-GAL4 collections: only files with '20x' AND 'multichannel' in filename
            - GAL4/LexA collections: only files with 'total' in filename
            Default: False
        calculate_specificity : bool
            If True, calculate specificity metrics for each line by calling line_to_neuron
            to determine how specific each line is to the queried neuron types.
            Adds columns: rank_sum, type_proportion, n_queried_types, n_total_types,
            selectivity, specificity_score. Only works for type/instance queries.
            Default: True
        specificity_top_n : int
            Controls both:
            1. Maximum number of lines to calculate specificity for (to limit API calls)
            2. Number of top neuron matches to consider when analyzing each line
            Lines are selected based on their ranking (top N by score).
            Default: 100
        
        Notes
        -----
        When `separate_splitgal4=True` is set on the NeuronBridgeFinder instance:
        - Results will include a 'line_type' column ('gal4_lexa' or 'split_gal4')
        - download_top_n_img applies separately to each category
        - GAL4/LexA lines: start with VT, R, GMR (e.g., VT037867, R10A06)
        - Split-GAL4 lines: start with SS, LH, MB, IS, OL, LC, etc.
        - Separate CSV files saved: gal4_lexa_lines.csv, split_gal4_lines.csv
        
        Specificity Metrics (when calculate_specificity=True):
        - rank_sum: Sum of ranks for queried neuron types (lower = better)
        - type_proportion: Queried types / Total types labeled (higher = more specific)
        - selectivity: 1 / n_total_types (higher = labels fewer cell types)
        - specificity_score: Combined score (type_proportion - 0.3 * normalized_rank_sum)
            
        Returns
        -------
        pd.DataFrame
            Combined DataFrame with all results, including:
            - 'source_query': original query string
            - 'source_bodyId': matching bodyId for type/instance queries
            - For match_type='both': cds_score, pppm_score, cds_rank, pppm_rank, combined_rank
            
        Example
        -------
        >>> nbf = NeuronBridgeFinder()
        >>> # Basic search
        >>> results = nbf.find_lines_batch('aMe12,MBON01', dataset='hemibrain:v1.2.1')
        >>> # With NeuronBridge images (top 50 lines only)
        >>> results = nbf.find_lines_batch('aMe12', download_images='neuronbridge', 
        ...                                 download_top_n_img=50, output_dir='./output')
        >>> # With FlyLight images
        >>> results = nbf.find_lines_batch('aMe12', download_images='flylight', output_dir='./output')
        >>> # With simple mode (reduced download volume)
        >>> results = nbf.find_lines_batch('aMe12', download_images='flylight', 
        ...                                 simple_mode=True, output_dir='./output')
        >>> # Separate GAL4/LexA from Split-GAL4 (set on instance)
        >>> nbf = NeuronBridgeFinder(separate_splitgal4=True)
        >>> results = nbf.find_lines_batch('MBON01', download_top_n_img=5)  # 5 GAL4 + 5 Split-GAL4
        """
        # Parse queries
        if isinstance(queries, str):
            query_list = [q.strip() for q in queries.split(',') if q.strip()]
        elif isinstance(queries, int):
            query_list = [queries]
        else:
            query_list = list(queries)
        
        # Use class-level match_type if not specified
        if match_type is None:
            match_type = self.match_type
        
        if not query_list:
            self._vprint("❌ No queries provided")
            return pd.DataFrame()
        
        self._vprint(f"🔍 Finding lines for {len(query_list)} query(s)")
        
        # Create output directory if needed
        output_path = None
        if output_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(output_dir, f'findlines_{timestamp}')
            os.makedirs(output_path, exist_ok=True)
            self._vprint(f"   Output: {output_path}")
        
        # Process each query
        all_results = []
        
        for q in query_list:
            self._vprint(f"\n📋 Processing: {q}")
            
            try:
                # Check if query is a body ID (integer)
                try:
                    body_id = int(q)
                    is_body_id = True
                except (ValueError, TypeError):
                    body_id = None
                    is_body_id = False
                
                if is_body_id:
                    # Direct body ID search
                    lines_df = self.id_to_lines(
                        body_id,
                        match_type=match_type
                    )
                    if not lines_df.empty:
                        lines_df = lines_df.copy()
                        lines_df['source_bodyId'] = body_id
                    query_name = str(body_id)
                else:
                    # Type/instance search - use neuron_to_lines with dataset parameter
                    results_dict = self.neuron_to_lines(
                        q,
                        dataset=dataset,
                        match_type=match_type
                    )
                    # Combine results from all matching neurons (source_bodyId already added)
                    if results_dict:
                        dfs = [df for df in results_dict.values() if not df.empty]
                        lines_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
                    else:
                        lines_df = pd.DataFrame()
                    query_name = str(q)
                
                if lines_df.empty:
                    self._vprint(f"   ⚠️ No matching lines found")
                    continue
                
                # Add source query info
                lines_df = lines_df.copy()
                lines_df['source_query'] = query_name
                all_results.append(lines_df)
                
                self._vprint(f"   ✅ Found {len(lines_df)} matching driver lines")
                
                # Save individual results
                if output_path:
                    safe_name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in query_name)
                    output_file = os.path.join(output_path, f'{safe_name}_lines.csv')
                    lines_df.to_csv(output_file, index=False)
                    self._vprint(f"   💾 Saved: {output_file}")
                    
            except Exception as e:
                self._vprint(f"   ❌ Error: {e}")
        
        # Combine results
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # Add line_type classification if separate_splitgal4 is enabled
            if self.separate_splitgal4 and 'line' in combined_df.columns:
                combined_df['line_type'] = combined_df['line'].apply(self._classify_line_type)
            
            # Aggregate line-level results across bodyIds for ranking
            # For each unique line, compute aggregate score/rank
            if 'line' in combined_df.columns:
                if match_type == 'both' and 'combined_rank' in combined_df.columns:
                    # For 'both', use mean combined_rank (lower is better)
                    agg_dict = {
                        'combined_rank': 'mean',
                        'score': 'max',
                        'source_bodyId': lambda x: ','.join(str(v) for v in x.unique())
                    }
                    if self.separate_splitgal4 and 'line_type' in combined_df.columns:
                        agg_dict['line_type'] = 'first'
                    
                    line_stats = combined_df.groupby('line').agg(agg_dict).reset_index()
                    line_stats = line_stats.rename(columns={
                        'combined_rank': 'agg_combined_rank',
                        'score': 'agg_max_score',
                        'source_bodyId': 'matched_bodyIds'
                    })
                    line_stats = line_stats.sort_values('agg_combined_rank', ascending=True)
                else:
                    # For cds/pppm, use mean score (higher is better)
                    agg_dict = {
                        'score': ['mean', 'max', 'count'],
                        'source_bodyId': lambda x: ','.join(str(v) for v in x.unique())
                    }
                    line_stats = combined_df.groupby('line').agg(agg_dict).reset_index()
                    line_stats.columns = ['line', 'agg_mean_score', 'agg_max_score', 
                                          'match_count', 'matched_bodyIds']
                    
                    # Add line_type if available
                    if self.separate_splitgal4 and 'line_type' in combined_df.columns:
                        line_type_map = combined_df.groupby('line')['line_type'].first()
                        line_stats['line_type'] = line_stats['line'].map(line_type_map)
                    
                    line_stats = line_stats.sort_values('agg_mean_score', ascending=False)
                
                # Add aggregated stats back to combined_df
                merge_cols = ['line', 'matched_bodyIds']
                combined_df = combined_df.merge(
                    line_stats[merge_cols], 
                    on='line', 
                    how='left'
                )
            
            # Calculate specificity metrics if requested
            if calculate_specificity and 'line' in combined_df.columns:
                # Collect queried types from non-bodyId queries
                queried_types = []
                for q in query_list:
                    # Check if this is NOT a bodyId (bodyIds are numeric)
                    q_str = str(q).strip()
                    if not q_str.isdigit():
                        # This is a type/instance query
                        queried_types.append(q_str)
                
                if queried_types:
                    self._vprint(f"\n📊 Calculating line specificity for {len(queried_types)} queried types...")
                    
                    # Calculate specificity on line_stats (limited to top N lines)
                    line_stats = self._calculate_line_specificity(
                        line_stats=line_stats,
                        queried_types=queried_types,
                        match_type=match_type,
                        top_n=specificity_top_n,
                        max_lines=specificity_top_n
                    )
                    
                    n_processed = min(specificity_top_n, len(line_stats)) if specificity_top_n else len(line_stats)
                    self._vprint(f"   ✅ Added specificity metrics to top {n_processed} of {len(line_stats)} lines")
                    
                    # Build and visualize co-labeling matrix if output_path is set
                    if output_path and len(line_stats) > 1:
                        # Limit to top N lines for co-labeling analysis
                        colabel_top_n = min(specificity_top_n or 50, len(line_stats))
                        colabel_lines = line_stats['line'].head(colabel_top_n).tolist()
                        
                        self._vprint(f"\n🔗 Building co-labeling matrix for top {colabel_top_n} lines...")
                        
                        # Build co-labeling matrix
                        co_labeling_matrix, _ = self._build_colabeling_matrix(
                            lines=colabel_lines,
                            match_type=match_type,
                            top_n=specificity_top_n
                        )
                        
                        # Calculate sparsity metrics
                        sparsity_scores = self._calculate_colabeling_sparsity(co_labeling_matrix)
                        
                        # Add sparsity columns to line_stats
                        for line_name, scores in sparsity_scores.items():
                            mask = line_stats['line'] == line_name
                            line_stats.loc[mask, 'colabel_sparsity'] = scores['colabel_sparsity']
                            line_stats.loc[mask, 'n_colabeling_lines'] = scores['n_colabeling_lines']
                            line_stats.loc[mask, 'mean_colabel_similarity'] = scores['mean_colabel_similarity']
                        
                        # Save co-labeling matrix CSV
                        colabel_csv = os.path.join(output_path, 'colabeling_matrix.csv')
                        co_labeling_matrix.to_csv(colabel_csv)
                        self._vprint(f"   💾 Co-labeling matrix: {colabel_csv}")
                        
                        # Visualize as heatmap
                        self.visualize_colabeling_matrix(
                            co_labeling_matrix=co_labeling_matrix,
                            output_path=output_path,
                            title=f"Co-Labeling Matrix (Jaccard Similarity) - Top {colabel_top_n} Lines",
                            color_scale='purple'
                        )
                        
                        # Calculate mutual information
                        self._vprint(f"\n📐 Calculating mutual information...")
                        mi_df, expression_df = self._calculate_mutual_information(
                            lines=colabel_lines,
                            queried_types=queried_types,
                            match_type=match_type,
                            top_n=specificity_top_n
                        )
                        
                        if not mi_df.empty:
                            # Merge MI columns into line_stats
                            mi_cols = ['line', 'mutual_information', 'normalized_mi', 'queried_type_coverage']
                            line_stats = line_stats.merge(mi_df[mi_cols], on='line', how='left')
                            
                            # Save MI results
                            mi_csv = os.path.join(output_path, 'mutual_information.csv')
                            mi_df.to_csv(mi_csv, index=False)
                            self._vprint(f"   💾 Mutual information: {mi_csv}")
                            
                            # Save and visualize expression matrix
                            if not expression_df.empty:
                                expr_csv = os.path.join(output_path, 'expression_matrix.csv')
                                expression_df.to_csv(expr_csv)
                                self._vprint(f"   💾 Expression matrix: {expr_csv}")
                                
                                # Visualize expression matrix
                                self.visualize_expression_matrix(
                                    expression_df=expression_df,
                                    output_path=output_path,
                                    queried_types=queried_types,
                                    title=f"Expression Matrix (Lines × Types) - Top {colabel_top_n} Lines"
                                )
                else:
                    self._vprint(f"\n⚠️ Skipping specificity: No type queries found (all queries are bodyIds)")
            
            # Save combined results
            if output_path:
                combined_file = os.path.join(output_path, 'all_lines.csv')
                combined_df.to_csv(combined_file, index=False)
                self._vprint(f"\n💾 Combined results: {combined_file}")
                self._vprint(f"   Total: {len(combined_df)} lines from {len(query_list)} query(s)")
                
                # Save line-level aggregate summary
                if 'line' in combined_df.columns:
                    summary_file = os.path.join(output_path, 'line_summary.csv')
                    line_stats.to_csv(summary_file, index=False)
                    self._vprint(f"   Summary: {summary_file} ({len(line_stats)} unique lines)")
                    
                    # Save separate files for GAL4/LexA and Split-GAL4 if enabled
                    if self.separate_splitgal4 and 'line_type' in line_stats.columns:
                        # Split combined_df by line type
                        gal4_lexa_combined = combined_df[combined_df['line_type'] == 'gal4_lexa']
                        split_gal4_combined = combined_df[combined_df['line_type'] == 'split_gal4']
                        
                        # Split line_stats by line type
                        gal4_lexa_stats = line_stats[line_stats['line_type'] == 'gal4_lexa']
                        split_gal4_stats = line_stats[line_stats['line_type'] == 'split_gal4']
                        
                        # Save GAL4/LexA files
                        if not gal4_lexa_combined.empty:
                            gal4_file = os.path.join(output_path, 'gal4_lexa_lines.csv')
                            gal4_lexa_combined.to_csv(gal4_file, index=False)
                            gal4_summary = os.path.join(output_path, 'gal4_lexa_summary.csv')
                            gal4_lexa_stats.to_csv(gal4_summary, index=False)
                            self._vprint(f"   GAL4/LexA: {gal4_file} ({len(gal4_lexa_combined)} matches, {len(gal4_lexa_stats)} unique)")
                        
                        # Save Split-GAL4 files
                        if not split_gal4_combined.empty:
                            split_file = os.path.join(output_path, 'split_gal4_lines.csv')
                            split_gal4_combined.to_csv(split_file, index=False)
                            split_summary = os.path.join(output_path, 'split_gal4_summary.csv')
                            split_gal4_stats.to_csv(split_summary, index=False)
                            self._vprint(f"   Split-GAL4: {split_file} ({len(split_gal4_combined)} matches, {len(split_gal4_stats)} unique)")
            
            # Download images if requested
            if download_images and output_path:
                # Normalize download_images parameter (case-insensitive)
                download_source = download_images.lower() if isinstance(download_images, str) else None
                
                if download_source in ('neuronbridge', 'flylight', 'both'):
                    # Get line names for image download (use aggregated ranking)
                    if 'line' in combined_df.columns:
                        # Use pre-computed line_stats for ordering
                        all_lines = line_stats['line'].tolist()
                        
                        # Handle separate_splitgal4 mode
                        if self.separate_splitgal4:
                            # Separate lines by type and get top N from each
                            gal4_lines = [l for l in all_lines if self._classify_line_type(l) == 'gal4_lexa']
                            split_lines = [l for l in all_lines if self._classify_line_type(l) == 'split_gal4']
                            
                            if download_top_n_img is not None and download_top_n_img > 0:
                                gal4_lines = gal4_lines[:download_top_n_img]
                                split_lines = split_lines[:download_top_n_img]
                                self._vprint(f"\n🖼️  Downloading images (separate mode):")
                                self._vprint(f"      GAL4/LexA: top {len(gal4_lines)} lines")
                                self._vprint(f"      Split-GAL4: top {len(split_lines)} lines")
                            else:
                                self._vprint(f"\n🖼️  Downloading images (separate mode):")
                                self._vprint(f"      GAL4/LexA: {len(gal4_lines)} lines")
                                self._vprint(f"      Split-GAL4: {len(split_lines)} lines")
                            
                            download_lines = gal4_lines + split_lines
                        else:
                            # Normal mode: just apply top_n limit to all lines
                            if download_top_n_img is not None and download_top_n_img > 0:
                                download_lines = all_lines[:download_top_n_img]
                                self._vprint(f"\n🖼️  Downloading images for top {len(download_lines)} lines...")
                            else:
                                download_lines = all_lines
                                self._vprint(f"\n🖼️  Downloading images for {len(download_lines)} lines...")
                    else:
                        download_lines = []
                    
                    if download_lines:
                        images_dir = os.path.join(output_path, 'images')
                        
                        # Download from NeuronBridge
                        if download_source in ('neuronbridge', 'both'):
                            nb_dir = os.path.join(images_dir, 'neuronbridge') if download_source == 'both' else images_dir
                            self._download_neuronbridge_images(
                                lines=download_lines,
                                output_dir=nb_dir,
                                formats=image_formats,
                                image_types=image_types,
                                max_files=max_images_per_line,
                                verbose=self.verbose
                            )
                        
                        # Download from FlyLight
                        if download_source in ('flylight', 'both'):
                            fl_dir = os.path.join(images_dir, 'flylight') if download_source == 'both' else images_dir
                            self._download_flylight_images_with_category(
                                lines=download_lines,
                                output_dir=fl_dir,
                                formats=image_formats,
                                image_types=image_types,
                                max_files=max_images_per_line,
                                category=flylight_category,
                                organize_by_region=organize_by_region,
                                simple_mode=simple_mode,
                                verbose=self.verbose
                            )
            
            return combined_df
        
        return pd.DataFrame()
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """
        Clear cached results.
        
        Parameters
        ----------
        cache_type : str, optional
            Type of cache to clear: 'id_to_lines', 'line_to_neuron', or None (all).
        """
        if not os.path.exists(self.cache_folder):
            return
        
        import glob
        
        if cache_type:
            pattern = os.path.join(self.cache_folder, f"{cache_type}_*.csv")
        else:
            pattern = os.path.join(self.cache_folder, "*.csv")
        
        files = glob.glob(pattern)
        for f in files:
            try:
                os.remove(f)
            except Exception:
                pass
        
        self._vprint(f"🗑️ Cleared {len(files)} cached files")
    
    # =========================================================================
    # Image Download Methods
    # =========================================================================
    
    def download_line_images(
        self,
        line_names: Union[str, List[str]],
        output_dir: str,
        source: str = 'neuronbridge',
        formats: Union[str, List[str]] = 'png',
        image_types: Union[str, List[str]] = 'cdm',
        max_files: Optional[int] = None,
        verbose: Optional[bool] = None
    ) -> List[str]:
        """
        Download images for driver lines from NeuronBridge or FlyLight.
        
        Parameters
        ----------
        line_names : str or list
            Driver line name(s) to download images for.
            Can be comma-separated string or list.
        output_dir : str
            Directory to save downloaded images.
        source : str
            Image source: 'neuronbridge' for CDM images, 'flylight' for raw images.
            Default: 'neuronbridge'
        formats : str or list
            File formats to download. For neuronbridge: 'png', 'jpg'. 
            For flylight: 'png', 'jpg', 'h5j', 'lsm', 'mp4', 'json', 'all'.
            Default: 'png'
        image_types : str or list
            Image types to download. For neuronbridge: 'cdm', 'mip'.
            For flylight: 'mip', 'cdm', 'aligned', 'translation', 'metadata', 'all'.
            Default: 'cdm'
        max_files : int, optional
            Maximum number of files to download per line. Default: None (no limit)
        verbose : bool, optional
            Override class verbose setting. Default: use class setting.
            
        Returns
        -------
        list
            List of downloaded file paths.
            
        Example
        -------
        >>> nbf = NeuronBridgeFinder()
        >>> files = nbf.download_line_images('LH173', './images', source='neuronbridge')
        >>> files = nbf.download_line_images('VT037867', './images', source='flylight')
        """
        if verbose is None:
            verbose = self.verbose
        
        # Parse line names
        if isinstance(line_names, str):
            lines = [l.strip() for l in line_names.split(',') if l.strip()]
        else:
            lines = list(line_names)
        
        if not lines:
            if verbose:
                print("❌ No line names provided")
            return []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        all_downloaded = []
        
        if source.lower() == 'neuronbridge':
            all_downloaded = self._download_neuronbridge_images(
                lines, output_dir, formats, image_types, max_files, verbose
            )
        elif source.lower() == 'flylight':
            all_downloaded = self._download_flylight_images(
                lines, output_dir, formats, image_types, max_files, verbose
            )
        else:
            if verbose:
                print(f"❌ Unknown source: {source}. Use 'neuronbridge' or 'flylight'.")
        
        return all_downloaded
    
    def _download_neuronbridge_images(
        self,
        lines: List[str],
        output_dir: str,
        formats: Union[str, List[str]],
        image_types: Union[str, List[str]],
        max_files: Optional[int],
        verbose: bool
    ) -> List[str]:
        """Download CDM images from NeuronBridge with progress bar."""
        import urllib.request
        
        downloaded = []
        
        # NeuronBridge CDM URL patterns
        # CDM files are at: https://s3.amazonaws.com/janelia-flylight-color-depth/{path}
        cdm_base_url = "https://s3.amazonaws.com/janelia-flylight-color-depth/"
        
        # Phase 1: Scan all lines to count total files
        if verbose:
            print(f"📊 Scanning {len(lines)} lines for NeuronBridge images...")
        
        line_files_map = {}  # line_name -> list of (file_type, file_path)
        total_file_count = 0
        
        for line_name in lines:
            try:
                lm_images = self._client.get_lm_images(line_name)
                if not lm_images:
                    line_files_map[line_name] = []
                    continue
                
                if not isinstance(lm_images, list):
                    lm_images = list(lm_images)
                
                file_paths = []
                for lm_image in lm_images:
                    if max_files and len(file_paths) >= max_files:
                        break
                    
                    files_obj = getattr(lm_image, 'files', None)
                    if not files_obj:
                        continue
                    
                    img_types = image_types if isinstance(image_types, list) else [image_types]
                    
                    for img_type in img_types:
                        if img_type.lower() in ['cdm', 'all']:
                            cdm_path = getattr(files_obj, 'CDM', None)
                            if cdm_path:
                                file_paths.append(('cdm', cdm_path))
                        if img_type.lower() in ['mip', 'all']:
                            mip_path = getattr(files_obj, 'SignalMip', None)
                            if mip_path:
                                file_paths.append(('mip', mip_path))
                            mip_masked = getattr(files_obj, 'SignalMipMasked', None)
                            if mip_masked:
                                file_paths.append(('mip_masked', mip_masked))
                
                # Filter by format
                fmt_list = formats if isinstance(formats, list) else [formats]
                filtered_paths = []
                for file_type, file_path in file_paths:
                    ext = os.path.splitext(file_path)[1].lower().lstrip('.')
                    if 'all' in fmt_list or ext in [f.lstrip('.').lower() for f in fmt_list]:
                        filtered_paths.append((file_type, file_path))
                
                if max_files:
                    filtered_paths = filtered_paths[:max_files]
                
                line_files_map[line_name] = filtered_paths
                total_file_count += len(filtered_paths)
                
            except Exception:
                line_files_map[line_name] = []
        
        if total_file_count == 0:
            if verbose:
                print("  ⚠️ No files found for any lines")
            return []
        
        lines_with_files = len([l for l, f in line_files_map.items() if f])
        if verbose:
            print(f"  📦 Found {total_file_count} files across {lines_with_files} lines")
        
        # Phase 2: Download with progress bar
        pbar = None
        if HAS_TQDM and verbose:
            pbar = tqdm(total=total_file_count, desc="  Downloading", unit="file")
        
        files_downloaded = 0
        for line_name in lines:
            if line_name not in line_files_map or not line_files_map[line_name]:
                continue
            
            try:
                line_dir = os.path.join(output_dir, line_name)
                os.makedirs(line_dir, exist_ok=True)
                
                for file_type, file_path in line_files_map[line_name]:
                    url = cdm_base_url + file_path
                    filename = os.path.basename(file_path)
                    local_path = os.path.join(line_dir, filename)
                    
                    try:
                        if pbar:
                            pbar.set_postfix(line=line_name[:15], refresh=False)
                        urllib.request.urlretrieve(url, local_path)
                        downloaded.append(local_path)
                        files_downloaded += 1
                        if pbar:
                            pbar.update(1)
                        elif verbose:
                            print(f"  [{files_downloaded}/{total_file_count}] {line_name}: {filename}")
                    except Exception as e:
                        if verbose and not pbar:
                            print(f"  ⚠️ Failed to download {filename}: {e}")
                        
            except Exception as e:
                if verbose and not pbar:
                    print(f"  ❌ Error processing '{line_name}': {e}")
        
        if pbar:
            pbar.close()
        
        if verbose:
            print(f"  ✅ Downloaded {len(downloaded)}/{total_file_count} files")
        
        return downloaded
    
    def _download_flylight_images(
        self,
        lines: List[str],
        output_dir: str,
        formats: Union[str, List[str]],
        image_types: Union[str, List[str]],
        max_files: Optional[int],
        verbose: bool
    ) -> List[str]:
        """Download images from FlyLight using flylight_downloader module."""
        return self._download_flylight_images_with_category(
            lines=lines,
            output_dir=output_dir,
            formats=formats,
            image_types=image_types,
            max_files=max_files,
            category=None,
            organize_by_region=False,
            verbose=verbose
        )
    
    def _parse_region_from_filename(self, filename: str) -> str:
        """
        Parse anatomical region from FlyLight filename.
        
        FlyLight filename format: {line}-{date}_{sample}-{sex}-{mag}-{region}-{driver}-...
        Example: SS01015-20131220_31_C3-f-20x-brain-Split_GAL4-JRC2018_Unisex_20x_HR-CDM_1.png
        
        Returns 'Brain', 'VNC', or 'Other' based on the region field.
        """
        # Split by '-' and try to find region field
        parts = filename.split('-')
        
        # Region is typically the 5th field (index 4) after line, date, sex, mag
        # But some files have extra parts, so search for known regions
        for part in parts:
            part_lower = part.lower()
            if 'brain' in part_lower:
                return 'Brain'
            elif 'ventral_nerve_cord' in part_lower or 'vnc' in part_lower:
                return 'VNC'
            elif 'dorsal' in part_lower or 'thorax' in part_lower or 'optic' in part_lower:
                return 'Other'
        
        # Fallback: try position-based parsing
        if len(parts) >= 5:
            region_field = parts[4].lower()
            if 'brain' in region_field:
                return 'Brain'
            elif 'ventral' in region_field or 'vnc' in region_field:
                return 'VNC'
        
        return 'Other'
    
    def _reorganize_files_by_region(
        self,
        downloaded_files: List[str],
        output_dir: str,
        verbose: bool = False
    ) -> List[str]:
        """
        Reorganize downloaded files into Brain/VNC/Other subfolders.
        
        Original structure: output_dir/Collection/LineName/filename.png
        New structure: output_dir/Brain/LineName/filename.png (or VNC or Other)
        
        Returns list of new file paths.
        """
        import shutil
        from pathlib import Path
        
        reorganized = []
        region_counts = {'Brain': 0, 'VNC': 0, 'Other': 0}
        
        for file_path in downloaded_files:
            file_path = Path(file_path)
            if not file_path.exists():
                continue
            
            filename = file_path.name
            
            # Parse region and line name from filename
            region = self._parse_region_from_filename(filename)
            
            # Extract line name (first field before '-')
            line_name = filename.split('-')[0] if '-' in filename else filename.split('.')[0]
            
            # Create new path: output_dir/Region/LineName/filename
            new_dir = Path(output_dir) / region / line_name
            new_dir.mkdir(parents=True, exist_ok=True)
            new_path = new_dir / filename
            
            try:
                # Move file to new location
                shutil.move(str(file_path), str(new_path))
                reorganized.append(str(new_path))
                region_counts[region] += 1
            except Exception as e:
                # If move fails, try copy
                try:
                    shutil.copy2(str(file_path), str(new_path))
                    file_path.unlink()  # Delete original
                    reorganized.append(str(new_path))
                    region_counts[region] += 1
                except Exception as e2:
                    if verbose:
                        print(f"  ⚠️  Could not reorganize {filename}: {e2}")
                    reorganized.append(str(file_path))  # Keep original
        
        # Clean up empty collection directories
        try:
            for item in Path(output_dir).iterdir():
                if item.is_dir() and item.name not in ('Brain', 'VNC', 'Other'):
                    # Check if empty (or has only empty subdirs)
                    remaining_files = list(item.rglob('*'))
                    remaining_files = [f for f in remaining_files if f.is_file()]
                    if not remaining_files:
                        shutil.rmtree(str(item), ignore_errors=True)
        except Exception:
            pass
        
        if verbose:
            print(f"  📁 Reorganized by region: Brain={region_counts['Brain']}, VNC={region_counts['VNC']}, Other={region_counts['Other']}")
        
        return reorganized
    
    def _download_flylight_images_with_category(
        self,
        lines: List[str],
        output_dir: str,
        formats: Union[str, List[str]],
        image_types: Union[str, List[str]],
        max_files: Optional[int],
        category: Optional[Union[str, List[str]]],
        organize_by_region: bool = False,
        simple_mode: bool = False,
        verbose: bool = False
    ) -> List[str]:
        """
        Download images from FlyLight with optional category filtering.
        
        Parameters
        ----------
        simple_mode : bool
            If True, apply filename filtering to reduce download volume:
            - Split-GAL4 collections: only files with '20x' AND 'multichannel' in filename
            - GAL4/LexA collections: only files with 'total' in filename
            Default: False
        """
        try:
            from flylight_downloader import FlyLightDownloader
        except ImportError:
            try:
                # Try relative import
                from .flylight_downloader import FlyLightDownloader
            except ImportError:
                if verbose:
                    print("❌ flylight_downloader module not available")
                return []
        
        def apply_simple_mode_filter(files, collection_name: str):
            """Filter files based on simple_mode rules."""
            if not simple_mode or not files:
                return files
            
            collection_lower = collection_name.lower() if collection_name else ''
            filtered = []
            
            for f in files:
                filename_lower = f.filename.lower()
                
                # Split-GAL4: only include '20x' AND 'multichannel', exclude 'image1'/'image2'
                if 'splitgal4' in collection_lower or 'split-gal4' in collection_lower or 'split_gal4' in collection_lower:
                    if '20x' in filename_lower and 'multichannel' in filename_lower:
                        # Exclude duplicate images (image1, image2, etc.)
                        if 'image1' not in filename_lower and 'image2' not in filename_lower:
                            filtered.append(f)
                # GAL4/LexA: only include 'total'
                elif 'gal4' in collection_lower or 'lexa' in collection_lower:
                    if 'total' in filename_lower:
                        filtered.append(f)
                else:
                    # For other collections (MCFO, etc.), keep all files
                    filtered.append(f)
            
            return filtered
        
        downloaded = []
        
        # Separate VT lines from other lines (they need different format settings)
        vt_lines = [l for l in lines if l.upper().startswith('VT')]
        other_lines = [l for l in lines if not l.upper().startswith('VT')]
        
        # Phase 1: Scan all lines to count total files
        if verbose:
            if simple_mode:
                print(f"📊 Scanning {len(lines)} lines to count files (simple mode: filtering filenames)...")
            else:
                print(f"📊 Scanning {len(lines)} lines to count files...")
        
        line_files_map = {}  # line_name -> list of FlyLightFile
        total_file_count = 0
        
        # Create a scanning progress bar
        scan_pbar = None
        if HAS_TQDM and verbose:
            scan_pbar = tqdm(total=len(lines), desc="  Scanning", unit="line", leave=False)
        
        # Create downloaders for scanning
        if other_lines:
            downloader = FlyLightDownloader(
                output_dir=output_dir,
                collection_category=category,
                formats=formats,
                image_types=image_types,
                verbose=False
            )
            
            for line_name in other_lines:
                try:
                    files = downloader.get_filtered_files(line_name)
                    # Apply simple_mode filtering based on collection
                    if files:
                        # Get collection from first file (all files same line share collection)
                        collection = files[0].collection if files else ''
                        files = apply_simple_mode_filter(files, collection)
                    if max_files:
                        files = files[:max_files]
                    line_files_map[line_name] = files
                    total_file_count += len(files)
                except Exception:
                    line_files_map[line_name] = []
                
                if scan_pbar:
                    scan_pbar.set_postfix(files=total_file_count, refresh=False)
                    scan_pbar.update(1)
        
        if vt_lines:
            vt_formats = formats
            if isinstance(vt_formats, str):
                vt_formats = [vt_formats]
            if 'png' in vt_formats and 'jpg' not in vt_formats:
                vt_formats = list(vt_formats) + ['jpg']
            
            downloader_vt = FlyLightDownloader(
                output_dir=output_dir,
                formats=vt_formats,
                image_types='all',
                verbose=False
            )
            
            for line_name in vt_lines:
                try:
                    files = downloader_vt.get_filtered_files(line_name)
                    # VT lines: apply simple_mode filter (usually GAL4 category)
                    if files:
                        collection = files[0].collection if files else 'GAL4'
                        files = apply_simple_mode_filter(files, collection)
                    if max_files:
                        files = files[:max_files]
                    line_files_map[line_name] = files
                    total_file_count += len(files)
                except Exception:
                    line_files_map[line_name] = []
                
                if scan_pbar:
                    scan_pbar.set_postfix(files=total_file_count, refresh=False)
                    scan_pbar.update(1)
        
        if scan_pbar:
            scan_pbar.close()
        
        if total_file_count == 0:
            if verbose:
                print("  ⚠️ No files found for any lines")
            return []
        
        lines_with_files = len([l for l, f in line_files_map.items() if f])
        if verbose:
            print(f"  📦 Found {total_file_count} files across {lines_with_files} lines")
        
        # Phase 2: Download with combined progress bar
        files_downloaded = [0]  # Use list to allow modification in callback
        current_line = ['']  # Track current line for progress display
        
        # Create a single progress bar for all files
        pbar = None
        if HAS_TQDM and verbose:
            pbar = tqdm(total=total_file_count, desc="  Downloading", unit="file")
        
        def on_file_downloaded(file_path, line_name):
            """Callback to update progress after each file download."""
            files_downloaded[0] += 1
            if pbar:
                # Update postfix to show current line
                if line_name != current_line[0]:
                    current_line[0] = line_name
                pbar.set_postfix(line=line_name[:15], refresh=False)
                pbar.update(1)
            elif verbose:
                # Fallback without tqdm
                print(f"  [{files_downloaded[0]}/{total_file_count}] {line_name}: {file_path.name if hasattr(file_path, 'name') else file_path}")
        
        # Download S3 lines
        if other_lines:
            downloader = FlyLightDownloader(
                output_dir=output_dir,
                collection_category=category,
                formats=formats,
                image_types=image_types,
                verbose=False
            )
            
            for line_name in other_lines:
                if line_name in line_files_map and line_files_map[line_name]:
                    try:
                        dl_files = downloader.download(
                            line_name=line_name,
                            max_files=max_files,
                            on_file_downloaded=on_file_downloaded,
                            flat_structure=True,
                            files=line_files_map[line_name]  # Use pre-filtered files
                        )
                        downloaded.extend([str(f) for f in dl_files])
                    except Exception as e:
                        if verbose and not pbar:
                            print(f"  ❌ {line_name}: {e}")
        
        # Download VT lines
        if vt_lines:
            vt_formats = formats
            if isinstance(vt_formats, str):
                vt_formats = [vt_formats]
            if 'png' in vt_formats and 'jpg' not in vt_formats:
                vt_formats = list(vt_formats) + ['jpg']
            
            downloader_vt = FlyLightDownloader(
                output_dir=output_dir,
                formats=vt_formats,
                image_types='all',
                verbose=False
            )
            
            for line_name in vt_lines:
                if line_name in line_files_map and line_files_map[line_name]:
                    try:
                        dl_files = downloader_vt.download(
                            line_name=line_name,
                            max_files=max_files,
                            on_file_downloaded=on_file_downloaded,
                            flat_structure=True,
                            files=line_files_map[line_name]  # Use pre-filtered files
                        )
                        downloaded.extend([str(f) for f in dl_files])
                    except Exception as e:
                        if verbose and not pbar:
                            print(f"  ❌ {line_name}: {e}")
        
        if pbar:
            pbar.close()
        
        if verbose:
            print(f"  ✅ Downloaded {len(downloaded)}/{total_file_count} files")
        
        # Reorganize files by anatomical region if requested
        if organize_by_region and downloaded:
            if verbose:
                print("  🔄 Reorganizing files by anatomical region...")
            downloaded = self._reorganize_files_by_region(
                downloaded_files=downloaded,
                output_dir=output_dir,
                verbose=verbose
            )
        
        return downloaded
    
    def find_lines_batch_with_images(
        self,
        queries: Union[str, int, List],
        dataset: Optional[str] = None,
        top_n: int = -1,
        match_type: str = 'cds',
        output_dir: Optional[str] = None,
        download_images: Union[bool, str] = False,
        image_source: str = 'neuronbridge',
        image_formats: Union[str, List[str]] = 'png',
        image_types: Union[str, List[str]] = 'cdm',
        max_images_per_line: Optional[int] = 5
    ) -> pd.DataFrame:
        """
        Find driver lines with optional image download.
        
        .. deprecated::
            This method is deprecated. Use find_lines_batch() with download_images parameter instead.
        
        Parameters
        ----------
        queries : str, int, or list
            Neuron query(s). Can be bodyId, type, instance, or comma-separated string.
        dataset : str, optional
            Dataset for type/instance lookups.
        top_n : int
            Maximum matches per neuron. Default: -1 (all matches)
        match_type : str
            Match algorithm: 'cds', 'pppm', or 'both'. Default: 'cds'
        output_dir : str, optional
            Directory to save results and images.
        download_images : bool or str
            Whether to download images. True uses image_source, or pass source directly.
        image_source : str
            'neuronbridge' for CDM images, 'flylight' for raw confocal. Default: 'neuronbridge'
        image_formats : str or list
            File formats for images. Default: 'png'
        image_types : str or list
            Image types to download. Default: 'cdm'
        max_images_per_line : int, optional
            Maximum images to download per line. Default: 5
            
        Returns
        -------
        pd.DataFrame
            Combined results DataFrame.
        """
        import warnings
        warnings.warn(
            "find_lines_batch_with_images() is deprecated. Use find_lines_batch() with download_images parameter.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Convert legacy parameters to new format
        if download_images:
            if isinstance(download_images, bool):
                dl_source = image_source
            else:
                dl_source = download_images
        else:
            dl_source = None
        
        return self.find_lines_batch(
            queries=queries,
            dataset=dataset,
            match_type=match_type,
            output_dir=output_dir,
            download_images=dl_source,
            image_formats=image_formats,
            image_types=image_types,
            max_images_per_line=max_images_per_line
        )


# Convenience function for quick usage
def find_lines_for_body(body_id: int, match_type: str = 'cds') -> pd.DataFrame:
    """
    Quick function to find driver lines matching a body ID.
    
    Parameters
    ----------
    body_id : int
        The EM body ID to search for.
    match_type : str
        Match algorithm: 'cds', 'pppm', or 'both'. Default: 'cds'
        
    Returns
    -------
    pd.DataFrame
        DataFrame of matching lines, sorted by score (cds/pppm) or combined_rank (both).
    """
    nbf = NeuronBridgeFinder(verbose=False)
    return nbf.id_to_lines(body_id, match_type=match_type)


def find_neurons_for_line(line_name: str, match_type: str = 'cds') -> pd.DataFrame:
    """
    Quick function to find neurons matching a driver line.
    
    Parameters
    ----------
    line_name : str
        The driver line name to search for.
    match_type : str
        Match algorithm: 'cds', 'pppm', or 'both'. Default: 'cds'
        
    Returns
    -------
    pd.DataFrame
        DataFrame of matching neurons, sorted by score (cds/pppm) or combined_rank (both).
    """
    nbf = NeuronBridgeFinder(verbose=False)
    return nbf.line_to_neuron(line_name, match_type=match_type)


if __name__ == '__main__':
    # Simple test
    print("Testing NeuronBridgeFinder...")
    
    # Initialize
    nbf = NeuronBridgeFinder()
    
    # Test id_to_lines
    print("\n--- Test id_to_lines ---")
    lines = nbf.id_to_lines(636798093)
    print(lines.head(10))
    
    # Test line_to_neuron
    print("\n--- Test line_to_neuron ---")
    neurons = nbf.line_to_neuron('LH173')
    print(neurons.head(10))
