import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional, Tuple
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from coana import FindNeuronConnection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TypeMapper:
    """
    Handles mapping of neuron types across datasets to a standardized nomenclature.
    Supports one-to-one and many-to-one mappings.
    """
    RESERVED_KEYWORDS = {'type', 'bodyId', 'instance', 'pre', 'post', 'weight', 'roi', 'connection_ratio'}

    def __init__(self, mapping: Optional[Union[Dict[str, str], str]] = None):
        """
        Initialize TypeMapper.
        
        Args:
            mapping: Dictionary mapping {original_type: std_type} or path to CSV file.
                     CSV should have columns 'original_type' and 'std_type'.
        """
        self.mapping = {}
        if mapping:
            if isinstance(mapping, str):
                self._load_from_csv(mapping)
            elif isinstance(mapping, dict):
                self.mapping = mapping
            else:
                raise ValueError("Mapping must be a dictionary or a path to a CSV file.")
        
        self._validate_mapping()

    def _load_from_csv(self, filepath: str):
        try:
            df = pd.read_csv(filepath)
            if 'original_type' not in df.columns or 'std_type' not in df.columns:
                raise ValueError("CSV must contain 'original_type' and 'std_type' columns.")
            self.mapping = dict(zip(df['original_type'], df['std_type']))
        except Exception as e:
            logger.error(f"Failed to load mapping from CSV: {e}")
            raise

    def _validate_mapping(self):
        """Ensure no mapped names conflict with reserved keywords."""
        for std_type in self.mapping.values():
            if std_type in self.RESERVED_KEYWORDS:
                raise ValueError(f"Mapped type '{std_type}' is a reserved keyword.")

    def get_std_type(self, original_type: str) -> str:
        """Return standardized type, defaulting to original if not found."""
        return self.mapping.get(original_type, original_type)

@dataclass
class DatasetConfig:
    dataset: str # Actual dataset name (e.g. "hemibrain:v1.2.1")
    name: str = '' # Optional label
    token: str = ''
    source_types: List[str] = field(default_factory=list)
    target_types: List[str] = field(default_factory=list)
    max_interlayer: int = 0 # 0 for direct connections, >0 for paths
    
    def __post_init__(self):
        if not self.name:
            self.name = self.dataset
            
        # Use TokenManager
        try:
            from utils.token_manager import token_manager
            self.token = token_manager.get_token('NEUPRINT_TOKEN', self.token)
        except ImportError:
            # Fallback
            try:
                from src.utils.token_manager import token_manager
                self.token = token_manager.get_token('NEUPRINT_TOKEN', self.token)
            except ImportError:
                pass

    
class InterDatasetComparator:
    """
    Unified framework for comparing connectivity across multiple datasets.
    """
    def __init__(self, 
                 configs: List[DatasetConfig], 
                 thresholds: List[int] = [1, 3, 5, 10, 20],
                 type_mapper: Optional[TypeMapper] = None,
                 output_dir: str = 'comparison_results'):
        
        self.configs = configs
        self.thresholds = sorted(thresholds)
        self.type_mapper = type_mapper or TypeMapper()
        self.output_dir = output_dir
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.datasets_data = {} # Store raw fetched data
        self.datasets_paths = {} # Store path data
        self.datasets_metadata = {} # Store metadata (neuron counts, etc.)
        self.aligned_data = None # Store aligned DataFrame
        self.aligned_paths = None # Store aligned paths DataFrame
        self.sensitivity_results = None

    def fetch_all_data(self):
        """
        Fetch data for all datasets at the lowest threshold using parallel execution.
        """
        min_threshold = min(self.thresholds)
        
        with ThreadPoolExecutor(max_workers=len(self.configs)) as executor:
            future_to_config = {executor.submit(self._fetch_single_dataset, config, min_threshold): config for config in self.configs}
            
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    if result:
                        conn_df, path_df, metadata = result
                        
                        if conn_df is not None and not conn_df.empty:
                            self.datasets_data[config.name] = conn_df
                            logger.info(f"Successfully fetched {len(conn_df)} connections for {config.name}")
                        else:
                            logger.warning(f"No connections found for {config.name}")
                            
                        if path_df is not None and not path_df.empty:
                            self.datasets_paths[config.name] = path_df
                            logger.info(f"Successfully fetched {len(path_df)} paths for {config.name}")
                            
                        self.datasets_metadata[config.name] = metadata
                except Exception as e:
                    logger.error(f"Error fetching data for {config.name}: {e}")

    def _fetch_single_dataset(self, config: DatasetConfig, min_threshold: int) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, Dict]]:
        """Helper to fetch data for a single dataset."""
        logger.info(f"Fetching data for {config.name} (source: {config.dataset})...")
        
        # Initialize FindNeuronConnection
        fc = FindNeuronConnection(
            token=config.token,
            dataset=config.dataset,
            sourceNeurons=config.source_types,
            targetNeurons=config.target_types,
            min_synapse_num=min_threshold,
            use_cache=True, # Leverage existing cache
            showfig=False,
            folder_prefix=config.name,
            max_interlayer=config.max_interlayer
        )
        
        # Run analysis to get the connection table
        fc.InitializeNeuronInfo()
        
        if config.max_interlayer > 0:
            logger.info(f"Running path finding (max_interlayer={config.max_interlayer}) for {config.name}...")
            fc.FindAllPath()
        else:
            logger.info(f"Running direct connection search for {config.name}...")
            fc.FindDirectConnections()
        
        metadata = {
            'source_count': len(fc.source_df) if hasattr(fc, 'source_df') else 0,
            'target_count': len(fc.target_df) if hasattr(fc, 'target_df') else 0,
            'source_types_found': fc.source_df['type'].nunique() if hasattr(fc, 'source_df') and 'type' in fc.source_df.columns else 0,
            'target_types_found': fc.target_df['type'].nunique() if hasattr(fc, 'target_df') and 'type' in fc.target_df.columns else 0,
        }

        conn_df = None
        path_df = None

        if hasattr(fc, 'conn_df') and not fc.conn_df.empty:
            conn_df = fc.conn_df.copy()
            # Standardize types immediately
            if 'type_pre' in conn_df.columns:
                conn_df['std_type_pre'] = conn_df['type_pre'].apply(self.type_mapper.get_std_type)
            if 'type_post' in conn_df.columns:
                conn_df['std_type_post'] = conn_df['type_post'].apply(self.type_mapper.get_std_type)
        
        if hasattr(fc, 'path_df') and not fc.path_df.empty:
            path_df = fc.path_df.copy()
            # Standardize path strings if possible?
            # For now, just return raw path_df
            
        return conn_df, path_df, metadata

    def compare_metadata(self):
        """
        Generate a comparison table of dataset vital signs.
        """
        if not self.datasets_metadata:
            logger.warning("No metadata available.")
            return

        df = pd.DataFrame(self.datasets_metadata).T
        df.to_csv(os.path.join(self.output_dir, 'dataset_metadata_comparison.csv'))
        logger.info("Metadata comparison saved.")
        return df


    def align_datasets(self, metric: str = 'weight'):
        """
        Align data from all datasets into a single DataFrame.
        Args:
            metric: Column to use for alignment ('weight', 'connection_ratio', etc.)
        """
        if not self.datasets_data:
            logger.warning("No data to align.")
            return

        dfs = []
        for name, df in self.datasets_data.items():
            if metric not in df.columns:
                logger.warning(f"Metric '{metric}' not found in {name} data. Available: {df.columns}")
                continue

            # Group by standardized types to handle many-to-one mappings
            # For weight, sum is correct. For ratio/prob, mean is a reasonable approximation if exact post counts aren't available
            agg_func = 'sum' if metric == 'weight' else 'mean'
            
            agg_df = df.groupby(['std_type_pre', 'std_type_post'])[metric].agg(agg_func).reset_index()
            agg_df = agg_df.set_index(['std_type_pre', 'std_type_post'])
            agg_df.columns = [name] # Rename column to dataset name
            dfs.append(agg_df)
            
        # Join all dataframes
        self.aligned_data = pd.concat(dfs, axis=1, join='outer').fillna(0)
        
        # Save aligned data
        self.aligned_data.to_csv(os.path.join(self.output_dir, f'aligned_connections_{metric}.csv'))
        logger.info(f"Datasets aligned by {metric} and saved.")

    def run_sensitivity_analysis(self):
        """
        Calculate similarity metrics across thresholds for all dataset pairs.
        """
        if self.aligned_data is None:
            self.align_datasets()
            
        if self.aligned_data is None or self.aligned_data.empty:
            logger.warning("Aligned data is empty, cannot run sensitivity analysis.")
            return

        results = []
        
        dataset_names = [c.name for c in self.configs if c.name in self.aligned_data.columns]
        if len(dataset_names) < 2:
            logger.warning("Need at least 2 datasets with data for comparison.")
            return

        # Pairwise comparison for all combinations
        pairs = list(combinations(dataset_names, 2))
        
        for d1, d2 in pairs:
            logger.info(f"Running sensitivity analysis between {d1} and {d2}")
            
            for th in self.thresholds:
                # Filter data by threshold
                mask = (self.aligned_data[d1] >= th) | (self.aligned_data[d2] >= th)
                subset = self.aligned_data[mask].copy()
                
                if subset.empty:
                    results.append({
                        'dataset_pair': f"{d1}_vs_{d2}",
                        'threshold': th,
                        'jaccard': 0,
                        'pearson': 0,
                        'edge_count_d1': 0,
                        'edge_count_d2': 0,
                        'common_edges': 0
                    })
                    continue

                # Binary vectors (presence/absence)
                v1_bin = (subset[d1] >= th).astype(int)
                v2_bin = (subset[d2] >= th).astype(int)
                
                # Jaccard Index
                intersection = (v1_bin & v2_bin).sum()
                union = (v1_bin | v2_bin).sum()
                jaccard = intersection / union if union > 0 else 0
                
                # Pearson Correlation (on weights of shared edges)
                shared_mask = (subset[d1] >= th) & (subset[d2] >= th)
                if shared_mask.sum() > 1:
                    correlation = subset.loc[shared_mask, d1].corr(subset.loc[shared_mask, d2])
                else:
                    correlation = 0 # Not enough data for correlation
                
                results.append({
                    'dataset_pair': f"{d1}_vs_{d2}",
                    'threshold': th,
                    'jaccard': jaccard,
                    'pearson': correlation,
                    'edge_count_d1': (subset[d1] >= th).sum(),
                    'edge_count_d2': (subset[d2] >= th).sum(),
                    'common_edges': intersection
                })
            
        self.sensitivity_results = pd.DataFrame(results)
        self.sensitivity_results.to_csv(os.path.join(self.output_dir, 'sensitivity_analysis.csv'), index=False)
        return self.sensitivity_results

    def plot_sensitivity(self):
        """
        Plot sensitivity analysis results.
        """
        if self.sensitivity_results is None:
            self.run_sensitivity_analysis()
            
        if self.sensitivity_results is None or self.sensitivity_results.empty:
            logger.warning("No sensitivity results to plot.")
            return
            
        df = self.sensitivity_results
        
        plt.figure(figsize=(12, 6))
        
        # Get unique pairs
        pairs = df['dataset_pair'].unique()
        
        # Use a colormap if many pairs
        colors = plt.cm.tab10(np.linspace(0, 1, len(pairs)))
        
        for i, pair in enumerate(pairs):
            pair_data = df[df['dataset_pair'] == pair]
            color = colors[i]
            plt.plot(pair_data['threshold'], pair_data['jaccard'], marker='o', linestyle='-', color=color, label=f'{pair} (Jaccard)')
            plt.plot(pair_data['threshold'], pair_data['pearson'], marker='s', linestyle='--', color=color, alpha=0.7, label=f'{pair} (Pearson)')
        
        plt.xlabel('Synapse Threshold')
        plt.ylabel('Similarity Metric')
        plt.title('Comparison Sensitivity to Threshold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sensitivity_plot.png'))
        plt.close()
        logger.info(f"Sensitivity plot saved to {os.path.join(self.output_dir, 'sensitivity_plot.png')}")

    def calculate_edge_robustness(self):
        """
        Calculate robustness score for each edge.
        Score = Max threshold where edge is present in ALL datasets.
        """
        if self.aligned_data is None:
            self.align_datasets()
            
        if self.aligned_data is None or self.aligned_data.empty:
            return None
            
        dataset_names = [c.name for c in self.configs if c.name in self.aligned_data.columns]
        if not dataset_names:
            return None
            
        # Initialize robustness score with 0
        robustness = pd.Series(0, index=self.aligned_data.index, name='robustness_score')
        
        for th in self.thresholds:
            # Check if edge is present in ALL datasets at this threshold
            # We assume 'present' means weight >= threshold
            present_in_all = (self.aligned_data[dataset_names] >= th).all(axis=1)
            robustness[present_in_all] = th
            
        # Add to aligned data
        self.aligned_data['robustness_score'] = robustness
        self.aligned_data.to_csv(os.path.join(self.output_dir, 'aligned_connections_with_robustness.csv'))
        logger.info("Robustness scores calculated and saved.")
        return robustness

    def plot_scatter(self, threshold: int = 1):
        """
        Plot scatter plot of weights for a specific threshold.
        """
        if self.aligned_data is None:
            self.align_datasets()
            
        if self.aligned_data is None or self.aligned_data.empty:
            logger.warning("Aligned data is empty, cannot plot scatter.")
            return
            
        dataset_names = [c.name for c in self.configs if c.name in self.aligned_data.columns]
        if len(dataset_names) < 2:
            return
            
        d1, d2 = dataset_names[0], dataset_names[1]
        
        # Filter data
        mask = (self.aligned_data[d1] >= threshold) | (self.aligned_data[d2] >= threshold)
        subset = self.aligned_data[mask]
        
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=subset[d1], y=subset[d2], alpha=0.6)
        
        # Add diagonal line
        max_val = max(subset[d1].max(), subset[d2].max())
        plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
        
        plt.xlabel(f"{d1} Weight")
        plt.ylabel(f"{d2} Weight")
        plt.title(f"Connection Weights Comparison (Threshold >= {threshold})")
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f'scatter_plot_th{threshold}.png'))
        plt.close()
        logger.info(f"Scatter plot saved to {os.path.join(self.output_dir, f'scatter_plot_th{threshold}.png')}")
    
    def compare_paths(self):
        """
        Compare paths across datasets.
        Aligns paths by their string representation and compares weights/probabilities.
        """
        if not self.datasets_paths:
            logger.warning("No path data available to compare.")
            return

        dfs = []
        for name, df in self.datasets_paths.items():
            # Ensure path column exists
            if 'path' not in df.columns and 'path_str' not in df.columns:
                continue
            
            # Create standardized path string
            # We need to map types in the path to standardized types
            
            def standardize_path(path_val):
                if isinstance(path_val, str):
                    if '->' in path_val:
                        parts = path_val.split('->')
                    else:
                        parts = [path_val]
                elif isinstance(path_val, list):
                    parts = path_val
                else:
                    return str(path_val)
                
                std_parts = [self.type_mapper.get_std_type(str(p).strip()) for p in parts]
                return '->'.join(std_parts)

            # Use 'path' or 'path_str'
            col = 'path' if 'path' in df.columns else 'path_str'
            
            # Create a copy to avoid modifying original
            df_std = df.copy()
            df_std['std_path'] = df_std[col].apply(standardize_path)
            
            # Aggregate if multiple paths map to same std_path (e.g. many-to-one mapping)
            # For paths, maybe sum weights? Or max probability?
            # Let's use max probability for now as it represents "best path"
            
            metric = 'traversal_probability' if 'traversal_probability' in df.columns else 'weight'
            agg_func = 'max' if metric == 'traversal_probability' else 'sum'
            
            agg_df = df_std.groupby('std_path')[metric].agg(agg_func).reset_index()
            agg_df = agg_df.set_index('std_path')
            agg_df.columns = [name]
            dfs.append(agg_df)
            
        if not dfs:
            logger.warning("No valid path data found.")
            return

        self.aligned_paths = pd.concat(dfs, axis=1, join='outer').fillna(0)
        self.aligned_paths.to_csv(os.path.join(self.output_dir, 'aligned_paths.csv'))
        logger.info("Paths aligned and saved.")
        return self.aligned_paths

