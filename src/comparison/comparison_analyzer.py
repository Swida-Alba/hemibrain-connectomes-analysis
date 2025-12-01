"""
ComparisonAnalyzer - Main orchestrator for cross-dataset comparison.

This module provides the primary interface for running path analyses across
multiple datasets and comparing results.

Optimized Workflow:
    1. Create ComparisonParameters with all settings (datasets, neurons, thresholds)
    2. Create ComparisonAnalyzer with parameters
    3. Run comparison and generate reports

Example:
    >>> params = ComparisonParameters(
    ...     datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
    ...     source_neurons=['MBON14.*_R'],
    ...     target_neurons=['KCg-d.*_R', 'PPL101.*_R'],
    ...     max_interlayer=2,
    ...     thresholds=[1, 3, 5, 10, 20],
    ... )
    >>> analyzer = ComparisonAnalyzer(params)
    >>> results = analyzer.run_comparison()
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np

from .dataset_config import DatasetConfig
from .comparison_parameters import ComparisonParameters
from .label_mapper import LabelMapper
from .data_loader import DataLoader
from .metrics import ComparisonMetrics


class ComparisonAnalyzer:
    """
    Main orchestrator for cross-dataset comparison analysis.
    
    Handles:
    - Running path analysis on multiple datasets
    - Coordinating label mapping
    - Computing comparison metrics
    - Generating reports
    
    Example:
        >>> # Simple workflow - ComparisonParameters first
        >>> params = ComparisonParameters(
        ...     datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
        ...     source_neurons=['MBON14.*_R'],
        ...     target_neurons=['KCg-d.*_R'],
        ...     max_interlayer=2,
        ...     thresholds=[1, 3, 5, 10, 20],
        ...     output_folder='./comparison_output'
        ... )
        >>> 
        >>> analyzer = ComparisonAnalyzer(params)
        >>> results = analyzer.run_comparison()
        >>> report = analyzer.generate_report()
    """
    
    def __init__(
        self,
        parameters: ComparisonParameters,
        label_mapper: Optional[LabelMapper] = None,
        verbose: bool = True
    ):
        """
        Initialize ComparisonAnalyzer.
        
        Args:
            parameters: ComparisonParameters with all analysis settings
            label_mapper: Optional LabelMapper for cross-dataset standardization
            verbose: Print progress messages
        """
        self.parameters = parameters
        self.label_mapper = label_mapper
        self.verbose = verbose
        
        # Initialize components
        self.metrics = ComparisonMetrics()
        
        # Storage for results
        self.raw_results: Dict[str, Dict[int, pd.DataFrame]] = {}
        self.aligned_results: Dict[int, pd.DataFrame] = {}
        self.comparison_report: Optional[Dict] = None
        
        # Cache for expensive calculations (reused across export/visualizations)
        self._similarity_cache: Dict[int, pd.DataFrame] = {}  # threshold -> similarities
        self._output_base_printed: bool = False  # Track if base dir was printed
        
        # Resolve dataset configurations from strings
        self._dataset_configs: Dict[str, DatasetConfig] = {}
        self._resolve_dataset_configs()
        
        # Setup output directory and data loader
        if parameters.output_folder:
            self.data_loader = DataLoader(parameters.full_output_path)
            self.data_loader.ensure_directories()
        else:
            self.data_loader = None
    
    def _resolve_dataset_configs(self):
        """
        Resolve dataset strings to DatasetConfig objects.
        
        Handles both string identifiers and existing DatasetConfig objects.
        For NeuPrint datasets, shares the client to avoid repeated login.
        """
        for ds in self.parameters.datasets:
            if isinstance(ds, str):
                # Create DatasetConfig from string
                # For NeuPrint datasets, we'll set the client when needed
                config = DatasetConfig.from_string(ds)
                self._dataset_configs[ds] = config
            elif isinstance(ds, DatasetConfig):
                # Use existing config
                self._dataset_configs[ds.dataset] = ds
            else:
                raise ValueError(f"Dataset must be string or DatasetConfig, got {type(ds)}")
    
    def _get_dataset_config(self, dataset_name: str) -> DatasetConfig:
        """Get DatasetConfig for a dataset name."""
        if dataset_name in self._dataset_configs:
            return self._dataset_configs[dataset_name]
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _log(self, message: str, level: str = 'info'):
        """Print message if verbose mode enabled.
        
        Args:
            message: Message to print
            level: Log level ('info', 'warn', 'debug'). Debug messages only shown with extra verbosity.
        """
        if not self.verbose:
            return
        # Skip repetitive debug messages
        if level == 'debug':
            return
        prefix = "⚠️ " if level == 'warn' else ""
        print(f"[Comparison] {prefix}{message}")
    
    def _log_file(self, filepath: str, description: str = "Saved"):
        """Log file save with relative path (prints base dir only once).
        
        Args:
            filepath: Full file path
            description: Action description (default: "Saved")
        """
        if not self.verbose:
            return
        
        base_dir = self.parameters.full_output_path if self.parameters else None
        
        # Print base directory once at the start
        if base_dir and not self._output_base_printed:
            print(f"[Comparison] Output directory: {base_dir}")
            self._output_base_printed = True
        
        # Show only relative path from base dir
        if base_dir and filepath.startswith(base_dir):
            rel_path = os.path.relpath(filepath, base_dir)
            print(f"[Comparison] {description}: {rel_path}")
        else:
            print(f"[Comparison] {description}: {filepath}")
    
    def _generate_mode_specific_note(self) -> str:
        """Generate HTML note specific to the comparison mode used."""
        mode = getattr(self.parameters, 'comparison_mode', 'path')
        
        if mode == 'edge':
            return '''
            <div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 12px 16px; margin-top: 12px; border-radius: 0 6px 6px 0;">
                <strong>⚠️ Edge-Based Comparison Mode:</strong> This analysis uses edge-based filtering where each synapse connection 
                is evaluated independently by its weight. Edges are aggregated at the neuron type level. This mode avoids the 
                path-filtering artifacts where strong edges might appear absent due to weak intermediate edges on paths. 
                <br><br>
                <strong>Caveat 1 - Dead-ends:</strong> Edge-based comparison may include <strong>dead-end connections</strong> in the network—edges 
                that are strongly connected but don't contribute to any complete source→target path. These appear in individual 
                datasets but may not be functionally relevant to the circuit. Check the path presence matrix to identify 
                edges that form complete paths versus isolated strong connections.
                <br><br>
                <strong>Caveat 2 - Weight Mismatch:</strong> Edge weights in the <em>Edge Presence Matrix</em> represent the <strong>total 
                synapse count between all neuron pairs</strong> of the source and target types that meet the threshold. However, edge weights 
                shown in the <em>Path Presence Matrix</em> (hop weights) represent only synapses from <strong>neurons actually participating 
                in paths</strong>. The same A→B edge may show different weights: e.g., edge matrix shows 500 synapses (all type-A to type-B 
                connections), while path matrix shows 120 (only neurons on paths from source to target). This is expected behavior—path 
                weights are subsets of edge weights.
            </div>'''
        else:
            return '''
            <div style="background: #fee2e2; border-left: 4px solid #ef4444; padding: 12px 16px; margin-top: 12px; border-radius: 0 6px 6px 0;">
                <strong>⚠️ Path-Based Filtering Caveat:</strong> This analysis uses path-based filtering where edges are discovered 
                through paths from source to target neurons. <strong>Strong edges may appear absent</strong> (marked ❌) if they only 
                exist on paths with weaker intermediate edges that fall below the threshold. An edge marked as "non-existent" in one 
                dataset may actually exist but was filtered due to path context, not edge absence. Compare results across threshold 
                levels to identify such cases—if an edge appears at lower thresholds but disappears at higher ones, it may indicate 
                path-filtering artifacts rather than true biological differences.
            </div>'''
    
    # =========================================================================
    # Path Analysis
    # =========================================================================
    
    def run_path_analysis(
        self,
        dataset_name: str,
        threshold: int
    ) -> pd.DataFrame:
        """
        Run path analysis for a single dataset at a specific threshold.
        
        Args:
            dataset_name: Dataset identifier string
            threshold: Weight threshold for path finding
            
        Returns:
            DataFrame with path analysis results
        """
        # Import here to avoid circular imports
        # Use absolute import since src is on sys.path
        from coana import FindNeuronConnection
        
        self._log(f"Running analysis: \033[94m{dataset_name} @ threshold={threshold}\033[0m")
        
        # Get dataset config
        config = self._get_dataset_config(dataset_name)
        
        # Get source/target neurons from ComparisonParameters
        source_neurons = self.parameters.get_source_neurons_for_dataset(dataset_name)
        target_neurons = self.parameters.get_target_neurons_for_dataset(dataset_name)
        
        # Get max_interlayer from ComparisonParameters (shared across all datasets)
        max_interlayer = self.parameters.max_interlayer
        
        # Set up output path to redirect FNC output into comparison folder structure
        # Output goes to: {comparison_output}/dataset_data/{dataset}/minsyn_{threshold}/
        safe_dataset_name = self.parameters._sanitize_name(dataset_name)
        fnc_output_path = self.parameters.get_dataset_output_path(dataset_name, threshold)
        
        # Let FindNeuronConnection handle client creation
        # It will auto-detect client_type from dataset name and create/reuse clients as needed
        # - If dataset contains 'flywire' or 'fafb' -> uses local data
        # - Otherwise -> uses NeuPrint (creates client using dataset name and token from env var)
        fnc = FindNeuronConnection(
            sourceNeurons=source_neurons,
            targetNeurons=target_neurons,
            max_interlayer=max_interlayer,
            min_synapse_num=threshold,
            min_traversal_probability=0,  # Use 0 to match FindPath.py - default 0.001 can miss weak but important edges
            min_ratio=0,
            dataset=dataset_name,
            # Redirect output to comparison folder structure
            saveas=fnc_output_path,  # Absolute path - overrides data_folder
            verbose_mode='simple',  # Use simplified progress output for comparison runs
        )
        
        # Initialize and run analysis
        # Use FindAllPath() as specified in TODO_comparison.md:
        # "DO NOT use the FindDirectConnection() function, because the FindAllPath() 
        # function can already include direct connections as 1-hop paths"
        fnc.InitializeNeuronInfo()
        fnc.FindAllPath()
        
        # Get results - FindAllPath saves both path data and connection data
        # For comparison metrics, we need the connection data (edge-level) format:
        # - data_details/connection_info_bodyId.csv has bodyId_pre, bodyId_post, weight, etc.
        # - This is the correct format for comparison metrics
        conn_df = pd.DataFrame()
        
        if hasattr(fnc, 'allpath_folder') and fnc.allpath_folder:
            # Try to load connection data (edge-level format for metrics)
            conn_file = os.path.join(
                fnc.allpath_folder, 'data_details', 'connection_info_bodyId.csv'
            )
            
            if os.path.exists(conn_file):
                try:
                    conn_df = pd.read_csv(conn_file)
                    self._log(f"Loaded {len(conn_df)} connections from connection_info_bodyId.csv")
                except Exception as e:
                    self._log(f"Warning: Could not read connection file: {e}")
            else:
                # Fallback: try connection_type.csv (type-level)
                conn_type_file = os.path.join(
                    fnc.allpath_folder, 'data_details', 'connection_type.csv'
                )
                if os.path.exists(conn_type_file):
                    try:
                        conn_df = pd.read_csv(conn_type_file)
                        self._log(f"Loaded {len(conn_df)} connections from connection_type.csv")
                    except Exception as e:
                        self._log(f"Warning: Could not read connection type file: {e}")
        
        # Add dataset identifier
        if not conn_df.empty:
            conn_df = conn_df.copy()
            conn_df['dataset'] = dataset_name
            conn_df['threshold'] = threshold
        
        return conn_df
    
    def run_edge_analysis(
        self,
        dataset_name: str,
        threshold: int,
        base_edges: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Run edge-based analysis for a single dataset at a specific threshold.
        
        Unlike path-based analysis, this queries edges directly between neuron types
        without path context. Edges are filtered by weight threshold independently.
        
        Args:
            dataset_name: Dataset identifier string
            threshold: Weight threshold for edge filtering
            base_edges: Optional pre-fetched edges at lowest threshold (for efficiency)
            
        Returns:
            DataFrame with edge analysis results
        """
        self._log(f"Running edge analysis: \033[94m{dataset_name} @ threshold={threshold}\033[0m")
        
        # Get source/target neurons
        source_neurons = self.parameters.get_source_neurons_for_dataset(dataset_name)
        target_neurons = self.parameters.get_target_neurons_for_dataset(dataset_name)
        
        # If base_edges provided, filter from it
        if base_edges is not None and not base_edges.empty:
            filtered = base_edges[base_edges['weight'] >= threshold].copy()
            filtered['threshold'] = threshold
            self._log(f"Filtered {len(filtered)} edges from base at threshold={threshold}", 'debug')
            return filtered
        
        # Otherwise, query edges directly
        conn_df = self._query_edges_for_dataset(
            dataset_name, source_neurons, target_neurons, threshold
        )
        
        if not conn_df.empty:
            conn_df = conn_df.copy()
            conn_df['dataset'] = dataset_name
            conn_df['threshold'] = threshold
        
        return conn_df
    
    def _query_edges_for_dataset(
        self,
        dataset_name: str,
        source_neurons: List,
        target_neurons: List,
        min_weight: int = 1
    ) -> pd.DataFrame:
        """
        Query all edges between source and target neuron types.
        
        This queries edges directly without path context, capturing all connections
        between relevant neuron types regardless of path existence.
        
        Args:
            dataset_name: Dataset identifier
            source_neurons: List of source neuron types/patterns
            target_neurons: List of target neuron types/patterns
            min_weight: Minimum edge weight
            
        Returns:
            DataFrame with columns: bodyId_pre, bodyId_post, type_pre, type_post, weight
        """
        # Check if dataset is local (FlyWire/FAFB) or NeuPrint
        is_local = 'flywire' in dataset_name.lower() or 'fafb' in dataset_name.lower() or 'banc' in dataset_name.lower()
        
        if is_local:
            return self._query_edges_local(dataset_name, source_neurons, target_neurons, min_weight)
        else:
            return self._query_edges_neuprint(dataset_name, source_neurons, target_neurons, min_weight)
    
    def _query_edges_neuprint(
        self,
        dataset_name: str,
        source_neurons: List,
        target_neurons: List,
        min_weight: int = 1
    ) -> pd.DataFrame:
        """Query edges from NeuPrint database."""
        try:
            from neuprint import Client
            
            token = self.parameters.resolve_token()
            client = Client('neuprint.janelia.org', dataset=dataset_name, token=token)
            
            # Build type patterns for Cypher query
            # Handle regex patterns (convert .* to Cypher regex)
            def format_types_for_cypher(types_list):
                formatted = []
                for t in types_list:
                    if isinstance(t, str):
                        if '.*' in t or '*' in t:
                            # Convert to Cypher regex pattern
                            pattern = t.replace('.*', '.*').replace('*', '.*')
                            formatted.append(f"a.type =~ '{pattern}'")
                        else:
                            formatted.append(f"a.type = '{t}'")
                return formatted
            
            # Build source and target type conditions
            source_conditions = format_types_for_cypher(source_neurons)
            target_conditions = format_types_for_cypher(target_neurons)
            
            # Also include intermediate types (neurons that connect source to target)
            # We want edges where:
            # 1. pre is source type and post is any type (outgoing from sources)
            # 2. pre is any type and post is target type (incoming to targets)
            # 3. edges between any neurons that could be intermediates
            
            # For simplicity, query edges involving source or target types
            source_cond_str = ' OR '.join([c.replace('a.type', 'pre.type') for c in source_conditions]) if source_conditions else 'false'
            target_cond_str = ' OR '.join([c.replace('a.type', 'post.type') for c in target_conditions]) if target_conditions else 'false'
            
            # Query edges where pre is source-like OR post is target-like
            # This captures the relevant subgraph
            query = f"""
            MATCH (pre:Neuron)-[c:ConnectsTo]->(post:Neuron)
            WHERE c.weight >= {min_weight}
            AND (({source_cond_str}) OR ({target_cond_str}))
            AND pre.type IS NOT NULL AND post.type IS NOT NULL
            RETURN pre.bodyId AS bodyId_pre, pre.type AS type_pre,
                   post.bodyId AS bodyId_post, post.type AS type_post,
                   c.weight AS weight
            """
            
            result = client.fetch_custom(query)
            
            if not result.empty:
                self._log(f"Queried {len(result)} edges from NeuPrint for {dataset_name}")
            
            return result
            
        except Exception as e:
            self._log(f"Warning: Failed to query NeuPrint edges for {dataset_name}: {e}")
            return pd.DataFrame()
    
    def _query_edges_local(
        self,
        dataset_name: str,
        source_neurons: List,
        target_neurons: List,
        min_weight: int = 1
    ) -> pd.DataFrame:
        """Query edges from local dataset files."""
        import re
        
        # Load local connection data
        safe_name = self.parameters._sanitize_name(dataset_name)
        datasets_folder = self._get_datasets_folder()
        
        # Try different file patterns
        conn_files = [
            os.path.join(datasets_folder, safe_name, f'{safe_name}_connections.parquet'),
            os.path.join(datasets_folder, safe_name, f'{safe_name}_connections.csv'),
            os.path.join(datasets_folder, safe_name, 'connections.parquet'),
            os.path.join(datasets_folder, safe_name, 'connections.csv'),
        ]
        
        conn_df = None
        for conn_file in conn_files:
            if os.path.exists(conn_file):
                try:
                    if conn_file.endswith('.parquet'):
                        conn_df = pd.read_parquet(conn_file)
                    else:
                        conn_df = pd.read_csv(conn_file)
                    self._log(f"Loaded connections from {conn_file}")
                    break
                except Exception as e:
                    self._log(f"Warning: Could not load {conn_file}: {e}")
        
        if conn_df is None or conn_df.empty:
            self._log(f"Warning: No connection data found for {dataset_name}")
            return pd.DataFrame()
        
        # Standardize column names
        col_mapping = {
            'pre_pt_root_id': 'bodyId_pre',
            'post_pt_root_id': 'bodyId_post',
            'pre_type': 'type_pre',
            'post_type': 'type_post',
            'syn_count': 'weight',
            'neuropil': 'roi'
        }
        conn_df = conn_df.rename(columns={k: v for k, v in col_mapping.items() if k in conn_df.columns})
        
        # Filter by weight
        if 'weight' in conn_df.columns:
            conn_df = conn_df[conn_df['weight'] >= min_weight]
        
        # Filter by source/target types if type columns exist
        if 'type_pre' in conn_df.columns and 'type_post' in conn_df.columns:
            def matches_patterns(type_val, patterns):
                if pd.isna(type_val):
                    return False
                for p in patterns:
                    if isinstance(p, str):
                        if '.*' in p or '*' in p:
                            pattern = p.replace('.*', '.*').replace('*', '.*')
                            if re.match(pattern, str(type_val)):
                                return True
                        elif str(type_val) == p:
                            return True
                return False
            
            # Keep edges where pre matches source OR post matches target
            mask = (
                conn_df['type_pre'].apply(lambda x: matches_patterns(x, source_neurons)) |
                conn_df['type_post'].apply(lambda x: matches_patterns(x, target_neurons))
            )
            conn_df = conn_df[mask]
        
        self._log(f"Filtered to {len(conn_df)} edges for {dataset_name}")
        return conn_df
    
    def run_all_analyses(self, skip_existing: bool = True) -> Dict[str, Dict[int, pd.DataFrame]]:
        """
        Run analysis for all datasets and thresholds.
        
        Uses comparison_mode from parameters:
        - 'path': Path-based analysis using FindAllPath()
        - 'edge': Edge-based analysis querying edges directly
        
        Args:
            skip_existing: Skip if results already cached
            
        Returns:
            Nested dict {dataset_name: {threshold: DataFrame}}
        """
        mode = self.parameters.comparison_mode
        self._log(f"Starting analysis across all datasets and thresholds (mode={mode})")
        
        dataset_names = self.parameters.get_dataset_names()
        
        if mode == 'edge':
            return self._run_all_edge_analyses(skip_existing)
        else:
            return self._run_all_path_analyses(skip_existing)
    
    def _run_all_path_analyses(self, skip_existing: bool = True) -> Dict[str, Dict[int, pd.DataFrame]]:
        """Run path-based analyses for all datasets and thresholds."""
        dataset_names = self.parameters.get_dataset_names()
        
        for dataset_name in dataset_names:
            if dataset_name not in self.raw_results:
                self.raw_results[dataset_name] = {}
            
            for threshold in self.parameters.thresholds:
                # Check if already computed
                if skip_existing and threshold in self.raw_results[dataset_name]:
                    self._log(f"Skipping \033[94m{dataset_name} @ {threshold}\033[0m (already computed)")
                    continue
                
                # Check if cached on disk
                if skip_existing and self.parameters.output_folder:
                    cached = self._try_load_cached(dataset_name, threshold)
                    if cached is not None:
                        self.raw_results[dataset_name][threshold] = cached
                        continue
                
                # Run path analysis
                result_df = self.run_path_analysis(dataset_name, threshold)
                self.raw_results[dataset_name][threshold] = result_df
                
                # Save to disk
                if self.parameters.output_folder:
                    self._save_result(dataset_name, threshold, result_df)
        
        self._log(f"Completed path analysis for {len(dataset_names)} datasets")
        return self.raw_results
    
    def _run_all_edge_analyses(self, skip_existing: bool = True) -> Dict[str, Dict[int, pd.DataFrame]]:
        """
        Run edge-based analyses for all datasets and thresholds.
        
        Edge mode workflow:
        1. Run FindAllPath() at lowest threshold to get the full graph
        2. For each higher threshold, filter edges by weight AND verify paths exist
        3. Store both edge results and path data per threshold
        
        This preserves strong edges that might be filtered in path mode
        due to weak intermediate edges on the path, while also maintaining
        valid path information for each threshold level.
        """
        dataset_names = self.parameters.get_dataset_names()
        lowest_threshold = min(self.parameters.thresholds)
        
        # Store base data for each dataset (at lowest threshold)
        base_results_cache = {}
        
        for dataset_name in dataset_names:
            if dataset_name not in self.raw_results:
                self.raw_results[dataset_name] = {}
            
            # First, run path analysis at lowest threshold to get base graph
            base_result = None
            
            # Try to load from cache
            if skip_existing and self.parameters.output_folder:
                cached = self._try_load_cached(dataset_name, lowest_threshold)
                if cached is not None:
                    base_result = cached
                    self._log(f"Loaded base results from cache for \033[94m{dataset_name} at lowest threshold={lowest_threshold}\033[0m")
            
            # Run path analysis if not cached
            if base_result is None:
                base_result = self.run_path_analysis(dataset_name, lowest_threshold)
                self._log(f"Built base graph for \033[94m{dataset_name} at threshold={lowest_threshold}\033[0m: {len(base_result)} connections")
            
            base_results_cache[dataset_name] = base_result
            self.raw_results[dataset_name][lowest_threshold] = base_result
            
            # Save lowest threshold result
            if self.parameters.output_folder:
                self._save_result(dataset_name, lowest_threshold, base_result)
            
            # Now filter for each higher threshold
            for threshold in self.parameters.thresholds:
                if threshold == lowest_threshold:
                    continue  # Already processed
                
                # Check if already computed
                if skip_existing and threshold in self.raw_results[dataset_name]:
                    self._log(f"Skipping \033[94m{dataset_name} @ {threshold}\033[0m (already computed)")
                    continue
                
                # Check if cached on disk
                if skip_existing and self.parameters.output_folder:
                    cached = self._try_load_cached(dataset_name, threshold)
                    if cached is not None:
                        self.raw_results[dataset_name][threshold] = cached
                        continue
                
                # Filter edges from base result by weight
                # AND also run path analysis to find valid paths at this threshold
                if base_result is not None and not base_result.empty:
                    # First: filter edges by weight
                    if 'weight' in base_result.columns:
                        filtered_edges = base_result[base_result['weight'] >= threshold].copy()
                    else:
                        filtered_edges = base_result.copy()
                    
                    # Also run path analysis at this threshold to get valid paths
                    # This captures paths that are valid at this threshold level
                    path_result = self.run_path_analysis(dataset_name, threshold)
                    
                    # Merge: keep all filtered edges, mark which ones have valid paths
                    if not path_result.empty and not filtered_edges.empty:
                        # Create edge keys for comparison
                        if 'type_pre' in filtered_edges.columns and 'type_post' in filtered_edges.columns:
                            filtered_edges['_edge_key'] = filtered_edges['type_pre'].astype(str) + '->' + filtered_edges['type_post'].astype(str)
                            path_edges = set(path_result['type_pre'].astype(str) + '->' + path_result['type_post'].astype(str))
                            filtered_edges['has_valid_path'] = filtered_edges['_edge_key'].isin(path_edges)
                            filtered_edges = filtered_edges.drop(columns=['_edge_key'])
                        else:
                            filtered_edges['has_valid_path'] = True
                    elif filtered_edges.empty:
                        filtered_edges = path_result.copy()
                        filtered_edges['has_valid_path'] = True
                    else:
                        filtered_edges['has_valid_path'] = False
                    
                    filtered_edges['threshold'] = threshold
                    result_df = filtered_edges
                    self._log(f"Edge mode: {len(result_df)} edges at \033[94mthreshold={threshold}\033[0m for \033[94m{dataset_name}\033[0m "
                             f"({result_df['has_valid_path'].sum() if 'has_valid_path' in result_df.columns else 0} with valid paths)")
                else:
                    result_df = pd.DataFrame()
                    result_df = pd.DataFrame()
                
                self.raw_results[dataset_name][threshold] = result_df
                
                # Save to disk
                if self.parameters.output_folder:
                    self._save_result(dataset_name, threshold, result_df)
        
        self._log(f"Completed edge analysis for {len(dataset_names)} datasets")
        return self.raw_results
    
    # =========================================================================
    # Dataset Metadata Collection
    # =========================================================================
    
    def _get_datasets_folder(self) -> str:
        """Get the path to the datasets folder."""
        # Assume datasets folder is at project root level
        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(src_dir)
        return os.path.join(project_root, 'datasets')
    
    def _get_metadata_path(self, dataset_name: str) -> str:
        """Get path to the metadata file for a dataset."""
        safe_name = self.parameters._sanitize_name(dataset_name)
        datasets_folder = self._get_datasets_folder()
        return os.path.join(datasets_folder, safe_name, f'{safe_name}_metadata.json')
    
    def _load_cached_metadata(self, dataset_name: str) -> Optional[Dict]:
        """Try to load cached metadata from local file."""
        metadata_path = self._get_metadata_path(dataset_name)
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self._log(f"Warning: Failed to load cached metadata for {dataset_name}: {e}")
        return None
    
    def _save_metadata(self, dataset_name: str, metadata: Dict) -> None:
        """Save metadata to local cache file."""
        metadata_path = self._get_metadata_path(dataset_name)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            self._log(f"Saved metadata to: {metadata_path}")
        except Exception as e:
            self._log(f"Warning: Failed to save metadata for {dataset_name}: {e}")
    
    def _fetch_neuprint_metadata(self, dataset_name: str) -> Dict:
        """Fetch metadata from NeuPrint server."""
        try:
            from neuprint import Client
            
            # Parse dataset name for server connection
            token = self.parameters.resolve_token()
            client = Client('neuprint.janelia.org', dataset=dataset_name, token=token)
            
            # Get neuron counts
            q = "MATCH (n:Neuron) RETURN count(n) AS total_neurons"
            result = client.fetch_custom(q)
            total_neurons = int(result.iloc[0]['total_neurons']) if not result.empty else 0
            
            # Get typed neuron count
            q = "MATCH (n:Neuron) WHERE n.type IS NOT NULL AND n.type <> '' RETURN count(n) AS typed_neurons"
            result = client.fetch_custom(q)
            typed_neurons = int(result.iloc[0]['typed_neurons']) if not result.empty else 0
            
            # Get synapse counts
            q = "MATCH (n:Neuron) RETURN sum(n.pre) AS total_pre, sum(n.post) AS total_post"
            result = client.fetch_custom(q)
            total_pre = int(result.iloc[0]['total_pre']) if not result.empty and result.iloc[0]['total_pre'] else 0
            total_post = int(result.iloc[0]['total_post']) if not result.empty and result.iloc[0]['total_post'] else 0
            
            # Get ROI coverage
            q = "MATCH (m:Meta) RETURN m.primaryRois AS rois"
            result = client.fetch_custom(q)
            rois = result.iloc[0]['rois'] if not result.empty else []
            
            # Get neuron counts per ROI
            roi_counts = {}
            if rois:
                for roi in rois[:20]:  # Limit to first 20 ROIs for performance
                    try:
                        q = f"MATCH (n:Neuron) WHERE n.`{roi}` = true RETURN count(n) AS count"
                        result = client.fetch_custom(q)
                        roi_counts[roi] = int(result.iloc[0]['count']) if not result.empty else 0
                    except:
                        pass
            
            metadata = {
                'dataset': dataset_name,
                'source': 'neuprint',
                'fetched_at': datetime.now().isoformat(),
                'neuron_counts': {
                    'total': total_neurons,
                    'typed': typed_neurons,
                    'untyped': total_neurons - typed_neurons,
                    'type_coverage': typed_neurons / total_neurons if total_neurons > 0 else 0
                },
                'synapse_counts': {
                    'total_presynaptic': total_pre,
                    'total_postsynaptic': total_post,
                    'total': total_pre + total_post
                },
                'roi_coverage': {
                    'roi_list': rois if rois else [],
                    'roi_count': len(rois) if rois else 0,
                    'neuron_counts_per_roi': roi_counts
                },
                'coverage_notes': self._get_coverage_notes(dataset_name)
            }
            
            return metadata
            
        except Exception as e:
            self._log(f"Warning: Failed to fetch NeuPrint metadata for {dataset_name}: {e}")
            return self._create_empty_metadata(dataset_name, str(e))
    
    def _fetch_local_metadata(self, dataset_name: str) -> Dict:
        """Fetch metadata from local dataset files."""
        safe_name = self.parameters._sanitize_name(dataset_name)
        datasets_folder = self._get_datasets_folder()
        dataset_path = os.path.join(datasets_folder, safe_name)
        
        neuron_file = os.path.join(dataset_path, f'{safe_name}_allneurons_neuron_df.csv')
        neuron_parquet = os.path.join(dataset_path, f'{safe_name}_allneurons_neuron_df.parquet')
        
        # Try to load neuron data
        neuron_df = None
        if os.path.exists(neuron_parquet):
            try:
                neuron_df = pd.read_parquet(neuron_parquet)
            except:
                pass
        if neuron_df is None and os.path.exists(neuron_file):
            try:
                neuron_df = pd.read_csv(neuron_file)
            except:
                pass
        
        if neuron_df is None:
            return self._create_empty_metadata(dataset_name, "No local data found")
        
        total_neurons = len(neuron_df)
        
        # Count typed neurons
        type_col = None
        for col in ['type', 'cell_type', 'hemibrain_type', 'class']:
            if col in neuron_df.columns:
                type_col = col
                break
        
        if type_col:
            typed_neurons = neuron_df[type_col].notna().sum()
            typed_neurons = int(typed_neurons - (neuron_df[type_col] == '').sum())
        else:
            typed_neurons = 0
        
        # Synapse counts
        pre_col = 'pre' if 'pre' in neuron_df.columns else None
        post_col = 'post' if 'post' in neuron_df.columns else None
        
        total_pre = int(neuron_df[pre_col].sum()) if pre_col else 0
        total_post = int(neuron_df[post_col].sum()) if post_col else 0
        
        # Check for ROI data
        roi_file = os.path.join(dataset_path, f'{safe_name}_allneurons_roi_count_df.csv')
        roi_counts = {}
        rois = []
        
        if os.path.exists(roi_file):
            try:
                roi_df = pd.read_csv(roi_file)
                # Get ROI columns (usually all except bodyId)
                roi_cols = [c for c in roi_df.columns if c not in ['bodyId', 'Unnamed: 0']]
                rois = roi_cols
                for col in roi_cols[:20]:  # Limit
                    roi_counts[col] = int((roi_df[col] > 0).sum())
            except:
                pass
        
        metadata = {
            'dataset': dataset_name,
            'source': 'local',
            'fetched_at': datetime.now().isoformat(),
            'neuron_counts': {
                'total': total_neurons,
                'typed': typed_neurons,
                'untyped': total_neurons - typed_neurons,
                'type_coverage': typed_neurons / total_neurons if total_neurons > 0 else 0
            },
            'synapse_counts': {
                'total_presynaptic': total_pre,
                'total_postsynaptic': total_post,
                'total': total_pre + total_post
            },
            'roi_coverage': {
                'roi_list': rois,
                'roi_count': len(rois),
                'neuron_counts_per_roi': roi_counts
            },
            'coverage_notes': self._get_coverage_notes(dataset_name)
        }
        
        return metadata
    
    def _create_empty_metadata(self, dataset_name: str, error_msg: str) -> Dict:
        """Create empty metadata structure with error message."""
        return {
            'dataset': dataset_name,
            'source': 'error',
            'fetched_at': datetime.now().isoformat(),
            'error': error_msg,
            'neuron_counts': {'total': 0, 'typed': 0, 'untyped': 0, 'type_coverage': 0},
            'synapse_counts': {'total_presynaptic': 0, 'total_postsynaptic': 0, 'total': 0},
            'roi_coverage': {'roi_list': [], 'roi_count': 0, 'neuron_counts_per_roi': {}},
            'coverage_notes': self._get_coverage_notes(dataset_name)
        }
    
    def _get_coverage_notes(self, dataset_name: str) -> str:
        """Get known coverage notes for a dataset."""
        notes = {
            'hemibrain': "Central brain only. Missing: optic lobe, ventral nerve cord, subesophageal zone.",
            'male-cns': "Full male CNS including central brain, optic lobes, VNC. Mostly bilateral symmetric.",
            'flywire': "Full adult female brain (FAFB). Complete brain coverage with optic lobes.",
            'fafb': "Full adult female brain. Complete brain coverage with optic lobes.",
            'optic-lobe': "Optic lobe only. Missing: central brain, VNC.",
            'banc': "Full brain and VNC connectome."
        }
        
        dataset_lower = dataset_name.lower()
        for key, note in notes.items():
            if key in dataset_lower:
                return note
        return "Coverage information not available."
    
    def collect_dataset_metadata(self, force_refresh: bool = False) -> Dict[str, Dict]:
        """
        Collect metadata for all datasets.
        
        Metadata is cached locally in datasets/{dataset}/{dataset}_metadata.json.
        If cached file exists and force_refresh=False, uses cached data.
        
        Args:
            force_refresh: If True, fetch fresh metadata even if cached exists
            
        Returns:
            Dict mapping dataset name to metadata dict
        """
        self._log("Collecting dataset metadata...")
        
        all_metadata = {}
        
        for dataset_name in self.parameters.get_dataset_names():
            # Try cached first
            if not force_refresh:
                cached = self._load_cached_metadata(dataset_name)
                if cached:
                    self._log(f"Loaded cached metadata for {dataset_name}")
                    all_metadata[dataset_name] = cached
                    continue
            
            # Fetch fresh metadata
            self._log(f"Fetching metadata for {dataset_name}...")
            
            # Determine if local or NeuPrint dataset
            dataset_lower = dataset_name.lower()
            if 'flywire' in dataset_lower or 'fafb' in dataset_lower or 'banc' in dataset_lower:
                metadata = self._fetch_local_metadata(dataset_name)
            else:
                metadata = self._fetch_neuprint_metadata(dataset_name)
            
            # Save to cache
            self._save_metadata(dataset_name, metadata)
            all_metadata[dataset_name] = metadata
        
        # Store for later use
        self._dataset_metadata = all_metadata
        
        return all_metadata
    
    def generate_metadata_comparison_table(self) -> pd.DataFrame:
        """
        Generate a comparison table from collected metadata.
        
        Returns:
            DataFrame comparing key metrics across datasets
        """
        if not hasattr(self, '_dataset_metadata') or not self._dataset_metadata:
            self.collect_dataset_metadata()
        
        rows = []
        for dataset_name, metadata in self._dataset_metadata.items():
            nc = metadata.get('neuron_counts', {})
            sc = metadata.get('synapse_counts', {})
            rc = metadata.get('roi_coverage', {})
            
            rows.append({
                'dataset': dataset_name,
                'total_neurons': nc.get('total', 0),
                'typed_neurons': nc.get('typed', 0),
                'untyped_neurons': nc.get('untyped', 0),
                'type_coverage_pct': round(nc.get('type_coverage', 0) * 100, 2),
                'total_presynaptic': sc.get('total_presynaptic', 0),
                'total_postsynaptic': sc.get('total_postsynaptic', 0),
                'total_synapses': sc.get('total', 0),
                'roi_count': rc.get('roi_count', 0),
                'coverage_notes': metadata.get('coverage_notes', '')
            })
        
        return pd.DataFrame(rows)
    
    def _try_load_cached(self, dataset_name: str, threshold: int) -> Optional[pd.DataFrame]:
        """Try to load cached result from disk."""
        if not self.parameters.output_folder:
            return None
        
        output_dir = self.parameters.get_dataset_output_path(dataset_name, threshold)
        
        # First try our own cached connections.csv
        filepath = os.path.join(output_dir, "connections.csv")
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                if not df.empty:
                    self._log(f"Loading cached: {dataset_name} @ {threshold}", 'debug')
                    return df
            except Exception:
                pass
        
        # Also try legacy paths.csv (for backward compatibility)
        filepath = os.path.join(output_dir, "paths.csv")
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                if not df.empty:
                    self._log(f"Loading cached: {dataset_name} @ {threshold}", 'debug')
                    return df
            except Exception:
                pass
        
        # Also try to find the FindNeuronConnection output file (connection_info_bodyId.csv)
        conn_file = os.path.join(output_dir, 'data_details', 'connection_info_bodyId.csv')
        if os.path.exists(conn_file):
            try:
                df = pd.read_csv(conn_file)
                if not df.empty:
                    self._log(f"Loading cached: {dataset_name} @ {threshold}", 'debug')
                    # Add dataset info if missing
                    if 'dataset' not in df.columns:
                        df['dataset'] = dataset_name
                        df['threshold'] = threshold
                    return df
            except Exception:
                pass
        
        return None
    
    def _save_result(self, dataset_name: str, threshold: int, df: pd.DataFrame):
        """Save result to disk."""
        if not self.parameters.output_folder:
            return
        
        dirpath = self.parameters.get_dataset_output_path(dataset_name, threshold)
        os.makedirs(dirpath, exist_ok=True)
        
        # Save to connections.csv as our own cached version
        filepath = os.path.join(dirpath, "connections.csv")
        df.to_csv(filepath, index=False)
        self._log_file(filepath)
    
    # =========================================================================
    # Comparison Analysis
    # =========================================================================
    
    def run_comparison(self, skip_existing: bool = True) -> Dict[str, Any]:
        """
        Run full comparison analysis: path analysis + comparison metrics.
        
        This is the main entry point for running a complete comparison.
        
        Args:
            skip_existing: Skip if results already cached
            
        Returns:
            Dictionary with all comparison metrics and findings
        """
        # Run path analyses
        self.run_all_analyses(skip_existing=skip_existing)
        
        # Compute comparison metrics
        return self.run_comparison_analysis()
    
    def run_comparison_analysis(self) -> Dict[str, Any]:
        """
        Run full comparison analysis on results.
        
        Returns:
            Dictionary with all comparison metrics and findings
        """
        # Ensure analyses have been run
        if not self.raw_results:
            self.run_all_analyses()
        
        self._log("Computing comparison metrics")
        
        dataset_names = self.parameters.get_dataset_names()
        
        # Generate comprehensive summary
        summary = self.metrics.generate_comparison_summary(
            results=self.raw_results,
            datasets=dataset_names,
            thresholds=self.parameters.thresholds,
            label_mapper=self.label_mapper
        )
        
        # Calculate cross-threshold similarities and cache them
        similarities = self.metrics.calculate_similarity_across_thresholds(
            results=self.raw_results,
            datasets=dataset_names,
            thresholds=self.parameters.thresholds,
            label_mapper=self.label_mapper
        )
        summary['threshold_similarities'] = similarities
        
        # Cache per-threshold similarities for reuse in visualizations
        if not similarities.empty and 'threshold' in similarities.columns:
            for threshold in self.parameters.thresholds:
                thresh_sims = similarities[similarities['threshold'] == threshold]
                if not thresh_sims.empty:
                    self._similarity_cache[threshold] = thresh_sims.copy()
        
        # Store for later use
        self.comparison_report = summary
        
        return summary
    
    def get_cached_similarities(self, threshold: int) -> pd.DataFrame:
        """
        Get cached pairwise similarities at a threshold.
        
        Uses cached values if available, otherwise computes and caches.
        
        Args:
            threshold: Weight threshold
            
        Returns:
            DataFrame with pairwise similarities
        """
        if threshold in self._similarity_cache:
            return self._similarity_cache[threshold].copy()
        
        # Compute if not cached
        aligned = self.get_aligned_data(threshold)
        if aligned.empty:
            return pd.DataFrame()
        
        dataset_names = self.parameters.get_dataset_names()
        similarities = self.metrics.calculate_all_pairwise_similarities(
            aligned, dataset_names, threshold=1, include_advanced_metrics=True
        )
        
        # Cache for future use
        if not similarities.empty:
            self._similarity_cache[threshold] = similarities.copy()
        
        return similarities
    
    def get_aligned_data(self, threshold: int) -> pd.DataFrame:
        """
        Get aligned edge data at a specific threshold.
        
        Args:
            threshold: Weight threshold
            
        Returns:
            DataFrame with edges aligned across datasets
        """
        if threshold in self.aligned_results:
            return self.aligned_results[threshold]
        
        dataset_names = self.parameters.get_dataset_names()
        
        aligned = self.metrics._align_results_at_threshold(
            self.raw_results,
            dataset_names,
            threshold,
            self.label_mapper
        )
        
        self.aligned_results[threshold] = aligned
        return aligned
    
    def get_common_connections(self, threshold: int) -> pd.DataFrame:
        """
        Get connections present in all datasets at threshold.
        
        Args:
            threshold: Weight threshold
            
        Returns:
            DataFrame with common connections
        """
        aligned = self.get_aligned_data(threshold)
        dataset_names = self.parameters.get_dataset_names()
        return self.metrics.find_common_connections(aligned, dataset_names, threshold)
    
    def get_unique_connections(self, threshold: int) -> Dict[str, pd.DataFrame]:
        """
        Get connections unique to each dataset.
        
        Args:
            threshold: Weight threshold
            
        Returns:
            Dict mapping dataset name to unique connections DataFrame
        """
        aligned = self.get_aligned_data(threshold)
        dataset_names = self.parameters.get_dataset_names()
        return self.metrics.find_unique_connections(aligned, dataset_names, threshold)
    
    def get_differential_connections(self, threshold: int, fold_threshold: float = 2.0) -> pd.DataFrame:
        """
        Get connections with large weight differences.
        
        Args:
            threshold: Weight threshold for presence
            fold_threshold: Minimum fold change
            
        Returns:
            DataFrame with differential connections
        """
        aligned = self.get_aligned_data(threshold)
        dataset_names = self.parameters.get_dataset_names()
        return self.metrics.find_differential_connections(aligned, dataset_names, fold_threshold)
    
    # =========================================================================
    # Report Generation
    # =========================================================================
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate human-readable comparison report.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Report text
        """
        # Ensure comparison has been run
        if self.comparison_report is None:
            self.run_comparison_analysis()
        
        thresholds_str = ', '.join(str(t) for t in self.parameters.thresholds)
        mid_threshold = self.parameters.thresholds[len(self.parameters.thresholds) // 2]
        dataset_names = self.parameters.get_dataset_names()
        
        lines = []
        lines.append("=" * 70)
        lines.append("CROSS-DATASET COMPARISON REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append("")
        
        # Datasets
        lines.append("DATASETS:")
        lines.append("-" * 40)
        for dataset_name in dataset_names:
            lines.append(f"  • {dataset_name}")
        lines.append("")
        
        # Analysis parameters with explicit threshold info
        lines.append("ANALYSIS PARAMETERS:")
        lines.append("-" * 40)
        lines.append(f"  Source neurons: {self.parameters.source_neurons}")
        lines.append(f"  Target neurons: {self.parameters.target_neurons}")
        lines.append(f"  Max interlayer: {self.parameters.max_interlayer}")
        lines.append(f"  Comparison mode: {self.parameters.comparison_mode}")
        lines.append("")
        
        # Explicit threshold section
        lines.append("SYNAPSE CUTOFF THRESHOLDS (min_synapse_num):")
        lines.append("-" * 40)
        lines.append(f"  Thresholds analyzed: {thresholds_str}")
        lines.append("  NOTE: Results are highly sensitive to threshold choice.")
        lines.append("")
        
        # Key metrics by threshold (matches HTML summary section)
        lines.append("KEY METRICS BY THRESHOLD:")
        lines.append("-" * 70)
        lines.append(f"{'Threshold':>10} | {'Total Edges':>12} | {'Conserved':>10} | {'Edge Rate':>10} | {'Total Paths':>12} | {'Path Rate':>10}")
        lines.append("-" * 70)
        
        for threshold in self.parameters.thresholds:
            aligned = self.get_aligned_data(threshold)
            if aligned.empty:
                continue
            
            available_ds = [d for d in dataset_names if d in aligned.columns]
            total_edges = len(aligned)
            
            # Conserved edges (present in ALL datasets)
            if available_ds:
                mask_all = (aligned[available_ds] > 0).all(axis=1)
                common_edges = int(mask_all.sum())
            else:
                common_edges = 0
            
            edge_rate = (common_edges / total_edges * 100) if total_edges > 0 else 0
            
            # Path data
            try:
                path_data = self._get_path_data_for_threshold(threshold)
                if not path_data.empty:
                    total_paths = len(path_data)
                    path_mask = (path_data[available_ds] > 0).all(axis=1) if available_ds else pd.Series([False])
                    common_paths = int(path_mask.sum())
                    path_rate = (common_paths / total_paths * 100) if total_paths > 0 else 0
                else:
                    total_paths, common_paths, path_rate = 0, 0, 0
            except:
                total_paths, common_paths, path_rate = 0, 0, 0
            
            lines.append(f"{threshold:>10} | {total_edges:>12} | {common_edges:>10} | {edge_rate:>9.1f}% | {total_paths:>12} | {path_rate:>9.1f}%")
        
        lines.append("")
        
        # Edge counts per dataset per threshold
        lines.append("EDGE COUNTS PER DATASET:")
        lines.append("-" * 70)
        header = f"{'Threshold':>10}"
        for d in dataset_names:
            short_name = d.split(':')[0][:10] if ':' in d else d[:10]
            header += f" | {short_name:>12}"
        lines.append(header)
        lines.append("-" * 70)
        
        for threshold in self.parameters.thresholds:
            aligned = self.get_aligned_data(threshold)
            if aligned.empty:
                continue
            row = f"{threshold:>10}"
            for d in dataset_names:
                count = int((aligned[d] > 0).sum()) if d in aligned.columns else 0
                row += f" | {count:>12}"
            lines.append(row)
        
        lines.append("")
        
        # Pairwise similarities at ALL thresholds
        lines.append("PAIRWISE SIMILARITIES AT ALL THRESHOLD LEVELS:")
        lines.append("-" * 70)
        
        for threshold in self.parameters.thresholds:
            aligned = self.get_aligned_data(threshold)
            if aligned.empty:
                continue
            
            sim_t = self.metrics.calculate_all_pairwise_similarities(aligned, dataset_names, threshold=1)
            if sim_t.empty:
                continue
            
            lines.append(f"\n  Threshold = {threshold}:")
            for _, row in sim_t.iterrows():
                jaccard = row.get('jaccard_similarity', 0)
                svd = row.get('svd_similarity', 0)
                if pd.isna(svd):
                    svd = 0
                pearson = row.get('pearson_correlation', 0)
                if pd.isna(pearson):
                    pearson = 0
                common = row.get('common_edges', 0)
                unique_d1 = row.get('unique_to_d1', 0)
                unique_d2 = row.get('unique_to_d2', 0)
                lines.append(f"    {row['dataset_1']} vs {row['dataset_2']}:")
                lines.append(f"      Jaccard: {jaccard:.3f} | SVD: {svd:.3f} | Weight Corr: {pearson:.3f}")
                lines.append(f"      Common: {common} | Unique to D1: {unique_d1} | Unique to D2: {unique_d2}")
        
        lines.append("")
        
        # Dataset overlap summary
        lines.append("DATASET OVERLAP SUMMARY:")
        lines.append("-" * 70)
        for threshold in self.parameters.thresholds:
            aligned = self.get_aligned_data(threshold)
            if aligned.empty:
                continue
            available_ds = [d for d in dataset_names if d in aligned.columns]
            n = len(available_ds)
            
            lines.append(f"\n  Threshold = {threshold}:")
            # Edge overlap
            for i, d1 in enumerate(available_ds):
                edges_in_d1 = set(aligned.index[aligned[d1] > 0])
                for j, d2 in enumerate(available_ds):
                    if i < j:
                        edges_in_d2 = set(aligned.index[aligned[d2] > 0])
                        overlap = len(edges_in_d1 & edges_in_d2)
                        pct1 = (overlap / len(edges_in_d1) * 100) if edges_in_d1 else 0
                        pct2 = (overlap / len(edges_in_d2) * 100) if edges_in_d2 else 0
                        short1 = d1.split(':')[0][:8] if ':' in d1 else d1[:8]
                        short2 = d2.split(':')[0][:8] if ':' in d2 else d2[:8]
                        lines.append(f"    {short1} ∩ {short2}: {overlap} edges ({pct1:.0f}% of D1, {pct2:.0f}% of D2)")
        
        lines.append("")
        lines.append("=" * 70)
        lines.append("For interactive visualizations, see: comparison_report.html")
        lines.append("For connectivity profile verification, see: connectivity_profile_comparison.html")
        lines.append("=" * 70)
        
        report_text = "\n".join(lines)
        
        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            self._log(f"Report saved to: {output_path}")
        
        return report_text
    
    def export_results(self, output_dir: Optional[str] = None):
        """
        Export all results to files following TODO_comparison.md structure.
        
        Output structure:
            {output_folder}/{saveas}/
            ├── parameters.json
            ├── comparison_report.txt
            ├── comparison_report.html          # Interactive HTML report
            ├── comparison_visualizations/       # Visualizations (renamed from visualizations/)
            │   ├── path_counts.png
            │   ├── edge_heatmap.png
            │   ├── similarity_matrix.png
            │   ├── similarity_per_threshold.png
            │   ├── edge_overlap.png
            │   ├── threshold_comparison.png
            │   ├── heatmaps/                   # Interactive HTML heatmaps
            │   │   ├── edge_heatmap.html
            │   │   └── edge_heatmap_all_thresholds.html
            │   └── visualization_data/         # Data files for visualizations
            │       ├── path_counts.csv
            │       ├── edge_heatmap.csv
            │       ├── similarity_matrix.csv
            │       └── ...
            ├── dataset_data/               # Raw FindNeuronConnection outputs (created by FNC)
            │   ├── {dataset_1}/minsyn_{threshold}/
            │   └── {dataset_2}/minsyn_{threshold}/
            └── comparison_results/         # Cross-dataset comparison outputs
                ├── path_count_comparison.csv
                ├── edge_weight_comparison.csv
                ├── edge_presence_matrix_minsyn_{threshold}.csv
                ├── path_presence_matrix_minsyn_{threshold}.csv
                └── ...
        
        Args:
            output_dir: Directory to save results (defaults to parameters.full_output_path)
        """
        out_dir = output_dir or self.parameters.full_output_path
        if not out_dir:
            raise ValueError("No output directory specified")
        
        # Ensure data loader is initialized
        if self.data_loader is None:
            self.data_loader = DataLoader(out_dir)
        self.data_loader.ensure_directories()
        
        # Create comparison_results subfolder
        comparison_results_dir = os.path.join(out_dir, "comparison_results")
        os.makedirs(comparison_results_dir, exist_ok=True)
        
        # Save parameters
        params_path = os.path.join(out_dir, "parameters.json")
        with open(params_path, 'w') as f:
            import json
            json.dump(self.parameters.to_dict(), f, indent=2, default=str)
        
        # Save report
        report_path = os.path.join(out_dir, "comparison_report.txt")
        self.generate_report(report_path)
        
        # === Cross-dataset comparison results ===
        self._export_cross_dataset_comparisons(comparison_results_dir)
        
        # === Intra-dataset threshold sensitivity ===
        self._export_intra_dataset_comparisons(comparison_results_dir)
        
        # Save comparison summary as JSON
        if self.comparison_report:
            import json
            summary_path = os.path.join(comparison_results_dir, "comparison_summary.json")
            
            # Convert non-serializable items
            summary_export = {}
            for key, value in self.comparison_report.items():
                if isinstance(value, pd.DataFrame):
                    summary_export[key] = value.to_dict(orient='records')
                else:
                    summary_export[key] = value
            
            with open(summary_path, 'w') as f:
                json.dump(summary_export, f, indent=2, default=str)

        # Generate matplotlib visualizations to comparison_visualizations/ at base level
        try:
            self._generate_visualizations(out_dir)
        except Exception as e:
            self._log(f"Warning: Failed to generate visualizations: {e}")
        
        # NOTE: Connectivity profile verification is NOT run here automatically.
        # Call run_connectivity_profile_verification() separately after export_results()
        # if needed. Parameters can be set in ComparisonParameters:
        #   - verification_direction, verification_mode, verification_top_k, etc.
        # Example:
        #   analyzer.run_connectivity_profile_verification()  # Uses params from ComparisonParameters
        
        # Generate interactive HTML report at base level
        try:
            html_report_path = os.path.join(out_dir, "comparison_report.html")
            self.generate_html_report(html_report_path)
        except Exception as e:
            self._log(f"Warning: Failed to generate HTML report: {e}")
        
        self._log(f"All results exported to: {out_dir}")
        self._log(f"Note: Run run_connectivity_profile_verification() separately for profile verification.")
    
    def _export_cross_dataset_comparisons(self, comparison_results_dir: str):
        """
        Export cross-dataset comparison results at each threshold level.
        
        Compares connections across datasets at the same synapse count cutoff.
        """
        dataset_names = self.parameters.get_dataset_names()
        
        # 1. Path count comparison across datasets
        path_counts = []
        for dataset in dataset_names:
            for threshold in self.parameters.thresholds:
                df = self.raw_results.get(dataset, {}).get(threshold, pd.DataFrame())
                count = len(df) if not df.empty else 0
                total_weight = df['weight'].sum() if 'weight' in df.columns else 0
                path_counts.append({
                    'dataset': dataset,
                    'threshold': threshold,
                    'connection_count': count,
                    'total_weight': total_weight,
                    'avg_weight': total_weight / count if count > 0 else 0
                })
        
        if path_counts:
            path_count_df = pd.DataFrame(path_counts)
            path_count_df.to_csv(
                os.path.join(comparison_results_dir, "path_count_comparison.csv"),
                index=False
            )
            self._log("Saved: path_count_comparison.csv")
        
        # 2. Common and unique connections at each threshold
        # To-Do List 5 Item 2: Remove redundant files
        # - unique_to_*.csv per threshold → merged into unique_to_{dataset}.csv 
        # - common_connections_*.csv → redundant with edge_presence_matrix
        # - conserved_strong_connections_*.csv → redundant with edge_presence_matrix
        # - aligned_data/ folder → removed
        
        # Collect all motif data for unified export (To-Do List 5 Item 6)
        all_motif_data = []
        
        for threshold in self.parameters.thresholds:
            aligned = self.get_aligned_data(threshold)
            if aligned.empty:
                continue
            
            # Export presence matrices directly to comparison_results/ (no output_cutoffs subfolder)
            self._export_presence_matrix(comparison_results_dir, threshold)
            
            # Export path presence matrix (multi-hop paths)
            self._export_path_presence_matrix(comparison_results_dir, threshold)
            
            # Collect motif analysis data (for unified export)
            motif_data = self._export_motif_analysis(comparison_results_dir, threshold)
            if motif_data:
                all_motif_data.extend(motif_data)
        
        # To-Do List 5 Item 6: Save unified motif_analysis.csv with all thresholds
        if all_motif_data:
            motif_df = pd.DataFrame(all_motif_data)
            # Reorder columns: dataset, threshold first
            cols = ['dataset', 'threshold'] + [c for c in motif_df.columns if c not in ['dataset', 'threshold']]
            motif_df = motif_df[cols]
            motif_df.to_csv(
                os.path.join(comparison_results_dir, "motif_analysis.csv"),
                index=False
            )
            self._log(f"Saved: motif_analysis.csv (unified, {len(motif_df)} rows)")
        
        # 3. Edge weight comparison matrix - includes all datasets, thresholds, with presence/difference cols
        edge_weights = []
        for threshold in self.parameters.thresholds:
            aligned = self.get_aligned_data(threshold)
            if aligned.empty:
                continue
            
            available = [d for d in dataset_names if d in aligned.columns]
            
            for edge_key, row in aligned.iterrows():
                # Parse source/target from edge key
                if ' -> ' in str(edge_key):
                    parts = str(edge_key).split(' -> ')
                    source = parts[0]
                    target = parts[1] if len(parts) > 1 else ''
                else:
                    source = str(edge_key)
                    target = ''
                
                edge_data = {
                    'edge_key': edge_key,
                    'source': source,
                    'target': target,
                    'threshold': threshold
                }
                
                # Add weight per dataset
                weights = []
                for dataset in available:
                    safe_name = self.parameters._sanitize_name(dataset)
                    weight = row[dataset]
                    edge_data[f'weight_{safe_name}'] = weight
                    if weight > 0:
                        weights.append(weight)
                
                # Add computed columns
                edge_data['presence_count'] = sum(1 for d in available if row[d] > 0)
                edge_data['total_datasets'] = len(available)
                
                if len(weights) > 0:
                    edge_data['max_weight'] = max(weights)
                    edge_data['avg_weight'] = round(np.mean(weights), 2)
                    if len(weights) > 1:
                        edge_data['weight_diff'] = max(weights) - min(weights)
                        edge_data['weight_ratio'] = round(max(weights) / min(weights), 2) if min(weights) > 0 else ''
                    else:
                        edge_data['weight_diff'] = 0
                        edge_data['weight_ratio'] = 1.0
                else:
                    edge_data['max_weight'] = 0
                    edge_data['avg_weight'] = 0
                    edge_data['weight_diff'] = 0
                    edge_data['weight_ratio'] = ''
                
                edge_weights.append(edge_data)
        
        if edge_weights:
            edge_weight_df = pd.DataFrame(edge_weights)
            # Order columns logically
            col_order = ['edge_key', 'source', 'target', 'threshold', 'presence_count', 'total_datasets']
            # Add weight columns
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                weight_col = f'weight_{safe_name}'
                if weight_col in edge_weight_df.columns:
                    col_order.append(weight_col)
            col_order.extend(['max_weight', 'avg_weight', 'weight_diff', 'weight_ratio'])
            col_order = [c for c in col_order if c in edge_weight_df.columns]
            edge_weight_df = edge_weight_df[col_order]
            
            edge_weight_df.to_csv(
                os.path.join(comparison_results_dir, "edge_weight_comparison.csv"),
                index=False
            )
            self._log(f"Saved: edge_weight_comparison.csv ({len(edge_weight_df)} edges)")
        
        # 4. Top edges comparison
        self._export_top_edges_comparison(comparison_results_dir)
        
        # 5. Degree distribution analysis
        self._export_degree_distribution(comparison_results_dir)
        
        # 6. Dataset metadata comparison
        self._export_metadata_comparison(comparison_results_dir)
        
        # 7. Unified summary CSVs (merged across thresholds)
        self._export_unified_summary(comparison_results_dir)
        
        # 8. Source/target neuron counts comparison
        self._export_neuron_counts_comparison(comparison_results_dir)
    
    def _export_intra_dataset_comparisons(self, comparison_results_dir: str):
        """
        Export intra-dataset threshold sensitivity analysis.
        
        Compares how connections change across different threshold levels within each dataset.
        """
        dataset_names = self.parameters.get_dataset_names()
        
        sensitivity_data = []
        
        for dataset in dataset_names:
            prev_edges = None
            prev_threshold = None
            
            for threshold in self.parameters.thresholds:
                df = self.raw_results.get(dataset, {}).get(threshold, pd.DataFrame())
                
                if df.empty:
                    sensitivity_data.append({
                        'dataset': dataset,
                        'threshold': threshold,
                        'edge_count': 0,
                        'edges_retained_from_prev': None,
                        'retention_rate': None,
                        'edges_lost': None,
                    })
                    prev_edges = set()
                    prev_threshold = threshold
                    continue
                
                # Create edge identifiers
                if 'type_pre' in df.columns and 'type_post' in df.columns:
                    current_edges = set(zip(df['type_pre'], df['type_post']))
                elif 'bodyId_pre' in df.columns and 'bodyId_post' in df.columns:
                    current_edges = set(zip(df['bodyId_pre'], df['bodyId_post']))
                else:
                    current_edges = set(range(len(df)))
                
                edge_count = len(current_edges)
                
                if prev_edges is not None:
                    retained = len(current_edges & prev_edges)
                    lost = len(prev_edges - current_edges)
                    retention_rate = retained / len(prev_edges) if prev_edges else None
                else:
                    retained = None
                    lost = None
                    retention_rate = None
                
                sensitivity_data.append({
                    'dataset': dataset,
                    'threshold': threshold,
                    'edge_count': edge_count,
                    'edges_retained_from_prev': retained,
                    'retention_rate': retention_rate,
                    'edges_lost': lost,
                })
                
                prev_edges = current_edges
                prev_threshold = threshold
        
        if sensitivity_data:
            sensitivity_df = pd.DataFrame(sensitivity_data)
            sensitivity_df.to_csv(
                os.path.join(comparison_results_dir, "threshold_sensitivity.csv"),
                index=False
            )
            self._log("Saved: threshold_sensitivity.csv")
    
    def _export_top_edges_comparison(self, comparison_results_dir: str):
        """Export top edges comparison to CSV."""
        dataset_names = self.parameters.get_dataset_names()
        top_n = self.parameters.top_edges
        
        # Use the middle threshold for top edges analysis
        mid_threshold = self.parameters.thresholds[len(self.parameters.thresholds) // 2]
        aligned = self.get_aligned_data(mid_threshold)
        
        if aligned.empty:
            return
        
        # Get top edges per dataset
        top_edges = self.metrics.get_top_edges_per_dataset(aligned, dataset_names, top_n)
        if not top_edges.empty:
            top_edges.to_csv(
                os.path.join(comparison_results_dir, "top_edges_comparison.csv"),
                index=False
            )
            self._log("Saved: top_edges_comparison.csv")
        
        # Get overlap statistics
        overlap = self.metrics.compare_top_edges_overlap(aligned, dataset_names, top_n)
        if not overlap.empty:
            overlap.to_csv(
                os.path.join(comparison_results_dir, "top_edges_overlap.csv"),
                index=False
            )
            self._log("Saved: top_edges_overlap.csv")
    
    def _export_degree_distribution(self, comparison_results_dir: str):
        """
        Export degree distribution analysis to CSV.
        
        To-Do List 5 Item 5: Renamed files to degree_out and degree_in, 
        added threshold indication and merged all thresholds into unified files.
        """
        dataset_names = self.parameters.get_dataset_names()
        
        # Collect degree data for ALL thresholds (To-Do List 5 Item 5)
        all_out_degree = []
        all_in_degree = []
        
        for threshold in self.parameters.thresholds:
            degree_data = self.metrics.calculate_degree_distribution(
                self.raw_results, dataset_names, threshold
            )
            
            # Add threshold column to each row
            out_degree = degree_data.get('out_degree', pd.DataFrame())
            if not out_degree.empty:
                out_degree = out_degree.copy()
                out_degree['threshold'] = threshold
                all_out_degree.append(out_degree)
            
            in_degree = degree_data.get('in_degree', pd.DataFrame())
            if not in_degree.empty:
                in_degree = in_degree.copy()
                in_degree['threshold'] = threshold
                all_in_degree.append(in_degree)
        
        # Save unified out-degree data (renamed from out_degree_distribution.csv)
        if all_out_degree:
            unified_out = pd.concat(all_out_degree, ignore_index=True)
            # Reorder columns: threshold first
            cols = ['threshold'] + [c for c in unified_out.columns if c != 'threshold']
            unified_out = unified_out[cols]
            unified_out.to_csv(
                os.path.join(comparison_results_dir, "degree_out.csv"),
                index=False
            )
            self._log("Saved: degree_out.csv (unified across thresholds)")
        
        # Save unified in-degree data (renamed from in_degree_distribution.csv)
        if all_in_degree:
            unified_in = pd.concat(all_in_degree, ignore_index=True)
            # Reorder columns: threshold first
            cols = ['threshold'] + [c for c in unified_in.columns if c != 'threshold']
            unified_in = unified_in[cols]
            unified_in.to_csv(
                os.path.join(comparison_results_dir, "degree_in.csv"),
                index=False
            )
            self._log("Saved: degree_in.csv (unified across thresholds)")
        
        # Save degree statistics summary for ALL thresholds
        all_degree_stats = []
        
        for threshold in self.parameters.thresholds:
            degree_data = self.metrics.calculate_degree_distribution(
                self.raw_results, dataset_names, threshold
            )
            
            if degree_data.get('out_degree', pd.DataFrame()).empty and degree_data.get('in_degree', pd.DataFrame()).empty:
                continue
            
            degree_stats = self.metrics.calculate_degree_statistics(degree_data)
            if not degree_stats.empty:
                degree_stats['threshold'] = threshold
                all_degree_stats.append(degree_stats)
        
        if all_degree_stats:
            unified_stats = pd.concat(all_degree_stats, ignore_index=True)
            # Reorder columns: threshold first
            cols = ['threshold'] + [c for c in unified_stats.columns if c != 'threshold']
            unified_stats = unified_stats[cols]
            unified_stats.to_csv(
                os.path.join(comparison_results_dir, "degree_statistics.csv"),
                index=False
            )
            self._log("Saved: degree_statistics.csv (all thresholds)")
    
    def _export_metadata_comparison(self, comparison_results_dir: str):
        """Export dataset metadata comparison to CSV."""
        try:
            # Collect metadata (uses cache if available)
            self.collect_dataset_metadata(force_refresh=False)
            
            # Generate comparison table
            metadata_df = self.generate_metadata_comparison_table()
            if not metadata_df.empty:
                # Save to comparison_results folder
                metadata_df.to_csv(
                    os.path.join(comparison_results_dir, "dataset_metadata_comparison.csv"),
                    index=False
                )
                self._log("Saved: dataset_metadata_comparison.csv")
                
                # Also save to the main output folder
                out_dir = os.path.dirname(comparison_results_dir)
                metadata_df.to_csv(
                    os.path.join(out_dir, "dataset_metadata_comparison.csv"),
                    index=False
                )
        except Exception as e:
            self._log(f"Warning: Failed to export metadata comparison: {e}")
    
    def _export_unified_summary(self, comparison_results_dir: str):
        """
        Export unified summary CSVs that merge data across all thresholds.
        
        Creates:
        1. unified_edge_comparison.csv - All edges across all thresholds with presence/weights
        2. unified_summary.csv - High-level summary per dataset per threshold
        
        This reduces the number of output files and provides a comprehensive view.
        """
        dataset_names = self.parameters.get_dataset_names()
        
        # 1. Unified edge comparison across all thresholds
        unified_edges = []
        
        for threshold in self.parameters.thresholds:
            aligned = self.get_aligned_data(threshold)
            if aligned.empty:
                continue
            
            available = [d for d in dataset_names if d in aligned.columns]
            
            for edge_key, row in aligned.iterrows():
                if ' -> ' in str(edge_key):
                    parts = str(edge_key).split(' -> ')
                    source = parts[0]
                    target = parts[1] if len(parts) > 1 else ''
                else:
                    source = str(edge_key)
                    target = ''
                
                edge_data = {
                    'edge_key': edge_key,
                    'source': source,
                    'target': target,
                    'threshold': threshold,
                }
                
                # Add weights per dataset
                conservation_count = 0
                for dataset in available:
                    safe_name = self.parameters._sanitize_name(dataset)
                    weight = row[dataset] if dataset in row else 0
                    edge_data[f'{safe_name}_weight'] = weight
                    edge_data[f'{safe_name}_present'] = True if weight > 0 else 0
                    if weight > 0:
                        conservation_count += 1
                
                edge_data['conservation'] = f"{conservation_count}/{len(available)}"
                unified_edges.append(edge_data)
        
        if unified_edges:
            unified_df = pd.DataFrame(unified_edges)
            unified_df.to_csv(
                os.path.join(comparison_results_dir, "unified_edge_comparison.csv"),
                index=False
            )
            self._log(f"Saved: unified_edge_comparison.csv ({len(unified_df)} entries)")
        
        # 2. Unified summary per dataset per threshold
        summary_data = []
        
        for dataset in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset)
            
            for threshold in self.parameters.thresholds:
                df = self.raw_results.get(dataset, {}).get(threshold, pd.DataFrame())
                
                if df.empty:
                    summary_data.append({
                        'dataset': safe_name,
                        'threshold': threshold,
                        'total_edges': 0,
                        'total_weight': 0,
                        'mean_weight': 0,
                        'unique_sources': 0,
                        'unique_targets': 0,
                    })
                    continue
                
                # Extract source/target columns
                if 'type_pre' in df.columns and 'type_post' in df.columns:
                    sources = df['type_pre'].nunique()
                    targets = df['type_post'].nunique()
                elif 'source' in df.columns and 'target' in df.columns:
                    sources = df['source'].nunique()
                    targets = df['target'].nunique()
                else:
                    sources = 0
                    targets = 0
                
                weight_col = 'weight' if 'weight' in df.columns else None
                total_weight = df[weight_col].sum() if weight_col else len(df)
                mean_weight = df[weight_col].mean() if weight_col else 1
                
                summary_data.append({
                    'dataset': safe_name,
                    'threshold': threshold,
                    'total_edges': len(df),
                    'total_weight': round(total_weight, 2),
                    'mean_weight': round(mean_weight, 2),
                    'unique_sources': sources,
                    'unique_targets': targets,
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(
                os.path.join(comparison_results_dir, "unified_summary.csv"),
                index=False
            )
            self._log(f"Saved: unified_summary.csv ({len(summary_df)} entries)")
        
        # 3. Export unified presence matrix with all thresholds as columns
        self._export_unified_presence_matrix(comparison_results_dir)
        
        # 4. Export merged unique connections per dataset
        self._export_merged_unique_connections(comparison_results_dir)
        
        # 5. To-Do List 5 Item 1: Export unified path presence matrix with all thresholds
        self._export_unified_path_presence_matrix(comparison_results_dir)

    def _export_unified_presence_matrix(self, comparison_results_dir: str):
        """
        Export a unified edge presence matrix with all thresholds expanded horizontally.
        
        Creates edge_presence_matrix.csv with columns:
        - edge_key, source, target
        - For each dataset and threshold: presence marker (✔️/❌)
        - For each dataset and threshold: weight
        - Conservation summary
        
        This provides a single-file view of how edges are affected by threshold changes.
        """
        dataset_names = self.parameters.get_dataset_names()
        thresholds = self.parameters.thresholds
        
        # Collect all unique edges across all thresholds
        all_edges = {}
        
        for threshold in thresholds:
            aligned = self.get_aligned_data(threshold)
            if aligned.empty:
                continue
            
            available = [d for d in dataset_names if d in aligned.columns]
            
            for edge_key, row in aligned.iterrows():
                if edge_key not in all_edges:
                    # Parse source/target from edge key
                    if ' -> ' in str(edge_key):
                        parts = str(edge_key).split(' -> ')
                        source = parts[0]
                        target = parts[1] if len(parts) > 1 else ''
                    else:
                        source = str(edge_key)
                        target = ''
                    
                    all_edges[edge_key] = {
                        'edge_key': edge_key,
                        'source': source,
                        'target': target,
                    }
                
                # Add threshold-specific presence and weight for each dataset
                for dataset in available:
                    safe_name = self.parameters._sanitize_name(dataset)
                    weight = row[dataset] if dataset in row else 0
                    
                    # Presence marker: dataset_threshold (True/0 for CSV readability)
                    pres_col = f'{safe_name}_t{threshold}'
                    all_edges[edge_key][pres_col] = True if weight > 0 else 0
                    
                    # Weight column: weight_dataset_threshold
                    weight_col = f'w_{safe_name}_t{threshold}'
                    all_edges[edge_key][weight_col] = weight if weight > 0 else ''
        
        if not all_edges:
            return
        
        # Build DataFrame
        rows = list(all_edges.values())
        presence_df = pd.DataFrame(rows)
        
        # Add summary columns - count thresholds where edge is present per dataset
        for dataset in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset)
            # Count how many thresholds this edge appears in for this dataset
            presence_cols = [f'{safe_name}_t{t}' for t in thresholds if f'{safe_name}_t{t}' in presence_df.columns]
            if presence_cols:
                presence_df[f'{safe_name}_count'] = presence_df[presence_cols].apply(
                    lambda x: sum(1 for v in x if v == True), axis=1
                )
        
        # Add total conservation count (edge present in any dataset at lowest threshold)
        lowest_threshold = min(thresholds)
        available = [d for d in dataset_names if f'{self.parameters._sanitize_name(d)}_t{lowest_threshold}' in presence_df.columns]
        presence_cols = [f'{self.parameters._sanitize_name(d)}_t{lowest_threshold}' for d in available]
        if presence_cols:
            presence_df['conserved_at_lowest'] = presence_df[presence_cols].apply(
                lambda x: sum(1 for v in x if v == True), axis=1
            )
        
        # Reorder columns: edge info first, then by threshold
        col_order = ['edge_key', 'source', 'target']
        
        # Add presence columns grouped by threshold
        for threshold in thresholds:
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                pres_col = f'{safe_name}_t{threshold}'
                if pres_col in presence_df.columns:
                    col_order.append(pres_col)
        
        # Add weight columns grouped by threshold
        for threshold in thresholds:
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                weight_col = f'w_{safe_name}_t{threshold}'
                if weight_col in presence_df.columns:
                    col_order.append(weight_col)
        
        # Add summary columns
        for dataset in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset)
            count_col = f'{safe_name}_count'
            if count_col in presence_df.columns:
                col_order.append(count_col)
        
        if 'conserved_at_lowest' in presence_df.columns:
            col_order.append('conserved_at_lowest')
        
        # Filter and reorder
        col_order = [c for c in col_order if c in presence_df.columns]
        presence_df = presence_df[col_order]
        
        # Sort by conservation at lowest threshold, then alphabetically
        if 'conserved_at_lowest' in presence_df.columns:
            presence_df = presence_df.sort_values(['conserved_at_lowest', 'edge_key'], ascending=[False, True])
        
        # Save
        presence_df.to_csv(
            os.path.join(comparison_results_dir, "edge_presence_matrix.csv"),
            index=False
        )
        self._log(f"Saved: edge_presence_matrix.csv (unified, {len(presence_df)} edges, {len(thresholds)} thresholds)")

    def _export_merged_unique_connections(self, comparison_results_dir: str):
        """
        Export merged unique connections - one file per dataset with all thresholds.
        
        Instead of separate unique_to_{dataset}_minsyn_{threshold}.csv files,
        creates a single unique_to_{dataset}.csv with a threshold column.
        
        Columns include:
        - edge_key, source, target
        - threshold (synapse cutoff level)
        - weight
        - Additional context columns
        """
        dataset_names = self.parameters.get_dataset_names()
        
        for dataset in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset)
            merged_unique = []
            
            for threshold in self.parameters.thresholds:
                unique = self.get_unique_connections(threshold)
                if safe_name not in unique and dataset not in unique:
                    continue
                
                # Get unique_df - check safe_name first, then dataset name
                # Cannot use `or` with DataFrames due to ambiguity
                if safe_name in unique:
                    unique_df = unique[safe_name]
                elif dataset in unique:
                    unique_df = unique[dataset]
                else:
                    unique_df = pd.DataFrame()
                
                if unique_df.empty:
                    continue
                
                for _, row in unique_df.iterrows():
                    # Get edge info from row or index
                    edge_key = row.get('edge_key', row.name if hasattr(row, 'name') else '')
                    if not edge_key:
                        # Try to construct from source/target
                        source = row.get('source_type', row.get('source', row.get('type_pre', '')))
                        target = row.get('target_type', row.get('target', row.get('type_post', '')))
                        edge_key = f"{source} -> {target}"
                    else:
                        # Parse from edge_key
                        if ' -> ' in str(edge_key):
                            parts = str(edge_key).split(' -> ')
                            source = parts[0]
                            target = parts[1] if len(parts) > 1 else ''
                        else:
                            source = str(edge_key)
                            target = ''
                    
                    weight = row.get('weight', row.get(safe_name, row.get(dataset, 0)))
                    
                    merged_unique.append({
                        'edge_key': edge_key,
                        'source': source,
                        'target': target,
                        'threshold': threshold,
                        'weight': weight,
                        'unique_to': safe_name,
                    })
            
            if merged_unique:
                merged_df = pd.DataFrame(merged_unique)
                merged_df = merged_df.sort_values(['threshold', 'weight'], ascending=[True, False])
                merged_df.to_csv(
                    os.path.join(comparison_results_dir, f"unique_to_{safe_name}.csv"),
                    index=False
                )
                self._log(f"Saved: unique_to_{safe_name}.csv ({len(merged_df)} unique edges)")

    def _export_neuron_counts_comparison(self, comparison_results_dir: str):
        """
        Export source/target neuron counts comparison across datasets.
        
        Creates:
        1. neuron_counts_summary.csv - Total counts per dataset (source/target)
        2. neuron_counts_by_type.csv - Count per neuron type per dataset
        3. neuron_counts_by_group.csv - Count per custom group per dataset (if custom groups exist)
        
        Data is loaded from source_neurons.csv and target_neurons.csv saved by FindAllPath.
        """
        dataset_names = self.parameters.get_dataset_names()
        lowest_threshold = min(self.parameters.thresholds)
        
        # Collect neuron data from each dataset
        all_source_data = []
        all_target_data = []
        summary_data = []
        type_counts = {}  # type -> {dataset: count}
        group_counts = {}  # group -> {dataset: count}
        
        for dataset in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset)
            
            # Load from the lowest threshold output (source/target neurons are the same across thresholds)
            dataset_output_path = self.parameters.get_dataset_output_path(dataset, lowest_threshold)
            source_file = os.path.join(dataset_output_path, 'data_details', 'source_neurons.csv')
            target_file = os.path.join(dataset_output_path, 'data_details', 'target_neurons.csv')
            
            source_count = 0
            target_count = 0
            source_df = None
            target_df = None
            
            # Load source neurons
            if os.path.exists(source_file):
                try:
                    source_df = pd.read_csv(source_file, dtype={'bodyId': str})
                    source_count = len(source_df)
                    
                    # Count by type
                    if 'type' in source_df.columns:
                        for type_val in source_df['type'].dropna().unique():
                            type_cnt = len(source_df[source_df['type'] == type_val])
                            if type_val not in type_counts:
                                type_counts[type_val] = {'role': 'source'}
                            type_counts[type_val][f'{safe_name}_source'] = type_cnt
                    
                    # Count by custom group
                    if 'custom_group' in source_df.columns:
                        for group_val in source_df['custom_group'].dropna().unique():
                            group_cnt = len(source_df[source_df['custom_group'] == group_val])
                            if group_val not in group_counts:
                                group_counts[group_val] = {'role': 'source'}
                            group_counts[group_val][f'{safe_name}_source'] = group_cnt
                            
                except Exception as e:
                    self._log(f"Warning: Could not load source neurons for {dataset}: {e}")
            
            # Load target neurons
            if os.path.exists(target_file):
                try:
                    target_df = pd.read_csv(target_file, dtype={'bodyId': str})
                    target_count = len(target_df)
                    
                    # Count by type
                    if 'type' in target_df.columns:
                        for type_val in target_df['type'].dropna().unique():
                            type_cnt = len(target_df[target_df['type'] == type_val])
                            if type_val not in type_counts:
                                type_counts[type_val] = {'role': 'target'}
                            type_counts[type_val][f'{safe_name}_target'] = type_cnt
                    
                    # Count by custom group  
                    if 'custom_group' in target_df.columns:
                        for group_val in target_df['custom_group'].dropna().unique():
                            group_cnt = len(target_df[target_df['custom_group'] == group_val])
                            if group_val not in group_counts:
                                group_counts[group_val] = {'role': 'target'}
                            group_counts[group_val][f'{safe_name}_target'] = group_cnt
                            
                except Exception as e:
                    self._log(f"Warning: Could not load target neurons for {dataset}: {e}")
            
            # Summary row
            summary_data.append({
                'dataset': safe_name,
                'source_count': source_count,
                'target_count': target_count,
                'total_neurons': source_count + target_count,
                'source_types': source_df['type'].nunique() if source_df is not None and 'type' in source_df.columns else 0,
                'target_types': target_df['type'].nunique() if target_df is not None and 'type' in target_df.columns else 0,
            })
        
        # Save summary CSV
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(
                os.path.join(comparison_results_dir, "neuron_counts_summary.csv"),
                index=False
            )
            self._log(f"Saved: neuron_counts_summary.csv ({len(summary_df)} datasets)")
        
        # Save type counts CSV (presence matrix style)
        if type_counts:
            type_rows = []
            for type_val, counts in type_counts.items():
                row = {'type': type_val}
                row.update(counts)
                type_rows.append(row)
            
            type_df = pd.DataFrame(type_rows)
            # Sort by type name
            type_df = type_df.sort_values('type')
            type_df.to_csv(
                os.path.join(comparison_results_dir, "neuron_counts_by_type.csv"),
                index=False
            )
            self._log(f"Saved: neuron_counts_by_type.csv ({len(type_df)} types)")
            
            # Store for HTML report
            self._neuron_type_counts = type_df
        
        # Save group counts CSV (if any groups exist)
        if group_counts:
            group_rows = []
            for group_val, counts in group_counts.items():
                row = {'custom_group': group_val}
                row.update(counts)
                group_rows.append(row)
            
            group_df = pd.DataFrame(group_rows)
            group_df = group_df.sort_values('custom_group')
            group_df.to_csv(
                os.path.join(comparison_results_dir, "neuron_counts_by_group.csv"),
                index=False
            )
            self._log(f"Saved: neuron_counts_by_group.csv ({len(group_df)} groups)")
            
            # Store for HTML report
            self._neuron_group_counts = group_df
        
        # Store summary for HTML report
        self._neuron_counts_summary = pd.DataFrame(summary_data) if summary_data else pd.DataFrame()

    def _export_unified_path_presence_matrix(self, comparison_results_dir: str):
        """
        To-Do List 5 Item 1: Export unified path presence matrix with all thresholds horizontally.
        
        Similar to edge_presence_matrix.csv but for multi-hop paths, with columns:
        - path_key, source, target, hops, intermediates
        - For each dataset and threshold: presence marker (✔️/❌)
        - For each dataset and threshold: weight
        - For each dataset: hop weights as [w1, w2, ...]
        - Conservation summary
        """
        import ast
        
        dataset_names = self.parameters.get_dataset_names()
        thresholds = self.parameters.thresholds
        
        # Collect all paths across all thresholds
        all_paths = {}  # path_key -> {base_info, threshold_data}
        
        for threshold in thresholds:
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                
                # Find path data file
                dataset_output_path = self.parameters.get_dataset_output_path(dataset, threshold)
                
                path_files_to_try = [
                    os.path.join(dataset_output_path, f"minsyn_{threshold}_data_original_paths.csv"),
                ]
                
                if os.path.exists(dataset_output_path):
                    for f in os.listdir(dataset_output_path):
                        if f.endswith('_allpaths_type.csv'):
                            path_files_to_try.append(os.path.join(dataset_output_path, f))
                
                path_df = None
                for path_file in path_files_to_try:
                    if os.path.exists(path_file):
                        try:
                            path_df = pd.read_csv(path_file)
                            break
                        except:
                            continue
                
                if path_df is None or path_df.empty:
                    continue
                
                # Extract path information
                for _, row in path_df.iterrows():
                    path_str = str(row.get('path', row.get('path_str', '')))
                    if not path_str or path_str == 'nan':
                        continue
                    
                    # Parse path string
                    if '->' in path_str:
                        path_nodes = [n.strip() for n in path_str.split('->')]
                    elif path_str.startswith('['):
                        try:
                            path_nodes = ast.literal_eval(path_str)
                        except:
                            continue
                    else:
                        continue
                    
                    if len(path_nodes) < 2:
                        continue
                    
                    source = path_nodes[0]
                    target = path_nodes[-1]
                    intermediates = path_nodes[1:-1] if len(path_nodes) > 2 else []
                    path_key = ' → '.join(path_nodes)
                    
                    # Initialize path data
                    if path_key not in all_paths:
                        all_paths[path_key] = {
                            'path_key': path_key,
                            'source': source,
                            'target': target,
                            'hops': len(path_nodes) - 1,
                            'intermediates': ' → '.join(intermediates) if intermediates else '',
                        }
                    
                    # Add threshold-specific data
                    weight = row.get('min_weight', row.get('weight', 1))
                    if pd.isna(weight):
                        weight = 1
                    
                    # Presence and weight columns (True/0 for CSV readability)
                    pres_col = f'{safe_name}_t{threshold}'
                    weight_col = f'w_{safe_name}_t{threshold}'
                    all_paths[path_key][pres_col] = True
                    all_paths[path_key][weight_col] = float(weight)
                    
                    # Parse hop weights
                    weights_str = str(row.get('weights', row.get('hop_weights', '')))
                    if weights_str and weights_str != 'nan':
                        hop_weights_col = f'hop_{safe_name}_t{threshold}'
                        if weights_str.startswith('['):
                            all_paths[path_key][hop_weights_col] = weights_str
                        elif ',' in weights_str:
                            try:
                                hw = [float(w.strip()) for w in weights_str.split(',')]
                                all_paths[path_key][hop_weights_col] = f"[{', '.join(str(int(w)) for w in hw)}]"
                            except:
                                pass
        
        if not all_paths:
            self._log("No path data available for unified path presence matrix")
            return
        
        # Build DataFrame
        rows = list(all_paths.values())
        path_df = pd.DataFrame(rows)
        
        # Fill missing presence markers with 0 (absent)
        for threshold in thresholds:
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                pres_col = f'{safe_name}_t{threshold}'
                if pres_col in path_df.columns:
                    path_df[pres_col] = path_df[pres_col].fillna(0)
                else:
                    path_df[pres_col] = 0
        
        # Add conservation count at lowest threshold
        lowest_threshold = min(thresholds)
        presence_cols = [f'{self.parameters._sanitize_name(d)}_t{lowest_threshold}' for d in dataset_names]
        presence_cols = [c for c in presence_cols if c in path_df.columns]
        if presence_cols:
            path_df['conserved_at_lowest'] = path_df[presence_cols].apply(
                lambda x: sum(1 for v in x if v == True), axis=1
            )
        
        # Reorder columns
        col_order = ['path_key', 'source', 'target', 'hops', 'intermediates']
        
        # Add presence columns by threshold
        for threshold in thresholds:
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                pres_col = f'{safe_name}_t{threshold}'
                if pres_col in path_df.columns:
                    col_order.append(pres_col)
        
        # Add weight columns by threshold
        for threshold in thresholds:
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                weight_col = f'w_{safe_name}_t{threshold}'
                if weight_col in path_df.columns:
                    col_order.append(weight_col)
        
        # Add hop weight columns by threshold  
        for threshold in thresholds:
            for dataset in dataset_names:
                safe_name = self.parameters._sanitize_name(dataset)
                hop_col = f'hop_{safe_name}_t{threshold}'
                if hop_col in path_df.columns:
                    col_order.append(hop_col)
        
        if 'conserved_at_lowest' in path_df.columns:
            col_order.append('conserved_at_lowest')
        
        col_order = [c for c in col_order if c in path_df.columns]
        path_df = path_df[col_order]
        
        # Sort by conservation
        if 'conserved_at_lowest' in path_df.columns:
            path_df = path_df.sort_values(['conserved_at_lowest', 'path_key'], ascending=[False, True])
        
        # Save
        path_df.to_csv(
            os.path.join(comparison_results_dir, "path_presence_matrix.csv"),
            index=False
        )
        self._log(f"Saved: path_presence_matrix.csv (unified, {len(path_df)} paths, {len(thresholds)} thresholds)")

    def _export_presence_matrix(self, comparison_results_dir: str, threshold: int):
        """
        Export edge and path presence matrices showing conservation across datasets.
        
        Creates unified tables with:
        - ✔️/❌ presence markers per dataset
        - Weights per dataset
        - Conservation count (number of datasets with edge)
        - Conservation metrics (CV, max_weight)
        
        Args:
            comparison_results_dir: Directory to save output files
            threshold: Weight threshold for analysis
        """
        dataset_names = self.parameters.get_dataset_names()
        aligned = self.get_aligned_data(threshold)
        
        if aligned.empty:
            self._log(f"No aligned data for presence matrix at threshold {threshold}")
            return
        
        # Get top edges union from all datasets
        top_n = self.parameters.top_edges
        top_edges_union = set()
        
        for dataset in dataset_names:
            if dataset in aligned.columns:
                dataset_top = set(aligned.nlargest(top_n, dataset).index)
                top_edges_union.update(dataset_top)
        
        # Limit total rows to reasonable number (2x top_edges)
        max_rows = top_n * 2
        if len(top_edges_union) > max_rows:
            # Sort by max weight across all datasets
            available = [d for d in dataset_names if d in aligned.columns]
            aligned['_max_weight'] = aligned[available].max(axis=1)
            top_edges_union = set(aligned.nlargest(max_rows, '_max_weight').index)
            aligned = aligned.drop(columns=['_max_weight'])
        
        # Filter to top edges
        matrix_df = aligned.loc[aligned.index.isin(top_edges_union)].copy()
        
        if matrix_df.empty:
            return
        
        # Build presence matrix
        rows = []
        available = [d for d in dataset_names if d in matrix_df.columns]
        
        for edge_key, row in matrix_df.iterrows():
            # Parse edge key
            if ' -> ' in str(edge_key):
                parts = str(edge_key).split(' -> ')
                source_type = parts[0]
                target_type = parts[1] if len(parts) > 1 else ''
            else:
                source_type = str(edge_key)
                target_type = ''
            
            edge_data = {
                'edge_key': edge_key,
                'source_type': source_type,
                'target_type': target_type,
            }
            
            # Calculate presence and weights
            weights = []
            conservation_count = 0
            
            for dataset in available:
                safe_name = self.parameters._sanitize_name(dataset)
                weight = row[dataset]
                is_present = weight > 0
                
                # Presence marker (True/0 for CSV readability)
                edge_data[safe_name] = True if is_present else 0
                # Weight column
                edge_data[f'weight_{safe_name}'] = weight if is_present else ''
                
                if is_present:
                    conservation_count += 1
                    weights.append(weight)
            
            edge_data['conservation_count'] = conservation_count
            
            # Calculate statistics
            if weights:
                edge_data['max_weight'] = max(weights)
                edge_data['avg_weight'] = np.mean(weights)
                if len(weights) > 1:
                    edge_data['weight_cv'] = round(np.std(weights) / np.mean(weights), 3)
                else:
                    edge_data['weight_cv'] = ''
            else:
                edge_data['max_weight'] = ''
                edge_data['avg_weight'] = ''
                edge_data['weight_cv'] = ''
            
            rows.append(edge_data)
        
        if not rows:
            return
        
        # Create DataFrame and reorder columns
        presence_df = pd.DataFrame(rows)
        
        # Order columns: edge_key, source, target, conservation_count, presence markers, weights, stats
        col_order = ['edge_key', 'source_type', 'target_type', 'conservation_count']
        
        # Add presence marker columns
        for dataset in available:
            safe_name = self.parameters._sanitize_name(dataset)
            if safe_name in presence_df.columns:
                col_order.append(safe_name)
        
        # Add weight columns
        for dataset in available:
            safe_name = self.parameters._sanitize_name(dataset)
            weight_col = f'weight_{safe_name}'
            if weight_col in presence_df.columns:
                col_order.append(weight_col)
        
        # Add statistics
        col_order.extend(['max_weight', 'avg_weight', 'weight_cv'])
        
        # Ensure all columns exist
        col_order = [c for c in col_order if c in presence_df.columns]
        presence_df = presence_df[col_order]
        
        # Sort by conservation count (desc) then max weight (desc)
        presence_df = presence_df.sort_values(
            ['conservation_count', 'max_weight'],
            ascending=[False, False]
        )
        
        # Save edge presence matrix
        presence_df.to_csv(
            os.path.join(comparison_results_dir, f"edge_presence_matrix_minsyn_{threshold}.csv"),
            index=False
        )
        self._log(f"Saved: edge_presence_matrix_minsyn_{threshold}.csv ({len(presence_df)} edges)")
        
        # Also save a threshold-independent version at the middle threshold
        mid_threshold = self.parameters.thresholds[len(self.parameters.thresholds) // 2]
        if threshold == mid_threshold:
            presence_df.to_csv(
                os.path.join(comparison_results_dir, "edge_presence_matrix.csv"),
                index=False
            )
            self._log("Saved: edge_presence_matrix.csv (default)")
    
    def _export_conserved_strong_connections(self, comparison_results_dir: str, threshold: int):
        """
        Export conserved strong connections - edges present in ALL datasets 
        that are also high-weight (top percentile).
        
        Args:
            comparison_results_dir: Directory to save output files
            threshold: Weight threshold for analysis
        """
        dataset_names = self.parameters.get_dataset_names()
        aligned = self.get_aligned_data(threshold)
        
        if aligned.empty:
            return
        
        available = [d for d in dataset_names if d in aligned.columns]
        num_datasets = len(available)
        
        if num_datasets < 2:
            return
        
        # Find edges present in ALL datasets
        all_present_mask = (aligned[available] > 0).all(axis=1)
        common_edges = aligned[all_present_mask].copy()
        
        if common_edges.empty:
            return
        
        # Calculate percentile threshold for each dataset (top 10%)
        top_percentile = 0.1
        top_n = max(1, int(len(aligned) * top_percentile))
        
        # Get top edges for each dataset
        is_top_in_all = pd.Series(True, index=common_edges.index)
        ranks = {}
        
        for dataset in available:
            safe_name = self.parameters._sanitize_name(dataset)
            # Get rank within all edges
            aligned_sorted = aligned[dataset].sort_values(ascending=False)
            rank_series = pd.Series(range(1, len(aligned_sorted) + 1), index=aligned_sorted.index)
            ranks[safe_name] = rank_series
            
            # Check if in top percentile
            top_edges_set = set(aligned.nlargest(top_n, dataset).index)
            is_top_in_all &= common_edges.index.isin(top_edges_set)
        
        # Filter to conserved strong connections
        conserved_strong = common_edges[is_top_in_all].copy()
        
        if conserved_strong.empty:
            self._log(f"No conserved strong connections at threshold {threshold}")
            return
        
        # Build output dataframe
        rows = []
        for edge_key, row in conserved_strong.iterrows():
            if ' -> ' in str(edge_key):
                parts = str(edge_key).split(' -> ')
                source_type = parts[0]
                target_type = parts[1] if len(parts) > 1 else ''
            else:
                source_type = str(edge_key)
                target_type = ''
            
            edge_data = {
                'edge_key': edge_key,
                'source_type': source_type,
                'target_type': target_type,
            }
            
            weights = []
            for dataset in available:
                safe_name = self.parameters._sanitize_name(dataset)
                weight = row[dataset]
                edge_data[f'rank_{safe_name}'] = int(ranks[safe_name].get(edge_key, 0))
                edge_data[f'weight_{safe_name}'] = weight
                weights.append(weight)
            
            # Calculate conservation score
            presence_ratio = 1.0  # All present by definition
            if len(weights) > 1:
                weight_cv = np.std(weights) / np.mean(weights)
                weight_consistency = max(0, 1 - weight_cv)
            else:
                weight_consistency = 1.0
            
            edge_data['conservation_score'] = round(
                (presence_ratio * 0.5) + (weight_consistency * 0.5), 3
            )
            edge_data['avg_weight'] = round(np.mean(weights), 2)
            edge_data['weight_cv'] = round(weight_cv, 3) if len(weights) > 1 else ''
            
            rows.append(edge_data)
        
        if not rows:
            return
        
        conserved_df = pd.DataFrame(rows)
        conserved_df = conserved_df.sort_values('conservation_score', ascending=False)
        
        # Save
        conserved_df.to_csv(
            os.path.join(comparison_results_dir, f"conserved_strong_connections_minsyn_{threshold}.csv"),
            index=False
        )
        self._log(f"Saved: conserved_strong_connections_minsyn_{threshold}.csv ({len(conserved_df)} edges)")
        
        # Also save default version at middle threshold
        mid_threshold = self.parameters.thresholds[len(self.parameters.thresholds) // 2]
        if threshold == mid_threshold:
            conserved_df.to_csv(
                os.path.join(comparison_results_dir, "conserved_strong_connections.csv"),
                index=False
            )
            self._log("Saved: conserved_strong_connections.csv (default)")
    
    def _export_path_presence_matrix(self, comparison_results_dir: str, threshold: int):
        """
        Export path presence matrix showing multi-hop path conservation across datasets.
        
        Similar to edge presence matrix but for source → intermediate → target paths.
        Creates unified tables with:
        - ✔️/❌ presence markers per dataset
        - Path structure (source → inter1 → inter2 → target)
        - Hop weights as [w1, w2, ...] for each dataset
        - Conservation metrics
        
        Args:
            comparison_results_dir: Directory to save output files
            threshold: Weight threshold for analysis
        """
        import ast
        
        dataset_names = self.parameters.get_dataset_names()
        
        # Collect paths from path CSV files (not connection data)
        path_data = {}  # path_key -> {dataset: True/False}
        path_details = {}  # path_key -> {source, intermediates, target, weights, hop_weights}
        
        for dataset in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset)
            
            # Find path data file: minsyn_X_data_original_paths.csv or {source}_to_{target}_allpaths_type.csv
            dataset_output_path = self.parameters.get_dataset_output_path(dataset, threshold)
            
            # Try multiple path file patterns
            path_files_to_try = [
                os.path.join(dataset_output_path, f"minsyn_{threshold}_data_original_paths.csv"),
            ]
            
            # Also check for source_to_target_allpaths_type.csv pattern
            if os.path.exists(dataset_output_path):
                for f in os.listdir(dataset_output_path):
                    if f.endswith('_allpaths_type.csv'):
                        path_files_to_try.append(os.path.join(dataset_output_path, f))
            
            path_df = None
            for path_file in path_files_to_try:
                if os.path.exists(path_file):
                    try:
                        path_df = pd.read_csv(path_file)
                        self._log(f"Loaded path data from {os.path.basename(path_file)} for {dataset}")
                        break
                    except Exception as e:
                        self._log(f"Warning: Could not read {path_file}: {e}")
            
            if path_df is None or path_df.empty:
                self._log(f"No path data found for {dataset} at threshold {threshold}")
                continue
            
            # Extract path information from DataFrame
            # Expected columns: path_str, path, weights, min_weight, length, etc.
            for _, row in path_df.iterrows():
                # Get path string (e.g., "aMe12->KCg-d->PPL101")
                path_str = str(row.get('path', row.get('path_str', '')))
                if not path_str or path_str == 'nan':
                    continue
                
                # Parse path string to extract nodes
                # Handle both "A->B->C" and "['A', 'B', 'C']" formats
                if '->' in path_str:
                    path_nodes = [n.strip() for n in path_str.split('->')]
                elif path_str.startswith('['):
                    # Parse list format like "['aMe12', 'KCg-d', 'PPL101']"
                    try:
                        path_nodes = ast.literal_eval(path_str)
                    except:
                        continue
                else:
                    continue
                
                if len(path_nodes) < 2:
                    continue
                
                source = path_nodes[0]
                target = path_nodes[-1]
                intermediates = path_nodes[1:-1] if len(path_nodes) > 2 else []
                
                # Build path key using arrow notation
                path_key = ' → '.join(path_nodes)
                
                # Initialize path data if not exists
                if path_key not in path_data:
                    path_data[path_key] = {}
                    path_details[path_key] = {
                        'source': source,
                        'target': target,
                        'intermediates': intermediates,
                        'weights': {},
                        'hop_weights': {}  # Store individual hop weights
                    }
                
                # Mark as present
                path_data[path_key][safe_name] = True
                
                # Parse hop weights from weights column (e.g., "[10, 5]" or "10,5")
                hop_weights_list = []
                weights_str = str(row.get('weights', row.get('hop_weights', '')))
                if weights_str and weights_str != 'nan':
                    if weights_str.startswith('['):
                        try:
                            hop_weights_list = ast.literal_eval(weights_str)
                        except:
                            pass
                    elif ',' in weights_str:
                        try:
                            hop_weights_list = [float(w.strip()) for w in weights_str.split(',')]
                        except:
                            pass
                
                # Store hop weights
                if hop_weights_list:
                    path_details[path_key]['hop_weights'][safe_name] = hop_weights_list
                
                # Record min weight as the path weight
                weight = row.get('min_weight', row.get('weight', 1))
                if pd.isna(weight):
                    if hop_weights_list:
                        weight = min(hop_weights_list)
                    else:
                        weight = 1
                
                if safe_name not in path_details[path_key]['weights']:
                    path_details[path_key]['weights'][safe_name] = []
                path_details[path_key]['weights'][safe_name].append(float(weight))
        
        if not path_data:
            self._log(f"No path data for path presence matrix at threshold {threshold}")
            return
        
        # Build presence matrix rows
        rows = []
        available_datasets = [self.parameters._sanitize_name(d) for d in dataset_names]
        
        for path_key, presence in path_data.items():
            details = path_details[path_key]
            
            row_data = {
                'path_key': path_key,
                'source': details['source'],
                'target': details['target'],
                'hops': len(details['intermediates']) + 1,
                'intermediates': ' → '.join(details['intermediates']) if details['intermediates'] else '',
            }
            
            # Add presence markers, weights, and hop weights
            conservation_count = 0
            all_weights = []
            
            for safe_name in available_datasets:
                is_present = presence.get(safe_name, False)
                row_data[safe_name] = True if is_present else 0  # True/0 for CSV readability
                
                if is_present:
                    conservation_count += 1
                    weights = details['weights'].get(safe_name, [])
                    hop_weights = details['hop_weights'].get(safe_name, [])
                    
                    if weights:
                        avg_weight = np.mean(weights)
                        row_data[f'weight_{safe_name}'] = round(avg_weight, 2)
                        all_weights.extend(weights)
                    else:
                        row_data[f'weight_{safe_name}'] = ''
                    
                    # Add hop weights as formatted string -w1-w2- with dashes
                    if hop_weights:
                        row_data[f'hop_weights_{safe_name}'] = '-' + '-'.join(str(int(w)) for w in hop_weights) + '-'
                    else:
                        row_data[f'hop_weights_{safe_name}'] = ''
                else:
                    row_data[f'weight_{safe_name}'] = ''
                    row_data[f'hop_weights_{safe_name}'] = ''
            
            row_data['conservation_count'] = conservation_count
            
            # Calculate statistics
            if all_weights:
                row_data['max_weight'] = round(max(all_weights), 2)
                row_data['avg_weight'] = round(np.mean(all_weights), 2)
                if len(all_weights) > 1:
                    row_data['weight_cv'] = round(np.std(all_weights) / np.mean(all_weights), 3)
                else:
                    row_data['weight_cv'] = ''
            else:
                row_data['max_weight'] = ''
                row_data['avg_weight'] = ''
                row_data['weight_cv'] = ''
            
            rows.append(row_data)
        
        if not rows:
            return
        
        # Create DataFrame and sort
        path_presence_df = pd.DataFrame(rows)
        
        # Reorder columns for better readability
        col_order = ['path_key', 'source', 'target', 'hops', 'intermediates', 'conservation_count']
        
        # Add presence columns
        for safe_name in available_datasets:
            if safe_name in path_presence_df.columns:
                col_order.append(safe_name)
        
        # Add weight columns  
        for safe_name in available_datasets:
            weight_col = f'weight_{safe_name}'
            if weight_col in path_presence_df.columns:
                col_order.append(weight_col)
        
        # Add hop weights columns
        for safe_name in available_datasets:
            hop_col = f'hop_weights_{safe_name}'
            if hop_col in path_presence_df.columns:
                col_order.append(hop_col)
        
        # Add statistics
        col_order.extend(['max_weight', 'avg_weight', 'weight_cv'])
        col_order = [c for c in col_order if c in path_presence_df.columns]
        path_presence_df = path_presence_df[col_order]
        
        path_presence_df = path_presence_df.sort_values(
            ['conservation_count', 'max_weight'],
            ascending=[False, False]
        )
        
        # Limit to top paths
        max_paths = self.parameters.top_edges * 2
        if len(path_presence_df) > max_paths:
            path_presence_df = path_presence_df.head(max_paths)
        
        # Save
        path_presence_df.to_csv(
            os.path.join(comparison_results_dir, f"path_presence_matrix_minsyn_{threshold}.csv"),
            index=False
        )
        self._log(f"Saved: path_presence_matrix_minsyn_{threshold}.csv ({len(path_presence_df)} paths)")
        
        # Save default version at middle threshold
        mid_threshold = self.parameters.thresholds[len(self.parameters.thresholds) // 2]
        if threshold == mid_threshold:
            path_presence_df.to_csv(
                os.path.join(comparison_results_dir, "path_presence_matrix.csv"),
                index=False
            )
            self._log("Saved: path_presence_matrix.csv (default)")
    
    def _export_motif_analysis(self, comparison_results_dir: str, threshold: int):
        """
        Export network motif analysis for each dataset.
        
        Detects and compares common network motifs:
        - Feedforward loops (A→B→C, A→C)
        - Feedback loops (A→B→A)
        - Fan-in patterns (multiple inputs to one node)
        - Fan-out patterns (one node with multiple outputs)
        - Reciprocal connections
        
        Args:
            comparison_results_dir: Directory to save output files
            threshold: Weight threshold for analysis
        """
        dataset_names = self.parameters.get_dataset_names()
        motif_data = []
        
        for dataset in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset)
            
            # Build graph from connection data
            df = self.raw_results.get(dataset, {}).get(threshold, pd.DataFrame())
            if df.empty:
                continue
            
            # Extract edges
            edges = {}  # (source, target) -> weight
            nodes = set()
            
            for _, row in df.iterrows():
                source = str(row.get('type_pre', row.get('source', '')))
                target = str(row.get('type_post', row.get('target', '')))
                weight = row.get('weight', 1)
                
                if source and target and source != 'nan' and target != 'nan':
                    edges[(source, target)] = weight
                    nodes.add(source)
                    nodes.add(target)
            
            # Calculate motif counts
            feedforward_loops = 0
            feedback_loops = 0
            reciprocal_connections = 0
            
            # Fan-in/fan-out analysis
            in_degree = {}
            out_degree = {}
            
            for (s, t) in edges:
                out_degree[s] = out_degree.get(s, 0) + 1
                in_degree[t] = in_degree.get(t, 0) + 1
            
            # Find reciprocal connections
            for (s, t) in edges:
                if (t, s) in edges:
                    reciprocal_connections += 1
            reciprocal_connections //= 2  # Count each pair once
            
            # Find feedforward loops (A→B→C where A→C also exists)
            for a in nodes:
                # Get B nodes that A connects to
                b_nodes = [t for (s, t) in edges if s == a]
                for b in b_nodes:
                    # Get C nodes that B connects to
                    c_nodes = [t for (s, t) in edges if s == b]
                    for c in c_nodes:
                        # Check if A→C exists (feedforward)
                        if (a, c) in edges and c != a:
                            feedforward_loops += 1
            
            # Find feedback loops (A→B→A cycles)
            for (a, b) in edges:
                if (b, a) in edges:
                    feedback_loops += 1
            feedback_loops //= 2  # Count each pair once
            
            # Calculate hub metrics
            max_out_degree = max(out_degree.values()) if out_degree else 0
            max_in_degree = max(in_degree.values()) if in_degree else 0
            avg_out_degree = np.mean(list(out_degree.values())) if out_degree else 0
            avg_in_degree = np.mean(list(in_degree.values())) if in_degree else 0
            
            # Find fan-in/fan-out hubs
            fan_out_hubs = [n for n, d in out_degree.items() if d >= 3]  # Nodes with 3+ outputs
            fan_in_hubs = [n for n, d in in_degree.items() if d >= 3]  # Nodes with 3+ inputs
            
            motif_data.append({
                'dataset': safe_name,
                'threshold': threshold,  # To-Do List 5 Item 6: Add threshold column
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'feedforward_loops': feedforward_loops,
                'feedback_loops': feedback_loops,
                'reciprocal_connections': reciprocal_connections,
                'fan_out_hubs': len(fan_out_hubs),
                'fan_in_hubs': len(fan_in_hubs),
                'max_out_degree': max_out_degree,
                'max_in_degree': max_in_degree,
                'avg_out_degree': round(avg_out_degree, 2),
                'avg_in_degree': round(avg_in_degree, 2),
                'density': round(len(edges) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0, 4),
            })
        
        if not motif_data:
            self._log(f"No motif data available at threshold {threshold}")
            return motif_data  # Return for later merging
        
        return motif_data  # Return for merging in export_cross_dataset_comparisons

    def _generate_visualizations(self, output_dir: str):
        """
        Generate and save all matplotlib visualizations.
        
        Creates visualization PNG files in the comparison_visualizations/ subfolder at base level.
        Also generates interactive HTML heatmaps using VisualizePath.
        Uses cached similarities to avoid redundant calculations.
        
        Args:
            output_dir: Base output directory
        """
        try:
            from .visualizations import ComparisonVisualizer
        except ImportError:
            self._log("Warning: ComparisonVisualizer not available, skipping visualizations")
            return
        
        # Save visualizations to comparison_visualizations/ at base level (not inside comparison_results/)
        vis_dir = os.path.join(output_dir, "comparison_visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        mid_threshold = self.parameters.thresholds[len(self.parameters.thresholds) // 2]
        aligned = self.get_aligned_data(mid_threshold)
        dataset_names = self.parameters.get_dataset_names()
        
        # Get pairwise similarities if available
        pairwise_sim = pd.DataFrame()
        if self.comparison_report and 'pairwise_similarities' in self.comparison_report:
            pairwise_sim = self.comparison_report['pairwise_similarities']
            if pairwise_sim is None:
                pairwise_sim = pd.DataFrame()
        
        try:
            visualizer = ComparisonVisualizer(verbose=self.verbose)
            
            # Build nickname map from parameters
            dataset_names = self.parameters.get_dataset_names()
            nicknames = self.parameters.get_dataset_nicknames()
            nickname_map = dict(zip(dataset_names, nicknames))
            
            # Generate all standard plots, passing cached similarity function
            visualizer.save_all_plots(
                results=self.raw_results,
                aligned_data=aligned,
                similarities=pairwise_sim,
                output_dir=vis_dir,
                thresholds=self.parameters.thresholds,
                align_func=self.get_aligned_data,  # Pass function to get aligned data at any threshold
                similarity_func=self.get_cached_similarities,  # Pass cached similarity function
                current_threshold=mid_threshold,
                path_data_func=self._get_path_data_for_threshold,
                ratio_data_func=self._get_ratio_data_for_threshold,
                prob_data_func=self._get_prob_data_for_threshold,
                output_base_path=self.parameters.full_output_path,
                nickname_map=nickname_map
            )
            
            self._log_file(vis_dir, "Saved visualizations")
        except Exception as e:
            self._log(f"Warning: Failed to generate some visualizations: {e}")
        
        # Generate VisualizePath interactive heatmaps (no separate network files)
        self._generate_vispath_visualizations(vis_dir)
    
    def _get_path_data_for_threshold(self, threshold: int) -> pd.DataFrame:
        """
        Get path min_weight data aligned across all datasets for a given threshold.
        
        Reads from dataset_data/{dataset}/minsyn_{threshold}/*_original_paths.csv
        or *_allpaths_type.csv and extracts min_weight values, aligning paths across datasets.
        
        Args:
            threshold: The threshold level
            
        Returns:
            DataFrame with path index and dataset columns containing min_weight
        """
        dataset_names = self.parameters.get_dataset_names()
        all_path_data = {}
        
        for dataset_name in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset_name)
            dataset_output_path = os.path.join(
                self.parameters.full_output_path,
                'dataset_data',
                safe_name,
                f'minsyn_{threshold}'
            )
            
            # Try multiple path file patterns
            path_files_to_try = [
                os.path.join(dataset_output_path, f'minsyn_{threshold}_data_original_paths.csv'),
            ]
            
            # Also check for source_to_target_allpaths_type.csv pattern
            if os.path.exists(dataset_output_path):
                for f in os.listdir(dataset_output_path):
                    if f.endswith('_allpaths_type.csv'):
                        path_files_to_try.append(os.path.join(dataset_output_path, f))
            
            # Try to read from available files
            df = None
            for path_file in path_files_to_try:
                if os.path.exists(path_file):
                    try:
                        df = pd.read_csv(path_file)
                        break
                    except Exception as e:
                        self._log(f"Warning: Could not read {path_file}: {e}")
            
            if df is not None and 'path' in df.columns and 'min_weight' in df.columns:
                for _, row in df.iterrows():
                    path_key = row['path']
                    min_weight = row['min_weight']
                    if path_key not in all_path_data:
                        all_path_data[path_key] = {}
                    all_path_data[path_key][dataset_name] = min_weight
        
        if not all_path_data:
            return pd.DataFrame()
        
        return pd.DataFrame(all_path_data).T.fillna(0)
    
    def _get_path_hop_weights_for_threshold(self, threshold: int) -> Dict[str, Dict[str, List[float]]]:
        """
        Get hop weights for each path across all datasets for a given threshold.
        
        Returns:
            Dict mapping path_key -> {safe_dataset_name: [hop_weight1, hop_weight2, ...]}
        """
        import ast
        dataset_names = self.parameters.get_dataset_names()
        all_hop_weights = {}  # path_key -> {safe_name: [weights]}
        
        for dataset_name in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset_name)
            dataset_output_path = os.path.join(
                self.parameters.full_output_path,
                'dataset_data',
                safe_name,
                f'minsyn_{threshold}'
            )
            
            # Try multiple path file patterns
            path_files_to_try = [
                os.path.join(dataset_output_path, f'minsyn_{threshold}_data_original_paths.csv'),
            ]
            
            # Also check for source_to_target_allpaths_type.csv pattern
            if os.path.exists(dataset_output_path):
                for f in os.listdir(dataset_output_path):
                    if f.endswith('_allpaths_type.csv'):
                        path_files_to_try.append(os.path.join(dataset_output_path, f))
            
            # Try to read from available files
            df = None
            for path_file in path_files_to_try:
                if os.path.exists(path_file):
                    try:
                        df = pd.read_csv(path_file)
                        break
                    except Exception as e:
                        self._log(f"Warning: Could not read {path_file}: {e}")
            
            if df is not None and 'path' in df.columns and 'weights' in df.columns:
                for _, row in df.iterrows():
                    path_key = row['path']
                    weights_str = str(row.get('weights', ''))
                    
                    # Parse weights column
                    hop_weights = []
                    if weights_str and weights_str != 'nan':
                        if weights_str.startswith('['):
                            try:
                                hop_weights = ast.literal_eval(weights_str)
                            except:
                                pass
                        elif ',' in weights_str:
                            try:
                                hop_weights = [float(w.strip()) for w in weights_str.split(',')]
                            except:
                                pass
                    
                    if hop_weights:
                        if path_key not in all_hop_weights:
                            all_hop_weights[path_key] = {}
                        all_hop_weights[path_key][safe_name] = hop_weights
        
        return all_hop_weights
    
    def _get_ratio_data_for_threshold(self, threshold: int) -> pd.DataFrame:
        """
        Get min_ratio data aligned across all datasets for a given threshold.
        
        Reads from dataset_data/{dataset}/minsyn_{threshold}/*_original_paths.csv
        or *_allpaths_type.csv and extracts min_ratio values.
        
        Args:
            threshold: The threshold level
            
        Returns:
            DataFrame with path index and dataset columns containing min_ratio
        """
        dataset_names = self.parameters.get_dataset_names()
        all_ratio_data = {}
        
        for dataset_name in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset_name)
            dataset_output_path = os.path.join(
                self.parameters.full_output_path,
                'dataset_data',
                safe_name,
                f'minsyn_{threshold}'
            )
            
            # Try multiple path file patterns
            path_files_to_try = [
                os.path.join(dataset_output_path, f'minsyn_{threshold}_data_original_paths.csv'),
            ]
            
            # Also check for source_to_target_allpaths_type.csv pattern
            if os.path.exists(dataset_output_path):
                for f in os.listdir(dataset_output_path):
                    if f.endswith('_allpaths_type.csv'):
                        path_files_to_try.append(os.path.join(dataset_output_path, f))
            
            # Try to read from available files
            df = None
            for path_file in path_files_to_try:
                if os.path.exists(path_file):
                    try:
                        df = pd.read_csv(path_file)
                        break
                    except Exception as e:
                        self._log(f"Warning: Could not read {path_file}: {e}")
            
            if df is not None and 'path' in df.columns and 'min_ratio' in df.columns:
                for _, row in df.iterrows():
                    path_key = row['path']
                    min_ratio = row['min_ratio']
                    if path_key not in all_ratio_data:
                        all_ratio_data[path_key] = {}
                    all_ratio_data[path_key][dataset_name] = min_ratio
        
        if not all_ratio_data:
            return pd.DataFrame()
        
        return pd.DataFrame(all_ratio_data).T.fillna(0)
    
    def _get_prob_data_for_threshold(self, threshold: int) -> pd.DataFrame:
        """
        Get traversal probability (path_prob) data aligned across datasets for a threshold.
        
        Reads from dataset_data/{dataset}/minsyn_{threshold}/*_original_paths.csv
        or *_allpaths_type.csv and extracts path_prob values.
        
        Args:
            threshold: The threshold level
            
        Returns:
            DataFrame with path index and dataset columns containing path_prob
        """
        dataset_names = self.parameters.get_dataset_names()
        all_prob_data = {}
        
        for dataset_name in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset_name)
            dataset_output_path = os.path.join(
                self.parameters.full_output_path,
                'dataset_data',
                safe_name,
                f'minsyn_{threshold}'
            )
            
            # Try multiple path file patterns
            path_files_to_try = [
                os.path.join(dataset_output_path, f'minsyn_{threshold}_data_original_paths.csv'),
            ]
            
            # Also check for source_to_target_allpaths_type.csv pattern
            if os.path.exists(dataset_output_path):
                for f in os.listdir(dataset_output_path):
                    if f.endswith('_allpaths_type.csv'):
                        path_files_to_try.append(os.path.join(dataset_output_path, f))
            
            # Try to read from available files
            df = None
            for path_file in path_files_to_try:
                if os.path.exists(path_file):
                    try:
                        df = pd.read_csv(path_file)
                        break
                    except Exception as e:
                        self._log(f"Warning: Could not read {path_file}: {e}")
            
            if df is not None and 'path' in df.columns and 'path_prob' in df.columns:
                for _, row in df.iterrows():
                    path_key = row['path']
                    path_prob = row['path_prob']
                    if path_key not in all_prob_data:
                        all_prob_data[path_key] = {}
                    all_prob_data[path_key][dataset_name] = path_prob
        
        if not all_prob_data:
            return pd.DataFrame()
        
        return pd.DataFrame(all_prob_data).T.fillna(0)

    def _generate_vispath_visualizations(self, vis_dir: str):
        """
        Generate interactive HTML visualizations using VisualizePath.
        
        Note: Interactive heatmaps have been removed (redundant with PNG visualizations
        and comparison_report.html). Network visualizations are embedded directly
        in comparison_report.html using Cytoscape.js.
        
        This method is kept for future extension but currently is a no-op.
        
        Args:
            vis_dir: Directory to save visualization files
        """
        # Note: _generate_vispath_heatmaps and heatmaps/ folder removed as redundant
        # All heatmap needs are served by PNG visualizations in visualizations/ folder
        pass

    def _generate_combined_networks(self, vis_dir: str, dataset_names: List[str], 
                                     VisualizePath):
        """
        Generate combined network visualizations.
        
        Creates:
        - network_all_datasets_minsyn_{threshold}.html - All datasets side-by-side at each cutoff
        - network_{dataset}_all_thresholds.html - All cutoffs for each dataset
        
        Args:
            vis_dir: Directory to save network files
            dataset_names: List of dataset names
            VisualizePath: The VisualizePath class
        """
        network_dir = os.path.join(vis_dir, "networks")
        os.makedirs(network_dir, exist_ok=True)
        
        # 1. Generate networks combining ALL datasets at each threshold
        for threshold in self.parameters.thresholds:
            aligned = self.get_aligned_data(threshold)
            if aligned.empty:
                continue
            
            available = [d for d in dataset_names if d in aligned.columns]
            if not available:
                continue
            
            # Build combined connection data with dataset tags
            conn_rows = []
            for edge_key, row in aligned.iterrows():
                if ' -> ' not in str(edge_key):
                    continue
                parts = str(edge_key).split(' -> ')
                source = parts[0]
                target = parts[1] if len(parts) > 1 else ''
                
                for dataset in available:
                    weight = row.get(dataset, 0)
                    if weight > 0:
                        safe_name = self.parameters._sanitize_name(dataset)
                        conn_rows.append({
                            'source': f"{source}",
                            'target': f"{target}",
                            'weight': weight,
                            'dataset': safe_name,
                            'path_block': f"{source} → {target} ({safe_name})"
                        })
            
            if conn_rows:
                output_file = os.path.join(network_dir, f"network_all_datasets_minsyn_{threshold}.html")
                self._create_combined_network_html(conn_rows, available, threshold, output_file, 
                                                   title=f"All Datasets at Threshold {threshold}")
        
        # 2. Generate networks combining ALL thresholds for each dataset
        for dataset in dataset_names:
            safe_name = self.parameters._sanitize_name(dataset)
            conn_rows = []
            
            for threshold in self.parameters.thresholds:
                aligned = self.get_aligned_data(threshold)
                if aligned.empty or dataset not in aligned.columns:
                    continue
                
                for edge_key, row in aligned.iterrows():
                    weight = row.get(dataset, 0)
                    if weight <= 0:
                        continue
                    
                    if ' -> ' not in str(edge_key):
                        continue
                    parts = str(edge_key).split(' -> ')
                    source = parts[0]
                    target = parts[1] if len(parts) > 1 else ''
                    
                    conn_rows.append({
                        'source': source,
                        'target': target,
                        'weight': weight,
                        'threshold': threshold,
                        'path_block': f"{source} → {target} (t={threshold})"
                    })
            
            if conn_rows:
                output_file = os.path.join(network_dir, f"network_{safe_name}_all_thresholds.html")
                self._create_threshold_comparison_network_html(
                    conn_rows, safe_name, self.parameters.thresholds, output_file,
                    title=f"{safe_name}: All Thresholds"
                )

    def _create_combined_network_html(self, conn_rows: List[Dict], datasets: List[str],
                                       threshold: int, output_path: str, title: str):
        """
        Create a combined network HTML with all datasets side by side.
        
        Uses vis.js with dataset-based coloring to distinguish connections.
        """
        import json
        
        # Build nodes and edges
        nodes = []
        edges = []
        node_ids = {}
        node_counter = 0
        
        # Color palette for datasets
        colors = ['#3b82f6', '#8b5cf6', '#22c55e', '#f59e0b', '#ef4444', '#06b6d4']
        dataset_colors = {ds: colors[i % len(colors)] for i, ds in enumerate(datasets)}
        
        for row in conn_rows:
            source = row['source']
            target = row['target']
            weight = row['weight']
            dataset = row.get('dataset', 'unknown')
            
            # Add source node
            if source not in node_ids:
                node_ids[source] = node_counter
                nodes.append({
                    'id': node_counter, 
                    'label': source, 
                    'group': 'source',
                    'title': f'{source}'
                })
                node_counter += 1
            
            # Add target node  
            if target not in node_ids:
                node_ids[target] = node_counter
                nodes.append({
                    'id': node_counter, 
                    'label': target, 
                    'group': 'target',
                    'title': f'{target}'
                })
                node_counter += 1
            
            # Add edge with dataset color
            edge_color = dataset_colors.get(self.parameters._sanitize_name(dataset), '#64748b')
            edges.append({
                'from': node_ids[source],
                'to': node_ids[target],
                'value': int(weight),
                'title': f"{source} → {target}: {int(weight)} ({dataset})",
                'arrows': 'to',
                'color': {'color': edge_color, 'highlight': edge_color}
            })
        
        # Create legend HTML
        legend_items = ''.join([
            f'<span style="color:{dataset_colors.get(self.parameters._sanitize_name(ds), "#64748b")}; margin-right:15px;">● {self.parameters._sanitize_name(ds)}</span>'
            for ds in datasets
        ])
        
        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        h1 {{ color: #2563eb; }}
        .legend {{ margin: 10px 0; font-size: 14px; }}
        #network {{ width: 100%; height: 700px; border: 1px solid #e2e8f0; border-radius: 8px; }}
        .stats {{ background: #f8fafc; padding: 10px 15px; border-radius: 8px; margin-bottom: 15px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="stats">
        <strong>{len(nodes)}</strong> neurons | <strong>{len(edges)}</strong> connections | 
        <strong>{len(datasets)}</strong> datasets
    </div>
    <div class="legend">{legend_items}</div>
    <div id="network"></div>
    <script>
        const nodes = new vis.DataSet({json.dumps(nodes)});
        const edges = new vis.DataSet({json.dumps(edges)});
        const container = document.getElementById('network');
        const data = {{ nodes: nodes, edges: edges }};
        const options = {{
            nodes: {{
                shape: 'dot',
                size: 20,
                font: {{ size: 12 }},
                borderWidth: 2
            }},
            edges: {{
                width: 2,
                arrows: {{ to: {{ enabled: true, scaleFactor: 0.8 }} }},
                smooth: {{ type: 'curvedCW', roundness: 0.15 }}
            }},
            groups: {{
                source: {{ color: {{ background: '#3b82f6', border: '#1d4ed8' }} }},
                target: {{ color: {{ background: '#8b5cf6', border: '#6d28d9' }} }},
                intermediate: {{ color: {{ background: '#22c55e', border: '#15803d' }} }}
            }},
            layout: {{
                hierarchical: {{
                    enabled: true,
                    direction: 'LR',
                    sortMethod: 'directed',
                    levelSeparation: 200,
                    nodeSpacing: 80
                }}
            }},
            physics: {{ enabled: false }},
            interaction: {{ hover: true, tooltipDelay: 100 }}
        }};
        new vis.Network(container, data, options);
    </script>
</body>
</html>'''
        
        with open(output_path, 'w') as f:
            f.write(html)
        self._log(f"Saved: {os.path.basename(output_path)}")

    def _create_threshold_comparison_network_html(self, conn_rows: List[Dict], 
                                                   dataset: str, thresholds: List[int],
                                                   output_path: str, title: str):
        """
        Create a network HTML comparing all thresholds for a single dataset.
        
        Uses edge width/opacity to show threshold sensitivity.
        """
        import json
        
        # Aggregate connections across thresholds
        # For each edge, track which thresholds it appears at and max weight
        edge_data = {}
        for row in conn_rows:
            source = row['source']
            target = row['target']
            weight = row['weight']
            threshold = row['threshold']
            
            key = (source, target)
            if key not in edge_data:
                edge_data[key] = {'thresholds': [], 'weights': [], 'max_weight': 0}
            edge_data[key]['thresholds'].append(threshold)
            edge_data[key]['weights'].append(weight)
            edge_data[key]['max_weight'] = max(edge_data[key]['max_weight'], weight)
        
        # Build nodes and edges
        nodes = []
        edges = []
        node_ids = {}
        node_counter = 0
        
        # Color by threshold count (more thresholds = more "conserved")
        num_thresholds = len(thresholds)
        
        for (source, target), data in edge_data.items():
            # Add source node
            if source not in node_ids:
                node_ids[source] = node_counter
                nodes.append({
                    'id': node_counter, 
                    'label': source, 
                    'group': 'source',
                    'title': f'{source}'
                })
                node_counter += 1
            
            # Add target node
            if target not in node_ids:
                node_ids[target] = node_counter
                nodes.append({
                    'id': node_counter, 
                    'label': target, 
                    'group': 'target',
                    'title': f'{target}'
                })
                node_counter += 1
            
            # Edge color based on threshold conservation
            t_count = len(data['thresholds'])
            if t_count == num_thresholds:
                color = '#22c55e'  # Green: in all thresholds
            elif t_count >= num_thresholds * 0.5:
                color = '#f59e0b'  # Orange: in half
            else:
                color = '#ef4444'  # Red: in few
            
            t_list = ', '.join(str(t) for t in sorted(data['thresholds']))
            edges.append({
                'from': node_ids[source],
                'to': node_ids[target],
                'value': int(data['max_weight']),
                'title': f"{source} → {target}\\nMax weight: {int(data['max_weight'])}\\nThresholds: {t_list}",
                'arrows': 'to',
                'color': {'color': color, 'highlight': color}
            })
        
        # Legend
        legend_html = '''
        <span style="color:#22c55e; margin-right:15px;">● All thresholds</span>
        <span style="color:#f59e0b; margin-right:15px;">● Half thresholds</span>
        <span style="color:#ef4444; margin-right:15px;">● Few thresholds</span>
        '''
        
        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        h1 {{ color: #2563eb; }}
        .legend {{ margin: 10px 0; font-size: 14px; }}
        #network {{ width: 100%; height: 700px; border: 1px solid #e2e8f0; border-radius: 8px; }}
        .stats {{ background: #f8fafc; padding: 10px 15px; border-radius: 8px; margin-bottom: 15px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="stats">
        <strong>{len(nodes)}</strong> neurons | <strong>{len(edges)}</strong> unique connections | 
        Thresholds: <strong>{', '.join(str(t) for t in thresholds)}</strong>
    </div>
    <div class="legend">{legend_html}</div>
    <div id="network"></div>
    <script>
        const nodes = new vis.DataSet({json.dumps(nodes)});
        const edges = new vis.DataSet({json.dumps(edges)});
        const container = document.getElementById('network');
        const data = {{ nodes: nodes, edges: edges }};
        const options = {{
            nodes: {{
                shape: 'dot',
                size: 20,
                font: {{ size: 12 }},
                borderWidth: 2
            }},
            edges: {{
                width: 2,
                arrows: {{ to: {{ enabled: true, scaleFactor: 0.8 }} }},
                smooth: {{ type: 'curvedCW', roundness: 0.15 }}
            }},
            groups: {{
                source: {{ color: {{ background: '#3b82f6', border: '#1d4ed8' }} }},
                target: {{ color: {{ background: '#8b5cf6', border: '#6d28d9' }} }},
                intermediate: {{ color: {{ background: '#22c55e', border: '#15803d' }} }}
            }},
            layout: {{
                hierarchical: {{
                    enabled: true,
                    direction: 'LR',
                    sortMethod: 'directed',
                    levelSeparation: 200,
                    nodeSpacing: 80
                }}
            }},
            physics: {{ enabled: false }},
            interaction: {{ hover: true, tooltipDelay: 100 }}
        }};
        new vis.Network(container, data, options);
    </script>
</body>
</html>'''
        
        with open(output_path, 'w') as f:
            f.write(html)
        self._log(f"Saved: {os.path.basename(output_path)}")

    # =========================================================================
    # Connectivity Profile Verification
    # =========================================================================
    
    def run_connectivity_profile_verification(
        self,
        output_dir: Optional[str] = None,
        direction: Optional[str] = None,
        comparison_mode: Optional[str] = None,
        include_partner_details: Optional[bool] = None,
        include_visualizations: Optional[bool] = None,
        top_k: Optional[int] = None,
        top_m: Optional[int] = None,
        min_synapse_threshold: Optional[int] = None,
        include_untyped_partners: Optional[bool] = None,
        # Strict mode parameters
        min_common_partners: Optional[int] = None,
        # Score weights for combined score
        score_weights: Optional[Dict[str, float]] = None,
        # Parallel processing parameters
        parallel: Optional[bool] = None,
        max_workers: Optional[int] = None,
        _skip_html_regeneration: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Run connectivity profile verification for all neuron types in the comparison.
        
        This method integrates the connectivity profile analysis with the comparison
        results, verifying that neurons with the same type labels have similar
        connectivity patterns across datasets.
        
        Two comparison modes are supported:
        
        **Loose Mode** (default): Type-level aggregated comparison
        - Aggregates all neurons of the same type into one profile
        - Computes Jaccard similarity, cosine similarity, rank correlation
        - Faster and good for initial screening
        - Best when: You want to compare overall type connectivity patterns
        
        **Strict Mode**: Per-bodyId individual comparison  
        - Compares individual neuron profiles within each type
        - Uses rank correlation on matching partner types (always uses ranks)
        - Includes 2-hop profile matching via profiler config
        - More computationally intensive but more precise
        - Best when: You want to verify individual neuron assignments
        
        The verification workflow:
        1. Extracts source, target, and intermediate types from comparison results
        2. Computes connectivity profiles for each type in each dataset
        3. Compares profiles using the selected mode's similarity metrics
        4. Generates verification report with confidence scores
        5. 2-hop expansion handled by profiler if config.expand_untyped_2hop is True
        
        All parameters default to values from ComparisonParameters if not provided.
        
        Args:
            output_dir: Directory to save verification results (default: comparison output folder)
            direction: 'upstream', 'downstream', or 'both' for profile comparison
                      (default: params.verification_direction)
            comparison_mode: 'loose' (type-aggregated) or 'strict' (per-bodyId)
                           (default: params.verification_mode)
            include_partner_details: Include per-type partner overlap CSVs
                                    (default: params.verification_include_partner_details)
            include_visualizations: Generate visualization plots/heatmaps
                                   (default: params.verification_include_visualizations)
            top_k: Number of top partners per direction 
                  (default: params.verification_top_k)
            top_m: Minimum unique partner types 
                  (default: params.verification_top_m)
            min_synapse_threshold: Minimum synapse count for connections
                                  (default: params.verification_min_synapse_threshold)
            include_untyped_partners: Include partners without type annotations
                                     (default: params.verification_include_untyped)
            min_common_partners: (strict mode) Minimum shared partners required for comparison
                                (default: params.verification_min_common_partners)
            score_weights: Custom weights for combined score {'jaccard': 0.5, 'rank': 0.5}
                          (default: params.verification_score_weights)
            parallel: Enable parallel processing for batch operations
                     (default: params.parallel)
            max_workers: Maximum parallel workers (default: params.max_workers)
            _skip_html_regeneration: Internal flag - skip HTML regeneration (used when called from export_results)
        
        Returns:
            Dict with keys:
            - 'source': DataFrame with source type verification
            - 'target': DataFrame with target type verification  
            - 'intermediate': DataFrame with intermediate type verification
            - 'summary': Overall summary DataFrame
            - 'similarity_matrix': Cross-dataset similarity matrix
            - 'comparison_mode': The mode used ('loose' or 'strict')
        
        Note:
            - Ranks are always used for bodyId-level comparison (proportions not used)
            - 2-hop expansion is controlled via profiler config.expand_untyped_2hop
            - If 2-hop doesn't return typed neurons in top_k, untyped partners are ignored
        
        Example:
            >>> analyzer = ComparisonAnalyzer(params)
            >>> results = analyzer.run_comparison()
            >>> 
            >>> # Use defaults from ComparisonParameters
            >>> verification = analyzer.run_connectivity_profile_verification()
            >>> 
            >>> # Override specific parameters
            >>> verification = analyzer.run_connectivity_profile_verification(
            ...     comparison_mode='strict',
            ...     top_k=10
            ... )
            >>> print(verification['summary'])
        """
        # Apply defaults from ComparisonParameters
        p = self.parameters
        direction = direction if direction is not None else p.verification_direction
        comparison_mode = comparison_mode if comparison_mode is not None else p.verification_mode
        include_partner_details = include_partner_details if include_partner_details is not None else p.verification_include_partner_details
        include_visualizations = include_visualizations if include_visualizations is not None else p.verification_include_visualizations
        top_k = top_k if top_k is not None else p.verification_top_k
        top_m = top_m if top_m is not None else p.verification_top_m
        min_synapse_threshold = min_synapse_threshold if min_synapse_threshold is not None else p.verification_min_synapse_threshold
        include_untyped_partners = include_untyped_partners if include_untyped_partners is not None else p.verification_include_untyped
        min_common_partners = min_common_partners if min_common_partners is not None else p.verification_min_common_partners
        score_weights = score_weights if score_weights is not None else p.verification_score_weights
        parallel = parallel if parallel is not None else p.parallel
        max_workers = max_workers if max_workers is not None else p.max_workers
        
        try:
            from .connectivity_profiler import ConnectivityProfiler, ProfilerConfig
            from .cross_dataset_verifier import CrossDatasetVerifier
            from .profile_visualizations import ProfileVisualizer
        except ImportError as e:
            self._log(f"Warning: Connectivity profile modules not available: {e}")
            return {}
        
        # Validate comparison_mode
        if comparison_mode not in ['loose', 'strict']:
            self._log(f"Warning: Invalid comparison_mode '{comparison_mode}', using 'loose'")
            comparison_mode = 'loose'
        
        self._log(f"Running connectivity profile verification (mode: {comparison_mode})...")
        
        # Setup output directory
        if output_dir is None:
            output_dir = os.path.join(
                self.parameters.full_output_path,
                "connectivity_profile_verification"
            )
        os.makedirs(output_dir, exist_ok=True)
        
        # Get dataset names
        dataset_names = self.parameters.get_dataset_names()
        
        if len(dataset_names) < 2:
            self._log("Need at least 2 datasets for verification")
            return {}
        
        # Extract neuron types from comparison results
        source_types, target_types, intermediate_types = self._extract_types_from_results()
        
        # Combine types with priority: source first, then target, then intermediate
        # This ensures source/target types are always included in the similarity matrix
        all_types_ordered = list(source_types) + [t for t in target_types if t not in source_types]
        seen = set(all_types_ordered)
        all_types_ordered += [t for t in intermediate_types if t not in seen]
        
        self._log(f"Found {len(source_types)} source, {len(target_types)} target, "
                  f"{len(intermediate_types)} intermediate types")
        
        if not all_types_ordered:
            self._log("No neuron types found in comparison results")
            return {}
        
        # Create profiler with config from parameters
        config = ProfilerConfig(
            top_k_bodyid=top_k,
            top_m_type=top_m,
            min_synapse_threshold=min_synapse_threshold,
            include_untyped_partners=include_untyped_partners,
            use_cache=True
        )
        
        profiler = ConnectivityProfiler(
            datasets=dataset_names,
            config=config,
            token=self.parameters.resolve_token(),
            verbose=self.verbose
        )
        
        # Pre-load profiles from parquet cache for efficiency
        # This leverages the build_connectivity_profile_cache / read_connectivity_profile_cache system
        self._log("Pre-loading connectivity profiles from cache...")
        preloaded_count = 0
        for dataset in dataset_names:
            try:
                # Read all cached profiles for this dataset
                cached_profiles = profiler.read_connectivity_profile_cache(
                    dataset=dataset,
                    neuron_types=all_types_ordered  # Only load types we need
                )
                if cached_profiles:
                    self._log(f"  Loaded {len(cached_profiles)} cached profiles for {dataset}")
                    preloaded_count += len(cached_profiles)
            except Exception as e:
                self._log(f"  Cache read failed for {dataset}: {e}")
        
        if preloaded_count > 0:
            self._log(f"Pre-loaded {preloaded_count} profiles from cache")
        else:
            self._log("No cached profiles found - will extract on demand")
        
        # Create verifier with comparison mode and score weights
        verifier = CrossDatasetVerifier(
            profiler=profiler,
            label_mapper=self.label_mapper,
            verbose=self.verbose,
            comparison_mode=comparison_mode,
            min_common_partners=min_common_partners,
            score_weights=score_weights
        )
        
        # Run verification for all types
        results = verifier.verify_comparison_results(
            source_types=source_types,
            target_types=target_types,
            intermediate_types=intermediate_types,
            datasets=dataset_names,
            direction=direction,
            parallel=parallel,
            max_workers=max_workers
        )
        
        # Store the comparison mode used
        results['comparison_mode'] = comparison_mode
        
        # Build cross-dataset similarity matrix (use rank_corr as primary metric)
        # Use all_types_ordered to ensure source/target types are always included
        if all_types_ordered:
            self._log("Building similarity matrix (rank correlation)...")
            similarity_matrix = verifier.build_cross_dataset_similarity_matrix(
                neuron_types=all_types_ordered[:50],  # Limit for performance, but source/target types are first
                datasets=dataset_names,
                metric='rank',  # Use rank correlation as primary metric
                direction=direction,
                parallel=parallel,
                max_workers=max_workers
            )
            results['similarity_matrix'] = similarity_matrix
            
            # Store for HTML report access
            self._profile_similarity_matrix = similarity_matrix
            
            # Build metric-specific matrices for additional visualizations
            self._log("Building metric-specific matrices...")
            metric_matrices = verifier.build_multi_metric_matrices(
                neuron_types=all_types_ordered[:50],
                datasets=dataset_names,
                direction=direction,
                parallel=parallel,
                max_workers=max_workers
            )
            results['metric_matrices'] = metric_matrices
            
            # Save metric matrices to CSVs
            metric_dir = os.path.join(output_dir, 'metric_matrices')
            os.makedirs(metric_dir, exist_ok=True)
            for metric_name, matrix in metric_matrices.items():
                matrix.to_csv(os.path.join(metric_dir, f'similarity_matrix_{metric_name}.csv'))
            self._log("Saved: metric_matrices/*.csv")
        
        # Save results
        self._save_profile_verification_results(
            results, output_dir, include_partner_details, verifier, dataset_names, direction
        )
        
        # Generate visualizations if requested
        if include_visualizations:
            self._generate_profile_visualizations(
                results, output_dir, profiler, dataset_names
            )
        
        # Regenerate main HTML report to include profile similarity matrix
        # Skip if called from export_results (HTML will be generated after this)
        if not _skip_html_regeneration:
            try:
                html_report_path = os.path.join(os.path.dirname(output_dir), "comparison_report.html")
                self.generate_html_report(html_report_path)
                self._log("Updated: comparison_report.html (with profile similarity)")
            except Exception as e:
                self._log(f"Warning: Could not update HTML report with profile data: {e}")
        
        self._log(f"Connectivity profile verification complete. Results saved to: {output_dir}")
        
        return results
    
    def _extract_types_from_results(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Extract source, target, and intermediate neuron types from comparison results.
        
        Returns:
            Tuple of (source_types, target_types, intermediate_types)
        """
        source_types = set()
        target_types = set()
        intermediate_types = set()
        
        # Get from parameters
        source_neurons = self.parameters.source_neurons
        target_neurons = self.parameters.target_neurons
        
        # Process source neurons - extract concrete types if possible
        for sn in source_neurons:
            if isinstance(sn, str) and '.*' not in sn and '*' not in sn:
                source_types.add(sn)
        
        # Process target neurons
        for tn in target_neurons:
            if isinstance(tn, str) and '.*' not in tn and '*' not in tn:
                target_types.add(tn)
        
        # Extract from raw results if available
        for dataset_name, threshold_results in self.raw_results.items():
            for threshold, df in threshold_results.items():
                if df.empty:
                    continue
                
                # Source types from type_pre column
                if 'type_pre' in df.columns:
                    for t in df['type_pre'].dropna().unique():
                        t_str = str(t)
                        if t_str and t_str != 'nan':
                            # Check if it matches source pattern
                            for sn in source_neurons:
                                if self._type_matches_pattern(t_str, sn):
                                    source_types.add(t_str)
                                    break
                            else:
                                # Not a source, might be intermediate
                                for tn in target_neurons:
                                    if self._type_matches_pattern(t_str, tn):
                                        target_types.add(t_str)
                                        break
                                else:
                                    intermediate_types.add(t_str)
                
                # Target types from type_post column
                if 'type_post' in df.columns:
                    for t in df['type_post'].dropna().unique():
                        t_str = str(t)
                        if t_str and t_str != 'nan':
                            for tn in target_neurons:
                                if self._type_matches_pattern(t_str, tn):
                                    target_types.add(t_str)
                                    break
                            else:
                                # Check if source
                                for sn in source_neurons:
                                    if self._type_matches_pattern(t_str, sn):
                                        source_types.add(t_str)
                                        break
                                else:
                                    intermediate_types.add(t_str)
        
        return list(source_types), list(target_types), list(intermediate_types)
    
    def _type_matches_pattern(self, type_name: str, pattern: Union[str, int]) -> bool:
        """Check if type_name matches the pattern (supports regex patterns)."""
        import re
        
        if isinstance(pattern, int):
            return str(pattern) == type_name
        
        if '.*' in pattern or '*' in pattern:
            regex_pattern = pattern.replace('.*', '.*').replace('*', '.*')
            try:
                return bool(re.match(f'^{regex_pattern}$', type_name))
            except re.error:
                return pattern == type_name
        
        return pattern == type_name
    
    def _save_profile_verification_results(
        self,
        results: Dict[str, pd.DataFrame],
        output_dir: str,
        include_partner_details: bool,
        verifier,
        dataset_names: List[str],
        direction: str
    ):
        """Save verification results to CSV files."""
        # Save summary
        if 'summary' in results and not results['summary'].empty:
            results['summary'].to_csv(
                os.path.join(output_dir, 'verification_summary.csv'),
                index=False
            )
            self._log("Saved: verification_summary.csv")
        
        # Save by role
        for role in ['source', 'target', 'intermediate']:
            if role in results and not results[role].empty:
                results[role].to_csv(
                    os.path.join(output_dir, f'verification_{role}.csv'),
                    index=False
                )
                self._log(f"Saved: verification_{role}.csv")
        
        # Save similarity matrix
        if 'similarity_matrix' in results and not results['similarity_matrix'].empty:
            results['similarity_matrix'].to_csv(
                os.path.join(output_dir, 'similarity_matrix.csv')
            )
            self._log("Saved: similarity_matrix.csv")
        
        # Generate partner details if requested
        if include_partner_details and verifier is not None:
            all_types = set()  # Use set to avoid duplicates
            for role in ['source', 'target', 'intermediate']:
                if role in results and not results[role].empty:
                    types = results[role]['neuron_type'].tolist()
                    all_types.update(types[:10])  # Limit to first 10 per role
            
            if all_types:
                verifier.generate_verification_report(
                    neuron_types=list(all_types),  # Convert back to list
                    datasets=dataset_names,
                    output_path=output_dir,
                    direction=direction,
                    include_partner_details=True
                )
    
    def _generate_profile_visualizations(
        self,
        results: Dict[str, pd.DataFrame],
        output_dir: str,
        profiler,
        dataset_names: List[str]
    ):
        """Generate visualizations for connectivity profile verification."""
        try:
            from .profile_visualizations import ProfileVisualizer
        except ImportError:
            self._log("Warning: ProfileVisualizer not available, skipping visualizations")
            return
        
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate similarity heatmap from rank_corr matrix (primary metric)
        if 'similarity_matrix' in results and not results['similarity_matrix'].empty:
            try:
                ProfileVisualizer.plot_similarity_heatmap(
                    results['similarity_matrix'],
                    title="Cross-Dataset Profile Similarity (Rank Correlation)",
                    output_path=os.path.join(vis_dir, 'similarity_heatmap.png'),
                    vmin=-1.0,  # Rank corr can be negative
                    vmax=1.0
                )
                self._log("Saved: similarity_heatmap.png")
            except Exception as e:
                self._log(f"Warning: Could not generate similarity heatmap: {e}")
        
        # Generate metric-specific heatmaps if available
        if 'metric_matrices' in results and results['metric_matrices']:
            metric_dir = os.path.join(vis_dir, 'metric_heatmaps')
            os.makedirs(metric_dir, exist_ok=True)
            
            metric_titles = {
                'jaccard': 'Jaccard Similarity',
                'cosine': 'Cosine Similarity',
                'rank': 'Rank Correlation',
                'combined': 'Combined Score'
            }
            
            for metric_name, matrix in results['metric_matrices'].items():
                if matrix is not None and not matrix.empty:
                    try:
                        vmin = -1.0 if metric_name == 'rank' else 0.0
                        ProfileVisualizer.plot_similarity_heatmap(
                            matrix,
                            title=f"Cross-Dataset Profile Similarity ({metric_titles.get(metric_name, metric_name)})",
                            output_path=os.path.join(metric_dir, f'similarity_{metric_name}.png'),
                            vmin=vmin,
                            vmax=1.0
                        )
                        self._log(f"Saved: metric_heatmaps/similarity_{metric_name}.png")
                    except Exception as e:
                        self._log(f"Warning: Could not generate {metric_name} heatmap: {e}")
        
        # Generate role comparison chart if available
        if 'summary' in results and not results['summary'].empty and 'role' in results['summary'].columns:
            try:
                ProfileVisualizer.plot_role_comparison(
                    results['summary'],
                    title="Connectivity Profile Similarity by Role",
                    output_path=os.path.join(vis_dir, 'role_comparison.png')
                )
                self._log("Saved: role_comparison.png")
            except Exception as e:
                self._log(f"Warning: Could not generate role comparison: {e}")
        
        # Generate verification summary bar chart
        if 'summary' in results and not results['summary'].empty:
            try:
                ProfileVisualizer.plot_verification_summary(
                    results['summary'],
                    title="Neuron Type Verification Summary",
                    output_path=os.path.join(vis_dir, 'verification_summary.png')
                )
                self._log("Saved: verification_summary.png")
            except Exception as e:
                self._log(f"Warning: Could not generate verification summary: {e}")
        
        # Generate profile comparisons for top types
        if 'summary' in results and not results['summary'].empty:
            top_types = results['summary'].head(5)['neuron_type'].tolist()
            
            for neuron_type in top_types:
                try:
                    # Get profiles for all datasets
                    profiles = {}
                    for ds in dataset_names:
                        try:
                            profile = profiler.get_profile(neuron_type, ds)
                            profiles[ds] = profile
                        except Exception:
                            pass
                    
                    if len(profiles) >= 2:
                        # Use the static method with correct signature
                        ProfileVisualizer.plot_profile_comparison(
                            profiles=profiles,
                            neuron_type=neuron_type,
                            direction='both',
                            output_path=os.path.join(vis_dir, f'profile_{neuron_type}.png')
                        )
                        self._log(f"Saved: profile_{neuron_type}.png")
                        
                except Exception as e:
                    self._log(f"Warning: Could not generate profile comparison for {neuron_type}: {e}")
        
        # Generate HTML report - save to parent folder (comparison_results_{timestamp})
        try:
            # output_dir is .../connectivity_profile_verification
            # We want .../connectivity_profile_comparison.html
            parent_dir = os.path.dirname(output_dir)
            html_output_path = os.path.join(parent_dir, 'connectivity_profile_comparison.html')
            
            # Link back to main comparison report (if exists in same folder)
            main_report_url = 'comparison_report.html'
            
            ProfileVisualizer.generate_html_report(
                results,
                similarity_matrix=results.get('similarity_matrix'),
                metric_matrices=results.get('metric_matrices'),
                output_path=html_output_path,
                title="Connectivity Profile Comparison Report",
                main_report_url=main_report_url
            )
            self._log(f"Saved: connectivity_profile_comparison.html")
        except Exception as e:
            self._log(f"Warning: Could not generate HTML report: {e}")
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_dataset_config(self, name: str) -> Optional[DatasetConfig]:
        """Get dataset configuration by name."""
        return self._dataset_configs.get(name)
    
    def clear_results(self):
        """Clear all cached results."""
        self.raw_results.clear()
        self.aligned_results.clear()
        self.comparison_report = None
        self._log("Results cleared")
    
    def set_label_mapper(self, mapper: LabelMapper):
        """
        Set or update label mapper.
        
        Args:
            mapper: LabelMapper instance
        """
        self.label_mapper = mapper
        # Clear aligned results since labels may change
        self.aligned_results.clear()
        self._log("Label mapper updated")
    
    def generate_html_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate an interactive HTML report with Plotly charts.
        
        Creates a self-contained HTML file with:
        - Executive summary with key findings
        - Interactive path count charts
        - Edge weight heatmap
        - Conservation analysis tables
        - Dataset comparison dashboard
        
        Args:
            output_path: Path to save the HTML report (default: comparison_results/comparison_report.html)
            
        Returns:
            Path to the generated HTML file
        """
        # Ensure comparison has been run
        if self.comparison_report is None:
            self.run_comparison_analysis()
        
        if output_path is None:
            output_path = os.path.join(
                self.parameters.full_output_path,
                "comparison_results",
                "comparison_report.html"
            )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Try to import plotly, fall back to basic HTML if not available
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            has_plotly = True
        except ImportError:
            has_plotly = False
            self._log("Warning: Plotly not installed. Generating basic HTML report.")
        
        html_content = self._generate_html_content(has_plotly)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self._log(f"HTML report saved to: {output_path}")
        return output_path
    
    def _generate_html_content(self, has_plotly: bool = True) -> str:
        """Generate the HTML content for the report - static version showing all thresholds."""
        from .html_report_generator import generate_html_report
        
        dataset_names = self.parameters.get_dataset_names()
        thresholds = self.parameters.thresholds
        
        # Generate mode-specific note
        mode_specific_note = self._generate_mode_specific_note()
        
        # Collect path count data for charts
        path_count_data = []
        for dataset in dataset_names:
            for threshold in thresholds:
                df = self.raw_results.get(dataset, {}).get(threshold, pd.DataFrame())
                count = len(df) if not df.empty else 0
                path_count_data.append({
                    'dataset': self.parameters._sanitize_name(dataset),
                    'threshold': threshold,
                    'count': count
                })
        
        # Get key findings per threshold and add path stats
        key_findings_per_threshold = self.comparison_report.get('key_findings_per_threshold', {})
        
        # Add path statistics to key findings
        for threshold in thresholds:
            path_data = self._get_path_data_for_threshold(threshold)
            if path_data is not None and not path_data.empty:
                available = [d for d in dataset_names if d in path_data.columns]
                total_paths = len(path_data)
                if available:
                    mask_all = (path_data[available] > 0).all(axis=1)
                    common_paths = int(mask_all.sum())
                else:
                    common_paths = 0
                
                if threshold in key_findings_per_threshold:
                    key_findings_per_threshold[threshold]['total_paths'] = total_paths
                    key_findings_per_threshold[threshold]['common_paths'] = common_paths
                    if total_paths > 0:
                        key_findings_per_threshold[threshold]['path_conservation_rate'] = common_paths / total_paths
        
        # Generate HTML using the new generator
        return generate_html_report(
            analyzer=self,
            dataset_names=dataset_names,
            thresholds=thresholds,
            mode_specific_note=mode_specific_note,
            path_count_data=path_count_data,
            key_findings_per_threshold=key_findings_per_threshold
        )


def quick_compare(
    datasets: List[Union[str, DatasetConfig]],
    source_neurons: List[Union[str, int]],
    target_neurons: List[Union[str, int]],
    max_interlayer: int = 2,
    thresholds: Optional[List[int]] = None,
    label_mapper: Optional[LabelMapper] = None,
    output_folder: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for quick cross-dataset comparison.
    
    Args:
        datasets: List of dataset identifiers (strings) or DatasetConfig objects
        source_neurons: Source neuron types/patterns (shared across all datasets)
        target_neurons: Target neuron types/patterns (shared across all datasets)
        max_interlayer: Maximum interlayer hops (default: 2)
        thresholds: List of weight thresholds (default: [1, 3, 5, 10, 20])
        label_mapper: Optional LabelMapper for standardization
        output_folder: Optional output directory
        verbose: Print progress
        
    Returns:
        Comparison results dictionary
        
    Example:
        >>> results = quick_compare(
        ...     datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
        ...     source_neurons=['MBON14.*_R'],
        ...     target_neurons=['KCg-d.*_R'],
        ...     max_interlayer=2,
        ...     thresholds=[1, 5, 10]
        ... )
    """
    if thresholds is None:
        thresholds = [1, 3, 5, 10, 20]
    
    params = ComparisonParameters(
        datasets=datasets,
        source_neurons=source_neurons,
        target_neurons=target_neurons,
        max_interlayer=max_interlayer,
        thresholds=thresholds,
        output_folder=output_folder or '.',
    )
    
    analyzer = ComparisonAnalyzer(params, label_mapper, verbose)
    results = analyzer.run_comparison()
    
    if output_folder:
        analyzer.export_results()
    
    return results
