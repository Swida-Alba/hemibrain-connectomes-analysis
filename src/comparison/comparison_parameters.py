"""
ComparisonParameters - Global comparison settings for cross-dataset analysis.

This module defines the ComparisonParameters dataclass that holds all global settings
for running cross-dataset comparisons. ComparisonParameters is the PRIMARY ENTRY POINT
for comparison workflows - all necessary parameters are defined here.

Workflow:
    1. Create ComparisonParameters with datasets (strings or DatasetConfig), neurons, and settings
    2. Pass to ComparisonAnalyzer for execution
    3. Optionally provide LabelMapper for cross-dataset label standardization
"""

from dataclasses import dataclass, field
from typing import List, Union, Optional, Any, Dict, TYPE_CHECKING
from datetime import datetime
import os

if TYPE_CHECKING:
    from .label_mapper import LabelMapper


@dataclass
class ComparisonParameters:
    """
    Global comparison settings for cross-dataset analysis.
    
    This is the PRIMARY ENTRY POINT for comparison workflows. All necessary parameters
    are defined here, including datasets, source/target neurons, and analysis settings.
    
    Attributes:
        datasets (List[str | DatasetConfig]): List of dataset identifiers 
            (e.g., 'hemibrain:v1.2.1', 'male-cns:v0.9') or DatasetConfig objects.
            At least 2 datasets required for comparison (or 1 with allow_single_dataset=True).
            
        datasets_nickname (List[str], optional): Short display names for datasets in
            visualizations. If provided, must match length of datasets list.
            Example: ['Hemi', 'Male', 'FAFB'] for shorter labels in charts/networks.
            
        source_neurons (List[str | int] | List[List] | LabelMapper): Source neuron 
            types, bodyIds, or patterns to analyze. Supports:
            - Simple list: ['MBON14.*_R', 'PPL101.*_R']
            - Nested list (groups): [['MBON14.*_R'], ['MBON06.*_R']]
            - LabelMapper: for dataset-specific neuron mapping
            
        target_neurons (List[str | int] | List[List] | LabelMapper): Target neuron
            types, bodyIds, or patterns. Same format options as source_neurons.
            
        max_interlayer (int): Maximum number of intermediate hops for path finding.
            Default: 2. Shared across ALL datasets for fair comparison.
            
        thresholds (List[int]): Minimum synapse count cutoffs for filtering.
            Default: [1, 3, 5, 10, 20]. Results generated at each threshold level.
            
        source_labels (str | List[str]): Custom label(s) for source neuron group(s)
            in output files and visualizations.
            
        target_labels (str | List[str]): Custom label(s) for target neuron group(s)
            in output files and visualizations.
            
        top_edges (int): Number of top edges to highlight in visualizations.
            Default: 50.
            
        comparison_mode (str): Analysis mode - 'path' or 'edge'.
            - 'path': Discover edges through source-to-target paths (may filter
              strong edges if path intermediates are weak)
            - 'edge': Query all edges independently (preserves all strong edges)
            Default: 'path'
            
        output_folder (str): Base directory for all output files. Default: '.'
        
        saveas (str, optional): Custom folder name for output. If None, generates
            timestamped name like 'comparison_results_20251127_120000'.
            
        token (str): NeuPrint authentication token. Empty string loads from
            NEUPRINT_APPLICATION_TOKEN environment variable.
            
        allow_single_dataset (bool): Allow single dataset mode for threshold
            sensitivity analysis. Default: False.
    
    Example:
        >>> # Basic comparison workflow
        >>> params = ComparisonParameters(
        ...     datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
        ...     datasets_nickname=['Hemi', 'Male'],  # Short display names
        ...     source_neurons=['MBON14.*_R'],
        ...     target_neurons=['KCg-d.*_R', 'PPL101.*_R'],
        ...     max_interlayer=2,
        ...     thresholds=[1, 3, 5, 10],
        ...     output_folder='./comparison_output',
        ...     saveas='mbon14_comparison'
        ... )
        >>> analyzer = ComparisonAnalyzer(params)
        >>> analyzer.run()
        >>>
        >>> # With grouped neurons (for separate analysis per group)
        >>> params = ComparisonParameters(
        ...     datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
        ...     source_neurons=[['MBON14.*_R'], ['MBON06.*_R']],  # 2 source groups
        ...     target_neurons=[['KCg-d.*_R'], ['PPL101.*_R']],   # 2 target groups
        ...     source_labels=['MBON14_grp', 'MBON06_grp'],
        ...     target_labels=['KCg-d_grp', 'PPL101_grp'],
        ... )
    """
    
    # Primary configuration (REQUIRED)
    datasets: List[Union[str, Any]] = field(default_factory=list)
    """List of dataset identifiers (e.g., 'hemibrain:v1.2.1', 'male-cns:v0.9') or DatasetConfig objects.
    At least 2 datasets required for cross-dataset comparison."""
    
    datasets_nickname: Optional[List[str]] = None
    """Short display names for datasets in visualizations (e.g., ['Hemi', 'Male', 'FAFB']).
    If provided, must match length of datasets list. Falls back to sanitized dataset names if None."""
    
    source_neurons: Union[List[Union[str, int]], List[List[Union[str, int]]], Any] = field(default_factory=list)
    """Source neuron types, bodyIds, or regex patterns. Supports:
    - Simple list: ['MBON14.*_R', 'PPL101.*_R']
    - Grouped list: [['MBON14.*_R'], ['MBON06.*_R']] for separate group analysis
    - LabelMapper object for dataset-specific neuron mapping"""
    
    target_neurons: Union[List[Union[str, int]], List[List[Union[str, int]]], Any] = field(default_factory=list)
    """Target neuron types, bodyIds, or regex patterns. Same format options as source_neurons."""
    
    max_interlayer: int = 2
    """Maximum interlayer hops for path finding (shared across ALL datasets)"""
    
    # Analysis settings
    thresholds: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    """Min synapse count thresholds for comparison (bodyId level filtering)"""
    
    source_labels: Union[str, List[str]] = ''
    """Unified label(s) for source group(s) - string or list matching group count"""
    
    target_labels: Union[str, List[str]] = ''
    """Unified label(s) for target group(s) - string or list matching group count"""
    
    intermediate_labels: Union[str, List[str]] = ''
    """Unified label(s) for intermediate group(s) - string or list matching group count"""
    
    top_edges: int = -1
    """Number of top edges for visualization focus. -1 means include all edges."""
    
    overall_label_mapper: Optional[Any] = None
    """LabelMapper object or configuration for defining source, target, and intermediate neuron labels."""
    
    verbose: bool = True
    """Whether to print initialization summary and progress updates."""
    
    # Connectivity Profile Verification Settings
    # These are used by run_connectivity_profile_verification() when called separately
    verification_direction: str = 'both'
    """Direction for profile verification: 'upstream', 'downstream', or 'both'"""
    
    verification_mode: str = 'loose'
    """Verification comparison mode: 'strict' or 'loose'.
    - 'strict': Only compare explicitly typed partners
    - 'loose': Include untyped partners in comparison"""
    
    verification_top_k: int = 15
    """Number of top partners to include per type for profile comparison"""
    
    verification_top_m: int = 5
    """Minimum synapse count for top partners (0 = no filter)"""
    
    verification_min_synapse_threshold: int = 3
    """Minimum synapse count for edges to include in verification"""
    
    verification_include_untyped: bool = True
    """Include untyped partners in profile comparison"""
    
    verification_min_common_partners: int = 3
    """Minimum common partners required for valid comparison"""
    
    verification_score_weights: Dict[str, float] = field(default_factory=lambda: {'jaccard': 0.50, 'rank': 0.50})
    """Weights for combining Jaccard and rank correlation scores"""
    
    verification_include_partner_details: bool = True
    """Include detailed partner information in verification output"""
    
    verification_include_visualizations: bool = True
    """Generate visualization files for verification results"""
    
    # Parallel processing settings
    parallel: bool = True
    """Enable parallel processing for batch operations (verification, similarity matrix)"""
    
    max_workers: Optional[int] = None
    """Maximum number of parallel workers. None = auto (min(32, cpu_count + 4))"""
    
    # Comparison mode
    comparison_mode: str = 'path'
    """Comparison mode: 'path' (default) or 'edge'.
    
    - 'path': Uses FindAllPath() to discover edges through paths from source to target.
              Edges are only included if they appear on valid paths. This may cause
              strong edges to be filtered if they only exist on paths with weak
              intermediate edges below the threshold.
              
    - 'edge': Queries all edges between relevant neuron types at the lowest threshold,
              then filters edges independently at each threshold level. This preserves
              all strong edges regardless of path context, providing more accurate
              edge presence comparison across datasets.
    """

    pathfinding: str = 'MemoizedDFS'
    """Pathfinding algorithm to use in FindAllPath:
    - 'MemoizedDFS': Meet-in-the-middle DFS - optimized for deep paths (L>=5) (default)
    - 'Bidirectional': Bidirectional BFS - optimized for shortest paths (high memory)
    - 'DP': Backward Reachability (DP) - optimized for pruning dead ends (low memory)
    - 'DFS': Backward Memoized DFS - standard traversal, no memoization (lowest memory, slower)
    - 'Backtracking': Backward DFS with backtracking
    """
    
    # Output settings
    output_folder: str = '.'
    """Base folder for all output files"""
    
    saveas: Optional[str] = None
    """Custom folder name (auto-generates with timestamp if None)"""
    
    # Authentication
    token: str = ''
    """NeuPrint token. Empty string = load from NEUPRINT_APPLICATION_TOKEN env var"""
    
    # Internal fixed settings (NOT user-configurable for comparison consistency)
    _use_cache: bool = field(default=True, repr=False)
    """Internal: caching enabled for performance"""
    
    _min_ratio: float = field(default=0.0, repr=False)
    """Internal: ratio filtering done post-hoc"""
    
    _min_prob: float = field(default=0.0, repr=False)
    """Internal: probability filtering done post-hoc"""
    
    _output_format: str = field(default='csv', repr=False)
    """Internal: output format fixed to CSV for comparison"""
    
    allow_single_dataset: bool = True
    """Allow single dataset for threshold sensitivity analysis only"""

    skip_bodyId: bool = False
    """If True, skip bodyId-level data saving, visualization, and calculations.
    Useful for large-scale analyses where only type-level data is needed."""

    force_API_fetching: bool = False
    """If True, use CAVE API to fetch FlyWire (FAFB) data instead of local files.
    This fetches connection data directly from the CAVE API for more up-to-date data.
    Note: Only applies to FAFB datasets. BANC does not support API fetching."""

    auto_type_mapping: bool = True
    """Enable automatic cross-dataset type mapping using male-cns neuron_df.
    
    When enabled, uses the type mapping columns from male-cns (flywireType, hemibrainType, mancType)
    to automatically resolve equivalent neuron types across datasets. This allows comparing neurons
    by their biological identity even when type names differ (e.g., MeVPLo2 in male-cns = MTe07 in flywire).
    
    Features:
    - Auto-loads mappings from male-cns_v0_9_allneurons_neuron_df.csv
    - Handles 1-to-1 type mappings automatically
    - Warns about N-to-1 mappings that could cause incorrect aggregation
    - Priority order: male-cns > flywire > manc > hemibrain > optic-lobe
    - LabelMapper has higher priority (manual mappings override auto-mapping)
    
    Note: Requires male-cns dataset to be initialized first for mapping file to exist."""

    def __post_init__(self):
        """Validate and process parameters after initialization."""
        from .label_mapper import LabelMapper
        
        # Ensure datasets is a list
        if isinstance(self.datasets, str):
            self.datasets = [self.datasets]

        # 1. Enforce Exclusivity: overall_label_mapper vs source/target LabelMappers
        has_overall = self.overall_label_mapper is not None
        is_source_mapper = isinstance(self.source_neurons, LabelMapper)
        is_target_mapper = isinstance(self.target_neurons, LabelMapper)

        if has_overall and (is_source_mapper or is_target_mapper):
            raise ValueError(
                "Ambiguous LabelMapper configuration: 'overall_label_mapper' cannot be used "
                "simultaneously with 'source_neurons' or 'target_neurons' as LabelMappers. "
                "Please provide either a single overall mapper OR specific source/target mappers."
            )
        
        # 2. Handle overall_label_mapper logic
        if has_overall:
            # If source_neurons provided alongside overall_label_mapper, raise error
            if self.source_neurons:
                raise ValueError(
                    "Ambiguous configuration: 'source_neurons' cannot be provided when "
                    "'overall_label_mapper' is used. The mapper defines the source neurons."
                )
            # If target_neurons provided alongside overall_label_mapper, raise error
            if self.target_neurons:
                raise ValueError(
                    "Ambiguous configuration: 'target_neurons' cannot be provided when "
                    "'overall_label_mapper' is used. The mapper defines the target neurons."
                )
            
            # Extract source/target info from overall_label_mapper
            # We don't populate self.source_neurons list here because it's dataset-specific.
            # Instead, we rely on the mapper being present.
            
            # Auto-populate labels if not provided
            if not self.source_labels:
                self.source_labels = self.overall_label_mapper.get_all_std_labels('source')
            if not self.target_labels:
                self.target_labels = self.overall_label_mapper.get_all_std_labels('target')
            if not self.intermediate_labels:
                self.intermediate_labels = self.overall_label_mapper.get_all_std_labels('intermediate')
                
            # Set internal mapper references
            self._source_mapper = self.overall_label_mapper
            self._target_mapper = self.overall_label_mapper

        # 3. Merge Logic: If source/target are mappers (and no overall), merge into overall_label_mapper
        elif is_source_mapper or is_target_mapper:
            self.overall_label_mapper = LabelMapper()
            
            if is_source_mapper:
                self.overall_label_mapper.merge(self.source_neurons)
                self._source_mapper = self.source_neurons
                self.source_neurons = []  # Clear list, rely on mapper
                
                if not self.source_labels:
                    self.source_labels = self._source_mapper.get_all_std_labels('source')

            if is_target_mapper:
                self.overall_label_mapper.merge(self.target_neurons)
                self._target_mapper = self.target_neurons
                self.target_neurons = []  # Clear list, rely on mapper
                
                if not self.target_labels:
                    self.target_labels = self._target_mapper.get_all_std_labels('target')
        
        # 4. Handle string/list inputs (if not mappers)
        else:
            if isinstance(self.source_neurons, str):
                self.source_neurons = [self.source_neurons]
            elif not self.source_neurons:
                self.source_neurons = []
                
            if isinstance(self.target_neurons, str):
                self.target_neurons = [self.target_neurons]
            elif not self.target_neurons:
                self.target_neurons = []
            
            # Initialize mapper references to None if not set
            self._source_mapper = None
            self._target_mapper = None
        
        # Ensure source_labels is a list
        if isinstance(self.source_labels, str) and self.source_labels:
            self.source_labels = [self.source_labels]
        elif not self.source_labels:
            self.source_labels = []
        
        # Ensure target_labels is a list
        if isinstance(self.target_labels, str) and self.target_labels:
            self.target_labels = [self.target_labels]
        elif not self.target_labels:
            self.target_labels = []
            
        # Ensure intermediate_labels is a list
        if isinstance(self.intermediate_labels, str) and self.intermediate_labels:
            self.intermediate_labels = [self.intermediate_labels]
        elif not self.intermediate_labels:
            self.intermediate_labels = []
        
        # Sort thresholds
        self.thresholds = sorted(self.thresholds)
        
        # Validate comparison_mode
        valid_modes = ['path', 'edge']
        if self.comparison_mode not in valid_modes:
            raise ValueError(f"comparison_mode must be one of {valid_modes}, got: {self.comparison_mode}")
        
        # Initialize auto type mapper (stores CrossDatasetTypeMapper instance if enabled)
        self._auto_type_mapper = None
        if self.auto_type_mapping:
            self._initialize_auto_type_mapping()
        
        # Validate minimum requirements
        if len(self.datasets) < 1:
            raise ValueError("At least 1 dataset is required")
        if len(self.datasets) < 2 and not self.allow_single_dataset:
            raise ValueError(
                "At least 2 datasets are required for cross-dataset comparison. "
                "Set allow_single_dataset=True for single-dataset threshold sensitivity analysis."
            )

        # Print initialization summary
        if self.verbose:
            print("\n=== ComparisonParameters Initialization Summary ===")
            print(f"Datasets Included ({len(self.datasets)}):")
            dataset_names = self.get_dataset_names()
            for i, ds in enumerate(self.datasets):
                nickname = self.datasets_nickname[i] if self.datasets_nickname and i < len(self.datasets_nickname) else "N/A"
                print(f"  - {ds} (Nickname: {nickname})")
                
            print(f"\nSource Neurons:")
            if self.overall_label_mapper:
                print(f"  Defined by LabelMapper (Labels: {self.source_labels})")
            else:
                print(f"  {self.source_neurons}")
                # Show auto type mapping results for source neurons
                if self.auto_type_mapping and self._auto_type_mapper:
                    self._print_neuron_mapping_summary(
                        self._ensure_flat_list(self.source_neurons),
                        dataset_names,
                        indent="  "
                    )
                 
            print(f"\nTarget Neurons:")
            if self.overall_label_mapper:
                print(f"  Defined by LabelMapper (Labels: {self.target_labels})")
            else:
                print(f"  {self.target_neurons}")
                # Show auto type mapping results for target neurons
                if self.auto_type_mapping and self._auto_type_mapper:
                    self._print_neuron_mapping_summary(
                        self._ensure_flat_list(self.target_neurons),
                        dataset_names,
                        indent="  "
                    )
            
            if self.intermediate_labels:
                print(f"\nIntermediate Neurons:")
                if self.overall_label_mapper:
                    print(f"  Defined by LabelMapper (Labels: {self.intermediate_labels})")
                else:
                    # If intermediate labels provided but no mapper, just print labels
                    print(f"  Labels: {self.intermediate_labels}")
                 
            print(f"\nLabel Mapper Provided: {'Yes' if self.overall_label_mapper else 'No'}")
            print(f"Auto Type Mapping: {'Enabled' if self.auto_type_mapping else 'Disabled'}")
            if self.auto_type_mapping and self._auto_type_mapper:
                print(f"  └─ Type mapper loaded successfully")
            print(f"Comparison Mode: {self.comparison_mode}")
            print("===================================================\n")
        
        if not self.thresholds:
            raise ValueError("At least one threshold is required")
        
        # Validate neurons: Must have either explicit list OR a mapper
        if not self._source_mapper and not self.source_neurons:
             # If we have an overall mapper but it has no source mappings, that might be an issue,
             # but technically _source_mapper should be set if overall_label_mapper is present.
             # Double check if overall_label_mapper is set but empty?
             if self.overall_label_mapper:
                 # If overall mapper exists, we assume it handles source neurons unless it's empty
                 pass 
             else:
                raise ValueError("source_neurons cannot be empty (provide list, nested list, or LabelMapper)")
                
        if not self._target_mapper and not self.target_neurons:
             if self.overall_label_mapper:
                 pass
             else:
                raise ValueError("target_neurons cannot be empty (provide list, nested list, or LabelMapper)")
        
        # Cache the timestamp for consistent output_name
        self._cached_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 5. Verify LabelMapper consistency (if used)
        if self.overall_label_mapper and not self.overall_label_mapper.is_empty:
            # Get dataset names from parameters
            param_datasets = set()
            for ds in self.datasets:
                # Handle DatasetConfig objects or strings
                ds_name = ds.dataset if hasattr(ds, 'dataset') else str(ds)
                param_datasets.add(ds_name)
            
            # Get datasets from mapper
            source_datasets = set()
            if self._source_mapper:
                for label_map in self._source_mapper._source_mapping.values():
                    source_datasets.update(label_map.keys())
            
            target_datasets = set()
            if self._target_mapper:
                for label_map in self._target_mapper._target_mapping.values():
                    target_datasets.update(label_map.keys())
            
            mapper_datasets = source_datasets.union(target_datasets)
            
            # Check 1: Datasets in LabelMapper should match ComparisonParameters.datasets
            missing_in_mapper = param_datasets - mapper_datasets
            if missing_in_mapper:
                print(f"\033[93mWarning: Datasets defined in parameters but missing from LabelMapper: {missing_in_mapper}\033[0m")
                
            # Check 2: Source vs Target consistency in Mapper
            if source_datasets != target_datasets:
                 print(f"\033[93mWarning: Inconsistent datasets between source and target mappings in LabelMapper.\033[0m")
                 if source_datasets - target_datasets:
                     print(f"\033[93m  Source only: {source_datasets - target_datasets}\033[0m")
                 if target_datasets - source_datasets:
                     print(f"\033[93m  Target only: {target_datasets - source_datasets}\033[0m")
    
    @property
    def run_timestamp(self) -> str:
        """Get the cached timestamp string for this run."""
        return self._cached_timestamp
    
    @property
    def output_name(self) -> str:
        """
        Get the output folder name.
        
        Returns:
            saveas value if provided, otherwise auto-generated name with timestamp
        """
        if self.saveas:
            return self.saveas
        return f"comparison_results_{self.run_timestamp}"
    
    @property
    def full_output_path(self) -> str:
        """
        Get the full path to the output folder.
        
        Returns:
            Absolute path combining output_folder and output_name
        """
        return os.path.abspath(os.path.join(self.output_folder, self.output_name))
    
    @property
    def dataset_data_path(self) -> str:
        """Get path to dataset_data subfolder."""
        return os.path.join(self.full_output_path, 'dataset_data')
    
    @property
    def comparison_results_path(self) -> str:
        """Get path to comparison_results subfolder."""
        return os.path.join(self.full_output_path, 'comparison_results')
    
    @property
    def visualizations_path(self) -> str:
        """Get path to visualizations subfolder."""
        return os.path.join(self.comparison_results_path, 'visualizations')
    
    def get_dataset_names(self) -> List[str]:
        """
        Get dataset identifier strings from datasets list.
        
        Handles both string datasets and DatasetConfig objects.
        
        Returns:
            List of dataset identifier strings
        """
        names = []
        for ds in self.datasets:
            if isinstance(ds, str):
                names.append(ds)
            else:
                # Assume DatasetConfig with dataset attribute
                names.append(getattr(ds, 'dataset', str(ds)))
        return names
    
    def get_dataset_nicknames(self) -> List[str]:
        """
        Get short nicknames for datasets for visualization labels.
        
        Returns datasets_nickname if provided, otherwise returns sanitized dataset names.
        
        Returns:
            List of short dataset labels for visualizations
        """
        if self.datasets_nickname and len(self.datasets_nickname) == len(self.datasets):
            return self.datasets_nickname
        # Fallback to sanitized names
        return [self._sanitize_name(n) for n in self.get_dataset_names()]
    
    def get_source_neurons_for_dataset(self, dataset: str) -> List[Union[str, int]]:
        """
        Get source neurons for a specific dataset.
        
        Resolution priority:
        1. LabelMapper (explicit mappings)
        2. Auto type mapping (from male-cns neuron_df)
        3. Shared source_neurons list (same names across all datasets)
        
        Args:
            dataset: Dataset identifier
            
        Returns:
            List of source neuron types/patterns/bodyIds
        """
        # Priority 1: LabelMapper
        if self._source_mapper is not None:
            # Get neurons from LabelMapper for this dataset
            return self._source_mapper.get_all_neurons_for_dataset(dataset, 'source')
        
        # Priority 2: Auto type mapping
        if self.auto_type_mapping and self._auto_type_mapper:
            return self._resolve_neurons_with_auto_mapping(
                self._ensure_flat_list(self.source_neurons), 
                dataset
            )
        
        # Priority 3: Shared list
        return self._ensure_flat_list(self.source_neurons)
    
    def get_target_neurons_for_dataset(self, dataset: str) -> List[Union[str, int]]:
        """
        Get target neurons for a specific dataset.
        
        Resolution priority:
        1. LabelMapper (explicit mappings)
        2. Auto type mapping (from male-cns neuron_df)
        3. Shared target_neurons list (same names across all datasets)
        
        Args:
            dataset: Dataset identifier
            
        Returns:
            List of target neuron types/patterns/bodyIds
        """
        # Priority 1: LabelMapper
        if self._target_mapper is not None:
            return self._target_mapper.get_all_neurons_for_dataset(dataset, 'target')
        
        # Priority 2: Auto type mapping
        if self.auto_type_mapping and self._auto_type_mapper:
            return self._resolve_neurons_with_auto_mapping(
                self._ensure_flat_list(self.target_neurons),
                dataset
            )
        
        # Priority 3: Shared list
        return self._ensure_flat_list(self.target_neurons)
    
    def get_source_groups(self) -> List[List[Union[str, int]]]:
        """
        Get source neurons as a list of groups.
        
        Returns:
            List of neuron groups (for grouped analysis)
        """
        return self._ensure_grouped(self.source_neurons)
    
    def get_target_groups(self) -> List[List[Union[str, int]]]:
        """
        Get target neurons as a list of groups.
        
        Returns:
            List of neuron groups (for grouped analysis)
        """
        return self._ensure_grouped(self.target_neurons)
    
    def _ensure_flat_list(self, neurons: List) -> List[Union[str, int]]:
        """Flatten nested list of neurons to single list."""
        flat = []
        for item in neurons:
            if isinstance(item, list):
                flat.extend(self._ensure_flat_list(item))
            else:
                flat.append(item)
        return flat
    
    def _ensure_grouped(self, neurons: List) -> List[List[Union[str, int]]]:
        """
        Ensure neurons are in grouped format (list of lists).
        """
        if not neurons:
            return [[]]
        
        # Check if first element is a list (already grouped)
        if isinstance(neurons[0], list):
            return neurons
        else:
            # Wrap in single group
            return [neurons]
    
    def get_dataset_output_path(self, dataset: str, threshold: int) -> str:
        """
        Get output path for a specific dataset and threshold.
        
        Args:
            dataset: Dataset identifier
            threshold: Synapse threshold
            
        Returns:
            Path to dataset/threshold output folder
        """
        safe_name = self._sanitize_name(dataset)
        return os.path.join(self.dataset_data_path, safe_name, f'minsyn_{threshold}')
    
    def _sanitize_name(self, name: str) -> str:
        """Convert name to filesystem-safe format.
        
        Replaces ':' and '.' with '_' but preserves '-' for readability.
        Examples:
            'hemibrain:v1.2.1' -> 'hemibrain_v1_2_1'
            'male-cns:v0.9' -> 'male-cns_v0_9'
            'flywire_FAFB_v783' -> 'flywire_FAFB_v783'
        """
        return name.replace(':', '_').replace('.', '_')
    
    def create_output_directories(self) -> None:
        """Create all necessary output directories."""
        # Main output folder
        os.makedirs(self.full_output_path, exist_ok=True)
        
        # Dataset data folder and subfolders
        for dataset in self.get_dataset_names():
            for threshold in self.thresholds:
                os.makedirs(self.get_dataset_output_path(dataset, threshold), exist_ok=True)
        
        # Comparison results folder
        os.makedirs(self.comparison_results_path, exist_ok=True)
        
        # Visualizations folder
        os.makedirs(self.visualizations_path, exist_ok=True)
    
    def resolve_token(self) -> str:
        """
        Resolve authentication token.
        
        Returns:
            Token from parameter, or from NEUPRINT_APPLICATION_TOKEN env var
        """
        if self.token:
            return self.token
        return os.environ.get('NEUPRINT_APPLICATION_TOKEN', '')
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation suitable for parameters.json
        """
        # Build source/target groups per dataset
        source_groups = {}
        target_groups = {}
        
        for dataset in self.get_dataset_names():
            source_groups[dataset] = self.get_source_neurons_for_dataset(dataset)
            target_groups[dataset] = self.get_target_neurons_for_dataset(dataset)
        
        return {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '2.1',
                'run_timestamp': self.run_timestamp
            },
            # Dataset configuration
            'datasets': self.get_dataset_names(),
            'datasets_nickname': self.get_dataset_nicknames(),
            
            # Neuron configuration
            'source_neurons': self._ensure_flat_list(self.source_neurons),
            'target_neurons': self._ensure_flat_list(self.target_neurons),
            'source_groups': source_groups,
            'target_groups': target_groups,
            'source_labels': self.source_labels,
            'target_labels': self.target_labels,
            
            # Analysis parameters
            'thresholds': self.thresholds,
            'max_interlayer': self.max_interlayer,
            'top_edges': self.top_edges,
            'comparison_mode': self.comparison_mode,
            
            # Verification settings
            'verification_settings': {
                'direction': self.verification_direction,
                'mode': self.verification_mode,
                'top_k': self.verification_top_k,
                'top_m': self.verification_top_m,
                'min_synapse_threshold': self.verification_min_synapse_threshold,
                'include_untyped': self.verification_include_untyped,
                'min_common_partners': self.verification_min_common_partners,
                'score_weights': self.verification_score_weights,
                'include_partner_details': self.verification_include_partner_details,
                'include_visualizations': self.verification_include_visualizations
            },
            
            # Performance settings
            'performance_settings': {
                'parallel': self.parallel,
                'max_workers': self.max_workers,
                'use_cache': self._use_cache
            },
            
            # Internal analysis settings
            'analysis_settings': {
                'min_ratio': self._min_ratio,
                'min_prob': self._min_prob,
                'output_format': self._output_format
            },
            
            # Output configuration
            'output_folder': self.output_folder,
            'saveas': self.saveas,
            'full_output_path': self.full_output_path
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ComparisonParameters':
        """
        Create ComparisonParameters from dictionary.
        
        Args:
            data: Dictionary with parameter data (e.g., from parameters.json)
            
        Returns:
            ComparisonParameters instance
        """
        # Extract nested settings with defaults
        verification = data.get('verification_settings', {})
        performance = data.get('performance_settings', {})
        
        return cls(
            # Dataset configuration
            datasets=data.get('datasets', []),
            datasets_nickname=data.get('datasets_nickname'),
            
            # Neuron configuration
            source_neurons=data.get('source_neurons', []),
            target_neurons=data.get('target_neurons', []),
            source_labels=data.get('source_labels', []),
            target_labels=data.get('target_labels', []),
            
            # Analysis parameters
            max_interlayer=data.get('max_interlayer', 2),
            thresholds=data.get('thresholds', [1, 3, 5, 10, 20]),
            top_edges=data.get('top_edges', 50),
            comparison_mode=data.get('comparison_mode', 'path'),
            
            # Verification settings
            verification_direction=verification.get('direction', 'both'),
            verification_mode=verification.get('mode', 'loose'),
            verification_top_k=verification.get('top_k', 5),
            verification_top_m=verification.get('top_m', 0),
            verification_min_synapse_threshold=verification.get('min_synapse_threshold', 3),
            verification_include_untyped=verification.get('include_untyped', True),
            verification_min_common_partners=verification.get('min_common_partners', 3),
            verification_score_weights=verification.get('score_weights', {'jaccard': 0.50, 'rank': 0.50}),
            verification_include_partner_details=verification.get('include_partner_details', True),
            verification_include_visualizations=verification.get('include_visualizations', True),
            
            # Performance settings
            parallel=performance.get('parallel', True),
            max_workers=performance.get('max_workers'),
            
            # Output configuration
            output_folder=data.get('output_folder', '.'),
            saveas=data.get('saveas'),
            token=data.get('token', ''),
        )
    
    def _initialize_auto_type_mapping(self) -> None:
        """
        Initialize auto type mapping from male-cns neuron_df.
        
        Loads CrossDatasetTypeMapper. Mapping warnings will be shown during
        the initialization summary print.
        """
        from .cross_dataset_type_mapper import CrossDatasetTypeMapper
        import os
        
        # Determine workspace path
        # Try to find from common paths
        workspace_candidates = [
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  # From this file
            os.getcwd(),
            os.path.expanduser('~/Documents/GitHub/hemibrain-connectomes-analysis-v3.1'),
        ]
        
        workspace_path = None
        for candidate in workspace_candidates:
            neuron_df_path = os.path.join(
                candidate, 'datasets', 'male-cns_v0_9', 
                'male-cns_v0_9_allneurons_neuron_df.csv'
            )
            if os.path.exists(neuron_df_path):
                workspace_path = candidate
                break
        
        if workspace_path is None:
            if self.verbose:
                print("\n⚠️ Auto type mapping: Could not find male-cns neuron_df file.")
                print("   Initialize male-cns dataset first, or disable auto_type_mapping.")
            self._auto_type_mapper = None
            return
        
        # Initialize type mapper
        self._auto_type_mapper = CrossDatasetTypeMapper(
            workspace_path=workspace_path,
            verbose=self.verbose,
        )
        
        if not self._auto_type_mapper.load():
            self._auto_type_mapper = None
            return
    
    def _resolve_neurons_with_auto_mapping(
        self, 
        neurons: List[Union[str, int]], 
        dataset: str
    ) -> List[Union[str, int]]:
        """
        Resolve neuron type names using auto type mapping.
        
        Args:
            neurons: List of neuron types/patterns/bodyIds
            dataset: Target dataset to resolve for
            
        Returns:
            List of resolved neuron types for the target dataset
        """
        if not self._auto_type_mapper:
            return neurons
        
        resolved = []
        for neuron in neurons:
            # Skip non-string identifiers (bodyIds)
            if not isinstance(neuron, str):
                resolved.append(neuron)
                continue
            
            # Skip regex patterns - pass through as-is
            if '*' in neuron or ('.' in neuron and '.*' in neuron):
                resolved.append(neuron)
                continue
            
            # Try to find mapping
            source_ds = self._auto_type_mapper._detect_type_source(neuron)
            if source_ds:
                mapped = self._auto_type_mapper.get_mapped_type(neuron, source_ds, dataset)
                if mapped:
                    resolved.append(mapped)
                else:
                    # No mapping found, use original (might be missing in target dataset)
                    resolved.append(neuron)
            else:
                # Type not found in any known dataset, pass through
                resolved.append(neuron)
        
        return resolved
    
    def _print_neuron_mapping_summary(
        self,
        neurons: List[Union[str, int]],
        datasets: List[str],
        indent: str = "  ",
    ) -> None:
        """
        Print mapping summary for source/target neurons.
        
        Shows:
        - Per-dataset resolved types (only if different from query)
        - N-to-1 warnings with recommendation
        - 1-to-N warnings with recommendation
        """
        if not self._auto_type_mapper:
            return
        
        # Get mapping summary
        summary = self._auto_type_mapper.get_source_target_mapping_summary(
            [n for n in neurons if isinstance(n, str)],
            datasets,
        )
        
        # Print different mappings
        if summary['different_mappings']:
            print(f"{indent}Auto-mapped types:")
            for type_name, mappings in summary['different_mappings']:
                # Show: type_name → dataset1:mapped1, dataset2:mapped2
                mapping_strs = []
                for ds, mapped in mappings.items():
                    if mapped != type_name:
                        # Use nickname if available
                        ds_idx = datasets.index(ds) if ds in datasets else -1
                        ds_label = self.datasets_nickname[ds_idx] if (
                            self.datasets_nickname and ds_idx >= 0 and ds_idx < len(self.datasets_nickname)
                        ) else ds
                        mapping_strs.append(f"{ds_label}:{mapped}")
                if mapping_strs:
                    print(f"{indent}  • {type_name} → {', '.join(mapping_strs)}")
        
        # Print N-to-1 warnings
        if summary['n_to_1_warnings']:
            print(f"\n{indent}⚠️  N-to-1 type mapping detected:")
            for type_name, src_ds, tgt_ds, conflict_types in summary['n_to_1_warnings']:
                # Determine which dataset is "A" (N types) and which is "B" (1 type)
                # The source_type in the conflict is the "1" side
                conflict_types_str = ", ".join(sorted(conflict_types))
                if type_name in conflict_types:
                    # User queried a type from the "N" side
                    # Recommend using the "1" type from target dataset
                    print(f"{indent}  • '{type_name}' is one of N types ({conflict_types_str}) in {tgt_ds}")
                    print(f"{indent}    that all map to 1 type in {src_ds}.")
                    print(f"{indent}    💡 Recommend: Query by the {src_ds} type name to compare aggregated results.")
                else:
                    # User queried the "1" type
                    print(f"{indent}  • '{type_name}' in {src_ds} maps to multiple types in {tgt_ds}:")
                    print(f"{indent}    ({conflict_types_str})")
                    print(f"{indent}    💡 Results from {tgt_ds} will aggregate these {len(conflict_types)} types.")
        
        # Print 1-to-N warnings
        if summary['one_to_n_warnings']:
            print(f"\n{indent}⚠️  1-to-N type mapping detected:")
            for type_name, src_ds, tgt_ds, split_types in summary['one_to_n_warnings']:
                split_types_str = ", ".join(sorted(split_types))
                print(f"{indent}  • '{type_name}' in {src_ds} splits into {len(split_types)} types in {tgt_ds}:")
                print(f"{indent}    ({split_types_str})")
                print(f"{indent}    💡 Results from {tgt_ds} will aggregate these {len(split_types)} types.")
                print(f"{indent}    To avoid aggregation, use custom LabelMapper or set auto_type_mapping=False.")
    
    def get_auto_type_mapper(self) -> Optional[Any]:
        """
        Get the auto type mapper instance if enabled.
        
        Returns:
            CrossDatasetTypeMapper instance or None
        """
        return self._auto_type_mapper
    
    def export_auto_mapping(self, output_path: Optional[str] = None) -> Optional[str]:
        """
        Export the auto type mapping used in this comparison.
        
        Args:
            output_path: Path to save CSV file. If None, saves to output folder.
            
        Returns:
            Path to saved file, or None if auto mapping not enabled.
        """
        if not self._auto_type_mapper:
            return None
        
        if output_path is None:
            output_path = os.path.join(
                self.full_output_path, 
                'auto_type_mapping.csv'
            )
        
        self._auto_type_mapper.export_mapping(output_path)
        return output_path
    
    def __repr__(self) -> str:
        datasets_str = self.get_dataset_names()
        return (f"ComparisonParameters(datasets={datasets_str}, "
                f"sources={len(self._ensure_flat_list(self.source_neurons))}, "
                f"targets={len(self._ensure_flat_list(self.target_neurons))}, "
                f"max_interlayer={self.max_interlayer}, "
                f"thresholds={self.thresholds})")
