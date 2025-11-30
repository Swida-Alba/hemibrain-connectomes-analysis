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
    
    top_edges: int = 50
    """Number of top edges for visualization focus"""
    
    # Connectivity Profile Verification Settings
    # These are used by run_connectivity_profile_verification() when called separately
    verification_direction: str = 'both'
    """Direction for profile verification: 'upstream', 'downstream', or 'both'"""
    
    verification_mode: str = 'loose'
    """Verification comparison mode: 'strict' or 'loose'.
    - 'strict': Only compare explicitly typed partners
    - 'loose': Include untyped partners in comparison"""
    
    verification_top_k: int = 5
    """Number of top partners to include per type for profile comparison"""
    
    verification_top_m: int = 0
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
    
    allow_single_dataset: bool = False
    """Allow single dataset for threshold sensitivity analysis only"""
    
    def __post_init__(self):
        """Validate and process parameters after initialization."""
        from .label_mapper import LabelMapper
        
        # Ensure datasets is a list
        if isinstance(self.datasets, str):
            self.datasets = [self.datasets]
        
        # Process source_neurons: handle LabelMapper, string, or list
        if isinstance(self.source_neurons, LabelMapper):
            # LabelMapper provides neurons per dataset - keep reference
            self._source_mapper = self.source_neurons
            self.source_neurons = []  # Will be resolved per dataset
        elif isinstance(self.source_neurons, str):
            self.source_neurons = [self.source_neurons]
        elif not self.source_neurons:
            self.source_neurons = []
        
        # Process target_neurons: same handling
        if isinstance(self.target_neurons, LabelMapper):
            self._target_mapper = self.target_neurons
            self.target_neurons = []
        elif isinstance(self.target_neurons, str):
            self.target_neurons = [self.target_neurons]
        elif not self.target_neurons:
            self.target_neurons = []
        
        # Initialize mapper references if not set
        if not hasattr(self, '_source_mapper'):
            self._source_mapper = None
        if not hasattr(self, '_target_mapper'):
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
        
        # Sort thresholds
        self.thresholds = sorted(self.thresholds)
        
        # Validate comparison_mode
        valid_modes = ['path', 'edge']
        if self.comparison_mode not in valid_modes:
            raise ValueError(f"comparison_mode must be one of {valid_modes}, got: {self.comparison_mode}")
        
        # Validate minimum requirements
        # Allow single dataset if allow_single_dataset=True (for threshold sensitivity analysis)
        if len(self.datasets) < 1:
            raise ValueError("At least 1 dataset is required")
        if len(self.datasets) < 2 and not self.allow_single_dataset:
            raise ValueError(
                "At least 2 datasets are required for cross-dataset comparison. "
                "Set allow_single_dataset=True for single-dataset threshold sensitivity analysis."
            )
        
        if not self.thresholds:
            raise ValueError("At least one threshold is required")
        
        # Validate neurons (unless using LabelMapper)
        if not self._source_mapper and not self.source_neurons:
            raise ValueError("source_neurons cannot be empty (provide list, nested list, or LabelMapper)")
        if not self._target_mapper and not self.target_neurons:
            raise ValueError("target_neurons cannot be empty (provide list, nested list, or LabelMapper)")
        
        # Cache the timestamp for consistent output_name
        self._cached_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
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
        
        If using LabelMapper, resolves neurons from mapper.
        Otherwise returns the shared source_neurons list.
        
        Args:
            dataset: Dataset identifier
            
        Returns:
            List of source neuron types/patterns/bodyIds
        """
        if self._source_mapper is not None:
            # Get neurons from LabelMapper for this dataset
            return self._source_mapper.get_all_neurons_for_dataset(dataset, 'source')
        return self._ensure_flat_list(self.source_neurons)
    
    def get_target_neurons_for_dataset(self, dataset: str) -> List[Union[str, int]]:
        """
        Get target neurons for a specific dataset.
        
        If using LabelMapper, resolves neurons from mapper.
        Otherwise returns the shared target_neurons list.
        
        Args:
            dataset: Dataset identifier
            
        Returns:
            List of target neuron types/patterns/bodyIds
        """
        if self._target_mapper is not None:
            return self._target_mapper.get_all_neurons_for_dataset(dataset, 'target')
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
                'version': '2.0'
            },
            'datasets': self.get_dataset_names(),
            'source_groups': source_groups,
            'target_groups': target_groups,
            'source_labels': self.source_labels,
            'target_labels': self.target_labels,
            'thresholds': self.thresholds,
            'max_interlayer': self.max_interlayer,
            'top_edges': self.top_edges,
            'analysis_settings': {
                'min_ratio': self._min_ratio,
                'min_prob': self._min_prob,
                'use_cache': self._use_cache,
                'output_format': self._output_format
            },
            'output_folder': self.output_folder,
            'saveas': self.saveas,
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
        return cls(
            datasets=data.get('datasets', []),
            source_neurons=data.get('source_neurons', []),
            target_neurons=data.get('target_neurons', []),
            max_interlayer=data.get('max_interlayer', 2),
            source_labels=data.get('source_labels', []),
            target_labels=data.get('target_labels', []),
            thresholds=data.get('thresholds', [1, 3, 5, 10, 20]),
            top_edges=data.get('top_edges', 50),
            output_folder=data.get('output_folder', '.'),
            saveas=data.get('saveas'),
            token=data.get('token', ''),
        )
    
    def __repr__(self) -> str:
        datasets_str = self.get_dataset_names()
        return (f"ComparisonParameters(datasets={datasets_str}, "
                f"sources={len(self._ensure_flat_list(self.source_neurons))}, "
                f"targets={len(self._ensure_flat_list(self.target_neurons))}, "
                f"max_interlayer={self.max_interlayer}, "
                f"thresholds={self.thresholds})")
