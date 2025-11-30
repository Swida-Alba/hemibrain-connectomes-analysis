"""
Connectivity Profile Extraction Module

This module provides tools for extracting and comparing connectivity profiles
(fingerprints) across neurons and datasets. A connectivity profile captures
the top upstream and downstream partners of a neuron, enabling cross-dataset
verification of neuron type assignments and homolog discovery.

Main components:
- ConnectivityProfile: Dataclass storing a neuron's connectivity fingerprint
- ProfilerConfig: Configuration for profile extraction
- FuzzyMatchConfig: Configuration for fuzzy partner type matching
- ConnectivityProfiler: Main class for extracting profiles from datasets

Example:
    >>> from src.comparison import ConnectivityProfiler, ProfilerConfig
    >>> 
    >>> config = ProfilerConfig(top_k_bodyid=10)
    >>> profiler = ConnectivityProfiler(
    ...     datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
    ...     config=config
    ... )
    >>> 
    >>> profile = profiler.get_profile('aMe12', 'hemibrain:v1.2.1')
    >>> print(profile.upstream_partners)
"""

import hashlib
import json
import os
import re
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class FuzzyMatchConfig:
    """
    Configuration for fuzzy partner type matching across datasets.
    
    Cross-dataset type names may have variations like:
    - T4a_R vs T4a vs T4a_L (laterality suffix)
    - aMe12_1 vs aMe12 (numeric suffix)
    - T4a-like vs T4a (qualitative suffix)
    
    This config controls how partner types are normalized for comparison.
    
    Attributes:
        enabled: Enable fuzzy matching (default: True)
        strip_lr_suffix: Remove _L, _R, _left, _right suffixes
        strip_numeric_suffix: Remove trailing numeric suffixes (_1, _2, etc.)
        strip_like_suffix: Remove -like or _like suffixes
        case_insensitive: Convert to lowercase for matching
        custom_mappings: User-defined type name mappings
    
    Example:
        >>> config = FuzzyMatchConfig(strip_lr_suffix=True)
        >>> normalize_partner_type('T4a_R', config)
        't4a'
    
    Note:
        Default is now DISABLED (exact matching) to prevent grouping of similar types.
        Types like R8, R7, Dm9 will be kept as-is instead of being stripped to 'r', 'dm'.
    """
    enabled: bool = False  # Disabled by default - use exact type matching
    strip_lr_suffix: bool = False  # Don't strip _L/_R suffixes
    strip_numeric_suffix: bool = False  # Don't strip trailing numbers (keeps R8, Dm9 etc.)
    strip_like_suffix: bool = False  # Don't strip -like suffixes
    case_insensitive: bool = False  # Case sensitive matching
    custom_mappings: Dict[str, str] = field(default_factory=dict)


@dataclass
class ProfilerConfig:
    """
    Configuration for ConnectivityProfiler.
    
    Round 5 additions:
    - top_k_bodyid: Top K connections by bodyId (default: 20)
    - top_m_type: Minimum unique types in profile (default: 5)
    - dynamic_expansion: Expand K until M types reached (default: True)
    - max_expansion_factor: Don't expand beyond K * factor (default: 5)
    
    Attributes:
        top_k_bodyid: Top K connections by bodyId (used for both upstream and downstream)
        top_m_type: Minimum unique types to ensure in profile (default: 5)
        min_synapse_threshold: Filter connections with fewer synapses (default: 3)
        include_untyped_partners: Include partners without type annotations (default: False)
        normalize_method: Weight normalization method ('proportion', 'rank', or 'both')
        use_cache: Enable profile caching to disk (default: True)
        fuzzy_match: Fuzzy matching configuration for partner type names
        dynamic_expansion: Dynamically expand K until M types reached (default: True)
        max_expansion_factor: Maximum expansion multiplier for K (default: 5)
    
    Example:
        >>> config = ProfilerConfig(
        ...     top_k_bodyid=20,
        ...     top_m_type=5,
        ...     min_synapse_threshold=5,
        ...     include_untyped_partners=False
        ... )
    """
    top_k_bodyid: int = 5  # Top K connections by bodyId (used for both upstream and downstream)
    top_m_type: int = 2  # Round 5: minimum unique types
    min_synapse_threshold: int = 3
    include_untyped_partners: bool = False
    normalize_method: str = 'both'  # 'proportion', 'rank', or 'both'
    use_cache: bool = True
    fuzzy_match: FuzzyMatchConfig = field(default_factory=FuzzyMatchConfig)
    dynamic_expansion: bool = True  # Round 5: expand K until M types
    max_expansion_factor: int = 5  # Round 5: max K expansion


# ============================================================================
# Connectivity Profile Dataclass
# ============================================================================

@dataclass
class ConnectivityProfile:
    """
    Connectivity fingerprint for a neuron or neuron group.
    
    Contains normalized weight distributions and rank information for
    upstream and downstream partners, enabling cross-dataset comparison.
    
    Round 5 additions:
    - unique_types_upstream/downstream: Count of unique types in profile
    - is_sparse: Flag if unique_types < top_m_type after expansion
    - partner_type_mapping: bodyId → type mapping for partners (metadata)
    - top_k_bodyid_used: Actual K used after dynamic expansion
    - top_m_type_target: Target M for unique types
    
    Attributes:
        neuron_id: Neuron type name, bodyId, or list of bodyIds
        dataset: Dataset identifier (e.g., 'hemibrain:v1.2.1')
        upstream_partners: Partner type → normalized weight (proportions)
        downstream_partners: Partner type → normalized weight (proportions)
        upstream_ranks: Partner type → rank (1 = strongest)
        downstream_ranks: Partner type → rank (1 = strongest)
        upstream_top_k: Number of top partners stored
        downstream_top_k: Number of top partners stored
        total_upstream_weight: Raw total weight (pre-normalization)
        total_downstream_weight: Raw total weight
        num_neurons_aggregated: Count of neurons if type-based aggregation
        upstream_weight_variance: Per-partner weight variance (for multi-neuron types)
        downstream_weight_variance: Per-partner weight variance
        untyped_upstream_count: Number of untyped upstream partners excluded
        untyped_downstream_count: Number of untyped downstream partners excluded
        untyped_upstream_weight_fraction: Fraction of weight from untyped partners
        untyped_downstream_weight_fraction: Fraction of weight from untyped partners
        actual_upstream_count: Actual number of partners found
        actual_downstream_count: Actual number of partners found
        is_weak_connectivity: Flag if < 5 partners in either direction
        unique_types_upstream: Count of unique types in upstream profile (Round 5)
        unique_types_downstream: Count of unique types in downstream profile (Round 5)
        is_sparse: Flag if unique_types < top_m_type after expansion (Round 5)
        partner_type_mapping_upstream: bodyId → type mapping for upstream partners (Round 5)
        partner_type_mapping_downstream: bodyId → type mapping for downstream partners (Round 5)
        top_k_bodyid_used: Actual K used after dynamic expansion (Round 5)
        top_m_type_target: Target M for unique types (Round 5)
    """
    neuron_id: Union[str, int, List]
    dataset: str
    
    # Partner data (typed partners only by default)
    upstream_partners: Dict[str, float] = field(default_factory=dict)
    downstream_partners: Dict[str, float] = field(default_factory=dict)
    
    # Rank-based representation
    upstream_ranks: Dict[str, int] = field(default_factory=dict)
    downstream_ranks: Dict[str, int] = field(default_factory=dict)
    
    # Metadata
    upstream_top_k: int = 10
    downstream_top_k: int = 10
    total_upstream_weight: float = 0.0
    total_downstream_weight: float = 0.0
    num_neurons_aggregated: int = 1
    
    # Variance for multi-neuron types
    upstream_weight_variance: Optional[Dict[str, float]] = None
    downstream_weight_variance: Optional[Dict[str, float]] = None
    
    # Untyped partner statistics
    untyped_upstream_count: int = 0
    untyped_downstream_count: int = 0
    untyped_upstream_weight_fraction: float = 0.0
    untyped_downstream_weight_fraction: float = 0.0
    
    # Sparse profile detection
    actual_upstream_count: int = 0
    actual_downstream_count: int = 0
    is_weak_connectivity: bool = False
    
    # Round 5: Unique type counts and sparse detection
    unique_types_upstream: int = 0
    unique_types_downstream: int = 0
    is_sparse: bool = False  # True if unique_types < top_m_type after expansion
    
    # Round 5: Partner type mapping (bodyId → type) for metadata
    partner_type_mapping_upstream: Optional[Dict[int, str]] = None
    partner_type_mapping_downstream: Optional[Dict[int, str]] = None
    
    # Round 5: Dynamic expansion tracking
    top_k_bodyid_used: int = 20  # Actual K used after expansion
    top_m_type_target: int = 5  # Target M for unique types
    
    # Minimum partners threshold for weak connectivity warning
    MIN_PARTNERS_THRESHOLD: int = field(default=5, repr=False)
    
    def __post_init__(self):
        """Check for weak connectivity and sparse profiles."""
        # Weak connectivity check - record but don't warn
        # The is_weak_connectivity flag is recorded in the profile for later use
        if (self.actual_upstream_count < self.MIN_PARTNERS_THRESHOLD or 
            self.actual_downstream_count < self.MIN_PARTNERS_THRESHOLD):
            self.is_weak_connectivity = True
            # Suppressed warning - recorded in profile.is_weak_connectivity instead
            # Callers should check this flag and handle appropriately
        
        # Sparse profile check (Round 5)
        if self.unique_types_upstream > 0 or self.unique_types_downstream > 0:
            if (self.unique_types_upstream < self.top_m_type_target or 
                self.unique_types_downstream < self.top_m_type_target):
                self.is_sparse = True
    
    def to_dict(self) -> dict:
        """Convert profile to serializable dictionary."""
        # Helper to convert numpy types to Python native types
        def to_native(val):
            if hasattr(val, 'item'):
                return val.item()  # numpy scalar -> Python scalar
            return val
        
        result = {
            'neuron_id': str(self.neuron_id) if not isinstance(self.neuron_id, str) else self.neuron_id,
            'dataset': self.dataset,
            'upstream_partners': {str(k): float(v) for k, v in self.upstream_partners.items()},
            'downstream_partners': {str(k): float(v) for k, v in self.downstream_partners.items()},
            'upstream_ranks': {str(k): int(v) for k, v in self.upstream_ranks.items()},
            'downstream_ranks': {str(k): int(v) for k, v in self.downstream_ranks.items()},
            'upstream_top_k': int(self.upstream_top_k),
            'downstream_top_k': int(self.downstream_top_k),
            'total_upstream_weight': to_native(self.total_upstream_weight),
            'total_downstream_weight': to_native(self.total_downstream_weight),
            'num_neurons_aggregated': int(self.num_neurons_aggregated),
            'upstream_weight_variance': to_native(self.upstream_weight_variance) if self.upstream_weight_variance else None,
            'downstream_weight_variance': to_native(self.downstream_weight_variance) if self.downstream_weight_variance else None,
            'untyped_upstream_count': int(self.untyped_upstream_count),
            'untyped_downstream_count': int(self.untyped_downstream_count),
            'untyped_upstream_weight_fraction': float(self.untyped_upstream_weight_fraction),
            'untyped_downstream_weight_fraction': float(self.untyped_downstream_weight_fraction),
            'actual_upstream_count': int(self.actual_upstream_count),
            'actual_downstream_count': int(self.actual_downstream_count),
            'is_weak_connectivity': bool(self.is_weak_connectivity),
            # Round 5 additions
            'unique_types_upstream': int(self.unique_types_upstream),
            'unique_types_downstream': int(self.unique_types_downstream),
            'is_sparse': bool(self.is_sparse),
            'top_k_bodyid_used': int(self.top_k_bodyid_used),
            'top_m_type_target': int(self.top_m_type_target),
        }
        
        # Add partner type mappings if present
        if self.partner_type_mapping_upstream:
            result['partner_type_mapping_upstream'] = {str(k): str(v) for k, v in self.partner_type_mapping_upstream.items()}
        if self.partner_type_mapping_downstream:
            result['partner_type_mapping_downstream'] = {str(k): str(v) for k, v in self.partner_type_mapping_downstream.items()}
        
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ConnectivityProfile':
        """Create profile from dictionary."""
        # Remove the non-init field before creating instance
        data = data.copy()
        data.pop('MIN_PARTNERS_THRESHOLD', None)
        
        # Convert partner_type_mapping keys back to int if present
        if 'partner_type_mapping_upstream' in data and data['partner_type_mapping_upstream']:
            data['partner_type_mapping_upstream'] = {
                int(k): v for k, v in data['partner_type_mapping_upstream'].items()
            }
        if 'partner_type_mapping_downstream' in data and data['partner_type_mapping_downstream']:
            data['partner_type_mapping_downstream'] = {
                int(k): v for k, v in data['partner_type_mapping_downstream'].items()
            }
        
        return cls(**data)
    
    def get_all_partner_types(self) -> set:
        """Get set of all partner types (upstream + downstream)."""
        return set(self.upstream_partners.keys()) | set(self.downstream_partners.keys())
    
    def get_partner_weight(self, partner_type: str, direction: str = 'both') -> float:
        """Get normalized weight for a partner type."""
        up = self.upstream_partners.get(partner_type, 0.0)
        down = self.downstream_partners.get(partner_type, 0.0)
        
        if direction == 'upstream':
            return up
        elif direction == 'downstream':
            return down
        else:
            return up + down
    
    def get_partner_bodyids(self, partner_type: str, direction: str = 'upstream') -> List[int]:
        """Get bodyIds for a specific partner type (Round 5)."""
        mapping = (self.partner_type_mapping_upstream if direction == 'upstream' 
                   else self.partner_type_mapping_downstream)
        if not mapping:
            return []
        return [bid for bid, ptype in mapping.items() if ptype == partner_type]
    
    def summary(self) -> str:
        """Generate human-readable summary of the profile."""
        lines = [
            f"ConnectivityProfile for {self.neuron_id} ({self.dataset})",
            f"  Neurons aggregated: {self.num_neurons_aggregated}",
            f"  Upstream partners: {self.actual_upstream_count} (top-{self.upstream_top_k} stored, {self.unique_types_upstream} unique types)",
            f"  Downstream partners: {self.actual_downstream_count} (top-{self.downstream_top_k} stored, {self.unique_types_downstream} unique types)",
        ]
        
        if self.is_weak_connectivity:
            lines.append("  ⚠️ WEAK CONNECTIVITY: May have unreliable comparisons")
        
        if self.is_sparse:
            lines.append(f"  ⚠️ SPARSE PROFILE: < {self.top_m_type_target} unique types in one or both directions")
        
        if self.untyped_upstream_count > 0 or self.untyped_downstream_count > 0:
            lines.append(f"  Untyped excluded: {self.untyped_upstream_count} up, {self.untyped_downstream_count} down")
            lines.append(f"  Untyped weight fraction: {self.untyped_upstream_weight_fraction:.1%} up, {self.untyped_downstream_weight_fraction:.1%} down")
        
        lines.append(f"  Top upstream: {list(self.upstream_partners.keys())[:5]}")
        lines.append(f"  Top downstream: {list(self.downstream_partners.keys())[:5]}")
        
        # Round 5: Show expansion info
        lines.append(f"  Dynamic expansion: K={self.top_k_bodyid_used} bodyIds, M={self.top_m_type_target} target types")
        
        return "\n".join(lines)
    
    # ========================================================================
    # Vectorization Methods (Round 2)
    # ========================================================================
    
    def to_weight_vector(
        self,
        vocabulary: List[str],
        direction: str = 'both',
        normalize: bool = True
    ) -> np.ndarray:
        """
        Convert connectivity profile to a normalized weight vector.
        
        Args:
            vocabulary: Ordered list of partner types (shared across datasets)
            direction: 'upstream', 'downstream', or 'both'
            normalize: If True, L2-normalize the vector
        
        Returns:
            numpy array of weights, where position i corresponds to vocabulary[i]
            Missing partners have weight 0.
        
        Example:
            >>> vocab = ['L1', 'L2', 'L3', 'Mi1', 'Tm1']
            >>> profile.to_weight_vector(vocab, 'upstream')
            array([0.32, 0.25, 0.18, 0.0, 0.0])
        """
        if direction == 'upstream':
            partners = self.upstream_partners
        elif direction == 'downstream':
            partners = self.downstream_partners
        else:  # both - sum upstream and downstream weights for each partner
            # Sum weights from both directions (consistent with ProfileComparator)
            up_vec = self.to_weight_vector(vocabulary, 'upstream', normalize=False)
            down_vec = self.to_weight_vector(vocabulary, 'downstream', normalize=False)
            combined = up_vec + down_vec
            if normalize and np.linalg.norm(combined) > 0:
                combined = combined / np.linalg.norm(combined)
            return combined
        
        # Build vector for single direction
        vector = np.array([partners.get(p, 0.0) for p in vocabulary])
        
        if normalize and np.linalg.norm(vector) > 0:
            vector = vector / np.linalg.norm(vector)
        
        return vector
    
    def to_rank_vector(
        self,
        vocabulary: List[str],
        direction: str = 'both',
        default_rank: Optional[int] = None
    ) -> np.ndarray:
        """
        Convert connectivity profile to a rank vector.
        
        Args:
            vocabulary: Ordered list of partner types (shared across datasets)
            direction: 'upstream', 'downstream', or 'both'
            default_rank: Rank to assign for missing partners.
                         If None, uses len(profile_partners) + 1
        
        Returns:
            numpy array of ranks, where position i corresponds to vocabulary[i]
            Missing partners get default_rank (low priority).
        
        Example:
            >>> vocab = ['L1', 'L2', 'L3', 'Mi1', 'Tm1']
            >>> profile.to_rank_vector(vocab, 'upstream')
            array([1, 2, 3, 6, 6])  # L1 is rank 1, Mi1/Tm1 missing (default 6)
        """
        if direction == 'upstream':
            ranks = self.upstream_ranks
            k = len(self.upstream_partners)
        elif direction == 'downstream':
            ranks = self.downstream_ranks
            k = len(self.downstream_partners)
        else:  # both - average upstream and downstream ranks for each partner
            up_vec = self.to_rank_vector(vocabulary, 'upstream', default_rank)
            down_vec = self.to_rank_vector(vocabulary, 'downstream', default_rank)
            # Average ranks from both directions
            return (up_vec + down_vec) / 2.0
        
        # Default rank is K+1 (one past the last partner)
        actual_default = default_rank if default_rank is not None else (k + 1)
        
        # Build vector
        vector = np.array([ranks.get(p, actual_default) for p in vocabulary])
        
        return vector
    
    @staticmethod
    def build_shared_vocabulary(
        profiles: Dict[str, 'ConnectivityProfile'],
        direction: str = 'both',
        min_occurrence: int = 1
    ) -> List[str]:
        """
        Build a shared vocabulary of partner types across multiple profiles.
        
        Args:
            profiles: Dict of {profile_id: ConnectivityProfile}
            direction: 'upstream', 'downstream', or 'both'
            min_occurrence: Minimum number of profiles a partner must appear in
        
        Returns:
            Sorted list of partner types appearing in at least min_occurrence profiles
        
        Example:
            >>> profiles = {'hb_Mi1': profile1, 'fafb_Mi1': profile2}
            >>> vocab = ConnectivityProfile.build_shared_vocabulary(profiles, 'upstream')
            ['L1', 'L2', 'L3', 'L4', 'L5', 'Tm1', 'Tm2']
        """
        partner_counts: Dict[str, int] = {}
        
        for profile in profiles.values():
            if direction in ('upstream', 'both'):
                for partner in profile.upstream_partners.keys():
                    partner_counts[partner] = partner_counts.get(partner, 0) + 1
            if direction in ('downstream', 'both'):
                for partner in profile.downstream_partners.keys():
                    partner_counts[partner] = partner_counts.get(partner, 0) + 1
        
        # Filter by minimum occurrence and sort
        vocabulary = [p for p, count in partner_counts.items() if count >= min_occurrence]
        return sorted(vocabulary)
    
    @staticmethod
    def profiles_to_weight_matrix(
        profiles: Dict[str, 'ConnectivityProfile'],
        vocabulary: Optional[List[str]] = None,
        direction: str = 'both',
        normalize: bool = True
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Convert multiple profiles to a weight matrix.
        
        Args:
            profiles: Dict of {profile_id: ConnectivityProfile}
            vocabulary: Optional shared vocabulary. If None, built from profiles.
            direction: 'upstream', 'downstream', or 'both'
            normalize: L2-normalize each row
        
        Returns:
            Tuple of:
            - matrix: (n_profiles, n_vocabulary) numpy array
            - profile_ids: List of profile IDs (row order)
            - vocabulary: List of partner types (column order)
        
        Example:
            >>> matrix, ids, vocab = ConnectivityProfile.profiles_to_weight_matrix(profiles)
            >>> matrix.shape
            (5, 20)  # 5 profiles, 20 unique partner types
        """
        if vocabulary is None:
            vocabulary = ConnectivityProfile.build_shared_vocabulary(profiles, direction)
        
        profile_ids = list(profiles.keys())
        matrix = np.zeros((len(profile_ids), len(vocabulary)))
        
        for i, pid in enumerate(profile_ids):
            matrix[i] = profiles[pid].to_weight_vector(vocabulary, direction, normalize)
        
        return matrix, profile_ids, vocabulary
    
    @staticmethod
    def profiles_to_rank_matrix(
        profiles: Dict[str, 'ConnectivityProfile'],
        vocabulary: Optional[List[str]] = None,
        direction: str = 'both',
        default_rank: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Convert multiple profiles to a rank matrix.
        
        Args:
            profiles: Dict of {profile_id: ConnectivityProfile}
            vocabulary: Optional shared vocabulary. If None, built from profiles.
            direction: 'upstream', 'downstream', or 'both'
            default_rank: Rank for missing partners. If None, uses profile's K+1.
        
        Returns:
            Tuple of:
            - matrix: (n_profiles, n_vocabulary) numpy array of ranks
            - profile_ids: List of profile IDs (row order)
            - vocabulary: List of partner types (column order)
        
        Example:
            >>> matrix, ids, vocab = ConnectivityProfile.profiles_to_rank_matrix(profiles)
            >>> # Compute pairwise Spearman correlations efficiently
            >>> from scipy.stats import spearmanr
            >>> corr_matrix, _ = spearmanr(matrix, axis=1)
        """
        if vocabulary is None:
            vocabulary = ConnectivityProfile.build_shared_vocabulary(profiles, direction)
        
        profile_ids = list(profiles.keys())
        matrix = np.zeros((len(profile_ids), len(vocabulary)))
        
        for i, pid in enumerate(profile_ids):
            matrix[i] = profiles[pid].to_rank_vector(vocabulary, direction, default_rank)
        
        return matrix, profile_ids, vocabulary


# ============================================================================
# Helper Functions
# ============================================================================

def normalize_partner_type(type_name: str, config: FuzzyMatchConfig) -> str:
    """
    Normalize partner type name for cross-dataset comparison.
    
    Applies fuzzy matching rules to handle naming variations:
    - T4a_R → t4a (strip laterality suffix + lowercase)
    - aMe12_1 → ame12 (strip numeric suffix + lowercase)
    - T4a-like → t4a (strip like suffix)
    
    Args:
        type_name: Original type name
        config: FuzzyMatchConfig with matching rules
    
    Returns:
        Normalized type name
    """
    if not config.enabled or pd.isna(type_name) or type_name is None:
        return str(type_name) if type_name else ''
    
    result = str(type_name)
    
    # Apply custom mappings first (exact match on original)
    if result in config.custom_mappings:
        result = config.custom_mappings[result]
    
    # Case normalization
    if config.case_insensitive:
        result = result.lower()
    
    # Strip laterality suffixes: _L, _R, _left, _right, -L, -R
    if config.strip_lr_suffix:
        result = re.sub(r'[_-]?([LR]|left|right)$', '', result, flags=re.IGNORECASE)
    
    # Strip numeric suffixes: _1, _2, -1, -2, etc.
    if config.strip_numeric_suffix:
        result = re.sub(r'[_-]?\d+$', '', result)
    
    # Strip "-like" or "_like" suffixes
    if config.strip_like_suffix:
        result = re.sub(r'[_-]?like$', '', result, flags=re.IGNORECASE)
    
    # Apply custom mappings again after normalization (for mapped normalized forms)
    if result in config.custom_mappings:
        result = config.custom_mappings[result]
    
    return result


def compute_ranks(weights: Dict[str, float]) -> Dict[str, int]:
    """
    Compute ranks from weight dictionary.
    
    Args:
        weights: Partner type → weight mapping
    
    Returns:
        Partner type → rank mapping (1 = strongest)
    """
    if not weights:
        return {}
    
    # Sort by weight descending
    sorted_partners = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    
    # Assign ranks (handle ties by giving same rank)
    ranks = {}
    current_rank = 1
    prev_weight = None
    
    for i, (partner, weight) in enumerate(sorted_partners):
        if prev_weight is not None and weight < prev_weight:
            current_rank = i + 1
        ranks[partner] = current_rank
        prev_weight = weight
    
    return ranks


# ============================================================================
# ConnectivityProfiler Class
# ============================================================================

class ConnectivityProfiler:
    """
    Extract and manage connectivity profiles across datasets.
    
    This class queries neuron connectivity from NeuPrint or local datasets,
    normalizes the data, and produces ConnectivityProfile objects for
    cross-dataset comparison.
    
    Features:
    - Support for neuron types (str), bodyIds (int), or lists of bodyIds
    - Multi-neuron type aggregation with variance computation
    - Configurable filtering (min synapses, exclude untyped)
    - Profile caching to disk
    - Fuzzy partner type matching
    
    Example:
        >>> profiler = ConnectivityProfiler(
        ...     datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
        ...     config=ProfilerConfig(top_k_bodyid=10)
        ... )
        >>> profile = profiler.get_profile('aMe12', 'hemibrain:v1.2.1')
        >>> print(profile.upstream_partners)
    """
    
    def __init__(
        self,
        datasets: List[str],
        client: Optional[Any] = None,
        config: Optional[ProfilerConfig] = None,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize ConnectivityProfiler.
        
        Args:
            datasets: List of dataset identifiers to work with
            client: Optional NeuPrint client (will be created if needed)
            config: ProfilerConfig with extraction settings
            cache_dir: Custom cache directory (default: cache/connectivity_profiles)
            token: NeuPrint API token (used when creating clients)
            verbose: Print progress messages
        """
        self.datasets = datasets
        self.client = client
        self.config = config or ProfilerConfig()
        self.token = token
        self.verbose = verbose
        
        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # Use project-level cache folder
            src_dir = Path(__file__).parent.parent
            project_root = src_dir.parent
            self.cache_dir = project_root / 'cache' / 'connectivity_profiles'
        
        if self.config.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._memory_cache: Dict[Tuple[str, str], ConnectivityProfile] = {}
        
        # Client cache per dataset
        self._clients: Dict[str, Any] = {}
    
    def _log(self, message: str, level: str = 'info'):
        """Print message if verbose mode enabled.
        
        Args:
            message: Message to print
            level: 'info', 'debug', or 'warning'. Debug messages are suppressed.
        """
        if not self.verbose:
            return
        # Only print warnings and key info messages, suppress debug-level messages
        if level == 'debug':
            return  # Suppress verbose loading/joining messages
        print(f"[ConnectivityProfiler] {message}")
    
    def _get_config_hash(self) -> str:
        """Get hash of current configuration for cache invalidation."""
        config_dict = {
            'top_k_bodyid': self.config.top_k_bodyid,
            'min_synapse_threshold': self.config.min_synapse_threshold,
            'include_untyped_partners': self.config.include_untyped_partners,
            'normalize_method': self.config.normalize_method,
            'fuzzy_enabled': self.config.fuzzy_match.enabled,
            'fuzzy_strip_lr': self.config.fuzzy_match.strip_lr_suffix,
            'fuzzy_strip_numeric': self.config.fuzzy_match.strip_numeric_suffix,
        }
        return hashlib.md5(
            json.dumps(config_dict, sort_keys=True).encode()
        ).hexdigest()[:8]
    
    def _get_cache_path(self, neuron_id: Union[str, int], dataset: str) -> Path:
        """Get cache file path for a profile."""
        safe_dataset = dataset.replace(':', '_').replace('.', '_')
        safe_neuron = str(neuron_id).replace('/', '_').replace('\\', '_')
        config_hash = self._get_config_hash()
        
        return self.cache_dir / safe_dataset / f"{safe_neuron}_{config_hash}.json"
    
    def _load_from_cache(self, neuron_id: Union[str, int], dataset: str) -> Optional[ConnectivityProfile]:
        """Load profile from cache if exists."""
        # Check memory cache first
        cache_key = (str(neuron_id), dataset)
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        
        # Check disk cache
        if not self.config.use_cache:
            return None
        
        cache_path = self._get_cache_path(neuron_id, dataset)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                profile = ConnectivityProfile.from_dict(data)
                self._memory_cache[cache_key] = profile
                return profile
            except Exception as e:
                self._log(f"Warning: Could not load cache for {neuron_id}: {e}")
        
        return None
    
    def _save_to_cache(self, profile: ConnectivityProfile):
        """Save profile to cache."""
        # Memory cache
        cache_key = (str(profile.neuron_id), profile.dataset)
        self._memory_cache[cache_key] = profile
        
        # Disk cache
        if not self.config.use_cache:
            return
        
        cache_path = self._get_cache_path(profile.neuron_id, profile.dataset)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)
        except Exception as e:
            self._log(f"Warning: Could not save cache for {profile.neuron_id}: {e}")
    
    def _normalize_neuprint_dataset_name(self, dataset: str) -> str:
        """
        Convert dataset folder name to NeuPrint dataset name.
        
        Examples:
            hemibrain_v1_2_1 → hemibrain:v1.2.1
            male-cns_v0_9 → male-cns:v0.9
            optic-lobe_v1_1 → optic-lobe:v1.1
        """
        # If already in NeuPrint format, return as-is
        if ':' in dataset:
            return dataset
        
        # Convert underscores back to proper format
        # Pattern: name_vX_Y_Z → name:vX.Y.Z
        import re
        match = re.match(r'^(.+?)_v(\d+(?:_\d+)*)$', dataset)
        if match:
            name = match.group(1).replace('_', '-')  # hemibrain_v1_2_1 → hemibrain
            version = match.group(2).replace('_', '.')  # 1_2_1 → 1.2.1
            return f"{name}:v{version}"
        
        return dataset
    
    def _get_client_for_dataset(self, dataset: str) -> Optional[Any]:
        """Get or create NeuPrint client for a dataset."""
        if dataset in self._clients:
            return self._clients[dataset]
        
        # Check if dataset is local (FlyWire/FAFB/BANC)
        dataset_lower = dataset.lower()
        if 'flywire' in dataset_lower or 'fafb' in dataset_lower or 'banc' in dataset_lower:
            # Local dataset - no client needed
            self._clients[dataset] = None
            return None
        
        # Create NeuPrint client
        if self.client is not None:
            self._clients[dataset] = self.client
            return self.client
        
        try:
            from neuprint import Client
            
            # Use provided token or fall back to environment variable
            token = self.token or os.environ.get('NEUPRINT_APPLICATION_CREDENTIALS', '')
            if not token:
                self._log(f"Warning: No NeuPrint token found for {dataset}")
            
            # Normalize dataset name for NeuPrint API
            neuprint_dataset = self._normalize_neuprint_dataset_name(dataset)
            client = Client('neuprint.janelia.org', dataset=neuprint_dataset, token=token)
            self._clients[dataset] = client
            return client
        except Exception as e:
            self._log(f"Warning: Could not create client for {dataset}: {e}")
            return None
    
    def _join_type_info_from_neuron_df(
        self,
        conn_df: pd.DataFrame,
        dataset_path: Path,
        safe_name: str
    ) -> pd.DataFrame:
        """
        Join type information from neuron_df to connection DataFrame.
        
        This is needed for FlyWire/FAFB/BANC datasets where merged_connections
        files don't include type columns. Type info must be looked up from
        the neuron_df file.
        
        Args:
            conn_df: Connection DataFrame with bodyId_pre, bodyId_post columns
            dataset_path: Path to dataset folder
            safe_name: Sanitized dataset name
        
        Returns:
            conn_df with type_pre and type_post columns added
        """
        # Try to load neuron_df
        neuron_files = [
            dataset_path / f'{safe_name}_allneurons_neuron_df.parquet',
            dataset_path / f'{safe_name}_allneurons_neuron_df.csv',
            dataset_path / f'{safe_name}_neuron_df.parquet',
            dataset_path / f'{safe_name}_neuron_df.csv',
            dataset_path / 'neuron_df.parquet',
            dataset_path / 'neuron_df.csv',
        ]
        
        neuron_df = None
        for neuron_file in neuron_files:
            if neuron_file.exists():
                try:
                    if str(neuron_file).endswith('.parquet'):
                        neuron_df = pd.read_parquet(neuron_file)
                    else:
                        neuron_df = pd.read_csv(neuron_file)
                    self._log(f"Loaded neuron info from {neuron_file}", level='debug')
                    break
                except Exception as e:
                    self._log(f"Warning: Could not load {neuron_file}: {e}")
        
        if neuron_df is None or neuron_df.empty:
            self._log("Warning: No neuron_df found to join type information")
            return conn_df
        
        # Determine type column name in neuron_df
        # IMPORTANT: 'type' should be checked BEFORE 'cell_type' because:
        # - FAFB/FlyWire 'type' has specific types like 'aMe12', 'PPL101'
        # - FAFB/FlyWire 'cell_type' has generic categories like 't5_neuron', 'transmedullary'
        type_col = None
        for col in ['type', 'Type', 'cellType', 'cell_type']:
            if col in neuron_df.columns:
                type_col = col
                break
        
        if type_col is None:
            self._log("Warning: No type column found in neuron_df")
            return conn_df
        
        # Standardize bodyId column
        if 'bodyId' not in neuron_df.columns:
            for col in ['root_id', 'pt_root_id', 'body_id']:
                if col in neuron_df.columns:
                    neuron_df = neuron_df.rename(columns={col: 'bodyId'})
                    break
        
        if 'bodyId' not in neuron_df.columns:
            self._log("Warning: No bodyId column found in neuron_df")
            return conn_df
        
        # Create type lookup - ensure bodyId types match
        neuron_df['bodyId'] = neuron_df['bodyId'].astype(str)
        type_lookup = neuron_df.set_index('bodyId')[type_col].to_dict()
        
        # Ensure connection bodyIds are strings too
        conn_df['bodyId_pre'] = conn_df['bodyId_pre'].astype(str)
        conn_df['bodyId_post'] = conn_df['bodyId_post'].astype(str)
        
        # Map types
        conn_df['type_pre'] = conn_df['bodyId_pre'].map(type_lookup)
        conn_df['type_post'] = conn_df['bodyId_post'].map(type_lookup)
        
        # Log stats
        typed_pre = conn_df['type_pre'].notna().sum()
        typed_post = conn_df['type_post'].notna().sum()
        total = len(conn_df)
        self._log(f"Joined type info: {typed_pre}/{total} pre-types, {typed_post}/{total} post-types", level='debug')
        
        return conn_df
    
    def _query_connections_neuprint(
        self,
        neuron: Union[str, int, List],
        dataset: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Query upstream and downstream connections from NeuPrint.
        
        Returns:
            Tuple of (upstream_df, downstream_df)
            Each has columns: partner_bodyId, partner_type, weight
        """
        client = self._get_client_for_dataset(dataset)
        if client is None:
            self._log(f"ERROR: No NeuPrint client available for {dataset}. "
                     f"Set NEUPRINT_APPLICATION_CREDENTIALS env variable or provide token.")
            return pd.DataFrame(), pd.DataFrame()
        
        min_syn = self.config.min_synapse_threshold
        
        # Build neuron condition
        if isinstance(neuron, str):
            # Type-based query with regex support
            if '.*' in neuron or '*' in neuron:
                neuron_cond = f"n.type =~ '{neuron}'"
            else:
                neuron_cond = f"n.type = '{neuron}'"
        elif isinstance(neuron, int):
            neuron_cond = f"n.bodyId = {neuron}"
        elif isinstance(neuron, list):
            bodyids_str = ', '.join(str(b) for b in neuron)
            neuron_cond = f"n.bodyId IN [{bodyids_str}]"
        else:
            raise ValueError(f"Unsupported neuron type: {type(neuron)}")
        
        # Query upstream (inputs to the neuron)
        upstream_query = f"""
        MATCH (pre:Neuron)-[c:ConnectsTo]->(n:Neuron)
        WHERE {neuron_cond} AND c.weight >= {min_syn}
        RETURN pre.bodyId AS partner_bodyId, pre.type AS partner_type, 
               n.bodyId AS neuron_bodyId, c.weight AS weight
        """
        
        # Query downstream (outputs from the neuron)
        downstream_query = f"""
        MATCH (n:Neuron)-[c:ConnectsTo]->(post:Neuron)
        WHERE {neuron_cond} AND c.weight >= {min_syn}
        RETURN post.bodyId AS partner_bodyId, post.type AS partner_type,
               n.bodyId AS neuron_bodyId, c.weight AS weight
        """
        
        try:
            upstream_df = client.fetch_custom(upstream_query)
            downstream_df = client.fetch_custom(downstream_query)
            return upstream_df, downstream_df
        except Exception as e:
            self._log(f"Warning: Query failed for {neuron} in {dataset}: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def _query_connections_local(
        self,
        neuron: Union[str, int, List],
        dataset: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Query upstream and downstream connections from local dataset files.
        
        Returns:
            Tuple of (upstream_df, downstream_df)
        """
        # Find dataset folder
        src_dir = Path(__file__).parent.parent
        project_root = src_dir.parent
        datasets_folder = project_root / 'datasets'
        
        safe_name = dataset.replace(':', '_').replace('.', '_')
        dataset_path = datasets_folder / safe_name
        
        # Try to load connections file - check multiple naming conventions
        # Priority order: merged_connections > connections > generic
        conn_files = [
            # Merged connections format (used by FlyWire FAFB/BANC converters)
            dataset_path / f'{safe_name}_merged_connections.parquet',
            dataset_path / f'{safe_name}_merged_connections.csv',
            # Standard connections format
            dataset_path / f'{safe_name}_connections.parquet',
            dataset_path / f'{safe_name}_connections.csv',
            # Generic fallback
            dataset_path / 'connections.parquet',
            dataset_path / 'connections.csv',
        ]
        
        conn_df = None
        for conn_file in conn_files:
            if conn_file.exists():
                try:
                    if str(conn_file).endswith('.parquet'):
                        conn_df = pd.read_parquet(conn_file)
                    else:
                        conn_df = pd.read_csv(conn_file)
                    self._log(f"Loaded connections from {conn_file}", level='debug')
                    break
                except Exception as e:
                    self._log(f"Warning: Could not load {conn_file}: {e}")
        
        if conn_df is None or conn_df.empty:
            self._log(f"Warning: No connection data found for {dataset}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Standardize column names
        col_mapping = {
            'pre_pt_root_id': 'bodyId_pre',
            'post_pt_root_id': 'bodyId_post',
            'pre_type': 'type_pre',
            'post_type': 'type_post',
            'syn_count': 'weight',
        }
        conn_df = conn_df.rename(columns={k: v for k, v in col_mapping.items() if k in conn_df.columns})
        
        # If type columns missing, try to join from neuron_df
        if 'type_pre' not in conn_df.columns or 'type_post' not in conn_df.columns:
            conn_df = self._join_type_info_from_neuron_df(conn_df, dataset_path, safe_name)
        
        # Ensure required columns exist
        if 'weight' not in conn_df.columns:
            if 'syn_count' in conn_df.columns:
                conn_df['weight'] = conn_df['syn_count']
            else:
                conn_df['weight'] = 1
        
        # Filter by minimum synapses
        conn_df = conn_df[conn_df['weight'] >= self.config.min_synapse_threshold]
        
        # Build neuron mask
        if isinstance(neuron, str):
            # Type-based query
            if '.*' in neuron or '*' in neuron:
                pattern = neuron.replace('.*', '.*').replace('*', '.*')
                if 'type_pre' in conn_df.columns:
                    mask_up = conn_df['type_post'].astype(str).str.match(pattern, na=False)
                    mask_down = conn_df['type_pre'].astype(str).str.match(pattern, na=False)
                else:
                    return pd.DataFrame(), pd.DataFrame()
            else:
                if 'type_pre' in conn_df.columns:
                    mask_up = conn_df['type_post'] == neuron
                    mask_down = conn_df['type_pre'] == neuron
                else:
                    return pd.DataFrame(), pd.DataFrame()
        elif isinstance(neuron, int):
            if 'bodyId_post' in conn_df.columns:
                mask_up = conn_df['bodyId_post'] == neuron
                mask_down = conn_df['bodyId_pre'] == neuron
            else:
                return pd.DataFrame(), pd.DataFrame()
        elif isinstance(neuron, list):
            if 'bodyId_post' in conn_df.columns:
                mask_up = conn_df['bodyId_post'].isin(neuron)
                mask_down = conn_df['bodyId_pre'].isin(neuron)
            else:
                return pd.DataFrame(), pd.DataFrame()
        else:
            raise ValueError(f"Unsupported neuron type: {type(neuron)}")
        
        # Extract upstream connections
        upstream = conn_df[mask_up].copy()
        if not upstream.empty:
            upstream = upstream.rename(columns={
                'bodyId_pre': 'partner_bodyId',
                'type_pre': 'partner_type',
                'bodyId_post': 'neuron_bodyId',
            })
            upstream = upstream[['partner_bodyId', 'partner_type', 'neuron_bodyId', 'weight']]
        
        # Extract downstream connections
        downstream = conn_df[mask_down].copy()
        if not downstream.empty:
            downstream = downstream.rename(columns={
                'bodyId_post': 'partner_bodyId',
                'type_post': 'partner_type',
                'bodyId_pre': 'neuron_bodyId',
            })
            downstream = downstream[['partner_bodyId', 'partner_type', 'neuron_bodyId', 'weight']]
        
        return upstream, downstream
    
    def _process_connections(
        self,
        conn_df: pd.DataFrame,
        direction: str,
        top_k: int
    ) -> Tuple[Dict[str, float], Dict[str, int], int, int, float, int, int, Dict[int, str], int]:
        """
        Process connection DataFrame into normalized weights and ranks.
        
        Round 5: Added dynamic expansion to ensure minimum unique types,
        and bodyId→type mapping for metadata.
        
        Args:
            conn_df: DataFrame with partner_type, weight columns
            direction: 'upstream' or 'downstream' (for logging)
            top_k: Number of top partners to keep
        
        Returns:
            Tuple of (partners_dict, ranks_dict, untyped_count, 
                     untyped_weight_fraction, total_weight, actual_count,
                     unique_types, type_mapping, k_used)
        """
        if conn_df.empty:
            return {}, {}, 0, 0.0, 0.0, 0, 0, {}, top_k
        
        # Ensure partner_type column exists
        if 'partner_type' not in conn_df.columns:
            return {}, {}, 0, 0.0, 0.0, 0, 0, {}, top_k
        
        # Track untyped partners
        untyped_mask = conn_df['partner_type'].isna() | (conn_df['partner_type'].astype(str).str.strip() == '')
        untyped_count = untyped_mask.sum()
        untyped_weight = conn_df.loc[untyped_mask, 'weight'].sum() if untyped_count > 0 else 0.0
        total_weight = conn_df['weight'].sum()
        
        # Store original conn_df for dynamic expansion
        original_conn_df = conn_df.copy()
        
        # Filter untyped if configured
        if not self.config.include_untyped_partners:
            conn_df = conn_df[~untyped_mask].copy()
        else:
            # Use 'untyped' as collective type for untyped partners (Round 5)
            conn_df = conn_df.copy()
            conn_df.loc[untyped_mask, 'partner_type'] = 'untyped'
        
        if conn_df.empty:
            return {}, {}, untyped_count, untyped_weight / total_weight if total_weight > 0 else 0.0, total_weight, 0, 0, {}, top_k
        
        # Apply fuzzy matching to partner types
        if self.config.fuzzy_match.enabled:
            conn_df = conn_df.copy()
            conn_df['partner_type_normalized'] = conn_df['partner_type'].apply(
                lambda x: normalize_partner_type(x, self.config.fuzzy_match)
            )
        else:
            conn_df['partner_type_normalized'] = conn_df['partner_type'].astype(str)
        
        # Round 5: Dynamic expansion to ensure minimum unique types
        k_used = top_k
        top_m = self.config.top_m_type
        max_k = top_k * self.config.max_expansion_factor
        
        if self.config.dynamic_expansion:
            # Sort by weight first
            conn_df = conn_df.sort_values('weight', ascending=False)
            
            # Initial selection
            selected = conn_df.head(k_used)
            unique_types = selected['partner_type_normalized'].nunique()
            
            # Expand until M types or max_k reached
            while unique_types < top_m and k_used < max_k and k_used < len(conn_df):
                k_used += 5  # Expand by 5
                selected = conn_df.head(k_used)
                unique_types = selected['partner_type_normalized'].nunique()
            
            conn_df = selected
        else:
            # No dynamic expansion - just sort and take top_k
            conn_df = conn_df.sort_values('weight', ascending=False).head(top_k)
        
        # Aggregate by normalized partner type
        aggregated = conn_df.groupby('partner_type_normalized').agg({
            'weight': 'sum'
        }).reset_index()
        
        actual_count = len(aggregated)
        unique_types = actual_count
        
        # Sort by weight (already done but re-sort aggregated)
        aggregated = aggregated.sort_values('weight', ascending=False)
        
        # Compute normalized weights (proportions)
        typed_weight = aggregated['weight'].sum()
        partners_dict = {}
        for _, row in aggregated.iterrows():
            partners_dict[row['partner_type_normalized']] = row['weight'] / typed_weight if typed_weight > 0 else 0.0
        
        # Compute ranks
        ranks_dict = compute_ranks(partners_dict)
        
        # Round 5: Build bodyId → type mapping
        type_mapping = {}
        if 'partner_bodyId' in conn_df.columns:
            for _, row in conn_df.iterrows():
                bid = row.get('partner_bodyId')
                ptype = row.get('partner_type_normalized')
                if bid is not None and ptype is not None:
                    try:
                        type_mapping[int(bid)] = str(ptype)
                    except (ValueError, TypeError):
                        pass
        
        return (
            partners_dict,
            ranks_dict,
            untyped_count,
            untyped_weight / total_weight if total_weight > 0 else 0.0,
            total_weight,
            actual_count,
            unique_types,
            type_mapping,
            k_used
        )
    
    def get_profile(
        self,
        neuron: Union[str, int, List],
        dataset: str,
        force_refresh: bool = False
    ) -> ConnectivityProfile:
        """
        Extract connectivity profile for a neuron or neuron group.
        
        Round 5: Now includes dynamic expansion to ensure minimum unique types,
        and stores bodyId→type mapping metadata.
        
        Args:
            neuron: Neuron type (str), bodyId (int), or list of bodyIds
            dataset: Dataset identifier (e.g., 'hemibrain:v1.2.1')
            force_refresh: Bypass cache if True
        
        Returns:
            ConnectivityProfile with both proportion and rank representations
        
        Example:
            >>> profile = profiler.get_profile('aMe12', 'hemibrain:v1.2.1')
            >>> print(profile.upstream_partners)
        """
        # Check cache first
        if not force_refresh:
            cached = self._load_from_cache(neuron, dataset)
            if cached is not None:
                self._log(f"Loaded from cache: {neuron} in {dataset}", level='debug')
                return cached
        
        self._log(f"Extracting profile for {neuron} in {dataset}", level='debug')
        
        # Query connections based on dataset type
        dataset_lower = dataset.lower()
        if 'flywire' in dataset_lower or 'fafb' in dataset_lower or 'banc' in dataset_lower:
            upstream_df, downstream_df = self._query_connections_local(neuron, dataset)
        else:
            upstream_df, downstream_df = self._query_connections_neuprint(neuron, dataset)
        
        # Count neurons aggregated
        neurons_aggregated = 1
        if isinstance(neuron, list):
            neurons_aggregated = len(neuron)
        elif isinstance(neuron, str) and not upstream_df.empty:
            # Type-based query - count unique bodyIds
            if 'neuron_bodyId' in upstream_df.columns:
                neurons_aggregated = upstream_df['neuron_bodyId'].nunique()
            elif not downstream_df.empty and 'neuron_bodyId' in downstream_df.columns:
                neurons_aggregated = downstream_df['neuron_bodyId'].nunique()
        
        # Process upstream (Round 5: expanded return tuple)
        (up_partners, up_ranks, up_untyped_count, up_untyped_frac, 
         up_total_weight, up_actual_count, up_unique_types,
         up_type_mapping, up_k_used) = self._process_connections(
            upstream_df, 'upstream', self.config.top_k_bodyid
        )
        
        # Process downstream (Round 5: expanded return tuple)
        (down_partners, down_ranks, down_untyped_count, down_untyped_frac,
         down_total_weight, down_actual_count, down_unique_types,
         down_type_mapping, down_k_used) = self._process_connections(
            downstream_df, 'downstream', self.config.top_k_bodyid
        )
        
        # Create profile with Round 5 fields
        profile = ConnectivityProfile(
            neuron_id=neuron,
            dataset=dataset,
            upstream_partners=up_partners,
            downstream_partners=down_partners,
            upstream_ranks=up_ranks,
            downstream_ranks=down_ranks,
            upstream_top_k=self.config.top_k_bodyid,
            downstream_top_k=self.config.top_k_bodyid,
            total_upstream_weight=up_total_weight,
            total_downstream_weight=down_total_weight,
            num_neurons_aggregated=neurons_aggregated,
            untyped_upstream_count=up_untyped_count,
            untyped_downstream_count=down_untyped_count,
            untyped_upstream_weight_fraction=up_untyped_frac,
            untyped_downstream_weight_fraction=down_untyped_frac,
            actual_upstream_count=up_actual_count,
            actual_downstream_count=down_actual_count,
            # Round 5 fields
            unique_types_upstream=up_unique_types,
            unique_types_downstream=down_unique_types,
            partner_type_mapping_upstream=up_type_mapping if up_type_mapping else None,
            partner_type_mapping_downstream=down_type_mapping if down_type_mapping else None,
            top_k_bodyid_used=max(up_k_used, down_k_used),
            top_m_type_target=self.config.top_m_type,
        )
        
        # Save to cache
        self._save_to_cache(profile)
        
        self._log(f"Profile extracted: {up_actual_count} upstream ({up_unique_types} types), "
                  f"{down_actual_count} downstream ({down_unique_types} types)", level='debug')
        
        return profile
    
    def get_profiles_batch(
        self,
        neurons: List[Union[str, int]],
        dataset: str,
        force_refresh: bool = False
    ) -> Dict[Union[str, int], ConnectivityProfile]:
        """
        Batch extraction for multiple neurons.
        
        More efficient than calling get_profile repeatedly for large queries.
        
        Args:
            neurons: List of neuron types or bodyIds
            dataset: Dataset identifier
            force_refresh: Bypass cache if True
        
        Returns:
            Dict mapping neuron identifier to ConnectivityProfile
        """
        results = {}
        
        for neuron in neurons:
            try:
                profile = self.get_profile(neuron, dataset, force_refresh)
                results[neuron] = profile
            except Exception as e:
                self._log(f"Warning: Failed to extract profile for {neuron}: {e}")
        
        return results
    
    def get_profiles_for_type_across_datasets(
        self,
        neuron_type: str,
        datasets: Optional[List[str]] = None,
        force_refresh: bool = False
    ) -> Dict[str, ConnectivityProfile]:
        """
        Get profiles for a neuron type across multiple datasets.
        
        Useful for cross-dataset verification of type assignments.
        
        Args:
            neuron_type: Type name to query
            datasets: List of datasets (defaults to self.datasets)
            force_refresh: Bypass cache if True
        
        Returns:
            Dict mapping dataset name to ConnectivityProfile
        """
        datasets = datasets or self.datasets
        
        results = {}
        for dataset in datasets:
            try:
                profile = self.get_profile(neuron_type, dataset, force_refresh)
                results[dataset] = profile
            except Exception as e:
                self._log(f"Warning: Failed to extract profile for {neuron_type} in {dataset}: {e}")
        
        return results
    
    def get_vectorized_profiles(
        self,
        neuron_types: List[str],
        datasets: Optional[List[str]] = None,
        direction: str = 'both',
        vector_type: str = 'weight',
        normalize: bool = True,
        min_occurrence: int = 1
    ) -> Dict[str, Any]:
        """
        Get vectorized profiles for multiple neurons across datasets.
        
        This method provides efficient batch vectorization with a shared
        vocabulary, suitable for batch comparisons and clustering.
        
        Args:
            neuron_types: List of neuron types to vectorize
            datasets: List of datasets (default: all configured datasets)
            direction: 'upstream', 'downstream', or 'both'
            vector_type: 'weight' for normalized weights, 'rank' for ranks
            normalize: L2-normalize weight vectors (ignored for rank)
            min_occurrence: Minimum profiles a partner must appear in
        
        Returns:
            Dict with keys:
            - 'matrix': (n_profiles, n_features) numpy array
            - 'profile_ids': List of profile IDs in row order
            - 'vocabulary': List of partner types in column order
            - 'profiles': Dict of {profile_id: ConnectivityProfile}
        
        Example:
            >>> result = profiler.get_vectorized_profiles(
            ...     ['Mi1', 'Tm1', 'L1'],
            ...     datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
            ...     vector_type='weight'
            ... )
            >>> result['matrix'].shape
            (6, 25)  # 3 types x 2 datasets = 6 profiles, 25 partners
        """
        datasets = datasets or self.datasets
        
        # Collect all profiles
        profiles = {}
        for neuron_type in neuron_types:
            for dataset in datasets:
                try:
                    profile = self.get_profile(neuron_type, dataset)
                    profile_id = f"{dataset}:{neuron_type}"
                    profiles[profile_id] = profile
                except Exception as e:
                    self._log(f"Warning: Could not get profile for {neuron_type} in {dataset}: {e}")
        
        if not profiles:
            return {
                'matrix': np.empty((0, 0)),
                'profile_ids': [],
                'vocabulary': [],
                'profiles': {}
            }
        
        # Build shared vocabulary
        vocabulary = ConnectivityProfile.build_shared_vocabulary(
            profiles, direction, min_occurrence
        )
        
        # Build matrix
        if vector_type == 'rank':
            matrix, profile_ids, vocab = ConnectivityProfile.profiles_to_rank_matrix(
                profiles, vocabulary, direction
            )
        else:  # weight
            matrix, profile_ids, vocab = ConnectivityProfile.profiles_to_weight_matrix(
                profiles, vocabulary, direction, normalize
            )
        
        return {
            'matrix': matrix,
            'profile_ids': profile_ids,
            'vocabulary': vocab,
            'profiles': profiles
        }
    
    def save_vectorized_profiles(
        self,
        output_path: Union[str, Path],
        neuron_types: List[str],
        datasets: Optional[List[str]] = None,
        direction: str = 'both',
        vector_type: str = 'weight'
    ) -> Path:
        """
        Save vectorized profiles to a numpy archive for fast loading.
        
        Args:
            output_path: Path to save the .npz file
            neuron_types: List of neuron types to vectorize
            datasets: List of datasets (default: all configured datasets)
            direction: 'upstream', 'downstream', or 'both'
            vector_type: 'weight' or 'rank'
        
        Returns:
            Path to the saved file
        
        Example:
            >>> path = profiler.save_vectorized_profiles(
            ...     'cache/vectorized_profiles.npz',
            ...     ['Mi1', 'Tm1', 'L1']
            ... )
        """
        result = self.get_vectorized_profiles(
            neuron_types, datasets, direction, vector_type
        )
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            output_path,
            matrix=result['matrix'],
            profile_ids=result['profile_ids'],
            vocabulary=result['vocabulary'],
            vector_type=vector_type,
            direction=direction
        )
        
        self._log(f"Saved vectorized profiles to {output_path}")
        return output_path
    
    @staticmethod
    def load_vectorized_profiles(input_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load vectorized profiles from a numpy archive.
        
        Args:
            input_path: Path to the .npz file
        
        Returns:
            Dict with matrix, profile_ids, vocabulary, vector_type, direction
        """
        data = np.load(input_path, allow_pickle=True)
        return {
            'matrix': data['matrix'],
            'profile_ids': list(data['profile_ids']),
            'vocabulary': list(data['vocabulary']),
            'vector_type': str(data['vector_type']),
            'direction': str(data['direction'])
        }
    
    def clear_cache(self, dataset: Optional[str] = None):
        """
        Clear cached profiles.
        
        Args:
            dataset: Clear only this dataset's cache (None = clear all)
        """
        # Clear memory cache
        if dataset:
            keys_to_remove = [k for k in self._memory_cache if k[1] == dataset]
            for key in keys_to_remove:
                del self._memory_cache[key]
        else:
            self._memory_cache.clear()
        
        # Clear disk cache
        if self.config.use_cache:
            if dataset:
                safe_dataset = dataset.replace(':', '_').replace('.', '_')
                dataset_cache_dir = self.cache_dir / safe_dataset
                if dataset_cache_dir.exists():
                    import shutil
                    shutil.rmtree(dataset_cache_dir)
                    self._log(f"Cleared disk cache for {dataset}")
            else:
                if self.cache_dir.exists():
                    import shutil
                    shutil.rmtree(self.cache_dir)
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    self._log("Cleared all disk cache")
    
    def get_available_types(self, dataset: str) -> Optional[List[str]]:
        """
        Get list of all neuron types available in a dataset.
        
        This method queries the dataset to find all unique neuron types
        that can be used with get_profile().
        
        Args:
            dataset: Dataset identifier (e.g., 'hemibrain_v1_2_1', 'flywire_FAFB_v783')
        
        Returns:
            List of neuron type names, or None if dataset not found
        
        Example:
            >>> profiler = ConnectivityProfiler(datasets=['hemibrain_v1_2_1'])
            >>> types = profiler.get_available_types('hemibrain_v1_2_1')
            >>> print(f"Found {len(types)} types: {types[:10]}...")
        """
        dataset_lower = dataset.lower()
        
        # Check if local dataset (FlyWire/FAFB/BANC/optic-lobe/male-cns)
        is_local = any(x in dataset_lower for x in ['flywire', 'fafb', 'banc', 'optic', 'male'])
        
        if is_local:
            return self._get_local_dataset_types(dataset)
        else:
            return self._get_neuprint_dataset_types(dataset)
    
    def _get_local_dataset_types(self, dataset: str) -> Optional[List[str]]:
        """Get types from local dataset files."""
        # Sanitize dataset name
        safe_name = dataset.replace(':', '_').replace('.', '_')
        
        # Find dataset folder
        src_dir = Path(__file__).parent.parent
        project_root = src_dir.parent
        
        # Try multiple possible paths
        possible_paths = [
            project_root / 'datasets' / safe_name,
            project_root / 'datasets' / dataset,
            project_root / 'cache' / safe_name,
        ]
        
        dataset_path = None
        for path in possible_paths:
            if path.exists():
                dataset_path = path
                break
        
        if dataset_path is None:
            self._log(f"Dataset folder not found for {dataset}")
            return None
        
        # Try to load neuron_df to get types
        neuron_files = [
            dataset_path / f'{safe_name}_allneurons_neuron_df.parquet',
            dataset_path / f'{safe_name}_allneurons_neuron_df.csv',
            dataset_path / f'{safe_name}_neuron_df.parquet',
            dataset_path / f'{safe_name}_neuron_df.csv',
            dataset_path / 'neuron_df.parquet',
            dataset_path / 'neuron_df.csv',
        ]
        
        neuron_df = None
        for neuron_file in neuron_files:
            if neuron_file.exists():
                try:
                    if str(neuron_file).endswith('.parquet'):
                        neuron_df = pd.read_parquet(neuron_file)
                    else:
                        neuron_df = pd.read_csv(neuron_file)
                    self._log(f"Loaded types from {neuron_file.name}")
                    break
                except Exception as e:
                    self._log(f"Warning: Could not load {neuron_file}: {e}")
        
        if neuron_df is None or neuron_df.empty:
            self._log(f"No neuron_df found for {dataset}")
            return None
        
        # Find type column
        type_col = None
        for col in ['type', 'Type', 'cellType', 'cell_type']:
            if col in neuron_df.columns:
                type_col = col
                break
        
        if type_col is None:
            self._log(f"No type column found in neuron_df for {dataset}")
            return None
        
        # Get unique non-null types
        types = neuron_df[type_col].dropna().unique().tolist()
        types = [str(t) for t in types if t and str(t).strip()]
        types = sorted(set(types))
        
        self._log(f"Found {len(types)} unique types in {dataset}")
        return types
    
    def _get_neuprint_dataset_types(self, dataset: str) -> Optional[List[str]]:
        """Get types from NeuPrint dataset."""
        try:
            client = self._get_client_for_dataset(dataset)
            if client is None:
                return None
            
            from neuprint import fetch_neurons
            
            # Fetch all neurons with type info
            # Note: This can be slow for large datasets
            neurons, _ = fetch_neurons(None)
            
            if neurons is None or neurons.empty:
                return None
            
            # Get type column
            type_col = None
            for col in ['type', 'Type', 'cellType']:
                if col in neurons.columns:
                    type_col = col
                    break
            
            if type_col is None:
                return None
            
            types = neurons[type_col].dropna().unique().tolist()
            types = [str(t) for t in types if t and str(t).strip()]
            types = sorted(set(types))
            
            self._log(f"Found {len(types)} unique types in {dataset}")
            return types
            
        except Exception as e:
            self._log(f"Error getting types from NeuPrint: {e}")
            return None
