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
# Module-Level Connection Data Cache
# ============================================================================
# Cache structure: {dataset_key: {'conn_df': DataFrame, 'type_lookup': dict}}
# This is shared across all ConnectivityProfiler instances to avoid repeated
# disk reads when processing multiple neurons/types from the same dataset.

_PROFILER_CONN_CACHE: Dict[str, Dict[str, Any]] = {}


def clear_profiler_conn_cache():
    """Clear the module-level connection data cache."""
    global _PROFILER_CONN_CACHE
    _PROFILER_CONN_CACHE.clear()


def get_profiler_conn_cache_info() -> Dict[str, Any]:
    """Get info about cached connection data."""
    info = {}
    for dataset, data in _PROFILER_CONN_CACHE.items():
        info[dataset] = {
            'conn_df_rows': len(data.get('conn_df', [])) if data.get('conn_df') is not None else 0,
            'type_lookup_size': len(data.get('type_lookup', {})),
        }
    return info


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
    
    Configuration for connectivity profile extraction and comparison.
    
    Key defaults:
    - top_k_bodyid=5: Focus on top 5 strongest connections per direction
    - top_m_type=0: No minimum type requirement (set >0 for type diversity needs)
    - normalize_method='rank': Use rank-based comparison (always used for bodyId comparison)
    
    Hybrid 2-hop profile features:
    - expand_untyped_2hop: Fetch 2-hop typed partners for 1-hop untyped neurons (default: True)
    - top_k_2hop: Max 2-hop partners to fetch per untyped 1-hop neuron (default: 5)
    - If 2-hop doesn't return any typed neuron in top_k, the untyped partner is ignored
    
    Note on removed parameters:
    - max_untyped_fraction: No longer needed - hybrid implementation handles untyped automatically
    - allow_2hop_expansion: Replaced by expand_untyped_2hop
    - use_ranks: Always True for bodyId comparison (proportions not used)
    
    Attributes:
        top_k_bodyid: Top K connections by bodyId (default: 5)
        top_m_type: Minimum unique types to ensure in profile (default: 0 = no expansion)
        min_synapse_threshold: Filter connections with fewer synapses (default: 3)
        include_untyped_partners: Include partners without type annotations (default: False)
        normalize_method: Weight normalization method ('rank' or 'both')
        use_cache: Enable profile caching to disk (default: True)
        fuzzy_match: Fuzzy matching configuration for partner type names
        dynamic_expansion: Dynamically expand K until M types reached (default: True)
        max_expansion_factor: Maximum expansion multiplier for K (default: 5)
        expand_untyped_2hop: Enable 2-hop expansion for untyped 1-hop partners
        top_k_2hop: Number of top 2-hop partners to fetch per untyped neuron
        use_bodyid_for_intra: Keep bodyId for intra-dataset comparison
    
    Example:
        >>> config = ProfilerConfig(
        ...     top_k_bodyid=5,  # Focus on top 5 partners
        ...     top_m_type=0,    # No minimum type requirement
        ...     min_synapse_threshold=3,
        ...     expand_untyped_2hop=True  # Enable 2-hop expansion for untyped
        ... )
    """
    top_k_bodyid: int = 5  # Top K connections by bodyId (default: 5)
    top_m_type: int = 0  # Minimum unique types (default: 0 = no expansion)
    min_synapse_threshold: int = 3
    include_untyped_partners: bool = False
    normalize_method: str = 'rank'  # 'rank' (default, always used for comparison) or 'both'
    use_cache: bool = True
    fuzzy_match: FuzzyMatchConfig = field(default_factory=FuzzyMatchConfig)
    dynamic_expansion: bool = True  # Expand K until M types reached
    max_expansion_factor: int = 5  # Max K expansion multiplier
    # Hybrid 2-hop profile options
    expand_untyped_2hop: bool = True  # Expand untyped 1-hop partners to 2-hop typed partners
    top_k_2hop: int = 5  # Max 2-hop partners per untyped 1-hop neuron
    use_bodyid_for_intra: bool = True  # Keep bodyId for intra-dataset matching


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
        upstream_partners: Partner type → actual synapse weight (not normalized)
        downstream_partners: Partner type → actual synapse weight (not normalized)
        upstream_ranks: Partner type → rank (1 = strongest, derived from weights)
        downstream_ranks: Partner type → rank (1 = strongest, derived from weights)
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
    
    # Partner data: partner_type → actual synapse weight (enables easy aggregation)
    # Weights are stored as actual synapse counts (typically < 4096 per pair)
    # This preserves rank structure and allows bodyId→type aggregation via sum
    upstream_partners: Dict[str, float] = field(default_factory=dict)
    downstream_partners: Dict[str, float] = field(default_factory=dict)
    
    # Rank-based representation (derived from weights: higher weight = lower rank)
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
    
    # Round 6: Hybrid 2-hop profile for untyped 1-hop partners
    # For untyped 1-hop partners, we store:
    # - bodyId (for intra-dataset comparison)
    # - 2-hop typed partners (for cross-dataset comparison)
    untyped_upstream_bodyids: Optional[Dict[int, float]] = None  # bodyId → weight for untyped 1-hop upstream
    untyped_downstream_bodyids: Optional[Dict[int, float]] = None  # bodyId → weight for untyped 1-hop downstream
    untyped_upstream_2hop: Optional[Dict[int, Dict[str, float]]] = None  # untyped_bodyId → {2hop_type → weight}
    untyped_downstream_2hop: Optional[Dict[int, Dict[str, float]]] = None  # untyped_bodyId → {2hop_type → weight}
    untyped_upstream_2hop_ranks: Optional[Dict[int, Dict[str, int]]] = None  # untyped_bodyId → {2hop_type → rank}
    untyped_downstream_2hop_ranks: Optional[Dict[int, Dict[str, int]]] = None  # untyped_bodyId → {2hop_type → rank}
    
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
        
        # Round 6: Add untyped 1-hop bodyIds and 2-hop partner data
        if self.untyped_upstream_bodyids:
            result['untyped_upstream_bodyids'] = {str(k): float(v) for k, v in self.untyped_upstream_bodyids.items()}
        if self.untyped_downstream_bodyids:
            result['untyped_downstream_bodyids'] = {str(k): float(v) for k, v in self.untyped_downstream_bodyids.items()}
        
        # 2-hop typed partners for untyped 1-hop: {untyped_bodyId: {2hop_type: weight}}
        if self.untyped_upstream_2hop:
            result['untyped_upstream_2hop'] = {
                str(bid): {str(t): float(w) for t, w in types.items()}
                for bid, types in self.untyped_upstream_2hop.items()
            }
        if self.untyped_downstream_2hop:
            result['untyped_downstream_2hop'] = {
                str(bid): {str(t): float(w) for t, w in types.items()}
                for bid, types in self.untyped_downstream_2hop.items()
            }
        
        # 2-hop ranks
        if self.untyped_upstream_2hop_ranks:
            result['untyped_upstream_2hop_ranks'] = {
                str(bid): {str(t): int(r) for t, r in types.items()}
                for bid, types in self.untyped_upstream_2hop_ranks.items()
            }
        if self.untyped_downstream_2hop_ranks:
            result['untyped_downstream_2hop_ranks'] = {
                str(bid): {str(t): int(r) for t, r in types.items()}
                for bid, types in self.untyped_downstream_2hop_ranks.items()
            }
        
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
        
        # Round 6: Convert untyped bodyId keys back to int
        if 'untyped_upstream_bodyids' in data and data['untyped_upstream_bodyids']:
            data['untyped_upstream_bodyids'] = {
                int(k): v for k, v in data['untyped_upstream_bodyids'].items()
            }
        if 'untyped_downstream_bodyids' in data and data['untyped_downstream_bodyids']:
            data['untyped_downstream_bodyids'] = {
                int(k): v for k, v in data['untyped_downstream_bodyids'].items()
            }
        
        # Convert 2-hop data keys back to int
        if 'untyped_upstream_2hop' in data and data['untyped_upstream_2hop']:
            data['untyped_upstream_2hop'] = {
                int(bid): types for bid, types in data['untyped_upstream_2hop'].items()
            }
        if 'untyped_downstream_2hop' in data and data['untyped_downstream_2hop']:
            data['untyped_downstream_2hop'] = {
                int(bid): types for bid, types in data['untyped_downstream_2hop'].items()
            }
        if 'untyped_upstream_2hop_ranks' in data and data['untyped_upstream_2hop_ranks']:
            data['untyped_upstream_2hop_ranks'] = {
                int(bid): types for bid, types in data['untyped_upstream_2hop_ranks'].items()
            }
        if 'untyped_downstream_2hop_ranks' in data and data['untyped_downstream_2hop_ranks']:
            data['untyped_downstream_2hop_ranks'] = {
                int(bid): types for bid, types in data['untyped_downstream_2hop_ranks'].items()
            }
        
        return cls(**data)
    
    def get_all_partner_types(self) -> set:
        """Get set of all partner types (upstream + downstream)."""
        return set(self.upstream_partners.keys()) | set(self.downstream_partners.keys())
    
    def get_partner_weight(self, partner_type: str, direction: str = 'both') -> float:
        """Get actual synapse weight for a partner type."""
        up = self.upstream_partners.get(partner_type, 0.0)
        down = self.downstream_partners.get(partner_type, 0.0)
        
        if direction == 'upstream':
            return up
        elif direction == 'downstream':
            return down
        else:
            return up + down
    
    def get_proportions(self, direction: str = 'both') -> Dict[str, float]:
        """
        Get normalized proportions from actual weights (on-demand).
        
        This converts actual synapse weights to proportions (sum to 1.0)
        for comparison methods that need normalized values.
        
        Args:
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Dict of partner_type → proportion (0.0 to 1.0)
        """
        if direction == 'upstream':
            partners = self.upstream_partners
        elif direction == 'downstream':
            partners = self.downstream_partners
        else:
            # Combine both directions
            partners = {}
            for k, v in self.upstream_partners.items():
                partners[k] = partners.get(k, 0.0) + v
            for k, v in self.downstream_partners.items():
                partners[k] = partners.get(k, 0.0) + v
        
        total = sum(partners.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in partners.items()}
    
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
            
            # Round 6: Show 2-hop expansion info for untyped partners
            if self.untyped_upstream_bodyids:
                lines.append(f"  Untyped upstream bodyIds (for 2-hop): {list(self.untyped_upstream_bodyids.keys())[:5]}")
            if self.untyped_downstream_bodyids:
                lines.append(f"  Untyped downstream bodyIds (for 2-hop): {list(self.untyped_downstream_bodyids.keys())[:5]}")
            if self.untyped_upstream_2hop:
                n_with_2hop = len([b for b, t in self.untyped_upstream_2hop.items() if t])
                lines.append(f"  Upstream untyped with 2-hop partners: {n_with_2hop}/{len(self.untyped_upstream_bodyids or {})}")
            if self.untyped_downstream_2hop:
                n_with_2hop = len([b for b, t in self.untyped_downstream_2hop.items() if t])
                lines.append(f"  Downstream untyped with 2-hop partners: {n_with_2hop}/{len(self.untyped_downstream_bodyids or {})}")
        
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
    
    def to_hybrid_rank_vector(
        self,
        typed_vocabulary: List[str],
        bodyid_vocabulary: Optional[List[int]] = None,
        direction: str = 'upstream',
        mode: str = 'cross_dataset',
        default_rank: Optional[int] = None
    ) -> np.ndarray:
        """
        Round 6: Convert connectivity profile to a hybrid rank vector.
        
        For cross-dataset comparison, this creates a vector where:
        - Typed 1-hop partners use their type-based ranks directly
        - Untyped 1-hop partners are expanded to their 2-hop typed partners
        
        For intra-dataset comparison, untyped partners can be compared by bodyId.
        
        Args:
            typed_vocabulary: Ordered list of typed partner types (for cross-dataset)
            bodyid_vocabulary: Ordered list of bodyIds (for intra-dataset, optional)
            direction: 'upstream' or 'downstream'
            mode: 'cross_dataset' (use typed + 2-hop) or 'intra_dataset' (use typed + bodyId)
            default_rank: Rank for missing partners. If None, uses K+1.
        
        Returns:
            numpy array of ranks
        
        For cross_dataset mode:
            - Vocabulary should include both typed 1-hop types AND 2-hop types
            - Untyped 1-hop partners contribute their 2-hop typed partner ranks
        
        For intra_dataset mode:
            - Two vectors: typed_vector (for typed partners) and bodyid_vector (for untyped)
        """
        if direction == 'upstream':
            typed_ranks = self.upstream_ranks
            typed_partners = self.upstream_partners
            untyped_bodyids = self.untyped_upstream_bodyids
            hop2_ranks = self.untyped_upstream_2hop_ranks
        else:
            typed_ranks = self.downstream_ranks
            typed_partners = self.downstream_partners
            untyped_bodyids = self.untyped_downstream_bodyids
            hop2_ranks = self.untyped_downstream_2hop_ranks
        
        k = len(typed_partners) + len(untyped_bodyids or {})
        actual_default = default_rank if default_rank is not None else (k + 1)
        
        if mode == 'cross_dataset':
            # Build vector that includes typed 1-hop + aggregated 2-hop types
            # For each vocabulary entry, check if it's in typed_ranks OR in 2-hop ranks
            
            # Start with typed partner ranks
            combined_ranks = dict(typed_ranks)
            
            # Add 2-hop ranks (for untyped 1-hop partners)
            # If a type appears in multiple untyped partners' 2-hop, use the best (lowest) rank
            if hop2_ranks:
                for untyped_bid, type_ranks in hop2_ranks.items():
                    for ptype, rank in type_ranks.items():
                        # Adjust rank: 2-hop partners should rank after typed 1-hop
                        # Add offset based on untyped bodyId's original rank
                        adjusted_rank = rank + len(typed_partners)
                        if ptype not in combined_ranks or combined_ranks[ptype] > adjusted_rank:
                            combined_ranks[ptype] = adjusted_rank
            
            vector = np.array([combined_ranks.get(p, actual_default) for p in typed_vocabulary])
            return vector
        
        else:  # intra_dataset
            # For intra-dataset, we want separate vectors for typed and untyped
            # Return typed vector; bodyid vector would be separate
            vector = np.array([typed_ranks.get(p, actual_default) for p in typed_vocabulary])
            return vector
    
    def get_untyped_bodyid_ranks(self, direction: str = 'upstream') -> Dict[int, int]:
        """
        Round 6: Get ranks for untyped bodyIds (for intra-dataset comparison).
        
        Returns:
            Dict mapping bodyId → rank based on weight
        """
        if direction == 'upstream':
            bodyids = self.untyped_upstream_bodyids
        else:
            bodyids = self.untyped_downstream_bodyids
        
        if not bodyids:
            return {}
        
        # Rank by weight (highest weight = rank 1)
        sorted_bids = sorted(bodyids.items(), key=lambda x: x[1], reverse=True)
        return {bid: rank + 1 for rank, (bid, _) in enumerate(sorted_bids)}
    
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
    def build_hybrid_vocabulary(
        profiles: Dict[str, 'ConnectivityProfile'],
        direction: str = 'upstream',
        min_occurrence: int = 1,
        include_2hop: bool = True
    ) -> List[str]:
        """
        Round 6: Build vocabulary including both typed 1-hop and 2-hop types.
        
        For cross-dataset comparison with untyped 1-hop partners expanded.
        
        Args:
            profiles: Dict of {profile_id: ConnectivityProfile}
            direction: 'upstream' or 'downstream'
            min_occurrence: Minimum profiles a type must appear in
            include_2hop: Whether to include 2-hop types from untyped partners
        
        Returns:
            Sorted list of all partner types (1-hop typed + 2-hop from untyped)
        """
        partner_counts: Dict[str, int] = {}
        
        for profile in profiles.values():
            # Add typed 1-hop partners
            if direction == 'upstream':
                for partner in profile.upstream_partners.keys():
                    partner_counts[partner] = partner_counts.get(partner, 0) + 1
                
                # Add 2-hop types from untyped partners
                if include_2hop and profile.untyped_upstream_2hop:
                    for untyped_bid, type_weights in profile.untyped_upstream_2hop.items():
                        for ptype in type_weights.keys():
                            partner_counts[ptype] = partner_counts.get(ptype, 0) + 1
            else:
                for partner in profile.downstream_partners.keys():
                    partner_counts[partner] = partner_counts.get(partner, 0) + 1
                
                if include_2hop and profile.untyped_downstream_2hop:
                    for untyped_bid, type_weights in profile.untyped_downstream_2hop.items():
                        for ptype in type_weights.keys():
                            partner_counts[ptype] = partner_counts.get(ptype, 0) + 1
        
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
    
    @staticmethod
    def aggregate_bodyid_profiles(
        profiles: List['ConnectivityProfile'],
        neuron_type: str,
        dataset: str
    ) -> 'ConnectivityProfile':
        """
        Aggregate multiple bodyId-level profiles into a single type-level profile.
        
        Since profiles store actual synapse weights (not proportions), aggregation
        is simply summing the weights across all bodyIds. This preserves the rank
        structure naturally - partners with more total synapses get lower ranks.
        
        Args:
            profiles: List of bodyId-level ConnectivityProfile objects
            neuron_type: Name for the aggregated type
            dataset: Dataset identifier
        
        Returns:
            Aggregated ConnectivityProfile representing the entire type
        
        Example:
            >>> # Get profiles for all neurons of type "aMe12"
            >>> bodyid_profiles = [profiler.get_profile(bid, dataset) for bid in body_ids]
            >>> # Aggregate into single type profile
            >>> type_profile = ConnectivityProfile.aggregate_bodyid_profiles(
            ...     bodyid_profiles, 'aMe12', dataset
            ... )
        """
        if not profiles:
            raise ValueError("Cannot aggregate empty profile list")
        
        # Sum weights across all profiles
        upstream_weights: Dict[str, float] = {}
        downstream_weights: Dict[str, float] = {}
        
        total_upstream_weight = 0.0
        total_downstream_weight = 0.0
        untyped_upstream_count = 0
        untyped_downstream_count = 0
        untyped_up_weight = 0.0
        untyped_down_weight = 0.0
        
        for profile in profiles:
            # Sum upstream weights
            for partner, weight in profile.upstream_partners.items():
                upstream_weights[partner] = upstream_weights.get(partner, 0.0) + weight
            
            # Sum downstream weights
            for partner, weight in profile.downstream_partners.items():
                downstream_weights[partner] = downstream_weights.get(partner, 0.0) + weight
            
            # Accumulate totals
            total_upstream_weight += profile.total_upstream_weight
            total_downstream_weight += profile.total_downstream_weight
            untyped_upstream_count += profile.untyped_upstream_count
            untyped_downstream_count += profile.untyped_downstream_count
            untyped_up_weight += profile.untyped_upstream_weight_fraction * profile.total_upstream_weight
            untyped_down_weight += profile.untyped_downstream_weight_fraction * profile.total_downstream_weight
        
        # Compute ranks from aggregated weights
        upstream_ranks = compute_ranks(upstream_weights)
        downstream_ranks = compute_ranks(downstream_weights)
        
        # Compute untyped fractions
        untyped_up_frac = untyped_up_weight / total_upstream_weight if total_upstream_weight > 0 else 0.0
        untyped_down_frac = untyped_down_weight / total_downstream_weight if total_downstream_weight > 0 else 0.0
        
        return ConnectivityProfile(
            neuron_id=neuron_type,
            dataset=dataset,
            upstream_partners=upstream_weights,
            downstream_partners=downstream_weights,
            upstream_ranks=upstream_ranks,
            downstream_ranks=downstream_ranks,
            upstream_top_k=profiles[0].upstream_top_k,
            downstream_top_k=profiles[0].downstream_top_k,
            total_upstream_weight=total_upstream_weight,
            total_downstream_weight=total_downstream_weight,
            num_neurons_aggregated=len(profiles),
            untyped_upstream_count=untyped_upstream_count,
            untyped_downstream_count=untyped_downstream_count,
            untyped_upstream_weight_fraction=untyped_up_frac,
            untyped_downstream_weight_fraction=untyped_down_frac,
            actual_upstream_count=len(upstream_weights),
            actual_downstream_count=len(downstream_weights),
            unique_types_upstream=len(upstream_weights),
            unique_types_downstream=len(downstream_weights),
        )


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
            self.cache_dir = project_root / 'cache'
        
        if self.config.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for profiles (neuron_id, dataset) -> ConnectivityProfile
        self._memory_cache: Dict[Tuple[str, str], ConnectivityProfile] = {}
        
        # In-memory cache for disk parquet DataFrames: dataset -> DataFrame
        # Avoids repeated disk reads when looking up multiple profiles
        self._disk_cache_df: Dict[str, pd.DataFrame] = {}
        
        # Index for O(1) lookups: dataset -> {neuron_id -> row_index}
        self._disk_cache_index: Dict[str, Dict[str, int]] = {}
        
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
    
    def _get_cache_parquet_path(self, dataset: str) -> Path:
        """Get path to parquet cache file for a dataset."""
        safe_dataset = dataset.replace(':', '_').replace('.', '_')
        return self.cache_dir / safe_dataset / 'connectivity_profiles.parquet'
    
    def _load_cache_dataframe(self, dataset: str, force_reload: bool = False) -> Optional[pd.DataFrame]:
        """
        Load the cache dataframe for a dataset with in-memory caching.
        
        First load reads from disk and caches in memory.
        Subsequent loads return the cached DataFrame instantly.
        
        Args:
            dataset: Dataset name
            force_reload: If True, reload from disk even if cached
            
        Returns:
            Cached DataFrame or None if not found
        """
        # Check in-memory cache first
        if not force_reload and dataset in self._disk_cache_df:
            return self._disk_cache_df[dataset]
        
        # Load from disk
        cache_path = self._get_cache_parquet_path(dataset)
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                # Cache in memory
                self._disk_cache_df[dataset] = df
                # Build index for O(1) lookups
                self._build_disk_cache_index(dataset)
                return df
            except Exception as e:
                self._log(f"Warning: Could not load cache parquet: {e}")
        return None
    
    def _build_disk_cache_index(self, dataset: str):
        """Build O(1) lookup index for disk cache DataFrame."""
        if dataset not in self._disk_cache_df:
            self._disk_cache_index[dataset] = {}
            return
        
        df = self._disk_cache_df[dataset]
        if 'neuron_id' not in df.columns:
            self._disk_cache_index[dataset] = {}
            return
        
        # Build index: neuron_id -> row_index
        self._disk_cache_index[dataset] = {
            str(row['neuron_id']): idx 
            for idx, row in df.iterrows()
        }
    
    def _save_cache_dataframe(self, df: pd.DataFrame, dataset: str):
        """
        Save the cache dataframe for a dataset.
        Also updates the in-memory cache.
        """
        cache_path = self._get_cache_parquet_path(dataset)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(cache_path, index=False)
            # Update in-memory cache
            self._disk_cache_df[dataset] = df
            self._build_disk_cache_index(dataset)
        except Exception as e:
            self._log(f"Warning: Could not save cache parquet: {e}")
    
    def _load_from_cache(
        self, 
        neuron_id: Union[str, int], 
        dataset: str,
        required_top_k: Optional[int] = None
    ) -> Optional[ConnectivityProfile]:
        """
        Load profile from cache if exists and meets requirements.
        
        Uses a 3-tier cache strategy:
        1. Memory cache (instant): Check _memory_cache dict first
        2. Disk cache index (O(1)): Use _disk_cache_index for fast row lookup
        3. Disk cache load: Only loads parquet once, cached in _disk_cache_df
        
        Option A Cache Strategy:
        - If cached_k >= requested_k: Use cache (slice if needed)
        - If cached_k < requested_k: Return None (need to re-fetch with higher k)
        
        Args:
            neuron_id: Neuron identifier
            dataset: Dataset name
            required_top_k: Minimum top_k required. If None, uses config.top_k_bodyid
        
        Returns:
            Cached profile if available and sufficient, None otherwise
        """
        required_k = required_top_k or self.config.top_k_bodyid
        neuron_id_str = str(neuron_id)
        
        # Tier 1: Check memory cache first (instant O(1))
        cache_key = (neuron_id_str, dataset)
        if cache_key in self._memory_cache:
            cached = self._memory_cache[cache_key]
            # Check if cached profile has sufficient top_k
            if cached.top_k_bodyid_used >= required_k:
                return cached
            # Cached profile has lower k than required - need re-fetch
            return None
        
        # Tier 2 & 3: Check disk cache with in-memory DataFrame
        if not self.config.use_cache:
            return None
        
        # This loads from disk only once, then uses cached DataFrame
        cache_df = self._load_cache_dataframe(dataset)
        if cache_df is None or 'neuron_id' not in cache_df.columns:
            return None
        
        # Use index for O(1) row lookup instead of DataFrame filter
        if dataset in self._disk_cache_index:
            row_idx = self._disk_cache_index[dataset].get(neuron_id_str)
            if row_idx is not None:
                try:
                    # Direct row access by index (O(1))
                    row_data = cache_df.iloc[row_idx]
                    profile = self._row_to_profile(row_data)
                    
                    # Check if cached profile has sufficient top_k
                    if profile.top_k_bodyid_used >= required_k:
                        self._memory_cache[cache_key] = profile
                        return profile
                    
                    # Cached profile has lower k - return None to trigger re-fetch
                    return None
                except Exception as e:
                    self._log(f"Warning: Could not load cache for {neuron_id}: {e}")
        
        return None
    
    def _row_to_profile(self, row: pd.Series) -> ConnectivityProfile:
        """Convert a parquet row to a ConnectivityProfile."""
        # Parse JSON strings back to dicts
        def parse_json(val):
            if pd.isna(val) or val is None or val == '':
                return None
            if isinstance(val, dict):
                return val
            try:
                return json.loads(val)
            except:
                return None
        
        return ConnectivityProfile(
            neuron_id=row['neuron_id'],
            dataset=row['dataset'],
            upstream_partners=parse_json(row.get('upstream_partners')) or {},
            downstream_partners=parse_json(row.get('downstream_partners')) or {},
            upstream_ranks=parse_json(row.get('upstream_ranks')) or {},
            downstream_ranks=parse_json(row.get('downstream_ranks')) or {},
            upstream_top_k=int(row.get('upstream_top_k', 10)),
            downstream_top_k=int(row.get('downstream_top_k', 10)),
            total_upstream_weight=float(row.get('total_upstream_weight', 0)),
            total_downstream_weight=float(row.get('total_downstream_weight', 0)),
            num_neurons_aggregated=int(row.get('num_neurons_aggregated', 1)),
            untyped_upstream_count=int(row.get('untyped_upstream_count', 0)),
            untyped_downstream_count=int(row.get('untyped_downstream_count', 0)),
            untyped_upstream_weight_fraction=float(row.get('untyped_upstream_weight_fraction', 0)),
            untyped_downstream_weight_fraction=float(row.get('untyped_downstream_weight_fraction', 0)),
            actual_upstream_count=int(row.get('actual_upstream_count', 0)),
            actual_downstream_count=int(row.get('actual_downstream_count', 0)),
            is_weak_connectivity=bool(row.get('is_weak_connectivity', False)),
            unique_types_upstream=int(row.get('unique_types_upstream', 0)),
            unique_types_downstream=int(row.get('unique_types_downstream', 0)),
            is_sparse=bool(row.get('is_sparse', False)),
            partner_type_mapping_upstream=parse_json(row.get('partner_type_mapping_upstream')),
            partner_type_mapping_downstream=parse_json(row.get('partner_type_mapping_downstream')),
            top_k_bodyid_used=int(row.get('top_k_bodyid_used', 20)),
            top_m_type_target=int(row.get('top_m_type_target', 5)),
            untyped_upstream_bodyids=parse_json(row.get('untyped_upstream_bodyids')),
            untyped_downstream_bodyids=parse_json(row.get('untyped_downstream_bodyids')),
            untyped_upstream_2hop=parse_json(row.get('untyped_upstream_2hop')),
            untyped_downstream_2hop=parse_json(row.get('untyped_downstream_2hop')),
            untyped_upstream_2hop_ranks=parse_json(row.get('untyped_upstream_2hop_ranks')),
            untyped_downstream_2hop_ranks=parse_json(row.get('untyped_downstream_2hop_ranks')),
        )
    
    def _profile_to_row(self, profile: ConnectivityProfile) -> dict:
        """Convert a ConnectivityProfile to a flat dict for parquet storage."""
        data = profile.to_dict()
        
        # Convert nested dicts to JSON strings for parquet storage
        for key in ['upstream_partners', 'downstream_partners', 'upstream_ranks', 'downstream_ranks',
                    'partner_type_mapping_upstream', 'partner_type_mapping_downstream',
                    'untyped_upstream_bodyids', 'untyped_downstream_bodyids',
                    'untyped_upstream_2hop', 'untyped_downstream_2hop',
                    'untyped_upstream_2hop_ranks', 'untyped_downstream_2hop_ranks']:
            if key in data and data[key] is not None:
                data[key] = json.dumps(data[key])
            else:
                data[key] = None
        
        return data
    
    def _save_to_cache(self, profile: ConnectivityProfile):
        """Save profile to cache (parquet)."""
        # Memory cache
        cache_key = (str(profile.neuron_id), profile.dataset)
        self._memory_cache[cache_key] = profile
        
        # Disk cache (parquet)
        if not self.config.use_cache:
            return
        
        # Load existing cache or create new
        cache_df = self._load_cache_dataframe(profile.dataset)
        
        # Convert profile to row
        new_row = self._profile_to_row(profile)
        new_row_df = pd.DataFrame([new_row])
        
        if cache_df is not None and not cache_df.empty:
            # Remove existing entry for this neuron if present
            cache_df = cache_df[cache_df['neuron_id'] != str(profile.neuron_id)]
            # Append new row
            cache_df = pd.concat([cache_df, new_row_df], ignore_index=True)
        else:
            cache_df = new_row_df
        
        # Save back to parquet
        self._save_cache_dataframe(cache_df, profile.dataset)
    
    def _save_profiles_to_cache_batch(
        self, 
        profiles: Dict[str, ConnectivityProfile], 
        dataset: str
    ):
        """
        Save multiple profiles to cache in a single batch operation.
        
        This is more efficient than calling _save_to_cache repeatedly
        when building cache for many profiles.
        
        Args:
            profiles: Dict mapping neuron_id to ConnectivityProfile
            dataset: Dataset identifier
        """
        if not profiles:
            return
        
        # Memory cache update
        for neuron_id, profile in profiles.items():
            cache_key = (str(neuron_id), dataset)
            self._memory_cache[cache_key] = profile
        
        # Disk cache (parquet)
        if not self.config.use_cache:
            return
        
        # Load existing cache
        cache_df = self._load_cache_dataframe(dataset)
        
        # Convert all new profiles to rows
        new_rows = [self._profile_to_row(p) for p in profiles.values()]
        new_df = pd.DataFrame(new_rows)
        
        if cache_df is not None and not cache_df.empty:
            # Remove existing entries for these neurons
            neuron_ids = set(str(p.neuron_id) for p in profiles.values())
            cache_df = cache_df[~cache_df['neuron_id'].isin(neuron_ids)]
            # Append new rows
            cache_df = pd.concat([cache_df, new_df], ignore_index=True)
        else:
            cache_df = new_df
        
        # Save back to parquet
        self._save_cache_dataframe(cache_df, dataset)
        self._log(f"Saved {len(profiles)} profiles to cache")

    def read_connectivity_profile_cache(
        self,
        dataset: str,
        neuron_types: Optional[List[str]] = None,
        min_top_k: Optional[int] = None
    ) -> Dict[str, ConnectivityProfile]:
        """
        Read pre-built connectivity profiles from cache (parquet format).
        
        This method loads cached profiles from disk for efficient batch access.
        Uses the 3-tier cache system for O(1) lookups when neuron_types are specified.
        
        Args:
            dataset: Dataset identifier (e.g., 'hemibrain_v1_2_1', 'flywire_FAFB_v783')
            neuron_types: Specific neuron types to load. If None, loads all cached profiles.
            min_top_k: Minimum top_k required. Profiles with lower top_k are skipped.
        
        Returns:
            Dict mapping neuron_type/id to ConnectivityProfile
        """
        self._log(f"Reading connectivity profile cache for {dataset}...")
        
        # Load cache DataFrame (uses in-memory cache if available)
        cache_df = self._load_cache_dataframe(dataset)
        
        if cache_df is None or cache_df.empty:
            cache_path = self._get_cache_parquet_path(dataset)
            self._log(f"Cache not found: {cache_path}")
            return {}
        
        profiles = {}
        loaded = 0
        skipped_k = 0
        failed = 0
        
        self._log(f"Found {len(cache_df)} profiles in cache")
        
        # Use index for O(1) lookups if specific neuron_types requested
        if neuron_types is not None and dataset in self._disk_cache_index:
            # Fast path: Use index for specific types
            for neuron_id in neuron_types:
                neuron_id_str = str(neuron_id)
                row_idx = self._disk_cache_index[dataset].get(neuron_id_str)
                
                if row_idx is None:
                    continue
                
                try:
                    row = cache_df.iloc[row_idx]
                    profile = self._row_to_profile(row)
                    
                    # Check min_top_k requirement
                    if min_top_k is not None and profile.top_k_bodyid_used < min_top_k:
                        skipped_k += 1
                        continue
                    
                    profiles[neuron_id_str] = profile
                    loaded += 1
                    
                    # Add to memory cache
                    cache_key = (neuron_id_str, dataset)
                    self._memory_cache[cache_key] = profile
                    
                except Exception as e:
                    self._log(f"Warning: Failed to load profile for {neuron_id}: {e}")
                    failed += 1
        else:
            # Slow path: Iterate all rows (only when loading ALL profiles)
            for idx, row in cache_df.iterrows():
                try:
                    neuron_id = row['neuron_id']
                    
                    # Convert row to profile
                    profile = self._row_to_profile(row)
                    
                    # Check min_top_k requirement
                    if min_top_k is not None:
                        if profile.top_k_bodyid_used < min_top_k:
                            skipped_k += 1
                            continue
                    
                    profiles[neuron_id] = profile
                    loaded += 1
                    
                    # Also add to memory cache
                    cache_key = (str(neuron_id), dataset)
                    self._memory_cache[cache_key] = profile
                    
                except Exception as e:
                    self._log(f"Warning: Failed to load profile from row {idx}: {e}")
                    failed += 1
        
        self._log(f"Loaded {loaded} profiles from cache "
                  f"(skipped {skipped_k} with insufficient top_k, {failed} failed)")
        
        return profiles
    
    def get_cache_stats(self, dataset: str) -> Dict[str, Any]:
        """
        Get statistics about cached profiles for a dataset.
        
        Args:
            dataset: Dataset identifier
        
        Returns:
            Dict with cache statistics:
            - 'total_profiles': Number of cached profiles
            - 'total_size_mb': Total cache size in MB
            - 'neuron_types': List of cached neuron types
            - 'top_k_distribution': Dict of top_k_used -> count
            - 'cache_modified': Last modification time of cache file
        """
        cache_path = self._get_cache_parquet_path(dataset)
        
        if not cache_path.exists():
            return {
                'total_profiles': 0,
                'total_size_mb': 0,
                'neuron_types': [],
                'top_k_distribution': {},
                'cache_modified': None
            }
        
        cache_df = self._load_cache_dataframe(dataset)
        
        if cache_df is None or cache_df.empty:
            return {
                'total_profiles': 0,
                'total_size_mb': 0,
                'neuron_types': [],
                'top_k_distribution': {},
                'cache_modified': None
            }
        
        total_size = cache_path.stat().st_size
        neuron_types = cache_df['neuron_id'].tolist() if 'neuron_id' in cache_df.columns else []
        
        # Get top_k distribution
        top_k_dist = {}
        if 'top_k_bodyid_used' in cache_df.columns:
            for top_k in cache_df['top_k_bodyid_used'].dropna():
                top_k = int(top_k)
                top_k_dist[top_k] = top_k_dist.get(top_k, 0) + 1
        
        from datetime import datetime
        cache_mtime = cache_path.stat().st_mtime
        
        return {
            'total_profiles': len(cache_df),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'neuron_types': neuron_types,
            'top_k_distribution': top_k_dist,
            'cache_modified': datetime.fromtimestamp(cache_mtime).isoformat()
        }

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
    
    def _get_cached_conn_df(self, dataset: str) -> Optional[pd.DataFrame]:
        """
        Get connection DataFrame from module-level cache, loading from disk only once.
        
        This method handles:
        1. Check module-level cache first
        2. Load from disk if not cached
        3. Standardize column names
        4. Join type info from neuron_df if needed
        5. Cache for future use
        
        Returns:
            Cached and preprocessed connection DataFrame, or None if not available
        """
        global _PROFILER_CONN_CACHE
        
        # Build cache key
        safe_name = dataset.replace(':', '_').replace('.', '_')
        
        # Check module-level cache first
        if safe_name in _PROFILER_CONN_CACHE and 'conn_df' in _PROFILER_CONN_CACHE[safe_name]:
            cached_df = _PROFILER_CONN_CACHE[safe_name]['conn_df']
            if cached_df is not None:
                return cached_df
        
        # Load from disk
        src_dir = Path(__file__).parent.parent
        project_root = src_dir.parent
        datasets_folder = project_root / 'datasets'
        dataset_path = datasets_folder / safe_name
        
        # Try to load connections file - check multiple naming conventions
        conn_files = [
            dataset_path / f'{safe_name}_merged_connections.parquet',
            dataset_path / f'{safe_name}_merged_connections.csv',
            dataset_path / f'{safe_name}_connections.parquet',
            dataset_path / f'{safe_name}_connections.csv',
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
                    self._log(f"Loaded connections from {conn_file.name} ({len(conn_df):,} rows)", level='debug')
                    break
                except Exception as e:
                    self._log(f"Warning: Could not load {conn_file}: {e}")
        
        if conn_df is None or conn_df.empty:
            # Cache the None result to avoid repeated disk checks
            if safe_name not in _PROFILER_CONN_CACHE:
                _PROFILER_CONN_CACHE[safe_name] = {}
            _PROFILER_CONN_CACHE[safe_name]['conn_df'] = None
            return None
        
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
        
        # Cache the preprocessed DataFrame
        if safe_name not in _PROFILER_CONN_CACHE:
            _PROFILER_CONN_CACHE[safe_name] = {}
        _PROFILER_CONN_CACHE[safe_name]['conn_df'] = conn_df
        
        self._log(f"Cached connection data for {dataset} ({len(conn_df):,} rows)")
        
        return conn_df
    
    def _query_connections_local(
        self,
        neuron: Union[str, int, List],
        dataset: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Query upstream and downstream connections from local dataset files.
        
        Uses module-level cache to load connection data once per dataset.
        
        Returns:
            Tuple of (upstream_df, downstream_df)
        """
        # Get cached connection DataFrame
        conn_df = self._get_cached_conn_df(dataset)
        
        if conn_df is None or conn_df.empty:
            self._log(f"Warning: No connection data found for {dataset}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Filter by minimum synapses (apply on a view, not copy yet)
        min_syn = self.config.min_synapse_threshold
        if min_syn > 0:
            conn_df = conn_df[conn_df['weight'] >= min_syn]
        
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
    
    def _fetch_2hop_partners(
        self,
        untyped_bodyids: List[int],
        dataset: str,
        direction: str
    ) -> Dict[int, Tuple[Dict[str, float], Dict[str, int]]]:
        """
        Round 6: Fetch 2-hop typed partners for untyped 1-hop partners.
        
        For each untyped 1-hop partner, query their connections and return
        only typed 2-hop partners with their weights and ranks.
        
        Uses module-level cache for connection data.
        
        Args:
            untyped_bodyids: List of untyped 1-hop partner bodyIds
            dataset: Dataset identifier
            direction: 'upstream' or 'downstream' - indicates the direction from original neuron
                       For upstream untyped partners, we get their upstream (2-hop upstream)
                       For downstream untyped partners, we get their downstream (2-hop downstream)
        
        Returns:
            Dict mapping untyped_bodyId → (weights_dict, ranks_dict)
            where weights_dict = {typed_type → normalized_weight}
            and ranks_dict = {typed_type → rank}
        """
        if not untyped_bodyids:
            return {}
        
        # Use cached connection data
        conn_df = self._get_cached_conn_df(dataset)
        
        if conn_df is None or conn_df.empty:
            return {}
        
        # Filter by minimum synapses
        min_syn = self.config.min_synapse_threshold
        if min_syn > 0:
            conn_df = conn_df[conn_df['weight'] >= min_syn]
        
        results = {}
        top_k_2hop = self.config.top_k_2hop
        
        for untyped_bid in untyped_bodyids:
            # Query connections for this untyped neuron
            # Direction determines which connections to fetch:
            # - upstream untyped partner → get their upstream (further back in circuit)
            # - downstream untyped partner → get their downstream (further forward in circuit)
            if direction == 'upstream':
                # Untyped partner is upstream of original neuron
                # Get their upstream connections (2-hop upstream)
                if 'bodyId_post' in conn_df.columns:
                    mask = conn_df['bodyId_post'] == untyped_bid
                    partner_type_col = 'type_pre'
                    partner_bid_col = 'bodyId_pre'
                else:
                    continue
            else:
                # Untyped partner is downstream of original neuron
                # Get their downstream connections (2-hop downstream)
                if 'bodyId_pre' in conn_df.columns:
                    mask = conn_df['bodyId_pre'] == untyped_bid
                    partner_type_col = 'type_post'
                    partner_bid_col = 'bodyId_post'
                else:
                    continue
            
            hop2_df = conn_df[mask].copy()
            
            if hop2_df.empty:
                results[untyped_bid] = ({}, {})
                continue
            
            # Filter to only typed 2-hop partners
            if partner_type_col in hop2_df.columns:
                typed_mask = hop2_df[partner_type_col].notna() & (hop2_df[partner_type_col].astype(str).str.strip() != '')
                hop2_df = hop2_df[typed_mask]
            else:
                results[untyped_bid] = ({}, {})
                continue
            
            if hop2_df.empty:
                results[untyped_bid] = ({}, {})
                continue
            
            # Apply fuzzy matching
            if self.config.fuzzy_match.enabled:
                hop2_df['partner_type_normalized'] = hop2_df[partner_type_col].apply(
                    lambda x: normalize_partner_type(x, self.config.fuzzy_match)
                )
            else:
                hop2_df['partner_type_normalized'] = hop2_df[partner_type_col].astype(str)
            
            # Sort by weight and take top-k
            hop2_df = hop2_df.sort_values('weight', ascending=False).head(top_k_2hop)
            
            # Aggregate by type
            aggregated = hop2_df.groupby('partner_type_normalized').agg({
                'weight': 'sum'
            }).reset_index()
            
            # Normalize weights
            total_weight = aggregated['weight'].sum()
            weights_dict = {}
            for _, row in aggregated.iterrows():
                ptype = row['partner_type_normalized']
                weights_dict[ptype] = row['weight'] / total_weight if total_weight > 0 else 0.0
            
            # Compute ranks
            ranks_dict = compute_ranks(weights_dict)
            
            results[untyped_bid] = (weights_dict, ranks_dict)
        
        return results
    
    def _process_connections(
        self,
        conn_df: pd.DataFrame,
        direction: str,
        top_k: int
    ) -> Tuple[Dict[str, float], Dict[str, int], int, float, float, int, int, Dict[int, str], int, Dict[int, float]]:
        """
        Process connection DataFrame into normalized weights and ranks.
        
        Round 5: Added dynamic expansion to ensure minimum unique types,
        and bodyId→type mapping for metadata.
        
        Round 6: Also returns untyped partner bodyIds with their weights
        for 2-hop expansion.
        
        Args:
            conn_df: DataFrame with partner_type, weight columns
            direction: 'upstream' or 'downstream' (for logging)
            top_k: Number of top partners to keep
        
        Returns:
            Tuple of (partners_dict, ranks_dict, untyped_count, 
                     untyped_weight_fraction, total_weight, actual_count,
                     unique_types, type_mapping, k_used, untyped_bodyids)
            where untyped_bodyids = {bodyId → weight} for untyped 1-hop partners
        """
        empty_result = ({}, {}, 0, 0.0, 0.0, 0, 0, {}, top_k, {})
        
        if conn_df.empty:
            return empty_result
        
        # Ensure partner_type column exists
        if 'partner_type' not in conn_df.columns:
            return empty_result
        
        # Track untyped partners
        untyped_mask = conn_df['partner_type'].isna() | (conn_df['partner_type'].astype(str).str.strip() == '')
        untyped_count = untyped_mask.sum()
        untyped_weight = conn_df.loc[untyped_mask, 'weight'].sum() if untyped_count > 0 else 0.0
        total_weight = conn_df['weight'].sum()
        
        # Round 6: Extract untyped partner bodyIds and weights for 2-hop expansion
        untyped_bodyids_dict = {}
        if self.config.expand_untyped_2hop and 'partner_bodyId' in conn_df.columns:
            untyped_df = conn_df[untyped_mask].copy()
            if not untyped_df.empty:
                # Sort by weight and take top-k untyped
                untyped_df = untyped_df.sort_values('weight', ascending=False).head(self.config.top_k_bodyid)
                for _, row in untyped_df.iterrows():
                    try:
                        bid = int(row['partner_bodyId'])
                        w = float(row['weight'])
                        untyped_bodyids_dict[bid] = w
                    except (ValueError, TypeError):
                        pass
        
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
            # Round 6: Return with untyped bodyids even if no typed partners
            return ({}, {}, untyped_count, untyped_weight / total_weight if total_weight > 0 else 0.0, 
                    total_weight, 0, 0, {}, top_k, untyped_bodyids_dict)
        
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
        
        # Store actual synapse weights (not normalized)
        # This preserves rank structure and enables easy bodyId→type aggregation via sum
        partners_dict = {}
        for _, row in aggregated.iterrows():
            partners_dict[row['partner_type_normalized']] = float(row['weight'])
        
        # Compute ranks (derived from weights: higher weight = lower rank number)
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
            k_used,
            untyped_bodyids_dict,  # Round 6: untyped bodyId → weight
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
        
        # Process upstream (Round 5/6: expanded return tuple with untyped bodyids)
        (up_partners, up_ranks, up_untyped_count, up_untyped_frac, 
         up_total_weight, up_actual_count, up_unique_types,
         up_type_mapping, up_k_used, up_untyped_bodyids) = self._process_connections(
            upstream_df, 'upstream', self.config.top_k_bodyid
        )
        
        # Process downstream (Round 5/6: expanded return tuple with untyped bodyids)
        (down_partners, down_ranks, down_untyped_count, down_untyped_frac,
         down_total_weight, down_actual_count, down_unique_types,
         down_type_mapping, down_k_used, down_untyped_bodyids) = self._process_connections(
            downstream_df, 'downstream', self.config.top_k_bodyid
        )
        
        # Round 6: Fetch 2-hop partners for untyped 1-hop partners
        up_2hop_weights = None
        up_2hop_ranks = None
        down_2hop_weights = None
        down_2hop_ranks = None
        
        if self.config.expand_untyped_2hop:
            # Fetch 2-hop for upstream untyped partners
            if up_untyped_bodyids:
                up_2hop_data = self._fetch_2hop_partners(
                    list(up_untyped_bodyids.keys()), dataset, 'upstream'
                )
                if up_2hop_data:
                    up_2hop_weights = {bid: data[0] for bid, data in up_2hop_data.items() if data[0]}
                    up_2hop_ranks = {bid: data[1] for bid, data in up_2hop_data.items() if data[1]}
            
            # Fetch 2-hop for downstream untyped partners
            if down_untyped_bodyids:
                down_2hop_data = self._fetch_2hop_partners(
                    list(down_untyped_bodyids.keys()), dataset, 'downstream'
                )
                if down_2hop_data:
                    down_2hop_weights = {bid: data[0] for bid, data in down_2hop_data.items() if data[0]}
                    down_2hop_ranks = {bid: data[1] for bid, data in down_2hop_data.items() if data[1]}
        
        # Create profile with Round 5 and Round 6 fields
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
            # Round 6 fields: 2-hop expansion for untyped partners
            untyped_upstream_bodyids=up_untyped_bodyids if up_untyped_bodyids else None,
            untyped_downstream_bodyids=down_untyped_bodyids if down_untyped_bodyids else None,
            untyped_upstream_2hop=up_2hop_weights if up_2hop_weights else None,
            untyped_downstream_2hop=down_2hop_weights if down_2hop_weights else None,
            untyped_upstream_2hop_ranks=up_2hop_ranks if up_2hop_ranks else None,
            untyped_downstream_2hop_ranks=down_2hop_ranks if down_2hop_ranks else None,
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
    
    def get_bodyids_for_type(
        self,
        neuron_type: str,
        dataset: str
    ) -> List[int]:
        """
        Get all bodyIds for a neuron type in a dataset.
        
        Args:
            neuron_type: Type name
            dataset: Dataset identifier
        
        Returns:
            List of bodyIds
        """
        dataset_lower = dataset.lower()
        
        if 'flywire' in dataset_lower or 'fafb' in dataset_lower or 'banc' in dataset_lower:
            # Local dataset - load neurons file
            src_dir = Path(__file__).parent.parent
            project_root = src_dir.parent
            datasets_folder = project_root / 'datasets'
            safe_name = dataset.replace(':', '_').replace('.', '_')
            dataset_path = datasets_folder / safe_name
            
            # Try to load neurons file
            neurons_files = [
                dataset_path / f'{safe_name}_neurons.parquet',
                dataset_path / f'{safe_name}_neurons.csv',
                dataset_path / 'neurons.parquet',
                dataset_path / 'neurons.csv',
            ]
            
            for neurons_file in neurons_files:
                if neurons_file.exists():
                    try:
                        if str(neurons_file).endswith('.parquet'):
                            df = pd.read_parquet(neurons_file)
                        else:
                            df = pd.read_csv(neurons_file)
                        
                        # Find bodyId column
                        bid_col = None
                        for col in ['bodyId', 'pt_root_id', 'root_id', 'body_id']:
                            if col in df.columns:
                                bid_col = col
                                break
                        
                        # Find type column
                        type_col = None
                        for col in ['type', 'cell_type', 'celltype']:
                            if col in df.columns:
                                type_col = col
                                break
                        
                        if bid_col and type_col:
                            mask = df[type_col] == neuron_type
                            return df.loc[mask, bid_col].astype(int).tolist()
                    except Exception as e:
                        self._log(f"Warning: Could not load neurons from {neurons_file}: {e}")
            
            return []
        
        else:
            # NeuPrint query
            client = self._get_client_for_dataset(dataset)
            if client is None:
                return []
            
            try:
                query = f"""
                MATCH (n:Neuron)
                WHERE n.type = '{neuron_type}'
                RETURN n.bodyId AS bodyId
                """
                result = client.fetch_custom(query)
                return result['bodyId'].astype(int).tolist()
            except Exception as e:
                self._log(f"Warning: Could not get bodyIds for {neuron_type} in {dataset}: {e}")
                return []
    
    def get_type_profile_from_bodyids(
        self,
        neuron_type: str,
        dataset: str,
        force_refresh: bool = False
    ) -> ConnectivityProfile:
        """
        Get type-level profile by fetching individual bodyId profiles and aggregating.
        
        This is the recommended method for type-level profiles as it:
        1. Fetches each bodyId's profile separately (ensuring bodyId-level caching)
        2. Aggregates using weight summation (proper for actual synapse weights)
        3. Derives ranks from aggregated weights
        
        Args:
            neuron_type: Type name
            dataset: Dataset identifier
            force_refresh: Bypass cache if True
        
        Returns:
            Aggregated ConnectivityProfile for the type
        
        Example:
            >>> type_profile = profiler.get_type_profile_from_bodyids('aMe12', 'hemibrain:v1.2.1')
            >>> print(type_profile.num_neurons_aggregated)  # Number of bodyIds aggregated
        """
        # Get all bodyIds for this type
        bodyids = self.get_bodyids_for_type(neuron_type, dataset)
        
        if not bodyids:
            self._log(f"No bodyIds found for {neuron_type} in {dataset}")
            # Return empty profile
            return ConnectivityProfile(
                neuron_id=neuron_type,
                dataset=dataset,
                upstream_partners={},
                downstream_partners={},
                upstream_ranks={},
                downstream_ranks={},
            )
        
        self._log(f"Fetching profiles for {len(bodyids)} bodyIds of {neuron_type} in {dataset}")
        
        # Fetch individual profiles
        bodyid_profiles = []
        for bid in bodyids:
            try:
                profile = self.get_profile(bid, dataset, force_refresh)
                if profile is not None:
                    bodyid_profiles.append(profile)
            except Exception as e:
                self._log(f"Warning: Could not get profile for bodyId {bid}: {e}")
        
        if not bodyid_profiles:
            self._log(f"No profiles extracted for {neuron_type}")
            return ConnectivityProfile(
                neuron_id=neuron_type,
                dataset=dataset,
                upstream_partners={},
                downstream_partners={},
                upstream_ranks={},
                downstream_ranks={},
            )
        
        # Aggregate using the static method
        return ConnectivityProfile.aggregate_bodyid_profiles(
            bodyid_profiles, neuron_type, dataset
        )

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
    
    # =========================================================================
    # Round 7: Cache Building and Homolog Finding Methods
    # =========================================================================
    
    def build_connectivity_profile_cache(
        self,
        dataset: str,
        neuron_types: Optional[List[str]] = None,
        top_k_bodyid: int = 10,
        top_m_type: int = 5,
        expand_untyped_2hop: bool = True,
        force_refresh: bool = False,
        max_neurons: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, ConnectivityProfile]:
        """
        Build connectivity profile cache for all neurons in a dataset.
        
        This method efficiently pre-builds profiles for all (or selected) neuron
        types in a dataset, storing them in the cache for fast lookup during
        homolog finding operations.
        
        Args:
            dataset: Dataset identifier (e.g., 'hemibrain_v1_2_1', 'flywire_FAFB_v783')
            neuron_types: Specific neuron types to cache. If None, caches all types.
            top_k_bodyid: Number of top partners by bodyId to store
            top_m_type: Minimum unique types to ensure via dynamic expansion
            expand_untyped_2hop: Whether to fetch 2-hop profiles for untyped partners
            force_refresh: If True, rebuild cache even if profiles exist
            max_neurons: Maximum number of neurons to cache (for testing)
            progress_callback: Optional callback(current, total, type_name) for progress
        
        Returns:
            Dict mapping neuron_type to ConnectivityProfile
        
        Example:
            >>> profiler = ConnectivityProfiler()
            >>> profiles = profiler.build_connectivity_profile_cache(
            ...     'hemibrain_v1_2_1',
            ...     top_k_bodyid=10,
            ...     expand_untyped_2hop=True
            ... )
            >>> print(f"Cached {len(profiles)} profiles")
        """
        self._log(f"Building connectivity profile cache for {dataset}...")
        
        # Update config for this operation
        original_config = self.config
        self.config = ProfilerConfig(
            top_k_bodyid=top_k_bodyid,
            top_m_type=top_m_type,
            expand_untyped_2hop=expand_untyped_2hop,
            top_k_2hop=self.config.top_k_2hop,
            use_bodyid_for_intra=self.config.use_bodyid_for_intra,
            use_cache=True,
            cache_dir=original_config.cache_dir or Path("cache"),
            verbose=original_config.verbose,
        )
        
        # Get neuron types if not specified
        if neuron_types is None:
            neuron_types = self.get_all_types(dataset)
            if neuron_types is None:
                self._log(f"Could not retrieve types for {dataset}")
                self.config = original_config
                return {}
        
        # Apply max_neurons limit if specified
        if max_neurons is not None and len(neuron_types) > max_neurons:
            neuron_types = neuron_types[:max_neurons]
            self._log(f"Limited to first {max_neurons} neuron types")
        
        total = len(neuron_types)
        self._log(f"Building profiles for {total} neuron types...")
        
        profiles = {}
        failed = []
        batch_size = 50  # Save to parquet every N profiles
        batch_profiles = {}
        
        # Temporarily disable per-profile cache saving (we'll batch save)
        save_per_profile = self.config.use_cache
        
        for i, neuron_type in enumerate(neuron_types):
            try:
                # Progress callback
                if progress_callback:
                    progress_callback(i + 1, total, neuron_type)
                
                # Log progress
                if (i + 1) % 50 == 0 or i == 0:
                    self._log(f"Progress: {i + 1}/{total} ({100*(i+1)/total:.1f}%)")
                
                # Get profile (check cache first unless force_refresh)
                if not force_refresh:
                    cached = self._load_from_cache(neuron_type, dataset)
                    if cached is not None:
                        profiles[neuron_type] = cached
                        continue
                
                # Temporarily disable per-profile cache save
                self.config.use_cache = False
                profile = self.get_profile(neuron_type, dataset, force_refresh=True)
                self.config.use_cache = save_per_profile
                
                profiles[neuron_type] = profile
                batch_profiles[neuron_type] = profile
                
                # Batch save every N profiles
                if len(batch_profiles) >= batch_size:
                    self._save_profiles_to_cache_batch(batch_profiles, dataset)
                    batch_profiles = {}
                
            except Exception as e:
                self._log(f"Warning: Failed to build profile for {neuron_type}: {e}")
                failed.append(neuron_type)
        
        # Save remaining batch
        if batch_profiles:
            self._save_profiles_to_cache_batch(batch_profiles, dataset)
        
        # Restore original config
        self.config = original_config
        
        self._log(f"Cache build complete: {len(profiles)} profiles built, "
                  f"{len(failed)} failed")
        
        if failed:
            self._log(f"Failed types: {failed[:10]}{'...' if len(failed) > 10 else ''}")
        
        return profiles
    
    def find_homologs_loose(
        self,
        query_type: str,
        query_dataset: str,
        target_dataset: str,
        direction: str = 'both',
        top_n: int = 10,
        use_ranks: bool = True
    ) -> pd.DataFrame:
        """
        Find potential homologs using loose matching (Jaccard similarity).
        
        This mode compares the typed connectivity profiles at the type level,
        computing both intersection-based and union-based Jaccard similarity.
        
        Similarity metrics:
        - Jaccard Intersection: |A ∩ B| / |A ∩ B| = overlap quality
        - Jaccard Union: |A ∩ B| / |A ∪ B| = standard Jaccard
        - Weighted variants: Use connection weights instead of binary sets
        
        Args:
            query_type: The neuron type to find homologs for
            query_dataset: Dataset containing the query neuron
            target_dataset: Dataset to search for homologs
            direction: 'upstream', 'downstream', or 'both'
            top_n: Number of top candidates to return
            use_ranks: If True, compare ranks (ordinal); if False, compare weights
        
        Returns:
            DataFrame with columns:
            - target_type: Candidate homolog type
            - jaccard_intersection: Intersection-based Jaccard
            - jaccard_union: Union-based Jaccard
            - weighted_jaccard: Weight-adjusted Jaccard
            - common_partners: Number of shared partner types
            - direction: Which direction was compared
        
        Example:
            >>> results = profiler.find_homologs_loose(
            ...     'aMe12', 'hemibrain_v1_2_1', 
            ...     'flywire_FAFB_v783',
            ...     direction='both'
            ... )
            >>> print(results.head())
        """
        self._log(f"Finding loose homologs for {query_type} from {query_dataset} "
                  f"in {target_dataset}...")
        
        # Get query profile
        query_profile = self.get_profile(query_type, query_dataset)
        
        # Get all types in target dataset
        target_types = self.get_all_types(target_dataset)
        if not target_types:
            self._log(f"No types found in {target_dataset}")
            return pd.DataFrame()
        
        results = []
        
        for target_type in target_types:
            try:
                target_profile = self.get_profile(target_type, target_dataset)
                
                # Compute similarities
                scores = self._compute_jaccard_similarity(
                    query_profile, target_profile, direction, use_ranks
                )
                
                if scores:
                    for score_dict in scores:
                        score_dict['target_type'] = target_type
                        results.append(score_dict)
                        
            except Exception as e:
                # Skip types that fail
                continue
        
        if not results:
            return pd.DataFrame()
        
        # Create results DataFrame
        df = pd.DataFrame(results)
        
        # Sort by combined score (average of intersection and union Jaccard)
        if 'jaccard_union' in df.columns:
            df['combined_score'] = (df['jaccard_intersection'] + df['jaccard_union']) / 2
            df = df.sort_values('combined_score', ascending=False)
        
        return df.head(top_n)
    
    def _compute_jaccard_similarity(
        self,
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str,
        use_ranks: bool
    ) -> List[Dict]:
        """
        Compute Jaccard similarity between two profiles.
        
        Returns list of dicts with scores for each direction.
        """
        results = []
        
        directions = []
        if direction in ['both', 'upstream']:
            directions.append('upstream')
        if direction in ['both', 'downstream']:
            directions.append('downstream')
        
        for dir_name in directions:
            if use_ranks:
                partners_a = getattr(profile_a, f'{dir_name}_ranks', {}) or {}
                partners_b = getattr(profile_b, f'{dir_name}_ranks', {}) or {}
            else:
                partners_a = getattr(profile_a, f'{dir_name}_partners', {}) or {}
                partners_b = getattr(profile_b, f'{dir_name}_partners', {}) or {}
            
            if not partners_a or not partners_b:
                continue
            
            # Get partner type sets
            set_a = set(partners_a.keys())
            set_b = set(partners_b.keys())
            
            intersection = set_a & set_b
            union = set_a | set_b
            
            if not union:
                continue
            
            # Jaccard metrics
            jaccard_intersection = len(intersection) / len(intersection) if intersection else 0.0
            jaccard_union = len(intersection) / len(union) if union else 0.0
            
            # Weighted Jaccard: sum of min weights for intersection / sum of max weights for union
            weighted_intersection = 0.0
            weighted_union = 0.0
            
            for partner in union:
                val_a = partners_a.get(partner, 0.0)
                val_b = partners_b.get(partner, 0.0)
                weighted_intersection += min(val_a, val_b)
                weighted_union += max(val_a, val_b)
            
            weighted_jaccard = weighted_intersection / weighted_union if weighted_union > 0 else 0.0
            
            results.append({
                'direction': dir_name,
                'jaccard_intersection': jaccard_intersection,
                'jaccard_union': jaccard_union,
                'weighted_jaccard': weighted_jaccard,
                'common_partners': len(intersection),
                'total_partners_query': len(set_a),
                'total_partners_target': len(set_b),
            })
        
        return results
    
    def find_homologs_strict(
        self,
        query_type: str,
        query_dataset: str,
        target_dataset: str,
        direction: str = 'both',
        top_n: int = 10,
        min_common_partners: int = 3
    ) -> pd.DataFrame:
        """
        Find potential homologs using strict matching (per-bodyId profiles).
        
        This mode compares profiles at the individual neuron level, computing:
        1. Rank correlation between matching partner types
        2. Per-bodyId Jaccard similarities
        3. 2-hop profile matching for untyped partners (if available)
        
        This is more stringent than loose matching and works well when:
        - You want to match individual neurons, not just types
        - The datasets have good type coverage
        - 2-hop expansion data is available
        
        Args:
            query_type: The neuron type to find homologs for
            query_dataset: Dataset containing the query neuron
            target_dataset: Dataset to search for homologs
            direction: 'upstream', 'downstream', or 'both'
            top_n: Number of top candidates to return
            min_common_partners: Minimum shared partners required
        
        Returns:
            DataFrame with columns:
            - target_type: Candidate homolog type
            - rank_correlation: Spearman correlation of partner ranks
            - jaccard_typed: Jaccard on typed partners only
            - jaccard_2hop: Jaccard including 2-hop expanded partners
            - combined_score: Weighted combination of metrics
            - direction: Which direction was compared
        
        Example:
            >>> results = profiler.find_homologs_strict(
            ...     'aMe12', 'hemibrain_v1_2_1',
            ...     'flywire_FAFB_v783',
            ...     direction='both',
            ...     min_common_partners=3
            ... )
        """
        self._log(f"Finding strict homologs for {query_type} from {query_dataset} "
                  f"in {target_dataset}...")
        
        from scipy.stats import spearmanr
        
        # Get query profile
        query_profile = self.get_profile(query_type, query_dataset)
        
        # Get all types in target dataset
        target_types = self.get_all_types(target_dataset)
        if not target_types:
            self._log(f"No types found in {target_dataset}")
            return pd.DataFrame()
        
        results = []
        
        for target_type in target_types:
            try:
                target_profile = self.get_profile(target_type, target_dataset)
                
                # Compute strict similarity
                scores = self._compute_strict_similarity(
                    query_profile, target_profile, direction, min_common_partners
                )
                
                if scores:
                    for score_dict in scores:
                        score_dict['target_type'] = target_type
                        results.append(score_dict)
                        
            except Exception as e:
                continue
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # Sort by combined score
        if 'combined_score' in df.columns:
            df = df.sort_values('combined_score', ascending=False)
        elif 'rank_correlation' in df.columns:
            df = df.sort_values('rank_correlation', ascending=False)
        
        return df.head(top_n)
    
    def _compute_strict_similarity(
        self,
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str,
        min_common: int
    ) -> List[Dict]:
        """
        Compute strict similarity using rank correlation and 2-hop matching.
        """
        from scipy.stats import spearmanr
        
        results = []
        
        directions = []
        if direction in ['both', 'upstream']:
            directions.append('upstream')
        if direction in ['both', 'downstream']:
            directions.append('downstream')
        
        for dir_name in directions:
            # Get typed partner ranks
            ranks_a = getattr(profile_a, f'{dir_name}_ranks', {}) or {}
            ranks_b = getattr(profile_b, f'{dir_name}_ranks', {}) or {}
            
            if not ranks_a or not ranks_b:
                continue
            
            # Find common partners
            set_a = set(ranks_a.keys())
            set_b = set(ranks_b.keys())
            common = set_a & set_b
            
            if len(common) < min_common:
                continue
            
            # Compute rank correlation on common partners
            ranks_list_a = [ranks_a[p] for p in common]
            ranks_list_b = [ranks_b[p] for p in common]
            
            try:
                corr, p_val = spearmanr(ranks_list_a, ranks_list_b)
                if np.isnan(corr):
                    corr = 0.0
            except Exception:
                corr = 0.0
                p_val = 1.0
            
            # Jaccard on typed partners
            union = set_a | set_b
            jaccard_typed = len(common) / len(union) if union else 0.0
            
            # 2-hop Jaccard (if available)
            jaccard_2hop = 0.0
            twohhop_a = getattr(profile_a, f'untyped_{dir_name}_2hop', {}) or {}
            twohop_b = getattr(profile_b, f'untyped_{dir_name}_2hop', {}) or {}
            
            if twohhop_a and twohop_b:
                # Flatten 2-hop partners
                all_2hop_a = set()
                all_2hop_b = set()
                
                for bid, partners in twohhop_a.items():
                    if isinstance(partners, dict):
                        all_2hop_a.update(partners.keys())
                
                for bid, partners in twohop_b.items():
                    if isinstance(partners, dict):
                        all_2hop_b.update(partners.keys())
                
                if all_2hop_a and all_2hop_b:
                    common_2hop = all_2hop_a & all_2hop_b
                    union_2hop = all_2hop_a | all_2hop_b
                    jaccard_2hop = len(common_2hop) / len(union_2hop) if union_2hop else 0.0
            
            # Combined score: weighted average
            # Weight rank correlation highest, then typed Jaccard, then 2-hop
            combined_score = 0.5 * max(0, corr) + 0.3 * jaccard_typed + 0.2 * jaccard_2hop
            
            results.append({
                'direction': dir_name,
                'rank_correlation': corr,
                'rank_correlation_pval': p_val,
                'jaccard_typed': jaccard_typed,
                'jaccard_2hop': jaccard_2hop,
                'combined_score': combined_score,
                'common_partners': len(common),
                'total_partners_query': len(set_a),
                'total_partners_target': len(set_b),
            })
        
        return results
    
    def get_hybrid_profile_vector(
        self,
        neuron_type: str,
        dataset: str,
        direction: str = 'both'
    ) -> Dict[str, Dict]:
        """
        Get hybrid 1-hop/2-hop profile as a dict (not vectorized).
        
        For typed 1-hop partners: stores the partner type and weight
        For untyped 1-hop partners: stores bodyId with 2-hop type profile
        
        This is useful for intra-dataset comparisons where you want to
        preserve bodyId information, or for cross-dataset comparisons
        where 2-hop profiles help disambiguate untyped partners.
        
        Args:
            neuron_type: Neuron type to get profile for
            dataset: Dataset identifier
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Dict with structure:
            {
                'upstream': {
                    'typed': {partner_type: weight, ...},
                    'typed_ranks': {partner_type: rank, ...},
                    'untyped_2hop': {
                        bodyId: {2hop_type: weight, ...},
                        ...
                    }
                },
                'downstream': {...}
            }
        
        Example:
            >>> hybrid = profiler.get_hybrid_profile_vector('aMe12', 'hemibrain_v1_2_1')
            >>> print(hybrid['upstream']['typed'])
        """
        profile = self.get_profile(neuron_type, dataset)
        
        result = {}
        
        directions = []
        if direction in ['both', 'upstream']:
            directions.append('upstream')
        if direction in ['both', 'downstream']:
            directions.append('downstream')
        
        for dir_name in directions:
            typed_partners = getattr(profile, f'{dir_name}_partners', {}) or {}
            typed_ranks = getattr(profile, f'{dir_name}_ranks', {}) or {}
            untyped_2hop = getattr(profile, f'untyped_{dir_name}_2hop', {}) or {}
            
            result[dir_name] = {
                'typed': typed_partners,
                'typed_ranks': typed_ranks,
                'untyped_2hop': untyped_2hop,
            }
        
        return result
