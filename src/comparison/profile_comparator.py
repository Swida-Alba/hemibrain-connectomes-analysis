"""
Profile Comparison Module

This module provides tools for comparing connectivity profiles across neurons
and datasets using various similarity metrics.

Main components:
- ProfileComparator: Class with methods for computing similarity scores
- ComparisonResult: Dataclass storing comparison results with all metrics

Supported Metrics:
- Jaccard similarity: Set-based overlap of partner types
- Cosine similarity: Weight vector similarity
- Rank correlation: Spearman correlation of partner rankings
- Overlap fraction: Asymmetric overlap measures
- Combined score: Weighted combination of all metrics

Example:
    >>> from src.comparison import ProfileComparator
    >>> 
    >>> similarity = ProfileComparator.weighted_cosine_similarity(
    ...     profile_a, profile_b, direction='both'
    ... )
    >>> print(f"Cosine similarity: {similarity:.3f}")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, TYPE_CHECKING
import os
import warnings
import math
import heapq

import numpy as np
import pandas as pd
from scipy import stats

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

from .connectivity_profiler import ConnectivityProfile

if TYPE_CHECKING:
    from .connectivity_profiler import ConnectivityStatus, ConnectivityProfiler


# ============================================================================
# Default Score Weights
# ============================================================================

# Simplified comparison metrics:
# - Jaccard: Set-based overlap of partner types (0-1)
# - Rank correlation: Spearman correlation normalized to 0-1 using (x+1)/2
# These two metrics work well for both bodyId-level and type-level comparison
DEFAULT_SCORE_WEIGHTS = {
    'jaccard': 0.50,
    'rank': 0.50
}


# ============================================================================
# Comparison Result Dataclass
# ============================================================================

@dataclass
class ComparisonResult:
    """
    Complete comparison result with all metrics.
    
    Attributes:
        profile_a_id: Identifier of first profile
        profile_b_id: Identifier of second profile
        dataset_a: Dataset of first profile
        dataset_b: Dataset of second profile
        direction: Direction compared ('upstream', 'downstream', 'both')
        jaccard: Jaccard similarity score
        cosine: Cosine similarity score
        rank_correlation: Spearman rank correlation
        overlap_a_in_b: Fraction of A's partners found in B (|A ∩ B| / |A|)
        overlap_b_in_a: Fraction of B's partners found in A (|A ∩ B| / |B|)
        combined: Weighted combined score
        confidence: Confidence level ('High', 'Medium', 'Low', 'Very Low')
        weak_connectivity_a: Flag if profile A has weak connectivity
        weak_connectivity_b: Flag if profile B has weak connectivity
    
    Confidence Interpretation:
        - High (>0.7): Profiles match well - type assignment likely correct
        - Medium (0.5-0.7): Some differences - may need review
        - Low (0.3-0.5): Significant differences - type assignment questionable
        - Very Low (<0.3): Profiles differ substantially - likely different neurons
    """
    profile_a_id: str
    profile_b_id: str
    dataset_a: str
    dataset_b: str
    direction: str
    
    # Individual metrics
    jaccard: float
    cosine: float
    rank_correlation: float
    overlap_a_in_b: float  # |A ∩ B| / |A| - fraction of A's partners found in B
    overlap_b_in_a: float  # |A ∩ B| / |B| - fraction of B's partners found in A
    
    # Combined score
    combined: float
    
    # Confidence level
    confidence: str
    
    # Flags
    rank_union: float = np.nan
    weak_connectivity_a: bool = False
    weak_connectivity_b: bool = False
    notes: str = ""  # Additional notes (e.g., "constant_array" when rank correlation is NaN)
    
    @property
    def rank_correlation_norm(self) -> float:
        """Normalized rank correlation in [0, 1] range using (x+1)/2."""
        if np.isnan(self.rank_correlation):
            return np.nan
        return (self.rank_correlation + 1) / 2
    
    def to_dict(self) -> dict:
        """Convert to dictionary for CSV/JSON export."""
        result = {
            'profile_a': self.profile_a_id,
            'profile_b': self.profile_b_id,
            'dataset_a': self.dataset_a,
            'dataset_b': self.dataset_b,
            'direction': self.direction,
            'jaccard': round(self.jaccard, 4),
            'cosine': round(self.cosine, 4),
            'rank_correlation': round(self.rank_correlation, 4) if not np.isnan(self.rank_correlation) else np.nan,
            'rank_correlation_norm': round(self.rank_correlation_norm, 4) if not np.isnan(self.rank_correlation_norm) else np.nan,
            'rank_union': round(self.rank_union, 4) if not np.isnan(self.rank_union) else np.nan,
            'overlap_a_in_b': round(self.overlap_a_in_b, 4),
            'overlap_b_in_a': round(self.overlap_b_in_a, 4),
            'combined': round(self.combined, 4),
            'confidence': self.confidence,
            'weak_connectivity_warning': self.weak_connectivity_a or self.weak_connectivity_b
        }
        if self.notes:
            result['notes'] = self.notes
        return result
    
    def summary(self) -> str:
        """Generate a human-readable summary string."""
        # Handle NaN in rank correlation display
        rank_corr_str = f"{self.rank_correlation:.3f}" if not np.isnan(self.rank_correlation) else "NaN"
        rank_norm_str = f"{self.rank_correlation_norm:.3f}" if not np.isnan(self.rank_correlation_norm) else "NaN"
        
        lines = [
            f"Comparison: {self.profile_a_id} ({self.dataset_a}) vs {self.profile_b_id} ({self.dataset_b})",
            f"  Direction: {self.direction}",
            f"  Combined Score: {self.combined:.4f} ({self.confidence})",
            f"  Metrics: Jaccard={self.jaccard:.3f}, Cosine={self.cosine:.3f}, RankCorr={rank_corr_str} (norm={rank_norm_str})",
            f"  Overlap: A_in_B={self.overlap_a_in_b:.3f}, B_in_A={self.overlap_b_in_a:.3f}"
        ]
        if self.weak_connectivity_a or self.weak_connectivity_b:
            lines.append("  ⚠️ Weak connectivity warning")
        if self.notes:
            lines.append(f"  ℹ️ {self.notes}")
        return "\n".join(lines)
    
    @staticmethod
    def determine_confidence(combined_score: float) -> str:
        """Determine confidence level from combined score."""
        if combined_score >= 0.7:
            return 'High'
        elif combined_score >= 0.5:
            return 'Medium'
        elif combined_score >= 0.3:
            return 'Low'
        else:
            return 'Very Low'


# ============================================================================
# Profile Comparator Class
# ============================================================================

class ProfileComparator:
    """
    Compare connectivity profiles using various similarity metrics.
    
    This class provides static methods for computing different similarity
    measures between ConnectivityProfile objects, enabling cross-dataset
    verification and homolog discovery.
    
    Example:
        >>> comparator = ProfileComparator()
        >>> result = comparator.compare_profiles(profile_a, profile_b)
        >>> print(f"Combined score: {result.combined:.3f}")
    """
    
    @staticmethod
    def _get_partner_sets(
        profile: ConnectivityProfile,
        direction: str
    ) -> set:
        """Get set of partner types for a given direction."""
        if direction == 'upstream':
            return set(profile.upstream_partners.keys())
        elif direction == 'downstream':
            return set(profile.downstream_partners.keys())
        else:  # both
            return (set(profile.upstream_partners.keys()) | 
                    set(profile.downstream_partners.keys()))
    
    @staticmethod
    def _get_weight_vector(
        profile: ConnectivityProfile,
        partners: List[str],
        direction: str
    ) -> np.ndarray:
        """
        Get weight vector for specified partners.
        
        Args:
            profile: ConnectivityProfile to extract weights from
            partners: Ordered list of partner types
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            numpy array of weights (0 if partner not present)
        """
        weights = []
        
        for partner in partners:
            if direction == 'upstream':
                w = profile.upstream_partners.get(partner, 0.0)
            elif direction == 'downstream':
                w = profile.downstream_partners.get(partner, 0.0)
            else:  # both - sum of upstream and downstream
                w = (profile.upstream_partners.get(partner, 0.0) +
                     profile.downstream_partners.get(partner, 0.0))
            weights.append(w)
        
        return np.array(weights)
    
    @staticmethod
    def jaccard_similarity(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both'
    ) -> float:
        """
        Jaccard similarity of partner type sets (binary presence).
        
        Formula: |A ∩ B| / |A ∪ B|
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Float in [0, 1], where 1 = identical partner sets
        
        Notes:
            - Ignores weights, purely set-based
            - Good for "do they connect to the same types?"
            - Returns 0 if both sets are empty
        """
        set_a = ProfileComparator._get_partner_sets(profile_a, direction)
        set_b = ProfileComparator._get_partner_sets(profile_b, direction)
        
        if not set_a and not set_b:
            return 0.0
        
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def weighted_cosine_similarity(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both',
        use_proportions: bool = True
    ) -> float:
        """
        Cosine similarity of weight vectors.
        
        Formula: (A · B) / (||A|| × ||B||)
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            direction: 'upstream', 'downstream', or 'both'
            use_proportions: If True, use normalized proportions (default)
        
        Returns:
            Float in [-1, 1] (typically [0, 1] for non-negative weights)
        
        Notes:
            - Considers relative importance of each partner
            - Robust to total weight differences when using proportions
            - Returns 0 if either vector is all zeros
        """
        # Get union of all partner types
        set_a = ProfileComparator._get_partner_sets(profile_a, direction)
        set_b = ProfileComparator._get_partner_sets(profile_b, direction)
        all_partners = sorted(set_a | set_b)
        
        if not all_partners:
            return 0.0
        
        # Get weight vectors
        vec_a = ProfileComparator._get_weight_vector(profile_a, all_partners, direction)
        vec_b = ProfileComparator._get_weight_vector(profile_b, all_partners, direction)
        
        # Compute cosine similarity
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        dot_product = np.dot(vec_a, vec_b)
        cosine_sim = dot_product / (norm_a * norm_b)
        
        return float(cosine_sim)
    
    @staticmethod
    def rank_correlation(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both',
        method: str = 'spearman',
        use_all_partners: bool = False
    ) -> float:
        """
        Spearman or Kendall rank correlation of partner rankings.
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            direction: 'upstream', 'downstream', or 'both'
            method: 'spearman' or 'kendall'
            use_all_partners: If True, use all partners from both profiles with
                             default low ranks (K+1/M+1) for missing partners.
                             If False (default), only consider shared partners.
        
        Returns:
            Float in [-1, 1], where 1 = identical rankings.
            Returns NaN if insufficient partners (< 3).
        
        Notes:
            - When use_all_partners=False (default), focuses on shared partners only,
              avoiding the problem where many unique partners dilute the correlation
            - When use_all_partners=True, captures overall similarity more comprehensively
              by penalizing missing partners with low default ranks
            - Handles ties appropriately
        """
        # Get ranks for each direction
        if direction == 'upstream':
            ranks_a = profile_a.upstream_ranks
            ranks_b = profile_b.upstream_ranks
        elif direction == 'downstream':
            ranks_a = profile_a.downstream_ranks
            ranks_b = profile_b.downstream_ranks
        else:  # both - average of upstream and downstream correlations
            # Compute upstream and downstream correlations separately, then average
            # This prevents the issue where partners appearing in different directions
            # in different datasets cause artificially low combined correlation
            upstream_corr = ProfileComparator.rank_correlation(
                profile_a, profile_b, direction='upstream', method=method,
                use_all_partners=use_all_partners
            )
            downstream_corr = ProfileComparator.rank_correlation(
                profile_a, profile_b, direction='downstream', method=method,
                use_all_partners=use_all_partners
            )
            
            # Handle NaN cases - use the valid one, or NaN if both are NaN
            if np.isnan(upstream_corr) and np.isnan(downstream_corr):
                return np.nan
            elif np.isnan(upstream_corr):
                return downstream_corr
            elif np.isnan(downstream_corr):
                return upstream_corr
            else:
                return (upstream_corr + downstream_corr) / 2
        
        if use_all_partners:
            # Use union of all partners with default ranks for missing
            all_partners = set(ranks_a.keys()) | set(ranks_b.keys())
            
            if len(all_partners) < 3:
                return np.nan
            
            # Default ranks: K+1 for missing in A, M+1 for missing in B
            k_a = len(ranks_a)  # Number of partners in A
            k_b = len(ranks_b)  # Number of partners in B
            default_rank_a = k_a + 1  # Default rank for partners missing in A
            default_rank_b = k_b + 1  # Default rank for partners missing in B
            
            # Build rank arrays for all partners
            # Use str() key for consistent sorting (may have mixed int/str types)
            all_list = sorted(all_partners, key=lambda x: str(x))
            rank_array_a = [ranks_a.get(p, default_rank_a) for p in all_list]
            rank_array_b = [ranks_b.get(p, default_rank_b) for p in all_list]
        else:
            # Shared partners mode with expansion if needed
            shared = set(ranks_a.keys()) & set(ranks_b.keys())
            all_partners = set(ranks_a.keys()) | set(ranks_b.keys())
            
            min_partners = 5  # Minimum partners needed for meaningful correlation
            
            if len(shared) >= min_partners:
                # Enough shared partners, use them directly
                partner_list = sorted(shared, key=lambda x: str(x))
                rank_array_a = [ranks_a[p] for p in partner_list]
                rank_array_b = [ranks_b[p] for p in partner_list]
            elif len(all_partners) < 2:
                # Not enough total partners
                return np.nan
            else:
                # Expand to min_partners by adding top-ranked partners from union
                # Priority: shared first, then lowest-ranked (most important) from each side
                
                # Start with shared partners
                selected = set(shared)
                
                # Get non-shared partners sorted by their rank (lower = more important)
                non_shared_a = [(p, ranks_a[p]) for p in ranks_a.keys() if p not in shared]
                non_shared_b = [(p, ranks_b[p]) for p in ranks_b.keys() if p not in shared]
                
                # Sort by rank (ascending = most important first)
                non_shared_a.sort(key=lambda x: x[1])
                non_shared_b.sort(key=lambda x: x[1])
                
                # Alternate adding from A and B to balance expansion
                idx_a, idx_b = 0, 0
                while len(selected) < min_partners and (idx_a < len(non_shared_a) or idx_b < len(non_shared_b)):
                    # Add from A
                    if idx_a < len(non_shared_a) and len(selected) < min_partners:
                        selected.add(non_shared_a[idx_a][0])
                        idx_a += 1
                    # Add from B
                    if idx_b < len(non_shared_b) and len(selected) < min_partners:
                        selected.add(non_shared_b[idx_b][0])
                        idx_b += 1
                
                if len(selected) < 2:
                    return np.nan
                
                # Build rank arrays with default ranks for missing partners
                k_a = len(ranks_a)
                k_b = len(ranks_b)
                default_rank_a = k_a + 1
                default_rank_b = k_b + 1
                
                partner_list = sorted(selected)
                rank_array_a = [ranks_a.get(p, default_rank_a) for p in partner_list]
                rank_array_b = [ranks_b.get(p, default_rank_b) for p in partner_list]
        
        # Check for constant arrays before computing correlation
        # (correlation is undefined when one array has no variance)
        arr_a = np.array(rank_array_a)
        arr_b = np.array(rank_array_b)
        if np.std(arr_a) == 0 or np.std(arr_b) == 0:
            # Return NaN and let caller handle it (marked as 'constant_array' in results)
            return np.nan
        
        # Compute correlation (suppress ConstantInputWarning just in case of edge cases)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='An input array is constant')
            if method == 'kendall':
                corr, _ = stats.kendalltau(rank_array_a, rank_array_b)
            else:  # spearman
                corr, _ = stats.spearmanr(rank_array_a, rank_array_b)
        
        # Handle NaN (can occur with constant arrays)
        if np.isnan(corr):
            return np.nan
        
        return float(corr)
    
    @staticmethod
    def overlap_fraction(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both'
    ) -> Tuple[float, float]:
        """
        Fraction of top-K partners that overlap.
        
        Returns:
            Tuple of (overlap_fraction_a, overlap_fraction_b)
            - overlap_fraction_a: |A ∩ B| / |A|
            - overlap_fraction_b: |A ∩ B| / |B|
        
        Notes:
            - Asymmetric measure (A→B may differ from B→A)
            - Useful when one profile has more specific annotations
            - Returns (0, 0) if either profile is empty
        """
        set_a = ProfileComparator._get_partner_sets(profile_a, direction)
        set_b = ProfileComparator._get_partner_sets(profile_b, direction)
        
        if not set_a or not set_b:
            return 0.0, 0.0
        
        intersection = len(set_a & set_b)
        
        overlap_a = intersection / len(set_a) if set_a else 0.0
        overlap_b = intersection / len(set_b) if set_b else 0.0
        
        return overlap_a, overlap_b
    
    @staticmethod
    def combined_score(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        weights: Optional[Dict[str, float]] = None,
        direction: str = 'both'
    ) -> Dict[str, float]:
        """
        Combined similarity score using Jaccard and rank correlation.
        
        Default weights (equal):
        - jaccard: 0.50 (set-based overlap of partner types)
        - rank: 0.50 (Spearman correlation normalized to 0-1)
        
        These metrics work well for both bodyId-level and type-level comparison.
        Cosine similarity is computed but not included in combined score.
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            weights: Custom weight dictionary (optional)
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Dict with scores:
            - 'combined': Weighted combination of jaccard and rank_norm
            - 'jaccard': Jaccard similarity (0-1)
            - 'rank': Original rank correlation (-1 to 1)
            - 'rank_norm': Normalized rank correlation (0-1)
            - 'cosine': Cosine similarity (for reference, not in combined)
            - 'overlap_a_in_b', 'overlap_b_in_a': Overlap fractions (for reference)
        """
        if weights is None:
            weights = DEFAULT_SCORE_WEIGHTS
        
        # Compute individual metrics
        jaccard = ProfileComparator.jaccard_similarity(profile_a, profile_b, direction)
        cosine = ProfileComparator.weighted_cosine_similarity(profile_a, profile_b, direction)
        rank_corr = ProfileComparator.rank_correlation(profile_a, profile_b, direction)
        overlap_a, overlap_b = ProfileComparator.overlap_fraction(profile_a, profile_b, direction)
        
        # Normalize rank_corr from [-1, 1] to [0, 1]
        rank_norm = (rank_corr + 1) / 2 if not np.isnan(rank_corr) else np.nan
        
        # Compute weighted combined score (only Jaccard + normalized rank)
        # Use 0.5 for missing rank (neutral)
        rank_for_combined = 0.5 if np.isnan(rank_norm) else rank_norm
        combined = (
            weights.get('jaccard', 0.50) * jaccard +
            weights.get('rank', 0.50) * rank_for_combined
        )
        
        # Keep overlap values for backward compatibility
        overlap_avg = (overlap_a + overlap_b) / 2
        
        return {
            'combined': combined,
            'jaccard': jaccard,
            'cosine': cosine,
            'rank': rank_corr,  # Original [-1, 1]
            'rank_norm': rank_norm,  # Normalized [0, 1]
            'overlap_a_in_b': overlap_a,
            'overlap_b_in_a': overlap_b,
            'overlap_avg': overlap_avg
        }
    
    @staticmethod
    def _get_all_bodyids(
        profile: ConnectivityProfile,
        direction: str
    ) -> Dict[int, float]:
        """
        Get all partner bodyIds (typed + untyped) for bodyId-level comparison.
        
        Args:
            profile: ConnectivityProfile
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Dict[bodyId, weight] for all partners
        """
        all_bodyids: Dict[int, float] = {}
        
        if direction in ('upstream', 'both'):
            # Add typed upstream bodyIds
            if profile.typed_upstream_bodyids:
                for bid, weight in profile.typed_upstream_bodyids.items():
                    if weight is not None:
                        all_bodyids[bid] = all_bodyids.get(bid, 0.0) + float(weight)
            # Add untyped upstream bodyIds
            if profile.untyped_upstream_bodyids:
                for bid, weight in profile.untyped_upstream_bodyids.items():
                    if weight is not None:
                        all_bodyids[bid] = all_bodyids.get(bid, 0.0) + float(weight)
        
        if direction in ('downstream', 'both'):
            # Add typed downstream bodyIds
            if profile.typed_downstream_bodyids:
                for bid, weight in profile.typed_downstream_bodyids.items():
                    if weight is not None:
                        all_bodyids[bid] = all_bodyids.get(bid, 0.0) + float(weight)
            # Add untyped downstream bodyIds
            if profile.untyped_downstream_bodyids:
                for bid, weight in profile.untyped_downstream_bodyids.items():
                    if weight is not None:
                        all_bodyids[bid] = all_bodyids.get(bid, 0.0) + float(weight)
        
        return all_bodyids
    
    @staticmethod
    def _get_expanded_types(
        profile: ConnectivityProfile,
        direction: str,
        prefix_2hop: bool = True,
        top_n_for_2hop_check: int = 5
    ) -> Dict[str, float]:
        """
        Get expanded type profile including 2-hop expansion for untyped partners.
        
        IMPORTANT: 2-hop expansion is ONLY used when the top-N connections are ALL untyped.
        This prevents diluting the similarity signal with indirect connections when
        direct typed connections are available.
        
        For cross-dataset comparison:
        - If top-N connections include ANY typed partner: use only 1-hop typed partners
        - If top-N connections are ALL untyped: expand to their 2-hop typed partners
          with "2hop:" prefix (e.g., "2hop:Mi1") to distinguish from direct partners
        
        The 2-hop prefix ensures that:
        1. Direct partners and indirect partners are tracked separately
        2. A type appearing as both 1-hop and 2-hop creates two distinct entries
        3. Comparisons can match 1-hop↔1-hop and 2-hop↔2-hop connections
        
        Args:
            profile: ConnectivityProfile
            direction: 'upstream', 'downstream', or 'both'
            prefix_2hop: If True, prefix 2-hop types with "2hop:" (default: True)
            top_n_for_2hop_check: Only use 2-hop if top N connections are all untyped (default: 5)
        
        Returns:
            Dict[type, weight] with 2-hop types prefixed to distinguish from 1-hop
        """
        expanded_types: Dict[str, float] = {}
        hop2_prefix = "2hop:" if prefix_2hop else ""
        
        if direction in ('upstream', 'both'):
            # Check if top-N upstream are all untyped
            # Get typed partner weights
            typed_weights = [(ptype, weight) for ptype, weight in profile.upstream_partners.items()
                           if ptype and ptype != 'untyped']
            # Get untyped partner weights
            untyped_weights = []
            if profile.untyped_upstream_bodyids:
                untyped_weights = list(profile.untyped_upstream_bodyids.values())
            
            # Combine and sort by weight to check top-N
            all_weights = [(w, True) for _, w in typed_weights] + [(w, False) for w in untyped_weights]
            all_weights.sort(key=lambda x: x[0], reverse=True)
            top_n = all_weights[:top_n_for_2hop_check]
            
            # Only use 2-hop if ALL top-N are untyped (is_typed=False)
            use_2hop_upstream = len(top_n) > 0 and all(not is_typed for _, is_typed in top_n)
            
            # Add 1-hop typed partners (no prefix) - always add these
            for ptype, weight in profile.upstream_partners.items():
                if ptype and ptype != 'untyped':
                    expanded_types[ptype] = expanded_types.get(ptype, 0.0) + weight
            
            # Add 2-hop types for untyped upstream partners (with prefix) - only if top-N all untyped
            if use_2hop_upstream and profile.untyped_upstream_2hop:
                for untyped_bid, hop2_types in profile.untyped_upstream_2hop.items():
                    # Weight the 2-hop types by the 1-hop connection weight
                    untyped_weight = 1.0
                    if profile.untyped_upstream_bodyids and untyped_bid in profile.untyped_upstream_bodyids:
                        untyped_weight = profile.untyped_upstream_bodyids[untyped_bid]
                    
                    # Normalize 2-hop weights and scale by 1-hop weight
                    hop2_total = sum(hop2_types.values()) if hop2_types else 1.0
                    for hop2_type, hop2_weight in hop2_types.items():
                        if hop2_type and hop2_type != 'untyped':
                            # Scale by relative 2-hop weight and 1-hop weight
                            scaled_weight = (hop2_weight / hop2_total) * untyped_weight if hop2_total > 0 else 0
                            prefixed_type = f"{hop2_prefix}{hop2_type}"
                            expanded_types[prefixed_type] = expanded_types.get(prefixed_type, 0.0) + scaled_weight
        
        if direction in ('downstream', 'both'):
            # Check if top-N downstream are all untyped
            typed_weights = [(ptype, weight) for ptype, weight in profile.downstream_partners.items()
                           if ptype and ptype != 'untyped']
            untyped_weights = []
            if profile.untyped_downstream_bodyids:
                untyped_weights = list(profile.untyped_downstream_bodyids.values())
            
            all_weights = [(w, True) for _, w in typed_weights] + [(w, False) for w in untyped_weights]
            all_weights.sort(key=lambda x: x[0], reverse=True)
            top_n = all_weights[:top_n_for_2hop_check]
            
            # Only use 2-hop if ALL top-N are untyped
            use_2hop_downstream = len(top_n) > 0 and all(not is_typed for _, is_typed in top_n)
            
            # Add 1-hop typed partners (no prefix) - always add these
            for ptype, weight in profile.downstream_partners.items():
                if ptype and ptype != 'untyped':
                    expanded_types[ptype] = expanded_types.get(ptype, 0.0) + weight
            
            # Add 2-hop types for untyped downstream partners (with prefix) - only if top-N all untyped
            if use_2hop_downstream and profile.untyped_downstream_2hop:
                for untyped_bid, hop2_types in profile.untyped_downstream_2hop.items():
                    # Weight the 2-hop types by the 1-hop connection weight
                    untyped_weight = 1.0
                    if profile.untyped_downstream_bodyids and untyped_bid in profile.untyped_downstream_bodyids:
                        untyped_weight = profile.untyped_downstream_bodyids[untyped_bid]
                    
                    # Normalize 2-hop weights and scale by 1-hop weight
                    hop2_total = sum(hop2_types.values()) if hop2_types else 1.0
                    for hop2_type, hop2_weight in hop2_types.items():
                        if hop2_type and hop2_type != 'untyped':
                            scaled_weight = (hop2_weight / hop2_total) * untyped_weight if hop2_total > 0 else 0
                            prefixed_type = f"{hop2_prefix}{hop2_type}"
                            expanded_types[prefixed_type] = expanded_types.get(prefixed_type, 0.0) + scaled_weight
        
        return expanded_types
    
    @staticmethod
    def bodyid_jaccard(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both'
    ) -> float:
        """
        Jaccard similarity using partner bodyIds directly (for intra-dataset).
        
        Uses all partner bodyIds (typed + untyped) without type aggregation.
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Float in [0, 1], where 1 = identical partner bodyId sets
        """
        bodyids_a = set(ProfileComparator._get_all_bodyids(profile_a, direction).keys())
        bodyids_b = set(ProfileComparator._get_all_bodyids(profile_b, direction).keys())
        
        if not bodyids_a and not bodyids_b:
            return 0.0
        
        intersection = len(bodyids_a & bodyids_b)
        union = len(bodyids_a | bodyids_b)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def bodyid_rank_correlation(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both'
    ) -> float:
        """
        Rank correlation using partner bodyIds directly (for intra-dataset).
        
        Uses weight-based ranks on shared bodyIds.
        
        Args:
            profile_a: First connectivity profile  
            profile_b: Second connectivity profile
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Float in [-1, 1], where 1 = identical rankings
        """
        from scipy.stats import spearmanr
        
        bodyids_a = ProfileComparator._get_all_bodyids(profile_a, direction)
        bodyids_b = ProfileComparator._get_all_bodyids(profile_b, direction)
        
        # Find shared bodyIds
        shared = set(bodyids_a.keys()) & set(bodyids_b.keys())
        
        if len(shared) < 3:
            return np.nan
        
        # Build weight vectors for shared bodyIds
        shared_list = sorted(shared)
        weights_a = [bodyids_a[bid] for bid in shared_list]
        weights_b = [bodyids_b[bid] for bid in shared_list]
        
        # Check for constant arrays (would cause ConstantInputWarning)
        if len(set(weights_a)) <= 1 or len(set(weights_b)) <= 1:
            return np.nan
        
        # Compute Spearman correlation
        try:
            corr, _ = spearmanr(weights_a, weights_b)
            return float(corr) if not np.isnan(corr) else np.nan
        except Exception:
            return np.nan
    
    @staticmethod
    def bodyid_rank_correlation_union(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both'
    ) -> float:
        """
        Rank correlation using UNION of partner bodyIds (for intra-dataset).
        
        Unlike the shared-based version, this uses ALL bodyIds from both profiles.
        Missing partners are assigned weight 0.
        
        Args:
            profile_a: First connectivity profile  
            profile_b: Second connectivity profile
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Float in [-1, 1], where 1 = identical rankings
        """
        from scipy.stats import spearmanr
        
        bodyids_a = ProfileComparator._get_all_bodyids(profile_a, direction)
        bodyids_b = ProfileComparator._get_all_bodyids(profile_b, direction)
        
        # Use UNION of all bodyIds (missing = weight 0)
        all_bodyids = set(bodyids_a.keys()) | set(bodyids_b.keys())
        
        if len(all_bodyids) < 3:
            return np.nan
        
        # Build weight vectors for ALL bodyIds (0 if missing)
        # Convert to string for consistent sorting (bodyIds may be mixed int/str)
        bodyids_list = sorted(all_bodyids, key=lambda x: str(x))
        weights_a = [bodyids_a.get(bid, 0.0) for bid in bodyids_list]
        weights_b = [bodyids_b.get(bid, 0.0) for bid in bodyids_list]
        
        # Check for constant arrays (would cause ConstantInputWarning)
        if len(set(weights_a)) <= 1 or len(set(weights_b)) <= 1:
            return np.nan
        
        # Compute Spearman correlation
        try:
            corr, _ = spearmanr(weights_a, weights_b)
            return float(corr) if not np.isnan(corr) else np.nan
        except Exception:
            return np.nan
    
    @staticmethod
    def expanded_type_rank_correlation_union(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both'
    ) -> float:
        """
        Rank correlation using UNION of expanded types (for cross-dataset).
        
        Unlike the shared-based version, this uses ALL types from both profiles.
        Missing types are assigned weight 0.
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Float in [-1, 1], where 1 = identical rankings
        """
        from scipy.stats import spearmanr
        
        types_a = ProfileComparator._get_expanded_types(profile_a, direction)
        types_b = ProfileComparator._get_expanded_types(profile_b, direction)
        
        # Use UNION of all types (missing = weight 0)
        all_types = set(types_a.keys()) | set(types_b.keys())
        
        if len(all_types) < 3:
            return np.nan
        
        # Build weight vectors for ALL types (0 if missing)
        # Use str() key for safety (types should already be strings)
        types_list = sorted(all_types, key=lambda x: str(x))
        weights_a = [types_a.get(t, 0.0) for t in types_list]
        weights_b = [types_b.get(t, 0.0) for t in types_list]
        
        # Check for constant arrays (would cause ConstantInputWarning)
        if len(set(weights_a)) <= 1 or len(set(weights_b)) <= 1:
            return np.nan
        
        # Compute Spearman correlation
        try:
            corr, _ = spearmanr(weights_a, weights_b)
            return float(corr) if not np.isnan(corr) else np.nan
        except Exception:
            return np.nan
    
    @staticmethod
    def expanded_type_jaccard(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both'
    ) -> float:
        """
        Jaccard similarity using expanded types (for cross-dataset).
        
        Includes 2-hop expansion for untyped 1-hop partners.
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Float in [0, 1], where 1 = identical partner type sets
        """
        types_a = set(ProfileComparator._get_expanded_types(profile_a, direction).keys())
        types_b = set(ProfileComparator._get_expanded_types(profile_b, direction).keys())
        
        if not types_a and not types_b:
            return 0.0
        
        intersection = len(types_a & types_b)
        union = len(types_a | types_b)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def expanded_type_rank_correlation(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both'
    ) -> float:
        """
        Rank correlation using expanded types (for cross-dataset).
        
        Includes 2-hop expansion for untyped 1-hop partners.
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Float in [-1, 1], where 1 = identical rankings
        """
        from scipy.stats import spearmanr
        
        types_a = ProfileComparator._get_expanded_types(profile_a, direction)
        types_b = ProfileComparator._get_expanded_types(profile_b, direction)
        
        # Find shared types
        shared = set(types_a.keys()) & set(types_b.keys())
        
        if len(shared) < 3:
            return np.nan
        
        # Build weight vectors for shared types
        shared_list = sorted(shared)
        weights_a = [types_a[t] for t in shared_list]
        weights_b = [types_b[t] for t in shared_list]
        
        # Check for constant arrays (would cause ConstantInputWarning)
        if len(set(weights_a)) <= 1 or len(set(weights_b)) <= 1:
            return np.nan
        
        # Compute Spearman correlation
        try:
            corr, _ = spearmanr(weights_a, weights_b)
            return float(corr) if not np.isnan(corr) else np.nan
        except Exception:
            return np.nan
    
    @staticmethod
    def combined_score_intra_dataset(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        weights: Optional[Dict[str, float]] = None,
        direction: str = 'both'
    ) -> Dict[str, float]:
        """
        Combined similarity score for intra-dataset (same dataset) comparison.
        
        Uses bodyId-level comparison directly:
        - All partner bodyIds (typed + untyped) are compared
        - No type aggregation or 2-hop expansion
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            weights: Custom weight dictionary (optional)
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Dict with scores (same format as combined_score)
        """
        if weights is None:
            weights = DEFAULT_SCORE_WEIGHTS
        
        # Check for empty profiles (0 partners = no valid connectivity profile)
        bodyids_a = ProfileComparator._get_all_bodyids(profile_a, direction)
        bodyids_b = ProfileComparator._get_all_bodyids(profile_b, direction)
        
        # If either profile has 0 partners, return NaN for all metrics
        if len(bodyids_a) == 0 or len(bodyids_b) == 0:
            return {
                'combined': np.nan,
                'jaccard': np.nan,
                'cosine': np.nan,
                'rank': np.nan,
                'rank_norm': np.nan,
                'rank_union': np.nan,
                'rank_union_norm': np.nan,
                'overlap_a_in_b': np.nan,
                'overlap_b_in_a': np.nan,
                'overlap_avg': np.nan,
                'shared_type_count': 0,
                'union_type_count': len(bodyids_a) + len(bodyids_b)  # 0 for both empty
            }
        
        # Compute bodyId-level metrics
        jaccard = ProfileComparator.bodyid_jaccard(profile_a, profile_b, direction)
        rank_corr = ProfileComparator.bodyid_rank_correlation(profile_a, profile_b, direction)
        rank_union = ProfileComparator.bodyid_rank_correlation_union(profile_a, profile_b, direction)
        
        # Normalize rank_corr from [-1, 1] to [0, 1]
        rank_norm = (rank_corr + 1) / 2 if not np.isnan(rank_corr) else np.nan
        rank_union_norm = (rank_union + 1) / 2 if not np.isnan(rank_union) else np.nan
        
        # Compute weighted combined score
        rank_for_combined = 0.5 if np.isnan(rank_norm) else rank_norm
        combined = (
            weights.get('jaccard', 0.50) * jaccard +
            weights.get('rank', 0.50) * rank_for_combined
        )
        
        # Cosine similarity using bodyId weights (bodyids_a/b already computed above)
        # Ensure all bodyIds are strings for sorting
        all_bodyids = sorted(set(str(k) for k in bodyids_a.keys()) | set(str(k) for k in bodyids_b.keys()))
        
        # Re-map bodyids_a/b to string keys for lookup
        bodyids_a_str = {str(k): v for k, v in bodyids_a.items()}
        bodyids_b_str = {str(k): v for k, v in bodyids_b.items()}
        
        shared_bodyids = set(bodyids_a_str.keys()) & set(bodyids_b_str.keys())
        
        if all_bodyids:
            vec_a = np.array([bodyids_a_str.get(bid, 0.0) for bid in all_bodyids])
            vec_b = np.array([bodyids_b_str.get(bid, 0.0) for bid in all_bodyids])
            norm_a, norm_b = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
            cosine = float(np.dot(vec_a, vec_b) / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0
        else:
            cosine = 0.0
        
        return {
            'combined': combined,
            'jaccard': jaccard,
            'cosine': cosine,
            'rank': rank_corr,
            'rank_norm': rank_norm,
            'rank_union': rank_union,
            'rank_union_norm': rank_union_norm,
            'overlap_a_in_b': 0.0,  # Not computed for bodyId-level
            'overlap_b_in_a': 0.0,
            'overlap_avg': 0.0,
            'shared_type_count': len(shared_bodyids),  # For bodyId comparison, this is shared bodyIds
            'union_type_count': len(all_bodyids)  # Total unique bodyIds in union
        }
    
    @staticmethod
    def combined_score_cross_dataset(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        weights: Optional[Dict[str, float]] = None,
        direction: str = 'both'
    ) -> Dict[str, float]:
        """
        Combined similarity score for cross-dataset comparison.
        
        Uses type-based comparison with 2-hop expansion:
        - 1-hop typed partners are compared directly
        - 1-hop untyped partners are expanded to their 2-hop typed partners
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            weights: Custom weight dictionary (optional)
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Dict with scores (same format as combined_score)
        """
        if weights is None:
            weights = DEFAULT_SCORE_WEIGHTS
        
        # Get expanded types first for empty profile check
        types_a = ProfileComparator._get_expanded_types(profile_a, direction)
        types_b = ProfileComparator._get_expanded_types(profile_b, direction)
        
        # If either profile has 0 partners, return NaN for all metrics
        if len(types_a) == 0 or len(types_b) == 0:
            return {
                'combined': np.nan,
                'jaccard': np.nan,
                'cosine': np.nan,
                'rank': np.nan,
                'rank_norm': np.nan,
                'rank_union': np.nan,
                'rank_union_norm': np.nan,
                'overlap_a_in_b': np.nan,
                'overlap_b_in_a': np.nan,
                'overlap_avg': np.nan,
                'shared_type_count': 0,
                'union_type_count': len(types_a) + len(types_b)  # 0 for both empty
            }
        
        # Compute expanded type metrics
        jaccard = ProfileComparator.expanded_type_jaccard(profile_a, profile_b, direction)
        rank_corr = ProfileComparator.expanded_type_rank_correlation(profile_a, profile_b, direction)
        rank_union = ProfileComparator.expanded_type_rank_correlation_union(profile_a, profile_b, direction)
        
        # Normalize rank_corr from [-1, 1] to [0, 1]
        rank_norm = (rank_corr + 1) / 2 if not np.isnan(rank_corr) else np.nan
        rank_union_norm = (rank_union + 1) / 2 if not np.isnan(rank_union) else np.nan
        
        # Compute weighted combined score
        rank_for_combined = 0.5 if np.isnan(rank_norm) else rank_norm
        combined = (
            weights.get('jaccard', 0.50) * jaccard +
            weights.get('rank', 0.50) * rank_for_combined
        )
        
        # Cosine similarity using expanded type weights (types_a/b already computed above)
        all_types = sorted(set(types_a.keys()) | set(types_b.keys()))
        shared_types = set(types_a.keys()) & set(types_b.keys())
        
        if all_types:
            vec_a = np.array([types_a.get(t, 0.0) for t in all_types])
            vec_b = np.array([types_b.get(t, 0.0) for t in all_types])
            norm_a, norm_b = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
            cosine = float(np.dot(vec_a, vec_b) / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0
        else:
            cosine = 0.0
        
        return {
            'combined': combined,
            'jaccard': jaccard,
            'cosine': cosine,
            'rank': rank_corr,
            'rank_norm': rank_norm,
            'rank_union': rank_union,
            'rank_union_norm': rank_union_norm,
            'overlap_a_in_b': 0.0,  # Not computed for expanded types
            'overlap_b_in_a': 0.0,
            'overlap_avg': 0.0,
            'shared_type_count': len(shared_types),  # Number of shared types for rank correlation
            'union_type_count': len(all_types)  # Total unique types in union
        }

    
    @staticmethod
    def batch_compare_cross_dataset(
        source_profile: ConnectivityProfile,
        target_profiles: Dict[int, ConnectivityProfile],
        candidate_map: Dict[int, int],
        direction: str = 'both'
    ) -> List[Dict[str, Any]]:
        """
        Batch compare one source profile against many targets efficiently.
        
        Optimized for speed by:
        1. Pre-computing source expanded types once
        2. Using vectorized operations where possible
        3. Avoiding repeated function calls
        
        Args:
            source_profile: Source connectivity profile
            target_profiles: Dict mapping target_bid -> ConnectivityProfile
            candidate_map: Dict mapping target_bid -> shared_count (adjacency score)
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            List of score dicts for each valid comparison
        """
        from scipy.stats import spearmanr
        
        results = []
        
        # Pre-compute source expanded types ONCE
        source_types = ProfileComparator._get_expanded_types(source_profile, direction)
        source_type_set = set(source_types.keys())
        source_type_count = len(source_types)
        
        if source_type_count == 0:
            return results
        
        # Process each target
        for target_bid, shared_count in candidate_map.items():
            target_profile = target_profiles.get(target_bid)
            if target_profile is None:
                continue
            
            # Get target expanded types
            target_types = ProfileComparator._get_expanded_types(target_profile, direction)
            target_type_set = set(target_types.keys())
            target_type_count = len(target_types)
            
            if target_type_count == 0:
                results.append({
                    'target_bid': target_bid,
                    'shared_count': shared_count,
                    'combined': np.nan,
                    'jaccard': np.nan,
                    'cosine': np.nan,
                    'rank': np.nan,
                    'rank_union': np.nan,
                    'shared_type_count': 0,
                    'union_type_count': source_type_count,
                    'target_type_count': 0
                })
                continue
            
            # Compute Jaccard (set-based, fast)
            intersection = source_type_set & target_type_set
            union = source_type_set | target_type_set
            jaccard = len(intersection) / len(union) if union else 0.0
            
            # Compute rank correlation on shared types
            shared_types = sorted(intersection)
            if len(shared_types) >= 3:
                weights_a = [source_types[t] for t in shared_types]
                weights_b = [target_types[t] for t in shared_types]
                
                # Check for constant arrays
                if len(set(weights_a)) > 1 and len(set(weights_b)) > 1:
                    try:
                        rank_corr, _ = spearmanr(weights_a, weights_b)
                        rank_corr = float(rank_corr) if not np.isnan(rank_corr) else np.nan
                    except:
                        rank_corr = np.nan
                else:
                    rank_corr = np.nan
            else:
                rank_corr = np.nan
            
            # Compute rank correlation on UNION of types
            all_types = sorted(union)
            if len(all_types) >= 3:
                weights_a_union = [source_types.get(t, 0.0) for t in all_types]
                weights_b_union = [target_types.get(t, 0.0) for t in all_types]
                
                if len(set(weights_a_union)) > 1 and len(set(weights_b_union)) > 1:
                    try:
                        rank_union, _ = spearmanr(weights_a_union, weights_b_union)
                        rank_union = float(rank_union) if not np.isnan(rank_union) else np.nan
                    except:
                        rank_union = np.nan
                else:
                    rank_union = np.nan
            else:
                rank_union = np.nan
            
            # Compute cosine similarity
            if all_types:
                vec_a = np.array([source_types.get(t, 0.0) for t in all_types])
                vec_b = np.array([target_types.get(t, 0.0) for t in all_types])
                norm_a, norm_b = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
                cosine = float(np.dot(vec_a, vec_b) / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0
            else:
                cosine = 0.0
            
            # Compute combined score
            rank_norm = (rank_corr + 1) / 2 if not np.isnan(rank_corr) else 0.5
            
            # If jaccard is 0 (no overlap), combined score should be 0
            if jaccard == 0:
                combined = 0.0
            else:
                combined = 0.5 * jaccard + 0.5 * rank_norm
            
            results.append({
                'target_bid': target_bid,
                'shared_count': shared_count,
                'combined': combined,
                'jaccard': jaccard,
                'cosine': cosine,
                'rank': rank_corr,
                'rank_union': rank_union,
                'shared_type_count': len(shared_types),
                'union_type_count': len(all_types),
                'target_type_count': target_type_count
            })
        
        return results
    
    @staticmethod
    def compare_profiles(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both',
        weights: Optional[Dict[str, float]] = None,
        score_weights: Optional[Dict[str, float]] = None,
    ) -> ComparisonResult:
        """
        Full comparison between two profiles.
        
        Computes all metrics and returns a ComparisonResult object.
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            direction: 'upstream', 'downstream', or 'both'
            weights: Custom score weights (optional)
            score_weights: Alias for weights (compatibility)
        
        Returns:
            ComparisonResult with all metrics and confidence level
        """
        merged_weights = weights if weights is not None else score_weights
        scores = ProfileComparator.combined_score(profile_a, profile_b, weights=merged_weights, direction=direction)
        rank_union = ProfileComparator.expanded_type_rank_correlation_union(profile_a, profile_b, direction)
        
        confidence = ComparisonResult.determine_confidence(scores['combined'])
        
        # Add note if rank correlation is NaN (due to constant arrays)
        notes = ""
        if np.isnan(scores['rank']):
            notes = "Rank correlation undefined (constant input array)"
        
        return ComparisonResult(
            profile_a_id=str(profile_a.neuron_id),
            profile_b_id=str(profile_b.neuron_id),
            dataset_a=profile_a.dataset,
            dataset_b=profile_b.dataset,
            direction=direction,
            jaccard=scores['jaccard'],
            cosine=scores['cosine'],
            rank_correlation=scores['rank'],
            rank_union=rank_union,
            overlap_a_in_b=scores['overlap_a_in_b'],
            overlap_b_in_a=scores['overlap_b_in_a'],
            combined=scores['combined'],
            confidence=confidence,
            weak_connectivity_a=profile_a.is_weak_connectivity,
            weak_connectivity_b=profile_b.is_weak_connectivity,
            notes=notes
        )
    
    @staticmethod
    def compare_profiles_simple(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both'
    ) -> Dict[str, float]:
        """
        Simple comparison between two profiles, returning a dictionary.
        
        This is a lightweight version of compare_profiles() that returns
        a dictionary instead of a ComparisonResult object. Useful for
        bulk comparisons where object creation overhead matters.
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Dict with keys: jaccard, cosine, rank_correlation, shared_count, combined
        """
        scores = ProfileComparator.combined_score(profile_a, profile_b, direction=direction)
        
        # Count shared partners
        if direction == 'upstream':
            shared = set(profile_a.upstream_partners.keys()) & set(profile_b.upstream_partners.keys())
        elif direction == 'downstream':
            shared = set(profile_a.downstream_partners.keys()) & set(profile_b.downstream_partners.keys())
        else:
            shared_up = set(profile_a.upstream_partners.keys()) & set(profile_b.upstream_partners.keys())
            shared_down = set(profile_a.downstream_partners.keys()) & set(profile_b.downstream_partners.keys())
            shared = shared_up | shared_down
        
        return {
            'jaccard': scores['jaccard'],
            'cosine': scores['cosine'],
            'rank_correlation': scores['rank'],
            'shared_count': len(shared),
            'combined': scores['combined']
        }
    
    @staticmethod
    def get_partner_overlap_details(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'upstream'
    ) -> pd.DataFrame:
        """
        Detailed breakdown of partner overlap.
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            direction: 'upstream' or 'downstream'
        
        Returns DataFrame:
            | partner_type | in_a | in_b | weight_a | weight_b | rank_a | rank_b |
            |--------------|------|------|----------|----------|--------|--------|
            | Mi1          | True | True | 0.25     | 0.22     | 1      | 2      |
            | Tm3          | True | True | 0.18     | 0.20     | 2      | 1      |
            | L2           | True | False| 0.12     | NaN      | 3      | NaN    |
        """
        if direction == 'upstream':
            partners_a = profile_a.upstream_partners
            partners_b = profile_b.upstream_partners
            ranks_a = profile_a.upstream_ranks
            ranks_b = profile_b.upstream_ranks
        else:
            partners_a = profile_a.downstream_partners
            partners_b = profile_b.downstream_partners
            ranks_a = profile_a.downstream_ranks
            ranks_b = profile_b.downstream_ranks
        
        # Get all partner types
        all_partners = set(partners_a.keys()) | set(partners_b.keys())
        
        rows = []
        for partner in sorted(all_partners, key=lambda x: str(x)):
            in_a = partner in partners_a
            in_b = partner in partners_b
            
            rows.append({
                'partner_type': partner,
                'in_a': in_a,
                'in_b': in_b,
                'weight_a': partners_a.get(partner, np.nan),
                'weight_b': partners_b.get(partner, np.nan),
                'rank_a': ranks_a.get(partner, np.nan),
                'rank_b': ranks_b.get(partner, np.nan),
                'status': 'shared' if (in_a and in_b) else ('a_only' if in_a else 'b_only')
            })
        
        df = pd.DataFrame(rows)
        
        # Sort by status (shared first) then by average rank
        df['sort_key'] = df.apply(
            lambda r: (0 if r['status'] == 'shared' else 1, 
                       min(r['rank_a'] if not np.isnan(r['rank_a']) else 999,
                           r['rank_b'] if not np.isnan(r['rank_b']) else 999)),
            axis=1
        )
        df = df.sort_values('sort_key').drop(columns=['sort_key'])
        
        return df
    
    @staticmethod
    def compare_multiple_profiles(
        profiles: Dict[str, ConnectivityProfile],
        direction: str = 'both',
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Pairwise comparison of multiple profiles.
        
        Args:
            profiles: Dict mapping identifier to profile
            direction: 'upstream', 'downstream', or 'both'
            weights: Custom score weights (optional)
        
        Returns:
            DataFrame with pairwise comparison results
        """
        results = []
        keys = list(profiles.keys())
        
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                key_a, key_b = keys[i], keys[j]
                profile_a, profile_b = profiles[key_a], profiles[key_b]
                
                result = ProfileComparator.compare_profiles(
                    profile_a, profile_b, direction, weights
                )
                
                result_dict = result.to_dict()
                result_dict['pair'] = f"{key_a} vs {key_b}"
                results.append(result_dict)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def direct_comparison(
        neurons_a: Union[str, int, List[Union[str, int]]],
        neurons_b: Union[str, int, List[Union[str, int]]],
        dataset_a: Optional[str] = None,
        dataset_b: Optional[str] = None,
        # Alternative names to align with callers (e.g., HomologFinder)
        source_dataset: Optional[str] = None,
        target_dataset: Optional[str] = None,
        profiler: Optional[Any] = None,
        direction: str = 'both',
        comparison_mode: str = 'loose',
        label_mapper: Optional[Any] = None,
        score_weights: Optional[Dict[str, float]] = None,
        top_k: int = 15,
        top_m: int = 5,
        min_synapse_threshold: int = 3,
        include_untyped_partners: bool = True,
        min_common_partners: Optional[int] = None,
        same_label_only: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Direct comparison of specific neurons between datasets.
        
        This method provides a unified interface for comparing neurons by:
        - Type name (string)
        - BodyId (integer)
        - LabelMapper (cross-dataset name resolution)
        
        Args:
            neurons_a: Neuron(s) in dataset_a (type names, bodyIds, or list)
            neurons_b: Neuron(s) in dataset_b (type names, bodyIds, or list)
                       If None, uses neurons_a with label_mapper for name resolution
            dataset_a / source_dataset: First dataset identifier
            dataset_b / target_dataset: Second dataset identifier
            profiler: ConnectivityProfiler instance (created if None)
            direction: 'upstream', 'downstream', or 'both'
            comparison_mode: 'loose' (type-level) or 'strict' (bodyId-level). Legacy
                             values 'type' -> 'loose', 'bodyid' -> 'strict'.
            label_mapper: Optional LabelMapper for cross-dataset type name resolution
            score_weights: Custom weights {'jaccard': 0.5, 'rank': 0.5}
            top_k: Number of top partners per direction
            top_m: Minimum unique partner types
            min_synapse_threshold: Minimum synapse count
            include_untyped_partners: Include partners without type
            min_common_partners: Minimum shared partners when using strict/type path
            verbose: Print progress messages
        
        Returns:
            Dict with keys:
            - 'results': DataFrame with pairwise comparison results
            - 'profiles_a': Dict of profiles from dataset_a
            - 'profiles_b': Dict of profiles from dataset_b
            - 'type_summary': Aggregated per-type statistics (strict/type path)
            - 'bodyid_results': BodyId-level pairs (strict/type path)
            - 'summary': Summary statistics
            - 'comparison_mode': Mode used ('loose' or 'strict')
        
        Example:
            >>> # Compare by type name
            >>> results = ProfileComparator.direct_comparison(
            ...     'aMe12', 'aMe12',
            ...     'flywire_FAFB_v783', 'male-cns:v0.9'
            ... )
            >>> 
            >>> # Compare by bodyId
            >>> results = ProfileComparator.direct_comparison(
            ...     [720575940609627403, 720575940622446106],
            ...     [12517, 13190],
            ...     'flywire_FAFB_v783', 'male-cns:v0.9',
            ...     comparison_mode='strict'
            ... )
        """
        from .connectivity_profiler import ConnectivityProfiler, ProfilerConfig
        
        def _log(msg):
            if verbose:
                print(f"[DirectComparison] {msg}")

        # Resolve dataset names, honoring both legacy and new parameter names
        dataset_a = dataset_a or source_dataset
        dataset_b = dataset_b or target_dataset
        if not dataset_a or not dataset_b:
            raise ValueError("Both dataset_a/source_dataset and dataset_b/target_dataset must be provided")
        
        # Normalize inputs to lists
        if not isinstance(neurons_a, list):
            neurons_a = [neurons_a]
        if neurons_b is None:
            # Use neurons_a with label_mapper for name resolution
            neurons_b = neurons_a
        if not isinstance(neurons_b, list):
            neurons_b = [neurons_b]

        # Determine whether inputs are bodyIds (ints) or type names (str)
        def _is_bodyid(x: Any) -> bool:
            return isinstance(x, int)

        all_bodyid_input = all(_is_bodyid(x) for x in neurons_a + neurons_b)

        # Normalize comparison mode
        mode = (comparison_mode or 'loose').lower()
        if mode == 'type':
            mode = 'loose'
        if mode == 'bodyid':
            mode = 'strict'
        if all_bodyid_input:
            mode = 'strict'
        
        # Create profiler if not provided
        if profiler is None:
            config = ProfilerConfig(
                top_k_bodyid=top_k,
                top_m_type=top_m,
                min_synapse_threshold=min_synapse_threshold,
                include_untyped_partners=include_untyped_partners,
                use_cache=True
            )
            profiler = ConnectivityProfiler(
                datasets=[dataset_a, dataset_b],
                config=config,
                verbose=verbose
            )
        
        # Ensure connection caches are complete and pre-warm profiles
        # This mimics HomologFinder.find_homologs_fast progress
        def _ensure_cache_and_prewarm(dataset_name):
            if not dataset_name:
                return
                
            # 1. Ensure connection cache is complete (using FindNeuronConnection)
            try:
                # Try relative import first
                try:
                    from ..coana import FindNeuronConnection
                except ImportError:
                    from coana import FindNeuronConnection
            except ImportError:
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from coana import FindNeuronConnection
            
            if verbose:
                print(f"[DirectComparison] Ensuring connection cache for {dataset_name}...")
            
            # Use token from profiler if available
            token = getattr(profiler, 'token', None)
            
            fnc = FindNeuronConnection(
                dataset=dataset_name,
                token=token,
                use_cache=True,
                verbose_mode='simple',
                simple_fetch=True
            )
            # Build connection cache
            fnc.build_connection_cache(batch_size=1000, quiet=not verbose)
            
            # Cleanup FNC to free memory
            fnc._conn_df_cache = None
            del fnc
            import gc
            gc.collect()
            
            # 2. Pre-warm profile cache
            if verbose:
                print(f"[DirectComparison] Pre-warming profile cache for {dataset_name}...")
            try:
                profiler.consolidate_profile_cache(dataset_name)
            except Exception:
                pass
            try:
                profiler._load_cache_dataframe(dataset_name)
            except Exception:
                pass
            try:
                profiler._get_cached_conn_df(dataset_name)
            except Exception:
                pass

        # Run for both datasets
        _ensure_cache_and_prewarm(dataset_a)
        if dataset_b != dataset_a:
            _ensure_cache_and_prewarm(dataset_b)

        weights = score_weights or DEFAULT_SCORE_WEIGHTS
        
        # Resolve neuron names via label_mapper if provided
        resolved_a = []
        resolved_b = []
        for na in neurons_a:
            if label_mapper and isinstance(na, str):
                mapped = label_mapper.get_mapped_label(na, dataset_a)
                resolved_a.append(mapped if mapped else na)
            else:
                resolved_a.append(na)
        
        for nb in neurons_b:
            if label_mapper and isinstance(nb, str):
                mapped = label_mapper.get_mapped_label(nb, dataset_b)
                resolved_b.append(mapped if mapped else nb)
            else:
                resolved_b.append(nb)
        _log(f"Comparing {len(resolved_a)} neuron(s) from {dataset_a} vs {len(resolved_b)} from {dataset_b} (mode={mode})")

        # Strict mode on type names: delegate to bodyId core for per-bodyId coverage
        if mode == 'strict' and not all_bodyid_input and hasattr(ProfileComparator, 'compare_types_bodyid_core'):
            strict_types = sorted(set(str(t) for t in resolved_a) & set(str(t) for t in resolved_b))
            if strict_types:
                bodyid_df, type_summary = ProfileComparator.compare_types_bodyid_core(
                    profiler=profiler,
                    neuron_types=strict_types,
                    dataset_a=dataset_a,
                    dataset_b=dataset_b,
                    direction=direction,
                    min_common_partners=min_common_partners or 0,
                    score_weights=weights,
                    verbose=verbose,
                )

                rows = []
                for _, row in type_summary.iterrows():
                    rank_corr = row.get('avg_rank_corr', np.nan)
                    rank_union = row.get('avg_rank_union', np.nan)
                    rank_norm = (rank_corr + 1) / 2 if not np.isnan(rank_corr) else np.nan
                    combined = (
                        weights.get('jaccard', 0.5) * row.get('avg_jaccard', np.nan) +
                        weights.get('rank', 0.5) * (0.5 if np.isnan(rank_norm) else rank_norm)
                    )
                    rows.append({
                        'neuron_a': row['neuron_type'],
                        'neuron_b': row['neuron_type'],
                        'type_a': row['neuron_type'],
                        'type_b': row['neuron_type'],
                        'dataset_a': dataset_a,
                        'dataset_b': dataset_b,
                        'direction': direction,
                        'combined': combined,
                        'rank_corr': rank_corr,
                        'rank_corr_norm': rank_norm,
                        'rank_union': rank_union,
                        'rank_union_norm': (rank_union + 1) / 2 if not np.isnan(rank_union) else np.nan,
                        'jaccard': row.get('avg_jaccard', np.nan),
                        'cosine': row.get('avg_cosine', np.nan),
                        'shared_type_count': np.nan,
                        'union_type_count': np.nan,
                        'is_same_type': True,
                        'confidence': ComparisonResult.determine_confidence(combined),
                        'n_source_bodyIds': row.get('n_source_bodyIds', np.nan),
                        'n_target_bodyIds': row.get('n_target_bodyIds', np.nan),
                    })

                results_df = pd.DataFrame(rows)
                summary = {
                    'n_types': len(strict_types),
                    'avg_combined': results_df['combined'].mean() if not results_df.empty else np.nan,
                    'avg_rank_corr': results_df['rank_corr'].mean() if not results_df.empty else np.nan,
                    'avg_jaccard': results_df['jaccard'].mean() if not results_df.empty else np.nan,
                }

                return {
                    'results': results_df.sort_values('combined', ascending=False) if not results_df.empty else results_df,
                    'profiles_a': {},
                    'profiles_b': {},
                    'type_summary': type_summary,
                    'bodyid_results': bodyid_df,
                    'summary': summary,
                    'comparison_mode': mode,
                }

        # Build profiles for loose mode or bodyId-driven strict comparisons
        profiles_a: Dict[str, ConnectivityProfile] = {}
        profiles_b: Dict[str, ConnectivityProfile] = {}

        _log(f"Building profiles for dataset_a ({dataset_a})...")
        for neuron in resolved_a:
            try:
                profile = profiler.get_profile(neuron, dataset_a)
                if profile is not None:
                    key = str(neuron)
                    profiles_a[key] = profile
                else:
                    _log(f"  Warning: No profile found for {neuron} in {dataset_a}")
            except Exception as e:
                _log(f"  Error getting profile for {neuron}: {e}")

        _log(f"Building profiles for dataset_b ({dataset_b})...")
        for neuron in resolved_b:
            try:
                profile = profiler.get_profile(neuron, dataset_b)
                if profile is not None:
                    key = str(neuron)
                    profiles_b[key] = profile
                else:
                    _log(f"  Warning: No profile found for {neuron} in {dataset_b}")
            except Exception as e:
                _log(f"  Error getting profile for {neuron}: {e}")

        if not profiles_a or not profiles_b:
            _log("Warning: Insufficient profiles for comparison")
            return {
                'results': pd.DataFrame(),
                'profiles_a': profiles_a,
                'profiles_b': profiles_b,
                'summary': {},
                'comparison_mode': mode
            }

        # Pairwise comparison
        _log(f"Computing pairwise comparisons ({len(profiles_a)} x {len(profiles_b)})...")
        rows = []

        # Prepare candidate map for batch comparison (all profiles_b are candidates)
        # Use 0 for shared_count as we don't have adjacency info here
        candidate_map = {k: 0 for k in profiles_b.keys()}

        for key_a, profile_a in profiles_a.items():
            # Use batch comparison for efficiency and consistency with HomologFinder
            # This ensures we use the exact same metric calculation logic
            batch_results = ProfileComparator.batch_compare_cross_dataset(
                source_profile=profile_a,
                target_profiles=profiles_b,
                candidate_map=candidate_map,
                direction=direction
            )
            
            # Handle case where source profile is empty (batch_results will be empty)
            if not batch_results:
                # Check if source is empty
                source_types = ProfileComparator._get_expanded_types(profile_a, direction)
                if len(source_types) == 0:
                    # Add NaN rows for all targets to indicate invalid comparison
                    for key_b, profile_b in profiles_b.items():
                        type_a = getattr(profile_a, 'neuron_type', None) or key_a
                        type_b = getattr(profile_b, 'neuron_type', None) or key_b
                        
                        if same_label_only and type_a != type_b:
                            continue
                            
                        rows.append({
                            'neuron_a': key_a,
                            'neuron_b': key_b,
                            'type_a': type_a,
                            'type_b': type_b,
                            'dataset_a': dataset_a,
                            'dataset_b': dataset_b,
                            'direction': direction,
                            'combined': np.nan,
                            'rank_corr': np.nan,
                            'rank_corr_norm': np.nan,
                            'rank_union': np.nan,
                            'rank_union_norm': np.nan,
                            'jaccard': np.nan,
                            'cosine': np.nan,
                            'shared_type_count': 0,
                            'union_type_count': 0,
                            'is_same_type': type_a == type_b,
                            'confidence': 'Low'
                        })
                continue

            for res in batch_results:
                key_b = res['target_bid']
                profile_b = profiles_b[key_b]
                
                # Determine if same type (for reference)
                type_a = getattr(profile_a, 'neuron_type', None) or key_a
                type_b = getattr(profile_b, 'neuron_type', None) or key_b
                if same_label_only and type_a != type_b:
                    continue
                is_same_type = type_a == type_b

                # Recalculate combined score using user-provided weights
                # batch_compare uses default 0.5/0.5, but we want to respect 'weights'
                rank_corr = res['rank']
                rank_union = res['rank_union']
                jaccard = res['jaccard']
                
                rank_corr_norm = (rank_corr + 1) / 2 if not np.isnan(rank_corr) else np.nan
                rank_union_norm = (rank_union + 1) / 2 if not np.isnan(rank_union) else np.nan
                
                # Use rank_norm for combined score (fallback to 0.5 if NaN)
                rank_for_combined = 0.5 if np.isnan(rank_corr_norm) else rank_corr_norm
                
                if jaccard == 0:
                    combined = 0.0
                else:
                    combined = (
                        weights.get('jaccard', 0.50) * jaccard +
                        weights.get('rank', 0.50) * rank_for_combined
                    )

                rows.append({
                    'neuron_a': key_a,
                    'neuron_b': key_b,
                    'type_a': type_a,
                    'type_b': type_b,
                    'dataset_a': dataset_a,
                    'dataset_b': dataset_b,
                    'direction': direction,
                    'combined': combined,
                    'rank_corr': rank_corr,
                    'rank_corr_norm': rank_corr_norm,
                    'rank_union': rank_union,
                    'rank_union_norm': rank_union_norm,
                    'jaccard': jaccard,
                    'cosine': res['cosine'],
                    'shared_type_count': res['shared_type_count'],
                    'union_type_count': res['union_type_count'],
                    'is_same_type': is_same_type,
                    'confidence': ComparisonResult.determine_confidence(combined)
                })

        results_df = pd.DataFrame(rows)
        if not results_df.empty:
            results_df = results_df.sort_values('combined', ascending=False)

        # Build summary
        summary = {}
        if not results_df.empty:
            summary = {
                'n_comparisons': len(results_df),
                'n_neurons_a': len(profiles_a),
                'n_neurons_b': len(profiles_b),
                'avg_combined': results_df['combined'].mean(),
                'max_combined': results_df['combined'].max(),
                'avg_rank_corr': results_df['rank_corr'].mean(),
                'avg_jaccard': results_df['jaccard'].mean(),
                'same_type_matches': results_df['is_same_type'].sum(),
            }
            _log(f"Comparison complete: {summary['n_comparisons']} pairs, avg_combined={summary['avg_combined']:.3f}")

        return {
            'results': results_df,
            'profiles_a': profiles_a,
            'profiles_b': profiles_b,
            'summary': summary,
            'comparison_mode': mode
        }
    
    @staticmethod
    def build_similarity_matrix(
        profiles: Dict[str, ConnectivityProfile],
        metric: str = 'combined',
        direction: str = 'both'
    ) -> pd.DataFrame:
        """
        Build similarity matrix for multiple profiles.
        
        Args:
            profiles: Dict mapping identifier to profile
            metric: 'combined', 'jaccard', 'cosine', or 'rank'
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            DataFrame with similarity matrix (symmetric)
        """
        keys = list(profiles.keys())
        n = len(keys)
        
        # Initialize matrix
        matrix = np.zeros((n, n))
        
        for i in range(n):
            matrix[i, i] = 1.0  # Self-similarity
            for j in range(i + 1, n):
                profile_a = profiles[keys[i]]
                profile_b = profiles[keys[j]]
                
                if metric == 'jaccard':
                    sim = ProfileComparator.jaccard_similarity(profile_a, profile_b, direction)
                elif metric == 'cosine':
                    sim = ProfileComparator.weighted_cosine_similarity(profile_a, profile_b, direction)
                elif metric == 'rank':
                    sim = ProfileComparator.rank_correlation(profile_a, profile_b, direction)
                else:  # combined
                    scores = ProfileComparator.combined_score(profile_a, profile_b, direction=direction)
                    sim = scores['combined']
                
                matrix[i, j] = sim
                matrix[j, i] = sim
        
        return pd.DataFrame(matrix, index=keys, columns=keys)
    
    # ========================================================================
    # Inter-Type Rank Correlation (Cross-Dataset and Intra-Dataset)
    # ========================================================================
    
    @staticmethod
    def compute_inter_type_rank_correlation_matrix(
        profiles_by_dataset: Dict[str, Dict[str, ConnectivityProfile]],
        metric: str = 'rank',
        direction: str = 'both'
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute inter-type rank correlation matrices.
        
        Builds similarity matrices comparing connectivity profiles between 
        different neuron types, both within datasets and across datasets.
        
        Args:
            profiles_by_dataset: Dict of {dataset: {type_name: ConnectivityProfile}}
            metric: Similarity metric to use ('rank', 'combined', 'jaccard', 'cosine')
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Dict containing:
            - 'cross_dataset': DataFrame with cross-dataset pairwise similarities
            - 'intra_dataset': Dict of {dataset: DataFrame} with intra-dataset matrices
            - 'all_types_matrix': Combined matrix of all types across all datasets
        """
        results = {
            'cross_dataset': None,
            'intra_dataset': {},
            'all_types_matrix': None
        }
        
        if not profiles_by_dataset:
            return results
        
        # 1. Build intra-dataset matrices
        for dataset, profiles in profiles_by_dataset.items():
            if len(profiles) >= 2:
                matrix = ProfileComparator.build_similarity_matrix(profiles, metric=metric, direction=direction)
                results['intra_dataset'][dataset] = matrix
        
        # 2. Build cross-dataset inter-type comparison
        # Collect all profiles with dataset-prefixed keys
        all_profiles: Dict[str, ConnectivityProfile] = {}
        for dataset, profiles in profiles_by_dataset.items():
            for type_name, profile in profiles.items():
                key = f"{dataset}:{type_name}"
                all_profiles[key] = profile
        
        if len(all_profiles) >= 2:
            # Build combined matrix
            results['all_types_matrix'] = ProfileComparator.build_similarity_matrix(
                all_profiles, metric=metric, direction=direction
            )
            
            # Build cross-dataset specific comparisons
            cross_dataset_rows = []
            datasets = list(profiles_by_dataset.keys())
            
            for i, ds1 in enumerate(datasets):
                for ds2 in datasets[i+1:]:
                    # Compare each type in ds1 with each type in ds2
                    for type1, profile1 in profiles_by_dataset[ds1].items():
                        for type2, profile2 in profiles_by_dataset[ds2].items():
                            if metric == 'rank':
                                sim = ProfileComparator.rank_correlation(profile1, profile2, direction)
                            elif metric == 'jaccard':
                                sim = ProfileComparator.jaccard_similarity(profile1, profile2, direction)
                            elif metric == 'cosine':
                                sim = ProfileComparator.weighted_cosine_similarity(profile1, profile2, direction)
                            else:  # combined
                                scores = ProfileComparator.combined_score(profile1, profile2, direction=direction)
                                sim = scores['combined']
                            
                            cross_dataset_rows.append({
                                'dataset_1': ds1,
                                'type_1': type1,
                                'dataset_2': ds2,
                                'type_2': type2,
                                'similarity': sim,
                                'same_type': type1 == type2
                            })
            
            if cross_dataset_rows:
                results['cross_dataset'] = pd.DataFrame(cross_dataset_rows)
        
        return results
    
    @staticmethod
    def find_similar_types_across_datasets(
        profiles_by_dataset: Dict[str, Dict[str, ConnectivityProfile]],
        type_name: str,
        metric: str = 'rank',
        direction: str = 'both',
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Find types in other datasets most similar to a given type.
        
        Args:
            profiles_by_dataset: Dict of {dataset: {type_name: ConnectivityProfile}}
            type_name: Name of the type to find similar types for
            metric: Similarity metric ('rank', 'combined', 'jaccard', 'cosine')
            direction: 'upstream', 'downstream', or 'both'
            top_n: Number of top similar types to return per dataset
        
        Returns:
            DataFrame with similar types ranked by similarity
        """
        rows = []
        
        # Find the profile for the target type in any dataset
        source_profile = None
        source_dataset = None
        for dataset, profiles in profiles_by_dataset.items():
            if type_name in profiles:
                source_profile = profiles[type_name]
                source_dataset = dataset
                break
        
        if source_profile is None:
            return pd.DataFrame(columns=['dataset', 'type', 'similarity'])
        
        # Compare with all other types
        for dataset, profiles in profiles_by_dataset.items():
            for other_type, other_profile in profiles.items():
                # Skip self-comparison
                if dataset == source_dataset and other_type == type_name:
                    continue
                
                if metric == 'rank':
                    sim = ProfileComparator.rank_correlation(source_profile, other_profile, direction)
                elif metric == 'jaccard':
                    sim = ProfileComparator.jaccard_similarity(source_profile, other_profile, direction)
                elif metric == 'cosine':
                    sim = ProfileComparator.weighted_cosine_similarity(source_profile, other_profile, direction)
                else:  # combined
                    scores = ProfileComparator.combined_score(source_profile, other_profile, direction=direction)
                    sim = scores['combined']
                
                rows.append({
                    'source_type': type_name,
                    'source_dataset': source_dataset,
                    'target_dataset': dataset,
                    'target_type': other_type,
                    'similarity': sim,
                    'is_same_type': type_name == other_type
                })
        
        if not rows:
            return pd.DataFrame(columns=['source_type', 'source_dataset', 'target_dataset', 
                                         'target_type', 'similarity', 'is_same_type'])
        
        df = pd.DataFrame(rows)
        df = df.sort_values('similarity', ascending=False)
        
        # Return top_n per target dataset
        if top_n is not None and top_n > 0:
            grouped = df.groupby('target_dataset').head(top_n).reset_index(drop=True)
            return grouped
        
        return df


class HomologFinder:
    """
    A class for finding homologs of neurons across datasets using connectivity profiles.
    
    This class provides a unified interface for homolog discovery, supporting:
    - Query by type name or bodyId
    - Search across multiple datasets
    - Novel homolog discovery within the same dataset
    - BodyId-level comparisons with automatic type-level aggregation
    - Fast finding via adjacency expansion for candidate discovery
    - Optional 3D skeleton visualization of top candidates
    
    Profile Rules (consistent with ConnectivityProfiler):
        - top_k_bodyid: Top K partners per direction (default: 15)
        - top_m_type: Minimum unique types to ensure via dynamic expansion (default: 5)
        - Dynamic expansion: If fewer than top_m types after top_k, expand K until M types reached
        - max_expansion_factor: Maximum K = top_k * max_expansion_factor (default: 5)
        - Profiles use the ConnectivityProfiler's 3-tier cache system
    
    Output (always both levels):
        - bodyid_results.csv: BodyId-level comparisons (sorted by source_bodyId, rank_corr)
        - type_summary.csv: Type-level aggregated summary (avg/best/std metrics)
        - homolog_results.csv: Legacy format (sorted by rank_corr only)
    
    Visualization:
        - Optional 3D skeleton visualization of top-n candidates
        - Uses VisualizeSkeleton module for interactive HTML + PNG export
    
    Cache Integration:
        - Uses ConnectivityProfiler's get_profile() for cache-aware profile building
        - Profiles are stored in 3-tier cache: memory → disk-index → disk-df (parquet)
        - Fast mode builds profiles from pre-aggregated connection data
    
    Example:
        >>> from src.comparison import HomologFinder
        >>> 
        >>> # Initialize finder with visualization enabled
        >>> finder = HomologFinder(
        ...     top_k=15,
        ...     top_m=5,
        ...     visualize_skeleton=True,  # Enable 3D visualization
        ...     visualize_top_n=5         # Visualize top 5 candidates
        ... )
        >>> 
        >>> # Find homologs with automatic visualization
        >>> results = finder.find_homologs_fast(
        ...     query='Mi1',
        ...     source_dataset='hemibrain_v1_2_1',
        ...     target_dataset='hemibrain_v1_2_1',
        ...     output_dir='./results'
        ... )
        >>> 
        >>> # Results include both bodyId-level and type-level files
        >>> # Plus visualization/ folder with 3D plots
    """
    
    def __init__(
        self,
        source: Optional[Union[str, int]] = None,
        source_dataset: Optional[str] = None,
        target_dataset: Optional[str] = None,
        output_dir: Optional[str] = None,
        saveas: Optional[str] = None,
        top_n: int = 20,
        top_k: int = 15,
        top_m: int = 5,
        min_synapse_threshold: int = 3,
        include_untyped_partners: bool = True,
        use_cache: bool = True,
        visualize_skeleton: bool = False,
        visualize_top_n: int = 5,
        similarity_metric: Union[str, Dict[str, float]] = 'rank_union',
        score_weights: Optional[Dict[str, float]] = None,
        verbose: bool = True,
        token: str = '',
        vector_prefiltering: bool = False,
    ):
        """
        Initialize HomologFinder with configuration and default parameters.
        
        Module-Level Defaults:
            These attributes are used as defaults by find_homologs() and 
            find_homologs_fast() when not provided as method arguments:
            - source: Query neuron (type name or bodyId)
            - source_dataset: Source dataset
            - target_dataset: Target dataset to search
            - output_dir: Directory to save results
            - saveas: Custom folder name for results
        
        Profile Construction Rules:
            - top_k: Top K partners by weight per direction
            - top_m: Minimum unique partner types to ensure
            - Dynamic expansion: If top_k yields < top_m types, expand K
            - Maximum K = top_k * 5 (max_expansion_factor)
        
        Visualization:
            - visualize_skeleton: If True, generate 3D skeleton plots for top matches
            - visualize_top_n: Number of top candidates to visualize
        
        Args:
            source: Default query neuron (type name or bodyId)
            source_dataset: Default source dataset
            target_dataset: Default target dataset
            output_dir: Default directory to save results
            saveas: Default folder name (auto-generates timestamp if None)
            top_n: Maximum number of candidates to return per source neuron (default: 20)
            top_k: Number of top partners to include per direction (default: 15)
            top_m: Minimum unique partner types to ensure via expansion (default: 5)
            min_synapse_threshold: Minimum synapse count for connections
            include_untyped_partners: Include partners without type annotations
            use_cache: Enable profile caching (uses ConnectivityProfiler cache)
            visualize_skeleton: Generate 3D skeleton visualizations (default: False)
            visualize_top_n: Number of top candidates to visualize (default: 5)
            similarity_metric: Metric for sorting top-N candidates. Can be:
                - str: One of 'rank_union', 'rank_corr', 'combined', 'jaccard', 'cosine'
                - dict: Custom weights like {'jaccard': 0.3, 'rank': 0.7} for combined score
                When dict is provided, computes weighted combination of metrics.
            score_weights: (Deprecated) Use similarity_metric dict instead.
                Custom weights for combined score: {'jaccard': 0.5, 'rank': 0.5}
            verbose: Print progress messages
            token: API token for NeuPrint (if empty, FNC auto-handles from env vars)
        
        Example:
            >>> # Set up finder with visualization
            >>> finder = HomologFinder(
            ...     source='Mi1',
            ...     source_dataset='hemibrain:v1.2.1',
            ...     target_dataset='hemibrain:v1.2.1',
            ...     output_dir='./results',
            ...     visualize_skeleton=True,
            ...     visualize_top_n=5
            ... )
            >>> 
            >>> # Run search - always saves both bodyId and type-level results
            >>> results = finder.find_homologs_fast()
        """
        from .connectivity_profiler import ConnectivityProfiler, ProfilerConfig
        
        # Module-level default parameters
        self.source = source
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.token = token  # API token passed to FNC
        self.top_n = top_n
        
        # Output directory - use default if not specified
        # Default: ~/connectome_analysis/homolog_finding/
        if output_dir is None:
            self.output_dir = os.path.join(
                os.path.expanduser('~'),
                'connectome_analysis',
                'homolog_finding'
            )
        else:
            self.output_dir = output_dir
        self.saveas = saveas
        
        # Visualization settings
        self.visualize_skeleton = visualize_skeleton
        self.visualize_top_n = visualize_top_n
        self.vector_prefiltering = vector_prefiltering
        
        # Similarity metric for sorting - can be str or dict
        # If dict: custom weights for computing combined score
        if isinstance(similarity_metric, dict):
            self.similarity_metric = 'combined'
            self.score_weights = similarity_metric
        else:
            self.similarity_metric = similarity_metric
            self.score_weights = score_weights if score_weights else DEFAULT_SCORE_WEIGHTS
        
        # Profiler configuration
        self.verbose = verbose
        self.use_cache = use_cache
        self.min_synapse_threshold = min_synapse_threshold
        
        config = ProfilerConfig(
            top_k_bodyid=top_k,
            top_m_type=top_m,
            min_synapse_threshold=min_synapse_threshold,
            include_untyped_partners=include_untyped_partners,
            use_cache=use_cache
        )
        
        self.profiler = ConnectivityProfiler(
            datasets=[],  # Empty - datasets determined at query time
            config=config,
            verbose=verbose
        )
        
        # Cache references to profiler's caches
        # We use the profiler's cache system directly for consistency
        
        # Cache for available types per dataset
        self._types_cache: Dict[str, List[str]] = {}
        
        # Cache for bodyIds per type per dataset
        self._bodyids_by_type_cache: Dict[str, Dict[str, List[int]]] = {}
        
        # Cache for loaded connection DataFrames
        self._conn_cache: Dict[str, pd.DataFrame] = {}
        
        # Cache for FindNeuronConnection instances per dataset
        self._fnc_cache: Dict[str, Any] = {}
        self._in_progress_bar = False  # Track if we're inside a progress bar context
        self._batch_size = 1000  # Batch size for profile building with intermediate saves
    
    def _log(self, msg: str):
        """Print message if verbose. Uses tqdm.write() if inside a progress bar."""
        if self.verbose:
            if self._in_progress_bar:
                tqdm.write(f"[HomologFinder] {msg}")
            else:
                print(f"[HomologFinder] {msg}")

    # ------------------------------------------------------------------
    # Shared bodyId-level type comparison core (for ComparisonAnalyzer)
    # ------------------------------------------------------------------
    @staticmethod
    def compare_types_bodyid_core(
        profiler: 'ConnectivityProfiler',
        neuron_types: List[str],
        dataset_a: str,
        dataset_b: str,
        direction: str = 'both',
        min_common_partners: int = 3,
        score_weights: Optional[Dict[str, float]] = None,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compare types via bodyId-level pairs then aggregate to type level.

        Returns
        -------
        bodyid_df : pd.DataFrame
            Per-pair results with columns: neuron_type, source_bodyId, target_bodyId,
            jaccard, cosine, rank_corr, rank_union.
        type_summary : pd.DataFrame
            Aggregated per-type averages (avg_* columns) plus counts.
        """
        results = []
        type_iter = neuron_types
        if verbose:
            type_iter = tqdm(neuron_types, desc=f"{dataset_a} vs {dataset_b} types", unit="type")

        for neuron_type in type_iter:
            try:
                bodyids_a = profiler.get_bodyids_for_type(neuron_type, dataset_a) or []
                bodyids_b = profiler.get_bodyids_for_type(neuron_type, dataset_b) or []
            except Exception:
                bodyids_a, bodyids_b = [], []

            if verbose:
                print(f"[ProfileComparator] Comparing type '{neuron_type}' ({len(bodyids_a)} vs {len(bodyids_b)} bodyIds)")

            for bid_a in bodyids_a:
                profile_a = profiler.get_profile(bid_a, dataset_a)
                if profile_a is None:
                    continue
                for bid_b in bodyids_b:
                    profile_b = profiler.get_profile(bid_b, dataset_b)
                    if profile_b is None:
                        continue

                    scores = ProfileComparator.combined_score_intra_dataset(
                        profile_a,
                        profile_b,
                        weights=score_weights or DEFAULT_SCORE_WEIGHTS,
                        direction=direction,
                    )

                    # Enforce minimum shared partners if requested
                    overlap_count = 0
                    try:
                        partners_a = set(profile_a.get_partner_types(direction))
                        partners_b = set(profile_b.get_partner_types(direction))
                        overlap_count = len(partners_a & partners_b)
                    except Exception:
                        pass

                    if min_common_partners and overlap_count < min_common_partners:
                        scores['rank'] = np.nan
                        scores['rank_union'] = np.nan

                    results.append({
                        'neuron_type': neuron_type,
                        'source_bodyId': bid_a,
                        'target_bodyId': bid_b,
                        'jaccard': scores.get('jaccard'),
                        'cosine': scores.get('cosine'),
                        'rank_corr': scores.get('rank'),
                        'rank_union': scores.get('rank_union'),
                        'overlap_partners': overlap_count,
                    })

        bodyid_df = pd.DataFrame(results)

        if bodyid_df.empty:
            return bodyid_df, pd.DataFrame()

        # Aggregate to type-level averages
        agg_dict = {
            'jaccard': 'mean',
            'cosine': 'mean',
            'rank_corr': 'mean',
            'rank_union': 'mean',
            'source_bodyId': 'nunique',
            'target_bodyId': 'nunique',
        }
        type_summary = bodyid_df.groupby('neuron_type').agg(agg_dict).rename(columns={
            'source_bodyId': 'n_source_bodyIds',
            'target_bodyId': 'n_target_bodyIds',
            'jaccard': 'avg_jaccard',
            'cosine': 'avg_cosine',
            'rank_corr': 'avg_rank_corr',
            'rank_union': 'avg_rank_union',
        }).reset_index()

        return bodyid_df, type_summary


    # ------------------------------------------------------------------------------
    # Compatibility shim: ensure ProfileComparator has compare_types_bodyid_core
    # ------------------------------------------------------------------------------
    # Some entry points (e.g., ComparisonAnalyzer) rely on ProfileComparator exposing
    # a bodyId-level type comparison core. In case this attribute is missing in older
    # environments, attach a fallback implementation here.
    if not hasattr(ProfileComparator, 'compare_types_bodyid_core'):

        def _compare_types_bodyid_core(
            profiler: 'ConnectivityProfiler',
            neuron_types: List[str],
            dataset_a: str,
            dataset_b: str,
            direction: str = 'both',
            min_common_partners: int = 3,
            score_weights: Optional[Dict[str, float]] = None,
            verbose: bool = True,
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
            """Fallback bodyId-level comparison then aggregate to type level."""
            results = []

            for neuron_type in neuron_types:
                try:
                    bodyids_a = profiler.get_bodyids_for_type(neuron_type, dataset_a) or []
                    bodyids_b = profiler.get_bodyids_for_type(neuron_type, dataset_b) or []
                except Exception:
                    bodyids_a, bodyids_b = [], []

                if verbose:
                    print(
                        f"[ProfileComparator] Comparing type '{neuron_type}' "
                        f"({len(bodyids_a)} vs {len(bodyids_b)} bodyIds)"
                    )

                for bid_a in bodyids_a:
                    profile_a = profiler.get_profile(bid_a, dataset_a)
                    if profile_a is None:
                        continue
                    for bid_b in bodyids_b:
                        profile_b = profiler.get_profile(bid_b, dataset_b)
                        if profile_b is None:
                            continue

                        scores = ProfileComparator.combined_score_intra_dataset(
                            profile_a,
                            profile_b,
                            weights=score_weights or DEFAULT_SCORE_WEIGHTS,
                            direction=direction,
                        )

                        # Enforce minimum shared partners if requested
                        overlap_count = 0
                        try:
                            partners_a = set(profile_a.get_partner_types(direction))
                            partners_b = set(profile_b.get_partner_types(direction))
                            overlap_count = len(partners_a & partners_b)
                        except Exception:
                            pass

                        if min_common_partners and overlap_count < min_common_partners:
                            scores['rank'] = np.nan
                            scores['rank_union'] = np.nan

                        results.append(
                            {
                                'neuron_type': neuron_type,
                                'source_bodyId': bid_a,
                                'target_bodyId': bid_b,
                                'jaccard': scores.get('jaccard'),
                                'cosine': scores.get('cosine'),
                                'rank_corr': scores.get('rank'),
                                'rank_union': scores.get('rank_union'),
                                'overlap_partners': overlap_count,
                            }
                        )

            bodyid_df = pd.DataFrame(results)

            if bodyid_df.empty:
                return bodyid_df, pd.DataFrame()

            agg_dict = {
                'jaccard': 'mean',
                'cosine': 'mean',
                'rank_corr': 'mean',
                'rank_union': 'mean',
                'source_bodyId': 'nunique',
                'target_bodyId': 'nunique',
            }

            type_summary = (
                bodyid_df.groupby('neuron_type')
                .agg(agg_dict)
                .rename(
                    columns={
                        'source_bodyId': 'n_source_bodyIds',
                        'target_bodyId': 'n_target_bodyIds',
                        'jaccard': 'avg_jaccard',
                        'cosine': 'avg_cosine',
                        'rank_corr': 'avg_rank_corr',
                        'rank_union': 'avg_rank_union',
                    }
                )
                .reset_index()
            )

            return bodyid_df, type_summary

        ProfileComparator.compare_types_bodyid_core = staticmethod(_compare_types_bodyid_core)  # type: ignore[attr-defined]
    
    def direct_comparison(
        self,
        neurons_a: Union[str, int, List[Union[str, int]]],
        neurons_b: Optional[Union[str, int, List[Union[str, int]]]] = None,
        dataset_a: Optional[str] = None,
        dataset_b: Optional[str] = None,
        direction: str = 'both',
        comparison_mode: str = 'type',
        label_mapper: Optional[Any] = None,
        output_dir: Optional[str] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Direct comparison of specific neurons between datasets.
        
        This is a convenience wrapper around ProfileComparator.direct_comparison()
        that uses the finder's pre-configured profiler and settings.
        
        Args:
            neurons_a: Neuron(s) to compare from dataset_a (type names or bodyIds)
            neurons_b: Neuron(s) to compare from dataset_b. If None and label_mapper
                       is provided, uses neurons_a with name resolution.
            dataset_a: First dataset (default: self.source_dataset)
            dataset_b: Second dataset (default: self.target_dataset)
            direction: 'upstream', 'downstream', or 'both'
            comparison_mode: 'type' (aggregate by type) or 'bodyid' (individual neurons)
            label_mapper: Optional LabelMapper for cross-dataset type name resolution
            output_dir: Directory to save results (default: self.output_dir)
            save_results: If True, save results to CSV
        
        Returns:
            Dict with 'results' DataFrame and 'summary' statistics
        
        Example:
            >>> finder = HomologFinder(source_dataset='flywire_FAFB_v783',
            ...                        target_dataset='male-cns:v0.9')
            >>> results = finder.direct_comparison('aMe12', 'aMe12')
            >>> print(results['summary'])
        """
        # Use instance defaults if not specified
        ds_a = dataset_a or self.source_dataset
        ds_b = dataset_b or self.target_dataset
        out_dir = output_dir or self.output_dir
        
        if not ds_a or not ds_b:
            raise ValueError("Both dataset_a and dataset_b must be specified")
        
        # Call ProfileComparator.direct_comparison
        results = ProfileComparator.direct_comparison(
            neurons_a=neurons_a,
            neurons_b=neurons_b,
            dataset_a=ds_a,
            dataset_b=ds_b,
            profiler=self.profiler,
            direction=direction,
            comparison_mode=comparison_mode,
            label_mapper=label_mapper,
            score_weights=self.score_weights,
            top_k=self.profiler.config.top_k_bodyid,
            top_m=self.profiler.config.top_m_type,
            min_synapse_threshold=self.min_synapse_threshold,
            include_untyped_partners=self.profiler.config.include_untyped_partners,
            verbose=self.verbose
        )
        
        # Save results if requested
        if save_results and not results['results'].empty:
            os.makedirs(out_dir, exist_ok=True)
            
            # Generate filename
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            neurons_str = str(neurons_a[0] if isinstance(neurons_a, list) else neurons_a)
            safe_neurons = neurons_str.replace('/', '_').replace(':', '_')[:20]
            filename = f'direct_comparison_{safe_neurons}_{timestamp}.csv'
            
            filepath = os.path.join(out_dir, filename)
            results['results'].to_csv(filepath, index=False)
            self._log(f"Saved results to: {filepath}")
            results['output_file'] = filepath

            # Save parameters to JSON
            import json
            params = {
                'neurons_a': neurons_a,
                'neurons_b': neurons_b,
                'dataset_a': ds_a,
                'dataset_b': ds_b,
                'direction': direction,
                'comparison_mode': comparison_mode,
                'top_k': self.profiler.config.top_k_bodyid,
                'top_m': self.profiler.config.top_m_type,
                'min_synapse_threshold': self.min_synapse_threshold,
                'include_untyped_partners': self.profiler.config.include_untyped_partners,
                'score_weights': self.score_weights,
                'timestamp': timestamp
            }
            json_filename = f'direct_comparison_{safe_neurons}_{timestamp}_params.json'
            json_filepath = os.path.join(out_dir, json_filename)
            with open(json_filepath, 'w') as f:
                json.dump(params, f, indent=2, default=str)
            self._log(f"Saved parameters to: {json_filepath}")
        
        return results

    def _get_target_bodyids_and_types(self, dataset: str) -> Tuple[List[int], Dict[int, str]]:
        """
        Get all bodyIds and their types from a dataset WITHOUT loading full connection DataFrame.
        
        Uses neuron_index.parquet which is much smaller than connections.parquet.
        Falls back to datasets/ neuron_df if index doesn't exist.
        
        Args:
            dataset: Dataset identifier
            
        Returns:
            Tuple of (list of bodyIds, dict mapping bodyId -> type)
        """
        from pathlib import Path
        import gc
        
        # Try to use Polars for memory-efficient loading
        try:
            import polars as pl
            use_polars = True
        except ImportError:
            use_polars = False
        
        safe_name = dataset.replace(':', '_').replace('.', '_')
        src_dir = Path(__file__).parent.parent
        project_root = src_dir.parent
        
        bodyids = []
        type_lookup = {}
        
        # Try neuron_index.parquet first (smallest file)
        index_path = project_root / 'cache' / safe_name / 'neuron_index.parquet'
        if index_path.exists():
            try:
                if use_polars:
                    index_df_pl = pl.read_parquet(str(index_path))
                    index_df = index_df_pl.to_pandas()
                    del index_df_pl
                else:
                    index_df = pd.read_parquet(index_path)
                if 'bodyId' in index_df.columns:
                    # Convert to int for bodyIds
                    index_df['bodyId'] = pd.to_numeric(index_df['bodyId'], errors='coerce')
                    bodyids = index_df['bodyId'].dropna().astype(int).tolist()
                    
                    if 'type' in index_df.columns:
                        for _, row in index_df.iterrows():
                            bid = int(row['bodyId']) if pd.notna(row['bodyId']) else None
                            if bid is not None:
                                type_lookup[bid] = row['type'] if pd.notna(row['type']) else ''
                    
                    self._log(f"Loaded {len(bodyids)} bodyIds from neuron_index")
                    del index_df
                    gc.collect()
                    return bodyids, type_lookup
            except Exception as e:
                self._log(f"Warning: Could not read neuron_index: {e}")
        
        # Fall back to datasets/ neuron_df
        neuron_df_paths = [
            project_root / 'datasets' / safe_name / f'{safe_name}_allneurons_neuron_df.parquet',
            project_root / 'datasets' / safe_name / f'{safe_name}_allneurons_neuron_df.csv'
        ]
        
        for path in neuron_df_paths:
            if path.exists():
                try:
                    if path.suffix == '.parquet':
                        if use_polars:
                            neuron_df_pl = pl.read_parquet(str(path))
                            neuron_df = neuron_df_pl.to_pandas()
                            del neuron_df_pl
                        else:
                            neuron_df = pd.read_parquet(path)
                    else:
                        neuron_df = pd.read_csv(path)
                    
                    if 'bodyId' in neuron_df.columns:
                        neuron_df['bodyId'] = pd.to_numeric(neuron_df['bodyId'], errors='coerce')
                        bodyids = neuron_df['bodyId'].dropna().astype(int).tolist()
                        
                        type_col = 'type' if 'type' in neuron_df.columns else None
                        if type_col:
                            for _, row in neuron_df.iterrows():
                                bid = int(row['bodyId']) if pd.notna(row['bodyId']) else None
                                if bid is not None:
                                    type_lookup[bid] = row[type_col] if pd.notna(row[type_col]) else ''
                        
                        self._log(f"Loaded {len(bodyids)} bodyIds from {path.name}")
                        del neuron_df
                        gc.collect()
                        return bodyids, type_lookup
                except Exception as e:
                    self._log(f"Warning: Could not read {path}: {e}")
        
        self._log(f"Warning: No neuron list found for {dataset}")
        return bodyids, type_lookup
    
    def _build_profiles_memory_safe(
        self,
        bodyids: List[int],
        dataset: str,
        show_progress: bool = True,
        label: str = "neurons"
    ) -> Dict[int, 'ConnectivityProfile']:
        """
        Build connectivity profiles, then release connection data to free memory.
        
        This method builds profiles with indexed connections for O(1) lookups,
        then releases the connection DataFrame after building to allow loading
        another dataset's connections without OOM.
        
        Memory-efficient cross-dataset workflow:
        1. Load source connections → build source profiles → release
        2. Load target connections → build target profiles → release
        3. Compare profiles (only profiles in memory)
        
        Args:
            bodyids: List of bodyIds to build profiles for
            dataset: Dataset identifier
            show_progress: Show progress bars
            label: Label for progress bar
            
        Returns:
            Dict mapping bodyId -> ConnectivityProfile
        """
        import gc
        
        # Step 1: Ensure connection cache is complete (will fetch if needed)
        self._log(f"Ensuring connection data for {dataset}...")
        cache_ready = self._ensure_connection_cache_complete(dataset)
        if not cache_ready:
            self._log(f"Warning: Connection cache may be incomplete for {dataset}")
        
        # Step 2: Build profiles using existing batch method (indexed)
        # This loads connections, builds indexes, and processes efficiently
        profiles = self._build_profiles_batch(
            bodyids,
            dataset,
            show_progress=show_progress,
            label=label
        )
        
        # Step 3: Release connection data to free memory for next dataset
        self._log(f"Releasing connection data for {dataset}...")
        
        # Clear profiler's connection cache
        try:
            from .connectivity_profiler import _PROFILER_CONN_CACHE
            safe_name = dataset.replace(':', '_').replace('.', '_')
            if safe_name in _PROFILER_CONN_CACHE:
                _PROFILER_CONN_CACHE[safe_name] = {}
        except:
            pass
        
        # Clear FNC's module-level cache
        try:
            from coana import _FNC_CACHE
            safe_name = dataset.replace(':', '_').replace('.', '_')
            if safe_name in _FNC_CACHE:
                if 'conn_df' in _FNC_CACHE[safe_name]:
                    _FNC_CACHE[safe_name]['conn_df'] = None
                # Keep indexes info but clear the data
                _FNC_CACHE[safe_name] = {}
        except:
            pass
        
        # Clear our local cache
        if dataset in self._conn_cache:
            del self._conn_cache[dataset]
        
        # Force garbage collection
        gc.collect()
        
        return profiles
    
    def _build_profiles_batch(
        self,
        bodyids: List[int],
        dataset: str,
        show_progress: bool = True,
        label: str = "target"
    ) -> Dict[int, 'ConnectivityProfile']:
        """
        Build connectivity profiles for a list of bodyIds with proper cache management.
        
        This method:
        1. Loads cached profiles first (checking top-k/top-m requirements)
        2. Identifies which profiles need to be built
        3. Builds profiles in batches with intermediate cache saves
        
        Args:
            bodyids: List of bodyIds to build profiles for
            dataset: Dataset identifier
            show_progress: Show progress bars
            label: Label for progress bar (e.g., "target", "source")
            
        Returns:
            Dict mapping bodyId -> ConnectivityProfile
        """
        import os
        
        profiles: Dict[int, 'ConnectivityProfile'] = {}
        required_top_k = self.profiler.config.top_k_bodyid
        
        # Note: Connection cache completeness is ensured by caller (find_homologs/find_homologs_fast)
        # before calling this method, so we don't need to check again here.

        # Merge any pending batch files first so cached profiles are visible
        try:
            self.profiler.consolidate_profile_cache(dataset)
        except Exception:
            pass
        
        # Step 1: Pre-warm caches to avoid repeated disk reads
        self._log(f"Pre-loading connection cache for {dataset}...")
        try:
            self.profiler._get_cached_conn_df(dataset)
            self._log(f"Connection cache ready for {dataset}")
        except Exception:
            pass
        try:
            self.profiler._load_cache_dataframe(dataset)
        except Exception:
            pass
        
        # Step 2: Check which profiles are already cached with sufficient top-k
        cached_count = 0
        need_build: List[int] = []
        
        self._in_progress_bar = True
        self.profiler._in_progress_bar = True  # Tell profiler to use tqdm.write
        try:
            for bid in tqdm(
                bodyids,
                desc=f"Loading cached {label} profiles",
                disable=not show_progress or not self.verbose,
                leave=True
            ):
                cached = self.profiler._load_from_cache(bid, dataset, required_top_k=required_top_k)
                if cached is not None:
                    profiles[bid] = cached
                    cached_count += 1
                else:
                    need_build.append(bid)
        finally:
            self._in_progress_bar = False
            self.profiler._in_progress_bar = False
        
        if cached_count > 0:
            self._log(f"Loaded {cached_count} {label} profiles from cache (top_k>={required_top_k})")
        
        if not need_build:
            self._log(f"All {len(bodyids)} {label} profiles found in cache")
            return profiles
        
        # Step 3: Build profiles for remaining bodyIds
        n_to_build = len(need_build)
        self._log(f"Building {n_to_build} {label} profiles (batch_size={self._batch_size})")
        
        # Check data availability before building
        try:
            self.profiler.ensure_data_available(dataset, raise_on_missing=True)
        except Exception as e:
            self._log(f"WARNING: Data availability check failed: {e}")
        
        # Flag to track if we were interrupted
        interrupted = False
        
        # Sequential execution with batch saves
        self._log(f"Caching to disk every {self._batch_size} profiles")
        batch_count = 0
        built_in_batch = 0
        total_built = 0
        
        # Enable deferred cache writes for batch saving
        self.profiler._defer_cache_writes = True
        
        self._in_progress_bar = True
        self.profiler._in_progress_bar = True  # Tell profiler to use tqdm.write
        try:
            pbar = tqdm(
                need_build,
                desc=f"Building {label} profiles",
                disable=not show_progress or not self.verbose,
                leave=True
            )
            
            for bid in pbar:
                try:
                    profile = self.profiler.get_profile(bid, dataset)
                    if profile is not None:
                        profiles[bid] = profile
                        built_in_batch += 1
                        total_built += 1
                    
                    # Save cache at batch boundaries (silently)
                    if built_in_batch >= self._batch_size:
                        batch_count += 1
                        # Flush pending cache writes silently
                        self.profiler._defer_cache_writes = False
                        try:
                            self.profiler.flush_pending_cache_writes(silent=True)
                        except Exception:
                            pass
                        # Re-enable deferred writes for next batch
                        self.profiler._defer_cache_writes = True
                        built_in_batch = 0
                except Exception:
                    pass
                    
            pbar.close()
        except KeyboardInterrupt:
            interrupted = True
            self._log(f"\n⚠️  Interrupted! Saving {len(profiles)} profiles built so far...")
        finally:
            self._in_progress_bar = False
            self.profiler._in_progress_bar = False
            # Final flush (silently) - save what we have
            self.profiler._defer_cache_writes = False
            try:
                self.profiler.flush_pending_cache_writes(silent=True)
            except Exception:
                pass
            
            # Consolidate batch files into main cache file (if not interrupted)
            # Even when total_built == 0, consolidation merges any prior pending batches
            if not interrupted:
                try:
                    self.profiler.consolidate_profile_cache(dataset)
                except Exception:
                    pass
        
        if interrupted:
            self._log(f"Saved {len(profiles)} {label} profiles before interrupt")
            raise KeyboardInterrupt("Profile building interrupted by user")
        
        self._log(f"Built {total_built} {label} profiles")
        return profiles
    
    def ensure_data_available(self, datasets: List[str] = None, raise_on_missing: bool = True) -> Dict[str, bool]:
        """
        Ensure connection data is available for datasets before running homolog search.
        
        This method checks if connection data exists for all required datasets.
        Call this before find_homologs() or find_homologs_fast() to get clear
        error messages if data is missing.
        
        Args:
            datasets: List of datasets to check. If None, checks source and target datasets.
            raise_on_missing: If True, raise DataNotAvailableError on first missing dataset.
            
        Returns:
            Dict mapping dataset -> availability (True/False)
            
        Raises:
            DataNotAvailableError: If raise_on_missing=True and any data is missing
            
        Example:
            >>> finder = HomologFinder(source_dataset='flywire_FAFB_v783', ...)
            >>> finder.ensure_data_available()  # Checks both source and target
            {'flywire_FAFB_v783': True, 'hemibrain:v1.2.1': True}
        """
        if datasets is None:
            datasets = []
            if self.source_dataset:
                datasets.append(self.source_dataset)
            if self.target_dataset and self.target_dataset != self.source_dataset:
                datasets.append(self.target_dataset)
        
        availability = {}
        for dataset in datasets:
            try:
                self.profiler.ensure_data_available(dataset, raise_on_missing=raise_on_missing)
                availability[dataset] = True
            except Exception as e:
                availability[dataset] = False
                if raise_on_missing:
                    raise
        
        return availability
    
    def get_data_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed status of connection data for configured datasets.
        
        Returns:
            Dict with status info for each dataset (available, type, path, rows, error)
            
        Example:
            >>> finder.get_data_status()
            {'hemibrain:v1.2.1': {'available': True, 'type': 'neuprint', ...}}
        """
        datasets = []
        if self.source_dataset:
            datasets.append(self.source_dataset)
        if self.target_dataset and self.target_dataset != self.source_dataset:
            datasets.append(self.target_dataset)
        
        return self.profiler.get_data_status(datasets)
    
    def _get_fnc_for_dataset(self, dataset: str) -> Optional[Any]:
        """
        Get or create a FindNeuronConnection instance for a dataset.
        
        Uses FindNeuronConnection module for fetching neuron info instead of
        direct database queries. This provides consistent data access patterns.
        
        Args:
            dataset: Dataset identifier (e.g., 'hemibrain:v1.2.1')
            
        Returns:
            FindNeuronConnection instance or None if not available
        """
        if dataset in self._fnc_cache:
            return self._fnc_cache[dataset]
        
        try:
            from ..coana import FindNeuronConnection
            
            # Determine if this is a FlyWire/local dataset
            dataset_lower = dataset.lower()
            is_flywire = any(x in dataset_lower for x in ['flywire', 'fafb', 'banc'])
            
            # Create FNC instance with minimal settings
            # Pass token from HomologFinder; FNC handles env vars if empty
            fnc = FindNeuronConnection(
                dataset=dataset,
                client_type='flywire' if is_flywire else 'neuprint',
                max_interlayer=-1,  # No connection fetching, just neuron info
                verbose_mode='silent',
                token=getattr(self, 'token', '')
            )
            
            self._fnc_cache[dataset] = fnc
            return fnc
            
        except Exception as e:
            self._log(f"Warning: Could not create FindNeuronConnection for {dataset}: {e}")
            return None
    
    def get_neuron_info(
        self,
        neurons: List[Union[str, int]],
        dataset: str,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get neuron information using the FindNeuronConnection module.
        
        This method provides consistent data access by delegating to FNC,
        which handles caching, local dataset files, and API fallback.
        
        Args:
            neurons: List of bodyIds (int) or type names (str)
            dataset: Dataset identifier
            columns: Specific columns to return (optional)
            
        Returns:
            DataFrame with neuron information
        """
        fnc = self._get_fnc_for_dataset(dataset)
        if fnc is None:
            return pd.DataFrame()
        
        try:
            # Determine if input is bodyIds or types
            if neurons and isinstance(neurons[0], int):
                # BodyId query
                return fnc._fetch_neurons_local_or_api(neurons, columns=columns)
            else:
                # Type query
                return fnc._fetch_neurons_by_types(list(neurons), columns=columns)
        except Exception as e:
            self._log(f"Warning: Could not fetch neuron info via FNC: {e}")
            return pd.DataFrame()
    
    def get_profile(
        self, 
        query: Union[str, int], 
        dataset: str,
        force_refresh: bool = False
    ) -> Optional['ConnectivityProfile']:
        """
        Get connectivity profile for a neuron, using the profiler's cache system.
        
        This delegates to ConnectivityProfiler.get_profile() which uses the 3-tier
        cache system (memory → disk-index → disk-df).
        
        Args:
            query: Neuron type (str) or bodyId (int)
            dataset: Dataset identifier
            force_refresh: Bypass cache if True
            
        Returns:
            ConnectivityProfile or None if not found
        """
        return self.profiler.get_profile(query, dataset, force_refresh=force_refresh)
    
    def get_profiles_batch(
        self,
        neurons: List[Union[str, int]],
        dataset: str,
        force_refresh: bool = False,
        show_progress: bool = True
    ) -> Dict[Union[str, int], 'ConnectivityProfile']:
        """
        Get connectivity profiles for multiple neurons efficiently.
        
        Uses the profiler's batch method with deferred cache writes for
        parallel-safe operation.
        
        Args:
            neurons: List of neuron types or bodyIds
            dataset: Dataset identifier
            force_refresh: Bypass cache if True
            show_progress: Show progress bar
            
        Returns:
            Dict mapping neuron identifier to ConnectivityProfile
        """
        profiles = {}
        
        # Enable deferred writes for batch operation
        self.profiler._defer_cache_writes = True
        self.profiler._pending_cache_writes = {}
        
        try:
            iterator = tqdm(
                neurons,
                desc=f"Building profiles for {dataset}",
                disable=not show_progress or not self.verbose,
                leave=False
            )
            
            for neuron in iterator:
                try:
                    profile = self.profiler.get_profile(neuron, dataset, force_refresh)
                    if profile is not None:
                        profiles[neuron] = profile
                except Exception as e:
                    self._log(f"Warning: Failed for {neuron}: {e}")
                    
        finally:
            # Flush all pending writes at once
            self.profiler._defer_cache_writes = False
            self.profiler.flush_pending_cache_writes()
        
        return profiles

    def _prewarm_profile_cache(self, dataset: str) -> None:
        """Warm up profile cache/index so first access avoids cold disk reads."""
        try:
            self.profiler.consolidate_profile_cache(dataset)
        except Exception:
            pass
        try:
            self.profiler._load_cache_dataframe(dataset)
        except Exception:
            pass
        try:
            self.profiler._get_cached_conn_df(dataset)
        except Exception:
            pass
    
    def get_available_types(self, dataset: str) -> List[str]:
        """Get available types in a dataset, with caching."""
        if dataset not in self._types_cache:
            types = self.profiler.get_available_types(dataset)
            self._types_cache[dataset] = types or []
        return self._types_cache[dataset]
    
    def get_bodyids_for_type(self, neuron_type: str, dataset: str) -> List[int]:
        """
        Get all bodyIds for a neuron type in a dataset.
        
        Uses caching for repeated lookups.
        
        Args:
            neuron_type: Type name
            dataset: Dataset identifier
            
        Returns:
            List of bodyIds
        """
        if dataset not in self._bodyids_by_type_cache:
            self._bodyids_by_type_cache[dataset] = {}
        
        if neuron_type not in self._bodyids_by_type_cache[dataset]:
            bodyids = self.profiler.get_bodyids_for_type(neuron_type, dataset)
            self._bodyids_by_type_cache[dataset][neuron_type] = bodyids or []
        
        return self._bodyids_by_type_cache[dataset][neuron_type]
    
    def find_homologs(
        self,
        source: Optional[Union[str, int]] = None,
        source_dataset: Optional[str] = None,
        target_dataset: Optional[str] = None,
        top_n: Optional[int] = None,
        metric: str = 'combined',
        direction: str = 'both',
        min_score: float = 0.0,
        show_progress: bool = True,
        output_dir: Optional[str] = None,
        saveas: Optional[str] = None,
        include_partner_details: bool = True,
        top_n_details: int = 10,
        run_shuffle_test: bool = False,
        n_shuffles: int = 100,
        shuffle_seed: Optional[int] = None,
        visualize_skeleton: Optional[bool] = None,
        visualize_top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Comprehensive homolog discovery - searches the ENTIRE target dataset.
        
        This method builds connectivity profiles for ALL neurons (typed + untyped)
        in the target dataset, then compares them with source neurons. This is the
        most thorough search but can be slow for large datasets.
        
        Note: Untyped neurons are only ignored in 2-hop profiles during profile
        building, but they are included as candidates in this comprehensive search.
        
        For faster search using adjacency expansion, use find_homologs_fast().
        
        Algorithm:
            1. Build profiles for source neurons (by type or bodyId)
            2. Get ALL bodyIds in target dataset (typed + untyped)
            3. Build profiles for ALL target bodyIds with batch saves
            4. Compare source profiles against all target profiles
            5. Return top-n matches sorted by rank_corr
        
        Uses module-level defaults (self.source, self.source_dataset, etc.) 
        when parameters are not provided.
        
        Args:
            source: Neuron identifier - bodyId (int) or type name (str).
                   Uses self.source if not provided.
            source_dataset: Dataset containing the query neuron.
                           Uses self.source_dataset if not provided.
            target_dataset: Target dataset to search for homologs.
                           Uses self.target_dataset if not provided.
            top_n: Number of top candidates to return per source neuron
            metric: Similarity metric ('combined', 'jaccard', 'cosine', 'rank')
            direction: Connection direction ('upstream', 'downstream', 'both')
            min_score: Minimum similarity score to include
            show_progress: Show progress bar
            output_dir: Directory to save results. Uses self.output_dir if not provided.
            saveas: Custom folder name. Uses self.saveas if not provided.
            include_partner_details: Include per-match partner overlap CSVs (when saving)
            top_n_details: Number of top matches to include detailed profiles for
            run_shuffle_test: If True, run random control test to validate results
            n_shuffles: Number of shuffle iterations for control test (default: 100)
            shuffle_seed: Random seed for reproducibility of shuffle test
            visualize_skeleton: Generate 3D visualizations. Uses self.visualize_skeleton if None.
            visualize_top_n: Number of top candidates to visualize. Uses self.visualize_top_n if None.
        
        Returns:
            DataFrame with columns:
                - source_bodyId: Source neuron bodyId
                - source_type: Source neuron type
                - target_bodyId: Target neuron bodyId
                - target_type: Target neuron type (empty string for untyped)
                - target_dataset: Target dataset
                - jaccard: Jaccard similarity
                - rank_corr: Rank correlation (primary metric for sorting)
                - is_same_type: True if type name matches query
                
            Results are sorted by source_bodyId, then rank_corr (descending).
            
            If run_shuffle_test=True, additional columns are added:
                - shuffle_p_value: P-value from shuffle test
                - shuffle_z_score: Z-score from shuffle test
                - shuffle_significant: True if p < 0.05
        """
        # Use module-level defaults where not provided
        query = source if source is not None else self.source
        source_dataset = source_dataset if source_dataset is not None else self.source_dataset
        target_dataset = target_dataset if target_dataset is not None else self.target_dataset
        output_dir = output_dir if output_dir is not None else self.output_dir
        saveas = saveas if saveas is not None else self.saveas
        visualize_skeleton = visualize_skeleton if visualize_skeleton is not None else self.visualize_skeleton
        visualize_top_n = visualize_top_n if visualize_top_n is not None else self.visualize_top_n
        top_n = top_n if top_n is not None else self.top_n
        
        # Validate required parameters
        if query is None:
            raise ValueError("No source neuron specified. Set 'source' parameter or self.source.")
        if source_dataset is None:
            raise ValueError("No source_dataset specified. Set 'source_dataset' parameter or self.source_dataset.")
        if target_dataset is None:
            raise ValueError("No target_dataset specified. Set 'target_dataset' parameter or self.target_dataset.")
        
        # Determine if query is bodyId or type
        is_bodyid = isinstance(query, int) or (isinstance(query, str) and query.isdigit())
        query_label = f"bodyId:{query}" if is_bodyid else query
        is_cross_dataset = (source_dataset != target_dataset)
        
        self._log(f"Comprehensive homolog search: {query_label} from {source_dataset} → {target_dataset}")
        
        # =====================================================================
        # Step 0: Ensure connection caches are complete for both datasets
        # =====================================================================
        # MEMORY-EFFICIENT APPROACH:
        # 1. Build source profiles (load connections only if needed, then release)
        # 2. Build target profiles (load connections only if needed, then release)
        # 3. Compare profiles (only profiles in memory, not connection DataFrames)
        # =====================================================================
        
        # =====================================================================
        # Step 1: Get source neurons and build their profiles
        # =====================================================================
        if not is_bodyid:
            # Type query: get all bodyIds for this type
            source_bodyids = self.get_bodyids_for_type(str(query), source_dataset)
            if not source_bodyids:
                self._log(f"ERROR: No bodyIds found for type '{query}' in {source_dataset}")
                return pd.DataFrame()
            self._log(f"Found {len(source_bodyids)} source neurons for type '{query}'")
        else:
            source_bodyids = [int(query)]
            self._log(f"Source bodyId: {query}")
        
        # Build source profiles using memory-efficient method
        self._log(f"Building source profiles for {len(source_bodyids)} neurons...")
        source_profiles = self._build_profiles_memory_safe(
            source_bodyids, 
            source_dataset, 
            show_progress=show_progress,
            label="source"
        )
        
        if not source_profiles:
            self._log(f"ERROR: Could not build profiles for any source neurons")
            return pd.DataFrame()
        
        source_type = str(query) if not is_bodyid else None
        
        # =====================================================================
        # Step 2: Get target bodyIds and build their profiles
        # Memory is now free from source connection data
        # =====================================================================
        # Get target bodyIds from neuron index (doesn't require loading full connections)
        all_target_bodyids, target_type_lookup = self._get_target_bodyids_and_types(target_dataset)
        
        if not all_target_bodyids:
            self._log(f"ERROR: Could not get target bodyIds for {target_dataset}")
            return pd.DataFrame()
        
        # Count typed vs untyped
        typed_count = sum(1 for bid in all_target_bodyids 
                         if target_type_lookup.get(bid, '') and 
                         not pd.isna(target_type_lookup.get(bid, '')) and
                         target_type_lookup.get(bid, '') != '')
        untyped_count = len(all_target_bodyids) - typed_count
        
        self._log(f"Found {len(all_target_bodyids)} neurons in target dataset ({typed_count} typed, {untyped_count} untyped)")
        
        # Build target profiles using memory-efficient method
        target_profiles = self._build_profiles_memory_safe(
            all_target_bodyids,
            target_dataset,
            show_progress=show_progress,
            label="target"
        )
        
        self._log(f"Built {len(target_profiles)} target profiles")
        
        # =====================================================================
        # Step 4: Compare source profiles against target profiles (shared core)
        # =====================================================================
        # Build status maps
        from .connectivity_profiler import ConnectivityStatus
        source_status_map = {bid: profile.connectivity_status for bid, profile in source_profiles.items()}
        source_status_counts = {s.value: 0 for s in ConnectivityStatus}
        for status in source_status_map.values():
            source_status_counts[status.value] += 1

        # Candidate map: comprehensive search includes every target bodyId
        candidate_map: Dict[int, Dict[int, int]] = {}
        for source_bid in source_profiles.keys():
            candidate_map[source_bid] = {bid: 1 for bid in target_profiles.keys() if (is_cross_dataset or bid != source_bid)}

        # Build source_type_lookup
        source_type_lookup = {bid: (profile.neuron_type if profile.neuron_type else str(query)) for bid, profile in source_profiles.items()}

        # Run shared comparison core
        results_df, intra_type_df, skipped_sources, warned_sources, source_status_map, target_status_map, target_status_counts = self._compare_candidates_core(
            source_bodyids=list(source_profiles.keys()),
            source_profiles_cache=source_profiles,
            source_status_map=source_status_map,
            target_profiles_cache=target_profiles,
            target_type_lookup=target_type_lookup,
            source_type_lookup=source_type_lookup,
            candidate_map=candidate_map,
            is_cross_dataset=is_cross_dataset,
            target_dataset=target_dataset,
            show_progress=show_progress,
            similarity_metric=metric,
            top_n=top_n,
            min_score=min_score,
            include_intra_type=not is_cross_dataset,
            vector_prefiltering=self.vector_prefiltering
        )

        if results_df.empty:
            self._log("No candidates found")
            return pd.DataFrame()

        total_skipped = len(skipped_sources['none'])
        total_warned = len(warned_sources['rare_or_uni'])
        if total_skipped > 0:
            self._log(f"Skipped {total_skipped} source neurons with no partners (NONE/ORPHAN): {skipped_sources['none'][:5]}{'...' if total_skipped>5 else ''}")
        if total_warned > 0:
            self._log(f"WARNING: {total_warned} source neurons are sparse/unidirectional (RARE/UNIDIRECTIONAL)")
        
        self._log(f"Found {len(results_df)} bodyId-level matches")
        
        # Run shuffle test if requested
        shuffle_stats = None
        if run_shuffle_test and not results_df.empty:
            self._log(f"\nRunning shuffle test with {n_shuffles} iterations...")
            shuffle_stats = self.run_random_control_test(
                source=query,
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                n_shuffles=n_shuffles,
                top_n=top_n,
                seed=shuffle_seed,
                show_progress=show_progress
            )
            
            if shuffle_stats and 'p_value' in shuffle_stats:
                results_df['shuffle_p_value'] = shuffle_stats['p_value']
                results_df['shuffle_z_score'] = shuffle_stats['z_score']
                results_df['shuffle_effect_size'] = shuffle_stats['effect_size']
                results_df['shuffle_significant'] = shuffle_stats['is_significant']
                self._log(shuffle_stats.get('summary', ''))
        
        # Save results if output_dir is provided
        if output_dir is not None:
            save_result = self._save_homolog_results_internal(
                results_df=results_df,
                query=query,
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                output_dir=output_dir,
                saveas=saveas,
                direction=direction,
                include_partner_details=include_partner_details,
                top_n_details=top_n_details,
                params={
                    'query': query,
                    'source_dataset': source_dataset,
                    'target_dataset': target_dataset,
                    'top_n': top_n,
                    'metric': metric,
                    'direction': direction,
                    'min_score': min_score,
                    'method': 'find_homologs (comprehensive)',
                    'run_shuffle_test': run_shuffle_test,
                    'n_shuffles': n_shuffles if run_shuffle_test else None
                },
                shuffle_stats=shuffle_stats,
                visualize_skeleton=visualize_skeleton,
                visualize_top_n=visualize_top_n,
                similarity_metric=self.similarity_metric
            )
        else:
            # Always save - use default output_dir
            save_result = self._save_homolog_results_internal(
                results_df=results_df,
                query=query,
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                output_dir=self.output_dir,
                saveas=saveas,
                direction=direction,
                include_partner_details=include_partner_details,
                top_n_details=top_n_details,
                params={
                    'query': query,
                    'source_dataset': source_dataset,
                    'target_dataset': target_dataset,
                    'top_n': top_n,
                    'metric': metric,
                    'direction': direction,
                    'min_score': min_score,
                    'method': 'find_homologs (comprehensive)',
                    'run_shuffle_test': run_shuffle_test,
                    'n_shuffles': n_shuffles if run_shuffle_test else None
                },
                shuffle_stats=shuffle_stats,
                visualize_skeleton=visualize_skeleton,
                visualize_top_n=visualize_top_n,
                similarity_metric=self.similarity_metric
            )
        
        return results_df
    
    def find_novel_homologs(
        self,
        query: Union[str, int],
        dataset: str,
        top_n: int = 20,
        min_score: float = 0.3
    ) -> pd.DataFrame:
        """
        Find potential novel homologs within the same dataset.
        
        Shortcut for find_homologs() with target_dataset = source_dataset.
        
        Args:
            query: Neuron identifier
            dataset: Dataset to search
            top_n: Number of top candidates
            min_score: Minimum similarity score
        
        Returns:
            DataFrame with similar types/neurons (excluding self)
        """
        return self.find_homologs(
            query=query,
            source_dataset=dataset,
            target_dataset=dataset,
            top_n=top_n,
            min_score=min_score
        )
    
    def get_partner_overlap(
        self,
        query: Union[str, int],
        source_dataset: str,
        target_type: str,
        target_dataset: str,
        direction: str = 'upstream'
    ) -> pd.DataFrame:
        """
        Get detailed partner overlap between query and a candidate.
        
        Args:
            query: Source neuron identifier
            source_dataset: Source dataset
            target_type: Target type to compare
            target_dataset: Target dataset
            direction: 'upstream' or 'downstream'
        
        Returns:
            DataFrame with partner overlap details
        """
        source_profile = self.get_profile(query, source_dataset)
        target_profile = self.get_profile(target_type, target_dataset)
        
        if source_profile is None or target_profile is None:
            return pd.DataFrame()
        
        return ProfileComparator.get_partner_overlap_details(
            source_profile, target_profile, direction=direction
        )
    
    # =========================================================================
    # Fast Homolog Finding with Connection Cache & Adjacency Expansion
    # =========================================================================
    
    def _get_connection_cache_path(self, dataset: str) -> 'Path':
        """Get path to existing connection cache."""
        from pathlib import Path
        safe_name = dataset.replace(':', '_').replace('.', '_')
        src_dir = Path(__file__).parent.parent
        project_root = src_dir.parent
        return project_root / 'cache' / safe_name / 'connections.parquet'
    
    def _get_neuron_index_path(self, dataset: str) -> 'Path':
        """Get path to neuron index for type mapping."""
        from pathlib import Path
        safe_name = dataset.replace(':', '_').replace('.', '_')
        src_dir = Path(__file__).parent.parent
        project_root = src_dir.parent
        return project_root / 'cache' / safe_name / 'neuron_index.parquet'
    
    def _load_connection_cache(self, dataset: str, auto_build: bool = True) -> Optional[pd.DataFrame]:
        """
        Load complete connections from cache with type information.
        
        If the cache doesn't exist and auto_build=True, attempts to build it
        automatically by fetching from the API or local dataset files.
        
        Also verifies cache completeness by checking against neuron_df if available.
        
        Uses internal caching to avoid reloading the same dataset multiple times.
        Prioritizes FNC's module-level cache to avoid memory duplication.
        
        Type mapping priority:
        1. datasets/{dataset}/neuron_df.parquet (authoritative source)
        2. cache/{dataset}/neuron_index.parquet (fallback)
        
        Args:
            dataset: Dataset identifier
            auto_build: If True, automatically build cache if missing
        
        Returns:
            DataFrame with columns: bodyId_pre, bodyId_post, weight, type_pre, type_post
            or None if cache not available and cannot be built
        """
        # Check internal cache first
        if dataset in self._conn_cache:
            return self._conn_cache[dataset]
        
        from pathlib import Path
        safe_name = dataset.replace(':', '_').replace('.', '_')
        src_dir = Path(__file__).parent.parent
        project_root = src_dir.parent
        
        # Check FNC's module-level cache first (avoids loading from disk twice)
        try:
            from coana import _FNC_CACHE
            if safe_name in _FNC_CACHE and 'conn_df' in _FNC_CACHE[safe_name]:
                fnc_df = _FNC_CACHE[safe_name]['conn_df']
                if fnc_df is not None and not fnc_df.empty:
                    self._log(f"Using FNC cache for {dataset} ({len(fnc_df):,} connections)")
                    # FNC cache may not have type columns - add them IN-PLACE to avoid copying
                    if 'type_pre' not in fnc_df.columns or 'type_post' not in fnc_df.columns:
                        index_path = self._get_neuron_index_path(dataset)
                        type_map = self._load_type_mapping(dataset, project_root, safe_name, index_path)
                        if type_map:
                            # Convert bodyIds to numeric in-place
                            fnc_df['bodyId_pre'] = pd.to_numeric(fnc_df['bodyId_pre'], errors='coerce')
                            fnc_df['bodyId_post'] = pd.to_numeric(fnc_df['bodyId_post'], errors='coerce')
                            type_map_int = {int(k) if str(k).isdigit() else k: v for k, v in type_map.items()}
                            fnc_df['type_pre'] = fnc_df['bodyId_pre'].map(type_map_int)
                            fnc_df['type_post'] = fnc_df['bodyId_post'].map(type_map_int)
                    # Don't store in self._conn_cache - just reference FNC's cache
                    return fnc_df
        except ImportError:
            pass
        
        cache_path = self._get_connection_cache_path(dataset)
        index_path = self._get_neuron_index_path(dataset)
        
        # Check if cache exists, if not try to build it
        if not cache_path.exists():
            self._log(f"Connection cache not found: {cache_path}")
            
            if auto_build:
                self._log(f"Attempting to build connection cache for {dataset}...")
                if self._build_connection_cache_for_dataset(dataset, cache_path, project_root, safe_name):
                    self._log(f"Successfully built connection cache for {dataset}")
                else:
                    self._log(f"ERROR: Could not build connection cache for {dataset}")
                    return None
            else:
                return None
        
        # Verify cache exists now
        if not cache_path.exists():
            return None
        
        # Load connections using Polars for memory efficiency, then convert to pandas
        self._log(f"Loading connection cache for {dataset}...")
        try:
            import polars as pl
            # Use Polars for memory-efficient loading of large caches
            conn_df_pl = pl.read_parquet(cache_path)
            conn_df = conn_df_pl.to_pandas()
            del conn_df_pl
            self._log(f"Loaded {len(conn_df):,} connections from cache (via Polars)")
        except ImportError:
            # Fallback to pandas if polars not available
            conn_df = pd.read_parquet(cache_path)
            self._log(f"Loaded {len(conn_df):,} connections from cache")
        
        # Note: Cache is incremental - build_connection_cache always adds missing neurons
        # No need for threshold-based completeness checking since cache grows over time
        
        # Load type mapping
        self._log(f"Loading type mapping for {dataset}...")
        type_map = self._load_type_mapping(dataset, project_root, safe_name, index_path)
        
        if type_map is None:
            self._log(f"No neuron type mapping available for {dataset}")
            return None
        
        # Ensure bodyId columns are numeric for proper type mapping
        try:
            conn_df['bodyId_pre'] = pd.to_numeric(conn_df['bodyId_pre'], errors='coerce')
            conn_df['bodyId_post'] = pd.to_numeric(conn_df['bodyId_post'], errors='coerce')
        except Exception:
            pass  # Keep original types if conversion fails
        
        # Convert type_map keys to match bodyId dtype if needed
        if conn_df['bodyId_pre'].dtype in ['int64', 'float64']:
            self._log(f"Converting type map keys ({len(type_map):,} entries)...")
            type_map = {int(k) if isinstance(k, (int, float, str)) and str(k).isdigit() else k: v 
                        for k, v in type_map.items()}
        
        # Add type columns
        self._log(f"Mapping types to connections ({len(conn_df):,} rows)...")
        conn_df['type_pre'] = conn_df['bodyId_pre'].map(type_map)
        conn_df['type_post'] = conn_df['bodyId_post'].map(type_map)
        
        # Log mapping success
        mapped_pre = conn_df['type_pre'].notna().sum()
        mapped_post = conn_df['type_post'].notna().sum()
        self._log(f"Loaded {len(conn_df):,} connections from cache")
        self._log(f"Type mapping: {mapped_pre:,}/{len(conn_df):,} pre, {mapped_post:,}/{len(conn_df):,} post")
        
        # NOTE: Do NOT store in self._conn_cache to avoid memory issues
        # For large datasets (20M+ rows), keeping in memory causes OOM
        # The caller should use the returned DataFrame and discard when done
        # self._conn_cache[dataset] = conn_df  # Disabled to save memory
        
        return conn_df
    
    def _load_type_mapping(self, dataset: str, project_root: Path, safe_name: str, index_path: Path) -> Optional[Dict]:
        """Load neuron type mapping from available sources."""
        type_map = None
        
        # Priority 1: datasets/{dataset}/neuron_df.parquet
        datasets_neuron_parquet = project_root / 'datasets' / safe_name / f'{safe_name}_allneurons_neuron_df.parquet'
        if datasets_neuron_parquet.exists():
            neuron_df = pd.read_parquet(datasets_neuron_parquet)
            if 'type' in neuron_df.columns:
                type_map = neuron_df.set_index('bodyId')['type'].to_dict()
                self._log(f"Loaded type mapping from {datasets_neuron_parquet.name}")
        
        # Priority 2: datasets/{dataset}/neuron_df.csv (fallback for CSV files)
        if type_map is None:
            datasets_neuron_csv = project_root / 'datasets' / safe_name / f'{safe_name}_allneurons_neuron_df.csv'
            if datasets_neuron_csv.exists():
                neuron_df = pd.read_csv(datasets_neuron_csv, low_memory=False)
                if 'type' in neuron_df.columns:
                    type_map = neuron_df.set_index('bodyId')['type'].to_dict()
                    self._log(f"Loaded type mapping from {datasets_neuron_csv.name}")
        
        # Priority 3: cache/{dataset}/neuron_index.parquet
        if type_map is None and index_path.exists():
            neuron_df = pd.read_parquet(index_path)
            if 'type' in neuron_df.columns:
                type_map = neuron_df.set_index('bodyId')['type'].to_dict()
                self._log(f"Loaded type mapping from neuron_index.parquet ({len(type_map)} neurons)")
        
        return type_map
    
    def _verify_cache_completeness(self, conn_df: pd.DataFrame, dataset: str, 
                                    project_root: Path, safe_name: str) -> Optional[Dict]:
        """
        Verify that the connection cache is complete by comparing against neuron_df.
        
        Returns:
            Dict with 'expected_neurons', 'cache_neurons', 'missing_ratio'
            or None if cannot verify
        """
        # Try to load neuron_df to get expected neuron count
        neuron_df = None
        
        neuron_files = [
            project_root / 'datasets' / safe_name / f'{safe_name}_allneurons_neuron_df.parquet',
            project_root / 'datasets' / safe_name / f'{safe_name}_allneurons_neuron_df.csv',
        ]
        
        for neuron_file in neuron_files:
            if neuron_file.exists():
                try:
                    if str(neuron_file).endswith('.parquet'):
                        neuron_df = pd.read_parquet(neuron_file)
                    else:
                        neuron_df = pd.read_csv(neuron_file, low_memory=False)
                    break
                except Exception:
                    continue
        
        if neuron_df is None:
            return None  # Cannot verify without neuron_df
        
        # Count unique neurons in cache
        cache_neurons = set()
        if 'bodyId_pre' in conn_df.columns:
            cache_neurons.update(conn_df['bodyId_pre'].dropna().unique())
        if 'bodyId_post' in conn_df.columns:
            cache_neurons.update(conn_df['bodyId_post'].dropna().unique())
        
        # Expected neurons from neuron_df
        expected_neurons = len(neuron_df)
        cache_neuron_count = len(cache_neurons)
        
        # Calculate missing ratio (some neurons may have no connections, so allow some tolerance)
        # But if significantly fewer neurons are in cache, it's likely incomplete
        missing_ratio = max(0, (expected_neurons - cache_neuron_count) / expected_neurons) if expected_neurons > 0 else 0
        
        return {
            'expected_neurons': expected_neurons,
            'cache_neurons': cache_neuron_count,
            'missing_ratio': missing_ratio
        }
    
    def _build_connection_cache_for_dataset(self, dataset: str, cache_path: Path, 
                                             project_root: Path, safe_name: str) -> bool:
        """
        Build connection cache using FindNeuronConnection.
        
        This method uses FindNeuronConnection.build_connection_cache() which:
        - Handles both NeuPrint datasets and local datasets
        - Builds cache incrementally (only fetches missing neurons)
        - Builds both connections.parquet AND neuron_index.parquet
        - Uses O(1) indexed lookups for fast cache queries
        
        Cache Hierarchy:
        ---------------
        Level 0: datasets/{dataset}/*_neuron_df - Authoritative neuron list
        Level 1: cache/{dataset}/neuron_index.parquet - Tracks cached neurons
        Level 2: cache/{dataset}/connections.parquet - Connection data
        
        Returns:
            True if cache was built successfully, False otherwise
        """
        try:
            # Import FindNeuronConnection
            try:
                from .coana import FindNeuronConnection
            except ImportError:
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from coana import FindNeuronConnection
            
            self._log(f"Building connection cache for {dataset}...")
            
            # Get token from environment (needed for NeuPrint datasets)
            import os
            token = os.environ.get('NEUPRINT_APPLICATION_CREDENTIALS') or os.environ.get('NEUPRINT_TOKEN')
            
            # Initialize FindNeuronConnection with cache enabled
            fnc = FindNeuronConnection(
                dataset=dataset,
                token=token,  # Will be None for local datasets, which is fine
                use_cache=True,
                verbose_mode='normal'
            )
            
            # Build cache incrementally - only fetches missing neurons
            result = fnc.build_connection_cache(batch_size=1000, quiet=False)
            
            total_connections = result.get('total_connections', 0)
            newly_cached = result.get('newly_cached', 0)
            already_cached = result.get('already_cached', 0)
            
            if total_connections > 0 or already_cached > 0:
                self._log(f"Cache ready: {total_connections:,} connections")
                if newly_cached > 0:
                    self._log(f"  Newly cached: {newly_cached:,} neurons")
                if already_cached > 0:
                    self._log(f"  Already cached: {already_cached:,} neurons")
                return True
            else:
                self._log("WARNING: No connections found - dataset may be empty or inaccessible")
                return False
                
        except Exception as e:
            self._log(f"ERROR: Failed to build connection cache: {e}")
            import traceback
            self._log(f"Details: {traceback.format_exc()}", level='debug')
            return False
    
    def _ensure_connection_cache_complete(self, dataset: str) -> bool:
        """
        Ensure connection cache is complete for the entire dataset.
        
        This method calls FNC's build_connection_cache which:
        - Checks which neurons are already cached (O(1) lookup)
        - Only fetches missing neurons (incremental)
        - Uses progress bar for long operations
        
        Cache Hierarchy (must be built in order):
        -----------------------------------------
        Level 0: datasets/{dataset}/*_neuron_df - Authoritative neuron list  
        Level 1: cache/{dataset}/neuron_index.parquet - Tracks cached neurons
        Level 2: cache/{dataset}/connections.parquet - Connection data
        Level 3: Connectivity profiles (built after this check)
        
        Args:
            dataset: Dataset identifier
            
        Returns:
            True if cache is ready, False otherwise
        """
        try:
            # Import FindNeuronConnection - try relative first (src.coana), then absolute
            try:
                from ..coana import FindNeuronConnection
            except ImportError:
                try:
                    from coana import FindNeuronConnection
                except ImportError:
                    import sys
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from coana import FindNeuronConnection
            
            # Initialize FindNeuronConnection
            # Use 'simple' mode to show loading progress, not 'silent' which hides everything
            # Use simple_fetch=True to reduce memory usage during bulk fetch
            # Token is passed from HomologFinder; FNC auto-handles env vars if empty
            fnc = FindNeuronConnection(
                dataset=dataset,
                token=self.token,
                use_cache=True,
                verbose_mode='simple',  # Show loading progress (not 'silent')
                simple_fetch=True
            )
            
            # Pass the progress bar flag so FNC uses tqdm.write if needed
            fnc._in_progress_bar = getattr(self, '_in_progress_bar', False)
            
            # Build connection cache - this is incremental (skips already cached neurons)
            # If cache is already complete, this returns quickly
            # Use quiet=False to show progress for large fetches
            # Use smaller batch size (50) to reduce memory pressure during fetch
            result = fnc.build_connection_cache(batch_size=1000, quiet=False)
            
            total_neurons = result.get('total_neurons', 0)
            already_cached = result.get('already_cached', 0)
            newly_cached = result.get('newly_cached', 0)
            total_connections = result.get('total_connections', 0)
            
            # Clear FNC's in-memory cache to free memory
            # Data is already saved to disk, we'll load only what we need later
            fnc._conn_df_cache = None
            fnc._conn_index = {}
            fnc._conn_index_post = {}
            
            # Also clear the module-level cache for this dataset
            # This prevents accumulating large DataFrames across datasets
            safe_name = dataset.replace(':', '_').replace('.', '_')
            
            # Try clearing from all possible module locations to handle import variations
            modules_to_check = ['coana', 'src.coana', 'src.comparison.coana']
            for mod_name in modules_to_check:
                try:
                    import sys
                    if mod_name in sys.modules:
                        mod = sys.modules[mod_name]
                        if hasattr(mod, '_FNC_CACHE'):
                            if safe_name in mod._FNC_CACHE:
                                mod._FNC_CACHE[safe_name] = {}
                except Exception:
                    pass
            
            del fnc
            import gc
            gc.collect()
            
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                mem = process.memory_info().rss / 1024 / 1024
                self._log(f"Memory after cache check for {dataset}: {mem:.1f} MB")
            except:
                pass
            
            if total_neurons > 0:
                if newly_cached == 0:
                    self._log(f"Connection cache complete: {already_cached:,} neurons, {total_connections:,} connections")
                else:
                    self._log(f"Connection cache updated: +{newly_cached:,} neurons (total: {total_connections:,} connections)")
                return True
            else:
                self._log("WARNING: No neurons found in dataset")
                return False
            
        except Exception as e:
            self._log(f"WARNING: Could not ensure connection cache: {e}")
            # Continue anyway - may work with partial data
            return False
    
    def _get_adjacency_candidates(
        self,
        conn_df: pd.DataFrame,
        query: Union[str, int],
        min_weight: int = 3
    ) -> set:
        """
        Get candidate field via adjacency expansion.
        
        Algorithm:
        1. Find upstream and downstream neurons of the source neurons
        2. Fetch the downstream of the upstream neurons (2-hop)
        3. Fetch the upstream of the downstream neurons (2-hop)
        4. Return the union as candidate field
        
        Args:
            conn_df: Connection DataFrame with type columns
            query: Type name (str) or bodyId (int)
            min_weight: Minimum synapse weight
            
        Returns:
            Set of candidate type names
        """
        # Filter by weight
        filtered = conn_df[conn_df['weight'] >= min_weight].copy()
        
        # Find 1-hop partners
        if isinstance(query, str):
            # Type-based query
            upstream_mask = filtered['type_post'] == query
            downstream_mask = filtered['type_pre'] == query
        else:
            # BodyId query
            upstream_mask = filtered['bodyId_post'] == query
            downstream_mask = filtered['bodyId_pre'] == query
        
        # Get 1-hop upstream types (neurons that synapse onto query)
        upstream_1hop = set(filtered.loc[upstream_mask, 'type_pre'].dropna().unique())
        # Get 1-hop downstream types (neurons that query synapses onto)
        downstream_1hop = set(filtered.loc[downstream_mask, 'type_post'].dropna().unique())
        
        # Get 2-hop: downstream of upstream (types that also receive from same upstream)
        downstream_of_upstream = set()
        if upstream_1hop:
            mask = filtered['type_pre'].isin(upstream_1hop)
            downstream_of_upstream = set(filtered.loc[mask, 'type_post'].dropna().unique())
        
        # Get 2-hop: upstream of downstream (types that also send to same downstream)
        upstream_of_downstream = set()
        if downstream_1hop:
            mask = filtered['type_post'].isin(downstream_1hop)
            upstream_of_downstream = set(filtered.loc[mask, 'type_pre'].dropna().unique())
        
        # Candidate field = 2-hop partners (excluding empty strings)
        candidates = downstream_of_upstream | upstream_of_downstream
        candidates = {c for c in candidates if c and str(c).strip()}
        
        # Remove query itself from candidates
        query_str = str(query)
        candidates.discard(query_str)
        
        return candidates
    
    def _build_type_aggregates(
        self,
        conn_df: pd.DataFrame,
        min_weight: int = 1,
        show_progress: bool = True
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """
        Pre-aggregate connection data into type-level dictionaries.
        
        This is the key optimization for find_homologs_fast - we aggregate
        once and then look up types in O(1) instead of O(N) per type.
        
        Args:
            conn_df: Connection DataFrame with type columns
            min_weight: Minimum synapse weight
            show_progress: Show progress bar
            
        Returns:
            Tuple of (upstream_by_type, downstream_by_type)
            Each is Dict[type -> Dict[partner_type -> total_weight]]
        """
        if show_progress and self.verbose:
            self._log("Pre-aggregating connection data by type...")
        
        # Filter by weight
        filtered = conn_df[conn_df['weight'] >= min_weight]
        
        # Filter to only typed connections
        typed = filtered[
            filtered['type_pre'].notna() & 
            filtered['type_post'].notna() &
            (filtered['type_pre'].astype(str).str.strip() != '') &
            (filtered['type_post'].astype(str).str.strip() != '')
        ]

        upstream_by_type: Dict[str, Dict[str, float]] = {}
        downstream_by_type: Dict[str, Dict[str, float]] = {}

        # Use Polars aggregation when available (notably faster for large caches)
        use_polars = False
        try:
            import polars as pl
            typed_pl = pl.from_pandas(typed[['type_pre', 'type_post', 'weight']])
            use_polars = True
        except Exception:
            typed_pl = None
            use_polars = False

        if use_polars and typed_pl is not None:
            upstream_agg = typed_pl.group_by(['type_post', 'type_pre']).agg(pl.col('weight').sum())
            for row in upstream_agg.iter_rows(named=True):
                post_type = row['type_post']
                pre_type = row['type_pre']
                weight = row['weight']
                upstream_by_type.setdefault(post_type, {})[pre_type] = weight

            downstream_agg = typed_pl.group_by(['type_pre', 'type_post']).agg(pl.col('weight').sum())
            for row in downstream_agg.iter_rows(named=True):
                pre_type = row['type_pre']
                post_type = row['type_post']
                weight = row['weight']
                downstream_by_type.setdefault(pre_type, {})[post_type] = weight
        else:
            # Pandas fallback
            upstream_agg = typed.groupby(['type_post', 'type_pre'])['weight'].sum()
            for (post_type, pre_type), weight in upstream_agg.items():
                upstream_by_type.setdefault(post_type, {})[pre_type] = weight

            downstream_agg = typed.groupby(['type_pre', 'type_post'])['weight'].sum()
            for (pre_type, post_type), weight in downstream_agg.items():
                downstream_by_type.setdefault(pre_type, {})[post_type] = weight
        
        if show_progress and self.verbose:
            self._log(f"Aggregated {len(upstream_by_type):,} types with upstream, "
                      f"{len(downstream_by_type):,} with downstream connections")
        
        return upstream_by_type, downstream_by_type
    
    def _build_bodyid_aggregates(
        self,
        conn_df: pd.DataFrame,
        min_weight: int = 1,
        show_progress: bool = True
    ) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
        """
        Pre-aggregate connection data into bodyId-level dictionaries.
        
        Similar to _build_type_aggregates but uses bodyId as primary key
        and partner types as values. Used for bodyId-level comparisons.
        
        Args:
            conn_df: Connection DataFrame with type columns
            min_weight: Minimum synapse weight
            show_progress: Show progress bar
            
        Returns:
            Tuple of (upstream_by_bodyid, downstream_by_bodyid)
            Each is Dict[bodyId -> Dict[partner_type -> total_weight]]
        """
        if show_progress and self.verbose:
            self._log("Pre-aggregating connection data by bodyId...")
        
        # Filter by weight
        filtered = conn_df[conn_df['weight'] >= min_weight]

        upstream_by_bodyid: Dict[int, Dict[str, float]] = {}
        downstream_by_bodyid: Dict[int, Dict[str, float]] = {}

        use_polars = False
        try:
            import polars as pl
            filtered_pl = pl.from_pandas(filtered[['bodyId_pre', 'bodyId_post', 'type_pre', 'type_post', 'weight']])
            use_polars = True
        except Exception:
            filtered_pl = None
            use_polars = False

        if use_polars and filtered_pl is not None:
            upstream_agg = filtered_pl.group_by(['bodyId_post', 'type_pre']).agg(pl.col('weight').sum())
            for row in upstream_agg.iter_rows(named=True):
                post_bodyid = row['bodyId_post']
                pre_type = row['type_pre']
                weight = row['weight']
                if pre_type is None or str(pre_type).strip() == '':
                    continue
                upstream_by_bodyid.setdefault(post_bodyid, {})[pre_type] = weight

            downstream_agg = filtered_pl.group_by(['bodyId_pre', 'type_post']).agg(pl.col('weight').sum())
            for row in downstream_agg.iter_rows(named=True):
                pre_bodyid = row['bodyId_pre']
                post_type = row['type_post']
                weight = row['weight']
                if post_type is None or str(post_type).strip() == '':
                    continue
                downstream_by_bodyid.setdefault(pre_bodyid, {})[post_type] = weight
        else:
            # Pandas fallback
            upstream_agg = filtered.groupby(['bodyId_post', 'type_pre'])['weight'].sum()
            for (post_bodyid, pre_type), weight in upstream_agg.items():
                if pd.isna(pre_type) or str(pre_type).strip() == '':
                    continue
                upstream_by_bodyid.setdefault(post_bodyid, {})[pre_type] = weight

            downstream_agg = filtered.groupby(['bodyId_pre', 'type_post'])['weight'].sum()
            for (pre_bodyid, post_type), weight in downstream_agg.items():
                if pd.isna(post_type) or str(post_type).strip() == '':
                    continue
                downstream_by_bodyid.setdefault(pre_bodyid, {})[post_type] = weight
        
        if show_progress and self.verbose:
            self._log(f"Aggregated {len(upstream_by_bodyid):,} bodyIds with upstream, "
                      f"{len(downstream_by_bodyid):,} with downstream connections")
        
        return upstream_by_bodyid, downstream_by_bodyid

    def _compare_candidates_core(
        self,
        source_bodyids: List[int],
        source_profiles_cache: Dict[int, 'ConnectivityProfile'],
        source_status_map: Dict[int, 'ConnectivityStatus'],
        target_profiles_cache: Dict[int, 'ConnectivityProfile'],
        target_type_lookup: Dict[int, str],
        source_type_lookup: Dict[int, str],
        candidate_map: Dict[int, Dict[int, int]],
        is_cross_dataset: bool,
        target_dataset: str,
        show_progress: bool,
        similarity_metric: str,
        top_n: int,
        min_score: Optional[float] = None,
        include_intra_type: bool = True,
        vector_prefiltering: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[int]], Dict[str, List[int]], Dict[int, 'ConnectivityStatus'], Dict[int, 'ConnectivityStatus'], Dict[str, int]]: # 
        """Shared bodyId comparison core used by find_homologs and find_homologs_fast."""
        from .connectivity_profiler import ConnectivityStatus

        target_status_counts: Dict[str, int] = {s.value: 0 for s in ConnectivityStatus}
        target_status_map: Dict[int, ConnectivityStatus] = {}

        for bid, profile in target_profiles_cache.items():
            status = profile.connectivity_status
            target_status_counts[status.value] += 1
            target_status_map[bid] = status

        all_results = []
        skipped_sources: Dict[str, List[int]] = {'none': []}
        warned_sources: Dict[str, List[int]] = {'rare_or_uni': []}

        if vector_prefiltering and self.verbose:
            self._log("Vector prefiltering enabled: keep top 5% by adjacency, then cosine>0 before full scoring")

        # Set progress bar flag so _log uses tqdm.write
        was_in_progress = self._in_progress_bar
        self._in_progress_bar = True

        # Comparison loop
        for source_bid in tqdm(
            source_bodyids,
            desc="Comparing source neurons",
            disable=not show_progress or not self.verbose,
            leave=True
        ):
            source_profile = source_profiles_cache.get(source_bid)
            if source_profile is None:
                continue

            source_status = source_status_map.get(source_bid, ConnectivityStatus.NONE)

            if not source_status.is_valid_for_comparison():
                skipped_sources['none'].append(source_bid)
                continue

            if source_status in {ConnectivityStatus.RARE, ConnectivityStatus.UNIDIRECTIONAL}:
                warned_sources['rare_or_uni'].append(source_bid)

            source_type_str = source_type_lookup.get(source_bid, str(source_bid))

            source_partner_count = len(ProfileComparator._get_expanded_types(source_profile, 'both')) if is_cross_dataset else len(ProfileComparator._get_all_bodyids(source_profile, 'both'))

            source_candidates = candidate_map.get(source_bid, {})

            if vector_prefiltering and source_candidates:
                total_candidates = len(source_candidates)

                # First filter by cosine > 0 (quick rejection of dissimilar vectors)
                cos_filtered: Dict[int, Tuple[int, float]] = {}
                for target_bid, shared_count in source_candidates.items():
                    target_profile = target_profiles_cache.get(target_bid)
                    if target_profile is None:
                        continue
                    cos_val = ProfileComparator.weighted_cosine_similarity(source_profile, target_profile, 'both')
                    if cos_val > 0:
                        cos_filtered[target_bid] = (shared_count, cos_val)

                cos_kept = len(cos_filtered)

                # Then keep top 5% of TOTAL candidates by adjacency score among cosine-positive set
                if cos_kept > 0:
                    top_k = max(1, math.ceil(total_candidates * 0.05))
                    top_candidates = dict(heapq.nlargest(top_k, cos_filtered.items(), key=lambda x: x[1][0]))
                    source_candidates = {bid: shared for bid, (shared, _) in top_candidates.items()}
                else:
                    source_candidates = {}

                if self.verbose and show_progress:
                    self._log(
                        f"Prefiltered {total_candidates}→{cos_kept} (cos>0) →{len(source_candidates)} (top 5% of total) for source {source_bid}"
                    )

            if not source_candidates:
                continue

            if is_cross_dataset:
                batch_scores = ProfileComparator.batch_compare_cross_dataset(
                    source_profile, target_profiles_cache, source_candidates, 'both'
                )

                for score_dict in batch_scores:
                    target_bid = score_dict['target_bid']
                    target_status = target_status_map.get(target_bid, ConnectivityStatus.NONE)
                    target_type_str = target_type_lookup.get(target_bid, '')

                    rank_corr_raw = score_dict['rank']
                    rank_corr_norm = (rank_corr_raw + 1) / 2 if not np.isnan(rank_corr_raw) else np.nan
                    rank_union_raw = score_dict['rank_union']
                    rank_union_norm = (rank_union_raw + 1) / 2 if not np.isnan(rank_union_raw) else np.nan

                    # Optional score filter
                    if min_score is not None and (pd.isna(rank_corr_norm) or rank_corr_norm < min_score):
                        continue

                    all_results.append({
                        'source_bodyId': source_bid,
                        'source_type': source_type_str,
                        'target_bodyId': target_bid,
                        'target_type': target_type_str,
                        'target_dataset': target_dataset,
                        'adjacency_score': int(score_dict.get('shared_count', 0)),
                        'shared_type_count': score_dict['shared_type_count'],
                        'union_type_count': score_dict['union_type_count'],
                        'rank_corr': rank_corr_norm,
                        'rank_corr_raw': rank_corr_raw,
                        'rank_union': rank_union_norm,
                        'rank_union_raw': rank_union_raw,
                        'jaccard': score_dict['jaccard'],
                        'cosine': score_dict['cosine'],
                        'combined': score_dict['combined'],
                        'is_same_type': source_type_str == target_type_str if target_type_str else False,
                        'is_same_dataset': not is_cross_dataset,
                        'source_status': source_status.value,
                        'target_status': target_status.value,
                        'weak_source': not source_status.is_valid_for_comparison(),
                        'weak_target': not target_status.is_valid_for_comparison(),
                        'source_partner_count': source_partner_count,
                        'target_partner_count': score_dict['target_type_count']
                    })
            else:
                for target_bid, shared_count in source_candidates.items():
                    target_profile = target_profiles_cache.get(target_bid)
                    if target_profile is None:
                        continue

                    target_status = target_status_map.get(target_bid, ConnectivityStatus.NONE)
                    scores = ProfileComparator.combined_score_intra_dataset(source_profile, target_profile)

                    rank_corr_raw = scores['rank']
                    rank_corr_norm = (rank_corr_raw + 1) / 2 if not np.isnan(rank_corr_raw) else np.nan
                    rank_union_raw = scores.get('rank_union', np.nan)
                    rank_union_norm = (rank_union_raw + 1) / 2 if not np.isnan(rank_union_raw) else np.nan

                    if min_score is not None and (pd.isna(rank_corr_norm) or rank_corr_norm < min_score):
                        continue

                    target_type_str = target_type_lookup.get(target_bid, '')
                    target_partner_count = len(ProfileComparator._get_all_bodyids(target_profile, 'both'))

                    all_results.append({
                        'source_bodyId': source_bid,
                        'source_type': source_type_str,
                        'target_bodyId': target_bid,
                        'target_type': target_type_str,
                        'target_dataset': target_dataset,
                        'adjacency_score': int(shared_count),
                        'shared_type_count': scores.get('shared_type_count', 0),
                        'union_type_count': scores.get('union_type_count', 0),
                        'rank_corr': rank_corr_norm,
                        'rank_corr_raw': rank_corr_raw,
                        'rank_union': rank_union_norm,
                        'rank_union_raw': rank_union_raw,
                        'jaccard': scores['jaccard'],
                        'cosine': scores['cosine'],
                        'combined': scores['combined'],
                        'is_same_type': source_type_str == target_type_str if target_type_str else False,
                        'is_same_dataset': not is_cross_dataset,
                        'source_status': source_status.value,
                        'target_status': target_status.value,
                        'weak_source': not source_status.is_valid_for_comparison(),
                        'weak_target': not target_status.is_valid_for_comparison(),
                        'source_partner_count': source_partner_count,
                        'target_partner_count': target_partner_count
                    })

        # Build intra-type comparison if requested
        intra_type_df = pd.DataFrame()
        if include_intra_type and len(source_bodyids) > 1:
            intra_type_results = []
            source_bid_list = sorted(source_bodyids)
            for i, bid_a in enumerate(source_bid_list):
                profile_a = source_profiles_cache.get(bid_a)
                if profile_a is None:
                    continue
                for bid_b in source_bid_list[i+1:]:
                    profile_b = source_profiles_cache.get(bid_b)
                    if profile_b is None:
                        continue

                    scores = ProfileComparator.combined_score_intra_dataset(profile_a, profile_b)
                    rank_corr_raw = scores['rank']
                    rank_corr_norm = (rank_corr_raw + 1) / 2 if not np.isnan(rank_corr_raw) else np.nan
                    rank_union_raw = scores.get('rank_union', np.nan)
                    rank_union_norm = (rank_union_raw + 1) / 2 if not np.isnan(rank_union_raw) else np.nan

                    source_partner_count = len(ProfileComparator._get_all_bodyids(profile_a, 'both'))
                    target_partner_count = len(ProfileComparator._get_all_bodyids(profile_b, 'both'))

                    status_a = source_status_map.get(bid_a, ConnectivityStatus.NONE)
                    status_b = source_status_map.get(bid_b, ConnectivityStatus.NONE)

                    intra_type_results.append({
                        'source_bodyId': bid_a,
                        'source_type': source_type_lookup.get(bid_a, str(bid_a)),
                        'target_bodyId': bid_b,
                        'target_type': source_type_lookup.get(bid_b, str(bid_b)),
                        'target_dataset': target_dataset,
                        'adjacency_score': 0,
                        'shared_type_count': scores.get('shared_type_count', 0),
                        'union_type_count': scores.get('union_type_count', 0),
                        'rank_corr': rank_corr_norm,
                        'rank_corr_raw': rank_corr_raw,
                        'rank_union': rank_union_norm,
                        'rank_union_raw': rank_union_raw,
                        'jaccard': scores['jaccard'],
                        'cosine': scores['cosine'],
                        'combined': scores['combined'],
                        'is_same_type': True,
                        'is_same_dataset': not is_cross_dataset,
                        'source_status': status_a.value,
                        'target_status': status_b.value,
                        'weak_source': not status_a.is_valid_for_comparison(),
                        'weak_target': not status_b.is_valid_for_comparison(),
                        'source_partner_count': source_partner_count,
                        'target_partner_count': target_partner_count
                    })

            if intra_type_results:
                intra_type_df = pd.DataFrame(intra_type_results)

        results_df = pd.DataFrame(all_results)

        # Sorting and top-N trimming
        if not results_df.empty:
            sort_col = similarity_metric if similarity_metric in results_df.columns else 'combined'
            results_df = results_df.sort_values(['source_bodyId', sort_col], ascending=[True, False], na_position='last')
            if top_n > 0:
                results_df = results_df.groupby('source_bodyId').head(top_n).reset_index(drop=True)

        self._in_progress_bar = was_in_progress
        return (
            results_df,
            intra_type_df,
            skipped_sources,
            warned_sources,
            source_status_map,
            target_status_map,
            target_status_counts,
        )
    
    def _build_bodyid_type_lookup(
        self,
        conn_df: pd.DataFrame
    ) -> Dict[int, str]:
        """
        Build a bodyId to type lookup dictionary from connection DataFrame.
        
        Args:
            conn_df: Connection DataFrame with bodyId_pre, bodyId_post, type_pre, type_post columns
            
        Returns:
            Dict mapping bodyId (int) to type (str)
        """
        type_lookup: Dict[int, str] = {}
        
        # Get types from pre neurons
        for _, row in conn_df[['bodyId_pre', 'type_pre']].drop_duplicates().iterrows():
            bid = row['bodyId_pre']
            typ = row['type_pre']
            if pd.notna(bid) and pd.notna(typ) and str(typ).strip():
                try:
                    type_lookup[int(bid)] = str(typ)
                except (ValueError, TypeError):
                    pass
        
        # Get types from post neurons (may add new entries or override)
        for _, row in conn_df[['bodyId_post', 'type_post']].drop_duplicates().iterrows():
            bid = row['bodyId_post']
            typ = row['type_post']
            if pd.notna(bid) and pd.notna(typ) and str(typ).strip():
                try:
                    type_lookup[int(bid)] = str(typ)
                except (ValueError, TypeError):
                    pass
        
        return type_lookup
    
    def _build_profile_from_bodyid_aggregates(
        self,
        bodyid: int,
        dataset: str,
        upstream_by_bodyid: Dict[int, Dict[str, float]],
        downstream_by_bodyid: Dict[int, Dict[str, float]],
        top_k: Optional[int] = None,
        top_m: Optional[int] = None
    ) -> Optional['ConnectivityProfile']:
        """
        DEPRECATED: Use self.profiler.get_profile(bodyid, dataset) instead.
        
        This method builds profiles from pre-aggregated data WITHOUT the 1-hop/2-hop
        hybrid approach. All profile building should now go through ConnectivityProfiler
        which provides proper 2-hop expansion for untyped 1-hop partners.
        
        Args:
            bodyid: BodyId of the neuron
            dataset: Dataset name
            upstream_by_bodyid: Pre-aggregated upstream connections by bodyId
            downstream_by_bodyid: Pre-aggregated downstream connections by bodyId
            top_k: Top K partners to keep per direction
            top_m: Minimum unique types to ensure
            
        Returns:
            ConnectivityProfile or None if no connections found
            
        .. deprecated::
            Use `self.profiler.get_profile(bodyid, dataset)` for proper 1-hop/2-hop
            hybrid profile building with 2-hop expansion for untyped neurons.
        """
        import warnings
        warnings.warn(
            "_build_profile_from_bodyid_aggregates is deprecated. "
            "Use self.profiler.get_profile(bodyid, dataset) for proper 1-hop/2-hop hybrid profiles.",
            DeprecationWarning,
            stacklevel=2
        )
        from .connectivity_profiler import ConnectivityProfile
        
        # Get defaults from profiler config
        if top_k is None:
            top_k = self.profiler.config.top_k_bodyid
        if top_m is None:
            top_m = self.profiler.config.top_m_type
        
        max_expansion_factor = self.profiler.config.max_expansion_factor
        dynamic_expansion = self.profiler.config.dynamic_expansion
        
        upstream_all = upstream_by_bodyid.get(bodyid, {})
        downstream_all = downstream_by_bodyid.get(bodyid, {})
        
        if not upstream_all and not downstream_all:
            return None
        
        def apply_top_k_m(partners: Dict[str, float], top_k: int, top_m: int) -> Tuple[Dict[str, float], int]:
            """Apply top-k filtering with optional top-m expansion."""
            if not partners:
                return {}, top_k
            
            # Sort by weight descending
            sorted_partners = sorted(partners.items(), key=lambda x: -x[1])
            
            k_used = top_k
            max_k = top_k * max_expansion_factor
            
            if dynamic_expansion and top_m > 0:
                selected = dict(sorted_partners[:k_used])
                unique_types = len(selected)
                
                while unique_types < top_m and k_used < max_k and k_used < len(sorted_partners):
                    k_used += 5
                    selected = dict(sorted_partners[:k_used])
                    unique_types = len(selected)
                
                return selected, k_used
            else:
                return dict(sorted_partners[:top_k]), top_k
        
        upstream_partners, up_k_used = apply_top_k_m(upstream_all, top_k, top_m)
        downstream_partners, down_k_used = apply_top_k_m(downstream_all, top_k, top_m)
        
        up_total = sum(upstream_partners.values()) if upstream_partners else 0.0
        down_total = sum(downstream_partners.values()) if downstream_partners else 0.0
        
        upstream_ranked = sorted(upstream_partners.items(), key=lambda x: -x[1])
        downstream_ranked = sorted(downstream_partners.items(), key=lambda x: -x[1])
        
        upstream_ranks = {k: i+1 for i, (k, _) in enumerate(upstream_ranked)}
        downstream_ranks = {k: i+1 for i, (k, _) in enumerate(downstream_ranked)}
        
        is_weak = len(upstream_partners) < 5 or len(downstream_partners) < 5
        is_sparse = (
            (top_m > 0 and len(upstream_partners) < top_m) or
            (top_m > 0 and len(downstream_partners) < top_m)
        )
        
        return ConnectivityProfile(
            neuron_id=bodyid,
            dataset=dataset,
            upstream_partners=upstream_partners,
            downstream_partners=downstream_partners,
            upstream_ranks=upstream_ranks,
            downstream_ranks=downstream_ranks,
            upstream_top_k=up_k_used,
            downstream_top_k=down_k_used,
            total_upstream_weight=up_total,
            total_downstream_weight=down_total,
            actual_upstream_count=len(upstream_partners),
            actual_downstream_count=len(downstream_partners),
            is_weak_connectivity=is_weak,
            unique_types_upstream=len(upstream_partners),
            unique_types_downstream=len(downstream_partners),
            is_sparse=is_sparse,
            top_k_bodyid_used=max(up_k_used, down_k_used),
            top_m_type_target=top_m,
        )
    
    def _build_profile_from_aggregates(
        self,
        neuron_type: str,
        dataset: str,
        upstream_by_type: Dict[str, Dict[str, float]],
        downstream_by_type: Dict[str, Dict[str, float]],
        top_k: Optional[int] = None,
        top_m: Optional[int] = None
    ) -> Optional['ConnectivityProfile']:
        """
        DEPRECATED: Use self.profiler.get_profile(neuron_type, dataset) instead.
        
        This method builds profiles from pre-aggregated data WITHOUT the 1-hop/2-hop
        hybrid approach. All profile building should now go through ConnectivityProfiler
        which provides proper 2-hop expansion for untyped 1-hop partners.
        
        The proper approach is:
        1. Build bodyId-level profiles via ConnectivityProfiler.get_profile()
        2. For type-level comparison, aggregate bodyId profiles
        
        Args:
            neuron_type: Type name
            dataset: Dataset name
            upstream_by_type: Pre-aggregated upstream connections
            downstream_by_type: Pre-aggregated downstream connections
            top_k: Top K partners to keep per direction (uses config default if None)
            top_m: Minimum unique types to ensure (uses config default if None)
            
        Returns:
            ConnectivityProfile or None if no connections found
            
        .. deprecated::
            Use `self.profiler.get_profile(neuron_type, dataset)` for proper 1-hop/2-hop
            hybrid profile building with 2-hop expansion for untyped neurons.
        """
        import warnings
        warnings.warn(
            "_build_profile_from_aggregates is deprecated. "
            "Use self.profiler.get_profile(neuron_type, dataset) for proper 1-hop/2-hop hybrid profiles.",
            DeprecationWarning,
            stacklevel=2
        )
        from .connectivity_profiler import ConnectivityProfile
        
        # Get defaults from profiler config
        if top_k is None:
            top_k = self.profiler.config.top_k_bodyid
        if top_m is None:
            top_m = self.profiler.config.top_m_type
        
        max_expansion_factor = self.profiler.config.max_expansion_factor
        dynamic_expansion = self.profiler.config.dynamic_expansion
        
        upstream_all = upstream_by_type.get(neuron_type, {})
        downstream_all = downstream_by_type.get(neuron_type, {})
        
        if not upstream_all and not downstream_all:
            return None
        
        def apply_top_k_m(partners: Dict[str, float], top_k: int, top_m: int) -> Tuple[Dict[str, float], int]:
            """Apply top-k filtering with optional top-m expansion."""
            if not partners:
                return {}, top_k
            
            # Sort by weight descending
            sorted_partners = sorted(partners.items(), key=lambda x: -x[1])
            
            k_used = top_k
            max_k = top_k * max_expansion_factor
            
            if dynamic_expansion and top_m > 0:
                # Expand k until we have at least top_m unique types
                selected = dict(sorted_partners[:k_used])
                unique_types = len(selected)
                
                while unique_types < top_m and k_used < max_k and k_used < len(sorted_partners):
                    k_used += 5  # Expand by 5
                    selected = dict(sorted_partners[:k_used])
                    unique_types = len(selected)
                
                return selected, k_used
            else:
                # Just take top-k
                return dict(sorted_partners[:top_k]), top_k
        
        # Apply top-k/top-m to both directions
        upstream_partners, up_k_used = apply_top_k_m(upstream_all, top_k, top_m)
        downstream_partners, down_k_used = apply_top_k_m(downstream_all, top_k, top_m)
        
        # Store actual synapse weights (not normalized to proportions)
        up_total = sum(upstream_partners.values()) if upstream_partners else 0.0
        down_total = sum(downstream_partners.values()) if downstream_partners else 0.0
        
        # Create ranks from weights (higher weight = lower rank)
        upstream_ranked = sorted(upstream_partners.items(), key=lambda x: -x[1])
        downstream_ranked = sorted(downstream_partners.items(), key=lambda x: -x[1])
        
        upstream_ranks = {k: i+1 for i, (k, _) in enumerate(upstream_ranked)}
        downstream_ranks = {k: i+1 for i, (k, _) in enumerate(downstream_ranked)}
        
        # Determine if weak connectivity (fewer than 5 partners in either direction)
        is_weak = len(upstream_partners) < 5 or len(downstream_partners) < 5
        
        # Check if sparse (couldn't reach top_m types even after expansion)
        is_sparse = (
            (top_m > 0 and len(upstream_partners) < top_m) or
            (top_m > 0 and len(downstream_partners) < top_m)
        )
        
        return ConnectivityProfile(
            neuron_id=neuron_type,
            dataset=dataset,
            upstream_partners=upstream_partners,
            downstream_partners=downstream_partners,
            upstream_ranks=upstream_ranks,
            downstream_ranks=downstream_ranks,
            upstream_top_k=up_k_used,
            downstream_top_k=down_k_used,
            total_upstream_weight=up_total,
            total_downstream_weight=down_total,
            actual_upstream_count=len(upstream_partners),
            actual_downstream_count=len(downstream_partners),
            is_weak_connectivity=is_weak,
            unique_types_upstream=len(upstream_partners),
            unique_types_downstream=len(downstream_partners),
            is_sparse=is_sparse,
            top_k_bodyid_used=max(up_k_used, down_k_used),
            top_m_type_target=top_m,
        )
    
    def find_homologs_fast(
        self,
        source: Optional[Union[str, int]] = None,
        source_dataset: Optional[str] = None,
        target_dataset: Optional[str] = None,
        top_n: Optional[int] = None,
        min_shared_partners: int = 2,
        min_weight: int = 3,
        show_progress: bool = True,
        output_dir: Optional[str] = None,
        saveas: Optional[str] = None,
        include_partner_details: bool = True,
        top_n_details: int = 10,
        run_shuffle_test: bool = False,
        n_shuffles: int = 100,
        shuffle_seed: Optional[int] = None,
        visualize_skeleton: Optional[bool] = None,
        visualize_top_n: Optional[int] = None,
        similarity_metric: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fast homolog discovery via adjacency expansion.
        
        This method uses adjacency expansion to efficiently find candidates:
        
        Cross-Dataset Algorithm (type-based expansion):
            1. Get source neuron's upstream types (A) and downstream types (B)
            2. Find target bodyIds with types in A → set A' (same types as source's upstream)
            3. Find target bodyIds with types in B → set B' (same types as source's downstream)
            4. Get A' neurons' downstream types → set C (upstream's downstream)
            5. Get B' neurons' upstream types → set D (downstream's upstream)
            6. Build profiles for candidates in A' ∪ B' ∪ C ∪ D
            7. Compare with source profiles
        
        Same-Dataset Algorithm (shared-partner expansion):
            1. Find bodyIds that share upstream or downstream partners with source
            2. Build profiles for those candidates
            3. Compare with source profiles
        
        Profile Building (via ConnectivityProfiler):
            - top_k: Top K partners per direction by synapse weight
            - top_m: Minimum unique types via dynamic K expansion
            - expand_untyped_2hop: 2-hop typed partners for untyped 1-hop neurons
        
        Uses module-level defaults (self.source, self.source_dataset, etc.) 
        when parameters are not provided.
        
        For comprehensive search of the ENTIRE target dataset, use find_homologs().
        
        Args:
            source: Neuron type name (str) or bodyId (int).
                   Uses self.source if not provided.
            source_dataset: Source dataset. Uses self.source_dataset if not provided.
            target_dataset: Target dataset to search. Uses self.target_dataset if not provided.
            top_n: Maximum candidates to return per source neuron (default: 20)
            min_shared_partners: Minimum shared partners to be a candidate (default: 2)
            min_weight: Minimum synapse weight for candidate discovery (default: 3)
            show_progress: Show progress bar
            output_dir: Directory to save results. Uses self.output_dir if not provided.
            saveas: Custom folder name. Uses self.saveas if not provided.
            include_partner_details: Include per-match partner overlap CSVs (when saving)
            top_n_details: Number of top matches to include detailed profiles for
            run_shuffle_test: If True, run random control test to validate results
            n_shuffles: Number of shuffle iterations for control test (default: 100)
            shuffle_seed: Random seed for reproducibility of shuffle test
            visualize_skeleton: Generate 3D visualizations. Uses self.visualize_skeleton if None.
            visualize_top_n: Number of top candidates to visualize. Uses self.visualize_top_n if None.
            similarity_metric: Metric for sorting top-N candidates. Options:
                - 'combined': Weighted average of jaccard and rank_corr (default)
                - 'rank_corr': Rank correlation on shared partners only
                - 'rank_union': Rank correlation on union of all partners (missing=0)
                - 'jaccard': Jaccard similarity
                - 'cosine': Cosine similarity
        
        Returns:
            DataFrame with columns (bodyId-to-bodyId comparison):
                - source_bodyId: Source neuron bodyId
                - source_type: Source neuron type
                - target_bodyId: Target neuron bodyId
                - target_type: Target neuron type
                - target_dataset: Target dataset
                - adjacency_score: Number of shared partners found during candidate discovery
                - shared_type_count: Number of types used for rank_corr calculation
                - union_type_count: Total unique types/bodyIds in union
                - rank_corr: Normalized rank correlation [0,1] (shared partners only)
                - rank_corr_raw: Raw Spearman correlation [-1,1]
                - rank_union: Normalized rank correlation [0,1] (all partners, missing=0)
                - rank_union_raw: Raw Spearman correlation [-1,1]
                - jaccard: Jaccard similarity
                - cosine: Cosine similarity
                - combined: Weighted combined score
                - is_same_type: True if source and target types match
            
            Results are sorted by source_bodyId then selected similarity_metric (descending).
            
            If run_shuffle_test=True, additional columns are added:
                - shuffle_p_value: P-value from shuffle test
                - shuffle_z_score: Z-score from shuffle test
                - shuffle_significant: True if p < 0.05
        
        Output Files (always saved when output_dir is set):
            - bodyid_results.csv: BodyId-level comparisons (sorted by source_bodyId, metric)
            - intra_type_results.csv: Intra-type self-comparisons (when source is a type)
            - type_summary.csv: Type-level aggregated summary (avg/best/std metrics)
            - homolog_results.csv: Legacy format (sorted by metric only)
            - visualization/: 3D skeleton plots (if visualize_skeleton=True)
        
        Note:
            - All profile building uses ConnectivityProfiler's 1-hop/2-hop hybrid approach
            - Always compares individual bodyIds for comprehensive results
            - Uses adjacency expansion to find candidates with shared partners
            - For comprehensive search without candidate filtering, use find_homologs()
        """
        # Use module-level defaults where not provided
        query = source if source is not None else self.source
        source_dataset = source_dataset if source_dataset is not None else self.source_dataset
        target_dataset = target_dataset if target_dataset is not None else self.target_dataset
        output_dir = output_dir if output_dir is not None else self.output_dir
        saveas = saveas if saveas is not None else self.saveas
        visualize_skeleton = visualize_skeleton if visualize_skeleton is not None else self.visualize_skeleton
        visualize_top_n = visualize_top_n if visualize_top_n is not None else self.visualize_top_n
        similarity_metric = similarity_metric if similarity_metric is not None else self.similarity_metric
        top_n = top_n if top_n is not None else self.top_n
        
        # Validate required parameters
        if query is None:
            raise ValueError("No source neuron specified. Set 'source' parameter or self.source.")
        if source_dataset is None:
            raise ValueError("No source_dataset specified. Set 'source_dataset' parameter or self.source_dataset.")
        if target_dataset is None:
            raise ValueError("No target_dataset specified. Set 'target_dataset' parameter or self.target_dataset.")
        
        is_bodyid = isinstance(query, int) or (isinstance(query, str) and str(query).isdigit())
        query_label = f"bodyId:{query}" if is_bodyid else query
        is_cross_dataset = (source_dataset != target_dataset)
        
        self._log(f"Fast homolog search: {query_label} from {source_dataset} → {target_dataset}")
        
        # Step 0: Ensure connection caches are complete BEFORE loading
        # This must happen first to respect cache hierarchy (connections → profiles)
        self._log("Ensuring connection caches are complete...")
        self._ensure_connection_cache_complete(source_dataset)
        if is_cross_dataset:
            self._log(f"Checking target dataset: {target_dataset}...")
            self._ensure_connection_cache_complete(target_dataset)
        self._log("Connection caches verified.")

        # Step 0.5: Pre-warm profile caches so subsequent loads avoid cold reads
        self._log("Pre-warming profile caches...")
        self._prewarm_profile_cache(source_dataset)
        if is_cross_dataset:
            self._prewarm_profile_cache(target_dataset)
        
        # Step 1: Load connection caches
        self._log("Loading and processing connection data...")
        self._in_progress_bar = True
        try:
            with tqdm(total=4, desc="Loading data", disable=not show_progress or not self.verbose, leave=True) as pbar:
                # Load source cache (already complete from Step 0)
                source_conn = self._load_connection_cache(source_dataset, auto_build=False)
                if source_conn is None:
                    self._log(f"ERROR: Could not load connection cache for {source_dataset}")
                    return pd.DataFrame()
                pbar.update(1)
                pbar.set_description("Aggregating source")
                
                # Pre-aggregate source (type-level)
                source_up, source_down = self._build_type_aggregates(source_conn, min_weight, show_progress=False)
                
                # Build bodyId-level aggregates and lookups up-front so both type and bodyId
                # queries can reuse the shared comparison core.
                source_bodyid_up, source_bodyid_down = self._build_bodyid_aggregates(
                    source_conn, min_weight, show_progress=False
                )
                source_type_lookup = self._build_bodyid_type_lookup(source_conn)

                # WORKFLOW OPTIMIZATION: Build source profiles NOW while source connections are in memory
                # This avoids reloading source connections later when building profiles
                # Get source bodyIds for the query type
                if not is_bodyid:
                    source_bodyids_early = self.get_bodyids_for_type(str(query), source_dataset)
                    if source_bodyids_early:
                        # Set progress bar flag for profiler to use tqdm.write
                        self.profiler._in_progress_bar = True
                        
                        # Pre-load connection cache into profiler BEFORE logging (avoids log interruption)
                        try:
                            self.profiler._get_cached_conn_df(source_dataset)
                        except Exception:
                            pass
                        
                        # Consolidate any existing batch files first
                        try:
                            self.profiler.consolidate_profile_cache(source_dataset)
                        except Exception:
                            pass
                        
                        tqdm.write(f"[HomologFinder] Pre-building {len(source_bodyids_early)} source profiles while connections are loaded...")
                        
                        # Build source profiles with deferred writes
                        self.profiler._defer_cache_writes = True
                        source_profiles_early: Dict[int, Any] = {}
                        
                        for bid in source_bodyids_early:
                            try:
                                profile = self.profiler.get_profile(bid, source_dataset)
                                if profile is not None:
                                    source_profiles_early[bid] = profile
                            except Exception:
                                pass
                        
                        # Flush cache writes
                        self.profiler._defer_cache_writes = False
                        try:
                            self.profiler.flush_pending_cache_writes(silent=True)
                        except Exception:
                            pass
                        
                        # Consolidate batch files into main cache
                        try:
                            self.profiler.consolidate_profile_cache(source_dataset)
                        except Exception:
                            pass
                        
                        tqdm.write(f"[HomologFinder] Pre-built {len(source_profiles_early)} source profiles")
                        self.profiler._in_progress_bar = False
                pbar.update(1)
                
                # Release source before loading target to avoid OOM
                if is_cross_dataset:
                    self._log("Releasing source DataFrame before loading target...")
                    del source_conn
                    
                    # Also clear FNC's cache for source dataset to free memory
                    source_safe_name = source_dataset.replace(':', '_').replace('.', '_')
                    try:
                        from coana import _FNC_CACHE
                        if source_safe_name in _FNC_CACHE:
                            if 'conn_df' in _FNC_CACHE[source_safe_name]:
                                _FNC_CACHE[source_safe_name]['conn_df'] = None
                            self._log(f"Cleared FNC cache for {source_dataset}")
                    except ImportError:
                        pass
                    
                    import gc
                    gc.collect()
                    
                    pbar.set_description("Loading target")
                    target_conn = self._load_connection_cache(target_dataset, auto_build=False)
                    if target_conn is None:
                        self._log(f"ERROR: Could not load connection cache for {target_dataset}")
                        return pd.DataFrame()
                    pbar.update(1)
                    pbar.set_description("Aggregating target")
                    target_up, target_down = self._build_type_aggregates(target_conn, min_weight, show_progress=False)
                    
                    # Build target bodyId aggregates and lookups for shared-core comparison
                    target_bodyid_up, target_bodyid_down = self._build_bodyid_aggregates(
                        target_conn, min_weight, show_progress=False
                    )
                    target_type_lookup = self._build_bodyid_type_lookup(target_conn)
                    
                    # Release target after building aggregates
                    del target_conn
                    gc.collect()
                    pbar.update(1)
                else:
                    # Same dataset - reuse source aggregates
                    target_conn = None  # Don't need a separate reference
                    target_up, target_down = source_up, source_down
                    target_bodyid_up, target_bodyid_down = source_bodyid_up, source_bodyid_down
                    target_type_lookup = source_type_lookup
                    pbar.update(2)
        finally:
            self._in_progress_bar = False
        
        # =====================================================================
        # BodyId-to-bodyId comparison (always used for comprehensive results)
        # Type-level aggregation is done when saving results
        # =====================================================================
        if not is_bodyid:
            # Type query: Get all bodyIds for this type
            source_bodyids = self.get_bodyids_for_type(str(query), source_dataset)
            if not source_bodyids:
                self._log(f"ERROR: No bodyIds found for type '{query}' in {source_dataset}")
                return pd.DataFrame()
            self._log(f"Running bodyId-to-bodyId comparison for {len(source_bodyids)} neurons in type '{query}'")
            
            # Aggregates already built above during loading phase
            # Memory already freed - source_conn and target_conn no longer exist
            if is_cross_dataset:
                # source_type_lookup already set
                pass
            else:
                source_type_lookup = target_type_lookup
            
            # Use pre-built source profiles (built earlier while source connections were loaded)
            # This avoids reloading source connections
            from .connectivity_profiler import ConnectivityStatus
            source_profiles_cache: Dict[int, 'ConnectivityProfile'] = {}
            source_status_counts: Dict[str, int] = {s.value: 0 for s in ConnectivityStatus}
            source_status_map: Dict[int, ConnectivityStatus] = {}
            
            # Check if profiles were pre-built during the loading phase
            if 'source_profiles_early' in dir() and source_profiles_early:
                self._log(f"Using {len(source_profiles_early)} pre-built source profiles")
                source_profiles_cache = source_profiles_early
                for bid, profile in source_profiles_cache.items():
                    status = profile.connectivity_status
                    source_status_counts[status.value] += 1
                    source_status_map[bid] = status
            else:
                # Fallback: Build profiles now (may require reloading connection data)
                self._log("Building source profiles via ConnectivityProfiler (1-hop/2-hop hybrid)")
                
                # Consolidate any existing batch files first
                try:
                    self.profiler.consolidate_profile_cache(source_dataset)
                except Exception:
                    pass
                
                # Pre-load connection cache BEFORE starting progress bar
                self._log(f"Pre-loading connection cache for {source_dataset}...")
                try:
                    self.profiler._get_cached_conn_df(source_dataset)
                except Exception:
                    pass
                
                # Enable deferred cache writes for batch saving
                self.profiler._defer_cache_writes = True
                self._in_progress_bar = True
                self.profiler._in_progress_bar = True
                
                try:
                    pbar = tqdm(
                        source_bodyids, 
                        desc="Building source profiles",
                        disable=not show_progress or not self.verbose,
                        leave=False
                    )
                    
                    for bid in pbar:
                        try:
                            profile = self.profiler.get_profile(bid, source_dataset)
                            if profile is not None:
                                source_profiles_cache[bid] = profile
                                status = profile.connectivity_status
                                source_status_counts[status.value] += 1
                                source_status_map[bid] = status
                        except Exception as e:
                            tqdm.write(f"[HomologFinder] Warning: Could not build profile for source {bid}: {e}")
                    
                    pbar.close()
                finally:
                    self._in_progress_bar = False
                    self.profiler._in_progress_bar = False
                    self.profiler._defer_cache_writes = False
                    try:
                        self.profiler.flush_pending_cache_writes(silent=True)
                    except Exception:
                        pass
                    # Consolidate batch files into main cache
                    try:
                        self.profiler.consolidate_profile_cache(source_dataset)
                    except Exception:
                        pass
            
            # Log connectivity status breakdown
            status_summary = ", ".join([f"{k.upper()}: {v}" for k, v in source_status_counts.items() if v > 0])
            self._log(f"Source connectivity status breakdown: {status_summary}")
            
            # Identify sources to skip (NONE/ORPHAN) or warn (RARE/UNIDIRECTIONAL)
            none_sources = [bid for bid, status in source_status_map.items() 
                           if status in {ConnectivityStatus.NONE, ConnectivityStatus.ORPHAN}]
            warn_sources = [bid for bid, status in source_status_map.items() 
                           if status in {ConnectivityStatus.RARE, ConnectivityStatus.UNIDIRECTIONAL}]
            if none_sources:
                self._log(f"⚠️ {len(none_sources)} source neurons have no partners (NONE/ORPHAN) and will be skipped")
            if warn_sources:
                self._log(f"⚠️ WARNING: {len(warn_sources)} source neurons are sparse or unidirectional (RARE/UNIDIRECTIONAL) - results may be unreliable")
            
            all_results = []
            # Track which target bodyIds we need profiles for
            all_candidate_bodyids: set = set()
            candidate_map: Dict[int, Dict[int, int]] = {}  # source_bid -> {target_bid: shared_count}
            
            # For both cross-dataset and same-dataset: use adjacency expansion
            # Cross-dataset: find target neurons by TYPE matching (source partner types → target bodyIds)
            # Same-dataset: find target neurons by shared partners
            
            if is_cross_dataset:
                # Cross-dataset adjacency expansion using TYPE matching:
                # Step 1: Get upstream types (A) and downstream types (B) from source neurons
                # Step 2: Find target bodyIds with those types (set A', B') - includes untyped
                # Step 3: Get A' downstream (C) and B' upstream (D) - includes untyped
                # Step 4: Candidate set = A' ∪ B' ∪ C ∪ D (includes both typed and untyped)
                # Note: Untyped neurons are only ignored in 2-hop profiles, not in candidate set
                
                # Collect all unique upstream/downstream types from source neurons
                # Include all source neurons (even weak ones) for comprehensive type collection
                self._log("Collecting source partner types...")
                all_upstream_types: set = set()
                all_downstream_types: set = set()
                
                for source_bid in source_bodyids:
                    source_profile = source_profiles_cache.get(source_bid)
                    if source_profile is None:
                        continue
                    all_upstream_types.update(source_profile.upstream_partners.keys())
                    all_downstream_types.update(source_profile.downstream_partners.keys())
                
                # Remove untyped placeholders from partner types
                all_upstream_types = {t for t in all_upstream_types if t and not pd.isna(t) and t != ''}
                all_downstream_types = {t for t in all_downstream_types if t and not pd.isna(t) and t != ''}
                
                self._log(f"Source partner types: {len(all_upstream_types)} upstream, {len(all_downstream_types)} downstream")
                
                # Build reverse lookup: type -> bodyIds in target dataset (typed neurons only)
                self._log("Building type-to-bodyId lookup for target dataset...")
                target_type_to_bodyids: Dict[str, set] = {}
                for bid, t in target_type_lookup.items():
                    if t and not pd.isna(t) and t != '':
                        if t not in target_type_to_bodyids:
                            target_type_to_bodyids[t] = set()
                        target_type_to_bodyids[t].add(bid)
                
                # Get all bodyIds in target dataset (for including untyped in C and D)
                all_target_bodyids = set(target_bodyid_up.keys()) | set(target_bodyid_down.keys())
                self._log(f"Target dataset: {len(all_target_bodyids)} total bodyIds")
                
                # Set A': Target bodyIds with types matching source's upstream partners
                self._log("Computing Set A' (type-matched upstream)...")
                set_a_prime: set = set()
                for up_type in all_upstream_types:
                    set_a_prime.update(target_type_to_bodyids.get(up_type, set()))
                
                # Set B': Target bodyIds with types matching source's downstream partners
                self._log("Computing Set B' (type-matched downstream)...")
                set_b_prime: set = set()
                for down_type in all_downstream_types:
                    set_b_prime.update(target_type_to_bodyids.get(down_type, set()))
                
                self._log(f"Type-matched targets: A'={len(set_a_prime)}, B'={len(set_b_prime)}")
                
                # Precompute partner type sets per bodyId for fast union
                downstream_sets_by_bodyid = {bid: set(partners.keys()) for bid, partners in target_bodyid_down.items()}
                upstream_sets_by_bodyid = {bid: set(partners.keys()) for bid, partners in target_bodyid_up.items()}

                # Set C: Downstream of A' (upstream's downstream)
                self._log("Computing Set C (downstream of A') via type unions...")
                downstream_types_c: set = set()
                for bid in set_a_prime:
                    downstream_types_c.update(downstream_sets_by_bodyid.get(bid, set()))
                set_c: set = set()
                for partner_type in downstream_types_c:
                    if partner_type and not pd.isna(partner_type) and partner_type != '':
                        set_c.update(target_type_to_bodyids.get(partner_type, set()))

                # Set D: Upstream of B' (downstream's upstream)
                self._log("Computing Set D (upstream of B') via type unions...")
                upstream_types_d: set = set()
                for bid in set_b_prime:
                    upstream_types_d.update(upstream_sets_by_bodyid.get(bid, set()))
                set_d: set = set()
                for partner_type in upstream_types_d:
                    if partner_type and not pd.isna(partner_type) and partner_type != '':
                        set_d.update(target_type_to_bodyids.get(partner_type, set()))
                
                # Candidate set = A' ∪ B' ∪ C ∪ D (includes both typed and untyped)
                all_candidate_bodyids = set_a_prime | set_b_prime | set_c | set_d
                
                # Count typed vs untyped
                typed_candidates = {bid for bid in all_candidate_bodyids 
                                   if target_type_lookup.get(bid, '') and 
                                   not pd.isna(target_type_lookup.get(bid, '')) and
                                   target_type_lookup.get(bid, '') != ''}
                untyped_candidates = all_candidate_bodyids - typed_candidates
                
                self._log(f"Cross-dataset candidates: A'={len(set_a_prime)}, B'={len(set_b_prime)}, C={len(set_c)}, D={len(set_d)} → {len(all_candidate_bodyids)} unique ({len(typed_candidates)} typed, {len(untyped_candidates)} untyped)")
                
                # For cross-dataset, all source neurons compare against same candidate pool
                # Compute adjacency_score as the count of partner-type overlaps per candidate
                # (shared upstream/downstream partner types relative to the source partner sets)
                for source_bid in source_bodyids:
                    candidate_scores: Dict[int, int] = {}
                    for bid in all_candidate_bodyids:
                        shared_downstream = len(downstream_sets_by_bodyid.get(bid, set()) & all_upstream_types)
                        shared_upstream = len(upstream_sets_by_bodyid.get(bid, set()) & all_downstream_types)
                        score = shared_downstream + shared_upstream
                        candidate_scores[bid] = score
                    candidate_map[source_bid] = candidate_scores
            else:
                # Same-dataset: use adjacency expansion (shared partner types)
                # Include all source neurons, even with weak connectivity
                # Precompute partner-type → bodyId indexes for fast lookups
                from collections import defaultdict
                upstream_partner_to_bodyids: Dict[str, set] = defaultdict(set)
                downstream_partner_to_bodyids: Dict[str, set] = defaultdict(set)
                for bid, partners in target_bodyid_up.items():
                    for ptype in partners.keys():
                        if ptype and not pd.isna(ptype) and ptype != '':
                            upstream_partner_to_bodyids[ptype].add(bid)
                for bid, partners in target_bodyid_down.items():
                    for ptype in partners.keys():
                        if ptype and not pd.isna(ptype) and ptype != '':
                            downstream_partner_to_bodyids[ptype].add(bid)

                for source_bid in tqdm(
                    source_bodyids,
                    desc=f"Finding candidates for {query} neurons",
                    disable=not show_progress or not self.verbose,
                    leave=False
                ):
                    # Get source profile for candidate finding
                    source_profile = source_profiles_cache.get(source_bid)
                    
                    if source_profile is None:
                        continue
                    
                    # Get candidate bodyIds via adjacency expansion
                    # Find bodyIds that share partners with this source bodyId
                    upstream_types = set(source_profile.upstream_partners.keys())
                    downstream_types = set(source_profile.downstream_partners.keys())
                    
                    # Find target bodyIds that share upstream/downstream partners
                    candidate_bodyids: Dict[int, int] = {}  # bodyId -> shared partner count
                    
                    # Get bodyIds receiving from same upstream types
                    for up_type in upstream_types:
                        for target_bid in upstream_partner_to_bodyids.get(up_type, set()):
                            if target_bid == source_bid:
                                continue
                            candidate_bodyids[target_bid] = candidate_bodyids.get(target_bid, 0) + 1
                    
                    # Get bodyIds sending to same downstream types
                    for down_type in downstream_types:
                        for target_bid in downstream_partner_to_bodyids.get(down_type, set()):
                            if target_bid == source_bid:
                                continue
                            candidate_bodyids[target_bid] = candidate_bodyids.get(target_bid, 0) + 1
                    
                    # Filter by minimum shared partners
                    candidate_bodyids = {k: v for k, v in candidate_bodyids.items() if v >= min_shared_partners}
                    
                    candidate_map[source_bid] = candidate_bodyids
                    all_candidate_bodyids.update(candidate_bodyids.keys())
            
            # Build target profiles using the batch method with cache management
            target_profiles_cache: Dict[int, 'ConnectivityProfile'] = {}
            target_status_counts: Dict[str, int] = {s.value: 0 for s in ConnectivityStatus}
            target_status_map: Dict[int, ConnectivityStatus] = {}  # bodyId -> status
            
            if all_candidate_bodyids:
                target_profiles_cache = self._build_profiles_batch(
                    list(all_candidate_bodyids),
                    target_dataset,
                    show_progress=show_progress,
                    label="target"
                )
                # Track target connectivity status
                for bid, profile in target_profiles_cache.items():
                    status = profile.connectivity_status
                    target_status_counts[status.value] += 1
                    target_status_map[bid] = status
                
                # Log target status breakdown
                status_summary = ", ".join([f"{k.upper()}: {v}" for k, v in target_status_counts.items() if v > 0])
                self._log(f"Target connectivity status breakdown: {status_summary}")
            
            # Shared comparison core
            results_df, intra_type_df, skipped_sources, warned_sources, source_status_map, target_status_map, target_status_counts = self._compare_candidates_core(
                source_bodyids=source_bodyids,
                source_profiles_cache=source_profiles_cache,
                source_status_map=source_status_map,
                target_profiles_cache=target_profiles_cache,
                target_type_lookup=target_type_lookup,
                source_type_lookup=source_type_lookup,
                candidate_map=candidate_map,
                is_cross_dataset=is_cross_dataset,
                target_dataset=target_dataset,
                show_progress=show_progress,
                similarity_metric=similarity_metric,
                top_n=top_n,
                min_score=None,
                include_intra_type=not is_cross_dataset,
                vector_prefiltering=self.vector_prefiltering,
            )

            # Log skipped and warned sources by status
            total_skipped = len(skipped_sources['none'])
            total_warned = len(warned_sources['rare_or_uni'])
            if total_skipped > 0:
                self._log(f"Skipped {total_skipped} source neurons with no partners (NONE/ORPHAN):")
                self._log(f"  - {skipped_sources['none'][:5]}{'...' if len(skipped_sources['none']) > 5 else ''}")
            if total_warned > 0:
                self._log(f"WARNING: {total_warned} source neurons are sparse/unidirectional (RARE/UNIDIRECTIONAL) - results may be unreliable:")
                self._log(f"  - {warned_sources['rare_or_uni'][:5]}{'...' if len(warned_sources['rare_or_uni']) > 5 else ''}")

            if results_df.empty:
                self._log("No valid candidates after profile comparison")
                return pd.DataFrame()

            self._log(f"Found {len(results_df)} bodyId-level matches across {len(source_bodyids)} source neurons")

            # Build source status summary for JSON export
            source_status_summary = {
                'query': str(query),
                'source_dataset': source_dataset,
                'total_source_neurons': len(source_bodyids),
                'status_breakdown': source_status_counts,
                'skipped_sources': {
                    'none': skipped_sources['none'],
                    'total': len(skipped_sources['none'])
                },
                'warned_sources': {
                    'rare_or_uni': warned_sources['rare_or_uni'],
                    'total': len(warned_sources['rare_or_uni'])
                },
                'included_for_comparison': len(source_bodyids) - len(skipped_sources['none']),
                'status_hierarchy': {
                    'none': 'No connections (0 partners) - SKIPPED entirely',
                    'rare_or_uni': 'Rare (<5 partners) or unidirectional - INCLUDED with WARNING',
                    'incomplete': 'Fewer than top_k partners - included with Warning',
                    'incomplete_expansion': 'Fewer than top_m unique types - included with Warning',
                    'complete': 'Full profile meeting all criteria'
                }
            }
            
            # Save results - always save (use default output_dir if not specified)
            save_output_dir = output_dir if output_dir is not None else self.output_dir
            self._save_homolog_results_internal(
                results_df=results_df,
                query=query,
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                output_dir=save_output_dir,
                saveas=saveas,
                direction='both',
                include_partner_details=include_partner_details,
                top_n_details=top_n_details,
                params={
                    'query': query,
                    'source_dataset': source_dataset,
                    'target_dataset': target_dataset,
                    'top_n': top_n,
                    'min_shared_partners': min_shared_partners,
                    'min_weight': min_weight,
                    'method': 'find_homologs_fast',
                    'source_bodyids_count': len(source_bodyids),
                    'profile_method': '1-hop/2-hop hybrid via ConnectivityProfiler'
                },
                visualize_skeleton=visualize_skeleton,
                visualize_top_n=visualize_top_n,
                intra_type_df=intra_type_df,  # Pass intra-type results for saving
                similarity_metric=self.similarity_metric,
                source_status_summary=source_status_summary
            )
            
            return results_df
        
        # =====================================================================
        # BodyId query: single neuron comparison (also used for single-bodyId type)
        # =====================================================================
        
            from .connectivity_profiler import ConnectivityStatus

            # Build source profile using ConnectivityProfiler (1-hop/2-hop hybrid)
            self._log("Building source profile via ConnectivityProfiler (1-hop/2-hop hybrid)")
            source_profile = self.profiler.get_profile(query, source_dataset)
            if source_profile is None:
                self._log(f"ERROR: No connections found for {query_label}")
                return pd.DataFrame()

            source_bid = int(query)
            source_bodyids = [source_bid]

            upstream_types = set(source_profile.upstream_partners.keys())
            downstream_types = set(source_profile.downstream_partners.keys())

            self._log(f"Source profile: {len(upstream_types)} upstream, {len(downstream_types)} downstream types")

            # Build source caches for shared comparison core
            source_profiles_cache: Dict[int, 'ConnectivityProfile'] = {source_bid: source_profile}
            source_status_map: Dict[int, ConnectivityStatus] = {source_bid: source_profile.connectivity_status}
            source_status_counts: Dict[str, int] = {s.value: 0 for s in ConnectivityStatus}
            source_status_counts[source_profile.connectivity_status.value] += 1
            source_type_lookup_single = {
                source_bid: source_type_lookup.get(source_bid, source_profile.neuron_type or str(query))
            }

            # Step 3: Find candidates via adjacency expansion (2-hop neighbors, bodyId-level)
            self._log("Finding candidate field via adjacency expansion (bodyId-level)...")

            candidate_map: Dict[int, Dict[int, int]] = {}
            all_candidate_bodyids: set = set()

            if is_cross_dataset:
                # Cross-dataset: use partner TYPE overlap to expand candidates, then map to bodyIds
                self._log("Collecting source partner types for cross-dataset expansion...")
                all_upstream_types = {t for t in upstream_types if t and not pd.isna(t) and t != ''}
                all_downstream_types = {t for t in downstream_types if t and not pd.isna(t) and t != ''}

                self._log(f"Source partner types: {len(all_upstream_types)} upstream, {len(all_downstream_types)} downstream")

                # Build reverse lookup: type -> bodyIds in target dataset (typed neurons only)
                self._log("Building type-to-bodyId lookup for target dataset...")
                target_type_to_bodyids: Dict[str, set] = {}
                for bid, t in target_type_lookup.items():
                    if t and not pd.isna(t) and t != '':
                        target_type_to_bodyids.setdefault(t, set()).add(bid)

                # Get all bodyIds in target dataset (for including untyped in C and D)
                all_target_bodyids = set(target_bodyid_up.keys()) | set(target_bodyid_down.keys())
                self._log(f"Target dataset: {len(all_target_bodyids)} total bodyIds")

                # Set A': Target bodyIds with types matching source's upstream partners
                self._log("Computing Set A' (type-matched upstream)...")
                set_a_prime: set = set()
                for up_type in all_upstream_types:
                    set_a_prime.update(target_type_to_bodyids.get(up_type, set()))

                # Set B': Target bodyIds with types matching source's downstream partners
                self._log("Computing Set B' (type-matched downstream)...")
                set_b_prime: set = set()
                for down_type in all_downstream_types:
                    set_b_prime.update(target_type_to_bodyids.get(down_type, set()))

                # Precompute partner type sets per bodyId for fast union
                downstream_sets_by_bodyid = {bid: set(partners.keys()) for bid, partners in target_bodyid_down.items()}
                upstream_sets_by_bodyid = {bid: set(partners.keys()) for bid, partners in target_bodyid_up.items()}

                # Set C: Downstream of A' (upstream's downstream)
                self._log("Computing Set C (downstream of A') via type unions...")
                downstream_types_c: set = set()
                for bid in set_a_prime:
                    downstream_types_c.update(downstream_sets_by_bodyid.get(bid, set()))
                set_c: set = set()
                for partner_type in downstream_types_c:
                    if partner_type and not pd.isna(partner_type) and partner_type != '':
                        set_c.update(target_type_to_bodyids.get(partner_type, set()))

                # Set D: Upstream of B' (downstream's upstream)
                self._log("Computing Set D (upstream of B') via type unions...")
                upstream_types_d: set = set()
                for bid in set_b_prime:
                    upstream_types_d.update(upstream_sets_by_bodyid.get(bid, set()))
                set_d: set = set()
                for partner_type in upstream_types_d:
                    if partner_type and not pd.isna(partner_type) and partner_type != '':
                        set_d.update(target_type_to_bodyids.get(partner_type, set()))

                # Candidate set = A' ∪ B' ∪ C ∪ D (includes both typed and untyped)
                all_candidate_bodyids = set_a_prime | set_b_prime | set_c | set_d

                # Count typed vs untyped
                typed_candidates = {bid for bid in all_candidate_bodyids 
                                   if target_type_lookup.get(bid, '') and 
                                   not pd.isna(target_type_lookup.get(bid, '')) and
                                   target_type_lookup.get(bid, '') != ''}
                untyped_candidates = all_candidate_bodyids - typed_candidates

                self._log(f"Cross-dataset candidates: A'={len(set_a_prime)}, B'={len(set_b_prime)}, C={len(set_c)}, D={len(set_d)} → {len(all_candidate_bodyids)} unique ({len(typed_candidates)} typed, {len(untyped_candidates)} untyped)")

                candidate_scores: Dict[int, int] = {}
                for bid in all_candidate_bodyids:
                    shared_downstream = len(downstream_sets_by_bodyid.get(bid, set()) & all_upstream_types)
                    shared_upstream = len(upstream_sets_by_bodyid.get(bid, set()) & all_downstream_types)
                    candidate_scores[bid] = shared_downstream + shared_upstream
                candidate_map[source_bid] = candidate_scores
            else:
                # Same-dataset: adjacency expansion using shared partner types
                from collections import defaultdict
                upstream_partner_to_bodyids: Dict[str, set] = defaultdict(set)
                downstream_partner_to_bodyids: Dict[str, set] = defaultdict(set)
                for bid, partners in target_bodyid_up.items():
                    for ptype in partners.keys():
                        if ptype and not pd.isna(ptype) and ptype != '':
                            upstream_partner_to_bodyids[ptype].add(bid)
                for bid, partners in target_bodyid_down.items():
                    for ptype in partners.keys():
                        if ptype and not pd.isna(ptype) and ptype != '':
                            downstream_partner_to_bodyids[ptype].add(bid)

                candidate_bodyids: Dict[int, int] = {}

                # Get bodyIds receiving from same upstream types
                for up_type in upstream_types:
                    for target_bid in upstream_partner_to_bodyids.get(up_type, set()):
                        if target_bid == source_bid:
                            continue
                        candidate_bodyids[target_bid] = candidate_bodyids.get(target_bid, 0) + 1

                # Get bodyIds sending to same downstream types
                for down_type in downstream_types:
                    for target_bid in downstream_partner_to_bodyids.get(down_type, set()):
                        if target_bid == source_bid:
                            continue
                        candidate_bodyids[target_bid] = candidate_bodyids.get(target_bid, 0) + 1

                # Filter by minimum shared partners
                candidate_bodyids = {k: v for k, v in candidate_bodyids.items() if v >= min_shared_partners}

                candidate_map[source_bid] = candidate_bodyids
                all_candidate_bodyids.update(candidate_bodyids.keys())

            if not all_candidate_bodyids:
                self._log("No candidates found after adjacency expansion")
                return pd.DataFrame()

            # Step 4: Build profiles for all candidates using ConnectivityProfiler (1-hop/2-hop hybrid)
            target_profiles_cache: Dict[int, 'ConnectivityProfile'] = {}
            target_status_counts: Dict[str, int] = {s.value: 0 for s in ConnectivityStatus}
            target_status_map: Dict[int, ConnectivityStatus] = {}

            target_profiles_cache = self._build_profiles_batch(
                list(all_candidate_bodyids),
                target_dataset,
                show_progress=show_progress,
                label="target"
            )
            for bid, profile in target_profiles_cache.items():
                status = profile.connectivity_status
                target_status_counts[status.value] += 1
                target_status_map[bid] = status

            if target_status_counts:
                status_summary = ", ".join([f"{k.upper()}: {v}" for k, v in target_status_counts.items() if v > 0])
                self._log(f"Target connectivity status breakdown: {status_summary}")

            # Step 5: Compare via shared core
            results_df, intra_type_df, skipped_sources, warned_sources, source_status_map, target_status_map, target_status_counts = self._compare_candidates_core(
                source_bodyids=source_bodyids,
                source_profiles_cache=source_profiles_cache,
                source_status_map=source_status_map,
                target_profiles_cache=target_profiles_cache,
                target_type_lookup=target_type_lookup,
                source_type_lookup=source_type_lookup_single,
                candidate_map=candidate_map,
                is_cross_dataset=is_cross_dataset,
                target_dataset=target_dataset,
                show_progress=show_progress,
                similarity_metric=similarity_metric,
                top_n=top_n,
                min_score=None,
                include_intra_type=False,
                vector_prefiltering=self.vector_prefiltering,
            )

            # Log skipped and warned sources by status
            total_skipped = len(skipped_sources['none'])
            total_warned = len(warned_sources['rare_or_uni'])
            if total_skipped > 0:
                self._log(f"Skipped {total_skipped} source neurons with no partners (NONE/ORPHAN):")
                self._log(f"  - {skipped_sources['none'][:5]}{'...' if len(skipped_sources['none']) > 5 else ''}")
            if total_warned > 0:
                self._log(f"WARNING: {total_warned} source neurons are sparse/unidirectional (RARE/UNIDIRECTIONAL) - results may be unreliable:")
                self._log(f"  - {warned_sources['rare_or_uni'][:5]}{'...' if len(warned_sources['rare_or_uni']) > 5 else ''}")

            if results_df.empty:
                self._log("No valid candidates after profile comparison")
                return pd.DataFrame()

            self._log(f"Found {len(results_df)} bodyId-level matches")

            # Run shuffle test if requested (before saving so we include results)
            shuffle_stats = None
            if run_shuffle_test:
                self._log(f"Running shuffle control test with {n_shuffles} iterations...")

                query_profile = self.profiler.get_profile(
                    query,
                    source_dataset
                )

                if query_profile is None:
                    self._log("Warning: Could not get query profile for shuffle test")
                else:
                    shuffle_stats = self.run_random_control_test(
                        source=query,
                        source_dataset=source_dataset,
                        target_dataset=target_dataset,
                        n_shuffles=n_shuffles,
                        top_n=top_n,
                        seed=shuffle_seed,
                        show_progress=show_progress
                    )

                    if shuffle_stats and 'p_value' in shuffle_stats:
                        results_df['shuffle_p_value'] = shuffle_stats['p_value']
                        results_df['shuffle_z_score'] = shuffle_stats['z_score']
                        results_df['shuffle_effect_size'] = shuffle_stats['effect_size']
                        results_df['shuffle_significant'] = shuffle_stats.get('is_significant', 
                                                                             shuffle_stats['p_value'] < 0.05)

                    self._log(f"Shuffle test: p={shuffle_stats['p_value']:.4f}, "
                             f"z={shuffle_stats['z_score']:.2f}, d={shuffle_stats['effect_size']:.2f}")

            # Save results - always save (use default output_dir if not specified)
            save_output_dir = output_dir if output_dir is not None else self.output_dir

            # Build source status summary for JSON export
            source_status_summary = {
                'query': str(query),
                'source_dataset': source_dataset,
                'total_source_neurons': len(source_bodyids),
                'status_breakdown': source_status_counts,
                'skipped_sources': {
                    'none': skipped_sources['none'],
                    'total': len(skipped_sources['none'])
                },
                'warned_sources': {
                    'rare_or_uni': warned_sources['rare_or_uni'],
                    'total': len(warned_sources['rare_or_uni'])
                },
                'included_for_comparison': len(source_bodyids) - len(skipped_sources['none']),
                'status_hierarchy': {
                    'none': 'No connections (0 partners) - SKIPPED entirely',
                    'rare_or_uni': 'Rare (<5 partners) or unidirectional - INCLUDED with WARNING',
                    'incomplete': 'Fewer than top_k partners - included with Warning',
                    'incomplete_expansion': 'Fewer than top_m unique types - included with Warning',
                    'complete': 'Full profile meeting all criteria'
                }
            }

            self._save_homolog_results_internal(
                results_df=results_df,
                query=query,
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                output_dir=save_output_dir,
                saveas=saveas,
                direction='both',
                include_partner_details=include_partner_details,
                top_n_details=top_n_details,
                params={
                    'query': query,
                    'source_dataset': source_dataset,
                    'target_dataset': target_dataset,
                    'top_n': top_n,
                    'min_shared_partners': min_shared_partners,
                    'min_weight': min_weight,
                    'method': 'find_homologs_fast',
                    'source_bodyids_count': len(source_bodyids),
                    'profile_method': '1-hop/2-hop hybrid via ConnectivityProfiler'
                },
                shuffle_stats=shuffle_stats,
                visualize_skeleton=visualize_skeleton,
                visualize_top_n=visualize_top_n,
                similarity_metric=self.similarity_metric,
                intra_type_df=intra_type_df,
                source_status_summary=source_status_summary
            )

            return results_df
    
    def find_homologs_intra_dataset(
        self,
        query: Union[str, int],
        dataset: str,
        search_untyped: bool = False,
        top_n: int = 20,
        min_weight: int = 3,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Find homologs within the same dataset.
        
        For intra-dataset search, useful for:
        - Finding untyped neurons similar to a typed query
        - Finding sub-populations within a type
        - Finding neurons with similar connectivity profiles
        
        Args:
            query: Neuron type (str) or bodyId (int)
            dataset: Dataset to search
            search_untyped: If True, search untyped neurons; else search typed
            top_n: Number of top results
            min_weight: Minimum synapse weight
            show_progress: Show progress bar
        
        Returns:
            DataFrame with matches
        """
        # Use find_homologs with same source/target dataset
        return self.find_homologs(
            query=query,
            source_dataset=dataset,
            target_dataset=dataset,
            top_n=top_n,
            show_progress=show_progress
        )

    def _save_homolog_results_internal(
        self,
        results_df: pd.DataFrame,
        query: Union[str, int],
        source_dataset: str,
        target_dataset: str,
        output_dir: str,
        saveas: Optional[str],
        direction: str,
        include_partner_details: bool,
        top_n_details: int,
        params: Dict[str, Any],
        shuffle_stats: Optional[Dict[str, Any]] = None,
        visualize_skeleton: bool = False,
        visualize_top_n: int = 5,
        intra_type_df: Optional[pd.DataFrame] = None,
        similarity_metric: str = 'rank_union',
        source_status_summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Internal method to save homolog finding results with organized folder structure.
        
        Called by find_homologs() and find_homologs_fast() when output_dir is provided.
        
        Output folder structure:
            {output_dir}/{saveas or timestamp}/
            ├── README.txt                    # Parameters, summary, and column descriptions
            ├── results/                      # Main results
            │   ├── homolog_results.csv       # Full results with all columns (sorted by metric)
            │   ├── type_summary.csv          # Type-level aggregated summary
            │   ├── bodyid_results.csv        # BodyId-level results (sorted by source, metric)
            │   ├── intra_type_results.csv    # Intra-type comparisons (if type query)
            │   ├── shuffle_test.json         # Shuffle test statistics (if run)
            │   └── source_status_summary.json # Source neuron connectivity status breakdown
            ├── profiles/                     # Connectivity profiles
            │   ├── query/                    # Query neuron profiles
            │   │   ├── {query}.csv           # Type-level profile
            │   │   └── source_bodyids.csv    # Source bodyId summary (if bodyId-level)
            │   └── matches/                  # Top match profiles  
            │       ├── top_target_bodyids.csv # Target bodyId summary (if bodyId-level)
            │       ├── {match_type1}.csv     # Type-level profile for top match types
            │       └── {match_type2}.csv
            ├── overlaps/                     # Partner overlap details
            │   ├── {query}_vs_{match1}.csv
            │   └── {query}_vs_{match2}.csv
            └── visualization/                # 3D skeleton plots (if enabled)
                ├── {type}_{bodyId}.html      # Interactive HTML per match
                └── ...
        
        Args:
            results_df: Results DataFrame
            query: Original query neuron
            source_dataset: Source dataset
            target_dataset: Target dataset
            output_dir: Base output directory
            saveas: Custom folder name (None = auto-generate with timestamp)
            direction: 'upstream', 'downstream', or 'both' for overlap details
            include_partner_details: Include per-match partner overlap CSVs
            top_n_details: Number of top matches to include detailed profiles for
            params: Dict of input parameters to save
            shuffle_stats: Optional shuffle test statistics dict to save
            visualize_skeleton: If True, generate 3D skeleton visualizations
            visualize_top_n: Number of top matches to visualize
            intra_type_df: Optional DataFrame with intra-type comparison results
            similarity_metric: The metric to use for sorting results 
                             ('rank_union', 'rank_corr', 'jaccard', 'cosine', 'combined')
            source_status_summary: Optional dict with source neuron connectivity status breakdown
        
        Returns:
            Dict with output_dir, files_saved, query_profile, match_profiles
        """
        import os
        import json
        from pathlib import Path
        from datetime import datetime
        
        # Check for bodyId-level columns
        has_bodyid_cols = 'source_bodyId' in results_df.columns if not results_df.empty else False
        
        # Generate folder name with timestamp if not provided
        if saveas is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_query = str(query).replace('/', '_').replace(':', '_').replace('*', '_')
            folder_name = f'homologs_{safe_query}_{timestamp}'
        else:
            folder_name = saveas
        
        # Create organized folder structure
        output_path = Path(output_dir) / folder_name
        results_dir = output_path / 'results'
        profiles_dir = output_path / 'profiles'
        query_profiles_dir = profiles_dir / 'query'
        match_profiles_dir = profiles_dir / 'matches'
        overlaps_dir = output_path / 'overlaps'
        visualization_dir = output_path / 'visualization'
        
        # Create all directories
        dirs_to_create = [output_path, results_dir, query_profiles_dir, match_profiles_dir, overlaps_dir]
        if visualize_skeleton:
            dirs_to_create.append(visualization_dir)
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        files_saved = []
        
        self._log(f"Saving homolog results to {output_path}")
        
        # 1. Save README.txt with parameters, summary, and shuffle test results
        readme_file = output_path / 'README.txt'
        with open(readme_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("  HOMOLOG FINDING RESULTS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Query: {query}\n")
            f.write(f"Source Dataset: {source_dataset}\n")
            f.write(f"Target Dataset: {target_dataset}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("  INPUT PARAMETERS\n")
            f.write("-" * 70 + "\n")
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\n" + "-" * 70 + "\n")
            f.write("  RESULTS SUMMARY\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Total matches: {len(results_df)}\n")
            
            # BodyId-level stats
            if has_bodyid_cols:
                unique_sources = results_df['source_bodyId'].nunique() if 'source_bodyId' in results_df.columns else 0
                unique_targets = results_df['target_bodyId'].nunique() if 'target_bodyId' in results_df.columns else 0
                f.write(f"  Unique source bodyIds: {unique_sources}\n")
                f.write(f"  Unique target bodyIds: {unique_targets}\n")
            
            if not results_df.empty and 'rank_corr' in results_df.columns:
                f.write(f"  Best rank_corr: {results_df['rank_corr'].max():.4f}\n")
                f.write(f"  Mean rank_corr: {results_df['rank_corr'].mean():.4f}\n")
                f.write(f"  Median rank_corr: {results_df['rank_corr'].median():.4f}\n")
            if not results_df.empty and 'jaccard' in results_df.columns:
                f.write(f"  Best jaccard: {results_df['jaccard'].max():.4f}\n")
                f.write(f"  Mean jaccard: {results_df['jaccard'].mean():.4f}\n")
            
            # Shuffle test results section
            if shuffle_stats:
                f.write("\n" + "-" * 70 + "\n")
                f.write("  SHUFFLE TEST RESULTS (Statistical Validation)\n")
                f.write("-" * 70 + "\n")
                f.write(f"  Number of shuffles: {shuffle_stats.get('n_shuffles', 'N/A')}\n")
                f.write(f"  Observed score: {shuffle_stats.get('observed_score', 0):.4f}\n")
                f.write(f"  Mean shuffled score: {shuffle_stats.get('mean_shuffled', 0):.4f}\n")
                f.write(f"  Std shuffled score: {shuffle_stats.get('std_shuffled', 0):.4f}\n")
                f.write(f"  P-value: {shuffle_stats.get('p_value', 1.0):.4f}\n")
                f.write(f"  Z-score: {shuffle_stats.get('z_score', 0):.2f}\n")
                f.write(f"  Effect size (Cohen's d): {shuffle_stats.get('effect_size', 0):.2f}\n")
                f.write(f"  Interpretation: {shuffle_stats.get('interpretation', 'N/A')}\n")
            
            f.write("\n" + "-" * 70 + "\n")
            f.write("  FOLDER STRUCTURE\n")
            f.write("-" * 70 + "\n")
            f.write("  results/\n")
            f.write("    ├── bodyid_results.csv     BodyId-level comparisons (sorted by source_bodyId, rank_corr)\n")
            f.write("    ├── type_summary.csv       Type-level aggregated summary\n")
            f.write("    ├── homolog_results.csv    Full results sorted by rank_corr (legacy format)\n")
            if shuffle_stats:
                f.write("    └── shuffle_test.json      Shuffle test statistics\n")
            f.write("  profiles/\n")
            f.write("    ├── query/                 Query neuron connectivity profile\n")
            f.write("    └── matches/               Top match connectivity profiles\n")
            f.write("  overlaps/\n")
            f.write("    └── *_vs_*.csv             Partner overlap details per match\n")
            if visualize_skeleton:
                f.write("  visualization/\n")
                f.write("    └── *.html, *.png          3D skeleton plots of top matches\n")
            
            # Add interpretation guide
            f.write("\n" + "-" * 70 + "\n")
            f.write("  INTERPRETATION GUIDE\n")
            f.write("-" * 70 + "\n")
            f.write("  combined: Weighted score of jaccard + rank_corr (0-1, higher=better)\n")
            f.write("  rank_corr: Spearman correlation on SHARED partners (0-1, higher=better)\n")
            f.write("  rank_corr_raw: Raw Spearman correlation (-1 to 1)\n")
            f.write("  rank_union: Spearman correlation on ALL partners, missing=0 (0-1)\n")
            f.write("  rank_union_raw: Raw Spearman correlation for union (-1 to 1)\n")
            f.write("  jaccard: Jaccard similarity of partner sets (0-1, higher=better)\n")
            f.write("  cosine: Cosine similarity of weight vectors (0-1, higher=better)\n")
            f.write("  adjacency_score: Number of shared partners found during candidate discovery\n")
            f.write("                   (from 2-hop adjacency expansion, NOT from profile comparison)\n")
            f.write("  shared_type_count: Number of types/bodyIds used for rank_corr calculation\n")
            f.write("                     (from actual profile comparison)\n")
            f.write("  union_type_count: Total unique types/bodyIds in both profiles\n")
            if shuffle_stats:
                f.write("\n  Shuffle Test Interpretation:\n")
                f.write("    P-value < 0.05: Result is statistically significant\n")
                f.write("    Z-score > 2: Observed score is 2+ std deviations above random\n")
                f.write("    Effect size > 0.8: Large effect (meaningful difference)\n")
        
        files_saved.append('README.txt')
        self._log("Saved: README.txt")
        
        # 2. Save bodyId-level results (always saved, sorted by source_bodyId then rank_corr)
        # This is the foundational comparison data showing each source-target bodyId pair
        if not results_df.empty:
            # Round numeric columns for better readability
            numeric_cols = ['rank_corr', 'rank_corr_raw', 'rank_union', 'rank_union_raw', 'jaccard', 'cosine', 'combined']
            results_rounded = results_df.copy()
            for col in numeric_cols:
                if col in results_rounded.columns:
                    results_rounded[col] = results_rounded[col].round(4)
            
            # Determine sorting columns based on available data
            # Use the user-selected similarity_metric for sorting
            sort_metric = similarity_metric if similarity_metric in results_rounded.columns else 'rank_union'
            if sort_metric not in results_rounded.columns:
                # Fallback to any available metric
                for fallback in ['rank_corr', 'combined', 'jaccard', 'cosine']:
                    if fallback in results_rounded.columns:
                        sort_metric = fallback
                        break
            
            if 'source_bodyId' in results_rounded.columns:
                # BodyId-level data: sort by source_bodyId, then similarity_metric
                bodyid_sort_cols = ['source_bodyId', sort_metric]
                bodyid_sorted = results_rounded.sort_values(
                    bodyid_sort_cols, 
                    ascending=[True, False],
                    na_position='last'
                )
                
                # Save bodyId-level results with clear column order
                # Include source_status/target_status to indicate connectivity quality
                # Include weak_source/weak_target columns (legacy) to indicate neurons with <5 partners
                # Include source/target_partner_count to diagnose comparison (bodyIds for intra, types for cross)
                # shared_type_count shows how many shared types were used for rank correlation
                # union_type_count shows total unique types in union (for rank_union)
                bodyid_cols = ['source_bodyId', 'source_type', 'target_bodyId', 'target_type',
                              'combined', 'rank_corr', 'rank_corr_raw', 'rank_union', 'rank_union_raw',
                              'jaccard', 'cosine', 
                              'adjacency_score', 'shared_type_count', 'union_type_count',
                              'is_same_type', 'is_same_dataset', 
                              'source_status', 'target_status',  # Hierarchical status
                              'weak_source', 'weak_target',  # Legacy flags
                              'source_partner_count', 'target_partner_count']
                available_bodyid_cols = [c for c in bodyid_cols if c in bodyid_sorted.columns]
                bodyid_df = bodyid_sorted[available_bodyid_cols].copy()
                bodyid_df.to_csv(results_dir / 'bodyid_results.csv', index=False)
                files_saved.append('results/bodyid_results.csv')
                self._log(f"Saved: results/bodyid_results.csv (sorted by source_bodyId, {sort_metric})")
            else:
                # Type-level only data (source_neuron instead of source_bodyId)
                bodyid_sorted = results_rounded.sort_values(sort_metric, ascending=False, na_position='last')
            
            # Also save as homolog_results.csv (legacy format, sorted by similarity_metric)
            results_by_score = results_rounded.sort_values(sort_metric, ascending=False, na_position='last')
            results_by_score.to_csv(results_dir / 'homolog_results.csv', index=False)
            files_saved.append('results/homolog_results.csv')
            self._log(f"Saved: results/homolog_results.csv (sorted by {sort_metric})")
        
        # 2b. Save intra-type comparison results if provided
        if intra_type_df is not None and not intra_type_df.empty:
            # Round numeric columns
            intra_type_rounded = intra_type_df.copy()
            for col in ['rank_corr', 'rank_corr_raw', 'rank_union', 'rank_union_raw', 'jaccard', 'cosine', 'combined']:
                if col in intra_type_rounded.columns:
                    intra_type_rounded[col] = intra_type_rounded[col].round(4)
            
            # Sort by combined (descending) to show most similar pairs first
            sort_col = 'combined' if 'combined' in intra_type_rounded.columns else 'jaccard'
            intra_type_sorted = intra_type_rounded.sort_values(sort_col, ascending=False)
            intra_type_sorted.to_csv(results_dir / 'intra_type_results.csv', index=False)
            files_saved.append('results/intra_type_results.csv')
            self._log(f"Saved: results/intra_type_results.csv ({len(intra_type_sorted)} pairs)")
        
        # 3. Save shuffle test results as JSON
        if shuffle_stats:
            shuffle_file = results_dir / 'shuffle_test.json'
            with open(shuffle_file, 'w') as f:
                # Convert to JSON-serializable format, excluding DataFrames and complex objects
                serializable_stats = {}
                exclude_keys = ['real_results', 'shuffled_results']  # These are DataFrames
                
                for k, v in shuffle_stats.items():
                    if k in exclude_keys:
                        continue  # Skip DataFrames
                    if v is None:
                        serializable_stats[k] = None
                    elif isinstance(v, pd.DataFrame):
                        continue  # Skip DataFrames
                    elif hasattr(v, 'item'):  # numpy scalar
                        serializable_stats[k] = v.item()
                    elif isinstance(v, np.ndarray):
                        serializable_stats[k] = v.tolist()
                    elif isinstance(v, list):
                        # Handle lists that might contain numpy types
                        serializable_stats[k] = [
                            x.item() if hasattr(x, 'item') else x 
                            for x in v 
                            if not isinstance(x, pd.DataFrame)
                        ]
                    elif isinstance(v, (str, int, float, bool)):
                        serializable_stats[k] = v
                    else:
                        # Try to convert to string for other types
                        try:
                            serializable_stats[k] = str(v)
                        except:
                            pass
                
                json.dump(serializable_stats, f, indent=2)
            files_saved.append('results/shuffle_test.json')
            self._log("Saved: results/shuffle_test.json")
        
        # 3b. Save source status summary as JSON
        # This provides detailed breakdown of source neuron connectivity status
        if source_status_summary is not None:
            status_summary_file = results_dir / 'source_status_summary.json'
            with open(status_summary_file, 'w') as f:
                json.dump(source_status_summary, f, indent=2)
            files_saved.append('results/source_status_summary.json')
            self._log("Saved: results/source_status_summary.json")
        
        # 4. Get and save query profile
        # For bodyId-level comparisons, save both type-level and individual bodyId profiles
        query_profile = self.get_profile(query, source_dataset)
        query_profile_data = None
        
        if query_profile is not None:
            query_profile_data = self._profile_to_dataframe(query_profile, source_dataset)
            safe_query = str(query).replace('/', '_').replace(':', '_')
            query_profile_data.to_csv(query_profiles_dir / f'{safe_query}.csv', index=False)
            files_saved.append(f'profiles/query/{safe_query}.csv')
            self._log(f"Saved: profiles/query/{safe_query}.csv (type-level)")
        
        # If bodyId-level comparison, also save source bodyId summary
        if has_bodyid_cols and not results_df.empty:
            # Create summary of source bodyIds with their status and connectivity
            source_summary = results_df[['source_bodyId', 'source_type', 'source_status', 
                                         'source_partner_count']].drop_duplicates()
            source_summary = source_summary.sort_values('source_bodyId')
            source_summary.to_csv(query_profiles_dir / 'source_bodyids.csv', index=False)
            files_saved.append('profiles/query/source_bodyids.csv')
            self._log(f"Saved: profiles/query/source_bodyids.csv ({len(source_summary)} source neurons)")
        
        # 5. Get and save top match profiles + partner overlaps
        match_profiles = {}
        
        if not results_df.empty and include_partner_details:
            top_matches = results_df.head(top_n_details)
            
            # For bodyId-level comparisons, save target bodyId summary for top matches
            if has_bodyid_cols:
                target_cols = ['target_bodyId', 'target_type', 'target_status', 
                              'target_partner_count', 'rank_corr', 'jaccard', 'combined']
                available_target_cols = [c for c in target_cols if c in top_matches.columns]
                target_summary = top_matches[available_target_cols].drop_duplicates(subset=['target_bodyId'])
                target_summary = target_summary.sort_values('rank_corr', ascending=False, na_position='last')
                target_summary.to_csv(match_profiles_dir / 'top_target_bodyids.csv', index=False)
                files_saved.append('profiles/matches/top_target_bodyids.csv')
                self._log(f"Saved: profiles/matches/top_target_bodyids.csv ({len(target_summary)} top targets)")
            
            for _, row in top_matches.iterrows():
                target_type = row.get('target_type', row.get('target', None))
                if target_type is None or pd.isna(target_type) or str(target_type) == '':
                    continue
                
                try:
                    # Get match profile (type-level)
                    match_profile = self.get_profile(target_type, target_dataset)
                    if match_profile is None:
                        continue
                    
                    # Save match profile to matches subfolder
                    match_data = self._profile_to_dataframe(match_profile, target_dataset)
                    safe_type = str(target_type).replace('/', '_').replace(':', '_')
                    match_data.to_csv(match_profiles_dir / f'{safe_type}.csv', index=False)
                    files_saved.append(f'profiles/matches/{safe_type}.csv')
                    match_profiles[target_type] = match_data
                    
                    # Get partner overlap details and save to overlaps subfolder
                    if query_profile is not None:
                        overlap_dfs = []
                        
                        for dir_name in ['upstream', 'downstream'] if direction == 'both' else [direction]:
                            overlap = ProfileComparator.get_partner_overlap_details(
                                query_profile, match_profile, dir_name
                            )
                            if not overlap.empty:
                                overlap['direction'] = dir_name
                                overlap_dfs.append(overlap)
                        
                        if overlap_dfs:
                            combined_overlap = pd.concat(overlap_dfs, ignore_index=True)
                            safe_query_name = str(query).replace('/', '_').replace(':', '_')
                            overlap_file = f'{safe_query_name}_vs_{safe_type}.csv'
                            combined_overlap.to_csv(overlaps_dir / overlap_file, index=False)
                            files_saved.append(f'overlaps/{overlap_file}')
                
                except Exception as e:
                    self._log(f"Warning: Could not save details for {target_type}: {e}")
        
        # 6. Save type-level summary (always generated by aggregating results)
        # This aggregates bodyId-level comparisons to show type-level patterns
        if not results_df.empty:
            # Determine the target column name (could be target_type or target)
            target_col = 'target_type' if 'target_type' in results_df.columns else 'target'
            
            if has_bodyid_cols and target_col in results_df.columns:
                # Filter out rows with NaN rank_corr (from weak sources with 0 partners)
                # These have no valid connectivity profile and shouldn't be included in type summary
                valid_results = results_df[results_df['rank_corr'].notna()].copy()
                
                if valid_results.empty:
                    self._log("Warning: No valid comparisons for type summary (all sources have invalid status)")
                    # Create empty type summary with proper columns (averages only)
                    type_summary = pd.DataFrame(columns=[
                        'query', 'source_dataset', 'target_dataset', 'source_type', target_col,
                        'avg_rank_corr', 'avg_jaccard', 'avg_combined', 'avg_rank_union',
                        'avg_cosine', 'avg_adjacency_score', 'n_bodyid_comparisons',
                        'n_complete_sources', 'n_incomplete_sources'
                    ])
                else:
                    # Aggregate bodyId results to type level
                    # Group by both source_type and target_type to show pairwise type comparisons
                    if 'source_type' in valid_results.columns:
                        group_cols = ['source_type', target_col]
                    else:
                        group_cols = [target_col]
                    
                    # Aggregate with averages only for all similarity metrics
                    agg_dict = {'rank_corr': ['mean', 'count']}
                    if 'jaccard' in valid_results.columns:
                        agg_dict['jaccard'] = 'mean'
                    if 'combined' in valid_results.columns:
                        agg_dict['combined'] = 'mean'
                    if 'rank_union' in valid_results.columns:
                        agg_dict['rank_union'] = 'mean'
                    if 'cosine' in valid_results.columns:
                        agg_dict['cosine'] = 'mean'
                    if 'adjacency_score' in valid_results.columns:
                        agg_dict['adjacency_score'] = 'mean'
                    if 'shared_type_count' in valid_results.columns:
                        agg_dict['shared_type_count'] = 'mean'
                    if 'union_type_count' in valid_results.columns:
                        agg_dict['union_type_count'] = 'mean'
                    
                    # Add connectivity status aggregation
                    if 'source_status' in valid_results.columns:
                        # Count sources by status type
                        agg_dict['source_status'] = lambda x: (x == 'complete').sum()
                    
                    type_summary = valid_results.groupby(group_cols).agg(agg_dict).round(4)
                    type_summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in type_summary.columns.values]
                    type_summary = type_summary.reset_index()
                    
                    # Rename columns - use avg_ prefix for all metrics (no best_/std_)
                    rename_map = {
                        'rank_corr_mean': 'avg_rank_corr',
                        'rank_corr_count': 'n_bodyid_comparisons',
                        'jaccard_mean': 'avg_jaccard',
                        'combined_mean': 'avg_combined',
                        'rank_union_mean': 'avg_rank_union',
                        'cosine_mean': 'avg_cosine',
                        'adjacency_score_mean': 'avg_adjacency_score',
                        'shared_type_count_mean': 'avg_shared_type_count',
                        'union_type_count_mean': 'avg_union_type_count'
                    }
                    
                    # Handle the lambda aggregation column name
                    for col in type_summary.columns:
                        if 'source_status' in col and 'lambda' in col.lower():
                            rename_map[col] = 'n_complete_sources'
                    
                    type_summary = type_summary.rename(columns=rename_map)
                    
                    # If we have source_status, also count incomplete sources
                    if 'n_complete_sources' in type_summary.columns:
                        type_summary['n_incomplete_sources'] = type_summary['n_bodyid_comparisons'] - type_summary['n_complete_sources']
                    
                    # Sort by the average of the selected similarity_metric
                    sort_by_col = f'avg_{sort_metric}' if f'avg_{sort_metric}' in type_summary.columns else 'avg_rank_corr'
                    type_summary = type_summary.sort_values(sort_by_col, ascending=False, na_position='last')

                    # Mark which rows correspond to the types we will visualize (top visualize_top_n)
                    if visualize_top_n and visualize_top_n > 0:
                        type_summary['visualized'] = False
                        type_summary.loc[type_summary.index[:visualize_top_n], 'visualized'] = True
                        type_summary['visualization_rank'] = type_summary.index.to_series() + 1
                    
                    type_summary.insert(0, 'query', query)
                    type_summary.insert(1, 'source_dataset', source_dataset)
                    type_summary.insert(2, 'target_dataset', target_dataset)
            elif target_col in results_df.columns:
                # Already type-level, just format
                # Filter out NaN rows here too
                valid_results = results_df[results_df['rank_corr'].notna()].copy() if 'rank_corr' in results_df.columns else results_df.copy()
                pairwise_cols = [target_col, 'rank_corr', 'rank_corr_raw', 'jaccard', 
                                'adjacency_score', 'is_same_type']
                available_cols = [c for c in pairwise_cols if c in valid_results.columns]
                type_summary = valid_results[available_cols].drop_duplicates().copy()
                type_summary = type_summary.sort_values('rank_corr', ascending=False, na_position='last')
                if visualize_top_n and visualize_top_n > 0:
                    type_summary['visualized'] = False
                    type_summary.loc[type_summary.index[:visualize_top_n], 'visualized'] = True
                    type_summary['visualization_rank'] = type_summary.index.to_series() + 1
                type_summary.insert(0, 'query', query)
                type_summary.insert(1, 'source_dataset', source_dataset)
                type_summary.insert(2, 'target_dataset', target_dataset)
            else:
                # Fallback: just save as-is
                type_summary = results_df.copy()
                type_summary.insert(0, 'query', query)
                type_summary.insert(1, 'source_dataset', source_dataset)
                type_summary.insert(2, 'target_dataset', target_dataset)
            
            type_summary.to_csv(results_dir / 'type_summary.csv', index=False)
            files_saved.append('results/type_summary.csv')
            self._log("Saved: results/type_summary.csv (aggregated type-level summary)")
        
        # 7. Generate 3D skeleton visualizations if enabled
        if visualize_skeleton and not results_df.empty:
            self._visualize_homolog_candidates(
                results_df=results_df,
                query=query,
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                visualization_dir=visualization_dir,
                top_n=visualize_top_n,
                files_saved=files_saved,
                type_summary=type_summary
            )
        
        self._log(f"Saved {len(files_saved)} files to {output_path}")
        
        return {
            'output_dir': str(output_path),
            'files_saved': files_saved,
            'query_profile': query_profile_data,
            'match_profiles': match_profiles
        }
    
    def _visualize_homolog_candidates(
        self,
        results_df: pd.DataFrame,
        query: Union[str, int],
        source_dataset: str,
        target_dataset: str,
        visualization_dir: 'Path',
        top_n: int = 5,
        files_saved: List[str] = None,
        type_summary: Optional[pd.DataFrame] = None
    ):
        """
        Generate 3D skeleton visualizations for top homolog candidates.
        
        Creates three types of visualizations:
        1. bodyId_level/: Individual target bodyIds (single neuron plots)
        2. type_level/: Target types (may include multiple neurons per type)
        3. source_neurons/: All source neurons plotted together
        
        Uses VisualizeSkeleton module to create interactive HTML and PNG files.
        
        Args:
            results_df: Results DataFrame with homolog matches
            query: Query neuron identifier
            source_dataset: Source dataset
            target_dataset: Target dataset
            visualization_dir: Directory to save visualization files
            top_n: Number of top candidates to visualize
            files_saved: List to append saved file names to
        """
        try:
            # Import VisualizeSkeleton from coana module
            import sys
            from pathlib import Path
            
            # Ensure src is in path
            src_dir = Path(__file__).parent.parent
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            
            from coana import VisualizeSkeleton
            
            self._log(f"Generating 3D visualizations for top {top_n} candidates...")

            # Use original dataset names for VisualizeSkeleton to ensure correct NeuPrint connection
            # (e.g. 'hemibrain:v1.2.1' must be preserved, not converted to 'hemibrain_v1_2_1')
            vis_source_dataset = source_dataset
            vis_target_dataset = target_dataset
            
            # Create subdirectories for organized output
            bodyid_dir = visualization_dir / 'bodyId_level'
            type_dir = visualization_dir / 'type_level'
            source_dir = visualization_dir / 'source_neurons'
            
            bodyid_dir.mkdir(parents=True, exist_ok=True)
            type_dir.mkdir(parents=True, exist_ok=True)
            source_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine target column name
            target_col = 'target_type' if 'target_type' in results_df.columns else 'target'
            target_bodyid_col = 'target_bodyId' if 'target_bodyId' in results_df.columns else None
            
            # Get matches to visualize; for bodyId-level use union of per-source top-N targets
            if top_n and top_n > 0:
                if target_bodyid_col:
                    if 'source_bodyId' in results_df.columns:
                        per_source_matches = []
                        for _, group in results_df.groupby('source_bodyId'):
                            # Sort best-to-worst for stable top-N per source
                            if 'rank_corr' in group.columns:
                                sorted_group = group.sort_values('rank_corr', ascending=False, na_position='last')
                            else:
                                sorted_group = group
                            per_source_matches.append(sorted_group.head(top_n))

                        if per_source_matches:
                            top_matches = pd.concat(per_source_matches, ignore_index=True)
                        else:
                            top_matches = results_df.iloc[0:0]
                    else:
                        # Fallback when source_bodyId is unavailable
                        if 'rank_corr' in results_df.columns:
                            top_matches = results_df.nlargest(top_n, 'rank_corr')
                        else:
                            top_matches = results_df.head(top_n)

                    # Union of targets across sources to avoid duplicate visualizations
                    top_matches = top_matches.drop_duplicates(subset=[target_bodyid_col], keep='first')
                else:
                    top_matches = results_df.head(top_n)
            else:
                top_matches = results_df
            
            # =====================================================================
            # 1. BodyId-level visualizations: Individual target bodyIds
            # =====================================================================
            if target_bodyid_col:
                self._log(f"  Creating bodyId-level visualizations...")
                for idx, row in top_matches.iterrows():
                    target_type = row.get(target_col, '')
                    target_bodyid = row.get(target_bodyid_col)
                    rank_corr = row.get('rank_corr', 0)
                    
                    if not target_bodyid:
                        continue
                    
                    # Single neuron - use bodyId
                    layers = [int(target_bodyid)]
                    safe_name = f"{target_type}_{target_bodyid}".replace('/', '_').replace(':', '_').replace('*', '_')
                    
                    try:
                        vs = VisualizeSkeleton(
                            dataset=vis_target_dataset,
                            neuron_layers=layers,
                            saveas=safe_name,
                            data_folder=str(bodyid_dir),
                            show_fig=False,
                            brain_mesh='whole',
                            neuron_alpha=0.6,  # Higher alpha for single neuron
                            legend_mode='normal',
                            merge_neurons=False,
                            verbose='simple',
                        )
                        vs.plot_neurons()
                        
                        if files_saved is not None:
                            files_saved.append(f'visualization/bodyId_level/{safe_name}.html')
                            files_saved.append(f'visualization/bodyId_level/{safe_name}.png')
                        
                        self._log(f"    Saved: bodyId_level/{safe_name}.html (rank_corr={rank_corr:.3f})")
                        
                    except Exception as e:
                        self._log(f"    Warning: Could not visualize bodyId {target_bodyid}: {e}")
            
            # =====================================================================
            # 2. Type-level visualizations: Use type names (fetches all neurons of type)
            # =====================================================================
            if type_summary is not None and not type_summary.empty:
                ts = type_summary
                type_col = 'target_type' if 'target_type' in ts.columns else ('target' if 'target' in ts.columns else None)
                if type_col:
                    top_type_rows = ts.head(top_n) if top_n and top_n > 0 else ts
                    unique_types = top_type_rows[type_col].dropna().unique()
                else:
                    unique_types = []
            else:
                unique_types = top_matches[target_col].dropna().unique()

            self._log(f"  Creating type-level visualizations for {len(unique_types)} types...")
            
            for target_type in unique_types:
                if not target_type:
                    continue
                
                # Get best rank_corr for this type
                type_rows = results_df[results_df[target_col] == target_type]
                best_rank_corr = type_rows['rank_corr'].max() if 'rank_corr' in type_rows.columns else 0
                bodyid_layers: List[Union[int, str]] = []

                if target_bodyid_col:
                    bodyids_for_type = type_rows[target_bodyid_col].dropna().unique().tolist()
                    if bodyids_for_type:
                        try:
                            bodyid_layers = [int(bid) for bid in bodyids_for_type]
                        except (ValueError, TypeError):
                            bodyid_layers = [str(bid) for bid in bodyids_for_type]

                # Prefer explicit bodyIds from results; fall back to type name lookup
                if bodyid_layers:
                    # Group bodyIds so the type renders as a single combined layer
                    layers = [bodyid_layers] if len(bodyid_layers) > 1 else bodyid_layers
                else:
                    layers = [str(target_type)]
                safe_name = f"{target_type}".replace('/', '_').replace(':', '_').replace('*', '_')
                
                try:
                    vs = VisualizeSkeleton(
                        dataset=vis_target_dataset,
                        neuron_layers=layers,
                        saveas=safe_name,
                        data_folder=str(type_dir),
                        show_fig=False,
                        brain_mesh='whole',
                        verbose='simple',
                    )
                    vs.plot_neurons()
                    
                    if files_saved is not None:
                        files_saved.append(f'visualization/type_level/{safe_name}.html')
                        files_saved.append(f'visualization/type_level/{safe_name}.png')
                    
                    bodyid_note = f", bodyIds={len(bodyid_layers)}" if bodyid_layers else ""
                    self._log(f"    Saved: type_level/{safe_name}.html (best_rank_corr={best_rank_corr:.3f}{bodyid_note})")
                    
                except Exception as e:
                    self._log(f"    Warning: Could not visualize type {target_type}: {e}")
            
            # =====================================================================
            # 3. Source neurons visualization: All source neurons together
            # =====================================================================
            self._log(f"  Creating source neurons visualization...")
            
            # Collect all source bodyIds
            if 'source_bodyId' in results_df.columns:
                source_bodyids = results_df['source_bodyId'].dropna().unique().tolist()
            else:
                # Fall back to query as the source
                source_bodyids = [query]
            
            if source_bodyids:
                # Convert to int if they look like bodyIds
                try:
                    source_layers = [int(bid) for bid in source_bodyids]
                except (ValueError, TypeError):
                    # They might be type names
                    source_layers = [str(bid) for bid in source_bodyids]

                # Group multiple neurons so they render as a single combined layer
                if len(source_layers) > 1:
                    source_layers = [source_layers]
                
                safe_name = f"source_{query}".replace('/', '_').replace(':', '_').replace('*', '_')
                
                try:
                    vs = VisualizeSkeleton(
                        dataset=vis_source_dataset,
                        neuron_layers=source_layers,
                        saveas=safe_name,
                        data_folder=str(source_dir),
                        show_fig=False,
                        brain_mesh='whole',
                        legend_mode='normal',  # Show legend for multiple neurons
                        neuron_alpha=0.6 if len(source_layers) == 1 else 0.2,  # Higher alpha for single neuron
                        verbose='simple',
                    )
                    vs.plot_neurons()
                    
                    if files_saved is not None:
                        files_saved.append(f'visualization/source_neurons/{safe_name}.html')
                        files_saved.append(f'visualization/source_neurons/{safe_name}.png')
                    
                    self._log(f"    Saved: source_neurons/{safe_name}.html ({len(source_layers)} neurons)")
                    
                except Exception as e:
                    self._log(f"    Warning: Could not visualize source neurons: {e}")
            
        except ImportError as e:
            self._log(f"Warning: Could not import VisualizeSkeleton for visualization: {e}")
        except Exception as e:
            self._log(f"Warning: Visualization failed: {e}")
    
    def _profile_to_dataframe(
        self,
        profile: 'ConnectivityProfile',
        dataset: str
    ) -> pd.DataFrame:
        """
        Convert a ConnectivityProfile to a DataFrame with partners, weights, and ranks.
        
        Args:
            profile: ConnectivityProfile to convert
            dataset: Dataset identifier for context
        
        Returns:
            DataFrame with columns: neuron_type, dataset, direction, partner_type, weight, rank
        """
        rows = []
        neuron_id = str(profile.neuron_id)
        
        # Export upstream partners (sorted by rank)
        for partner_type, weight in sorted(
            profile.upstream_partners.items(),
            key=lambda x: profile.upstream_ranks.get(x[0], 999)
        ):
            rank = profile.upstream_ranks.get(partner_type, None)
            rows.append({
                'neuron_type': neuron_id,
                'dataset': dataset,
                'direction': 'upstream',
                'partner_type': partner_type,
                'weight': weight,
                'rank': rank
            })
        
        # Export downstream partners (sorted by rank)
        for partner_type, weight in sorted(
            profile.downstream_partners.items(),
            key=lambda x: profile.downstream_ranks.get(x[0], 999)
        ):
            rank = profile.downstream_ranks.get(partner_type, None)
            rows.append({
                'neuron_type': neuron_id,
                'dataset': dataset,
                'direction': 'downstream',
                'partner_type': partner_type,
                'weight': weight,
                'rank': rank
            })
        
        return pd.DataFrame(rows)

    # =========================================================================
    # Random Control Test
    # =========================================================================
    
    def shuffle_profile(
        self,
        profile: 'ConnectivityProfile',
        seed: Optional[int] = None
    ) -> 'ConnectivityProfile':
        """
        Create a shuffled version of a connectivity profile for control testing.
        
        Shuffles the partner types while keeping the same number of partners and
        weight distribution. This creates a randomized profile that maintains
        the statistical properties but loses the biological meaning.
        
        The shuffle randomly reassigns weights to different partner types from
        the same dataset's type pool, preserving:
        - Total number of upstream/downstream partners
        - Weight distribution (same set of weights, different assignment)
        - Rank structure (maintains the same ranking pattern)
        
        Args:
            profile: The original ConnectivityProfile to shuffle
            seed: Random seed for reproducibility (optional)
        
        Returns:
            A new ConnectivityProfile with shuffled partner assignments
        
        Example:
            >>> finder = HomologFinder(...)
            >>> profile = finder.profiler.get_profile('Mi1', 'hemibrain:v1.2.1')
            >>> shuffled = finder.shuffle_profile(profile, seed=42)
            >>> # Compare results with shuffled vs original profile
        """
        from .connectivity_profiler import ConnectivityProfile
        import copy
        
        if seed is not None:
            np.random.seed(seed)
        
        # Get all available types from the dataset for shuffling
        # We use the type pool from the dataset's connection cache
        dataset_key = profile.dataset.replace(':', '_').replace('.', '_')
        type_pool = self._get_type_pool(profile.dataset)
        
        if not type_pool:
            warnings.warn(f"No type pool available for {profile.dataset}, using profile types only")
            # Fall back to shuffling within existing types
            type_pool = list(set(list(profile.upstream_partners.keys()) + 
                               list(profile.downstream_partners.keys())))
        
        # Shuffle upstream partners
        upstream_weights = list(profile.upstream_partners.values())
        n_upstream = len(upstream_weights)
        if n_upstream > 0 and len(type_pool) >= n_upstream:
            shuffled_upstream_types = list(np.random.choice(
                type_pool, size=n_upstream, replace=False
            ))
        else:
            shuffled_upstream_types = list(np.random.permutation(
                list(profile.upstream_partners.keys())
            ))
        
        shuffled_upstream_partners = dict(zip(shuffled_upstream_types, upstream_weights))
        
        # Compute ranks for shuffled upstream
        sorted_upstream = sorted(shuffled_upstream_partners.items(), key=lambda x: -x[1])
        shuffled_upstream_ranks = {t: i+1 for i, (t, _) in enumerate(sorted_upstream)}
        
        # Shuffle downstream partners
        downstream_weights = list(profile.downstream_partners.values())
        n_downstream = len(downstream_weights)
        if n_downstream > 0 and len(type_pool) >= n_downstream:
            shuffled_downstream_types = list(np.random.choice(
                type_pool, size=n_downstream, replace=False
            ))
        else:
            shuffled_downstream_types = list(np.random.permutation(
                list(profile.downstream_partners.keys())
            ))
        
        shuffled_downstream_partners = dict(zip(shuffled_downstream_types, downstream_weights))
        
        # Compute ranks for shuffled downstream
        sorted_downstream = sorted(shuffled_downstream_partners.items(), key=lambda x: -x[1])
        shuffled_downstream_ranks = {t: i+1 for i, (t, _) in enumerate(sorted_downstream)}
        
        # Create new profile with shuffled data
        shuffled_profile = ConnectivityProfile(
            neuron_id=f"{profile.neuron_id}_shuffled",
            dataset=profile.dataset,
            upstream_partners=shuffled_upstream_partners,
            downstream_partners=shuffled_downstream_partners,
            upstream_ranks=shuffled_upstream_ranks,
            downstream_ranks=shuffled_downstream_ranks,
            upstream_top_k=profile.upstream_top_k,
            downstream_top_k=profile.downstream_top_k,
            total_upstream_weight=profile.total_upstream_weight,
            total_downstream_weight=profile.total_downstream_weight,
            num_neurons_aggregated=profile.num_neurons_aggregated,
            actual_upstream_count=profile.actual_upstream_count,
            actual_downstream_count=profile.actual_downstream_count,
            is_weak_connectivity=profile.is_weak_connectivity,
            unique_types_upstream=len(shuffled_upstream_partners),
            unique_types_downstream=len(shuffled_downstream_partners),
        )
        
        return shuffled_profile
    
    def _get_type_pool(self, dataset: str) -> List[str]:
        """
        Get all available neuron types from a dataset for shuffling.
        
        Args:
            dataset: Dataset identifier
        
        Returns:
            List of all unique type names in the dataset
        """
        # Try to get types from the profiler's connection cache
        dataset_key = dataset.replace(':', '_').replace('.', '_')
        
        # Check profiler cache first
        from .connectivity_profiler import _PROFILER_CONN_CACHE
        if dataset_key in _PROFILER_CONN_CACHE:
            type_lookup = _PROFILER_CONN_CACHE[dataset_key].get('type_lookup', {})
            return list(set(type_lookup.values()))
        
        # Try neuron index from the profiler
        try:
            if hasattr(self.profiler, '_load_neuron_index'):
                neuron_idx = self.profiler._load_neuron_index(dataset)
                if neuron_idx is not None and 'type' in neuron_idx.columns:
                    return list(neuron_idx['type'].dropna().unique())
        except Exception:
            pass
        
        # Fall back to connection data
        try:
            conn_data = self.profiler._load_connection_data(dataset)
            if conn_data is not None:
                types = set()
                if 'type_pre' in conn_data.columns:
                    types.update(conn_data['type_pre'].dropna().unique())
                if 'type_post' in conn_data.columns:
                    types.update(conn_data['type_post'].dropna().unique())
                return list(types)
        except Exception:
            pass
        
        return []
    
    def run_random_control_test(
        self,
        source: Union[str, int],
        source_dataset: str,
        target_dataset: str,
        n_shuffles: int = 100,
        top_n: int = 20,
        seed: Optional[int] = None,
        show_progress: bool = True,
        output_dir: Optional[str] = None,
        saveas: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a random control test for homolog finding.
        
        This method performs homolog finding on both the real profile and multiple
        shuffled profiles to assess whether the homolog finding results are
        statistically significant. If the real profile produces significantly
        higher similarity scores than shuffled profiles, the homolog finding
        is considered meaningful.
        
        Algorithm:
        1. Build the real connectivity profile for the source neuron
        2. Run homolog finding on the real profile → get real_scores
        3. For n_shuffles iterations:
           a. Shuffle the profile (randomize partner type assignments)
           b. Run homolog finding on shuffled profile → get shuffled_scores
        4. Compare real_scores vs distribution of shuffled_scores
        5. Compute p-values and effect sizes
        
        Args:
            source: Source neuron (type name or bodyId)
            source_dataset: Dataset of source neuron
            target_dataset: Dataset to search for homologs
            n_shuffles: Number of shuffle iterations (default: 100)
            top_n: Number of top homolog candidates to consider
            seed: Random seed for reproducibility
            show_progress: Whether to show progress bar
            output_dir: Directory to save results (optional)
            saveas: Custom name for output folder (optional)
        
        Returns:
            Dict containing:
                - real_results: DataFrame of homolog finding results with real profile
                - shuffled_results: List of DataFrames from shuffled profiles
                - real_mean_score: Mean similarity score for real profile
                - shuffled_mean_scores: List of mean scores for each shuffle
                - p_value: Proportion of shuffles with higher mean score than real
                - z_score: Z-score of real score vs shuffle distribution
                - effect_size: Cohen's d effect size
                - is_significant: Whether result is significant (p < 0.05)
                - summary: Human-readable summary string
        
        Example:
            >>> finder = HomologFinder(
            ...     source_dataset='hemibrain:v1.2.1',
            ...     target_dataset='male-cns:v0.9'
            ... )
            >>> results = finder.run_random_control_test(
            ...     source='Mi1',
            ...     source_dataset='hemibrain:v1.2.1',
            ...     target_dataset='male-cns:v0.9',
            ...     n_shuffles=100,
            ...     seed=42
            ... )
            >>> print(results['summary'])
            >>> if results['is_significant']:
            ...     print("Homolog finding is meaningful!")
        """
        from datetime import datetime
        from pathlib import Path
        
        if seed is not None:
            np.random.seed(seed)
        
        self._log(f"\n{'='*60}")
        self._log(f"Running Random Control Test for: {source}")
        self._log(f"Source dataset: {source_dataset}")
        self._log(f"Target dataset: {target_dataset}")
        self._log(f"Number of shuffles: {n_shuffles}")
        self._log(f"{'='*60}\n")
        
        # 1. Get the real profile
        self._log("Step 1: Building real connectivity profile...")
        try:
            real_profile = self.profiler.get_profile(source, source_dataset)
        except Exception as e:
            self._log(f"Error building profile: {e}")
            return {'error': str(e), 'is_significant': False}
        
        # 2. Run homolog finding with real profile
        self._log("Step 2: Running homolog finding with real profile...")
        real_results = self.find_homologs_fast(
            source=source,
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            top_n=top_n,
            show_progress=False,
        )
        
        if real_results.empty:
            self._log("No homologs found with real profile")
            return {
                'real_results': real_results,
                'shuffled_results': [],
                'real_mean_score': 0.0,
                'shuffled_mean_scores': [],
                'p_value': 1.0,
                'z_score': 0.0,
                'effect_size': 0.0,
                'is_significant': False,
                'summary': "No homologs found with real profile."
            }
        
        # Get mean rank_corr from real results (use normalized version)
        score_col = 'rank_corr' if 'rank_corr' in real_results.columns else 'jaccard'
        real_mean_score = real_results[score_col].mean()
        real_max_score = real_results[score_col].max()
        
        self._log(f"Real profile - Mean {score_col}: {real_mean_score:.4f}, Max: {real_max_score:.4f}")
        
        # 3. Run homolog finding with shuffled profiles
        self._log(f"\nStep 3: Running {n_shuffles} shuffle iterations...")
        shuffled_mean_scores = []
        shuffled_max_scores = []
        shuffled_results_list = []
        
        iterator = range(n_shuffles)
        if show_progress:
            try:
                iterator = tqdm(iterator, desc="Shuffle iterations")
            except:
                pass
        
        for i in iterator:
            # Create shuffled profile
            shuffle_seed = seed + i if seed is not None else None
            shuffled_profile = self.shuffle_profile(real_profile, seed=shuffle_seed)
            
            # Run homolog finding with shuffled profile
            # We need to temporarily replace the profile in the comparison
            shuffled_df = self._run_homolog_with_profile(
                shuffled_profile,
                target_dataset,
                top_n
            )
            
            if not shuffled_df.empty and score_col in shuffled_df.columns:
                shuffled_mean_scores.append(shuffled_df[score_col].mean())
                shuffled_max_scores.append(shuffled_df[score_col].max())
                shuffled_results_list.append(shuffled_df)
            else:
                shuffled_mean_scores.append(0.0)
                shuffled_max_scores.append(0.0)
        
        # 4. Compute statistics
        self._log("\nStep 4: Computing statistics...")
        shuffled_mean_scores = np.array(shuffled_mean_scores)
        shuffled_max_scores = np.array(shuffled_max_scores)
        
        # P-value: proportion of shuffles with mean score >= real score
        p_value = np.mean(shuffled_mean_scores >= real_mean_score)
        p_value_max = np.mean(shuffled_max_scores >= real_max_score)
        
        # Z-score
        if np.std(shuffled_mean_scores) > 0:
            z_score = (real_mean_score - np.mean(shuffled_mean_scores)) / np.std(shuffled_mean_scores)
        else:
            z_score = 0.0
        
        # Cohen's d effect size
        if np.std(shuffled_mean_scores) > 0:
            effect_size = (real_mean_score - np.mean(shuffled_mean_scores)) / np.std(shuffled_mean_scores)
        else:
            effect_size = 0.0
        
        is_significant = p_value < 0.05
        
        # Create summary
        summary_lines = [
            f"\n{'='*60}",
            "RANDOM CONTROL TEST RESULTS",
            f"{'='*60}",
            f"Source neuron: {source}",
            f"Source dataset: {source_dataset}",
            f"Target dataset: {target_dataset}",
            f"Number of shuffles: {n_shuffles}",
            f"Metric used: {score_col}",
            "",
            "SCORES:",
            f"  Real profile mean {score_col}: {real_mean_score:.4f}",
            f"  Real profile max {score_col}: {real_max_score:.4f}",
            f"  Shuffled mean (avg): {np.mean(shuffled_mean_scores):.4f} ± {np.std(shuffled_mean_scores):.4f}",
            f"  Shuffled max (avg): {np.mean(shuffled_max_scores):.4f} ± {np.std(shuffled_max_scores):.4f}",
            "",
            "STATISTICS:",
            f"  P-value (mean): {p_value:.4f} {'*' if p_value < 0.05 else ''} {'**' if p_value < 0.01 else ''} {'***' if p_value < 0.001 else ''}",
            f"  P-value (max): {p_value_max:.4f}",
            f"  Z-score: {z_score:.4f}",
            f"  Effect size (Cohen's d): {effect_size:.4f}",
            "",
            "INTERPRETATION:",
        ]
        
        if is_significant:
            if effect_size > 0.8:
                summary_lines.append("  ✓ SIGNIFICANT with LARGE effect size")
                summary_lines.append("  → Homolog finding results are highly meaningful")
            elif effect_size > 0.5:
                summary_lines.append("  ✓ SIGNIFICANT with MEDIUM effect size")
                summary_lines.append("  → Homolog finding results are meaningful")
            else:
                summary_lines.append("  ✓ SIGNIFICANT with SMALL effect size")
                summary_lines.append("  → Homolog finding results may be meaningful, interpret with caution")
        else:
            summary_lines.append("  ✗ NOT SIGNIFICANT")
            summary_lines.append("  → Homolog finding results may be due to chance")
            summary_lines.append("  → Consider using different parameters or more shuffles")
        
        summary_lines.append(f"{'='*60}")
        summary = "\n".join(summary_lines)
        
        self._log(summary)
        
        # 5. Save results if output_dir specified
        results = {
            'real_results': real_results,
            'shuffled_results': shuffled_results_list,
            'real_mean_score': float(real_mean_score),
            'real_max_score': float(real_max_score),
            'shuffled_mean_scores': list(shuffled_mean_scores),
            'shuffled_max_scores': list(shuffled_max_scores),
            'p_value': float(p_value),
            'p_value_max': float(p_value_max),
            'z_score': float(z_score),
            'effect_size': float(effect_size),
            'is_significant': bool(is_significant),
            'summary': summary,
            'n_shuffles': n_shuffles,
            'score_metric': score_col,
        }
        
        if output_dir:
            output_path = Path(output_dir)
            if saveas:
                output_path = output_path / saveas
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = output_path / f"control_test_{source}_{timestamp}"
            
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save real results
            real_results.to_csv(output_path / 'real_results.csv', index=False)
            
            # Save statistics
            stats_df = pd.DataFrame({
                'metric': ['real_mean_score', 'real_max_score', 'shuffled_mean_avg', 
                          'shuffled_mean_std', 'p_value', 'p_value_max', 'z_score', 
                          'effect_size', 'is_significant', 'n_shuffles'],
                'value': [real_mean_score, real_max_score, np.mean(shuffled_mean_scores),
                         np.std(shuffled_mean_scores), p_value, p_value_max, z_score,
                         effect_size, is_significant, n_shuffles]
            })
            stats_df.to_csv(output_path / 'control_test_stats.csv', index=False)
            
            # Save shuffled score distribution
            shuffle_dist_df = pd.DataFrame({
                'iteration': range(n_shuffles),
                'mean_score': shuffled_mean_scores,
                'max_score': shuffled_max_scores
            })
            shuffle_dist_df.to_csv(output_path / 'shuffled_score_distribution.csv', index=False)
            
            # Save summary
            with open(output_path / 'summary.txt', 'w') as f:
                f.write(summary)
            
            self._log(f"\nResults saved to: {output_path}")
            results['output_path'] = str(output_path)
        
        return results
    
    def _run_homolog_with_profile(
        self,
        profile: 'ConnectivityProfile',
        target_dataset: str,
        top_n: int
    ) -> pd.DataFrame:
        """
        Run homolog finding using a pre-built profile (for shuffle testing).
        
        This is an internal method that compares a given profile against
        all types in the target dataset.
        
        Args:
            profile: The connectivity profile to compare
            target_dataset: Dataset to search for homologs
            top_n: Number of top results to return
        
        Returns:
            DataFrame with homolog results
        """
        # Get target types from the target dataset
        target_types = self._get_type_pool(target_dataset)
        if not target_types:
            return pd.DataFrame()
        
        results = []
        
        for target_type in target_types:
            try:
                target_profile = self.profiler.get_profile(target_type, target_dataset)
                if target_profile is None:
                    continue
                
                # Compare profiles
                comparison = ProfileComparator.compare_profiles_simple(
                    profile, target_profile, direction='both'
                )
                
                results.append({
                    'target_type': target_type,
                    'target_dataset': target_dataset,
                    'rank_corr': comparison.get('rank_correlation', 0.0),
                    'jaccard': comparison.get('jaccard', 0.0),
                    'adjacency_score': comparison.get('shared_count', 0),
                })
            except Exception:
                continue
        
        if not results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        
        # Normalize rank correlation to 0-1 scale
        if 'rank_corr' in results_df.columns:
            results_df['rank_corr'] = (results_df['rank_corr'] + 1) / 2
        
        # Sort by rank_corr and return top_n
        results_df = results_df.sort_values('rank_corr', ascending=False).head(top_n)
        
        return results_df
