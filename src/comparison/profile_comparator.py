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
from typing import Dict, List, Optional, Tuple, Union
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


# ============================================================================
# Default Score Weights
# ============================================================================

DEFAULT_SCORE_WEIGHTS = {
    'jaccard': 0.30,
    'cosine': 0.35,
    'rank': 0.35
    # Note: overlap removed (highly correlated with Jaccard)
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
    weak_connectivity_a: bool = False
    weak_connectivity_b: bool = False
    
    @property
    def rank_correlation_norm(self) -> float:
        """Normalized rank correlation in [0, 1] range using (x+1)/2."""
        if np.isnan(self.rank_correlation):
            return np.nan
        return (self.rank_correlation + 1) / 2
    
    def to_dict(self) -> dict:
        """Convert to dictionary for CSV/JSON export."""
        return {
            'profile_a': self.profile_a_id,
            'profile_b': self.profile_b_id,
            'dataset_a': self.dataset_a,
            'dataset_b': self.dataset_b,
            'direction': self.direction,
            'jaccard': round(self.jaccard, 4),
            'cosine': round(self.cosine, 4),
            'rank_correlation': round(self.rank_correlation, 4),
            'rank_correlation_norm': round(self.rank_correlation_norm, 4) if not np.isnan(self.rank_correlation_norm) else np.nan,
            'overlap_a_in_b': round(self.overlap_a_in_b, 4),
            'overlap_b_in_a': round(self.overlap_b_in_a, 4),
            'combined': round(self.combined, 4),
            'confidence': self.confidence,
            'weak_connectivity_warning': self.weak_connectivity_a or self.weak_connectivity_b
        }
    
    def summary(self) -> str:
        """Generate a human-readable summary string."""
        lines = [
            f"Comparison: {self.profile_a_id} ({self.dataset_a}) vs {self.profile_b_id} ({self.dataset_b})",
            f"  Direction: {self.direction}",
            f"  Combined Score: {self.combined:.4f} ({self.confidence})",
            f"  Metrics: Jaccard={self.jaccard:.3f}, Cosine={self.cosine:.3f}, RankCorr={self.rank_correlation:.3f} (norm={self.rank_correlation_norm:.3f})",
            f"  Overlap: A_in_B={self.overlap_a_in_b:.3f}, B_in_A={self.overlap_b_in_a:.3f}"
        ]
        if self.weak_connectivity_a or self.weak_connectivity_b:
            lines.append("  ⚠️ Weak connectivity warning")
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
            all_list = sorted(all_partners)
            rank_array_a = [ranks_a.get(p, default_rank_a) for p in all_list]
            rank_array_b = [ranks_b.get(p, default_rank_b) for p in all_list]
        else:
            # Shared partners mode with expansion if needed
            shared = set(ranks_a.keys()) & set(ranks_b.keys())
            all_partners = set(ranks_a.keys()) | set(ranks_b.keys())
            
            min_partners = 5  # Minimum partners needed for meaningful correlation
            
            if len(shared) >= min_partners:
                # Enough shared partners, use them directly
                partner_list = sorted(shared)
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
        
        # Compute correlation
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
        Weighted combination of multiple metrics.
        
        Default weights:
        - jaccard: 0.25
        - cosine: 0.35
        - rank: 0.25
        - overlap: 0.15
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            weights: Custom weight dictionary (optional)
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            Dict with 'combined', 'jaccard', 'cosine', 'rank', 'rank_norm', 'overlap' scores
            - 'rank': Original rank correlation in [-1, 1] range
            - 'rank_norm': Normalized rank correlation in [0, 1] range using (x+1)/2
        """
        if weights is None:
            weights = DEFAULT_SCORE_WEIGHTS
        
        # Compute individual metrics
        jaccard = ProfileComparator.jaccard_similarity(profile_a, profile_b, direction)
        cosine = ProfileComparator.weighted_cosine_similarity(profile_a, profile_b, direction)
        rank_corr = ProfileComparator.rank_correlation(profile_a, profile_b, direction)
        overlap_a, overlap_b = ProfileComparator.overlap_fraction(profile_a, profile_b, direction)
        
        # Normalize rank_corr from [-1, 1] to [0, 1] for display
        rank_norm = (rank_corr + 1) / 2 if not np.isnan(rank_corr) else np.nan
        
        # Compute weighted combined score
        # Use normalized rank_corr for consistency (in [0, 1] range)
        rank_for_combined = 0 if np.isnan(rank_norm) else rank_norm
        combined = (
            weights.get('jaccard', 0.30) * jaccard +
            weights.get('cosine', 0.35) * cosine +
            weights.get('rank', 0.35) * rank_for_combined
        )
        
        # Keep overlap values for backward compatibility but don't use in combined
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
    def compare_profiles(
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both',
        weights: Optional[Dict[str, float]] = None
    ) -> ComparisonResult:
        """
        Full comparison between two profiles.
        
        Computes all metrics and returns a ComparisonResult object.
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            direction: 'upstream', 'downstream', or 'both'
            weights: Custom score weights (optional)
        
        Returns:
            ComparisonResult with all metrics and confidence level
        """
        scores = ProfileComparator.combined_score(profile_a, profile_b, weights=weights, direction=direction)
        
        confidence = ComparisonResult.determine_confidence(scores['combined'])
        
        return ComparisonResult(
            profile_a_id=str(profile_a.neuron_id),
            profile_b_id=str(profile_b.neuron_id),
            dataset_a=profile_a.dataset,
            dataset_b=profile_b.dataset,
            direction=direction,
            jaccard=scores['jaccard'],
            cosine=scores['cosine'],
            rank_correlation=scores['rank'],
            overlap_a_in_b=scores['overlap_a_in_b'],
            overlap_b_in_a=scores['overlap_b_in_a'],
            combined=scores['combined'],
            confidence=confidence,
            weak_connectivity_a=profile_a.is_weak_connectivity,
            weak_connectivity_b=profile_b.is_weak_connectivity
        )
    
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
        for partner in sorted(all_partners):
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
    - Configurable profiling parameters
    
    Example:
        >>> from src.comparison import HomologFinder
        >>> 
        >>> # Initialize finder (no datasets required upfront)
        >>> finder = HomologFinder()
        >>> 
        >>> # Find homologs of Mi1 in FAFB
        >>> results = finder.find_homologs(
        ...     query='Mi1',
        ...     source_dataset='hemibrain_v1_2_1',
        ...     target_datasets='flywire_FAFB_v783'
        ... )
        >>> 
        >>> # Find novel homologs within hemibrain
        >>> novel = finder.find_homologs(
        ...     query='Mi1',
        ...     source_dataset='hemibrain_v1_2_1',
        ...     target_datasets='hemibrain_v1_2_1'
        ... )
    """
    
    def __init__(
        self,
        top_k: int = 15,
        top_m: int = 5,
        min_synapse_threshold: int = 3,
        include_untyped_partners: bool = True,
        use_cache: bool = True,
        verbose: bool = True
    ):
        """
        Initialize HomologFinder with configuration.
        
        Args:
            top_k: Number of top partners to include (both upstream and downstream)
            top_m: Minimum unique partner types to ensure in profile
            min_synapse_threshold: Minimum synapse count for connections
            include_untyped_partners: Include partners without type annotations
            use_cache: Enable profile caching
            verbose: Print progress messages
        """
        from .connectivity_profiler import ConnectivityProfiler, ProfilerConfig
        
        self.verbose = verbose
        
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
        
        # Cache for profiles
        self._profile_cache: Dict[Tuple[str, str], 'ConnectivityProfile'] = {}
        
        # Cache for available types per dataset
        self._types_cache: Dict[str, List[str]] = {}
    
    def _log(self, msg: str):
        """Print message if verbose."""
        if self.verbose:
            print(f"[HomologFinder] {msg}")
    
    def get_profile(self, query: Union[str, int], dataset: str) -> Optional['ConnectivityProfile']:
        """Get connectivity profile for a neuron, with caching."""
        cache_key = (str(query), dataset)
        if cache_key not in self._profile_cache:
            profile = self.profiler.get_profile(query, dataset)
            if profile is not None:
                self._profile_cache[cache_key] = profile
        return self._profile_cache.get(cache_key)
    
    def get_available_types(self, dataset: str) -> List[str]:
        """Get available types in a dataset, with caching."""
        if dataset not in self._types_cache:
            types = self.profiler.get_available_types(dataset)
            self._types_cache[dataset] = types or []
        return self._types_cache[dataset]
    
    def find_homologs(
        self,
        query: Union[str, int],
        source_dataset: str,
        target_datasets: Union[str, List[str]],
        top_n: int = 20,
        metric: str = 'combined',
        direction: str = 'both',
        min_score: float = 0.0
    ) -> pd.DataFrame:
        """
        Find homologs of a neuron in target datasets.
        
        Args:
            query: Neuron identifier - bodyId (int) or type name (str)
            source_dataset: Dataset containing the query neuron
            target_datasets: Dataset(s) to search for homologs
                - Single: 'flywire_FAFB_v783'
                - Multiple: ['flywire_FAFB_v783', 'male-cns_v0_9']
                - Same as source: finds novel homologs within same dataset
            top_n: Number of top candidates per target dataset
            metric: Similarity metric ('combined', 'jaccard', 'cosine', 'rank')
            direction: Connection direction ('upstream', 'downstream', 'both')
            min_score: Minimum similarity score to include
        
        Returns:
            DataFrame with columns:
                - target_dataset: Dataset where candidate was found
                - target_type: Type name of candidate
                - similarity: Similarity score
                - jaccard: Jaccard similarity (if metric='combined')
                - cosine: Cosine similarity (if metric='combined')
                - rank_corr: Rank correlation (if metric='combined')
                - is_same_type: True if type name matches query
                - is_same_dataset: True if target = source dataset
        """
        # Normalize target_datasets
        if isinstance(target_datasets, str):
            target_datasets = [target_datasets]
        
        # Get source profile
        is_bodyid = isinstance(query, int) or (isinstance(query, str) and query.isdigit())
        query_label = f"bodyId:{query}" if is_bodyid else query
        
        self._log(f"Finding homologs for {query_label} from {source_dataset}")
        
        source_profile = self.get_profile(query, source_dataset)
        if source_profile is None:
            self._log(f"ERROR: Could not find '{query}' in {source_dataset}")
            return pd.DataFrame()
        
        if source_profile.is_weak_connectivity:
            self._log(f"WARNING: Source profile has weak connectivity ({source_profile.actual_upstream_count} up, {source_profile.actual_downstream_count} down)")
        
        source_type = str(source_profile.neuron_id)
        
        # Search each target dataset
        all_results = []
        
        # Collect all target types first to compute total for progress bar
        all_target_types = []
        for target_dataset in target_datasets:
            target_types = self.get_available_types(target_dataset)
            if target_types:
                for t in target_types:
                    all_target_types.append((target_dataset, t))
        
        if not all_target_types:
            self._log("No target types found in any dataset")
            return pd.DataFrame()
        
        self._log(f"Comparing against {len(all_target_types)} types across {len(target_datasets)} dataset(s)")
        
        # Progress bar for comparing types
        weak_connectivity_skipped = 0
        for target_dataset, target_type in tqdm(
            all_target_types,
            desc=f"Finding homologs",
            disable=not self.verbose,
            leave=False
        ):
            is_same_dataset = (target_dataset == source_dataset)
            
            # Skip self-comparison
            if is_same_dataset and target_type == source_type:
                continue
            
            try:
                target_profile = self.get_profile(target_type, target_dataset)
                if target_profile is None:
                    continue
                if target_profile.is_weak_connectivity:
                    weak_connectivity_skipped += 1
                    continue
                
                # Compute similarity
                if metric == 'combined':
                    scores = ProfileComparator.combined_score(
                        source_profile, target_profile, direction=direction
                    )
                    sim = scores['combined']
                    result = {
                        'target_dataset': target_dataset,
                        'target_type': target_type,
                        'similarity': sim,
                        'jaccard': scores['jaccard'],
                        'cosine': scores['cosine'],
                        'rank_corr': scores['rank'],
                        'is_same_type': target_type == source_type,
                        'is_same_dataset': is_same_dataset
                    }
                else:
                    if metric == 'jaccard':
                        sim = ProfileComparator.jaccard_similarity(
                            source_profile, target_profile, direction
                        )
                    elif metric == 'cosine':
                        sim = ProfileComparator.weighted_cosine_similarity(
                            source_profile, target_profile, direction
                        )
                    else:  # rank
                        sim = ProfileComparator.rank_correlation(
                            source_profile, target_profile, direction
                        )
                    
                    result = {
                        'target_dataset': target_dataset,
                        'target_type': target_type,
                        'similarity': sim,
                        'is_same_type': target_type == source_type,
                        'is_same_dataset': is_same_dataset
                    }
                
                if not pd.isna(sim) and sim >= min_score:
                    all_results.append(result)
                    
            except Exception:
                pass
        
        if weak_connectivity_skipped > 0:
            self._log(f"Skipped {weak_connectivity_skipped} types with weak connectivity")
        
        if not all_results:
            self._log("No candidates found")
            return pd.DataFrame()
        
        # Create and sort results
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('similarity', ascending=False)
        
        # Get top N per dataset
        if top_n > 0:
            results_df = results_df.groupby('target_dataset').head(top_n).reset_index(drop=True)
        
        self._log(f"Found {len(results_df)} candidates")
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
        
        Shortcut for find_homologs() with target_datasets = source_dataset.
        
        Args:
            query: Neuron identifier
            dataset: Dataset to search
            top_n: Number of top candidates
            min_score: Minimum similarity score
        
        Returns:
            DataFrame with similar types (excluding self)
        """
        return self.find_homologs(
            query=query,
            source_dataset=dataset,
            target_datasets=dataset,
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
    # Round 6: Fast Homolog Finding with Connection Cache
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
    
    def _load_connection_cache(self, dataset: str) -> Optional[pd.DataFrame]:
        """
        Load complete connections from existing cache with type information.
        
        Cache is built by FindNeuronConnection with threshold=1 (BANC=3).
        Joins with neuron data to get type information.
        
        Type mapping priority:
        1. datasets/{dataset}/neuron_df.parquet (authoritative source)
        2. cache/{dataset}/neuron_index.parquet (fallback)
        
        Returns:
            DataFrame with columns: bodyId_pre, bodyId_post, weight, type_pre, type_post
            or None if cache not available
        """
        cache_path = self._get_connection_cache_path(dataset)
        index_path = self._get_neuron_index_path(dataset)
        
        if not cache_path.exists():
            self._log(f"Connection cache not found: {cache_path}")
            return None
        
        # Load connections
        conn_df = pd.read_parquet(cache_path)
        
        # Try to load type mapping from datasets folder first (authoritative)
        from pathlib import Path
        safe_name = dataset.replace(':', '_').replace('.', '_')
        src_dir = Path(__file__).parent.parent
        project_root = src_dir.parent
        datasets_neuron_path = project_root / 'datasets' / safe_name / f'{safe_name}_allneurons_neuron_df.parquet'
        
        type_map = None
        if datasets_neuron_path.exists():
            neuron_df = pd.read_parquet(datasets_neuron_path)
            if 'type' in neuron_df.columns:
                type_map = neuron_df.set_index('bodyId')['type'].to_dict()
        
        # Fallback to cache neuron_index
        if type_map is None and index_path.exists():
            neuron_df = pd.read_parquet(index_path)
            type_map = neuron_df.set_index('bodyId')['type'].to_dict()
        
        if type_map is None:
            self._log(f"No neuron type mapping available for {dataset}")
            return None
        
        # Add type columns
        conn_df['type_pre'] = conn_df['bodyId_pre'].map(type_map)
        conn_df['type_post'] = conn_df['bodyId_post'].map(type_map)
        
        self._log(f"Loaded {len(conn_df):,} connections from cache")
        
        return conn_df
    
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
        
        # Aggregate upstream: for each post-type, sum weights by pre-type
        # upstream_by_type[target_type] = {source_type: total_weight}
        upstream_agg = typed.groupby(['type_post', 'type_pre'])['weight'].sum()
        upstream_by_type: Dict[str, Dict[str, float]] = {}
        for (post_type, pre_type), weight in upstream_agg.items():
            if post_type not in upstream_by_type:
                upstream_by_type[post_type] = {}
            upstream_by_type[post_type][pre_type] = weight
        
        # Aggregate downstream: for each pre-type, sum weights by post-type
        # downstream_by_type[source_type] = {target_type: total_weight}
        downstream_agg = typed.groupby(['type_pre', 'type_post'])['weight'].sum()
        downstream_by_type: Dict[str, Dict[str, float]] = {}
        for (pre_type, post_type), weight in downstream_agg.items():
            if pre_type not in downstream_by_type:
                downstream_by_type[pre_type] = {}
            downstream_by_type[pre_type][post_type] = weight
        
        if show_progress and self.verbose:
            self._log(f"Aggregated {len(upstream_by_type):,} types with upstream, "
                      f"{len(downstream_by_type):,} with downstream connections")
        
        return upstream_by_type, downstream_by_type
    
    def _build_profile_from_aggregates(
        self,
        neuron_type: str,
        dataset: str,
        upstream_by_type: Dict[str, Dict[str, float]],
        downstream_by_type: Dict[str, Dict[str, float]]
    ) -> Optional['ConnectivityProfile']:
        """
        Build a ConnectivityProfile from pre-aggregated type dictionaries.
        
        This is O(1) lookup instead of O(N) filtering on the full DataFrame.
        
        Args:
            neuron_type: Type name
            dataset: Dataset name
            upstream_by_type: Pre-aggregated upstream connections
            downstream_by_type: Pre-aggregated downstream connections
            
        Returns:
            ConnectivityProfile or None if no connections found
        """
        from .connectivity_profiler import ConnectivityProfile
        
        upstream_partners = upstream_by_type.get(neuron_type, {})
        downstream_partners = downstream_by_type.get(neuron_type, {})
        
        if not upstream_partners and not downstream_partners:
            return None
        
        # Normalize weights to proportions
        up_total = sum(upstream_partners.values()) if upstream_partners else 1.0
        down_total = sum(downstream_partners.values()) if downstream_partners else 1.0
        
        upstream_norm = {k: v / up_total for k, v in upstream_partners.items()}
        downstream_norm = {k: v / down_total for k, v in downstream_partners.items()}
        
        # Create ranks
        upstream_ranked = sorted(upstream_partners.items(), key=lambda x: -x[1])
        downstream_ranked = sorted(downstream_partners.items(), key=lambda x: -x[1])
        
        upstream_ranks = {k: i+1 for i, (k, _) in enumerate(upstream_ranked)}
        downstream_ranks = {k: i+1 for i, (k, _) in enumerate(downstream_ranked)}
        
        # Determine if weak connectivity
        is_weak = len(upstream_partners) < 5 or len(downstream_partners) < 5
        
        return ConnectivityProfile(
            neuron_id=neuron_type,
            dataset=dataset,
            upstream_partners=upstream_norm,
            downstream_partners=downstream_norm,
            upstream_ranks=upstream_ranks,
            downstream_ranks=downstream_ranks,
            upstream_top_k=len(upstream_partners),
            downstream_top_k=len(downstream_partners),
            total_upstream_weight=up_total,
            total_downstream_weight=down_total,
            actual_upstream_count=len(upstream_partners),
            actual_downstream_count=len(downstream_partners),
            is_weak_connectivity=is_weak,
            unique_types_upstream=len(upstream_partners),
            unique_types_downstream=len(downstream_partners),
        )
    
    def _get_typed_connections_from_cache(
        self,
        conn_df: pd.DataFrame,
        neuron: Union[str, int],
        min_weight: int = 1
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get upstream and downstream connections for a neuron from cached data.
        
        Args:
            conn_df: Full connection DataFrame with type columns
            neuron: Type name (str) or bodyId (int)
            min_weight: Minimum synapse weight
        
        Returns:
            Tuple of (upstream_df, downstream_df)
        """
        if isinstance(neuron, str):
            # Type-based query
            mask_up = conn_df['type_post'] == neuron
            mask_down = conn_df['type_pre'] == neuron
        else:
            # BodyId query
            mask_up = conn_df['bodyId_post'] == neuron
            mask_down = conn_df['bodyId_pre'] == neuron
        
        upstream = conn_df[mask_up & (conn_df['weight'] >= min_weight)].copy()
        downstream = conn_df[mask_down & (conn_df['weight'] >= min_weight)].copy()
        
        return upstream, downstream
    
    def _build_2hop_profile(
        self,
        conn_df: pd.DataFrame,
        neuron: Union[str, int],
        min_weight: int = 1,
        expand_untyped: bool = False,
        max_untyped_expand: int = 10
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Build connectivity profile from cached connections.
        
        For typed partners: uses direct connection weight
        For untyped partners (optional): expands 1 level to their typed partners
        
        Args:
            conn_df: Full connection DataFrame with type columns
            neuron: Type name or bodyId
            min_weight: Minimum synapse weight
            expand_untyped: If True, expand through untyped partners (slower)
            max_untyped_expand: Max untyped partners to expand (for performance)
        
        Returns:
            Tuple of (upstream_partners, downstream_partners) as {type: weight}
        """
        upstream, downstream = self._get_typed_connections_from_cache(conn_df, neuron, min_weight)
        
        def process_direction(df: pd.DataFrame, partner_type_col: str, 
                            partner_id_col: str, expand_direction: str) -> Dict[str, float]:
            """Process one direction."""
            partner_weights: Dict[str, float] = {}
            
            if df.empty:
                return partner_weights
            
            # Separate typed and untyped partners
            typed_mask = df[partner_type_col].notna() & (df[partner_type_col].astype(str).str.strip() != '')
            typed_df = df[typed_mask]
            untyped_df = df[~typed_mask]
            
            # Direct typed partners - aggregate by type (vectorized)
            if not typed_df.empty:
                typed_agg = typed_df.groupby(partner_type_col)['weight'].sum()
                partner_weights = typed_agg.to_dict()
            
            # Expand untyped partners (2-hop) - optional and limited
            if expand_untyped and not untyped_df.empty:
                # Only expand top N untyped by weight
                untyped_sorted = untyped_df.nlargest(max_untyped_expand, 'weight')
                untyped_bodyids = untyped_sorted[partner_id_col].tolist()
                untyped_weights = untyped_sorted.set_index(partner_id_col)['weight'].to_dict()
                
                # Batch query for all untyped partners
                if expand_direction == 'upstream':
                    exp_mask = conn_df['bodyId_post'].isin(untyped_bodyids) & \
                               (conn_df['weight'] >= min_weight)
                    exp_type_col = 'type_pre'
                    exp_id_col = 'bodyId_post'
                else:
                    exp_mask = conn_df['bodyId_pre'].isin(untyped_bodyids) & \
                               (conn_df['weight'] >= min_weight)
                    exp_type_col = 'type_post'
                    exp_id_col = 'bodyId_pre'
                
                exp_df = conn_df[exp_mask]
                exp_typed = exp_df[exp_df[exp_type_col].notna() & 
                                   (exp_df[exp_type_col].astype(str).str.strip() != '')]
                
                if not exp_typed.empty:
                    # Group by (untyped_bodyid, typed_partner) and aggregate
                    for untyped_id in untyped_bodyids:
                        partner_exp = exp_typed[exp_typed[exp_id_col] == untyped_id]
                        if partner_exp.empty:
                            continue
                        
                        weight_through = untyped_weights.get(untyped_id, 0)
                        total_exp = partner_exp['weight'].sum()
                        
                        for ptype, pweight in partner_exp.groupby(exp_type_col)['weight'].sum().items():
                            contribution = weight_through * (pweight / total_exp)
                            partner_weights[ptype] = partner_weights.get(ptype, 0) + contribution
            
            return partner_weights
        
        upstream_partners = process_direction(upstream, 'type_pre', 'bodyId_pre', 'upstream')
        downstream_partners = process_direction(downstream, 'type_post', 'bodyId_post', 'downstream')
        
        return upstream_partners, downstream_partners
    
    def _build_profile_from_cache(
        self,
        conn_df: pd.DataFrame,
        neuron: Union[str, int],
        dataset: str,
        min_weight: int = 1
    ) -> Optional['ConnectivityProfile']:
        """
        Build a ConnectivityProfile from cached connection data.
        
        This creates a profile without API calls, using the local cache.
        Used by find_homologs_fast() for candidate comparison.
        
        Args:
            conn_df: Connection DataFrame from cache
            neuron: Type name or bodyId
            dataset: Dataset name
            min_weight: Minimum synapse weight
            
        Returns:
            ConnectivityProfile or None if no connections found
        """
        from .connectivity_profiler import ConnectivityProfile
        
        upstream_partners, downstream_partners = self._build_2hop_profile(
            conn_df, neuron, min_weight
        )
        
        if not upstream_partners and not downstream_partners:
            return None
        
        # Normalize weights to proportions
        up_total = sum(upstream_partners.values()) if upstream_partners else 1.0
        down_total = sum(downstream_partners.values()) if downstream_partners else 1.0
        
        upstream_norm = {k: v / up_total for k, v in upstream_partners.items()}
        downstream_norm = {k: v / down_total for k, v in downstream_partners.items()}
        
        # Create ranks
        upstream_ranked = sorted(upstream_partners.items(), key=lambda x: -x[1])
        downstream_ranked = sorted(downstream_partners.items(), key=lambda x: -x[1])
        
        upstream_ranks = {k: i+1 for i, (k, _) in enumerate(upstream_ranked)}
        downstream_ranks = {k: i+1 for i, (k, _) in enumerate(downstream_ranked)}
        
        # Determine if weak connectivity
        is_weak = len(upstream_partners) < 5 or len(downstream_partners) < 5
        
        return ConnectivityProfile(
            neuron_id=neuron,
            dataset=dataset,
            upstream_partners=upstream_norm,
            downstream_partners=downstream_norm,
            upstream_ranks=upstream_ranks,
            downstream_ranks=downstream_ranks,
            upstream_top_k=len(upstream_partners),
            downstream_top_k=len(downstream_partners),
            total_upstream_weight=up_total,
            total_downstream_weight=down_total,
            actual_upstream_count=len(upstream_partners),
            actual_downstream_count=len(downstream_partners),
            is_weak_connectivity=is_weak,
            unique_types_upstream=len(upstream_partners),
            unique_types_downstream=len(downstream_partners),
        )
    
    def find_homologs_fast(
        self,
        query: Union[str, int],
        source_dataset: str,
        target_dataset: str,
        top_n_candidates: int = 50,
        min_shared_partners: int = 2,
        min_weight: int = 3,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Fast homolog discovery via adjacency expansion.
        
        This method is much faster than find_homologs() because it:
        1. Uses local connection cache instead of neuprint queries
        2. Pre-aggregates connections by type for O(1) lookups
        3. Uses adjacency expansion to find candidates (not brute force)
        4. Only runs full profile comparison on top 50 candidates
        
        Algorithm:
        1. Pre-aggregate connection data into type-level dictionaries
        2. Build profile for query from aggregates
        3. Find candidate types by adjacency expansion
        4. Run full profile comparison on top 50 candidates
        
        Args:
            query: Neuron type name (str) - bodyId not supported in fast mode
            source_dataset: Source dataset
            target_dataset: Target dataset to search
            top_n_candidates: Max candidates for full comparison (default: 50)
            min_shared_partners: Minimum shared partners to be a candidate (default: 2)
            min_weight: Minimum synapse weight (default: 3)
            show_progress: Show progress bar
        
        Returns:
            DataFrame with columns:
                - target_type: Candidate type name
                - target_dataset: Target dataset
                - shared_partner_count: Number of shared partners (pre-filtering score)
                - similarity: Combined similarity score
                - jaccard, cosine, rank_corr: Individual metrics
                - is_same_type: True if type matches query
        
        Note:
            If fewer than top_n_candidates found, a warning suggests
            running find_homologs() for comprehensive search.
        """
        # Fast mode only supports type queries, not bodyId
        if isinstance(query, int):
            self._log("WARNING: find_homologs_fast() only supports type queries.")
            self._log("For bodyId queries, use find_homologs() instead.")
            return pd.DataFrame()
        
        query_type = str(query)
        is_cross_dataset = (source_dataset != target_dataset)
        
        self._log(f"Fast homolog search: {query_type} from {source_dataset} → {target_dataset}")
        
        # Step 1: Load and pre-aggregate connection data
        with tqdm(total=4, desc="Loading data", disable=not show_progress or not self.verbose, leave=False) as pbar:
            # Load source cache
            source_conn = self._load_connection_cache(source_dataset)
            if source_conn is None:
                self._log(f"ERROR: Connection cache not available for {source_dataset}")
                self._log("Build with: FindNeuronConnection(dataset, source=None, target=None, max_interlayer=0)")
                return pd.DataFrame()
            pbar.update(1)
            pbar.set_description("Aggregating source")
            
            # Pre-aggregate source
            source_up, source_down = self._build_type_aggregates(source_conn, min_weight, show_progress=False)
            pbar.update(1)
            
            # Load and aggregate target if cross-dataset
            if is_cross_dataset:
                pbar.set_description("Loading target")
                target_conn = self._load_connection_cache(target_dataset)
                if target_conn is None:
                    self._log(f"ERROR: Connection cache not available for {target_dataset}")
                    return pd.DataFrame()
                pbar.update(1)
                pbar.set_description("Aggregating target")
                target_up, target_down = self._build_type_aggregates(target_conn, min_weight, show_progress=False)
                pbar.update(1)
            else:
                target_up, target_down = source_up, source_down
                pbar.update(2)
        
        # Step 2: Build source profile
        source_profile = self._build_profile_from_aggregates(query_type, source_dataset, source_up, source_down)
        if source_profile is None:
            self._log(f"ERROR: No connections found for {query_type}")
            return pd.DataFrame()
        
        upstream_types = set(source_profile.upstream_partners.keys())
        downstream_types = set(source_profile.downstream_partners.keys())
        
        self._log(f"Source profile: {len(upstream_types)} upstream, {len(downstream_types)} downstream types")
        
        # Step 3: Find candidates via adjacency expansion (using aggregates)
        candidate_scores: Dict[str, int] = {}
        
        # Types that share upstream partners (receive from same types)
        for up_type in upstream_types:
            # Find all types that also receive from up_type
            for target_type in target_down.get(up_type, {}).keys():
                if target_type and target_type != query_type:
                    candidate_scores[target_type] = candidate_scores.get(target_type, 0) + 1
        
        # Types that share downstream partners (send to same types)
        for down_type in downstream_types:
            # Find all types that also send to down_type
            for target_type in target_up.get(down_type, {}).keys():
                if target_type and target_type != query_type:
                    candidate_scores[target_type] = candidate_scores.get(target_type, 0) + 1
        
        # Filter by minimum shared partners
        candidate_scores = {k: v for k, v in candidate_scores.items() if v >= min_shared_partners}
        
        # Step 4: Take top N candidates
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])[:top_n_candidates]
        
        if len(sorted_candidates) < top_n_candidates:
            self._log(f"NOTE: Found {len(sorted_candidates)} candidates (< {top_n_candidates}). "
                      f"Consider find_homologs() for comprehensive search.")
        
        if not sorted_candidates:
            self._log("No candidates found with sufficient shared partners")
            return pd.DataFrame()
        
        self._log(f"Comparing top {len(sorted_candidates)} candidates")
        
        # Step 5: Full profile comparison on candidates
        results = []
        weak_skipped = 0
        
        iterator = tqdm(
            sorted_candidates,
            desc="Comparing profiles",
            disable=not show_progress or not self.verbose,
            leave=False
        )
        
        for ctype, shared_count in iterator:
            iterator.set_postfix_str(ctype[:20])
            
            # Build profile from aggregates (O(1))
            target_profile = self._build_profile_from_aggregates(ctype, target_dataset, target_up, target_down)
            if target_profile is None:
                continue
            if target_profile.is_weak_connectivity:
                weak_skipped += 1
                continue
            
            scores = ProfileComparator.combined_score(source_profile, target_profile)
            
            # Normalize rank_corr from [-1,1] to [0,1]
            rank_corr_raw = scores['rank']
            rank_corr_norm = (rank_corr_raw + 1) / 2 if not np.isnan(rank_corr_raw) else np.nan
            
            results.append({
                'target_type': ctype,
                'target_dataset': target_dataset,
                'shared_partner_count': int(shared_count),
                'similarity': scores['combined'],
                'jaccard': scores['jaccard'],
                'cosine': scores['cosine'],
                'rank_corr': rank_corr_norm,  # Normalized to [0,1]
                'rank_corr_raw': rank_corr_raw,  # Original [-1,1]
                'is_same_type': ctype == query_type,
                'is_same_dataset': not is_cross_dataset
            })
        
        if weak_skipped > 0:
            self._log(f"Skipped {weak_skipped} types with weak connectivity")
        
        if not results:
            self._log("No valid candidates after profile comparison")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results).sort_values('similarity', ascending=False)
        self._log(f"Found {len(results_df)} matches")
        
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
        Find homologs within the same dataset, comparing by bodyId.
        
        For intra-dataset search, we can compare individual neurons (bodyIds)
        rather than aggregated types. This is useful for:
        - Finding untyped neurons similar to a typed query
        - Finding sub-populations within a type
        
        Args:
            query: Neuron type (str) or bodyId (int)
            dataset: Dataset to search
            search_untyped: If True, search untyped neurons; else search typed first
            top_n: Number of top results
            min_weight: Minimum synapse weight
            show_progress: Show progress bar
        
        Returns:
            DataFrame with matches including bodyId-level results
        """
        is_bodyid = isinstance(query, int)
        query_label = f"bodyId:{query}" if is_bodyid else query
        
        self._log(f"Intra-dataset search: {query_label} in {dataset}")
        
        # Load connection cache
        conn_df = self._load_connection_cache(dataset)
        if conn_df is None:
            self._log(f"ERROR: Connection cache not available for {dataset}")
            return pd.DataFrame()
        
        # Get query profile
        source_profile = self.get_profile(query, dataset)
        if source_profile is None:
            self._log(f"ERROR: Could not get profile for {query_label}")
            return pd.DataFrame()
        
        # Get list of neurons to compare
        neuron_index_path = self._get_neuron_index_path(dataset)
        if not neuron_index_path.exists():
            self._log("ERROR: Neuron index not found")
            return pd.DataFrame()
        
        neuron_df = pd.read_parquet(neuron_index_path)
        
        if search_untyped:
            # Search untyped neurons
            candidates = neuron_df[
                neuron_df['type'].isna() | (neuron_df['type'].astype(str).str.strip() == '')
            ]['bodyId'].tolist()
            self._log(f"Searching {len(candidates)} untyped neurons")
        else:
            # Search typed neurons first
            typed_neurons = neuron_df[
                neuron_df['type'].notna() & (neuron_df['type'].astype(str).str.strip() != '')
            ]
            # Get unique types
            candidates = typed_neurons['type'].unique().tolist()
            self._log(f"Searching {len(candidates)} typed neurons")
        
        # Exclude query from candidates
        query_str = str(query)
        candidates = [c for c in candidates if str(c) != query_str]
        
        # Compare against candidates
        results = []
        iterator = tqdm(
            candidates,
            desc="Comparing",
            disable=not show_progress or not self.verbose,
            leave=False
        )
        
        weak_skipped = 0
        for candidate in iterator:
            target_profile = self.get_profile(candidate, dataset)
            if target_profile is None:
                continue
            if target_profile.is_weak_connectivity:
                weak_skipped += 1
                continue
            
            scores = ProfileComparator.combined_score(source_profile, target_profile)
            
            # Get type info for bodyId candidates
            if search_untyped:
                candidate_type = None
                candidate_bodyid = candidate
            else:
                candidate_type = candidate
                candidate_bodyid = None
            
            results.append({
                'target': str(candidate),
                'target_type': candidate_type,
                'target_bodyId': candidate_bodyid,
                'target_dataset': dataset,
                'similarity': scores['combined'],
                'jaccard': scores['jaccard'],
                'cosine': scores['cosine'],
                'rank_corr': scores['rank']
            })
        
        if weak_skipped > 0:
            self._log(f"Skipped {weak_skipped} with weak connectivity")
        
        if not results:
            self._log("No matches found")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results).sort_values('similarity', ascending=False).head(top_n)
        self._log(f"Found {len(results_df)} matches")
        
        return results_df
