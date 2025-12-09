"""
Cross-Dataset Verification Module

This module provides tools for verifying neuron type assignments across datasets
by comparing connectivity profiles.

Main components:
- CrossDatasetVerifier: Main class for verification workflows
- VerificationResult: Dataclass storing verification results

Use Cases:
1. Verify that neurons with the same type label have similar connectivity
2. Find potential homologs for untyped neurons
3. Generate confidence scores for type assignments
4. Batch verification for multiple types

Example:
    >>> from src.comparison import CrossDatasetVerifier, ConnectivityProfiler
    >>> 
    >>> profiler = ConnectivityProfiler(datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'])
    >>> verifier = CrossDatasetVerifier(profiler)
    >>> 
    >>> results = verifier.verify_type_assignment(
    ...     'aMe12', datasets=['hemibrain:v1.2.1', 'male-cns:v0.9']
    ... )
    >>> print(results)
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd

from .connectivity_profiler import (
    ConnectivityProfile, 
    ConnectivityProfiler, 
    ProfilerConfig
)
from .profile_comparator import (
    ProfileComparator, 
    ComparisonResult,
    DEFAULT_SCORE_WEIGHTS
)


# ============================================================================
# Verification Result Dataclass
# ============================================================================

@dataclass
class VerificationResult:
    """
    Result of verifying a neuron type across datasets.
    
    Attributes:
        neuron_type: Type name being verified
        datasets: List of datasets compared
        pairwise_scores: List of ComparisonResult for each dataset pair
        avg_combined_score: Average combined score across all pairs
        min_score: Minimum combined score
        max_score: Maximum combined score
        confidence: Overall confidence level
        weak_connectivity_datasets: Datasets with weak connectivity profiles
        verification_status: 'verified', 'needs_review', 'questionable', 'failed'
    """
    neuron_type: str
    datasets: List[str]
    pairwise_scores: List[ComparisonResult]
    avg_combined_score: float
    min_score: float
    max_score: float
    confidence: str
    weak_connectivity_datasets: List[str] = field(default_factory=list)
    verification_status: str = 'verified'
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        # Handle NaN values for JSON serialization
        def safe_round(val, decimals=4):
            if np.isnan(val):
                return None  # JSON-safe representation of NaN
            return round(val, decimals)
        
        return {
            'neuron_type': self.neuron_type,
            'datasets': self.datasets,
            'num_pairs': len(self.pairwise_scores),
            'avg_combined_score': safe_round(self.avg_combined_score),
            'min_score': safe_round(self.min_score),
            'max_score': safe_round(self.max_score),
            'confidence': self.confidence,
            'weak_connectivity_datasets': self.weak_connectivity_datasets,
            'verification_status': self.verification_status,
            'pairwise_details': [s.to_dict() for s in self.pairwise_scores]
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Verification Result: {self.neuron_type}",
            f"  Datasets: {', '.join(self.datasets)}",
            f"  Status: {self.verification_status}",
            f"  Confidence: {self.confidence}",
        ]
        
        # Handle NaN scores
        if np.isnan(self.avg_combined_score):
            lines.append(f"  Combined Score: N/A (type not found in enough datasets)")
        else:
            lines.append(f"  Combined Score: {self.avg_combined_score:.3f} (range: {self.min_score:.3f}-{self.max_score:.3f})")
        
        if self.weak_connectivity_datasets:
            lines.append(f"  ⚠️ Weak connectivity in: {', '.join(self.weak_connectivity_datasets)}")
        
        return "\n".join(lines)


# ============================================================================
# Cross-Dataset Verifier Class
# ============================================================================

class CrossDatasetVerifier:
    """
    Verify neuron type consistency across datasets.
    
    This class orchestrates the extraction and comparison of connectivity
    profiles to verify that neurons labeled with the same type have
    similar connectivity patterns across datasets.
    
    Features:
    - Type assignment verification with confidence scores
    - Homolog discovery for untyped neurons
    - Batch verification for multiple types
    - Integration with ComparisonAnalyzer results
    - Report generation with visualizations
    - 2-hop profile expansion for high-untyped profiles
    
    Example:
        >>> verifier = CrossDatasetVerifier(profiler)
        >>> results = verifier.verify_type_assignment('aMe12', datasets)
        >>> print(results.summary())
    """
    
    def __init__(
        self,
        profiler: ConnectivityProfiler,
        comparator: Optional[ProfileComparator] = None,
        label_mapper: Optional[Any] = None,
        verbose: bool = True,
        comparison_mode: str = 'loose',
        min_common_partners: int = 3,
        score_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize CrossDatasetVerifier.
        
        Args:
            profiler: ConnectivityProfiler instance for extracting profiles
            comparator: ProfileComparator instance (created if not provided)
            label_mapper: Optional LabelMapper for cross-dataset name mapping
            verbose: Print progress messages
            comparison_mode: 'loose' (type-aggregated) or 'strict' (per-bodyId)
            min_common_partners: (strict mode) Minimum shared partners required
            score_weights: Custom weights for combined score {'jaccard': 0.3, 'cosine': 0.35, 'rank': 0.35}
        
        Note:
            - use_ranks parameter removed: Ranks are always used for bodyId-level comparison
            - max_untyped_fraction removed: Hybrid 1-hop/2-hop handles untyped automatically
            - allow_2hop_expansion removed: Controlled via profiler.config.expand_untyped_2hop
        """
        self.profiler = profiler
        self.comparator = comparator or ProfileComparator()
        self.label_mapper = label_mapper
        self.verbose = verbose
        self.comparison_mode = comparison_mode
        self.min_common_partners = min_common_partners
        self.score_weights = score_weights
        
        # Cache for profiles
        self._profile_cache: Dict[Tuple[str, str], ConnectivityProfile] = {}
    
    def _log(self, message: str):
        """Print message if verbose mode enabled."""
        if self.verbose:
            print(f"[CrossDatasetVerifier] {message}")
    
    def _get_mapped_type(self, neuron_type: str, dataset: str) -> str:
        """Get mapped type name for a dataset using label_mapper if available."""
        if self.label_mapper is None:
            return neuron_type
        
        # Use label_mapper to get dataset-specific type name
        try:
            mapped = self.label_mapper.get_mapped_label(neuron_type, dataset)
            return mapped if mapped else neuron_type
        except Exception:
            return neuron_type
    
    def _compare_profiles_strict(
        self,
        profile_a: ConnectivityProfile,
        profile_b: ConnectivityProfile,
        direction: str = 'both',
        score_weights: Optional[Dict[str, float]] = None
    ) -> ComparisonResult:
        """
        Compare profiles using strict mode (per-bodyId with rank correlation).
        
        This delegates to ProfileComparator.compare_profiles() which handles:
        1. Rank correlation between matching partner types (always uses ranks)
        2. Jaccard on typed partners only
        3. Combined score weighted by metrics
        
        Strict mode adds the min_common_partners check - if fewer common partners
        than threshold, rank correlation is treated as undefined.
        
        Args:
            profile_a: First connectivity profile
            profile_b: Second connectivity profile
            direction: 'upstream', 'downstream', or 'both'
            score_weights: Custom weights for combined score
        
        Returns:
            ComparisonResult with strict mode metrics
        """
        # Delegate to ProfileComparator for core comparison
        weights = score_weights or self.score_weights or DEFAULT_SCORE_WEIGHTS
        result = ProfileComparator.compare_profiles(profile_a, profile_b, direction, weights)
        
        # Check min_common_partners for strict mode
        min_common = self.min_common_partners
        directions = []
        if direction in ['both', 'upstream']:
            directions.append('upstream')
        if direction in ['both', 'downstream']:
            directions.append('downstream')
        
        # Count common partners across directions
        total_common = 0
        for dir_name in directions:
            if dir_name == 'upstream':
                partners_a = set(profile_a.upstream_partners.keys()) if profile_a.upstream_partners else set()
                partners_b = set(profile_b.upstream_partners.keys()) if profile_b.upstream_partners else set()
            else:
                partners_a = set(profile_a.downstream_partners.keys()) if profile_a.downstream_partners else set()
                partners_b = set(profile_b.downstream_partners.keys()) if profile_b.downstream_partners else set()
            total_common += len(partners_a & partners_b)
        
        # If insufficient common partners, mark rank as undefined
        if total_common < min_common:
            # Recompute with adjusted notes
            notes = result.notes or ""
            if notes:
                notes += "; "
            notes += f"Insufficient common partners ({total_common} < {min_common})"
            
            # Create new result with NaN rank and adjusted combined score
            result = ComparisonResult(
                profile_a_id=result.profile_a_id,
                profile_b_id=result.profile_b_id,
                dataset_a=result.dataset_a,
                dataset_b=result.dataset_b,
                direction=result.direction,
                jaccard=result.jaccard,
                cosine=result.cosine,
                rank_correlation=np.nan,
                overlap_a_in_b=result.overlap_a_in_b,
                overlap_b_in_a=result.overlap_b_in_a,
                combined=weights.get('jaccard', 0.50) * result.jaccard + weights.get('rank', 0.50) * 0.5,
                confidence=ComparisonResult.determine_confidence(
                    weights.get('jaccard', 0.50) * result.jaccard + weights.get('rank', 0.50) * 0.5
                ),
                weak_connectivity_a=result.weak_connectivity_a,
                weak_connectivity_b=result.weak_connectivity_b,
                notes=notes
            )
        
        return result


    def _get_profile(
        self,
        neuron: Union[str, int],
        dataset: str,
        force_refresh: bool = False
    ) -> ConnectivityProfile:
        """
        Get profile with caching.
        
        The profiler handles 2-hop expansion automatically via config.expand_untyped_2hop.
        If 2-hop expansion doesn't return any typed neuron in top_k, the untyped partner
        is ignored by the profiler.
        
        Args:
            neuron: Neuron identifier
            dataset: Dataset identifier
            force_refresh: Bypass cache
        
        Returns:
            ConnectivityProfile (with 2-hop expansion handled by profiler if enabled)
        """
        cache_key = (str(neuron), dataset)
        
        # Check cache first
        if not force_refresh and cache_key in self._profile_cache:
            return self._profile_cache[cache_key]
        
        # Get profile (profiler handles 2-hop expansion via config.expand_untyped_2hop)
        profile = self.profiler.get_profile(neuron, dataset, force_refresh)
        
        if profile is None:
            return None
        
        self._profile_cache[cache_key] = profile
        return profile
    
    def verify_type_assignment(
        self,
        neuron_type: str,
        datasets: List[str],
        direction: str = 'both',
        score_weights: Optional[Dict[str, float]] = None,
        show_variance: bool = True,
        comparison_mode: Optional[str] = None
    ) -> VerificationResult:
        """
        Verify that a neuron type has consistent connectivity across datasets.
        
        Supports two comparison modes:
        - 'loose': Type-aggregated profiles (faster, default)
        - 'strict': Per-bodyId profile comparison (more precise)
        
        Args:
            neuron_type: Type name to verify (uses label_mapper if provided)
            datasets: List of datasets to compare
            direction: 'upstream', 'downstream', or 'both'
            score_weights: Custom weights for combined score
            show_variance: Include within-type variance in results
            comparison_mode: 'loose' or 'strict' (default: use self.comparison_mode)
        
        Returns:
            VerificationResult with all metrics and confidence level
        
        Example:
            >>> result = verifier.verify_type_assignment(
            ...     'aMe12', ['hemibrain:v1.2.1', 'male-cns:v0.9']
            ... )
            >>> print(result.confidence)
            'High'
        """
        # Use instance-level comparison_mode if not specified
        mode = comparison_mode or getattr(self, 'comparison_mode', 'loose')
        
        # Only log if not in batch mode (to reduce noise)
        if not getattr(self, '_in_batch_mode', False):
            self._log(f"Verifying type assignment: {neuron_type} (mode: {mode})")
        
        # Use score_weights from instance if not provided
        weights = score_weights or self.score_weights
        
        # Extract profiles for each dataset
        profiles: Dict[str, ConnectivityProfile] = {}
        weak_connectivity_datasets = []
        
        for dataset in datasets:
            # Get mapped type name if label_mapper is available
            mapped_type = self._get_mapped_type(neuron_type, dataset)
            
            try:
                profile = self._get_profile(mapped_type, dataset)
                profiles[dataset] = profile
                
                if profile.is_weak_connectivity:
                    weak_connectivity_datasets.append(dataset)
                    
            except Exception as e:
                self._log(f"Warning: Could not extract profile for {neuron_type} in {dataset}: {e}")
        
        if len(profiles) < 2:
            self._log(f"Insufficient profiles for verification (need >= 2, got {len(profiles)})")
            return VerificationResult(
                neuron_type=neuron_type,
                datasets=datasets,
                pairwise_scores=[],
                avg_combined_score=0.0,
                min_score=0.0,
                max_score=0.0,
                confidence='Very Low',
                weak_connectivity_datasets=weak_connectivity_datasets,
                verification_status='failed'
            )
        
        # Compute pairwise comparisons
        pairwise_scores: List[ComparisonResult] = []
        dataset_keys = list(profiles.keys())
        
        # Track which datasets have empty profiles (type not found)
        empty_profile_datasets = []
        for dataset in dataset_keys:
            profile = profiles[dataset]
            if not profile.upstream_partners and not profile.downstream_partners:
                empty_profile_datasets.append(dataset)
        
        for i in range(len(dataset_keys)):
            for j in range(i + 1, len(dataset_keys)):
                ds_a, ds_b = dataset_keys[i], dataset_keys[j]
                profile_a, profile_b = profiles[ds_a], profiles[ds_b]
                
                if mode == 'strict':
                    # Strict mode: Use per-bodyId comparison with rank correlation
                    result = self._compare_profiles_strict(
                        profile_a, profile_b, direction, weights
                    )
                else:
                    # Loose mode (default): Type-aggregated comparison
                    result = ProfileComparator.compare_profiles(
                        profile_a, profile_b, direction, weights
                    )
                pairwise_scores.append(result)
        
        # Aggregate scores - EXCLUDE pairs where either profile is empty (type not found)
        # These should use NaN, not 0, to avoid dragging down the average
        valid_scores = []
        for idx, score in enumerate(pairwise_scores):
            # Reconstruct which datasets this pair compares
            pair_idx = 0
            found_ds_a, found_ds_b = None, None
            for i in range(len(dataset_keys)):
                for j in range(i + 1, len(dataset_keys)):
                    if pair_idx == idx:
                        found_ds_a, found_ds_b = dataset_keys[i], dataset_keys[j]
                        break
                    pair_idx += 1
                if found_ds_a is not None:
                    break
            
            # Only include score if BOTH profiles have data
            profile_a, profile_b = profiles[found_ds_a], profiles[found_ds_b]
            a_has_data = bool(profile_a.upstream_partners or profile_a.downstream_partners)
            b_has_data = bool(profile_b.upstream_partners or profile_b.downstream_partners)
            
            if a_has_data and b_has_data:
                valid_scores.append(score.combined)
        
        # Calculate stats from valid scores only
        if valid_scores:
            avg_score = np.mean(valid_scores)
            min_score = np.min(valid_scores)
            max_score = np.max(valid_scores)
        else:
            # All profiles are empty - type not found in any dataset
            avg_score = np.nan
            min_score = np.nan
            max_score = np.nan
        
        # Calculate average rank correlation for confidence determination
        # Edge case handling: If rank_correlation for 'both' is NaN but directional scores exist,
        # use the average of available directional scores instead
        if valid_scores:
            valid_rank_corrs = []
            for idx, score in enumerate(pairwise_scores):
                pair_idx = 0
                found_ds_a, found_ds_b = None, None
                for i in range(len(dataset_keys)):
                    for j in range(i + 1, len(dataset_keys)):
                        if pair_idx == idx:
                            found_ds_a, found_ds_b = dataset_keys[i], dataset_keys[j]
                            break
                        pair_idx += 1
                    if found_ds_a is not None:
                        break
                profile_a, profile_b = profiles[found_ds_a], profiles[found_ds_b]
                a_has_data = bool(profile_a.upstream_partners or profile_a.downstream_partners)
                b_has_data = bool(profile_b.upstream_partners or profile_b.downstream_partners)
                if a_has_data and b_has_data:
                    rank_val = score.rank_correlation
                    # Edge case: If combined rank_corr is NaN, try to compute from directional
                    if np.isnan(rank_val):
                        # Compute directional rank correlations
                        up_corr = ProfileComparator.rank_correlation(profile_a, profile_b, 'upstream')
                        down_corr = ProfileComparator.rank_correlation(profile_a, profile_b, 'downstream')
                        # Use average of available directional scores
                        dir_scores = [s for s in [up_corr, down_corr] if not np.isnan(s)]
                        if dir_scores:
                            rank_val = np.mean(dir_scores)
                    if not np.isnan(rank_val):
                        valid_rank_corrs.append(rank_val)
            avg_rank_corr = np.mean(valid_rank_corrs) if valid_rank_corrs else np.nan
        else:
            avg_rank_corr = np.nan
        
        # Determine overall confidence and status based on RANK CORRELATION
        # Normalize rank_corr from [-1, 1] to [0, 1] for consistent thresholds
        avg_rank_corr_norm = (avg_rank_corr + 1) / 2 if not np.isnan(avg_rank_corr) else np.nan
        
        if np.isnan(avg_rank_corr_norm):
            confidence = 'N/A'
            status = 'type_not_found'
        else:
            # Use normalized rank_corr [0, 1] for confidence determination
            # Thresholds: Very High >= 0.85, High >= 0.7, Medium >= 0.5, Low >= 0.3, Very Low < 0.3
            if avg_rank_corr_norm >= 0.85:
                confidence = 'Very High'
                status = 'verified'
            elif avg_rank_corr_norm >= 0.7:
                confidence = 'High'
                status = 'verified'
            elif avg_rank_corr_norm >= 0.5:
                confidence = 'Medium'
                status = 'needs_review'
            elif avg_rank_corr_norm >= 0.3:
                confidence = 'Low'
                status = 'questionable'
            else:
                confidence = 'Very Low'
                status = 'failed'
        
        return VerificationResult(
            neuron_type=neuron_type,
            datasets=list(profiles.keys()),
            pairwise_scores=pairwise_scores,
            avg_combined_score=avg_score,
            min_score=min_score,
            max_score=max_score,
            confidence=confidence,
            weak_connectivity_datasets=weak_connectivity_datasets,
            verification_status=status
        )
    
    def find_similar_neurons(
        self,
        query_neuron: Union[str, int],
        query_dataset: str,
        target_dataset: str,
        candidate_types: Optional[List[str]] = None,
        top_k: int = 10,
        direction: str = 'both'
    ) -> pd.DataFrame:
        """
        Find neurons in target_dataset most similar to query_neuron.
        
        Args:
            query_neuron: Type name or bodyId to find homologs for
            query_dataset: Dataset containing query neuron
            target_dataset: Dataset to search for similar neurons
            candidate_types: Limit search to these types (None = search common types)
            top_k: Number of top matches to return
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            DataFrame with columns:
            - target_type: Matched type name
            - combined_score: Combined similarity score
            - jaccard, cosine, rank_corr: Individual metrics
            - confidence: Confidence level
        
        Example:
            >>> matches = verifier.find_similar_neurons(
            ...     'aMe12', 'hemibrain:v1.2.1', 'male-cns:v0.9', top_k=5
            ... )
            >>> print(matches['target_type'].tolist())
        """
        self._log(f"Finding similar neurons: {query_neuron} ({query_dataset} -> {target_dataset})")
        
        # Get query profile
        query_profile = self._get_profile(query_neuron, query_dataset)
        
        # If no candidate types specified, use types from the query profile's partners
        if candidate_types is None:
            # Use partner types from query profile as candidates
            candidate_types = list(
                set(query_profile.upstream_partners.keys()) | 
                set(query_profile.downstream_partners.keys())
            )
            self._log(f"Using {len(candidate_types)} partner types as candidates")
        
        if not candidate_types:
            self._log("No candidate types to search")
            return pd.DataFrame()
        
        # Limit candidates to avoid excessive computation
        max_candidates = 100
        if len(candidate_types) > max_candidates:
            self._log(f"Limiting candidates from {len(candidate_types)} to {max_candidates}")
            candidate_types = candidate_types[:max_candidates]
        
        # Compare against each candidate
        results = []
        
        for candidate in candidate_types:
            try:
                candidate_profile = self._get_profile(candidate, target_dataset)
                
                # Skip if no connectivity data
                if not candidate_profile.upstream_partners and not candidate_profile.downstream_partners:
                    continue
                
                comparison = ProfileComparator.compare_profiles(
                    query_profile, candidate_profile, direction
                )
                
                results.append({
                    'target_type': candidate,
                    'combined_score': comparison.combined,
                    'jaccard': comparison.jaccard,
                    'cosine': comparison.cosine,
                    'rank_corr': comparison.rank_correlation,
                    'overlap_a_in_b': comparison.overlap_a_in_b,
                    'overlap_b_in_a': comparison.overlap_b_in_a,
                    'confidence': comparison.confidence,
                    'weak_connectivity': comparison.weak_connectivity_b
                })
                
            except Exception as e:
                self._log(f"Warning: Could not compare with {candidate}: {e}")
        
        if not results:
            return pd.DataFrame()
        
        # Create DataFrame and sort by combined score
        df = pd.DataFrame(results)
        df = df.sort_values('combined_score', ascending=False).head(top_k)
        df = df.reset_index(drop=True)
        
        return df
    
    def find_homologs_for_untyped(
        self,
        query_bodyids: List[int],
        query_dataset: str,
        target_dataset: str,
        top_k: int = 5,
        direction: str = 'both'
    ) -> pd.DataFrame:
        """
        Find potential homologs for untyped neurons.
        
        Args:
            query_bodyids: List of untyped bodyIds in query dataset
            query_dataset: Dataset containing query neurons
            target_dataset: Dataset to search for homologs
            top_k: Number of top matches per query
            direction: 'upstream', 'downstream', or 'both'
        
        Returns:
            DataFrame with columns:
            - query_bodyid: Query neuron bodyId
            - target_type: Suggested type assignment
            - combined_score: Similarity score
            - confidence: Confidence level
        
        Example:
            >>> homologs = verifier.find_homologs_for_untyped(
            ...     [720575940610453042], 'flywire_FAFB_v783', 'hemibrain:v1.2.1'
            ... )
        """
        self._log(f"Finding homologs for {len(query_bodyids)} untyped neurons")
        
        all_results = []
        
        for bodyid in query_bodyids:
            try:
                # Get profile for untyped neuron (using bodyId)
                query_profile = self._get_profile(bodyid, query_dataset)
                
                # Skip if weak connectivity
                if query_profile.is_weak_connectivity:
                    self._log(f"Warning: Weak connectivity for bodyId {bodyid}, results may be unreliable")
                
                # Get candidate types from query's partners
                # These are the types the untyped neuron connects to
                partner_types = (
                    set(query_profile.upstream_partners.keys()) | 
                    set(query_profile.downstream_partners.keys())
                )
                
                # Find similar typed neurons in target dataset
                matches = self.find_similar_neurons(
                    query_neuron=bodyid,
                    query_dataset=query_dataset,
                    target_dataset=target_dataset,
                    candidate_types=list(partner_types),
                    top_k=top_k,
                    direction=direction
                )
                
                if not matches.empty:
                    matches = matches.copy()
                    matches['query_bodyid'] = bodyid
                    all_results.append(matches)
                    
            except Exception as e:
                self._log(f"Warning: Could not process bodyId {bodyid}: {e}")
        
        if not all_results:
            return pd.DataFrame()
        
        # Combine results
        combined = pd.concat(all_results, ignore_index=True)
        
        # Reorder columns
        cols = ['query_bodyid', 'target_type', 'combined_score', 'confidence', 
                'jaccard', 'cosine', 'rank_corr']
        cols = [c for c in cols if c in combined.columns]
        combined = combined[cols + [c for c in combined.columns if c not in cols]]
        
        return combined
    
    def _verify_single_type(
        self,
        neuron_type: str,
        datasets: List[str],
        direction: str,
        score_weights: Optional[Dict[str, float]],
        include_directional: bool
    ) -> Dict[str, Any]:
        """
        Verify a single neuron type (helper for parallel execution).
        
        This method is thread-safe as it only reads from shared caches.
        
        Args:
            neuron_type: Type to verify
            datasets: List of datasets to compare
            direction: 'upstream', 'downstream', or 'both'
            score_weights: Custom weights for combined score
            include_directional: Include directional scores
            
        Returns:
            Dict with verification results for this type
        """
        try:
            verification = self.verify_type_assignment(
                neuron_type, datasets, direction, score_weights
            )
            
            # Count datasets where type was found (has profile data)
            datasets_found = 0
            for dataset in datasets:
                try:
                    profile = self._get_profile(neuron_type, dataset)
                    if profile.upstream_partners or profile.downstream_partners:
                        datasets_found += 1
                except Exception:
                    pass
            
            # Calculate average individual metrics from pairwise scores
            pairwise = verification.pairwise_scores
            if pairwise:
                valid_pairs = [p for p in pairwise if not (np.isnan(p.jaccard) and np.isnan(p.cosine))]
                if valid_pairs:
                    avg_jaccard = np.nanmean([p.jaccard for p in valid_pairs])
                    avg_cosine = np.nanmean([p.cosine for p in valid_pairs])
                    rank_values = [p.rank_correlation_norm for p in valid_pairs if not np.isnan(p.rank_correlation)]
                    avg_rank = np.mean(rank_values) if rank_values else np.nan
                    rank_union_values = [p.rank_union for p in valid_pairs if not np.isnan(p.rank_union)]
                    avg_rank_union = np.mean(rank_union_values) if rank_union_values else np.nan
                    avg_overlap_a_in_b = np.nanmean([p.overlap_a_in_b for p in valid_pairs])
                    avg_overlap_b_in_a = np.nanmean([p.overlap_b_in_a for p in valid_pairs])
                    avg_overlap = np.nanmean([avg_overlap_a_in_b, avg_overlap_b_in_a])
                else:
                    avg_jaccard = avg_cosine = avg_rank = avg_rank_union = avg_overlap_a_in_b = avg_overlap_b_in_a = avg_overlap = np.nan
            else:
                avg_jaccard = avg_cosine = avg_rank = avg_rank_union = avg_overlap_a_in_b = avg_overlap_b_in_a = avg_overlap = np.nan
            
            # Get unique_types from profiles
            total_unique_types = 0
            for dataset in datasets:
                try:
                    profile = self._get_profile(neuron_type, dataset)
                    total_unique_types += profile.unique_types_upstream + profile.unique_types_downstream
                except Exception:
                    pass
            
            # Determine confidence
            if np.isnan(avg_rank):
                confidence_level = 'Error'
            elif avg_rank >= 0.85:
                confidence_level = 'Very High'
            elif avg_rank >= 0.70:
                confidence_level = 'High'
            elif avg_rank >= 0.50:
                confidence_level = 'Medium'
            elif avg_rank >= 0.30:
                confidence_level = 'Low'
            else:
                confidence_level = 'Very Low'
            
            result_row = {
                'neuron_type': neuron_type,
                'datasets_found': datasets_found,
                'total_datasets': len(datasets),
                'avg_jaccard': avg_jaccard,
                'avg_rank_corr': avg_rank,
                'avg_rank_union': avg_rank_union,
                'avg_overlap': avg_overlap,
                'confidence': confidence_level,
                'datasets_compared': len(verification.datasets),
                'total_unique_types': total_unique_types,
            }
            
            # Directional scores
            if include_directional:
                profiles_cache: Dict[str, ConnectivityProfile] = {}
                for dataset in datasets:
                    try:
                        profiles_cache[dataset] = self._get_profile(neuron_type, dataset)
                    except Exception:
                        pass
                
                for dir_name in ['upstream', 'downstream', 'both']:
                    try:
                        dir_rank_values = []
                        dir_jaccard_values = []
                        
                        dataset_keys = list(profiles_cache.keys())
                        for i in range(len(dataset_keys)):
                            for j in range(i + 1, len(dataset_keys)):
                                ds_a, ds_b = dataset_keys[i], dataset_keys[j]
                                profile_a, profile_b = profiles_cache[ds_a], profiles_cache[ds_b]
                                
                                a_has_data = bool(profile_a.upstream_partners or profile_a.downstream_partners)
                                b_has_data = bool(profile_b.upstream_partners or profile_b.downstream_partners)
                                if not (a_has_data and b_has_data):
                                    continue
                                
                                scores = ProfileComparator.combined_score(
                                    profile_a, profile_b, 
                                    weights=score_weights or self.score_weights,
                                    direction=dir_name
                                )
                                
                                if not np.isnan(scores['rank']):
                                    dir_rank_values.append((scores['rank'] + 1) / 2)
                                if not np.isnan(scores['jaccard']):
                                    dir_jaccard_values.append(scores['jaccard'])
                        
                        result_row[f'avg_rank_corr_{dir_name}'] = np.mean(dir_rank_values) if dir_rank_values else np.nan
                        result_row[f'avg_jaccard_{dir_name}'] = np.mean(dir_jaccard_values) if dir_jaccard_values else np.nan
                        
                    except Exception:
                        result_row[f'avg_rank_corr_{dir_name}'] = np.nan
                        result_row[f'avg_jaccard_{dir_name}'] = np.nan
            
            return result_row
            
        except Exception as e:
            error_row = {
                'neuron_type': neuron_type,
                'datasets_found': 0,
                'total_datasets': len(datasets),
                'avg_jaccard': np.nan,
                'avg_rank_corr': np.nan,
                'avg_overlap': np.nan,
                'confidence': 'Error',
                'datasets_compared': 0,
                'total_unique_types': 0,
            }
            if include_directional:
                error_row['avg_rank_corr_upstream'] = np.nan
                error_row['avg_rank_corr_downstream'] = np.nan
                error_row['avg_rank_corr_both'] = np.nan
                error_row['avg_jaccard_upstream'] = np.nan
                error_row['avg_jaccard_downstream'] = np.nan
                error_row['avg_jaccard_both'] = np.nan
            return error_row
    
    def batch_verify_types(
        self,
        neuron_types: List[str],
        datasets: List[str],
        direction: str = 'both',
        score_weights: Optional[Dict[str, float]] = None,
        include_directional: bool = True,
        parallel: bool = True,
        max_workers: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Batch verification for multiple neuron types.
        
        Supports parallel execution for faster processing when verifying
        many types. The parallel mode is thread-safe as all operations
        read from shared in-memory caches (profile cache, connection cache).
        
        Args:
            neuron_types: List of types to verify
            datasets: List of datasets to compare
            direction: 'upstream', 'downstream', or 'both'
            score_weights: Custom weights for combined score
            include_directional: If True, also compute separate upstream/downstream scores
            parallel: If True, use parallel threads (default: True)
            max_workers: Max parallel workers (default: min(32, cpu_count + 4))
        
        Returns:
            Summary DataFrame with verification results for each type
        """
        self._log(f"Batch verifying {len(neuron_types)} types across {len(datasets)} datasets")
        
        # Set batch mode flag to suppress per-type logging
        self._in_batch_mode = True
        
        results = []
        
        # Use tqdm for progress
        from tqdm import tqdm
        
        if parallel and len(neuron_types) > 1:
            # Parallel execution using ThreadPoolExecutor
            # Thread-safe: all operations read from shared caches
            import os
            if max_workers is None:
                max_workers = min(32, (os.cpu_count() or 1) + 4)
            
            self._log(f"Using parallel execution with {max_workers} workers")
            
            # Pre-warm ALL caches BEFORE starting parallel execution
            # This ensures all cache loading messages appear before the progress bar
            self._log("Pre-loading caches...")
            
            # 1. Pre-warm connection data cache
            for dataset in datasets:
                try:
                    self.profiler._get_cached_conn_df(dataset)
                except Exception:
                    pass
            
            # 2. Pre-warm connectivity profiles cache (load parquet into memory)
            for dataset in datasets:
                try:
                    self.profiler._load_cache_dataframe(dataset)
                except Exception:
                    pass
            
            # 3. Enable deferred cache writes to prevent parallel write corruption
            self.profiler._defer_cache_writes = True
            
            try:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all verification tasks
                    future_to_type = {
                        executor.submit(
                            self._verify_single_type,
                            neuron_type, datasets, direction, score_weights, include_directional
                        ): neuron_type
                        for neuron_type in neuron_types
                    }
                    
                    # Collect results with progress bar
                    for future in tqdm(
                        as_completed(future_to_type), 
                        total=len(neuron_types),
                        desc="Verifying types (parallel)",
                        disable=not self.verbose,
                        leave=True
                    ):
                        neuron_type = future_to_type[future]
                        try:
                            result_row = future.result()
                            results.append(result_row)
                        except Exception as e:
                            self._log(f"Warning: Failed to verify {neuron_type}: {e}")
                            error_row = {
                                'neuron_type': neuron_type,
                                'datasets_found': 0,
                                'total_datasets': len(datasets),
                                'avg_jaccard': np.nan,
                                'avg_rank_corr': np.nan,
                                'avg_overlap': np.nan,
                                'confidence': 'Error',
                                'datasets_compared': 0,
                                'total_unique_types': 0,
                            }
                            if include_directional:
                                error_row['avg_rank_corr_upstream'] = np.nan
                                error_row['avg_rank_corr_downstream'] = np.nan
                                error_row['avg_rank_corr_both'] = np.nan
                                error_row['avg_jaccard_upstream'] = np.nan
                                error_row['avg_jaccard_downstream'] = np.nan
                                error_row['avg_jaccard_both'] = np.nan
                            results.append(error_row)
            finally:
                # Disable deferred writes and flush pending cache writes
                self.profiler._defer_cache_writes = False
                self.profiler.flush_pending_cache_writes()
        else:
            # Sequential execution (original behavior)
            iterator = tqdm(neuron_types, desc="Verifying types", disable=not self.verbose, leave=True)
            
            for neuron_type in iterator:
                iterator.set_postfix_str(neuron_type[:20])
                result_row = self._verify_single_type(
                    neuron_type, datasets, direction, score_weights, include_directional
                )
                results.append(result_row)
        
        # Reset batch mode flag
        self._in_batch_mode = False
        
        df = pd.DataFrame(results)
        
        # Sorting: 1) by role (if present), 2) by datasets_found, 3) by rank_corr
        sort_cols = []
        ascending_flags = []
        if 'datasets_found' in df.columns:
            sort_cols.append('datasets_found')
            ascending_flags.append(False)  # More datasets found = better
        sort_cols.append('avg_rank_corr' if 'avg_rank_corr' in df.columns else 'avg_combined_score')
        ascending_flags.append(False)
        
        df = df.sort_values(sort_cols, ascending=ascending_flags)
        df = df.reset_index(drop=True)
        
        return df
    
    def verify_comparison_results(
        self,
        source_types: List[str],
        target_types: List[str],
        intermediate_types: List[str],
        datasets: List[str],
        direction: str = 'both',
        parallel: bool = True,
        max_workers: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Verify all neuron types from a comparison analysis.
        
        This method is designed to integrate with ComparisonAnalyzer results,
        verifying connectivity profile consistency for source, target, and
        intermediate neurons.
        
        Deduplication: If a type appears in multiple roles (e.g., source=target),
        it's merged with role='source/target' to avoid duplicates.
        
        Args:
            source_types: Source neuron types from comparison
            target_types: Target neuron types from comparison
            intermediate_types: Intermediate neuron types from path analysis
            datasets: Datasets being compared
            direction: 'upstream', 'downstream', or 'both'
            parallel: If True, use parallel threads (default: True)
            max_workers: Max parallel workers (default: min(32, cpu_count + 4))
        
        Returns:
            Dict with keys:
            - 'source': DataFrame with source type verification
            - 'target': DataFrame with target type verification
            - 'intermediate': DataFrame with intermediate type verification
            - 'summary': Overall summary DataFrame (deduplicated with merged roles)
        """
        self._log("Verifying comparison results")
        
        # Convert to sets for efficient lookup
        source_set = set(source_types) if source_types else set()
        target_set = set(target_types) if target_types else set()
        intermediate_set = set(intermediate_types) if intermediate_types else set()
        
        # Determine role for each type (merging duplicates)
        type_roles = {}
        for t in source_set:
            type_roles[t] = ['source']
        for t in target_set:
            if t in type_roles:
                type_roles[t].append('target')
            else:
                type_roles[t] = ['target']
        for t in intermediate_set:
            if t not in type_roles:
                type_roles[t] = ['intermediate']
            # If already source/target, intermediate is redundant
        
        # Build merged role strings
        role_map = {}
        for t, roles in type_roles.items():
            role_map[t] = '/'.join(roles)  # e.g., 'source/target'
        
        results = {}
        
        # Get all unique types
        all_unique_types = list(type_roles.keys())
        
        # Verify all types in one batch
        if all_unique_types:
            self._log(f"Verifying {len(all_unique_types)} unique types across {len(datasets)} datasets")
            all_verified = self.batch_verify_types(
                all_unique_types, datasets, direction,
                parallel=parallel, max_workers=max_workers
            )
            all_verified['role'] = all_verified['neuron_type'].map(role_map)
        else:
            all_verified = pd.DataFrame()
        
        # Split into role-specific DataFrames for backward compatibility
        results['source'] = all_verified[all_verified['role'].str.contains('source', na=False)].copy() if not all_verified.empty else pd.DataFrame()
        results['target'] = all_verified[all_verified['role'].str.contains('target', na=False)].copy() if not all_verified.empty else pd.DataFrame()
        results['intermediate'] = all_verified[all_verified['role'] == 'intermediate'].copy() if not all_verified.empty else pd.DataFrame()
        
        # Create deduplicated summary (no duplicate types!)
        if not all_verified.empty:
            results['summary'] = all_verified.copy()
            
            # Filter to keep only types appearing in at least 2 datasets (avg_rank_corr is not NaN)
            if 'avg_rank_corr' in results['summary'].columns:
                results['summary'] = results['summary'][
                    results['summary']['avg_rank_corr'].notna()
                ].copy()
            
            # Sort by: 1) role order, 2) datasets_found, 3) avg_rank_corr
            role_order = {'source': 0, 'source/target': 0.5, 'target': 1, 'intermediate': 2}
            if 'role' in results['summary'].columns:
                results['summary']['_role_order'] = results['summary']['role'].map(
                    lambda x: min([role_order.get(r.strip(), 3) for r in x.split('/')]) if pd.notna(x) else 3
                )
                sort_cols = ['_role_order']
                ascending = [True]
                if 'datasets_found' in results['summary'].columns:
                    sort_cols.append('datasets_found')
                    ascending.append(False)
                sort_cols.append('avg_rank_corr' if 'avg_rank_corr' in results['summary'].columns else 'avg_combined_score')
                ascending.append(False)
                results['summary'] = results['summary'].sort_values(
                    sort_cols, ascending=ascending
                ).drop(columns=['_role_order'])
            results['summary'] = results['summary'].reset_index(drop=True)
        else:
            results['summary'] = pd.DataFrame()
        
        return results
    
    def build_cross_dataset_similarity_matrix(
        self,
        neuron_types: List[str],
        datasets: List[str],
        metric: str = 'combined',
        direction: str = 'both',
        dataset_nicknames: Optional[Dict[str, str]] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Build similarity matrix showing profile agreement for types across dataset pairs.
        
        This produces the visualization shown in TODO_connprofile.md:
        Rows = neuron types, Columns = dataset pairs, Values = similarity scores
        
        Supports parallel execution for faster processing.
        
        Args:
            neuron_types: Types to include in matrix
            datasets: Datasets to compare
            metric: 'combined', 'jaccard', 'cosine', or 'rank'
            direction: 'upstream', 'downstream', or 'both'
            dataset_nicknames: Optional dict mapping dataset names to short nicknames
                              e.g., {'hemibrain:v1.2.1': 'HB', 'male-cns:v0.9': 'MCNS'}
            parallel: If True, use parallel threads (default: True)
            max_workers: Max parallel workers (default: min(32, cpu_count + 4))
        
        Returns:
            DataFrame with neuron types as index and dataset pairs as columns.
            Failed type lookups return NaN (not 0).
        """
        self._log(f"Building similarity matrix for {len(neuron_types)} types")
        
        # Generate dataset pair columns
        dataset_pairs = []
        for i in range(len(datasets)):
            for j in range(i + 1, len(datasets)):
                # Use nicknames if provided
                if dataset_nicknames:
                    name_a = dataset_nicknames.get(datasets[i], datasets[i])
                    name_b = dataset_nicknames.get(datasets[j], datasets[j])
                else:
                    name_a = datasets[i]
                    name_b = datasets[j]
                pair_name = f"{name_a} vs {name_b}"
                dataset_pairs.append((datasets[i], datasets[j], pair_name))
        
        def compute_row_for_type(neuron_type: str) -> Tuple[str, Dict[str, float]]:
            """Compute similarity row for a single neuron type."""
            row = {}
            for ds_a, ds_b, pair_name in dataset_pairs:
                try:
                    profile_a = self._get_profile(neuron_type, ds_a)
                    profile_b = self._get_profile(neuron_type, ds_b)
                    
                    profile_a_empty = (not profile_a.upstream_partners and not profile_a.downstream_partners)
                    profile_b_empty = (not profile_b.upstream_partners and not profile_b.downstream_partners)
                    
                    if profile_a_empty and profile_b_empty:
                        row[pair_name] = np.nan
                        continue
                    
                    if metric == 'jaccard':
                        score = ProfileComparator.jaccard_similarity(profile_a, profile_b, direction)
                    elif metric == 'cosine':
                        score = ProfileComparator.weighted_cosine_similarity(profile_a, profile_b, direction)
                    elif metric == 'rank':
                        raw_score = ProfileComparator.rank_correlation(profile_a, profile_b, direction)
                        score = (raw_score + 1) / 2 if not np.isnan(raw_score) else np.nan
                    else:  # combined
                        scores = ProfileComparator.combined_score(profile_a, profile_b, direction=direction)
                        score = scores['combined']
                    
                    row[pair_name] = score
                except Exception:
                    row[pair_name] = np.nan
            return neuron_type, row
        
        # Build matrix
        matrix_data = {}
        
        if parallel and len(neuron_types) > 1:
            import os
            if max_workers is None:
                max_workers = min(32, (os.cpu_count() or 1) + 4)
            
            # Enable deferred cache writes to prevent parallel write corruption
            self.profiler._defer_cache_writes = True
            
            try:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(compute_row_for_type, nt): nt for nt in neuron_types}
                    for future in as_completed(futures):
                        neuron_type, row = future.result()
                        matrix_data[neuron_type] = row
            finally:
                # Disable deferred writes and flush pending cache writes
                self.profiler._defer_cache_writes = False
                self.profiler.flush_pending_cache_writes()
        else:
            for neuron_type in neuron_types:
                _, row = compute_row_for_type(neuron_type)
                matrix_data[neuron_type] = row
        
        df = pd.DataFrame(matrix_data).T
        df.index.name = 'neuron_type'
        
        return df
    
    def build_directional_similarity_matrices(
        self,
        neuron_types: List[str],
        datasets: List[str],
        metric: str = 'combined',
        dataset_nicknames: Optional[Dict[str, str]] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Build separate similarity matrices for upstream and downstream directions.
        
        This supports Round 4 requirement: keep upstream/downstream scores separate.
        
        Args:
            neuron_types: Types to include in matrix
            datasets: Datasets to compare
            metric: 'combined', 'jaccard', 'cosine', or 'rank'
            dataset_nicknames: Optional dict mapping dataset names to short nicknames
            parallel: If True, use parallel threads (default: True)
            max_workers: Max parallel workers (default: min(32, cpu_count + 4))
        
        Returns:
            Dict with keys 'upstream', 'downstream', 'both' containing DataFrames.
            Each DataFrame has neuron types as rows and dataset pairs as columns.
            Failed type lookups return NaN (not 0).
        """
        self._log(f"Building directional similarity matrices for {len(neuron_types)} types")
        
        result = {}
        
        for direction in ['upstream', 'downstream', 'both']:
            result[direction] = self.build_cross_dataset_similarity_matrix(
                neuron_types=neuron_types,
                datasets=datasets,
                metric=metric,
                direction=direction,
                dataset_nicknames=dataset_nicknames,
                parallel=parallel,
                max_workers=max_workers
            )
        
        return result
    
    def build_multi_metric_matrices(
        self,
        neuron_types: List[str],
        datasets: List[str],
        direction: str = 'both',
        dataset_nicknames: Optional[Dict[str, str]] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Build similarity matrices for each individual metric.
        
        This supports Round 4 requirement: visualize individual score matrices.
        
        Args:
            neuron_types: Types to include
            datasets: Datasets to compare
            direction: 'upstream', 'downstream', or 'both'
            dataset_nicknames: Optional dict mapping dataset names to short nicknames
            parallel: If True, use parallel threads (default: True)
            max_workers: Max parallel workers (default: min(32, cpu_count + 4))
        
        Returns:
            Dict with keys 'combined', 'jaccard', 'cosine', 'rank' containing DataFrames.
            Failed type lookups return NaN (not 0).
        """
        self._log(f"Building multi-metric matrices for {len(neuron_types)} types")
        
        result = {}
        
        for metric in ['combined', 'jaccard', 'cosine', 'rank']:
            result[metric] = self.build_cross_dataset_similarity_matrix(
                neuron_types=neuron_types,
                datasets=datasets,
                metric=metric,
                direction=direction,
                dataset_nicknames=dataset_nicknames,
                parallel=parallel,
                max_workers=max_workers
            )
        
        return result
    
    def generate_verification_report(
        self,
        neuron_types: List[str],
        datasets: List[str],
        output_path: str,
        direction: str = 'both',
        include_partner_details: bool = True,
        dataset_nicknames: Optional[Dict[str, str]] = None,
        save_directional_matrices: bool = True,
        save_metric_matrices: bool = True,
        parallel: bool = True,
        max_workers: Optional[int] = None
    ) -> None:
        """
        Generate comprehensive verification report.
        
        Outputs (Round 4 enhanced):
        - verification_summary.csv: High-level summary per type
        - verification_details.csv: Full pairwise comparisons
        - similarity_matrix_combined.csv: Combined score matrix
        - similarity_matrix_upstream.csv: Upstream direction only
        - similarity_matrix_downstream.csv: Downstream direction only
        - similarity_matrix_jaccard.csv: Jaccard scores
        - similarity_matrix_cosine.csv: Cosine scores
        - similarity_matrix_rank.csv: Rank correlation scores
        - partner_overlap_{type}.csv: Per-type partner analysis (if include_partner_details)
        
        Round 4 Features:
        - Failed type lookups marked as NaN (not 0)
        - Individual score matrices for each metric
        - Dataset nicknames for shorter labels
        - Separate upstream/downstream matrices
        - Parallel processing support for faster execution
        
        Args:
            neuron_types: Types to verify
            datasets: Datasets to compare
            output_path: Directory to save report files
            direction: 'upstream', 'downstream', or 'both' (for summary/details)
            include_partner_details: Include per-type partner overlap CSVs
            dataset_nicknames: Optional dict for shorter dataset labels
            save_directional_matrices: Save separate upstream/downstream matrices
            save_metric_matrices: Save individual metric matrices (jaccard, cosine, rank)
            parallel: If True, use parallel threads (default: True)
            max_workers: Max parallel workers (default: min(32, cpu_count + 4))
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self._log(f"Generating verification report to {output_dir}")
        
        # 1. Summary CSV
        summary = self.batch_verify_types(
            neuron_types, datasets, direction,
            parallel=parallel, max_workers=max_workers
        )
        summary.to_csv(output_dir / 'verification_summary.csv', index=False)
        self._log("Saved: verification_summary.csv")
        
        # 2. Detailed pairwise comparisons
        detailed_results = []
        for neuron_type in neuron_types:
            try:
                verification = self.verify_type_assignment(neuron_type, datasets, direction)
                for comparison in verification.pairwise_scores:
                    row = comparison.to_dict()
                    row['neuron_type'] = neuron_type
                    detailed_results.append(row)
            except Exception as e:
                self._log(f"Warning: Could not get details for {neuron_type}: {e}")
        
        if detailed_results:
            details_df = pd.DataFrame(detailed_results)
            details_df.to_csv(output_dir / 'verification_details.csv', index=False)
            self._log("Saved: verification_details.csv")
        
        # 3. Combined similarity matrix (main)
        similarity_matrix = self.build_cross_dataset_similarity_matrix(
            neuron_types, datasets, metric='combined', direction=direction,
            dataset_nicknames=dataset_nicknames,
            parallel=parallel, max_workers=max_workers
        )
        similarity_matrix.to_csv(output_dir / 'similarity_matrix_combined.csv')
        self._log("Saved: similarity_matrix_combined.csv")
        
        # 4. Directional matrices (Round 4: keep upstream/downstream separate)
        if save_directional_matrices:
            matrices_dir = output_dir / 'directional_matrices'
            matrices_dir.mkdir(exist_ok=True)
            
            directional = self.build_directional_similarity_matrices(
                neuron_types, datasets, metric='combined',
                dataset_nicknames=dataset_nicknames,
                parallel=parallel, max_workers=max_workers
            )
            for dir_name, matrix in directional.items():
                matrix.to_csv(matrices_dir / f'similarity_matrix_{dir_name}.csv')
            self._log("Saved: directional similarity matrices (upstream, downstream, both)")
        
        # 5. Individual metric matrices (Round 4: visualize each single score)
        if save_metric_matrices:
            metrics_dir = output_dir / 'metric_matrices'
            metrics_dir.mkdir(exist_ok=True)
            
            metric_matrices = self.build_multi_metric_matrices(
                neuron_types, datasets, direction=direction,
                dataset_nicknames=dataset_nicknames,
                parallel=parallel, max_workers=max_workers
            )
            for metric_name, matrix in metric_matrices.items():
                matrix.to_csv(metrics_dir / f'similarity_matrix_{metric_name}.csv')
            self._log("Saved: metric-specific matrices (combined, jaccard, cosine, rank)")
        
        # 6. Partner overlap details (optional) - now supports ALL dataset pairs
        if include_partner_details:
            from itertools import combinations
            partner_dir = output_dir / 'partner_details'
            partner_dir.mkdir(exist_ok=True)
            
            for neuron_type in neuron_types[:20]:  # Limit to first 20 types
                try:
                    profiles = {}
                    for dataset in datasets:
                        profiles[dataset] = self._get_profile(neuron_type, dataset)
                    
                    # Get partner overlap for ALL dataset pairs (not just first two)
                    if len(profiles) >= 2:
                        ds_list = list(profiles.keys())
                        all_overlaps = []
                        
                        # Iterate over all pairs of datasets
                        for dataset_a, dataset_b in combinations(ds_list, 2):
                            # Upstream overlap
                            overlap_up = ProfileComparator.get_partner_overlap_details(
                                profiles[dataset_a], profiles[dataset_b], 'upstream'
                            )
                            overlap_up['direction'] = 'upstream'
                            overlap_up['dataset_a'] = dataset_a
                            overlap_up['dataset_b'] = dataset_b
                            all_overlaps.append(overlap_up)
                            
                            # Downstream overlap
                            overlap_down = ProfileComparator.get_partner_overlap_details(
                                profiles[dataset_a], profiles[dataset_b], 'downstream'
                            )
                            overlap_down['direction'] = 'downstream'
                            overlap_down['dataset_a'] = dataset_a
                            overlap_down['dataset_b'] = dataset_b
                            all_overlaps.append(overlap_down)
                        
                        if all_overlaps:
                            combined = pd.concat(all_overlaps, ignore_index=True)
                            # Reorder columns to put dataset info first
                            cols = ['dataset_a', 'dataset_b', 'direction'] + [c for c in combined.columns if c not in ['dataset_a', 'dataset_b', 'direction']]
                            combined = combined[cols]
                            combined.to_csv(
                                partner_dir / f'partner_overlap_{neuron_type}.csv', 
                                index=False
                            )
                        
                except Exception as e:
                    self._log(f"Warning: Could not save partner details for {neuron_type}: {e}")
            
            self._log("Saved: partner details CSVs (all dataset pairs)")
        
        self._log(f"Verification report complete: {output_dir}")
    
    def clear_cache(self):
        """Clear the profile cache."""
        self._profile_cache.clear()
        self._log("Profile cache cleared")
