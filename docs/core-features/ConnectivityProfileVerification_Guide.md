# Connectivity Profile Verification Guide

**Version:** 4.1  
**Last Updated:** December 2025

## Overview

The Connectivity Profile Verification module enables systematic verification of neuron type assignments across datasets by comparing connectivity fingerprints. A connectivity profile captures the top upstream and downstream partners of a neuron, enabling cross-dataset validation of type labels.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Similarity Metrics](#similarity-metrics)
4. [Usage Examples](#usage-examples)
5. [HTML Report Features](#html-report-features)
6. [API Reference](#api-reference)

---

## Quick Start

### Minimal Example

```python
from comparison import ConnectivityProfiler, CrossDatasetVerifier, ProfilerConfig

# Configure profiler (v4.1 defaults)
config = ProfilerConfig(
    top_k_bodyid=5,         # Top 5 connections per partner type (default)
    top_m_type=0,           # No minimum type limit (include all)
    min_synapse_threshold=3 # Filter weak connections
)

# Create profiler and verifier
profiler = ConnectivityProfiler(
    datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
    config=config
)
verifier = CrossDatasetVerifier(profiler)

# Verify a neuron type across datasets
results = verifier.verify_type_assignment(
    'aMe12', 
    datasets=['hemibrain:v1.2.1', 'male-cns:v0.9']
)
print(results.summary())
```

---

## Core Concepts

### What is a Connectivity Profile?

A connectivity profile (or fingerprint) captures:
- **Upstream partners**: Neurons providing synaptic input (top K by weight)
- **Downstream partners**: Neurons receiving synaptic output (top K by weight)
- **Actual synapse weights**: Raw synapse counts stored for each partner (not proportions)
- **Ranks**: Ordinal rank of each partner (1 = strongest connection)

### Data Storage (v4.1)

Profiles now store **actual synapse weights** instead of proportions:

```python
profile.upstream_partners    # {'Mi1': 245.0, 'Tm3': 180.0, ...}  - actual weights
profile.upstream_ranks       # {'Mi1': 1, 'Tm3': 2, ...}  - ordinal ranks
profile.get_proportions()    # {'Mi1': 0.25, 'Tm3': 0.18, ...}  - on-demand
```

**Benefits of storing weights:**
- Enables simple bodyId→type aggregation by summing weights
- Preserves rank structure naturally after aggregation
- Proportions can be computed on-demand when needed

### BodyId vs Type-Level Profiles

Profiles can be fetched at two levels:

1. **BodyId-level** (individual neurons):
   ```python
   profile = profiler.get_profile(1234567890, dataset)  # Single neuron
   ```

2. **Type-level** (aggregated from all neurons of a type):
   ```python
   # Fetch individual profiles for all neurons of a type
   bodyid_profiles = profiler.get_bodyid_profiles_for_type('aMe12', dataset)
   
   # Aggregate into type-level profile
   type_profile = ConnectivityProfile.aggregate_bodyid_profiles(
       list(bodyid_profiles.values()), 'aMe12', dataset
   )
   ```

Aggregation works by **summing weights** across bodyIds, then recomputing ranks.

### Why Verify with Profiles?

Neurons with the same type annotation across datasets should have similar connectivity:
1. **Conservation**: Same partners with similar ranks = likely homologs
2. **Divergence**: Different partners = may be misassigned or dataset-specific
3. **Confidence**: Similarity scores quantify assignment quality

### Configuration Options

```python
from comparison import ProfilerConfig

config = ProfilerConfig(
    # Profile extraction
    top_k_bodyid=5,            # Top K connections per partner type (default: 5)
    top_m_type=0,              # Minimum unique partner types (default: 0 = no limit)
    min_synapse_threshold=3,   # Filter weak connections
    
    # Expansion behavior
    dynamic_expansion=True,    # Expand K until M types reached
    max_expansion_factor=5,    # Don't expand beyond K * factor
    
    # Normalization (for rank computation)
    normalize_method='rank',   # 'rank' (default) - ranks are always used for comparison
    
    # Caching
    use_cache=True,            # Cache profiles to disk (parquet format)
)
```

**Note:** The `normalize_method` parameter controls how ranks are computed. Proportions are not stored but can be computed on-demand using `profile.get_proportions()`.

---

## Similarity Metrics

The comparison system uses only **Jaccard similarity** and **Rank correlation** for the combined score. Cosine similarity is computed for reference but not included in the combined score (as of v4.1).

### Jaccard Similarity

Measures set overlap of partner types between two profiles.

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

**Confidence Thresholds (Updated v4.1):**
| Score | Confidence | Interpretation |
|-------|------------|----------------|
| > 0.5 | Very High | Excellent overlap, very likely same type |
| > 0.3 | High | Good overlap, likely same type |
| > 0.2 | Medium | Moderate overlap, may need review |
| > 0.1 | Low | Limited overlap, questionable assignment |
| ≤ 0.1 | Very Low | Poor overlap, likely different types |

### Rank Correlation (Spearman)

Measures similarity of partner rankings (are the same partners top-ranked in both profiles?).

The raw correlation is in range [-1, 1] and is **normalized to [0, 1]** using `(x + 1) / 2` for the combined score.

**Normalized Confidence Thresholds:**
| Normalized Score | Raw Correlation | Interpretation |
|------------------|-----------------|----------------|
| ≥ 0.925 | ≥ 0.85 | Near-identical partner ordering |
| 0.85 - 0.925 | 0.7 - 0.85 | Similar partner priorities |
| 0.75 - 0.85 | 0.5 - 0.7 | Some agreement in rankings |
| 0.65 - 0.75 | 0.3 - 0.5 | Weak correlation |
| < 0.65 | < 0.3 | Essentially uncorrelated |

### Cosine Similarity (Reference Only)

Measures weight vector similarity. Computed but **not included** in the combined score as of v4.1.

```
Cosine(A, B) = (A · B) / (||A|| × ||B||)
```

### Combined Score (v4.1)

Weighted average of Jaccard and normalized rank correlation only:

```python
DEFAULT_SCORE_WEIGHTS = {
    'jaccard': 0.50,  # Set-based overlap
    'rank': 0.50      # Rank correlation (normalized 0-1)
}

# Combined score calculation:
combined = 0.50 * jaccard + 0.50 * rank_normalized
# where rank_normalized = (rank_correlation + 1) / 2
```

**Rationale:** Jaccard captures partner overlap (what partners are shared), while rank correlation captures ordering similarity (are the same partners prioritized). These two metrics are complementary and work well for both bodyId-level and type-level comparison.

---

## Usage Examples

### Example 1: Single Type Verification

```python
from comparison import ConnectivityProfiler, CrossDatasetVerifier

# Setup
profiler = ConnectivityProfiler(datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'])
verifier = CrossDatasetVerifier(profiler)

# Verify
results = verifier.verify_type_assignment('aMe12', datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'])

# Check results
print(results.summary())
print(f"Status: {results.verification_status}")
print(f"Confidence: {results.confidence}")
print(f"Combined Score: {results.avg_combined_score:.3f}")
```

### Example 2: Batch Verification

```python
# Verify multiple types
neuron_types = ['aMe12', 'PPL101', 'KCg-d', 'MBON01', 'LHAV1a1']

batch_results = verifier.batch_verify(
    neuron_types=neuron_types,
    datasets=['hemibrain:v1.2.1', 'male-cns:v0.9']
)

# Summarize
for result in batch_results:
    status = '✅' if result.verification_status == 'verified' else '⚠️'
    print(f"{status} {result.neuron_type}: {result.confidence} ({result.avg_combined_score:.3f})")
```

### Example 3: Generate HTML Report

```python
# After batch verification
verifier.generate_html_report(
    batch_results,
    output_path='/path/to/verification_report.html',
    title='Neuron Type Verification Report'
)
```

### Example 4: Access Pairwise Scores

```python
results = verifier.verify_type_assignment('aMe12', datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'])

# Access individual comparison results
for pair in results.pairwise_scores:
    print(f"{pair.dataset_a} vs {pair.dataset_b}:")
    print(f"  Jaccard: {pair.jaccard:.3f}")
    print(f"  Cosine: {pair.cosine:.3f}")
    print(f"  Rank: {pair.rank_correlation:.3f}")
    print(f"  Combined: {pair.combined:.3f}")
```

### Example 5: Directional Analysis

```python
# Get upstream/downstream breakdown
results = verifier.verify_type_assignment(
    'aMe12',
    datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
    direction='both'  # 'upstream', 'downstream', or 'both'
)

# Access directional Jaccard (added in v4.0)
for pair in results.pairwise_scores:
    print(f"Upstream Jaccard: {pair.jaccard_upstream:.3f}")
    print(f"Downstream Jaccard: {pair.jaccard_downstream:.3f}")
```

---

## HTML Report Features

The verification HTML report includes:

### Verification Summary Table
- All verified types with combined scores
- Confidence badges (Very High, High, Medium, Low, Very Low)
- Verification status (verified, needs_review, questionable, failed)

### Jaccard Similarity Matrix
- Pairwise dataset comparison heatmap
- Separate matrices for upstream and downstream (v4.0)
- Color-coded cells by confidence level

### Directional Breakdown
- Avg Jaccard Upstream column
- Avg Jaccard Downstream column
- Helps identify input vs output connectivity differences

### Confidence Indicators
Color-coded badges:
- 🟢 **confidence-very-high**: Green (Very High)
- 🔵 **confidence-high**: Blue (High)
- 🟡 **confidence-medium**: Yellow (Medium)
- 🟠 **confidence-low**: Orange (Low)
- 🔴 **confidence-very-low**: Red (Very Low)

---

## API Reference

### ConnectivityProfile

```python
@dataclass
class ConnectivityProfile:
    neuron_id: Union[str, int]          # Type name or bodyId
    dataset: str                         # Dataset identifier
    upstream_partners: Dict[str, float]  # partner_type → actual weight
    downstream_partners: Dict[str, float]
    upstream_ranks: Dict[str, int]       # partner_type → rank (1 = strongest)
    downstream_ranks: Dict[str, int]
    
    # Methods
    def get_proportions(direction='both') -> Dict[str, float]:
        """Convert actual weights to proportions (sum to 1.0)"""
    
    @staticmethod
    def aggregate_bodyid_profiles(
        profiles: List[ConnectivityProfile],
        neuron_type: str,
        dataset: str
    ) -> ConnectivityProfile:
        """Aggregate bodyId profiles into type-level profile by summing weights"""
```

### ConnectivityProfiler

```python
ConnectivityProfiler(
    datasets: List[str],           # Dataset identifiers
    config: ProfilerConfig = None, # Configuration options
    token: str = '',               # NeuPrint API token
)

# Methods
profile = profiler.get_profile(
    neuron: Union[str, int],  # Type name or bodyId
    dataset: str,             # Dataset to query
    direction: str = 'both'   # 'upstream', 'downstream', or 'both'
)

# Get all bodyId profiles for a type (for aggregation)
profiles = profiler.get_bodyid_profiles_for_type(
    neuron_type: str,
    dataset: str
) -> Dict[int, ConnectivityProfile]

# Cache management
profiler.build_connectivity_profile_cache(dataset)
profiler.read_connectivity_profile_cache(dataset)
```

### CrossDatasetVerifier

```python
CrossDatasetVerifier(
    profiler: ConnectivityProfiler,
    comparator: ProfileComparator = None,
    score_weights: Dict[str, float] = {'jaccard': 0.50, 'rank': 0.50}
)

# Single verification
result = verifier.verify_type_assignment(
    neuron_type: str,
    datasets: List[str],
    direction: str = 'both'
)

# Batch verification
results = verifier.batch_verify(
    neuron_types: List[str],
    datasets: List[str]
)

# Report generation
verifier.generate_html_report(
    results: List[VerificationResult],
    output_path: str,
    title: str = 'Verification Report'
)
```

### ProfileComparator

```python
# Compare two profiles (works for both bodyId and type level)
result = ProfileComparator.compare_profiles(
    profile_a: ConnectivityProfile,
    profile_b: ConnectivityProfile,
    direction: str = 'both',
    weights: Dict[str, float] = None  # defaults to {'jaccard': 0.50, 'rank': 0.50}
) -> ComparisonResult

# Get combined score dictionary
scores = ProfileComparator.combined_score(
    profile_a, profile_b
)
# Returns: {'combined': 0.75, 'jaccard': 0.6, 'rank': 0.8, 'rank_norm': 0.9, ...}
```

### VerificationResult

```python
@dataclass
class VerificationResult:
    neuron_type: str                    # Type name
    datasets: List[str]                 # Compared datasets
    pairwise_scores: List[ComparisonResult]  # Individual comparisons
    avg_combined_score: float           # Average across pairs
    min_score: float                    # Minimum score
    max_score: float                    # Maximum score
    confidence: str                     # Confidence level
    verification_status: str            # 'verified', 'needs_review', etc.
```

---

## See Also

- [Cross-Dataset Comparison Guide](CrossDatasetComparison_Guide.md) - Full comparison workflows
- [Graph Similarity Metrics](GraphSimilarityMetrics_Documentation.md) - Metric calculations
- [Cache System Guide](CacheSystem_Guide.md) - Profile caching
