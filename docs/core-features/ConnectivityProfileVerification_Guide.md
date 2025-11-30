# Connectivity Profile Verification Guide

**Version:** 4.0  
**Last Updated:** November 2025

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

# Configure profiler
config = ProfilerConfig(
    top_k_bodyid=20,        # Top 20 connections per direction
    top_m_type=5,           # Ensure at least 5 unique partner types
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
- **Normalized weights**: Proportion of total input/output for each partner

### Why Verify with Profiles?

Neurons with the same type annotation across datasets should have similar connectivity:
1. **Conservation**: Same partners with similar weights = likely homologs
2. **Divergence**: Different partners = may be misassigned or dataset-specific
3. **Confidence**: Similarity scores quantify assignment quality

### Configuration Options

```python
from comparison import ProfilerConfig

config = ProfilerConfig(
    # Profile extraction
    top_k_bodyid=20,           # Top K connections by body ID
    top_m_type=5,              # Minimum unique partner types
    min_synapse_threshold=3,   # Filter weak connections
    
    # Expansion behavior
    dynamic_expansion=True,    # Expand K until M types reached
    max_expansion_factor=5,    # Don't expand beyond K * factor
    
    # Normalization
    normalize_method='proportion',  # 'proportion', 'rank', or 'both'
    
    # Caching
    use_cache=True,            # Cache profiles to disk
)
```

---

## Similarity Metrics

### Jaccard Similarity

Measures set overlap of partner types between two profiles.

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

**Confidence Thresholds (Updated v4.0):**
| Score | Confidence | Interpretation |
|-------|------------|----------------|
| > 0.5 | Very High | Excellent overlap, very likely same type |
| > 0.3 | High | Good overlap, likely same type |
| > 0.2 | Medium | Moderate overlap, may need review |
| > 0.1 | Low | Limited overlap, questionable assignment |
| ≤ 0.1 | Very Low | Poor overlap, likely different types |

### Rank Correlation (Spearman)

Measures similarity of partner rankings (are the same partners top-ranked in both profiles?).

**Confidence Thresholds:**
| Score | Confidence | Interpretation |
|-------|------------|----------------|
| ≥ 0.85 | Very High | Near-identical partner ordering |
| 0.7 - 0.85 | High | Similar partner priorities |
| 0.5 - 0.7 | Medium | Some agreement in rankings |
| 0.3 - 0.5 | Low | Weak correlation |
| < 0.3 | Very Low | Essentially uncorrelated |

### Cosine Similarity

Measures weight vector similarity (do shared partners have similar weights?).

```
Cosine(A, B) = (A · B) / (||A|| × ||B||)
```

### Combined Score

Weighted average of all metrics:
```python
DEFAULT_SCORE_WEIGHTS = {
    'jaccard': 0.30,
    'cosine': 0.35,
    'rank': 0.35
}
```

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

### ConnectivityProfiler

```python
ConnectivityProfiler(
    datasets: List[str],           # Dataset identifiers
    config: ProfilerConfig = None, # Configuration options
    token: str = '',               # NeuPrint API token
)

# Methods
profile = profiler.get_profile(
    neuron_type: str,     # Type name to profile
    dataset: str,         # Dataset to query
    direction: str = 'both'  # 'upstream', 'downstream', or 'both'
)
```

### CrossDatasetVerifier

```python
CrossDatasetVerifier(
    profiler: ConnectivityProfiler,
    comparator: ProfileComparator = None,
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
