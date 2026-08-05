# Score and Probability Calculation Guide

This document provides comprehensive explanations of all score and probability calculations used throughout the toolkit, including their formulas, biological rationale, and where they are applied.

---

## Table of Contents

- [Score and Probability Calculation Guide](#score-and-probability-calculation-guide)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Connection-Level Metrics](#connection-level-metrics)
    - [Connection Ratio](#connection-ratio)
    - [Traversal Probability](#traversal-probability)
    - [Block Probability](#block-probability)
  - [Population-Level Aggregation](#population-level-aggregation)
    - [Type-to-Type Connection Ratio](#type-to-type-connection-ratio)
    - [Type-to-Type Traversal Probability](#type-to-type-traversal-probability)
      - [1. Product Method (Default for Paths)](#1-product-method-default-for-paths)
      - [2. Average Method (for Direct Connections)](#2-average-method-for-direct-connections)
  - [Path-Level Metrics](#path-level-metrics)
    - [Path Probability](#path-probability)
    - [Minimum Edge Weight](#minimum-edge-weight)
    - [Path Ranking](#path-ranking)
  - [NeuronBridge Matching Scores](#neuronbridge-matching-scores)
    - [Match Score](#match-score)
    - [Coverage Ratio](#coverage-ratio)
    - [Weighted Score](#weighted-score)
    - [Cross-Dataset Score](#cross-dataset-score)
  - [Filter Parameters Summary](#filter-parameters-summary)
  - [Threshold Selection Guide](#threshold-selection-guide)
    - [Connection Ratio (`min_ratio`)](#connection-ratio-min_ratio)
    - [Traversal Probability (`min_traversal_probability`)](#traversal-probability-min_traversal_probability)
    - [Conversion Formulas](#conversion-formulas)
  - [Code References](#code-references)
    - [Core Calculation Locations](#core-calculation-locations)
    - [Parameter Definitions](#parameter-definitions)
    - [Related Documentation](#related-documentation)
  - [Summary](#summary)

---

## Overview

The toolkit uses several metrics to quantify connection strength and signal propagation probability in the connectome:

| Metric                    | Formula                      | Range     | Level |
| ------------------------- | ---------------------------- | --------- | ----- |
| **Connection Ratio**      | `weight / post`              | 0.0 - 1.0 | Edge  |
| **Traversal Probability** | `min(1.0, ratio / 0.3)`      | 0.0 - 1.0 | Edge  |
| **Path Probability**      | `∏(edge probabilities)`      | 0.0 - 1.0 | Path  |
| **Weighted Score**        | `avg_score × coverage_ratio` | 0.0 - ∞   | Line  |

---

## Connection-Level Metrics

These metrics are calculated for each individual synaptic connection (edge) in the network.

### Connection Ratio

**Definition**: The fraction of a neuron's **total input from ALL sources in the dataset** provided by a specific upstream connection.

**Formula (Global Ratio)**:

$$\mathit{ratio}_{ij}^{(t)} = \frac{w_{ij}^{(t)}}{\sum_{\forall k \in \text{dataset}} w_{kj}^{(t)}}$$

Where:
- $w_{ij}^{(t)}$ = Number of synapses from neuron $i$ to neuron $j$ passing threshold $t$
- $\sum_{\forall k \in \text{dataset}} w_{kj}^{(t)}$ = Total incoming synapses to neuron $j$ from **ALL neurons in the dataset** at threshold $t$

**Why Global Ratio?**

The ratio is calculated using the **entire connection cache**, not just the neurons in the current query:
- Denominator includes ALL incoming connections to B, from ANY source neuron
- This gives the true biological fraction "how much of B's total input comes from A"
- Without global calculation, ratios would be inflated when only a subset of source neurons is queried

**Interpretation**:
- Value of `0.01` means 1% of neuron $j$'s **total input** comes from neuron $i$
- Higher values indicate stronger relative input influence
- Range: 0.0 to 1.0 (NaN if no incoming connections)

**Example**:
```
Query: aMe12 → KCg-d at threshold = 3 synapses

aMe12 → KCg-d: 49 synapses (the connection we're analyzing)
All other neurons → KCg-d: 4951 synapses (from connection cache)
Total incoming to KCg-d: 49 + 4951 = 5000

connection_ratio(aMe12→KCg-d) = 49 / 5000 = 0.0098 (0.98%)

Note: If we only counted from provided source neurons, we'd get a misleading
ratio like 0.98 (98%) which doesn't represent true biological significance.
```

**Used in**:
- [`src/coana.py`](../../src/coana.py) - `_fetch_total_incoming_weight()`, `_apply_bodyid_level_filters()`, `_apply_type_level_filters()`
- [`src/statvis.py`](../../src/statvis.py) - `EnrichConnectionTable()` (preserves pre-calculated ratios)
- Edge filtering via `min_ratio` parameter

---

### Traversal Probability

**Definition**: The probability that a signal will traverse from one neuron to another, scaled by a biological threshold.

**Formula**:

$$p_{ij} = \min\left(1.0, \frac{\mathit{ratio}_{ij}^{(t)}}{0.3}\right)$$

**The 0.3 Biological Threshold**:

The 0.3 scaling factor represents a **30% input threshold**—connections providing ≥30% of a neuron's inputs are considered maximally effective for signal propagation. This is based on neurobiological observations that:
- Connections exceeding ~30% of post-synaptic input typically reliably activate the downstream neuron
- Below this threshold, activation probability scales roughly linearly with input fraction

**Examples**:

| Scenario    | Weight | Post | Ratio | Traversal Prob |
| ----------- | ------ | ---- | ----- | -------------- |
| Weak        | 2      | 1000 | 0.2%  | 0.67%          |
| Moderate    | 10     | 500  | 2%    | 6.7%           |
| Strong      | 50     | 200  | 25%   | 83.3%          |
| Very Strong | 100    | 200  | 50%   | 100% (capped)  |

**Used in**:
- [`src/coana.py`](../../src/coana.py#L2737-L2738) - Calculation and capping
- [`src/statvis.py`](../../src/statvis.py#L4601-L4603) - `EnrichConnectionTable()`
- Edge filtering via `min_traversal_probability` parameter
- Sankey diagrams and network visualizations

---

### Block Probability

**Definition**: The probability that a signal will NOT traverse a connection.

**Formula**:

$$\mathit{block}_{ij} = 1 - \mathit{p}_{ij}$$

**Used in**:
- [`src/statvis.py`](../../src/statvis.py#L4607-L4609) - Path probability calculations
- Type-level enrichment output (`block_probability = 1 - p_AB`)

---

## Population-Level Aggregation

When analyzing connections between neuron **types** (populations), individual connections must be aggregated.

### Type-to-Type Connection Ratio

**Definition**: The aggregate connection strength between two neuron types, using total input from ALL sources in the dataset.

**Formula (Global Ratio)**:

$$\mathit{ratio}_{AB}^{(t)} = \frac{\sum_{i \in A, j \in B} w_{ij}^{(t)}}{\sum_{\forall k \in \text{dataset}, j \in B} w_{kj}^{(t)}}$$

Where:
- $A$ = Set of neurons of type A (presynaptic population)
- $B$ = Set of neurons of type B (postsynaptic population)
- Numerator = Total synapses from type A to type B at threshold $t$
- Denominator = Total incoming synapses to type B from **ALL neuron types in the dataset** at threshold $t$

**Key Point**: The denominator includes ALL incoming connections to type B from any neuron type in the entire dataset (not just the source types in your query), giving the true fraction of type B's total input that comes from type A.

**Implementation** ([`src/coana.py`](../../src/coana.py)):

```python
# Step 1: Sum weights for each type pair
weight_sum = conn_df.groupby([type_pre, type_post])['weight'].sum()

# Step 2: Get total incoming weight for each target type from ENTIRE CACHE
# This queries the connection database, not just the current connections
total_incoming_per_type = self._fetch_total_incoming_weight_by_type(post_types, min_weight)

# Step 3: Calculate global ratio
conn_type['connection_ratio'] = weight_sum / total_incoming_per_type
```

**Aggregation via matrix operations** ([`src/coana.py`](../../src/coana.py)):

```python
# For type-level pivot tables, use 'max' aggregation
mat_ratio = df.pivot(values='connection_ratio', index='type_pre', 
                     columns='type_post', aggregate_function='max')
```

---

### Type-to-Type Traversal Probability

**Definition**: The aggregate probability that a signal will propagate from any neuron of type A to any neuron of type B.

**Formula (current model, consistent at every level)**:

$$p_{AB} = \min\left(1.0, \frac{\mathit{ratio}_{AB}}{0.3}\right)$$

The type-level traversal probability uses the same `ratio / 0.3` scaling as the
bodyId level (see [Traversal Probability](#traversal-probability)), applied to
the **global** type-to-type ratio above. This keeps the filter thresholds
(`min_traversal_probability`), the enrichment output, and the path metrics on
one consistent scale.

**Implementation**:

- [`src/coana.py`](../../src/coana.py) - `_apply_type_level_filters()` (type-level filtering)
- [`src/statvis.py`](../../src/statvis.py) - `EnrichConnectionTable()` (with `global_incoming_weights`)
- [`src/statvis.py`](../../src/statvis.py) - `EnrichConnectionTable()` (unified entry; polars engine via `engine='auto'`, with `global_incoming_weights`)

**Note (legacy)**: Earlier versions offered 'product'
($1 - \prod_{i \in A, j \in B}(1 - p_{ij})$) and 'average' (weight-weighted mean)
aggregations of bodyId-level probabilities via the `aggregate_method` parameter.
These were replaced by the uniform `min(ratio / 0.3, 1)` model so that the two
implementations (pandas/Polars) produce identical numbers. `aggregate_method` is
still accepted for API compatibility but no longer changes the result.

---

## Path-Level Metrics

For multi-hop pathways (A → B → C → D), path-level metrics aggregate edge metrics along the entire path.

### Path Probability

**Definition**: The probability that a signal successfully traverses an entire multi-hop pathway.

**Formula**:

$$P_\mathrm{path} = \prod_{k=1}^{n} p_{k,k+1}$$

Where the path has $n$ edges with individual traversal probabilities $p_{k,k+1}$.

**Example**:
```
Path: A → B → C → D
Edge probabilities: [0.8, 0.6, 0.9]
Path probability: 0.8 × 0.6 × 0.9 = 0.432 (43.2%)
```

**Implementation** ([`src/statvis.py`](../../src/statvis.py#L4199)):

```python
row = {
    'path_prob': np.prod(probs) if probs else 0,
    'min_weight': min(weights) if weights else 0,
    'min_ratio': min(ratios) if ratios else 0,
    'length': len(path) - 1
}
```

**Note on Path Length**: Longer paths have exponentially lower probabilities:
- 3 edges at 0.5 each: $0.5^3 = 0.125$
- 6 edges at 0.5 each: $0.5^6 = 0.016$

---

### Minimum Edge Weight

**Definition**: The "bottleneck" of a path—the weakest connection along the route.

**Formula**:

$$w_\mathrm{min} = \min_{k=1}^{n} w_{k,k+1}$$

**Used for**: Identifying the limiting factor in signal transmission.

---

### Path Ranking

Paths are ranked by `path_prob` (descending) for output files and visualizations:

**Implementation** ([`src/coana.py`](../../src/coana.py#L6581-L6585)):

```python
if 'path_prob' in df_paths.columns:
    sort_cols.append('path_prob')
elif 'path_probability' in df_paths.columns:
    sort_cols.append('path_probability')
```

The `show_top_n_paths` parameter limits output to the highest-probability paths.

---

## NeuronBridge Matching Scores

These scores are used for EM↔LM mapping via NeuronBridge.

### Match Score

**Definition**: The raw morphological similarity score from NeuronBridge's Color Depth MIP (CDM) search algorithm.

**Provided by**: NeuronBridge API (not calculated locally)

**Range**: Typically 0 - 500,000+ (higher = better match)

**Types**:
- `cds_score`: Color Depth MIP Search score
- `pppm_score`: Point Pattern Matching score
- `combined_rank`: Rank when using both algorithms

---

### Coverage Ratio

**Definition**: The fraction of queried neurons that a driver line labels.

**Formula**:

$$\mathit{coverage} = \frac{\mathit{matches}}{\mathit{total}}$$

Where:
- `match_count` = Number of unique queried neurons this line matches
- `total_query_neurons` = Total number of neurons in the query

**Example**:
```
Query: ["aMe12", "MBON01", "KC"] (3 neurons)
Line SS01234 matches: aMe12, MBON01 (2 neurons)

coverage_ratio = 2 / 3 = 0.667 (66.7%)
```

**Implementation** ([`src/neuronbridge_finder.py`](../../src/neuronbridge_finder.py#L7073)):

```python
line_stats['coverage_ratio'] = line_stats['match_count'] / total_query_neurons
```

---

### Weighted Score

**Definition**: A composite score prioritizing lines that label more of the queried neurons with high morphological similarity.

**Formula**:

$$\mathit{weighted} = \mathit{mean} \times \mathit{coverage}$$

Or equivalently:

$$\mathit{weighted} = \mathit{mean} \times \frac{\mathit{matches}}{\mathit{total}}$$

**Sorting Modes** (controlled by `sort_by` parameter):

1. **`sort_by='max'`** (default) - Sort by average score
   - Formula: Sort by `agg_mean_score` (descending)
   - Prioritizes: Lines with highest morphological similarity
   - Use when: Finding best morphological matches regardless of coverage

2. **`sort_by='completeness'`** - Sort by weighted score
   - Formula: Sort by `weighted_score` (descending)
   - Prioritizes: Lines labeling MORE queried neurons
   - Use when: Finding lines that label ALL queried neuron types

**Interpretation**:
- A line labeling ALL N queried neurons with score S gets `weighted_score = S`
- A line labeling only 1 of N neurons with score S gets `weighted_score = S/N`

**Example**:
```
Query: 3 neurons

sort_by='max':
  Line A: avg_score=100,000, matches 3/3 → ranks #1 (high score)
  Line B: avg_score=80,000, matches 3/3 → ranks #2

sort_by='completeness':
  Line B: weighted_score=80,000 × 1.0 = 80,000 → ranks #1 (labels all)
  Line A: weighted_score=100,000 × 0.33 = 33,000 → ranks #2 (if only 1/3)
```

**Implementation** ([`src/neuronbridge_finder.py`](../../src/neuronbridge_finder.py#L7191-L7193)):

```python
line_stats['coverage_ratio'] = line_stats['match_count'] / total_query_neurons
line_stats['weighted_score'] = line_stats['agg_mean_score'] * line_stats['coverage_ratio']

# Sorting
if sort_by == 'completeness':
    line_stats = line_stats.sort_values('weighted_score', ascending=False)
else:  # sort_by == 'max'
    line_stats = line_stats.sort_values('agg_mean_score', ascending=False)
```

**Used in**: `find_lines_batch()` output ranking

---

### Cross-Dataset Score

**Definition**: For multi-dataset queries, measures how consistently a line labels neurons across different datasets.

**Formulas**:

1. **Min Score per Dataset**: Worst-case performance across datasets
   $$\mathit{minScore} = \min_{d \in \mathit{datasets}} \max_{n \in d} score_{n}$$

2. **Cross-Dataset Score**: Average of max scores across datasets
   $$\mathit{crossScore} = \frac{1}{|D|} \sum_{d \in D} \max_{n \in d} score_{n}$$

**Used for**: Finding lines that reliably label homologous neurons in multiple connectome datasets.

---

## Filter Parameters Summary

| Parameter                   | Level | Formula Check           | Description                     |
| --------------------------- | ----- | ----------------------- | ------------------------------- |
| `min_synapse_num`           | Edge  | `weight >= X`           | Minimum synapse count           |
| `min_ratio`                 | Edge  | `connection_ratio >= X` | Minimum input fraction          |
| `min_traversal_probability` | Edge  | `traversal_prob >= X`   | Minimum propagation probability |

**Filtering Order** ([`src/coana.py`](../../src/coana.py)):

```python
# Applied BEFORE pathfinding (edge-level pre-filtering)
1. Fetch connections (weight ≥ 1)
2. Calculate connection_ratio, traversal_probability
3. Filter: weight >= min_synapse_num
4. Filter: connection_ratio >= min_ratio
5. Filter: traversal_probability >= min_traversal_probability
6. Build graph with filtered edges
7. Find paths
```

---

## Threshold Selection Guide

### Connection Ratio (`min_ratio`)

| Threshold | Meaning         | Use Case             |
| --------- | --------------- | -------------------- |
| 0.0       | No filter       | Exploratory analysis |
| 0.001     | ≥0.1% of inputs | Very permissive      |
| 0.01      | ≥1% of inputs   | Moderate filtering   |
| 0.03      | ≥3% of inputs   | Significant inputs   |
| 0.05      | ≥5% of inputs   | Strong connections   |
| 0.1       | ≥10% of inputs  | Major inputs only    |

### Traversal Probability (`min_traversal_probability`)

| Threshold | Equiv. Ratio | Recommendation            |
| --------- | ------------ | ------------------------- |
| 0.001     | 0.03%        | Conservative (most paths) |
| 0.01      | 0.3%         | Moderate (balanced)       |
| 0.05      | 1.5%         | Strict (strong paths)     |
| 0.1       | 3%           | Very strict               |

### Conversion Formulas

```python
# ratio → probability
min_traversal_probability = min_ratio / 0.3

# probability → ratio
min_ratio = min_traversal_probability * 0.3
```

---

## Code References

### Core Calculation Locations

| Metric                  | File                         | Function/Line                                                        |
| ----------------------- | ---------------------------- | -------------------------------------------------------------------- |
| `connection_ratio`      | `src/coana.py`               | [`_fetch_connections_with_cache()`](../../src/coana.py#L2736)        |
| `traversal_probability` | `src/coana.py`               | [`_fetch_connections_with_cache()`](../../src/coana.py#L2737-L2738)  |
| `path_prob`             | `src/statvis.py`             | [`process_paths()`](../../src/statvis.py#L4199)                      |
| Type aggregation        | `src/statvis.py`             | [`EnrichConnectionTable()`](../../src/statvis.py#L4754-L4780)        |
| `weighted_score`        | `src/neuronbridge_finder.py` | [`find_lines_batch()`](../../src/neuronbridge_finder.py#L7191-L7193) |

### Parameter Definitions

| Parameter                   | File           | Line                                      |
| --------------------------- | -------------- | ----------------------------------------- |
| `min_ratio`                 | `src/coana.py` | [L528-L533](../../src/coana.py#L528-L533) |
| `min_traversal_probability` | `src/coana.py` | [L535-L541](../../src/coana.py#L535-L541) |
| `filter_by`                 | `src/coana.py` | [L543-L550](../../src/coana.py#L543-L550) |

### Related Documentation

- [Connection Ratio Filter](./ConnectionRatio_Filter.md) - Detailed `min_ratio` guide
- [Traversal Probability Edge Filter](./TraversalProbability_EdgeLevelFilter.md) - Edge-level filtering
- [NeuronBridge Guide](./NeuronBridge_Guide.md) - EM↔LM matching scores
- [PathFinding Methods](./PathFinding_Methods.md) - Algorithm selection

---

## Summary

| Context               | Primary Metric          | Formula                   |
| --------------------- | ----------------------- | ------------------------- |
| **Single connection** | `traversal_probability` | `min(1, weight/post/0.3)` |
| **Type→Type direct**  | `connection_ratio`      | `Σweights / Σposts`       |
| **Type→Type path**    | `traversal_probability` | `1 - Π(1-p_edge)`         |
| **Multi-hop path**    | `path_prob`             | `Π(p_edge)`               |
| **Driver line rank**  | `weighted_score`        | `avg_score × coverage`    |

---

*Last updated: January 2026*
