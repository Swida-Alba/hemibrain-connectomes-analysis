# Score and Probability Calculation Guide

This document provides comprehensive explanations of all score and probability calculations used throughout the toolkit, including their formulas, biological rationale, and where they are applied.

---

## Table of Contents

- [Overview](#overview)
- [Connection-Level Metrics](#connection-level-metrics)
  - [Connection Ratio](#connection-ratio)
  - [Traversal Probability](#traversal-probability)
  - [Block Probability](#block-probability)
- [Population-Level Aggregation](#population-level-aggregation)
  - [Type-to-Type Connection Ratio](#type-to-type-connection-ratio)
  - [Type-to-Type Traversal Probability](#type-to-type-traversal-probability)
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
- [Code References](#code-references)

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

**Definition**: The fraction of a neuron's total inputs provided by a specific upstream neuron.

**Formula**:

$$\text{connection\_ratio}_{ij} = \frac{w_{ij}}{W_j}$$

Where:
- $w_{ij}$ = Number of synapses from neuron $i$ to neuron $j$ (weight)
- $W_j$ = Total number of post-synaptic sites on neuron $j$ (post)

**Interpretation**:
- Value of `0.25` means 25% of neuron $j$'s inputs come from neuron $i$
- Higher values indicate stronger input influence
- Range: 0.0 to 1.0

**Example**:
```
Neuron A → Neuron B
Synapses (weight): 50
Total inputs to B (post): 200

connection_ratio = 50 / 200 = 0.25 (25%)
```

**Used in**:
- [`src/coana.py`](../../src/coana.py#L2736) - `_fetch_connections_with_cache()`
- [`src/statvis.py`](../../src/statvis.py#L4599) - `EnrichConnectionTable()`
- Edge filtering via `min_ratio` parameter

---

### Traversal Probability

**Definition**: The probability that a signal will traverse from one neuron to another, scaled by a biological threshold.

**Formula**:

$$p_{ij} = \min\left(1.0, \frac{w_{ij}}{W_j \times 0.3}\right) = \min\left(1.0, \frac{\text{connection\_ratio}_{ij}}{0.3}\right)$$

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

$$\text{block\_probability}_{ij} = 1 - \text{traversal\_probability}_{ij}$$

**Used in**:
- [`src/statvis.py`](../../src/statvis.py#L4607-L4609) - Path probability calculations
- Type-to-type aggregation (product method)

---

## Population-Level Aggregation

When analyzing connections between neuron **types** (populations), individual connections must be aggregated.

### Type-to-Type Connection Ratio

**Definition**: The aggregate connection strength between two neuron types.

**Formula**:

$$\text{connection\_ratio}_{AB} = \frac{\sum_{i \in A, j \in B} w_{ij}}{\sum_{j \in B} W_j}$$

Where:
- $A$ = Set of neurons of type A (presynaptic population)
- $B$ = Set of neurons of type B (postsynaptic population)
- Numerator = Total synapses from type A to type B
- Denominator = Total post-synaptic sites across all neurons of type B

**Implementation** ([`src/statvis.py`](../../src/statvis.py#L4740-L4780)):

```python
# Step 1: Sum weights for each type pair
weight_sum = conn_df.groupby([type_pre, type_post])['weight'].sum()

# Step 2: Get total post-synaptic sites for each target type
type_post_totals = target_neurons_df.groupby('type')['post'].sum()

# Step 3: Calculate ratio
conn_type['connection_ratio'] = weight_sum / type_post_totals
```

**Aggregation via matrix operations** ([`src/coana.py`](../../src/coana.py#L384-L387)):

```python
# For type-level pivot tables, use 'max' aggregation
mat_ratio = df.pivot(values='connection_ratio', index='type_pre', 
                     columns='type_post', aggregate_function='max')
```

---

### Type-to-Type Traversal Probability

**Definition**: The aggregate probability that a signal will propagate from any neuron of type A to any neuron of type B.

**Two Aggregation Methods**:

#### 1. Product Method (Default for Paths)

Models signal propagation as requiring ALL individual connections to transmit:

$$p_{AB}^{\text{product}} = 1 - \prod_{i \in A, j \in B} (1 - p_{ij})$$

**Implementation** ([`src/statvis.py`](../../src/statvis.py#L4754-L4761)):

```python
# Group block probabilities and take product
conn_traversal = conn_df.groupby([type_pre, type_post])['block_probability'].prod()
conn_type['traversal_probability'] = 1 - conn_traversal['block_probability']
```

**Interpretation**: The probability that at least one signal path succeeds between the populations.

#### 2. Average Method (for Direct Connections)

Uses weighted average of individual traversal probabilities:

$$p_{AB}^{\text{average}} = \frac{\sum_{i \in A, j \in B} w_{ij} \cdot p_{ij}}{\sum_{i \in A, j \in B} w_{ij}}$$

**Implementation** ([`src/statvis.py`](../../src/statvis.py#L4763-L4768)):

```python
# Weight-weighted average of traversal probabilities
weighted_sum = (conn_df['weight'] * conn_df['traversal_probability']).sum()
total_weight = conn_df['weight'].sum()
conn_type['traversal_probability'] = weighted_sum / total_weight
```

**When to use**:
- **Product**: Multi-hop pathways, signal cascade analysis
- **Average**: Direct connection strength assessment

---

## Path-Level Metrics

For multi-hop pathways (A → B → C → D), path-level metrics aggregate edge metrics along the entire path.

### Path Probability

**Definition**: The probability that a signal successfully traverses an entire multi-hop pathway.

**Formula**:

$$P_{\text{path}} = \prod_{k=1}^{n} p_{k,k+1}$$

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

$$w_{\text{min}} = \min_{k=1}^{n} w_{k,k+1}$$

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

$$\text{coverage\_ratio} = \frac{\text{match\_count}}{\text{total\_query\_neurons}}$$

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

$$\text{weighted\_score} = \text{agg\_mean\_score} \times \text{coverage\_ratio}$$

Or equivalently:

$$\text{weighted\_score} = \text{agg\_mean\_score} \times \frac{\text{match\_count}}{\text{total\_query\_neurons}}$$

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
   $$\text{min\_score\_per\_dataset} = \min_{d \in \text{datasets}} \max_{n \in d} \text{score}_{n}$$

2. **Cross-Dataset Score**: Average of max scores across datasets
   $$\text{cross\_dataset\_score} = \frac{1}{|D|} \sum_{d \in D} \max_{n \in d} \text{score}_{n}$$

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
