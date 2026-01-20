# Graph Similarity Metrics Documentation

This document provides comprehensive documentation for all graph similarity metrics used in the inter-dataset comparison module.

## Overview

The comparison module provides **8 complementary similarity metrics** organized into three categories:

### Active Metrics (Used in Current Implementation)

| Category     | Metric         | Range   | Weight-Sensitive | Scale-Sensitive             | Description                               |
| ------------ | -------------- | ------- | ---------------- | --------------------------- | ----------------------------------------- |
| **Topology** | Jaccard        | [0, 1]  | ❌ No             | N/A                         | Binary edge overlap                       |
| **Topology** | GED            | [0, 1]  | ⚠️ Partial        | ⚠️ Partial                   | Graph edit distance                       |
| **Topology** | WL Kernel      | [0, 1]  | ⚠️ Partial        | N/A                         | Local neighborhood patterns               |
| **Union**    | Edge Rank      | [-1, 1] | ✅ Yes            | ❌ No (rank-based)           | Spearman correlation on union of edges    |
| **Union**    | Cosine         | [0, 1]  | ✅ Yes            | ❌ No (normalized)           | Cosine similarity on union of edges       |
| **Shared**   | Spearman Rank  | [-1, 1] | ✅ Yes            | ❌ No (rank-based)           | Spearman correlation on shared edges only |
| **Matrix**   | Ruzicka        | [0, 1]  | ✅ Yes            | ❌ No (inherently invariant) | Weight-aware edge overlap                 |
| **Matrix**   | RV Coefficient | [0, 1]  | ✅ Yes            | ❌ No (normalized)           | Multivariate matrix similarity            |

### Deprecated Metrics (Not Used)

| Metric                  | Reason for Deprecation                                         |
| ----------------------- | -------------------------------------------------------------- |
| **Frobenius**           | Too sensitive to absolute scale differences between datasets   |
| **Pearson Correlation** | Diluted by zeros in sparse adjacency matrices (union approach) |

---

## Important: Raw vs Normalized Correlation Values

### Cross-Dataset Comparison (Raw Values)

In the **cross-dataset comparison** module, rank correlation metrics return **raw Spearman correlation values** in the range **[-1, 1]**:

- **-1**: Perfect negative correlation (inverse relationship)
- **0**: No correlation
- **+1**: Perfect positive correlation (identical ranking)

This preserves the full statistical meaning of the correlation coefficient, allowing users to identify both positive and negative relationships between datasets.

### Homolog Finding & Connectivity Profiling (Normalized Values)

In the **homolog finding** and **connectivity profiling** modules, correlation values are **normalized to [0, 1]** using the formula:

$$S_{normalized} = \frac{\rho + 1}{2}$$

This normalization is appropriate in these contexts because:
1. Negative correlations indicate dissimilar patterns (not useful for homolog matching)
2. A uniform [0, 1] scale allows combining with other metrics for composite scoring
3. The focus is on "similarity" rather than "correlation direction"

### Choosing the Right Value

| Use Case                 | Value Type | Range   | Method Parameter            |
| ------------------------ | ---------- | ------- | --------------------------- |
| Cross-dataset comparison | Raw        | [-1, 1] | `normalize=False` (default) |
| Homolog finding          | Normalized | [0, 1]  | `normalize=True`            |
| Connectivity profiling   | Normalized | [0, 1]  | `normalize=True`            |

---

## NaN Handling for Undefined Cases

Several metrics can return `NaN` (Not a Number) for undefined cases:

| Metric                    | Returns NaN When            | Reason                             |
| ------------------------- | --------------------------- | ---------------------------------- |
| **Edge Rank Correlation** | Fewer than 3 non-zero edges | Spearman correlation undefined     |
| **Path Rank Correlation** | Fewer than 3 common paths   | Spearman correlation undefined     |
| **Cosine Similarity**     | Both vectors are zero       | Division by zero (0·0/0 undefined) |
| **Spearman Rank**         | Fewer than 3 shared edges   | Correlation undefined              |

### How NaN Values Are Handled

1. **Heatmaps**: NaN values are displayed as "N/A" with a gray background
2. **Trend Plots**: NaN values are filtered out from traces and averages
3. **Average Calculations**: NaN values are excluded from the average computation
4. **Export Data**: NaN values are preserved as `NaN` in CSV/DataFrame exports

---

## Category 1: Topology-Based Metrics

These metrics compare **graph structure only**, treating edges as present or absent regardless of weight magnitude.

### 1.1 Jaccard Similarity (Binary)

#### Formula

$$J(A, B) = \frac{|E_A \cap E_B|}{|E_A \cup E_B|}$$

Where:
- $E_A$ = set of edges in graph A (above threshold)
- $E_B$ = set of edges in graph B (above threshold)

#### Properties

| Property             | Value                                                |
| -------------------- | ---------------------------------------------------- |
| **Range**            | [0, 1]                                               |
| **Interpretation**   | 1.0 = identical edge sets, 0.0 = no overlap          |
| **Weight-sensitive** | ❌ No - treats all edges equally once above threshold |
| **Scale-sensitive**  | N/A (binary)                                         |
| **Speed**            | ⚡ Fast                                               |

#### Pros & Cons

| ✅ Pros                         | ❌ Cons                            |
| ------------------------------ | --------------------------------- |
| Simple and interpretable       | Ignores weight information        |
| Fast computation               | Threshold-dependent               |
| Good for structural comparison | Strong/weak edges treated equally |

#### When to Use

- Quick structural comparison
- When edge weights are unreliable
- With appropriate threshold to filter noise

---

### 1.2 Graph Edit Distance (GED) Similarity

#### Formula

$$S_{GED} = 1 - \frac{GED(G_A, G_B)}{|V_A| + |V_B| + |E_A| + |E_B|}$$

Where:
- $GED(G_A, G_B)$ = minimum edit operations to transform $G_A$ into $G_B$
- Edit operations: insert/delete node, insert/delete edge

#### Properties

| Property             | Value                                               |
| -------------------- | --------------------------------------------------- |
| **Range**            | [0, 1]                                              |
| **Interpretation**   | 1.0 = identical graphs, lower = more edits needed   |
| **Weight-sensitive** | ⚠️ Partial - edge matching uses 10% weight tolerance |
| **Scale-sensitive**  | ⚠️ Partial - tolerance is relative                   |
| **Speed**            | 🐢 Slow (NP-hard, may timeout)                       |

#### Pros & Cons

| ✅ Pros                              | ❌ Cons                       |
| ----------------------------------- | ---------------------------- |
| Intuitive "distance" interpretation | Computationally expensive    |
| Captures node+edge structure        | May timeout for large graphs |
| Partial weight awareness            | Approximation for >20 nodes  |

#### When to Use

- Small graphs (<20 nodes)
- When you need intuitive "how different?" measure
- Detailed structural comparison

---

### 1.3 Weisfeiler-Lehman (WL) Kernel Similarity

#### Background

The WL kernel compares graphs by iteratively aggregating neighborhood information. Each node's label is updated based on its neighbors' labels, creating a "fingerprint" of local structure.

Edge weights are **discretized into 5 bins** (percentile-based) and included in neighbor aggregation, making the kernel partially weight-aware.

#### Properties

| Property             | Value                                       |
| -------------------- | ------------------------------------------- |
| **Range**            | [0, 1]                                      |
| **Interpretation**   | 1.0 = identical neighborhood patterns       |
| **Weight-sensitive** | ⚠️ Partial - via edge weight bins (5 levels) |
| **Scale-sensitive**  | ❌ No (percentile-based bins)                |
| **Speed**            | 🟡 Medium                                    |

#### Baseline Value (~0.25)

When graphs share node vocabulary but have different edges, WL kernel shows ~0.25 similarity because:
- Iteration 0 features (node labels) are shared
- Later iteration features differ due to different neighborhoods

#### Pros & Cons

| ✅ Pros                      | ❌ Cons                              |
| --------------------------- | ----------------------------------- |
| Captures multi-hop patterns | Coarse weight discretization        |
| Scale-invariant             | Baseline ~0.25 for different graphs |
| Fast for large graphs       | Less interpretable than Jaccard     |

#### When to Use

- Comparing local neighborhood structure
- When multi-hop patterns matter
- Large graphs where GED would timeout

---

## Category 2: Union-Based Metrics (Cross-Dataset Comparison)

These metrics compare edge weights using the **union of all edges** from both graphs. Missing edges are assigned weight 0. This approach captures both:
- How well the edge sets overlap
- How well the weight rankings agree

### 2.1 Edge Rank Correlation

#### Formula

$$\rho_{edge} = \text{spearman}(R_A^{union}, R_B^{union})$$

Where:
- $R_A^{union}$ = ranks of weights in graph A for all edges in union
- $R_B^{union}$ = ranks of weights in graph B for all edges in union
- Missing edges have weight 0 (lowest rank)

#### Properties

| Property             | Value                                                            |
| -------------------- | ---------------------------------------------------------------- |
| **Range**            | [-1, 1] (raw) or [0, 1] (normalized)                             |
| **Interpretation**   | +1 = identical ranking, 0 = no correlation, -1 = inverse ranking |
| **Weight-sensitive** | ✅ Yes - uses weight ranks                                        |
| **Scale-sensitive**  | ❌ No (rank-based)                                                |
| **Speed**            | ⚡ Fast                                                           |
| **NaN condition**    | Fewer than 3 non-zero edges                                      |

#### How Missing Edges Are Handled

```
Graph A edges: {A→B: 100, B→C: 50}
Graph B edges: {A→B: 80, C→D: 60}

Union edges: {A→B, B→C, C→D}
Weights A: [100, 50, 0]   → Ranks: [3, 2, 1]
Weights B: [80, 0, 60]    → Ranks: [3, 1, 2]

Spearman correlation computed on these rank vectors
```

#### Raw vs Normalized Values

| Context                  | Value Type | Formula          | Range   |
| ------------------------ | ---------- | ---------------- | ------- |
| Cross-dataset comparison | Raw        | $\rho$           | [-1, 1] |
| Homolog finding          | Normalized | $(\rho + 1) / 2$ | [0, 1]  |

#### Pros & Cons

| ✅ Pros                            | ❌ Cons                          |
| --------------------------------- | ------------------------------- |
| Captures both overlap and ranking | Zeros affect ranking            |
| Scale-invariant                   | Penalizes non-overlapping edges |
| Full statistical interpretation   | Returns NaN for sparse data     |

#### When to Use

- Cross-dataset comparison where you want to see how well rankings agree
- When unique edges should contribute to the comparison
- When negative correlations are meaningful

---

### 2.2 Cosine Similarity

#### Formula

$$\cos(A, B) = \frac{A \cdot B}{||A|| \cdot ||B||} = \frac{\sum_{e} W_A(e) W_B(e)}{\sqrt{\sum_{e} W_A(e)^2} \cdot \sqrt{\sum_{e} W_B(e)^2}}$$

Where:
- $W_A(e)$ = weight of edge $e$ in graph A (0 if missing)
- Summation over all edges in union

#### Properties

| Property             | Value                                                              |
| -------------------- | ------------------------------------------------------------------ |
| **Range**            | [0, 1]                                                             |
| **Interpretation**   | 1.0 = identical direction (proportional weights), 0.0 = orthogonal |
| **Weight-sensitive** | ✅ Yes - uses actual weight values                                  |
| **Scale-sensitive**  | ❌ No (inherently normalized by magnitude)                          |
| **Speed**            | ⚡ Fast                                                             |
| **NaN condition**    | Both vectors are zero (no edges in either graph)                   |

#### Why Scale-Invariant?

Cosine similarity only measures the **angle** between weight vectors, not their magnitude:

```
A = [100, 50, 20]
B = [200, 100, 40]  (2x scale of A)

cos(A, B) = 1.0  (identical direction despite different magnitudes)
```

#### Special Case: Zero Vectors

When **both graphs have no edges** (both vectors are zero), the cosine similarity is **undefined** (division by zero). In this implementation, we return `NaN` rather than an arbitrary value:

```python
if np.allclose(norm_a, 0) and np.allclose(norm_b, 0):
    return np.nan  # Both vectors are zero - undefined
```

#### Pros & Cons

| ✅ Pros                     | ❌ Cons                              |
| -------------------------- | ----------------------------------- |
| Inherently scale-invariant | Only considers angle, not magnitude |
| Intuitive interpretation   | Zeros in vector reduce similarity   |
| Fast computation           | NaN for both-zero case              |

#### When to Use

- When datasets have very different total synapse counts
- When you care about relative weight proportions
- As complement to Jaccard (topology) and Edge Rank (ranking)

---

## Category 3: Shared-Edge Metrics

These metrics compare edge weights using **only the edges present in both graphs** (intersection). This avoids the problem of zeros dominating the comparison.

### 3.1 Spearman Rank Correlation (Shared Edges)

#### Formula

$$\rho_{shared} = \text{spearman}(R_A^{shared}, R_B^{shared})$$

Where $R_A^{shared}, R_B^{shared}$ are the ranks of weights for **shared edges only**.

#### Key Design Choice: Shared Edges Only

The implementation uses **SHARED edges only** (not union) to avoid the problem where many (0, weight) comparisons dilute the correlation coefficient:

```python
# Union approach (problematic):
# Many edges: (0, 50), (0, 30), (0, 25), (100, 80), (50, 40)
# Correlation dominated by zeros

# Shared approach (current):
# Only compare edges present in BOTH graphs
# (100, 80), (50, 40), (30, 35) → meaningful rank comparison
```

#### Properties

| Property             | Value                                         |
| -------------------- | --------------------------------------------- |
| **Range**            | [-1, 1] (raw) or [0, 1] (normalized)          |
| **Interpretation**   | +1 = identical ranking of shared edges        |
| **Weight-sensitive** | ✅ Yes - uses weight ranks                     |
| **Scale-sensitive**  | ❌ No (rank-based, inherently scale-invariant) |
| **Speed**            | ⚡ Fast                                        |
| **NaN condition**    | Fewer than 3 shared edges                     |

#### Pros & Cons

| ✅ Pros                   | ❌ Cons                         |
| ------------------------ | ------------------------------ |
| Scale-invariant          | Ignores unique edges           |
| Robust to outliers       | Only compares shared edges     |
| No zero-dilution problem | Returns NaN if <3 shared edges |

#### When to Use

- When you want to compare "relative importance" of edges
- When datasets have different annotation depths
- When unique edges should NOT affect the comparison

---

## Category 4: Matrix-Based Metrics

These metrics compare **edge weight magnitudes** using the full weight information. They use normalized weights to handle different scales across datasets.

### 4.1 Ruzicka Similarity (Weighted Jaccard)

#### Formula

$$R(A, B) = \frac{\sum_{e \in E_{union}} \min(W_A(e), W_B(e))}{\sum_{e \in E_{union}} \max(W_A(e), W_B(e))}$$

Where:
- $E_{union}$ = union of all edges from both graphs
- $W_A(e)$ = weight of edge $e$ in graph A (0 if missing)

#### Properties

| Property             | Value                                            |
| -------------------- | ------------------------------------------------ |
| **Range**            | [0, 1]                                           |
| **Interpretation**   | 1.0 = identical weights, 0.0 = no overlap        |
| **Weight-sensitive** | ✅ Yes - weak edges contribute proportionally     |
| **Scale-sensitive**  | ✅ Yes, but inherently scale-invariant by formula |
| **Speed**            | ⚡ Fast                                           |

#### Why Scale-Invariant?

Ruzicka is **inherently scale-invariant** because of its min/max ratio formula:
```
R(A, kB) = Σmin(a, kb) / Σmax(a, kb) 
         ≈ R(A, B) for consistent patterns
```

If all weights in B are scaled by the same factor k, the relative min/max ratios remain similar.

#### Pros & Cons

| ✅ Pros                 | ❌ Cons                                     |
| ---------------------- | ------------------------------------------ |
| Weight-aware           | Sensitive to unique edges (penalizes them) |
| Strong edges dominate  | Less intuitive than Jaccard                |
| Natural scale handling |                                            |

#### When to Use

- When edge weights are biologically meaningful (synapse counts)
- When strong connections should dominate similarity
- As alternative to Jaccard + threshold

---

### 4.2 RV Coefficient

#### Formula

$$RV = \frac{\langle A, B \rangle_F^2}{||A||_F^2 \cdot ||B||_F^2}$$

Where:
- $\langle A, B \rangle_F = \sum_{ij} A_{ij} B_{ij}$ (Frobenius inner product)
- $||A||_F = \sqrt{\sum_{ij} A_{ij}^2}$ (Frobenius norm)

The RV coefficient (Robert & Escoufier, 1976) is a multivariate generalization of the squared Pearson correlation.

#### Properties

| Property             | Value                                         |
| -------------------- | --------------------------------------------- |
| **Range**            | [0, 1]                                        |
| **Interpretation**   | 1.0 = proportional matrices, 0.0 = orthogonal |
| **Weight-sensitive** | ✅ Yes - uses full weight information          |
| **Scale-sensitive**  | ❌ No (uses normalized weights)                |
| **Speed**            | ⚡ Fast                                        |

#### Normalization

The implementation normalizes weights to proportions before computing RV:
```python
norm_a = weights_a / weights_a.sum()
norm_b = weights_b / weights_b.sum()
```

This makes the metric invariant to different total synapse counts.

#### Pros & Cons

| ✅ Pros                              | ❌ Cons                       |
| ----------------------------------- | ---------------------------- |
| Multivariate (considers all edges)  | More complex interpretation  |
| Scale-invariant via normalization   | Requires aligned matrices    |
| Captures overall pattern similarity | Sensitive to sparse matrices |

#### When to Use

- Overall matrix-level pattern comparison
- When comparing full weight distributions
- As complement to edge-specific metrics

---

## Deprecated Metrics

### Frobenius Similarity (Not Used)

#### Why Deprecated

**Too sensitive to absolute scale differences.** Datasets often have different total synapse counts due to:
- Different annotation completeness
- Different reconstruction quality
- Biological variation

#### Formula

$$S_{Frobenius} = 1 - \frac{||A - B||_F}{||A||_F + ||B||_F}$$

#### Problem Demonstration

| Scenario                 | Frobenius | Problem                              |
| ------------------------ | --------- | ------------------------------------ |
| Identical                | 1.00      | ✅ Correct                            |
| Same pattern, 2x scale   | 0.67      | ⚠️ Penalizes scale difference         |
| Same pattern, 10x scale  | 0.18      | ❌ Very low despite identical pattern |
| Same pattern, 100x scale | 0.02      | ❌ Near-zero for same pattern!        |

**Conclusion**: Frobenius is inappropriate for comparing datasets with different scales.

---

### Pearson Correlation on Full Matrix (Not Used)

#### Why Deprecated

**Diluted by zeros in sparse adjacency matrices.** When using the union of all edges, many cells are (0, weight) comparisons:

```
Graph A edges: {A→B: 100, B→C: 50}
Graph B edges: {A→B: 80, C→D: 60, D→E: 40}

Union matrix positions: A→B, B→C, C→D, D→E
Values A: [100, 50, 0, 0]
Values B: [80, 0, 60, 40]

Correlation is dominated by the (50, 0), (0, 60), (0, 40) pairs!
```

#### Problem Demonstration

| Scenario                          | Pearson (union) | Problem            |
| --------------------------------- | --------------- | ------------------ |
| Identical                         | 1.00            | ✅ Correct          |
| 50% edge overlap, similar weights | 0.55            | ⚠️ Low due to zeros |
| 30% edge overlap, similar weights | 0.35            | ❌ Very low         |
| Different edges, both strong      | 0.10            | ❌ Near-zero        |

**Conclusion**: Pearson on union is dominated by zero-weight comparisons. The Spearman Rank approach on **shared edges only** avoids this problem.

---

## Comprehensive Comparison Table

### All Active Metrics

| Metric                | Category | Range   | Weight | Scale | Speed | Best For                  |
| --------------------- | -------- | ------- | ------ | ----- | ----- | ------------------------- |
| **Jaccard**           | Topology | [0, 1]  | ❌      | N/A   | ⚡     | Quick structure check     |
| **GED**               | Topology | [0, 1]  | ⚠️      | ⚠️     | 🐢     | Detailed small graph      |
| **WL Kernel**         | Topology | [0, 1]  | ⚠️      | ❌     | 🟡     | Neighborhood patterns     |
| **Edge Rank**         | Union    | [-1, 1] | ✅      | ❌     | ⚡     | Ranking with unique edges |
| **Cosine**            | Union    | [0, 1]  | ✅      | ❌     | ⚡     | Scale-invariant pattern   |
| **Spearman (shared)** | Shared   | [-1, 1] | ✅      | ❌     | ⚡     | Ranking of common edges   |
| **Ruzicka**           | Matrix   | [0, 1]  | ✅      | ❌*    | ⚡     | Weight-aware overlap      |
| **RV Coef**           | Matrix   | [0, 1]  | ✅      | ❌     | ⚡     | Overall pattern           |

\* Ruzicka is inherently scale-invariant by its min/max formula

### Deprecated Metrics

| Metric      | Category | Reason for Deprecation                      |
| ----------- | -------- | ------------------------------------------- |
| *Frobenius* | Matrix   | Too sensitive to absolute scale differences |
| *Pearson*   | Matrix   | Diluted by zeros in sparse matrices         |

### Key Differences: Edge Rank vs Spearman (Shared)

| Aspect             | Edge Rank (Union)                     | Spearman (Shared)                 |
| ------------------ | ------------------------------------- | --------------------------------- |
| **Edge set**       | Union (missing = 0)                   | Intersection only                 |
| **Unique edges**   | Contribute (low rank)                 | Ignored                           |
| **Sparse graphs**  | Can penalize uniqueness               | Focuses on overlap                |
| **Interpretation** | "How well do rankings agree overall?" | "How well do common edges agree?" |

### Test Results with Current Metrics

| Test Case               | Jaccard | Edge Rank | Cosine | Spearman | Ruzicka | RV   |
| ----------------------- | ------- | --------- | ------ | -------- | ------- | ---- |
| Identical graphs        | 1.00    | 1.00      | 1.00   | 1.00     | 1.00    | 1.00 |
| Same pattern, 2x scale  | 1.00    | 1.00      | 1.00   | 1.00     | 0.67    | 1.00 |
| Same pattern, 10x scale | 1.00    | 1.00      | 1.00   | 1.00     | 0.18    | 1.00 |
| No edge overlap         | 0.00    | -1.00*    | 0.00   | NaN      | 0.00    | 0.00 |
| 50% edge overlap        | 0.50    | 0.60      | 0.75   | 0.85     | 0.60    | 0.78 |
| Different patterns      | 0.50    | 0.30      | 0.55   | 0.45     | 0.45    | 0.55 |

\* Negative correlation indicates inverse ranking due to one dataset having edges the other lacks

### Recommended Metric Selection

| Scenario                                 | Primary Metric    | Secondary      |
| ---------------------------------------- | ----------------- | -------------- |
| Quick structural check                   | Jaccard           | -              |
| Compare edge presence                    | Jaccard           | GED            |
| Cross-dataset comparison (general)       | Edge Rank         | Cosine         |
| Compare weight importance (common edges) | Spearman (shared) | Ruzicka        |
| Overall pattern similarity               | Cosine            | RV Coefficient |
| Scale-invariant comparison               | Cosine            | Spearman       |
| Local neighborhood patterns              | WL Kernel         | GED            |
| Small detailed graphs                    | GED               | Jaccard        |
| Robust multi-metric                      | All active        | -              |

---

## Usage Example

```python
from src.comparison.metrics import ComparisonMetrics
import pandas as pd

metrics = ComparisonMetrics()

# Edge data as Series indexed by "source -> target" or tuples
edges_a = pd.Series({
    ("TypeA", "TypeB"): 100,
    ("TypeB", "TypeC"): 80,
    ("TypeC", "TypeD"): 60,
})

edges_b = pd.Series({
    ("TypeA", "TypeB"): 100,
    ("TypeB", "TypeC"): 80,
    ("TypeX", "TypeY"): 50,  # Different edge
})

# Calculate all active metrics
jaccard = metrics.calculate_jaccard_similarity(
    set(edges_a.index), set(edges_b.index)
)
edge_rank = metrics.calculate_edge_list_rank_correlation(edges_a, edges_b)  # Raw [-1, 1]
edge_rank_norm = metrics.calculate_edge_list_rank_correlation(edges_a, edges_b, normalize=True)  # [0, 1]
cosine = metrics.calculate_cosine_similarity(edges_a, edges_b)
spearman = metrics.calculate_spearman_rank_correlation(edges_a, edges_b)
ruzicka = metrics.calculate_ruzicka_similarity(edges_a, edges_b)
rv = metrics.calculate_rv_coefficient(edges_a, edges_b)

print(f"Topology Metrics:")
print(f"  Jaccard: {jaccard:.3f}")
print(f"Union-Based Metrics:")
print(f"  Edge Rank (raw): {edge_rank:.3f}")
print(f"  Edge Rank (norm): {edge_rank_norm:.3f}")
print(f"  Cosine: {cosine:.3f}")
print(f"Shared-Edge Metrics:")
print(f"  Spearman Rank: {spearman:.3f}")
print(f"Matrix Metrics:")
print(f"  Ruzicka: {ruzicka:.3f}")
print(f"  RV Coefficient: {rv:.3f}")
```

---

## Batch Comparison

For comparing multiple datasets, use `calculate_all_pairwise_similarities()`:

```python
import pandas as pd
from src.comparison.metrics import ComparisonMetrics

metrics = ComparisonMetrics()

# Aligned data: DataFrame with datasets as columns, edges as rows
aligned = pd.DataFrame({
    "hemibrain": {("A", "B"): 100, ("B", "C"): 80, ("C", "D"): 60, ("X", "Y"): 0},
    "male-cns": {("A", "B"): 90, ("B", "C"): 70, ("C", "D"): 0, ("X", "Y"): 50},
})

# Calculate all pairwise similarities (6 metrics)
similarities = metrics.calculate_all_pairwise_similarities(
    aligned,
    datasets=["hemibrain", "male-cns"],
    threshold=1,
    include_advanced_metrics=True
)

print(similarities)
# Output columns:
# Output columns:
# - dataset_1, dataset_2
# - jaccard_similarity, edge_rank_correlation, cosine_similarity (Union metrics)
# - spearman_rank_correlation (Shared-edge metric)
# - ruzicka_similarity, rv_coefficient (Matrix metrics)
# - ged_similarity, kernel_similarity (Topology metrics - if include_advanced_metrics=True)
```

---

## Related Documentation

- [CrossDatasetComparison_Guide.md](CrossDatasetComparison_Guide.md) - Overall comparison workflow
- [HomologFinding_Guide.md](HomologFinding_Guide.md) - Homolog finding (uses normalized [0, 1] values)
- [ConnectivityProfiler_Guide.md](ConnectivityProfiler_Guide.md) - Connectivity profiling (uses normalized [0, 1] values)

---

*Last updated: January 2025*
