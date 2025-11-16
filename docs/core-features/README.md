# Core Features Documentation

Documentation for the fundamental features of the Hemibrain Connectomes Analysis toolkit.

## Overview

This directory contains documentation for the core analytical capabilities:
- **Path Finding**: Finding direct and multi-hop connections
- **Custom Groups**: Flexible neuron grouping for custom analysis
- **Cache System**: High-performance local data storage
- **Parallel Processing**: Multi-core acceleration
- **Filtering**: Connection and neuron filtering options

---

## Path Finding

### [FindAllPath Documentation](./FindAllPath_Documentation.md)
Complete guide to graph-based pathfinding algorithms.

**Key Topics**:
- Finding multi-hop paths between neuron populations
- Depth and weight cutoffs
- Forward-only mode for biological realism
- Path validation and filtering

**Quick Start**:
```python
from findpath import FindAllPath

fap = FindAllPath(
    token='your_token',
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['KC.*'],
    targetNeurons=['MBON03'],
    max_depth=3,
    min_weight=1,
    forward_only=True
)

paths_df = fap.find_all_paths()
```

### Related Documents
- [Custom Groups Feature](./CustomGroups_Feature.md)
- [Forward-Only Mode Guide](./ForwardOnly_Guide.md)
- [Multiple Connections Example](./FindAllPath_MultipleConnections_Example.md)
- [Caching Improvements](./FindAllPath_CachingImprovement.md)
- [Path Analysis Zero Weight Filter](./PathAnalysis_ZeroWeightFilter.md)

---

## Cache System

### [Cache System Guide](./CacheSystem_Guide.md)
Comprehensive guide to the local caching system that provides 10-100x speedup.

**Key Topics**:
- First-time setup and configuration
- Cache database architecture
- Storage optimization (40-60% smaller)
- Handling complete datasets

**Quick Start**:
```python
from findpath import FindNeuronConnection

fc = FindNeuronConnection(
    token='your_token',
    dataset='hemibrain:v1.2.1',
    data_folder='/path/to/cache',  # Enable caching
    use_cache=True
)
```

### Cache System Documents

#### Getting Started
- **[Quick Start](./CacheSystem_QuickStart.md)**: Get caching up and running in 5 minutes
- **[Complete Dataset](./CacheSystem_CompleteDataset.md)**: Handling type=None neurons efficiently

#### Technical Details
- **[Architecture v3](./CacheSystem_v3_DatabaseArchitecture.md)**: Database design and implementation
- **[Storage Optimization](./CacheSystem_StorageOptimization.md)**: Why cache is 40-60% smaller
- **[v4 Implementation](./CacheSystem_v4_Implementation.md)**: Latest improvements and pair-level caching
- **[v4 Complete](./CacheSystem_v4_Complete.md)**: Comprehensive v4 documentation
- **[v4 Pair-Level Proposal](./CacheSystem_v4_PairLevel_Proposal.md)**: Design rationale

### Performance Impact

| Operation | Without Cache | With Cache | Speedup |
|-----------|---------------|-----------|---------|
| First query (type-level) | ~2-5 seconds | ~2-5 seconds | 1x |
| Repeat query (same types) | ~2-5 seconds | ~0.05 seconds | **40-100x** |
| Repeat query (different types, cached) | ~2-5 seconds | ~0.1-0.2 seconds | **10-25x** |
| Large pathway analysis | Minutes | Seconds | **10-60x** |

---

## Parallel Processing

### [Parallel Processing Documentation](./ParallelProcessing_Documentation.md)
Multi-core acceleration for large-scale analyses.

**Key Topics**:
- Multi-core neuron fetching
- Progress tracking and ETA
- Memory management
- Worker pool optimization

**Quick Start**:
```python
from findpath import FindAllPath

fap = FindAllPath(
    # ... other parameters ...
    use_parallel=True,      # Enable parallel processing
    n_workers=4             # Number of CPU cores to use
)
```

### Related Documents
- **[Implementation Summary](./ParallelProcessing_Implementation_Summary.md)**: Technical details
- **[Quick Reference](./ParallelProcessing_QuickReference.md)**: Command reference
- **[Progress Tracking](./ParallelProcessing_ProgressTracking.md)**: Real-time progress displays
- **[Improved Progress v2](./ParallelProcessing_ImprovedProgress_v2.md)**: Enhanced ETA and statistics

### Performance Impact

| Dataset Size | Sequential | Parallel (4 cores) | Speedup |
|--------------|------------|-------------------|---------|
| 10 neurons | 1 sec | 0.8 sec | 1.25x |
| 100 neurons | 15 sec | 4 sec | **3.75x** |
| 500 neurons | 90 sec | 15 sec | **6x** |
| 2000 neurons | 8 min | 70 sec | **6.9x** |

**Best speedup**: 4-14x on large datasets with 4-8 cores

---

## Filtering

### Connection Filtering

#### [Connection Ratio Filter](./ConnectionRatio_Filter.md)
Filter connections by the proportion of source neuron's output.

**Use case**: Find connections that represent significant portion of source neuron's output
```python
min_connection_ratio = 0.05  # 5% or more of source output
```

#### [Traversal Probability Filter](./TraversalProbability_EdgeLevelFilter.md)
Filter by likelihood of signal traversal through connection.

**Use case**: Focus on functionally significant connections
```python
min_traversal_probability = 0.01  # 1% or higher probability
```

#### [Filter Modes Fix](./FILTER_MODES_FIX.md)
Documentation of filter mode corrections and proper usage.

### Neuron Filtering

#### [FilterBy Feature](./FilterBy_Feature.md)
Filter neurons by various criteria before analysis.

**Criteria**:
- Instance vs type
- Status (Traced, Roughly traced, etc.)
- Size (synapse count)
- ROI (brain region)

**Quick Reference**: [FilterBy_QuickRef.md](./FilterBy_QuickRef.md)

---

## Path Finding Optimization

### Algorithm Improvements

#### [Pathfinding Optimization](./PathFinding_Optimization.md)
Depth-first search optimization for faster path finding.

**Key improvements**:
- Early termination
- Memory-efficient recursion
- Cutoff optimization
- Dynamic ETA calculation

#### [Pathfinding Optimization Summary](./PATHFINDING_OPTIMIZATION_SUMMARY.md)
Complete summary of all pathfinding optimizations.

#### [Realistic Estimation](./PathFinding_RealisticEstimation.md)
Improved time estimation for long-running searches.

#### [Dynamic ETA](./PathFinding_DynamicETA.md)
Real-time estimation updates based on actual progress.

### DFS Implementation

#### [PathfindingOptimization_DFS.md](./PathfindingOptimization_DFS.md)
Depth-first search algorithm details and optimizations.

**Optimizations**:
- Branch pruning
- Weight-based early stopping
- Path deduplication
- Memory reuse

---

## Quick Reference

### Connection Filtering Options

The toolkit provides multiple filtering methods:

```python
# Synapse count threshold
min_synapse_num = 10

# Connection ratio (direct proportion)
min_ratio = 0.01  # weight/post

# Traversal probability (scaled)
min_traversal_probability = 0.001  # min(1, weight/(post*0.3))

# Filtering level
filter_by = 'bodyId'  # or 'type' for aggregation

# Exclude intra-type connections (NEW)
exclude_intra_type_connections = True  # Remove type_pre == type_post
```

**Intra-Type Exclusion Use Cases:**
- Cross-type connectivity analysis
- Cleaner network visualizations
- Inter-type communication studies
- See **[Example_ExcludeIntraType.py](../../examples/Example_ExcludeIntraType.py)**

### Path Finding Commands

```python
# Basic path finding
from findpath import FindAllPath

fap = FindAllPath(
    token='token',
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['source_type'],
    targetNeurons=['target_type'],
    max_depth=3,
    min_weight=1,
    forward_only=True,
    use_cache=True,
    use_parallel=True
)

paths = fap.find_all_paths()
```

### Cache Setup

```python
# First time setup
fc = FindNeuronConnection(
    token='token',
    dataset='hemibrain:v1.2.1',
    data_folder='/path/to/cache',  # Creates cache here
    use_cache=True
)

# Cache is automatically used for all subsequent queries
```

### Filtering

```python
# Connection strength filters
min_synapse_num = 3
min_connection_ratio = 0.05
min_traversal_probability = 0.01

# Neuron filters
filterBy = {
    'status': 'Traced',
    'min_pre': 10,
    'min_post': 10,
    'rois': ['MB', 'LH']
}
```

---

## Workflow Integration

### Typical Analysis Pipeline

```
1. Setup cache (first time only)
   ↓
2. Find paths with filters:
   - min_weight for weak connections
   - forward_only for biological realism
   - Use cache for speedup
   ↓
3. Enable parallel processing for large datasets
   ↓
4. Filter results:
   - By connection ratio
   - By traversal probability
   ↓
5. Visualize (see visualizations/)
```

### Performance Tips

1. **Always use cache**: 10-100x speedup on repeated queries
2. **Enable parallel processing**: 4-14x speedup for large datasets
3. **Use appropriate filters**: Reduce unnecessary computation
4. **Forward-only mode**: Faster and more biologically realistic
5. **Optimize depth**: Lower max_depth = faster results

---

## Troubleshooting

### Cache Issues
- **Cache not working**: Check `data_folder` path is writable
- **Cache too large**: Normal for complete datasets (~40-60GB for hemibrain)
- **Cache corrupted**: Delete cache directory and rebuild

### Path Finding Issues
- **Too many paths**: Increase `min_weight`, reduce `max_depth`
- **No paths found**: Check source/target names, reduce `min_weight`
- **Slow performance**: Enable cache and parallel processing

### Parallel Processing Issues
- **No speedup**: Dataset too small, parallel overhead dominates
- **Memory error**: Reduce `n_workers` or dataset size
- **Progress bar frozen**: Large batches being processed, wait for next update

---

## Best Practices

### Cache Management
1. Use one cache directory per dataset
2. Let cache build naturally through queries
3. Don't manually edit cache files
4. Backup important caches

### Path Finding
1. Start with restrictive filters (high `min_weight`)
2. Gradually relax if needed
3. Use `forward_only=True` for biological accuracy
4. Monitor progress for long searches

### Filtering
1. Apply neuron filters before path finding
2. Use connection filters during path finding
3. Post-process results for final filtering
4. Document filter choices for reproducibility

---

## Related Documentation

- [Visualization Documentation](../visualizations/)
- [Technical Documentation](../technical/)
- [Main README](../../README.md)
- [Quick Start Guide](../QUICK_START_AFTER_REORGANIZATION.md)

---

## Examples

See `examples/` directory for complete workflows:
- `example_pathfinding.py`: Basic path finding
- `example_cache_setup.py`: Cache configuration
- `example_parallel.py`: Parallel processing
- `example_filtering.py`: Various filter combinations
