# ConnectivityProfiler Guide

## Overview

The `ConnectivityProfiler` is the **foundational module** for building connectivity profiles in the hemibrain-connectomes-analysis project. It implements the **1-hop/2-hop hybrid approach** with **top-k/top-m dynamic expansion**, which is the authoritative method for all profile construction.

**Key Principle**: All connectivity profiles are built at the bodyId level. Type-level profiles are aggregations of bodyId-level profiles.

## Architecture

```
ConnectivityProfiler
├── ProfilerConfig         # Configuration for profile construction
├── FuzzyMatchConfig       # Type name normalization rules
├── ConnectivityProfile    # Profile data structure (bodyId-level)
└── Cache System           # 3-tier caching (memory → disk-index → disk-df)
```

## The 1-hop/2-hop Hybrid Approach

### Concept

For each neuron, we build a connectivity profile containing:
1. **1-hop typed partners**: Direct synaptic partners with known types
2. **2-hop typed partners**: For untyped 1-hop partners, we fetch their typed partners

This hybrid approach ensures that:
- Neurons with many untyped 1-hop partners still have meaningful profiles
- The profile captures indirect connectivity through untyped neurons
- Cross-dataset comparisons are more robust (different datasets have different typing coverage)

### Algorithm

```
For each neuron (bodyId):
    1. Query upstream connections (neurons synapsing onto this neuron)
    2. Query downstream connections (neurons this neuron synapses onto)
    3. For each direction:
        a. Identify typed partners (have known type)
        b. Identify untyped partners (no type annotation)
        c. For untyped partners, fetch their typed partners (2-hop)
    4. Apply top-k/top-m expansion to select partners
    5. Store weights and ranks for profile
```

### Data Flow

```
Raw Connections (bodyId_pre, bodyId_post, weight)
          │
          ▼
    ┌─────────────┐
    │ 1-hop Query │
    └─────────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
  Typed      Untyped
  Partners   Partners
    │           │
    │     ┌─────┴─────┐
    │     │ 2-hop     │
    │     │ Query     │
    │     └─────┬─────┘
    │           ▼
    │     2-hop Typed
    │     Partners
    │           │
    └─────┬─────┘
          ▼
    ┌─────────────┐
    │ top-k/top-m │
    │ Expansion   │
    └─────────────┘
          │
          ▼
    ConnectivityProfile
```

## Top-k/Top-m Dynamic Expansion

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k_bodyid` | 15 | Initial number of top partners by synapse weight |
| `top_m_type` | 5 | Minimum unique partner types to ensure |
| `max_expansion_factor` | 5 | Maximum K = top_k × factor |
| `dynamic_expansion` | True | Enable automatic K expansion |

### Algorithm

```python
def apply_top_k_m(partners, top_k, top_m):
    # Sort partners by synapse weight (descending)
    sorted_partners = sort(partners, by=weight, descending=True)
    
    # Start with top-k
    k_used = top_k
    selected = sorted_partners[:k_used]
    
    # Expand until top-m unique types or max_k reached
    while unique_types(selected) < top_m and k_used < max_k:
        k_used += 5
        selected = sorted_partners[:k_used]
    
    return selected, k_used
```

### Rationale

- **top-k**: Limits profile size for performance and focuses on strongest connections
- **top-m**: Ensures profiles have enough type diversity for meaningful comparison
- **Dynamic expansion**: Automatically increases k when needed to reach m types
- **max_expansion_factor**: Prevents runaway expansion for pathological cases

## Configuration

### ProfilerConfig

```python
from src.comparison.connectivity_profiler import ProfilerConfig, ConnectivityProfiler

config = ProfilerConfig(
    # Core expansion parameters
    top_k_bodyid=15,           # Top K partners by weight
    top_m_type=5,              # Minimum unique types
    dynamic_expansion=True,    # Enable K expansion
    max_expansion_factor=5,    # Max K = 15 × 5 = 75
    
    # Filtering
    min_synapse_threshold=3,   # Minimum synapses for connection
    include_untyped=True,      # Include untyped in 1-hop query
    
    # 2-hop expansion for untyped
    expand_untyped_2hop=True,  # Enable 2-hop for untyped partners
    top_k_2hop=5,              # Top 5 2-hop partners per untyped
    
    # Type normalization
    fuzzy_match=FuzzyMatchConfig(enabled=True),
    
    # Caching
    use_cache=True,
    cache_profiles=True
)

profiler = ConnectivityProfiler(config)
```

### FuzzyMatchConfig

```python
fuzzy_match = FuzzyMatchConfig(
    enabled=True,
    strip_hemisphere=True,    # Remove _L, _R, (L), (R) suffixes
    strip_numbers=False,      # Keep numeric suffixes
    normalize_case=True       # Convert to lowercase
)
```

## ConnectivityProfile Structure

```python
@dataclass
class ConnectivityProfile:
    # Identity
    neuron_id: Union[str, int]      # BodyId or type name
    dataset: str                     # Dataset identifier
    
    # Partner data (1-hop typed + aggregated 2-hop)
    upstream_partners: Dict[str, float]     # type → total weight
    downstream_partners: Dict[str, float]   # type → total weight
    upstream_ranks: Dict[str, int]          # type → rank (1=highest)
    downstream_ranks: Dict[str, int]        # type → rank
    
    # Metadata
    num_neurons_aggregated: int = 1
    total_upstream_weight: float = 0.0
    total_downstream_weight: float = 0.0
    
    # Untyped partner info
    untyped_upstream_count: int = 0
    untyped_downstream_count: int = 0
    untyped_upstream_weight_fraction: float = 0.0
    untyped_downstream_weight_fraction: float = 0.0
    
    # 2-hop data for untyped partners
    untyped_upstream_2hop: Optional[Dict[int, Dict[str, float]]] = None
    untyped_downstream_2hop: Optional[Dict[int, Dict[str, float]]] = None
    
    # Expansion info
    top_k_bodyid_used: int = 15
    top_m_type_target: int = 5
    unique_types_upstream: int = 0
    unique_types_downstream: int = 0
    
    # Flags
    is_weak_connectivity: bool = False
    is_sparse: bool = False
```

## Usage Examples

### Basic Profile Extraction

```python
from src.comparison import ConnectivityProfiler, ProfilerConfig

# Initialize profiler
profiler = ConnectivityProfiler(ProfilerConfig(
    top_k_bodyid=15,
    top_m_type=5,
    expand_untyped_2hop=True
))

# Get profile for a specific bodyId
profile = profiler.get_profile(720575940631089589, 'hemibrain:v1.2.1')

print(f"Upstream partners: {len(profile.upstream_partners)}")
print(f"Downstream partners: {len(profile.downstream_partners)}")
print(f"Unique upstream types: {profile.unique_types_upstream}")
print(f"K used: {profile.top_k_bodyid_used}")
```

### Type-Level Profile (Aggregated)

```python
# When query is a type name, profiler aggregates all bodyIds in that type
type_profile = profiler.get_profile('Mi1', 'hemibrain:v1.2.1')

print(f"Neurons aggregated: {type_profile.num_neurons_aggregated}")
print(f"Top upstream: {list(type_profile.upstream_partners.keys())[:5]}")
```

### Batch Extraction

```python
# Get profiles for multiple neurons efficiently
bodyids = [720575940631089589, 720575940631089590, 720575940631089591]
profiles = profiler.get_profiles_batch(bodyids, 'hemibrain:v1.2.1')

for bid, profile in profiles.items():
    print(f"{bid}: {len(profile.upstream_partners)} upstream types")
```

### Profile Comparison

```python
from src.comparison import ProfileComparator

# Compare two profiles
scores = ProfileComparator.combined_score(profile1, profile2)
print(f"Jaccard: {scores['jaccard']:.3f}")
print(f"Cosine: {scores['cosine']:.3f}")
print(f"Rank correlation: {scores['rank']:.3f}")
```

## Cache System

### 3-Tier Cache

```
Memory Cache (instant)
    ↓ miss
Disk Index (O(1) lookup via SQLite/parquet index)
    ↓ miss
Disk DataFrame (full parquet scan)
    ↓ miss
API Query / Local Data
```

### Cache Locations

```
project_root/
└── cache/
    └── {dataset_safe_name}/
        ├── connections.parquet      # Connection data
        ├── neuron_index.parquet     # BodyId → type mapping
        └── profiles/
            └── {neuron_id}.json     # Cached profiles
```

### Cache Control

```python
# Force refresh (bypass cache)
profile = profiler.get_profile(bodyid, dataset, force_refresh=True)

# Clear cache
profiler.clear_cache()

# Disable caching
config = ProfilerConfig(use_cache=False, cache_profiles=False)
```

## Integration with HomologFinder

The `HomologFinder` module **exclusively** uses `ConnectivityProfiler` for all profile building:

```python
# HomologFinder internally calls:
profile = self.profiler.get_profile(neuron, dataset)

# This ensures:
# 1. 1-hop/2-hop hybrid approach is always used
# 2. top-k/top-m expansion is applied consistently
# 3. Profiles are cached for reuse
```

**Important**: The deprecated methods `_build_profile_from_aggregates()` and `_build_profile_from_bodyid_aggregates()` should NOT be used. They bypass the 2-hop expansion and are only kept for backwards compatibility.

## Best Practices

1. **Always use ConnectivityProfiler.get_profile()** for profile building
2. **Configure top-k/top-m appropriately** for your use case:
   - Cross-dataset comparison: Higher top_m (10+) for more overlap
   - Same-dataset comparison: Lower top_m (5) is usually sufficient
3. **Enable 2-hop expansion** (`expand_untyped_2hop=True`) for datasets with many untyped neurons
4. **Use caching** for repeated comparisons
5. **Check profile flags** (`is_weak_connectivity`, `is_sparse`) before trusting comparison results

## API Reference

### ConnectivityProfiler

```python
class ConnectivityProfiler:
    def __init__(self, config: ProfilerConfig = None)
    
    def get_profile(
        self,
        neuron: Union[str, int, List],
        dataset: str,
        force_refresh: bool = False
    ) -> ConnectivityProfile
    
    def get_profiles_batch(
        self,
        neurons: List[Union[str, int]],
        dataset: str,
        force_refresh: bool = False
    ) -> Dict[Union[str, int], ConnectivityProfile]
    
    def get_available_types(self, dataset: str) -> List[str]
    
    def get_bodyids_for_type(self, type_name: str, dataset: str) -> List[int]
```

## See Also

- [HomologFinding_Guide.md](./HomologFinding_Guide.md) - Using profiles for homolog discovery
- [CrossDatasetComparison_Guide.md](./CrossDatasetComparison_Guide.md) - Comparing across datasets
- [CacheSystem_Guide_v4.md](./CacheSystem_Guide_v4.md) - Cache architecture details
