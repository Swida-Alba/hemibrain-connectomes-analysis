# Cache Management Enhancements

## Overview
Five features added to improve cache visibility, documentation, customization, and performance for brain transforms and neuron data.

**v4.1.7 Update (Dec 2025):** Added 3-tier in-memory caching for 100,000x+ speedup on repeated lookups.

## Features

### 1. In-Memory Cache with O(1) Lookups (v4.1.7) 🚀

**Problem:** Every cache lookup was reading from disk (parquet files), causing ~100-150ms delays.

**Solution:** Implemented a 3-tier cache system with in-memory DataFrame caching and O(1) dict lookups.

#### Cache Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      3-Tier Cache System                        │
├─────────────────────────────────────────────────────────────────┤
│  Tier 1: Memory Cache (O(1))                                    │
│  ├── _memory_cache: Dict[(neuron_id, dataset) → Profile]        │
│  └── Lookup: ~0.001 ms                                          │
├─────────────────────────────────────────────────────────────────┤
│  Tier 2: Disk Cache Index (O(1))                                │
│  ├── _disk_cache_index: Dict[dataset → Dict[id → row_idx]]      │
│  └── Lookup: ~0.05 ms                                           │
├─────────────────────────────────────────────────────────────────┤
│  Tier 3: Disk Cache DataFrame (in-memory)                       │
│  ├── _disk_cache_df: Dict[dataset → DataFrame]                  │
│  ├── First load from disk: ~2 ms                                │
│  └── Subsequent: cached in memory                               │
├─────────────────────────────────────────────────────────────────┤
│  Tier 4: API Fetch (fallback)                                   │
│  └── ~900-2000 ms per neuron                                    │
└─────────────────────────────────────────────────────────────────┘
```

#### Performance Results (Tested Dec 2025)

**ConnectivityProfiler:**
| Operation | Time | Speedup |
|-----------|------|---------|
| First fetch (API) | 1144 ms | baseline |
| Memory cache | 0.002 ms | **570,000x** |
| Disk cache (in-memory DF) | 0.07 ms | **16,000x** |
| Cold disk load (first) | 1.9 ms | 600x |

**FindNeuronConnection (coana.py):**
| Operation | Time | Speedup |
|-----------|------|---------|
| First load (disk) | 126 ms | baseline |
| Second load (memory) | 0.0007 ms | **178,000x** |
| Dict lookup | 0.00002 ms | per lookup |

#### Implementation Details

**ConnectivityProfiler (`src/comparison/connectivity_profiler.py`):**
```python
class ConnectivityProfiler:
    def __init__(self, ...):
        # Tier 1: Memory cache for profiles
        self._memory_cache: Dict[Tuple[str, str], ConnectivityProfile] = {}
        
        # Tier 2 & 3: Disk cache with in-memory DataFrame
        self._disk_cache_df: Dict[str, pd.DataFrame] = {}
        self._disk_cache_index: Dict[str, Dict[str, int]] = {}
    
    def _load_cache_dataframe(self, dataset, force_reload=False):
        # Returns cached DataFrame or loads from disk once
        if dataset in self._disk_cache_df:
            return self._disk_cache_df[dataset]  # Instant
        # Load from disk, cache in memory
        df = pd.read_parquet(cache_path)
        self._disk_cache_df[dataset] = df
        self._build_disk_cache_index(dataset)  # O(1) lookup index
        return df
    
    def _load_from_cache(self, neuron_id, dataset, ...):
        # Tier 1: Check memory cache
        if (neuron_id, dataset) in self._memory_cache:
            return self._memory_cache[(neuron_id, dataset)]  # ~0.001 ms
        
        # Tier 2: Use index for O(1) row lookup
        row_idx = self._disk_cache_index[dataset].get(neuron_id)
        if row_idx is not None:
            profile = self._row_to_profile(cache_df.iloc[row_idx])
            self._memory_cache[(neuron_id, dataset)] = profile
            return profile  # ~0.05 ms
        
        return None  # → Fallback to API fetch
```

**FindNeuronConnection (`src/coana.py`):**
```python
@dataclass
class FindNeuronConnection:
    def __post_init__(self):
        # Connection cache (O(1) by bodyId_pre)
        self._conn_df_cache: Optional[pd.DataFrame] = None
        self._conn_index: Dict[str, List[int]] = {}  # bodyId → row indices
        
        # Neuron index cache (O(1) by bodyId)
        self._neuron_index_cache: Optional[pd.DataFrame] = None
        self._neuron_index_dict: Dict[str, Dict] = {}  # bodyId → metadata
    
    def _load_connection_db(self, force_reload=False):
        if self._conn_df_cache is not None and not force_reload:
            return self._conn_df_cache  # Instant (~0.0007 ms)
        # Load from disk, build index
        df = pd.read_parquet(db_path)
        self._conn_df_cache = df
        self._build_conn_index()  # Dict[bodyId_pre → row_indices]
        return df
    
    def _query_connection_db(self, upstream_bodyIds, ...):
        # O(1) dict lookup instead of DataFrame filter
        for bodyId in upstream_bodyIds:
            neuron_data = self._neuron_index_dict.get(bodyId)  # O(1)
            if neuron_data and neuron_data['downstream_complete']:
                cached_upstream.append(bodyId)
        
        # Retrieve using index
        row_indices = []
        for bodyId in cached_upstream:
            if bodyId in self._conn_index:
                row_indices.extend(self._conn_index[bodyId])  # O(1)
        cached_conn = conn_db.iloc[row_indices]  # Direct row access
```

---

### 2. Yellow Path Display 🎨
**Problem:** Transform cache path was displayed in plain text, hard to spot in terminal output.

**Solution:** Added ANSI color codes to highlight cache path in yellow.

**Code Changes:**
- `src/coana.py` lines ~6006, ~6090: Added `YELLOW = '\033[93m'` and `RESET = '\033[0m'`
- Transform path now displays as: `Location: /Users/username/flybrain-data/` (in yellow)

**Example Output:**
```
The transforms will be cached in:
  /Users/username/flybrain-data/  ← (shown in yellow)
```

---

### 2. Transform Path Documentation File 📄
**Problem:** Users needed a permanent record of where transform files are stored.

**Solution:** Automatically generates `brain_transforms_info.txt` in data folder.

**File Location:** `{data_folder}/brain_transforms_info.txt`

**Contents:**
- Dataset information
- Transform path details
- Storage location (highlighted)
- All 8 transform files with sizes
- Instructions for changing location
- Links to documentation

**Example:**
```
Brain Transforms Information
======================================================================

Dataset: hemibrain:v1.2.1
Transform path: JRCFIB2018Fraw → JRCFIB2018F → JRCFIB2018Fum → JRC2018F

Storage Location:
  /Users/username/flybrain-data/

Transform Files (8 files, ~10 GB total):
  • JRC2018F_JRCFIB2018F.h5   (~1.29 GB)
  • JRC2018F_FAFB.h5          (~580 MB)
  ...

Download Status: SUCCESS
Downloaded at: 2025-01-13 10:23:45
```

---

### 3. Custom Transform Cache Path 🗂️
**Problem:** Users couldn't customize where transform files are stored.

**Solution:** Added `transforms_dir` attribute to `VisualizeSkeleton` class.

**New Attribute:**
```python
transforms_dir: str = '~/flybrain-data'
```

**Default:** `~/flybrain-data` (flybrains package default)

**Usage:**
```python
vis = VisualizeSkeleton(
    neuron_layers=[criteria],
    transforms_dir='~/custom/path/transforms',  # Custom location
    brain_mesh='whole'
)
```

**Implementation:**
- Expands user path (`~` → full path)
- Sets `FLYBRAINS_DATA` environment variable for custom paths
- Updates all transform operations to use custom directory
- Prints confirmation when using custom path

**Notes:**
⚠️ Changing this requires:
1. Setting attribute before visualization
2. OR setting `FLYBRAINS_DATA` environment variable before importing flybrains
3. OR manually moving existing transform files to new location

---

### 4. Neuron & Synapse Data Caching 💾
**Problem:** Repeatedly fetching skeleton and synapse data from NeuPrint was slow.

**Solution:** Added intelligent disk caching with automatic management.

#### New Attributes:
```python
cache_neurons: bool = False     # Cache skeleton data
cache_synapses: bool = False    # Cache synapse connections
```

#### Cache Structure:
```
cache/
├── hemibrain_v1_2_1/
│   ├── available_rois.json
│   ├── connections.parquet
│   ├── neuron_index.parquet
│   ├── meshes/
│   │   ├── AL_L.json
│   │   ├── AL_R.json
│   │   └── ... (63 ROI meshes)
│   ├── neurons/
│   │   └── {bodyId_key}_n{count}.pkl
│   └── synapses/
│       └── layer{i}_{hash}.parquet
└── optic-lobe_v1_1/
    ├── available_rois.json
    ├── connections.parquet
    ├── neuron_index.parquet
    ├── meshes/
    ├── neurons/
    └── synapses/
```

#### Usage:
```python
vis = VisualizeSkeleton(
    neuron_layers=[criteria1, criteria2],
    cache_neurons=True,    # Cache skeleton fetches
    cache_synapses=True,   # Cache synapse fetches
    dataset='hemibrain:v1.2.1'
)
```

#### Features:
- **Automatic Cache Keys:** 
  - Neurons: First 5 bodyIds + total count
  - Synapses: MD5 hash of layer criteria
- **Smart Loading:** Checks cache before fetching from NeuPrint
- **Dataset Isolation:** Separate cache per dataset
- **Format:**
  - Neurons: Pickle (preserves navis objects)
  - Synapses: Parquet (efficient for DataFrames)
- **Progress Feedback:**
  - `✓ Loaded N neurons from cache`
  - `💾 Saved N neurons to cache`
  - `⚠ Cache load failed: {error}, fetching from NeuPrint...`

#### Cache Methods:
```python
# Neuron caching
_get_cache_path(cache_type)          # Get cache directory
_load_cached_neurons(neuron_df)      # Load from cache
_save_cached_neurons(neuron_df, vols) # Save to cache

# Synapse caching
_load_cached_synapses(src, tgt, idx)  # Load from cache
_save_cached_synapses(df, src, tgt, idx) # Save to cache
```

---

## Benefits

### Performance 🚀
- **First Run:** Normal NeuPrint fetch times
- **Subsequent Runs:** Near-instant loading from disk
- **Typical Speedup:** 10-50x faster (network-dependent)

### Visibility 👁️
- Yellow highlighting makes cache paths easy to spot
- No more scrolling to find where files are stored

### Documentation 📋
- Permanent record of transform locations
- Includes download status and timestamps
- Self-documenting for future reference

### Flexibility 🔧
- Custom cache locations for different projects
- Separate caches per dataset
- Easy cache management and cleanup

---

## Code Locations

### Transform Features (1-3):
- **Attributes:** `src/coana.py` lines ~5480-5520
- **Path Display:** `src/coana.py` lines ~6006-6090
- **File Generation:** `src/coana.py` lines ~6040-6070

### Caching Features (4):
- **Attributes:** `src/coana.py` lines ~5505-5525
- **Cache Methods:** `src/coana.py` lines ~5677-5730
- **Integration:** 
  - Neurons: `plot_skeleton()` line ~5730
  - Synapses: `plot_synapses()` line ~5830

---

## Migration Guide

### Existing Code (No Changes Needed)
```python
# This still works exactly as before
vis = VisualizeSkeleton(
    neuron_layers=[criteria],
    brain_mesh='whole'
)
```

### With New Features
```python
# Use all new features
vis = VisualizeSkeleton(
    neuron_layers=[criteria1, criteria2],
    dataset='hemibrain:v1.2.1',
    
    # Custom transform location
    transforms_dir='~/project/transforms',
    
    # Enable caching
    cache_neurons=True,
    cache_synapses=True,
    
    brain_mesh='whole'
)
vis.plot_skeleton()  # Uses cache if available
vis.plot_synapses()  # Uses cache if available
```

---

## Cache Management

### View Cache Contents
```bash
# Neuron cache
ls -lh connection_data/neuron_cache/hemibrain_v1_2_1/

# Synapse cache
ls -lh connection_data/synapse_cache/hemibrain_v1_2_1/

# Transform info
cat connection_data/brain_transforms_info.txt
```

### Clear Cache
```bash
# Clear neuron cache
rm -rf connection_data/neuron_cache/

# Clear synapse cache
rm -rf connection_data/synapse_cache/

# Clear specific dataset
rm -rf connection_data/neuron_cache/hemibrain_v1_2_1/
```

### Cache Sizes
- **Neuron cache:** ~100KB - 10MB per file (depends on skeleton complexity)
- **Synapse cache:** ~10KB - 5MB per file (depends on connection count)
- **Transform files:** ~10GB total (in `~/flybrain-data/`)

---

## Implementation Details

### Yellow Terminal Output
Uses ANSI escape codes:
- `\033[93m` - Yellow foreground
- `\033[0m` - Reset to default

Works in: macOS Terminal, iTerm2, Linux terminals, Windows Terminal

### Cache Key Generation

**Neurons:**
```python
# Example: bodyIds = [123, 456, 789, 1011, 1213, 1415]
cache_key = "123_456_789_1011_1213_n6"
#           ↑ first 5 IDs      ↑ total count
```

**Synapses:**
```python
import hashlib
criteria_str = f'{source_criteria}_{target_criteria}_{layer_idx}'
cache_key = hashlib.md5(criteria_str.encode()).hexdigest()[:16]
# Example: "a1b2c3d4e5f6g7h8"
```

### Error Handling
All cache operations include try-except blocks:
- Cache load failure → Falls back to NeuPrint fetch
- Cache save failure → Prints warning, continues execution
- Never crashes due to cache issues

---

## Testing

### Test Yellow Output
```python
vis = VisualizeSkeleton(
    neuron_layers=[{"type": "TmY9"}],
    dataset='hemibrain:v1.2.1',
    brain_mesh='whole'
)
# Check terminal output for yellow cache path
```

### Test Documentation File
```python
vis = VisualizeSkeleton(
    neuron_layers=[{"type": "TmY9"}],
    brain_mesh='whole'
)
# Check: connection_data/brain_transforms_info.txt exists
```

### Test Custom Path
```python
vis = VisualizeSkeleton(
    neuron_layers=[{"type": "TmY9"}],
    transforms_dir='~/test/transforms',
    brain_mesh='whole'
)
# Check terminal for "Using custom transform directory: /Users/x/test/transforms"
```

### Test Caching
```python
import time

# First run (no cache)
start = time.time()
vis1 = VisualizeSkeleton(
    neuron_layers=[{"type": "TmY9"}],
    cache_neurons=True,
    cache_synapses=True
)
vis1.plot_skeleton()
print(f"First run: {time.time()-start:.2f}s")

# Second run (with cache)
start = time.time()
vis2 = VisualizeSkeleton(
    neuron_layers=[{"type": "TmY9"}],
    cache_neurons=True,
    cache_synapses=True
)
vis2.plot_skeleton()
print(f"Second run: {time.time()-start:.2f}s")  # Should be much faster
```

---

## Future Enhancements

Possible future additions:
1. **Cache Expiration:** Auto-delete old cache files
2. **Cache Statistics:** Report cache hit/miss rates
3. **Compression:** Compress cache files to save disk space
4. **Smart Invalidation:** Detect when NeuPrint data has changed
5. **Cache Browser:** GUI to inspect and manage cache contents

---

## Testing the Cache System

Run the cache performance test:
```bash
cd /path/to/hemibrain-connectomes-analysis
python dev/test_cache_performance.py
```

Expected output:
```
======================================================================
Testing coana.py FindNeuronConnection Cache Performance
======================================================================
  First load (disk):    126.4 ms
  Second load (memory): 0.0007 ms
  Speedup:              178257x

======================================================================
Testing ConnectivityProfiler Cache Performance
======================================================================
  First fetch (no cache):     1144.3 ms average
  Memory cache:               0.002 ms average
  Disk cache (in-memory DF):  0.072 ms average
  Speedup (memory cache):     773106x
  Speedup (disk cache):       19104x
```

---

## Related Documentation

- [Multi-Dataset Support](QUICK_REFERENCE.md)
- [Installation Guide](INSTALLATION.md)
- [Visualization Guide](visualizations/README.md)
- [flybrains package](https://github.com/navis-org/navis-flybrains)

---

**Version:** 1.1  
**Date:** 2025-12-01  
**Author:** Cache Enhancement Update (v4.1.7: In-memory caching)
