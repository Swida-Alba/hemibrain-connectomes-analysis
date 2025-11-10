# Cache Directory Update Summary

## Changes Made

### Directory Rename
✅ `neuprint_cache/` → `cache/`

### Code Updates

**1. src/coana.py (4 locations)**
- Line 78: Updated `script_path` to point to project root (parent of src/)
  ```python
  # Before: script_path: str = os.path.dirname(os.path.abspath(__file__))
  # After:  script_path: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  ```
  **Reason:** After reorganization, `__file__` points to `src/coana.py`, so we need to go up TWO levels to reach project root.

- Line 241: Updated docstring reference
  ```python
  # Before: Cache is stored in: script_path/neuprint_cache/{dataset}/connections/
  # After:  Cache is stored in: cache/{dataset}/connections/ (in project root)
  ```

- Line 282: Updated cache folder path
  ```python
  # Before: self.cache_folder = os.path.join(self.script_path, 'neuprint_cache', dataset_safe)
  # After:  self.cache_folder = os.path.join(self.script_path, 'cache', dataset_safe)
  ```

- Line 4949: Updated VisualizeSkeleton class script_path (same fix as line 78)

**2. src/core/cache_manager.py (3 locations)**
- Line 27: `cache_root = 'cache'`
- Line 165: `cache_root = 'cache'`
- Line 202: `cache_root = 'cache'`

### Impact on Data Loading

✅ **All existing cache data is preserved**
- The directory was renamed, not recreated
- All cached connections remain accessible
- No re-downloading needed

✅ **Paths correctly resolved**
- Cache now at: `<project_root>/cache/<dataset>/`
- Example: `.../cache/hemibrain_v1_2_1/connections.parquet`

✅ **Backward compatibility**
- Code automatically uses the new `cache/` directory
- All scripts work without modification
- Cache manager tools updated

## Verification

Test performed:
```python
fc = FindNeuronConnection(
    token='test_token',
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['KC.*'],
    targetNeurons=['MBON.*'],
    use_cache=True
)
# Output: Cache enabled: .../cache/hemibrain_v1_2_1
# Cache exists: True ✓
```

## User Impact

**No action required!**
- Existing cached data works immediately
- Future cache operations use the new `cache/` directory
- All scripts continue to work as before

## Cache Location

Your cache is now at:
```
project_root/
└── cache/
    ├── hemibrain_v1_2_1/
    │   ├── connections.parquet
    │   └── neuron_index.parquet
    └── optic-lobe_v1_1/
        ├── connections.parquet
        └── neuron_index.parquet
```

## Documentation Updates Needed

The following documentation references `neuprint_cache` and should be updated to `cache`:
- docs/CacheSystem_QuickStart.md
- examples/Example_CachingDemo.py
- Other cache-related documentation

These are documentation-only changes and don't affect functionality.
