# Cache Reorganization & Fixes (November 2025)

## Overview
Three major improvements to cache management and dataset support.

---

## 1. Unified Cache Structure in `cache/` Folder

### Problem
Cache files were scattered across multiple locations:
- Neuron/synapse cache: `connection_data/{cache_type}_cache/{dataset}/`
- ROI meshes: `navis_roi_meshes_json/{dataset}/`
- ROI lists: `navis_roi_meshes_json/{dataset}_available_rois.json`
- Dataset files: `cache/{dataset}/` (connections, neuron_index)

This was confusing and made cache management difficult.

### Solution
Unified all dataset-specific files in `cache/{dataset}/`:

```
cache/
├── hemibrain_v1_2_1/
│   ├── available_rois.json        ← ROI list (230 ROIs)
│   ├── connections.parquet         ← Pre-processed connections
│   ├── neuron_index.parquet        ← Neuron metadata index
│   ├── meshes/                     ← ROI 3D meshes (63 files)
│   │   ├── AL_L.json
│   │   ├── AL_R.json
│   │   └── ... (other ROIs)
│   ├── neurons/                    ← Cached skeleton data
│   │   └── {bodyIds}_n{count}.pkl
│   └── synapses/                   ← Cached synapse connections
│       └── layer{i}_{hash}.parquet
│
└── optic-lobe_v1_1/
    ├── available_rois.json        ← ROI list (2690 ROIs!)
    ├── connections.parquet
    ├── neuron_index.parquet
    ├── meshes/                     ← Empty for now (can add as needed)
    ├── neurons/
    └── synapses/
```

### Benefits
- ✅ All dataset files in one place
- ✅ Easy to see what's cached for each dataset
- ✅ Simple cache cleanup: `rm -rf cache/hemibrain_v1_2_1/neurons/`
- ✅ Easy to backup/transfer: zip entire `cache/{dataset}/` folder
- ✅ Clear separation between datasets

### Code Changes
**`_get_cache_path()` method:**
```python
# OLD: connection_data/neuron_cache/hemibrain_v1_2_1/
# NEW: cache/hemibrain_v1_2_1/neurons/
cache_dir = os.path.join(self.script_path, 'cache', dataset_normalized, cache_type)
```

**`_get_dataset_mesh_dir()` method:**
```python
# Try new structure first
cache_mesh_dir = os.path.join(self.script_path, 'cache', dataset_normalized, 'meshes')

# Fallback to old structure for backward compatibility
old_mesh_dir = os.path.join(self.script_path, 'navis_roi_meshes_json', dataset_normalized)
```

**ROI list location:**
```python
# OLD: navis_roi_meshes_json/hemibrain_v1_2_1_available_rois.json
# NEW: cache/hemibrain_v1_2_1/available_rois.json
cache_file = os.path.join(self.script_path, 'cache', dataset_normalized, 'available_rois.json')
```

---

## 2. Fixed Optic-Lobe Transform Source

### Problem
Optic-lobe dataset was using `JRCFIB2018Fraw` as source, which is incorrect.

**Incorrect transform path:**
```
JRCFIB2018Fraw → JRCFIB2018F → JRC2018F  ❌ WRONG!
```

This caused transformation errors because optic-lobe neurons are stored in **OL (Optic Lobe)** coordinate space in NeuPrint, not JRCFIB2018Fraw.

### Solution
Use correct source template for optic-lobe: `OL`

**Correct transform path:**
```
OL → JRCFIB2018F → JRC2018F  ✓ CORRECT!
```

### Code Changes
**`_get_template_info()` method:**
```python
# OLD: Both hemibrain and optic-lobe used same source
if 'hemibrain' in dataset_lower or 'optic' in dataset_lower:
    return {'source': 'JRCFIB2018Fraw', ...}  # ❌ WRONG for optic-lobe

# NEW: Separate handling for each dataset
if 'hemibrain' in dataset_lower:
    return {'source': 'JRCFIB2018Fraw', ...}  # ✓ Correct for hemibrain

elif 'optic' in dataset_lower:
    return {
        'source': 'OL',  # ✓ Correct for optic-lobe
        'target': 'JRC2018F' if self.brain_mesh == 'whole' else 'JRCFIB2018F',
        'template_obj': flybrains.JRCFIB2018F,
        'mesh_name': 'JRCFIB2018F (optic lobe)'
    }
```

### Template Coordinate Systems

| Dataset | NeuPrint Space | Transform Path | Notes |
|---------|---------------|----------------|-------|
| hemibrain:v1.2.1 | JRCFIB2018Fraw | JRCFIB2018Fraw → JRCFIB2018F → JRC2018F | Full brain EM |
| optic-lobe:v1.1 | OL | OL → JRCFIB2018F → JRC2018F | Optic lobe EM |
| manc:v1.2.3 | MANCraw | MANCraw → MANC | VNC EM |
| male-cns:v0.9 | JRCFIB2022Mraw | JRCFIB2022Mraw → JRCFIB2022M | Male CNS |

### References
- flybrains package: https://github.com/navis-org/navis-flybrains
- OL template: https://neuprint.janelia.org/optic-lobe
- Transform registrations: See flybrains.report() output

---

## 3. Reorganized ROI Mesh Directories

### Problem
Old structure mixed dataset-specific meshes with generic meshes:
```
navis_roi_meshes_json/
├── hemibrain_v1_2_1/          ← Dataset-specific
│   └── AL_L.json
├── hemibrain_v1_2_1_available_rois.json  ← Awkward naming
└── primary_rois/              ← Generic/fallback
    └── some_roi.json
```

### Solution
Dataset-specific organization in cache/:
```
cache/
├── hemibrain_v1_2_1/
│   ├── available_rois.json    ← Clean naming
│   └── meshes/                ← All meshes together
│       └── AL_L.json
└── optic-lobe_v1_1/
    ├── available_rois.json
    └── meshes/
```

### Backward Compatibility
Code still checks old location if new location doesn't exist:

```python
# 1. Check new location: cache/{dataset}/meshes/
cache_mesh_dir = os.path.join(self.script_path, 'cache', dataset_normalized, 'meshes')
if os.path.exists(cache_mesh_dir):
    return cache_mesh_dir

# 2. Fallback to old location: navis_roi_meshes_json/{dataset}/
old_mesh_dir = os.path.join(self.script_path, 'navis_roi_meshes_json', dataset_normalized)
if os.path.exists(old_mesh_dir):
    return old_mesh_dir

# 3. Final fallback: navis_roi_meshes_json/primary_rois/
return os.path.join(self.script_path, 'navis_roi_meshes_json', 'primary_rois')
```

---

## ROI Availability by Dataset

### Hemibrain v1.2.1
- **Total ROIs:** 230
- **Location:** `cache/hemibrain_v1_2_1/available_rois.json`
- **Examples:** AL(L), AL(R), MB(L), MB(R), PB, EB, FB, NO, etc.
- **Coverage:** Whole brain neuropils

### Optic-Lobe v1.1
- **Total ROIs:** 2,690 (!)
- **Location:** `cache/optic-lobe_v1_1/available_rois.json`
- **Includes:**
  - Standard brain ROIs: AL, MB, PB, EB, FB, etc. (same as hemibrain)
  - **Detailed optic lobe columns:** 
    - ME_R_col_01_07 through ME_R_col_36_35 (medulla columns)
    - LO_R_col_01_07 through LO_R_col_36_35 (lobula columns)
    - LOP_R_col_01_07 through LOP_R_col_36_35 (lobula plate columns)
  - **Optic lobe layers:**
    - ME_R_layer_01 through ME_R_layer_10 (medulla layers)
    - LO_R_layer_1 through LO_R_layer_7 (lobula layers)
    - LOP_R_layer_1 through LOP_R_layer_4 (lobula plate layers)

**Why so many?** Optic lobe dataset includes fine-grained retinotopic column segmentation!

---

## Migration Steps

### For Existing Projects

1. **Check current cache usage:**
```bash
# See what's cached
du -sh connection_data/*_cache/*
du -sh cache/*/
```

2. **No action needed!**
   - Code automatically uses new structure
   - Falls back to old locations if files exist
   - New cache files go to new location

3. **Optional: Clean up old cache (saves disk space):**
```bash
# Remove old scattered caches
rm -rf connection_data/neuron_cache/
rm -rf connection_data/synapse_cache/

# Keep: connection_data/brain_transforms_info.txt (documentation)
# Keep: connection_data/plot3d_* (visualization outputs)
```

### For New Projects

Just use the code normally - everything goes to `cache/{dataset}/` automatically!

```python
from src import coana as ca

vis = ca.VisualizeSkeleton(
    neuron_layers=[{"type": "TmY9"}],
    dataset='optic-lobe:v1.1',  # ✓ Will use cache/optic-lobe_v1_1/
    cache_neurons=True,          # ✓ Saves to cache/optic-lobe_v1_1/neurons/
    cache_synapses=True,         # ✓ Saves to cache/optic-lobe_v1_1/synapses/
    brain_mesh='whole'           # ✓ Uses correct OL → JRC2018F transform
)
```

---

## Cache Management Commands

### View Cache Contents
```bash
# Show all cached datasets
ls -lh cache/

# Show hemibrain cache breakdown
du -sh cache/hemibrain_v1_2_1/*

# Show optic-lobe cache breakdown
du -sh cache/optic-lobe_v1_1/*

# Count cached neurons
ls cache/hemibrain_v1_2_1/neurons/ | wc -l

# Count cached synapse files
ls cache/hemibrain_v1_2_1/synapses/ | wc -l

# List available ROIs
cat cache/hemibrain_v1_2_1/available_rois.json | grep -o '"[^"]*"' | wc -l
```

### Clear Specific Caches
```bash
# Clear all hemibrain neuron cache
rm -rf cache/hemibrain_v1_2_1/neurons/

# Clear all hemibrain synapse cache
rm -rf cache/hemibrain_v1_2_1/synapses/

# Clear entire hemibrain cache (keeps directory structure)
rm -rf cache/hemibrain_v1_2_1/neurons/ cache/hemibrain_v1_2_1/synapses/

# Clear all cached data (nuclear option)
rm -rf cache/*/neurons/ cache/*/synapses/
```

### Backup Cache
```bash
# Backup specific dataset
tar -czf hemibrain_cache_backup.tar.gz cache/hemibrain_v1_2_1/

# Backup all cached neurons (for transfer between machines)
tar -czf neurons_backup.tar.gz cache/*/neurons/

# Restore backup
tar -xzf hemibrain_cache_backup.tar.gz
```

---

## Testing

### Test 1: Hemibrain Caching (Old Behavior Should Still Work)
```python
from src import coana as ca

vis = ca.VisualizeSkeleton(
    neuron_layers=[{"type": "TmY9"}],
    dataset='hemibrain:v1.2.1',
    cache_neurons=True,
    cache_synapses=True
)
vis.plot_skeleton()
vis.plot_synapses()

# Check: cache/hemibrain_v1_2_1/neurons/ should have .pkl files
# Check: cache/hemibrain_v1_2_1/synapses/ should have .parquet files
```

### Test 2: Optic-Lobe Transform (Fixed)
```python
vis = ca.VisualizeSkeleton(
    neuron_layers=[{"type": "Tm1"}],
    dataset='optic-lobe:v1.1',
    brain_mesh='whole'  # Should transform OL → JRC2018F (no errors!)
)
vis.plot_skeleton()

# Check: Should complete without "Target 'JRC2018F' has no known bridging" error
```

### Test 3: ROI Loading from Cache
```python
# Should load from cache/hemibrain_v1_2_1/available_rois.json
rois_hb = vis._get_available_rois(use_cache=True, fetch_online=False)
print(f"Hemibrain ROIs: {len(rois_hb)}")  # Should be 230

# Should load from cache/optic-lobe_v1_1/available_rois.json
vis2 = ca.VisualizeSkeleton(
    neuron_layers=[{"type": "Tm1"}],
    dataset='optic-lobe:v1.1'
)
rois_ol = vis2._get_available_rois(use_cache=True, fetch_online=False)
print(f"Optic-lobe ROIs: {len(rois_ol)}")  # Should be 2690
```

---

## Disk Space Estimates

### Per Dataset
```
cache/hemibrain_v1_2_1/
├── available_rois.json      ~3 KB
├── connections.parquet      ~15 MB
├── neuron_index.parquet     ~700 KB
├── meshes/                  ~5 MB (63 ROI meshes)
├── neurons/                 Variable (100KB - 10MB per file)
└── synapses/                Variable (10KB - 5MB per file)
```

### Typical Usage
- **Minimal** (no caching): ~21 MB
- **Light** (cache 10 neuron groups): ~50 MB
- **Moderate** (cache 50 neuron groups): ~200 MB
- **Heavy** (cache 200+ neuron groups): ~1 GB

### Optic-Lobe
- More ROIs → Potentially more mesh files
- More detailed columns → Same neuron cache size per neuron

---

## Benefits Summary

### 1. Organization
- ✅ One folder per dataset
- ✅ Clear cache structure
- ✅ Easy to understand what's cached

### 2. Correctness
- ✅ Fixed optic-lobe transforms
- ✅ Proper template coordinate systems
- ✅ Dataset-specific handling

### 3. Maintainability
- ✅ Easy cache management
- ✅ Simple backup/restore
- ✅ Clear separation of concerns

### 4. Performance
- ✅ Fast cache lookups (organized by dataset)
- ✅ Efficient disk usage
- ✅ No duplicate files

---

## Related Documentation

- [Cache Enhancements](CACHE_ENHANCEMENTS.md) - Caching features
- [Quick Reference](QUICK_REFERENCE.md) - Multi-dataset support
- [Installation Guide](INSTALLATION.md) - Setup instructions

---

**Version:** 1.0  
**Date:** November 21, 2025  
**Changes:** Unified cache structure, fixed optic-lobe transforms, reorganized mesh directories
