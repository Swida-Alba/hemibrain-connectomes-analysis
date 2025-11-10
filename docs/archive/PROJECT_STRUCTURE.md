# Project Structure

**Last Updated:** October 27, 2025

---

## Overview

This document describes the organization of the hemibrain-connectomes-analysis project.

---

## Directory Structure

```
hemibrain-connectomes-analysis-now/
│
├── README.md                          # Main documentation
├── DOCUMENTATION_STRUCTURE.md         # Documentation index
├── PROJECT_STRUCTURE.md              # This file
│
├── Core Library Files
│   ├── coana.py                      # Main analysis library
│   ├── statvis.py                    # Visualization library  
│   ├── ManageCache.py                # Cache management utility
│   └── setup.py                      # Installation script
│
├── Template Scripts (User-friendly entry points)
│   ├── FindDirect.py                 # Template for finding direct connections
│   ├── FindPath.py                   # Template for finding paths
│   ├── FindSynapse.py                # Template for synapse analysis
│   └── plot3dSkeleton.py             # Template for 3D visualization
│
├── User-Specific Analysis Scripts
│   ├── FindDirect_VTaMe.py           # VTaMe direct connections
│   ├── FindPath_Kun.py               # Kun's path analysis (L3 → l-LNv)
│   ├── FindPath_PPL1_VTaMe.py        # PPL1 → VTaMe paths
│   └── FindPath_VTaMe.py             # VTaMe path analysis
│
├── examples/                          # Example scripts
│   ├── Example_CachingDemo.py        # Cache system demonstration
│   └── Example_ParallelProcessing.py # Parallel processing example
│
├── tests/                             # Test suite
│   ├── test_cache_v4.py              # Cache v4 tests
│   ├── test_connection_metrics.py     # Connection metrics tests
│   ├── test_filter_by.py             # Filter functionality tests
│   ├── test_folder_naming.py         # Folder naming tests
│   ├── test_neuprint_login.py        # Neuprint login tests
│   └── Test_DraggableNodes.py        # Interactive visualization tests
│
├── dev/                               # Development files
│   ├── cache_v4_methods.py           # v4 cache implementation (proposal)
│   └── integrate_cache_v4.py         # v4 integration script (WIP)
│
├── docs/                              # Documentation (37 files)
│   ├── CacheSystem_Guide.md          # Consolidated cache guide
│   ├── ForwardOnly_Guide.md          # Consolidated forward_only guide
│   ├── [27 feature documentation files]
│   ├── [3 project documentation files]
│   └── archive/                      # Historical fixes (10 files)
│
├── datasets/                          # Neuron data files
│   └── [Various .csv and .xlsx files]
│
├── navis_roi_meshes_json/            # ROI mesh data
├── neuprint_cache/                   # Cache storage (git-ignored)
├── assets/                            # Documentation images
└── __pycache__/                      # Python cache (git-ignored)
```

---

## File Categories

### 🔧 Core Library (4 files)

**Purpose:** Main library code that powers the analysis

| File | Description | Lines |
|------|-------------|-------|
| `coana.py` | Main connectome analysis class | 4,331 |
| `statvis.py` | Statistics and visualization functions | 1,218 |
| `ManageCache.py` | Interactive cache management utility | 249 |
| `setup.py` | Package installation and dependencies | 19 |

**Usage:** These are imported by user scripts, not run directly

---

### 📝 Template Scripts (4 files)

**Purpose:** Ready-to-use templates for common analyses

| File | Description | Use Case |
|------|-------------|----------|
| `FindDirect.py` | Find direct connections | Source → Target (1-hop) |
| `FindPath.py` | Find indirect paths | Source → ... → Target (multi-hop) |
| `FindSynapse.py` | Analyze synapses | Synapse-level analysis |
| `plot3dSkeleton.py` | 3D visualization | Neuron skeleton rendering |

**Usage:** Copy and modify these templates for your analysis

**How to use:**
1. Open the template file
2. Edit the parameters (token, sourceNeurons, targetNeurons, etc.)
3. Run: `python FindDirect.py` or `python FindPath.py`

---

### 👤 User-Specific Scripts (4 files)

**Purpose:** Specific analyses for particular research questions

| File | Analysis |
|------|----------|
| `FindDirect_VTaMe.py` | VTaMe direct connections |
| `FindPath_Kun.py` | L3_R → l-LNv_R pathfinding |
| `FindPath_PPL1_VTaMe.py` | PPL1 → VTaMe paths |
| `FindPath_VTaMe.py` | VTaMe pathway analysis |

**Usage:** Run directly for specific analyses

---

### 📚 Examples (2 files in `examples/`)

**Purpose:** Demonstrate specific features

| File | Demonstrates |
|------|-------------|
| `Example_CachingDemo.py` | Cache system usage and benefits |
| `Example_ParallelProcessing.py` | Parallel processing setup |

**Usage:** 
```bash
python examples/Example_CachingDemo.py
python examples/Example_ParallelProcessing.py
```

---

### 🧪 Tests (6 files in `tests/`)

**Purpose:** Automated testing for quality assurance

| File | Tests |
|------|-------|
| `test_cache_v4.py` | Cache v4 functionality |
| `test_connection_metrics.py` | Connection calculations |
| `test_filter_by.py` | Filter functionality |
| `test_folder_naming.py` | Output folder naming |
| `test_neuprint_login.py` | Neuprint authentication |
| `Test_DraggableNodes.py` | Interactive network features |

**Usage:**
```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_cache_v4.py
```

---

### 🚧 Development (2 files in `dev/`)

**Purpose:** Work-in-progress features and proposals

| File | Purpose |
|------|---------|
| `cache_v4_methods.py` | Proposed v4 cache implementation |
| `integrate_cache_v4.py` | Integration script for v4 cache |

**Status:** Experimental - not used in production code

**Note:** v3 is the current stable cache system. v4 is a proposal for future enhancement.

---

## Quick Start Guide

### For Users

1. **Copy a template:**
   ```bash
   cp FindDirect.py my_analysis.py
   ```

2. **Edit parameters:**
   - Add your Neuprint token
   - Set source and target neurons
   - Adjust filters (min_synapse_num, etc.)

3. **Run:**
   ```bash
   python my_analysis.py
   ```

### For Developers

1. **Library code:** Edit `coana.py` or `statvis.py`
2. **Add tests:** Create `tests/test_new_feature.py`
3. **Add examples:** Create `examples/Example_NewFeature.py`
4. **Update docs:** Add `docs/NewFeature_Guide.md`

---

## File Naming Conventions

### Scripts
- **Templates:** `FindXXX.py` (e.g., `FindDirect.py`, `FindPath.py`)
- **User analyses:** `FindXXX_ProjectName.py` (e.g., `FindPath_Kun.py`)
- **Examples:** `Example_XXX.py` (e.g., `Example_CachingDemo.py`)
- **Tests:** `test_xxx.py` or `Test_XXX.py`

### Documentation
- **Guides:** `XXX_Guide.md` (e.g., `CacheSystem_Guide.md`)
- **Features:** `XXX_Documentation.md` or `XXX_Feature.md`
- **References:** `XXX_QuickRef.md` or `XXX_QuickStart.md`

---

## Data Directories

### `datasets/`
Contains neuron metadata files downloaded from Neuprint:
- `*_allneurons_neuron_df.csv` - All neurons (including type=None)
- `*_alltypes_neuron_df.csv` - Only typed neurons
- `*_roi_count_df.csv` - ROI counts

### `neuprint_cache/`
Local cache storage (automatically created):
```
neuprint_cache/
  optic-lobe_v1.1/
    connections/          # Cached connection data
    neuron_registry.parquet
    cache_index.parquet
```

**Note:** This folder is git-ignored and created automatically when caching is enabled.

---

## Installation

### Standard Installation
```bash
python setup.py
```

### Development Installation
```bash
pip install -e .
pytest tests/  # Run tests
```

---

## Key Files to Know

### Essential
- `README.md` - Start here
- `coana.py` - Main library
- `FindDirect.py` or `FindPath.py` - Templates to copy

### Documentation
- `DOCUMENTATION_STRUCTURE.md` - Doc index
- `docs/CacheSystem_Guide.md` - Cache guide
- `docs/ForwardOnly_Guide.md` - Path validation guide

### Utilities
- `ManageCache.py` - Manage cache interactively
- `setup.py` - Install dependencies

---

## Contributing

1. **Add new feature:** Edit `coana.py`
2. **Add test:** Create `tests/test_feature.py`
3. **Add example:** Create `examples/Example_Feature.py`
4. **Document:** Create `docs/Feature_Guide.md`
5. **Update:** Update this `PROJECT_STRUCTURE.md`

---

## Maintenance

### Regular Tasks
- Clear old cache: `python ManageCache.py`
- Run tests: `pytest tests/`
- Update documentation: Edit files in `docs/`

### When Adding Files
- Template script → Root directory
- Example → `examples/`
- Test → `tests/`
- Development → `dev/`
- Documentation → `docs/`

---

## Summary Statistics

**Total Files by Type:**
- Core library: 4 files (~6,000 lines)
- Templates: 4 files (~120 lines)
- User scripts: 4 files (~140 lines)
- Examples: 2 files (~390 lines)
- Tests: 6 files (~440 lines)
- Development: 2 files (~500 lines)
- Documentation: 38 files

**Total:** 60+ Python/markdown files

---

**See Also:**
- [README.md](README.md) - Main documentation
- [DOCUMENTATION_STRUCTURE.md](DOCUMENTATION_STRUCTURE.md) - Documentation index
- [docs/](docs/) - All documentation files
