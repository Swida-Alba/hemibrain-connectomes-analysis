# Project Reorganization Plan

## Current Issues
- Main scripts scattered in root directory
- Test files mixed with production code
- HTML output files in root
- Unclear separation between core functionality and utilities

## Proposed Structure

```
hemibrain-connectomes-analysis/
├── README.md
├── LICENSE
├── setup.py
├── pyproject.toml
├── requirements.txt
│
├── src/                           # Main source code
│   ├── __init__.py
│   ├── coana.py                   # Core analysis module
│   ├── statvis.py                 # Statistics and visualization
│   ├── vispath.py                 # Path visualization
│   │
│   ├── core/                      # Core functionality
│   │   ├── __init__.py
│   │   ├── find_direct.py         # Direct connection finding
│   │   ├── find_path.py           # Path finding algorithms
│   │   ├── find_synapse.py        # Synapse analysis
│   │   └── cache_manager.py       # Cache management (ManageCache.py)
│   │
│   ├── plotting/                  # Visualization modules
│   │   ├── __init__.py
│   │   ├── plot_path.py           # Path plotting
│   │   ├── plot_skeleton.py       # 3D skeleton plotting
│   │   └── network_viz.py         # Network visualizations
│   │
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       ├── path_utils.py
│       └── data_processing.py
│
├── scripts/                       # Executable scripts
│   ├── FindDirect.py              # Main direct connection script
│   ├── FindPath.py                # Main pathfinding script
│   ├── FindPath_Kun.py            # Variant scripts
│   ├── FindPath_Kun_loop.py
│   ├── FindPath_VTaMe.py
│   ├── FindPath_PPL1_VTaMe.py
│   ├── FindDirect_VTaMe.py
│   ├── PlotPath.py
│   └── PlotPath_kun.py
│
├── notebooks/                     # Jupyter notebooks
│   └── FetchNeurons.ipynb
│
├── tests/                         # All test files
│   ├── __init__.py
│   ├── test_type_filter.py
│   ├── test_bodyid_filter.py
│   ├── test_both_filters.py
│   ├── test_metric_toggle.py
│   ├── debug_type_filter.py
│   ├── inspect_type_filter.py
│   └── check_paths.py
│
├── test_output/                   # Test output files
│   ├── html/                      # HTML test outputs
│   │   ├── test_3point_and_toggle_fix.html
│   │   ├── test_3point_debug.html
│   │   ├── test_heatmap_3point_fix.html
│   │   ├── test_heatmap_improvements.html
│   │   ├── test_rect_10x20.html
│   │   ├── test_rect_20x10.html
│   │   ├── test_rect_30x10.html
│   │   ├── test_square_15x15.html
│   │   └── test_square_cells_fixed.html
│   └── data/                      # Test data outputs
│
├── examples/                      # Example scripts and usage
│   └── (existing example files)
│
├── docs/                          # Documentation
│   ├── (existing doc files)
│   └── REORGANIZATION_PLAN.md
│
├── datasets/                      # Data files
│   └── (existing CSV files)
│
├── assets/                        # Images and resources
│   └── (existing asset files)
│
├── csv_visualization/             # CSV visualization outputs
│   └── (existing HTML files)
│
├── cache/                         # Cache directory (renamed from neuprint_cache)
│   └── (cache files)
│
├── navis_roi_meshes_json/         # ROI mesh data
│   └── (existing mesh files)
│
├── dev/                           # Development and experimental code
│   └── (existing dev files)
│
└── output/                        # Generated outputs (gitignored)
    ├── sankey/
    ├── networks/
    └── plots/
```

## Migration Steps

### Phase 1: Create New Structure
1. Create `src/` directory with subdirectories
2. Create `scripts/` directory
3. Create `test_output/html/` directory

### Phase 2: Move Core Modules
1. Keep `coana.py`, `statvis.py`, `vispath.py` in `src/`
2. Move utility scripts to `src/utils/`
3. Create organized submodules in `src/core/` and `src/plotting/`

### Phase 3: Move Scripts
1. Move all `Find*.py` and `Plot*.py` to `scripts/`
2. Update imports in scripts to reference `src/`

### Phase 4: Organize Tests
1. Move all `test_*.py` files to `tests/`
2. Move all `.html` test outputs to `test_output/html/`
3. Move `debug_*.py` and `inspect_*.py` to `tests/`

### Phase 5: Move Notebooks
1. Move `.ipynb` files to `notebooks/`

### Phase 6: Rename Cache
1. Rename `neuprint_cache/` to `cache/`

### Phase 7: Update Imports
1. Update all import statements in scripts
2. Update documentation references
3. Test all scripts still work

### Phase 8: Update Configuration
1. Update `.gitignore` to include `output/`
2. Update `setup.py` and `pyproject.toml` with new structure
3. Update README.md with new structure

## Benefits
- Clear separation of concerns
- Easier to find files
- Better for IDE navigation
- Cleaner root directory
- Easier testing and CI/CD integration
- Better for package distribution
