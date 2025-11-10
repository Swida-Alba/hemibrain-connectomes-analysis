# Import Strategy Update - Using vispath-subproject

**Date**: 2025-11-07  
**Status**: ✅ COMPLETE

## Overview

Updated all scripts and examples to use the **vispath-subproject** as the primary import source, with automatic fallback to the `src/vispath.py` for backward compatibility. This provides better maintainability and aligns with the standalone sub-project architecture.

## Changes Made

### 1. **Import Strategy**

**New Pattern (Applied to all scripts and examples):**

```python
import sys
from pathlib import Path

# Add vispath-subproject to Python path for local development
vispath_pkg_path = Path(__file__).parent.parent / 'vispath-subproject' / 'src'
if vispath_pkg_path.exists():
    sys.path.insert(0, str(vispath_pkg_path))

# Try to import from vispath_pkg (standalone sub-project) first,
# fall back to vispath (src) for backward compatibility
try:
    from vispath_pkg import VisualizePath
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from vispath import VisualizePath
```

**Old Pattern (Replaced):**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vispath import VisualizePath
```

### 2. **Files Updated**

#### Scripts (`scripts/`)
1. ✅ `PlotPath.py` - Updated import strategy

#### Examples (`examples/`)
1. ✅ `Example_Clustering_Demo.py` - Updated import strategy
2. ✅ `Example_FilePicker.py` - Updated import strategy
3. ✅ `Example_SimpleEdgeList.py` - Updated import strategy
4. ✅ `Example_VisualizeSelectedPaths.py` - Updated import strategy (includes `visualize_paths`)

**Total Files Updated**: 5

### 3. **Documentation Updates**

#### vispath-subproject/README.md
- ✅ **Fixed incorrect data format description**
- Added comprehensive format specifications for both:
  - **Path-based format**: `path_block` with `weights` as lists
  - **Edge-list format**: Flexible column naming with auto-conversion

**Before (Incorrect)**:
```markdown
### Path-based format:
| path_id | source | layer_1 | target | weight | probability |
```

**After (Correct)**:
```markdown
### 1. Path-based format (Multi-hop paths):
| path_block     | weights      | connection_ratios | traversal_probabilities |
|----------------|--------------|-------------------|-------------------------|
| A -> B -> C    | [10, 5]      | [0.5, 0.3]        | [0.8, 0.6]              |
```

#### vispath-subproject/INSTALLATION.md
- ✅ Added "Input Data Formats" section with code examples
- Documented both path-based and edge-list formats
- Provided DataFrame creation examples

## Benefits

### 1. **Better Architecture**
- Scripts use the **standalone sub-project** as intended
- Clear separation between standalone (vispath_pkg) and integrated (vispath) usage
- Aligns with the modular design goals

### 2. **Backward Compatibility**
- Automatic fallback to `src/vispath.py` if sub-project not available
- Works with both local development and installed packages
- No breaking changes for existing users

### 3. **Improved Maintainability**
- Single source of truth: vispath-subproject
- Changes to vispath automatically reflected in scripts/examples
- Easier to test standalone functionality

### 4. **Flexible Usage**
Works in three scenarios:
1. **Local development**: Uses symlinked vispath-subproject
2. **Standalone installation**: `pip install vispath-subproject`
3. **Full installation**: `pip install hemibrain-connectomes-analysis`

## Import Behavior

### Scenario 1: Local Development
```
Project Root/
├── vispath-subproject/
│   └── src/vispath_pkg/  ← USED (via symlink to src/vispath.py)
├── src/
│   └── vispath.py
└── scripts/
    └── PlotPath.py  ← Imports from vispath_pkg
```

### Scenario 2: Standalone Package Installed
```bash
pip install git+https://...#subdirectory=vispath-subproject

# Scripts use installed vispath_pkg
from vispath_pkg import VisualizePath  # ✅ Works
```

### Scenario 3: Full Package Installed
```bash
pip install git+https://...

# Scripts can use either
from vispath_pkg import VisualizePath  # ✅ Works (if sub-project installed)
from vispath import VisualizePath      # ✅ Works (fallback)
```

## Testing

### Test Import Strategy
```python
# Test 1: vispath-subproject available
python scripts/PlotPath.py
# Expected: Uses vispath_pkg

# Test 2: Only src/vispath.py available
# (Temporarily rename vispath-subproject)
python scripts/PlotPath.py
# Expected: Falls back to src/vispath, still works

# Test 3: Standalone installation
pip install -e vispath-subproject
python -c "from vispath_pkg import VisualizePath; print('✅ OK')"
# Expected: ✅ OK
```

### Verified Files
All 5 updated files tested:
- ✅ No syntax errors
- ✅ Import logic correct
- ✅ Fallback mechanism works

## Migration Guide

### For Users

**No action required!** Scripts work automatically in all scenarios:
- Local development
- Standalone package installation
- Full package installation

### For Developers

**Adding new scripts?** Use the new import pattern:

```python
import sys
from pathlib import Path

# Add vispath-subproject to Python path
vispath_pkg_path = Path(__file__).parent.parent / 'vispath-subproject' / 'src'
if vispath_pkg_path.exists():
    sys.path.insert(0, str(vispath_pkg_path))

# Import with fallback
try:
    from vispath_pkg import VisualizePath
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from vispath import VisualizePath
```

**Key Points:**
1. Try vispath_pkg first (sub-project)
2. Fallback to src/vispath (backward compatibility)
3. Works in all installation scenarios

## Documentation Corrections

### Data Format Clarification

**Issue**: README.md showed incorrect path-based format
**Fix**: Documented actual format used by VisualizePath

**Correct Formats:**

#### Path-based (Multi-hop)
```python
df = pd.DataFrame({
    'path_block': ['A -> B -> C', 'A -> D -> C'],
    'weights': [[10, 5], [15, 8]],  # List of weights for each hop
    'connection_ratios': [[0.5, 0.3], [0.6, 0.4]],  # Optional
    'traversal_probabilities': [[0.8, 0.6], [0.9, 0.7]]  # Optional
})
```

#### Edge-list (Direct connections)
```python
df = pd.DataFrame({
    'source': ['A', 'B', 'D'],
    'target': ['B', 'C', 'C'],
    'weight': [10, 5, 8],
    'ratio': [0.5, 0.3, 0.4],  # Optional
    'probability': [0.8, 0.6, 0.7]  # Optional
})
```

**Column Name Flexibility:**
- Source: `source`, `from`, `pre`, `*_pre`
- Target: `target`, `to`, `post`, `*_post`
- Weight: `weight`, `weights`, `synapse_count`, `count`

## Related Documentation

- **Standalone Architecture**: [VISPATH_STANDALONE_REORGANIZATION.md](VISPATH_STANDALONE_REORGANIZATION.md)
- **Sub-project Installation**: [vispath-subproject/INSTALLATION.md](../vispath-subproject/INSTALLATION.md)
- **Data Format Guide**: [vispath-subproject/README.md](../vispath-subproject/README.md)
- **Main Package README**: [README.md](../README.md)

## Summary

✅ **All scripts and examples now use vispath-subproject as primary import**  
✅ **Automatic fallback ensures backward compatibility**  
✅ **Data format documentation corrected and expanded**  
✅ **No breaking changes for existing users**  

The import strategy update is complete and ready for use! 🎉
