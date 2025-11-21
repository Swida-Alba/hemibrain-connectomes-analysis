# VisualizePath Updates - November 2025

## Summary of Changes

This document summarizes the recent enhancements to the `VisualizePath` class, including connection matrix input support, individual visualization control, and improved format detection.

---

## 1. Connection Matrix Input Support ✨

### Overview
`VisualizePath` now accepts connection matrices as direct input, in addition to edge-lists and path-based formats.

### Supported Matrix Types
- **Square matrices (NxN)**: Symmetric or asymmetric neuron-to-neuron connectivity
- **Rectangular matrices (NxM)**: Source population to target population connections
- **Named or numeric indices**: Auto-generates node names (N0, N1, ...) if needed

### Detection Logic
```python
# Automatic detection criteria:
# 1. 2D DataFrame with at least 2x2 shape
# 2. All columns contain numeric data
# 3. Not an edge-list (no source/target/weight columns)
```

### Example Usage
```python
import pandas as pd
import numpy as np
from vispath_pkg import VisualizePath

# Square matrix (5x5)
matrix_square = pd.DataFrame(
    np.random.poisson(3, (5, 5)),
    index=['A', 'B', 'C', 'D', 'E'],
    columns=['A', 'B', 'C', 'D', 'E']
)
vp = VisualizePath(matrix_square)
vp.visualize()
# Console: ✓ Recognized input format: connection matrix (5x5 DataFrame)

# Rectangular matrix (10x12)
matrix_rect = pd.DataFrame(
    np.random.poisson(2, (10, 12)),
    index=[f"Source_{i}" for i in range(10)],
    columns=[f"Target_{j}" for j in range(12)]
)
vp = VisualizePath(matrix_rect)
vp.visualize()
# Console: ✓ Recognized input format: connection matrix (10x12 DataFrame)

# Numeric indices (auto-generates node names)
matrix_numeric = pd.DataFrame(np.random.poisson(2, (6, 6)))
vp = VisualizePath(matrix_numeric)
vp.visualize()
# Node names auto-generated: N0, N1, N2, N3, N4, N5
```

### Internal Processing
1. **Detection**: Identifies matrix format based on shape and data types
2. **Conversion**: Converts matrix to edge-list format internally
3. **Filtering**: Removes zero and NaN connections
4. **Processing**: Proceeds with standard visualization pipeline

---

## 2. Individual Visualization Control 🎛️

### Overview
The `visualize()` method now accepts three boolean parameters to control which visualizations are generated.

### New Parameters
```python
def visualize(self, plot_heatmap=True, plot_Sankey=True, plot_network=True):
    """
    Generate selected visualizations.
    
    Parameters
    ----------
    plot_heatmap : bool, default=True
        Generate interactive heatmap with hierarchical clustering
    plot_Sankey : bool, default=True
        Generate Sankey flow diagram
    plot_network : bool, default=True
        Generate interactive network graph
    """
```

### Example Usage
```python
from vispath_pkg import VisualizePath

vp = VisualizePath('data.xlsx')

# Generate all visualizations (default)
vp.visualize()

# Generate only heatmap
vp.visualize(plot_heatmap=True, plot_Sankey=False, plot_network=False)

# Generate only Sankey and network (skip heatmap)
vp.visualize(plot_heatmap=False, plot_Sankey=True, plot_network=True)

# Skip all visualizations (data processing only)
vp.visualize(plot_heatmap=False, plot_Sankey=False, plot_network=False)
```

### Benefits
- **Performance**: Faster execution when only specific visualizations are needed
- **Disk space**: Saves storage by generating only required files
- **Flexibility**: Adapt to different use cases (exploration, publication, analysis)
- **Workflow**: Iteratively refine specific visualizations without regenerating all

### Use Cases
| Scenario | Configuration | Output |
|----------|---------------|--------|
| Quick exploration | `plot_network=True` only | Fast network view |
| Publication figure | `plot_heatmap=True` only | Clean matrix visualization |
| Flow analysis | `plot_Sankey=True` only | Sankey diagram |
| Complete analysis | All `True` (default) | All three visualizations |
| Data processing | All `False` | Excel data only |

---

## 3. Enhanced Format Detection 🔍

### Overview
Clear console messages now indicate the recognized input format during data loading.

### Detection Messages

#### Connection Matrix
```
✓ Recognized input format: connection matrix (10x12 DataFrame)
```

#### Edge-List (Various Formats)
```
✓ Recognized input format: edge-list DataFrame (columns: ['source', 'target', 'weight'])
✓ Recognized input format: edge-list DataFrame (columns: ['from', 'to', 'weight'])
✓ Recognized input format: edge-list DataFrame (columns: ['bodyId_pre', 'bodyId_post', 'weight'])
```

#### Path-Based
```
✓ Recognized input format: generic DataFrame (columns: ['path_block', 'weights'])
```

### Edge-List Column Name Flexibility
The system recognizes multiple column naming conventions:

| Column Type | Accepted Names | Examples |
|-------------|---------------|----------|
| **Source** | `source`, `from`, `pre`, `*_pre` | `source`, `bodyId_pre`, `type_pre` |
| **Target** | `target`, `to`, `post`, `*_post` | `target`, `bodyId_post`, `type_post` |
| **Weight** | `weight`, `weights` | `weight` |

### Example
```python
# Standard format
df1 = pd.DataFrame({
    'source': ['A', 'B'],
    'target': ['B', 'C'],
    'weight': [10, 15]
})

# Alternative format
df2 = pd.DataFrame({
    'from': ['A', 'B'],
    'to': ['B', 'C'],
    'weight': [10, 15]
})

# NeuPrint bodyId format
df3 = pd.DataFrame({
    'bodyId_pre': [123, 456],
    'bodyId_post': [456, 789],
    'weight': [10, 15]
})

# All automatically recognized!
```

---

## 4. Code Optimizations 🚀

### Bug Fixes
- **Numeric column names**: Fixed AttributeError when DataFrames have numeric column names
- **String conversion**: Added robust string conversion in column name detection

### Improvements
```python
# Before (would crash with numeric columns)
def has_prefixed_cols(cols):
    has_pre = any(c.endswith('_pre') for c in cols)  # AttributeError if c is int

# After (handles numeric columns)
def has_prefixed_cols(cols):
    str_cols = [str(c) for c in cols]
    has_pre = any(c.endswith('_pre') for c in str_cols)
```

---

## 5. Testing & Validation ✅

### Comprehensive Test Suite
Created `Example_AllInputFormats_Test.py` to validate all features:

**Test Coverage:**
1. ✅ Square connection matrix (5x5)
2. ✅ Rectangular connection matrix (10x12)
3. ✅ Matrix without named index/columns
4. ✅ Edge-list (source, target, weight)
5. ✅ Edge-list (from, to, weight)
6. ✅ Edge-list (bodyId_pre, bodyId_post, weight)
7. ✅ Path-based format
8. ✅ Individual visualization control

**All tests passed successfully!**

### Running Tests
```bash
# Run comprehensive test
PYTHONPATH=vispath-subproject/src python examples/Example_AllInputFormats_Test.py

# Output:
# ================================================================================
# ✓ All tests passed successfully!
# ================================================================================
```

---

## 6. Documentation Updates 📚

### Updated Files
1. **`vispath-subproject/README.md`**
   - Added connection matrix format documentation
   - Updated Quick Start with matrix examples
   - Documented individual visualization parameters

2. **`VISUALIZATION_UPDATE.md`**
   - Added section on connection matrix support
   - Documented individual visualization control
   - Added enhanced format detection details

3. **`docs/visualizations/README.md`**
   - Added connection matrix to data type recommendations
   - Updated workflow examples with matrix input
   - Added individual visualization examples

### Quick Reference

#### Input Format Summary
| Format | Columns | Shape | Auto-Detect |
|--------|---------|-------|-------------|
| **Connection Matrix** | Numeric data | NxM | ✓ |
| **Edge-List** | source/target/weight (flexible) | Nx3+ | ✓ |
| **Path-Based** | path_block, weights | Nx2+ | ✓ |

#### Visualization Control Summary
| Method Call | Heatmap | Sankey | Network |
|-------------|---------|--------|---------|
| `visualize()` | ✓ | ✓ | ✓ |
| `visualize(True, False, False)` | ✓ | ✗ | ✗ |
| `visualize(False, True, True)` | ✗ | ✓ | ✓ |

---

## 7. Migration Guide 🔄

### For Existing Users
No breaking changes! All existing code continues to work:

```python
# Existing code (still works)
vp = VisualizePath('paths.xlsx')
vp.visualize()  # Generates all three visualizations

# New features are opt-in
vp.visualize(plot_heatmap=True, plot_Sankey=False, plot_network=False)
```

### New Capabilities
```python
# Connection matrix input (NEW)
import pandas as pd
matrix = pd.read_csv('matrix.csv', index_col=0)
vp = VisualizePath(matrix)
vp.visualize()

# Individual visualization control (NEW)
vp.visualize(plot_heatmap=True, plot_Sankey=False, plot_network=True)
```

---

## 8. Examples 💡

### Example 1: Connection Matrix Heatmap Only
```python
import pandas as pd
import numpy as np
from vispath_pkg import VisualizePath

# Create connectivity matrix
matrix = pd.DataFrame(
    np.random.poisson(3, (10, 10)),
    index=[f"Neuron_{i}" for i in range(10)],
    columns=[f"Neuron_{i}" for i in range(10)]
)

# Generate only heatmap for publication
vp = VisualizePath(matrix, output_folder='./publication_figures')
vp.visualize(plot_heatmap=True, plot_Sankey=False, plot_network=False)
```

### Example 2: Edge-List Network Exploration
```python
# Load edge-list
edges = pd.read_csv('connections.csv')  # bodyId_pre, bodyId_post, weight

# Generate only network for interactive exploration
vp = VisualizePath(edges)
vp.showfig = True  # Open in browser
vp.visualize(plot_heatmap=False, plot_Sankey=False, plot_network=True)
```

### Example 3: Multi-Format Analysis
```python
# Compare different input formats
formats = {
    'matrix': connection_matrix,
    'edgelist': edge_dataframe,
    'paths': path_dataframe
}

for name, data in formats.items():
    vp = VisualizePath(data, output_folder=f'./analysis/{name}')
    vp.visualize()  # All visualizations for comparison
```

---

## Summary

### Key Features Added
✅ Connection matrix input support (NxM, auto-naming)  
✅ Individual visualization control (plot_heatmap, plot_Sankey, plot_network)  
✅ Enhanced format detection with clear console messages  
✅ Robust handling of numeric column names  
✅ Comprehensive test coverage  
✅ Updated documentation across all files  

### Benefits
- **Flexibility**: Support for more input formats
- **Efficiency**: Generate only needed visualizations
- **Usability**: Clear feedback on data format recognition
- **Reliability**: Comprehensive testing ensures robustness

### Next Steps
- Try connection matrix input with your data
- Experiment with individual visualization control
- Check console messages to verify format detection
- Run `Example_AllInputFormats_Test.py` to see all features in action

---

**Version**: November 2025  
**Status**: All features tested and documented ✓
