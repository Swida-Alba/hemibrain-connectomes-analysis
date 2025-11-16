# Visualization Update Summary

## Changes Made

### 1. Added `output_format` Attribute

**Location**: `src/coana.py`, line ~253

```python
output_format: str = 'xlsx'
'''
output data format: 'xlsx' (default) or 'csv'
'xlsx': save all data in Excel files
'csv': save all data in CSV files in a subfolder named 'output_data'
'''
```

**Usage**:
```python
fc = FindNeuronConnection(
    token='your_token',
    sourceNeurons=['KC.*'],
    targetNeurons=['MBON03'],
    output_format='csv',  # Save as CSV instead of Excel
    ...
)
```

**Implementation Status**: ⚠️ Attribute added, CSV saving logic needs to be implemented in:
- `FindDirectConnections()` - Excel saving at lines ~1475-1567
- `FindPath()` - Excel saving at lines ~1935-2010  
- `FindAllPath()` - Excel saving at lines ~3555-3605

### 2. Deprecated Old Visualization Code

**Deprecated Visualizations**:
- Direct Sankey diagram generation (lines ~2225-2540 in FindPath, ~3720-4080 in FindAllPath)
- Direct heatmap generation using `CreateHeatmap` (lines ~2580-2610, ~4140-4170)

**Current Recommended Approach**:
- Use **VisualizePath** class for all visualizations
- Generates 3 consistent visualizations:
  1. `network_*.html` - Interactive Cytoscape.js network
  2. `sankey_*.html` - Plotly Sankey diagram  
  3. `heatmap_*.html` - Connection matrix heatmap

**VisualizePath Usage** (already implemented):
```python
# In FindDirectConnections (lines ~1695-1780)
vp = VisualizePath(
    path_file=self.conn_type,  # or self.conn_group for custom groups
    output_folder=self.direct_folder,
    showfig=self.showfig
)
vp.visualize()

# In FindPath (lines ~2545-2577)  
vp = VisualizePath(
    path_file=paths_to_visualize,
    output_folder=self.path_folder,
    showfig=self.showfig
)
vp.visualize()

# In FindAllPath (lines ~4095-4138)
vp = VisualizePath(
    path_file=paths_to_visualize,
    output_folder=self.allpath_folder,
    showfig=self.showfig
)
vp.visualize()
```

## Files That Will Be Removed (Deprecated)

When old visualization code is fully removed, these files will no longer be generated:

**FindPath output**:
- ❌ `Sankey_type_path_snp.html` (replaced by `sankey_selected_paths.html`)
- ❌ `Sankey_type_path_ratio.html` (use VisualizePath sankey)
- ❌ `Sankey_type_path_prob.html` (use VisualizePath sankey)
- ❌ `Sankey_bodyId_path.html` (use VisualizePath sankey)
- ❌ `heatmap_path_type.html` (replaced by `heatmap_selected_paths.html`)

**FindAllPath output**:
- ❌ `Sankey_type_allpaths_snp.html` (replaced by `sankey_selected_paths.html`)
- ❌ `Sankey_type_allpaths_ratio.html` (use VisualizePath sankey)
- ❌ `Sankey_type_allpaths_prob.html` (use VisualizePath sankey)
- ❌ `Sankey_bodyId_allpaths.html` (use VisualizePath sankey)
- ❌ `heatmap_allpaths_type.html` (replaced by `heatmap_selected_paths.html`)

## Current VisualizePath Output Files

**FindDirectConnections** (lines ~1695-1780):
- ✅ `network_type.html` - Type-based network
- ✅ `sankey_type.html` - Type-based Sankey
- ✅ `heatmap_type.html` - Type-based heatmap
- ✅ `custom_groups/network_custom_groups.html` - Custom group network (if applicable)
- ✅ `custom_groups/sankey_custom_groups.html` - Custom group Sankey (if applicable)
- ✅ `custom_groups/heatmap_custom_groups.html` - Custom group heatmap (if applicable)

**FindPath / FindAllPath**:
- ✅ `network_selected_paths.html` - Interactive network of paths
- ✅ `sankey_selected_paths.html` - Sankey diagram of paths
- ✅ `heatmap_selected_paths.html` - Connection matrix heatmap

## Migration Guide

### For Users

**Before (Deprecated)**:
```python
# Old approach generated multiple redundant Sankey files
fc.FindPath()
# Generated: Sankey_type_path_snp.html, Sankey_type_path_ratio.html,
#            Sankey_type_path_prob.html, Sankey_bodyId_path.html, etc.
```

**After (Current)**:
```python
# New approach uses VisualizePath for consistent visualizations
fc.FindPath()
# Generates: network_selected_paths.html, sankey_selected_paths.html,
#            heatmap_selected_paths.html
```

### For Developers

To fully remove old visualization code:

1. **Comment out Sankey generation blocks**:
   - Lines ~2225-2540 in `FindPath`
   - Lines ~3720-4080 in `FindAllPath`

2. **Comment out heatmap generation blocks**:
   - Lines ~2580-2610 in `FindPath`  
   - Lines ~4140-4170 in `FindAllPath`

3. **Keep only VisualizePath calls**:
   - Lines ~2545-2577 in `FindPath`
   - Lines ~4095-4138 in `FindAllPath`

## TODO: Implement CSV Output

The `output_format='csv'` attribute is defined but not yet implemented. Implementation needed in:

1. **FindDirectConnections** (lines ~1475-1567):
```python
if self.output_format == 'csv':
    csv_folder = os.path.join(self.direct_folder, 'output_data')
    os.makedirs(csv_folder, exist_ok=True)
    self.parameter_df.to_csv(os.path.join(csv_folder, 'parameters.csv'), index=False)
    self.source_df.to_csv(os.path.join(csv_folder, 'source_info.csv'))
    # ... save all dataframes as CSV
else:
    # Existing Excel save code
    with pd.ExcelWriter(output_excel_name, ...) as writer:
        ...
```

2. **FindPath** (lines ~1935-2010):
   - Similar pattern for path data

3. **FindAllPath** (lines ~3555-3605):
   - Similar pattern for allpath data
   - Already has CSV logic for large bodyId data

## Benefits of VisualizePath Approach

1. **Consistency**: Same visualization style across all methods
2. **Maintainability**: Single codebase for all visualizations
3. **Flexibility**: Easy to customize colors, layouts, and styles
4. **Completeness**: Always generates network + sankey + heatmap trio
5. **Standalone**: Can be used independently with any edge-list data

## Summary

- ✅ `output_format` attribute added
- ✅ Old visualization code documented as deprecated
- ✅ VisualizePath already implemented and working
- ⚠️ CSV saving logic still needs implementation
- ⚠️ Old visualization code can be removed in future cleanup

---

*Last updated: 2025-11-14*
