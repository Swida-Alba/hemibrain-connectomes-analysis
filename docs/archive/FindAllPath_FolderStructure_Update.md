# FindAllPath Folder Structure and Visualization Update

## Summary
Updated the `FindAllPath()` method in `coana.py` to reorganize the folder structure and create three separate Sankey diagram visualizations showing different metrics: synapse count, connection ratio, and traversal probability.

## Changes Made

### 1. Folder Structure Reorganization

**Old Structure:**
```
L3_to_l-LNv_L2w10r0p0_20251025_143022/
└── allpaths_2L_10snp/
    ├── files...
```

**New Structure:**
```
L3_to_l-LNv/
└── allpaths_L2w10r0p0_20251025_143022/
    ├── all_attributes.json
    ├── parameters.txt
    ├── L3_to_l-LNv_allpaths_info.xlsx
    ├── Sankey_type_allpaths_snp.html
    ├── Sankey_type_allpaths_ratio.html
    ├── Sankey_type_allpaths_prob.html
    └── Sankey_bodyId_allpaths.html
```

**Key Changes:**
- Base folder (`L3_to_l-LNv`) contains only source and target neuron names
- All analysis parameters moved to the `allpaths_` subfolder name
- Configuration files (`all_attributes.json`, `parameters.txt`) now saved in the allpaths folder
- Excel filename simplified from `_allpaths_info_snp10.xlsx` to `_allpaths_info.xlsx`
- Visualization HTML files updated with cleaner naming

### 2. Parameter Suffix Format

The allpaths folder includes a parameter suffix with the following format:
```
allpaths_L{max_interlayer}w{min_weight}r{min_ratio}p{min_prob}_{timestamp}
```

Example: `allpaths_L2w10r0_01p0_05_20251025_143022`

**Decimal Notation:** Uses underscore `_` instead of `p` for decimal points
- `0.01` becomes `0_01`
- `0.5` becomes `0_5`

### 3. Three Sankey Diagram Visualizations

Created three separate Sankey diagrams to visualize connection paths using different metrics:

#### Visualization 1: Synapse Count (Weight-based)
- **File:** `Sankey_type_allpaths_snp.html`
- **Title:** "Sankey diagram of all connection paths based on neuron type (by synapse count)"
- **Metric:** Direct synapse count (`weight`)
- **Use case:** Shows the raw number of synapses in each connection

#### Visualization 2: Connection Ratio
- **File:** `Sankey_type_allpaths_ratio.html`
- **Title:** "Sankey diagram of all connection paths based on neuron type (by connection ratio)"
- **Metric:** `connection_ratio * weight` (weighted average)
- **Use case:** Shows the proportion of connections relative to total possible connections

#### Visualization 3: Traversal Probability
- **File:** `Sankey_type_allpaths_prob.html`
- **Title:** "Sankey diagram of all connection paths based on neuron type (by traversal probability)"
- **Metric:** `traversal_probability * weight` (weighted average)
- **Use case:** Shows the likelihood of signal propagation through each path

### 4. Code Changes in `coana.py`

#### Lines 820-851: Modified `InitializeNeuronInfo()`
- Simplified base folder naming (removed parameters)
- Parameter dict and DataFrame still created but not saved immediately
- Will be saved in method-specific subfolders instead

#### Lines 1470-1520: Modified `FindAllPath()` initialization
- Added `format_decimal()` helper function
- Created `param_suffix` with all parameters
- Updated `allpath_folder` path with parameter suffix
- Saves `all_attributes.json` and `parameters.txt` to allpaths folder

#### Lines 1942-1987: Enhanced Sankey parameter collection
- Added `ratio_type = []` list to collect connection ratio data
- Added `prob_type = []` list to collect traversal probability data
- Modified the data collection loop to calculate:
  - `ratio_type`: Uses `connection_ratio * weight` if available
  - `prob_type`: Uses `traversal_probability * weight` if available
- Both fall back to `weight` if columns don't exist

#### Lines 2000-2065: Created three Sankey visualizations
- **Weight-based Sankey** (existing, renamed file)
- **Ratio-based Sankey** (NEW)
- **Probability-based Sankey** (NEW)
- All share the same node layout, only link values differ

#### Line 2067: Updated bodyId Sankey filename
- Changed from: `Sankey_bodyId_allpaths_snp{min_synapse_num}.html`
- To: `Sankey_bodyId_allpaths.html`
- Removed redundant synapse number suffix (already in folder name)

### 5. Implementation Details

**Connection Ratio Calculation:**
```python
if 'connection_ratio' in conn_type.columns:
    ratio_type.append(conn_type.at[j,'connection_ratio'] * conn_type.at[j,'weight'])
else:
    ratio_type.append(conn_type.at[j,'weight'])
```

**Traversal Probability Calculation:**
```python
if 'traversal_probability' in conn_type.columns:
    prob_type.append(conn_type.at[j,'traversal_probability'] * conn_type.at[j,'weight'])
else:
    prob_type.append(conn_type.at[j,'weight'])
```

Both calculations use weighted values to properly represent the metric across connections with different synapse counts.

### 6. Benefits of New Structure

1. **Cleaner Organization:** Parameters at analysis level, not neuron pair level
2. **Better Scalability:** Multiple analyses on same neuron pair create separate subfolders
3. **Comprehensive Visualization:** Three different views of the same data
4. **Consistent Naming:** All files follow the same naming convention
5. **Self-Documenting:** Folder and file names clearly indicate their content

### 7. Testing Recommendations

To test these changes:

1. Run `FindAllPath()` with your standard parameters
2. Verify folder structure matches the new pattern
3. Check that all three Sankey HTML files are created
4. Open each visualization and confirm:
   - Weight-based shows raw synapse counts
   - Ratio-based shows proportional connections
   - Probability-based shows traversal likelihood
5. Verify that `all_attributes.json` and `parameters.txt` are in the allpaths folder

### 8. Related Files

This update complements previous fixes and features:
- `FilterBy_DecimalNotation_Fix.md`: Decimal notation using `_` instead of `p`
- `BUGFIX_TypeColumn_Merge.md`: Fixed AttributeError in type_pre column
- `FilterBy_Implementation.md`: Added `filter_by` parameter for bodyId vs type filtering

## Date
January 2025

## Status
✅ Complete - All changes implemented and tested for syntax errors
