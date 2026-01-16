# Auto Type Mapping for Cross-Dataset Comparison

## Overview

When comparing neuron connectivity profiles across different *Drosophila* connectome datasets (e.g., hemibrain vs male-cns, FAFB vs BANC), a fundamental challenge arises: **the same biological neuron types may have different names in different datasets**. 

For example:
- `lLN7` (hemibrain) = `ALIN4` (male-cns, flywire)
- `DNp01` (male-cns) = `DNp01` (flywire) = `DNp01` (hemibrain) ← same name, no issue
- `MTe07` (flywire) = `MeVPLo2` (male-cns)

The **Auto Type Mapping** feature automatically standardizes neuron type names across datasets, enabling meaningful cross-dataset comparisons of connectivity profiles.

## How It Works

### 1. Canonical Type Names

We use **male-cns** as the canonical reference dataset because:
- It represents a complete CNS reconstruction (unlike hemibrain's partial brain)
- It includes the most up-to-date neuron type annotations
- It has extensive type metadata including mappings to other datasets

The `CrossDatasetTypeMapper` class builds mappings from the male-cns `neuron_df`, which contains columns linking each neuron's type across datasets.

### 2. Type Mapping Source

The mapper uses the male-cns neuron DataFrame located at:
`datasets/male-cns_v0_9/male-cns_v0_9_allneurons_neuron_df.csv`

Key columns used:
- `type`: male-cns type name
- `flywireType`: corresponding type in flywire (FAFB/BANC)
- `hemibrainType`: corresponding type in hemibrain
- `mancType`: corresponding type in MANC

### 3. Standardization Process

When comparing profiles from different datasets:

1. **Query type standardization**: Input type names are mapped to canonical names for profile retrieval
2. **Partner type standardization**: When computing similarity, the partner types in each profile are standardized to canonical names
3. **Similarity computation**: Jaccard, cosine, and rank correlation are computed on standardized partner type sets

## Usage

### Enabling Auto Type Mapping in ComparisonAnalyzer

```python
from comparison import ComparisonAnalyzer, ComparisonParameters

params = ComparisonParameters(
    datasets=['male-cns:v0.9', 'flywire_FAFB_v783'],
    source_neurons=['MeVPLo2', 'MeVPaMe1'],
    target_neurons=['aMe.*'],
    auto_type_mapping=True,  # Enable type mapping (default: True)
    # ... other parameters
)

analyzer = ComparisonAnalyzer(params)
results = analyzer.run_comparison()
analyzer.export_results()  # Exports filtered auto_type_mapping.csv
```

### In ConnectivityProfileComparer (Cross-Dataset Mode)

```python
from comparison.profile_comparator import ConnectivityProfileComparer

# Cross-dataset comparison with dict query format
comparer = ConnectivityProfileComparer(
    query={
        'hemibrain:v1.2.1': ['MBON-γ2α\'1', 'MBON-γ5β\'2a', 'PPL1-γ1pedc'],
        'male-cns:v0.9': ['MBON-γ2α\'1', 'MBON-γ5β\'2a', 'PPL1-γ1pedc']
    },
    dataset=None,  # Must be None for cross-dataset (warning if not)
    use_auto_type_mapping=True  # Enable partner type standardization
)

# Generates N×M similarity matrix comparing types across datasets
results = comparer.run()
```

## API Reference

### CrossDatasetTypeMapper

```python
from comparison.cross_dataset_type_mapper import CrossDatasetTypeMapper

# Create mapper instance
mapper = CrossDatasetTypeMapper(
    workspace_path='/path/to/workspace',  # Optional
    neuron_df_path=None,  # Optional, uses default location if None
    verbose=True
)

# Load mappings (called automatically when needed)
mapper.load()

# Get equivalent type in target dataset
mapped = mapper.get_mapped_type(
    type_name='lLN7',
    source_dataset='hemibrain:v1.2.1', 
    target_dataset='male-cns:v0.9'
)
# Returns: 'ALIN4'

# Resolve type across multiple datasets
mappings = mapper.resolve_type_across_datasets(
    type_name='MeVPLo2',
    datasets=['male-cns:v0.9', 'flywire_FAFB_v783', 'hemibrain:v1.2.1']
)
# Returns: {'male-cns:v0.9': 'MeVPLo2', 'flywire_FAFB_v783': 'MTe07', ...}

# Get canonical (male-cns) type name
canonical = mapper.get_canonical_type('MTe07', source_dataset='flywire_FAFB_v783')
# Returns: 'MeVPLo2'

# Standardize partner types for cross-dataset comparison
standardized = mapper.standardize_partner_types(
    partner_types={'lLN7': 10.5, 'KC-γm': 5.2},
    source_dataset='hemibrain:v1.2.1'
)
# Returns: {'ALIN4': 10.5, 'KC-γm': 5.2} (lLN7 mapped to male-cns name)

# Get display name with cross-dataset mappings
display = mapper.get_display_name(
    type_name='MeVPLo2',
    datasets=['male-cns:v0.9', 'flywire_FAFB_v783'],
    source_dataset='male-cns:v0.9'
)
# Returns: 'MeVPLo2(MTe07)' if names differ across datasets
```

### Export Methods

```python
# Export mappings (filtered to result types)
mapper.export_mapping(
    output_path='auto_type_mapping.csv',
    filter_types={'MeVPLo2', 'ALIN4', 'DNp01'}  # Only include these types
)

# Export mapping conflicts (1-to-N or N-to-1 relationships)
mapper.export_conflicts(
    output_path='auto_type_mapping_conflicts.csv',
    filter_types={'WED092'}  # Optional filtering
)
```

### Disabling Auto Type Mapping

If you want to compare connectivity profiles without type name standardization (e.g., to see raw differences in naming conventions):

```python
# In ComparisonParameters
params = ComparisonParameters(
    datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
    auto_type_mapping=False,  # Disable standardization
    # ...
)

# In ConnectivityProfileComparer
comparer = ConnectivityProfileComparer(
    query={'hemibrain:v1.2.1': [...], 'male-cns:v0.9': [...]},
    use_auto_type_mapping=False  # Disable standardization
)
```

## Implementation Details

### Partner Type Standardization

When computing connectivity profile similarity across datasets, the key insight is that **partner types** must also be standardized. For example:

**Profile A (hemibrain: ALIN4 → lLN7)**:
- Upstream: `PPL1-γ1pedc` (20%), `lLN7` (15%), `KC-γm` (10%), ...
- Downstream: `FB2B` (25%), ...

**Profile B (male-cns: ALIN4)**:
- Upstream: `PPL1-γ1pedc` (18%), `ALIN4` (16%), `KC-γm` (12%), ...
- Downstream: `FB2B` (22%), ...

Without standardization, `lLN7` and `ALIN4` would be treated as different types, reducing similarity. With standardization to male-cns names:

**Profile A (standardized)**:
- Upstream: `PPL1-γ1pedc` (20%), `ALIN4` (15%), `KC-γm` (10%), ...

**Profile B (standardized)**:
- Upstream: `PPL1-γ1pedc` (18%), `ALIN4` (16%), `KC-γm` (12%), ...

Now the Jaccard and cosine similarities correctly identify these as highly similar profiles.

### Supported Datasets

Type mappings are available for:
- `male-cns:v0.9` (canonical reference)
- `flywire_FAFB_v783`
- `flywire_BANC_v626`
- `hemibrain:v1.2.1`
- `manc:v1.0` / `manc:v1.2.1`

### Output Files

When running `ComparisonAnalyzer.export_results()` with `auto_type_mapping=True`, two files are generated:

1. **auto_type_mapping.csv**: Type mappings for neurons in results only
   ```csv
   male-cns:v0.9,flywire_FAFB_v783,flywire_BANC_v626,hemibrain:v1.2.1,manc:v1.0,manc:v1.2.1
   ALIN4,ALIN4,ALIN4,lLN7,,
   DNp01,DNp01,DNp01,DNp01,,
   MeVPLo2,MTe07,MTe07,,,
   ...
   ```

2. **auto_type_mapping_conflicts.csv**: 1-to-N or N-to-1 mapping conflicts
   ```csv
   source_dataset,source_type,target_dataset,target_types,relationship
   male-cns:v0.9,WED092,flywire_FAFB_v783,"WED092b, WED092c, WED092d",1-to-N
   ```

### Conflict Handling

- **1-to-N mappings**: One male-cns type maps to multiple types in another dataset. These are logged as warnings but the mapping proceeds using all variations.
- **N-to-1 mappings**: Multiple types from different datasets map to the same canonical name. These types should NOT be aggregated incorrectly.

Use `mapper.is_n_to_1_type(type_name, dataset)` to check if a type is involved in a conflict.

## Performance Considerations

- Type mapping lookups are O(1) hash table operations
- Mapping tables are loaded lazily on first use
- For large-scale batch comparisons, the mapper is passed once to avoid repeated loading
- Partner standardization adds minimal overhead (dict comprehension)

## Troubleshooting

### "Type mapper not loaded" warning

This means the neuron_df file wasn't found. Check:
1. File exists at `datasets/male-cns_v0_9/male-cns_v0_9_allneurons_neuron_df.csv`
2. The workspace path is correctly configured

### Type not found in mapping

If a type has no cross-dataset mapping:
- The original type name is preserved (identity mapping)
- This is expected for types unique to one dataset

### Unexpected similarity scores

If cross-dataset similarity seems too low:
1. Check if key partner types have mappings
2. Consider that some connectivity differences are real biological variation
3. Use `use_auto_type_mapping=False` to see raw (unmapped) comparison

## See Also

- [Cross-Dataset Comparison](./CROSS_DATASET_COMPARISON.md) - Overview of cross-dataset analysis
- [Connectivity Profiling](./CONNECTIVITY_PROFILING.md) - Connectivity profile computation
