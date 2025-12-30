# LabelMapper Guide

The `LabelMapper` class provides a centralized mechanism for standardizing neuron identifiers across different datasets. This is crucial for cross-dataset comparison analysis where naming conventions may vary (e.g., `aMe12` in hemibrain vs `aMe12-like` in male-cns).

## Key Features

*   **Centralized Mapping**: Manages mapping logic across all datasets in one place.
*   **Flexible Input Formats**: Supports both CSV and JSON formats.
*   **Asymmetric Mapping**: Handles cases where one dataset has more neurons for a given type than another.
*   **Group-Based Mapping**: Maps multiple neurons from different datasets to a single "standard label" (group).
*   **Auto-Generation**: Can automatically generate standard labels if not provided.

## Input Formats and Example Files

The toolkit provides several example files in `examples/data/` to demonstrate how to structure your mapping data.

### 1. CSV Format
The CSV format is the most straightforward way to define mappings. It requires a `custom_label` column and one column per dataset.

**Example (`examples/data/example_source_mapping.csv`):**
```csv
custom_label,hemibrain:v1.2.1,male-cns:v0.9,flywire_FAFB_v783
aMe12,aMe12,aMe12,720575940610453042
MBON14,MBON14.*_R,MBON14.*_R,MBON14
MBON06,MBON06.*_R,MBON06.*_R,MBON06
PPL101,PPL101,PPL101,720575940621886666
```
*   **custom_label**: The standardized name used in your analysis.
*   **Dataset Columns**: The specific neuron IDs or regex patterns for each dataset.
*   **Regex Support**: You can use regex patterns (e.g., `MBON14.*_R`) to match multiple neurons.

### 2. JSON Format
The JSON format allows for defining source, target, and intermediate mappings in a single file and supports nested structures.

**Example (`examples/data/example_label_mapping.json`):**
```json
{
  "source_mapping": {
    "custom_label": ["aMe12"],
    "hemibrain:v1.2.1": [["aMe12"]],
    "male-cns:v0.9": [["aMe12"]],
    "flywire_FAFB_v783": [["720575940610453042"]]
  },
  "target_mapping": {
    "custom_label": ["KCg-s1"],
    "hemibrain:v1.2.1": [["KCg-s1"]],
    "male-cns:v0.9": [["KCg-s1"]],
    "flywire_FAFB_v783": [["KCg-s1"]]
  },
  "intermediate_mapping": {
    "custom_label": ["LPLC2"],
    "hemibrain:v1.2.1": [["LPLC2"]],
    "male-cns:v0.9": [["LPLC2"]],
    "flywire_FAFB_v783": [["LPLC2"]]
  }
}
```

## Usage Examples

### Initialization from Files

```python
from comparison.label_mapper import LabelMapper

# Using a single unified JSON file (Recommended)
# Note: ONLY JSON format is supported for overall_mapping_json
mapper = LabelMapper(
    overall_mapping_json='examples/data/example_label_mapping.json'
)

# Using separate CSV files for source and target
mapper_csv = LabelMapper(
    source_mapping_file='examples/data/example_source_mapping.csv',
    target_mapping_file='examples/data/example_target_mapping.csv'
)
```

### Initialization from Dictionaries

```python
from comparison.label_mapper import LabelMapper

mapper = LabelMapper(
    source_mapping_dict={
        'hemibrain:v1.2.1': [['aMe12', 'aMe12_R']],
        'male-cns:v0.9': [['aMe12', 'aMe12-like']]
    },
    source_labels=['aMe12_grp']
)
```

### Retrieving Standard Labels

```python
# Get the standard label for a neuron in a specific dataset
std_label = mapper.get_std_label(
    dataset='hemibrain:v1.2.1', 
    neuron_id='aMe12', 
    group_type='source'
)
print(std_label) # Output: aMe12_grp1
```

### Retrieving Neurons by Standard Label

```python
# Get all neurons associated with a standard label in a specific dataset
neurons = mapper.get_neurons_by_label(
    std_label='aMe12_grp1', 
    dataset='male-cns:v0.9', 
    group_type='source'
)
print(neurons) # Output: ['aMe12', 'aMe12-like', 'aMe12_variant']
```

## Integration with ComparisonParameters

`LabelMapper` is the core component for defining neuron identity in `ComparisonParameters`. It allows you to abstract away dataset-specific naming differences.

### Using Example Files in Comparison

Here is how you would set up a comparison using the provided example files:

```python
from comparison.comparison_parameters import ComparisonParameters
from comparison.label_mapper import LabelMapper

# 1. Initialize the mapper with your mapping files
mapper = LabelMapper(
    source_mapping_file='examples/data/example_source_mapping.csv',
    target_mapping_file='examples/data/example_target_mapping.csv'
)

# 2. Create ComparisonParameters
# The mapper handles the translation of 'aMe12' and 'KCg-s1' to dataset-specific IDs
params = ComparisonParameters(
    datasets=['hemibrain:v1.2.1', 'male-cns:v0.9', 'flywire_FAFB_v783'],
    source_neurons=mapper,  # Pass the mapper object here
    target_neurons=mapper,  # Pass the mapper object here (or a list if no mapping needed)
    
    # Optional: Manually specify which standard labels to use from the mapper
    # If omitted, it might use all available in the mapper or require manual specification depending on implementation
    # source_labels=['aMe12'], 
    # target_labels=['KCg-s1'] 
)

# 3. Run the comparison
# The analyzer will use the mapper to look up 'aMe12' in hemibrain, male-cns, etc.
```

### Why use LabelMapper in ComparisonParameters?

*   **Consistency**: Ensures you are comparing "apples to apples" across datasets.
*   **Simplicity**: Your analysis code only deals with standard labels (e.g., `aMe12`), not dataset-specific IDs (e.g., `720575940610453042`).
*   **Scalability**: Easily add new datasets by just updating the CSV mapping file, without changing your analysis code.

