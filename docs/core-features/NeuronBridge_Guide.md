# NeuronBridge Integration Guide

The NeuronBridge module provides programmatic access to the [NeuronBridge](https://neuronbridge.janelia.org/) database, enabling bidirectional mapping between electron microscopy (EM) reconstructions and light microscopy (LM) driver lines.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [NeuronBridgeFinder Class](#neuronbridgefinder-class)
  - [Initialization Parameters](#initialization-parameters)
  - [Line Type Separation](#line-type-separation)
  - [Line Specificity Metrics](#line-specificity-metrics)
  - [Mutual Information](#mutual-information)
  - [Core Methods](#core-methods)
  - [Batch Processing Methods](#batch-processing-methods)
  - [Image Download Methods](#image-download-methods)
- [Match Types](#match-types)
- [Dataset Support](#dataset-support)
- [Examples](#examples)
- [Integration with FlyLight Downloader](#integration-with-flylight-downloader)
- [Troubleshooting](#troubleshooting)

---

## Overview

NeuronBridge is a web application that uses Color Depth MIP (CDM) search to find morphological matches between:
- **EM reconstructions** from hemibrain, male-cns, FlyWire FAFB, and other datasets
- **LM driver lines** (GAL4, LexA, Split-GAL4) from the FlyLight project

This module wraps the NeuronBridge API to provide:
- **EM → LM mapping**: Find driver lines matching a given EM body ID
- **LM → EM mapping**: Find EM neurons matching a driver line name
- **Batch processing**: Process multiple queries with automatic result aggregation
- **Image downloads**: Download CDM images from NeuronBridge or full imagery from FlyLight

---

## Installation

The NeuronBridge module requires the `neuronbridge-python` package:

```bash
pip install neuronbridge-python
```

For FlyLight image downloads, optionally install boto3 for faster S3 access:

```bash
pip install boto3
```

---

## Quick Start

### Basic Usage

```python
from src.neuronbridge_finder import NeuronBridgeFinder

# Initialize the finder
nbf = NeuronBridgeFinder(verbose=True)

# Find driver lines matching an EM body ID
lines_df = nbf.id_to_lines(720575940621039145)
print(lines_df[['line', 'score', 'library']].head())

# Find EM neurons matching a driver line
neurons_df = nbf.line_to_neuron('LH173')
print(neurons_df[['bodyId', 'dataset', 'score']].head())
```

### Batch Processing with Image Download

```python
from src.neuronbridge_finder import NeuronBridgeFinder

nbf = NeuronBridgeFinder()

# Find lines for multiple neurons and download images
results = nbf.find_lines_batch(
    queries='MBON01,aMe12',
    dataset='hemibrain:v1.2.1',
    match_type='both',
    output_dir='./neuronbridge_results',
    download_images='flylight',
    download_top_n_img=10,
    simple_mode=True
)
```

---

## NeuronBridgeFinder Class

### Initialization Parameters

```python
@dataclass
class NeuronBridgeFinder:
    datasets_path: Optional[str] = None
    use_cache: bool = True
    cache_folder: Optional[str] = None
    verbose: bool = True
    separate_splitgal4: bool = False
    neuprint_token: Optional[str] = None
    neuprint_server: str = 'https://neuprint.janelia.org'
    match_type: str = 'cds'
    region: str = 'All'
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `datasets_path` | `str` or `None` | Auto-detect | Path to the datasets folder containing neuron_df CSV files. Used for enriching results with local neuron metadata. |
| `use_cache` | `bool` | `True` | Whether to cache API results locally. Cached results are stored as CSV files and reused on subsequent calls. |
| `cache_folder` | `str` or `None` | Auto-detect | Folder for cached results. Default: `cache/neuronbridge/` in the project directory. |
| `verbose` | `bool` | `True` | Print progress messages during operations. |
| `separate_splitgal4` | `bool` | `False` | **NEW**: If True, separate results into GAL4/LexA and Split-GAL4 categories. When enabled, `download_top_n_img` applies separately to each category (see [Line Type Separation](#line-type-separation)). |
| `neuprint_token` | `str` or `None` | `None` | **NEW**: NeuPrint API token for pulling missing datasets. If not set, checks `NEUPRINT_TOKEN` or `NEUPRINT_APPLICATION_CREDENTIALS` environment variables. Get your token at: https://neuprint.janelia.org/account |
| `neuprint_server` | `str` | `'https://neuprint.janelia.org'` | **NEW**: NeuPrint server URL. |
| `match_type` | `str` | `'cds'` | **NEW**: Default match algorithm for all operations: `'cds'` (Color Depth MIP Search), `'pppm'` (Pattern Matching), or `'both'`. Can be overridden at method level. |
| `region` | `str` | `'All'` | **NEW**: Filter images by anatomical region: `'Brain'`, `'VNC'`, or `'All'`. Filters out images from non-matching regions to reduce processing time and improve specificity. |

**Setting up NeuPrint Token** (required for pulling missing datasets):

```python
# Option 1: Pass token directly
nbf = NeuronBridgeFinder(neuprint_token='your_token_here')

# Option 2: Set environment variable (recommended)
# In terminal: export NEUPRINT_TOKEN="your_token_here"
nbf = NeuronBridgeFinder()
```

**Region Filtering** (filter images by anatomical area):

```python
# Filter for Brain region only (exclude VNC images)
nbf = NeuronBridgeFinder(region='Brain')

# Filter for VNC region only (exclude Brain images)
nbf = NeuronBridgeFinder(region='VNC')

# Process all regions (default)
nbf = NeuronBridgeFinder(region='All')

# Combine with match_type for complete configuration
nbf = NeuronBridgeFinder(region='Brain', match_type='both')
```

**Match Type Configuration** (set default algorithm):

```python
# Use CDS matching by default (faster, color-based)
nbf = NeuronBridgeFinder(match_type='cds')  # Default

# Use PPPM matching by default (pattern-based)
nbf = NeuronBridgeFinder(match_type='pppm')

# Use both algorithms by default (combines results with rank-based scoring)
nbf = NeuronBridgeFinder(match_type='both')

# Override at method level if needed
results = nbf.find_neurons_batch('VT037867', match_type='cds')  # Override
```

### Line Type Separation

When `separate_splitgal4=True`, driver lines are classified into two categories based on their name prefixes:

**GAL4/LexA lines** (traditional enhancer trap lines):
- VT lines (Vienna Tile): `VT037867`, `VT001234`
- R lines (Rubin lab): `R10A06`, `R65E11`
- GMR lines: `GMR_01A01`

**Split-GAL4 lines** (intersectional genetic approach):
- SS lines (Split Screen): `SS01015`, `SS00734`
- LH lines (Lateral Horn): `LH173`, `LH1234`
- MB lines (Mushroom Body): `MB011B`, `MB543`
- IS lines: `IS49879`, `IS83928`
- OL lines (Optic Lobe): `OL0042B`
- LC lines (Lobula Columnar): `LC16`
- And others: `LLPC`, `LPC`, `JRC_SS`, `BJD_SS`

**Usage Example**:
```python
# Enable separate mode on instance
nbf = NeuronBridgeFinder(separate_splitgal4=True)

# With download_top_n_img=5, downloads:
# - Top 5 GAL4/LexA lines
# - Top 5 Split-GAL4 lines
# = 10 lines total
results = nbf.find_lines_batch(
    queries='MBON01',
    download_images='flylight',
    download_top_n_img=5,
    output_dir='./output'
)

# Results include 'line_type' column
print(results[results['line_type'] == 'gal4_lexa']['line'].unique()[:5])
print(results[results['line_type'] == 'split_gal4']['line'].unique()[:5])
```

**Separate Output Files** (when `separate_splitgal4=True`):
```
output/findlines_20241223_123456/
├── all_lines.csv              # Combined results (all lines)
├── line_summary.csv           # Aggregated stats (all lines)
├── gal4_lexa_lines.csv        # GAL4/LexA results only
├── gal4_lexa_summary.csv      # GAL4/LexA summary
├── split_gal4_lines.csv       # Split-GAL4 results only
├── split_gal4_summary.csv     # Split-GAL4 summary
└── images/
```

---

### Line Specificity Metrics

**NEW**: Calculate how specific each driver line is to your queried neuron types.

```python
results = nbf.find_lines_batch(
    queries='MBON01,aMe12,EPG',     # Type queries (not bodyIds)
    dataset='hemibrain:v1.2.1',
    calculate_specificity=True,     # Enable specificity calculation
    specificity_top_n=100,          # Analyze top 100 matches per line
    output_dir='./output'
)
```

**Specificity Columns** (added to `line_summary.csv`):

| Column | Description | Interpretation |
|--------|-------------|----------------|
| `rank_sum` | Sum of ranks for queried types in top-N results | Lower = better (queried types appear higher in rankings) |
| `type_proportion` | Queried types / Total types labeled | Higher = better (line labels fewer other types) |
| `n_queried_types` | Number of queried types found in top-N | Higher = better coverage |
| `n_total_types` | Total distinct types labeled by this line | Lower = more selective |
| `selectivity` | 1 / n_total_types | Higher = more selective |
| `expression_entropy` | Shannon entropy of type distribution (bits) | Lower = more specific (labels fewer types more strongly) |
| `normalized_entropy` | Entropy / max_entropy (0-1 scale) | Lower = more specific |
| `weighted_type_proportion` | Type proportion weighted by match scores | Higher = queried types have stronger matches |
| `mean_queried_score` | Mean NeuronBridge score for queried types | Higher = stronger expression in queried types |
| `colabel_sparsity` | 1 - (proportion of co-labeling lines) | Higher = more unique labeling pattern |
| `n_colabeling_lines` | Number of lines with >10% neuron overlap | Lower = more unique |
| `mean_colabel_similarity` | Average Jaccard similarity with other lines | Lower = more unique |
| `specificity_score` | Composite score (0-1) | Higher = more specific to queried types |

**Expression Entropy** ($H = -\sum p_i \log_2(p_i)$):
- Measures diversity of neuron types labeled by the line
- A line labeling 10 types equally has entropy ≈ 3.32 bits
- A line labeling mostly one type has entropy ≈ 0 bits
- Use `normalized_entropy` (0-1) for easier comparison

**Expression Strength Weighting**:
- Weights each neuron's contribution by its NeuronBridge match score
- High-confidence matches count more than low-confidence ones
- `weighted_type_proportion = Σ(scores for queried types) / Σ(all scores)`

**Co-labeling Matrix & Sparsity**:
- Builds a Jaccard similarity matrix showing how often pairs of lines label the same neurons
- `colabel_sparsity` = 1 - (fraction of lines with significant overlap)
- High sparsity = line labels unique set of neurons not covered by other lines
- Saves `colabeling_matrix.csv` and interactive heatmap visualization

**Notes**:
- Requires type/instance queries (not bodyIds) to calculate meaningful specificity
- Uses `line_to_neuron()` to query NeuronBridge for each line's labeled neurons
- Automatically falls back to NeuPrint data if local dataset files are missing

**Interpreting Specificity**:
- A line with high `type_proportion` (e.g., 0.8) and low `normalized_entropy` (e.g., 0.3) is strongly specific to your queried types
- A line with low `rank_sum` means your queried types consistently appear in top matches
- A line with high `colabel_sparsity` labels unique neurons not covered by other lines
- Use `specificity_score` as a combined metric for prioritizing lines

**Output Files** (when `calculate_specificity=True`):
```
output/findlines_20241223_123456/
├── line_summary.csv           # All specificity metrics (incl. entropy, MI)
├── colabeling_matrix.csv      # Jaccard similarity matrix
├── colabeling_matrix.html     # Interactive heatmap visualization
├── mutual_information.csv     # MI values per line
├── expression_matrix.csv      # Binary lines × types matrix
└── expression_matrix.html     # Lines × types heatmap
```

---

### Mutual Information

**Mutual Information** quantifies how much knowing a line's expression pattern tells you about neuron type identity:

$$I(L; T) = H(T) - H(T|L) = \sum_{l,t} p(l,t) \log_2 \frac{p(l,t)}{p(l)p(t)}$$

Where:
- $H(T)$ = entropy of type distribution (uncertainty about neuron type)
- $H(T|L)$ = conditional entropy (remaining uncertainty after knowing line expression)
- $p(l,t)$ = joint probability of line $l$ labeling type $t$
- $p(l)$, $p(t)$ = marginal probabilities

**New Columns** (added to `line_summary.csv`):

| Column | Description | Interpretation |
|--------|-------------|----------------|
| `mutual_information` | MI in bits | Higher = more informative |
| `normalized_mi` | MI / H(T) (0-1 scale) | Higher = more informative |
| `queried_type_coverage` | Fraction of queried types labeled | Higher = better coverage |

**Interpretation**:
- High MI = Line expression is highly informative about neuron type
- MI = 0 means line expression is independent of neuron type (random labeling)
- A line with MI = 2 bits reduces type uncertainty by a factor of 4
- `normalized_mi` of 0.5 means knowing this line halves your uncertainty about type

**Why Useful**:
- Unlike entropy (which only considers one line), MI considers the relationship between line expression and type identity
- Captures both specificity (labeling few types) and selectivity (types not labeled by other lines)
- Use to find the most informative genetic tools for your target cell types

**Output Files** (when `calculate_specificity=True`):
```
output/findlines_20241223_123456/
├── mutual_information.csv    # MI values per line
├── expression_matrix.csv     # Binary lines × types matrix
└── expression_matrix.html    # Interactive heatmap visualization
```

---

### Core Methods

#### `id_to_lines(body_id, match_type='cds', expected_dataset=None)`

Find driver lines matching a given EM body ID.

```python
lines_df = nbf.id_to_lines(
    body_id=720575940621039145,
    match_type='cds',
    expected_dataset='hemibrain:v1.2.1'
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `body_id` | `int` | Required | The EM body ID to search for. |
| `match_type` | `str` | `'cds'` | Match algorithm: `'cds'` (Color Depth Search), `'pppm'` (PatchPerPixMatch), or `'both'`. |
| `expected_dataset` | `str` or `None` | `None` | Expected dataset name (e.g., `'male-cns:v0.9'`). Filters results to match this dataset when body ID exists in multiple datasets. |

**Returns**: `pd.DataFrame` with columns:
- `line`: Driver line name (e.g., 'LH173', 'SS01015')
- `library`: Source library (e.g., 'FlyLight Gen1 MCFO')
- `score`: Match score (higher is better)
- `image_id`: NeuronBridge image identifier
- `match_type`: Algorithm used ('cds' or 'pppm')
- For `match_type='both'`: Additional columns `cds_score`, `pppm_score`, `cds_rank`, `pppm_rank`, `combined_rank`

---

#### `line_to_neuron(line_name, match_type='cds')`

Find EM neurons matching a driver line name.

```python
neurons_df = nbf.line_to_neuron(
    line_name='LH173',
    match_type='cds'
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `line_name` | `str` | Required | The driver line name to search for (e.g., 'LH173', 'VT037867'). |
| `match_type` | `str` | `'cds'` | Match algorithm: `'cds'`, `'pppm'`, or `'both'`. |
| `top_n` | `int` | `-1` | Maximum matches to return. `-1` for all. |

**Returns**: `pd.DataFrame` with columns:
- `bodyId`: EM body ID
- `dataset`: Dataset name (e.g., 'hemibrain:v1.2.1')
- `instance`: Neuron instance name
- `type`: Neuron type
- `status`: Reconstruction status
- `score`: Match score
- `image_id`: NeuronBridge image identifier
- `lm_sample`: LM sample identifier
- `match_type`: Algorithm used
- `library`: Source library

---

#### `neuron_to_lines(query, dataset=None, match_type='cds')`

Find driver lines matching neurons specified by bodyId, instance, or type.

```python
# Search by neuron type
results = nbf.neuron_to_lines(
    query='MBON01',
    dataset='hemibrain:v1.2.1',
    match_type='both'
)

# Search by body ID
results = nbf.neuron_to_lines(query=720575940621039145)

# Search multiple datasets
results = nbf.neuron_to_lines(
    query='aMe12',
    dataset=['hemibrain:v1.2.1', 'flywire_FAFB_v783']
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str`, `int`, or `list` | Required | Query to search: body ID (int), neuron type/instance (str), or list of queries. Supports regex patterns. |
| `dataset` | `str`, `list`, or `None` | `None` | Dataset(s) to search. Can be a single string, list of datasets, or `None` to search all available datasets. |
| `match_type` | `str` | `'cds'` | Match algorithm: `'cds'`, `'pppm'`, or `'both'`. |

**Returns**: `Dict[str, pd.DataFrame]` mapping body IDs to DataFrames of matching lines.

---

### Batch Processing Methods

#### `find_lines_batch(...)`

Find driver lines for multiple EM neurons with automatic saving and optional image download.

```python
results = nbf.find_lines_batch(
    queries='MBON01,aMe12,720575940621039145',
    dataset='hemibrain:v1.2.1',
    match_type='both',
    output_dir='./output',
    download_images='flylight',
    download_top_n_img=20,
    image_formats=['png', 'jpg'],
    image_types='mip',
    max_images_per_line=10,
    flylight_category=['GAL4/LEXA', 'SplitGAL4'],
    organize_by_region=False,
    simple_mode=True
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `queries` | `str`, `int`, or `list` | Required | Neuron query(s). Can be comma-separated string, body ID, or list. |
| `dataset` | `str`, `list`, or `None` | `None` | Dataset(s) to search. Use list for multiple datasets. |
| `match_type` | `str` or `None` | `None` | Match algorithm: `'cds'`, `'pppm'`, or `'both'`. If `None`, uses instance-level `match_type`. |
| `output_dir` | `str` or `None` | `None` | Directory to save results. Creates timestamped subfolder. |
| `download_images` | `str` or `None` | `None` | Image source: `'neuronbridge'`, `'flylight'`, `'both'`, or `None`. |
| `download_top_n_img` | `int` or `None` | `10` | Download images for top N lines only (by aggregate score/rank). |
| `image_formats` | `str` or `list` | `['png', 'jpg']` | File formats to download. |
| `image_types` | `str` or `list` | `'mip'` | Image types: `'mip'`, `'cdm'`, `'aligned'`, etc. |
| `max_images_per_line` | `int` or `None` | `20` | Maximum images per line. |
| `flylight_category` | `str` or `list` | `['GAL4/LEXA', 'SplitGAL4']` | FlyLight collection category. |
| `organize_by_region` | `bool` | `False` | Organize images into Brain/VNC subfolders. |
| `simple_mode` | `bool` | `False` | Apply filename filtering to reduce download volume (see [Simple Mode](#simple-mode)). |
| `calculate_specificity` | `bool` | `False` | **NEW**: Calculate line specificity metrics (see [Line Specificity Metrics](#line-specificity-metrics)). |
| `specificity_top_n` | `int` | `100` | Number of top matches per line to analyze for specificity. |

**Returns**: `pd.DataFrame` with combined results including:
- All columns from `id_to_lines()`
- `source_query`: Original query string
- `source_bodyId`: Matching body ID
- `source_dataset`: Source dataset
- `matched_bodyIds`: Comma-separated list of all body IDs matching each line
- When `calculate_specificity=True`: Additional specificity columns in summary

**Output Files** (when `output_dir` is specified):
```
output/findlines_20241223_123456/
├── all_lines.csv           # Combined results
├── line_summary.csv        # Aggregated statistics per line (+ specificity if enabled)
├── {query}_lines.csv       # Individual query results
├── gal4_lexa_*.csv         # Separate files if separate_splitgal4=True
├── split_gal4_*.csv        # Separate files if separate_splitgal4=True
└── images/                 # Downloaded images (if requested)
    └── {line_name}/
        └── *.png
```

---

#### `find_neurons_batch(line_names, top_n=-1, match_type=None, output_dir=None)`

Find EM neurons for multiple driver lines.

```python
results = nbf.find_neurons_batch(
    line_names='LH173,VT037867,SS01015',
    top_n=50,
    match_type='both',
    output_dir='./output'
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `line_names` | `str` or `list` | Required | Driver line name(s). Can be comma-separated. |
| `top_n` | `int` | `-1` | Maximum matches per line. `-1` for all. |
| `match_type` | `str` or `None` | `None` | Match algorithm. If `None`, uses instance-level `match_type`. |
| `output_dir` | `str` or `None` | `None` | Directory to save results. |

---

### Image Download Methods

#### `download_line_images(line_names, output_dir, source='neuronbridge', ...)`

Download images for driver lines from NeuronBridge or FlyLight.

```python
files = nbf.download_line_images(
    line_names='LH173,SS01015',
    output_dir='./images',
    source='flylight',
    formats=['png', 'jpg'],
    image_types='mip',
    max_files=10
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `line_names` | `str` or `list` | Required | Driver line name(s). |
| `output_dir` | `str` | Required | Directory to save images. |
| `source` | `str` | `'neuronbridge'` | Image source: `'neuronbridge'` or `'flylight'`. |
| `formats` | `str` or `list` | `'png'` | File formats. |
| `image_types` | `str` or `list` | `'cdm'` | Image types. |
| `max_files` | `int` or `None` | `None` | Maximum files per line. |

---

## Match Types

### CDS (Color Depth Search)
- **Algorithm**: Compares color depth MIP images pixel-by-pixel
- **Speed**: Fast (~seconds per query)
- **Best for**: Quick initial searches, finding similar morphologies

### PPPM (PatchPerPixMatch)
- **Algorithm**: Advanced patch-based matching with geometric constraints
- **Speed**: Slower but more accurate
- **Best for**: Refined matching, reducing false positives

### Both (Combined Ranking)
When using `match_type='both'`, results are ranked by combined score:
1. Each line gets a CDS rank (1-based, by descending score)
2. Each line gets a PPPM rank (1-based, by descending score)
3. Combined rank = CDS rank + PPPM rank
4. Results sorted by combined rank (lower is better)

This ensures lines that rank well in both algorithms appear first.

---

## Dataset Support

### Supported EM Datasets

| Dataset | Format | Description |
|---------|--------|-------------|
| `hemibrain:v1.2.1` | NeuPrint | Hemibrain connectome |
| `male-cns:v0.9` | NeuPrint | Male CNS (brain + VNC) |
| `flywire_FAFB_v783` | FlyWire | FlyWire FAFB full brain |
| `flywire_BANC_v626` | FlyWire | FlyWire BANC VNC |
| `vnc:v0.5` | NeuPrint | VNC (older version) |
| `manc:v1.2.1` | NeuPrint | MANC dataset |

### Dataset Name Normalization

The module automatically normalizes dataset names for comparison:
- `flywire_FAFB_v783` ↔ `flywire_fafb:v783`
- `hemibrain_v1_2_1` ↔ `hemibrain:v1.2.1`
- `male-cns_v0_9` ↔ `male-cns:v0.9`

This allows seamless matching between local folder names and NeuronBridge API responses.

---

## Examples

### Example 1: Find Lines for a Neuron Type

```python
from src.neuronbridge_finder import NeuronBridgeFinder

nbf = NeuronBridgeFinder()

# Find all body IDs of type MBON01 in hemibrain
# Then find matching driver lines for each
results = nbf.neuron_to_lines(
    query='MBON01',
    dataset='hemibrain:v1.2.1',
    match_type='both'
)

for body_id, lines_df in results.items():
    print(f"\nBody ID {body_id}:")
    print(lines_df[['line', 'score', 'combined_rank']].head(5))
```

### Example 2: Cross-Dataset Search

```python
# Search for aMe12 neurons in both hemibrain and FlyWire FAFB
results = nbf.neuron_to_lines(
    query='aMe12',
    dataset=['hemibrain:v1.2.1', 'flywire_FAFB_v783'],
    match_type='cds'
)
```

### Example 3: Batch Processing with FlyLight Downloads

```python
# Find lines for multiple queries and download representative images
results = nbf.find_lines_batch(
    queries='MBON01,MBON02,MBON03',
    dataset='hemibrain:v1.2.1',
    match_type='both',
    output_dir='./mbon_analysis',
    download_images='flylight',
    download_top_n_img=5,
    flylight_category='SplitGAL4',
    simple_mode=True  # Only download 20x multichannel MIPs
)

print(f"Found {len(results)} total matches")
print(f"Unique lines: {results['line'].nunique()}")
```

### Example 4: Find EM Neurons for a Split-GAL4 Line

```python
# Find which neurons are labeled by SS01015
neurons = nbf.line_to_neuron('SS01015', match_type='both')

# Group by dataset
for dataset in neurons['dataset'].unique():
    subset = neurons[neurons['dataset'] == dataset]
    print(f"\n{dataset}: {len(subset)} neurons")
    print(subset[['bodyId', 'type', 'score']].head())
```

### Example 5: Separate GAL4/LexA from Split-GAL4 Lines

```python
from src.neuronbridge_finder import NeuronBridgeFinder

# Enable separate_splitgal4 mode
nbf = NeuronBridgeFinder(separate_splitgal4=True)

# Find lines for MBON01
# With download_top_n_img=5, downloads:
#   - Top 5 GAL4/LexA lines (VT*, R*, GMR*)
#   - Top 5 Split-GAL4 lines (SS*, LH*, MB*, etc.)
results = nbf.find_lines_batch(
    queries='MBON01',
    dataset='hemibrain:v1.2.1',
    match_type='cds',
    output_dir='./mbon_separated',
    download_images='flylight',
    download_top_n_img=5,  # 5 GAL4/LexA + 5 Split-GAL4 = 10 lines
    simple_mode=True
)

# Results include 'line_type' column
print("\nLine type distribution:")
print(results.groupby('line_type')['line'].nunique())

# Top GAL4/LexA matches
gal4_lines = results[results['line_type'] == 'gal4_lexa']
print("\nTop GAL4/LexA lines:")
print(gal4_lines.groupby('line')['score'].max().nlargest(5))

# Top Split-GAL4 matches
split_lines = results[results['line_type'] == 'split_gal4']
print("\nTop Split-GAL4 lines:")
print(split_lines.groupby('line')['score'].max().nlargest(5))
```

**Output:**
```
Line type distribution:
line_type
gal4_lexa      834
split_gal4     785
Name: line, dtype: int64

Top GAL4/LexA lines:
line
R65E11    0.89
R20H11    0.87
R14C08    0.85
...

Top Split-GAL4 lines:
line
IS49879    0.92
IS83928    0.90
MB011B     0.88
...
```

---

## Simple Mode

Simple mode (`simple_mode=True`) reduces download volume by applying filename-based filtering:

| Collection | Filter | Effect |
|------------|--------|--------|
| Split-GAL4 | `20x` AND `multichannel`, excluding `image1`/`image2` | ~95% reduction |
| VT GAL4 | Files with `total` in filename | ~90% reduction |
| Gen1 R-lines | CDM and MIP files only | Keeps representative images |
| MCFO | Keep all files | Full stochastic labeling data |

This is useful when you only need representative images rather than all available imagery.

---

## Integration with FlyLight Downloader

The NeuronBridgeFinder integrates with the [FlyLight Downloader](./FlyLight_Guide.md) for downloading full-resolution imagery:

```python
# NeuronBridge provides CDM images (color depth masks)
# FlyLight provides full MIP images, aligned stacks, and metadata

results = nbf.find_lines_batch(
    queries='MBON01',
    download_images='flylight',  # Uses FlyLightDownloader internally
    flylight_category=['GAL4/LEXA', 'SplitGAL4'],
    simple_mode=True
)
```

For direct FlyLight access, see the [FlyLight Downloader Guide](./FlyLight_Guide.md).

---

## Troubleshooting

### Common Issues

**Q: API returns no results for a known body ID**
- Check if the body ID exists in NeuronBridge (not all neurons have matches)
- Verify the expected dataset matches the actual dataset in NeuronBridge

**Q: Dataset mismatch errors**
- Use the `dataset` parameter to specify which dataset to search
- The module validates body IDs against their actual dataset in NeuronBridge

**Q: Slow performance**
- Enable caching with `use_cache=True`
- Use `download_top_n_img` to limit image downloads
- Consider using `simple_mode=True` for FlyLight downloads

**Q: Missing neuronbridge-python package**
```bash
pip install neuronbridge-python
```

### Clearing Cache

```python
nbf = NeuronBridgeFinder()

# Clear all cache
nbf.clear_cache()

# Clear specific cache type
nbf.clear_cache(cache_type='id_to_lines')
nbf.clear_cache(cache_type='line_to_neuron')
```

---

## See Also

- [FlyLight Downloader Guide](./FlyLight_Guide.md) - Direct FlyLight imagery access
- [NeuronBridge Website](https://neuronbridge.janelia.org/) - Web interface
- [NeuronBridge Python Package](https://github.com/JaneliaSciComp/neuronbridge-python) - API client
