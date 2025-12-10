# Folder Naming Convention

## Overview

Data folders are automatically named with descriptive parameter suffixes and timestamps that make it easy to identify analysis settings and run times at a glance.

## Format

```
{source}_to_{target}_L{layer}w{weight}r{ratio}p{prob}_{timestamp}
```

### Components

- **Source and Target**: Neuron type names (e.g., `L3_to_Tm3`)
- **L{layer}**: `max_interlayer` value (number of intermediate layers)
- **w{weight}**: `min_synapse_num` value (minimum synapse count)
- **r{ratio}**: `min_ratio` value (**ALWAYS included**, even if 0)
- **p{prob}**: `min_traversal_probability` value (**ALWAYS included**, even if 0)
- **{timestamp}**: Run timestamp in format `YYYYMMDD_HHMMSS`

### Decimal Formatting

To avoid dots (`.`) in folder names and prevent ambiguity with the `p` (probability) parameter:
- **Decimal points are replaced with underscore `_`**
- Trailing zeros removed: `0.100000` → `0_1`
- Integer values show no decimal: `1.0` → `1`
- Scientific notation expanded: `1e-6` → `0_000001`
- Zero values: `0.0` → `0`

**Examples:**
- `0.01` → `0_01`
- `0.5` → `0_5`
- `1.0` → `1`
- `0.000001` → `0_000001`
- `0.0` → `0`

**Why underscore instead of 'p'?**
Using `_` avoids confusion with the `p` parameter (traversal probability). Compare:
- **Old (ambiguous)**: `r0p01p0p1` - Which 'p' is the parameter vs decimal?
- **New (clear)**: `r0_01p0_1` - Clearly shows r=0.01, p=0.1

## Examples

### Basic Analysis (no ratio/probability filters)
```python
ca = FindNeuronConnection(
    sourceNeurons=['L3'],
    targetNeurons=['Tm3'],
    max_interlayer=2,
    min_synapse_num=10,
    min_ratio=0.0,
    min_traversal_probability=0.0
)
```
**Folder**: `L3_to_Tm3_L2w10r0p0_20251025_200012`

### With Connection Ratio Filter
```python
ca = FindNeuronConnection(
    sourceNeurons=['L3'],
    targetNeurons=['Tm3'],
    max_interlayer=2,
    min_synapse_num=10,
    min_ratio=0.01,  # ≥1% of inputs
    min_traversal_probability=0.0
)
```
**Folder**: `L3_to_Tm3_L2w10r0_01p0_20251025_200012`

### With Traversal Probability Filter
```python
ca = FindNeuronConnection(
    sourceNeurons=['L3'],
    targetNeurons=['Tm3'],
    max_interlayer=2,
    min_synapse_num=10,
    min_ratio=0.0,
    min_traversal_probability=0.000001  # Very weak connections allowed
)
```
**Folder**: `L3_to_Tm3_L2w10r0p0_000001_20251025_200012`

### With Both Filters
```python
ca = FindNeuronConnection(
    sourceNeurons=['L3'],
    targetNeurons=['Tm3'],
    max_interlayer=3,
    min_synapse_num=5,
    min_ratio=0.05,  # ≥5% of inputs
    min_traversal_probability=0.1  # ≥10% chance
)
```
**Folder**: `L3_to_Tm3_L3w5r0_05p0_1_20251025_200012`

### Deep Network
```python
ca = FindNeuronConnection(
    sourceNeurons=['PPL1'],
    targetNeurons=['VTa'],
    max_interlayer=5,
    min_synapse_num=3,
    min_ratio=0.02,
    min_traversal_probability=0.05
)
```
**Folder**: `PPL1_to_VTa_L5w3r0_02p0_05_20251025_200012`

### Integer-like Decimals
```python
ca = FindNeuronConnection(
    sourceNeurons=['L3'],
    targetNeurons=['Tm3'],
    max_interlayer=2,
    min_synapse_num=10,
    min_ratio=1.0,
    min_traversal_probability=0.5
)
```
**Folder**: `L3_to_Tm3_L2w10r1p0_5_20251025_200012`

## Folder Contents

Each folder contains:

### Configuration Files
- **`all_attributes.json`**: Complete object state (all parameters and settings)
- **`parameters.txt`**: Human-readable parameter summary

### Analysis Outputs
- **Direct connections** (if `FindDirectConnections()` called):
  - `direct_{weight}snp/` subfolder
  - Excel file with connection matrix
  - Heatmaps and visualizations

- **Path analysis** (if `FindPath()` or `FindAllPath()` called):
  - `allpaths_L{layer}w{weight}/` subfolder
  - Excel file with all paths
  - Path visualizations

## Benefits

### 1. **Easy Identification**
Folder name tells you immediately what filters were used:
```
L3_to_Tm3_L2w10r0_01p0_20251025_200012
          ^  ^  ^    ^  ^
          |  |  |    |  └─ timestamp (Oct 25, 2025 at 20:00:12)
          |  |  |    └──── probability = 0 (no filter)
          |  |  └────────── ratio = 0.01 (1%)
          |  └─────────────── weight ≥ 10 synapses
          └────────────────── max 2 intermediate layers
```

### 2. **Unique Runs**
Every analysis run gets a unique timestamp:
```
L3_to_Tm3_L2w10r0_01p0_20251025_143022/  # Morning run
L3_to_Tm3_L2w10r0_01p0_20251025_200012/  # Evening run
```

### 3. **Compare Analyses**
Easily compare results from different filter settings or times:
```
connection_data/
├── L3_to_Tm3_L2w5r0p0_20251025_100000/        # Lenient (5 synapses)
├── L3_to_Tm3_L2w10r0p0_20251025_110000/       # Moderate (10 synapses)
├── L3_to_Tm3_L2w20r0p0_20251025_120000/       # Stringent (20 synapses)
└── L3_to_Tm3_L2w10r0_01p0_20251025_130000/    # With ratio filter (1%)
```

### 4. **Self-Documenting**
No need to open files to know analysis settings:
```bash
ls connection_data/
# L3_to_Tm3_L2w10r0_01p0_20251025_200012  ← see all parameters + when it ran
```

### 5. **Filesystem Safe**
No dots in folder names (uses underscore notation) avoids:
- Filesystem parsing issues
- File extension confusion
- Cross-platform compatibility problems

### 6. **Complete Visibility**
r0 and p0 shown even when filters not applied:
- **Clear intent**: You see that ratio/probability filters exist (just set to 0)
- **No ambiguity**: `r0` means "no ratio filter" vs missing `r` (unclear)
- **Consistent format**: All folders have same structure

### 7. **Unambiguous Decimals**
Using `_` for decimal points prevents confusion with `p` (probability parameter):
- **Clear separation**: `r0_01p0_1` clearly shows r=0.01, p=0.1
- **No confusion**: Unlike `r0p01p0p1` where 'p' has multiple meanings

## Parameter Reference

### max_interlayer (L)
- Number of intermediate layers between source and target
- L0 = direct connections only
- L1 = source → intermediate → target
- L2 = source → inter1 → inter2 → target
- L3 = source → inter1 → inter2 → inter3 → target

### min_synapse_num (w)
- Minimum number of synapses for a connection
- Absolute threshold (e.g., w10 = at least 10 synapses)
- Higher values = stronger connections only

### min_ratio (r)
- Minimum connection ratio (weight / post-synaptic inputs)
- Relative threshold (e.g., r0_01 = connection is ≥1% of inputs)
- Range: 0.0 to 1.0
- **ALWAYS shown** (r0 if no filter applied)

### min_traversal_probability (p)
- Minimum traversal probability (connection_ratio / 0.3, capped at 1.0)
- Probabilistic threshold (e.g., p0_1 = ≥10% chance of signal transmission)
- Range: 0.0 to 1.0
- **ALWAYS shown** (p0 if no filter applied)

### timestamp
- Format: YYYYMMDD_HHMMSS (year-month-day_hour-minute-second)
- Automatically generated when analysis starts
- Ensures each run has unique folder (no overwrites)
- Sortable chronologically

## Custom Folder Names

You can still specify custom folder names:

```python
ca = FindNeuronConnection(
    sourceNeurons=['L3'],
    targetNeurons=['Tm3'],
    save_folder='my_custom_analysis'  # Will use this instead
)
```
**Folder**: `my_custom_analysis` (no parameter suffix or timestamp)

## Backward Compatibility

Old scripts without `min_ratio` will work fine:
- `min_ratio` defaults to 0.0 (shown as r0 in folder name)
- `min_traversal_probability` defaults to 0.0 (shown as p0)
- Existing analyses not affected
- New parameters are completely optional

## Migration

Existing folders from old analyses remain unchanged:
```
connection_data/
├── L3_to_Tm3/                                    # Very old format
├── L3_to_Tm3_L2w10/                             # Old format (before r/p always shown)
├── L3_to_Tm3_L2w10r0p01/                        # Old format (before timestamps, used 'p' for decimals)
└── L3_to_Tm3_L2w10r0_01p0_20251025_200012/     # New format (underscore for decimals)
```

All formats coexist peacefully!

## Implementation

The folder name is generated in `InitializeNeuronInfo()` method in `coana.py` (lines 728-760):

```python
import datetime

def format_decimal(val):
    """Format decimal number for folder name (replace . with _)"""
    if val == int(val):
        return str(int(val))
    else:
        formatted = f"{val:.6f}".rstrip('0').rstrip('.')
        # Replace decimal point with '_' to avoid dots and confusion with 'p' parameter
        return formatted.replace('.', '_')

# ALWAYS include all parameters (even if 0) and timestamp
param_suffix = f"_L{self.max_interlayer}w{self.min_synapse_num}"
param_suffix += f"r{format_decimal(self.min_ratio)}"
param_suffix += f"p{format_decimal(self.min_traversal_probability)}"
param_suffix += f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

self.save_folder = os.path.join(
    self.output_dir, 
    self.source_fname + '_to_' + self.target_fname + param_suffix
)
```

## Key Design Decisions

### Why always show r0/p0?
- **Clarity**: Makes it obvious that ratio/probability filters exist in the system
- **Consistency**: All folders have same format (easier to parse programmatically)
- **No ambiguity**: `r0` clearly means "no ratio filter" vs missing `r` (could mean old version)

### Why use '_' instead of '.'?
- **Filesystem safety**: Dots can cause issues on some systems
- **No file extension confusion**: `folder.01` might be interpreted as file with extension `.01`
- **Parsing simplicity**: Easier to split on known characters

### Why use '_' instead of 'p'?
- **Avoid ambiguity**: 'p' is the traversal probability parameter
- **Clear distinction**: `r0_01p0_1` is immediately readable as r=0.01, p=0.1
- **No confusion**: Unlike `r0p01p0p1` where 'p' has multiple meanings

### Why add timestamps?
- **Unique runs**: Same parameters at different times create different folders
- **No overwrites**: Never accidentally replace old results
- **Time tracking**: Know when each analysis was performed
- **Easy cleanup**: Delete old runs by date

### Why this timestamp format?
- **Sortable**: YYYYMMDD_HHMMSS sorts chronologically
- **Readable**: Clear year/month/day and hour/minute/second
- **No special chars**: No colons or slashes that could cause filesystem issues

---

**Date**: October 25, 2025  
**Version**: Updated with timestamp and underscore notation for decimals  
**Benefits**: Self-documenting, unique runs, filesystem-safe, complete visibility, unambiguous
