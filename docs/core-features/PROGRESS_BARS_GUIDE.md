# Using Progress Bars in NeuronBridge Finder

## Overview

The NeuronBridge Finder now includes detailed progress bars for time-consuming operations, particularly during specificity calculation and line-to-neuron fetching. This helps you monitor long-running queries and estimate completion times.

## Requirements

Progress bars require the `tqdm` library:

```bash
pip install tqdm
```

If `tqdm` is not installed, the code will work normally but without progress bars.

## Enabling Progress Bars

Progress bars are automatically shown when:
1. `tqdm` is installed
2. `verbose=True` is set when creating the NeuronBridgeFinder instance

```python
from neuronbridge_finder import NeuronBridgeFinder

# Enable progress bars
nbf = NeuronBridgeFinder(verbose=True)

# Without progress bars
nbf_quiet = NeuronBridgeFinder(verbose=False)
```

## Batch Mode & Warning Summary (v4.3.2+)

During batch operations, verbose output is automatically compressed:

- **Cache messages suppressed**: Load/save cache messages are hidden during batch loops
- **Warnings collected**: Non-critical warnings are collected and summarized at the end
- **Progress bar active**: Success messages (e.g., "✓ Found X images") are hidden when progress bar is running

**Warning Summary Output** (printed after batch completion):
```
⚠️  Warnings encountered during processing:
   • VT lines: VT037867 not found in FlyLight Gen1-GAL4 collection (2 occurrences)
   • VT lines: VT001234 not found in FlyLight Gen1-GAL4 collection
```

Warnings are grouped by category and deduplicated with occurrence counts, keeping the output clean and actionable.

## What Gets Progress Bars?

### 1. Specificity Calculation

When `calculate_specificity=True` in `find_lines_batch()`:

```python
results = nbf.find_lines_batch(
    queries='aMe12,MBON01',
    dataset='hemibrain:v1.2.1',
    calculate_specificity=True,
    specificity_top_n=100  # Process top 100 lines
)
```

**Output:**
```
📊 Calculating specificity for top 100 of 250 lines...
   Queried types: ['aMe12', 'MBON01']
   ⏱️  Note: Each line requires an API call to fetch neuron matches (may take time)
   🔬 [23/100] Fetching neurons for VT037867: 23%|██▎       | 23/100 [0:45<2:30, 0.51line/s]
```

The progress bar shows:
- Current line being processed (`VT037867`)
- Progress: `23/100` (23%)
- Elapsed time: `0:45` (45 seconds)
- Estimated remaining: `2:30` (2 minutes 30 seconds)
- Processing rate: `0.51line/s`

After completion:
```
   ✓ Specificity calculated for 250 lines
   ⏱️  Processing time: 127.3s total, ~1.27s per line
```

### 2. Co-labeling Matrix

When building co-labeling similarity matrices:

```
🔗 Building co-labeling matrix for 50 lines...
   ⏱️  Note: Fetching neurons for each line to build similarity matrix
   🔍 [15/50] Fetching neurons for SS00324: 30%|███       | 15/50 [0:28<1:05, 0.53line/s]
   🔢 Computing Jaccard similarities between 50 lines...
```

### 3. Mutual Information

When calculating type-line mutual information:

```
📊 Calculating mutual information for 50 lines...
   ⏱️  Note: Fetching neuron types for each line (may take time)
   🧬 [35/50] Processing VT045123: 70%|███████   | 35/50 [1:12<0:31, 0.48line/s]
```

## Understanding the Progress Output

### Progress Bar Format

```
🔬 [23/100] Fetching neurons for VT037867: 23%|██▎       | 23/100 [0:45<2:30, 0.51line/s]
│    │       │                             │     │         │       │      │      │
│    │       │                             │     │         │       │      │      └─ Rate (lines/second)
│    │       │                             │     │         │       │      └─ Remaining time
│    │       │                             │     │         │       └─ Elapsed time
│    │       │                             │     │         └─ Completed/Total count
│    │       │                             │     └─ Visual progress bar
│    │       │                             └─ Percentage
│    │       └─ Current line name
│    └─ Current position/Total
└─ Operation icon
```

### Time Estimates

- **Elapsed time**: How long the operation has been running
- **Remaining time**: Estimated time to completion (based on current rate)
- **Processing rate**: Lines processed per second

**Note:** Time estimates become more accurate as more items are processed.

### Icons

- 🔬 - Specificity analysis
- 🔍 - Collecting/fetching data
- 🧬 - Building matrices
- 🔢 - Computing statistics
- ⏱️ - Timing/performance hint
- ✓ - Completion

## Performance Tips

### 1. Limit Processing with `specificity_top_n`

Only calculate specificity for top N lines to reduce API calls:

```python
# Process only top 50 lines (much faster)
results = nbf.find_lines_batch(
    queries='aMe12',
    calculate_specificity=True,
    specificity_top_n=50  # Default: 100
)
```

### 2. Monitor Cache Usage

The progress bar will show faster processing for cached lines. If you see:
- Fast processing (>2 lines/sec): Lines are cached
- Slow processing (<1 line/sec): Making API calls

### 3. Estimate Total Time

After processing a few lines, use the rate to estimate:

```
Rate: 0.5 lines/sec
Lines to process: 100
Estimated time: 100 / 0.5 = 200 seconds (~3.3 minutes)
```

## Example Workflow

```python
from neuronbridge_finder import NeuronBridgeFinder

# Initialize with verbose mode
nbf = NeuronBridgeFinder(verbose=True)

# Query with progress monitoring
results = nbf.find_lines_batch(
    queries='aMe12,MBON01,aIPg',  # Multiple neurons
    dataset='hemibrain:v1.2.1',
    match_type='cds',
    calculate_specificity=True,
    specificity_top_n=100,  # Limit to top 100 lines
    output_dir='./output'
)
```

**Expected output:**
```
🔍 Finding lines for 3 query(s)
   Output: ./output/findlines_20231223_143022

📋 Processing: aMe12
  Found 3 neurons to search
  ...
   ✅ Found 142 matching driver lines

📊 Calculating specificity for top 100 of 142 lines...
   Queried types: ['aMe12', 'MBON01', 'aIPg']
   ⏱️  Note: Each line requires an API call to fetch neuron matches (may take time)
   🔬 [50/100] Fetching neurons for VT037867: 50%|█████     | 50/100 [1:23<1:23, 0.60line/s]
   ...
   ✓ Specificity calculated for 142 lines
   ⏱️  Processing time: 167.5s total, ~1.68s per line

🔗 Building co-labeling matrix for top 100 lines...
   ⏱️  Note: Fetching neurons for each line to build similarity matrix
   🔍 [75/100] Fetching neurons for SS00324: 75%|███████▌  | 75/100 [2:05<0:42, 0.60line/s]
   🔢 Computing Jaccard similarities between 100 lines...
   💾 Co-labeling matrix: ./output/findlines_20231223_143022/colabeling_matrix.csv

📐 Calculating mutual information...
   ⏱️  Note: Fetching neuron types for each line (may take time)
   🧬 [100/100] Processing VT045123: 100%|██████████| 100/100 [2:47<0:00, 0.60line/s]
   ✓ MI calculated for 100 lines
```

## Troubleshooting

### Progress Bar Not Showing

**Problem:** No progress bars appear

**Solutions:**
1. Check if `tqdm` is installed: `pip install tqdm`
2. Ensure `verbose=True`: `nbf = NeuronBridgeFinder(verbose=True)`
3. Progress bars only appear for operations with multiple items

### Progress Stuck at 0%

**Problem:** Progress bar shows 0% for a long time

**Cause:** First API call is slow or line has many matches

**Solution:** Wait for first item to complete, then rate will stabilize

### Inaccurate Time Estimates

**Problem:** "Remaining time" jumps around

**Cause:** Variable API response times (network, server load, cache hits)

**Solution:** Time estimates become more accurate after ~10% completion

## Advanced Usage

### Disable Progress Bars Temporarily

```python
# Create with progress enabled
nbf = NeuronBridgeFinder(verbose=True)

# Temporarily disable for a single operation
nbf.verbose = False
results = nbf.find_lines_batch(...)  # No progress bars
nbf.verbose = True  # Re-enable
```

### Capture Progress for Logging

Progress information is printed using the internal `_vprint()` method, which goes to stdout when `verbose=True`. You can redirect stdout to capture progress:

```python
import sys
from io import StringIO

# Redirect stdout
old_stdout = sys.stdout
sys.stdout = log_buffer = StringIO()

# Run operation (progress goes to buffer)
results = nbf.find_lines_batch(...)

# Restore stdout
sys.stdout = old_stdout

# Get progress log
progress_log = log_buffer.getvalue()
```
