# Nested Progress Bar Enhancement

## Overview

Enhanced the NeuronBridge specificity calculation with **nested progress bars** showing both line-level and image-level processing, plus cache status indicators for better visibility into what's happening during long-running operations.

## Changes Made

### 1. Nested Progress Bars

**Outer Progress Bar (Line Level)**
```
🔬 [5/20] VT000770:  25%|████████          | 5/20 [0:15<0:45, 0.33line/s] 💾cached
```
- Shows current line being processed
- Position in overall queue (5/20)
- Processing rate and time estimates
- Cache status (💾cached or 🌐fetching)

**Inner Progress Bar (Image Level)**
```
     Processing 75 images:  40%|████████  | 30/75 [0:08<0:12]
```
- Only shown for lines with >10 LM images
- Shows image processing progress within each line
- Auto-disappears when done (`leave=False`)
- Positioned below the main progress bar

### 2. Cache Status Indicators

Added real-time cache status in the progress bar postfix:
- **💾cached**: Line data loaded from cache (fast)
- **🌐fetching**: Making API call to NeuronBridge (slower)

This helps users understand:
- Why some lines process faster than others
- Whether their cache is working effectively
- Expected processing times for remaining lines

### 3. Optimized Verbose Output

**Suppressed redundant messages** when progress bars are active:
- No "🔍 Searching for neurons matching line: X" for each line (already shown in progress bar)
- No "Found N LM images" messages (shown in nested progress bar)
- Cleaner output focused on progress visualization

**Preserved messages** for important events:
- Initial operation start
- Warnings and errors
- Final completion summaries with timing stats

### 4. Smart Progress Detection

Uses frame inspection to detect when running in batch/progress mode:
```python
# Automatically detects if called from within a progress bar loop
in_progress_context = any('tqdm' in str(type(v)).lower() for v in caller_locals.values())
```

This ensures:
- Clean output in batch operations
- Detailed output when called individually
- No duplicate messages

## Visual Example

### Before
```
🔍 Searching for neurons matching line: VT000770
  Found 75 LM images for 'VT000770'
  ✓ Found 234 matches
🔍 Searching for neurons matching line: VT001234
  Found 82 LM images for 'VT001234'
  ✓ Found 189 matches
...
```

### After (with nested progress)
```
📊 Calculating specificity for top 20 of 2171 lines...
   Queried types: ['aMe12']
   ⏱️  Note: Each line requires an API call to fetch neuron matches (may take time)
   🔬 [1/20] VT000770:   5%|█         | 1/20 [0:03<0:57, 0.33line/s] 🌐fetching
     Processing 75 images:  40%|████████  | 30/75 [0:08<0:12]
   🔬 [2/20] VT001234:  10%|██        | 2/20 [0:06<0:54, 0.33line/s] 💾cached
   🔬 [3/20] SS00145:   15%|███       | 3/20 [0:09<0:51, 0.33line/s] 🌐fetching
     Processing 63 images:  60%|████████████| 38/63 [0:06<0:04]
   ...
   ✓ Specificity calculated for 2171 lines
   ⏱️  Processing time: 127.3s total, ~6.4s per line
```

## Technical Details

### Progress Bar Configuration

**Main Progress Bar (Line Level)**
```python
tqdm_progress(
    lines_to_process,
    desc="   🔬 Analyzing specificity",
    unit="line",
    bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
    ncols=110,
    position=0  # Top position
)
```

**Nested Progress Bar (Image Level)**
```python
tqdm_progress(
    lm_images,
    desc=f"     Processing {n_images} images",
    unit="img",
    leave=False,  # Disappears when done
    bar_format='     {desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
    ncols=100,
    position=1  # Below main progress bar
)
```

### Image Progress Threshold

Nested progress bar only shows for lines with >10 images:
```python
show_image_progress = HAS_TQDM and self.verbose and n_images > 10
```

This prevents clutter for lines with few images while providing useful feedback for lines with many images.

### Cache Detection

Checks cache before processing to show accurate status:
```python
cache_key = f"{line_name}_{match_type}"
is_cached = self._load_from_cache('line_to_neuron', cache_key) is not None

if HAS_TQDM and self.verbose:
    status = "💾cached" if is_cached else "🌐fetching"
    iterator.set_postfix_str(status)
```

## Performance Impact

### Cache Effectiveness Visualization

The cache indicators help users see:
- **High cache hit rate**: Many 💾cached → fast processing
- **Low cache hit rate**: Many 🌐fetching → slower, but building cache for next time
- **Mixed pattern**: Some cached, some fetched → typical during incremental work

### Time Estimation Improvements

With nested progress bars, users can better estimate:
1. **Per-line time**: Shown in main progress bar rate
2. **Per-image time**: Visible in nested progress when active
3. **Cache vs. API time**: Compare processing times for cached vs. fetched lines

Example:
```
Cached lines: ~0.5s per line
API fetch lines: ~6.0s per line (with 75 images)
```

## Benefits

1. **Better visibility**: See exactly what's happening at both levels
2. **Time awareness**: Accurate estimates for both line and image processing
3. **Cache feedback**: Know when cache is helping performance
4. **Cleaner output**: No redundant messages cluttering the display
5. **Professional UX**: Nested progress bars like modern CLI tools

## Testing

Run the test script to see the nested progress bars in action:

```bash
python test_progress_bars.py
```

Or test with a real query:
```python
from neuronbridge_finder import NeuronBridgeFinder

nbf = NeuronBridgeFinder(verbose=True)
results = nbf.find_lines_batch(
    queries='aMe12',
    dataset='hemibrain:v1.2.1',
    calculate_specificity=True,
    specificity_top_n=20
)
```

## Dependencies

Requires `tqdm` library (already in requirements):
```bash
pip install tqdm
```

Falls back gracefully if tqdm is not available.
