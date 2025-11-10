# Documentation Corrections Summary

## Issue
The parallel processing documentation initially created used incorrect syntax:
1. **Wrong class name**: Used `Connectome` instead of `FindNeuronConnection`
2. **Wrong parameter format**: Used separate `dataset='hemibrain'` and `version='v1.2.1'` instead of combined `dataset='hemibrain:v1.2.1'`

## Root Cause
When implementing the parallel processing feature, I incorrectly assumed the class was named `Connectome` and that dataset/version were separate parameters. The actual implementation uses:
- Class name: `FindNeuronConnection`
- Dataset parameter: `dataset='hemibrain:v1.2.1'` (combined format)

## Files Corrected

### 1. FindPath_Kun.py ✅
**Changes:**
- Added `use_cache=True` parameter
- Added `use_parallel=True` parameter  
- Added `n_jobs=-1` parameter
- Added explanatory comments
- Fixed spacing/formatting for consistency

**Before:**
```python
fc = FindNeuronConnection(
    token='',
    data_folder='/Users/apple/Local/connection_data',
    dataset = 'optic-lobe:v1.1', 
    sourceNeurons = ['L3.*_R'],
    targetNeurons = ['l-LNv.*_R'],
    ...
)
```

**After:**
```python
fc = FindNeuronConnection(
    token='',
    data_folder='/Users/apple/Local/connection_data',
    dataset='optic-lobe:v1.1',  # Combined dataset and version
    sourceNeurons=['L3.*_R'],
    targetNeurons=['l-LNv.*_R'],
    ...
    use_cache=True,  # Enable caching
    use_parallel=True,  # Enable parallel processing
    n_jobs=-1,  # Use all CPU cores
)
```

### 2. ParallelProcessing_Documentation.md ✅
**Changes:**
- Replaced all `Connectome(` with `FindNeuronConnection(`
- Combined `dataset='hemibrain', version='v1.2.1'` into `dataset='hemibrain:v1.2.1'`
- Added required parameters (`token`, `sourceNeurons`, `targetNeurons`)
- Added method calls (`InitializeNeuronInfo()`, `FindAllPath()`)

**Example Fix:**
```python
# Before (WRONG):
conn = Connectome(
    dataset='hemibrain',
    version='v1.2.1',
    use_parallel=True
)
result = conn.FindAllPath(source=['PPL1-01'], target=['MBON14'])

# After (CORRECT):
fc = FindNeuronConnection(
    token='your_token_here',
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['PPL1-01'],
    targetNeurons=['MBON14'],
    max_interlayer=3,
    use_parallel=True
)
fc.InitializeNeuronInfo()
fc.FindAllPath()
```

### 3. README.md ✅
**Changes:**
- Replaced all `Connectome(` with `FindNeuronConnection(`
- Combined `dataset='hemibrain', version='v1.2.1'` into `dataset='hemibrain:v1.2.1'`
- Added required parameters and method calls
- Updated all example code blocks

### 4. ParallelProcessing_QuickReference.md ✅
**Changes:**
- Fixed TL;DR section with correct class and parameters
- Updated all "Quick Settings" examples
- Fixed "One-Line Enable" comparison at the end

### 5. Example_ParallelProcessing.py ✅
**Changes:**
- Changed import from `from coana import Connectome` to `from coana import FindNeuronConnection`
- Replaced all `Connectome(` with `FindNeuronConnection(`
- Combined dataset/version parameters
- Added required parameters (`token`, `sourceNeurons`, `targetNeurons`, `max_interlayer`, `showfig`)
- Added `InitializeNeuronInfo()` and `FindAllPath()` method calls
- Removed return statements (methods don't return values directly)

## Verification

All corrected files have been verified:
- ✅ No syntax errors in FindPath_Kun.py
- ✅ No syntax errors in Example_ParallelProcessing.py
- ✅ Documentation uses correct class name throughout
- ✅ All examples use combined dataset parameter format
- ✅ All examples include required parameters and method calls

## Correct Usage Pattern

The correct pattern for using FindNeuronConnection with parallel processing:

```python
from coana import FindNeuronConnection

# Create instance with all required parameters
fc = FindNeuronConnection(
    token='your_auth_token',               # Required: from neuprint.janelia.org
    dataset='hemibrain:v1.2.1',            # Combined dataset:version format
    sourceNeurons=['PPL1-01', 'PPL1-02'],  # List of source neuron types/bodyIds
    targetNeurons=['MBON14', 'MBON11'],    # List of target neuron types/bodyIds
    max_interlayer=3,                      # Maximum path length
    use_cache=True,                        # Enable caching (default)
    use_parallel=True,                     # Enable parallel processing
    n_jobs=-1                              # Number of cores (-1 = all)
)

# Initialize neuron information
fc.InitializeNeuronInfo()

# Find all paths (uses parallel processing if enabled)
fc.FindAllPath()
```

## Available Datasets

All datasets use the combined format `'dataset:version'`:
- `'fib19:v1.0'`
- `'hemibrain:v0.9'`
- `'hemibrain:v1.0.1'`
- `'hemibrain:v1.1'`
- `'hemibrain:v1.2.1'`
- `'manc:v1.0'`
- `'optic-lobe:v1.1'`

## Summary

All documentation has been corrected to use:
1. ✅ Correct class name: `FindNeuronConnection`
2. ✅ Combined dataset parameter: `dataset='hemibrain:v1.2.1'`
3. ✅ Required parameters: token, sourceNeurons, targetNeurons
4. ✅ Proper method calls: InitializeNeuronInfo(), FindAllPath()
5. ✅ Correct parallel processing parameters: use_parallel, n_jobs

---

**Date**: October 25, 2025
**Status**: All corrections complete and verified
