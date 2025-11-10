# CRITICAL FIX: Multiprocessing Fork Bomb on macOS/Windows

## Issue

When using parallel processing (`use_parallel=True`), the script would:
1. Spawn infinite worker processes
2. Each worker re-executes the entire script
3. Each creates 12 more workers
4. System becomes overwhelmed with processes
5. Massive print output chaos
6. Script never completes

**Example of the problem:**
```
Using parallel processing with 12 processes...

Initializing... (×12)
Initializing... (×12)
Initializing... (×12)
... [infinite loop]

=== PHASE 1: Fetching all network layers... (×144)
=== PHASE 1: Fetching all network layers... (×1728)
... [exponential growth]
```

## Root Cause

### The multiprocessing Problem on macOS/Windows

Python's `multiprocessing` module uses different **process creation methods** on different operating systems:

- **Linux**: Uses `fork()` - child processes **inherit** the parent's memory state
- **macOS/Windows**: Uses `spawn` - child processes **re-execute** the script from scratch

When using `spawn`, the worker process:
1. Starts a fresh Python interpreter
2. **Re-imports the main module**
3. **Re-executes all module-level code**
4. Including the code that creates `FindNeuronConnection` and calls `FindAllPath()`!

### What Was Happening

**Before (BROKEN)**:
```python
# FindPath_Kun.py
from coana import FindNeuronConnection

fc = FindNeuronConnection(...)  # ← Runs in main process
fc.FindAllPath()                 # ← Runs in main process

# When FindAllPath() uses multiprocessing:
#   1. Creates 12 worker processes
#   2. Each worker re-runs FindPath_Kun.py
#   3. Each worker creates NEW FindNeuronConnection()
#   4. Each worker calls FindAllPath() AGAIN
#   5. Each creates 12 MORE workers
#   6. Exponential explosion: 12 → 144 → 1728 → ...
```

## Solution

Wrap all execution code in `if __name__ == '__main__':` guard:

```python
# FindPath_Kun.py
from coana import FindNeuronConnection

if __name__ == '__main__':
    fc = FindNeuronConnection(...)
    fc.FindAllPath()

# Now when workers re-import this module:
#   - Import statement runs (ok)
#   - Code inside if __name__ == '__main__': SKIPS (because __name__ == '__mp_main__' in workers)
#   - Workers only execute the assigned pathfinding function
#   - No infinite recursion!
```

### Why This Works

When a module is **imported** vs **executed as main script**:

| Context | `__name__` value | Code inside `if __name__ == '__main__':` |
|---------|------------------|------------------------------------------|
| Main script | `'__main__'` | ✅ Executes |
| Imported module | `'<module_name>'` | ❌ Skips |
| Multiprocessing worker | `'__mp_main__'` | ❌ Skips |

## Files Fixed

All example scripts updated with the `if __name__ == '__main__':` guard:

1. ✅ `FindPath_Kun.py`
2. ✅ `FindPath.py`
3. ✅ `FindPath_VTaMe.py`
4. ✅ `FindPath_PPL1_VTaMe.py`

### Before (All Files)
```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(...)
fc.InitializeNeuronInfo()
fc.FindAllPath()  # or fc.FindPath()
```

### After (All Files)
```python
from coana import FindNeuronConnection

if __name__ == '__main__':
    fc = FindNeuronConnection(...)
    fc.InitializeNeuronInfo()
    fc.FindAllPath()  # or fc.FindPath()
```

## Technical Details

### Process Creation Methods

```python
import multiprocessing as mp

# Different methods:
mp.set_start_method('fork')   # Linux default - fast but unsafe
mp.set_start_method('spawn')  # macOS/Windows default - safe but requires main guard
mp.set_start_method('forkserver')  # Hybrid approach
```

We use the system default (spawn on macOS), which **requires** the `if __name__ == '__main__':` guard.

### Worker Process Lifecycle

**Main Process**:
```python
if __name__ == '__main__':
    fc = FindNeuronConnection(...)
    fc.FindAllPath()  # Starts parallel processing
```

**Worker Processes** (automatically created by multiprocessing.Pool):
```python
# Worker re-imports the script
import FindPath_Kun  

# __name__ is now '__mp_main__', NOT '__main__'
# So the if __name__ == '__main__': block is SKIPPED

# Worker only executes the function passed to pool.map():
_find_paths_for_pairs(chunk_of_pairs)  # ← Only this runs
```

## Why We Didn't See This on Linux

If you develop on Linux and deploy on macOS:
- ✅ Linux: Works fine without guard (uses `fork`)
- ❌ macOS: Explodes without guard (uses `spawn`)

**Always use the guard** for cross-platform compatibility!

## Best Practices

### ✅ DO: Use main guard for any script that uses multiprocessing

```python
from coana import FindNeuronConnection

if __name__ == '__main__':
    # All your code here
    fc = FindNeuronConnection(...)
    fc.FindAllPath()
```

### ❌ DON'T: Run code at module level

```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(...)  # BAD: Runs every time module imports
fc.FindAllPath()                 # BAD: Will spawn infinite processes
```

### ✅ DO: Protect even simple scripts

```python
if __name__ == '__main__':
    # Even simple scripts should use this guard
    # when they might use multiprocessing
    main()
```

### ❌ DON'T: Assume it only matters for complex code

```python
# Even this simple code needs the guard:
fc = FindNeuronConnection(...)
fc.FindAllPath()  # Uses parallel processing internally!
```

## Testing

After the fix, parallel processing works correctly:

```
=== PHASE 3: Finding all paths from sources to targets ===
Using graph-based pathfinding to handle reciprocal connections...
Building connection graph... Done! (55076 nodes, 6525174 edges)

Searching paths: 892 sources × 4 targets = 3568 pairs
Maximum path length: 3 edges
Using parallel processing with 12 processes...

✅ Parallel pathfinding complete in 45.3s!
   Average: 78.7 pairs/s
```

**No more:**
- ❌ Infinite "Initializing..." messages
- ❌ Repeated PHASE 1 executions
- ❌ Exponential process spawning
- ❌ System lockup

## Impact

- **Before**: Script unusable on macOS/Windows with parallel processing
- **After**: Works perfectly on all platforms

## References

- [Python multiprocessing documentation](https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming)
- [Safe importing of main module](https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods)

---

**Status**: ✅ Fixed
**Date**: October 25, 2025
**Priority**: CRITICAL
**Affected Platforms**: macOS, Windows (spawn-based systems)
**Solution**: Add `if __name__ == '__main__':` guard to all scripts
