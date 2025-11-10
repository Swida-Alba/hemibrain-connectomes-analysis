# Folder Naming - Decimal Notation Fix

## Issue
Using 'p' to represent both decimal points and the probability parameter created ambiguity:

**Problem example:**
```
L3_to_Tm3_L2w10r0p01p0p1
                ^  ^^^ ^
                |  ||| └─ Which 'p' is this? Decimal or parameter?
                |  ||└─── Is this 0.1 or parameter p?
                |  |└──── Is this p0 or part of 0p01?
                |  └───── Is this decimal point or probability?
                └──────── This 'p' is clearly decimal point in 0.01
```

Hard to read: Is it `r=0.01, p=0.1` or `r=0, p=01, p=0, p=1`?

## Solution
Use underscore `_` for decimal points instead of 'p':

**Clear example:**
```
L3_to_Tm3_L2w10r0_01p0_1
                ^   ^
                |   └─ Clearly p = 0.1 (probability parameter)
                └───── Clearly r = 0.01 (decimal underscore)
```

Easy to read: `r=0.01, p=0.1` ✓

## Comparison Table

| Scenario | Old (using 'p') | New (using '_') | Clarity |
|----------|----------------|-----------------|---------|
| Basic | `L2w10r0p0` | `L2w10r0p0` | Same (no decimals) |
| Ratio 0.01 | `L2w10r0p01p0` | `L2w10r0_01p0` | ✓ Much clearer |
| Prob 0.000001 | `L2w10r0p0p000001` | `L2w10r0p0_000001` | ✓ Much clearer |
| Both 0.05, 0.1 | `L2w10r0p05p0p1` | `L2w10r0_05p0_1` | ✓ Much clearer |
| Ratio 1.0, Prob 0.5 | `L2w10r1p0p5` | `L2w10r1p0_5` | ✓ Clearer |

## Examples

### Basic Analysis (no decimals needed)
```
Parameters: max_interlayer=2, min_synapse_num=10, min_ratio=0.0, min_traversal_probability=0.0
Old: L3_to_Tm3_L2w10r0p0_20251025_195447
New: L3_to_Tm3_L2w10r0p0_20251025_200012
✓ No change (no decimals involved)
```

### With Ratio Filter (0.01 = 1%)
```
Parameters: max_interlayer=2, min_synapse_num=10, min_ratio=0.01, min_traversal_probability=0.0
Old: L3_to_Tm3_L2w10r0p01p0_20251025_195447 ❌ Confusing (0p01p0)
New: L3_to_Tm3_L2w10r0_01p0_20251025_200012 ✓ Clear (0_01 is decimal)
```

### With Probability Filter (0.000001)
```
Parameters: max_interlayer=2, min_synapse_num=10, min_ratio=0.0, min_traversal_probability=0.000001
Old: L3_to_Tm3_L2w10r0p0p000001_20251025_195447 ❌ Very confusing (p0p000001)
New: L3_to_Tm3_L2w10r0p0_000001_20251025_200012 ✓ Clear (0_000001 is decimal)
```

### With Both Filters (0.05 and 0.1)
```
Parameters: max_interlayer=3, min_synapse_num=5, min_ratio=0.05, min_traversal_probability=0.1
Old: L3_to_Tm3_L3w5r0p05p0p1_20251025_195447 ❌ Ambiguous (0p05p0p1)
New: L3_to_Tm3_L3w5r0_05p0_1_20251025_200012 ✓ Very clear (0_05 and 0_1)
```

### Integer-like Decimals
```
Parameters: max_interlayer=2, min_synapse_num=10, min_ratio=1.0, min_traversal_probability=0.5
Old: L3_to_Tm3_L2w10r1p0p5_20251025_195447 ❌ Confusing (p0p5)
New: L3_to_Tm3_L2w10r1p0_5_20251025_200012 ✓ Clear (0_5 is decimal)
```

## Benefits of Using Underscore

1. **Unambiguous**: `_` can only mean decimal point, never a parameter
2. **Readable**: `r0_01p0_1` is immediately parsed as r=0.01, p=0.1
3. **Consistent**: Parameter 'p' always means probability, never decimal
4. **Standard**: Underscore is commonly used in programming for readability
5. **Filesystem safe**: Like 'p', underscore works on all filesystems

## Implementation

### Code Change (coana.py, lines 732-739)
```python
def format_decimal(val):
    """Format decimal number for folder name, replacing '.' with '_'"""
    if val == int(val):
        return str(int(val))
    else:
        formatted = f"{val:.6f}".rstrip('0').rstrip('.')
        # Replace decimal point with '_' (underscore)
        formatted = formatted.replace('.', '_')  # Changed from 'p' to '_'
        return formatted
```

## Backward Compatibility

Old folders with 'p' notation still exist and work fine:
```
connection_data/
├── L3_to_Tm3_L2w10r0p01p0_20251025_195447/    # Old (confusing but functional)
└── L3_to_Tm3_L2w10r0_01p0_20251025_200012/    # New (clear and functional)
```

New analyses will use underscore notation going forward.

## Summary

**Problem**: 'p' was overloaded (decimal point AND probability parameter)
**Solution**: Use '_' for decimal points, reserve 'p' for probability parameter only
**Result**: Much clearer, unambiguous folder names

---

**Date**: October 25, 2025  
**Change**: Decimal notation from 'p' to '_'  
**Reason**: Avoid ambiguity with probability parameter 'p'
