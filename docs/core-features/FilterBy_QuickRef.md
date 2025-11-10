# filter_by Quick Reference

## Syntax

```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(
    sourceNeurons=['L3'],
    targetNeurons=['Tm3'],
    min_synapse_num=10,
    min_ratio=0.01,
    min_traversal_probability=0.001,
    filter_by='bodyId',  # or 'type'
    # ... other parameters
)
```

## Options

| Value | Behavior | Use When |
|-------|----------|----------|
| `'bodyId'` | Filter each connection individually | You want strong individual connections |
| `'type'` | Filter aggregated type pairs | You want type-level patterns |

## Default
`filter_by='bodyId'` (original behavior, most conservative)

## Quick Examples

### Conservative (Individual Neurons)
```python
fc = FindNeuronConnection(
    sourceNeurons=['L3'],
    targetNeurons=['Tm3'],
    min_ratio=0.05,  # Each connection must be ≥5%
    filter_by='bodyId'
)
```
**Output**: Only connections where individual L3[i]→Tm3[j] ≥ 5% of Tm3[j]'s inputs

### Inclusive (Type Level)
```python
fc = FindNeuronConnection(
    sourceNeurons=['L3'],
    targetNeurons=['Tm3'],
    min_ratio=0.01,  # Total L3→Tm3 must be ≥1%
    filter_by='type'
)
```
**Output**: All L3→Tm3 connections if total aggregate ≥ 1% of all Tm3 inputs

## Visual Comparison

### filter_by='bodyId'
```
Before: 10,000 connections
  ↓
Filter each connection: weight/post ≥ 0.01
  ↓
After: 150 strong connections
```

### filter_by='type'
```
Before: 10,000 connections
  ↓
Group by type → 50 type pairs
Filter type pairs: Σweight/Σpost ≥ 0.01
  ↓
Keep connections from 12 passing type pairs
  ↓
After: 7,500 connections
```

## Decision Tree

```
Do you care about individual neuron identities?
├─ YES → Use filter_by='bodyId'
│         (Find specific strong connections)
│
└─ NO → Do you want type-level circuit view?
         ├─ YES → Use filter_by='type'
         │         (See complete type connectivity)
         │
         └─ NOT SURE → Start with 'bodyId', then try 'type'
```

## Common Mistakes

### ❌ Wrong: Using same thresholds for both modes
```python
# Too strict for type-level
fc = FindNeuronConnection(
    min_ratio=0.1,  # 10% is very high!
    filter_by='type'
)
# Result: May filter out ALL type pairs
```

### ✅ Right: Adjust thresholds for filtering mode
```python
# BodyId: Strict (looking for strong connections)
fc1 = FindNeuronConnection(
    min_ratio=0.05,  # 5% for individual connections
    filter_by='bodyId'
)

# Type: More permissive (aggregate is already conservative)
fc2 = FindNeuronConnection(
    min_ratio=0.01,  # 1% for type aggregates
    filter_by='type'
)
```

## Folder Names

Both modes create separate folders:
```
connection_data/
├── L3_to_Tm3_L2w10r0_05p0_20251025_143022/  # bodyId mode (fewer connections)
└── L3_to_Tm3_L2w10r0_01p0_20251025_143145/  # type mode (more connections)
```

Check `parameters.txt` in each folder to see which mode was used.

## Performance

| Mode | Speed | Memory | Output Size |
|------|-------|--------|-------------|
| `bodyId` | Faster ⚡ | Lower 💾 | Smaller 📄 |
| `type` | Slower 🐌 | Higher 💾💾 | Larger 📄📄 |

## Validation

Invalid values raise error:
```python
fc = FindNeuronConnection(filter_by='invalid')
# ValueError: filter_by must be 'bodyId' or 'type', got 'invalid'
```

## When in Doubt

**Default to `filter_by='bodyId'`** - it's more conservative and faster.

Then explore with `filter_by='type'` if you need the full circuit view.

---

**Quick Start**: Just add `filter_by='type'` to any existing analysis to see type-level patterns!
