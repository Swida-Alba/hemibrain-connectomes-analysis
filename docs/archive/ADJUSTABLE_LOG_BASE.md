# Adjustable Logarithmic Base for Edge Width Scaling

## Overview

Edge width scaling now supports **adjustable logarithm base** when using the `'log'` scaling method. This allows fine-tuned control over how edge weight differences are compressed, from subtle (high base) to pronounced (low base).

## Why Adjustable Log Base?

Different log bases compress weight ranges differently:

| Log Base | Formula | Compression | Best For |
|----------|---------|-------------|----------|
| **e ≈ 2.718** | ln(x) | Moderate | General purpose, balanced view |
| **2** | log₂(x) | More spread | Binary scales, showing more differences |
| **10** | log₁₀(x) | Least spread | Wide ranges (10¹ to 10⁶), orders of magnitude |
| **Custom** | log_b(x) | Variable | Specific data distributions |

### Example with Weights: 10, 100, 1000, 10000

```
Natural log (base e):
  log(10) ≈ 2.4,  log(100) ≈ 4.6,  log(1000) ≈ 6.9,  log(10000) ≈ 9.2
  Range: 6.8 units

Binary log (base 2):
  log₂(10) ≈ 3.3, log₂(100) ≈ 6.6, log₂(1000) ≈ 10.0, log₂(10000) ≈ 13.3
  Range: 10.0 units (more spread than base e)

Common log (base 10):
  log₁₀(10) = 1,  log₁₀(100) = 2,  log₁₀(1000) = 3,  log₁₀(10000) = 4
  Range: 3.0 units (least spread, linear in orders of magnitude)
```

**Higher base = less spread (more compression)**
**Lower base = more spread (less compression)**

## API Usage

### Python Initialization

```python
from vispath import VisualizePath

# Natural logarithm (default, base e)
vis = VisualizePath(
    'pathways.csv',
    edge_width_scale='log',
    edge_width_log_base=None  # or 'e'
)

# Binary logarithm (base 2)
vis = VisualizePath(
    'pathways.csv',
    edge_width_scale='log',
    edge_width_log_base=2
)

# Common logarithm (base 10)
vis = VisualizePath(
    'pathways.csv',
    edge_width_scale='log',
    edge_width_log_base=10
)

# Custom logarithm (any base > 1)
vis = VisualizePath(
    'pathways.csv',
    edge_width_scale='log',
    edge_width_log_base=5.0  # Base 5
)

vis.create_network()
```

### Interactive Controls (Browser)

1. **Open network visualization** (`network_selected_paths.html`)
2. **Click color palette icon** (🎨)
3. **Select "Logarithmic" from Edge Width Scale dropdown**
4. **Log Base dropdown appears** with options:
   - **e (natural)** - Default, moderate compression
   - **2 (binary)** - More spread than natural log
   - **10 (common)** - Shows orders of magnitude
   - **Custom...** - Enter any base value (1.1 to 100)

5. **For Custom:**
   - Select "Custom..." from dropdown
   - Enter base value in number input (e.g., 5, 7, 20)
   - Changes apply instantly

## Mathematical Implementation

### Change of Base Formula

For any logarithm base `b`:
```
log_b(x) = ln(x) / ln(b)
```

### Python Backend (`_calculate_edge_widths`)

```python
if self.edge_width_scale == 'log':
    if self.edge_width_log_base == 'e' or self.edge_width_log_base is None:
        # Natural logarithm
        scaled = np.log(weights + 1)
    else:
        # Custom base
        log_base = float(self.edge_width_log_base)
        if log_base > 1:
            scaled = np.log(weights + 1) / np.log(log_base)
        else:
            # Fallback to natural log for invalid base
            scaled = np.log(weights + 1)
```

### JavaScript Frontend (`updateEdgeWidths`)

```javascript
if (scalingMethod === 'log') {
    if (logBase === 'e') {
        scaled = Math.log(w + 1);  // Natural log
    } else {
        const base = parseFloat(logBase);
        scaled = Math.log(w + 1) / Math.log(base);  // Change of base
    }
}
```

## Visual Comparison

Created test networks with weights 10, 100, 1000, 10000:

```bash
# Generate comparison visualizations
python -c "
from vispath import VisualizePath
import pandas as pd

data = pd.DataFrame({
    'path_block': ['A->B', 'C->D', 'E->F', 'G->H'],
    'weights': ['[10]', '[100]', '[1000]', '[10000]']
})

for base in ['e', 2, 10]:
    VisualizePath(
        data,
        output_folder=f'test_output/base_{base}',
        edge_width_scale='log',
        edge_width_log_base=base
    ).create_network()
"
```

**Visual Results:**
- **Base e:** Moderate differences, balanced appearance
- **Base 2:** Larger differences, more variation visible
- **Base 10:** Most compressed, focuses on order of magnitude

## Use Cases

### 1. Wide Range of Synapse Counts (1 to 100,000)

**Recommendation:** Base 10 (common log)

```python
vis = VisualizePath(
    'pathways.csv',
    edge_width_scale='log',
    edge_width_log_base=10,
    edge_width_factor=1.5
)
```

**Why:** Common log groups weights by orders of magnitude (10¹, 10², 10³, etc.), making it easier to see which connections differ by 10x vs 100x.

### 2. Moderate Range (10 to 10,000)

**Recommendation:** Base e (natural log, default)

```python
vis = VisualizePath(
    'pathways.csv',
    edge_width_scale='log',
    edge_width_log_base=None  # or 'e'
)
```

**Why:** Natural log provides good balance without over-compressing differences.

### 3. Binary/Computing Context

**Recommendation:** Base 2 (binary log)

```python
vis = VisualizePath(
    'pathways.csv',
    edge_width_scale='log',
    edge_width_log_base=2
)
```

**Why:** Binary log shows doublings (2¹, 2², 2³...), natural for computing/information theory contexts.

### 4. Emphasize Small Differences

**Recommendation:** Low base (e.g., 1.5 to 3)

```python
vis = VisualizePath(
    'pathways.csv',
    edge_width_scale='log',
    edge_width_log_base=1.5,  # Very low compression
    edge_width_factor=0.8      # Scale down to avoid too-thick edges
)
```

**Why:** Lower bases spread values more, making small differences more visible.

### 5. Compress Large Differences

**Recommendation:** High base (e.g., 20 to 50)

```python
vis = VisualizePath(
    'pathways.csv',
    edge_width_scale='log',
    edge_width_log_base=20,   # High compression
    edge_width_factor=2.0      # Scale up to maintain visibility
)
```

**Why:** Higher bases compress differences more, preventing a few huge edges from dominating.

## Interactive Workflow

**Scenario:** You have weights ranging from 5 to 50,000 and want to find the best visualization.

### Step 1: Start with defaults
```python
vis = VisualizePath('pathways.csv')  # Uses log base e
vis.create_network()
```

### Step 2: Open in browser and experiment
1. Open `network_selected_paths.html`
2. Click palette icon (🎨)
3. Try different bases:
   - **Base e:** Starting point
   - **Base 2:** If differences too subtle
   - **Base 10:** If weights span 3+ orders of magnitude
   - **Custom (5):** For more spread
   - **Custom (20):** For more compression

### Step 3: Adjust width factor alongside base
- High base + high factor: Compressed but visible
- Low base + low factor: Spread but not overwhelming

### Step 4: Save optimal settings for future use
```python
# Once you find the best base through interactive exploration
vis = VisualizePath(
    'pathways.csv',
    edge_width_scale='log',
    edge_width_log_base=5,     # Your optimal base
    edge_width_factor=1.5       # Your optimal factor
)
```

## Parameter Reference

### `edge_width_log_base`

**Type:** `float`, `str`, or `None`

**Valid Values:**
- `None` - Natural log (base e) ← **DEFAULT**
- `'e'` - Natural log (base e ≈ 2.718)
- `2` - Binary log
- `10` - Common log
- `1.1 to 100` - Any custom base

**Invalid Values:**
- `≤ 1.0` - Will fallback to natural log with warning
- Non-numeric strings (except 'e') - Will cause error

**Examples:**
```python
edge_width_log_base=None   # Natural log (default)
edge_width_log_base='e'    # Natural log (explicit)
edge_width_log_base=2      # Binary log
edge_width_log_base=10     # Common log
edge_width_log_base=5.5    # Custom base 5.5
edge_width_log_base=1.5    # Custom base 1.5 (subtle compression)
edge_width_log_base=50     # Custom base 50 (extreme compression)
```

## Relationship to Edge Width Factor

Both `edge_width_log_base` and `edge_width_factor` affect edge widths, but differently:

| Parameter | What it Controls | Effect |
|-----------|-----------------|---------|
| `edge_width_log_base` | **Relative** differences between edges | How spread out edge widths are |
| `edge_width_factor` | **Absolute** thickness of all edges | Overall scale (thicker/thinner) |

**Example:**
```python
# Compress differences, but make all edges thicker
VisualizePath(
    'pathways.csv',
    edge_width_scale='log',
    edge_width_log_base=20,    # High base = compressed differences
    edge_width_factor=3.0       # 3x thicker = easier to see
)

# Spread differences, but make edges thinner
VisualizePath(
    'pathways.csv',
    edge_width_scale='log',
    edge_width_log_base=1.5,   # Low base = spread differences
    edge_width_factor=0.5       # 0.5x thinner = avoid clutter
)
```

## Testing

Run the test to compare different log bases:

```bash
python test_log_base.py
```

This generates networks with bases e, 2, and 10, using weights from 10 to 10,000.

**Compare visually:**
```
test_output/log_base_test/
├── base_e/network_selected_paths.html    ← Moderate spread
├── base_2/network_selected_paths.html    ← More spread
└── base_10/network_selected_paths.html   ← Least spread
```

## Troubleshooting

### All edges still look the same

**Possible causes:**
1. Log base too high (e.g., 100) - over-compressing
2. All weights very similar (e.g., 95-105)
3. Width factor too low

**Solutions:**
- Try lower log base (2 instead of 10)
- Use linear scaling to see raw proportions
- Increase width factor to 2.0+

### Edges too thick/thin

**Solution:** Adjust `edge_width_factor` independently of log base

### Custom base not working

**Check:**
- Base value > 1.0
- Scaling method set to 'log'
- Selected "Custom..." from dropdown
- Entered valid number in custom input

## Summary

Adjustable log base provides:
- ✅ **Fine-grained control** over edge width compression
- ✅ **Interactive exploration** - change base in browser without regenerating
- ✅ **Mathematical flexibility** - any base > 1
- ✅ **Common presets** - e, 2, 10 for typical use cases
- ✅ **Custom bases** - tailor to specific data distributions

**Quick Guide:**
- **More differences visible:** Lower base (2, e)
- **Less differences visible:** Higher base (10, 20+)
- **Orders of magnitude:** Base 10
- **General use:** Base e (default)

This feature makes logarithmic edge width scaling adaptable to any weight distribution!
