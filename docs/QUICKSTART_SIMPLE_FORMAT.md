# Quick Start: Simple Network Visualization

## TL;DR - Get Started in 30 Seconds

```python
from vispath import VisualizePath
import pandas as pd

# Create your network data (just 3 columns!)
df = pd.DataFrame({
    'source': ['A', 'B', 'C'],
    'target': ['B', 'C', 'D'],
    'weight': [10, 20, 15]
})

# Visualize it
vis = VisualizePath(path_file=df, output_folder='./output')
vis.create_network()  # Interactive network graph
vis.create_sankey()   # Layered flow diagram
```

**Done!** Open the HTML files in `./output/` to see your visualizations.

---

## What You Need

### Minimum Input: Just 3 Columns

| Column | What it means | Example values |
|--------|---------------|----------------|
| **source** | Starting node | 'A', 'neuron1', 123456 |
| **target** | Ending node | 'B', 'neuron2', 234567 |
| **weight** | Connection strength | 10, 50, 100 |

### Column Names - We're Flexible!

**Source column** can be named:
- `source`, `from`, `pre`
- `bodyId_pre`, `neuron_pre`, `type_pre` (anything ending in `_pre`)

**Target column** can be named:
- `target`, `to`, `post`
- `bodyId_post`, `neuron_post`, `type_post` (anything ending in `_post`)

**Weight column** can be named:
- `weight`, `weights`, `synapse_count`, `count`

---

## Quick Examples

### Example 1: Simple CSV File

**Create `network.csv`:**
```csv
source,target,weight
A,B,10
B,C,20
C,D,15
```

**Visualize:**
```python
from vispath import VisualizePath

vis = VisualizePath(path_file='network.csv')
vis.create_network()
vis.create_sankey()
```

### Example 2: Neuroscience Data

**Your data:**
```csv
bodyId_pre,bodyId_post,weight
123456,234567,25
234567,345678,30
345678,456789,20
```

**Code:**
```python
from vispath import VisualizePath

vis = VisualizePath(path_file='neurons.csv')
vis.create_network()
```

### Example 3: Custom Colors

```python
from vispath import VisualizePath

vis = VisualizePath(
    path_file='network.csv',
    source_color='#3498db',           # Blue
    target_color='#e74c3c',           # Red
    intermediate_color='#2ecc71',     # Green
    link_color='rgba(100,100,100,0.3)'  # Semi-transparent
)

vis.create_network()
vis.create_sankey()
```

### Example 4: From DataFrame

```python
from vispath import VisualizePath
import pandas as pd

# Create data directly
edges = pd.DataFrame({
    'from': ['KC_a', 'KC_b', 'KC_c'],
    'to': ['MBON_a', 'MBON_b', 'MBON_a'],
    'weight': [100, 80, 120]
})

# Visualize
vis = VisualizePath(path_file=edges)
vis.create_network()
```

---

## What You Get

### 1. Interactive Network Graph
**File:** `network_selected_paths.html`

Features:
- ✨ Drag nodes to rearrange
- 🔍 Zoom and pan
- 📊 Layered layout (sources → intermediates → targets)
- 🎨 Color-coded by node type
- 💡 Hover to see details

### 2. Sankey Flow Diagram
**File:** `sankey_selected_paths.html`

Features:
- 🌊 Flow visualization showing connection weights
- 🎛️ Interactive controls (color, opacity, visibility)
- 📈 Multi-layer support
- 🎨 Customizable colors
- 📊 Proportional link widths

---

## Common Use Cases

### Neuroscience Networks
```python
# Synaptic connections
df = pd.DataFrame({
    'neuron_pre': ['DA1_PN', 'DA1_PN', 'VA1d_PN'],
    'neuron_post': ['LHON1', 'LHON2', 'LHON1'],
    'synapse_count': [150, 120, 180]
})

vis = VisualizePath(path_file=df)
vis.create_network()
```

### Social Networks
```python
# Connections between people
df = pd.DataFrame({
    'from': ['Alice', 'Bob', 'Charlie'],
    'to': ['Bob', 'Charlie', 'Alice'],
    'weight': [10, 15, 8]
})

vis = VisualizePath(path_file=df)
vis.create_sankey()
```

### Citation Networks
```python
# Paper citations
df = pd.DataFrame({
    'source': ['Paper_A', 'Paper_B', 'Paper_C'],
    'target': ['Paper_D', 'Paper_D', 'Paper_E'],
    'weight': [5, 3, 7]
})

vis = VisualizePath(path_file=df)
vis.create_network()
```

---

## Installation

```bash
pip install -r requirements.txt
```

That's it! All dependencies will be installed.

---

## Need Help?

### Quick Diagnostics

**Test your installation:**
```bash
python test_simple_format.py
```

**Run comprehensive examples:**
```bash
python Example_SimpleEdgeList.py
```

### Common Issues

**Q: "Column not found" error**
- Check your column names match one of the supported variants
- Try renaming: `df.rename(columns={'my_col': 'source'}, inplace=True)`

**Q: "No module named 'vispath'"**
- Make sure you're in the correct directory
- Install requirements: `pip install -r requirements.txt`

**Q: Excel file not working**
- Make sure you have openpyxl installed: `pip install openpyxl`
- Check the file path is correct

### Documentation

- 📖 **[Simple Input Format Guide](SIMPLE_INPUT_FORMAT.md)** - Complete reference
- 📖 **[Implementation Details](SIMPLE_FORMAT_IMPLEMENTATION.md)** - Technical details
- 📖 **[Full README](README.md)** - Complete project documentation

---

## Advanced Features

All the power of VisualizePath is available with simple input:

```python
vis = VisualizePath(
    path_file='network.csv',
    output_folder='./results',
    
    # Colors
    source_color='rgba(52, 152, 219, 0.8)',
    target_color='rgba(231, 76, 60, 0.8)',
    intermediate_color='rgba(46, 204, 113, 0.8)',
    link_color='rgba(100, 100, 100, 0.4)',
    
    # Sizes
    node_size=30,
    font_size=12,
    
    # And many more options...
)

vis.create_network()
vis.create_sankey()
```

---

## What's Next?

1. ✅ Load your data (CSV, Excel, or DataFrame)
2. ✅ Create visualizations (network and/or Sankey)
3. ✅ Open HTML files in browser
4. ✅ Customize colors and styles as needed

**That's it!** You're ready to visualize networks with just 3 columns of data.

---

## Examples Included

| File | Description |
|------|-------------|
| `Example_SimpleEdgeList.py` | Comprehensive examples (5 formats) |
| `example_simple_network.csv` | Basic source/target/weight |
| `example_bodyid_network.csv` | BodyId format |
| `example_neuron_network.csv` | Neuron format with synapse counts |
| `test_simple_format.py` | Quick validation test |

Run any of these to see the system in action!

---

**Questions?** See [SIMPLE_INPUT_FORMAT.md](SIMPLE_INPUT_FORMAT.md) for complete documentation.

**Happy Visualizing! 🎉**
