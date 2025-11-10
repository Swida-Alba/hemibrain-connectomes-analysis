# Enhanced Edge-List Format - Quick Reference

## Minimum Requirements

**3 columns**: Source + Target + Weight

```python
df = pd.DataFrame({
    'source': ['A', 'B', 'C'],
    'target': ['B', 'C', 'D'],
    'weight': [10, 15, 20]
})
```

## Supported Column Names

### Source Columns
- `source` ✓
- `from` ✓
- `pre` ✓
- `*_pre` ✓ (e.g., `bodyId_pre`, `type_pre`, `neuron_pre`)

### Target Columns
- `target` ✓
- `to` ✓
- `post` ✓
- `*_post` ✓ (e.g., `bodyId_post`, `type_post`, `neuron_post`)

### Weight Columns
- `weight` ✓
- `weights` ✓
- `synapse_count` ✓
- `count` ✓

## Automatic Metric Detection

All numeric columns (except source, target, color) are detected as metrics.

### Standard Metrics (Auto-mapped for Toggle Support)

| Input Column | Maps To | Toggle? |
|-------------|---------|---------|
| `ratio` | `connection_ratios` | ✓ |
| `connection_ratio` | `connection_ratios` | ✓ |
| `probability` | `traversal_probabilities` | ✓ |
| `prob` | `traversal_probabilities` | ✓ |
| `trav_prob` | `traversal_probabilities` | ✓ |
| `traversal_probability` | `traversal_probabilities` | ✓ |

### Custom Metrics (Preserved for Export)

Any other numeric column:
- `strength` → Custom metric (no toggle, but preserved)
- `confidence` → Custom metric (no toggle, but preserved)
- `reliability` → Custom metric (no toggle, but preserved)

## Input Format Options

### 1. DataFrame
```python
vis = VisualizePath(path_file=dataframe, output_folder='./output')
```

### 2. CSV File
```python
vis = VisualizePath(path_file='edges.csv', output_folder='./output')
```

### 3. Excel File
```python
vis = VisualizePath(path_file='edges.xlsx', output_folder='./output')
```

## Format Examples

### Basic Formats

```python
# Format 1: source/target/weight
{'source': [...], 'target': [...], 'weight': [...]}

# Format 2: from/to/weight
{'from': [...], 'to': [...], 'weight': [...]}

# Format 3: pre/post/weight
{'pre': [...], 'post': [...], 'weight': [...]}

# Format 4: bodyId_pre/bodyId_post/weight
{'bodyId_pre': [...], 'bodyId_post': [...], 'weight': [...]}

# Format 5: neuron_pre/neuron_post/synapse_count
{'neuron_pre': [...], 'neuron_post': [...], 'synapse_count': [...]}
```

### With Standard Metrics

```python
# With ratio (toggleable in visualizations)
{
    'source': ['A', 'B'],
    'target': ['B', 'C'],
    'weight': [10, 15],
    'ratio': [0.5, 0.7]  # Auto-mapped to connection_ratios
}

# With probability (toggleable in visualizations)
{
    'source': ['A', 'B'],
    'target': ['B', 'C'],
    'weight': [10, 15],
    'probability': [0.9, 0.85]  # Auto-mapped to traversal_probabilities
}

# With both
{
    'source': ['A', 'B'],
    'target': ['B', 'C'],
    'weight': [10, 15],
    'ratio': [0.5, 0.7],
    'probability': [0.9, 0.85]
}
```

### With Custom Metrics

```python
# Custom metrics preserved for export
{
    'source': ['A', 'B'],
    'target': ['B', 'C'],
    'weight': [10, 15],
    'strength': [0.85, 0.72],      # Custom metric
    'confidence': [0.95, 0.88],    # Custom metric
    'reliability': [0.87, 0.82]    # Custom metric
}
```

## Detection Output

When loading data, you'll see:

```
Path-based format not detected, checking for edge-list format...
✓ Detected edge-list format
  Source column: 'source'
  Target column: 'target'
  Weight column: 'weight'
  Converting 5 edges to path format...
  Detected numeric columns: ['weight', 'ratio', 'probability']
  ✓ Mapped 'ratio' → 'connection_ratios' for toggle support
  ✓ Mapped 'probability' → 'traversal_probabilities' for toggle support
✓ Converted to 5 paths
```

## Common Use Cases

### Use Case 1: Simple Network Visualization
```python
df = pd.DataFrame({
    'source': ['A', 'B', 'C'],
    'target': ['B', 'C', 'D'],
    'weight': [10, 15, 20]
})
vis = VisualizePath(path_file=df, output_folder='./output')
vis.create_network()
```

### Use Case 2: With Connection Ratios
```python
df = pd.DataFrame({
    'pre': ['KC', 'KC', 'MBON'],
    'post': ['MBON', 'DAN', 'DAN'],
    'weight': [100, 80, 60],
    'ratio': [0.67, 0.57, 1.0]
})
vis = VisualizePath(path_file=df, output_folder='./output')
vis.create_network()  # Toggle ratio on/off in browser
```

### Use Case 3: Full Pipeline with Metrics
```python
df = pd.DataFrame({
    'bodyId_pre': [123, 234, 345],
    'bodyId_post': [234, 345, 456],
    'synapse_count': [150, 120, 100],
    'probability': [0.95, 0.88, 0.92],
    'strength': [0.85, 0.82, 0.91]
})
vis = VisualizePath(path_file=df, output_folder='./output')
vis.create_heatmap()
vis.create_sankey()
vis.create_network()
```

### Use Case 4: Load from File
```python
# CSV file with headers: source, target, weight, ratio
vis = VisualizePath(path_file='my_edges.csv', output_folder='./output')
vis.visualize()  # Creates all three visualizations
```

## Testing

### Quick Test
```python
# Create test data
test_data = {
    'source': ['A', 'B'],
    'target': ['B', 'C'],
    'weight': [10, 15],
    'ratio': [0.5, 0.7]
}

# Visualize
from vispath import VisualizePath
vis = VisualizePath(
    path_file=pd.DataFrame(test_data),
    output_folder='./test_output'
)
vis.create_network()
```

### Run Full Test Suite
```bash
python tests/test_enhanced_edgelist.py
```

### Run Examples
```bash
python examples/Example_SimpleEdgeList.py
```

## Troubleshooting

### Issue: Column Not Detected
**Problem**: "Could not find source/target column"

**Solution**: Check column names match supported formats:
- Source: `source`, `from`, `pre`, or `*_pre`
- Target: `target`, `to`, `post`, or `*_post`

### Issue: Metric Not Toggleable
**Problem**: Custom metric doesn't have toggle control

**Solution**: Only standard metrics get toggles:
- `ratio` → toggleable
- `probability` → toggleable
- Custom metrics → preserved for export only

### Issue: Weight Column Not Found
**Problem**: "Could not find weight column"

**Solution**: Rename to: `weight`, `weights`, `synapse_count`, or `count`

## Tips

1. **Use consistent naming**: Stick to one convention (e.g., always use `*_pre`/`*_post`)
2. **Include standard metrics**: Use `ratio` and `probability` for toggle support
3. **Custom metrics**: Any numeric column is preserved in exports
4. **File formats**: CSV and Excel work identically - choose based on your workflow
5. **Testing**: Start with small datasets to verify format detection

## See Also

- [Enhanced_EdgeList_Format.md](Enhanced_EdgeList_Format.md) - Complete guide
- [examples/Example_SimpleEdgeList.py](../examples/Example_SimpleEdgeList.py) - Working examples
- [tests/test_enhanced_edgelist.py](../tests/test_enhanced_edgelist.py) - Test suite
- [CacheSystem_QuickStart.md](CacheSystem_QuickStart.md) - Getting started
