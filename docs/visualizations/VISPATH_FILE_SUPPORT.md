# VisualizePath - File Format Support & File Picker

## Overview

The `VisualizePath` class now supports multiple input methods:
- **Excel files (.xlsx, .xls)** with automatic or interactive sheet selection
- **CSV files (.csv)** with direct loading
- **Interactive file picker** when file path is missing or invalid
- **Direct DataFrame** input for programmatic use

---

## File Format Support

### Excel Files (.xlsx, .xls)

**Features:**
- Automatic sheet detection (looks for 'path_type', 'path_bodyId', 'path_block', 'paths')
- Interactive sheet selection when multiple sheets exist
- Manual sheet specification via `sheet_name` parameter

**Usage:**
```python
from vispath import VisualizePath

# Auto-detect sheet
vp = VisualizePath(
    path_file='my_paths.xlsx',
    sheet_name=None  # Auto-detect or prompt
)

# Specify sheet explicitly
vp = VisualizePath(
    path_file='my_paths.xlsx',
    sheet_name='path_type'
)
```

**Sheet Selection Priority:**
1. If `sheet_name` is specified → Use that sheet
2. If only 1 sheet exists → Use it automatically
3. If common sheet names found ('path_type', 'path_bodyId', etc.) → Auto-select
4. Otherwise → Interactive prompt with sheet info (rows, columns)

**Interactive Sheet Selection Example:**
```
================================================================
Multiple sheets found. Please select one:
================================================================
  [1] connection_info            (1234 rows, 15 cols)
  [2] connection_type            (567 rows, 12 cols)
  [3] path_type                  (89 rows, 8 cols)
  [4] path_bodyId                (234 rows, 10 cols)
================================================================
Enter number (1-4) or sheet name: 3
✓ Selected: 'path_type'
```

---

### CSV Files (.csv)

**Features:**
- Direct loading without sheet selection
- `sheet_name` parameter is automatically ignored
- Faster loading for simple datasets

**Usage:**
```python
vp = VisualizePath(
    path_file='my_paths.csv',
    # sheet_name is ignored for CSV files
)
```

**Note:** CSV files must have the same column structure as Excel sheets:
- Required: `path_block`, `weights`
- Optional: `connection_ratios`, `traversal_probabilities`

---

## Interactive File Picker

### When File Picker Opens

The file picker dialog opens automatically when:
1. `path_file=None` (explicitly request file picker)
2. `path_file=''` (empty string)
3. File path doesn't exist

**Example:**
```python
# Open file picker
vp = VisualizePath(
    path_file=None,  # Opens file picker dialog
    showfig=True
)
```

**File Picker Features:**
- Browse your filesystem with native OS dialog
- Filter by file type (Excel, CSV, or All files)
- Preview file information before selection
- Cross-platform support (macOS, Windows, Linux)

### File Picker Workflow

```
⚠️ Path file not found: /nonexistent/path.xlsx
Please select a path file...
[File picker dialog opens]
✓ Selected file: /Users/data/my_paths.xlsx
Loading Excel file: /Users/data/my_paths.xlsx
Auto-selected sheet: 'path_type'
  Loaded sheet: 'path_type'
Loaded 5 pathways from data
```

---

## DataFrame Input

### Direct DataFrame Usage

For programmatic workflows, you can pass a DataFrame directly:

```python
import pandas as pd
from vispath import VisualizePath

# Create or filter path data
path_data = pd.DataFrame({
    'path_block': ['A -> B -> C', 'A -> D -> C'],
    'weights': [[10, 20], [15, 25]]
})

# Visualize directly
vp = VisualizePath(
    path_file=path_data,  # DataFrame instead of file path
    output_folder='./my_output',
    showfig=True
)

conn_df, G = vp.visualize()
```

**Benefits:**
- No file I/O required
- Easy filtering and preprocessing
- Integration with data pipelines

---

## Complete Examples

### Example 1: File Picker with Excel

```python
from vispath import VisualizePath

# Open file picker, then select sheet interactively
vp = VisualizePath(
    path_file=None,              # Triggers file picker
    sheet_name=None,              # Triggers sheet selection
    source_color='#1f77b4',
    intermediate_color='#2ca02c',
    target_color='#d62728',
    showfig=True
)

conn_df, G = vp.visualize()
```

### Example 2: CSV with Custom Colors

```python
from vispath import VisualizePath

vp = VisualizePath(
    path_file='my_paths.csv',
    source_color='#FF6B6B',
    intermediate_color='#FFA500',
    target_color='#FFD700',
    link_color='rgba(255,107,107,0.3)',
    output_folder='./csv_output',
    showfig=True
)

conn_df, G = vp.visualize()
```

### Example 3: Filtered Paths from Excel

```python
import pandas as pd
from vispath import VisualizePath

# Read and filter paths
all_paths = pd.read_excel('results.xlsx', sheet_name='path_type')

# Filter high-quality paths
high_quality = all_paths[
    (all_paths['traversal_probability'] > 0.5) &
    (all_paths['inter_layer_num'] <= 2)
]

# Visualize filtered subset
vp = VisualizePath(
    path_file=high_quality,      # Pass filtered DataFrame
    output_folder='./high_quality',
    showfig=True
)

conn_df, G = vp.visualize()
```

### Example 4: Auto-detect Everything

```python
from vispath import VisualizePath

# Minimal code - auto-detect file format and sheets
vp = VisualizePath(
    path_file='my_data.xlsx',    # Or .csv
    # sheet_name=None by default
    showfig=True
)

conn_df, G = vp.visualize()
```

---

## Required Data Format

### Minimum Required Columns

Both CSV and Excel files must contain:

| Column | Type | Description |
|--------|------|-------------|
| `path_block` | str | Path in format "A -> B -> C -> D" |
| `weights` | list | List of synapse counts [w1, w2, w3] |

### Optional Columns

| Column | Type | Description |
|--------|------|-------------|
| `connection_ratios` | list | List of connection ratios [r1, r2, r3] |
| `traversal_probabilities` | list | List of probabilities [p1, p2, p3] |

### Example Data Structure

**CSV Format:**
```csv
path_block,weights,connection_ratios,traversal_probabilities
"A -> B -> C","[100, 50]","[0.5, 0.3]","[0.8, 0.6]"
"A -> D -> C","[80, 60]","[0.4, 0.35]","[0.7, 0.65]"
```

**Excel Format:**
Same structure, but can have multiple sheets for different analyses.

---

## Error Handling

### File Not Found
```
⚠️ Path file not found: /path/to/file.xlsx
Please select a path file...
[Opens file picker]
```

### No Sheet Selected
```
✗ No sheet selected
ValueError: No sheet selected. Cannot proceed without sheet selection.
```

### Missing Columns
```
ValueError: Missing required columns: ['weights']
Available columns: ['path_block', 'connections']
Required columns: ['path_block', 'weights']
```

### Unsupported Format
```
ValueError: Unsupported file format: .txt. Use .csv, .xlsx, or .xls
```

---

## Tips & Best Practices

### Performance

- **CSV files** load faster than Excel for large datasets
- Use **DataFrame input** for repeated analyses (avoid file I/O)
- Filter paths before visualization to reduce processing time

### Sheet Selection

- Name sheets descriptively ('path_type', 'path_bodyId', etc.) for auto-detection
- Use `sheet_name` parameter to skip interactive selection
- Put most important data in first sheet for default selection

### File Organization

```
my_project/
├── data/
│   ├── raw_paths.xlsx           # Original FindAllPath output
│   ├── filtered_paths.csv       # Filtered subset for quick loading
│   └── custom_paths.csv         # Manual path analysis
└── visualizations/
    ├── all_paths/               # Auto-created output folders
    ├── high_quality/
    └── custom_analysis/
```

### Automation

```python
# Batch process multiple files
import glob
from vispath import VisualizePath

for csv_file in glob.glob('data/*.csv'):
    print(f"Processing {csv_file}...")
    vp = VisualizePath(
        path_file=csv_file,
        output_folder=f'output/{Path(csv_file).stem}',
        showfig=False  # Don't open browser for batch processing
    )
    vp.visualize()
```

---

## Troubleshooting

### File Picker Doesn't Open
- **Issue:** `tkinter` not installed
- **Solution:** File picker requires `tkinter` (usually included with Python)
- **Workaround:** Specify `path_file` explicitly

### Sheet Selection Hangs
- **Issue:** Running in non-interactive environment (Jupyter, remote server)
- **Solution:** Always specify `sheet_name` when running non-interactively

### CSV Column Parsing Issues
- **Issue:** List columns stored as strings
- **Solution:** Lists are automatically parsed using `ast.literal_eval()`
- **Format:** Use Python list syntax: `"[1, 2, 3]"` not `"1,2,3"`

---

## API Reference

### VisualizePath Constructor

```python
VisualizePath(
    path_file=None,                    # str, DataFrame, or None (opens picker)
    sheet_name=None,                   # str or None (auto-detect/prompt)
    output_folder=None,                # str or None (auto-create)
    source_color='#1f77b4',
    intermediate_color='#2ca02c',
    target_color='#d62728',
    link_color='rgba(100,100,100,0.3)',
    network_layout='hierarchical',
    showfig=True
)
```

### Key Methods

- `visualize()` → Generate all visualizations
- `build_network()` → Create network graph
- `create_sankey()` → Create Sankey diagram
- `create_network()` → Create interactive network

---

## See Also

- `PlotPath.py` - Main example script
- `Example_FilePicker.py` - File picker examples
- `Example_VisualizeSelectedPaths.py` - Advanced usage examples
