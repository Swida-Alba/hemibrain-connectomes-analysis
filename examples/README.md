# Examples Directory

This directory contains example scripts demonstrating the key features of the Hemibrain Connectomes Analysis toolkit.

## 📋 Quick Start Examples

### 1. Caching System
**File:** `Example_CachingDemo.py`

Demonstrates the local caching feature that provides 10-100x speedup for repeated analyses.

```bash
python examples/Example_CachingDemo.py
```

**Features shown:**
- Enabling/disabling cache
- Performance comparison (first run vs. cached run)
- Cache management tips
- Cache location and file structure

**Related documentation:** [docs/CacheSystem_Guide.md](../docs/CacheSystem_Guide.md)

---

### 2. Parallel Processing
**File:** `Example_ParallelProcessing.py`

Shows how to use multi-core parallel processing for 4-14x speedup on large datasets.

```bash
python examples/Example_ParallelProcessing.py
```

**Features shown:**
- Basic parallel processing setup
- Comparing sequential vs. parallel performance
- Configuring number of CPU cores
- Automatic optimization for different dataset sizes

**Related documentation:** [docs/ParallelProcessing_Documentation.md](../docs/ParallelProcessing_Documentation.md)

---

### 3. Simple Edge-List Visualization
**File:** `Example_SimpleEdgeList.py`

Demonstrates how to visualize any network data with just 3 columns (source, target, weight).

```bash
python examples/Example_SimpleEdgeList.py
```

**Features shown:**
- Multiple input formats (source/target, from/to, pre/post, bodyId_pre/post)
- Loading from CSV/Excel files
- Auto-detecting column names
- Creating interactive networks and Sankey diagrams

**Sample data included:**
- `example_simple_network.csv`
- `example_bodyid_network.csv`
- `example_neuron_network.csv`

**Related documentation:** [docs/SIMPLE_FORMAT_IMPLEMENTATION.md](../docs/SIMPLE_FORMAT_IMPLEMENTATION.md)

---

### 4. File Picker for Visualizations
**File:** `Example_FilePicker.py`

Shows how to use the interactive file picker to load pathway data for visualization.

```bash
python examples/Example_FilePicker.py
```

**Features shown:**
- Opening file picker dialog
- Loading from different file formats
- Sheet selection for multi-sheet Excel files
- GUI backend options (PyQt5/PyQt6/wxPython/tkinter)

**Related documentation:** [docs/DIALOG_PERFORMANCE_GUIDE.md](../docs/DIALOG_PERFORMANCE_GUIDE.md)

---

### 5. Visualizing Selected Pathways
**File:** `Example_VisualizeSelectedPaths.py`

Comprehensive example of creating publication-ready pathway visualizations.

```bash
python examples/Example_VisualizeSelectedPaths.py
```

**Features shown:**
- Standalone VisualizePath usage (no NeuPrint connection needed)
- Loading pathway data from FindAllPath results
- Filtering specific paths
- Custom colors and layouts
- Creating both network and Sankey visualizations
- Interactive customization options

**Related documentation:** [docs/VisualizeSelectedPaths_Guide.md](../docs/VisualizeSelectedPaths_Guide.md)

---

## 🎯 Usage Pattern

All example scripts follow the same pattern:

1. **Import required modules:**
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
   
   from coana import FindNeuronConnection
   from vispath import VisualizePath
   ```

2. **Set up parameters:**
   ```python
   fc = FindNeuronConnection(
       token='your_token_here',
       dataset='hemibrain:v1.2.1',
       sourceNeurons=['KC.*'],
       targetNeurons=['MBON.*'],
       use_cache=True,
       use_parallel=True
   )
   ```

3. **Run analysis:**
   ```python
   fc.InitializeNeuronInfo()
   fc.FindAllPath()
   ```

4. **Visualize results:**
   ```python
   vis = VisualizePath(path_file='results.xlsx')
   vis.create_network()
   vis.create_sankey()
   ```

## 📦 Sample Data

The directory includes several sample CSV files for testing:

- **`example_simple_network.csv`** - Basic 3-column edge list
- **`example_bodyid_network.csv`** - BodyId format example
- **`example_neuron_network.csv`** - Neuron type format example
- **`example_paths.csv`** - Multi-hop pathway example

## 🔗 Related Resources

### Main Documentation
- [README.md](../README.md) - Main project documentation
- [docs/](../docs/) - Complete documentation library

### Quick Start Guides
- [Installation Guide](../docs/INSTALLATION.md)
- [Quick Start After Reorganization](../docs/QUICK_START_AFTER_REORGANIZATION.md)

### Core Features
- [Cache System Guide](../docs/CacheSystem_Guide.md)
- [Parallel Processing](../docs/ParallelProcessing_Documentation.md)
- [Path Finding](../docs/FindAllPath_Documentation.md)
- [Visualization](../docs/VisualizeSelectedPaths_Guide.md)

### Configuration
- [Connection Filters](../docs/ConnectionRatio_Filter.md)
- [Folder Naming Convention](../docs/FolderNaming_Convention.md)
- [Custom Colors Guide](../docs/CUSTOM_COLORS_GUIDE.md)

## 💡 Tips

1. **Get a NeuPrint Token:** Visit [https://neuprint.janelia.org/account](https://neuprint.janelia.org/account)
2. **Start with small datasets:** Use limited source/target neurons for testing
3. **Enable caching:** Significant speedup for repeated analyses
4. **Use parallel processing:** For datasets with many neuron pairs
5. **Check documentation:** Each example references relevant detailed guides

## 🆘 Troubleshooting

**Import errors?**
- Make sure you're in the project root when running examples
- Check that all dependencies are installed: `pip install -r requirements.txt`

**Slow performance?**
- Enable caching: `use_cache=True`
- Use parallel processing: `use_parallel=True`
- Reduce dataset size for testing

**File picker not working?**
- Install PyQt5 for best performance: `pip install PyQt5`
- See [DIALOG_PERFORMANCE_GUIDE.md](../docs/DIALOG_PERFORMANCE_GUIDE.md)

---

**Need more help?** Check the [main documentation](../README.md) or see detailed guides in [docs/](../docs/).
