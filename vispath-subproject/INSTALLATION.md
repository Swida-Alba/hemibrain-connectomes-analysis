# VisualizePath Sub-Project Installation Guide

## Overview

The `vispath-subproject` directory contains a standalone installable version of the VisualizePath visualization toolkit. This allows users to install just the visualization component without the full hemibrain connectomes analysis suite.

## Installation Options

### Recommended: Using Conda Environment

**Create a new conda environment for the standalone package:**

```bash
# Create environment with Python 3.11
conda create -n vispath python=3.11 -y

# Activate the environment
conda activate vispath

# Install the package
cd vispath-subproject
pip install -e .
```

### Option 1: Standalone Installation (Minimal Dependencies)

Install only the visualization toolkit:

```bash
# From the vispath-subproject directory
cd vispath-subproject
pip install -e .

# Or directly from GitHub
pip install git+https://github.com/Swida-Alba/hemibrain-connectomes-analysis.git#subdirectory=vispath-subproject
```

**Dependencies (minimal):**
- numpy >=1.20.0, <2.0.0 (constrained for pandas compatibility)
- pandas >=1.3.0, <2.0.0
- scipy >=1.7.0
- plotly >=5.0.0
- networkx >=2.6.0
- openpyxl >=3.0.0
- PyQt5 >=5.15.0 (optional, for GUI dialogs)

### Option 2: Full Installation (Recommended)

Install the complete hemibrain-connectomes-analysis package, which includes VisualizePath:

```bash
# From the root directory
cd ..
pip install -e .

# Or
pip install -e ".[vispath]"  # Install with vispath dependencies highlighted
```

**Dependencies (full):**
- All visualization dependencies above, plus:
- neuprint-python, navis, flybrains
- matplotlib, seaborn, bokeh, opencv-python
- img2pdf

## Project Structure

```
vispath-subproject/
├── README.md              # This file
├── pyproject.toml         # Standalone package configuration
├── setup.py               # Standalone installation script
├── requirements.txt       # Minimal dependencies
└── src/
    └── vispath_pkg/
        ├── __init__.py    # Package initialization
        └── vispath.py     # Symlink to ../../../src/vispath.py
```

## How It Works

The sub-project uses a **symbolic link** to share the main `vispath.py` file from the parent project. This ensures:

1. **Single source of truth** - Code is maintained in one place
2. **No duplication** - Same code for both standalone and full installation
3. **Automatic updates** - Changes to the main vispath.py are immediately available

## Limitations of Standalone Installation

The standalone installation is **fully functional** for all visualization features:

✅ **All features available:**
- Interactive Sankey diagrams with edge filtering
- Network graphs with multiple layout algorithms
- Connection heatmaps with hierarchical clustering
- Metric toggle (weight/ratio/probability)
- Custom node and edge coloring
- Export to SVG/PNG

The only features NOT available in standalone mode are:
- **NeuPrint database connectivity** - requires neuprint-python (part of full package)
- **3D neuron skeleton rendering** - requires navis/flybrains (part of full package)
- **Automated pathfinding algorithms** - requires full connectome analysis suite

## Development

### Making Changes

Since `vispath.py` is symlinked, changes should be made to the parent file:

```bash
# Edit the main file
vim ../../../src/vispath.py

# Changes are immediately available in both locations
```

### Testing Standalone Installation

```bash
cd vispath-subproject

# Create conda environment (recommended)
conda create -n vispath_test python=3.11 -y
conda activate vispath_test

# Install in development mode
pip install -e .

# Test import
python -c "from vispath_pkg.vispath import VisualizePath; print('✓ vispath-subproject works!')"
```

### Testing Full Installation

```bash
cd ..

# Use existing conda environment or create new one
conda create -n hemibrain python=3.11 -y
conda activate hemibrain

# Install in development mode
pip install -e .

# Test import
python -c "from vispath import VisualizePath; print('✓ Full package works!')"
```

## Package Distribution

### Publishing Standalone Package

The standalone package can be published to PyPI separately:

```bash
cd vispath-subproject
python -m build
twine upload dist/*
```

### Including in Main Distribution

The main package automatically includes vispath when installed:

```bash
cd ..
python -m build
twine upload dist/*
```

## Usage Examples

### Input Data Formats

VisualizePath accepts two data formats:

#### Path-based format (multi-hop paths):
```python
import pandas as pd

# Create path data
paths = pd.DataFrame({
    'path_block': ['A -> B -> C', 'A -> D -> C'],
    'weights': [[10, 5], [15, 8]],
    'connection_ratios': [[0.5, 0.3], [0.6, 0.4]],  # Optional
    'traversal_probabilities': [[0.8, 0.6], [0.9, 0.7]]  # Optional
})
```

#### Edge-list format (direct connections):
```python
# Create edge data (automatically converted to paths)
edges = pd.DataFrame({
    'source': ['A', 'B', 'D'],
    'target': ['B', 'C', 'C'],
    'weight': [10, 5, 8],
    'ratio': [0.5, 0.3, 0.4],  # Optional
    'probability': [0.8, 0.6, 0.7]  # Optional
})
```

See [README.md](README.md) for complete format specifications.

### Standalone Usage

```python
from vispath_pkg import VisualizePath

vp = VisualizePath(
    path_file='paths.xlsx',
    output_folder='./output'
)
vp.visualize_all()
```

### Full Package Usage

```python
from vispath import VisualizePath

# Same API as standalone, but with additional features
vp = VisualizePath(
    path_file='paths.xlsx',
    output_folder='./output'
)
vp.visualize_all()
```

## Maintenance

### Updating Dependencies

Dependencies are maintained in:
- `requirements.txt` - Standalone minimal deps
- `pyproject.toml` - Standalone package config
- `../pyproject.toml` - Main package config (includes `[project.optional-dependencies]` for vispath)

To update:

1. Update `requirements.txt` for standalone
2. Update `pyproject.toml` dependencies for standalone
3. Update `../pyproject.toml` optional dependencies `[vispath]` section

### Synchronizing Versions

The standalone package version should match the parent project version:

```toml
# vispath-subproject/pyproject.toml
version = "1.0.0"

# ../pyproject.toml
version = "3.0.0"  # Can be different, but document relationship
```

## Troubleshooting

### numpy/pandas Binary Compatibility Error

If you see an error like `numpy.dtype size changed, may indicate binary incompatibility`:

```bash
# Ensure numpy <2.0.0 is installed
pip install 'numpy<2.0.0' --force-reinstall
```

This is already handled in requirements.txt and pyproject.toml with the constraint `numpy>=1.20.0,<2.0.0`.

### Symlink Issues on Windows

If symlinks don't work on Windows:

1. Enable Developer Mode in Windows Settings
2. Or copy `vispath.py` instead of symlinking
3. Use `mklink` instead of `ln -sf`

### Import Errors

If you see import errors about `statvis`:

```python
# This is expected in standalone mode
# Install the full package for complete functionality
pip install hemibrain-connectomes-analysis
```

### Module Not Found

If Python can't find the module:

```bash
# Reinstall in editable mode
pip install -e .

# Or check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
```

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/Swida-Alba/hemibrain-connectomes-analysis/issues
- Main Documentation: ../README.md

## License

MIT License - Same as parent project
