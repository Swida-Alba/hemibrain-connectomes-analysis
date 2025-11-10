# Installation Guide

## Quick Start

### Recommended: Using Conda Environment

**Create a new conda environment (recommended for isolation):**

```bash
# Create environment with Python 3.11
conda create -n hemibrain python=3.11 -y

# Activate the environment
conda activate hemibrain

# Install dependencies
pip install -r requirements.txt
```

**Or install the package in editable mode:**

```bash
conda activate hemibrain
pip install -e .
```

### Option 1: Using requirements.txt

```bash
pip install -r requirements.txt
```

This will install all necessary dependencies including PyQt5 for fast GUI dialogs.

### Option 2: Using pip install (Modern Python)

```bash
# Standard installation
pip install .

# Editable installation (for development)
pip install -e .
```

Or with optional dependencies:

```bash
# Install with development tools
pip install -e ".[dev]"

# Install with visualization extras
pip install -e ".[viz]"

# Install with all GUI backends
pip install -e ".[gui]"

# Install everything
pip install -e ".[all]"
```

### Option 3: vispath-subproject Only

If you only need the visualization path component:

```bash
cd vispath-subproject
pip install -e .
```

## Core Dependencies

The following packages are **required**:

| Package | Purpose | Version |
|---------|---------|---------|
| numpy | Numerical computing | >=1.20.0, <2.0.0 |
| pandas | Data manipulation | >=1.3.0, <2.0.0 |
| scipy | Scientific computing | >=1.7.0 |
| plotly | Interactive visualizations | >=5.0.0 |
| matplotlib | Static plots | >=3.4.0 |
| networkx | Network analysis | >=2.6.0 |
| openpyxl | Excel file support | >=3.0.0 |
| **PyQt5** | **Fast GUI dialogs** | **>=5.15.0** |
| neuprint-python | NeUPRINT database | >=0.4.0 |

**Important:** numpy is constrained to <2.0.0 for binary compatibility with pandas 1.x.

## Optional Dependencies

### For Better Performance

```bash
# Install PyQt6 as alternative to PyQt5
pip install PyQt6

# Install wxPython as another GUI option
pip install wxPython
```

### For 3D Visualization

```bash
pip install navis trimesh
```

### For Development

```bash
pip install pytest pytest-cov jupyter ipykernel
```

## GUI Backend Selection

The system automatically uses the **fastest available** GUI backend:

1. **PyQt5** ⚡⚡⚡ (Recommended - Included in requirements.txt)
2. **PyQt6** ⚡⚡⚡ (Alternative)
3. **wxPython** ⚡⚡ (Alternative)
4. **tkinter** ⚡ (Fallback - Built-in with Python)

**For best performance**, PyQt5 is included by default in `requirements.txt`.

## Platform-Specific Notes

### macOS

```bash
# All dependencies should install via pip
pip install -r requirements.txt
```

**Note:** tkinter is included with Python on macOS, but PyQt5 is much faster.

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies first
sudo apt-get update
sudo apt-get install python3-pip python3-dev python3-tk

# Then install Python packages
pip install -r requirements.txt
```

### Windows

```bash
# Install from requirements.txt
pip install -r requirements.txt
```

**Note:** Make sure Python was installed with tkinter support (should be default).

## Verification

Check your installation:

```bash
python -c "import numpy, pandas, plotly, networkx, PyQt5; print('✓ All core packages installed')"
```

Test GUI backend:

```bash
python -c "from PyQt5.QtWidgets import QApplication; print('✓ PyQt5 is working')"
```

## Troubleshooting

### numpy/pandas Binary Compatibility Error

If you see an error like `numpy.dtype size changed, may indicate binary incompatibility`:

```bash
# Ensure numpy <2.0.0 is installed
pip install 'numpy<2.0.0' --force-reinstall
```

This is already handled in requirements.txt and pyproject.toml.

### PyQt5 Installation Issues

**macOS:**
```bash
pip install --upgrade pip
pip install PyQt5
```

**Linux:**
```bash
sudo apt-get install python3-pyqt5
# or
pip install PyQt5
```

**Windows:**
```bash
pip install PyQt5
```

### Pandas Version Conflict

This project requires pandas < 2.0.0. If you have pandas 2.x:

```bash
pip install "pandas<2"
```

### Missing tkinter (Linux)

```bash
sudo apt-get install python3-tk
```

### Import Errors

If you get import errors, try reinstalling:

```bash
pip install --upgrade --force-reinstall -r requirements.txt
```

## Minimal Installation

For a minimal setup (without GUI file pickers):

```bash
pip install numpy pandas scipy plotly matplotlib networkx openpyxl neuprint-python
```

**Note:** File picker will fall back to terminal input without GUI libraries.

## Development Installation

For contributing to the project:

```bash
# Clone repository
git clone https://github.com/Swida-Alba/hemibrain-connectomes-analysis.git
cd hemibrain-connectomes-analysis

# Create conda environment (recommended)
conda create -n hemibrain python=3.11 -y
conda activate hemibrain

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

**Testing the installation:**

```bash
# Test vispath-subproject
cd vispath-subproject
pip install -e .
python -c "from vispath_pkg.vispath import VisualizePath; print('✓ vispath-subproject works')"
cd ..

# Test root package
pip install -e .
python -c "import coana, statvis, vispath; print('✓ Root package works')"
```

## Docker Installation (Optional)

If you prefer containerized setup:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
```

## Summary

**Recommended (with conda):**
```bash
conda create -n hemibrain python=3.11 -y
conda activate hemibrain
pip install -r requirements.txt
```

**Easiest (without conda):**
```bash
pip install -r requirements.txt
```

**Editable install (for development):**
```bash
pip install -e .
```

**With extras:**
```bash
pip install -e ".[all]"
```

All methods will install PyQt5 for optimal GUI performance and numpy <2.0.0 for compatibility! 🚀
