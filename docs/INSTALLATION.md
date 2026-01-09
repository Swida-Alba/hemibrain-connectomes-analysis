# Installation Guide

## Quick Start

### Recommended: Using Conda Environment

**Create a new conda environment (recommended for isolation):**

```bash
# Create environment with Python 3.11
conda create -n drocat python=3.11 -y

# Activate the environment
conda activate drocat
```

**Install dependencies (platform-specific):**

**Linux/macOS:**
```bash
pip install -r requirements.txt
```

**Windows:**
```bash
pip install -r requirements-windows.txt
pip install neuronbridge-python --no-deps
```

**Or install the package in editable mode:**

```bash
conda activate drocat
pip install -e .
```

### Option 1: Using requirements files

**Linux/macOS:**
```bash
pip install -r requirements.txt
```

**Windows:**
```bash
# Step 1: Install all dependencies from Windows-specific requirements
pip install -r requirements-windows.txt

# Step 2: Install neuronbridge-python without its dependencies
pip install neuronbridge-python --no-deps
```

This will install all necessary dependencies including PyQt5 for fast GUI dialogs.

> **Note for Windows users:** The `requirements-windows.txt` file is specifically designed to handle the `memray` incompatibility issue. It includes all the same packages as `requirements.txt` but excludes `neuronbridge-python` (which will be installed separately without dependencies in Step 2).

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
## Authentication Setup

To access NeuPrint and FlyWire datasets, you need to configure your authentication tokens.

1.  **Get your tokens:**
    *   **NeuPrint Token:** Log in to [neuprint.janelia.org](https://neuprint.janelia.org/account) and copy your token.
    *   **CAVE Token (FlyWire/FAFB):** Log in to [codex.flywire.ai](https://codex.flywire.ai/auth_token) and copy your token.
    *   **NeuronBridge:** No authentication required - NeuronBridge API is publicly accessible.

2.  **Configure local tokens:**
    The project uses a local file `token_info_local.txt` to store your secrets. This file is gitignored to prevent accidental commits.

    *   Copy the template file:
        ```bash
        cp token_info.txt token_info_local.txt
        ```
    *   Edit `token_info_local.txt` and paste your tokens:
        ```text
        NEUPRINT_TOKEN='your_actual_neuprint_token_here'
        CAVE_TOKEN='your_actual_cave_token_here'
        ```

    Alternatively, you can set environment variables `NEUPRINT_TOKEN` and `CAVE_TOKEN`.

**Token Requirements:**
- `NEUPRINT_TOKEN`: **Required** for accessing all NeuPrint datasets (hemibrain, male-cns, MANC, optic-lobe)
- `CAVE_TOKEN`: **Required** for accessing FlyWire datasets (FAFB, BANC)
- NeuronBridge API: **No token required** (publicly accessible)
## Core Dependencies

The following packages are **required**:

| Package         | Purpose                     | Version          |
| --------------- | --------------------------- | ---------------- |
| numpy           | Numerical computing         | >=1.20.0, <2.0.0 |
| pandas          | Data manipulation           | >=1.3.0, <2.0.0  |
| polars          | Memory-efficient large data | >=1.0.0          |
| scipy           | Scientific computing        | >=1.7.0          |
| plotly          | Interactive visualizations  | >=5.0.0          |
| matplotlib      | Static plots                | >=3.4.0          |
| networkx        | Network analysis            | >=2.6.0          |
| openpyxl        | Excel file support          | >=3.0.0          |
| **PyQt5**       | **Fast GUI dialogs**        | **>=5.15.0**     |
| neuprint-python | NeUPRINT database           | >=0.4.0          |

**Important:** numpy is constrained to <2.0.0 for binary compatibility with pandas 1.x.

**Note on Polars:** Polars is used for memory-efficient operations when building and loading large connection caches (millions of connections). It gracefully falls back to pandas if not available, but Polars is recommended for cross-dataset comparison workflows.

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
pip install navis trimesh k3d ipywidgets open3d
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
# Install system dependencies first (if needed)
sudo apt-get update
sudo apt-get install python3-pip python3-dev python3-tk

# Then install Python packages
pip install -r requirements.txt
```

**Note:** Most Linux distributions include these dependencies by default.

### Windows

**Windows requires a two-step installation process** to handle the `neuronbridge-python` package:

```bash
# Step 1: Install all dependencies from Windows-specific requirements
pip install -r requirements-windows.txt

# Step 2: Install neuronbridge-python without its dependencies
pip install neuronbridge-python --no-deps
```

**Why this is necessary:**
- The `neuronbridge-python` package on PyPI lists `memray` as a dependency
- `memray` is a memory profiler that only supports Linux and macOS (no Windows builds)
- However, `memray` is **not actually required** for the NeuronBridge API—it's only used for internal development
- The `requirements-windows.txt` file includes all necessary dependencies (`pydantic`, `python-rapidjson`, `ray[default]`) that `neuronbridge-python` needs to function
- Installing with `--no-deps` skips the `memray` requirement while keeping full functionality

**What's included in requirements-windows.txt:**
- All core packages (same as requirements.txt)
- NeuronBridge dependencies: `pydantic~=2.9.1`, `python-rapidjson~=1.20`, `ray[default]~=2.39.0`
- **Excludes:** `neuronbridge-python` itself (installed separately in Step 2)

**Note:** Make sure Python was installed with tkinter support (should be default on Windows).

## NeuronBridge Installation

The `neuronbridge-python` package is included in requirements.txt and works directly on **Linux and macOS**. However, **Windows users must use the special `requirements-windows.txt` file** due to platform compatibility issues.

### The Issue (Windows Only)

The `neuronbridge-python` package on PyPI includes `memray` as a dependency:
- **`memray`**: A memory profiler that only supports **Linux and macOS** (no Windows support)

This causes `pip install -r requirements.txt` to fail on Windows with an error like:
```
ERROR: Could not find a version that satisfies the requirement memray
```

However, **the core NeuronBridge API does not actually require memray**. It's only used for internal validation scripts at Janelia, not the public API.

### Solution: Use requirements-windows.txt

**The `requirements-windows.txt` file is the recommended solution for Windows users:**

```bash
# Step 1: Install all dependencies from Windows-specific requirements
pip install -r requirements-windows.txt

# Step 2: Install neuronbridge-python without its problematic dependencies
pip install neuronbridge-python --no-deps
```

**What requirements-windows.txt provides:**
- All the same core packages as `requirements.txt`
- Explicit versions of NeuronBridge dependencies: `pydantic~=2.9.1`, `python-rapidjson~=1.20`, `ray[default]~=2.39.0`
- Comments and documentation explaining the Windows-specific setup
- Excludes `neuronbridge-python` itself (which is installed separately in Step 2 without dependencies)

The core NeuronBridge API only needs: `pydantic`, `python-rapidjson`, and `ray[default]` — all included in requirements-windows.txt.

### Affected Features

If `neuronbridge-python` is not installed, the following features will be unavailable:
- `NeuronBridgeFinder` class in `src/neuronbridge_finder.py`
- Scripts: `NeuronBridge_FindNeuron.py`, `NeuronBridge_FindLines.py`
- EM-to-LM line matching functionality

**All other features of hemibrain-connectomes-analysis work normally without NeuronBridge.**

### Verification

Test your NeuronBridge installation:

```python
# Quick test
python -c "from neuronbridge.client import Client; print('✓ NeuronBridge API works')"

# Full test
python -c "from neuronbridge.client import Client; c = Client(); print(f'✓ NeuronBridge v{c.version}')"
```

## Verification

Check your installation:

```bash
python -c "import numpy, pandas, plotly, networkx, PyQt5; print('✓ All core packages installed')"
```

Test GUI backend:

```bash
python -c "from PyQt5.QtWidgets import QApplication; print('✓ PyQt5 is working')"
```

## Brain Transform Downloads

**For 3D visualization with whole-brain meshes** (`brain_mesh='whole'` in VisualizeSkeleton), the system may prompt you to download brain transforms (~10GB total, one-time).

### Transform Storage Location

Transforms are stored in `~/flybrain-data` (managed by the flybrains package):

```bash
# Check if transforms exist
ls ~/flybrain-data/

# Expected contents after download (8 files):
# JRC2018F_FAFB.h5          (~580 MB)
# JRC2018F_JFRC2013.h5      (~1.39 GB)
# JRC2018F_FCWB.h5          (~1.29 GB)
# JRC2018F_JRCFIB2018F.h5   (~1.29 GB) - For hemibrain/optic-lobe
# JRC2018U_JRC2018F.h5      (~717 MB)
# JRC2018U_JRC2018M.h5      (~1.10 GB)
# JRC2018F_JFRC2010.h5      (~1.65 GB)
# JRCFIB2022M_JRC2018M.h5   (~2.12 GB) - For male-cns
```

### Automatic Download Process

When using `brain_mesh='whole'` for hemibrain or optic-lobe datasets, the system will automatically check for required transforms:

```
⚠ Brain transforms not found for hemibrain:v1.2.1
  Target template: JRC2018F (whole brain)
  Transform size: ~10GB total (8 files, one-time download)
  Download time: ~1-2 hours
  Storage location: ~/flybrain-data

Download transforms? [y/N]:
```

**Important:** The flybrains package downloads ALL JRC transforms as a bundle (~10GB), even though hemibrain/optic-lobe only needs JRC2018F_JRCFIB2018F.h5 (~1.3GB). Selective download is not supported by the library.

**Why download everything?**
- The actual transform path is: `JRCFIB2018Fraw → JRCFIB2018F → JRCFIB2018Fum → JRC2018F`
- Only 1 file (JRC2018F_JRCFIB2018F.h5) is used for your dataset
- Other files enable cross-dataset registration and future flexibility
- One-time setup benefits all projects and datasets

Type `y` to proceed with the download. Transforms are shared across all NeuPrint projects.

### Dataset-Specific Requirements

| Dataset              | Template Mesh        | Whole Brain/VNC     | Transform Required     |
| -------------------- | -------------------- | ------------------- | ---------------------- |
| **hemibrain:v1.2.1** | JRCFIB2018F (EM)     | JRC2018F (confocal) | ✅ Yes (~10GB download) |
| **optic-lobe:v1.1**  | JRCFIB2018F (EM)     | JRC2018F (confocal) | ✅ Yes (~10GB download) |
| **manc:v1.2.3**      | MANC (native VNC)    | —                   | ❌ No (native)          |
| **male-cns:v0.9**    | JRCFIB2022M (native) | —                   | ❌ No (native)          |

### Manual Transform Verification

Check if transforms are available:

```bash
python -c "from flybrains import JRCFIB2018Fraw; print('✓ Template available')"
```

### Disk Space Requirements

- **Initial installation:** ~100MB (Python packages)
- **Transform download:** ~10GB (one-time, optional, for whole-brain visualization)
- **Cache storage:** Variable (depends on datasets used, typically 100MB-1GB)

**Total recommended:** ~12GB free disk space (if using whole-brain transforms)

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
conda create -n drocat python=3.11 -y
conda activate drocat

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
conda create -n drocat python=3.11 -y
conda activate drocat
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
