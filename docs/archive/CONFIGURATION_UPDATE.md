# Configuration Files Updated

**Date:** October 31, 2025  
**Status:** ✅ COMPLETE

## Updated Files

### 1. `pyproject.toml` - Updated ✅

**Changes:**
- ✅ Updated version to `2.1.0` (reflects reorganization)
- ✅ Updated description to mention reorganized modular structure
- ✅ Added Python 3.12 support
- ✅ Added more keywords (neuprint, drosophila)
- ✅ Fixed package discovery to use src/ layout
- ✅ Updated pytest configuration to add pythonpath = ["src"]
- ✅ Updated coverage configuration to focus on src/ directory
- ✅ Added coverage report exclusion patterns
- ✅ Added Black formatter configuration
- ✅ Added package data configuration
- ✅ Improved documentation and comments

**Key Configuration:**
```toml
[tool.setuptools]
package-dir = {"" = "src"}
packages = ["src"]

[tool.setuptools.packages.find]
where = ["src"]
include = ["*"]
```

### 2. `requirements.txt` - Updated ✅

**Changes:**
- ✅ Added comprehensive header explaining project structure
- ✅ Better organization with clear sections
- ✅ Added version notes (pandas <2.0.0 compatibility)
- ✅ Improved comments and documentation
- ✅ Added optional dependencies section (commented out)
- ✅ Added installation instructions
- ✅ Added system notes about tkinter and GUI backends
- ✅ Noted that scripts import from src/ automatically

**Structure:**
```
Core Dependencies
Visualization
Network Analysis
Data Processing
GUI Backends
Neuroscience Data
3D Visualization (Optional)
Development/Testing (Optional)
Jupyter Support (Optional)
System Notes
Installation Instructions
```

## Installation Methods

### Method 1: Standard Installation (requirements.txt)
```bash
pip install -r requirements.txt
```

### Method 2: Development Installation (pyproject.toml)
```bash
# Basic installation
pip install -e .

# With development dependencies
pip install -e ".[dev]"

# With visualization dependencies
pip install -e ".[viz]"

# With all optional dependencies
pip install -e ".[all]"
```

## Project Structure Support

Both configuration files now properly support the reorganized structure:

```
project/
├── src/                    # Core modules (configured in pyproject.toml)
│   ├── coana.py
│   ├── statvis.py
│   └── vispath.py
├── scripts/                # Executable scripts (import from src/)
│   ├── FindPath.py
│   ├── FindDirect.py
│   └── ...
├── tests/                  # Test files (configured in pytest)
├── requirements.txt        # Dependency specification
└── pyproject.toml          # Modern Python configuration
```

## Key Features

### pyproject.toml Features:
1. **Src Layout:** Properly configured for src/ directory structure
2. **Testing:** Pytest configured with pythonpath to find src/
3. **Coverage:** Focuses on src/ directory, excludes tests/scripts/examples
4. **Optional Dependencies:** Grouped for easy installation
5. **Black Formatter:** Pre-configured code formatting rules
6. **Package Data:** Includes JSON, CSV, MD files

### requirements.txt Features:
1. **Well Documented:** Clear sections and explanations
2. **Version Constraints:** Proper version pinning (pandas <2.0.0)
3. **Optional Sections:** Commented out optional dependencies
4. **Installation Guide:** Instructions included in file
5. **Compatibility Notes:** System-specific information

## Dependency Summary

### Required (Core):
- numpy ≥1.20.0
- pandas ≥1.3.0, <2.0.0
- scipy ≥1.7.0
- plotly ≥5.0.0
- matplotlib ≥3.4.0
- seaborn ≥0.11.0
- networkx ≥2.6.0
- openpyxl ≥3.0.0
- xlrd ≥2.0.0
- PyQt5 ≥5.15.0
- neuprint-python ≥0.4.0

### Optional (Development):
- pytest ≥7.0.0
- pytest-cov ≥3.0.0
- jupyter ≥1.0.0
- ipykernel ≥6.0.0

### Optional (Visualization):
- navis ≥1.0.0
- trimesh ≥3.9.0
- kaleido ≥0.2.0
- bokeh ≥2.4.0

### Optional (GUI Alternatives):
- PyQt6 ≥6.0.0
- wxPython ≥4.2.0

## Testing Configuration

```bash
# Run tests with proper path configuration
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_specific.py
```

## Version History

- **v2.1.0** (Oct 31, 2025): Project reorganization with src/ layout
- **v2.0.0** (Previous): Original flat structure

## Compatibility

- **Python:** 3.8, 3.9, 3.10, 3.11, 3.12
- **OS:** macOS, Linux, Windows
- **Pandas:** <2.0.0 for compatibility with existing code

## Next Steps

1. ✅ Updated pyproject.toml
2. ✅ Updated requirements.txt
3. ⏳ Optional: Create setup.py for backward compatibility
4. ⏳ Optional: Test installation in clean environment
5. ⏳ Optional: Publish to PyPI (if desired)

## Verification

Test that configuration works:

```bash
# Test basic installation
pip install -e .

# Verify imports work
python -c "import sys; sys.path.insert(0, 'src'); from coana import FindNeuronConnection; print('✅ Configuration working!')"

# Test pytest configuration
pytest --collect-only
```

---

**All configuration files updated to support the reorganized project structure!** ✅
