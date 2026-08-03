#!/bin/bash
# =============================================================================
# DROCAT UI Launcher for macOS/Linux
#
# Self-healing: creates the `drocat` conda environment and installs
# dependencies on first run, then launches the web UI.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="drocat"

echo "Starting DROCAT UI..."

# ---------------------------------------------------------------------------
# Locate conda
# ---------------------------------------------------------------------------
CONDA_BIN=""
if command -v conda &> /dev/null; then
    CONDA_BIN="$(command -v conda)"
elif [[ -f "$HOME/miniconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/miniconda3/bin/conda"
elif [[ -f "$HOME/anaconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/anaconda3/bin/conda"
fi

if [[ -z "$CONDA_BIN" ]]; then
    echo "ERROR: Conda was not found." >&2
    echo "Please install Miniconda (https://docs.conda.io/miniconda.html) and re-run this script," >&2
    echo "or run: ./install.sh" >&2
    exit 1
fi

source "$(dirname "$(dirname "$CONDA_BIN")")/etc/profile.d/conda.sh" 2>/dev/null || \
    eval "$($CONDA_BIN shell.bash hook)"

# ---------------------------------------------------------------------------
# Create environment on first run
# ---------------------------------------------------------------------------
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '$ENV_NAME' not found. Creating it (one-time setup)..."
    conda create -n "$ENV_NAME" python=3.11 -y
fi

conda activate "$ENV_NAME"

PYVER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo '')"
if [[ -n "$PYVER" && "$PYVER" != "3.11" ]]; then
    echo "ERROR: the existing '$ENV_NAME' env uses Python $PYVER (expected 3.11)." >&2
    echo "Recreate it with: conda env remove -n $ENV_NAME -y && ./install.sh" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Install dependencies if missing
# ---------------------------------------------------------------------------
if ! python -c "import nicegui, pandas, numpy, neuprint" &> /dev/null; then
    echo "Installing dependencies (this may take a few minutes)..."
    pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
    pip install -r "$SCRIPT_DIR/ui/requirements.txt" --quiet
    pip install -e "$SCRIPT_DIR" --quiet
fi

# ---------------------------------------------------------------------------
# Launch UI
# ---------------------------------------------------------------------------
cd "$SCRIPT_DIR"
python ui/app.py
