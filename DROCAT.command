#!/bin/zsh

# =============================================================================
# DROCAT - Drosophila Connectome Analysis Toolkit
# Double-click this file to launch the DROCAT web UI on macOS.
# =============================================================================

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

clear 2>/dev/null || true
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     DROCAT - Drosophila Connectome Analysis Toolkit          ║"
echo "║                    Web UI Launcher                           ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo

# -----------------------------------------------------------------------------
# Step 1: Check for Python 3
# -----------------------------------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is required."
  echo "Install it from https://python.org, then double-click this file again."
  echo
  read "?Press Return to close."
  exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0")
echo "Found Python $PYTHON_VERSION"

# -----------------------------------------------------------------------------
# Step 2: Check for Conda environment
# -----------------------------------------------------------------------------
CONDA_ENV="drocat"
CONDA_AVAILABLE=false

if command -v conda >/dev/null 2>&1; then
  CONDA_AVAILABLE=true
  echo "Found Conda: $(conda --version 2>/dev/null || echo 'unknown version')"
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
  export PATH="$HOME/miniconda3/bin:$PATH"
  CONDA_AVAILABLE=true
  echo "Found Miniconda at $HOME/miniconda3"
elif [ -f "$HOME/anaconda3/bin/conda" ]; then
  export PATH="$HOME/anaconda3/bin:$PATH"
  CONDA_AVAILABLE=true
  echo "Found Anaconda at $HOME/anaconda3"
fi

# -----------------------------------------------------------------------------
# Step 3: Create / Activate conda environment
# -----------------------------------------------------------------------------
if [ "$CONDA_AVAILABLE" = true ]; then
  eval "$(conda shell.zsh hook 2>/dev/null || conda shell.bash hook 2>/dev/null)"

  if conda env list 2>/dev/null | grep -q "^$CONDA_ENV "; then
    echo "Activating conda environment: $CONDA_ENV"
    conda activate "$CONDA_ENV" 2>/dev/null
  else
    echo "Conda environment '$CONDA_ENV' not found."
    echo "Creating conda environment with Python 3.11..."
    conda create -n "$CONDA_ENV" python=3.11 -y
    conda activate "$CONDA_ENV"
    echo "Installing dependencies (this may take a few minutes)..."
    pip install -r requirements.txt --quiet
    pip install -r ui/requirements.txt --quiet
    pip install -e . --quiet
    echo "Setup complete!"
  fi
else
  echo "Conda not found. Using system Python."
  echo "ERROR: DROCAT requires the 'drocat' conda environment."
  echo "Please install Miniconda from https://docs.conda.io/miniconda.html,"
  echo "then run ./install.sh once, and double-click this file again."
  echo
  read "?Press Return to close."
  exit 1
fi

# -----------------------------------------------------------------------------
# Step 4: Verify nicegui is installed
# -----------------------------------------------------------------------------
if ! python3 -c "import nicegui" >/dev/null 2>&1; then
  echo "NiceGUI not found. Installing..."
  pip install nicegui --quiet
fi

# -----------------------------------------------------------------------------
# Step 5: Launch the UI
# -----------------------------------------------------------------------------
echo
echo "Starting DROCAT Web UI..."
echo "The UI will open in your browser at http://127.0.0.1:8080"
echo "Press Ctrl+C to stop the server."
echo

python3 ui/app.py
