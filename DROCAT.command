#!/bin/zsh

# =============================================================================
# DROCAT - Drosophila Connectome Analysis Toolkit
# Double-click this file to launch the DROCAT web UI on macOS.
# =============================================================================

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Keep conda environments self-contained: ignore ~/.local user-site packages
export PYTHONNOUSERSITE=1

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
elif [ -f "$HOME/miniforge3/bin/conda" ]; then
  export PATH="$HOME/miniforge3/bin:$PATH"
  CONDA_AVAILABLE=true
  echo "Found Miniforge at $HOME/miniforge3"
else
  # No local conda: download and install Miniconda automatically
  echo "Conda not found. Installing Miniconda into ~/miniconda3 ..."
  ARCH="$(uname -m)"
  if [ "$ARCH" = "arm64" ]; then
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
  else
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
  fi
  INSTALLER="/tmp/miniconda_drocat.sh"
  # The .sh installer refuses to run into an existing directory (broken or
  # partial earlier install); -u repairs/updates it in that case.
  MC_FLAGS="-b"
  if [ -d "$HOME/miniconda3" ]; then
    MC_FLAGS="-b -u"
  fi
  if curl -fsSL "$MINICONDA_URL" -o "$INSTALLER" && bash "$INSTALLER" $MC_FLAGS -p "$HOME/miniconda3" && [ -f "$HOME/miniconda3/bin/conda" ]; then
    rm -f "$INSTALLER"
    export PATH="$HOME/miniconda3/bin:$PATH"
    "$HOME/miniconda3/bin/conda" init zsh >/dev/null 2>&1 || true
    "$HOME/miniconda3/bin/conda" init bash >/dev/null 2>&1 || true
    CONDA_AVAILABLE=true
    echo "Miniconda installed successfully."
  else
    echo "ERROR: could not install Miniconda automatically."
    echo "Install it manually from https://docs.conda.io/miniconda.html and re-run."
    read -r -p "Press Enter to exit..." _
    exit 1
  fi
fi

# -----------------------------------------------------------------------------
# Step 3: Create / Activate conda environment
# -----------------------------------------------------------------------------
if [ "$CONDA_AVAILABLE" = true ]; then
  eval "$(conda shell.zsh hook 2>/dev/null || conda shell.bash hook 2>/dev/null)"

  ROOT="$(cd "$(dirname "$0")" && pwd)"
  DROCAT_VERSION="$(sed -n 's/^APP_VERSION = "\([^"]*\)"/\1/p' "$ROOT/ui/config.py" | head -1)"
  DROCAT_VERSION="${DROCAT_VERSION:-4.5.0}"
  ENV_BASE="drocat-${DROCAT_VERSION}"

  # Resolve the environment: use 'drocat-<version>' if it is free/usable;
  # otherwise leave existing envs untouched and pick the next free name.
  CONDA_ENV=""
  env_num=0
  while [[ -z "$CONDA_ENV" && $env_num -le 20 ]]; do
    if [[ $env_num -eq 0 ]]; then candidate="$ENV_BASE"; else candidate="${ENV_BASE}-${env_num}"; fi
    if conda run -n "$candidate" python -c "import sys, nicegui; assert sys.version_info[:2]==(3,11)" >/dev/null 2>&1; then
      CONDA_ENV="$candidate"
    elif ! conda env list 2>/dev/null | grep -qE "^$candidate "; then
      echo "Conda environment '$candidate' not found. Creating it..."
      conda create -n "$candidate" python=3.11 -y
      CONDA_ENV="$candidate"
      conda activate "$CONDA_ENV"
      echo "Installing dependencies (this may take a few minutes)..."
      pip install -r requirements.txt --quiet
      pip install -r ui/requirements.txt --quiet
      pip install -e . --quiet || pip install -e . --no-deps --quiet
      echo "Setup complete!"
    else
      if [[ $env_num -eq 0 ]]; then
        echo "WARNING: existing '$ENV_BASE' env is not usable - using a new environment."
      fi
      env_num=$((env_num + 1))
    fi
  done
  if [[ -z "$CONDA_ENV" ]]; then
    echo "ERROR: could not resolve a usable $ENV_BASE environment."
    echo
    read "?Press Return to close."
    exit 1
  fi
  if [[ "$CONDA_ENV" != "$ENV_BASE" ]]; then
    echo "Using environment: $CONDA_ENV"
  fi
  conda activate "$CONDA_ENV" 2>/dev/null || true
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
# Step 4: Verify key packages (with NeuronBridge memray fallback)
# -----------------------------------------------------------------------------
if ! python3 -c "import neuronbridge" >/dev/null 2>&1; then
  echo "NeuronBridge not importable - retrying with --no-deps..."
  pip install neuronbridge-python --no-deps --quiet || \
    echo "Warning: neuronbridge-python unavailable; NeuronBridge panels will be limited."
fi

if ! python3 -c "import nicegui" >/dev/null 2>&1; then
  echo "NiceGUI not found. Installing..."
  pip install nicegui --quiet
fi

python3 -c "import numpy, pandas, polars, neuprint, nicegui, neuronbridge" >/dev/null 2>&1 && \
  echo "✓ Core imports verified" || \
  echo "⚠️  Import check failed - see messages above."

# -----------------------------------------------------------------------------
# Step 5: Launch the UI
# -----------------------------------------------------------------------------
echo
echo "Starting DROCAT Web UI..."
echo "The UI will open in your browser at http://127.0.0.1:8080"
echo "Press Ctrl+C to stop the server."
echo

python3 ui/app.py
