#!/bin/bash
# =============================================================================
# DROCAT UI Launcher for macOS/Linux
#
# Self-healing: creates the `drocat` conda environment and installs
# dependencies on first run, then launches the web UI.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
# ---------------------------------------------------------------------------
DROCAT_VERSION="$(sed -n 's/^APP_VERSION = "\([^"]*\)"/\1/p' "$SCRIPT_DIR/ui/config.py" | head -1)"
DROCAT_VERSION="${DROCAT_VERSION:-4.5.0}"
ENV_BASE="drocat-${DROCAT_VERSION}"
# Resolve environment: use 'drocat-<version>' if it is free/usable; otherwise
# leave existing envs untouched and pick the next free name
# (drocat-<version>-2, drocat-<version>-3, ...).
# ---------------------------------------------------------------------------
ENV_NAME=""
env_num=0
while [[ -z "$ENV_NAME" && $env_num -le 20 ]]; do
    if [[ $env_num -eq 0 ]]; then candidate="$ENV_BASE"; else candidate="${ENV_BASE}-${env_num}"; fi
    if conda run -n "$candidate" python -c "import sys, nicegui; assert sys.version_info[:2]==(3,11)" >/dev/null 2>&1; then
        ENV_NAME="$candidate"
    elif ! conda env list | grep -qE "^${candidate} "; then
        echo "Conda environment '$candidate' not found. Creating it (one-time setup)..."
        conda create -n "$candidate" python=3.11 -y
        ENV_NAME="$candidate"
    else
        if [[ $env_num -eq 0 ]]; then
            echo "WARNING: existing '$ENV_BASE' env is not usable (wrong Python or missing deps) - using a new env."
        fi
        env_num=$((env_num + 1))
    fi
done
if [[ -z "$ENV_NAME" ]]; then
    echo "ERROR: could not resolve a usable $ENV_BASE environment." >&2
    exit 1
fi
if [[ "$ENV_NAME" != "$ENV_BASE" ]]; then
    echo "Using environment: $ENV_NAME"
fi
conda activate "$ENV_NAME"

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
