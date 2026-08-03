#!/bin/bash
# =============================================================================
# DROCAT One-Click Installer for macOS/Linux
# =============================================================================
# This script installs DROCAT with all dependencies using Miniconda.
# Usage: chmod +x install.sh && ./install.sh
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     DROCAT - Drosophila Connectome Analysis Toolkit          ║"
echo "║                    One-Click Installer                        ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Configuration
ENV_NAME="drocat"
PYTHON_VERSION="3.11"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect OS and architecture
OS="$(uname -s)"
ARCH="$(uname -m)"

echo -e "${YELLOW}Detected system: ${OS} ${ARCH}${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# =============================================================================
# Step 1: Check/Install Miniconda
# =============================================================================
echo -e "\n${BLUE}[1/5] Checking for Conda...${NC}"

if command_exists conda; then
    echo -e "${GREEN}✓ Conda found: $(conda --version)${NC}"
    CONDA_AVAILABLE=true
else
    echo -e "${YELLOW}Conda not found. Installing Miniconda...${NC}"
    
    # Determine installer URL
    if [[ "$OS" == "Darwin" ]]; then
        if [[ "$ARCH" == "arm64" ]]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        else
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        fi
    else
        if [[ "$ARCH" == "aarch64" ]]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
        else
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        fi
    fi
    
    # Download and install
    INSTALLER_PATH="/tmp/miniconda_installer.sh"
    echo "Downloading Miniconda from: $MINICONDA_URL"
    curl -fsSL "$MINICONDA_URL" -o "$INSTALLER_PATH"
    
    echo "Installing Miniconda..."
    bash "$INSTALLER_PATH" -b -p "$HOME/miniconda3"
    rm "$INSTALLER_PATH"
    
    # Initialize conda
    "$HOME/miniconda3/bin/conda" init bash 2>/dev/null || true
    "$HOME/miniconda3/bin/conda" init zsh 2>/dev/null || true
    
    # Add to current session
    export PATH="$HOME/miniconda3/bin:$PATH"
    
    echo -e "${GREEN}✓ Miniconda installed successfully${NC}"
    CONDA_AVAILABLE=true
fi

# Ensure conda is in PATH
if ! command_exists conda; then
    if [[ -f "$HOME/miniconda3/bin/conda" ]]; then
        export PATH="$HOME/miniconda3/bin:$PATH"
    elif [[ -f "$HOME/anaconda3/bin/conda" ]]; then
        export PATH="$HOME/anaconda3/bin:$PATH"
    fi
fi

# Resolve the conda binary and initialize the shell hook so that
# `conda activate` works in this non-interactive session.
CONDA_BIN="$(command -v conda 2>/dev/null || true)"
if [[ -z "$CONDA_BIN" && -f "$HOME/miniconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/miniconda3/bin/conda"
fi
if [[ -z "$CONDA_BIN" && -f "$HOME/anaconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/anaconda3/bin/conda"
fi
if [[ -n "$CONDA_BIN" ]]; then
    CONDA_BASE="$("$CONDA_BIN" info --base 2>/dev/null)"
    if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1090
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    fi
fi

# =============================================================================
# Step 2: Select / Create Conda Environment
# =============================================================================
echo -e "\n${BLUE}[2/5] Selecting conda environment...${NC}"

# If 'drocat' already exists, NEVER touch it: warn and pick the next free
# name (drocat-2, drocat-3, ...) so no environment name conflicts occur.
if conda env list | grep -qE "^${ENV_NAME} "; then
    echo -e "${YELLOW}WARNING: conda env '${ENV_NAME}' already exists - leaving it untouched.${NC}"
    env_num=2
    while conda env list | grep -qE "^${ENV_NAME}-${env_num} "; do
        env_num=$((env_num + 1))
    done
    ENV_NAME="${ENV_NAME}-${env_num}"
    echo -e "${GREEN}Using a new environment instead: '${ENV_NAME}'${NC}"
fi

echo -e "${BLUE}Creating conda environment '${ENV_NAME}' (Python ${PYTHON_VERSION})...${NC}"
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
conda activate "$ENV_NAME" 2>/dev/null || source activate "$ENV_NAME"

echo -e "${GREEN}✓ Environment ready (Python $(python --version 2>&1 | cut -d' ' -f2))${NC}"

# =============================================================================
# Step 3: Install Dependencies
# =============================================================================
echo -e "\n${BLUE}[3/5] Installing dependencies...${NC}"

cd "$SCRIPT_DIR"

# Install main requirements
echo "Installing core dependencies (this may take a few minutes)..."
pip install -r requirements.txt --quiet

# NeuronBridge safety net: memray (a neuronbridge-python dependency) can fail
# to build on some platforms. If the import is missing, retry without deps -
# the compatible deps are already covered by requirements.txt.
if ! python -c "import neuronbridge" >/dev/null 2>&1; then
    echo "NeuronBridge not importable - retrying with --no-deps..."
    pip install neuronbridge-python --no-deps --quiet || \
        echo -e "${YELLOW}[WARN] neuronbridge-python could not be installed; NeuronBridge panels will be limited${NC}"
fi

# Install UI dependencies
echo "Installing UI dependencies..."
pip install -r ui/requirements.txt --quiet

echo -e "${GREEN}✓ Dependencies installed${NC}"

# =============================================================================
# Step 4: Install DROCAT Package
# =============================================================================
echo -e "\n${BLUE}[4/5] Installing DROCAT package...${NC}"

pip install -e . --quiet || pip install -e . --no-deps --quiet

echo -e "${GREEN}✓ DROCAT installed in editable mode${NC}"

# =============================================================================
# Step 5: Create Launcher Scripts (never overwrite an existing launcher)
# =============================================================================
echo -e "\n${BLUE}[5/5] Creating launcher scripts...${NC}"

# Create run_ui.sh
if [[ -f "$SCRIPT_DIR/run_ui.sh" ]]; then
    echo -e "${YELLOW}run_ui.sh already exists - keeping it${NC}"
else
cat > "$SCRIPT_DIR/run_ui.sh" << 'EOF'
#!/bin/bash
# DROCAT UI Launcher (self-healing: creates the env on first run)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_BIN="$(command -v conda 2>/dev/null || true)"
if [[ -z "$CONDA_BIN" && -f "$HOME/miniconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/miniconda3/bin/conda"
fi
if [[ -z "$CONDA_BIN" && -f "$HOME/anaconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/anaconda3/bin/conda"
fi
if [[ -z "$CONDA_BIN" ]]; then
    echo "ERROR: conda not found. Run ./install.sh first." >&2
    exit 1
fi
source "$(dirname "$(dirname "$CONDA_BIN")")/etc/profile.d/conda.sh" 2>/dev/null || \
    eval "$($CONDA_BIN shell.bash hook)"

# Resolve the environment: use 'drocat' if it does not exist yet; otherwise
# leave it untouched and use the next free name (drocat-2, drocat-3, ...).
ENV_NAME=""
env_num=0
while [[ -z "$ENV_NAME" && $env_num -le 20 ]]; do
    if [[ $env_num -eq 0 ]]; then candidate="drocat"; else candidate="drocat-${env_num}"; fi
    if conda run -n "$candidate" python -c "import sys, nicegui; assert sys.version_info[:2]==(3,11)" >/dev/null 2>&1; then
        ENV_NAME="$candidate"
    elif ! conda env list | grep -qE "^${candidate} "; then
        echo "Creating conda environment '$candidate' (one-time setup)..."
        conda create -n "$candidate" python=3.11 -y
        ENV_NAME="$candidate"
    else
        if [[ $env_num -eq 0 ]]; then
            echo "WARNING: existing 'drocat' env is not usable (wrong Python or missing deps) - using a new env."
        fi
        env_num=$((env_num + 1))
    fi
done
if [[ -z "$ENV_NAME" ]]; then
    echo "ERROR: could not resolve a usable drocat environment." >&2
    exit 1
fi
if [[ "$ENV_NAME" != "drocat" ]]; then
    echo "Using environment: $ENV_NAME"
fi
conda activate "$ENV_NAME"

if ! python -c "import nicegui, pandas, numpy, neuprint" &> /dev/null; then
    echo "Installing dependencies (first run)..."
    pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
    pip install neuronbridge-python --no-deps --quiet 2>/dev/null || true
    pip install -r "$SCRIPT_DIR/ui/requirements.txt" --quiet
    pip install -e "$SCRIPT_DIR" --no-deps --quiet
fi

# Launch UI
cd "$SCRIPT_DIR"
python ui/app.py
EOF
chmod +x "$SCRIPT_DIR/run_ui.sh"
fi

echo -e "${GREEN}✓ Launcher created: run_ui.sh${NC}"

# =============================================================================
# Step 6: Verify the installation
# =============================================================================
echo -e "\n${BLUE}[6/6] Verifying installation...${NC}"
python -m pip check || echo -e "${YELLOW}[WARN] pip check reported dependency conflicts${NC}"
if python -c "import numpy, pandas, polars, scipy, matplotlib, plotly, networkx, neuprint, nicegui, neuronbridge" 2>/dev/null; then
    echo -e "${GREEN}✓ Core imports verified${NC}"
else
    echo -e "${YELLOW}[WARN] Some imports failed - check the messages above.${NC}"
fi

# =============================================================================
# Installation Complete
# =============================================================================
echo -e "\n${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║              Installation Complete!                           ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}To launch DROCAT UI:${NC}"
echo "  ./run_ui.sh"
echo ""
echo -e "${BLUE}Or manually:${NC}"
echo "  conda activate $ENV_NAME"
echo "  python ui/app.py"
echo ""
echo -e "${BLUE}The UI will open at: http://127.0.0.1:8080${NC}"
echo ""
echo -e "${YELLOW}Note: You may need to restart your terminal or run:${NC}"
echo "  source ~/.zshrc  (or source ~/.bashrc)"
echo ""
echo -e "${BLUE}First time? Configure your NeuPrint token in the Settings tab.${NC}"
