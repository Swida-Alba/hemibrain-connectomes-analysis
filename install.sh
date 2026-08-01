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

# =============================================================================
# Step 2: Create Conda Environment
# =============================================================================
echo -e "\n${BLUE}[2/5] Creating conda environment '${ENV_NAME}'...${NC}"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Environment '${ENV_NAME}' already exists. Updating...${NC}"
    conda activate "$ENV_NAME" 2>/dev/null || source activate "$ENV_NAME"
else
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    conda activate "$ENV_NAME" 2>/dev/null || source activate "$ENV_NAME"
fi

echo -e "${GREEN}✓ Environment ready (Python $(python --version 2>&1 | cut -d' ' -f2))${NC}"

# =============================================================================
# Step 3: Install Dependencies
# =============================================================================
echo -e "\n${BLUE}[3/5] Installing dependencies...${NC}"

cd "$SCRIPT_DIR"

# Install main requirements
echo "Installing core dependencies (this may take a few minutes)..."
pip install -r requirements.txt --quiet

# Install UI dependencies
echo "Installing UI dependencies..."
pip install -r ui/requirements.txt --quiet

echo -e "${GREEN}✓ Dependencies installed${NC}"

# =============================================================================
# Step 4: Install DROCAT Package
# =============================================================================
echo -e "\n${BLUE}[4/5] Installing DROCAT package...${NC}"

pip install -e . --quiet

echo -e "${GREEN}✓ DROCAT installed in editable mode${NC}"

# =============================================================================
# Step 5: Create Launcher Scripts
# =============================================================================
echo -e "\n${BLUE}[5/5] Creating launcher scripts...${NC}"

# Create run_ui.sh
cat > "$SCRIPT_DIR/run_ui.sh" << 'EOF'
#!/bin/bash
# DROCAT UI Launcher
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate conda environment
if command -v conda &> /dev/null; then
    conda activate drocat 2>/dev/null || source activate drocat
elif [[ -f "$HOME/miniconda3/bin/activate" ]]; then
    source "$HOME/miniconda3/bin/activate" drocat
fi

# Launch UI
cd "$SCRIPT_DIR"
python ui/app.py
EOF
chmod +x "$SCRIPT_DIR/run_ui.sh"

echo -e "${GREEN}✓ Launcher created: run_ui.sh${NC}"

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
