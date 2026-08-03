#!/usr/bin/env bash
# DROCAT one-click installer for macOS and Linux.

set -euo pipefail
export PYTHONNOUSERSITE=1

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VERSION="3.11"
DROCAT_VERSION=""
if [[ -f "$SCRIPT_DIR/ui/config.py" ]]; then
    DROCAT_VERSION="$(sed -n 's/^APP_VERSION = "\([^"]*\)"/\1/p' "$SCRIPT_DIR/ui/config.py" | head -1)"
fi
if [[ -z "$DROCAT_VERSION" ]]; then
    DROCAT_VERSION="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$SCRIPT_DIR/pyproject.toml" | head -1)"
fi
DROCAT_VERSION="${DROCAT_VERSION:-4.4.5}"
ENV_BASE="drocat-${DROCAT_VERSION}"

printf "%b\n" "$BLUE"
printf '%s\n' '╔═══════════════════════════════════════════════════════════════╗'
printf '%s\n' '║     DROCAT - Drosophila Connectome Analysis Toolkit          ║'
printf '%s\n' '║                    One-Click Installer                        ║'
printf '%s\n' '╚═══════════════════════════════════════════════════════════════╝'
printf "%b\n" "$NC"

command_exists() { command -v "$1" >/dev/null 2>&1; }

find_conda() {
    if command_exists conda; then
        command -v conda
        return
    fi
    local candidate
    for candidate in \
        "$HOME/miniconda3/bin/conda" \
        "$HOME/anaconda3/bin/conda" \
        "$HOME/miniforge3/bin/conda" \
        "/opt/miniconda3/bin/conda" \
        "/opt/anaconda3/bin/conda" \
        "/usr/local/miniconda3/bin/conda" \
        "/usr/local/anaconda3/bin/conda"; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
}

install_miniconda() {
    local os arch url installer flags
    os="$(uname -s)"
    arch="$(uname -m)"
    case "$os:$arch" in
        Darwin:arm64) url="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh" ;;
        Darwin:*) url="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh" ;;
        Linux:aarch64|Linux:arm64) url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh" ;;
        Linux:*) url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" ;;
        *) printf "%bUnsupported operating system: %s%b\n" "$RED" "$os" "$NC" >&2; exit 1 ;;
    esac

    installer="$(mktemp "${TMPDIR:-/tmp}/drocat-miniconda.XXXXXX.sh")"
    trap 'rm -f "${installer:-}"' EXIT
    printf '%s\n' "Downloading Miniconda..."
    if command_exists curl; then
        curl -fsSL "$url" -o "$installer"
    elif command_exists wget; then
        wget -q "$url" -O "$installer"
    else
        printf "%bERROR: curl or wget is required to download Miniconda.%b\n" "$RED" "$NC" >&2
        exit 1
    fi
    flags=(-b -p "$HOME/miniconda3")
    [[ -d "$HOME/miniconda3" ]] && flags=(-b -u -p "$HOME/miniconda3")
    bash "$installer" "${flags[@]}"
    [[ -x "$HOME/miniconda3/bin/conda" ]] || {
        printf "%bERROR: Miniconda installation did not produce conda.%b\n" "$RED" "$NC" >&2
        exit 1
    }
    rm -f "$installer"
    trap - EXIT
    CONDA_BIN="$HOME/miniconda3/bin/conda"
}

printf "%b[1/5] Checking for Conda...%b\n" "$BLUE" "$NC"
CONDA_BIN="$(find_conda || true)"
if [[ -z "$CONDA_BIN" ]]; then
    printf "%bConda was not found; installing Miniconda.%b\n" "$YELLOW" "$NC"
    install_miniconda
fi
printf "%b✓ Using %s%b\n" "$GREEN" "$CONDA_BIN" "$NC"

env_exists() {
    "$CONDA_BIN" env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$1"
}

env_usable() {
    "$CONDA_BIN" run -n "$1" python -c \
        'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)' \
        >/dev/null 2>&1
}

printf "\n%b[2/5] Selecting a Python %s environment...%b\n" "$BLUE" "$PYTHON_VERSION" "$NC"
ENV_NAME=""
for index in $(seq 0 20); do
    if [[ "$index" -eq 0 ]]; then
        candidate="$ENV_BASE"
    else
        candidate="${ENV_BASE}-$((index + 1))"
    fi
    if env_exists "$candidate"; then
        if env_usable "$candidate"; then
            ENV_NAME="$candidate"
            printf "%b✓ Reusing %s%b\n" "$GREEN" "$ENV_NAME" "$NC"
            break
        fi
        printf "%bSkipping %s because it is not Python %s.%b\n" \
            "$YELLOW" "$candidate" "$PYTHON_VERSION" "$NC"
        continue
    fi
    ENV_NAME="$candidate"
    printf '%s\n' "Creating $ENV_NAME..."
    "$CONDA_BIN" create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
    break
done
[[ -n "$ENV_NAME" ]] || {
    printf "%bERROR: could not select a usable %s environment.%b\n" "$RED" "$ENV_BASE" "$NC" >&2
    exit 1
}

run_in_env() {
    "$CONDA_BIN" run -n "$ENV_NAME" --no-capture-output "$@"
}

printf "\n%b[3/5] Installing pinned dependencies...%b\n" "$BLUE" "$NC"
cd "$SCRIPT_DIR"
run_in_env python -m pip install --upgrade pip setuptools wheel

# Releases before this one installed the upstream client with an incompatible
# Pydantic constraint. DROCAT now bundles its API client, so remove the stale
# distribution when repairing an existing versioned environment.
if run_in_env python -m pip show neuronbridge-python >/dev/null 2>&1; then
    printf '%s\n' "Removing legacy neuronbridge-python dependency..."
    run_in_env python -m pip uninstall -y neuronbridge-python
fi
run_in_env python -m pip install --upgrade -r requirements.txt

printf "\n%b[4/5] Installing DROCAT...%b\n" "$BLUE" "$NC"
run_in_env python -m pip install -e . --no-deps

printf "\n%b[5/5] Verifying the environment...%b\n" "$BLUE" "$NC"
run_in_env python -m pip check
run_in_env python skills/drocat-install/scripts/verify_install.py --project "$SCRIPT_DIR"

printf "\n%bInstallation complete.%b\n" "$GREEN" "$NC"
printf 'Environment: %s\n' "$ENV_NAME"
printf 'Run a workflow with: conda run -n %s python scripts/FindDirect.py\n' "$ENV_NAME"
printf '%s\n' 'Configure NeuPrint/CAVE tokens in token_info_local.txt before online workflows.'
