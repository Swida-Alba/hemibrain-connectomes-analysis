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
# The installer lives in archive/install; the repository root is two levels up.
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_VERSION="3.11"
DROCAT_VERSION=""
if [[ -f "$PROJECT_ROOT/ui/config.py" ]]; then
    DROCAT_VERSION="$(sed -n 's/^APP_VERSION = "\([^"]*\)"/\1/p' "$PROJECT_ROOT/ui/config.py" | head -1)"
fi
if [[ -z "$DROCAT_VERSION" ]]; then
    DROCAT_VERSION="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$PROJECT_ROOT/pyproject.toml" | head -1)"
fi
DROCAT_VERSION="${DROCAT_VERSION:-4.5.0}"
ENV_BASE="drocat-${DROCAT_VERSION}"
# config.json ships clean on GitHub (committed defaults) and wins per key -
# it is the file a GitHub-pulled copy edits directly; the gitignored
# config_local.json is the developer-specific fallback for empty entries.
# It is NEVER created automatically: developers add it manually
# (cp config.json config_local.json) when local overrides are needed.
CONFIG_FILE="$PROJECT_ROOT/config.json"
CONFIG_LOCAL="$PROJECT_ROOT/config_local.json"

# Minimal JSON reader for config.json - a format this project generates and
# keeps simple (string values only, one level of section objects). Commas are
# normalized to newlines first so pretty-printed and single-line JSON both
# work.
json_value() {
    # $1 = section ("envs"|"tokens"), $2 = key, $3 = file
    tr ',' '\n' < "$3" | awk -v section="$1" -v key="$2" '
        {
            line = $0
            sub(/^[[:space:]]*/, "", line)
            sub(/[[:space:]]*$/, "", line)
            if (!in_section) {
                if (line ~ "\"" section "\"") {
                    in_section = 1
                    sub(/^.*\"" section "\"/, "", line)
                } else {
                    next
                }
            }
            if (line ~ "\"" key "\":") {
                rest = substr(line, index(line, "\"" key "\":") + length("\"" key "\":"))
                sub(/^[[:space:]]*/, "", rest)
                gsub(/[{}]/, "", rest)
                gsub(/^"|"$/, "", rest)
                print rest
                exit
            }
            if (line ~ /^[[:space:]]*}/) { in_section = 0 }
        }
    '
}

# Update envs.<version> in the LOCAL config (config_local.json) with the
# environment actually used, so an empty entry is filled with the
# auto-created name. The committed config.json is never rewritten.
update_config_env() {
    # $1 = version, $2 = env name, $3 = file
    [[ -f "$3" ]] || return 0
    local tmp
    tmp="$(mktemp "${TMPDIR:-/tmp}/drocat-config.XXXXXX")"
    awk -v version="$1" -v envname="$2" '
        BEGIN { in_envs = 0; done = 0 }
        {
            line = $0
            if (!done && line ~ /"envs"/) { in_envs = 1 }
            pattern = "\"" version "\"[[:space:]]*:[[:space:]]*\"[^\"]*\""
            if (in_envs && !done && line ~ pattern) {
                sub(pattern, "\"" version "\": \"" envname "\"", line)
                done = 1
                in_envs = 0
            }
            print line
        }
    ' "$3" > "$tmp" && mv "$tmp" "$3"
    chmod 600 "$3"
}

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
    "$CONDA_BIN" env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$1" \
        || [[ -d "$(dirname "$(dirname "$CONDA_BIN")")/envs/$1" ]]
}

env_usable() {
    "$CONDA_BIN" run -n "$1" python -c \
        'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)' \
        >/dev/null 2>&1
}

printf "\n%b[2/5] Selecting a Python %s environment...%b\n" "$BLUE" "$PYTHON_VERSION" "$NC"
# Version-specific custom env override: envs.<version> is only consulted
# for the CURRENT release, so upgrading DROCAT never reuses an older
# release's custom environment. config.json wins per key; the gitignored
# config_local.json fills empty entries.
ENV_NAME=""
ENV_OVERRIDE=""
ENV_SOURCE=""
for cfg in "$CONFIG_FILE" "$CONFIG_LOCAL"; do
    if [[ -f "$cfg" ]]; then
        ENV_OVERRIDE="$(json_value envs "$DROCAT_VERSION" "$cfg" || true)"
        ENV_OVERRIDE="$(printf '%s' "$ENV_OVERRIDE" | tr -d '[:space:]')"
        if [[ -n "$ENV_OVERRIDE" ]]; then
            ENV_SOURCE="$(basename "$cfg")"
            break
        fi
    fi
done
if [[ -n "$ENV_OVERRIDE" ]]; then
    if env_exists "$ENV_OVERRIDE"; then
        if env_usable "$ENV_OVERRIDE"; then
            ENV_NAME="$ENV_OVERRIDE"
            printf "%b✓ Reusing %s (custom env from %s)%b\n" "$GREEN" "$ENV_NAME" "$ENV_SOURCE" "$NC"
        else
            # A configured custom env must be the env used: never silently
            # switch to a default name and clobber the config entry.
            printf "%bERROR: %s (custom env from %s) exists but is not Python %s. Fix or remove it, or clear the envs entry.%b\n" \
                "$RED" "$ENV_OVERRIDE" "$ENV_SOURCE" "$PYTHON_VERSION" "$NC" >&2
            exit 1
        fi
    else
        ENV_NAME="$ENV_OVERRIDE"
        printf '%s\n' "Creating $ENV_NAME (custom env from $ENV_SOURCE)..."
        "$CONDA_BIN" create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
    fi
fi
if [[ -z "$ENV_NAME" ]]; then
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
fi
[[ -n "$ENV_NAME" ]] || {
    printf "%bERROR: could not select a usable %s environment.%b\n" "$RED" "$ENV_BASE" "$NC" >&2
    exit 1
}
# Persist the auto-selected environment into the LOCAL config
# (config_local.json) when no custom name was configured - an empty entry
# is filled with the auto-created name so launchers and scripts resolve it
# directly (it only applies while the config.json entry stays empty). The
# committed config.json is never rewritten, and a configured custom name is
# never touched.
if [[ -z "$ENV_OVERRIDE" ]]; then
    update_config_env "$DROCAT_VERSION" "$ENV_NAME" "$CONFIG_LOCAL"
fi

run_in_env() {
    "$CONDA_BIN" run -n "$ENV_NAME" --no-capture-output "$@"
}

printf "\n%b[3/5] Installing pinned dependencies...%b\n" "$BLUE" "$NC"
cd "$PROJECT_ROOT"
run_in_env python -m pip install --upgrade pip setuptools wheel

# Releases before this one installed the upstream client with an incompatible
# Pydantic constraint. DROCAT now bundles its API client, so remove the stale
# distribution when repairing an existing versioned environment.
if run_in_env python -m pip show neuronbridge-python >/dev/null 2>&1; then
    printf '%s\n' "Removing legacy neuronbridge-python dependency..."
    run_in_env python -m pip uninstall -y neuronbridge-python
fi
run_in_env python -m pip install --upgrade -r requirements.txt -r ui/requirements.txt

printf "\n%b[4/5] Installing DROCAT...%b\n" "$BLUE" "$NC"
run_in_env python -m pip install -e . --no-deps

printf "\n%b[5/5] Verifying the environment...%b\n" "$BLUE" "$NC"
run_in_env python -m pip check
# pip can silently install a metadata-only wheel when a PEP 517 sdist build
# is corrupted mid-build on a flaky network (observed with img2pdf and
# asciitree: "Successfully installed" but the modules are missing). The
# bundled verifier catches it; before aborting, clear the pip wheel cache
# (which would otherwise re-serve the bad wheels) and rebuild once.
VERIFY_OK=""
for attempt in 1 2; do
    if [[ "$attempt" -eq 2 ]]; then
        printf "%bVerification failed; clearing the pip wheel cache and rebuilding dependencies once.%b\n" "$YELLOW" "$NC"
        pip_cache_dir="${PIP_CACHE_DIR:-}"
        if [[ -z "$pip_cache_dir" ]]; then
            if [[ "$(uname -s)" == "Darwin" ]]; then
                pip_cache_dir="$HOME/Library/Caches/pip"
            else
                pip_cache_dir="$HOME/.cache/pip"
            fi
        fi
        rm -rf "$pip_cache_dir/wheels"
        run_in_env python -m pip install --upgrade -r requirements.txt -r ui/requirements.txt
    fi
    if run_in_env python skills/drocat-install/scripts/verify_install.py --project "$PROJECT_ROOT"; then
        VERIFY_OK=1
        break
    fi
done
if [[ -z "$VERIFY_OK" ]]; then
    printf "%bERROR: environment verification failed after one rebuild retry.%b\n" "$RED" "$NC" >&2
    exit 1
fi

printf "\n%bInstallation complete.%b\n" "$GREEN" "$NC"
printf 'Environment: %s\n' "$ENV_NAME"
printf '%s\n' 'Launch with: ./run_DROCAT.command'

# --- Token configuration notice ---
# Tokens are NOT collected in the terminal: they are set in the UI Settings
# tab after launch, or by editing config_local.json at the repository root
# (the gitignored override; config.json ships clean). The NeuPrint token is
# required for NeuPrint datasets; the CAVE token is optional and only needed
# for FlyWire FAFB online fetching.
printf '\n%b[Token setup]%b\n' "$BLUE" "$NC"
configure_tokens() {
    local neuprint_now="" cave_now=""
    for cfg in "$CONFIG_FILE" "$CONFIG_LOCAL"; do
        if [[ -f "$cfg" ]]; then
            neuprint_now="$(json_value tokens neuprint "$cfg" || true)"
            neuprint_now="$(printf '%s' "$neuprint_now" | tr -d '[:space:]')"
            [[ -n "$neuprint_now" ]] && break
        fi
    done
    for cfg in "$CONFIG_FILE" "$CONFIG_LOCAL"; do
        if [[ -f "$cfg" ]]; then
            cave_now="$(json_value tokens cave "$cfg" || true)"
            cave_now="$(printf '%s' "$cave_now" | tr -d '[:space:]')"
            [[ -n "$cave_now" ]] && break
        fi
    done
    if [[ -n "$neuprint_now" && "$neuprint_now" != "YOUR_NEUPRINT_TOKEN_HERE" ]]; then
        printf '%s\n' "✓ NeuPrint token already configured in config_local.json or config.json."
    else
        printf '%s\n' "⚠ NeuPrint token not configured - required for NeuPrint datasets."
    fi
    if [[ -n "$cave_now" && "$cave_now" != "YOUR_CAVE_TOKEN_HERE" ]]; then
        printf '%s\n' "✓ CAVE token already configured in config_local.json or config.json."
    else
        printf '%s\n' "ℹ CAVE token optional - only needed for FlyWire FAFB online fetching."
    fi
    printf '%s\n' "Set tokens in the UI Settings tab after launching, or edit config_local.json"
    printf '%s\n' "(repository root, format: tokens.neuprint / tokens.cave)."
}
configure_tokens
