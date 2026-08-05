#!/usr/bin/env bash
# Self-healing DROCAT UI launcher for macOS/Linux.

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DROCAT_VERSION=""
if [[ -f "$SCRIPT_DIR/ui/config.py" ]]; then
    DROCAT_VERSION="$(sed -n 's/^APP_VERSION = "\([^"]*\)"/\1/p' "$SCRIPT_DIR/ui/config.py" | head -1)"
fi
if [[ -z "$DROCAT_VERSION" ]]; then
    DROCAT_VERSION="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$SCRIPT_DIR/pyproject.toml" | head -1)"
fi
DROCAT_VERSION="${DROCAT_VERSION:-4.5.0}"
ENV_BASE="drocat-${DROCAT_VERSION}"

# --- Port-conflict guard -------------------------------------------------
# If the UI is already running (e.g. a previous launch is still up), open it
# in the browser instead of crashing with "address already in use".
APP_PORT="${DROCAT_UI_PORT:-$(sed -n 's/^APP_PORT = \([0-9][0-9]*\)/\1/p' "$SCRIPT_DIR/ui/config.py" | head -1)}"
APP_PORT="${APP_PORT:-8080}"

port_in_use() {
    if command -v lsof >/dev/null 2>&1; then
        lsof -ti "tcp:$APP_PORT" -sTCP:LISTEN >/dev/null 2>&1
    elif command -v nc >/dev/null 2>&1; then
        nc -z 127.0.0.1 "$APP_PORT" >/dev/null 2>&1
    else
        (exec 3<>"/dev/tcp/127.0.0.1/$APP_PORT") >/dev/null 2>&1
    fi
}

if port_in_use; then
    owner_pid="$(lsof -ti "tcp:$APP_PORT" -sTCP:LISTEN 2>/dev/null | head -1)"
    owner_cmd="$(ps -p "$owner_pid" -o command= 2>/dev/null || true)"
    # A DROCAT process owns the port (still booting or fully up).
    if echo "$owner_cmd" | grep -qE "ui/app\.py|drocat"; then
        printf 'DROCAT is already running at http://127.0.0.1:%s\n' "$APP_PORT"
        printf 'Opening it in your browser...\n'
        if command -v open >/dev/null 2>&1; then
            open "http://127.0.0.1:$APP_PORT"
        fi
        exit 0
    fi
    # Otherwise probe the page to rule out a non-DROCAT server on the port.
    if curl -s --max-time 2 "http://127.0.0.1:$APP_PORT/" 2>/dev/null | grep -q "drocat-cobalt"; then
        printf 'DROCAT is already running at http://127.0.0.1:%s\n' "$APP_PORT"
        printf 'Opening it in your browser...\n'
        if command -v open >/dev/null 2>&1; then
            open "http://127.0.0.1:$APP_PORT"
        fi
        exit 0
    fi
    printf 'ERROR: Port %s is already in use by another application.\n' "$APP_PORT" >&2
    printf 'Close that application, or start the UI with DROCAT_UI_PORT=<free port>.\n' >&2
    exit 1
fi

find_conda() {
    if command -v conda >/dev/null 2>&1; then
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
        [[ -x "$candidate" ]] && { printf '%s\n' "$candidate"; return; }
    done
}

resolve_env() {
    local index candidate
    ENV_NAME=""
    for index in $(seq 0 20); do
        if [[ "$index" -eq 0 ]]; then
            candidate="$ENV_BASE"
        else
            candidate="${ENV_BASE}-$((index + 1))"
        fi
        if "$CONDA_BIN" run -n "$candidate" python -c \
            'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)' \
            >/dev/null 2>&1; then
            ENV_NAME="$candidate"
            return
        fi
    done
}

CONDA_BIN="$(find_conda || true)"
if [[ -z "$CONDA_BIN" ]]; then
    printf '%s\n' "Conda is not installed; starting the one-click installer."
    "$SCRIPT_DIR/install.sh"
    CONDA_BIN="$(find_conda || true)"
fi
[[ -n "$CONDA_BIN" ]] || { printf '%s\n' "ERROR: Conda installation failed." >&2; exit 1; }

resolve_env
if [[ -z "$ENV_NAME" ]]; then
    "$SCRIPT_DIR/install.sh"
    resolve_env
fi
[[ -n "$ENV_NAME" ]] || { printf '%s\n' "ERROR: no usable $ENV_BASE environment was found." >&2; exit 1; }

# An older environment can have importable packages while still containing
# incompatible distributions. Repair it through the same pinned installer.
if ! "$CONDA_BIN" run -n "$ENV_NAME" python -c \
    'import nicegui, numpy, pandas, neuprint, neuronbridge_client' >/dev/null 2>&1 \
    || ! "$CONDA_BIN" run -n "$ENV_NAME" python -m pip check >/dev/null 2>&1; then
    printf '%s\n' "Repairing dependencies in $ENV_NAME..."
    "$SCRIPT_DIR/install.sh"
    resolve_env
fi

printf 'Starting DROCAT v%s in %s...\n' "$DROCAT_VERSION" "$ENV_NAME"
cd "$SCRIPT_DIR"
exec "$CONDA_BIN" run -n "$ENV_NAME" --no-capture-output python ui/app.py
