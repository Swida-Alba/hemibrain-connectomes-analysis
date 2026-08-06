#!/usr/bin/env bash
# Double-click / terminal launcher for macOS and Linux.
# Self-healing: prepares the versioned environment on first run (via
# archive/install/install.sh), repairs it when inconsistent, resolves port
# conflicts interactively, and launches the web UI.

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALLER="$SCRIPT_DIR/archive/install/install.sh"

clear 2>/dev/null || true
printf '%s\n' "DROCAT - Drosophila Connectome Analysis Toolkit"
printf '%s\n' "Preparing the versioned environment and launching the UI..."
printf '%s\n' ""

main() {
    DROCAT_VERSION=""
    if [[ -f "$SCRIPT_DIR/ui/config.py" ]]; then
        DROCAT_VERSION="$(sed -n 's/^APP_VERSION = "\([^"]*\)"/\1/p' "$SCRIPT_DIR/ui/config.py" | head -1)"
    fi
    if [[ -z "$DROCAT_VERSION" ]]; then
        DROCAT_VERSION="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$SCRIPT_DIR/pyproject.toml" | head -1)"
    fi
    DROCAT_VERSION="${DROCAT_VERSION:-4.5.0}"
    ENV_BASE="drocat-${DROCAT_VERSION}"

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
        printf '%s\n' "Conda is not installed; running the one-click installer."
        "$INSTALLER"
        CONDA_BIN="$(find_conda || true)"
    fi
    [[ -n "$CONDA_BIN" ]] || { printf '%s\n' "ERROR: Conda installation failed." >&2; return 1; }

    resolve_env
    if [[ -z "$ENV_NAME" ]]; then
        printf '%s\n' "Environment $ENV_BASE not found; running the one-click installer."
        "$INSTALLER"
        resolve_env
    fi
    [[ -n "$ENV_NAME" ]] || { printf '%s\n' "ERROR: no usable $ENV_BASE environment was found." >&2; return 1; }

    # An older environment can have importable packages while still containing
    # incompatible distributions. Repair it through the same pinned installer.
    if ! "$CONDA_BIN" run -n "$ENV_NAME" python -c \
        'import nicegui, numpy, pandas, neuprint, neuronbridge_client' >/dev/null 2>&1 \
        || ! "$CONDA_BIN" run -n "$ENV_NAME" python -m pip check >/dev/null 2>&1; then
        printf '%s\n' "Repairing dependencies in $ENV_NAME..."
        "$INSTALLER"
        resolve_env
    fi

    # --- Token hint -----------------------------------------------------
    # Remind users who skipped the installer prompt that tokens can be set
    # later in the UI Settings tab or in token_info_local.txt.
    local neuprint_token=""
    [[ -f "$SCRIPT_DIR/token_info_local.txt" ]] && \
        neuprint_token="$(sed -n "s/^NEUPRINT_TOKEN='\([^']*\)'/\1/p" "$SCRIPT_DIR/token_info_local.txt" | head -1)"
    if [[ -z "$neuprint_token" || "$neuprint_token" == "YOUR_NEUPRINT_TOKEN_HERE" ]]; then
        printf '%s\n' "Tip: NeuPrint/CAVE tokens are not configured yet - set them in the UI Settings tab or in token_info_local.txt."
    fi

    # --- Port-conflict guard ---------------------------------------------
    # If the UI port is already in use, ask the user whether to start on a new
    # port, kill the existing DROCAT process and restart, or cancel. When the
    # script is not interactive (agents, CI), keep the previous automatic
    # behavior: open the browser if a DROCAT instance owns the port, otherwise
    # fail with a hint.
    APP_PORT="${DROCAT_UI_PORT:-$(sed -n 's/^APP_PORT = \([0-9][0-9]*\)/\1/p' "$SCRIPT_DIR/ui/config.py" | head -1)}"
    APP_PORT="${APP_PORT:-8080}"

    port_in_use() {
        local port="${1:-$APP_PORT}"
        if command -v lsof >/dev/null 2>&1; then
            lsof -ti "tcp:$port" -sTCP:LISTEN >/dev/null 2>&1
        elif command -v nc >/dev/null 2>&1; then
            nc -z 127.0.0.1 "$port" >/dev/null 2>&1
        else
            (exec 3<>"/dev/tcp/127.0.0.1/$port") >/dev/null 2>&1
        fi
    }

    is_drocat_owner() {
        local pid="$1" cmd
        [[ -n "$pid" ]] || return 1
        cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
        echo "$cmd" | grep -qE "ui/app\.py|drocat"
    }

    launch_ui() {
        local port="${1:-$APP_PORT}"
        printf 'Starting DROCAT v%s in %s at http://127.0.0.1:%s...\n' "$DROCAT_VERSION" "$ENV_NAME" "$port"
        cd "$SCRIPT_DIR"
        export DROCAT_UI_PORT="$port"
        exec "$CONDA_BIN" run -n "$ENV_NAME" --no-capture-output python ui/app.py
    }

    if port_in_use; then
        owner_pid="$(lsof -ti "tcp:$APP_PORT" -sTCP:LISTEN 2>/dev/null | head -1)"
        owner_cmd="$(ps -p "$owner_pid" -o command= 2>/dev/null || true)"
        if [[ -t 0 ]]; then
            # Interactive: let the user decide what to do with the busy port.
            while true; do
                printf '\nPort %s is already in use by PID %s:\n    %s\n' "$APP_PORT" "$owner_pid" "$owner_cmd"
                printf '  [1] Start DROCAT on a new port\n'
                if is_drocat_owner "$owner_pid"; then
                    printf '  [2] Kill the existing DROCAT process and restart on port %s\n' "$APP_PORT"
                else
                    printf '  [2] Not allowed: the process on port %s is not DROCAT - stop it manually, then retry\n' "$APP_PORT"
                fi
                printf '  [3] Cancel\n'
                printf 'Your choice [1-3]: '
                read -r choice || break
                case "$choice" in
                    1)
                        new_port="$APP_PORT"
                        while port_in_use "$new_port"; do
                            new_port=$((new_port + 1))
                            [[ "$new_port" -le 65535 ]] || { printf 'No free port found.\n' >&2; return 1; }
                        done
                        printf 'Port %s is busy; starting on port %s instead.\n' "$APP_PORT" "$new_port"
                        launch_ui "$new_port"
                        ;;
                    2)
                        if is_drocat_owner "$owner_pid"; then
                            printf 'Stopping PID %s...\n' "$owner_pid"
                            kill "$owner_pid" 2>/dev/null || true
                            for _ in $(seq 1 20); do
                                port_in_use || break
                                sleep 0.5
                            done
                            if port_in_use; then
                                printf 'ERROR: the process on port %s did not stop.\n' "$APP_PORT" >&2
                                return 1
                            fi
                            launch_ui "$APP_PORT"
                        else
                            printf 'The process on port %s is not DROCAT; it will not be killed automatically.\n' "$APP_PORT"
                        fi
                        ;;
                    3)
                        printf 'Cancelled.\n'
                        return 1
                        ;;
                    *)
                        printf 'Invalid choice.\n'
                        ;;
                esac
            done
            return 1
        else
            # Non-interactive (agent, CI): previous automatic behavior.
            if is_drocat_owner "$owner_pid"; then
                printf 'DROCAT is already running at http://127.0.0.1:%s\n' "$APP_PORT"
                printf 'Opening it in your browser...\n'
                if command -v open >/dev/null 2>&1; then
                    open "http://127.0.0.1:$APP_PORT"
                fi
                return 0
            fi
            # Otherwise probe the page to rule out a non-DROCAT server on the port.
            if curl -s --max-time 2 "http://127.0.0.1:$APP_PORT/" 2>/dev/null | grep -q "drocat-cobalt"; then
                printf 'DROCAT is already running at http://127.0.0.1:%s\n' "$APP_PORT"
                printf 'Opening it in your browser...\n'
                if command -v open >/dev/null 2>&1; then
                    open "http://127.0.0.1:$APP_PORT"
                fi
                return 0
            fi
            printf 'ERROR: Port %s is already in use by another application (PID %s).\n' "$APP_PORT" "$owner_pid" >&2
            printf 'Stop it manually, run with DROCAT_UI_PORT=<free port>, or run interactively to choose.\n' >&2
            return 1
        fi
    fi

    printf 'Starting DROCAT v%s in %s...\n' "$DROCAT_VERSION" "$ENV_NAME"
    launch_ui "$APP_PORT"
}

main || rc=$?
if [[ "${rc:-0}" -ne 0 ]]; then
    printf '\n%s\n' "DROCAT could not start. Review the messages above."
    if [[ -t 0 ]]; then
        read -r -p "Press Return to close." _
    fi
    exit "$rc"
fi
