#!/bin/zsh
# Double-click installer/launcher for macOS.

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

clear 2>/dev/null || true
echo "DROCAT - Drosophila Connectome Analysis Toolkit"
echo "Preparing the versioned environment and launching the UI..."
echo

"$ROOT/run_ui.sh"
status=$?
if [[ "$status" -ne 0 ]]; then
    echo
    echo "DROCAT could not start. Review the messages above."
    read "?Press Return to close."
    exit "$status"
fi
