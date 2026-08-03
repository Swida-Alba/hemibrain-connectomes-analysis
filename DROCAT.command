#!/bin/zsh
# Double-click one-click installer for DROCAT v4.4.5 on macOS.

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

clear 2>/dev/null || true
echo "DROCAT v4.4.5 - Drosophila Connectome Analysis Toolkit"
echo "Preparing the versioned command-line environment..."
echo

"$ROOT/install.sh"
status=$?
echo
if [[ "$status" -eq 0 ]]; then
    echo "Installation complete. See README.md for standalone script examples."
else
    echo "Installation failed. Review the messages above."
fi
read "?Press Return to close."
exit "$status"
