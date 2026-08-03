# DROCAT v4.4.5 Installation Notes

## Platform notes

### macOS and Linux

- Run `bash install.sh`; on macOS, `DROCAT.command` is the double-click wrapper.
- Conda is detected in PATH and common Miniconda/Anaconda/Miniforge locations.
- Matplotlib font-cache messages on first import are harmless.
- `PlotPath.py` may need a working GUI session for native file dialogs.

### Windows

- Use `install.bat` or `install.ps1`.
- The bundled NeuronBridge client eliminates the former Memray/Ray workaround.
- Installers use `conda run`, so Conda shell initialization is not required.

## Known issues

- Supported Python is 3.10-3.11; one-click setup uses 3.11.
- A successful release environment must report `No broken requirements found` from `python -m pip check`.
- If Requests warns about an unsupported character detector, rerun the installer; this release pins `chardet==5.2.0`.
- Real tokens belong only in gitignored `token_info_local.txt`.
- First online queries create/download dataset caches. FlyWire FAFB/BANC also require their documented local files.
- Chrome/WebDriver is required for WebGL PNG/video exports.

## Verifier coverage

The verifier checks:

1. One-click installers, standalone scripts, backend modules, and VisPath layout.
2. Python 3.10-3.11.
3. Core imports, including the XLS/XLSX readers and `neuronbridge_client`.
4. Optional integration imports (warn-only).
5. Every installed version against the pinned platform manifest.
6. Backend imports with `src/` on `sys.path`.
7. A complete `pip check` with no ignored conflicts.
8. Token configuration (advisory unless `--require-token` is passed).
