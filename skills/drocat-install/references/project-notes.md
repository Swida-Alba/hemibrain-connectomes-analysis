# DROCAT Installation - Platform Notes & Known Issues

## Platform notes

### macOS

- Conda: Miniconda at `~/miniconda3` or Anaconda at `~/anaconda3`; both are auto-detected by `archive/install/install.sh` and `run_DROCAT.command`.
- `run_DROCAT.command` (double-click) repairs/creates the versioned environment through the same installer used by the terminal launcher, then launches the UI.
- Matplotlib "font cache" warnings on first import are benign.
- Opening the UI uses `python ui/app.py`; set `DROCAT_UI_SHOW=0` for a headless launch.

### Windows

- Use `archive/install/install.ps1` (PowerShell, `-ExecutionPolicy Bypass`) or `archive/install/install.bat`.
- Do not install the upstream `neuronbridge-python` distribution. DROCAT ships its own API client so the Windows install has no `memray`, Ray, or conflicting Pydantic requirement.
- Installers and launchers use `conda run`; shell initialization is not required.

### Linux

- Same flow as macOS (`bash archive/install/install.sh`, `./run_DROCAT.command`).
- `xdg-open` is used to open output folders; install `xdg-utils` if missing.
- If `tkinter` is unavailable, native folder pickers in the UI fall back to typing paths (the UI still works).

## Known issues and how to handle them

- **Pinned dependency conflicts**: use the exact pins in `requirements.txt`, `requirements-windows.txt`, and `ui/requirements.txt`; do not upgrade individual packages. A successful install must pass `python -m pip check`.
- **Requests warning about chardet**: repair the environment by rerunning the installer. DROCAT pins `chardet==5.2.0`; user-site packages are disabled with `PYTHONNOUSERSITE=1`.
- **Token placeholders**: `config.example.json` ships empty token values; the real tokens must live in the gitignored `config.json`. The legacy `token_info*.txt` files are deprecated and no longer read (token_info.txt documents the migration).
- **Missing datasets**: `datasets/` and `cache/` are created automatically; the first query downloads the full neuron table for the selected dataset (needs token + network, can take minutes). FlyWire FAFB/BANC require manual downloads (see the Settings tab guide in the UI).
- **UI port 8080 busy**: set `DROCAT_UI_PORT` to a free port (for example,
  `DROCAT_UI_PORT=8081 ./run_DROCAT.command`) or stop the other process.
- **3D export requires Chrome**: PNG/video exports in `plot3dSkeleton` use Chrome + WebDriver (or the slower Kaleido fallback). Install Chrome if exports fail.
- **Sandboxed agents**: dependency downloads, token checks, and dataset fetches need network; request escalation when running inside a restricted sandbox.

## Verification recap

The bundled `scripts/verify_install.py` checks:

1. Project layout (installers, UI, backend modules, vispath subproject).
2. Python version (3.10-3.11).
3. Core imports (`numpy`, `pandas`, `polars`, `scipy`, `matplotlib`, `seaborn`, `plotly`, `networkx`, `openpyxl`, `xlrd`, `xlsxwriter`, `bokeh`, `requests`, `jinja2`, `PIL`, `pydantic`, `neuprint`, `nicegui`, `tqdm`, `cv2`, `reportlab`, `pptx`, `img2pdf`).
4. Bundled NeuronBridge client import plus optional integrations (warn-only): `caveclient`, `cloudvolume`, `navis`, `flybrains`, `selenium`, `webdriver_manager`, `fitz`, `psutil`, `trimesh`.
5. Installed versions against the platform and UI requirement manifests.
6. `pip check` dependency consistency.
7. Token file with a non-placeholder `NEUPRINT_TOKEN` (advisory unless `--require-token` is used).
8. UI package import (`ui.app`).
