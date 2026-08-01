# DROCAT Installation - Platform Notes & Known Issues

## Platform notes

### macOS

- Conda: Miniconda at `~/miniconda3` or Anaconda at `~/anaconda3`; both are auto-detected by `install.sh` and `DROCAT.command`.
- `DROCAT.command` (double-click) creates/activates the `drocat` env, installs dependencies, and launches the UI.
- Matplotlib "font cache" warnings on first import are benign.
- Opening the UI uses `python ui/app.py`; the browser opens automatically unless launched with `show=False`.

### Windows

- Use `install.ps1` (PowerShell, `-ExecutionPolicy Bypass`) or `install.bat`.
- `neuronbridge-python` may fail to install because of its `memray` dependency; `install.ps1` falls back to `--no-deps`. If it still fails, install it manually with `pip install neuronbridge-python --no-deps` and report that NeuronBridge tabs may be limited.
- Conda activation inside scripts can fail when conda is not initialized for the current shell; run from "Anaconda Prompt" or call `conda activate drocat` first.

### Linux

- Same flow as macOS (`bash install.sh`, `./run_ui.sh`).
- `xdg-open` is used to open output folders; install `xdg-utils` if missing.
- If `tkinter` is unavailable, native folder pickers in the UI fall back to typing paths (the UI still works).

## Known issues and how to handle them

- **Pinned dependency conflicts**: use the exact pins in `requirements.txt`; do not upgrade individual packages (numpy/pandas/polars/matplotlib versions are mutually pinned).
- **Token placeholders**: `token_info.txt` contains `YOUR_...` placeholders; the real tokens must live in the gitignored `token_info_local.txt`.
- **Missing datasets**: `datasets/` and `cache/` are created automatically; the first query downloads the full neuron table for the selected dataset (needs token + network, can take minutes). FlyWire FAFB/BANC require manual downloads (see the Settings tab guide in the UI).
- **UI port 8080 busy**: change `APP_PORT` in `ui/config.py` or stop the other process.
- **3D export requires Chrome**: PNG/video exports in `plot3dSkeleton` use Chrome + WebDriver (or the slower Kaleido fallback). Install Chrome if exports fail.
- **Sandboxed agents**: dependency downloads, token checks, and dataset fetches need network; request escalation when running inside a restricted sandbox.

## Verification recap

The bundled `scripts/verify_install.py` checks:

1. Project layout (installers, UI, backend modules, vispath subproject).
2. Python version (>= 3.9).
3. Core imports (`numpy`, `pandas`, `polars`, `scipy`, `matplotlib`, `seaborn`, `plotly`, `networkx`, `openpyxl`, `xlsxwriter`, `bokeh`, `requests`, `jinja2`, `PIL`, `pydantic`, `neuprint`, `nicegui`, `tqdm`, `cv2`, `reportlab`, `pptx`, `img2pdf`).
4. Optional imports (warn-only): `neuronbridge`, `caveclient`, `cloudvolume`, `navis`, `flybrains`, `selenium`, `webdriver_manager`, `fitz`, `rapidjson`, `psutil`, `trimesh`.
5. Token file with a non-placeholder `NEUPRINT_TOKEN`.
6. UI package import (`ui.app`).
