# DROCAT v4.4.3 Installation - Platform Notes & Known Issues

## Platform notes

### macOS

- Conda: Miniconda at `~/miniconda3` or Anaconda at `~/anaconda3`; either works with the conda commands in SKILL.md.
- Matplotlib "font cache" warnings on first import are benign.
- `scripts/PlotPath.py` may open native dialogs (tkinter) for file/sheet selection; a working GUI session is required for those prompts.

### Windows

- Use `requirements-windows.txt` instead of `requirements.txt`.
- `neuronbridge-python` may fail to install because of its `memray` dependency; install with `pip install neuronbridge-python --no-deps`. If it still fails, report that NeuronBridge scripts may be limited.
- Conda activation can fail when conda is not initialized for the current shell; run from "Anaconda Prompt" or call `conda activate drocat` first.

### Linux

- Same flow as macOS (`pip install -r requirements.txt`).
- If `tkinter` is unavailable, `PlotPath.py`'s native sheet picker will not open; pass `sheet_name` explicitly in the script instead.

## Known issues and how to handle them

- **Pinned dependency conflicts**: use the exact pins in `requirements.txt`; do not upgrade individual packages (numpy/pandas/polars/matplotlib versions are mutually pinned).
- **Token placeholders**: `token_info.txt` contains `YOUR_...` placeholders; the real tokens must live in the gitignored `token_info_local.txt` (it takes precedence over `token_info.txt`).
- **Missing datasets**: `datasets/` and `cache/` are created automatically; the first query downloads the full neuron table for the selected dataset (needs token + network, can take minutes). FlyWire FAFB/BANC require manual downloads.
- **3D export requires Chrome**: PNG/video exports in `plot3dSkeleton.py` use Chrome + WebDriver (or the slower Kaleido fallback). Install Chrome if exports fail.
- **Sandboxed agents**: dependency downloads, token checks, and dataset fetches need network; request escalation when running inside a restricted sandbox.

## Verification recap

The bundled `scripts/verify_install.py` checks:

1. Project layout (scripts/, src/, requirements, vispath subproject).
2. Python version (>= 3.9).
3. Core imports (numpy, pandas, polars, scipy, matplotlib, seaborn, plotly, networkx, openpyxl, xlsxwriter, bokeh, requests, jinja2, PIL, pydantic, neuprint, tqdm, cv2, reportlab, pptx, img2pdf, neuronbridge).
4. Optional imports (warn-only): caveclient, cloudvolume, navis, flybrains, selenium, webdriver_manager, fitz, rapidjson, psutil, trimesh.
5. Backend module imports with `src/` on `sys.path` (coana, statvis, neuronbridge_finder, visualize_skeleton, comparison.*, core.fast_graph).
6. Token file with a non-placeholder `NEUPRINT_TOKEN`.
