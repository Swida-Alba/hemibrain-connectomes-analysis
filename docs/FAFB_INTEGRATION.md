# FAFB (FlyWire) Data Integration

This project supports the FAFB (Full Adult Fly Brain) dataset from FlyWire using a local file-based approach for high performance.

## Data Preparation

We provide a dedicated script to prepare the FAFB data for analysis. This script handles file organization, format conversion (to Parquet), and data enrichment.

### 1. Download Data

Download the following files from [FlyWire Codex Downloads](https://codex.flywire.ai/api/download?dataset=fafb):

**Required Files:**
*   `classification.csv.gz` (Neuron Classification) - **~1 MB**
*   `connections_princeton_no_threshold.csv.gz` (Connectivity) - **~263 MB**
    *   *Note: `connections_princeton.csv.gz` or `connections.csv.gz` are also accepted as fallbacks.*
*   `consolidated_cell_types.csv.gz` (Consolidated Cell Types) - **~1 MB**
*   `names.csv.gz` (Neuron Names) - **~1 MB**
*   `coordinates.csv.gz` (Soma Coordinates) - **~5 MB**
*   `neurons.csv.gz` (Neurotransmitters) - **~2 MB**
*   `cell_stats.csv.gz` (Cell Statistics) - **~2.5 MB**

*Note: The converter script requires ALL the above files to build a complete neuron database. If any are missing, the script will stop and ask you to download them.*

**Optional Visualization Files:**
*   `sk_lod1_783_healed.zip` (Skeletons - Required for 3D skeleton visualization) - **~13 GB**
*   `fafb_v783_princeton_synapse_table.csv.gz` (Synapses - Required for synapse visualization) - **~2.5 GB**

**Storage Summary:**
*   **Minimal Analysis:** ~300 MB download → ~200 MB final size
*   **With Synapses:** +2.5 GB download → +1.7 GB final size
*   **With Skeletons:** +13 GB download → +13 GB final size
*   **Full Dataset:** ~16 GB total storage required

### 2. Run the Converter

You can trigger the data preparation in two ways:

**Option A: Run the converter directly (Recommended)**
```bash
# Run from the project root
python src/FAFB_file_converter.py
```

**Option B: Run any analysis script**
Simply running any script that initializes the dataset (e.g., `FindPath_flywire.py` or initializing `FindNeuronConnection` with `dataset='flywire_FAFB_v783'`) will automatically check for and convert the data if needed.

The script will check for the required files and guide you if anything is missing.

**What the script does:**
1.  Creates the `datasets/flywire_FAFB_v783` directory structure.
2.  Checks the `datasets/flywire_FAFB_v783/downloads` folder for source files.
3.  If files are missing, it prints a list of what to download and where to put them.
4.  Converts CSVs to optimized **Parquet** files for fast loading.
5.  Merges metadata (names, coordinates, types) into a single neuron dataframe.
6.  Moves the skeleton zip file to the correct location.

### 3. Cleanup (Removable Files)

After the script successfully completes (look for "✓ Conversion complete" messages), you can safely delete the entire `downloads` folder to save space.

**Removable Folder:**
*   `datasets/flywire_FAFB_v783/downloads/` (The entire folder can be deleted)

**Do NOT delete:**
*   The generated `.parquet` files in `datasets/flywire_FAFB_v783/`.
*   The `sk_lod1_783_healed.zip` file in `datasets/flywire_FAFB_v783/` (The converter moves this file out of downloads, so it is safe).

## Usage

### Path Finding

Use `FindNeuronConnection` with the FAFB dataset name.

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from coana import FindNeuronConnection

fnc = FindNeuronConnection(
    dataset='flywire_FAFB_v783',
    sourceNeurons=['l-LNv.*'],  # Regex patterns supported
    targetNeurons=['s-LNv.*'],
    max_interlayer=2,
    min_synapse_num=5,
    verbose_mode='simple',
)

fnc.InitializeNeuronInfo()
fnc.FindAllPath()
```

### Visualization

Visualize skeletons using `VisualizeSkeleton` (or the shorthand `Vis3S`).

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from coana import VisualizeSkeleton

vs = VisualizeSkeleton(
    dataset='flywire_FAFB_v783',
    token='',  # Not needed for FAFB local data
    output_dir='/path/to/output',
    neuron_layers=['l-LNv'],  # Neuron types or bodyIds
    skip_synapse=True,
    neuron_alpha=0.2,
    min_synapse_num=3,
    skeleton_mode='tube',
    merge_neurons=True,
    show_fig=True,
    brain_mesh='template',  # Use native FAFB coordinates
    cache_neurons=True,
)

vs.plot_neurons()
```

### CAVE API Fetching (force_API_fetching)

For more up-to-date skeleton data, you can fetch skeletons directly from the CAVE API instead of using the local ZIP file:

```python
from coana import VisualizeSkeleton

vs = VisualizeSkeleton(
    dataset='flywire_FAFB_v783',
    output_dir='/path/to/output',
    neuron_layers=['l-LNv'],
    skip_synapse=True,
    neuron_alpha=0.2,
    skeleton_mode='tube',
    merge_neurons=True,
    show_fig=True,
    brain_mesh='template',
    cache_neurons=True,
    force_API_fetching=True,  # Fetch from CAVE API instead of local ZIP
)

vs.plot_neurons()
```

**Key Features:**
- **API Cache**: When `force_API_fetching=True`, skeletons fetched via API are cached locally in `cache/{dataset}/API_cache/skeletons/`. On subsequent runs, cached skeletons are loaded first before fetching new ones.
- **Local ZIP Mode**: When `force_API_fetching=False` (default), only local ZIP data is used. The system will NOT check API cache - this ensures consistency with the downloaded dataset.
- **Automatic Fallback**: If `force_API_fetching=False` and the local ZIP is missing or empty, the system will automatically fall back to API fetching as a last resort.
- **Updated Data**: Use `force_API_fetching=True` to ensure you're using the most up-to-date neuron morphologies from the CAVE API.

### Fixing Skeleton Extrusion Issues

The downloaded FAFB skeleton ZIP (`sk_lod1_783_healed.zip`) may contain neurons with extrusion artifacts (mesh errors that appear as spikes or protrusions). You can fix specific neurons by fetching fresh skeletons via CAVE API:

```python
from coana import VisualizeSkeleton

# Method 1: Fix specific neurons by fetching them via API
# Run with force_API_fetching=True for the problematic neurons only
vs = VisualizeSkeleton(
    dataset='flywire_FAFB_v783',
    neuron_layers=[720575940596125868, 720575940597856265],  # Problematic bodyIds
    force_API_fetching=True,  # Fetch fresh data from CAVE API
    show_fig=False,  # Just cache the fixed skeletons
    cache_neurons=True,
)
vs.plot_neurons()  # This caches the API-fetched skeletons

# Method 2: Once fixed, run your full visualization
# API-cached skeletons are automatically prioritized over ZIP data
vs2 = VisualizeSkeleton(
    dataset='flywire_FAFB_v783',
    neuron_layers=['l-LNv', 's-LNv'],  # Mix of neurons
    force_API_fetching=False,  # Use local data, but API cache takes priority
    show_fig=True,
    brain_mesh='template',
)
vs2.plot_neurons()
```

**How Extrusion Fixes Work:**
1. Neurons fetched via API are cached in `cache/{dataset}/API_cache/skeletons/`
2. **VisualizeSkeleton** ALWAYS checks API cache first, even when `force_API_fetching=False`
3. This allows you to selectively fix problematic neurons without re-downloading the entire 13GB ZIP
4. Fixed neurons persist across sessions via the cache

**Note on force_API_fetching Behavior:**
- **VisualizeSkeleton**: Prioritizes API cache even when `force_API_fetching=False` (for extrusion fixes)
- **FindNeuronConnection**: Uses API only when `force_API_fetching=True` (for consistency with local data)

**Requirements:**
- CAVE token (obtain from https://global.daf-apis.com/auth/api/v1/create_token)
- Set token in `token_info_local.txt` or as environment variable `CAVE_TOKEN`

**Note:** BANC dataset does not support `force_API_fetching` due to API access restrictions (requires community membership at brain-and-nerve-cord.org).

## Notes

*   **Root IDs**: FAFB root IDs are very large integers. The system handles them as strings internally to avoid precision loss, but you can pass them as integers in your scripts.
*   **Caching**: The caching system currently produces warnings for FAFB IDs due to their size, but this does not affect the analysis results.
*   **Skeletons**: Skeleton visualization requires either the `sk_lod1_783_healed.zip` file or CAVE API access (via `force_API_fetching=True`). VisualizeSkeleton always prioritizes API-cached skeletons over ZIP data.
*   **Extrusion Issues**: The downloaded `sk_lod1_783_healed.zip` may contain neurons with extrusion artifacts (mesh errors appearing as spikes). Use `VisualizeSkeleton.fix_fafb_extrusions([bodyId1, bodyId2, ...])` to fetch fresh skeletons for problematic neurons. Once cached, they will be automatically used instead of ZIP versions.
