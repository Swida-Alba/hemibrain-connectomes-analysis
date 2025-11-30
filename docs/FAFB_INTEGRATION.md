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
from coana import FindNeuronConnection

fnc = FindNeuronConnection()
fnc.dataset = 'flywire_FAFB_v783'

# Use FAFB Root IDs (as integers or strings)
fnc.sourceNeurons = [720575940596125868] 
fnc.targetNeurons = [720575940597856265]

fnc.InitializeNeuronInfo()
fnc.FindAllPath()
```

### Visualization

You can visualize skeletons using `Vis3S` (or `VisualizeSkeleton`).

```python
import statvis as sv
import pandas as pd

# Create a DataFrame with neurons to visualize
df = pd.DataFrame({
    'bodyId': ['720575940596125868'],
    'type': ['T5c']
})

sv.Vis3S(
    df,
    toPlot='skeleton',
    dataset='flywire_FAFB_v783',
    showfig=True
)
```

## Notes

*   **Root IDs**: FAFB root IDs are very large integers. The system handles them as strings internally to avoid precision loss, but you can pass them as integers in your scripts.
*   **Caching**: The caching system currently produces warnings for FAFB IDs due to their size, but this does not affect the analysis results.
*   **Skeletons**: Skeleton visualization requires the `sk_lod1_783_healed.zip` file. Only neurons present in this zip file can be visualized as skeletons.
