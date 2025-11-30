# BANC (FlyWire) Data Integration

This project supports the BANC (Brain Analysis of Neuronal Connectivity) dataset from FlyWire using a local file-based approach for high performance.

## Data Preparation

We provide a dedicated script to prepare the BANC data for analysis. This script handles file organization, format conversion (to Parquet), and data enrichment.

### 1. Download Data

Download the following files from [FlyWire Codex Downloads](https://codex.flywire.ai/api/download?dataset=banc):

**Required Files:**
*   `neurons.csv.gz` (Neuron Metadata)
*   `connections_princeton.csv.gz` (Connectivity)

*Note: The converter script requires these files to build a complete neuron database. If any are missing, the script will stop and ask you to download them.*

**Important Note on Visualization:**
*   **Skeleton Visualization is NOT available for BANC.** The skeleton data (`sk_lod1_783_healed.zip` or similar) is not currently available for the BANC dataset on the FlyWire Codex.
*   Pathfinding, network visualization, and heatmap analysis are fully supported.
*   3D skeleton rendering will be skipped automatically.

### 2. Run the Converter

You can trigger the data preparation in two ways:

**Option A: Run the converter directly (Recommended)**
```bash
# Run from the project root
python src/BANC_file_converter.py
```

**Option B: Run any analysis script**
Simply running any script that initializes the dataset (e.g., `FindPath_flywire.py` with `dataset='flywire_BANC_v626'`) will automatically check for and convert the data if needed.

The script will check for the required files and guide you if anything is missing.

**What the script does:**
1.  Creates the `datasets/flywire_BANC_v626` directory structure.
2.  Checks the `datasets/flywire_BANC_v626/downloads` folder for source files.
3.  If files are missing, it prints a list of what to download and where to put them.
4.  Converts CSVs to optimized **Parquet** files for fast loading.
5.  Merges metadata into a single neuron dataframe.
6.  Aggregates connection weights across ROIs.

### 3. Cleanup (Removable Files)

After the script successfully completes (look for "✓ Conversion complete" messages), you can safely delete the entire `downloads` folder to save space.

**Removable Folder:**
*   `datasets/flywire_BANC_v626/downloads/` (The entire folder can be deleted)

**Do NOT delete:**
*   The generated `.parquet` files in `datasets/flywire_BANC_v626/`.

## Usage

### Path Finding

Use `FindNeuronConnection` with the BANC dataset name.

```python
from coana import FindNeuronConnection

fnc = FindNeuronConnection()
fnc.dataset = 'flywire_BANC_v626'

# Use BANC Root IDs (as strings)
fnc.sourceNeurons = ['720575940596125868'] 
fnc.targetNeurons = ['720575940597856265']

fnc.InitializeNeuronInfo()
fnc.FindAllPath()
```

## Notes

*   **Root IDs**: BANC root IDs are very large integers. The system handles them as strings internally to avoid precision loss.
*   **Column Mapping**: The converter automatically maps BANC-specific columns (e.g., `Primary Cell Type` -> `type`, `Root ID` -> `bodyId`) to the standard format used by this toolkit.
*   **Visualization Restrictions**: As mentioned above, 3D skeleton visualization is disabled for BANC due to missing data.
