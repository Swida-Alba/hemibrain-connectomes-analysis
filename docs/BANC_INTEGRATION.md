# BANC (FlyWire) Data Integration

This project supports the BANC (Brain Analysis of Neuronal Connectivity) dataset from FlyWire using a local file-based approach for high performance.

Use the exact dataset identifier that matches the downloaded release:
`flywire_BANC_v626` or `flywire_BANC_v888`. Keep each release in its own
`datasets/<dataset>/` directory; do not mix versions. BANC is local-file only
in this toolkit and does not support CAVE API fetching.

## Data Preparation

We provide a dedicated script to prepare the BANC data for analysis. This script handles file organization, format conversion (to Parquet), and data enrichment.

### 1. Download Data

Download the following files from [FlyWire Codex Downloads](https://codex.flywire.ai/api/download?dataset=banc):

**Required Files:**
*   `neurons.csv.gz` (Neuron Metadata)
*   `connections_princeton.csv.gz` (Connectivity)

*Note: The converter requires both files to build a usable local database. If
either is missing, it will stop and print the download location.*

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

The direct command defaults to `flywire_BANC_v626`. For v888, pass the
dataset name through the converter function:

```bash
python -c "import sys; sys.path.insert(0, 'src'); from BANC_file_converter import ensure_banc_data; d='flywire_BANC_v888'; ensure_banc_data(d, 'datasets/' + d)"
```

**Option B: Run any analysis script**
Simply selecting the matching dataset in a tool and running it will
automatically check for and convert the data if needed.

The script will check for the required files and guide you if anything is missing.

**What the script does:**
1.  Creates the `datasets/<dataset>` directory structure.
2.  Checks the `datasets/<dataset>/downloads` folder for source files.
3.  If files are missing, it prints a list of what to download and where to put them.
4.  Converts CSVs to optimized **Parquet** files for fast loading.
5.  Merges metadata into a single neuron dataframe.
6.  Aggregates connection weights across ROIs.

### 3. Cleanup (Removable Files)

After the script successfully completes (look for "✓ Conversion complete" messages), you can safely delete the entire `downloads` folder to save space.

**Removable Folder:**
*   `datasets/<dataset>/downloads/` (The entire folder can be deleted)

**Do NOT delete:**
*   The generated `.parquet` files in `datasets/<dataset>/`.

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
