# FlyWire/FAFB/BANC Dataset Usage Guide

This toolkit supports analysis of FlyWire (FAFB) and BANC datasets using a **local file-based workflow**. Due to the complexity and size of these datasets, we do not support direct API fetching. Instead, you must download the required data files from the Codex/FlyWire portal and place them in the correct directory. The toolkit will then automatically process and convert them for analysis.

## 1. Data Preparation

### Step 1: Download Data Files
You need to download the required CSV files from the Codex Download Page (or equivalent source).

**For FAFB (flywire_FAFB_v783):**
- See [FAFB Integration Guide](FAFB_INTEGRATION.md) for the full file list.

**For BANC (flywire_BANC_v626):**
- See [BANC Integration Guide](BANC_INTEGRATION.md) for the full file list.
- **Note:** Skeleton visualization is NOT available for BANC.

### Step 2: Place Files in Directory
Create the directory structure in your project folder and place the downloaded files there.

**For FAFB:**
```
datasets/
  └── flywire_FAFB_v783/
      └── downloads/
            ├── classification.csv.gz
            ├── names.csv.gz
            ├── coordinates.csv.gz
            ├── neurons.csv.gz
            ├── cell_stats.csv.gz
            ├── consolidated_cell_types.csv.gz
            ├── connections_princeton_no_threshold.csv.gz
            ├── fafb_v783_princeton_synapse_table.csv.gz  (optional)
            └── sk_lod1_783_healed.zip                    (optional)
```

**For BANC:**
```
datasets/
  └── flywire_BANC_v626/
      └── downloads/
            ├── neurons.csv.gz
            ├── connections_princeton.csv.gz
```

### Step 3: Run Conversion
You can run the conversion script manually, or it will run automatically the first time you try to use the dataset.

**Manual Conversion (FAFB):**
```bash
python src/FAFB_file_converter.py
```

**Manual Conversion (BANC):**
```bash
python src/BANC_file_converter.py
```

This script will:
1.  Read the CSV files from the `downloads` folder.
2.  Merge and enrich the neuron metadata.
3.  Convert the data into optimized Parquet files (`.parquet`) for fast loading.
4.  Save the processed files in the dataset folder.

## 2. Using FlyWire/BANC Data in Analysis

Once the data is prepared, you can use it just like any other NeuPrint dataset.

### Example: Finding Connections

```python
from coana import FindNeuronConnection

# Initialize connection finder
fc = FindNeuronConnection(
    token='dummy_token',  # Token is ignored for local files
    dataset='flywire_BANC_v626', # or 'flywire_FAFB_v783'
    sourceNeurons=['720575940621039145'],  # Use Root IDs
    targetNeurons=['720575940619419758'],
    min_synapse_num=5
)

# Run analysis
fc.InitializeNeuronInfo()
fc.FindDirectConnection()
```

### Example Script

We provide a ready-to-use script for FlyWire/BANC pathfinding: `scripts/FindPath_flywire.py`.

```bash
python scripts/FindPath_flywire.py
```

This script demonstrates:
- Setting up the `FindNeuronConnection` class.
- Finding paths between neuron types.
- Visualizing the results (Note: 3D skeletons only for FAFB).

### Example: Visualizing Skeletons (FAFB Only)

If you downloaded the `sk_lod1_783_healed.zip` file for FAFB, you can visualize 3D skeletons.

```python
from coana import VisualizeSkeleton

vs = VisualizeSkeleton(
    dataset='flywire_FAFB_v783',
    neuron_layers=['720575940621039145'],
    brain_mesh='template'  # Uses FAFB template
)

vs.plot_neurons()
```

**Note:** FlyWire-based BANC dataset does not currently support skeleton visualization.

## 3. Important Notes

-   **Storage**: The converted Parquet files are much smaller and faster than the raw CSVs, but the initial raw files can be large. Ensure you have enough disk space.
-   **Updates**: If you want to update the data, simply delete the files in the dataset folder and place new CSV files in the `downloads` folder. The converter will run again.
-   **IDs**: FlyWire/BANC use long integer Root IDs (e.g., `720575940...`). Ensure you use these IDs in your queries.
