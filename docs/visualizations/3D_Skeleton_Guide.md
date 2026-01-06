# 3D Skeleton Visualization Guide

Three-dimensional rendering of neuron morphologies and spatial connectivity in the Drosophila brain.

## Overview

The 3D skeleton visualization displays:
- **Neuron morphologies** as 3D skeletons with accurate spatial coordinates
- **Brain regions** (ROIs) as semi-transparent meshes
- **Connections** between neurons with spatial context
- **Interactive exploration** with rotation, zoom, and selection

## Example Input Data

### Required Input

3D skeleton visualization requires **neuron bodyIds** from NeuPrint:

| Input Type              | Description                      | Example                     | Source                   |
| ----------------------- | -------------------------------- | --------------------------- | ------------------------ |
| **bodyId**              | Unique neuron identifier         | 123456789, 987654321        | NeuPrint database        |
| **Neuron types**        | Type name (converted to bodyIds) | 'KC_alpha', 'MBON03'        | Fetched via NeuPrint API |
| **Layer specification** | Multi-layer pathway              | ['KC.*', 'MBON.*', 'DAN.*'] | From path analysis       |

### Example Input Files

**From FindAllPath results**:
- Path: `connection_data/*/allpaths_*/path_bodyId_info.xlsx`
- Extract bodyIds from 'path_bodyId' sheet
- Example script: [`scripts/plot3dSkeleton.py`](../../scripts/plot3dSkeleton.py)

**Direct bodyId list**:
```python
# Option 1: Direct bodyId list
bodyIds = [
    123456789,  # KC neuron
    987654321,  # MBON neuron
    111222333   # DAN neuron
]

# Option 2: From neuron types
neuron_types = ['KC_alpha', 'MBON03', 'PPL1-01']
# Will be converted to bodyIds automatically

# Option 3: From path layers (recommended)
neuron_layers = ['KC.*', 'MBON03', 'DAN']
# Or as string: 'KC.* -> MBON03 -> DAN'
```

**Example datasets**:
- Sample bodyIds: See [`examples/README.md`](../../examples/README.md) for neuron lists
- Type-level data: `datasets/hemibrain_v1_2_1_alltypes_neuron_df.csv`
- Path results: Output from `scripts/FindPath.py` contains bodyId information

### Getting BodyIds

**From neuron types**:
```python
from neuprint import fetch_neurons, Client

client = Client('neuprint.janelia.org', dataset='hemibrain:v1.2.1', token='your_token')
neurons_df, roi_counts = fetch_neurons(NC(type='KC_alpha'))
bodyIds = neurons_df['bodyId'].tolist()
```

**From path analysis results**:
```python
import pandas as pd

# Load path results
paths = pd.read_excel('path_results.xlsx', sheet_name='path_bodyId')

# Extract unique bodyIds
all_bodyIds = set()
for col in paths.columns:
    if 'bodyId' in col or col in ['source', 'target']:
        all_bodyIds.update(paths[col].dropna().unique())
```

## Quick Start

### Basic Usage

```python
import statvis as sv
from coana import VisualizeSkeleton

# Login to NeuPrint
sv.LogInHemibrain(token='your_token', dataset='hemibrain:v1.2.1')

# Visualize neurons with 3D skeletons
vs = VisualizeSkeleton(
    neuron_layers=['KC.*', 'MBON03'],  # Neuron types to visualize
    brain_mesh='template',              # 'none', 'template', or 'whole'
    mesh_roi=['MB(R)', 'CA(R)'],       # ROI meshes to display
    neuron_alpha=0.2,                   # Neuron transparency
    synapse_size=3,                     # Synapse marker size
    skeleton_mode='tube',               # 'tube' or 'line'
    show_fig=True
)

vs.plot_neurons()

# Export rotating video
vs.export_video(fps=30, degree_per_frame=1.0, rotate='horizontal')
```

### Video Export

Export rotating 3D visualization to MP4 video:

```python
# Basic video export (after plot_neurons())
vs.export_video(fps=30, degree_per_frame=1.0)  # 12 sec video at 30fps

# Faster rotation (shorter video)
vs.export_video(fps=30, degree_per_frame=2.0)  # 6 sec video

# Vertical rotation (tumbling motion)
vs.export_video(fps=30, rotate='vertical')

# High quality export
vs.export_video(fps=30, scale=4, width=1920, height=1080)

# Reuse cached frames (fast regeneration)
vs.export_video(fps=60, use_existing_images=True)
```

**Standalone Export (from existing HTML file):**

```python
from visualize_skeleton import export_video_from_html

# No VisualizeSkeleton initialization needed
export_video_from_html('/path/to/my_neurons.html', fps=30)

# With options
export_video_from_html(
    '/path/to/my_neurons.html',
    fps=30,
    degree_per_frame=1.0,
    rotate='horizontal',
    scale=4
)
```

**Video Export Parameters:**

| Parameter             | Default      | Description                                    |
| --------------------- | ------------ | ---------------------------------------------- |
| `fps`                 | 30           | Frames per second                              |
| `degree_per_frame`    | 1.0          | Rotation angle per frame (1.0 → 360 frames)    |
| `rotate`              | 'horizontal' | Rotation direction: 'horizontal' or 'vertical' |
| `scale`               | 2            | Resolution multiplier                          |
| `use_existing_images` | True         | Reuse cached frame images if available         |

**Output Files:**
- `{folder}/pics_{fps}fps_{plane}/` - Cached frame images
- `{folder}/{name}_video_forward.mp4` - Forward rotation
- `{folder}/{name}_video_backward.mp4` - Reverse rotation

---

## Export Methods: WebDriver vs Kaleido

VisualizeSkeleton supports two export engines for PNG/video generation, each with different trade-offs:

### Method Comparison

| Feature           | **Kaleido** (default)           | **WebDriver**                      |
| ----------------- | ------------------------------- | ---------------------------------- |
| **Speed**         | 🐢 Slower (one-by-one rendering) | ⚡ Fast (browser-based screenshot ) |
| **Quality**       | ✅ Excellent                     | ✅ Excellent                        |
| **Max HTML Size** | ~100 MB                         | ~200 MB+                           |
| **WebGL Support** | ❌ Limited (rasterization)       | ✅ Full (native WebGL)              |
| **Dependencies**  | `kaleido` only                  | `selenium` + Chrome                |
| **Reliability**   | ✅ Very stable                   | ⚠️ Requires Chrome version match    |
| **Large Figures** | ❌ May timeout                   | ✅ Better handling                  |

### When to Use Each Method

**Use `export_method='kaleido'` (default) when:**
- HTML file size < 100 MB
- You want minimal dependencies
- Fast export is priority
- Simple 3D scenes without complex WebGL

**Use `export_method='webdriver'` when:**
- HTML file size > 100 MB
- Kaleido times out on complex figures
- You need native WebGL rendering quality
- Exporting rotating videos (efficient frame generation)

### Configuration Example

```python
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['MBON.*'],
    
    # Export method selection
    export_method='webdriver',      # 'kaleido' or 'webdriver'
    
    # WebDriver-specific settings
    webdriver_render_wait=0.3,      # Seconds between frames (None = auto-calibrate)
    
    # Shared export settings
    export_scale=3,                 # Resolution multiplier (1-5)
    export_timeout=60,              # Timeout per frame (kaleido only)
    
    # Auto-simplification threshold
    html_size_cap=150,              # MB threshold for auto-simplification
                                    # None = auto (100MB kaleido, 200MB webdriver)
)
```

### Auto-Simplification

For large figures, VisualizeSkeleton automatically simplifies meshes before export:

| Export Method | Default Threshold | Behavior                                       |
| ------------- | ----------------- | ---------------------------------------------- |
| Kaleido       | 100 MB            | Simplify if HTML > threshold, retry on timeout |
| WebDriver     | 200 MB            | Simplify if HTML > threshold                   |

**Manual override:**
```python
# Disable auto-simplification (may cause timeouts)
vs = VisualizeSkeleton(..., html_size_cap=9999)

# Force aggressive simplification
vs = VisualizeSkeleton(..., html_size_cap=50)
```

When simplification occurs, a `{name}_simplified.html` file is saved for reference.

---

## WebDriver Setup & Requirements

### Dependencies

WebDriver export requires:

1. **Python packages:**
   ```bash
   pip install selenium webdriver-manager
   ```

2. **Google Chrome browser** (version 109 or later)
   - The `--headless=new` mode requires Chrome 109+
   - Download: https://www.google.com/chrome/

### Platform-Specific Setup

#### macOS

```bash
# Install via Homebrew (recommended)
brew install --cask google-chrome

# Or download directly from google.com/chrome

# Verify Chrome version (must be 109+)
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --version
```

**ChromeDriver is managed automatically** by `webdriver-manager`. It:
- Detects your Chrome version
- Downloads matching ChromeDriver
- Caches it at `~/.wdm/drivers/chromedriver/`

#### Linux

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install google-chrome-stable

# Or download .deb from google.com/chrome
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb

# Verify version
google-chrome --version
```

#### Windows

1. Download Chrome from https://www.google.com/chrome/
2. Install normally
3. ChromeDriver is auto-managed by `webdriver-manager`

**Note:** On Windows, Chrome is typically found at:
- `C:\Program Files\Google\Chrome\Application\chrome.exe`
- `C:\Program Files (x86)\Google\Chrome\Application\chrome.exe`

### Troubleshooting WebDriver

#### Error: "Could not initialize Chrome WebDriver"

**Causes and solutions:**

1. **Chrome not installed:**
   ```
   Install Google Chrome from google.com/chrome
   ```

2. **Chrome version too old:**
   ```
   Update Chrome to version 109 or later
   Current minimum: Chrome 109 (for --headless=new WebGL support)
   ```

3. **ChromeDriver version mismatch:**
   ```bash
   # Clear cached ChromeDriver
   rm -rf ~/.wdm/drivers/chromedriver/
   
   # Re-run export (will download fresh ChromeDriver)
   ```

4. **webdriver-manager not installed:**
   ```bash
   pip install webdriver-manager
   ```

5. **Network error downloading ChromeDriver:**
   ```
   Check internet connection
   Retry the export
   ```

#### Error: "WebGL not supported"

This typically means Chrome is running in legacy headless mode. The code now uses `--headless=new` which supports WebGL on all platforms.

#### Fallback to Kaleido

If WebDriver fails, you can always fall back:
```python
vs = VisualizeSkeleton(..., export_method='kaleido')
```

### Chrome Version Check

To verify your Chrome version meets requirements:

```bash
# macOS
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --version

# Linux
google-chrome --version

# Windows (PowerShell)
(Get-Item "C:\Program Files\Google\Chrome\Application\chrome.exe").VersionInfo.FileVersion
```

**Minimum required: Chrome 109** (released January 2023)

---

### Prerequisites

**Required packages**:
```bash
pip install navis
pip install flybrains  # For brain templates and transforms
pip install plotly
pip install neuprint-python
```

**For WebDriver export:**
```bash
pip install selenium webdriver-manager
# + Google Chrome browser (version 109+)
```

**Data requirements**:
- Valid NeuPrint authentication token ([get yours here](https://neuprint.janelia.org/account))
- Neuron bodyIds or types from chosen dataset
- Internet connection (first time, to fetch skeleton data)
- **Optional**: ~10GB disk space for whole-brain transforms (one-time download, ~1-2 hours)

### Multi-Dataset Support (NEW)

**Supported Datasets:**
- **hemibrain:v1.2.1** / **optic-lobe:v1.1**: Adult brain
- **manc:v1.2.3**: Male VNC (ventral nerve cord)
- **male-cns:v0.9**: Full CNS (brain + VNC combined)

**Brain Mesh Options:**

```python
# Option 1: No brain mesh (fastest, clearest for single neurons)
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['KC.*'],
    brain_mesh='none'  # No background mesh
)

# Option 2: Template mesh (native EM resolution, 0.5-2s load time)
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['KC.*'],
    brain_mesh='template'  # JRCFIB2018F for hemibrain, MANC for manc
)

# Option 3: Whole brain/VNC mesh (standard resolution, requires download)
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['KC.*'],
    brain_mesh='whole'  # JRC2018F whole brain, ~500MB one-time download
)
```

**Dataset-Specific Templates:**

| Dataset              | `brain_mesh='template'` | `brain_mesh='whole'` | Transform Required |
| -------------------- | ----------------------- | -------------------- | ------------------ |
| **hemibrain:v1.2.1** | JRCFIB2018F (EM)        | JRC2018F (confocal)  | ✅ Yes (~10GB)      |
| **optic-lobe:v1.1**  | JRCFIB2018F (EM)        | JRC2018F (confocal)  | ✅ Yes (~10GB)      |
| **manc:v1.2.3**      | MANC (native VNC)       | —                    | ❌ No               |
| **male-cns:v0.9**    | JRCFIB2022M (native)    | —                    | ❌ No               |

**Transform Storage:**

Transforms for `brain_mesh='whole'` are stored in `~/flybrain-data` (managed by flybrains). First use will prompt:

```
⚠ Brain transforms not found for hemibrain:v1.2.1
  Target template: JRC2018F (whole brain)
  Transform size: ~10GB total (8 files, one-time download)
  Download time: ~1-2 hours
  Storage location: ~/flybrain-data

Download transforms? [y/N]:
```

**Important:** Only **JRC2018F_JRCFIB2018F.h5 (~1.3 GB)** is needed for hemibrain/optic-lobe whole-brain visualization. However, the flybrains package downloads ALL JRC transforms (8 files, ~10GB total) as a bundle - selective download is not supported.

**Actual transform path used:**
```
JRCFIB2018Fraw → JRCFIB2018F → JRCFIB2018Fum → JRC2018F
```

The other 7 files (~8.7GB) enable cross-dataset registration and support for other brain templates (FAFB, JFRC2013, etc.).

Type `y` to download. Transforms persist and are shared across all NeuPrint projects.

## Key Features

### 1. Neuron Skeleton Rendering

#### Morphology Display
- **Skeleton structure**: Node-and-edge representation
- **Compartments**: Soma, dendrites, axon (if labeled)
- **Spatial accuracy**: Real coordinates from EM reconstruction
- **Color coding**: By neuron, by type, or by compartment

#### Multiple Neurons
```python
# Visualize multiple neurons together
plot_navis_3d(
    bodyIds=[
        123456789,  # Neuron 1
        987654321,  # Neuron 2
        111222333   # Neuron 3
    ],
    color_by='neuron'  # Color each neuron differently
)
```

### 2. Brain Region (ROI) Meshes

Display anatomical context with 3D brain region meshes:

#### ROI Selection
```python
# Show specific ROIs
plot_navis_3d(
    bodyIds=[123456789],
    show_rois=True,
    roi_list=['MB_CA', 'MB_PED', 'SLP_R'],  # Mushroom body regions
    roi_opacity=0.2  # Semi-transparent
)
```

#### Available ROIs
Major brain regions include:
- **Mushroom Body**: MB_CA (calyx), MB_PED (pedunculus), MB_VL (vertical lobe)
- **Central Complex**: FB (fan-shaped body), EB (ellipsoid body), PB (protocerebral bridge)
- **Optic Lobes**: ME (medulla), LO (lobula), LOP (lobula plate)
- **Antennal Lobe**: AL_R, AL_L
- **Lateral Horn**: LH_R, LH_L

**Complete list**: Available in `navis_roi_meshes_json/primary_rois/` directory

### 3. Synapse Visualization

Show synaptic connections in 3D space:

#### Presynaptic Sites (Outputs)
- **Color**: Red (default)
- **Size**: Proportional to partner count
- **Location**: Exact spatial coordinates

#### Postsynaptic Sites (Inputs)
- **Color**: Blue (default)
- **Size**: Proportional to partner count
- **Location**: Exact spatial coordinates

#### Connection Lines
```python
# Show connections between neurons
plot_navis_3d(
    bodyIds=[source_id, target_id],
    show_connectors=True,
    show_connections=True,  # Draw lines between connected synapses
    connection_color='green',
    connection_width=2
)
```

### 4. Color Schemes

#### By Neuron
Each neuron gets a unique color:
```python
color_by='neuron'  # Default
```

#### By Type
Neurons of same type share color:
```python
plot_navis_3d(
    bodyIds=neuron_list,
    color_by='type',
    neuron_types=['KC', 'MBON', 'DAN']  # Provide type info
)
```

#### By Compartment
Different colors for soma, dendrites, axon:
```python
color_by='compartment'
```

#### Custom Colors
Specify exact colors:
```python
plot_navis_3d(
    bodyIds=[id1, id2, id3],
    custom_colors=['#FF0000', '#00FF00', '#0000FF']  # Red, Green, Blue
)
```

### 5. Interactive Controls

#### Rotation
- **Click and drag**: Rotate view in any direction
- **Shift + drag**: Pan view
- **Scroll**: Zoom in/out

#### Selection
- **Click neuron**: Highlight and show info
- **Click synapse**: Show partner details
- **Click ROI**: Display region name and volume

#### Visibility Toggle
- **Neurons**: Toggle individual neurons on/off
- **ROIs**: Toggle brain regions
- **Synapses**: Show/hide connectors
- **Connections**: Show/hide connection lines

### 6. Camera Views

Preset viewing angles:

#### Anterior View
Front view of the brain:
```python
camera_view='anterior'
```

#### Posterior View
Back view:
```python
camera_view='posterior'
```

#### Dorsal View
Top-down view:
```python
camera_view='dorsal'
```

#### Ventral View
Bottom-up view:
```python
camera_view='ventral'
```

#### Custom View
Define exact camera position:
```python
camera_position={
    'eye': {'x': 1000, 'y': 1000, 'z': 1000},
    'center': {'x': 0, 'y': 0, 'z': 0},
    'up': {'x': 0, 'y': 0, 'z': 1}
}
```

### 7. Export Options

#### HTML Interactive
Default output format:
```python
output_format='html'  # Interactive 3D in browser
```

#### Static Image
```python
# Export as PNG
plot_navis_3d(
    bodyIds=[123456789],
    output_format='png',
    image_width=1920,
    image_height=1080,
    camera_view='anterior'
)
```

#### Animation
Create rotating view:
```python
# Generate rotation animation
plot_navis_3d(
    bodyIds=[123456789],
    output_format='gif',
    rotation_frames=360,  # Full rotation
    rotation_axis='z'     # Rotate around z-axis
)
```

### 8. Performance Options

#### Level of Detail
Control skeleton complexity:

```python
# High detail (slow)
skeleton_detail='high'  # All nodes and edges

# Medium detail (balanced)
skeleton_detail='medium'  # Simplified skeleton

# Low detail (fast)
skeleton_detail='low'  # Major branches only
```

#### ROI Mesh Quality
```python
# Lower quality for faster rendering
roi_mesh_quality='low'   # Fewer triangles
roi_mesh_quality='medium'
roi_mesh_quality='high'  # Maximum detail
```

### 9. Measurement Tools

#### Distance Measurement
Measure spatial distances:
```python
# Enable measurement mode
enable_measurements=True
```

**Usage**: Click two points to measure distance in micrometers

#### Volume Calculation
Calculate neuron volume:
```python
calculate_volumes=True  # Adds volume info to neuron labels
```

#### Path Length
Calculate cable length:
```python
calculate_path_lengths=True  # Total skeleton length
```

### 10. Advanced Visualization

#### Density Maps
Show synaptic density in 3D:
```python
plot_density_map=True
density_radius=10  # Micrometers
density_type='presynaptic'  # or 'postsynaptic'
```

#### Connectivity Matrix Overlay
Show connection strength as colors:
```python
plot_navis_3d(
    bodyIds=neuron_list,
    connectivity_matrix=conn_matrix,  # Pandas DataFrame
    color_by_connectivity=True
)
```

## Advanced Usage

### Pathway Visualization

Visualize complete pathways in 3D:

```python
# Get pathway neurons from path analysis
pathway_df = pd.read_excel('path_results.xlsx')
all_neurons = list(set(pathway_df['source'].tolist() + 
                      pathway_df['target'].tolist()))

# Extract bodyIds (if type names)
from neuprint import fetch_neurons
bodyIds = fetch_neurons(all_neurons, dataset='hemibrain:v1.2.1')

# Visualize pathway in 3D
plot_navis_3d(
    bodyIds=bodyIds,
    show_rois=True,
    show_connectors=True,
    color_by='type',
    roi_list=['MB_CA', 'MB_PED', 'MB_VL']  # Relevant ROIs
)
```

### Comparison Views

Compare neuron morphologies:

```python
# Create side-by-side views
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Two neurons in separate panels
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
)

# Add neuron 1 to left panel
# Add neuron 2 to right panel
# (detailed implementation in examples/)
```

### Synapse-Specific Filtering

Show only specific synapse types:

```python
# Only show strong connections (>10 synapses)
plot_navis_3d(
    bodyIds=[source_id, target_id],
    show_connectors=True,
    synapse_threshold=10,  # Minimum synapse count
    synapse_type='presynaptic'  # or 'postsynaptic'
)
```

### Custom Mesh Import

Use custom brain region meshes:

```python
# Load custom mesh
custom_mesh = load_mesh('custom_roi.obj')

plot_navis_3d(
    bodyIds=[123456789],
    custom_meshes=[custom_mesh],
    mesh_colors=['rgba(255,0,0,0.2)']
)
```

### Time-Series Animation

Show neuron growth or changes:

```python
# Multiple time points
neuron_versions = [
    bodyIds_t0,  # Time point 0
    bodyIds_t1,  # Time point 1
    bodyIds_t2   # Time point 2
]

# Create animation
create_temporal_animation(
    neuron_versions,
    frame_duration=1000,  # ms per frame
    output='neuron_growth.html'
)
```

## Coordinate Systems

### Hemibrain Coordinates
- **X-axis**: Anterior (front) to Posterior (back)
- **Y-axis**: Left to Right
- **Z-axis**: Ventral (bottom) to Dorsal (top)
- **Units**: Micrometers (μm)
- **Origin**: Defined by FlyEM/DVID system

### Transformations
Convert between coordinate systems:
```python
from navis import xform_brain

# Transform to JRC2018 template
neurons_transformed = xform_brain(
    neurons,
    source='FAFB14',
    target='JRC2018F'
)
```

## Performance Tips

### Large Neuron Sets
- **Limit to <20 neurons** for interactive performance
- **Use lower detail** for faster rendering
- **Hide connectors** if not needed
- **Reduce ROI mesh quality**

### High-Resolution Exports
- **Increase image dimensions** (2K, 4K)
- **Use static format** (PNG) instead of HTML
- **Render without connectors** for cleaner images
- **Set specific camera view** to avoid re-positioning

### Memory Management
- **Process neurons in batches**
- **Clear cache** between visualizations
- **Use lazy loading** for large datasets

## Troubleshooting

**Skeletons don't appear**: Check bodyIds are valid and dataset is correct

**ROIs not loading**: Ensure `navis-flybrains` is installed and ROI names are correct

**Slow performance**: Reduce neuron count, lower skeleton detail, hide connectors

**Export fails**: Check output directory permissions, try different format

**Colors wrong**: Verify color specification format (hex or rgba)

**Memory error**: Reduce number of neurons, disable connectors, lower mesh quality

## Integration with Other Visualizations

### 3D + Network
- **Network**: Shows connectivity topology
- **3D**: Shows spatial arrangement
- **Together**: Correlate topology with anatomy

### 3D + Heatmap
- **Heatmap**: Quantifies connection strengths
- **3D**: Visualizes spatial positions
- **Workflow**: Identify strong connections in heatmap → visualize in 3D

### 3D + Sankey
- **Sankey**: Shows flow and magnitude
- **3D**: Shows physical pathways
- **Use case**: Understand how flow maps to brain regions

## Example Workflows

### 1. Mushroom Body Pathway
```python
# Kenyon Cells → MBONs
kc_ids = [...]  # KC bodyIds
mbon_ids = [...]  # MBON bodyIds

plot_navis_3d(
    bodyIds=kc_ids + mbon_ids,
    show_rois=True,
    roi_list=['MB_CA', 'MB_PED', 'MB_VL', 'MB_ML'],
    show_connectors=True,
    color_by='type'
)
```

### 2. Visual Pathway
```python
# Optic lobe to central brain
plot_navis_3d(
    bodyIds=visual_neurons,
    show_rois=True,
    roi_list=['ME', 'LO', 'LOP', 'AOTU', 'BU'],
    camera_view='lateral'
)
```

### 3. Comparison Study
```python
# Compare two neuron types
plot_navis_3d(
    bodyIds=type1_ids,
    output_file='type1_3d.html'
)

plot_navis_3d(
    bodyIds=type2_ids,
    output_file='type2_3d.html'
)
```

## Related Documentation

- [Advanced Visualization Features](../AdvancedVisualizationFeatures.md)
- [NAVIS Integration](https://navis.readthedocs.io/)
- [FlyBrains Resources](https://github.com/navis-org/navis-flybrains)
- [NeuPrint Documentation](https://neuprint.janelia.org/help)

## External Resources

- **NAVIS**: https://navis.readthedocs.io/
- **FlyWire**: https://flywire.ai/
- **Virtual Fly Brain**: https://www.virtualflybrain.org/
- **FlyEM**: https://www.janelia.org/project-team/flyem
