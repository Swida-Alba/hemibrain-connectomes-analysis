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

| Input Type | Description | Example | Source |
|------------|-------------|---------|--------|
| **bodyId** | Unique neuron identifier | 123456789, 987654321 | NeuPrint database |
| **Neuron types** | Type name (converted to bodyIds) | 'KC_alpha', 'MBON03' | Fetched via NeuPrint API |
| **Layer specification** | Multi-layer pathway | ['KC.*', 'MBON.*', 'DAN.*'] | From path analysis |

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
from navis_related import plot_navis_3d

# Visualize neurons with 3D skeletons
plot_navis_3d(
    bodyIds=[123456789, 987654321],  # List of neuron bodyIds
    dataset='hemibrain:v1.2.1',
    token='your_neuprint_token',
    show_rois=True,                   # Display brain regions
    show_connectors=True,             # Show synapse locations
    output_folder='./3d_output',
    showfig=True
)
```

### Prerequisites

**Required packages**:
```bash
pip install navis
pip install navis-flybrains
pip install plotly
```

**Data requirements**:
- Valid NeuPrint authentication token ([get yours here](https://neuprint.janelia.org/account))
- Neuron bodyIds from chosen dataset
- Internet connection (first time, to fetch skeleton data)

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
