# Visualization - Instruction

Two tools: **3D Skeleton (plot3dSkeleton)** and **Path Network (PlotPath)**.

## Quick start (3D)

1. Add neurons/layers as chips (e.g. `aMe12`, or `aMe12 -> MBON01`).
2. Pick the dataset; keep **Brain Mesh = template** and **ROIs = EB, LH, AL**.
3. Click **Generate Visualization**.

## Quick start (Path Network)

1. Run **Find All Paths** first.
2. Upload its `*_allpaths_type.csv` (or `*_allpaths_info.xlsx`) file.
3. Click **Generate Visualization**.

## 3D Skeleton

- **Neurons / Layers**: type names, bodyIds, or `A -> B -> C` layer chains.
  Custom layer names map 1:1 to the chips.
- **Appearance**: skeleton mode (tube/line), legend mode, neuron opacity and
  a color palette (full bokeh catalog) with range/custom editing.
- **Synapses**: skip toggle, minimum count, size, opacity, cone/scatter mode.
- **Brain Region ROIs**: select ROIs (presets or regex like `ME.*`); give
  each ROI its own color from a palette, or use Auto gray. Start/End % range
  selects part of a palette; Custom colors mode supports single colors with
  reordering.
- **Brain/VNC mesh**: template or whole brain, VNC toggle, mesh color.
- **Advanced**: caching, smoothing, soma/connectors, export method
  (webdriver needs Chrome; kaleido is the stable fallback), export scale and
  views, individual-profile PDF/PPTX, rotating video/GIF.

## Path Network (PlotPath)

- Upload a Find All Paths output file
  ([Input File Formats](input_formats.md)), choose a color scheme (swatch
  preview), layout and output folder.
- Produces Sankey, heatmap, network HTML and an XLSX connection table.

## Output

- 3D: `plot3d_{DATASET}_{layers}_{timestamp}/`
- Path: the timestamped folder inside the chosen path output directory.
