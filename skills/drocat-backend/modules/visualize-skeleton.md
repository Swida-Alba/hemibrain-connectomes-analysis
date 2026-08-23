# visualize_skeleton — VisualizeSkeleton

Module `src/visualize_skeleton.py`. `VisualizeSkeleton` produces interactive 3D
plotly HTML for neuron skeletons, synapses, brain-region ROI meshes, and supports
PNG/PDF/PPTX/GIF/video exports. `WebDriverExportSession` (in the same module)
drives Chrome for PNG/video when `export_method="webdriver"`.

## Constructor (key params)

```python
from visualize_skeleton import VisualizeSkeleton

vs = VisualizeSkeleton(
    dataset="male-cns:v0.9",
    neuron_layers=[["aMe12"], ["aMe10", "aMe9"]],   # nested list model
    search_columns="auto",              # "auto" | "type" | "instance" | "bodyId"
    hemisphere="both",
    custom_layer_names=[],              # optional layer labels
    output_dir="/abs/output/skeleton",
    output_format="csv",                # merged synapse export: "csv" | "xlsx"
    skeleton_mode="line",               # "line" | "tube" (start with line)
    brain_mesh="template",
    vnc_mesh=None,
    legend_mode="layer",                # "single" | "type" | "layer"
    neuron_alpha=1.0,
    neuron_colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
    synapse_colors=["#50E3C2"],
    background_color="#ffffff",
    skip_synapse=False,
    min_synapse_num=3,
    synapse_size="real",                # "real" or a numeric size; uniform sizing uses uniform_synapse_size
    uniform_synapse_size=False,
    synapse_alpha=1.0,
    synapse_mode="scatter",             # "scatter" | "sphere" | "cone" | "tetrahedron"
    mesh_roi=["EB", "LH", "AL"],
    mesh_color=["#4A90E2", "#50E3C2", "#B8E986"],
    mesh_alpha=0.1,
    cache_neurons=True,
    cache_synapses=True,
    smooth_skeleton=True,
    show_soma=False,
    show_connectors=False,
    export_method="webdriver",          # or "kaleido"
    export_scale=2,
    export_views=False,
    show_fig=False,
    brain_mesh_color=None,
    neuprint_skeleton_pipeline="fast",  # "fast" | "fine" | "artistic" | "direct" | ...
    skeleton_mesh_simplification=0.0,
)
```

## Key methods

| Method | Purpose |
| --- | --- |
| `plot_neurons()` | Build the interactive HTML (main entry). |
| `plot_individuals(pdf_images_per_page=(3,2), views=["front"], summary_format=["pdf"])` | Per-neuron PDF/PPTX profile export. |
| `export_video(fps=30, degree_per_frame=1.0, rotate="horizontal", export_gif=True, gif_scale=0.2, ...)` | Rotating video/GIF export. |
| `list_available_rois(refresh=False, fetch_online=True)` | List available ROI meshes for the dataset. |

```python
vs = VisualizeSkeleton(dataset="male-cns:v0.9", neuron_layers=[["aMe12", "aMe10"]],
                       output_dir="/abs/output/skeleton", skeleton_mode="line",
                       mesh_roi=["EB"], show_fig=False, skip_synapse=True)
vs.plot_neurons()
# optional, after the base HTML succeeds:
vs.plot_individuals(pdf_images_per_page=(3, 2), views=["front"], summary_format=["pdf"])
vs.export_video(fps=30, export_gif=True, gif_scale=0.2)
```

## Notes

- BANC skeleton visualization is unavailable — choose a non-BANC dataset.
- Mesh fixes are dataset-specific; verify ROI availability and coordinates before
  changing transforms.
- `export_method="webdriver"` needs Chrome + WebDriver (Kaleido is the slower
  fallback). Use `skeleton_mode="line"` and no exports for a first smoke test.
- `synapse_size` accepts `"real"` or a numeric value; uniform sizing uses the
  `uniform_synapse_size` bool. Invalid/empty values fall back to `"real"`.
- `cache_neurons`/`cache_synapses` persist raw skeletons/synapses, reused by the
  Settings skeleton pull.
