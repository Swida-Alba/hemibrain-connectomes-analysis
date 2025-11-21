# VisualizeSkeleton Quick Reference Card

## New Features (November 2024)

### 1. Multi-Dataset Support

```python
from coana import VisualizeSkeleton

# Hemibrain dataset
vs_hemi = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['EB'],
    mesh_roi=['EB', 'PB', 'FB']
)

# Optic-lobe dataset
vs_optic = VisualizeSkeleton(
    dataset='optic-lobe:v1.1',
    neuron_layers=['LNd'],
    mesh_roi=['ME(R)', 'AME(R)']
)
```

**✓ Automatic dataset-specific caching**  
**✓ No configuration needed**  
**✓ Backward compatible**

---

### 2. Discover Available ROIs

```python
# List all ROIs for current dataset
available_rois = vs.list_available_rois()

# Force refresh from NeuPrint
fresh_rois = vs.list_available_rois(refresh=True)
```

**✓ Queries NeuPrint database**  
**✓ Cached locally for speed**  
**✓ Fallback to local meshes**

---

### 3. Brain Transformation Confirmation

```python
# User will be prompted if transforms are missing
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['EB'],
    brain_mesh='whole'  # Requires JRC2018F transforms (~500MB)
)
```

**✓ One-time download**  
**✓ User confirmation required**  
**✓ Clear error messages**

---

## Directory Structure

```
navis_roi_meshes_json/
├── hemibrain_v1_2_1/     # Hemibrain ROI meshes
├── optic-lobe_v1_1/      # Optic-lobe ROI meshes
├── fib/                  # FIB ROI meshes
├── manc/                 # MANC ROI meshes
└── primary_rois/         # Backward compatibility
```

---

## Common Use Cases

### Load Specific ROIs
```python
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['EB', 'PB'],
    mesh_roi=['EB', 'PB', 'FB', 'NO']  # Central complex
)
vs.plot_neurons()
```

### Work with Multiple Datasets
```python
# Hemibrain
vs1 = VisualizeSkeleton(dataset='hemibrain:v1.2.1', neuron_layers=['EB'])
vs1.plot_neurons()

# Optic-lobe  
vs2 = VisualizeSkeleton(dataset='optic-lobe:v1.1', neuron_layers=['LNd'])
vs2.plot_neurons()
```

### Enable Whole Brain Mesh
```python
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['EB'],
    brain_mesh='whole',  # Requires transform confirmation
    brain_mesh_color='rgba(200, 230, 240, 0.1)'
)
vs.plot_neurons()
```

---

## Troubleshooting

### ROI Not Found
```
⚠️  mesh file custom_roi.json not found
```
**Fix:** Check available ROIs with `vs.list_available_rois()`

### Transform Download Failed
```
⚠️  Failed to load whole brain mesh
```
**Fix:** 
1. Install flybrains: `pip install navis[flybrains]`
2. Check internet connection
3. Manually download: `flybrains.download_jrc_transforms()`

### API Timeout
```
⚠️  Failed to fetch available ROIs from NeuPrint: timeout
```
**Fix:** Uses cached/local data automatically (no action needed)

---

## Performance Tips

✓ **Use cached ROI lists:** Avoid `refresh=True` unless needed  
✓ **Simplify meshes:** `mesh.simplify(factor=0.5)` for faster rendering  
✓ **Use line mode:** `skeleton_mode='line'` for large datasets  
✓ **Disable auto-open:** `show_fig=False` for batch processing  

---

## Documentation

- **Full Guide:** [docs/visualizations/VisualizeSkeleton_Updates_Nov2024.md](../visualizations/VisualizeSkeleton_Updates_Nov2024.md)
- **Examples:** [examples/Example_VisualizeSkeleton_MultiDataset.py](../../examples/Example_VisualizeSkeleton_MultiDataset.py)
- **Main README:** [README.md](../../README.md)

---

## External References

- [navis documentation](https://navis.readthedocs.io/)
- [flybrains GitHub](https://github.com/navis-org/navis-flybrains)
- [NeuPrint](https://neuprint.janelia.org/)

---

**Version:** v3.1 | **Last Updated:** November 2024
