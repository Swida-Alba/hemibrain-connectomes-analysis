# Dataset Transform Paths Reference

## Current NeuPrint Datasets (2025)

### Brain Datasets

#### 1. hemibrain:v1.2.1
- **Type:** Female adult brain
- **Server:** https://neuprint.janelia.org
- **Native Space:** JRCFIB2018Fraw
- **Transform Path:** 
  ```
  JRCFIB2018Fraw → JRCFIB2018F → JRCFIB2018Fum → JRC2018F
  ```
- **ROIs:** 230 brain neuropils
- **brain_mesh options:**
  - `'none'`: No brain mesh
  - `'template'`: JRCFIB2018F (hemibrain-only mesh)
  - `'whole'`: JRC2018F (whole brain template, requires transforms)
  - `'hemi'`: Hemisphere mesh (left or right, HEMIBRAIN ONLY)

#### 2. optic-lobe:v1.1
- **Type:** Optic lobe connectome (part of Male CNS volume)
- **Server:** https://neuprint.janelia.org
- **Native Space:** JRCFIB2022Mraw
- **Transform Path:**
  ```
  JRCFIB2022Mraw → JRCFIB2022M
  ```
- **Note:** Optic-lobe is a focused reconstruction from the Male CNS dataset, so it uses the Male CNS coordinate system (JRCFIB2022M), NOT the hemibrain (JRCFIB2018F) system.
- **ROIs:** 2,690 (includes fine-grained columns)
- **brain_mesh options:**
  - `'none'`: No brain mesh
  - `'template'`: JRCFIB2018F (optic lobe region)
  - `'whole'`: JRC2018F (whole brain template, requires transforms)
  - ❌ `'hemi'`: NOT SUPPORTED (brain dataset only)

---

### VNC Datasets

#### 3. manc:v1.0 / manc:v1.2
- **Type:** Male Adult Nerve Cord (VNC only)
- **Server:** https://neuprint.janelia.org
- **Native Space:** MANCraw
- **Transform Path:**
  ```
  MANCraw → MANC
  ```
- **ROIs:** VNC neuropils
- **brain_mesh options:**
  - `'none'`: No VNC mesh
  - `'template'`: MANC (VNC envelope)
  - `'whole'`: MANC (VNC envelope, no transform needed)
  - ❌ `'hemi'`: NOT SUPPORTED (VNC has no hemispheres)

**Note:** For VNC datasets, `'whole'` and `'template'` are equivalent - both show the VNC envelope.

---

### Brain + VNC Datasets

#### 4. male-cns:v1.0
- **Type:** Male Central Nervous System (Brain + VNC)
- **Server:** https://neuprint.janelia.org
- **Native Space:** JRCFIB2022Mraw
- **Transform Path:**
  ```
  JRCFIB2022Mraw → JRCFIB2022M
  ```
- **ROIs:** Brain + VNC neuropils
- **brain_mesh options:**
  - `'none'`: No CNS mesh
  - `'template'`: JRCFIB2022M (full CNS: brain + VNC)
  - `'whole'`: JRCFIB2022M (full CNS envelope)
  - ❌ `'hemi'`: NOT SUPPORTED (use separate brain/VNC meshes)

---

## Template Space Comparison

| Dataset | Native Space | Requires Transform? | Whole Brain/VNC Template | Notes |
|---------|--------------|---------------------|-------------------------|-------|
| hemibrain:v1.2.1 | JRCFIB2018Fraw | Yes (for 'whole') | JRC2018F | Supports 'hemi' |
| optic-lobe:v1.1 | JRCFIB2022Mraw | Yes | JRCFIB2022M | Part of Male CNS |
| manc:v1.x | MANCraw | No | MANC | VNC only |
| male-cns:v1.0 | JRCFIB2022Mraw | No | JRCFIB2022M | Brain + VNC |

---

## brain_mesh Parameter Guide

### For Brain Datasets (hemibrain, optic-lobe)
```python
vis = ca.VisualizeSkeleton(
    neuron_layers=[{"type": "TmY9"}],
    dataset='hemibrain:v1.2.1',
    brain_mesh='whole'  # Shows JRC2018F (requires transforms)
)
```

### For VNC Datasets (manc)
```python
vis = ca.VisualizeSkeleton(
    neuron_layers=[{"type": "SomeVNCNeuron"}],
    dataset='manc:v1.2',
    brain_mesh='whole'  # Shows MANC VNC envelope (no transform needed)
)

# 'hemi' is NOT SUPPORTED for VNC
vis = ca.VisualizeSkeleton(
    dataset='manc:v1.2',
    brain_mesh='hemi'  # ❌ Will auto-switch to 'whole' with warning
)
```

### For Brain+VNC Datasets (male-cns)
```python
vis = ca.VisualizeSkeleton(
    neuron_layers=[{"type": "SomeNeuron"}],
    dataset='male-cns:v1.0',
    brain_mesh='whole'  # Shows JRCFIB2022M (full CNS envelope)
)
```

---

## Transform Requirements

### Downloads Required
Only brain datasets need transform downloads:

```python
# Hemibrain/Optic-lobe with brain_mesh='whole'
# → Downloads ~10GB of JRC transforms (one-time)
import flybrains
flybrains.download_jrc_transforms()
```

### No Downloads Needed
VNC and Brain+VNC datasets work without transforms:

```python
# MANC or male-cns
# → No transform download needed!
vis = ca.VisualizeSkeleton(
    dataset='manc:v1.2',
    brain_mesh='whole'  # Works immediately
)
```

---

## Error Handling

### Invalid brain_mesh for VNC
```python
# This will trigger automatic correction:
vis = ca.VisualizeSkeleton(
    dataset='manc:v1.2',
    brain_mesh='hemi'  # ❌ Not supported for VNC
)
# Output:
# ⚠️  brain_mesh="hemi" only works with hemibrain:v1.2.1 dataset
#    VNC datasets (manc, male-cns) do not support hemisphere mode
#    Automatically switching to brain_mesh="whole"
```

### Unknown Dataset
```python
vis = ca.VisualizeSkeleton(
    dataset='unknown:v1.0',
    brain_mesh='whole'
)
# Output:
# ⚠️  Unknown dataset "unknown:v1.0", defaulting to hemibrain template
```

---

## Migration from Old Code

### Old Code (Pre-November 2025)
```python
# All datasets used same logic
vis = ca.VisualizeSkeleton(
    dataset='manc:v1.0',
    brain_mesh='whole'  # Tried to transform to JRC2018F (wrong!)
)
```

### New Code (Post-November 2025)
```python
# VNC datasets handled correctly
vis = ca.VisualizeSkeleton(
    dataset='manc:v1.0',
    brain_mesh='whole'  # Uses MANC VNC envelope (correct!)
)
```

---

## References

- **flybrains:** https://github.com/navis-org/navis-flybrains
- **NeuPrint:** https://neuprint.janelia.org/
- **JRC Templates:** https://www.janelia.org/open-science/jrc-2018-brain-templates
- **MANC Paper:** https://www.nature.com/articles/s41586-023-06683-4

---

**Last Updated:** November 21, 2025  
**Version:** 3.1
