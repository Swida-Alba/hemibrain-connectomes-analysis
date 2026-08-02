# DROCAT UI - Instruction Guides

Every panel in the DROCAT web UI has an **Instructions** link that opens its
local guide below. The guides explain what each function does, how to fill
the form, the most useful options, and where results are saved.

## Guides by panel

| Panel | Guide | Covers |
| --- | --- | --- |
| Find All Paths | [find_path.md](find_path.md) | Multi-hop pathfinding |
| Direct Connections | [find_direct.md](find_direct.md) | One-hop connections |
| Connectivity Profiling | [connectivity_profiling.md](connectivity_profiling.md) | Profile similarity |
| Homolog Finding | [find_homologs.md](find_homologs.md) | Cross-dataset homologs |
| Cross-Dataset | [cross_dataset.md](cross_dataset.md) | Multi-dataset comparison |
| Find Lines (NeuronBridge) | [nb_find_lines.md](nb_find_lines.md) | EM → driver lines |
| Find Neurons (NeuronBridge) | [nb_find_neuron.md](nb_find_neuron.md) | Driver line → EM |
| Co-Labeling (NeuronBridge) | [nb_colabel.md](nb_colabel.md) | Line overlap analysis |
| Visualization | [visualization.md](visualization.md) | 3D skeletons + path network |
| Settings | [settings.md](settings.md) | Tokens, datasets, output dir |

## Shared references

- [Input File Formats](input_formats.md) - neuron CSV/XLSX files, path files
  and layer-mapping CSVs. Linked directly from every upload menu.

## Quick tips

- **Neuron lists**: type an entry and press Enter to make a chip, paste a
  comma/newline list, or upload a CSV/XLSX (first column). Numeric bodyIds
  are kept as integers.
- **Output folders**: every run creates its own timestamped folder named
  `{tool}_{dataset}_{detail}_{timestamp}` (e.g.
  `findpath_MCNS_aMe12_to_aMe10_L2w3r0p0_20260801_183000`). The results
  panel always opens exactly that run's folder.
- **Long runs**: the execution log streams live - watch it while the tool
  works; output files appear when it finishes.
- **First use of a dataset**: DROCAT downloads the neuron table once, so the
  first query takes longer than later cached runs.
