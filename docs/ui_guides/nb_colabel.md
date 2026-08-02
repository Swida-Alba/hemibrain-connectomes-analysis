# Co-Labeling Analysis (NeuronBridge) - Instruction

## Purpose

Analyze overlap between driver lines: which neuron types each line labels,
Jaccard / weighted-Jaccard co-labeling matrices, and line specificity.

## Quick start

1. Enter at least two **Driver Lines** (e.g. `VT037867`, `R10A06`).
2. Keep Jaccard and Weighted Jaccard enabled.
3. Click **Run Co-Labeling** - matrices, statistics and a report are saved.

## Inputs

- **Driver Lines**: at least two names.
- **Similarity methods**: Jaccard, weighted Jaccard.

## Advanced

- Top N neurons per line and minimum score filters.
- Heatmaps, HTML report, 3D skeletons of top co-labeled types.
- Profile PDF layout and background.

## Output

`colabel_{lines}_{timestamp}/` with expression matrices, co-labeling
matrices, line statistics and an HTML report.
