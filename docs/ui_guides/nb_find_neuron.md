# Find EM Neurons (NeuronBridge) - Instruction

## Purpose

Find EM neurons whose morphology matches a GAL4/Split-GAL4 driver line.

## Quick start

1. Enter one or more **Driver Line Names** (e.g. `VT037867`).
2. Click **Find EM Neurons**.

## Inputs

- **Driver Line Names**: e.g. `VT037867`, `R10A06`, `SS00731`.
- **Algorithm**: `cds`, `pppm`, or `both`.
- **Top N Results** per line.

## Advanced

- Visualize top matches (grouped by type or bodyId).
- Individual profile PDFs: images per page, background color.
- Sort by max score or type-average score.

## Output

`findneuron_{lines}_{timestamp}/` with match CSVs, type summaries and
visualizations.
