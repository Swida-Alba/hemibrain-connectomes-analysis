# Cross-Dataset Comparison - Instruction

## Purpose

Compare connectivity between two or more datasets (e.g. `male-cns:v0.9` vs
`hemibrain:v1.2.1`) for the same source→target query.

## Quick start

1. Select 2+ **Datasets** (optional nicknames in the same order).
2. Add source and target neurons.
3. Click **Run Comparison** - a report plus matrices are produced for every
   threshold.

## Inputs

- **Datasets (2+)**, optional nicknames in the same order.
- Source / Target neurons (chips, paste, CSV/XLSX).
- **Mode**: `path` (edges discovered through paths) or `edge` (independent
  edge weights).
- **Synapse Thresholds**: comma-separated list, e.g. `3, 5, 10`.

## Advanced

- Pathfinding algorithm, top edges, skip bodyId level, cache-only offline,
  auto type mapping, min ratio/probability, output format, parallelism,
  hemisphere analysis (L/R separation, symmetry, conserved edges).

## Output

`comp_{source}_to_{target}_{datasets}_{timestamp}/` with per-threshold data,
matrices, visualizations and an interactive HTML report.
