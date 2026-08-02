# Find All Paths - Instruction

## Purpose

Discover multi-hop pathways between source and target neuron groups in a
single dataset (e.g. `aMe12` → `aMe10`).

## Quick start

1. Add **Source Neurons** (e.g. `aMe12`, `aMe10`) and **Target Neurons**
   (e.g. `PPL101`, `PPL103`) as chips, paste, or upload a list.
2. Pick the **Dataset** and an **Output Directory**.
3. Keep **Max Intermediate Layers = 2**, then click **Find All Paths** and
   watch the live log.

## Inputs

- **Source / Target Neurons**: type names, bodyIds, or patterns
  (`KC.*`, `PPL.*`). Add chips with Enter, paste a list, or upload
  CSV/XLSX (see [Input File Formats](input_formats.md)).
- **Filter mode**: applies the chosen rule (exact / starts with / contains /
  ends with / regex) to every entry.
- **Dataset**: NeuPrint (`male-cns:v1.0`, `hemibrain:v1.2.1`, ...) or local
  FlyWire datasets.

## Key parameters

- **Max Intermediate Layers**: path length (0 = direct, 2 = two intermediates).
- **Min Synapse Count**: filters weak connections.
- **Min Connection Ratio / Traversal Prob.**: strength thresholds.
- **Algorithm**: `Bidirectional` (fastest), `DP`, `MemoizedDFS`, `DFS`.
- **Edge Limit**: caps memory for highly connected neurons.

## Advanced

- Hemisphere analysis (L/R separation, symmetry, conserved edges, reciprocal).
- Cache-only offline mode and a custom save folder name.

## Output

One folder per run:
`findpath_{DATASET}_{source}_to_{target}_{params}_{timestamp}/` with path
CSVs, connection matrices and interactive HTML (Sankey/heatmap/network).

## Tips

- Keep the query small first; larger networks (hundreds of thousands of
  neurons) take minutes.
- Re-running with the same dataset uses the local cache for 10-100x speedup.
