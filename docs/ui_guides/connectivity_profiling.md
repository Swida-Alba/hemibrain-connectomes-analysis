# Connectivity Profiling - Instruction

## Purpose

Compare connectivity profiles of neuron types (or bodyIds) inside one
dataset using Jaccard, cosine and rank-correlation similarities.

## Quick start

1. Enter 2+ neurons (e.g. `aMe12`, `aMe10`, `aMe9`).
2. Pick the dataset; keep **Top K Partners = 15**.
3. Click **Run Profiling** - heatmaps and similarity matrices are generated.

## Inputs

- **Neurons to Compare**: at least two entries (types, bodyIds or patterns).
- Dataset and output directory.

## Profile construction

- **Top K Partners**: partners per direction used in each profile.
- **Min Unique Types (M)**: minimum partner-type count; K expands
  automatically when needed.
- **Min Synapse Threshold**: connection weight cutoff.
- **Aggregation Level**: `type` (grouped) or `bodyid` (per neuron).
- **Direction**: upstream / downstream / both.

## Advanced

- BodyId-level computation: auto / skip / always.
- Ward-clustered heatmaps, show figures.
- "Pre-build Full Dataset Cache" is **off** by default - leave it off; the
  tool uses the connections already cached.

## Output

`profiling_{DATASET}_{query}_{timestamp}/` with profile CSVs, similarity
matrices and interactive heatmaps.
