# Homolog Finding - Instruction

## Purpose

Find potential cross-dataset homologs of a source neuron via connectivity
profile similarity (e.g. `aMe12` in `male-cns` → candidates in `hemibrain`).

## Quick start

1. Enter the **Source Neuron** (type or bodyId).
2. Choose **Source Dataset** and **Target Dataset**.
3. Click **Find Homologs** (Fast Search is on by default).

## Inputs

- **Source Neuron**: one type or bodyId.
- **Source / Target Dataset**.
- **Top N Candidates**, **Top K Partners**, **Min Types (M)**.
- **Similarity Metric**: rank_union (default), rank_corr, jaccard, cosine.

## Options

- **Fast Search**: adjacency-expansion candidate discovery (recommended).
- **Vector Pre-filtering**: cosine pre-filter for speed.
- **Include untyped partners**: 2-hop expansion for untyped 1-hop partners.
- **Visualize Candidates**: render top matches as 3D skeletons.
- **Auto Type Mapping**: canonicalize partner type names across datasets.

## Output

`homologs_{SRC}_to_{TGT}_{query}_{timestamp}/` with bodyId results,
type summary and visualization subfolders.
