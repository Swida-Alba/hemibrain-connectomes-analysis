# Shortest Paths

Find only the **shortest (minimum hop-count) paths** between source and
target neuron groups. Backend: `FindNeuronConnection.FindShortestPath` — the
full FindAllPath pipeline (discovery, enrichment, type-path derivation,
outputs, visualization) with shortest-only enumeration.

## Semantics

- For every reachable (source, target) pair, the minimum-hop paths under the
  search criteria (Min Synapse Count / Min Connection Ratio / Min Traversal
  Prob.) are returned. **All tied shortest paths are kept**, each once.
- Distances are hop counts on the threshold-filtered graph (synapse weights
  are edge attributes, not path costs).

## Depth: unlimited by default

- **Max Layers (optional) = 0** means unlimited search depth. Layer
  discovery is a BFS and stops as soon as *every* target has been
  discovered: a target's first-discovery layer is its exact shortest hop
  distance, so deeper layers cannot change any result. This keeps an
  "unlimited" search cheap in practice.
- Set `Max Layers > 0` only as a safety cap for queries where targets may be
  unreachable (the worst case then fetches the whole forward cone of the
  sources).

## Why there is no algorithm selector

Shortest enumeration is a backward-BFS distance pass plus a guided DFS over
the shortest-path DAG — polynomial in the graph size. The combinatorial
path explosion that makes Find All Paths expensive does not exist here, so
the algorithm choice and the deep-search warning do not apply.

## Edge Limit – BodyIds: off by default

Unlike Find All Paths, the bodyId graph edge limit is **disabled by
default**:

- Its purpose (bounding the exponential path count) is unnecessary for
  shortest enumeration.
- Trimming keeps the strongest edges: it preserves pair *reachability* but
  not *shortest distance* — a dropped weak edge can silently inflate a
  reported distance.

Set it only to cap memory on extremely large graphs; the run then reports
shortest paths **within the trimmed graph** (noted in
`user_warning_notes.txt`).

## Everything else matches Find All Paths

Graph cache (depth-aware; shallow caches are extended instead of rebuilt),
early network preview, keyword exclusion, custom grouping, hemisphere
options (separate L/R, conserved-edge filter, symmetry analysis),
reciprocal enrichment, CSV/XLSX outputs, and the final
network/Sankey/heatmap visualizations.

## Output

```
{output_dir}/findshortestpath_{DATASET}_{source}_to_{target}_L{depth}w{minsyn}r{ratio}p{prob}_{timestamp}/
  *_allpaths_type.csv            # type-level shortest paths
  data_details/                  # parameters, neuron lists, connection tables
  visualization/                 # network / Sankey / heatmap HTML + inputs
  bodyId_visualization/          # (when Skip BodyId is off)
```

`Lmax` in the folder name means the search ran with unlimited depth.
