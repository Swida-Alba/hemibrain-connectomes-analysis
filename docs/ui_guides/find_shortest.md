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

## Depth cap (default: 2 intermediate layers)

- **Max Intermediate Layers (optional) = 0** means unlimited search depth. Layer
  discovery is a BFS and stops as soon as *every* target has been
  discovered: a target's first-discovery layer is its exact shortest hop
  distance, so deeper layers cannot change any result. This keeps an
  "unlimited" search cheap in practice.
- The default is 2, matching the Find All Paths tab. Increase it when a
  query needs deeper intermediate layers; there is no artificial upper
  bound on the integer value.
- Set `Max Intermediate Layers > 0` only as a safety cap for queries where targets may be
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

## Visualization Edge Limit

Visualization Edge Limit only caps the number of edges drawn in each
network, heatmap, or Sankey view. It does not trim the fetched graph and
does not change shortest-path discovery.

## Everything else matches Find All Paths

Graph cache (depth-aware; shallow caches are extended instead of rebuilt),
early network preview, keyword exclusion, custom grouping, hemisphere
options (separate L/R, conserved-edge filter, symmetry analysis),
reciprocal enrichment, CSV/XLSX outputs, and the final
network/Sankey/heatmap visualizations.

## All-neurons queries (all_neurons)

Typing the special chip `all_neurons` as the **source or target** loads the
full (typed) neuron set on that side, so the run fetches every adjacent
neuron at the given thresholds (Min Synapse Count / Min Connection Ratio /
Min Traversal Prob.):

- `all_neurons` replaces every other chip in the same input.
- Both source and target = `all_neurons` is not allowed.
- An `all_neurons` side forces **Max Intermediate Layers = 0** (direct
  connections only) — the search stays bounded instead of exploring the
  whole graph. With shortest-path semantics this reports every direct edge
  (one hop) from or to the other side.

The backend (`InitializeNeuronInfo`) enforces the same rules, so script and
API callers get identical behavior.

## Output

Same layout as Find All Paths (see the Complete Paths guide):

```
{output_dir}/find-paths-shortest_MCNS_aMe12_to_PPL101_L1w3r0p0_20260815_142717/
  aMe12_to_PPL101_allpaths_type.csv  # shortest paths (same columns as Find All Paths)
  all_attributes.json
  parameters.txt, user_warning_notes.txt
  data_details/                      # matrices, connection_type.csv, neuron lists, parameters.csv
  visualization/                     # Network / Sankey / Heatmap HTML + visualization_data/ inputs
  hemisphere_symmetry/...            # when Symmetry Analysis is on
  find_reciprocal/...                # when Find Reciprocal Connections is on
```

`Lmax` in the folder name means the search ran with unlimited depth.

With **Skip BodyId** off, bodyId-level tables (`..._allpaths_bodyId.csv`) are
written next to the type-level ones. The full reference is in
`docs/OUTPUT_FILES.md`.
