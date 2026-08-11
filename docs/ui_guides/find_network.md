# Network (FindNetwork)

Build the **mutual direct-connection network among the queried neurons** —
equivalent to Direct Connections with source == target == the query set.
Backend: `FindNeuronConnection.FindNetwork`, using the FindAllPath pipeline
for enrichment, hemisphere-aware analysis, and visualization.

<div class="callout warn">
  <b>Limited scope:</b> the network contains only direct connections whose
  BOTH endpoints are in the queried set — intermediate neurons are never
  involved. For a more complete network that also involves intermediate
  neurons, use the <b>Find Path</b> tab with <b>Find Reciprocal
  Connections</b> enabled.
</div>

## Quick start

1. Add the **Query Neurons** (types, bodyIds, patterns, or an uploaded list).
2. Set **Min Synapse Count** (and the ratio/probability thresholds if needed).
3. Click **Find Network**.

## What is computed

- One fetch of all downstream connections of the query set, filtered to
  edges whose post-synaptic partner is also in the set (both directions
  kept as directed edges).
- Enrichment with the FindAllPath semantics: connection ratios and
  traversal probabilities with **global** incoming-weight denominators,
  LabelMapper standardization, custom grouping when a mapping is selected.
- Hemisphere-aware analysis (same helpers as Find All Paths):
  **Separate Hemispheres** labels, **Symmetry Analysis** on the unfiltered
  table, and the **Keep Only Hemisphere-Conserved Edges** filter.

## Outputs (nothing redundant)

```
{output_dir}/findnetwork_{DATASET}_{group}_w{minsyn}r{ratio}p{prob}_{timestamp}/
  parameters.txt, all_attributes.json
  data_details/
    neurons.csv                     # all queried neurons
    connection_type.csv             # type-level mutual connections
    connection_info_bodyId.csv      # (when Skip BodyId is off)
    connection_custom_groups.csv    # (when a custom grouping is used)
    hemisphere_unconserved_edges.csv# (when the conserved filter is on)
  visualization/
    Network_*.html, Heatmap_*.html  # NO Sankey
    visualization_data/             # visualization inputs
  user_warning_notes.txt            # (when any output-affecting option applied)
```

Visualizations are produced by the FindAllPath visualization backend
(VisualizePath) as **network + heatmap only** — the Sankey diagram and the
path/matrix exports of the path tools are intentionally omitted.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| Min Synapse Count | 3 | Minimum edge weight |
| Min Connection Ratio | 0.0 | Minimum weight/post ratio |
| Min Traversal Prob. | 0.0 | Traversal probability threshold |
| Search Columns | auto | Columns searched when resolving neuron names |
| Output Format | csv | csv or xlsx |
| Skip BodyId in Output | on | Skip the bodyId-level connection table |
| Hemisphere options | off | Same semantics as Find All Paths |

## Find Network vs Find Path + Find Reciprocal

| | Find Network | Find Path + Find Reciprocal |
|---|---|---|
| Node set | exactly the queried neurons | queried neurons + all discovered intermediates |
| Edges | direct connections within the query set | path edges, plus ALL direct connections among path-graph nodes |
| Depth | 1 hop only | up to Max Layers |
| Use when | you want the induced connectivity of a known neuron set | you want the neighborhood reachable between sources and targets |
