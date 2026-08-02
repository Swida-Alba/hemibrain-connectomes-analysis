# Direct Connections - Instruction

## Purpose

Find one-hop synaptic connections between source and target neuron groups.
Leave targets empty to get all downstream targets of the sources.

## Inputs

Same neuron-list input as Find All Paths (chips, paste, CSV/XLSX upload).
See [Input File Formats](input_formats.md).

## Key parameters

- **Min Synapse Count**, **Min Ratio**, **Min Traversal Prob.**: connection
  strength filters.
- **Edge Limit**: maximum output rows.
- **Exclude Intra-type Connections**: removes `type_pre == type_post` edges.
- **Filter By**: `bodyId` (per neuron) or `type` (aggregated).

## Advanced

- Custom source/target names, cache-only mode, custom save folder, and
  hemisphere options.

## Output

`finddirect_{DATASET}_{source}_to_{target}_{params}_{timestamp}/` containing
connection tables, matrices and an interactive network HTML.
