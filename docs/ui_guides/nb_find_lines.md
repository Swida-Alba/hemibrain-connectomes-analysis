# Find Driver Lines (NeuronBridge) - Instruction

## Purpose

Find GAL4 / Split-GAL4 driver lines whose expression matches EM neurons,
using NeuronBridge color-depth search (CDS) or point-pattern matching (PPPM).

## Inputs

- **EM Neurons**: bodyIds, types, instances, or patterns.
- **Dataset**: restrict to one dataset or search all.
- **Algorithm**: `cds` (fast), `pppm`, or `both`.

## Advanced

- Region (Brain/VNC), workers, sort by max score or completeness.
- Image downloads: formats/types, per-line cap, FlyLight collections,
  simple mode, organize by region.
- PDF/PPTX summary layout and background color.

## Output

`findlines_{DATASET or ALL}_{query}_{timestamp}/` with per-query CSVs,
summary tables (GAL4/Split separated when enabled) and downloaded images.

## Tip

Querying multiple types together finds lines labeling ALL of them; run
separate queries for different groups.
