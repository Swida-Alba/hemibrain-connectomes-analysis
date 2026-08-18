# FAFB and NeuPrint Skeleton Pipeline Plan

Status: planning only. No implementation is included in this document.

Date: 2026-08-18

## Goals

- Make `fast` the default tube-mode skeleton pipeline.
- Use a two-stage FAFB fast pipeline: reduce the raw SWC to 25% of its nodes, then run the fine tube-mesh and surface-decimation path.
- Keep raw SWC and CAVE MeshNeuron representations separate and unambiguous.
- Make line rendering lightweight by simplifying TreeNeurons before plotting.
- Make sub-called skeleton visualizations from Similarity tabs and NeuronBridge Find Neurons use line mode by default.
- Fall back to CAVE when the local FAFB healed bundle is unavailable or unprepared and no suitable local cache exists.

## User-facing defaults

| Context | Default representation | Node simplification | Surface simplification |
| --- | --- | ---: | ---: |
| Dedicated Skeleton tab, NeuPrint tube | Tube mesh | Existing NeuPrint `fast` behavior | Existing fast target |
| Dedicated Skeleton tab, FAFB tube | Tube mesh | Retain 25% of raw SWC nodes in `fast` | 95% removal |
| Similarity-tab sub-visualization | Lines | NeuPrint: 50% reduction; FAFB: 90% reduction | Not applicable |
| NeuronBridge Find Neurons sub-visualization | Lines | NeuPrint: 50% reduction; FAFB: 90% reduction | Not applicable |
| Explicit `fine` or `artistic` selection | User-selected | No implicit fast node reduction | Existing selected method |

Percentages in this plan refer to reduction unless stated as “retain”. Thus:

- FAFB fast retains 25% of nodes.
- NeuPrint line mode targets 50% node reduction, approximately factor 2.
- FAFB line mode targets 90% node reduction, approximately factor 10.

Topology-preserving simplification may retain more nodes than the nominal target because roots, branch points, and terminal nodes must remain.

## Dataset-specific pipelines

### FAFB fast tube mode

```text
raw healed-ZIP SWC
  -> in-memory topology-aware reduction to a 25% node target
  -> TreeNeuron -> tube MeshNeuron
  -> fine surface decimation
  -> render/export
```

The 25% target is a node-count target, not merely a fixed downsampling factor. A calibrated factor near 4 may be used as the initial estimate, but the implementation must account for preserved topology nodes and the achievable minimum for each neuron.

The FAFB fine path after SWC loading means tube conversion plus fine surface decimation. NeuPrint-specific smoothing, resampling, and radius transformation must not be applied to native FAFB SWCs.

### FAFB fine mode

```text
raw healed-ZIP SWC
  -> TreeNeuron -> tube MeshNeuron
  -> fine surface decimation
  -> render/export
```

Fine mode remains available as an explicit alternative and does not perform the fast 25% node reduction first.

### FAFB artistic mode

Artistic mode remains an explicit alternative using its vertex-clustering surface-decimation behavior. It must not accidentally combine fast node reduction with artistic clustering unless explicitly requested.

### NeuPrint tube mode

`fast` becomes the public default, while `fine` and `artistic` remain selectable. Existing NeuPrint method-specific morphology and mesh behavior should be preserved unless required by the new default selection.

### Line mode

Local TreeNeuron sources follow this order:

```text
raw SWC
  -> in-memory node simplification
  -> direct line plotting
```

The dataset defaults are:

- NeuPrint: 50% node reduction.
- FAFB: 90% node reduction.

CAVE returns MeshNeurons rather than SWCs. If a CAVE MeshNeuron is used in line mode, it is skeletonized only at the line-render boundary, in memory, and then receives the applicable node simplification. CAVE MeshNeurons are never skeletonized for tube rendering or cache writing.

## FAFB source-resolution and fallback policy

The source resolver must not require the local healed bundle when CAVE access is available.

For each FAFB body ID, resolve sources in this order:

1. Local healed-ZIP SWC, when the local bundle is prepared and the member exists.
2. Existing local raw-skeleton cache, when applicable.
3. Prepared CAVE MeshNeuron cache at the configured cache policy.
4. CAVE API fetch for remaining or missing bodies.

This applies when the healed ZIP is absent, has not been prepared, or does not contain a requested body. The access guard should report the missing local source but continue to CAVE rather than failing prematurely when a CAVE token is configured.

If CAVE access is unavailable, the user should receive an actionable message identifying the missing local bundle/cache and the required CAVE configuration.

## Cache and representation rules

- The healed ZIP and raw SWC caches remain canonical raw skeleton sources.
- CAVE-fetched MeshNeurons use the separate prepared mesh cache in `.pkl.zst` format.
- The prepared CAVE cache keeps the established policy: 95% branch/surface simplification, 80% soma-region simplification, and the configured soma radius.
- A MeshNeuron must never be serialized as an SWC merely to satisfy a skeleton API.
- A skeletonized CAVE replacement must remain an in-memory line-mode product and must not overwrite the MeshNeuron cache.
- `use_cache=True`: read prepared CAVE cache first, fetch missing meshes, and write prepared mesh results.
- `use_cache=False`: fetch online only; do not read or write local CAVE, SWC, MeshNeuron, or state caches.
- Raw SWCs must remain unchanged by render-time node simplification.

## UI and caller behavior

- Set the public tube pipeline default to `fast`.
- Keep `fine` and `artistic` as explicit user choices.
- Set `skeleton_mode="line"` by default for Similarity-tab sub-called visualizations.
- Set `skeleton_mode="line"` by default for NeuronBridge Find Neurons sub-called visualizations.
- Preserve an explicit caller override to request tube mode.
- In line mode, the tube pipeline selector should not trigger mesh conversion or surface decimation.
- Keep the dedicated Skeleton tab independently configurable rather than inheriting the analysis-caller line default.

## Progress reporting

Use the existing shared progress display and expose distinct stages:

- resolving local SWC, prepared CAVE cache, or CAVE API source;
- node simplification, including current body ID and resulting node count;
- TreeNeuron-to-tube conversion;
- fine surface decimation;
- line plotting or export.

The line path must not display a tube/mesh stage. CAVE fallback progress must distinguish cache hits from online fetches.

## Tests and validation

Add focused tests for:

- FAFB fallback when the healed ZIP is absent or unprepared;
- prepared CAVE cache hit, CAVE fetch miss, and cache write;
- `use_cache=False` producing no local cache access or writes;
- exact source-priority behavior when both SWC and CAVE mesh sources exist;
- FAFB fast ordering: SWC node reduction before tube conversion and fine surface decimation;
- 25% FAFB node-retention target and topology-preservation floor;
- NeuPrint 50% line simplification default;
- FAFB 90% line simplification default;
- Similarity and NeuronBridge sub-callers defaulting to line mode;
- line-only CAVE MeshNeuron skeletonization;
- no CAVE MeshNeuron skeletonization in tube mode;
- raw SWC and `.pkl.zst` cache representations remaining unchanged by rendering;
- progress stages and body-level diagnostics.

After implementation, run the focused tests, the full suite, and real-data benchmarks for:

- FAFB fast: 25% SWC nodes followed by fine mesh decimation;
- FAFB line: 90% node reduction;
- NeuPrint line: 50% node reduction;
- Similarity and NeuronBridge line-mode defaults;
- local FAFB source present, absent, and CAVE-fallback cases.

## Existing benchmark baseline

The read-only benchmarks used real aMe12 data and the direct `navis.plot3d(..., radius=False)` path.

| Dataset | Raw median nodes | Line setting | Median output nodes | Direct line plot |
| --- | ---: | ---: | ---: | ---: |
| FAFB | 36,443 | 90% reduction | 4,338 | 0.0129 s |
| NeuPrint male-cns v1.0 | 10,152 | 50% reduction | 3,578 | 0.0054 s |

The complete benchmark artifacts are under:

- `visualization_exports/fafb_line_mode_swc_benchmark_20260818_172837/`
- `visualization_exports/neuprint_line_mode_benchmark_20260818_190834/`

## Non-goal for this turn

This document records the approved-direction plan only. No source code, UI code, tests, caches, or configuration defaults are changed by creating this document.
