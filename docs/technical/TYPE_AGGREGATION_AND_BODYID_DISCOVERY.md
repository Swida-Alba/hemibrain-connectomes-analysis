# Why FindAllPath Finds Paths at the bodyId Level First

Design rationale for the discovery-first pipeline in `FindAllPath`:

> The type-level results are **derived from** discovered bodyId-level paths —
> never computed independently of them. Aggregating all per-bodyId pairs
> into a type-level graph and running pathfinding on that graph alone would
> hide the differences *within* a type and report paths that no real chain
> of neurons can realize.

## 1. The pipeline (current design)

```
per-bodyId pair tables (layer by layer, from the source neurons)
   │
   │ 1. pan-graph edge limit (path-integrity trim: reachability filter +
   │    source/target reservation + adaptive dead-end refill)
   ▼
bodyId graph G ──────────► 2. bodyId-level pathfinding  (THE DISCOVERY)
   │                              │
   │                              ▼
   │                    edges_in_paths — the SET of unique (pre, post)
   │                    pairs that lie on at least one found bodyId path
   ▼                              │
   │                              ▼
   │                    3. each layer table is filtered down to its
   │                       in-path pairs (set intersection)
   ▼                              │
   ▼                              ▼
   │                    4. EnrichConnectionTable (pair dedup + weight
   │                       summation + type aggregation)
   ▼                              │
   ▼                              ▼
conn_types — type-level edges, every one backed by a real bodyId path
   │
   ▼
5. derive the type-level paths: map each discovered bodyId path to its
   type sequence, then VERIFY every hop against the type-edge table (no
   second pathfinding on a type-level graph, and no type-level edge
   limit — the bodyId discovery already bounds the search space; the
   Visualization Edge Limit `edgeN_limit` remains the only type-level
   cap, applied when drawing)
```

Every step after the discovery consumes only pairs that were **observed on
an actual bodyId path**. The type-level output in step 5 is a *summary* of
the discovery — the unique type sequences of the discovered bodyId paths —
not an independent source of truth.

### Why the type-level paths are derived, not re-searched

Step 5 was historically a second pathfinding on the type-level graph
(built from `conn_types`). Running a search there re-introduces the bundle
effect at the type level: it can report a type chain A→B→C whose hops are
each backed by *some* bodyId pair, but by **different** pairs, so no single
bodyId path realizes the chain. Deriving the type paths from the
discovered bodyId paths instead guarantees every reported type path is the
type sequence of a real bodyId path. It also preserves repeated-type
routes (A→B→A through two distinct B neurons) that a simple-path search on
the type graph silently drops. Measured on the battery in Section 5: the
old re-search added 0–3 phantom paths per query and missed 0–31 real
repeated-type paths; the derivation matches the discovered bodyId paths
exactly.

## 2. Why the type level cannot simply aggregate all pairs (the bundle effect)

A type is a **class label shared by many neurons**, and a type-level edge
A→B aggregates **all** bodyId pairs (aᵢ, bⱼ) with those types. Once
aggregated, the individual identities of aᵢ and bⱼ are lost — the type edge
is a *bundle* of parallel channels. A type-level path therefore exists
whenever each consecutive type pair has **some** channel, but nothing
guarantees that the channels **connect to each other**.

### Minimal synthetic example (the core failure mode)

```
bodyId pairs:                  type aggregation:
  a1 → b1   (A → B)              A → B   (w = 18)
  a2 → b2   (A → B)              B → C   (w = 16)
  b2 → c1   (B → C)
  b3 → c2   (B → C)

type-level graph:  A → B → C   ⇒   type path "A → B → C" EXISTS
```

But **no bodyId path** a → b → c exists:

- the only A→B edges land on **b1** and **b2**;
- the only B→C edges leave **b2** and **b3**;
- b1 has no outgoing edge, so a1 → b1 is a dead end;
- b2 → c1 works for a2, but b3 → c2 has no incoming A-edge.

The aggregation mixed the **B neurons that receive from A** with the **B
neurons that project to C** — two different populations sharing the same
type label. The type path A→B→C is a *phantom*: no neuron of type B is both
reached from an A-neuron and able to reach a C-neuron within one hop.

## 3. Where "hidden differences under the same type" appear

The design intent of the discovery step is exactly to prevent phantom paths
by requiring a **concrete chain of neurons**. The hidden within-type
differences that a pure type-level aggregation would gloss over:

| Hidden difference | Consequence for a type-only pipeline |
|---|---|
| **Split membership** — the B-neurons receiving from A differ from the B-neurons projecting to C | the type path A→B→C exists, no bodyId route does (synthetic example above) |
| **Dead-end members** — some B-neurons have only incoming (or only outgoing) edges | they inflate the type edge weight and can create phantom continuations through neurons that can never reach a target |
| **Cutoff interplay** — an A→B edge exists only at a depth where B→C would exceed `max_interlayer` | the type graph has no depth concept, so it reports routes the discovery correctly rejects as too long |
| **Target identity** — a B-neuron projects to *a* neuron of type C, but not to any **queried** target | the type path ends at the right *type* but at the wrong *neuron* |
| **Weak-route trimming** — the pan-graph edge limit keeps only the strongest usable edges | type aggregation preserves the type-pair presence of edges the discovery pruned as unusable |

The discovery is the only step that can answer the question *"does a
concrete chain of neurons exist, within the depth limit, ending at a
queried target?"* — the type level cannot.

## 4. Real-data example (male-cns:v1.0, R1-R6 → Tm3)

Query: 6 R1-R6 sources → 6 Tm3 targets, `max_interlayer=2`,
`min_synapse_num=3`. The type-only pipeline reports the phantom type path

```
R1-R6 → L1 → Tm1 → Tm3
```

Every hop exists at the bodyId level:

- an R1-R6 neuron (e.g. bodyId `274589`) → an L1 neuron (`34008`),
- that L1 neuron → a Tm1 neuron (`38214`),
- that Tm1 neuron → **a** Tm3 neuron (`117197`).

Yet the old pipeline does **not** report this path — and correctly so: the
Tm3 neuron `117197` is *not one of the six queried targets*. The Tm1 that
receives from the discovered L1 population projects to a different Tm3
neuron than the ones being searched. After aggregation the two Tm3 neurons
are indistinguishable ("same type"), and the phantom path appears.

## 5. Comparative measurements

`local_data/type_agg_eval.py` compares the two pipelines on real cached
connectomes: **OLD** (discovery + in-path aggregation, the current
pipeline) vs **NEW** (aggregate all per-bodyId pairs → type graph →
type-level pathfinding). Every new-only type path is then checked for a
simple bodyId route **from a queried source to a queried target** in (a)
the trimmed discovery graph (what the old pipeline searches) and (b) the
full untrimmed bodyId graph (routes the bodyId edge-limit trim discarded).
All queries: 6 sources × 6 targets, `min_synapse_num=3`, seed 7.

| Query (dataset, L) | bodyId paths | type OLD | type NEW | OLD missing | NEW-only | backed (trimmed) | backed (full only) | **phantom** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R1-R6→Tm3 L2 (male-cns, seed 3) | 50 | 11 | 21 | 0 | 10 | 0 | 0 | **10** |
| R1-R6→Tm3 L2 (male-cns) | 46 | 12 | 23 | 0 | 11 | 0 | 0 | **11** |
| R1-R6→Tm1 L2 (male-cns) | 101 | 24 | 29 | 0 | 5 | 0 | 0 | **5** |
| L1→Tm3 L2 (male-cns) | 399 | 27 | 133 | 0 | 106 | 0 | 0 | **106** |
| R1-R6→Tm3 L3 (male-cns) | 176 | 35 | 3 281 | 0 | 3 246 | 0 | 95 | **3 151** |
| KCg-m→MBON01 L2 (hemibrain) | 14 749 | 334 | 262 | **233** | 161 | 0 | 24 | **137** |
| **TOTAL** | | 443 | 3 749 | 233 | 3 539 | 0 | 119 | **3 420 (97 %)** |

Three findings:

1. **Phantom inflation dominates (97 % of new-only paths).** The bundle
   effect is not a corner case: on every query the shortcut reports paths
   with no bodyId route from a queried source to a queried target. At L3
   the type graph is dense enough that the shortcut explodes (3 281 vs 35
   backed paths) while staying ~97 % phantom.

2. **The shortcut can also LOSE real paths (hemibrain: 233 of 334 old
   paths missing).** The type-level edge limit re-ranks edges by the
   *new* weights (summed over all pairs instead of in-path pairs), so a
   weak-ranked type edge that the old pipeline keeps can be trimmed from
   the new type graph. `OLD ⊆ NEW` is *not* guaranteed — it happened to
   hold on the male-cns runs and broke on hemibrain.

3. **Deep layers: the recovery-vs-noise trade-off is unfavourable.** At
   L3, 95 new-only paths are backed in the full bodyId graph but not in
   the trimmed discovery graph (their routes use weak edges the bodyId
   edge-limit trim removed). Keeping them costs 3 151 phantoms — a 33:1
   noise-to-signal ratio — and still loses nothing from OLD there.

## 6. When is the type-only shortcut acceptable?

The shortcut (aggregate all pairs → type-level pathfinding, skipping the
bodyId discovery) is legitimate **only as a different artifact**, never as
a drop-in replacement for `allpaths`:

- ✅ as a **type-topology overview** of the discovered network (the role
  `network_early` plays) — it shows which type pairs connect, without
  claiming any path is real;
- ✅ when bodyId-level data is unavailable and only type-level edges exist;
- ❌ as the "allpaths" result — its phantom paths have no path metrics, no
  real route, and would silently change downstream analyses that assume
  each path is realized by neurons.

## 7. Related documents

- [Pathfinding Methods in FindAllPath](../core-features/PathFinding_Methods.md)
- [Pathfinding Algorithm Evaluation](PATHFINDING_ALGORITHM_EVALUATION.md)
- `user_warning_notes.txt` (written per run when edge limits or filters
  tilt the outputs)
