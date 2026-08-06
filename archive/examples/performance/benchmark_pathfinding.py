"""
Benchmark the FastGraph pathfinding algorithms (DP, Bidirectional,
MemoizedDFS forward, DFS backward, MeetInMiddle).

Measures wall time, peak memory (tracemalloc) and path counts on:

1. Synthetic layered DAGs with controlled branching (clean scaling behaviour).
2. A real hemibrain v1.2.1 connectome subgraph (LC -> MBON/PPL queries),
   pruned the same way the FindAllPath pipeline prunes dead ends.

Each algorithm runs twice per scenario: once without tracing (wall time)
and once under tracemalloc (peak memory + traced time). A 180 s timeout
guards the slower methods; runs that exceed it are reported as timeouts.

Usage:
    python examples/performance/benchmark_pathfinding.py [--json out.json]
"""

import argparse
import json
import random
import signal
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vispath-subproject" / "src"))

from vispath_pkg.fast_graph_core import FastGraph  # noqa: E402

# UI name -> FastGraph generator. Names match the algorithms:
# MemoizedDFS = memoized DFS (forward), DFS = memoized DFS (backward),
# MeetInMiddle = meet-in-the-middle. Backtracking is intentionally not
# benchmarked: it is unusable on real connectomes beyond 3 intermediate
# layers (see the evaluation docs).
ALGORITHMS = {
    "DP": lambda G, s, t, c: G.find_paths_backward_dp(s, t, c),
    "Bidirectional": lambda G, s, t, c: G.find_paths_bidirectional_bfs(s, t, c),
    "MemoizedDFS": lambda G, s, t, c: G.find_paths_memoized_dfs(s, t, c),
    "DFS": lambda G, s, t, c: G.find_paths_memoized_dfs(s, t, c, direction="backward"),
    "MeetInMiddle": lambda G, s, t, c: G.find_paths_meet_in_the_middle(s, t, c),
}

TIMEOUT_S = 180
MAX_PATHS = 5_000_000  # guard: stop collecting beyond this many paths


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


def collect(gen):
    """Collect generator output into a list, guarding runtime/memory."""
    paths = []
    for p in gen:
        paths.append(p)
        if len(paths) >= MAX_PATHS:
            break
    return paths


def run_one(G, sources, targets, cutoff, trace):
    """Run one algorithm scenario; returns a stats dict (or timeout dict)."""
    stats = {}
    for name, make_gen in ALGORITHMS.items():
        if trace:
            tracemalloc.start()
        t0 = time.perf_counter()
        try:
            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(TIMEOUT_S)
            try:
                paths = collect(make_gen(G, sources, targets, cutoff))
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            if trace:
                peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
                tracemalloc.stop()
                stats[name] = {
                    "time_s": round(time.perf_counter() - t0, 3),
                    "peak_mb": round(peak_mb, 1),
                    "paths": len(paths),
                    "aborted": len(paths) >= MAX_PATHS,
                }
            else:
                stats[name] = {
                    "time_s": round(time.perf_counter() - t0, 3),
                    "paths": len(paths),
                    "aborted": len(paths) >= MAX_PATHS,
                }
        except _Timeout:
            if trace:
                tracemalloc.stop()
            stats[name] = {"timeout": True}
    return stats


# ---------------------------------------------------------------------------
# Synthetic graphs
# ---------------------------------------------------------------------------

def layered_dag(layers, width, branching, seed=0):
    """Directed layered DAG: each node connects to `branching` random nodes
    of the next layer. Sources = first layer, targets = last layer."""
    rng = random.Random(seed)
    G = FastGraph()
    nodes = [[f"L{l}_{i}" for i in range(width)] for l in range(layers)]
    for l in range(layers - 1):
        for u in nodes[l]:
            for v in rng.sample(nodes[l + 1], k=min(branching, width)):
                G.add_edge(u, v, 1.0)
    return G, nodes[0][:2], nodes[-1][:2]


# ---------------------------------------------------------------------------
# Real connectome subgraph
# ---------------------------------------------------------------------------

def real_subgraph(conn_path, neuron_index_path, source_types, target_types,
                  depth=3, node_cap=30000, min_weight=5):
    """Build the search-space subgraph for a real query, mirroring the
    FindAllPath pipeline: keep only connections with >= `min_weight`
    synapses (default 5, the UI default Min Synapse Count), take the nodes
    within `depth` hops of the sources (forward) or the targets (backward),
    then the induced edges, then drop nodes that can no longer reach a
    target within `depth` hops.

    The closures run as Polars joins over the parquet cache so the full
    connectome is never materialised as Python objects.
    """
    import polars as pl

    conn = (
        pl.read_parquet(conn_path)
        .select(["bodyId_pre", "bodyId_post", "weight"])
        .filter(pl.col("weight") >= min_weight)
        .drop_nulls()
        .unique(subset=["bodyId_pre", "bodyId_post"])
    )
    ni = pl.read_parquet(neuron_index_path).filter(
        pl.col("type").is_not_null() & (pl.col("type") != "")
    )

    def bodyids(types, limit):
        out = []
        for t in types:
            ids = (
                ni.filter(pl.col("type") == t)
                .select("bodyId")
                .head(limit)
                .to_series()
                .to_list()
            )
            out.extend(ids)
        return out

    sources = bodyids(source_types, limit=2)
    targets = bodyids(target_types, limit=2)
    if not sources or not targets:
        raise SystemExit(f"No bodyIds resolved: sources={sources} targets={targets}")

    def closure(start_ids, pre_col, post_col, d):
        """All nodes reachable from start_ids within d hops (set-expansion)."""
        seen = set(start_ids)
        frontier = pl.DataFrame({pre_col: start_ids}).unique()
        for _ in range(d):
            reached = (
                conn.join(frontier, on=pre_col, how="inner")
                .select(post_col)
                .unique()
                .to_series()
                .to_list()
            )
            new = [n for n in reached if n not in seen]
            if not new or len(seen) > node_cap:
                break
            seen.update(new)
            frontier = pl.DataFrame({pre_col: new})
        return seen

    fwd_zone = closure(sources, "bodyId_pre", "bodyId_post", depth)
    bwd_zone = closure(targets, "bodyId_post", "bodyId_pre", depth)
    zone = fwd_zone & bwd_zone
    print(f"    search zone: {len(fwd_zone)} fwd, {len(bwd_zone)} bwd, "
          f"{len(zone)} intersecting nodes", flush=True)

    zone_df = pl.DataFrame({"node": sorted(zone)})
    edges = (
        conn.join(zone_df.rename({"node": "bodyId_pre"}), on="bodyId_pre", how="inner")
        .join(zone_df.rename({"node": "bodyId_post"}), on="bodyId_post", how="inner")
        .unique()
    )
    print(f"    induced edges: {edges.height:,}", flush=True)

    G = FastGraph()
    for u, v, _w in edges.iter_rows():
        G.add_edge(u, v, 1.0)

    # Dead-end pruning: keep only nodes that can reach a target within depth.
    keep = set(targets)
    frontier = set(targets)
    for _ in range(depth):
        if not frontier:
            break
        nxt = set()
        f = pl.DataFrame({"bodyId_post": pl.Series(sorted(frontier), dtype=pl.Utf8)})
        for u in (
            edges.join(f, on="bodyId_post", how="inner")
            .select("bodyId_pre")
            .unique()
            .to_series()
            .to_list()
        ):
            if u in zone and u not in keep:
                keep.add(u)
                nxt.add(u)
        frontier = nxt
    G = G.subgraph(keep)
    return G, sources, targets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=str, default="",
                        help="Write results as JSON to this path")
    args = parser.parse_args()

    results = {"scenarios": [], "timeout_s": TIMEOUT_S}

    scenarios = [
        ("synthetic L5 (cutoff 4)", *layered_dag(5, 25, 3), 4),
        ("synthetic L7 (cutoff 6)", *layered_dag(7, 25, 3), 6),
        ("synthetic L6 dense (cutoff 5)", *layered_dag(6, 30, 5), 5),
    ]

    conn_path = ROOT / "cache" / "hemibrain_v1_2_1" / "connections.parquet"
    ni_path = ROOT / "cache" / "hemibrain_v1_2_1" / "neuron_index.parquet"
    if conn_path.exists() and ni_path.exists():
        G, sources, targets = real_subgraph(
            conn_path, ni_path,
            source_types=["LPLC1", "LC4"],
            target_types=["MBON01", "MBON03", "PPL101"],
            depth=3,
        )
        scenarios.append(
            (f"hemibrain v1.2.1 LC->MBON/PPL (cutoff 3) "
             f"[{G.number_of_nodes()} nodes, {G.number_of_edges()} edges]",
             G, sources, targets, 3)
        )
        scenarios.append(
            (f"hemibrain v1.2.1 LC->MBON/PPL (cutoff 4) "
             f"[{G.number_of_nodes()} nodes, {G.number_of_edges()} edges]",
             G, sources, targets, 4)
        )

        # Deep scenarios (4 and 5 intermediate layers = cutoff 5/6) need a
        # smaller, tighter query to stay tractable: a single source type
        # (LPLC1) to a single target type (MBON01), same ≥5-synapse filter.
        # The closure depth must match the cutoff (6) or the targets fall
        # outside the search zone entirely.
        G_small, sources_small, targets_small = real_subgraph(
            conn_path, ni_path,
            source_types=["LPLC1"],
            target_types=["MBON01"],
            depth=6,
        )
        small_label = (f"hemibrain v1.2.1 LPLC1->MBON01 "
                       f"[{G_small.number_of_nodes()} nodes, "
                       f"{G_small.number_of_edges()} edges]")
        scenarios.append((f"{small_label} (cutoff 5)", G_small, sources_small, targets_small, 5))
        scenarios.append((f"{small_label} (cutoff 6)", G_small, sources_small, targets_small, 6))
    else:
        print("SKIP real-connectome scenario: cache files not found")

    print("=" * 100)
    for label, G, sources, targets, cutoff in scenarios:
        print(f"\n### {label}")
        print(f"    graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
              f"{len(sources)} sources, {len(targets)} targets, cutoff {cutoff}")
        row = {"label": label, "nodes": G.number_of_nodes(),
               "edges": G.number_of_edges(), "sources": len(sources),
               "targets": len(targets), "cutoff": cutoff}

        fast = run_one(G, sources, targets, cutoff, trace=False)
        traced = run_one(G, sources, targets, cutoff, trace=True)

        # Equivalence check: all algorithms must return the same path set
        # (only when the runs are small enough to recollect cheaply).
        if not any(v.get("timeout") for v in fast.values()):
            if all(v["paths"] < 300_000 for v in fast.values()):
                ref = None
                equal = True
                for name, make_gen in ALGORITHMS.items():
                    got = frozenset(tuple(p) for p in collect(
                        make_gen(G, sources, targets, cutoff)))
                    if ref is None:
                        ref = got
                    elif got != ref:
                        equal = False
                row["equivalent"] = bool(equal)
                row["path_set_size"] = len(ref)

        print(f"    {'algorithm':<14} {'time (s)':>10} {'peak (MB)':>11} "
              f"{'paths':>10}   note")
        for name in ALGORITHMS:
            u = fast[name]
            t = traced[name]
            if u.get("timeout"):
                print(f"    {name:<14} {'TIMEOUT >180s':>22}   (both runs)")
                row[name] = {"time_s": None, "peak_mb": None, "timeout": True}
                continue
            note = "aborted@5M" if (u.get("aborted") or t.get("aborted")) else ""
            peak = t.get("peak_mb", "n/a")
            print(f"    {name:<14} {u['time_s']:>10.2f} {str(peak):>11} "
                  f"{u['paths']:>10,}   {note}")
            row[name] = {
                "time_s": u["time_s"],
                "traced_time_s": t.get("time_s"),
                "peak_mb": t.get("peak_mb"),
                "paths": u["paths"],
                "aborted": bool(u.get("aborted") or t.get("aborted")),
            }
        results["scenarios"].append(row)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {args.json}")


if __name__ == "__main__":
    main()
