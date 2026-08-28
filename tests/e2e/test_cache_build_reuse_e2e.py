"""Cache BUILD and REUSE e2e — 3D visualization + similar-neuron finding.

Runs against the REAL hemibrain v1.2.1 NeuPrint server (skipped when the
network is unavailable) and verifies the post-Mission-1 cache contract end
to end:

3D skeleton visualization (``VisualizeSkeleton.plot_neurons``):
1. The explicit ``fast`` pipeline BUILDS the shared raw compressed-SWC cache.
2. A second render REUSES the cache: with the skeleton fetcher patched to
   raise, the render still completes — pure cache hits, zero fetches.
3. A less-simplified render (< 0.9) reuses the same shared raw skeletons;
   simplification happens only in memory during visualization.

Similar-neuron finding (``MorphologyComparer``):
4. A bodyId query BUILDS its vector: ``fetch_skeleton_on_demand`` fetches
   raw, persists a shared compressed raw skeleton AND emits the raw vector.
   Both writes are SANDBOXED (tmp skeleton dir + in-memory vector append), so
   the test never mutates the production caches.
5. The search REUSES the vector: with the fetcher patched to raise, the
   query still returns same-type neurons — pure vector-cache hits, and the
   second run reproduces identical results.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import navis  # noqa: E402
import navis.interfaces.neuprint as neu  # noqa: E402
import morphology as morph  # noqa: E402
import visualize_skeleton as vs_mod  # noqa: E402
from visualize_skeleton import VisualizeSkeleton  # noqa: E402

DATASET = "hemibrain:v1.2.1"
FOLDER = morph._dataset_folder(DATASET)
NEUPRINT_SERVER = "neuprint.janelia.org"
# aMe12 members (verified fetchable, type has many vector-cached members)
PROBE_IDS = [911332304, 1158631810]


def _network_available() -> bool:
    try:
        import socket
        socket.create_connection((NEUPRINT_SERVER, 443), timeout=5).close()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not _network_available(),
        reason="NeuPrint server unreachable (network required for this e2e)",
    ),
]


@pytest.fixture(scope="module")
def neuprint_client():
    """Production NeuPrint client (token from the standard manager)."""
    import neuprint
    from utils.token_manager import token_manager
    token = ""
    try:
        token = token_manager.get_token("NEUPRINT_TOKEN")
    except Exception:
        pass
    client = neuprint.Client(NEUPRINT_SERVER, dataset=DATASET, token=token)
    neuprint.set_default_client(client)
    return client


def _patch_layer_resolution(monkeypatch):
    """Construction resolves layers via statvis; the cache pipeline itself is
    what is under test, so layer data is injected directly."""
    monkeypatch.setattr(
        vs_mod.sv, "getNeurons",
        lambda *a, **k: (pd.DataFrame(columns=["bodyId"]), pd.DataFrame(),
                         "e2e_probe", None),
    )


def _make_vs(work_dir, client, simplification=0.9, cache_neurons=True):
    vs = VisualizeSkeleton(
        dataset=DATASET,
        neuron_layers=["aMe12"],
        client=client,
        verbose=False,
        output_dir=str(work_dir),
        include_timestamp=False,
        cache_neurons=cache_neurons,
        data_folder=str(work_dir),
        script_path=str(work_dir),  # sandbox the cache root
        skeleton_mesh_simplification=simplification,
        # Fast is a visualization simplification pipeline. It still uses the
        # shared raw cache and never selects a pull-time skeleton format.
        neuprint_skeleton_pipeline="fast",
        brain_mesh="none",
        show_fig=False,
        skip_synapse=True,
        backend="plotly",
    )
    vs.client_type = "neuprint"
    vs.neuron_dfs = [pd.DataFrame({"bodyId": PROBE_IDS})]
    vs.roi_dfs = [None]
    vs.layer_criteria = [None]
    vs.layer_names = ["e2e_probe"]
    return vs


@pytest.fixture(scope="module")
def built_cache(tmp_path_factory, neuprint_client):
    """One full first render: BUILD phase shared by the reuse tests."""
    work = tmp_path_factory.mktemp("vs_build")
    # module-scoped fixture: patch layer resolution manually (the cache
    # pipeline itself is what is under test, not statvis layer lookup)
    orig_get_neurons = vs_mod.sv.getNeurons
    try:
        vs_mod.sv.getNeurons = lambda *a, **k: (
            pd.DataFrame(columns=["bodyId"]), pd.DataFrame(), "e2e_probe", None)
        vs = _make_vs(work, neuprint_client)
    finally:
        vs_mod.sv.getNeurons = orig_get_neurons
    vs.plot_neurons()
    cache_dir = Path(work) / "cache" / FOLDER / "skeletons"
    for bid in PROBE_IDS:
        assert (cache_dir / "raw_skeletons" / f"{bid}.swc.zst").exists()
    assert not (cache_dir / ".level").exists()
    assert not list(cache_dir.glob("*.pkl"))
    return work


# ---------------------------------------------------------------------------
# 3D visualization: build then reuse
# ---------------------------------------------------------------------------

class TestVisualizationCacheBuildReuse:
    def test_first_render_builds_shared_raw_cache(
            self, built_cache, neuprint_client):
        """The BUILD render persists raw level-0 SWC only."""
        work = built_cache
        cache_dir = Path(work) / "cache" / FOLDER / "skeletons"
        raw_dir = cache_dir / "raw_skeletons"
        for bid in PROBE_IDS:
            assert (raw_dir / f"{bid}.swc.zst").exists()

        html = Path(work) / "plot-3d_HEMI_e2e_probe" / "e2e_probe.html"
        assert html.exists() and html.stat().st_size > 100_000

        # Every cached file holds the raw source (level recorded in the
        # header); the render applies mesh decimation from that source.
        for bid in PROBE_IDS:
            cached = morph._load_cached_skeleton_file(
                raw_dir / f"{bid}.swc.zst")
            assert cached._drocat_simplification == 0
            raw = neuprint_client.fetch_skeleton(bid)
            raw_n = len(navis.TreeNeuron(raw).nodes)
            assert len(cached.nodes) == raw_n

    def test_second_render_reuses_cache_without_fetching(self, built_cache,
                                                         neuprint_client,
                                                         monkeypatch):
        """REUSE: with the skeleton fetcher patched to raise, the second
        render still completes from the raw SWC cache."""
        work = built_cache
        _patch_layer_resolution(monkeypatch)

        def forbidden_fetch(*args, **kwargs):
            raise AssertionError("skeleton fetch called on a cache-hit render")

        monkeypatch.setattr(neu, "fetch_skeletons", forbidden_fetch)
        vs = _make_vs(work, neuprint_client)
        t0 = time.time()
        vs.plot_neurons()
        elapsed = time.time() - t0
        html = Path(work) / "plot-3d_HEMI_e2e_probe" / "e2e_probe.html"
        assert html.exists()
        assert elapsed < 60.0, f"cache-hit render too slow: {elapsed:.1f}s"

    def test_less_simplified_render_reuses_raw_cache(
            self, built_cache, neuprint_client, monkeypatch):
        """< 0.9 render reuses the shared raw cache."""
        work = built_cache
        cache_dir = Path(work) / "cache" / FOLDER / "skeletons"

        def forbidden_fetch(*args, **kwargs):
            raise AssertionError(
                "the less-simplified render should reuse the shared raw cache"
            )

        monkeypatch.setattr(neu, "fetch_skeletons", forbidden_fetch)
        vs = _make_vs(work, neuprint_client, simplification=0.5)
        vs.plot_neurons()

        # The cached simplified files remain the source and are not rewritten
        # as simplified skeleton pickles.
        for bid in PROBE_IDS:
            assert (cache_dir / "raw_skeletons" / f"{bid}.swc.zst").exists()
        assert not (cache_dir / ".level").exists()
        assert not list(cache_dir.glob("*.pkl"))


# ---------------------------------------------------------------------------
# Similar-neuron finding: build vector, then reuse it
# ---------------------------------------------------------------------------

# A type whose cached members disagree this much about their own shape
# (median intra-type standardized cosine below this bound) cannot be used
# for the same-type top-N assertion: for such types the pipeline may
# LEGITIMATELY rank other types above the query's own members (observed for
# PAM08_e: median pairwise cosine 0.072 across its 10 cached members).
MIN_INTRA_TYPE_COHERENCE = 0.45

# The legacy V1 raw vector cache was retired when "vector_v2" became the
# only vector method: these hemibrain tests were seeded from that retired
# production cache. Build/reuse e2e coverage for the current pipeline lives
# in tests/e2e/test_similar_bodyid_e2e.py (live male-cns run) and in
# tests/core/test_morphology_coverage.py (sandboxed fetch -> persist ->
# reuse unit flows).
V1_CACHE_RETIRED = pytest.mark.skip(
    reason="legacy V1 raw vector cache retired; vector_v2 cache build/reuse "
           "is covered by test_similar_bodyid_e2e.py and core tests")


def _type_coherence(X: np.ndarray, row_idxs) -> float:
    """Median pairwise standardized cosine among a type's cached members."""
    sims = []
    for i in range(len(row_idxs)):
        for j in range(i + 1, len(row_idxs)):
            a, b = X[row_idxs[i]], X[row_idxs[j]]
            sims.append(float(
                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
            ))
    return float(np.median(sims)) if sims else 0.0


def _pick_uncached_bid(seed=3):
    """A hemibrain bodyId NOT in the vector cache whose type is
    morphologically COHERENT and has >= 3 cached members — so the same-type
    top-N assertion is a property of the pipeline, not of type-label noise.
    The cache is never mutated by this test, so the seeded selection is
    reproducible across runs and machines."""
    vc = morph.find_similar_raw_cache(
        DATASET, project_root=str(PROJECT_ROOT), verbose=False)
    data = vc.load()
    assert data is not None and len(data["bodyIds"]) > 0
    X = np.asarray(data["X"], dtype=float)
    by_type = {}
    for row_i, (bid, t) in enumerate(zip(data["bodyIds"], data["types"])):
        if t:
            by_type.setdefault(t, []).append((int(bid), row_i))
    tmap, _ = morph._load_neuron_type_map(DATASET, str(PROJECT_ROOT))
    cached_ids = set(int(b) for b in data["bodyIds"])
    rng = np.random.default_rng(seed)
    types = [t for t, v in by_type.items() if len(v) >= 3]
    rng.shuffle(types)
    for t in types:
        coherence = _type_coherence(X, [i for _, i in by_type[t]])
        if coherence < MIN_INTRA_TYPE_COHERENCE:
            continue
        members = [int(b) for b in tmap if tmap[b] == t]
        rng.shuffle(members)
        for bid in members:
            if bid not in cached_ids:
                return bid, t
    pytest.skip(
        "requires a seeded raw vector cache with a coherent type cohort and "
        "at least one uncached member"
    )


class TestSimilarFindingCacheBuildReuse:
    @V1_CACHE_RETIRED
    def test_query_builds_vector_then_reuses_it(self, tmp_path,
                                                neuprint_client, monkeypatch):
        bid, qtype = _pick_uncached_bid()

        # ---- Sandbox: BUILD must never mutate the production caches ----
        # Vector appends are captured in memory and merged into load()
        # results (mirroring append_vectors semantics: dedupe by bodyId,
        # types refreshed from the neuron-index map, meta stats untouched);
        # the raw compressed-SWC skeleton is written under tmp_path.
        captured = []
        real_load = morph.SkeletonVectorCache.load

        def captured_append(self, records, vector_basis=morph.VECTOR_BASIS_RAW):
            captured.extend(
                (int(b), np.asarray(v, dtype=float), str(r)) for b, v, r in records
            )
            return len(records)

        def merged_load(self):
            base = real_load(self)
            if base is None or not captured:
                return base
            known = set(int(b) for b in base["bodyIds"])
            new = [c for c in captured if c[0] not in known]
            if not new:
                return base
            meta = base["meta"]
            mean = np.asarray(meta["mean"], dtype=float)
            std = np.asarray(meta["std"], dtype=float)
            std = np.where(std <= 0, 1.0, std)
            type_map, instance_map = morph._load_neuron_type_map(
                self.dataset, str(PROJECT_ROOT))
            rows = []
            for cbid, vec, rep in new:
                row = {"bodyId": cbid, "rep": rep}
                for i, name in enumerate(morph.MORPHOMETRIC_FEATURES):
                    row[name] = float(vec[i])
                for i in range(morph.PERSISTENCE_DIM):
                    row[f"pv_{i}"] = float(
                        vec[len(morph.MORPHOMETRIC_FEATURES) + i])
                row["type"] = (type_map or {}).get(cbid, "")
                row["instance"] = (instance_map or {}).get(cbid, "")
                rows.append(row)
            df_new = pd.DataFrame(rows)[list(base["df"].columns)]
            df = pd.concat([base["df"], df_new], ignore_index=True)
            raw_new = morph.SkeletonVectorCache._raw_matrix(df_new)
            # Legacy caches may lack the 'rep' column (load() uses .get()).
            if "rep" in df.columns:
                reps = df["rep"].fillna("").astype(str).tolist()
            else:
                reps = [""] * len(df)
            return {
                "meta": meta,
                "df": df,
                "raw": np.vstack([base["raw"], raw_new]),
                "X": np.vstack([base["X"], (raw_new - mean) / std]),
                "bodyIds": df["bodyId"].astype(np.int64).to_numpy(),
                "types": df.get("type", pd.Series([""] * len(df)))
                          .fillna("").astype(str).tolist(),
                "instances": df.get("instance", pd.Series([""] * len(df)))
                             .fillna("").astype(str).tolist(),
                "rep": reps,
                "dataset_rep": base["dataset_rep"],
            }

        monkeypatch.setattr(morph.SkeletonVectorCache, "append_vectors",
                            captured_append)
        monkeypatch.setattr(morph.SkeletonVectorCache, "load", merged_load)
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()

        # ---- BUILD: fetch raw -> persist shared raw + emit raw vector ----
        nrn = morph.fetch_skeleton_on_demand(DATASET, bid, level="raw",
                                             persist=True,
                                             project_root=str(sandbox))
        assert nrn is not None
        cache_file = (
            sandbox / "cache" / FOLDER / "skeletons" / "raw_skeletons"
            / f"{bid}.swc.zst"
        )
        assert cache_file.exists()
        assert morph._skeleton_folder_level(DATASET, str(sandbox)) == "raw"
        with open(cache_file, "rb") as f:
            # zstd frame magic
            assert f.read(4) == b"\x28\xb5\x2f\xfd"
        data = morph.find_similar_raw_cache(
            DATASET, project_root=str(PROJECT_ROOT), verbose=False).load()
        assert int(bid) in set(int(b) for b in data["bodyIds"]), \
            "fetch must expose the raw vector to the cache layer"
        assert ((data.get("meta") or {}).get("vector_basis")
                or "raw") == "raw"
        # The production vector cache itself stays untouched.
        prod_df = pd.read_parquet(
            morph.find_similar_raw_cache(
                DATASET, project_root=str(PROJECT_ROOT), verbose=False
            ).parquet_path
        )
        assert int(bid) not in set(int(b) for b in prod_df["bodyId"]), \
            "BUILD leaked into the production vector cache"

        # ---- REUSE: cache-only search, zero fetches, deterministic ----
        def forbidden_fetch(*args, **kwargs):
            raise AssertionError("fetch called on a vector-cache-hit search")

        monkeypatch.setattr(morph, "fetch_skeleton_on_demand", forbidden_fetch)
        out = tmp_path / "similar"
        comparer = morph.MorphologyComparer(
            query=bid, dataset=DATASET, level="bodyid", method="vector_v2",
            candidate_source="cache", output_dir=str(out),
            project_root=str(PROJECT_ROOT), verbose=False,
        )
        res1 = comparer.find_similar()
        assert not res1.empty
        same = res1[res1["is_same_type"] == True]  # noqa: E712
        assert len(same) >= 1, f"no same-type ({qtype}) neurons returned"
        # second run: identical results (pure cache reuse)
        res2 = comparer.find_similar()
        pd.testing.assert_frame_equal(res1, res2)


# ---------------------------------------------------------------------------
# Uncached-bid selection: deterministic + morphologically coherent
# ---------------------------------------------------------------------------

class TestUncachedBidSelection:
    def test_coherence_metric_synthetic(self):
        """Identical members -> 1.0; orthogonal members -> 0.0."""
        same = np.vstack([np.ones(8), np.ones(8) * 2, np.ones(8) * 0.5])
        assert abs(_type_coherence(same, [0, 1, 2]) - 1.0) < 1e-9
        ortho = np.eye(3)
        assert abs(_type_coherence(ortho, [0, 1, 2]) - 0.0) < 1e-9

    @V1_CACHE_RETIRED
    def test_selection_is_deterministic(self):
        # The test never mutates the cache, so the seeded pick is stable.
        assert _pick_uncached_bid() == _pick_uncached_bid()

    @V1_CACHE_RETIRED
    def test_selected_type_is_coherent_and_has_uncached_member(self):
        bid, qtype = _pick_uncached_bid()
        vc = morph.find_similar_raw_cache(
            DATASET, project_root=str(PROJECT_ROOT), verbose=False)
        data = vc.load()
        X = np.asarray(data["X"], dtype=float)
        rows = [i for i, t in enumerate(data["types"]) if t == qtype]
        assert len(rows) >= 3
        assert _type_coherence(X, rows) >= MIN_INTRA_TYPE_COHERENCE, \
            f"{qtype} is too incoherent for a same-type top-N assertion"
        assert int(bid) not in set(int(b) for b in data["bodyIds"])
        tmap, _ = morph._load_neuron_type_map(DATASET, str(PROJECT_ROOT))
        assert tmap.get(int(bid)) == qtype
