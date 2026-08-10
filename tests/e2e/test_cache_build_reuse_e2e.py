"""Cache BUILD and REUSE e2e — 3D visualization + similar-neuron finding.

Runs against the REAL hemibrain v1.2.1 NeuPrint server (skipped when the
network is unavailable) and verifies the post-Mission-1 cache contract end
to end:

3D skeleton visualization (``VisualizeSkeleton.plot_neurons``):
1. First render BUILDS the cache: raw skeletons are fetched online and ONLY
   the 90%-simplified versions are persisted (``skeletons/.level`` marker).
2. A second render REUSES the cache: with the skeleton fetcher patched to
   raise, the render still completes — pure cache hits, zero fetches.
3. A less-simplified render (< 0.9) IGNORES the simplified cache and
   transiently re-fetches RAW skeletons; the cached simp90 files stay
   untouched (still written/read at the fixed cache level).

Similar-neuron finding (``MorphologyComparer``):
4. A bodyId query BUILDS its vector: ``fetch_skeleton_on_demand`` fetches
   raw, persists the simplified skeleton AND appends the raw vector.
5. The search REUSES the vector: with the fetcher patched to raise, the
   query still returns same-type neurons — pure vector-cache hits, and the
   second run reproduces identical results.
"""

import pickle
import sys
import time
from pathlib import Path

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
    assert (cache_dir / ".level").read_text().strip() == "simp90"
    for bid in PROBE_IDS:
        assert (cache_dir / f"{bid}.pkl").exists()
    return work


# ---------------------------------------------------------------------------
# 3D visualization: build then reuse
# ---------------------------------------------------------------------------

class TestVisualizationCacheBuildReuse:
    def test_first_render_builds_simplified_cache_only(self, built_cache,
                                                       neuprint_client):
        """The BUILD render persists ONLY the 90%-simplified skeletons."""
        work = built_cache
        cache_dir = Path(work) / "cache" / FOLDER / "skeletons"
        assert (cache_dir / ".level").read_text().strip() == "simp90"

        html = Path(work) / "plot3d_HEMI_e2e_probe" / "e2e_probe.html"
        assert html.exists() and html.stat().st_size > 100_000

        # every cached file holds the simplified skeleton (<= ~20% of the
        # raw node count; factor-10 downsampling keeps 10-17%)
        for bid in PROBE_IDS:
            with open(cache_dir / f"{bid}.pkl", "rb") as f:
                cached = pickle.load(f)
            raw = neuprint_client.fetch_skeleton(bid)
            raw_n = len(navis.TreeNeuron(raw).nodes)
            assert len(cached.nodes) <= max(10, raw_n // 2), \
                f"cached {bid} not simplified ({len(cached.nodes)} vs raw {raw_n})"

    def test_second_render_reuses_cache_without_fetching(self, built_cache,
                                                         neuprint_client,
                                                         monkeypatch):
        """REUSE: with the skeleton fetcher patched to raise, the second
        render still completes — every neuron comes from the simp90 cache."""
        work = built_cache
        _patch_layer_resolution(monkeypatch)

        def forbidden_fetch(*args, **kwargs):
            raise AssertionError("skeleton fetch called on a cache-hit render")

        monkeypatch.setattr(neu, "fetch_skeletons", forbidden_fetch)
        vs = _make_vs(work, neuprint_client)
        t0 = time.time()
        vs.plot_neurons()
        elapsed = time.time() - t0
        html = Path(work) / "plot3d_HEMI_e2e_probe" / "e2e_probe.html"
        assert html.exists()
        assert elapsed < 60.0, f"cache-hit render too slow: {elapsed:.1f}s"

    def test_less_simplified_render_fetches_raw_and_keeps_simp90_cache(
            self, built_cache, neuprint_client, monkeypatch):
        """< 0.9 render: the simp90 cache is too coarse -> transient RAW
        fetch; the on-disk simplified cache stays untouched."""
        work = built_cache
        cache_dir = Path(work) / "cache" / FOLDER / "skeletons"
        before = {}
        for bid in PROBE_IDS:
            with open(cache_dir / f"{bid}.pkl", "rb") as f:
                before[bid] = len(pickle.load(f).nodes)

        calls = {"n": 0}
        real_fetch = neu.fetch_skeletons

        def counting_fetch(*args, **kwargs):
            calls["n"] += 1
            return real_fetch(*args, **kwargs)

        monkeypatch.setattr(neu, "fetch_skeletons", counting_fetch)
        vs = _make_vs(work, neuprint_client, simplification=0.5)
        vs.plot_neurons()
        assert calls["n"] >= 1, "less-simplified render must fetch RAW skeletons"

        # the fixed cache level is preserved (files never overwritten)
        for bid in PROBE_IDS:
            with open(cache_dir / f"{bid}.pkl", "rb") as f:
                after = len(pickle.load(f).nodes)
            assert after == before[bid], f"cache file {bid} was rewritten"


# ---------------------------------------------------------------------------
# Similar-neuron finding: build vector, then reuse it
# ---------------------------------------------------------------------------

def _pick_uncached_bid(neuprint_client, seed=3):
    """A hemibrain bodyId NOT in the vector cache whose type has >= 3 members
    in the cache (so same-type hits are assertable)."""
    import numpy as np
    vc = morph.SkeletonVectorCache(DATASET, project_root=str(PROJECT_ROOT), verbose=False)
    data = vc.load()
    assert data is not None and len(data["bodyIds"]) > 0
    by_type = {}
    for bid, t in zip(data["bodyIds"], data["types"]):
        if t:
            by_type.setdefault(t, []).append(int(bid))
    tmap, _ = morph._load_neuron_type_map(DATASET, str(PROJECT_ROOT))
    cached_ids = set(int(b) for b in data["bodyIds"])
    rng = np.random.default_rng(seed)
    types = [t for t, v in by_type.items() if len(v) >= 3]
    rng.shuffle(types)
    for t in types:
        members = [int(b) for b in tmap if tmap[b] == t]
        rng.shuffle(members)
        for bid in members:
            if bid not in cached_ids:
                return bid, t
    pytest.fail("no uncached typed bodyId found (cache too complete?)")


class TestSimilarFindingCacheBuildReuse:
    def test_query_builds_vector_then_reuses_it(self, tmp_path,
                                                neuprint_client, monkeypatch):
        bid, qtype = _pick_uncached_bid(neuprint_client)
        vc = morph.SkeletonVectorCache(DATASET, project_root=str(PROJECT_ROOT),
                                       verbose=False)

        # ---- BUILD: fetch raw -> persist simplified + append raw vector ----
        nrn = morph.fetch_skeleton_on_demand(DATASET, bid, level="raw",
                                             persist=True)
        assert nrn is not None
        cache_file = (PROJECT_ROOT / "cache" / FOLDER / "skeletons" / f"{bid}.pkl")
        assert cache_file.exists()
        assert morph._skeleton_folder_level(DATASET, str(PROJECT_ROOT)) == "simp90"
        with open(cache_file, "rb") as f:
            simp = pickle.load(f)
        assert len(simp.nodes) <= max(10, len(nrn.nodes) // 2)
        data = vc.load()
        assert int(bid) in set(int(b) for b in data["bodyIds"]), \
            "fetch must append the raw vector to the cache"
        assert ((data.get("meta") or {}).get("vector_basis")
                or "raw") == "raw"

        # ---- REUSE: cache-only search, zero fetches, deterministic ----
        def forbidden_fetch(*args, **kwargs):
            raise AssertionError("fetch called on a vector-cache-hit search")

        monkeypatch.setattr(morph, "fetch_skeleton_on_demand", forbidden_fetch)
        out = tmp_path / "similar"
        comparer = morph.MorphologyComparer(
            query=bid, dataset=DATASET, level="bodyid", method="vector",
            candidate_source="cache", top_n=30, output_dir=str(out),
            project_root=str(PROJECT_ROOT), verbose=False,
        )
        res1 = comparer.find_similar()
        assert not res1.empty
        same = res1[res1["is_same_type"] == True]  # noqa: E712
        assert len(same) >= 1, f"no same-type ({qtype}) neurons returned"
        # second run: identical results (pure cache reuse)
        res2 = comparer.find_similar()
        pd.testing.assert_frame_equal(res1, res2)
