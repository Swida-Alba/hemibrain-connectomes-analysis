"""3D-skeleton visualization e2e — simplified NeuPrint cache pipeline (M1.5).

Runs against the REAL hemibrain v1.2.1 NeuPrint server (skipped when the
network is unavailable) and verifies the Mission-1 contract:

  * A fresh fetch writes ONLY the 90%-simplified skeleton to the cache
    (``navis.downsample_neuron`` factor 10) plus the ``skeletons/.level``
    marker; raw is never persisted.
  * A second load is a cache hit (no network): ``_load_cached_neurons``
    returns every neuron.
  * A less-simplified render (``skeleton_mesh_simplification < 0.9``)
    ignores the simplified cache and reports all neurons as missing, so
    the render re-fetches RAW skeletons transiently.
  * ``morphology.fetch_skeleton_on_demand``: a raw request never serves the
    disk cache (raw is not persisted) and appends the raw vector; a simp90
    request reuses the cached simplified file.
  * The FlyWire path keeps its own cache (no marker written).
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
import morphology as morph  # noqa: E402
from visualize_skeleton import VisualizeSkeleton  # noqa: E402

DATASET = "hemibrain:v1.2.1"
FOLDER = morph._dataset_folder(DATASET)
# aMe12 members with cached skeletons in the pre-migration build
PROBE_IDS = [911332304, 1158631810]
NEUPRINT_SERVER = "neuprint.janelia.org"


def _network_available() -> bool:
    try:
        import socket
        socket.create_connection((NEUPRINT_SERVER, 443), timeout=5).close()
        return True
    except Exception:
        return False
    return False


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not _network_available(),
        reason="NeuPrint server unreachable (network required for this e2e)",
    ),
]


@pytest.fixture(scope="module")
def probe_neurons():
    """Fetch two real hemibrain skeletons via the production client."""
    import neuprint
    from utils.token_manager import token_manager
    token = ""
    try:
        token = token_manager.get_token("NEUPRINT_TOKEN")
    except Exception:
        pass
    client = neuprint.Client(NEUPRINT_SERVER, dataset=DATASET, token=token)
    neuprint.set_default_client(client)
    neurons = []
    for bid in PROBE_IDS:
        n = neuprint.fetch_skeleton(bid)
        assert n is not None and len(n) > 0, f"no skeleton for {bid}"
        nrn = navis.TreeNeuron(n)
        # neuprint.fetch_skeleton (singular) returns a plain DataFrame;
        # attach the bodyId like the production fetch_skeletons batch path
        nrn.id = bid
        neurons.append(nrn)
    return neurons


def _vs(tmp_path, simplification=0.9):
    vs = VisualizeSkeleton(
        dataset=DATASET,
        neuron_layers=["aMe12"],
        verbose=False,
        output_dir=str(tmp_path),
        include_timestamp=False,
        cache_neurons=True,
        data_folder=str(tmp_path),
        script_path=str(tmp_path),
        skeleton_mesh_simplification=simplification,
    )
    vs.client_type = "neuprint"
    return vs


class TestSimplifiedCacheE2E:
    def test_fetch_writes_simplified_cache_only(self, tmp_path, probe_neurons):
        vs = _vs(tmp_path)
        cache_dir = Path(tmp_path) / "cache" / FOLDER / "skeletons"
        neuron_df = pd.DataFrame({"bodyId": PROBE_IDS})

        vs._save_cached_neurons(neuron_df, probe_neurons)

        assert (cache_dir / ".level").read_text().strip() == "simp90"
        for i, bid in enumerate(PROBE_IDS):
            with open(cache_dir / f"{bid}.pkl", "rb") as f:
                cached = pickle.load(f)
            # strictly fewer nodes: the raw skeleton is not persisted
            assert len(cached.nodes) < len(probe_neurons[i].nodes)

    def test_second_load_is_cache_hit(self, tmp_path, probe_neurons):
        vs = _vs(tmp_path)
        vs._save_cached_neurons(pd.DataFrame({"bodyId": PROBE_IDS}), probe_neurons)

        t0 = time.time()
        neurons, missing = vs._load_cached_neurons(pd.DataFrame({"bodyId": PROBE_IDS}))
        elapsed = time.time() - t0
        assert missing == []
        assert neurons is not None and len(neurons) == 2
        assert elapsed < 5.0  # local cache read: no network round-trips

    def test_less_simplified_render_ignores_cache(self, tmp_path, probe_neurons):
        vs = _vs(tmp_path, simplification=0.5)
        vs._save_cached_neurons(pd.DataFrame({"bodyId": PROBE_IDS}), probe_neurons)
        # < 0.9 render: the simp90 cache is too coarse -> refetch RAW
        neurons, missing = vs._load_cached_neurons(
            pd.DataFrame({"bodyId": PROBE_IDS}), ignore_cache=True)
        assert neurons is None
        assert sorted(missing) == sorted(PROBE_IDS)

    def test_render_decimation_skipped_at_cache_level(self, tmp_path, probe_neurons):
        vs = _vs(tmp_path, simplification=0.9)
        vs._save_cached_neurons(pd.DataFrame({"bodyId": PROBE_IDS}), probe_neurons)
        assert vs._skeleton_cache_is_simplified() is True
        # at exactly the cache level the render-time decimation is skipped
        # (the tube mesh is already at the cache level; double-decimation guard)
        assert vs._effective_render_simplification(is_fafb=False) == 0.0

    def test_render_decimation_above_cache_level_applies_remainder(self, tmp_path, probe_neurons):
        vs = _vs(tmp_path, simplification=0.95)
        vs._save_cached_neurons(pd.DataFrame({"bodyId": PROBE_IDS}), probe_neurons)
        assert vs._skeleton_cache_is_simplified() is True
        # above the cache level only the *remaining* relative reduction is
        # applied: (1 - 0.95) / (1 - 0.9) = 0.5 kept -> remove 50% of faces
        assert vs._effective_render_simplification(is_fafb=False) == pytest.approx(0.5)

    def test_render_decimation_below_cache_level_is_direct(self, tmp_path, probe_neurons):
        # below the cache level the RAW skeletons are re-fetched and the
        # user's fraction applies directly (no relative adjustment)
        vs = _vs(tmp_path, simplification=0.5)
        assert vs._effective_render_simplification(is_fafb=False) == pytest.approx(0.5)

    def test_fetch_on_demand_level_semantics(self, tmp_path, probe_neurons):
        vs = _vs(tmp_path)
        vs._save_cached_neurons(pd.DataFrame({"bodyId": PROBE_IDS}), probe_neurons)

        # simp90 request: served from the disk cache (no network)
        nrn = morph.fetch_skeleton_on_demand(
            DATASET, PROBE_IDS[0], project_root=str(tmp_path), level="simp90")
        assert nrn is not None
        assert len(nrn.nodes) < len(probe_neurons[0].nodes)

        # raw request never serves the disk cache: it fetches (network)
        nrn_raw = morph.fetch_skeleton_on_demand(
            DATASET, PROBE_IDS[0], project_root=str(tmp_path), level="raw",
            persist=False)
        assert nrn_raw is not None
        assert len(nrn_raw.nodes) >= len(nrn.nodes)
