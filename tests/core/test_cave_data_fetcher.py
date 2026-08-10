"""FAFB skeleton denoising (twig pruning) in cave_data_fetcher.

`fetch_skeleton`/`fetch_skeletons` prune terminal twigs shorter than the
denoise threshold (default 10 µm) after skeletonization — and also on cache
hits, so cached skeletons return denoised consistently.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cave_data_fetcher as cdf  # noqa: E402


def twiggy_neuron():
    """A long neurite (200 µm) with a short 2 µm terminal twig."""
    import navis
    pts = [(i * 1000.0, 0.0, 0.0) for i in range(201)]
    parents = [-1] + list(range(200))
    # twig: 2 extra nodes at 1 µm spacing off node 50
    pts += [(50.0 * 1000.0, 1000.0, 0.0), (50.0 * 1000.0, 2000.0, 0.0)]
    parents += [50, 201]
    n = len(pts)
    nodes = pd.DataFrame({
        "node_id": np.arange(n, dtype=np.int64),
        "parent_id": np.array(parents, dtype=np.int64),
        "x": [p[0] for p in pts],
        "y": [p[1] for p in pts],
        "z": [p[2] for p in pts],
        "radius": np.ones(n),
        "type": ["slab"] * n,
    })
    nodes.loc[0, "type"] = "root"
    nodes.loc[200, "type"] = "end"
    nodes.loc[201, "type"] = "end"
    nodes.loc[202, "type"] = "end"
    nrn = navis.TreeNeuron(nodes)
    nrn.soma = None
    nrn.units = "nm"
    return nrn


class TestDenoiseSkeleton:
    def test_prune_twigs_removes_short_twig(self):
        nrn = twiggy_neuron()
        assert len(nrn.nodes) == 203
        out = cdf.CAVEDataFetcher._denoise_skeleton(nrn, 10000.0)  # 10 µm
        # the 2 µm twig is pruned; the 200 µm neurite survives
        assert len(out.nodes) < len(nrn.nodes)
        assert len(out.nodes) <= 201
        # the original is untouched (pruning is not in-place)
        assert len(nrn.nodes) == 203

    def test_fetch_skeleton_denoises_cache_hits(self, monkeypatch):
        """A cached skeleton is returned DENOISED (consistent with fresh
        fetches that cache the denoised skeleton)."""
        fetcher = object.__new__(cdf.CAVEDataFetcher)
        fetcher.verbose = False
        monkeypatch.setattr(fetcher, "_get_skeleton_cache_path",
                            lambda bid: "/tmp/x.pkl")
        monkeypatch.setattr(fetcher, "_load_from_cache",
                            lambda path: twiggy_neuron())
        # denoise_twigs=None keeps the cached skeleton as-is
        raw = fetcher.fetch_skeleton(42, use_cache=True, denoise_twigs=None)
        assert len(raw.nodes) == 203
        # default 10 µm prunes the twig even on a cache hit
        den = fetcher.fetch_skeleton(42, use_cache=True)
        assert len(den.nodes) < len(raw.nodes)

    def test_fetch_skeleton_caches_denoised(self, monkeypatch):
        """A fresh fetch caches the DENOISED skeleton."""
        import pickle
        import navis
        fetcher = object.__new__(cdf.CAVEDataFetcher)
        fetcher.verbose = False
        saved = {}

        mesh_pkl = (PROJECT_ROOT / "cache" / "flywire_FAFB_v783" / "skeletons"
                    / "FLYWIRE_simp95_soma80_r20" / "720575940597856265.pkl")
        if not mesh_pkl.exists():
            pytest.skip("FAFB mesh cache missing")
        with open(mesh_pkl, "rb") as f:
            real_mesh = pickle.load(f)

        def fake_fetch_mesh(bid, use_cache=False):
            return navis.MeshNeuron(real_mesh.trimesh,
                                    id=bid, name=str(bid), units="nm")

        monkeypatch.setattr(fetcher, "fetch_mesh", fake_fetch_mesh)
        monkeypatch.setattr(fetcher, "_get_skeleton_cache_path",
                            lambda bid: f"/tmp/{bid}.pkl")
        monkeypatch.setattr(fetcher, "_load_from_cache", lambda path: None)

        def fake_save(neuron, path):
            saved[path] = len(neuron.nodes)

        monkeypatch.setattr(fetcher, "_save_to_cache", fake_save)
        skel = fetcher.fetch_skeleton(42, use_cache=True, denoise_twigs=10000.0)
        assert skel is not None
        assert "/tmp/42.pkl" in saved
        assert saved["/tmp/42.pkl"] == len(skel.nodes)  # cache == returned
