"""Tests for the mesh-native FlyWire preparation helpers."""

import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def make_mesh(body_id=42):
    import navis
    import numpy as np

    mesh = navis.MeshNeuron({
        "vertices": np.array([
            (0.0, 0.0, 0.0),
            (100.0, 0.0, 0.0),
            (0.0, 100.0, 0.0),
            (0.0, 0.0, 100.0),
        ]),
        "faces": np.array([
            (0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3),
        ], dtype=np.int64),
    })
    mesh.id = body_id
    return mesh


def test_mesh_cache_writes_zstd_and_roundtrips(tmp_path):
    import navis
    import flywire_mesh_cache as cache

    mesh_cache = cache.FlyWireMeshCache(
        "flywire_FAFB_v783", project_root=tmp_path)
    assert mesh_cache.save({42: make_mesh()}) == 1

    compressed_path = mesh_cache.path(42)
    assert compressed_path.name == "42.pkl.zst"
    assert compressed_path.exists()
    assert not compressed_path.with_suffix("").with_suffix(".pkl").exists()
    assert mesh_cache.existing_ids() == {"42"}

    loaded = mesh_cache.load(42)
    assert isinstance(loaded, navis.MeshNeuron)
    assert str(loaded.id) == "42"


def test_mesh_cache_migrates_legacy_pickle_to_zstd(tmp_path):
    import navis
    import flywire_mesh_cache as cache

    mesh_cache = cache.FlyWireMeshCache(
        "flywire_FAFB_v783", project_root=tmp_path)
    mesh_cache.cache_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = mesh_cache.cache_dir / "42.pkl"
    with legacy_path.open("wb") as handle:
        pickle.dump(make_mesh(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = mesh_cache.load(42)

    assert isinstance(loaded, navis.MeshNeuron)
    assert mesh_cache.path(42).exists()
    assert legacy_path.exists()


def test_fine_decimator_prefilters_before_final_qem(monkeypatch):
    import trimesh
    import flywire_mesh_cache as cache

    source = trimesh.creation.icosphere(subdivisions=1, radius=10)
    prefiltered = trimesh.creation.icosphere(subdivisions=1, radius=9)
    calls = []

    monkeypatch.setattr(
        cache,
        "_vertex_cluster_prefilter",
        lambda mesh, target, prefilter_ratio=cache.FAFB_FINE_PREFILTER_RATIO:
            prefiltered,
    )
    monkeypatch.setattr(
        cache,
        "_simplify_mesh_open3d",
        lambda mesh, target: calls.append((mesh, target)) or mesh,
    )

    result = cache.simplify_mesh_fine(source, 10)

    assert result is prefiltered
    assert calls == [(prefiltered, 10)]


def test_soma_aware_default_uses_fine_decimator(monkeypatch):
    import flywire_mesh_cache as cache
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=1, radius=10)
    calls = []

    monkeypatch.setattr(
        cache,
        "simplify_mesh_fine",
        lambda mesh, target: calls.append(target) or mesh,
    )

    cache.simplify_mesh_with_soma_awareness(
        mesh,
        skeleton_simp=0.95,
        soma_simp=0.8,
        soma_pos=None,
    )

    assert calls == [max(100, int(len(mesh.faces) * 0.05))]
