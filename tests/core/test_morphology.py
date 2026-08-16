"""Tests for src/morphology.py — morphological similarity, vector cache,
NBLAST wrapper, on-demand fetch, and homolog enrichment."""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import morphology as morph  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic neuron builders
# ---------------------------------------------------------------------------

def make_neuron(points, parents, radius=1.0):
    """Build a navis TreeNeuron from (x, y, z) points and parent indices
    (-1 = root). Coordinates are in microns."""
    import navis
    n = len(points)
    nodes = pd.DataFrame({
        "node_id": np.arange(n, dtype=np.int64),
        "parent_id": np.array([p if p >= 0 else -1 for p in parents], dtype=np.int64),
        "x": [p[0] for p in points],
        "y": [p[1] for p in points],
        "z": [p[2] for p in points],
        "radius": [radius] * n,
        "type": ["0"] * n,
    })
    return navis.TreeNeuron(nodes)


def line_neuron(length=4, step=1.0):
    pts = [(i * step, 0.0, 0.0) for i in range(length)]
    return make_neuron(pts, [-1] + list(range(length - 1)))


def y_neuron():
    return make_neuron([(0, 0, 0), (1, 0, 0), (2, 1, 0), (2, -1, 0)], [-1, 0, 1, 1])


def bushy_y_neuron():
    """A Y with long arms (richer geometry for NBLAST discrimination)."""
    pts = [(0, 0, 0), (1, 0, 0)]
    for i in range(1, 6):
        pts.append((1 + i, i * 0.4, 0))
        pts.append((1 + i, -i * 0.4, 0))
    # arm points attach alternately to the stem node (1)
    parents = [-1, 0] + [1] * 10
    return make_neuron(pts, parents)


def translated(neuron, shift=(100.0, 50.0, -30.0)):
    """A translated copy of a neuron (identical morphology, different position)."""
    nodes = neuron.nodes.copy()
    for i, axis in enumerate("xyz"):
        nodes[axis] = nodes[axis] + shift[i]
    return type(neuron)(nodes)


def write_skeleton(tmp_path, dataset, body_id, neuron):
    """Persist a neuron to a tmp skeleton cache."""
    folder = tmp_path / "cache" / morph._dataset_folder(dataset) / "skeletons"
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{body_id}.pkl", "wb") as f:
        pickle.dump(neuron, f)
    return folder / f"{body_id}.pkl"


def write_neuron_index(tmp_path, dataset, rows):
    """Persist a bodyId->type/instance index to the tmp cache."""
    folder = tmp_path / "neuron_indexes" / morph._dataset_folder(dataset)
    folder.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["bodyId", "type", "instance"])
    df["bodyId"] = df["bodyId"].astype(np.int64)
    df.to_parquet(folder / "neuron_index.parquet", index=False)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class TestMorphometrics:
    def test_line_neuron_values(self):
        m = morph.compute_morphometrics(line_neuron())
        assert m["cable_length"] == pytest.approx(3.0)
        assert m["n_nodes"] == 4.0
        assert m["n_branch"] == 0.0
        assert m["n_leaf"] == 1.0
        assert m["n_root"] == 1.0
        assert m["n_primary"] == 1.0
        assert m["bbox_x"] == pytest.approx(3.0)
        assert m["bbox_diagonal"] == pytest.approx(3.0)
        assert m["tortuosity"] == pytest.approx(1.0)
        assert m["strahler_max"] == 1.0

    def test_y_branch_values(self):
        m = morph.compute_morphometrics(y_neuron())
        assert m["n_branch"] == 1.0
        assert m["n_leaf"] == 2.0
        assert m["n_root"] == 1.0
        assert m["strahler_max"] == 2.0
        # orders: branch node 2, root inherits 2, tips 1 -> [2, 2, 1, 1]
        assert m["strahler_mean"] == pytest.approx(1.5)
        # path 1+sqrt(2) over straight sqrt(5) for each tip
        expected = (1 + np.sqrt(2)) / np.sqrt(5)
        assert m["tortuosity"] == pytest.approx(expected, rel=1e-3)

    def test_all_features_finite_and_constant_dimension(self):
        for neuron in (line_neuron(), y_neuron(), translated(line_neuron())):
            m = morph.compute_morphometrics(neuron)
            assert set(m.keys()) == set(morph.MORPHOMETRIC_FEATURES)
            assert all(np.isfinite(v) for v in m.values())

    def test_translated_copy_identical_features(self):
        a = morph.compute_morphometrics(line_neuron())
        b = morph.compute_morphometrics(translated(line_neuron()))
        for k in morph.MORPHOMETRIC_FEATURES:
            assert a[k] == pytest.approx(b[k]), k

    def test_soma_radius_read(self):
        import navis
        neuron = line_neuron()
        neuron.nodes.loc[0, "radius"] = 5.0
        neuron.soma = np.array([0])
        m = morph.compute_morphometrics(neuron)
        assert m["soma_radius"] == 5.0


class TestPersistenceVector:
    def test_shape_and_stability(self):
        pv = morph.compute_persistence_vector(line_neuron())
        assert pv.shape == (100,)
        assert np.allclose(pv, morph.compute_persistence_vector(line_neuron()))

    def test_distinct_geometries(self):
        pv_line = morph.compute_persistence_vector(line_neuron())
        pv_y = morph.compute_persistence_vector(y_neuron())
        assert not np.allclose(pv_line, pv_y)

    def test_translated_copy_same_vector(self):
        a = morph.compute_persistence_vector(line_neuron())
        b = morph.compute_persistence_vector(translated(line_neuron()))
        assert np.allclose(a, b)

    def test_full_vector_dimension(self):
        _, vec = morph.vectorize_neuron(line_neuron())
        assert vec.shape == (morph.VECTOR_DIM,)


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_cosine_self_is_one(self):
        v = np.random.default_rng(0).normal(size=50)
        assert morph.cosine_similarity_matrix(v, v.reshape(1, -1))[0] == pytest.approx(1.0)

    def test_pearson_self_is_one(self):
        v = np.random.default_rng(1).normal(size=50)
        assert morph.pearson_similarity_matrix(v, v.reshape(1, -1))[0] == pytest.approx(1.0)

    def test_rank_corresponds_to_distance(self):
        rng = np.random.default_rng(2)
        q = rng.normal(size=30)
        near = q + rng.normal(scale=0.1, size=30)
        far = rng.normal(size=30)
        cos = morph.cosine_similarity_matrix(q, np.vstack([far, near]))
        assert cos[1] > cos[0]


# ---------------------------------------------------------------------------
# Vector cache
# ---------------------------------------------------------------------------

class TestSkeletonVectorCache:
    def _setup(self, tmp_path, dataset="test:v1", n_line=2, n_y=1):
        write_skeleton(tmp_path, dataset, 101, line_neuron())
        write_skeleton(tmp_path, dataset, 102, line_neuron(length=6))
        write_skeleton(tmp_path, dataset, 103, y_neuron())
        write_neuron_index(tmp_path, dataset, [
            (101, "LINE", "LINE_1"), (102, "LINE", "LINE_2"), (103, "Y", "Y_1"),
        ])
        return morph.SkeletonVectorCache(dataset, project_root=str(tmp_path), n_workers=2, verbose=False)

    def test_build_load_roundtrip(self, tmp_path):
        cache = self._setup(tmp_path)
        stats = cache.build()
        assert stats["rows"] == 3 and stats["new"] == 3 and stats["fetched"] == 0
        assert cache.parquet_path.exists() and cache.meta_path.exists()

        data = cache.load()
        assert len(data["bodyIds"]) == 3
        assert data["X"].shape == (3, morph.VECTOR_DIM)
        assert data["types"] == ["LINE", "LINE", "Y"]
        assert data["instances"] == ["LINE_1", "LINE_2", "Y_1"]

        # z-score standardization across the population (constant features
        # keep std 0; all others are standardized to unit variance)
        assert np.allclose(data["X"].mean(axis=0), 0.0, atol=1e-9)
        stds = data["X"].std(axis=0)
        assert np.allclose(stds[stds > 0], 1.0, atol=1e-9)

    def test_incremental_rebuild_skips_existing(self, tmp_path):
        cache = self._setup(tmp_path)
        cache.build()
        stats2 = cache.build()
        assert stats2["new"] == 0 and stats2["rows"] == 3
        # adding a skeleton only vectorizes the new one
        write_skeleton(tmp_path, "test:v1", 104, line_neuron(length=3))
        stats3 = cache.build()
        assert stats3["new"] == 1 and stats3["rows"] == 4

    def test_ensure_autobuilds(self, tmp_path):
        cache = self._setup(tmp_path)
        assert cache.cache_exists() is False
        cache.ensure()
        assert cache.cache_exists() is True

    def test_coverage(self, tmp_path):
        cache = self._setup(tmp_path)
        cov = cache.coverage()
        assert cov["skeletons"] == 3 and cov["vectors"] == 0
        cache.build()
        assert cache.coverage()["vectors"] == 3

    def test_vectors_for_missing_computes_from_skeleton(self, tmp_path):
        cache = self._setup(tmp_path)
        cache.build()
        X, mask, _ = cache.vectors_for([101, 103, 999])
        assert mask.tolist() == [True, True, False]
        assert np.isnan(X[2]).all()
        assert np.isfinite(X[0]).all()

    def test_vectors_for_without_built_cache(self, tmp_path):
        cache = self._setup(tmp_path)
        # no parquet yet: still resolves from existing skeleton files
        X, mask, _ = cache.vectors_for([101, 102])
        assert mask.all() and np.isfinite(X).all()

    def test_fetch_missing_extends_cache(self, tmp_path, monkeypatch):
        cache = self._setup(tmp_path)
        # neuron 999 exists in the index but has no skeleton yet
        write_neuron_index(tmp_path, "test:v1", [
            (101, "LINE", "LINE_1"), (102, "LINE", "LINE_2"),
            (103, "Y", "Y_1"), (999, "NEW", "NEW_1"),
        ])
        fetched = {}

        def fake_fetch(dataset, body_id, project_root=None):
            fetched[body_id] = True
            neuron = line_neuron(length=5)
            folder = (Path(project_root) / "cache"
                      / morph._dataset_folder(dataset) / "skeletons")
            folder.mkdir(parents=True, exist_ok=True)
            with open(folder / f"{body_id}.pkl", "wb") as f:
                pickle.dump(neuron, f)
            return neuron

        monkeypatch.setattr(morph, "fetch_skeleton_on_demand", fake_fetch)
        stats = cache.build(fetch_missing=2)
        assert fetched == {999: True}
        assert stats["fetched"] == 1 and stats["rows"] == 4


# ---------------------------------------------------------------------------
# On-demand fetch
# ---------------------------------------------------------------------------

class TestFetchOnDemand:
    def test_neuprint_fetch_persists_simplified_and_reuses_at_simp90(self, tmp_path, monkeypatch):
        """persist=True writes the SIMPLIFIED skeleton (raw never persisted);
        a raw request never hits the disk cache, a simp90 request does."""
        neuron = line_neuron()
        monkeypatch.setattr(morph, "_fetch_neuprint_skeleton",
                            lambda dataset, bid: neuron)
        pkl = write_skeleton(tmp_path, "test:v1", 101, None)  # placeholder path
        pkl.unlink()  # ensure missing

        nrn_raw = morph.fetch_skeleton_on_demand(
            "test:v1", 101, project_root=str(tmp_path), level="raw"
        )
        assert nrn_raw is not None
        assert pkl.exists()  # simplified cache written
        # the persisted file must hold the simplified skeleton (fewer nodes)
        import pickle as _p
        with open(pkl, "rb") as f:
            cached = _p.load(f)
        assert len(cached.nodes) <= len(neuron.nodes)
        # the .level marker records the simplified cache level
        marker = pkl.parent / ".level"
        assert marker.read_text().strip() == "simp90"

        # A RAW request never hits the cache (raw is not persisted).
        calls = {"n": 0}

        def counting_fetch(dataset, bid):
            calls["n"] += 1
            return neuron

        monkeypatch.setattr(morph, "_fetch_neuprint_skeleton", counting_fetch)
        again_raw = morph.fetch_skeleton_on_demand(
            "test:v1", 101, project_root=str(tmp_path), level="raw"
        )
        assert calls["n"] == 1
        assert again_raw is not None

        # A SIMP90 request reuses the persisted file without the fetcher.
        again_simp = morph.fetch_skeleton_on_demand(
            "test:v1", 101, project_root=str(tmp_path), level="simp90"
        )
        assert calls["n"] == 1
        assert again_simp is not None
        assert len(again_simp.nodes) <= len(neuron.nodes)

    def test_persist_false_does_not_write_cache(self, tmp_path, monkeypatch):
        """Transient fetches (persist=False) must not create skeleton files."""
        monkeypatch.setattr(morph, "_fetch_neuprint_skeleton",
                            lambda d, b: line_neuron())
        skel_dir = tmp_path / "cache" / morph._dataset_folder("np:v1") / "skeletons"
        skel_dir.mkdir(parents=True)
        neuron = morph.fetch_skeleton_on_demand(
            "np:v1", 42, project_root=str(tmp_path), persist=False
        )
        assert neuron is not None
        assert not list(skel_dir.glob("*.pkl"))
        # persist=True (default) writes the file for reuse
        neuron2 = morph.fetch_skeleton_on_demand(
            "np:v1", 43, project_root=str(tmp_path)
        )
        assert neuron2 is not None
        assert (skel_dir / "43.pkl").exists()

    def test_cave_fetch_used_for_flywire(self, monkeypatch, tmp_path):
        used = {}

        def fake_cave(dataset, bid):
            used["dataset"] = dataset
            return y_neuron()

        monkeypatch.setattr(morph, "_fetch_cave_skeleton", fake_cave)
        monkeypatch.setattr(morph, "_fetch_neuprint_skeleton",
                            lambda d, b: (_ for _ in ()).throw(AssertionError("neuprint used")))
        nrn = morph.fetch_skeleton_on_demand(
            "flywire_FAFB_v783", 42, project_root=str(tmp_path)
        )
        assert used["dataset"] == "flywire_FAFB_v783"
        assert nrn is not None


# ---------------------------------------------------------------------------
# MorphologyComparer
# ---------------------------------------------------------------------------

class TestMorphologyComparer:
    def _setup(self, tmp_path, monkeypatch, dataset="test:v1"):
        write_skeleton(tmp_path, dataset, 101, line_neuron(length=20))
        write_skeleton(tmp_path, dataset, 102, translated(line_neuron(length=20)))
        write_skeleton(tmp_path, dataset, 103, bushy_y_neuron())
        write_neuron_index(tmp_path, dataset, [
            (101, "LINE", "LINE_1"), (102, "LINE", "LINE_2"), (103, "Y", "Y_1"),
        ])

        def fake_getNeurons(required, dataset_name, **kwargs):
            rows = {
                101: ("LINE", "LINE_1"),
                102: ("LINE", "LINE_2"),
                103: ("Y", "Y_1"),
            }
            if isinstance(required, (list, tuple)):
                b = [int(r) for r in required]
            else:
                b = [int(required)]
            df = pd.DataFrame({
                "bodyId": b,
                "type": [rows[x][0] for x in b],
                "instance": [rows[x][1] for x in b],
            })
            # real getNeurons returns (neuron_df, roi_count_df, auto_name, criteria)
            return df, pd.DataFrame(), "auto", None

        monkeypatch.setattr(morph, "getNeurons", fake_getNeurons)
        return tmp_path

    def _make_comparer(self, root, **kwargs):
        params = dict(
            dataset="test:v1", level="bodyid", method="vector", metric="cosine",
            top_n=5, output_dir=str(root / "out"),
            project_root=str(root), verbose=False, candidate_source="cache",
        )
        params.update(kwargs)
        return morph.MorphologyComparer(**params)

    def test_vector_search_ranks_translated_copy_first(self, tmp_path, monkeypatch):
        root = self._setup(tmp_path, monkeypatch)
        comparer = self._make_comparer(root, query=101)
        res = comparer.find_similar()
        assert not res.empty
        assert res.iloc[0]["target_bodyId"] == 102  # translated copy = identical morphology
        assert 101 not in res["target_bodyId"].tolist()  # self excluded
        assert res.iloc[0]["similarity"] >= res.iloc[1]["similarity"]

    def test_vector_search_type_level_includes_intra_reference(self, tmp_path, monkeypatch):
        """Type queries must include the query type itself as the intra-type
        reference row (rank 1, is_intra_type=True) plus its intra-type
        similarity value."""
        root = self._setup(tmp_path, monkeypatch)
        comparer = self._make_comparer(root, query=101, level="type")
        res = comparer.find_similar()
        assert not res.empty
        # The query type (LINE) is present as the intra-type reference row.
        intra = res[res["is_intra_type"]]
        assert len(intra) == 1
        assert intra.iloc[0]["target_type"] == "LINE"
        assert intra.iloc[0]["rank"] == 1
        # 2 identical-morphology LINE members -> intra-type similarity ~1.
        assert intra.iloc[0]["intra_type_similarity"] == pytest.approx(1.0, abs=1e-6)
        # Inter-type rows exclude the query type and rank after it.
        inter = res[~res["is_intra_type"]]
        assert inter.iloc[0]["target_type"] == "Y"
        assert inter.iloc[0]["rank"] == 2
        assert "LINE" not in inter["target_type"].tolist()

    def test_vector_search_bodyid_carries_intra_columns(self, tmp_path, monkeypatch):
        """bodyId queries must tag same-type rows (is_same_type) and report
        the query type's intra-type similarity."""
        root = self._setup(tmp_path, monkeypatch)
        comparer = self._make_comparer(root, query=101)
        res = comparer.find_similar()
        assert {"is_same_type", "intra_type_similarity"}.issubset(res.columns)
        # 102 is the same type (LINE) and morphologically identical -> top hit
        assert res.iloc[0]["target_bodyId"] == 102
        assert res.iloc[0]["is_same_type"] == True
        # LINE has 2 identical members -> intra-type similarity ~1
        assert res.iloc[0]["intra_type_similarity"] == pytest.approx(1.0, abs=1e-6)
        # the Y neuron is a different type
        row_y = res[res["target_bodyId"] == 103]
        assert row_y.iloc[0]["is_same_type"] == False

    def test_intra_type_similarity_values(self, tmp_path, monkeypatch):
        root = self._setup(tmp_path, monkeypatch)
        cache = morph.SkeletonVectorCache("test:v1", project_root=str(root), verbose=False)
        cache.build()
        data = cache.load()
        comparer = self._make_comparer(root, query=101)
        # two identical LINE members -> 1.0; single-member type -> 1.0;
        # unknown/empty types -> NaN
        assert comparer._intra_type_similarity(
            "LINE", data["bodyIds"], data["types"], data["X"], "cosine"
        ) == pytest.approx(1.0, abs=1e-6)
        assert comparer._intra_type_similarity(
            "Y", data["bodyIds"], data["types"], data["X"], "cosine"
        ) == 1.0
        assert np.isnan(comparer._intra_type_similarity(
            "ZZZ", data["bodyIds"], data["types"], data["X"], "cosine"))
        assert np.isnan(comparer._intra_type_similarity(
            "", data["bodyIds"], data["types"], data["X"], "cosine"))

    def test_nblast_type_level_includes_intra_reference(self, tmp_path, monkeypatch):
        root = self._setup(tmp_path, monkeypatch)
        comparer = self._make_comparer(
            root, query=101, level="type", method="nblast",
            nblast_prefilter=10, n_workers=1,
        )
        res = comparer.find_similar()
        assert not res.empty
        intra = res[res["is_intra_type"]]
        assert len(intra) == 1
        assert intra.iloc[0]["target_type"] == "LINE"
        assert intra.iloc[0]["rank"] == 1
        # identical members -> both the NBLAST mean and the vector intra ~1
        assert intra.iloc[0]["similarity"] > 0.9
        assert intra.iloc[0]["intra_type_similarity"] == pytest.approx(1.0, abs=1e-6)
        # inter-type rows rank after the intra reference
        inter = res[~res["is_intra_type"]]
        assert inter.iloc[0]["target_type"] == "Y"
        assert inter.iloc[0]["rank"] == 2

    def test_nblast_method_and_outputs(self, tmp_path, monkeypatch):
        root = self._setup(tmp_path, monkeypatch)
        comparer = self._make_comparer(
            root, query=101, method="nblast", nblast_prefilter=10, n_workers=1,
        )
        res = comparer.find_similar()
        assert not res.empty
        assert res.iloc[0]["target_bodyId"] == 102  # translated copy scores highest
        assert res.iloc[0]["similarity"] >= res.iloc[1]["similarity"]
        # dotprops must never be persisted to disk
        assert not list((root / "cache" / "test_v1").rglob("*.dp")) 
        # results saved with the similar-morphology_ prefix
        run_dirs = [p for p in (root / "out").iterdir() if p.is_dir()]
        assert len(run_dirs) == 1
        assert run_dirs[0].name.startswith("similar-morphology_")
        assert (run_dirs[0] / "results.csv").exists()
        assert (run_dirs[0] / "README.txt").exists()

    def test_invalid_parameters(self, tmp_path):
        with pytest.raises(ValueError):
            morph.MorphologyComparer(level="bad", project_root=str(tmp_path), verbose=False)
        with pytest.raises(ValueError):
            morph.MorphologyComparer(method="bad", project_root=str(tmp_path), verbose=False)
        with pytest.raises(ValueError):
            morph.MorphologyComparer(metric="bad", project_root=str(tmp_path), verbose=False)
        with pytest.raises(ValueError, match="visualize_by"):
            morph.MorphologyComparer(query=1, visualize_by="bad",
                                     project_root=str(tmp_path), verbose=False)


    def test_auto_level_resolution(self, tmp_path, monkeypatch):
        """level='auto': type queries -> type-to-type, bodyId queries ->
        bodyId-to-bodyId, mixed/multi-type lists -> bodyId."""
        root = self._setup(tmp_path, monkeypatch)
        comparer = self._make_comparer(root, query=101, level="auto")

        comparer.query = "LINE"
        qdf = pd.DataFrame({"bodyId": [101, 102], "type": ["LINE", "LINE"],
                            "instance": ["L1", "L2"]})
        assert comparer._resolve_level(qdf) == "type"

        comparer.query = 101
        qdf1 = pd.DataFrame({"bodyId": [101], "type": ["LINE"], "instance": ["L1"]})
        assert comparer._resolve_level(qdf1) == "bodyid"

        comparer.query = [101, 102]
        assert comparer._resolve_level(qdf) == "bodyid"  # all-numeric list

        comparer.query = "SMP"  # single-member type still a type query
        qdf2 = pd.DataFrame({"bodyId": [999], "type": ["SMP"], "instance": ["S"]})
        assert comparer._resolve_level(qdf2) == "type"

        comparer.query = ["LINE", "Y"]  # multi-type list -> bodyId rows
        qdf3 = pd.DataFrame({"bodyId": [101, 103], "type": ["LINE", "Y"],
                             "instance": ["L1", "Y1"]})
        assert comparer._resolve_level(qdf3) == "bodyid"

    def test_auto_level_end_to_end(self, tmp_path, monkeypatch):
        root = self._setup(tmp_path, monkeypatch)

        def fake_getNeurons(required, dataset_name, **kwargs):
            rows = {101: ("LINE", "L1"), 102: ("LINE", "L2"), 103: ("Y", "Y1")}
            if isinstance(required, str) and not required.isdigit():
                b = [bid for bid, (t, _) in rows.items() if t == required]
            else:
                b = [int(r) for r in (required if isinstance(required, (list, tuple))
                                      else [required])]
            df = pd.DataFrame({"bodyId": b, "type": [rows[x][0] for x in b],
                               "instance": [rows[x][1] for x in b]})
            return df, pd.DataFrame(), "auto", None

        monkeypatch.setattr(morph, "getNeurons", fake_getNeurons)
        # type query -> type-to-type rows with the intra reference
        comparer = self._make_comparer(root, query="LINE", level="auto")
        res = comparer.find_similar()
        assert "is_intra_type" in res.columns
        assert res.iloc[0]["target_type"] == "LINE"
        # bodyId query -> bodyId-to-bodyId rows
        comparer2 = self._make_comparer(root, query=101, level="auto")
        res2 = comparer2.find_similar()
        assert "target_bodyId" in res2.columns
        assert "is_same_type" in res2.columns

    def test_type_summary_written_for_bodyid_level(self, tmp_path, monkeypatch):
        root = self._setup(tmp_path, monkeypatch)
        comparer = self._make_comparer(root, query=101, level="bodyid")
        comparer.find_similar()
        run_dir = Path(comparer.output_folder)
        # results.csv is bodyId-level; type_summary.csv the type rows
        res = pd.read_csv(run_dir / "results.csv")
        assert "target_bodyId" in res.columns
        summary = pd.read_csv(run_dir / "type_summary.csv")
        assert {"target_type", "similarity", "n_bodyids", "is_intra_type",
                "intra_type_similarity"}.issubset(summary.columns)
        # the query type (LINE, both members) is the intra reference first
        assert summary.iloc[0]["target_type"] == "LINE"
        assert summary.iloc[0]["is_intra_type"] == True  # noqa: E712
        assert summary.iloc[0]["n_bodyids"] == 2
        assert summary.iloc[1]["target_type"] == "Y"
        assert summary.iloc[1]["is_intra_type"] == False  # noqa: E712

    def test_type_summary_written_for_type_level(self, tmp_path, monkeypatch):
        root = self._setup(tmp_path, monkeypatch)
        comparer = self._make_comparer(root, query=101, level="type")
        comparer.find_similar()
        run_dir = Path(comparer.output_folder)
        summary = pd.read_csv(run_dir / "type_summary.csv")
        assert {"target_type", "similarity", "n_bodyids", "is_intra_type"}.issubset(summary.columns)
        assert summary.iloc[0]["target_type"] == "LINE"
        assert summary.iloc[0]["is_intra_type"] == True  # noqa: E712

    def test_type_search_writes_bodyid_results_csv(self, tmp_path, monkeypatch):
        """results.csv is ALWAYS bodyId-level: a type query's results.csv
        holds ranked bodyId rows, including ordered intra-type pairs when
        multiple query members are resolved, while type_summary.csv holds
        the type rows."""
        root = self._setup(tmp_path, monkeypatch)
        comparer = self._make_comparer(root, query=101, level="type")
        comparer.find_similar()
        run_dir = Path(comparer.output_folder)
        res = pd.read_csv(run_dir / "results.csv")
        assert "target_bodyId" in res.columns
        # This scalar fixture resolves one member, so there is no intra pair.
        assert 101 not in res["target_bodyId"].tolist()
        assert res["target_type"].tolist() == ["LINE", "Y"]
        summary = pd.read_csv(run_dir / "type_summary.csv")
        assert "target_type" in summary.columns
        assert summary.iloc[0]["is_intra_type"] == True  # noqa: E712
        assert summary.iloc[0]["target_type"] == "LINE"


class TestVisualizeTopResults:
    """3D skeleton visualization of the top-N found types (NB-style)."""

    class FakeVisualizer:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            # nested class: its own name is not in method scope
            type(self).instances.append(self)

        def plot_neurons(self):
            pass

    def _setup(self, tmp_path, monkeypatch, **kwargs):
        root = TestMorphologyComparer()._setup(tmp_path, monkeypatch)
        params = dict(query=101, dataset="test:v1", top_n=5,
                      output_dir=str(root / "out"), project_root=str(root),
                      verbose=False, candidate_source="cache",
                      visualize_top_n=3, visualize_by="type")
        params.update(kwargs)
        self.FakeVisualizer.instances = []
        monkeypatch.setattr(morph, "_import_visualizer",
                            lambda: self.FakeVisualizer)
        return morph.MorphologyComparer(**params)

    def test_disabled_by_default(self, tmp_path, monkeypatch):
        comparer = self._setup(tmp_path, monkeypatch, visualize_top_n=0)
        res = comparer.find_similar()
        assert not res.empty
        assert self.FakeVisualizer.instances == []

    def test_type_mode_groups_result_members(self, tmp_path, monkeypatch):
        comparer = self._setup(tmp_path, monkeypatch)
        res = comparer.find_similar()
        assert not res.empty
        assert len(self.FakeVisualizer.instances) == 1
        vs = self.FakeVisualizer.instances[0]
        # The query is rendered as the first (reference) layer; the requested
        # top-N count applies only to result types.
        assert vs.kwargs["neuron_layers"] == [[101], [102], [103]]
        assert vs.kwargs["custom_layer_names"] == [
            "query_101_x1", "r1_LINE_x1", "r2_Y_x1"
        ]
        assert vs.kwargs["legend_mode"] == "layer"
        assert vs.kwargs["skip_synapse"] is True
        assert vs.kwargs["show_fig"] is False
        assert vs.kwargs["saveas"] == morph._dataset_folder("test:v1")
        assert vs.kwargs["skeleton_mesh_simplification"] == 0.95

    def test_type_level_excludes_intra_reference_row(self, tmp_path, monkeypatch):
        """The intra-type reference row (rank 1) must never be rendered."""
        comparer = self._setup(tmp_path, monkeypatch, query=101, level="type")
        res = comparer.find_similar()
        assert not res.empty
        assert len(self.FakeVisualizer.instances) == 1
        vs = self.FakeVisualizer.instances[0]
        # The query is the first layer; the LINE intra reference is excluded,
        # so only the Y row remains, with its members from the vector cache.
        assert vs.kwargs["neuron_layers"] == [[101], [103]]
        assert vs.kwargs["custom_layer_names"] == ["query_101_x1", "r1_Y_x1"]

    def test_bodyid_mode_one_layer_per_row(self, tmp_path, monkeypatch):
        comparer = self._setup(tmp_path, monkeypatch, visualize_by="bodyId")
        comparer.find_similar()
        assert len(self.FakeVisualizer.instances) == 1
        vs = self.FakeVisualizer.instances[0]
        assert vs.kwargs["neuron_layers"] == [[101], [102], [103]]
        assert vs.kwargs["custom_layer_names"] == [
            "query_101_x1", "r1_LINE_102", "r2_Y_103"
        ]
        assert vs.kwargs["legend_mode"] == "single"

    def test_query_rows_do_not_consume_bodyid_top_n(self, tmp_path, monkeypatch):
        comparer = self._setup(
            tmp_path, monkeypatch, visualize_by="bodyId", visualize_top_n=1
        )
        results = pd.DataFrame({
            "target_bodyId": [101, 102, 103],
            "target_type": ["LINE", "LINE", "Y"],
            "similarity": [1.0, 0.9, 0.8],
        })
        query_df = pd.DataFrame({"bodyId": [101], "type": ["LINE"]})

        comparer._visualize_top_results(results, query_df=query_df)

        vs = self.FakeVisualizer.instances[0]
        # The query reference layer never consumes the top-N budget: the
        # query row (101) is excluded from the results, leaving one layer.
        assert vs.kwargs["neuron_layers"] == [[101], [102]]
        assert vs.kwargs["custom_layer_names"] == ["query_101_x1", "r1_LINE_102"]

    def test_visualization_settings_are_forwarded(self, tmp_path, monkeypatch):
        comparer = self._setup(
            tmp_path,
            monkeypatch,
            visualization_settings={
                "brain_mesh": "none",
                "skeleton_mode": "line",
                "background_color": "black",
                "show_fig": True,
                "export_views": True,
            },
        )
        comparer.find_similar()
        vs = self.FakeVisualizer.instances[0]
        assert vs.kwargs["brain_mesh"] == "none"
        assert vs.kwargs["skeleton_mode"] == "line"
        assert vs.kwargs["background_color"] == "black"
        assert vs.kwargs["show_fig"] is True
        assert vs.kwargs["export_views"] is True

    def test_failure_never_breaks_search(self, tmp_path, monkeypatch):
        def boom(self):
            raise RuntimeError("render failed")
        self.FakeVisualizer.plot_neurons = boom
        comparer = self._setup(tmp_path, monkeypatch)
        res = comparer.find_similar()  # must not raise
        assert not res.empty

    def test_visualizer_unavailable_skips(self, tmp_path, monkeypatch):
        comparer = self._setup(tmp_path, monkeypatch)
        monkeypatch.setattr(morph, "_import_visualizer", lambda: None)
        res = comparer.find_similar()
        assert not res.empty
        assert self.FakeVisualizer.instances == []


# ---------------------------------------------------------------------------
# Mesh-based features (FlyWire bulk caches)
# ---------------------------------------------------------------------------

def cube_mesh(size=1.0):
    """A unit cube as a navis MeshNeuron (12 triangles)."""
    import navis
    v = np.array([
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
        (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
    ], dtype=float) * size
    faces = np.array([
        [0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6],
        [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7],
        [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5],
    ], dtype=np.int64)
    return navis.MeshNeuron({"vertices": v, "faces": faces})


def translated_mesh(mesh, shift=(100.0, 50.0, -30.0)):
    vertices = np.asarray(mesh.vertices, dtype=float) + np.array(shift)
    return type(mesh)({"vertices": vertices, "faces": np.asarray(mesh.faces)})


class TestMeshFeatures:
    def test_cube_morphometrics(self):
        m = morph.compute_mesh_morphometrics(cube_mesh())
        assert m["n_nodes"] == 8.0
        assert m["n_branch"] == 12.0
        assert m["bbox_x"] == pytest.approx(1.0)
        assert m["bbox_diagonal"] == pytest.approx(np.sqrt(3))
        assert m["leaf_density"] == pytest.approx(6.0 / 3.0)  # area / bbox_diag^2
        assert m["cable_length"] > 0

    def test_histogram_shape_and_normalization(self):
        hist = morph.compute_spatial_histogram(cube_mesh())
        assert hist.shape == (100,)
        assert hist.sum() == pytest.approx(1.0)

    def test_mesh_vector_schema_and_translation_invariance(self):
        v1 = morph.vectorize_neuron(cube_mesh())[1]
        v2 = morph.vectorize_neuron(translated_mesh(cube_mesh()))[1]
        assert v1.shape == (morph.VECTOR_DIM,)
        assert np.isfinite(v1).all()
        assert np.allclose(v1, v2)  # bbox-normalized histogram is translation-invariant

    def test_mesh_and_skeleton_share_schema(self):
        v_skel = morph.vectorize_neuron(line_neuron())[1]
        v_mesh = morph.vectorize_neuron(cube_mesh())[1]
        assert v_skel.shape == v_mesh.shape == (morph.VECTOR_DIM,)

    def test_unsupported_neuron_raises(self):
        class Bad:
            pass
        with pytest.raises(ValueError, match="Unsupported neuron type"):
            morph.vectorize_neuron(Bad())


# ---------------------------------------------------------------------------
# Nested skeleton folders (FlyWire bulk caches)
# ---------------------------------------------------------------------------

class TestNestedDiscovery:
    def test_build_and_coverage_find_nested_files(self, tmp_path):
        dataset = "fw:v1"
        folder = (tmp_path / "cache" / morph._dataset_folder(dataset)
                  / "skeletons" / "bulk_v1")
        folder.mkdir(parents=True)
        for bid, nrn in ((301, line_neuron()), (302, y_neuron())):
            with open(folder / f"{bid}.pkl", "wb") as f:
                pickle.dump(nrn, f)
        cache = morph.SkeletonVectorCache(dataset, project_root=str(tmp_path), verbose=False)
        assert cache.coverage()["skeletons"] == 2
        stats = cache.build()
        assert stats["rows"] == 2
        X, mask, _ = cache.vectors_for([301, 302])
        assert mask.all() and np.isfinite(X).all()

    def test_vectors_for_resolves_nested(self, tmp_path):
        dataset = "fw:v1"
        folder = (tmp_path / "cache" / morph._dataset_folder(dataset)
                  / "skeletons" / "bulk_v1")
        folder.mkdir(parents=True)
        with open(folder / "301.pkl", "wb") as f:
            pickle.dump(line_neuron(), f)
        cache = morph.SkeletonVectorCache(dataset, project_root=str(tmp_path), verbose=False)
        X, mask, _ = cache.vectors_for([301, 999])
        assert mask.tolist() == [True, False]

    def test_fetch_reuses_nested_file(self, tmp_path, monkeypatch):
        """A simp90 request reuses a nested (bulk-folder) cache file."""
        dataset = "fw:v1"
        folder = (tmp_path / "cache" / morph._dataset_folder(dataset)
                  / "skeletons" / "bulk_v1")
        folder.mkdir(parents=True)
        with open(folder / "301.pkl", "wb") as f:
            pickle.dump(line_neuron(), f)
        monkeypatch.setattr(morph, "_fetch_neuprint_skeleton",
                            lambda d, b: (_ for _ in ()).throw(AssertionError("fetcher used")))
        nrn = morph.fetch_skeleton_on_demand(dataset, 301, project_root=str(tmp_path),
                                             level="simp90")
        assert nrn is not None


# ---------------------------------------------------------------------------
# Vectorization-level guards (vector_basis / .level marker)
# ---------------------------------------------------------------------------

class TestLevelGuards:
    def test_folder_level_defaults_to_raw(self, tmp_path):
        dataset = "test:v1"
        write_skeleton(tmp_path, dataset, 101, line_neuron())
        assert morph._skeleton_folder_level(dataset, str(tmp_path)) == "raw"

    def test_folder_level_reads_marker(self, tmp_path):
        dataset = "test:v1"
        folder = tmp_path / "cache" / morph._dataset_folder(dataset) / "skeletons"
        folder.mkdir(parents=True)
        (folder / ".level").write_text("simp90\n")
        assert morph._skeleton_folder_level(dataset, str(tmp_path)) == "simp90"
        # unknown values are treated as raw (never guessed)
        (folder / ".level").write_text("weird\n")
        assert morph._skeleton_folder_level(dataset, str(tmp_path)) == "raw"

    def test_marker_is_idempotent(self, tmp_path):
        dataset = "test:v1"
        morph._write_skeleton_level_marker(dataset, str(tmp_path))
        morph._write_skeleton_level_marker(dataset, str(tmp_path))
        marker = (tmp_path / "cache" / morph._dataset_folder(dataset)
                  / "skeletons" / ".level")
        assert marker.read_text().strip() == "simp90"

    def test_downsample_for_cache_reduces_nodes(self):
        import navis
        pts = [(i * 1.0, 0.0, 0.0) for i in range(200)]
        nrn = make_neuron(pts, [-1] + list(range(199)))
        out = morph._downsample_for_cache(nrn)
        assert isinstance(out, navis.TreeNeuron)
        assert len(out.nodes) <= len(nrn.nodes)
        # deterministic
        out2 = morph._downsample_for_cache(nrn)
        assert len(out.nodes) == len(out2.nodes)

    def test_downsample_ignores_bogus_multi_node_soma(self):
        """navis' radius>=1 soma detection flags every neuprint node (nm
        radii); a whole-neuron "soma" must not freeze the cache level at
        full resolution."""
        pts = [(i * 1.0, 0.0, 0.0) for i in range(200)]
        nrn = make_neuron(pts, [-1] + list(range(199)), radius=10.0)
        # reproduce the bogus detection: every node qualifies as soma
        assert nrn.soma is not None and len(nrn.soma) > 1
        out = morph._downsample_for_cache(nrn)
        # strictly reduced: the bogus soma is treated as no soma
        assert len(out.nodes) < len(nrn.nodes)
        # the original neuron is untouched (caller keeps vectorizing raw)
        assert nrn.soma is not None and len(nrn.soma) > 1

    def test_vectors_for_skips_simp90_files_when_basis_raw(self, tmp_path):
        """On-disk simplified files are never vectorized into a raw cache."""
        cache = TestSkeletonVectorCache()._setup(tmp_path)
        cache.build()
        # mark the folder simplified (post-cleanup NeuPrint state)
        morph._write_skeleton_level_marker("test:v1", str(tmp_path))
        # a NEW simplified file appears on disk: must NOT be vectorized
        folder = tmp_path / "cache" / morph._dataset_folder("test:v1") / "skeletons"
        with open(folder / "999.pkl", "wb") as f:
            pickle.dump(line_neuron(), f)
        X, mask, reps = cache.vectors_for([101, 999], compute_missing=True)
        assert mask.tolist() == [True, False]  # 999 stays unscorable
        assert np.isnan(X[1]).all()
        assert reps[1] == ""
        # the cache basis was recorded as raw
        assert (cache._load_meta() or {}).get("vector_basis") == "raw"

    def test_vectors_for_computes_from_raw_files(self, tmp_path):
        """Raw folder level + raw basis: missing rows still compute (legacy)."""
        cache = TestSkeletonVectorCache()._setup(tmp_path)
        cache.build()
        folder = tmp_path / "cache" / morph._dataset_folder("test:v1") / "skeletons"
        with open(folder / "999.pkl", "wb") as f:
            pickle.dump(line_neuron(), f)
        X, mask, _ = cache.vectors_for([101, 999], compute_missing=True)
        assert mask.tolist() == [True, True]
        assert np.isfinite(X[1]).all()

    def test_build_skips_files_at_wrong_level(self, tmp_path):
        """build() must not vectorize on-disk files whose level != basis."""
        cache = TestSkeletonVectorCache()._setup(tmp_path)
        cache.build()
        morph._write_skeleton_level_marker("test:v1", str(tmp_path))
        # new simp90 file: build() skips it entirely
        folder = tmp_path / "cache" / morph._dataset_folder("test:v1") / "skeletons"
        with open(folder / "999.pkl", "wb") as f:
            pickle.dump(line_neuron(), f)
        stats = cache.build()
        assert stats["new"] == 0 and stats["rows"] == 3

    def test_append_vectors_rejects_different_basis(self, tmp_path):
        cache = TestSkeletonVectorCache()._setup(tmp_path)
        cache.build()
        _, vec = morph.vectorize_neuron(line_neuron())
        n = cache.append_vectors([(999, vec, "skeleton")],
                                 vector_basis="simp90")
        assert n == 0  # basis mismatch: never mixed
        # same basis appends fine
        n = cache.append_vectors([(999, vec, "skeleton")],
                                 vector_basis="raw")
        assert n == 1

    def test_population_stats_falls_back_to_cache_meta(self, tmp_path):
        """simp90 on-disk files -> population stats come from the raw vector
        cache meta (no raw skeleton sample exists any more)."""
        cache = TestSkeletonVectorCache()._setup(tmp_path)
        cache.build()
        morph._write_skeleton_level_marker("test:v1", str(tmp_path))
        mu, sd = morph.population_stats("test:v1", str(tmp_path))
        meta = cache._load_meta()
        assert mu is not None and sd is not None
        assert np.allclose(mu, meta["mean"])
        assert np.allclose(sd, meta["std"])

    def test_population_stats_computes_from_raw_files(self, tmp_path, monkeypatch):
        """Raw folder: sample-based stats still computed from on-disk files."""
        TestPopulationStats()._patch_sequential(monkeypatch)
        cache = TestSkeletonVectorCache()._setup(tmp_path)
        cache.build()
        mu, sd = morph.population_stats("test:v1", str(tmp_path))
        assert mu is not None and sd is not None
        assert mu.shape == (morph.VECTOR_DIM,)


# ---------------------------------------------------------------------------
# Candidate source & profile-first search
# ---------------------------------------------------------------------------

class TestCandidateSource:
    def test_auto_resolution(self):
        c = morph.MorphologyComparer(query=1, dataset="hemibrain:v1.2.1",
                                     project_root="/tmp", verbose=False)
        assert c._resolved_candidate_source() == "profile"
        c2 = morph.MorphologyComparer(query=1, dataset="flywire_FAFB_v783",
                                      project_root="/tmp", verbose=False)
        assert c2._resolved_candidate_source() == "cache"
        c3 = morph.MorphologyComparer(query=1, dataset="male-cns:v0.9",
                                      candidate_source="cache",
                                      project_root="/tmp", verbose=False)
        assert c3._resolved_candidate_source() == "cache"

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError, match="candidate_source"):
            morph.MorphologyComparer(query=1, dataset="x", candidate_source="bad",
                                     project_root="/tmp", verbose=False)

    def test_fetch_missing_alias(self):
        c = morph.MorphologyComparer(query=1, dataset="x", fetch_missing=7,
                                     project_root="/tmp", verbose=False)
        assert c.fetch_top_n == 7


class TestProfileFirst:
    """Connectivity-expanded pipeline: connection-cache candidates ->
    top-(top_n x candidate_expansion) types -> all members -> morphology."""

    def _setup(self, tmp_path, monkeypatch, query=101, **kwargs):
        write_skeleton(tmp_path, "np:v1", query, line_neuron(length=20))
        write_skeleton(tmp_path, "np:v1", 201, line_neuron(length=20))  # identical
        write_skeleton(tmp_path, "np:v1", 202, bushy_y_neuron())

        params = dict(
            query=query, dataset="np:v1", level="bodyid", method="vector",
            candidate_source="profile", candidate_expansion=3, top_n=5,
            output_dir=str(tmp_path / "out"), project_root=str(tmp_path),
            verbose=False,
        )
        params.update(kwargs)
        comparer = morph.MorphologyComparer(**params)
        monkeypatch.setattr(comparer, "_resolve_query", lambda: pd.DataFrame({
            "bodyId": [query], "type": ["T"], "instance": ["T_1"],
        }))
        return comparer

    def _candidates(self, rows):
        """Build a fake _connection_cache_candidates result."""
        return pd.DataFrame({
            "target_bodyId": [r[0] for r in rows],
            "shared_count": [r[1] for r in rows],
            "profile_similarity": [r[2] for r in rows],
            "target_type": [r[3] for r in rows],
        })

    def _index(self, tmp_path, entries):
        write_neuron_index(tmp_path, "np:v1", entries)

    def test_type_expansion_pool_and_transient_fetch(self, tmp_path, monkeypatch):
        """Candidates -> top types -> ALL their members become the pool;
        missing members are fetched transiently (never persisted)."""
        comparer = self._setup(tmp_path, monkeypatch)
        self._index(tmp_path, [
            (101, "T", "T_1"), (201, "T", "T_2"),
            (202, "Y", "Y_1"), (204, "Z", "Z_1"),
        ])
        monkeypatch.setattr(comparer, "_connection_cache_candidates",
                            lambda q: self._candidates([
                                (201, 9, 1.0, "T"), (202, 5, 0.6, "Y"),
                                (204, 3, 0.3, "Z"),
                            ]))
        fetched = []

        def fake_fetch(dataset, bid, project_root=None, persist=True):
            fetched.append(bid)
            # transient: the pipeline must not persist fetched skeletons
            assert persist is False
            return line_neuron(length=25)

        monkeypatch.setattr(morph, "fetch_skeleton_on_demand", fake_fetch)
        res = comparer.find_similar()
        assert not res.empty
        # pool = T {101, 201} + Y {202} + Z {204}; only 204 is missing
        assert fetched == [204]
        # no skeleton cache files were created by the transient fetches
        skel_dir = tmp_path / "cache" / morph._dataset_folder("np:v1") / "skeletons"
        assert sorted(p.name for p in skel_dir.glob("*.pkl")) ==             ["101.pkl", "201.pkl", "202.pkl"]
        # query neuron excluded from its own results
        assert 101 not in res["target_bodyId"].tolist()
        # identical morphology ranks first; all pool members are ranked
        assert res.iloc[0]["target_bodyId"] == 201
        assert {"profile_similarity", "target_type"} <= set(res.columns)
        assert set(res["target_bodyId"]) == {201, 202, 204}

    def test_cache_fetched_skeletons_controls_persistence(self, tmp_path, monkeypatch):
        """cache_fetched_skeletons=True persists transient fetches to the
        skeleton cache; the default keeps them memory-only (persist=False)."""
        comparer = self._setup(tmp_path, monkeypatch, cache_fetched_skeletons=True)
        self._index(tmp_path, [
            (101, "T", "T_1"), (201, "T", "T_2"),
            (202, "Y", "Y_1"), (204, "Z", "Z_1"),
        ])
        monkeypatch.setattr(comparer, "_connection_cache_candidates",
                            lambda q: self._candidates([
                                (201, 9, 1.0, "T"), (202, 5, 0.6, "Y"),
                                (204, 3, 0.3, "Z"),
                            ]))
        fetched = []

        def fake_fetch(dataset, bid, project_root=None, persist=True):
            fetched.append((bid, persist))
            return line_neuron(length=25)

        monkeypatch.setattr(morph, "fetch_skeleton_on_demand", fake_fetch)
        res = comparer.find_similar()
        assert not res.empty
        # pool = T {101, 201} + Y {202} + Z {204}; only 204 is missing, and
        # with the option on the fetch requests a permanent cache write.
        assert fetched == [(204, True)]
        # default (option off) stays memory-only
        default = self._setup(tmp_path, monkeypatch)
        assert default.cache_fetched_skeletons is False

    def test_type_expansion_caps_types(self, tmp_path, monkeypatch):
        """Only the top (top_n x expansion) TYPES enter the pool."""
        comparer = self._setup(tmp_path, monkeypatch, top_n=1, candidate_expansion=2)
        self._index(tmp_path, [
            (101, "T", "T_1"), (201, "T", "T_2"),
            (202, "Y", "Y_1"), (204, "Z", "Z_1"), (205, "Z", "Z_2"),
        ])
        write_skeleton(tmp_path, "np:v1", 204, line_neuron(length=25))
        write_skeleton(tmp_path, "np:v1", 205, line_neuron(length=25))
        monkeypatch.setattr(comparer, "_connection_cache_candidates",
                            lambda q: self._candidates([
                                (201, 9, 1.0, "T"), (202, 5, 0.6, "Y"),
                                (204, 3, 0.3, "Z"), (205, 3, 0.3, "Z"),
                            ]))
        monkeypatch.setattr(morph, "fetch_skeleton_on_demand",
                            lambda d, b, project_root=None, persist=True:
                            line_neuron(length=25))
        res = comparer.find_similar()
        # top-1 x 2 types = T, Y -> Z members never scored
        assert not res.empty
        assert "Z" not in res["target_type"].tolist()
        assert 204 not in res["target_bodyId"].tolist()

    def test_profile_first_bodyid_carries_intra_columns(self, tmp_path, monkeypatch):
        comparer = self._setup(tmp_path, monkeypatch)
        self._index(tmp_path, [
            (101, "T", "T_1"), (201, "T", "T_2"), (202, "Y", "Y_1"),
        ])
        monkeypatch.setattr(comparer, "_connection_cache_candidates",
                            lambda q: self._candidates([
                                (201, 9, 1.0, "T"), (202, 5, 0.6, "Y"),
                            ]))
        monkeypatch.setattr(morph, "fetch_skeleton_on_demand",
                            lambda d, b, project_root=None, persist=True:
                            line_neuron(length=25))
        res = comparer.find_similar()
        assert {"is_same_type", "intra_type_similarity"}.issubset(res.columns)
        # 201 is the same type (T) and identical morphology -> top hit
        assert res.iloc[0]["target_bodyId"] == 201
        assert res.iloc[0]["is_same_type"] == True
        # T has 2 identical members -> intra-type similarity ~1
        assert res.iloc[0]["intra_type_similarity"] == pytest.approx(1.0, abs=1e-6)
        assert res.iloc[1]["is_same_type"] == False

    def test_profile_first_type_level_includes_intra_reference(self, tmp_path, monkeypatch):
        comparer = self._setup(tmp_path, monkeypatch)
        self._index(tmp_path, [
            (101, "T", "T_1"), (201, "T", "T_2"), (202, "Y", "Y_1"),
        ])
        comparer.level = "type"
        monkeypatch.setattr(comparer, "_connection_cache_candidates",
                            lambda q: self._candidates([
                                (201, 9, 1.0, "T"), (202, 5, 0.6, "Y"),
                            ]))
        monkeypatch.setattr(morph, "fetch_skeleton_on_demand",
                            lambda d, b, project_root=None, persist=True:
                            line_neuron(length=25))
        res = comparer.find_similar()
        assert not res.empty
        intra = res[res["is_intra_type"]]
        assert len(intra) == 1
        assert intra.iloc[0]["target_type"] == "T"
        assert intra.iloc[0]["rank"] == 1
        assert intra.iloc[0]["similarity"] == pytest.approx(1.0, abs=1e-6)
        assert intra.iloc[0]["intra_type_similarity"] == pytest.approx(1.0, abs=1e-6)
        assert intra.iloc[0]["n_bodyids"] == 2
        inter = res[~res["is_intra_type"]]
        assert inter.iloc[0]["target_type"] == "Y"

    def test_type_level_works_without_vector_cache(self, tmp_path, monkeypatch):
        """Type-level profile-first works with skeletons + neuron index but
        NO vector cache (male-cns v1.0 situation)."""
        comparer = self._setup(tmp_path, monkeypatch)
        self._index(tmp_path, [
            (101, "T", "T_1"), (201, "T", "T_2"), (202, "Y", "Y_1"),
        ])
        comparer.level = "type"
        monkeypatch.setattr(comparer, "_connection_cache_candidates",
                            lambda q: self._candidates([
                                (201, 9, 1.0, "T"), (202, 5, 0.6, "Y"),
                            ]))
        monkeypatch.setattr(morph, "fetch_skeleton_on_demand",
                            lambda d, b, project_root=None, persist=True:
                            line_neuron(length=25))
        res = comparer.find_similar()
        assert not res.empty
        assert {"target_type", "is_intra_type", "intra_type_similarity"}.issubset(res.columns)
        intra = res[res["is_intra_type"] == True]  # noqa: E712
        assert len(intra) == 1
        assert intra.iloc[0]["target_type"] == "T"
        assert intra.iloc[0]["intra_type_similarity"] == pytest.approx(1.0, abs=1e-6)
        assert (res["target_type"] == "Y").any()

    def test_type_query_exports_all_members_and_intra_pairs(
        self, tmp_path, monkeypatch
    ):
        """A type query must not collapse to query_ids[0] in results.csv.

        The body-level export contains unique intra-type pairs for every
        resolved query member, while the type summary counts the type
        population rather than the pre-fetch cache snapshot.
        """
        comparer = self._setup(tmp_path, monkeypatch)
        # Make the two query members morphologically distinct so a
        # first-bodyId-only implementation cannot pass the type-mean check.
        write_skeleton(tmp_path, "np:v1", 102, bushy_y_neuron())
        self._index(tmp_path, [
            (101, "Q", "Q_1"), (102, "Q", "Q_2"),
            (201, "T", "T_1"), (202, "Y", "Y_1"),
        ])
        comparer.level = "type"
        monkeypatch.setattr(comparer, "_resolve_query", lambda: pd.DataFrame({
            "bodyId": [101, 102], "type": ["Q", "Q"],
            "instance": ["Q_1", "Q_2"],
        }))
        monkeypatch.setattr(comparer, "_connection_cache_candidates",
                            lambda q: self._candidates([
                                (201, 9, 1.0, "T"), (202, 5, 0.6, "Y"),
                            ]))
        monkeypatch.setattr(morph, "fetch_skeleton_on_demand",
                            lambda d, b, project_root=None, persist=True:
                            line_neuron(length=25))

        res = comparer.find_similar()
        assert not res.empty
        body = pd.read_csv(Path(comparer.output_folder) / "results.csv")
        endpoints = set(body["source_bodyId"]) | set(body["target_bodyId"])
        assert {101, 102}.issubset(endpoints)
        intra = body[
            (body["target_type"] == "Q")
            & (body["is_same_type"] == True)  # noqa: E712
        ]
        assert set(zip(intra["source_bodyId"], intra["target_bodyId"])) == {
            (101, 102)
        }
        assert intra["intra_type_similarity"].iloc[0] == pytest.approx(
            intra["similarity"].mean(),
            abs=1e-6,
        )

        summary = pd.read_csv(Path(comparer.output_folder) / "type_summary.csv")
        row = summary[summary["target_type"] == "Q"].iloc[0]
        assert row["n_bodyids"] == 2
        assert row["is_intra_type"] == True  # noqa: E712
        target_rows = body[body["target_type"] == "T"]
        assert set(target_rows["source_bodyId"]) == {101, 102}
        target_row = summary[summary["target_type"] == "T"].iloc[0]
        assert target_row["similarity"] == pytest.approx(
            target_rows["similarity"].mean(), abs=1e-6
        )

    def test_type_search_uses_only_connectivity_candidates(self, tmp_path, monkeypatch):
        """Regression: a cached-only type must NOT rank; a member of an
        expanded candidate type that is not itself a candidate row must."""
        comparer = self._setup(tmp_path, monkeypatch)
        # 103 (Y) is cached but Y is NOT among the connectivity candidates
        write_skeleton(tmp_path, "np:v1", 103, y_neuron())
        self._index(tmp_path, [
            (101, "T", "T_1"), (201, "T", "T_2"),
            (202, "Y", "Y_1"), (103, "Y", "Y_2"),
        ])
        comparer.level = "type"
        monkeypatch.setattr(comparer, "_connection_cache_candidates",
                            lambda q: self._candidates([(201, 9, 1.0, "T")]))
        monkeypatch.setattr(morph, "fetch_skeleton_on_demand",
                            lambda d, b, project_root=None, persist=True:
                            line_neuron(length=25))
        res = comparer.find_similar()
        assert not res.empty
        assert not (res["target_type"] == "Y").any(), \
            "cached-only type leaked into the connectivity-only pool"
        # now Y enters via a candidate: all Y members (even non-candidate 103)
        # join the pool through the type expansion
        comparer2 = self._setup(tmp_path, monkeypatch)
        comparer2.level = "type"
        monkeypatch.setattr(comparer2, "_connection_cache_candidates",
                            lambda q: self._candidates([
                                (201, 9, 1.0, "T"), (103, 4, 0.4, "Y"),
                            ]))
        res2 = comparer2.find_similar()
        assert (res2["target_type"] == "Y").any(),             "type -> individual expansion failed (non-candidate member missing)"

    def test_pool_standardization_without_vector_cache(self, tmp_path, monkeypatch):
        """Without a vector cache the pool vectors are z-scored with
        pool-computed stats; scores match the manual z-scored cosine."""
        comparer = self._setup(tmp_path, monkeypatch)
        self._index(tmp_path, [
            (101, "T", "T_1"), (201, "T", "T_2"), (202, "Y", "Y_1"),
        ])
        monkeypatch.setattr(comparer, "_connection_cache_candidates",
                            lambda q: self._candidates([
                                (201, 9, 1.0, "T"), (202, 5, 0.6, "Y"),
                            ]))
        monkeypatch.setattr(morph, "fetch_skeleton_on_demand",
                            lambda d, b, project_root=None, persist=True:
                            line_neuron(length=25))
        res = comparer.find_similar()
        assert not res.empty

        # manual: raw vectors for query + pool, z-scored with pool stats
        cache = morph.SkeletonVectorCache("np:v1", project_root=str(tmp_path), verbose=False)
        ids = [101, 201, 202]
        Xr, mask, _ = cache.vectors_for(ids, compute_missing=True)
        assert mask.all()
        mu = Xr.mean(axis=0)
        sd = Xr.std(axis=0)
        sd[sd <= 0] = 1.0
        Xz = (Xr - mu) / sd
        qz = Xz[0]
        for _, row in res.iterrows():
            idx = ids.index(int(row["target_bodyId"]))
            expected = morph.cosine_similarity_matrix(qz, Xz[idx].reshape(1, -1))[0]
            assert row["similarity"] == pytest.approx(expected, abs=1e-9),                 row["target_bodyId"]

    def test_download_all_skeletons_resumable(self, tmp_path, monkeypatch):
        """download_all_skeletons: persist=True, skips existing, respects
        limit and the cancel event."""
        write_neuron_index(tmp_path, "np:v1", [
            (101, "T", "T_1"), (201, "T", "T_2"), (202, "Y", "Y_1"),
        ])
        fetched = []

        def fake_fetch(dataset, bid, project_root=None, persist=True):
            assert persist is True
            fetched.append(bid)
            folder = (Path(project_root) / "cache"
                      / morph._dataset_folder(dataset) / "skeletons")
            folder.mkdir(parents=True, exist_ok=True)
            with open(folder / f"{bid}.pkl", "wb") as f:
                pickle.dump(line_neuron(length=25), f)
            return line_neuron(length=25)

        monkeypatch.setattr(morph, "fetch_skeleton_on_demand", fake_fetch)
        progress = []

        summary = morph.download_all_skeletons(
            "np:v1", project_root=str(tmp_path), max_workers=2, limit=2,
            progress_callback=lambda c, t, i: progress.append((c, t)),
            verbose=False,
        )
        assert summary["fetched"] == 2 and summary["errors"] == 0
        assert set(fetched) == {101, 201}  # deterministic first two of the index
        assert progress and progress[-1][0] == progress[-1][1] == 2
        # resume: the two fetched are now cached; limit=2 fetches the rest
        summary2 = morph.download_all_skeletons(
            "np:v1", project_root=str(tmp_path), max_workers=2, limit=2,
            verbose=False,
        )
        assert summary2["fetched"] == 1 and 202 in fetched
        assert summary2["skipped_existing"] == 2

    def test_download_all_skeletons_cancel(self, tmp_path, monkeypatch):
        import threading
        write_neuron_index(tmp_path, "np:v1", [
            (101, "T", "T_1"), (201, "T", "T_2"), (202, "Y", "Y_1"),
        ])
        monkeypatch.setattr(morph, "fetch_skeleton_on_demand",
                            lambda d, b, project_root=None, persist=True:
                            line_neuron(length=25))
        ev = threading.Event()
        ev.set()  # pre-cancelled
        summary = morph.download_all_skeletons(
            "np:v1", project_root=str(tmp_path), max_workers=2,
            cancel_event=ev, verbose=False,
        )
        assert summary["cancelled"] is True
        assert summary["fetched"] == 0

    def test_find_homologs_fast_bodyid_reaches_comparison(self, tmp_path, monkeypatch):
        """Regression: the bodyId branch of find_homologs_fast was nested
        inside ``if not is_bodyid:`` (dead code), so bodyId queries returned
        None and profile-first searches always fell back to the vector
        cache. The branch must run and return candidates."""
        from comparison.profile_comparator import HomologFinder
        from comparison.connectivity_profiler import ConnectivityStatus

        class FakeProfile:
            neuron_type = "T"
            upstream_partners = {"A": 10.0, "B": 20.0}
            downstream_partners = {"C": 15.0}
            connectivity_status = ConnectivityStatus.COMPLETE

        conn = pd.DataFrame({
            "bodyId_pre": [1, 2, 3],
            "bodyId_post": [2, 1, 2],
            "weight": [5, 5, 5],
            "type_pre": ["A", "C", "A"],
            "type_post": ["C", "A", "C"],
        })
        finder = HomologFinder(
            source="1", source_dataset="d:v1", target_dataset="d:v1",
            output_dir=str(tmp_path), top_n=5, verbose=False,
            min_shared_partners=1, vector_prune_fraction=1.0,
            vector_prefiltering=False, min_synapse_threshold=1,
            include_untyped_partners=True, use_cache=True,
            ensure_cache_complete=False,
        )
        monkeypatch.setattr(finder, "_load_connection_cache",
                            lambda ds, auto_build=False: conn)
        monkeypatch.setattr(finder, "_prewarm_profile_cache", lambda ds: None)
        monkeypatch.setattr(finder.profiler, "get_profile",
                            lambda bid, ds: FakeProfile())
        monkeypatch.setattr(finder.profiler, "consolidate_profile_cache",
                            lambda ds: None)
        monkeypatch.setattr(finder.profiler, "_get_cached_conn_df",
                            lambda ds: conn)
        monkeypatch.setattr(finder.profiler, "flush_pending_cache_writes",
                            lambda silent=False: None)
        monkeypatch.setattr(finder, "_build_profiles_batch",
                            lambda ids, ds, **kw: {int(b): FakeProfile() for b in ids})
        reached = {}

        def fake_core(**kw):
            reached["source_bodyids"] = kw.get("source_bodyids")
            res = pd.DataFrame({
                "source_bodyId": [1], "target_bodyId": [2],
                "target_type": ["T"], "rank_union": [0.9],
            })
            return (res, pd.DataFrame(), {"none": []}, {"rare_or_uni": []},
                    {}, {}, {})

        monkeypatch.setattr(finder, "_compare_candidates_core", fake_core)
        monkeypatch.setattr(finder, "_save_homolog_results_internal",
                            lambda **kw: None)
        res = finder.find_homologs_fast()
        assert res is not None and len(res) == 1
        assert reached["source_bodyids"] == [1]


# ---------------------------------------------------------------------------
# Homolog enrichment
# ---------------------------------------------------------------------------

class TestConnectionCacheCandidates:
    """Candidate discovery straight from the connections parquet."""

    def _setup(self, tmp_path, monkeypatch, with_roi=True):
        folder = tmp_path / "cache" / morph._dataset_folder("np:v1")
        folder.mkdir(parents=True, exist_ok=True)
        rows = {
            "bodyId_pre": ["2", "3", "4", "5", "1", "1", "2", "3", "3", "6"],
            "bodyId_post": ["1", "1", "1", "1", "2", "3", "6", "6", "7", "2"],
            "weight": [10, 8, 7, 2, 10, 5, 9, 4, 6, 9],
        }
        if with_roi:
            rows["roi"] = ["AL(L)", "AL(R)", "AL(R)", "", "AL(L)", "AL(L)",
                           "AL(L)", "AL(L)", "AL(R)", "AL(L)"]
        rows["cached_date"] = ["x"] * len(rows["bodyId_pre"])
        pd.DataFrame(rows).to_parquet(folder / "connections.parquet", index=False)
        write_neuron_index(tmp_path, "np:v1", [
            (1, "Q", "Q_1"), (2, "A", "A_1"), (3, "B", "B_1"),
            (4, "C", "C_1"), (5, "D", "D_1"), (6, "E", "E_1"), (7, "F", "F_1"),
        ])
        comparer = morph.MorphologyComparer(query=1, dataset="np:v1",
                                            project_root=str(tmp_path), verbose=False)
        monkeypatch.setattr(comparer, "_resolve_query", lambda: pd.DataFrame({
            "bodyId": [1], "type": ["Q"], "instance": ["Q_1"],
        }))
        return comparer

    def test_shared_partner_ranking_and_type_join(self, tmp_path, monkeypatch):
        comparer = self._setup(tmp_path, monkeypatch)
        qdf = comparer._resolve_query()
        res = comparer._connection_cache_candidates(
            qdf, min_weight=3, min_shared_partners=2
        )
        # query 1 partners (w>=3): up {2,3,4}, down {2,3}; 5 excluded (w=2).
        # candidate 6 shares upstream {2,3} + downstream {2} = 3; candidate 7
        # shares upstream {3} = 1 -> below min_shared_partners.
        assert res["target_bodyId"].tolist() == [6]
        assert res["shared_count"].tolist() == [3]
        assert res["profile_similarity"].tolist() == [1.0]
        assert res["target_type"].tolist() == ["E"]

    def test_roi_filter_restricts_candidates(self, tmp_path, monkeypatch):
        comparer = self._setup(tmp_path, monkeypatch)
        qdf = comparer._resolve_query()
        # AL(R) rows only: 3->1, 4->1, 3->7. Query partners {3,4}; shared
        # upstream for 7 = {3} -> 1 shared -> below the minimum.
        res = comparer._connection_cache_candidates(
            qdf, min_weight=3, min_shared_partners=2, roi_filter=["AL(R)"]
        )
        assert res.empty

    def test_dataset_without_roi_column_falls_back_to_all_rows(self, tmp_path, monkeypatch):
        comparer = self._setup(tmp_path, monkeypatch, with_roi=False)
        qdf = comparer._resolve_query()
        res = comparer._connection_cache_candidates(
            qdf, min_weight=3, min_shared_partners=2
        )
        assert res["target_bodyId"].tolist() == [6]
        assert res["shared_count"].tolist() == [3]

    def test_missing_connection_cache_returns_empty(self, tmp_path, monkeypatch):
        comparer = morph.MorphologyComparer(query=1, dataset="np:v1",
                                            project_root=str(tmp_path), verbose=False)
        monkeypatch.setattr(comparer, "_resolve_query", lambda: pd.DataFrame({
            "bodyId": [1], "type": ["Q"], "instance": ["Q_1"],
        }))
        assert comparer._connection_cache_candidates(comparer._resolve_query()).empty


class TestEnrichment:
    def test_enrichment_adds_columns_with_valid_range(self, tmp_path):
        write_skeleton(tmp_path, "src:v1", 101, line_neuron())
        write_skeleton(tmp_path, "src:v1", 102, translated(line_neuron()))
        write_skeleton(tmp_path, "tgt:v1", 201, line_neuron())
        write_skeleton(tmp_path, "tgt:v1", 202, y_neuron())
        results = pd.DataFrame({
            "source_bodyId": [101, 102, 101],
            "target_bodyId": [201, 202, 999],
        })
        enriched = morph.enrich_homolog_results(
            results, "src:v1", "tgt:v1", project_root=str(tmp_path), verbose=False
        )
        assert {"morph_cosine", "morph_pearson"}.issubset(enriched.columns)
        # 101 vs 201 = identical morphology -> cosine ~1
        assert enriched.loc[0, "morph_cosine"] > 0.9
        # missing skeleton -> NaN, row preserved
        assert np.isnan(enriched.loc[2, "morph_cosine"])
        assert len(enriched) == len(results)
        # pearson in [-1, 1] (float tolerance)
        assert -1 - 1e-9 <= enriched.loc[0, "morph_pearson"] <= 1 + 1e-9

    def test_enrichment_skips_without_required_columns(self, tmp_path):
        results = pd.DataFrame({"a": [1], "b": [2]})
        out = morph.enrich_homolog_results(results, "x", "y", project_root=str(tmp_path))
        assert out.equals(results)

    def test_enrichment_empty_df(self, tmp_path):
        out = morph.enrich_homolog_results(pd.DataFrame(), "x", "y", project_root=str(tmp_path))
        assert out.empty

    def test_homolog_finder_flag_disables_enrichment(self, tmp_path, monkeypatch):
        from comparison.profile_comparator import HomologFinder
        finder = HomologFinder(source="a", source_dataset="d", target_dataset="d",
                               verbose=False, morphological_enrichment=False)
        assert finder.morphological_enrichment is False
        df = pd.DataFrame({"source_bodyId": [1], "target_bodyId": [2]})
        out = finder._enrich_with_morphology(df, "d", "d")
        assert "morph_cosine" not in out.columns

    def test_homolog_finder_enrichment_uses_backend(self, tmp_path, monkeypatch):
        write_skeleton(tmp_path, "d:v1", 1, line_neuron())
        write_skeleton(tmp_path, "d:v1", 2, line_neuron(length=5))
        from comparison.profile_comparator import HomologFinder
        finder = HomologFinder(source="a", source_dataset="d:v1", target_dataset="d:v1",
                               verbose=False, morphological_enrichment=True)
        finder.project_root = str(tmp_path)
        df = pd.DataFrame({"source_bodyId": [1], "target_bodyId": [2]})
        out = finder._enrich_with_morphology(df, "d:v1", "d:v1")
        assert "morph_cosine" in out.columns
        assert np.isfinite(out.loc[0, "morph_cosine"])


# ---------------------------------------------------------------------------
# Step-progress events ([DROCAT][progress] protocol for the web UI)
# ---------------------------------------------------------------------------

class TestStepProgress:
    """Structured step-progress events emitted by the similarity pipeline."""

    def test_progress_event_format_and_verbose_gating(self, capsys):
        """The event line is '[DROCAT][progress] <step>/<total> <label>' and
        is only emitted when verbose is enabled."""
        c = morph.MorphologyComparer(query=1, dataset="np:v1",
                                     project_root="/tmp", verbose=True)
        c._progress(2, 6, "Discovering candidates (connection cache)")
        out = capsys.readouterr().out
        assert "[DROCAT][progress] 2/6 Discovering candidates (connection cache)" in out

        # Labels are optional; trailing whitespace is stripped.
        c._progress(6, 6, "")
        assert "[DROCAT][progress] 6/6" in capsys.readouterr().out

        # Silent when verbose is off (nothing leaks into quiet runs).
        c2 = morph.MorphologyComparer(query=1, dataset="np:v1",
                                      project_root="/tmp", verbose=False)
        c2._progress(1, 4, "Loading vector cache")
        assert capsys.readouterr().out == ""

    def test_find_similar_emits_ordered_step_events(self, tmp_path, monkeypatch, capsys):
        """A full profile-first run reports steps 1/6..6/6 in pipeline order."""
        write_skeleton(tmp_path, "np:v1", 101, line_neuron(length=20))
        write_skeleton(tmp_path, "np:v1", 201, line_neuron(length=20))
        write_skeleton(tmp_path, "np:v1", 202, bushy_y_neuron())
        write_neuron_index(tmp_path, "np:v1", [
            (101, "T", "T_1"), (201, "T", "T_2"), (202, "Y", "Y_1"),
        ])

        comparer = morph.MorphologyComparer(
            query=101, dataset="np:v1", level="bodyid", method="vector",
            candidate_source="profile", top_n=5, candidate_expansion=3,
            output_dir=str(tmp_path / "out"), project_root=str(tmp_path),
            verbose=True,
        )
        monkeypatch.setattr(comparer, "_resolve_query", lambda: pd.DataFrame({
            "bodyId": [101], "type": ["T"], "instance": ["T_1"],
        }))
        monkeypatch.setattr(comparer, "_connection_cache_candidates",
                            lambda q: pd.DataFrame({
                                "target_bodyId": [201, 202],
                                "shared_count": [9, 5],
                                "profile_similarity": [1.0, 0.6],
                                "target_type": ["T", "Y"],
                            }))

        res = comparer.find_similar()
        assert not res.empty

        out = capsys.readouterr().out
        assert "Step 2/6 — Discovering candidates" in out
        assert "Step 3/6 — Expanding 2 candidate types" in out
        assert "Step 4/6 — Loading & vectorizing skeletons" in out
        assert "Profile-first:" in out and "pool neurons" in out
        events = [ln.strip() for ln in out.splitlines()
                  if "[DROCAT][progress]" in ln]
        # visualize_top_n defaults to 0 here, so the final label reflects
        # the visualization-free save phase.
        assert events == [
            "[DROCAT][progress] 1/6 Resolving query neuron",
            "[DROCAT][progress] 2/6 Discovering candidates (connection cache)",
            "[DROCAT][progress] 3/6 Expanding 2 candidate types to the scoring pool",
            "[DROCAT][progress] 4/6 Loading & vectorizing skeletons",
            "[DROCAT][progress] 5/6 Scoring similarity (vector)",
            "[DROCAT][progress] 6/6 Saving results",
        ]

        # The 'Results saved to:' marker must carry the path ONLY (the UI
        # splits after the marker and checks isdir to find the run folder
        # for streaming output files).
        marker = [ln.strip() for ln in out.splitlines()
                  if "Results saved to:" in ln]
        assert marker, "expected a 'Results saved to:' marker line"
        run_dir = marker[-1].split("Results saved to:", 1)[1].strip()
        assert os.path.isdir(run_dir)
        assert run_dir == comparer.output_folder

    def test_homolog_finder_progress_event(self, capsys):
        """HomologFinder emits the same structured event protocol (plain print
        outside a tqdm context, tqdm.write inside one)."""
        from comparison.profile_comparator import HomologFinder
        finder = HomologFinder(source="a", source_dataset="d:v1",
                               target_dataset="d:v1", verbose=True)
        finder._progress(2, 6, "Building source profiles")
        assert ("[DROCAT][progress] 2/6 Building source profiles"
                in capsys.readouterr().out)

        # Inside a progress bar the event goes through tqdm.write (the UI
        # consumes it the same way regardless of the transport).
        finder._in_progress_bar = True
        finder._progress(4, 6, "Building target profiles")
        assert ("[DROCAT][progress] 4/6 Building target profiles"
                in capsys.readouterr().out)
        finder._in_progress_bar = False

        # Silent when verbose is off.
        quiet = HomologFinder(source="a", source_dataset="d:v1",
                              target_dataset="d:v1", verbose=False)
        quiet._progress(1, 6, "Loading connection data")
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Population standardization statistics
# ---------------------------------------------------------------------------

class TestPopulationStats:
    """population_stats(): stable standardization stats for datasets without
    a vector cache, with version-sibling borrowing for sparse caches."""

    def _patch_sequential(self, monkeypatch):
        """Fork-based parallel vectorization is not deterministic under
        pytest; these tests exercise the statistics logic, not the worker
        pool, so run the vectorization sequentially."""
        monkeypatch.setattr(
            morph.SkeletonVectorCache, "_vectorize_parallel",
            lambda self, files: [_vectorize_one_file(p) for p in files],
        )

    def test_persists_and_reuses(self, tmp_path, monkeypatch):
        self._patch_sequential(monkeypatch)
        write_skeleton(tmp_path, "np:v1", 101, line_neuron(length=20))
        write_skeleton(tmp_path, "np:v1", 201, bushy_y_neuron())
        write_skeleton(tmp_path, "np:v1", 202, y_neuron())
        mu, sd = morph.population_stats("np:v1", str(tmp_path))
        assert mu.shape == (morph.VECTOR_DIM,)
        assert sd.shape == (morph.VECTOR_DIM,)
        assert (sd > 0).all()
        stats_file = tmp_path / "cache" / "np_v1" / "morphology" / "population_stats.json"
        assert stats_file.exists()
        # second call loads the persisted file (same arrays)
        mu2, sd2 = morph.population_stats("np:v1", str(tmp_path))
        np.testing.assert_array_equal(mu2, mu)
        np.testing.assert_array_equal(sd2, sd)

    def test_none_without_skeletons(self, tmp_path):
        mu, sd = morph.population_stats("empty:v1", str(tmp_path))
        assert mu is None and sd is None

    def test_borrows_version_sibling_stats(self, tmp_path, monkeypatch):
        """A dataset with too few cached skeletons samples its stats from
        the version sibling's cache (shared reconstruction, e.g. male-cns
        v1.0 <- v0.9), so both datasets share the same statistics."""
        self._patch_sequential(monkeypatch)
        write_skeleton(tmp_path, "mc:v0.9", 1, line_neuron(length=20))
        write_skeleton(tmp_path, "mc:v0.9", 2, bushy_y_neuron())
        write_neuron_index(tmp_path, "mc:v0.9", [(1, "A", "A1"), (2, "B", "B1")])
        write_skeleton(tmp_path, "mc:v1.0", 1, line_neuron(length=20))
        write_neuron_index(tmp_path, "mc:v1.0", [(1, "A", "A1"), (3, "C", "C1")])
        # 50% of the smaller index is shared -> valid sibling

        mu09, sd09 = morph.population_stats("mc:v0.9", str(tmp_path))
        assert mu09 is not None
        mu10, sd10 = morph.population_stats("mc:v1.0", str(tmp_path))
        assert mu10 is not None
        # v1.0 samples the sibling's (larger) cache: identical stats
        np.testing.assert_array_equal(mu10, mu09)
        np.testing.assert_array_equal(sd10, sd09)
        stats_file = (tmp_path / "cache" / "mc_v1_0" / "morphology"
                      / "population_stats.json")
        assert stats_file.exists()

    def test_does_not_borrow_from_unrelated_dataset(self, tmp_path, monkeypatch):
        """Same name prefix but no shared population -> no borrowing."""
        self._patch_sequential(monkeypatch)
        write_skeleton(tmp_path, "mc:v0.9", 1, line_neuron(length=20))
        write_neuron_index(tmp_path, "mc:v0.9", [(1, "A", "A1")])
        # unrelated sibling: no overlapping bodyIds
        write_neuron_index(tmp_path, "mc:v1.0", [(999, "X", "X1")])
        morph.population_stats("mc:v0.9", str(tmp_path))
        mu, sd = morph.population_stats("mc:v1.0", str(tmp_path))
        assert mu is None and sd is None


# ---------------------------------------------------------------------------
# Representation consistency (skeleton vs mesh; one level per comparison)
# ---------------------------------------------------------------------------

class TestRepresentationConsistency:
    """Skeletons and meshes (or two simplification levels) are never mixed
    within one vector cache or one comparison."""

    @staticmethod
    def _write_mesh(tmp_path, dataset, body_id):
        folder = tmp_path / "cache" / morph._dataset_folder(dataset) / "skeletons"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{body_id}.pkl"
        with open(path, "wb") as f:
            pickle.dump(cube_mesh(), f)
        return path

    def test_vectorize_one_file_reports_rep(self, tmp_path):
        p = write_skeleton(tmp_path, "np:v1", 101, line_neuron(length=20))
        bid, morph_vals, pv_vals, rep = morph._vectorize_one_file(str(p))
        assert (bid, rep) == (101, "skeleton")
        assert len(morph_vals) == 24 and len(pv_vals) == 100
        mp = self._write_mesh(tmp_path, "np:v1", 901)
        assert morph._vectorize_one_file(str(mp))[3] == "mesh"

    def test_build_skips_foreign_representation(self, tmp_path):
        """A cache holds ONE representation: a mesh pickle beside skeleton
        pickles is skipped at build time."""
        write_skeleton(tmp_path, "np:v1", 101, line_neuron(length=20))
        write_skeleton(tmp_path, "np:v1", 201, bushy_y_neuron())
        self._write_mesh(tmp_path, "np:v1", 901)
        cache = morph.SkeletonVectorCache(
            "np:v1", project_root=str(tmp_path), verbose=False
        )
        stats = cache.build()
        assert stats["rows"] == 2  # the mesh row never entered the cache
        data = cache.load()
        assert data["dataset_rep"] == "skeleton"
        assert set(data["rep"]) == {"skeleton"}

    def test_vectors_for_skips_foreign_rep_files(self, tmp_path):
        write_skeleton(tmp_path, "np:v1", 101, line_neuron(length=20))
        cache = morph.SkeletonVectorCache(
            "np:v1", project_root=str(tmp_path), verbose=False
        )
        cache.build()
        # a mesh pickle added after the build is not vectorized into the
        # comparison (different representation than the cache's)
        self._write_mesh(tmp_path, "np:v1", 901)
        X, mask, reps = cache.vectors_for([101, 901], compute_missing=True)
        assert mask.tolist() == [True, False]
        assert reps == ["skeleton", ""]

    def test_profile_first_drops_foreign_rep_pool(self, tmp_path, monkeypatch):
        """A pool neuron cached as a mesh is unscorable against a skeleton
        query: representations are never mixed within one comparison."""
        write_skeleton(tmp_path, "np:v1", 101, line_neuron(length=20))
        write_skeleton(tmp_path, "np:v1", 201, line_neuron(length=20))
        self._write_mesh(tmp_path, "np:v1", 301)
        write_neuron_index(tmp_path, "np:v1", [
            (101, "T", "T_1"), (201, "T", "T_2"), (301, "M", "M_1"),
        ])
        comparer = morph.MorphologyComparer(
            query=101, dataset="np:v1", level="bodyid", method="vector",
            candidate_source="profile", top_n=5, candidate_expansion=3,
            output_dir=str(tmp_path / "out"), project_root=str(tmp_path),
            verbose=False,
        )
        monkeypatch.setattr(comparer, "_resolve_query", lambda: pd.DataFrame({
            "bodyId": [101], "type": ["T"], "instance": ["T_1"],
        }))
        monkeypatch.setattr(comparer, "_connection_cache_candidates",
                            lambda q: pd.DataFrame({
                                "target_bodyId": [201, 301],
                                "shared_count": [9, 5],
                                "profile_similarity": [1.0, 0.6],
                                "target_type": ["T", "M"],
                            }))
        res = comparer.find_similar()
        # the skeleton pool neuron is scored; the mesh is excluded
        assert 201 in res["target_bodyId"].tolist()
        assert 301 not in res["target_bodyId"].tolist()


# ---------------------------------------------------------------------------
# Vector persistence (always cache the vector, even without the skeleton)
# ---------------------------------------------------------------------------

class TestVectorPersistence:
    """Computed vectors are always persisted into the vector cache: the
    workflow is vector cache -> cached skeleton file -> online fetch, and
    the vector survives even when the original skeleton does not."""

    def test_vectors_for_persists_computed_vectors(self, tmp_path):
        """A vector computed from a cached skeleton file is appended, so a
        later call is a cache hit (no re-vectorization)."""
        write_skeleton(tmp_path, "np:v1", 101, line_neuron(length=20))
        cache = morph.SkeletonVectorCache(
            "np:v1", project_root=str(tmp_path), verbose=False
        )
        cache.build()
        # a new skeleton file appears after the build
        write_skeleton(tmp_path, "np:v1", 202, bushy_y_neuron())
        X, mask, _ = cache.vectors_for([101, 202], compute_missing=True)
        assert mask.tolist() == [True, True]
        # the computed vector was persisted: a fresh call hits the cache
        data = cache.load()
        assert len(data["bodyIds"]) == 2
        X2, mask2, _ = cache.vectors_for([202], compute_missing=True)
        assert mask2[0]
        # the stored row is the same RAW vector, standardized on load
        idx = int(np.where(data["bodyIds"] == 202)[0][0])
        meta = data["meta"]
        raw_recovered = (data["X"][idx] * np.asarray(meta["std"])
                         + np.asarray(meta["mean"]))
        np.testing.assert_allclose(raw_recovered, X[1], rtol=1e-6)

    def test_profile_first_persists_transient_fetch_vector(self, tmp_path, monkeypatch):
        """A transiently-fetched skeleton (never written to the skeleton
        cache) still persists its VECTOR; the next run reuses the vector
        and skips the online fetch entirely."""
        write_skeleton(tmp_path, "np:v1", 101, line_neuron(length=20))
        write_skeleton(tmp_path, "np:v1", 201, line_neuron(length=20))
        write_neuron_index(tmp_path, "np:v1", [
            (101, "T", "T_1"), (201, "T", "T_2"), (204, "Z", "Z_1"),
        ])
        fetched = []

        def fake_fetch(dataset, bid, project_root=None, persist=True):
            fetched.append(bid)
            return line_neuron(length=25)

        monkeypatch.setattr(morph, "fetch_skeleton_on_demand", fake_fetch)
        params = dict(
            query=101, dataset="np:v1", level="bodyid", method="vector",
            candidate_source="profile", top_n=5, candidate_expansion=3,
            output_dir=str(tmp_path / "out"), project_root=str(tmp_path),
            verbose=False,
        )

        def make():
            c = morph.MorphologyComparer(**params)
            monkeypatch.setattr(c, "_resolve_query", lambda: pd.DataFrame({
                "bodyId": [101], "type": ["T"], "instance": ["T_1"],
            }))
            monkeypatch.setattr(c, "_connection_cache_candidates",
                                lambda q: pd.DataFrame({
                                    "target_bodyId": [201, 204],
                                    "shared_count": [9, 5],
                                    "profile_similarity": [1.0, 0.6],
                                    "target_type": ["T", "Z"],
                                }))
            return c

        # run 1: 204 has no cached skeleton -> transient fetch
        res = make().find_similar()
        assert not res.empty
        assert fetched == [204]
        # the vector was persisted even though the skeleton was not
        cache = morph.SkeletonVectorCache(
            "np:v1", project_root=str(tmp_path), verbose=False
        )
        data = cache.load()
        assert data is not None
        assert int(204) in set(int(b) for b in data["bodyIds"])
        assert data["dataset_rep"] == "skeleton"
        assert not (tmp_path / "cache" / "np_v1" / "skeletons" / "204.pkl").exists()

        # run 2: the cached vector suffices -> no online fetch at all
        fetched.clear()
        res2 = make().find_similar()
        assert not res2.empty
        assert fetched == []

    def test_append_ignores_duplicate_bodyids(self, tmp_path):
        write_skeleton(tmp_path, "np:v1", 101, line_neuron(length=20))
        cache = morph.SkeletonVectorCache(
            "np:v1", project_root=str(tmp_path), verbose=False
        )
        cache.build()
        before = len(cache.load()["bodyIds"])
        X, _, _ = cache.vectors_for([101], compute_missing=True)
        added = cache.append_vectors([(101, X[0], "skeleton")])
        assert added == 0
        assert len(cache.load()["bodyIds"]) == before
