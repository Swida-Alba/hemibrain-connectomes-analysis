"""Tests for src/morphology.py — morphological similarity, vector cache,
NBLAST wrapper, on-demand fetch, and homolog enrichment."""

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
    folder = tmp_path / "cache" / morph._dataset_folder(dataset)
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
        X, mask = cache.vectors_for([101, 103, 999])
        assert mask.tolist() == [True, True, False]
        assert np.isnan(X[2]).all()
        assert np.isfinite(X[0]).all()

    def test_vectors_for_without_built_cache(self, tmp_path):
        cache = self._setup(tmp_path)
        # no parquet yet: still resolves from existing skeleton files
        X, mask = cache.vectors_for([101, 102])
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
    def test_neuprint_fetch_persists_and_reuses(self, tmp_path, monkeypatch):
        monkeypatch.setattr(morph, "_fetch_neuprint_skeleton",
                            lambda dataset, bid: line_neuron())
        pkl = write_skeleton(tmp_path, "test:v1", 101, None)  # placeholder path
        pkl.unlink()  # ensure missing

        neuron = morph.fetch_skeleton_on_demand(
            "test:v1", 101, project_root=str(tmp_path)
        )
        assert neuron is not None
        assert pkl.exists()
        # second call reuses the persisted file without hitting the fetcher
        calls = {"n": 0}

        def counting_fetch(dataset, bid):
            calls["n"] += 1
            raise AssertionError("should not be called again")

        monkeypatch.setattr(morph, "_fetch_neuprint_skeleton", counting_fetch)
        again = morph.fetch_skeleton_on_demand("test:v1", 101, project_root=str(tmp_path))
        assert calls["n"] == 0
        assert again is not None

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
        # results saved with the findsimilar_ prefix
        run_dirs = [p for p in (root / "out").iterdir() if p.is_dir()]
        assert len(run_dirs) == 1
        assert run_dirs[0].name.startswith("findsimilar_")
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
        summary = pd.read_csv(run_dir / "type_summary.csv")
        assert {"target_type", "avg_similarity", "max_similarity", "min_similarity",
                "std_similarity", "n_bodyids", "is_query_type"}.issubset(summary.columns)
        # identical-morphology LINE partner ranks first; flagged as query type
        assert summary.iloc[0]["target_type"] == "LINE"
        assert summary.iloc[0]["is_query_type"] == True  # noqa: E712
        assert summary.iloc[0]["n_bodyids"] == 1
        assert summary.iloc[1]["target_type"] == "Y"
        assert summary.iloc[1]["is_query_type"] == False  # noqa: E712

    def test_type_summary_written_for_type_level(self, tmp_path, monkeypatch):
        root = self._setup(tmp_path, monkeypatch)
        comparer = self._make_comparer(root, query=101, level="type")
        comparer.find_similar()
        run_dir = Path(comparer.output_folder)
        summary = pd.read_csv(run_dir / "type_summary.csv")
        assert {"target_type", "similarity", "n_bodyids", "is_intra_type"}.issubset(summary.columns)
        assert summary.iloc[0]["target_type"] == "LINE"
        assert summary.iloc[0]["is_intra_type"] == True  # noqa: E712


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
        # top types by similarity: LINE (102) then Y (103), one layer each
        assert vs.kwargs["neuron_layers"] == [[102], [103]]
        assert vs.kwargs["custom_layer_names"] == ["r1_LINE_x1", "r2_Y_x1"]
        assert vs.kwargs["legend_mode"] == "layer"
        assert vs.kwargs["skip_synapse"] is True
        assert vs.kwargs["show_fig"] is False
        assert vs.kwargs["saveas"] == morph._dataset_folder("test:v1")

    def test_type_level_excludes_intra_reference_row(self, tmp_path, monkeypatch):
        """The intra-type reference row (rank 1) must never be rendered."""
        comparer = self._setup(tmp_path, monkeypatch, query=101, level="type")
        res = comparer.find_similar()
        assert not res.empty
        assert len(self.FakeVisualizer.instances) == 1
        vs = self.FakeVisualizer.instances[0]
        # only the Y row remains after dropping the LINE intra reference;
        # its members come from the vector cache
        assert vs.kwargs["neuron_layers"] == [[103]]
        assert vs.kwargs["custom_layer_names"] == ["r1_Y_x1"]

    def test_bodyid_mode_one_layer_per_row(self, tmp_path, monkeypatch):
        comparer = self._setup(tmp_path, monkeypatch, visualize_by="bodyId")
        comparer.find_similar()
        assert len(self.FakeVisualizer.instances) == 1
        vs = self.FakeVisualizer.instances[0]
        assert vs.kwargs["neuron_layers"] == [[102], [103]]
        assert vs.kwargs["custom_layer_names"] == ["r1_LINE_102", "r2_Y_103"]
        assert vs.kwargs["legend_mode"] == "single"

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
        X, mask = cache.vectors_for([301, 302])
        assert mask.all() and np.isfinite(X).all()

    def test_vectors_for_resolves_nested(self, tmp_path):
        dataset = "fw:v1"
        folder = (tmp_path / "cache" / morph._dataset_folder(dataset)
                  / "skeletons" / "bulk_v1")
        folder.mkdir(parents=True)
        with open(folder / "301.pkl", "wb") as f:
            pickle.dump(line_neuron(), f)
        cache = morph.SkeletonVectorCache(dataset, project_root=str(tmp_path), verbose=False)
        X, mask = cache.vectors_for([301, 999])
        assert mask.tolist() == [True, False]

    def test_fetch_reuses_nested_file(self, tmp_path, monkeypatch):
        dataset = "fw:v1"
        folder = (tmp_path / "cache" / morph._dataset_folder(dataset)
                  / "skeletons" / "bulk_v1")
        folder.mkdir(parents=True)
        with open(folder / "301.pkl", "wb") as f:
            pickle.dump(line_neuron(), f)
        monkeypatch.setattr(morph, "_fetch_neuprint_skeleton",
                            lambda d, b: (_ for _ in ()).throw(AssertionError("fetcher used")))
        nrn = morph.fetch_skeleton_on_demand(dataset, 301, project_root=str(tmp_path))
        assert nrn is not None


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
    def _setup(self, tmp_path, monkeypatch, query=101):
        write_skeleton(tmp_path, "np:v1", query, line_neuron(length=20))
        write_skeleton(tmp_path, "np:v1", 201, line_neuron(length=20))  # identical
        write_skeleton(tmp_path, "np:v1", 202, bushy_y_neuron())

        comparer = morph.MorphologyComparer(
            query=query, dataset="np:v1", level="bodyid", method="vector",
            candidate_source="profile", fetch_top_n=3, top_n=5,
            output_dir=str(tmp_path / "out"), project_root=str(tmp_path), verbose=False,
        )
        monkeypatch.setattr(comparer, "_resolve_query", lambda: pd.DataFrame({
            "bodyId": [query], "type": ["T"], "instance": ["T_1"],
        }))
        return comparer

    def test_profile_first_ranks_and_bounds_fetch(self, tmp_path, monkeypatch):
        comparer = self._setup(tmp_path, monkeypatch)
        monkeypatch.setattr(comparer, "_profile_candidates", lambda q: pd.DataFrame({
            "target_bodyId": [201, 202, 203, 204, 205],
            "profile_similarity": [0.9, 0.8, 0.7, 0.6, 0.5],
        }))
        fetched = []

        def fake_fetch(dataset, bid, project_root=None, persist=True):
            fetched.append(bid)
            # transient: the pipeline must not persist fetched skeletons
            assert persist is False
            return line_neuron(length=25)

        monkeypatch.setattr(morph, "fetch_skeleton_on_demand", fake_fetch)
        res = comparer.find_similar()

        # only the top-3 candidates are considered; among them only the
        # missing skeleton (203) is fetched — the fetch is bounded by top-N
        assert res["target_bodyId"].tolist() == [201, 203, 202] or \
               res["target_bodyId"].tolist() == [201, 202, 203]
        assert 204 not in res["target_bodyId"].tolist()
        assert 205 not in res["target_bodyId"].tolist()
        assert fetched == [203]
        # no skeleton cache files were created by the transient fetches
        skel_dir = tmp_path / "cache" / morph._dataset_folder("np:v1") / "skeletons"
        assert sorted(p.name for p in skel_dir.glob("*.pkl")) == ["101.pkl", "201.pkl", "202.pkl"]
        assert "profile_similarity" in res.columns
        # identical morphology ranks first
        assert res.iloc[0]["target_bodyId"] == 201
        assert res.iloc[0]["similarity"] >= res.iloc[1]["similarity"]

    def test_profile_first_fetch_top_n_caps_candidates(self, tmp_path, monkeypatch):
        comparer = self._setup(tmp_path, monkeypatch)
        # only 201 has a skeleton; fetch_top_n=1 -> only 201 considered
        comparer.fetch_top_n = 1
        monkeypatch.setattr(comparer, "_profile_candidates", lambda q: pd.DataFrame({
            "target_bodyId": [201, 202, 203],
            "profile_similarity": [0.9, 0.8, 0.7],
        }))
        fetched = []

        def fake_fetch(dataset, bid, project_root=None, persist=True):
            fetched.append(bid)
            return line_neuron(length=25)

        monkeypatch.setattr(morph, "fetch_skeleton_on_demand", fake_fetch)
        res = comparer.find_similar()
        assert res["target_bodyId"].tolist() == [201]
        assert fetched == []  # 201's skeleton already exists

    def test_nblast_rejected_for_flywire_mesh_cache(self, tmp_path, monkeypatch):
        comparer = morph.MorphologyComparer(
            query=1, dataset="flywire_FAFB_v783", method="nblast",
            project_root=str(tmp_path), verbose=False,
        )
        monkeypatch.setattr(comparer, "_resolve_query", lambda: pd.DataFrame({
            "bodyId": [1], "type": ["T"], "instance": ["t"],
        }))
        with pytest.raises(ValueError, match="NBLAST requires neuron skeletons"):
            comparer.find_similar()

    def _type_populated(self, tmp_path):
        """Give np:v1 typed members + a vector cache (like the real datasets)."""
        write_neuron_index(tmp_path, "np:v1", [
            (101, "T", "T_1"), (201, "T", "T_2"), (202, "Y", "Y_1"),
        ])
        morph.SkeletonVectorCache(
            "np:v1", project_root=str(tmp_path), n_workers=1, verbose=False
        ).build()

    def test_profile_first_bodyid_carries_intra_columns(self, tmp_path, monkeypatch):
        comparer = self._setup(tmp_path, monkeypatch)
        self._type_populated(tmp_path)
        monkeypatch.setattr(comparer, "_profile_candidates", lambda q: pd.DataFrame({
            "target_bodyId": [201, 202],
            "profile_similarity": [0.9, 0.8],
        }))
        fetched = []
        monkeypatch.setattr(morph, "fetch_skeleton_on_demand",
                            lambda d, b, project_root=None, persist=True:
                            fetched.append(b) or line_neuron(length=25))
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
        self._type_populated(tmp_path)
        comparer.level = "type"
        # only a different-type candidate reaches the top-N: the query type's
        # intra reference must still be injected from the vector cache
        monkeypatch.setattr(comparer, "_profile_candidates", lambda q: pd.DataFrame({
            "target_bodyId": [202],
            "profile_similarity": [0.8],
        }))
        monkeypatch.setattr(morph, "fetch_skeleton_on_demand",
                            lambda d, b, project_root=None, persist=True: line_neuron(length=25))
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
        """Regression: type-level profile-first returned empty when the
        dataset had skeletons + a neuron index but NO vector cache — candidate
        types were only resolved from the vector cache, so every row was
        untyped and the type aggregation produced nothing."""
        comparer = self._setup(tmp_path, monkeypatch)
        # index + skeletons only; the vector cache parquet is never built
        # (this is the male-cns v1.0 situation)
        write_neuron_index(tmp_path, "np:v1", [
            (101, "T", "T_1"), (201, "T", "T_2"), (202, "Y", "Y_1"),
        ])
        comparer.level = "type"
        monkeypatch.setattr(comparer, "_profile_candidates", lambda q: pd.DataFrame({
            "target_bodyId": [201, 202],
            "profile_similarity": [0.9, 0.8],
        }))
        monkeypatch.setattr(morph, "fetch_skeleton_on_demand",
                            lambda d, b, project_root=None, persist=True: line_neuron(length=25))
        res = comparer.find_similar()
        assert not res.empty
        assert {"target_type", "is_intra_type", "intra_type_similarity"}.issubset(res.columns)
        intra = res[res["is_intra_type"] == True]  # noqa: E712
        assert len(intra) == 1
        assert intra.iloc[0]["target_type"] == "T"
        # intra computed from the query member's own vector (single member)
        assert intra.iloc[0]["intra_type_similarity"] == pytest.approx(1.0, abs=1e-6)
        # the Y candidate is typed through the neuron index fallback
        assert (res["target_type"] == "Y").any()

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
