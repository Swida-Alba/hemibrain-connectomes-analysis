"""Real-data end-to-end tests for the intra/inter-type similarity contract.

Runs against the real hemibrain v1.2.1 vector cache (starts from the 755
skeleton build; queries append freshly-computed vectors, so the cache only
grows) and verifies the semantics the Similar tab promises:

  * Type-level queries return the query type as the intra-type reference
    row (rank 1, ``is_intra_type=True``) with a real ``intra_type_similarity``
    value (mean pairwise similarity of the type's members).
  * Intra-type similarity > inter-type similarity for the named aMe pairs.
  * sim(aMe12, aMe10) > sim(aMe12, aMe5) and sim(aMe10, aMe12) >
    sim(aMe10, aMe5) — both as direct cache statistics and through the
    pipeline's type aggregation.
  * bodyId queries tag same-type hits (``is_same_type``) and report the
    query type's intra-type similarity; same-type hits are highly similar
    (at least as similar as the type's average member).
  * The NBLAST and profile-first paths provide the same intra-type data.

Module-level skip when the real cache / type table is absent.

Real-data findings encoded here (2026-08-08 build, cosine on z-scored
124-dim vectors):
  * aMe12 is compact: intra cohesion 0.975 > all inter types (max CL255
    0.880); bodyId queries return the other member at rank 1 (0.902, exactly
    the intra-type similarity).
  * aMe5 is cohesive for bodyId queries: 12 same-type members in the top-30,
    mean 0.909 > inter mean 0.834. Its type-level intra cohesion (0.869) is
    below the single-cell type SMP581 (0.913) — a documented exception, so
    only the named-pair and mean-inter assertions apply to aMe5.
  * aMe10 is diffuse: intra cohesion 0.894 barely edges out SMP581 (0.894);
    its off-diagonal member-pair mean (0.762) is below sim(aMe10, aMe12)
    (0.799), and same-type members do not outrank inter-type hits in bodyId
    queries (same-type mean 0.788 < top-30 inter mean 0.826). The test
    asserts the machinery + the weaker but still meaningful contract
    (same-type hits >= the type's average member similarity).

Run:  pytest tests/e2e/test_real_morphology_e2e.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import morphology as morph  # noqa: E402

DATASET = "hemibrain:v1.2.1"
FOLDER = morph._dataset_folder(DATASET)
PARQUET = PROJECT_ROOT / "cache" / FOLDER / "morphology" / "skeleton_vectors.parquet"
TYPE_CSV = PROJECT_ROOT / "datasets" / FOLDER / f"{FOLDER}_allneurons_neuron_df.csv"

AME_TYPES = ("aMe12", "aMe10", "aMe5")
BODYID_PROBES = (
    (911332304, "aMe12"),   # compact: same-type partner ranks 1st
    (1158631810, "aMe5"),   # cohesive: 12 same-type members in top-30
    (1252409535, "aMe10"),  # diffuse: same-type present but inter rows interleave
)


def _cache_usable() -> bool:
    if not (PARQUET.exists() and TYPE_CSV.exists()):
        return False
    try:
        df = pd.read_parquet(PARQUET, columns=["bodyId", "type"])
        return bool((df["type"].fillna("") != "").any())
    except Exception:
        return False


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not _cache_usable(),
        reason="hemibrain vector cache / type table missing or untyped "
               "(run SkeletonVectorCache(...).build() first)",
    ),
]


# ---------------------------------------------------------------------------
# Helpers (all statistics are recomputed from the cache, not from pipeline
# outputs, so the tests cross-check the pipeline against the raw data).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data():
    cache = morph.SkeletonVectorCache(DATASET, project_root=str(PROJECT_ROOT), verbose=False)
    d = cache.load()
    # The cache only grows: every query persists freshly-computed vectors
    # (even for transiently-fetched skeletons), so the base build size is
    # the lower bound, not an exact count.
    assert d is not None and len(d["bodyIds"]) >= 755
    return d


def member_indices(d, type_name):
    return [i for i, t in enumerate(d["types"]) if t == type_name]


def pair_mean(d, type_a, type_b, metric="cosine"):
    """Mean similarity of every a-member vs every b-member (incl. self when
    a == b)."""
    ia, ib = member_indices(d, type_a), member_indices(d, type_b)
    if not ia or not ib:
        pytest.fail(f"type missing from cache: {type_a}/{type_b}")
    m = np.vstack([
        morph.similarity_matrix(d["X"][i], d["X"][ib], metric) for i in ia
    ])
    return float(m.mean())


def off_diag_mean(d, type_name, metric="cosine"):
    """Mean pairwise similarity among a type's members, excluding self."""
    idx = member_indices(d, type_name)
    n = len(idx)
    if n < 2:
        pytest.fail(f"type {type_name} has <2 members in the cache")
    m = np.vstack([
        morph.similarity_matrix(d["X"][i], d["X"][idx], metric) for i in idx
    ])
    return float((m.sum() - n) / (n * (n - 1)))


def run_query(query, level, output_dir, method="vector", **kw):
    return morph.MorphologyComparer(
        query=query, dataset=DATASET, level=level, method=method,
        metric="cosine", output_dir=str(output_dir),
        project_root=str(PROJECT_ROOT), candidate_source="cache",
        verbose=False, **kw,
    ).find_similar()


# ---------------------------------------------------------------------------
# Type-level queries: intra-type reference row
# ---------------------------------------------------------------------------

class TestTypeLevelIntraReference:
    @pytest.mark.parametrize("type_name", AME_TYPES)
    def test_intra_reference_row_rank1(self, data, tmp_path_factory, type_name):
        out = tmp_path_factory.mktemp("e2e_out")
        res = run_query(type_name, "type", output_dir=out)
        assert not res.empty
        assert {"is_intra_type", "intra_type_similarity"}.issubset(res.columns)
        intra = res[res["is_intra_type"] == True]  # noqa: E712
        assert len(intra) == 1
        row = intra.iloc[0]
        assert row["target_type"] == type_name
        assert row["rank"] == 1
        assert row["n_bodyids"] == len(member_indices(data, type_name))
        # the intra-type similarity column is the off-diagonal pairwise mean
        assert row["intra_type_similarity"] == pytest.approx(
            off_diag_mean(data, type_name), abs=1e-6
        )

    @pytest.mark.parametrize("type_name", AME_TYPES)
    def test_intra_greater_than_inter_mean(self, data, tmp_path_factory, type_name):
        """The intra row outranks the mean of all inter-type rows (holds for
        all three types; the aMe5/SMP581 single-cell exception only breaks
        the strict max comparison, not the mean one)."""
        out = tmp_path_factory.mktemp("e2e_out")
        res = run_query(type_name, "type", output_dir=out)
        intra_sim = res.loc[res["is_intra_type"] == True, "similarity"].iloc[0]  # noqa: E712
        inter = res[res["is_intra_type"] == False]  # noqa: E712
        assert not inter.empty
        assert intra_sim > inter["similarity"].mean()

    def test_ame12_intra_is_global_max(self, data, tmp_path_factory):
        """aMe12 is compact enough that its intra row beats every inter row
        (0.975 vs 0.880); aMe10/aMe5 are not asserted here (see module
        docstring: SMP581 ties/beats their cohesion)."""
        out = tmp_path_factory.mktemp("e2e_out")
        res = run_query("aMe12", "type", output_dir=out)
        intra_sim = res.loc[res["is_intra_type"] == True, "similarity"].iloc[0]  # noqa: E712
        assert intra_sim > res.loc[res["is_intra_type"] == False, "similarity"].max()  # noqa: E712


# ---------------------------------------------------------------------------
# Named-pair relationships: aMe12-aMe10 > their similarities to aMe5
# ---------------------------------------------------------------------------

class TestNamedPairRelationships:
    def test_pair_means_satisfy_hierarchy(self, data):
        s1210 = pair_mean(data, "aMe12", "aMe10")
        s125 = pair_mean(data, "aMe12", "aMe5")
        s1012 = pair_mean(data, "aMe10", "aMe12")
        s105 = pair_mean(data, "aMe10", "aMe5")
        s512 = pair_mean(data, "aMe5", "aMe12")
        s510 = pair_mean(data, "aMe5", "aMe10")
        # symmetry of the cache statistics
        assert s1210 == pytest.approx(s1012, abs=1e-9)
        assert s125 == pytest.approx(s512, abs=1e-9)
        # the user contract: aMe12-aMe10 closer than either's tie to aMe5
        assert s1210 > s125
        assert s1012 > s105

    @pytest.mark.parametrize(
        ("query_type", "closer", "farther"),
        [
            ("aMe12", "aMe10", "aMe5"),
            ("aMe10", "aMe12", "aMe5"),
            ("aMe5", "aMe10", "aMe12"),
        ],
    )
    def test_pipeline_aggregation_preserves_hierarchy(
        self, data, tmp_path_factory, query_type, closer, farther
    ):
        """The type-level result rows must rank the closer type above the
        farther one, and both below the intra reference row."""
        out = tmp_path_factory.mktemp("e2e_out")
        res = run_query(query_type, "type", output_dir=out)
        intra_sim = res.loc[res["is_intra_type"] == True, "similarity"].iloc[0]  # noqa: E712
        close_sim = float(res.loc[res["target_type"] == closer, "similarity"].iloc[0])
        far_sim = float(res.loc[res["target_type"] == farther, "similarity"].iloc[0])
        assert intra_sim > close_sim > far_sim

    @pytest.mark.parametrize("type_name", AME_TYPES)
    def test_intra_greater_than_named_inter_pairs(self, data, type_name):
        """intra > inter for every named pair, computed directly from the
        cache (intra = mean pairwise incl. self, matching what a type query
        reports)."""
        intra = pair_mean(data, type_name, type_name)
        others = [t for t in AME_TYPES if t != type_name]
        for other in others:
            assert intra > pair_mean(data, type_name, other), (
                f"intra({type_name}) should exceed inter({type_name},{other})"
            )


# ---------------------------------------------------------------------------
# bodyId queries: same-type hits
# ---------------------------------------------------------------------------

class TestBodyIdSameType:
    @pytest.mark.parametrize(("body_id", "type_name"), BODYID_PROBES)
    def test_same_type_rows_present_and_columns(self, data, tmp_path_factory,
                                                body_id, type_name):
        out = tmp_path_factory.mktemp("e2e_out")
        res = run_query(body_id, "bodyid", output_dir=out)
        assert not res.empty
        assert {"is_same_type", "intra_type_similarity"}.issubset(res.columns)
        same = res[res["is_same_type"] == True]  # noqa: E712
        assert not same.empty, "same-type neurons must appear in the results"
        assert (same["target_type"] == type_name).all()
        # the intra column reports the query type's mean pairwise similarity
        assert same["intra_type_similarity"].iloc[0] == pytest.approx(
            off_diag_mean(data, type_name), abs=1e-6
        )
        # same-type hits are at least as similar as the type's average member
        # (tolerance for the last-ULP difference between the pipeline's
        # per-row cosine and the test's pairwise-matrix mean)
        assert same["similarity"].mean() >= same["intra_type_similarity"].iloc[0] - 1e-9

    def test_ame12_member_ranks_own_type_first(self, data, tmp_path_factory):
        """aMe12's two members are each other's top hit (0.902 = the
        intra-type similarity)."""
        out = tmp_path_factory.mktemp("e2e_out")
        res = run_query(911332304, "bodyid", output_dir=out)
        row = res.iloc[0]
        assert row["target_type"] == "aMe12"
        assert row["target_bodyId"] == 5813058431
        assert row["is_same_type"] == True  # noqa: E712
        assert row["similarity"] == pytest.approx(off_diag_mean(data, "aMe12"), abs=1e-6)

    def test_ame5_member_same_type_mean_exceeds_inter(self, data, tmp_path_factory):
        out = tmp_path_factory.mktemp("e2e_out")
        res = run_query(1158631810, "bodyid", output_dir=out)
        same = res[res["is_same_type"] == True]  # noqa: E712
        inter = res[res["is_same_type"] == False]  # noqa: E712
        # The vector cache grows with every query (freshly-computed vectors
        # are persisted), so the exact same-type count varies; the type must
        # stay clearly dominant either way.
        assert len(same) >= 8
        assert same["similarity"].mean() > inter["similarity"].mean()
        # cohesive: same-type members dominate the top ranks
        assert (same["rank"] <= 3).any()


# ---------------------------------------------------------------------------
# NBLAST and profile-first paths
# ---------------------------------------------------------------------------

class TestNblastPath:
    def test_type_level_includes_intra_row(self, data, tmp_path_factory):
        """NBLAST type queries must carry the same intra-type data. n_workers
        stays 1: navis's process pool cannot re-import the test module under
        spawn on macOS."""
        out = tmp_path_factory.mktemp("e2e_out")
        res = run_query("aMe12", "type", method="nblast",
                        output_dir=out, candidate_cap=40, n_workers=1)
        assert not res.empty
        intra = res[res["is_intra_type"] == True]  # noqa: E712
        assert len(intra) == 1
        assert intra.iloc[0]["target_type"] == "aMe12"
        assert intra.iloc[0]["rank"] == 1
        # the two aMe12 members are near-identical under NBLAST as well
        assert intra.iloc[0]["similarity"] > 0.8
        assert intra.iloc[0]["intra_type_similarity"] == pytest.approx(
            off_diag_mean(data, "aMe12"), abs=1e-6
        )


class TestProfileFirstPath:
    def test_bodyid_carries_intra_data(self, data, tmp_path_factory):
        """The default neuprint pipeline (connection-profile-first) must tag
        same-type hits and report the intra-type similarity. Network-bound:
        skipped when the live fetch fails."""
        out = tmp_path_factory.mktemp("e2e_out")
        comparer = morph.MorphologyComparer(
            query=911332304, dataset=DATASET, level="bodyid", method="vector",
            metric="cosine", output_dir=str(out),
            project_root=str(PROJECT_ROOT), candidate_source="auto",
            verbose=False,
        )
        try:
            res = comparer.find_similar()
        except Exception as exc:  # network / token issues -> graceful skip
            pytest.skip(f"profile-first run failed ({exc}); needs live NeuPrint access")
        assert not res.empty
        assert {"profile_similarity", "is_same_type", "intra_type_similarity"} \
            .issubset(res.columns)
        same = res[res["is_same_type"] == True]  # noqa: E712
        assert not same.empty
        assert same["similarity"].iloc[0] > 0.85
        assert same["intra_type_similarity"].iloc[0] == pytest.approx(
            off_diag_mean(data, "aMe12"), abs=1e-6
        )


class TestOutputsAndAutoLevel:
    def test_auto_level_follows_query_kind(self, tmp_path_factory):
        """level='auto': a type query yields type-to-type rows; a bodyId
        query yields bodyId-to-bodyId rows. results.csv is always
        bodyId-level and type_summary.csv always type-level."""
        out = tmp_path_factory.mktemp("e2e_out")
        c_type = morph.MorphologyComparer(
            query="aMe12", dataset=DATASET, level="auto", method="vector",
            metric="cosine", output_dir=str(out),
            project_root=str(PROJECT_ROOT), candidate_source="cache",
            verbose=False,
        )
        res = c_type.find_similar()
        assert "is_intra_type" in res.columns
        assert res.iloc[0]["target_type"] == "aMe12"
        assert res.iloc[0]["is_intra_type"] == True  # noqa: E712
        run_dir = Path(c_type.output_folder)
        body_rows = pd.read_csv(run_dir / "results.csv")
        assert "target_bodyId" in body_rows.columns  # bodyId-level ALWAYS
        type_rows = pd.read_csv(run_dir / "type_summary.csv")
        assert "target_type" in type_rows.columns  # type-level ALWAYS
        assert type_rows.iloc[0]["is_intra_type"] == True  # noqa: E712

        out2 = tmp_path_factory.mktemp("e2e_out")
        c_bid = morph.MorphologyComparer(
            query=911332304, dataset=DATASET, level="auto", method="vector",
            metric="cosine", output_dir=str(out2),
            project_root=str(PROJECT_ROOT), candidate_source="cache",
            verbose=False,
        )
        res2 = c_bid.find_similar()
        assert "target_bodyId" in res2.columns
        assert "is_same_type" in res2.columns
        run_dir2 = Path(c_bid.output_folder)
        assert "target_bodyId" in pd.read_csv(run_dir2 / "results.csv").columns
        assert "target_type" in pd.read_csv(run_dir2 / "type_summary.csv").columns

    def test_type_search_ranks_top_types_over_expanded_connectivity_types(
        self, tmp_path_factory
    ):
        """A type search ranks the types of the connection cache's top
        candidate_cap shared-partner neurons; every returned type must be
        within that screened candidate set."""
        out = tmp_path_factory.mktemp("e2e_out")
        comparer = morph.MorphologyComparer(
            query="aMe12", dataset=DATASET, level="type", method="vector",
            metric="cosine", output_dir=str(out),
            project_root=str(PROJECT_ROOT), candidate_source="profile",
            verbose=False,
        )
        res = comparer.find_similar()
        assert not res.empty
        assert len(res) >= 5, f"type search covered too few types: {res['target_type'].tolist()}"
        # the intra reference row is rank 1
        assert res.iloc[0]["target_type"] == "aMe12"
        assert res.iloc[0]["is_intra_type"] == True  # noqa: E712
        # every inter type is within the screened connectivity candidate
        # types (direct cross-check of the discovery step)
        qdf = comparer._resolve_query()
        cand = comparer._connection_cache_candidates(qdf)
        assert not cand.empty
        top_types = {str(t) for t in cand["target_type"] if str(t)}
        inter = res[res["is_intra_type"] == False]  # noqa: E712
        assert set(inter["target_type"]) <= top_types, \
            f"types outside the connectivity candidate set: " \
            f"{set(inter['target_type']) - top_types}"

    def test_run_saves_type_summary_like_homologs(self, tmp_path_factory):
        """Every run writes results.csv (bodyId) + type_summary.csv (type)."""
        out = tmp_path_factory.mktemp("e2e_out")
        comparer = morph.MorphologyComparer(
            query=5813058431, dataset=DATASET, level="bodyid", method="vector",
            metric="cosine", output_dir=str(out),
            project_root=str(PROJECT_ROOT), candidate_source="cache",
            verbose=False,
        )
        res = comparer.find_similar()
        assert not res.empty
        run_dir = Path(comparer.output_folder)
        assert (run_dir / "results.csv").exists()
        summary = pd.read_csv(run_dir / "type_summary.csv")
        assert {"target_type", "similarity", "n_bodyids",
                "is_intra_type"}.issubset(summary.columns)
        # the query type (aMe12) is present and flagged as the intra row
        row = summary[summary["target_type"] == "aMe12"]
        assert len(row) == 1
        assert row.iloc[0]["is_intra_type"] == True  # noqa: E712


class TestVisualization:
    def test_type_level_renders_top_types_html(self, tmp_path_factory):
        """visualize_top_n renders one layer per top found type (the intra
        reference row excluded) and produces an interactive HTML next to the
        results."""
        out = tmp_path_factory.mktemp("e2e_out")
        comparer = morph.MorphologyComparer(
            query="aMe12", dataset=DATASET, level="type", method="vector",
            metric="cosine", output_dir=str(out),
            project_root=str(PROJECT_ROOT), candidate_source="cache",
            visualize_top_n=3, visualize_by="type", verbose=False,
        )
        res = comparer.find_similar()
        assert not res.empty
        run_dir = Path(comparer.output_folder)
        htmls = list(run_dir.glob(f"plot-3d_*/*.html"))
        assert len(htmls) == 1, f"expected one plot3d html in {run_dir}"
        assert htmls[0].stat().st_size > 100_000
        content = htmls[0].read_text(errors="ignore")
        assert "plotly" in content

    def test_bodyid_mode_renders_top_rows(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("e2e_out")
        comparer = morph.MorphologyComparer(
            query=911332304, dataset=DATASET, level="bodyid", method="vector",
            metric="cosine", output_dir=str(out),
            project_root=str(PROJECT_ROOT), candidate_source="cache",
            visualize_top_n=2, visualize_by="bodyId", verbose=False,
        )
        res = comparer.find_similar()
        assert not res.empty
        htmls = list(Path(comparer.output_folder).glob(f"plot-3d_*/*.html"))
        assert len(htmls) == 1
        assert htmls[0].stat().st_size > 100_000
