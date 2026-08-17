from pathlib import Path

import pandas as pd

from neuronbridge_cache import NeuronBridgeParquetCache
import neuronbridge_finder as neuronbridge_finder_module
from neuronbridge_finder import NeuronBridgeFinder


class _DummyClient:
    version = "test-version"


def _finder(tmp_path, monkeypatch):
    monkeypatch.setattr(neuronbridge_finder_module, "NBClient", _DummyClient)
    return NeuronBridgeFinder(
        cache_folder=str(tmp_path),
        verbose=False,
        max_workers=1,
    )


def test_image_parquet_schema_and_deduplication(tmp_path):
    cache = NeuronBridgeParquetCache(tmp_path)
    frame = pd.DataFrame(
        [
            {
                "bodyId": 123,
                "score": 0.4,
                "image_id": "em-1",
                "lm_sample": "lm-1",
                "match_type": "cds",
                "dataset": "hemibrain:v1.2.1",
                "library": "FlyEM_Hemibrain_v1.2.1",
                "type": "T1",
                "instance": "I1",
                "status": "Traced",
                "dataset_folder": "discarded",
            },
            {
                "bodyId": 123,
                "score": 0.9,
                "image_id": "em-1",
                "lm_sample": "lm-1",
                "match_type": "cds",
                "dataset": "hemibrain:v1.2.1",
                "library": "FlyEM_Hemibrain_v1.2.1",
                "type": "T1",
                "instance": "I1",
                "status": "Traced",
            },
        ]
    )

    path = cache.save_image("lm-1", "cds", frame)
    assert path is not None and path.suffix == ".parquet"
    loaded = cache.load_image("lm-1", "cds")

    assert loaded is not None
    assert len(loaded) == 1
    assert loaded.iloc[0]["score"] == 0.9
    assert list(loaded.columns) == [
        "bodyId",
        "score",
        "image_id",
        "lm_sample",
        "match_type",
        "dataset",
        "library",
        "type",
        "instance",
        "status",
    ]


def test_id_parquet_reload_preserves_score_order(tmp_path):
    cache = NeuronBridgeParquetCache(tmp_path)
    cache.save_id(
        "123_cds_any",
        pd.DataFrame(
            [
                {"line": "LH-low", "library": "lib", "score": 0.2, "image_id": "lm-2", "match_type": "cds"},
                {"line": "LH-high", "library": "lib", "score": 0.9, "image_id": "lm-1", "match_type": "cds"},
            ]
        ),
    )
    loaded = cache.load_id("123_cds_any")
    assert loaded is not None
    assert loaded["line"].tolist() == ["LH-high", "LH-low"]


def test_id_key_ignores_non_id_parameters(tmp_path, monkeypatch):
    finder = _finder(tmp_path, monkeypatch)
    finder.region = "Brain"
    finder.max_api_images_per_line = 3
    first = finder._get_id_to_lines_cache_key(123, "cds", "hemibrain:v1.2.1")
    finder.region = "VNC"
    finder.max_api_images_per_line = 50
    second = finder._get_id_to_lines_cache_key(123, "cds", "hemibrain:v1.2.1")
    assert first == second == "123_cds_hemibrain:v1.2.1"

    old_cache = NeuronBridgeParquetCache(tmp_path, version="old-version")
    old_cache.save_id(
        first,
        pd.DataFrame(
            [{"line": "LH1", "library": "lib", "score": 0.8, "image_id": "lm1", "match_type": "cds"}]
        ),
    )
    assert not finder._is_cached("id_to_lines", first)


def test_both_id_results_are_reconstructed_from_algorithm_tables(tmp_path, monkeypatch):
    finder = _finder(tmp_path, monkeypatch)

    def fake_em_matches(body_id, match_type=None, expected_dataset=None, **kwargs):
        return [
            {
                "line": "LH1",
                "library": "lib",
                "score": 0.8 if match_type == "cds" else 0.7,
                "image_id": f"{match_type}-image",
                "match_type": match_type,
            }
        ]

    monkeypatch.setattr(finder, "_get_em_matches", fake_em_matches)
    combined = finder.id_to_lines(123, match_type="both", expected_dataset="ds:v1")

    assert len(combined) == 1
    assert combined.iloc[0]["match_type"] == "both"
    parquet_root = Path(tmp_path) / "parquet" / "v_test-version"
    assert not list((parquet_root / "id_to_lines").glob("*both*"))
    assert (parquet_root / "id_to_lines" / "id_to_lines_123_cds_ds_v1.parquet").exists()
    assert (parquet_root / "id_to_lines" / "id_to_lines_123_pppm_ds_v1.parquet").exists()

    monkeypatch.setattr(
        finder,
        "_get_em_matches",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    cached_combined = finder.id_to_lines(123, match_type="both", expected_dataset="ds:v1")
    assert cached_combined[["line", "combined_rank"]].equals(
        combined[["line", "combined_rank"]]
    )


def test_line_results_use_image_cache_without_line_snapshot(tmp_path, monkeypatch):
    finder = _finder(tmp_path, monkeypatch)
    monkeypatch.setattr(
        finder,
        "_get_lm_matches",
        lambda line_name, match_type=None: [
            {
                "bodyId": "123",
                "dataset": "ds:v1",
                "instance": "I1",
                "type": "T1",
                "status": "Traced",
                "score": 0.8,
                "image_id": "em-1",
                "lm_sample": "lm-1",
                "match_type": match_type,
                "library": "lib",
            },
            {
                "bodyId": "124",
                "dataset": "ds:v1",
                "instance": "I2",
                "type": "T2",
                "status": "Traced",
                "score": 0.7,
                "image_id": "em-2",
                "lm_sample": "lm-1",
                "match_type": match_type,
                "library": "lib",
            },
        ],
    )

    top_one = finder.line_to_neuron("LH1", top_n=1)
    all_rows = finder.line_to_neuron("LH1", top_n=-1)

    assert len(top_one) == 1
    assert len(all_rows) == 2
    assert not list(Path(tmp_path).glob("line_to_neuron_*.parquet"))
    assert not list(Path(tmp_path).glob("line_to_neuron_*.csv"))


def test_legacy_migration_writes_parquet_and_can_remove_sources(tmp_path, monkeypatch):
    finder = _finder(tmp_path, monkeypatch)
    image_dir = Path(tmp_path) / "image_cache"
    image_dir.mkdir(parents=True, exist_ok=True)

    line_csv = Path(tmp_path) / "line_to_neuron_LH1_cds_Brain_all.csv"
    pd.DataFrame(
        [
            {
                "bodyId": 123,
                "dataset": "ds:v1",
                "instance": "I1",
                "type": "T1",
                "status": "Traced",
                "score": 0.8,
                "image_id": "em-1",
                "lm_sample": "lm-1",
                "match_type": "cds",
                "library": "lib",
            }
        ]
    ).to_csv(line_csv, index=False)

    id_csv = Path(tmp_path) / "id_to_lines_123_cds_ds_v1_Brain_all.csv"
    pd.DataFrame(
        [
            {
                "line": "LH1",
                "library": "lib",
                "score": 0.8,
                "image_id": "lm-1",
                "match_type": "cds",
            }
        ]
    ).to_csv(id_csv, index=False)

    stats = finder.migrate_cache_to_parquet(remove_legacy=True)

    assert stats["errors"] == 0
    assert not line_csv.exists()
    assert not id_csv.exists()
    parquet_root = Path(tmp_path) / "parquet" / "v_test-version"
    assert (parquet_root / "image_cache" / "cds_lm-1.parquet").exists()
    assert (parquet_root / "id_to_lines" / "id_to_lines_123_cds_ds_v1.parquet").exists()
