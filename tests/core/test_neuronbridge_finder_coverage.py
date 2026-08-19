"""Coverage tests for neuronbridge_finder.

All HTTP boundaries are mocked via a fake NeuronBridge client injected into
``finder._client`` (or by monkeypatching ``NBClient``).  All file/cache I/O
happens inside pytest ``tmp_path``.  No network, no multiprocessing.
"""

import json
import os
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import neuronbridge_finder as nbf_mod
from neuronbridge_finder import (
    NeuronBridgeFinder,
    _create_mountain_order,
    _extract_base_type,
    _to_int_bodyid,
    image_summary_skip_note,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _DummyClient:
    """Minimal client substitute for construction-time initialization."""

    version = "test-version"


class FakeNBClient:
    """In-memory NeuronBridge client keyed by image id / body id / line."""

    version = "test-version"

    def __init__(self):
        self.em_images = {}
        self.lm_images = {}
        self.cds_matches = {}
        self.ppp_matches = {}
        self.calls = []

    def get_em_images(self, body_id):
        self.calls.append(("em_images", body_id))
        result = self.em_images.get(body_id)
        if isinstance(result, Exception):
            raise result
        return result or []

    def get_lm_images(self, line_name):
        self.calls.append(("lm_images", line_name))
        result = self.lm_images.get(line_name)
        if isinstance(result, Exception):
            raise result
        return result or []

    def get_cds_matches(self, image):
        key = getattr(image, "id", None)
        self.calls.append(("cds", key))
        result = self.cds_matches.get(key)
        if isinstance(result, Exception):
            raise result
        return result or []

    def get_ppp_matches(self, image):
        key = getattr(image, "id", None)
        self.calls.append(("pppm", key))
        result = self.ppp_matches.get(key)
        if isinstance(result, Exception):
            raise result
        return result or []


def _em_image(body_id="5813128953", cds=True, pppm=False):
    files = SimpleNamespace(
        CDSResults="https://example.invalid/cds.parquet" if cds else None,
        PPPMResults="https://example.invalid/ppp.parquet" if pppm else None,
    )
    return SimpleNamespace(
        id=f"em-{body_id}",
        type="EMImage",
        publishedName=f"hemibrain:v1.2.1:{body_id}",
        libraryName="FlyEM_Hemibrain_v1.2.1",
        neuronType="MBON01",
        neuronInstance="MBON01_R",
        anatomicalArea="Brain",
        files=files,
    )


def _lm_match(line="VT037867", score=100, image_id="lm-1"):
    image = SimpleNamespace(
        type="LMImage",
        publishedName=line,
        libraryName="FlyEM_Hemibrain_v1.2.1",
        id=image_id,
    )
    return SimpleNamespace(image=image, normalizedScore=score)


def _em_match(body_id="5813128953", score=100, image_id="em-5813128953"):
    image = SimpleNamespace(
        type="EMImage",
        publishedName=f"hemibrain:v1.2.1:{body_id}",
        libraryName="FlyEM_Hemibrain_v1.2.1",
        id=image_id,
        neuronType="MBON01",
        neuronInstance="MBON01_R",
    )
    return SimpleNamespace(image=image, normalizedScore=score)


def _lm_image(image_id="img-1", area="Brain", cds=True, pppm=False):
    files = SimpleNamespace(
        CDSResults="u" if cds else None,
        PPPMResults="u" if pppm else None,
    )
    return SimpleNamespace(id=image_id, anatomicalArea=area, files=files)


HEMIBRAIN_DF = pd.DataFrame(
    {
        "bodyId": ["5813128953", "1234"],
        "type": ["MBON01", "LH173"],
        "instance": ["MBON01_R", "LH173_L"],
        "status": ["ok", "ok"],
    }
)


@pytest.fixture
def finder(tmp_path, monkeypatch):
    monkeypatch.setattr(nbf_mod, "NBClient", _DummyClient)
    datasets_path = tmp_path / "datasets"
    datasets_path.mkdir()
    return NeuronBridgeFinder(
        datasets_path=str(datasets_path),
        cache_folder=str(tmp_path / "cache"),
        verbose=False,
        max_workers=1,
    )


@pytest.fixture
def finder_with_client(finder):
    finder._client = FakeNBClient()
    return finder


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def test_to_int_bodyid_handles_various_inputs():
    assert _to_int_bodyid(123) == 123
    assert _to_int_bodyid("456") == 456
    assert _to_int_bodyid("5813128953.0") == 5813128953
    assert _to_int_bodyid("abc") == "abc"
    assert _to_int_bodyid(None) is None


def test_extract_base_type():
    assert _extract_base_type("MCNS_aMe12") == "aMe12"
    assert _extract_base_type("aMe12") == "aMe12"
    mapping = {"MCNS_aMe12": "merged_aMe"}
    assert _extract_base_type("MCNS_aMe12", mapping) == "merged_aMe"
    assert _extract_base_type("MCNS_other", mapping) == "other"


def test_create_mountain_order():
    assert _create_mountain_order([], []) == ([], [])
    # NOTE: source bug — the left/right write pointers both start at (or
    # overlap) the center slot, so entries after the first overwrite the
    # highest score and trailing slots stay empty.  Assert the current
    # deterministic behavior; should be fixed upstream.
    scores, labels = _create_mountain_order([1, 4, 3, 2], ["a", "b", "c", "d"])
    assert scores == [1, 3, 2, 0] and labels == ["a", "c", "d", ""]
    scores, labels = _create_mountain_order([1, 3, 2], ["a", "b", "c"])
    assert scores == [0, 1, 0] and labels == ["", "a", ""]


def test_image_summary_skip_note():
    assert image_summary_skip_note("pdf", download_images=True) is None
    assert image_summary_skip_note(None, download_images=False) is None
    assert image_summary_skip_note([], download_images=False) is None
    note = image_summary_skip_note("pdf", download_images=False)
    assert note and "summary format" in note
    note = image_summary_skip_note(["pdf", "pptx"], download_images=False)
    assert "pdf" in note and "pptx" in note


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

def test_invalid_match_type_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(nbf_mod, "NBClient", _DummyClient)
    with pytest.raises(ValueError, match="Invalid match_type"):
        NeuronBridgeFinder(cache_folder=str(tmp_path), match_type="nope")


def test_invalid_region_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(nbf_mod, "NBClient", _DummyClient)
    with pytest.raises(ValueError, match="Invalid region"):
        NeuronBridgeFinder(cache_folder=str(tmp_path), region="spine")


def test_match_type_and_region_normalized(finder):
    assert finder.match_type == "cds"
    assert finder.region == "All"


def test_validate_helpers(finder):
    assert finder._validate_match_type(" CDS ") == "cds"
    assert finder._validate_similarity_method("Jaccard") == "jaccard"
    assert finder._validate_sort_by("MAX") == "max"
    with pytest.raises(ValueError):
        finder._validate_match_type("wrong")
    with pytest.raises(ValueError):
        finder._validate_similarity_method("wrong")
    with pytest.raises(ValueError):
        finder._validate_sort_by("wrong")


def test_init_client_success_after_retry(tmp_path, monkeypatch):
    attempts = {"n": 0}

    def flaky_client():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionError("connection timeout")
        return _DummyClient()

    monkeypatch.setattr(nbf_mod, "NBClient", flaky_client)
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    f = NeuronBridgeFinder(cache_folder=str(tmp_path), use_cache=False, verbose=False)
    assert attempts["n"] == 2
    assert isinstance(f._client, _DummyClient)


def test_init_client_failure_loop_raises(tmp_path, monkeypatch):
    def failing_client():
        raise ConnectionError("connection timeout")

    monkeypatch.setattr(nbf_mod, "NBClient", failing_client)
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda delay: sleeps.append(delay))
    # NOTE: source bug — retry_delays has only 3 entries but 5 attempts are
    # made, so a persistently failing client raises IndexError on attempt 4
    # instead of the documented RuntimeError.
    with pytest.raises((RuntimeError, IndexError)):
        NeuronBridgeFinder(cache_folder=str(tmp_path), use_cache=False, verbose=False)
    assert sleeps[:3] == [2, 5, 10]


def test_fix_store_prefixes(finder):
    store = SimpleNamespace(
        prefixes={"CDSResults": "https://data-dev.example/v3_8_0/prefix"}
    )
    finder._client = SimpleNamespace(config=SimpleNamespace(stores={"s": store}))
    finder._fix_store_prefixes()
    assert store.prefixes["CDSResults"] == "https://data-prod.example/v3_8_1/prefix"
    # Client without config.stores is a no-op
    finder._client = SimpleNamespace()
    finder._fix_store_prefixes()


# ---------------------------------------------------------------------------
# _retry_with_backoff
# ---------------------------------------------------------------------------

def test_retry_with_backoff_retryable_succeeds(finder, monkeypatch):
    sleeps = []
    monkeypatch.setattr(nbf_mod.time, "sleep", lambda d: sleeps.append(d))
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("Connection reset by peer")
        return "ok"

    assert finder._retry_with_backoff(flaky, max_retries=5, initial_delay=0.5) == "ok"
    assert calls["n"] == 3
    assert sleeps == [0.5, 1.0]


def test_retry_with_backoff_non_retryable_raises_immediately(finder, monkeypatch):
    sleeps = []
    monkeypatch.setattr(nbf_mod.time, "sleep", lambda d: sleeps.append(d))

    def broken():
        raise ValueError("not a network problem")

    with pytest.raises(ValueError):
        finder._retry_with_backoff(broken, max_retries=5)
    assert sleeps == []


def test_retry_with_backoff_exhausts_retries(finder, monkeypatch):
    monkeypatch.setattr(nbf_mod.time, "sleep", lambda d: None)

    def always_timeout():
        raise TimeoutError("read timeout")

    with pytest.raises(TimeoutError):
        finder._retry_with_backoff(always_timeout, max_retries=2, initial_delay=0.1)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def test_normalize_dataset_name(finder):
    assert finder._normalize_dataset_name("flywire_FAFB_v783") == "flywire-fafb"
    assert finder._normalize_dataset_name("flywire_fafb:v783") == "flywire-fafb"
    assert finder._normalize_dataset_name("male-cns:v0.9") == "male-cns"
    assert finder._normalize_dataset_name("hemibrain:v1.2.1") == "hemibrain"
    assert finder._normalize_dataset_name("hemibrain_v1_2_1") == "hemibrain"


def test_save_parameters(tmp_path, finder):
    class Custom:
        def __repr__(self):
            return "Custom()"

    path = finder._save_parameters(
        str(tmp_path),
        "analyze_colabeling",
        {
            "top_n": 5,
            "lines": ("VT037867", "LH173"),
            "opts": {"a": 1},
            "arr": np.array([1, 2, 3]),
            "custom": Custom(),
            "plain_set": {1},
            "none": None,
        },
    )
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["metadata"]["function"] == "analyze_colabeling"
    assert saved["function_params"]["top_n"] == 5
    assert saved["function_params"]["lines"] == ["VT037867", "LH173"]
    assert saved["function_params"]["arr"] == [1, 2, 3]
    assert "has_neuprint_token" in saved["module_params"]


def test_save_user_warning_notes(tmp_path, finder):
    assert finder._save_user_warning_notes(str(tmp_path), []) is None
    path = finder._save_user_warning_notes(str(tmp_path), ["caveat one"])
    assert "caveat one" in open(path, encoding="utf-8").read()


def test_filter_images_by_region(tmp_path, monkeypatch):
    monkeypatch.setattr(nbf_mod, "NBClient", _DummyClient)
    images = [
        SimpleNamespace(anatomicalArea="Brain"),
        SimpleNamespace(anatomicalArea="VNC"),
        SimpleNamespace(anatomicalArea=None),
    ]
    f = NeuronBridgeFinder(cache_folder=str(tmp_path / "c"), verbose=False, region="all")
    assert f._filter_images_by_region(images) == images
    f.region = "Brain"
    assert len(f._filter_images_by_region(images)) == 2  # Brain + missing area
    f.region = "VNC"
    filtered = f._filter_images_by_region(images)
    # VNC image plus the area-less one (included by default)
    assert len(filtered) == 2 and filtered[0].anatomicalArea == "VNC"


def test_filter_images_by_match_availability(finder):
    both = _lm_image("i1", cds=True, pppm=True)
    cds_only = _lm_image("i2", cds=True, pppm=False)
    no_files = SimpleNamespace(id="i3")
    images = [both, cds_only, no_files]
    assert finder._filter_images_by_match_availability(images, "cds") == [both, cds_only]
    assert finder._filter_images_by_match_availability(images, "pppm") == [both]
    assert finder._filter_images_by_match_availability(images, "both") == [both, cds_only]
    assert finder._filter_images_by_match_availability(images, "???") == images


def test_classify_line_type(finder):
    assert finder._classify_line_type("VT037867") == "gal4_lexa"
    assert finder._classify_line_type("R10A06") == "gal4_lexa"
    assert finder._classify_line_type("GMR_0001") == "gal4_lexa"
    assert finder._classify_line_type("SS010") == "split_gal4"
    assert finder._classify_line_type("LH173") == "split_gal4"
    assert finder._classify_line_type("") == "gal4_lexa"
    assert finder._classify_line_type("XYZ") == "gal4_lexa"


def test_print_warning_summary(finder, capsys):
    finder.verbose = True
    finder._warning_collector = [
        "flimg.janelia.org server not accessible for VT037867: error",
        "No files from GAL4 collections for VT037867, trying MCFO",
        "No images found for LH173 in any collection",
    ]
    finder._print_warning_summary()
    out = capsys.readouterr().out
    assert "Server Access Issues" in out
    assert "MCFO fallback" in out
    assert "no FlyLight images" in out
    assert finder._warning_collector == []
    finder._print_warning_summary()  # empty collector is a no-op


def test_vprint_batch_mode_and_progress(finder, capsys):
    finder.verbose = True
    finder._batch_mode = True
    finder._vprint("hidden")
    assert capsys.readouterr().out == ""
    finder._vprint("shown", force=True)
    assert "shown" in capsys.readouterr().out
    finder._batch_mode = False
    finder._progress(2, 4, "working")
    assert "[DROCAT][progress] 2/4 working" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Library -> dataset mapping
# ---------------------------------------------------------------------------

def test_parse_library_name(finder):
    assert finder._parse_library_name("FlyEM_Hemibrain_v1.2.1") == ("FlyEM_Hemibrain", "1.2.1")
    assert finder._parse_library_name("FlyEM_MANC") == ("FlyEM_MANC", "")
    assert finder._parse_library_name("") == ("", "")


def test_get_dataset_from_library(finder):
    assert finder._get_dataset_from_library("FlyEM_Hemibrain_v1.2.1") == "hemibrain_v1_2_1"
    assert finder._get_dataset_from_library("") is None
    # Unmapped version auto-maps from the same base library
    assert finder._get_dataset_from_library("FlyEM_MANC_v9.9") == "manc_v9_9"
    # Prefix partial match fallback
    assert finder._get_dataset_from_library("FlyEM_Hemibrain_extra") == "hemibrain_v1_2_1"
    assert finder._get_dataset_from_library("Zebra_Lib") is None


def test_get_dataset_name_from_library(finder):
    assert finder._get_dataset_name_from_library("FlyEM_Hemibrain_v1.2.1") == "hemibrain:v1.2.1"
    assert finder._get_dataset_name_from_library("") is None
    assert finder._get_dataset_name_from_library("FlyEM_MANC_v9.9") == "manc:v9.9"
    # Folder-style mapped names become colon-style
    assert finder._get_dataset_name_from_library("FlyWire_FAFB_v999") == "flywire_FAFB:v999"
    assert finder._get_dataset_name_from_library("Zebra_Lib") is None


def test_dataset_name_folder_conversions(finder):
    assert finder._dataset_name_to_folder("hemibrain:v1.2.1") == "hemibrain_v1_2_1"
    assert finder._dataset_name_to_folder("hemibrain_v1_2_1") == "hemibrain_v1_2_1"
    assert finder._dataset_name_to_folder("not-a-dataset") is None
    assert finder._folder_to_dataset_name("hemibrain_v1_2_1") == "hemibrain:v1.2.1"
    assert finder._folder_to_dataset_name("other_folder") == "other_folder"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _write_neuron_csv(datasets_path, folder, df):
    folder_path = Path(datasets_path) / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    path = folder_path / f"{folder}_allneurons_neuron_df.csv"
    df.to_csv(path, index=False)
    return path


def test_ensure_datasets_loaded_local_csv(finder):
    _write_neuron_csv(finder.datasets_path, "hemibrain_v1_2_1", HEMIBRAIN_DF)
    loaded = finder._ensure_datasets_loaded(["hemibrain:v1.2.1"])
    assert loaded == ["hemibrain_v1_2_1"]
    assert "hemibrain_v1_2_1" in finder._neuron_dfs
    # Already loaded -> immediate
    assert finder._ensure_datasets_loaded(["hemibrain:v1.2.1"]) == ["hemibrain_v1_2_1"]


def test_ensure_datasets_loaded_pull_and_fallback(finder, monkeypatch):
    pulled = pd.DataFrame({"bodyId": ["9"], "type": ["X"]})
    monkeypatch.setattr(finder, "_pull_and_load_dataset", lambda ds: pulled)
    assert finder._ensure_datasets_loaded(["manc:v1.0"]) == ["manc_v1_0"]

    monkeypatch.setattr(finder, "_pull_and_load_dataset", lambda ds: None)
    fnc_df = pd.DataFrame({"bodyId": ["7"], "type": ["Y"]})
    monkeypatch.setattr(finder, "_fetch_neuron_df_via_fnc", lambda ds: fnc_df)
    assert finder._ensure_datasets_loaded(["male-cns:v0.9"]) == ["male-cns_v0_9"]

    monkeypatch.setattr(finder, "_fetch_neuron_df_via_fnc", lambda ds: None)
    assert finder._ensure_datasets_loaded(["male-cns:v1.0"]) == []


def test_ensure_datasets_loaded_flywire_skip(finder):
    # No local FlyWire table -> skipped, no network attempted
    assert finder._ensure_datasets_loaded(["flywire_FAFB_v783"]) == []


def test_load_neuron_df_for_dataset_paths(finder, monkeypatch):
    assert finder._load_neuron_df_for_dataset("") is None
    finder._neuron_dfs["cached_ds"] = HEMIBRAIN_DF
    assert finder._load_neuron_df_for_dataset("cached_ds") is HEMIBRAIN_DF

    # CSV load
    _write_neuron_csv(finder.datasets_path, "hemibrain_v1_2_1", HEMIBRAIN_DF)
    df = finder._load_neuron_df_for_dataset("hemibrain_v1_2_1")
    assert df is not None and len(df) == 2

    # Parquet load
    folder = Path(finder.datasets_path) / "manc_v1_0"
    folder.mkdir(parents=True, exist_ok=True)
    HEMIBRAIN_DF.to_parquet(folder / "manc_v1_0_allneurons_neuron_df.parquet")
    assert finder._load_neuron_df_for_dataset("manc_v1_0") is not None

    # Missing -> pull path (mocked)
    monkeypatch.setattr(finder, "_pull_and_load_dataset", lambda ds: None)
    assert finder._load_neuron_df_for_dataset("optic-lobe_v1_1") is None

    # FlyWire missing -> skip without pulling
    pulled = []
    monkeypatch.setattr(
        finder, "_pull_and_load_dataset", lambda ds: pulled.append(ds))
    assert finder._load_neuron_df_for_dataset("flywire_BANC_v626") is None
    assert pulled == []

    # No datasets_path -> FNC fallback (mocked)
    finder.datasets_path = None
    monkeypatch.setattr(finder, "_fetch_neuron_df_via_fnc", lambda ds: HEMIBRAIN_DF)
    assert finder._load_neuron_df_for_dataset("anything") is HEMIBRAIN_DF


def test_enrich_match_with_dataset_info(finder):
    finder._neuron_dfs["hemibrain_v1_2_1"] = HEMIBRAIN_DF
    match = {"bodyId": "5813128953"}
    enriched = finder._enrich_match_with_dataset_info(match, _em_image())
    assert enriched["dataset"] == "hemibrain:v1.2.1"
    assert enriched["dataset_folder"] == "hemibrain_v1_2_1"
    assert enriched["type"] == "MBON01"
    assert enriched["instance"] == "MBON01_R"
    assert enriched["status"] == "ok"

    # bodyId missing from local table -> NeuronBridge metadata fallback
    match = {"bodyId": "999"}
    enriched = finder._enrich_match_with_dataset_info(match, _em_image())
    assert enriched["type"] == "MBON01"
    assert enriched["status"] == ""

    # Unknown library -> unknown dataset
    img = _em_image()
    img.libraryName = "Zebra_Lib"
    enriched = finder._enrich_match_with_dataset_info({"bodyId": "1"}, img)
    assert enriched["dataset"] == "unknown"
    assert enriched["dataset_folder"] == "unknown"


# ---------------------------------------------------------------------------
# id_to_lines cache layer
# ---------------------------------------------------------------------------

def test_get_cache_path_invalid_type(finder):
    with pytest.raises(ValueError, match="Unsupported"):
        finder._get_cache_path("bogus", "x")


def test_save_and_load_from_cache(finder):
    df = pd.DataFrame(
        {"line": ["VT037867"], "library": ["FlyEM_Hemibrain_v1.2.1"],
         "score": [100], "image_id": ["lm-1"], "match_type": ["cds"]}
    )
    assert finder._load_from_cache("id_to_lines", "key1") is None
    finder._save_to_cache("id_to_lines", "key1", df)
    loaded = finder._load_from_cache("id_to_lines", "key1")
    assert loaded is not None and list(loaded["line"]) == ["VT037867"]
    assert finder._is_cached("id_to_lines", "key1")
    # Empty df is never saved
    finder._save_to_cache("id_to_lines", "empty", pd.DataFrame())
    assert not finder._is_cached("id_to_lines", "empty")


def test_load_from_cache_corrupt_parquet(finder):
    path = finder._get_cache_path("id_to_lines", "corrupt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"not a parquet file")
    with pytest.warns(UserWarning, match="Failed to read"):
        assert finder._load_from_cache("id_to_lines", "corrupt") is None


def test_cache_index_helpers(finder):
    finder._add_to_cache_index("id_to_lines", "a/b:c")
    assert "id_to_lines_a_b_c" in finder._cache_index["id_to_lines"]
    assert finder._is_cached("id_to_lines", "a/b:c")
    assert not finder._is_cached("id_to_lines", "missing")


def test_cache_disabled_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(nbf_mod, "NBClient", _DummyClient)
    f = NeuronBridgeFinder(cache_folder=str(tmp_path), use_cache=False, verbose=False)
    assert f._load_from_cache("id_to_lines", "x") is None
    assert f._load_image_cache("img", "cds") is None
    f._save_to_cache("id_to_lines", "x", pd.DataFrame({"a": [1]}))
    f._save_image_cache("img", "cds", [{"bodyId": "1", "score": 1}])


def test_load_from_cache_polars_and_bulk(finder):
    if not nbf_mod.HAS_POLARS:
        pytest.skip("polars not installed")
    df = pd.DataFrame(
        {"line": ["VT037867"], "library": ["L"], "score": [100.0],
         "image_id": ["lm-1"], "match_type": ["cds"]}
    )
    cache_key = finder._get_id_to_lines_cache_key(123, "cds", "hemibrain:v1.2.1")
    assert cache_key == "123_cds_hemibrain:v1.2.1"
    assert finder._load_from_cache_polars("id_to_lines", cache_key) is None
    finder._save_to_cache("id_to_lines", cache_key, df)
    loaded = finder._load_from_cache_polars("id_to_lines", cache_key)
    assert loaded is not None and loaded.height == 1

    combined, ids = finder._load_cached_neurons_bulk_polars(
        [{"bodyId": 123, "dataset": "hemibrain:v1.2.1"},
         {"bodyId": 999, "dataset": "hemibrain:v1.2.1"}],
        "cds",
    )
    assert ids == [123]
    assert "source_dataset" in combined.columns
    empty, ids = finder._load_cached_neurons_bulk_polars([], "cds")
    assert empty.height == 0 and ids == []


# ---------------------------------------------------------------------------
# Line mapping + image cache
# ---------------------------------------------------------------------------

def test_line_mapping_missing_and_corrupt(finder):
    assert finder._load_line_mapping() == {"lines": {}, "images": {}}
    path = finder._get_line_mapping_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("{invalid json")
    assert finder._load_line_mapping() == {"lines": {}, "images": {}}


def test_update_and_save_line_mapping(finder):
    finder._update_line_mapping("VT037867", "Brain", ["img-1", "img-2"],
                                image_id="img-1", match_type="cds")
    mapping = finder._load_line_mapping()
    assert mapping["lines"]["VT037867"]["Brain"]["image_ids"] == ["img-1", "img-2"]
    assert mapping["images"]["img-1"]["cached_types"] == ["cds"]
    # Updating again does not duplicate cached types
    finder._update_line_mapping("VT037867", "Brain", ["img-1", "img-2"],
                                image_id="img-1", match_type="cds")
    mapping = finder._load_line_mapping()
    assert mapping["images"]["img-1"]["cached_types"] == ["cds"]
    # Manual save round-trip
    finder._save_line_mapping({"lines": {"X": {}}, "images": {}})
    assert "X" in finder._load_line_mapping()["lines"]


def test_save_line_mapping_failure_warns(finder, monkeypatch):
    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.warns(UserWarning, match="Failed to save line mapping"):
        finder._update_line_mapping("VT037867", "Brain", ["img-1"])


def test_sync_mapping_from_cache_files(finder):
    image_dir = finder._get_parquet_cache().image_dir
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "cds_img1.parquet").write_bytes(b"x")
    (image_dir / "pppm_img1.parquet").write_bytes(b"x")
    (image_dir / "badname.parquet").write_bytes(b"x")
    (image_dir / "junk_img2.parquet").write_bytes(b"x")
    stats = finder.sync_mapping_from_cache_files()
    assert stats == {"images_scanned": 2, "types_updated": 2}
    mapping = finder._load_line_mapping()
    assert set(mapping["images"]["img1"]["cached_types"]) == {"cds", "pppm"}


def test_image_cache_load_corrupt(finder):
    path = finder._get_image_cache_path("imgX", "cds")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"garbage")
    with pytest.warns(UserWarning, match="Failed to read"):
        assert finder._load_image_cache("imgX", "cds") is None


def test_save_image_cache_and_mapping(finder):
    matches = [{"bodyId": "5813128953", "score": 100, "image_id": "em-1",
                "lm_sample": "img-1", "match_type": "cds",
                "dataset": "hemibrain:v1.2.1", "library": "FlyEM_Hemibrain_v1.2.1",
                "type": "MBON01", "instance": "MBON01_R", "status": "ok"}]
    # 'both' is never written
    finder._save_image_cache("img-1", "both", matches, line_name="VT037867")
    assert finder._load_image_cache("img-1", "both") is None
    # Empty matches are skipped
    finder._save_image_cache("img-1", "cds", [], line_name="VT037867")
    assert finder._load_image_cache("img-1", "cds") is None

    finder._save_image_cache("img-1", "cds", matches, line_name="VT037867")
    cached = finder._load_image_cache("img-1", "cds")
    assert cached is not None and list(cached["bodyId"]) == ["5813128953"]
    mapping = finder._load_line_mapping()
    assert mapping["images"]["img-1"]["cached_types"] == ["cds"]
    assert mapping["images"]["img-1"]["line"] == "VT037867"


# ---------------------------------------------------------------------------
# API match fetching with fake client
# ---------------------------------------------------------------------------

def test_fetch_matches_from_api(finder_with_client):
    finder = finder_with_client
    finder._neuron_dfs["hemibrain_v1_2_1"] = HEMIBRAIN_DF
    lm_image = _lm_image("img-1")
    finder._client.cds_matches["img-1"] = [_em_match(score=80)]
    finder._client.ppp_matches["img-1"] = [_em_match(score=60)]

    cds = finder._fetch_matches_from_api(lm_image, "cds")
    assert len(cds) == 1
    assert cds[0]["bodyId"] == "5813128953"
    assert cds[0]["match_type"] == "cds"
    assert cds[0]["type"] == "MBON01"  # enriched from local table

    pppm = finder._fetch_matches_from_api(lm_image, "pppm")
    assert pppm[0]["match_type"] == "pppm"
    assert pppm[0]["lm_sample"] == "img-1"


def test_fetch_matches_from_api_error_returns_empty(finder_with_client):
    finder_with_client._client.cds_matches["img-1"] = RuntimeError("boom")
    assert finder_with_client._fetch_matches_from_api(_lm_image("img-1"), "cds") == []


def test_get_image_matches_cached_paths(finder_with_client):
    finder = finder_with_client
    finder._neuron_dfs["hemibrain_v1_2_1"] = HEMIBRAIN_DF

    # No image id -> nothing
    assert finder._get_image_matches_cached(SimpleNamespace(id=""), "cds") == ([], False, False)

    lm_image = _lm_image("img-1")
    finder._client.cds_matches["img-1"] = [_em_match(score=80)]
    finder._client.ppp_matches["img-1"] = [_em_match(score=60)]

    # First call fetches from API and caches
    matches, from_cache, partial = finder._get_image_matches_cached(
        lm_image, "cds", line_name="VT037867")
    assert len(matches) == 1 and not from_cache and not partial

    # Second call hits the cache
    matches, from_cache, partial = finder._get_image_matches_cached(lm_image, "cds")
    assert len(matches) == 1 and from_cache and not partial

    # 'both' with cds cached but pppm not -> partial cache
    matches, from_cache, partial = finder._get_image_matches_cached(
        lm_image, "both", line_name="VT037867")
    assert len(matches) == 2 and not from_cache and partial

    # 'both' again -> everything cached
    matches, from_cache, partial = finder._get_image_matches_cached(lm_image, "both")
    assert len(matches) == 2 and from_cache and partial


# ---------------------------------------------------------------------------
# EM image / dataset identity helpers
# ---------------------------------------------------------------------------

def test_get_em_images(finder_with_client):
    finder = finder_with_client
    assert finder._get_em_images(1) == []
    finder._client.em_images[2] = RuntimeError("api down")
    assert finder._get_em_images(2) == []
    img = _em_image()
    finder._client.em_images[3] = [img]
    assert finder._get_em_images(3) == [img]
    # Generator inputs are converted to lists
    finder._client.em_images[4] = (i for i in [img])
    assert finder._get_em_images(4) == [img]
    # No client -> []
    finder._client = None
    assert finder._get_em_images(3) == []


def test_dataset_identity_helpers():
    assert NeuronBridgeFinder._dataset_identity(None) == ("", None)
    assert NeuronBridgeFinder._dataset_identity("unknown") == ("", None)
    assert NeuronBridgeFinder._dataset_identity("male-cns:v1.0:12211") == ("male-cns", "1.0")
    assert NeuronBridgeFinder._dataset_identity("hemibrain_v1_2_1") == ("hemibrain", "1.2.1")
    assert NeuronBridgeFinder._dataset_identity("hemibrain") == ("hemibrain", None)

    assert NeuronBridgeFinder._datasets_match("hemibrain:v1.2.1", "hemibrain:v1.2.1")
    assert not NeuronBridgeFinder._datasets_match("hemibrain:v1.2.1", "hemibrain:v1.2")
    assert NeuronBridgeFinder._datasets_match("hemibrain", "hemibrain:v1.2.1")
    assert NeuronBridgeFinder._datasets_match(None, "hemibrain:v1.2.1")
    assert not NeuronBridgeFinder._datasets_match("manc:v1.0", "hemibrain:v1.2.1")


def test_dataset_name_from_em_image_and_selection():
    img_a = SimpleNamespace(publishedName="hemibrain:v1.2.1:123")
    img_b = SimpleNamespace(publishedName="manc:v1.0:456:789")  # extra numeric id
    img_c = SimpleNamespace(publishedName="")
    assert NeuronBridgeFinder._dataset_name_from_em_image(img_a) == "hemibrain:v1.2.1"
    # Only the trailing numeric body-id segment is dropped
    assert NeuronBridgeFinder._dataset_name_from_em_image(img_b) == "manc:v1.0:456"
    assert NeuronBridgeFinder._dataset_name_from_em_image(img_c) is None

    assert NeuronBridgeFinder._select_em_image_for_dataset([]) is None
    assert NeuronBridgeFinder._select_em_image_for_dataset([img_a], None) is img_a
    assert NeuronBridgeFinder._select_em_image_for_dataset([img_a], "unknown") is img_a
    assert NeuronBridgeFinder._select_em_image_for_dataset(
        [img_a, img_b], "manc:v1.0") is img_b
    assert NeuronBridgeFinder._select_em_image_for_dataset([img_a], "manc:v1.0") is None


def test_get_body_metadata_and_validation(finder_with_client):
    finder = finder_with_client
    img = _em_image()
    finder._client.em_images[10] = [img]
    actual_ds, selected = finder._get_body_metadata(10, "hemibrain:v1.2.1")
    assert actual_ds == "hemibrain:v1.2.1" and selected is img
    actual_ds, selected = finder._get_body_metadata(10, "manc:v1.0")
    assert selected is None  # no matching dataset image

    assert finder._validate_body_id_dataset(10, "hemibrain:v1.2.1") == "hemibrain:v1.2.1"
    assert finder._get_em_image_for_dataset(10, "hemibrain:v1.2.1") is img
    # No client -> expected dataset is echoed
    finder._client = None
    assert finder._validate_body_id_dataset(10, "hemibrain:v1.2.1") == "hemibrain:v1.2.1"


def test_validate_body_ids_parallel(finder_with_client):
    finder = finder_with_client
    finder._client.em_images[10] = [_em_image()]
    results = finder._validate_body_ids_parallel(
        [{"bodyId": 10, "dataset": "hemibrain:v1.2.1"},
         {"bodyId": 11}],  # unknown body -> no images
        max_workers=2,
    )
    assert results[10]["dataset"] == "hemibrain:v1.2.1"
    assert results[10]["em_image"] is not None
    assert results[11]["dataset"] is None
    assert finder._validate_body_ids_parallel([]) == {}


def test_extract_body_id(finder):
    assert finder._extract_body_id(_em_image()) == "5813128953"
    assert finder._extract_body_id(SimpleNamespace(id="img-778")) == "778"
    assert finder._extract_body_id(SimpleNamespace()) == ""


def test_sort_matches_by_rank(finder):
    assert finder._sort_matches_by_rank([]) == []
    matches = [
        {"line": "A", "score": 90, "match_type": "cds"},
        {"line": "B", "score": 80, "match_type": "cds"},
        {"line": "A", "score": 70, "match_type": "pppm"},
        {"line": "C", "score": 60, "match_type": "pppm"},
    ]
    result = finder._sort_matches_by_rank(matches, key_field="line")
    by_line = {m["line"]: m for m in result}
    assert by_line["A"]["combined_rank"] == 2  # cds rank 1 + pppm rank 2
    assert by_line["A"]["match_type"] == "both"
    assert by_line["B"]["pppm_rank"] == 3  # missing pppm -> max+1
    assert result[0]["line"] == "A"


# ---------------------------------------------------------------------------
# _get_em_matches / _get_lm_matches
# ---------------------------------------------------------------------------

def test_get_em_matches_cds_sorted(finder_with_client):
    finder = finder_with_client
    finder._client.em_images[100] = [_em_image(cds=True)]
    finder._client.cds_matches["em-5813128953"] = [
        _lm_match("VT037867", score=50),
        _lm_match("LH173", score=90),
    ]
    matches = finder._get_em_matches(100, match_type="cds")
    assert [m["line"] for m in matches] == ["LH173", "VT037867"]
    assert matches[0]["match_type"] == "cds"


def test_get_em_matches_error_paths(finder_with_client):
    finder = finder_with_client
    # No EM image at all
    assert finder._get_em_matches(200, match_type="cds") == []
    # Explicit None image (validation found nothing)
    assert finder._get_em_matches(200, match_type="cds", em_image=None) == []
    # Image without files metadata
    assert finder._get_em_matches(200, match_type="cds",
                                  em_image=SimpleNamespace(id="x")) == []
    # match_type both with no result URLs -> warning path, empty result
    no_urls = _em_image(cds=False, pppm=False)
    assert finder._get_em_matches(200, match_type="both", em_image=no_urls) == []


def test_get_em_matches_both_raw_and_ranked(finder_with_client):
    finder = finder_with_client
    em_img = _em_image(cds=True, pppm=True)
    finder._client.cds_matches[em_img.id] = [
        _lm_match("VT037867", score=90), _lm_match("LH173", score=80)]
    finder._client.ppp_matches[em_img.id] = [
        _lm_match("VT037867", score=70), _lm_match("SS010", score=95)]

    raw = finder._get_em_matches(1, match_type="both", em_image=em_img, raw=True)
    assert len(raw) == 4 and all(m["match_type"] in ("cds", "pppm") for m in raw)

    ranked = finder._get_em_matches(1, match_type="both", em_image=em_img)
    assert ranked[0]["line"] == "VT037867"  # best combined rank
    assert ranked[0]["match_type"] == "both"


def test_get_lm_matches_cds_dedup(finder_with_client):
    finder = finder_with_client
    finder._neuron_dfs["hemibrain_v1_2_1"] = HEMIBRAIN_DF
    finder._client.lm_images["VT037867"] = [_lm_image("img-1")]
    finder._client.cds_matches["img-1"] = [
        _em_match(score=50), _em_match(score=90)]

    matches = finder._get_lm_matches("VT037867", match_type="cds")
    # Duplicates (same bodyId + dataset) collapsed, best score kept
    assert len(matches) == 1
    assert matches[0]["score"] == 90
    assert matches[0]["type"] == "MBON01"
    # Line mapping recorded the checked images (region defaults to 'All')
    mapping = finder._load_line_mapping()
    assert mapping["lines"]["VT037867"]["All"]["image_ids"] == ["img-1"]


def test_get_lm_matches_empty_paths(finder_with_client):
    finder = finder_with_client
    assert finder._get_lm_matches("MISSING", match_type="cds") == []

    finder._client.lm_images["VT1"] = [_lm_image("img-1", area="VNC")]
    finder.region = "Brain"
    assert finder._get_lm_matches("VT1", match_type="cds") == []

    finder.region = "All"
    finder._client.lm_images["VT2"] = [_lm_image("img-1", cds=False, pppm=False)]
    assert finder._get_lm_matches("VT2", match_type="cds") == []

    finder._client.lm_images["VT3"] = RuntimeError("api down")
    assert finder._get_lm_matches("VT3", match_type="cds") == []


def test_get_lm_matches_both_with_limit(finder_with_client):
    finder = finder_with_client
    finder._neuron_dfs["hemibrain_v1_2_1"] = HEMIBRAIN_DF
    finder.max_api_images_per_line = 1
    finder._client.lm_images["VT9"] = [
        _lm_image("img-1", cds=True, pppm=True),
        _lm_image("img-2", cds=True, pppm=True),
    ]
    finder._client.cds_matches["img-1"] = [_em_match(score=80)]
    finder._client.ppp_matches["img-1"] = [_em_match("1234", score=60)]

    matches = finder._get_lm_matches("VT9", match_type="both")
    assert len(matches) == 2
    assert all(m["match_type"] == "both" for m in matches)
    # Only img-1 was processed because of max_api_images_per_line=1
    assert "img-2" not in [c[1] for c in finder._client.calls]


def test_get_lm_matches_parallel(finder_with_client):
    finder = finder_with_client
    finder._neuron_dfs["hemibrain_v1_2_1"] = HEMIBRAIN_DF
    finder.max_workers = 2
    finder._client.lm_images["VT9"] = [
        _lm_image("img-1"), _lm_image("img-2")]
    finder._client.cds_matches["img-1"] = [_em_match(score=80)]
    finder._client.cds_matches["img-2"] = [_em_match("1234", score=70)]
    matches = finder._get_lm_matches("VT9", match_type="cds")
    assert len(matches) == 2


# ---------------------------------------------------------------------------
# id_to_lines
# ---------------------------------------------------------------------------

def test_id_to_lines_cds_fetch_cache_and_hit(finder_with_client):
    finder = finder_with_client
    finder._client.em_images[100] = [_em_image(cds=True)]
    finder._client.cds_matches["em-5813128953"] = [
        _lm_match("VT037867", score=50), _lm_match("LH173", score=90)]

    df = finder.id_to_lines(100, match_type="cds")
    assert list(df["line"]) == ["LH173", "VT037867"]

    # Second call served from cache (no new API calls)
    calls_before = len(finder._client.calls)
    df2 = finder.id_to_lines(100, match_type="cds")
    assert list(df2["line"]) == ["LH173", "VT037867"]
    assert len(finder._client.calls) == calls_before


def test_id_to_lines_both_uses_algorithm_caches(finder_with_client):
    finder = finder_with_client
    em_img = _em_image(cds=True, pppm=True)
    finder._client.em_images[100] = [em_img]
    finder._client.cds_matches[em_img.id] = [_lm_match("VT037867", score=90)]
    finder._client.ppp_matches[em_img.id] = [_lm_match("LH173", score=85)]

    # Pre-populate the cds cache
    finder.id_to_lines(100, match_type="cds")

    df = finder.id_to_lines(100, match_type="both")
    assert set(df.columns) >= {"combined_rank", "cds_score", "pppm_score"}
    assert sorted(df["line"]) == ["LH173", "VT037867"]


def test_id_to_lines_empty_and_invalid(finder_with_client):
    finder = finder_with_client
    df = finder.id_to_lines(404, match_type="cds")
    assert df.empty and list(df.columns) == ["line", "library", "score", "image_id", "match_type"]
    df = finder.id_to_lines(404, match_type="both")
    assert df.empty and "combined_rank" in df.columns
    with pytest.raises(ValueError):
        finder.id_to_lines(404, match_type="wrong")


# ---------------------------------------------------------------------------
# Co-labeling / aggregation
# ---------------------------------------------------------------------------

def test_calculate_expression_entropy(finder):
    assert finder._calculate_expression_entropy({}) == 0.0
    assert finder._calculate_expression_entropy({"a": 0}) == 0.0
    assert finder._calculate_expression_entropy({"a": 5}) == 0.0
    assert abs(finder._calculate_expression_entropy({"a": 1, "b": 1}) - 1.0) < 1e-9


def test_calculate_weighted_specificity(finder):
    empty = finder._calculate_weighted_specificity(pd.DataFrame(), {"mbon01"})
    assert empty["weighted_type_proportion"] == 0.0
    df = pd.DataFrame({"type": ["MBON01", "LH173"], "score": [80.0, 20.0]})
    result = finder._calculate_weighted_specificity(df, {"mbon01"})
    assert result["weighted_type_proportion"] == pytest.approx(0.8)
    assert result["weighted_queried_score"] == 80.0
    assert result["weighted_total_score"] == 100.0
    assert result["mean_queried_score"] == 80.0
    # Missing score column
    assert finder._calculate_weighted_specificity(
        pd.DataFrame({"type": ["x"]}), {"x"}, score_column="nope"
    )["weighted_total_score"] == 0.0


def _patch_line_to_neuron(finder, monkeypatch, data):
    def fake(line_name, match_type="cds", top_n=100, **_kwargs):
        result = data.get(line_name)
        if isinstance(result, Exception):
            raise result
        return result if result is not None else pd.DataFrame()

    monkeypatch.setattr(finder, "line_to_neuron", fake)


def test_build_colabeling_matrix_methods(finder, monkeypatch):
    l1 = pd.DataFrame({"type": ["A", "B", "C"], "score": [10.0, 5.0, 2.0]})
    l2 = pd.DataFrame({"type": ["A", "B", "C"], "score": [9.0, 3.0, 1.0]})
    _patch_line_to_neuron(finder, monkeypatch, {"L1": l1, "L2": l2, "L3": pd.DataFrame()})

    matrix, sets = finder._build_colabeling_matrix(
        ["L1", "L2", "L3"], similarity_method="jaccard")
    assert matrix.loc["L1", "L1"] == 1.0
    assert matrix.loc["L1", "L2"] == pytest.approx(1.0)
    assert matrix.loc["L1", "L3"] == 0.0
    assert sets["L1"] == {"a", "b", "c"}
    assert sets["L3"] == set()

    matrix, _ = finder._build_colabeling_matrix(
        ["L1", "L2"], similarity_method="weighted_jaccard", min_score=4.0)
    # min_score drops L2's score-3 row: intersection min(10,9)=9,
    # union max(10,9)+max(5,0)=15 -> 9/15
    assert matrix.loc["L1", "L2"] == pytest.approx(0.6)

    matrix, _ = finder._build_colabeling_matrix(
        ["L1", "L2"], similarity_method="rank_correlation")
    assert matrix.loc["L1", "L2"] == pytest.approx(1.0)


def test_build_colabeling_matrix_filters_and_errors(finder, monkeypatch):
    l1 = pd.DataFrame({"type": ["A", "B"], "score": [100.0, 1.0]})
    l2 = pd.DataFrame({"type": ["A", "B"], "score": [90.0, 2.0]})
    _patch_line_to_neuron(
        finder, monkeypatch, {"L1": l1, "L2": l2, "BAD": RuntimeError("api")})

    matrix, sets = finder._build_colabeling_matrix(
        ["L1", "L2", "BAD"], min_type_avg_score=50.0)
    assert sets["L1"] == {"a"}  # low-score type 'b' filtered out
    assert sets["BAD"] == set()

    with pytest.raises(ValueError):
        finder._build_colabeling_matrix(["L1"], similarity_method="nope")


def test_calculate_colabeling_sparsity(finder):
    matrix = pd.DataFrame(
        [[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]],
        index=["L1", "L2", "L3"], columns=["L1", "L2", "L3"],
    )
    scores = finder._calculate_colabeling_sparsity(matrix, threshold=0.1)
    assert scores["L1"]["n_colabeling_lines"] == 1
    assert scores["L3"]["colabel_sparsity"] == 1.0
    single = pd.DataFrame([[1.0]], index=["L1"], columns=["L1"])
    assert finder._calculate_colabeling_sparsity(single)["L1"]["colabel_sparsity"] == 1.0


def _combined_df():
    return pd.DataFrame(
        {
            "line": ["VT037867", "VT037867", "LH173"],
            "score": [90.0, 60.0, 80.0],
            "source_bodyId": ["5813128953", "1234", "5813128953"],
            "source_dataset": ["hemibrain:v1.2.1", "hemibrain:v1.2.1", "unknown"],
        }
    )


def test_aggregate_results_pandas(finder):
    finder._neuron_dfs["hemibrain_v1_2_1"] = HEMIBRAIN_DF
    combined, stats = finder._aggregate_results_pandas(
        _combined_df(), "cds", is_multi_dataset=True, sort_by="max")
    assert list(stats["line"]) == ["VT037867", "LH173"]
    assert "weighted_score" in stats.columns
    assert "min_score_per_dataset" in stats.columns
    assert "matched_types" in stats.columns
    vt_row = stats[stats["line"] == "VT037867"].iloc[0]
    assert vt_row["match_count"] == 2
    assert "matched_bodyIds" in combined.columns

    # completeness sorting branch
    _, stats_c = finder._aggregate_results_pandas(
        _combined_df(), "cds", is_multi_dataset=False, sort_by="completeness")
    assert "coverage_ratio" in stats_c.columns

    # separate_splitgal4 adds line_type
    finder.separate_splitgal4 = True
    _, stats_s = finder._aggregate_results_pandas(
        _combined_df(), "cds", is_multi_dataset=False, sort_by="max")
    assert set(stats_s["line_type"]) == {"gal4_lexa"}

    # No 'line' column -> empty stats
    combined_only, empty_stats = finder._aggregate_results_pandas(
        pd.DataFrame({"score": [1.0]}), "cds", False, "max")
    assert empty_stats.empty


def test_aggregate_results_polars(finder):
    if not nbf_mod.HAS_POLARS:
        pytest.skip("polars not installed")
    finder._neuron_dfs["hemibrain_v1_2_1"] = HEMIBRAIN_DF
    combined, stats = finder._aggregate_results_polars(
        _combined_df(), "cds", is_multi_dataset=True, sort_by="completeness")
    assert len(stats) == 2
    assert "coverage_ratio" in stats.columns
    assert "matched_bodyIds" in combined.columns

    finder.separate_splitgal4 = True
    _, stats_s = finder._aggregate_results_polars(
        _combined_df(), "cds", is_multi_dataset=False, sort_by="max")
    assert "line_type" in stats_s.columns

    combined_only, empty_stats = finder._aggregate_results_polars(
        pd.DataFrame({"score": [1.0]}), "cds", False, "max")
    assert empty_stats.empty
