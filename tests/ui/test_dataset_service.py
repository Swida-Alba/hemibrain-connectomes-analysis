"""Tests for DatasetService: dataset lists, name conversions, live-list
hidden-dataset filtering, and token file precedence."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ui.dataset_service as ds_mod
from ui.config import DATASETS, DEFAULTS, FLYWIRE_DATASETS, NEUPRINT_DATASETS
from ui.dataset_service import (
    DatasetService,
    dataset_to_folder,
    folder_to_dataset,
    is_flywire_dataset,
)

NEUPRINT_EXPECTED = {
    "male-cns:v1.0",
    "male-cns:v0.9",
    "hemibrain:v1.2.1",
    "hemibrain:v1.1",
    "optic-lobe:v1.1",
    "optic-lobe:v1.0.1",
    "manc:v1.2.3",
    "manc:v1.2.1",
    "manc:v1.0",
    "fib19:v1.0",
    "mushroombody",
}
FLYWIRE_EXPECTED = {"flywire_FAFB_v783", "flywire_BANC_v888", "flywire_BANC_v626"}


class TestDatasetLists:
    def test_neuprint_lists_are_complete_and_consistent(self):
        assert set(NEUPRINT_DATASETS) == NEUPRINT_EXPECTED
        assert set(NEUPRINT_DATASETS) <= set(DATASETS)
        assert set(FLYWIRE_DATASETS) == FLYWIRE_EXPECTED

    def test_banc_v888_not_supported_via_neuprint(self):
        # The NeuPrint server lists banc:v888 as hidden and not queryable;
        # BANC is served through FlyWire/Codex instead.
        assert "banc:v888" not in NEUPRINT_DATASETS
        assert "banc:v888" not in DATASETS
        assert "banc:v888" not in DatasetService.NEUPRINT_CANDIDATES
        assert "flywire_BANC_v888" in DATASETS

    def test_defaults_contain_core_parameters(self):
        for key in ("min_synapse_num", "min_ratio", "min_traversal_probability",
                    "max_interlayer", "output_format"):
            assert key in DEFAULTS


class TestDatasetNameConversion:
    def test_roundtrip_neuprint(self):
        for ds in NEUPRINT_EXPECTED:
            assert folder_to_dataset(dataset_to_folder(ds)) == ds

    def test_flywire_passthrough(self):
        for ds in FLYWIRE_EXPECTED:
            assert dataset_to_folder(ds) == ds
            assert folder_to_dataset(ds) == ds

    def test_examples(self):
        assert folder_to_dataset("hemibrain_v1_2_1") == "hemibrain:v1.2.1"
        assert folder_to_dataset("male-cns_v0_9") == "male-cns:v0.9"
        assert folder_to_dataset("manc_v1_2_3") == "manc:v1.2.3"
        assert dataset_to_folder("manc:v1.2.3") == "manc_v1_2_3"
        assert dataset_to_folder("hemibrain:v1.2.1") == "hemibrain_v1_2_1"


class TestIsFlywireDataset:
    def test_positive(self):
        for ds in list(FLYWIRE_EXPECTED) + ["fafb", "flywire_fafb:v783"]:
            assert is_flywire_dataset(ds)

    def test_negative(self):
        # banc:v888 must never be classified as the FlyWire BANC release.
        for ds in ["banc:v888", "male-cns:v0.9", "hemibrain:v1.2.1", "manc:v1.0"]:
            assert not is_flywire_dataset(ds)


class TestHiddenDatasetFilter:
    """The live /api/dbmeta/datasets listing is filtered: hidden entries such
    as banc:v888 are listed by the server but are not queryable."""

    SERVER_META = {
        "banc:v888": {"hidden": "True", "description": "BANC"},
        "male-cns:v0.9": {"hidden": "False", "description": "MaleCNS"},
        "mushroombody": {"hidden": "False"},
        "hemibrain:v1.2.1": {},
    }

    class _FakeResponse:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def test_hidden_datasets_excluded_from_available(self, monkeypatch):
        monkeypatch.setattr(
            "requests.get",
            lambda *a, **k: self._FakeResponse(dict(self.SERVER_META)),
        )
        svc = DatasetService()
        svc._token = "tok"
        svc._cave_token = "cav"
        available = svc.fetch_neuprint_datasets()
        assert "banc:v888" not in available
        assert available == ["hemibrain:v1.2.1", "male-cns:v0.9", "mushroombody"]

    def test_full_metadata_still_stored(self, monkeypatch):
        monkeypatch.setattr(
            "requests.get",
            lambda *a, **k: self._FakeResponse(dict(self.SERVER_META)),
        )
        svc = DatasetService()
        svc._token = "tok"
        svc._cave_token = "cav"
        svc.fetch_neuprint_datasets()
        assert "banc:v888" in svc._server_datasets  # kept for status display

    def test_api_failure_falls_back_to_candidates(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("network down")

        monkeypatch.setattr("requests.get", boom)
        svc = DatasetService()
        svc._token = "tok"
        svc._cave_token = "cav"
        monkeypatch.setattr(
            svc, "_probe_neuprint_dataset",
            lambda ds: type("Info", (), {"available": True})(),
        )
        available = svc.fetch_neuprint_datasets()
        assert set(available) == NEUPRINT_EXPECTED


class TestTokenFilePrecedence:
    """_load_tokens reads token_info.txt then token_info_local.txt; the local
    file overrides per-key, and a blank local value clears the template."""

    def _svc(self, monkeypatch, tmp_path, template, local):
        if template is not None:
            (tmp_path / "token_info.txt").write_text(template, encoding="utf-8")
        if local is not None:
            (tmp_path / "token_info_local.txt").write_text(local, encoding="utf-8")
        monkeypatch.setattr(ds_mod, "PROJECT_ROOT", tmp_path)
        return DatasetService()

    def test_local_overrides_template(self, monkeypatch, tmp_path):
        svc = self._svc(
            monkeypatch, tmp_path,
            "NEUPRINT_TOKEN='template-tok'\nCAVE_TOKEN='template-cave'\n",
            "NEUPRINT_TOKEN='local-tok'\n",
        )
        assert svc.get_token() == "local-tok"
        assert svc.get_cave_token() == "template-cave"

    def test_blank_local_clears_template(self, monkeypatch, tmp_path):
        svc = self._svc(
            monkeypatch, tmp_path,
            "NEUPRINT_TOKEN='template-tok'\nCAVE_TOKEN='template-cave'\n",
            "NEUPRINT_TOKEN=''\n",
        )
        assert svc.get_token() is None
        assert svc.get_cave_token() == "template-cave"

    def test_placeholder_local_ignored(self, monkeypatch, tmp_path):
        svc = self._svc(
            monkeypatch, tmp_path,
            "NEUPRINT_TOKEN='template-tok'\n",
            "NEUPRINT_TOKEN='YOUR_NEUPRINT_TOKEN_HERE'\n",
        )
        assert svc.get_token() is None

    def test_no_files_returns_none(self, monkeypatch, tmp_path):
        svc = self._svc(monkeypatch, tmp_path, None, None)
        assert svc.get_token() is None
        assert svc.get_cave_token() is None
