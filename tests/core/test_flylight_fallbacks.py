"""Regression tests for the FlyLight/NeuronBridge image fallback chain."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import flylight_downloader as fld  # noqa: E402
from flylight_downloader import FlyLightFile, FlyLightDownloader  # noqa: E402
from neuronbridge_finder import NeuronBridgeFinder  # noqa: E402


class _Response:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self._body


def _downloader(**kwargs):
    return FlyLightDownloader(
        auto_discover=False,
        use_boto3=False,
        verbose=False,
        **kwargs,
    )


def test_gen1_mcfo_parser_discovers_r_line_s3_images(monkeypatch):
    """The Gen1 MCFO viewer must work for R-lines such as R96A08."""
    html = (
        '<img src="https://s3.amazonaws.com/janelia-flylight-imagery/'
        'Gen1+MCFO/R96A08/R96A08-40x-central-multichannel_mip.png">'
        '<a href="https://s3.amazonaws.com/janelia-flylight-imagery/'
        'Gen1+MCFO/R96A08/R96A08-40x-central-CDM_1.png">image</a>'
        '<img src="https://s3.amazonaws.com/janelia-flylight-imagery/'
        'Gen1+MCFO/R96A08/R96A08-40x-central-CDM_1.png?width=250">'
        '<img src="https://s3.amazonaws.com/janelia-flylight-imagery/'
        'Gen1+MCFO/R96A08/R96A08-40x-central-thumbnail.png">'
    )

    monkeypatch.setattr(
        fld.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(html),
    )

    files = _downloader(collection_category="MCFO")._get_gen1_mcfo_files("R96A08")

    assert [file.filename for file in files] == [
        "R96A08-40x-central-multichannel_mip.png",
        "R96A08-40x-central-CDM_1.png",
    ]
    assert files[0].key.startswith("Gen1 MCFO/R96A08/")
    assert "Gen1%20MCFO" in files[0].url
    assert files[0].url.endswith("R96A08-40x-central-multichannel_mip.png")


def test_r_line_mcfo_category_uses_gen1_mcfo_viewer(monkeypatch):
    downloader = _downloader(collection_category="MCFO")
    expected = FlyLightFile(
        key="Gen1 MCFO/R96A08/R96A08-multichannel_mip.png",
        size=1,
        last_modified="",
        collection="Gen1 MCFO",
        line_name="R96A08",
    )
    monkeypatch.setattr(
        downloader,
        "_get_gen1_mcfo_files",
        lambda line_name: [expected] if line_name == "R96A08" else [],
    )
    monkeypatch.setattr(downloader, "_list_bucket_http", lambda *_args: [])

    files = downloader.list_files("R96A08")

    assert files == [expected]


def test_r_line_splitgal4_query_does_not_leak_gal4_files(monkeypatch):
    downloader = _downloader(collection_category="SplitGAL4")
    called = []
    monkeypatch.setattr(
        downloader,
        "_get_r_line_files",
        lambda line_name: called.append(line_name) or [],
    )
    monkeypatch.setattr(downloader, "_list_bucket_http", lambda *_args: [])

    assert downloader.list_files("R96A08") == []
    assert called == []


def test_r_line_rawimages_category_lists_direct_s3_files(monkeypatch):
    downloader = _downloader(collection_category="RawImages")
    raw_file = FlyLightFile(
        key="Gen1 MCFO/R96A08/R96A08-raw.lsm.bz2",
        size=10,
        last_modified="",
        collection="Gen1 MCFO",
        line_name="R96A08",
    )
    monkeypatch.setattr(
        downloader,
        "_list_bucket_http",
        lambda prefix: [raw_file] if prefix.endswith("R96A08/") else [],
    )
    monkeypatch.setattr(
        downloader,
        "_get_gen1_mcfo_files",
        lambda *_args: pytest.fail("RawImages must not use the MCFO page parser"),
    )

    assert downloader.list_files("R96A08") == [raw_file]


def test_nb_flylight_fallback_order_reaches_rawimages(monkeypatch, tmp_path):
    """Primary categories are followed by MCFO, then RawImages."""
    calls = []

    class FakeDownloader:
        def __init__(self, **kwargs):
            calls.append(kwargs.get("collection_category"))

        def get_filtered_files(self, _line_name):
            category = calls[-1]
            if category == "RawImages":
                return [
                    SimpleNamespace(
                        filename="R96A08-raw.png",
                        collection="Gen1 MCFO",
                        source="s3",
                        key="Gen1 MCFO/R96A08/R96A08-raw.png",
                        line_name="R96A08",
                    )
                ]
            return []

        def download(self, **_kwargs):
            return []

    monkeypatch.setattr(fld, "FlyLightDownloader", FakeDownloader)

    finder = object.__new__(NeuronBridgeFinder)
    finder.max_workers = 1
    finder.verbose = False
    finder.region = "All"
    finder._warning_collector = []

    _downloaded, missing = finder._download_flylight_images_with_category(
        lines=["R96A08"],
        output_dir=str(tmp_path),
        formats=["png"],
        image_types=["mip"],
        max_files=1,
        category=["GAL4/LEXA", "SplitGAL4"],
        simple_mode=False,
        verbose=False,
    )

    assert missing == []
    assert calls[:4] == ["GAL4/LEXA", "SplitGAL4", "MCFO", "RawImages"]


def test_standalone_primary_categories_fall_back_to_mcfo(monkeypatch):
    """The FlyLight UI path must fall back without going through NeuronBridge."""
    downloader = _downloader(
        formats=["png", "jpg"],
        image_types=["mip", "cdm"],
        region="Brain",
        collection_category=["GAL4/LEXA", "SplitGAL4"],
        simple_mode=True,
    )
    monkeypatch.setattr(downloader, "list_files", lambda _line_name: [])

    fallback_files = [
        FlyLightFile(
            key=(
                "Gen1 MCFO/R96A08/"
                f"R96A08-40x-central-multichannel_mip_{index}.png"
            ),
            size=1,
            last_modified="",
            collection="Gen1 MCFO",
            line_name="R96A08",
        )
        for index in range(8)
    ]
    calls = []

    class FakeFallbackDownloader:
        def __init__(self, category):
            self.category = category

        def get_filtered_files(self, _line_name, apply_simple_mode=None):
            calls.append((self.category, apply_simple_mode))
            return fallback_files if self.category == "MCFO" else []

    monkeypatch.setattr(
        downloader,
        "_get_fallback_downloader",
        lambda category: FakeFallbackDownloader(category),
    )

    files = downloader.get_filtered_files("R96A08", max_files_per_line=6)

    assert len(files) == 6
    assert all(file.collection == "Gen1 MCFO" for file in files)
    assert calls == [("MCFO", False)]
