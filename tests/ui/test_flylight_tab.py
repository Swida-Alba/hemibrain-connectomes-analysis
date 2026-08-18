"""Tests for the FlyLight downloader UI tab and its runner integration."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ui.runner import TOOL_REGISTRY, ScriptRunner  # noqa: E402
from ui.tabs.flylight import (  # noqa: E402
    _fetch_file_preview,
    _fetch_lines,
    _parse_lines,
    CATEGORY_OPTIONS,
    DEFAULT_FLAT_STRUCTURE,
    DEFAULT_IMAGE_SUMMARY,
    FORMAT_OPTIONS,
    IMAGE_TYPE_OPTIONS,
    REGION_OPTIONS,
)


# ---------------------------------------------------------------------------
# Line-name parsing
# ---------------------------------------------------------------------------


class TestParseLines:
    def test_comma_separated(self):
        assert _parse_lines("R10A06, VT037867, SS00731") == [
            "R10A06", "VT037867", "SS00731",
        ]

    def test_whitespace_and_empty(self):
        assert _parse_lines("  R10A06 ,  ") == ["R10A06"]
        assert _parse_lines("") == []
        assert _parse_lines(None) == []

    def test_single_line(self):
        assert _parse_lines("VT037867") == ["VT037867"]


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------


class TestRunnerIntegration:
    def test_download_defaults(self):
        assert DEFAULT_FLAT_STRUCTURE is True
        assert DEFAULT_IMAGE_SUMMARY == "pdf"

    def test_registry_entry(self):
        tool = TOOL_REGISTRY["flylight_download"]
        assert tool["label"] == "FlyLight Image Download"
        assert tool["import"] == "from flylight_downloader import FlyLightDownloader"
        assert tool["class"] == "FlyLightDownloader"
        assert "download" in tool["methods"]

    def test_script_generation(self):
        runner = ScriptRunner()
        script = runner._generate_flylight_script(
            {
                "formats": ["png", "jpg"],
                "image_types": ["mip", "cdm"],
                "region": "Brain",
                "collection_category": ["GAL4/LEXA"],
                "max_workers": 4,
                "simple_mode": True,
                "verbose": "pbar",
            },
            {
                "line_name": ["R10A06", "VT037867"],
                "output_dir": "/tmp/out",
                "max_files": 6,
                "flat_structure": DEFAULT_FLAT_STRUCTURE,
                "generate_summary": DEFAULT_IMAGE_SUMMARY,
                "summary_images_per_page": (3, 2),
            },
        )
        assert "from flylight_downloader import FlyLightDownloader" in script
        assert "downloader = FlyLightDownloader(" in script
        assert "downloader.download(" in script
        assert "formats=['png', 'jpg']" in script
        assert "line_name=['R10A06', 'VT037867']" in script
        assert "output_dir='/tmp/out'" in script
        assert "flat_structure=True" in script
        assert "generate_summary='pdf'" in script
        assert "summary_images_per_page=(3, 2)" in script
        assert 'print("[DROCAT] Done.")' in script

    def test_timestamped_downloads_use_requested_prefix(self, tmp_path, monkeypatch):
        from flylight_downloader import FlyLightDownloader, FlyLightFile

        downloader = FlyLightDownloader(output_dir=str(tmp_path), verbose=False)
        file_info = FlyLightFile(
            key="R10A06/R10A06_mip.png",
            size=1,
            last_modified="",
            line_name="R10A06",
        )
        monkeypatch.setattr(
            downloader,
            "download_file",
            lambda file, output_dir, flat_structure=False: Path(output_dir) / file.filename,
        )

        downloader.download("R10A06", files=[file_info], add_timestamp=True)

        folders = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(folders) == 1
        assert folders[0].name.startswith("flylight-downloads_R10A06_")

    def test_tab_builds(self):
        """The FlyLight tab renders all controls and the output panel."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.flylight import create_flylight_tab

        client = Client(page("/flylight-tab-test"))
        with client:
            create_flylight_tab()

        texts = {getattr(el, "text", "") for el in client.elements.values()}
        labels = {
            getattr(el, "_props", {}).get("label")
            for el in client.elements.values()
        }
        assert "FlyLight Downloader" in texts
        assert "Driver line name(s)" in labels
        assert "Formats" in labels
        assert "Image Types" in labels
        assert "Collections" in labels
        assert "Search Lines" in texts
        assert "List Files" in texts
        assert "Download Images" in texts


# ---------------------------------------------------------------------------
# Preview helpers (network mocked)
# ---------------------------------------------------------------------------


class _FakeDownloader:
    """Network-free stand-in for FlyLightDownloader."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def search_lines(self, pattern):
        return [f"{pattern}001", f"{pattern}002"]

    def get_filtered_files(self, line_names, max_files_per_line=None):
        from types import SimpleNamespace

        files = []
        for name in line_names:
            files.append(
                SimpleNamespace(
                    line_name=name,
                    filename=f"{name}_mip.png",
                    extension=".png",
                    size_mb=1.5,
                    url=f"https://s3.example/{name}_mip.png",
                )
            )
        return files


@pytest.fixture()
def fake_downloader(monkeypatch):
    import flylight_downloader as fld

    monkeypatch.setattr(fld, "FlyLightDownloader", _FakeDownloader)
    return fld


class TestPreviewHelpers:
    def test_fetch_lines_uses_downloader(self, fake_downloader):
        lines = _fetch_lines(
            "R10A0", ["png"], ["mip"], "Brain", ["GAL4/LEXA"], True
        )
        assert lines == ["R10A0001", "R10A0002"]

    def test_fetch_file_preview_returns_files(self, fake_downloader):
        files = _fetch_file_preview(
            ["R10A06", "VT037867"],
            ["png", "jpg"], ["mip", "cdm"], "Brain",
            ["GAL4/LEXA", "SplitGAL4"], True,
            max_files_per_line=6,
        )
        assert [f.line_name for f in files] == ["R10A06", "VT037867"]
        assert files[0].filename == "R10A06_mip.png"
        assert files[0].url.startswith("https://")

    def test_option_sets_cover_backend_choices(self):
        # The tab's option lists must be a subset of the backend's supported
        # values so the generated scripts never pass unknown filters.
        assert set(FORMAT_OPTIONS) <= {"png", "jpg", "h5j", "lsm", "mp4", "json"}
        assert set(REGION_OPTIONS) <= {"Brain", "VNC", "All"}
        assert set(CATEGORY_OPTIONS) <= {
            "GAL4/LEXA", "SplitGAL4", "MCFO", "RawImages", "All",
        }
        assert IMAGE_TYPE_OPTIONS  # non-empty; backend matches by substring


# ---------------------------------------------------------------------------
# Backend search_lines: nested line folders (Gen1/CDM/R10A01/...) must be found
# ---------------------------------------------------------------------------


class _FakeS3Client:
    """S3 client stub whose CommonPrefixes mirror the real bucket layout:

    Gen1/CDM/R10A01/...  (nested: collection -> CDM -> line)
    Flat/SS01001/...     (flat: collection -> line)
    """

    def __init__(self):
        self.calls = []

    def list_objects_v2(self, Bucket, Prefix, Delimiter="/", MaxKeys=None):
        self.calls.append(Prefix)
        lines = ["R10A01", "R10A02", "R10A06", "SS01001", "SS01002"]
        # Strip trailing slash and drop the last segment to get the parent
        # folder prefix; then return the matching child folders.
        base = Prefix.rstrip("/")
        parent = base.rsplit("/", 1)[0] + "/" if "/" in base else ""
        children = []
        for line in lines:
            if line.startswith(base.split("/")[-1]) or base == "":
                children.append({"Prefix": f"{parent}{line}/"})
        return {"CommonPrefixes": children}


class TestSearchLines:
    def test_nested_and_flat_line_folders_found(self, monkeypatch):
        import flylight_downloader as fld

        fake_s3 = _FakeS3Client()
        downloader = fld.FlyLightDownloader(
            collections=["Gen1", "Flat"],
            formats=["png"], image_types=["mip"], region="Brain",
            simple_mode=False, verbose=False, use_boto3=True,
        )
        downloader._s3_client = fake_s3

        found = downloader.search_lines("R10A0")
        assert "R10A01" in found and "R10A06" in found
        assert "SS01001" not in found  # pattern must not leak across layouts

        found_ss = downloader.search_lines("SS0100")
        assert "SS01001" in found_ss and "SS01002" in found_ss

        # The prefix trick must be used: direct pattern-prefixed requests,
        # never a full two-level walk for flat collections.
        assert "Flat/R10A0" in fake_s3.calls or "Flat/CDM/R10A0" in fake_s3.calls

    def test_regex_fallback_walks_shallow_collections(self, monkeypatch):
        import flylight_downloader as fld

        class _WalkS3(_FakeS3Client):
            def list_objects_v2(self, Bucket, Prefix, Delimiter="/", MaxKeys=None):
                self.calls.append(Prefix)
                base = Prefix.rstrip("/")
                leaf = base.split("/")[-1]
                if leaf == "Gen1":
                    return {"CommonPrefixes": [{"Prefix": "Gen1/CDM/"}]}
                if leaf == "CDM":
                    return {"CommonPrefixes": [
                        {"Prefix": "Gen1/CDM/R10A01/"},
                        {"Prefix": "Gen1/CDM/R10A02/"},
                        {"Prefix": "Gen1/CDM/R10A06/"},
                    ]}
                if leaf == "Flat":
                    return {"CommonPrefixes": [{"Prefix": "Flat/SS01001/"}]}
                return {"CommonPrefixes": []}

        fake_s3 = _WalkS3()
        downloader = fld.FlyLightDownloader(
            collections=["Gen1", "Flat"],
            formats=["png"], image_types=["mip"], region="Brain",
            simple_mode=False, verbose=False, use_boto3=True,
        )
        downloader._s3_client = fake_s3

        # A pure regex cannot use the literal prefix trick, so the bounded
        # walk must find the nested Gen1/CDM line folders.
        found = downloader.search_lines(r"R10A0[0-9]$")
        assert found == ["R10A01", "R10A02", "R10A06"]
        assert any(call == "Gen1/CDM/" for call in fake_s3.calls)
