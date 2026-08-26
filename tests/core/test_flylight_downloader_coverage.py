"""Coverage tests for flylight_downloader.py.

All HTTP is mocked by monkeypatching ``fld.urllib.request.urlopen`` /
``urlretrieve``; all downloads write into pytest ``tmp_path``.  Module-level
collection globals are replaced with deep copies so tests never leak state.
"""

import copy
import json
import ssl
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import flylight_downloader as fld
from flylight_downloader import (
    FlyLightDownloader,
    FlyLightFile,
    VTSampleInfo,
    discover_s3_collections,
    update_collection_categories,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _Response:
    def __init__(self, body, status=200):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self._body = body
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self._body


def _downloader(tmp_path, **kwargs):
    defaults = dict(auto_discover=False, use_boto3=False, verbose=False)
    defaults.update(kwargs)
    defaults.setdefault("output_dir", str(tmp_path / "out"))
    return FlyLightDownloader(**defaults)


def _file(key, name=None, collection="Gen1", source="s3", http_url="", line_name=""):
    return FlyLightFile(
        key=key,
        size=0,
        last_modified="",
        collection=collection,
        line_name=line_name,
        source=source,
        http_url=http_url,
    )


@pytest.fixture
def isolated_collection_globals(monkeypatch):
    """Replace mutable module globals with deep copies for one test."""
    monkeypatch.setattr(fld, "COLLECTION_CATEGORIES", copy.deepcopy(fld.COLLECTION_CATEGORIES))
    monkeypatch.setattr(fld, "COLLECTION_TO_CATEGORY", dict(fld.COLLECTION_TO_CATEGORY))
    monkeypatch.setattr(fld, "_DISCOVERED_COLLECTIONS_CACHE", None)
    return fld


# ---------------------------------------------------------------------------
# FlyLightFile
# ---------------------------------------------------------------------------


def test_flylight_file_properties():
    f = FlyLightFile(key="Coll/Line/raw.lsm.bz2", size=2 * 1024 * 1024, last_modified="now")
    assert f.filename == "raw.lsm.bz2"
    assert f.extension == ".lsm.bz2"
    assert f.size_mb == pytest.approx(2.0)
    assert f.url.startswith("https://s3.amazonaws.com/janelia-flylight-imagery/")

    png = FlyLightFile(key="Coll/Line/img.png", size=0, last_modified="")
    assert png.extension == ".png"

    http = FlyLightFile(
        key="VT GAL4/VT1/brain/x.jpg", size=0, last_modified="",
        source="http", http_url="https://flimg.janelia.org/x.jpg",
    )
    assert http.url == "https://flimg.janelia.org/x.jpg"


# ---------------------------------------------------------------------------
# collection discovery / categories
# ---------------------------------------------------------------------------


_DISCOVER_XML = (
    '<?xml version="1.0"?>'
    '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
    "<CommonPrefixes><Prefix>Gen1/</Prefix></CommonPrefixes>"
    "<CommonPrefixes><Prefix>content/</Prefix></CommonPrefixes>"
    "<CommonPrefixes><Prefix>Split-GAL4 Omnibus Broad/</Prefix></CommonPrefixes>"
    "</ListBucketResult>"
)


def test_discover_s3_collections_http(monkeypatch, isolated_collection_globals):
    monkeypatch.setattr(fld, "HAS_BOTO3", False)
    monkeypatch.setattr(fld.urllib.request, "urlopen", lambda *a, **k: _Response(_DISCOVER_XML))

    collections = discover_s3_collections(verbose=True)

    assert collections == ["Gen1", "Split-GAL4 Omnibus Broad"]
    # Second call uses the module cache
    assert discover_s3_collections() is collections


def test_discover_s3_collections_http_failure_falls_back(
    monkeypatch, isolated_collection_globals
):
    monkeypatch.setattr(fld, "HAS_BOTO3", False)

    def _raise(*_a, **_k):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(fld.urllib.request, "urlopen", _raise)
    assert discover_s3_collections(verbose=True) == fld.FLYLIGHT_COLLECTIONS.copy()


def test_update_collection_categories(monkeypatch, isolated_collection_globals):
    # No new collections -> returns categories untouched
    assert update_collection_categories(["Gen1"]) is fld.COLLECTION_CATEGORIES

    discovered = ["Gen1", "Brand New MCFO", "Gen1 Extra", "Some Paper 2026"]
    update_collection_categories(discovered)

    assert "Brand New MCFO" in fld.COLLECTION_CATEGORIES["MCFO"]
    assert "Gen1 Extra" in fld.COLLECTION_CATEGORIES["GAL4/LEXA"]
    assert "Some Paper 2026" in fld.COLLECTION_CATEGORIES["SPLITGAL4"]
    assert fld.COLLECTION_TO_CATEGORY["Brand New MCFO"] == "MCFO"
    assert fld.COLLECTION_CATEGORIES["ALL"] == discovered


# ---------------------------------------------------------------------------
# initialization / parameter normalization
# ---------------------------------------------------------------------------


def test_post_init_normalizes_string_params(tmp_path):
    downloader = _downloader(tmp_path, formats="png", image_types="mip")
    assert downloader.formats == ["png"]
    assert downloader.image_types == ["mip"]


def test_post_init_boto3_success_and_failure(tmp_path, monkeypatch, capsys):
    class _FakeBoto3:
        def __init__(self, fail=False):
            self.fail = fail

        def client(self, *_a, **_k):
            if self.fail:
                raise RuntimeError("no boto3 config")
            return object()

    monkeypatch.setattr(fld, "HAS_BOTO3", True)
    monkeypatch.setattr(fld, "boto3", _FakeBoto3(fail=False), raising=False)
    monkeypatch.setattr(fld, "Config", lambda **_k: None, raising=False)
    monkeypatch.setattr(fld, "UNSIGNED", None, raising=False)

    ok = _downloader(tmp_path, use_boto3=True, verbose=True)
    assert ok._s3_client is not None
    assert "Using boto3" in capsys.readouterr().out

    monkeypatch.setattr(fld, "boto3", _FakeBoto3(fail=True), raising=False)
    failed = _downloader(tmp_path, use_boto3=True, verbose=True)
    assert failed._s3_client is None
    assert "boto3 initialization failed" in capsys.readouterr().out


def test_post_init_no_boto3_message(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fld, "HAS_BOTO3", False)
    downloader = _downloader(tmp_path, use_boto3=True, verbose=True)
    assert downloader._s3_client is None
    assert "boto3 not installed" in capsys.readouterr().out


def test_resolve_collections_by_category(tmp_path, capsys):
    downloader = _downloader(tmp_path, collection_category="MCFO", verbose=True)
    assert set(downloader._resolved_collections) == set(fld.COLLECTION_CATEGORIES["MCFO"])
    assert "collections" in capsys.readouterr().out

    unknown = _downloader(tmp_path, collection_category="NoSuchCategory", verbose=True)
    out = capsys.readouterr().out
    assert "Unknown category" in out
    assert unknown._resolved_collections == fld.FLYLIGHT_COLLECTIONS

    listed = _downloader(tmp_path, collection_category=["MCFO", "MCFO"])
    assert set(listed._resolved_collections) == set(fld.COLLECTION_CATEGORIES["MCFO"])

    explicit = _downloader(tmp_path, collections=["Gen1"])
    assert explicit._resolved_collections == ["Gen1"]


def test_category_helpers(tmp_path):
    downloader = _downloader(tmp_path)
    assert downloader.get_collection_category("Gen1") == "GAL4/LEXA"
    assert downloader.get_collection_category("Unknown Coll") == "Other"
    assert "MCFO" in FlyLightDownloader.list_categories()


def test_log_and_progress_modes(tmp_path, capsys):
    downloader = _downloader(tmp_path, verbose=False)
    downloader._log("Downloading something")
    downloader._progress(1, 3, "step")
    assert capsys.readouterr().out == ""

    pbar = _downloader(tmp_path, verbose="pbar")
    pbar._log("Downloading file.png")  # suppressed in pbar mode
    pbar._log("Summary message")  # kept
    pbar._progress(2, 3, "listing")
    out = capsys.readouterr().out
    assert "Downloading" not in out
    assert "Summary message" in out
    assert "[DROCAT][progress] 2/3 listing" in out


def test_requested_category_keys(tmp_path):
    assert _downloader(tmp_path)._requested_category_keys() == {"ALL"}
    assert (
        _downloader(tmp_path, collection_category="GAL4/LEXA")._requested_category_keys()
        == {"GAL4LEXA"}
    )
    assert (
        _downloader(tmp_path, collections=["Gen1 MCFO"])._requested_category_keys()
        == {"GEN1MCFO"}
    )


def test_automatic_fallback_categories(tmp_path):
    gal4 = _downloader(tmp_path, collection_category="GAL4/LEXA")
    assert gal4._automatic_fallback_categories() == ["MCFO", "RawImages"]

    assert _downloader(tmp_path)._automatic_fallback_categories() == []
    assert (
        _downloader(tmp_path, collection_category="GAL4/LEXA", automatic_fallback=False)
        ._automatic_fallback_categories()
        == []
    )
    assert (
        _downloader(tmp_path, collection_category="MCFO")._automatic_fallback_categories()
        == []
    )
    assert (
        _downloader(tmp_path, collection_category="GAL4/LEXA", collections=["Gen1"])
        ._automatic_fallback_categories()
        == []
    )


def test_get_fallback_downloader_cached(tmp_path):
    parent = _downloader(tmp_path, collection_category="GAL4/LEXA")
    first = parent._get_fallback_downloader("MCFO")
    second = parent._get_fallback_downloader("MCFO")
    assert first is second
    assert first.automatic_fallback is False
    assert first.verbose is False
    assert first.collection_category == "MCFO"


# ---------------------------------------------------------------------------
# VT / R-line / MCFO page parsing
# ---------------------------------------------------------------------------


_VT_HTML = """
<html>
<a href="download.cgi?id=111">sample brain</a>
<a href="download.cgi?id=222">sample vnc</a>
sid=abcdef123456
<img src="https://flimg.janelia.org/flylight-image/external-data/adult/secdata/projections/130101/VT037867-fA00b/VT037867_fA00b_total.jpg">
<img src="https://flimg.janelia.org/flylight-image/external-data/adult/secdata/projections/130101/VT037867-fA00v/VT037867_fA00v_total.jpg">
</html>
"""


def test_parse_vt_page(tmp_path, monkeypatch):
    downloader = _downloader(tmp_path)
    monkeypatch.setattr(fld.urllib.request, "urlopen", lambda *a, **k: _Response(_VT_HTML))

    samples, session_id = downloader._parse_vt_page("VT037867")

    assert session_id == "abcdef123456"
    assert len(samples) == 2
    regions = sorted(s.region for s in samples)
    assert regions == ["brain", "vnc"]
    assert [s.sample_id for s in samples] == ["111", "222"]


def test_parse_vt_page_error(tmp_path, monkeypatch):
    downloader = _downloader(tmp_path)

    def _raise(*_a, **_k):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(fld.urllib.request, "urlopen", _raise)
    samples, session_id = downloader._parse_vt_page("VT037867")
    assert samples == [] and session_id == ""


def test_get_vt_files_from_cache(tmp_path):
    downloader = _downloader(tmp_path)
    downloader._vt_sample_cache["VT037867"] = [
        VTSampleInfo(
            sample_id="111",
            line_name="VT037867",
            region="brain",
            date="130101",
            sample_path="VT037867-fA00b",
            session_id="abc",
        )
    ]
    files = downloader._get_vt_files("VT037867")
    # 2 standard projections + 20 substacks + 1 translation video
    assert len(files) == 23
    assert all(f.source == "http" and f.collection == "VT GAL4" for f in files)
    assert files[-1].http_url.endswith(".t.mp4")


_R_LINE_HTML = """
<html>
<img src="https://flimg.janelia.org/flylight-image/external-data/adult/secdata/projections/130101/R78H08-fA00b/R78H08_fA00b_total.jpg">
<img src="https://flimg.janelia.org/flylight-image/external-data/adult/secdata/projections/130101/R78H08-fA00b/R78H08_fA00b_total.jpg?cachebust=1">
<img src="https://flimg.janelia.org/flylight-image/external-data/adult/secdata/projections/130101/R78H08-fA00v/R78H08_fA00v_total.jpg">
<img src="https://flimg.janelia.org/flylight-image/external-data/adult/secdata/translations/130101/R78H08-fA00b/R78H08_fA00b.t.mp4">
</html>
"""


def test_get_r_line_files(tmp_path, monkeypatch):
    downloader = _downloader(tmp_path)
    monkeypatch.setattr(fld.urllib.request, "urlopen", lambda *a, **k: _Response(_R_LINE_HTML))

    files = downloader._get_r_line_files("R78H08")

    assert len(files) == 3  # duplicate URL with query string deduplicated
    regions = {f.key.split("/")[2] for f in files}
    assert regions == {"brain", "vnc"}
    assert all(f.collection == "Gen1 GAL4" for f in files)


def test_get_r_line_files_error(tmp_path, monkeypatch):
    downloader = _downloader(tmp_path)

    def _raise(*_a, **_k):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(fld.urllib.request, "urlopen", _raise)
    assert downloader._get_r_line_files("R78H08") == []


_MCFO_HTML = """
<html>
<img src="https://s3.amazonaws.com/janelia-flylight-imagery/Gen1+MCFO/VT000770/VT000770-image.png">
<img src="https://s3.amazonaws.com/janelia-flylight-imagery/Gen1+MCFO/VT000770/VT000770-image.png">
<img src="https://s3.amazonaws.com/janelia-flylight-imagery/Annotator+Gen1+MCFO/VT000770/VT000770-thumbnail.jpg">
</html>
"""


def test_get_gen1_mcfo_files(tmp_path, monkeypatch):
    downloader = _downloader(tmp_path)
    monkeypatch.setattr(fld.urllib.request, "urlopen", lambda *a, **k: _Response(_MCFO_HTML))

    files = downloader._get_gen1_mcfo_files("VT000770")

    # Thumbnail skipped, duplicate deduplicated
    assert len(files) == 1
    assert files[0].collection == "Gen1 MCFO"
    assert files[0].source == "s3"
    assert files[0].key == "Gen1 MCFO/VT000770/VT000770-image.png"
    # Backward-compatible alias
    monkeypatch.setattr(
        downloader, "_get_gen1_mcfo_files", lambda line: ["sentinel"]
    )
    assert downloader._get_vt_mcfo_files("VT000770") == ["sentinel"]


def test_get_gen1_mcfo_files_all_endpoints_fail(tmp_path, monkeypatch):
    downloader = _downloader(tmp_path)

    def _raise(*_a, **_k):
        raise urllib.error.URLError("tls handshake failed")

    monkeypatch.setattr(fld.urllib.request, "urlopen", _raise)
    assert downloader._get_gen1_mcfo_files("VT000770") == []


def test_list_vt_files_with_verify(tmp_path, monkeypatch):
    downloader = _downloader(tmp_path)
    sample = VTSampleInfo(
        sample_id="1", line_name="VT037867", region="brain",
        date="130101", sample_path="VT037867-fA00b",
    )
    downloader._vt_sample_cache["VT037867"] = [sample]
    monkeypatch.setattr(downloader, "_get_vt_mcfo_files", lambda line: [])
    monkeypatch.setattr(
        fld.urllib.request, "urlopen", lambda *a, **k: _Response(b"", status=200)
    )

    unverified = downloader.list_vt_files("VT037867", verify=False, include_mcfo=True)
    assert len(unverified) == 23

    verified = downloader.list_vt_files("VT037867", verify=True, include_mcfo=False)
    assert len(verified) == 23  # HEAD requests all report status 200


# ---------------------------------------------------------------------------
# filtering
# ---------------------------------------------------------------------------


def test_format_extensions_and_image_type_patterns(tmp_path):
    downloader = _downloader(tmp_path, formats=["png", "h5j", "tiff"], image_types=["mip", "custom_.*"])
    extensions = downloader._get_format_extensions()
    assert set(extensions) == {".png", ".h5j", ".tiff"}

    patterns = downloader._get_image_type_patterns()
    assert len(patterns) == 2
    assert patterns[1].search("my_custom_file.png")


def test_matches_filters(tmp_path):
    downloader = _downloader(tmp_path, formats=["png"], image_types=["mip"], region="Brain")

    brain = _file("Gen1/R1/R1-fA00b_mip.png")
    vnc = _file("Gen1/R1/R1-fA00v_mip.png")
    wrong_format = _file("Gen1/R1/R1_mip.h5j")
    wrong_type = _file("Gen1/R1/R1_cdm.png")

    assert downloader._matches_filters(brain)
    assert not downloader._matches_filters(vnc)
    assert not downloader._matches_filters(wrong_format)
    assert not downloader._matches_filters(wrong_type)

    vnc_downloader = _downloader(tmp_path, formats=["png"], image_types=["mip"], region="VNC")
    assert vnc_downloader._matches_filters(vnc)
    assert not vnc_downloader._matches_filters(brain)
    assert not vnc_downloader._matches_filters(_file("Gen1/R1/R1_mip.png"))


def test_apply_simple_mode_filter(tmp_path):
    downloader = _downloader(tmp_path, simple_mode=True)

    splitgal4 = [
        _file("SG/SS01/SS01_20x_multichannel_mip.png", collection="Split-GAL4 Omnibus Broad"),
        _file("SG/SS01/SS01_20x_multichannel_image1.png", collection="Split-GAL4 Omnibus Broad"),
        _file("SG/SS01/SS01_63x_mip.png", collection="Split-GAL4 Omnibus Broad"),
    ]
    kept = downloader.apply_simple_mode_filter(splitgal4)
    assert [f.filename for f in kept] == ["SS01_20x_multichannel_mip.png"]

    vt = [
        _file("VT GAL4/VT1/VT1_total.jpg", collection="VT GAL4"),
        _file("VT GAL4/VT1/VT1_01.jpg", collection="VT GAL4"),
    ]
    assert [f.filename for f in downloader.apply_simple_mode_filter(vt)] == ["VT1_total.jpg"]

    gen1 = [
        _file("Gen1/R1/R1_total.jpg", collection="Gen1"),
        _file("Gen1/R1/R1-CDM_1.png", collection="Gen1"),
        _file("Gen1/R1/R1_other.png", collection="Gen1"),
    ]
    assert len(downloader.apply_simple_mode_filter(gen1)) == 2

    mcfo = [_file("Gen1 MCFO/R1/R1_anything.png", collection="Gen1 MCFO")]
    assert len(downloader.apply_simple_mode_filter(mcfo)) == 1

    # simple_mode disabled -> input returned untouched
    off = _downloader(tmp_path, simple_mode=False)
    assert off.apply_simple_mode_filter(splitgal4) is splitgal4
    assert downloader.apply_simple_mode_filter([]) == []


# ---------------------------------------------------------------------------
# bucket listing
# ---------------------------------------------------------------------------


def _listing_xml(keys, truncated=False):
    contents = "".join(
        f"<Contents><Key>{key}</Key><Size>10</Size>"
        f"<LastModified>2024-01-01T00:00:00Z</LastModified></Contents>"
        for key in keys
    )
    return (
        '<?xml version="1.0"?>'
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"{contents}<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>"
        "</ListBucketResult>"
    )


def test_list_bucket_http_with_truncation(tmp_path, monkeypatch):
    downloader = _downloader(tmp_path)
    pages = [
        _Response(_listing_xml(["Gen1/CDM/R78H08/a.png"], truncated=True)),
        _Response(_listing_xml(["Gen1/CDM/R78H08/b.png"])),
    ]
    seen_urls = []

    def fake_urlopen(url, *args, **kwargs):
        seen_urls.append(url)
        return pages.pop(0)

    monkeypatch.setattr(fld.urllib.request, "urlopen", fake_urlopen)

    files = downloader._list_bucket_http("Gen1/CDM/R78H08/")

    assert [f.filename for f in files] == ["a.png", "b.png"]
    assert files[0].collection == "Gen1"
    assert files[0].line_name == "R78H08"  # CDM subfolder structure detected
    assert len(seen_urls) == 2
    assert "marker=" in seen_urls[1]


def test_list_bucket_http_error(tmp_path, monkeypatch):
    downloader = _downloader(tmp_path)

    def _raise(*_a, **_k):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(fld.urllib.request, "urlopen", _raise)
    assert downloader._list_bucket_http("Gen1/") == []


def test_list_bucket_boto3(tmp_path):
    downloader = _downloader(tmp_path)

    class _FakePaginator:
        def paginate(self, Bucket, Prefix):
            yield {
                "Contents": [
                    {"Key": "Gen1/CDM/R78H08/a.png", "Size": 10, "LastModified": "now"},
                    {"Key": "Other/Line/b.png", "Size": 5, "LastModified": "now"},
                ]
            }

    class _FakeClient:
        def get_paginator(self, _name):
            return _FakePaginator()

    downloader._s3_client = _FakeClient()
    files = downloader._list_bucket_boto3("Gen1/")
    assert len(files) == 2
    assert files[0].line_name == "R78H08"
    assert files[1].line_name == "Line"


def test_list_bucket_boto3_error(tmp_path):
    downloader = _downloader(tmp_path)

    class _BrokenPaginator:
        def paginate(self, **_kwargs):
            raise RuntimeError("no access")
            yield  # pragma: no cover

    class _BrokenClient:
        def get_paginator(self, _name):
            return _BrokenPaginator()

    downloader._s3_client = _BrokenClient()
    assert downloader._list_bucket_boto3("Gen1/") == []


# ---------------------------------------------------------------------------
# list_files dispatch
# ---------------------------------------------------------------------------


def test_list_files_cache_hit(tmp_path):
    downloader = _downloader(tmp_path)
    cached = [_file("x/y.png")]
    downloader._file_cache["SS00001"] = cached
    assert downloader.list_files("SS00001") is cached


def test_list_files_vt_category_branches(tmp_path, monkeypatch):
    sentinel_gal4 = [_file("VT GAL4/VT1/brain/x_total.jpg")]
    sentinel_mcfo = [_file("Gen1 MCFO/VT1/x.png")]

    def _make(category):
        downloader = _downloader(tmp_path, collection_category=category)
        monkeypatch.setattr(downloader, "_get_vt_files", lambda line: sentinel_gal4)
        monkeypatch.setattr(downloader, "_get_gen1_mcfo_files", lambda line: sentinel_mcfo)
        monkeypatch.setattr(downloader, "list_vt_files", lambda *a, **k: sentinel_gal4 + sentinel_mcfo)
        return downloader

    both = _make(None)
    assert len(both.list_files("VT037867")) == 2

    gal4_only = _make("GAL4/LEXA")
    assert gal4_only.list_files("VT037867") == sentinel_gal4

    mcfo_only = _make("MCFO")
    assert mcfo_only.list_files("VT037867") == sentinel_mcfo

    neither = _make("RawImages")
    assert neither.list_files("VT037867") == []


def test_list_files_r_line_branches(tmp_path, monkeypatch):
    downloader = _downloader(tmp_path, collections=["Gen1", "Gen1 MCFO"])

    gal4_files = [_file("Gen1/R78H08/brain/R78H08_total.jpg", source="http", line_name="R78H08")]
    cdm_files = [_file("Gen1/CDM/R78H08/R78H08-CDM_1.png", line_name="R78H08")]
    mcfo_files = [_file("Gen1 MCFO/R78H08/R78H08-mcfo.png", line_name="R78H08")]
    raw_files = [_file("Gen1 MCFO/R78H08/R78H08.lsm.bz2", line_name="R78H08")]

    monkeypatch.setattr(downloader, "_get_r_line_files", lambda line: gal4_files)
    monkeypatch.setattr(downloader, "_get_gen1_mcfo_files", lambda line: mcfo_files)
    prefixes = []

    def fake_list_http(prefix, *args):
        prefixes.append(prefix)
        if prefix == "Gen1/CDM/R78H08/":
            return cdm_files
        if prefix == "Gen1 MCFO/R78H08/":
            return raw_files
        return []

    monkeypatch.setattr(downloader, "_list_bucket_http", fake_list_http)

    files = downloader.list_files("R78H08")
    keys = {f.key for f in files}
    assert gal4_files[0].key in keys
    assert cdm_files[0].key in keys
    assert mcfo_files[0].key in keys
    assert "Gen1/CDM/R78H08/" in prefixes

    # Split-GAL4-only query must not touch any R-line source
    split_only = _downloader(tmp_path, collections=["Split-GAL4 Omnibus Broad"])
    monkeypatch.setattr(split_only, "_get_r_line_files", lambda line: gal4_files)
    assert split_only.list_files("R78H08") == []


def test_list_files_s3_branch_cdm_fallback(tmp_path, monkeypatch):
    downloader = _downloader(tmp_path, collections=["CollA"])
    cdm_files = [_file("CollA/CDM/SS01015/SS01015-CDM_1.png", line_name="SS01015")]
    calls = []

    def fake_list_http(prefix, *args):
        calls.append(prefix)
        if prefix == "CollA/CDM/SS01015/":
            return cdm_files
        return []

    monkeypatch.setattr(downloader, "_list_bucket_http", fake_list_http)

    files = downloader.list_files("SS01015")
    assert files == cdm_files
    assert calls == ["CollA/SS01015/", "CollA/CDM/SS01015/"]


# ---------------------------------------------------------------------------
# get_filtered_files
# ---------------------------------------------------------------------------


def test_get_filtered_files_input_forms_and_limit(tmp_path, monkeypatch):
    downloader = _downloader(tmp_path, formats=["png"], image_types=["all"], simple_mode=False)
    seen = []

    def fake_list_files(name, **kwargs):
        seen.append(name)
        return [
            _file(f"Coll/{name}/{name}_a.png", line_name=name),
            _file(f"Coll/{name}/{name}_b.png", line_name=name),
        ]

    monkeypatch.setattr(downloader, "list_files", fake_list_files)

    files = downloader.get_filtered_files("VT000001, R78H08")
    assert seen == ["VT000001", "R78H08"]
    assert len(files) == 4

    limited = downloader.get_filtered_files(["VT000001"], max_files_per_line=1)
    assert len(limited) == 1


def test_get_filtered_files_simple_mode_fallback(tmp_path, monkeypatch):
    downloader = _downloader(
        tmp_path, formats=["png"], image_types=["mip"], simple_mode=True,
        simple_mode_min_files=2, verbose=True,
    )
    pre_files = [
        _file("SG/SS01/SS01_63x_multichannel_mip.png", collection="Split-GAL4 Omnibus Broad", line_name="SS00001"),
        _file("SG/SS01/SS01_other_mip.png", collection="Split-GAL4 Omnibus Broad", line_name="SS00001"),
        _file("SG/SS01/SS01_another_mip.png", collection="Split-GAL4 Omnibus Broad", line_name="SS00001"),
    ]
    monkeypatch.setattr(downloader, "list_files", lambda name, **k: pre_files)

    files = downloader.get_filtered_files("SS00001")
    # simple_mode removed everything -> fallback kept min_files entries
    assert len(files) == 2
    assert files[0].filename == "SS01_63x_multichannel_mip.png"  # multichannel first


def test_get_filtered_files_category_fallback(tmp_path, monkeypatch):
    downloader = _downloader(
        tmp_path, collection_category="GAL4/LEXA", formats=["png"],
        image_types=["mip"], simple_mode=False,
    )
    # Primary query finds files but none pass the format filter
    monkeypatch.setattr(
        downloader, "list_files",
        lambda name, **k: [_file("Gen1/VT1/VT1.h5j", line_name=name)],
    )
    fallback_files = [_file("Gen1 MCFO/VT1/VT1_mip.png", line_name="VT000770")]

    class _StubFallback:
        def get_filtered_files(self, name, apply_simple_mode=False):
            return fallback_files

    monkeypatch.setattr(downloader, "_get_fallback_downloader", lambda cat: _StubFallback())

    files = downloader.get_filtered_files("VT000770")
    assert files == fallback_files


# ---------------------------------------------------------------------------
# downloads
# ---------------------------------------------------------------------------


def test_download_file_http_retries(tmp_path, monkeypatch):
    downloader = _downloader(tmp_path)
    target = tmp_path / "dl" / "x.png"
    file = _file("VT GAL4/VT1/brain/x.png", source="http", http_url="https://cdn/x.png")

    calls = {"n": 0}

    def flaky(url, path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ssl.SSLError("UNEXPECTED_EOF_WHILE_READING")
        Path(path).write_bytes(b"img")

    monkeypatch.setattr(fld.urllib.request, "urlretrieve", flaky)
    import time as time_module
    monkeypatch.setattr(time_module, "sleep", lambda *_a, **_k: None)
    assert downloader._download_file_http(file, target) is True
    assert target.read_bytes() == b"img"

    # URL errors exhaust retries
    def always_fail(url, path):
        raise urllib.error.URLError("network unreachable")

    monkeypatch.setattr(fld.urllib.request, "urlretrieve", always_fail)
    assert downloader._download_file_http(file, target, max_retries=2) is False

    # Non-network errors are not retried
    def bad(url, path):
        raise ValueError("bad url")

    monkeypatch.setattr(fld.urllib.request, "urlretrieve", bad)
    assert downloader._download_file_http(file, target) is False


def test_download_file_boto3(tmp_path):
    downloader = _downloader(tmp_path)
    target = tmp_path / "s3" / "x.png"
    file = _file("Gen1/R1/x.png")

    class _FakeClient:
        def __init__(self, fail=False):
            self.fail = fail

        def download_file(self, bucket, key, path):
            if self.fail:
                raise RuntimeError("denied")
            Path(path).write_bytes(b"data")

    downloader._s3_client = _FakeClient()
    assert downloader._download_file_boto3(file, target) is True
    assert target.read_bytes() == b"data"

    downloader._s3_client = _FakeClient(fail=True)
    assert downloader._download_file_boto3(file, target) is False


def test_download_file_dispatch(tmp_path, monkeypatch):
    downloader = _downloader(tmp_path)
    file = _file("VT GAL4/VT1/brain/x.png", source="http",
                  http_url="https://cdn/x.png", line_name="VT000001")

    def fake_urlretrieve(url, path):
        Path(path).write_bytes(b"bytes")

    monkeypatch.setattr(fld.urllib.request, "urlretrieve", fake_urlretrieve)

    result = downloader.download_file(file, output_dir=str(tmp_path), flat_structure=True)
    assert result == tmp_path / "VT000001" / "x.png"
    assert result.exists()

    result2 = downloader.download_file(file, output_dir=str(tmp_path))
    assert result2 == tmp_path / file.key

    # Failing download returns None
    def fail_urlretrieve(url, path):
        raise ValueError("nope")

    monkeypatch.setattr(fld.urllib.request, "urlretrieve", fail_urlretrieve)
    assert downloader.download_file(file, output_dir=str(tmp_path)) is None


def test_download_no_files_and_dry_run(tmp_path, monkeypatch):
    downloader = _downloader(tmp_path)
    monkeypatch.setattr(downloader, "get_filtered_files", lambda *a, **k: [])
    assert downloader.download("VT000770") == []

    files = [_file("VT GAL4/VT1/brain/x_total.jpg", source="http",
                    http_url="https://cdn/x.jpg", line_name="VT000770")]
    assert downloader.download("VT000770", files=files, dry_run=True) == []


def test_download_sequential_and_parallel(tmp_path, monkeypatch):
    def fake_urlretrieve(url, path):
        Path(path).write_bytes(b"content")

    monkeypatch.setattr(fld.urllib.request, "urlretrieve", fake_urlretrieve)

    files = [
        _file("VT GAL4/VT1/brain/a_total.jpg", source="http",
              http_url="https://cdn/a.jpg", line_name="VT000770"),
        _file("VT GAL4/VT1/brain/b_total.jpg", source="http",
              http_url="https://cdn/b.jpg", line_name="VT000770"),
    ]

    seen_callbacks = []

    def on_file(path, name):
        seen_callbacks.append((Path(path).name, name))

    sequential = _downloader(tmp_path, max_workers=1)
    downloaded = sequential.download(
        "VT000770", output_dir=str(tmp_path / "seq"), files=files,
        add_timestamp=False, on_file_downloaded=on_file,
    )
    assert len(downloaded) == 2
    assert all(p.exists() for p in downloaded)
    assert len(seen_callbacks) == 2

    parallel = _downloader(tmp_path, max_workers=2, verbose="pbar")
    downloaded2 = parallel.download(
        ["VT000770", "R78H08", "SS00001", "LH173"],
        output_dir=str(tmp_path / "par"), files=files, add_timestamp=False,
    )
    assert len(downloaded2) == 2


def test_generate_image_summary(tmp_path, monkeypatch):
    downloader = _downloader(tmp_path, verbose=True)

    # No image files -> early return
    downloader._generate_image_summary(
        downloaded_files=[tmp_path / "note.txt"],
        output_dir=tmp_path, line_name="VT1", timestamp="ts",
        generate_summary="pdf",
    )

    img = tmp_path / "VT1_x.png"
    img.write_bytes(b"png-bytes")

    created = []
    import neuronbridge_finder as nbf_module
    monkeypatch.setattr(
        nbf_module, "create_image_pdf",
        lambda **kwargs: created.append(("pdf", kwargs["output_pdf"])),
        raising=False,
    )
    downloader._generate_image_summary(
        downloaded_files=[img], output_dir=tmp_path, line_name="VT1",
        timestamp="ts", generate_summary="pdf",
    )
    assert len(created) == 1
    assert created[0][1].endswith("VT1_ts_summary.pdf")

    # generate_summary False -> no formats processed
    downloader._generate_image_summary(
        downloaded_files=[img], output_dir=tmp_path, line_name="VT1",
        timestamp="ts", generate_summary=False,
    )


def test_get_metadata(tmp_path, monkeypatch):
    downloader = _downloader(tmp_path)
    files = [
        _file("Coll/VT1/VT1-metadata.json", source="http",
              http_url="https://flimg.janelia.org/VT1-metadata.json", line_name="VT000770"),
        _file("Coll/VT1/VT1-bad.json", source="http",
              http_url="https://other.example.com/VT1-bad.json", line_name="VT000770"),
    ]
    monkeypatch.setattr(downloader, "get_filtered_files", lambda name, **k: files)

    payload = {"line": "VT000770", "driver": "GAL4"}
    calls = []

    def fake_urlopen(req_or_url, *args, **kwargs):
        url = getattr(req_or_url, "full_url", req_or_url)
        calls.append(url)
        if "janelia.org" in url:
            return _Response(json.dumps(payload))
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(fld.urllib.request, "urlopen", fake_urlopen)

    metadata = downloader.get_metadata("VT000770")
    assert metadata == [payload]
    assert downloader.formats != ["json"]  # original filters restored


def test_search_lines_http_fallback(tmp_path, monkeypatch):
    downloader = _downloader(tmp_path, collections=["Gen1"])

    def fake_list_http(prefix, *args):
        if prefix.startswith("Gen1/R78"):
            return [_file("Gen1/R78H08/x.png", line_name="R78H08"),
                    _file("Gen1/R79A01/x.png", line_name="R79A01")]
        return []

    monkeypatch.setattr(downloader, "_list_bucket_http", fake_list_http)
    assert downloader.search_lines("R78") == ["R78H08"]


def test_search_lines_boto3_walk(tmp_path):
    downloader = _downloader(tmp_path, collections=["Gen1"])

    class _FakeS3:
        def list_objects_v2(self, Bucket, Prefix, Delimiter=None):
            if Prefix == "Gen1/":
                return {"CommonPrefixes": [{"Prefix": "Gen1/CDM/"}]}
            if Prefix == "Gen1/CDM/":
                return {"CommonPrefixes": [{"Prefix": "Gen1/CDM/R78H08/"}]}
            return {}

    downloader._s3_client = _FakeS3()
    assert downloader.search_lines("R78H08") == ["R78H08"]


def test_convenience_functions(tmp_path, monkeypatch):
    class _FakeDownloader:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            _FakeDownloader.instances.append(self)

        def download(self, line_name, max_files=None):
            return [Path("downloaded.png")]

        def get_filtered_files(self, line_name):
            return [_file("x/y.png")]

    monkeypatch.setattr(fld, "FlyLightDownloader", _FakeDownloader)

    downloaded = fld.download_flylight_images("R10A06", output_dir=str(tmp_path))
    assert downloaded == [Path("downloaded.png")]

    listed = fld.list_flylight_files("R10A06")
    assert len(listed) == 1
    assert len(_FakeDownloader.instances) == 2


# ---------------------------------------------------------------------------
# Appended coverage batch: boto3 discovery, auto-discover init, VT page
# parsing, R-line branches, generic S3 early-exit, simple-mode fallback,
# download flows, image summaries and search_lines edge branches.
# ---------------------------------------------------------------------------


def test_discover_s3_collections_boto3(monkeypatch, isolated_collection_globals):
    class _FakeClient:
        def list_objects_v2(self, **kwargs):
            return {"CommonPrefixes": [
                {"Prefix": "Gen1/"},
                {"Prefix": "content/"},
                {"Prefix": "Other Collection/"},
            ]}

    class _FakeBoto3:
        @staticmethod
        def client(*args, **kwargs):
            return _FakeClient()

    monkeypatch.setattr(fld, "HAS_BOTO3", True)
    monkeypatch.setattr(fld, "boto3", _FakeBoto3)
    monkeypatch.setattr(fld, "Config", lambda **kw: object())
    monkeypatch.setattr(fld, "UNSIGNED", "UNSIGNED")

    assert discover_s3_collections(verbose=True) == ["Gen1", "Other Collection"]


def test_discover_s3_collections_boto3_error_falls_back(monkeypatch, isolated_collection_globals):
    class _BoomBoto3:
        @staticmethod
        def client(*args, **kwargs):
            raise RuntimeError("no s3 access")

    monkeypatch.setattr(fld, "HAS_BOTO3", True)
    monkeypatch.setattr(fld, "boto3", _BoomBoto3)
    monkeypatch.setattr(fld, "Config", lambda **kw: object())
    monkeypatch.setattr(fld, "UNSIGNED", "UNSIGNED")
    monkeypatch.setattr(fld.urllib.request, "urlopen",
                        lambda *a, **k: _Response(_DISCOVER_XML))

    assert discover_s3_collections(verbose=True) == ["Gen1", "Split-GAL4 Omnibus Broad"]


def test_auto_discover_updates_categories(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fld, "discover_s3_collections",
                        lambda verbose=False: ["BrandNewColl"])
    monkeypatch.setattr(fld, "update_collection_categories",
                        lambda names: calls.append(list(names)))
    FlyLightDownloader(output_dir=str(tmp_path / "out"), auto_discover=True,
                       use_boto3=False, verbose=False)
    assert calls == [["BrandNewColl"]]

    # empty discovery result -> no category update
    calls.clear()
    monkeypatch.setattr(fld, "discover_s3_collections", lambda verbose=False: [])
    FlyLightDownloader(output_dir=str(tmp_path / "out"), auto_discover=True,
                       use_boto3=False, verbose=False)
    assert calls == []


_VT_PAGE_HTML = (
    '<a href="download.cgi?id=111&sid=abcdef12">session</a>'
    "<h3>Brain</h3>"
    '<img src="https://flimg.janelia.org/flylight-image/external-data/adult/secdata/'
    'projections/20200101/VT000770_custombrain/VT000770_x.jpg">'
    '<a href="download.cgi?id=222">other</a>'
    "<h3>Misc</h3>"
    '<img src="https://flimg.janelia.org/flylight-image/external-data/adult/secdata/'
    'projections/20200101/VT000770_miscz/VT000770_y.jpg">'
)


def test_parse_vt_page_region_inference(tmp_path, monkeypatch):
    d = _downloader(tmp_path)
    monkeypatch.setattr(fld.urllib.request, "urlopen",
                        lambda *a, **k: _Response(_VT_PAGE_HTML))
    samples, sid = d._parse_vt_page("VT000770")
    assert sid == "abcdef12"
    assert [s.region for s in samples] == ["brain", "unknown"]
    assert [s.sample_id for s in samples] == ["111", "222"]


def test_get_vt_files_cache_miss_then_hit(tmp_path, monkeypatch):
    d = _downloader(tmp_path)
    sample = VTSampleInfo(sample_id="1", line_name="VT000770", region="brain",
                          date="20200101", sample_path="VT000770-fA00b")
    calls = []

    def fake_parse(name):
        calls.append(name)
        return [sample], "sid123"

    monkeypatch.setattr(d, "_parse_vt_page", fake_parse)
    files = d._get_vt_files("VT000770")
    assert calls == ["VT000770"]
    assert files and all(f.source == "http" for f in files)
    # second call must hit the cache (no extra parse)
    d._get_vt_files("VT000770")
    assert calls == ["VT000770"]


_RLINE_HTML = (
    '<img src="https://flimg.janelia.org/flylight-image/external-data/adult/secdata/'
    'projections/20200101/R78H08custom/R78H08_pattern.jpg">'
)


def test_get_r_line_files_unknown_region(tmp_path, monkeypatch):
    d = _downloader(tmp_path)
    monkeypatch.setattr(fld.urllib.request, "urlopen",
                        lambda *a, **k: _Response(_RLINE_HTML))
    files = d._get_r_line_files("R78H08")
    assert len(files) == 1
    assert "/unknown/" in files[0].key
    assert files[0].collection == "Gen1 GAL4"


def test_verify_vt_file_exists_error(tmp_path, monkeypatch):
    d = _downloader(tmp_path)
    monkeypatch.setattr(
        fld.urllib.request, "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError("boom")))
    f = _file("VT GAL4/VT1/brain/x.jpg", source="http", http_url="https://x/y.jpg")
    assert d._verify_vt_file_exists(f) is False


def test_list_vt_files_mcfo_merged(tmp_path, monkeypatch):
    d = _downloader(tmp_path)
    gal4 = _file("VT GAL4/VT000770/brain/a.jpg", source="http",
                 http_url="https://x/a.jpg", line_name="VT000770")
    mcfo = _file("Gen1 MCFO/VT000770/brain/b.jpg", source="http",
                 http_url="https://x/b.jpg", line_name="VT000770",
                 collection="Gen1 MCFO")
    monkeypatch.setattr(d, "_get_vt_files", lambda name: [gal4])
    monkeypatch.setattr(d, "_get_vt_mcfo_files", lambda name: [mcfo])
    files = d.list_vt_files("VT000770", verify=False, include_mcfo=True)
    assert files == [gal4, mcfo]


def test_list_bucket_http_short_keys(tmp_path, monkeypatch):
    d = _downloader(tmp_path)
    xml = (
        '<?xml version="1.0"?>'
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        "<Contents><Key>root.png</Key><Size>1</Size>"
        "<LastModified>now</LastModified></Contents>"
        "<Contents><Key>Coll/x.png</Key><Size>2</Size>"
        "<LastModified>now</LastModified></Contents>"
        "</ListBucketResult>"
    )
    monkeypatch.setattr(fld.urllib.request, "urlopen",
                        lambda *a, **k: _Response(xml))
    files = d._list_bucket_http("prefix/")
    assert [f.line_name for f in files] == ["", ""]
    assert files[0].collection == ""
    assert files[1].collection == "Coll"


def test_r_line_boto3_branches(tmp_path, monkeypatch):
    # GAL4/LEXA R-line: CDM lookup via boto3
    d = _downloader(tmp_path, collections=["Gen1", "Gen1 MCFO"],
                    collection_category="GAL4/LEXA")
    d._s3_client = object()
    calls = []

    def fake_boto3_list(prefix, *args):
        calls.append(prefix)
        if prefix == "Gen1/CDM/R78H08/":
            return [_file("Gen1/CDM/R78H08/x.png", line_name="R78H08")]
        return []

    monkeypatch.setattr(d, "_list_bucket_boto3", fake_boto3_list)
    monkeypatch.setattr(d, "_get_r_line_files", lambda name: [])
    files = d.list_files("R78H08", use_cache=False)
    assert calls == ["Gen1/CDM/R78H08/"]
    assert len(files) == 1

    # RawImages R-line: non-MCFO collections skipped, MCFO listed via boto3
    d2 = _downloader(tmp_path, collections=["Gen1", "Gen1 MCFO"],
                     collection_category="RawImages")
    d2._s3_client = object()
    calls2 = []

    def fake_boto3_list2(prefix, *args):
        calls2.append(prefix)
        return [_file("Gen1 MCFO/R78H08/raw.lsm", line_name="R78H08",
                      collection="Gen1 MCFO")]

    monkeypatch.setattr(d2, "_list_bucket_boto3", fake_boto3_list2)
    files2 = d2.list_files("R78H08", use_cache=False)
    assert calls2 == ["Gen1 MCFO/R78H08/"]
    assert len(files2) == 1


def test_generic_s3_boto3_and_early_exit(tmp_path, monkeypatch):
    # direct hit via boto3, then early-exit break (>3 collections, no MCFO ahead)
    d = _downloader(tmp_path, collections=["CollA", "CollB", "CollC", "CollD"],
                    simple_mode=False)
    d._s3_client = object()
    calls = []

    def fake_list(prefix, *args):
        calls.append(prefix)
        if prefix == "CollA/SS00001/":
            return [_file("CollA/SS00001/x.png", collection="CollA",
                          line_name="SS00001")]
        return []

    monkeypatch.setattr(d, "_list_bucket_boto3", fake_list)
    files = d.list_files("SS00001", use_cache=False)
    assert calls == ["CollA/SS00001/"]
    assert len(files) == 1

    # MCFO collection ahead -> keep searching (continue), then break
    d2 = _downloader(tmp_path,
                     collections=["CollA", "Gen1 MCFO", "CollC", "CollD"],
                     simple_mode=False)
    d2._s3_client = object()
    calls2 = []

    def fake_list2(prefix, *args):
        calls2.append(prefix)
        if prefix in ("CollA/SS00001/", "Gen1 MCFO/SS00001/"):
            return [_file(prefix + "x.png", collection=prefix.split("/")[0],
                          line_name="SS00001")]
        return []

    monkeypatch.setattr(d2, "_list_bucket_boto3", fake_list2)
    files2 = d2.list_files("SS00001", use_cache=False)
    assert calls2 == ["CollA/SS00001/", "Gen1 MCFO/SS00001/"]
    assert len(files2) == 2

    # CDM fallback via boto3 when the direct path is empty
    d3 = _downloader(tmp_path, collections=["CollX"])
    d3._s3_client = object()
    calls3 = []

    def fake_list3(prefix, *args):
        calls3.append(prefix)
        if prefix == "CollX/CDM/SS00001/":
            return [_file("CollX/CDM/SS00001/y.png", collection="CollX",
                          line_name="SS00001")]
        return []

    monkeypatch.setattr(d3, "_list_bucket_boto3", fake_list3)
    files3 = d3.list_files("SS00001", use_cache=False)
    assert calls3 == ["CollX/SS00001/", "CollX/CDM/SS00001/"]
    assert len(files3) == 1


def test_simple_mode_fallback_chain(tmp_path, monkeypatch):
    d = _downloader(tmp_path, collections=["Gen1"], simple_mode=True,
                    simple_mode_min_files=2, automatic_fallback=False,
                    verbose=True)
    line_files = {
        "SS00001": [],  # empty pre-filter list -> skipped entirely
        "SS00002": [_file(f"Gen1/SS00002/mc{i}_multichannel_mip.png",
                          line_name="SS00002") for i in range(3)],
        "SS00003": [_file("Gen1/SS00003/a_mip.png", line_name="SS00003"),
                     _file("Gen1/SS00003/b.png", line_name="SS00003"),
                     _file("Gen1/SS00003/c.png", line_name="SS00003")],
    }
    monkeypatch.setattr(d, "list_files",
                        lambda name, use_cache=True: line_files[name])
    monkeypatch.setattr(d, "apply_simple_mode_filter", lambda files: [])

    result = d.get_filtered_files(["SS00001", "SS00002", "SS00003"])
    names = [f.filename for f in result]
    # SS00002: two multichannel_mip files (break once min reached)
    assert names[:2] == ["mc0_multichannel_mip.png", "mc1_multichannel_mip.png"]
    # SS00003: one _mip file, then last-resort first other file
    assert names[2:] == ["a_mip.png", "b.png"]


def test_download_file_http_ssl_exhaustion_and_zero_retries(tmp_path, monkeypatch):
    d = _downloader(tmp_path)
    f = _file("Gen1/R1/x.png", source="http", http_url="https://cdn/x.png")
    monkeypatch.setattr(fld.urllib.request, "urlretrieve",
                        lambda *a, **k: (_ for _ in ()).throw(ssl.SSLError("eof")))
    monkeypatch.setattr("time.sleep", lambda s: None)
    out = tmp_path / "x.png"
    assert d._download_file_http(f, out, max_retries=2) is False
    # zero retries -> loop body never runs, tail return False
    assert d._download_file_http(f, out, max_retries=0) is False


def test_download_file_dispatch(tmp_path, monkeypatch):
    d = _downloader(tmp_path)
    f = _file("Gen1/SS00001/x.png", line_name="SS00001")

    class _FakeS3:
        def download_file(self, bucket, key, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("data")

    d._s3_client = _FakeS3()
    result = d.download_file(f)
    assert result is not None and result.exists()

    # no S3 client -> HTTP fallback dispatch
    d2 = _downloader(tmp_path / "b")
    monkeypatch.setattr(d2, "_download_file_http", lambda file, path: True)
    result2 = d2.download_file(f, flat_structure=True)
    assert result2 == Path(d2.output_dir) / "SS00001" / "x.png"

    # failed download -> None
    d3 = _downloader(tmp_path / "c")
    monkeypatch.setattr(d3, "_download_file_http", lambda file, path: False)
    assert d3.download_file(f) is None


def test_download_flows(tmp_path, monkeypatch):
    # comma-separated input, <=3 joined name, timestamped folder,
    # parallel download with callback
    d = _downloader(tmp_path, max_workers=2, verbose="pbar")
    files = [_file("Gen1/SS1/a.png", line_name="SS1"),
             _file("Gen1/SS2/b.png", line_name="SS2")]
    monkeypatch.setattr(d, "get_filtered_files",
                        lambda name, max_files_per_line=None: files)
    monkeypatch.setattr(d, "download_file",
                        lambda f, out, flat: Path(out) / f.filename)
    events = []
    downloaded = d.download(
        "SS1, SS2",
        on_file_downloaded=lambda p, n: events.append((str(p), n)),
        add_timestamp=True)
    assert len(downloaded) == 2
    assert all(name == "SS1_SS2" for _, name in events)
    assert all("flylight-downloads_SS1_SS2_" in str(p) for p in downloaded)

    # sequential download with pbar + summary generation, >3 lines _etc name
    d2 = _downloader(tmp_path / "seq", max_workers=1, verbose="pbar")
    monkeypatch.setattr(d2, "get_filtered_files",
                        lambda name, max_files_per_line=None: files)
    monkeypatch.setattr(d2, "download_file",
                        lambda f, out, flat: Path(out) / f.filename)
    summary_calls = []
    monkeypatch.setattr(d2, "_generate_image_summary",
                        lambda **kw: summary_calls.append(kw))
    downloaded2 = d2.download(["L1", "L2", "L3", "L4"],
                              generate_summary="pdf", add_timestamp=False)
    assert len(downloaded2) == 2
    assert summary_calls and summary_calls[0]["line_name"] == "L1_L2_L3_etc4"


def test_generate_image_summary_formats(tmp_path, monkeypatch):
    import sys
    import types

    d = _downloader(tmp_path)
    img = tmp_path / "a.png"
    img.write_text("x")
    files = [img]

    # pdf success via fake neuronbridge_finder
    nbf = types.ModuleType("neuronbridge_finder")
    pdf_calls = []
    nbf.create_image_pdf = lambda **kw: pdf_calls.append(kw)
    monkeypatch.setitem(sys.modules, "neuronbridge_finder", nbf)
    d._generate_image_summary(files, tmp_path, "L", "T", "pdf")
    assert len(pdf_calls) == 1

    # list input + generic Exception during pdf creation
    nbf.create_image_pdf = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
    d._generate_image_summary(files, tmp_path, "L", "T", ["pdf"])
    assert len(pdf_calls) == 1

    # pdf ImportError branch (module entry halted)
    monkeypatch.setitem(sys.modules, "neuronbridge_finder", None)
    d._generate_image_summary(files, tmp_path, "L", "T", "pdf")

    # pptx success via fake utils.report_utils; True -> ['pdf', 'pptx']
    ru = types.ModuleType("utils.report_utils")
    pptx_calls = []
    ru.img2pptx = lambda **kw: pptx_calls.append(kw)
    monkeypatch.setitem(sys.modules, "utils.report_utils", ru)
    d._generate_image_summary(files, tmp_path, "L", "T", True)
    assert len(pptx_calls) == 1

    # pptx ImportError branch (both import paths halted)
    monkeypatch.setitem(sys.modules, "utils.report_utils", None)
    monkeypatch.setitem(sys.modules, "src.utils.report_utils", None)
    d._generate_image_summary(files, tmp_path, "L", "T", "pptx")

    # pptx generic Exception branch
    ru2 = types.ModuleType("utils.report_utils")
    ru2.img2pptx = lambda **kw: (_ for _ in ()).throw(RuntimeError("pptx boom"))
    monkeypatch.setitem(sys.modules, "utils.report_utils", ru2)
    d._generate_image_summary(files, tmp_path, "L", "T", "pptx")

    # falsy summary value -> formats fall through and return
    d._generate_image_summary(files, tmp_path, "L", "T", False)


def test_get_metadata_non_janelia_success(tmp_path, monkeypatch):
    d = _downloader(tmp_path)
    files = [_file("Coll/VT1/VT1-meta.json", source="http",
                   http_url="https://other.example.com/meta.json",
                   line_name="VT000770")]
    monkeypatch.setattr(d, "get_filtered_files", lambda name, **k: files)
    monkeypatch.setattr(fld.urllib.request, "urlopen",
                        lambda *a, **k: _Response(json.dumps({"k": 1})))
    assert d.get_metadata("VT000770") == [{"k": 1}]


def test_search_lines_error_and_walk_branches(tmp_path, monkeypatch):
    # boto3 client errors everywhere -> caught in _collect and level-1 walk
    class _ErrS3:
        def list_objects_v2(self, **kw):
            raise RuntimeError("s3 down")

    d = _downloader(tmp_path, collections=["Gen1"])
    d._s3_client = _ErrS3()
    assert d.search_lines("R78") == []

    # flat collection (>30 level-1 prefixes) is skipped during the walk
    class _FlatS3:
        def list_objects_v2(self, **kw):
            if kw.get("Prefix") == "Gen1/":
                return {"CommonPrefixes": [
                    {"Prefix": f"Gen1/sub{i}/"} for i in range(31)]}
            return {}

    d2 = _downloader(tmp_path, collections=["Gen1"])
    d2._s3_client = _FlatS3()
    assert d2.search_lines("ZZZ") == []

    # no S3 client -> prefix _collect uses HTTP, walk skipped entirely
    d3 = _downloader(tmp_path, collections=["Gen1"])
    monkeypatch.setattr(d3, "_list_bucket_http", lambda prefix, *a: [])
    assert d3.search_lines("ZZZ") == []
