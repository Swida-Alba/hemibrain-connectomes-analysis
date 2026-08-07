#!/usr/bin/env python
"""
Tests for the NeuronBridge image pipeline:

- ``_collect_line_images`` handles every layout the downloaders produce
  (per-line subdirs, flat files, and the 'both' source subfolders), so the
  PDF/PPTX summary never reports "No images found" when images are nested.
- ``create_image_pdf`` builds a real PDF from the 'both' layout.
- ``_download_neuronbridge_images`` scans lines in PARALLEL (every
  get_lm_images() is a network call) with a streamed progress bar, and
  downloads files with a bounded ThreadPoolExecutor honoring max_workers.
"""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neuronbridge_finder import _collect_line_images  # noqa: E402
from neuronbridge_finder import create_image_pdf  # noqa: E402
from neuronbridge_finder import NeuronBridgeFinder  # noqa: E402


def _make_image(path: Path, fmt: str = "png") -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (10, 20, 30)).save(path, format=fmt.upper())
    return path


# ---------------------------------------------------------------------------
# _collect_line_images
# ---------------------------------------------------------------------------

class TestCollectLineImages:
    def test_per_line_subdirectories(self, tmp_path):
        _make_image(tmp_path / "VT001" / "a.png")
        _make_image(tmp_path / "VT001" / "b.jpg")
        _make_image(tmp_path / "SS002" / "c.png")
        result = _collect_line_images(tmp_path)
        assert set(result) == {"VT001", "SS002"}
        assert [p.name for p in result["VT001"]] == ["a.png", "b.jpg"]

    def test_flat_files_grouped_by_prefix(self, tmp_path):
        _make_image(tmp_path / "VT001-20x-multichannel.png")
        _make_image(tmp_path / "VT001-total.png")
        _make_image(tmp_path / "GMR002-other.png")
        result = _collect_line_images(tmp_path)
        assert set(result) == {"VT001", "GMR002"}
        assert len(result["VT001"]) == 2

    def test_both_source_subfolders_are_merged(self, tmp_path):
        # download_source='both' layout: images/neuronbridge/<line>/*.png
        # and flat files in images/flylight/
        nb_line = _make_image(tmp_path / "neuronbridge" / "VT001" / "cdm.png")
        fl_flat = _make_image(tmp_path / "flylight" / "VT001-20x-multichannel.png")
        fl_other = _make_image(tmp_path / "flylight" / "SS002-total.png")
        result = _collect_line_images(tmp_path)
        assert set(result) == {"VT001", "SS002"}
        # both sources merged under the same line, no 'neuronbridge'/'flylight' keys
        assert {p.name for p in result["VT001"]} == {"cdm.png", "VT001-20x-multichannel.png"}
        assert result["SS002"] == [fl_other]
        assert nb_line in result["VT001"] and fl_flat in result["VT001"]

    def test_unknown_flat_file_without_dash(self, tmp_path):
        _make_image(tmp_path / "neuronbridge" / "random.png")
        result = _collect_line_images(tmp_path)
        assert result["Unknown"] == [tmp_path / "neuronbridge" / "random.png"]

    def test_ignores_non_image_files(self, tmp_path):
        _make_image(tmp_path / "VT001" / "a.png")
        (tmp_path / "VT001" / "notes.txt").write_text("x")
        (tmp_path / "images_summary.pdf").write_text("x")
        result = _collect_line_images(tmp_path)
        assert result == {"VT001": [tmp_path / "VT001" / "a.png"]}


# ---------------------------------------------------------------------------
# create_image_pdf with the 'both' layout (the reported bug)
# ---------------------------------------------------------------------------

class TestCreateImagePdf:
    def test_pdf_generated_from_both_layout(self, tmp_path):
        images = tmp_path / "images"
        _make_image(images / "neuronbridge" / "VT001" / "cdm.png")
        _make_image(images / "neuronbridge" / "VT001" / "mip.png")
        _make_image(images / "flylight" / "VT002-20x-multichannel.png")
        _make_image(images / "flylight" / "SS003-total.png")

        out_pdf = tmp_path / "images_summary.pdf"
        result = create_image_pdf(
            images_dir=str(images),
            output_pdf=str(out_pdf),
            images_per_page=(3, 2),
            verbose=False,
        )
        assert result == str(out_pdf)
        assert out_pdf.exists() and out_pdf.stat().st_size > 0

    def test_empty_dir_returns_none(self, tmp_path):
        result = create_image_pdf(str(tmp_path / "nope"), verbose=False)
        assert result is None


# ---------------------------------------------------------------------------
# _download_neuronbridge_images: parallel scan + bounded parallel download
# ---------------------------------------------------------------------------

def _fake_lm_image(path: str):
    return SimpleNamespace(files=SimpleNamespace(
        CDM=path, SignalMip=None, SignalMipMasked=None,
    ))


class TestNeuronBridgeDownloaderParallelism:
    def _make_finder(self, n_lines: int = 6, latency: float = 0.2):
        finder = object.__new__(NeuronBridgeFinder)
        finder.max_workers = n_lines
        finder.verbose = False

        class FakeClient:
            def get_lm_images(self, line_name):
                time.sleep(latency)  # simulate a network call per line
                return [_fake_lm_image(f"prefix/{line_name}.png")]

        finder._client = FakeClient()
        return finder

    def test_scan_runs_in_parallel(self, tmp_path, monkeypatch):
        """Scanning N lines must take ~1x latency, not N x latency."""
        n_lines = 6
        latency = 0.2
        finder = self._make_finder(n_lines, latency)

        # Patch the actual file download so no network is touched; the
        # parallel SCAN is what this test verifies.
        def fake_urlretrieve(url, local_path):
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            Path(local_path).write_bytes(b"fake")

        monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

        start = time.monotonic()
        downloaded = finder._download_neuronbridge_images(
            lines=[f"L{i}" for i in range(n_lines)],
            output_dir=str(tmp_path / "out"),
            formats="png",
            image_types="cdm",
            max_files=None,
            verbose=False,
        )
        elapsed = time.monotonic() - start

        assert len(downloaded) == n_lines
        # Parallel: ~latency + download time. Sequential would be n*latency.
        assert elapsed < latency * (n_lines - 1), (
            f"scan appears sequential: {elapsed:.2f}s for {n_lines} lines "
            f"at {latency}s each"
        )

    def test_scan_reports_progress_bar_when_verbose(self, tmp_path, monkeypatch):
        """The scan phase streams a tqdm progress bar (the UI shows it live)."""
        finder = self._make_finder(n_lines=3, latency=0.01)

        def fake_urlretrieve(url, local_path):
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            Path(local_path).write_bytes(b"fake")

        monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)
        captured = []

        import tqdm as tqdm_mod

        real_tqdm = tqdm_mod.tqdm

        class RecordingTqdm(real_tqdm):
            def __init__(self, *args, **kwargs):
                captured.append(kwargs.get("desc", ""))
                super().__init__(*args, **kwargs)

        monkeypatch.setattr("neuronbridge_finder.tqdm", RecordingTqdm)
        finder._download_neuronbridge_images(
            lines=[f"L{i}" for i in range(3)],
            output_dir=str(tmp_path / "out"),
            formats="png",
            image_types="cdm",
            max_files=None,
            verbose=True,
        )
        # One bar for the scan phase, one for the download phase
        assert any("Scanning" in d for d in captured), f"no scan bar: {captured}"
        assert any("Downloading" in d for d in captured), f"no download bar: {captured}"
