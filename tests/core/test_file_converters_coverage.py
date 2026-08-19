"""Hermetic coverage tests for BANC_file_converter and FAFB_file_converter.

All raw inputs are tiny synthetic CSV/CSV.gz/ZIP files written into tmp_path,
mirroring the column shapes the converters expect.  The only parallelism in
the FAFB skeleton pipeline (ProcessPoolExecutor) is replaced with a serial
in-process executor so no child processes are spawned.
"""

import concurrent.futures
import gzip
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import BANC_file_converter as banc  # noqa: E402
import FAFB_file_converter as fafb  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_gz(path: Path, text: str) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)
    return path


def _read_pq(path):
    assert Path(path).exists(), f"missing output: {path}"
    return pd.read_parquet(path)


_BANC_NEURONS_CSV = (
    "Root ID,Primary Cell Type,Super Class,Flow,Nerve,Hemilineage,Predicted NT type,Soma side\n"
    "102,Beta,Sensory,Output,nerve1,HL1,GABA,L\n"
    "101,aMe12,Clock,Input,nerve2,HL2,ACh,R\n"
    "101,aMe12,Clock,Input,nerve2,HL2,ACh,R\n"
    "103,,Motor,Mixed,nerve3,HL3,Glu,M\n"
)

_BANC_CONNECTIONS_CSV = (
    "pre_root_id,post_root_id,neuropil,syn_count,nt_type\n"
    "101,102,A,3,GABA\n"
    "101,102,B,2,GABA\n"
    "101,102,,1,GABA\n"
    "102,103,A,5,ACh\n"
)


# ===========================================================================
# BANC_file_converter
# ===========================================================================

class TestBancNeurons:
    def test_existing_output_short_circuits(self, tmp_path, capsys):
        save = tmp_path / "neurons.parquet"
        save.write_bytes(b"sentinel")
        assert banc.process_neurons_to_parquet(
            str(tmp_path / "missing.csv"), str(save)) is True
        assert "Found existing converted file" in capsys.readouterr().out
        assert save.read_bytes() == b"sentinel"

    def test_missing_input_returns_false(self, tmp_path):
        assert banc.process_neurons_to_parquet(
            str(tmp_path / "nope.csv"), str(tmp_path / "out.parquet")) is False

    def test_conversion_renames_dedupes_and_sorts(self, tmp_path):
        read_path = tmp_path / "neurons.csv"
        read_path.write_text(_BANC_NEURONS_CSV, encoding="utf-8")
        save = tmp_path / "neurons.parquet"
        save_csv = tmp_path / "neurons_out.csv"

        assert banc.process_neurons_to_parquet(
            str(read_path), str(save), save_csv_path=str(save_csv)) is True

        df = _read_pq(save)
        assert df["bodyId"].tolist() == ["101", "102", "103"]  # sorted + dedup
        assert str(df["bodyId"].dtype) == "string"
        assert df.loc[df["bodyId"] == "103", "type"].iloc[0] == "Unknown"
        assert df.loc[df["bodyId"] == "101", "type"].iloc[0] == "aMe12"
        assert (df["instance"] == df["type"]).all()
        assert (df["post"] == 0).all()
        assert "super_class" in df.columns
        assert "nt_type" in df.columns
        # Standard columns come first in the documented order.
        assert list(df.columns[:4]) == ["bodyId", "type", "instance", "post"]
        assert save_csv.exists()

    def test_invalid_input_returns_false(self, tmp_path):
        read_path = tmp_path / "broken.csv"
        read_path.mkdir()  # a directory makes read_csv raise
        assert banc.process_neurons_to_parquet(
            str(read_path), str(tmp_path / "out.parquet")) is False


class TestBancConnections:
    def test_existing_output_short_circuits(self, tmp_path):
        save = tmp_path / "conn.parquet"
        save.write_bytes(b"sentinel")
        assert banc.process_connections_to_parquet(
            str(tmp_path / "missing.csv"), str(save)) is True

    def test_missing_input_returns_false(self, tmp_path):
        assert banc.process_connections_to_parquet(
            str(tmp_path / "nope.csv"), str(tmp_path / "out.parquet")) is False

    def test_aggregates_weights_and_rois(self, tmp_path):
        read_path = tmp_path / "connections.csv"
        read_path.write_text(_BANC_CONNECTIONS_CSV, encoding="utf-8")
        save = tmp_path / "conn.parquet"

        assert banc.process_connections_to_parquet(str(read_path), str(save)) is True

        df = _read_pq(save)
        assert df[["bodyId_pre", "bodyId_post"]].values.tolist() == [
            ["101", "102"], ["102", "103"]]
        row = df[df["bodyId_pre"] == "101"].iloc[0]
        assert row["weight"] == 6          # 3 + 2 + 1 summed across ROIs
        assert row["roi"] == "A|B"         # null ROI dropped, sorted join
        assert df[df["bodyId_pre"] == "102"].iloc[0]["weight"] == 5

    def test_without_roi_column_defaults_to_wholebrain(self, tmp_path):
        read_path = tmp_path / "connections.csv"
        read_path.write_text(
            "pre_root_id,post_root_id,syn_count\n"
            "101,102,3\n101,102,4\n",
            encoding="utf-8")
        save = tmp_path / "conn.parquet"

        assert banc.process_connections_to_parquet(str(read_path), str(save)) is True

        df = _read_pq(save)
        assert df.iloc[0]["weight"] == 7
        assert df.iloc[0]["roi"] == "WholeBrain"

    def test_missing_weight_column_returns_false(self, tmp_path):
        read_path = tmp_path / "connections.csv"
        read_path.write_text(
            "pre_root_id,post_root_id,neuropil\n101,102,A\n", encoding="utf-8")
        assert banc.process_connections_to_parquet(
            str(read_path), str(tmp_path / "out.parquet")) is False


class TestBancPostCounts:
    def _write_pair(self, tmp_path, fmt):
        neurons = pd.DataFrame({
            "bodyId": ["101", "102", "103"],
            "type": ["A", "B", "C"],
            "post": [0, 0, 0],
        })
        conn = pd.DataFrame({
            "bodyId_pre": ["101", "102"],
            "bodyId_post": ["102", "103"],
            "weight": [5, 7],
        })
        if fmt == "parquet":
            neuron_path = tmp_path / "neurons.parquet"
            conn_path = tmp_path / "conn.parquet"
            neurons.to_parquet(neuron_path, index=False)
            conn.to_parquet(conn_path, index=False)
        else:
            neuron_path = tmp_path / "neurons.csv"
            conn_path = tmp_path / "conn.csv"
            neurons.to_csv(neuron_path, index=False)
            conn.to_csv(conn_path, index=False)
        return neuron_path, conn_path

    def test_updates_parquet_and_writes_csv_copy(self, tmp_path):
        neuron_path, conn_path = self._write_pair(tmp_path, "parquet")
        csv_copy = tmp_path / "neurons_updated.csv"

        assert banc.update_neuron_post_counts(
            str(neuron_path), str(conn_path), save_csv_path=str(csv_copy)) is True

        df = _read_pq(neuron_path)
        posts = dict(zip(df["bodyId"], df["post"]))
        assert posts == {"101": 0, "102": 5, "103": 7}
        assert csv_copy.exists()

    def test_updates_csv_paths(self, tmp_path):
        neuron_path, conn_path = self._write_pair(tmp_path, "csv")
        assert banc.update_neuron_post_counts(
            str(neuron_path), str(conn_path)) is True
        df = pd.read_csv(neuron_path, dtype={"bodyId": str})
        posts = dict(zip(df["bodyId"], df["post"]))
        assert posts == {"101": 0, "102": 5, "103": 7}

    def test_missing_connections_returns_false(self, tmp_path):
        neuron_path, _ = self._write_pair(tmp_path, "parquet")
        assert banc.update_neuron_post_counts(
            str(neuron_path), str(tmp_path / "missing.parquet")) is False


class TestBancEnsure:
    def _write_sources(self, tmp_path):
        downloads = tmp_path / "downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        _write_gz(downloads / "neurons.csv.gz", _BANC_NEURONS_CSV)
        _write_gz(downloads / "connections_princeton.csv.gz", _BANC_CONNECTIONS_CSV)

    def test_full_pipeline_converts_and_updates_posts(self, tmp_path, capsys):
        dataset_dir = tmp_path / "flywire_BANC_v626"
        self._write_sources(dataset_dir)

        assert banc.ensure_banc_data("flywire_BANC_v626", str(dataset_dir)) is True

        neuron_pq = dataset_dir / "flywire_BANC_v626_allneurons_neuron_df.parquet"
        conn_pq = dataset_dir / "flywire_BANC_v626_merged_connections.parquet"
        assert neuron_pq.exists() and conn_pq.exists()
        assert (dataset_dir / "flywire_BANC_v626_allneurons_neuron_df.csv").exists()

        df = _read_pq(neuron_pq)
        posts = dict(zip(df["bodyId"], df["post"]))
        assert posts == {"101": 0, "102": 6, "103": 5}

    def test_second_run_reuses_existing_outputs(self, tmp_path, capsys):
        dataset_dir = tmp_path / "flywire_BANC_v626"
        self._write_sources(dataset_dir)
        assert banc.ensure_banc_data("flywire_BANC_v626", str(dataset_dir)) is True
        capsys.readouterr()

        assert banc.ensure_banc_data("flywire_BANC_v626", str(dataset_dir)) is True
        out = capsys.readouterr().out
        assert "Found existing neurons" in out
        assert "Found existing connections" in out
        assert "Post counts already populated." in out

    def test_missing_sources_returns_false(self, tmp_path, capsys):
        dataset_dir = tmp_path / "flywire_BANC_v626"
        assert banc.ensure_banc_data("flywire_BANC_v626", str(dataset_dir)) is False
        out = capsys.readouterr().out
        assert "MISSING CRITICAL FILES" in out
        assert (dataset_dir / "downloads").is_dir()


# ===========================================================================
# FAFB_file_converter
# ===========================================================================

_SWC_100 = (
    "# skeleton for body 100\n"
    "1 0 10.6 20.4 30.2 1.2 -1\n"
    "2 3 11 21 31 2.0 1\n"
    "short line\n"
)
_SWC_200 = "1 1 5.0 6.0 7.0 0.5 -1\n"


def _make_swc_zip(path: Path, entries):
    with zipfile.ZipFile(path, "w") as handle:
        for name, content in entries.items():
            handle.writestr(name, content)
    return path


class TestFafbSwcParsing:
    def test_parse_swc_batch_reads_nodes(self, tmp_path):
        zip_path = _make_swc_zip(tmp_path / "skeletons.zip", {
            "100.swc": _SWC_100,
            "sub/200.swc": _SWC_200,
        })

        data = fafb._parse_swc_batch(
            str(zip_path), ["100.swc", "sub/200.swc", "missing.swc"])

        assert data["bodyId"] == ["100", "100", "200"]
        assert data["node_id"] == [1, 2, 1]
        assert data["x"][0] == pytest.approx(10.6)
        assert data["parent_id"] == [-1, 1, -1]

    def test_parse_swc_batch_bad_zip_returns_empty(self, tmp_path, capsys):
        bad = tmp_path / "not_a_zip.zip"
        bad.write_text("nope", encoding="utf-8")
        data = fafb._parse_swc_batch(str(bad), ["100.swc"])
        assert data["bodyId"] == []
        assert "Error in batch" in capsys.readouterr().out


_CLASSIFICATION_CSV = (
    "root_id,super_class,class,sub_class,side\n"
    "1,,Cls1,CellB,L\n"
    "2,Sens,,CellD,R\n"
    "3,SupC,,,L\n"
)


class TestFafbNeurons:
    def test_existing_output_short_circuits(self, tmp_path):
        save = tmp_path / "neurons.parquet"
        save.write_bytes(b"sentinel")
        assert fafb.process_neurons_to_parquet(
            str(tmp_path / "missing.csv"), str(save)) is True

    def test_missing_input_returns_false(self, tmp_path):
        assert fafb.process_neurons_to_parquet(
            str(tmp_path / "nope.csv"), str(tmp_path / "out.parquet")) is False

    def test_conversion_with_all_enrichments(self, tmp_path):
        downloads = tmp_path / "downloads"
        downloads.mkdir()
        class_path = downloads / "classification.csv"
        class_path.write_text(_CLASSIFICATION_CSV, encoding="utf-8")
        (downloads / "names.csv").write_text(
            "root_id,name,group\n1,InstA,0\n", encoding="utf-8")
        (downloads / "coordinates.csv").write_text(
            "root_id,supervoxel_id,x,y,z\n1,77,1.0,2.0,3.0\n2,78,4.0,5.0,6.0\n",
            encoding="utf-8")
        (downloads / "neurons.csv").write_text(
            "root_id,nt_type,group\n1,ACh,0\n2,GABA,0\n", encoding="utf-8")
        (downloads / "cell_stats.csv").write_text(
            "root_id,cable_length\n1,1000.5\n", encoding="utf-8")
        (downloads / "consolidated_cell_types.csv").write_text(
            "root_id,primary_type\n1,TypeA\n", encoding="utf-8")

        save = tmp_path / "neurons.parquet"
        enrichment = {
            "names": str(downloads / "names.csv"),
            "coordinates": str(downloads / "coordinates.csv"),
            "neurons": str(downloads / "neurons.csv"),
            "cell_stats": str(downloads / "cell_stats.csv"),
            "cell_types": str(downloads / "consolidated_cell_types.csv"),
        }
        assert fafb.process_neurons_to_parquet(
            str(class_path), str(save),
            save_csv_path=str(tmp_path / "neurons.csv"),
            enrichment_files=enrichment) is True

        df = _read_pq(save)
        assert df["bodyId"].tolist() == ["1", "2", "3"]
        types = dict(zip(df["bodyId"], df["type"]))
        # primary_type beats cell_type; cell_type beats super_class fallback.
        assert types == {"1": "TypeA", "2": "CellD", "3": "SupC"}
        instances = dict(zip(df["bodyId"], df["instance"]))
        assert instances == {"1": "InstA", "2": "CellD", "3": "SupC"}
        assert "group" not in df.columns
        assert "supervoxel_id" not in df.columns
        for column in ("x", "y", "z", "nt_type", "cable_length", "hemisphere"):
            assert column in df.columns
        assert (df["post"] == 0).all()
        assert (tmp_path / "neurons.csv").exists()

    def test_conversion_without_enrichment(self, tmp_path):
        class_path = tmp_path / "classification.csv"
        class_path.write_text(_CLASSIFICATION_CSV, encoding="utf-8")
        save = tmp_path / "neurons.parquet"

        assert fafb.process_neurons_to_parquet(str(class_path), str(save)) is True

        df = _read_pq(save)
        types = dict(zip(df["bodyId"], df["type"]))
        assert types == {"1": "CellB", "2": "CellD", "3": "SupC"}
        assert (df["instance"] == df["type"]).all()

    def test_conversion_dedupes_duplicate_root_ids(self, tmp_path):
        class_path = tmp_path / "classification.csv"
        class_path.write_text(
            "root_id,class\n1,X\n1,X\n2,Y\n", encoding="utf-8")
        save = tmp_path / "neurons.parquet"

        assert fafb.process_neurons_to_parquet(str(class_path), str(save)) is True

        df = _read_pq(save)
        assert df["bodyId"].tolist() == ["1", "2"]

    def test_processing_error_returns_false(self, tmp_path, capsys):
        read_path = tmp_path / "classification.csv"
        read_path.mkdir()  # a directory makes read_csv raise
        assert fafb.process_neurons_to_parquet(
            str(read_path), str(tmp_path / "out.parquet")) is False
        assert "Error processing neurons" in capsys.readouterr().out


_FAFB_CONNECTIONS_CSV = (
    "pre_root_id,post_root_id,neuropil,syn_count\n"
    "1,2,A,5\n"
    "1,2,B,3\n"
    "2,3,A,4\n"
)


class TestFafbConnections:
    def test_existing_output_short_circuits(self, tmp_path):
        save = tmp_path / "conn.parquet"
        save.write_bytes(b"sentinel")
        assert fafb.process_connections_to_parquet(
            str(tmp_path / "missing.csv.gz"), str(save)) is True

    def test_missing_input_returns_false(self, tmp_path):
        assert fafb.process_connections_to_parquet(
            str(tmp_path / "nope.csv.gz"), str(tmp_path / "out.parquet")) is False

    def test_aggregates_weights_and_rois(self, tmp_path):
        read_path = _write_gz(tmp_path / "connections.csv.gz", _FAFB_CONNECTIONS_CSV)
        save = tmp_path / "conn.parquet"

        assert fafb.process_connections_to_parquet(str(read_path), str(save)) is True

        df = _read_pq(save)
        assert df[["bodyId_pre", "bodyId_post"]].values.tolist() == [
            ["1", "2"], ["2", "3"]]
        row = df[df["bodyId_pre"] == "1"].iloc[0]
        assert row["weight"] == 8
        assert row["roi"] == "A|B"

    def test_without_roi_column_defaults_to_wholebrain(self, tmp_path):
        read_path = _write_gz(
            tmp_path / "connections.csv.gz",
            "pre_root_id,post_root_id,syn_count\n1,2,3\n")
        save = tmp_path / "conn.parquet"

        assert fafb.process_connections_to_parquet(str(read_path), str(save)) is True

        df = _read_pq(save)
        assert df.iloc[0]["weight"] == 3
        assert df.iloc[0]["roi"] == "WholeBrain"

    def test_corrupt_input_returns_false(self, tmp_path, capsys):
        read_path = tmp_path / "connections.csv.gz"
        read_path.write_bytes(b"not really gzip")
        assert fafb.process_connections_to_parquet(
            str(read_path), str(tmp_path / "out.parquet")) is False
        assert "Error processing connections" in capsys.readouterr().out


class TestFafbSynapseTable:
    def test_existing_output_short_circuits(self, tmp_path):
        save = tmp_path / "syn.parquet"
        save.write_bytes(b"sentinel")
        assert fafb.process_synapse_table_to_parquet(
            str(tmp_path / "missing.csv"), str(save)) is True

    def test_missing_input_returns_false(self, tmp_path):
        assert fafb.process_synapse_table_to_parquet(
            str(tmp_path / "nope.csv"), str(tmp_path / "out.parquet")) is False

    def test_missing_root_id_columns_returns_false(self, tmp_path, capsys):
        read_path = tmp_path / "syn.csv"
        read_path.write_text("a,b\n1,2\n", encoding="utf-8")
        assert fafb.process_synapse_table_to_parquet(
            str(read_path), str(tmp_path / "out.parquet")) is False
        assert "Could not find root_id columns" in capsys.readouterr().out

    def test_converts_and_fixes_short_ids(self, tmp_path):
        read_path = tmp_path / "syn.csv"
        read_path.write_text(
            "x,y,z,pre_root_id,post_root_id\n"
            "1,2,3,609749525,609749526\n"
            "4,5,6,609749527,609749525\n",
            encoding="utf-8")
        save = tmp_path / "syn.parquet"

        assert fafb.process_synapse_table_to_parquet(
            str(read_path), str(save), chunksize=1) is True

        df = _read_pq(save)
        assert len(df) == 2
        # 9-char short IDs are expanded to full FlyWire IDs and sorted.
        assert df["pre_root_id"].tolist() == [
            "720575940609749525", "720575940609749527"]

    def test_no_chunks_returns_false(self, tmp_path, monkeypatch, capsys):
        read_path = tmp_path / "syn.csv"
        read_path.write_text(
            "x,y,z,pre_root_id,post_root_id\n1,2,3,4,5\n", encoding="utf-8")

        class _EmptyReader:
            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

            def __iter__(self):
                return iter([])

        real_read_csv = pd.read_csv

        def fake_read_csv(path, *args, **kwargs):
            if kwargs.get("chunksize"):
                return _EmptyReader()
            return real_read_csv(path, *args, **kwargs)

        monkeypatch.setattr(fafb.pd, "read_csv", fake_read_csv)

        assert fafb.process_synapse_table_to_parquet(
            str(read_path), str(tmp_path / "out.parquet")) is False
        assert "No data found" in capsys.readouterr().out

    def test_corrupt_input_returns_false(self, tmp_path, capsys):
        read_path = tmp_path / "syn.csv.gz"
        read_path.write_bytes(b"not really gzip")
        assert fafb.process_synapse_table_to_parquet(
            str(read_path), str(tmp_path / "out.parquet")) is False
        assert "Error processing synapse table" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Skeleton pipeline (ProcessPoolExecutor replaced with a serial stand-in)
# ---------------------------------------------------------------------------

class _SerialExecutor:
    """In-process stand-in for ProcessPoolExecutor (no child processes)."""

    def __init__(self, max_workers=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def submit(self, fn, *args, **kwargs):
        future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001 - surfaced via result()
            future.set_exception(exc)
        return future


@pytest.fixture
def serial_pool(monkeypatch):
    """Keep the skeleton pipeline in-process: no child workers are spawned."""
    monkeypatch.setattr(
        fafb.concurrent.futures, "ProcessPoolExecutor", _SerialExecutor)


class TestFafbSkeletons:
    def test_existing_output_short_circuits(self, tmp_path):
        save = tmp_path / "sk.parquet"
        save.write_bytes(b"sentinel")
        assert fafb.process_skeletons_to_parquet(
            str(tmp_path / "missing.zip"), str(save)) is True

    def test_missing_zip_returns_false(self, tmp_path):
        assert fafb.process_skeletons_to_parquet(
            str(tmp_path / "nope.zip"), str(tmp_path / "out.parquet")) is False

    def test_zip_without_swc_returns_false(self, tmp_path, capsys):
        zip_path = _make_swc_zip(tmp_path / "skeletons.zip", {"readme.txt": "hi"})
        assert fafb.process_skeletons_to_parquet(
            str(zip_path), str(tmp_path / "out.parquet")) is False
        assert "No SWC files found" in capsys.readouterr().out

    def test_converts_swc_zip(self, tmp_path, serial_pool):
        zip_path = _make_swc_zip(tmp_path / "skeletons.zip", {
            "100.swc": _SWC_100,
            "200.swc": _SWC_200,
        })
        save = tmp_path / "sk.parquet"

        assert fafb.process_skeletons_to_parquet(
            str(zip_path), str(save), batch_size=1) is True

        df = _read_pq(save)
        # Batches land as separate row groups; order across batches is free.
        assert sorted(df["bodyId"].tolist()) == ["100", "100", "200"]
        # Coordinates are rounded to int32 at conversion time.
        body100 = df[df["bodyId"] == "100"]
        assert body100.loc[body100["node_id"] == 1, "x"].iloc[0] == 11
        assert not Path(str(save) + ".tmp").exists()

    def test_no_valid_nodes_returns_false(self, tmp_path, serial_pool, capsys):
        zip_path = _make_swc_zip(
            tmp_path / "skeletons.zip", {"100.swc": "# only a comment\n"})
        assert fafb.process_skeletons_to_parquet(
            str(zip_path), str(tmp_path / "out.parquet")) is False
        assert "No valid skeleton data found" in capsys.readouterr().out

    def test_batch_failure_returns_false_and_cleans_temp(
            self, tmp_path, serial_pool, monkeypatch, capsys):
        zip_path = _make_swc_zip(
            tmp_path / "skeletons.zip", {"100.swc": _SWC_100})

        def boom(zip_arg, names):
            raise RuntimeError("parse exploded")

        monkeypatch.setattr(fafb, "_parse_swc_batch", boom)
        assert fafb.process_skeletons_to_parquet(
            str(zip_path), str(tmp_path / "out.parquet")) is False
        assert "Error processing skeletons" in capsys.readouterr().out
        assert not Path(str(tmp_path / "out.parquet") + ".tmp").exists()

    def test_midstream_failure_closes_writer_and_cleans_temp(
            self, tmp_path, monkeypatch, capsys):
        """A worker dying after the first batch must close the writer + sweep."""
        zip_path = _make_swc_zip(tmp_path / "skeletons.zip", {
            "100.swc": _SWC_100,
            "200.swc": _SWC_200,
        })

        class _FlakyExecutor:
            def __init__(self, max_workers=None):
                self.calls = 0

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

            def submit(self, fn, *args, **kwargs):
                self.calls += 1
                future = concurrent.futures.Future()
                if self.calls == 1:
                    future.set_result(fn(*args, **kwargs))
                else:
                    future.set_exception(RuntimeError("worker died"))
                return future

        monkeypatch.setattr(
            fafb.concurrent.futures, "ProcessPoolExecutor", _FlakyExecutor)

        assert fafb.process_skeletons_to_parquet(
            str(zip_path), str(tmp_path / "out.parquet"), batch_size=1) is False
        assert "Error processing skeletons" in capsys.readouterr().out
        assert not Path(str(tmp_path / "out.parquet") + ".tmp").exists()


class TestFafbPostCounts:
    def _write_pair(self, tmp_path, fmt):
        neurons = pd.DataFrame({
            "bodyId": ["1", "2", "3"],
            "type": ["A", "B", "C"],
            "post": [0, 0, 0],
        })
        conn = pd.DataFrame({
            "bodyId_pre": ["1", "2"],
            "bodyId_post": ["2", "3"],
            "weight": [8, 4],
        })
        if fmt == "parquet":
            neuron_path = tmp_path / "neurons.parquet"
            conn_path = tmp_path / "conn.parquet"
            neurons.to_parquet(neuron_path, index=False)
            conn.to_parquet(conn_path, index=False)
        else:
            neuron_path = tmp_path / "neurons.csv"
            conn_path = tmp_path / "conn.csv"
            neurons.to_csv(neuron_path, index=False)
            conn.to_csv(conn_path, index=False)
        return neuron_path, conn_path

    def test_updates_parquet_and_writes_csv_copy(self, tmp_path):
        neuron_path, conn_path = self._write_pair(tmp_path, "parquet")
        csv_copy = tmp_path / "neurons_updated.csv"

        assert fafb.update_neuron_post_counts(
            str(neuron_path), str(conn_path), save_csv_path=str(csv_copy)) is True

        df = _read_pq(neuron_path)
        posts = dict(zip(df["bodyId"], df["post"]))
        assert posts == {"1": 0, "2": 8, "3": 4}
        assert csv_copy.exists()

    def test_updates_csv_paths(self, tmp_path):
        neuron_path, conn_path = self._write_pair(tmp_path, "csv")
        assert fafb.update_neuron_post_counts(
            str(neuron_path), str(conn_path)) is True
        df = pd.read_csv(neuron_path, dtype={"bodyId": str})
        posts = dict(zip(df["bodyId"], df["post"]))
        assert posts == {"1": 0, "2": 8, "3": 4}

    def test_missing_connections_returns_false(self, tmp_path):
        neuron_path, _ = self._write_pair(tmp_path, "parquet")
        assert fafb.update_neuron_post_counts(
            str(neuron_path), str(tmp_path / "missing.parquet")) is False


class TestFafbDownloadInstructions:
    def test_marks_present_files(self, tmp_path, capsys):
        downloads = tmp_path / "downloads"
        downloads.mkdir()
        _write_gz(downloads / "classification.csv.gz", "root_id,class\n1,X\n")
        # Uncompressed variant satisfies the gz expectation.
        (downloads / "names.csv").write_text("root_id,name\n1,A\n", encoding="utf-8")
        # Fallback connections filename.
        _write_gz(downloads / "connections_princeton.csv.gz", "a\n1\n")

        fafb.print_download_instructions(str(downloads))

        out = capsys.readouterr().out
        assert "[existed] classification.csv.gz" in out
        assert "[existed] [optional] names.csv.gz" in out
        assert "[existed] connections_princeton_no_threshold.csv.gz" in out
        assert "❌" not in out

    def test_marks_missing_critical_files(self, tmp_path, capsys):
        fafb.print_download_instructions(str(tmp_path))
        out = capsys.readouterr().out
        assert "❌ classification.csv.gz" in out
        assert "- [optional] names.csv.gz" in out


# ---------------------------------------------------------------------------
# ensure_flywire_data orchestration
# ---------------------------------------------------------------------------

def _fafb_sources(downloads: Path, connections_name="connections.csv.gz"):
    _write_gz(downloads / "classification.csv.gz", _CLASSIFICATION_CSV)
    _write_gz(downloads / "names.csv.gz", "root_id,name,group\n1,InstA,0\n")
    # Uncompressed enrichment exercises the .gz-fallback branch.
    (downloads / "coordinates.csv").write_text(
        "root_id,supervoxel_id,x,y,z\n1,77,1.0,2.0,3.0\n", encoding="utf-8")
    _write_gz(downloads / connections_name, _FAFB_CONNECTIONS_CSV)
    _write_gz(
        downloads / "my_synapse_table.csv.gz",
        "x,y,z,pre_root_id,post_root_id\n1,2,3,609749525,609749526\n")
    _make_swc_zip(downloads / "skeletons.zip", {"100.swc": _SWC_100})


class TestFafbEnsure:
    def test_missing_sources_returns_false(self, tmp_path, capsys):
        dataset_dir = tmp_path / "flywire_FAFB_v783"
        assert fafb.ensure_flywire_data(
            "flywire_FAFB_v783", str(dataset_dir)) is False
        out = capsys.readouterr().out
        assert "MISSING CRITICAL FILES" in out
        assert (dataset_dir / "downloads").is_dir()

    def test_full_pipeline_converts_everything(self, tmp_path, serial_pool, capsys):
        dataset_dir = tmp_path / "flywire_FAFB_v783"
        downloads = dataset_dir / "downloads"
        downloads.mkdir(parents=True)
        _fafb_sources(downloads)

        assert fafb.ensure_flywire_data(
            "flywire_FAFB_v783", str(dataset_dir)) is True

        neuron_pq = dataset_dir / "flywire_FAFB_v783_allneurons_neuron_df.parquet"
        conn_pq = dataset_dir / "flywire_FAFB_v783_merged_connections.parquet"
        syn_pq = dataset_dir / "flywire_FAFB_v783_synapse_table.parquet"
        assert neuron_pq.exists() and conn_pq.exists() and syn_pq.exists()
        assert (dataset_dir / "sk_lod1_783_healed.zip").exists()

        df = _read_pq(neuron_pq)
        posts = dict(zip(df["bodyId"], df["post"]))
        assert posts == {"1": 0, "2": 8, "3": 4}

    def test_second_run_reuses_existing_outputs(self, tmp_path, serial_pool, capsys):
        dataset_dir = tmp_path / "flywire_FAFB_v783"
        downloads = dataset_dir / "downloads"
        downloads.mkdir(parents=True)
        _fafb_sources(downloads)
        assert fafb.ensure_flywire_data(
            "flywire_FAFB_v783", str(dataset_dir)) is True
        capsys.readouterr()

        assert fafb.ensure_flywire_data(
            "flywire_FAFB_v783", str(dataset_dir)) is True
        out = capsys.readouterr().out
        assert "Found existing neurons" in out
        assert "Found existing connections" in out
        assert "Found existing synapse table" in out
        assert "Found existing skeletons" in out
        assert "Post counts already populated." in out

    def test_neuron_conversion_failure_marks_critical(self, tmp_path, capsys):
        dataset_dir = tmp_path / "flywire_FAFB_v783"
        downloads = dataset_dir / "downloads"
        downloads.mkdir(parents=True)
        (downloads / "classification.csv.gz").write_bytes(b"not really gzip")
        _write_gz(downloads / "connections_princeton_no_threshold.csv.gz",
                  _FAFB_CONNECTIONS_CSV)

        assert fafb.ensure_flywire_data(
            "flywire_FAFB_v783", str(dataset_dir)) is False
        assert "MISSING CRITICAL FILES" in capsys.readouterr().out

    def test_connection_conversion_failure_marks_critical(self, tmp_path, capsys):
        dataset_dir = tmp_path / "flywire_FAFB_v783"
        downloads = dataset_dir / "downloads"
        downloads.mkdir(parents=True)
        _write_gz(downloads / "classification.csv.gz", _CLASSIFICATION_CSV)
        (downloads / "connections_princeton_no_threshold.csv.gz").write_bytes(
            b"bad gzip")

        assert fafb.ensure_flywire_data(
            "flywire_FAFB_v783", str(dataset_dir)) is False
        assert "MISSING CRITICAL FILES" in capsys.readouterr().out

    def test_post_count_check_failure_warns_but_succeeds(self, tmp_path, capsys):
        dataset_dir = tmp_path / "flywire_FAFB_v783"
        dataset_dir.mkdir(parents=True)
        neuron_pq = dataset_dir / "flywire_FAFB_v783_allneurons_neuron_df.parquet"
        conn_pq = dataset_dir / "flywire_FAFB_v783_merged_connections.parquet"
        neuron_pq.write_text("not parquet")
        conn_pq.write_text("not parquet either")

        assert fafb.ensure_flywire_data(
            "flywire_FAFB_v783", str(dataset_dir)) is True
        assert "Could not check post counts" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# BANC ensure_* failure branches
# ---------------------------------------------------------------------------

class TestBancEnsureFailures:
    def test_neuron_conversion_failure_marks_critical(self, tmp_path, capsys):
        dataset_dir = tmp_path / "flywire_BANC_v626"
        downloads = dataset_dir / "downloads"
        downloads.mkdir(parents=True)
        (downloads / "neurons.csv.gz").write_bytes(b"not really gzip")
        _write_gz(
            downloads / "connections_princeton.csv.gz", _BANC_CONNECTIONS_CSV)

        assert banc.ensure_banc_data("flywire_BANC_v626", str(dataset_dir)) is False
        assert "MISSING CRITICAL FILES" in capsys.readouterr().out

    def test_connection_conversion_failure_marks_critical(self, tmp_path, capsys):
        dataset_dir = tmp_path / "flywire_BANC_v626"
        downloads = dataset_dir / "downloads"
        downloads.mkdir(parents=True)
        _write_gz(downloads / "neurons.csv.gz", _BANC_NEURONS_CSV)
        (downloads / "connections_princeton.csv.gz").write_bytes(b"bad gzip")

        assert banc.ensure_banc_data("flywire_BANC_v626", str(dataset_dir)) is False
        assert "MISSING CRITICAL FILES" in capsys.readouterr().out

    def test_post_count_check_failure_warns_but_succeeds(self, tmp_path, capsys):
        dataset_dir = tmp_path / "flywire_BANC_v626"
        dataset_dir.mkdir(parents=True)
        neuron_pq = dataset_dir / "flywire_BANC_v626_allneurons_neuron_df.parquet"
        conn_pq = dataset_dir / "flywire_BANC_v626_merged_connections.parquet"
        neuron_pq.write_text("not parquet")
        conn_pq.write_text("not parquet either")

        assert banc.ensure_banc_data("flywire_BANC_v626", str(dataset_dir)) is True
        assert "Could not check post counts" in capsys.readouterr().out
