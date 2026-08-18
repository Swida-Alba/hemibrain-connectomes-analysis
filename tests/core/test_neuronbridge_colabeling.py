"""Regression tests for NeuronBridge top-N and score-cutoff semantics."""

from pathlib import Path

import pandas as pd

import neuronbridge_finder as neuronbridge_finder_module
from neuronbridge_finder import NeuronBridgeFinder


class _DummyClient:
    version = "test-version"


def _finder(tmp_path, monkeypatch):
    monkeypatch.setattr(neuronbridge_finder_module, "NBClient", _DummyClient)
    return NeuronBridgeFinder(
        cache_folder=str(tmp_path / "cache"),
        verbose=False,
        max_workers=1,
    )


def _matches():
    rows = []
    for line, scores in (("L1", [100.0, 40.0, 30.0, 20.0]),
                         ("L2", [90.0, 45.0, 35.0, 25.0])):
        for index, score in enumerate(scores, start=1):
            rows.append({
                "bodyId": f"{line}-{index}",
                "dataset": "flywire_FAFB_v783",
                "instance": f"{line}-instance-{index}",
                "type": f"{line}_T{index}",
                "status": "",
                "score": score,
                "image_id": f"image-{line}-{index}",
                "lm_sample": f"sample-{line}",
                "match_type": "cds",
                "library": "FlyWire_FAFB_v783",
            })
    frame = pd.DataFrame(rows)
    return {
        line: frame[frame["bodyId"].str.startswith(f"{line}-")].reset_index(drop=True)
        for line in ("L1", "L2")
    }


def test_expression_matrix_filters_after_top_n_retrieval(tmp_path, monkeypatch):
    finder = _finder(tmp_path, monkeypatch)
    frames = _matches()

    def fake_line_to_neuron(line_name, match_type="cds", top_n=-1):
        result = frames[line_name].copy()
        return result.head(top_n) if top_n > 0 else result

    monkeypatch.setattr(finder, "line_to_neuron", fake_line_to_neuron)

    _, expression, line_neurons, labeling_info = finder._calculate_mutual_information(
        lines=["L1", "L2"],
        queried_types=[],
        match_type="cds",
        top_n=3,
        min_score=50.0,
    )

    # Top-N is applied first and the raw per-line table keeps all three rows,
    # including the two rows below the cutoff.
    assert len(line_neurons["L1"]) == 3
    assert line_neurons["L1"]["score"].tolist() == [100.0, 40.0, 30.0]
    assert line_neurons["L1"]["_passes_min_score"].tolist() == [True, False, False]

    # Only score-qualified rows contribute to aggregate expression outputs.
    assert list(expression.columns) == ["FAFB_L1_T1", "FAFB_L2_T1"]
    assert expression.loc["L1", "FAFB_L1_T1"] == 100.0
    assert expression.loc["L2", "FAFB_L2_T1"] == 90.0
    assert set(labeling_info["type"]) == {"L1_T1", "L2_T1"}


def test_colabeling_writes_cutoff_warning_notes(tmp_path, monkeypatch):
    finder = _finder(tmp_path, monkeypatch)
    frames = _matches()

    def fake_line_to_neuron(line_name, match_type="cds", top_n=-1):
        result = frames[line_name].copy()
        return result.head(top_n) if top_n > 0 else result

    monkeypatch.setattr(finder, "line_to_neuron", fake_line_to_neuron)
    monkeypatch.setattr(finder, "_save_dataset_categorized_files", lambda *args, **kwargs: None)

    results = finder.analyze_colabeling(
        lines=["L1", "L2"],
        top_n_neurons=3,
        similarity_methods=["jaccard"],
        output_dir=str(tmp_path / "output"),
        generate_report=False,
        visualize=False,
        min_score=50.0,
        min_type_avg_score=0.0,
    )

    assert all(len(frame) == 3 for frame in results["line_neurons"].values())
    notes = next(Path(tmp_path / "output").glob("NB-colabeling_*/user_warning_notes.txt"))
    text = notes.read_text(encoding="utf-8")
    assert "expression_matrix.csv" in text
    assert "top-N (3)" in text
