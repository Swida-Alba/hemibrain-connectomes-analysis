"""UI regression tests for shared NeuronBridge controls."""

import sys
from pathlib import Path

import pytest
from nicegui import Client
from nicegui.page import page

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ui.config import DEFAULTS  # noqa: E402
from ui.tabs.nb_colabel import create_nb_colabel_tab  # noqa: E402
from ui.tabs.nb_find_neuron import create_nb_find_neuron_tab  # noqa: E402


@pytest.mark.parametrize(
    ("factory", "url"),
    [
        (create_nb_find_neuron_tab, "/nb-find-neuron-parameters"),
        (create_nb_colabel_tab, "/nb-colabel-parameters"),
    ],
)
def test_neuronbridge_tabs_share_top_n_and_score_cutoff_defaults(factory, url):
    client = Client(page(url))
    with client:
        factory()

    controls = {
        element._props.get("label"): element
        for element in client.elements.values()
        if getattr(element, "_props", {}).get("label")
    }

    assert controls["Algorithm"].value == DEFAULTS["match_algorithm"] == "cds"
    assert controls["Top N Matches Per Line"].value == DEFAULTS["nb_top_n"] == 50
    assert controls["Score Cutoff"].value == DEFAULTS["nb_min_score"] == 30000
    assert "Top N Results" not in controls
    assert "Top N Neurons Per Line" not in controls
