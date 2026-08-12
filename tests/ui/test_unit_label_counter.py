"""Structural UI tests for the chip-input count badge unit label.

The Cross-Dataset tab reuses the chip-based neuron list input for its
Synapse Thresholds control; its counter badge must name items "thresholds",
not "neurons" (bug found by the 2026-08 E2E audit: badge read "3 neurons").
Neuron inputs everywhere else keep the default "neuron" unit.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _badge_texts(client):
    return [
        el.text
        for el in client.elements.values()
        if type(el).__name__ == "Badge" and getattr(el, "text", "")
    ]


class TestChipCounterUnitLabel:
    def test_component_default_unit_is_neuron(self):
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        client = Client(page("/unit-label-default"))
        with client:
            neuron_list_input(label="Query", initial=["aMe12"])
        texts = _badge_texts(client)
        assert "1 neuron" in texts, texts

    def test_component_custom_unit_pluralizes(self):
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        client = Client(page("/unit-label-threshold"))
        with client:
            neuron_list_input(
                label="Synapse Thresholds",
                initial=[3, 5, 10],
                unit_label="threshold",
                show_filter=False,
                show_upload=False,
            )
        texts = _badge_texts(client)
        assert "3 thresholds" in texts, texts
        assert not any("neuron" in t for t in texts), texts

    def test_component_singular_form(self):
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        client = Client(page("/unit-label-singular"))
        with client:
            neuron_list_input(
                label="Synapse Thresholds",
                initial=[3],
                unit_label="threshold",
                show_filter=False,
                show_upload=False,
            )
        texts = _badge_texts(client)
        assert "1 threshold" in texts, texts

    def test_interdataset_thresholds_badge_reads_thresholds(self):
        """The full Cross-Dataset tab wires unit_label='threshold'."""
        from nicegui import Client
        from nicegui.page import page
        from ui.tabs.inter_dataset import create_inter_dataset_tab

        client = Client(page("/interdataset-unit-label"))
        with client:
            create_inter_dataset_tab()
        texts = _badge_texts(client)
        assert "3 thresholds" in texts, texts
        # The neuron inputs on the same tab still count neurons.
        assert any("neuron" in t for t in texts), texts
