"""Tab-level query-history recording tests.

Every UI tab records its query inputs into the shared history store after a
successful run: dataset-scoped tabs record the selected dataset(s), while
NeuronBridge / FlyLight queries are recorded without dataset provenance so
they appear in every dataset's history list. Failed or cancelled runs never
record.
"""

import asyncio
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from nicegui import Client  # noqa: E402
from nicegui.page import page  # noqa: E402

import ui.history_store as hs  # noqa: E402
from ui.components.output_panel import OutputPanel  # noqa: E402
from ui.tabs.connectivity_profiling import (  # noqa: E402
    create_connectivity_profiling_tab,
)
from ui.tabs.find_homologs import create_find_homologs_tab  # noqa: E402
from ui.tabs.nb_colabel import create_nb_colabel_tab  # noqa: E402


@pytest.fixture
def isolated_history(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "_HISTORY_PATH", tmp_path / "neuron_history.json")
    return hs


def _mock_output_panel_run(monkeypatch, returncode=0):
    """Stub the panel runner and its UI-touching methods.

    The run handlers are driven through ``asyncio.run``, which runs on a
    different task than the page build; the panel methods that create
    elements or emit JavaScript therefore get no-ops so the test focuses on
    the history recording.
    """
    captured = []

    async def fake_run(self, runner, tool_name, constructor_params,
                       method_name, method_params=None, output_dir=None):
        captured.append((tool_name, constructor_params, method_params))
        return {"returncode": returncode, "files": [], "duration": 0,
                "cancelled": False, "output_folder": None,
                "neuron_match": None}

    monkeypatch.setattr(OutputPanel, "run", fake_run)
    monkeypatch.setattr(OutputPanel, "set_running", lambda self, value: None)
    monkeypatch.setattr(OutputPanel, "set_status",
                        lambda self, status, color="grey": None)
    monkeypatch.setattr(OutputPanel, "clear", lambda self: None)
    monkeypatch.setattr(OutputPanel, "show_files",
                        lambda self, files, output_dir=None: None)
    monkeypatch.setattr(OutputPanel, "log",
                        lambda self, message, level="stdout": None)
    return captured


def _chip_inputs(client):
    return [
        element for element in client.elements.values()
        if getattr(element, "chip_input", None) is not None
    ]


def _chip_input(client, label):
    return next(
        element for element in _chip_inputs(client)
        if element.chip_input._props.get("label") == label
    )


def _click_run(client, label):
    """Invoke the tab's async run handler directly (NiceGUI wraps the raw
    callback in a ``lambda _: handle_event(...)``; cell 0 of that closure
    holds the original handler)."""
    button = next(
        element for element in client.elements.values()
        if type(element).__name__ == "Button"
        and getattr(element, "text", "") == label
    )
    click = next(
        listener for listener in button._event_listeners.values()
        if listener.type == "click"
    )
    handler = click.handler.__closure__[0].cell_contents
    result = handler()
    if asyncio.iscoroutine(result):
        asyncio.run(result)


class TestProfilingHistory:
    def test_profiling_records_query_with_selected_datasets(
        self, isolated_history, monkeypatch
    ):
        captured = _mock_output_panel_run(monkeypatch)
        client = Client(page("/history-profiling"))
        with client:
            create_connectivity_profiling_tab()
        _chip_input(client, "Neurons to Compare").add_values(["aMe12", "aMe10"])
        _click_run(client, "Run Profiling")

        assert captured and captured[0][0] == "connectivity_profiling"
        assert isolated_history.recent() == ["aMe12", "aMe10"]
        assert isolated_history.datasets_of("aMe12") == ["male-cns:v1.0"]
        assert isolated_history.datasets_of("aMe10") == ["male-cns:v1.0"]

    def test_profiling_failed_run_leaves_history_untouched(
        self, isolated_history, monkeypatch
    ):
        _mock_output_panel_run(monkeypatch, returncode=1)
        client = Client(page("/history-profiling-fail"))
        with client:
            create_connectivity_profiling_tab()
        _chip_input(client, "Neurons to Compare").add_values(["aMe12"])
        _click_run(client, "Run Profiling")

        assert isolated_history.recent() == []


class TestHomologsHistory:
    def test_homologs_records_each_source_with_source_dataset(
        self, isolated_history, monkeypatch
    ):
        captured = _mock_output_panel_run(monkeypatch)
        client = Client(page("/history-homologs"))
        with client:
            create_find_homologs_tab()
        _chip_input(client, "Source Neuron(s) (type or bodyId)").add_values(
            ["aMe12", "aMe10"]
        )
        _click_run(client, "Find Homologs")

        assert captured and captured[0][0] == "find_homologs"
        assert set(isolated_history.recent()) == {"aMe12", "aMe10"}
        recorded = isolated_history.datasets_of("aMe12")
        assert recorded and recorded == isolated_history.datasets_of("aMe10")


class TestColabelHistory:
    def test_colabel_records_driver_lines_without_dataset_scope(
        self, isolated_history, monkeypatch
    ):
        captured = _mock_output_panel_run(monkeypatch)
        client = Client(page("/history-colabel"))
        with client:
            create_nb_colabel_tab()
        _chip_input(client, "Driver Line Names").add_values(
            ["R10A06", "R10A07"]
        )
        _click_run(client, "Run Co-Labeling")

        assert captured and captured[0][0] == "nb_colabel"
        assert isolated_history.recent() == ["R10A06", "R10A07"]
        # NeuronBridge searches every dataset: no provenance, visible in all
        # dataset scopes.
        assert isolated_history.datasets_of("R10A06") == []
