"""UI affordances for unavailable BANC morphology/skeleton workflows."""

from nicegui import Client
from nicegui.page import page


def test_dataset_selector_disables_banc_options():
    from ui.components.common import dataset_selector

    client = Client(page("/disabled-banc-dataset"))
    with client:
        selector = dataset_selector(
            datasets=["male-cns:v1.0", "flywire_BANC_v626", "flywire_FAFB_v783"],
            show_local_status=False,
            disable_banc=True,
        )

    assert ":option-disable" in selector._props
    assert "banc" in selector._props[":option-disable"]
    assert selector._drocat_disabled_datasets == ["flywire_BANC_v626"]


def test_morphology_tab_shows_persistent_banc_warning_when_selected():
    from ui.tabs.find_similar import create_find_similar_tab

    client = Client(page("/similar-banc-warning"))
    with client:
        create_find_similar_tab()

    morph_selector = next(
        element
        for element in client.elements.values()
        if getattr(element, "_props", {}).get("label") == "Dataset"
    )
    warning = next(
        element
        for element in client.elements.values()
        if "BANC morphological similarity is unavailable" in str(
            getattr(element, "text", "")
        )
    )
    assert warning.visible is False
    assert morph_selector._props.get(":option-disable")

    morph_selector.set_value("flywire_BANC_v626")
    assert warning.visible is True
