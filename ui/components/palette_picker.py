"""
Visual color palette pickers for DROCAT.

Provides a reusable palette selector with direct color-swatch previews backed
by the full bokeh.palettes catalog, plus a small single-color swatch picker.
"""

from typing import List, Optional, Tuple

from nicegui import ui


_PALETTE_CATALOG: Optional[List[Tuple[str, List[str]]]] = None


def get_palette_catalog() -> List[Tuple[str, List[str]]]:
    """
    Build a curated catalog of palettes from bokeh.palettes.

    Includes categorical palettes, sequential palettes (256-level sample) and
    diverging palettes (11-level sample). Returns [(name, [hex colors]), ...].
    """
    global _PALETTE_CATALOG
    if _PALETTE_CATALOG is not None:
        return _PALETTE_CATALOG

    import bokeh.palettes as bp

    catalog: List[Tuple[str, List[str]]] = []
    seen = set()

    def add(name: str, colors) -> None:
        colors = list(colors)
        if not colors or name in seen:
            return
        seen.add(name)
        catalog.append((name, colors))

    # Categorical palettes (dicts keyed by size -> use the largest entry)
    for name in [
        "Category10", "Category20", "Category20b", "Category20c",
        "Accent", "Dark2", "Paired", "Pastel1", "Pastel2",
        "Set1", "Set2", "Set3", "Colorblind",
    ]:
        obj = getattr(bp, name, None)
        if obj is None:
            continue
        if isinstance(obj, dict):
            add(name, obj[max(obj.keys())])
        else:
            add(name, obj)

    # Sequential palettes (prefer 256-level list, fall back to 9-level)
    for base in [
        "Blues", "Greens", "Greys", "Oranges", "Purples", "Reds",
        "BuGn", "BuPu", "GnBu", "OrRd", "PuBu", "PuBuGn", "PuRd", "RdPu",
        "YlGn", "YlGnBu", "YlOrBr", "YlOrRd",
        "Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo",
    ]:
        for size in (256, 9):
            obj = getattr(bp, f"{base}{size}", None)
            if obj is not None:
                add(base, obj)
                break

    # Diverging palettes (prefer 11-level list, fall back to 9)
    for base in [
        "BrBG", "PiYG", "PRGn", "PuOr", "RdBu", "RdGy",
        "RdYlBu", "RdYlGn", "Spectral", "Coolwarm",
    ]:
        for size in (256, 11, 9):
            obj = getattr(bp, f"{base}{size}", None)
            if obj is not None:
                add(base, obj)
                break

    _PALETTE_CATALOG = catalog
    return catalog


def sample_palette(colors: List[str], n: int) -> List[str]:
    """Evenly sample n colors from a palette (handles n larger than the palette)."""
    if n <= 0 or not colors:
        return []
    if n == 1:
        return [colors[0]]
    indices = [round(i * (len(colors) - 1) / (n - 1)) for i in range(n)]
    return [colors[i] for i in indices]


def palette_picker(
    label: str,
    value: Optional[str] = None,
    catalog: Optional[List[Tuple[str, List[str]]]] = None,
    include_auto: bool = False,
    max_height: int = 240,
) -> ui.element:
    """
    Palette selector with direct color-swatch preview.

    Renders a grid of palette cards (name + swatch row); the selected card is
    highlighted and shown in a preview bar. Returns an element with:
      - .value: current palette name
      - .get_value(): same
      - .get_colors(): palette color list
    """
    catalog = list(catalog) if catalog is not None else list(get_palette_catalog())
    if include_auto:
        catalog = [("Auto (single gray)", ["#94a3b8"])] + catalog
    names = [name for name, _ in catalog]
    if value not in names:
        value = names[0]

    state = {"value": value}
    cards: List[Tuple[ui.element, str]] = []

    with ui.column().classes("w-full gap-1") as container:
        ui.label(label).classes("drocat-mini-label")

        preview = ui.row().classes("items-center gap-2 w-full drocat-palette-preview")

        def render_preview():
            preview.clear()
            with preview:
                colors = dict(catalog)[state["value"]]
                for color in colors[:12]:
                    ui.element("div").style(
                        f"background:{color}; width:18px; height:18px;"
                        "border-radius:4px; border:1px solid rgba(11,31,58,.12);"
                    )
                ui.label(state["value"]).classes("text-caption drocat-muted")

        with ui.expansion("Choose palette", icon="palette").classes(
            "w-full drocat-palette-expansion"
        ):
            with ui.element("div").classes("drocat-palette-grid").style(
                f"max-height:{max_height}px"
            ):
                for name, colors in catalog:
                    card = ui.element("div").classes(
                        "drocat-palette-card"
                        + (" selected" if name == state["value"] else "")
                    ).on("click", lambda n=name: select(n))
                    with card:
                        with ui.row().classes("gap-0 w-full drocat-palette-swatches"):
                            for color in colors[:14]:
                                ui.element("div").style(
                                    f"background:{color}; flex:1; min-width:5px;"
                                    "height:18px;"
                                )
                        ui.label(name).classes("drocat-palette-name")
                    cards.append((card, name))

        def select(name: str):
            state["value"] = name
            for element, card_name in cards:
                element.classes(
                    replace="drocat-palette-card"
                    + (" selected" if card_name == name else "")
                )
            container.value = name
            render_preview()

        render_preview()

    container.value = state["value"]
    container.get_value = lambda: state["value"]
    container.get_colors = lambda: dict(catalog)[state["value"]]
    return container


def color_swatch_picker(
    label: str,
    value: str = "auto",
    options: Optional[List[Tuple[str, str]]] = None,
) -> ui.element:
    """
    Single-color picker with round swatch previews.

    options: list of (value, display_name); value may be a hex color or 'auto'.
    Returns an element with .value / .get_value().
    """
    options = options or [
        ("auto", "Auto"),
        ("#94a3b8", "Gray"),
        ("#145cff", "Blue"),
        ("#22c55e", "Green"),
        ("#eab308", "Yellow"),
        ("#ef4444", "Red"),
        ("#8b5cf6", "Purple"),
        ("#000000", "Black"),
        ("#ffffff", "White"),
    ]
    state = {"value": value if any(v == value for v, _ in options) else "auto"}

    with ui.column().classes("w-full gap-1") as container:
        ui.label(label).classes("drocat-mini-label")
        with ui.row().classes("items-center gap-2 w-full drocat-swatch-row") as row:
            swatches = []
            for option_value, display in options:
                swatch = ui.element("div").classes(
                    "drocat-swatch"
                    + (" selected" if option_value == state["value"] else "")
                ).tooltip(display)
                with swatch:
                    ui.element("div").style(
                        f"background:{option_value if option_value != 'auto' else '#e5e7eb'};"
                        "width:22px; height:22px; border-radius:50%;"
                        "border:2px solid rgba(11,31,58,.15);"
                    )
                swatches.append((swatch, option_value))

        def select(option_value: str):
            state["value"] = option_value
            container.value = option_value
            for element, swatch_value in swatches:
                element.classes(
                    replace="drocat-swatch"
                    + (" selected" if swatch_value == option_value else "")
                )

        for element, option_value in swatches:
            element.on("click", lambda v=option_value: select(v))

    container.value = state["value"]
    container.get_value = lambda: state["value"]
    return container
