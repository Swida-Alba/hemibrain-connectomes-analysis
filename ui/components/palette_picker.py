"""
Visual color palette tools for DROCAT.

- ``palette_picker``: preset palette cards with full-gradient previews.
- ``palette_editor``: full palette editor - preset palettes with selectable
  range, custom single-color assignment, click-to-add from the palette strip,
  and reordering of the custom color list.
- ``color_swatch_picker``: single-color swatches with a custom color input.
"""

from typing import List, Optional, Tuple

from nicegui import ui


_PALETTE_CATALOG: Optional[List[Tuple[str, List[str]]]] = None


def get_palette_catalog() -> List[Tuple[str, List[str]]]:
    """Catalog of palettes from bokeh.palettes (categorical/sequential/diverging)."""
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

    for name in [
        "Category10", "Category20", "Category20b", "Category20c",
        "Accent", "Dark2", "Paired", "Pastel1", "Pastel2",
        "Set1", "Set2", "Set3", "Colorblind",
    ]:
        obj = getattr(bp, name, None)
        if obj is None:
            continue
        add(name, obj[max(obj.keys())] if isinstance(obj, dict) else obj)

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

    for base in [
        "BrBG", "PiYG", "PRGn", "PuOr", "RdBu", "RdGy",
        "RdYlBu", "RdYlGn", "Spectral",
    ]:
        for size in (256, 11, 9):
            obj = getattr(bp, f"{base}{size}", None)
            if obj is not None:
                add(base, obj)
                break

    _PALETTE_CATALOG = catalog
    return catalog


def sample_palette(colors: List[str], n: int) -> List[str]:
    """Evenly sample n colors from a palette (repeats when n > palette size)."""
    if n <= 0 or not colors:
        return []
    if n == 1:
        return [colors[0]]
    indices = [round(i * (len(colors) - 1) / (n - 1)) for i in range(n)]
    return [colors[i] for i in indices]


def palette_slice(colors: List[str], start_pct: int, end_pct: int) -> List[str]:
    """Select a sub-range of a palette by percentage (0-100)."""
    if not colors:
        return []
    start_pct = max(0, min(99, int(start_pct)))
    end_pct = max(1, min(100, int(end_pct)))
    if end_pct <= start_pct:
        end_pct = start_pct + 1
    start_idx = round(start_pct / 100 * (len(colors) - 1))
    end_idx = round(end_pct / 100 * (len(colors) - 1))
    return colors[start_idx:end_idx + 1]


def _gradient_style(colors: List[str], height: int = 18) -> str:
    """CSS linear-gradient over the whole palette (full preview for long palettes)."""
    if not colors:
        return ""
    stops = sample_palette(colors, min(len(colors), 32))
    return f"background:linear-gradient(90deg,{','.join(stops)});height:{height}px;"


def _render_color_strip(
    colors: List[str],
    height: int = 18,
    click=None,
    classes: str = "",
) -> None:
    """Render a color strip in the current context (gradient for long palettes)."""
    if len(colors) > 20:
        strip = ui.element("div").classes(f"drocat-palette-strip {classes}").style(
            _gradient_style(colors, height)
        )
        if click is not None:
            strip.on("click", click)
        return
    with ui.row().classes(f"gap-0 w-full drocat-palette-swatches {classes}"):
        for color in colors:
            sw = ui.element("div").style(
                f"background:{color}; flex:1; min-width:5px; height:{height}px;"
            )
            if click is not None:
                sw.on("click", lambda c=color: click(c))


def palette_picker(
    label: str,
    value: Optional[str] = None,
    catalog: Optional[List[Tuple[str, List[str]]]] = None,
    include_auto: bool = False,
    max_height: int = 220,
) -> ui.element:
    """Preset palette selector with full-color previews (used for small schemes)."""
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
                _render_color_strip(
                    dict(catalog)[state["value"]][:24],
                    height=20,
                    classes="flex-grow",
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
                        _render_color_strip(colors[:24], height=18)
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


def palette_editor(
    label: str,
    value: Optional[str] = None,
    include_auto: bool = False,
    max_height: int = 220,
) -> ui.element:
    """
    Full palette editor with direct previews.

    Features:
    - Preset palettes from bokeh.palettes with full-gradient previews
    - Select part of a palette with Start/End range controls
    - Custom colors: add single colors via a native color picker, click the
      palette strip to append its colors, reorder (move left/right, reverse)
      and remove entries
    """
    catalog = list(get_palette_catalog())
    if include_auto:
        catalog = [("Auto (single gray)", ["#94a3b8"])] + catalog
    names = [name for name, _ in catalog]
    if value not in names:
        value = names[0]

    state = {
        "mode": "preset",
        "palette": value,
        "start": 0,
        "end": 100,
        "custom": [],
    }
    cards: List[Tuple[ui.element, str]] = []

    def palette_colors(name: Optional[str] = None) -> List[str]:
        return dict(catalog)[name or state["palette"]]

    def effective_slice() -> List[str]:
        return palette_slice(
            palette_colors(), state["start"], state["end"]
        )

    def effective_colors() -> List[str]:
        if state["mode"] == "custom" and state["custom"]:
            return state["custom"]
        return effective_slice()

    with ui.column().classes("w-full gap-1") as container:
        ui.label(label).classes("drocat-mini-label")

        mode_toggle = ui.toggle(
            ["Preset palette", "Custom colors"], value="Preset palette"
        ).props("dense outlined")

        preview = ui.row().classes("items-center gap-2 w-full drocat-palette-preview")

        def render_preview():
            preview.clear()
            with preview:
                _render_color_strip(
                    effective_colors()[:32],
                    height=20,
                    classes="flex-grow",
                )
                mode_text = (
                    "custom colors"
                    if state["mode"] == "custom" and state["custom"]
                    else f"{state['palette']}  {state['start']}–{state['end']}%"
                )
                ui.label(mode_text).classes("text-caption drocat-muted")

        # ---------------- Preset panel ----------------
        with ui.column().classes("w-full gap-1") as preset_panel:
            with ui.expansion("Choose palette", icon="palette").classes(
                "w-full drocat-palette-expansion"
            ):
                with ui.element("div").classes("drocat-palette-grid").style(
                    f"max-height:{max_height}px"
                ):
                    for name, colors in catalog:
                        card = ui.element("div").classes(
                            "drocat-palette-card"
                            + (" selected" if name == state["palette"] else "")
                        ).on("click", lambda n=name: select_preset(n))
                    with card:
                        _render_color_strip(colors, height=18)
                        ui.label(name).classes("drocat-palette-name")
                        cards.append((card, name))

            with ui.row().classes("w-full items-center gap-2"):
                ui.label("Palette range").classes("text-caption drocat-muted")
                start_input = ui.number("Start %", value=0, min=0, max=99, step=1).props(
                    "dense outlined"
                ).classes("w-28")
                end_input = ui.number("End %", value=100, min=1, max=100, step=1).props(
                    "dense outlined"
                ).classes("w-28")

                def apply_range():
                    state["start"] = int(start_input.value)
                    state["end"] = int(end_input.value)
                    render_preview()

                ui.button("Apply", icon="check", on_click=apply_range).props(
                    "flat dense color=primary"
                )
            ui.label(
                "Full palette preview below; use Start/End % to select part of it."
            ).classes("text-caption drocat-muted")
            _render_color_strip(palette_colors(), height=22, classes="w-full")

        # ---------------- Custom panel ----------------
        with ui.column().classes("w-full gap-1") as custom_panel:
            with ui.row().classes("w-full items-center gap-2"):
                color_input = ui.color_input(value="#145cff").props("dense")
                ui.button(
                    "Add color", icon="add", on_click=lambda: add_custom_color()
                ).props("flat dense color=primary")
                ui.button(
                    "Reverse list", icon="swap_vert", on_click=lambda: reverse_custom()
                ).props("flat dense")
            ui.label(
                "Click the palette strip below to append individual colors, "
                "then reorder with ◀ ▶."
            ).classes("text-caption drocat-muted")
            _render_color_strip(
                palette_colors(),
                height=22,
                classes="w-full",
                click=lambda color: add_single_color(color),
            )
            custom_list = ui.column().classes("w-full gap-1")

            def render_custom_list():
                custom_list.clear()
                with custom_list:
                    if not state["custom"]:
                        ui.label("No custom colors yet.").classes(
                            "text-caption drocat-muted"
                        )
                        return
                    for index, color in enumerate(state["custom"]):
                        with ui.row().classes(
                            "items-center gap-2 w-full drocat-custom-color-row"
                        ):
                            ui.element("div").style(
                                f"background:{color}; width:24px; height:24px;"
                                "border-radius:6px; border:1px solid rgba(11,31,58,.15);"
                            )
                            ui.label(color).classes(
                                "text-caption font-mono drocat-muted drocat-truncate"
                            ).classes("flex-grow")
                            if index > 0:
                                ui.button(
                                    "◀", on_click=lambda i=index: move_custom(i, -1)
                                ).props("flat dense round")
                            if index < len(state["custom"]) - 1:
                                ui.button(
                                    "▶", on_click=lambda i=index: move_custom(i, 1)
                                ).props("flat dense round")
                            ui.button(
                                "✕", on_click=lambda i=index: remove_custom(i)
                            ).props("flat dense round color=negative")

            def add_single_color(color: str):
                if color and color not in state["custom"]:
                    state["custom"].append(color)
                    render_custom_list()
                    render_preview()

            def add_custom_color():
                add_single_color(color_input.value or "#145cff")

            def reverse_custom():
                state["custom"] = list(reversed(state["custom"]))
                render_custom_list()
                render_preview()

            def move_custom(index: int, delta: int):
                target = index + delta
                if target < 0 or target >= len(state["custom"]):
                    return
                colors = state["custom"]
                colors[index], colors[target] = colors[target], colors[index]
                render_custom_list()
                render_preview()

            def remove_custom(index: int):
                del state["custom"][index]
                render_custom_list()
                render_preview()

            render_custom_list()

        def select_preset(name: str):
            state["palette"] = name
            for element, card_name in cards:
                element.classes(
                    replace="drocat-palette-card"
                    + (" selected" if card_name == name else "")
                )
            render_preview()

        def on_mode_change():
            state["mode"] = "custom" if mode_toggle.value == "Custom colors" else "preset"
            preset_panel.set_visibility(state["mode"] == "preset")
            custom_panel.set_visibility(state["mode"] == "custom")
            render_preview()

        mode_toggle.on_value_change(lambda _e: on_mode_change())
        on_mode_change()
        render_preview()

    container.value = state["palette"]
    container.get_value = lambda: state["palette"]
    container.get_mode = lambda: state["mode"]
    container.get_custom_colors = lambda: list(state["custom"])
    container.get_range = lambda: (state["start"], state["end"])
    container.get_colors = effective_colors
    return container


def color_swatch_picker(
    label: str,
    value: str = "auto",
    options: Optional[List[Tuple[str, str]]] = None,
) -> ui.element:
    """Single-color swatches plus a native custom color picker."""
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
        with ui.row().classes("items-center gap-2 w-full drocat-swatch-row"):
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

        with ui.row().classes("w-full items-center gap-2"):
            custom_input = ui.color_input(value="#3b82f6").props("dense")
            ui.button(
                "Use custom color", on_click=lambda: select(custom_input.value)
            ).props("flat dense color=primary")

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
