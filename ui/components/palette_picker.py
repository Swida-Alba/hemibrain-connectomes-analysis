"""
Visual color palette tools for DROCAT.

- ``palette_picker``: preset palette cards with full-gradient previews.
- ``palette_editor``: palette editor with ONE interactive preview row -
  drag-and-drop reordering of discrete colors, a live range slider, and a
  reset button beside the preview (the current state is the only preview).
  Custom colors (mode toggle) are added via an input with color picker and
  reordered by dragging the list rows.
- ``color_swatch_picker``: single-color swatches with a custom color input.
"""

from typing import Callable, List, Optional, Tuple

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


def assign_palette_colors(colors: List[str], n: int) -> List[str]:
    """Assign colors in their exact displayed order, cycling only if needed."""
    if n <= 0 or not colors:
        return []
    return [colors[index % len(colors)] for index in range(n)]


def normalize_palette_range(start_pct: int, end_pct: int) -> Tuple[int, int]:
    """Clamp a palette range and guarantee at least one percentage point."""
    start = max(0, min(99, int(start_pct)))
    end = max(1, min(100, int(end_pct)))
    if end <= start:
        end = start + 1
    return start, end


def palette_slice(colors: List[str], start_pct: int, end_pct: int) -> List[str]:
    """Select a sub-range of a palette by percentage (0-100)."""
    if not colors:
        return []
    start_pct, end_pct = normalize_palette_range(start_pct, end_pct)
    start_idx = round(start_pct / 100 * (len(colors) - 1))
    end_idx = round(end_pct / 100 * (len(colors) - 1))
    return colors[start_idx:end_idx + 1]


def move_color(colors: List[str], index: int, delta: int) -> List[str]:
    """Return a copy with one color moved left/right by ``delta`` places."""
    reordered = list(colors)
    target = index + delta
    if index < 0 or index >= len(reordered) or target < 0 or target >= len(reordered):
        return reordered
    color = reordered.pop(index)
    reordered.insert(target, color)
    return reordered


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
    if len(colors) > 20 and click is None:
        strip = ui.element("div").classes(f"drocat-palette-strip {classes}").style(
            _gradient_style(colors, height)
        )
        return
    display_colors = sample_palette(colors, 20) if len(colors) > 20 else colors
    with ui.row().classes(f"gap-0 w-full drocat-palette-swatches {classes}"):
        for color in display_colors:
            sw = ui.element("div").style(
                f"background:{color}; flex:1; min-width:5px; height:{height}px;"
            )
            if click is not None:
                sw.on("click", lambda c=color: click(c))


# Drag-and-drop reordering of discrete palette swatches. The drop handler
# computes the insertion index from the cursor position over the swatch row
# and emits {from, to} to the server (which reorders and re-renders).
_DRAG_OVER_JS = (
    "(event) => { event.preventDefault(); event.dataTransfer.dropEffect = 'move'; }"
)
_DRAG_START_JS = (
    "(event) => { event.dataTransfer.setData('text/plain', "
    "String(event.currentTarget.dataset.index)); "
    "event.dataTransfer.effectAllowed = 'move'; }"
)


def _drop_js(axis: str = "x") -> str:
    """Drop-handler JS computing the insertion index from the cursor
    position along an axis ('x' for horizontal swatch rows, 'y' for
    vertical lists). Emits {from, to} to the server."""
    coord = "clientX" if axis == "x" else "clientY"
    pos = "left" if axis == "x" else "top"
    size = "width" if axis == "x" else "height"
    return (
        "(event) => { event.preventDefault(); "
        "const from = Number(event.dataTransfer.getData('text/plain')); "
        "const rect = event.currentTarget.getBoundingClientRect(); "
        f"const {axis} = event.{coord} - rect.{pos}; "
        "let to = 0; "
        "for (const child of event.currentTarget.querySelectorAll('[draggable=\"true\"]')) { "
        f"  const c = child.getBoundingClientRect(); "
        f"  if ({axis} > c.{pos} + c.{size} / 2 - rect.{pos}) to++; "
        "} "
        "emit({from, to}); }"
    )


def _render_draggable_swatches(
    colors: List[str],
    height: int = 20,
    on_drop: Optional[Callable] = None,
) -> None:
    """Render individual draggable swatches for drag-and-drop reordering."""
    with ui.row().classes(
        "gap-0 w-full drocat-palette-swatches"
    ).on("dragover", None, js_handler=_DRAG_OVER_JS) as row:
        for index, color in enumerate(colors):
            sw = ui.element("div").style(
                f"background:{color}; flex:1; min-width:5px; height:{height}px;"
            ).tooltip("Drag to reorder")
            sw._props["draggable"] = "true"
            sw._props["data-index"] = str(index)
            sw.on("dragstart", None, js_handler=_DRAG_START_JS)
    row.on("drop", on_drop, js_handler=_drop_js("x"))


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
    on_change: Optional[Callable] = None,
) -> ui.element:
    """
    Palette editor with ONE interactive preview row.

    Features:
    - Preset palettes from bokeh.palettes with full-gradient previews
    - The preview row shows the current state only: drag swatches to
      reorder discrete palettes, use the range slider to select part of
      the palette live, and hit reset (beside the preview) to restore the
      original order and the full range.
    - Custom colors: add single colors via the color input/picker; the
      custom list reorders with the same drag-and-drop as the preset
      preview, and entries can be removed

    ``on_change`` fires when the user manually edits the palette state
    (picking a preset card, drag-reordering, adjusting the range, resetting);
    callers use it to stop auto-following e.g. the background color. The
    returned container also gains ``set_palette(name)`` for programmatic
    palette switching, which does NOT fire ``on_change``.
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
        "preset_orders": {},
    }
    cards: List[Tuple[ui.element, str]] = []

    def palette_colors(name: Optional[str] = None) -> List[str]:
        palette_name = name or state["palette"]
        original = list(dict(catalog)[palette_name])
        return list(state["preset_orders"].get(palette_name, original))

    def is_discrete() -> bool:
        return len(palette_colors()) <= 20

    def effective_slice() -> List[str]:
        return palette_slice(
            palette_colors(), state["start"], state["end"]
        )

    def effective_colors() -> List[str]:
        if state["mode"] == "custom" and state["custom"]:
            return state["custom"]
        return effective_slice()

    def slice_start_index(colors: List[str]) -> int:
        """Index in ``colors`` where the displayed range slice starts."""
        return round(state["start"] / 100 * (len(colors) - 1))

    with ui.column().classes("w-full gap-1") as container:
        ui.label(label).classes("drocat-mini-label")

        mode_toggle = ui.toggle(
            ["Preset palette", "Custom colors"], value="Preset palette"
        ).props("dense outlined")

        # ---- Single interactive preview row (current state only) ----
        preview = ui.row().classes("items-center gap-2 w-full drocat-palette-preview")
        with preview:
            swatch_area = ui.row().classes(
                "gap-0 flex-grow drocat-palette-swatches"
            )
            reset_button = ui.button(
                icon="restart_alt", on_click=lambda: reset_all()
            ).props(
                'flat dense round aria-label="Reset palette"'
            ).tooltip("Reset order and range")

        status_label = ui.label("").classes("text-caption drocat-muted")

        def render_preview():
            colors = effective_colors()
            # Drag-and-drop only makes sense for discrete color lists: preset
            # palettes with <= 20 colors and any non-empty custom list.
            draggable = (
                (state["mode"] == "preset" and is_discrete())
                or (state["mode"] == "custom" and bool(state["custom"]))
            )
            swatch_area.clear()
            with swatch_area:
                if draggable:
                    _render_draggable_swatches(
                        colors[:32], height=20, on_drop=handle_drop
                    )
                else:
                    _render_color_strip(
                        colors[:32], height=20, classes="w-full"
                    )
            if state["mode"] == "custom" and state["custom"]:
                status = "custom colors"
            else:
                status = f"{state['palette']}  {state['start']}–{state['end']}%"
            if draggable:
                status += " · drag swatches to reorder"
            status_label.text = status

        def handle_drop(e):
            """Reorder the displayed colors after a drag-and-drop."""
            args = e.args or {}
            from_idx = int(args.get("from", 0) or 0)
            to_idx = int(args.get("to", 0) or 0)
            if state["mode"] == "custom" and state["custom"]:
                colors = list(state["custom"])
                if not (0 <= from_idx < len(colors)):
                    return
                to_idx = max(0, min(to_idx, len(colors)))
                item = colors.pop(from_idx)
                colors.insert(
                    to_idx - 1 if from_idx < to_idx else to_idx, item
                )
                state["custom"] = colors
                render_custom_list()
            else:
                colors = palette_colors()
                offset = slice_start_index(colors)
                displayed = effective_slice()
                if not (0 <= from_idx < len(displayed)):
                    return
                to_idx = max(0, min(to_idx, len(displayed)))
                item = colors.pop(offset + from_idx)
                insert_at = offset + to_idx
                if from_idx < to_idx:
                    insert_at -= 1
                colors.insert(insert_at, item)
                state["preset_orders"][state["palette"]] = colors
            render_preview()
            if on_change:
                on_change()

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
                # Thumb-position bubbles would overlap neighboring text, so
                # the start/end values are shown as lateral labels aligned
                # with the track ends instead (they update live on drag).
                range_start_label = ui.label("0").classes(
                    "text-caption font-mono drocat-muted"
                ).style("width: 26px; text-align: right; flex: none;")
                range_slider = ui.range(
                    min=0, max=100, step=1, value={"min": 0, "max": 100}
                ).props("dense color=primary").style(
                    # q-range renders at full width; a zero flex-basis lets it
                    # share the row with the lateral labels and grow into the
                    # leftover space (single line, labels at the track ends).
                    "flex: 1 1 0%; min-width: 80px;"
                )
                range_end_label = ui.label("100").classes(
                    "text-caption font-mono drocat-muted"
                ).style("width: 30px; flex: none;")

            def apply_range(e):
                value = e.args if isinstance(e.args, dict) else {}
                start, end = normalize_palette_range(
                    value.get("min", state["start"]),
                    value.get("max", state["end"]),
                )
                state["start"], state["end"] = start, end
                range_slider.value = {"min": start, "max": end}
                range_start_label.text = str(start)
                range_end_label.text = str(end)
                render_preview()
                if on_change:
                    on_change()

            range_slider.on(
                "update:model-value", apply_range,
                throttle=0.1, leading_events=True, trailing_events=True,
            )

        # ---------------- Custom panel ----------------
        with ui.column().classes("w-full gap-1") as custom_panel:
            with ui.row().classes("w-full items-center gap-2"):
                color_input = ui.color_input(value="#145cff").props("dense")
                ui.button(
                    "Add color", icon="add", on_click=lambda: add_custom_color()
                ).props("flat dense color=primary")
            ui.label(
                "Pick a color and press Add; drag rows to reorder."
            ).classes("text-caption drocat-muted")

            # Draggable custom-color rows: same drag-and-drop mechanism as
            # the preset preview row, but along the vertical list axis.
            custom_list = ui.column().classes("w-full gap-1").on(
                "dragover", None, js_handler=_DRAG_OVER_JS
            ).on("drop", handle_drop, js_handler=_drop_js("y"))

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
                        ) as row:
                            row._props["draggable"] = "true"
                            row._props["data-index"] = str(index)
                            row.on("dragstart", None, js_handler=_DRAG_START_JS)
                            ui.element("div").style(
                                f"background:{color}; width:24px; height:24px;"
                                "border-radius:6px; border:1px solid rgba(11,31,58,.15);"
                            )
                            ui.label(color).classes(
                                "text-caption font-mono drocat-muted drocat-truncate"
                            ).classes("flex-grow")
                            ui.button(
                                "✕", on_click=lambda i=index: remove_custom(i)
                            ).props(
                                'flat dense round color=negative '
                                'aria-label="Remove custom color"'
                            ).tooltip("Remove color")

            def add_single_color(color: str):
                if color and color not in state["custom"]:
                    state["custom"].append(color)
                    render_custom_list()
                    render_preview()

            def add_custom_color():
                add_single_color(color_input.value or "#145cff")

            def remove_custom(index: int):
                del state["custom"][index]
                render_custom_list()
                render_preview()

            render_custom_list()

        def apply_palette(name: str, notify: bool = False):
            state["palette"] = name
            container.value = name
            for element, card_name in cards:
                element.classes(
                    replace="drocat-palette-card"
                    + (" selected" if card_name == name else "")
                )
            range_slider.value = {"min": state["start"], "max": state["end"]}
            range_start_label.text = str(state["start"])
            range_end_label.text = str(state["end"])
            render_preview()
            if notify and on_change:
                on_change()

        def select_preset(name: str):
            apply_palette(name, notify=True)

        def reset_all():
            """Restore the original palette order/range (or clear custom colors)."""
            if state["mode"] == "custom":
                state["custom"] = []
                render_custom_list()
            else:
                state["preset_orders"].pop(state["palette"], None)
                state["start"], state["end"] = 0, 100
                range_slider.value = {"min": 0, "max": 100}
                range_start_label.text = "0"
                range_end_label.text = "100"
            render_preview()
            if on_change:
                on_change()

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
    container.get_palette_order = lambda: list(palette_colors())
    container.get_range = lambda: (state["start"], state["end"])
    container.get_colors = effective_colors
    container.set_palette = lambda name: apply_palette(name, notify=False)
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
