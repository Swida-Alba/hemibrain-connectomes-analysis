"""Reusable single-color picker popup with alpha and a Bokeh palette.

A modal dialog containing an inline Quasar ``q-color`` grid (the same picker
panel NiceGUI shows inside a color-input popup) so a color is selected with a
single click — no nested popup. It is augmented with:
- a Bokeh palette selector with the same horizontal strip previews used by
  the standard palette controls, plus a clickable swatch grid,
- an alpha (0-1) control with an explicit opt-in opacity channel,
- a live ``rgba()`` preview.

The handle exposes ``open(initial)``, ``get_value()`` (the chosen color, with an
explicit alpha when one is set), and ``on_submit(callback)``. When ``Override
alpha`` is unchecked the raw picked color is returned (so it inherits the
visualization's global opacity); when it is checked a ``rgba(...)`` string that
carries the alpha channel is returned even for alpha == 1.0, so an explicit
"opaque" override is not mistaken for "no override" by the backend's
``color_has_explicit_alpha``.
"""

from typing import Callable, List, Optional, Tuple

from nicegui import ui

try:
    from utils.color_utils import color_to_rgba_string, color_has_explicit_alpha, extract_rgba_tuple
except ModuleNotFoundError:  # Allow importing the UI package directly from the repo.
    from src.utils.color_utils import color_to_rgba_string, color_has_explicit_alpha, extract_rgba_tuple

from .palette_picker import _embed_palette_strips, get_palette_catalog, sample_palette


def _to_hex(colors: List[str]) -> List[str]:
    """Resolve palette swatches to '#rrggbb' strings for the swatch grid."""
    return [color_to_rgba_string(c, alpha=1.0) if c.startswith("rgba") else c for c in colors]


class ColorPickerPopupHandle:
    """State + actions of one single-color picker popup."""

    def __init__(self):
        self.dialog: Optional[ui.dialog] = None
        # Inline Quasar q-color grid (single-click selection, no nested popup).
        self.q_color: Optional[ui.element] = None
        # Keep the editable value as a plain text input so the alpha control
        # has no number-input spinner/dropdown.  The paired slider is the
        # canonical 0.05-step editor; typed values are snapped to that grid.
        self.alpha: Optional[ui.input] = None
        self.alpha_slider: Optional[ui.slider] = None
        self.apply_alpha: Optional[ui.checkbox] = None
        self.preview_swatch: Optional[ui.element] = None
        self.preview_text: Optional[ui.input] = None
        self._palette_select: Optional[ui.select] = None
        self._swatch_row: Optional[ui.row] = None
        self._catalog: List[Tuple[str, List[str]]] = []
        self._current: str = "#145cff"
        self._last_alpha: float = 1.0
        self._syncing_alpha: bool = False
        self._submit_callback: Optional[Callable[[str], None]] = None
        self._committed: bool = False
        self._cancelled: bool = False
        self._session_active: bool = False

    # ------------------------------------------------------------------ value
    def get_value(self) -> str:
        """Return the chosen color, optionally carrying an explicit alpha.

        Alpha is deliberately opt-in.  This keeps a newly opened picker
        compatible with the renderer's global opacity until the user checks
        ``Override alpha``.  When checked, an alpha below 1.0 is encoded as an
        ``rgba(...)`` string.
        """
        color = self._current
        alpha = self._normalize_alpha(
            self.alpha.value if self.alpha is not None else self._last_alpha
        )
        if not self.apply_alpha or not bool(self.apply_alpha.value):
            return color
        # Always carry the alpha channel when the user opts in, even for 1.0:
        # a bare '#rrggbb' has no alpha, so the backend treats it as "no override"
        # and falls back to the global opacity instead of forcing opaque.
        return color_to_rgba_string(color, alpha=alpha)

    @staticmethod
    def _normalize_alpha(value) -> float:
        """Clamp and snap an alpha value to the 0:0.05:1 slider grid."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 1.0
        numeric = max(0.0, min(1.0, numeric))
        return round(round(numeric / 0.05) * 0.05, 2)

    def _set_alpha_value(self, value, *, refresh: bool = True) -> float:
        """Synchronize the text input and slider without recursive events."""
        normalized = self._normalize_alpha(value)
        self._last_alpha = normalized
        if self._syncing_alpha:
            return normalized
        self._syncing_alpha = True
        try:
            if self.alpha is not None:
                self.alpha.value = normalized
            if self.alpha_slider is not None:
                self.alpha_slider.value = normalized
        finally:
            self._syncing_alpha = False
        if refresh:
            self._refresh_preview()
        return normalized

    def on_submit(self, callback: Callable[[str], None]) -> None:
        """Register the callback invoked with the committed color."""
        self._submit_callback = callback

    # ------------------------------------------------------------------ picker
    def _set_picker_color(self, color: str) -> None:
        """Update the inline q-color grid model-value (no popup involved)."""
        if self.q_color is not None:
            self.q_color.props(f'model-value="{color}"')

    # ------------------------------------------------------------------ open
    def open(self, initial: str = "#145cff") -> None:
        """Open the picker, seeding it with *initial* (any supported color)."""
        self._committed = False
        self._cancelled = False
        self._session_active = True
        try:
            r, g, b, a = extract_rgba_tuple(initial or "#145cff", default_alpha=1.0)
        except (TypeError, ValueError):
            r, g, b, a = 0x14, 0x5C, 0xFF, 1.0
        # Keep the base color alpha-free.  When Override alpha is off, returning
        # an ``rgba(..., 1)`` string would accidentally override a renderer's
        # global opacity even though the user did not opt in.
        self._current = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
        self._set_picker_color(self._current)
        self._set_alpha_value(a, refresh=False)
        # Reflect an existing alpha channel: when the opened color carries an
        # explicit alpha (rgba(...), #RGBA/#RRGGBBAA …) opt into "Override alpha"
        # and show that alpha; a bare hex keeps the override unchecked so it
        # inherits the renderer's global opacity.
        if self.apply_alpha is not None:
            self.apply_alpha.value = color_has_explicit_alpha(initial or "#145cff")
        self._refresh_preview()
        if self.dialog is not None:
            self.dialog.open()

    # ------------------------------------------------------------ internals
    def _refresh_preview(self) -> None:
        rgba = self.get_value()
        if self.preview_swatch is not None:
            self.preview_swatch.style(
                "background:" + rgba + "; width:22px; height:22px; "
                "border-radius:4px; border:1px solid rgba(11,31,58,.25);"
            )
        if self.preview_text is not None:
            self.preview_text.value = rgba

    def _on_picker_change(self, event) -> None:
        raw = str(
            getattr(event, "args", None)
            or getattr(event, "value", None)
            or self._current
        )
        try:
            r, g, b, _ = extract_rgba_tuple(raw, default_alpha=1.0)
            self._current = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
        except (TypeError, ValueError):
            self._current = raw
        self._refresh_preview()

    def _on_alpha_change(self, _event=None) -> None:
        value = getattr(_event, "value", None) if _event is not None else None
        if value is None and self.alpha is not None:
            value = self.alpha.value
        # Moving the alpha control implies the user wants to apply it, so opt in.
        if self.apply_alpha is not None:
            self.apply_alpha.value = True
        self._set_alpha_value(value, refresh=True)

    def _on_alpha_slider_change(self, _event=None) -> None:
        value = getattr(_event, "value", None) if _event is not None else None
        if value is None and self.alpha_slider is not None:
            value = self.alpha_slider.value
        # Moving the slider implies the user wants to apply it, so opt in.
        if self.apply_alpha is not None:
            self.apply_alpha.value = True
        self._set_alpha_value(value, refresh=True)

    def _on_apply_alpha_change(self, _event=None) -> None:
        """Refresh the preview when the explicit-alpha opt-in changes."""
        self._refresh_preview()

    def _on_palette_change(self, _event=None) -> None:
        name = self._palette_select.value if self._palette_select else None
        colors = get_palette_catalog()
        chosen = next((c for n, c in colors if n == name), None)
        if chosen is None:
            return
        if self._swatch_row is not None:
            self._swatch_row.clear()
            with self._swatch_row:
                # Discrete palettes (<=20 colors) are shown exactly as configured.
                # Continuous palettes (>20, e.g. sequential/diverging colormaps)
                # are sampled to 40 colors evenly across their full range, so the
                # 10x4 grid spans the whole gradient instead of the first stops.
                swatch_colors = (
                    _to_hex(chosen)
                    if len(chosen) <= 20
                    else _to_hex(sample_palette(chosen, 40))
                )
                for hex_color in swatch_colors:
                    swatch = ui.element("div").classes("drocat-swatch").tooltip(hex_color).on(
                        "click", lambda v=hex_color: self._set_from_swatch(v)
                    )
                    with swatch:
                        # Circular swatch sized to fit the 10-column grid.
                        ui.element("div").style(
                            f"background:{hex_color}; width:20px; height:20px; "
                            "border-radius:50%; border:2px solid rgba(11,31,58,.15);"
                        )

    def _set_from_swatch(self, hex_color: str) -> None:
        self._current = hex_color
        self._set_picker_color(hex_color)
        self._refresh_preview()

    def _commit(self) -> None:
        """Commit once, then close the dialog.

        The dialog's ``hide`` event also commits when the user clicks outside
        the panel.  The guard makes that event idempotent when this method
        closes the dialog itself.
        """
        if self._committed or self._cancelled:
            return
        self._committed = True
        if self._submit_callback is not None:
            try:
                self._submit_callback(self.get_value())
            except Exception:
                pass
        if self.dialog is not None:
            self.dialog.close()
            # Direct handle callers (and integrations that use the picker as
            # a non-modal color editor) can commit while the dialog is not
            # open. That is a new logical commit each time; keep the guard for
            # a session started through ``open`` so a hide event cannot apply
            # the same color twice.
            if not self._session_active:
                self._committed = False

    def _cancel(self) -> None:
        """Close without applying the current value."""
        self._cancelled = True
        if self.dialog is not None:
            self.dialog.close()

    def _on_dialog_hide(self, _event=None) -> None:
        """Apply a color when a dismissible dialog closes by outside click."""
        if not self._committed and not self._cancelled:
            self._commit()


def color_picker_popup(
    value: str = "#145cff",
    card_id: str = "card-color-picker-popup",
) -> ColorPickerPopupHandle:
    """Build the single-color picker dialog and return its handle.

    Use ``handle.on_submit(cb)`` to receive the committed color and
    ``handle.open(initial=...)`` to show it.
    """
    handle = ColorPickerPopupHandle()
    catalog = get_palette_catalog()
    handle._catalog = catalog
    palette_names = [name for name, _ in catalog]

    with ui.dialog().props(f'id="{card_id}"') as dialog:
        handle.dialog = dialog
        # QDialog emits ``hide`` for an outside click or Escape.  Treat that
        # dismiss action as the color picker's focus-out commit, while the
        # explicit Cancel button opts out through ``_cancel``.
        dialog.on("hide", handle._on_dialog_hide)
        with ui.card().classes("w-96"):
            ui.label("Single color picker").classes("drocat-mini-label")
            # Inline q-color grid: a click on a cell picks the colour directly.
            handle.q_color = ui.element("q-color").props(
                f'model-value="{value}"'
            ).on("change", handle._on_picker_change)
            with ui.column().classes("w-full gap-1"):
                handle._palette_select = ui.select(
                    palette_names,
                    value=palette_names[0] if palette_names else None,
                    label="Picker palette",
                ).props("outlined dense").classes("w-full")
                # Keep the picker palette selector visually consistent with
                # the standard Neuron Colors > Custom Colors control.  The
                # option slot shows each palette as a horizontal strip before
                # it is selected, without changing the selected value from
                # the palette name.
                _embed_palette_strips(
                    handle._palette_select,
                    palette_names,
                    dict(catalog),
                )
                handle._palette_select.on_value_change(handle._on_palette_change)
                # A 10-column CSS grid so every candidate palette shows ten swatches
                # per row regardless of the palette width.
                handle._swatch_row = ui.row().classes("w-full drocat-swatch-grid")
                handle._on_palette_change()
            with ui.row().classes("w-full items-center gap-1"):
                handle.apply_alpha = ui.checkbox(
                    "Override alpha", value=False
                ).props("dense").classes("shrink-0").tooltip(
                    "Embed the alpha value in the submitted color; leave off "
                    "to inherit the visualization's global opacity."
                )
                handle.apply_alpha.on_value_change(handle._on_apply_alpha_change)
                handle.alpha_slider = ui.slider(
                    min=0,
                    max=1,
                    step=0.05,
                    value=1.0,
                ).props("label-always").classes("grow min-w-0").tooltip(
                    "Alpha from 0 to 1 in 0.05 steps."
                )
                handle.alpha_slider.on_value_change(handle._on_alpha_slider_change)
                # A text input deliberately avoids the q-number spinner/dropdown.
                handle.alpha = ui.input(
                    label="Alpha (0-1)",
                    value=1.0,
                ).props("outlined dense inputmode=decimal").classes("w-24 shrink-0").tooltip(
                    "Type an opacity; it is snapped to 0.05 steps."
                ).on_value_change(handle._on_alpha_change)
            with ui.row().classes("items-center gap-2 w-full"):
                handle.preview_swatch = ui.element("div")
                handle.preview_text = ui.input(
                    label="Color format",
                    value="",
                ).props("dense outlined").classes("grow")
            handle._refresh_preview()
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=handle._cancel).props("flat outline dense")
                ui.button("OK", on_click=handle._commit).props("color=primary dense")

    return handle
