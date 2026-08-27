"""Free-scrolling execution log element.

NiceGUI's ``ui.log`` snaps the viewport back to the newest line whenever any
content arrives, so an actively streaming execution log cannot be scrolled
up to read earlier lines. ``FreeLog`` swaps in the sibling ``free_log.js``
component: the view follows the tail only while it already rests at the
bottom, and otherwise stays put (line appends happen below it, and trims of
old lines at the ``max_lines`` cap are compensated). The Python API is
inherited unchanged from ``ui.log`` (``push``, ``max_lines``, ``clear``),
so callers only need to swap the class.
"""

from nicegui.elements.log import Log


class FreeLog(Log, component='free_log.js'):
    """Drop-in replacement for ``ui.log`` with a freely scrollable viewport."""
