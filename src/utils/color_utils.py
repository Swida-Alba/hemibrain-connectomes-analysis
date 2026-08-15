"""
Color Utilities Module
======================

This module provides comprehensive color parsing, standardization, and conversion
utilities for the visualization components in the connectome analysis toolkit.

Supported Input Formats
-----------------------
- **Named colors**: 'red', 'blue', 'lightgray', 'rebeccapurple', etc.
- **Hex colors**: '#RGB', '#RGBA', '#RRGGBB', '#RRGGBBAA' (the leading '#'
  is optional for hexadecimal strings)
- **RGB tuples/lists**: (255, 0, 0) or (1.0, 0.0, 0.0). Integer channels
  use 0-255; floating-point channels in 0-1 use normalized RGB.
- **RGBA tuples/lists**: (255, 0, 0, 128), (1.0, 0.0, 0.0, 0.5)
- **CSS rgb/rgba strings**: comma or modern space/slash syntax, including
  percentages and normalized decimal channels
- **CSS hsl/hsla strings**: e.g. 'hsl(210 100% 50% / 50%)'
- **Bokeh palette values**: bokeh.palettes.Category10[10], etc.

When a palette contains both alpha-bearing and alpha-less colors, only the
alpha-bearing entries override the caller's default alpha. Entries without an
alpha inherit that default.

Standard Output Format
----------------------
All colors are standardized to RGBA format: 'rgba(r, g, b, a)' where:
- r, g, b: integers 0-255
- a: float 0.0-1.0

Usage Examples
--------------
>>> from src.utils.color_utils import standardize_color, standardize_color_list
>>> 
>>> # Single color conversion
>>> standardize_color('red')
'rgba(255, 0, 0, 1.0)'
>>> standardize_color('#ff0000')
'rgba(255, 0, 0, 1.0)'
>>> standardize_color((255, 0, 0))
'rgba(255, 0, 0, 1.0)'
>>> standardize_color('rgba(255, 0, 0, 0.5)')
'rgba(255, 0, 0, 0.5)'
>>>
>>> # Color list/palette conversion  
>>> standardize_color_list(['red', '#00ff00', (0, 0, 255)])
['rgba(255, 0, 0, 1.0)', 'rgba(0, 255, 0, 1.0)', 'rgba(0, 0, 255, 1.0)']
>>>
>>> # With default alpha
>>> standardize_color('red', default_alpha=0.5)
'rgba(255, 0, 0, 0.5)'
"""

import colorsys
import math
import re
from numbers import Integral, Real
from typing import Union, List, Tuple, Any


# CSS named colors (subset of most common ones)
CSS_NAMED_COLORS = {
    'aliceblue': (240, 248, 255),
    'antiquewhite': (250, 235, 215),
    'aqua': (0, 255, 255),
    'aquamarine': (127, 255, 212),
    'azure': (240, 255, 255),
    'beige': (245, 245, 220),
    'bisque': (255, 228, 196),
    'black': (0, 0, 0),
    'blanchedalmond': (255, 235, 205),
    'blue': (0, 0, 255),
    'blueviolet': (138, 43, 226),
    'brown': (165, 42, 42),
    'burlywood': (222, 184, 135),
    'cadetblue': (95, 158, 160),
    'chartreuse': (127, 255, 0),
    'chocolate': (210, 105, 30),
    'coral': (255, 127, 80),
    'cornflowerblue': (100, 149, 237),
    'cornsilk': (255, 248, 220),
    'crimson': (220, 20, 60),
    'cyan': (0, 255, 255),
    'darkblue': (0, 0, 139),
    'darkcyan': (0, 139, 139),
    'darkgoldenrod': (184, 134, 11),
    'darkgray': (169, 169, 169),
    'darkgrey': (169, 169, 169),
    'darkgreen': (0, 100, 0),
    'darkkhaki': (189, 183, 107),
    'darkmagenta': (139, 0, 139),
    'darkolivegreen': (85, 107, 47),
    'darkorange': (255, 140, 0),
    'darkorchid': (153, 50, 204),
    'darkred': (139, 0, 0),
    'darksalmon': (233, 150, 122),
    'darkseagreen': (143, 188, 143),
    'darkslateblue': (72, 61, 139),
    'darkslategray': (47, 79, 79),
    'darkslategrey': (47, 79, 79),
    'darkturquoise': (0, 206, 209),
    'darkviolet': (148, 0, 211),
    'deeppink': (255, 20, 147),
    'deepskyblue': (0, 191, 255),
    'dimgray': (105, 105, 105),
    'dimgrey': (105, 105, 105),
    'dodgerblue': (30, 144, 255),
    'firebrick': (178, 34, 34),
    'floralwhite': (255, 250, 240),
    'forestgreen': (34, 139, 34),
    'fuchsia': (255, 0, 255),
    'gainsboro': (220, 220, 220),
    'ghostwhite': (248, 248, 255),
    'gold': (255, 215, 0),
    'goldenrod': (218, 165, 32),
    'gray': (128, 128, 128),
    'grey': (128, 128, 128),
    'green': (0, 128, 0),
    'greenyellow': (173, 255, 47),
    'honeydew': (240, 255, 240),
    'hotpink': (255, 105, 180),
    'indianred': (205, 92, 92),
    'indigo': (75, 0, 130),
    'ivory': (255, 255, 240),
    'khaki': (240, 230, 140),
    'lavender': (230, 230, 250),
    'lavenderblush': (255, 240, 245),
    'lawngreen': (124, 252, 0),
    'lemonchiffon': (255, 250, 205),
    'lightblue': (173, 216, 230),
    'lightcoral': (240, 128, 128),
    'lightcyan': (224, 255, 255),
    'lightgoldenrodyellow': (250, 250, 210),
    'lightgray': (211, 211, 211),
    'lightgrey': (211, 211, 211),
    'lightgreen': (144, 238, 144),
    'lightpink': (255, 182, 193),
    'lightsalmon': (255, 160, 122),
    'lightseagreen': (32, 178, 170),
    'lightskyblue': (135, 206, 250),
    'lightslategray': (119, 136, 153),
    'lightslategrey': (119, 136, 153),
    'lightsteelblue': (176, 196, 222),
    'lightyellow': (255, 255, 224),
    'lime': (0, 255, 0),
    'limegreen': (50, 205, 50),
    'linen': (250, 240, 230),
    'magenta': (255, 0, 255),
    'maroon': (128, 0, 0),
    'mediumaquamarine': (102, 205, 170),
    'mediumblue': (0, 0, 205),
    'mediumorchid': (186, 85, 211),
    'mediumpurple': (147, 112, 219),
    'mediumseagreen': (60, 179, 113),
    'mediumslateblue': (123, 104, 238),
    'mediumspringgreen': (0, 250, 154),
    'mediumturquoise': (72, 205, 204),
    'mediumvioletred': (199, 21, 133),
    'midnightblue': (25, 25, 112),
    'mintcream': (245, 255, 250),
    'mistyrose': (255, 228, 225),
    'moccasin': (255, 228, 181),
    'navajowhite': (255, 222, 173),
    'navy': (0, 0, 128),
    'oldlace': (253, 245, 230),
    'olive': (128, 128, 0),
    'olivedrab': (107, 142, 35),
    'orange': (255, 165, 0),
    'orangered': (255, 69, 0),
    'orchid': (218, 112, 214),
    'palegoldenrod': (238, 232, 170),
    'palegreen': (152, 251, 152),
    'paleturquoise': (175, 238, 238),
    'palevioletred': (219, 112, 147),
    'papayawhip': (255, 239, 213),
    'peachpuff': (255, 218, 185),
    'peru': (205, 133, 63),
    'pink': (255, 192, 203),
    'plum': (221, 160, 221),
    'powderblue': (176, 224, 230),
    'purple': (128, 0, 128),
    'rebeccapurple': (102, 51, 153),
    'red': (255, 0, 0),
    'rosybrown': (188, 143, 143),
    'royalblue': (65, 105, 225),
    'saddlebrown': (139, 69, 19),
    'salmon': (250, 128, 114),
    'sandybrown': (244, 164, 96),
    'seagreen': (46, 139, 87),
    'seashell': (255, 245, 238),
    'sienna': (160, 82, 45),
    'silver': (192, 192, 192),
    'skyblue': (135, 206, 235),
    'slateblue': (106, 90, 205),
    'slategray': (112, 128, 144),
    'slategrey': (112, 128, 144),
    'snow': (255, 250, 250),
    'springgreen': (0, 255, 127),
    'steelblue': (70, 130, 180),
    'tan': (210, 180, 140),
    'teal': (0, 128, 128),
    'thistle': (216, 191, 216),
    'tomato': (255, 99, 71),
    'turquoise': (64, 224, 208),
    'violet': (238, 130, 238),
    'wheat': (245, 222, 179),
    'white': (255, 255, 255),
    'whitesmoke': (245, 245, 245),
    'yellow': (255, 255, 0),
    'yellowgreen': (154, 205, 50),
}


def _is_numeric(value: Any) -> bool:
    """Return whether *value* is a finite real number (excluding booleans)."""
    if isinstance(value, bool):
        return False
    try:
        return isinstance(value, Real) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _as_color_sequence(value: Any) -> Any:
    """Convert numpy-like color arrays to ordinary Python lists when possible."""
    if isinstance(value, (str, bytes, tuple, list)):
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except (TypeError, ValueError):
            pass
    return value


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _normalize_alpha(value: Any, default: float = 1.0) -> float:
    """Normalize alpha from 0–1, percentage, or 0–255 notation."""
    if value is None:
        value = default
    if isinstance(value, str):
        token = value.strip()
        if token.endswith("%"):
            try:
                return _clamp(float(token[:-1]) / 100.0, 0.0, 1.0)
            except ValueError as exc:
                raise ValueError(f"Invalid alpha value: {value!r}") from exc
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f"Invalid alpha value: {value!r}") from exc
    if not _is_numeric(value):
        raise ValueError(f"Alpha must be numeric, got {value!r}")
    number = float(value)
    if number > 1.0:
        number /= 255.0
    return _clamp(number, 0.0, 1.0)


def _is_color_sequence(value: Any) -> bool:
    """Return whether *value* is one RGB/RGBA tuple or list.

    RGB channels must be numeric.  The alpha channel may additionally use
    the same percentage/string notation accepted by CSS alpha values.
    """
    value = _as_color_sequence(value)
    if not isinstance(value, (tuple, list)) or len(value) not in (3, 4):
        return False
    if not all(_is_numeric(channel) for channel in value[:3]):
        return False
    if len(value) == 4:
        if value[3] is None:
            return False
        try:
            _normalize_alpha(value[3])
        except ValueError:
            return False
    return True


def _parse_hex_color(hex_str: str) -> Tuple[int, int, int, float]:
    """Parse ``#RGB``, ``#RGBA``, ``#RRGGBB`` or ``#RRGGBBAA``."""
    value = hex_str.strip().lstrip("#")
    if len(value) not in (3, 4, 6, 8) or not re.fullmatch(r"[0-9a-fA-F]+", value):
        raise ValueError(
            f"Invalid hex color format: {hex_str!r}; use #RGB, #RGBA, "
            "#RRGGBB, or #RRGGBBAA"
        )
    if len(value) in (3, 4):
        value = "".join(char * 2 for char in value)
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    a = int(value[6:8], 16) / 255.0 if len(value) == 8 else 1.0
    return (r, g, b, a)


def _split_css_components(body: str) -> List[str]:
    """Split comma or modern space/slash CSS color components."""
    body = body.strip()
    if "," in body:
        return [part.strip() for part in body.split(",") if part.strip()]
    if "/" in body:
        color_part, alpha_part = body.split("/", 1)
        return color_part.split() + [alpha_part.strip()]
    return body.split()


def _token_is_fractional(token: str) -> bool:
    return "." in token or "e" in token.lower()


def _parse_rgb_channel(value: Any, normalized: bool = False) -> int:
    """Parse one RGB channel, accepting 0–255, normalized 0–1, or percent."""
    if isinstance(value, str) and value.strip().endswith("%"):
        # Divide explicitly instead of multiplying by 2.55; the latter can
        # turn an exact 50% channel into 127.499999... before rounding.
        number = float(value.strip()[:-1]) * 255.0 / 100.0
    else:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid RGB channel: {value!r}") from exc
        if normalized:
            number *= 255.0
    return int(round(_clamp(number, 0.0, 255.0)))


def _parse_rgb_string(rgb_str: str) -> Tuple[int, int, int, float]:
    """Parse CSS ``rgb()/rgba()`` including modern space/slash syntax."""
    match = re.fullmatch(r"\s*(rgba?)\s*\((.*)\)\s*", rgb_str, re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid rgb/rgba format: {rgb_str!r}")

    function_name = match.group(1).lower()
    components = _split_css_components(match.group(2))
    if len(components) not in (3, 4):
        raise ValueError(
            f"{function_name}() requires 3 RGB channels and optional alpha: {rgb_str!r}"
        )

    rgb_components = components[:3]
    has_percent = any(str(component).strip().endswith("%") for component in rgb_components)
    numeric_components = []
    for component in rgb_components:
        try:
            numeric_components.append(float(str(component).strip()))
        except ValueError:
            numeric_components.append(255.0)
    # CSS integer channels are 0–255.  Decimal channels in the 0–1 range are
    # also accepted as normalized RGB for the color editor's common scientific
    # notation (e.g. rgb(1.0, 0.0, 0.5)).
    normalized = (
        not has_percent
        and all(0.0 <= value <= 1.0 for value in numeric_components)
        and any(_token_is_fractional(str(component).strip()) for component in rgb_components)
    )
    r, g, b = (
        _parse_rgb_channel(component, normalized=normalized)
        for component in rgb_components
    )
    alpha = (
        _normalize_alpha(components[3], default=1.0)
        if len(components) == 4
        else 1.0
    )
    return (r, g, b, alpha)


def _parse_hsl_string(hsl_str: str) -> Tuple[int, int, int, float]:
    """Parse CSS ``hsl()/hsla()`` into an RGBA tuple."""
    match = re.fullmatch(r"\s*(hsla?)\s*\((.*)\)\s*", hsl_str, re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid hsl/hsla format: {hsl_str!r}")
    components = _split_css_components(match.group(2))
    if len(components) not in (3, 4):
        raise ValueError(
            f"hsl() requires hue, saturation, lightness, and optional alpha: {hsl_str!r}"
        )

    hue_token = components[0].strip().lower()
    if hue_token.endswith("turn"):
        hue = float(hue_token[:-4]) * 360.0
    elif hue_token.endswith("rad"):
        hue = math.degrees(float(hue_token[:-3]))
    elif hue_token.endswith("deg"):
        hue = float(hue_token[:-3])
    else:
        hue = float(hue_token)

    def parse_percentage(token: str, name: str) -> float:
        token = token.strip()
        if token.endswith("%"):
            value = float(token[:-1]) / 100.0
        else:
            value = float(token)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be 0–100% or 0–1, got {token!r}")
        return value

    saturation = parse_percentage(components[1], "Saturation")
    lightness = parse_percentage(components[2], "Lightness")
    red, green, blue = colorsys.hls_to_rgb(
        (hue % 360.0) / 360.0, lightness, saturation
    )
    alpha = (
        _normalize_alpha(components[3], default=1.0)
        if len(components) == 4
        else 1.0
    )
    return (
        int(round(red * 255)),
        int(round(green * 255)),
        int(round(blue * 255)),
        alpha,
    )


def _parse_tuple_color(color_tuple: Tuple, default_alpha: float = 1.0) -> Tuple[int, int, int, float]:
    """Parse RGB/RGBA tuples or lists in absolute or normalized notation."""
    color_tuple = _as_color_sequence(color_tuple)
    if not isinstance(color_tuple, (tuple, list)) or len(color_tuple) not in (3, 4):
        raise ValueError(
            f"Color tuple must contain 3 RGB or 4 RGBA values, got {color_tuple!r}"
        )
    if not all(_is_numeric(value) for value in color_tuple[:3]):
        raise ValueError(f"RGB tuple channels must be numeric, got {color_tuple!r}")
    if len(color_tuple) == 4 and color_tuple[3] is None:
        raise ValueError(f"RGBA tuple alpha must be numeric or a percentage, got {color_tuple!r}")

    rgb_values = color_tuple[:3]
    # The unambiguous normalized tuple form is floats in 0–1. Integer tuples
    # such as (1, 0, 0) remain the conventional 0–255 form.
    normalized = (
        all(0.0 <= float(value) <= 1.0 for value in rgb_values)
        and any(not isinstance(value, Integral) for value in rgb_values)
    )
    rgb = tuple(_parse_rgb_channel(value, normalized=normalized) for value in rgb_values)
    alpha = (
        _normalize_alpha(color_tuple[3], default=default_alpha)
        if len(color_tuple) == 4
        else _normalize_alpha(default_alpha, default=1.0)
    )
    return (rgb[0], rgb[1], rgb[2], alpha)


def color_has_explicit_alpha(color: Any) -> bool:
    """Return whether *color* carries its own alpha channel.

    This deliberately distinguishes a color with no alpha (which should use
    the caller's global opacity) from a color whose alpha happens to normalize
    to ``1.0``.  It understands RGBA tuples, ``#RGBA``/``#RRGGBBAA``, CSS
    ``rgba()/hsla()`` and modern CSS ``/ alpha`` syntax, including mixed lists.
    """
    color = _as_color_sequence(color)
    if isinstance(color, str):
        value = color.strip().lower()
        if value in ("transparent", "none"):
            return True
        if value.startswith("#"):
            return len(value) in (5, 9)
        if len(value) in (4, 8) and re.fullmatch(r"[0-9a-f]+", value):
            return True
        if re.match(r"^(rgba|hsla)\s*\(", value):
            return True
        modern_match = re.match(r"^(rgb|hsl)\s*\((.*)\)$", value)
        if modern_match:
            return len(_split_css_components(modern_match.group(2))) == 4
        return False
    if isinstance(color, (tuple, list)):
        if _is_color_sequence(color) and len(color) == 4:
            return True
        return any(color_has_explicit_alpha(item) for item in color)
    return False


def standardize_color(
    color: Any,
    default_alpha: float = 1.0,
    output_format: str = 'rgba'
) -> Any:
    """
    Standardize a color input to a consistent RGBA format string.
    
    This function accepts multiple color input formats and converts them
    to a standard RGBA string format suitable for Plotly/CSS usage.
    
    Parameters
    ----------
    color : str, tuple, list
        Color in any supported format:
        - Named colors: 'red', 'blue', 'lightgray', etc.
        - Hex colors: '#RGB', '#RGBA', '#RRGGBB', or '#RRGGBBAA'
        - RGB tuples/lists: 0-255 channels or normalized 0-1 floats
        - RGBA tuples/lists: byte or normalized channels with alpha
        - CSS strings: rgb()/rgba() and hsl()/hsla(), including modern
          space/slash syntax and percentage alpha
    default_alpha : float, default 1.0
        Default alpha value to use if not specified in the color
    output_format : str, default 'rgba'
        Output format. Options:
        - 'rgba': Returns 'rgba(r, g, b, a)' string
        - 'rgb': Returns 'rgb(r, g, b)' string (ignores alpha)
        - 'hex': Returns '#rrggbb' string (ignores alpha)
        - 'hex_alpha': Returns '#rrggbbaa' string
        - 'tuple': Returns (r, g, b, a) tuple
        - 'tuple_normalized': Returns (r, g, b, a) with 0-1 values
        
    Returns
    -------
    str or tuple
        Standardized color in the requested format
        
    Raises
    ------
    ValueError
        If the color format cannot be parsed
        
    Examples
    --------
    >>> standardize_color('red')
    'rgba(255, 0, 0, 1.0)'
    >>> standardize_color('#ff0000')
    'rgba(255, 0, 0, 1.0)'
    >>> standardize_color((255, 0, 0))
    'rgba(255, 0, 0, 1.0)'
    >>> standardize_color('rgba(255, 0, 0, 0.5)')
    'rgba(255, 0, 0, 0.5)'
    >>> standardize_color('red', default_alpha=0.5)
    'rgba(255, 0, 0, 0.5)'
    >>> standardize_color('red', output_format='hex')
    '#ff0000'
    """
    if color is None:
        raise ValueError("Color cannot be None")
    default_alpha = _normalize_alpha(default_alpha, default=1.0)
    color = _as_color_sequence(color)

    if isinstance(color, str):
        color_stripped = color.strip()
        color_lower = color_stripped.lower()
        if color_lower == 'auto':
            return 'auto'
        if color_lower == 'transparent':
            r, g, b, a = 0, 0, 0, 0.0
        elif color_stripped.startswith('#') or (
            len(color_stripped) in (3, 4, 6, 8)
            and re.fullmatch(r"[0-9a-fA-F]+", color_stripped)
        ):
            r, g, b, a = _parse_hex_color(color_stripped)
            if not color_has_explicit_alpha(color_stripped):
                a = default_alpha
        elif re.match(r"^rgba?\s*\(", color_lower):
            r, g, b, a = _parse_rgb_string(color_stripped)
            if not color_has_explicit_alpha(color_stripped):
                a = default_alpha
        elif re.match(r"^hsla?\s*\(", color_lower):
            r, g, b, a = _parse_hsl_string(color_stripped)
            if not color_has_explicit_alpha(color_stripped):
                a = default_alpha
        elif color_lower in CSS_NAMED_COLORS:
            r, g, b = CSS_NAMED_COLORS[color_lower]
            a = default_alpha
        else:
            # Matplotlib adds the rest of the common CSS/X11 names and
            # aliases such as tab:blue without making it a hard dependency
            # for the core parser.
            try:
                from matplotlib.colors import to_rgba
                rgba = to_rgba(color_stripped)
                r = int(round(rgba[0] * 255))
                g = int(round(rgba[1] * 255))
                b = int(round(rgba[2] * 255))
                a = rgba[3] if color_has_explicit_alpha(color_stripped) else default_alpha
            except (ImportError, ValueError) as exc:
                raise ValueError(
                    f"Cannot parse color {color!r}. Use a named color, "
                    "#RGB/#RGBA/#RRGGBB/#RRGGBBAA, rgb()/rgba(), hsl()/hsla(), "
                    "or a 3/4-value RGB(A) tuple."
                ) from exc
    elif isinstance(color, (tuple, list)):
        r, g, b, a = _parse_tuple_color(color, default_alpha)
    elif _is_numeric(color):
        number = float(color)
        gray = number * 255.0 if not isinstance(color, Integral) and 0.0 <= number <= 1.0 else number
        r = g = b = int(round(_clamp(gray, 0.0, 255.0)))
        a = default_alpha
    else:
        raise ValueError(f"Unsupported color type: {type(color)}")

    if output_format == 'rgba':
        return f'rgba({r}, {g}, {b}, {a})'
    elif output_format == 'rgb':
        return f'rgb({r}, {g}, {b})'
    elif output_format == 'hex':
        return f'#{r:02x}{g:02x}{b:02x}'
    elif output_format == 'hex_alpha':
        alpha_int = int(round(a * 255))
        return f'#{r:02x}{g:02x}{b:02x}{alpha_int:02x}'
    elif output_format == 'tuple':
        return (r, g, b, a)
    elif output_format == 'tuple_normalized':
        return (r / 255.0, g / 255.0, b / 255.0, a)
    else:
        raise ValueError(f"Unknown output format: {output_format}")


def standardize_color_list(
    colors: Union[List, Tuple],
    default_alpha: float = 1.0,
    output_format: str = 'rgba'
) -> List[str]:
    """
    Standardize a list/tuple of colors to consistent RGBA format strings.
    
    This function handles color palettes from various sources (bokeh, matplotlib,
    custom lists) and converts them to a standard format.
    
    Parameters
    ----------
    colors : list or tuple
        List of colors in any supported format. Can also be a bokeh palette
        or a Bokeh palette. Mixed entries may use different supported formats.
    default_alpha : float, default 1.0
        Default alpha value to use if not specified in colors
    output_format : str, default 'rgba'
        Output format (see standardize_color for options)
        
    Returns
    -------
    list
        List of standardized color strings
        
    Examples
    --------
    >>> standardize_color_list(['red', '#00ff00', (0, 0, 255)])
    ['rgba(255, 0, 0, 1.0)', 'rgba(0, 255, 0, 1.0)', 'rgba(0, 0, 255, 1.0)']
    >>> 
    >>> import bokeh.palettes
    >>> standardize_color_list(bokeh.palettes.Category10[3])
    ['rgba(31, 119, 180, 1.0)', 'rgba(255, 127, 14, 1.0)', 'rgba(44, 160, 44, 1.0)']
    """
    if colors is None:
        return []

    colors = _as_color_sequence(colors)
    if isinstance(colors, str):
        colors = [colors]
    elif _is_color_sequence(colors):
        # A single RGB(A) tuple is one color, not three/four grayscale colors.
        colors = [colors]
    elif not isinstance(colors, (tuple, list)):
        raise ValueError(f"Color list must be a sequence, got {type(colors)}")

    if len(colors) == 0:
        return []

    result = []
    for c in colors:
        try:
            std_color = standardize_color(c, default_alpha=default_alpha, output_format=output_format)
            result.append(std_color)
        except ValueError as e:
            # Skip invalid colors with warning
            import warnings
            warnings.warn(f"Skipping invalid color: {c} - {e}")
            continue
    
    return result


def extract_rgb_tuple(color: Any) -> Tuple[int, int, int]:
    """
    Extract RGB tuple (0-255) from any color format.
    
    Parameters
    ----------
    color : any
        Color in any supported format
        
    Returns
    -------
    tuple
        (r, g, b) where values are 0-255
        
    Examples
    --------
    >>> extract_rgb_tuple('red')
    (255, 0, 0)
    >>> extract_rgb_tuple('#00ff00')
    (0, 255, 0)
    """
    rgba = standardize_color(color, output_format='tuple')
    return (rgba[0], rgba[1], rgba[2])


def extract_rgba_tuple(color: Any, default_alpha: float = 1.0) -> Tuple[int, int, int, float]:
    """
    Extract RGBA tuple from any color format.
    
    Parameters
    ----------
    color : any
        Color in any supported format
    default_alpha : float
        Default alpha if not specified
        
    Returns
    -------
    tuple
        (r, g, b, a) where r,g,b are 0-255 and a is 0.0-1.0
        
    Examples
    --------
    >>> extract_rgba_tuple('red')
    (255, 0, 0, 1.0)
    >>> extract_rgba_tuple('rgba(255, 0, 0, 0.5)')
    (255, 0, 0, 0.5)
    """
    return standardize_color(color, default_alpha=default_alpha, output_format='tuple')


def color_to_hex(color: Any) -> str:
    """
    Convert any color format to hex string.
    
    Parameters
    ----------
    color : any
        Color in any supported format
        
    Returns
    -------
    str
        Hex color string like '#ff0000'
        
    Examples
    --------
    >>> color_to_hex('red')
    '#ff0000'
    >>> color_to_hex((255, 128, 0))
    '#ff8000'
    """
    return standardize_color(color, output_format='hex')


def color_to_rgba_string(color: Any, alpha: float = None) -> str:
    """
    Convert any color format to rgba() CSS string.
    
    Parameters
    ----------
    color : any
        Color in any supported format
    alpha : float, optional
        Override alpha value. If None, uses the color's own alpha or 1.0
        
    Returns
    -------
    str
        RGBA color string like 'rgba(255, 0, 0, 1.0)'
        
    Examples
    --------
    >>> color_to_rgba_string('red')
    'rgba(255, 0, 0, 1.0)'
    >>> color_to_rgba_string('red', alpha=0.5)
    'rgba(255, 0, 0, 0.5)'
    """
    normalized_alpha = _normalize_alpha(alpha) if alpha is not None else None
    default_alpha = normalized_alpha if normalized_alpha is not None else 1.0
    rgba = standardize_color(color, default_alpha=default_alpha, output_format='tuple')
    
    # Override alpha if explicitly specified
    if normalized_alpha is not None:
        return f'rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {normalized_alpha})'
    return f'rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {rgba[3]})'


def is_dark_color(color: Any, threshold: float = 0.5) -> bool:
    """
    Determine if a color is dark (low luminance).
    
    Uses the standard luminance formula: 0.299*R + 0.587*G + 0.114*B
    
    Parameters
    ----------
    color : any
        Color in any supported format
    threshold : float, default 0.5
        Luminance threshold. Colors below this are considered dark.
        
    Returns
    -------
    bool
        True if the color is dark, False if light
        
    Examples
    --------
    >>> is_dark_color('black')
    True
    >>> is_dark_color('white')
    False
    >>> is_dark_color('#808080')  # Medium gray
    False
    >>> is_dark_color('navy')
    True
    """
    if isinstance(color, str) and color.lower() == 'auto':
        return False  # Default assumption for 'auto'
    
    try:
        r, g, b, _ = standardize_color(color, output_format='tuple')
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        return luminance < threshold
    except ValueError:
        return False  # Default to light if color cannot be parsed


def darken_color(color: Any, factor: float = 0.7) -> str:
    """
    Darken a color by a given factor.
    
    Parameters
    ----------
    color : any
        Color in any supported format
    factor : float, default 0.7
        Brightness factor from 0.0 (black) to 1.0 (original color)
        
    Returns
    -------
    str
        Darkened color in rgba format
        
    Examples
    --------
    >>> darken_color('red', 0.5)
    'rgba(127, 0, 0, 1.0)'
    """
    r, g, b, a = standardize_color(color, output_format='tuple')
    
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    
    return f'rgba({r}, {g}, {b}, {a})'


def lighten_color(color: Any, factor: float = 0.3) -> str:
    """
    Lighten a color by blending with white.
    
    Parameters
    ----------
    color : any
        Color in any supported format
    factor : float, default 0.3
        Amount to lighten (0.0 = original, 1.0 = white)
        
    Returns
    -------
    str
        Lightened color in rgba format
        
    Examples
    --------
    >>> lighten_color('red', 0.5)
    'rgba(255, 127, 127, 1.0)'
    """
    r, g, b, a = standardize_color(color, output_format='tuple')
    
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    
    return f'rgba({r}, {g}, {b}, {a})'


def set_alpha(color: Any, alpha: float) -> str:
    """
    Set the alpha (transparency) of a color.
    
    Parameters
    ----------
    color : any
        Color in any supported format
    alpha : float
        Alpha value from 0.0 (transparent) to 1.0 (opaque)
        
    Returns
    -------
    str
        Color with new alpha in rgba format
        
    Examples
    --------
    >>> set_alpha('red', 0.5)
    'rgba(255, 0, 0, 0.5)'
    """
    r, g, b, _ = standardize_color(color, output_format='tuple')
    alpha = _normalize_alpha(alpha)
    return f'rgba({r}, {g}, {b}, {alpha})'


def interpolate_colors(colors: List, n_colors: int) -> List[str]:
    """
    Generate n_colors by interpolating through a list of base colors.
    
    Parameters
    ----------
    colors : list
        Base color palette (any format)
    n_colors : int
        Number of colors to generate
        
    Returns
    -------
    list
        List of n_colors interpolated rgba color strings
        
    Examples
    --------
    >>> interpolate_colors(['red', 'blue'], 5)
    ['rgba(255, 0, 0, 1.0)', 'rgba(191, 0, 63, 1.0)', 'rgba(127, 0, 127, 1.0)', 
     'rgba(63, 0, 191, 1.0)', 'rgba(0, 0, 255, 1.0)']
    """
    if n_colors <= 0:
        return []
    
    if len(colors) == 0:
        return ['rgba(128, 128, 128, 1.0)'] * n_colors
    
    if n_colors == 1:
        return [standardize_color(colors[0])]
    
    # Convert all colors to RGBA tuples
    rgba_colors = [standardize_color(c, output_format='tuple') for c in colors]
    
    result = []
    for i in range(n_colors):
        # Calculate position in the gradient
        pos = i / (n_colors - 1) if n_colors > 1 else 0
        
        # Find which two colors to interpolate between
        scaled_pos = pos * (len(rgba_colors) - 1)
        idx1 = int(scaled_pos)
        idx2 = min(idx1 + 1, len(rgba_colors) - 1)
        t = scaled_pos - idx1  # Interpolation factor
        
        # Interpolate
        r1, g1, b1, a1 = rgba_colors[idx1]
        r2, g2, b2, a2 = rgba_colors[idx2]
        
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        a = a1 + (a2 - a1) * t
        
        result.append(f'rgba({r}, {g}, {b}, {a:.3f})')
    
    return result


def generate_color_palette(
    n_colors: int,
    palette: str = 'category10',
    alpha: float = 1.0
) -> List[str]:
    """
    Generate a color palette of n colors.
    
    Parameters
    ----------
    n_colors : int
        Number of colors to generate
    palette : str, default 'category10'
        Palette name. Options:
        - 'category10', 'category20': Categorical palettes (good for discrete data)
        - 'viridis', 'plasma', 'inferno', 'magma': Perceptually uniform sequential
        - 'rainbow', 'hsv': Full spectrum
        - 'cool', 'warm': Temperature-based
    alpha : float, default 1.0
        Alpha value for all colors
        
    Returns
    -------
    list
        List of rgba color strings
        
    Examples
    --------
    >>> generate_color_palette(5, 'category10')
    ['rgba(31, 119, 180, 1.0)', 'rgba(255, 127, 14, 1.0)', ...]
    """
    if n_colors <= 0:
        return []
    
    # Try bokeh palettes first
    try:
        import bokeh.palettes as bp
        
        palette_map = {
            # Bokeh categorical palettes have minimum sizes (Category10: 2,
            # Category20: 3); generate at the minimum and slice for small n.
            'category10': lambda n: bp.Category10[min(max(n, 3), 10)][:n],
            'category20': lambda n: bp.Category20[min(max(n, 3), 20)][:n],
            'category20b': lambda n: bp.Category20b[min(max(n, 3), 20)][:n],
            'category20c': lambda n: bp.Category20c[min(max(n, 3), 20)][:n],
            'viridis': lambda n: bp.Viridis256[::(256 // max(n, 1))][:n],
            'plasma': lambda n: bp.Plasma256[::(256 // max(n, 1))][:n],
            'inferno': lambda n: bp.Inferno256[::(256 // max(n, 1))][:n],
            'magma': lambda n: bp.Magma256[::(256 // max(n, 1))][:n],
        }
        
        if palette.lower() in palette_map:
            colors = palette_map[palette.lower()](n_colors)
            return standardize_color_list(colors, default_alpha=alpha)
    except ImportError:
        pass
    
    # Fallback to matplotlib
    try:
        from matplotlib import colormaps
        from matplotlib.colors import to_rgba
        
        cmap = colormaps.get_cmap(palette)
        colors = [cmap(i / max(n_colors - 1, 1)) for i in range(n_colors)]
        
        result = []
        for c in colors:
            r, g, b, a = c
            result.append(f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {alpha})')
        return result
    except (ImportError, ValueError):
        pass
    
    # Ultimate fallback: generate evenly spaced hues
    result = []
    for i in range(n_colors):
        hue = i / n_colors
        # HSV to RGB conversion
        h = hue * 6
        x = 1 - abs(h % 2 - 1)
        if h < 1:
            r, g, b = 1, x, 0
        elif h < 2:
            r, g, b = x, 1, 0
        elif h < 3:
            r, g, b = 0, 1, x
        elif h < 4:
            r, g, b = 0, x, 1
        elif h < 5:
            r, g, b = x, 0, 1
        else:
            r, g, b = 1, 0, x
        
        result.append(f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {alpha})')
    
    return result


# Convenience aliases
parse_color = standardize_color
normalize_color = standardize_color
to_rgba = standardize_color
