"""
Color Utilities Module
======================

This module provides comprehensive color parsing, standardization, and conversion
utilities for the visualization components in the connectome analysis toolkit.

Supported Input Formats
-----------------------
- **Named colors**: 'red', 'blue', 'lightgray', 'darkslategray', etc.
- **Hex colors**: '#ff0000', '#f00', '#FF0000FF' (with alpha)
- **RGB tuples**: (255, 0, 0) or (1.0, 0.0, 0.0) (normalized)
- **RGBA tuples**: (255, 0, 0, 0.5) or (1.0, 0.0, 0.0, 0.5)
- **CSS rgb/rgba strings**: 'rgb(255, 0, 0)', 'rgba(255, 0, 0, 0.5)'
- **Bokeh palettes**: bokeh.palettes.Category10[10], etc.
- **Matplotlib colormaps**: 'viridis', 'plasma', etc.

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

import re
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


def _parse_hex_color(hex_str: str) -> Tuple[int, int, int, float]:
    """
    Parse hex color string to RGBA tuple.
    
    Parameters
    ----------
    hex_str : str
        Hex color string like '#ff0000', '#f00', '#ff0000ff'
        
    Returns
    -------
    tuple
        (r, g, b, a) where r, g, b are 0-255 and a is 0.0-1.0
        
    Examples
    --------
    >>> _parse_hex_color('#ff0000')
    (255, 0, 0, 1.0)
    >>> _parse_hex_color('#f00')
    (255, 0, 0, 1.0)
    >>> _parse_hex_color('#ff000080')
    (255, 0, 0, 0.502)
    """
    hex_str = hex_str.lstrip('#')
    
    if len(hex_str) == 3:
        # Short form: #RGB -> #RRGGBB
        hex_str = ''.join(c * 2 for c in hex_str)
    elif len(hex_str) == 4:
        # Short form with alpha: #RGBA -> #RRGGBBAA
        hex_str = ''.join(c * 2 for c in hex_str)
    
    if len(hex_str) == 6:
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        a = 1.0
    elif len(hex_str) == 8:
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        a = int(hex_str[6:8], 16) / 255.0
    else:
        raise ValueError(f"Invalid hex color format: #{hex_str}")
    
    return (r, g, b, a)


def _parse_rgb_string(rgb_str: str) -> Tuple[int, int, int, float]:
    """
    Parse CSS rgb/rgba string to RGBA tuple.
    
    Parameters
    ----------
    rgb_str : str
        CSS color string like 'rgb(255, 0, 0)' or 'rgba(255, 0, 0, 0.5)'
        
    Returns
    -------
    tuple
        (r, g, b, a) where r, g, b are 0-255 and a is 0.0-1.0
        
    Examples
    --------
    >>> _parse_rgb_string('rgb(255, 0, 0)')
    (255, 0, 0, 1.0)
    >>> _parse_rgb_string('rgba(255, 0, 0, 0.5)')
    (255, 0, 0, 0.5)
    """
    # Extract numbers from the string
    numbers = re.findall(r'[\d.]+', rgb_str)
    
    if len(numbers) < 3:
        raise ValueError(f"Invalid rgb/rgba format: {rgb_str}")
    
    r = float(numbers[0])
    g = float(numbers[1])
    b = float(numbers[2])
    a = float(numbers[3]) if len(numbers) > 3 else 1.0
    
    # Handle normalized values (0-1 range)
    if r <= 1 and g <= 1 and b <= 1 and max(r, g, b) <= 1:
        # Could be normalized, but only if all values are <= 1
        # Check if any non-zero value is greater than 1 to determine
        if max(r, g, b) > 0 or (r == 0 and g == 0 and b == 0):
            # Looks like 0-1 range only if max is <= 1 and not all zeros with large value
            pass  # Keep as is, will normalize below
    
    # Convert to 0-255 range if in normalized form
    if r <= 1 and g <= 1 and b <= 1:
        # Heuristic: if all values are small decimals, assume normalized
        # This check looks for values like (0.5, 0.3, 0.2) vs (128, 64, 32)
        if all(v <= 1 for v in [r, g, b]) and any(0 < v < 1 for v in [r, g, b]):
            r = int(r * 255)
            g = int(g * 255)
            b = int(b * 255)
        else:
            r = int(r)
            g = int(g)
            b = int(b)
    else:
        r = int(r)
        g = int(g)
        b = int(b)
    
    # Clamp values
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    a = max(0.0, min(1.0, a))
    
    return (r, g, b, a)


def _parse_tuple_color(color_tuple: Tuple, default_alpha: float = 1.0) -> Tuple[int, int, int, float]:
    """
    Parse RGB or RGBA tuple to standardized RGBA tuple.
    
    Parameters
    ----------
    color_tuple : tuple
        RGB tuple (r, g, b) or RGBA tuple (r, g, b, a)
        Values can be 0-255 integers or 0.0-1.0 floats
    default_alpha : float
        Default alpha value if not provided in tuple
        
    Returns
    -------
    tuple
        (r, g, b, a) where r, g, b are 0-255 and a is 0.0-1.0
        
    Examples
    --------
    >>> _parse_tuple_color((255, 0, 0))
    (255, 0, 0, 1.0)
    >>> _parse_tuple_color((1.0, 0.0, 0.0))
    (255, 0, 0, 1.0)
    >>> _parse_tuple_color((255, 0, 0, 0.5))
    (255, 0, 0, 0.5)
    """
    if len(color_tuple) < 3:
        raise ValueError(f"Color tuple must have at least 3 values, got {len(color_tuple)}")
    
    r, g, b = color_tuple[0], color_tuple[1], color_tuple[2]
    a = color_tuple[3] if len(color_tuple) > 3 else default_alpha
    
    # Determine if values are normalized (0-1) or absolute (0-255)
    # Heuristic: if all RGB values are <= 1.0 and at least one is a float with decimal
    is_normalized = False
    if all(isinstance(v, float) for v in [r, g, b]):
        if all(v <= 1.0 for v in [r, g, b]):
            is_normalized = True
    elif all(v <= 1.0 for v in [r, g, b]) and any(0 < v < 1 for v in [r, g, b]):
        is_normalized = True
    
    if is_normalized:
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)
    else:
        r = int(r)
        g = int(g)
        b = int(b)
    
    # Handle alpha - only treat as 0-255 range if it's clearly an integer > 1
    # Values like 1.5 should be clamped, not divided by 255
    if isinstance(a, (int, float)):
        if isinstance(a, int) and a > 1:
            # Integer > 1 is likely 0-255 range (e.g., 128 for 50% opacity)
            a = a / 255.0
        # Otherwise keep as-is (will be clamped below)
        a = float(a)
    else:
        a = default_alpha
    
    # Clamp values
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    a = max(0.0, min(1.0, a))
    
    return (r, g, b, a)


def standardize_color(
    color: Any,
    default_alpha: float = 1.0,
    output_format: str = 'rgba'
) -> str:
    """
    Standardize a color input to a consistent RGBA format string.
    
    This function accepts multiple color input formats and converts them
    to a standard RGBA string format suitable for Plotly/CSS usage.
    
    Parameters
    ----------
    color : str, tuple, list
        Color in any supported format:
        - Named colors: 'red', 'blue', 'lightgray', etc.
        - Hex colors: '#ff0000', '#f00', '#FF0000FF'
        - RGB tuples: (255, 0, 0) or (1.0, 0.0, 0.0)
        - RGBA tuples: (255, 0, 0, 0.5) or (1.0, 0.0, 0.0, 0.5)
        - CSS strings: 'rgb(255, 0, 0)', 'rgba(255, 0, 0, 0.5)'
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
    r, g, b, a = 0, 0, 0, default_alpha
    
    if color is None:
        raise ValueError("Color cannot be None")
    
    # Handle string inputs
    if isinstance(color, str):
        color_stripped = color.strip()
        color_lower = color_stripped.lower()
        
        # Check if it's already in rgba format
        if color_lower.startswith('rgba('):
            r, g, b, a = _parse_rgb_string(color_stripped)
        # Check if it's in rgb format
        elif color_lower.startswith('rgb('):
            r, g, b, a = _parse_rgb_string(color_stripped)
            a = default_alpha if a == 1.0 else a
        # Check if it's a hex color
        elif color_stripped.startswith('#') or (len(color_stripped) in [3, 6, 8] and all(c in '0123456789abcdefABCDEF' for c in color_stripped)):
            if not color_stripped.startswith('#'):
                color_stripped = '#' + color_stripped
            r, g, b, a = _parse_hex_color(color_stripped)
            # If hex didn't have alpha, use default
            if len(color_stripped.lstrip('#')) <= 6:
                a = default_alpha
        # Check if it's a named color
        elif color_lower in CSS_NAMED_COLORS:
            r, g, b = CSS_NAMED_COLORS[color_lower]
            a = default_alpha
        # Check if it's 'auto' or other special value
        elif color_lower == 'auto':
            return 'auto'  # Return as-is for special handling
        else:
            # Try matplotlib color converter as fallback
            try:
                from matplotlib.colors import to_rgba
                rgba = to_rgba(color_stripped)
                r = int(rgba[0] * 255)
                g = int(rgba[1] * 255)
                b = int(rgba[2] * 255)
                a = rgba[3] if rgba[3] != 1.0 else default_alpha
            except (ImportError, ValueError):
                raise ValueError(f"Cannot parse color: {color}")
    
    # Handle tuple/list inputs
    elif isinstance(color, (tuple, list)):
        r, g, b, a = _parse_tuple_color(color, default_alpha)
    
    # Handle numeric (single value = grayscale)
    elif isinstance(color, (int, float)):
        if isinstance(color, float) and color <= 1.0:
            gray = int(color * 255)
        else:
            gray = int(color)
        gray = max(0, min(255, gray))
        r = g = b = gray
        a = default_alpha
    
    else:
        raise ValueError(f"Unsupported color type: {type(color)}")
    
    # Format output
    if output_format == 'rgba':
        return f'rgba({r}, {g}, {b}, {a})'
    elif output_format == 'rgb':
        return f'rgb({r}, {g}, {b})'
    elif output_format == 'hex':
        return f'#{r:02x}{g:02x}{b:02x}'
    elif output_format == 'hex_alpha':
        alpha_int = int(a * 255)
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
        or matplotlib colormap name.
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
    if colors is None or len(colors) == 0:
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
    default_alpha = alpha if alpha is not None else 1.0
    rgba = standardize_color(color, default_alpha=default_alpha, output_format='tuple')
    
    # Override alpha if explicitly specified
    if alpha is not None:
        return f'rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {alpha})'
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
    alpha = max(0.0, min(1.0, alpha))
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
