"""
Utility modules for DROCAT.

Modules:
    api_utils: API call utilities with timeout, retry, and Cypher escaping
    report_utils: Report generation utilities (PPTX, PDF)
    color_utils: Color parsing, standardization, and conversion utilities
"""

from .api_utils import (
    api_call_with_retry,
    escape_cypher_string,
    build_cypher_type_condition,
    process_batches_with_retry,
    APITimeoutError,
    APIRetryExhaustedError,
)

from .report_utils import img2pptx

from .color_utils import (
    standardize_color,
    standardize_color_list,
    color_has_explicit_alpha,
    extract_rgb_tuple,
    extract_rgba_tuple,
    color_to_hex,
    color_to_rgba_string,
    is_dark_color,
    darken_color,
    lighten_color,
    set_alpha,
    interpolate_colors,
    generate_color_palette,
)

__all__ = [
    # API utilities
    'api_call_with_retry',
    'escape_cypher_string',
    'build_cypher_type_condition',
    'process_batches_with_retry',
    'APITimeoutError',
    'APIRetryExhaustedError',
    # Report utilities
    'img2pptx',
    # Color utilities
    'standardize_color',
    'standardize_color_list',
    'color_has_explicit_alpha',
    'extract_rgb_tuple',
    'extract_rgba_tuple',
    'color_to_hex',
    'color_to_rgba_string',
    'is_dark_color',
    'darken_color',
    'lighten_color',
    'set_alpha',
    'interpolate_colors',
    'generate_color_palette',
]
