"""
Comprehensive tests for color_utils module.

Run with: python -m pytest tests/test_color_utils.py -v
Or standalone: python tests/test_color_utils.py
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import unittest
from utils.color_utils import (
    standardize_color,
    standardize_color_list,
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


class TestStandardizeColor(unittest.TestCase):
    """Test the main standardize_color function."""
    
    def test_named_colors(self):
        """Test named CSS color parsing."""
        # Common named colors
        self.assertEqual(standardize_color('red'), 'rgba(255, 0, 0, 1.0)')
        self.assertEqual(standardize_color('green'), 'rgba(0, 128, 0, 1.0)')
        self.assertEqual(standardize_color('blue'), 'rgba(0, 0, 255, 1.0)')
        self.assertEqual(standardize_color('white'), 'rgba(255, 255, 255, 1.0)')
        self.assertEqual(standardize_color('black'), 'rgba(0, 0, 0, 1.0)')
        
        # Case insensitivity
        self.assertEqual(standardize_color('RED'), 'rgba(255, 0, 0, 1.0)')
        self.assertEqual(standardize_color('Red'), 'rgba(255, 0, 0, 1.0)')
        
        # Complex named colors
        self.assertEqual(standardize_color('lightblue'), 'rgba(173, 216, 230, 1.0)')
        self.assertEqual(standardize_color('darkslategray'), 'rgba(47, 79, 79, 1.0)')
    
    def test_hex_colors(self):
        """Test hex color parsing."""
        # 6-digit hex
        self.assertEqual(standardize_color('#ff0000'), 'rgba(255, 0, 0, 1.0)')
        self.assertEqual(standardize_color('#00ff00'), 'rgba(0, 255, 0, 1.0)')
        self.assertEqual(standardize_color('#0000ff'), 'rgba(0, 0, 255, 1.0)')
        
        # Without hash
        self.assertEqual(standardize_color('ff0000'), 'rgba(255, 0, 0, 1.0)')
        
        # 3-digit hex (shorthand)
        self.assertEqual(standardize_color('#f00'), 'rgba(255, 0, 0, 1.0)')
        self.assertEqual(standardize_color('#0f0'), 'rgba(0, 255, 0, 1.0)')
        
        # 8-digit hex with alpha
        self.assertEqual(standardize_color('#ff000080'), 'rgba(255, 0, 0, 0.5019607843137255)')
        
        # Case insensitivity
        self.assertEqual(standardize_color('#FF0000'), 'rgba(255, 0, 0, 1.0)')
    
    def test_rgb_tuples(self):
        """Test RGB tuple parsing."""
        # Integer RGB tuples (0-255)
        self.assertEqual(standardize_color((255, 0, 0)), 'rgba(255, 0, 0, 1.0)')
        self.assertEqual(standardize_color((128, 128, 128)), 'rgba(128, 128, 128, 1.0)')
        
        # Float RGB tuples (0.0-1.0)
        self.assertEqual(standardize_color((1.0, 0.0, 0.0)), 'rgba(255, 0, 0, 1.0)')
        self.assertEqual(standardize_color((0.5, 0.5, 0.5)), 'rgba(127, 127, 127, 1.0)')
        
        # Lists work too
        self.assertEqual(standardize_color([255, 0, 0]), 'rgba(255, 0, 0, 1.0)')
    
    def test_rgba_tuples(self):
        """Test RGBA tuple parsing."""
        # Integer RGBA tuples
        self.assertEqual(standardize_color((255, 0, 0, 0.5)), 'rgba(255, 0, 0, 0.5)')
        
        # Float RGBA tuples
        self.assertEqual(standardize_color((1.0, 0.0, 0.0, 0.5)), 'rgba(255, 0, 0, 0.5)')
    
    def test_css_rgb_strings(self):
        """Test CSS rgb() string parsing."""
        self.assertEqual(standardize_color('rgb(255, 0, 0)'), 'rgba(255, 0, 0, 1.0)')
        self.assertEqual(standardize_color('RGB(255, 0, 0)'), 'rgba(255, 0, 0, 1.0)')
        
    def test_css_rgba_strings(self):
        """Test CSS rgba() string parsing."""
        self.assertEqual(standardize_color('rgba(255, 0, 0, 0.5)'), 'rgba(255, 0, 0, 0.5)')
        self.assertEqual(standardize_color('RGBA(255, 0, 0, 0.5)'), 'rgba(255, 0, 0, 0.5)')
        
        # Already correct format should pass through
        result = standardize_color('rgba(255, 128, 64, 0.75)')
        self.assertEqual(result, 'rgba(255, 128, 64, 0.75)')
    
    def test_default_alpha(self):
        """Test default_alpha parameter."""
        self.assertEqual(standardize_color('red', default_alpha=0.5), 'rgba(255, 0, 0, 0.5)')
        self.assertEqual(standardize_color('#00ff00', default_alpha=0.3), 'rgba(0, 255, 0, 0.3)')
        self.assertEqual(standardize_color((0, 0, 255), default_alpha=0.8), 'rgba(0, 0, 255, 0.8)')
    
    def test_output_formats(self):
        """Test different output formats."""
        # Default rgba
        self.assertEqual(standardize_color('red', output_format='rgba'), 'rgba(255, 0, 0, 1.0)')
        
        # RGB string
        self.assertEqual(standardize_color('red', output_format='rgb'), 'rgb(255, 0, 0)')
        
        # Hex
        self.assertEqual(standardize_color('red', output_format='hex'), '#ff0000')
        
        # Hex with alpha
        result = standardize_color('red', default_alpha=0.5, output_format='hex_alpha')
        self.assertTrue(result.startswith('#ff0000'))
        
        # Tuple
        self.assertEqual(standardize_color('red', output_format='tuple'), (255, 0, 0, 1.0))
        
        # Normalized tuple
        result = standardize_color('red', output_format='tuple_normalized')
        self.assertEqual(result, (1.0, 0.0, 0.0, 1.0))
    
    def test_auto_special_value(self):
        """Test 'auto' special value passthrough."""
        self.assertEqual(standardize_color('auto'), 'auto')
    
    def test_invalid_colors(self):
        """Test error handling for invalid colors."""
        with self.assertRaises(ValueError):
            standardize_color(None)
        
        with self.assertRaises(ValueError):
            standardize_color('notacolor')
        
        with self.assertRaises(ValueError):
            standardize_color((1, 2))  # Too few values


class TestStandardizeColorList(unittest.TestCase):
    """Test the standardize_color_list function."""
    
    def test_simple_list(self):
        """Test simple color list."""
        colors = ['red', 'green', 'blue']
        result = standardize_color_list(colors)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], 'rgba(255, 0, 0, 1.0)')
    
    def test_mixed_formats(self):
        """Test list with mixed color formats."""
        colors = ['red', '#00ff00', (0, 0, 255), 'rgba(128, 128, 128, 0.5)']
        result = standardize_color_list(colors)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], 'rgba(255, 0, 0, 1.0)')
        self.assertEqual(result[1], 'rgba(0, 255, 0, 1.0)')
        self.assertEqual(result[2], 'rgba(0, 0, 255, 1.0)')
        self.assertEqual(result[3], 'rgba(128, 128, 128, 0.5)')
    
    def test_with_default_alpha(self):
        """Test list with default alpha."""
        colors = ['red', 'green']
        result = standardize_color_list(colors, default_alpha=0.5)
        self.assertEqual(result[0], 'rgba(255, 0, 0, 0.5)')
        self.assertEqual(result[1], 'rgba(0, 128, 0, 0.5)')
    
    def test_bokeh_palette(self):
        """Test with bokeh palette."""
        try:
            import bokeh.palettes as bp
            colors = bp.Category10[3]
            result = standardize_color_list(colors)
            self.assertEqual(len(result), 3)
            # Category10 first color is #1f77b4 (blue)
            self.assertEqual(result[0], 'rgba(31, 119, 180, 1.0)')
        except ImportError:
            self.skipTest("bokeh not installed")
    
    def test_empty_list(self):
        """Test empty list."""
        result = standardize_color_list([])
        self.assertEqual(result, [])


class TestExtractFunctions(unittest.TestCase):
    """Test extract_rgb_tuple and extract_rgba_tuple functions."""
    
    def test_extract_rgb(self):
        """Test RGB tuple extraction."""
        self.assertEqual(extract_rgb_tuple('red'), (255, 0, 0))
        self.assertEqual(extract_rgb_tuple('#00ff00'), (0, 255, 0))
        self.assertEqual(extract_rgb_tuple((0, 0, 255)), (0, 0, 255))
    
    def test_extract_rgba(self):
        """Test RGBA tuple extraction."""
        self.assertEqual(extract_rgba_tuple('red'), (255, 0, 0, 1.0))
        self.assertEqual(extract_rgba_tuple('rgba(255, 0, 0, 0.5)'), (255, 0, 0, 0.5))
        self.assertEqual(extract_rgba_tuple((128, 128, 128, 0.5)), (128, 128, 128, 0.5))


class TestColorConversions(unittest.TestCase):
    """Test color conversion functions."""
    
    def test_color_to_hex(self):
        """Test hex conversion."""
        self.assertEqual(color_to_hex('red'), '#ff0000')
        self.assertEqual(color_to_hex((255, 128, 0)), '#ff8000')
        self.assertEqual(color_to_hex('rgba(0, 255, 0, 0.5)'), '#00ff00')
    
    def test_color_to_rgba_string(self):
        """Test rgba string conversion."""
        self.assertEqual(color_to_rgba_string('red'), 'rgba(255, 0, 0, 1.0)')
        self.assertEqual(color_to_rgba_string('red', alpha=0.5), 'rgba(255, 0, 0, 0.5)')


class TestIsDarkColor(unittest.TestCase):
    """Test is_dark_color function."""
    
    def test_dark_colors(self):
        """Test dark color detection."""
        self.assertTrue(is_dark_color('black'))
        self.assertTrue(is_dark_color('#000000'))
        self.assertTrue(is_dark_color('navy'))
        self.assertTrue(is_dark_color('darkgreen'))
        self.assertTrue(is_dark_color('maroon'))
    
    def test_light_colors(self):
        """Test light color detection."""
        self.assertFalse(is_dark_color('white'))
        self.assertFalse(is_dark_color('#ffffff'))
        self.assertFalse(is_dark_color('yellow'))
        self.assertFalse(is_dark_color('lightblue'))
    
    def test_auto_value(self):
        """Test 'auto' special value."""
        self.assertFalse(is_dark_color('auto'))
    
    def test_custom_threshold(self):
        """Test custom threshold."""
        # Gray is around 0.5 luminance
        self.assertFalse(is_dark_color('gray', threshold=0.3))
        self.assertTrue(is_dark_color('gray', threshold=0.7))


class TestColorModification(unittest.TestCase):
    """Test color modification functions."""
    
    def test_darken_color(self):
        """Test color darkening."""
        result = darken_color('red', 0.5)
        # Should be approximately half brightness
        self.assertIn('127', result)  # R value should be ~127
    
    def test_lighten_color(self):
        """Test color lightening."""
        result = lighten_color('red', 0.5)
        # Should be blended toward white
        self.assertIn('255', result)  # R stays 255
        self.assertIn('127', result)  # G and B should be ~127
    
    def test_set_alpha(self):
        """Test alpha setting."""
        result = set_alpha('red', 0.5)
        self.assertEqual(result, 'rgba(255, 0, 0, 0.5)')
        
        result = set_alpha('#00ff00', 0.3)
        self.assertEqual(result, 'rgba(0, 255, 0, 0.3)')


class TestInterpolateColors(unittest.TestCase):
    """Test color interpolation function."""
    
    def test_basic_interpolation(self):
        """Test basic color interpolation."""
        colors = ['red', 'blue']
        result = interpolate_colors(colors, 3)
        self.assertEqual(len(result), 3)
        # First should be red
        self.assertIn('255, 0, 0', result[0])
        # Last should be blue
        self.assertIn('0, 0, 255', result[2])
        # Middle should be purple-ish
        self.assertIn('127', result[1])  # Interpolated value
    
    def test_single_color(self):
        """Test interpolation with single color."""
        result = interpolate_colors(['red'], 1)
        self.assertEqual(len(result), 1)
        self.assertIn('255, 0, 0', result[0])
    
    def test_many_colors(self):
        """Test interpolation to many colors."""
        colors = ['red', 'green', 'blue']
        result = interpolate_colors(colors, 10)
        self.assertEqual(len(result), 10)


class TestGenerateColorPalette(unittest.TestCase):
    """Test palette generation function."""
    
    def test_category10(self):
        """Test category10 palette generation."""
        try:
            result = generate_color_palette(5, 'category10')
            self.assertEqual(len(result), 5)
            # All should be valid rgba strings
            for c in result:
                self.assertTrue(c.startswith('rgba('))
        except ImportError:
            self.skipTest("Required palette library not installed")
    
    def test_with_alpha(self):
        """Test palette with custom alpha."""
        try:
            result = generate_color_palette(3, 'category10', alpha=0.5)
            for c in result:
                self.assertIn('0.5', c)
        except ImportError:
            self.skipTest("Required palette library not installed")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""
    
    def test_whitespace_handling(self):
        """Test whitespace in color strings."""
        self.assertEqual(standardize_color(' red '), 'rgba(255, 0, 0, 1.0)')
        self.assertEqual(standardize_color(' #ff0000 '), 'rgba(255, 0, 0, 1.0)')
    
    def test_grayscale_numeric(self):
        """Test numeric grayscale input."""
        # Integer grayscale
        result = standardize_color(128)
        self.assertEqual(result, 'rgba(128, 128, 128, 1.0)')
        
        # Float grayscale (0-1)
        result = standardize_color(0.5)
        self.assertEqual(result, 'rgba(127, 127, 127, 1.0)')
    
    def test_boundary_values(self):
        """Test boundary RGB values."""
        self.assertEqual(standardize_color((0, 0, 0)), 'rgba(0, 0, 0, 1.0)')
        self.assertEqual(standardize_color((255, 255, 255)), 'rgba(255, 255, 255, 1.0)')
    
    def test_alpha_clamping(self):
        """Test alpha value clamping."""
        result = standardize_color((255, 0, 0, 1.5))  # Alpha > 1
        self.assertIn('1.0', result)  # Should be clamped
        
        result = standardize_color((255, 0, 0, -0.5))  # Alpha < 0
        self.assertIn('0.0', result)  # Should be clamped


def run_tests():
    """Run all tests and print summary."""
    print("=" * 70)
    print("Color Utilities Test Suite")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestStandardizeColor))
    suite.addTests(loader.loadTestsFromTestCase(TestStandardizeColorList))
    suite.addTests(loader.loadTestsFromTestCase(TestExtractFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestColorConversions))
    suite.addTests(loader.loadTestsFromTestCase(TestIsDarkColor))
    suite.addTests(loader.loadTestsFromTestCase(TestColorModification))
    suite.addTests(loader.loadTestsFromTestCase(TestInterpolateColors))
    suite.addTests(loader.loadTestsFromTestCase(TestGenerateColorPalette))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split(chr(10))[0]}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split(chr(10))[0]}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
