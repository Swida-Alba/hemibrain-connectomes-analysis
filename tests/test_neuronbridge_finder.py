"""
Tests for NeuronBridge Finder Module
====================================

Tests the NeuronBridgeFinder class functionality including:
- Client initialization
- ID to lines conversion
- Neuron query to lines conversion  
- Line to neuron conversion
- Caching functionality
- CSV export

Note: These tests require network access to the NeuronBridge API.
Some tests may be slow due to API calls.
"""

import os
import sys
import pytest
import pandas as pd
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neuronbridge_finder import NeuronBridgeFinder, find_lines_for_body, find_neurons_for_line


class TestNeuronBridgeFinderInit:
    """Test NeuronBridgeFinder initialization."""
    
    def test_init_default(self):
        """Test default initialization."""
        nbf = NeuronBridgeFinder(verbose=False)
        assert nbf._client is not None
        assert nbf.use_cache == True
    
    def test_init_custom_cache_folder(self):
        """Test initialization with custom cache folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_folder = os.path.join(tmpdir, 'test_cache')
            nbf = NeuronBridgeFinder(
                cache_folder=cache_folder,
                verbose=False
            )
            assert os.path.exists(cache_folder)
    
    def test_init_no_cache(self):
        """Test initialization with caching disabled."""
        nbf = NeuronBridgeFinder(use_cache=False, verbose=False)
        assert nbf.use_cache == False


class TestIdToLines:
    """Test id_to_lines method."""
    
    @pytest.fixture
    def nbf(self):
        """Create NeuronBridgeFinder instance with temp cache."""
        tmpdir = tempfile.mkdtemp()
        nbf = NeuronBridgeFinder(
            cache_folder=tmpdir,
            verbose=False
        )
        yield nbf
        # Cleanup
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    def test_id_to_lines_returns_dataframe(self, nbf):
        """Test that id_to_lines returns a DataFrame."""
        # Use a known hemibrain body ID
        result = nbf.id_to_lines(636798093, top_n=5)
        assert isinstance(result, pd.DataFrame)
    
    def test_id_to_lines_columns(self, nbf):
        """Test that result has expected columns."""
        result = nbf.id_to_lines(636798093, top_n=5)
        expected_cols = ['line', 'library', 'score', 'image_id', 'match_type']
        for col in expected_cols:
            assert col in result.columns
    
    def test_id_to_lines_top_n(self, nbf):
        """Test that top_n limits results."""
        result = nbf.id_to_lines(636798093, top_n=3)
        assert len(result) <= 3
    
    def test_id_to_lines_invalid_id(self, nbf):
        """Test behavior with invalid body ID."""
        result = nbf.id_to_lines(99999999999, top_n=5)
        assert isinstance(result, pd.DataFrame)
        # Should return empty or have no matches


class TestLineToNeuron:
    """Test line_to_neuron method."""
    
    @pytest.fixture
    def nbf(self):
        """Create NeuronBridgeFinder instance with temp cache."""
        tmpdir = tempfile.mkdtemp()
        nbf = NeuronBridgeFinder(
            cache_folder=tmpdir,
            verbose=False
        )
        yield nbf
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    def test_line_to_neuron_returns_dataframe(self, nbf):
        """Test that line_to_neuron returns a DataFrame."""
        result = nbf.line_to_neuron('LH173', top_n=5)
        assert isinstance(result, pd.DataFrame)
    
    def test_line_to_neuron_columns(self, nbf):
        """Test that result has expected columns."""
        result = nbf.line_to_neuron('LH173', top_n=5)
        # At minimum should have bodyId and score
        if not result.empty:
            assert 'bodyId' in result.columns
            assert 'score' in result.columns
    
    def test_line_to_neuron_invalid_line(self, nbf):
        """Test behavior with invalid line name."""
        result = nbf.line_to_neuron('INVALID_LINE_NAME_12345', top_n=5)
        assert isinstance(result, pd.DataFrame)


class TestNeuronToLines:
    """Test neuron_to_lines method."""
    
    @pytest.fixture
    def nbf(self):
        """Create NeuronBridgeFinder instance."""
        tmpdir = tempfile.mkdtemp()
        nbf = NeuronBridgeFinder(
            cache_folder=tmpdir,
            verbose=False
        )
        yield nbf
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    def test_neuron_to_lines_by_id(self, nbf):
        """Test neuron_to_lines with integer body ID."""
        result = nbf.neuron_to_lines(636798093, top_n=3)
        assert isinstance(result, dict)
        assert '636798093' in result or 636798093 in result
    
    def test_neuron_to_lines_returns_dict(self, nbf):
        """Test that neuron_to_lines returns a dict."""
        result = nbf.neuron_to_lines(636798093, top_n=3)
        assert isinstance(result, dict)


class TestCaching:
    """Test caching functionality."""
    
    def test_cache_saves_results(self):
        """Test that results are cached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nbf = NeuronBridgeFinder(
                cache_folder=tmpdir,
                use_cache=True,
                verbose=False
            )
            
            # First call - should hit API
            result1 = nbf.id_to_lines(636798093, top_n=5)
            
            # Check cache file exists
            cache_files = os.listdir(tmpdir)
            assert any('id_to_lines' in f for f in cache_files)
    
    def test_cache_loads_results(self):
        """Test that cached results are loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nbf = NeuronBridgeFinder(
                cache_folder=tmpdir,
                use_cache=True,
                verbose=False
            )
            
            # First call
            result1 = nbf.id_to_lines(636798093, top_n=5)
            
            # Second call should use cache
            result2 = nbf.id_to_lines(636798093, top_n=5)
            
            # Results should be equivalent
            if not result1.empty and not result2.empty:
                assert len(result1) == len(result2)
    
    def test_clear_cache(self):
        """Test cache clearing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nbf = NeuronBridgeFinder(
                cache_folder=tmpdir,
                use_cache=True,
                verbose=False
            )
            
            # Create cache
            nbf.id_to_lines(636798093, top_n=5)
            
            # Clear cache
            nbf.clear_cache()
            
            # Check cache is empty
            cache_files = [f for f in os.listdir(tmpdir) if f.endswith('.csv')]
            assert len(cache_files) == 0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_find_lines_for_body(self):
        """Test find_lines_for_body function."""
        result = find_lines_for_body(636798093, top_n=3)
        assert isinstance(result, pd.DataFrame)
    
    def test_find_neurons_for_line(self):
        """Test find_neurons_for_line function."""
        result = find_neurons_for_line('LH173', top_n=3)
        assert isinstance(result, pd.DataFrame)


class TestMatchTypes:
    """Test different match types (CDS, PPPM, both)."""
    
    @pytest.fixture
    def nbf(self):
        """Create NeuronBridgeFinder instance."""
        tmpdir = tempfile.mkdtemp()
        nbf = NeuronBridgeFinder(
            cache_folder=tmpdir,
            verbose=False
        )
        yield nbf
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    def test_cds_match_type(self, nbf):
        """Test CDS match type."""
        result = nbf.id_to_lines(636798093, top_n=5, match_type='cds')
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert all(result['match_type'] == 'cds')
    
    def test_pppm_match_type(self, nbf):
        """Test PPPM match type."""
        result = nbf.id_to_lines(636798093, top_n=5, match_type='pppm')
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert all(result['match_type'] == 'pppm')
    
    def test_both_match_type(self, nbf):
        """Test 'both' match type."""
        result = nbf.id_to_lines(636798093, top_n=10, match_type='both')
        assert isinstance(result, pd.DataFrame)


class TestSaveResults:
    """Test save_results method."""
    
    def test_save_dataframe(self):
        """Test saving DataFrame results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nbf = NeuronBridgeFinder(
                cache_folder=tmpdir,
                verbose=False
            )
            
            result = nbf.id_to_lines(636798093, top_n=3)
            output_path = os.path.join(tmpdir, 'test_output.csv')
            
            saved_path = nbf.save_results(result, output_path, include_timestamp=False)
            
            assert os.path.exists(saved_path)
            loaded = pd.read_csv(saved_path)
            assert len(loaded) == len(result)
    
    def test_save_dict(self):
        """Test saving dict of DataFrame results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nbf = NeuronBridgeFinder(
                cache_folder=tmpdir,
                verbose=False
            )
            
            result = nbf.neuron_to_lines(636798093, top_n=3)
            output_path = os.path.join(tmpdir, 'test_output.csv')
            
            saved_path = nbf.save_results(result, output_path, include_timestamp=False)
            
            assert os.path.exists(saved_path)


if __name__ == '__main__':
    # Run a quick smoke test
    print("Running quick smoke test...")
    
    print("\n1. Testing initialization...")
    nbf = NeuronBridgeFinder(verbose=True)
    print("   ✓ Initialization successful")
    
    print("\n2. Testing id_to_lines...")
    lines = nbf.id_to_lines(636798093, top_n=5)
    print(f"   ✓ Found {len(lines)} lines")
    if not lines.empty:
        print(lines.head())
    
    print("\n3. Testing line_to_neuron...")
    neurons = nbf.line_to_neuron('LH173', top_n=5)
    print(f"   ✓ Found {len(neurons)} neurons")
    if not neurons.empty:
        print(neurons.head())
    
    print("\n✓ All smoke tests passed!")
