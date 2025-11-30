
import sys
import os
import io
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from coana import FindNeuronConnection, VisualizeSkeleton

class TestMissingDatasetInstructions(unittest.TestCase):
    
    def setUp(self):
        # Use a fake FAFB dataset name that won't have local files
        self.dataset = "flywire_FAFB_TEST_MISSING"
        self.download_url = "https://codex.flywire.ai/api/download?dataset=fafb"
        
        # Mock FlyWire client to avoid actual connection attempts if logic fails
        self.mock_client = MagicMock()
        self.mock_client.fetch_neurons.return_value = (pd.DataFrame(), None)
        self.mock_client.fetch_adjacencies.return_value = (pd.DataFrame(), pd.DataFrame())
        self.mock_client.fetch_synapse_connections.return_value = pd.DataFrame()

    def test_FindPath_missing_dataset(self):
        print("\nTesting FindPath with missing dataset...")
        
        original_exists = os.path.exists
        
        with patch('coana.os.path.exists') as mock_exists:
            def side_effect(path):
                if "datasets" in str(path) and "flywire" in str(path):
                    return False
                return original_exists(path)
            mock_exists.side_effect = side_effect
            
            # Setup FindNeuronConnection
            fnc = FindNeuronConnection()
            fnc.dataset = self.dataset
            fnc.client_type = 'flywire'
            fnc.client_flywire = self.mock_client
            fnc.save_folder = "tests/output"
            fnc.min_synapse_num = 1
            fnc.min_ratio = 0.01
            fnc.min_traversal_probability = 0.01
            fnc.max_interlayer = 1
            
            # Mock _query_connection_db to ensure we try to fetch
            # Returns: cached_conn, uncached_upstream, partially_cached
            fnc._query_connection_db = MagicMock(return_value=(pd.DataFrame(), ['123'], []))
            
            # Mock source and target DFs to start the process
            fnc.source_df = pd.DataFrame({'bodyId': ['123'], 'type': ['typeA'], 'post': [100]})
            fnc.target_df = pd.DataFrame({'bodyId': ['456'], 'type': ['typeB'], 'post': [100]})
            
            # Capture stdout
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            try:
                # Run FindPath
                # It will try to fetch connections. Since local file is missing, it should print instructions and return empty
                fnc.FindPath()
            except Exception as e:
                print(f"Caught expected exception or finished: {e}")
            finally:
                sys.stdout = sys.__stdout__
                
            output = captured_output.getvalue()
            print("Output captured:")
            print(output)
            
            # Check for download instructions
            self.assertIn(self.download_url, output)
            self.assertIn("Local connection data not found", output)

    def test_FindAllPath_missing_dataset(self):
        print("\nTesting FindAllPath with missing dataset...")
        
        # Save original exists
        original_exists = os.path.exists
        
        with patch('coana.os.path.exists') as mock_exists:
            def side_effect(path):
                # Force False for dataset path to simulate missing file
                if "datasets" in str(path) and "flywire" in str(path):
                    return False
                # Use real exists for everything else (like creating output dirs)
                return original_exists(path)
            mock_exists.side_effect = side_effect
            
            fnc = FindNeuronConnection()
            fnc.dataset = self.dataset
            fnc.client_type = 'flywire'
            fnc.client_flywire = self.mock_client
            fnc.save_folder = "tests/output"
            fnc.min_synapse_num = 1
            fnc.min_ratio = 0.01
            fnc.min_traversal_probability = 0.01
            fnc.max_interlayer = 1
            fnc.source_fname = "test_source"
            fnc.target_fname = "test_target"
            fnc.parameter_dict = {}
            
            fnc._query_connection_db = MagicMock(return_value=(pd.DataFrame(), ['123'], []))
            
            fnc.source_df = pd.DataFrame({'bodyId': ['123'], 'type': ['typeA'], 'post': [100]})
            fnc.target_df = pd.DataFrame({'bodyId': ['456'], 'type': ['typeB'], 'post': [100]})
            
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            try:
                fnc.FindAllPath()
            except Exception as e:
                print(f"Caught exception in FindAllPath: {e}")
                # import traceback
                # traceback.print_exc()
            finally:
                sys.stdout = sys.__stdout__
                
            output = captured_output.getvalue()
            print("Output captured:")
            print(output)
            
            self.assertIn(self.download_url, output)
            self.assertIn("Local connection data not found", output)

    def test_VisualizeSkeleton_plot_synapses_missing_dataset(self):
        print("\nTesting VisualizeSkeleton.plot_synapses with missing dataset...")
        
        vs = VisualizeSkeleton()
        vs.dataset = self.dataset
        vs.client_type = 'flywire'
        vs.client_flywire = self.mock_client
        vs.save_folder = "tests/output"
        vs.saveas = "test_vis"
        vs.neuron_layers = [pd.DataFrame({'bodyId': ['123']}), pd.DataFrame({'bodyId': ['456']})]
        vs.layer_criteria = ["criteria1", "criteria2"]
        vs.script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        vs.ignore_synapses = False
        vs.synapse_mode = 'scatter'
        vs.backend = 'plotly'
        
        # Mock _load_cached_synapses to return None (force fetch)
        vs._load_cached_synapses = MagicMock(return_value=None)
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            vs.plot_synapses()
        except Exception as e:
            pass
        finally:
            sys.stdout = sys.__stdout__
            
        output = captured_output.getvalue()
        print("Output captured:")
        print(output)
        
        self.assertIn(self.download_url, output)
        self.assertIn("Local synapse file not found", output)

    def test_fetch_neurons_missing_dataset(self):
        print("\nTesting _fetch_neurons_local_or_api with missing dataset...")
        
        original_exists = os.path.exists
        
        with patch('coana.os.path.exists') as mock_exists:
            def side_effect(path):
                if "datasets" in str(path) and "flywire" in str(path):
                    return False
                return original_exists(path)
            mock_exists.side_effect = side_effect
            
            fnc = FindNeuronConnection()
            fnc.dataset = self.dataset
            fnc.client_type = 'flywire'
            fnc.client_flywire = self.mock_client
            fnc.script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
            
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            try:
                fnc._fetch_neurons_local_or_api(['123'])
            except Exception as e:
                print(f"Caught exception: {e}")
            finally:
                sys.stdout = sys.__stdout__
                
            output = captured_output.getvalue()
            print("Output captured:")
            print(output)
            
            self.assertIn(self.download_url, output)
            self.assertIn("Local neuron data not found", output)

if __name__ == '__main__':
    unittest.main()
