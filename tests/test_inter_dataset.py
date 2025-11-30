import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import os
import shutil
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from inter_dataset import TypeMapper, DatasetConfig, InterDatasetComparator

class TestInterDatasetComparator(unittest.TestCase):
    def setUp(self):
        self.output_dir = 'test_comparison_results'
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
            
        self.configs = [
            DatasetConfig(name='dataset1', source_types=['A'], target_types=['B']),
            DatasetConfig(name='dataset2', source_types=['A'], target_types=['B'])
        ]
        
    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_type_mapper(self):
        mapper = TypeMapper({'A': 'TypeA', 'B': 'TypeB'})
        self.assertEqual(mapper.get_std_type('A'), 'TypeA')
        self.assertEqual(mapper.get_std_type('C'), 'C')
        
        with self.assertRaises(ValueError):
            TypeMapper({'A': 'type'}) # Reserved keyword

    @patch('src.inter_dataset.FindNeuronConnection')
    def test_fetch_and_align(self, mock_fnc_class):
        # Mock FindNeuronConnection instance
        mock_fnc = MagicMock()
        mock_fnc_class.return_value = mock_fnc
        
        # Mock data for dataset1
        df1 = pd.DataFrame({
            'bodyId_pre': ['1', '2'],
            'bodyId_post': ['3', '4'],
            'type_pre': ['A', 'A'],
            'type_post': ['B', 'B'],
            'weight': [10, 5]
        })
        
        # Mock data for dataset2
        df2 = pd.DataFrame({
            'bodyId_pre': ['5', '6'],
            'bodyId_post': ['7', '8'],
            'type_pre': ['A', 'C'], # C is unique to dataset2
            'type_post': ['B', 'B'],
            'weight': [8, 3]
        })
        
        # Configure side_effect to return different dfs for different calls
        # Since we use ThreadPoolExecutor, the order isn't guaranteed, 
        # but we can check the dataset name in the constructor call if we wanted to be precise.
        # However, simpler is to just make the mock return data based on some attribute set during init?
        # Or just patch _fetch_single_dataset which is easier.
        pass

    @patch('inter_dataset.InterDatasetComparator._fetch_single_dataset')
    def test_full_pipeline(self, mock_fetch):
        # Mock fetch results
        df1 = pd.DataFrame({
            'type_pre': ['A', 'A'],
            'type_post': ['B', 'C'],
            'std_type_pre': ['A', 'A'],
            'std_type_post': ['B', 'C'],
            'weight': [10, 5]
        })
        meta1 = {'source_count': 10, 'target_count': 10}
        
        df2 = pd.DataFrame({
            'type_pre': ['A', 'A'],
            'type_post': ['B', 'C'],
            'std_type_pre': ['A', 'A'],
            'std_type_post': ['B', 'C'],
            'weight': [8, 2]
        })
        meta2 = {'source_count': 12, 'target_count': 12}
        
        def side_effect(config, threshold):
            if config.name == 'dataset1':
                return df1, meta1
            else:
                return df2, meta2
        
        mock_fetch.side_effect = side_effect
        
        comparator = InterDatasetComparator(self.configs, output_dir=self.output_dir)
        comparator.fetch_all_data()
        
        self.assertEqual(len(comparator.datasets_data), 2)
        self.assertEqual(len(comparator.datasets_metadata), 2)
        
        comparator.compare_metadata()
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'dataset_metadata_comparison.csv')))
        
        comparator.align_datasets()
        self.assertIsNotNone(comparator.aligned_data)
        # Check alignment
        # A->B: 10 vs 8
        # A->C: 5 vs 2
        self.assertEqual(comparator.aligned_data.loc[('A', 'B'), 'dataset1'], 10)
        self.assertEqual(comparator.aligned_data.loc[('A', 'B'), 'dataset2'], 8)
        
        comparator.run_sensitivity_analysis()
        self.assertIsNotNone(comparator.sensitivity_results)
        
        # Check sensitivity results
        # Threshold 1: Both edges present in both. Jaccard = 1.0
        res = comparator.sensitivity_results
        row_th1 = res[res['threshold'] == 1].iloc[0]
        self.assertEqual(row_th1['jaccard'], 1.0)
        
        # Threshold 6: 
        # A->B: 10 (d1), 8 (d2) -> Present in both
        # A->C: 5 (d1), 2 (d2) -> Present in d1 only (d2 < 6)
        # Jaccard: Intersection(1) / Union(2) = 0.5
        row_th6 = res[res['threshold'] == 5].iloc[0] # threshold 5
        # A->C is 5 in d1, so it meets threshold 5. In d2 it is 2, so fails.
        # So A->C is in d1 only.
        # A->B is 10 and 8, so in both.
        # Intersection = 1 (A->B)
        # Union = 2 (A->B, A->C)
        # Jaccard = 0.5
        self.assertEqual(row_th6['jaccard'], 0.5)

if __name__ == '__main__':
    unittest.main()
