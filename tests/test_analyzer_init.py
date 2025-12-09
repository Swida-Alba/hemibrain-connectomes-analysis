import unittest
from src.comparison.comparison_analyzer import ComparisonAnalyzer, ComparisonParameters
from src.comparison.label_mapper import LabelMapper

class TestAnalyzerInit(unittest.TestCase):
    def test_label_mapper_extraction(self):
        mapper = LabelMapper()
        params = ComparisonParameters(
            datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
            source_neurons=['neuron1'],
            target_neurons=mapper,
            output_folder='./test_output'
        )
        
        # Initialize analyzer without explicit label_mapper
        analyzer = ComparisonAnalyzer(params)
        
        # Check if label_mapper was extracted
        self.assertIsNotNone(analyzer.label_mapper)
        self.assertEqual(analyzer.label_mapper, mapper)

if __name__ == '__main__':
    unittest.main()
