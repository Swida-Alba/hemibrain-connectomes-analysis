"""
Example: Using LabelMapper for Cross-Dataset Comparison

This example demonstrates how to use the LabelMapper class to handle complex
cross-dataset neuron mapping scenarios where neuron names or IDs differ
significantly between datasets.

The LabelMapper allows you to:
1. Define explicit mappings between a "standard label" and dataset-specific IDs/names.
2. Handle cases where one neuron in Dataset A maps to multiple neurons in Dataset B.
3. Use a dictionary or CSV/JSON files as input for the mapping.

This is particularly useful when:
- Datasets use different naming conventions (e.g., 'aMe12' vs 'aMe12-like').
- You are comparing specific bodyIds that have no common type name.
- You want to group multiple neurons under a single logical label for comparison.
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from comparison.comparison_parameters import ComparisonParameters
from comparison.comparison_analyzer import ComparisonAnalyzer
from comparison.label_mapper import LabelMapper

def run_label_mapper_example():
    print("=== Starting LabelMapper Example ===\n")

    # -------------------------------------------------------------------------
    # 1. Define the Mapping
    # -------------------------------------------------------------------------
    # In this scenario, we want to compare a neuron group we'll call "My_Custom_Group".
    # We have specific IDs/names for this group in each dataset.
    
    # Note: These are hypothetical IDs for demonstration. 
    # In a real scenario, you would use actual bodyIds or type names.
    source_mapping_dict = {
        'hemibrain_v1_2_1': [['5813020828', '5813057000']],  # Example bodyIds in Hemibrain
        'flywire_FAFB_v783': [['720575940621039145']]       # Example bodyId in Flywire
    }
    
    # The 'source_labels' list corresponds to the groups defined in the values above.
    # Since we have a list of lists (one inner list), we have one group.
    source_group_labels = ['My_Custom_Group']

    print("Defining Source Mapping (Dict):")
    print(f"  Standard Label: {source_group_labels[0]}")
    print(f"  Hemibrain IDs: {source_mapping_dict['hemibrain_v1_2_1'][0]}")
    print(f"  Flywire IDs:   {source_mapping_dict['flywire_FAFB_v783'][0]}")
    print("-" * 40)

    # -------------------------------------------------------------------------
    # 2. Initialize LabelMapper
    # -------------------------------------------------------------------------
    
    # --- Option 1: From Dictionary ---
    print("\n--- Option 1: Initialize from Dictionary ---")
    mapper_dict = LabelMapper(
        source_mapping_dict=source_mapping_dict,
        source_labels=source_group_labels
    )
    print("LabelMapper (Dict) initialized successfully.")

    # --- Option 2: From CSV File ---
    print("\n--- Option 2: Initialize from CSV File ---")
    # Create a dummy CSV file for demonstration
    csv_content = """std_label,hemibrain_v1_2_1,flywire_FAFB_v783
My_Custom_Group,5813020828;5813057000,720575940621039145
Another_Group,123456789,987654321"""
    
    csv_filename = 'example_mapping.csv'
    with open(csv_filename, 'w') as f:
        f.write(csv_content)
    print(f"Created temporary mapping file: {csv_filename}")
    print(f"Content:\n{csv_content}\n")

    # Initialize mapper from file
    mapper_file = LabelMapper(
        source_mapping_file=csv_filename
    )
    print("LabelMapper (File) initialized successfully.")
    
    # We'll use the file-based mapper for the rest of the example
    mapper = mapper_file

    # -------------------------------------------------------------------------
    # 3. Configure Comparison Parameters
    # -------------------------------------------------------------------------
    # We pass the `mapper` object directly to `source_neurons`.
    # The ComparisonParameters class detects this and uses it to resolve neurons for each dataset.
    
    params = ComparisonParameters(
        datasets=['hemibrain_v1_2_1', 'flywire_FAFB_v783'],
        datasets_nickname=['Hemi', 'Flywire'],
        
        # PASS THE MAPPER HERE
        source_neurons=mapper,
        
        # For target neurons, we can still use standard regex or lists if they are consistent,
        # or use another mapper (or the same one if it contained target mappings).
        # Here we'll just use a broad regex for demonstration.
        target_neurons=['KC.*'], 
        
        max_interlayer=1,
        thresholds=[10],
        output_folder='../../local_data/dataset_comparison',
        saveas='label_mapper_demo'
    )
    
    print("\nComparisonParameters configured with LabelMapper.")

    # -------------------------------------------------------------------------
    # 4. Verify Resolution (Optional but recommended)
    # -------------------------------------------------------------------------
    # You can verify what the mapper resolves to for a specific dataset using the helper method
    # on the parameters object (which delegates to the mapper).
    
    print("\nVerifying Neuron Resolution:")
    for dataset in params.datasets:
        resolved = params.get_source_neurons_for_dataset(dataset)
        print(f"  Dataset: {dataset}")
        print(f"  Resolved Source Neurons: {resolved}")

    # -------------------------------------------------------------------------
    # 5. Run Analysis
    # -------------------------------------------------------------------------
    # Initialize the analyzer
    analyzer = ComparisonAnalyzer(params)
    
    print("\nAnalyzer initialized. Ready to run comparison.")
    print("Note: This example stops before running the full heavy comparison to avoid")
    print("      waiting for data fetching in this demo script.")
    print("      Uncomment 'analyzer.run()' below to execute the full pipeline.")
    
    # analyzer.run()
    
    # Cleanup temporary file
    if os.path.exists(csv_filename):
        os.remove(csv_filename)
        print(f"\nRemoved temporary file: {csv_filename}")
    
    print("\n=== LabelMapper Example Completed Successfully ===")

if __name__ == "__main__":
    run_label_mapper_example()
