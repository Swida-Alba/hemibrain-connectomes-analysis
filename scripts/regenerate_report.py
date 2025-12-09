import os
import sys
import pandas as pd
import glob

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.comparison.interactive_heatmap import generate_interactive_heatmap

def regenerate_report(input_dir, timestamp, output_path):
    print(f"Looking for files in {input_dir} with timestamp {timestamp}")
    
    # Find matrix files
    # Pattern: comparison_matrix_{metric}_{timestamp}.csv
    pattern = os.path.join(input_dir, f"comparison_matrix_*_{timestamp}.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No matrix files found matching pattern: {pattern}")
        return
        
    matrices = {}
    for f in files:
        # Extract metric name
        basename = os.path.basename(f)
        # comparison_matrix_{metric}_{timestamp}.csv
        # Remove prefix and suffix
        parts = basename.replace('comparison_matrix_', '').replace(f'_{timestamp}.csv', '')
        metric = parts
        
        print(f"Loading {metric} from {f}")
        try:
            df = pd.read_csv(f)
            if 'neuron_type' in df.columns:
                df = df.set_index('neuron_type')
            matrices[metric] = df
        except Exception as e:
            print(f"Error loading {f}: {e}")
        
    if matrices:
        print(f"Generating interactive heatmap to {output_path}")
        try:
            generate_interactive_heatmap(
                matrices, 
                output_path, 
                title=f"Connectivity Profile Comparison ({timestamp})", 
                showfig=False
            )
            print(f"Successfully generated {output_path}")
        except Exception as e:
            print(f"Error generating heatmap: {e}")
    else:
        print("No matrices loaded.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python regenerate_report.py <input_dir> <timestamp> [output_path]")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    timestamp = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "connectivity_profile_report.html"
    
    regenerate_report(input_dir, timestamp, output_path)
