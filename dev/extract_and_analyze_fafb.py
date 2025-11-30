import os
import gzip
import shutil
import pandas as pd
from pathlib import Path

# Define paths
base_dir = Path('/Users/apple/Documents/GitHub/hemibrain-connectomes-analysis-v3.1/datasets/flywire_FAFB_v783')
output_dir = base_dir / 'extracted'

# Create output directory if it doesn't exist
if not output_dir.exists():
    output_dir.mkdir()

# List of .csv.gz files
files = [f for f in os.listdir(base_dir) if f.endswith('.csv.gz')]

print(f"Found {len(files)} .csv.gz files.")

analysis_results = []

for filename in files:
    file_path = base_dir / filename
    output_filename = filename[:-3] # Remove .gz
    output_path = output_dir / output_filename
    
    print(f"Extracting {filename}...")
    
    try:
        with gzip.open(file_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"  -> Extracted to {output_path}")
        
        # Analyze the file
        try:
            # Read just the header and first row to be fast
            df = pd.read_csv(output_path, nrows=5)
            columns = list(df.columns)
            row_count = "Unknown (did not read full file)"
            
            analysis_results.append({
                'filename': output_filename,
                'columns': columns,
                'sample_data': df.head(1).to_dict(orient='records')
            })
            
        except Exception as e:
            print(f"  -> Error reading CSV: {e}")
            
    except Exception as e:
        print(f"  -> Error extracting: {e}")

print("\n" + "="*50)
print("FILE ANALYSIS")
print("="*50)

for result in analysis_results:
    print(f"\nFile: {result['filename']}")
    print(f"Columns: {result['columns']}")
    # print(f"Sample: {result['sample_data']}")

print("\n" + "="*50)
print("RECOMMENDATIONS FOR PATH FINDING")
print("="*50)

# Logic to identify required files
neuron_info_candidates = []
connection_candidates = []

for result in analysis_results:
    cols = result['columns']
    filename = result['filename']
    
    # Check for neuron info (bodyId, type, instance, etc.)
    # FAFB/FlyWire often uses 'root_id' instead of 'bodyId'
    has_id = any(c in cols for c in ['bodyId', 'root_id', 'id', 'pt_root_id'])
    has_type = any(c in cols for c in ['type', 'cell_type', 'tag', 'class'])
    has_synapses = any(c in cols for c in ['pre', 'post', 'synapses'])
    
    if has_id and (has_type or has_synapses):
        neuron_info_candidates.append(filename)
        
    # Check for connection info
    has_source = any(c in cols for c in ['pre_root_id', 'bodyId_pre', 'source'])
    has_target = any(c in cols for c in ['post_root_id', 'bodyId_post', 'target'])
    has_weight = any(c in cols for c in ['weight', 'syn_count', 'synapses'])
    
    if has_source and has_target:
        connection_candidates.append(filename)

print("Potential Neuron Info Files:")
for f in neuron_info_candidates:
    print(f"  - {f}")

print("\nPotential Connection Files:")
for f in connection_candidates:
    print(f"  - {f}")
