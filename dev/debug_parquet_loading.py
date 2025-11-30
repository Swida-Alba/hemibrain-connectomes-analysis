import pandas as pd
import os
import sys

# Path to the parquet file
parquet_path = "/Users/apple/Documents/GitHub/hemibrain-connectomes-analysis-v3.1/datasets/flywire_FAFB_v783/flywire_FAFB_v783_skeletons.parquet"

if not os.path.exists(parquet_path):
    print(f"File not found: {parquet_path}")
    sys.exit(1)

print(f"Checking {parquet_path}...")

try:
    # Read only bodyId column to count unique IDs
    df_ids = pd.read_parquet(parquet_path, columns=['bodyId'])
    unique_ids = df_ids['bodyId'].nunique()
    total_rows = len(df_ids)
    
    print(f"Total rows: {total_rows}")
    print(f"Unique bodyIds: {unique_ids}")
    
    print("\nFirst 5 IDs:")
    print(df_ids['bodyId'].head().tolist())
    
    print("\nLast 5 IDs:")
    print(df_ids['bodyId'].tail().tolist())

except Exception as e:
    print(f"Error: {e}")
