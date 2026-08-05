import os
import pandas as pd

def process_neurons_to_parquet(read_path, save_path, save_csv_path=None):
    """
    Process neurons.csv.gz into neuron_df parquet format for BANC.
    """
    if os.path.exists(save_path):
        print(f"  ✓ Found existing converted file: {save_path}")
        return True

    print(f"  ⏳ Processing {read_path} -> {save_path}...")
    
    if not os.path.exists(read_path):
        print(f"  ⚠️ Error: Input file not found: {read_path}")
        return False

    try:
        # Read neurons
        print("  Reading neuron data...")
        df = pd.read_csv(read_path, compression='gzip' if read_path.endswith('.gz') else None)
        
        # Rename columns to match coana expectations
        # BANC columns: ['Root ID', 'Top in/out region', 'Community labels', 'Predicted NT type', 
        # 'Predicted NT confidence', 'Verified NT type', 'Verified Neuropeptide', 'Body Part', 
        # 'Function', 'Flow', 'Super Class', 'Class', 'Sub Class', 'Hemilineage', 'Nerve', 
        # 'Soma side', 'Primary Cell Type', 'Alternative Cell Type(s)', 'Cable length (nm)', 
        # 'Surface area (nm^2)', 'Volume (nm^3)']
        
        rename_map = {
            'Root ID': 'bodyId',
            'Primary Cell Type': 'type',
            'Super Class': 'super_class',
            'Flow': 'flow',
            'Nerve': 'nerve',
            'Hemilineage': 'hemilineage',
            'Predicted NT type': 'nt_type'
        }
        
        # Check which columns actually exist
        existing_cols = df.columns.tolist()
        actual_rename = {k: v for k, v in rename_map.items() if k in existing_cols}
        
        df = df.rename(columns=actual_rename)
        df['bodyId'] = df['bodyId'].astype(str)
        
        # Deduplicate if needed
        if df.duplicated(subset=['bodyId']).any():
            print("  Deduplicating dataframe...")
            df = df.drop_duplicates(subset=['bodyId'])

        # Fill missing standard columns
        if 'type' not in df.columns: df['type'] = 'Unknown'
        df['type'] = df['type'].fillna('Unknown')
        
        # Instance: use type if no specific name column
        df['instance'] = df['type']
        
        df['post'] = 0 # Placeholder
        
        # Select columns (keep all renamed + others)
        standard_cols = ['bodyId', 'type', 'instance', 'post', 'super_class', 'Class', 'Sub Class', 'Soma side', 'hemilineage', 'nerve', 'flow', 'nt_type']
        
        # Add any other columns
        all_cols = list(df.columns)
        ordered_cols = [c for c in standard_cols if c in all_cols] + [c for c in all_cols if c not in standard_cols]
        df = df[ordered_cols]
        
        # Sort by bodyId
        df = df.sort_values('bodyId')
        
        print(f"  Saving to Parquet: {save_path}...")
        df.to_parquet(save_path, index=False, compression='snappy')
        
        if save_csv_path:
            print(f"  Saving to CSV: {save_csv_path}...")
            df.to_csv(save_csv_path, index=False)
        
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"  ✓ Conversion complete. Output size: {file_size_mb:.2f} MB")
        return True
        
    except Exception as e:
        print(f"  ⚠️ Error processing neurons: {e}")
        return False

def process_connections_to_parquet(read_path, save_path, save_csv_path=None):
    """
    Process connections_princeton.csv.gz into merged_connections parquet format.
    Aggregates weights across ROIs.
    """
    if os.path.exists(save_path):
        print(f"  ✓ Found existing converted file: {save_path}")
        return True

    print(f"  ⏳ Processing {read_path} -> {save_path}...")
    
    if not os.path.exists(read_path):
        print(f"  ⚠️ Error: Input file not found: {read_path}")
        return False

    try:
        df = pd.read_csv(read_path, compression='gzip' if read_path.endswith('.gz') else None)
        
        # Rename for consistency
        # BANC columns: ['pre_root_id', 'post_root_id', 'neuropil', 'syn_count', 'nt_type']
        rename_map = {
            'pre_root_id': 'bodyId_pre',
            'post_root_id': 'bodyId_post',
            'syn_count': 'weight',
            'neuropil': 'roi'
        }
        df = df.rename(columns=rename_map)
        
        # Ensure strings
        df['bodyId_pre'] = df['bodyId_pre'].astype(str)
        df['bodyId_post'] = df['bodyId_post'].astype(str)
        
        # Aggregate weights and ROIs (sum weights, join ROIs)
        print("  Aggregating connections across ROIs...")
        if 'roi' in df.columns:
            df = df.groupby(['bodyId_pre', 'bodyId_post'], as_index=False).agg({
                'weight': 'sum',
                'roi': lambda x: '|'.join(sorted(set(str(v) for v in x if pd.notnull(v) and str(v) != 'nan')))
            })
        else:
            print("  Note: 'roi' column not found, aggregating weights only.")
            df = df.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)['weight'].sum()
            df['roi'] = 'WholeBrain'

        # Sort by pre, post
        print("  Sorting connections...")
        df = df.sort_values(['bodyId_pre', 'bodyId_post'])
        
        print(f"  Saving to Parquet: {save_path}...")
        df.to_parquet(save_path, index=False, compression='snappy')
        
        if save_csv_path:
            print(f"  Saving to CSV: {save_csv_path}...")
            df.to_csv(save_csv_path, index=False)
        
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"  ✓ Conversion complete. Output size: {file_size_mb:.2f} MB")
        return True
        
    except Exception as e:
        print(f"  ⚠️ Error processing connections: {e}")
        return False

def update_neuron_post_counts(neuron_path, conn_path, save_csv_path=None):
    """
    Update the 'post' column in the neuron DataFrame by summing weights from the connections DataFrame.
    """
    print("  Updating neuron post-synaptic counts from connections...")
    try:
        # Load neurons
        if neuron_path.endswith('.parquet'):
            df_neuron = pd.read_parquet(neuron_path)
        else:
            df_neuron = pd.read_csv(neuron_path, dtype={'bodyId': str})
            
        # Load connections (only need bodyId_post and weight)
        print("  Loading connections for count calculation...")
        if conn_path.endswith('.parquet'):
            df_conn = pd.read_parquet(conn_path, columns=['bodyId_post', 'weight'])
        else:
            df_conn = pd.read_csv(conn_path, usecols=['bodyId_post', 'weight'], dtype={'bodyId_post': str})
            
        # Calculate post counts
        print("  Calculating post counts...")
        # Ensure bodyId_post is string
        df_conn['bodyId_post'] = df_conn['bodyId_post'].astype(str)
        post_counts = df_conn.groupby('bodyId_post')['weight'].sum()
        
        # Update df_neuron
        # Ensure bodyId is string
        df_neuron['bodyId'] = df_neuron['bodyId'].astype(str)
        
        # Map counts
        print("  Mapping counts to neurons...")
        df_neuron['post'] = df_neuron['bodyId'].map(post_counts).fillna(0).astype(int)
        
        # Save
        print(f"  Saving updated neurons to {neuron_path}...")
        if neuron_path.endswith('.parquet'):
            df_neuron.to_parquet(neuron_path, index=False, compression='snappy')
        else:
            df_neuron.to_csv(neuron_path, index=False)
            
        if save_csv_path:
            print(f"  Saving updated neurons to {save_csv_path}...")
            df_neuron.to_csv(save_csv_path, index=False)
            
        print("  ✓ Post counts updated.")
        return True
    except Exception as e:
        print(f"  ⚠️ Error updating post counts: {e}")
        return False

def ensure_banc_data(dataset_name, dataset_dir):
    """
    Ensure BANC data is available and converted for the given dataset.
    """
    print(f"\nChecking BANC data for {dataset_name}...")
    
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
        print(f"Created dataset folder: {dataset_dir}")

    downloads_dir = os.path.join(dataset_dir, "downloads")
    if not os.path.exists(downloads_dir):
        os.makedirs(downloads_dir, exist_ok=True)

    # Define target files
    neuron_pq = os.path.join(dataset_dir, f"{dataset_name}_allneurons_neuron_df.parquet")
    neuron_csv = os.path.join(dataset_dir, f"{dataset_name}_allneurons_neuron_df.csv")
    conn_pq = os.path.join(dataset_dir, f"{dataset_name}_merged_connections.parquet")
    conn_csv = os.path.join(dataset_dir, f"{dataset_name}_merged_connections.csv")

    all_critical_present = True

    # --- 1. Neurons ---
    if os.path.exists(neuron_pq):
        print(f"  ✓ Found existing neurons: {os.path.basename(neuron_pq)}")
    else:
        print("  Checking neuron source files...")
        neurons_raw = os.path.join(downloads_dir, "neurons.csv.gz")
        
        if not os.path.exists(neurons_raw):
            print(f"  ❌ Missing required file: neurons.csv.gz")
            all_critical_present = False
        else:
            if not process_neurons_to_parquet(neurons_raw, neuron_pq, save_csv_path=neuron_csv):
                all_critical_present = False

    # --- 2. Connections ---
    if os.path.exists(conn_pq):
        print(f"  ✓ Found existing connections: {os.path.basename(conn_pq)}")
    else:
        print("  Checking connection source files...")
        conn_raw = os.path.join(downloads_dir, "connections_princeton.csv.gz")
        
        if not os.path.exists(conn_raw):
            print(f"  ❌ Missing required file: connections_princeton.csv.gz")
            all_critical_present = False
        else:
            if not process_connections_to_parquet(conn_raw, conn_pq, save_csv_path=conn_csv):
                all_critical_present = False

    # --- Post Counts Update ---
    if os.path.exists(neuron_pq) and os.path.exists(conn_pq):
        try:
            # Read just the post column to check if it's all zeros
            df_check = pd.read_parquet(neuron_pq, columns=['post'])
            if df_check['post'].sum() == 0:
                print("  ℹ️  Post counts are 0. Updating from connections...")
                update_neuron_post_counts(neuron_pq, conn_pq, save_csv_path=neuron_csv)
            else:
                print("  ✓ Post counts already populated.")
        except Exception as e:
            print(f"  ⚠️ Could not check post counts: {e}")

    if not all_critical_present:
        print("\n" + "="*60)
        print("MISSING CRITICAL FILES")
        print("Please download missing files to:", downloads_dir)
        print("See https://codex.flywire.ai/api/download?dataset=banc")
        print("="*60 + "\n")
        return False

    return True

if __name__ == "__main__":
    # Standard run for installation
    dataset_name = "flywire_BANC_v626"
    # Determine project root (parent of src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_root, "datasets", dataset_name)
    
    print(f"Running BANC data preparation for {dataset_name}...")
    print(f"Target directory: {dataset_dir}")
    
    ensure_banc_data(dataset_name, dataset_dir)
