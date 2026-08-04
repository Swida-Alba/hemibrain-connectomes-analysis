import os
import shutil
import pandas as pd
import numpy as np
import zipfile
import io
import navis
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import concurrent.futures
import multiprocessing

def _parse_swc_batch(zip_path, filenames):
    """Helper function to parse a batch of SWC files from a zip."""
    # Use column-oriented storage for memory efficiency
    data = {
        'bodyId': [], 'node_id': [], 'type': [], 
        'x': [], 'y': [], 'z': [], 'radius': [], 'parent_id': []
    }
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            for filename in filenames:
                try:
                    # Handle potential folder prefixes in zip
                    body_id = os.path.basename(filename).split('.')[0]
                    
                    with z.open(filename) as f:
                        content = f.read().decode('utf-8')
                    
                    # Manual parsing
                    lines = [l.strip() for l in content.split('\n') if l.strip() and not l.strip().startswith('#')]
                    
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 7:
                            # SWC: id type x y z radius parent
                            data['bodyId'].append(body_id)
                            data['node_id'].append(int(parts[0]))
                            data['type'].append(int(parts[1]))
                            data['x'].append(float(parts[2]))
                            data['y'].append(float(parts[3]))
                            data['z'].append(float(parts[4]))
                            data['radius'].append(float(parts[5]))
                            data['parent_id'].append(int(parts[6]))
                except Exception:
                    continue
    except Exception as e:
        print(f"Error in batch: {e}")
    return data

def process_neurons_to_parquet(read_path, save_path, save_csv_path=None, enrichment_files=None):
    """
    Process classification.csv.gz and optional enrichment files into neuron_df parquet format.
    Enrichment files: names, coordinates, neurons (neurotransmitters), cell_stats.
    """
    if os.path.exists(save_path):
        print(f"  ✓ Found existing converted file: {save_path}")
        return True

    print(f"  ⏳ Processing {read_path} -> {save_path}...")
    
    if not os.path.exists(read_path):
        print(f"  ⚠️ Error: Input file not found: {read_path}")
        return False

    try:
        # Read classifications (Base)
        print("  Reading classification data...")
        df = pd.read_csv(read_path, compression='gzip' if read_path.endswith('.gz') else None)
        
        # Rename columns to match coana expectations
        rename_map = {
            'root_id': 'bodyId',
            'class': 'cell_class',
            'sub_class': 'cell_type',
            'side': 'hemisphere'
        }
        df = df.rename(columns=rename_map)
        df['bodyId'] = df['bodyId'].astype(str)
        
        # Deduplicate base if needed
        if df.duplicated(subset=['bodyId']).any():
            print("  Deduplicating base dataframe...")
            df = df.drop_duplicates(subset=['bodyId'])

        # --- Enrichment ---
        if enrichment_files:
            # 1. Names
            if enrichment_files.get('names'):
                fpath = enrichment_files['names']
                print(f"  Merging names from {os.path.basename(fpath)}...")
                df_names = pd.read_csv(fpath, compression='gzip' if fpath.endswith('.gz') else None, dtype={'root_id': str})
                df_names = df_names.rename(columns={'root_id': 'bodyId', 'name': 'instance'})
                if 'group' in df_names.columns: df_names = df_names.drop(columns=['group'])
                df = pd.merge(df, df_names, on='bodyId', how='left')

            # 2. Coordinates
            if enrichment_files.get('coordinates'):
                fpath = enrichment_files['coordinates']
                print(f"  Merging coordinates from {os.path.basename(fpath)}...")
                df_coords = pd.read_csv(fpath, compression='gzip' if fpath.endswith('.gz') else None, dtype={'root_id': str})
                df_coords = df_coords.rename(columns={'root_id': 'bodyId'})
                if 'supervoxel_id' in df_coords.columns: df_coords = df_coords.drop(columns=['supervoxel_id'])
                df_coords = df_coords.drop_duplicates(subset=['bodyId'])
                df = pd.merge(df, df_coords, on='bodyId', how='left')

            # 3. Neurotransmitters (neurons.csv)
            if enrichment_files.get('neurons'):
                fpath = enrichment_files['neurons']
                print(f"  Merging neurotransmitters from {os.path.basename(fpath)}...")
                df_nt = pd.read_csv(fpath, compression='gzip' if fpath.endswith('.gz') else None, dtype={'root_id': str})
                df_nt = df_nt.rename(columns={'root_id': 'bodyId'})
                if 'group' in df_nt.columns: df_nt = df_nt.drop(columns=['group'])
                df = pd.merge(df, df_nt, on='bodyId', how='left')

            # 4. Cell Stats
            if enrichment_files.get('cell_stats'):
                fpath = enrichment_files['cell_stats']
                print(f"  Merging cell stats from {os.path.basename(fpath)}...")
                df_stats = pd.read_csv(fpath, compression='gzip' if fpath.endswith('.gz') else None, dtype={'root_id': str})
                df_stats = df_stats.rename(columns={'root_id': 'bodyId'})
                df = pd.merge(df, df_stats, on='bodyId', how='left')

            # 5. Cell Types (consolidated_cell_types.csv.gz)
            if enrichment_files.get('cell_types'):
                fpath = enrichment_files['cell_types']
                print(f"  Merging cell types from {os.path.basename(fpath)}...")
                df_types = pd.read_csv(fpath, compression='gzip' if fpath.endswith('.gz') else None, dtype={'root_id': str})
                
                # Rename root_id -> bodyId, primary_type -> type
                rename_dict = {'root_id': 'bodyId'}
                if 'primary_type' in df_types.columns:
                    rename_dict['primary_type'] = 'type'
                
                df_types = df_types.rename(columns=rename_dict)
                
                # Merge all columns (keep all original col names, use suffix for conflicts)
                df = pd.merge(df, df_types, on='bodyId', how='left', suffixes=(None, '_enriched'))

        # Logic for type: type (from consolidated) > cell_type > cell_class > super_class
        if 'type' not in df.columns: df['type'] = np.nan
        if 'cell_type' not in df.columns: df['cell_type'] = np.nan
        if 'cell_class' not in df.columns: df['cell_class'] = np.nan
        if 'super_class' not in df.columns: df['super_class'] = np.nan
        
        df['type'] = df['type'].fillna(df['cell_type']).fillna(df['cell_class']).fillna(df['super_class']).fillna('Unknown')
        
        # Instance: usually same as type or specific name. 
        if 'instance' not in df.columns:
            df['instance'] = df['type']
        else:
            df['instance'] = df['instance'].fillna(df['type'])
        
        df['post'] = 0 # Placeholder
        
        # Select columns (keep all enriched columns + standard ones)
        standard_cols = ['bodyId', 'type', 'instance', 'post', 'super_class', 'cell_class', 'cell_type', 'hemisphere', 'nucleus_id', 'hemilineage', 'nerve', 'flow']
        # Add any other columns found in enrichment (e.g. x, y, z, nt_type, etc.)
        all_cols = list(df.columns)
        # Prioritize standard columns order
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
    Process connections.csv.gz into merged_connections parquet format.
    Logic matches coana.py _prepare_flywire_data.
    """
    if os.path.exists(save_path):
        print(f"  ✓ Found existing converted file: {save_path}")
        return True

    print(f"  ⏳ Processing {read_path} -> {save_path}...")
    
    if not os.path.exists(read_path):
        print(f"  ⚠️ Error: Input file not found: {read_path}")
        return False

    try:
        # Read connections (can be large, but usually fits in memory for FlyWire ~1GB compressed)
        # If too large, we might need chunking, but sorting requires full dataset or external sort.
        # For now, assume memory is sufficient as per coana.py assumption.
        df = pd.read_csv(read_path, compression='gzip')
        
        # Rename for consistency
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

def process_synapse_table_to_parquet(read_path, save_path, chunksize=100000):
    """
    Process synapse table CSV to Parquet.
    Includes logic to fix short IDs (FlyWire specific).
    """
    if os.path.exists(save_path):
        print(f"  ✓ Found existing converted file: {save_path}")
        return True

    print(f"  ⏳ Processing {read_path} -> {save_path}...")
    
    if not os.path.exists(read_path):
        print(f"  ⚠️ Error: Input file not found: {read_path}")
        return False

    try:
        compression = 'gzip' if read_path.endswith('.gz') else None
        
        chunks = []
        pre_root_id_col = None
        post_root_id_col = None
        
        with pd.read_csv(read_path, compression=compression, chunksize=chunksize) as reader:
            for i, chunk in enumerate(tqdm(reader, desc="  Reading chunks")):
                # Work on a copy to avoid SettingWithCopyWarning and internal block manager issues
                chunk = chunk.copy()
                
                # Identify ID columns dynamically (only once)
                if pre_root_id_col is None:
                    pre_cols = [c for c in chunk.columns if c.startswith('pre_root_id')]
                    post_cols = [c for c in chunk.columns if c.startswith('post_root_id')]
                    if pre_cols and post_cols:
                        pre_root_id_col = pre_cols[0]
                        post_root_id_col = post_cols[0]
                    else:
                        print("  ⚠️ Error: Could not find root_id columns in CSV")
                        return False

                # Ensure strings
                chunk[pre_root_id_col] = chunk[pre_root_id_col].astype(str)
                chunk[post_root_id_col] = chunk[post_root_id_col].astype(str)
                
                # Fix short IDs if needed (FlyWire specific issue)
                # Check first row of chunk
                if len(chunk[pre_root_id_col].iloc[0]) == 9:
                    chunk[pre_root_id_col] = '720575940' + chunk[pre_root_id_col]
                    chunk[post_root_id_col] = '720575940' + chunk[post_root_id_col]
                
                chunks.append(chunk)
        
        if not chunks:
            print("  ⚠️ Error: No data found.")
            return False

        print("  Concatenating chunks...")
        full_df = pd.concat(chunks, ignore_index=True)
        
        print("  Sorting by root IDs...")
        full_df = full_df.sort_values([pre_root_id_col, post_root_id_col])
        
        print(f"  Saving to Parquet: {save_path}...")
        full_df.to_parquet(save_path, index=False, compression='snappy')
        
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"  ✓ Conversion complete. Output size: {file_size_mb:.2f} MB")
        return True
        
    except Exception as e:
        print(f"  ⚠️ Error processing synapse table: {e}")
        return False

def process_skeletons_to_parquet(zip_path, save_path, batch_size=500):
    """
    Process skeletons from a ZIP of SWC files to a single Parquet file.
    Optimized for space and speed using parallel processing and batched writing.
    """
    if os.path.exists(save_path):
        print(f"  ✓ Found existing converted file: {save_path}")
        return True

    print(f"  ⏳ Processing {zip_path} -> {save_path}...")
    
    if not os.path.exists(zip_path):
        print(f"  ⚠️ Error: Input file not found: {zip_path}")
        return False

    # Use a temporary file to avoid corrupting the destination if interrupted
    temp_path = save_path + '.tmp'
    writer = None
    
    try:
        # Get list of files first
        with zipfile.ZipFile(zip_path, 'r') as z:
            swc_files = [f for f in z.namelist() if f.endswith('.swc')]
            swc_files.sort()
            
        if not swc_files:
            print("  ⚠️ No SWC files found in zip.")
            return False
            
        print(f"  Found {len(swc_files)} skeletons. Starting parallel processing...")
        
        # Define schema
        schema = pa.schema([
            ('bodyId', pa.string()),
            ('node_id', pa.int32()),
            ('type', pa.int8()),
            ('x', pa.int32()),
            ('y', pa.int32()),
            ('z', pa.int32()),
            ('radius', pa.int32()),
            ('parent_id', pa.int32())
        ])
        
        # Prepare chunks
        chunks = [swc_files[i:i + batch_size] for i in range(0, len(swc_files), batch_size)]
        
        # Cap workers to avoid OOM on high-core machines with limited RAM per core
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        num_workers = min(num_workers, 8) 
        
        print(f"  Using {num_workers} workers for {len(chunks)} batches (batch_size={batch_size})...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all jobs
            futures = {executor.submit(_parse_swc_batch, zip_path, chunk): chunk for chunk in chunks}
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(chunks), desc="  Converting"):
                batch_nodes = future.result()
                
                # Check if batch has data (using column-oriented check)
                if batch_nodes and batch_nodes['bodyId']:
                    df_batch = pd.DataFrame(batch_nodes)
                    
                    # Optimize types
                    df_batch['bodyId'] = df_batch['bodyId'].astype(str)
                    df_batch['node_id'] = df_batch['node_id'].astype('int32')
                    df_batch['type'] = df_batch['type'].astype('int8')
                    df_batch['parent_id'] = df_batch['parent_id'].astype('int32')
                    
                    # Optimize coords
                    for col in ['x', 'y', 'z']:
                        df_batch[col] = df_batch[col].round().astype('int32')
                        
                    df_batch['radius'] = df_batch['radius'].astype('int32')
                    
                    # Sort batch
                    df_batch = df_batch.sort_values(['bodyId', 'node_id'])
                    
                    # Convert to Table
                    table = pa.Table.from_pandas(df_batch, schema=schema)
                    
                    # Initialize writer if first batch
                    if writer is None:
                        writer = pq.ParquetWriter(temp_path, schema=schema, compression='snappy')
                    
                    writer.write_table(table)
                    
        if writer:
            writer.close()
            # Rename temp file to final path
            if os.path.exists(save_path):
                os.remove(save_path)
            os.rename(temp_path, save_path)
            
            file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
            print(f"  ✓ Conversion complete. Output size: {file_size_mb:.2f} MB")
            return True
        else:
            print("  ⚠️ No valid skeleton data found.")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
        
    except Exception as e:
        print(f"  ⚠️ Error processing skeletons: {e}")
        if writer:
            writer.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def update_neuron_post_counts(neuron_path, conn_path, save_csv_path=None):
    """
    Update the 'post' column in the neuron DataFrame by summing weights from the connections DataFrame.
    This is necessary because FlyWire classification files don't include post-synaptic counts.
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

def print_download_instructions(downloads_dir):
    print(f"\033[31mError: Missing converted files or source files.\033[0m")
    print(f"\033[33mPlease download the following files from: \033[34mhttps://codex.flywire.ai/api/download?dataset=fafb\033[0m")
    print(f"\033[33mAnd save them to: \033[34m{downloads_dir}\033[0m")
    
    files_info = [
        ("classification.csv.gz", "Neuron Classification"),
        ("names.csv.gz", "Optional neuron metadata: names"),
        ("coordinates.csv.gz", "Optional neuron metadata: soma coordinates"),
        ("neurons.csv.gz", "Optional neuron metadata: neurotransmitters"),
        ("cell_stats.csv.gz", "Optional neuron metadata: cell statistics"),
        ("consolidated_cell_types.csv.gz", "Optional neuron metadata: consolidated cell types"),
        ("connections_princeton_no_threshold.csv.gz", "Connectivity Data"),
        ("fafb_v783_princeton_synapse_table.csv.gz", "Optional synapse coordinates for visualization"),
        ("sk_lod1_783_healed.zip", "Optional skeletons for visualization")
    ]

    for fname, desc in files_info:
        fpath = os.path.join(downloads_dir, fname)
        exists = os.path.exists(fpath)
        
        # Check uncompressed version if .gz
        if not exists and fname.endswith('.gz'):
            fpath_uncompressed = os.path.join(downloads_dir, fname[:-3])
            if os.path.exists(fpath_uncompressed):
                exists = True
        
        # Check fallbacks for connections
        if not exists and fname == "connections_princeton_no_threshold.csv.gz":
            if os.path.exists(os.path.join(downloads_dir, "connections_princeton.csv.gz")) or \
               os.path.exists(os.path.join(downloads_dir, "connections.csv.gz")):
                exists = True

        is_optional = desc.startswith("Optional") or "Visualization" in desc
        fname_display = f"[optional] {fname}" if is_optional else fname

        if exists:
            print(f"  ✅ [existed] {fname_display} ({desc})")
        else:
            if is_optional:
                print(f"  - {fname_display} ({desc})")
            else:
                print(f"  ❌ {fname_display} ({desc})")

def ensure_flywire_data(dataset_name, dataset_dir):
    """
    Ensure FlyWire data is available and converted for the given dataset.
    Checks each component (neurons, connections, synapses, skeletons) independently.
    """
    print(f"\nChecking FlyWire data for {dataset_name}...")
    
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
    syn_pq = os.path.join(dataset_dir, f"{dataset_name}_synapse_table.parquet")
    sk_zip_dst = os.path.join(dataset_dir, "sk_lod1_783_healed.zip")

    all_critical_present = True

    # --- 1. Neurons ---
    if os.path.exists(neuron_pq):
        print(f"  ✓ Found existing neurons: {os.path.basename(neuron_pq)}")
    else:
        print("  Checking neuron source files...")
        class_raw = os.path.join(downloads_dir, "classification.csv.gz")
        
        if not os.path.exists(class_raw):
            print(f"  ❌ Missing required file: classification.csv.gz")
            all_critical_present = False
        else:
            # Check enrichment files
            required_enrichment = {
                'names': os.path.join(downloads_dir, "names.csv.gz"),
                'coordinates': os.path.join(downloads_dir, "coordinates.csv.gz"),
                'neurons': os.path.join(downloads_dir, "neurons.csv.gz"),
                'cell_stats': os.path.join(downloads_dir, "cell_stats.csv.gz"),
                'cell_types': os.path.join(downloads_dir, "consolidated_cell_types.csv.gz")
            }
            
            enrichment_files = {}
            missing_enrichment = []

            for key, path in required_enrichment.items():
                if os.path.exists(path):
                    enrichment_files[key] = path
                else:
                    uncompressed = path.replace('.gz', '')
                    if os.path.exists(uncompressed):
                        enrichment_files[key] = uncompressed
                    else:
                        missing_enrichment.append(os.path.basename(path))
            
            if missing_enrichment:
                print(f"  ⚠️ Missing neuron metadata files: {', '.join(missing_enrichment)}")
                print("  (Will proceed without them, but neuron info will be incomplete)")
            
            if not process_neurons_to_parquet(class_raw, neuron_pq, save_csv_path=neuron_csv, enrichment_files=enrichment_files):
                all_critical_present = False

    # --- 2. Connections ---
    if os.path.exists(conn_pq):
        print(f"  ✓ Found existing connections: {os.path.basename(conn_pq)}")
    else:
        print("  Checking connection source files...")
        conn_raw = os.path.join(downloads_dir, "connections_princeton_no_threshold.csv.gz")
        if not os.path.exists(conn_raw):
            # Fallbacks
            conn_raw = os.path.join(downloads_dir, "connections_princeton.csv.gz")
            if not os.path.exists(conn_raw):
                conn_raw = os.path.join(downloads_dir, "connections.csv.gz")

        if os.path.exists(conn_raw):
            if not process_connections_to_parquet(conn_raw, conn_pq, save_csv_path=conn_csv):
                all_critical_present = False
        else:
            print(f"  ❌ Missing required file: connections_princeton_no_threshold.csv.gz")
            all_critical_present = False

    # --- 3. Synapses (Optional) ---
    if os.path.exists(syn_pq):
        print(f"  ✓ Found existing synapse table: {os.path.basename(syn_pq)}")
    else:
        syn_raw = os.path.join(downloads_dir, "fafb_v783_princeton_synapse_table.csv.gz")
        if not os.path.exists(syn_raw):
            for f in os.listdir(downloads_dir):
                if 'synapse' in f and f.endswith('.csv.gz'):
                    syn_raw = os.path.join(downloads_dir, f)
                    break
        
        if syn_raw and os.path.exists(syn_raw):
            process_synapse_table_to_parquet(syn_raw, syn_pq)
        else:
            print("  ℹ️  Synapse table source not found (optional).")

    # --- 4. Skeletons (Optional) ---
    if os.path.exists(sk_zip_dst):
        print(f"  ✓ Found existing skeletons: {os.path.basename(sk_zip_dst)}")
    else:
        sk_raw = os.path.join(downloads_dir, "sk_lod1_783_healed.zip")
        if not os.path.exists(sk_raw):
            for f in os.listdir(downloads_dir):
                if f.endswith('.zip') and ('sk' in f or 'skeleton' in f):
                    sk_raw = os.path.join(downloads_dir, f)
                    break
                
        if sk_raw and os.path.exists(sk_raw):
            print(f"  Moving {os.path.basename(sk_raw)} -> {os.path.basename(sk_zip_dst)}...")
            shutil.move(sk_raw, sk_zip_dst)
        else:
            print("  ℹ️  Skeletons zip not found (optional).")

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
        print("="*60)
        print_download_instructions(downloads_dir)
        return False

    return True

if __name__ == "__main__":
    # Standard run for installation
    dataset_name = "flywire_FAFB_v783"
    # Determine project root (parent of src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_root, "datasets", dataset_name)
    
    print(f"Running FAFB data preparation for {dataset_name}...")
    print(f"Target directory: {dataset_dir}")
    
    ensure_flywire_data(dataset_name, dataset_dir)
