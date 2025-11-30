import os
import shutil
import zipfile
import pandas as pd
import sys
import glob
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import FAFB_file_converter
from coana import VisualizeSkeleton

def setup_test_env():
    dataset_name = "test_flywire_skeleton_loading"
    project_root = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.join(project_root, "datasets", dataset_name)
    
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)
    
    print(f"Created test dataset dir: {dataset_dir}")
    
    # 1. Find real skeletons
    zip_paths = glob.glob(os.path.join(project_root, "datasets/**/*skeletons.zip"), recursive=True) + \
                glob.glob(os.path.join(project_root, "datasets/**/sk_lod1_783_healed.zip"), recursive=True) + \
                glob.glob(os.path.join(project_root, "datasets/**/downloads/sk_lod1_783_healed.zip"), recursive=True)
    
    if not zip_paths:
        print("No source skeleton zip found!")
        return None, None
        
    src_zip = zip_paths[0]
    print(f"Using source zip: {src_zip}")
    
    # 2. Extract 5 random SWCs
    extracted_ids = []
    dst_zip = os.path.join(dataset_dir, f"{dataset_name}_skeletons.zip")
    
    with zipfile.ZipFile(src_zip, 'r') as z_in:
        swc_files = [f for f in z_in.namelist() if f.endswith('.swc')]
        selected_files = swc_files[:5]
        
        with zipfile.ZipFile(dst_zip, 'w') as z_out:
            for f in selected_files:
                content = z_in.read(f)
                z_out.writestr(f, content)
                extracted_ids.append(f.split('.')[0])
                
    print(f"Created test zip with {len(extracted_ids)} skeletons: {extracted_ids}")
    
    # 3. Create dummy neuron DF
    neuron_df = pd.DataFrame({
        'bodyId': extracted_ids,
        'type': ['test_type'] * len(extracted_ids),
        'instance': ['test_instance'] * len(extracted_ids),
        'pre': [0] * len(extracted_ids),
        'post': [0] * len(extracted_ids)
    })
    neuron_pq = os.path.join(dataset_dir, f"{dataset_name}_allneurons_neuron_df.parquet")
    neuron_df.to_parquet(neuron_pq, index=False)
    
    # Remove CSV if exists (cleanup from previous runs)
    neuron_csv = os.path.join(dataset_dir, f"{dataset_name}_allneurons_neuron_df.csv")
    if os.path.exists(neuron_csv):
        os.remove(neuron_csv)
    
    # 4. Create dummy connections DF (empty)
    conn_df = pd.DataFrame(columns=['bodyId_pre', 'bodyId_post', 'weight', 'roi'])
    conn_pq = os.path.join(dataset_dir, f"{dataset_name}_merged_connections.parquet")
    conn_df.to_parquet(conn_pq, index=False)
    
    # Remove CSV if exists
    conn_csv = os.path.join(dataset_dir, f"{dataset_name}_merged_connections.csv")
    if os.path.exists(conn_csv):
        os.remove(conn_csv)
    
    return dataset_name, extracted_ids

def run_test():
    dataset_name, bodyIds = setup_test_env()
    if not dataset_name:
        return
        
    project_root = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.join(project_root, "datasets", dataset_name)
    
    # 1. Convert to Parquet
    print("\n--- Testing Conversion ---")
    zip_path = os.path.join(dataset_dir, f"{dataset_name}_skeletons.zip")
    pq_path = os.path.join(dataset_dir, f"{dataset_name}_skeletons.parquet")
    
    start_time = time.time()
    FAFB_file_converter.process_skeletons_to_parquet(zip_path, pq_path)
    print(f"Conversion took {time.time() - start_time:.2f}s")
    
    if os.path.exists(pq_path):
        print("Conversion successful!")
        # Inspect parquet columns
        df_check = pd.read_parquet(pq_path)
        print(f"Parquet columns: {df_check.columns.tolist()}")
        print(f"First row: {df_check.iloc[0].to_dict()}")
    else:
        print("Conversion failed!")
        return

    # 2. Test Loading with VisualizeSkeleton
    print("\n--- Testing VisualizeSkeleton Loading ---")
    
    # Initialize VisualizeSkeleton
    # We use brain_mesh='none' to avoid transform issues
    vs = VisualizeSkeleton(
        dataset=dataset_name,
        neuron_layers=[bodyIds],
        brain_mesh='none',
        skeleton_mode='tube',
        show_connectors=False,
        ignore_synapses=True
    )
    
    print("Calling plot_skeleton()...")
    start_time = time.time()
    vs.plot_skeleton()
    print(f"plot_skeleton took {time.time() - start_time:.2f}s")
    
    print("\nTest Complete.")

if __name__ == "__main__":
    run_test()
