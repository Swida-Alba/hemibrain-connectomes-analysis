"""
Cache Management Utility

This script helps you inspect and manage the Neuprint connection cache.
"""

import warnings
warnings.filterwarnings("ignore")
from coana import FindNeuronConnection
import os
import sys

def print_menu():
    print("\n" + "="*70)
    print("NEUPRINT CACHE MANAGEMENT")
    print("="*70)
    print("1. View cache information")
    print("2. View cache for specific dataset")
    print("3. Search cached neurons")
    print("4. Clear cache for specific dataset")
    print("5. Clear all cache")
    print("6. Exit")
    print("="*70)

def view_cache_info():
    """View cache information for all datasets"""
    cache_root = 'neuprint_cache'
    
    if not os.path.exists(cache_root):
        print(f'\n❌ No cache found at: {cache_root}')
        return
    
    datasets = [d for d in os.listdir(cache_root) if os.path.isdir(os.path.join(cache_root, d))]
    
    if not datasets:
        print(f'\n❌ No cached datasets found')
        return
    
    print(f'\n📁 Found cache for {len(datasets)} dataset(s):\n')
    
    for dataset in sorted(datasets):
        # Convert folder name back to dataset format
        dataset_name = dataset.replace('_', ':', 1).replace('_', '.')
        
        # Create temporary instance to read cache
        fc = FindNeuronConnection(
            dataset=dataset_name,
            sourceNeurons=[],
            targetNeurons=[],
            use_cache=True,
            token=''  # No token needed for cache inspection
        )
        
        fc.print_cache_info()

def view_dataset_cache():
    """View cache for a specific dataset"""
    print("\nAvailable datasets:")
    datasets = [
        'hemibrain:v1.2.1',
        'optic-lobe:v1.1',
        'manc:v1.0',
        'Other (enter manually)'
    ]
    
    for i, ds in enumerate(datasets, 1):
        print(f"{i}. {ds}")
    
    choice = input("\nSelect dataset (1-{}): ".format(len(datasets)))
    
    try:
        idx = int(choice) - 1
        if idx == len(datasets) - 1:
            dataset = input("Enter dataset name (e.g., hemibrain:v1.2.1): ")
        else:
            dataset = datasets[idx]
        
        fc = FindNeuronConnection(
            dataset=dataset,
            sourceNeurons=[],
            targetNeurons=[],
            use_cache=True,
            token=''
        )
        
        fc.print_cache_info()
    except (ValueError, IndexError):
        print("❌ Invalid selection")

def search_cached_neurons():
    """Search for neurons in the cache registry"""
    print("\nAvailable datasets:")
    datasets = [
        'hemibrain:v1.2.1',
        'optic-lobe:v1.1',
        'manc:v1.0',
        'Other (enter manually)'
    ]
    
    for i, ds in enumerate(datasets, 1):
        print(f"{i}. {ds}")
    
    choice = input("\nSelect dataset (1-{}): ".format(len(datasets)))
    
    try:
        idx = int(choice) - 1
        if idx == len(datasets) - 1:
            dataset = input("Enter dataset name (e.g., hemibrain:v1.2.1): ")
        else:
            dataset = datasets[idx]
        
        fc = FindNeuronConnection(
            dataset=dataset,
            sourceNeurons=[],
            targetNeurons=[],
            use_cache=True,
            token=''
        )
        
        # Check if registry exists
        registry = fc._load_neuron_registry()
        if registry.empty:
            print(f"\n❌ No neuron registry found for {dataset}")
            return
        
        print(f"\n✅ Found {len(registry)} neurons in registry")
        print("\nSearch by:")
        print("1. Neuron type (regex pattern)")
        print("2. Neuron instance (regex pattern)")
        print("3. BodyId (exact match)")
        
        search_choice = input("\nSelect search method (1-3): ")
        
        if search_choice == '1':
            pattern = input("Enter type pattern (e.g., 'L3.*' or 'PPL1'): ")
            results = fc.search_cached_neurons(pattern, 'type')
            print(f"\n🔍 Found {len(results)} neurons matching type pattern '{pattern}':")
            if not results.empty:
                print(results.to_string(index=False))
        elif search_choice == '2':
            pattern = input("Enter instance pattern (e.g., '.*_R' for right side): ")
            results = fc.search_cached_neurons(pattern, 'instance')
            print(f"\n🔍 Found {len(results)} neurons matching instance pattern '{pattern}':")
            if not results.empty:
                print(results.to_string(index=False))
        elif search_choice == '3':
            bodyid = input("Enter bodyId: ")
            results = fc.search_cached_neurons(int(bodyid), 'bodyId')
            print(f"\n🔍 Results for bodyId {bodyid}:")
            if not results.empty:
                print(results.to_string(index=False))
            else:
                print("No matching neuron found")
        else:
            print("❌ Invalid search method")
            
    except (ValueError, IndexError) as e:
        print(f"❌ Invalid selection: {e}")
    except Exception as e:
        print(f"❌ Search failed: {e}")

def clear_dataset_cache():
    """Clear cache for a specific dataset"""
    print("\nAvailable datasets:")
    cache_root = 'neuprint_cache'
    
    if not os.path.exists(cache_root):
        print(f'\n❌ No cache found')
        return
    
    # os.listdir order is unspecified and differs across filesystems (NTFS vs
    # APFS), so sort to keep the menu numbers stable on every platform.
    datasets = sorted(
        d for d in os.listdir(cache_root)
        if os.path.isdir(os.path.join(cache_root, d))
    )
    
    if not datasets:
        print(f'\n❌ No cached datasets found')
        return
    
    for i, ds in enumerate(datasets, 1):
        dataset_name = ds.replace('_', ':', 1).replace('_', '.')
        print(f"{i}. {dataset_name}")
    
    choice = input(f"\nSelect dataset to clear (1-{len(datasets)}): ")
    
    try:
        idx = int(choice) - 1
        dataset_folder = datasets[idx]
        dataset_name = dataset_folder.replace('_', ':', 1).replace('_', '.')
        
        fc = FindNeuronConnection(
            dataset=dataset_name,
            sourceNeurons=[],
            targetNeurons=[],
            use_cache=True,
            token=''
        )
        
        fc.clear_cache(confirm=True)
    except (ValueError, IndexError):
        print("❌ Invalid selection")

def clear_all_cache():
    """Clear all cache"""
    cache_root = 'neuprint_cache'
    
    if not os.path.exists(cache_root):
        print(f'\n❌ No cache found')
        return
    
    response = input(f'⚠️  Clear ALL cache? This cannot be undone! (yes/no): ')
    
    if response.lower() != 'yes':
        print('Operation cancelled.')
        return
    
    try:
        import shutil
        shutil.rmtree(cache_root)
        print(f'✅ All cache cleared: {cache_root}')
    except Exception as e:
        print(f'❌ Failed to clear cache: {e}')

def main():
    while True:
        print_menu()
        choice = input("\nSelect option (1-6): ")
        
        if choice == '1':
            view_cache_info()
        elif choice == '2':
            view_dataset_cache()
        elif choice == '3':
            search_cached_neurons()
        elif choice == '4':
            clear_dataset_cache()
        elif choice == '5':
            clear_all_cache()
        elif choice == '6':
            print("\n👋 Goodbye!")
            sys.exit(0)
        else:
            print("\n❌ Invalid option. Please select 1-6.")
        
        input("\nPress Enter to continue...")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
