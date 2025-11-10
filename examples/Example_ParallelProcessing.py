"""
Example: Parallel Processing for Pathfinding

This example demonstrates how to use parallel processing to accelerate
pathfinding operations in large connectome datasets.

Performance Comparison:
- Sequential: Processes pairs one at a time (slower but predictable)
- Parallel: Distributes work across CPU cores (faster for large datasets)
"""

import sys
from pathlib import Path
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from coana import FindNeuronConnection
import time

def example_basic_parallel():
    """
    Basic example: Enable parallel processing with default settings
    """
    print("="*80)
    print("EXAMPLE 1: Basic Parallel Processing")
    print("="*80)
    
    # Create connectome with parallel processing enabled
    fc = FindNeuronConnection(
        token='',  # Add your token here
        dataset='hemibrain:v1.2.1',
        sourceNeurons=['PPL1-01', 'PPL1-02', 'PPL1-03', 'PPL1-04', 'PPL1-05'],
        targetNeurons=['MBON14', 'MBON11', 'MBON01', 'MBON06', 'MBON08'],
        max_interlayer=3,
        showfig=False,
        use_parallel=True  # Enable parallel processing (uses all CPU cores)
    )
    
    # Find paths from PPL1 neurons to MBON neurons
    # This creates many source-target pairs, ideal for parallel processing
    fc.InitializeNeuronInfo()
    fc.FindAllPath()
    
    print(f"\nPathfinding complete!")
    

def example_custom_cores():
    """
    Advanced example: Control number of parallel processes
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom Number of Cores")
    print("="*80)
    
    import os
    n_cores = os.cpu_count()
    print(f"System has {n_cores} CPU cores")
    print(f"Using {n_cores - 2} cores (reserving 2 for system)")
    
    # Create connectome with specific number of cores
    fc = FindNeuronConnection(
        token='',  # Add your token here
        dataset='hemibrain:v1.2.1',
        sourceNeurons=['PPL1-01', 'PPL1-02', 'PPL1-03'],
        targetNeurons=['MBON14', 'MBON11', 'MBON01'],
        max_interlayer=2,
        showfig=False,
        use_parallel=True,
        n_jobs=max(1, n_cores - 2)  # Leave 2 cores for system
    )
    
    # Find paths
    fc.InitializeNeuronInfo()
    fc.FindAllPath()
    
    print(f"\nPathfinding complete!")
    

def compare_sequential_vs_parallel():
    """
    Performance comparison: Sequential vs Parallel processing
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Performance Comparison")
    print("="*80)
    
    # Test configuration
    sources = ['PPL1-01', 'PPL1-02', 'PPL1-03', 'PPL1-04', 'PPL1-05']
    targets = ['MBON14', 'MBON11', 'MBON01', 'MBON06', 'MBON08', 'MBON12']
    max_layers = 3
    
    total_pairs = len(sources) * len(targets)
    print(f"Test: {len(sources)} sources × {len(targets)} targets = {total_pairs} pairs")
    print(f"Max path length: {max_layers + 1} edges\n")
    
    # Test 1: Sequential processing
    print("-" * 80)
    print("Test 1: Sequential Processing")
    print("-" * 80)
    
    fc_sequential = FindNeuronConnection(
        token='',  # Add your token here
        dataset='hemibrain:v1.2.1',
        sourceNeurons=sources,
        targetNeurons=targets,
        max_interlayer=max_layers,
        showfig=False,
        use_parallel=False  # Disable parallel processing
    )
    
    start = time.time()
    fc_sequential.InitializeNeuronInfo()
    fc_sequential.FindAllPath()
    sequential_time = time.time() - start
    
    print(f"\nSequential time: {sequential_time:.2f}s")
    
    # Test 2: Parallel processing
    print("\n" + "-" * 80)
    print("Test 2: Parallel Processing")
    print("-" * 80)
    
    fc_parallel = FindNeuronConnection(
        token='',  # Add your token here
        dataset='hemibrain:v1.2.1',
        sourceNeurons=sources,
        targetNeurons=targets,
        max_interlayer=max_layers,
        showfig=False,
        use_parallel=True,  # Enable parallel processing
        n_jobs=-1  # Use all CPU cores
    )
    
    start = time.time()
    fc_parallel.InitializeNeuronInfo()
    fc_parallel.FindAllPath()
    parallel_time = time.time() - start
    
    print(f"\nParallel time:   {parallel_time:.2f}s")
    
    # Compare results
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Sequential time: {sequential_time:.2f}s")
    print(f"Parallel time:   {parallel_time:.2f}s")
    
    if parallel_time < sequential_time:
        speedup = sequential_time / parallel_time
        print(f"\n✅ Parallel is {speedup:.2f}x faster!")
        print(f"   Time saved: {sequential_time - parallel_time:.2f}s ({(1-parallel_time/sequential_time)*100:.1f}%)")
    else:
        slowdown = parallel_time / sequential_time
        print(f"\n⚠️  Parallel is {slowdown:.2f}x slower")
        print(f"   Dataset may be too small for parallel processing benefits")
        print(f"   Recommendation: Use sequential processing for <100 pairs")
    

def example_small_dataset():
    """
    Example: Automatic fallback to sequential for small datasets
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Small Dataset (Automatic Sequential Fallback)")
    print("="*80)
    
    # Create connectome with parallel enabled
    fc = FindNeuronConnection(
        token='',  # Add your token here
        dataset='hemibrain:v1.2.1',
        sourceNeurons=['PPL1-01'],  # Just 1 source
        targetNeurons=['MBON14', 'MBON11'],  # Just 2 targets = 2 pairs
        max_interlayer=2,
        showfig=False,
        use_parallel=True  # Enabled, but will auto-fallback to sequential
    )
    
    # Find paths with small number of pairs (<100)
    # System will automatically use sequential processing
    fc.InitializeNeuronInfo()
    fc.FindAllPath()
    
    print(f"\n✅ System automatically used sequential processing (only 2 pairs)")
    print(f"   Parallel processing overhead not worth it for <100 pairs")
    

def example_recommended_settings():
    """
    Example: Recommended settings for different use cases
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Recommended Settings")
    print("="*80)
    
    import os
    n_cores = os.cpu_count()
    
    print(f"Your system: {n_cores} CPU cores\n")
    
    # Laptop setting (4-8 cores)
    if 4 <= n_cores <= 8:
        print("📱 Laptop detected (4-8 cores)")
        print("   Recommended: use_parallel=True, n_jobs=4")
        print("   Reason: Leave cores for system responsiveness")
        recommended_jobs = 4
    
    # Workstation setting (16+ cores)
    elif n_cores >= 16:
        print("🖥️  Workstation detected (16+ cores)")
        print("   Recommended: use_parallel=True, n_jobs=-1")
        print("   Reason: You have plenty of cores, use them all!")
        recommended_jobs = -1
    
    # Server setting (shared resource)
    else:
        print("🖧 Server/shared resource")
        print(f"   Recommended: use_parallel=True, n_jobs={max(1, n_cores // 2)}")
        print("   Reason: Be considerate of other users")
        recommended_jobs = max(1, n_cores // 2)
    
    print(f"\n✅ Recommended settings for your system:")
    print(f"   use_parallel=True")
    print(f"   n_jobs={recommended_jobs}")


if __name__ == '__main__':
    """
    Run all examples
    """
    print("\n" + "="*80)
    print("PARALLEL PROCESSING EXAMPLES")
    print("="*80)
    print("\nThese examples demonstrate parallel processing for pathfinding.")
    print("Choose an example to run:\n")
    print("  1. Basic parallel processing (default settings)")
    print("  2. Custom number of cores")
    print("  3. Performance comparison (sequential vs parallel)")
    print("  4. Small dataset (automatic fallback)")
    print("  5. Recommended settings for your system")
    print("  6. Run all examples")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == '1':
        example_basic_parallel()
    elif choice == '2':
        example_custom_cores()
    elif choice == '3':
        compare_sequential_vs_parallel()
    elif choice == '4':
        example_small_dataset()
    elif choice == '5':
        example_recommended_settings()
    elif choice == '6':
        print("\nRunning all examples...\n")
        example_basic_parallel()
        example_custom_cores()
        compare_sequential_vs_parallel()
        example_small_dataset()
        example_recommended_settings()
    else:
        print("Invalid choice. Running Example 1 (basic parallel processing)...")
        example_basic_parallel()
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)
    print("\nFor more information, see: ParallelProcessing_Documentation.md")
    print("="*80)
