"""
Example script demonstrating the local caching feature.

This script shows how cached data speeds up repeated analyses.
Run it twice to see the difference!
"""

import sys
from pathlib import Path
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import warnings
import time
warnings.filterwarnings("ignore")
from coana import FindNeuronConnection

print("="*70)
print("CACHING FEATURE DEMONSTRATION")
print("="*70)
print("\nThis script will run the same analysis twice to show cache benefits.")
print("First run: Fetches from API and caches")
print("Second run: Loads from cache (much faster!)\n")

# Example 1: With caching enabled (default)
print("\n" + "="*70)
print("EXAMPLE 1: With Caching (use_cache=True)")
print("="*70)

start_time = time.time()

fc1 = FindNeuronConnection(
    token='',  # Add your token
    dataset='optic-lobe:v1.1',
    sourceNeurons=['L3_R'],  # Single source for faster demo
    targetNeurons=['l-LNv_R'],
    min_synapse_num=10,
    max_interlayer=1,  # Limited to 1 layer for quick demo
    use_cache=True,  # ← CACHING ENABLED (default)
    showfig=False
)

fc1.InitializeNeuronInfo()
fc1.FindAllPath(find_bodyId_path=False)  # Skip bodyId paths for speed

elapsed1 = time.time() - start_time
print(f"\n✅ Analysis completed in {elapsed1:.2f} seconds")

# Example 2: Same analysis again (should use cache)
print("\n" + "="*70)
print("RUNNING SAME ANALYSIS AGAIN...")
print("="*70)

start_time = time.time()

fc2 = FindNeuronConnection(
    token='',
    dataset='optic-lobe:v1.1',
    sourceNeurons=['L3_R'],  # SAME parameters
    targetNeurons=['l-LNv_R'],
    min_synapse_num=10,
    max_interlayer=1,
    use_cache=True,
    showfig=False
)

fc2.InitializeNeuronInfo()
fc2.FindAllPath(find_bodyId_path=False)

elapsed2 = time.time() - start_time
print(f"\n✅ Analysis completed in {elapsed2:.2f} seconds")

# Summary
print("\n" + "="*70)
print("PERFORMANCE COMPARISON")
print("="*70)
print(f"First run:  {elapsed1:.2f} seconds (fetched from API)")
print(f"Second run: {elapsed2:.2f} seconds (loaded from cache)")
if elapsed1 > elapsed2:
    speedup = elapsed1 / elapsed2
    print(f"Speed improvement: {speedup:.1f}x faster with cache!")
print("="*70)

# Example 3: Disabling cache
print("\n" + "="*70)
print("EXAMPLE 3: Without Caching (use_cache=False)")
print("="*70)
print("This will always fetch from API (slower but ensures latest data)\n")

start_time = time.time()

fc3 = FindNeuronConnection(
    token='',
    dataset='optic-lobe:v1.1',
    sourceNeurons=['L3_R'],
    targetNeurons=['l-LNv_R'],
    min_synapse_num=10,
    max_interlayer=1,
    use_cache=False,  # ← CACHING DISABLED
    showfig=False
)

fc3.InitializeNeuronInfo()
fc3.FindAllPath(find_bodyId_path=False)

elapsed3 = time.time() - start_time
print(f"\n✅ Analysis completed in {elapsed3:.2f} seconds")
print("(Always fetches from API, no caching)")

# Cache management tips
print("\n" + "="*70)
print("CACHE MANAGEMENT TIPS")
print("="*70)
print("""
Cache location: cache/{dataset}/connections/

To clear cache for optic-lobe:v1.1:
  rm -rf cache/optic-lobe_v1_1/

To clear all cache:
  rm -rf cache/

To check cache size:
  du -sh cache/

Cache files are named:
  conn_{N}neurons_minw{weight}_{hash}.parquet

See docs/CacheSystem_Guide.md for complete guide.
""")
print("="*70)
