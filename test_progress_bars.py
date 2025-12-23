#!/usr/bin/env python3
"""
Test script to verify progress bars during specificity calculation and neuron finding.
"""

import sys
sys.path.insert(0, 'src')

from neuronbridge_finder import NeuronBridgeFinder

def test_find_lines_progress():
    """Test progress bars for find_lines_batch."""
    print("=" * 80)
    print("Test 1: find_lines_batch Progress Bars")
    print("=" * 80)
    print()
    print("This test will:")
    print("  1. Query for a neuron type (aMe12)")
    print("  2. Calculate specificity metrics (limited to top 10 lines)")
    print("  3. Show progress bars for:")
    print("     - Line-to-neuron API fetches")
    print("     - LM image processing (nested)")
    print("     - Specificity calculation")
    print()
    print("⏱️  Note: Progress bars will show detailed hints about what's happening")
    print("=" * 80)
    print()
    
    # Initialize finder with verbose mode (required for progress bars)
    nbf = NeuronBridgeFinder(verbose=True)
    
    # Run a small test query with specificity calculation
    results = nbf.find_lines_batch(
        queries='aMe12',
        dataset='hemibrain:v1.2.1',
        match_type='cds',
        calculate_specificity=True,
        specificity_top_n=10,  # Limit to 10 lines to make test faster
        output_dir='./test_output'
    )
    
    print()
    print("=" * 80)
    print(f"✅ Test 1 complete! Found {len(results)} results")
    print("=" * 80)
    print()

def test_find_neurons_progress():
    """Test progress bars for find_neurons_batch."""
    print()
    print("=" * 80)
    print("Test 2: find_neurons_batch Progress Bars")
    print("=" * 80)
    print()
    print("This test will:")
    print("  1. Query for multiple driver lines")
    print("  2. Show progress bars for:")
    print("     - Line-by-line processing")
    print("     - LM image processing (nested)")
    print("     - Cache status indicators")
    print()
    print("⏱️  Note: Progress bars will show which lines are cached vs. fetched")
    print("=" * 80)
    print()
    
    # Initialize finder with verbose mode
    nbf = NeuronBridgeFinder(verbose=True)
    
    # Run batch search for multiple lines
    results = nbf.find_neurons_batch(
        line_names='VT037867,LH173,SS00324',
        top_n=50,
        match_type='cds',
        output_dir='./test_output'
    )
    
    print()
    print("=" * 80)
    print(f"✅ Test 2 complete! Found {len(results)} neurons")
    print("=" * 80)
    print()

def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "NeuronBridge Progress Bar Tests" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Test 1: find_lines_batch
    test_find_lines_progress()
    
    # Test 2: find_neurons_batch
    test_find_neurons_progress()
    
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 28 + "All Tests Complete!" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    print("Progress bar features tested:")
    print("  ✓ Line specificity calculation with progress")
    print("  ✓ Nested progress bars for LM image processing")
    print("  ✓ Cache status indicators (💾cached vs 🌐fetching)")
    print("  ✓ Batch neuron finding with progress")
    print("  ✓ Dynamic line names in progress description")
    print("  ✓ Time estimates and elapsed time")
    print()

if __name__ == '__main__':
    main()
