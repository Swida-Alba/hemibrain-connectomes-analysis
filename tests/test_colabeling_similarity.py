#!/usr/bin/env python
"""
Test script for the new co-labeling similarity methods.

This script tests the three similarity methods:
1. Binary Jaccard (original)
2. Weighted Jaccard (using match scores)
3. Rank Correlation (Spearman correlation on overlapping types)
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neuronbridge_finder import NeuronBridgeFinder

def main():
    """Test the new similarity methods."""
    
    print("=" * 80)
    print("Testing Co-Labeling Similarity Methods")
    print("=" * 80)
    
    # Create finder instance
    nbf = NeuronBridgeFinder(verbose=True)
    
    # Test with a small set of lines
    test_lines = ['LH173', 'LH174', 'LH175']
    
    print(f"\n🔍 Testing with {len(test_lines)} lines: {', '.join(test_lines)}")
    
    # Test each similarity method
    methods = ['jaccard', 'weighted_jaccard', 'rank_correlation']
    
    for method in methods:
        print(f"\n{'=' * 80}")
        print(f"Method: {method}")
        print(f"{'=' * 80}")
        
        try:
            matrix, type_sets = nbf._build_colabeling_matrix(
                lines=test_lines,
                match_type='cds',
                top_n=50,
                similarity_method=method
            )
            
            print(f"\n✅ Successfully computed {method} similarity matrix")
            print(f"\nMatrix shape: {matrix.shape}")
            print(f"\nSimilarity scores:")
            print(matrix.to_string())
            
            # Print some statistics
            print(f"\n📊 Statistics:")
            # Get upper triangle (excluding diagonal)
            import numpy as np
            n = len(test_lines)
            upper_triangle = []
            for i in range(n):
                for j in range(i+1, n):
                    upper_triangle.append(matrix.iloc[i, j])
            
            if upper_triangle:
                print(f"   Mean similarity: {np.mean(upper_triangle):.4f}")
                print(f"   Max similarity:  {np.max(upper_triangle):.4f}")
                print(f"   Min similarity:  {np.min(upper_triangle):.4f}")
            
        except Exception as e:
            print(f"❌ Error with {method}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 80}")
    print("✅ Testing complete!")
    print(f"{'=' * 80}")

if __name__ == '__main__':
    main()
