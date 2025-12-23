#!/usr/bin/env python3
"""
NeuronBridge_FindNeuron.py - Find EM neurons matching LM driver lines

This script searches NeuronBridge to find EM neurons that match the morphology
of a given GAL4 driver line.

Usage:
    Edit the parameters in the script and run directly:
    python NeuronBridge_FindNeuron.py

Key Features:
    - Search by driver line name (e.g., 'LH173', 'VT037867', 'SS00731')
    - Multiple lines supported as comma-separated string or list
    - Match algorithms: 'cds' (Color Depth Search), 'pppm', or 'both'
    - Automatic dataset detection and neuron enrichment
    - Results saved to CSV with dataset, type, and instance information

Output Files:
    - {line_name}_neurons.csv: Matched EM neurons with scores and metadata
    - all_neurons.csv: Aggregated results from all searches

Author: Hemibrain Connectomes Analysis Project
"""

import sys
from pathlib import Path

# Add repo src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from neuronbridge_finder import NeuronBridgeFinder

if __name__ == "__main__":
    # ==========================================================================
    # CONFIGURATION - Edit these parameters
    # ==========================================================================
    
    # Driver line(s) to search - can be:
    #   - Single line: 'LH173'
    #   - Multiple lines as string: 'LH173,VT037867,SS00731'
    #   - Multiple lines as list: ['LH173', 'VT037867', 'SS00731']
    lines = 'VT037867'
    
    # Match algorithm: 'cds' (Color Depth Search), 'pppm', or 'both'
    # Can be set here at class level, or passed to find_neurons_batch()
    match_type = 'both'
    
    # Anatomical region filter: 'Brain', 'VNC', or 'All'
    # Filters out images from non-matching regions
    region = 'Brain'
    
    # Maximum LM images to process per line (10 by default, -1 for all)
    # Images are pre-filtered by match_type availability before limiting
    max_images_per_line = 10
    
    # Maximum number of matches to return per line (-1 for all matches)
    top_n = -1
    
    # Output directory (set to None for stdout only)
    output_dir = '/Users/apple/Local/connection_data/neuronbridge'
    
    # Path to datasets folder for neuron enrichment (None = auto-detect)
    datasets_path = None
    
    # Enable result caching
    use_cache = True
    
    # Verbose output
    verbose = True
    
    
    # ==========================================================================
    # EXECUTION - No need to edit below this line
    # ==========================================================================
    
    # Initialize finder
    finder = NeuronBridgeFinder(
        datasets_path=datasets_path,
        use_cache=use_cache,
        verbose=verbose,
        match_type=match_type,           # Set default match algorithm
        region=region,                    # Set region filter
        max_images_per_line=max_images_per_line  # Limit images per line
    )
    
    # Run batch search (handles parsing, processing, and saving automatically)
    # match_type parameter is now optional - uses instance setting by default
    results = finder.find_neurons_batch(
        line_names=lines,
        top_n=top_n,
        # match_type=match_type,  # Can override here if needed
        output_dir=output_dir
    )
    
    # Display summary
    if not results.empty:
        print(f"\n{'='*60}")
        print(f"📊 Summary")
        print('='*60)
        print(f"   Total neurons: {len(results)}")
        
        if 'dataset' in results.columns:
            print(f"\n   By dataset:")
            for ds, count in results['dataset'].value_counts().items():
                print(f"      {ds}: {count}")
        
        if 'source_line' in results.columns:
            print(f"\n   By source line:")
            for line, count in results['source_line'].value_counts().items():
                print(f"      {line}: {count}")
        
        # Show top results
        print(f"\n   Top matches:")
        display_cols = ['bodyId', 'dataset', 'type', 'instance', 'score', 'source_line']
        display_cols = [c for c in display_cols if c in results.columns]
        print(results[display_cols].head(10).to_string(index=False))
    
    print("\n✅ Done!")
