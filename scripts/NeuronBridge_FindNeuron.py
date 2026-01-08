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
    - 3D skeleton visualization with automatic mesh simplification
    - PDF summary generation with natural sorting (r1, r2, ..., r10)
    - Results saved to CSV with dataset, type, and instance information

Output Files:
    - {line_name}_neurons.csv: Matched EM neurons with scores and metadata
    - all_neurons.csv: Aggregated results from all searches
    - plot3d_{dataset}/: 3D visualization folder (if visualize_top_n > 0)
      - {dataset}.html: Interactive 3D skeleton visualization
      - exported_views/: PNG exports (front, back, top, bottom, left, right)
      - individual_profiles/: Per-neuron PNG profiles + PDF summary
      - parameters.txt: Visualization settings record

Visualization Features:
    - Neurons grouped by type with r{rank}_{type}_x{N} legend labels
    - Automatic mesh simplification (95% reduction) for large visualizations
    - Natural sorting in PDF: r1, r2, ..., r9, r10 (not r1, r10, r11...)
    - Line or tube skeleton modes based on neuron count (>50 uses line mode)

Author: Hemibrain Connectomes Analysis Project
"""

import sys
import time
from pathlib import Path

# Add repo src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from neuronbridge_finder import NeuronBridgeFinder

if __name__ == "__main__":
    t0 = time.time()
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
    match_type = 'cds'
    
    # Anatomical region filter: 'Brain', 'VNC', or 'All'
    # Filters out images from non-matching regions
    region = 'Brain'
    
    # Maximum LM images to process per line for API calls (10 by default, -1 for all)
    # Images are pre-filtered by match_type availability before limiting
    max_api_images_per_line = -1
    
    # Maximum number of matches to return per line (-1 for all matches)
    top_n = -1
    
    # Output directory (set to None for stdout only)
    output_dir = '../local_data/neuronbridge_finding'
    
    # Path to datasets folder for neuron enrichment (None = auto-detect)
    datasets_path = None
    
    # Enable result caching
    use_cache = True
    
    # Verbose output
    verbose = True
    
    # ==========================================================================
    # VISUALIZATION - 3D skeleton visualization options
    # ==========================================================================
    
    # Visualize top N types/bodyIds per dataset using 3D skeleton (0 = disabled)
    # This creates interactive HTML visualizations showing the top N neuron types or bodyIds
    # Output folder: plot3d_{dataset}/ (no timestamp, overwrites previous)
    visualize_top_n = 20
    
    # How to organize visualization: 'type' or 'bodyId'
    # - 'type': Group neurons by type (legend_mode='layer', shows combined morphology)
    # - 'bodyId': Show individual neurons grouped by type (legend_mode='single')
    #            Legend labels: r{rank}_{type}_x{N} where N is neuron count per type
    visualize_by = 'type'
    
    # Create separate visualization per dataset (True) or combined (False)
    visualize_per_dataset = True
    
    # Generate individual profile PNGs for each neuron type/bodyId
    # Creates a subfolder 'individual_profiles' with one PNG per type + summary files
    # Options:
    #   - 'pdf' or ['pdf']: Generate PDF only
    #   - 'pptx' or ['pptx']: Generate PPTX only
    #   - ['pdf', 'pptx']: Generate both formats
    #   - False or None: Disable generation
    # Uses natural sorting: r1, r2, ..., r9, r10 (not r1, r10, r11...)
    # Only uses front view
    generate_individual_profiles = ['pdf', 'pptx']
    
    # Layout for individual profiles (columns, rows)
    # Default: (3, 2) = 3 columns x 2 rows = 6 images per page/slide
    pdf_images_per_page = (3, 2)
    
    
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
        max_api_images_per_line=max_api_images_per_line  # Limit images per line for API
    )
    
    # Run batch search (handles parsing, processing, and saving automatically)
    # match_type parameter is now optional - uses instance setting by default
    results = finder.find_neurons_batch(
        line_names=lines,
        top_n=top_n,
        # match_type=match_type,  # Can override here if needed
        output_dir=output_dir,
        visualize_top_n=visualize_top_n,
        visualize_by=visualize_by,
        visualize_per_dataset=visualize_per_dataset,
        generate_individual_profiles=generate_individual_profiles,
        pdf_images_per_page=pdf_images_per_page,
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
    t1 = time.time()
    elapsed = t1 - t0
    print(f"⏱️ Elapsed time: {elapsed/60:.2f} minutes\n")
