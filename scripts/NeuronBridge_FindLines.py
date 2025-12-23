#!/usr/bin/env python3
"""
NeuronBridge_FindLines.py - Find LM driver lines matching EM neurons

This script searches NeuronBridge to find GAL4 driver lines that express
neurons matching the morphology of a given EM body ID or neuron type.

Usage:
    Edit the parameters in the script and run directly:
    python NeuronBridge_FindLines.py

Key Features:
    - Search by body ID (integer), type name, or instance name
    - Multiple queries supported as comma-separated string or list
    - Regex patterns supported for type/instance matching
    - Match algorithms: 'cds' (Color Depth Search), 'pppm', or 'both'
    - Multi-dataset search: when dataset=None, searches all available datasets
    - Optional image download from NeuronBridge or FlyLight
    - Results saved to CSV with driver line information

Output Files:
    - {query}_lines.csv: Matched driver lines with scores
    - all_lines.csv: Combined results from all queries
    - images/: Downloaded images (if enabled)

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
    
    # Neuron query - can be:
    #   - Body ID (integer): 636798093
    #   - Type name: 'aMe12'
    #   - Instance name: 'aMe12_R'
    #   - Regex pattern: 'MBON.*'
    #   - Multiple as string: 'aMe12,MBON01,636798093'
    #   - Multiple as list: ['aMe12', 'MBON01', 636798093]
    query = 'aMe12'
    
    # Dataset to search in (for type/instance lookups)
    # Options: 'hemibrain:v1.2.1', 'male-cns:v0.9', 'flywire_FAFB_v783', etc.
    # Set to None to search ALL available datasets (recommended for broad search)
    dataset = ['male-cns:v0.9', 'hemibrain:v1.2.1', 'flywire_FAFB_v783']
    
    # Match algorithm: 'cds' (Color Depth Search), 'pppm', or 'both'
    match_type = 'cds'
    
    # Maximum LM images to process per line (10 by default, -1 for all)
    # Images are pre-filtered by match_type availability before limiting
    max_images_per_line = 10
    
    # Output directory (set to None for stdout only)
    output_dir = '/Users/apple/Local/connection_data/neuronbridge'
    
    # Path to datasets folder for neuron lookup (None = auto-detect)
    datasets_path = None
    
    # Enable result caching
    use_cache = True
    
    # Verbose output
    verbose = True
    
    # Region of interest
    # -------------------------------------------------------------------
    # 'Brain', 'VNC', or 'All' for both (default is 'All')
    region = 'Brain'
    
    # ==========================================================================
    # IMAGE DOWNLOAD OPTIONS (Optional)
    # ==========================================================================
    
    # Download images for matched lines
    # Options (case-insensitive):
    #   - 'neuronbridge': Download CDM images from NeuronBridge
    #   - 'flylight': Download images from FlyLight (S3/HTTP CDN)
    #   - 'both': Download from both sources
    #   - None/False: No image download (default)
    download_images = 'flylight'
    
    # Download images only for top N lines (by aggregate score/rank)
    # Set to None to download for all lines
    download_top_n_img = 5
    
    # File formats to download
    # For neuronbridge: 'png', 'jpg'
    # For flylight: 'png', 'jpg', 'h5j', 'mp4', 'all'
    image_formats = ['png','jpg']
    
    # Image types to download
    # For neuronbridge: 'cdm', 'mip', 'all'
    # For flylight: 'mip', 'cdm', 'aligned', 'translation', 'all'
    image_types = 'mip'
    
    # Maximum images to download per line (None = no limit)
    max_images_per_line = 10
    
    # FlyLight collection category (only used when download_images='flylight' or 'both')
    # Options (case-insensitive): 'GAL4/LEXA', 'SplitGAL4', 'MCFO', 'RawImages', 'All'
    # Can also be a list: ['GAL4/LEXA', 'SplitGAL4']
    # None = search all collections
    flylight_category = ['GAL4/LEXA', 'SplitGAL4']
    
    # Organize images by anatomical region (Brain/VNC folders)
    # If True, images will be organized into:
    #   images/Brain/LineName/...
    #   images/VNC/LineName/...
    # If False, images will be organized by collection (Gen1/, Split-GAL4 Omnibus Broad/, etc.)
    organize_by_region = False
    
    # Simple mode for FlyLight downloads - reduces download volume by filtering filenames
    # If True:
    #   - Split-GAL4 collections: only download files with '20x' AND 'multichannel' in filename
    #   - GAL4/LexA collections: only download files with 'total' in filename
    # If False: download all matching files (default)
    simple_mode = True
    
    # Separate GAL4/LexA from Split-GAL4 lines (affects download and results)
    # If True:
    #   - Results will include a 'line_type' column ('gal4_lexa
    #     or 'split_gal4')
    #   - download_top_n_img applies separately to each category
    # If False: all lines treated together (default)
    separate_splitgal4 = True
    # ==========================================================================
    # SPECIFICITY OPTIONS
    # ==========================================================================
    
    # Calculate line specificity metrics (how specific each line is to queried types)
    calculate_specificity = True
    
    # Limit for specificity calculation:
    #  - Maximum number of lines to calculate specificity for (to limit API calls)
    #  - Number of top neuron matches to consider when analyzing each line
    # Set to None to calculate for all lines (can be very slow)
    specificity_top_n = 10
    
    # ==========================================================================
    # EXECUTION - No need to edit below this line
    # ==========================================================================
    
    # Initialize finder
    finder = NeuronBridgeFinder(
        datasets_path=datasets_path,
        use_cache=use_cache,
        verbose=verbose,
        separate_splitgal4=separate_splitgal4,
        match_type=match_type,                    # Set default match algorithm
        region=region,                            # Set region filter
        max_images_per_line=max_images_per_line   # Limit images per line
    )
    
    # Run batch search (with optional image download)
    # match_type parameter is now optional - uses instance setting by default
    results = finder.find_lines_batch(
        queries=query,
        dataset=dataset,
        # match_type=match_type,  # Can override here if needed
        output_dir=output_dir,
        download_images=download_images,
        download_top_n_img=download_top_n_img,
        image_formats=image_formats,
        image_types=image_types,
        max_images_per_line=max_images_per_line,
        flylight_category=flylight_category,
        organize_by_region=organize_by_region,
        simple_mode=simple_mode,
        calculate_specificity=calculate_specificity,
        specificity_top_n=specificity_top_n
    )
    
    # Display summary
    if not results.empty:
        print(f"\n{'='*60}")
        print(f"📊 Summary")
        print('='*60)
        print(f"   Total lines: {len(results)}")
        
        if 'source_query' in results.columns:
            print(f"\n   By source query:")
            for q, count in results['source_query'].value_counts().items():
                print(f"      {q}: {count}")
        
        if 'source_dataset' in results.columns:
            print(f"\n   By source dataset:")
            for ds, count in results['source_dataset'].value_counts().items():
                print(f"      {ds}: {count}")
        
        # Show top results
        print(f"\n   Top matches:")
        display_cols = ['line', 'library', 'score', 'source_query']
        display_cols = [c for c in display_cols if c in results.columns]
        if display_cols:
            print(results[display_cols].head(10).to_string(index=False))
        else:
            print(results.head(10).to_string(index=False))
    
    print("\n✅ Done!")
