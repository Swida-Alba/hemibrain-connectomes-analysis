#!/usr/bin/env python3
"""
FlyLight_downloader.py - Download FlyLight imagery data by driver line name

This script downloads images and data from multiple FlyLight sources:
    1. S3 bucket (janelia-flylight-imagery): Gen1 MCFO, Split-GAL4 lines
    2. HTTP CDN (flimg.janelia.org): VT GAL4 lines

The script automatically detects the line type and uses the appropriate source.

Usage:
    Edit the parameters in the script and run directly:
    python FlyLight_downloader.py

Key Features:
    - Download by driver line name (e.g., 'R10A06', 'VT037867', 'SS00731')
    - Automatic source detection (S3 for R-lines, HTTP for VT lines)
    - Filter by file format: 'png', 'jpg', 'h5j', 'lsm', 'mp4', 'json', or 'all'
    - Filter by image type: 'mip', 'cdm', 'aligned', 'metadata', etc.
    - List files without downloading (dry_run mode)
    - Search for lines matching a pattern
    - Parallel downloads for speed

Line Types & Sources:
    - R lines (R10A06, R23A12): S3 bucket - Gen1 MCFO collections
    - SS lines (SS00731): S3 bucket - Split-GAL4 collections
    - VT lines (VT037867): HTTP CDN - flimg.janelia.org (auto-detected)

File Formats:
    - png: PNG images (MIP, CDM, aligned stack previews)
    - jpg: JPEG images (VT line projections)
    - h5j: H5J 3D stacks (aligned and unaligned)
    - lsm: Raw confocal data (compressed .lsm.bz2)
    - mp4: Translation videos
    - json: Metadata files

Image Types:
    - mip: Maximum Intensity Projections
    - cdm: Color Depth Masks (for NeuronBridge matching)
    - aligned: Aligned brain stacks
    - unaligned: Unaligned stacks
    - translation: Translation videos
    - signals: Signal channel images
    - multichannel: Multichannel images
    - metadata: Specimen metadata JSON files
    - raw: Raw confocal data
    - all: All image types

Collections (S3 only):
    - Annotator Gen1 MCFO (R-lines like R10A06)
    - Gen1 MCFO (more R-lines)
    - Gen1 (CDM images for R-lines)
    - Split-GAL4 Omnibus Broad/Rescreen (SS lines)
    - Various paper-specific collections

Author: Hemibrain Connectomes Analysis Project
"""

import sys
from pathlib import Path

# Add repo src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from flylight_downloader import FlyLightDownloader

if __name__ == "__main__":
    # ==========================================================================
    # CONFIGURATION - Edit these parameters
    # ==========================================================================
    
    # Driver line name to download
    # Examples: 
    #   'R10A06'   - Gen1 line (from S3 bucket)
    #   'VT037867' - VT line (from HTTP CDN - automatically detected)
    #   'SS00731'  - Split-GAL4 line (from S3 bucket)
    line_name = 'SS01015'
    
    # File formats to download
    # Options: 'png', 'jpg', 'h5j', 'lsm', 'mp4', 'json', 'all'
    # Note: VT lines primarily have 'jpg' and 'mp4' files
    # Can be a single format or list: ['png', 'jpg', 'json']
    formats = ['jpg', 'png']
    
    # Image types to download
    # Options: 'mip', 'cdm', 'aligned', 'unaligned', 'translation', 
    #          'signals', 'multichannel', 'metadata', 'raw', 'all'
    # Can be a single type or list: ['mip', 'cdm']
    image_types = 'all'
    
    # Collections to search (only for S3 lines, not VT lines)
    # Main options: 'Annotator Gen1 MCFO', 'Gen1 MCFO', 'Gen1', 'Split-GAL4 Omnibus Broad'
    # VT lines are auto-detected and fetched from HTTP CDN (this setting is ignored for VT lines)
    collections = None  # None = search all collections (recommended)
    
    # Collection category (case-insensitive, alternative to collections)
    # Options: 'GAL4/LEXA', 'SplitGAL4', 'MCFO', 'RawImages', 'All'
    # Can be a single category or list: ['GAL4/LEXA', 'SplitGAL4']
    collection_category = None  # None = search all collections
    
    # Output directory for downloaded files
    output_dir = '../local_data/connection_data/flylight'
    
    # Maximum number of files to download (None = no limit)
    max_files = 30
    
    # Dry run mode - list files without downloading
    dry_run = False
    
    # Number of parallel download threads
    max_workers = 6
    
    # Use boto3 for downloads (faster if installed)
    use_boto3 = True
    
    # Verbose output
    verbose = True
    
    # ==========================================================================
    # EXECUTION - No need to edit below this line
    # ==========================================================================
    
    print("="*70)
    print("🪰 FlyLight Image Downloader")
    print("="*70)
    print(f"\n📋 Configuration:")
    print(f"   Line name: {line_name}")
    print(f"   Formats: {formats}")
    print(f"   Image types: {image_types}")
    print(f"   Collections: {collections or 'all'}")
    print(f"   Output: {output_dir}")
    print(f"   Max files: {max_files or 'unlimited'}")
    print(f"   Dry run: {dry_run}")
    print()
    
    # Initialize downloader
    downloader = FlyLightDownloader(
        output_dir=output_dir,
        collections=collections,
        collection_category=collection_category,
        formats=formats,
        image_types=image_types,
        max_workers=max_workers,
        use_boto3=use_boto3,
        verbose=verbose,
        simple_mode=True, # Enable simple mode for reduced download volume
    )
    
    # Download files
    downloaded_files = downloader.download(
        line_name=line_name,
        max_files=max_files,
        dry_run=dry_run
    )
    
    # Summary
    if downloaded_files:
        print(f"\n{'='*70}")
        print(f"📁 Downloaded {len(downloaded_files)} files to:")
        print(f"   {output_dir}")
        print("\n   Files:")
        for f in downloaded_files[:10]:
            print(f"   - {f.name}")
        if len(downloaded_files) > 10:
            print(f"   ... and {len(downloaded_files) - 10} more")
    elif not dry_run:
        print(f"\n⚠️  No files downloaded for '{line_name}'")
        print("   Try different filters or check the line name.")
    
    print("\n✅ Done!")
    
    # ==========================================================================
    # ADDITIONAL USAGE EXAMPLES (uncomment to use)
    # ==========================================================================
    
    # # Example 1: Search for lines matching a pattern
    # matching_lines = downloader.search_lines('R10A.*')
    # print(f"Found {len(matching_lines)} lines matching 'R10A.*'")
    # for line in matching_lines[:20]:
    #     print(f"  - {line}")
    
    # # Example 2: Get metadata without downloading images
    # metadata = downloader.get_metadata('R10A06')
    # print(f"Found {len(metadata)} metadata records")
    # if metadata:
    #     print(f"First record: {metadata[0]}")
    
    # # Example 3: List all files without downloading
    # files = downloader.list_files('R10A06')
    # print(f"Found {len(files)} total files")
    # for f in files[:20]:
    #     print(f"  - {f.filename} ({f.size_mb:.1f} MB)")
    
    # # Example 4: Download aligned stacks (H5J format)
    # downloader.formats = ['h5j']
    # downloader.image_types = ['aligned']
    # downloaded = downloader.download('R10A06', max_files=2)
    
    # # Example 5: Download all CDM masks for NeuronBridge
    # downloader.formats = ['png']
    # downloader.image_types = ['cdm']
    # downloaded = downloader.download('SS00731')
