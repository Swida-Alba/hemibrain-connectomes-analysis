# FlyLight Downloader Guide

The FlyLight Downloader module provides programmatic access to FlyLight imagery data from multiple sources, including the Janelia S3 bucket and HTTP CDN.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [FlyLightDownloader Class](#flylightdownloader-class)
  - [Initialization Parameters](#initialization-parameters)
  - [Collection Categories](#collection-categories)
  - [Core Methods](#core-methods)
- [File Types and Formats](#file-types-and-formats)
- [Simple Mode](#simple-mode)
- [VT Lines (HTTP CDN)](#vt-lines-http-cdn)
- [Examples](#examples)
- [Convenience Functions](#convenience-functions)
- [Troubleshooting](#troubleshooting)

---

## Overview

FlyLight is a large-scale effort to characterize Drosophila neural anatomy using light microscopy. This module provides access to:

### Data Sources

1. **S3 Bucket** (`janelia-flylight-imagery/`)
   - Gen1 GAL4/LexA lines (R-lines)
   - Gen1 MCFO stochastic labeling
   - Split-GAL4 lines (SS-lines)
   - Paper-specific collections (Lateral Horn, Descending Neurons, etc.)

2. **HTTP CDN** (`flimg.janelia.org`)
   - VT GAL4 lines
   - Projections, translations, and LSM stacks

### Key Features

- **Multi-source access**: Automatically routes to S3 or HTTP based on line name
- **Collection filtering**: Search specific collections (GAL4, Split-GAL4, MCFO)
- **Format filtering**: Download specific file types (PNG, H5J, LSM, MP4)
- **Image type filtering**: Select MIP, CDM, aligned stacks, metadata, etc.
- **Simple mode**: Reduce download volume with intelligent filename filtering
- **Parallel downloads**: Multi-threaded downloading for efficiency
- **Caching**: File list caching to speed up repeated queries

---

## Installation

```bash
# Required for HTTP access (always available)
pip install requests

# Optional: boto3 for faster S3 access
pip install boto3
```

If boto3 is not installed, the module falls back to HTTP access automatically.

---

## Quick Start

### List Available Files

```python
from src.flylight_downloader import FlyLightDownloader, list_flylight_files

# List all PNG files for a line
files = list_flylight_files('SS01015', formats='png')
print(f"Found {len(files)} files")

for f in files[:5]:
    print(f"  {f.filename} ({f.size_mb:.1f} MB)")
```

### Download Files

```python
from src.flylight_downloader import download_flylight_images

# Download MIP images for a Split-GAL4 line
paths = download_flylight_images(
    line_name='SS01015',
    output_dir='./flylight_data',
    formats='png',
    image_types='mip',
    max_files=10
)

print(f"Downloaded {len(paths)} files")
```

### Using the Class Directly

```python
from src.flylight_downloader import FlyLightDownloader

downloader = FlyLightDownloader(
    output_dir='./downloads',
    collection_category='SplitGAL4',
    formats=['png', 'jpg'],
    image_types=['mip', 'cdm'],
    simple_mode=True,
    verbose=True
)

# Download files
downloaded = downloader.download('SS01015', max_files=20)
```

---

## FlyLightDownloader Class

### Initialization Parameters

```python
@dataclass
class FlyLightDownloader:
    output_dir: str = './flylight_downloads'
    collections: Optional[List[str]] = None
    collection_category: Optional[Union[str, List[str]]] = None
    formats: Union[str, List[str]] = 'png'
    image_types: Union[str, List[str]] = 'all'
    max_workers: int = 4
    verbose: bool = True
    use_boto3: bool = True
    include_vt_lines: bool = True
    simple_mode: bool = False
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `str` | `'./flylight_downloads'` | Default directory to save downloaded files. |
| `collections` | `list` or `None` | `None` | Explicit list of collection folder names to search. Overrides `collection_category` if both specified. |
| `collection_category` | `str`, `list`, or `None` | `None` | Collection category: `'GAL4/LEXA'`, `'SplitGAL4'`, `'MCFO'`, `'RawImages'`, `'All'`. Can be a list. |
| `formats` | `str` or `list` | `'png'` | File formats to include: `'png'`, `'jpg'`, `'h5j'`, `'lsm'`, `'mp4'`, `'json'`, `'all'`. |
| `image_types` | `str` or `list` | `'all'` | Image types to filter: `'mip'`, `'cdm'`, `'aligned'`, `'metadata'`, etc. |
| `max_workers` | `int` | `4` | Number of parallel download threads. |
| `verbose` | `bool` | `True` | Print progress messages. |
| `use_boto3` | `bool` | `True` | Use boto3 for S3 access if available (faster). |
| `include_vt_lines` | `bool` | `True` | Also search VT lines via HTTP CDN. |
| `simple_mode` | `bool` | `False` | Apply filename filtering to reduce download volume. |

---

### Collection Categories

Collection categories provide an easy way to filter by data type:

| Category | Collections | Description |
|----------|-------------|-------------|
| `'GAL4/LEXA'` | Gen1 | Gen1 GAL4/LexA R-lines (CDM images) |
| `'SplitGAL4'` | Split-GAL4 Omnibus Broad, Split-GAL4 Omnibus Rescreen, Lateral Horn 2019, Descending Neurons 2018/2025, SEZ 2021, MB Paper 2014 | SS-lines from various collections |
| `'MCFO'` | Annotator Gen1 MCFO, Gen1 MCFO | MCFO stochastic labeling |
| `'RawImages'` | Collections with LSM files | Raw confocal data |
| `'All'` | All collections | Search everything |

**Using multiple categories:**
```python
downloader = FlyLightDownloader(
    collection_category=['GAL4/LEXA', 'SplitGAL4']
)
```

---

### Core Methods

#### `list_files(line_name, use_cache=True)`

List all available files for a driver line.

```python
files = downloader.list_files('SS01015')
for f in files:
    print(f"{f.collection}/{f.filename} - {f.size_mb:.1f} MB")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `line_name` | `str` | Required | Driver line name (e.g., 'R10A06', 'SS01015', 'VT037867'). |
| `use_cache` | `bool` | `True` | Use cached file list if available. |

**Returns**: `List[FlyLightFile]` objects with attributes:
- `key`: S3 key or file path
- `filename`: Base filename
- `size`: File size in bytes
- `size_mb`: File size in MB
- `collection`: Collection name
- `line_name`: Driver line name
- `source`: `'s3'` or `'http'`
- `url`: Direct download URL

---

#### `get_filtered_files(line_name, apply_simple_mode=None)`

Get files matching format, image type, and simple mode filters.

```python
# Get filtered files
files = downloader.get_filtered_files('SS01015')

# Override simple_mode setting
files = downloader.get_filtered_files('SS01015', apply_simple_mode=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `line_name` | `str` | Required | Driver line name. |
| `apply_simple_mode` | `bool` or `None` | `None` | Override class `simple_mode` setting. |

---

#### `download(line_name, output_dir=None, max_files=None, dry_run=False, flat_structure=False, files=None)`

Download files for a driver line.

```python
# Basic download
paths = downloader.download('SS01015')

# With options
paths = downloader.download(
    line_name='SS01015',
    output_dir='./my_downloads',
    max_files=10,
    flat_structure=True
)

# Dry run (list files without downloading)
downloader.download('SS01015', dry_run=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `line_name` | `str` | Required | Driver line name. |
| `output_dir` | `str` or `None` | `None` | Override output directory. |
| `max_files` | `int` or `None` | `None` | Maximum files to download. |
| `dry_run` | `bool` | `False` | List files without downloading. |
| `on_file_downloaded` | `callable` or `None` | `None` | Callback after each file: `callback(path, line_name)`. |
| `flat_structure` | `bool` | `False` | Save as `{line_name}/{filename}` instead of preserving S3 structure. |
| `files` | `list` or `None` | `None` | Pre-filtered file list (skips `get_filtered_files()`). |

**Returns**: `List[Path]` of downloaded file paths.

---

#### `download_file(file, output_dir=None, flat_structure=False)`

Download a single file.

```python
file = files[0]  # A FlyLightFile object
path = downloader.download_file(file, flat_structure=True)
```

---

#### `apply_simple_mode_filter(files, collection_name=None)`

Apply simple mode filtering to a file list.

```python
# Filter files based on collection-specific rules
filtered = downloader.apply_simple_mode_filter(files, collection_name='Split-GAL4 Omnibus Broad')
```

---

#### `get_metadata(line_name)`

Get metadata JSON files without downloading full images.

```python
metadata = downloader.get_metadata('SS01015')
for m in metadata:
    print(f"Sample: {m.get('sampleName')}")
    print(f"Region: {m.get('anatomicalArea')}")
```

---

#### `search_lines(pattern)`

Search for driver lines matching a regex pattern.

```python
# Find all SS lines starting with SS010
lines = downloader.search_lines(r'^SS010')
print(f"Found {len(lines)} matching lines")
```

---

## File Types and Formats

### File Formats

| Format | Extensions | Description |
|--------|------------|-------------|
| `'png'` | `.png` | PNG images (MIPs, CDMs) |
| `'jpg'` | `.jpg`, `.jpeg` | JPEG images (VT projections) |
| `'h5j'` | `.h5j` | H5J 3D stacks (aligned/unaligned) |
| `'lsm'` | `.lsm`, `.lsm.bz2` | Raw confocal data |
| `'mp4'` | `.mp4` | Translation videos |
| `'json'` | `.json` | Metadata files |
| `'all'` | All above | All formats |

### Image Types

| Type | Pattern | Description |
|------|---------|-------------|
| `'mip'` | `*_mip.png`, `*_signals_mip.png`, `*_multichannel_mip.png` | Maximum Intensity Projections |
| `'cdm'` | `*-CDM_*.png` | Color Depth Mask images |
| `'aligned'` | `*-aligned_stack.h5j` | Aligned 3D stacks |
| `'unaligned'` | `*-unaligned_stack.h5j` | Unaligned 3D stacks |
| `'translation'` | `*_translation.mp4`, `*.t.mp4` | Translation videos |
| `'signals'` | `*_signals*` | Signal channel images |
| `'multichannel'` | `*_multichannel*` | Multichannel images |
| `'metadata'` | `*-metadata.json` | Specimen metadata |
| `'raw'` | `*.lsm`, `*.lsm.bz2` | Raw confocal data |
| `'projection'` | `*_total.jpg` | VT line projections |
| `'all'` | `.*` | All types |

---

## Simple Mode

Simple mode (`simple_mode=True`) intelligently filters files based on collection type to reduce download volume while keeping representative images.

### Filtering Rules

| Collection Type | Filter Applied | Typical Reduction |
|-----------------|----------------|-------------------|
| **Split-GAL4** | Only `20x` AND `multichannel` files, excluding `image1`/`image2` duplicates | 241 → 13 files (~95%) |
| **VT GAL4** | Only files with `total` in filename | 44 → 4 files (~90%) |
| **Gen1 R-lines** | Keep CDM and MIP files | Keeps all (already minimal) |
| **MCFO** | Keep all files | No reduction (need full stochastic data) |

### Example

```python
# Without simple_mode: 241 files for SS01015
downloader = FlyLightDownloader(
    collection_category='SplitGAL4',
    formats=['png', 'jpg'],
    simple_mode=False
)
files = downloader.get_filtered_files('SS01015')  # 241 files

# With simple_mode: 13 files for SS01015
downloader.simple_mode = True
files = downloader.get_filtered_files('SS01015')  # 13 files
```

---

## VT Lines (HTTP CDN)

VT lines (e.g., 'VT037867') are served from a different source than R-lines and SS-lines:

### Automatic Detection

The module automatically detects VT lines and routes to the HTTP CDN:

```python
downloader = FlyLightDownloader()

# This automatically uses HTTP CDN
files = downloader.list_files('VT037867')
```

### VT File Types

| Type | Filename Pattern | Description |
|------|------------------|-------------|
| Total Projection | `*_total.jpg` | Main projection image |
| Pattern Channel | `*_ch2_total.jpg` | Pattern channel projection |
| Substacks | `*_01.jpg` to `*_10.jpg` | Substack projections |
| Translation | `*.t.mp4` | Fly-through video |

### Verifying VT Files

VT file URLs are constructed but may not all exist. Use `verify=True` to check:

```python
# List potential files (fast but may include non-existent)
files = downloader.list_vt_files('VT037867', verify=False)

# List only verified files (slower but accurate)
files = downloader.list_vt_files('VT037867', verify=True)
```

---

## Examples

### Example 1: Download Split-GAL4 MIP Images

```python
from src.flylight_downloader import FlyLightDownloader

downloader = FlyLightDownloader(
    output_dir='./splitgal4_images',
    collection_category='SplitGAL4',
    formats='png',
    image_types='mip',
    simple_mode=True,
    verbose=True
)

# Download for multiple lines
for line in ['SS01015', 'SS01540', 'SS02017']:
    paths = downloader.download(line, flat_structure=True)
    print(f"{line}: {len(paths)} files downloaded")
```

### Example 2: Get Metadata Without Downloading Images

```python
downloader = FlyLightDownloader(verbose=False)
metadata = downloader.get_metadata('SS01015')

for m in metadata:
    print(f"Sample: {m.get('sampleName', 'N/A')}")
    print(f"Anatomical Area: {m.get('anatomicalArea', 'N/A')}")
    print(f"Objective: {m.get('objective', 'N/A')}")
    print("---")
```

### Example 3: Search for Lines by Pattern

```python
downloader = FlyLightDownloader(
    collection_category='SplitGAL4',
    verbose=True
)

# Find all SS lines in the 010xx range
lines = downloader.search_lines(r'^SS010\d{2}$')
print(f"Found {len(lines)} lines: {lines[:10]}...")
```

### Example 4: Download MCFO Data

```python
downloader = FlyLightDownloader(
    output_dir='./mcfo_data',
    collection_category='MCFO',
    formats=['png', 'json'],
    image_types=['mip', 'metadata'],
    verbose=True
)

# MCFO has many files per line - use max_files
paths = downloader.download('R10A06', max_files=50)
```

### Example 5: Download Raw LSM Stacks

```python
downloader = FlyLightDownloader(
    output_dir='./raw_data',
    collection_category='RawImages',
    formats='lsm',
    image_types='raw',
    verbose=True
)

# Warning: LSM files can be very large (GB each)
paths = downloader.download('R10A06', max_files=2, dry_run=True)
```

### Example 6: VT Line Images

```python
downloader = FlyLightDownloader(
    output_dir='./vt_images',
    formats=['jpg', 'mp4'],
    simple_mode=True,  # Only total projections
    verbose=True
)

# VT lines automatically use HTTP CDN
paths = downloader.download('VT037867')
```

---

## Convenience Functions

### `download_flylight_images(...)`

Quick download function for simple use cases.

```python
from src.flylight_downloader import download_flylight_images

paths = download_flylight_images(
    line_name='SS01015',
    output_dir='./images',
    formats='png',
    image_types='mip',
    max_files=10,
    simple_mode=True,
    verbose=True
)
```

### `list_flylight_files(...)`

Quick list function without downloading.

```python
from src.flylight_downloader import list_flylight_files

files = list_flylight_files(
    line_name='SS01015',
    formats='png',
    image_types='mip',
    simple_mode=True,
    verbose=True
)

for f in files:
    print(f"{f.filename}: {f.url}")
```

---

## FlyLightFile Class

The `FlyLightFile` dataclass represents a file in the FlyLight system:

```python
@dataclass
class FlyLightFile:
    key: str           # S3 key or file path
    size: int          # Size in bytes
    last_modified: str # Modification timestamp
    collection: str    # Collection name
    line_name: str     # Driver line name
    source: str        # 's3' or 'http'
    http_url: str      # Direct URL for HTTP sources
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `filename` | `str` | Base filename from key |
| `extension` | `str` | File extension (e.g., '.png') |
| `size_mb` | `float` | Size in megabytes |
| `url` | `str` | Direct download URL |

---

## Troubleshooting

### Common Issues

**Q: boto3 not working**
```
⚠️  boto3 initialization failed
```
- The module will automatically fall back to HTTP access
- HTTP access is slightly slower but fully functional

**Q: No files found for a line**
- Check if the line name is correct (case-sensitive for some collections)
- Try without category filter to search all collections
- VT lines require `include_vt_lines=True`

**Q: Download errors for VT lines**
- Some VT file URLs may not exist; use `verify=True` in `list_vt_files()`
- VT lines may have partial data available

**Q: Files too large**
- Use `max_files` to limit downloads
- Use `simple_mode=True` to reduce file count
- Filter by `image_types` to exclude large H5J/LSM files

### Checking Available Collections

```python
from src.flylight_downloader import FlyLightDownloader

# List all categories and their collections
categories = FlyLightDownloader.list_categories()
for cat, collections in categories.items():
    print(f"\n{cat}:")
    for c in collections:
        print(f"  - {c}")
```

---

## See Also

- [NeuronBridge Integration Guide](./NeuronBridge_Guide.md) - Find EM↔LM matches
- [FlyLight Website](https://www.janelia.org/project-team/flylight) - Project homepage
- [NeuronBridge](https://neuronbridge.janelia.org/) - Web search interface
