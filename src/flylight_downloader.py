"""
FlyLight Downloader Module

This module provides programmatic access to FlyLight imagery data from multiple sources:
    1. S3 bucket (janelia-flylight-imagery): Gen1 MCFO, Split-GAL4 lines (R-lines)
    2. HTTP CDN (flimg.janelia.org): VT GAL4 lines (standard expression patterns)
    3. Gen1 MCFO CDN (gen1mcfo.janelia.org): VT MCFO lines (sparse labeling clones)

Key Features:
    - Search by driver line name (e.g., 'R10A06', 'VT037867', 'SS00731')
    - Automatic source detection (S3 for R-lines, HTTP CDN for VT lines)
    - Filter by file format: png, jpg, h5j, lsm, mp4, json, or all
    - Filter by image type: mip, cdm, aligned, unaligned, translation, signals, etc.
    - Filter by collection: Gen1 GAL4, Gen1 MCFO, Split-GAL4, VT GAL4, etc.
    - List available files without downloading
    - Automatic directory structure preservation

Data Sources:

1. S3 Bucket (janelia-flylight-imagery/) - R-lines:
    ├── Annotator Gen1 MCFO/
    │   ├── R10A06/
    │   │   ├── R10A06-...-metadata.json
    │   │   ├── R10A06-...-aligned_stack.h5j
    │   │   ├── R10A06-...-CDM_1.png
    │   │   └── ...
    │   └── ...
    ├── Split-GAL4 Omnibus/
    │   └── ...
    └── ...

2. HTTP CDN (flimg.janelia.org) - VT GAL4 Lines:
    ├── projections/  - JPEG projection images
    ├── translations/ - MP4 fly-through movies
    └── (LSM stacks via CGI download)

3. Gen1 MCFO for VT Lines (discovered via gen1mcfo.janelia.org):
    VT lines with MCFO data are hosted in the S3 bucket under 'Gen1 MCFO'
    and 'Annotator Gen1 MCFO' collections, but are not easily discoverable
    via direct S3 listing. This module parses the gen1mcfo.janelia.org viewer
    page to find all available S3-hosted MCFO images for VT lines.

**VT Line Search Order:**
When searching for VT line images, this module searches in priority order:
    1. VT GAL4 (flimg.janelia.org) - Standard expression patterns
    2. VT MCFO (S3 via gen1mcfo.janelia.org) - Sparse labeling clones (included automatically)

This ensures that lines like VT000770, which have no GAL4 images but have
MCFO data, will still return results.

File Types:
    - *-metadata.json: Specimen metadata
    - *-aligned_stack.h5j: Aligned 3D stack (H5J format)
    - *-unaligned_stack.h5j: Unaligned 3D stack
    - *-CDM_*.png: Color Depth Mask images (for NeuronBridge)
    - *-mip.png: Maximum Intensity Projection
    - *-signals_mip.png: Signal channel MIP
    - *-multichannel_mip.png: Multichannel MIP
    - *-translation.mp4: Translation video
    - *.lsm.bz2: Raw confocal data (compressed)
    - VT lines: *_total.jpg, *_ch2_total.jpg (projections), *.t.mp4 (translations)

Author: Hemibrain Connectomes Analysis Project
"""

import os
import re
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


# S3 bucket configuration
FLYLIGHT_BUCKET = 'janelia-flylight-imagery'
FLYLIGHT_REGION = 'us-east-1'

# VT line HTTP CDN configuration
VT_CDN_BASE = 'https://flimg.janelia.org/flylight-image/external-data/adult/secdata'
VT_VIEW_CGI = 'https://flweb.janelia.org/cgi-bin/view_flew_imagery.cgi'
VT_DOWNLOAD_CGI = 'https://flweb.janelia.org/cgi-bin/download.cgi'

# Gen1 MCFO configuration (for VT lines MCFO data)
# VT lines have MCFO data at gen1mcfo.janelia.org, not in the S3 bucket
GEN1_MCFO_VIEW_CGI = 'https://gen1mcfo.janelia.org/cgi-bin/view_gen1mcfo_imagery.cgi'
GEN1_MCFO_SEARCH_CGI = 'https://gen1mcfo.janelia.org/cgi-bin/gen1mcfo.cgi'
GEN1_MCFO_CDN_BASE = 'https://gen1mcfo.janelia.org/imagery'

# Known collections in the S3 bucket (top-level folders)
# Note: VT GAL4 lines are NOT in this S3 bucket - they use the HTTP CDN above
FLYLIGHT_COLLECTIONS = [
    'Annotator Gen1 MCFO',  # R-lines with MCFO stochastic labeling
    'Gen1 MCFO',            # More R-lines with MCFO
    'Gen1',                 # Gen1 CDM images (R-lines)
    'Split-GAL4 Omnibus Broad',    # Split-GAL4 broad collection
    'Split-GAL4 Omnibus Rescreen', # Split-GAL4 rescreen collection
    # Paper-specific collections:
    'Descending Neurons 2018',
    'Descending Neurons 2025',
    'Lateral Horn 2019',
    'MB Paper 2014',
    'SEZ 2021',
]

# Collection categories for easier filtering
# Maps category name (case-insensitive) to list of collection folder names
COLLECTION_CATEGORIES = {
    'GAL4/LEXA': [
        'Gen1',                  # Gen1 GAL4/LexA R-lines (CDM images)
        # VT GAL4 lines are handled separately via HTTP CDN
    ],
    'SPLITGAL4': [
        'Split-GAL4 Omnibus Broad',    # SS-lines broad collection
        'Split-GAL4 Omnibus Rescreen', # SS-lines rescreen
        'Lateral Horn 2019',           # LH paper Split-GAL4
        'Descending Neurons 2018',     # DN paper
        'Descending Neurons 2025',     # DN paper 2025
        'SEZ 2021',                    # SEZ paper
        'MB Paper 2014',               # MB paper
    ],
    'MCFO': [
        'Annotator Gen1 MCFO',  # R-lines with MCFO labeling
        'Gen1 MCFO',            # More MCFO data
    ],
    'RAWIMAGES': [
        # Collections with raw LSM confocal data
        'Annotator Gen1 MCFO',  # Has .lsm.bz2 files
        'Gen1 MCFO',            # Has .lsm.bz2 files
    ],
    'ALL': FLYLIGHT_COLLECTIONS,  # All collections
}

# Normalized lookup for case-insensitive access
_CATEGORY_LOOKUP = {k.upper().replace('-', '').replace('_', ''): k for k in COLLECTION_CATEGORIES.keys()}

# Reverse mapping: collection folder -> category
COLLECTION_TO_CATEGORY = {}
for category, collections in COLLECTION_CATEGORIES.items():
    if category != 'All':
        for coll in collections:
            COLLECTION_TO_CATEGORY[coll] = category

# File format extensions
FILE_FORMATS = {
    'png': ['.png'],
    'jpg': ['.jpg', '.jpeg'],
    'h5j': ['.h5j'],
    'lsm': ['.lsm.bz2', '.lsm'],
    'mp4': ['.mp4'],
    'json': ['.json'],
    'all': ['.png', '.jpg', '.jpeg', '.h5j', '.lsm.bz2', '.lsm', '.mp4', '.json'],
}

# Image type patterns
IMAGE_TYPES = {
    'mip': r'_(mip|signals_mip|multichannel_mip)\.png$',
    'cdm': r'-CDM_\d+\.png$',
    'aligned': r'-aligned_stack\.(h5j|png)$',
    'unaligned': r'-unaligned_stack\.(h5j|png)$',
    'translation': r'(_translation\.mp4$|\.t\.mp4$)',  # S3 and VT formats
    'signals': r'[_-]signals',
    'multichannel': r'[_-]multichannel',
    'metadata': r'-metadata\.json$',
    'raw': r'\.lsm(\.bz2)?$',
    # VT line specific patterns
    'projection': r'_total\.jpg$',  # VT line main projections
    'pattern': r'_ch2_.*\.jpg$',  # VT line pattern channel
    'substack': r'_\d{2}\.jpg$',  # VT line substacks
    'vt': r'(_(total|ch2_).*\.jpg$|\.t\.mp4$)',  # All VT file types
    'all': r'.*',
}


@dataclass
class FlyLightFile:
    """Represents a file in the FlyLight S3 bucket or HTTP CDN."""
    key: str
    size: int
    last_modified: str
    collection: str = ''
    line_name: str = ''
    source: str = 's3'  # 's3' or 'http'
    http_url: str = ''  # Direct URL for HTTP sources
    
    @property
    def filename(self) -> str:
        """Get the filename from the S3 key."""
        return os.path.basename(self.key)
    
    @property
    def extension(self) -> str:
        """Get the file extension."""
        name = self.filename.lower()
        if name.endswith('.lsm.bz2'):
            return '.lsm.bz2'
        return os.path.splitext(name)[1]
    
    @property
    def size_mb(self) -> float:
        """Get the file size in MB."""
        return self.size / (1024 * 1024)
    
    @property
    def url(self) -> str:
        """Get the URL for this file."""
        if self.source == 'http' and self.http_url:
            return self.http_url
        encoded_key = urllib.parse.quote(self.key)
        return f"https://s3.amazonaws.com/{FLYLIGHT_BUCKET}/{encoded_key}"


@dataclass
class VTSampleInfo:
    """Information about a VT line sample (brain or VNC)."""
    sample_id: str
    line_name: str
    region: str  # 'brain' or 'vnc'
    date: str
    sample_path: str  # Full sample path for URL construction
    session_id: str = ''  # For LSM download
    
    # Metadata from the page
    vector: str = ''
    landing_site: str = ''
    age: str = ''
    gender: str = ''
    reporter: str = ''


@dataclass
class FlyLightDownloader:
    """
    Download FlyLight imagery data from the Janelia S3 bucket.
    
    Attributes:
        output_dir: Directory to save downloaded files
        collections: List of collections to search (None = all, or use collection_category)
        collection_category: Category to filter ('GAL4', 'SplitGAL4', 'MCFO', 'All')
        formats: File formats to include ('png', 'h5j', 'lsm', 'mp4', 'json', 'all')
        image_types: Image types to include ('mip', 'cdm', 'aligned', etc.)
        max_workers: Number of parallel download threads
        verbose: Print progress messages
        use_boto3: Use boto3 for downloads (faster) if available
        simple_mode: Apply filename filtering to reduce download volume
        
    Collection Categories (case-insensitive):
        - 'GAL4/LEXA': Gen1 GAL4/LexA R-lines (+ VT lines via HTTP CDN if include_vt_lines=True)
        - 'SplitGAL4': SS-lines from Split-GAL4 Omnibus and paper-specific collections
        - 'MCFO': MCFO stochastic labeling collections
        - 'RawImages': Collections with raw LSM confocal data
        - 'All': All available collections
        
    Simple Mode Filtering:
        When simple_mode=True, applies filename-based filtering to reduce download volume:
        - Split-GAL4 collections: only files with '20x' AND 'multichannel' in filename,
          excluding files with 'image1' or 'image2' (duplicates)
        - VT GAL4 lines: only files with 'total' in filename
        - Gen1 R-lines: keep CDM and MIP files (they don't have 'total' variants)
        - Other collections (MCFO, etc.): keep all files
        
    Can also pass a list of categories: ['GAL4/LEXA', 'SplitGAL4']
        
    Note: If both `collections` and `collection_category` are specified,
          `collections` takes precedence.
    """
    output_dir: str = './flylight_downloads'
    collections: Optional[List[str]] = None
    collection_category: Optional[Union[str, List[str]]] = None  # 'GAL4/LEXA', 'SplitGAL4', 'MCFO', 'RawImages', 'All'
    formats: Union[str, List[str]] = 'png'
    image_types: Union[str, List[str]] = 'all'
    max_workers: int = 4
    verbose: bool = True
    use_boto3: bool = True
    include_vt_lines: bool = True  # Also search VT lines via HTTP CDN
    simple_mode: bool = False  # Apply filename filtering to reduce download volume
    
    # Internal state
    _s3_client: Any = field(default=None, repr=False, init=False)
    _file_cache: Dict[str, List[FlyLightFile]] = field(default_factory=dict, repr=False, init=False)
    _vt_sample_cache: Dict[str, List[VTSampleInfo]] = field(default_factory=dict, repr=False, init=False)
    _resolved_collections: List[str] = field(default=None, repr=False, init=False)
    
    def __post_init__(self):
        """Initialize S3 client and normalize parameters."""
        # Normalize formats
        if isinstance(self.formats, str):
            self.formats = [self.formats]
        
        # Normalize image types
        if isinstance(self.image_types, str):
            self.image_types = [self.image_types]
        
        # Resolve collections from category if not explicitly set
        self._resolve_collections()
        
        # Initialize boto3 client if available
        if self.use_boto3 and HAS_BOTO3:
            try:
                self._s3_client = boto3.client(
                    's3',
                    region_name=FLYLIGHT_REGION,
                    config=Config(signature_version=UNSIGNED)
                )
                if self.verbose:
                    print("✅ Using boto3 for S3 access")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  boto3 initialization failed: {e}")
                    print("   Falling back to HTTP access")
                self._s3_client = None
        else:
            self._s3_client = None
            if self.verbose:
                if not HAS_BOTO3:
                    print("ℹ️  boto3 not installed, using HTTP access")
                    print("   Install boto3 for faster downloads: pip install boto3")
    
    def _resolve_collections(self):
        """Resolve collections from category if not explicitly provided."""
        if self.collections is not None:
            # Use explicitly specified collections
            self._resolved_collections = self.collections
        elif self.collection_category:
            # Handle list of categories or single category
            categories = self.collection_category if isinstance(self.collection_category, list) else [self.collection_category]
            
            resolved = []
            for cat in categories:
                # Normalize for case-insensitive lookup
                normalized = cat.upper().replace('-', '').replace('_', '').replace(' ', '')
                
                # Find matching category
                if normalized in _CATEGORY_LOOKUP:
                    cat_key = _CATEGORY_LOOKUP[normalized]
                    resolved.extend(COLLECTION_CATEGORIES[cat_key])
                    if self.verbose:
                        print(f"📁 Category '{cat}' → {len(COLLECTION_CATEGORIES[cat_key])} collections")
                else:
                    if self.verbose:
                        print(f"⚠️ Unknown category: {cat}")
            
            if resolved:
                # Remove duplicates while preserving order
                seen = set()
                self._resolved_collections = [x for x in resolved if not (x in seen or seen.add(x))]
            else:
                if self.verbose:
                    print(f"⚠️ No valid categories found, using all collections")
                self._resolved_collections = FLYLIGHT_COLLECTIONS
        else:
            # Default: all collections
            self._resolved_collections = None  # None means search all
    
    def get_collection_category(self, collection_name: str) -> str:
        """Get the category for a given collection name."""
        return COLLECTION_TO_CATEGORY.get(collection_name, 'Other')
    
    @staticmethod
    def list_categories() -> Dict[str, List[str]]:
        """Return available collection categories and their collections."""
        return COLLECTION_CATEGORIES.copy()
    
    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def _is_vt_line(self, line_name: str) -> bool:
        """Check if a line name is a VT line (served from HTTP CDN)."""
        return bool(re.match(r'^VT\d+', line_name, re.IGNORECASE))
    
    def _parse_vt_page(self, line_name: str) -> tuple[List[VTSampleInfo], str]:
        """
        Parse the VT line view page to extract sample information.
        
        Returns:
            Tuple of (list of VTSampleInfo, session_id)
        """
        url = f"{VT_VIEW_CGI}?line={line_name}"
        samples = []
        session_id = ''
        
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                html = response.read().decode('utf-8')
        except Exception as e:
            self._log(f"❌ Error fetching VT page: {e}")
            return samples, session_id
        
        # Extract session ID from download URLs
        sid_match = re.search(r'sid=([a-f0-9]+)', html)
        if sid_match:
            session_id = sid_match.group(1)
        
        # Parse sample sections (Brain, VNC)
        # Pattern: download.cgi?id={sample_id} and projection URLs
        sample_pattern = re.compile(
            r"download\.cgi\?id=(\d+).*?"
            r"<h3>([^<]+)</h3>.*?"  # Section title (Brain, VNC)
            r"projections/(\d+)/"  # Date folder
            r"([^/]+)/"  # Sample path
            r"([^\"]+)\.jpg",  # First image filename
            re.DOTALL
        )
        
        # Simplified pattern for extracting projection URLs
        projection_pattern = re.compile(
            r'https://flimg\.janelia\.org/flylight-image/external-data/adult/secdata/projections/'
            r'(\d+)/'  # date
            r'([^/]+)/'  # sample folder
            r'([^\"\']+\.jpg)',  # filename
            re.IGNORECASE
        )
        
        # Find all projection URLs
        projection_matches = projection_pattern.findall(html)
        
        # Group by sample folder to identify unique samples
        sample_folders = {}
        for date, folder, filename in projection_matches:
            if folder not in sample_folders:
                sample_folders[folder] = {'date': date, 'files': []}
            sample_folders[folder]['files'].append(filename)
        
        # Extract sample IDs from download links
        download_pattern = re.compile(r"download\.cgi\?id=(\d+)")
        sample_ids = download_pattern.findall(html)
        
        # Determine region from folder name 
        # fA00b = brain, fA00v = VNC (case-insensitive search)
        for i, (folder, info) in enumerate(sample_folders.items()):
            folder_lower = folder.lower()
            if '-fa00b' in folder_lower or 'fa00b' in folder_lower:
                region = 'brain'
            elif '-fa00v' in folder_lower or 'fa00v' in folder_lower:
                region = 'vnc'
            else:
                # Try to infer from the sample path structure
                region = 'brain' if 'brain' in folder_lower else 'vnc' if 'vnc' in folder_lower else 'unknown'
            
            sample_id = sample_ids[i] if i < len(sample_ids) else ''
            
            samples.append(VTSampleInfo(
                sample_id=sample_id,
                line_name=line_name,
                region=region,
                date=info['date'],
                sample_path=folder,
                session_id=session_id
            ))
        
        return samples, session_id
    
    def _get_vt_files(self, line_name: str) -> List[FlyLightFile]:
        """
        Get available files for a VT line from the HTTP CDN.
        
        VT lines have a different structure than S3 files:
        - Projections: JPG images at flimg.janelia.org
        - Translations: MP4 movies
        - LSM: Full confocal stacks (requires CGI download)
        """
        if line_name in self._vt_sample_cache:
            samples = self._vt_sample_cache[line_name]
        else:
            samples, session_id = self._parse_vt_page(line_name)
            self._vt_sample_cache[line_name] = samples
        
        files = []
        
        for sample in samples:
            base_url = f"{VT_CDN_BASE}/projections/{sample.date}/{sample.sample_path}"
            translation_base = f"{VT_CDN_BASE}/translations/{sample.date}"
            
            # Construct filenames based on VT naming convention
            # Main projection: {line}_{sample_path}_total.jpg
            filename_base = f"{line_name}_{sample.sample_path.split('_', 1)[1] if '_' in sample.sample_path else sample.sample_path}"
            
            # Standard VT files (these exist for most samples):
            vt_file_types = [
                ('_total.jpg', 'projection'),
                ('_ch2_total.jpg', 'pattern_projection'),
            ]
            
            # Substack files (01-10 typically)
            for i in range(1, 11):
                vt_file_types.append((f'_{i:02d}.jpg', f'substack_{i:02d}'))
                vt_file_types.append((f'_ch2_{i:02d}.jpg', f'pattern_substack_{i:02d}'))
            
            for suffix, file_type in vt_file_types:
                filename = f"{filename_base}{suffix}"
                url = f"{base_url}/{filename}"
                
                files.append(FlyLightFile(
                    key=f"VT GAL4/{line_name}/{sample.region}/{filename}",
                    size=0,  # Unknown until we HEAD the URL
                    last_modified='',
                    collection='VT GAL4',
                    line_name=line_name,
                    source='http',
                    http_url=url
                ))
            
            # Translation video
            video_filename = f"{filename_base}.t.mp4"
            video_url = f"{translation_base}/{video_filename}"
            files.append(FlyLightFile(
                key=f"VT GAL4/{line_name}/{sample.region}/{video_filename}",
                size=0,
                last_modified='',
                collection='VT GAL4',
                line_name=line_name,
                source='http',
                http_url=video_url
            ))
        
        return files
    
    def _get_vt_mcfo_files(self, line_name: str) -> List[FlyLightFile]:
        """
        Get MCFO images for a VT line from gen1mcfo.janelia.org.
        
        VT lines have MCFO (Multi-Color Flip-Out) data that is viewable at
        gen1mcfo.janelia.org. The actual images are hosted in the S3 bucket
        (janelia-flylight-imagery) under 'Gen1 MCFO' and 'Annotator Gen1 MCFO'
        collections, but for VT lines, the images may not be discoverable via
        direct S3 listing. This method parses the gen1mcfo viewer page to find
        all available S3 image URLs.
        
        Args:
            line_name: VT line name (e.g., 'VT000770')
            
        Returns:
            List of FlyLightFile objects from the Gen1 MCFO collection on S3
        """
        import ssl
        files = []
        
        # Try multiple endpoints - gen1mcfo.janelia.org may have SSL issues
        # flweb.janelia.org serves the same content and is more reliable
        endpoints = [
            f"https://flweb.janelia.org/cgi-bin/view_gen1mcfo_imagery.cgi?line={line_name}",
            f"{GEN1_MCFO_VIEW_CGI}?line={line_name}",
        ]
        
        html = None
        for url in endpoints:
            try:
                # Use SSL context to avoid SSL handshake errors
                ctx = ssl.create_default_context()
                with urllib.request.urlopen(url, timeout=30, context=ctx) as response:
                    html = response.read().decode('utf-8')
                break  # Success - stop trying endpoints
            except Exception as e:
                self._log(f"   ⚠️ Error fetching MCFO page from {url.split('/')[2]}: {e}")
                continue
        
        if html is None:
            self._log(f"   ⚠️ Could not fetch Gen1 MCFO page for {line_name} from any endpoint")
            return files
        
        # Parse S3 image URLs from the page
        # Gen1 MCFO images are hosted on S3: https://s3.amazonaws.com/janelia-flylight-imagery/...
        # Pattern: src="https://s3.amazonaws.com/janelia-flylight-imagery/Gen1+MCFO/VT000770/..."
        # Also: "Annotator+Gen1+MCFO" collection
        
        # Find all S3 image URLs (PNG and JPG)
        s3_pattern = re.compile(
            r'https://s3\.amazonaws\.com/janelia-flylight-imagery/'
            r'((?:Gen1\+MCFO|Annotator\+Gen1\+MCFO)/[^"\'?]+\.(?:png|jpg))',
            re.IGNORECASE
        )
        
        matches = s3_pattern.findall(html)
        seen_keys = set()
        
        for key in matches:
            # Normalize key (URL decode the + signs)
            key = key.replace('+', ' ')
            
            # Skip duplicates
            if key in seen_keys:
                continue
            seen_keys.add(key)
            
            # Extract filename and collection
            parts = key.split('/')
            collection = parts[0] if parts else 'Gen1 MCFO'
            filename = parts[-1] if parts else key
            
            # Skip thumbnails and query-string variants
            if 'thumbnail' in filename.lower() or '_thumb' in filename.lower():
                continue
            
            files.append(FlyLightFile(
                key=key,
                size=0,
                last_modified='',
                collection=collection,
                line_name=line_name,
                source='s3',  # These are S3-hosted files
                http_url=f"https://s3.amazonaws.com/janelia-flylight-imagery/{key.replace(' ', '+')}"
            ))
        
        if files:
            self._log(f"   📦 Found {len(files)} Gen1 MCFO images for {line_name}")
        
        return files
    
    def _verify_vt_file_exists(self, file: FlyLightFile) -> bool:
        """Check if a VT file URL is accessible."""
        try:
            req = urllib.request.Request(file.http_url, method='HEAD')
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status == 200
        except:
            return False
    
    def list_vt_files(
        self, 
        line_name: str, 
        verify: bool = False,
        include_mcfo: bool = True
    ) -> List[FlyLightFile]:
        """
        List available files for a VT line.
        
        VT lines have images from two sources:
        1. GAL4 images from flimg.janelia.org (standard projections)
        2. MCFO (Multi-Color Flip-Out) images from gen1mcfo.janelia.org
        
        The search order follows the FlyLight category priority:
        1. GAL4/LEXA - Standard expression patterns (flimg.janelia.org)
        2. MCFO - Sparse labeling clones (gen1mcfo.janelia.org)
        
        Args:
            line_name: VT line name (e.g., 'VT037867', 'VT000770')
            verify: If True, verify each URL exists (slower but accurate)
            include_mcfo: If True, also search Gen1 MCFO collection (default: True)
            
        Returns:
            List of FlyLightFile objects from both GAL4 and MCFO sources
        """
        self._log(f"🔍 Searching VT line files for '{line_name}'...")
        
        # 1. Get standard VT GAL4 files from flimg.janelia.org
        files = self._get_vt_files(line_name)
        self._log(f"   Found {len(files)} GAL4 files from flimg.janelia.org")
        
        # 2. Get MCFO files from gen1mcfo.janelia.org
        if include_mcfo:
            mcfo_files = self._get_vt_mcfo_files(line_name)
            if mcfo_files:
                self._log(f"   Found {len(mcfo_files)} MCFO files from gen1mcfo.janelia.org")
                files.extend(mcfo_files)
        
        if verify:
            self._log("   Verifying file URLs (this may take a moment)...")
            verified_files = []
            for f in files:
                if self._verify_vt_file_exists(f):
                    verified_files.append(f)
            files = verified_files
            self._log(f"   Found {len(files)} verified files")
        else:
            self._log(f"   Found {len(files)} total potential files (use verify=True to confirm)")
        
        return files

    def _get_format_extensions(self) -> List[str]:
        """Get list of file extensions to include."""
        extensions = []
        for fmt in self.formats:
            fmt = fmt.lower()
            if fmt in FILE_FORMATS:
                extensions.extend(FILE_FORMATS[fmt])
            else:
                # Treat as raw extension
                if not fmt.startswith('.'):
                    fmt = '.' + fmt
                extensions.append(fmt)
        return list(set(extensions))
    
    def _get_image_type_patterns(self) -> List[re.Pattern]:
        """Get compiled regex patterns for image types."""
        patterns = []
        for img_type in self.image_types:
            img_type = img_type.lower()
            if img_type in IMAGE_TYPES:
                patterns.append(re.compile(IMAGE_TYPES[img_type], re.IGNORECASE))
            else:
                # Treat as raw pattern
                patterns.append(re.compile(img_type, re.IGNORECASE))
        return patterns
    
    def _matches_filters(self, file: FlyLightFile) -> bool:
        """Check if a file matches the format and image type filters."""
        # Check format
        extensions = self._get_format_extensions()
        if not any(file.filename.lower().endswith(ext) for ext in extensions):
            return False
        
        # Check image type
        patterns = self._get_image_type_patterns()
        if not any(p.search(file.filename) for p in patterns):
            return False
        
        return True
    
    def _list_bucket_http(self, prefix: str = '', marker: str = '') -> List[FlyLightFile]:
        """List bucket contents using HTTP (no boto3 required)."""
        files = []
        
        # Build request URL
        params = {'prefix': prefix}
        if marker:
            params['marker'] = marker
        
        url = f"https://s3.amazonaws.com/{FLYLIGHT_BUCKET}"
        if params:
            url += '?' + urllib.parse.urlencode(params)
        
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                xml_data = response.read()
        except Exception as e:
            self._log(f"❌ Error listing bucket: {e}")
            return files
        
        # Parse XML response
        root = ET.fromstring(xml_data)
        ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
        
        for content in root.findall('s3:Contents', ns):
            key = content.find('s3:Key', ns).text
            size = int(content.find('s3:Size', ns).text)
            last_modified = content.find('s3:LastModified', ns).text
            
            # Extract collection and line name from key
            # Handle structure: Collection/LineName/... or Collection/CDM/LineName/...
            parts = key.split('/')
            collection = parts[0] if len(parts) > 1 else ''
            
            # Check for CDM subfolder structure (Gen1/CDM/R21B12/...)
            if len(parts) > 2 and parts[1].upper() == 'CDM':
                line_name = parts[2] if len(parts) > 3 else ''
            else:
                line_name = parts[1] if len(parts) > 2 else ''
            
            files.append(FlyLightFile(
                key=key,
                size=size,
                last_modified=last_modified,
                collection=collection,
                line_name=line_name
            ))
        
        # Check for truncation
        is_truncated = root.find('s3:IsTruncated', ns)
        if is_truncated is not None and is_truncated.text == 'true':
            # Get more results
            next_marker = files[-1].key if files else marker
            files.extend(self._list_bucket_http(prefix, next_marker))
        
        return files
    
    def _list_bucket_boto3(self, prefix: str = '') -> List[FlyLightFile]:
        """List bucket contents using boto3."""
        files = []
        
        paginator = self._s3_client.get_paginator('list_objects_v2')
        
        try:
            for page in paginator.paginate(Bucket=FLYLIGHT_BUCKET, Prefix=prefix):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    parts = key.split('/')
                    collection = parts[0] if len(parts) > 1 else ''
                    
                    # Check for CDM subfolder structure (Gen1/CDM/R21B12/...)
                    if len(parts) > 2 and parts[1].upper() == 'CDM':
                        line_name = parts[2] if len(parts) > 3 else ''
                    else:
                        line_name = parts[1] if len(parts) > 2 else ''
                    
                    files.append(FlyLightFile(
                        key=key,
                        size=obj['Size'],
                        last_modified=str(obj['LastModified']),
                        collection=collection,
                        line_name=line_name
                    ))
        except Exception as e:
            self._log(f"❌ Error listing bucket: {e}")
        
        return files
    
    def list_files(self, line_name: str, use_cache: bool = True) -> List[FlyLightFile]:
        """
        List available files for a driver line.
        
        Automatically detects whether to use S3 (R-lines) or HTTP CDN (VT lines).
        
        Args:
            line_name: Driver line name (e.g., 'R10A06', 'VT037867')
            use_cache: Use cached file list if available
            
        Returns:
            List of FlyLightFile objects matching the line name
        """
        # Check cache
        if use_cache and line_name in self._file_cache:
            return self._file_cache[line_name]
        
        self._log(f"🔍 Searching for files matching '{line_name}'...")
        
        all_files = []
        
        # Check if this is a VT line - use HTTP CDN
        if self._is_vt_line(line_name) and self.include_vt_lines:
            self._log(f"   Detected VT line - searching HTTP CDN...")
            
            # Determine whether to include MCFO based on collection category
            # MCFO should be included if:
            # - No specific category is set (default: search all)
            # - Category is 'All' or 'MCFO'
            # - Category list includes 'MCFO' or 'All'
            include_mcfo = True  # Default: include MCFO
            include_gal4 = True  # Default: include GAL4
            
            if self.collection_category:
                categories = self.collection_category if isinstance(self.collection_category, list) else [self.collection_category]
                categories_normalized = [c.upper().replace('-', '').replace('_', '').replace(' ', '').replace('/', '') for c in categories]
                
                # Check if MCFO should be included
                mcfo_keywords = {'MCFO', 'ALL'}
                include_mcfo = any(c in mcfo_keywords for c in categories_normalized)
                
                # Check if GAL4/LEXA should be included
                gal4_keywords = {'GAL4LEXA', 'GAL4', 'LEXA', 'ALL'}
                include_gal4 = any(c in gal4_keywords for c in categories_normalized)
            
            # Get VT files based on category
            if include_gal4 and include_mcfo:
                # Get both GAL4 and MCFO
                all_files = self.list_vt_files(line_name, verify=False, include_mcfo=True)
            elif include_gal4:
                # GAL4 only - skip MCFO
                self._log(f"   Category filter: GAL4/LEXA only (no MCFO)")
                all_files = self._get_vt_files(line_name)
                self._log(f"   Found {len(all_files)} GAL4 files from flimg.janelia.org")
            elif include_mcfo:
                # MCFO only - skip GAL4
                self._log(f"   Category filter: MCFO only (no GAL4)")
                all_files = self._get_vt_mcfo_files(line_name)
                self._log(f"   Found {len(all_files)} MCFO files from gen1mcfo.janelia.org")
            else:
                # No matching category - return empty
                self._log(f"   ⚠️ No matching VT collections for category")
                all_files = []
        else:
            # Search S3 bucket for R-lines, Split-GAL4, etc.
            collections_to_search = self._resolved_collections or FLYLIGHT_COLLECTIONS
            
            # MCFO collections that commonly overlap - always search both
            mcfo_collections = {'Annotator Gen1 MCFO', 'Gen1 MCFO'}
            
            for i, collection in enumerate(collections_to_search):
                self._log(f"   Searching {collection}...")
                
                # Try direct path first: Collection/LineName/
                prefix = f"{collection}/{line_name}/"
                
                if self._s3_client:
                    files = self._list_bucket_boto3(prefix)
                else:
                    files = self._list_bucket_http(prefix)
                
                # If no files found, try nested CDM path: Collection/CDM/LineName/
                # (Gen1 collection has this structure)
                if not files:
                    prefix_cdm = f"{collection}/CDM/{line_name}/"
                    if self._s3_client:
                        files = self._list_bucket_boto3(prefix_cdm)
                    else:
                        files = self._list_bucket_http(prefix_cdm)
                
                all_files.extend(files)
                
                # Early exit logic - skip remaining collections if we have files,
                # BUT always check both MCFO collections since they commonly overlap
                if all_files and len(collections_to_search) > 3:
                    # Check if the next collection is also MCFO
                    next_collections = [c for c in collections_to_search[i+1:] if c in mcfo_collections]
                    if next_collections:
                        continue  # Keep searching MCFO collections
                    self._log(f"   ✓ Found files in {collection}, skipping remaining collections")
                    break
        
        # Cache results
        self._file_cache[line_name] = all_files
        
        self._log(f"   Found {len(all_files)} total files")
        
        return all_files
    
    def apply_simple_mode_filter(
        self, 
        files: List[FlyLightFile], 
        collection_name: Optional[str] = None
    ) -> List[FlyLightFile]:
        """
        Filter files based on simple_mode rules.
        
        Simple mode reduces download volume by selecting only representative files:
        - Split-GAL4 collections: only '20x' AND 'multichannel' files, excluding 'image1'/'image2'
        - GAL4/LexA collections: only 'total' files
        - VT lines: only 'total' files
        - Other collections (MCFO, etc.): keep all files
        
        Args:
            files: List of FlyLightFile objects to filter
            collection_name: Collection name for determining filter rules.
                           If None, uses file.collection attribute.
                           
        Returns:
            List of filtered FlyLightFile objects
        """
        if not self.simple_mode or not files:
            return files
        
        filtered = []
        
        for f in files:
            # Determine collection to use for filtering
            coll = collection_name or f.collection or ''
            coll_lower = coll.lower()
            filename_lower = f.filename.lower()
            
            # Split-GAL4: only include '20x' AND 'multichannel', exclude 'image1'/'image2'
            if any(x in coll_lower for x in ['splitgal4', 'split-gal4', 'split_gal4', 'omnibus']):
                if '20x' in filename_lower and 'multichannel' in filename_lower:
                    # Exclude duplicate images (image1, image2, etc.)
                    if 'image1' not in filename_lower and 'image2' not in filename_lower:
                        filtered.append(f)
            # VT GAL4 lines (from HTTP CDN): only include 'total'
            elif coll_lower.startswith('vt') or 'vt gal4' in coll_lower:
                if 'total' in filename_lower:
                    filtered.append(f)
            # Gen1 R-lines: keep CDM files (they don't have 'total' files)
            elif 'gen1' in coll_lower and 'mcfo' not in coll_lower:
                # Gen1 CDM images - keep them all (they're already filtered)
                if 'cdm' in filename_lower or 'mip' in filename_lower:
                    filtered.append(f)
            else:
                # For other collections (MCFO, etc.), keep all files
                filtered.append(f)
        
        if self.verbose and len(filtered) != len(files):
            self._log(f"   Simple mode: {len(files)} → {len(filtered)} files")
        
        return filtered
    
    def get_filtered_files(self, line_name: str, apply_simple_mode: Optional[bool] = None) -> List[FlyLightFile]:
        """
        Get files matching the configured format and image type filters.
        
        Args:
            line_name: Driver line name
            apply_simple_mode: Override class simple_mode setting. If None, uses class setting.
            
        Returns:
            List of FlyLightFile objects matching filters
        """
        all_files = self.list_files(line_name)
        filtered = [f for f in all_files if self._matches_filters(f)]
        
        self._log(f"   {len(filtered)} files match format/type filters")
        
        # Apply simple_mode filtering if enabled
        use_simple = apply_simple_mode if apply_simple_mode is not None else self.simple_mode
        if use_simple:
            filtered = self.apply_simple_mode_filter(filtered)
        
        return filtered
    
    def _download_file_http(self, file: FlyLightFile, local_path: Path) -> bool:
        """Download a file using HTTP."""
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(file.url, str(local_path))
            return True
        except Exception as e:
            self._log(f"❌ Error downloading {file.filename}: {e}")
            return False
    
    def _download_file_boto3(self, file: FlyLightFile, local_path: Path) -> bool:
        """Download a file using boto3."""
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self._s3_client.download_file(FLYLIGHT_BUCKET, file.key, str(local_path))
            return True
        except Exception as e:
            self._log(f"❌ Error downloading {file.filename}: {e}")
            return False
    
    def download_file(self, file: FlyLightFile, output_dir: Optional[str] = None, flat_structure: bool = False) -> Optional[Path]:
        """
        Download a single file.
        
        Args:
            file: FlyLightFile object to download
            output_dir: Override output directory (optional)
            flat_structure: If True, save to {line_name}/{filename} instead of preserving S3 key structure
            
        Returns:
            Path to downloaded file, or None if failed
        """
        output_path = Path(output_dir or self.output_dir)
        
        if flat_structure and file.line_name:
            # Save as {output_dir}/{line_name}/{filename}
            local_path = output_path / file.line_name / file.filename
        else:
            # Preserve directory structure
            local_path = output_path / file.key
        
        size_str = f"({file.size_mb:.1f} MB)" if file.size > 0 else ""
        self._log(f"⬇️  Downloading {file.filename} {size_str}...")
        
        # Use HTTP for VT files (source='http'), otherwise S3
        if file.source == 'http':
            success = self._download_file_http(file, local_path)
        elif self._s3_client:
            success = self._download_file_boto3(file, local_path)
        else:
            success = self._download_file_http(file, local_path)
        
        if success:
            self._log(f"   ✅ Saved: {local_path}")
            return local_path
        return None
    
    def download(
        self,
        line_name: str,
        output_dir: Optional[str] = None,
        max_files: Optional[int] = None,
        dry_run: bool = False,
        on_file_downloaded: Optional[Callable[[Path, str], None]] = None,
        flat_structure: bool = False,
        files: Optional[List] = None
    ) -> List[Path]:
        """
        Download files for a driver line.
        
        Args:
            line_name: Driver line name (e.g., 'R10A06')
            output_dir: Override output directory (optional)
            max_files: Maximum number of files to download (optional)
            dry_run: If True, only list files without downloading
            on_file_downloaded: Optional callback called after each file download
                               Signature: callback(file_path, line_name)
            flat_structure: If True, save to {line_name}/{filename} instead of preserving S3 key structure
            files: Optional pre-filtered list of FlyLightFile objects. If provided, skips get_filtered_files() call.
            
        Returns:
            List of paths to downloaded files
        """
        if files is None:
            files = self.get_filtered_files(line_name)
        
        if not files:
            self._log(f"⚠️  No files found for '{line_name}' matching filters")
            return []
        
        if max_files:
            files = files[:max_files]
        
        # Calculate total size
        total_size = sum(f.size for f in files)
        self._log(f"\n📦 {len(files)} files to download ({total_size / (1024*1024):.1f} MB total)")
        
        if dry_run:
            self._log("\n📋 Files (dry run - not downloading):")
            for f in files:
                self._log(f"   {f.filename} ({f.size_mb:.1f} MB)")
            return []
        
        # Download files
        output_path = Path(output_dir or self.output_dir)
        downloaded = []
        
        if self.max_workers > 1 and len(files) > 1:
            # Parallel download
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.download_file, f, str(output_path), flat_structure): f
                    for f in files
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        downloaded.append(result)
                        if on_file_downloaded:
                            on_file_downloaded(result, line_name)
        else:
            # Sequential download
            for f in files:
                result = self.download_file(f, str(output_path), flat_structure)
                if result:
                    downloaded.append(result)
                    if on_file_downloaded:
                        on_file_downloaded(result, line_name)
        
        self._log(f"\n✅ Downloaded {len(downloaded)}/{len(files)} files")
        
        return downloaded
    
    def get_metadata(self, line_name: str) -> List[Dict[str, Any]]:
        """
        Get metadata for a driver line without downloading full files.
        
        Args:
            line_name: Driver line name
            
        Returns:
            List of metadata dictionaries
        """
        # Temporarily override formats to get only JSON
        original_formats = self.formats
        original_types = self.image_types
        
        self.formats = ['json']
        self.image_types = ['metadata']
        
        files = self.get_filtered_files(line_name)
        
        # Restore original filters
        self.formats = original_formats
        self.image_types = original_types
        
        metadata_list = []
        
        for f in files:
            try:
                with urllib.request.urlopen(f.url, timeout=30) as response:
                    data = json.loads(response.read())
                    metadata_list.append(data)
            except Exception as e:
                self._log(f"⚠️  Error fetching metadata from {f.filename}: {e}")
        
        return metadata_list
    
    def search_lines(self, pattern: str) -> List[str]:
        """
        Search for driver lines matching a pattern.
        
        Args:
            pattern: Regex pattern to match line names
            
        Returns:
            List of matching line names
        """
        self._log(f"🔍 Searching for lines matching '{pattern}'...")
        
        matching_lines = set()
        regex = re.compile(pattern, re.IGNORECASE)
        
        # Search each collection
        collections_to_search = self._resolved_collections or FLYLIGHT_COLLECTIONS
        
        for collection in collections_to_search:
            self._log(f"   Searching {collection}...")
            
            # List top-level folders (line names)
            prefix = f"{collection}/"
            
            if self._s3_client:
                try:
                    result = self._s3_client.list_objects_v2(
                        Bucket=FLYLIGHT_BUCKET,
                        Prefix=prefix,
                        Delimiter='/'
                    )
                    for prefix_obj in result.get('CommonPrefixes', []):
                        line_name = prefix_obj['Prefix'].rstrip('/').split('/')[-1]
                        if regex.search(line_name):
                            matching_lines.add(line_name)
                except Exception as e:
                    self._log(f"   ⚠️  Error: {e}")
            else:
                # Use HTTP - less efficient
                files = self._list_bucket_http(prefix)
                for f in files:
                    if f.line_name and regex.search(f.line_name):
                        matching_lines.add(f.line_name)
        
        result = sorted(matching_lines)
        self._log(f"   Found {len(result)} matching lines")
        
        return result


def download_flylight_images(
    line_name: str,
    output_dir: str = './flylight_downloads',
    formats: Union[str, List[str]] = 'png',
    image_types: Union[str, List[str]] = 'mip',
    max_files: Optional[int] = None,
    simple_mode: bool = False,
    verbose: bool = True
) -> List[Path]:
    """
    Convenience function to download FlyLight images for a driver line.
    
    Args:
        line_name: Driver line name (e.g., 'R10A06')
        output_dir: Directory to save files
        formats: File formats ('png', 'h5j', 'lsm', 'mp4', 'json', 'all')
        image_types: Image types ('mip', 'cdm', 'aligned', 'metadata', 'all')
        max_files: Maximum files to download
        simple_mode: Apply filename filtering to reduce download volume:
            - Split-GAL4: only '20x' AND 'multichannel' files, excluding 'image1'/'image2'
            - GAL4/LexA: only 'total' files
        verbose: Print progress
        
    Returns:
        List of paths to downloaded files
        
    Example:
        >>> download_flylight_images('R10A06', formats='png', image_types='cdm')
        >>> download_flylight_images('VT037867', formats=['png', 'json'], image_types='all')
        >>> download_flylight_images('SS01015', simple_mode=True)  # Reduced download
    """
    downloader = FlyLightDownloader(
        output_dir=output_dir,
        formats=formats,
        image_types=image_types,
        simple_mode=simple_mode,
        verbose=verbose
    )
    
    return downloader.download(line_name, max_files=max_files)


def list_flylight_files(
    line_name: str,
    formats: Union[str, List[str]] = 'all',
    image_types: Union[str, List[str]] = 'all',
    simple_mode: bool = False,
    verbose: bool = True
) -> List[FlyLightFile]:
    """
    List available FlyLight files for a driver line without downloading.
    
    Args:
        line_name: Driver line name
        formats: File formats to filter
        image_types: Image types to filter
        simple_mode: Apply filename filtering to reduce file list:
            - Split-GAL4: only '20x' AND 'multichannel' files, excluding 'image1'/'image2'
            - GAL4/LexA: only 'total' files
        verbose: Print progress
        
    Returns:
        List of FlyLightFile objects
    """
    downloader = FlyLightDownloader(
        formats=formats,
        image_types=image_types,
        simple_mode=simple_mode,
        verbose=verbose
    )
    
    return downloader.get_filtered_files(line_name)
