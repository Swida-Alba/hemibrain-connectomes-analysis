# flylight_downloader — FlyLightDownloader

Module `src/flylight_downloader.py`. A single dataclass `FlyLightDownloader`
downloads FlyLight expression images/metadata for GAL4/Split-GAL4 driver lines
and can build PDF/PPTX summaries.

## Constructor (dataclass defaults)

```python
from flylight_downloader import FlyLightDownloader

downloader = FlyLightDownloader(
    output_dir="/abs/output/flylight",
    formats=["png"],                    # ["png", "jpg", "tif", ...]
    image_types=["mip"],
    region=None,                        # filter by region
    collection_category=None,           # "GAL4/LEXA" | "SplitGAL4" | "MCFO" | "RawImages" | "All"
    max_workers=4,
    simple_mode=False,
    use_boto3=True,                     # or HTTP fallback
    include_vt_lines=True,
    verbose="pbar",                     # "pbar" | True | False
)
```

## Key methods

| Method | Purpose |
| --- | --- |
| `download(line_name=..., output_dir=..., max_files=None, flat_structure=False, add_timestamp=True, generate_summary=None, summary_images_per_page=..., **filters)` | Download images/metadata; optionally summarize. |
| `list_vt_files(line_name=..., **filters)` | List available VT-line files before downloading. |
| `list_categories()` | List the supported image collection categories. |

```python
# dry-run style: bound the download with a small max_files
downloader.download(
    line_name=["SS00001"],
    output_dir="/abs/output/flylight",
    max_files=20,
    flat_structure=False,
    add_timestamp=True,
    generate_summary=None,              # "pdf" | "pptx" | ["pdf","pptx"] | None
    summary_images_per_page=(3, 2),
)
```

## Supports `dry_run`

Use a tiny `max_files` (or the backend dry-run flag) before downloading many
images. `generate_summary` accepts a string or a list (`["pdf","pptx"]`).

## Notes

- `simple_mode=True` skips broad category fallbacks for a faster, narrower download.
- `collection_category` narrows the download to a category; leave it `None` for the
  automatic fallback chain.
- Verify the file count/size after a download before generating large summaries.
