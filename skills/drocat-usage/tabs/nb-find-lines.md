# Find Driver Lines (nb_find_lines)

Reproduce the **Find Driver Lines** UI tab (EM → LM). Uses `NeuronBridgeFinder`
to map EM neurons to GAL4/Split-GAL4 driver lines. No token required.

## Backend contract

- **tool_key:** `nb_find_lines`
- **import:** `from neuronbridge_finder import NeuronBridgeFinder`
- **class:** `NeuronBridgeFinder` (var `finder`)
- **method:** `finder.find_lines_batch(**method_params)`

## Parameters the UI builds

```python
from neuronbridge_finder import NeuronBridgeFinder

finder = NeuronBridgeFinder(
    verbose=True,
    separate_splitgal4=False,
    region=None,                        # filter by brain region
    max_workers=4,
)
# optional summary/download method params:
result = finder.find_lines_batch(
    queries=["aMe12"],                  # EM neuron queries
    dataset="male-cns:v0.9",
    output_dir="/absolute/output/nb_lines",
    match_type="both",                 # "cds" | "pppm" | "both"
    download_images=None,               # "neuronbridge" | "flylight" | "both" | None
    download_img_for_top_n_lines=None,
    summary_format=None,                # ["pdf"], ["pptx"], or None
    sort_by="score",
    image_formats=["png"],
    image_types=["cdm"],
    max_download_images_per_line=None,
    flylight_category=None,
    simple_mode=False,
    organize_by_region=False,
    pdf_images_per_page=(3, 2),
    summary_background_color="#ffffff",
)
```

## Run

```bash
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 --script archive/scripts_local/agent_NBLines_<date>.py
```

## Outputs

- Ranked line summaries (CSV) and optional FlyLight/NeuronBridge images and
  PDF/PPTX reports.

## Notes

- `download_images` drives image download: `"neuronbridge"`, `"flylight"`, `"both"`,
  or `None`. Start without downloads and a small query.
- `simple_mode` speeds up broad searches; `organize_by_region` groups the report
  by brain region.
