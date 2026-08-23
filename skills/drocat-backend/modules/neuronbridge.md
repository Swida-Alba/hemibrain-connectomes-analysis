# neuronbridge_finder — NeuronBridgeFinder

Module `src/neuronbridge_finder.py`. A single dataclass `NeuronBridgeFinder`
carries out EM↔LM mapping: find GAL4/Split-GAL4 driver lines for EM neurons (EM →
LM), find candidate EM neurons for LM lines (LM → EM), and analyze line
co-labeling. **No token is required** for the NeuronBridge lookup.

## Constructor (dataclass defaults)

```python
from neuronbridge_finder import NeuronBridgeFinder

finder = NeuronBridgeFinder(
    verbose=True,
    separate_splitgal4=False,
    region=None,                        # filter by brain region
    max_workers=4,
)
```

## Key methods

| Method | Direction | Purpose |
| --- | --- | --- |
| `find_lines_batch(queries=..., dataset=..., output_dir=..., match_type=..., download_images=None, download_img_for_top_n_lines=None, summary_format=None, sort_by=..., image_formats=..., image_types=..., max_download_images_per_line=None, flylight_category=None, simple_mode=False, organize_by_region=False, pdf_images_per_page=..., summary_background_color=...)` | EM → LM | Ranked driver lines for EM neuron queries. |
| `find_neurons_batch(line_names=..., output_dir=..., match_type=..., top_n=..., min_score=..., visualize_top_n=..., generate_individual_profiles=None, visualize_by=..., visualization_settings=..., sort_by=..., pdf_images_per_page=..., background_color=...)` | LM → EM | Ranked candidate EM neurons for line names. |
| `analyze_colabeling(lines=..., match_type=..., output_dir=..., similarity_methods=..., generate_report=..., visualize=..., visualize_top_n=..., top_n_neurons=..., min_score=..., min_type_avg_score=..., sort_by=..., background_color=..., pdf_images_per_page=..., datasets_to_visualize=..., visualize_by=..., visualization_settings=...)` | LM ↔ LM | Co-labeling overlap/similarity/specificity + heatmaps. |

```python
# EM → LM
finder.find_lines_batch(queries=["aMe12"], dataset="male-cns:v0.9",
                        output_dir="/abs/output/nb_lines", match_type="both")

# LM → EM
finder.find_neurons_batch(line_names=["SS00001"], output_dir="/abs/output/nb_neuron",
                          match_type="both", top_n=10)

# Co-labeling
finder.analyze_colabeling(lines=["SS00001", "SS00002"], output_dir="/abs/output/nb_colabel",
                          similarity_methods=["jaccard"], generate_report=True)
```

## Optional visualization methods

- `visualize_colabeling_matrix(...)`, `visualize_expression_matrix(...)`,
  `visualize_expression_matrix_merged(...)`, `visualize_labeling_distribution(...)`,
  `visualize_colabeling_distribution(...)` — standalone heatmaps/plots.

## Notes

- `download_images` toggles image download (`"neuronbridge"`, `"flylight"`,
  `"both"`); `max_download_images_per_line` bounds it. Start without downloads.
- `simple_mode=True` speeds broad searches; `organize_by_region=True` groups the
  report by region.
- `match_type` is the NeuronBridge matching algorithm, one of `"cds"`, `"pppm"`, `"both"`.
- A 3D skeleton view (`visualize_top_n>0`) needs `visualization_settings`.
