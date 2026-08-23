# Homolog Finding (find_homologs)

Reproduce the **Homolog Finding** UI tab as a direct backend call. Find
cross-dataset homologs of a source neuron using a connectivity-profile similarity
metric (fast adjacency search or the slower comprehensive search).

## Backend contract

- **tool_key:** `find_homologs`
- **import:** `from comparison.profile_comparator import HomologFinder`
- **class:** `HomologFinder` (var `finder`)
- **method:** `finder.find_homologs_fast()` (UI default) or `finder.find_homologs()`

## Parameters the UI builds

```python
from comparison.profile_comparator import HomologFinder

finder = HomologFinder(
    source_dataset="male-cns:v0.9",
    target_dataset="hemibrain:v1.2.1",
    output_dir="/absolute/output/homologs",
    top_n=30,
    top_k=15,
    top_m=5,
    similarity_metric="rank_union",
    vector_prefiltering=True,
    include_untyped_partners=False,     # expand 2-hop (untyped) partners
    visualize_skeleton=False,
    visualize_top_n=0,
    visualization_settings={},          # skeleton visualization options (only if visualize_skeleton)
    min_synapse_threshold=3,
    use_cache=True,
    saveas="",
    use_auto_type_mapping=True,
    ensure_cache_complete=False,
)
# The UI passes `source` per query:
finder = HomologFinder(..., source="aMe12", saveas="")
results = finder.find_homologs_fast()
# or finder.find_homologs() for the slower comprehensive search
```

## Run

```bash
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 --script archive/scripts_local/agent_Homologs_<date>.py
```

For multiple source queries, the UI loops `source` and gives each query its own
`saveas` suffix.

## Outputs

- bodyId/type homolog tables (CSV/XLSX) and summaries.

## Notes

- `use_auto_type_mapping=True` relies on the cross-dataset type mapper; set it
  explicitly when names differ between datasets.
- `visualize_skeleton=True` requires a valid `visualization_settings` dict.
- Start with `find_homologs_fast`; escalate to `find_homologs` only when the fast
  adjacency search is insufficient.
