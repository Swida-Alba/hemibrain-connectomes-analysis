# Similar Neurons (find_similar)

Reproduce the **Similar Neurons** UI tab. It drives **two** backend tools: a
morphological similarity search and a connectivity-profile similarity search. They
can run independently or be chained (morphology candidates can feed the profile
search). BANC datasets are excluded from morphological similarity.

## Backend contract

| Sub-analysis | tool_key | import · class | method |
| --- | --- | --- | --- |
| Morphological similarity | `find_similar_morphology` | `from morphology import MorphologyComparer` · `MorphologyComparer` | `comparer.find_similar()` |
| Connection-profile similarity | `find_similar_profile` | `from comparison.profile_comparator import HomologFinder` · `HomologFinder` | `finder.find_homologs_fast()` or `finder.find_novel_homologs()` |

## Morphological similarity (MorphologyComparer.find_similar)

```python
from morphology import MorphologyComparer

comparer = MorphologyComparer(
    dataset="male-cns:v0.9",
    level="type",                        # query level
    method="vector",                     # or "nblast"
    metric="cosine",                     # vector similarity metric
    candidate_cap=500,
    candidate_source="auto",             # "auto" | "profile" | "combined" | "cache" | "roi"
    roi_filter=None,                     # or ["EB", "LH", "AL"]
    visualize_top_n=0,                   # >0 to render top-N skeletons
    visualize_by="bodyId",
    visualization_settings={},           # skeleton settings when visualizing
    output_dir="/absolute/output/similar_morph",
    saveas="",
    verbose=True,
    n_workers=8,
    use_cache=True,
    cache_fetched_skeletons=True,
)
# The UI passes `query` per query (loop for multiple queries):
comparer = MorphologyComparer(..., query="aMe12")
results = comparer.find_similar()
```

## Connection-profile similarity (HomologFinder.find_homologs_fast / find_novel_homologs)

```python
from comparison.profile_comparator import HomologFinder

finder = HomologFinder(
    source_dataset="male-cns:v0.9",
    target_dataset="male-cns:v0.9",
    output_dir="/absolute/output/similar_profile",
    top_n=30,
    top_k=15,
    min_shared_partners=3,
    vector_prune_fraction=0.05,
    similarity_metric="rank_union",
    vector_prefiltering=True,
    include_untyped_partners=False,
    use_cache=True,
    saveas="",
    min_synapse_threshold=3,
    ensure_cache_complete=False,
    morphological_enrichment=True,       # use morphology candidates to enrich the search
    output_folder_prefix="similar-connectivity",
    visualize_skeleton=False,
    visualize_top_n=0,
    visualization_settings={},
    verbose=True,
)
# The UI passes `source` per query:
finder = HomologFinder(..., source="aMe12", saveas="")
results = finder.find_homologs_fast()   # or finder.find_novel_homologs()
```

## Run

```bash
# morphology (BANC excluded)
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 --script archive/scripts_local/agent_SimilarMorph_<date>.py
# profile
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 --script archive/scripts_local/agent_SimilarProfile_<date>.py
```

## Outputs

- Morphology: bodyId-level `results.csv` + type-level `type_summary.csv`; optional
  query-plus-top-N 3D skeleton HTML.
- Profile: ranked connectivity-similarity tables; optional skeleton/summary output.

## Notes

- Use `morphological_enrichment=True` to feed morphology candidates into the
  profile search — this is the tab's cross-tool combination.
- Multiple queries loop the `query`/`source` field; each query gets its own
  `saveas` suffix.
- `candidate_source` options are `["auto", "roi", "combined", "profile", "cache"]`;
  `"profile"`/`"combined"`/`"cache"` read the connection cache, `"roi"` uses ROI
  screening. Ensure cache coverage first or set `cache_fetched_skeletons` accordingly.
