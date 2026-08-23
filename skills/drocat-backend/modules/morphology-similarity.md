# morphology — MorphologyComparer

Morphological similarity search (module `src/morphology.py`). The main class is
`MorphologyComparer` (query-vs-all similarity); `SkeletonVectorCache` manages the
skeleton vector cache used by morphological and `candidate_source` paths. BANC
datasets are not supported for morphology.

## MorphologyComparer — key params

```python
from morphology import MorphologyComparer

comparer = MorphologyComparer(
    query="aMe12",                      # str/int query (per query; loop for many)
    dataset="male-cns:v0.9",            # non-BANC only
    level="auto",                       # "auto" | "type" | "bodyid"
    method="vector",                    # "vector" | "nblast"
    metric="cosine",                    # vector similarity metric
    candidate_cap=500,                  # screen candidates entering comparison
    candidate_source="auto",            # "auto" | "roi" | "combined" | "profile" | "cache"
    visualize_top_n=0,                  # >0 to render top-N skeletons
    visualize_by="type",                # or "bodyId"
    min_weight=3,                       # connection-cache screen threshold
    min_shared_partners=2,
    roi_filter=None,                    # or ["EB", "LH", "AL"]
    output_dir="/abs/output/similar_morph",
    saveas="",
    verbose=True,
    n_workers=8,
    use_cache=True,
    cache_fetched_skeletons=True,       # persist raw skeletons
)

results = comparer.find_similar()       # -> pd.DataFrame
```

The two **size knobs** shared by every candidate-source mode are `candidate_cap`
(bounds how many screened candidates enter the comparison) and `visualize_top_n`
(bounds rendering only). Results are never truncated — every compared candidate is
returned and written.

## SkeletonVectorCache

```python
from morphology import SkeletonVectorCache

cache = SkeletonVectorCache(dataset="male-cns:v0.9")
cache.build(fetch_missing=0)            # vectorize cached skeletons incrementally
cache.ensure(fetch_missing=0)           # build + load, ready to query
cache.coverage()                        # dict of coverage stats
vecs = cache.vectors_for(body_ids, compute_missing=True)
```

## Supporting functions

- `find_similar_raw_cache(dataset, ...)` — the raw skeleton cache helper.
- `find_similar_dataset_cache(dataset, ...)` — dataset-level cache helper.
- `find_similar_flywire_mesh_cache(...)` — FlyWire mesh cache helper.

## Notes

- `candidate_source` options are `["auto", "roi", "combined", "profile", "cache"]`;
  `"profile"`/`"combined"`/`"cache"` read the connection cache, `"roi"` uses ROI
  screening. Ensure cache coverage first (or set `cache_fetched_skeletons`).
- `visualize_top_n>0` triggers 3D rendering of the top matches — the UI feeds
  this through the shared skeleton settings.
- Multiple queries are a loop: one `MorphologyComparer` per query.
