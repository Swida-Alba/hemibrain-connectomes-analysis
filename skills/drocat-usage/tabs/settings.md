# Settings / Datasets & Cache

Reproduce the **Settings** UI tab as a direct-backend workflow. The Settings tab
is the data-management hub: it pulls dataset metadata/connections, pulls skeletons,
builds connection/profile caches, checks FlyWire local readiness, and configures
tokens — none of which are a single `FindNeuronConnection`-style scientific tool.
Use this recipe when the task is about dataset readiness, cache building, indexes,
or token configuration rather than a specific scientific analysis.

## Backend contract

| Task | Entry point | Notes |
| --- | --- | --- |
| Dataset metadata pull (neurons + ROI table + index) | `DatasetPuller` (`ui/dataset_pull.py`) or `FindNeuronConnection._ensure_complete_dataset()` + `_ensure_neuron_index_from_metadata()` | Idempotent; first pull downloads the full neuron table. |
| Connections pull (cache) | `DatasetPuller` (`_run_connections`) or `python src/build_connection_cache.py` | Builds the connection cache. |
| Skeleton pull | `SkeletonPuller` (`ui/skeleton_pull.py`) | Resumable, cancelable. |
| Connectivity-profile cache | `python src/build_connectivity_profile_cache.py` | Used by profiling/similar tools. |
| Seed neuron indexes | `python src/build_seed_indexes.py` | Refreshes committed `neuron_indexes/` seeds. |
| FlyWire local readiness | `src/utils/flywire_readiness.py` (`flywire_skeleton_readiness`, `require_flywire_skeleton_access`) | Detects missing local FAFB/BANC files. |
| FlyWire/FAFB conversion | `src/FAFB_file_converter.py`, `src/BANC_file_converter.py` | Converts raw Codex downloads to parquet. |
| Token configuration | `config.json` / `config_local.json` `tokens` section | See `src/utils/token_manager.py`. |

## Metadata pull (first run for a new dataset)

```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(
    dataset="male-cns:v0.9",
    use_cache=True,
    cache_only=False,
)
fc.InitializeNeuronInfo()               # ensures metadata + neuron index
```

## Connection cache build

```bash
# Option A: script
python src/build_connection_cache.py male-cns:v0.9
# Option B: the UI puller (asynchronous, via DatasetPuller)
#   DatasetPuller().start(dataset="male-cns:v0.9", operation="connections", ...)
```

## Skeleton pull

```python
from ui.skeleton_pull import SkeletonPuller

puller = SkeletonPuller()
puller.start(dataset="male-cns:v0.9", max_workers=4)
# puller.state / puller.running / puller.cancel()
```

## Connectivity-profile cache

```bash
python src/build_connectivity_profile_cache.py male-cns:v0.9
```

## Seed indexes refresh

```bash
python src/build_seed_indexes.py
```

## FlyWire local readiness & conversion

```python
from src.utils.flywire_readiness import flywire_skeleton_readiness, require_flywire_skeleton_access
from src.flywire_ids import is_flywire_dataset, is_banc_dataset

# check the converted parquet layout for a FlyWire dataset
readiness = flywire_skeleton_readiness("flywire_FAFB_v783")
```

For FlyWire datasets, the raw Codex downloads belong under
`datasets/<dataset>/downloads/` and must be converted (via
`FAFB_file_converter.py` / `BANC_file_converter.py`) before analysis. See
[references/datasets-and-auth.md](../references/datasets-and-auth.md).

## Tokens

```python
from src.utils.token_manager import get_access_token, get_cave_token  # no values printed
```

Set `neuprint` and `cave` in the `tokens` section of `config.json` (wins per key)
or the gitignored `config_local.json`. Never print or commit the values.

## Notes

- FlyWire FAFB/BANC are local-file datasets: they require converted local files
  (CAVE token only for explicit remote fetch/fallback). NeuPrint datasets are
  API-only and need the NeuPrint token.
- Pulls stream and are resumable/cancelable; metadata must exist before
  connection/skeleton pulls (the puller enforces this).
- `neuron_indexes/` are persistent "system files" (not `cache/`); clearing
  `cache/` never removes them.
