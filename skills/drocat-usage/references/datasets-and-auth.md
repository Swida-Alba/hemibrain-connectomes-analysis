# Datasets, authentication, and outputs

## Dataset identifiers

Use the exact identifiers below. Version suffixes are significant.

| Identifier | Source | Notes |
| --- | --- | --- |
| `male-cns:v1.0` | NeuPrint | current male CNS dataset |
| `male-cns:v0.9` | NeuPrint | legacy male CNS; local cache may be useful |
| `hemibrain:v1.2.1` | NeuPrint | adult central brain |
| `optic-lobe:v1.1` | NeuPrint | optic-lobe dataset |
| `manc:v1.2.1` | NeuPrint | male VNC |
| `flywire_FAFB_v783` | FlyWire/CAVE + local files | requires CAVE token and downloaded files |
| `flywire_BANC_v888` | FlyWire/CAVE + local files | requires CAVE token and downloaded files |
| `flywire_BANC_v626` | FlyWire/CAVE + local files | legacy BANC files may be required |

Validate local FlyWire/BANC file layout with the repository's integration
guides before running a large query. Do not substitute a similarly named
dataset silently.

## Tokens

Preferred sources, in order:

1. `token_info_local.txt` at the repository root (gitignored);
2. environment variables handled by the project's token manager;
3. an explicit token argument only when the user has supplied it for this run.

Typical file entries are:

```text
NEUPRINT_TOKEN='...'
CAVE_TOKEN='...'
```

Never include token contents in logs, patches, notebooks, output reports, or
agent prompts. Use `--require-token` with the install verifier when the user
wants an authentication check.

## Output conventions

Keep outputs outside source directories, preferably under a timestamped
`local_data/agent_runs/<analysis-name>/` folder. Common artifacts include:

- FindPath/FindDirect: CSV/XLSX edge/path tables, summaries, network/heatmap HTML;
- PlotPath: network, Sankey, heatmap HTML plus connection tables;
- 3D skeleton: HTML, parameters, PNG views, individual profiles, PDF/PPTX, GIF/video;
- comparisons: per-dataset tables, threshold summaries, reports, conserved-path HTML;
- NeuronBridge: ranked CSV summaries, expression matrices, images, PDF/PPTX reports.

After a run, list files and inspect schemas. Preserve the run folder until the
user confirms that it can be removed.

## Offline/cache rules

- `use_cache=True` is the default for repeat work.
- `cache_only=True` prevents remote fallback and should be used only when cache
  coverage is known.
- FlyWire/BANC local files and NeuPrint caches are different resources; one does
  not replace the other.
- A missing token and a missing local file are different failures. Report which
  one occurred.
