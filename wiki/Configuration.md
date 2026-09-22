# Configuration Reference

teval is controlled entirely by a YAML configuration file. Generate a default:

```bash
python -m teval --init
```

Or view inline help for every parameter:
```bash
python -m teval --help-config
```

## Top-level sections

| Section | Description |
|---|---|
| `io` | Input/output file paths |
| `system` | CPU, Dask, stream-to-disk settings |
| `data` | Time slicing and feature filtering |
| `stats` | Ensemble statistic options |
| `metrics` | Skill metric selection |
| `viz` | All visualization flags and options |

## `io`

| Key | Type | Description |
|---|---|---|
| `troute_netcdf_dir` | path | Directory containing T-Route `*_output/` folders |
| `ensemble_netcdf_dir` | path | Pre-computed ensemble NC directory (skip recompute) |
| `hydrofabric_dir` | path | Directory containing `.gpkg` hydrofabric file(s) |
| `observations_file` | path | USGS observations Parquet or CSV file |
| `auto_download_usgs` | bool | Download observations via USGS API if file not found |
| `offline` | bool | Never use the network: no USGS download, no basemaps. Stops at startup if `observations_file` lacks what the run needs (see `--fetch` in [HPC](HPC.md)) |
| `save_downloaded_obs` | path | Save downloaded observations to this path |
| `output_dir` | path | Root output directory |
| `per_domain_output` | bool | Create per-domain subdirectories under `output_dir` |
| `directory_naming` | `suffix`\|`parent` | T-Route directory naming convention |
| `metrics_output_file` | str | Filename for metrics CSV (default: `metrics.csv`) |

### Directory naming conventions

**`suffix`** (default) — flat layout:
```
troute_netcdf_dir/
  {formulation}_{domain}_output/
```

**`parent`** — nested layout:
```
troute_netcdf_dir/
  {domain}/
    {formulation}_output/
```

## `system`

| Key | Type | Default | Description |
|---|---|---|---|
| `cpu` | int | `-1` | Worker count for all parallel work, used as given. `-1` = the job's allocation (`SLURM_CPUS_PER_TASK`, else the cores this process may use). |
| `stream_to_disk` | bool | `true` | Write ensemble NC during compute pass (recommended for CONUS) |
| `use_dask` | bool | `true` | Use Dask lazy evaluation |
| `logging_level` | str | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` |
| `timing` | str | `simple` | `none`, `simple`, or `verbose` |

## `stats`

| Key | Type | Default | Description |
|---|---|---|---|
| `quantiles` | list[float] | `[0.05, 0.95]` | Spread band quantiles (used when ensemble size >= `small_domain_threshold`) |
| `small_domain_threshold` | int | `10` | Below this member count, use min/max instead of quantiles |
| `weights` | block | *(absent)* | Optional spatially varying ensemble weights. Omit for the default simple mean. See [`stats.weights`](#statsweights) |

### `stats.weights`

Supplying this block switches `streamflow_mean` from an equal-weight arithmetic
mean to a weighted sum over the formulation dimension. Weights are supplied per
**nexus**; every flowpath draining to a nexus receives that nexus' weights.

Omit the block entirely (or set it to `null`) and nothing changes — no weight
file is read and the unweighted code path runs exactly as before.

| Key | Type | Default | Description |
|---|---|---|---|
| `file` | path | *(required)* | Weight file, `.csv` or `.parquet` |
| `formulation_index_map` | dict[int, str] | *(required)* | Binding from the file's `formulation_index` to formulation name, e.g. `{1: cfe, 2: noahowp}` |
| `on_missing` | `warn`\|`error` | `warn` | Policy for features whose nexus is absent from the weight file |
| `normalize` | bool | `false` | Divide each nexus' weights by their sum instead of requiring them to sum to 1.0 |

```yaml
stats:
  quantiles: [0.05, 0.95]
  weights:
    file: /path/to/nexus_weights.csv
    formulation_index_map:
      1: cfe
      2: noahowp
      3: lstm
    on_missing: warn
    normalize: false
```

#### Weight file schema

A tidy table with one row per (nexus, formulation) pair. Any additional column
is ignored.

| Column | Type | Description |
|---|---|---|
| `nexus_id` | str | Nexus identifier, `nex-` prefix retained (e.g. `nex-9001`) |
| `formulation_index` | int | 1-based index into `formulation_index_map` |
| `weight` | float | Non-negative weight for that formulation at that nexus |

```csv
nexus_id,formulation_index,weight
nex-9001,1,0.5
nex-9001,2,0.3
nex-9001,3,0.2
nex-9002,1,0.1
nex-9002,2,0.9
nex-9002,3,0.0
```

> **The weight file format is provisional.** The process that will generate
> these weights does not yet exist, and the format is expected to change.

#### `formulation_index_map`

This binding is required, not optional, because the order of the formulation
dimension follows directory scan order and is not stable across machines — so
the file's integer indices cannot be read positionally.

The map must name **exactly** the formulations discovered in the run. Both
directions of a mismatch are errors: an index naming a formulation that is not
in the run (a stale legend), and a discovered formulation that no index names
(which would silently drop a member from the product).

#### Rules every weight group must obey

A "group" is all the rows for one nexus. Violations raise before any compute is
triggered, naming the offending nexus ids.

| Rule | Behaviour |
|---|---|
| Completeness | A nexus with *any* rows must carry exactly one row per configured index. A missing row or a duplicate row is a hard error, and neither is configurable. A nexus with *no* rows is not an error — that is coverage, governed by `on_missing`. |
| Sign | Negative weights are rejected. |
| All-zero groups | A group whose weights are all zero is rejected; it would silently produce zero flow at that location. An individual zero inside an otherwise non-zero group is permitted and meaningful — it excludes one formulation at one location deliberately. |
| Sums | Each group must sum to 1.0 within a tolerance of `1e-6`. With `normalize: true` each group is divided by its own sum instead, accepting any positive scale. |

#### Coverage and `on_missing`

A feature whose nexus is absent from the weight file — or which drains to no
nexus at all — is *uncovered*. `on_missing` decides what that means:

- **`warn`** (default) — uncovered features fall back to equal weights, which
  is exactly the simple mean, so they behave as they did before weighting was
  configured. Covered/uncovered counts and the coverage fraction are logged at
  warning level.
- **`error`** — incomplete coverage aborts the run. Coverage is determined
  before any Dask graph is built, so the failure lands in the first seconds of
  a run rather than after a long compute.

The achieved coverage fraction is written into the output NetCDF — see
[Outputs](https://github.com/shorvath-noaa/teval/wiki/Outputs#weighting-provenance).

#### Limits and interactions

- **Median and the spread band remain unweighted.** Only `streamflow_mean` is
  weighted. `streamflow_median`, `streamflow_min`/`max` and
  `streamflow_p05`/`p95` are computed across formulations exactly as they are
  in an unweighted run.
- **A pre-computed ensemble bypasses weighting entirely.** Weighting lives in
  the statistics builder, which is skipped when an existing ensemble NetCDF
  from `io.ensemble_netcdf_dir` is reused. The run logs a prominent warning and
  otherwise succeeds, reusing whatever the file already holds. Delete the
  cached ensemble file to have weights applied.
- **A hydrofabric is required.** The per-nexus weights are joined to features
  through a crosswalk derived from the hydrofabric's flowpaths, so a domain
  with no hydrofabric aborts the run with an explicit error rather than
  quietly falling back to the unweighted mean.

  **When teval loads a hydrofabric.** Only when `metrics.enabled` or
  `viz.interactive_map.enabled` is set — nothing else reads
  `io.hydrofabric_dir`. Switching both off therefore leaves every domain
  without a hydrofabric even when that directory holds a perfectly good
  GeoPackage, which is the usual cause of the error above in a run whose
  configuration otherwise looks complete.
- **`stats.weights` is global while the hydrofabric is per-domain.** A single
  hydrofabric-less domain therefore aborts a whole multi-domain run. This is
  intended: a weighted run that silently produced unweighted output for some
  domains would be worse. (A domain reusing a pre-computed ensemble is exempt,
  since it needs no crosswalk; it gets the bypass warning instead.)

## `metrics`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Compute skill metrics |
| `variables` | list | `[nse, kge, pbias]` | Metrics to compute. Options: `nse`, `kge`, `pbias`, `peak_flow_error`, `peak_timing_error` |
| `per_formulation` | bool | `true` | Compute metrics for each formulation in addition to ensemble mean |
| `bootstrap_enabled` | bool | `false` | Compute bootstrap confidence intervals (slow) |
| `bootstrap_samples` | int | `1000` | Number of bootstrap resamples |
| `confidence_level` | float | `0.95` | CI confidence level |

## `viz`

### `hydrographs`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Render per-gage hydrograph PNGs |
| `target_ids` | list | `[]` | Specific gage IDs to plot. Empty = all gages with observations |
| `plot_uncertainty` | bool | `false` | Shade the ensemble spread band |
| `plot_members` | bool | `true` | Plot individual formulation lines (spaghetti plot) |

### `skill_maps`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Render skill map figures |
| `score_maps` | bool | `true` | Spatial scatter map coloured by metric value |
| `winner_maps` | bool | `true` | Map showing which formulation wins at each gage |
| `boxplots` | bool | `true` | Score distribution boxplots per formulation |
| `vpu_breakdown` | bool | `true` | Stacked-bar win-rate by VPU (CONUS only) |
| `variables` | list | `[nse, kge, pbias]` | Metrics to map |
| `basemap` | bool | `true` | Add contextily basemap tiles |

### `interactive_map`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Render Folium HTML interactive map |
| `variable` | str | `streamflow_mean` | Variable to display |

### `animation`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Render GIF animation |
| `variable` | str | `streamflow_mean` | Variable to animate |
| `fps` | int | `8` | Frames per second |
| `log_scale` | bool | `true` | Use log colour scale |
| `cmap` | str | `hydro_flow` | Matplotlib colormap |
| `time_step` | str | `1W` | Pandas offset string for frame interval (e.g. `1D`, `1W`) |
| `min_stream_order` | int | `4` | Minimum stream order to include (reduces feature count) |
