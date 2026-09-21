"""Core domain data loading, metrics calculation, and per-domain visualization dispatch."""

import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import logging
import multiprocessing
from joblib import Parallel, delayed
import gc

from teval.config import IOConfig, MetricsConfig, StatsConfig, VizConfig
from teval.ensemble_methods.stats import build_stats
from teval.io import load_hydrofabric, fetch_observations
from teval.utils import Timer
from teval.metrics import deterministic as det
from teval.weights import (
    WeightPlan,
    prepare_weight_plan,
    resolve_domain_weights,
    weighting_attrs,
)
import teval.viz.static as tviz
import teval.viz.animation as tanim

logger = logging.getLogger(__name__)


def get_gage_fids(domain_data: Dict, viz_config: VizConfig) -> np.ndarray:
    """
    Return the array of feature IDs needed for gage-level metrics and
    hydrographs.

    Used by __main__ to build the gage-subset selection that is computed
    alongside the full-domain disk write in a single Dask pass.
    """
    gage_to_fids = domain_data.get("gage_to_fids", {})
    obs_df       = domain_data.get("gage_obs", pd.DataFrame())
    ds_stats     = domain_data["formulations"].get("combined")

    if not gage_to_fids or ds_stats is None:
        return np.array([], dtype=int)

    obs_gages  = set(str(g) for g in obs_df.columns) if not obs_df.empty else set()
    target_ids = set(str(t) for t in (viz_config.hydrographs.target_ids or []))

    needed_gages = set(str(g) for g in gage_to_fids)
    if obs_gages:
        needed_gages &= obs_gages
    if target_ids:
        needed_gages |= (set(str(g) for g in gage_to_fids) & target_ids)

    required_fids: List[int] = []
    for g in needed_gages:
        if g in gage_to_fids:
            required_fids.extend(gage_to_fids[g])

    return np.intersect1d(list(set(required_fids)), ds_stats.feature_id.values)


def reuses_precomputed_ensemble(formulation_dict: Dict) -> bool:
    """
    Say whether this domain returns a pre-computed ensemble instead of building one.

    Answered from the domain map alone, so it is known before anything is
    loaded.  Both weighting guard rails turn on this one question -- the
    missing-hydrofabric error is not applied to a domain that needs no
    crosswalk, and the bypass warning is made for exactly the domains it is
    not applied to -- and they must agree about a given run, which they can
    only be relied on to do while they ask the same function.

    Naming the file is what decides it, because naming one that is not there is
    refused outright: a configuration pointing at a missing ensemble is far more
    likely a stale path than a request to rebuild, and rebuilding would hand
    back numbers the run was not asked for without saying so.  Refusing here
    keeps the question decidable from the domain map alone, which is what lets
    every site share this one answer.

    Raises
    ------
    FileNotFoundError
        The domain map names an ensemble file that is not on disk.
    """
    ensemble_file = formulation_dict.get("ensemble_file")
    if ensemble_file is None:
        return False

    if not ensemble_file.exists():
        raise FileNotFoundError(
            f"The domain map names the pre-computed ensemble {ensemble_file}, "
            f"which does not exist. Point it at the ensemble to reuse, or "
            f"remove the entry to build the statistics from the raw members."
        )
    return True


# Functions for loading domain data based on the domain map created in initialize_domains
def load_domain_data(domain_dict: Dict, io: IOConfig, stats_config: StatsConfig) -> Dict:
    """
    Load all data needed for one domain: hydrofabric, formulations, and observations.

    The three steps are ordered by what they depend on, not by convenience:

    1. Hydrofabric first.  It depends only on domain_dict['hydrofabric'], so it
       can be loaded before anything else -- and it must be, because the
       nexus-to-feature crosswalk it yields is what a weighted run needs in
       hand before the ensemble stats graph is built.
    2. Formulations second, which is where the stats graph is constructed.
       When stats.weights is configured, the weight plan built from steps 1's
       hydrofabric is resolved here, against the formulation names and feature
       ids the opened dataset reports, and handed to build_stats.
    3. Observations last, because the fetch window is derived from the time
       bounds the formulation files report.

    With no stats.weights block the plan is None and every step behaves exactly
    as it did before weighting existed -- including for a domain with no
    hydrofabric, which stays a supported entry.

    Raises
    ------
    ValueError
        Weights are configured for a domain that will build statistics but has
        no hydrofabric, which makes the nexus-to-feature crosswalk impossible.
        Raised in step 1, before a single formulation file is opened.  A domain
        reusing a pre-computed ensemble is exempt: it needs no crosswalk, and
        the bypass warning describes it accurately where this error would not.
    """
    results = {}

    # Process Hydrofabric
    results['hydrofabric'], all_gage_ids, results['gage_to_fids'], results['gage_to_nexus'] = load_hydrofabric(
        domain_dict['hydrofabric'], io.hydrofabric_layer
        )
    
    # Prepare Weights (no-op when stats.weights is absent)
    weight_plan = prepare_weight_plan(
        stats_config,
        results['hydrofabric'],
        reusing_ensemble=reuses_precomputed_ensemble(domain_dict['formulations']),
    )

    # Process Formulations
    results['formulations'] = {'combined': None, 'ensemble_members': None}
    ds_stats, ds_members, t_min, t_max = _process_formulation_files(domain_dict['formulations'], stats_config, weight_plan)

    results['formulations']['combined'] = ds_stats
    results['formulations']['ensemble_members'] = ds_members

    initial_gages = domain_dict.get('gage_obs', {}).get('domain_name', [])
    if "CONUS" in initial_gages: initial_gages.remove("CONUS")
    gage_ids = list(set(initial_gages + all_gage_ids))
    
    # Fetch/Load Observations
    results['gage_obs'] = fetch_observations(gage_ids, t_min, t_max, io)

    return results

def _process_formulation_files(
    formulation_dict: Dict,
    stats_config: StatsConfig,
    weight_plan: Optional[WeightPlan] = None,
) -> tuple:
    """
    Loads pre-computed ensemble if available, and raw members if available.
    Calculates stats only if a pre-computed ensemble is not provided.

    *weight_plan* carries the weight file and the nexus crosswalk gathered by
    ``load_domain_data`` before this step ran; it is resolved against the
    dataset opened here and passed to ``build_stats``.  ``None`` -- the default,
    and what an unconfigured run supplies -- takes the unweighted path
    unchanged.  A pre-computed ensemble is returned as it was written and the
    plan goes unused, since the statistics it holds were built elsewhere; that
    combination is legal but almost never what was meant, so it warns loudly
    rather than passing in silence.

    Whichever path the mean took is recorded on the returned statistics dataset
    as provenance attributes, so the output NetCDF the pipeline writes from it
    says for itself whether it holds a weighted mean.  A pre-computed ensemble
    keeps whatever attributes it was written with, untouched: this step did not
    build those statistics and so has nothing to attest about them.
    """
    raw_files = formulation_dict.get("raw_files", {})
    ensemble_file = formulation_dict.get("ensemble_file")
    
    ds_stats = None
    combined_ds = None
    t_min, t_max = None, None

    # Load Pre-Computed Ensemble (if it exists)
    if reuses_precomputed_ensemble(formulation_dict):
        logger.debug(f"Loading pre-computed ensemble from {ensemble_file.name}")

        ds_stats = xr.open_dataset(ensemble_file, engine="h5netcdf", chunks={'feature_id': 'auto'})

        # Weighting lives in build_stats, which this branch skips, so the run
        # otherwise succeeds looking weighted
        if weight_plan is not None:
            logger.warning(
                f"ENSEMBLE WEIGHTS NOT APPLIED: stats.weights configures weights "
                f"from {weight_plan.config.file}, but this domain reuses the "
                f"pre-computed ensemble {ensemble_file}. Its statistics were "
                f"built elsewhere, and weighting is applied where they are built, "
                f"so the mean in the output is whatever that file already held -- "
                f"weighted or not. Remove or repoint the pre-computed ensemble to "
                f"rebuild the statistics with these weights."
            )

        if 'time' in ds_stats.coords:
            t_min = pd.to_datetime(ds_stats.time.min().values)
            t_max = pd.to_datetime(ds_stats.time.max().values)

    # Load Raw Formulation Files (if they exist)
    if raw_files:
        logger.debug(f"Loading {len(raw_files)} raw formulation files (strategy=mfdataset)...")
        
        combined_ds = xr.open_mfdataset(
            paths=list(raw_files.values()),
            combine='nested',
            concat_dim="formulation",
            engine="h5netcdf",
            chunks={},
            parallel=True,
        )
        
        # Assign coordinates properly
        combined_ds = combined_ds.assign_coords(formulation=list(raw_files.keys()))

        if t_min is None:
            t_min = pd.to_datetime(combined_ds.time.min().values)
            t_max = pd.to_datetime(combined_ds.time.max().values)

    # Calculate Stats
    if ds_stats is None and combined_ds is not None:
        weights, applied = (
            (None, None) if weight_plan is None
            else resolve_domain_weights(weight_plan, combined_ds)
        )
        ds_stats = build_stats(combined_ds, raw_files, stats_config, weights=weights)

        # Recorded on the dataset itself, so it travels to the NetCDF the
        # pipeline writes from it.  Both branches: see teval.weights.provenance
        ds_stats.attrs.update(weighting_attrs(applied))

    elif ds_stats is None and combined_ds is None:
        raise ValueError("No ensemble file or raw formulation files found to process.")
        
    return ds_stats, combined_ds, t_min, t_max

# Functions for calculating metrics based on the loaded domain data
def _calc_row(sim_series: pd.Series, obs_series: pd.Series, source_name: str, 
              fid: int, gage_id_str: str, lat: float, lon: float, 
              metrics_config: MetricsConfig) -> dict:
    """Helper to calculate metrics for a single simulation series vs observations."""
    if sim_series.index.tz is not None:
        sim_series = sim_series.tz_localize(None)
        
    df_aligned = pd.concat([obs_series, sim_series], axis=1, join="inner").dropna()
    if len(df_aligned) < 5:
        logger.debug(
            f"Skipping gage {gage_id_str} / {source_name}: only {len(df_aligned)} "
            "aligned obs+sim rows after inner join (need ≥5). "
            f"Sim index type: {type(sim_series.index).__name__}, "
            f"obs index type: {type(obs_series.index).__name__}."
        )
        return None
    
    obs_aligned = df_aligned.iloc[:, 0]
    sim_aligned = df_aligned.iloc[:, 1]

    # Cast to float64 before metric functions.
    # T-Route NCs are written as float32; np.sum/np.mean on float32 return float32,
    # and in numpy >= 2.0 np.float32 is not a subclass of Python float.
    # np.corrcoef always promotes to float64, which can lead to skill scores being silently dropped.
    obs_aligned = obs_aligned.astype(np.float64)
    sim_aligned = sim_aligned.astype(np.float64)

    row = {'feature_id': int(fid), 'gage_id': str(gage_id_str), 'lat': lat, 'lon': lon, 'source': source_name}
    
    for metric in metrics_config.variables:
        row[metric] = det.calculate_deterministic_metric(obs_aligned, sim_aligned, metric_name=metric)
    
    return row
            
def calculate_metrics(domain_data: Dict[str, Dict], 
                      metrics: MetricsConfig,
                      sim_var: str = 'streamflow_mean'
                      ) -> Dict[str, Dict]:
    """
    Given the loaded domain data, calculates specified metrics and returns results in a structured format.
    Dynamically sums streamflow for gages that receive flow from multiple feature_ids.
    """
    metric_results = []
    
    obs_df = domain_data.get('gage_obs', pd.DataFrame())
    gdf = domain_data.get('hydrofabric', gpd.GeoDataFrame())
    ds_stats = domain_data.get('formulations', {}).get('combined', xr.Dataset())
    ds_ensemble = domain_data.get('formulations', {}).get('ensemble_members', xr.Dataset())
    gage_to_fids = domain_data.get('gage_to_fids', {})
    
    if not obs_df.empty and not gdf.empty and len(ds_stats.dims) > 0 and gage_to_fids:
        
        # Filter to gages we actually have observations for
        valid_gages = [str(g) for g in gage_to_fids.keys() if str(g) in obs_df.columns]
        
        # Pull data from xarray
        all_req_fids = []
        for g in valid_gages:
            all_req_fids.extend(gage_to_fids[g])
        all_req_fids = list(set(all_req_fids)) 
        valid_fids_in_ds = [f for f in all_req_fids if f in ds_stats.feature_id.values]
        
        logger.debug("Converting Xarray slice to Pandas and summing multi-flowpath gages...")
        sim_df_mean_raw = ds_stats[sim_var].sel(feature_id=valid_fids_in_ds).to_pandas()
        
        # Sum multiple upstream flow values
        mean_dict = {}
        for g in valid_gages:
            g_fids = [f for f in gage_to_fids[g] if f in valid_fids_in_ds]
            if g_fids:
                mean_dict[g] = sim_df_mean_raw[g_fids].sum(axis=1)
        sim_df_mean = pd.DataFrame(mean_dict)
        
        # If calculating per formulation, do the exact same dictionary aggregation
        sim_df_members = None
        formulation_names = []
        
        if metrics.per_formulation and ds_ensemble is not None:
            formulation_names = ds_ensemble.formulation.values
            sim_df_members = {}
            for form in formulation_names:
                raw_members = ds_ensemble['streamflow'].sel(formulation=form, feature_id=valid_fids_in_ds).to_pandas()
                
                # Build a dictionary for this formulation
                form_dict = {}
                for g in valid_gages:
                    g_fids = [f for f in gage_to_fids[g] if f in valid_fids_in_ds]
                    if g_fids:
                        form_dict[g] = raw_members[g_fids].sum(axis=1)
                        
                sim_df_members[form] = pd.DataFrame(form_dict)
        
        # Calculate final metrics
        for gage_id_str in valid_gages:
            if gage_id_str not in sim_df_mean.columns: continue
            
            obs_series = obs_df[gage_id_str].tz_localize(None) if obs_df[gage_id_str].index.tz else obs_df[gage_id_str]
            
            # Use the primary fid for assigning spatial data (lat/lon)
            primary_fid = gage_to_fids[gage_id_str][0]
            geom = gdf.loc[primary_fid].geometry.centroid if primary_fid in gdf.index else None
            lat, lon = (geom.y, geom.x) if geom else (None, None)

            # Calculate Ensemble Mean
            row_mean = _calc_row(
                sim_series=sim_df_mean[gage_id_str], 
                obs_series=obs_series, 
                source_name="ensemble_mean", 
                fid=primary_fid, 
                gage_id_str=gage_id_str, 
                lat=lat, 
                lon=lon, 
                metrics_config=metrics
            )
            if row_mean: metric_results.append(row_mean)
            
            # Calculate Per Formulation
            if metrics.per_formulation and sim_df_members:
                for form in formulation_names:
                    row_form = _calc_row(
                        sim_series=sim_df_members[form][gage_id_str], 
                        obs_series=obs_series, 
                        source_name=form, 
                        fid=primary_fid, 
                        gage_id_str=gage_id_str, 
                        lat=lat, 
                        lon=lon, 
                        metrics_config=metrics
                    )
                    if row_form: metric_results.append(row_form)
    
    return metric_results


# Functions for producing visualizations based on the loaded domain data and calculated metrics
def _render_single_hydrograph(gage, fids, nexus_id, ds_stats, obs_df, viz, stats, ds_ensemble, hydro_dir, metrics_df):
    """Joblib worker function to render a single hydrograph."""
    valid_fids = [f for f in fids if f in ds_stats.feature_id.values]
    if not valid_fids: return
    
    # Aggregate the flow across all incoming feature_ids
    ds_stats_summed = ds_stats.sel(feature_id=valid_fids).sum(dim='feature_id', keep_attrs=True)
    
    ds_ensemble_summed = None
    if viz.hydrographs.plot_members and ds_ensemble is not None:
        ds_ensemble_summed = ds_ensemble.sel(feature_id=valid_fids).sum(dim='feature_id', keep_attrs=True)
        
    fig, ax = plt.subplots(figsize=(12, 6))
    
    series_obs = None
    if obs_df is not None and gage in obs_df.columns:
        series_obs = obs_df[gage]
        
    # Isolate metrics for just this specific Gage ID
    fid_metrics = None
    if metrics_df is not None and not metrics_df.empty:
        fid_metrics = metrics_df[metrics_df['gage_id'].astype(str) == str(gage)]
    
    tviz.hydrograph(
        stats_ds=ds_stats_summed, 
        gage_id=gage,
        nexus_id=nexus_id,
        ax=ax, 
        obs_series=series_obs,
        plot_uncertainty=viz.hydrographs.plot_uncertainty,
        plot_members=viz.hydrographs.plot_members,
        ensemble_ds=ds_ensemble_summed,
        quantiles=stats.quantiles,
        metrics_df=fid_metrics
    )
    fig.savefig(hydro_dir / f"hydrograph_gage_{gage}.png", bbox_inches='tight')
    plt.close(fig)
    gc.collect()


def produce_domain_specific_visualizations(domain_data: Dict, viz: VizConfig, io: IOConfig, stats: StatsConfig):
    """
    Given the loaded domain data and visualization config, produces and saves visualizations.
    """
    ds_stats = domain_data.get('formulations', {}).get('combined', xr.Dataset())
    ds_ensemble = domain_data.get('formulations', {}).get('ensemble_members', xr.Dataset())
    gdf = domain_data.get('hydrofabric', gpd.GeoDataFrame())
    obs_df = domain_data.get('gage_obs', pd.DataFrame())
    
    # Extract calculated metrics to pass to the plots
    metrics_list = domain_data.get('metrics', [])
    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        # Ensure gage_id is a string.
        metrics_df['gage_id'] = metrics_df['gage_id'].astype(str)
    else:
        metrics_df = None
    
    # Hydrographs
    if viz.hydrographs.enabled:
        hydro_dir = io.output_dir / "hydrographs"
        hydro_dir.mkdir(parents=True, exist_ok=True)
        
        target_ids = viz.hydrographs.target_ids
        gage_to_fids = domain_data.get('gage_to_fids', {})
        gage_to_nexus = domain_data.get('gage_to_nexus', {})
        
        # Determine which gages to plot
        if not target_ids:
            valid_gages = [str(g) for g in gage_to_fids.keys() if str(g) in [str(c) for c in obs_df.columns]]
        else:
            # Match user input if they provided specific gage IDs
            valid_gages = [g for g in gage_to_fids.keys() if any(str(t) in str(g) for t in target_ids)]
            if not valid_gages: valid_gages = list(gage_to_fids.keys())[:5]
            
        # Collect all required FIDs to pre-load into memory
        required_fids = []
        for g in valid_gages:
            required_fids.extend(gage_to_fids[g])
        required_fids = list(set(required_fids))
        valid_targets = np.intersect1d(required_fids, ds_stats.feature_id.values)
        
        with Timer(f"Pre-load Hydrograph Data ({len(valid_gages)} gages)", category="loading"):
            logger.debug(f"Pre-loading data for {len(valid_gages)} gage hydrographs into RAM...")
            ds_stats_subset = ds_stats.sel(feature_id=valid_targets).compute()

        ds_ensemble_subset = None
        if viz.hydrographs.plot_members and ds_ensemble is not None:
            if 'feature_id' in ds_ensemble.dims:
                with Timer("Pre-load Ensemble Members", category="loading"):
                    ds_ensemble_subset = ds_ensemble.sel(feature_id=valid_targets).compute()

        logger.debug(f"Generating {len(valid_gages)} hydrographs in parallel...")
        
        with Timer("Plotting Hydrographs", category="visualization"):
            n_cores = max(1, multiprocessing.cpu_count() - 1)
            Parallel(n_jobs=n_cores)(
                delayed(_render_single_hydrograph)(
                    gage, 
                    gage_to_fids[gage],
                    gage_to_nexus.get(gage),
                    ds_stats_subset, 
                    obs_df, 
                    viz, 
                    stats, 
                    ds_ensemble_subset, 
                    hydro_dir, 
                    metrics_df
                ) for gage in valid_gages
            )
    
    # Animation
    if viz.animation.enabled:
        logger.info("Generating Animation...")
        anim_dir = io.output_dir / "animations"
        anim_dir.mkdir(parents=True, exist_ok=True)

        full_nc_path = domain_data["formulations"].get("_full_nc_path")
        if full_nc_path:
            with Timer("Open Full Dataset for Animation", category="loading"):
                logger.info(f"Opening full domain dataset for animation: {full_nc_path}")
                ds_anim_full = xr.open_dataset(
                    str(full_nc_path), engine="h5netcdf",
                    chunks={"time": "auto", "feature_id": -1},
                )
        else:
            logger.warning(
                "No _full_nc_path in domain_data; animation will use the current "
                "'combined' dataset (may contain only gage-subset features)."
            )
            ds_anim_full = ds_stats

        # Filter by stream order
        if 'order' in gdf.columns:
            gdf_anim = gdf[gdf['order'] >= viz.animation.min_stream_order]
        else:
            logger.warning("Stream order not found in geopackage, using all paths.")
            gdf_anim = gdf

        # Extract valid feature IDs and drop duplicates
        anim_fids = gdf_anim.index.values
        common_anim_ids = np.intersect1d(anim_fids, ds_anim_full.feature_id.values)
        
        gdf_anim = gdf_anim.loc[common_anim_ids]
        gdf_anim = gdf_anim[~gdf_anim.index.duplicated(keep='first')]
        
        # Time subsetting (e.g., '1H', '1D', '3D', '1W')
        time_step_str = str(viz.animation.time_step).strip().upper()
        
        try:
            # If the user passed a raw integer (e.g., "168")
            step = int(time_step_str)
        except ValueError:
            # If the user passed a string, calculate the dataset's native time resolution
            if len(ds_anim_full.time) >= 2:
                # Get difference between first two time steps
                dt_native = pd.Timedelta(ds_anim_full.time.values[1] - ds_anim_full.time.values[0])

                if time_step_str.endswith('W'):
                    val = int(time_step_str[:-1]) if len(time_step_str) > 1 else 1
                    dt_target = pd.Timedelta(days=val * 7)
                else:
                    dt_target = pd.to_timedelta(time_step_str)

                # Calculate how many integer steps to jump per frame
                step = max(1, int(dt_target / dt_native))
            else:
                step = 1

        ds_anim_sliced = ds_anim_full.isel(time=slice(0, None, step))
        ds_anim_sliced = ds_anim_sliced.sel(feature_id=common_anim_ids).sortby('feature_id')
        
        # Generate
        domain_name = domain_data.get('gage_obs', {}).get('domain_name', ['domain'])[0]
        out_gif = anim_dir / f"streamflow_animation_{domain_name}.gif"
        
        with Timer("Generating GIF Animation", category="visualization"):
            tanim.animate_network(
                gdf=gdf_anim,
                stats_ds=ds_anim_sliced,
                output_path=str(out_gif),
                var_name=viz.animation.variable,
                fps=viz.animation.fps,
                add_basemap=True
            )