"""
teval.pipeline

Single-domain execution and post-processing orchestration.

This module owns all compute logic that was previously inline in __main__.
__main__ calls these functions; it never touches dask, xarray, or
ProcessPoolExecutor directly.

Public API
----------
run_domain(domain_name, domain_dict, config)
    Full lifecycle for one domain: time-slice → compute/write → metrics → viz.
    Returns a list of metric-row dicts (empty list if metrics are disabled).

run_skill_maps(metrics_df, config, domain_map)
    Build all skill-map figures in parallel using ProcessPoolExecutor.

run_interactive_map(metrics_df, config)
    Render the Folium HTML interactive map.

get_worker_count(config)
    Return the number of workers to use: system.cpu, or the CPUs allocated to
    this job when it is -1.

configure_dask(config)
    Size Dask's threaded scheduler to get_worker_count(config).
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

import dask
import pandas as pd

from teval.config import TevalConfig
from teval import workflow
from teval.utils import Timer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker count helper
# ---------------------------------------------------------------------------

def get_worker_count(config) -> int:
    """
    Return the number of parallel workers for this run.

    An explicit ``system.cpu`` is used as given; keeping it within a Slurm
    allocation is up to the user.  -1 means the CPUs this job was allocated:
    SLURM_CPUS_PER_TASK inside a Slurm job, otherwise the cores this process
    may run on.
    """
    if config.system.cpu != -1:
        return config.system.cpu
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return int(slurm_cpus)
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def configure_dask(config) -> None:
    """Size Dask's threaded scheduler, used by every compute, to the run."""
    dask.config.set(num_workers=get_worker_count(config))


# ---------------------------------------------------------------------------
# Dask compute + NC write
# ---------------------------------------------------------------------------

def compute_and_write(
    domain_name: str,
    domain_data: Dict,
    domain_dict: Dict,
    config: TevalConfig,
) -> Dict:
    """
    Trigger the Dask compute graph and optionally write the ensemble NetCDF.

    Two branches depending on config.system.stream_to_disk:

    stream_to_disk = True (default)
        Single dask.compute() pass that simultaneously:
        - writes the full ensemble stats to {domain_name}_ensemble.nc
        - extracts the gage-subset into RAM for metrics and hydrographs

    stream_to_disk = False
        Skips the NC write; extracts only the gage subset into RAM.

    This function mutates domain_data in-place and returns it.

    Parameters
    ----------
    domain_name:
        Domain identifier used for logging and file naming.
    domain_data:
        The dict produced by workflow.load_domain_data().
    domain_dict:
        The raw entry from the domain map (used to check if a pre-computed
        ensemble file already exists).
    config:
        Full TevalConfig instance.

    Returns
    -------
    dict
        The mutated domain_data dict with datasets replaced by
        computed results.
    """
    build_from_raw = not workflow.reuses_precomputed_ensemble(
        domain_dict["formulations"]
    )

    if build_from_raw:
        ds_stats_lazy   = domain_data["formulations"]["combined"]
        ds_members_lazy = domain_data["formulations"].get("ensemble_members")

        valid_fids  = workflow.get_gage_fids(domain_data, config.viz)
        need_members = (
            (config.metrics.per_formulation or config.viz.hydrographs.plot_members)
            and ds_members_lazy is not None
            and "feature_id" in ds_members_lazy.dims
        )

        write_flag = config.system.stream_to_disk is not False

        if write_flag:
            # Determine output path
            if config.io.per_domain_output:
                domain_out_dir = config.io.output_dir / domain_name
            else:
                domain_out_dir = config.io.output_dir
            domain_out_dir.mkdir(parents=True, exist_ok=True)
            out_nc = domain_out_dir / f"{domain_name}_ensemble.nc"

            n_feats = ds_stats_lazy.sizes.get("feature_id", 0)
            logger.debug(
                f"[{domain_name}] Writing {n_feats} features to disk and "
                f"extracting {len(valid_fids)} gage FIDs into RAM..."
            )

            with Timer(
                f"[{domain_name}] Compute + Write + Extract Gage Subset",
                category="output",
            ):
                write_task        = ds_stats_lazy.astype("float32").to_netcdf(
                    out_nc, engine="h5netcdf", compute=False
                )
                gage_stats_lazy   = ds_stats_lazy.sel(feature_id=valid_fids)
                compute_targets   = [write_task, gage_stats_lazy]
                if need_members:
                    compute_targets.append(ds_members_lazy.sel(feature_id=valid_fids))
                results = dask.compute(*compute_targets)

            domain_data["formulations"]["combined"] = results[1]
            domain_data["formulations"]["ensemble_members"] = results[2] if need_members else None
            domain_data["formulations"]["_full_nc_path"] = out_nc
            logger.debug(f"[{domain_name}] Ensemble written -> {out_nc}")

        else:
            logger.debug(
                f"[{domain_name}] Extracting {len(valid_fids)} gage FIDs into RAM "
                "(stream_to_disk=false, skipping NC write)..."
            )
            with Timer(f"[{domain_name}] Compute Gage Subset", category="loading"):
                compute_targets = [ds_stats_lazy.sel(feature_id=valid_fids)]
                if need_members:
                    compute_targets.append(ds_members_lazy.sel(feature_id=valid_fids))
                results = dask.compute(*compute_targets)

            domain_data["formulations"]["combined"] = results[0]
            domain_data["formulations"]["ensemble_members"] = results[1] if need_members else None
            domain_data["formulations"]["_full_nc_path"] = None

    else:
        # Pre-computed ensemble: record its path for animation, then pull the
        # gage subset into RAM for metrics / hydrographs.
        domain_data["formulations"]["_full_nc_path"] = (
            domain_dict["formulations"]["ensemble_file"]
        )
        valid_fids = workflow.get_gage_fids(domain_data, config.viz)
        if len(valid_fids) > 0:
            with Timer(
                f"[{domain_name}] Extract Gage Subset (pre-computed)",
                category="loading",
            ):
                ds_full = domain_data["formulations"]["combined"].chunk(
                    {"time": -1, "feature_id": "auto"}
                )
                domain_data["formulations"]["combined"] = (
                    ds_full.sel(feature_id=valid_fids).compute()
                )

    return domain_data


# ---------------------------------------------------------------------------
# Single-domain processing
# ---------------------------------------------------------------------------

def run_domain(
    domain_name: str,
    domain_dict: Dict,
    config: TevalConfig,
) -> List[Dict]:
    """
    Execute the full pipeline for a single domain.

    Steps
    -----
    1. Load data (hydrofabric, observations, lazy formulation datasets)
    2. Apply optional time slice
    3. Compute ensemble stats and write NC (compute_and_write)
    4. Calculate metrics
    5. Produce per-domain visualisations (hydrographs, animation)

    Parameters
    ----------
    domain_name:
        Domain identifier.
    domain_dict:
        Entry from the domain map produced by io.initialize_domains().
    config:
        Full TevalConfig instance.

    Returns
    -------
    list[dict]
        Metric rows for this domain (empty list if metrics are disabled or
        no valid obs/sim overlap exists).
    """
    # ------------------------------------------------------------------
    # Load domain data
    # ------------------------------------------------------------------
    with Timer(f"[{domain_name}] Load Data", category="loading"):
        domain_data = workflow.load_domain_data(domain_dict, config.io, config.stats)

    # ------------------------------------------------------------------ 
    # Time slice
    # ------------------------------------------------------------------
    if config.data.time_slice and len(config.data.time_slice) == 2:
        t_start, t_end = config.data.time_slice
        logger.debug(f"[{domain_name}] Slicing time -> {t_start} to {t_end}.")
        combined = domain_data["formulations"].get("combined")
        members  = domain_data["formulations"].get("ensemble_members")
        if combined is not None:
            domain_data["formulations"]["combined"] = combined.sel(
                time=slice(t_start, t_end)
            )
        if members is not None:
            domain_data["formulations"]["ensemble_members"] = members.sel(
                time=slice(t_start, t_end)
            )

    # ------------------------------------------------------------------
    # Compute + write
    # ------------------------------------------------------------------
    domain_data = compute_and_write(domain_name, domain_data, domain_dict, config)

    # ------------------------------------------------------------------
    # Calculate metrics
    # ------------------------------------------------------------------
    metric_rows: List[Dict] = []
    if config.metrics.enabled:
        n_gages = len(domain_data.get("gage_to_fids", {}))
        logger.debug(f"[{domain_name}] Computing metrics for {n_gages} gages...")
        with Timer(f"[{domain_name}] Metrics", category="metrics"):
            metric_rows = workflow.calculate_metrics(domain_data, config.metrics)
            domain_data["metrics"] = metric_rows

    # ------------------------------------------------------------------
    # Per-domain visualisations
    # ------------------------------------------------------------------
    viz_enabled = config.viz.hydrographs.enabled or config.viz.animation.enabled
    if viz_enabled:
        with Timer(f"[{domain_name}] Visualizations", category="visualization"):
            workflow.produce_domain_specific_visualizations(
                domain_data, config.viz, config.io, config.stats,
                get_worker_count(config),
            )

    logger.debug(f"[{domain_name}] Done.")
    return metric_rows


# ---------------------------------------------------------------------------
# Picklable top-level worker (needed by ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _run_skill_map_task(task_spec):
    """
    Top-level picklable worker for parallel skill-map rendering.
    ``task_spec`` is a ``(fn_name, kwargs)`` tuple.
    """
    import logging as _logging
    _logging.basicConfig(level=_logging.WARNING)
    from teval.viz import static as tviz
    fn_name, kwargs = task_spec
    getattr(tviz, fn_name)(**kwargs)


# ---------------------------------------------------------------------------
# Post-processing: skill maps and interactive map
# ---------------------------------------------------------------------------

def run_skill_maps(
    metrics_df: pd.DataFrame,
    config: TevalConfig,
    domain_map: Optional[Dict] = None,
) -> None:
    """
    Build all configured skill-map figures in parallel.
    Handles score maps, winner maps, boxplots, and VPU breakdowns.

    Parameters
    ----------
    metrics_df:
        Combined metrics DataFrame from all processed domains.
    config:
        Full TevalConfig instance.
    domain_map:
        The domain map returned by io.initialize_domains().  Required
        only when config.io.per_domain_output is True (used to filter
        metrics to tailwater gages).  Pass None or {} otherwise.
    """
    from teval.viz.static import build_vpu_map

    skill_dir = config.io.output_dir / "skill_maps"
    skill_dir.mkdir(parents=True, exist_ok=True)

    # Filter to tailwater gages when per-domain output is enabled
    if config.io.per_domain_output and domain_map:
        tailwater_gages = set(domain_map.keys())
        n_before = len(metrics_df)
        metrics_df_skill = metrics_df[metrics_df["gage_id"].isin(tailwater_gages)].copy()
        logger.info(
            f"Skill maps: filtered to {len(tailwater_gages)} tailwater gages "
            f"({len(metrics_df_skill)} rows from {n_before} total)."
        )
    else:
        metrics_df_skill = metrics_df

    # Build VPU map if needed
    vpu_map: dict = {}
    if config.viz.skill_maps.vpu_breakdown:
        with Timer("Build VPU Map", category="loading"):
            vpu_map = build_vpu_map(config.io.hydrofabric_dir)
            logger.info(
                f"VPU map: {len(vpu_map)} gages mapped."
                if vpu_map else "No VPU map available; skipping breakdown."
            )

    # Assemble task list
    tasks: list = []
    for metric in config.viz.skill_maps.variables:
        if metric not in metrics_df_skill.columns:
            logger.warning(f"Metric '{metric}' not in metrics DataFrame. Skipping.")
            continue

        if config.viz.skill_maps.score_maps:
            for src in metrics_df_skill["source"].unique():
                tasks.append(("map_metrics", dict(
                    metrics_df=metrics_df_skill[metrics_df_skill["source"] == src],
                    variable=metric,
                    output_path=skill_dir / f"map_{metric}_{src}.png",
                    add_basemap=config.viz.skill_maps.basemap,
                    title=f"{metric.upper()} — {src}",
                )))

        if config.viz.skill_maps.winner_maps:
            tasks.append(("plot_winner_map", dict(
                metrics_df=metrics_df_skill,
                metric=metric,
                output_path=skill_dir / f"map_winner_{metric}.png",
                add_basemap=config.viz.skill_maps.basemap,
            )))

        if config.viz.skill_maps.boxplots:
            tasks.append(("plot_boxplots", dict(
                metrics_df=metrics_df_skill,
                metric=metric,
                output_path=skill_dir / f"boxplot_{metric}.png",
            )))

        if config.viz.skill_maps.vpu_breakdown and vpu_map:
            tasks.append(("plot_vpu_breakdown", dict(
                metrics_df=metrics_df_skill,
                metric=metric,
                output_path=skill_dir / f"vpu_breakdown_{metric}.png",
                vpu_map=vpu_map,
            )))

    if not tasks:
        logger.info("No skill map tasks to run.")
        return

    n_workers = min(len(tasks), get_worker_count(config))
    logger.info(f"Rendering {len(tasks)} skill map(s) with {n_workers} workers...")

    with Timer(f"Skill Maps ({len(tasks)} total)", category="visualization"):
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_skill_map_task, t): t for t in tasks}
            for fut in as_completed(futures):
                fn_name, kw = futures[fut]
                out_path = kw.get("output_path", "?")
                try:
                    fut.result()
                    logger.info(f"  + {out_path.name}")
                except Exception as exc:
                    logger.error(f"  X {out_path.name}: {exc}")


def run_interactive_map(metrics_df: pd.DataFrame, config: TevalConfig) -> None:
    """
    Render the Folium interactive HTML metrics map.

    Parameters
    ----------
    metrics_df:
        Combined metrics DataFrame from all processed domains.
    config:
        Full TevalConfig instance.
    """
    from teval.viz.interactive import plot_interactive_metrics_map

    with Timer("Interactive Map", category="visualization"):
        plot_interactive_metrics_map(
            metrics_df,
            output_path=config.io.output_dir / "interactive_metrics_map.html",
        )