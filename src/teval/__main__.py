"""Command-line entry point for the teval pipeline."""

import argparse
import logging
import os
import sys

import pandas as pd

from teval.config import TevalConfig, generate_default_config, generate_config_help
from teval.io import initialize_domains
from teval.pipeline import (
    configure_dask, get_worker_count, run_domain, run_skill_maps, run_interactive_map,
)
from teval.utils import Timer, configure_logging, configure_timing, print_timing_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Parse CLI arguments, load config, and run the full teval pipeline."""
    parser = argparse.ArgumentParser(description="Run the TEVAL Evaluation Pipeline")
    parser.add_argument(
        "-c", "--config", type=str, default="teval_default_config.yaml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--init", action="store_true",
        help="Generate a default 'teval_config.yaml' in the current directory.",
    )
    parser.add_argument(
        "--help-config", action="store_true",
        help="Print a detailed guide of all configuration parameters.",
    )

    args = parser.parse_args()

    if args.init:
        generate_default_config("teval_default_config.yaml")
        print("Generated default configuration: teval_default_config.yaml")
        sys.exit(0)
    if args.help_config:
        print(generate_config_help())
        sys.exit(0)

    # ------------------------------------------------------------------
    # Setup                                                             
    # ------------------------------------------------------------------
    config = TevalConfig.from_yaml(args.config)
    logger.info(f"Loaded config from: {args.config}")

    configure_logging(getattr(config.system, "logging_level", "INFO"))
    configure_timing(getattr(config.system, "timing", "simple"))
    logger.info(
        f"Runtime: logging={getattr(config.system, 'logging_level', 'INFO')}, "
        f"timing={getattr(config.system, 'timing', 'simple')}"
    )

    # xr.open_mfdataset(parallel=True) uses Dask's threaded scheduler.
    # Disable POSIX file locking so multiple threads can safely read HDF5 files
    # concurrently.
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    configure_dask(config)
    logger.info(f"Parallel workers: {get_worker_count(config)}")

    # ------------------------------------------------------------------ #
    # Domain processing                                                   #
    # ------------------------------------------------------------------ #
    run_domain_processing = any([
        config.stats.enabled,
        config.metrics.enabled,
        config.viz.hydrographs.enabled,
        config.viz.animation.enabled,
    ])

    metrics_df = pd.DataFrame()
    out_csv = config.io.output_dir / (config.io.metrics_output_file or "metrics.csv")
    domain_map = {}

    if run_domain_processing:
        with Timer("Domain Discovery", category="discovery"):
            domain_map = initialize_domains(config.io, config.stats, config.metrics, config.viz)
        
        n = len(domain_map)
        logger.info(f"Found {n} domain(s) to process.")
        metrics_list = []

        for domain_idx, (domain_name, domain_dict) in enumerate(domain_map.items(), 1):
            logger.info(f"[{domain_idx}/{n}] {domain_name}")
            metric_rows = run_domain(domain_name, domain_dict, config)
            if metric_rows:
                metrics_list.append(pd.DataFrame(metric_rows))

        if metrics_list:
            metrics_df = pd.concat(metrics_list, ignore_index=True)
            if not metrics_df.empty:
                metrics_df["gage_id"] = metrics_df["gage_id"].astype(str)
                metrics_df.to_csv(out_csv, index=False)
                logger.info(f"Metrics saved -> {out_csv}")

    else:
        logger.info("Domain processing disabled. Checking for existing metrics CSV.")
        if out_csv.exists():
            logger.info(f"Loading pre-calculated metrics from {out_csv}")
            metrics_df = pd.read_csv(out_csv, dtype={"gage_id": str})
        elif config.viz.skill_maps.enabled or config.viz.interactive_map.enabled:
            logger.error(
                f"Cannot find metrics file {out_csv}. "
                "Run with metrics.enabled: true first."
            )

    # ------------------------------------------------------------------
    # Post-processing: skill maps and interactive map                   
    # ------------------------------------------------------------------
    if not metrics_df.empty:
        if config.viz.skill_maps.enabled:
            run_skill_maps(metrics_df, config, domain_map if run_domain_processing else None)

        if config.viz.interactive_map.enabled:
            run_interactive_map(metrics_df, config)

    # ------------------------------------------------------------------
    # Done                                                              
    # ------------------------------------------------------------------
    print_timing_summary()
    logger.info("TEVAL run complete.")


if __name__ == "__main__":
    main()