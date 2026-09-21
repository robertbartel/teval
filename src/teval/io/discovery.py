"""
teval.io.discovery

Scan T-Route output directories and build the domain map consumed by the
rest of the pipeline.

Public API
----------
parse_run_directory(path, naming)
    Extract (formulation_name, domain_name) from a directory path.

discover_formulation_files(troute_dir, naming)
    Scan a directory tree and return {domain: {formulation: Path}}.

initialize_domains(io, metrics, viz)
    Full initialisation pass: discover files, match hydrofabrics, prepare
    observation info.  Returns the domain_map dict consumed by the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

from teval.config import IOConfig, MetricsConfig, VizConfig, StatsConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Directory name parsing
# ---------------------------------------------------------------------------

def parse_run_directory(
    path: Path,
    naming: Literal["suffix", "parent"] = "suffix",
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract (formulation_name, domain_name) from a T-Route output directory.

    Parameters
    ----------
    path:
        Path to the *_output directory.
    naming:
        "suffix" — flat layout:
            {formulation}_{domain}_output/
            e.g. sloth_noahowp_cfe_s_12009000_output
                 -> formulation='sloth_noahowp_cfe_s', domain='12009000'

        "parent" — nested layout:
            {root}/{domain}/{formulation}_output/
            e.g. runs/12009000/cfe_s_output
                 -> formulation='cfe_s', domain='12009000'

    Returns
    -------
    (formulation_name, domain_name) or (None, None) if parsing fails.
    """
    name = path.name

    if not name.endswith("_output"):
        logger.warning(f"Directory '{name}' does not end in '_output'. Skipping.")
        return None, None

    base = name[:-7]  # strip '_output'

    if naming == "parent":
        domain_name = path.parent.name
        formulation_name = base
        if not formulation_name:
            logger.warning(f"Could not extract formulation from '{name}'.")
            return None, None
        return formulation_name, domain_name

    # Default: suffix mode
    parts = base.split("_")
    if len(parts) < 2:
        logger.warning(f"Directory '{name}' has too few underscore-separated parts.")
        return None, None

    domain_name = parts[-1]
    formulation_name = "_".join(parts[:-1])
    return formulation_name, domain_name


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_formulation_files(troute_dir, naming: str = "suffix") -> dict:
    """
    Scan troute_dir for T-Route output directories.

    Returns
    -------
    dict
        {domain_name: {formulation_name: Path(nc_file)}}

    Parameters
    ----------
    troute_dir:
        Root directory to scan (str or Path).
    naming:
        Directory naming convention — "suffix" or "parent".
    """
    troute_dir = Path(troute_dir)
    result: dict = {}

    if naming == "suffix":
        subdirs = [p for p in troute_dir.iterdir()
                   if p.is_dir() and p.name.endswith("_output")]
        for folder in subdirs:
            form_name, domain_name = parse_run_directory(folder, naming="suffix")
            if not form_name or not domain_name:
                continue
            nc_files = list(folder.glob("*.nc"))
            if nc_files:
                result.setdefault(domain_name, {})[form_name] = nc_files[0]

    elif naming == "parent":
        for domain_dir in troute_dir.iterdir():
            if not domain_dir.is_dir():
                continue
            for form_dir in domain_dir.iterdir():
                if not form_dir.is_dir() or not form_dir.name.endswith("_output"):
                    continue
                form_name, domain_name = parse_run_directory(form_dir, naming="parent")
                if not form_name or not domain_name:
                    continue
                nc_files = list(form_dir.glob("*.nc"))
                if nc_files:
                    result.setdefault(domain_name, {})[form_name] = nc_files[0]
    else:
        logger.warning(f"discover_formulation_files: unknown naming='{naming}'.")

    return result


# ---------------------------------------------------------------------------
# Domain map initialization
# ---------------------------------------------------------------------------

def initialize_domains(io: IOConfig, stats: StatsConfig, metrics: MetricsConfig, viz: VizConfig) -> Dict:
    """
    Scan input directories, group files by domain, and return a prepared
    domain map ready for the pipeline.

    The returned dict has the shape:

        {
            domain_name: {
                "formulations": {
                    "raw_files": {form_name: Path, ...},
                    "ensemble_file": Path | None,
                },
                "hydrofabric": Path | None,
                "gage_obs": {"domain_name": [...], "obs_file": [...]},
            },
            ...
        }
    """
    load_gpkgs = stats.enabled or metrics.enabled or viz.interactive_map.enabled
    fetch_obs = metrics.enabled or viz.hydrographs.enabled

    domain_map: Dict = {}

    # Discover T-Route outputs
    if io.troute_netcdf_dir:
        naming = getattr(io, "directory_naming", "suffix")
        form_files = discover_formulation_files(io.troute_netcdf_dir, naming=naming)
        for domain_name, formulations in form_files.items():
            if domain_name not in domain_map:
                domain_map[domain_name] = _create_empty_domain_dict(
                    domain_name, io, load_gpkgs, fetch_obs
                )
            for form_name, nc_path in formulations.items():
                domain_map[domain_name]["formulations"]["raw_files"][form_name] = nc_path

    # Discover pre-computed ensemble files
    if io.ensemble_netcdf_dir:
        for e_file in io.ensemble_netcdf_dir.rglob("*.nc"):
            domain_name = (
                e_file.stem.split("_ensemble")[0]
                if "_ensemble" in e_file.stem
                else e_file.stem
            )
            if domain_name not in domain_map:
                domain_map[domain_name] = _create_empty_domain_dict(
                    domain_name, io, load_gpkgs, fetch_obs
                )
            domain_map[domain_name]["formulations"]["ensemble_file"] = e_file

    if not domain_map:
        logger.error("No valid inputs found in troute_netcdf_dir or ensemble_netcdf_dir.")

    return domain_map


def _create_empty_domain_dict(
    domain_name: str,
    io: IOConfig,
    load_gpkgs: bool,
    fetch_obs: bool,
) -> dict:
    """Initialise the dictionary structure for a new domain."""
    gpkg_path = None
    if load_gpkgs:
        if domain_name == "CONUS":
            def _case_insensitive(s: str) -> str:
                """Convert a string into a glob pattern that matches it case-insensitively."""
                return "".join(
                    f"[{c.lower()}{c.upper()}]" if c.isalpha() else c for c in s
                )
            pattern = f"*{_case_insensitive(domain_name)}*.gpkg"
            gpkgs = list(io.hydrofabric_dir.glob(pattern))
        else:
            gpkgs = list(io.hydrofabric_dir.glob(f"*{domain_name}*.gpkg"))
        gpkg_path = gpkgs[0] if gpkgs else None

    obs_info: dict = {}
    if fetch_obs:
        obs_file_path = (
            io.observations_file
            if (io.observations_file and io.observations_file.exists())
            else None
        )
        obs_info = {"domain_name": [domain_name], "obs_file": [obs_file_path]}

    return {
        "formulations": {"raw_files": {}, "ensemble_file": None},
        "hydrofabric": gpkg_path,
        "gage_obs": obs_info,
    }