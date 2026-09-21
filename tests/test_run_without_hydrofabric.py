"""
The run that needs no hydrofabric GeoPackage.

``initialize_domains`` loads a GeoPackage only for the consumers that need the
gage crosswalk -- statistics, metrics, and the interactive map.  A run with all
three disabled and only hydrographs asked for takes the other path, where
every domain's ``hydrofabric`` is ``None``.
That is the path the ensemble orchestrator's configuration always takes.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from teval import __main__ as teval_main
from teval.config import IOConfig, MetricsConfig, StatsConfig, VizConfig
from teval.io.discovery import initialize_domains

#: Domain name, which discovery reads from the run directory suffix.
DOMAIN = "02020202"

#: Formulation name to the constant its file carries.
FORMULATIONS = {"cfe": 100.0, "noahowp": 200.0}

FEATURE_IDS = [101, 201]
N_TIMES = 4
TIMES = pd.date_range("2020-06-01", periods=N_TIMES, freq="h")


@pytest.fixture(autouse=True)
def restore_global_state():
    """
    Put the logger levels and Dask's worker count back after each run.

    ``main`` sets all three globally; left alone they would outlive this
    module and change what every later test sees.
    """
    root_level = logging.getLogger().level
    teval_level = logging.getLogger("teval").level
    with dask.config.set(num_workers=dask.config.get("num_workers", None)):
        yield
    logging.getLogger().setLevel(root_level)
    logging.getLogger("teval").setLevel(teval_level)


# --------------------------------------------------------------------- #
# Inputs: T-Route outputs and nothing else                              #
# --------------------------------------------------------------------- #
def _write_troute_outputs(troute_dir: Path) -> None:
    """One T-Route output directory per formulation, named for discovery."""
    for name, value in FORMULATIONS.items():
        run_dir = troute_dir / f"{name}_{DOMAIN}_output"
        run_dir.mkdir(parents=True)
        xr.Dataset(
            {
                "streamflow": (
                    ("time", "feature_id"),
                    np.full((N_TIMES, len(FEATURE_IDS)), value),
                )
            },
            coords={"time": TIMES, "feature_id": FEATURE_IDS},
        ).to_netcdf(run_dir / "troute_output.nc", engine="h5netcdf")


@pytest.fixture
def run_inputs(tmp_path) -> dict:
    """
    The inputs a hydrograph-only run is given.

    ``hydrofabric_dir`` is a real but empty directory, which is what the
    orchestrator's rendered configuration points at: the directory exists, and
    no GeoPackage in it is ever meant to be read.
    """
    troute_dir = tmp_path / "troute"
    _write_troute_outputs(troute_dir)
    hydrofabric_dir = tmp_path / "hydrofabric"
    hydrofabric_dir.mkdir()

    return {
        "root": tmp_path,
        "troute_dir": troute_dir,
        "hydrofabric_dir": hydrofabric_dir,
        "output_dir": tmp_path / "output",
    }


def _no_gpkg_config_dict(run_inputs: dict) -> dict:
    """
    Stats and metrics at their disabled defaults and the interactive map off;
    hydrographs stay on so the run still processes its domains.
    """
    return {
        "io": {
            "troute_netcdf_dir": str(run_inputs["troute_dir"]),
            "hydrofabric_dir": str(run_inputs["hydrofabric_dir"]),
            "output_dir": str(run_inputs["output_dir"]),
            "directory_naming": "suffix",
            "per_domain_output": True,
            "auto_download_usgs": False,
        },
        "system": {"cpu": 1, "logging_level": "DEBUG", "timing": "none"},
        "viz": {
            "hydrographs": {"enabled": True},
            "skill_maps": {"enabled": False},
            "interactive_map": {"enabled": False},
            "animation": {"enabled": False},
        },
    }


def _disabled_configs(run_inputs: dict) -> tuple:
    """``(io, stats, metrics, viz)`` for the hydrograph-only run."""
    config = _no_gpkg_config_dict(run_inputs)
    return (
        IOConfig(**config["io"]),
        StatsConfig(),
        MetricsConfig(),
        VizConfig(**config["viz"]),
    )


# --------------------------------------------------------------------- #
# Discovery                                                             #
# --------------------------------------------------------------------- #
def test_discovery_completes_with_stats_metrics_and_map_all_disabled(run_inputs):
    """With stats, metrics and the map off, discovery still returns the domain."""
    io, stats, metrics, viz = _disabled_configs(run_inputs)
    assert not (stats.enabled or metrics.enabled or viz.interactive_map.enabled)

    domain_map = initialize_domains(io, stats, metrics, viz)

    assert DOMAIN in domain_map
    assert domain_map[DOMAIN]["hydrofabric"] is None


# --------------------------------------------------------------------- #
# The whole run                                                         #
# --------------------------------------------------------------------- #
def test_a_hydrograph_only_run_writes_its_ensemble_netcdf(run_inputs, monkeypatch):
    """
    The run reaches its primary output.  Nothing is stubbed, so this also covers
    the steps after discovery that receive a domain with no hydrofabric.
    """
    config_path = run_inputs["root"] / "teval_config.yaml"
    with open(config_path, "w") as handle:
        yaml.safe_dump(_no_gpkg_config_dict(run_inputs), handle, sort_keys=False)

    monkeypatch.setattr(sys, "argv", ["teval", "-c", str(config_path)])
    teval_main.main()

    written = run_inputs["output_dir"] / DOMAIN / f"{DOMAIN}_ensemble.nc"
    assert written.exists(), f"the run wrote no ensemble NetCDF at {written}"

    with xr.open_dataset(written, engine="h5netcdf") as ds:
        assert sorted(int(f) for f in ds.feature_id.values) == FEATURE_IDS
