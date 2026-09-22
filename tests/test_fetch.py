"""
``--fetch`` downloads what a run needs so the run can go offline.

The download itself is stubbed: what is under test is which observations are
planned, where they land, when a re-run skips them, and what an offline run
checks before it starts.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from teval import __main__ as teval_main
from teval import fetch as tfetch
from teval.config import TevalConfig
from teval.io import observations
from teval.io.observations import fetch_observations

DOMAIN = "02020202"
FEATURE_IDS = [101, 201]
TIMES = pd.date_range("2020-06-01", periods=4, freq="h")


def _write_troute_outputs(troute_dir, times=TIMES):
    for name in ("cfe", "noahowp"):
        run_dir = troute_dir / f"{name}_{DOMAIN}_output"
        run_dir.mkdir(parents=True)
        xr.Dataset(
            {"streamflow": (("time", "feature_id"), np.ones((len(times), len(FEATURE_IDS))))},
            coords={"time": times, "feature_id": FEATURE_IDS},
        ).to_netcdf(run_dir / "troute_output.nc", engine="h5netcdf")


def _config_dict(tmp_path, observations_file) -> dict:
    """A hydrograph-only run, which observes but loads no GeoPackage."""
    return {
        "io": {
            "troute_netcdf_dir": str(tmp_path / "troute"),
            "hydrofabric_dir": str(tmp_path / "hydrofabric"),
            "output_dir": str(tmp_path / "output"),
            "observations_file": str(observations_file) if observations_file else None,
            "directory_naming": "suffix",
            "per_domain_output": True,
        },
        "system": {"cpu": 1, "timing": "none"},
        "viz": {
            "hydrographs": {"enabled": True},
            "skill_maps": {"enabled": False},
            "interactive_map": {"enabled": False},
            "animation": {"enabled": False},
        },
    }


@pytest.fixture
def run_dir(tmp_path):
    _write_troute_outputs(tmp_path / "troute")
    (tmp_path / "hydrofabric").mkdir()
    return tmp_path


@pytest.fixture
def config(run_dir):
    return TevalConfig(**_config_dict(run_dir, run_dir / "obs" / "obs.parquet"))


@pytest.fixture
def downloads(monkeypatch):
    """Stub the NWIS download; record each call and return a frame for it."""
    calls = []

    def download(gage_ids, t_min, t_max, raise_errors=False):
        calls.append((list(gage_ids), t_min, t_max, raise_errors))
        index = pd.date_range(t_min, t_max + pd.Timedelta(hours=23), freq="h", tz="UTC")
        return pd.DataFrame({g: 1.5 for g in gage_ids}, index=index)

    monkeypatch.setattr(tfetch, "download_observations", download)
    return calls


# --------------------------------------------------------------------- #
# The plan                                                              #
# --------------------------------------------------------------------- #
def test_each_observing_domain_is_planned_over_its_files_dates(config):
    assert tfetch.plan_observations(config) == [
        tfetch.ObservationRequest(DOMAIN, (DOMAIN,), date(2020, 6, 1), date(2020, 6, 1))
    ]


def test_a_run_that_observes_nothing_plans_nothing(config):
    config.viz.hydrographs.enabled = False

    assert tfetch.plan_observations(config) == []


# --------------------------------------------------------------------- #
# Fetching                                                              #
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_fetched_observations_are_what_the_run_reads(run_dir, downloads, suffix):
    config = TevalConfig(**_config_dict(run_dir, run_dir / "obs" / f"obs{suffix}"))

    tfetch.fetch(config)

    obs = fetch_observations([DOMAIN], TIMES[0], TIMES[-1], config.io)
    assert list(obs.columns) == [DOMAIN]
    assert (obs[DOMAIN] == 1.5).all()
    assert downloads[0][3] is True, "fetch must not swallow download errors"


def test_a_second_fetch_downloads_nothing(config, downloads):
    tfetch.fetch(config)
    tfetch.fetch(config)

    assert len(downloads) == 1


def test_a_gage_nwis_has_no_data_for_still_counts_as_fetched(config, monkeypatch):
    calls = []

    def nothing_found(gage_ids, t_min, t_max, raise_errors=False):
        calls.append(gage_ids)
        return pd.DataFrame()

    monkeypatch.setattr(tfetch, "download_observations", nothing_found)

    tfetch.fetch(config)
    tfetch.fetch(config)

    assert len(calls) == 1
    assert tfetch.offline_problems(config) == []


def test_a_longer_window_is_fetched_again(run_dir, config, downloads):
    tfetch.fetch(config)
    later = pd.date_range("2020-07-01", periods=4, freq="h")
    xr.Dataset(
        {"streamflow": (("time", "feature_id"), np.ones((4, len(FEATURE_IDS))))},
        coords={"time": later, "feature_id": FEATURE_IDS},
    ).to_netcdf(run_dir / "troute" / f"cfe_{DOMAIN}_output" / "troute_output.nc", engine="h5netcdf")

    tfetch.fetch(config)

    assert len(downloads) == 2
    assert downloads[1][1:3] == (pd.Timestamp("2020-06-01"), pd.Timestamp("2020-07-01"))


def test_an_observations_file_fetch_did_not_write_is_left_alone(config, downloads):
    config.io.observations_file.parent.mkdir()
    config.io.observations_file.write_bytes(b"not ours")

    tfetch.fetch(config)

    assert downloads == []
    assert config.io.observations_file.read_bytes() == b"not ours"


def test_fetch_needs_somewhere_to_write(run_dir, downloads):
    config = TevalConfig(**_config_dict(run_dir, None))

    with pytest.raises(ValueError, match="observations_file"):
        tfetch.fetch(config)


def test_a_failed_download_writes_nothing(config, monkeypatch):
    def fail(*args, **kwargs):
        raise ConnectionError("no route to host")

    monkeypatch.setattr(tfetch, "download_observations", fail)

    with pytest.raises(ConnectionError):
        tfetch.fetch(config)

    assert not config.io.observations_file.exists()


# --------------------------------------------------------------------- #
# Offline                                                               #
# --------------------------------------------------------------------- #
def test_offline_needs_the_observations_file(config):
    assert "does not exist" in tfetch.offline_problems(config)[0]


def test_offline_is_ready_after_a_fetch(config, downloads):
    tfetch.fetch(config)

    assert tfetch.offline_problems(config) == []


def test_offline_names_a_domain_the_fetch_did_not_cover(config, downloads):
    tfetch.fetch(config)
    record = config.io.observations_file.with_name("obs.parquet.fetch.json")
    record.write_text(json.dumps({"requests": []}))

    assert tfetch.offline_problems(config)[0].startswith(f"[{DOMAIN}]")


def test_offline_trusts_an_observations_file_fetch_did_not_write(config):
    config.io.observations_file.parent.mkdir()
    pd.DataFrame({DOMAIN: [1.0]}, index=TIMES[:1]).to_parquet(config.io.observations_file)

    assert tfetch.offline_problems(config) == []


def test_offline_never_downloads(config, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("an offline run reached for the network")

    monkeypatch.setattr(observations, "download_observations", fail)
    config.io.auto_download_usgs = True
    config.io.offline = True

    obs = fetch_observations([DOMAIN], TIMES[0], TIMES[-1], config.io)

    assert obs.empty


@pytest.fixture
def restore_global_state():
    """``main`` sets logger levels and Dask's worker count globally."""
    root_level = logging.getLogger().level
    teval_level = logging.getLogger("teval").level
    with dask.config.set(num_workers=dask.config.get("num_workers", None)):
        yield
    logging.getLogger().setLevel(root_level)
    logging.getLogger("teval").setLevel(teval_level)


def _main(run_dir, monkeypatch, *flags, offline=False):
    config = _config_dict(run_dir, run_dir / "obs" / "obs.parquet")
    config["io"]["offline"] = offline
    config_path = run_dir / "teval_config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["teval", "-c", str(config_path), *flags])
    with pytest.raises(SystemExit) as exit_info:
        teval_main.main()
    return exit_info.value.code


def test_an_offline_run_without_fetched_observations_stops_before_processing(
    run_dir, monkeypatch, restore_global_state
):
    assert _main(run_dir, monkeypatch, offline=True) == 1
    assert not (run_dir / "output" / DOMAIN).exists()


def test_fetch_then_an_offline_run(run_dir, monkeypatch, downloads, restore_global_state):
    assert _main(run_dir, monkeypatch, "--fetch") == 0

    monkeypatch.setattr(sys, "argv", [
        "teval", "-c", str(run_dir / "teval_config.yaml"),
    ])
    config = yaml.safe_load((run_dir / "teval_config.yaml").read_text())
    config["io"]["offline"] = True
    (run_dir / "teval_config.yaml").write_text(yaml.safe_dump(config))
    teval_main.main()

    assert (run_dir / "output" / DOMAIN / f"{DOMAIN}_ensemble.nc").exists()
