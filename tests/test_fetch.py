"""
``--fetch`` downloads what a run needs so the run can go offline.

The download itself is stubbed: what is under test is which observations are
planned, where they land, when a re-run skips them, and what an offline run
checks before it starts.
"""

from __future__ import annotations

import json
import sys

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


def _write_troute_outputs(troute_dir, times=TIMES, names=("cfe", "noahowp")):
    for name in names:
        run_dir = troute_dir / f"{name}_{DOMAIN}_output"
        run_dir.mkdir(parents=True, exist_ok=True)
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


def _record_path(config):
    return config.io.observations_file.with_name("obs.parquet.fetch.json")


@pytest.fixture
def downloads(monkeypatch):
    """Stub the NWIS download; record each call and return a frame for it."""
    calls = []

    def download(gage_ids, t_min, t_max, raise_errors=False):
        calls.append((list(gage_ids), t_min, t_max, raise_errors))
        index = pd.date_range(t_min, t_max, freq="h", tz="UTC")
        return pd.DataFrame({g: 1.5 for g in gage_ids}, index=index)

    monkeypatch.setattr(tfetch, "download_observations", download)
    return calls


# --------------------------------------------------------------------- #
# The plan                                                              #
# --------------------------------------------------------------------- #
def test_each_observing_domain_is_planned_over_its_files_times(config):
    assert tfetch.plan_observations(config) == [
        tfetch.ObservationRequest(DOMAIN, (DOMAIN,), TIMES[0], TIMES[-1])
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
    _write_troute_outputs(run_dir / "troute", later, names=("cfe",))

    tfetch.fetch(config)

    assert len(downloads) == 2
    assert downloads[1][1:3] == (TIMES[0], later[-1])


def test_an_earlier_start_on_the_same_day_is_fetched_again(run_dir, config, downloads):
    _write_troute_outputs(run_dir / "troute", TIMES[2:])
    tfetch.fetch(config)
    _write_troute_outputs(run_dir / "troute", TIMES, names=("cfe",))

    tfetch.fetch(config)

    assert len(downloads) == 2
    assert downloads[0][1] == TIMES[2]
    assert downloads[1][1:3] == (TIMES[0], TIMES[-1])


def test_the_record_keeps_times_and_their_version(config, downloads):
    tfetch.fetch(config)

    record = json.loads(_record_path(config).read_text())
    assert record["version"] == tfetch.RECORD_VERSION
    assert record["requests"][0]["start"] == "2020-06-01T00:00:00Z"
    assert record["requests"][0]["end"] == "2020-06-01T03:00:00Z"


def _write_date_only_record(config):
    """A record as --fetch wrote it before RECORD_VERSION: no version, dates only."""
    _record_path(config).write_text(json.dumps({
        "requests": [
            {"domain": DOMAIN, "gages": [DOMAIN], "start": "2020-06-01", "end": "2020-06-01"}
        ]
    }))


def test_observations_fetched_under_a_date_only_record_are_fetched_again(config, downloads):
    tfetch.fetch(config)
    _write_date_only_record(config)

    tfetch.fetch(config)

    assert len(downloads) == 2
    assert json.loads(_record_path(config).read_text())["version"] == tfetch.RECORD_VERSION


def test_a_record_round_trips_and_covers_its_request():
    request = tfetch.ObservationRequest(
        DOMAIN, (DOMAIN,), pd.Timestamp("2020-06-01 01:30"), pd.Timestamp("2020-06-02 23:00")
    )

    restored = tfetch.ObservationRequest.from_json(json.loads(json.dumps(request.to_json())))

    assert restored == request
    assert request.covered_by(restored)


def test_a_request_is_not_covered_by_one_starting_later_that_day():
    request = tfetch.ObservationRequest(
        DOMAIN, (DOMAIN,), pd.Timestamp("2020-06-01 01:00"), pd.Timestamp("2020-06-01 03:00")
    )
    later = tfetch.ObservationRequest(
        DOMAIN, (DOMAIN,), pd.Timestamp("2020-06-01 02:00"), pd.Timestamp("2020-06-01 03:00")
    )

    assert not request.covered_by(later)


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
    _record_path(config).write_text(
        json.dumps({"version": tfetch.RECORD_VERSION, "requests": []})
    )

    assert tfetch.offline_problems(config)[0].startswith(f"[{DOMAIN}]")


def test_offline_does_not_trust_a_date_only_record(config, downloads):
    tfetch.fetch(config)
    _write_date_only_record(config)

    problems = tfetch.offline_problems(config)

    assert len(problems) == 1
    assert problems[0].startswith(f"[{DOMAIN}]")
    assert "2020-06-01 00:00 -> 2020-06-01 03:00 UTC" in problems[0]


def test_offline_trusts_an_observations_file_fetch_did_not_write(config):
    config.io.observations_file.parent.mkdir()
    pd.DataFrame({DOMAIN: [1.0]}, index=TIMES[:1]).to_parquet(config.io.observations_file)

    assert tfetch.offline_problems(config) == []


def _offline_config(run_dir, **io) -> TevalConfig:
    config = _config_dict(run_dir, run_dir / "obs" / "obs.parquet")
    config["io"].update(offline=True, **io)
    config["viz"]["skill_maps"] = {"enabled": True, "basemap": True}
    return TevalConfig(**config)


def test_offline_never_downloads(run_dir, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("an offline run reached for the network")

    monkeypatch.setattr(observations, "download_observations", fail)
    config = _offline_config(run_dir, auto_download_usgs=True)

    obs = fetch_observations([DOMAIN], TIMES[0], TIMES[-1], config.io)

    assert obs.empty


def test_offline_draws_no_skill_map_basemaps(run_dir):
    assert not _offline_config(run_dir).viz.skill_maps.basemap


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
