"""
Tests for the ordering of the loading steps in ``load_domain_data``.

``load_domain_data`` performs three loads, and the order among them is a
contract rather than an accident:

* the hydrofabric depends on nothing but ``domain_dict['hydrofabric']``, and
  must come first because the nexus-to-feature crosswalk it yields is what a
  weighted run needs *before* the ensemble stats graph is built;
* the formulations come second, since that is where the graph is built;
* the observations come last, because their fetch window is derived from the
  time bounds the formulation files report.

Only the first two moved.  These tests pin the order, pin the one real
dependency that forces observations to stay last, and assert that reordering
changed no result — both against a transcription of the previous
implementation and against a real end-to-end unweighted load.

The wiring the reorder was made for is covered separately, in
``test_workflow_weights.py``.
"""

from __future__ import annotations

from pathlib import Path

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from teval import workflow
from teval.config import IOConfig, StatsConfig


# --------------------------------------------------------------------- #
# Stub harness                                                          #
# --------------------------------------------------------------------- #
HYDROFABRIC_RESULT = (
    "gdf-sentinel", ["11111111"], {"11111111": [101]}, {"11111111": "nex-9001"},
)
FORMULATION_RESULT = (
    "stats-sentinel", "members-sentinel", "t-min-sentinel", "t-max-sentinel",
)
OBSERVATIONS_RESULT = "obs-sentinel"


@pytest.fixture
def domain_dict():
    """A domain map entry in the shape ``initialize_domains`` produces."""
    return {
        "formulations": {
            "raw_files": {"formA": Path("a.nc"), "formB": Path("b.nc")},
            "ensemble_file": None,
        },
        "hydrofabric": Path("domain.gpkg"),
        "gage_obs": {"domain_name": ["22222222"], "obs_file": [None]},
    }


@pytest.fixture
def calls(monkeypatch):
    """
    Replace the three loaders with recorders and return the call log.

    Each entry is ``(step_name, args)``, so tests can assert both the order of
    the steps and what each one was handed.
    """
    log: list = []

    def fake_load_hydrofabric(gpkg_path, *args, **kwargs):
        log.append(("hydrofabric", (gpkg_path,) + args, kwargs))
        return HYDROFABRIC_RESULT

    def fake_process_formulations(formulation_dict, stats_config, *args, **kwargs):
        log.append(("formulations", (formulation_dict, stats_config) + args, kwargs))
        return FORMULATION_RESULT

    def fake_fetch_observations(gage_ids, t_min, t_max, io, *args, **kwargs):
        log.append(("observations", (gage_ids, t_min, t_max, io) + args, kwargs))
        return OBSERVATIONS_RESULT

    monkeypatch.setattr(workflow, "load_hydrofabric", fake_load_hydrofabric)
    monkeypatch.setattr(
        workflow, "_process_formulation_files", fake_process_formulations
    )
    monkeypatch.setattr(workflow, "fetch_observations", fake_fetch_observations)
    return log


def _steps(log):
    return [name for name, _args, _kwargs in log]


# --------------------------------------------------------------------- #
# The reorder itself                                                    #
# --------------------------------------------------------------------- #
def test_hydrofabric_loads_before_the_formulations(domain_dict, calls):
    """The step that was second is now first."""
    workflow.load_domain_data(domain_dict, IOConfig(), StatsConfig())
    assert _steps(calls).index("hydrofabric") < _steps(calls).index("formulations")


def test_observations_load_last(domain_dict, calls):
    """The observation fetch stays behind both loads."""
    workflow.load_domain_data(domain_dict, IOConfig(), StatsConfig())
    assert _steps(calls) == ["hydrofabric", "formulations", "observations"]


def test_hydrofabric_is_loaded_before_the_stats_graph_can_fail(
    domain_dict, monkeypatch, calls,
):
    """
    The hydrofabric is in hand even when formulation processing blows up.

    This is the property the weighting work needs: whatever the formulation
    step does, the crosswalk was already built by the time it ran, so a
    weighted run can resolve weights before the stats graph is constructed.
    """
    def exploding(*args, **kwargs):
        calls.append(("formulations", args, kwargs))
        raise ValueError("No ensemble file or raw formulation files found to process.")

    monkeypatch.setattr(workflow, "_process_formulation_files", exploding)

    with pytest.raises(ValueError):
        workflow.load_domain_data(domain_dict, IOConfig(), StatsConfig())

    assert _steps(calls) == ["hydrofabric", "formulations"]


# --------------------------------------------------------------------- #
# What each step depends on                                             #
# --------------------------------------------------------------------- #
def test_hydrofabric_receives_only_the_domain_hydrofabric_entry(domain_dict, calls):
    """
    ``load_hydrofabric`` is handed the hydrofabric entry and the configured
    layer, nothing else.

    If it took anything derived from the formulations, hoisting it above the
    formulation step would be wrong; this asserts the independence the reorder
    relies on rather than assuming it.
    """
    io = IOConfig(hydrofabric_layer="flowlines")
    workflow.load_domain_data(domain_dict, io, StatsConfig())

    _name, args, kwargs = calls[0]
    assert args == (domain_dict["hydrofabric"], "flowlines")
    assert kwargs == {}


def test_only_the_hydrofabric_key_is_read_before_the_hydrofabric_loads(monkeypatch):
    """
    Reading the domain entry is itself evidence of a dependency.

    A mapping that records every key looked up shows the hydrofabric step
    touching ``hydrofabric`` alone — no peek at ``formulations`` sneaking in
    through a shared value.
    """
    read_keys: list = []
    touched_at_hydrofabric_time: list = []

    class RecordingDict(dict):
        def __getitem__(self, key):
            read_keys.append(key)
            return super().__getitem__(key)

        def get(self, key, default=None):
            read_keys.append(key)
            return super().get(key, default)

    entry = RecordingDict(
        formulations={"raw_files": {}, "ensemble_file": None},
        hydrofabric=Path("domain.gpkg"),
        gage_obs={"domain_name": [], "obs_file": []},
    )

    def fake_load_hydrofabric(gpkg_path, gpkg_layer):
        touched_at_hydrofabric_time.extend(read_keys)
        return HYDROFABRIC_RESULT

    monkeypatch.setattr(workflow, "load_hydrofabric", fake_load_hydrofabric)
    monkeypatch.setattr(
        workflow, "_process_formulation_files", lambda *a, **k: FORMULATION_RESULT
    )
    monkeypatch.setattr(
        workflow, "fetch_observations", lambda *a, **k: OBSERVATIONS_RESULT
    )

    workflow.load_domain_data(entry, IOConfig(), StatsConfig())

    assert touched_at_hydrofabric_time == ["hydrofabric"]


def test_observations_still_take_their_window_from_the_formulations(domain_dict, calls):
    """
    The one real ordering constraint that survives: the fetch window is the
    time bounds the formulation step reported, which is why observations
    cannot move up.
    """
    workflow.load_domain_data(domain_dict, IOConfig(), StatsConfig())

    _name, (gage_ids, t_min, t_max, io), _kwargs = calls[2]
    assert (t_min, t_max) == (FORMULATION_RESULT[2], FORMULATION_RESULT[3])
    assert isinstance(io, IOConfig)
    # ...and the gage list still merges the hydrofabric's gages with the
    # domain's own, which is the reason it cannot move up either.
    assert set(gage_ids) == {"11111111", "22222222"}


# --------------------------------------------------------------------- #
# Results unchanged by the reorder                                      #
# --------------------------------------------------------------------- #
def _load_domain_data_previous_order(domain_dict, io, stats_config):
    """
    The implementation as it stood before the reorder — previous *order*,
    current signature.

    Kept here so "results are unchanged" is asserted against the old code
    rather than against a restatement of the new code's behaviour.  Attribute
    lookups go through the ``workflow`` module so the same patches apply.

    Tracks the previous *order*, not its signature: ``_process_formulation_files``
    takes the weight plan the later wiring added, forwarded as the ``None``
    these tests' ``StatsConfig()`` produces.
    """
    results = {}

    results['formulations'] = {'combined': None, 'ensemble_members': None}
    ds_stats, ds_members, t_min, t_max = workflow._process_formulation_files(
        domain_dict['formulations'], stats_config, None
    )

    results['formulations']['combined'] = ds_stats
    results['formulations']['ensemble_members'] = ds_members

    (results['hydrofabric'], all_gage_ids, results['gage_to_fids'],
     results['gage_to_nexus']) = workflow.load_hydrofabric(
        domain_dict['hydrofabric'], io.hydrofabric_layer
    )

    initial_gages = domain_dict.get('gage_obs', {}).get('domain_name', [])
    if "CONUS" in initial_gages:
        initial_gages.remove("CONUS")
    gage_ids = list(set(initial_gages + all_gage_ids))

    results['gage_obs'] = workflow.fetch_observations(gage_ids, t_min, t_max, io)

    return results


def test_returns_what_the_previous_order_returned(domain_dict, calls):
    """Same inputs through both orders give the same dict."""
    io, stats_config = IOConfig(), StatsConfig()

    new = workflow.load_domain_data(domain_dict, io, stats_config)
    old = _load_domain_data_previous_order(domain_dict, io, stats_config)

    assert new == old
    assert set(new) == {
        "formulations", "hydrofabric", "gage_to_fids", "gage_to_nexus", "gage_obs",
    }


def test_both_orders_hand_the_same_arguments_to_each_step(domain_dict, calls):
    """
    Not just the same result — the same calls.

    Comparing the recorded calls of the two orders catches an argument that
    silently changed identity in the move, which an equal return value could
    hide.
    """
    io, stats_config = IOConfig(), StatsConfig()

    workflow.load_domain_data(domain_dict, io, stats_config)
    new_calls = list(calls)
    calls.clear()

    _load_domain_data_previous_order(domain_dict, io, stats_config)
    old_calls = list(calls)

    def normalise(log):
        # Sort by step so only the arguments are compared, not the order the
        # other tests already pin.  gage_ids is a set-derived list, so its
        # order is not meaningful.
        out = {}
        for name, args, kwargs in log:
            args = tuple(sorted(a) if isinstance(a, list) else a for a in args)
            out[name] = (args, kwargs)
        return out

    assert normalise(new_calls) == normalise(old_calls)


def test_conus_is_still_dropped_from_the_domain_gage_list(calls):
    """The CONUS special case rides along with the observation step, unmoved."""
    entry = {
        "formulations": {"raw_files": {}, "ensemble_file": None},
        "hydrofabric": None,
        "gage_obs": {"domain_name": ["CONUS", "22222222"], "obs_file": [None]},
    }

    workflow.load_domain_data(entry, IOConfig(), StatsConfig())

    _name, (gage_ids, _t_min, _t_max, _io), _kwargs = calls[2]
    assert "CONUS" not in gage_ids
    assert set(gage_ids) == {"11111111", "22222222"}


def test_a_domain_without_a_hydrofabric_still_loads(domain_dict, monkeypatch, calls):
    """
    ``hydrofabric: None`` is a supported domain entry, and hoisting the load
    must not turn a missing hydrofabric into an early failure.
    """
    domain_dict["hydrofabric"] = None

    def empty_hydrofabric(gpkg_path, gpkg_layer):
        calls.append(("hydrofabric", (gpkg_path, gpkg_layer), {}))
        return gpd.GeoDataFrame(), [], {}, {}

    monkeypatch.setattr(workflow, "load_hydrofabric", empty_hydrofabric)

    results = workflow.load_domain_data(domain_dict, IOConfig(), StatsConfig())

    assert _steps(calls) == ["hydrofabric", "formulations", "observations"]
    assert results["hydrofabric"].empty


# --------------------------------------------------------------------- #
# End to end, no stubs                                                  #
# --------------------------------------------------------------------- #
@pytest.fixture
def raw_formulation_files(tmp_path, combined_ds, formulation_names):
    """
    Write the synthetic ensemble out as one NetCDF per formulation.

    This is what ``_process_formulation_files`` actually reads, so the
    end-to-end test exercises ``open_mfdataset`` and ``build_stats`` rather
    than a stand-in for them.
    """
    files = {}
    for name in formulation_names:
        path = tmp_path / f"{name}.nc"
        combined_ds.sel(formulation=name).drop_vars("formulation").to_netcdf(
            path, engine="h5netcdf"
        )
        files[name] = path
    return files


def test_unweighted_run_end_to_end_is_unchanged(raw_formulation_files, combined_ds):
    """
    A real unweighted load: real files, real stats graph, no hydrofabric and
    no observations available.  The mean must still be the plain mean over the
    formulation dimension, and the result must still be lazy.
    """
    entry = {
        "formulations": {"raw_files": raw_formulation_files, "ensemble_file": None},
        "hydrofabric": None,
        "gage_obs": {"domain_name": [], "obs_file": [None]},
    }

    results = workflow.load_domain_data(entry, IOConfig(), StatsConfig())

    ds_stats = results["formulations"]["combined"]
    assert isinstance(ds_stats.streamflow_mean.data, da.Array)

    expected = combined_ds.streamflow.mean(dim="formulation").compute()
    got = ds_stats.streamflow_mean.compute().transpose(*expected.dims)
    np.testing.assert_allclose(got.values, expected.values)

    # The unloaded pieces are still their documented empties, not failures.
    assert results["hydrofabric"].empty
    assert results["gage_to_fids"] == {}
    assert results["gage_to_nexus"] == {}
    assert isinstance(results["gage_obs"], pd.DataFrame)
    assert results["gage_obs"].empty


def test_end_to_end_matches_calling_the_steps_by_hand(raw_formulation_files):
    """
    The reordered function equals the composition of its parts.

    Building the stats directly from the same files and comparing every
    variable pins that the reorder moved code without changing what any step
    produced.
    """
    from teval.ensemble_methods.stats import build_stats

    entry = {
        "formulations": {"raw_files": raw_formulation_files, "ensemble_file": None},
        "hydrofabric": None,
        "gage_obs": {"domain_name": [], "obs_file": [None]},
    }
    stats_config = StatsConfig()

    results = workflow.load_domain_data(entry, IOConfig(), stats_config)

    direct_combined = xr.open_mfdataset(
        paths=list(raw_formulation_files.values()),
        combine="nested",
        concat_dim="formulation",
        engine="h5netcdf",
        chunks={},
        parallel=True,
    ).assign_coords(formulation=list(raw_formulation_files.keys()))
    direct_stats = build_stats(direct_combined, raw_formulation_files, stats_config)

    got = results["formulations"]["combined"]
    assert set(got.data_vars) == set(direct_stats.data_vars)
    for var in direct_stats.data_vars:
        np.testing.assert_allclose(
            got[var].compute().values,
            direct_stats[var].compute().values,
        )
