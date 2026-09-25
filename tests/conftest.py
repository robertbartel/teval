"""
Shared pytest fixtures for the teval test suite.

The fixtures here provide the small synthetic inputs the ensemble machinery
operates on: a combined dataset carrying a ``formulation`` dimension, a
minimal flowpaths frame in the shape ``load_hydrofabric`` returns, a tidy
weight frame in the provisional weight-file schema, and the same inputs
written to disk for the tests that drive ``load_domain_data`` end to end.

All values are chosen to be hand-checkable so tests can assert against
expectations worked out on paper rather than against a second implementation.
Everything describing the same synthetic domain lives here so the several
weighting test modules cannot drift apart on what that domain is; the plain
helper functions they share are in ``weighting_support.py``.
"""

from __future__ import annotations

import logging

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from teval import workflow


# --------------------------------------------------------------------- #
# Formulations                                                          #
# --------------------------------------------------------------------- #
@pytest.fixture
def formulation_names():
    """Names of the synthetic ensemble members, in dataset order."""
    return ["formA", "formB", "formC"]


@pytest.fixture
def formulation_index_map():
    """1-based index to formulation name binding, as configuration carries it."""
    return {1: "formA", 2: "formB", 3: "formC"}


# --------------------------------------------------------------------- #
# Combined ensemble dataset                                             #
# --------------------------------------------------------------------- #
@pytest.fixture
def feature_ids():
    """Integer feature ids (prefix-stripped flowpath ids) of the synthetic run."""
    return [101, 102, 103, 201]


@pytest.fixture
def combined_ds(formulation_names, feature_ids):
    """
    Small lazy ensemble dataset shaped like an ``open_mfdataset`` result.

    Dimensions are ``(formulation, time, feature_id)`` with a single
    ``streamflow`` variable.  Values are ``formulation_offset + feature_offset
    + time_step`` so any reduction over ``formulation`` has an obvious
    closed-form expectation.

    The array is dask-backed, so tests can assert that graph construction
    stays lazy.
    """
    n_form = len(formulation_names)
    times = pd.date_range("2020-01-01", periods=4, freq="h")
    n_time = len(times)
    n_feat = len(feature_ids)

    form_offset = (np.arange(n_form) + 1)[:, None, None] * 100.0
    time_offset = np.arange(n_time)[None, :, None] * 1.0
    feat_offset = (np.arange(n_feat) + 1)[None, None, :] * 10.0
    values = form_offset + time_offset + feat_offset

    ds = xr.Dataset(
        {"streamflow": (("formulation", "time", "feature_id"), values)},
        coords={
            "formulation": formulation_names,
            "time": times,
            "feature_id": feature_ids,
        },
        attrs={"description": "synthetic ensemble"},
    )
    return ds.chunk({"formulation": 1, "time": -1, "feature_id": -1})


# --------------------------------------------------------------------- #
# Hydrofabric                                                           #
# --------------------------------------------------------------------- #
@pytest.fixture
def flowpaths_frame(feature_ids):
    """
    Minimal flowpaths frame in the shape ``load_hydrofabric`` returns.

    Indexed by integer ``id`` with an integer ``toid`` naming the downstream
    nexus.  Features 101, 102 and 103 converge on nexus 9001 (a confluence, so
    the nexus-to-feature relationship is genuinely many-to-one); 201 drains to
    its own nexus 9002.
    """
    return pd.DataFrame(
        {
            "toid": [9001, 9001, 9001, 9002],
            "hydroseq": [4, 3, 2, 1],
            "order": [1, 1, 2, 1],
        },
        index=pd.Index(feature_ids, name="id"),
    )


# --------------------------------------------------------------------- #
# Weights                                                               #
# --------------------------------------------------------------------- #
@pytest.fixture
def weight_frame():
    """
    Tidy weight frame in the provisional weight-file schema.

    Columns are ``nexus_id`` (string, ``nex-`` prefix retained),
    ``formulation_index`` (1-based int) and ``weight`` (float).  Both groups
    are complete over the three formulations and sum to 1; nexus 9002 carries
    an individual zero weight, which is permitted.
    """
    return pd.DataFrame(
        {
            "nexus_id": ["nex-9001"] * 3 + ["nex-9002"] * 3,
            "formulation_index": [1, 2, 3, 1, 2, 3],
            "weight": [0.5, 0.3, 0.2, 0.25, 0.75, 0.0],
        }
    )


@pytest.fixture
def crosswalk():
    """
    Nexus to draining features, matching the ``flowpaths_frame`` fixture.

    Features 101, 102 and 103 converge on nexus 9001 — the confluence that
    makes the relationship genuinely many-to-one — and 201 drains to 9002.
    Built here rather than derived from the hydrofabric so the resolver is
    tested against the crosswalk *contract*, not against a second module.
    """
    return {9001: [101, 102, 103], 9002: [201]}


@pytest.fixture
def one_nexus_crosswalk():
    """Crosswalk covering only the confluence, leaving 201 uncovered."""
    return {9001: [101, 102, 103]}


# --------------------------------------------------------------------- #
# The same domain, on disk                                              #
# --------------------------------------------------------------------- #
@pytest.fixture
def raw_files(tmp_path, combined_ds, formulation_names):
    """The synthetic ensemble written out as one NetCDF per formulation."""
    files = {}
    for name in formulation_names:
        path = tmp_path / f"{name}.nc"
        combined_ds.sel(formulation=name).drop_vars("formulation").to_netcdf(
            path, engine="h5netcdf"
        )
        files[name] = path
    return files


@pytest.fixture
def weight_file(tmp_path, weight_frame):
    """The tidy weight frame written out in the provisional csv schema."""
    path = tmp_path / "weights.csv"
    weight_frame.to_csv(path, index=False)
    return path


@pytest.fixture
def partial_weight_file(tmp_path, weight_frame):
    """A weight file covering nexus 9001 only, leaving feature 201 uncovered."""
    path = tmp_path / "partial_weights.csv"
    weight_frame[weight_frame["nexus_id"] == "nex-9001"].to_csv(path, index=False)
    return path


@pytest.fixture
def hydrofabric(monkeypatch, flowpaths_frame):
    """
    Stand in for ``load_hydrofabric`` with the synthetic flowpaths frame.

    Only the frame matters here — the crosswalk is derived from its ``toid``
    column — so the gage structures are returned empty.  Patching the loader
    rather than writing a GeoPackage keeps this about the wiring; reading a
    ``.gpkg`` is ``teval.io.hydrofabric``'s own concern.
    """
    monkeypatch.setattr(
        workflow,
        "load_hydrofabric",
        lambda gpkg_path, gpkg_layer: (flowpaths_frame, [], {}, {}),
    )
    return flowpaths_frame


@pytest.fixture
def no_hydrofabric(monkeypatch):
    """A domain with no hydrofabric at all, so no crosswalk can be built."""
    monkeypatch.setattr(
        workflow,
        "load_hydrofabric",
        lambda gpkg_path, gpkg_layer: (gpd.GeoDataFrame(), [], {}, {}),
    )


# --------------------------------------------------------------------- #
# Whole runs                                                            #
# --------------------------------------------------------------------- #
@pytest.fixture
def restore_global_state():
    """Undo the logger levels and Dask worker count ``teval.__main__.main`` sets."""
    root_level = logging.getLogger().level
    teval_level = logging.getLogger("teval").level
    with dask.config.set(num_workers=dask.config.get("num_workers", None)):
        yield
    logging.getLogger().setLevel(root_level)
    logging.getLogger("teval").setLevel(teval_level)
