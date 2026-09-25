"""
``load_hydrofabric`` reduces its identifiers through the canonical helper.

The GeoPackage is the point where prefixed identifiers enter teval, so it is
where the reduction belongs.  These tests drive the loader with a stubbed
``read_file`` rather than a real ``.gpkg``: what is under test is the
identifier handling and the gage crosswalk, not geopandas' ability to read a
file.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest

from teval.io import hydrofabric
from teval.io.hydrofabric import load_hydrofabric


def _layers(monkeypatch, flowpaths: pd.DataFrame, network: pd.DataFrame) -> None:
    """Stub ``gpd.read_file`` so each layer comes back as the given frame."""
    frames = {
        "flowpaths": gpd.GeoDataFrame(flowpaths.assign(geometry=None)),
        "network": gpd.GeoDataFrame(network),
    }

    def read_file(path, layer, columns=None, ignore_geometry=False):
        frame = frames[layer]
        return (frame[columns] if columns else frame).copy()

    monkeypatch.setattr(hydrofabric.gpd, "read_file", read_file)


def _flowpaths(ids, toids) -> pd.DataFrame:
    return pd.DataFrame(
        {"id": ids, "toid": toids, "hydroseq": range(len(ids)), "order": [1] * len(ids)}
    )


def _no_gages() -> pd.DataFrame:
    """A network layer carrying rows, none of which is a gage."""
    return pd.DataFrame(
        {"id": ["wb-101"], "toid": ["nex-9001"], "hl_uri": [None]}, dtype=object
    )


def test_prefixed_identifiers_are_stripped_to_integers(monkeypatch):
    """The established behaviour: ``wb-101`` indexes as 101, ``nex-9001`` as 9001."""
    _layers(
        monkeypatch,
        _flowpaths(["wb-101", "wb-102"], ["nex-9001", "nex-9001"]),
        _no_gages(),
    )

    gdf, _, _, _ = load_hydrofabric("domain.gpkg")

    assert list(gdf.index) == [101, 102]
    assert list(gdf["toid"]) == [9001, 9001]
    assert gdf.index.dtype == "int64" and gdf["toid"].dtype == "int64"


def test_numeric_identifier_columns_are_accepted(monkeypatch):
    """
    A GeoPackage storing its ids as numbers rather than prefixed strings loads.

    The hand-rolled ``.str.replace(...).astype(int)`` this replaced raised
    ``AttributeError: Can only use .str accessor with string values`` on such a
    column -- a crash that named pandas rather than the hydrofabric.
    """
    _layers(monkeypatch, _flowpaths([101, 102], [9001, 9002]), _no_gages())

    gdf, _, _, _ = load_hydrofabric("domain.gpkg")

    assert list(gdf.index) == [101, 102]
    assert list(gdf["toid"]) == [9001, 9002]


def test_an_identifier_with_no_digits_names_its_column(monkeypatch):
    """A column that cannot be reduced fails as a hydrofabric problem."""
    _layers(monkeypatch, _flowpaths(["wb-101", "wb-"], ["nex-9001", "nex-9001"]), _no_gages())

    with pytest.raises(ValueError, match=r"flowpaths 'id' column"):
        load_hydrofabric("domain.gpkg")


def test_the_gage_crosswalk_reduces_its_ids_the_same_way(monkeypatch):
    """The network layer's ids go through the same reduction as the flowpaths'."""
    _layers(
        monkeypatch,
        _flowpaths(["wb-101", "wb-102"], ["nex-9001", "nex-9001"]),
        pd.DataFrame(
            {
                "id": ["wb-101", "wb-102"],
                "toid": ["nex-9001", "nex-9001"],
                "hl_uri": ["gages-01010101", "gages-02020202"],
            }
        ),
    )

    gdf, gage_ids, gage_to_fids, gage_to_nexus = load_hydrofabric("domain.gpkg")

    assert sorted(gage_ids) == ["01010101", "02020202"]
    assert gage_to_fids == {"01010101": [101], "02020202": [102]}
    assert gage_to_nexus == {"01010101": "nex-9001", "02020202": "nex-9001"}
    assert list(gdf.index) == [101, 102]
