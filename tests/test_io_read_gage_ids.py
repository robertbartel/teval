"""
``read_gage_ids`` finds the gages ``load_hydrofabric`` would, without geometry.

Both schemas are written as real GeoPackages, since what is under test is
which layers and columns are read.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from teval.io.hydrofabric import load_hydrofabric, read_gage_ids


def _line(i):
    return LineString([(i, 0), (i + 1, 0)])


def _write_network_schema(path):
    """Pre-v4.0: standard flowpath columns, gages in the network layer."""
    gpd.GeoDataFrame(
        {
            "id": ["wb-1", "wb-2"], "toid": ["nex-10", "nex-20"],
            "hydroseq": [2, 1], "order": [1, 2],
        },
        geometry=[_line(0), _line(1)], crs="EPSG:4326",
    ).to_file(path, layer="flowpaths")
    gpd.GeoDataFrame(
        {
            "id": ["wb-1", "wb-2"], "toid": ["nex-10", "nex-20"],
            "hl_uri": ["gages-01010101", None],
        },
        geometry=[_line(0), _line(1)], crs="EPSG:4326",
    ).to_file(path, layer="network")


def _write_v4_schema(path):
    """v4.0: prefixed flowpath columns, gages in the hydrolocations layer."""
    gpd.GeoDataFrame(
        {
            "flowpath_id": [1, 2], "flowpath_toid": [10, 20],
            "flowpath_hydroseq": [2, 1], "streamorder": [1, 2],
        },
        geometry=[_line(0), _line(1)], crs="EPSG:4326",
    ).to_file(path, layer="flowpaths")
    gpd.GeoDataFrame(
        {
            "flowpath_id": [1, 2, None],
            "hl_class": ["gage", "gage", "gage"],
            "hl_reference": ["nwis-02020202 | other-9", "nwis-03030303", "nwis-04040404"],
        },
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0)], crs="EPSG:4326",
    ).to_file(path, layer="hydrolocations")


def test_no_hydrofabric_has_no_gages():
    assert read_gage_ids(None) == []


def test_network_schema_gages_match_the_full_load(tmp_path):
    path = tmp_path / "domain.gpkg"
    _write_network_schema(path)

    _, gage_ids, _, _ = load_hydrofabric(path)

    assert read_gage_ids(path) == sorted(gage_ids) == ["01010101"]


def test_v4_schema_reads_nwis_gages_on_a_flowpath(tmp_path):
    path = tmp_path / "domain.gpkg"
    _write_v4_schema(path)

    _, gage_ids, _, _ = load_hydrofabric(path)

    assert read_gage_ids(path) == ["02020202", "03030303"]
    assert set(gage_ids) <= set(read_gage_ids(path))
