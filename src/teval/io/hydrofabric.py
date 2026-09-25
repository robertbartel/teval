"""
teval.io.hydrofabric

Load and prepare NextGen hydrofabric GeoPackages.

Public API
----------
load_hydrofabric(gpkg_path, gpkg_layer="flowpaths")
    Read a ``.gpkg`` file and return the flowpath GeoDataFrame together with
    the gage crosswalk structures needed downstream.

read_gage_ids(gpkg_path, gpkg_layer="flowpaths")
    Read only the gage IDs, without geometries, for planning downloads.

build_nexus_crosswalk(gdf_hydro)
    Map each nexus to the feature ids draining to it, for weighting.

find_tailwater_feature(gdf_hydro)
    Identify outlet (tailwater) flowpaths in a hydrofabric GeoDataFrame.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import pyogrio
import pandas as pd
import numpy as np

from teval.identifiers import as_identifiers

logger = logging.getLogger(__name__)

_FLOWPATH_COLUMNS = {
    "id": ["id", "flowline_id", "flowpath_id"],
    "toid": ["toid", "flowpath_toid"],
    "hydroseq": ["hydroseq", "flowpath_hydroseq", "flowline_hydroseq"],
    "order": ["order", "streamorder"],
    "geometry": ["geometry"]
}


def _flowpath_columns(available_cols) -> Tuple[List[str], Dict[str, str]]:
    """
    Match the flowpath layer's columns to their standard names.

    Returns the columns to keep and the renames to standard names.  A non-empty
    rename means the v4.0 schema, whose gages live in the hydrolocations layer.
    """
    cols_to_keep = []
    rename_dict = {}

    for standard_name, possible_names in _FLOWPATH_COLUMNS.items():
        match = next((col for col in possible_names if col in available_cols), None)
        if match:
            cols_to_keep.append(match)
            if match != standard_name:
                rename_dict[match] = standard_name
        else:
            raise KeyError(f"Could not find a valid column for '{standard_name}'. Checked: {possible_names}")
    return cols_to_keep, rename_dict


def _nwis_gages(hydrolocations: pd.DataFrame) -> pd.DataFrame:
    """One row per NWIS gage in ``hl_reference``, its ID in a ``gage`` column."""
    gages = hydrolocations.copy()
    gages['hl_reference'] = gages['hl_reference'].str.split('|')
    gages = gages.explode('hl_reference')
    gages['hl_reference'] = gages['hl_reference'].str.strip()
    gages[['type', 'gage_id']] = gages['hl_reference'].str.split('-', n=1, expand=True)
    gages['type'] = gages['type'].str.strip()
    gages['gage_id'] = gages['gage_id'].str.strip()
    gages = gages[gages['type']=='nwis'].drop(columns=['hl_reference'])
    return gages.rename(columns={'gage_id': 'gage'})


def _network_gages(network: pd.DataFrame) -> pd.DataFrame:
    """The network layer's gage rows, their ID in a ``gage`` column."""
    gages = network[network["hl_uri"].str.startswith("gages-", na=False)].copy()
    gages["gage"] = gages["hl_uri"].str.replace("gages-", "")
    return gages


def read_gage_ids(gpkg_path: Optional[Path], gpkg_layer: str = "flowpaths") -> List[str]:
    """
    Read the hydrofabric's gage IDs without loading any geometry.

    A superset of the IDs ``load_hydrofabric`` returns: for the v4.0 schema it
    keeps every NWIS gage in the hydrolocations layer, including any the full
    crosswalk later drops.
    """
    if not gpkg_path:
        return []

    # read_info lists attribute fields only; geometry is always present
    fields = [*pyogrio.read_info(gpkg_path, layer=gpkg_layer)["fields"], "geometry"]
    _, rename_dict = _flowpath_columns(fields)

    if rename_dict:
        hydrolocations = gpd.read_file(
            gpkg_path, layer="hydrolocations",
            columns=["hl_class", "hl_reference", "flowpath_id"], ignore_geometry=True,
        )
        hydrolocations = hydrolocations[hydrolocations['hl_class']=='gage']
        gages = _nwis_gages(hydrolocations.dropna(subset='flowpath_id'))
    else:
        network = gpd.read_file(
            gpkg_path, layer="network", columns=["hl_uri"], ignore_geometry=True
        )
        gages = _network_gages(network)

    return sorted(gages["gage"].unique())


def load_hydrofabric(
    gpkg_path: Optional[Path],
    gpkg_layer: str = "flowpaths",
) -> Tuple[gpd.GeoDataFrame, List[str], Dict, Dict]:
    """
    Load a hydrofabric GeoPackage and build the gage crosswalk.
    Reads the flowpaths and network layers.

    Parameters
    ----------
    gpkg_path:
        Path to the .gpkg file, or None if no hydrofabric is needed
        for this domain (metrics and interactive map will be skipped).
    gpkg_layer:
        Layer to read from the hydrofabric. As of hydrofabric v4.0, this could
        be either "flowpaths" or "flowlines".

    Returns
    -------
    gdf : gpd.GeoDataFrame
        Flowpath geometries indexed by integer feature_id.  Has columns toid, hydroseq, order,
        geometry, and optionally gage.
    gage_ids : list[str]
        All USGS gage IDs found in the hydrofabric network layer.
    gage_to_fids : dict[str, list[int]]
        Maps each gage ID to the list of upstream feature IDs whose flows
        should be summed at that gage location.
    gage_to_nexus : dict[str, str]
        Maps each gage ID to the nexus toid of the gage row in the
        network layer (used for hydrograph labelling).
    """
    gdf = gpd.GeoDataFrame()
    gage_ids: List[str] = []
    gage_to_fids: Dict = {}
    gage_to_nexus: Dict = {}

    if not gpkg_path:
        return gdf, gage_ids, gage_to_fids, gage_to_nexus

    # Flowpaths
    flowpaths = gpd.read_file(gpkg_path, layer=gpkg_layer)
    cols_to_keep, rename_dict = _flowpath_columns(flowpaths.columns)

    fp_outlet_fl_df = pd.DataFrame()
    if 'flowline_id' in cols_to_keep:
        cols_to_keep.append('flowpath_id')
        flowpaths = promote_connecting_flowlines(flowpaths, id_col="flowline_id",toid_col="flowline_toid",flag_col="routeable")
        flowpaths = flowpaths[flowpaths['routeable']]
        def get_outlet_flowline_per_flowpath(df):
            # Create a lookup mapping every flowline_id to its parent flowpath_id
            id_to_path = df.set_index('flowline_id')['flowpath_id']
            
            # Look up the flowpath_id for the downstream segment (the 'toid')
            # If the toid is an outlet or not in the dataset, this returns NaN
            downstream_flowpath = df['flowline_toid'].map(id_to_path)
            
            # A flowline is the furthest downstream for its flowpath if it exits that flowpath
            is_terminal = df['flowpath_id'] != downstream_flowpath
            
            # Filter and return the results
            return df[is_terminal][['flowpath_id', 'flowline_id', 'flowline_toid']]
        
        fp_outlet_fl_df = get_outlet_flowline_per_flowpath(flowpaths)
    
    flowpaths = flowpaths[cols_to_keep]
    flowpaths = flowpaths.rename(columns=rename_dict)
    
    if "flowapth_id" not in flowpaths.columns:
        flowpaths["flowpath_id"] = flowpaths["id"]
            
    flowpaths["id"] = as_identifiers(
        flowpaths["id"], "The flowpaths 'id' column", required=True
    ).astype("int64")
    flowpaths["toid"] = as_identifiers(
        flowpaths["toid"], "The flowpaths 'toid' column", required=True
    ).astype("int64")
    flowpaths.set_index("id", inplace=True)

    # Network / gage crosswalk
    if rename_dict:
        hydrolocations = gpd.read_file(gpkg_path, layer='hydrolocations')
        hydrolocations = hydrolocations[hydrolocations['hl_class']=='gage'][['flowpath_id','hl_reference']].drop_duplicates().dropna(subset='flowpath_id')
        gages_net = _nwis_gages(
            pd.merge(hydrolocations, flowpaths[['flowpath_id','toid']].reset_index(), on='flowpath_id')
        )
        
        #TODO: Test that this works with an updated hydrofabric
        if not fp_outlet_fl_df.empty:
            fp_outlet_fl_df = fp_outlet_fl_df[fp_outlet_fl_df['flowpath_id'].isin(gages_net.flowpath_id)]
            gages_net = gages_net[gages_net['id'].isin(fp_outlet_fl_df.flowline_id)]
        

    else:
        gages_net = _network_gages(gpd.read_file(gpkg_path, layer="network"))
    
    if not gages_net.empty:
        # Nexus ID
        gage_to_nexus = gages_net.groupby("gage")["toid"].first().to_dict()

        gages_net["id"] = as_identifiers(
            gages_net["id"], "The network layer's 'id' column", required=True
        ).astype("int64")

        # All upstream feature IDs per gage
        gage_to_fids = (
            gages_net.groupby("gage")["id"]
            .unique()
            .apply(list)
            .to_dict()
        )
        gage_ids = list(gage_to_fids.keys())

        # Assign each gage to the highest stream-order flowpath for map display
        flowpath_gage_df = (
            pd.merge(
                flowpaths["order"].reset_index(),
                gages_net[["id", "gage"]],
                on="id",
            )
            .drop_duplicates()
        )
        flowpath_gage_df = (
            flowpath_gage_df
            .loc[flowpath_gage_df.groupby("gage")["order"].idxmax()][["gage", "id"]]
            .set_index("id")
        )
        flowpaths["gage"] = pd.Series(flowpath_gage_df.to_dict().get("gage"))
    else:
        flowpaths["gage"] = None

    if flowpaths.crs and flowpaths.crs.to_epsg() != 4326:
        gdf = flowpaths.to_crs(epsg=4326)
    else:
        gdf = flowpaths
    
    return gdf, gage_ids, gage_to_fids, gage_to_nexus


def build_nexus_crosswalk(
    gdf_hydro: Optional[gpd.GeoDataFrame],
) -> Dict[int, List[int]]:
    """
    Map each nexus to the feature ids draining to it.

    Ensemble weights are supplied per nexus while the ensemble dataset is
    indexed by ``feature_id``, so applying them needs this crosswalk.  The
    relationship is many-to-one: every flowpath at a confluence drains to the
    same nexus and therefore shares its weights.

    The mapping is derived entirely from the ``toid`` column of the flowpaths
    frame that has already been loaded, so this reads no file and adds no
    GeoPackage access to a run.

    Nexus keys come from ``toid`` and feature ids from the frame's index, and
    the two are never crossed; both are reduced by
    :func:`teval.identifiers.as_identifiers`, whose docstring gives the reason
    for each.

    Parameters
    ----------
    gdf_hydro:
        Flowpaths frame as ``load_hydrofabric`` returns it: indexed by integer
        feature id with a ``toid`` column naming the downstream nexus.  ``None``
        or an empty frame means no hydrofabric was configured for this domain.

    Returns
    -------
    dict[int, list[int]]
        Nexus id to the feature ids draining to it, both as plain integers.
        Nexus keys follow first appearance in the frame and each list follows
        frame order, so the result is deterministic.  A nexus appears only if
        at least one flowpath drains to it.  An absent or empty hydrofabric
        gives an empty dict, which resolves to no coverage rather than to an
        error — the caller decides whether missing weights matter.

    Raises
    ------
    ValueError
        A non-empty frame has no ``toid`` column, an identifier column holds a
        non-integral value, or a feature id is missing.

    Notes
    -----
    Flowpaths whose ``toid`` is missing are dropped with a warning: they cannot
    be placed at a nexus, and they surface downstream as uncovered features
    under the weight coverage policy.  A feature that drains to two different
    nexuses is left in both groups for the resolver to reject, so that rule
    lives in one place.
    """
    if gdf_hydro is None or len(gdf_hydro) == 0:
        logger.debug(
            "No hydrofabric flowpaths available; nexus crosswalk is empty."
        )
        return {}

    if "toid" not in gdf_hydro.columns:
        found = ", ".join(str(c) for c in gdf_hydro.columns) or "(none)"
        raise ValueError(
            f"Flowpaths frame has no 'toid' column, so the nexus each flowpath "
            f"drains to is unknown and no crosswalk can be built. Found "
            f"column(s): {found}."
        )

    # Both sides are pulled onto a fresh positional index so the id taken from
    # the frame's index and the toid taken from the column stay row-aligned.
    features = as_identifiers(
        pd.Series(gdf_hydro.index.to_numpy()), "The flowpath id column"
    )
    nexus = as_identifiers(pd.Series(gdf_hydro["toid"].to_numpy()), "The toid column")

    if features.isna().any():
        raise ValueError(
            f"{int(features.isna().sum())} flowpath(s) carry no usable feature "
            f"id in the frame's index, so they cannot be crosswalked to a nexus."
        )

    unplaced = int(nexus.isna().sum())
    if unplaced:
        logger.warning(
            f"{unplaced} flowpath(s) carry no usable 'toid' and are left out of "
            f"the nexus crosswalk; they will count as uncovered if weights are "
            f"applied."
        )

    pairs = (
        pd.DataFrame({"feature_id": features, "nexus_id": nexus})
        .dropna()
        .astype({"feature_id": "int64", "nexus_id": "int64"})
        .drop_duplicates()
    )

    crosswalk = {
        int(nexus_id): [int(feature_id) for feature_id in group]
        for nexus_id, group in pairs.groupby("nexus_id", sort=False)["feature_id"]
    }

    logger.debug(
        f"Built nexus crosswalk: {len(crosswalk)} nexus over "
        f"{len(pairs)} flowpath(s)."
    )
    return crosswalk


def find_tailwater_feature(gdf_hydro: gpd.GeoDataFrame) -> np.ndarray:
    """
    Identify tailwater flowpaths in a hydrofabric GeoDataFrame.

    Parameters
    ----------
    gdf_hydro:
        GeoDataFrame with integer index (feature IDs) and a toid column.

    Returns
    -------
    np.ndarray
        Array of tailwater feature IDs.
    """
    ids       = gdf_hydro.index
    toids     = gdf_hydro["toid"]
    missing   = ~toids.isin(ids)
    return gdf_hydro.loc[missing].index.values

def promote_connecting_flowlines(
    flowlines_df,
    id_col="flowline_id",
    toid_col="flowline_toid",
    flag_col="routeable",
):
    if flag_col not in flowlines_df.columns:
        return flowlines_df
 
    df = flowlines_df.copy()
    df[flag_col] = df[flag_col].fillna(False).astype(bool)
 
    next_id = dict(zip(df[id_col], df[toid_col]))
    is_routed = dict(zip(df[id_col], df[flag_col]))
 
    promoted = set()
    for flowline_id, routed in is_routed.items():
        if not routed:
            continue
        current = next_id.get(flowline_id)
        while (
            current is not None
            and current in is_routed
            and not is_routed[current]
            and current not in promoted
        ):
            promoted.add(current)
            current = next_id.get(current)
 
    if promoted:
        df.loc[df[id_col].isin(promoted), flag_col] = True
 
    return df