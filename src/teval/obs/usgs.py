"""USGS NWIS streamflow data retrieval utilities."""


import pandas as pd
import dataretrieval.nwis as nwis
from dataretrieval.exceptions import NoSitesError
from typing import List, Union, Optional

# Conversion constant: CFS to CMS
CFS_TO_CMS = 0.028316847

def find_gages_in_domain(min_x: float, min_y: float, max_x: float, max_y: float) -> pd.DataFrame:
    """
    Queries USGS NWIS for stream gages within a bounding box.
    
    Args:
        min_x, min_y, max_x, max_y: Bounding box coordinates (Long/Lat).
        
    Returns:
        pd.DataFrame: Metadata of found sites (site_no, station_nm, dec_lat_va, dec_long_va).
    """
    print(f"Searching for USGS gages in bbox: {min_x}, {min_y}, {max_x}, {max_y}")
    
    # "00060" = Streamflow, "iv" = Instantaneous
    try:
        sites_df, _ = nwis.what_sites(
            bBox=f"{min_x},{min_y},{max_x},{max_y}",
            parameterCd="00060",
            hasDataTypeCd="iv"
        )
    except Exception as e:
        print(f"Error querying NWIS sites: {e}")
        return pd.DataFrame()
    
    if sites_df is None or sites_df.empty:
        print("No gages found in this domain.")
        return pd.DataFrame()
        
    print(f"Found {len(sites_df)} gages.")
    return sites_df

def fetch_usgs_streamflow(
    site_ids: List[str],
    start_date: str,
    end_date: str,
    to_cms: bool = True,
    to_utc: bool = True,
    raise_errors: bool = False,
) -> pd.DataFrame:
    """
    Fetches daily or instantaneous streamflow (parameter 00060) from USGS NWIS.
    
    Args:
        site_ids: List of USGS gage IDs (strings, e.g. ["01111500"]).
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        to_cms: If True, converts from CFS to CMS.
        to_utc: If True, converts index to UTC timezone.
        raise_errors: If True, a failed request raises instead of returning an
            empty DataFrame.  NWIS finding no data is still an empty result.
        
    Returns:
        pd.DataFrame: Index is Datetime, Columns are site_ids. Values are flow.
    """
    # 00060 = Discharge (cfs), 00065 = Gage height
    # iv = instantaneous values (usually 15-min)
    # dv = daily values
    # Try 'iv' first for high-res validation, fall back to 'dv' if needed. 
    # For t-route, 'iv' is usually preferred.
    
    if isinstance(site_ids, str):
        site_ids = [site_ids]
    
    try:
        df_flow = nwis.get_record(
            sites=site_ids, 
            service='iv', 
            start=start_date, 
            end=end_date, 
            parameterCd='00060'
        )
    except Exception as e:
        # Sometimes dataretrieval fails if no data found
        if raise_errors and not isinstance(e, NoSitesError):
            raise
        print(f"Error fetching from NWIS: {e}")
        return pd.DataFrame()

    if df_flow is None or df_flow.empty:
        print("Warning: No USGS data returned for these sites/dates.")
        return pd.DataFrame()

    # Clean up the DataFrame
    # 1. Reset index to ensure site_no and datetime are accessible columns
    # (dataretrieval usually returns them as a MultiIndex)
    df_reset = df_flow.reset_index()
    
    # 2. Identify ALL potential flow columns
    # Rule: Must contain "00060" and NOT end with "cd" (which is a quality flag)
    flow_cols = [c for c in df_reset.columns if "00060" in c and not c.endswith("cd")]
    
    if not flow_cols:
        print("Columns found:", df_reset.columns)
        raise ValueError("Could not identify streamflow value column in NWIS response.")
    
    # 3. Merge columns
    if len(flow_cols) > 1:
        df_reset["_consolidated_flow"] = df_reset[flow_cols].bfill(axis=1).iloc[:, 0]
        val_col = "_consolidated_flow"
    else:
        val_col = flow_cols[0]
    
    # Pivot so each column is a site
    df_pivot = df_reset.pivot_table(
        index='datetime', 
        columns='site_no', 
        values=val_col,
        aggfunc='first'
    )
    
    # Timezone conversion
    if to_utc:
        # USGS usually returns timezone-aware timestamps (e.g. America/New_York)
        # We convert to UTC.
        if df_pivot.index.tz is not None:
            df_pivot.index = df_pivot.index.tz_convert('UTC')
        else:
            # If naive, assume UTC or warn user
            print("Warning: USGS data returned timezone-naive. Assuming UTC.")
            df_pivot.index = df_pivot.index.tz_localize('UTC')

    # Unit conversion
    if to_cms:
        df_pivot = df_pivot * CFS_TO_CMS

    return df_pivot