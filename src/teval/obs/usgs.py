"""USGS NWIS streamflow data retrieval utilities."""


import time

import pandas as pd
import dataretrieval.nwis as nwis
from dataretrieval.exceptions import NetworkError, NoSitesError, TransientError
from typing import List, Union, Optional

# Conversion constant: CFS to CMS
CFS_TO_CMS = 0.028316847

# Sites per NWIS request.  USGS water services take at most 100 sites per
# request, and one request for every gage in a CONUS hydrofabric is rejected
# as too long before it is sent.
MAX_SITES_PER_REQUEST = 100

# Seconds to wait before each further attempt at a request that failed
# transiently (429, 5xx, no connection).  dataretrieval's own retries give up
# within seconds, while NWIS 503 bursts can last longer.
RETRY_WAITS_SECONDS = (30, 60, 120)

# Seconds between requests, so a CONUS fetch does not provoke throttling.
PAUSE_BETWEEN_REQUESTS_SECONDS = 1

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

def _fetch_record(
    site_ids: List[str], start_date: str, end_date: str, raise_errors: bool
) -> Optional[pd.DataFrame]:
    """Makes one NWIS 'iv' request; returns None if it fails without raising."""
    def get_record():
        return nwis.get_record(
            sites=site_ids,
            service='iv',
            start=start_date,
            end=end_date,
            parameterCd='00060'
        )

    try:
        for wait in RETRY_WAITS_SECONDS:
            try:
                return get_record()
            except (TransientError, NetworkError) as e:
                print(f"NWIS request for {len(site_ids)} sites failed, retrying in {wait}s: {e}")
            time.sleep(wait)
        return get_record()
    except Exception as e:
        # Sometimes dataretrieval fails if no data found
        if raise_errors and not isinstance(e, NoSitesError):
            raise
        print(f"Error fetching {len(site_ids)} sites from NWIS: {e}")
        return None

def fetch_usgs_streamflow(
    site_ids: List[str],
    start_date: str,
    end_date: str,
    to_cms: bool = True,
    to_utc: bool = True,
    raise_errors: bool = False,
    chunk_size: int = MAX_SITES_PER_REQUEST,
) -> pd.DataFrame:
    """
    Fetches daily or instantaneous streamflow (parameter 00060) from USGS NWIS.

    Sites are requested ``chunk_size`` at a time and the results merged.

    Args:
        site_ids: List of USGS gage IDs (strings, e.g. ["01111500"]).
        start_date: Start of the period, as NWIS takes it: a time with its
            offset (e.g. ``2020-06-01T00:00Z``), or a date (``YYYY-MM-DD``),
            which NWIS reads as midnight local time at each site.
        end_date: End of the period, in the same form.
        to_cms: If True, converts from CFS to CMS.
        to_utc: If True, converts index to UTC timezone.
        raise_errors: If True, a failed request raises instead of being
            skipped, which leaves its sites out of the result.  A transient
            failure only counts once its retries (``RETRY_WAITS_SECONDS``) run
            out.  NWIS finding no data is still an empty result.
        chunk_size: Maximum number of sites in one NWIS request.

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
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    # Distinct chunks keep a site from appearing in two responses.
    site_ids = list(dict.fromkeys(site_ids))

    records = []
    for i in range(0, len(site_ids), chunk_size):
        if i:
            time.sleep(PAUSE_BETWEEN_REQUESTS_SECONDS)
        record = _fetch_record(site_ids[i:i + chunk_size], start_date, end_date, raise_errors)
        if record is not None and not record.empty:
            records.append(record)

    if not records:
        print("Warning: No USGS data returned for these sites/dates.")
        return pd.DataFrame()

    df_flow = pd.concat(records)

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