"""
teval.io.observations

Load USGS streamflow observations from file or the USGS API.

Public API
----------
fetch_observations(gage_ids, t_min, t_max, io)
    Return a wide-format DataFrame of observed streamflow (m³/s) with
    datetime index and zero-padded 8-digit USGS gage ID columns.

download_observations(gage_ids, t_min, t_max)
    Download hourly USGS NWIS streamflow (m³/s) for the gages and period.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from teval.config import IOConfig
from teval.obs import usgs

logger = logging.getLogger(__name__)


def _normalize_gage_id(g) -> str:
    """
    Normalize a raw gage identifier to a zero-padded 8-digit string.

    USGS gage IDs are 8-digit zero-padded strings, but pre-downloaded parquet
    files store them as plain integers or unpadded strings.
    IDs already longer than 8 characters (some newer stations) are kept
    as-is.
    """
    s = str(g).strip()
    return s.zfill(8) if len(s) <= 8 else s


# How far beyond each end of the period observations are requested, so the
# hourly means at its ends average whole hours and interpolation reaches them.
DOWNLOAD_PADDING = pd.Timedelta(hours=1)


def download_observations(
    gage_ids: List[str],
    t_min: pd.Timestamp,
    t_max: pd.Timestamp,
    raise_errors: bool = False,
) -> pd.DataFrame:
    """
    Download USGS NWIS streamflow for the gages over the period.

    *t_min* and *t_max* are naive UTC, as t-route's time coordinates are.
    NWIS is asked for the whole hours spanning them, padded by
    ``DOWNLOAD_PADDING``, as exact UTC times; given dates, NWIS would start at
    local midnight at each gage, hours into the UTC day.

    Only all-digit IDs are requested.  Returns hourly means in m^3/s on a UTC
    index, interpolated across gaps; empty if nothing was requested or found.
    With *raise_errors*, a failed request raises rather than returning empty.
    """
    clean_gages = [str(g) for g in gage_ids if str(g).isdigit()]
    if not clean_gages:
        return pd.DataFrame()

    obs_df = usgs.fetch_usgs_streamflow(
        clean_gages,
        f"{t_min.floor('h') - DOWNLOAD_PADDING:%Y-%m-%dT%H:%MZ}",
        f"{t_max.ceil('h') + DOWNLOAD_PADDING:%Y-%m-%dT%H:%MZ}",
        to_cms=True,
        to_utc=True,
        raise_errors=raise_errors,
    )
    if not obs_df.empty:
        obs_df = obs_df.resample("1h").mean().interpolate()
    return obs_df


def fetch_observations(
    gage_ids: List[str],
    t_min: Optional[pd.Timestamp],
    t_max: Optional[pd.Timestamp],
    io: IOConfig,
) -> pd.DataFrame:
    """
    Load USGS streamflow observations for the requested gages and period.

    Priority
    --------
    1. Read from io.observations_file (Parquet or CSV) if it exists.
    2. Fall back to the USGS NWIS API if io.auto_download_usgs is True.
    3. Return an empty DataFrame and log a warning otherwise.

    Gage ID normalization
    ---------------------
    All column names in the returned DataFrame are normalised to zero-padded 8-digit 
    strings via _normalize_gage_id.  The caller's gage_ids list is normalised identically 
    so the intersection is correct regardless of how IDs were stored in the source file.

    Parameters
    ----------
    gage_ids:
        USGS gage IDs from the hydrofabric crosswalk.
    t_min, t_max:
        Simulation period boundaries used to check temporal overlap and to
        bound the USGS API request.
    io:
        IOConfig instance carrying file paths and download flags.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame: datetime index (tz-naive), one column per
        gage (zero-padded 8-digit string), values in m^3/s.  Empty DataFrame
        if no observations are available.
    """
    obs_df = pd.DataFrame()

    if not gage_ids:
        return obs_df

    # Load from file
    if io.observations_file and io.observations_file.exists():
        file_path = io.observations_file
        logger.debug(f"Loading observations from {file_path.suffix} file: {file_path}")

        if file_path.suffix == ".parquet":
            obs_df = pd.read_parquet(file_path)
        elif file_path.suffix == ".csv":
            obs_df = pd.read_csv(file_path)
        else:
            logger.error(f"Unsupported observation file format: {file_path.suffix}")
            return obs_df

        # Standardise index
        if "time" in obs_df.columns:
            obs_df.set_index("time", inplace=True)
        elif "datetime" in obs_df.columns:
            obs_df.set_index("datetime", inplace=True)
        obs_df.index = pd.to_datetime(obs_df.index)

        # Normalise column names and filter to requested gages
        obs_df.columns = [_normalize_gage_id(c) for c in obs_df.columns]
        norm_gage_ids  = {_normalize_gage_id(g) for g in gage_ids}
        valid_gages    = list(set(obs_df.columns) & norm_gage_ids)
        obs_df         = obs_df[valid_gages]

        if obs_df.empty:
            logger.warning(
                f"No observation columns matched the {len(gage_ids)} gage IDs from "
                "the hydrofabric. Check that gage ID format matches between the "
                "file and the hydrofabric 'hl_uri' field (e.g. zero-padding)."
            )
            return obs_df

        obs_t0 = obs_df.index.min()
        obs_t1 = obs_df.index.max()
        logger.debug(
            f"Observations loaded: {len(obs_df.columns)} gages, "
            f"{obs_t0.date()} -> {obs_t1.date()}"
        )

        # Temporal overlap check
        if t_min is not None and t_max is not None:
            sim_t0 = pd.Timestamp(t_min)
            sim_t1 = pd.Timestamp(t_max)
            # Strip timezone info for comparison
            sim_t0 = sim_t0.tz_localize(None) if sim_t0.tzinfo else sim_t0
            sim_t1 = sim_t1.tz_localize(None) if sim_t1.tzinfo else sim_t1
            obs_t0n = obs_t0.tz_convert(None) if obs_t0.tzinfo else obs_t0
            obs_t1n = obs_t1.tz_convert(None) if obs_t1.tzinfo else obs_t1

            if not (obs_t0n <= sim_t1 and sim_t0 <= obs_t1n):
                logger.warning(
                    f"OBS TIME RANGE ({obs_t0.date()} -> {obs_t1.date()}) DOES NOT "
                    f"OVERLAP WITH SIMULATION PERIOD ({sim_t0.date()} -> "
                    f"{sim_t1.date()}). Metrics and hydrograph observations will be "
                    "empty. Ensure the observations file covers the simulation period."
                )
            else:
                logger.debug(
                    f"Simulation period: {sim_t0.date()} -> {sim_t1.date()} "
                    "(overlap with obs confirmed)"
                )

        return obs_df

    # USGS API fallback
    if io.auto_download_usgs:
        logger.info("Fetching USGS data via API...")
        obs_df = download_observations(gage_ids, t_min, t_max)

        if not obs_df.empty and io.save_downloaded_obs:
            suffix = io.save_downloaded_obs.suffix
            if suffix == ".csv":
                obs_df.to_csv(io.save_downloaded_obs)
            elif suffix == ".parquet":
                obs_df.to_parquet(io.save_downloaded_obs)
            else:
                logger.warning(f"Observation file type '{suffix}' not supported for saving.")

        return obs_df

    # Nothing available
    logger.warning("No observation file provided and auto_download is disabled.")
    return obs_df