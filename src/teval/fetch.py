"""
teval.fetch

Download ahead of a run what it would otherwise download mid-run, so the run
itself can go without network access (``io.offline``).

Only observations are fetched.  Basemap tiles are not: which tiles a map needs
depends on the extent it is drawn at, which depends on run results, so an
offline run draws no basemaps instead.

Public API
----------
plan_observations(domain_map, io)
    One request per domain that needs observations: its gages and period.

fetch(config)
    Download every planned request into ``io.observations_file``, unless the
    file already holds them.

offline_problems(domain_map, io)
    Why an offline run of these domains would lack observations; empty if it
    would not.

RECORD_VERSION
    The version of the record ``fetch`` writes beside the observations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from teval.config import IOConfig, TevalConfig
from teval.io import initialize_domains
from teval.io.hydrofabric import read_gage_ids
from teval.io.observations import download_observations, nwis_gage_ids
from teval.workflow import domain_gage_ids, formulation_time_bounds

logger = logging.getLogger(__name__)

# A record of any other version covers nothing.  Unversioned records held
# dates, which NWIS read as local midnight at each gage, so their observations
# lack the first hours of each period.
RECORD_VERSION = 2


@dataclass(frozen=True)
class ObservationRequest:
    """The gages a domain observes and the period its formulation files span.

    ``start`` and ``end`` are naive UTC timestamps, as t-route's are.
    """

    domain: str
    gages: Tuple[str, ...]
    start: pd.Timestamp
    end: pd.Timestamp

    def covered_by(self, other: ObservationRequest) -> bool:
        """Whether *other* asked for all of these gages over all of this period."""
        return (
            set(self.gages) <= set(other.gages)
            and other.start <= self.start
            and self.end <= other.end
        )

    def describe_period(self) -> str:
        return f"{self.start:%Y-%m-%d %H:%M} -> {self.end:%Y-%m-%d %H:%M} UTC"

    def to_json(self) -> dict:
        return {
            "domain": self.domain,
            "gages": list(self.gages),
            "start": self.start.isoformat() + "Z",
            "end": self.end.isoformat() + "Z",
        }

    @classmethod
    def from_json(cls, data: dict) -> ObservationRequest:
        return cls(
            domain=data["domain"],
            gages=tuple(data["gages"]),
            start=pd.Timestamp(data["start"]).tz_localize(None),
            end=pd.Timestamp(data["end"]).tz_localize(None),
        )


def plan_observations(domain_map: Dict, io: IOConfig) -> List[ObservationRequest]:
    """
    One request per discovered domain that the run would fetch observations for.

    Asks the same questions the run does -- which domains observe, which gages,
    over which window -- but reads only gage IDs and time coordinates, so it is
    cheap enough for a login node.  Gages are the all-digit IDs NWIS is asked
    for, and a superset of those the run's crosswalk keeps.
    """
    requests = []
    for domain_name, entry in domain_map.items():
        # Discovery leaves gage_obs empty when nothing in the run uses observations
        if not entry["gage_obs"]:
            continue
        hydrofabric_gages = read_gage_ids(entry["hydrofabric"], io.hydrofabric_layer)
        gages = sorted(nwis_gage_ids(domain_gage_ids(entry, hydrofabric_gages)))
        t_min, t_max = formulation_time_bounds(entry["formulations"])
        if gages and t_min is not None:
            requests.append(
                ObservationRequest(domain_name, tuple(gages), t_min, t_max)
            )
    return requests


def _record_path(observations_file: Path) -> Path:
    """The file beside the observations recording what ``fetch`` requested."""
    return observations_file.with_name(observations_file.name + ".fetch.json")


def _unfetched(
    plan: List[ObservationRequest], observations_file: Path
) -> Optional[List[ObservationRequest]]:
    """
    The planned requests ``fetch`` has not made into this file: all of them if
    the file does not exist or its record is not of ``RECORD_VERSION``, None if
    ``fetch`` did not write it.
    """
    if not observations_file.exists():
        return plan
    record = _record_path(observations_file)
    if not record.exists():
        return None
    data = json.loads(record.read_text())
    if data.get("version") != RECORD_VERSION:
        logger.warning(
            f"{record} was written by another version of teval --fetch, so "
            f"{observations_file} counts as holding no observations. Older "
            "versions left out the first hours of each period."
        )
        return plan
    recorded = [ObservationRequest.from_json(r) for r in data["requests"]]
    return [r for r in plan if not any(r.covered_by(done) for done in recorded)]


def fetch(config: TevalConfig) -> None:
    """
    Download the observations this configuration's run needs.

    They go to ``io.observations_file``, with a record of the requests beside
    it.  If the record shows every planned request already made, nothing is
    downloaded; a gage NWIS has no data for counts as requested.  Otherwise
    every request is downloaded afresh and the file replaced.  An existing
    file without a record was not written here and is left alone.

    Raises
    ------
    ValueError
        Observations are needed but ``io.observations_file`` is unset or not
        ``.csv`` or ``.parquet``.
    """
    domain_map = initialize_domains(config.io, config.stats, config.metrics, config.viz)
    plan = plan_observations(domain_map, config.io)
    if not plan:
        logger.info("This configuration needs no observations; nothing to fetch.")
        return

    observations_file = config.io.observations_file
    if observations_file is None:
        raise ValueError(
            "--fetch writes observations to io.observations_file, which is not set."
        )
    if observations_file.suffix not in (".csv", ".parquet"):
        raise ValueError(
            f"io.observations_file must be .csv or .parquet, not {observations_file.name}."
        )

    unfetched = _unfetched(plan, observations_file)
    if unfetched is None:
        logger.warning(
            f"{observations_file} was not written by --fetch, so it is left as it "
            "is. Point io.observations_file elsewhere to fetch."
        )
        return
    if not unfetched:
        logger.info(f"{observations_file} already holds every observation needed.")
        return

    # Domains over the same period share one download, so no gage is fetched twice
    by_period: Dict[Tuple[pd.Timestamp, pd.Timestamp], List[ObservationRequest]] = {}
    for request in plan:
        by_period.setdefault((request.start, request.end), []).append(request)

    obs_df = pd.DataFrame()
    for (start, end), requests in by_period.items():
        gages = sorted(set().union(*(r.gages for r in requests)))
        logger.info(
            f"[{', '.join(r.domain for r in requests)}] Downloading {len(gages)} "
            f"gage(s), {requests[0].describe_period()}"
        )
        downloaded = download_observations(gages, start, end, raise_errors=True)
        obs_df = obs_df.combine_first(downloaded)

    observations_file.parent.mkdir(parents=True, exist_ok=True)
    obs_df.index.name = "time"
    if observations_file.suffix == ".csv":
        obs_df.to_csv(observations_file)
    else:
        obs_df.to_parquet(observations_file)
    _record_path(observations_file).write_text(
        json.dumps(
            {"version": RECORD_VERSION, "requests": [r.to_json() for r in plan]}, indent=2
        )
    )
    logger.info(f"Observations for {len(obs_df.columns)} gage(s) saved -> {observations_file}")


def offline_problems(domain_map: Dict, io: IOConfig) -> List[str]:
    """
    Why an offline run of these discovered domains would lack observations.

    A file ``fetch`` wrote must cover every planned request.  A file it did not
    write is trusted as it is.  Empty when nothing is missing.
    """
    plan = plan_observations(domain_map, io)
    if not plan:
        return []

    observations_file = io.observations_file
    if observations_file is None:
        return ["The run needs observations, but io.observations_file is not set."]
    if not observations_file.exists():
        return [f"The run needs observations, but {observations_file} does not exist."]

    unfetched = _unfetched(plan, observations_file)
    if unfetched is None:
        return []
    return [
        f"[{r.domain}] {observations_file} lacks observations for "
        f"{len(r.gages)} gage(s), {r.describe_period()}."
        for r in unfetched
    ]
