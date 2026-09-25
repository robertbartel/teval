"""
teval.fetch

Download ahead of a run what it would otherwise download mid-run, so the run
itself can go without network access (``io.offline``).

Only observations are fetched.  Basemap tiles are not: which tiles a map needs
depends on the extent it is drawn at, which depends on run results, so an
offline run draws no basemaps instead.

Public API
----------
plan_observations(config)
    One request per domain that needs observations: its gages and period.

fetch(config)
    Download every planned request into ``io.observations_file``, unless the
    file already holds them.

offline_problems(config)
    Why an offline run of this configuration would lack observations; empty if
    it would not.

RECORD_VERSION
    The version of the record ``fetch`` writes beside the observations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from teval.config import TevalConfig
from teval.io import initialize_domains
from teval.io.hydrofabric import read_gage_ids
from teval.io.observations import download_observations
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


def plan_observations(config: TevalConfig) -> List[ObservationRequest]:
    """
    One request per domain that the run would fetch observations for.

    Asks the same questions the run does -- which domains observe, which gages,
    over which window -- but reads only gage IDs and time coordinates, so it is
    cheap enough for a login node.  Gages are the all-digit IDs NWIS is asked
    for, and a superset of those the run's crosswalk keeps.
    """
    domain_map = initialize_domains(config.io, config.stats, config.metrics, config.viz)

    requests = []
    for domain_name, entry in domain_map.items():
        # Discovery leaves gage_obs empty when nothing in the run uses observations
        if not entry["gage_obs"]:
            continue
        hydrofabric_gages = read_gage_ids(entry["hydrofabric"], config.io.hydrofabric_layer)
        gages = sorted(
            g for g in domain_gage_ids(entry, hydrofabric_gages) if str(g).isdigit()
        )
        t_min, t_max = formulation_time_bounds(entry["formulations"])
        if gages and t_min is not None:
            requests.append(
                ObservationRequest(domain_name, tuple(gages), t_min, t_max)
            )
    return requests


def _record_path(observations_file: Path) -> Path:
    """The file beside the observations recording what ``fetch`` requested."""
    return observations_file.with_name(observations_file.name + ".fetch.json")


def _recorded_requests(observations_file: Path) -> Optional[List[ObservationRequest]]:
    """
    What ``fetch`` requested for this file: None if it did not write it, empty
    if its record is not of ``RECORD_VERSION``.
    """
    record = _record_path(observations_file)
    if not (observations_file.exists() and record.exists()):
        return None
    data = json.loads(record.read_text())
    if data.get("version") != RECORD_VERSION:
        logger.warning(
            f"{record} was written by another version of teval --fetch, so "
            f"{observations_file} counts as holding no observations. Older "
            "versions left out the first hours of each period."
        )
        return []
    return [ObservationRequest.from_json(r) for r in data["requests"]]


def _uncovered(
    plan: List[ObservationRequest], recorded: List[ObservationRequest]
) -> List[ObservationRequest]:
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
    plan = plan_observations(config)
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

    recorded = _recorded_requests(observations_file)
    if recorded is None and observations_file.exists():
        logger.warning(
            f"{observations_file} was not written by --fetch, so it is left as it "
            "is. Point io.observations_file elsewhere to fetch."
        )
        return
    if recorded is not None and not _uncovered(plan, recorded):
        logger.info(f"{observations_file} already holds every observation needed.")
        return

    obs_df = pd.DataFrame()
    for request in plan:
        logger.info(
            f"[{request.domain}] Downloading {len(request.gages)} gage(s), "
            f"{request.describe_period()}"
        )
        downloaded = download_observations(
            list(request.gages), request.start, request.end, raise_errors=True
        )
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


def offline_problems(config: TevalConfig) -> List[str]:
    """
    Why an offline run of this configuration would lack observations.

    A file ``fetch`` wrote must cover every planned request.  A file it did not
    write is trusted as it is.  Empty when nothing is missing.
    """
    plan = plan_observations(config)
    if not plan:
        return []

    observations_file = config.io.observations_file
    if observations_file is None:
        return ["The run needs observations, but io.observations_file is not set."]
    if not observations_file.exists():
        return [f"The run needs observations, but {observations_file} does not exist."]

    recorded = _recorded_requests(observations_file)
    if recorded is None:
        return []
    return [
        f"[{r.domain}] {observations_file} lacks observations for "
        f"{len(r.gages)} gage(s), {r.describe_period()}."
        for r in _uncovered(plan, recorded)
    ]
