"""
``download_observations`` asks NWIS for the period in UTC.

Given only dates, NWIS starts at midnight local time at each gage, hours into
the UTC day, and the hours before that are left without observations.  NWIS
is faked here, including that reading of dates.
"""

from __future__ import annotations

import pandas as pd
import pytest

from teval.io.observations import download_observations
from teval.obs import usgs

SITE = "11447650"
SITE_TZ = "America/Los_Angeles"


class _FakeNWIS:
    """Answers like NWIS 'iv' for a Pacific-time gage: 15-minute readings, each
    its minute past the hour in cfs, from ``start`` through ``end``."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        start = self._parse(kwargs["start"])
        end = self._parse(kwargs["end"], end_of_day=True)
        times = pd.date_range(start, end, freq="15min").tz_convert(SITE_TZ)
        index = pd.MultiIndex.from_product(
            [kwargs["sites"], times], names=["site_no", "datetime"]
        )
        flow = [float(t.minute) for t in times] * len(kwargs["sites"])
        return pd.DataFrame({"00060": flow, "00060_cd": "P"}, index=index)

    @staticmethod
    def _parse(value, end_of_day=False):
        if "T" in value:
            return pd.Timestamp(value)
        midnight = pd.Timestamp(value).tz_localize(SITE_TZ)
        return midnight + pd.Timedelta(hours=23, minutes=45) if end_of_day else midnight


@pytest.fixture
def nwis(monkeypatch):
    fake = _FakeNWIS()
    monkeypatch.setattr(usgs.nwis, "get_record", fake)
    return fake


def test_nwis_is_asked_for_utc_times(nwis):
    download_observations(
        [SITE], pd.Timestamp("2020-06-01 00:00"), pd.Timestamp("2020-06-01 03:00")
    )

    assert nwis.calls[0]["start"] == "2020-05-31T23:00Z"
    assert nwis.calls[0]["end"] == "2020-06-01T04:00Z"


def test_a_period_off_the_hour_is_asked_for_in_whole_hours(nwis):
    download_observations(
        [SITE], pd.Timestamp("2020-06-01 00:30"), pd.Timestamp("2020-06-01 02:10")
    )

    assert nwis.calls[0]["start"] == "2020-05-31T23:00Z"
    assert nwis.calls[0]["end"] == "2020-06-01T04:00Z"


def test_every_hour_of_the_period_is_observed(nwis):
    t_min, t_max = pd.Timestamp("2020-06-01 00:00"), pd.Timestamp("2020-06-01 09:00")

    obs = download_observations([SITE], t_min, t_max)

    hours = pd.date_range(t_min, t_max, freq="h", tz="UTC")
    assert obs[SITE].reindex(hours).notna().all()


def test_the_hours_at_the_ends_of_the_period_average_whole_hours(nwis):
    t_min, t_max = pd.Timestamp("2020-06-01 00:00"), pd.Timestamp("2020-06-01 03:00")

    obs = download_observations([SITE], t_min, t_max)

    whole_hour_mean = (0 + 15 + 30 + 45) / 4 * usgs.CFS_TO_CMS
    ends = pd.DatetimeIndex([t_min, t_max], tz="UTC")
    assert obs[SITE].reindex(ends).tolist() == pytest.approx([whole_hour_mean] * 2)
