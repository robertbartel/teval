"""NWIS requests are chunked by site, transient failures are retried, and
``raise_errors`` tells a failed request apart from one that found nothing."""

from __future__ import annotations

import pandas as pd
import pytest
from dataretrieval.exceptions import (
    HTTPError, NetworkError, NoSitesError, ServiceUnavailable, URLTooLong,
)

from teval.obs import usgs


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    """Records each wait instead of sleeping through it."""
    waits = []
    monkeypatch.setattr(usgs.time, "sleep", waits.append)
    return waits


def _get_record_raising(error):
    def get_record(**kwargs):
        raise error
    return get_record


def _long_record(sites, times, flow=10.0):
    """An NWIS 'iv' result: one 00060 row per (site_no, datetime)."""
    index = pd.MultiIndex.from_product(
        [sites, pd.DatetimeIndex(times, tz="UTC")], names=["site_no", "datetime"]
    )
    return pd.DataFrame({"00060": flow, "00060_cd": "P"}, index=index)


class _RecordingGetRecord:
    """Stands in for ``nwis.get_record``, answering each call from ``respond``."""

    def __init__(self, respond):
        self.respond = respond
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(list(kwargs["sites"]))
        return self.respond(kwargs["sites"])


def _one_time_record(sites):
    return _long_record(sites, ["2020-06-01 00:00"])


def test_a_failed_request_raises_when_asked(monkeypatch):
    monkeypatch.setattr(usgs.nwis, "get_record", _get_record_raising(NetworkError("down")))

    with pytest.raises(NetworkError):
        usgs.fetch_usgs_streamflow(["01010101"], "2020-06-01", "2020-06-02", raise_errors=True)


def test_finding_no_data_is_still_an_empty_result(monkeypatch):
    monkeypatch.setattr(usgs.nwis, "get_record", _get_record_raising(NoSitesError("url")))

    df = usgs.fetch_usgs_streamflow(["01010101"], "2020-06-01", "2020-06-02", raise_errors=True)

    assert df.empty


def test_a_failed_request_is_empty_by_default(monkeypatch):
    monkeypatch.setattr(usgs.nwis, "get_record", _get_record_raising(NetworkError("down")))

    assert usgs.fetch_usgs_streamflow(["01010101"], "2020-06-01", "2020-06-02").empty


def test_sites_are_requested_in_chunks(monkeypatch):
    get_record = _RecordingGetRecord(_one_time_record)
    monkeypatch.setattr(usgs.nwis, "get_record", get_record)
    sites = [f"{i:08d}" for i in range(7)]

    usgs.fetch_usgs_streamflow(sites, "2020-06-01", "2020-06-02", chunk_size=3)

    assert get_record.calls == [sites[0:3], sites[3:6], sites[6:7]]


def test_a_list_under_the_chunk_size_is_one_request(monkeypatch):
    get_record = _RecordingGetRecord(_one_time_record)
    monkeypatch.setattr(usgs.nwis, "get_record", get_record)

    usgs.fetch_usgs_streamflow(["01010101", "02020202"], "2020-06-01", "2020-06-02")

    assert len(get_record.calls) == 1


def test_a_repeated_site_is_requested_once(monkeypatch):
    get_record = _RecordingGetRecord(_one_time_record)
    monkeypatch.setattr(usgs.nwis, "get_record", get_record)

    df = usgs.fetch_usgs_streamflow(
        ["01010101", "02020202", "01010101"], "2020-06-01", "2020-06-02", chunk_size=2
    )

    assert get_record.calls == [["01010101", "02020202"]]
    assert list(df.columns) == ["01010101", "02020202"]


def test_chunks_merge_on_the_union_of_their_times(monkeypatch):
    times = {"01010101": ["2020-06-01 00:00", "2020-06-01 00:15"],
             "02020202": ["2020-06-01 00:15", "2020-06-01 00:30"]}
    monkeypatch.setattr(
        usgs.nwis, "get_record",
        _RecordingGetRecord(lambda sites: _long_record(sites, times[sites[0]], flow=100.0)),
    )

    df = usgs.fetch_usgs_streamflow(
        ["01010101", "02020202"], "2020-06-01", "2020-06-02", chunk_size=1
    )

    assert list(df.columns) == ["01010101", "02020202"]
    assert list(df.index) == list(pd.DatetimeIndex(
        ["2020-06-01 00:00", "2020-06-01 00:15", "2020-06-01 00:30"], tz="UTC"))
    assert df.iloc[1].tolist() == pytest.approx([100 * usgs.CFS_TO_CMS] * 2)
    assert df["01010101"].isna().tolist() == [False, False, True]


@pytest.mark.parametrize("nothing", [NoSitesError("url"), pd.DataFrame(), None],
                         ids=["no-sites", "empty", "none"])
def test_a_chunk_with_nothing_is_skipped(monkeypatch, nothing):
    def respond(sites):
        if sites == ["01010101"]:
            if isinstance(nothing, Exception):
                raise nothing
            return nothing
        return _one_time_record(sites)
    monkeypatch.setattr(usgs.nwis, "get_record", _RecordingGetRecord(respond))

    df = usgs.fetch_usgs_streamflow(
        ["01010101", "02020202"], "2020-06-01", "2020-06-02", raise_errors=True, chunk_size=1
    )

    assert list(df.columns) == ["02020202"]


def _second_chunk_fails(sites):
    if sites == ["02020202"]:
        raise NetworkError("down")
    return _one_time_record(sites)


def test_a_failed_chunk_raises_when_asked(monkeypatch):
    monkeypatch.setattr(usgs.nwis, "get_record", _RecordingGetRecord(_second_chunk_fails))

    with pytest.raises(NetworkError):
        usgs.fetch_usgs_streamflow(
            ["01010101", "02020202"], "2020-06-01", "2020-06-02", raise_errors=True, chunk_size=1
        )


def test_a_failed_chunk_is_left_out_by_default(monkeypatch):
    monkeypatch.setattr(usgs.nwis, "get_record", _RecordingGetRecord(_second_chunk_fails))

    df = usgs.fetch_usgs_streamflow(
        ["01010101", "02020202"], "2020-06-01", "2020-06-02", chunk_size=1
    )

    assert list(df.columns) == ["01010101"]


def _unavailable():
    return ServiceUnavailable("HTTP 503", status_code=503)


class _FailingFirst:
    """Raises each of ``errors`` in turn, then answers like NWIS."""

    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = 0

    def __call__(self, sites):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return _one_time_record(sites)


def test_a_transient_failure_is_retried_after_waiting(monkeypatch, sleeps):
    respond = _FailingFirst(_unavailable(), NetworkError("reset"))
    monkeypatch.setattr(usgs.nwis, "get_record", _RecordingGetRecord(respond))

    df = usgs.fetch_usgs_streamflow(["01010101"], "2020-06-01", "2020-06-02", raise_errors=True)

    assert list(df.columns) == ["01010101"]
    assert respond.calls == 3
    assert sleeps == list(usgs.RETRY_WAITS_SECONDS[:2])


def test_a_lasting_transient_failure_raises_after_the_last_retry(monkeypatch, sleeps):
    respond = _FailingFirst(*(_unavailable() for _ in range(10)))
    monkeypatch.setattr(usgs.nwis, "get_record", _RecordingGetRecord(respond))

    with pytest.raises(ServiceUnavailable):
        usgs.fetch_usgs_streamflow(["01010101"], "2020-06-01", "2020-06-02", raise_errors=True)

    assert respond.calls == len(usgs.RETRY_WAITS_SECONDS) + 1
    assert sleeps == list(usgs.RETRY_WAITS_SECONDS)


def test_a_lasting_transient_failure_is_left_out_by_default(monkeypatch):
    def respond(sites):
        if sites == ["02020202"]:
            raise _unavailable()
        return _one_time_record(sites)
    monkeypatch.setattr(usgs.nwis, "get_record", _RecordingGetRecord(respond))

    df = usgs.fetch_usgs_streamflow(
        ["01010101", "02020202"], "2020-06-01", "2020-06-02", chunk_size=1
    )

    assert list(df.columns) == ["01010101"]


@pytest.mark.parametrize("error", [
    NoSitesError("url"), URLTooLong("too long"), HTTPError("HTTP 400", status_code=400),
], ids=["no-sites", "url-too-long", "bad-request"])
def test_a_lasting_failure_is_not_retried(monkeypatch, sleeps, error):
    respond = _FailingFirst(error)
    monkeypatch.setattr(usgs.nwis, "get_record", _RecordingGetRecord(respond))

    usgs.fetch_usgs_streamflow(["01010101"], "2020-06-01", "2020-06-02")

    assert respond.calls == 1
    assert sleeps == []


def test_requests_are_paused_between(monkeypatch, sleeps):
    monkeypatch.setattr(usgs.nwis, "get_record", _RecordingGetRecord(_one_time_record))

    usgs.fetch_usgs_streamflow(
        ["01010101", "02020202", "03030303"], "2020-06-01", "2020-06-02", chunk_size=1
    )

    assert sleeps == [usgs.PAUSE_BETWEEN_REQUESTS_SECONDS] * 2
