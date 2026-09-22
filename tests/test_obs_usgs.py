"""``raise_errors`` tells a failed NWIS request apart from one that found nothing."""

from __future__ import annotations

import pytest
from dataretrieval.exceptions import NetworkError, NoSitesError

from teval.obs import usgs


def _get_record_raising(error):
    def get_record(**kwargs):
        raise error
    return get_record


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
