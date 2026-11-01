"""Offline tests replay recorded Statistikbanken responses.

The files in tests/fixtures/ are unmodified JSON bodies from
https://api.statbank.dk/v1, captured on 2026-10-31 and 2026-11-01. They are
pretty-printed only so they can be read; the objects are the API's.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from statistikbanken_mcp.server import set_client
from statistikbanken_mcp.statbank import StatbankClient

FIXTURES = Path(__file__).parent / "fixtures"
_CACHE: dict[str, object] = {}


def load_fixture(name: str) -> object:
    if name not in _CACHE:
        _CACHE[name] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return _CACHE[name]


def _response(status: int, name: str) -> httpx.Response:
    return httpx.Response(status, json=load_fixture(name))


def _handler(calls: list[dict[str, object]]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        calls.append({"path": request.url.path, "body": body})
        path = request.url.path
        if path.endswith("/tables"):
            return _response(200, "tables.json")
        if path.endswith("/tableinfo"):
            table = str(body.get("table", "")).strip().casefold()
            if table == "folk1a":
                return _response(200, "tableinfo_FOLK1A.json")
            if table == "indkp105":
                return _response(200, "tableinfo_INDKP105.json")
            if table == "idrfor01":
                return _response(200, "tableinfo_IDRFOR01.json")
            if table == "hotel1":
                return _response(200, "tableinfo_HOTEL1.json")
            if table == "straf10":
                return _response(200, "tableinfo_STRAF10.json")
            if table == "hfudd11":
                return _response(200, "tableinfo_HFUDD11.json")
            if table == "notatable":
                return _response(400, "tableinfo_NOTATABLE.json")
            return httpx.Response(500, json={"message": f"no tableinfo fixture for {table}"})
        if path.endswith("/data"):
            return _data_response(body)
        return httpx.Response(500, json={"message": f"no fixture for {path}"})

    return httpx.MockTransport(handle)


def _data_response(body: dict[str, object]) -> httpx.Response:
    table = str(body.get("table", "")).upper()
    if table == "INDKP105":
        return _response(200, "data_indkp105.json")
    if table == "IDRFOR01":
        return _response(200, "data_idrfor01.json")
    if table == "HOTEL1":
        return _response(200, "data_hotel_months.json")
    if table == "HFUDD11":
        return _response(200, "data_hfudd11_ages.json")
    if table == "STRAF10":
        tid_values = _variable_values(body, "Tid")
        if any(str(value).startswith("sum(") for value in tid_values):
            return _response(400, "data_straf10_sum_rejected.json")
        return _response(200, "data_straf10_quarters.json")
    variables = {
        item["code"]: item["values"] for item in body.get("variables", []) if isinstance(item, dict)
    }
    alder = (variables.get("ALDER") or [""])[0]
    areas = variables.get("OMRÅDE") or []
    tid = variables.get("Tid") or []
    if isinstance(alder, str) and alder.startswith("sum(0-17") and areas == ["561", "791"]:
        return _response(200, "data_children_0_17.json")
    if isinstance(alder, str) and "99999" in alder:
        return _response(400, "data_bad_sum.json")
    if isinstance(alder, str) and alder.startswith("sum(80+"):
        if areas == ["561", "791", "157"]:
            return _response(200, "data_three_municipalities_80plus.json")
        if areas == ["561"]:
            return _response(200, "data_esbjerg_80plus.json")
    if tid and str(tid[0]).startswith(">="):
        return _response(200, "data_viborg_10y.json")
    if areas == ["561"] and alder == "IALT" and "Tid" not in variables:
        return _response(200, "data_esbjerg_total_latest.json")
    if areas == ["740"]:
        return _response(200, "data_silkeborg.json")
    if areas == ["101"]:
        return _response(200, "data_kobenhavn.json")
    return httpx.Response(500, json={"message": "no data fixture", "variables": variables})


def _variable_values(body: dict[str, object], code: str) -> list[object]:
    variables = body.get("variables", [])
    if not isinstance(variables, list):
        return []
    for item in variables:
        if isinstance(item, dict) and item.get("code") == code:
            values = item.get("values", [])
            return list(values) if isinstance(values, list) else []
    return []


@pytest.fixture
def calls() -> list[dict[str, object]]:
    return []


@pytest.fixture(autouse=True)
def offline_client(request: pytest.FixtureRequest, calls: list[dict[str, object]]):
    if request.node.get_closest_marker("live"):
        yield None
        return
    http = httpx.Client(
        base_url="https://api.statbank.dk/v1",
        transport=_handler(calls),
    )
    client = StatbankClient(http)
    previous = set_client(client)
    yield client
    set_client(previous)
    client.close()
