"""Tool behaviour against recorded API responses."""

import httpx
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from statistikbanken_mcp.server import find_tabel, hent_data, set_client, tabelinfo
from statistikbanken_mcp.statbank import StatbankClient


def test_find_tabel_befolkning_leads_with_folk1a() -> None:
    hits = find_tabel("befolkning", limit=5)
    assert hits[0] == {
        "id": "FOLK1A",
        "title": "Befolkningen den 1. i kvartalet",
        "unit": "Antal",
        "latest_period": "2026K3",
        "variables": ["område", "køn", "alder", "civilstand", "tid"],
    }
    assert len(hits) == 5


def test_find_tabel_matches_id_case_insensitively() -> None:
    hits = find_tabel("folk1a", limit=1)
    assert [hit["id"] for hit in hits] == ["FOLK1A"]


def test_find_tabel_empty_and_unknown() -> None:
    assert find_tabel("zzzxxyyqqq") == []
    with pytest.raises(ToolError, match="befolkning"):
        find_tabel("   ")
    with pytest.raises(ToolError, match="1 to 25"):
        find_tabel("befolkning", limit=0)


def test_tabelinfo_compacts_ages_and_lists_areas() -> None:
    info = tabelinfo("folk1a")
    assert info["id"] == "FOLK1A"
    assert info["unit"] == "Antal"
    assert info["updated"] == "2026-08-10T08:00:00"
    variables = {item["id"]: item for item in info["variables"]}
    assert list(variables) == ["OMRÅDE", "KØN", "ALDER", "CIVILSTAND", "Tid"]
    assert variables["ALDER"]["numeric_codes"] == {"from": "0", "to": "125"}
    assert variables["ALDER"]["values"] == [{"id": "IALT", "text": "Alder i alt"}]
    assert variables["ALDER"]["eliminable"] is True
    assert variables["Tid"]["eliminable"] is False
    assert variables["Tid"]["time"] is True
    assert variables["Tid"]["values"][0]["id"] == "2008K1"
    assert variables["Tid"]["values"][-1]["id"] == "2026K3"
    areas = {item["id"]: item["text"] for item in variables["OMRÅDE"]["values"]}
    assert areas["230"] == "Rudersdal"
    assert areas["223"] == "Hørsholm"
    assert areas["157"] == "Gentofte"
    assert variables["KØN"]["values"][0] == {"id": "TOT", "text": "I alt"}


def test_unknown_table_uses_the_api_error() -> None:
    with pytest.raises(ToolError, match="Unknown table 'NOTATABLE'") as caught:
        tabelinfo("NOTATABLE")
    message = str(caught.value)
    assert "find_tabel" in message
    assert "Traceback" not in message
    assert "errorTypeCode" not in message


def test_unknown_value_and_variable_are_actionable() -> None:
    with pytest.raises(ToolError, match="Unknown value '99999'") as caught:
        hent_data("FOLK1A", {"ALDER": ["99999"]})
    message = str(caught.value)
    assert "0 to 125" in message
    assert "sum_from" in message
    assert "Traceback" not in message

    with pytest.raises(ToolError, match="Unknown value '99999' for variable OMRÅDE") as caught:
        hent_data("FOLK1A", {"OMRÅDE": ["99999"]})
    assert "tabelinfo" in str(caught.value)

    with pytest.raises(ToolError, match="Unknown variable 'NOPE'") as caught:
        hent_data("FOLK1A", {"NOPE": ["1"]})
    assert "ALDER" in str(caught.value)


def test_api_rejects_a_bad_sum_expression(calls: list[dict[str, object]]) -> None:
    with pytest.raises(ToolError, match="Unknown value '99999' for variable ALDER") as caught:
        hent_data(
            "FOLK1A",
            {"OMRÅDE": ["230"], "ALDER": ["sum(bad=99999)"], "Tid": ["(1)"]},
        )
    message = str(caught.value)
    assert "sum_from" in message
    assert "Traceback" not in message
    data_calls = [call for call in calls if str(call["path"]).endswith("/data")]
    assert data_calls


def test_sum_range_rejects_bad_bounds() -> None:
    with pytest.raises(ToolError, match="lower code to a higher one"):
        hent_data("FOLK1A", sum_range={"ALDER": ["17", "0"]})
    with pytest.raises(ToolError, match="two numeric codes"):
        hent_data("FOLK1A", sum_range={"ALDER": ["0"]})
    with pytest.raises(ToolError, match="sum_range and also"):
        hent_data("FOLK1A", {"ALDER": ["IALT"]}, sum_range={"ALDER": ["0", "17"]})
    with pytest.raises(ToolError, match="between 200 and 210"):
        hent_data("FOLK1A", sum_range={"ALDER": ["200", "210"]})


def test_sum_from_and_bad_sum_arguments() -> None:
    with pytest.raises(ToolError, match="both values and sum_from"):
        hent_data("FOLK1A", {"ALDER": ["80"]}, {"ALDER": "80"})
    with pytest.raises(ToolError, match="numeric code"):
        hent_data("FOLK1A", sum_from={"ALDER": "IALT"})
    with pytest.raises(ToolError, match=">= 200"):
        hent_data("FOLK1A", sum_from={"ALDER": "200"})
    with pytest.raises(ToolError, match="list of codes"):
        hent_data("FOLK1A", {"OMRÅDE": []})


def test_omitted_time_is_the_latest_total() -> None:
    result = hent_data("FOLK1A", {"område": [" 230 "], "alder": ["ialt"]})
    assert result["rows"] == [
        {
            "OMRÅDE": "Rudersdal",
            "ALDER": "Alder i alt",
            "Tid": "2026K3",
            "value": 58381,
        }
    ]
    assert result["unit"] == "Antal"
    assert result["source"] == "Danmarks Statistik"


def test_unreachable_api_has_no_stack_trace() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    http = httpx.Client(base_url="https://api.statbank.dk/v1", transport=httpx.MockTransport(fail))
    previous = set_client(StatbankClient(http))
    try:
        with pytest.raises(ToolError, match="Could not reach Statistikbanken") as caught:
            tabelinfo("FOLK1A")
        assert "Traceback" not in str(caught.value)
    finally:
        set_client(previous)
        http.close()
