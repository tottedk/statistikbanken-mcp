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
    with pytest.raises(ToolError, match="No table matches"):
        find_tabel("zzzxxyyqqq")
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
    assert variables["Tid"]["time_span"]["first"]["id"] == "2008K1"
    assert variables["Tid"]["time_span"]["last"]["id"] == "2026K3"
    assert variables["Tid"]["time_span"]["count"] == 75
    assert "values" not in variables["Tid"]
    assert variables["OMRÅDE"]["values_truncated"] is True
    assert variables["OMRÅDE"]["value_count"] == 105
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


def test_search_partial_synonym_and_place_names() -> None:
    befolkning = [hit["id"] for hit in find_tabel("befolkning kommune", limit=5)]
    assert "FOLK1A" in befolkning
    assert any(hit["id"] == "AUP01" for hit in find_tabel("bruttoledighed"))
    assert find_tabel("population", limit=1)[0]["id"] == "FOLK1A"
    assert any(hit["id"] == "EJ56" for hit in find_tabel("huspriser"))
    assert any(hit["id"] == "HDYR07" for hit in find_tabel("husdyr"))
    assert any(hit["id"] == "STRAF10" for hit in find_tabel("indbrud"))
    silkeborg = find_tabel("Silkeborg", limit=1)[0]
    assert silkeborg["id"] == "FOLK1A"
    assert silkeborg["matched_values"] == [{"variable": "OMRÅDE", "id": "740", "text": "Silkeborg"}]
    with pytest.raises(ToolError, match="befolkning"):
        find_tabel("enhjørninger")


def test_values_accept_labels_and_suggest_near_misses(calls: list[dict[str, object]]) -> None:
    result = hent_data("FOLK1A", {"OMRÅDE": ["Silkeborg"], "ALDER": ["IALT"], "Tid": ["(1)"]})
    assert result["rows"][0]["OMRÅDE"] == "Silkeborg"
    assert result["rows"][0]["value"] == 103293
    sent = [call["body"] for call in calls if str(call["path"]).endswith("/data")][-1]
    assert isinstance(sent, dict)
    areas = next(item["values"] for item in sent["variables"] if item["code"] == "OMRÅDE")
    assert areas == ["740"]

    copenhagen = hent_data("FOLK1A", {"område": ["Kobenhavn"], "alder": ["ialt"], "Tid": ["(1)"]})
    assert copenhagen["rows"][0]["OMRÅDE"] == "København"
    with pytest.raises(ToolError, match="Latest period is 2026K3"):
        hent_data("FOLK1A", {"Tid": ["2035K1"]})
    with pytest.raises(ToolError, match="Narnia"):
        hent_data("FOLK1A", {"OMRÅDE": ["Narnia"]})


def test_required_variable_without_a_total_lists_codes() -> None:
    with pytest.raises(ToolError, match="ENHED") as caught:
        hent_data("INDKP105", {"OMRÅDE": ["157"], "Tid": ["2024"]})
    message = str(caught.value)
    assert "118" in message
    assert "Der skal vælges" not in message
    assert "Traceback" not in message


def test_ambiguous_time_sum_is_rejected_before_any_number(
    calls: list[dict[str, object]],
) -> None:
    with pytest.raises(ToolError, match="sum_from") as caught:
        hent_data(
            "FOLK1A",
            values={"ALDER": [">=80"], "OMRÅDE": ["Rudersdal", "Hørsholm"]},
            sum_range={"Tid": ["2015K1", "2026K3"]},
        )
    message = str(caught.value)
    assert ">=2015K1" in message
    assert 'not values [">=80"]' in message
    assert "13078" not in message
    assert "hide rows" in message
    assert not any(str(call["path"]).endswith("/data") for call in calls)


def test_two_period_sum_is_not_a_time_range(calls: list[dict[str, object]]) -> None:
    with pytest.raises(ToolError, match="not every period") as caught:
        hent_data(
            "FOLK1A",
            values={"OMRÅDE": ["230"], "ALDER": ["IALT"]},
            sum_range={"Tid": ["2015K1", "2026K3"]},
        )
    assert ">=2015K1" in str(caught.value)
    assert not any(str(call["path"]).endswith("/data") for call in calls)


def test_local_sum_keeps_each_series() -> None:
    from statistikbanken_mcp.statbank import _sum_rows

    rows = [
        {"OMRÅDE": "Rudersdal", "ALDER": "80 år", "Tid": "2015K1", "value": 3524},
        {"OMRÅDE": "Rudersdal", "ALDER": "80 år", "Tid": "2026K3", "value": 5045},
        {"OMRÅDE": "Hørsholm", "ALDER": "80 år", "Tid": "2015K1", "value": 1719},
        {"OMRÅDE": "Hørsholm", "ALDER": "80 år", "Tid": "2026K3", "value": 2790},
        {"OMRÅDE": "Rudersdal", "ALDER": "81 år", "Tid": "2015K1", "value": 1},
        {"OMRÅDE": "Rudersdal", "ALDER": "81 år", "Tid": "2026K3", "value": 2},
    ]
    summed = _sum_rows(rows, "Tid", ["2015K1", "2026K3"])
    by_series = {(row["OMRÅDE"], row["ALDER"]): row for row in summed}
    assert len(by_series) == 3
    assert by_series[("Rudersdal", "80 år")]["value"] == 8569
    assert by_series[("Hørsholm", "80 år")]["value"] == 4509
    assert by_series[("Rudersdal", "81 år")]["Tid"] == "2015K1+2026K3"
    assert 13078 not in {row["value"] for row in summed}


def test_values_filter_splits_words_and_keeps_unmatched_variables() -> None:
    info = tabelinfo("FOLK1A", values_filter="Rudersdal, Hørsholm")
    variables = {item["id"]: item for item in info["variables"]}
    areas = {item["id"]: item["text"] for item in variables["OMRÅDE"]["values"]}
    assert areas == {"223": "Hørsholm", "230": "Rudersdal"}
    assert "filter_note" not in variables["OMRÅDE"]
    assert variables["Tid"]["time_span"]["first"]["id"] == "2008K1"
    assert variables["Tid"]["time_span"]["count"] == 75
    assert variables["Tid"]["filter_note"] == "No match for the filter on this variable."
    assert "values" not in variables["Tid"]
    assert variables["KØN"]["values"]
    assert "No match" in variables["KØN"]["filter_note"]

    spaced = tabelinfo("FOLK1A", values_filter="Rudersdal Hørsholm")
    spaced_areas = next(item for item in spaced["variables"] if item["id"] == "OMRÅDE")
    assert {item["text"] for item in spaced_areas["values"]} == {"Rudersdal", "Hørsholm"}


def test_string_codes_are_coerced_to_lists(calls: list[dict[str, object]]) -> None:
    from statistikbanken_mcp.server import mcp

    tool = mcp._tool_manager.get_tool("hent_data")
    parsed = tool.fn_metadata.validate_arguments(
        {
            "table": "FOLK1A",
            "values": {"OMRÅDE": "Rudersdal, Hørsholm", "ALDER": "IALT", "Tid": ">=2016K3"},
        }
    )
    assert parsed["values"]["Tid"] == ">=2016K3"
    result = hent_data("FOLK1A", values=parsed["values"])
    assert result["rows"]
    sent = [call["body"] for call in calls if str(call["path"]).endswith("/data")][-1]
    assert isinstance(sent, dict)
    variables = {item["code"]: item["values"] for item in sent["variables"]}
    assert variables["OMRÅDE"] == ["230", "223"]
    assert variables["ALDER"] == ["IALT"]
    assert variables["Tid"] == [">=2016K3"]

    children = hent_data(
        "FOLK1A",
        values={"OMRÅDE": "230, 223"},
        sum_range={"ALDER": "0, 17"},
    )
    assert {row["OMRÅDE"] for row in children["rows"]} == {"Rudersdal", "Hørsholm"}
    assert children["rows"][0]["ALDER"] == "0-17"


def test_equivalent_age_expression_and_sum_from_are_one_request(
    calls: list[dict[str, object]],
) -> None:
    result = hent_data(
        "FOLK1A",
        values={"OMRÅDE": ["230"], "ALDER": [">=80"]},
        sum_from={"ALDER": "80"},
    )
    assert result["rows"][0]["ALDER"] == "80+"
    assert result["rows"][0]["value"] == 5045
    assert any("same request" in note for note in result["notes"])
    sent = [call["body"] for call in calls if str(call["path"]).endswith("/data")][-1]
    assert isinstance(sent, dict)
    ages = next(item["values"] for item in sent["variables"] if item["code"] == "ALDER")
    assert len(ages) == 1
    assert str(ages[0]).startswith("sum(80+")

    as_text = hent_data(
        "FOLK1A",
        values={"OMRÅDE": "230", "ALDER": ">=80"},
        sum_from={"ALDER": "80"},
    )
    assert as_text["rows"][0]["value"] == 5045


def test_sum_from_nested_in_values_is_moved_out(calls: list[dict[str, object]]) -> None:
    from statistikbanken_mcp.server import mcp

    tool = mcp._tool_manager.get_tool("hent_data")
    arguments = {"table": "FOLK1A", "values": {"OMRÅDE": "230", "sum_from": {"ALDER": "80"}}}
    tool.fn_metadata.validate_arguments(arguments)
    result = hent_data("FOLK1A", values=arguments["values"])
    assert result["rows"][0]["value"] == 5045
    assert any("Moved sum_from" in note for note in result["notes"])
    assert "validation error" not in " ".join(result["notes"]).casefold()

    with pytest.raises(ToolError, match="beside values") as caught:
        hent_data("FOLK1A", values={"OMRÅDE": "230", "sum_from": "80"})
    message = str(caught.value)
    assert "sum_from" in message
    assert 'sum_from {"ALDER": "80"}' in message
    assert "pydantic" not in message.casefold()
    assert "list_type" not in message


def test_tabelinfo_filter_finds_a_value_past_the_cap() -> None:
    info = tabelinfo("FOLK1A", variable="OMRÅDE", values_filter="Silkeborg")
    assert info["variables"] == [
        {
            "id": "OMRÅDE",
            "text": "område",
            "eliminable": True,
            "time": False,
            "values": [{"id": "740", "text": "Silkeborg"}],
            "value_count": 1,
        }
    ]


def test_explicit_code_sum_and_unpublished_zeros() -> None:
    ages = hent_data(
        "HFUDD11",
        values={"HFUDD": ["H70"], "Tid": ["2025"]},
        sum_range={
            "ALDER": ["25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64"]
        },
    )
    assert ages["rows"] == [
        {
            "HFUDD": "H70 Lange videregående uddannelser, LVU",
            "ALDER": "25-29+30-34+35-39+40-44+45-49+50-54+55-59+60-64",
            "Tid": "2025",
            "value": 493701,
        }
    ]
    quarters = hent_data(
        "STRAF10",
        values={"OVERTRÆD": ["1"]},
        sum_range={"Tid": ["2023K1", "2023K2", "2023K3", "2023K4"]},
    )
    assert quarters["rows"] == [
        {"OVERTRÆD": "Straffelov i alt", "Tid": "2023K1+2023K2+2023K3+2023K4", "value": 331828}
    ]
    hotel = hent_data(
        "HOTEL1",
        values={
            "OMRÅDE": ["000"],
            "NATION1": ["TOT"],
            "TAL": ["1030"],
            "PERIODE": ["07", "08", "09", "10"],
            "Tid": ["2026"],
        },
    )
    by_month = {row["PERIODE"]: row for row in hotel["rows"]}
    assert by_month["August"]["value"] == 3238862
    assert "unpublished" not in by_month["August"]
    assert by_month["September"]["unpublished"] is True
    assert by_month["September"]["value"] == 0
    assert by_month["Oktober"]["unpublished"] is True


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
        hent_data("FOLK1A", {"ALDER": ["IALT"]}, {"ALDER": "80"})
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
