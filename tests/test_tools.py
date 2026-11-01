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
            {"OMRÅDE": ["561"], "ALDER": ["sum(bad=99999)"], "Tid": ["(1)"]},
        )
    message = str(caught.value)
    assert "sum_from" in message
    assert "Traceback" not in message
    data_calls = [call for call in calls if str(call["path"]).endswith("/data")]
    assert data_calls


@pytest.mark.parametrize(
    "query",
    [
        "gennemsnitlig indkomst før skat",
        "indkomst i alt før skat",
        "gennemsnitsindkomst før skat",
    ],
)
def test_pre_tax_income_search_ranks_indkp105(query: str) -> None:
    ids = [hit["id"] for hit in find_tabel(query, limit=3)]
    assert "INDKP105" in ids


def test_labour_force_survey_keeps_the_whole_word() -> None:
    ids = [
        hit["id"] for hit in find_tabel("arbejdskraftundersøgelsen beskæftigelsesstatus", limit=3)
    ]
    assert "AKU101K" in ids


def test_income_before_tax_by_sex_is_not_overridden() -> None:
    assert find_tabel("indkomst før skat køn", limit=1)[0]["id"] == "INDKP221"


@pytest.mark.parametrize("query", ["foreningsmedlemmer", "foreningsmedlemmer Esbjerg"])
def test_club_members_are_not_board_members(query: str) -> None:
    hits = find_tabel(query, limit=15)
    ids = [hit["id"] for hit in hits]
    assert ids.index("IDRFOR01") < 3
    club = ids.index("IDRFOR01")
    for index, hit in enumerate(hits):
        if "bestyrelsesmedlemmer" in hit["title"].casefold():
            assert index > club


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


@pytest.mark.parametrize(
    ("query", "table_id"),
    [
        ("togpassagerer", "BANE21"),
        ("vareeksport", "KN8MEST"),
        ("dødsårsager", "DODA1"),
        ("gini", "IFOR41"),
        ("nye virksomheder", "DEMO11"),
        ("iværksættere", "DEMO11"),
        ("månedsfortjeneste", "LONS20"),
        ("AKU", "AKU100K"),
        ("påbegyndte boliger", "BYGV33"),
        ("unemployed", "AUS09"),
        ("jobless", "AUS09"),
    ],
)
def test_everyday_words_find_the_subject(query: str, table_id: str) -> None:
    ids = [hit["id"] for hit in find_tabel(query, limit=3)]
    assert table_id in ids


def test_denmark_does_not_outrank_the_subject() -> None:
    unemployment = [hit["id"] for hit in find_tabel("unemployment Denmark", limit=3)]
    assert "AUS09" in unemployment
    assert "BEF5F" not in unemployment

    cancer = [hit["id"] for hit in find_tabel("Hvor mange døde af kræft i Danmark", limit=3)]
    assert cancer[0] == "DODA1"
    assert "BEF5F" not in cancer

    assert find_tabel("befolkning i Danmark", limit=1)[0]["id"] == "FOLK1A"
    assert find_tabel("Danmark", limit=1)[0]["id"] == "BEF5F"
    assert find_tabel("Denmark", limit=1)[0]["id"] == "BEF5F"


def test_lyngby_suggests_lyngby_taarbaek_and_is_not_accepted(
    calls: list[dict[str, object]],
) -> None:
    with pytest.raises(ToolError, match=r"Lyngby-Taarbæk \(173\)") as caught:
        hent_data("FOLK1A", {"OMRÅDE": ["Lyngby"]})
    message = str(caught.value)
    assert "Did you mean" in message
    assert "Unknown value 'Lyngby'" in message
    with pytest.raises(ToolError, match=r"Høje-Taastrup \(169\)"):
        hent_data("FOLK1A", {"OMRÅDE": ["Taastrup"]})
    assert not any(str(call["path"]).endswith("/data") for call in calls)


def test_out_of_range_period_states_the_available_range() -> None:
    with pytest.raises(ToolError, match="Available: 2008K1") as caught:
        hent_data("FOLK1A", {"Tid": ["2030K1"]})
    message = str(caught.value)
    assert "2026K3" in message
    assert "Did you mean" not in message
    assert "2023K1" not in message

    with pytest.raises(ToolError, match="Available: 2008K1") as caught:
        hent_data("FOLK1A", {"Tid": ["1850"]})
    assert "Did you mean" not in str(caught.value)
    assert "2023K1" not in str(caught.value)


def test_bare_year_on_a_quarterly_table_suggests_the_quarters() -> None:
    with pytest.raises(ToolError, match=r">=2025K1<=2025K4") as caught:
        hent_data("STRAF10", {"Tid": ["2025"]})
    message = str(caught.value)
    assert "Available: 1995K1" in message
    assert "2026K3" in message

    with pytest.raises(ToolError, match=r">=2026K1<=2026K3"):
        hent_data("FOLK1A", {"Tid": ["2026"]})


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
    with pytest.raises(ToolError, match="Available: 2008K1"):
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
    with pytest.raises(ToolError, match="not every period") as caught:
        hent_data(
            "FOLK1A",
            values={"ALDER": [">=80"], "OMRÅDE": ["Esbjerg", "Viborg"]},
            sum_range={"Tid": ["2015K1", "2026K3"]},
        )
    message = str(caught.value)
    assert ">=2015K1" in message
    assert "22780" not in message
    assert "hide rows" in message
    assert not any(str(call["path"]).endswith("/data") for call in calls)


def test_two_period_sum_is_not_a_time_range(calls: list[dict[str, object]]) -> None:
    with pytest.raises(ToolError, match="not every period") as caught:
        hent_data(
            "FOLK1A",
            values={"OMRÅDE": ["561"], "ALDER": ["IALT"]},
            sum_range={"Tid": ["2015K1", "2026K3"]},
        )
    assert ">=2015K1" in str(caught.value)
    assert not any(str(call["path"]).endswith("/data") for call in calls)


def test_local_sum_keeps_each_series() -> None:
    from statistikbanken_mcp.statbank import _sum_rows

    rows = [
        {"OMRÅDE": "Esbjerg", "ALDER": "80 år", "Tid": "2015K1", "value": 4678},
        {"OMRÅDE": "Esbjerg", "ALDER": "80 år", "Tid": "2026K3", "value": 7556},
        {"OMRÅDE": "Viborg", "ALDER": "80 år", "Tid": "2015K1", "value": 4126},
        {"OMRÅDE": "Viborg", "ALDER": "80 år", "Tid": "2026K3", "value": 6420},
        {"OMRÅDE": "Esbjerg", "ALDER": "81 år", "Tid": "2015K1", "value": 1},
        {"OMRÅDE": "Esbjerg", "ALDER": "81 år", "Tid": "2026K3", "value": 2},
    ]
    summed = _sum_rows(rows, "Tid", ["2015K1", "2026K3"])
    by_series = {(row["OMRÅDE"], row["ALDER"]): row for row in summed}
    assert len(by_series) == 3
    assert by_series[("Esbjerg", "80 år")]["value"] == 12234
    assert by_series[("Viborg", "80 år")]["value"] == 10546
    assert by_series[("Esbjerg", "81 år")]["Tid"] == "2015K1+2026K3"
    assert 22780 not in {row["value"] for row in summed}


def test_values_filter_splits_words_and_keeps_unmatched_variables() -> None:
    info = tabelinfo("FOLK1A", values_filter="Esbjerg, Viborg")
    variables = {item["id"]: item for item in info["variables"]}
    areas = {item["id"]: item["text"] for item in variables["OMRÅDE"]["values"]}
    assert areas == {"791": "Viborg", "561": "Esbjerg"}
    assert "filter_note" not in variables["OMRÅDE"]
    assert variables["Tid"]["time_span"]["first"]["id"] == "2008K1"
    assert variables["Tid"]["time_span"]["count"] == 75
    assert variables["Tid"]["filter_note"] == "No match for the filter on this variable."
    assert "values" not in variables["Tid"]
    assert variables["KØN"]["values"]
    assert "No match" in variables["KØN"]["filter_note"]

    spaced = tabelinfo("FOLK1A", values_filter="Esbjerg Viborg")
    spaced_areas = next(item for item in spaced["variables"] if item["id"] == "OMRÅDE")
    assert {item["text"] for item in spaced_areas["values"]} == {"Esbjerg", "Viborg"}


def test_string_codes_are_coerced_to_lists(calls: list[dict[str, object]]) -> None:
    from statistikbanken_mcp.server import mcp

    tool = mcp._tool_manager.get_tool("hent_data")
    parsed = tool.fn_metadata.validate_arguments(
        {
            "table": "FOLK1A",
            "values": {"OMRÅDE": "Esbjerg, Viborg", "ALDER": "IALT", "Tid": ">=2016K3"},
        }
    )
    assert parsed["values"]["Tid"] == ">=2016K3"
    result = hent_data("FOLK1A", values=parsed["values"])
    assert result["rows"]
    sent = [call["body"] for call in calls if str(call["path"]).endswith("/data")][-1]
    assert isinstance(sent, dict)
    variables = {item["code"]: item["values"] for item in sent["variables"]}
    assert variables["OMRÅDE"] == ["561", "791"]
    assert variables["ALDER"] == ["IALT"]
    assert variables["Tid"] == [">=2016K3"]

    children = hent_data(
        "FOLK1A",
        values={"OMRÅDE": "561, 791"},
        sum_range={"ALDER": "0, 17"},
    )
    assert {row["OMRÅDE"] for row in children["rows"]} == {"Esbjerg", "Viborg"}
    assert children["rows"][0]["ALDER"] == "0-17"


def test_equivalent_age_expression_and_sum_from_are_one_request(
    calls: list[dict[str, object]],
) -> None:
    result = hent_data(
        "FOLK1A",
        values={"OMRÅDE": ["561"], "ALDER": [">=80"]},
        sum_from={"ALDER": "80"},
    )
    assert result["rows"][0]["ALDER"] == "80+"
    assert result["rows"][0]["value"] == 7556
    assert any("sum_from" in note for note in result["notes"])
    sent = [call["body"] for call in calls if str(call["path"]).endswith("/data")][-1]
    assert isinstance(sent, dict)
    ages = next(item["values"] for item in sent["variables"] if item["code"] == "ALDER")
    assert len(ages) == 1
    assert str(ages[0]).startswith("sum(80+")

    as_text = hent_data(
        "FOLK1A",
        values={"OMRÅDE": "561", "ALDER": ">=80"},
        sum_from={"ALDER": "80"},
    )
    assert as_text["rows"][0]["value"] == 7556


def test_sum_from_nested_in_values_is_moved_out(calls: list[dict[str, object]]) -> None:
    from statistikbanken_mcp.server import mcp

    tool = mcp._tool_manager.get_tool("hent_data")
    arguments = {"table": "FOLK1A", "values": {"OMRÅDE": "561", "sum_from": {"ALDER": "80"}}}
    tool.fn_metadata.validate_arguments(arguments)
    result = hent_data("FOLK1A", values=arguments["values"])
    assert result["rows"][0]["value"] == 7556
    assert any("Moved sum_from" in note for note in result["notes"])
    assert "validation error" not in " ".join(result["notes"]).casefold()

    with pytest.raises(ToolError, match="beside values") as caught:
        hent_data("FOLK1A", values={"OMRÅDE": "561", "sum_from": "80"})
    message = str(caught.value)
    assert "sum_from" in message
    assert 'sum_from {"ALDER": "80"}' in message
    assert "pydantic" not in message.casefold()
    assert "list_type" not in message


def test_open_ended_age_becomes_sum_from(calls: list[dict[str, object]]) -> None:
    phrases = ["80+", ">=80", "80-", "80 og derover", "over 80"]
    for phrase in phrases:
        result = hent_data("FOLK1A", values={"OMRÅDE": ["561"], "ALDER": [phrase]})
        assert result["rows"][0]["ALDER"] == "80+", phrase
        assert result["rows"][0]["value"] == 7556, phrase
        assert any("sum_from" in note and phrase in note for note in result["notes"]), phrase
    sent = [call["body"] for call in calls if str(call["path"]).endswith("/data")][-1]
    assert isinstance(sent, dict)
    ages = next(item["values"] for item in sent["variables"] if item["code"] == "ALDER")
    assert str(ages[0]).startswith("sum(80+")


def test_plain_age_80_stays_one_year(calls: list[dict[str, object]]) -> None:
    with pytest.raises(ToolError):
        hent_data("FOLK1A", {"OMRÅDE": ["561"], "ALDER": ["80"], "Tid": ["2026K3"]})
    sent = [call["body"] for call in calls if str(call["path"]).endswith("/data")][-1]
    assert isinstance(sent, dict)
    ages = next(item["values"] for item in sent["variables"] if item["code"] == "ALDER")
    assert ages == ["80"]


def test_closed_age_text_becomes_sum_range() -> None:
    result = hent_data("FOLK1A", values={"OMRÅDE": ["561", "791"], "ALDER": ["0-17"]})
    assert {row["OMRÅDE"] for row in result["rows"]} == {"Esbjerg", "Viborg"}
    assert result["rows"][0]["ALDER"] == "0-17"
    assert any('sum_range {"ALDER": ["0", "17"]}' in note for note in result["notes"])


def test_kom_is_an_alias_for_area() -> None:
    result = hent_data("FOLK1A", values={"KOM": "Esbjerg", "alder": "80+"})
    assert result["rows"][0]["OMRÅDE"] == "Esbjerg"
    assert result["rows"][0]["value"] == 7556
    assert any("Used OMRÅDE for KOM." in note for note in result["notes"])


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
    result = hent_data("FOLK1A", {"område": [" 561 "], "alder": ["ialt"]})
    assert result["rows"] == [
        {
            "OMRÅDE": "Esbjerg",
            "ALDER": "Alder i alt",
            "Tid": "2026K3",
            "value": 114824,
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
