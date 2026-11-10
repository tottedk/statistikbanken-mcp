"""Tool behaviour against recorded API responses."""

import json
from datetime import date

import httpx
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from statistikbanken_mcp.server import find_tabel, hent_data, set_client, tabelinfo
from statistikbanken_mcp.statbank import StatbankClient


def test_subject_search_ranks_the_published_table() -> None:
    assert find_tabel("gennemsnitlig indkomst Danmark", limit=5)[0]["id"] == "INDKP105"
    assert find_tabel("levendefødte Aarhus", limit=5)[0]["id"] == "BEV22"
    assert find_tabel("beskæftigede nationalregnskab", limit=5)[0]["id"] == "NAN1"
    assert find_tabel("bruttoværditilvækst Danmark", limit=3)[0]["id"] == "NABP19"


def test_find_tabel_befolkning_leads_with_folk1a() -> None:
    hits = find_tabel("befolkning", limit=5)
    assert hits[0] == {
        "id": "FOLK1A",
        "title": "Befolkningen den 1. i kvartalet",
        "unit": "Antal",
        "first_period": "2008K1",
        "latest_period": "2026K3",
        "variables": ["område", "køn", "alder", "civilstand", "tid"],
    }
    assert len(hits) == 5


def test_net_and_annual_unemployment_rank_their_own_tables() -> None:
    net = [hit["id"] for hit in find_tabel("nettoledige Danmark sæsonkorrigeret", limit=5)]
    assert net[0] == "AUS07"
    annual = [hit["id"] for hit in find_tabel("fuldtidsledige Esbjerg gennemsnit 2025", limit=5)]
    assert annual[0] == "AUL01"
    gross = [hit["id"] for hit in find_tabel("bruttoledighed", limit=5)]
    assert gross[0] == "AUS08"


def test_a_year_before_folk1a_ranks_a_table_that_covers_it() -> None:
    hits = find_tabel("befolkning siden 1934", limit=8)
    assert hits[0]["id"] == "BEFOLK2"
    assert hits[0]["first_period"] == "1901"
    assert find_tabel("FOLK1A", limit=1)[0]["first_period"] == "2008K1"
    place = find_tabel("befolkning Viborg siden 2000", limit=3)
    assert place[0]["id"] == "FOLK1A"
    assert place[0]["first_period"] == "2008K1"


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


def test_a_short_name_resolves_to_the_hyphenated_municipality(
    calls: list[dict[str, object]],
) -> None:
    from statistikbanken_mcp.statbank import select_named_groups

    with pytest.raises(ToolError):
        hent_data("FOLK1A", {"OMRÅDE": ["Lyngby"], "ALDER": ["IALT"], "Tid": ["2026K3"]})
    sent = [call["body"] for call in calls if str(call["path"]).endswith("/data")][-1]
    assert isinstance(sent, dict)
    areas = next(item["values"] for item in sent["variables"] if item["code"] == "OMRÅDE")
    assert areas == ["173"]
    chosen = select_named_groups(
        "Hvor mange bor i Lyngby?",
        {
            "variables": [
                {
                    "id": "OMRÅDE",
                    "values": [
                        {"id": "173", "text": "Lyngby-Taarbæk"},
                        {"id": "101", "text": "København"},
                    ],
                }
            ]
        },
        {},
    )
    assert chosen["OMRÅDE"] == ["173"]


def test_food_words_do_not_select_the_price_index() -> None:
    """Food and price words are not a table rule. The index name still is."""
    food = find_tabel("årsstigningen i priserne på fødevarer", limit=5)
    assert food[0]["id"] != "PRIS111"
    assert find_tabel("forbrugerprisindekset i alt", limit=3)[0]["id"] == "PRIS111"


def test_the_unemployment_rate_ranks_above_the_count() -> None:
    assert find_tabel("ledighedsprocenten i Odense", limit=5)[0]["id"] == "AUP01"
    assert find_tabel("forbrugerprisindekset i alt", limit=3)[0]["id"] == "PRIS111"


def test_lyngby_suggests_lyngby_taarbaek_and_is_not_accepted(
    calls: list[dict[str, object]],
) -> None:
    """A unique first part is the municipality. The old suggestion is no longer an error."""
    with pytest.raises(ToolError):
        hent_data("FOLK1A", {"OMRÅDE": ["Lyngby"]})
    sent = [call["body"] for call in calls if str(call["path"]).endswith("/data")][-1]
    assert isinstance(sent, dict)
    areas = next(item["values"] for item in sent["variables"] if item["code"] == "OMRÅDE")
    assert areas == ["173"]
    calls.clear()
    with pytest.raises(ToolError):
        hent_data("FOLK1A", {"OMRÅDE": ["Taastrup"]})
    sent = [call["body"] for call in calls if str(call["path"]).endswith("/data")][-1]
    assert isinstance(sent, dict)
    areas = next(item["values"] for item in sent["variables"] if item["code"] == "OMRÅDE")
    assert areas == ["169"]


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


def test_bare_year_outside_the_table_states_the_range() -> None:
    with pytest.raises(ToolError, match="Available: 2008K1") as caught:
        hent_data("FOLK1A", {"OMRÅDE": ["Esbjerg"], "ALDER": ["IALT"], "Tid": ["1850"]})
    message = str(caught.value)
    assert "2026K3" in message
    assert "Did you mean" not in message


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


def test_a_comma_between_year_bounds_is_a_closed_range(
    calls: list[dict[str, object]],
) -> None:
    hent_data("FOLK1A", {"OMRÅDE": ["000"], "Tid": [">=1995,<=2001"]})
    sent = [call["body"] for call in calls if str(call["path"]).endswith("/data")][-1]
    assert isinstance(sent, dict)
    tid = next(item["values"] for item in sent["variables"] if item["code"] == "Tid")
    assert tid == [">=1995<=2001"]


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
            "total": {"id": "000", "text": "Hele landet"},
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


def test_unemployment_words_rank_aus_tables() -> None:
    for query in ("arbejdsløshed", "ledighed", "Arbejdsløshed"):
        ids = [hit["id"] for hit in find_tabel(query, limit=5)]
        assert ids[0].startswith("AUS"), (query, ids)
        assert any(table_id.startswith("AUS") for table_id in ids[:3])
    with pytest.raises(ToolError, match="No table matches"):
        find_tabel("arbejdsløshedxyz")


def test_hyphenated_query_matches_joined_spaced_and_hyphenated_titles() -> None:
    from statistikbanken_mcp.statbank import _query_tokens, _score_table

    ids = [hit["id"] for hit in find_tabel("fuldtids-ledige", limit=5)]
    assert ids[0].startswith("AUS"), ids
    tokens = _query_tokens("fuldtids-ledige")
    folded = "fuldtids-ledige"
    assert _score_table(_title("Fuldtidsledige"), tokens, folded)
    assert _score_table(_title("Fuldtids ledige"), tokens, folded)
    assert _score_table(_title("Fuldtids-ledige"), tokens, folded)
    assert _score_table(_title("Befolkning"), tokens, folded) is None
    with pytest.raises(ToolError, match="No table matches"):
        find_tabel("zzzz-qqqq")


def _title(text: str) -> dict[str, object]:
    return {"id": "X", "text": text, "variables": [], "active": True}


def test_a_registration_title_is_a_flow_and_unemployment_is_a_level() -> None:
    from statistikbanken_mcp.statbank import _one_measure

    assert _one_measure("Nyregistrerede personbiler Antal") == "sum"
    assert _one_measure("Fuldtidsledige (sæsonkorrigeret) -") == "average"


def test_an_open_range_collapses_one_complete_year_and_keeps_a_long_series() -> None:
    from statistikbanken_mcp.statbank import (
        _collapse_years,
        _plans_from_open_range,
        _Table,
        _Variable,
    )

    months = tuple(
        {"id": f"2025M{month:02d}", "text": f"2025M{month:02d}"} for month in range(1, 13)
    ) + tuple({"id": f"2026M{month:02d}", "text": f"2026M{month:02d}"} for month in range(1, 4))
    table = _Table(
        "BIL55",
        "Nyregistrerede personbiler",
        "",
        "Antal",
        None,
        True,
        (_Variable("Tid", "tid", False, True, months),),
    )
    rows = [{"Tid": f"2025M{month:02d}", "value": 10} for month in range(1, 13)]
    rows += [{"Tid": f"2026M{month:02d}", "value": 1} for month in range(1, 4)]
    plans = _plans_from_open_range(table, {"Tid": [">=2025M01"]}, rows)
    collapsed, _notes = _collapse_years(table, rows, plans)
    year = next(row for row in collapsed if row["Tid"] == "2025")
    assert year["value"] == 120
    assert year["year_total"]["method"] == "sum"
    assert any(row["Tid"] == "2026M01" for row in collapsed)
    long_rows = [{"Tid": f"{year}K1", "value": 1} for year in range(2016, 2027)]
    assert _plans_from_open_range(table, {"Tid": [">=2016K1"]}, long_rows) == []


def test_omitted_dwelling_status_adds_occupied_and_unoccupied_only() -> None:
    from statistikbanken_mcp.statbank import _headline_sums, _Table, _Variable

    table = _Table(
        "BOL101",
        "Boliger",
        "",
        "Antal",
        None,
        True,
        (
            _Variable(
                "BEBO",
                "beboertype",
                False,
                False,
                (
                    {"id": "1000", "text": "beboede"},
                    {"id": "2000", "text": "ubeboede"},
                    {"id": "5000", "text": "fritidshuse"},
                ),
            ),
        ),
    )
    selections: dict[str, list[str]] = {}
    sums = _headline_sums(table, selections)
    assert selections["BEBO"] == ["1000", "2000"]
    assert sums and "5000" not in sums[0][1]
    explicit = {"BEBO": ["1000", "2000", "5000"]}
    assert _headline_sums(table, explicit) == []


def test_a_timed_out_post_is_tried_once_more() -> None:
    from statistikbanken_mcp.statbank import StatbankClient

    calls = {"n": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.TimeoutException("timed out", request=request)
        return httpx.Response(200, json={"ok": True})

    client = StatbankClient(
        httpx.Client(
            base_url="https://api.statbank.dk/v1",
            transport=httpx.MockTransport(handle),
        )
    )
    assert client._post("data", {"table": "FOLK1A"}) == {"ok": True}
    assert calls["n"] == 2
    client.close()


def test_year_on_a_flow_table_is_the_sum() -> None:
    result = hent_data("STRAF10", {"OVERTRÆD": ["1"], "Tid": ["2025"]})
    assert result["rows"] == [
        {
            "OVERTRÆD": "Straffelov i alt",
            "Tid": "2025",
            "value": 305839,
            "year_total": {
                "method": "sum",
                "measure": "flow",
                "periods": ["2025K1", "2025K2", "2025K3", "2025K4"],
            },
        }
    ]
    note = " ".join(result["notes"])
    assert "sum of 2025K1, 2025K2, 2025K3 and 2025K4" in note
    assert "flow" in note
    assert result["periods"] == ["2025"]


def test_year_on_a_level_table_is_the_average() -> None:
    result = hent_data("FOLK1A", {"OMRÅDE": ["Esbjerg"], "ALDER": ["IALT"], "Tid": ["2025"]})
    assert result["rows"] == [
        {
            "OMRÅDE": "Esbjerg",
            "ALDER": "Alder i alt",
            "Tid": "2025",
            "value": 115029,
            "year_total": {
                "method": "average",
                "measure": "level",
                "periods": ["2025K1", "2025K2", "2025K3", "2025K4"],
            },
        }
    ]
    note = " ".join(result["notes"])
    assert "average of 2025K1, 2025K2, 2025K3 and 2025K4" in note
    assert "level" in note


def test_incomplete_year_is_not_collapsed() -> None:
    crimes = hent_data("STRAF10", {"OVERTRÆD": ["1"], "Tid": ["2026"]})
    assert [row["Tid"] for row in crimes["rows"]] == ["2026K1", "2026K2", "2026K3"]
    assert [row["value"] for row in crimes["rows"]] == [69970, 76275, 78389]
    assert "not a full year" in " ".join(crimes["notes"])
    assert "No yearly figure" in " ".join(crimes["notes"])
    population = hent_data("FOLK1A", {"OMRÅDE": ["Esbjerg"], "ALDER": ["IALT"], "Tid": ["i år"]})
    assert [row["Tid"] for row in population["rows"]] == ["2026K1", "2026K2", "2026K3"]
    assert "Read 'i år' as 2026." in population["notes"]
    assert "No yearly figure" in " ".join(population["notes"])


def test_row_label_chooses_sum_average_or_neither() -> None:
    births = hent_data(
        "BEV22",
        {"OMRÅDE": ["Esbjerg"], "BEVÆGELSE": ["B02"], "Tid": ["2024"]},
    )
    assert births["rows"] == [
        {
            "OMRÅDE": "Esbjerg",
            "BEVÆGELSE": "Levendefødte",
            "Tid": "2024",
            "value": 1025,
            "year_total": {
                "method": "sum",
                "measure": "flow",
                "periods": ["2024K1", "2024K2", "2024K3", "2024K4"],
            },
        }
    ]
    stock = hent_data(
        "BEV22",
        {"OMRÅDE": ["Esbjerg"], "BEVÆGELSE": ["B20K"], "Tid": ["2024"]},
    )
    assert stock["rows"][0]["Tid"] == "2024"
    assert stock["rows"][0]["value"] == 115235.5
    assert stock["rows"][0]["year_total"]["method"] == "average"
    corrections = hent_data(
        "BEV22",
        {"OMRÅDE": ["Esbjerg"], "BEVÆGELSE": ["B12"], "Tid": ["2024"]},
    )
    assert [row["Tid"] for row in corrections["rows"]] == ["2024K1", "2024K2", "2024K3", "2024K4"]
    assert [row["value"] for row in corrections["rows"]] == [-1, 0, -8, 3]
    assert "does not say whether" in " ".join(corrections["notes"])
    assert "listed separately" in " ".join(corrections["notes"])


def test_danish_period_phrases_map_to_codes() -> None:
    latest = hent_data(
        "FOLK1A",
        {"OMRÅDE": ["Esbjerg"], "ALDER": ["IALT"], "Tid": ["seneste"]},
    )
    assert latest["rows"][0]["Tid"] == "2026K3"
    assert latest["rows"][0]["value"] == 114824
    quarter = hent_data(
        "FOLK1A",
        {"OMRÅDE": ["Esbjerg"], "ALDER": ["IALT"], "Tid": ["3. kvartal 2026"]},
    )
    assert quarter["rows"][0]["Tid"] == "2026K3"
    coded = hent_data(
        "FOLK1A",
        {"OMRÅDE": ["Esbjerg"], "ALDER": ["IALT"], "Tid": ["K3 2026"]},
    )
    assert coded["rows"][0]["value"] == 114824
    last_year = hent_data(
        "FOLK1A",
        {"OMRÅDE": ["Esbjerg"], "ALDER": ["IALT"], "Tid": ["i fjor"]},
    )
    assert last_year["rows"][0]["Tid"] == "2025"
    assert last_year["rows"][0]["value"] == 115029
    assert "Read 'i fjor' as 2025." in last_year["notes"]
    also = hent_data(
        "FOLK1A",
        {"OMRÅDE": ["Esbjerg"], "ALDER": ["IALT"], "Tid": ["sidste år"]},
    )
    assert also["rows"][0]["value"] == 115029
    previous = hent_data(
        "FOLK1A",
        {"OMRÅDE": ["Esbjerg"], "ALDER": ["IALT"], "Tid": ["sidste kvartal"]},
    )
    assert previous["rows"][0]["Tid"] == "2026K2"
    assert previous["rows"][0]["value"] == 114867
    assert "Read 'sidste kvartal' as 2026K2." in previous["notes"]


def test_calendar_phrases_do_not_substitute_another_period(calls: list[dict[str, object]]) -> None:
    from datetime import date

    from statistikbanken_mcp.statbank import StatbankError, _period_meaning, _Table, _Variable

    result = hent_data(
        "IDRFOR01",
        {
            "BLSTKOM": ["561"],
            "ORGANISATION": ["100"],
            "AKTP": ["104", "105"],
            "Tid": ["i fjor"],
        },
    )
    sent = [call["body"] for call in calls if str(call["path"]).endswith("/data")][-1]
    assert isinstance(sent, dict)
    tid = next(item["values"] for item in sent["variables"] if item["code"] == "Tid")
    assert tid == ["2025"]
    assert {row["AKTP"]: row["value"] for row in result["rows"]} == {
        "Foreninger": 212,
        "Medlemmer": 54205,
    }
    calls.clear()
    with pytest.raises(ToolError, match="not published") as caught:
        hent_data(
            "IDRFOR01",
            {
                "BLSTKOM": ["561"],
                "ORGANISATION": ["100"],
                "AKTP": ["104", "105"],
                "Tid": ["i år"],
            },
        )
    message = str(caught.value)
    assert "2026" in message
    assert "Latest available period: 2025" in message
    assert not any(str(call["path"]).endswith("/data") for call in calls)

    codes = [f"2026M{month:02d}" for month in range(1, 10)]
    variable = _Variable(
        id="Tid",
        text="tid",
        eliminable=False,
        time=True,
        values=tuple({"id": code, "text": code} for code in codes),
    )
    table = _Table(
        id="AUS07",
        title="Fuldtidsledige",
        description="",
        unit="Antal",
        updated=None,
        active=True,
        variables=(variable,),
    )
    today = date(2026, 11, 2)
    with pytest.raises(StatbankError, match="2026M10") as missed:
        _period_meaning(table, variable, "sidste måned", today)
    assert "Latest available period: 2026M09" in str(missed.value)
    published = _Variable(
        id="Tid",
        text="tid",
        eliminable=False,
        time=True,
        values=tuple({"id": code, "text": code} for code in [*codes, "2026M10"]),
    )
    present = _Table(
        id="AUS07",
        title="Fuldtidsledige",
        description="",
        unit="Antal",
        updated=None,
        active=True,
        variables=(published,),
    )
    found, _plan, read_as = _period_meaning(present, published, "sidste måned", today)
    assert found == ["2026M10"]
    assert read_as == "2026M10"
    latest, _plan, read_as = _period_meaning(table, variable, "seneste", today)
    assert latest == ["2026M09"]
    assert read_as == "2026M09"
    span = _Variable(
        id="Tid",
        text="tid",
        eliminable=False,
        time=True,
        values=tuple({"id": code, "text": code} for code in ("2024M09", "2025M09", "2026M09")),
    )
    span_table = _Table(
        id="AUS07",
        title="Fuldtidsledige",
        description="",
        unit="Antal",
        updated=None,
        active=True,
        variables=(span,),
    )
    now, _plan, read_as = _period_meaning(span_table, span, "nu", today)
    assert now == ["2026M09"]
    today_phrase, _plan, read_as = _period_meaning(span_table, span, "i dag", today)
    assert today_phrase == ["2026M09"]
    ago, _plan, read_as = _period_meaning(span_table, span, "for 2 år siden", today)
    assert ago == ["2024M09"]
    assert "2024M09" in read_as


def test_danish_period_phrases_outside_the_table_name_the_range() -> None:
    with pytest.raises(ToolError, match="2025M03") as caught:
        hent_data("FOLK1A", {"OMRÅDE": ["Esbjerg"], "Tid": ["marts 2025"]})
    message = str(caught.value)
    assert "Available: 2008K1" in message
    assert "2026K3" in message
    with pytest.raises(ToolError, match="not a quarter") as caught:
        hent_data("FOLK1A", {"Tid": ["5. kvartal 2025"]})
    assert "Available: 2008K1" in str(caught.value)
    with pytest.raises(ToolError, match="Available: 1987") as caught:
        hent_data("INDKP105", {"Tid": ["sidste kvartal"]})
    assert "2025" in str(caught.value)
    with pytest.raises(ToolError, match="not in table FOLK1A") as caught:
        hent_data("FOLK1A", {"Tid": ["3. kvartal 1990"]})
    assert "Available: 2008K1" in str(caught.value)


def test_projection_tables_and_forecast_rows_are_labelled() -> None:
    info = tabelinfo("FRKM126")
    assert info["projection"] is True
    assert "projection" not in tabelinfo("FOLK1A")
    hits = find_tabel("befolkningsfremskrivning", limit=5)
    assert hits[0]["projection"] is True
    assert "projection" not in find_tabel("befolkning", limit=1)[0]
    projected = hent_data(
        "FRKM126",
        {"KOMMUNEDK": ["561"], "ALDER": ["TOT"], "KØN": ["M", "K"], "Tid": ["2030"]},
    )
    assert projected["projection"] is True
    assert "fremskrivning" in " ".join(projected["notes"])
    by_sex = {row["KØN"]: row for row in projected["rows"]}
    assert by_sex["Mænd"]["value"] == 57160
    assert by_sex["Kvinder"]["value"] == 56898
    assert all(row["projection"] is True for row in projected["rows"])
    forecast = hent_data(
        "HST5",
        {"AFGRØDE": ["H110"], "ENHED": ["3A", "4A"], "Tid": ["2026"]},
    )
    by_unit = {row["ENHED"]: row for row in forecast["rows"]}
    assert by_unit["Prognose"]["value"] == 426
    assert by_unit["Prognose"]["projection"] is True
    assert by_unit["Høstet areal"]["value"] is None
    assert "projection" not in by_unit["Høstet areal"]
    assert forecast["projection"] is True
    assert "forecast" in " ".join(forecast["notes"]).casefold()


def _sent_variables(calls: list[dict[str, object]]) -> dict[str, list[str]]:
    sent = [call["body"] for call in calls if str(call["path"]).endswith("/data")][-1]
    assert isinstance(sent, dict)
    variables = sent["variables"]
    assert isinstance(variables, list)
    return {str(item["code"]): list(item["values"]) for item in variables if isinstance(item, dict)}


def test_population_by_municipality_or_age_ranks_folk1a_first() -> None:
    for query in (
        "befolkning kommune alder",
        "befolkning efter kommune og alder",
        "population by municipality and age",
    ):
        ids = [hit["id"] for hit in find_tabel(query, limit=8)]
        assert ids[0] == "FOLK1A", query
        assert ids.index("FOLK1A") < ids.index("BEFOLK3"), query
    forecast = [hit["id"] for hit in find_tabel("befolkningsfremskrivning kommune", limit=3)]
    assert forecast[0] == "FRKM126"


@pytest.mark.parametrize(
    "query",
    ["registrerede ledige", "registreret ledighed", "bruttoledighed", "arbejdsløshed"],
)
def test_registered_unemployment_ranks_aus08_first(query: str) -> None:
    ids = [hit["id"] for hit in find_tabel(query, limit=8)]
    assert ids[0] == "AUS08", (query, ids)
    for other in ("AKU520K", "AUF01"):
        if other in ids:
            assert ids.index("AUS08") < ids.index(other), (query, ids)


def test_unemployment_by_age_and_sex_is_not_the_area_table() -> None:
    hit = find_tabel("ledighed efter alder og koen", limit=5)[0]
    assert hit["id"] == "AUF01"
    assert any("alder" in name.casefold() for name in hit["variables"])


def test_job_vacancies_are_not_ranked_as_unemployment() -> None:
    assert find_tabel("ledige stillinger", limit=1)[0]["id"].startswith("LSK")


@pytest.mark.parametrize(
    ("query", "table_id"),
    [
        ("elpris", "PRIS01"),
        ("brændstof benzin og diesel", "PRIS01"),
        ("aktive firmaer", "GF11"),
        ("udlån fra folkebibliotekerne", "BIB3A"),
        ("hvor mange boliger", "BOL101"),
        ("ejerboliger", "BOL101"),
        ("vareeksport", "UHM"),
        ("kontanthjælp", "KY21"),
        ("arbejdsstyrken ifølge AKU", "AKU100K"),
        ("enlige mødre indkomst", "INDKF111"),
        ("indvandrere i Rudersdal kommune", "FOLK1C"),
    ],
)
def test_headline_topics_rank_the_preferred_table(query: str, table_id: str) -> None:
    assert find_tabel(query, limit=5)[0]["id"] == table_id, query


def test_preference_unless_turns_the_boost_off() -> None:
    from statistikbanken_mcp.statbank import _preference_boost

    assert _preference_boost("PRIS01", "elpris", set()) == 400
    assert _preference_boost("PRIS01", "HICP inflation", set()) == 0
    assert _preference_boost("GF11", "aktive firmaer", set()) == 400
    assert _preference_boost("GF11", "nye virksomheder", set()) == 0
    assert _preference_boost("BIB3A", "udlån folkebibliotek", set()) == 400
    assert _preference_boost("BIB3A", "interurban udlån", set()) == 0
    assert _preference_boost("BOL101", "hvor mange boliger", set()) == 400
    assert _preference_boost("BOL101", "påbegyndte boliger", set()) == 0
    assert _preference_boost("AKU100K", "arbejdsstyrken ifølge AKU", set()) == 400
    assert _preference_boost("AKU100K", "AKU ledighed", set()) == 0


def test_a_named_table_beats_the_preference() -> None:
    assert find_tabel("ledighed AUF01", limit=5)[0]["id"] == "AUF01"
    assert find_tabel("AUF01 ledighed", limit=5)[0]["id"] == "AUF01"


def test_a_breakdown_word_beats_the_preference() -> None:
    aku = [hit["id"] for hit in find_tabel("AKU ledighed", limit=5)]
    assert aku[0].startswith("AKU"), aku
    assert find_tabel("befolkning efter herkomst", limit=5)[0]["id"] == "FOLK1E"
    assert find_tabel("befolkning kommune herkomst", limit=5)[0]["id"] == "FOLK1E"


def test_personal_income_ranks_indkp105_first() -> None:
    ids = [hit["id"] for hit in find_tabel("personlig indkomst", limit=8)]
    assert ids[0] == "INDKP105"
    assert ids.index("INDKP105") < ids.index("INDKP222")
    english = [hit["id"] for hit in find_tabel("personal income", limit=5)]
    assert english[0] == "INDKP105"


def test_reported_crimes_rank_straf10_first() -> None:
    ids = [hit["id"] for hit in find_tabel("anmeldte forbrydelser", limit=6)]
    assert ids[0] == "STRAF10"
    assert ids.index("STRAF10") < ids.index("STRAF20")
    victims = [hit["id"] for hit in find_tabel("ofre for anmeldte forbrydelser", limit=3)]
    assert victims[0] == "STRAF5"


def test_omitted_variables_default_to_the_total_and_the_latest_period(
    calls: list[dict[str, object]],
) -> None:
    result = hent_data("FOLK1A", {"OMRÅDE": ["Esbjerg"]})
    assert result["rows"][0]["value"] == 114824
    assert result["rows"][0]["Tid"] == "2026K3"
    chosen = {(item["variable"], item["id"], item["text"]) for item in result["defaults"]}
    assert ("KØN", "TOT", "I alt") in chosen
    assert ("CIVILSTAND", "TOT", "I alt") in chosen
    assert ("ALDER", "IALT", "Alder i alt") in chosen
    assert ("Tid", "2026K3", "2026K3") in chosen
    assert all(item["variable"] != "OMRÅDE" for item in result["defaults"])
    sent = _sent_variables(calls)
    assert sent["OMRÅDE"] == ["561"]
    assert sent["KØN"] == ["TOT"]
    assert sent["ALDER"] == ["IALT"]
    assert sent["CIVILSTAND"] == ["TOT"]
    assert sent["Tid"] == ["2026K3"]


def test_an_explicit_period_and_category_are_kept(calls: list[dict[str, object]]) -> None:
    result = hent_data(
        "FOLK1A",
        {"OMRÅDE": ["Esbjerg"], "KØN": ["1"], "ALDER": ["IALT"], "Tid": ["2025"]},
    )
    assert result["rows"][0]["Tid"] == "2025"
    filled = {item["variable"] for item in result["defaults"]}
    assert "KØN" not in filled
    assert "Tid" not in filled
    assert "CIVILSTAND" in filled
    sent = _sent_variables(calls)
    assert sent["KØN"] == ["1"]
    assert sent["Tid"] == ["2025K1", "2025K2", "2025K3", "2025K4"]


def test_omitted_area_and_income_groups_use_total_labels(
    calls: list[dict[str, object]],
) -> None:
    income = hent_data(
        "INDKP105",
        {"OMRÅDE": ["Esbjerg"], "ENHED": ["118"], "Tid": ["2025"]},
    )
    by_var = {item["variable"]: item for item in income["defaults"]}
    assert by_var["KOEN"] == {
        "variable": "KOEN",
        "id": "MOK",
        "text": "Mænd og kvinder i alt",
    }
    assert by_var["ALDER1"]["text"] == "I alt"
    assert by_var["INDKINTB"]["text"] == "Alle"
    assert "ENHED" not in by_var
    assert "OMRÅDE" not in by_var
    assert "Tid" not in by_var
    sent = _sent_variables(calls)
    assert sent["KOEN"] == ["MOK"]
    assert sent["ALDER1"] == ["00"]
    assert sent["INDKINTB"] == ["000"]

    calls.clear()
    hotel = hent_data("HOTEL1", {"TAL": ["1030"], "PERIODE": ["08"], "Tid": ["2026"]})
    by_var = {item["variable"]: item for item in hotel["defaults"]}
    assert by_var["OMRÅDE"]["id"] == "000"
    assert by_var["OMRÅDE"]["text"] == "Hele landet"
    assert by_var["NATION1"]["text"] == "I alt"


def _values(variable_id: str, pairs: list[tuple[str, str]]) -> dict[str, object]:
    return {
        "id": variable_id,
        "text": variable_id,
        "elimination": False,
        "time": False,
        "values": [{"id": code, "text": label} for code, label in pairs],
    }


def _shape(table_id: str, text: str, variables: list[dict[str, object]]) -> dict[str, object]:
    return {"id": table_id, "text": text, "variables": variables}


def test_a_required_subcategory_is_not_used_as_the_total() -> None:
    """EJ131, FF4, and MREG22 each have one required label that merely ends in 'i alt'."""
    tid = {
        "id": "Tid",
        "text": "tid",
        "elimination": False,
        "time": True,
        "values": [{"id": "2025", "text": "2025"}],
    }
    shapes = {
        "EJ131": _shape(
            "EJ131",
            "Ejendomssalg (endelig)",
            [
                _values(
                    "EJENDOMSKATE",
                    [("0111", "Enfamiliehuse"), ("2103", "Ejerlejligheder, i alt")],
                ),
                tid,
            ],
        ),
        "FF4": _shape(
            "FF4",
            "Ferierejser",
            [
                _values(
                    "OVERNATF",
                    [
                        ("110", "Hoteller og Feriecentre i alt"),
                        ("140", "Camping"),
                        ("150", "Vandrerhjem"),
                    ],
                ),
                {
                    "id": "DESTINA",
                    "text": "destination",
                    "elimination": False,
                    "time": False,
                    "values": [
                        {"id": "IALT", "text": "Danmark og udlandet i alt"},
                        {"id": "DAN", "text": "Danmark"},
                    ],
                },
                tid,
            ],
        ),
        "MREG22": _shape(
            "MREG22",
            "Offentlig miljøbeskyttelse",
            [
                _values(
                    "UI",
                    [
                        ("U1", "1.7. Driftsudgifter i alt (1+2+3+4+5+6)"),
                        ("I2", "2.5. Kapitalindtægter i alt"),
                        ("I", "2.6. Drifts- og kapitalindtægter i alt (4+5)"),
                    ],
                ),
                tid,
            ],
        ),
        "REQ": _shape(
            "REQ",
            "Required total",
            [
                _values("KON", [("X", "I alt"), ("M", "Mænd")]),
                tid,
            ],
        ),
    }
    sent: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        if request.url.path.endswith("/tableinfo"):
            table = str(body.get("table", "")).upper()
            return httpx.Response(200, json=shapes[table])
        sent.append(body)
        return httpx.Response(400, json={"message": "stop"})

    http = httpx.Client(
        base_url="https://api.statbank.dk/v1",
        transport=httpx.MockTransport(handle),
    )
    previous = set_client(StatbankClient(http, today=date(2026, 11, 2)))
    try:
        for table, label in (
            ("EJ131", "Ejerlejligheder, i alt"),
            ("FF4", "Hoteller og Feriecentre i alt"),
            ("MREG22", "2.5. Kapitalindtægter i alt"),
        ):
            sent.clear()
            with pytest.raises(ToolError, match="must be set") as caught:
                hent_data(table, {})
            message = str(caught.value)
            assert label in message
            assert sent == []

        sent.clear()
        with pytest.raises(ToolError, match="stop"):
            hent_data("FF4", {"OVERNATF": ["Camping"]})
        variables = {item["code"]: item["values"] for item in sent[-1]["variables"]}  # type: ignore[index]
        assert variables["OVERNATF"] == ["140"]
        assert variables["DESTINA"] == ["IALT"]

        sent.clear()
        with pytest.raises(ToolError, match="stop"):
            hent_data("REQ", {})
        variables = {item["code"]: item["values"] for item in sent[-1]["variables"]}  # type: ignore[index]
        assert variables["KON"] == ["X"]
    finally:
        set_client(previous)
        http.close()


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
