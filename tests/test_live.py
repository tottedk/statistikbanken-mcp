"""Live calls against https://api.statbank.dk. Run with pytest -m live."""

import asyncio
import sys

import pytest
from mcp import Client, StdioServerParameters
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import TextContent

from statistikbanken_mcp.server import find_tabel, hent_data, set_client, tabelinfo
from statistikbanken_mcp.statbank import StatbankClient

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def live_client():
    client = StatbankClient()
    previous = set_client(client)
    yield client
    set_client(previous)
    client.close()


def test_search_and_metadata(live_client: StatbankClient) -> None:
    hits = find_tabel("befolkning", limit=5)
    assert hits[0]["id"] == "FOLK1A"
    assert hits[0]["unit"] == "Antal"
    assert hits[0]["latest_period"]
    info = tabelinfo("FOLK1A")
    variables = {item["id"]: item for item in info["variables"]}
    assert variables["ALDER"]["numeric_codes"]["from"] == "0"
    assert int(variables["ALDER"]["numeric_codes"]["to"]) >= 100
    areas = {item["id"]: item["text"] for item in variables["OMRÅDE"]["values"]}
    assert areas["157"] == "Gentofte"
    filtered = tabelinfo("FOLK1A", values_filter="Esbjerg, Viborg")
    named = {
        item["id"]: item["text"]
        for item in next(item for item in filtered["variables"] if item["id"] == "OMRÅDE")["values"]
    }
    assert named == {"561": "Esbjerg", "791": "Viborg"}


def test_ambiguous_time_sum_is_rejected_live(live_client: StatbankClient) -> None:
    with pytest.raises(ToolError, match="not every period") as caught:
        hent_data(
            "FOLK1A",
            values={"ALDER": [">=80"], "OMRÅDE": ["Esbjerg", "Viborg"]},
            sum_range={"Tid": ["2015K1", "2026K3"]},
        )
    message = str(caught.value)
    assert ">=2015K1" in message
    assert "22780" not in message
    assert "Traceback" not in message


def test_equivalent_age_sum_is_one_number_live(live_client: StatbankClient) -> None:
    result = hent_data(
        "FOLK1A",
        values={"OMRÅDE": ["Esbjerg"], "ALDER": [">=80"], "Tid": ["(1)"]},
        sum_from={"ALDER": "80"},
    )
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["OMRÅDE"] == "Esbjerg"
    assert row["ALDER"] == "80+"
    assert isinstance(row["value"], int)
    assert 1000 < row["value"] < 20000
    assert any("sum_from" in note for note in result["notes"])


def test_open_ended_age_returns_the_80_plus_series(live_client: StatbankClient) -> None:
    for phrase in ("80+", ">=80"):
        result = hent_data(
            "FOLK1A",
            values={
                "OMRÅDE": ["Esbjerg", "Viborg"],
                "ALDER": [phrase],
                "Tid": [">=2015K1"],
            },
        )
        by_place = {}
        for row in result["rows"]:
            by_place.setdefault(row["OMRÅDE"], []).append(row)
        assert set(by_place) == {"Esbjerg", "Viborg"}, phrase
        esbjerg = by_place["Esbjerg"]
        viborg = by_place["Viborg"]
        assert esbjerg[0]["Tid"] == "2015K1", phrase
        assert esbjerg[0]["ALDER"] == "80+", phrase
        assert esbjerg[0]["value"] > 2000, phrase
        assert esbjerg[-1]["value"] > 2000, phrase
        assert viborg[0]["value"] > 1000, phrase
        assert any("sum_from" in note for note in result["notes"]), phrase


def test_question_over_80_is_one_number(live_client: StatbankClient) -> None:
    info = tabelinfo("FOLK1A")
    tid = next(item for item in info["variables"] if item["id"] == "Tid")
    latest = tid["time_span"]["last"]["id"]
    result = hent_data(
        "FOLK1A",
        values={
            "OMRÅDE": ["561"],
            "KØN": ["TOT"],
            "CIVILSTAND": ["TOT"],
            "Tid": ["(1)"],
        },
        sum_from={"ALDER": "80"},
    )
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["OMRÅDE"] == "Esbjerg"
    assert row["ALDER"] == "80+"
    assert row["Tid"] == latest
    assert isinstance(row["value"], int)
    assert 1000 < row["value"] < 20000


def test_question_ten_year_series(live_client: StatbankClient) -> None:
    info = tabelinfo("FOLK1A")
    tid = next(item for item in info["variables"] if item["id"] == "Tid")
    latest = tid["time_span"]["last"]["id"]
    result = hent_data(
        "FOLK1A",
        values={
            "OMRÅDE": ["791"],
            "KØN": ["TOT"],
            "ALDER": ["IALT"],
            "CIVILSTAND": ["TOT"],
            "Tid": [">=2016K3"],
        },
    )
    rows = result["rows"]
    assert rows[0]["Tid"] == "2016K3"
    assert rows[-1]["Tid"] == latest
    assert len(rows) >= 41
    assert all(row["OMRÅDE"] == "Viborg" for row in rows)
    assert all(row["ALDER"] == "Alder i alt" for row in rows)
    assert all(isinstance(row["value"], int) and row["value"] > 10000 for row in rows)
    years = {row["Tid"][:4] for row in rows if row["Tid"].endswith("K3")}
    assert int(rows[-1]["Tid"][:4]) - int(rows[0]["Tid"][:4]) >= 10
    assert "2016" in years


def test_question_three_municipalities(live_client: StatbankClient) -> None:
    result = hent_data(
        "FOLK1A",
        values={
            "OMRÅDE": ["561", "791", "157"],
            "KØN": ["TOT"],
            "CIVILSTAND": ["TOT"],
            "Tid": ["(1)"],
        },
        sum_from={"ALDER": "80"},
    )
    by_name = {row["OMRÅDE"]: row["value"] for row in result["rows"]}
    assert set(by_name) == {"Esbjerg", "Viborg", "Gentofte"}
    assert all(row["ALDER"] == "80+" for row in result["rows"])
    assert all(isinstance(value, int) and 500 < value < 20000 for value in by_name.values())
    winner = max(by_name, key=by_name.get)
    assert sum(value == by_name[winner] for value in by_name.values()) == 1


def test_question_children_0_to_17(live_client: StatbankClient) -> None:
    result = hent_data(
        "FOLK1A",
        values={
            "OMRÅDE": ["561", "791"],
            "KØN": ["TOT"],
            "CIVILSTAND": ["TOT"],
            "Tid": ["(1)"],
        },
        sum_range={"ALDER": ["0", "17"]},
    )
    assert result["periods"]
    assert len(result["rows"]) == 2
    by_name = {row["OMRÅDE"]: row["value"] for row in result["rows"]}
    assert set(by_name) == {"Esbjerg", "Viborg"}
    assert all(row["ALDER"] == "0-17" for row in result["rows"])
    assert all(isinstance(value, int) and 1000 < value < 50000 for value in by_name.values())
    assert by_name["Esbjerg"] > by_name["Viborg"]


def test_question_average_income(live_client: StatbankClient) -> None:
    hits = find_tabel("indkomst", limit=15)
    assert any(hit["id"] == "INDKP105" and hit["latest_period"] for hit in hits)
    result = hent_data(
        "INDKP105",
        values={
            "OMRÅDE": ["561", "791"],
            "ENHED": ["118"],
            "KOEN": ["MOK"],
            "ALDER1": ["00"],
            "INDKINTB": ["000"],
            "Tid": ["2025"],
        },
    )
    assert result["title"]
    assert result["unit"] == "-"
    assert result["periods"] == ["2025"]
    by_name = {row["OMRÅDE"]: row for row in result["rows"]}
    assert set(by_name) == {"Esbjerg", "Viborg"}
    for row in by_name.values():
        assert row["ENHED"] == "Gennemsnit for personer i gruppen (kr.)"
        assert isinstance(row["value"], int)
        assert 100_000 < row["value"] < 5_000_000


def test_question_association_membership_is_sports_clubs(live_client: StatbankClient) -> None:
    hits = find_tabel("forening")
    sports = next(hit for hit in hits if hit["id"] == "IDRFOR01")
    assert "idræt" in sports["title"].casefold()
    assert sports["unit"] == "Antal"
    assert sports["latest_period"]
    info = tabelinfo("IDRFOR01")
    assert "idræt" in info["description"].casefold()
    result = hent_data(
        "IDRFOR01",
        values={
            "BLSTKOM": ["561"],
            "ORGANISATION": ["100"],
            "AKTP": ["104", "105"],
            "Tid": ["(1)"],
        },
    )
    assert "idræt" in result["title"].casefold()
    assert "idræt" in result["description"].casefold()
    assert result["unit"] == "Antal"
    assert result["periods"]
    by_measure = {row["AKTP"]: row["value"] for row in result["rows"]}
    assert set(by_measure) == {"Foreninger", "Medlemmer"}
    assert isinstance(by_measure["Foreninger"], int) and by_measure["Foreninger"] < 500
    assert isinstance(by_measure["Medlemmer"], int) and by_measure["Medlemmer"] > 1000
    assert all(row["BLSTKOM"] == "Esbjerg" for row in result["rows"])


def test_live_invalid_input_is_actionable(live_client: StatbankClient) -> None:
    with pytest.raises(ToolError, match="Unknown table 'NOTATABLE'") as caught:
        tabelinfo("NOTATABLE")
    assert "Traceback" not in str(caught.value)
    assert "find_tabel" in str(caught.value)

    with pytest.raises(ToolError, match="99999") as caught:
        hent_data("FOLK1A", {"OMRÅDE": ["99999"]})
    message = str(caught.value)
    assert "OMRÅDE" in message
    assert "Traceback" not in message
    assert "errorTypeCode" not in message


def test_unemployment_and_hyphen_search_live(live_client: StatbankClient) -> None:
    for query in ("arbejdsløshed", "ledighed"):
        ids = [hit["id"] for hit in find_tabel(query, limit=5)]
        assert ids[0].startswith("AUS"), (query, ids)
    hyphen = [hit["id"] for hit in find_tabel("fuldtids-ledige", limit=5)]
    assert hyphen[0].startswith("AUS"), hyphen
    with pytest.raises(ToolError, match="No table matches"):
        find_tabel("zzzz-qqqq")


def test_year_total_matches_the_quarters_live(live_client: StatbankClient) -> None:
    quarters = hent_data(
        "FOLK1A",
        {"OMRÅDE": ["Esbjerg"], "ALDER": ["IALT"], "Tid": [">=2025K1<=2025K4"]},
    )
    values = [row["value"] for row in quarters["rows"]]
    assert len(values) == 4
    year = hent_data(
        "FOLK1A",
        {"OMRÅDE": ["Esbjerg"], "ALDER": ["IALT"], "Tid": ["2025"]},
    )
    assert year["rows"][0]["Tid"] == "2025"
    assert year["rows"][0]["value"] == sum(values) / len(values)
    assert "average" in " ".join(year["notes"])
    assert "level" in " ".join(year["notes"])

    parts = hent_data("STRAF10", {"OVERTRÆD": ["1"], "Tid": [">=2025K1<=2025K4"]})
    crimes = hent_data("STRAF10", {"OVERTRÆD": ["1"], "Tid": ["2025"]})
    assert crimes["rows"][0]["value"] == sum(row["value"] for row in parts["rows"])
    assert "sum" in " ".join(crimes["notes"])
    assert "flow" in " ".join(crimes["notes"])

    births = hent_data(
        "BEV22",
        {"OMRÅDE": ["561"], "BEVÆGELSE": ["B02"], "Tid": ["2024"]},
    )
    birth_parts = hent_data(
        "BEV22",
        {"OMRÅDE": ["561"], "BEVÆGELSE": ["B02"], "Tid": [">=2024K1<=2024K4"]},
    )
    assert births["rows"][0]["value"] == sum(row["value"] for row in birth_parts["rows"])
    corrections = hent_data(
        "BEV22",
        {"OMRÅDE": ["561"], "BEVÆGELSE": ["B12"], "Tid": ["2024"]},
    )
    assert len(corrections["rows"]) == 4
    assert all(row["Tid"] != "2024" for row in corrections["rows"])
    assert "no yearly figure" in " ".join(corrections["notes"]).casefold()


def test_calendar_phrases_follow_today_live(live_client: StatbankClient) -> None:
    from datetime import date, timedelta

    today = date.today()
    previous_year = str(today.year - 1)
    this_year = str(today.year)
    direct = hent_data(
        "IDRFOR01",
        {
            "BLSTKOM": ["561"],
            "ORGANISATION": ["100"],
            "AKTP": ["104", "105"],
            "Tid": [previous_year],
        },
    )
    phrase = hent_data(
        "IDRFOR01",
        {
            "BLSTKOM": ["561"],
            "ORGANISATION": ["100"],
            "AKTP": ["104", "105"],
            "Tid": ["i fjor"],
        },
    )
    assert {row["AKTP"]: row["value"] for row in phrase["rows"]} == {
        row["AKTP"]: row["value"] for row in direct["rows"]
    }
    assert all(row["Tid"] == previous_year for row in phrase["rows"])
    info = tabelinfo("IDRFOR01")
    latest = next(item for item in info["variables"] if item["id"] == "Tid")["time_span"]["last"][
        "id"
    ]
    if this_year > latest:
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
        assert this_year in message
        assert latest in message
    month = today.replace(day=1) - timedelta(days=1)
    code = f"{month.year}M{month.month:02d}"
    aus = tabelinfo("AUS07")
    aus_latest = next(item for item in aus["variables"] if item["id"] == "Tid")["time_span"][
        "last"
    ]["id"]
    if code > aus_latest:
        with pytest.raises(ToolError, match="not published") as caught:
            hent_data("AUS07", {"Tid": ["sidste måned"]})
        message = str(caught.value)
        assert code in message
        assert aus_latest in message
        assert "2026M08" not in message or code == "2026M08"


def test_danish_periods_and_projection_label_live(live_client: StatbankClient) -> None:
    info = tabelinfo("FOLK1A")
    latest = next(item for item in info["variables"] if item["id"] == "Tid")["time_span"]["last"][
        "id"
    ]
    phrase = hent_data(
        "FOLK1A",
        {"OMRÅDE": ["Esbjerg"], "ALDER": ["IALT"], "Tid": ["seneste"]},
    )
    assert phrase["rows"][0]["Tid"] == latest
    with pytest.raises(ToolError, match="Available") as caught:
        hent_data("FOLK1A", {"OMRÅDE": ["Esbjerg"], "Tid": ["marts 1990"]})
    assert "FOLK1A" in str(caught.value)
    with pytest.raises(ToolError, match="not a quarter"):
        hent_data("FOLK1A", {"Tid": ["8. kvartal 2025"]})
    projected = hent_data(
        "FRKM126",
        {"KOMMUNEDK": ["561"], "ALDER": ["TOT"], "KØN": ["M"], "Tid": ["2030"]},
    )
    assert projected["projection"] is True
    row = projected["rows"][0]
    assert row["projection"] is True
    assert row.get("KOMMUNEDK") == "Esbjerg" or row.get("OMRÅDE") == "Esbjerg"
    assert isinstance(row["value"], int)


def test_stdio_client_calls_each_tool() -> None:
    asyncio.run(_stdio_session())


async def _stdio_session() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "statistikbanken_mcp"],
    )
    async with Client(params) as client:
        listed = await client.list_tools()
        assert [tool.name for tool in listed.tools] == ["find_tabel", "tabelinfo", "hent_data"]

        found = await client.call_tool("find_tabel", {"query": "FOLK1A", "limit": 1})
        assert found.is_error is False
        assert "FOLK1A" in _text(found)

        missing = await client.call_tool("tabelinfo", {"table": "NOTATABLE"})
        assert missing.is_error is True
        assert "Unknown table" in _text(missing)
        assert "Traceback" not in _text(missing)

        data = await client.call_tool(
            "hent_data",
            {
                "table": "FOLK1A",
                "values": {"OMRÅDE": ["561"], "ALDER": ["IALT"], "Tid": ["(1)"]},
            },
        )
        assert data.is_error is False
        text = _text(data)
        assert "Esbjerg" in text
        assert "Alder i alt" in text

        bad = await client.call_tool(
            "hent_data",
            {"table": "FOLK1A", "values": {"ALDER": ["99999"]}},
        )
        assert bad.is_error is True
        text = _text(bad)
        assert "99999" in text
        assert "Traceback" not in text


def _text(result: object) -> str:
    content = getattr(result, "content", None)
    assert content
    block = content[0]
    assert isinstance(block, TextContent)
    return block.text
