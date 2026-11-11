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


def test_new_cars_and_gentofte_dwellings_match_the_known_totals(
    live_client: StatbankClient,
) -> None:
    cars = hent_data("BIL55", {"REGBIL": ["4000101002"], "Tid": ["2025"]})
    assert cars["rows"][0]["Tid"] == "2025"
    assert cars["rows"][0]["value"] == 185394
    assert cars["rows"][0]["year_total"]["method"] == "sum"
    homes = hent_data("BOL101", {"OMRÅDE": ["157"]})
    assert len(homes["rows"]) == 1
    assert homes["rows"][0]["value"] == 36145
    note = " ".join(homes["notes"]).casefold()
    assert "holiday" in note or "fritid" in note


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


def test_denmark_population_on_1_january_reaches_1934_and_1989(
    live_client: StatbankClient,
) -> None:
    """BEFOLK2 counts people. FOLK1A and FT do not publish 1934."""
    historical = hent_data(
        "BEFOLK2",
        {"KØN": ["TOT"], "ALDER": ["TOT"], "Tid": ["1934", "1989"]},
    )
    by_year = {row["Tid"]: row["value"] for row in historical["rows"]}
    assert by_year["1934"] == 3_650_500
    assert by_year["1989"] == 5_129_778
    stock_2026 = hent_data(
        "BEFOLK2",
        {"KØN": ["TOT"], "ALDER": ["TOT"], "Tid": ["2026"]},
    )
    assert stock_2026["rows"][0]["value"] == 6_025_603
    assert hent_data("BEFOLK1", {"Tid": ["1989"]})["rows"][0]["value"] == 5_129_778
    assert hent_data("FT", {"HOVEDDELE": ["000"], "Tid": ["1989"]})["rows"][0]["value"] == 5_129_778
    latest = hent_data("BEFOLK2", {"KØN": ["TOT"], "ALDER": ["TOT"]})
    assert len(latest["rows"]) == 1
    end = latest["rows"][0]
    assert isinstance(end["value"], int)
    assert end["value"] > by_year["1989"] > by_year["1934"]
    january = hent_data(
        "FOLK1A",
        {
            "OMRÅDE": ["000"],
            "KØN": ["TOT"],
            "ALDER": ["IALT"],
            "CIVILSTAND": ["TOT"],
            "Tid": [f"{end['Tid']}K1"],
        },
    )
    assert january["rows"][0]["value"] == end["value"]
    for year in ("1934", "1989"):
        with pytest.raises(ToolError, match="not in table FOLK1A"):
            hent_data("FOLK1A", {"OMRÅDE": ["000"], "Tid": [year]})
    with pytest.raises(ToolError, match="not in table FT") as caught:
        hent_data("FT", {"HOVEDDELE": ["000"], "Tid": ["1934"]})
    assert "1930" in str(caught.value) and "1935" in str(caught.value)
    for start in (by_year["1934"], by_year["1989"]):
        difference = end["value"] - start
        assert difference > 0
        percent = difference / start * 100
        assert 0 < percent < 100


def test_population_change_1995_to_2001_and_the_tenth_urban_area(
    live_client: StatbankClient,
) -> None:
    """Known stocks, the municipality, and the BY3 urban-area rank. 7 Nov 2026."""
    from statistikbanken_mcp.web import rank_urban_areas

    for table, values in (
        (
            "BEFOLK1",
            {"KØN": ["TOT"], "ALDER": ["TOT"], "CIVILSTAND": ["TOT"], "Tid": ["1995", "2001"]},
        ),
        ("BEFOLK2", {"KØN": ["TOT"], "ALDER": ["TOT"], "Tid": ["1995", "2001"]}),
    ):
        by_year = {row["Tid"]: row["value"] for row in hent_data(table, values)["rows"]}
        assert by_year["1995"] == 5_215_718
        assert by_year["2001"] == 5_349_212
    esbjerg = hent_data(
        "FOLK1A",
        {
            "OMRÅDE": ["561"],
            "KØN": ["TOT"],
            "ALDER": ["IALT"],
            "CIVILSTAND": ["TOT"],
            "Tid": ["2026K3"],
        },
    )
    assert esbjerg["rows"][0]["value"] == 114_824
    by = hent_data("BY3", {"FOLKARTAET": ["FOLKETAL"], "BYER": ["*"], "Tid": ["2026"]})
    ranked = rank_urban_areas(
        [
            (str(row.get("BYER") or ""), float(row["value"]))
            for row in by["rows"]
            if isinstance(row.get("value"), int | float) and not isinstance(row.get("value"), bool)
        ]
    )
    assert [(name, int(value)) for name, value in ranked[:10]] == [
        ("Hovedstadsområdet", 1_408_575),
        ("Aarhus", 305_227),
        ("Odense", 187_558),
        ("Aalborg", 122_550),
        ("Esbjerg", 71_371),
        ("Horsens", 65_037),
        ("Randers", 64_764),
        ("Kolding", 63_664),
        ("Vejle", 62_153),
        ("Roskilde", 53_897),
    ]
    assert int(next(value for name, value in ranked if name == "Esbjerg")) != 114_824


def test_stdio_client_calls_each_tool() -> None:
    asyncio.run(_stdio_session())


async def _stdio_session() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "statistikbanken_mcp"],
    )
    async with Client(params) as client:
        listed = await client.list_tools()
        assert [tool.name for tool in listed.tools] == [
            "find_tabel",
            "tabelinfo",
            "hent_data",
            "beregn",
        ]

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

        computed = await client.call_tool(
            "beregn",
            {"operation": "difference", "numerator": 114983, "denominator": 114766},
        )
        assert computed.is_error is False
        assert "217" in _text(computed)


def test_relative_time_inflation_and_a_published_change(live_client: StatbankClient) -> None:
    """nu is the table's latest period. One year earlier and CPI are fixed figures."""
    info = tabelinfo("FOLK1A")
    latest = next(item for item in info["variables"] if item["id"] == "Tid")["time_span"]["last"][
        "id"
    ]
    direct = hent_data(
        "FOLK1A",
        {
            "OMRÅDE": ["561"],
            "ALDER": ["IALT"],
            "KØN": ["TOT"],
            "CIVILSTAND": ["TOT"],
            "Tid": [latest],
        },
    )
    now = hent_data(
        "FOLK1A",
        {
            "OMRÅDE": ["561"],
            "ALDER": ["IALT"],
            "KØN": ["TOT"],
            "CIVILSTAND": ["TOT"],
            "Tid": ["nu"],
        },
    )
    assert now["rows"][0]["Tid"] == latest
    assert now["rows"][0]["value"] == direct["rows"][0]["value"]
    earlier = hent_data(
        "FOLK1A",
        {
            "OMRÅDE": ["561"],
            "ALDER": ["IALT"],
            "KØN": ["TOT"],
            "CIVILSTAND": ["TOT"],
            "Tid": ["for 1 år siden"],
        },
    )
    assert earlier["rows"][0]["Tid"] == f"{int(latest[:4]) - 1}{latest[4:]}"
    if earlier["rows"][0]["Tid"] == "2025K4":
        assert earlier["rows"][0]["value"] == 114_983
        assert direct["rows"][0]["value"] == 114_766
    with pytest.raises(ToolError, match="2099") as caught:
        hent_data("FOLK1A", {"OMRÅDE": ["561"], "Tid": ["2099"]})
    message = str(caught.value)
    assert latest in message
    assert "114766" not in message
    inflation = hent_data(
        "PRIS111",
        {"VAREGR": ["000000"], "ENHED": ["300"], "Tid": ["2025M12"]},
    )
    assert inflation["rows"][0]["value"] == 1.9
    areas = hent_data("BY3", {"FOLKARTAET": ["FOLKETAL"], "BYER": ["*"], "Tid": ["2026"]})
    from statistikbanken_mcp.web import rank_urban_areas

    ranked = rank_urban_areas(
        [
            (str(row.get("BYER") or ""), float(row["value"]))
            for row in areas["rows"]
            if isinstance(row.get("value"), int | float) and not isinstance(row.get("value"), bool)
        ]
    )
    assert ranked[1] == ("Aarhus", 305227) or (
        ranked[1][0] == "Aarhus" and int(ranked[1][1]) == 305_227
    )


def test_subject_tables_match_the_published_figures(live_client: StatbankClient) -> None:
    """Known figures checked against api.statbank.dk on 10 Nov 2026."""
    from statistikbanken_mcp.web import rank_urban_areas

    income = hent_data(
        "INDKP105",
        {
            "OMRÅDE": ["000"],
            "ENHED": ["118"],
            "KOEN": ["MOK"],
            "ALDER1": ["00"],
            "INDKINTB": ["000"],
            "Tid": ["2025"],
        },
    )
    assert income["rows"][0]["value"] == 428_795
    prices = hent_data(
        "PRIS111",
        {"VAREGR": ["000000"], "ENHED": ["100"], "Tid": ["2025M12"]},
    )
    assert prices["rows"][0]["value"] == 121.2
    education = hent_data(
        "HFUDD11",
        {
            "BOPOMR": ["000"],
            "HERKOMST": ["TOT"],
            "HFUDD": ["TOT"],
            "ALDER": ["TOT"],
            "KØN": ["TOT"],
            "Tid": ["2025"],
        },
    )
    assert education["rows"][0]["value"] == 4_157_750
    births = hent_data(
        "BEV22",
        {"OMRÅDE": ["751"], "BEVÆGELSE": ["B02"], "KØN": ["*"], "Tid": ["2026K3"]},
    )
    assert sum(row["value"] for row in births["rows"]) == 1_250
    areas = hent_data("BY3", {"FOLKARTAET": ["FOLKETAL"], "BYER": ["*"], "Tid": ["2026"]})
    ranked = rank_urban_areas(
        [
            (str(row.get("BYER") or ""), float(row["value"]))
            for row in areas["rows"]
            if isinstance(row.get("value"), int | float) and not isinstance(row.get("value"), bool)
        ]
    )
    assert ranked[1][0] == "Aarhus"
    assert int(ranked[1][1]) == 305_227


def test_a_verified_card_matches_the_esbjerg_stock(live_client: StatbankClient) -> None:
    """The card accepts the published 2026K4 stock and rejects a number that was not fetched."""
    from statistikbanken_mcp.answer_card import build_card, verify_card

    data = hent_data(
        "FOLK1A",
        {"OMRÅDE": ["561"], "ALDER": ["IALT"], "KØN": ["TOT"], "Tid": ["2026K4"]},
    )
    assert data["rows"][0]["value"] == 114_766
    fact = {
        "table": "FOLK1A",
        "period": data["rows"][0]["Tid"],
        "unit": data["unit"],
        "value": data["rows"][0]["value"],
        "detail": "OMRÅDE: Esbjerg",
    }
    card = build_card("Hvor mange indbyggere har Esbjerg Kommune?", [fact])
    assert verify_card(card, [fact]) == []
    rejected = dict(card)
    rejected["values"] = [1]
    assert verify_card(rejected, [fact])


def test_quarters_stay_quarters_and_the_price_table_ranks(live_client: StatbankClient) -> None:
    """A bare year in a question is not averaged. PRIS111 is the price index."""
    kept = hent_data(
        "FOLK1A",
        {"OMRÅDE": ["791"], "ALDER": ["IALT"], "KØN": ["TOT"], "Tid": ["2008", "2026"]},
        question="Hvor meget er befolkningen i Viborg Kommune vokset siden 2000?",
    )
    by_period = {row["Tid"]: row["value"] for row in kept["rows"]}
    assert "2008" not in by_period
    assert by_period["2008K1"] == 92_084
    assert isinstance(by_period["2008K1"], int)
    assert by_period["2026K4"] == 98_282
    assert isinstance(by_period["2026K4"], int)
    assert find_tabel("forbrugerprisindekset i alt", limit=3)[0]["id"] == "PRIS111"
    place = hent_data(
        "FOLK1A",
        {"OMRÅDE": ["Lyngby"], "ALDER": ["IALT"], "KØN": ["TOT"], "Tid": ["2026K4"]},
    )
    assert place["rows"][0]["OMRÅDE"] == "Lyngby-Taarbæk"


def test_compute_rank_group_and_required_code(live_client: StatbankClient) -> None:
    """Known Esbjerg stocks, the municipality rank, a named group, and a required code."""
    from statistikbanken_mcp.server import beregn
    from statistikbanken_mcp.web import _area_rank_answer

    earlier = hent_data(
        "FOLK1A",
        {"OMRÅDE": ["561"], "ALDER": ["IALT"], "KØN": ["TOT"], "Tid": ["2025K4"]},
    )
    latest = hent_data(
        "FOLK1A",
        {"OMRÅDE": ["561"], "ALDER": ["IALT"], "KØN": ["TOT"], "Tid": ["2026K4"]},
    )
    assert earlier["rows"][0]["value"] == 114_983
    assert latest["rows"][0]["value"] == 114_766
    diff = beregn("difference", 114_983, 114_766)
    assert diff["numerator"] == 114_983
    assert diff["denominator"] == 114_766
    assert diff["result"] == 217
    assert diff["unit"]
    share = beregn("percent", 114_766, 114_983)
    assert abs(share["result"] - (114_766 / 114_983 * 100)) < 1e-9

    both = hent_data(
        "FOLK1A",
        {"OMRÅDE": ["000", "561"], "ALDER": ["IALT"], "KØN": ["TOT"], "Tid": ["2026K4"]},
    )
    facts = []
    for row in both["rows"]:
        facts.append(
            {
                "table": "FOLK1A",
                "period": row["Tid"],
                "unit": "Antal",
                "value": row["value"],
                "detail": f"OMRÅDE: {row['OMRÅDE']}, ALDER: Alder i alt",
            }
        )
    ranked = _area_rank_answer("Which municipality has the largest population?", facts)
    assert ranked is not None
    assert "Esbjerg" in ranked
    assert "114.766" in ranked
    assert "Hele landet" not in ranked

    women = hent_data(
        "FOLK1A",
        {"OMRÅDE": ["561"], "ALDER": ["IALT"], "Tid": ["2026K4"]},
        question="Hvor mange kvinder bor i Esbjerg?",
    )
    explicit = hent_data(
        "FOLK1A",
        {"OMRÅDE": ["561"], "ALDER": ["IALT"], "KØN": ["2"], "Tid": ["2026K4"]},
    )
    assert women["rows"][0]["KØN"] == "Kvinder"
    assert women["rows"][0]["value"] == explicit["rows"][0]["value"]
    assert women["rows"][0]["value"] != 114_766

    with pytest.raises(ToolError, match="must be set") as caught:
        hent_data("BY3", {"FOLKARTAET": ["FOLKETAL"], "Tid": ["2026"]})
    message = str(caught.value)
    assert "Codes:" in message
    assert "Pass one of them" in message


def _text(result: object) -> str:
    content = getattr(result, "content", None)
    assert content
    block = content[0]
    assert isinstance(block, TextContent)
    return block.text
