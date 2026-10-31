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
    assert areas["230"] == "Rudersdal"
    assert areas["223"] == "Hørsholm"
    assert areas["157"] == "Gentofte"


def test_question_over_80_is_one_number(live_client: StatbankClient) -> None:
    info = tabelinfo("FOLK1A")
    latest = next(item for item in info["variables"] if item["id"] == "Tid")["values"][-1]["id"]
    result = hent_data(
        "FOLK1A",
        values={
            "OMRÅDE": ["230"],
            "KØN": ["TOT"],
            "CIVILSTAND": ["TOT"],
            "Tid": ["(1)"],
        },
        sum_from={"ALDER": "80"},
    )
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["OMRÅDE"] == "Rudersdal"
    assert row["ALDER"] == "80+"
    assert row["Tid"] == latest
    assert isinstance(row["value"], int)
    assert 1000 < row["value"] < 20000


def test_question_ten_year_series(live_client: StatbankClient) -> None:
    info = tabelinfo("FOLK1A")
    latest = next(item for item in info["variables"] if item["id"] == "Tid")["values"][-1]["id"]
    result = hent_data(
        "FOLK1A",
        values={
            "OMRÅDE": ["230"],
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
    assert all(row["OMRÅDE"] == "Rudersdal" for row in rows)
    assert all(row["ALDER"] == "Alder i alt" for row in rows)
    assert all(isinstance(row["value"], int) and row["value"] > 10000 for row in rows)
    years = {row["Tid"][:4] for row in rows if row["Tid"].endswith("K3")}
    assert int(rows[-1]["Tid"][:4]) - int(rows[0]["Tid"][:4]) >= 10
    assert "2016" in years


def test_question_three_municipalities(live_client: StatbankClient) -> None:
    result = hent_data(
        "FOLK1A",
        values={
            "OMRÅDE": ["230", "223", "157"],
            "KØN": ["TOT"],
            "CIVILSTAND": ["TOT"],
            "Tid": ["(1)"],
        },
        sum_from={"ALDER": "80"},
    )
    by_name = {row["OMRÅDE"]: row["value"] for row in result["rows"]}
    assert set(by_name) == {"Rudersdal", "Hørsholm", "Gentofte"}
    assert all(row["ALDER"] == "80+" for row in result["rows"])
    assert all(isinstance(value, int) and 500 < value < 20000 for value in by_name.values())
    winner = max(by_name, key=by_name.get)
    assert sum(value == by_name[winner] for value in by_name.values()) == 1


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
                "values": {"OMRÅDE": ["230"], "ALDER": ["IALT"], "Tid": ["(1)"]},
            },
        )
        assert data.is_error is False
        text = _text(data)
        assert "Rudersdal" in text
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
