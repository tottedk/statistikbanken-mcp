"""The tools as an MCP client sees them, in-process, on recorded responses."""

import asyncio
import json

from mcp import Client
from mcp.types import TextContent

from statistikbanken_mcp.server import mcp


def test_client_sees_three_tools_and_a_real_call() -> None:
    asyncio.run(_session())


async def _session() -> None:
    async with Client(mcp) as client:
        listed = await client.list_tools()
        names = [tool.name for tool in listed.tools]
        assert names == ["find_tabel", "tabelinfo", "hent_data", "beregn"]
        hent = next(tool for tool in listed.tools if tool.name == "hent_data")
        assert hent.description is not None
        assert "sum_from" in hent.description
        assert "sum_range" in hent.description

        found = await client.call_tool("find_tabel", {"query": "befolkning", "limit": 1})
        assert not found.is_error
        assert _payload(found) == [
            {
                "id": "FOLK1A",
                "title": "Befolkningen den 1. i kvartalet",
                "unit": "Antal",
                "first_period": "2008K1",
                "latest_period": "2026K3",
                "variables": ["område", "køn", "alder", "civilstand", "tid"],
            }
        ]

        data = await client.call_tool(
            "hent_data",
            {
                "table": "FOLK1A",
                "values": {
                    "OMRÅDE": ["561"],
                    "KØN": ["TOT"],
                    "CIVILSTAND": ["TOT"],
                    "Tid": ["(1)"],
                },
                "sum_from": {"ALDER": "80"},
            },
        )
        assert not data.is_error
        payload = _payload(data)
        assert isinstance(payload, dict)
        assert payload["rows"][0]["value"] == 7556
        assert payload["rows"][0]["ALDER"] == "80+"

        children = await client.call_tool(
            "hent_data",
            {
                "table": "FOLK1A",
                "values": {
                    "OMRÅDE": ["561", "791"],
                    "KØN": ["TOT"],
                    "CIVILSTAND": ["TOT"],
                    "Tid": ["2026K3"],
                },
                "sum_range": {"ALDER": ["0", "17"]},
            },
        )
        assert not children.is_error
        child_payload = _payload(children)
        assert isinstance(child_payload, dict)
        assert [row["value"] for row in child_payload["rows"]] == [21616, 19171]
        assert child_payload["rows"][0]["ALDER"] == "0-17"
        assert child_payload["periods"] == ["2026K3"]

        missing = await client.call_tool("tabelinfo", {"table": "NOTATABLE"})
        assert missing.is_error is True
        text = _text(missing)
        assert "Unknown table 'NOTATABLE'" in text
        assert "find_tabel" in text
        assert "Traceback" not in text

        bad_value = await client.call_tool(
            "hent_data",
            {"table": "FOLK1A", "values": {"OMRÅDE": ["99999"]}},
        )
        assert bad_value.is_error is True
        text = _text(bad_value)
        assert "99999" in text
        assert "OMRÅDE" in text
        assert "Traceback" not in text


def _payload(result: object) -> object:
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict) and "result" in structured:
        return structured["result"]
    return json.loads(_text(result))


def _text(result: object) -> str:
    content = getattr(result, "content", None)
    assert content
    block = content[0]
    assert isinstance(block, TextContent)
    return block.text
