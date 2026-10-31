"""Web demo: tool loop, limits, and the real MCP server's tool list."""

from __future__ import annotations

import asyncio
import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import httpx
import pytest

from statistikbanken_mcp.web import (
    ChatClient,
    ChatReply,
    Limiter,
    LLMError,
    McpRunner,
    Settings,
    ToolCall,
    ToolSpec,
    TurnResult,
    _handler,
    list_server_tools,
    run_turn,
    server_command,
    settings_from_env,
)


class ScriptedModel:
    def __init__(self, replies: list[ChatReply]) -> None:
        self.replies = list(replies)
        self.tools_seen: list[list[dict] | None] = []

    def complete(self, messages: list[dict], tools: list[dict] | None) -> ChatReply:
        self.tools_seen.append(tools)
        return self.replies.pop(0)


class FakeTools:
    def __init__(self, results: dict[str, str]) -> None:
        self.results = results
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                "find_tabel",
                "search",
                {"type": "object", "properties": {"query": {"type": "string"}}},
            ),
            ToolSpec("hent_data", "fetch", {"type": "object", "properties": {}}),
        ]

    async def call(self, name: str, arguments: dict) -> str:
        self.calls.append((name, arguments))
        return self.results[name]


def _call(name: str, arguments: dict, call_id: str = "call_1") -> ToolCall:
    raw = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
    return ToolCall(id=call_id, name=name, arguments=arguments, raw=raw)


def test_turn_cites_table_period_and_records_calls() -> None:
    payload = {
        "table": "FOLK1A",
        "unit": "Antal",
        "periods": ["2026K3"],
        "rows": [
            {
                "OMRÅDE": "Rudersdal",
                "ALDER": "80+",
                "Tid": "2026K3",
                "value": 5045,
            }
        ],
    }
    tools = FakeTools(
        {
            "find_tabel": '[{"id": "FOLK1A"}]',
            "hent_data": json.dumps(payload),
        }
    )
    model = ScriptedModel(
        [
            ChatReply("", [_call("find_tabel", {"query": "befolkning"}, "c1")]),
            ChatReply("", [_call("hent_data", {"table": "FOLK1A"}, "c2")]),
            ChatReply("Der bor 5.045 personer på 80 år eller derover i Rudersdal i 2026K3.", []),
        ]
    )
    result = asyncio.run(
        run_turn(model, tools, "Hvor mange over 80 bor i Rudersdal?", max_tool_calls=8)
    )
    assert "5.045" in result.answer
    assert [call["name"] for call in result.calls] == ["find_tabel", "hent_data"]
    assert result.facts == [
        {
            "table": "FOLK1A",
            "period": "2026K3",
            "unit": "Antal",
            "value": 5045,
            "detail": "OMRÅDE: Rudersdal, ALDER: 80+",
            "unpublished": False,
        }
    ]


def test_turn_stops_at_the_tool_cap() -> None:
    tools = FakeTools({"find_tabel": "[]"})
    model = ScriptedModel(
        [
            ChatReply("", [_call("find_tabel", {"query": "a"}, "c1")]),
            ChatReply("Delvist svar.", []),
        ]
    )
    result = asyncio.run(run_turn(model, tools, "spørgsmål", max_tool_calls=1))
    assert len(result.calls) == 1
    assert result.answer == "Delvist svar."


def test_invalid_tool_arguments_are_reported() -> None:
    tools = FakeTools({})
    bad = ToolCall(
        id="c1",
        name="find_tabel",
        arguments=None,
        raw={"id": "c1", "function": {"name": "find_tabel", "arguments": "{"}},
    )
    model = ScriptedModel(
        [
            ChatReply("", [bad]),
            ChatReply("Jeg kunne ikke læse kaldet.", []),
        ]
    )
    result = asyncio.run(run_turn(model, tools, "spørgsmål", max_tool_calls=4))
    assert result.calls[0]["result"] == "Tool arguments were not valid JSON."
    assert tools.calls == []


def test_limiter_rate_and_daily_cap() -> None:
    limiter = Limiter(daily_cap=3, per_minute=2)
    assert limiter.allow("10.0.0.1") is None
    assert limiter.allow("10.0.0.1") is None
    assert limiter.allow("10.0.0.1") == "rate"
    assert limiter.allow("10.0.0.2") is None
    assert limiter.allow("10.0.0.3") == "daily"


def test_settings_hide_the_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STATBANK_LLM", "xai")
    monkeypatch.setenv("XAI_API_KEY", "super-secret")
    settings = settings_from_env([])
    assert settings.llm == "xai"
    assert settings.base_url == "https://api.x.ai/v1"
    assert "super-secret" not in repr(settings)


def test_model_error_does_not_include_the_key() -> None:
    secret = "super-secret-key"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {secret}"
        return httpx.Response(401, json={"error": request.headers["authorization"]})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = ChatClient("https://api.x.ai/v1", secret, "grok-4", 5, http=http)
    with pytest.raises(LLMError) as caught:
        client.complete([{"role": "user", "content": "hej"}], [])
    assert "super-secret" not in str(caught.value)


def test_http_rejects_long_questions_and_the_rate_limit() -> None:
    settings = Settings(
        llm="ollama",
        base_url="http://127.0.0.1:11434/v1",
        model="qwen2.5",
        api_key="ollama",
        host="127.0.0.1",
        port=0,
        daily_cap=10,
        per_minute=1,
        max_question_chars=20,
        max_tool_calls=2,
        timeout=5,
    )

    def answer(question: str) -> TurnResult:
        return TurnResult(answer=f"Svar: {question}", facts=[], calls=[])

    handler = _handler(settings, Limiter(settings.daily_cap, settings.per_minute), answer)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        long = _post(port, {"question": "x" * 21})
        assert long.status == 400
        first = _post(port, {"question": "hej"})
        assert first.status == 200
        assert "Svar: hej" in first.read().decode()
        second = _post(port, {"question": "igen"})
        assert second.status == 429
    finally:
        server.shutdown()


def _post(port: int, payload: dict) -> HTTPConnection:
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    body = json.dumps(payload).encode()
    connection.request("POST", "/api/ask", body=body, headers={"Content-Type": "application/json"})
    return connection.getresponse()


def test_app_lists_tools_from_the_mcp_server() -> None:
    assert list_server_tools() == ["find_tabel", "tabelinfo", "hent_data"]


def test_real_server_schemas_become_llm_tools() -> None:
    """List tools over stdio and build the chat-completions payload. No data call."""

    async def _run() -> list[dict]:
        from mcp import Client

        model = ScriptedModel([ChatReply("Intet tal.", [])])
        async with Client(server_command(), read_timeout_seconds=30) as client:
            runner = McpRunner(client)
            await runner.prepare()
            await run_turn(model, runner, "Hej", max_tool_calls=2)
        assert model.tools_seen
        tools = model.tools_seen[0]
        assert tools is not None
        return tools

    tools = asyncio.run(_run())
    names = [tool["function"]["name"] for tool in tools]
    assert names == ["find_tabel", "tabelinfo", "hent_data"]
    for tool in tools:
        assert tool["type"] == "function"
        parameters = tool["function"]["parameters"]
        assert parameters["type"] == "object"
        assert "properties" in parameters
