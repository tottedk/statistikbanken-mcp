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
    NO_FETCHED_NUMBERS,
    PAGE,
    SYSTEM_PROMPT,
    UNVERIFIED_NUMBERS,
    ChatClient,
    ChatReply,
    Limiter,
    LLMError,
    McpRunner,
    Settings,
    ToolCall,
    ToolSpec,
    TurnResult,
    _closing_instruction,
    _facts_from,
    _fallback_answer,
    _handler,
    _turn,
    chart_series,
    facts_to_csv,
    guard_answer,
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
            "url": "https://www.statistikbanken.dk/FOLK1A",
        }
    ]


def test_empty_model_text_is_not_the_tool_cap() -> None:
    tools = FakeTools({"find_tabel": "[]", "tabelinfo": "{}"})
    model = ScriptedModel(
        [
            ChatReply("", [_call("find_tabel", {"query": "a"}, "c1")]),
            ChatReply("", [_call("tabelinfo", {"table": "FOLK1A"}, "c2")]),
            ChatReply("   ", []),
        ]
    )
    result = asyncio.run(run_turn(model, tools, "spørgsmål", max_tool_calls=8))
    assert len(result.calls) == 2
    assert "grænsen" not in result.answer
    assert result.answer == _fallback_answer(result.calls, capped=False)


def test_empty_final_answer_after_the_cap_names_the_limit() -> None:
    tools = FakeTools({"find_tabel": "[]"})
    model = ScriptedModel(
        [
            ChatReply("", [_call("find_tabel", {"query": "a"}, "c1")]),
            ChatReply("", []),
        ]
    )
    result = asyncio.run(run_turn(model, tools, "spørgsmål", max_tool_calls=1))
    assert "grænsen for opslag" in result.answer


def test_prompt_requires_the_question_language_and_danish_periods() -> None:
    assert "same language as the question" in SYSTEM_PROMPT
    assert "Do not answer in English when the question is Danish" in SYSTEM_PROMPT
    assert "only the latest period" in SYSTEM_PROMPT
    assert "5.045" in SYSTEM_PROMPT
    assert "2026K3" in SYSTEM_PROMPT
    assert "2026Q3" in SYSTEM_PROMPT
    assert "Do not put Tid in sum_range" in SYSTEM_PROMPT
    single = _closing_instruction("Hvor mange over 80 bor i Rudersdal?")
    assert "Danish" in single
    assert "latest period" in single
    assert "do not invent" in single.casefold()
    trend = _closing_instruction(
        "Hvordan har antallet af 80+ i Rudersdal og Hørsholm udviklet sig siden 2015?"
    )
    assert "change over time" in trend
    assert "Danish" in trend


# Excerpts from the qwen2.5 retest (pr7r-ui-log), where the model wrote numbers
# after hent_data failed and the page showed them with an empty fact list.
_HALLUCINATED = {
    "a1": (
        "Data for Rudersdal og Hørsholm er summen af alle personer 80 år og over.\n"
        '{"Tid": "2026K3", "OMRÅDE": "230", "ALDER": "80+", "value": 1234},\n'
        '{"Tid": "2026K3", "OMRÅDE": "223", "ALDER": "80+", "value": 567}'
    ),
    "a2": (
        "Jeg har hentet data for alle kvartaler siden 2015. "
        "For Rudersdal starter antallet på 120 i 2015 og stiger til 345 i 2026, "
        "mens for Hørsholm starter på 125 i 2015 og stiger til 340 i 2026."
    ),
    "a3": (
        "antallet af 80+ i Rudersdal har øget fra 1234 i kvartal 1 2015 til 1300 i kvartal 4 2015"
    ),
    "a6": (
        "For Rudersdal (område 230):\n- 2015K1: 1.125\n- 2026K3: 1.805\n"
        "For Hørsholm (område 223):\n- 2015K1: 1.120\n- 2026K3: 1.810"
    ),
}
_QUESTION_A = "Hvordan har antallet af 80+ i Rudersdal og Hørsholm udviklet sig siden 2015?"
_ENGLISH_WITH_FACTS = (
    "The population of people aged 80+ in Hørsholm has steadily increased "
    "from 1,813 in 2016 to 2,697 in 2026. This represents a growth of "
    "approximately 49.1% over the 10-year period. The population in 2025 "
    "(2,629) is the highest. Average annual increase of about 5.2%. "
    "A projected population of 2,743 in 2026."
)


def _population_fact(value: int, period: str = "2026K3", place: str = "Rudersdal") -> dict:
    return {
        "table": "FOLK1A",
        "period": period,
        "unit": "Antal",
        "value": value,
        "detail": f"OMRÅDE: {place}, ALDER: 80+",
        "unpublished": False,
        "url": "https://www.statistikbanken.dk/FOLK1A",
    }


def test_empty_facts_replace_the_hallucinated_answers() -> None:
    for key, answer in _HALLUCINATED.items():
        guarded = guard_answer(_QUESTION_A, answer, [])
        assert guarded == NO_FETCHED_NUMBERS, key
        assert "1234" not in guarded
        assert "1.125" not in guarded
        assert "1,813" not in guarded


def test_hallucinated_turn_keeps_the_tool_trace_and_hides_the_numbers() -> None:
    tools = FakeTools({})
    model = ScriptedModel(
        [
            ChatReply("", [_call("hent_data", {"table": "FOLK1A"}, "c1")]),
            ChatReply(_HALLUCINATED["a1"], []),
        ]
    )
    result = asyncio.run(run_turn(model, tools, _QUESTION_A, max_tool_calls=8))
    assert result.facts == []
    assert result.csv == ""
    assert result.answer == NO_FETCHED_NUMBERS
    assert "1234" not in result.answer
    assert result.calls[0]["name"] == "hent_data"


def test_unverified_numbers_are_removed_when_facts_exist() -> None:
    facts = [
        _population_fact(3524, "2015K1"),
        _population_fact(5045, "2026K3"),
        _population_fact(1719, "2015K1", "Hørsholm"),
        _population_fact(2790, "2026K3", "Hørsholm"),
    ]
    guarded = guard_answer(_QUESTION_A, _ENGLISH_WITH_FACTS, facts)
    assert guarded == UNVERIFIED_NUMBERS
    for invented in ("1,813", "2,697", "49.1", "5.2", "2,629", "2,743"):
        assert invented not in guarded


def test_single_fact_summary_does_not_depend_on_the_prose() -> None:
    fact = _population_fact(5045)
    question = "Hvor mange over 80 bor i Rudersdal?"
    line = "5.045 (FOLK1A, 2026K3, Antal)"
    honest = guard_answer(
        question,
        "Der bor 5.045 personer på 80 år eller derover i Rudersdal i 2026K3.",
        [fact],
    )
    invented = guard_answer(question, "There are 9,999 people.", [fact])
    english = guard_answer(question, "There are 5,045 people in 2026Q3.", [fact])
    assert honest == invented == english == line


def test_a_count_question_uses_only_the_latest_period() -> None:
    facts = [
        _population_fact(4200, "2016K3"),
        _population_fact(5045, "2026K3"),
        _population_fact(2100, "2016K3", "Hørsholm"),
        _population_fact(2790, "2026K3", "Hørsholm"),
    ]
    answer = guard_answer(
        "Hvor mange over 80 bor i Rudersdal?",
        "In 2016K3 there were 4,200 and in 2026K3 there were 5,045.",
        facts,
    )
    assert answer == "5.045 (FOLK1A, 2026K3, Antal)\n2.790 (FOLK1A, 2026K3, Antal)"
    assert "2016" not in answer
    assert "4.200" not in answer


def test_a_danish_refusal_without_new_numbers_is_kept() -> None:
    answer = "Narnia er ikke et område i statistikken. Der er ingen tal for den kommune."
    assert guard_answer("Hvor mange bor i Narnia?", answer, []) == answer
    repeated = guard_answer(
        "Hvor mange over 80 bor i Rudersdal?",
        "Jeg kan ikke slå 80 år op uden en tabel.",
        [],
    )
    assert repeated.startswith("Jeg kan ikke")


def test_each_question_starts_a_fresh_conversation() -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.seen: list[str] = []

        def complete(self, messages: list[dict], tools: list[dict] | None) -> ChatReply:
            self.seen.append("\n".join(str(item.get("content") or "") for item in messages))
            return ChatReply("Ingen tal her.", [])

    tools = FakeTools({})
    model = RecordingModel()
    first = asyncio.run(run_turn(model, tools, "Hvor mange bor i Silkeborg?", max_tool_calls=4))
    second = asyncio.run(run_turn(model, tools, "Hvor mange bor i Rudersdal?", max_tool_calls=4))
    assert "Silkeborg" in model.seen[0]
    assert "Silkeborg" not in model.seen[1]
    assert "Rudersdal" in model.seen[1]
    assert "latest period" in model.seen[1]
    assert "Danish" in model.seen[1]
    assert first.answer == "Ingen tal her."
    assert second.answer == "Ingen tal her."


def test_llm_timeout_is_distinct_from_a_statbank_timeout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def time_out(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("model hung")

    http = httpx.Client(transport=httpx.MockTransport(time_out))
    client = ChatClient("http://127.0.0.1:11434/v1", "ollama", "qwen2.5", 1, http=http)
    with pytest.raises(LLMError, match="Modellen svarede ikke i tide"):
        client.complete([{"role": "user", "content": "hej"}], [])

    settings = Settings(
        llm="ollama",
        base_url="http://127.0.0.1:11434/v1",
        model="qwen2.5",
        api_key="ollama",
        host="127.0.0.1",
        port=0,
        daily_cap=10,
        per_minute=10,
        max_question_chars=100,
        max_tool_calls=2,
        llm_timeout=5,
        tool_timeout=30,
    )

    def explode(question: str) -> TurnResult:
        raise TimeoutError(f"Statistikbanken stalled on {question}")

    handler = _handler(settings, Limiter(10, 10), explode)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        response = _post(port, {"question": "hej"})
        assert response.status == 504
        body = json.loads(response.read().decode())
        assert body["error"] == "Statistikbanken svarede ikke i tide."
    finally:
        server.shutdown()
    logged = capsys.readouterr().err
    assert "TimeoutError" in logged
    assert "Statistikbanken stalled" in logged


def test_timeout_defaults_are_split(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STATBANK_LLM_TIMEOUT", raising=False)
    monkeypatch.delenv("STATBANK_TOOL_TIMEOUT", raising=False)
    monkeypatch.delenv("STATBANK_TIMEOUT", raising=False)
    settings = settings_from_env([])
    assert settings.llm_timeout == 45
    assert settings.tool_timeout == 30


def test_csv_button_is_hidden_without_facts() -> None:
    assert "data.facts && data.facts.length && data.csv" in PAGE
    assert _turn("Hvor mange bor i Narnia?", "Ingen tal.", [], []).csv == ""


def test_chart_keeps_the_last_period_inside_the_frame() -> None:
    assert "right = 56" in PAGE
    assert 'text-anchor": last ? "end"' in PAGE
    assert "width - 4" in PAGE


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
        llm_timeout=5,
        tool_timeout=5,
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


def test_chart_series_groups_periods_and_skips_a_single_point() -> None:
    facts = [
        _fact("Rudersdal", "2015K1", 4200),
        _fact("Rudersdal", "2026K3", 5045),
        _fact("Hørsholm", "2015K1", 2100),
        _fact("Hørsholm", "2026K3", 2790),
        _fact("Rudersdal", "2026K4", 0, unpublished=True),
        {"table": "FOLK1A", "period": "2020", "unit": "Antal", "value": 1, "detail": "Kun ét"},
    ]
    series = chart_series(facts)
    assert [item["label"] for item in series] == [
        "OMRÅDE: Rudersdal",
        "OMRÅDE: Hørsholm",
    ]
    assert [point["period"] for point in series[0]["points"]] == ["2015K1", "2026K3"]
    assert series[0]["points"][1]["value"] == 5045
    assert series[1]["unit"] == "Antal"


def test_csv_uses_bom_semicolon_and_decimal_comma() -> None:
    facts = [
        _fact("Rudersdal", "2015K1", 5045),
        _fact("Rudersdal", "2015K2", 12.5),
    ]
    text = facts_to_csv(facts)
    assert text.startswith("\ufeff")
    header = text.splitlines()[0].lstrip("\ufeff")
    assert header == "tabel;periode;dimensioner;værdi;enhed"
    assert "FOLK1A;2015K1;OMRÅDE: Rudersdal;5045;Antal" in text
    assert "FOLK1A;2015K2;OMRÅDE: Rudersdal;12,5;Antal" in text


@pytest.mark.live
def test_multi_period_call_returns_several_facts() -> None:
    from statistikbanken_mcp.server import hent_data

    result = hent_data(
        "FOLK1A",
        values={
            "OMRÅDE": ["Rudersdal", "Hørsholm"],
            "KØN": ["TOT"],
            "CIVILSTAND": ["TOT"],
            "Tid": [">=2015K1"],
        },
        sum_from={"ALDER": "80"},
    )
    facts = _facts_from("hent_data", json.dumps(result))
    assert len(facts) >= 8
    assert facts[0]["url"] == "https://www.statistikbanken.dk/FOLK1A"
    series = chart_series(facts)
    labels = {item["label"] for item in series}
    assert any("Rudersdal" in label for label in labels)
    assert any("Hørsholm" in label for label in labels)
    assert all(len(item["points"]) >= 4 for item in series)


def _fact(place: str, period: str, value: int | float, unpublished: bool = False) -> dict:
    return {
        "table": "FOLK1A",
        "period": period,
        "unit": "Antal",
        "value": value,
        "detail": f"OMRÅDE: {place}",
        "unpublished": unpublished,
        "url": "https://www.statistikbanken.dk/FOLK1A",
    }


@pytest.mark.live
def test_scripted_model_recovers_from_the_bad_time_sum() -> None:
    """The exact failing call errors, then the corrected call charts both places."""

    bad = {
        "table": "FOLK1A",
        "values": {"ALDER": [">=80"], "OMRÅDE": ["Rudersdal", "Hørsholm"]},
        "sum_range": {"Tid": ["2015K1", "2026K3"]},
    }
    as_strings = {
        "table": "FOLK1A",
        "values": {"OMRÅDE": "230", "Tid": "(1)"},
        "sum_from": {"ALDER": "80"},
    }
    good = {
        "table": "FOLK1A",
        "values": {"OMRÅDE": ["Rudersdal", "Hørsholm"], "Tid": [">=2015K1"]},
        "sum_from": {"ALDER": "80"},
    }

    async def _run() -> TurnResult:
        from mcp import Client

        model = ScriptedModel(
            [
                ChatReply("", [_call("hent_data", as_strings, "c1")]),
                ChatReply("", [_call("hent_data", bad, "c2")]),
                ChatReply("", [_call("hent_data", good, "c3")]),
                ChatReply(
                    "I Rudersdal og Hørsholm er der flere på 80 år i 2026K3 end i 2015K1.",
                    [],
                ),
            ]
        )
        async with Client(server_command(), read_timeout_seconds=120) as client:
            runner = McpRunner(client)
            await runner.prepare()
            return await run_turn(
                model,
                runner,
                "Hvordan har antallet af personer på 80 år og derover udviklet sig "
                "i Rudersdal og Hørsholm siden 2015?",
                max_tool_calls=8,
            )

    result = asyncio.run(_run())
    assert "list_type" not in result.calls[0]["result"]
    assert "5045" in result.calls[0]["result"]
    assert "13078" not in result.calls[0]["result"]
    assert "sum_from" in result.calls[1]["result"]
    assert ">=2015K1" in result.calls[1]["result"]
    assert "13078" not in result.calls[1]["result"]
    assert all(fact.get("value") != 13078 for fact in result.facts)
    assert len(result.facts) >= 8
    assert len(result.chart) == 2
    assert "2026K3" in result.answer
    assert "5,045" not in result.answer


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
