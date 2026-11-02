"""Web demo: tool loop, limits, and the real MCP server's tool list."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import httpx
import pytest

from statistikbanken_mcp.web import (
    LLM_TIMEOUT_TEXT,
    NO_DATA_ANSWER,
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
    apply_age_intent,
    chart_axis,
    chart_series,
    display_label,
    facts_to_csv,
    facts_to_grid,
    facts_to_json,
    facts_to_xlsx,
    guard_answer,
    list_server_tools,
    nice_axis,
    period_label,
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
                "OMRÅDE": "Esbjerg",
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
            ChatReply("Der bor 5.045 personer på 80 år eller derover i Esbjerg i 2026K3.", []),
        ]
    )
    result = asyncio.run(
        run_turn(model, tools, "Hvor mange over 80 bor i Esbjerg?", max_tool_calls=8)
    )
    assert "5.045" in result.answer
    assert [call["name"] for call in result.calls] == ["find_tabel", "hent_data"]
    assert result.facts == [
        {
            "table": "FOLK1A",
            "period": "2026K3",
            "unit": "Antal",
            "value": 5045,
            "detail": "OMRÅDE: Esbjerg, ALDER: 80+",
            "title": "",
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
    assert "1. kvartal 2015" in SYSTEM_PROMPT
    assert "2015 1. kvt." in SYSTEM_PROMPT
    assert "Do not write 2015K1 or 2026Q3." in SYSTEM_PROMPT
    assert "Do not put Tid in sum_range" in SYSTEM_PROMPT
    assert "first value and the last value" in SYSTEM_PROMPT
    assert "percentage change" in SYSTEM_PROMPT
    assert "is allowed when it is computed" in SYSTEM_PROMPT
    assert "Refer to areas and categories by name, not by code." in SYSTEM_PROMPT
    assert "Do not open with the table id or the table title." in SYSTEM_PROMPT
    single = _closing_instruction("Hvor mange over 80 bor i Esbjerg?")
    assert "Do not open with the table id or the table title." in single
    assert "2015 1. kvt." in single
    assert "Danish" in single
    assert "latest period" in single
    assert "do not invent" in single.casefold()
    trend = _closing_instruction(
        "Hvordan har antallet af 80+ i Esbjerg og Viborg udviklet sig siden 2015?"
    )
    assert "change over time" in trend
    assert "Danish" in trend


# Excerpts from the qwen2.5 retest (pr7r-ui-log), where the model wrote numbers
# after hent_data failed and the page showed them with an empty fact list.
_HALLUCINATED = {
    "a1": (
        "Data for Esbjerg og Viborg er summen af alle personer 80 år og over.\n"
        '{"Tid": "2026K3", "OMRÅDE": "561", "ALDER": "80+", "value": 1234},\n'
        '{"Tid": "2026K3", "OMRÅDE": "791", "ALDER": "80+", "value": 567}'
    ),
    "a2": (
        "Jeg har hentet data for alle kvartaler siden 2015. "
        "For Esbjerg starter antallet på 120 i 2015 og stiger til 345 i 2026, "
        "mens for Viborg starter på 125 i 2015 og stiger til 340 i 2026."
    ),
    "a3": (
        "antallet af 80+ i Esbjerg har øget fra 1234 i kvartal 1 2015 til 1300 i kvartal 4 2015"
    ),
    "a6": (
        "For Esbjerg (område 561):\n- 2015K1: 1.125\n- 2026K3: 1.805\n"
        "For Viborg (område 791):\n- 2015K1: 1.120\n- 2026K3: 1.810"
    ),
}
_QUESTION_A = "Hvordan har antallet af 80+ i Esbjerg og Viborg udviklet sig siden 2015?"
_ENGLISH_WITH_FACTS = (
    "The population of people aged 80+ in Viborg has steadily increased "
    "from 1,813 in 2016 to 2,697 in 2026. This represents a growth of "
    "approximately 49.1% over the 10-year period. The population in 2025 "
    "(2,629) is the highest. Average annual increase of about 5.2%. "
    "A projected population of 2,743 in 2026."
)


def _population_fact(value: int, period: str = "2026K3", place: str = "Esbjerg") -> dict:
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


def test_a_number_in_the_tool_result_may_be_quoted() -> None:
    text = "Befolkningen den 1. i kvartalet: Esbjerg fra 3.524 (2015K1) til 5.045 (2026K3)"
    facts = [
        _population_fact(3524, "2015K1"),
        _population_fact(5045, "2026K3"),
    ]
    calls = [
        {
            "name": "hent_data",
            "arguments": "{}",
            "result": (
                '{"table": "FOLK1A", "title": "Befolkningen den 1. i kvartalet", '
                '"rows": [{"value": 3524}, {"value": 5045}]}'
            ),
        }
    ]
    question = "Hvordan har udviklingen i Esbjerg været siden 2015?"
    kept = guard_answer(question, text, facts, calls)
    assert kept == text
    invented = guard_answer(question, text + ", i alt 9.999", facts, calls)
    assert invented == UNVERIFIED_NUMBERS
    assert "9.999" not in invented


def test_suppressed_prose_logs_the_flagged_numbers(caplog: pytest.LogCaptureFixture[str]) -> None:
    text = (
        "Befolkningen den 1. i kvartalet: Esbjerg fra 3.524 (2015K1) "
        "til 5.045 (2026K3), i alt 9.999"
    )
    facts = [
        _population_fact(3524, "2015K1"),
        _population_fact(5045, "2026K3"),
    ]
    calls = [
        {
            "name": "hent_data",
            "arguments": "{}",
            "result": '{"table": "FOLK1A", "title": "Befolkningen den 1. i kvartalet"}',
        }
    ]
    question = "Hvordan har udviklingen i Esbjerg været siden 2015?"
    with caplog.at_level("INFO", logger="statistikbanken_mcp.web"):
        guarded = guard_answer(question, text, facts, calls)
    assert guarded == UNVERIFIED_NUMBERS
    logged = "\n".join(caplog.messages)
    assert "Suppressed prose; flagged numbers: 9999" in logged
    assert text not in logged
    assert "9.999" not in logged


def _fetched_series() -> tuple[str, list[dict], list[dict[str, str]]]:
    """Trend question, hent_data rows, and metadata that must not authorise numbers."""
    question = "Hvordan har udviklingen været?"
    facts = [
        _population_fact(3524, "2015K1"),
        _population_fact(5045, "2026K3"),
        _population_fact(1785.5, "2024"),
    ]
    for fact in facts:
        fact["detail"] = "ALDER: 80 år"
    calls = [
        {
            "name": "find_tabel",
            "arguments": "{}",
            "result": '[{"id": "FOLK1A", "title": "Tabel 42"}]',
        },
        {
            "name": "tabelinfo",
            "arguments": "{}",
            "result": (
                '{"numeric_codes": {"from": "0", "to": "125"}, '
                '"values": [{"id": "79", "text": "79 år"}, {"id": "561", "text": "Esbjerg"}], '
                '"time_span": {"first": "2008K1", "count": 75}}'
            ),
        },
        {
            "name": "hent_data",
            "arguments": "{}",
            "result": (
                '{"table": "FOLK1A", "title": "Befolkningen den 1. i kvartalet", '
                '"description": "Se også tabel 99", "rows": ['
            ),
        },
    ]
    return question, facts, calls


def test_prose_may_quote_question_values_labels_and_title() -> None:
    question, facts, calls = _fetched_series()
    kept = [
        "I 2015, 2024 og 2026 var tallene 3.524, 1.785,5 og 5.045.",
        "Fra 2015K1 til 2026K3.",
        "5045 er det samme som 5.045 og 5 045.",
        "Decimalen er 1.785,5 og 1785,5 og 1 785,5.",
        "Gruppen 80 år havde 3.524.",
        "Befolkningen den 1. i kvartalet havde 3.524.",
        "5\u00a0045 personer.",
    ]
    for text in kept:
        assert guard_answer(question, text, facts, calls) == text
    asked = "Hvordan har udviklingen været sammenlignet med 1.800?"
    quoted = "Tallet 3.524 er ikke 1.800."
    assert guard_answer(asked, quoted, facts, calls) == quoted


def test_prose_numbers_only_in_metadata_are_suppressed(
    caplog: pytest.LogCaptureFixture[str],
) -> None:
    question, facts, calls = _fetched_series()
    blocked = {
        125: "Der er 125 alderskoder, og tallet er 3.524.",
        79: "Gruppen 79 år havde 3.524.",
        561: "Område 561 havde 3.524.",
        1800: "Omkring 1.800 i 2024, mens det hentede tal er 3.524.",
        2008: "Siden 2008 er tallet 3.524.",
        2016: "I 2016K2 var tallet 3.524.",
        42: "Tabel 42 viser 3.524.",
        99: "Se tabel 99 med tallet 3.524.",
        75: "Der er 75 perioder, og tallet er 3.524.",
    }
    for number, text in blocked.items():
        caplog.clear()
        with caplog.at_level("INFO", logger="statistikbanken_mcp.web"):
            guarded = guard_answer(question, text, facts, calls)
        assert guarded == UNVERIFIED_NUMBERS
        assert str(number) not in guarded
        assert caplog.messages == [f"Suppressed prose; flagged numbers: {number}"]


def test_area_code_from_hent_data_arguments_may_be_quoted(
    caplog: pytest.LogCaptureFixture[str],
) -> None:
    question = "Hvordan har udviklingen været?"
    facts = [
        _population_fact(3524, "2015K1"),
        _population_fact(5045, "2026K3"),
    ]
    text = "Område 561 og 791 havde 3.524 og 5.045."
    metadata = '{"values": [{"id": "561", "text": "Esbjerg"}, {"id": "791", "text": "Viborg"}]}'
    with_codes = [
        {"name": "tabelinfo", "arguments": "{}", "result": metadata},
        {
            "name": "hent_data",
            "arguments": json.dumps({"table": "FOLK1A", "values": {"OMRÅDE": ["561", "791"]}}),
            "result": "{}",
        },
    ]
    assert guard_answer(question, text, facts, with_codes) == text
    names_only = [
        {"name": "tabelinfo", "arguments": "{}", "result": metadata},
        {
            "name": "find_tabel",
            "arguments": json.dumps({"query": "561 791"}),
            "result": metadata,
        },
        {
            "name": "hent_data",
            "arguments": json.dumps(
                {"table": "FOLK1A", "values": {"OMRÅDE": ["Esbjerg", "Viborg"]}}
            ),
            "result": "{}",
        },
    ]
    caplog.clear()
    with caplog.at_level("INFO", logger="statistikbanken_mcp.web"):
        guarded = guard_answer(question, text, facts, names_only)
    assert guarded == UNVERIFIED_NUMBERS
    assert caplog.messages == ["Suppressed prose; flagged numbers: 561, 791"]


def test_unverified_numbers_are_removed_when_facts_exist() -> None:
    facts = [
        _population_fact(3524, "2015K1"),
        _population_fact(5045, "2026K3"),
        _population_fact(1719, "2015K1", "Viborg"),
        _population_fact(2790, "2026K3", "Viborg"),
    ]
    guarded = guard_answer(_QUESTION_A, _ENGLISH_WITH_FACTS, facts)
    assert guarded == UNVERIFIED_NUMBERS
    for invented in ("1,813", "2,697", "49.1", "5.2", "2,629", "2,743"):
        assert invented not in guarded


def test_single_fact_summary_does_not_depend_on_the_prose() -> None:
    fact = _population_fact(5045)
    question = "Hvor mange over 80 bor i Esbjerg?"
    line = "5.045 (FOLK1A, 2026 3. kvt., Antal, Esbjerg, 80+)"
    honest = guard_answer(
        question,
        "Der bor 5.045 personer på 80 år eller derover i Esbjerg i 2026K3.",
        [fact],
    )
    invented = guard_answer(question, "There are 9,999 people.", [fact])
    english = guard_answer(question, "There are 5,045 people in 2026Q3.", [fact])
    assert honest == invented == english == line


def test_a_count_question_uses_only_the_latest_period() -> None:
    facts = [
        _population_fact(4200, "2016K3"),
        _population_fact(5045, "2026K3"),
        _population_fact(2100, "2016K3", "Viborg"),
        _population_fact(2790, "2026K3", "Viborg"),
    ]
    answer = guard_answer(
        "Hvor mange over 80 bor i Esbjerg?",
        "In 2016K3 there were 4,200 and in 2026K3 there were 5,045.",
        facts,
    )
    assert answer == (
        "5.045 (FOLK1A, 2026 3. kvt., Antal, Esbjerg, 80+)\n"
        "2.790 (FOLK1A, 2026 3. kvt., Antal, Viborg, 80+)"
    )
    assert "2016" not in answer
    assert "4.200" not in answer


def test_no_data_answers_use_a_fixed_danish_sentence() -> None:
    english = (
        'Tool calls failed because there is no data for "Narnia" in Statistikbanken. '
        "Narnia is not a real municipality in Denmark."
    )
    norwegian = (
        'Fant ingen data om "Narnia" i Statistikbanken. '
        "Denne navnet passer ikke bra til vores database."
    )
    assert guard_answer("Hvor mange bor i Narnia?", english, []) == NO_DATA_ANSWER
    assert guard_answer("Hvor mange bor i Narnia?", norwegian, []) == NO_DATA_ANSWER
    assert "Fant" not in NO_DATA_ANSWER


def test_wrong_municipality_is_not_presented_as_the_answer() -> None:
    fact = _population_fact(18114)
    fact["detail"] = "OMRÅDE: København, ALDER: 80+"
    answer = guard_answer("Hvor mange over 80 bor i Esbjerg?", "18.114", [fact])
    assert answer == "Jeg fik hentet tal for København, ikke for Esbjerg."
    assert "18.114" not in answer
    hidden = _turn("Hvor mange over 80 bor i Esbjerg?", "18.114", [fact], [])
    assert hidden.answer == answer
    assert hidden.facts == []
    assert hidden.chart == []
    assert hidden.csv == ""
    assert hidden.show_numbers is False


def test_question_age_replaces_the_codes_the_model_sent() -> None:
    question_a = "Hvordan har udviklingen for 80+ i Esbjerg og Viborg været siden 2015?"
    rewritten_a = apply_age_intent(
        question_a,
        {
            "table": "FOLK1A",
            "values": {"KOM": ["Esbjerg", "Viborg"], "ALDER": "80", "Tid": [">=2015K1"]},
        },
    )
    assert rewritten_a["values"] == {
        "KOM": ["Esbjerg", "Viborg"],
        "Tid": [">=2015K1"],
    }
    assert rewritten_a["sum_from"] == {"ALDER": "80"}
    assert "sum_range" not in rewritten_a

    question_c = "Hvor mange børn (0-17 år) bor i Viborg?"
    first = apply_age_intent(
        question_c,
        {
            "table": "FOLK1A",
            "values": {"OMRÅDE": "Viborg", "ALDER": "0-17"},
            "sum_from": {"ALDER": "0"},
        },
    )
    assert first["values"] == {"OMRÅDE": "Viborg"}
    assert "sum_from" not in first
    assert first["sum_range"] == {"ALDER": ["0", "17"]}
    second = apply_age_intent(
        question_c,
        {"table": "FOLK1A", "values": {"OMRÅDE": ["791"]}, "sum_from": {"ALDER": 0}},
    )
    assert second["sum_range"] == {"ALDER": ["0", "17"]}
    assert "sum_from" not in second


def test_wrong_age_is_not_presented_as_the_answer() -> None:
    fact = _population_fact(395, "2015K1")
    fact["detail"] = "OMRÅDE: Esbjerg, ALDER: 80 år"
    question = "Hvordan har udviklingen for 80+ i Esbjerg været siden 2015?"
    hidden = _turn(question, "Se grafen.", [fact], [])
    assert hidden.answer == "Jeg fik hentet tal for alder 80 år, ikke for 80+."
    assert hidden.show_numbers is False
    assert hidden.facts == []
    assert hidden.chart == []
    assert "395" not in hidden.answer


def test_llm_timeout_inside_a_task_group_is_http_504() -> None:
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
        llm_timeout=45,
        tool_timeout=30,
    )

    def timed_out(question: str) -> TurnResult:
        async def fail() -> None:
            raise LLMError(LLM_TIMEOUT_TEXT)

        async def grouped() -> None:
            async with asyncio.TaskGroup() as group:
                group.create_task(fail())

        asyncio.run(grouped())

    handler = _handler(settings, Limiter(10, 10), timed_out)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        response = _post(port, {"question": "hej"})
        assert response.status == 504
        body = json.loads(response.read().decode())
        assert body["error"] == "Modellen svarede ikke i tide. Prøv igen."
        assert "Kunne ikke hente et svar." not in body["error"]
    finally:
        server.shutdown()


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
    second = asyncio.run(run_turn(model, tools, "Hvor mange bor i Esbjerg?", max_tool_calls=4))
    assert "Silkeborg" in model.seen[0]
    assert "Silkeborg" not in model.seen[1]
    assert "Esbjerg" in model.seen[1]
    assert "latest period" in model.seen[1]
    assert "Danish" in model.seen[1]
    assert first.answer == NO_DATA_ANSWER
    assert second.answer == NO_DATA_ANSWER


def test_llm_timeout_is_distinct_from_a_statbank_timeout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def time_out(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("model hung")

    http = httpx.Client(transport=httpx.MockTransport(time_out))
    client = ChatClient("http://127.0.0.1:11434/v1", "ollama", "qwen2.5", 1, http=http)
    with pytest.raises(LLMError, match="Modellen svarede ikke i tide. Prøv igen."):
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


def test_llm_timeout_is_http_504() -> None:
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
        llm_timeout=45,
        tool_timeout=30,
    )

    def timed_out(question: str) -> TurnResult:
        raise LLMError("Modellen svarede ikke i tide. Prøv igen.")

    handler = _handler(settings, Limiter(10, 10), timed_out)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        response = _post(port, {"question": "hej"})
        assert response.status == 504
        body = json.loads(response.read().decode())
        assert body["error"] == "Modellen svarede ikke i tide. Prøv igen."
    finally:
        server.shutdown()


def test_timeout_defaults_are_split(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STATBANK_LLM_TIMEOUT", raising=False)
    monkeypatch.delenv("STATBANK_TOOL_TIMEOUT", raising=False)
    monkeypatch.delenv("STATBANK_TIMEOUT", raising=False)
    settings = settings_from_env([])
    assert settings.llm_timeout == 45
    assert settings.tool_timeout == 30


def test_fact_cards_and_the_axis_use_danish_dots() -> None:
    assert "function daDigits" in PAGE
    assert "daDigits(Math.round(value))" in PAGE
    chart = PAGE.split("function drawChart", 1)[1].split("function daDigits", 1)[0]
    assert "toLocaleString" not in chart
    assert "for (const value of axis.ticks)" in chart
    assert "daTick(value)" in chart


def test_csv_button_is_hidden_without_facts() -> None:
    assert "data.facts && data.facts.length && data.csv" in PAGE
    assert _turn("Hvor mange bor i Narnia?", "Ingen tal.", [], []).csv == ""


def test_page_has_basis_and_avanceret_modes() -> None:
    assert '<fieldset class="modes">' in PAGE
    assert "<legend>Visning</legend>" in PAGE
    assert '<input type="radio" name="mode" value="basic" checked>' in PAGE
    assert '<input type="radio" name="mode" value="advanced">' in PAGE
    assert "> Basis</label>" in PAGE
    assert "> Avanceret</label>" in PAGE
    assert 'const MODE_KEY = "statistikbanken-mode"' in PAGE
    assert "localStorage.getItem(MODE_KEY)" in PAGE
    assert "localStorage.setItem(MODE_KEY, mode)" in PAGE
    change = PAGE.split('input.addEventListener("change"', 1)[1].split("function advancedOnly", 1)[
        0
    ]
    assert "fetch(" not in change
    assert "if (last) render(last)" in change
    assert 'node.dataset.mode = "advanced"' in PAGE
    assert "advancedOnly(csvButton(data.csv))" in PAGE
    assert "advancedOnly(chart)" in PAGE
    assert "advancedOnly(link)" in PAGE
    assert 'return mode === "advanced" && data.show_numbers !== false;' in PAGE
    assert "advancedView(data) && data.facts && data.facts.length && data.csv" in PAGE
    assert "advancedView(data) && data.chart && data.chart.length" in PAGE
    assert "if (data.error)" in PAGE
    render = PAGE.split("function render", 1)[1].split("form.addEventListener", 1)[0]
    assert 'if (mode === "advanced") out.append(advancedOnly(toolCalls(data)));' in render
    assert "Værktøjskald" not in render
    assert render.index("if (data.error)") < render.index("toolCalls")
    assert render.index("out.append(facts)") < render.index("toolCalls(data)")
    assert "show_numbers === false) return" not in render
    assert render.index("drawGrid") < render.index("Hent Excel")
    assert "Hent Excel" in PAGE
    assert "Hent JSON" in PAGE
    assert "Hent graf (PNG)" in PAGE
    assert "/api/export" in PAGE
    assert "<script src=" not in PAGE


def test_chart_labels_sit_on_the_points() -> None:
    chart = PAGE.split("function drawChart", 1)[1].split("function daDigits", 1)[0]
    assert "right = 56" in chart
    assert "x: xOf(period)" in chart
    assert "width - 4" not in chart
    assert 'text-anchor": last ? "end"' in chart
    assert 'font-family": "system-ui, sans-serif"' in chart
    png = PAGE.split("function pngButton", 1)[1].split("function drawGrid", 1)[0]
    assert ".legend" in png
    assert "system-ui, sans-serif" in png
    assert "getBBox" in png
    assert "plotWidth" in png
    assert "canvas.width = width * scale" in png
    assert "640 * scale" not in png


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
    assert result.answer == NO_DATA_ANSWER


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


def _post(port: int, payload: dict, path: str = "/api/ask") -> HTTPConnection:
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    body = json.dumps(payload).encode()
    connection.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    return connection.getresponse()


def test_chart_series_groups_periods_and_skips_a_single_point() -> None:
    facts = [
        _fact("Esbjerg", "2015K1", 4200),
        _fact("Esbjerg", "2026K3", 5045),
        _fact("Viborg", "2015K1", 2100),
        _fact("Viborg", "2026K3", 2790),
        _fact("Esbjerg", "2026K4", 0, unpublished=True),
        {"table": "FOLK1A", "period": "2020", "unit": "Antal", "value": 1, "detail": "Kun ét"},
    ]
    series = chart_series(facts)
    assert [item["label"] for item in series] == [
        "Esbjerg",
        "Viborg",
    ]
    assert [point["period"] for point in series[0]["points"]] == ["2015K1", "2026K3"]
    assert series[0]["points"][1]["value"] == 5045
    assert series[0]["points"][0]["label"] == "2015 1. kvt."
    assert series[0]["points"][1]["label"] == "2026 3. kvt."
    assert series[1]["unit"] == "Antal"


def test_axis_ticks_are_nice_numbers_covering_min_and_max() -> None:
    low, high, ticks = nice_axis(1719, 5045)
    assert ticks == [1500, 2500, 3500, 4500, 5500]
    assert low == 1500 and high == 5500
    assert low <= 1719 and high >= 5045
    flat_low, flat_high, flat = nice_axis(5045, 5045)
    assert flat_low < 5045 < flat_high
    assert flat[0] == flat_low and flat[-1] == flat_high
    zero, hundred, even = nice_axis(0, 100)
    assert even == [0, 20, 40, 60, 80, 100]
    assert zero == 0 and hundred == 100
    series = chart_series(
        [
            _population_fact(3524, "2015K1"),
            _population_fact(5045, "2026K3"),
            _population_fact(1719, "2015K1", "Viborg"),
            _population_fact(2790, "2026K3", "Viborg"),
        ]
    )
    axis = chart_axis(series)
    assert axis == {"min": 1500, "max": 5500, "ticks": [1500, 2500, 3500, 4500, 5500]}
    assert "drawChart(data.chart, data.axis)" in PAGE


def test_exports_use_the_same_dimension_label_as_the_table() -> None:
    facts = [
        _population_fact(3524, "2015K1"),
        _population_fact(1719, "2015K1", "Viborg"),
    ]
    text = facts_to_csv(facts).lstrip("\ufeff")
    rows = list(csv.reader(io.StringIO(text), delimiter=";"))
    assert rows[1][2] == "Esbjerg, 80+"
    assert rows[2][2] == "Viborg, 80+"
    assert "OMRÅDE:" not in text
    assert "ALDER:" not in text
    sheet = _xlsx_rows(facts_to_xlsx(facts))
    assert [row[2] for row in sheet] == ["dimensioner", "Esbjerg, 80+", "Viborg, 80+"]
    exported = json.loads(facts_to_json(facts))
    assert exported[0]["dimensioner"] == "Esbjerg, 80+"
    assert exported[0]["dimensioner_rå"] == "OMRÅDE: Esbjerg, ALDER: 80+"
    assert exported[1]["dimensioner"] == "Viborg, 80+"


def test_display_label_drops_variable_names_and_totals() -> None:
    detail = "OMRÅDE: Esbjerg, KØN: I alt, ALDER: 80+, CIVILSTAND: I alt"
    assert display_label(detail, "FOLK1A") == "Esbjerg, 80+"


def test_grid_is_periods_by_series_with_plain_labels() -> None:
    one = facts_to_grid([_population_fact(5045)])
    assert one["columns"] == ["Periode", "Esbjerg, 80+"]
    assert one["rows"] == [["2026 3. kvt.", "5.045"]]
    two = facts_to_grid(
        [
            _population_fact(3524, "2015K1"),
            _population_fact(5045, "2026K3"),
            _population_fact(1719, "2015K1", "Viborg"),
            _population_fact(2790, "2026K3", "Viborg"),
        ]
    )
    assert two["columns"] == ["Periode", "Esbjerg, 80+", "Viborg, 80+"]
    assert two["rows"] == [
        ["2015 1. kvt.", "3.524", "1.719"],
        ["2026 3. kvt.", "5.045", "2.790"],
    ]
    long = facts_to_grid(
        [_population_fact(3000 + index, f"20{index:02d}K1") for index in range(20)]
    )
    assert len(long["rows"]) == 20
    assert long["rows"][0][0] == "2000 1. kvt."
    assert long["rows"][-1][0] == "2019 1. kvt."


def test_period_label_uses_danish_names_and_keeps_unknown_codes() -> None:
    assert period_label("2015K1") == "2015 1. kvt."
    assert period_label("2015k4") == "2015 4. kvt."
    assert period_label("2015M03") == "marts 2015"
    assert period_label("2015M12") == "december 2015"
    assert period_label("2015H1") == "2015 1. halvår"
    assert period_label("2015H2") == "2015 2. halvår"
    assert period_label("2015") == "2015"
    assert period_label("2015K5") == "2015K5"
    assert period_label(">=2015K1") == ">=2015K1"
    assert period_label("2015K1+2026K3") == "2015K1+2026K3"


def test_friendly_period_forms_are_not_extra_figures() -> None:
    question = "Hvordan har udviklingen været?"
    quarters = [
        _population_fact(3524, "2015K1"),
        _population_fact(5045, "2026K3"),
    ]
    kept = [
        "Fra 2015 1. kvt. til 2026 3. kvt. var tallene 3.524 og 5.045.",
        "Fra 1. kvartal 2015 til 3. kvartal 2026 var tallene 3.524 og 5.045.",
    ]
    for text in kept:
        assert guard_answer(question, text, quarters) == text
    invented = "Fra 2015 1. kvt. til 2026 3. kvt., i alt 1 person."
    assert guard_answer(question, invented, quarters) == UNVERIFIED_NUMBERS
    month = [_population_fact(100, "2015M03"), _population_fact(110, "2015M04")]
    march = "Fra marts 2015 til april 2015 var tallene 100 og 110."
    assert guard_answer(question, march, month) == march
    halves = [_population_fact(100, "2015H1"), _population_fact(110, "2015H2")]
    half = "Fra 2015 1. halvår til 2015 2. halvår var tallene 100 og 110."
    assert guard_answer(question, half, halves) == half


def _viborg_quarters() -> list[dict]:
    """94.486 is 2014K1. 2013K1 is 94.333. The acceptance prose swapped them."""
    return [
        _population_fact(94333, "2013K1", "Viborg"),
        _population_fact(94486, "2014K1", "Viborg"),
    ]


def test_value_from_another_period_is_suppressed(
    caplog: pytest.LogCaptureFixture[str],
) -> None:
    question = "Hvordan har befolkningen i Viborg udviklet sig siden 2013?"
    facts = _viborg_quarters()
    missed = [
        "The lowest was 94,486 (2013K1).",
        "Viborg havde 94.486 i 2013K1.",
        "Viborg havde 94\u00a0486 i 2013K1.",
        "94.486 (2013 1. kvt.).",
    ]
    for text in missed:
        caplog.clear()
        with caplog.at_level("INFO", logger="statistikbanken_mcp.web"):
            guarded = guard_answer(question, text, facts)
        assert guarded == UNVERIFIED_NUMBERS
        assert caplog.messages == ["Suppressed prose; flagged numbers: 94486"]


def test_value_next_to_its_period_or_with_no_period_is_kept() -> None:
    question = "Hvordan har befolkningen i Viborg udviklet sig siden 2013?"
    facts = _viborg_quarters()
    kept = [
        "The lowest was 94.333 (2013K1).",
        "Viborg havde 94.486 i 2014K1.",
        "94.486 (2014 1. kvt.) og 94.333 (2013K1).",
        "Fra 2013K1 til 2014K1 var tallene 94.333 og 94.486.",
        "Fra 2013 1. kvt. til 2014 1. kvt. var tallene 94.333 og 94.486.",
        "Viborg havde 94.486.",
        "Tallene er 94.333 og 94.486.",
        "Viborg voksede, en forskel på 153.",
    ]
    asked = "Hvordan har befolkningen i Viborg udviklet sig? Forskellen er 153."
    for text in kept:
        assert guard_answer(asked if "153" in text else question, text, facts) == text


def test_friendly_period_label_must_match_that_row() -> None:
    question = "Hvordan har udviklingen været?"
    quarters = _viborg_quarters()
    assert (
        guard_answer(question, "I 2014 1. kvt. var tallet 94.486.", quarters)
        == "I 2014 1. kvt. var tallet 94.486."
    )
    assert guard_answer(question, "I 2014 1. kvt. var tallet 94.333.", quarters) == (
        UNVERIFIED_NUMBERS
    )
    assert guard_answer(question, "2014K1 94.486.", quarters) == "2014K1 94.486."
    assert guard_answer(question, "2013K1 94.486.", quarters) == UNVERIFIED_NUMBERS
    months = [_population_fact(100, "2015M03"), _population_fact(110, "2015M04")]
    assert guard_answer(question, "I marts 2015 var tallet 100.", months) == (
        "I marts 2015 var tallet 100."
    )
    assert guard_answer(question, "110 i marts 2015.", months) == UNVERIFIED_NUMBERS
    assert guard_answer(question, "100 i marts 2015.", months) == "100 i marts 2015."


def test_question_and_label_numbers_are_not_paired_with_a_place() -> None:
    """80 from over 80 / (80+) and 10 from 10 år are not wrong-row values."""
    age = "Hvordan har antallet af personer over 80 i Rudersdal og Hørsholm udviklet sig?"
    series = [
        _population_fact(1719, "2015K1", "Hørsholm"),
        _population_fact(2790, "2026K3", "Hørsholm"),
        _population_fact(3524, "2015K1", "Rudersdal"),
        _population_fact(5045, "2026K3", "Rudersdal"),
        # 80 is also a real row, so the label alone must keep "(80+)" from pairing.
        _population_fact(80, "2015K1", "Esbjerg"),
    ]
    kept = [
        (
            "Hørsholm (80+): første værdi 2015K1 er 1.719, "
            "sidste værdi 2026K3 er 2.790. "
            "Rudersdal (80+): første værdi 2015K1 er 3.524, "
            "sidste værdi 2026K3 er 5.045."
        ),
        "Rudersdal 80+: 3.524 (2015K1) → 5.045 (2026K3)",
        "I 2026K3 boede der 5.045 personer over 80 i Rudersdal",
    ]
    for text in kept:
        assert guard_answer(age, text, series) == text
    decade = "Hvordan har befolkningen i Viborg udviklet sig de seneste 10 år?"
    viborg = [
        _population_fact(95776, "2016K1", "Viborg"),
        _population_fact(96166, "2016K3", "Viborg"),
        _population_fact(98104, "2026K3", "Viborg"),
        # 10 is also a real row, so the question's "10 år" must not be paired.
        _population_fact(10, "2015K1", "Esbjerg"),
    ]
    for text in (
        "Viborg: 10 år siden (2016K1) var tallet 95.776.",
        "I Viborg Kommune boede der 98.104 i 2026K3, mod 96.166 i 2016K3 for 10 år siden.",
        "Viborg (10 år): 95.776 → 98.104.",
    ):
        assert guard_answer(decade, text, viborg) == text


def test_correct_derived_figures_pass_and_invented_ones_do_not() -> None:
    """Viborg FOLK1A: 96,166 in 2016K3 and 98,104 in 2026K3 differ by 1,938."""
    question = "Hvordan har befolkningen i Viborg udviklet sig?"
    facts = [
        _population_fact(96166, "2016K3", "Viborg"),
        _population_fact(98104, "2026K3", "Viborg"),
    ]
    kept = [
        "Viborg voksede, en forskel på 1.938.",
        "Stigningen er 1.938 personer.",
        "I alt 194.270.",
        "En stigning på 2 procent.",
        "En stigning på 2,0 procent.",
        "En stigning på 2,02 procent.",
    ]
    for text in kept:
        assert guard_answer(question, text, facts) == text, text
    invented = [
        "Viborg voksede, en forskel på 1.999.",
        "I alt 9.999.",
        "En stigning på 9 procent.",
        "Der bor 1.938 personer.",
    ]
    for text in invented:
        assert guard_answer(question, text, facts) == UNVERIFIED_NUMBERS, text


def test_derived_figures_do_not_bypass_the_wrong_period_check() -> None:
    question = "Hvordan har befolkningen i Viborg udviklet sig siden 2013?"
    facts = _viborg_quarters()
    assert guard_answer(question, "Viborg havde 94.486 i 2013K1.", facts) == UNVERIFIED_NUMBERS
    assert guard_answer(question, "Viborg havde 94.486 i 2014K1.", facts) == (
        "Viborg havde 94.486 i 2014K1."
    )


def test_projection_and_year_derivation_are_visible() -> None:
    projected = _population_fact(57160, "2030", "Esbjerg")
    projected["projection"] = True
    question = "Hvor mange bor der i Esbjerg i 2030?"
    answer = guard_answer(question, "Der bor 57.160.", [projected])
    assert "57.160" in answer
    assert "fremskrivning" in answer
    grid = facts_to_grid([projected])
    assert "fremskrivning" in grid["rows"][0][1]
    exported = json.loads(facts_to_json([projected]))
    assert exported[0]["fremskrivning"] is True
    assert exported[0]["periode_label"] == "2030 (fremskrivning)"
    yearly = _population_fact(115029, "2025")
    yearly["derivation"] = (
        "Gennemsnit af 2025K1, 2025K2, 2025K3, 2025K4 (niveau, perioderne er gennemsnit)."
    )
    shown = guard_answer("Hvor mange bor i Esbjerg i 2025?", "115.029", [yearly])
    assert "Gennemsnit af 2025K1" in shown
    assert "115.029" in shown


def test_swapped_place_names_are_suppressed() -> None:
    question = "Hvordan har antallet udviklet sig?"
    facts = [
        _population_fact(7556, "2026K3", "Esbjerg"),
        _population_fact(6420, "2026K3", "Viborg"),
    ]
    assert guard_answer(question, "Esbjerg 7.556 / Viborg 6.420", facts) == (
        "Esbjerg 7.556 / Viborg 6.420"
    )
    assert guard_answer(question, "Esbjerg 6.420 / Viborg 7.556", facts) == UNVERIFIED_NUMBERS
    assert guard_answer(question, "Esbjerg 7.556, Viborg 6.420", facts) == (
        "Esbjerg 7.556, Viborg 6.420"
    )
    assert guard_answer(question, "Esbjerg: 6.420 / Viborg: 7.556", facts) == UNVERIFIED_NUMBERS
    # 6.420 is also an earlier Esbjerg figure, so the place alone is not a mismatch.
    both = [
        *facts,
        _population_fact(6420, "2015K1", "Esbjerg"),
    ]
    assert guard_answer(question, "Esbjerg 6.420", both) == "Esbjerg 6.420"
    assert guard_answer(question, "Esbjerg 6.420 i 2026 3. kvt.", both) == UNVERIFIED_NUMBERS
    assert guard_answer(question, "Viborg 6.420 i 2026K3.", both) == "Viborg 6.420 i 2026K3."


def test_json_keeps_the_period_code_and_csv_uses_the_label() -> None:
    facts = [
        _fact("Esbjerg", "2015K1", 5045),
        _fact("Esbjerg", "2015K2", 12.5),
    ]
    text = facts_to_csv(facts).lstrip("\ufeff")
    reader = csv.reader(io.StringIO(text), delimiter=";")
    header = next(reader)
    csv_rows = list(reader)
    assert header == ["tabel", "periode", "dimensioner", "værdi", "enhed"]
    assert csv_rows[0][1] == "2015 1. kvt."
    assert _xlsx_rows(facts_to_xlsx(facts)) == [header, *csv_rows]
    exported = json.loads(facts_to_json(facts))
    assert exported[0]["periode"] == "2015K1"
    assert exported[0]["periode_label"] == "2015 1. kvt."
    assert exported[0]["dimensioner"] == "Esbjerg"
    assert exported[0]["dimensioner_rå"] == "OMRÅDE: Esbjerg"
    assert exported[0]["værdi"] == "5045"
    assert exported[1]["periode"] == "2015K2"
    assert exported[1]["værdi"] == "12,5"


def test_export_endpoint_writes_a_file_and_does_not_ask_again() -> None:
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
        tool_timeout=5,
    )

    def answer(question: str) -> TurnResult:
        raise AssertionError(question)

    handler = _handler(settings, Limiter(10, 10), answer)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    facts = [_fact("Esbjerg", "2015K1", 5045)]
    try:
        workbook = _post(port, {"format": "xlsx", "facts": facts}, "/api/export")
        assert workbook.status == 200
        rows = _xlsx_rows(workbook.read())
        assert rows[0] == ["tabel", "periode", "dimensioner", "værdi", "enhed"]
        assert rows[1][0] == "FOLK1A"
        assert rows[1][1] == "2015 1. kvt."
        assert rows[1][3] == "5045"
        exported = _post(port, {"format": "json", "facts": facts}, "/api/export")
        assert exported.status == 200
        assert json.loads(exported.read().decode()) == json.loads(facts_to_json(facts))
    finally:
        server.shutdown()


def _xlsx_rows(blob: bytes) -> list[list[str]]:
    import xml.etree.ElementTree as ElementTree
    import zipfile

    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(blob)) as book:
        assert "[Content_Types].xml" in book.namelist()
        assert "xl/workbook.xml" in book.namelist()
        sheet = ElementTree.fromstring(book.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.findall("m:sheetData/m:row", ns):
        cells = []
        for cell in row.findall("m:c", ns):
            text = cell.find("m:is/m:t", ns)
            cells.append("" if text is None or text.text is None else text.text)
        rows.append(cells)
    return rows


def test_csv_uses_bom_semicolon_and_decimal_comma() -> None:
    facts = [
        _fact("Esbjerg", "2015K1", 5045),
        _fact("Esbjerg", "2015K2", 12.5),
    ]
    text = facts_to_csv(facts)
    assert text.startswith("\ufeff")
    header = text.splitlines()[0].lstrip("\ufeff")
    assert header == "tabel;periode;dimensioner;værdi;enhed"
    assert "FOLK1A;2015 1. kvt.;Esbjerg;5045;Antal" in text
    assert "FOLK1A;2015 2. kvt.;Esbjerg;12,5;Antal" in text


@pytest.mark.live
def test_multi_period_call_returns_several_facts() -> None:
    from statistikbanken_mcp.server import hent_data

    result = hent_data(
        "FOLK1A",
        values={
            "OMRÅDE": ["Esbjerg", "Viborg"],
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
    assert any("Esbjerg" in label for label in labels)
    assert any("Viborg" in label for label in labels)
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
@pytest.mark.live
def test_scripted_model_uses_the_question_age_not_its_codes() -> None:
    """14b sent age 80, and 0-17 together with sum_from 0. The question wins."""

    async def _run(question: str, arguments: dict, call_id: str) -> TurnResult:
        from mcp import Client

        model = ScriptedModel(
            [
                ChatReply("", [_call("hent_data", arguments, call_id)]),
                ChatReply("Se tallene.", []),
            ]
        )
        async with Client(server_command(), read_timeout_seconds=120) as client:
            runner = McpRunner(client)
            await runner.prepare()
            return await run_turn(model, runner, question, max_tool_calls=4)

    series = asyncio.run(
        _run(
            "Hvordan har udviklingen for 80+ i Esbjerg og Viborg været siden 2015?",
            {
                "table": "FOLK1A",
                "values": {
                    "KOM": ["Esbjerg", "Viborg"],
                    "ALDER": "80",
                    "Tid": [">=2015K1"],
                },
            },
            "c_a",
        )
    )
    by_place = {}
    for fact in series.facts:
        if "Esbjerg" in fact["detail"]:
            by_place.setdefault("Esbjerg", {})[fact["period"]] = fact["value"]
        if "Viborg" in fact["detail"]:
            by_place.setdefault("Viborg", {})[fact["period"]] = fact["value"]
    assert by_place["Esbjerg"]["2015K1"] == 4678
    assert by_place["Esbjerg"]["2026K3"] == 7556
    assert by_place["Viborg"]["2015K1"] == 4126
    assert by_place["Viborg"]["2026K3"] == 6420
    assert all("80+" in fact["detail"] for fact in series.facts)
    assert len(series.chart) == 2
    assert all(len(item["points"]) > 2 for item in series.chart)

    children = asyncio.run(
        _run(
            "Hvor mange børn (0-17 år) bor i Viborg?",
            {
                "table": "FOLK1A",
                "values": {"OMRÅDE": "Viborg", "ALDER": "0-17"},
                "sum_from": {"ALDER": "0"},
            },
            "c_c",
        )
    )
    assert len(children.facts) == 1
    assert children.facts[0]["value"] == 19171
    assert "Viborg" in children.facts[0]["detail"]
    assert "0-17" in children.facts[0]["detail"]
    assert children.answer.startswith("19.171")
    assert children.show_numbers is True


@pytest.mark.live
def test_scripted_model_recovers_from_the_bad_time_sum() -> None:
    """The exact failing call errors, then the corrected call charts both places."""

    bad = {
        "table": "FOLK1A",
        "values": {"ALDER": [">=80"], "OMRÅDE": ["Esbjerg", "Viborg"]},
        "sum_range": {"Tid": ["2015K1", "2026K3"]},
    }
    as_strings = {
        "table": "FOLK1A",
        "values": {"OMRÅDE": "561", "Tid": "(1)"},
        "sum_from": {"ALDER": "80"},
    }
    good = {
        "table": "FOLK1A",
        "values": {"OMRÅDE": ["Esbjerg", "Viborg"], "Tid": [">=2015K1"]},
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
                    "I Esbjerg og Viborg er der flere på 80 år i 2026K3 end i 2015K1.",
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
                "i Esbjerg og Viborg siden 2015?",
                max_tool_calls=8,
            )

    result = asyncio.run(_run())
    assert "list_type" not in result.calls[0]["result"]
    assert "7556" in result.calls[0]["result"]
    assert "22780" not in result.calls[0]["result"]
    assert "not every period" in result.calls[1]["result"]
    assert ">=2015K1" in result.calls[1]["result"]
    assert "22780" not in result.calls[1]["result"]
    assert all(fact.get("value") != 22780 for fact in result.facts)
    assert len(result.facts) >= 8
    assert len(result.chart) == 2
    assert "2026K3" in result.answer
    assert "7,556" not in result.answer


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
