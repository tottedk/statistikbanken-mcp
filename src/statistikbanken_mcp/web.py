"""Local web page that asks an LLM to call the Statistikbanken MCP server.

The model is any OpenAI-compatible chat-completions endpoint. Ollama and the
xAI API both accept the same tools payload on /v1/chat/completions.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import os
import re
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol

import httpx
from mcp import Client, StdioServerParameters

OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_OLLAMA_MODEL = "qwen2.5"
DEFAULT_XAI_MODEL = "grok-4"
TRACE_LIMIT = 2_000
MODEL_RESULT_LIMIT = 12_000
MAX_BODY_BYTES = 32_768
DEFAULT_LLM_TIMEOUT = 45.0
DEFAULT_TOOL_TIMEOUT = 30.0
NO_FETCHED_NUMBERS = "Jeg fik ikke hentet tal, så jeg viser ingen."
UNVERIFIED_NUMBERS = (
    "Jeg viser kun de hentede tal. Teksten er udeladt, fordi den indeholder tal, "
    "der ikke kommer fra tabellen."
)

SYSTEM_PROMPT = """You answer questions about Danish official statistics.
Use only the Statistikbanken tools. Do not invent numbers.
Never write a number that is not in a hent_data row from this question.
If a tool call failed, say so. Do not fill in example figures.
Call find_tabel, then tabelinfo when you need codes, then hent_data.
Always name the table id and the period in the answer.
If the table title, unit, or description only partly matches the question, say so.
If the tools have no data, say so.
Prefer sum_from or sum_range when a total is needed.
A value like >=80 lists rows. It does not add them.
Ages 80 and older are sum_from {"ALDER": "80"}.
You may also pass >=80 in values together with that sum_from. It is the same request.
sum_from and sum_range are arguments beside values, not keys inside values.
Unless the question asks for a change, a development, or several years, use only the latest period.
Do not list earlier periods for a question that asks how many there are now.
For a change over time, fetch every period in one hent_data call.
Pass Tid as one range in values, for example >=2015K1.
Each period comes back as its own row. Do not put Tid in sum_range.
Do not sum periods together with several places or an age range.
Municipality names may be passed as labels.
One code may be a string. Several codes are a list, or one comma-separated string.
Answer in the same language as the question.
A Danish question gets a Danish answer, even if the tool text is English.
Do not answer in English when the question is Danish.
Write Danish numbers with a dot as the thousands separator: 5.045, never 5,045.
Write periods as Statistikbanken does: 2026K3, never 2026Q3."""

PAGE = """<!DOCTYPE html>
<html lang="da">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Statistikbanken</title>
<style>
  :root { color-scheme: light; }
  body { margin: 0; font: 18px/1.45 system-ui, sans-serif; background: #f6f4ef; color: #1c1915; }
  main { max-width: 40rem; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }
  h1 { font-size: 1.6rem; font-weight: 650; margin: 0 0 0.25rem; }
  p.lead { margin: 0 0 1.5rem; color: #4a453e; }
  form { display: grid; gap: 0.75rem; }
  textarea { width: 100%; min-height: 6rem; box-sizing: border-box; font: inherit;
    padding: 0.75rem; border: 1px solid #c9c2b6; border-radius: 8px; background: #fff; }
  button { font: inherit; padding: 0.55rem 1rem; border: 0; border-radius: 8px;
    background: #1c1915; color: #fff; justify-self: start; }
  button:disabled { opacity: 0.55; }
  section { margin-top: 1.75rem; }
  h2 { font-size: 1rem; margin: 0 0 0.5rem; }
  .answer { white-space: pre-wrap; }
  .fact { background: #fff; border: 1px solid #e2dcd2; border-radius: 8px;
    padding: 0.75rem 1rem; margin: 0 0 0.5rem; }
  .fact strong { font-size: 1.25rem; }
  .meta { color: #4a453e; font-size: 0.95rem; }
  details { background: #fff; border: 1px solid #e2dcd2; border-radius: 8px;
    padding: 0.5rem 0.75rem; margin: 0 0 0.5rem; }
  summary { cursor: pointer; }
  pre { white-space: pre-wrap; word-break: break-word;
    font: 0.85rem/1.4 ui-monospace, monospace; margin: 0.5rem 0 0; }
  .error { color: #8a1f1f; }
  .facts { max-height: 16rem; overflow: auto; }
  .chart { background: #fff; border: 1px solid #e2dcd2; border-radius: 8px; padding: 0.75rem; }
  .chart svg { width: 100%; height: auto; display: block; }
  .legend { display: flex; flex-wrap: wrap; gap: 0.75rem; margin-top: 0.5rem; font-size: 0.95rem; }
  .swatch { display: inline-block; width: 0.8rem; height: 0.8rem; margin-right: 0.3rem; }
  .row { display: flex; gap: 0.75rem; align-items: center; }
  a { color: #1d4e89; }
  footer { margin-top: 2rem; color: #4a453e; font-size: 0.9rem; }
</style>
</head>
<body>
<main>
  <h1>Statistikbanken</h1>
  <p class="lead">Stil et spørgsmål. Svaret bygger på tal fra Danmarks Statistik.</p>
  <form>
    <label for="question">Dit spørgsmål</label>
    <textarea id="question" name="question" maxlength="__MAX__" required
      placeholder="Hvor mange over 80 bor i Rudersdal?"></textarea>
    <button type="submit">Spørg</button>
  </form>
  <div id="out"></div>
  <footer>Kilde: Danmarks Statistik. Tabellens titel viser, hvad tallet dækker.</footer>
</main>
<script>
const form = document.querySelector("form");
const out = document.querySelector("#out");
const button = form.querySelector("button");

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

const COLORS = ["#1c1915", "#9a3412", "#1d4e89", "#3f6212"];

function svgEl(name, attrs) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

function drawChart(series) {
  const width = 640, height = 280, left = 64, right = 56, top = 16, bottom = 36;
  const periods = [];
  const seen = new Set();
  for (const item of series) for (const point of item.points) {
    if (!seen.has(point.period)) { seen.add(point.period); periods.push(point.period); }
  }
  periods.sort();
  let min = Infinity, max = -Infinity;
  for (const item of series) for (const point of item.points) {
    min = Math.min(min, point.value);
    max = Math.max(max, point.value);
  }
  if (min === max) { min -= 1; max += 1; }
  const span = Math.max(periods.length - 1, 1);
  const xOf = (period) => left + (periods.indexOf(period) / span) * (width - left - right);
  const yOf = (value) => top + (1 - (value - min) / (max - min)) * (height - top - bottom);
  const svg = svgEl("svg", {viewBox: "0 0 " + width + " " + height, role: "img"});
  const title = svgEl("title", {});
  title.textContent = "Udvikling over tid";
  svg.append(title);
  svg.append(svgEl("line", {x1: left, y1: top, x2: left, y2: height - bottom, stroke: "#c9c2b6"}));
  svg.append(svgEl("line", {
    x1: left, y1: height - bottom, x2: width - right, y2: height - bottom, stroke: "#c9c2b6"
  }));
  for (const value of [max, (min + max) / 2, min]) {
    const label = svgEl("text", {
      x: left - 6, y: yOf(value) + 4, "text-anchor": "end", fill: "#4a453e", "font-size": "11"
    });
    label.textContent = Math.round(value).toLocaleString("da-DK");
    svg.append(label);
  }
  const step = Math.ceil(periods.length / 6);
  periods.forEach((period, index) => {
    if (index % step && index !== periods.length - 1) return;
    const last = index === periods.length - 1;
    const first = index === 0;
    const label = svgEl("text", {
      x: last ? width - 4 : (first ? left : xOf(period)),
      y: height - 12,
      "text-anchor": last ? "end" : (first ? "start" : "middle"),
      fill: "#4a453e",
      "font-size": "11"
    });
    label.textContent = period;
    svg.append(label);
  });
  series.forEach((item, index) => {
    const points = item.points.map((point) => xOf(point.period) + "," + yOf(point.value)).join(" ");
    svg.append(svgEl("polyline", {
      points, fill: "none", stroke: COLORS[index % COLORS.length], "stroke-width": "2"
    }));
  });
  const wrap = el("div", null, "chart");
  wrap.append(svg);
  return wrap;
}

function legend(series) {
  const row = el("div", null, "legend");
  series.forEach((item, index) => {
    const bit = el("span");
    const swatch = el("span", null, "swatch");
    swatch.style.background = COLORS[index % COLORS.length];
    bit.append(swatch, document.createTextNode(item.label || item.table || ""));
    row.append(bit);
  });
  return row;
}

function csvButton(csv) {
  const button = el("button", "Hent CSV");
  button.type = "button";
  button.addEventListener("click", () => {
    const blob = new Blob([csv], {type: "text/csv;charset=utf-8"});
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "statistikbanken.csv";
    link.click();
    URL.revokeObjectURL(link.href);
  });
  return button;
}

function render(data) {
  out.replaceChildren();
  if (data.error) {
    out.append(el("p", data.error, "error"));
    return;
  }
  const answer = el("section");
  const heading = el("div", null, "row");
  heading.append(el("h2", "Svar"));
  if (data.facts && data.facts.length && data.csv) heading.append(csvButton(data.csv));
  answer.append(heading, el("p", data.answer || "", "answer"));
  out.append(answer);

  if (data.chart && data.chart.length) {
    const chart = el("section");
    chart.append(el("h2", "Udvikling"), drawChart(data.chart), legend(data.chart));
    out.append(chart);
  }

  const facts = el("section");
  facts.append(el("h2", "Tal"));
  const linked = new Set();
  for (const fact of data.facts || []) {
    if (fact.url && fact.table && !linked.has(fact.table)) {
      linked.add(fact.table);
      const link = document.createElement("a");
      link.href = fact.url;
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = "Tabel " + fact.table + " på Statistikbanken";
      facts.append(link);
    }
  }
  const list = el("div", null, "facts");
  if (!data.facts || data.facts.length === 0) {
    list.append(el("p", "Ingen tal hentet.", "meta"));
  }
  for (const fact of data.facts || []) {
    const card = el("div", null, "fact");
    const value = fact.value == null ? "–" : String(fact.value);
    card.append(el("strong", value));
    const bits = [
      "Tabel " + (fact.table || ""),
      "Periode " + (fact.period || ""),
      "Enhed " + (fact.unit || ""),
    ];
    if (fact.detail) bits.push(fact.detail);
    if (fact.unpublished) bits.push("Ikke offentliggjort endnu");
    card.append(el("div", bits.filter(Boolean).join(" · "), "meta"));
    list.append(card);
  }
  facts.append(list);
  out.append(facts);

  const trace = el("section");
  trace.append(el("h2", "Værktøjskald"));
  if (!data.calls || data.calls.length === 0) {
    trace.append(el("p", "Ingen kald.", "meta"));
  }
  for (const call of data.calls || []) {
    const item = document.createElement("details");
    item.append(el("summary", call.name || "kald"));
    item.append(el("div", "Argumenter", "meta"));
    item.append(el("pre", call.arguments || ""));
    item.append(el("div", "Resultat", "meta"));
    item.append(el("pre", call.result || ""));
    trace.append(item);
  }
  out.append(trace);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = new FormData(form).get("question").toString().trim();
  button.disabled = true;
  out.replaceChildren(el("p", "Henter tal…", "meta"));
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await response.json();
    render(data);
  } catch (error) {
    render({ error: "Kunne ikke hente et svar." });
  } finally {
    button.disabled = false;
  }
});
</script>
</body>
</html>
"""


class LLMError(Exception):
    """The model could not be called. The message is safe to show."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] | None
    raw: dict[str, Any]


@dataclass
class ChatReply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


class ToolRunner(Protocol):
    async def list_tools(self) -> list[ToolSpec]: ...

    async def call(self, name: str, arguments: dict[str, Any]) -> str: ...


class ChatClient:
    """POST /v1/chat/completions. The same body works for Ollama and xAI."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        http: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._owns_http = http is None
        self._http = http or httpx.Client(timeout=timeout)

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> ChatReply:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0}
        if tools is not None:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        try:
            response = self._http.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException as exc:
            raise LLMError("Modellen svarede ikke i tide.") from exc
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise LLMError("Kunne ikke kontakte modellen.") from exc
        return _parse_reply(payload)


def _parse_reply(payload: dict[str, Any]) -> ChatReply:
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("Modellen svarede i et ukendt format.") from exc
    content = message.get("content") or ""
    if not isinstance(content, str):
        content = ""
    calls: list[ToolCall] = []
    for index, raw in enumerate(message.get("tool_calls") or []):
        function = raw.get("function") or {}
        name = str(function.get("name") or "")
        arguments = _parse_arguments(function.get("arguments"))
        call_id = str(raw.get("id") or f"call_{index}")
        calls.append(ToolCall(id=call_id, name=name, arguments=arguments, raw=raw))
    return ChatReply(content=content, tool_calls=calls)


def _parse_arguments(raw: Any) -> dict[str, Any] | None:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def openai_tools(specs: list[ToolSpec]) -> list[dict[str, Any]]:
    tools = []
    for spec in specs:
        parameters = spec.parameters or {"type": "object", "properties": {}}
        if "type" not in parameters:
            parameters = {"type": "object", **parameters}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": parameters,
                },
            }
        )
    return tools


@dataclass
class TurnResult:
    answer: str
    facts: list[dict[str, Any]]
    calls: list[dict[str, str]]
    chart: list[dict[str, Any]] = field(default_factory=list)
    csv: str = ""


async def run_turn(
    llm: ChatClient,
    runner: ToolRunner,
    question: str,
    *,
    max_tool_calls: int,
) -> TurnResult:
    specs = await runner.list_tools()
    tools = openai_tools(specs)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "system", "content": _closing_instruction(question)},
    ]
    calls: list[dict[str, str]] = []
    facts: list[dict[str, Any]] = []
    used = 0
    while True:
        allow_tools = used < max_tool_calls
        reply = await asyncio.to_thread(llm.complete, messages, tools if allow_tools else None)
        if not reply.tool_calls or not allow_tools:
            answer = reply.content.strip() or _fallback_answer(calls, capped=not allow_tools)
            return _turn(question, answer, facts, calls)
        messages.append(_assistant_message(reply))
        for call in reply.tool_calls:
            if used >= max_tool_calls:
                break
            result = await _invoke(runner, call)
            used += 1
            calls.append(
                {
                    "name": call.name,
                    "arguments": _short(
                        json.dumps(call.arguments, ensure_ascii=False, indent=2),
                        TRACE_LIMIT,
                    ),
                    "result": _short(result, TRACE_LIMIT),
                }
            )
            facts.extend(_facts_from(call.name, result))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": _short(result, MODEL_RESULT_LIMIT),
                }
            )
        if used >= max_tool_calls:
            final = await asyncio.to_thread(llm.complete, messages, None)
            answer = final.content.strip() or _fallback_answer(calls, capped=True)
            return _turn(question, answer, facts, calls)


def _assistant_message(reply: ChatReply) -> dict[str, Any]:
    tool_calls = []
    for call in reply.tool_calls:
        arguments = call.raw.get("function", {}).get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(call.arguments or {}, ensure_ascii=False)
        tool_calls.append(
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": arguments},
            }
        )
    return {"role": "assistant", "content": reply.content or "", "tool_calls": tool_calls}


async def _invoke(runner: ToolRunner, call: ToolCall) -> str:
    if not call.name:
        return "Tool call is missing a name."
    if call.arguments is None:
        return "Tool arguments were not valid JSON."
    try:
        return await runner.call(call.name, call.arguments)
    except Exception as exc:
        return f"Tool failed: {exc.__class__.__name__}"


def _fallback_answer(calls: list[dict[str, str]], *, capped: bool) -> str:
    if capped:
        return (
            "Jeg nåede grænsen for opslag og kan ikke svare færdigt. Se værktøjskaldene nedenfor."
        )
    if not calls:
        return "Jeg fandt ingen tal at svare ud fra."
    return "Modellen gav ikke et svar. Prøv at omformulere spørgsmålet."


def _facts_from(name: str, result: str) -> list[dict[str, Any]]:
    if name != "hent_data":
        return []
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    periods = payload.get("periods") if isinstance(payload.get("periods"), list) else []
    facts = []
    for row in payload.get("rows") or []:
        if not isinstance(row, dict):
            continue
        period = row.get("Tid")
        if not period and len(periods) == 1:
            period = periods[0]
        detail = ", ".join(
            f"{key}: {value}"
            for key, value in row.items()
            if key not in {"value", "unpublished", "Tid"}
        )
        facts.append(
            {
                "table": payload.get("table") or "",
                "period": "" if period is None else str(period),
                "unit": payload.get("unit") or "",
                "value": row.get("value"),
                "detail": detail,
                "unpublished": row.get("unpublished") is True,
                "url": _table_url(str(payload.get("table") or "")),
            }
        )
    return facts


def _table_url(table: str) -> str:
    if table.isalnum():
        return f"https://www.statistikbanken.dk/{table}"
    return ""


def chart_series(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group numeric facts that share a table and dimensions across periods."""
    grouped: dict[tuple[str, str, str], dict[str, int | float]] = {}
    order: list[tuple[str, str, str]] = []
    for fact in facts:
        if fact.get("unpublished"):
            continue
        value = fact.get("value")
        period = str(fact.get("period") or "")
        if not period or isinstance(value, bool) or not isinstance(value, int | float):
            continue
        key = (
            str(fact.get("table") or ""),
            str(fact.get("detail") or ""),
            str(fact.get("unit") or ""),
        )
        if key not in grouped:
            grouped[key] = {}
            order.append(key)
        grouped[key][period] = value
    series = []
    for key in order:
        points = grouped[key]
        if len(points) < 2:
            continue
        table, detail, unit = key
        series.append(
            {
                "label": detail or table,
                "table": table,
                "unit": unit,
                "points": [
                    {"period": period, "value": points[period]} for period in sorted(points)
                ],
            }
        )
    return series


def facts_to_csv(facts: list[dict[str, Any]]) -> str:
    """UTF-8 CSV with BOM, semicolon separator, and a decimal comma."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=";", lineterminator="\r\n")
    writer.writerow(["tabel", "periode", "dimensioner", "værdi", "enhed"])
    for fact in facts:
        writer.writerow(
            [
                fact.get("table") or "",
                fact.get("period") or "",
                fact.get("detail") or "",
                _csv_number(fact.get("value")),
                fact.get("unit") or "",
            ]
        )
    return "\ufeff" + buffer.getvalue()


def _csv_number(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        number = int(value) if value.is_integer() else value
        return str(number).replace(".", ",")
    return str(value)


def _turn(
    question: str,
    answer: str,
    facts: list[dict[str, Any]],
    calls: list[dict[str, str]],
) -> TurnResult:
    # The model text is not a source of numbers. A figure stays only when this
    # turn's hent_data rows contain it. Anything else is dropped, because a
    # wrong but plausible number is worse than no sentence.
    return TurnResult(
        answer=guard_answer(question, answer, facts),
        facts=facts,
        calls=calls,
        chart=chart_series(facts),
        csv=facts_to_csv(facts) if facts else "",
    )


_DANISH_WORDS = frozenset(
    """
    hvor mange hvad hvem hvordan hvilken hvilke hvorfor bor har er siden
    udviklet antallet over under børn kommune og til fra den det en et
    ikke personer tabellen
    """.split()
)
_TREND_MARKERS = (
    "udvikl",
    "siden",
    "over tid",
    "historik",
    "ændr",
    "aendr",
    "trend",
    "sammenlign",
    "development",
    "over time",
    "since",
    "history",
)
_PERIOD_RE = re.compile(r"\b(?:19|20)\d{2}(?:K[1-4]|Q[1-4]|M(?:0[1-9]|1[0-2]))\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_NUMBER_RE = re.compile(
    r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?"
)


def _looks_danish(question: str) -> bool:
    if re.search(r"[æøåÆØÅ]", question):
        return True
    words = set(re.findall(r"[a-zA-ZæøåÆØÅ]+", question.casefold()))
    return len(words & _DANISH_WORDS) >= 2


def _asks_for_trend(question: str) -> bool:
    folded = question.casefold()
    return any(marker in folded for marker in _TREND_MARKERS)


def _closing_instruction(question: str) -> str:
    """Last instruction before the model writes. Small models follow the end."""
    if _looks_danish(question):
        language = (
            "The question is Danish. Answer in Danish. Do not answer in English. "
            "Write 5.045 and 2026K3."
        )
    else:
        language = "Answer in the same language as the question."
    if _asks_for_trend(question):
        scope = (
            "The question asks how the numbers change over time. "
            "Fetch the periods in one hent_data call and cite those rows."
        )
    else:
        scope = (
            "The question asks for one figure. Use only the latest period. "
            "Do not return a long list of earlier periods."
        )
    return (
        f"{language} {scope} "
        "Use only numbers that appear in hent_data rows from this question. "
        "If a tool call failed, say so and do not invent figures."
    )


def guard_answer(question: str, answer: str, facts: list[dict[str, Any]]) -> str:
    """Drop prose that states a number this turn did not fetch.

    A question that asks for one figure is answered with the page's own line
    for the latest period. The model's sentence is not used, so a long series
    or an invented total cannot become the answer.
    """
    if not facts:
        if _unverified_numbers(answer, question, facts):
            return NO_FETCHED_NUMBERS
        return answer
    if not _asks_for_trend(question):
        lines = _latest_lines(facts)
        if lines:
            return "\n".join(lines)
    if _unverified_numbers(answer, question, facts):
        if _single_fact(facts):
            return _summary_line(facts[0])
        return UNVERIFIED_NUMBERS
    if _single_fact(facts):
        line = _summary_line(facts[0])
        return f"{line}\n\n{answer}" if answer else line
    return answer


def _latest_lines(facts: list[dict[str, Any]]) -> list[str]:
    numeric = [fact for fact in facts if _single_value(fact)]
    if not numeric:
        return []
    latest = max(str(fact.get("period") or "") for fact in numeric)
    chosen = [fact for fact in numeric if str(fact.get("period") or "") == latest]
    return [_summary_line(fact) for fact in chosen]


def _single_value(fact: dict[str, Any]) -> bool:
    value = fact.get("value")
    return isinstance(value, int | float) and not isinstance(value, bool)


def _single_fact(facts: list[dict[str, Any]]) -> bool:
    return len(facts) == 1 and _single_value(facts[0])


def _summary_line(fact: dict[str, Any]) -> str:
    value = _da_number(fact["value"])
    bits = [str(fact.get(key) or "") for key in ("table", "period", "unit")]
    label = ", ".join(bit for bit in bits if bit)
    return f"{value} ({label})" if label else value


def _da_number(value: int | float) -> str:
    if isinstance(value, float) and not value.is_integer():
        sign = "-" if value < 0 else ""
        whole, frac = f"{abs(value):.10f}".rstrip("0").rstrip(".").split(".")
        grouped = f"{int(whole):,}".replace(",", ".")
        return f"{sign}{grouped},{frac}"
    number = int(value)
    sign = "-" if number < 0 else ""
    return sign + f"{abs(number):,}".replace(",", ".")


def _unverified_numbers(answer: str, question: str, facts: list[dict[str, Any]]) -> list[float]:
    allowed = set(_numbers_in(question))
    for fact in facts:
        value = fact.get("value")
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        allowed.add(float(value))
    return [number for number in _numbers_in(answer) if not _near(number, allowed)]


def _near(number: float, allowed: set[float]) -> bool:
    return any(abs(number - item) < 1e-6 for item in allowed)


def _numbers_in(text: str) -> list[float]:
    masked = _YEAR_RE.sub(lambda match: " " * len(match.group(0)), _mask_periods(text))
    return [_parse_stat_number(match.group(0)) for match in _NUMBER_RE.finditer(masked)]


def _mask_periods(text: str) -> str:
    return _PERIOD_RE.sub(lambda match: " " * len(match.group(0)), text)


def _parse_stat_number(token: str) -> float:
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", token):
        return float(token.replace(".", ""))
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", token):
        return float(token.replace(",", ""))
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+,\d+", token):
        whole, frac = token.split(",")
        return float(f"{whole.replace('.', '')}.{frac}")
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+\.\d+", token):
        whole, frac = token.split(".")
        return float(f"{whole.replace(',', '')}.{frac}")
    if "," in token and "." not in token:
        return float(token.replace(",", "."))
    return float(token)


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return any(_is_timeout(item) for item in exc.exceptions)
    return "timeout" in type(exc).__name__.casefold()


def _short(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


class Limiter:
    def __init__(self, daily_cap: int, per_minute: int) -> None:
        self.daily_cap = daily_cap
        self.per_minute = per_minute
        self._lock = threading.Lock()
        self._day = date.today()
        self._count = 0
        self._hits: dict[str, deque[float]] = {}

    def allow(self, ip: str) -> str | None:
        now = time.monotonic()
        with self._lock:
            today = date.today()
            if today != self._day:
                self._day = today
                self._count = 0
                self._hits.clear()
            hits = self._hits.setdefault(ip, deque())
            while hits and now - hits[0] >= 60:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return "rate"
            if self._count >= self.daily_cap:
                return "daily"
            hits.append(now)
            self._count += 1
            return None


_LIMIT_TEXT = {
    "rate": "For mange spørgsmål fra denne adresse. Prøv igen om et minut.",
    "daily": "Dagens grænse for spørgsmål er nået.",
}


@dataclass
class Settings:
    llm: str
    base_url: str
    model: str
    api_key: str
    host: str
    port: int
    daily_cap: int
    per_minute: int
    max_question_chars: int
    max_tool_calls: int
    llm_timeout: float
    tool_timeout: float

    def __repr__(self) -> str:
        return (
            f"Settings(llm={self.llm!r}, model={self.model!r}, base_url={self.base_url!r}, "
            f"host={self.host!r}, port={self.port})"
        )


def load_dotenv(path: str = ".env") -> None:
    """Set variables that are not already in the environment. Never prints values."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            key, value = text.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def settings_from_env(argv: list[str] | None = None) -> Settings:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Local Statistikbanken web demo")
    parser.add_argument("--llm", choices=("ollama", "xai"))
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args(argv)
    llm = args.llm or os.environ.get("STATBANK_LLM", "ollama")
    if llm == "xai":
        base_url = args.base_url or os.environ.get("STATBANK_BASE_URL", XAI_BASE_URL)
        model = args.model or os.environ.get("STATBANK_MODEL", DEFAULT_XAI_MODEL)
        api_key = os.environ.get("XAI_API_KEY", "")
    else:
        llm = "ollama"
        base_url = args.base_url or os.environ.get("STATBANK_BASE_URL", OLLAMA_BASE_URL)
        model = args.model or os.environ.get("STATBANK_MODEL", DEFAULT_OLLAMA_MODEL)
        api_key = os.environ.get("STATBANK_API_KEY", "ollama")
    return Settings(
        llm=llm,
        base_url=base_url,
        model=model,
        api_key=api_key,
        host=args.host or os.environ.get("STATBANK_HOST", "127.0.0.1"),
        port=args.port or int(os.environ.get("STATBANK_PORT", "8080")),
        daily_cap=int(os.environ.get("STATBANK_DAILY_CAP", "200")),
        per_minute=int(os.environ.get("STATBANK_RATE_PER_MINUTE", "30")),
        max_question_chars=int(os.environ.get("STATBANK_MAX_QUESTION_CHARS", "1000")),
        max_tool_calls=int(os.environ.get("STATBANK_MAX_TOOL_CALLS", "8")),
        llm_timeout=float(os.environ.get("STATBANK_LLM_TIMEOUT", str(DEFAULT_LLM_TIMEOUT))),
        tool_timeout=float(os.environ.get("STATBANK_TOOL_TIMEOUT", str(DEFAULT_TOOL_TIMEOUT))),
    )


def server_command(tool_timeout: float | None = None) -> StdioServerParameters:
    env = dict(os.environ)
    if tool_timeout is not None:
        env["STATBANK_HTTP_TIMEOUT"] = str(tool_timeout)
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "statistikbanken_mcp"],
        env=env,
    )


def list_server_tools() -> list[str]:
    async def _list() -> list[str]:
        async with Client(server_command(), read_timeout_seconds=30) as client:
            listed = await client.list_tools()
            return [tool.name for tool in listed.tools]

    return asyncio.run(_list())


class McpRunner:
    def __init__(self, client: Any) -> None:
        self._client = client
        self._listed: list[Any] = []

    async def prepare(self) -> None:
        listed = await self._client.list_tools()
        self._listed = listed.tools

    async def list_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name=tool.name,
                description=tool.description or "",
                parameters=dict(tool.input_schema or {}),
            )
            for tool in self._listed
        ]

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        result = await self._client.call_tool(name, arguments)
        parts = [block.text for block in result.content if getattr(block, "text", None)]
        return "\n".join(parts)


def answer_with_server(settings: Settings, question: str) -> TurnResult:
    if settings.llm == "xai" and not settings.api_key:
        raise LLMError("XAI_API_KEY mangler. Sæt den i miljøet eller i .env.")

    async def _run() -> TurnResult:
        llm = ChatClient(
            settings.base_url,
            settings.api_key,
            settings.model,
            settings.llm_timeout,
        )
        try:
            async with Client(
                server_command(settings.tool_timeout),
                read_timeout_seconds=settings.tool_timeout + 5,
            ) as client:
                runner = McpRunner(client)
                await runner.prepare()
                return await run_turn(
                    llm,
                    runner,
                    question,
                    max_tool_calls=settings.max_tool_calls,
                )
        finally:
            llm.close()

    return asyncio.run(_run())


def serve(settings: Settings, answer: Any | None = None) -> None:
    limited = Limiter(settings.daily_cap, settings.per_minute)
    handler = _handler(settings, limited, answer or _default_answer(settings))
    server = ThreadingHTTPServer((settings.host, settings.port), handler)
    url = f"http://{settings.host}:{settings.port}"
    print(f"Statistikbanken på {url} ({settings.llm}, {settings.model})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopper.", flush=True)
    finally:
        server.server_close()


def _default_answer(settings: Settings):
    def answer(question: str) -> TurnResult:
        return answer_with_server(settings, question)

    return answer


def _handler(settings: Settings, limiter: Limiter, answer: Any) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/":
                self._send(404, b"Not found", "text/plain; charset=utf-8")
                return
            page = PAGE.replace("__MAX__", str(settings.max_question_chars))
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/api/ask":
                self._json(404, {"error": "Ukendt adresse."})
                return
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length > MAX_BODY_BYTES:
                self._json(400, {"error": "Spørgsmålet er for langt."})
                return
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
                question = str(payload.get("question", "")).strip()
            except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                self._json(400, {"error": "Spørgsmålet kunne ikke læses."})
                return
            if not question:
                self._json(400, {"error": "Skriv et spørgsmål."})
                return
            if len(question) > settings.max_question_chars:
                self._json(400, {"error": "Spørgsmålet er for langt."})
                return
            ip = self.client_address[0]
            limited = limiter.allow(ip)
            if limited:
                self._json(429, {"error": _LIMIT_TEXT[limited]})
                return
            try:
                result = answer(question)
            except LLMError as exc:
                sys.stderr.write(f"Question failed: {type(exc).__name__}: {exc}\n")
                self._json(502, {"error": str(exc)})
                return
            except Exception as exc:
                sys.stderr.write(f"Question failed: {type(exc).__name__}: {exc}\n")
                traceback.print_exc(file=sys.stderr)
                if _is_timeout(exc):
                    self._json(504, {"error": "Statistikbanken svarede ikke i tide."})
                else:
                    self._json(502, {"error": "Kunne ikke hente et svar."})
                return
            self._json(
                200,
                {
                    "answer": result.answer,
                    "facts": result.facts,
                    "calls": result.calls,
                    "chart": result.chart,
                    "csv": result.csv,
                },
            )

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:
            message = fmt % args  # noqa: UP031
            sys.stderr.write(f"{self.address_string()} - {message}\n")

    return Handler


def main(argv: list[str] | None = None) -> None:
    serve(settings_from_env(argv))


if __name__ == "__main__":
    main()
