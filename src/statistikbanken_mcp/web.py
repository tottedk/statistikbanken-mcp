"""Local web page that asks an LLM to call the Statistikbanken MCP server.

The model is any OpenAI-compatible chat-completions endpoint. Ollama and the
xAI API both accept the same tools payload on /v1/chat/completions.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hmac
import io
import json
import logging
import math
import os
import re
import sys
import threading
import time
import traceback
import zipfile
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol

import httpx
from mcp import Client, StdioServerParameters

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_OLLAMA_MODEL = "qwen2.5"
DEFAULT_XAI_MODEL = "grok-4"
TRACE_LIMIT = 2_000
MODEL_RESULT_LIMIT = 12_000
MAX_BODY_BYTES = 32_768
EXPORT_MAX_BYTES = 1_048_576
DEFAULT_LLM_TIMEOUT = 45.0
DEFAULT_TOOL_TIMEOUT = 30.0
PUBLIC_MAX_TOKENS = 2048
PUBLIC_MAX_CONCURRENT = 2
NO_FETCHED_NUMBERS = "Jeg fik ikke hentet tal, så jeg viser ingen."
NO_DATA_ANSWER = "Jeg fandt ingen tal i Statistikbanken for det, du spørger om."
LLM_TIMEOUT_TEXT = "Modellen svarede ikke i tide. Prøv igen."
UNVERIFIED_NUMBERS = (
    "Jeg viser kun de hentede tal. Teksten er udeladt, fordi den indeholder tal, "
    "der ikke kommer fra tabellen."
)

SYSTEM_PROMPT = """You answer questions about Danish official statistics.
Use only the Statistikbanken tools. Do not invent numbers.
Never write a number that is not in a hent_data row from this question.
If a tool call failed, say so. Do not fill in example figures.
Call find_tabel, then tabelinfo when you need codes, then hent_data.
Name the period in the answer. Do not open with the table id or the table title.
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
Quote only values from the tool results. For each series, quote the first value and the last value.
Do not compute a difference, a change, or a percentage.
Municipality names may be passed as labels.
One code may be a string. Several codes are a list, or one comma-separated string.
Refer to areas and categories by name, not by code.
Answer in the same language as the question.
A Danish question gets a Danish answer, even if the tool text is English.
Do not answer in English when the question is Danish.
Write Danish numbers with a dot as the thousands separator: 5.045, never 5,045.
Write periods like 1. kvartal 2015 or 2015 1. kvt., marts 2015, or 2015 1. halvår.
A year stays 2015. Do not write 2015K1 or 2026Q3."""

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
  p.demo { margin: -0.75rem 0 1.5rem; color: #4a453e; font-size: 0.95rem; }
  form { display: grid; gap: 0.75rem; }
  textarea { width: 100%; min-height: 6rem; box-sizing: border-box; font: inherit;
    padding: 0.75rem; border: 1px solid #c9c2b6; border-radius: 8px; background: #fff; }
  button { font: inherit; padding: 0.55rem 1rem; border: 0; border-radius: 8px;
    background: #1c1915; color: #fff; justify-self: start; }
  button:disabled { opacity: 0.55; }
  section { margin-top: 1.75rem; }
  h2 { font-size: 1rem; margin: 0 0 0.5rem; }
  .answer { white-space: pre-wrap; }
  .meta { color: #4a453e; font-size: 0.95rem; }
  table { width: 100%; border-collapse: collapse; background: #fff; }
  th, td { padding: 0.35rem 0.6rem; border-bottom: 1px solid #e2dcd2; text-align: left; }
  th { font-size: 0.9rem; color: #4a453e; position: sticky; top: 0; background: #fff; }
  th:not(:first-child), td:not(:first-child) { text-align: right; }
  .exports { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.75rem; }
  details { background: #fff; border: 1px solid #e2dcd2; border-radius: 8px;
    padding: 0.5rem 0.75rem; margin: 0 0 0.5rem; }
  summary { cursor: pointer; }
  pre { white-space: pre-wrap; word-break: break-word;
    font: 0.85rem/1.4 ui-monospace, monospace; margin: 0.5rem 0 0; }
  .error { color: #8a1f1f; }
  .facts { max-height: 16rem; overflow: auto; scrollbar-width: thin; }
  .facts::-webkit-scrollbar { width: 10px; height: 10px; }
  .facts::-webkit-scrollbar-thumb { background: #c9c2b6; border-radius: 8px; }
  .chart { background: #fff; border: 1px solid #e2dcd2; border-radius: 8px; padding: 0.75rem; }
  .chart svg { width: 100%; height: auto; display: block; }
  .legend { display: flex; flex-wrap: wrap; gap: 0.75rem; margin-top: 0.5rem; font-size: 0.95rem; }
  .swatch { display: inline-block; width: 0.8rem; height: 0.8rem; margin-right: 0.3rem; }
  a { color: #1d4e89; }
  footer { margin-top: 2rem; color: #4a453e; font-size: 0.9rem; }
  fieldset.modes { border: 0; margin: 0 0 1.25rem; padding: 0;
    display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: center; }
  fieldset.modes legend { padding: 0; font-size: 0.95rem; color: #4a453e; }
  fieldset.modes label { display: inline-flex; gap: 0.35rem; align-items: center; cursor: pointer; }
  fieldset.modes label:has(:checked) { font-weight: 650; }
  fieldset.modes input { width: 1rem; height: 1rem; accent-color: #1c1915; }
  fieldset.modes input:focus-visible { outline: 2px solid #1d4e89; outline-offset: 2px; }
</style>
</head>
<body>
<main>
  <h1>Statistikbanken</h1>
  <p class="lead">Stil et spørgsmål. Svaret bygger på tal fra Danmarks Statistik.</p>
  __DEMO__
  <fieldset class="modes">
    <legend>Visning</legend>
    <label><input type="radio" name="mode" value="basic" checked> Basis</label>
    <label><input type="radio" name="mode" value="advanced"> Avanceret</label>
  </fieldset>
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
const MODE_KEY = "statistikbanken-mode";
let mode = "basic";
let last = null;
try {
  if (localStorage.getItem(MODE_KEY) === "advanced") mode = "advanced";
} catch (error) {}
for (const input of document.querySelectorAll('input[name="mode"]')) {
  input.checked = input.value === mode;
  input.addEventListener("change", () => {
    if (!input.checked) return;
    mode = input.value;
    try { localStorage.setItem(MODE_KEY, mode); } catch (error) {}
    if (last) render(last);
  });
}

function advancedOnly(node) {
  node.dataset.mode = "advanced";
  return node;
}

function advancedView(data) {
  return mode === "advanced" && data.show_numbers !== false;
}

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

function drawChart(series, axis) {
  const width = 640, height = 280, left = 64, right = 56, top = 16, bottom = 36;
  const periods = [];
  const seen = new Set();
  const labelOf = new Map();
  for (const item of series) for (const point of item.points) {
    if (!seen.has(point.period)) { seen.add(point.period); periods.push(point.period); }
    if (point.label) labelOf.set(point.period, point.label);
  }
  periods.sort();
  let min = axis.min, max = axis.max;
  if (min === max) { min -= 1; max += 1; }
  const span = Math.max(periods.length - 1, 1);
  const xOf = (period) => left + (periods.indexOf(period) / span) * (width - left - right);
  const yOf = (value) => top + (1 - (value - min) / (max - min)) * (height - top - bottom);
  const svg = svgEl("svg", {
    viewBox: "0 0 " + width + " " + height,
    role: "img",
    "font-family": "system-ui, sans-serif"
  });
  const title = svgEl("title", {});
  title.textContent = "Udvikling over tid";
  svg.append(title);
  svg.append(svgEl("line", {x1: left, y1: top, x2: left, y2: height - bottom, stroke: "#c9c2b6"}));
  svg.append(svgEl("line", {
    x1: left, y1: height - bottom, x2: width - right, y2: height - bottom, stroke: "#c9c2b6"
  }));
  for (const value of axis.ticks) {
    const label = svgEl("text", {
      x: left - 6, y: yOf(value) + 4, "text-anchor": "end", fill: "#4a453e", "font-size": "11"
    });
    label.textContent = daTick(value);
    svg.append(label);
  }
  const step = Math.ceil(periods.length / 4);
  periods.forEach((period, index) => {
    if (index % step && index !== periods.length - 1) return;
    const last = index === periods.length - 1;
    const first = index === 0;
    const label = svgEl("text", {
      x: xOf(period),
      y: height - 12,
      "text-anchor": last ? "end" : (first ? "start" : "middle"),
      fill: "#4a453e",
      "font-size": "10"
    });
    const shown = labelOf.get(period) || period;
    label.textContent = shown;
    const tip = svgEl("title", {});
    tip.textContent = shown;
    label.append(tip);
    svg.append(label);
  });
  series.forEach((item, index) => {
    const color = COLORS[index % COLORS.length];
    const points = item.points.map((point) => xOf(point.period) + "," + yOf(point.value)).join(" ");
    svg.append(svgEl("polyline", {
      points, fill: "none", stroke: color, "stroke-width": "2"
    }));
    for (const point of item.points) {
      const dot = svgEl("circle", {
        cx: xOf(point.period), cy: yOf(point.value), r: "3.5", fill: color
      });
      const tip = svgEl("title", {});
      tip.textContent = (point.label || point.period) + ": " + daDigits(point.value);
      dot.append(tip);
      svg.append(dot);
    }
  });
  const wrap = el("div", null, "chart");
  wrap.append(svg);
  return wrap;
}

function daTick(value) {
  if (Math.abs(value - Math.round(value)) < 1e-6) return daDigits(Math.round(value));
  const negative = value < 0;
  const scaled = Math.round(Math.abs(value) * 100) / 100;
  const parts = String(scaled).split(".");
  const text = daDigits(Number(parts[0])) + (parts[1] ? "," + parts[1] : "");
  return negative ? "-" + text : text;
}

function daDigits(value) {
  const negative = value < 0;
  const digits = String(Math.abs(Math.round(value)));
  let out = "";
  for (let i = 0; i < digits.length; i++) {
    if (i && (digits.length - i) % 3 === 0) out += ".";
    out += digits[i];
  }
  return (negative ? "-" : "") + out;
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

function download(blob, filename) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
  URL.revokeObjectURL(link.href);
}

function csvButton(csv) {
  const button = el("button", "Hent CSV");
  button.type = "button";
  button.addEventListener("click", () => {
    download(new Blob([csv], {type: "text/csv;charset=utf-8"}), "statistikbanken.csv");
  });
  return button;
}

function fileButton(label, format, facts) {
  const button = el("button", label);
  button.type = "button";
  const filename = format === "xlsx" ? "statistikbanken.xlsx" : "statistikbanken.json";
  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      const response = await fetch("/api/export", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({format, facts}),
      });
      if (!response.ok) return;
      download(await response.blob(), filename);
    } finally {
      button.disabled = false;
    }
  });
  return button;
}

function pngButton(svg) {
  const button = el("button", "Hent graf (PNG)");
  button.type = "button";
  button.addEventListener("click", () => {
    const box = svg.getBBox();
    const plotWidth = Math.max(1, Math.ceil(box.x + box.width + 8));
    const plotHeight = 280;
    const clone = svg.cloneNode(true);
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    clone.setAttribute("viewBox", "0 0 " + plotWidth + " " + plotHeight);
    clone.setAttribute("width", String(plotWidth));
    clone.setAttribute("height", String(plotHeight));
    const xml = new XMLSerializer().serializeToString(clone);
    const url = URL.createObjectURL(new Blob([xml], {type: "image/svg+xml;charset=utf-8"}));
    const image = new Image();
    image.onload = () => {
      const legend = svg.closest("section").querySelector(".legend");
      const bits = legend ? [...legend.children] : [];
      const extra = bits.length ? 40 : 0;
      const scale = 2;
      const measure = document.createElement("canvas").getContext("2d");
      measure.font = "15px system-ui, sans-serif";
      let legendRight = 8;
      for (const bit of bits) {
        legendRight += 18 + measure.measureText(bit.textContent || "").width + 16;
      }
      const width = Math.max(plotWidth, Math.ceil(legendRight));
      const canvas = document.createElement("canvas");
      canvas.width = width * scale;
      canvas.height = (plotHeight + extra) * scale;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0, plotWidth * scale, plotHeight * scale);
      ctx.font = (15 * scale) + "px system-ui, sans-serif";
      ctx.textBaseline = "middle";
      let x = 8 * scale;
      const y = (plotHeight + extra / 2) * scale;
      bits.forEach((bit, index) => {
        ctx.fillStyle = COLORS[index % COLORS.length];
        ctx.fillRect(x, y - 6 * scale, 12 * scale, 12 * scale);
        x += 18 * scale;
        ctx.fillStyle = "#1c1915";
        const text = bit.textContent || "";
        ctx.fillText(text, x, y);
        x += ctx.measureText(text).width + 16 * scale;
      });
      URL.revokeObjectURL(url);
      canvas.toBlob((png) => download(png, "statistikbanken.png"));
    };
    image.src = url;
  });
  return button;
}

function drawGrid(grid) {
  const wrap = el("div", null, "facts");
  const rows = (grid && grid.rows) || [];
  if (!rows.length) {
    wrap.append(el("p", "Ingen tal hentet.", "meta"));
    return wrap;
  }
  const table = document.createElement("table");
  const head = document.createElement("tr");
  for (const name of grid.columns) head.append(el("th", name));
  const thead = document.createElement("thead");
  thead.append(head);
  const body = document.createElement("tbody");
  for (const row of rows) {
    const line = document.createElement("tr");
    for (const cell of row) line.append(el("td", cell));
    body.append(line);
  }
  table.append(thead, body);
  wrap.append(table);
  return wrap;
}

function toolCalls(data) {
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
  return trace;
}

function render(data) {
  out.replaceChildren();
  if (data.error) {
    out.append(el("p", data.error, "error"));
    return;
  }
  const answer = el("section");
  answer.append(el("h2", "Svar"), el("p", data.answer || "", "answer"));
  out.append(answer);

  let chartSvg = null;
  if (advancedView(data) && data.chart && data.chart.length) {
    const chart = el("section");
    const drawn = drawChart(data.chart, data.axis);
    chartSvg = drawn.querySelector("svg");
    chart.append(el("h2", "Udvikling"), drawn, legend(data.chart));
    out.append(advancedOnly(chart));
  }

  if (data.show_numbers !== false) {
    const facts = el("section");
    facts.append(el("h2", "Tal"));
    if (advancedView(data)) {
      const linked = new Set();
      for (const fact of data.facts || []) {
        if (fact.url && fact.table && !linked.has(fact.table)) {
          linked.add(fact.table);
          const link = document.createElement("a");
          link.href = fact.url;
          link.target = "_blank";
          link.rel = "noopener";
          const title = fact.title ? ", " + fact.title : "";
          link.textContent = "Tabel " + fact.table + title + " på Statistikbanken";
          facts.append(advancedOnly(link));
        }
      }
    }
    facts.append(drawGrid(data.grid));
    if (advancedView(data) && data.facts && data.facts.length && data.csv) {
      const row = el("div", null, "exports");
      row.append(advancedOnly(csvButton(data.csv)));
      row.append(advancedOnly(fileButton("Hent Excel", "xlsx", data.facts)));
      row.append(advancedOnly(fileButton("Hent JSON", "json", data.facts)));
      if (chartSvg) row.append(advancedOnly(pngButton(chartSvg)));
      facts.append(row);
    }
    out.append(facts);
  }
  if (mode === "advanced") out.append(advancedOnly(toolCalls(data)));
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
    last = data;
    render(data);
  } catch (error) {
    last = { error: "Kunne ikke hente et svar." };
    render(last);
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
        *,
        max_tokens: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
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
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
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
            raise LLMError(LLM_TIMEOUT_TEXT) from exc
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
    axis: dict[str, Any] = field(default_factory=dict)
    csv: str = ""
    grid: dict[str, Any] = field(default_factory=dict)
    show_numbers: bool = True


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
            if call.name == "hent_data" and isinstance(call.arguments, dict):
                call.arguments = apply_age_intent(question, call.arguments)
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
                "title": payload.get("title") if isinstance(payload.get("title"), str) else "",
                "unpublished": row.get("unpublished") is True,
                "url": _table_url(str(payload.get("table") or "")),
            }
        )
    return facts


def _table_url(table: str) -> str:
    if table.isalnum():
        return f"https://www.statistikbanken.dk/{table}"
    return ""


_TOTAL_LABELS = frozenset({"i alt", "total", "tot", "alle"})
_MONTHS = (
    "januar",
    "februar",
    "marts",
    "april",
    "maj",
    "juni",
    "juli",
    "august",
    "september",
    "oktober",
    "november",
    "december",
)
_QUARTER_RE = re.compile(r"((?:19|20)\d{2})K([1-4])\Z", re.IGNORECASE)
_MONTH_RE = re.compile(r"((?:19|20)\d{2})M(0[1-9]|1[0-2])\Z", re.IGNORECASE)
_HALF_RE = re.compile(r"((?:19|20)\d{2})H([12])\Z", re.IGNORECASE)
_YEAR_RE = re.compile(r"(?:19|20)\d{2}\Z")


def period_label(code: str) -> str:
    """Friendly Danish period. Unknown codes are returned unchanged."""
    text = str(code).strip()
    quarter = _QUARTER_RE.fullmatch(text)
    if quarter:
        return f"{quarter.group(1)} {quarter.group(2)}. kvt."
    month = _MONTH_RE.fullmatch(text)
    if month:
        return f"{_MONTHS[int(month.group(2)) - 1]} {month.group(1)}"
    half = _HALF_RE.fullmatch(text)
    if half:
        return f"{half.group(1)} {half.group(2)}. halvår"
    if _YEAR_RE.fullmatch(text):
        return text
    return text


def _period_phrases(code: str) -> list[str]:
    """Writings of one fetched period whose digits are not separate figures."""
    text = str(code).strip()
    label = period_label(text)
    phrases = [label] if label != text else []
    if label.endswith("."):
        phrases.append(label[:-1])
    quarter = _QUARTER_RE.fullmatch(text)
    if quarter:
        phrases.append(f"{quarter.group(2)}. kvartal {quarter.group(1)}")
    half = _HALF_RE.fullmatch(text)
    if half:
        phrases.append(f"{half.group(2)}. halvår {half.group(1)}")
    return phrases


def display_label(detail: str, table: str = "") -> str:
    """'Rudersdal, 80+' rather than 'OMRÅDE: Rudersdal, KØN: I alt, ALDER: 80+'."""
    if not detail:
        return table
    values = []
    for part in detail.split(", "):
        _name, sep, value = part.partition(": ")
        text = value if sep else part
        if text and text.casefold() not in _TOTAL_LABELS:
            values.append(text)
    return ", ".join(values) or table


def facts_to_grid(facts: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """One row per period, one column per series. Every period is kept."""
    keys: list[tuple[str, str, str]] = []
    by_period: dict[str, dict[tuple[str, str, str], str]] = {}
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        period = str(fact.get("period") or "")
        if not period:
            continue
        key = (
            str(fact.get("table") or ""),
            str(fact.get("detail") or ""),
            str(fact.get("unit") or ""),
        )
        if key not in keys:
            keys.append(key)
        by_period.setdefault(period, {})[key] = _grid_cell(fact)
    if not keys:
        return {"columns": [], "rows": []}
    labels = [display_label(detail, table) for table, detail, _unit in keys]
    if len(set(labels)) != len(labels):
        labels = [
            f"{label} ({table})" if table else label
            for label, (table, _detail, _unit) in zip(labels, keys, strict=True)
        ]
    return {
        "columns": ["Periode", *labels],
        "rows": [
            [period_label(period), *[by_period[period].get(key, "–") for key in keys]]
            for period in sorted(by_period)
        ],
    }


def _grid_cell(fact: dict[str, Any]) -> str:
    if fact.get("unpublished"):
        return "–"
    value = fact.get("value")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "–" if value is None else str(value)
    return _da_number(value)


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
                "label": display_label(detail, table),
                "table": table,
                "unit": unit,
                "points": [
                    {
                        "period": period,
                        "label": period_label(period),
                        "value": points[period],
                    }
                    for period in sorted(points)
                ],
            }
        )
    return series


def chart_axis(series: list[dict[str, Any]]) -> dict[str, Any]:
    """Nice y-axis covering every charted value, with about five ticks."""
    values: list[float] = []
    for item in series:
        for point in item.get("points") or []:
            value = point.get("value")
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            values.append(float(value))
    if not values:
        return {}
    low, high, ticks = nice_axis(min(values), max(values))
    return {"min": low, "max": high, "ticks": ticks}


def nice_axis(
    low: float, high: float, count: int = 5
) -> tuple[int | float, int | float, list[int | float]]:
    """Min, max, and about `count` ticks on a 1-2-5 grid. The range covers both ends."""
    low = float(low)
    high = float(high)
    if low > high:
        low, high = high, low
    if math.isclose(low, high):
        low -= 1
        high += 1
    span = _nice_num(high - low, round_=False)
    step = _nice_num(span / (count - 1), round_=True)
    axis_min = math.floor(low / step + 1e-10) * step
    axis_max = math.ceil(high / step - 1e-10) * step
    half = step / 2
    tightened_min = axis_min + half
    tightened_max = axis_max - half
    if tightened_min <= low and tightened_max >= high and tightened_max - tightened_min >= step:
        axis_min, axis_max = tightened_min, tightened_max
    ticks: list[int | float] = []
    value = axis_min
    for _ in range(24):
        if value > axis_max + step * 1e-6:
            break
        ticks.append(_plain_number(value))
        value += step
    return _plain_number(axis_min), _plain_number(axis_max), ticks


def _nice_num(value: float, *, round_: bool) -> float:
    """Heckbert's nice number: 1, 2, or 5 times a power of ten."""
    if value <= 0:
        return 1
    exp = math.floor(math.log10(value))
    frac = value / 10**exp
    if round_:
        if frac < 1.5:
            nice = 1
        elif frac < 3:
            nice = 2
        elif frac < 7:
            nice = 5
        else:
            nice = 10
    elif frac <= 1:
        nice = 1
    elif frac <= 2:
        nice = 2
    elif frac <= 5:
        nice = 5
    else:
        nice = 10
    return nice * 10**exp


def _plain_number(value: float) -> int | float:
    nearest = round(value)
    if math.isclose(value, nearest, abs_tol=1e-8):
        return int(nearest)
    return round(value, 10)


_EXPORT_HEADER = ["tabel", "periode", "dimensioner", "værdi", "enhed"]


def _dimension_label(fact: dict[str, Any]) -> str:
    """Same text as the table column: 'Rudersdal, 80+', not 'OMRÅDE: Rudersdal, ALDER: 80+'."""
    return display_label(str(fact.get("detail") or ""), str(fact.get("table") or ""))


def _export_rows(facts: list[dict[str, Any]]) -> tuple[list[str], list[list[str]]]:
    rows = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        rows.append(
            [
                str(fact.get("table") or ""),
                period_label(str(fact.get("period") or "")),
                _dimension_label(fact),
                _csv_number(fact.get("value")),
                str(fact.get("unit") or ""),
            ]
        )
    return list(_EXPORT_HEADER), rows


def facts_to_csv(facts: list[dict[str, Any]]) -> str:
    """UTF-8 CSV with BOM, semicolon separator, and a decimal comma."""
    header, rows = _export_rows(facts)
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=";", lineterminator="\r\n")
    writer.writerow(header)
    writer.writerows(rows)
    return "\ufeff" + buffer.getvalue()


def facts_to_json(facts: list[dict[str, Any]]) -> str:
    """CSV columns. periode stays the raw code; dimensioner is the table label."""
    rows = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        raw = str(fact.get("period") or "")
        detail = str(fact.get("detail") or "")
        rows.append(
            {
                "tabel": str(fact.get("table") or ""),
                "periode": raw,
                "periode_label": period_label(raw),
                "dimensioner": _dimension_label(fact),
                "dimensioner_rå": detail,
                "værdi": _csv_number(fact.get("value")),
                "enhed": str(fact.get("unit") or ""),
            }
        )
    return json.dumps(rows, ensure_ascii=False)


def facts_to_xlsx(facts: list[dict[str, Any]]) -> bytes:
    """The same columns and cell text as the CSV, as a workbook Excel can open."""
    header, rows = _export_rows(facts)
    sheet_rows = []
    for index, row in enumerate([header, *rows], start=1):
        cells = []
        for col, value in enumerate(row):
            ref = f"{_xlsx_col(col)}{index}"
            cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{_xml_text(value)}</t></is></c>')
        sheet_rows.append(f'<row r="{index}">{"".join(cells)}</row>')
    sheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(sheet_rows)}</sheetData></worksheet>"
    )
    parts = {
        "[Content_Types].xml": _XLSX_CONTENT_TYPES,
        "_rels/.rels": _XLSX_ROOT_RELS,
        "xl/workbook.xml": _XLSX_WORKBOOK,
        "xl/_rels/workbook.xml.rels": _XLSX_WORKBOOK_RELS,
        "xl/styles.xml": _XLSX_STYLES,
        "xl/worksheets/sheet1.xml": sheet,
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in parts.items():
            archive.writestr(name, body)
    return buffer.getvalue()


def _xlsx_col(index: int) -> str:
    letters = ""
    number = index + 1
    while number:
        number, rem = divmod(number - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _xml_text(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _xlsx_xml(*lines: str) -> str:
    return "".join(lines)


_XLSX_CONTENT_TYPES = _xlsx_xml(
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
    '<Default Extension="rels" ContentType=',
    '"application/vnd.openxmlformats-package.relationships+xml"/>',
    '<Default Extension="xml" ContentType="application/xml"/>',
    '<Override PartName="/xl/workbook.xml" ContentType=',
    '"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
    '<Override PartName="/xl/worksheets/sheet1.xml" ContentType=',
    '"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>',
    '<Override PartName="/xl/styles.xml" ContentType=',
    '"application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    "</Types>",
)

_XLSX_ROOT_RELS = _xlsx_xml(
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    '<Relationship Id="rId1" Type=',
    '"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"',
    ' Target="xl/workbook.xml"/>',
    "</Relationships>",
)

_XLSX_WORKBOOK = _xlsx_xml(
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
    '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"',
    ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
    '<sheets><sheet name="Tal" sheetId="1" r:id="rId1"/></sheets>',
    "</workbook>",
)

_XLSX_WORKBOOK_RELS = _xlsx_xml(
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    '<Relationship Id="rId1" Type=',
    '"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"',
    ' Target="worksheets/sheet1.xml"/>',
    '<Relationship Id="rId2" Type=',
    '"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles"',
    ' Target="styles.xml"/>',
    "</Relationships>",
)

_XLSX_STYLES = _xlsx_xml(
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
    '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
    '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>',
    '<fills count="2"><fill><patternFill patternType="none"/></fill>',
    '<fill><patternFill patternType="gray125"/></fill></fills>',
    '<borders count="1"><border>',
    "<left/><right/><top/><bottom/><diagonal/></border></borders>",
    '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>',
    '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>',
    "</styleSheet>",
)


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
    # A figure in the prose is kept when it is in the question, a hent_data
    # value, a period or row label from those rows, the fetched table title,
    # or a variable value on this turn's hent_data arguments.
    refusal = _relevance_refusal(question, facts)
    if refusal:
        # The rows are real, but they are not the question. Do not chart them.
        return TurnResult(
            answer=refusal, facts=[], calls=calls, chart=[], csv="", show_numbers=False
        )
    kept = facts
    chart = chart_series(kept)
    return TurnResult(
        answer=guard_answer(question, answer, kept, calls),
        facts=kept,
        calls=calls,
        chart=chart,
        axis=chart_axis(chart),
        csv=facts_to_csv(kept) if kept else "",
        grid=facts_to_grid(kept),
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
# The year in 2015K1, 2015H1, or a bare period such as 2024. The quarter digit is not a figure.
_PERIOD_YEAR_RE = re.compile(
    r"\b((?:19|20)\d{2})(?:K[1-4]|Q[1-4]|H[12]|M(?:0[1-9]|1[0-2]))?\b",
    re.IGNORECASE,
)
_TITLE_RE = re.compile(r'"title"\s*:\s*"((?:\\.|[^"\\])*)"')
# A figure stands alone. Digits inside FOLK1A or INDKP106 are not figures.
# A space is a Danish thousands separator, as in 5 045.
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])("
    r"\d{1,3}(?:\.\d{3})+(?:,\d+)?"
    r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:,\d+)?"
    r"|\d+(?:[.,]\d+)?"
    r")(?![A-Za-z0-9_])"
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
            "Write 5.045 and 2015 1. kvt."
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
        "Do not open with the table id or the table title. "
        "Use only numbers that appear in hent_data rows from this question. "
        "If a tool call failed, say so and do not invent figures."
    )


def guard_answer(
    question: str,
    answer: str,
    facts: list[dict[str, Any]],
    calls: list[dict[str, str]] | None = None,
) -> str:
    """Drop prose that states a number this turn did not fetch.

    Allowed figures are the question, a hent_data value, a period or row label
    from those rows, the fetched table title, and a variable value on this
    turn's hent_data arguments. A question that asks for one
    figure is answered with the page's own line for the latest period. The
    model's sentence is not used, so a long series or an invented total cannot
    become the answer.
    """
    if not facts:
        if _is_page_message(answer):
            return answer
        if _log_unverified(answer, question, facts, calls):
            return NO_FETCHED_NUMBERS
        return NO_DATA_ANSWER
    refusal = _relevance_refusal(question, facts)
    if refusal:
        return refusal
    if not _asks_for_trend(question):
        lines = _latest_lines(facts)
        if lines:
            return "\n".join(lines)
    if _log_unverified(answer, question, facts, calls):
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


def _is_page_message(answer: str) -> bool:
    if answer in {NO_FETCHED_NUMBERS, UNVERIFIED_NUMBERS, NO_DATA_ANSWER}:
        return True
    return (
        answer.startswith("Jeg nåede grænsen for opslag")
        or answer.startswith("Modellen gav ikke et svar")
        or answer.startswith("Jeg fandt ingen tal at svare ud fra")
    )


def _summary_line(fact: dict[str, Any]) -> str:
    value = _da_number(fact["value"])
    bits = [
        str(fact.get("table") or ""),
        period_label(str(fact.get("period") or "")),
        str(fact.get("unit") or ""),
    ]
    detail = display_label(str(fact.get("detail") or ""))
    if detail:
        bits.append(detail)
    label = ", ".join(bit for bit in bits if bit)
    return f"{value} ({label})" if label else value


def apply_age_intent(question: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Replace the model's age codes with the age the question asked for."""
    intent = _age_intent(question)
    if intent is None:
        return arguments
    if not any(_has_alder(arguments.get(key)) for key in ("values", "sum_from", "sum_range")):
        return arguments
    kind, low, high = intent
    updated = dict(arguments)
    values = _without_alder(updated.get("values"))
    sum_from = _without_alder(updated.get("sum_from"))
    sum_range = _without_alder(updated.get("sum_range"))
    if kind == "from":
        sum_from = dict(sum_from or {})
        sum_from["ALDER"] = str(low)
    else:
        sum_range = dict(sum_range or {})
        sum_range["ALDER"] = [str(low), str(high)]
    _put_map(updated, "values", values)
    _put_map(updated, "sum_from", sum_from)
    _put_map(updated, "sum_range", sum_range)
    return updated


def _age_intent(question: str) -> tuple[str, int, int | None] | None:
    text = question.casefold().replace("–", "-")
    between = re.search(r"mellem\s+(\d{1,3})\s+og\s+(\d{1,3})", text)
    if between and _age_span(int(between.group(1)), int(between.group(2))):
        low, high = int(between.group(1)), int(between.group(2))
        return ("range", low, high)
    span = re.search(r"(?<!\d)(\d{1,3})\s*-\s*(\d{1,3})(?!\d)", text)
    if span and _age_span(int(span.group(1)), int(span.group(2))):
        return ("range", int(span.group(1)), int(span.group(2)))
    if re.search(r"\bbørn\b", text):
        return ("range", 0, 17)
    opened = re.search(
        r"(\d{1,3})\s*år\s+og\s+derover|over\s+(\d{1,3})(?!\d)|>=\s*(\d{1,3})(?!\d)|(\d{1,3})\s*\+",
        text,
    )
    if opened is None:
        return None
    number = next(int(group) for group in opened.groups() if group)
    if number > 125:
        return None
    return ("from", number, None)


def _age_span(low: int, high: int) -> bool:
    return low <= high <= 125


def _has_alder(mapping: Any) -> bool:
    return isinstance(mapping, dict) and any(str(key).casefold() == "alder" for key in mapping)


def _without_alder(mapping: Any) -> dict[str, Any] | None:
    if not isinstance(mapping, dict):
        return None
    return {key: value for key, value in mapping.items() if str(key).casefold() != "alder"}


def _put_map(arguments: dict[str, Any], key: str, mapping: dict[str, Any] | None) -> None:
    if mapping:
        arguments[key] = mapping
    else:
        arguments.pop(key, None)


def _relevance_refusal(question: str, facts: list[dict[str, Any]]) -> str:
    if not facts:
        return ""
    place = _place_mismatch(question, facts)
    if place:
        return place
    return _age_mismatch(question, facts)


def _age_mismatch(question: str, facts: list[dict[str, Any]]) -> str:
    intent = _age_intent(question)
    if intent is None:
        return ""
    labels: list[str] = []
    for fact in facts:
        label = _detail_field(str(fact.get("detail") or ""), "ALDER")
        if label and label not in labels:
            labels.append(label)
    if not labels or all(_age_label_matches(intent, label) for label in labels):
        return ""
    kind, low, high = intent
    wanted = f"{low}+" if kind == "from" else f"{low}-{high}"
    return f"Jeg fik hentet tal for alder {' og '.join(labels)}, ikke for {wanted}."


def _age_label_matches(intent: tuple[str, int, int | None], label: str) -> bool:
    kind, low, high = intent
    folded = label.casefold().replace(" ", "").replace("–", "-")
    if kind == "from":
        return folded in {f"{low}+", f"{low}årogderover"}
    return folded == f"{low}-{high}"


def _detail_field(detail: str, field: str) -> str:
    match = re.search(rf"{field}:\s*(.*?)(?:,\s*[^,]+:|$)", detail)
    return match.group(1).strip() if match else ""


def _place_mismatch(question: str, facts: list[dict[str, Any]]) -> str:
    """Refuse a fetched place when the question named a different one."""
    tables = {str(fact.get("table") or "") for fact in facts if fact.get("table")}
    if len(tables) != 1:
        return ""
    known = _place_names(next(iter(tables)))
    asked = [name for name in known if _name_in_text(name, question)]
    if not asked:
        return ""
    got = _places_in_facts(facts)
    missing = [name for name in asked if not any(_same_place(name, item) for item in got)]
    if not missing:
        return ""
    missing_text = " og ".join(missing)
    if not got:
        return f"Jeg fik ikke hentet tal for {missing_text}."
    return f"Jeg fik hentet tal for {' og '.join(got)}, ikke for {missing_text}."


def _place_names(table: str) -> list[str]:
    from statistikbanken_mcp.server import get_client
    from statistikbanken_mcp.statbank import StatbankError

    try:
        return get_client().place_names(table)
    except (StatbankError, OSError):
        return []


def _places_in_facts(facts: list[dict[str, Any]]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for fact in facts:
        for name in re.findall(r"OMRÅDE:\s*(.*?)(?:,\s*[^,]+:|$)", str(fact.get("detail") or "")):
            label = name.strip()
            key = _fold_place(label)
            if label and key not in seen:
                seen.add(key)
                found.append(label)
    return found


def _name_in_text(name: str, text: str) -> bool:
    folded = _fold_place(name)
    if len(folded) < 4:
        return False
    return (
        re.search(rf"(?<![0-9a-z]){re.escape(folded)}(?![0-9a-z])", _fold_place(text)) is not None
    )


def _same_place(left: str, right: str) -> bool:
    return _fold_place(left) == _fold_place(right)


def _fold_place(text: str) -> str:
    return text.casefold().replace("æ", "ae").replace("ø", "o").replace("å", "aa")


def _da_number(value: int | float) -> str:
    if isinstance(value, float) and not value.is_integer():
        sign = "-" if value < 0 else ""
        whole, frac = f"{abs(value):.10f}".rstrip("0").rstrip(".").split(".")
        grouped = f"{int(whole):,}".replace(",", ".")
        return f"{sign}{grouped},{frac}"
    number = int(value)
    sign = "-" if number < 0 else ""
    return sign + f"{abs(number):,}".replace(",", ".")


def _log_unverified(
    answer: str,
    question: str,
    facts: list[dict[str, Any]],
    calls: list[dict[str, str]] | None,
) -> list[float]:
    flagged = _unverified_numbers(answer, question, facts, calls)
    if flagged:
        logger.info("Suppressed prose; flagged numbers: %s", _flagged_text(flagged))
    return flagged


def _flagged_text(numbers: list[float]) -> str:
    shown = sorted({int(number) if float(number).is_integer() else number for number in numbers})
    return ", ".join(str(number) for number in shown)


def _unverified_numbers(
    answer: str,
    question: str,
    facts: list[dict[str, Any]],
    calls: list[dict[str, str]] | None = None,
) -> list[float]:
    titles = _fetched_titles(calls)
    allowed = _numbers_in_context(question, facts, titles)
    allowed.update(_argument_figures(calls))
    fetched = {str(fact.get("period") or "") for fact in facts}
    scanned = _mask_friendly_periods(answer, fetched)
    flagged = [number for number in _numbers_in(scanned) if not _near(number, allowed)]
    known_periods = _period_tokens(question)
    for fact in facts:
        known_periods.update(_period_tokens(str(fact.get("period") or "")))
    for title in titles:
        known_periods.update(_period_tokens(title))
    for token in _period_tokens(answer):
        if token not in known_periods:
            flagged.append(float(token[:4]))
    return flagged


def _numbers_in_context(
    question: str,
    facts: list[dict[str, Any]],
    titles: list[str],
) -> set[float]:
    """Question, hent_data values, period and row labels, and the table title.

    A number that appears only in tabelinfo or find_tabel is not allowed.
    """
    found = _label_figures(question)
    for fact in facts:
        value = fact.get("value")
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        found.add(float(value))
        found.update(_label_figures(str(fact.get("period") or "")))
        found.update(_label_figures(str(fact.get("detail") or "")))
    for title in titles:
        found.update(_label_figures(title))
    return found


def _argument_figures(calls: list[dict[str, str]] | None) -> set[float]:
    """Numbers the model passed as hent_data variable values this turn."""
    found: set[float] = set()
    for call in calls or []:
        if call.get("name") != "hent_data":
            continue
        try:
            payload = json.loads(str(call.get("arguments") or ""))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        for key in ("values", "sum_from", "sum_range"):
            found.update(_value_figures(payload.get(key)))
    return found


def _value_figures(value: Any) -> set[float]:
    if isinstance(value, bool) or value is None:
        return set()
    if isinstance(value, int | float):
        return {float(value)}
    if isinstance(value, str):
        return _label_figures(value)
    if isinstance(value, dict):
        found: set[float] = set()
        for item in value.values():
            found.update(_value_figures(item))
        return found
    if isinstance(value, list):
        found = set()
        for item in value:
            found.update(_value_figures(item))
        return found
    return set()


def _label_figures(text: str) -> set[float]:
    return set(_numbers_in(text)) | _period_years(text)


def _period_years(text: str) -> set[float]:
    return {float(year) for year in _PERIOD_YEAR_RE.findall(text)}


def _period_tokens(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _PERIOD_RE.finditer(text)}


def _fetched_titles(calls: list[dict[str, str]] | None) -> list[str]:
    titles: list[str] = []
    for call in calls or []:
        if call.get("name") != "hent_data":
            continue
        title = _title_in(str(call.get("result") or ""))
        if title:
            titles.append(title)
    return titles


def _title_in(result: str) -> str:
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        title = payload.get("title")
        if isinstance(title, str):
            return title
    match = _TITLE_RE.search(result)
    if not match:
        return ""
    try:
        return json.loads(f'"{match.group(1)}"')
    except json.JSONDecodeError:
        return ""


def _near(number: float, allowed: set[float]) -> bool:
    return any(abs(number - item) < 1e-6 for item in allowed)


def _numbers_in(text: str) -> list[float]:
    masked = _mask_periods(text)
    return [_parse_stat_number(match.group(1)) for match in _NUMBER_RE.finditer(masked)]


def _mask_periods(text: str) -> str:
    return _PERIOD_RE.sub(lambda match: " " * len(match.group(0)), text)


def _mask_friendly_periods(text: str, periods: set[str]) -> str:
    """Hide digits that only name a fetched quarter, month, or half-year."""
    phrases: list[str] = []
    for code in periods:
        phrases.extend(_period_phrases(code))
    for phrase in sorted(set(phrases), key=len, reverse=True):
        text = re.sub(re.escape(phrase), " " * len(phrase), text, flags=re.IGNORECASE)
    return text


def _parse_stat_number(token: str) -> float:
    token = token.replace(" ", "").replace("\u00a0", "").replace("\u202f", "")
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


def _unwrap(exc: BaseException) -> list[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        leaves: list[BaseException] = []
        for child in exc.exceptions:
            leaves.extend(_unwrap(child))
        return leaves
    return [exc]


def _failure_response(exc: BaseException) -> tuple[int, str]:
    for item in _unwrap(exc):
        if isinstance(item, LLMError):
            text = str(item)
            return (504 if "ikke i tide" in text else 502), text
    if any(_is_timeout(item) for item in _unwrap(exc)):
        return 504, "Statistikbanken svarede ikke i tide."
    return 502, "Kunne ikke hente et svar."


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
    def __init__(self, daily_cap: int, per_minute: int, state_file: str = "") -> None:
        self.daily_cap = daily_cap
        self.per_minute = per_minute
        self.state_file = state_file
        self._lock = threading.Lock()
        self._day = date.today()
        self._count = 0
        self._hits: dict[str, deque[float]] = {}
        loaded = _load_daily_count(state_file)
        if loaded is not None and loaded[0] == self._day:
            self._count = loaded[1]

    def allow(self, ip: str) -> str | None:
        if self.daily_cap <= 0:
            return "stopped"
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
            _save_daily_count(self.state_file, self._day, self._count)
            return None


def _load_daily_count(path: str) -> tuple[date, int] | None:
    """Date and count from STATBANK_STATE_FILE. Addresses are not stored."""
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        day = date.fromisoformat(str(payload["day"]))
        count = payload["count"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return None
    return day, count


def _save_daily_count(path: str, day: date, count: int) -> None:
    if not path:
        return
    temporary = f"{path}.tmp"
    body = json.dumps({"day": day.isoformat(), "count": count})
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(body)
        os.replace(temporary, path)
    except OSError:
        logger.warning("Could not save the daily counter to %s", path)


_LIMIT_TEXT = {
    "rate": "For mange spørgsmål fra denne adresse. Prøv igen om et minut.",
    "daily": "Dagens grænse for spørgsmål er nået.",
    "stopped": "Demoen er midlertidigt lukket. Prøv igen senere.",
}
_BUSY_TEXT = "Der er travlt lige nu. Prøv igen."


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
    public: bool = False
    proxy_secret: str = ""
    frame_ancestors: str = "'self'"
    max_tokens: int | None = None
    max_concurrent: int | None = None
    state_file: str = ""

    def __repr__(self) -> str:
        return (
            f"Settings(llm={self.llm!r}, model={self.model!r}, base_url={self.base_url!r}, "
            f"host={self.host!r}, port={self.port}, public={self.public})"
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


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _listen_port(cli_port: int | None) -> int:
    """Platform PORT wins over STATBANK_PORT. A --port flag wins over both."""
    if cli_port is not None:
        return cli_port
    platform = os.environ.get("PORT", "").strip()
    if platform:
        return int(platform)
    return int(os.environ.get("STATBANK_PORT", "8080"))


def _frame_ancestors() -> str:
    raw = os.environ.get("STATBANK_FRAME_ANCESTORS", "'self'")
    cleaned = raw.replace("\r", "").replace("\n", "").strip()
    return cleaned or "'self'"


def _proxy_secret() -> str:
    raw = os.environ.get("STATBANK_PROXY_SECRET", "")
    return raw.replace("\r", "").replace("\n", "").strip()


def _state_file() -> str:
    raw = os.environ.get("STATBANK_STATE_FILE", "")
    return raw.replace("\r", "").replace("\n", "").strip()


def _xai_model(cli_model: str | None, *, public: bool) -> str:
    """Model sent to xAI. STATBANK_XAI_MODEL is the deploy-time choice."""
    if cli_model:
        return cli_model
    chosen = os.environ.get("STATBANK_XAI_MODEL", "").strip()
    if chosen:
        return chosen
    if not public:
        legacy = os.environ.get("STATBANK_MODEL", "").strip()
        if legacy:
            return legacy
    return DEFAULT_XAI_MODEL


def settings_from_env(argv: list[str] | None = None) -> Settings:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Local Statistikbanken web demo")
    parser.add_argument("--llm", choices=("ollama", "xai"))
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args(argv)
    public = _env_flag("STATBANK_PUBLIC")
    if public:
        # Ollama is not available on the public demo. A copied local .env is ignored.
        llm = "xai"
        base_url = XAI_BASE_URL
        model = _xai_model(args.model, public=True)
        api_key = os.environ.get("XAI_API_KEY", "").strip()
        daily_default, rate_default, chars_default, tools_default = 40, 3, 300, 6
    else:
        llm = args.llm or os.environ.get("STATBANK_LLM", "ollama")
        if llm == "xai":
            base_url = args.base_url or os.environ.get("STATBANK_BASE_URL", XAI_BASE_URL)
            model = _xai_model(args.model, public=False)
            api_key = os.environ.get("XAI_API_KEY", "")
        else:
            llm = "ollama"
            base_url = args.base_url or os.environ.get("STATBANK_BASE_URL", OLLAMA_BASE_URL)
            model = args.model or os.environ.get("STATBANK_MODEL", DEFAULT_OLLAMA_MODEL)
            api_key = os.environ.get("STATBANK_API_KEY", "ollama")
        daily_default, rate_default, chars_default, tools_default = 200, 30, 1000, 8
    return Settings(
        llm=llm,
        base_url=base_url,
        model=model,
        api_key=api_key,
        host=args.host or os.environ.get("STATBANK_HOST", "127.0.0.1"),
        port=_listen_port(args.port),
        daily_cap=_env_int("STATBANK_DAILY_CAP", daily_default),
        per_minute=_env_int("STATBANK_RATE_PER_MINUTE", rate_default),
        max_question_chars=_env_int("STATBANK_MAX_QUESTION_CHARS", chars_default),
        max_tool_calls=_env_int("STATBANK_MAX_TOOL_CALLS", tools_default),
        llm_timeout=float(os.environ.get("STATBANK_LLM_TIMEOUT", str(DEFAULT_LLM_TIMEOUT))),
        tool_timeout=float(os.environ.get("STATBANK_TOOL_TIMEOUT", str(DEFAULT_TOOL_TIMEOUT))),
        public=public,
        proxy_secret=_proxy_secret(),
        frame_ancestors=_frame_ancestors(),
        max_tokens=PUBLIC_MAX_TOKENS if llm == "xai" else None,
        max_concurrent=PUBLIC_MAX_CONCURRENT if public else None,
        state_file=_state_file(),
    )


def _is_secret_env(name: str) -> bool:
    upper = name.upper()
    if upper in {"XAI_API_KEY", "STATBANK_API_KEY", "STATBANK_PROXY_SECRET"}:
        return True
    return upper.endswith(("_API_KEY", "_SECRET", "_TOKEN", "_PASSWORD"))


def _child_env(tool_timeout: float | None) -> dict[str, str]:
    """Environment for the MCP subprocess. Secrets stay in the parent process."""
    env = {key: value for key, value in os.environ.items() if not _is_secret_env(key)}
    if tool_timeout is not None:
        env["STATBANK_HTTP_TIMEOUT"] = str(tool_timeout)
    return env


def server_command(tool_timeout: float | None = None) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "statistikbanken_mcp"],
        env=_child_env(tool_timeout),
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
            max_tokens=settings.max_tokens,
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


def check_settings(settings: Settings) -> None:
    """Refuse a public demo that would start without an xAI key."""
    if settings.public and not settings.api_key:
        raise SystemExit(
            "XAI_API_KEY is missing. Set it in the environment before starting. "
            "Do not write it into a file."
        )


def serve(settings: Settings, answer: Any | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    check_settings(settings)
    limited = Limiter(settings.daily_cap, settings.per_minute, settings.state_file)
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


def client_ip(peer: str, headers: Any, *, trust_proxy: bool) -> str:
    """Visitor address for the per-minute limit. Callers must not log it."""
    if trust_proxy:
        forwarded = _forwarded_ip(headers)
        if forwarded:
            return forwarded
    return peer


def _forwarded_ip(headers: Any) -> str:
    cf = headers.get("CF-Connecting-IP")
    if isinstance(cf, str) and cf.strip():
        return cf.strip()
    raw = headers.get("X-Forwarded-For")
    if not isinstance(raw, str):
        return ""
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    return parts[-1] if parts else ""


def _page_bytes(settings: Settings) -> bytes:
    if not settings.public:
        demo = ""
    elif settings.daily_cap <= 0:
        demo = '<p class="demo">Demoen er midlertidigt lukket.</p>'
    else:
        demo = (
            '<p class="demo">Demo: maks. '
            f"{settings.daily_cap} spørgsmål om dagen i alt. "
            "Tal fra Danmarks Statistik.</p>"
        )
    html = PAGE.replace("__MAX__", str(settings.max_question_chars)).replace("__DEMO__", demo)
    return html.encode("utf-8")


def _security_headers(settings: Settings) -> list[tuple[str, str]]:
    ancestors = settings.frame_ancestors or "'self'"
    return [
        ("Content-Security-Policy", f"frame-ancestors {ancestors}"),
        ("X-Content-Type-Options", "nosniff"),
        ("Referrer-Policy", "no-referrer"),
    ]


class _Gate:
    """How many questions may run at once. A full gate does not queue."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._active = 0
        self._lock = threading.Lock()

    def try_enter(self) -> bool:
        with self._lock:
            if self._active >= self._limit:
                return False
            self._active += 1
            return True

    def leave(self) -> None:
        with self._lock:
            self._active -= 1


def _handler(settings: Settings, limiter: Limiter, answer: Any) -> type[BaseHTTPRequestHandler]:
    gate = _Gate(settings.max_concurrent) if settings.max_concurrent else None

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/healthz":
                self._send(200, b"ok\n", "text/plain; charset=utf-8")
                return
            if not self._proxy_ok():
                return
            if path != "/":
                self._send(404, b"Not found\n", "text/plain; charset=utf-8")
                return
            self._send(200, _page_bytes(settings), "text/html; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802
            if not self._proxy_ok():
                return
            path = self.path.split("?", 1)[0]
            length = int(self.headers.get("Content-Length", "0") or 0)
            limit = EXPORT_MAX_BYTES if path == "/api/export" else MAX_BODY_BYTES
            if length > limit:
                self._json(400, {"error": "Forespørgslen er for lang."})
                return
            raw = self.rfile.read(length)
            if path == "/api/export":
                self._export(raw)
                return
            if path != "/api/ask":
                self._json(404, {"error": "Ukendt adresse."})
                return
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
            held = False
            if gate is not None:
                if not gate.try_enter():
                    self._json(503, {"error": _BUSY_TEXT})
                    return
                held = True
            try:
                self._answer(question)
            finally:
                if held and gate is not None:
                    gate.leave()

        def _answer(self, question: str) -> None:
            ip = client_ip(
                self.client_address[0],
                self.headers,
                trust_proxy=self._secret_matches(),
            )
            limited = limiter.allow(ip)
            if limited:
                status = 503 if limited == "stopped" else 429
                self._json(status, {"error": _LIMIT_TEXT[limited]})
                return
            try:
                result = answer(question)
            except Exception as exc:
                leaves = _unwrap(exc)
                names = ", ".join(type(item).__name__ for item in leaves)
                sys.stderr.write(f"Question failed: {type(exc).__name__} ({names}): {exc}\n")
                traceback.print_exc(file=sys.stderr)
                status, text = _failure_response(exc)
                self._json(status, {"error": text})
                return
            self._json(
                200,
                {
                    "answer": result.answer,
                    "facts": result.facts,
                    "calls": result.calls,
                    "chart": result.chart,
                    "axis": result.axis,
                    "csv": result.csv,
                    "grid": result.grid,
                    "show_numbers": result.show_numbers,
                },
            )

        def _secret_matches(self) -> bool:
            secret = settings.proxy_secret
            if not secret:
                return False
            provided = self.headers.get("X-Proxy-Secret", "")
            return hmac.compare_digest(provided.encode(), secret.encode())

        def _proxy_ok(self) -> bool:
            if not settings.proxy_secret or self._secret_matches():
                return True
            self._send(403, b"Forbidden\n", "text/plain; charset=utf-8")
            return False

        def _export(self, raw: bytes) -> None:
            """Turn already-fetched facts into a file. No model call, no new data call."""
            try:
                payload = json.loads(raw.decode("utf-8"))
                fmt = str(payload.get("format", ""))
                facts = payload.get("facts")
            except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                self._json(400, {"error": "Kunne ikke læse tallene."})
                return
            if not isinstance(facts, list) or any(not isinstance(item, dict) for item in facts):
                self._json(400, {"error": "Kunne ikke læse tallene."})
                return
            if fmt == "json":
                body = facts_to_json(facts).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
                return
            if fmt == "xlsx":
                self._send(
                    200,
                    facts_to_xlsx(facts),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                return
            self._json(400, {"error": "Ukendt format."})

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for name, value in _security_headers(settings):
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def address_string(self) -> str:  # noqa: N802
            return "-"

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write(f"{fmt % args}\n")  # noqa: UP031

    return Handler


def main(argv: list[str] | None = None) -> None:
    serve(settings_from_env(argv))


if __name__ == "__main__":
    main()
