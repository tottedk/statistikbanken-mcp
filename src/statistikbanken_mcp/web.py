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
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import httpx
from mcp import Client, StdioServerParameters

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_OLLAMA_MODEL = "gemma4:26b-a4b-it-qat"
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
Leave out a variable the question does not name. It is set to the total (I alt or Hele landet).
Leave out Tid for one figure. Omitted time is the latest published period.
For a change over time, fetch every period in one hent_data call.
Pass Tid as one range in values, for example >=2015K1.
Each period comes back as its own row. Do not put Tid in sum_range.
Do not sum periods together with several places or an age range.
Quote values from the tool results. A difference, a sum, or a percentage change
is allowed when it is computed from those values.
For each series, quote the first value and the last value.
If a note says the figure is a projection (fremskrivning) or a forecast, say so.
Do not present it as an observed figure.
If a note says a year was added or averaged, repeat that.
Municipality names may be passed as labels.
One code may be a string. Several codes are a list, or one comma-separated string.
Refer to areas and categories by name, not by code.
Answer in the same language as the question.
A Danish question gets a Danish answer, even if the tool text is English.
Do not answer in English when the question is Danish.
Write Danish numbers with a dot as the thousands separator: 5.045, never 5,045.
Write periods like 1. kvartal 2015 or 2015 1. kvt., marts 2015, or 2015 1. halvår.
A year stays 2015. Do not write 2015K1 or 2026Q3.
If find_tabel returned a table, call hent_data, or name the concrete blocker.
Do not say there are no figures.
A question with no metric asks which figure is wanted. Do not choose a population number.
When only one side of a comparison is in Statistikbanken,
give that figure and say the other is not covered.
steget, faldet, or ændret asks for the change: use the change unit,
or the first and last value of one series.
Do not answer with the index level alone.
vokset, flere end, and siden ask for the change between two dates.
Quote both values, the difference, and the percentage.
Read first_period on a find_tabel hit. If the question names a year before
that period, open a hit whose first_period covers the year before you stop.
If a year is missing inside one table, say which table and name the
neighbouring periods. Do not say the figure exists in no table.
Nettoledige is the net count. Bruttoledige is the gross count.
Sæsonkorrigeret is the seasonally adjusted count, not opregnede faktiske.
A whole year asked as an average is the published annual figure,
not a mean of the monthly actual series.
A named year is the Tid code for that year, such as 2025.
Do not pass an open >= range for one year.
A question about which town is largest uses byområder in BY3, folketal.
A municipality, a region, or the whole country is not that town.
Hovedstadsområdet is one urban area."""


_DANISH_MONTHS = (
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


def _danish_month(day: date, *, start: bool) -> str:
    name = _DANISH_MONTHS[day.month - 1]
    if start:
        return name[:1].upper() + name[1:]
    return name


def _unpublished_example(day: date) -> str:
    """The month just gone, and the month before it, so the example does not go stale."""
    requested = day.replace(day=1) - timedelta(days=1)
    earlier = requested.replace(day=1) - timedelta(days=1)
    return (
        f"{_danish_month(requested, start=True)} {requested.year} er ikke offentliggjort endnu; "
        f"seneste er {_danish_month(earlier, start=False)} {earlier.year}"
    )


def _previous_quarter(day: date) -> str:
    """Last finished quarter. November is K3 of the same year."""
    quarter = (day.month - 1) // 3
    if quarter == 0:
        return f"{day.year - 1}K4"
    return f"{day.year}K{quarter}"


def _years_earlier(code: str, years: int) -> str:
    return f"{int(code[:4]) - years}{code[4:]}"


def system_prompt(today: date | None = None) -> str:
    # None uses the clock in Europe/Copenhagen. Tests pass a date.
    day = today if today is not None else datetime.now(ZoneInfo("Europe/Copenhagen")).date()
    last_month = day.replace(day=1) - timedelta(days=1)
    month_code = f"{last_month.year}M{last_month.month:02d}"
    quarter_code = _previous_quarter(day)
    last_year = day.year - 1
    example = _unpublished_example(day)
    return (
        f"{SYSTEM_PROMPT}\n"
        f"Today's date is {day.isoformat()} (Europe/Copenhagen). "
        f"In hent_data, pass Tid as Statbank codes: a year is {last_year}, "
        f"a quarter is {quarter_code}, a month is {month_code}. "
        f"'i fjor' is {last_year}. 'i år' is {day.year}. "
        f"'sidste måned' is {month_code}. "
        f"These codes are the calendar meaning. "
        f"If that period is not published yet, call hent_data again with the latest "
        f"published period from the error and say so in the answer. "
        f"For example: '{example}.' "
        f"Do the same for 'i år' and 'i fjor' when that year is not published. "
        f"'seneste 10 år' is the latest published period and the same quarter or month "
        f"10 years earlier. If the latest quarter is {quarter_code}, pass "
        f"{_years_earlier(quarter_code, 10)} to {quarter_code}. "
        f"If the latest month is {month_code}, pass {_years_earlier(month_code, 10)} "
        f"to {month_code}."
    )


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
  .suggest { background: #fff; border: 1px solid #c9c2b6; border-radius: 8px;
    padding: 0.75rem 1rem; margin: 0; }
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
      placeholder="Hvor mange over 80 bor i Esbjerg?"></textarea>
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
  const marks = axis.labels || [];
  marks.forEach((mark, index) => {
    const period = mark.period;
    const last = index === marks.length - 1;
    const first = index === 0;
    const label = svgEl("text", {
      x: xOf(period),
      y: height - 12,
      "text-anchor": last ? "end" : (first ? "start" : "middle"),
      fill: "#4a453e",
      "font-size": "10"
    });
    const shown = mark.label || labelOf.get(period) || period;
    label.textContent = shown;
    const tip = svgEl("title", {});
    tip.textContent = labelOf.get(period) || shown;
    label.append(tip);
    svg.append(label);
  });
  series.forEach((item, index) => {
    const color = COLORS[index % COLORS.length];
    const points = item.points.map((point) => xOf(point.period) + "," + yOf(point.value)).join(" ");
    svg.append(svgEl("polyline", {
      points, fill: "none", stroke: color, "stroke-width": "2",
      "stroke-linejoin": "round", "stroke-linecap": "round"
    }));
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

function extraNotes(data) {
  const lines = [];
  const seen = new Set();
  let forecast = false;
  for (const fact of data.facts || []) {
    if (fact.projection) forecast = true;
    if (fact.derivation && !seen.has(fact.derivation)) {
      seen.add(fact.derivation);
      lines.push(fact.derivation);
    }
  }
  if (forecast) {
    lines.unshift("Rækker mærket fremskrivning er prognoser, ikke observerede tal.");
  }
  return lines;
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

function suggestionsBlock(data) {
  const lines = data.suggestions || [];
  if (!lines.length) return null;
  const box = el("section");
  box.append(el("h2", "Forslag"));
  for (const line of lines) box.append(el("p", line, "suggest"));
  return box;
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
  const suggestions = suggestionsBlock(data);
  if (suggestions) out.append(suggestions);

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
    for (const line of extraNotes(data)) facts.append(el("p", line, "meta"));
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
    """POST /v1/chat/completions for Ollama or xAI."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        http: httpx.Client | None = None,
        *,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
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
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
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
    suggestions: list[str] = field(default_factory=list)


async def run_turn(
    llm: ChatClient,
    runner: ToolRunner,
    question: str,
    *,
    max_tool_calls: int,
    today: date | None = None,
) -> TurnResult:
    specs = await runner.list_tools()
    tools = openai_tools(specs)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt(today)},
        {"role": "user", "content": question},
        {"role": "system", "content": _closing_instruction(question)},
    ]
    calls: list[dict[str, str]] = []
    facts: list[dict[str, Any]] = []
    ranked: str | None = None
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
            if call.name == "find_tabel" and isinstance(call.arguments, dict):
                call.arguments = apply_search_context(question, call.arguments)
            if call.name == "hent_data" and isinstance(call.arguments, dict):
                call.arguments = apply_age_intent(question, call.arguments)
                call.arguments = apply_fetch_defaults(question, call.arguments)
                if ranked and call.arguments.get("table"):
                    call.arguments["table"] = preferred_table(ranked, str(call.arguments["table"]))
            blocked = (
                net_series_block(question, call.arguments)
                if call.name == "hent_data" and isinstance(call.arguments, dict)
                else ""
            )
            result = blocked or await _invoke(runner, call)
            if (
                call.name == "hent_data"
                and isinstance(call.arguments, dict)
                and _fetch_timed_out(result)
            ):
                calls.append(_trace(call, result))
                used += 1
                call = ToolCall(
                    id=f"{call.id}:retry",
                    name=call.name,
                    arguments=_retry_fetch(call.arguments),
                    raw=call.raw,
                )
                result = await _invoke(runner, call)
            if call.name == "find_tabel":
                ranked = _top_table(result) or ranked
            used += 1
            calls.append(_trace(call, result))
            facts.extend(_facts_from(call.name, result))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": _model_result(call.name, result),
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


def _trace(call: ToolCall, result: str) -> dict[str, str]:
    arguments = call.arguments if isinstance(call.arguments, dict) else {}
    return {
        "name": call.name,
        "arguments": _short(json.dumps(arguments, ensure_ascii=False, indent=2), TRACE_LIMIT),
        "result": _short(result, TRACE_LIMIT),
    }


def _top_table(result: str) -> str | None:
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        table_id = payload[0].get("id")
        return str(table_id) if table_id else None
    return None


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
            if key not in {"value", "unpublished", "Tid", "projection", "year_total"}
            and not isinstance(value, dict)
        )
        fact: dict[str, Any] = {
            "table": payload.get("table") or "",
            "period": "" if period is None else str(period),
            "unit": payload.get("unit") or "",
            "value": row.get("value"),
            "detail": detail,
            "title": payload.get("title") if isinstance(payload.get("title"), str) else "",
            "unpublished": row.get("unpublished") is True,
            "url": _table_url(str(payload.get("table") or "")),
        }
        if row.get("projection") is True or payload.get("projection") is True:
            fact["projection"] = True
        derivation = _derivation_text(row.get("year_total"))
        if derivation:
            fact["derivation"] = derivation
        facts.append(fact)
    return facts


def _derivation_text(info: Any) -> str:
    """Danish sentence for a yearly figure built from shorter periods."""
    if not isinstance(info, dict):
        return ""
    periods = [str(period) for period in info.get("periods") or [] if str(period)]
    if not periods:
        return ""
    listed = ", ".join(periods)
    if info.get("method") == "sum":
        return f"Sum af {listed} (strøm, perioderne er lagt sammen)."
    if info.get("method") == "average":
        return f"Gennemsnit af {listed} (niveau, perioderne er gennemsnit)."
    return ""


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


def _shown_period(fact: dict[str, Any]) -> str:
    label = period_label(str(fact.get("period") or ""))
    if fact.get("projection"):
        return f"{label} (fremskrivning)"
    return label


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


_ALL_AGES_LABELS = frozenset({"alder i alt"})
_POPULATION_TABLES = ("FOLK", "BEFOLK", "FRKM", "BY")
_REGION_PREFIXES = ("region ", "landsdel ", "provins")


def display_label(detail: str, table: str = "") -> str:
    """'Esbjerg, 80+' rather than 'OMRÅDE: Esbjerg, KØN: I alt, ALDER: 80+'."""
    if not detail:
        return table
    values = []
    for part in detail.split(", "):
        _name, sep, value = part.partition(": ")
        text = value if sep else part
        if text and text.casefold() not in _TOTAL_LABELS:
            values.append(text)
    if not values:
        return table
    if table.upper().startswith(_POPULATION_TABLES) and any(
        item.casefold() in _ALL_AGES_LABELS for item in values
    ):
        shown = [
            f"{item} kommune" if _municipality_name(item) else item
            for item in values
            if item.casefold() not in _ALL_AGES_LABELS
        ]
        return ", ".join(["Befolkning", *shown, "alder i alt"])
    return ", ".join(values)


def _municipality_name(text: str) -> bool:
    """A bare area name on a population table is a municipality, not a town."""
    folded = text.casefold()
    if folded in {"hele landet", "i alt"}:
        return False
    return not folded.startswith(_REGION_PREFIXES)


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
        shown = "–" if value is None else str(value)
    else:
        shown = _da_number(value)
    if fact.get("projection") and shown != "–":
        return f"{shown} (fremskrivning)"
    return shown


def chart_series(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group numeric facts that share a table and dimensions across periods."""
    grouped: dict[tuple[str, str, str, bool], dict[str, int | float]] = {}
    order: list[tuple[str, str, str, bool]] = []
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
            fact.get("projection") is True,
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
        table, detail, unit, projection = key
        label = display_label(detail, table)
        if projection:
            label = f"{label} (fremskrivning)"
        series.append(
            {
                "label": label,
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


_MAX_X_LABELS = 6


def _chart_periods(series: list[dict[str, Any]]) -> list[str]:
    found: set[str] = set()
    periods: list[str] = []
    for item in series:
        for point in item.get("points") or []:
            period = str(point.get("period") or "")
            if period and period not in found:
                found.add(period)
                periods.append(period)
    return sorted(periods)


def _even_indexes(count: int, limit: int) -> list[int]:
    if count <= limit:
        return list(range(count))
    last = count - 1
    chosen: list[int] = []
    for slot in range(limit):
        index = round(slot * last / (limit - 1))
        if not chosen or index != chosen[-1]:
            chosen.append(index)
    return chosen


def _subyear_year(code: str) -> str | None:
    text = code.strip()
    for pattern in (_QUARTER_RE, _MONTH_RE, _HALF_RE):
        match = pattern.fullmatch(text)
        if match:
            return match.group(1)
    return None


def chart_x_labels(periods: list[str]) -> list[dict[str, str]]:
    """Bottom-axis labels. A long quarterly or monthly series uses years."""
    ordered = sorted({period for period in periods if period})
    if not ordered:
        return []
    years: list[str] = []
    anchor: dict[str, str] = {}
    for period in ordered:
        year = _subyear_year(period)
        if year is None:
            years = []
            break
        if year not in anchor:
            anchor[year] = period
            years.append(year)
    if len(years) >= 2 and len(ordered) > _MAX_X_LABELS:
        return [
            {"period": anchor[years[index]], "label": years[index]}
            for index in _even_indexes(len(years), _MAX_X_LABELS)
        ]
    indexes = _even_indexes(len(ordered), _MAX_X_LABELS)
    return [{"period": ordered[index], "label": period_label(ordered[index])} for index in indexes]


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
    return {
        "min": low,
        "max": high,
        "ticks": ticks,
        "labels": chart_x_labels(_chart_periods(series)),
    }


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
    """Same text as the table column: 'Esbjerg, 80+', not 'OMRÅDE: Esbjerg, ALDER: 80+'."""
    return display_label(str(fact.get("detail") or ""), str(fact.get("table") or ""))


def _export_rows(facts: list[dict[str, Any]]) -> tuple[list[str], list[list[str]]]:
    rows = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        rows.append(
            [
                str(fact.get("table") or ""),
                _shown_period(fact),
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
        row = {
            "tabel": str(fact.get("table") or ""),
            "periode": raw,
            "periode_label": _shown_period(fact),
            "dimensioner": _dimension_label(fact),
            "dimensioner_rå": detail,
            "værdi": _csv_number(fact.get("value")),
            "enhed": str(fact.get("unit") or ""),
        }
        if fact.get("projection"):
            row["fremskrivning"] = True
        rows.append(row)
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


_UNPUBLISHED_HOUSEHOLD = "Det præcise tal er ikke offentliggjort. Husstandsstørrelsen er topkodet."


def _household_exact_unpublished(question: str) -> bool:
    """No table publishes children living with more than two adults."""
    folded = question.casefold()
    if "børn" not in folded and "born" not in folded:
        return False
    return re.search(r"mere end\s+(?:2|to)\s+voksne", folded) is not None


def _turn(
    question: str,
    answer: str,
    facts: list[dict[str, Any]],
    calls: list[dict[str, str]],
) -> TurnResult:
    if _household_exact_unpublished(question):
        return TurnResult(
            answer=_UNPUBLISHED_HOUSEHOLD,
            facts=[],
            calls=calls,
            chart=[],
            csv="",
            show_numbers=False,
        )
    # A figure in the prose is kept when it is in the question, a hent_data
    # value, a period or row label from those rows, the fetched table title,
    # or a variable value on this turn's hent_data arguments. A fetched value
    # tied to one of those periods or row labels has to be that row's value.
    # A figure from the question or a label is not paired.
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
        suggestions=[] if kept else _lookup_suggestions(calls),
    )


_DID_YOU_MEAN_RE = re.compile(r"Did you mean ([^?]+)\?")
_NO_TABLE_RE = re.compile(r"No table matches (?P<query>'[^']*'|\"[^\"]*\")")
_NOT_FOUND_STARTS = ("No table matches", "Unknown table", "Unknown value")
# The MCP client prefixes a tool failure with "Error executing tool <name>:".
_TOOL_ERROR_RE = re.compile(r"^(?:Error executing tool\b[^:]*:\s*)+", re.IGNORECASE)


def _lookup_suggestions(calls: list[dict[str, str]]) -> list[str]:
    """Lines to show when a search, a table, or a value was not found."""
    lines: list[str] = []
    for call in calls:
        result = " ".join(str(call.get("result") or "").split())
        result = _TOOL_ERROR_RE.sub("", result).strip()
        if not result:
            continue
        meant = _DID_YOU_MEAN_RE.findall(result)
        if meant:
            lines.extend(f"Mente du {item.strip()}?" for item in meant)
            continue
        missed = _NO_TABLE_RE.search(result)
        if missed:
            quoted = missed.group("query")
            lines.append(
                f"Ingen tabel matcher {quoted}. Prøv færre ord, eller et emne som "
                "'befolkning', 'indkomst' eller 'priser'."
            )
            continue
        if result.startswith(_NOT_FOUND_STARTS):
            parts = re.split(r"(?<=\.)\s+", result)
            line = " ".join(parts[:2]).strip()
            if len(line) > 320:
                line = line[:317].rstrip() + "…"
            if line:
                lines.append(line)
    return list(dict.fromkeys(lines))


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
    "vokset",
    "voksede",
    "flere",
    "færre",
    "faerre",
    "steg",
    "steget",
    "tilvækst",
)
_CHANGE_MARKERS = (
    "vokset",
    "voksede",
    "flere",
    "faerre",
    "steg",
    "steget",
    "tilvaekst",
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
# Next to a period or row label: "94.486 (2013K1)", "2013K1 94.486", "i 2013K1".
_TIE_GAP_RE = re.compile(
    r"^[ \t]*(?:[([][ \t]*|[ \t]*[)\]]|:|in|for|på|pa|i)?[ \t]*$",
    re.IGNORECASE,
)
_RANGE_PREFIX_RE = re.compile(
    r"(?<![0-9A-Za-zÆØÅæøå])"
    r"(?:indtil|until|siden|since|mellem|between|from|fra|til)"
    r"[ \t]*[([]?[ \t]*$",
    re.IGNORECASE,
)
# "en forskel på 153" and "i alt 9.999" are not a claim that the row equals that figure.
_TOTAL_PHRASE_RE = re.compile(
    r"(?<![0-9A-Za-zÆØÅæøå])"
    r"(?:i alt|in total|forskellen|forskel|stigningen|stigning|faldet|fald|procent|"
    r"difference|percentage|percent)"
    r"(?![0-9A-Za-zÆØÅæøå])"
    r"(?:[ \t]+\S+){0,2}[ \t]*$",
    re.IGNORECASE,
)
# "." ends a sentence only before a capital or the end. "1. kvt." and "94.333" stay intact.
_CLAUSE_SPLIT_RE = re.compile(r"[!?;\n]|\s+/\s+|(?<=\.)(?=\s+[A-ZÆØÅ]|\s*$)")
_BARE_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_PERIOD_EDGE = "0-9A-Za-zÆØÅæøå"


def _looks_danish(question: str) -> bool:
    if re.search(r"[æøåÆØÅ]", question):
        return True
    words = set(re.findall(r"[a-zA-ZæøåÆØÅ]+", question.casefold()))
    return len(words & _DANISH_WORDS) >= 2


def _asks_named_difference(question: str) -> bool:
    """'Hvor stor var forskellen' asks for the change. 'Forskellen er 153' does not."""
    folded = question.casefold()
    return "forskel" in folded and re.search(r"\bhvor\b", folded) is not None


def _asks_for_trend(question: str) -> bool:
    folded = question.casefold()
    return any(marker in folded for marker in _TREND_MARKERS) or _asks_named_difference(question)


def _asks_for_change(question: str) -> bool:
    """A growth or 'how many more' question, not every time series."""
    folded = question.casefold().replace("æ", "ae").replace("ø", "o").replace("å", "aa")
    return any(marker in folded for marker in _CHANGE_MARKERS) or _asks_named_difference(question)


def _asks_city_rank(question: str) -> bool:
    """Which town is largest. Not a municipality stock and not a growth question."""
    folded = _fold_place(question)
    if not re.search(r"\bby\b|\bbyer\b|\bbyomr", folded):
        return False
    return any(word in folded for word in ("storst", "tiende", "flest", "indbygg"))


def _wants_city_list(question: str) -> bool:
    folded = question.casefold()
    if "flest" in folded or "rækkefølge" in folded or "raekkefolge" in folded:
        return True
    return re.search(r"\bti byer\b", folded) is not None


_BY_CODE_RE = re.compile(r"^\d{3}-\d+\s+")


def rank_urban_areas(pairs: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Whole byområder, largest first.

    Parts of Hovedstadsområdet are one city. The parent cell in BY3 is 0,
    so the city is the sum of those parts. Landdistrikter are not cities.
    """
    capital = 0.0
    cities: list[tuple[str, float]] = []
    for raw, value in pairs:
        name = _BY_CODE_RE.sub("", str(raw).strip())
        folded = _fold_place(name)
        if "landdistrik" in folded or "uden fast" in folded:
            continue
        if folded == "hovedstadsomraadet" or folded.startswith("hovedstadsomraadet ("):
            continue
        if "del af hovedstadsomraadet" in folded:
            if value > 0:
                capital += float(value)
            continue
        if value <= 0:
            continue
        cities.append((name, float(value)))
    if capital > 0:
        cities.append(("Hovedstadsområdet", capital))
    cities.sort(key=lambda item: (-item[1], _fold_place(item[0])))
    return cities


def _by_name(detail: str) -> str:
    for part in detail.split(", "):
        key, sep, value = part.partition(": ")
        if sep and _fold_place(key) in {"byer", "by"}:
            return value
    return ""


def _city_rank_answer(question: str, facts: list[dict[str, Any]]) -> str | None:
    """The requested urban-area rank from BY3 rows. Other tables are not towns."""
    pairs: list[tuple[str, float]] = []
    periods: list[str] = []
    for fact in facts:
        if str(fact.get("table") or "").upper() != "BY3":
            continue
        if not _single_value(fact) or fact.get("unpublished"):
            continue
        name = _by_name(str(fact.get("detail") or ""))
        if not name:
            continue
        pairs.append((name, float(fact["value"])))
        period = str(fact.get("period") or "")
        if period:
            periods.append(period)
    ranked = rank_urban_areas(pairs)
    if len(ranked) < 10:
        return None
    period = period_label(max(periods, key=_period_sort_key)) if periods else ""
    when = f", {period}" if period else ""
    definition = (
        f"BY3 folketal{when}, byområde. "
        "Hovedstadsområdet er ét byområde, summen af delene, fordi selve rækken er 0. "
        "Områder uden for byerne tæller ikke med."
    )
    if _wants_city_list(question):
        lines = [
            f"{index}. {name}: {_da_number(value)}"
            for index, (name, value) in enumerate(ranked[:10], start=1)
        ]
        return f"{definition}\n" + "\n".join(lines)
    number = 10 if "tiende" in question.casefold() else 1
    name, value = ranked[number - 1]
    return f"{name} er den {number}. største by med {_da_number(value)} indbyggere ({definition})"


def _with_municipality_note(question: str, text: str) -> str:
    """An unnamed place on a population table is the municipality, not the town."""
    folded = question.casefold()
    if "kommune" in folded or re.search(r"\bby\b|\bbyer\b|\bbyomr", folded):
        return text
    if "kommune" not in text.casefold():
        return text
    note = "Tallet er kommunen. Byområdet er et andet tal."
    if note.casefold() in text.casefold():
        return text
    return f"{text}\n{note}" if text else note


def _period_sort_key(code: str) -> tuple[int, int, int, str]:
    match = re.fullmatch(
        r"(\d{4})(?:(K|Q)([1-4])|(H)([12])|(M)(\d{2}))?",
        code.strip(),
        re.IGNORECASE,
    )
    if match is None:
        return (0, 0, 0, code)
    year = int(match.group(1))
    if match.group(2):
        return (year, 1, int(match.group(3)), code)
    if match.group(4):
        return (year, 2, int(match.group(5)), code)
    if match.group(6):
        return (year, 3, int(match.group(7)), code)
    return (year, 0, 0, code)


def _comparison_line(facts: list[dict[str, Any]]) -> str | None:
    """Both endpoints, the people difference, and the percentage. One series only."""
    numeric = [fact for fact in facts if _single_value(fact) and not fact.get("unpublished")]
    if len(numeric) < 2:
        return None
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for fact in numeric:
        key = (
            str(fact.get("table") or ""),
            str(fact.get("unit") or ""),
            str(fact.get("detail") or ""),
        )
        groups.setdefault(key, []).append(fact)
    if len(groups) != 1:
        return None
    series = next(iter(groups.values()))
    ordered = sorted(series, key=lambda fact: _period_sort_key(str(fact.get("period") or "")))
    first, last = ordered[0], ordered[-1]
    if str(first.get("period") or "") == str(last.get("period") or ""):
        return None
    start, end = float(first["value"]), float(last["value"])
    if start == 0:
        return None
    difference = abs(end - start)
    percent = difference / abs(start) * 100
    shown = f"{percent:.2f}".replace(".", ",")
    return (
        f"{_da_number(start)} ({period_label(str(first.get('period') or ''))}) og "
        f"{_da_number(end)} ({period_label(str(last.get('period') or ''))}). "
        f"Forskellen er {_da_number(difference)} ({shown} procent)."
    )


def _series_endpoints(
    facts: list[dict[str, Any]],
) -> tuple[float, float, float] | None:
    """Start, end, and absolute difference for one series. Period digits are not values."""
    numeric = [fact for fact in facts if _single_value(fact) and not fact.get("unpublished")]
    if len(numeric) < 2:
        return None
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for fact in numeric:
        key = (
            str(fact.get("table") or ""),
            str(fact.get("unit") or ""),
            str(fact.get("detail") or ""),
        )
        groups.setdefault(key, []).append(fact)
    if len(groups) != 1:
        return None
    series = next(iter(groups.values()))
    ordered = sorted(series, key=lambda fact: _period_sort_key(str(fact.get("period") or "")))
    first, last = ordered[0], ordered[-1]
    if str(first.get("period") or "") == str(last.get("period") or ""):
        return None
    start, end = float(first["value"]), float(last["value"])
    if start == 0:
        return None
    return start, end, abs(end - start)


def _states_change(answer: str, facts: list[dict[str, Any]]) -> bool:
    change = _series_endpoints(facts)
    if change is None:
        return False
    stated = _numbers_in(answer)

    def has(wanted: float) -> bool:
        return any(abs(item - wanted) <= max(0.05, abs(wanted) * 0.001) for item in stated)

    start, end, difference = change
    return has(start) and has(end) and has(difference)


_NEIGHBOUR_RE = re.compile(
    r"not in table (?P<table>[A-Za-z0-9]+)\. Neighbouring periods: "
    r"(?P<before>\S+) and (?P<after>\S+)"
)
_SPAN_RE = re.compile(
    r'"id"\s*:\s*"(?P<table>[A-Za-z0-9]+)".*?"first"\s*:\s*\{\s*"id"\s*:\s*"(?P<first>[^"]+)"',
    re.DOTALL,
)


_OMISSION_RE = re.compile(
    r"ikke med|ikke vist|afkortede|afkortet|not shown",
    re.IGNORECASE,
)


def _without_false_omissions(answer: str, facts: list[dict[str, Any]]) -> str:
    """Drop a claim that a period we fetched is missing from the tool result."""
    mentioned = []
    for fact in facts:
        code = str(fact.get("period") or "")
        if not code:
            continue
        mentioned.append(code)
        mentioned.append(period_label(code))
    kept: list[str] = []
    for sentence in re.split(r"\n+|(?<=[.!?])\s+", answer):
        piece = sentence.strip()
        if not piece:
            continue
        denies = _OMISSION_RE.search(piece) is not None
        names_fetched = any(item.casefold() in piece.casefold() for item in mentioned)
        if denies and names_fetched:
            continue
        kept.append(piece)
    return " ".join(kept)


def _with_boundary_note(question: str, facts: list[dict[str, Any]], text: str) -> str:
    """Keep the coverage caveat when the comparison line replaces the prose."""
    note = _reform_note(question, facts)
    if note and note.casefold() not in text.casefold():
        return f"{text}\n{note}" if text else note
    return text


def _reform_note(question: str, facts: list[dict[str, Any]]) -> str:
    """A municipality series that starts in 2008 is the post-2007 boundary."""
    years = [int(year) for year in re.findall(r"\b(?:1[6-9]|20)\d{2}\b", question)]
    numeric = [fact for fact in facts if _single_value(fact)]
    if not years or not numeric:
        return ""
    earliest = min(numeric, key=lambda fact: _period_sort_key(str(fact.get("period") or "")))
    period = str(earliest.get("period") or "")
    start = re.match(r"(\d{4})", period)
    if start is None or int(start.group(1)) <= min(years):
        return ""
    if int(start.group(1)) != 2008:
        return ""
    table = str(earliest.get("table") or "")
    if not table.upper().startswith(("FOLK", "BEFOLK")):
        return ""
    place = ""
    for part in str(earliest.get("detail") or "").split(", "):
        name, sep, value = part.partition(": ")
        if not sep or _fold_place(name) not in _AREA_FIELDS:
            continue
        if _fold_place(value) in {"hele landet", "i alt"}:
            continue
        place = value
    if not place:
        return ""
    return (
        f"{place} i {table} starter {period_label(period)}, "
        f"kommunegrænsen efter kommunalreformen i 2007. "
        f"{min(years)} er ikke med i denne serie."
    )


def _coverage_explanation(question: str, calls: list[dict[str, str]] | None) -> str:
    """Say which table lacks the year. Do not claim that no table has any data."""
    if not calls:
        return ""
    years = [int(year) for year in re.findall(r"\b(?:1[6-9]|20)\d{2}\b", question)]
    for call in calls:
        if call.get("name") != "hent_data":
            continue
        match = _NEIGHBOUR_RE.search(str(call.get("result") or ""))
        if match is None:
            continue
        year = years[-1] if years else ""
        return (
            f"{match.group('table')} har ikke {year}. "
            f"Nærmeste perioder i tabellen er {match.group('before')} og {match.group('after')}."
        )
    if not years:
        return ""
    requested = min(years)
    late: list[str] = []
    for call in calls:
        if call.get("name") != "tabelinfo":
            continue
        match = _SPAN_RE.search(str(call.get("result") or ""))
        if match is None:
            continue
        first = match.group("first")
        start = re.match(r"(\d{4})", first)
        if start and int(start.group(1)) > requested:
            late.append(f"{match.group('table')} starter ved {first}")
    if not late:
        return ""
    joined = " og ".join(dict.fromkeys(late))
    return f"{joined}, så {requested} er ikke med i den tabel."


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
            "Do not return a long list of earlier periods. "
            "Do not pass variables the question does not name."
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
    turn's hent_data arguments. A fetched value tied in the same clause to one
    period or row label has to be that row's value. A figure that only comes
    from the question or a label is left to that check. A question that asks for one
    figure is answered with the page's own line for the latest period. The
    model's sentence is not used, so a long series or an invented total cannot
    become the answer. A note that the requested period is not published yet
    is kept with that line.
    """
    if not facts:
        note = _timeout_after_table(calls)
        if note:
            return note
        gap = _coverage_explanation(question, calls)
        if gap and _log_unverified(answer, question, facts, calls):
            # The model's extra figures are dropped. The tool's own gap stays.
            return gap
        if _explicit_gap(answer) and not _log_unverified(answer, question, facts, calls):
            return answer
        if _log_unverified(answer, question, facts, calls):
            return NO_FETCHED_NUMBERS
        if gap:
            return gap
        if _is_page_message(answer):
            return answer
        if _household_exact_unpublished(question):
            return _UNPUBLISHED_HOUSEHOLD
        return NO_DATA_ANSWER
    refusal = _relevance_refusal(question, facts)
    if refusal:
        return refusal
    if _asks_city_rank(question):
        ranked = _city_rank_answer(question, facts)
        if ranked:
            return ranked
        return (
            "Spørgsmålet gælder byer (byområder), ikke kommuner, regioner eller hele landet. "
            "De hentede tal er ikke en sammenlignelig byrækkefølge."
        )
    if not _asks_for_trend(question):
        lines = _latest_lines(facts)
        if lines:
            return _with_municipality_note(
                question,
                _with_outside_note(
                    _with_unpublished_note("\n".join(lines), answer),
                    question,
                ),
            )
    if _log_unverified(answer, question, facts, calls):
        change = _comparison_line(facts) if _asks_for_change(question) else None
        if change:
            return _with_boundary_note(question, facts, change)
        if _single_fact(facts):
            return _summary_line(facts[0])
        return UNVERIFIED_NUMBERS
    if _asks_for_change(question):
        answer = _without_false_omissions(answer, facts)
        note = _reform_note(question, facts)
        if note and note.casefold() not in answer.casefold():
            answer = f"{answer}\n{note}".strip() if answer else note
        if not _states_change(answer, facts):
            change = _comparison_line(facts)
            if change:
                return _with_boundary_note(
                    question, facts, f"{change}\n{answer}" if answer else change
                )
    if _single_fact(facts):
        line = _summary_line(facts[0])
        return f"{line}\n\n{answer}" if answer else line
    return answer


_NOTE_GLUE_RE = re.compile(
    r"(?:\s*(?:,|;|:|—|-)?\s*\b(?:med|og|viser|showing|shows|show|er|på|pa|of|with|at|"
    r"ca|cirka|about|approx|approximately)\b\.?\s*)+$",
    re.IGNORECASE,
)
_DANGLING_OPEN_RE = re.compile(r"[\s(\[{]+$")
_QUARTER_WORD_RE = re.compile(
    r"\s*\.?\s*(?:kvartal|kvt|halvår|halvaar)\b",
    re.IGNORECASE,
)


def _is_sentence_end(text: str, index: int) -> bool:
    """A period inside 1.234,5 or before 'kvartal' does not end the sentence."""
    char = text[index]
    if char == "\n":
        return True
    if char in "!?":
        return True
    if char != ".":
        return False
    prev = text[index - 1] if index else ""
    nxt = text[index + 1] if index + 1 < len(text) else ""
    if prev.isdigit() and nxt.isdigit():
        return False
    if prev.isdigit() and _QUARTER_WORD_RE.match(text[index + 1 :]):
        return False
    return True


def _clause_around(text: str, start: int, end: int) -> str:
    left = start
    while left > 0 and not _is_sentence_end(text, left - 1):
        left -= 1
    right = end
    while right < len(text) and not _is_sentence_end(text, right):
        right += 1
    return text[left:right].strip(" \t;:—-")


def _without_unchecked_figures(note: str) -> str:
    """Keep the period words. Drop a statistic, including one Danish number."""
    cut = len(note)
    for match in _NUMBER_RE.finditer(note):
        token = match.group(1)
        if re.fullmatch(r"(?:19|20)\d{2}", token):
            continue
        if _QUARTER_WORD_RE.match(note[match.end() :]):
            continue
        cut = match.start()
        break
    kept = _NOTE_GLUE_RE.sub("", note[:cut])
    kept = _DANGLING_OPEN_RE.sub("", kept)
    kept = re.sub(r"\s{2,}", " ", kept)
    return kept.strip(" \t;:—-,")


def _unpublished_period_note(answer: str) -> str:
    """The not-published clause, with period names and no statistic."""
    match = re.search(r"ikke offentliggjort|not published", answer, re.IGNORECASE)
    if match is None:
        return ""
    note = _clause_around(answer, match.start(), match.end())
    if note and not re.search(r"\b(?:seneste|latest)\b", note, re.IGNORECASE):
        rest = answer[match.end() :]
        follow = re.search(r"\b(?:seneste|latest)\b", rest, re.IGNORECASE)
        if follow and follow.start() < 80:
            note = _clause_around(answer, match.start(), match.end() + follow.end())
    return _without_unchecked_figures(note)


_EXPLICIT_GAP_RE = re.compile(
    r"ikke offentliggjort|kan ikke opgøres|kan ikke opgoeres|topkod",
    re.IGNORECASE,
)
_OUTSIDE_RE = re.compile(r"sverige|sweden", re.IGNORECASE)


def _explicit_gap(answer: str) -> bool:
    return _EXPLICIT_GAP_RE.search(answer) is not None


def _with_outside_note(text: str, question: str) -> str:
    """A comparison with a country outside Statistikbanken keeps that caveat."""
    if _OUTSIDE_RE.search(question) is None or _OUTSIDE_RE.search(text):
        return text
    if not text:
        return "Sverige er ikke i Statistikbanken."
    return f"{text}\nSverige er ikke i Statistikbanken."


def _with_unpublished_note(text: str, answer: str) -> str:
    note = _unpublished_period_note(answer)
    if not note or note in text:
        return text
    return f"{text}\n{note}" if text else note


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
    detail = display_label(str(fact.get("detail") or ""), str(fact.get("table") or ""))
    if detail:
        bits.append(detail)
    if fact.get("projection"):
        bits.append("fremskrivning")
    label = ", ".join(bit for bit in bits if bit)
    line = f"{value} ({label})" if label else value
    derivation = str(fact.get("derivation") or "")
    if derivation:
        line = f"{line} {derivation}"
    return line


_STICK = {
    "FOLK1C": "FOLK1E",
    "AKU100K": "AKU110K",
    "BANE21": "BANE25",
    "LONS20": "LONS30",
}
_MONTH_NAME_RE = re.compile(
    r"januar|februar|marts|april|maj|juni|juli|august|september|oktober|november|december"
    r"|kvartal|kvt|halvår|halvaar",
    re.IGNORECASE,
)
_YEAR_IN_QUESTION_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_FUEL_CHILD_RE = re.compile(r"0722\d+")


_FETCH_TIMEOUT_RE = re.compile(r"did not respond in time|timed out|\bTimeout", re.IGNORECASE)


def _fetch_timed_out(result: str) -> bool:
    return _FETCH_TIMEOUT_RE.search(result) is not None


def _retry_fetch(arguments: dict[str, Any]) -> dict[str, Any]:
    """One smaller try. A timed-out sibling becomes the headline table."""
    updated = dict(arguments)
    table = str(updated.get("table") or "").upper()
    for kept, sibling in _STICK.items():
        if table == sibling:
            updated["table"] = kept
            break
    values = dict(updated.get("values") or {})
    periods = _codes(values, "Tid")
    if len(periods) > 1:
        _put_code(values, "Tid", periods[-1:])
        updated["values"] = values
    return updated


def _timeout_after_table(calls: list[dict[str, str]] | None) -> str:
    """A failed fetch is not the same as a table that has no such figure."""
    if not calls:
        return ""
    timed = False
    table = ""
    for call in calls:
        result = str(call.get("result") or "")
        if _fetch_timed_out(result):
            timed = True
        if call.get("name") == "find_tabel":
            found = _top_table(result)
            if found:
                table = found
        if call.get("name") == "hent_data":
            try:
                payload = json.loads(str(call.get("arguments") or ""))
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and payload.get("table"):
                table = str(payload["table"])
    if not timed or not table:
        return ""
    return f"Opslaget i {table} tog for lang tid. Prøv igen."


def _asks_net_unemployment(question: str) -> bool:
    """Net unemployment, not the gross total and not a sentence that only says brutto."""
    folded = question.casefold()
    if "netto" not in folded or not re.search(r"ledig", folded):
        return False
    return "brutto" not in folded or folded.index("netto") < folded.index("brutto")


def _national_area(values: dict[str, Any]) -> bool:
    codes = _codes(values, "OMRÅDE")
    if not codes:
        return True
    return {code.strip().casefold() for code in codes} <= {"000", "hele landet"}


def net_series_block(question: str, arguments: dict[str, Any]) -> str:
    """Refuse a gross table when the question asked for net and a place."""
    if not _asks_net_unemployment(question):
        return ""
    table = str(arguments.get("table") or "").upper()
    if table not in {"AUS08", "AUS09", "AUF01", "AUF02"}:
        return ""
    return (
        f"{table} has no net series. Nettoledige are not the gross total on this table. "
        "The national net count is AUS07, YD NET. This table cannot split a municipality into net."
    )


def _annual_unemployment_average(question: str) -> bool:
    """A named year asked as an average of full-time unemployed, not the net series."""
    folded = question.casefold()
    if "netto" in folded or "gennemsnit" not in folded:
        return False
    if not re.search(r"fuldtidsledig|ledige|ledighed", folded):
        return False
    return re.search(r"\b(?:19|20)\d{2}\b", question) is not None


def apply_search_context(question: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Keep words the question already used when the model searches without them."""
    updated = dict(arguments)
    if _asks_city_rank(question):
        query = str(updated.get("query") or "")
        if "byomr" not in query.casefold():
            updated["query"] = f"{query} byområder".strip()
        return updated
    if not _annual_unemployment_average(question):
        return updated
    query = str(updated.get("query") or "")
    if re.search(r"gennemsnit", query, re.IGNORECASE):
        return updated
    if not re.search(r"ledig|fuldtid", f"{query} {question}", re.IGNORECASE):
        return updated
    updated["query"] = f"{query} gennemsnit".strip()
    return updated


def preferred_table(ranked: str, requested: str) -> str:
    """If search ranked the headline table, do not switch to its sibling."""
    ranked_id = ranked.strip().upper()
    requested_id = requested.strip().upper()
    for kept, sibling in _STICK.items():
        if ranked_id == kept and requested_id == sibling:
            return kept
    return requested_id or requested


_CITY_TABLES = frozenset(
    {
        "",
        "FOLK1A",
        "FOLK1AM",
        "FOLK1B",
        "FOLK1C",
        "FOLK1E",
        "BEFOLK1",
        "BEFOLK2",
        "BEFOLK3",
        "BY1",
        "BY2",
        "BY3",
        "FRKM126",
    }
)


def apply_fetch_defaults(question: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Correct the code the question already named, before the API call."""
    updated = dict(arguments)
    values = dict(updated.get("values") or {})
    table = str(updated.get("table") or "").upper()
    folded = question.casefold()
    if _asks_city_rank(question) and table in _CITY_TABLES:
        # A town rank is byområder, not municipalities or the whole country.
        updated["table"] = "BY3"
        updated["values"] = {"FOLKARTAET": ["FOLKETAL"], "BYER": ["*"]}
        updated.pop("sum_from", None)
        updated.pop("sum_range", None)
        return updated
    if (
        _asks_net_unemployment(question)
        and table in {"AUS08", "AUS09", "AUF01", "AUF02"}
        and _national_area(values)
    ):
        # The area table is gross only. National net is a different series.
        table = "AUS07"
        updated["table"] = table
        for key in list(values):
            if str(key).upper() == "OMRÅDE":
                del values[key]
    if _annual_unemployment_average(question) and table in {"AUS08", "AUS09", "AUF01", "AUF02"}:
        table = "AUL01"
        updated["table"] = table
        for key in list(values):
            if str(key).upper() == "SAESONFAK":
                del values[key]
    if table in {"AUS07", "AUS08", "AUS09"} and not re.search(r"procent|pct\b|%", folded):
        if re.search(r"sæsonkorriger|saesonkorriger|seasonally", folded):
            _put_code(values, "SAESONFAK", ["10"])
        else:
            _put_code(values, "SAESONFAK", ["24"])
    if table == "AUS07" and re.search(r"netto", folded) and "brutto" not in folded:
        _put_code(values, "YD", ["NET"])
    if table == "PRIS01" and re.search(r"brændstof|braendstof|benzin|diesel", folded):
        codes = _codes(values, "VAREGR")
        if any(_FUEL_CHILD_RE.fullmatch(code) for code in codes):
            _put_code(values, "VAREGR", ["0722"])
    if table == "INDKP201" and "median" in folded and "disponibel" not in folded:
        _put_code(values, "INDKOMSTTYPE", ["105"])
        _put_code(values, "ENHED", ["0055"])
    year = _single_total_year(question)
    if year is not None:
        _put_code(values, "Tid", [year])
    if values:
        updated["values"] = values
    return updated


def _codes(values: dict[str, Any], key: str) -> list[str]:
    for name, raw in values.items():
        if str(name).casefold() != key.casefold():
            continue
        if isinstance(raw, list):
            return [str(item) for item in raw]
        return [str(raw)]
    return []


def _put_code(values: dict[str, Any], key: str, codes: list[str]) -> None:
    for name in list(values):
        if str(name).casefold() == key.casefold():
            del values[name]
    values[key] = codes


def _single_total_year(question: str) -> str | None:
    """'i alt i 2025' is that year. A named month or quarter is not."""
    if _MONTH_NAME_RE.search(question):
        return None
    years = _YEAR_IN_QUESTION_RE.findall(question)
    if len(years) != 1:
        return None
    year = years[0]
    folded = question.casefold()
    if "i alt" in folded or f"i {year}" in folded or "gennemsnit" in folded:
        return year
    return None


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
    flagged = []
    for start, end, number in _figure_mentions(scanned, []):
        if _near(number, allowed):
            continue
        # Spans match the unmasked answer; friendly periods stay visible there.
        if _matches_derivation(answer, start, end, number, facts):
            continue
        flagged.append(number)
    # _figure_mentions leaves bare years to the period check. A year that is not
    # in the question, a row, or a fetched period is still an invented figure.
    for match in _NUMBER_RE.finditer(scanned):
        token = match.group(1)
        if _BARE_YEAR_RE.fullmatch(token) is None:
            continue
        year = _parse_stat_number(token)
        if not _near(year, allowed):
            flagged.append(year)
    known_periods = _period_tokens(question)
    for fact in facts:
        known_periods.update(_period_tokens(str(fact.get("period") or "")))
    for title in titles:
        known_periods.update(_period_tokens(title))
    for token in _period_tokens(answer):
        if token not in known_periods:
            flagged.append(float(token[:4]))
    flagged.extend(_mispaired_figures(answer, facts, question))
    return flagged


def _mispaired_figures(
    answer: str,
    facts: list[dict[str, Any]],
    question: str = "",
) -> list[float]:
    """Fetched values the clause assigns to a period or row they do not belong to.

    A tie is either adjacency (parentheses, a colon, or i/in/for/på) or the only
    figure in the clause beside the only period or row label. Only a number that
    is itself a fetched row value can be flagged here. A figure from the question,
    a period or age label, or nowhere in the rows keeps the ordinary check.
    """
    periods = _period_mentions(answer, facts)
    labels = _label_mentions(answer, facts, [(start, end) for start, end, _code in periods])
    covered = [(start, end) for start, end, _code in periods]
    covered.extend((start, end) for start, end, _label in labels)
    numbers = _figure_mentions(answer, covered)
    if not numbers:
        return []
    period_for: dict[int, set[str]] = {}
    label_for: dict[int, set[str]] = {}

    def bind_period(index: int, code: str) -> None:
        period_for.setdefault(index, set()).add(code)

    def bind_label(index: int, label: str) -> None:
        label_for.setdefault(index, set()).add(label)

    anchors: list[tuple[int, int, str, int | str]] = []
    for start, end, code in periods:
        anchors.append((start, end, "period", code))
    for start, end, label in labels:
        anchors.append((start, end, "label", label))
    for index, (start, end, _value) in enumerate(numbers):
        anchors.append((start, end, "number", index))
    anchors.sort(key=lambda item: item[0])
    for position, (start, end, kind, payload) in enumerate(anchors):
        if kind != "number" or not isinstance(payload, int):
            continue
        for neighbor in (position - 1, position + 1):
            if not 0 <= neighbor < len(anchors):
                continue
            other_start, other_end, other_kind, other = anchors[neighbor]
            if other_kind == "number":
                continue
            gap = answer[end:other_start] if other_start >= end else answer[other_end:start]
            if _TIE_GAP_RE.fullmatch(gap) is None:
                continue
            if other_kind == "period" and isinstance(other, str):
                if _range_prefixed(answer, other_start):
                    continue
                bind_period(payload, other)
            elif isinstance(other, str):
                bind_label(payload, other)
    for clause_start, clause_end in _clause_ranges(answer):
        claimed = [
            index
            for index, (start, end, _value) in enumerate(numbers)
            if clause_start <= start and end <= clause_end and not _stated_as_total(answer, start)
        ]
        clause_periods = [
            code
            for start, end, code in periods
            if clause_start <= start and end <= clause_end and not _range_prefixed(answer, start)
        ]
        clause_labels = [
            label for start, end, label in labels if clause_start <= start and end <= clause_end
        ]
        if len(claimed) == 1 and len(clause_periods) == 1:
            bind_period(claimed[0], clause_periods[0])
        if len(claimed) == 1 and len(clause_labels) == 1:
            bind_label(claimed[0], clause_labels[0])
    row_values = _row_values(facts)
    exempt = _label_figures(question)
    for fact in facts:
        exempt.update(_label_figures(str(fact.get("detail") or "")))
        exempt.update(_period_years(str(fact.get("period") or "")))
    flagged: list[float] = []
    for index in set(period_for) | set(label_for):
        value = numbers[index][2]
        if not _near(value, row_values) or _near(value, exempt):
            continue
        codes = period_for.get(index, set())
        bound_labels = label_for.get(index, set())
        if not any(_row_has(fact, value, codes, bound_labels) for fact in facts):
            flagged.append(value)
    return flagged


def _row_values(facts: list[dict[str, Any]]) -> set[float]:
    found: set[float] = set()
    for fact in facts:
        value = fact.get("value")
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        found.add(float(value))
    return found


def _row_has(fact: dict[str, Any], number: float, codes: set[str], labels: set[str]) -> bool:
    value = fact.get("value")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    if not _near(float(number), {float(value)}):
        return False
    if codes and (len(codes) != 1 or str(fact.get("period") or "").strip().casefold() not in codes):
        return False
    return not labels or labels <= _fact_labels(fact)


def _fact_labels(fact: dict[str, Any]) -> set[str]:
    found: set[str] = set()
    for part in str(fact.get("detail") or "").split(", "):
        _name, sep, value = part.partition(": ")
        text = (value if sep else part).strip()
        folded = _fold_place(text)
        if not text or text.casefold() in _TOTAL_LABELS or len(folded) < 4:
            continue
        if re.search(r"[a-z]", folded):
            found.add(folded)
    return found


def _period_mentions(text: str, facts: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
    forms: list[tuple[str, str]] = []
    seen: set[str] = set()
    for fact in facts:
        code = str(fact.get("period") or "").strip()
        if not code:
            continue
        canonical = code.casefold()
        for form in (code, *_period_phrases(code)):
            key = form.casefold()
            if form and key not in seen:
                seen.add(key)
                forms.append((form, canonical))
    forms.sort(key=lambda item: len(item[0]), reverse=True)
    occupied: list[tuple[int, int]] = []
    found: list[tuple[int, int, str]] = []
    for form, canonical in forms:
        pattern = re.compile(
            rf"(?<![{_PERIOD_EDGE}]){re.escape(form)}(?![{_PERIOD_EDGE}])",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if _span_overlaps(span, occupied):
                continue
            occupied.append(span)
            found.append((span[0], span[1], canonical))
    return found


def _label_mentions(
    text: str,
    facts: list[dict[str, Any]],
    periods: list[tuple[int, int]],
) -> list[tuple[int, int, str]]:
    labels = {label for fact in facts for label in _fact_labels(fact)}
    labels_longest = sorted(labels, key=len, reverse=True)
    folded, origins = _fold_index(text)
    occupied = list(periods)
    found: list[tuple[int, int, str]] = []
    for label in labels_longest:
        pattern = re.compile(rf"(?<![0-9a-z]){re.escape(label)}s?(?![0-9a-z])")
        for match in pattern.finditer(folded):
            start = origins[match.start()]
            end = origins[match.end() - 1] + 1
            if _span_overlaps((start, end), occupied):
                continue
            occupied.append((start, end))
            found.append((start, end, label))
    return found


def _figure_mentions(
    text: str,
    covered: list[tuple[int, int]],
) -> list[tuple[int, int, float]]:
    found: list[tuple[int, int, float]] = []
    for match in _NUMBER_RE.finditer(text):
        span = match.span(1)
        token = match.group(1)
        if _span_overlaps(span, covered) or _BARE_YEAR_RE.fullmatch(token):
            continue
        found.append((span[0], span[1], _parse_stat_number(token)))
    return found


def _clause_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for match in _CLAUSE_SPLIT_RE.finditer(text):
        ranges.append((start, match.start()))
        start = match.end()
    ranges.append((start, len(text)))
    return ranges


def _range_prefixed(text: str, start: int) -> bool:
    return _RANGE_PREFIX_RE.search(text[max(0, start - 32) : start]) is not None


def _stated_as_total(text: str, start: int) -> bool:
    return _TOTAL_PHRASE_RE.search(text[max(0, start - 48) : start]) is not None


def _span_overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < other_end and other_start < end for other_start, other_end in occupied)


def _fold_index(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    origins: list[int] = []
    for index, char in enumerate(text):
        for piece in _fold_place(char):
            chars.append(piece)
            origins.append(index)
    return "".join(chars), origins


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


_DERIVED_BEFORE = {
    "difference": re.compile(
        r"(?:forskellen|forskel|difference|stigningen|stigning|faldet|fald"
        r"|vokset med|oegget med|øget med)\b"
        r"(?:[ \t]+\S+){0,3}[ \t]*$",
        re.IGNORECASE,
    ),
    "sum": re.compile(
        r"(?:i alt|in total|summen|sammenlagt|tilsammen)\b(?:[ \t]+\S+){0,3}[ \t]*$",
        re.IGNORECASE,
    ),
}
_PERCENT_AFTER = re.compile(r"^[ \t]*(?:procent|percentage|percent|pct\.?|%)\b", re.IGNORECASE)
_DIFF_AFTER = re.compile(
    r"^[ \t]*(?:flere|færre|faerre|større|stoerre|mindre)\b",
    re.IGNORECASE,
)


_AREA_FIELDS = frozenset({"omraade", "kommune", "blstkom", "kommunedk", "region", "landsdel"})


def _measure_of(fact: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    bits = []
    for part in str(fact.get("detail") or "").split(", "):
        name, sep, value = part.partition(": ")
        if sep and _fold_place(name) not in _AREA_FIELDS:
            bits.append(f"{name}={value}")
    return str(fact.get("unit") or ""), tuple(bits)


def _matches_derivation(
    text: str,
    start: int,
    end: int,
    number: float,
    facts: list[dict[str, Any]],
) -> bool:
    """One measure, named by this clause. A sum also has to be a single period."""
    kind = _derivation_kind(text, start, end)
    if kind is None:
        return False
    chosen = _referred_facts(text, start, facts)
    if chosen is None:
        chosen = [fact for fact in facts if _single_value(fact)]
        if kind == "sum" or len(chosen) != 2:
            return False
    if len({_measure_of(fact) for fact in chosen}) != 1 or len(chosen) < 2:
        return False
    if kind == "sum":
        if len({str(fact.get("period") or "") for fact in chosen}) != 1:
            return False
        return _close_enough(number, sum(fact["value"] for fact in chosen))
    if len(chosen) != 2:
        return False
    left, right = float(chosen[0]["value"]), float(chosen[1]["value"])
    if kind == "difference":
        return _close_enough(number, abs(left - right))
    return any(
        old and _close_enough(number, abs(left - right) / abs(old) * 100) for old in (left, right)
    )


def _referred_facts(
    text: str,
    start: int,
    facts: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    clause_start, clause_end = next(
        ((begin, end) for begin, end in _clause_ranges(text) if begin <= start <= end),
        (0, len(text)),
    )
    mentions = _period_mentions(text, facts)
    periods = {code for begin, end, code in mentions if clause_start <= begin and end <= clause_end}
    covered = [(begin, end) for begin, end, _code in mentions]
    labels = {
        label
        for begin, end, label in _label_mentions(text, facts, covered)
        if clause_start <= begin and end <= clause_end
    }
    quoted = {
        value
        for begin, end, value in _figure_mentions(text, covered)
        if clause_start <= begin
        and end <= clause_end
        and begin != start
        and _near(value, _row_values(facts))
    }
    if not periods and not labels and len(quoted) < 2:
        return None
    chosen = []
    for fact in facts:
        if not _single_value(fact):
            continue
        period = str(fact.get("period") or "").casefold()
        if periods and period not in periods:
            continue
        if labels and not (_fact_labels(fact) & labels):
            continue
        if not periods and not labels and not _near(float(fact["value"]), quoted):
            continue
        chosen.append(fact)
    return chosen


def _derivation_kind(text: str, start: int, end: int) -> str | None:
    before = text[max(0, start - 48) : start]
    after = text[end : end + 24]
    if _PERCENT_AFTER.search(after):
        return "percent"
    percent_before = re.search(
        r"(?:procent|percentage|percent|pct\.?|%)\b(?:[ \t]+\S+){0,2}[ \t]*$",
        before,
        re.IGNORECASE,
    )
    if percent_before:
        return "percent"
    if _DERIVED_BEFORE["difference"].search(before) or _DIFF_AFTER.search(after):
        return "difference"
    if _DERIVED_BEFORE["sum"].search(before):
        return "sum"
    return None


def _close_enough(stated: float, exact: float) -> bool:
    """Whole numbers within 0.5, one decimal within 0.05, and so on."""
    if abs(stated - exact) <= 1e-6:
        return True
    scale = 1.0
    for _ in range(4):
        if abs(stated * scale - round(stated * scale)) <= 1e-6:
            return abs(stated - exact) <= 0.5 / scale + 1e-6
        scale *= 10
    return False


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


def _model_result(name: str, result: str) -> str:
    """Keep the last row when a long fetch is shortened for the model."""
    if len(result) <= MODEL_RESULT_LIMIT:
        return result
    if name != "hent_data":
        return _short(result, MODEL_RESULT_LIMIT)
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return _short(result, MODEL_RESULT_LIMIT)
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) < 3:
        return _short(result, MODEL_RESULT_LIMIT)
    omitted = len(rows) - 2
    shown = dict(payload)
    shown["rows"] = [rows[0], rows[-1]]
    shown["rows_omitted"] = omitted
    first = rows[0].get("Tid") if isinstance(rows[0], dict) else ""
    last = rows[-1].get("Tid") if isinstance(rows[-1], dict) else ""
    shown["rows_note"] = (
        f"{omitted} rows omitted from this view. "
        f"First period {first}, last period {last}. Both values are included."
    )
    text = json.dumps(shown, ensure_ascii=False)
    if len(text) <= MODEL_RESULT_LIMIT:
        return text
    return _short(text, MODEL_RESULT_LIMIT)


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
    think: bool = False

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
        think=_env_flag("STATBANK_LLM_THINK"),
    )


def chat_client(settings: Settings, http: httpx.Client | None = None) -> ChatClient:
    # Ollama thinks unless this is none. xAI is left unchanged.
    effort = "none" if settings.llm == "ollama" and not settings.think else None
    return ChatClient(
        settings.base_url,
        settings.api_key,
        settings.model,
        settings.llm_timeout,
        http=http,
        max_tokens=settings.max_tokens,
        reasoning_effort=effort,
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


def _transient(exc: BaseException) -> bool:
    """One retry is worth it for a dropped model call or a dead MCP session."""
    for leaf in _unwrap(exc):
        if isinstance(leaf, (LLMError, TimeoutError, ConnectionError)):
            return True
        folded = type(leaf).__name__.casefold()
        if "timeout" in folded or "mcperror" in folded or folded.startswith("connect"):
            return True
    return False


def _answer_once(settings: Settings, question: str) -> TurnResult:
    async def _run() -> TurnResult:
        llm = chat_client(settings)
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


def answer_with_server(settings: Settings, question: str) -> TurnResult:
    if settings.llm == "xai" and not settings.api_key:
        raise LLMError("XAI_API_KEY mangler. Sæt den i miljøet eller i .env.")
    try:
        return _answer_once(settings, question)
    except Exception as exc:
        if not _transient(exc):
            raise
        return _answer_once(settings, question)


def check_settings(settings: Settings) -> None:
    """Refuse a public demo that would start without an xAI key or a proxy secret."""
    if settings.public and not settings.api_key:
        raise SystemExit(
            "XAI_API_KEY is missing. Set it in the environment before starting. "
            "Do not write it into a file."
        )
    if settings.public and not settings.proxy_secret.strip():
        raise SystemExit(
            "STATBANK_PROXY_SECRET is missing. Set it in the environment before starting. "
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
                    "suggestions": result.suggestions,
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
