"""Local web page that asks an LLM to call the Statistikbanken MCP server.

The model is any OpenAI-compatible chat-completions endpoint. Ollama and the
xAI API both accept the same tools payload on /v1/chat/completions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
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

SYSTEM_PROMPT = """You answer questions about Danish official statistics.
Use only the Statistikbanken tools. Do not invent numbers.
Call find_tabel, then tabelinfo when you need codes, then hent_data.
Always name the table id and the period in the answer.
If the table title, unit, or description only partly matches the question, say so.
If the tools have no data, say so.
Prefer sum_from or sum_range when a total is needed.
A value like >=80 lists rows. It does not add them.
Municipality names may be passed as labels.
Answer in the same language as the question, in a few sentences."""

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

function render(data) {
  out.replaceChildren();
  if (data.error) {
    out.append(el("p", data.error, "error"));
    return;
  }
  const answer = el("section");
  answer.append(el("h2", "Svar"), el("p", data.answer || "", "answer"));
  out.append(answer);

  const facts = el("section");
  facts.append(el("h2", "Tal"));
  if (!data.facts || data.facts.length === 0) {
    facts.append(el("p", "Ingen tal hentet.", "meta"));
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
    facts.append(card);
  }
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
    ]
    calls: list[dict[str, str]] = []
    facts: list[dict[str, Any]] = []
    used = 0
    while True:
        allow_tools = used < max_tool_calls
        reply = await asyncio.to_thread(llm.complete, messages, tools if allow_tools else None)
        if not reply.tool_calls or not allow_tools:
            answer = reply.content.strip() or _fallback_answer(calls)
            return TurnResult(answer=answer, facts=facts, calls=calls)
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
            answer = final.content.strip() or _fallback_answer(calls)
            return TurnResult(answer=answer, facts=facts, calls=calls)


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


def _fallback_answer(calls: list[dict[str, str]]) -> str:
    if not calls:
        return "Jeg fandt ingen tal at svare ud fra."
    return "Jeg nåede grænsen for opslag og kan ikke svare færdigt. Se værktøjskaldene nedenfor."


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
            }
        )
    return facts


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
    timeout: float

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
        timeout=float(os.environ.get("STATBANK_TIMEOUT", "120")),
    )


def server_command() -> StdioServerParameters:
    return StdioServerParameters(command=sys.executable, args=["-m", "statistikbanken_mcp"])


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
        llm = ChatClient(settings.base_url, settings.api_key, settings.model, settings.timeout)
        try:
            async with Client(server_command(), read_timeout_seconds=settings.timeout) as client:
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
                self._json(502, {"error": str(exc)})
                return
            except Exception:
                sys.stderr.write("Question failed.\n")
                self._json(502, {"error": "Kunne ikke hente et svar."})
                return
            self._json(
                200,
                {"answer": result.answer, "facts": result.facts, "calls": result.calls},
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
