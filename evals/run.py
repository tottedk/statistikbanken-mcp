"""Run the regression set through the web UI's xAI path.

Not part of pytest or CI. A full run calls the live model and costs money.
The API key is read from the environment and is never printed or written out.

    uv run python -m evals.run
    uv run python -m evals.run --limit 5 --ids 63,101-105 --model grok-4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from evals.score import classify, cost_usd, parse_numbers, prices_from_env, render_report
from evals.verify import VerifierError, truths_for
from statistikbanken_mcp import web
from statistikbanken_mcp.web import LLMError

ROOT = Path(__file__).resolve().parent
QUESTIONS = ROOT / "questions.json"
REPORTS = ROOT / "reports"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    questions = _load()
    chosen = _select(questions, limit=args.limit, ids=args.ids)
    if not chosen:
        print("No questions selected.", file=sys.stderr)
        return 2
    model_args = ["--llm", "xai"]
    if args.model:
        model_args.extend(["--model", args.model])
    settings = web.settings_from_env(model_args)
    if not settings.api_key:
        print(
            "XAI_API_KEY is missing. Set it in the environment. It is not read from the repo.",
            file=sys.stderr,
        )
        return 2
    prices = prices_from_env()
    records: list[dict[str, Any]] = []
    served: list[str] = []
    for question in chosen:
        record = _run_one(settings, question)
        records.append(record)
        if record.get("served_model"):
            served.append(str(record["served_model"]))
        print(
            f"Q{record['id']} {record['outcome']} {record.get('latency_s')}s "
            f"tools={record.get('tool_calls')}",
            flush=True,
        )
    day = datetime.now(ZoneInfo("Europe/Copenhagen")).date().isoformat()
    served_model = served[-1] if served else settings.model
    meta = {
        "date": day,
        "model": settings.model,
        "served_model": served_model,
        "set_size": len(questions),
        "prices": prices,
    }
    markdown = render_report(records, meta)
    out = Path(args.out) if args.out else REPORTS / f"{day}-{_safe(settings.model)}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    jsonl = out.with_suffix(".jsonl")
    with jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    correct = sum(1 for record in records if record["outcome"] == "correct")
    spent = cost_usd(
        prompt=sum(int(r.get("prompt_tokens") or 0) for r in records),
        cached=sum(int(r.get("cached_tokens") or 0) for r in records),
        completion=sum(int(r.get("completion_tokens") or 0) for r in records),
        reasoning=sum(int(r.get("reasoning_tokens") or 0) for r in records),
        prices=prices,
    )
    print(f"{correct}/{len(records)} correct  ${spent:.4f}  {out}", flush=True)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Statistikbanken regression set")
    parser.add_argument("--limit", type=int, help="run at most this many questions after --ids")
    parser.add_argument("--ids", default="", help="ids to run, for example 63,101-105")
    parser.add_argument("--model", help="xAI model. Default is STATBANK_XAI_MODEL or grok-4")
    parser.add_argument("--out", help="markdown report path. A .jsonl file is written beside it")
    return parser


def _load() -> list[dict[str, Any]]:
    payload = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit("evals/questions.json must be a list")
    return payload


def _select(
    questions: list[dict[str, Any]], *, limit: int | None, ids: str
) -> list[dict[str, Any]]:
    chosen = questions
    wanted = _parse_ids(ids)
    if wanted:
        chosen = [item for item in questions if int(item["id"]) in wanted]
    chosen = sorted(chosen, key=lambda item: int(item["id"]))
    if limit is not None:
        chosen = chosen[:limit]
    return chosen


def _parse_ids(text: str) -> set[int]:
    wanted: set[int] = set()
    for part in text.split(","):
        piece = part.strip()
        if not piece:
            continue
        if "-" in piece:
            start, end = piece.split("-", 1)
            wanted.update(range(int(start), int(end) + 1))
        else:
            wanted.add(int(piece))
    return wanted


def _run_one(settings: web.Settings, question: dict[str, Any]) -> dict[str, Any]:
    rounds: list[dict[str, Any]] = []
    original = web._parse_reply

    def parse(payload: dict[str, Any]) -> Any:
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else {}
        completion = usage.get("completion_tokens_details") if isinstance(usage, dict) else {}
        rounds.append(
            {
                "model": payload.get("model") if isinstance(payload, dict) else None,
                "prompt_tokens": int((usage or {}).get("prompt_tokens") or 0),
                "cached_tokens": int((details or {}).get("cached_tokens") or 0)
                if isinstance(details, dict)
                else 0,
                "completion_tokens": int((usage or {}).get("completion_tokens") or 0),
                "reasoning_tokens": int((completion or {}).get("reasoning_tokens") or 0)
                if isinstance(completion, dict)
                else 0,
                "cost_usd_ticks": int((usage or {}).get("cost_in_usd_ticks") or 0),
            }
        )
        return original(payload)

    web._parse_reply = parse
    started = time.perf_counter()
    error: str | None = None
    result: dict[str, Any] = {}
    try:
        try:
            answer = web.answer_with_server(settings, str(question["question"]))
            result = dataclasses.asdict(answer)
        except LLMError as exc:
            error = _redact(str(exc))
        except Exception as exc:
            error = _redact(f"{type(exc).__name__}: {exc}")
    finally:
        web._parse_reply = original
    latency = round(time.perf_counter() - started, 2)
    truths = []
    if error is None and (question.get("expect") or {}).get("checks"):
        try:
            truths = truths_for(question)
        except VerifierError as exc:
            error = f"verifier: {exc}"
    outcome = "timeout_error" if error else classify(question, result, truths)
    prompt = sum(item["prompt_tokens"] for item in rounds)
    cached = sum(item["cached_tokens"] for item in rounds)
    completion = sum(item["completion_tokens"] for item in rounds)
    reasoning = sum(item["reasoning_tokens"] for item in rounds)
    tables = sorted(
        {str(fact.get("table")) for fact in result.get("facts") or [] if fact.get("table")}
    )
    return {
        "id": question["id"],
        "question": question["question"],
        "category": question.get("category"),
        "source": question.get("source") or "regression",
        "note": question.get("note") or "",
        "outcome": outcome,
        "error": error,
        "answer": result.get("answer"),
        "show_numbers": result.get("show_numbers", True),
        "suggestions": result.get("suggestions") or [],
        "facts": [
            {
                "table": fact.get("table"),
                "period": fact.get("period"),
                "value": fact.get("value"),
                "detail": fact.get("detail"),
            }
            for fact in result.get("facts") or []
            if isinstance(fact, dict)
        ],
        "calls": result.get("calls") or [],
        "tables": tables,
        "figures_shown": _figures(result),
        "guard_hidden": outcome == "guard_hid_correct_answer",
        "expected": [
            {"table": truth.table, "values": truth.values, "periods": truth.periods}
            for truth in truths
        ],
        "latency_s": latency,
        "llm_rounds": len(rounds),
        "tool_calls": len(result.get("calls") or []),
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "cost_usd_ticks": sum(item["cost_usd_ticks"] for item in rounds),
        "served_model": next((item["model"] for item in rounds if item.get("model")), None),
    }


def _figures(result: dict[str, Any]) -> list[float]:
    return parse_numbers(str(result.get("answer") or ""))


def _redact(text: str) -> str:
    secret = os.environ.get("XAI_API_KEY", "").strip()
    if secret:
        text = text.replace(secret, "[redacted]")
    return text


def _safe(model: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-._" else "-" for char in model)
    return cleaned or "model"


if __name__ == "__main__":
    raise SystemExit(main())
