"""Score one answered question against a truth fetched from the raw API.

The live runner is separate. These functions take plain dicts so a unit test
can exercise every outcome without calling xAI or Statistikbanken.
"""

from __future__ import annotations

import json
import os
import re
import statistics
from dataclasses import dataclass
from typing import Any

from statistikbanken_mcp.web import (
    _NUMBER_RE,
    LLM_TIMEOUT_TEXT,
    NO_DATA_ANSWER,
    NO_FETCHED_NUMBERS,
    UNVERIFIED_NUMBERS,
)

# https://docs.x.ai/docs/models — grok-4 is served as grok-4.3 (3 Nov 2026).
# Text tokens, prompt under 200k. Reasoning is billed as output.
PRICE_SOURCE = "https://docs.x.ai/docs/models"
DEFAULT_INPUT_PER_M = 1.25
DEFAULT_CACHED_PER_M = 0.20
DEFAULT_OUTPUT_PER_M = 2.50

OUTCOMES = (
    "correct",
    "wrong_table",
    "wrong_number",
    "wrong_period",
    "guard_hid_correct_answer",
    "no_answer",
    "timeout_error",
    "edge_behaviour_wrong",
)
OUTCOME_LABEL = {
    "correct": "correct",
    "wrong_table": "wrong_table",
    "wrong_number": "wrong_number",
    "wrong_period": "wrong_period",
    "guard_hid_correct_answer": "guard_hid_correct_answer",
    "no_answer": "no_answer",
    "timeout_error": "timeout/error",
    "edge_behaviour_wrong": "edge_behaviour_wrong",
}

_PLACEHOLDERS = frozenset(
    " ".join(text.split()) for text in (UNVERIFIED_NUMBERS, NO_FETCHED_NUMBERS, NO_DATA_ANSWER)
)
# A refusal has to say the figure is missing. A bare "ikke" is too common.
_GAP_PHRASES = (
    "ingen",
    "findes ikke",
    "ikke i statistikbanken",
    "ikke offentliggjort",
    "not published",
    "har ikke",
    "kan ikke",
    "ikke finde",
    "ikke en ",
    "ikke et ",
    "ukendt",
    "not available",
    "no data",
    "no table",
    "does not exist",
    "not in stat",
)
_UNPUBLISHED_PHRASES = (
    "ikke offentliggjort",
    "not published",
    "ikke i tabellen",
    "findes ikke",
    "går ikke",
    "gaar ikke",
    "uden for",
    "ikke tilgængelig",
    "ikke tilgaengelig",
    "starter",
    "rækker ikke",
    "raekker ikke",
    "ikke med i",
    "ingen tal",
    "not in the table",
    "before the",
    "kan ikke",
)
_LOWER_BOUND_PHRASES = (
    "mindst",
    "nedre",
    "lavere grænse",
    "lavere graense",
    "ikke alle",
    "ikke præcis",
    "ikke praecis",
    "ikke præcist",
    "ikke praecist",
    "tilnærm",
    "tilnaerm",
    "cirka",
    "ca.",
    "ufuldstændig",
    "ufuldstaendig",
    "og derover",
    "6+",
    "6 personer",
    "topkod",
    "ikke det fulde",
    "ikke hele",
    "kun de husstande",
    "kan ikke opgøres",
    "kan ikke opgoeres",
    "lower bound",
    "at least",
)


@dataclass(frozen=True)
class Truth:
    """Figures the raw API returns for one acceptable selection."""

    table: str
    values: list[float]
    periods: list[str]
    names: tuple[str, ...] = ()


@dataclass(frozen=True)
class Prices:
    input_per_m: float
    cached_per_m: float
    output_per_m: float
    source: str
    from_env: bool


def prices_from_env() -> Prices:
    """List prices, unless the environment overrides them."""
    overrides = {
        "input_per_m": _env_price("XAI_PRICE_INPUT_PER_M"),
        "cached_per_m": _env_price("XAI_PRICE_CACHED_PER_M"),
        "output_per_m": _env_price("XAI_PRICE_OUTPUT_PER_M"),
    }
    from_env = any(value is not None for value in overrides.values())
    return Prices(
        input_per_m=overrides["input_per_m"]
        if overrides["input_per_m"] is not None
        else DEFAULT_INPUT_PER_M,
        cached_per_m=overrides["cached_per_m"]
        if overrides["cached_per_m"] is not None
        else DEFAULT_CACHED_PER_M,
        output_per_m=overrides["output_per_m"]
        if overrides["output_per_m"] is not None
        else DEFAULT_OUTPUT_PER_M,
        source=PRICE_SOURCE,
        from_env=from_env,
    )


def _env_price(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return float(raw)


def cost_usd(
    *,
    prompt: int,
    cached: int,
    completion: int,
    reasoning: int,
    prices: Prices,
) -> float:
    """Cached tokens are a subset of prompt tokens. Reasoning is output."""
    fresh = max(int(prompt) - int(cached), 0)
    output = int(completion) + int(reasoning)
    return (
        fresh * prices.input_per_m / 1_000_000
        + int(cached) * prices.cached_per_m / 1_000_000
        + output * prices.output_per_m / 1_000_000
    )


def parse_numbers(text: str) -> list[float]:
    """Figures in the answer, Danish or English grouping.

    A hyphen between two digits is a range (0-6), not a minus. A leading
    minus on its own is kept, so a price fall of -10,2 still counts.
    """
    found: list[float] = []
    for match in _NUMBER_RE.finditer(text or ""):
        value = _parse_token(match.group(1))
        if value is None:
            continue
        start = match.start(1)
        before = text[start - 1] if start else ""
        earlier = text[start - 2] if start >= 2 else ""
        # "" in "-−" is true in Python, so membership has to be a set of characters.
        if before in {"-", "−"} and not earlier.isdigit():
            value = -value
        found.append(value)
    return found


def _parse_token(token: str) -> float | None:
    text = token.replace("\u00a0", "").replace(" ", "")
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        parts = text.split(",")
        if len(parts) > 1 and all(len(part) == 3 for part in parts[1:]):
            text = "".join(parts)
        else:
            text = text.replace(",", ".")
    elif "." in text:
        parts = text.split(".")
        if len(parts) > 1 and all(len(part) == 3 for part in parts[1:]):
            text = "".join(parts)
    try:
        return float(text)
    except ValueError:
        return None


def close(expected: float, got: float) -> bool:
    """Formatting and ordinary rounding. Not a different official figure."""
    return abs(expected - got) <= max(0.05, abs(expected) * 0.001)


def _same_kroner_scale(expected: float, got: float, text: str) -> bool:
    """3.118.538 mio. kr. and 3.118,5 mia. kr. are one national-accounts figure."""
    folded = text.casefold()
    if not any(word in folded for word in ("mio", "mia", "million", "milliard")):
        return False
    if not expected or not got:
        return False
    ratio = abs(got / expected)
    return abs(ratio - 1000) <= 1 or abs(ratio - 0.001) <= 1e-6


def covers(expected: list[float], stated: list[float], text: str = "") -> bool:
    """Every expected figure has its own stated number. Extras are allowed."""
    if not expected:
        return False
    pool = list(stated)
    for value in expected:
        for index, got in enumerate(pool):
            if close(value, got) or (text and _same_kroner_scale(value, got, text)):
                pool.pop(index)
                break
        else:
            return False
    return True


def classify(question: dict[str, Any], result: dict[str, Any], truths: list[Truth]) -> str:
    """One taxonomy label. ``truths`` is empty for a pure behaviour question."""
    if result.get("error") or result.get("timeout"):
        return "timeout_error"
    answer = str(result.get("answer") or "")
    if LLM_TIMEOUT_TEXT in answer:
        return "timeout_error"
    kind = str((question.get("expect") or {}).get("kind") or "figures")
    if kind == "figures":
        return _score_figures(question, result, truths)
    if kind == "partial":
        return _score_partial(result, truths)
    if kind == "approximate":
        return _score_approximate(question, result, truths)
    if kind == "answer_or_clarify":
        return _score_clarify(question, result, truths)
    if kind == "not_published":
        return _score_not_published(result)
    if kind == "too_broad":
        return _score_too_broad(question, result)
    if kind in {"no_table", "not_available"}:
        return _score_gap(question, result)
    if kind == "coverage_gap":
        return _score_coverage_gap(question, result, truths)
    return "edge_behaviour_wrong"


def _figure_match(truth: Truth, stated: list[float], answer: str, text: str = "") -> bool:
    """Stocks, and for a rank the place names in order and the geography type."""
    if not covers(truth.values, stated, text):
        return False
    if not truth.names:
        return True
    return _names_in_order(truth.names, answer) and _names_urban_area(answer)


def _names_in_order(names: tuple[str, ...], answer: str) -> bool:
    folded = answer.casefold()
    cursor = 0
    for name in names:
        index = folded.find(name.casefold(), cursor)
        if index < 0:
            return False
        cursor = index + len(name)
    return True


def _names_urban_area(answer: str) -> bool:
    folded = answer.casefold()
    return "byområde" in folded or "byomraade" in folded


def _phrases_ok(question: dict[str, Any], result: dict[str, Any]) -> bool:
    """Each require_all group needs one of its phrases. Groups are AND."""
    text = _visible(result).casefold()
    for group in (question.get("expect") or {}).get("require_all") or []:
        phrases = [str(phrase).casefold() for phrase in group]
        if phrases and not any(phrase in text for phrase in phrases):
            return False
    return True


def _score_figures(question: dict[str, Any], result: dict[str, Any], truths: list[Truth]) -> str:
    if not truths:
        return "timeout_error"
    stated = _stated(result)
    facts = _fact_values(result)
    answer = str(result.get("answer") or "")
    if any(_figure_match(truth, stated, answer, text=answer) for truth in truths):
        if not _phrases_ok(question, result):
            return "edge_behaviour_wrong"
        return "correct"
    hidden = _is_placeholder(str(result.get("answer") or "")) or result.get("show_numbers") is False
    if hidden and any(covers(truth.values, facts) for truth in truths):
        return "guard_hid_correct_answer"
    if not stated and not facts:
        return "no_answer"
    used = _tables_used(result)
    if used.isdisjoint({truth.table.casefold() for truth in truths}):
        return "wrong_table"
    if _stated_period_is_wrong(result, truths, stated):
        return "wrong_period"
    return "wrong_number"


def _score_partial(result: dict[str, Any], truths: list[Truth]) -> str:
    """Danish figure plus an explicit 'Sweden is not covered'. No second figure."""
    if not truths:
        return "timeout_error"
    stated = _stated(result)
    facts = _fact_values(result)
    answer = _visible(result)
    if _is_placeholder(str(result.get("answer") or "")) and any(
        covers(t.values, facts) for t in truths
    ):
        return "guard_hid_correct_answer"
    has_figure = any(covers(truth.values, stated) for truth in truths)
    if _extra_large(stated, truths):
        return "edge_behaviour_wrong"
    if has_figure and _sweden_caveat(answer):
        return "correct"
    if not stated and not facts:
        return "no_answer"
    return "edge_behaviour_wrong"


# A generic "ingen tal" give-up is not an explanation that the exact count is unpublished.
_EXACT_UNPUBLISHED = (
    "ikke opgjort",
    "kan ikke opgøres",
    "kan ikke opgoeres",
    "ikke præcis",
    "ikke praecis",
    "det præcise",
    "det praecise",
    "ikke en kategori",
    "topkod",
    "exact count",
    "ikke offentliggjort",
)


def _score_approximate(
    question: dict[str, Any],
    result: dict[str, Any],
    truths: list[Truth],
) -> str:
    """A lower bound with a caveat, or a specific reason the exact count is unpublished.

    The fixed no-data sentence is not that reason. It also covers a search miss.
    """
    if not truths:
        return "timeout_error"
    stated = _stated(result)
    text = _visible(result).casefold()
    invented = _statistics(str(result.get("answer") or ""), str(question.get("question") or ""))
    if invented or _shown_facts(result):
        has_bound = any(covers(truth.values, stated) for truth in truths)
        if has_bound and any(phrase in text for phrase in _LOWER_BOUND_PHRASES):
            return "correct"
        return "edge_behaviour_wrong"
    if any(phrase in text for phrase in _EXACT_UNPUBLISHED):
        return "correct"
    return "edge_behaviour_wrong"


def _score_clarify(question: dict[str, Any], result: dict[str, Any], truths: list[Truth]) -> str:
    stated = _stated(result)
    facts = _fact_values(result) if result.get("show_numbers") is not False else []
    if truths and any(
        covers(truth.values, stated) or covers(truth.values, facts) for truth in truths
    ):
        return "correct"
    text = _visible(result).casefold().replace("æ", "ae")
    named = "lyngby-taarbaek" in text or "lyngby taarbaek" in text
    invented = _statistics(str(result.get("answer") or ""), str(question.get("question") or ""))
    if named and not invented and not _shown_facts(result):
        return "correct"
    return "edge_behaviour_wrong"


def _score_not_published(result: dict[str, Any]) -> str:
    answer = str(result.get("answer") or "")
    text = _visible(result).casefold()
    if _is_placeholder(answer) and " ".join(answer.split()) == " ".join(NO_FETCHED_NUMBERS.split()):
        return "correct"
    if _is_no_data(answer) or any(phrase in text for phrase in _UNPUBLISHED_PHRASES):
        return "correct"
    return "edge_behaviour_wrong"


def _score_too_broad(question: dict[str, Any], result: dict[str, Any]) -> str:
    if _shown_facts(result):
        return "edge_behaviour_wrong"
    if _statistics(str(result.get("answer") or ""), str(question.get("question") or "")):
        return "edge_behaviour_wrong"
    return "correct"


def _without_period_labels(text: str) -> str:
    """Drop quarter labels so '1. kvt.' is not a second statistic."""
    text = re.sub(r"\b(?:19|20)\d{2}[KkQq][1-4]\b", " ", text)
    text = re.sub(
        r"\b(?:19|20)\d{2}\s+[1-4]\s*\.\s*kvt\.?",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b[1-4]\s*\.\s*kvartal\s+(?:19|20)\d{2}\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _score_coverage_gap(
    question: dict[str, Any], result: dict[str, Any], truths: list[Truth]
) -> str:
    """An explanation of a missing period. The generic no-data sentence is not enough."""
    answer = str(result.get("answer") or "")
    if _is_placeholder(answer) or _is_no_data(answer):
        return "edge_behaviour_wrong"
    text = _visible(result).casefold()
    required = [
        str(phrase).casefold() for phrase in (question.get("expect") or {}).get("require_any") or []
    ]
    if required and not any(phrase in text for phrase in required):
        return "edge_behaviour_wrong"
    for group in (question.get("expect") or {}).get("require_all") or []:
        phrases = [str(phrase).casefold() for phrase in group]
        if phrases and not any(phrase in text for phrase in phrases):
            return "edge_behaviour_wrong"
    if _denies_fetched_period(text, truths):
        return "edge_behaviour_wrong"
    stated = _statistics(_without_period_labels(answer), str(question.get("question") or ""))
    if not stated or not truths:
        return "correct"
    stocks = [value for truth in truths for value in truth.values]
    allowed = list(stocks)
    if len(stocks) >= 2 and stocks[0]:
        difference = abs(stocks[-1] - stocks[0])
        allowed.append(difference)
        allowed.append(difference / abs(stocks[0]) * 100)
    for number in stated:
        if not any(close(number, value) for value in allowed):
            return "edge_behaviour_wrong"
    return "correct"


def _denies_fetched_period(text: str, truths: list[Truth]) -> bool:
    """A sentence that says a period we scored is missing from the answer."""
    periods = [period for truth in truths for period in truth.periods]
    for sentence in re.split(r"\n+|(?<=[.!?])\s+", text):
        if not re.search(r"ikke med|ikke vist|afkortede|afkortet", sentence, re.IGNORECASE):
            continue
        folded = sentence.casefold()
        if any(period.casefold() in folded for period in periods):
            return True
    return False


def _score_gap(question: dict[str, Any], result: dict[str, Any]) -> str:
    if _shown_facts(result):
        return "edge_behaviour_wrong"
    if _statistics(str(result.get("answer") or ""), str(question.get("question") or "")):
        return "edge_behaviour_wrong"
    text = _visible(result).casefold()
    if _is_no_data(str(result.get("answer") or "")) or any(
        phrase in text for phrase in _GAP_PHRASES
    ):
        return "correct"
    return "edge_behaviour_wrong"


def _stated(result: dict[str, Any]) -> list[float]:
    answer = str(result.get("answer") or "")
    if _is_placeholder(answer):
        return []
    return parse_numbers(answer)


def _is_placeholder(answer: str) -> bool:
    return " ".join(answer.split()) in _PLACEHOLDERS


def _is_no_data(answer: str) -> bool:
    text = " ".join(answer.split())
    return text == " ".join(NO_DATA_ANSWER.split()) or "fandt ingen tal" in text.casefold()


def _fact_values(result: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for fact in result.get("facts") or []:
        if not isinstance(fact, dict):
            continue
        value = fact.get("value")
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        values.append(float(value))
    return values


def _shown_facts(result: dict[str, Any]) -> list[float]:
    if result.get("show_numbers") is False:
        return []
    return _fact_values(result)


def _statistics(answer: str, question: str) -> list[float]:
    """Numbers that claim a measurement. Years and figures copied from the question are not."""
    asked = parse_numbers(question)
    kept: list[float] = []
    for number in parse_numbers(answer):
        if number == int(number) and 1800 <= number <= 2100:
            continue
        if any(close(number, item) for item in asked):
            continue
        kept.append(number)
    return kept


def _tables_used(result: dict[str, Any]) -> set[str]:
    tables: set[str] = set()
    for fact in result.get("facts") or []:
        if isinstance(fact, dict) and fact.get("table"):
            tables.add(str(fact["table"]).casefold())
    for call in result.get("calls") or []:
        if not isinstance(call, dict) or call.get("name") != "hent_data":
            continue
        raw = call.get("arguments")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                continue
        if isinstance(raw, dict) and raw.get("table"):
            tables.add(str(raw["table"]).casefold())
    return tables


def _stated_period_is_wrong(
    result: dict[str, Any],
    truths: list[Truth],
    stated: list[float],
) -> bool:
    if not stated:
        return False
    for truth in truths:
        allowed = {period.casefold() for period in truth.periods}
        matched: list[str] = []
        for fact in result.get("facts") or []:
            if not isinstance(fact, dict):
                continue
            if str(fact.get("table") or "").casefold() != truth.table.casefold():
                continue
            value = fact.get("value")
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            if any(close(float(value), number) for number in stated):
                matched.append(str(fact.get("period") or ""))
        if matched and all(period.casefold() not in allowed for period in matched):
            return True
    return False


def _sweden_caveat(text: str) -> bool:
    folded = text.casefold()
    if "sverige" not in folded and "sweden" not in folded:
        return False
    return any(
        phrase in folded
        for phrase in (
            "ikke",
            "not ",
            "uden for",
            "udenfor",
            "dækker",
            "daekker",
            "findes ikke",
            "ingen tal",
            "omfatter ikke",
            "not covered",
            "no data",
        )
    )


def _extra_large(stated: list[float], truths: list[Truth]) -> bool:
    expected = [value for truth in truths for value in truth.values]
    for number in stated:
        if any(close(value, number) for value in expected):
            continue
        if number == int(number) and 1800 <= number <= 2100:
            continue
        if abs(number) >= 1000:
            return True
    return False


def _visible(result: dict[str, Any]) -> str:
    parts = [str(result.get("answer") or "")]
    parts.extend(str(item) for item in result.get("suggestions") or [])
    return "\n".join(parts)


def render_report(records: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    """Markdown report. ``records`` are the per-question rows the runner stored."""
    total = len(records)
    counts = {outcome: 0 for outcome in OUTCOMES}
    for record in records:
        counts[str(record.get("outcome") or "timeout_error")] = (
            counts.get(str(record.get("outcome")), 0) + 1
        )
    correct = counts.get("correct", 0)
    prices: Prices = meta["prices"]
    lines = [
        f"# Evaluation {meta.get('date')} {meta.get('model')}",
        "",
        f"**{correct}/{total} correct**",
        "",
        f"Model requested: `{meta.get('model')}`. "
        f"Served as: `{meta.get('served_model') or meta.get('model')}`.",
        f"Questions: {total} of {meta.get('set_size')} in the set. "
        "Path: `answer_with_server` with `--llm xai`.",
        "",
        "## Taxonomy",
        "",
        "| Outcome | Count |",
        "|---|---:|",
    ]
    for outcome in OUTCOMES:
        lines.append(f"| {OUTCOME_LABEL[outcome]} | {counts.get(outcome, 0)} |")
    lines.extend(
        ["", "## By question category", "", "| Category | Questions | Correct |", "|---|---:|---:|"]
    )
    by_category: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_category.setdefault(str(record.get("category") or ""), []).append(record)
    for category in sorted(by_category):
        rows = by_category[category]
        got = sum(1 for row in rows if row.get("outcome") == "correct")
        lines.append(f"| {category} | {len(rows)} | {got} |")
    prompt = sum(int(row.get("prompt_tokens") or 0) for row in records)
    cached = sum(int(row.get("cached_tokens") or 0) for row in records)
    completion = sum(int(row.get("completion_tokens") or 0) for row in records)
    reasoning = sum(int(row.get("reasoning_tokens") or 0) for row in records)
    spent = cost_usd(
        prompt=prompt,
        cached=cached,
        completion=completion,
        reasoning=reasoning,
        prices=prices,
    )
    latencies = [float(row["latency_s"]) for row in records if row.get("latency_s") is not None]
    median = statistics.median(latencies) if latencies else 0.0
    fresh = max(prompt - cached, 0)
    output = completion + reasoning
    price_note = "environment overrides" if prices.from_env else prices.source
    lines.extend(
        [
            "",
            "## Tokens and cost",
            "",
            f"Prompt {prompt:,} (cached {cached:,}), completion {completion:,}, "
            f"reasoning {reasoning:,}.",
            f"Billable input {fresh:,} × ${prices.input_per_m}/M, "
            f"cached {cached:,} × ${prices.cached_per_m}/M, "
            f"output {output:,} × ${prices.output_per_m}/M (reasoning counted as output).",
            f"Cost: ${spent:.4f}. Median latency: {median:.1f} s.",
            f"Prices: {price_note}.",
            "",
            "## Notable failures",
            "",
        ]
    )
    failures = [row for row in records if row.get("outcome") != "correct"]
    if not failures:
        lines.append("None.")
    else:
        shown = failures[:15]
        for row in shown:
            answer = " ".join(str(row.get("answer") or "").split())
            if len(answer) > 180:
                answer = answer[:177].rstrip() + "..."
            lines.append(
                f"- Q{row.get('id')} {OUTCOME_LABEL.get(str(row.get('outcome')), '')}: "
                f"{answer or '(empty)'}"
            )
        if len(failures) > len(shown):
            lines.append(f"- {len(failures) - len(shown)} more in the table below.")
    owner = [row for row in records if row.get("source") == "owner"]
    if owner:
        lines.extend(["", "## Owner questions", ""])
        for row in owner:
            note = " ".join(str(row.get("note") or "").split())
            lines.append(
                f"- Q{row.get('id')} {OUTCOME_LABEL.get(str(row.get('outcome')), '')}. {note}"
            )
    lines.extend(
        [
            "",
            "## Per question",
            "",
            "| ID | Category | Outcome | Tables | Latency | Answer |",
            "|---:|---|---|---|---:|---|",
        ]
    )
    for row in records:
        answer = " ".join(str(row.get("answer") or "").split()).replace("|", "/")
        if len(answer) > 80:
            answer = answer[:77].rstrip() + "..."
        tables = ", ".join(row.get("tables") or []) or "—"
        latency = row.get("latency_s")
        latency_text = f"{float(latency):.1f}s" if latency is not None else ""
        lines.append(
            f"| {row.get('id')} | {row.get('category')} | "
            f"{OUTCOME_LABEL.get(str(row.get('outcome')), '')} | {tables} | "
            f"{latency_text} | {answer} |"
        )
    lines.append("")
    return "\n".join(lines)
