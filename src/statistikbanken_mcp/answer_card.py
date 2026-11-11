"""A fixed answer card, checked against the rows that were fetched.

The model may send the card. The same checks accept a card built from those
rows. A share's denominator has to be the total for that table and period,
not the other part.
"""

from __future__ import annotations

import json
import re
from typing import Any

_PLACE_RE = re.compile(
    r"([A-Za-zÆØÅæøå][A-Za-zÆØÅæøå-]*)\s+(?:kommune|municipality)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d{1,3}(?:\.\d{3})+|\d+")
_PERCENT_WORD = re.compile(r"%|procent|percent|\bpct\b", re.IGNORECASE)
_VALUE_SUFFIX = re.compile(
    r"(?:\s*%|\s+pct\.?|\s+procent|\s+percent|\s+personer|\s+antal|\s+kr\.?|\s+kroner|\s+indeks)\s*$",
    re.IGNORECASE,
)


def decompose(question: str) -> list[str]:
    """One sub-question per named municipality, or per question mark."""
    seen: list[str] = []
    for place in _PLACE_RE.findall(question):
        if place.casefold() not in {item.casefold() for item in seen}:
            seen.append(place)
    if len(seen) >= 2:
        parts: list[str] = []
        for place in seen:
            text = question
            for other in seen:
                if other.casefold() == place.casefold():
                    continue
                text = re.sub(
                    rf"{re.escape(other)}\s+(?:kommune|municipality)",
                    "",
                    text,
                    flags=re.IGNORECASE,
                )
            parts.append(" ".join(text.split()))
        return parts
    chunks = [part.strip() for part in re.split(r"\?\s+", question) if part.strip()]
    if len(chunks) >= 2:
        return [part if part.endswith("?") else f"{part}?" for part in chunks]
    return [question]


def allocate_budget(parts: int, cap: int) -> list[int]:
    """Split the remaining tool cap across parts. A part needs at least one call.

    When there are more parts than calls, the extra parts get 0. The caller
    says which part was not fetched. The overall cap is not raised.
    """
    if parts <= 0:
        return []
    if cap <= 0:
        return [0] * parts
    if cap < parts:
        return [1 if index < cap else 0 for index in range(parts)]
    base, extra = divmod(cap, parts)
    return [base + (1 if index < extra else 0) for index in range(parts)]


def parse_card(text: str) -> dict[str, Any] | None:
    """The first JSON object that carries premise_check, or None."""
    if "premise_check" not in text:
        return None
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "premise_check" in payload:
            return payload
    return None


def build_card(question: str, facts: list[dict[str, Any]]) -> dict[str, Any]:
    """A card that only restates the fetched rows."""
    numeric = [fact for fact in facts if _number(fact.get("value")) is not None]
    tables = _unique(str(fact.get("table") or "") for fact in numeric)
    periods = _unique(str(fact.get("period") or "") for fact in numeric)
    units = _unique(str(fact.get("unit") or "") for fact in numeric)
    values = [_number(fact["value"]) for fact in numeric]
    parts = [
        {
            "part": str(fact.get("detail") or fact.get("period") or "tal"),
            "value": _number(fact["value"]),
            "missing": None,
        }
        for fact in numeric
    ]
    if not parts:
        parts = [{"part": question, "value": None, "missing": "ingen hentede tal"}]
    premise = "na"
    figure = None
    for claimed in _claimed_numbers(question):
        if values and not any(_close(claimed, value) for value in values):
            premise = "false"
            figure = values[0]
            break
    return {
        "premise_check": premise,
        "premise_figure": figure,
        "table_id": tables[0] if tables else "",
        "periods": [period for period in periods if period],
        "unit": units[0] if units else "",
        "values": values,
        "sub_parts": parts,
        "share": None,
    }


def verify_card(card: dict[str, Any], facts: list[dict[str, Any]]) -> list[str]:
    """Concrete errors. An empty list means the card may be rendered."""
    errors: list[str] = []
    numeric = [fact for fact in facts if _number(fact.get("value")) is not None]
    fetched = [_number(fact["value"]) for fact in numeric]
    tables = {str(fact.get("table") or "") for fact in numeric}
    periods = {str(fact.get("period") or "").casefold() for fact in numeric}
    units = {str(fact.get("unit") or "").casefold() for fact in numeric if fact.get("unit")}

    table_id = str(card.get("table_id") or "")
    if table_id and tables and table_id not in tables:
        errors.append(f"table {table_id} was not fetched")
    for period in card.get("periods") or []:
        if str(period).casefold() not in periods:
            errors.append(f"period {period} is not in the fetched rows")
    unit = str(card.get("unit") or "").strip()
    if unit and units and unit.casefold() not in units:
        errors.append(f"unit {unit} does not match the table")
    for value in card.get("values") or []:
        number = _number(value)
        if number is None or not _known(number, fetched):
            errors.append(f"value {value} is not in the fetched data")
    premise = str(card.get("premise_check") or "na")
    if premise == "false":
        figure = _number(card.get("premise_figure"))
        if figure is None or not _known(figure, fetched):
            errors.append("a false premise needs the fetched figure")
    for part in card.get("sub_parts") or []:
        if not isinstance(part, dict):
            errors.append("sub_part is not an object")
            continue
        value = _number(part.get("value"))
        if value is None and not str(part.get("missing") or "").strip():
            errors.append(f"sub_part {part.get('part')} has no value")
        elif value is not None and not _known(value, fetched):
            errors.append(f"sub_part {part.get('part')} value {value} was not fetched")
    share = card.get("share")
    if isinstance(share, dict):
        errors.extend(_share_errors(share, numeric))
    return errors


def render_card(card: dict[str, Any]) -> str:
    """Prose that only uses fields on the card."""
    lines: list[str] = []
    if card.get("premise_check") == "false" and _number(card.get("premise_figure")) is not None:
        lines.append(f"Nej, det hentede tal er {_fmt(card['premise_figure'])}.")
    unit = str(card.get("unit") or "")
    periods = ", ".join(str(period) for period in card.get("periods") or [])
    table = str(card.get("table_id") or "")
    for part in card.get("sub_parts") or []:
        if not isinstance(part, dict):
            continue
        name = str(part.get("part") or "tal")
        if _number(part.get("value")) is None:
            lines.append(f"{name}: {part.get('missing') or 'mangler'}")
            continue
        bits = ", ".join(bit for bit in (unit, periods, table) if bit)
        suffix = f" ({bits})" if bits else ""
        lines.append(f"{name}: {_fmt(part['value'])}{suffix}")
    share = card.get("share")
    if isinstance(share, dict) and _number(share.get("result")) is not None:
        left = share.get("numerator")
        right = share.get("denominator")
        lines.append(f"{left} / {right} = {_fmt(share['result'])} procent.")
    for error in card.get("_errors") or []:
        lines.append(f"Ikke bekræftet: {error}")
    if lines:
        return "\n".join(lines)
    for value in card.get("values") or []:
        if _number(value) is None:
            continue
        shown = _fmt(value)
        suffix = f" ({unit})" if unit else ""
        return f"{shown}{suffix}"
    if table:
        return table
    return "Bekræftet svar."


def _share_errors(share: dict[str, Any], facts: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    numerator = _number(share.get("numerator_value"))
    denominator = _number(share.get("denominator_value"))
    result = _number(share.get("result"))
    if numerator is None or denominator is None or result is None:
        return ["share is missing numerator, denominator, or result"]
    if denominator > 0 and 0 < result <= 1 and _close(result * 100, numerator / denominator * 100):
        result = result * 100
    if result > 100 or numerator > denominator or denominator <= 0:
        errors.append("share exceeds 100 percent")
    period = str(share.get("period") or "")
    table = str(share.get("table_id") or "")
    same = [
        fact
        for fact in facts
        if (not period or str(fact.get("period") or "") == period)
        and (not table or str(fact.get("table") or "") == table)
    ]
    if len({str(fact.get("table") or "") for fact in same}) > 1:
        errors.append("share numerator and denominator are from different tables")
    if not _denominator_is_total(denominator, numerator, same):
        errors.append("share denominator is not the total for that period")
    fetched = [_number(fact["value"]) for fact in same]
    if not _known(numerator, fetched) or (
        not _known(denominator, fetched) and not _close(denominator, sum(fetched))
    ):
        errors.append("share inputs are not in the fetched rows")
    return errors


def _denominator_is_total(
    denominator: float, numerator: float, facts: list[dict[str, Any]]
) -> bool:
    for fact in facts:
        detail = str(fact.get("detail") or "").casefold()
        value = _number(fact.get("value"))
        if value is None or not _close(denominator, value):
            continue
        if any(word in detail for word in ("i alt", "total", "hele landet")):
            return True
    fetched = [_number(fact["value"]) for fact in facts if _number(fact.get("value")) is not None]
    part_sum = sum(fetched)
    numerator_is_part = any(_close(numerator, value) for value in fetched)
    if fetched and _close(denominator, part_sum) and numerator_is_part:
        return True
    return len(facts) == 1 and _close(denominator, _number(facts[0].get("value")) or -1)


def _known(value: Any, fetched: list[float]) -> bool:
    if any(_equivalent(value, item) for item in fetched):
        return True
    number = _number(value)
    if number is None:
        return False
    for left in fetched:
        for right in fetched:
            if _close(number, left - right) or _close(number, abs(left - right)):
                return True
            if _close(number, left + right):
                return True
            if right and left <= right and _close(number, left / right * 100):
                return True
    return False


def _equivalent(stated: Any, fetched: float) -> bool:
    """Thousand separators, decimal commas, a trailing unit, and a percent fraction."""
    parsed = _parsed(stated)
    if parsed is None:
        return False
    number, saw_percent = parsed
    if _close(number, fetched):
        return True
    if saw_percent and _close(number / 100, fetched):
        return True
    return (
        not saw_percent
        and 0 < abs(number) <= 1
        and abs(fetched) <= 100
        and _close(number * 100, fetched)
    )


def _claimed_numbers(question: str) -> list[float]:
    found: list[float] = []
    for token in _NUMBER_RE.findall(question):
        if re.fullmatch(r"(?:19|20)\d{2}", token):
            continue
        number = float(token.replace(".", "")) if "." in token else float(token)
        if number >= 1000:
            found.append(number)
    return found


def _number(value: Any) -> float | None:
    parsed = _parsed(value)
    if parsed is None:
        return None
    return parsed[0]


def _parsed(value: Any) -> tuple[float, bool] | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value), False
    if not isinstance(value, str):
        return None
    raw = value.strip().replace("\u00a0", " ")
    if not raw:
        return None
    saw_percent = _PERCENT_WORD.search(raw) is not None
    text = _VALUE_SUFFIX.sub("", raw).strip().replace(" ", "")
    number = _parse_token(text)
    if number is None:
        return None
    return number, saw_percent


def _parse_token(token: str) -> float | None:
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+,\d+", token):
        whole, frac = token.split(",")
        return float(f"{whole.replace('.', '')}.{frac}")
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+\.\d+", token):
        whole, frac = token.split(".")
        return float(f"{whole.replace(',', '')}.{frac}")
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", token) and not token.startswith("0"):
        return float(token.replace(".", ""))
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", token):
        return float(token.replace(",", ""))
    if re.fullmatch(r"\d+,\d+", token):
        return float(token.replace(",", "."))
    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        return float(token)
    return None


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= max(0.05, abs(right) * 0.001)


def _fmt(value: Any) -> str:
    number = _number(value)
    if number is None:
        return str(value)
    if number.is_integer():
        return f"{int(number):,}".replace(",", ".")
    text = f"{number:.10f}".rstrip("0").rstrip(".")
    whole, frac = text.split(".")
    return f"{int(whole):,}".replace(",", ".") + "," + frac


def _unique(items: Any) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.append(item)
    return seen
