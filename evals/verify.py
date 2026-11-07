"""Fetch the expected figure from the raw Statistikbanken API.

This does not use the MCP client. A wrong default inside hent_data cannot
confirm itself.
"""

from __future__ import annotations

import csv
import io
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from evals.score import Truth
from statistikbanken_mcp.web import rank_urban_areas

API = "https://api.statbank.dk/v1"
USER_AGENT = "statistikbanken-mcp-eval/0.1"

FetchRows = Callable[[str, dict[str, list[str]]], list[dict[str, Any]]]
TimeCodes = Callable[[str], tuple[str, list[str]]]


class VerifierError(Exception):
    """The expected selection could not be read from the API."""


def newest_table_id(tables: list[dict[str, Any]], spec: dict[str, Any]) -> str:
    """Highest numeric id with this prefix whose title matches every required part."""
    prefix = str(spec.get("prefix") or "")
    required = [str(part).casefold() for part in spec.get("title") or []]
    blocked = [str(part).casefold() for part in spec.get("not_title") or []]
    best_id = ""
    best_number = -1
    for table in tables:
        table_id = str(table.get("id") or "")
        if not table_id.startswith(prefix):
            continue
        suffix = table_id[len(prefix) :]
        if not suffix.isdigit():
            continue
        title = str(table.get("text") or table.get("title") or "").casefold()
        if any(part not in title for part in required):
            continue
        if any(part in title for part in blocked):
            continue
        number = int(suffix)
        if number > best_number:
            best_number = number
            best_id = table_id
    if not best_id:
        raise VerifierError(f"no table matches {prefix}")
    return best_id


def truths_for(question: dict[str, Any], *, client: ApiClient | None = None) -> list[Truth]:
    """One truth per acceptable selection. A failed alternative is skipped."""
    expect = question.get("expect") or {}
    checks = list(expect.get("checks") or [])
    if not checks:
        return []
    api = client or ApiClient()
    found: list[Truth] = []
    errors: list[str] = []
    for check in checks:
        try:
            resolved = _resolve_table(check, api)
            if str(check.get("aggregate") or "") == "urban_rank":
                found.append(_urban_truth(resolved, api))
            else:
                found.append(
                    evaluate_check(resolved, fetch_rows=api.rows, time_codes=api.time_codes)
                )
        except VerifierError as exc:
            errors.append(str(exc))
    if checks and not found:
        raise VerifierError("; ".join(errors) or "no selection could be read")
    return found


def _resolve_table(check: dict[str, Any], client: ApiClient) -> dict[str, Any]:
    table = check.get("table")
    if not isinstance(table, dict):
        return check
    resolved = dict(check)
    resolved["table"] = client.newest_id(table)
    return resolved


def evaluate_check(check: dict[str, Any], *, fetch_rows: FetchRows, time_codes: TimeCodes) -> Truth:
    """Turn one selection into the figures a correct answer has to contain."""
    table = str(check["table"])
    variables = _expand(check.get("variables") or {})
    for pair in check.get("sum_where") or []:
        for key, code in pair.items():
            bucket = variables.setdefault(str(key), [])
            if str(code) not in bucket:
                bucket.append(str(code))
    time_id, catalogue = time_codes(table)
    requested = _resolve_time(check.get("time", "latest"), catalogue)
    variables[time_id] = requested
    rows = fetch_rows(table, variables)
    if check.get("sum_where"):
        rows = [row for row in rows if _matches_any(row, check["sum_where"])]
    aggregate = str(check.get("aggregate") or "values")
    numbers = _numbers(rows)
    if aggregate == "sum_by_period":
        values = _sums_by_period(rows, time_id, requested)
        periods = list(requested)
    elif aggregate == "sum":
        if not numbers:
            raise VerifierError(f"{table} returned no numbers")
        values = [sum(numbers)]
        periods = _answer_periods(check, requested)
    elif aggregate == "mean":
        if not numbers:
            raise VerifierError(f"{table} returned no numbers")
        values = [sum(numbers) / len(numbers)]
        periods = _answer_periods(check, requested)
    elif aggregate == "urban_rank":
        return _rank_truth(table, rows, requested, check)
    else:
        values = numbers
        periods = list(requested)
    if not values:
        raise VerifierError(f"{table} returned no numbers for the expected selection")
    values = _with_change(values, rows, time_id, requested, check.get("also") or [])
    return Truth(table=table, values=[float(value) for value in values], periods=periods)


def _with_change(
    values: list[float],
    rows: list[dict[str, Any]],
    time_id: str,
    requested: list[str],
    also: Any,
) -> list[float]:
    """Append the difference and percent when the question asks for the change."""
    wanted = {str(item) for item in also} if isinstance(also, list) else set()
    if not wanted or len(requested) != 2:
        return values
    found: dict[str, float] = {}
    for row in rows:
        period = ""
        for key, value in row.items():
            if str(key).casefold() in {time_id.casefold(), "tid"}:
                period = str(value)
                break
        number = row.get("value")
        if period and isinstance(number, int | float) and not isinstance(number, bool):
            found[period.casefold()] = float(number)
    first = found.get(requested[0].casefold())
    second = found.get(requested[1].casefold())
    if first is None or second is None or first == 0:
        return values
    difference = abs(second - first)
    extra: list[float] = []
    if "difference" in wanted:
        extra.append(difference)
    if "percent" in wanted:
        extra.append(difference / abs(first) * 100)
    return [*values, *extra]


def _rank_truth(
    table: str,
    rows: list[dict[str, Any]],
    requested: list[str],
    check: dict[str, Any],
) -> Truth:
    pairs: list[tuple[str, float]] = []
    for row in rows:
        label = ""
        for key, value in row.items():
            if str(key).casefold() == "byer":
                label = str(value)
                break
        number = row.get("value")
        if label and isinstance(number, int | float) and not isinstance(number, bool):
            pairs.append((label, float(number)))
    ranked = rank_urban_areas(pairs)
    spec = check.get("rank", 1)
    if isinstance(spec, list) and len(spec) == 2:
        start, end = int(spec[0]), int(spec[1])
        chosen = ranked[start - 1 : end]
        if len(chosen) != end - start + 1:
            raise VerifierError(f"{table} does not have urban ranks {start} to {end}")
    else:
        index = int(spec) - 1
        if index < 0 or index >= len(ranked):
            raise VerifierError(f"{table} does not have urban rank {spec}")
        chosen = [ranked[index]]
    return Truth(
        table=table,
        values=[value for _name, value in chosen],
        periods=list(requested),
        names=tuple(name for name, _value in chosen),
    )


def _urban_truth(check: dict[str, Any], client: ApiClient) -> Truth:
    """Rank byområder by their names. The data call itself returns codes."""
    table = str(check["table"])
    names: dict[str, str] = {}
    for variable in client._tableinfo(table).get("variables") or []:
        if str(variable.get("id") or "").casefold() != "byer":
            continue
        for item in variable.get("values") or []:
            if item.get("id") is not None:
                names[str(item["id"])] = str(item.get("text") or item["id"])

    def fetch_rows(fetched: str, variables: dict[str, list[str]]) -> list[dict[str, Any]]:
        rows = client.rows(fetched, variables)
        for row in rows:
            for key in list(row):
                if str(key).casefold() == "byer":
                    row[key] = names.get(str(row[key]), str(row[key]))
        return rows

    return evaluate_check(check, fetch_rows=fetch_rows, time_codes=client.time_codes)


def _answer_periods(check: dict[str, Any], requested: list[str]) -> list[str]:
    """A year total is the year, not the month the guard might show instead."""
    spec = check.get("time")
    if isinstance(spec, dict) and spec.get("year"):
        return [str(spec["year"])]
    return list(requested)


def _expand(variables: dict[str, Any]) -> dict[str, list[str]]:
    expanded: dict[str, list[str]] = {}
    for code, spec in variables.items():
        if isinstance(spec, dict) and "from" in spec and "to" in spec:
            expanded[str(code)] = [str(i) for i in range(int(spec["from"]), int(spec["to"]) + 1)]
        elif isinstance(spec, list):
            expanded[str(code)] = [str(item) for item in spec]
        else:
            expanded[str(code)] = [str(spec)]
    return expanded


def _resolve_time(spec: Any, catalogue: list[str]) -> list[str]:
    if not catalogue:
        raise VerifierError("table has no time variable")
    if spec == "latest" or spec is None:
        return [catalogue[-1]]
    if isinstance(spec, list):
        resolved: list[str] = []
        for item in spec:
            code = catalogue[-1] if str(item) == "latest" else str(item)
            if code not in resolved:
                resolved.append(code)
        missing = [
            code for code in resolved if code.casefold() not in {c.casefold() for c in catalogue}
        ]
        if missing:
            raise VerifierError(f"period not in table: {', '.join(missing)}")
        return resolved
    if isinstance(spec, dict) and spec.get("year"):
        year = str(spec["year"])
        grain = str(spec.get("grain") or "month")
        if grain == "quarter":
            codes = [f"{year}K{quarter}" for quarter in range(1, 5)]
        elif grain == "month":
            codes = [f"{year}M{month:02d}" for month in range(1, 13)]
        else:
            raise VerifierError(f"unknown time grain {grain}")
        known = {code.casefold(): code for code in catalogue}
        if any(code.casefold() not in known for code in codes):
            raise VerifierError(f"{year} is not a complete {grain} in this table")
        return [known[code.casefold()] for code in codes]
    raise VerifierError(f"unknown time spec {spec!r}")


def _numbers(rows: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get("value")
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        values.append(float(value))
    return values


def _sums_by_period(rows: list[dict[str, Any]], time_id: str, periods: list[str]) -> list[float]:
    buckets = {period.casefold(): 0.0 for period in periods}
    seen = {period.casefold(): False for period in periods}
    for row in rows:
        period = ""
        for key, value in row.items():
            if str(key).casefold() == time_id.casefold() or str(key).casefold() == "tid":
                period = str(value)
                break
        number = row.get("value")
        if period.casefold() not in buckets:
            continue
        if isinstance(number, bool) or not isinstance(number, int | float):
            continue
        buckets[period.casefold()] += float(number)
        seen[period.casefold()] = True
    if not all(seen.values()):
        raise VerifierError("a requested period returned no number")
    return [buckets[period.casefold()] for period in periods]


def _matches_any(row: dict[str, Any], pairs: list[dict[str, str]]) -> bool:
    folded = {str(key).casefold(): str(value).casefold() for key, value in row.items()}
    for pair in pairs:
        if all(
            folded.get(str(key).casefold()) == str(code).casefold() for key, code in pair.items()
        ):
            return True
    return False


class ApiClient:
    """Small POST client for /v1/tableinfo and /v1/data. No MCP code."""

    def __init__(self) -> None:
        self._info: dict[str, dict[str, Any]] = {}
        self._tables: list[dict[str, Any]] | None = None

    def newest_id(self, spec: dict[str, Any]) -> str:
        if self._tables is None:
            payload = json.loads(_post("tables", {"lang": "da", "format": "JSON"}).decode())
            if not isinstance(payload, list):
                raise VerifierError("table catalogue was not a list")
            self._tables = [item for item in payload if isinstance(item, dict)]
        return newest_table_id(self._tables, spec)

    def time_codes(self, table: str) -> tuple[str, list[str]]:
        info = self._tableinfo(table)
        for variable in info.get("variables") or []:
            if variable.get("time"):
                codes = [str(item["id"]) for item in variable.get("values") or [] if item.get("id")]
                return str(variable.get("id") or "Tid"), codes
        raise VerifierError(f"{table} has no time variable")

    def rows(self, table: str, variables: dict[str, list[str]]) -> list[dict[str, Any]]:
        body = {
            "table": table,
            "format": "CSV",
            "valuePresentation": "Code",
            "delimiter": "Semicolon",
            "variables": [{"code": code, "values": values} for code, values in variables.items()],
        }
        text = _post("data", body).decode("utf-8-sig")
        parsed = csv.DictReader(io.StringIO(text), delimiter=";")
        rows: list[dict[str, Any]] = []
        for raw in parsed:
            row: dict[str, Any] = {str(key): value for key, value in raw.items() if key}
            cell = str(row.get("INDHOLD") or "")
            number = _cell(cell)
            if number is None:
                continue
            row["value"] = number
            rows.append(row)
        return rows

    def _tableinfo(self, table: str) -> dict[str, Any]:
        key = table.casefold()
        if key not in self._info:
            payload = json.loads(_post("tableinfo", {"table": table, "format": "JSON"}).decode())
            if not isinstance(payload, dict):
                raise VerifierError(f"{table} metadata was not an object")
            self._info[key] = payload
        return self._info[key]


def _cell(cell: str) -> float | None:
    text = cell.strip()
    if text in {"", ".", "..", "-"}:
        return None
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _post(path: str, body: dict[str, Any]) -> bytes:
    data = json.dumps(body).encode()
    request = urllib.request.Request(
        f"{API}/{path}",
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
    )
    last: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise VerifierError(f"{path} {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            time.sleep(1 + attempt)
    raise VerifierError(f"{path} failed: {last}") from last
