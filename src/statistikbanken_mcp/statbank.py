"""Small client for the Statistikbanken API (https://api.statbank.dk/v1).

The API has no free-text search, so table search filters the published table
list. Age and other numeric codes are single values; a total such as "80 and
older" has to be requested with the API's own ``sum()`` syntax. Passing
``>=80`` returns one row per code and does not add them up.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from typing import Any

import httpx

BASE_URL = "https://api.statbank.dk/v1"
USER_AGENT = "statistikbanken-mcp/0.1"

_TABLE_MISSING = re.compile(r"Tabel findes ikke:\s*(.*?)\s*$")
_VARIABLE_MISSING = re.compile(r"Kan ikke finde variablen:\s*(.*?)\s*$")
_VALUE_MISSING = re.compile(r"Kan ikke finde værdien:\s*(.*?)\s*\((.*?)\)\s*$")


class StatbankError(Exception):
    """A failure the caller can correct, with a message and no traceback."""


@dataclass(frozen=True)
class _Variable:
    id: str
    text: str
    eliminable: bool
    time: bool
    values: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class _Table:
    id: str
    title: str
    description: str
    unit: str
    updated: str | None
    active: bool | None
    variables: tuple[_Variable, ...]

    def variable(self, code: str) -> _Variable | None:
        key = code.casefold()
        for variable in self.variables:
            if variable.id.casefold() == key:
                return variable
        return None


class StatbankClient:
    """HTTP client. Pass an ``httpx.Client`` in tests; otherwise one is opened."""

    def __init__(
        self,
        http: httpx.Client | None = None,
        *,
        base_url: str = BASE_URL,
        timeout: float = 60.0,
    ) -> None:
        self._owns_http = http is None
        self._http = http or httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
        self._lock = threading.Lock()
        self._tables: list[dict[str, Any]] | None = None
        self._metadata: dict[str, _Table] = {}

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def find_tables(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise StatbankError("Provide a search query, for example 'befolkning' or 'FOLK1A'.")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 25:
            raise StatbankError("limit must be an integer from 1 to 25.")

        tokens = query.casefold().split()
        folded = query.casefold().strip()
        ranked: list[tuple[int, int, dict[str, Any]]] = []
        for index, table in enumerate(self._table_list()):
            score = _score_table(table, tokens, folded)
            if score is None:
                continue
            ranked.append((score, -index, _public_table(table)))
        ranked.sort(reverse=True)
        return [item for _, _, item in ranked[:limit]]

    def table_info(self, table: str) -> dict[str, Any]:
        return _public_metadata(self._metadata_for(table))

    def fetch(
        self,
        table: str,
        values: dict[str, list[str]] | None = None,
        sum_from: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        metadata = self._metadata_for(table)
        selections = _build_selections(metadata, values, sum_from)
        variables = [{"code": code, "values": selected} for code, selected in selections.items()]
        payload: dict[str, Any] = {
            "table": metadata.id,
            "format": "JSONSTAT",
            "lang": "da",
            "timeOrder": "Ascending",
        }
        if variables:
            payload["variables"] = variables
        data = self._post("data", payload)
        rows = _rows_from_jsonstat(data)
        source = None
        if isinstance(data, dict):
            source = (data.get("dataset") or {}).get("source")
        return {
            "table": metadata.id,
            "title": metadata.title,
            "unit": metadata.unit,
            "updated": metadata.updated,
            "source": source or "Danmarks Statistik",
            "rows": rows,
        }

    def _table_list(self) -> list[dict[str, Any]]:
        with self._lock:
            cached = self._tables
        if cached is not None:
            return cached
        data = self._post("tables", {"lang": "da", "format": "JSON"})
        if not isinstance(data, list):
            raise StatbankError("Statistikbanken returned an unexpected table list.")
        with self._lock:
            self._tables = data
        return data

    def _metadata_for(self, table: str) -> _Table:
        if not isinstance(table, str) or not table.strip():
            raise StatbankError(
                "Provide a table id, for example 'FOLK1A'. Use find_tabel to search."
            )
        key = table.strip().casefold()
        with self._lock:
            cached = self._metadata.get(key)
        if cached is not None:
            return cached
        data = self._post(
            "tableinfo",
            {"table": table.strip(), "lang": "da", "format": "JSON"},
        )
        parsed = _parse_table(data)
        with self._lock:
            self._metadata[key] = parsed
            self._metadata[parsed.id.casefold()] = parsed
        return parsed

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        try:
            response = self._http.post(path, json=payload)
        except httpx.TimeoutException as exc:
            raise StatbankError(
                "Statistikbanken did not respond in time. Try a smaller request."
            ) from exc
        except httpx.HTTPError as exc:
            raise StatbankError(
                "Could not reach Statistikbanken (https://api.statbank.dk). Try again."
            ) from exc
        if response.status_code >= 400:
            message = _message_from_api_error(_json_or_none(response), response.status_code)
            raise StatbankError(message)
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise StatbankError("Statistikbanken returned a response that was not JSON.") from exc


def _score_table(table: dict[str, Any], tokens: list[str], folded: str) -> int | None:
    table_id = str(table.get("id") or "")
    title = str(table.get("text") or "")
    variable_names = " ".join(str(name) for name in table.get("variables") or [])
    haystack_id = table_id.casefold()
    haystack_title = title.casefold()
    haystack_variables = variable_names.casefold()
    score = 0
    if haystack_id == folded:
        score += 1000
    elif folded in haystack_id:
        score += 200
    for token in tokens:
        if token in haystack_id:
            score += 50
        elif token in haystack_title:
            score += 30
        elif token in haystack_variables:
            score += 10
        else:
            return None
    if table.get("active") is True:
        score += 5
    return score


def _public_table(table: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": table.get("id"),
        "title": table.get("text") or "",
        "unit": table.get("unit") or "",
        "latest_period": table.get("latestPeriod"),
        "variables": list(table.get("variables") or []),
    }


def _parse_table(data: Any) -> _Table:
    variables_ok = isinstance(data, dict) and isinstance(data.get("variables"), list)
    if not variables_ok or not data.get("id"):
        raise StatbankError("Statistikbanken returned unexpected table metadata.")
    variables: list[_Variable] = []
    for raw in data["variables"]:
        if not isinstance(raw, dict) or not raw.get("id"):
            raise StatbankError("Statistikbanken returned unexpected table metadata.")
        values = tuple(
            {"id": str(item["id"]), "text": str(item.get("text") or "")}
            for item in raw.get("values") or []
            if isinstance(item, dict) and item.get("id") is not None
        )
        variables.append(
            _Variable(
                id=str(raw["id"]),
                text=str(raw.get("text") or ""),
                eliminable=bool(raw.get("elimination")),
                time=bool(raw.get("time")),
                values=values,
            )
        )
    return _Table(
        id=str(data["id"]),
        title=str(data.get("text") or ""),
        description=str(data.get("description") or ""),
        unit=str(data.get("unit") or ""),
        updated=data.get("updated"),
        active=data.get("active"),
        variables=tuple(variables),
    )


def _public_metadata(table: _Table) -> dict[str, Any]:
    return {
        "id": table.id,
        "title": table.title,
        "description": table.description,
        "unit": table.unit,
        "updated": table.updated,
        "active": table.active,
        "variables": [_public_variable(variable) for variable in table.variables],
    }


def _public_variable(variable: _Variable) -> dict[str, Any]:
    numeric, explicit = _split_numeric_run(variable.values)
    published: dict[str, Any] = {
        "id": variable.id,
        "text": variable.text,
        "eliminable": variable.eliminable,
        "time": variable.time,
        "values": explicit,
    }
    if numeric is not None:
        published["numeric_codes"] = numeric
    return published


def _split_numeric_run(
    values: tuple[dict[str, str], ...],
) -> tuple[dict[str, str] | None, list[dict[str, str]]]:
    """Collapse a long contiguous run such as ages 0–125.

    Codes with leading zeros (area code ``000``) stay listed. They are not
    ages, and ``str(int("000"))`` is not ``"000"``.
    """
    numbers: list[int] = []
    explicit: list[dict[str, str]] = []
    for item in values:
        code = item["id"]
        if code.isdigit() and str(int(code)) == code:
            numbers.append(int(code))
        else:
            explicit.append(item)
    contiguous = len(numbers) >= 12 and numbers == list(range(numbers[0], numbers[-1] + 1))
    if not contiguous:
        return None, [dict(item) for item in values]
    return {"from": str(numbers[0]), "to": str(numbers[-1])}, [dict(item) for item in explicit]


def _build_selections(
    table: _Table,
    values: dict[str, list[str]] | None,
    sum_from: dict[str, str] | None,
) -> dict[str, list[str]]:
    if values is None:
        values = {}
    if sum_from is None:
        sum_from = {}
    if not isinstance(values, dict) or not isinstance(sum_from, dict):
        raise StatbankError(
            "values must be a map of variable id to a list of codes, "
            "and sum_from a map of variable id to one numeric code."
        )

    chosen: dict[str, list[str]] = {}
    for code, raw_values in values.items():
        variable = _require_variable(table, str(code))
        if variable.id in chosen:
            raise StatbankError(f"Variable {variable.id} was given more than once.")
        chosen[variable.id] = _resolve_values(table, variable, raw_values)

    for code, start in sum_from.items():
        variable = _require_variable(table, str(code))
        if variable.id in chosen:
            raise StatbankError(
                f"Variable {variable.id} is listed in both values and sum_from. "
                "Put it in only one of them."
            )
        chosen[variable.id] = [_sum_expression(variable, start)]

    return {
        variable.id: chosen[variable.id] for variable in table.variables if variable.id in chosen
    }


def _require_variable(table: _Table, code: str) -> _Variable:
    variable = table.variable(code)
    if variable is not None:
        return variable
    known = ", ".join(item.id for item in table.variables)
    raise StatbankError(f"Unknown variable '{code}' on table {table.id}. Known variables: {known}.")


def _resolve_values(table: _Table, variable: _Variable, raw_values: Any) -> list[str]:
    if (
        not isinstance(raw_values, list)
        or not raw_values
        or not all(isinstance(item, str) for item in raw_values)
    ):
        example = ""
        if variable.values:
            sample = variable.values[0]["id"]
            example = f" For example ['{sample}']."
        raise StatbankError(f"Values for {variable.id} must be a non-empty list of codes.{example}")
    by_fold = {item["id"].casefold(): item["id"] for item in variable.values}
    resolved: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        token = raw.strip()
        if not token:
            raise StatbankError(f"Variable {variable.id} contains an empty value code.")
        if _is_expression(token):
            canonical = token
        else:
            canonical = by_fold.get(token.casefold())
            if canonical is None:
                raise StatbankError(
                    f"Unknown value '{token}' for variable {variable.id} on table {table.id}. "
                    f"{_value_hint(variable)}"
                )
        if canonical not in seen:
            resolved.append(canonical)
            seen.add(canonical)
    return resolved


def _is_expression(token: str) -> bool:
    return token == "*" or token.startswith((">", "<", "(", "sum(", "SUM(")) or "*" in token


def _sum_expression(variable: _Variable, start: Any) -> str:
    if not isinstance(start, str) or not start.strip():
        raise StatbankError(f"sum_from for {variable.id} must be a numeric code such as '80'.")
    token = start.strip()
    if not token.isdigit():
        raise StatbankError(
            f"sum_from for {variable.id} must be a numeric code such as '80' (got '{token}'). "
            f"{_value_hint(variable)}"
        )
    threshold = int(token)
    codes = [
        item["id"]
        for item in variable.values
        if item["id"].isdigit() and int(item["id"]) >= threshold
    ]
    if not codes:
        raise StatbankError(
            f"No codes on {variable.id} are numeric and >= {threshold}. {_value_hint(variable)}"
        )
    label = f"{threshold}+"
    return f"sum({label}|{label}={';'.join(codes)})"


def _value_hint(variable: _Variable) -> str:
    numeric, explicit = _split_numeric_run(variable.values)
    if numeric is not None and len(explicit) <= 20:
        others = ", ".join(f"{item['id']} ({item['text']})" for item in explicit)
        suffix = f" Other codes: {others}." if others else ""
        return (
            f"Numeric codes run from {numeric['from']} to {numeric['to']} "
            f"(the code is the number).{suffix} "
            "To total codes from N upward, use sum_from instead of listing them."
        )
    if len(variable.values) <= 20:
        listed = ", ".join(f"{item['id']} ({item['text']})" for item in variable.values)
        return f"Valid codes: {listed}."
    return "Call tabelinfo to list valid codes."


def _rows_from_jsonstat(payload: Any) -> list[dict[str, Any]]:
    # JSON-stat lays cells out row-major: the last dimension in `size` varies fastest.
    try:
        dataset = payload["dataset"]
        dimension = dataset["dimension"]
        var_ids = list(dimension["id"])
        sizes = [int(size) for size in dimension["size"]]
        values = dataset["value"]
    except (KeyError, TypeError, ValueError) as exc:
        raise StatbankError("Statistikbanken returned data in an unexpected shape.") from exc
    if not isinstance(values, list):
        raise StatbankError("Statistikbanken returned data in an unexpected shape.")

    axes: list[list[str]] = []
    for var_id, size in zip(var_ids, sizes, strict=True):
        try:
            category = dimension[var_id]["category"]
            index = category["index"]
            labels = category.get("label") or {}
        except (KeyError, TypeError) as exc:
            raise StatbankError("Statistikbanken returned data in an unexpected shape.") from exc
        if not isinstance(index, dict):
            raise StatbankError("Statistikbanken returned data in an unexpected shape.")
        ordered = [""] * size
        for code, position in index.items():
            pos = int(position)
            if pos < 0 or pos >= size:
                raise StatbankError("Statistikbanken returned data in an unexpected shape.")
            ordered[pos] = str(labels.get(code, code))
        if any(label == "" for label in ordered):
            raise StatbankError("Statistikbanken returned data in an unexpected shape.")
        axes.append(ordered)

    expected = 1
    for size in sizes:
        expected *= size
    if len(values) != expected:
        raise StatbankError(
            f"Statistikbanken returned {len(values)} cells but the dimensions describe {expected}."
        )

    rows: list[dict[str, Any]] = []
    for flat in range(expected):
        remaining = flat
        coords = [0] * len(sizes)
        for dim in range(len(sizes) - 1, -1, -1):
            coords[dim] = remaining % sizes[dim]
            remaining //= sizes[dim]
        row: dict[str, Any] = {}
        for dim, var_id in enumerate(var_ids):
            if var_id == "ContentsCode":
                continue
            row[var_id] = axes[dim][coords[dim]]
        row["value"] = _number(values[flat])
        rows.append(row)
    return rows


def _number(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _json_or_none(response: httpx.Response) -> Any:
    try:
        return response.json()
    except json.JSONDecodeError:
        return None


def _message_from_api_error(payload: Any, status: int) -> str:
    message = ""
    if isinstance(payload, dict) and payload.get("message"):
        message = str(payload["message"]).strip()

    table_missing = _TABLE_MISSING.search(message)
    if table_missing:
        table_id = table_missing.group(1) or "(empty)"
        return (
            f"Unknown table '{table_id}'. "
            "Use find_tabel to search for a table id, then tabelinfo to read its variables."
        )
    variable_missing = _VARIABLE_MISSING.search(message)
    if variable_missing:
        return (
            f"Unknown variable '{variable_missing.group(1)}'. "
            "Call tabelinfo and use a variable id such as OMRÅDE, KØN, ALDER or Tid."
        )
    value_missing = _VALUE_MISSING.search(message)
    if value_missing:
        value, variable = value_missing.group(1), value_missing.group(2)
        return (
            f"Unknown value '{value}' for variable {variable}. "
            "Call tabelinfo to list valid codes. "
            "To total numeric codes from N upward, use sum_from instead of listing them."
        )
    if message:
        return (
            f"Statistikbanken rejected the request: {message} "
            "Check the table id and the codes from tabelinfo."
        )
    return (
        f"Statistikbanken returned HTTP {status}. Check the table id and the codes from tabelinfo."
    )
