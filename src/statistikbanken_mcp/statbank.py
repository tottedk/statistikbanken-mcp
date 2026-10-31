"""Small client for the Statistikbanken API (https://api.statbank.dk/v1).

The API has no free-text search, so table search filters the published table
list. Age and other numeric codes are single values; a total such as "80 and
older" has to be requested with the API's own ``sum()`` syntax. Passing
``>=80`` returns one row per code and does not add them up.
"""

from __future__ import annotations

import difflib
import json
import re
import threading
from dataclasses import dataclass
from typing import Any

import httpx

BASE_URL = "https://api.statbank.dk/v1"
USER_AGENT = "statistikbanken-mcp/0.1"
VALUE_LIST_CAP = 50

# Everyday words and a few English ones, only where the table title uses another term.
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "bruttoledighed": ("fuldtidsledige",),
    "unemployment": ("fuldtidsledige",),
    "population": ("befolkning",),
    "people": ("befolkning",),
    "inhabitants": ("befolkning",),
    "huspriser": ("ejendomssalg",),
    "husdyr": ("dyr",),
    "livestock": ("dyr",),
    "cattle": ("dyr",),
    "indbrud": ("forbrydelser",),
    "burglary": ("forbrydelser",),
    "studerende": ("uddannelse",),
    "students": ("uddannelse",),
    "indvandrere": ("indvandr",),
    "immigrants": ("indvandr",),
    "income": ("indkomst",),
    "kommune": ("område",),
    "denmark": ("danmark",),
    "danish": ("danmark",),
}
_STOPWORDS = frozenset(
    {
        "og",
        "i",
        "på",
        "af",
        "til",
        "med",
        "for",
        "den",
        "det",
        "der",
        "som",
        "en",
        "et",
        "er",
        "var",
        "har",
        "hvor",
        "mange",
        "hvordan",
        "hvad",
        "the",
        "of",
        "in",
        "on",
        "a",
        "an",
        "to",
        "and",
        "how",
        "many",
        "what",
        "live",
        "lives",
        "there",
        "are",
        "is",
    }
)

_TABLE_MISSING = re.compile(r"Tabel findes ikke:\s*(.*?)\s*$")
_VARIABLE_MISSING = re.compile(r"Kan ikke finde variablen:\s*(.*?)\s*$")
_VALUE_MISSING = re.compile(r"Kan ikke finde værdien:\s*(.*?)\s*\((.*?)\)\s*$")


class StatbankError(Exception):
    """A failure the caller can correct, with a message and no traceback."""


class SumNotAllowed(StatbankError):
    """The API refused sum() for this variable. The caller can add the rows."""


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

        tokens = _query_tokens(query)
        folded = query.casefold().strip()
        places = self._matching_places(tokens)
        ranked: list[tuple[int, int, dict[str, Any]]] = []
        for index, table in enumerate(self._table_list()):
            score = _score_table(table, tokens, folded)
            place_hit = places if str(table.get("id")) == "FOLK1A" else []
            if place_hit:
                score = (score or 0) + 250
            if score is None:
                continue
            hit = _public_table(table)
            if place_hit:
                hit["matched_values"] = [
                    {"variable": "OMRÅDE", "id": item["id"], "text": item["text"]}
                    for item in place_hit[:5]
                ]
            ranked.append((score, -index, hit))
        if not ranked:
            raise StatbankError(
                f"No table matches {query.strip()!r}. Try fewer words, or a subject such as "
                "'befolkning', 'indkomst' or 'priser'. A place or category name is a value "
                "inside a table: search the subject, then pass the name to hent_data."
            )
        ranked.sort(reverse=True)
        return [item for _, _, item in ranked[:limit]]

    def table_info(
        self,
        table: str,
        *,
        variable: str | None = None,
        values_filter: str | None = None,
    ) -> dict[str, Any]:
        return _public_metadata(
            self._metadata_for(table),
            variable=variable,
            values_filter=values_filter,
        )

    def _matching_places(self, tokens: list[str]) -> list[dict[str, str]]:
        try:
            metadata = self._metadata_for("FOLK1A")
        except StatbankError:
            return []
        area = metadata.variable("OMRÅDE")
        if area is None:
            return []
        found: list[dict[str, str]] = []
        seen: set[str] = set()
        for token in tokens:
            folded = _fold(token)
            for item in area.values:
                if item["id"] in seen:
                    continue
                if _fold(item["text"]) == folded or _fold(item["id"]) == folded:
                    found.append(item)
                    seen.add(item["id"])
        return found

    def fetch(
        self,
        table: str,
        values: dict[str, list[str] | str] | None = None,
        sum_from: dict[str, str] | None = None,
        sum_range: dict[str, list[str] | str] | None = None,
    ) -> dict[str, Any]:
        values = _coerce_code_map(values, "values")
        sum_range = _coerce_code_map(sum_range, "sum_range")
        metadata = self._metadata_for(table)
        selections, explicit = _build_selections(metadata, values, sum_from, sum_range)
        assumed = _fill_required(metadata, selections)
        _guard_time_sums(metadata, selections, explicit)
        variables = [{"code": code, "values": selected} for code, selected in selections.items()]
        payload: dict[str, Any] = {
            "table": metadata.id,
            "format": "JSONSTAT",
            "lang": "da",
            "timeOrder": "Ascending",
        }
        if variables:
            payload["variables"] = variables
        source = "Danmarks Statistik"
        try:
            data = self._post("data", payload)
        except SumNotAllowed:
            rows = self._fetch_sum_locally(payload, explicit)
        else:
            rows = _rows_from_jsonstat(data)
            if isinstance(data, dict):
                source = (data.get("dataset") or {}).get("source") or source
        _flag_unpublished(rows)
        result = {
            "table": metadata.id,
            "title": metadata.title,
            "description": metadata.description,
            "unit": metadata.unit,
            "updated": metadata.updated,
            "periods": _periods(metadata, rows),
            "source": source,
            "rows": rows,
        }
        if assumed:
            result["defaults"] = assumed
        return result

    def _fetch_sum_locally(
        self,
        payload: dict[str, Any],
        explicit: dict[str, list[str]],
    ) -> list[dict[str, Any]]:
        if len(explicit) != 1:
            raise StatbankError(
                "Statistikbanken cannot add these codes. Request them in values and add the rows."
            )
        variable_id, codes = next(iter(explicit.items()))
        variables = []
        for item in payload.get("variables", []):
            selected = codes if item["code"] == variable_id else item["values"]
            variables.append({"code": item["code"], "values": selected})
        expanded = {**payload, "variables": variables}
        data = self._post("data", expanded)
        return _sum_rows(_rows_from_jsonstat(data), variable_id, codes)

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
            payload_error = _json_or_none(response)
            raw = ""
            if isinstance(payload_error, dict) and payload_error.get("message"):
                raw = str(payload_error["message"])
            message = _message_from_api_error(payload_error, response.status_code)
            if "ikke tilladt" in raw.casefold() and "sum" in raw.casefold():
                raise SumNotAllowed(message)
            raise StatbankError(message)
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise StatbankError("Statistikbanken returned a response that was not JSON.") from exc


def _query_tokens(query: str) -> list[str]:
    parts = [part for part in query.casefold().split() if part]
    kept = [part for part in parts if part not in _STOPWORDS and len(part) >= 3]
    return kept or parts


def _expansions(token: str) -> tuple[str, ...]:
    return (token, *_SYNONYMS.get(token, ()))


def _contains_token(haystack: str, token: str) -> bool:
    if len(token) >= 4:
        return token in haystack
    return re.search(rf"(?<!\w){re.escape(token)}(?!\w)", haystack) is not None


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
    elif folded and _contains_token(haystack_id, folded):
        score += 200
    matched = 0
    for token in tokens:
        best = 0
        for alt in _expansions(token):
            if _contains_token(haystack_id, alt):
                best = max(best, 80)
            elif _contains_token(haystack_title, alt):
                best = max(best, 40)
            elif _contains_token(haystack_variables, alt):
                best = max(best, 15)
        if best:
            matched += 1
            score += best
    if matched == 0 and score < 1000:
        return None
    score += matched * 25
    if tokens and matched == len(tokens):
        score += 30
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


def _public_metadata(
    table: _Table,
    *,
    variable: str | None = None,
    values_filter: str | None = None,
) -> dict[str, Any]:
    chosen = table.variables
    if variable is not None and str(variable).strip():
        match = table.variable(str(variable))
        if match is None:
            known = ", ".join(item.id for item in table.variables)
            raise StatbankError(
                f"Unknown variable '{variable}' on table {table.id}. Known variables: {known}."
            )
        chosen = (match,)
    return {
        "id": table.id,
        "title": table.title,
        "description": table.description,
        "unit": table.unit,
        "updated": table.updated,
        "active": table.active,
        "variables": [_public_variable(item, values_filter=values_filter) for item in chosen],
    }


def _public_variable(variable: _Variable, *, values_filter: str | None = None) -> dict[str, Any]:
    published: dict[str, Any] = {
        "id": variable.id,
        "text": variable.text,
        "eliminable": variable.eliminable,
        "time": variable.time,
    }
    total = _unique_total(variable)
    if total is not None:
        published["total"] = {"id": total["id"], "text": total["text"]}
    terms = _filter_terms(values_filter)
    if terms:
        shown = [
            {"id": item["id"], "text": item["text"]}
            for item in variable.values
            if _value_matches_terms(item, terms)
        ]
        if shown:
            published["values"] = shown
            published["value_count"] = len(shown)
            return published
        published["filter_note"] = "No match for the filter on this variable."
    if variable.time:
        values = variable.values
        if values:
            published["time_span"] = {
                "first": {"id": values[0]["id"], "text": values[0]["text"]},
                "last": {"id": values[-1]["id"], "text": values[-1]["text"]},
                "count": len(values),
            }
        else:
            published["time_span"] = {"first": None, "last": None, "count": 0}
        return published
    numeric, explicit = _split_numeric_run(variable.values)
    if numeric is not None:
        published["numeric_codes"] = numeric
    published["value_count"] = len(explicit)
    if len(explicit) > VALUE_LIST_CAP:
        published["values"] = explicit[:VALUE_LIST_CAP]
        published["values_truncated"] = True
    else:
        published["values"] = explicit
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


def _filter_terms(values_filter: str | None) -> list[str]:
    if not values_filter or not values_filter.strip():
        return []
    return [part for part in re.split(r"[\s,;]+", values_filter.casefold()) if len(part) >= 2]


def _value_matches_terms(item: dict[str, str], terms: list[str]) -> bool:
    fields = (
        item["id"].casefold(),
        item["text"].casefold(),
        _fold(item["id"]),
        _fold(item["text"]),
    )
    for term in terms:
        folded = _fold(term)
        if any(term in field or folded in field for field in fields):
            return True
    return False


def _coerce_code_map(raw: Any, name: str) -> dict[str, list[str]] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise StatbankError(f"{name} maps a variable id to a list of codes, or to one code.")
    return {str(key): _coerce_code_list(value, f"{name}.{key}") for key, value in raw.items()}


def _coerce_code_list(value: Any, name: str) -> list[str]:
    if isinstance(value, str):
        return _split_codes(value)
    if isinstance(value, list):
        codes: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise StatbankError(f"{name} must be a list of codes, or one code.")
            codes.extend(_split_codes(item))
        return codes
    raise StatbankError(f"{name} must be a list of codes, or one code.")


def _split_codes(text: str) -> list[str]:
    """Turn one code, or an unambiguous comma-separated list, into codes.

    Expressions such as ``>=2015K1`` are one value. A comma splits only when
    every piece is its own code.
    """
    stripped = text.strip()
    if "," not in stripped or _is_expression(stripped):
        return [stripped]
    parts = [part.strip() for part in stripped.split(",") if part.strip()]
    if len(parts) < 2 or any(_is_expression(part) for part in parts):
        return [stripped]
    return parts


def _guard_time_sums(
    table: _Table,
    selections: dict[str, list[str]],
    explicit: dict[str, list[str]],
) -> None:
    """Refuse a time sum that would hide rows or add only the two endpoints.

    The API often rejects ``sum()`` on Tid. Adding the cells afterwards used
    the first row's labels for every other variable, so a second municipality
    and every age except the first disappeared into one plausible number.
    """
    for var_id, codes in explicit.items():
        variable = table.variable(var_id)
        if variable is None or not variable.time or not codes:
            continue
        conflicts = [
            other_id
            for other_id, selected in selections.items()
            if other_id != var_id
            and selected
            and (len(selected) != 1 or _is_expression(selected[0]))
        ]
        if len(codes) != 2 and not conflicts:
            continue
        raise StatbankError(_time_sum_error(var_id, codes, conflicts, selections))


def _time_sum_error(
    var_id: str,
    codes: list[str],
    conflicts: list[str],
    selections: dict[str, list[str]],
) -> str:
    start, end = codes[0], codes[-1]
    parts = []
    if len(codes) == 2:
        parts.append(
            f"sum_range on {var_id} with {start} and {end} adds only those two periods, "
            "not every period between them."
        )
    if conflicts:
        listed = ", ".join(conflicts)
        parts.append(
            f"Refusing to add {var_id} while {listed} has several values or a range. "
            "That would hide rows."
        )
    parts.append(
        f'For the development over time, use values {{"{var_id}": [">={start}"]}} '
        f'or {{"{var_id}": [">={start}<={end}"]}}. '
        "Each period comes back as its own row."
    )
    advice = _older_than_advice(selections)
    if advice:
        parts.append(advice)
    return " ".join(parts)


def _older_than_advice(selections: dict[str, list[str]]) -> str:
    selected = selections.get("ALDER") or []
    if len(selected) != 1:
        return ""
    match = re.fullmatch(r">=(\d+)", selected[0].strip())
    if match is None:
        return ""
    start = match.group(1)
    return (
        f'Use sum_from {{"ALDER": "{start}"}} for ages {start} and older, not values [">={start}"].'
    )


def _build_selections(
    table: _Table,
    values: dict[str, list[str]] | None,
    sum_from: dict[str, str] | None,
    sum_range: dict[str, list[str]] | None = None,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    if values is None:
        values = {}
    if sum_from is None:
        sum_from = {}
    if sum_range is None:
        sum_range = {}
    maps_ok = (
        isinstance(values, dict) and isinstance(sum_from, dict) and isinstance(sum_range, dict)
    )
    if not maps_ok:
        raise StatbankError(
            "values maps a variable id to a list of codes. "
            "sum_from maps a variable id to one numeric code. "
            "sum_range maps a variable id to two numeric codes, low then high, "
            "or to a list of codes to add."
        )

    chosen: dict[str, list[str]] = {}
    explicit: dict[str, list[str]] = {}
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

    for code, bounds in sum_range.items():
        variable = _require_variable(table, str(code))
        if variable.id in chosen:
            raise StatbankError(
                f"Variable {variable.id} is listed in sum_range and also in values or sum_from. "
                "Put it in only one of them."
            )
        expression, codes = _sum_range_or_codes(table, variable, bounds)
        chosen[variable.id] = [expression]
        if codes is not None:
            explicit[variable.id] = codes

    ordered = {
        variable.id: chosen[variable.id] for variable in table.variables if variable.id in chosen
    }
    return ordered, explicit


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
    resolved: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        token = raw.strip()
        if not token:
            raise StatbankError(f"Variable {variable.id} contains an empty value code.")
        if _is_expression(token):
            canonical = token
        else:
            canonical = _canonical_value(table, variable, token)
        if canonical not in seen:
            resolved.append(canonical)
            seen.add(canonical)
    return resolved


def _canonical_value(table: _Table, variable: _Variable, token: str) -> str:
    matches = _value_matches(variable, token)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        listed = ", ".join(matches)
        raise StatbankError(
            f"Value '{token}' matches several codes on {variable.id}: {listed}. Pass one code."
        )
    suggestions = _close_values(variable, token)
    latest = ""
    if variable.time and variable.values:
        latest = f" Latest period is {variable.values[-1]['id']}."
    advice = suggestions or _value_hint(variable)
    raise StatbankError(
        f"Unknown value '{token}' for variable {variable.id} on table {table.id}.{latest} {advice}"
    )


def _value_matches(variable: _Variable, token: str) -> list[str]:
    folded = _fold(token)
    exact_code = [
        item["id"] for item in variable.values if item["id"].casefold() == token.casefold()
    ]
    if exact_code:
        return exact_code
    folded_code = [item["id"] for item in variable.values if _fold(item["id"]) == folded]
    if folded_code:
        return folded_code
    labels: list[str] = []
    for item in variable.values:
        keys = {_fold(item["text"])}
        for prefix in ("til ", "fra "):
            text = _fold(item["text"])
            if text.startswith(prefix):
                keys.add(text[len(prefix) :].strip())
        if folded in keys:
            labels.append(item["id"])
    return labels


def _close_values(variable: _Variable, token: str) -> str:
    folded = _fold(token)
    pool: list[str] = []
    owner: dict[str, dict[str, str]] = {}
    for item in variable.values:
        for key in (_fold(item["text"]), _fold(item["id"])):
            if key and key not in owner:
                pool.append(key)
                owner[key] = item
    found = difflib.get_close_matches(folded, pool, n=3, cutoff=0.72)
    if not found:
        return ""
    shown: list[str] = []
    seen: set[str] = set()
    for key in found:
        item = owner[key]
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        shown.append(f"{item['text']} ({item['id']})")
    return "Did you mean " + ", ".join(shown) + "?"


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


def _sum_range_or_codes(
    table: _Table,
    variable: _Variable,
    bounds: Any,
) -> tuple[str, list[str] | None]:
    if _is_numeric_span(bounds):
        return _sum_range_expression(variable, bounds), None
    codes_ok = (
        isinstance(bounds, list)
        and len(bounds) >= 2
        and all(isinstance(item, str) for item in bounds)
    )
    if not codes_ok:
        raise StatbankError(
            f"sum_range for {variable.id} must be two numeric codes such as ['0', '17'], "
            "or a list of codes to add such as ['25-29', '30-34']."
        )
    codes = [_canonical_value(table, variable, item) for item in bounds]
    return _sum_expression_from_codes(codes), codes


def _is_numeric_span(bounds: Any) -> bool:
    if not isinstance(bounds, list) or len(bounds) != 2:
        return False
    if not all(isinstance(item, str) for item in bounds):
        return False
    low, high = bounds[0].strip(), bounds[1].strip()
    return low.isdigit() and high.isdigit() and str(int(low)) == low and str(int(high)) == high


def _sum_expression_from_codes(codes: list[str]) -> str:
    label = "+".join(codes)
    if len(label) > 80:
        label = f"{codes[0]}+…+{codes[-1]}"
    return f"sum({label}|{label}={';'.join(codes)})"


def _sum_range_expression(variable: _Variable, bounds: Any) -> str:
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or not all(isinstance(item, str) for item in bounds)
    ):
        raise StatbankError(
            f"sum_range for {variable.id} must be two numeric codes, low then high, "
            "for example ['0', '17']."
        )
    low_token, high_token = bounds[0].strip(), bounds[1].strip()
    if not low_token.isdigit() or not high_token.isdigit():
        raise StatbankError(
            f"sum_range for {variable.id} must be two numeric codes "
            f"(got '{low_token}' and '{high_token}'). {_value_hint(variable)}"
        )
    low, high = int(low_token), int(high_token)
    if low > high:
        raise StatbankError(
            f"sum_range for {variable.id} must run from a lower code to a higher one "
            f"(got {low_token} then {high_token})."
        )
    codes = [
        item["id"]
        for item in variable.values
        if item["id"].isdigit() and low <= int(item["id"]) <= high
    ]
    if not codes:
        raise StatbankError(
            f"No numeric codes on {variable.id} fall between {low} and {high}. "
            f"{_value_hint(variable)}"
        )
    if len(codes) == 1:
        return codes[0]
    label = f"{low}-{high}"
    return f"sum({label}|{label}={';'.join(codes)})"


def _value_hint(variable: _Variable) -> str:
    numeric, explicit = _split_numeric_run(variable.values)
    if numeric is not None and len(explicit) <= 20:
        others = ", ".join(f"{item['id']} ({item['text']})" for item in explicit)
        suffix = f" Other codes: {others}." if others else ""
        return (
            f"Numeric codes run from {numeric['from']} to {numeric['to']} "
            f"(the code is the number).{suffix} "
            "To total a closed range, use sum_range, for example ['0', '17']. "
            "To total codes from N upward, use sum_from."
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


def _fold(text: str) -> str:
    return text.casefold().replace("æ", "ae").replace("ø", "o").replace("å", "aa")


def _is_total_value(item: dict[str, str]) -> bool:
    code = item["id"].casefold()
    text = item["text"].casefold().strip()
    return code in {"tot", "ialt"} or text in {"i alt", "alle", "total"}


def _unique_total(variable: _Variable) -> dict[str, str] | None:
    totals = [item for item in variable.values if _is_total_value(item)]
    if len(totals) == 1:
        return totals[0]
    return None


def _fill_required(table: _Table, selections: dict[str, list[str]]) -> list[dict[str, str]]:
    assumed: list[dict[str, str]] = []
    missing: list[_Variable] = []
    for variable in table.variables:
        if variable.time or variable.eliminable or variable.id in selections:
            continue
        total = _unique_total(variable)
        if total is None:
            missing.append(variable)
            continue
        selections[variable.id] = [total["id"]]
        assumed.append({"variable": variable.id, "id": total["id"], "text": total["text"]})
    if missing:
        raise StatbankError(_missing_variables_message(table, missing))
    ordered = {
        variable.id: selections[variable.id]
        for variable in table.variables
        if variable.id in selections
    }
    selections.clear()
    selections.update(ordered)
    return assumed


def _missing_variables_message(table: _Table, missing: list[_Variable]) -> str:
    parts = [
        f"Variable {variable.id} on {table.id} has no total and must be set. "
        f"{_code_sample(variable)}"
        for variable in missing
    ]
    return " ".join(parts)


def _code_sample(variable: _Variable) -> str:
    shown = variable.values[:10]
    listed = ", ".join(f"{item['id']} ({item['text']})" for item in shown)
    extra = ""
    if len(variable.values) > len(shown):
        extra = f" … {len(variable.values) - len(shown)} more."
    return f"Codes: {listed}.{extra} Pass one of them in values."


def _flag_unpublished(rows: list[dict[str, Any]]) -> None:
    if len(rows) < 2:
        return
    keys = [key for key in rows[0] if key != "value"]
    varying = [key for key in keys if len({row.get(key) for row in rows}) > 1]
    if len(varying) != 1:
        return
    trailing: list[dict[str, Any]] = []
    for row in reversed(rows):
        if row.get("value") == 0:
            trailing.append(row)
        else:
            break
    if trailing and len(trailing) < len(rows):
        for row in trailing:
            row["unpublished"] = True


def _sum_rows(
    rows: list[dict[str, Any]],
    variable_id: str,
    codes: list[str],
) -> list[dict[str, Any]]:
    """Add one variable inside each other combination of values.

    One group per series. The first row's labels are not reused for a second
    municipality or a second age.
    """
    if not rows:
        raise StatbankError(f"Statistikbanken returned no rows to add for {variable_id}.")
    grouped: dict[tuple[tuple[str, Any], ...], dict[str, Any]] = {}
    order: list[tuple[tuple[str, Any], ...]] = []
    for row in rows:
        value = row.get("value")
        if isinstance(value, bool) or not isinstance(value, int | float):
            label = row.get(variable_id, "")
            raise StatbankError(f"Could not add {variable_id} because '{label}' has no number.")
        identity = tuple(
            (key, row.get(key)) for key in row if key not in {variable_id, "value", "unpublished"}
        )
        current = grouped.get(identity)
        if current is None:
            current = {key: item for key, item in identity}
            current[variable_id] = "+".join(codes)
            current["value"] = 0
            grouped[identity] = current
            order.append(identity)
        current["value"] += value
    summed_rows: list[dict[str, Any]] = []
    for identity in order:
        summed = grouped[identity]
        total = summed["value"]
        if isinstance(total, float) and total.is_integer():
            total = int(total)
        summed["value"] = total
        summed_rows.append(summed)
    return summed_rows


def _periods(table: _Table, rows: list[dict[str, Any]]) -> list[str]:
    time_ids = [variable.id for variable in table.variables if variable.time]
    periods: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for time_id in time_ids:
            period = row.get(time_id)
            if isinstance(period, str) and period not in seen:
                seen.add(period)
                periods.append(period)
    return periods


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
            "To total a closed numeric range, use sum_range. "
            "To total codes from N upward, use sum_from."
        )
    must_choose = re.search(r"Der skal vælges værdier for variabel:\s*(.*?)\s*$", message)
    if must_choose:
        variable = must_choose.group(1)
        return (
            f"Variable {variable} must be set; it has no single total. "
            "Call tabelinfo and pass one of its codes in values."
        )
    if message:
        return (
            f"Statistikbanken rejected the request: {message} "
            "Check the table id and the codes from tabelinfo."
        )
    return (
        f"Statistikbanken returned HTTP {status}. Check the table id and the codes from tabelinfo."
    )
