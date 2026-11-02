"""Small client for the Statistikbanken API (https://api.statbank.dk/v1).

The API has no free-text search, so table search filters the published table
list. Age and other numeric codes are single values; a total such as "80 and
older" has to be requested with the API's own ``sum()`` syntax. Passing
``>=80`` returns one row per code and does not add them up.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import threading
from dataclasses import dataclass
from datetime import date, timedelta
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
    # Kept so a query that is only "Denmark" still resolves. Mixed queries drop
    # the word in _STOPWORDS before synonyms are applied.
    "denmark": ("danmark",),
    "danish": ("danmark",),
    "tog": ("jernbane",),
    "togpassagerer": ("jernbanetransport af passagerer",),
    "vareeksport": ("eksport",),
    "dødsårsager": ("dødsårsag",),
    "kræft": ("dødsårsag",),
    "gini": ("indkomstfordeling",),
    "iværksættere": ("erhvervsdemografi",),
    "iværksætter": ("erhvervsdemografi",),
    "nyetablerede": ("erhvervsdemografi",),
    "nyetableret": ("erhvervsdemografi",),
    "etablerede": ("erhvervsdemografi",),
    "etableret": ("erhvervsdemografi",),
    "månedsfortjeneste": ("løn",),
    "aku": ("arbejdsmarkedstilknytning",),
    "påbegyndte": ("boligbyggeri",),
    "påbegyndt": ("boligbyggeri",),
    "unemployed": ("ledige",),
    "jobless": ("ledige",),
}
# Replaced before tokenising. The separate words hit the wrong titles
# ("nye" + "virksomheder" is inside "højvækstvirksomheder").
_PHRASES: tuple[tuple[str, str], ...] = (
    ("nyetablerede virksomheder", "erhvervsdemografi"),
    ("etablerede virksomheder", "erhvervsdemografi"),
    ("nye virksomheder", "erhvervsdemografi"),
)
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
        "danmark",
        "danmarks",
        "denmark",
        "danish",
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

# Whole words only. "bruttoledighed" keeps its own synonym; replacing the
# inner "ledighed" would point that query at the wrong title.
_UNEMPLOYMENT_RE = re.compile(
    r"(?<![0-9a-zæøå])(?:arbejdsløshed|arbejdsloshed|ledighed)(?![0-9a-zæøå])"
)
# Stems are matched after _fold, which maps ø to o and æ to ae.
_FLOW_STEMS = (
    "anmeld",
    "overnatning",
    "fodt",
    "fodsel",
    "dodsfald",
    "indvandr",
    "udvandr",
    "flytt",
    "eksport",
    "import",
    "omsaet",
    "transaktion",
    "skilsmis",
    "viels",
    "forbryd",
)
_STOCK_RE = re.compile(
    r"den 1\.|pr\. 1\.|\bultimo\b|\bprimo\b|\bbestand\b|\bbeholdning\b|"
    r"\bpris(?:er|indeks)?\b|\bkurs\b|\bindeks\b|\bprocent\b|\bpct\b",
    re.IGNORECASE,
)
_LEVEL_UNITS = frozenset({"pct", "procent", "indeks", "index"})
_MONTH_NUMBERS = {
    "januar": 1,
    "februar": 2,
    "marts": 3,
    "april": 4,
    "maj": 5,
    "juni": 6,
    "juli": 7,
    "august": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "december": 12,
}
_QUARTER_WORDS = {"forste": 1, "anden": 2, "andet": 2, "tredje": 3, "fjerde": 4}

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
class _YearPlan:
    year: str
    codes: tuple[str, ...]
    complete: bool


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
        timeout: float = 30.0,
        today: date | None = None,
    ) -> None:
        # None uses the real clock. Tests pass a date so fixtures do not depend on it.
        self.today = today
        self._owns_http = http is None
        if http is None and os.environ.get("STATBANK_HTTP_TIMEOUT"):
            timeout = float(os.environ["STATBANK_HTTP_TIMEOUT"])
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
            title = table.get("text") or ""
            hit = {
                "id": table.get("id"),
                "title": title,
                "unit": table.get("unit") or "",
                "latest_period": table.get("latestPeriod"),
                "variables": list(table.get("variables") or []),
            }
            if _title_is_projection(str(title)):
                hit["projection"] = True
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
        values: dict[str, Any] | None = None,
        sum_from: dict[str, Any] | None = None,
        sum_range: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        values, sum_from, sum_range, notes = _hoist_nested_sums(values, sum_from, sum_range)
        values = _coerce_code_map(values, "values")
        sum_range = _coerce_code_map(sum_range, "sum_range")
        sum_from = _coerce_sum_from(sum_from)
        metadata = self._metadata_for(table)
        values, sum_from, sum_range, aliased = _alias_arguments(
            metadata, values, sum_from, sum_range
        )
        notes.extend(aliased)
        values, sum_from, sum_range, promoted = _promote_open_ended(
            metadata, values, sum_from, sum_range
        )
        notes.extend(promoted)
        values, year_plans, period_notes = _prepare_periods(
            metadata, values, self.today or date.today()
        )
        notes.extend(period_notes)
        values, dropped = _drop_equivalent_value_sums(values, sum_from)
        notes.extend(dropped)
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
        if year_plans:
            rows, year_notes = _collapse_years(metadata, rows, year_plans)
            notes.extend(year_notes)
        notes.extend(_mark_projections(metadata, rows))
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
        if any(row.get("projection") for row in rows):
            result["projection"] = True
        if notes:
            result["notes"] = notes
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

    def place_names(self, table: str) -> list[str]:
        """Every area label on the table, including those past the tabelinfo cap."""
        metadata = self._metadata_for(table)
        for variable in metadata.variables:
            if _fold(variable.id) == "omraade":
                return [item["text"] for item in variable.values]
        return []

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
            try:
                payload_error = response.json()
            except json.JSONDecodeError:
                payload_error = None
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
    folded = " ".join(query.casefold().split())
    # "ledighed" also sits in AKU titles. The unemployment tables are titled
    # "Fuldtidsledige", so the word is replaced rather than added as a synonym.
    folded = _UNEMPLOYMENT_RE.sub("fuldtidsledige", folded)
    for phrase, replacement in _PHRASES:
        folded = folded.replace(phrase, replacement)
    parts: list[str] = []
    for part in folded.split():
        cleaned = part.strip(".,;:!?\"'()[]")
        if cleaned:
            parts.extend(_hyphen_forms(cleaned))
    deduped = list(dict.fromkeys(parts))
    kept = [part for part in deduped if part not in _STOPWORDS and len(part) >= 3]
    if kept:
        return kept
    # A query that is only "Danmark" or "i Danmark" still searches Danmark.
    # Short function words are not put back, or "i" would match half the catalogue.
    content = [part for part in deduped if len(part) >= 3]
    return content or deduped


def _hyphen_forms(token: str) -> list[str]:
    """'fuldtids-ledige' is the joined word and each part.

    A title may write the same word joined, hyphenated, or with a space.
    The original token is kept so a hyphenated place name still matches.
    """
    if "-" not in token and "–" not in token and "—" not in token:
        return [token]
    parts = [part for part in re.split(r"[-–—]+", token) if part]
    forms = [token]
    joined = "".join(parts)
    if joined and joined not in forms:
        forms.append(joined)
    for part in parts:
        if part not in forms:
            forms.append(part)
    return forms


def _without_hyphen(text: str) -> str:
    return text.replace("-", "").replace("–", "").replace("—", "")


def _with_stem(token: str, found: list[str]) -> None:
    """Trailing e: 'gennemsnitlige' also matches 'gennemsnitlig'."""
    if len(token) >= 10 and token.endswith("e"):
        found.append(token[:-1])


# Person-dimension words. "før skat" must not outrank a table that actually
# breaks the figure down by one of these.
_BREAKDOWNS = frozenset({"kon", "alder", "uddannelse", "civilstand", "herkomst"})


def _compound_parts(token: str) -> tuple[str, ...]:
    """Split gennemsnitsindkomst on the linking s into gennemsnit + indkomst."""
    if len(token) < 14:
        return ()
    parts: list[str] = []
    for index in range(7, len(token) - 7):
        if token[index] != "s":
            continue
        left, right = token[:index], token[index + 1 :]
        if len(left) < 7 or len(right) < 7:
            continue
        for part in (left, right):
            parts.append(part)
            _with_stem(part, parts)
    return tuple(dict.fromkeys(parts))


def _primary_forms(token: str) -> tuple[str, ...]:
    """The word itself, a trailing e, and synonyms. Not the compound pieces."""
    found = [token]
    _with_stem(token, found)
    for item in tuple(found):
        found.extend(_SYNONYMS.get(item, ()))
    return tuple(dict.fromkeys(found))


def _contains_token(haystack: str, token: str) -> bool:
    hay = _without_hyphen(haystack)
    tok = _without_hyphen(token)
    if len(tok) >= 4:
        return tok in hay
    if not tok:
        return False
    return re.search(rf"(?<!\w){re.escape(tok)}(?!\w)", hay) is not None


def _field_weight(table_id: str, title: str, variables: str, term: str) -> int:
    if _contains_token(table_id, term):
        return 80
    if _contains_token(title, term):
        return 40
    if _contains_token(variables, term):
        return 15
    return 0


def _form_weight(table_id: str, title: str, variables: str, form: str) -> int:
    """One word uses the usual field weight.

    A synonym of several words has to hit every word, and those hits add up,
    so "jernbanetransport" plus "passagerer" outranks a table that only says
    jernbane. A single word is unchanged.
    """
    parts = [part for part in form.split() if part not in _STOPWORDS]
    if len(parts) <= 1:
        return _field_weight(table_id, title, variables, parts[0] if parts else form)
    total = 0
    for part in parts:
        weight = _field_weight(table_id, title, variables, part)
        if not weight:
            return 0
        total += weight
    return total


def _token_score(table_id: str, title: str, variables: str, token: str) -> int:
    """Full word first. Compound pieces count only when the word itself misses.

    A piece is worth less than a variable hit (15), so a real word always
    outranks a table that only matched forening inside foreningsmedlemmer.
    """
    primary = max(
        (_form_weight(table_id, title, variables, term) for term in _primary_forms(token)),
        default=0,
    )
    if primary:
        return primary
    # Title 8, id 6, variables 3. Several pieces add up, still under 15.
    fallback = 0
    for part in _compound_parts(token):
        weight = _field_weight(table_id, title, variables, part)
        if weight == 80:
            fallback += 6
        elif weight == 40:
            fallback += 8
        elif weight:
            fallback += 3
    return min(fallback, 14)


@dataclass(frozen=True)
class _TablePreference:
    """Rank one table first when a common question would otherwise tie.

    Each group in `all_of` is a set of alternatives, and every group has to
    match the query. `unless` turns the preference off, for a forecast, a
    breakdown, or a different subject that shares a word. The boost only
    reorders a table that already matched. An exact table id still wins.
    """

    table_id: str
    all_of: tuple[tuple[str, ...], ...]
    unless: tuple[str, ...] = ()


# One list. Search uses it and nothing else hard-codes these ids.
# FOLK1A is the quarterly population by municipality or age (BEFOLK3 is 1 January).
# AUS08 is seasonally adjusted gross unemployment, monthly (over AKU520K and AUF01).
# INDKP105 is personal income, the amount (INDKP222 counts people in brackets).
# STRAF10 is crimes reported (STRAF20 adds charges).
_TABLE_PREFERENCES: tuple[_TablePreference, ...] = (
    _TablePreference(
        "FOLK1A",
        (
            ("befolkning", "population", "people", "indbyggere"),
            ("kommune", "kommuner", "municipality", "alder", "age"),
        ),
        ("fremskriv",),
    ),
    _TablePreference(
        "AUS08",
        (
            (
                "ledighed",
                "ledige",
                "arbejdsløshed",
                "unemployment",
                "unemployed",
                "bruttoledighed",
                "fuldtidsledige",
                "jobless",
            ),
        ),
        ("stilling",),
    ),
    _TablePreference(
        "INDKP105",
        (("indkomst", "income"), ("personlig", "personal")),
        ("køn", "alder", "uddannelse", "sex"),
    ),
    _TablePreference(
        "STRAF10",
        (("forbrydelse", "forbrydelser", "indbrud", "burglary"),),
        ("ofre", "offer"),
    ),
)
# Enough to pass a title that matches every word of a short query (about 200).
_PREFERENCE_BOOST = 400


def _term_in_query(folded: str, term: str) -> bool:
    """A short word has to stand alone, so 'age' does not match 'average'."""
    needle = _fold(term)
    if not needle:
        return False
    if len(needle) <= 4:
        return re.search(rf"(?<![0-9a-z]){re.escape(needle)}(?![0-9a-z])", folded) is not None
    return needle in folded


def _preference_boost(table_id: str, query: str) -> int:
    folded = _fold(query)
    for preference in _TABLE_PREFERENCES:
        if preference.table_id.casefold() != table_id.casefold():
            continue
        if any(_term_in_query(folded, term) for term in preference.unless):
            return 0
        groups = preference.all_of
        if groups and all(any(_term_in_query(folded, term) for term in group) for group in groups):
            return _PREFERENCE_BOOST
    return 0


def _pre_tax_adjustment(query: str, title: str, tokens: list[str]) -> int:
    """Prefer "Indkomst i alt" for income before tax, unless a breakdown is named.

    The boost has to clear three ordinary title hits (about 200) or INDKP105
    never reaches the top three. A query that also says køn, alder, or
    uddannelse only gets a tie-breaker, so the table that carries that
    dimension stays first.
    """
    if "for skat" not in " ".join(_fold(query).split()):
        return 0
    folded = _fold(title)
    breakdown = any(_fold(token) in _BREAKDOWNS for token in tokens)
    score = 0
    if "indkomst i alt" in folded:
        score += 1 if breakdown else 170
    if "disponibel" in folded:
        score -= 1 if breakdown else 40
    return score


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
        best = _token_score(haystack_id, haystack_title, haystack_variables, token)
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
    return (
        score
        + _pre_tax_adjustment(folded, haystack_title, tokens)
        + _preference_boost(table_id, folded)
    )


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
    published: dict[str, Any] = {
        "id": table.id,
        "title": table.title,
        "description": table.description,
        "unit": table.unit,
        "updated": table.updated,
        "active": table.active,
        "variables": [_public_variable(item, values_filter=values_filter) for item in chosen],
    }
    if _title_is_projection(table.title):
        published["projection"] = True
    return published


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
    terms: list[str] = []
    if values_filter and values_filter.strip():
        terms = [part for part in re.split(r"[\s,;]+", values_filter.casefold()) if len(part) >= 2]
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


def _sum_argument_help(kind: str) -> str:
    if kind == "sum_range":
        return (
            "sum_range is an argument beside values, not a key inside values. "
            'Call hent_data with values {"OMRÅDE": ["561"]} and '
            'sum_range {"ALDER": ["0", "17"]}.'
        )
    return (
        "sum_from is an argument beside values, not a key inside values. "
        'Call hent_data with values {"OMRÅDE": ["561"]} and sum_from {"ALDER": "80"}.'
    )


def _hoist_nested_sums(
    values: Any,
    sum_from: Any,
    sum_range: Any,
) -> tuple[Any, Any, Any, list[str]]:
    """Lift sum_from and sum_range that the model nested inside values."""
    notes: list[str] = []
    bad_outer = any(
        item is not None and not isinstance(item, dict) for item in (sum_from, sum_range)
    )
    if not isinstance(values, dict) or bad_outer:
        return values, sum_from, sum_range, notes
    values = dict(values)
    lifted = {"sum_from": dict(sum_from or {}), "sum_range": dict(sum_range or {})}
    used = {"sum_from": sum_from is not None, "sum_range": sum_range is not None}

    def take(kind: str, var: Any, spec: Any, clash: str) -> None:
        previous = lifted[kind].get(var)
        if previous is not None and str(previous).strip() != str(spec).strip():
            raise StatbankError(f"{clash} {_sum_argument_help(kind)}")
        lifted[kind][var] = spec
        used[kind] = True

    for kind in ("sum_from", "sum_range"):
        if kind not in values:
            continue
        nested = values.pop(kind)
        if not isinstance(nested, dict) or not nested:
            raise StatbankError(_sum_argument_help(kind))
        for var, spec in nested.items():
            take(
                kind,
                var,
                spec,
                (
                    f"{kind} for {var} is inside values and also passed beside values, "
                    "and they differ."
                ),
            )
        notes.append(f"Moved {kind} out of values. {_sum_argument_help(kind)}")
    for var, raw in list(values.items()):
        if not isinstance(raw, dict):
            continue
        keys = {str(key) for key in raw}
        if keys not in ({"sum_from"}, {"sum_range"}):
            raise StatbankError(
                f"values for {var} must be a code or a list of codes. "
                f"{_sum_argument_help('sum_from')}"
            )
        kind = next(iter(raw))
        take(
            kind,
            var,
            raw[kind],
            f"{kind} for {var} disagrees with the value inside values.",
        )
        values.pop(var)
        notes.append(f"Moved {kind} for {var} out of values. {_sum_argument_help(kind)}")
    return (
        values,
        lifted["sum_from"] if used["sum_from"] else None,
        lifted["sum_range"] if used["sum_range"] else None,
        notes,
    )


def _coerce_sum_from(raw: Any) -> dict[str, str] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise StatbankError(_sum_argument_help("sum_from"))
    coerced: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, bool):
            raise StatbankError(_sum_argument_help("sum_from"))
        if isinstance(value, int):
            coerced[str(key)] = str(value)
        elif isinstance(value, str) and value.strip():
            coerced[str(key)] = value.strip()
        elif isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
            coerced[str(key)] = value[0].strip()
        else:
            raise StatbankError(
                f"sum_from for {key} must be one numeric code such as '80'. "
                f"{_sum_argument_help('sum_from')}"
            )
    return coerced


_VARIABLE_ALIASES = {
    "kom": "OMRÅDE",
    "kommune": "OMRÅDE",
    "kommuner": "OMRÅDE",
    "omraade": "OMRÅDE",
    "alder": "ALDER",
    "tid": "Tid",
    "aar": "Tid",
}


def _alias_arguments(
    table: _Table,
    values: dict[str, list[str]] | None,
    sum_from: dict[str, str] | None,
    sum_range: dict[str, list[str]] | None,
) -> tuple[
    dict[str, list[str]] | None, dict[str, str] | None, dict[str, list[str]] | None, list[str]
]:
    notes: list[str] = []

    def rename(mapping: dict[str, Any] | None) -> dict[str, Any] | None:
        if not mapping:
            return mapping
        renamed: dict[str, Any] = {}
        for key, value in mapping.items():
            found = table.variable(str(key))
            if found is not None:
                canonical = found.id
            else:
                target = _VARIABLE_ALIASES.get(_fold(str(key)).replace(" ", ""))
                match = table.variable(target) if target else None
                canonical = match.id if match is not None else str(key)
                if canonical != key:
                    notes.append(f"Used {canonical} for {key}.")
            if canonical in renamed:
                raise StatbankError(f"Variable {canonical} was given more than once.")
            renamed[canonical] = value
        return renamed

    return rename(values), rename(sum_from), rename(sum_range), notes


def _promote_open_ended(
    table: _Table,
    values: dict[str, list[str]] | None,
    sum_from: dict[str, str] | None,
    sum_range: dict[str, list[str]] | None,
) -> tuple[
    dict[str, list[str]] | None, dict[str, str] | None, dict[str, list[str]] | None, list[str]
]:
    """Turn 80+, >=80 and 0-17 into the sum the API actually understands.

    A code of 80 is one year of age. The models write 80+ for everyone older,
    and the close match '80 år' then charts that one year.
    """
    notes: list[str] = []
    if not values:
        return values, sum_from, sum_range, notes
    values = {key: list(codes) for key, codes in values.items()}
    from_map = dict(sum_from or {})
    range_map = dict(sum_range or {})
    for key in list(values):
        variable = table.variable(key)
        if variable is None or variable.time:
            continue
        numeric, _explicit = _split_numeric_run(variable.values)
        if numeric is None:
            continue
        kept: list[str] = []
        for token in values[key]:
            if _value_matches(variable, token):
                kept.append(token)
                continue
            threshold = _open_ended_age(token)
            span = None if threshold is not None else _closed_age_span(token)
            if threshold is not None:
                note = (
                    f'Treated {variable.id} "{token}" as '
                    f'sum_from {{"{variable.id}": "{threshold}"}}.'
                )
                _store_sum(from_map, variable.id, threshold, note, notes)
            elif span is not None:
                low, high = span
                _store_sum(
                    range_map,
                    variable.id,
                    [low, high],
                    f'Treated {variable.id} "{token}" as sum_range '
                    f'{{"{variable.id}": ["{low}", "{high}"]}}.',
                    notes,
                )
            else:
                kept.append(token)
        if kept:
            values[key] = kept
        else:
            values.pop(key)
    return (
        values,
        from_map or None,
        range_map or None,
        notes,
    )


def _store_sum(
    target: dict[str, Any], variable_id: str, value: Any, note: str, notes: list[str]
) -> None:
    previous = target.get(variable_id)
    if previous is not None and previous != value:
        raise StatbankError(
            f"Variable {variable_id} asks for two different totals ({previous} and {value}). "
            "Pass one sum_from or sum_range."
        )
    target[variable_id] = value
    if note not in notes:
        notes.append(note)


def _open_ended_age(token: str) -> str | None:
    folded = _fold(token.strip())
    match = re.fullmatch(
        r">=?\s*(\d+)|(\d+)\s*\+|(\d+)\s*-|(\d+)\s+og\s+derover|over\s+(\d+)|(\d+)\s+aar\s+og\s+derover",
        folded,
    )
    if match is None:
        return None
    return next(group for group in match.groups() if group is not None)


def _closed_age_span(token: str) -> tuple[str, str] | None:
    folded = _fold(token.strip())
    match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", folded) or re.fullmatch(
        r"(\d+)\s+til\s+(\d+)", folded
    )
    if match is None:
        return None
    low, high = int(match.group(1)), int(match.group(2))
    if low > high:
        return None
    return str(low), str(high)


def _drop_equivalent_value_sums(
    values: dict[str, list[str]] | None,
    sum_from: dict[str, str] | None,
) -> tuple[dict[str, list[str]] | None, list[str]]:
    """Treat values '>=80' plus sum_from '80' as one request."""
    if not values or not sum_from:
        return values, []
    notes: list[str] = []
    folded = {key.casefold(): key for key in values}
    for code, start in sum_from.items():
        key = folded.get(str(code).casefold())
        if key is None:
            continue
        if _equivalent_sum(values[key], start):
            shown = ", ".join(values.pop(key))
            notes.append(
                f"Ignored values for {key} ({shown}); it is the same request as "
                f'sum_from {{"{code}": "{start}"}}.'
            )
            continue
        raise StatbankError(
            f"Variable {key} is listed in both values and sum_from, and they differ. "
            f"Remove {key} from values. "
            f'Keep sum_from {{"{code}": "{start}"}} and pass the other variables in values.'
        )
    return values, notes


def _equivalent_sum(raw: list[str], start: str) -> bool:
    token = start.strip()
    if not token.isdigit():
        return False
    threshold = int(token)
    if not raw:
        return False
    for item in raw:
        text = item.strip()
        lower_bound = re.fullmatch(r">=\s*(\d+)", text)
        if lower_bound and int(lower_bound.group(1)) == threshold:
            continue
        if text.isdigit() and int(text) == threshold:
            continue
        if text == f"{threshold}+":
            continue
        return False
    return True


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
        ages = selections.get("ALDER") or []
        age_match = re.fullmatch(r">=(\d+)", ages[0].strip()) if len(ages) == 1 else None
        if age_match is not None:
            age = age_match.group(1)
            parts.append(
                f'Use sum_from {{"ALDER": "{age}"}} for ages {age} and older, '
                f'not values [">={age}"].'
            )
        raise StatbankError(" ".join(parts))


def _build_selections(
    table: _Table,
    values: dict[str, list[str]] | None,
    sum_from: dict[str, str] | None,
    sum_range: dict[str, list[str]] | None = None,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    values = values or {}
    sum_from = sum_from or {}
    sum_range = sum_range or {}
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
                f"Variable {variable.id} is listed in both values and sum_from, and they differ. "
                f"Remove {variable.id} from values."
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
    if variable.time:
        raise StatbankError(
            f"Unknown value '{token}' for variable {variable.id} on table {table.id}. "
            f"{_time_advice(variable, token)}"
        )
    suggestions = _close_values(variable, token)
    advice = suggestions or _value_hint(variable)
    raise StatbankError(
        f"Unknown value '{token}' for variable {variable.id} on table {table.id}. {advice}"
    )


def _time_advice(variable: _Variable, token: str) -> str:
    """The real period span. Near-miss codes such as 2023K1 are not useful."""
    if not variable.values:
        return "Call tabelinfo to list valid codes."
    first = variable.values[0]["id"]
    last = variable.values[-1]["id"]
    advice = f"Available: {first}–{last}."
    hint = _year_span_hint(variable, token)
    if hint:
        advice = f"{advice} {hint}"
    return advice


def _title_is_projection(title: str) -> bool:
    """Population projections are tables. A mixed table such as HST5 is not."""
    return "fremskriv" in _fold(title)


def _label_is_forecast(text: str) -> bool:
    folded = _fold(text)
    return "fremskriv" in folded or re.search(r"\bprognose", folded) is not None


def _mark_projections(table: _Table, rows: list[dict[str, Any]]) -> list[str]:
    table_level = _title_is_projection(table.title)
    marked = False
    for row in rows:
        labels = [
            str(value)
            for key, value in row.items()
            if key not in {"value", "unpublished", "year_total", "projection"}
            and isinstance(value, str)
        ]
        if table_level or any(_label_is_forecast(label) for label in labels):
            row["projection"] = True
            marked = True
    if table_level and marked:
        return [
            "This table is a projection (fremskrivning), not observed figures. "
            "Rows are marked projection."
        ]
    if marked:
        return [
            "Rows marked projection are forecasts (prognose or fremskrivning), "
            "not observed figures."
        ]
    return []


def _measure_sides(text: str) -> tuple[bool, bool]:
    folded = _fold(text)
    flow = any(stem in folded for stem in _FLOW_STEMS) or re.search(r"\bdode\b", folded) is not None
    unit = folded.strip(" .")
    stock = _STOCK_RE.search(folded) is not None or unit in _LEVEL_UNITS
    return flow, stock


def _one_measure(text: str) -> str | None:
    """'sum' for a flow, 'average' for a level, None when the text is silent or both."""
    flow, stock = _measure_sides(text)
    if flow == stock:
        return None
    return "sum" if flow else "average"


def _measure_method(table: _Table, labels: str) -> str | None:
    """Flow is added. A stock or level is averaged. Mixed or unknown signals are not guessed."""
    table_text = f"{table.title} {table.description} {table.unit}"
    table_method = _one_measure(table_text)
    label_method = _one_measure(labels) if labels.strip() else None
    if table_method and label_method and table_method != label_method:
        return None
    return label_method or table_method


def _prepare_periods(
    table: _Table,
    values: dict[str, list[str]] | None,
    today: date,
) -> tuple[dict[str, list[str]] | None, list[_YearPlan], list[str]]:
    """Rewrite Danish period phrases and bare years into the table's own codes."""
    if not values:
        return values, [], []
    time_var = next((variable for variable in table.variables if variable.time), None)
    if time_var is None or time_var.id not in values:
        return values, [], []
    plans: list[_YearPlan] = []
    notes: list[str] = []
    rewritten: list[str] = []
    for token in values[time_var.id]:
        if _is_expression(token) or _value_matches(time_var, token):
            rewritten.append(token)
            continue
        meaning = _period_meaning(table, time_var, token, today)
        if meaning is None:
            rewritten.append(token)
            continue
        codes, plan, read_as = meaning
        rewritten.extend(codes)
        if plan is not None:
            plans.append(plan)
        if read_as and read_as.casefold() != token.strip().casefold():
            notes.append(f"Read {token.strip()!r} as {read_as}.")
    updated = dict(values)
    updated[time_var.id] = rewritten
    return updated, plans, notes


def _period_meaning(
    table: _Table,
    variable: _Variable,
    token: str,
    today: date,
) -> tuple[list[str], _YearPlan | None, str] | None:
    """Codes to request, an optional year plan, and the text of the read-as note.

    None means the token is not a period phrase. The caller leaves it for the
    ordinary unknown-value error.
    """
    text = " ".join(_fold(token).replace("–", " ").replace("—", " ").split())
    text = text.replace(".", " ").replace("  ", " ").strip()
    text = " ".join(text.split())
    latest = _latest_period(variable)
    if latest is None:
        return None
    latest_code, latest_year, latest_quarter, latest_month = latest

    if text in {"seneste", "nyeste", "seneste periode", "nyeste periode"}:
        return [latest_code], None, latest_code
    # i år / i fjor / sidste måned follow the calendar, not the table's last period.
    if text in {"i aar", "iaar"}:
        return _year_meaning(table, variable, token, today.year, unpublished=True)
    if text in {"i fjor", "sidste aar", "forrige aar"}:
        return _year_meaning(table, variable, token, today.year - 1, unpublished=True)
    if text in {"seneste kvartal", "sidste kvartal"}:
        quarter = _relative_quarter(variable, token, latest_year, latest_quarter, text)
        return _quarter_meaning(table, variable, token, *quarter)
    if text == "seneste maaned":
        if latest_month is None:
            raise StatbankError(
                f"{token.strip()!r} needs a monthly table. {_time_advice(variable, token)}"
            )
        return [latest_code], None, latest_code
    if text == "sidste maaned":
        previous = today.replace(day=1) - timedelta(days=1)
        return _calendar_code(table, variable, token, f"{previous.year}M{previous.month:02d}")
    month = _month_phrase(text)
    if month is not None:
        year, number = month
        return _one_code(table, variable, token, f"{year}M{number:02d}")
    quarter = _quarter_phrase(text)
    if quarter is not None:
        year, number = quarter
        if number < 1 or number > 4:
            raise StatbankError(
                f"{token.strip()!r} is not a quarter. Use 1 to 4. {_time_advice(variable, token)}"
            )
        return _quarter_meaning(table, variable, token, year, number)
    if re.fullmatch(r"\d{4}", text):
        return _year_meaning(table, variable, token, int(text))
    return None


def _latest_period(
    variable: _Variable,
) -> tuple[str, int, int | None, int | None] | None:
    best: tuple[tuple[int, int, int], str, int, int | None, int | None] | None = None
    for item in variable.values:
        parsed = _parsed_period(item["id"])
        if parsed is None:
            continue
        year, kind, index = parsed
        key = (year, kind, index)
        if best is None or key > best[0]:
            quarter = index if kind == 1 else (index - 1) // 3 + 1 if kind == 3 else None
            month = index if kind == 3 else None
            best = (key, item["id"], year, quarter, month)
    if best is None:
        return None
    return best[1], best[2], best[3], best[4]


def _parsed_period(code: str) -> tuple[int, int, int] | None:
    """(year, kind, index). Kind is 0 year, 1 quarter, 2 half, 3 month."""
    match = re.fullmatch(r"(\d{4})(?:(K|Q)([1-4])|(H)([12])|(M)(\d{2}))?", code.strip(), re.I)
    if match is None:
        return None
    year = int(match.group(1))
    if match.group(2):
        return year, 1, int(match.group(3))
    if match.group(4):
        return year, 2, int(match.group(5))
    if match.group(6):
        return year, 3, int(match.group(7))
    return year, 0, 0


def _relative_quarter(
    variable: _Variable,
    token: str,
    year: int,
    quarter: int | None,
    text: str,
) -> tuple[int, int]:
    if quarter is None:
        raise StatbankError(
            f"{token.strip()!r} needs a quarterly or monthly table. {_time_advice(variable, token)}"
        )
    if text.startswith("sidste"):
        index = year * 4 + (quarter - 1) - 1
        return index // 4, index % 4 + 1
    return year, quarter


def _month_phrase(text: str) -> tuple[int, int] | None:
    named = "|".join(_MONTH_NUMBERS)
    front = re.fullmatch(rf"(?:i )?({named}) (\d{{4}})", text)
    if front:
        return int(front.group(2)), _MONTH_NUMBERS[front.group(1)]
    back = re.fullmatch(rf"(\d{{4}}) ({named})", text)
    if back:
        return int(back.group(1)), _MONTH_NUMBERS[back.group(2)]
    return None


def _quarter_phrase(text: str) -> tuple[int, int] | None:
    words = "|".join(_QUARTER_WORDS)
    patterns = (
        (r"(\d{4})\s*(\d)\s*(?:kvartal|kvt)", True),
        (r"(\d{4})\s*[kq]\s*(\d)", True),
        (r"(\d)\s*(?:kvartal|kvt)\s*(\d{4})", False),
        (r"[kq]\s*(\d)\s*(\d{4})", False),
        (rf"({words})\s+kvartal\s+(\d{{4}})", False),
    )
    for pattern, year_first in patterns:
        match = re.fullmatch(pattern, text)
        if match is None:
            continue
        first, second = match.group(1), match.group(2)
        if year_first:
            return int(first), int(second)
        quarter = _QUARTER_WORDS[first] if first in _QUARTER_WORDS else int(first)
        return int(second), quarter
    return None


def _not_published(token: str, code: str, table: _Table, variable: _Variable) -> StatbankError:
    latest = _latest_period(variable)
    shown = latest[0] if latest else "none"
    return StatbankError(
        f"{token.strip()!r} means {code}, which is not published in {table.id}. "
        f"Latest available period: {shown}."
    )


def _calendar_code(
    table: _Table,
    variable: _Variable,
    token: str,
    code: str,
) -> tuple[list[str], _YearPlan | None, str]:
    found = _code_on(variable, code)
    if found is None:
        raise _not_published(token, code, table, variable)
    return [found], None, found


def _one_code(
    table: _Table,
    variable: _Variable,
    token: str,
    code: str,
) -> tuple[list[str], _YearPlan | None, str]:
    found = _code_on(variable, code)
    if found is None:
        raise StatbankError(
            f"{token.strip()!r} is {code}, which is not in table {table.id}. "
            f"{_time_advice(variable, code)}"
        )
    return [found], None, found


def _quarter_meaning(
    table: _Table,
    variable: _Variable,
    token: str,
    year: int,
    quarter: int,
) -> tuple[list[str], _YearPlan | None, str]:
    quarter_code = _code_on(variable, f"{year}K{quarter}") or _code_on(
        variable, f"{year}Q{quarter}"
    )
    if quarter_code is not None:
        return [quarter_code], None, quarter_code
    months = [f"{year}M{month:02d}" for month in range((quarter - 1) * 3 + 1, quarter * 3 + 1)]
    found = [code for month in months if (code := _code_on(variable, month))]
    if len(found) == 3:
        shown = ", ".join(found)
        return found, None, shown
    target = quarter_code or f"{year}K{quarter}"
    raise StatbankError(
        f"{token.strip()!r} is {target}, which is not in table {table.id}. "
        f"{_time_advice(variable, token)}"
    )


def _year_meaning(
    table: _Table,
    variable: _Variable,
    token: str,
    year: int,
    *,
    unpublished: bool = False,
) -> tuple[list[str], _YearPlan | None, str]:
    exact = _code_on(variable, str(year))
    if exact is not None:
        return [exact], None, exact
    codes = [
        item["id"]
        for item in variable.values
        if re.fullmatch(
            rf"{year}(?:K[1-4]|Q[1-4]|M(?:0[1-9]|1[0-2])|H[12])",
            item["id"],
            re.IGNORECASE,
        )
    ]
    if not codes:
        if unpublished:
            raise _not_published(token, str(year), table, variable)
        raise StatbankError(
            f"{token.strip()!r} is {year}, which is not in table {table.id}. "
            f"{_time_advice(variable, str(year))}"
        )
    plan = _YearPlan(year=str(year), codes=tuple(codes), complete=_year_is_complete(codes))
    return list(codes), plan, str(year)


def _year_is_complete(codes: list[str]) -> bool:
    suffixes = {code[4:].upper() for code in codes}
    if suffixes <= {f"K{i}" for i in range(1, 5)}:
        return suffixes == {f"K{i}" for i in range(1, 5)}
    if suffixes <= {f"Q{i}" for i in range(1, 5)}:
        return suffixes == {f"Q{i}" for i in range(1, 5)}
    months = {f"M{i:02d}" for i in range(1, 13)}
    if suffixes <= months and all(item.startswith("M") for item in suffixes):
        return suffixes == months
    if suffixes <= {"H1", "H2"}:
        return suffixes == {"H1", "H2"}
    return False


def _code_on(variable: _Variable, code: str) -> str | None:
    for item in variable.values:
        if item["id"].casefold() == code.casefold():
            return item["id"]
    return None


def _collapse_years(
    table: _Table,
    rows: list[dict[str, Any]],
    plans: list[_YearPlan],
) -> tuple[list[dict[str, Any]], list[str]]:
    time_id = next((variable.id for variable in table.variables if variable.time), None)
    if time_id is None:
        return rows, []
    notes: list[str] = []
    consumed: set[int] = set()
    produced: dict[int, list[dict[str, Any]]] = {}
    seen_years: set[str] = set()
    for plan in plans:
        if plan.year in seen_years:
            continue
        seen_years.add(plan.year)
        wanted = {code.casefold() for code in plan.codes}
        indexes = [
            index
            for index, row in enumerate(rows)
            if index not in consumed and str(row.get(time_id) or "").casefold() in wanted
        ]
        chosen = [rows[index] for index in indexes]
        labels = " ".join(
            str(value)
            for row in chosen
            for key, value in row.items()
            if key not in {time_id, "value", "unpublished", "projection", "year_total"}
            and isinstance(value, str)
        )
        method = _measure_method(table, labels) if plan.complete else None
        aggregated = _aggregate_year(chosen, time_id, plan, method) if method else None
        if aggregated is None:
            notes.append(_year_skip_note(plan, method))
            continue
        notes.append(_year_method_note(plan, method))
        if indexes:
            consumed.update(indexes)
            produced[indexes[0]] = aggregated
    if not consumed:
        return rows, notes
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index in produced:
            output.extend(produced[index])
        if index not in consumed:
            output.append(row)
    return output, notes


def _listed_periods(codes: tuple[str, ...]) -> str:
    items = list(codes)
    if len(items) < 2:
        return ", ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _year_skip_note(plan: _YearPlan, method: str | None) -> str:
    listed = _listed_periods(plan.codes)
    if not plan.complete:
        return (
            f"{plan.year} is not a full year in this table ({listed}). "
            "No yearly figure was calculated."
        )
    if method is None:
        return (
            f"{plan.year} cannot be combined into one figure ({listed}). "
            "The table does not say whether the measure is a flow (add the periods) "
            "or a level (average them), so no yearly figure was calculated. "
            "The periods are listed separately."
        )
    return (
        f"{plan.year} could not be combined ({listed}) because a period has no number. "
        "No yearly figure was calculated. The periods are listed separately."
    )


def _year_method_note(plan: _YearPlan, method: str) -> str:
    listed = _listed_periods(plan.codes)
    if method == "sum":
        return (
            f"{plan.year} is the sum of {listed}. The measure is a flow, so the periods are added."
        )
    return (
        f"{plan.year} is the average of {listed}. "
        "The measure is a level at a point in time, so the periods are averaged."
    )


def _aggregate_year(
    rows: list[dict[str, Any]],
    time_id: str,
    plan: _YearPlan,
    method: str | None,
) -> list[dict[str, Any]] | None:
    if method not in {"sum", "average"} or not rows:
        return None
    expected = [code.casefold() for code in plan.codes]
    groups: dict[tuple[tuple[str, Any], ...], dict[str, dict[str, Any]]] = {}
    order: list[tuple[tuple[str, Any], ...]] = []
    for row in rows:
        period = str(row.get(time_id) or "").casefold()
        identity = tuple(
            (key, row.get(key))
            for key in row
            if key not in {time_id, "value", "unpublished", "projection", "year_total"}
        )
        bucket = groups.get(identity)
        if bucket is None:
            bucket = {}
            groups[identity] = bucket
            order.append(identity)
        bucket[period] = row
    aggregated: list[dict[str, Any]] = []
    for identity in order:
        series = groups[identity]
        if any(code not in series for code in expected):
            return None
        values: list[int | float] = []
        for code in expected:
            row = series[code]
            if row.get("unpublished"):
                return None
            value = row.get("value")
            if isinstance(value, bool) or not isinstance(value, int | float):
                return None
            values.append(value)
        total = _combine_period_values(values, method)
        combined = {key: item for key, item in identity}
        combined[time_id] = plan.year
        combined["value"] = total
        combined["year_total"] = {
            "method": method,
            "measure": "flow" if method == "sum" else "level",
            "periods": list(plan.codes),
        }
        aggregated.append(combined)
    return aggregated or None


def _combine_period_values(values: list[int | float], method: str) -> int | float:
    total = sum(values)
    if method == "sum":
        if isinstance(total, float) and total.is_integer():
            return int(total)
        return total
    if all(isinstance(value, int) for value in values) and total % len(values) == 0:
        return int(total // len(values))
    return total / len(values)


def _year_span_hint(variable: _Variable, token: str) -> str:
    """A bare year on a quarterly or monthly table names the sub-periods."""
    year = token.strip()
    if not re.fullmatch(r"\d{4}", year):
        return ""
    codes = [
        item["id"] for item in variable.values if item["id"].startswith(year) and item["id"] != year
    ]
    if len(codes) < 2:
        return ""
    return f"For {year}, pass >={codes[0]}<={codes[-1]}."


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


def _prefix_hits(variable: _Variable, token: str) -> list[dict[str, str]]:
    """Labels whose hyphen or space parts start with the token.

    Difflib's cutoff misses a short prefix of a hyphenated name, so Lyngby
    never suggests Lyngby-Taarbæk. Suggest only; the value is still unknown.
    """
    folded = _fold(token)
    if len(folded) < 4:
        return []
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in variable.values:
        if item["id"] in seen:
            continue
        parts = [part for part in re.split(r"[\s\-]+", _fold(item["text"])) if part]
        if any(part.startswith(folded) for part in parts):
            seen.add(item["id"])
            found.append(item)
    return found


def _similar_values(variable: _Variable, token: str) -> list[dict[str, str]]:
    folded = _fold(token)
    pool: list[str] = []
    owner: dict[str, dict[str, str]] = {}
    for item in variable.values:
        for key in (_fold(item["text"]), _fold(item["id"])):
            if key and key not in owner:
                pool.append(key)
                owner[key] = item
    found = difflib.get_close_matches(folded, pool, n=3, cutoff=0.72)
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for key in found:
        item = owner[key]
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        items.append(item)
    return items


def _close_values(variable: _Variable, token: str) -> str:
    items = _prefix_hits(variable, token)[:3] or _similar_values(variable, token)
    if not items:
        return ""
    shown = [f"{item['text']} ({item['id']})" for item in items]
    return "Did you mean " + ", ".join(shown) + "?"


def _is_expression(token: str) -> bool:
    return token == "*" or token.startswith((">", "<", "(", "sum(", "SUM(")) or "*" in token


def _sum_call(codes: list[str], label: str) -> str:
    if len(label) > 80:
        label = f"{codes[0]}+…+{codes[-1]}"
    return f"sum({label}|{label}={';'.join(codes)})"


def _numeric_codes(variable: _Variable, low: int, high: int | None = None) -> list[str]:
    codes = []
    for item in variable.values:
        if not item["id"].isdigit():
            continue
        number = int(item["id"])
        if number >= low and (high is None or number <= high):
            codes.append(item["id"])
    return codes


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
    codes = _numeric_codes(variable, threshold)
    if not codes:
        raise StatbankError(
            f"No codes on {variable.id} are numeric and >= {threshold}. {_value_hint(variable)}"
        )
    return _sum_call(codes, f"{threshold}+")


def _numeric_span(bounds: Any) -> tuple[int, int] | None:
    if not isinstance(bounds, list) or len(bounds) != 2:
        return None
    if not all(isinstance(item, str) for item in bounds):
        return None
    low, high = bounds[0].strip(), bounds[1].strip()
    if not (low.isdigit() and high.isdigit() and str(int(low)) == low and str(int(high)) == high):
        return None
    return int(low), int(high)


def _sum_range_or_codes(
    table: _Table,
    variable: _Variable,
    bounds: Any,
) -> tuple[str, list[str] | None]:
    span = _numeric_span(bounds)
    if span is not None:
        low, high = span
        if low > high:
            raise StatbankError(
                f"sum_range for {variable.id} must run from a lower code to a higher one "
                f"(got {low} then {high})."
            )
        codes = _numeric_codes(variable, low, high)
        if not codes:
            raise StatbankError(
                f"No numeric codes on {variable.id} fall between {low} and {high}. "
                f"{_value_hint(variable)}"
            )
        if len(codes) == 1:
            return codes[0], None
        return _sum_call(codes, f"{low}-{high}"), None
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
    return _sum_call(codes, "+".join(codes)), codes


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


def _label_key(text: str) -> str:
    return " ".join(_fold(text).split())


def _unique_total(variable: _Variable) -> dict[str, str] | None:
    """The one total, or none when several values could be it.

    A code TOT or IALT wins. Otherwise one label I alt, Total, Alle, or
    Hele landet. A single longer label that ends in "i alt" ("Mænd og kvinder
    i alt") is the same kind of total. Several of those, as on a crime
    breakdown, are not a total.
    """
    by_code = [item for item in variable.values if item["id"].casefold() in {"tot", "ialt"}]
    if len(by_code) == 1:
        return by_code[0]
    if by_code:
        return None
    exact_labels = {"i alt", "ialt", "alle", "total", "hele landet"}
    exact = [item for item in variable.values if _label_key(item["text"]) in exact_labels]
    if len(exact) == 1:
        return exact[0]
    if exact:
        return None
    styled = [item for item in variable.values if _label_key(item["text"]).endswith(" i alt")]
    if len(styled) == 1:
        return styled[0]
    return None


def _latest_value(variable: _Variable) -> dict[str, str] | None:
    latest = _latest_period(variable)
    if latest is None:
        return None
    code = latest[0]
    for item in variable.values:
        if item["id"] == code:
            return item
    return {"id": code, "text": code}


def _fill_required(table: _Table, selections: dict[str, list[str]]) -> list[dict[str, str]]:
    """Fill every variable the caller left out.

    A total (I alt, Total, Hele landet) is used when the question does not
    name that variable, so the model does not have to choose it. Omitted time
    is the latest published period. A required variable with no total is still
    an error that lists the codes.
    """
    assumed: list[dict[str, str]] = []
    missing: list[_Variable] = []
    for variable in table.variables:
        if variable.id in selections:
            continue
        if variable.time:
            latest = _latest_value(variable)
            if latest is None:
                continue
            selections[variable.id] = [latest["id"]]
            assumed.append({"variable": variable.id, "id": latest["id"], "text": latest["text"]})
            continue
        total = _unique_total(variable)
        if total is None:
            if not variable.eliminable:
                missing.append(variable)
            continue
        selections[variable.id] = [total["id"]]
        assumed.append({"variable": variable.id, "id": total["id"], "text": total["text"]})
    if missing:
        detail = " ".join(
            f"Variable {variable.id} on {table.id} has no total and must be set. "
            f"{_code_sample(variable)}"
            for variable in missing
        )
        raise StatbankError(detail)
    ordered = {
        variable.id: selections[variable.id]
        for variable in table.variables
        if variable.id in selections
    }
    selections.clear()
    selections.update(ordered)
    return assumed


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
