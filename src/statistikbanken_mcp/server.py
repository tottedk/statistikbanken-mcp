"""MCP server with three Statistikbanken tools: find_tabel, tabelinfo, hent_data."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any, TypeVar

import httpx
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from statistikbanken_mcp.statbank import StatbankClient, StatbankError

# The server logs at INFO. One line per HTTP call from httpx hides tool errors.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

INSTRUCTIONS = (
    "Tools for Danmarks Statistik (Statistikbanken). "
    "Search with find_tabel, read codes with tabelinfo, then fetch with hent_data. "
    "Search matches some of the words, a few everyday synonyms, and municipality names "
    "from FOLK1A. A miss is an error that suggests a broader query. "
    "The table title, description, unit, and period say what a number measures. "
    "If they only partly cover the question, say so. "
    "Values may be codes or labels, for example Silkeborg or København. "
    'sum_from {"ALDER": "80"} totals 80 and older. '
    'sum_range {"ALDER": ["0", "17"]} totals a numeric range. '
    'sum_range {"ALDER": ["25-29", "30-34"]} adds those codes. '
    'A value like ">=80" returns every code on its own row. It does not add them. '
    "Do not sum a time variable across several places or a range such as >=80. "
    'For the change over time, pass Tid in values as ">=2015K1". '
    "A values entry may be one code or several codes separated by a comma. "
    "A trailing 0 marked unpublished is a period the API has not published yet. "
    'Omit Tid or pass "(1)" for the latest period.'
)

mcp = MCPServer("statistikbanken", instructions=INSTRUCTIONS)

_client: StatbankClient | None = None
_client_lock = threading.Lock()
T = TypeVar("T")


def get_client() -> StatbankClient:
    global _client
    with _client_lock:
        if _client is None:
            _client = StatbankClient()
        return _client


def set_client(client: StatbankClient | None) -> StatbankClient | None:
    """Replace the client used by the tools. Returns the previous one."""
    global _client
    with _client_lock:
        previous = _client
        _client = client
        return previous


def _guard(fn: Callable[[], T]) -> T:
    try:
        return fn()
    except StatbankError as exc:
        raise ToolError(str(exc)) from exc
    except httpx.HTTPError as exc:
        raise ToolError(
            "Could not reach Statistikbanken (https://api.statbank.dk). Try again."
        ) from exc


@mcp.tool()
def find_tabel(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search Statistikbanken tables by free text.

    Matches table id, title, and variable names. A hit can match some of the
    words rather than every word, and a few everyday synonyms are recognised
    (for example bruttoledighed, huspriser, husdyr, population, people, indbrud).
    Municipality names are matched against FOLK1A areas and returned on that hit
    as matched_values with their codes. Matching is not case-sensitive.
    "befolkning" returns FOLK1A near the top. `limit` is 1–25 and defaults to 10.
    Each hit has id, title, unit, latest_period, and variables. The title is the
    table's own wording and can be narrower than the query. Read it, then
    tabelinfo, before treating the hit as the answer. If nothing matches, the
    tool returns an error that suggests a broader subject. Variable entries are
    names; hent_data also accepts the labels from tabelinfo.
    """
    return _guard(lambda: get_client().find_tables(query, limit=limit))


@mcp.tool()
def tabelinfo(
    table: str,
    variable: str | None = None,
    values_filter: str | None = None,
) -> dict[str, Any]:
    """Describe one table: its variables and the value codes hent_data accepts.

    Compare title, description, and unit with the question before fetching.
    They say what the table measures. Value labels do too: "Medlemmer" is a
    count of memberships, and "Gennemsnit for personer i gruppen (kr.)" is the
    unit when the table's own unit is "-".
    Long runs of numeric codes are collapsed. On FOLK1A, ALDER has numeric_codes
    from "0" to "125" plus IALT. Time variables are a time_span (first, last,
    count) rather than every period. Other value lists are capped; value_count
    is the full size and values_truncated means the list was cut. Pass
    values_filter to search codes and labels. Several words, separated by a
    comma or a space, match any of them (for example "Rudersdal, Hørsholm").
    A variable with no match keeps its normal list and says so. `total`, when
    present, is the code used if that required variable is omitted in hent_data.
    eliminable means the variable can be left out. A time variable also accepts
    "(1)" for the latest period and ranges such as ">=2016K3".
    """
    return _guard(
        lambda: get_client().table_info(table, variable=variable, values_filter=values_filter)
    )


@mcp.tool()
def hent_data(
    table: str,
    values: dict[str, list[str] | str] | None = None,
    sum_from: dict[str, str] | None = None,
    sum_range: dict[str, list[str] | str] | None = None,
) -> dict[str, Any]:
    """Fetch observations from a Statistikbanken table.

    Variables you omit are removed by the API, which keeps their total (for
    example both sexes). Tid is the exception: omitting it returns the latest
    period, the same as the value "(1)".

    `values` maps a variable id to codes or labels, matched without case
    sensitivity ("Silkeborg", "København", "kobenhavn"). Each entry may be a
    list, one code ("230"), or several codes separated by a comma
    ("Rudersdal, Hørsholm"). An unknown value names the closest labels and,
    for Tid, the latest period. These expressions are passed through: "*"
    (every code), "(1)" (latest period), "(-n+4)" (the four latest periods),
    and a range such as ">=2016K3". A range lists matching periods. It does
    not add them together.

    A variable that cannot be omitted and has one total (code TOT or IALT, or
    text "I alt") is filled with that total and listed in `defaults`. If it
    has no total, the call fails and lists the codes to pass.

    `sum_from` maps a variable id to a numeric code. Every numeric code from
    that one upward is added into a single figure. {"ALDER": "80"} is everyone
    aged 80 and older, labelled "80+".

    `sum_range` is either two numeric codes, low then high, or a longer list
    of codes to add. {"ALDER": ["0", "17"]} is ages 0 through 17. {"ALDER":
    ["25-29", "30-34"]} adds those bands. {"Tid": ["2023K1", "2023K2",
    "2023K3", "2023K4"]} adds those quarters when the API allows a sum, and
    otherwise adds the published rows, but only when every other variable is
    one plain code. Two periods in sum_range are not a range from the first
    to the second: that call is rejected. So is a time sum combined with
    several values, or with a range such as ">=80", because adding those
    cells would hide rows. For the change over time, pass
    {"Tid": [">=2015K1"]} in values and use sum_from for ages. Do not also
    put a summed variable in `values` or `sum_from`. ">=80" returns one row
    per code.

    The result contains table, title, description, unit, periods, updated,
    source, and rows. Title, description, unit, and period say what the number
    measures. If unit is "-", the selected value labels carry the measurement.
    If those words only partly cover the question, say so. A row with
    unpublished true is a trailing 0: the API uses 0 for a period it has not
    published yet. Each row uses variable ids as keys and labels as text, plus
    `value`.
    """
    return _guard(lambda: get_client().fetch(table, values, sum_from, sum_range))


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
