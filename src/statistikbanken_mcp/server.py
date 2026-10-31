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
    "Search with find_tabel, read value codes with tabelinfo, then fetch with hent_data. "
    "Ages in FOLK1A are single years from 0 to 125. To total an age group such as "
    '80 and older, pass sum_from, for example {"ALDER": "80"}. The API then '
    'returns one summed figure. A value like ">=80" does not sum; it returns '
    'every age on its own row. Omit Tid or pass "(1)" for the latest period. '
    'A span such as ">=2016K3" selects every period from that quarter onward.'
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

    Matches table id, title, and variable names. Matching is not case-sensitive.
    "befolkning" returns FOLK1A near the top. `limit` is 1–25 and defaults to 10.
    Each hit has id, title, unit, latest_period, and variables. The variable
    entries are names, not the codes hent_data expects; call tabelinfo for codes.
    An empty list means nothing matched.
    """
    return _guard(lambda: get_client().find_tables(query, limit=limit))


@mcp.tool()
def tabelinfo(table: str) -> dict[str, Any]:
    """Describe one table: its variables and the value codes hent_data accepts.

    Long runs of numeric codes are collapsed. On FOLK1A, ALDER has numeric_codes
    from "0" to "125" (the code is the age) plus the separate code IALT
    ("Alder i alt"). OMRÅDE, KØN, CIVILSTAND, and Tid are listed in full.
    eliminable means the variable can be left out of hent_data. A time variable
    also accepts "(1)" for the latest period and ranges such as ">=2016K3".
    """
    return _guard(lambda: get_client().table_info(table))


@mcp.tool()
def hent_data(
    table: str,
    values: dict[str, list[str]] | None = None,
    sum_from: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Fetch observations from a Statistikbanken table.

    Variables you omit are removed by the API, which keeps their total (for
    example both sexes). Tid is the exception: omitting it returns the latest
    period, the same as the value "(1)".

    `values` maps a variable id to the codes to return as separate rows. Take
    the ids from tabelinfo. These expressions are passed through: "*" (every
    code), "(1)" (latest period), "(-n+4)" (the four latest periods), and a
    range such as ">=2016K3". A range lists matching periods. It does not add
    them together.

    `sum_from` maps a variable id to a numeric code. Every numeric code from
    that one upward is added, by the API, into a single figure. For FOLK1A,
    {"ALDER": "80"} is everyone aged 80 through 125, labelled "80+". Do not
    also put ALDER in `values`, and do not pass ">=80" expecting a total.

    The result contains table, title, unit, updated, source, and rows. Each
    row uses variable ids as keys and labels as text, plus `value`.
    """
    return _guard(lambda: get_client().fetch(table, values, sum_from))


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
