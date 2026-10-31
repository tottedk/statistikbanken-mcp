# statistikbanken-mcp

An MCP server for [Danmarks Statistik](https://www.dst.dk)'s open Statistikbanken API (`https://api.statbank.dk/v1`). No API key. Three tools, stdio transport.

Data is published by Danmarks Statistik and is free to reuse with attribution ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Run

```bash
uv run statistikbanken-mcp
```

The process speaks MCP on stdin/stdout and waits for a client.

### Cursor

Add this to your MCP config (`~/.cursor/mcp.json`, or the project `.cursor/mcp.json`). Use the absolute path of this repo:

```json
{
  "mcpServers": {
    "statistikbanken": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/statistikbanken-mcp",
        "run",
        "statistikbanken-mcp"
      ]
    }
  }
}
```

### MCP Inspector

```bash
uv run mcp dev src/statistikbanken_mcp/server.py
```

## Tools

Names are Danish. Arguments and descriptions are English.

### `find_tabel`

Free-text search over table id, title, and variable names.

```text
find_tabel(query="befolkning", limit=5)
```

Returns `id`, `title`, `unit`, `latest_period`, and `variables` (names, not codes). `befolkning` returns FOLK1A first. `limit` is 1–25, default 10.

### `tabelinfo`

Variables and the value codes `hent_data` accepts.

```text
tabelinfo(table="FOLK1A")
```

On FOLK1A that is OMRÅDE (including Rudersdal `230`, Hørsholm `223`, Gentofte `157`), KØN, ALDER as single years `0`–`125` plus `IALT`, CIVILSTAND, and Tid (for example `2026K3`). A long run of numeric codes is collapsed to `numeric_codes` instead of listing every year.

### `hent_data`

Fetch rows for a table. Omitted variables are dropped by the API (the total remains, for example both sexes). Omitting Tid returns the latest period, same as `"(1)"`.

`>=80` returns one row per year. `sum_from` asks the API to add every numeric code from that code upward into one figure. `sum_range` does the same for a closed range, low then high:

```text
hent_data(
  table="FOLK1A",
  values={
    "OMRÅDE": ["230"],
    "KØN": ["TOT"],
    "CIVILSTAND": ["TOT"],
    "Tid": ["(1)"]
  },
  sum_from={"ALDER": "80"}
)
```

The row is labelled `80+` and `value` is one integer. Ages 0 through 17 are one call:

```text
hent_data(
  table="FOLK1A",
  values={"OMRÅDE": ["230", "223"], "KØN": ["TOT"], "CIVILSTAND": ["TOT"], "Tid": ["2026K3"]},
  sum_range={"ALDER": ["0", "17"]}
)
```

Those rows are labelled `0-17`. A period range such as `">=2016K3"` returns every matching period as its own row.

The result includes `title`, `description`, `unit`, and `periods`. Read them before answering. A search hit can be narrower than the question: `forening` returns both `Foreningsmedlemskab` and `Idrætsforeninger og medlemmer`, and only the title says which one it is. When `unit` is `-`, the selected value label carries the measurement, such as average income in kroner.

Unknown table ids and value codes come back as a short message that says what was wrong and whether to call `find_tabel` or `tabelinfo`.

## Acceptance questions

These are the questions the tools are meant to answer:

1. Hvor mange over 80 bor i Rudersdal? Latest quarter. `sum_from` `{"ALDER": "80"}`, område `230`, Tid `"(1)"`.
2. Hvordan har befolkningstallet i Rudersdal udviklet sig de sidste 10 år? ALDER `IALT`, Tid `">=2016K3"`.
3. Hvilken kommune har flest indbyggere over 80: Rudersdal, Hørsholm eller Gentofte? Same sum as question 1, områder `230`, `223`, and `157`.
4. How many children aged 0–17 live in Rudersdal and in Hørsholm? `sum_range` `{"ALDER": ["0", "17"]}`, områder `230` and `223`.
5. Average income in Rudersdal and Hørsholm? `INDKP105`, ENHED `118` (average per person, kr.), Tid `2025`.
6. How many people in Rudersdal are members of an association? There is no all-association table by municipality. The nearest hit, `IDRFOR01`, counts sports-club memberships, not unique persons. The title and description say so.

## Tests

Fixtures in `tests/fixtures/` are responses captured from the live API.

```bash
uv sync --group dev
uv run pytest                  # unit tests and live API tests
uv run pytest -m "not live"    # recorded fixtures only
uv run pytest -m live          # the real API, including one stdio client session
uv run ruff check .
uv run ruff format --check .
```
