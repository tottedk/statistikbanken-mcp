# statistikbanken-mcp

An MCP server for [Danmarks Statistik](https://www.dst.dk)'s public Statistikbanken API. No API key.

Statistikbanken has more than 2,000 tables and no free-text search. This server gives an assistant three small tools, so a question such as "How many people over 80 live in Rudersdal?" comes back as a number from a named table and period, with Danmarks Statistik as the source.

## Example

**How many people over 80 live in Rudersdal?**

`find_tabel("befolkning")` returns `FOLK1A`, "Befolkningen den 1. i kvartalet", latest period `2026K3`.

```text
hent_data(
  table="FOLK1A",
  values={"OMRÅDE": ["Rudersdal"], "KØN": ["TOT"], "CIVILSTAND": ["TOT"], "Tid": ["(1)"]},
  sum_from={"ALDER": "80"}
)
```

Ages in FOLK1A are single years. `sum_from` asks the API to add 80 through 125. The result is one row:

| Table | Period | Place | Age | Value |
|---|---|---|---|---|
| FOLK1A | 2026K3 | Rudersdal | 80+ | 5,045 |

Unit is people (`Antal`). Source: Danmarks Statistik.

**Children aged 0–17 in Rudersdal and Hørsholm**, same table and quarter, with `sum_range={"ALDER": ["0", "17"]}`:

| Place | Value |
|---|---|
| Rudersdal | 12,770 |
| Hørsholm | 4,916 |

**A misspelled name.** `hent_data` with `OMRÅDE: ["Kobenhavn"]` resolves to København. Total population in FOLK1A, 2026K3: 670,389.

These three calls were run against `https://api.statbank.dk` on 31 October 2026.

## Tools

Names are Danish. Arguments and descriptions are English.

| Tool | Purpose | Main inputs | Behaviour worth knowing |
|---|---|---|---|
| `find_tabel` | Find a table | `query`, `limit` (1–25, default 10) | Matches id, title, and variable names. Some words are enough. A few synonyms are built in (`population`, `bruttoledighed`, `huspriser`, `husdyr`, `indbrud`). Municipality names are matched against FOLK1A and returned with their codes. A miss is an error, not an empty list. |
| `tabelinfo` | Read codes | `table`, optional `variable`, `values_filter` | Time is first, last, and count. Long value lists are capped; `values_filter` looks up the rest. |
| `hent_data` | Fetch rows | `table`, `values`, `sum_from`, `sum_range` | Values may be codes or labels (`Rudersdal`, `Kobenhavn`). A required variable with one total is filled in. One with no total returns its codes. `sum_from` and `sum_range` use the API's own sum. A trailing 0 marked `unpublished` is a period the API has not published yet. |

Read `title`, `description`, `unit`, and `periods` on the result. They say what the number measures. A search hit can be narrower than the question.

## Quick start

Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run statistikbanken-mcp
```

The process speaks MCP on stdin and stdout and waits for a client. It does not open a port.

Cursor, in `~/.cursor/mcp.json` or the project `.cursor/mcp.json`. Use the absolute path of this repo:

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

Any other host that launches a local command over stdio can use the same `command` and `args`.

MCP Inspector:

```bash
uv run mcp dev src/statistikbanken_mcp/server.py
```

That starts a local page where you can list the tools and call them.

## Web demo

A local page asks a question in Danish or English. The answer sits above the numbers (table, period, unit). The tool calls that produced them can be expanded. The page is a client of the MCP server over stdio. It does not read Statistikbanken itself. A development over time is one `hent_data` call with a period range such as `>=2015K1`, so each quarter is its own row. Several periods in one series draw a line chart on the page. Several places draw several lines. A sum of time that would hide a place or an age, including two periods passed where a range was meant, is an error that says how to ask for the series. The answer follows the language of the question, with Danish grouping (`5.045`) and period codes (`2026K3`) for a Danish question. A number in the model's prose is kept only when it is in the question, a fetched `hent_data` value (including `5.045`, `5 045`, and a decimal comma), a period or row label from those rows, or the fetched table title. A figure that appears only in `tabelinfo` or `find_tabel` is dropped. One fetched figure is also written by the page itself, for example `5.045 (FOLK1A, 2026K3, Antal, OMRÅDE: Rudersdal, ALDER: 80+)`. `80+` and `0-17` on an age variable are totals. A fetched place that the question did not ask for is not presented as the answer. A question that does not ask for a change uses the latest period. The model and Statistikbanken have separate time limits (45 seconds and 30 seconds). Hent CSV downloads those rows as UTF-8 with a BOM, a semicolon between columns, and a decimal comma, which is what Danish Excel expects. Each table links to `https://www.statistikbanken.dk/` plus the table id.

```bash
uv run statistikbanken-web
```

Then open http://127.0.0.1:8080. The process listens on 127.0.0.1 only. Copy `.env.example` to `.env` to change the model, the daily question cap, or the per-address limit. `.env` is gitignored. The API key is never printed.

Ollama, in PowerShell or any other shell. Pull a model that can call tools, then start the page with the default settings:

```bash
ollama pull qwen2.5
uv run statistikbanken-web
```

Ollama's OpenAI-compatible API is `http://127.0.0.1:11434/v1`. The dummy key is ignored.

xAI uses the same chat-completions shape at `https://api.x.ai/v1`. Set the key in the environment, not in the repo:

```powershell
$env:XAI_API_KEY = "your-key"
uv run statistikbanken-web --llm xai --model grok-4
```

`--llm`, `--model`, `--base-url`, `--host`, and `--port` override the environment for one run.

## Tests

Unit tests replay responses recorded from the live API (`tests/fixtures/`). A smaller set calls the API directly, including one session over stdio. GitHub Actions runs Ruff and pytest on every push and pull request.

```bash
uv sync --group dev
uv run pytest -m "not live"   # fixtures only
uv run pytest -m live         # the real API
uv run pytest                 # both
uv run ruff check .
uv run ruff format --check .
```

On 31 October 2026 a real MCP client answered 50 Danish questions through this server. 37 were fine, 5 were near matches (the table was narrower than the question), and none of the numbers were wrong. Five invalid inputs were rejected clearly. Three were not: `Kobenhavn` without ø, an English question, and the name Silkeborg. Search was also too strict, and a missing required variable came back as a raw Danish API error.

Those cases were fixed afterwards: partial search, a small synonym list, label lookup with suggestions, English errors that list the valid codes, a shorter `tabelinfo`, and sums of explicit codes. Unpublished trailing zeros are flagged.

Limits that remain:

- When several tables score the same, catalogue order wins, so the best table is often on the first page rather than always first.
- Place names in search come from FOLK1A's area list only.
- There is no table of all association membership by municipality. The nearest table, `IDRFOR01`, counts sports-club memberships, and its title says so.

## Design

- Three tools, not one. Search, then codes, then data.
- Totals such as ages 80+ and 0–17 use Statistikbanken's own `sum()`.
- Labels map to codes. A near miss suggests the closest code and label.
- `tabelinfo` stays short enough to read: time is a span, long lists are capped.
- No API key. The repository contains no secrets.

## Data and licence

Numbers come from Danmarks Statistik, [Statistikbanken](https://www.statistikbanken.dk), via [the public API](https://www.dst.dk/en/Statistik/hjaelp-til-statistikbanken/api). Danmarks Statistik states that the data may be used free of charge, including commercially, if you cite the source. They describe those terms as [Creative Commons CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). A citation looks like: Danmarks Statistik, Statistikbanken, table FOLK1A.

The code is released under the [MIT licence](LICENSE).
