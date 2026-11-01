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
| `tabelinfo` | Read codes | `table`, optional `variable`, `values_filter` | Time is first, last, and count. Long value lists are capped; `values_filter` looks up the rest, and several words match any of them. |
| `hent_data` | Fetch rows | `table`, `values`, `sum_from`, `sum_range` | Values may be codes or labels (`Rudersdal`, `Kobenhavn`). A required variable with one total is filled in. One with no total returns its codes. `sum_from` and `sum_range` use the API's own sum. `>=80` on its own is one row per code. A time sum that would hide rows, including two periods passed where a range was meant, is an error. A trailing 0 marked `unpublished` is a period the API has not published yet. |

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

MCP Inspector. `mcp dev` starts the inspector with `npx`, so Node.js is required:

```bash
uv run mcp dev src/statistikbanken_mcp/server.py
```

That starts a local page where you can list the tools and call them.

## Web page

`statistikbanken-web` is a local page. It asks an OpenAI-compatible chat-completions client, Ollama or xAI, to call the same three tools over stdio.

```bash
uv run statistikbanken-web
```

Then open http://127.0.0.1:8080. The default bind is `127.0.0.1` port `8080`. `--host`, `--port`, `STATBANK_HOST`, and `STATBANK_PORT` change it. `PORT`, when set, is used instead of `STATBANK_PORT`. Copy `.env.example` to `.env` to change the model, the daily question cap, or the per-address limit. `.env` is gitignored. The API key is not printed.

Ollama is the default (`--llm ollama`). The model is `qwen2.5` and the base URL is `http://127.0.0.1:11434/v1`. Pull a model that can call tools, then start the page:

```bash
ollama pull qwen2.5
uv run statistikbanken-web
```

xAI uses `https://api.x.ai/v1`. The model is `STATBANK_XAI_MODEL`, or `grok-4` when that is unset. `grok-4` may be retired, so set the model when you deploy. Each xAI request sends `max_tokens` (1024). Set `XAI_API_KEY` in the environment or in `.env`, not in the repo:

```bash
uv run statistikbanken-web --llm xai --model grok-4
```

`--llm` is `ollama` or `xai`. `--model`, `--base-url`, `--host`, and `--port` override the environment for one run. The model and Statistikbanken time out separately (45 seconds and 30 seconds by default).

The page is in Danish. Ask in Danish or English. The model is told to answer in the language of the question.

The page opens in Basis. Basis shows the answer and a compact table: one row per period, one column per series. It hides the chart, the downloads, the source link, and the tool calls. When the numbers are shown, Avanceret adds the chart (a series needs more than one period), the source link (the table id and title on `https://www.statistikbanken.dk/`), and, under the table, CSV, Excel, and JSON. A PNG of the chart is included when a chart is shown. Excel and JSON are built from the rows already on the page. They do not ask the model or Statistikbanken again. Avanceret also lists the tool calls, including when the numbers are hidden. A failed request shows only the error. The browser remembers the choice. Switching redraws the last answer and does not ask again.

`2015K1` reads as `2015 1. kvt.` on the table, the chart, the page's own answer line, and in CSV and Excel. A month is `marts 2015`, a half-year is `2015 1. halvår`, and a year stays the year. Unknown codes are left as they are. JSON keeps the code in `periode` and adds `periode_label`. CSV is UTF-8 with a BOM, a semicolon between columns, and a decimal comma.

A number in the answer must come from the question, the fetched data, a period or row label from those rows, the fetched table title, or a variable value in that turn's `hent_data` arguments. A number that appears only in `tabelinfo` or `find_tabel` does not count. Digits that only name a fetched quarter, month, or half-year are not extra figures. Otherwise the prose is hidden. A question that does not ask how the figures change over time is answered with the page's own line for the latest period. If the question names a place or an age and the fetched rows are for a different one, they are not presented as the answer.

## Running as a public demo

```bash
docker build -t statistikbanken-web .
docker run --rm -p 8080:8080 \
  -e STATBANK_PUBLIC=1 \
  -e STATBANK_XAI_MODEL=grok-4 \
  -e XAI_API_KEY \
  statistikbanken-web
```

The image is Python 3.12. It listens on `STATBANK_HOST` (the image sets `0.0.0.0`) and on `PORT` when that is set, otherwise `STATBANK_PORT` (8080). `GET /healthz` returns 200 and does not call the model or Statistikbanken.

`STATBANK_PUBLIC=1` uses xAI only. Ollama is ignored, and so are `--llm`, `STATBANK_BASE_URL`, and `STATBANK_MODEL`. The process exits at startup when `XAI_API_KEY` is missing. Pass the key in from the environment (`-e XAI_API_KEY` with no value copies it from the shell). Do not write `XAI_API_KEY` or `STATBANK_PROXY_SECRET` into a file, the image, or the repo.

Set `STATBANK_XAI_MODEL` when you deploy. The default is `grok-4`, and that name may be retired. `--model` overrides it for one run. Every request to xAI includes `max_tokens` (1024).

Public defaults, each still overridable by its `STATBANK_*` variable: 40 questions a day in total (`STATBANK_DAILY_CAP`), 3 per minute per visitor (`STATBANK_RATE_PER_MINUTE`), 6 tool calls (`STATBANK_MAX_TOOL_CALLS`), and 300 characters (`STATBANK_MAX_QUESTION_CHARS`). At most two questions run at once. The page shows a short Danish line with the daily cap.

The daily count and the per-minute window are kept in memory. A free container that sleeps or restarts starts them again at zero. That is not a spending cap. The hard cap is the prepaid xAI credit. Turn auto top-up off.

`STATBANK_STATE_FILE`, if set to a path, stores the date and the daily count so a restart on the same day can continue the count. The file does not store addresses. It only helps when the disk outlives the process. An ephemeral disk is wiped when the container sleeps or restarts, and the count is forgotten there too.

Emergency stop: set `STATBANK_DAILY_CAP=0` and restart. Questions then get a closed message until the cap is raised.

The per-minute limit uses `CF-Connecting-IP`, or else the first `X-Forwarded-For` entry, only when `STATBANK_TRUST_PROXY` is on (the default in public mode). Addresses are not logged. When `STATBANK_PROXY_SECRET` is set, every route except `/healthz` requires the header `X-Proxy-Secret`. `STATBANK_FRAME_ANCESTORS` is the CSP `frame-ancestors` source list. The default is `'self'`.

## Tests

Unit tests replay responses recorded from the live API (`tests/fixtures/`). A smaller set calls the API directly, including one session over stdio.

```bash
uv sync --group dev
uv run pytest -m "not live"   # fixtures only
uv run pytest -m live         # the real API
uv run pytest                 # both
uv run ruff check .
uv run ruff format --check .
```

GitHub Actions runs those Ruff commands and `uv run pytest` with no marker, so the live tests run there, on every push and pull request.

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
