# statistikbanken-mcp

An MCP server for [Danmarks Statistik](https://www.dst.dk)'s public Statistikbanken API. No API key.

Statistikbanken has more than 2,000 tables and no free-text search. This server gives an assistant three small tools, so a question such as "How many people over 80 live in Esbjerg?" comes back as a number from a named table and period, with Danmarks Statistik as the source.

## Example

**How many people over 80 live in Esbjerg?**

`find_tabel("befolkning")` returns `FOLK1A`, "Befolkningen den 1. i kvartalet", latest period `2026K3`.

```text
hent_data(
  table="FOLK1A",
  values={"OMRÅDE": ["Esbjerg"], "KØN": ["TOT"], "CIVILSTAND": ["TOT"], "Tid": ["(1)"]},
  sum_from={"ALDER": "80"}
)
```

Ages in FOLK1A are single years. `sum_from` asks the API to add 80 through 125. The result is one row:

| Table | Period | Place | Age | Value |
|---|---|---|---|---|
| FOLK1A | 2026K3 | Esbjerg | 80+ | 7,556 |

Unit is people (`Antal`). Source: Danmarks Statistik.

**Children aged 0–17 in Esbjerg and Viborg**, same table and quarter, with `sum_range={"ALDER": ["0", "17"]}`:

| Place | Value |
|---|---|
| Esbjerg | 21,616 |
| Viborg | 19,171 |

**A misspelled name.** `hent_data` with `OMRÅDE: ["Kobenhavn"]` resolves to København. Total population in FOLK1A, 2026K3: 670,389.

These three calls were run against `https://api.statbank.dk` on 1 November 2026.

## Tools

Names are Danish. Arguments and descriptions are English.

| Tool | Purpose | Main inputs | Behaviour worth knowing |
|---|---|---|---|
| `find_tabel` | Find a table | `query`, `limit` (1–25, default 10) | Matches id, title, and variable names. Some words are enough. A long word drops a trailing e, and a compound splits on s (`gennemsnitsindkomst`). A hyphenated word also matches the joined form and the parts written separately (`fuldtids-ledige`). A few synonyms are built in (`population`, `bruttoledighed`, `huspriser`, `husdyr`, `indbrud`). `arbejdsløshed` and `ledighed` search the Fuldtidsledige tables, so the AUS tables come first. A query that says "før skat" prefers a title of "Indkomst i alt" over "Disponibel indkomst". Municipality names are matched against FOLK1A and returned with their codes. A short list ranks the usual table first: population by municipality or age is FOLK1A, population by origin is FOLK1E, registered unemployment is AUS08, personal income is INDKP105, and crimes reported are STRAF10. A table id in the query stays first. A hit whose title is a fremskrivning has `projection` true. A miss is an error, not an empty list. |
| `tabelinfo` | Read codes | `table`, optional `variable`, `values_filter` | Time is first, last, and count. Long value lists are capped; `values_filter` looks up the rest, and several words match any of them. |
| `hent_data` | Fetch rows | `table`, `values`, `sum_from`, `sum_range` | Values may be codes or labels (`Esbjerg`, `Kobenhavn`). A variable left out is set to a top-level total (`I alt`, `Total`, `Hele landet`, or code TOT or IALT) and listed in `defaults`. A longer label that only ends in "i alt" is used when the API allows elimination. On a variable that must be set, that label is a subcategory and the error lists the values. Time left out is the latest published period. `sum_from` and `sum_range` use the API's own sum. `>=80` on its own is one row per code. A time sum that would hide rows, including two periods passed where a range was meant, is an error. A bare year on a quarterly or monthly table becomes one yearly figure when the year is complete: flows are added, levels are averaged, and the note says which. If the measure is unclear, or a quarter or month is missing, the periods stay separate and no yearly figure is invented. `i år`, `i fjor`, `sidste år`, and `sidste måned` follow today's calendar. If that period is not published, the error names it and the latest period in the table. `seneste` is the latest published period. `sidste kvartal`, `marts 2025`, and `K3 2025` are unchanged. A trailing 0 marked `unpublished` is a period the API has not published yet. A fremskrivning table, and a row labelled prognose or fremskrivning, is marked `projection`. |

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

Ollama is the default (`--llm ollama`). The model is `gemma4:26b-a4b-it-qat` and the base URL is `http://127.0.0.1:11434/v1`. Pull it, then start the page:

```bash
ollama pull gemma4:26b-a4b-it-qat
uv run statistikbanken-web
```

That tag's context is shorter than a tool-calling turn. Create a 16k tag from a Modelfile and select it with `STATBANK_MODEL` or `--model`:

```text
FROM gemma4:26b-a4b-it-qat
PARAMETER num_ctx 16384
```

```bash
ollama create gemma4-16k -f Modelfile
STATBANK_MODEL=gemma4-16k uv run statistikbanken-web
```

`qwen2.5` is an alternative. `ollama pull qwen2.5`, then `STATBANK_MODEL=qwen2.5`.

Ollama thinking is off. The request sends `reasoning_effort` set to `none`. `STATBANK_LLM_THINK=1` leaves thinking on. xAI ignores that variable. The system prompt states today's date in Europe/Copenhagen and builds the unpublished-period example from that date. `i fjor` and `sidste måned` are Statbank codes counted from the date. `seneste 10 år` is the latest published period and the same quarter or month 10 years earlier.

xAI uses `https://api.x.ai/v1`. The model is `STATBANK_XAI_MODEL`, or `grok-4` when that is unset. `grok-4` may be retired, so set the model when you deploy. Each xAI request sends `max_tokens` (2048). Set `XAI_API_KEY` in the environment or in `.env`, not in the repo:

```bash
uv run statistikbanken-web --llm xai --model grok-4
```

`--llm` is `ollama` or `xai`. `--model`, `--base-url`, `--host`, and `--port` override the environment for one run. The model and Statistikbanken time out separately (45 seconds and 30 seconds by default).

The page is in Danish. Ask in Danish or English. The model is told to answer in the language of the question.

The page opens in Basis. Basis shows the answer and a compact table: one row per period, one column per series. It hides the chart, the downloads, the source link, and the tool calls. When the numbers are shown, Avanceret adds the chart (a series needs more than one period), the source link (the table id and title on `https://www.statistikbanken.dk/`), and, under the table, CSV, Excel, and JSON. A PNG of the chart is included when a chart is shown. Excel and JSON are built from the rows already on the page. They do not ask the model or Statistikbanken again. Avanceret also lists the tool calls, including when the numbers are hidden. A failed request shows only the error. A search, table, or value that is not found shows its suggestion (`Mente du …`, or the server's hint) in both Basis and Avanceret. The browser remembers the choice. Switching redraws the last answer and does not ask again.

`2015K1` reads as `2015 1. kvt.` on the table, the chart, the page's own answer line, and in CSV and Excel. A month is `marts 2015`, a half-year is `2015 1. halvår`, and a year stays the year. Unknown codes are left as they are. JSON keeps the code in `periode` and adds `periode_label`. CSV is UTF-8 with a BOM, a semicolon between columns, and a decimal comma.

A number in the answer must come from the question, the fetched data, a period or row label from those rows, the fetched table title, or a variable value in that turn's `hent_data` arguments. A difference, a sum, or a percentage change of those fetched values is also kept, including ordinary rounding (`1.938` when the values differ by 1,938, or `2 procent` for a change of about 2). A number that is not that arithmetic is still hidden. A number that appears only in `tabelinfo` or `find_tabel` does not count. Digits that only name a fetched quarter, month, or half-year are not extra figures. A fetched value tied to the wrong period or the wrong row is still hidden. Otherwise the prose is hidden. A question that does not ask how the figures change over time is answered with the page's own line for the latest period. That line names a projection, and it states when a year was added or averaged. If the model says the requested period is not published yet, that note stays with the line. If the question names a place or an age and the fetched rows are for a different one, they are not presented as the answer.

## Running as a public demo

```bash
docker build -t statistikbanken-web .
docker run --rm -p 8080:8080 \
  -e STATBANK_PUBLIC=1 \
  -e STATBANK_XAI_MODEL=grok-4 \
  -e XAI_API_KEY \
  statistikbanken-web
```

The image is Python 3.12. It listens on `STATBANK_HOST` (the image sets `0.0.0.0`) and on `PORT` when that is set, otherwise `STATBANK_PORT` (8080). `GET /healthz` returns 200 and does not call the model or Statistikbanken. The image health check requests that path.

`STATBANK_PUBLIC=1` uses xAI only. Ollama is ignored, and so are `--llm`, `STATBANK_BASE_URL`, and `STATBANK_MODEL`. The process exits at startup when `XAI_API_KEY` is missing. Pass the key in from the environment (`-e XAI_API_KEY` with no value copies it from the shell). Do not write `XAI_API_KEY` or `STATBANK_PROXY_SECRET` into a file, the image, or the repo.

Set `STATBANK_XAI_MODEL` when you deploy. The default is `grok-4`, and that name may be retired. `--model` overrides it for one run. Every request to xAI includes `max_tokens` (2048).

Public defaults, each still overridable by its `STATBANK_*` variable: 40 questions a day in total (`STATBANK_DAILY_CAP`), 3 per minute per visitor (`STATBANK_RATE_PER_MINUTE`), 6 tool calls (`STATBANK_MAX_TOOL_CALLS`), and 300 characters (`STATBANK_MAX_QUESTION_CHARS`). At most two questions run at once. The page shows a short Danish line with the daily cap.

The daily count and the per-minute window are kept in memory. A free container that sleeps or restarts starts them again at zero. That is not a spending cap. The hard cap is the prepaid xAI credit. Turn auto top-up off.

`STATBANK_STATE_FILE`, if set to a path, stores the date and the daily count so a restart on the same day can continue the count. The file does not store addresses. It only helps when the disk outlives the process. An ephemeral disk is wiped when the container sleeps or restarts, and the count is forgotten there too.

Emergency stop: set `STATBANK_DAILY_CAP=0` and restart. Questions then get a closed message until the cap is raised.

The per-minute limit uses the socket address. It reads `CF-Connecting-IP`, or else the last `X-Forwarded-For` entry, only when `STATBANK_PROXY_SECRET` is set and the request carried that value in `X-Proxy-Secret`. A fake `X-Forwarded-For` on a direct request does not change the limit. Addresses are not logged. When the secret is set, every route except `/healthz` requires the header. `STATBANK_FRAME_ANCESTORS` is the CSP `frame-ancestors` source list. The default is `'self'`.

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

Six questions, checked against the live API on 1 November 2026. The same figures are asserted from the recorded responses in `tests/test_acceptance.py` (`ACCEPTANCE_QUESTIONS`). Viborg's club counts are in that list; the recording used for the assert is Esbjerg only.

1. Hvor mange over 80 bor i Esbjerg? FOLK1A 2026K3: 7556.
2. Hvordan har befolkningstallet i Viborg udviklet sig de seneste 10 år? FOLK1A: 96166 in 2016K3, 98104 in 2026K3.
3. Hvor mange over 80 bor i Gentofte, Esbjerg og Viborg? FOLK1A 2026K3: 5486 / 7556 / 6420.
4. Hvor mange foreningsmedlemmer er der i Esbjerg? IDRFOR01 2025: 212 clubs and 54205 memberships. Viborg: 208 and 49821.
5. Hvor mange børn mellem 0 og 17 år bor i Esbjerg og Viborg? FOLK1A 2026K3: 21616 / 19171.
6. Hvad er den gennemsnitlige indkomst i alt før skat i Esbjerg og Viborg? INDKP105 2025: 407775 / 402605.

On 31 October 2026 a real MCP client answered 50 Danish questions through this server. 37 were fine, 5 were near matches (the table was narrower than the question), and none of the numbers were wrong. Five invalid inputs were rejected clearly. Three were not: `Kobenhavn` without ø, an English question, and the name Silkeborg. Search was also too strict, and a missing required variable came back as a raw Danish API error.

Those cases were fixed afterwards: partial search, a small synonym list, label lookup with suggestions, English errors that list the valid codes, a shorter `tabelinfo`, and sums of explicit codes. Unpublished trailing zeros are flagged.

Limits that remain:

- When several tables score the same, catalogue order wins, so the best table is often on the first page rather than always first.
- Place names in search come from FOLK1A's area list only.
- There is no table of all association membership by municipality. The nearest table, `IDRFOR01`, counts sports-club memberships, and its title says so.

## Evaluation

`evals/questions.json` is the regression set: 100 questions from 1 November 2026, plus five owner questions (101–105) marked `source: owner` and `category: hard/ambiguous`. Each item names the Statistikbanken selection that defines the right figure, or the behaviour expected when that figure is not published. A short `note` records a judgment, such as a lower bound or a series the table does not split.

The runner uses the same path as the web page: `answer_with_server` with `--llm xai`. It does not open a port. It is not part of `uv run pytest` or CI. A full run calls xAI and costs money.

```bash
uv run python -m evals.run
uv run python -m evals.run --limit 5
uv run python -m evals.run --ids 63,101-105
uv run python -m evals.run --model grok-4 --out evals/reports/custom.md
```

Set `XAI_API_KEY` in the environment, not in the repo. The key is not printed and not written to the report. The default model is `STATBANK_XAI_MODEL`, or `grok-4`. The report is `evals/reports/<date>-<model>.md` plus a `.jsonl` file next to it. That directory is gitignored. The M2 proof is `evals/reports/2026-11-05-grok-4.md`: 100/105, grok-4, $1.35.

Prices default to the grok-4.3 rates on <https://docs.x.ai/docs/models>: input $1.25, cached input $0.20, and output $2.50 per million tokens. `grok-4` is served as grok-4.3, and reasoning tokens are billed as output. `XAI_PRICE_INPUT_PER_M`, `XAI_PRICE_CACHED_PER_M`, and `XAI_PRICE_OUTPUT_PER_M` override the defaults.

To add a question after a real failure, append an object to `evals/questions.json`. Give it the next `id`, the Danish `question`, a `category`, and `expect`. A figure question lists `checks`: `table`, `variables` (a code list, or `{"from", "to"}` for a numeric range), `time` (`latest`, a list of codes, or `{"year", "grain"}`), and `aggregate` (`values`, `sum`, `mean`, or `sum_by_period`). Several checks mean any one of them is an acceptable official figure. A gap question sets `kind` to `no_table`, `not_published`, `too_broad`, `not_available`, `answer_or_clarify`, `partial`, or `approximate`. `tests/test_eval_score.py` checks the scorer with fixtures and does not call the API.

## Design

- Three tools, not one. Search, then codes, then data.
- Totals such as ages 80+ and 0–17 use Statistikbanken's own `sum()`.
- Labels map to codes. A near miss suggests the closest code and label.
- `tabelinfo` stays short enough to read: time is a span, long lists are capped.
- No API key. The repository contains no secrets.

## Data and licence

Numbers come from Danmarks Statistik, [Statistikbanken](https://www.statistikbanken.dk), via [the public API](https://www.dst.dk/en/Statistik/hjaelp-til-statistikbanken/api). Danmarks Statistik states that the data may be used free of charge, including commercially, if you cite the source. They describe those terms as [Creative Commons CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). A citation looks like: Danmarks Statistik, Statistikbanken, table FOLK1A.

The code is released under the [MIT licence](LICENSE).
