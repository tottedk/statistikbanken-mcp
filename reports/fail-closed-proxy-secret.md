# Fail-closed public startup check

Local verification only. Nothing was pushed, no pull request was opened, and nothing was deployed.

## Result

When `STATBANK_PUBLIC` is one of the existing truthy values (`1`, `true`, `yes`, `on`), `check_settings` (called from `serve` before the process listens) now exits if `STATBANK_PROXY_SECRET` is unset, empty, or whitespace. The message names `STATBANK_PROXY_SECRET` and does not include a value. A non-empty placeholder (`test-proxy-secret`) does not trigger that refusal. Local mode does not require the variable.

The same startup path already exited when `XAI_API_KEY` was missing. That check still runs first, so a public start with neither value still exits with the existing `XAI_API_KEY is missing` message.

## Failing run (before the check)

Command:

```bash
set -o pipefail
uv run pytest tests/test_public.py -k "proxy_secret_is_missing_or_blank or does_not_exit_for_a_proxy_secret or local_mode_does_not_require_a_proxy_secret" --tb=short -q
```

Exit code: 1

Collected on this filter: 19. Result: 12 failed, 7 passed, 24 deselected.

The 12 failures are public mode with a missing, empty, or whitespace-only `STATBANK_PROXY_SECRET`. Each failed with `Failed: DID NOT RAISE SystemExit` at `check_settings`. The 7 passes were the non-empty placeholder and local mode (`None`, empty, `0`, `false`, `no`, `off`), which current code already allowed.

Failed node ids:

- `test_public_mode_exits_when_the_proxy_secret_is_missing_or_blank[missing-1]`
- `test_public_mode_exits_when_the_proxy_secret_is_missing_or_blank[missing-true]`
- `test_public_mode_exits_when_the_proxy_secret_is_missing_or_blank[missing-yes]`
- `test_public_mode_exits_when_the_proxy_secret_is_missing_or_blank[missing-on]`
- `test_public_mode_exits_when_the_proxy_secret_is_missing_or_blank[empty-1]`
- `test_public_mode_exits_when_the_proxy_secret_is_missing_or_blank[empty-true]`
- `test_public_mode_exits_when_the_proxy_secret_is_missing_or_blank[empty-yes]`
- `test_public_mode_exits_when_the_proxy_secret_is_missing_or_blank[empty-on]`
- `test_public_mode_exits_when_the_proxy_secret_is_missing_or_blank[whitespace-1]`
- `test_public_mode_exits_when_the_proxy_secret_is_missing_or_blank[whitespace-true]`
- `test_public_mode_exits_when_the_proxy_secret_is_missing_or_blank[whitespace-yes]`
- `test_public_mode_exits_when_the_proxy_secret_is_missing_or_blank[whitespace-on]`

## Passing run (on the commit below)

Command:

```bash
set -o pipefail
uv run pytest -m "not live" -q
```

Exit code: 0

219 passed, 21 deselected in 11.85s.

Working tree was clean. `HEAD` was `1a2ad2733744263526cb43f81f9c304c497e84da`.

Before this change the same command collected 200 non-live tests (221 including the 21 live tests). It now collects 219 non-live tests (240 including live). The count went up by 19. No test was deleted.

`uv run ruff check .` and `uv run ruff format --check .` both exited 0.

## Commit

Branch: `cursor/fail-closed-proxy-secret-51e9`

Commit that contains the check and the tests, and on which the passing command was run:

`1a2ad2733744263526cb43f81f9c304c497e84da`

This report file is committed after that. It does not change the check or the tests.

## Test edit

`test_public_mode_forces_xai_and_tighter_defaults` now sets `STATBANK_PROXY_SECRET` to the placeholder `test-proxy-secret` before it calls `check_settings`. That test already required a public configuration with an xAI key to pass the startup check. Its assertions about xAI and the tighter limits are unchanged. Without the placeholder, the new rule would reject that fixture.

`test_public_mode_exits_when_the_key_is_missing` was not changed.
