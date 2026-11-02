"""Public demo hosting: limits, health, proxy, and headers. No network."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from http.client import HTTPConnection, HTTPResponse
from http.server import ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from statistikbanken_mcp.web import (
    PUBLIC_MAX_TOKENS,
    XAI_BASE_URL,
    ChatClient,
    Limiter,
    Settings,
    TurnResult,
    _handler,
    chat_client,
    check_settings,
    client_ip,
    server_command,
    settings_from_env,
)

_ENV_KEYS = (
    "STATBANK_PUBLIC",
    "STATBANK_LLM",
    "STATBANK_BASE_URL",
    "STATBANK_MODEL",
    "STATBANK_XAI_MODEL",
    "STATBANK_STATE_FILE",
    "STATBANK_API_KEY",
    "XAI_API_KEY",
    "STATBANK_HOST",
    "STATBANK_PORT",
    "PORT",
    "STATBANK_DAILY_CAP",
    "STATBANK_RATE_PER_MINUTE",
    "STATBANK_MAX_QUESTION_CHARS",
    "STATBANK_MAX_TOOL_CALLS",
    "STATBANK_PROXY_SECRET",
    "STATBANK_FRAME_ANCESTORS",
    "STATBANK_LLM_THINK",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _settings(**overrides: object) -> Settings:
    settings = Settings(
        llm="xai",
        base_url=XAI_BASE_URL,
        model="grok-4",
        api_key="dummy",
        host="127.0.0.1",
        port=0,
        daily_cap=40,
        per_minute=10,
        max_question_chars=300,
        max_tool_calls=6,
        llm_timeout=5,
        tool_timeout=5,
    )
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


def _ok(question: str) -> TurnResult:
    return TurnResult(answer=f"Svar: {question}", facts=[], calls=[])


@contextmanager
def _running(settings: Settings, answer: object) -> Iterator[int]:
    handler = _handler(settings, Limiter(settings.daily_cap, settings.per_minute), answer)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def _request(
    port: int,
    method: str,
    path: str,
    payload: dict | None = None,
    headers: dict[str, str] | None = None,
) -> HTTPResponse:
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    body = None if payload is None else json.dumps(payload).encode()
    connection.request(method, path, body=body, headers=headers or {})
    return connection.getresponse()


def _get(port: int, path: str = "/", headers: dict[str, str] | None = None) -> HTTPResponse:
    return _request(port, "GET", path, headers=headers)


def _post(
    port: int,
    payload: dict,
    path: str = "/api/ask",
    headers: dict[str, str] | None = None,
) -> HTTPResponse:
    sent = {"Content-Type": "application/json"}
    if headers:
        sent.update(headers)
    return _request(port, "POST", path, payload, sent)


def test_local_defaults_are_unchanged(clean_env: None) -> None:
    settings = settings_from_env([])
    assert settings.llm == "ollama"
    assert settings.host == "127.0.0.1"
    assert settings.port == 8080
    assert settings.daily_cap == 200
    assert settings.per_minute == 30
    assert settings.max_question_chars == 1000
    assert settings.max_tool_calls == 8
    assert settings.public is False
    assert settings.max_tokens is None
    assert settings.max_concurrent is None
    assert settings.proxy_secret == ""
    assert settings.frame_ancestors == "'self'"
    assert settings.state_file == ""
    assert settings.think is False


def _completion_body(settings: Settings) -> dict[str, object]:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"choices": [{"message": {"content": "hej"}}]})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    chat_client(settings, http).complete([{"role": "user", "content": "hej"}], None)
    body = seen["body"]
    assert isinstance(body, dict)
    return body


def test_ollama_thinking_is_off_unless_asked(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _completion_body(settings_from_env([]))
    assert body["model"] == "gemma4:26b-a4b-it-qat"
    assert body["reasoning_effort"] == "none"
    assert "think" not in body

    monkeypatch.setenv("STATBANK_MODEL", "gemma4-16k")
    monkeypatch.setenv("STATBANK_LLM_THINK", "1")
    thinking = _completion_body(settings_from_env([]))
    assert thinking["model"] == "gemma4-16k"
    assert "reasoning_effort" not in thinking
    assert "think" not in thinking

    monkeypatch.setenv("STATBANK_LLM_THINK", "0")
    off = _completion_body(settings_from_env([]))
    assert off["reasoning_effort"] == "none"
    assert "think" not in off


def test_xai_requests_omit_reasoning_effort(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STATBANK_LLM", "xai")
    monkeypatch.setenv("XAI_API_KEY", "dummy")
    monkeypatch.setenv("STATBANK_LLM_THINK", "0")
    body = _completion_body(settings_from_env([]))
    assert "reasoning_effort" not in body
    assert "think" not in body
    assert body["max_tokens"] == PUBLIC_MAX_TOKENS
    assert "dummy" not in json.dumps(body)

    monkeypatch.setenv("STATBANK_LLM_THINK", "1")
    again = _completion_body(settings_from_env([]))
    assert "reasoning_effort" not in again
    assert "think" not in again
    assert "dummy" not in json.dumps(again)

    monkeypatch.setenv("STATBANK_PUBLIC", "1")
    monkeypatch.setenv("STATBANK_LLM_THINK", "0")
    public = _completion_body(settings_from_env(["--llm", "ollama"]))
    assert public["model"] == "grok-4"
    assert "reasoning_effort" not in public
    assert "think" not in public


def test_port_prefers_platform_port_then_statbank_port(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STATBANK_PORT", "9090")
    assert settings_from_env([]).port == 9090
    monkeypatch.setenv("PORT", "10000")
    assert settings_from_env([]).port == 10000
    assert settings_from_env(["--port", "9091"]).port == 9091
    monkeypatch.delenv("PORT")
    monkeypatch.delenv("STATBANK_PORT")
    assert settings_from_env([]).port == 8080


def test_public_mode_forces_xai_and_tighter_defaults(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STATBANK_PUBLIC", "1")
    monkeypatch.setenv("XAI_API_KEY", "dummy")
    monkeypatch.setenv("STATBANK_LLM", "ollama")
    monkeypatch.setenv("STATBANK_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("STATBANK_MODEL", "qwen2.5")
    settings = settings_from_env(["--llm", "ollama", "--base-url", "http://127.0.0.1:11434/v1"])
    assert settings.public is True
    assert settings.llm == "xai"
    assert settings.base_url == XAI_BASE_URL
    assert settings.model == "grok-4"
    assert settings.api_key == "dummy"
    assert settings.daily_cap == 40
    assert settings.per_minute == 3
    assert settings.max_tool_calls == 6
    assert settings.max_question_chars == 300
    assert settings.max_concurrent == 2
    assert settings.max_tokens == PUBLIC_MAX_TOKENS
    assert settings.host == "127.0.0.1"
    check_settings(settings)


def test_public_limits_stay_overridable(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STATBANK_PUBLIC", "1")
    monkeypatch.setenv("XAI_API_KEY", "dummy")
    monkeypatch.setenv("STATBANK_DAILY_CAP", "7")
    monkeypatch.setenv("STATBANK_RATE_PER_MINUTE", "1")
    monkeypatch.setenv("STATBANK_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("STATBANK_MAX_QUESTION_CHARS", "50")
    monkeypatch.setenv("STATBANK_MODEL", "qwen2.5")
    monkeypatch.setenv("STATBANK_XAI_MODEL", "grok-4-fast")
    settings = settings_from_env([])
    assert settings.daily_cap == 7
    assert settings.per_minute == 1
    assert settings.max_tool_calls == 2
    assert settings.max_question_chars == 50
    assert settings.model == "grok-4-fast"
    assert settings.llm == "xai"
    assert settings_from_env(["--model", "grok-3"]).model == "grok-3"
    monkeypatch.setenv("STATBANK_DAILY_CAP", "0")
    assert settings_from_env([]).daily_cap == 0


def test_public_mode_exits_when_the_key_is_missing(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STATBANK_PUBLIC", "1")
    settings = settings_from_env([])
    assert settings.api_key == ""
    with pytest.raises(SystemExit, match="XAI_API_KEY is missing") as caught:
        check_settings(settings)
    assert "Do not write it into a file" in str(caught.value)


def test_healthz_skips_the_model_and_the_proxy_secret() -> None:
    def answer(question: str) -> TurnResult:
        raise AssertionError(question)

    settings = _settings(proxy_secret="test-secret", public=True)
    with _running(settings, answer) as port:
        health = _get(port, "/healthz")
        assert health.status == 200
        assert health.read() == b"ok\n"
        blocked = _get(port, "/")
        assert blocked.status == 403


def test_daily_cap_zero_stops_the_demo() -> None:
    def answer(question: str) -> TurnResult:
        raise AssertionError(question)

    settings = _settings(daily_cap=0, public=True)
    with _running(settings, answer) as port:
        response = _post(port, {"question": "hej"})
        assert response.status == 503
        body = json.loads(response.read().decode())
        assert body["error"] == "Demoen er midlertidigt lukket. Prøv igen senere."
        page = _get(port, "/").read().decode()
        assert "Demoen er midlertidigt lukket." in page


def test_public_page_shows_the_demo_line_and_local_page_does_not() -> None:
    settings = _settings(public=True, daily_cap=40, max_question_chars=300)
    with _running(settings, _ok) as port:
        page = _get(port, "/").read().decode()
    assert "Demo: maks. 40 spørgsmål om dagen i alt. Tal fra Danmarks Statistik." in page
    assert 'maxlength="300"' in page
    assert "__DEMO__" not in page
    local = _settings(public=False, daily_cap=200, max_question_chars=1000)
    with _running(local, _ok) as port:
        quiet = _get(port, "/").read().decode()
    assert "Demo:" not in quiet
    assert "__DEMO__" not in quiet


def test_concurrency_limit_returns_a_busy_error() -> None:
    entered = 0
    both = threading.Event()
    release = threading.Event()
    lock = threading.Lock()

    def answer(question: str) -> TurnResult:
        nonlocal entered
        with lock:
            entered += 1
            if entered >= 2:
                both.set()
        assert release.wait(timeout=5)
        return _ok(question)

    settings = _settings(max_concurrent=2, per_minute=10, daily_cap=40)
    with _running(settings, answer) as port:
        threads = [
            threading.Thread(target=lambda: _post(port, {"question": "a"}).read()),
            threading.Thread(target=lambda: _post(port, {"question": "b"}).read()),
        ]
        for thread in threads:
            thread.start()
        assert both.wait(timeout=5)
        busy = _post(port, {"question": "c"})
        assert busy.status == 503
        assert json.loads(busy.read().decode())["error"] == "Der er travlt lige nu. Prøv igen."
        assert entered == 2
        release.set()
        for thread in threads:
            thread.join(timeout=5)


def test_spoofed_forwarded_for_without_a_secret_is_ignored() -> None:
    settings = _settings(public=True, per_minute=1, daily_cap=10)
    with _running(settings, _ok) as port:
        first = _post(port, {"question": "a"}, headers={"X-Forwarded-For": "8.8.8.8"})
        assert first.status == 200
        first.read()
        blocked = _post(port, {"question": "b"}, headers={"X-Forwarded-For": "8.8.8.8"})
        assert blocked.status == 429
        blocked.read()
        other = _post(port, {"question": "c"}, headers={"X-Forwarded-For": "1.1.1.1"})
        assert other.status == 429


def test_valid_proxy_secret_uses_the_forwarded_address() -> None:
    settings = _settings(proxy_secret="test-secret", per_minute=1, daily_cap=10)
    secret = {"X-Proxy-Secret": "test-secret"}
    with _running(settings, _ok) as port:
        first = _post(
            port,
            {"question": "a"},
            headers={**secret, "X-Forwarded-For": "203.0.113.10"},
        )
        assert first.status == 200
        first.read()
        other = _post(
            port,
            {"question": "b"},
            headers={**secret, "X-Forwarded-For": "203.0.113.11"},
        )
        assert other.status == 200
        other.read()
        again = _post(
            port,
            {"question": "c"},
            headers={**secret, "X-Forwarded-For": "203.0.113.10"},
        )
        assert again.status == 429


def test_last_forwarded_for_entry_wins() -> None:
    chain = {"X-Forwarded-For": "8.8.8.8, 203.0.113.50"}
    assert client_ip("127.0.0.1", chain, trust_proxy=True) == "203.0.113.50"
    both = {
        "CF-Connecting-IP": " 203.0.113.5 ",
        "X-Forwarded-For": "8.8.8.8, 198.51.100.1",
    }
    assert client_ip("127.0.0.1", both, trust_proxy=True) == "203.0.113.5"
    assert client_ip("127.0.0.1", chain, trust_proxy=False) == "127.0.0.1"

    settings = _settings(proxy_secret="test-secret", per_minute=1, daily_cap=10)
    secret = {"X-Proxy-Secret": "test-secret"}
    with _running(settings, _ok) as port:
        first = _post(
            port,
            {"question": "a"},
            headers={**secret, "X-Forwarded-For": "8.8.8.8, 203.0.113.50"},
        )
        assert first.status == 200
        first.read()
        same_hop = _post(
            port,
            {"question": "b"},
            headers={**secret, "X-Forwarded-For": "1.1.1.1, 203.0.113.50"},
        )
        assert same_hop.status == 429
        same_hop.read()
        other_hop = _post(
            port,
            {"question": "c"},
            headers={**secret, "CF-Connecting-IP": "203.0.113.70", "X-Forwarded-For": "8.8.8.8"},
        )
        assert other_hop.status == 200


def test_requests_do_not_log_the_visitor_address(capsys: pytest.CaptureFixture[str]) -> None:
    settings = _settings(per_minute=5, daily_cap=10)
    with _running(settings, _ok) as port:
        response = _post(port, {"question": "a"}, headers={"CF-Connecting-IP": "203.0.113.77"})
        assert response.status == 200
        response.read()
    logged = capsys.readouterr().err
    assert "POST /api/ask" in logged
    assert "203.0.113.77" not in logged
    assert "127.0.0.1" not in logged


def test_proxy_secret_rejects_other_routes() -> None:
    settings = _settings(proxy_secret="test-secret", per_minute=10, daily_cap=10)
    with _running(settings, _ok) as port:
        open_health = _get(port, "/healthz")
        assert open_health.status == 200
        open_health.read()
        missing = _get(port, "/")
        assert missing.status == 403
        missing.read()
        wrong = _get(port, "/", headers={"X-Proxy-Secret": "test-secre"})
        assert wrong.status == 403
        wrong.read()
        longer = _post(port, {"question": "hej"}, headers={"X-Proxy-Secret": "test-secret-extra"})
        assert longer.status == 403
        longer.read()
        page = _get(port, "/", headers={"X-Proxy-Secret": "test-secret"})
        assert page.status == 200
        assert "Statistikbanken" in page.read().decode()
        allowed = _post(port, {"question": "hej"}, headers={"X-Proxy-Secret": "test-secret"})
        assert allowed.status == 200
        assert "Svar: hej" in allowed.read().decode()


def test_child_env_has_no_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "super-secret-value")
    monkeypatch.setenv("STATBANK_PROXY_SECRET", "proxy-secret-value")
    monkeypatch.setenv("STATBANK_API_KEY", "other-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    params = server_command(12)
    assert params.env is not None
    for name in ("XAI_API_KEY", "STATBANK_PROXY_SECRET", "STATBANK_API_KEY", "OPENAI_API_KEY"):
        assert name not in params.env
    blob = "\n".join(params.env.values())
    assert "super-secret-value" not in blob
    assert "proxy-secret-value" not in blob
    assert "openai-secret" not in blob
    assert params.env["STATBANK_HTTP_TIMEOUT"] == "12"
    assert "PATH" in params.env


def test_security_headers_and_no_cookie() -> None:
    custom = _settings(frame_ancestors="https://example.com")
    with _running(custom, _ok) as port:
        response = _get(port, "/")
        assert response.status == 200
        assert (
            response.getheader("Content-Security-Policy") == "frame-ancestors https://example.com"
        )
        assert response.getheader("X-Content-Type-Options") == "nosniff"
        assert response.getheader("Referrer-Policy") == "no-referrer"
        assert response.getheader("Set-Cookie") is None
        response.read()
        health = _get(port, "/healthz")
        assert health.getheader("X-Content-Type-Options") == "nosniff"
        health.read()
    default = _settings()
    with _running(default, _ok) as port:
        page = _get(port, "/")
        assert page.getheader("Content-Security-Policy") == "frame-ancestors 'self'"
        page.read()


def test_max_tokens_is_sent_only_when_set() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"choices": [{"message": {"content": "hej"}}]})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    capped = ChatClient(XAI_BASE_URL, "dummy", "grok-4", 5, http=http, max_tokens=PUBLIC_MAX_TOKENS)
    capped.complete([{"role": "user", "content": "hej"}], None)
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["max_tokens"] == PUBLIC_MAX_TOKENS
    assert "dummy" not in json.dumps(body)

    plain = ChatClient(XAI_BASE_URL, "dummy", "grok-4", 5, http=http)
    plain.complete([{"role": "user", "content": "hej"}], None)
    second = seen["body"]
    assert isinstance(second, dict)
    assert "max_tokens" not in second

    capped.complete(
        [{"role": "user", "content": "hej"}],
        [{"type": "function", "function": {"name": "find_tabel", "parameters": {}}}],
    )
    with_tools = seen["body"]
    assert isinstance(with_tools, dict)
    assert with_tools["max_tokens"] == PUBLIC_MAX_TOKENS
    assert with_tools["tools"][0]["function"]["name"] == "find_tabel"


def test_public_xai_request_uses_statbank_xai_model(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STATBANK_PUBLIC", "1")
    monkeypatch.setenv("XAI_API_KEY", "dummy")
    monkeypatch.setenv("STATBANK_XAI_MODEL", "grok-4-fast")
    monkeypatch.setenv("STATBANK_MODEL", "qwen2.5")
    settings = settings_from_env([])
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"choices": [{"message": {"content": "hej"}}]})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = ChatClient(
        settings.base_url,
        settings.api_key,
        settings.model,
        settings.llm_timeout,
        http=http,
        max_tokens=settings.max_tokens,
    )
    client.complete([{"role": "user", "content": "hej"}], [{"type": "function"}])
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["model"] == "grok-4-fast"
    assert body["max_tokens"] == PUBLIC_MAX_TOKENS
    assert "dummy" not in json.dumps(body)


def test_local_xai_prefers_statbank_xai_model(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STATBANK_LLM", "xai")
    monkeypatch.setenv("XAI_API_KEY", "dummy")
    monkeypatch.setenv("STATBANK_MODEL", "grok-2")
    settings = settings_from_env([])
    assert settings.model == "grok-2"
    assert settings.max_tokens == PUBLIC_MAX_TOKENS
    monkeypatch.setenv("STATBANK_XAI_MODEL", "grok-3")
    chosen = settings_from_env([])
    assert chosen.model == "grok-3"
    assert chosen.public is False


def test_state_file_keeps_the_daily_count_and_not_the_address(tmp_path: Path) -> None:
    path = tmp_path / "daily.json"
    address = "203.0.113.44"
    limiter = Limiter(2, 10, str(path))
    assert limiter.allow(address) is None
    assert limiter.allow("198.51.100.8") is None
    text = path.read_text(encoding="utf-8")
    assert address not in text
    assert "198.51.100.8" not in text
    assert json.loads(text) == {"day": date.today().isoformat(), "count": 2}
    restored = Limiter(2, 10, str(path))
    assert restored.allow("203.0.113.1") == "daily"


def test_state_file_from_another_day_does_not_count(tmp_path: Path) -> None:
    path = tmp_path / "daily.json"
    path.write_text(json.dumps({"day": "2000-01-01", "count": 40}), encoding="utf-8")
    limiter = Limiter(40, 10, str(path))
    assert limiter.allow("203.0.113.9") is None


def test_state_file_is_optional(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    assert settings_from_env([]).state_file == ""
    monkeypatch.setenv("STATBANK_STATE_FILE", "/var/data/statbank-daily.json")
    assert settings_from_env([]).state_file == "/var/data/statbank-daily.json"
