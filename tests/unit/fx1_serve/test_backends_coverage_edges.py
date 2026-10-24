"""Line and branch edges for fx1.serve.backends left open by the fx1 run."""

from __future__ import annotations

import contextlib
import email.message
import http.client
import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from fx1.serve import backends as backends
from fx1.serve.backends import (
    BackendNotConfiguredError,
    HostedK3Backend,
    LocalFx1Backend,
    OpenAICompatBackend,
    _embedding_item_shape,
    _env_float,
    _extract_token_count,
    _free_tcp_port,
    _open_byok_pinned,
    _openai_chat_complete,
    _openai_chat_complete_tools,
    _openai_chat_stream,
    _openai_embeddings_complete,
    _openai_sibling_url,
    _openai_tokenize_count,
    _RefuseRedirects,
    _tool_call_shape,
    byok_base_url_problem,
    hosted_api_url_problem,
)

_BASE = Path(__file__).resolve().parents[3] / "artifacts" / "fx1_tiny_lm"
_MSGS = [{"role": "user", "content": "hi"}]
_URL = "https://example.com/v1/chat/completions"


class _Body:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw

    def read(self) -> bytes:
        return self.raw


class _Lines:
    def __init__(self, lines: list[bytes]) -> None:
        self.lines = lines

    def __iter__(self) -> Any:
        return iter(self.lines)


class _Encode:
    def read(self) -> bytes:
        raise UnicodeEncodeError("utf-8", "x", 0, 1, "no")


def _http_error(code: int = 500) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(_URL, code, "no", email.message.Message(), None)


def _patch_open(monkeypatch: pytest.MonkeyPatch, outcome: Any) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    @contextlib.contextmanager
    def opener(request: urllib.request.Request, *, timeout_s: float) -> Any:
        seen["timeout"] = timeout_s
        seen["body"] = json.loads(request.data.decode()) if request.data else None
        seen["url"] = request.full_url
        if isinstance(outcome, BaseException):
            raise outcome
        yield outcome

    monkeypatch.setattr(backends, "_openai_urlopen", opener)
    return seen


def test_url_helpers_and_redirect_refusal() -> None:
    assert _openai_sibling_url("https://example.com/v1/embeddings", "embeddings").endswith(
        "/embeddings"
    )
    assert _openai_sibling_url(_URL, "embeddings").endswith("/v1/embeddings")
    with pytest.raises(ValueError, match="must be positive"):
        _env_float("FX1_TIMEOUT_S", -1.0, 1.0)
    assert (
        _RefuseRedirects().redirect_request(  # type: ignore[arg-type]
            urllib.request.Request(_URL),
            None,
            302,
            "found",
            {},
            "https://elsewhere.example/v1",
        )
        is None
    )
    assert hosted_api_url_problem("http://[::1") == "must be a well-formed HTTPS URL"
    assert byok_base_url_problem("http://[::1") == "must be a well-formed http(s) URL"
    assert hosted_api_url_problem("https://user:pass@example.com/v1") == (
        "must not embed userinfo credentials"
    )
    assert "params" in (hosted_api_url_problem("https://example.com/v1;p") or "")
    assert "query" in (hosted_api_url_problem("https://example.com/v1?x=1") or "")
    assert hosted_api_url_problem("https://example.com:70000/v1") == "port must be 1-65535"
    assert hosted_api_url_problem("https://example.com:0/v1") == "port must be 1-65535"
    assert hosted_api_url_problem("https://:443/x") == "must name a host"
    assert hosted_api_url_problem("http://8.8.8.8/v1") == (
        "must use HTTPS unless the host is a literal loopback address"
    )
    assert byok_base_url_problem("http://example.com:0/v1") == "port must be 1-65535"
    assert byok_base_url_problem("https://:443/x") == "must name a host"
    port = _free_tcp_port()
    assert 0 < port < 65536


def test_byok_pinned_open_rejects_a_hostless_url(monkeypatch: pytest.MonkeyPatch) -> None:
    real = backends.urllib.parse.urlparse

    def hostless(url: str, *args: Any, **kwargs: Any) -> Any:
        if url == "http://example.com/v1":
            return urllib.parse.ParseResult("http", "", "/v1", "", "", "")
        return real(url, *args, **kwargs)

    monkeypatch.setattr(backends.urllib.parse, "urlparse", hostless)
    request = urllib.request.Request("http://example.com/v1", data=b"{}", method="POST")
    with pytest.raises(urllib.error.URLError, match="no host"):
        _open_byok_pinned(request, timeout_s=1.0, allow_private=False)


def test_token_count_shapes_and_tokenize_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _extract_token_count(["nope"]) is None
    assert _extract_token_count({"tokens": "nope", "token_ids": [9, 8]}) == 2
    assert _extract_token_count({"data": {"total_tokens": True}, "tokens": [1]}) == 1
    assert _extract_token_count({"tokens": "nope", "token_ids": "nope"}) is None

    _patch_open(monkeypatch, _Body(b"{}"))
    with pytest.raises(RuntimeError, match="needs count"):
        _openai_tokenize_count(
            _URL, model="m", messages=_MSGS, timeout_s=1, api_key=None, label="tok"
        )

    _patch_open(monkeypatch, _Body(b'{"tokens":[1,2,3]}'))
    assert (
        _openai_tokenize_count(
            _URL, model="m", messages=_MSGS, timeout_s=1, api_key="k", label="tok"
        )
        == 3
    )

    _patch_open(monkeypatch, urllib.error.URLError(OSError("down")))
    with pytest.raises(RuntimeError, match="unreachable"):
        _openai_tokenize_count(
            _URL, model="m", messages=_MSGS, timeout_s=1, api_key=None, label="tok"
        )

    _patch_open(monkeypatch, TimeoutError("slow"))
    with pytest.raises(RuntimeError, match="transport fault"):
        _openai_tokenize_count(
            _URL, model="m", messages=_MSGS, timeout_s=1, api_key=None, label="tok"
        )

    _patch_open(monkeypatch, _Body(b"not-json"))
    with pytest.raises(RuntimeError, match="not JSON"):
        _openai_tokenize_count(
            _URL, model="m", messages=_MSGS, timeout_s=1, api_key=None, label="tok"
        )

    _patch_open(monkeypatch, _Encode())
    with pytest.raises(RuntimeError, match="UnicodeEncodeError"):
        _openai_tokenize_count(
            _URL, model="m", messages=_MSGS, timeout_s=1, api_key=None, label="tok"
        )


def test_chat_complete_transport_faults(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_open(monkeypatch, http.client.RemoteDisconnected("reset"))
    with pytest.raises(RuntimeError, match="transport fault"):
        _openai_chat_complete(
            _URL, model="m", messages=_MSGS, timeout_s=1, api_key=None, label="chat"
        )
    _patch_open(monkeypatch, _Encode())
    with pytest.raises(RuntimeError, match="UnicodeEncodeError"):
        _openai_chat_complete(
            _URL, model="m", messages=_MSGS, timeout_s=1, api_key=None, label="chat"
        )


def test_tool_completion_shape_and_faults(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="not an object"):
        _tool_call_shape("nope", "tools", 0)
    with pytest.raises(RuntimeError, match="function"):
        _tool_call_shape({"id": "call_1", "type": "function", "function": "nope"}, "tools", 1)
    shaped = _tool_call_shape(
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        },
        "tools",
        0,
    )
    assert shaped["function"]["name"] == "lookup"

    payload = {
        "choices": [
            {
                "message": {
                    "content": "ok",
                    "tool_calls": [shaped],
                },
                "finish_reason": "tool_calls",
                "logprobs": {"content": []},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    seen = _patch_open(monkeypatch, _Body(json.dumps(payload).encode()))
    result, usage = _openai_chat_complete_tools(
        _URL,
        model="m",
        messages=_MSGS,
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        tool_choice="auto",
        parallel_tool_calls=False,
        timeout_s=1,
        api_key="k",
        label="tools",
        logprobs=True,
        top_logprobs=2,
    )
    assert seen["body"]["tool_choice"] == "auto"
    assert seen["body"]["parallel_tool_calls"] is False
    assert seen["body"]["logprobs"] is True
    assert seen["body"]["top_logprobs"] == 2
    assert result.content == "ok"
    assert result.tool_calls is not None and result.tool_calls[0]["id"] == "call_1"
    assert result.logprobs == {"content": []}
    assert usage is not None and usage["total_tokens"] == 2

    def _refuse(body: dict[str, Any], match: str) -> None:
        _patch_open(monkeypatch, _Body(json.dumps(body).encode()))
        with pytest.raises(RuntimeError, match=match):
            _openai_chat_complete_tools(
                _URL,
                model="m",
                messages=_MSGS,
                tools=None,
                tool_choice=None,
                parallel_tool_calls=None,
                timeout_s=1,
                api_key=None,
                label="tools",
            )

    _refuse({"choices": []}, "missing choices")
    _refuse({"choices": [{"message": "hi"}]}, "not object")
    _refuse({"choices": [{"message": {"content": 1}}]}, "not str")
    _refuse({"choices": [{"message": {"content": "x", "tool_calls": {}}}]}, "not list")
    _refuse(
        {"choices": [{"message": {"content": "x"}, "logprobs": []}]},
        "not object",
    )
    _patch_open(monkeypatch, _http_error(502))
    with pytest.raises(RuntimeError, match="HTTP 502"):
        _openai_chat_complete_tools(
            _URL,
            model="m",
            messages=_MSGS,
            tools=None,
            tool_choice=None,
            parallel_tool_calls=None,
            timeout_s=1,
            api_key=None,
            label="tools",
        )
    _patch_open(monkeypatch, urllib.error.URLError("down"))
    with pytest.raises(RuntimeError, match="unreachable"):
        _openai_chat_complete_tools(
            _URL,
            model="m",
            messages=_MSGS,
            tools=None,
            tool_choice=None,
            parallel_tool_calls=None,
            timeout_s=1,
            api_key=None,
            label="tools",
        )
    _patch_open(monkeypatch, TimeoutError("slow"))
    with pytest.raises(RuntimeError, match="transport fault"):
        _openai_chat_complete_tools(
            _URL,
            model="m",
            messages=_MSGS,
            tools=None,
            tool_choice=None,
            parallel_tool_calls=None,
            timeout_s=1,
            api_key=None,
            label="tools",
        )
    _patch_open(monkeypatch, _Body(b"{"))
    with pytest.raises(RuntimeError, match="not JSON"):
        _openai_chat_complete_tools(
            _URL,
            model="m",
            messages=_MSGS,
            tools=None,
            tool_choice=None,
            parallel_tool_calls=None,
            timeout_s=1,
            api_key=None,
            label="tools",
        )
    _patch_open(monkeypatch, _Encode())
    with pytest.raises(RuntimeError, match="UnicodeEncodeError"):
        _openai_chat_complete_tools(
            _URL,
            model="m",
            messages=_MSGS,
            tools=None,
            tool_choice=None,
            parallel_tool_calls=None,
            timeout_s=1,
            api_key=None,
            label="tools",
        )


def test_stream_and_embedding_faults(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_open(
        monkeypatch,
        _Lines([b'data: {"choices":[{"index":0}]}']),
    )
    with pytest.raises(RuntimeError, match="missing delta"):
        list(
            _openai_chat_stream(
                _URL, model="m", messages=_MSGS, timeout_s=1, api_key=None, label="stream"
            )
        )
    _patch_open(
        monkeypatch,
        _Lines([b'data: {"choices":[{"delta":{"content":1}}]}']),
    )
    with pytest.raises(RuntimeError, match="not str"):
        list(
            _openai_chat_stream(
                _URL, model="m", messages=_MSGS, timeout_s=1, api_key=None, label="stream"
            )
        )
    _patch_open(monkeypatch, urllib.error.URLError("down"))
    with pytest.raises(RuntimeError, match="unreachable"):
        list(
            _openai_chat_stream(
                _URL, model="m", messages=_MSGS, timeout_s=1, api_key=None, label="stream"
            )
        )
    _patch_open(monkeypatch, TimeoutError("slow"))
    with pytest.raises(RuntimeError, match="transport fault"):
        list(
            _openai_chat_stream(
                _URL, model="m", messages=_MSGS, timeout_s=1, api_key=None, label="stream"
            )
        )

    class _IterEncode:
        def __iter__(self) -> Any:
            raise UnicodeEncodeError("utf-8", "x", 0, 1, "no")

    _patch_open(monkeypatch, _IterEncode())
    with pytest.raises(RuntimeError, match="UnicodeEncodeError"):
        list(
            _openai_chat_stream(
                _URL, model="m", messages=_MSGS, timeout_s=1, api_key=None, label="stream"
            )
        )

    with pytest.raises(RuntimeError, match="not an object"):
        _embedding_item_shape(["nope"], "emb", 0)
    with pytest.raises(RuntimeError, match="number"):
        _embedding_item_shape({"object": "embedding", "index": 0, "embedding": None}, "emb", 0)

    seen = _patch_open(
        monkeypatch,
        _Body(
            json.dumps(
                {
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.5]}],
                    "model": "embed-a",
                }
            ).encode()
        ),
    )
    result = _openai_embeddings_complete(
        "https://example.com/v1/embeddings",
        model="embed-a",
        input="hello",
        encoding_format="float",
        dimensions=2,
        user="acct",
        timeout_s=1,
        api_key="k",
        label="emb",
    )
    assert seen["body"]["encoding_format"] == "float"
    assert seen["body"]["dimensions"] == 2
    assert seen["body"]["user"] == "acct"
    assert result.model == "embed-a"
    assert result.data[0]["embedding"] == [0.25, 0.5]

    _patch_open(monkeypatch, _Body(b'{"data":"nope"}'))
    with pytest.raises(RuntimeError, match="not list"):
        _openai_embeddings_complete(
            _URL,
            model="m",
            input="hello",
            encoding_format=None,
            dimensions=None,
            user=None,
            timeout_s=1,
            api_key=None,
            label="emb",
        )
    for outcome, match in (
        (urllib.error.URLError("down"), "unreachable"),
        (TimeoutError("slow"), "transport fault"),
        (_Body(b"{"), "not JSON"),
        (_Encode(), "UnicodeEncodeError"),
    ):
        _patch_open(monkeypatch, outcome)
        with pytest.raises(RuntimeError, match=match):
            _openai_embeddings_complete(
                _URL,
                model="m",
                input="hello",
                encoding_format=None,
                dimensions=None,
                user=None,
                timeout_s=1,
                api_key=None,
                label="emb",
            )


def test_hosted_stream_without_usage_exits_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = HostedK3Backend(api_key="k", timeout_s=1, api_url="https://example.com/v1")
    _patch_open(
        monkeypatch,
        _Lines(
            [
                b'data: {"choices":[{"delta":{"content":"hi"}}]}\n',
                b"data: [DONE]\n",
            ]
        ),
    )
    assert list(backend.stream([{"role": "user", "content": "hi"}])) == ["hi"]
    assert backend.last_usage is None
    assert backend.total_usage == {}


def test_hosted_timeout_and_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        HostedK3Backend(api_key="k", timeout_s=0, api_url="https://example.com/v1")
    backend = HostedK3Backend(api_key="k", timeout_s=1, api_url="https://example.com/v1")
    for outcome, match in (
        (_http_error(429), "HTTP 429"),
        (urllib.error.URLError("down"), "unreachable"),
        (TimeoutError("slow"), "transport fault"),
        (_Body(b"{"), "not JSON"),
        (_Encode(), "UnicodeEncodeError"),
    ):
        _patch_open(monkeypatch, outcome)
        with pytest.raises(RuntimeError, match=match):
            backend.complete(_MSGS)


def test_local_backend_attach_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FX1_CHECKPOINT_DIR", raising=False)
    assert LocalFx1Backend._shares_env_checkpoint(_BASE) is False
    monkeypatch.setenv("FX1_CHECKPOINT_DIR", str(_BASE))
    assert LocalFx1Backend._shares_env_checkpoint(_BASE) is True

    def explode(self: Path, *args: Any, **kwargs: Any) -> Path:
        raise OSError("resolve failed")

    monkeypatch.setattr(backends.Path, "resolve", explode)
    monkeypatch.setenv("FX1_CHECKPOINT_DIR", "/tmp/fx1-unresolvable")
    assert LocalFx1Backend._shares_env_checkpoint(_BASE) is False

    monkeypatch.undo()
    monkeypatch.delenv("FX1_CHECKPOINT_DIR", raising=False)
    monkeypatch.delenv("FX1_LOCAL_SERVE_URL", raising=False)
    monkeypatch.delenv("FX1_LOCAL_SERVE_CMD", raising=False)
    dedicated = LocalFx1Backend(_BASE)
    assert "127.0.0.1" in dedicated._url
    assert "fx1.serve.local_engine" in dedicated._serve_cmd
    with dedicated as entered:
        assert entered is dedicated

    broken = tmp_path / "broken-manifest"
    shutil.copytree(_BASE, broken)
    (broken / "weights.manifest.json").write_text("{", encoding="utf-8")
    attached = LocalFx1Backend(broken, serve_url="http://127.0.0.1:9/v1", serve_cmd="")
    assert attached._expected_weights_sha is None

    def http_error(*args: Any, **kwargs: Any) -> Any:
        raise _http_error(503)

    monkeypatch.setattr(backends.urllib.request, "urlopen", http_error)
    assert attached._engine_up() is True

    def down(*args: Any, **kwargs: Any) -> Any:
        raise urllib.error.URLError("down")

    monkeypatch.setattr(backends.urllib.request, "urlopen", down)
    attached._expected_weights_sha = "ab" * 32
    attached._verified_attach = False
    attached._verify_served_weights()
    assert attached._verified_attach is False

    attached._url = ""
    with pytest.raises(BackendNotConfiguredError, match="not configured"):
        attached.complete_with_tools(_MSGS)
    with pytest.raises(BackendNotConfiguredError, match="not configured"):
        attached.count_tokens(_MSGS)


def test_ensure_engine_lock_races(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FX1_CHECKPOINT_DIR", raising=False)
    backend = LocalFx1Backend(_BASE, serve_url="http://127.0.0.1:9/v1", serve_cmd="echo fx1-engine")
    backend._verified_attach = False
    backend._proc = None

    class _Claim:
        def __enter__(self) -> _Claim:
            backend._proc = object()  # type: ignore[assignment]
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    backend._engine_lock = _Claim()  # type: ignore[assignment]
    monkeypatch.setattr(backend, "_engine_up", lambda: False)
    backend._ensure_engine()

    fresh = LocalFx1Backend(_BASE, serve_url="http://127.0.0.1:9/v1", serve_cmd="echo fx1-engine")
    fresh._verified_attach = False
    fresh._proc = None
    calls = {"n": 0}

    def up() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    verified = {"ok": False}
    monkeypatch.setattr(fresh, "_engine_up", up)
    monkeypatch.setattr(fresh, "_verify_served_weights", lambda: verified.__setitem__("ok", True))
    fresh._ensure_engine()
    assert verified["ok"] is True
    assert calls["n"] == 2


def test_byok_optional_fields_round_trip_through_compat(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAICompatBackend is constructed only to prove the helpers stay wired."""
    backend = OpenAICompatBackend(
        base_url="https://example.com/v1", api_key="k", model="m", timeout_s=1
    )
    seen = _patch_open(
        monkeypatch,
        _Body(json.dumps({"choices": [{"message": {"content": "pong"}}]}).encode()),
    )
    assert backend.complete(_MSGS) == "pong"
    assert seen["body"]["model"] == "m"
