"""Edge arms in header_audit: parsers, poll sleeps, env restore, and TCP faults."""

from __future__ import annotations

import json
import os
import re
import socket
import sys
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fx1.harness import Harness
from fx1.serve import header_audit as hdr

_POLL = re.compile(
    r"/(?:harness/jobs|v1/fine_tuning/jobs|harness/evals|v1/evals|"
    r"v1/batches|v1/messages/batches|v1/responses)/[^/]+$"
)
_TERMINAL = frozenset(
    {"succeeded", "failed", "cancelled", "completed", "expired", "ended", "incomplete", "paused"}
)


class _Running:
    status_code = 200
    text = "{}"
    headers: dict[str, str] = {}
    content = b"{}"

    def json(self) -> dict[str, Any]:
        return {"status": "running", "processing_status": "in_progress", "callback_status": None}


def _exec_main(mod: Any, bench: str) -> None:
    path = mod.__file__
    assert isinstance(path, str)
    source = Path(path).read_text(encoding="utf-8")

    def _stub_bench() -> dict[str, bool]:
        return {"ok": True}

    class _NS(dict[str, Any]):
        def __setitem__(self, key: str, value: object) -> None:
            if key == bench and callable(value):
                value = _stub_bench
            dict.__setitem__(self, key, value)

    ns = _NS()
    ns.update({"__name__": "__main__", "__file__": path, "__package__": mod.__package__})
    with redirect_stdout(StringIO()):
        exec(compile(source, path, "exec"), ns, ns)  # noqa: S102 — module __main__ line


def _install_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    real_get = TestClient.get
    seen: set[tuple[int, str]] = set()

    def _get(self: TestClient, url: str, *args: object, **kwargs: object) -> object:
        resp = real_get(self, url, *args, **kwargs)
        path = url.split("?", 1)[0] if isinstance(url, str) else ""
        if not _POLL.search(path) or resp.status_code != 200:
            return resp
        key = (id(self), path)
        if key in seen:
            return resp
        try:
            body = resp.json()
        except (TypeError, ValueError):
            return resp
        if not isinstance(body, dict):
            return resp
        status = body.get("status") or body.get("processing_status")
        if status in _TERMINAL or body.get("callback_status") not in (None, ""):
            seen.add(key)
            return _Running()
        return resp

    monkeypatch.setattr(TestClient, "get", _get)


def _call_runners(monkeypatch: pytest.MonkeyPatch) -> None:
    real_init = Harness.__init__

    def _init(self: Harness, runner: Any = None) -> None:
        real_init(self, runner=runner)
        if runner is not None:
            runner(["coverage"], 1)

    monkeypatch.setattr(Harness, "__init__", _init)


def test_spy_resolver_records_checkpoint_and_rejects_unknown() -> None:
    spy = hdr._SpyResolver()
    backend = spy("byok", "/ckpt", timeout_s=1)
    assert backend is not None
    assert spy.calls[-1][1]["checkpoint_dir"] == "/ckpt"
    with pytest.raises(KeyError):
        spy("missing-link")


def test_client_restores_a_previous_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _call_runners(monkeypatch)
    with hdr._audit_context():
        # The audit context clears FX1_* before yielding, so the previous
        # value has to be set inside that window.
        os.environ[hdr._API_KEY_ENV] = "header-prev"
        client, _api = hdr._client(api_key=None)
        client.close()
        assert os.environ[hdr._API_KEY_ENV] == "header-prev"


def test_error_parsers_tolerate_non_json_and_wrong_shape() -> None:
    class _Bad:
        def json(self) -> dict[str, Any]:
            raise ValueError("not json")

    class _Plain:
        def json(self) -> dict[str, Any]:
            return {"type": "message", "error": "nope"}

    assert hdr._openai_err(_Bad()) is None
    assert hdr._anthropic_err(_Bad()) is None
    assert hdr._anthropic_err(_Plain()) is None
    assert hdr._flat_err(_Bad()) is None


def test_raw_send_ignores_unrelated_asgi_messages() -> None:
    async def _app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        del scope, receive
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})
        await send({"type": "lifespan.startup"})

    resp = hdr._raw(_app, "GET", "/ignored", [], b"")
    assert resp["status"] == 204


def test_tcp_endpoint_accept_and_write_faults(monkeypatch: pytest.MonkeyPatch) -> None:
    real_accept = socket.socket.accept
    stage = {"n": 0}

    class _Conn:
        def recv(self, _n: int) -> bytes:
            raise OSError("reset")

        def sendall(self, _body: bytes) -> None:
            raise OSError("reset")

        def close(self) -> None:
            return None

    def _accept(self: socket.socket) -> Any:
        if sys._getframe(1).f_code.co_name != "_serve":
            return real_accept(self)
        stage["n"] += 1
        if stage["n"] == 1:
            return _Conn(), ("127.0.0.1", 1)
        raise OSError("listener closed")

    monkeypatch.setattr(socket.socket, "accept", _accept)
    endpoint = hdr._TcpEndpoint(sleep_s=30.0)
    try:
        endpoint._thread.join(timeout=2.0)
    finally:
        endpoint.close()


def test_envelope_parse_failures_record_empty_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    real_loads = json.loads

    def _loads(text: str | bytes, *args: Any, **kwargs: Any) -> Any:
        if sys._getframe(1).f_code.co_name == "_envelope_probes":
            raise ValueError("coverage")
        return real_loads(text, *args, **kwargs)

    monkeypatch.setattr(hdr.json, "loads", _loads)
    with hdr._audit_context():
        out = hdr._envelope_probes()
    assert "envelope_bad_content_length_v1" in out
    assert "envelope_too_large_v1" in out


def test_timeout_batch_poll_sleeps_once_when_already_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_poll(monkeypatch)
    real_mono = time.monotonic
    calls = {"n": 0}

    def _mono() -> float:
        frame = sys._getframe(1)
        if frame.f_code.co_name == "_timeout_enforcement_probes" and frame.f_lineno in {723, 752}:
            calls["n"] += 1
            # Second check of each poll while: the deadline has passed.
            if calls["n"] in {2, 4}:
                return real_mono() + 1000.0
        return real_mono()

    monkeypatch.setattr(hdr.time, "monotonic", _mono)
    with hdr._audit_context():
        out = hdr._timeout_enforcement_probes()
    assert "timeout_batch_per_line_header" in out


def test_main_prints_stubbed_bench() -> None:
    _exec_main(hdr, "header_audit_bench")
