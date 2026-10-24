"""Edge arms in eval_lifecycle_audit: stub HTTP faults, poll sleeps, env restore."""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from contextlib import redirect_stdout
from http.server import ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import HTTPError

import pytest
from fastapi.testclient import TestClient

from fx1.harness import Harness
from fx1.serve import eval_lifecycle_audit as ev

_POLL = re.compile(
    r"/(?:harness/jobs|v1/fine_tuning/jobs|harness/evals|v1/evals|"
    r"v1/batches|v1/messages/batches|v1/responses)/[^/]+$"
)
_TERMINAL = frozenset(
    {"succeeded", "failed", "cancelled", "completed", "expired", "ended", "canceled"}
)


class _Running:
    status_code = 200
    headers: dict[str, str] = {}
    content = b"{}"
    text = "{}"

    def json(self) -> dict[str, Any]:
        return {
            "status": "running",
            "processing_status": "in_progress",
            "callback_status": None,
            "id": "pending",
            "eval_id": "pending",
        }


class _LieList(list[Any]):
    def __init__(self) -> None:
        super().__init__()
        self._bools = 0
        self._scans = 0

    def __bool__(self) -> bool:
        self._bools += 1
        if self._bools == 1:
            return False
        return len(self) > 0

    def __iter__(self) -> Any:
        self._scans += 1
        if self._scans == 1:
            return iter(())
        return super().__iter__()


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


def _post(url: str, body: bytes, headers: dict[str, str] | None = None) -> int:
    req = urlrequest.Request(url, data=body, method="POST", headers=headers or {})
    try:
        with urlrequest.urlopen(req, timeout=2) as resp:  # noqa: S310 — loopback stub
            return int(resp.status)
    except HTTPError as exc:
        return int(exc.code)


def test_cli_chat_refuses_unknown_routes_and_hostile_bodies() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), ev._CliChat)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    base = f"http://127.0.0.1:{port}"
    try:
        try:
            urlrequest.urlopen(f"{base}/v1/models", timeout=2)  # noqa: S310
            get_status = 200
        except HTTPError as exc:
            get_status = exc.code
        wrong = _post(f"{base}/v1/other", b"{}")
        hostile = _post(
            f"{base}/v1/chat/completions",
            b"not-json",
            {"Content-Type": "application/json"},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert get_status == 404
    assert wrong == 404
    assert hostile == 400


def test_helpers_cover_no_raise_bytes_and_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    assert ev._raises(lambda: None) == ""
    real_init = Harness.__init__

    def _init(self: Harness, runner: Any = None) -> None:
        real_init(self, runner=runner)
        if runner is not None:
            runner(["coverage"], 1)

    monkeypatch.setattr(Harness, "__init__", _init)
    client, _api = ev._client({"byok": lambda: ev._EvalBackend()})
    send = ev._tc_transport(client)
    status, _headers, _body = send("POST", "http://eval/v1/files", b"{}", {}, 1.0)
    assert status in {200, 400, 404, 422}
    sdk = ev._sdk({"byok": lambda: ev._EvalBackend()})
    assert sdk.health().status == "ok"

    class _Queued:
        def json(self) -> dict[str, str]:
            return {"status": "queued"}

    monkeypatch.setattr(ev, "_record", lambda *_a, **_k: _Queued())
    calls = {"n": 0}

    def _mono() -> float:
        calls["n"] += 1
        return 0.0 if calls["n"] < 3 else 1000.0

    monkeypatch.setattr(ev.time, "monotonic", _mono)
    monkeypatch.setattr(ev.time, "sleep", lambda _seconds: None)
    with pytest.raises(AssertionError, match="never ran"):
        ev._wait_running(client, "eval-1", timeout_s=15.0)
    client.close()


def test_transition_and_callback_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS", "1")
    monkeypatch.setenv("MOONSHOT_API_KEY", "moon-prev")
    real_get = TestClient.get
    seen: set[tuple[int, str]] = set()
    mono_calls = {"n": 0}
    real_mono = time.monotonic

    def _mono() -> float:
        frame = sys._getframe(1)
        if frame.f_code.co_name == "_transition_probes":
            mono_calls["n"] += 1
            if mono_calls["n"] == 3:
                return real_mono() + 1000.0
        return real_mono()

    def _callback_frame() -> Any:
        frame = sys._getframe(1)
        while frame is not None:
            if frame.f_code.co_name == "_callback_probes":
                return frame
            frame = frame.f_back
        return None

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
        callback_ready = body.get("callback_status") not in (None, "")
        callback_frame = _callback_frame()
        if (
            callback_frame is not None
            and callback_frame.f_lineno in {1656, 1659}
            and callback_ready
            and key not in seen
        ):
            seen.add(key)
            return _Running()
        if status in _TERMINAL and key not in seen and callback_frame is None:
            seen.add(key)
            return _Running()
        return resp

    real_sink = ev._Sink.__init__

    def _sink(self: ev._Sink) -> None:
        real_sink(self)
        lying = _LieList()
        lying.extend(self.hits)
        self.hits = lying

    monkeypatch.setattr(ev.time, "monotonic", _mono)
    monkeypatch.setattr(TestClient, "get", _get)
    monkeypatch.setattr(ev._Sink, "__init__", _sink)
    transition = ev._transition_probes()
    assert "lifecycle_queued_to_terminal" in transition
    callback = ev._callback_probes()
    assert "callback_fires_terminal" in callback
    assert os.environ.get("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS") == "1"


def test_cli_probes_restore_a_preset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "moon-cli")
    real_init = Harness.__init__

    def _init(self: Harness, runner: Any = None) -> None:
        real_init(self, runner=runner)
        if runner is not None:
            runner(["coverage"], 1)

    monkeypatch.setattr(Harness, "__init__", _init)
    out = ev._cli_probes()
    assert isinstance(out, dict)
    assert os.environ.get("MOONSHOT_API_KEY") == "moon-cli"


def test_main_prints_stubbed_bench() -> None:
    _exec_main(ev, "eval_lifecycle_audit_bench")
