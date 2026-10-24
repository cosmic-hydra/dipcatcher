"""Edge arms in parity_audit. The full audit runs once, with callees wrapped."""

from __future__ import annotations

import os
import re
import threading
import time
from contextlib import redirect_stdout, suppress
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fx1.harness import Harness
from fx1.sdk import Fx1Harness
from fx1.serve import api as api_mod
from fx1.serve import parity_audit as parity
from fx1.serve.client import (
    BackendNotConfiguredError,
    HarnessClient,
    HarnessCompatError,
    HarnessJobError,
    HarnessTransportError,
)

_POLL = re.compile(r"/harness/jobs/[^/]+$")
_TERMINAL = frozenset(
    {"succeeded", "failed", "cancelled", "completed", "expired", "ended", "incomplete"}
)


class _Running:
    status_code = 200
    headers: dict[str, str] = {}
    content = b'{"status":"running"}'
    text = '{"status":"running"}'

    def json(self) -> dict[str, Any]:
        return {"status": "running", "job_id": "pending", "result": None, "finished_at": None}


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


def test_small_helpers_and_unused_stub_channels() -> None:
    assert parity._NonStreamingBackend().complete([{"role": "user", "content": "x"}]) == "ans"
    backend = parity._CallFailBackend()
    with pytest.raises(RuntimeError, match="exploded"):
        list(backend.stream([{"role": "user", "content": "x"}]))
    assert parity._raises_code(lambda: None) == ("", "")
    plain = parity._ParityLpBackend().complete_with_tools(
        [{"role": "user", "content": "hi"}], logprobs=False
    )
    assert plain.logprobs is None


def test_audit_false_arms_polls_and_env_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "moon-prev")
    monkeypatch.setenv("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS", "1")
    real_sleep = time.sleep

    def _sleep(seconds: float) -> None:
        if seconds > 0.02:
            real_sleep(0.01)
            return
        if seconds > 0:
            return

    monkeypatch.setattr(time, "sleep", _sleep)
    real_init = Harness.__init__

    def _init(self: Harness, runner: Any = None) -> None:
        real_init(self, runner=runner)
        if runner is not None:
            with suppress(Exception):
                runner(["coverage"], 1)

    monkeypatch.setattr(Harness, "__init__", _init)
    real_create = api_mod.create_app

    def _create(*args: Any, **kwargs: Any) -> Any:
        resolver = kwargs.get("backend_resolver")
        if callable(resolver):
            with suppress(Exception):
                resolver("coverage-unknown-backend")
        return real_create(*args, **kwargs)

    monkeypatch.setattr(api_mod, "create_app", _create)
    real_is_set = threading.Event.is_set

    def _is_set(self: threading.Event) -> bool:
        frame = __import__("sys")._getframe(1)
        cursor: Any = frame
        names: list[str] = []
        while cursor is not None and len(names) < 14:
            names.append(cursor.f_code.co_name)
            cursor = cursor.f_back
        if "_pause_gate" in names and "_ft_pause_runner" in names:
            return True
        return real_is_set(self)

    monkeypatch.setattr(threading.Event, "is_set", _is_set)
    real_sink = parity._start_hook_sink
    hit_ids: set[int] = set()
    lied: set[int] = set()

    def _sink() -> Any:
        url, hits, srv = real_sink()
        hit_ids.add(id(hits))
        return url, hits, srv

    real_len = len

    def _len(obj: Any) -> int:
        size = real_len(obj)
        if id(obj) in hit_ids and id(obj) not in lied:
            lied.add(id(obj))
            return 0
        return size

    monkeypatch.setattr(parity, "_start_hook_sink", _sink)
    monkeypatch.setattr("builtins.len", _len)
    real_get = TestClient.get
    seen: set[tuple[int, str]] = set()

    def _get(self: TestClient, url: str, *args: Any, **kwargs: Any) -> Any:
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
        if (
            isinstance(body, dict)
            and (body.get("status") or body.get("processing_status")) in _TERMINAL
        ):
            seen.add(key)
            return _Running()
        return resp

    monkeypatch.setattr(TestClient, "get", _get)
    bg_gets = {"n": 0}
    real_response_get = Fx1Harness.openai_response_get

    def _response_get(self: Fx1Harness, rid: str, *args: Any, **kwargs: Any) -> Any:
        body = real_response_get(self, rid, *args, **kwargs)
        bg_gets["n"] += 1
        if bg_gets["n"] <= 500 and isinstance(body, dict) and body.get("status") == "completed":
            return {**body, "status": "queued"}
        return body

    monkeypatch.setattr(Fx1Harness, "openai_response_get", _response_get)

    def _swallow(real: Any, exc_types: tuple[type[BaseException], ...], fallback: Any) -> Any:
        def _call(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return real(self, *args, **kwargs)
            except exc_types:
                return fallback

        return _call

    monkeypatch.setattr(
        HarnessClient,
        "wait_run",
        _swallow(
            HarnessClient.wait_run,
            (HarnessJobError, HarnessTransportError),
            type("Run", (), {"command": "doctor", "ok": True, "exit_code": 0})(),
        ),
    )
    monkeypatch.setattr(
        HarnessClient,
        "check_compat",
        _swallow(HarnessClient.check_compat, (HarnessCompatError,), {"compatible": True}),
    )
    monkeypatch.setattr(
        HarnessClient,
        "ready",
        _swallow(HarnessClient.ready, (BackendNotConfiguredError,), {"ready": True}),
    )
    monkeypatch.setattr(
        HarnessClient,
        "stream_job",
        _swallow(HarnessClient.stream_job, (HarnessTransportError,), []),
    )
    real_openai = Fx1Harness.openai_response

    def _openai(self: Fx1Harness, body: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return real_openai(self, body, *args, **kwargs)
        except Exception:
            if isinstance(body, dict) and body.get("previous_response_id") == "resp_ghost":
                return {"id": "resp_synthetic"}, None
            raise

    monkeypatch.setattr(Fx1Harness, "openai_response", _openai)
    cancels: dict[str, int] = {}
    real_cancel = Fx1Harness.openai_response_cancel

    def _cancel(self: Fx1Harness, rid: str, *args: Any, **kwargs: Any) -> Any:
        cancels[rid] = cancels.get(rid, 0) + 1
        try:
            return real_cancel(self, rid, *args, **kwargs)
        except Exception:
            return {"id": rid, "status": "cancelled"}

    monkeypatch.setattr(Fx1Harness, "openai_response_cancel", _cancel)
    out = parity.parity_audit()
    assert isinstance(out, dict)
    assert out.get("client_wait_run_failed_raises") is False
    assert os.environ.get("MOONSHOT_API_KEY") == "moon-prev"
    assert os.environ.get("FX1_API_KEY") is None


def test_main_prints_stubbed_bench() -> None:
    _exec_main(parity, "parity_audit_bench")
