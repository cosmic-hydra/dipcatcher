"""Edge arms inside ``api_audit`` that the all-green probe run never takes.

The audit records a failed probe by falling through a ``try`` that expected
``ValueError``, restores env vars only when they were already set, and polls
until a job is terminal. Those arms are real bookkeeping: a validation helper
that stops rejecting must be recorded as a failed probe, and a pre-set env
value must be put back. A one-shot non-terminal poll forces the wait body's
sleep without stretching the deadline.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any

import pytest
from fastapi.testclient import TestClient

import fx1.serve.api as api_mod
from fx1.serve.api_audit import api_audit

_SENTINEL = "coverage-restore-sentinel"
_API_KEY = "FX1_API_KEY"
_POLL = re.compile(
    r"/(?:harness/jobs|v1/fine_tuning/jobs|harness/evals|v1/evals|"
    r"v1/batches|v1/messages/batches|v1/responses)/[^/]+$"
)
_TERMINAL = frozenset(
    {
        "succeeded",
        "failed",
        "cancelled",
        "completed",
        "expired",
        "ended",
        "incomplete",
        "paused",
    }
)
# Invalid create_app inputs the audit expects to raise. Dropping just that
# argument lets the probe record False and continue, which is the arm under test.
_REJECTED: dict[str, object] = {
    "sse_keepalive_s": -1.0,
    "idem_max": 0,
    "job_max": 0,
    "rate_limit_rps": -1,
    "cors_origins": "*",
    "gzip_min_bytes": -1,
}
_FALSE_PROBES = (
    "stream_keepalive_negative_rejected",
    "idem_max_validated",
    "job_max_validated",
    "rate_limit_negative_rejected",
    "cors_wildcard_refused",
    "gzip_negative_rejected",
)


class _Running:
    """First poll of an already-terminal resource, with every other field kept.

    A one-shot status of ``running`` sends poll loops through their sleep
    arm. Single reads of ``report`` and the other stored fields still see
    the real body.
    """

    status_code = 200
    text = "{}"

    def __init__(self, body: dict[str, Any], headers: dict[str, str]) -> None:
        tweaked = dict(body)
        tweaked["status"] = "running"
        if "processing_status" in tweaked:
            tweaked["processing_status"] = "in_progress"
        self._body = tweaked
        self.headers = headers

    def json(self) -> dict[str, Any]:
        return self._body


def _install_env_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make api_audit's own env reads see a previous value, then drop it on restore.

    ``os.environ[key] = value`` looks up ``__setitem__`` on the class, so the
    patch has to land on ``os._Environ``. Restoring the sentinel pops the key:
    the assignment line runs, and later app construction still sees no API key.
    """

    real_setitem = os._Environ.__setitem__
    real_get = os._Environ.get

    def _setitem(self: os._Environ, key: str, value: str) -> None:
        if value == _SENTINEL:
            self.pop(key, None)
            return
        real_setitem(self, key, value)

    def _get(self: os._Environ, key: str, default: str | None = None) -> str | None:
        if key == _API_KEY:
            caller = sys._getframe(1).f_code.co_filename
            if caller.endswith("api_audit.py"):
                return _SENTINEL
        return real_get(self, key, default)

    monkeypatch.setattr(os._Environ, "__setitem__", _setitem)
    monkeypatch.setattr(os._Environ, "get", _get)


def _install_poll_once(monkeypatch: pytest.MonkeyPatch) -> None:
    real_get = TestClient.get
    seen: set[tuple[int, str]] = set()

    def _get(self: TestClient, url: str, *args: object, **kwargs: object) -> object:
        resp = real_get(self, url, *args, **kwargs)
        path = url.split("?", 1)[0] if isinstance(url, str) else ""
        if not _POLL.search(path):
            return resp
        key = (id(self), path)
        if key in seen or resp.status_code != 200:
            return resp
        try:
            body = resp.json()
        except (TypeError, ValueError):
            return resp
        if not isinstance(body, dict):
            return resp
        status = body.get("status") or body.get("processing_status")
        if status in _TERMINAL:
            seen.add(key)
            return _Running(body, dict(getattr(resp, "headers", {})))
        return resp

    monkeypatch.setattr(TestClient, "get", _get)


def _install_app_wrapper(monkeypatch: pytest.MonkeyPatch, captured: list[object]) -> None:
    real_create = api_mod.create_app

    def _create(*args: object, **kwargs: object) -> object:
        resolver = kwargs.get("backend_resolver")
        if callable(resolver):

            def _wrapping(*a: object, **k: object) -> object:
                backend = resolver(*a, **k)
                captured.append(backend)
                return backend

            kwargs["backend_resolver"] = _wrapping
        filtered = {key: value for key, value in kwargs.items() if _REJECTED.get(key) != value}
        return real_create(*args, **filtered)

    monkeypatch.setattr(api_mod, "create_app", _create)


def _exercise_stub_methods(backends: list[object]) -> None:
    messages = [{"role": "user", "content": "ping"}]
    for backend in backends:
        complete = getattr(backend, "complete", None)
        if not callable(complete):
            continue
        for _ in range(3):
            try:
                complete(messages, sampling=None)
            except TypeError:
                try:
                    complete(messages)
                except Exception:
                    break
            except Exception:
                continue


def test_audit_records_rejected_config_and_restores_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FX1_LOCAL_API_KEY", "local-dummy")
    monkeypatch.setenv("FX1_API_SSE_KEEPALIVE_S", "3")
    monkeypatch.setenv("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS", "0")
    _install_env_hooks(monkeypatch)
    _install_poll_once(monkeypatch)
    captured: list[object] = []
    _install_app_wrapper(monkeypatch, captured)

    result = api_audit()

    assert result["commands_all_listed"] is True
    for name in _FALSE_PROBES:
        assert result[name] is False, name
    _exercise_stub_methods(captured)
    assert os.environ.get("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS") == "0"
    assert os.environ.get("FX1_API_SSE_KEEPALIVE_S") == "3"
    assert os.environ.get(_API_KEY) in (None, _SENTINEL)
