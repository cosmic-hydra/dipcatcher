"""Edge arms in byok_audit: broken pipes, positional resolver args, and no-raise probes."""

from __future__ import annotations

import os
from contextlib import redirect_stdout, suppress
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fx1.harness import Harness
from fx1.sdk import Fx1Harness
from fx1.serve import api as api_mod
from fx1.serve import byok_audit as byok
from fx1.serve.backends import OpenAICompatBackend


@pytest.fixture(autouse=True)
def _resources() -> Any:
    with byok._audit_context():
        yield


_TERMINAL = frozenset({"completed", "failed", "expired", "cancelled"})


class _Body:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class _Handler:
    def send_response(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def send_header(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def end_headers(self) -> None:
        return None

    class wfile:
        @staticmethod
        def write(_body: bytes) -> None:
            raise BrokenPipeError("closed")

        @staticmethod
        def flush() -> None:
            return None


class _Json:
    def __init__(self, status: int, body: dict[str, Any]) -> None:
        self.status_code = status
        self._body = body
        self.text = ""
        self.headers: dict[str, str] = {}

    def json(self) -> dict[str, Any]:
        return dict(self._body)


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


def test_send_and_sse_swallow_a_dropped_client() -> None:
    upstream = byok._StubUpstream()
    upstream._send(_Handler(), 200, b"{}")
    upstream.mode = "sse_ok"
    upstream._respond(_Handler())  # type: ignore[arg-type]
    upstream.mode = "not-a-mode"
    upstream._respond(_Handler())  # type: ignore[arg-type]


def test_err_reads_an_envelope() -> None:
    assert byok._err(_Body({"error": {"code": "x"}})) == {"code": "x"}
    assert byok._err(_Body({"detail": "nope"})) == {"detail": "nope"}

    class _NotDict:
        def get(self, _key: str) -> str:
            return "text"

    assert byok._err(_Body(_NotDict())) == {}


def test_client_restores_api_key_and_calls_nested_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(byok._API_KEY_ENV, "byok-prev")
    real_init = Harness.__init__

    def _init(self: Harness, runner: Any = None) -> None:
        real_init(self, runner=runner)
        if runner is not None:
            runner(["coverage"], 1)

    monkeypatch.setattr(Harness, "__init__", _init)
    real_create = api_mod.create_app

    def _create(*args: Any, **kwargs: Any) -> Any:
        resolver = kwargs.get("backend_resolver")
        if callable(resolver):
            with suppress(Exception):
                resolver(
                    "byok",
                    "/ckpt",
                    {"base_url": "http://127.0.0.1:9/v1", "api_key": "k", "model": "m"},
                    2.5,
                )
        return real_create(*args, **kwargs)

    monkeypatch.setattr(api_mod, "create_app", _create)
    client, _api, calls = byok._client({"hosted_k3": lambda: byok._LocalStub("local")})
    client.close()
    assert os.environ[byok._API_KEY_ENV] == "byok-prev"
    assert isinstance(calls, list)


def test_backend_probes_record_a_missed_raise_and_restore_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FX1_BYOK_BASE_URL", "https://restore.example/v1")
    real_init = OpenAICompatBackend.__init__
    calls = {"n": 0}

    def _init(self: OpenAICompatBackend, *args: Any, **kwargs: Any) -> None:
        calls["n"] += 1
        if calls["n"] in {1, 4}:
            self._model = "skipped"  # type: ignore[attr-defined]
            self._url = "https://skipped"  # type: ignore[attr-defined]
            return
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(OpenAICompatBackend, "__init__", _init)
    out = byok._backend_probes()
    assert out["missing_all_raises"] is False
    assert out["error_names_env"] is False
    assert os.environ.get("FX1_BYOK_BASE_URL") == "https://restore.example/v1"


def test_fallback_wrong_link_records_false(monkeypatch: pytest.MonkeyPatch) -> None:
    real_init = Fx1Harness.__init__

    def _init(self: Fx1Harness, *args: Any, **kwargs: Any) -> None:
        real_init(self, *args, **kwargs)
        resolver = kwargs.get("backend_resolver")
        if callable(resolver) and getattr(resolver, "__name__", "") == "sdk_resolver":
            resolver("hosted_k3")
            resolver(
                "byok",
                "/ckpt",
                {"base_url": "http://127.0.0.1:9/v1", "api_key": "k", "model": "m"},
                {"model": "m2"},
            )

    monkeypatch.setattr(Fx1Harness, "__init__", _init)
    real_complete = Fx1Harness.complete

    def _complete(self: Fx1Harness, messages: Any, **kwargs: Any) -> Any:
        if kwargs.get("backend") == "hosted_k3" and "byok" in kwargs and "fallbacks" not in kwargs:
            kwargs = dict(kwargs)
            kwargs.pop("byok")
        return real_complete(self, messages, **kwargs)

    monkeypatch.setattr(Fx1Harness, "complete", _complete)
    out = byok._probe_fallback()
    assert out["fb_sdk_wrong_link_raises"] is False


def test_batch_upload_failure_returns_early(monkeypatch: pytest.MonkeyPatch) -> None:
    real_post = TestClient.post

    def _post(self: TestClient, url: str, *args: Any, **kwargs: Any) -> Any:
        if url == "/v1/files":
            return _Json(400, {"error": {"message": "no"}})
        return real_post(self, url, *args, **kwargs)

    monkeypatch.setattr(TestClient, "post", _post)
    out = byok._probe_store_replay_batch()
    assert out["batch_upload_200"] is False


def test_batch_without_output_file_records_false(monkeypatch: pytest.MonkeyPatch) -> None:
    real_get = TestClient.get

    def _get(self: TestClient, url: str, *args: Any, **kwargs: Any) -> Any:
        path = url.split("?", 1)[0]
        if path.count("/") == 3 and path.startswith("/v1/batches/"):
            return _Json(200, {"status": "in_progress", "id": "pending"})
        return real_get(self, url, *args, **kwargs)

    monkeypatch.setattr(TestClient, "get", _get)
    monkeypatch.setattr(byok.time, "sleep", lambda _seconds: None)
    out = byok._probe_store_replay_batch()
    assert out["batch_lines_answers"] is False
    assert out["batch_no_creds"] is False


def test_main_prints_stubbed_bench() -> None:
    _exec_main(byok, "byok_audit_bench")
