"""Edges inside observe_audit that the all-green battery never takes.

Parser rejects, the client transport's DELETE and raw-body arms, a poll
that is not already terminal, and the generic ``except`` in the client
error map. Happy-path ``observe_audit()`` stays untouched.
"""

from __future__ import annotations

import runpy
from typing import Any

import pytest

import fx1.serve.observe_audit as observe_audit
from fx1.serve.client import HarnessClient


class _Resp:
    def __init__(
        self,
        status: int = 200,
        payload: Any = None,
        *,
        content: bytes = b"",
        exc: BaseException | None = None,
    ) -> None:
        self.status_code = status
        self._payload = {} if payload is None else payload
        self._exc = exc
        self.content = content
        self.headers = {"content-type": "application/json"}

    def json(self) -> Any:
        if self._exc is not None:
            raise self._exc
        return self._payload


class _Transport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get(self, path: str, headers: dict[str, str] | None = None) -> _Resp:
        del headers
        self.calls.append(("GET", path))
        return _Resp(200, content=b"get")

    def delete(self, path: str, headers: dict[str, str] | None = None) -> _Resp:
        del headers
        self.calls.append(("DELETE", path))
        return _Resp(204, content=b"")

    def post(
        self,
        path: str,
        content: bytes | None = None,
        json: Any = None,
        headers: dict[str, str] | None = None,
    ) -> _Resp:
        del json, headers
        kind = "bytes" if isinstance(content, bytes) else "json"
        self.calls.append((kind, path))
        return _Resp(201, content=content or b"")


class _JobPoll:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, url: str, headers: dict[str, str] | None = None) -> _Resp:
        del url, headers
        self.calls += 1
        status = "running" if self.calls == 1 else "succeeded"
        return _Resp(200, {"status": status})


def test_resources_require_the_audit_context() -> None:
    with pytest.raises(RuntimeError, match="audit resource context"):
        observe_audit._resources()


def test_stub_stream_yields_two_tokens() -> None:
    backend = observe_audit._StubBackend(
        "m-0",
        {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    )
    assert list(backend.stream([{"role": "user", "content": "ping"}])) == ["tok-a", "tok-b"]
    assert backend.calls == 1
    assert backend.last_usage == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }


def test_enveloped_classifies_every_documented_shape() -> None:
    assert observe_audit._enveloped(_Resp(200, {"detail": "x", "code": "y"})) is False
    assert observe_audit._enveloped(_Resp(503, {"detail": "x", "code": "y"})) is False
    assert observe_audit._enveloped(_Resp(422, exc=ValueError("bad"))) is False
    assert observe_audit._enveloped(_Resp(422, ["nope"])) is False
    assert observe_audit._enveloped(_Resp(422, {"code": 1, "error": {"type": "api_error"}})) is True
    assert (
        observe_audit._enveloped(_Resp(422, {"type": "error", "error": {"message": "no"}})) is True
    )
    assert observe_audit._enveloped(_Resp(422, {"error": "nope", "type": "other"})) is False


async def _raw_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
    del scope
    message = await receive()
    assert message["type"] == "http.request"
    await send(
        {
            "type": "http.response.start",
            "status": 204,
            "headers": [(b"content-type", b"text/plain")],
        }
    )
    await send({"type": "http.response.body", "body": b"ok"})
    await send({"type": "http.response.trailers"})


def test_raw_http_reads_the_body_and_ignores_trailer_messages() -> None:
    seen = observe_audit._raw_http(_raw_app, "POST", "/metrics", [], body=b"{}")
    assert seen["status"] == 204
    assert seen["body"] == b"ok"


def test_prometheus_parser_rejects_comments_garbage_and_non_finite_values() -> None:
    commented = observe_audit._Prom("# HELP fx1_empty help text\n# TYPE fx1_empty gauge\n# note\n")
    assert commented.ordered() is True
    assert commented.bad_lines == []

    garbage = observe_audit._Prom("this is not a sample\n")
    assert garbage.bad_lines == ["this is not a sample"]
    assert garbage.all_typed() is False

    broken = observe_audit._Prom("fx1_bad 12abc\n")
    assert broken.finite() is False
    infinite = observe_audit._Prom("fx1_inf +Inf\n")
    assert infinite.finite() is False

    samples = observe_audit._Prom('fx1_c{job="1"} 2\n')
    assert samples.counter("missing") is None
    assert samples.counter("fx1_c", {"job": "2"}) is None
    assert samples.counter("fx1_c", {"job": "1"}) == 2.0


def test_wait_job_sleeps_once_and_times_out_without_spinning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(observe_audit.time, "sleep", lambda _seconds: None)
    poll = _JobPoll()
    assert observe_audit._wait_job(poll, "job-1", {}, timeout=5)["status"] == "succeeded"
    assert poll.calls == 2
    with pytest.raises(TimeoutError, match="did not finish"):
        observe_audit._wait_job(_JobPoll(), "job-1", {}, timeout=0)


def test_transport_adapter_deletes_and_posts_raw_bytes() -> None:
    client = _Transport()
    send = observe_audit._tc_transport(client)
    status, _headers, body = send("DELETE", "http://harness.test/x?y=1", None, {}, 1.0)
    assert status == 204
    assert body == b""
    status, _headers, body = send("POST", "http://harness.test/raw", b"raw", {}, 1.0)
    assert status == 201
    assert body == b"raw"
    assert client.calls == [("DELETE", "/x?y=1"), ("bytes", "/raw")]


def test_client_error_map_records_unexpected_and_missing_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics_calls = {"n": 0}
    orig_metrics = HarnessClient.metrics
    orig_json = HarnessClient._json

    def _metrics(self: HarnessClient) -> Any:
        metrics_calls["n"] += 1
        if metrics_calls["n"] > 1:
            raise RuntimeError("metrics-boom")
        return orig_metrics(self)

    def _json(self: HarnessClient, method: str, path: str, *args: Any, **kwargs: Any) -> Any:
        if path == "/nope":
            return {}
        return orig_json(self, method, path, *args, **kwargs)

    monkeypatch.setattr(HarnessClient, "metrics", _metrics)
    monkeypatch.setattr(HarnessClient, "_json", _json)
    with observe_audit._audit_resources():
        out = observe_audit._probe_client_surface()
    assert out["client_401_maps_auth_error"] is False
    assert out["client_404_maps_keyerror"] is False
    assert out["client_422_maps_valueerror"] is True
    assert out["client_ready_503_maps_backend_error"] is True


def test_main_guard_stops_before_the_full_bench(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stop(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("coverage-main-stop")

    monkeypatch.setattr("fx1.serve.api.create_app", _stop)
    with pytest.raises(RuntimeError, match="coverage-main-stop"):
        runpy.run_path(observe_audit.__file__, run_name="__main__")
