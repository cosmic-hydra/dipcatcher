"""Edges inside cap_audit that the all-green matrix never takes.

Poll timeouts, envelope parsers, upstream reply branches, and the
``if`` arms that skip a follow-up when a submit or fine-tune name is
absent. Happy-path ``cap_audit()`` stays untouched.
"""

from __future__ import annotations

import json
import os
import runpy
from typing import Any

import pytest

import fx1.serve.cap_audit as cap_audit
from fx1.serve.conv_audit import _audit_context

_TEXT = "stub-answer cap- [DONE] chat.completion.chunk"
_MODELS = {"data": [{"id": name} for name in ("fx1", "hosted_k3", "local_fx1", "byok")]}
_CHOICES = {"choices": [{"message": {"content": "stub-answer", "tool_calls": [{"id": "call_1"}]}}]}
_COUNTS = {"completed": 1, "failed": 1, "succeeded": 1, "errored": 1}
_OPENAI_ERR = {
    "error": {"message": "no", "type": "invalid_request_error", "code": "model_not_trainable"}
}


class _Resp:
    def __init__(
        self,
        status: int = 200,
        payload: Any = None,
        *,
        text: str = "",
        content: bytes = b"",
        exc: BaseException | None = None,
    ) -> None:
        self.status_code = status
        self._payload = {} if payload is None else payload
        self._exc = exc
        self.text = text
        self.content = content
        self.headers = {"content-type": "application/json"}

    def json(self) -> Any:
        if self._exc is not None:
            raise self._exc
        return self._payload


class _Script:
    """FIFO stand-in for the few calls a surface helper makes."""

    def __init__(self, responses: list[_Resp]) -> None:
        self._responses = list(responses)

    def post(self, url: str, **_kwargs: Any) -> _Resp:
        del url
        return self._responses.pop(0)

    def get(self, url: str, **_kwargs: Any) -> _Resp:
        del url
        return self._responses.pop(0)


class _Wire:
    """URL-dispatched client so a probe can run without booting an app."""

    def post(self, url: str, json: Any = None, **_kwargs: Any) -> _Resp:
        return _route("POST", url, json)

    def get(self, url: str, **_kwargs: Any) -> _Resp:
        return _route("GET", url, None)


class _Poll:
    def get(self, url: str, **_kwargs: Any) -> _Resp:
        if "/messages/batches/" in url:
            return _Resp(200, {"processing_status": "in_progress", "status": "running"})
        return _Resp(200, {"status": "running", "processing_status": "in_progress"})


class _Handler:
    def __init__(self, path: str) -> None:
        self.path = path
        self.wfile = _SinkFile()

    def send_response(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def send_header(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def end_headers(self) -> None:
        return None


class _SinkFile:
    def write(self, _data: bytes) -> None:
        return None

    def flush(self) -> None:
        return None


def _route(method: str, url: str, body: Any) -> _Resp:
    path = url.split("?", 1)[0]
    status = 200
    payload: dict[str, Any] = {}
    content = b""
    if path == "/v1/chat/completions":
        payload = _CHOICES
    elif path == "/v1/messages":
        payload = {"type": "message", "content": [{"type": "text", "text": "stub-answer"}]}
    elif path == "/v1/messages/count_tokens":
        payload = {"input_tokens": 5}
    elif path == "/v1/embeddings":
        payload = {"data": [{"index": 0}]}
    elif path == "/v1/models":
        payload = _MODELS
    elif path == "/v1/files" and method == "POST":
        payload = {"id": "file-1"}
    elif path.endswith("/content"):
        content = cap_audit._CORPUS
    elif path == "/v1/uploads" and method == "POST":
        payload = {"id": "up-1"}
    elif path.endswith("/parts"):
        payload = {"id": "part-1"}
    elif "uploads" in path and path.endswith("/complete"):
        payload = {"file": {"id": "file-2"}}
    elif path == "/v1/vector_stores":
        payload = {"id": "vs_1"}
    elif path == "/v1/conversations":
        payload = {"id": "conv_1"}
    elif path == "/v1/moderations":
        payload = {"results": []}
    elif path == "/harness/evals" and method == "POST":
        status, payload = 202, {"eval_id": "e1"}
    elif path.startswith("/harness/evals/"):
        payload = {"status": "succeeded", "error": "missing-backend"}
    elif path == "/v1/evals" and method == "POST":
        status, payload = 201, {"id": "ev1"}
    elif path.startswith("/v1/evals/") and path.endswith("/runs") and method == "POST":
        status, payload = 201, {"id": "run1"}
    elif "/runs/" in path:
        payload = {"status": "completed"}
    elif path == "/v1/batches" and method == "POST":
        payload = {"id": "batch1"}
    elif path.startswith("/v1/batches/"):
        payload = {"status": "completed", "request_counts": dict(_COUNTS)}
    elif path == "/v1/messages/batches" and method == "POST":
        payload = {"id": "ab1"}
    elif path.startswith("/v1/messages/batches/"):
        payload = {"processing_status": "ended", "request_counts": dict(_COUNTS)}
    elif path.startswith("/v1/fine_tuning/jobs"):
        if method == "POST":
            status, payload = 400, dict(_OPENAI_ERR)
        else:
            payload = {"status": "succeeded", "fine_tuned_model": "", "error": "x"}
    elif path == "/v1/responses" and method == "POST":
        if isinstance(body, dict) and body.get("background") is True:
            payload = {"status": "queued", "id": "resp1"}
        else:
            payload = {"status": "completed", "id": "resp1"}
    elif path.startswith("/v1/responses/"):
        payload = {"status": "completed", "id": "resp1"}
    elif path == "/harness/jobs" and method == "POST":
        status, payload = 202, {"job_id": "job1", "status": "queued"}
    elif path.startswith("/harness/jobs/"):
        payload = {"status": "succeeded"}
    elif path == "/harness/runs":
        payload = {"exit_code": 0}
    elif path == "/harness/score":
        payload = {"data": []}
    elif path == "/harness/gate/check":
        payload = {"ok": True}
    elif path == "/harness/keys" and method == "POST":
        status, payload = 201, {"key": "k", "id": "i"}
    elif path == "/harness/drain":
        payload = {"draining": True, "drained": True}
    return _Resp(status, payload, text=_TEXT, content=content)


def _bare_upstream(**overrides: Any) -> Any:
    upstream = cap_audit._CapUpstream.__new__(cap_audit._CapUpstream)
    upstream.delay_s = 0
    upstream.mode = "ok"
    upstream.requests = []
    upstream.answer = "stub-answer"
    upstream.resp_model = "stub-upstream-model"
    upstream.usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    for key, value in overrides.items():
        setattr(upstream, key, value)
    return upstream


def _install_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cap_audit,
        "_client",
        lambda _resolver, **_kwargs: (_Wire(), object()),
    )
    monkeypatch.setattr(
        cap_audit,
        "_ft_job",
        lambda _client, suffix="cap": ({"status": "succeeded"}, ""),
    )


def test_tool_stub_keeps_default_name_without_a_function_dict() -> None:
    stub = cap_audit._CapStub()
    unnamed = stub.complete_with_tools([{"role": "user", "content": "q"}])
    mistyped = stub.complete_with_tools(
        [{"role": "user", "content": "q"}],
        tools=[{"type": "function", "function": "calc"}],
    )
    assert unnamed.tool_calls[0]["function"]["name"] == "calc"
    assert mistyped.tool_calls[0]["function"]["name"] == "calc"


def test_upstream_reply_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(cap_audit.time, "sleep", lambda seconds: slept.append(seconds))

    delayed = _bare_upstream(delay_s=0.25, requests=[])
    delayed._respond(_Handler("/v1/not-a-route"))
    assert slept == [0.25]

    broken = _bare_upstream(requests=[{"body": b"{"}])
    broken._respond(_Handler("/v1/not-a-route"))

    listed = _bare_upstream(requests=[{"body": b"[1, 2]"}])
    listed._respond(_Handler("/v1/not-a-route"))

    def _frames(stop_at: int) -> Any:
        seen = {"n": 0}

        def _frame(_handler: Any, _payload: bytes) -> bool:
            seen["n"] += 1
            return seen["n"] < stop_at

        upstream = _bare_upstream(requests=[{"body": json.dumps({"stream": True}).encode()}])
        upstream._sse_frame = _frame
        upstream._respond(_Handler("/v1/chat/completions"))
        return seen["n"]

    assert _frames(1) == 1
    assert _frames(3) == 3


def test_byok_factory_applies_timeout() -> None:
    class _Up:
        url = "http://127.0.0.1:9/v1"

    resolver = cap_audit._byok_resolver(_Up())
    backend = resolver(
        "byok",
        timeout_s=4,
        base_url="http://127.0.0.1:9/v1",
        api_key="k",
        model="m",
    )
    assert backend._timeout_s == 4


def test_client_restores_a_preexisting_api_key() -> None:
    with _audit_context():
        os.environ["FX1_API_KEY"] = "preexisting-dummy"
        client, _api = cap_audit._client(cap_audit._Resolver({}))
        try:
            assert os.environ.get("FX1_API_KEY") == "preexisting-dummy"
        finally:
            client.close()


def test_verdict_helpers_on_non_objects_and_bad_json() -> None:
    assert cap_audit._verdict(_Resp(200, [1, 2])) == "200"
    assert cap_audit._err(_Resp(exc=ValueError("bad"))) == {}
    assert (
        cap_audit._is_anthropic_envelope(_Resp(exc=json.JSONDecodeError("bad", "doc", 0))) is False
    )
    openai = _Resp(
        400,
        {"error": {"message": "no", "type": "invalid_request_error", "code": "bad"}},
    )
    assert cap_audit._enveloped(openai) is True
    assert cap_audit._enveloped(_Resp(exc=ValueError("bad"))) is False


def test_polls_raise_when_the_resource_never_terminates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cap_audit.time, "sleep", lambda _seconds: None)
    client = _Poll()
    for waiter in (
        cap_audit._wait_ft,
        cap_audit._wait_eval,
        cap_audit._wait_batch,
        cap_audit._wait_abatch,
        cap_audit._wait_bg,
        cap_audit._wait_job,
    ):
        with pytest.raises(AssertionError, match="never"):
            waiter(client, "id-1")
    with pytest.raises(AssertionError, match="never"):
        cap_audit._wait_v1_run(client, "eval-1", "run-1")


def test_eval_batch_and_state_helpers_skip_when_submit_fails() -> None:
    skipped = cap_audit._Probe("edge")
    cap_audit._eval_surfaces(
        skipped,
        _Script([_Resp(400, {}), _Resp(400, {})]),
        expect="failed",
        submit={},
        run={},
        vexpect="failed",
    )
    assert "edge__eval_terminal_failed" not in skipped.out
    assert "edge__v1eval_run_201" not in skipped.out

    partial = cap_audit._Probe("edge")
    cap_audit._eval_surfaces(
        partial,
        _Script([_Resp(400, {}), _Resp(201, {"id": "ev1"}), _Resp(400, {})]),
        expect="succeeded",
        submit={},
        run={},
        vexpect="completed",
    )
    assert partial.out["edge__v1eval_run_201"] is False
    assert "edge__v1eval_terminal_completed" not in partial.out

    batch = cap_audit._Probe("edge")
    cap_audit._batch_surfaces(
        batch,
        _Script([_Resp(200, {"id": "file-1"}), _Resp(400, {}), _Resp(400, {})]),
        lines_ok=True,
    )
    assert "edge__batch_terminal_honest" not in batch.out
    assert "edge__abatch_terminal_honest" not in batch.out

    state = cap_audit._Probe("edge")
    cap_audit._stateful_surfaces(state, _Script([_Resp(400, {}) for _ in range(5)]))
    assert "edge__files_get_200" not in state.out
    assert "edge__uploads__200" in state.out
    assert "edge__vs_attach_file" not in state.out
    assert "edge__conv_items_200" not in state.out

    background = cap_audit._Probe("edge")
    cap_audit._bg_surface(background, _Script([_Resp(503, {})]), expect="completed")
    assert background.out["edge__bg_submit_200"] is False
    assert "edge__bg_terminal_completed" not in background.out

    class _Ops:
        def get(self, url: str, **_kwargs: Any) -> _Resp:
            del url
            return _Resp(200, {})

        def post(self, url: str, **_kwargs: Any) -> _Resp:
            if url == cap_audit._P_JOBS:
                return _Resp(400, {})
            if url == cap_audit._P_RUNS:
                return _Resp(200, {"exit_code": 0})
            return _Resp(200, {})

    ops = cap_audit._Probe("edge")
    cap_audit._ops_surfaces(ops, _Ops())
    assert "edge__ops_jobs_terminal_succeeded" not in ops.out


def test_probes_skip_ft_followups_when_the_name_or_submit_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_wire(monkeypatch)
    with _audit_context():
        stub = cap_audit._probe_stub()
        plain = cap_audit._probe_plain()
        unconf = cap_audit._probe_unconf()
        byok = cap_audit._probe_byok()
        local = cap_audit._probe_local()
    assert stub.out["stub__ft_job_succeeded"] is True
    assert "stub__ft_serve__200" not in stub.out
    assert "plain__ft_serve__200" not in plain.out
    assert unconf.out["unconf__ft_submit__200"] is False
    assert "unconf__ft_job_failed_honest" not in unconf.out
    assert "unconf__ft_serve__503:backend_unavailable" not in unconf.out
    assert byok.out["byok__chat__200"] is True
    assert "byok__ft_job_failed_honest" not in byok.out
    assert "byok__ft_serve__503:backend_unavailable" not in byok.out
    assert "local__ft_serve__200" not in local.out


def test_main_guard_stops_before_the_full_bench(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stop(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("coverage-main-stop")

    monkeypatch.setattr("fx1.serve.conv_audit._audit_context", _stop)
    with pytest.raises(RuntimeError, match="coverage-main-stop"):
        runpy.run_path(cap_audit.__file__, run_name="__main__")
