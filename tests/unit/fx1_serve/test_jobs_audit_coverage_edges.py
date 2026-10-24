"""Edges inside jobs_audit that the all-green battery never takes.

The default runner, poll deadlines, cancel/status loops that return
before the first sleep, and the client ``try`` arms that record a probe
as False. Happy-path ``jobs_audit()`` stays untouched.
"""

from __future__ import annotations

import json
import os
import re
import runpy
import sys
import threading
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

import fx1.serve.jobs_audit as jobs_audit
from fx1.harness import HarnessResult
from fx1.serve.client import HarnessClient
from fx1.serve.conv_audit import _RESOURCES, _audit_context

_JOB = re.compile(r"/harness/jobs/[^/]+$")
_YIELD = time.sleep


class _EnteredOnce(threading.Event):
    """First ``is_set`` is false so a poll body runs once, then true."""

    def __init__(self) -> None:
        super().__init__()
        self._polls = 0

    def is_set(self) -> bool:
        self._polls += 1
        return self._polls != 1


class _Plain:
    headers = {"content-type": "text/plain"}

    def json(self) -> dict[str, Any]:
        raise AssertionError("non-json body was parsed")


class _Delete:
    status_code = 418
    headers: dict[str, str] = {}
    text = "{}"

    def json(self) -> dict[str, Any]:
        return {"status": "weird", "finished_at": None}


class _View:
    def __init__(self, response: Any, payload: dict[str, Any]) -> None:
        self.status_code = response.status_code
        self.headers = response.headers
        self.text = response.text
        self.content = response.content
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class _Lines:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __enter__(self) -> _Lines:
        return self

    def __exit__(self, *_args: Any) -> bool:
        return False

    def iter_lines(self) -> Any:
        yield from self._lines


class _ClosedStream:
    def __enter__(self) -> Any:
        raise RuntimeError("stream-closed")

    def __exit__(self, *_args: Any) -> bool:
        return False


def _install_entered_once(monkeypatch: pytest.MonkeyPatch) -> None:
    orig = jobs_audit._Runner.__init__

    def _init(self: Any, gate: Any = None, **kwargs: Any) -> None:
        orig(self, gate, **kwargs)
        self.entered = _EnteredOnce()

    monkeypatch.setattr(jobs_audit._Runner, "__init__", _init)


def _yield_sleep(_seconds: float) -> None:
    _YIELD(0)


def _shutdown(client: Any) -> None:
    client.app.state.jobs_executor.shutdown(wait=True, cancel_futures=True)
    client.close()


def test_default_runner_outside_the_resource_stack() -> None:
    assert _RESOURCES.get(None) is None
    saved = os.environ.get("FX1_API_KEY")
    client = None
    try:
        client = jobs_audit._client()
        created = jobs_audit._submit(client)
        assert created.status_code == 202
        record = jobs_audit._wait(client, created.json()["job_id"], timeout=5)
        assert record["status"] == "succeeded"
        assert record["result"]["stdout"] == "ok"
        assert record["result"]["exit_code"] == 0
    finally:
        if client is not None:
            _shutdown(client)
        if saved is None:
            os.environ.pop("FX1_API_KEY", None)
        else:
            os.environ["FX1_API_KEY"] = saved


def test_err_code_ignores_a_non_json_body() -> None:
    assert jobs_audit._err_code(_Plain()) is None


def test_wait_returns_the_empty_record_when_the_deadline_has_passed() -> None:
    assert jobs_audit._wait(object(), "missing", timeout=0) == {}


def test_wait_hits_sleeps_until_the_sink_grows(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Sink:
        def __init__(self) -> None:
            self.hits: list[str] = []

    sink = _Sink()

    def _sleep(_seconds: float) -> None:
        sink.hits.append("hit")

    monkeypatch.setattr(jobs_audit.time, "sleep", _sleep)
    jobs_audit._wait_hits(sink, 1, timeout=5)
    assert sink.hits == ["hit"]


def test_lifecycle_reads_status_before_the_runner_is_marked_entered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entered_once(monkeypatch)
    with _audit_context():
        out = jobs_audit._submit_lifecycle_probes()
    assert out["lifecycle_queued_to_succeeded"] is True
    assert out["status_never_regresses"] is True


def test_config_escape_that_is_not_accepted_skips_the_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_post = TestClient.post

    def _post(self: TestClient, url: str, *args: Any, **kwargs: Any) -> Any:
        body = kwargs.get("json")
        if isinstance(body, dict) and body.get("config") == "/etc/hostname":
            return _Delete()
        return real_post(self, url, *args, **kwargs)

    monkeypatch.setattr(TestClient, "post", _post)
    with _audit_context():
        out = jobs_audit._validation_probes()
    assert out["config_escape_submit_accepted"] is False
    assert "config_escape_fails_closed" not in out


def test_transition_treats_a_nonstandard_cancel_as_neither_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entered_once(monkeypatch)
    monkeypatch.setattr(jobs_audit.time, "sleep", _yield_sleep)
    monkeypatch.setattr(jobs_audit, "_busy_executor", lambda *_a, **_k: None)

    def _delete(self: TestClient, url: str, *args: Any, **kwargs: Any) -> _Delete:
        del self, url, args, kwargs
        return _Delete()

    monkeypatch.setattr(TestClient, "delete", _delete)
    with _audit_context():
        out = jobs_audit._transition_probes()
    assert out["cancel_race_outcome_honest"] is False
    assert isinstance(out["cancel_running_409"], bool)


def test_gate_polls_once_before_the_runner_is_marked_entered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entered_once(monkeypatch)
    with _audit_context():
        out = jobs_audit._gate_probes()
    assert "over_capacity_503" in out
    assert "drain_200_latches" in out


def test_webhook_bookkeeping_and_callback_polls_take_both_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force the delivered-callback re-read, the 5xx deadline exit, and one
    4xx sleep followed by a deadline exit. Line numbers are the poll sites
    inside ``_webhook_probes``.
    """

    real_get = TestClient.get
    real_mono = time.monotonic
    blanked: set[str] = set()
    stripped: set[str] = set()
    polls = {"841": 0}

    def _mono() -> float:
        frame = sys._getframe(1)
        if frame.f_code.co_filename.endswith("jobs_audit.py"):
            if frame.f_lineno == 823:
                return 1e18
            if frame.f_lineno == 841:
                polls["841"] += 1
                if polls["841"] >= 2:
                    return 1e18
        return real_mono()

    def _get(self: TestClient, url: str, *args: Any, **kwargs: Any) -> Any:
        response = real_get(self, url, *args, **kwargs)
        path = str(url).split("?", 1)[0]
        if _JOB.fullmatch(path) is None or response.status_code != 200:
            return response
        try:
            body = response.json()
        except (TypeError, ValueError):
            return response
        if not isinstance(body, dict):
            return response
        outer = sys._getframe(2)
        if (
            outer.f_code.co_filename.endswith("jobs_audit.py")
            and outer.f_lineno == 842
            and path not in stripped
        ):
            stripped.add(path)
            cloned = dict(body)
            cloned["callback_status"] = None
            return _View(response, cloned)
        if (
            body.get("status") in jobs_audit._TERMINAL
            and body.get("callback_status") == "delivered"
            and path not in blanked
        ):
            blanked.add(path)
            cloned = dict(body)
            cloned["callback_status"] = None
            return _View(response, cloned)
        return response

    orig_busy = jobs_audit._busy_executor

    def _short_busy(client: Any, sleep_s: float = 1.5) -> None:
        del sleep_s
        orig_busy(client, 0.05)

    monkeypatch.setattr(jobs_audit.time, "sleep", _yield_sleep)
    monkeypatch.setattr(jobs_audit.time, "monotonic", _mono)
    monkeypatch.setattr(TestClient, "get", _get)
    monkeypatch.setattr(jobs_audit, "_busy_executor", _short_busy)
    with _audit_context():
        os.environ["FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"] = "1"
        out = jobs_audit._webhook_probes()
    assert out["callback_fires_once"] is True
    assert isinstance(out["callback_failure_marks_failed"], bool)
    assert isinstance(out["callback_4xx_single_attempt"], bool)
    assert polls["841"] >= 2


def test_batch_items_without_ids_are_not_waited(monkeypatch: pytest.MonkeyPatch) -> None:
    real_post = TestClient.post

    def _post(self: TestClient, url: str, *args: Any, **kwargs: Any) -> Any:
        response = real_post(self, url, *args, **kwargs)
        body = kwargs.get("json")
        jobs = body.get("jobs") if isinstance(body, dict) else None
        if not str(url).endswith("/harness/jobs/batch") or not isinstance(jobs, list):
            return response
        keys = [item.get("idempotency_key") for item in jobs if isinstance(item, dict)]
        try:
            payload = response.json()
        except (TypeError, ValueError):
            return response
        if not isinstance(payload, dict):
            return response
        cloned = json.loads(json.dumps(payload))
        rows = cloned.get("jobs")
        if not isinstance(rows, list):
            return response
        if keys == ["b-k1", "b-k2"]:
            rows.append({"index": 99})
            return _View(response, cloned)
        if keys == ["b-k1", "b-k3"] and len(rows) > 1 and isinstance(rows[1], dict):
            rows[1].pop("job_id", None)
            return _View(response, cloned)
        return response

    monkeypatch.setattr(TestClient, "post", _post)
    with _audit_context():
        out = jobs_audit._batch_probes()
    assert "batch_idem_replay_per_item" in out


@pytest.mark.parametrize("mode", ["quiet", "boom"])
def test_wait_stream_covers_partial_frames_and_unexpected_client_errors(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    _install_entered_once(monkeypatch)
    real_stream = TestClient.stream
    orig_wait = HarnessClient.wait_run
    waits = {"n": 0}

    def _stream(self: TestClient, method: str, url: str, *args: Any, **kwargs: Any) -> Any:
        if "timeout_s=10" in str(url):
            if mode == "boom":
                return _ClosedStream()
            return _Lines(['data: {"status": "running"}', 'data: {"status": "queued"}'])
        return real_stream(self, method, url, *args, **kwargs)

    def _wait(self: HarnessClient, job_id: str, **kwargs: Any) -> Any:
        waits["n"] += 1
        if waits["n"] == 1:
            return orig_wait(self, job_id, **kwargs)
        if mode == "quiet":
            return HarnessResult(command="doctor", exit_code=0, stdout="ok", stderr="")
        raise RuntimeError("wait-boom")

    monkeypatch.setattr(TestClient, "stream", _stream)
    monkeypatch.setattr(HarnessClient, "wait_run", _wait)
    with _audit_context():
        out = jobs_audit._wait_stream_probes()
    if mode == "quiet":
        assert out["client_wait_failed_raises"] is False
        assert out["client_wait_timeout_raises"] is False
    else:
        assert out["client_wait_failed_raises"] is False
        assert out["client_wait_timeout_raises"] is False
    assert out["sse_emits_terminal_frame"] is False


@pytest.mark.parametrize("mode", ["quiet", "boom"])
def test_client_probe_records_false_when_the_mapped_error_does_not_fire(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    orig_submit = HarnessClient.submit_run
    orig_status = HarnessClient.job_status
    submits = {"n": 0}

    def _submit(self: HarnessClient, command: str, extra_args: Any = None, **kwargs: Any) -> Any:
        if self._headers.get("X-API-Key") == "bogus":
            if mode == "quiet":
                return "job-ignored"
            raise RuntimeError("auth-map")
        submits["n"] += 1
        if submits["n"] == 2:
            if mode == "quiet":
                return "job-ignored"
            raise RuntimeError("drain-map")
        return orig_submit(self, command, extra_args, **kwargs)

    def _status(self: HarnessClient, job_id: str, *args: Any, **kwargs: Any) -> Any:
        if job_id == "never-existed":
            if mode == "quiet":
                return {"job_id": job_id}
            raise RuntimeError("missing-map")
        return orig_status(self, job_id, *args, **kwargs)

    monkeypatch.setattr(HarnessClient, "submit_run", _submit)
    monkeypatch.setattr(HarnessClient, "job_status", _status)
    monkeypatch.setattr(HarnessClient, "cancel_job", lambda _self, _job_id: {"status": "cancelled"})
    monkeypatch.setattr(jobs_audit, "_busy_executor", lambda *_a, **_k: None)
    with _audit_context():
        out = jobs_audit._client_probes()
    if mode == "quiet":
        assert out["client_auth_maps"] is False
        assert out["client_404_maps_keyerror"] is False
        assert out["client_503_maps_backend_not_configured"] is False
    else:
        assert out["client_auth_maps"] is False
        assert out["client_404_maps_keyerror"] is False
        assert out["client_503_maps_backend_not_configured"] is False
    assert out["client_cancel_job"] is True


def test_main_guard_stops_before_the_full_bench(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stop(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("coverage-main-stop")

    monkeypatch.setattr("fx1.serve.conv_audit._audit_context", _stop)
    with pytest.raises(RuntimeError, match="coverage-main-stop"):
        runpy.run_path(jobs_audit.__file__, run_name="__main__")
