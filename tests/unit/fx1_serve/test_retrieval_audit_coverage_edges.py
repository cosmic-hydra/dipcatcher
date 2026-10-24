"""Edge arms in retrieval_audit: waits, torn reads, and a pause that does not cancel."""

from __future__ import annotations

import json
import os
import threading
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fx1.serve import retrieval_audit as ret


@pytest.fixture(autouse=True)
def _resources() -> Any:
    with ret._audit_context():
        yield


_RELEASE = threading.Event()


class _Resp:
    def __init__(self, body: dict[str, Any], status: int = 200, content: bytes = b"") -> None:
        self.status_code = status
        self._body = body
        self.content = content or json.dumps(body).encode()
        self.text = self.content.decode(errors="replace")
        self.headers: dict[str, str] = {}

    def json(self) -> dict[str, Any]:
        return self._body


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


def test_client_pops_and_restores_a_previous_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ret._API_KEY_ENV, "retr-prev")
    client, _api = ret._client({"byok": lambda: ret._StubBackend()}, api_key=None)
    client.close()
    assert os.environ[ret._API_KEY_ENV] == "retr-prev"


def test_waits_sleep_then_return_the_open_record(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _mono() -> float:
        calls["n"] += 1
        return 1000.0 if calls["n"] % 3 == 0 else 0.0

    class _Client:
        def get(self, url: str, **_kwargs: Any) -> _Resp:
            del url
            return _Resp({"status": "running", "processing_status": "in_progress"})

    monkeypatch.setattr(ret.time, "monotonic", _mono)
    monkeypatch.setattr(ret.time, "sleep", lambda _seconds: None)
    client = _Client()
    assert ret._wait_eval(client, "eval-1", timeout=10).get("status") == "running"  # type: ignore[arg-type]
    assert ret._wait_ft(client, "ft-1", timeout=10).get("status") == "running"  # type: ignore[arg-type]
    assert ret._wait_job(client, "job-1", timeout=10).get("status") == "running"  # type: ignore[arg-type]
    assert ret._wait_abatch(client, "batch-1", timeout=10).get("processing_status") == "in_progress"  # type: ignore[arg-type]


def test_cursor_walk_raises_when_pages_do_not_end() -> None:
    class _Client:
        def get(self, url: str, **_kwargs: Any) -> _Resp:
            del url
            return _Resp({"data": [{"id": "a"}], "has_more": True, "last_id": "a"})

    with pytest.raises(AssertionError, match="did not terminate"):
        ret._walk_cursor(_Client(), "/v1/files", limit=1, max_pages=1)  # type: ignore[arg-type]


def test_checkpoint_absence_and_bad_json_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    real_get = TestClient.get

    def _get(self: TestClient, url: str, *args: Any, **kwargs: Any) -> Any:
        resp = real_get(self, url, *args, **kwargs)
        if "/checkpoints" in url:
            return _Resp({"data": []})
        if url.endswith("/results"):
            return _Resp({}, content=b"not-json\n")
        return resp

    real_stream = TestClient.stream

    class _ShortStream:
        def __init__(self, inner: Any, kind: str) -> None:
            self._inner = inner
            self._kind = kind
            self.status_code = 200

        def __enter__(self) -> _ShortStream:
            self._resp = self._inner.__enter__()
            self.status_code = self._resp.status_code
            return self

        def __exit__(self, *args: Any) -> Any:
            return self._inner.__exit__(*args)

        def iter_text(self) -> Any:
            if self._kind == "response":
                yield "id: 1\nevent: response.completed\n\n"
                return
            if self._kind == "job":
                # One bad frame, then two non-terminal snapshots so the loop
                # both catches JSONDecodeError and continues.
                yield 'data: {bad\n\ndata: {"status": "running"}\n\ndata: {"status": "running"}\n\n'
                return
            yield from self._resp.iter_text()

    def _stream(self: TestClient, method: str, url: str, *args: Any, **kwargs: Any) -> Any:
        inner = real_stream(self, method, url, *args, **kwargs)
        kind = ""
        if "/v1/responses/" in str(url):
            kind = "response"
        elif "/events" in str(url):
            kind = "job"
        return _ShortStream(inner, kind)

    monkeypatch.setattr(TestClient, "get", _get)
    monkeypatch.setattr(TestClient, "stream", _stream)
    deleted = ret._deletion_boundary_probes()
    assert deleted["ft_deleted_model_drops_checkpoint"] is False
    streamed = ret._stream_probes()
    assert "results_jsonl_eof_terminated" in streamed
    assert streamed["response_replay_starting_after_resumes"] is True


def test_pause_gate_false_emits_the_second_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    real_is_set = threading.Event.is_set

    def _is_set(self: threading.Event) -> bool:
        frame = __import__("sys")._getframe(1)
        names: list[str] = []
        cursor = frame
        while cursor is not None and len(names) < 12:
            names.append(cursor.f_code.co_name)
            cursor = cursor.f_back
        if "_pause_gate" in names and "pausable_runner" in names:
            return False
        return real_is_set(self)

    monkeypatch.setattr(threading.Event, "is_set", _is_set)
    out = ret._lifecycle_probes()
    assert "ft_cancelled_job_events_honest_tail" in out


def test_concurrent_readers_see_torn_rows_and_time_out(monkeypatch: pytest.MonkeyPatch) -> None:
    _RELEASE.clear()
    real_post = TestClient.post
    real_get = TestClient.get
    real_sleep = ret.time.sleep

    def _sleep(seconds: float) -> None:
        if seconds >= 0.5:
            real_sleep(0.05)
        elif seconds <= 0.01:
            real_sleep(seconds)

    def _post(self: TestClient, url: str, *args: Any, **kwargs: Any) -> Any:
        if "/conversations/" in url and url.endswith("/items"):
            _RELEASE.wait(5)
            return _Resp({}, status=500)
        return real_post(self, url, *args, **kwargs)

    def _get(self: TestClient, url: str, *args: Any, **kwargs: Any) -> Any:
        if "/conversations/" in url and url.endswith("/items"):
            return _Resp(
                {
                    "object": "nope",
                    "data": [{"id": "", "type": "", "content": None}],
                    "first_id": "x",
                    "last_id": "y",
                }
            )
        if url.endswith("/events"):
            _RELEASE.wait(5)
            return _Resp({"data": [{"id": "not-an-event"}]})
        return real_get(self, url, *args, **kwargs)

    monkeypatch.setattr(ret.time, "sleep", _sleep)
    monkeypatch.setattr(TestClient, "post", _post)
    monkeypatch.setattr(TestClient, "get", _get)
    monkeypatch.setattr(threading.Thread, "join", lambda self, timeout=None: None)
    try:
        out = ret._concurrent_read_probes()
    finally:
        _RELEASE.set()
    assert out["conv_items_concurrent_no_torn_reads"] is False
    assert out["ft_events_concurrent_consistent_snapshots"] is False


def test_main_prints_stubbed_bench() -> None:
    _exec_main(ret, "retrieval_audit_bench")
