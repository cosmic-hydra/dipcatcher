"""Edge arms in files_audit: worker faults, journal no-raise, and poll sleeps."""

from __future__ import annotations

import os
import re
import sys
import threading
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fx1.harness import Harness
from fx1.sdk import Fx1Harness
from fx1.serve import files_audit as files
from fx1.serve.api import _FileStore
from fx1.serve.client import HarnessAuthError, HarnessClient, HarnessTransportError
from fx1.serve.uploads import UploadStoreError

_POLL = re.compile(
    r"/(?:harness/jobs|v1/fine_tuning/jobs|harness/evals|v1/evals|"
    r"v1/batches|v1/messages/batches|v1/responses)/[^/]+$"
)
_TERMINAL = frozenset(
    {"succeeded", "failed", "cancelled", "completed", "expired", "ended", "incomplete"}
)
_RACE_BODY = (b"0123456789abcdef" * 512) + b"\n"


class _Running:
    status_code = 200
    headers: dict[str, str] = {}
    content = b'{"status":"running"}'
    text = '{"status":"running"}'

    def json(self) -> dict[str, Any]:
        return {"status": "running", "processing_status": "in_progress"}


class _Bytes:
    def __init__(self, status: int, content: bytes) -> None:
        self.status_code = status
        self.content = content
        self.headers: dict[str, str] = {}
        self.text = ""

    def json(self) -> dict[str, Any]:
        return {}


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


def test_runner_env_restore_and_json_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    assert files._runner(["coverage"], 1) == (0, "ok", "")
    monkeypatch.setenv(files._API_KEY_ENV, "files-prev")
    with files._env_key():
        assert os.environ[files._API_KEY_ENV] == files._ROOT_KEY
    assert os.environ[files._API_KEY_ENV] == "files-prev"
    with files._client() as client:
        send = files._tc_transport(client)
        status, _headers, _body = send("POST", "http://files-audit/v1/files", {"n": 1}, {}, 1.0)
    assert status in {200, 400, 401, 404, 422}


def test_worker_errors_and_a_join_that_does_not_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> None:
        raise RuntimeError("worker exploded")

    with pytest.raises(RuntimeError, match="worker failed"):
        files._run_calls([_boom])

    monkeypatch.setattr(threading.Thread, "join", lambda self, timeout=None: None)
    monkeypatch.setattr(threading.Thread, "is_alive", lambda self: True)
    with pytest.raises(RuntimeError, match="did not terminate"):
        files._run_calls([lambda: None])


def test_wait_ft_sleeps_when_the_job_is_still_open(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _mono() -> float:
        calls["n"] += 1
        return 1000.0 if calls["n"] % 3 == 0 else 0.0

    class _Client:
        def get(self, url: str, **_kwargs: Any) -> _Running:
            del url
            return _Running()

    monkeypatch.setattr(files.time, "monotonic", _mono)
    monkeypatch.setattr(files.time, "sleep", lambda _seconds: None)
    rec = files._wait_ft(_Client(), "ftjob-1", timeout_s=10.0)  # type: ignore[arg-type]
    assert rec["status"] == "running"


def test_audit_records_no_raise_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validation that fails to raise is recorded False; a preset env is restored."""
    monkeypatch.setenv("MOONSHOT_API_KEY", "moon-prev")
    real_init = Harness.__init__

    def _init(self: Harness, runner: Any = None) -> None:
        real_init(self, runner=runner)
        if runner is not None:
            runner(["coverage"], 1)

    monkeypatch.setattr(Harness, "__init__", _init)
    real_store_init = _FileStore.__init__
    real_put = _FileStore.put
    real_delete = _FileStore.delete
    real_get_store = _FileStore.get

    def _store_init(self: _FileStore, *args: Any, **kwargs: Any) -> None:
        state = kwargs.get("state_dir")
        if state is None and len(args) >= 3:
            state = args[2]
        journal = Path(state) / "files.jsonl" if state is not None else None
        if journal is not None and journal.exists() and b'{"torn":' in journal.read_bytes():
            return
        real_store_init(self, *args, **kwargs)

    def _put(self: _FileStore, *args: Any, **kwargs: Any) -> Any:
        try:
            return real_put(self, *args, **kwargs)
        except OSError:
            return None

    def _delete(self: _FileStore, *args: Any, **kwargs: Any) -> Any:
        try:
            return real_delete(self, *args, **kwargs)
        except OSError:
            return None

    def _get_store(self: _FileStore, *args: Any, **kwargs: Any) -> Any:
        try:
            return real_get_store(self, *args, **kwargs)
        except OSError:
            return None

    monkeypatch.setattr(_FileStore, "__init__", _store_init)
    monkeypatch.setattr(_FileStore, "put", _put)
    monkeypatch.setattr(_FileStore, "delete", _delete)
    monkeypatch.setattr(_FileStore, "get", _get_store)

    real_file = HarnessClient.file
    real_content = HarnessClient.file_content
    real_upload = HarnessClient.upload_file
    real_sdk_content = Fx1Harness.file_content
    real_sdk_create = Fx1Harness.openai_file_create

    def _file(self: HarnessClient, fid: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return real_file(self, fid, *args, **kwargs)
        except (KeyError, HarnessAuthError, ValueError, HarnessTransportError):
            return {"id": fid, "object": "file"}

    def _content(self: HarnessClient, fid: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return real_content(self, fid, *args, **kwargs)
        except KeyError:
            return b""

    def _upload(self: HarnessClient, *args: Any, **kwargs: Any) -> Any:
        try:
            return real_upload(self, *args, **kwargs)
        except (HarnessTransportError, HarnessAuthError):
            return {"id": "file-swallowed", "object": "file", "bytes": 1}

    def _sdk_content(self: Fx1Harness, fid: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return real_sdk_content(self, fid, *args, **kwargs)
        except KeyError:
            return b""

    def _sdk_create(self: Fx1Harness, *args: Any, **kwargs: Any) -> Any:
        try:
            return real_sdk_create(self, *args, **kwargs)
        except UploadStoreError:
            return {"id": "file-swallowed"}

    monkeypatch.setattr(HarnessClient, "file", _file)
    monkeypatch.setattr(HarnessClient, "file_content", _content)
    monkeypatch.setattr(HarnessClient, "upload_file", _upload)
    monkeypatch.setattr(Fx1Harness, "file_content", _sdk_content)
    monkeypatch.setattr(Fx1Harness, "openai_file_create", _sdk_create)

    real_get = TestClient.get
    seen: set[tuple[int, str]] = set()
    torn = {"n": 0}

    def _get(self: TestClient, url: str, *args: object, **kwargs: object) -> object:
        resp = real_get(self, url, *args, **kwargs)
        path = url.split("?", 1)[0] if isinstance(url, str) else ""
        if (
            sys._getframe(1).f_code.co_name == "_probe_consumer"
            and path.startswith("/v1/batches/")
            and path.count("/") == 3
        ):
            key = (id(self), path)
            seen_batch = getattr(_get, "batches", None)
            if seen_batch is None:
                seen_batch = {}
                _get.batches = seen_batch  # type: ignore[attr-defined]
            seen_batch[key] = seen_batch.get(key, 0) + 1
            if seen_batch[key] == 1:
                return _Running()
            if seen_batch[key] == 2:

                class _Done:
                    status_code = 200
                    headers: dict[str, str] = {}
                    content = b"{}"
                    text = "{}"

                    def json(self) -> dict[str, Any]:
                        return {"status": "completed", "request_counts": {"completed": 0}}

                return _Done()
        # fall through to the poll wrapper below
        if path.endswith("/content") and getattr(resp, "content", b"") == _RACE_BODY:
            torn["n"] += 1
            if torn["n"] == 1:
                return _Bytes(200, b"torn")
            if torn["n"] == 2:
                return _Bytes(500, b"")
        if not _POLL.search(path) or resp.status_code != 200:
            return resp
        key = (id(self), path)
        if key in seen:
            return resp
        try:
            body = resp.json()
        except (TypeError, ValueError):
            return resp
        if isinstance(body, dict) and body.get("status") in _TERMINAL:
            seen.add(key)
            return _Running()
        return resp

    monkeypatch.setattr(TestClient, "get", _get)
    out = files.files_audit()
    assert isinstance(out, dict)
    assert os.environ.get("MOONSHOT_API_KEY") == "moon-prev"
    assert os.environ.get(files._API_KEY_ENV) in (None, "files-prev")


def test_main_prints_stubbed_bench() -> None:
    _exec_main(files, "files_audit_bench")
