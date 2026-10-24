"""Edge arms in e2e_audit: stub engine faults, absent receipts, and retry bookkeeping."""

from __future__ import annotations

import contextlib
import threading
from http.server import ThreadingHTTPServer
from typing import Any
from urllib import request as urlrequest
from urllib.error import HTTPError

import pytest

from fx1.serve import api as api_mod
from fx1.serve import e2e_audit as e2e
from fx1.serve.client import HarnessClient
from fx1.serve.receipt_store import ReceiptIndex


def _request(url: str, data: bytes | None = None, headers: dict[str, str] | None = None) -> int:
    method = "POST" if data is not None else "GET"
    req = urlrequest.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urlrequest.urlopen(req, timeout=2) as resp:  # noqa: S310 — loopback stub
            return int(resp.status)
    except HTTPError as exc:
        return int(exc.code)


def test_stub_engine_covers_get_post_and_hostile_input() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), e2e._StubChat)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    base = f"http://127.0.0.1:{port}"
    try:
        models = _request(f"{base}/v1/models")
        missing = _request(f"{base}/nope")
        post_missing = _request(f"{base}/v1/other", b"{}")
        hostile = _request(
            f"{base}/v1/chat/completions",
            b"not-json",
            {"Content-Type": "application/json"},
        )
        status, _headers = e2e._raw_headers(f"{base}/nope", None)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert models == 200
    assert missing == 404
    assert post_missing == 404
    assert hostile == 400
    assert status == 404


def test_cleanup_skips_servers_that_never_started(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("create_app refused")

    monkeypatch.setattr(api_mod, "create_app", _boom)
    with pytest.raises(RuntimeError, match="create_app refused"):
        e2e.e2e_audit()


def _instant_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(e2e.time, "sleep", lambda _seconds: None)
    import fx1.serve.client as client_mod

    monkeypatch.setattr(client_mod.time, "sleep", lambda _seconds: None)


def test_audit_records_absent_receipt_and_unraised_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One pass where the resilient complete raises before any retry sleep."""
    _instant_sleep(monkeypatch)
    real_exists = e2e.Path.exists

    def _exists(self: e2e.Path) -> bool:
        if self.name == "fx1_parity_audit.json":
            return False
        return real_exists(self)

    monkeypatch.setattr(e2e.Path, "exists", _exists)
    monkeypatch.setattr(ReceiptIndex, "available", lambda self: False)
    real_complete = HarnessClient.complete
    real_run = HarnessClient.run
    real_submit = HarnessClient.submit_run
    real_status = HarnessClient.job_status
    real_metrics = HarnessClient.metrics
    drained: set[int] = set()
    status_seen: set[tuple[int, str]] = set()
    metrics_seen = {"n": 0}
    real_drain = HarnessClient.drain

    def _drain(self: HarnessClient, *args: Any, **kwargs: Any) -> Any:
        drained.add(id(self))
        return real_drain(self, *args, **kwargs)

    def _complete(self: HarnessClient, *args: Any, **kwargs: Any) -> Any:
        if self._max_retries >= 3:
            raise RuntimeError("retry client refused before sleep")
        return real_complete(self, *args, **kwargs)

    def _run(self: HarnessClient, command: str, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("idempotency_key") is None and self._max_retries >= 3:
            return type("Run", (), {"command": command, "exit_code": 0, "ok": True})()
        return real_run(self, command, *args, **kwargs)

    def _submit(self: HarnessClient, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("idempotency_key") is None and id(self) in drained:
            return "job-synthetic"
        return real_submit(self, *args, **kwargs)

    def _status(self: HarnessClient, job_id: str, *args: Any, **kwargs: Any) -> Any:
        body = real_status(self, job_id, *args, **kwargs)
        key = (id(self), job_id)
        if (
            isinstance(body, dict)
            and body.get("status") not in {"queued", None}
            and key not in status_seen
        ):
            status_seen.add(key)
            return {**body, "status": "queued"}
        return body

    def _metrics(self: HarnessClient, *args: Any, **kwargs: Any) -> Any:
        metrics_seen["n"] += 1
        snap = real_metrics(self, *args, **kwargs)
        if metrics_seen["n"] == 4 and getattr(snap, "rate_limited_total", 0) >= 1:
            clone = (
                type(snap).model_validate(snap.model_dump())
                if hasattr(snap, "model_dump")
                else snap
            )
            if hasattr(clone, "rate_limited_total"):
                try:
                    object.__setattr__(clone, "rate_limited_total", 0)
                except (AttributeError, TypeError):
                    clone = type("Snap", (), {**snap.__dict__, "rate_limited_total": 0})()
            return clone
        return snap

    monkeypatch.setattr(HarnessClient, "drain", _drain)
    monkeypatch.setattr(HarnessClient, "complete", _complete)
    monkeypatch.setattr(HarnessClient, "run", _run)
    monkeypatch.setattr(HarnessClient, "submit_run", _submit)
    monkeypatch.setattr(HarnessClient, "job_status", _status)
    monkeypatch.setattr(HarnessClient, "metrics", _metrics)
    out = e2e.e2e_audit()
    assert out["e2e_verify_sealed"] is False
    assert out["e2e_verify_tamper"] is False
    assert out["e2e_retry_recovers_under_cap"] is False
    assert out["e2e_drain_blocks_new_work"] is False
    assert out["e2e_error_code_draining"] is False


def test_retry_sleep_runs_again_after_the_slot_stays_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second 503 after the release hook already ran takes the other branch."""
    _instant_sleep(monkeypatch)
    hold = {"on": False}
    sems: list[Any] = []
    real_acquire = threading.BoundedSemaphore.acquire
    real_release = threading.BoundedSemaphore.release

    def _acquire(
        self: threading.BoundedSemaphore, blocking: bool = True, timeout: float | None = None
    ) -> bool:
        if getattr(self, "_cap1", False) and blocking is False:
            sems.append(self)
        return bool(real_acquire(self, blocking, timeout))

    def _release(self: threading.BoundedSemaphore) -> None:
        if hold["on"] and getattr(self, "_cap1", False):
            return
        real_release(self)

    monkeypatch.setattr(threading.BoundedSemaphore, "acquire", _acquire)
    monkeypatch.setattr(threading.BoundedSemaphore, "release", _release)
    real_create = api_mod.create_app

    def _create(*args: Any, **kwargs: Any) -> Any:
        app = real_create(*args, **kwargs)
        if kwargs.get("max_inflight") == 1:
            app.state.inflight_slots._cap1 = True  # type: ignore[attr-defined]
            hold["on"] = True
        return app

    monkeypatch.setattr(api_mod, "create_app", _create)
    real_complete = HarnessClient.complete
    real_run = HarnessClient.run

    def _complete(self: HarnessClient, *args: Any, **kwargs: Any) -> Any:
        try:
            return real_complete(self, *args, **kwargs)
        finally:
            if self._max_retries >= 3:
                hold["on"] = False
                for sem in sems:
                    with contextlib.suppress(ValueError):
                        real_release(sem)

    def _run(self: HarnessClient, command: str, *args: Any, **kwargs: Any) -> Any:
        if self._max_retries >= 3:
            return type("Run", (), {"command": command, "exit_code": 0, "ok": True})()
        return real_run(self, command, *args, **kwargs)

    monkeypatch.setattr(HarnessClient, "complete", _complete)
    monkeypatch.setattr(HarnessClient, "run", _run)
    out = e2e.e2e_audit()
    assert out["e2e_retry_recovers_under_cap"] is False
