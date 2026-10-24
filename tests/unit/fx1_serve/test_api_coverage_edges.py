"""Line and branch edges for fx1.serve.api left open by the fx1 run."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from starlette.requests import Request

from fx1.harness import Harness
from fx1.serve import api as api
from fx1.serve.api import (
    ApiError,
    ApiKeyPatchRequest,
    ApiKeyRotateRequest,
    ChatMessage,
    CompleteRequest,
    CompletionRecord,
    EvalRunCreate,
    HarnessRunRequest,
    JobStatusResponse,
    ScoreRequest,
    _AnthropicBatchRecord,
    _AnthropicBatchStore,
    _BatchRecord,
    _BatchStore,
    _breaker_key,
    _charge_key_tokens,
    _declare_response_headers,
    _FileStore,
    _grace_pipe,
    _idem_lookup,
    _IdemStore,
    _insufficient_scope,
    _JobStore,
    _JsonIdemRecord,
    _make_lifespan,
    _RateLimiter,
    _redact_credential_inputs,
    _request_ingress_refusal,
    _request_or_422,
    _require_admin,
    _resolve_auth,
    _responses_replay_frames,
    _sampling_extras_valid,
    _stored_receipt_matches,
    _submit_job,
)
from fx1.serve.evals import EvalRecord, EvalStore
from fx1.serve.finetune import FTJob, FTJobStore
from fx1.serve.journal import JobJournal
from fx1.serve.keys import ApiKeyStore
from fx1.serve.openai_compat import ByokOverride

_FID = "file-" + "ab" * 16
_SHA = "cd" * 32


def _http(
    method: str,
    path: str,
    *,
    headers: list[tuple[bytes, bytes]] | None = None,
    body: bytes = b"",
) -> Request:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers or [],
        "client": ("127.0.0.1", 9),
        "server": ("test", 80),
    }

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def _validation(model_cls: type[Any], **kwargs: Any) -> ValidationError:
    with pytest.raises(ValidationError) as exc:
        model_cls(**kwargs)
    return exc.value


def _job(status: str = "queued") -> JobStatusResponse:
    return JobStatusResponse(
        job_id="job-1",
        status=status,  # type: ignore[arg-type]
        created_at=1.0,
        finished_at=None,
        result=None,
        error=None,
    )


def _batch(status: str = "completed") -> _BatchRecord:
    return _BatchRecord(
        batch_id="batch_1",
        input_file_id="file-1",
        endpoint="/v1/chat/completions",
        completion_window="24h",
        status=status,
        created_at=1,
        expires_at=2,
    )


def _abatch(status: str = "ended") -> _AnthropicBatchRecord:
    return _AnthropicBatchRecord(
        batch_id="msgbatch_1",
        status=status,  # type: ignore[arg-type]
        created_at=1,
        expires_at=2,
    )


def _route(app: Any, name: str) -> Any:
    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        if getattr(endpoint, "__name__", "") == name:
            return endpoint
    raise AssertionError(name)


def _closure(fn: Any, kind: type[Any]) -> Any:
    for cell in fn.__closure__ or ():
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if isinstance(value, kind):
            return value
    raise AssertionError(kind)


def test_declared_headers_skip_non_objects_and_stamp_retry_after() -> None:
    seen: dict[str, Any] = {}

    class _App:
        def openapi(self) -> dict[str, Any]:
            spec = {
                "paths": {
                    "/x": {
                        "get": "not-an-operation",
                        "post": {
                            "responses": {"200": "not-an-object", "202": {}, "429": {}},
                            "operationId": "submit_job",
                        },
                    }
                }
            }
            seen["spec"] = spec
            return spec

    _declare_response_headers(_App(), rate_limited=True)  # type: ignore[arg-type]
    responses = seen["spec"]["paths"]["/x"]["post"]["responses"]
    assert "Retry-After" in responses["429"]["headers"]
    assert "Location" in responses["202"]["headers"]


def test_content_length_mismatch_is_refused() -> None:
    request = _http(
        "POST",
        "/harness/runs",
        headers=[(b"content-length", b"4")],
        body=b"hi",
    )
    refused = asyncio.run(_request_ingress_refusal(request))
    assert refused is not None
    assert refused.status_code == 400
    assert refused.body is not None
    assert b"content-length" in refused.body


def test_message_sampling_and_request_validators_fail_closed() -> None:
    assert "content is required" in str(_validation(ChatMessage, role="user"))
    assert "tool_call_id" in str(
        _validation(ChatMessage, role="tool", content=None, tool_call_id="")
    )
    assert "belongs on role 'tool'" in str(
        _validation(ChatMessage, role="user", content="hi", tool_call_id="c1")
    )
    assert "belongs on role 'assistant'" in str(
        _validation(ChatMessage, role="user", content="hi", tool_calls=[{"id": "c"}])
    )
    assert "tool_calls[0]" in str(
        _validation(
            ChatMessage,
            role="assistant",
            content=None,
            tool_calls=[{"id": "c", "type": "function", "function": "nope"}],
        )
    )

    with pytest.raises(ValueError, match="stop accepts"):
        _sampling_extras_valid([""], None, None)
    with pytest.raises(ValueError, match="token ids"):
        _sampling_extras_valid(None, {"nope": 1}, None)
    with pytest.raises(ValueError, match="outside"):
        _sampling_extras_valid(None, {"1": 101}, None)
    with pytest.raises(ValueError, match="metadata"):
        _sampling_extras_valid(None, None, {"k" * 65: "v"})

    assert "non-empty list" in str(
        _validation(
            CompleteRequest,
            backend="hosted_k3",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
        )
    )
    assert "tool_choice" in str(
        _validation(
            CompleteRequest,
            backend="hosted_k3",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "lookup"}}],
            tool_choice={"type": "nope"},
        )
    )
    assert "require a non-empty tools" in str(
        _validation(
            CompleteRequest,
            backend="hosted_k3",
            messages=[{"role": "user", "content": "hi"}],
            tool_choice="auto",
        )
    )
    assert "top_logprobs" in str(
        _validation(
            CompleteRequest,
            backend="hosted_k3",
            messages=[{"role": "user", "content": "hi"}],
            top_logprobs=1,
        )
    )
    assert "262144" in str(_validation(ScoreRequest, input="x" * 262145))
    assert "262144" in str(_validation(ScoreRequest, input=["ok", "y" * 262145]))
    shaped = ScoreRequest(input=["ok"])
    shaped.input = [1]  # type: ignore[assignment]
    with pytest.raises(ValueError, match="list of strings"):
        shaped._input_shape()

    created = EvalRunCreate(model="fx1", callback_url="https://example.com/hook")
    assert created.callback_url == "https://example.com/hook"
    assert "callback_url" in str(_validation(EvalRunCreate, model="fx1", callback_secret="sek"))
    assert "concrete" in str(_validation(ApiKeyPatchRequest, scopes=None))
    assert "concrete" in str(_validation(ApiKeyPatchRequest, admin=None))


def test_breaker_idempotency_redaction_and_grace_pipe() -> None:
    body = CompleteRequest(
        backend="byok",
        messages=[ChatMessage(role="user", content="hi")],
        byok=ByokOverride(base_url="https://example.com/v1", api_key="k", model="m"),
    )
    assert _breaker_key(body).startswith("byok:")

    store = _IdemStore(4, model=_JsonIdemRecord)
    with pytest.raises(ApiError, match="256") as exc:
        _idem_lookup("k" * 257, store, "fp")
    assert exc.value.status_code == 400

    errs: list[dict[str, Any]] = [
        {"loc": "byok", "input": "kept"},
        {"loc": ("byok", "api_key"), "input": "secret", "ctx": {"a": 1}, "url": "u"},
        {"loc": ("api_key",), "ctx": {"a": 1}},
    ]
    _redact_credential_inputs(errs)
    assert errs[0]["input"] == "kept"
    assert errs[1]["input"] == "[redacted]"
    assert "ctx" not in errs[1] and "url" not in errs[1]
    assert "ctx" not in errs[2]

    with pytest.raises(ApiError, match="at least 1 item") as exc:
        _request_or_422(CompleteRequest, {"backend": "hosted_k3", "messages": []})
    assert exc.value.status_code == 422
    assert exc.value.code == "invalid_request"

    pipe = _grace_pipe(lambda: (_ for _ in ()).throw(RuntimeError("backend down")))
    kind, err = pipe.get(timeout=2)
    assert kind == "error"
    assert isinstance(err, HTTPException)
    assert err.status_code == 502
    assert "backend down" in str(err.detail)


def test_submit_replays_only_a_live_job_and_refuses_a_dead_executor() -> None:
    lab = Harness(runner=lambda argv, timeout_s: (0, "ok", ""))
    metrics = api._Metrics(2)
    slots = threading.BoundedSemaphore(2)
    executor = ThreadPoolExecutor(max_workers=1)
    store = _JobStore(4)
    body = HarnessRunRequest(command="doctor")
    fingerprint = api._body_fp(body, exclude={"idempotency_key"})
    store._keys[api._idem_scope("idem-1") or ""] = (fingerprint, "missing-job")
    try:
        submitted = _submit_job(body, "idem-1", lab, store, metrics, slots, executor)
        assert submitted.replayed is False
        assert store.get(submitted.job_id) is not None
        executor.shutdown(wait=True, cancel_futures=True)
        with pytest.raises(ApiError, match="256") as too_long:
            _submit_job(body, "k" * 257, lab, store, metrics, slots, executor)
        assert too_long.value.status_code == 400
        with pytest.raises(ApiError, match="executor unavailable") as exc:
            _submit_job(body, None, lab, store, metrics, slots, executor)
        assert exc.value.status_code == 503
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def test_lifespan_cancels_queued_eval_and_finetune_work() -> None:
    metrics = api._Metrics(1)
    jobs = _JobStore(2)
    evals = EvalStore(2)
    fine = FTJobStore(2)
    executor = ThreadPoolExecutor(max_workers=1)
    evals.put(
        EvalRecord(
            eval_id="ev-1",
            suite="s",
            backend="hosted_k3",
            seed=0,
            status="queued",
            created_at=1.0,
        ),
        None,
        None,
    )
    fine.put(FTJob(id="ft-1", model="fx1", created_at=1, training_file="file-1"), None, "fp")
    lifespan = _make_lifespan(metrics, jobs, evals, executor, fine)
    bare = _make_lifespan(metrics, jobs, EvalStore(2), executor, None)

    async def _enter(manager: Any) -> None:
        async with manager(None):
            return None

    try:
        asyncio.run(_enter(lifespan))
        assert evals.get("ev-1").status == "cancelled"  # type: ignore[union-attr]
        assert fine.get("ft-1").job.status == "cancelled"  # type: ignore[union-attr]
        asyncio.run(_enter(bare))
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def test_receipt_match_and_batch_verifier_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _stored_receipt_matches(b"not-json", _SHA) is False
    assert _stored_receipt_matches(b'{"receipt_sha256": 1}', _SHA) is False

    root = tmp_path / "receipts"
    root.mkdir()
    (root / "one.json").write_text(f'{{"receipt_sha256": "{_SHA}"}}', encoding="utf-8")
    app = api.create_app(receipts_dir=root)
    from fastapi.testclient import TestClient

    def explode(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("verifier down")

    monkeypatch.setattr(api, "verify_receipt_payload", explode)
    batch = TestClient(app).post("/receipts/verify/batch", json={"receipts": [{"no": 1}]})
    assert batch.status_code == 200
    item = batch.json()["results"][0]
    assert item["valid"] is False
    assert item["errors"] == ["verifier down"]

    monkeypatch.setattr(api, "verify_receipt_bytes", lambda *_a, **_k: {"valid": True})
    monkeypatch.setattr(api, "_stored_receipt_matches", lambda *_a, **_k: False)
    fetched = TestClient(app).get(f"/receipts/{_SHA}")
    assert fetched.status_code == 409
    assert fetched.json()["code"] == "receipt_integrity_failed"

    real_read = Path.read_bytes

    def unreadable(self: Path) -> bytes:
        if self.name == "one.json":
            raise OSError("receipt unreadable")
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    down = TestClient(app).get(f"/receipts/{_SHA}")
    assert down.status_code == 503
    assert down.json()["code"] == "receipts_unavailable"


def test_replay_follow_exits_when_the_record_never_terminates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "_REPLAY_FOLLOW_MAX_ITERS", 2)
    monkeypatch.setattr(api.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(api.time, "monotonic", lambda: 0.0)

    class _Store:
        def get(self, response_id: str) -> dict[str, Any]:
            return {
                "id": response_id,
                "object": "response",
                "status": "in_progress",
                "output": [],
                "created_at": 1,
            }

    frames = list(
        _responses_replay_frames(
            _Store(),  # type: ignore[arg-type]
            "resp_1",
            timeout_s=5.0,
            keepalive_s=1.0,
        )
    )
    assert frames
    assert any("in_progress" in frame for frame in frames)


def test_job_event_stream_ends_when_the_record_is_evicted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()
    seen: dict[str, str] = {}

    def runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        release.wait(2)
        return 0, "ok", ""

    app = api.create_app(harness=Harness(runner=runner), sse_keepalive_s=0.01)

    def drop_during_poll(seconds: float) -> None:
        job_id = seen.get("id")
        if job_id:
            app.state.job_store.delete(job_id)
        del seconds

    monkeypatch.setattr(api.time, "sleep", drop_during_poll)
    from fastapi.testclient import TestClient

    client = TestClient(app)
    try:
        submitted = client.post("/harness/jobs", json={"command": "doctor"})
        assert submitted.status_code == 202
        seen["id"] = submitted.json()["job_id"]
        with client.stream("GET", f"/harness/jobs/{seen['id']}/events") as stream:
            body = b"".join(stream.iter_bytes())
        assert b"event: job" in body
        assert app.state.job_store.get(seen["id"]) is None
    finally:
        release.set()


def test_journal_replay_skips_tombstones_and_bare_payloads(tmp_path: Path) -> None:
    idem_path = tmp_path / "idem.jsonl"
    idem = JobJournal(idem_path)
    idem.append({"evicted": ["old"]})
    idem.append({"note": "no record"})
    idem.append(
        {
            "idem": {
                "key": "k",
                "fp": "f",
                "resp": _JsonIdemRecord(envelope={"ok": True}).model_dump(),
            }
        }
    )
    restored = _IdemStore(4, JobJournal(idem_path), _JsonIdemRecord)
    assert restored.get("k") == ("f", _JsonIdemRecord(envelope={"ok": True}))
    _IdemStore(2)._compact_locked()

    job_path = tmp_path / "jobs.jsonl"
    jobs = JobJournal(job_path)
    jobs.append({"deleted": "gone"})
    jobs.append({"evicted": ["old"]})
    jobs.append({"job": _job().model_dump(mode="json")})
    job_store = _JobStore(4, JobJournal(job_path))
    assert job_store.get("job-1") is not None
    _JobStore(2)._compact_locked()
    assert job_store.delete("missing") is None
    live = _JobStore(2, JobJournal(tmp_path / "live-jobs.jsonl"))
    live.put(_job(), None, None)
    cancelled = live.cancel_pending()
    assert cancelled and cancelled[0].status == "cancelled"
    live._compact_locked()
    assert live._journal is not None

    batch_path = tmp_path / "batches.jsonl"
    batches = JobJournal(batch_path)
    batches.append({"evicted": ["old"]})
    batches.append({"batch": _batch().model_dump(mode="json")})
    batch_store = _BatchStore(4, JobJournal(batch_path))
    assert batch_store.get("batch_1") is not None
    _BatchStore(2)._compact_locked()
    orphan = _batch("in_progress")
    orphan.batch_id = "batch_orphan"
    batch_store.mark(orphan)
    assert batch_store.get("batch_orphan") is None

    ab_path = tmp_path / "abatches.jsonl"
    abatches = JobJournal(ab_path)
    abatches.append({"evicted": ["old"]})
    abatches.append({"batch": _abatch().model_dump(mode="json")})
    ab_store = _AnthropicBatchStore(1, JobJournal(ab_path))
    assert ab_store.get("msgbatch_1") is not None
    second = _abatch()
    second.batch_id = "msgbatch_2"
    ab_store.put(second)
    assert ab_store.get("msgbatch_1") is None
    assert ab_store.delete("missing") is None


def test_file_store_replay_put_and_cleanup_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="max_entries"):
        _FileStore(0, 10)
    with pytest.raises(ValueError, match="max_bytes"):
        _FileStore(1, 0)
    memory = _FileStore(2, 10)
    memory._compact_locked()
    with pytest.raises(RuntimeError, match="no state_dir"):
        memory._blob(_FID)

    class _Boom:
        def __init__(self) -> None:
            self.path = Path("/tmp/fx1-missing-journal")

        def append(self, payload: dict[str, Any]) -> None:
            del payload
            raise RuntimeError("journal refused")

    memory._journal = _Boom()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="journal refused"):
        memory.put(filename="m.txt", purpose="batch", content=b"m")
    orphan = _FileStore(2, 100, tmp_path / "journal-none")
    orphan._journal = None

    class _BadLock:
        def __enter__(self) -> None:
            raise RuntimeError("lock refused")

        def __exit__(self, *_exc: object) -> bool:
            return False

    orphan._lock = _BadLock()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="lock refused"):
        orphan.put(filename="o.txt", purpose="batch", content=b"o")
    durable = _FileStore(2, 10, tmp_path / "files-a")
    with pytest.raises(ValueError, match="invalid file id"):
        durable._blob("nope")
    with pytest.raises(ValueError, match="must be an object"):
        durable._apply_replay_op(["nope"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="evicted"):
        durable._apply_replay_op({"evicted": ["nope"]})
    with pytest.raises(ValueError, match="metadata must be an object"):
        durable._apply_replay_op({"file": "nope"})
    with pytest.raises(ValueError, match="delete"):
        durable._apply_replay_op({"file_deleted": "nope"})
    with pytest.raises(ValueError, match="touch"):
        durable._apply_replay_op({"file_touched": _FID})
    with pytest.raises(ValueError, match="unknown"):
        durable._apply_replay_op({"other": 1})
    durable._apply_replay_op(
        {
            "file": {
                "file_id": _FID,
                "filename": "a.txt",
                "purpose": "batch",
                "size": 1,
                "created_at": 1,
                "content_sha256": None,
            }
        }
    )
    with pytest.raises(ValueError, match="unexpected"):
        durable._apply_replay_op({"file_deleted": _FID, "extra": 1})

    root = tmp_path / "capped"
    wide = _FileStore(2, 100, root)
    wide.put(filename="a.txt", purpose="batch", content=b"a")
    wide.put(filename="b.txt", purpose="batch", content=b"b")
    with pytest.raises(RuntimeError, match="invalid operation"):
        _FileStore(1, 100, root)

    failing = _FileStore(2, 100, tmp_path / "replace")

    def refuse_replace(*_args: object, **_kwargs: object) -> None:
        raise OSError("replace refused")

    real_replace = api.os.replace
    monkeypatch.setattr(api.os, "replace", refuse_replace)
    try:
        with pytest.raises(OSError, match="replace refused"):
            failing.put(filename="c.txt", purpose="batch", content=b"c")
    finally:
        monkeypatch.setattr(api.os, "replace", real_replace)

    cleanup = _FileStore(2, 100, tmp_path / "cleanup")
    calls = {"n": 0}
    real_stat = cleanup._journal.path.stat  # type: ignore[union-attr]

    class _Path:
        def stat(self) -> Any:
            calls["n"] += 1
            if calls["n"] > 1:
                raise OSError("stat refused")
            return real_stat()

    cleanup._journal.path = _Path()  # type: ignore[union-attr]

    def refuse_append(_payload: dict[str, Any]) -> None:
        raise RuntimeError("append refused")

    cleanup._journal.append = refuse_append  # type: ignore[union-attr, method-assign]
    with pytest.raises(RuntimeError, match="append refused"):
        cleanup.put(filename="d.txt", purpose="batch", content=b"d")
    assert calls["n"] >= 2
    assert failing.max_bytes == 100


def test_rate_limit_evicts_and_usage_scope_and_auth_edges() -> None:
    limiter = _RateLimiter(0.01, max_keys=1)
    assert limiter.allow("alpha")[0] == 0.0
    assert limiter.allow("beta")[0] == 0.0
    wait, remaining = limiter.allow("alpha")
    assert wait > 0 and remaining == 0.0
    assert len(limiter._buckets) == 1

    store = ApiKeyStore()
    raw, rec = store.mint(name="meter")
    record = CompletionRecord(
        completion_id="c1",
        backend="hosted_k3",
        ok=True,
        latency_ms=1.0,
        at=1.0,
        usage={"prompt_tokens": 2, "completion_tokens": 3},
        prompt_sha256="ab" * 32,
        key_id=rec["key_id"],
    )
    record.usage = {  # type: ignore[assignment]
        "total_tokens": True,
        "prompt_tokens": 2,
        "completion_tokens": 3,
    }
    _charge_key_tokens(record, store)
    assert store.get(rec["key_id"])["tokens_used"] == 5  # type: ignore[index]

    denied = _http("POST", "/harness/runs")
    denied.state.admin = False
    with pytest.raises(ApiError, match="bootstrap") as exc:
        _require_admin(denied)
    assert exc.value.status_code == 403

    plain = _insufficient_scope(_http("POST", "/harness/runs"), frozenset({"read"}))
    assert plain is not None and plain.status_code == 403
    assert b"insufficient_scope" in plain.body  # type: ignore[operator]
    openai = _insufficient_scope(_http("POST", "/v1/chat/completions"), frozenset({"read"}))
    assert openai is not None and b'"error"' in openai.body  # type: ignore[operator]
    assert _insufficient_scope(_http("GET", "/health"), None) is None
    assert _insufficient_scope(_http("GET", "/harness/commands"), frozenset({"read"})) is None

    def authenticate(
        self: ApiKeyStore, provided: str, required_scope: str | None = None
    ) -> dict[str, Any]:
        del self, provided, required_scope
        return {"key_id": "kid", "scopes": ["read"], "admin": False}

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ApiKeyStore, "authenticate", authenticate)
    try:
        refused = _resolve_auth(
            _http("POST", "/harness/runs", headers=[(b"x-api-key", b"not-the-env-key")]),
            "env-secret",
            ApiKeyStore(),
        )
    finally:
        monkeypatch.undo()
    assert getattr(refused, "status_code", None) == 403


def test_key_routes_record_invalid_rotation_and_missing_self(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = api.create_app()
    rotate = _route(app, "key_rotate")
    patch = _route(app, "key_patch")
    self_usage = _route(app, "self_usage")
    keys = _closure(rotate, ApiKeyStore)
    _raw, rec = keys.mint(name="ops")

    def bad_rotate(*_args: object, **_kwargs: object) -> Any:
        raise ValueError("ttl_s must be finite and > 0")

    def bad_update(*_args: object, **_kwargs: object) -> Any:
        raise ValueError("expires_at must be finite and > 0")

    monkeypatch.setattr(keys, "rotate", bad_rotate)
    monkeypatch.setattr(keys, "update", bad_update)
    admin = _http("POST", f"/harness/keys/{rec['key_id']}/rotate")
    admin.state.admin = True
    with pytest.raises(ApiError, match="finite") as rotated:
        rotate(rec["key_id"], ApiKeyRotateRequest(), admin, None, None)
    assert rotated.value.status_code == 422
    with pytest.raises(ApiError, match="finite") as patched:
        patch(rec["key_id"], ApiKeyPatchRequest(), admin, None, None)
    assert patched.value.status_code == 422

    missing = _http("GET", "/harness/self")
    missing.state.key_id = "missing"
    with pytest.raises(ApiError, match="unknown key") as usage:
        self_usage(missing)
    assert usage.value.status_code == 404


def test_budget_header_and_logger_skip_optional_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ApiKeyStore()
    _raw, rec = store.mint(name="window", rpm=10)
    store.authenticate(_raw)
    monkeypatch.setattr(api, "_rfc3339", lambda _moment: None)
    headers = api._anthropic_budget_headers(store, rec["key_id"])
    assert "anthropic-ratelimit-requests-limit" in headers
    assert "anthropic-ratelimit-requests-reset" not in headers

    source = Path(api.__file__).read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(source) if line.startswith("logger = "))
    end = next(
        i for i, line in enumerate(source) if i > start + 1 and line.startswith("def _finish")
    )
    # Include the following def so the skipped-handler branch lands on it.
    nxt = next(
        (
            i
            for i, line in enumerate(source)
            if i > end and line and not line.startswith((" ", "\t"))
        ),
        len(source),
    )
    padded = "\n" * start + "\n".join(source[start:nxt])
    exec(compile(padded, api.__file__, "exec"), dict(api.__dict__))  # noqa: S102
    assert api.logger.handlers


def test_byok_override_is_refused_off_the_byok_link() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(api.create_app())
    response = client.post(
        "/harness/backends/hosted_k3/probe",
        json={"byok": {"base_url": "https://example.com/v1", "api_key": "k", "model": "m"}},
    )
    assert response.status_code == 422
    assert "byok" in response.json()["detail"]


def test_module_main_refuses_open_bind_and_serves_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    source = Path(api.__file__).read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(source) if line.startswith("if __name__"))
    padded = "\n" * start + "\n".join(source[start:])
    code = compile(padded, api.__file__, "exec")
    ran: list[dict[str, Any]] = []

    def _run(host: str, key: str | None) -> None:
        monkeypatch.setenv("FX1_API_HOST", host)
        if key is None:
            monkeypatch.delenv("FX1_API_KEY", raising=False)
        else:
            monkeypatch.setenv("FX1_API_KEY", key)
        namespace = dict(api.__dict__)
        namespace["__name__"] = "__main__"
        exec(code, namespace)  # noqa: S102 - execute the module's own main block

    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: ran.append(kwargs))
    with pytest.raises(SystemExit, match="FX1_API_KEY"):
        _run("0.0.0.0", None)
    _run("0.0.0.0", "sek")
    assert ran and ran[-1]["host"] == "0.0.0.0"
    _run("127.0.0.1", None)
    assert ran[-1]["host"] == "127.0.0.1"
