"""batch_audit — deep-audit battery for the fx1 batch surfaces.

``api_audit`` pins the synchronous wire shapes and ``webhook_audit`` pins
the delivery contract; this battery attacks the *batch lifecycle +
output correctness* contract end to end on BOTH dialects —
``POST /v1/batches`` (OpenAI: input file of JSONL request lines, polled
status, output/error files) and ``POST /v1/messages/batches`` (Anthropic:
inline ``requests[]``, ``processing``/``ended`` tallies, results JSONL).

Probe map:

- *Lifecycle* — ``validating`` observed while the worker is queued,
  ``in_progress`` observed while a gated line is parked inside the
  backend, ``finalizing_at``/``completed_at`` stamped, timestamps
  monotone, ``request_counts`` honest against the rows actually
  written, list newest-first with ``after``/``limit`` cursors.
- *Output correctness* — ``output_file_id`` mints on completion, the
  output file serves one JSONL row per input line, ``custom_id`` joins
  back exactly, the per-line ``response.body`` is the *same envelope a
  live call returns* (parity with ``/v1/chat/completions``,
  ``/v1/responses``, and ``/v1/embeddings`` is the key probe), and a
  failed line carries the OpenAI error body shape.
- *Per-line error isolation* — a line whose body fails the endpoint's
  own validation lands a ``4xx`` error row while its neighbors still
  succeed; line-*shape* garbage (bad JSON, no ``custom_id``, wrong
  method, url mismatch) refuses the whole submit 400 — a batch whose
  input can't be trusted never starts.
- *Cancel window* — cancel a queued batch lands ``cancelled`` with zero
  output; cancel mid-flight lands ``cancelled`` with partial output
  minted; cancel on a terminal batch 409s ``batch_terminal``; a second
  cancel while ``cancelling`` is an idempotent 200.
- *Endpoint validation* — ``endpoint`` outside the served trio refuses
  422 at parse; ``stream``/``background``/``conversation`` lines land
  honest ``invalid_request`` rows instead of running.
- *Completion window + expiry* — ``"24h"`` accepted, any other window
  422s; a batch past ``expires_at`` flips ``expired`` on read, fires the
  terminal webhook once, and the terminal status is never overwritten
  by a worker that finishes afterwards.
- *Anthropic dialect* — ``msgbatch_*`` ids, ``in_progress``/``ended``
  statuses with tallies frozen all-``processing`` until the terminal
  flip, results rows as ``{custom_id, result}`` mapping the live
  ``/v1/messages`` envelope, ``canceling`` + ``cancel_initiated_at``,
  delete only once ``ended``, ``x-api-key`` + ``anthropic-version``
  auth, and the ``{type: "error"}`` error grammar on every refusal.
- *Durability* — a ``--state-dir`` restart mid-batch recovers the
  record as ``failed`` (OpenAI) / ``ended`` with per-item restart
  ``errored`` rows (Anthropic) — never lost, never silently re-run; a
  completed batch's output file still serves post-restart; a fired
  webhook never re-fires after recovery.
- *Concurrency* — two batches run in parallel; the same input file can
  back two batches; a cancel racing a completion lands exactly one
  honest terminal state.
- *Idempotency* — ``Idempotency-Key`` replays the same ``id`` with
  ``X-Fx1-Idempotent-Replay``, and a key reused under a different body
  409s.
- *Usage accounting* — every per-line completion record attributes the
  *submitting* credential's ``key_id`` fingerprint and charges its
  token meter — a batch can never run unmetered lines.
- *Submit refusals* — empty/missing/wrong-purpose/oversized input
  files, non-UTF-8 bytes, over-``batch_line_max`` line counts, drain
  latch, and inflight saturation all fail closed with enveloped bodies.

Probes are literal bools: ``True`` pins a contract that holds; ``False``
pins a measured divergence — the sealed receipt names every defect by
probe name so the finding survives byte-for-byte.

Honesty: every verdict is measured against a live in-process app and a
real loopback webhook sink; no probe stubs its way past the pipeline.
Restart probes replay the real on-disk journal — nothing is simulated.

Composition: complements ``api_audit`` (happy-path wire shapes),
``webhook_audit`` (signature/retry/fire-once delivery),
``usage_audit`` (ledger conservation), ``observe_audit`` (ops surface),
``fault_audit`` (backend-fault taxonomy), ``concurrency_audit``
(inflight saturation), and ``uploads_audit`` (multipart staging).
``batch_audit`` owns the batch *lifecycle + result correctness* claims
those siblings only touch incidentally.

Sealed ``batch_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest import mock

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from fx1.serve.backends import EmbeddingResult, SamplingParams

__all__ = ["batch_audit", "batch_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT = "b4tch-root"
_SWEPT_ENVS = (
    _API_KEY_ENV,
    "MOONSHOT_API_KEY",
    "FX1_API_STATE_DIR",
    "FX1_BYOK_BASE_URL",
    "FX1_BYOK_API_KEY",
    "FX1_BYOK_MODEL",
    "FX1_LOCAL_SERVE_URL",
    "FX1_LOCAL_SERVE_CMD",
    "FX1_LOCAL_MODEL",
    "FX1_LOCAL_API_KEY",
    "FX1_CHECKPOINT_DIR",
    "FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS",
)
_WAIT_S = 20.0
_OPENAI_TERMINAL = ("completed", "failed", "expired", "cancelled")
_STATUS_ORDER = {
    "validating": 0,
    "in_progress": 1,
    "finalizing": 2,
    "cancelling": 2,
    "completed": 3,
    "failed": 3,
    "expired": 3,
    "cancelled": 3,
}
_U = {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}
_EMB_U = {"prompt_tokens": 2, "total_tokens": 2}
_24H_S = 86400


_RESOURCE_STACK: ContextVar[ExitStack | None] = ContextVar("batch_audit_resources", default=None)


@contextmanager
def _audit_resources() -> Iterator[None]:
    """Close every client, sink, and worker pool even when a probe raises."""
    with ExitStack() as stack:
        token = _RESOURCE_STACK.set(stack)
        try:
            yield
        finally:
            _RESOURCE_STACK.reset(token)


def _resources() -> ExitStack:
    stack = _RESOURCE_STACK.get()
    if stack is None:
        raise RuntimeError("audit app creation requires an audit resource context")
    return stack


def _fast_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    del argv, timeout_s
    return 0, "ok", ""


class _StubBackend:
    """Deterministic completion stub: clean content + scripted usage."""

    def __init__(self, model: str = "fx1", usage: dict[str, int] | None = None) -> None:
        self._model = model
        self._usage = dict(usage) if usage else None
        self.calls = 0
        self.last_usage: dict[str, int] | None = None
        self._lock = threading.Lock()

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        sampling: SamplingParams | None = None,  # NOSONAR(S1172)
    ) -> str:
        with self._lock:
            self.calls += 1
        self.last_usage = dict(self._usage) if self._usage is not None else None
        return f"ok:{messages[-1]['content']}"

    def close(self) -> None:
        """No resources to release — the stub holds nothing."""


class _GateBackend(_StubBackend):
    """A backend whose calls park until ``release`` — deterministic
    in-flight occupancy for the cancel/expiry probes (no sleeps, no
    races: ``entered`` is set once the call is inside the backend)."""

    def __init__(self, usage: dict[str, int] | None = None) -> None:
        super().__init__("fx1", usage)
        self.entered = threading.Event()
        self.release = threading.Event()

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        sampling: SamplingParams | None = None,  # NOSONAR(S1172)
    ) -> str:
        with self._lock:
            self.calls += 1
        self.entered.set()
        self.release.wait(timeout=60.0)
        self.last_usage = dict(self._usage) if self._usage is not None else None
        return f"ok:{messages[-1]['content']}"


class _EmbedBackend(_StubBackend):
    """Stub with the optional ``embeddings`` channel — satisfies the
    ``isinstance(EmbeddingBackend)`` capability check structurally."""

    def embeddings(
        self,
        input: Any,  # noqa: A002 — the protocol field's own name
        *,
        model: str,
        encoding_format: str | None = None,  # NOSONAR(S1172)
        dimensions: int | None = None,  # NOSONAR(S1172)
        user: str | None = None,  # NOSONAR(S1172)
    ) -> EmbeddingResult:
        from fx1.serve.backends import EmbeddingResult  # noqa: PLC0415

        with self._lock:
            self.calls += 1
        self.last_usage = dict(_EMB_U)
        return EmbeddingResult(
            data=(
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]},
            )[: 2 if isinstance(input, list) else 1],
            model=model,
            usage=dict(_EMB_U),
        )


class _Hit:
    """One captured webhook delivery: path, headers, raw body bytes."""

    __slots__ = ("body", "headers", "path")

    def __init__(self, path: str, headers: dict[str, str], body: bytes) -> None:
        self.path = path
        self.headers = headers
        self.body = body


class _Sink:
    """A real loopback HTTP webhook sink — ``ThreadingHTTPServer`` on
    ``127.0.0.1:0``; each POST lands as a ``_Hit`` and 200s."""

    def __init__(self) -> None:
        self.hits: list[_Hit] = []
        self.path_n: dict[str, int] = {}
        self._lock = threading.Lock()
        sink = self

        class _H(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 — http.server name
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n)
                with sink._lock:  # noqa: SLF001 — same-module closure state
                    sink.hits.append(_Hit(self.path, dict(self.headers.items()), raw))
                    sink.path_n[self.path] = sink.path_n.get(self.path, 0) + 1
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args: Any) -> None:
                del args

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self._srv.server_address[1]}{path}"

    def close(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()
        self._thread.join(timeout=5)


def _make_app(
    *,
    backend_map: dict[str, Callable[[], Any]] | None = None,
    api_key: str | None = _ROOT,
    resolved: list[str] | None = None,
    **create_kw: Any,
) -> Any:
    """create_app under an isolated env; backends resolve from
    ``backend_map[name]`` zero-arg factories (a factory may itself raise
    to model an unconfigured link)."""
    import fx1.serve.api as api_mod  # noqa: PLC0415
    from fx1.harness import Harness  # noqa: PLC0415

    backends = backend_map or {"hosted_k3": lambda: _StubBackend("fx1", dict(_U))}

    def fake_resolve(name: str, *a: Any, **k: Any) -> Any:
        del a, k
        if resolved is not None:
            resolved.append(name)
        return backends[name]()

    resources = _resources()
    saved = {k: os.environ.get(k) for k in _SWEPT_ENVS}
    try:
        for k in _SWEPT_ENVS:
            os.environ.pop(k, None)
        if api_key is not None:
            os.environ[_API_KEY_ENV] = api_key
        app = api_mod.create_app(
            harness=Harness(runner=_fast_runner),
            backend_resolver=fake_resolve,
            **create_kw,
        )
        resources.callback(app.state.jobs_executor.shutdown, wait=True, cancel_futures=True)
        return app
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _client(
    *,
    backend_map: dict[str, Callable[[], Any]] | None = None,
    api_key: str | None = _ROOT,
    resolved: list[str] | None = None,
    **create_kw: Any,
) -> tuple[TestClient, Any]:
    """(TestClient, FastAPI app) — the battery's standard wired app."""
    from fastapi.testclient import TestClient

    app = _make_app(backend_map=backend_map, api_key=api_key, resolved=resolved, **create_kw)
    stack = _resources()
    client = TestClient(app, raise_server_exceptions=False)
    stack.callback(client.close)
    return stack.enter_context(client), app


def _root_h() -> dict[str, str]:
    return {"X-API-Key": _ROOT}


def _mint(client: TestClient, **policy: Any) -> tuple[str, str]:
    """Mint a managed key under the root credential → (raw, key_id)."""
    r = client.post("/harness/keys", json=policy, headers=_root_h())
    assert r.status_code == 201, r.text
    body = r.json()
    return str(body["key"]), str(body["id"])


def _upload(
    client: TestClient, lines: list[dict[str, Any]], headers: dict[str, str] | None = None
) -> str:
    """Serialize request lines to JSONL and upload purpose=batch → file id."""
    blob = "".join(json.dumps(ln) + "\n" for ln in lines).encode()
    r = client.post(
        "/v1/files",
        files={"file": ("in.jsonl", blob, "application/jsonl")},
        data={"purpose": "batch"},
        headers=headers if headers is not None else _root_h(),
    )
    assert r.status_code == 200, r.text
    return str(r.json()["id"])


def _line(
    custom_id: str, body: dict[str, Any], url: str = "/v1/chat/completions"
) -> dict[str, Any]:
    return {"custom_id": custom_id, "method": "POST", "url": url, "body": body}


def _chat_body(content: str = "hi") -> dict[str, Any]:
    return {"model": "fx1", "messages": [{"role": "user", "content": content}]}


def _batch_create(
    client: TestClient,
    lines: list[dict[str, Any]],
    *,
    endpoint: str = "/v1/chat/completions",
    headers: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
    upload_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    fid = _upload(client, lines, headers=upload_headers)
    body: dict[str, Any] = {
        "input_file_id": fid,
        "endpoint": endpoint,
        "completion_window": "24h",
    }
    body.update(extra or {})
    r = client.post("/v1/batches", json=body, headers=headers or _root_h())
    assert r.status_code == 200, f"batch submit refused: {r.status_code} {r.text}"
    return dict(r.json())


def _abatch_create(
    client: TestClient,
    items: list[dict[str, Any]],
    *,
    headers: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"requests": items}
    body.update(extra or {})
    r = client.post(
        "/v1/messages/batches",
        json=body,
        headers=headers or {**_root_h(), "anthropic-version": "2023-06-01"},
    )
    assert r.status_code == 200, f"abatch submit refused: {r.status_code} {r.text}"
    return dict(r.json())


def _abatch_item(custom_id: str, params_extra: dict[str, Any] | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": "fx1",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "ping"}],
    }
    params.update(params_extra or {})
    return {"custom_id": custom_id, "params": params}


def _wait_batch(
    client: TestClient,
    batch_id: str,
    *,
    timeout: float = _WAIT_S,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    end = time.monotonic() + timeout
    b: dict[str, Any] = {}
    h = headers if headers is not None else _root_h()
    while time.monotonic() < end:
        b = client.get(f"/v1/batches/{batch_id}", headers=h).json()
        if b.get("status") in _OPENAI_TERMINAL:
            return b
        time.sleep(0.03)
    return b


def _wait_abatch(client: TestClient, batch_id: str, *, timeout: float = _WAIT_S) -> dict[str, Any]:
    end = time.monotonic() + timeout
    b: dict[str, Any] = {}
    while time.monotonic() < end:
        b = client.get(
            f"/v1/messages/batches/{batch_id}",
            headers={**_root_h(), "anthropic-version": "2023-06-01"},
        ).json()
        if b.get("processing_status") == "ended":
            return b
        time.sleep(0.03)
    return b


def _wait_until(pred: Callable[[], bool], timeout: float = _WAIT_S) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def _wait_hits(sink: _Sink, n: int, timeout: float = _WAIT_S) -> None:
    end = time.monotonic() + timeout
    while len(sink.hits) < n and time.monotonic() < end:
        time.sleep(0.03)


def _busy_executor(app: Any, slots: int | None = None, sleep_s: float = 1.2) -> None:
    """Occupy every worker thread with a sleeper so a submitted batch
    stays ``validating`` — the deterministic way to reach pre-start
    cancel/expiry/restart states. Defaults to filling the pool."""
    executor = app.state.jobs_executor
    n = int(slots or executor._max_workers)  # noqa: SLF001 — audit reads pool depth
    release = threading.Event()
    for _ in range(n):
        executor.submit(lambda: release.wait(timeout=sleep_s))
    time.sleep(0.05)


def _occupy_executor(app: Any) -> threading.Event:
    """Fill the pool until ``release`` is set.

    ``_busy_executor``'s sleep is shorter than a slow create on a loaded
    runner. A worker then drains and runs the batch the restart probe
    needs left mid-flight. Callers cancel queued work, then set the event.
    """
    executor = app.state.jobs_executor
    n = int(executor._max_workers)  # noqa: SLF001 — audit reads pool depth
    release = threading.Event()
    for _ in range(n):
        executor.submit(release.wait)
    time.sleep(0.05)
    return release


_VOLATILE_KEYS = frozenset({"id", "created", "created_at"})


def _strip_volatile(obj: Any) -> Any:
    """Recursively drop minted ids/timestamps so a batch row can be
    compared shape-for-shape against a live call's envelope."""
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items() if k not in _VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


def _output_rows(client: TestClient, batch: dict[str, Any]) -> list[dict[str, Any]]:
    """Fetch the output file's JSONL rows for a finished batch."""
    ofid = batch.get("output_file_id")
    if not ofid:
        return []
    r = client.get(f"/v1/files/{ofid}/content", headers=_root_h())
    assert r.status_code == 200, r.text
    return [json.loads(ln) for ln in r.text.splitlines() if ln.strip()]


def _abatch_rows(client: TestClient, batch_id: str) -> list[dict[str, Any]]:
    r = client.get(
        f"/v1/messages/batches/{batch_id}/results",
        headers={**_root_h(), "anthropic-version": "2023-06-01"},
    )
    assert r.status_code == 200, r.text
    return [json.loads(ln) for ln in r.text.splitlines() if ln.strip()]


def _probe_lifecycle(client: TestClient, app: Any) -> dict[str, bool]:
    """validating → in_progress → finalizing → completed + counts."""
    out: dict[str, bool] = {}
    h = _root_h()

    # queued behind a busy executor: the submit response IS the
    # validating projection, and a read confirms it survives the wire
    _busy_executor(app)
    b = _batch_create(client, [_line("lc-1", _chat_body("a")), _line("lc-2", _chat_body("b"))])
    got = client.get(f"/v1/batches/{b['id']}", headers=h).json()
    out["create_envelope_shape"] = (
        b.get("object") == "batch"
        and str(b.get("id", "")).startswith("batch_")
        and b.get("endpoint") == "/v1/chat/completions"
        and b.get("completion_window") == "24h"
        and b.get("expires_at") == b.get("created_at", 0) + _24H_S
        and b.get("errors") is None
        and b.get("output_file_id") is None
    )
    out["validating_while_queued"] = (
        b.get("status") == "validating"
        and got.get("status") == "validating"
        and b.get("request_counts") == {"total": 2, "completed": 0, "failed": 0}
        and b.get("in_progress_at") is None
    )
    fin = _wait_batch(client, b["id"])
    out["queued_batch_completes"] = fin.get("status") == "completed"

    # gated backend: the batch parks inside line 1 — in_progress is
    # observable mid-flight, and every poll is monotone forward
    gate = _GateBackend(dict(_U))
    gclient, _ = _client(
        backend_map={"hosted_k3": lambda: gate},
        max_inflight=1,
    )
    try:
        gb = _batch_create(
            gclient,
            [_line("g1", _chat_body("x")), _line("g2", _chat_body("y"))],
        )
        assert _wait_until(lambda: gate.calls >= 1), "worker never entered line 1"
        mid = gclient.get(f"/v1/batches/{gb['id']}", headers=h).json()
        out["in_progress_observed"] = mid.get("status") == "in_progress" and isinstance(
            mid.get("in_progress_at"), int
        )
        gate.release.set()
        gfin = _wait_batch(gclient, gb["id"])
        out["finalizing_stamped_ordered"] = (
            gfin.get("status") == "completed"
            and isinstance(gfin.get("finalizing_at"), int)
            and isinstance(gfin.get("completed_at"), int)
            and gfin["created_at"]
            <= gfin["in_progress_at"]
            <= gfin["finalizing_at"]
            <= gfin["completed_at"]
            and gfin.get("request_counts") == {"total": 2, "completed": 2, "failed": 0}
        )
    finally:
        gate.release.set()

    # status monotonicity: collect the polled sequence on another run —
    # intermediate states may be skipped but never regress
    seq: list[str] = []
    gate2 = _GateBackend(dict(_U))
    sclient, _ = _client(backend_map={"hosted_k3": lambda: gate2}, max_inflight=1)
    try:
        sb = _batch_create(sclient, [_line("s1", _chat_body("z"))])
        end = time.monotonic() + _WAIT_S
        while time.monotonic() < end:
            st = sclient.get(f"/v1/batches/{sb['id']}", headers=h).json().get("status")
            if not seq or seq[-1] != st:
                seq.append(str(st))
            if st in _OPENAI_TERMINAL:
                break
            gate2.release.set()
            time.sleep(0.02)
        gate2.release.set()
        _wait_batch(sclient, sb["id"])
    finally:
        gate2.release.set()
    out["status_monotone"] = seq == sorted(seq, key=lambda s: _STATUS_ORDER.get(s, 0)) and bool(seq)

    # listing: newest-first, object=list, cursor paging
    lclient, _ = _client()
    lb1 = _batch_create(lclient, [_line("l1", _chat_body("1"))])
    lb2 = _batch_create(lclient, [_line("l2", _chat_body("2"))])
    lst = lclient.get("/v1/batches", headers=h).json()
    ids = [it["id"] for it in lst.get("data", [])]
    out["list_newest_first"] = lst.get("object") == "list" and ids[:2] == [lb2["id"], lb1["id"]]
    p1 = lclient.get("/v1/batches?limit=1", headers=h).json()
    p2 = lclient.get(f"/v1/batches?limit=1&after={p1['data'][0]['id']}", headers=h).json()
    out["list_after_cursor_pages"] = (
        p1.get("has_more") is True
        and len(p1.get("data", [])) == 1
        and p1.get("first_id") == p1.get("last_id") == lb2["id"]
        and len(p2.get("data", [])) == 1
        and p2["data"][0]["id"] == lb1["id"]
        and p2.get("has_more") is False
    )
    out["metadata_echoed"] = _batch_create(
        lclient, [_line("m1", _chat_body("m"))], extra={"metadata": {"tag": "lane153"}}
    ).get("metadata") == {"tag": "lane153"}
    out["get_unknown_404"] = (
        lambda r: (
            r.status_code == 404 and r.json().get("error", {}).get("code") == "batch_not_found"
        )
    )(client.get("/v1/batches/batch_nope", headers=h))
    return out


def _probe_output(client: TestClient) -> dict[str, bool]:
    """output_file_id minted; JSONL rows join + mirror live envelopes."""
    out: dict[str, bool] = {}
    h = _root_h()
    chat = _chat_body("parity-check")
    b = _batch_create(client, [_line("p1", chat)])
    fin = _wait_batch(client, b["id"])
    rows = _output_rows(client, fin)
    out["output_file_minted"] = fin.get("status") == "completed" and str(
        fin.get("output_file_id") or ""
    ).startswith("file-")
    out["output_file_record_purpose"] = (
        client.get(f"/v1/files/{fin['output_file_id']}", headers=h).json().get("purpose")
        == "batch_output"
    )
    out["output_rows_one_per_input"] = len(rows) == 1
    out["output_custom_ids_join"] = {r.get("custom_id") for r in rows} == {"p1"}
    out["output_row_shape"] = all(
        str(r.get("id", "")).startswith("batch_req_")
        and r.get("response", {}).get("status_code") == 200
        and str(r.get("response", {}).get("request_id", "")).startswith("req_")
        and r.get("error") is None
        for r in rows
    )
    out["output_content_type_jsonl"] = (
        client.get(f"/v1/files/{fin['output_file_id']}/content", headers=h)
        .headers.get("content-type", "")
        .startswith("application/jsonl")
    )

    # the key probe: the batch line body is shape-identical to a live
    # call for the same request, modulo minted ids/timestamps
    live = client.post("/v1/chat/completions", json=chat, headers=h).json()
    row1 = next(r for r in rows if r["custom_id"] == "p1")["response"]["body"]
    out["line_parity_live_call"] = _strip_volatile(live) == _strip_volatile(row1)
    out["line_usage_is_provider_reported"] = row1.get("usage") == _U

    # /v1/responses line parity (its own endpoint batch)
    rb = _batch_create(
        client,
        [_line("p2", {"model": "fx1", "input": "tell me about dips"}, url="/v1/responses")],
        endpoint="/v1/responses",
    )
    rfin = _wait_batch(client, rb["id"])
    rrows = _output_rows(client, rfin)
    row2 = rrows[0]["response"]["body"] if rrows else {}
    live2 = client.post(
        "/v1/responses", json={"model": "fx1", "input": "tell me about dips"}, headers=h
    ).json()
    out["responses_endpoint_runs"] = (
        rfin.get("status") == "completed"
        and row2.get("object") == "response"
        and row2.get("status") == "completed"
    )
    out["responses_line_parity"] = _strip_volatile(live2) == _strip_volatile(row2)

    # /v1/embeddings batch leg over an embeddings-capable backend
    eclient, _ = _client(backend_map={"hosted_k3": lambda: _EmbedBackend("fx1", dict(_EMB_U))})
    eb = _batch_create(
        eclient,
        [_line("e1", {"model": "fx1", "input": ["a", "b"]}, url="/v1/embeddings")],
        endpoint="/v1/embeddings",
    )
    efin = _wait_batch(eclient, eb["id"])
    erows = _output_rows(eclient, efin)
    ebody = erows[0]["response"]["body"] if erows else {}
    out["embeddings_endpoint_runs"] = (
        efin.get("status") == "completed"
        and ebody.get("object") == "list"
        and [d.get("object") for d in ebody.get("data", [])] == ["embedding", "embedding"]
    )
    # a backend without the channel fails the line 501 honestly
    nb = _batch_create(
        client,
        [_line("n1", {"model": "fx1", "input": "x"}, url="/v1/embeddings")],
        endpoint="/v1/embeddings",
    )
    nfin = _wait_batch(client, nb["id"])
    nrows = _output_rows(client, nfin)
    nrow = nrows[0] if nrows else {}
    out["embeddings_unsupported_501_row"] = (
        nfin.get("status") == "completed"
        and nrow.get("response", {}).get("status_code") == 501
        and nrow.get("response", {}).get("body", {}).get("error", {}).get("code") == "not_supported"
    )
    return out


def _probe_line_isolation(client: TestClient) -> dict[str, bool]:
    """A bad line is a row verdict, never a batch failure."""
    out: dict[str, bool] = {}
    h = _root_h()
    b = _batch_create(
        client,
        [
            _line("ok-1", _chat_body("one")),
            _line(
                "bad-1",
                {
                    "model": "fx1",
                    "messages": [{"role": "user", "content": "x"}],
                    "temperature": 5.0,
                },
            ),
            _line("ok-2", _chat_body("two")),
        ],
    )
    fin = _wait_batch(client, b["id"])
    rows = _output_rows(client, fin)
    bad = next((r for r in rows if r.get("custom_id") == "bad-1"), {})
    out["bad_line_isolated"] = (
        fin.get("status") == "completed"
        and fin.get("request_counts") == {"total": 3, "completed": 2, "failed": 1}
        and len(rows) == 3
        and bad.get("response", {}).get("status_code") == 400
        and isinstance(bad.get("response", {}).get("body", {}).get("error", {}).get("message"), str)
    )
    out["bad_row_error_shape"] = (
        set(bad.get("response", {}).get("body", {}).get("error", {}).keys())
        >= {"message", "type", "code", "param"}
        and bad.get("response", {}).get("body", {}).get("error", {}).get("type")
        == "invalid_request_error"
        and bad.get("error") is None
        and str(bad.get("response", {}).get("request_id", "")).startswith("req_")
    )
    out["no_error_file_when_clean"] = fin.get("error_file_id") is None

    # lines whose requests are structurally legal but batch-illegal land
    # as per-line invalid_request rows — the batch still completes
    b2 = _batch_create(
        client,
        [_line("streamy", {**_chat_body("s"), "stream": True}), _line("fine", _chat_body("ok"))],
    )
    fin2 = _wait_batch(client, b2["id"])
    rows2 = _output_rows(client, fin2)
    srow = next((r for r in rows2 if r.get("custom_id") == "streamy"), {})
    out["stream_line_error_row"] = (
        fin2.get("status") == "completed"
        and srow.get("response", {}).get("status_code") == 400
        and srow.get("response", {}).get("body", {}).get("error", {}).get("code")
        == "invalid_request"
    )
    b3 = _batch_create(
        client,
        [
            _line("bg", {"model": "fx1", "input": "i", "background": True}, url="/v1/responses"),
            _line(
                "conv",
                {"model": "fx1", "input": "i", "conversation": "conv_nope"},
                url="/v1/responses",
            ),
        ],
        endpoint="/v1/responses",
    )
    fin3 = _wait_batch(client, b3["id"])
    rows3 = _output_rows(client, fin3)
    out["background_and_conv_lines_refuse"] = (
        fin3.get("status") == "completed"
        and all(
            r.get("response", {}).get("status_code") == 400
            and r.get("response", {}).get("body", {}).get("error", {}).get("code")
            == "invalid_request"
            for r in rows3
        )
        and len(rows3) == 2
    )

    # line-SHAPE garbage is a submit-time refusal for the whole batch —
    # a batch whose input can't be trusted never starts
    def _submit_blob(blob: bytes, endpoint: str = "/v1/chat/completions") -> int:
        up = client.post(
            "/v1/files",
            files={"file": ("in.jsonl", blob, "application/jsonl")},
            data={"purpose": "batch"},
            headers=h,
        )
        assert up.status_code == 200, up.text
        r = client.post(
            "/v1/batches",
            json={
                "input_file_id": up.json()["id"],
                "endpoint": endpoint,
                "completion_window": "24h",
            },
            headers=h,
        )
        return int(r.status_code)

    out["non_jsonl_line_refuses_submit"] = _submit_blob(b"{not json}\n") == 400
    out["missing_custom_id_refuses_submit"] = (
        _submit_blob(b'{"method":"POST","url":"/v1/chat/completions","body":{}}\n') == 400
    )
    out["method_not_post_refuses_submit"] = (
        _submit_blob(b'{"custom_id":"m","method":"GET","url":"/v1/chat/completions","body":{}}\n')
        == 400
    )
    out["url_endpoint_mismatch_refuses_submit"] = (
        _submit_blob(b'{"custom_id":"u","method":"POST","url":"/v1/responses","body":{}}\n') == 400
    )
    out["non_object_body_refuses_submit"] = (
        _submit_blob(b'{"custom_id":"b","method":"POST","url":"/v1/chat/completions","body":[1]}\n')
        == 400
    )
    out["missing_file_404"] = (
        lambda r: r.status_code == 404 and r.json().get("error", {}).get("code") == "file_not_found"
    )(
        client.post(
            "/v1/batches",
            json={
                "input_file_id": "file-missing",
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
            headers=h,
        )
    )
    return out


def _probe_submit_refusals(client: TestClient) -> dict[str, bool]:
    """Files, windows, sizes, and server-state refusals are enveloped."""
    out: dict[str, bool] = {}
    h = _root_h()

    def _upload_raw(blob: bytes, purpose: str = "batch", name: str = "in.jsonl") -> int | str:
        r = client.post(
            "/v1/files",
            files={"file": (name, blob, "application/jsonl")},
            data={"purpose": purpose},
            headers=h,
        )
        return str(r.json()["id"]) if r.status_code == 200 else int(r.status_code)

    out["empty_lines_400"] = (
        lambda fid: (
            client.post(
                "/v1/batches",
                json={"input_file_id": fid, "endpoint": "/v1/chat/completions"},
                headers=h,
            ).status_code
            == 400
        )
    )(_upload_raw(b"\n\n"))
    out["non_utf8_400"] = (
        lambda fid: (
            client.post(
                "/v1/batches",
                json={"input_file_id": fid, "endpoint": "/v1/chat/completions"},
                headers=h,
            ).status_code
            == 400
        )
    )(_upload_raw(b"\xff\xfe" + json.dumps(_line("x", _chat_body("x"))).encode() + b"\n"))
    out["purpose_not_batch_400"] = (
        lambda fid: (
            client.post(
                "/v1/batches",
                json={"input_file_id": fid, "endpoint": "/v1/chat/completions"},
                headers=h,
            ).status_code
            == 400
        )
    )(
        _upload_raw(
            b'{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n',
            purpose="fine-tune",
            name="c.jsonl",
        )
    )
    # bogus endpoint literal and completion window refuse at body parse
    out["endpoint_literal_422"] = (
        lambda fid: (
            client.post(
                "/v1/batches",
                json={"input_file_id": fid, "endpoint": "/v1/evals"},
                headers=h,
            ).status_code
            == 422
        )
    )(_upload(client, [_line("e", _chat_body("e"), url="/v1/chat/completions")]))
    out["window_bogus_422"] = (
        client.post(
            "/v1/batches",
            json={
                "input_file_id": _upload(client, [_line("w", _chat_body("w"))]),
                "endpoint": "/v1/chat/completions",
                "completion_window": "7d",
            },
            headers=h,
        ).status_code
        == 422
    )
    # over the per-batch line cap
    cap_client, _ = _client(batch_line_max=2)
    cap_fid = _upload(cap_client, [_line(f"c{i}", _chat_body(str(i))) for i in range(3)])
    cap_r = cap_client.post(
        "/v1/batches",
        json={"input_file_id": cap_fid, "endpoint": "/v1/chat/completions"},
        headers=h,
    )
    out["over_line_cap_400"] = (
        cap_r.status_code == 400
        and cap_r.json().get("error", {}).get("code") == "batch_input_limit"
    )
    # drain latch refuses new batches with an enveloped 503 — the input
    # file uploads pre-drain (file writes refuse under the latch too)
    dclient, _ = _client()
    d_fid = _upload(dclient, [_line("d", _chat_body("d"))])
    dclient.post("/harness/drain", headers=h)
    d_openai = dclient.post(
        "/v1/batches",
        json={
            "input_file_id": d_fid,
            "endpoint": "/v1/chat/completions",
        },
        headers=h,
    )
    d_ant = dclient.post(
        "/v1/messages/batches",
        json={"requests": [_abatch_item("d")]},
        headers={**h, "anthropic-version": "2023-06-01"},
    )
    out["drain_refuses_503"] = (
        d_openai.status_code == 503
        and d_openai.json().get("error", {}).get("code") == "draining"
        and d_ant.status_code == 503
        and d_ant.json().get("error", {}).get("type") == "api_error"
    )
    # inflight saturation → 503 over_capacity + Retry-After
    gate = _GateBackend(dict(_U))
    cclient, _ = _client(backend_map={"hosted_k3": lambda: gate}, max_inflight=1)
    try:
        b1 = _batch_create(cclient, [_line("cap", _chat_body("hold"))])
        assert _wait_until(lambda: gate.calls >= 1)
        r2 = cclient.post(
            "/v1/batches",
            json={
                "input_file_id": _upload(cclient, [_line("cap2", _chat_body("x"))]),
                "endpoint": "/v1/chat/completions",
            },
            headers=h,
        )
        out["over_capacity_503_enveloped"] = (
            r2.status_code == 503
            and r2.json().get("error", {}).get("code") == "over_capacity"
            and bool(r2.headers.get("retry-after"))
        )
    finally:
        gate.release.set()
        _wait_batch(cclient, b1["id"])
    return out


def _probe_cancel(client: TestClient, app: Any) -> dict[str, bool]:
    """Queued/mid-flight/terminal/idempotent cancel semantics."""
    out: dict[str, bool] = {}
    h = _root_h()

    # queued: never ran a line → cancelled with no output
    _busy_executor(app)
    b = _batch_create(client, [_line("c1", _chat_body("a")), _line("c2", _chat_body("b"))])
    r = client.post(f"/v1/batches/{b['id']}/cancel", headers=h)
    fin = _wait_batch(client, b["id"])
    out["cancel_queued_cancelled_no_output"] = (
        r.status_code == 200
        and fin.get("status") == "cancelled"
        and fin.get("output_file_id") is None
        and fin.get("request_counts") == {"total": 2, "completed": 0, "failed": 0}
        and isinstance(fin.get("cancelling_at"), int)
        and isinstance(fin.get("cancelled_at"), int)
    )

    # mid-flight: gate parks line 1 of 3; cancel lands while it runs →
    # line 1's row survives as partial output, the tail never runs
    gate = _GateBackend(dict(_U))
    gclient, _ = _client(backend_map={"hosted_k3": lambda: gate}, max_inflight=1)
    try:
        gb = _batch_create(
            gclient,
            [
                _line("k1", _chat_body("1")),
                _line("k2", _chat_body("2")),
                _line("k3", _chat_body("3")),
            ],
        )
        assert _wait_until(lambda: gate.calls >= 1)
        cr = gclient.post(f"/v1/batches/{gb['id']}/cancel", headers=h)
        cbody = cr.json()
        out["cancel_midflight_response_cancelling"] = (
            cr.status_code == 200
            and cbody.get("status") == "cancelling"
            and isinstance(cbody.get("cancelling_at"), int)
        )
        # a second cancel while cancelling is an idempotent re-read
        r2 = gclient.post(f"/v1/batches/{gb['id']}/cancel", headers=h)
        out["cancel_twice_idempotent"] = r2.status_code == 200 and r2.json().get("status") in (
            "cancelling",
            "cancelled",
        )
        gate.release.set()
        gfin = _wait_batch(gclient, gb["id"])
        grows = _output_rows(gclient, gfin)
        out["cancel_midflight_partial_output"] = (
            gfin.get("status") == "cancelled"
            and gfin.get("request_counts") == {"total": 3, "completed": 1, "failed": 0}
            and [r.get("custom_id") for r in grows] == ["k1"]
            and grows[0].get("response", {}).get("status_code") == 200
        )
    finally:
        gate.release.set()

    # terminal batch: 409 batch_terminal envelope; unknown: 404
    done = _wait_batch(client, _batch_create(client, [_line("t", _chat_body("t"))])["id"])
    tr = client.post(f"/v1/batches/{done['id']}/cancel", headers=h)
    out["cancel_terminal_409_envelope"] = (
        tr.status_code == 409 and tr.json().get("error", {}).get("code") == "batch_terminal"
    )
    out["cancel_unknown_404"] = (
        client.post("/v1/batches/batch_nope/cancel", headers=h).status_code == 404
    )
    return out


def _probe_expiry(client: TestClient, sink: _Sink) -> dict[str, bool]:
    """A batch past expires_at flips expired on read — honestly, once."""
    out: dict[str, bool] = {}
    h = _root_h()
    gate = _GateBackend(dict(_U))
    gclient, _ = _client(backend_map={"hosted_k3": lambda: gate}, max_inflight=1)
    try:
        real = time.time()
        # created_at lands 25h back: expires_at is already in the past
        # while the worker is still parked inside line 1
        with mock.patch("time.time", return_value=real - 90000):
            b = _batch_create(
                gclient,
                [_line("x1", _chat_body("slow")), _line("x2", _chat_body("tail"))],
                extra={"callback_url": sink.url("/exp")},
            )
        assert _wait_until(lambda: gate.calls >= 1), "worker never entered line 1"
        n0 = len(sink.hits)
        got = gclient.get(f"/v1/batches/{b['id']}", headers=h).json()
        out["expired_observed_on_read"] = got.get("status") == "expired" and isinstance(
            got.get("expired_at"), int
        )
        _wait_hits(sink, n0 + 1)
        exp_hit = sink.hits[-1]
        gate.release.set()
        # Let the worker settle (it exits the loop at the next line), then
        # read back.  Expiry already terminalized the record: the in-flight
        # provider call is still metered, but its late result is not attached
        # to a terminal payload that may already have been delivered.
        time.sleep(0.6)
        fin = gclient.get(f"/v1/batches/{b['id']}", headers=h).json()
        out["expired_never_overwritten"] = fin.get("status") == "expired"
        out["expired_tail_never_ran"] = (
            gate.calls == 1 and fin.get("request_counts", {}).get("completed") == 0
        )
        _wait_hits(sink, n0 + 1, timeout=0.4)
        out["expired_webhook_once_terminal_payload"] = (
            sink.path_n.get("/exp") == 1
            and json.loads(exp_hit.body).get("status") == "expired"
            and json.loads(exp_hit.body).get("id") == b["id"]
        )
    finally:
        gate.release.set()
    return out


def _probe_anthropic(client: TestClient, app: Any) -> dict[str, bool]:
    """The /v1/messages/batches dialect: envelope, tallies, results."""
    out: dict[str, bool] = {}
    ah = {**_root_h(), "anthropic-version": "2023-06-01"}
    # busy executor keeps the submit response's projection honest —
    # the worker can't beat the envelope read to a terminal state
    _busy_executor(app)
    b = _abatch_create(
        client,
        [_abatch_item("a-1"), _abatch_item("a-2"), _abatch_item("a-3")],
    )
    out["abatch_create_envelope"] = (
        str(b.get("id", "")).startswith("msgbatch_")
        and b.get("type") == "message_batch"
        and b.get("processing_status") == "in_progress"
        and b.get("request_counts")
        == {
            "processing": 3,
            "succeeded": 0,
            "errored": 0,
            "canceled": 0,
            "expired": 0,
        }
        and b.get("results_url") is None
        and b.get("cancel_initiated_at") is None
        and b.get("ended_at") is None
    )
    # RFC3339 timestamps with a +24h expiry
    try:
        created = datetime.fromisoformat(str(b["created_at"]).replace("Z", "+00:00"))
        expires = datetime.fromisoformat(str(b["expires_at"]).replace("Z", "+00:00"))
        out["abatch_timestamps_rfc3339_24h"] = (
            str(b["created_at"]).endswith("Z")
            and str(b["expires_at"]).endswith("Z")
            and int((expires - created).total_seconds()) == _24H_S
        )
    except (ValueError, TypeError, KeyError):
        out["abatch_timestamps_rfc3339_24h"] = False

    fin = _wait_abatch(client, b["id"])
    rows = _abatch_rows(client, b["id"])
    out["abatch_ended_envelope"] = (
        fin.get("processing_status") == "ended"
        and fin.get("results_url") == f"/v1/messages/batches/{b['id']}/results"
        and isinstance(fin.get("ended_at"), str)
    )
    out["abatch_counts_terminal_split"] = fin.get("request_counts") == {
        "processing": 0,
        "succeeded": 3,
        "errored": 0,
        "canceled": 0,
        "expired": 0,
    }
    out["abatch_rows_join_and_shape"] = {r.get("custom_id") for r in rows} == {
        "a-1",
        "a-2",
        "a-3",
    } and all(r.get("result", {}).get("type") == "succeeded" for r in rows)
    msg = rows[0].get("result", {}).get("message", {}) if rows else {}
    out["abatch_result_message_shape"] = (
        msg.get("type") == "message"
        and str(msg.get("id", "")).startswith("msg_")
        and msg.get("role") == "assistant"
        and isinstance(msg.get("content"), list)
        and msg["content"][0].get("type") == "text"
        and isinstance(msg.get("usage", {}).get("input_tokens"), int)
        and isinstance(msg.get("usage", {}).get("output_tokens"), int)
    )
    # parity: the row's message mirrors a live /v1/messages envelope for
    # the same params, modulo the minted id
    live = client.post(
        "/v1/messages",
        json=_abatch_item("x")["params"],
        headers=ah,
    ).json()
    prow = next(r for r in rows if r["custom_id"] == "a-1")["result"]["message"]
    out["abatch_result_parity_live"] = {k: v for k, v in live.items() if k != "id"} == {
        k: v for k, v in prow.items() if k != "id"
    }

    # per-item failure is an errored row, never a batch failure
    eb = _abatch_create(
        client,
        [_abatch_item("ok"), _abatch_item("err", {"fx1": {"backend": "local_fx1"}})],
    )
    efin = _wait_abatch(client, eb["id"])
    erows = _abatch_rows(client, eb["id"])
    erow = next((r for r in erows if r.get("custom_id") == "err"), {})
    out["abatch_errored_row_isolated"] = (
        efin.get("processing_status") == "ended"
        and efin.get("request_counts", {}).get("succeeded") == 1
        and efin.get("request_counts", {}).get("errored") == 1
        and erow.get("result", {}).get("type") == "errored"
        and isinstance(erow.get("result", {}).get("error", {}).get("type"), str)
        and isinstance(erow.get("result", {}).get("error", {}).get("message"), str)
    )
    return out


def _probe_anthropic_lifecycle(client: TestClient, app: Any) -> dict[str, bool]:
    """cancel → canceling → ended(+canceled tail), delete, cursors."""
    out: dict[str, bool] = {}
    ah = {**_root_h(), "anthropic-version": "2023-06-01"}

    # queued-cancel: busy executor keeps the batch pre-start
    _busy_executor(app)
    b = _abatch_create(client, [_abatch_item("c1"), _abatch_item("c2")])
    cr = client.post(f"/v1/messages/batches/{b['id']}/cancel", headers=ah)
    out["abatch_cancel_response_canceling"] = (
        cr.status_code == 200
        and cr.json().get("processing_status") == "canceling"
        and isinstance(cr.json().get("cancel_initiated_at"), str)
    )
    fin = _wait_abatch(client, b["id"])
    rows = _abatch_rows(client, b["id"])
    out["abatch_cancel_queued_all_canceled"] = (
        fin.get("processing_status") == "ended"
        and fin.get("request_counts", {}).get("canceled") == 2
        and all(r.get("result", {}).get("type") == "canceled" for r in rows)
    )

    # mid-flight: gate parks item 1; the in-flight item completes, the
    # tail lands canceled rows (Anthropic semantics)
    gate = _GateBackend(dict(_U))
    gclient, _ = _client(backend_map={"hosted_k3": lambda: gate}, max_inflight=1)
    try:
        gb = _abatch_create(gclient, [_abatch_item("m1"), _abatch_item("m2"), _abatch_item("m3")])
        assert _wait_until(lambda: gate.calls >= 1)
        client2 = gclient.post(f"/v1/messages/batches/{gb['id']}/cancel", headers=ah)
        out["abatch_cancel_twice_idempotent"] = (
            gclient.post(f"/v1/messages/batches/{gb['id']}/cancel", headers=ah).status_code == 200
            and client2.json().get("processing_status") == "canceling"
        )
        gate.release.set()
        gfin = _wait_abatch(gclient, gb["id"])
        grows = _abatch_rows(gclient, gb["id"])
        types = {r["custom_id"]: r["result"]["type"] for r in grows}
        out["abatch_cancel_midflight_tail_canceled"] = (
            gfin.get("processing_status") == "ended"
            and types.get("m1") == "succeeded"
            and types.get("m2") == "canceled"
            and types.get("m3") == "canceled"
            and gfin.get("request_counts")
            == {
                "processing": 0,
                "succeeded": 1,
                "errored": 0,
                "canceled": 2,
                "expired": 0,
            }
        )
    finally:
        gate.release.set()

    # ended-batch cancel is a 400; delete gates on ended; results 400
    # mid-flight — the busy executor keeps both batches pre-start so the
    # mid-flight reads are deterministic
    _busy_executor(app)
    rb = _abatch_create(client, [_abatch_item("r1")])
    db = _abatch_create(client, [_abatch_item("d1")])
    out["abatch_results_400_until_ended"] = (
        client.get(f"/v1/messages/batches/{rb['id']}/results", headers=ah).status_code == 400
    )
    dget = client.request("DELETE", f"/v1/messages/batches/{db['id']}", headers=ah)
    out["abatch_delete_midflight_400"] = (
        dget.status_code == 400
        and dget.json().get("error", {}).get("type") == "invalid_request_error"
    )
    _wait_abatch(client, rb["id"])
    out["abatch_cancel_ended_400"] = (
        client.post(f"/v1/messages/batches/{rb['id']}/cancel", headers=ah)
        .json()
        .get("error", {})
        .get("type")
        == "invalid_request_error"
    )
    out["abatch_delete_ended_shape"] = (
        client.request("DELETE", f"/v1/messages/batches/{rb['id']}", headers=ah).json()
        == {"id": rb["id"], "type": "message_batch_deleted"}
        and client.get(f"/v1/messages/batches/{rb['id']}", headers=ah)
        .json()
        .get("error", {})
        .get("type")
        == "not_found_error"
    )
    out["abatch_get_unknown_404"] = (
        lambda r: (
            r.status_code == 404 and r.json().get("error", {}).get("type") == "not_found_error"
        )
    )(client.get("/v1/messages/batches/msgbatch_nope", headers=ah))

    # cursors: after_id pages forward-older, before_id backward-newer
    lc, _ = _client()
    a1 = _abatch_create(lc, [_abatch_item("l1")])
    a2 = _abatch_create(lc, [_abatch_item("l2")])
    a3 = _abatch_create(lc, [_abatch_item("l3")])
    lst = lc.get("/v1/messages/batches", headers=ah).json()
    ids = [it["id"] for it in lst.get("data", [])]
    paged = lc.get(f"/v1/messages/batches?after_id={a3['id']}", headers=ah).json()
    paged2 = lc.get(f"/v1/messages/batches?before_id={a1['id']}", headers=ah).json()
    out["abatch_list_cursors"] = (
        ids[:3] == [a3["id"], a2["id"], a1["id"]]
        and [it["id"] for it in paged.get("data", [])][:2] == [a2["id"], a1["id"]]
        and paged2.get("data", [])[0].get("id") == a3["id"]
        and lst.get("has_more") is False
    )

    # body-validation refusals stay in the Anthropic error grammar
    def _st(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        r = client.post("/v1/messages/batches", json=payload, headers=ah)
        return r.status_code, r.json()

    code, body = _st({"requests": []})
    out["abatch_empty_requests_422"] = code == 422 and body.get("error", {}).get("type") == (
        "invalid_request_error"
    )
    code, body = _st({"requests": [_abatch_item("dup"), _abatch_item("dup")]})
    out["abatch_dup_custom_id_422"] = code == 422
    code, _ = _st({"requests": [_abatch_item("s", {"stream": True})]})
    out["abatch_stream_param_422"] = code == 422
    code, _ = _st({"requests": [{"custom_id": "noparams"}]})
    out["abatch_missing_params_422"] = code == 422
    code, body = _st({"requests": [_abatch_item("mt", {"max_tokens": 0})]})
    out["abatch_bad_max_tokens_422"] = code == 422
    capc, _ = _client(batch_line_max=1)
    cap_r = capc.post(
        "/v1/messages/batches",
        json={"requests": [_abatch_item("a"), _abatch_item("b")]},
        headers=ah,
    )
    out["abatch_over_cap_400"] = (
        cap_r.status_code == 400
        and cap_r.json().get("error", {}).get("type") == "invalid_request_error"
    )
    return out


def _probe_anthropic_auth() -> dict[str, bool]:
    """x-api-key + anthropic-version authenticate the batch surface."""
    out: dict[str, bool] = {}
    client, _ = _client()
    ah = {**_root_h(), "anthropic-version": "2023-06-01"}
    r = client.post(
        "/v1/messages/batches",
        json={"requests": [_abatch_item("au")]},
        headers=ah,
    )
    out["abatch_xapikey_anthropicversion_ok"] = r.status_code == 200
    out["abatch_request_id_header"] = bool(r.headers.get("request-id"))
    r2 = client.post("/v1/messages/batches", json={"requests": [_abatch_item("au2")]})
    out["abatch_missing_key_401_envelope"] = (
        r2.status_code == 401 and r2.json().get("error", {}).get("type") == "authentication_error"
    )
    r3 = client.post(
        "/v1/batches",
        json={
            "input_file_id": "file-x",
            "endpoint": "/v1/chat/completions",
        },
    )
    out["openai_missing_key_401_envelope"] = (
        r3.status_code == 401
        and r3.json().get("error", {}).get("type") == "authentication_error"
        and r3.json().get("error", {}).get("code") == "unauthorized"
    )
    return out


def _probe_durability(workdir: Path, sink: _Sink) -> dict[str, bool]:
    """A --state-dir restart mid-batch: honest failure, never lost."""
    out: dict[str, bool] = {}
    h = _root_h()

    # OpenAI: batch journaled validating, workers occupied, then "crash"
    sd = workdir / "state-openai"
    c1, app1 = _client(state_dir=sd)
    release_openai = _occupy_executor(app1)
    b = _batch_create(c1, [_line("d1", _chat_body("dur"))])
    app1.state.jobs_executor.shutdown(wait=False, cancel_futures=True)
    release_openai.set()
    c1.close()
    c2, _ = _client(state_dir=sd)
    rec = c2.get(f"/v1/batches/{b['id']}", headers=h).json()
    out["restart_midbatch_fails_honest"] = (
        rec.get("status") == "failed"
        and isinstance(rec.get("failed_at"), int)
        and rec.get("errors", {}).get("object") == "list"
        and rec.get("errors", {}).get("data", [{}])[0].get("code") == "internal_error"
        and "restart" in rec.get("errors", {}).get("data", [{}])[0].get("message", "")
        and rec.get("output_file_id") is None
    )
    # a completed batch + its output file survive the restart, and the
    # delivered webhook never refires (callback_fired claimed on replay)
    sd2 = workdir / "state-done"
    c3, app3 = _client(state_dir=sd2)
    n0 = len(sink.hits)
    cb = _batch_create(
        c3, [_line("ok", _chat_body("kept"))], extra={"callback_url": sink.url("/dur")}
    )
    fin = _wait_batch(c3, cb["id"])
    _wait_hits(sink, n0 + 1)
    app3.state.jobs_executor.shutdown(wait=True, cancel_futures=True)
    c3.close()
    c4, _ = _client(state_dir=sd2)
    rec2 = c4.get(f"/v1/batches/{cb['id']}", headers=h).json()
    rows2 = _output_rows(c4, rec2)
    got3 = c4.get(f"/v1/batches/{cb['id']}", headers=h).json()
    time.sleep(0.3)
    out["restart_completed_survives"] = (
        rec2.get("status") == "completed"
        and rec2.get("output_file_id") == fin.get("output_file_id")
        and len(rows2) == 1
        and rows2[0].get("response", {}).get("status_code") == 200
        and got3.get("status") == "completed"
    )
    out["restart_webhook_no_refire"] = sink.path_n.get("/dur") == 1

    # Anthropic: mid-flight batch recovers ended with restart errored rows
    sd3 = workdir / "state-abatch"
    c5, app5 = _client(state_dir=sd3)
    release_abatch = _occupy_executor(app5)
    ab = _abatch_create(c5, [_abatch_item("r1"), _abatch_item("r2")])
    app5.state.jobs_executor.shutdown(wait=False, cancel_futures=True)
    release_abatch.set()
    c5.close()
    c6, _ = _client(state_dir=sd3)
    rec3 = c6.get(
        f"/v1/messages/batches/{ab['id']}",
        headers={**h, "anthropic-version": "2023-06-01"},
    ).json()
    rows3 = _abatch_rows(c6, ab["id"])
    out["abatch_restart_ends_with_rows"] = (
        rec3.get("processing_status") == "ended"
        and isinstance(rec3.get("ended_at"), str)
        and rec3.get("request_counts", {}).get("errored") == 2
        and len(rows3) == 2
        and all(
            r.get("result", {}).get("type") == "errored"
            and "restart" in r.get("result", {}).get("error", {}).get("message", "")
            for r in rows3
        )
        and {r.get("custom_id") for r in rows3} == {"r1", "r2"}
    )
    return out


def _probe_concurrency(client: TestClient) -> dict[str, bool]:
    """Parallel batches, shared input files, cancel/completion races."""
    out: dict[str, bool] = {}
    h = _root_h()
    fid = _upload(client, [_line("s1", _chat_body("same")), _line("s2", _chat_body("same2"))])
    r1 = client.post(
        "/v1/batches",
        json={"input_file_id": fid, "endpoint": "/v1/chat/completions"},
        headers=h,
    )
    r2 = client.post(
        "/v1/batches",
        json={"input_file_id": fid, "endpoint": "/v1/chat/completions"},
        headers=h,
    )
    f1 = _wait_batch(client, r1.json()["id"])
    f2 = _wait_batch(client, r2.json()["id"])
    rows1 = _output_rows(client, f1)
    rows2 = _output_rows(client, f2)

    def _strip(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop volatile ids so two runs of one input compare equal."""
        return [
            _strip_volatile({k: v for k, v in r.items() if k != "id"})
            | {"response": {**_strip_volatile(r["response"]), "request_id": ""}}
            for r in rows
        ]

    out["same_input_two_batches_equal"] = (
        r1.status_code == 200
        and r2.status_code == 200
        and r1.json()["id"] != r2.json()["id"]
        and f1.get("status") == "completed"
        and f2.get("status") == "completed"
        and _strip(rows1) == _strip(rows2)
    )

    # cancel racing a completion lands exactly one honest terminal state
    raced = _batch_create(client, [_line("rc", _chat_body("race"))])
    rr = client.post(f"/v1/batches/{raced['id']}/cancel", headers=h)
    fin = _wait_batch(client, raced["id"])
    counts = fin.get("request_counts", {})
    out["cancel_race_one_terminal_state"] = (
        (
            rr.status_code == 409
            and rr.json().get("error", {}).get("code") == "batch_terminal"
            and fin.get("status") in _OPENAI_TERMINAL
        )
        or (rr.status_code == 200 and fin.get("status") in ("cancelled", "completed"))
    ) and counts.get("completed", 0) + counts.get("failed", 0) <= counts.get("total", -1)
    return out


def _probe_idempotency(client: TestClient) -> dict[str, bool]:
    """Idempotency-Key replays the same batch id; mismatch 409s."""
    out: dict[str, bool] = {}
    h = _root_h()
    fid = _upload(client, [_line("i1", _chat_body("idem"))])
    body = {"input_file_id": fid, "endpoint": "/v1/chat/completions"}
    r1 = client.post("/v1/batches", json=body, headers={**h, "Idempotency-Key": "bi-1"})
    r2 = client.post("/v1/batches", json=body, headers={**h, "Idempotency-Key": "bi-1"})
    out["idem_replay_same_id"] = (
        r1.status_code == 200
        and r2.status_code == 200
        and r2.headers.get("x-fx1-idempotent-replay") == "true"
        and r2.json() == r1.json()
        and r2.json()["id"] == r1.json()["id"]
    )
    _wait_batch(client, r1.json()["id"])
    r3 = client.post(
        "/v1/batches",
        json={"input_file_id": fid, "endpoint": "/v1/responses"},
        headers={**h, "Idempotency-Key": "bi-1"},
    )
    out["idem_mismatch_409"] = r3.status_code == 409 and isinstance(
        r3.json().get("error", {}).get("code"), str
    )
    # Anthropic store dedupes the same way
    abody = {"requests": [_abatch_item("i1")]}
    ah = {**h, "anthropic-version": "2023-06-01"}
    a1 = client.post("/v1/messages/batches", json=abody, headers={**ah, "Idempotency-Key": "ai-1"})
    a2 = client.post("/v1/messages/batches", json=abody, headers={**ah, "Idempotency-Key": "ai-1"})
    a3 = client.post(
        "/v1/messages/batches",
        json={"requests": [_abatch_item("i2")]},
        headers={**ah, "Idempotency-Key": "ai-1"},
    )
    out["abatch_idem_replay_same_id"] = (
        a1.status_code == 200
        and a2.status_code == 200
        and a2.headers.get("x-fx1-idempotent-replay") == "true"
        and a2.json()["id"] == a1.json()["id"]
    )
    out["abatch_idem_mismatch_409"] = a3.status_code == 409
    _wait_abatch(client, a1.json()["id"])
    return out


def _probe_usage_accounting() -> dict[str, bool]:
    """Per-line records attribute the submitter's key and charge its meter."""
    out: dict[str, bool] = {}
    h = _root_h()
    client, _ = _client(backend_map={"hosted_k3": lambda: _StubBackend("fx1", dict(_U))})
    raw, kid = _mint(client)
    k1 = {"Authorization": f"Bearer {raw}"}
    before = client.get(f"/harness/keys/{kid}/usage", headers=h).json()
    # two managed-key authed calls (upload + create), then poll under root
    # so only the submit touches the key's `uses` counter
    bid = _batch_create(
        client,
        [_line("u1", _chat_body("u1")), _line("u2", _chat_body("u2"))],
        headers=k1,
        upload_headers=k1,
    )["id"]
    _wait_batch(client, bid)
    after = client.get(f"/harness/keys/{kid}/usage", headers=h).json()
    usage = client.get("/harness/usage", headers=h).json()
    by_key = usage.get("by_key", {}).get(kid, {})
    out["batch_lines_attribute_key_id"] = by_key.get("requests") == 2 and by_key.get("ok") == 2
    out["batch_lines_charge_tokens"] = (
        after.get("tokens_used", 0) - before.get("tokens_used", 0) == 2 * _U["total_tokens"]
    )
    out["batch_lines_in_served_card"] = (
        after.get("served", {}).get("calls") == 2
        and after.get("served", {}).get("total_tokens") == 2 * _U["total_tokens"]
    )
    out["batch_uses_counts_http_only"] = after.get("uses", 0) - before.get("uses", 0) == 2

    # the Anthropic dialect attributes identically
    raw2, kid2 = _mint(client)
    ka = {"x-api-key": raw2, "anthropic-version": "2023-06-01"}
    ab = _abatch_create(client, [_abatch_item("au1"), _abatch_item("au2")], headers=ka)
    _wait_abatch(client, ab["id"])
    usage2 = client.get("/harness/usage", headers=h).json()
    out["abatch_lines_attribute_key_id"] = (
        usage2.get("by_key", {}).get(kid2, {}).get("requests") == 2
        and client.get(f"/harness/keys/{kid2}/usage", headers=h).json().get("tokens_used")
        == 2 * _U["total_tokens"]
    )
    # loopback (unkeyed app) records stay unattributed — the "(none)"
    # bucket holds loopback calls, never a keyed caller's work
    c2, _ = _client(api_key=None)
    _wait_batch(c2, _batch_create(c2, [_line("n", _chat_body("n"))], headers={})["id"], headers={})
    u2 = c2.get("/harness/usage").json()
    out["loopback_lines_bucket_none"] = (
        u2.get("by_key", {}).get("(none)", {}).get("requests") or 0
    ) >= 1
    return out


def batch_audit() -> dict[str, Any]:
    """Run the whole batch-surface battery; results are literal bools."""
    prev = {k: os.environ.get(k) for k in _SWEPT_ENVS}
    for k in _SWEPT_ENVS:
        os.environ.pop(k, None)
    # The audit's webhook sink is deliberately loopback-only. Production
    # callback delivery remains public-network-only unless explicitly opted in.
    os.environ["FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"] = "1"
    out: dict[str, Any] = {}
    try:
        with _audit_resources(), tempfile.TemporaryDirectory() as td:
            sink = _Sink()
            _resources().callback(sink.close)
            client, app = _client()
            out.update(_probe_lifecycle(client, app))
            out.update(_probe_output(client))
            out.update(_probe_line_isolation(client))
            out.update(_probe_submit_refusals(client))
            out.update(_probe_cancel(client, app))
            out.update(_probe_expiry(client, sink))
            out.update(_probe_anthropic(client, app))
            out.update(_probe_anthropic_lifecycle(client, app))
            out.update(_probe_anthropic_auth())
            out.update(_probe_durability(Path(td), sink))
            out.update(_probe_concurrency(client))
            out.update(_probe_idempotency(client))
            out.update(_probe_usage_accounting())
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out


def batch_audit_bench() -> dict[str, Any]:
    """Sealed receipt: contract probes True, divergences named."""
    r = batch_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "batch_audit",
        "schema": "batch_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok, "defects": defects},
        "coverage": {
            "transport": "Starlette TestClient in-process + a real loopback webhook sink",
            "not_verified": [
                "real network transport behavior",
                "external process restart (in-process app recreation over the same journal)",
                "24h wall-clock expiry (expiry observed via the record's own expires_at)",
            ],
        },
        "interpretation": (
            "The batch contract holds end to end on both dialects: "
            "/v1/batches walks validating → in_progress → finalizing → "
            "completed with honest request_counts, mints output_file_id "
            "with one JSONL row per input line whose bodies are the same "
            "envelopes a live call returns (chat, responses, and "
            "embeddings parity), isolates per-line failures into error "
            "rows while line-shape garbage refuses the whole submit, "
            "cancels honestly at every window (queued → cancelled with no "
            "output, mid-flight → partial output, terminal → 409), "
            "expires on read without ever being overwritten by a late "
            "worker, and meters every line under the submitting key. The "
            "Anthropic dialect keeps its own grammar (msgbatch_ ids, "
            "processing/ended tallies, {custom_id, result} rows mirroring "
            "the live /v1/messages envelope, canceling + delete-gated "
            "lifecycle, x-api-key auth). A state-dir restart fails a "
            "mid-flight OpenAI batch honestly and ends an Anthropic batch "
            "with restart-errored rows — nothing is lost or silently "
            "re-run, and a fired webhook never refires."
            if ok
            else f"BATCH AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(batch_audit_bench(), indent=2, sort_keys=True))
