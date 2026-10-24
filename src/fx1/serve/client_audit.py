"""client_audit — adversarial probes on the HarnessClient error boundary.

The claim under test: a caller of ``HarnessClient`` — and of the
``fx1 harness`` CLI leg that wraps it — meets exactly the declared error
taxonomy, end to end. Every non-2xx lands on its declared exception
class carrying the server's machine ``code`` and human ``detail``; the
retry policy retries only statuses the wire declares retryable
(429/503 carrying a parseable ``Retry-After``), bounded by
``max_retries`` and ``max_retry_wait_s``; transport faults and timeouts
are honest ``HarnessTransportError``s, never silent empties; SSE
surfaces route an in-band ``error`` frame through the same map and
enforce the terminal-frame contract; idempotent retries collapse to one
server-side execution; the in-process ``Fx1Harness`` SDK raises the same
exception classes the remote client does for the same refusal wherever
both legs implement the surface.

Coverage map:

- *Error map* — each status class lands on its declared exception:
  401/403→HarnessAuthError, 404→KeyError, 422→ValueError,
  501→NotImplementedError, 503→BackendNotConfiguredError, 502 honesty
  gate→Fx1HonestyError, 502 other→RuntimeError, everything
  else→HarnessTransportError. ``detail`` and the machine ``code``
  survive through both the ``{detail, code}`` envelope and the OpenAI
  ``{error: {message, code}}`` shape; pydantic's list-shaped 422 joins
  its field messages; a non-JSON error body is carried, not swallowed.
  Verified synthetically per status and end-to-end through a real app
  for every refusal that has a wire trigger.
- *Retry policy* — 429/503 with ``Retry-After`` retry; the same
  statuses bare fail fast on the first response; the bound is
  ``max_retries``; ``Retry-After`` is honored over the fallback
  backoff and capped by ``max_retry_wait_s`` — a refusal asking for a
  longer wait breaks immediately instead of sleeping it out; writes
  never retry unless keyed idempotent or ``retry_writes``; transport
  faults retry only retryable calls with a doubling backoff; a
  persisted refusal ends in ``HarnessTransportError`` naming the last
  status. The circuit breaker counts transport-class faults and
  fails fast while open; non-transport mapped errors (auth, 4xx) do
  not trip it.
- *Timeout + transport faults* — ``timeout_s`` reaches the transport
  verbatim; the constructor rejects bad knobs before the wire; refused
  connections and resolver failures surface as
  ``HarnessTransportError``; every ``wait_*`` poller honors the
  injected ``clock``/``sleep`` pair, sleeps bounded by the remaining
  budget, and times out with ``HarnessTransportError`` while the
  server-side record keeps running.
- *Streaming honesty* — an in-band ``error`` frame maps through the
  same table and carries the server ``code``; a stream that ends
  without its terminal frame (``[DONE]``, ``response.completed``,
  a terminal job record) is an error on every SSE surface; malformed
  frames fail loudly; every stream surface returns an eager list —
  nothing leaks a half-consumed generator.
- *Idempotency interplay* — same ``Idempotency-Key`` + same body
  dedupes server-side (the runner executes once, the reply is marked
  replayed); same key + different body is a mapped 409
  ``idempotency_conflict``; a transport-fault retry under one key
  still produces exactly one job.
- *Auth edges* — missing, wrong, revoked, and scope-limited
  credentials all map to ``HarnessAuthError``; a hard
  ``quota_exceeded`` 429 carries no ``Retry-After`` and fails fast
  instead of burning retries; a key-level ``rate_limited`` 429
  carries ``Retry-After`` and does retry.
- *CLI legs* — ``fx1 harness`` surfaces the same taxonomy: a wire
  refusal is one clean ``error: Class: msg`` stderr line and exit 2
  with stdout left machine-readable; bad flags are usage errors; the
  in-process and ``--remote`` legs of ``harness commands`` return the
  same registry set.
- *SDK parity* — the in-process ``Fx1Harness`` raises the same
  exception classes the remote client does for the same refusal
  (unknown key → KeyError, contract violation → ValueError,
  streaming on a non-streaming backend → NotImplementedError,
  honesty refusal → Fx1HonestyError); the documented divergence —
  revoke-on-revoked is ``ValueError`` in-process but a 409
  ``HarnessTransportError`` over the wire — is pinned, not hidden.
- *Cancellation* — an abandoned ``wait_run`` leaves the job running
  server-side, untouched; job cancel is queued-only by design (the
  runner has no mid-run kill handle) so running and terminal records
  refuse with a mapped 409; a background response carries a real
  cooperative cancel — it lands mid-flight and refuses once terminal.

Defects found and fixed while building this battery:

1. ``HarnessClient.wait_message_batch`` polled on the wall clock —
   ``time.time()``/``time.sleep()`` — while every sibling waiter
   (``wait_run``/``wait_eval``/``wait_finetune_job``/``wait_batch``)
   honors the injected ``clock``/``sleep`` pair. A caller configured
   for deterministic or test-time polling still hit the real clock;
   the waiter now uses the same injected pair and the
   ``min(poll_s, remaining)`` bound its siblings use.
2. The SSE in-band ``error`` frame in ``stream_complete`` dropped the
   server's machine ``code`` — the mapper re-encoded only ``detail``,
   so a keepalived 503 surfaced as ``BackendNotConfiguredError`` with
   ``code=None`` instead of ``"backend_unavailable"``. The frame's
   ``code`` now flows through ``_map_error``.
3. The in-process legs of ``fx1 harness run`` and ``fx1 harness
   list`` skipped the ``_or_exit`` mapper every other surface uses:
   an unknown command raised a raw ``KeyError`` traceback exiting 1,
   and a bogus ``--role`` raised a raw ``ValueError`` — while the
   ``--remote`` twins of both printed one clean ``error:`` line and
   exited 2. Both legs now map through ``_or_exit`` for parity.

Sealed ``client_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import email.message
import json
import os
import secrets
import socket
import threading
import time
import urllib.error
import urllib.parse
from typing import TYPE_CHECKING, Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import ModuleType

    from fastapi.testclient import TestClient

    from fx1.serve.backends import SamplingParams
    from fx1.serve.client import Transport

__all__ = ["client_audit", "client_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT = secrets.token_urlsafe(32)
_SWEPT_ENVS = (
    _API_KEY_ENV,
    "MOONSHOT_API_KEY",
    "FX1_API_STATE_DIR",
    "FX1_API_MAX_INFLIGHT",
    "FX1_API_SSE_KEEPALIVE_S",
    "FX1_API_IDEM_MAX",
    "FX1_API_JOB_MAX",
    "FX1_API_RATE_LIMIT_RPS",
    "FX1_API_GZIP_MIN_BYTES",
    "FX1_API_CORS_ORIGINS",
    "FX1_API_BREAKER_THRESHOLD",
    "FX1_API_BREAKER_COOLDOWN_S",
    "FX1_API_RECEIPTS_DIR",
    "FX1_API_BYOK_OVERRIDE",
    "FX1_API_FILE_MAX",
    "FX1_API_FILE_BYTES",
    "FX1_API_BATCH_MAX",
    "FX1_API_BATCH_LINES",
    "FX1_API_STORE_MAX",
    "FX1_BYOK_BASE_URL",
    "FX1_BYOK_API_KEY",
    "FX1_BYOK_MODEL",
    "FX1_LOCAL_SERVE_URL",
    "FX1_LOCAL_SERVE_CMD",
    "FX1_LOCAL_MODEL",
    "FX1_LOCAL_API_KEY",
    "FX1_CHECKPOINT_DIR",
)
# Fake host for scripted transports — construction requires a valid
# http(s) base_url; the transport seam means it is never dialed.
_BASE = "http://harness.test"  # NOSONAR(S5332) — never dialed
_MSGS = [{"role": "user", "content": "hi"}]
_NO_CREDS = "no credentials configured"
_ERR_BODY = b'{"detail":"server says no","code":"x_probe"}'
_ITEMS_BODY = b'{"items":[{"name":"alpha"},{"name":"beta"}]}'
_RA = "Retry-After"


class _StubBackend:
    """Deterministic gated backend: echoes the last message, streams two
    chunks, counts calls; ``sleep_s`` stretches the call for the
    inflight-saturation leg. ``entered`` flags a call in flight."""

    def __init__(
        self,
        model: str = "stub-v0",
        *,
        sleep_s: float = 0.0,
        release: threading.Event | None = None,
    ) -> None:
        self._model = model
        self._sleep_s = sleep_s
        # ``release`` holds the call until the probe says so. A wall-clock
        # sleep loses that race when the suite is under coverage: the
        # response completes before cancel is observed.
        self._release = release
        self.calls = 0
        self.entered = threading.Event()
        self.done = threading.Event()

    def _gate(self) -> None:
        self.calls += 1
        self.entered.set()
        if self._release is not None:
            self._release.wait(30.0)
            self.done.set()
        elif self._sleep_s:
            time.sleep(self._sleep_s)
            self.done.set()

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        sampling: SamplingParams | None = None,  # NOSONAR(S1172) — protocol signature
    ) -> str:
        self._gate()
        return f"ok:{messages[-1]['content']}"

    def stream(
        self,
        messages: list[dict[str, str]],  # NOSONAR(S1172) — protocol signature
        *,
        sampling: SamplingParams | None = None,  # NOSONAR(S1172)
    ) -> Any:
        self._gate()
        yield "tok-a"
        yield "tok-b"

    def close(self) -> None:
        """No resources to release — the stub holds nothing."""


class _FailBackend:
    """Refuses before producing output: ``complete`` and ``stream`` raise
    ``BackendNotConfiguredError``; ``sleep_s`` delays the refusal so the
    keepalived stream leg commits the 200 first."""

    def __init__(self, model: str = "fail-v0", *, sleep_s: float = 0.0) -> None:
        self._model = model
        self._sleep_s = sleep_s

    def _refuse(self) -> None:
        if self._sleep_s:
            time.sleep(self._sleep_s)
        from fx1.serve.backends import BackendNotConfiguredError  # noqa: PLC0415

        raise BackendNotConfiguredError(_NO_CREDS)

    def complete(
        self,
        messages: list[dict[str, str]],  # NOSONAR(S1172) — protocol signature
        *,
        sampling: SamplingParams | None = None,  # NOSONAR(S1172)
    ) -> str:
        self._refuse()
        return ""  # unreachable — _refuse raises

    def stream(
        self,
        messages: list[dict[str, str]],  # NOSONAR(S1172) — protocol signature
        *,
        sampling: SamplingParams | None = None,  # NOSONAR(S1172)
    ) -> Any:
        self._refuse()
        yield ""  # unreachable — _refuse raises

    def close(self) -> None:
        """No resources to release — the stub holds nothing."""


class _DirtyBackend:
    """Honesty-gate tripper: the output carries a forbidden headline
    metric the output gate must refuse (502 → Fx1HonestyError)."""

    def __init__(self, model: str = "dirty-v0") -> None:
        self._model = model

    def complete(
        self,
        messages: list[dict[str, str]],  # NOSONAR(S1172) — protocol signature
        *,
        sampling: SamplingParams | None = None,  # NOSONAR(S1172)
    ) -> str:
        return "total Sharpe 4.2 on NAV"  # forbidden headline metric

    def stream(
        self,
        messages: list[dict[str, str]],  # NOSONAR(S1172) — protocol signature
        *,
        sampling: SamplingParams | None = None,  # NOSONAR(S1172)
    ) -> Any:
        yield "total Sharpe 4.2 on NAV"

    def close(self) -> None:
        """No resources to release — the stub holds nothing."""


class _NoStreamBackend:
    """Complete-only backend — no ``stream`` member, so the streaming
    route must refuse 501, never a dropped request."""

    def __init__(self, model: str = "ns-v0") -> None:
        self._model = model

    def complete(
        self,
        messages: list[dict[str, str]],  # NOSONAR(S1172) — protocol signature
        *,
        sampling: SamplingParams | None = None,  # NOSONAR(S1172)
    ) -> str:
        return "ok"

    def close(self) -> None:
        """No resources to release — the stub holds nothing."""


def _client(
    backend_map: dict[str, Any] | None = None,
    api_key: str | None = None,
    *,
    runner: Any = None,
    **app_kw: Any,
) -> tuple[TestClient, ModuleType]:
    """(TestClient, api_module) — isolated env per construction; backends
    resolve from ``backend_map[name]`` — a zero-arg factory or a plain
    instance (an unmapped name raises KeyError → 404). ``runner`` swaps
    the harness executor, ``app_kw`` forwards ``create_app`` knobs
    (rate_limit_rps, max_inflight, sse_keepalive_s)."""
    from fastapi.testclient import TestClient

    import fx1.serve.api as api_mod
    from fx1.harness import Harness

    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:  # NOSONAR(S1172)
        return 0, "ok", ""

    backends = backend_map or {}

    saved = {k: os.environ.get(k) for k in _SWEPT_ENVS}
    try:
        for k in _SWEPT_ENVS:
            os.environ.pop(k, None)
        if api_key is not None:
            os.environ[_API_KEY_ENV] = api_key

        def _resolve(name: str, *a: Any, **k: Any) -> Any:
            entry = backends[name]
            return entry() if callable(entry) else entry

        app = api_mod.create_app(
            harness=Harness(runner=runner if runner is not None else fake_runner),
            backend_resolver=_resolve,
            **app_kw,
        )
        return TestClient(app, raise_server_exceptions=False), api_mod
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _tc_transport(client: TestClient) -> Any:
    """Adapt HarnessClient's transport contract to a TestClient."""

    def send(
        method: str,
        url: str,
        payload: dict[str, Any] | bytes | None,
        headers: dict[str, str],
        timeout_s: float,  # NOSONAR(S1172) — TestClient has no timeout knob
    ) -> tuple[int, Mapping[str, str], bytes]:
        p = urllib.parse.urlparse(url)
        path = p.path + (f"?{p.query}" if p.query else "")
        if method == "GET":
            resp = client.get(path, headers=headers)
        elif method == "DELETE":
            resp = client.delete(path, headers=headers)
        elif method == "PATCH":
            resp = client.patch(path, json=payload, headers=headers)
        elif isinstance(payload, bytes):
            resp = client.post(path, content=payload, headers=headers)
        else:
            resp = client.post(path, json=payload, headers=headers)
        return resp.status_code, dict(resp.headers), resp.content

    return send


def _remote(client: TestClient, api_key: str | None = None, **kw: Any) -> Any:
    """A HarnessClient bound to ``client``'s TestClient transport."""
    from fx1.serve.client import HarnessClient

    return HarnessClient(_BASE, api_key=api_key, transport=_tc_transport(client), **kw)


def _mk(transport: Any, **kw: Any) -> Any:
    """A HarnessClient on a scripted/fake transport."""
    from fx1.serve.client import HarnessClient

    return HarnessClient(_BASE, transport=transport, **kw)


def _scripted(
    *steps: tuple[int, dict[str, str], bytes] | BaseException,
) -> tuple[Transport, list[tuple[str, str]]]:
    """Canned transport: replays ``steps`` — (status, headers, body)
    triples or exceptions to raise — and records (method, path) per call.
    Past the script the last HTTP triple repeats, so a "refusal forever"
    leg is written as a single step."""
    calls: list[tuple[str, str]] = []
    it = iter(steps)
    last: tuple[int, dict[str, str], bytes] = (500, {}, b'{"detail":"script exhausted"}')

    def send(
        method: str,
        url: str,
        payload: dict[str, Any] | bytes | None,  # NOSONAR(S1172) — contract signature
        headers: dict[str, str],  # NOSONAR(S1172)
        timeout_s: float,  # NOSONAR(S1172)
    ) -> tuple[int, Mapping[str, str], bytes]:
        nonlocal last
        calls.append((method, urllib.parse.urlparse(url).path))
        try:
            step = next(it)
        except StopIteration:
            step = last
        else:
            if not isinstance(step, BaseException):
                last = step
        if isinstance(step, BaseException):
            raise step
        return step

    return send, calls


def _scripted_timed(
    *steps: tuple[int, dict[str, str], bytes] | BaseException,
) -> tuple[Transport, list[tuple[str, str]], list[float]]:
    """_scripted + a record of the ``timeout_s`` each call was handed."""
    send, calls = _scripted(*steps)
    timeouts: list[float] = []

    def timed(
        method: str,
        url: str,
        payload: dict[str, Any] | bytes | None,
        headers: dict[str, str],
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        timeouts.append(timeout_s)
        return send(method, url, payload, headers, timeout_s)

    return timed, calls, timeouts


def _vclock() -> tuple[Callable[[], float], Callable[[float], None], list[float]]:
    """Virtual clock + recording sleep: ``sleep`` advances the clock so a
    waiter exhausts its deadline in exactly its declared iterations."""
    t = [0.0]
    sleeps: list[float] = []

    def clock() -> float:
        return t[0]

    def sleep(s: float) -> None:
        sleeps.append(s)
        t[0] += s

    return clock, sleep, sleeps


def _exc(fn: Callable[[], Any]) -> BaseException | None:
    """The exception ``fn`` raised (None when it returned)."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 — probe captures the class
        return exc
    return None


def _closed_port() -> int:
    """A real port nothing listens on — refused deterministically."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _mint(client: TestClient, root_h: dict[str, str], **policy: Any) -> tuple[str, str]:
    """Mint a managed key → (raw, key_id)."""
    r = client.post("/harness/keys", json=policy, headers=root_h)
    assert r.status_code == 201, r.text
    body = r.json()
    return str(body["key"]), str(body["id"])


def _error_map_probes() -> dict[str, bool]:  # NOSONAR(S3776) — status table fans out per class
    """Every status class → its declared exception, synthetic + e2e."""
    out: dict[str, bool] = {}

    def mapped(
        status: int, body: bytes = _ERR_BODY, headers: dict[str, str] | None = None
    ) -> tuple[BaseException | None, Any]:
        tr, calls = _scripted((status, headers or {}, body))
        exc = _exc(lambda: _mk(tr).commands())
        return exc, calls

    # ---- synthetic status table ----
    exc, _ = mapped(401)
    out["map_401_auth"] = type(exc).__name__ == "HarnessAuthError" and "401" in str(exc)
    exc, _ = mapped(403)
    out["map_403_auth"] = type(exc).__name__ == "HarnessAuthError" and "403" in str(exc)
    exc, _ = mapped(404)
    out["map_404_keyerror"] = type(exc).__name__ == "KeyError" and "server says no" in str(exc)
    exc, _ = mapped(422)
    out["map_422_valueerror"] = type(exc).__name__ == "ValueError"
    exc, _ = mapped(501)
    out["map_501_notimplemented"] = type(exc).__name__ == "NotImplementedError"
    exc, calls = mapped(503)
    out["map_503_backendnotconfigured"] = (
        type(exc).__name__ == "BackendNotConfiguredError" and getattr(exc, "code", "") == "x_probe"
    )
    exc, _ = mapped(
        502, b'{"detail":"honesty gate refused model output: sharpe","code":"honesty_gate"}'
    )
    out["map_502_honesty_fx1honesty"] = (
        type(exc).__name__ == "Fx1HonestyError"
        and getattr(exc, "code", "") == "honesty_gate"
        and "honesty gate refused model output" not in str(exc)
    )
    exc, _ = mapped(502, b'{"detail":"upstream died","code":"backend_failure"}')
    out["map_502_other_runtime"] = type(exc).__name__ == "RuntimeError" and "upstream died" in str(
        exc
    )
    for status, name in (
        (400, "map_400"),
        (409, "map_409"),
        (410, "map_410"),
        (413, "map_413"),
        (429, "map_429"),
        (500, "map_500"),
        (418, "map_418"),
    ):
        exc, _ = mapped(status)
        out[f"{name}_transport"] = (
            type(exc).__name__ == "HarnessTransportError"
            and str(status) in str(exc)
            and getattr(exc, "code", "") == "x_probe"
        )

    # ---- envelope + message fidelity ----
    exc, _ = mapped(429, b'{"error":{"message":"slow down","code":"bad_code","type":"rate"}}')
    out["openai_error_envelope"] = (
        type(exc).__name__ == "HarnessTransportError"
        and getattr(exc, "code", "") == "bad_code"
        and "slow down" in str(exc)
    )
    exc, _ = mapped(401, b'{"error":{"message":"invalid key","code":"invalid_api_key"}}')
    out["openai_401_carries_code"] = (
        type(exc).__name__ == "HarnessAuthError"
        and getattr(exc, "code", "") == "invalid_api_key"
        and "invalid key" in str(exc)
    )
    exc, _ = mapped(422, b'{"detail":[{"msg":"field required"},{"msg":"bad type"}]}')
    out["pydantic_detail_joins"] = (
        type(exc).__name__ == "ValueError"
        and "field required" in str(exc)
        and "bad type" in str(exc)
    )
    exc, _ = mapped(500, b"plain crash")
    out["nonjson_error_body_carried"] = type(
        exc
    ).__name__ == "HarnessTransportError" and "plain crash" in str(exc)
    exc, _ = mapped(503, b"{}")
    out["empty_error_body_honest"] = type(exc).__name__ == "BackendNotConfiguredError"

    # ---- end-to-end through a real app ----
    # The request models constrain ``backend`` to the three literal
    # names, so the stubs ride those names through the resolver: an
    # enum-valid name the resolver doesn't know is the wire's 404.
    stub = _StubBackend()
    client, _ = _client({"hosted_k3": stub, "byok": _FailBackend()})
    remote = _remote(client)

    exc = _exc(
        lambda: remote.complete(_MSGS, backend="local_fx1", checkpoint_dir="no/such/checkpoint")
    )
    out["e2e_404_unresolved_backend"] = type(exc).__name__ == "KeyError"
    exc = _exc(lambda: remote.submit_run("no-such-command"))
    out["e2e_404_unknown_command"] = type(exc).__name__ == "KeyError"
    exc = _exc(lambda: remote.retrieve_model("nope-model"))
    out["e2e_404_unknown_model"] = type(exc).__name__ == "KeyError"
    exc = _exc(lambda: remote.job_status("job_nope"))
    out["e2e_404_unknown_job"] = type(exc).__name__ == "KeyError"
    exc = _exc(lambda: remote.complete(_MSGS, backend="local_fx1"))
    out["e2e_422_checkpoint_missing"] = type(exc).__name__ == "ValueError"
    exc = _exc(lambda: remote.complete(_MSGS, backend="hosted_k3", byok={"api_key": "x"}))
    out["e2e_422_byok_on_nonbyok"] = type(exc).__name__ == "ValueError"
    exc = _exc(lambda: remote.complete(_MSGS, backend="hosted_k3", timeout_s=-1))
    out["e2e_422_bad_timeout_field"] = type(exc).__name__ == "ValueError"
    exc = _exc(lambda: remote.complete(_MSGS, backend="byok"))
    out["e2e_503_backend_unavailable"] = (
        type(exc).__name__ == "BackendNotConfiguredError"
        and getattr(exc, "code", "") == "backend_unavailable"
        and _NO_CREDS in str(exc)
    )
    client_ns, _ = _client({"byok": _NoStreamBackend()})
    exc = _exc(lambda: _remote(client_ns).stream_complete(_MSGS, backend="byok"))
    out["e2e_501_no_stream"] = type(exc).__name__ == "NotImplementedError"
    client_d, _ = _client({"byok": _DirtyBackend()})
    exc = _exc(lambda: _remote(client_d).complete(_MSGS, backend="byok"))
    out["e2e_502_honesty_gate"] = type(exc).__name__ == "Fx1HonestyError"

    # authed surface
    client_a, _ = _client({"hosted_k3": _StubBackend()}, api_key=_ROOT)
    exc = _exc(lambda: _remote(client_a).commands())
    out["e2e_401_no_key"] = (
        type(exc).__name__ == "HarnessAuthError" and getattr(exc, "code", "") == "unauthorized"
    )
    return out


def _retry_probes() -> dict[str, bool]:  # NOSONAR(S3776) — retry matrix fans out per policy edge
    """The retry contract: which statuses retry, how many times, what
    the waits look like, what a persisted refusal costs."""
    out: dict[str, bool] = {}
    from fx1.serve.client import HarnessTransportError  # noqa: PLC0415

    _Step = tuple[int, dict[str, str], bytes]
    ok: _Step = (200, {}, _ITEMS_BODY)
    ra429: _Step = (429, {_RA: "1"}, b'{"detail":"slow","code":"too_many_requests"}')
    ra503: _Step = (503, {_RA: "1"}, b'{"detail":"busy","code":"over_capacity"}')
    bare429: _Step = (429, {}, _ERR_BODY)
    bare503: _Step = (503, {}, _ERR_BODY)
    fault = HarnessTransportError("dial failed")

    def client_for(*steps: Any, **kw: Any) -> tuple[Any, Any, list[float]]:
        tr, calls = _scripted(*steps)
        sleeps: list[float] = []
        c = _mk(tr, sleep=lambda s: sleeps.append(s), **kw)
        return c, calls, sleeps

    c, calls, _ = client_for(ra429, ok, max_retries=2)
    out["retry_429_ra_succeeds"] = c.commands() == ["alpha", "beta"] and len(calls) == 2
    c, calls, _ = client_for(ra503, ok, max_retries=2)
    out["retry_503_ra_succeeds"] = c.commands() == ["alpha", "beta"] and len(calls) == 2

    c, calls, _ = client_for(bare429, ok, max_retries=2)
    exc = _exc(c.commands)
    out["no_retry_bare_429"] = type(exc).__name__ == "HarnessTransportError" and len(calls) == 1
    c, calls, _ = client_for(bare503, ok, max_retries=2)
    exc = _exc(c.commands)
    out["no_retry_bare_503"] = type(exc).__name__ == "BackendNotConfiguredError" and len(calls) == 1

    c, calls, sleeps = client_for(ra429, max_retries=2)
    exc = _exc(c.commands)
    # budget gone → the FINAL refusal surfaces mapped (the literal
    # "exhausted N retries" message is only the long-wait break path).
    out["persisted_429_raises_final"] = (
        type(exc).__name__ == "HarnessTransportError"
        and len(calls) == 3
        and "harness API returned 429" in str(exc)
        and sleeps == [1.0, 1.0]
    )
    c, calls, sleeps = client_for(
        (429, {_RA: "0.5"}, _ERR_BODY),
        (429, {_RA: "0.5"}, _ERR_BODY),
        ok,
        max_retries=3,
        retry_backoff_s=0.1,
    )
    out["retry_after_honored_over_backoff"] = c.commands() == ["alpha", "beta"] and sleeps == [
        0.5,
        0.5,
    ]
    c, calls, sleeps = client_for((429, {_RA: "9"}, _ERR_BODY), max_retries=3, max_retry_wait_s=1)
    exc = _exc(c.commands)
    out["long_wait_breaks_fast"] = (
        type(exc).__name__ == "HarnessTransportError" and len(calls) == 1 and sleeps == []
    )
    c, calls, sleeps = client_for((429, {_RA: "-5"}, _ERR_BODY), ok, max_retries=1)
    out["negative_ra_retries_immediately"] = c.commands() == ["alpha", "beta"] and sleeps == [0.0]
    c, calls, _ = client_for(
        (429, {_RA: "Wed, 21 Oct 2015 07:28:00 GMT"}, _ERR_BODY), ok, max_retries=2
    )
    exc = _exc(c.commands)
    out["http_date_ra_not_retried"] = (
        type(exc).__name__ == "HarnessTransportError" and len(calls) == 1
    )
    c, calls, _ = client_for((429, {_RA: "soon"}, _ERR_BODY), ok, max_retries=2)
    exc = _exc(c.commands)
    out["malformed_ra_not_retried"] = (
        type(exc).__name__ == "HarnessTransportError" and len(calls) == 1
    )

    c, calls, sleeps = client_for(fault, ok, max_retries=1, retry_backoff_s=0.1)
    out["fault_retried_on_idempotent"] = c.commands() == ["alpha", "beta"] and sleeps == [0.1]
    c, calls, sleeps = client_for(fault, fault, ok, max_retries=2, retry_backoff_s=0.1)
    out["fault_backoff_doubles"] = c.commands() == ["alpha", "beta"] and sleeps == [0.1, 0.2]

    job_body: _Step = (200, {}, b'{"job_id":"j1","status":"queued","replayed":false}')
    comp_body: _Step = (
        200,
        {},
        b'{"backend":"hosted_k3","model":"m","content":"ok",'
        b'"receipt_hashes":[],"replayed":false,"attempts":[],'
        b'"usage":{"tokens_in":1,"tokens_out":1},"completion_id":"c-1"}',
    )
    # ``complete`` unkeyed is the genuinely non-idempotent write: no
    # retry under faults, no retry under a retryable refusal.
    c, calls, _ = client_for(fault, comp_body, max_retries=3)
    exc = _exc(lambda: c.complete(_MSGS))
    out["no_retry_write_by_default"] = (
        type(exc).__name__ == "HarnessTransportError" and len(calls) == 1
    )
    c, calls, _ = client_for(fault, comp_body, max_retries=3, retry_writes=True)
    out["retry_writes_retries_post"] = c.complete(_MSGS).content == "ok" and len(calls) == 2
    c, calls, _ = client_for(ra429, comp_body, max_retries=3)
    exc = _exc(lambda: c.complete(_MSGS))
    out["no_retry_429_on_unkeyed_write"] = (
        type(exc).__name__ == "HarnessTransportError" and len(calls) == 1
    )
    # an explicit Idempotency-Key marks the write idempotent → retryable.
    c, calls, _ = client_for(ra429, comp_body, max_retries=3)
    out["keyed_write_429_retries"] = (
        c.complete(_MSGS, idempotency_key="k-1").content == "ok" and len(calls) == 2
    )
    # ``submit_run`` mints a key itself → retried by construction.
    c, calls, _ = client_for(ra429, job_body, max_retries=3)
    out["submit_autokeyed_retries"] = c.submit_run("doctor") == "j1" and len(calls) == 2
    c, calls, _ = client_for(fault, fault, fault, ok, max_retries=2)
    exc = _exc(c.commands)
    out["persisted_fault_exhausts"] = (
        type(exc).__name__ == "HarnessTransportError"
        and len(calls) == 3
        and "dial failed" in str(exc)
    )

    # circuit breaker: threshold faults open the circuit; it fails fast
    # while open and half-opens after the reset window.
    tr_cb, calls_cb = _scripted(fault)
    clock_cb, sleep_cb, _ = _vclock()
    cb = _mk(
        tr_cb,
        max_retries=0,
        sleep=sleep_cb,
        clock=clock_cb,
        circuit_breaker_threshold=2,
        circuit_reset_s=30.0,
    )
    e1, e2 = _exc(cb.commands), _exc(cb.commands)
    e3 = _exc(cb.commands)  # circuit open — no transport call
    out["circuit_fails_fast_open"] = (
        type(e1).__name__ == "HarnessTransportError"
        and type(e2).__name__ == "HarnessTransportError"
        and type(e3).__name__ == "HarnessTransportError"
        and "circuit open" in str(e3)
        and len(calls_cb) == 2
    )
    # mapped refusals (HarnessTransportError-classed) count toward the
    # breaker too — pinned, the docstring says "transport fault" and the
    # refusal taxonomy shares the class.
    tr_cb2, calls_cb2 = _scripted(bare429)
    cb2 = _mk(
        tr_cb2,
        max_retries=0,
        sleep=lambda s: None,
        circuit_breaker_threshold=2,
        circuit_reset_s=30.0,
    )
    _exc(cb2.commands)
    _exc(cb2.commands)
    e3 = _exc(cb2.commands)
    out["circuit_counts_mapped_refusals"] = "circuit open" in str(e3) and len(calls_cb2) == 2
    # auth/4xx classes do not trip the breaker.
    tr_cb3, calls_cb3 = _scripted((401, {}, _ERR_BODY))
    cb3 = _mk(tr_cb3, max_retries=0, circuit_breaker_threshold=2, circuit_reset_s=30.0)
    _exc(cb3.commands)
    _exc(cb3.commands)
    _exc(cb3.commands)
    out["auth_refusals_skip_circuit"] = len(calls_cb3) == 3
    return out


def _timeout_probes() -> dict[str, bool]:  # NOSONAR(S3776) — ctor matrix + waiter matrix
    """Timeouts bounded and the waiters' injectable clock honored."""
    out: dict[str, bool] = {}
    from fx1.serve.client import HarnessClient  # noqa: PLC0415

    out["ctor_rejects_bad_scheme"] = type(_exc(lambda: HarnessClient("ftp://x"))).__name__ == (
        "ValueError"
    )
    out["ctor_rejects_no_host"] = type(_exc(lambda: HarnessClient("http://"))).__name__ == (
        "ValueError"
    )
    out["ctor_rejects_bare_host"] = type(_exc(lambda: HarnessClient("harness.test"))).__name__ == (
        "ValueError"
    )
    out["ctor_rejects_zero_timeout"] = (
        type(_exc(lambda: HarnessClient(_BASE, timeout_s=0))).__name__ == "ValueError"
    )
    out["ctor_rejects_negative_retries"] = (
        type(_exc(lambda: HarnessClient(_BASE, max_retries=-1))).__name__ == "ValueError"
    )
    out["ctor_rejects_zero_backoff"] = (
        type(_exc(lambda: HarnessClient(_BASE, retry_backoff_s=0))).__name__ == "ValueError"
    )
    out["ctor_rejects_zero_max_wait"] = (
        type(_exc(lambda: HarnessClient(_BASE, max_retry_wait_s=0))).__name__ == "ValueError"
    )
    out["ctor_rejects_negative_cb_threshold"] = (
        type(_exc(lambda: HarnessClient(_BASE, circuit_breaker_threshold=-1))).__name__
        == "ValueError"
    )
    out["ctor_rejects_zero_cb_reset"] = (
        type(_exc(lambda: HarnessClient(_BASE, circuit_reset_s=0))).__name__ == "ValueError"
    )

    tr_t, _, timeouts = _scripted_timed((200, {}, _ITEMS_BODY))
    _mk(tr_t, timeout_s=7.5).commands()
    out["timeout_s_reaches_transport"] = timeouts == [7.5]

    # waiters on a never-terminal record + injected clock: bounded by
    # timeout_s, each sleep capped by the remaining budget.
    _Step = tuple[int, dict[str, str], bytes]
    running_job: _Step = (200, {}, b'{"job_id":"j1","status":"running"}')
    clock, sleep, sleeps = _vclock()
    tr, calls = _scripted(running_job)
    c = _mk(tr, clock=clock, sleep=sleep)
    exc = _exc(lambda: c.wait_run("j1", poll_s=0.5, timeout_s=5.0))
    out["wait_run_timeout_bounded"] = (
        type(exc).__name__ == "HarnessTransportError"
        and "5.0" in str(exc)
        and len(calls) == 11
        and sleeps == [0.5] * 10
    )
    tr, calls = _scripted(running_job)
    clock, sleep, sleeps = _vclock()
    exc = _exc(lambda: _mk(tr, clock=clock, sleep=sleep).wait_run("j1", poll_s=0.5, timeout_s=0.3))
    out["wait_run_sleep_capped_by_remaining"] = sleeps == [0.3] and len(calls) == 2

    running_eval: _Step = (200, {}, b'{"eval_id":"e1","status":"running"}')
    clock, sleep, sleeps = _vclock()
    tr, calls = _scripted(running_eval)
    exc = _exc(lambda: _mk(tr, clock=clock, sleep=sleep).wait_eval("e1", poll_s=0.5, timeout_s=5.0))
    out["wait_eval_timeout_bounded"] = (
        type(exc).__name__ == "HarnessTransportError" and len(calls) == 11 and sleeps == [0.5] * 10
    )
    open_batch: _Step = (200, {}, b'{"id":"b1","status":"in_progress"}')
    clock, sleep, sleeps = _vclock()
    tr, calls = _scripted(open_batch)
    exc = _exc(
        lambda: _mk(tr, clock=clock, sleep=sleep).wait_batch("b1", poll_s=0.5, timeout_s=5.0)
    )
    out["wait_batch_timeout_bounded"] = (
        type(exc).__name__ == "HarnessTransportError" and len(calls) == 11 and sleeps == [0.5] * 10
    )
    running_ft: _Step = (200, {}, b'{"id":"f1","status":"running"}')
    clock, sleep, sleeps = _vclock()
    tr, calls = _scripted(running_ft)
    exc = _exc(
        lambda: _mk(tr, clock=clock, sleep=sleep).wait_finetune_job("f1", poll_s=0.5, timeout_s=5.0)
    )
    out["wait_finetune_timeout_bounded"] = (
        type(exc).__name__ == "HarnessTransportError" and len(calls) == 11 and sleeps == [0.5] * 10
    )
    # defect-fix pin: wait_message_batch must honor the injected
    # clock/sleep exactly like its siblings (it used wall time before).
    pending_mb: _Step = (200, {}, b'{"id":"mb1","processing_status":"in_progress"}')
    clock, sleep, sleeps = _vclock()
    tr, calls = _scripted(pending_mb)
    exc = _exc(
        lambda: _mk(tr, clock=clock, sleep=sleep).wait_message_batch(
            "mb1", poll_s=0.5, timeout_s=5.0
        )
    )
    out["wait_message_batch_injected_clock"] = (
        type(exc).__name__ == "HarnessTransportError" and len(calls) == 11 and sleeps == [0.5] * 10
    )
    clock, sleep, sleeps = _vclock()
    tr, calls = _scripted(pending_mb)
    exc = _exc(
        lambda: _mk(tr, clock=clock, sleep=sleep).wait_message_batch(
            "mb1", poll_s=0.5, timeout_s=0.3
        )
    )
    out["wait_message_batch_sleep_bounded"] = sleeps == [0.3] and len(calls) == 2
    return out


def _transport_fault_probes() -> dict[str, bool]:  # NOSONAR(S3776)
    """Wire faults, socket-level honesty, malformed success payloads."""
    out: dict[str, bool] = {}
    import http.client as _http  # noqa: PLC0415
    from unittest import mock  # noqa: PLC0415

    from fx1.serve.client import HarnessClient, _urllib_transport  # noqa: PLC0415

    t0 = time.monotonic()
    exc = _exc(lambda: HarnessClient(f"http://127.0.0.1:{_closed_port()}", timeout_s=5).commands())
    out["conn_refused_transport_error"] = (
        type(exc).__name__ == "HarnessTransportError" and time.monotonic() - t0 < 5
    )
    with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("no dns")):
        exc = _exc(lambda: _urllib_transport("GET", "http://dead.invalid/x", None, {}, 1.0))
    out["dns_failure_transport_error"] = type(exc).__name__ == "HarnessTransportError"
    with mock.patch("urllib.request.urlopen", side_effect=OSError("socket reset")):
        exc = _exc(lambda: _urllib_transport("GET", _BASE, None, {}, 1.0))
    out["oserror_wrapped_transport"] = type(exc).__name__ == "HarnessTransportError"
    with mock.patch("urllib.request.urlopen", side_effect=_http.HTTPException("broken")):
        exc = _exc(lambda: _urllib_transport("GET", _BASE, None, {}, 1.0))
    out["httpexception_wrapped_transport"] = type(exc).__name__ == "HarnessTransportError"
    # HTTPError is not a fault — it owns the refused response and
    # returns it for mapping (the wire gets a status, not an exception).
    hdrs = email.message.Message()
    http_err = urllib.error.HTTPError(_BASE, 418, "teapot", hdrs, None)
    with mock.patch("urllib.request.urlopen", side_effect=http_err):
        tr_res = _urllib_transport("GET", _BASE, None, {}, 1.0)
    out["http_error_returns_status"] = tr_res[0] == 418

    # a custom transport that raises a foreign (non-HarnessTransportError)
    # exception leaks it raw — the transport injection owns its own
    # taxonomy; pinned so callers know the seam's contract.
    def oserror_send(
        method: str,
        url: str,
        payload: Any,
        headers: dict[str, str],
        timeout_s: float,
    ) -> None:  # NOSONAR(S1172)
        raise OSError("foreign fault")

    exc = _exc(lambda: _mk(oserror_send, max_retries=2).commands())
    out["foreign_exception_unwrapped"] = type(exc).__name__ == "OSError"

    # response-headers bookkeeping: refused-but-answered calls keep the
    # headers (x-request-id stays quotable); transport faults clear them.
    tr, _ = _scripted(
        (401, {"X-Request-Id": "req-1"}, _ERR_BODY),
    )
    c = _mk(tr)
    _exc(c.commands)
    out["refusal_keeps_last_headers"] = c.last_response_headers.get("x-request-id") == "req-1"
    from fx1.serve.client import HarnessTransportError  # noqa: PLC0415

    tr, _ = _scripted(HarnessTransportError("dial"))
    c = _mk(tr)
    _exc(c.commands)
    out["transport_fault_clears_headers"] = c.last_response_headers == {}

    # malformed success payloads — honest exceptions, never fabricated.
    tr, _ = _scripted((200, {}, b"not json"))
    exc = _exc(lambda: _mk(tr).commands())
    out["malformed_json_2xx_raises"] = type(exc).__name__ == "JSONDecodeError"
    tr, _ = _scripted((200, {}, b""))
    exc = _exc(lambda: _mk(tr).commands())
    out["empty_body_2xx_raises"] = type(exc).__name__ == "JSONDecodeError"
    tr, _ = _scripted((200, {}, b'{"nope": 1}'))
    exc = _exc(lambda: _mk(tr).commands())
    out["wrong_shape_2xx_raises"] = type(exc).__name__ in ("KeyError", "TypeError")
    tr, _ = _scripted((200, {}, b"[1,2]"))
    exc = _exc(lambda: _mk(tr).commands())
    out["list_body_2xx_raises"] = type(exc).__name__ in ("KeyError", "TypeError")
    return out


def _stream_probes() -> dict[
    str, bool
]:  # NOSONAR(S3776) — SSE surfaces each pin their terminal frame
    """SSE honesty: in-band error frames map through the table (code
    included), missing terminal frames error, garbage fails loudly."""
    out: dict[str, bool] = {}
    tok = b'data: {"type":"token","content":"x"}\n\n'
    done = b"data: [DONE]\n\n"

    tr, _ = _scripted((200, {}, tok + done))
    out["stream_complete_ok"] = _mk(tr).stream_complete(_MSGS) == ["x"]
    tr, _ = _scripted((200, {}, tok))
    exc = _exc(lambda: _mk(tr).stream_complete(_MSGS))
    out["stream_truncated_errors"] = type(exc).__name__ == "HarnessTransportError"
    tr, _ = _scripted((200, {}, b""))
    exc = _exc(lambda: _mk(tr).stream_complete(_MSGS))
    out["stream_empty_errors"] = type(exc).__name__ == "HarnessTransportError"
    tr, _ = _scripted((200, {}, b"data: {oops\n\n" + done))
    exc = _exc(lambda: _mk(tr).stream_complete(_MSGS))
    out["stream_bad_frame_raises"] = type(exc).__name__ == "JSONDecodeError"
    tr, _ = _scripted((200, {}, b": keepalive\n\n" + tok + done))
    out["stream_keepalive_comments_skipped"] = _mk(tr).stream_complete(_MSGS) == ["x"]

    # in-band error frames: status AND machine code flow through the map.
    err503 = (
        b'data: {"type":"error","status":503,"detail":"no backend",'
        b'"code":"backend_unavailable"}\n\n' + done
    )
    tr, _ = _scripted((200, {}, err503))
    exc = _exc(lambda: _mk(tr).stream_complete(_MSGS))
    out["stream_error_frame_maps_503"] = type(exc).__name__ == "BackendNotConfiguredError"
    out["stream_error_frame_carries_code"] = getattr(exc, "code", "") == "backend_unavailable"
    err401 = (
        b'data: {"type":"error","status":401,"detail":"bad key","code":"unauthorized"}\n\n' + done
    )
    tr, _ = _scripted((200, {}, err401))
    exc = _exc(lambda: _mk(tr).stream_complete(_MSGS))
    out["stream_error_frame_maps_401"] = (
        type(exc).__name__ == "HarnessAuthError" and getattr(exc, "code", "") == "unauthorized"
    )
    err_hon = (
        b'data: {"type":"error","status":502,'
        b'"detail":"honesty gate refused model output: sharpe","code":"honesty_gate"}\n\n' + done
    )
    tr, _ = _scripted((200, {}, err_hon))
    exc = _exc(lambda: _mk(tr).stream_complete(_MSGS))
    out["stream_error_frame_maps_honesty"] = type(exc).__name__ == "Fx1HonestyError"

    # /v1/chat/completions stream
    chunk = (
        b'data: {"id":"c1","object":"chat.completion.chunk","created":1,'
        b'"model":"fx1","choices":[{"index":0,"delta":{"content":"a"}}]}\n\n'
    )
    tr, _ = _scripted((200, {}, chunk + done))
    chunks, cid = _mk(tr).chat_completion_stream(_MSGS)
    out["openai_stream_ok"] = isinstance(chunks, list) and len(chunks) == 1
    tr, _ = _scripted((200, {}, chunk))
    exc = _exc(lambda: _mk(tr).chat_completion_stream(_MSGS))
    out["openai_stream_truncated_errors"] = type(exc).__name__ == "HarnessTransportError"

    # /v1/responses stream + replay terminal-frame contracts
    delta = b'data: {"type":"response.output_text.delta","delta":"x"}\n\n'
    completed = b'data: {"type":"response.completed","response":{}}\n\n'
    tr, _ = _scripted((200, {}, delta + completed))
    evs, _cid = _mk(tr).responses_create_stream("hi")
    out["responses_stream_terminal_ok"] = isinstance(evs, list) and evs[-1]["type"] == (
        "response.completed"
    )
    tr, _ = _scripted((200, {}, delta))
    exc = _exc(lambda: _mk(tr).responses_create_stream("hi"))
    out["responses_stream_truncated_errors"] = type(exc).__name__ == "HarnessTransportError"
    tr, _ = _scripted((200, {}, delta))
    exc = _exc(lambda: _mk(tr).responses_replay("resp_1"))
    out["responses_replay_terminal_required"] = type(exc).__name__ == "HarnessTransportError"
    tr, _ = _scripted((200, {}, delta + b'data: {"type":"response.failed","response":{}}\n\n'))
    evs, _cid = _mk(tr).responses_replay("resp_1")
    out["responses_replay_failed_terminal"] = evs[-1]["type"] == "response.failed"

    # job-event stream
    jrec = b'data: {"job_id":"j1","status":"running"}\n\n'
    jdone = (
        b'data: {"job_id":"j1","status":"succeeded","result":{"command":"x",'
        b'"exit_code":0,"stdout":"s","stderr":""}}\n\n'
    )
    tr, _ = _scripted((200, {}, jrec + jdone))
    frames = _mk(tr).stream_job("j1")
    out["job_stream_returns_frames"] = len(frames) == 2 and frames[-1]["status"] == "succeeded"
    tr, _ = _scripted((200, {}, jrec + jdone))
    res = _mk(tr).wait_run_stream("j1")
    out["wait_run_stream_terminal"] = res.exit_code == 0 and res.stdout == "s"
    tr, _ = _scripted((200, {}, jrec))
    exc = _exc(lambda: _mk(tr).wait_run_stream("j1"))
    out["wait_run_stream_nonterminal_errors"] = type(exc).__name__ == "HarnessTransportError"
    tr, _ = _scripted((200, {}, b""))
    exc = _exc(lambda: _mk(tr).stream_job("j1"))
    out["job_stream_empty_errors"] = type(exc).__name__ == "HarnessTransportError"
    tr, _ = _scripted((200, {}, b'data: {"job_id":"j1","status":"failed","error":"boom"}\n\n'))
    exc = _exc(lambda: _mk(tr).wait_run_stream("j1"))
    out["wait_run_stream_failed_joberror"] = type(exc).__name__ == "HarnessJobError"

    # ---- end-to-end over a real app ----
    stub = _StubBackend()
    client, _ = _client({"hosted_k3": stub})
    remote = _remote(client)
    chunks = remote.stream_complete(_MSGS, backend="hosted_k3")
    out["e2e_stream_returns_list"] = isinstance(chunks, list) and chunks == ["tok-a", "tok-b"]

    # keepalived post-commit failure → in-band error frame → mapped, code
    # intact (defect-fix pin).
    slow_fail = _FailBackend(sleep_s=0.4)
    client_k, _ = _client({"byok": slow_fail}, sse_keepalive_s=0.05)
    exc = _exc(lambda: _remote(client_k).stream_complete(_MSGS, backend="byok"))
    out["e2e_keepalive_error_frame_mapped"] = type(exc).__name__ == "BackendNotConfiguredError"
    out["e2e_keepalive_error_code"] = getattr(exc, "code", "") == "backend_unavailable"

    # e2e job stream: submit a fast job, then follow it — last frame is
    # terminal and wait_run_stream maps it to the result.
    job_id = remote.submit_run("doctor")
    frames = remote.stream_job(job_id)
    res = remote.wait_run_stream(job_id)
    out["e2e_job_stream_terminal"] = (
        frames[-1]["status"] in ("succeeded", "failed") and res.exit_code == 0
    )
    return out


def _idem_probes() -> dict[str, bool]:  # NOSONAR(S3776)
    """Idempotency-Key interplay: dedup on same-body retry, 409 on
    conflict, a transport-fault retry still producing exactly one job."""
    out: dict[str, bool] = {}
    from fx1.serve.client import HarnessTransportError  # noqa: PLC0415

    ran: list[list[str]] = []

    def count_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:  # NOSONAR(S1172)
        ran.append(list(argv))
        return 0, "ok", ""

    stub = _StubBackend()
    client, _ = _client({"hosted_k3": stub}, runner=count_runner)
    remote = _remote(client)

    j1 = remote.submit_run("doctor", idempotency_key="idem-a")
    j2 = remote.submit_run("doctor", idempotency_key="idem-a")
    jobs = remote.list_jobs()
    out["submit_same_key_dedupes"] = j1 == j2 and jobs["total"] == 1 and len(ran) <= 1
    exc = _exc(lambda: remote.submit_run("doctor", ["--different"], idempotency_key="idem-a"))
    out["idem_conflict_409_mapped"] = (
        type(exc).__name__ == "HarnessTransportError"
        and getattr(exc, "code", "") == "idempotency_conflict"
    )
    r1 = remote.run("doctor", idempotency_key="idem-run")
    n_after_first = len(ran)
    r2 = remote.run("doctor", idempotency_key="idem-run")
    out["run_same_key_executes_once"] = (
        r1.exit_code == r2.exit_code == 0 and len(ran) == n_after_first
    )
    remote.run("doctor")
    remote.run("doctor")
    out["auto_minted_keys_do_not_dedup"] = len(ran) == n_after_first + 2

    c1 = remote.complete(_MSGS, backend="hosted_k3", idempotency_key="idem-comp")
    calls_after_first = stub.calls
    c2 = remote.complete(_MSGS, backend="hosted_k3", idempotency_key="idem-comp")
    out["complete_same_key_replays"] = (
        c2.replayed is True and stub.calls == calls_after_first and c1.content == c2.content
    )

    # a transport-fault retry under one key still produces exactly one
    # job — the "server sees exactly one claim" pin.
    real_send = _tc_transport(client)
    state = {"n": 0}

    def flaky(
        method: str, url: str, payload: Any, headers: dict[str, str], timeout_s: float
    ) -> Any:
        state["n"] += 1
        res = real_send(method, url, payload, headers, timeout_s)
        if state["n"] == 1:
            # the work landed server-side; the wire answer was lost.
            raise HarnessTransportError("response lost on the wire")
        return res

    tr_client = _mk(flaky, max_retries=2, retry_writes=True, sleep=lambda s: None)
    jid = tr_client.submit_run("doctor", idempotency_key="idem-flaky")
    jobs2 = remote.list_jobs()
    out["retry_fault_server_one_claim"] = (
        jobs2["total"] == 2 and isinstance(jid, str) and jid == jobs2["jobs"][0]["job_id"]
    )
    out["retry_fault_wire_saw_two_posts"] = state["n"] == 2
    return out


def _auth_probes() -> dict[str, bool]:  # NOSONAR(S3776) — credential matrix
    """Auth edges: missing/wrong/revoked/scope-limited creds →
    HarnessAuthError; quota refusals fail fast; window refusals retry."""
    out: dict[str, bool] = {}
    client, _ = _client({"hosted_k3": _StubBackend()}, api_key=_ROOT)
    root_h = {"X-API-Key": _ROOT}

    exc = _exc(lambda: _remote(client).commands())
    out["no_key_401_auth"] = (
        type(exc).__name__ == "HarnessAuthError" and getattr(exc, "code", "") == "unauthorized"
    )
    exc = _exc(lambda: _remote(client, api_key="fx1k_bad").commands())
    out["bad_key_401_auth"] = type(exc).__name__ == "HarnessAuthError"
    out["good_key_passes"] = _remote(client, api_key=_ROOT).commands() != []

    raw, kid = _mint(client, root_h, name="probe-key")
    out["minted_key_passes"] = _remote(client, api_key=raw).commands() != []
    r = client.delete(f"/harness/keys/{kid}", headers=root_h)
    exc = _exc(lambda: _remote(client, api_key=raw).commands())
    out["revoked_key_401"] = r.status_code == 200 and type(exc).__name__ == "HarnessAuthError"

    rraw, _rid = _mint(client, root_h, name="read-only", scopes=["read"])
    scoped = _remote(client, api_key=rraw)
    out["read_scope_reads"] = scoped.commands() != []
    exc = _exc(lambda: scoped.complete(_MSGS, backend="hosted_k3"))
    out["read_scope_write_403_auth"] = type(exc).__name__ == "HarnessAuthError"

    # hard budget: 429 quota_exceeded with no Retry-After must not retry.
    qraw, _qid = _mint(client, root_h, name="quota-key", max_requests=1)
    sleeps: list[float] = []
    qremote = _remote(client, api_key=qraw, max_retries=3, sleep=lambda s: sleeps.append(s))
    qremote.commands()
    exc = _exc(qremote.commands)
    out["quota_429_fails_fast"] = (
        type(exc).__name__ == "HarnessTransportError"
        and getattr(exc, "code", "") == "quota_exceeded"
        and sleeps == []
    )

    # per-key rate window: 429 rate_limited carries the ~60s window in
    # Retry-After — inside the wait budget the client retries once and
    # the persisted refusal surfaces mapped; past it the client breaks
    # fast rather than park a call for a minute.
    lraw, _lid = _mint(client, root_h, name="rpm-key", rpm=1)
    sleeps2: list[float] = []
    lremote = _remote(
        client,
        api_key=lraw,
        max_retries=1,
        max_retry_wait_s=70,
        sleep=lambda s: sleeps2.append(s),
    )
    lremote.commands()
    exc = _exc(lremote.commands)
    out["key_rpm_429_retries_within_budget"] = (
        type(exc).__name__ == "HarnessTransportError"
        and getattr(exc, "code", "") == "rate_limited"
        and sleeps2 == [60.0]
    )
    lraw2, _lid2 = _mint(client, root_h, name="rpm-key-2", rpm=1)
    sleeps3: list[float] = []
    lremote2 = _remote(client, api_key=lraw2, max_retries=3, sleep=lambda s: sleeps3.append(s))
    lremote2.commands()
    exc = _exc(lremote2.commands)
    out["key_rpm_429_breaks_over_budget"] = (
        type(exc).__name__ == "HarnessTransportError"
        and "exhausted 3 retries" in str(exc)
        and sleeps3 == []
    )
    return out


def _cli_probes() -> dict[str, bool]:  # NOSONAR(S3776) — CLI legs each pin exit code + stderr shape
    """``fx1 harness`` legs: wire refusals are one clean ``error:`` line
    on stderr + exit 2, stdout stays machine-readable, in-process and
    remote legs agree."""
    out: dict[str, bool] = {}
    from unittest import mock  # noqa: PLC0415

    from typer.testing import CliRunner  # noqa: PLC0415

    import fx1.serve.client as client_mod  # noqa: PLC0415
    from fx1.cli import app  # noqa: PLC0415

    runner = CliRunner()

    r = runner.invoke(app, ["harness", "metrics"])
    out["metrics_needs_remote_2"] = r.exit_code == 2 and "error:" in r.stderr
    r = runner.invoke(app, ["harness", "submit", "doctor"])
    out["submit_needs_remote_2"] = r.exit_code == 2 and "error:" in r.stderr
    r = runner.invoke(app, ["harness", "job", "no-such"])
    out["job_needs_remote_2"] = r.exit_code == 2 and "error:" in r.stderr

    # defect-fix pins: in-process legs map refusals through _or_exit —
    # clean error line + exit 2, never a traceback exit 1.
    r = runner.invoke(app, ["harness", "run", "no-such-command"])
    out["run_unknown_inprocess_clean_2"] = (
        r.exit_code == 2 and r.stderr.startswith("error: KeyError") and "Traceback" not in r.output
    )
    r = runner.invoke(app, ["harness", "list", "--role", "bogus"])
    out["list_bogus_role_clean_2"] = (
        r.exit_code == 2 and r.stderr.startswith("error:") and "Traceback" not in r.output
    )
    r = runner.invoke(app, ["harness", "commands", "--role", "bogus"])
    out["commands_bogus_role_2"] = r.exit_code == 2 and r.stderr.startswith("error:")

    # remote refusal through a real authed app: the caller has no key →
    # 401 → HarnessAuthError → one error line, exit 2, stdout untouched.
    client_a, _ = _client({"hosted_k3": _StubBackend()}, api_key=_ROOT)
    remote_nokey = _remote(client_a)
    with mock.patch.object(client_mod, "HarnessClient", lambda *a, **k: remote_nokey):
        r = runner.invoke(app, ["harness", "commands", "--remote", "http://x"])
    out["remote_auth_refusal_stderr_2"] = (
        r.exit_code == 2 and r.stderr.startswith("error: HarnessAuthError") and r.stdout == ""
    )
    # unreachable remote → HarnessTransportError → same shape.
    import fx1.serve.client as _cm  # noqa: PLC0415

    def _dial(
        method: str, url: str, payload: Any, headers: dict[str, str], timeout_s: float
    ) -> Any:  # NOSONAR(S1172)
        raise _cm.HarnessTransportError("dial failed")

    dead_c = _mk(_dial)
    with mock.patch.object(client_mod, "HarnessClient", lambda *a, **k: dead_c):
        r = runner.invoke(app, ["harness", "commands", "--remote", "http://x"])
    out["remote_transport_fault_stderr_2"] = (
        r.exit_code == 2 and r.stderr.startswith("error: HarnessTransportError") and r.stdout == ""
    )
    dead2 = _mk(_dial)
    with mock.patch.object(client_mod, "HarnessClient", lambda *a, **k: dead2):
        r = runner.invoke(app, ["harness", "job", "j1", "--remote", "http://x"])
    out["remote_job_refusal_2"] = r.exit_code == 2 and r.stderr.startswith("error:")

    # --remote vs in-process parity on the same op: the commands set is
    # identical whichever leg serves it.
    client_b, _ = _client({"hosted_k3": _StubBackend()})
    remote_b = _remote(client_b)
    with mock.patch.object(client_mod, "HarnessClient", lambda *a, **k: remote_b):
        r_rem = runner.invoke(app, ["harness", "commands", "--remote", "http://x"])
    r_loc = runner.invoke(app, ["harness", "commands"])
    names_rem = json.loads(r_rem.stdout) if r_rem.exit_code == 0 else {}
    names_loc = json.loads(r_loc.stdout) if r_loc.exit_code == 0 else {}
    out["remote_inprocess_commands_parity"] = (
        r_rem.exit_code == 0
        and r_loc.exit_code == 0
        and sorted(names_rem.get("commands", [])) == sorted(names_loc.get("commands", []))
        and len(names_loc.get("commands", [])) > 0
    )
    r = runner.invoke(app, ["harness", "commands"])
    out["commands_stdout_machine_readable"] = r.exit_code == 0 and isinstance(
        json.loads(r.stdout)["commands"], list
    )
    return out


def _parity_probes() -> dict[str, bool]:
    """In-process SDK vs remote client: same refusal → same exception
    class wherever both legs implement the surface."""
    out: dict[str, bool] = {}
    from fastapi.testclient import TestClient as _TC  # noqa: PLC0415

    import fx1.serve.api as api_mod  # noqa: PLC0415
    from fx1.harness import Harness  # noqa: PLC0415
    from fx1.sdk import Fx1Harness  # noqa: PLC0415

    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:  # NOSONAR(S1172)
        return 0, "ok", ""

    def surfaces(backend_map: dict[str, Any]) -> tuple[Any, Any]:
        """(Fx1Harness, remote HarnessClient) over one resolver map —
        enum-valid names only reach the resolver; an unmapped name is the
        wire's 404 and the SDK's KeyError alike."""

        def resolver(name: str, **kw: Any) -> Any:
            if name not in backend_map:
                raise KeyError(name)
            return backend_map[name]()

        saved = {k: os.environ.get(k) for k in _SWEPT_ENVS}
        try:
            for k in _SWEPT_ENVS:
                os.environ.pop(k, None)
            # the key-lifecycle routes are admin-gated once a key exists —
            # the remote leg carries the root credential like any real caller.
            os.environ[_API_KEY_ENV] = _ROOT
            sdk = Fx1Harness(harness=Harness(runner=fake_runner), backend_resolver=resolver)
            app = api_mod.create_app(harness=Harness(runner=fake_runner), backend_resolver=resolver)
            remote = _remote(_TC(app, raise_server_exceptions=False), api_key=_ROOT)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        return sdk, remote

    sdk, remote = surfaces({"byok": _StubBackend})
    out["key_get_unknown_parity"] = (
        type(_exc(lambda: sdk.key_get("kid_nope"))).__name__ == "KeyError"
        and type(_exc(lambda: remote.key_get("kid_nope"))).__name__ == "KeyError"
    )
    out["unknown_backend_parity"] = (
        type(_exc(lambda: sdk.complete(_MSGS, backend="hosted_k3"))).__name__ == "KeyError"
        and type(_exc(lambda: remote.complete(_MSGS, backend="hosted_k3"))).__name__ == "KeyError"
    )
    out["local_fx1_checkpoint_parity"] = (
        type(_exc(lambda: sdk.complete(_MSGS, backend="local_fx1"))).__name__ == "ValueError"
        and type(_exc(lambda: remote.complete(_MSGS, backend="local_fx1"))).__name__ == "ValueError"
    )
    sdk_ns, remote_ns = surfaces({"byok": _NoStreamBackend})
    out["no_stream_501_parity"] = (
        type(_exc(lambda: sdk_ns.stream_complete(_MSGS, backend="byok"))).__name__
        == "NotImplementedError"
        and type(_exc(lambda: remote_ns.stream_complete(_MSGS, backend="byok"))).__name__
        == "NotImplementedError"
    )
    sdk_d, remote_d = surfaces({"byok": _DirtyBackend})
    out["honesty_gate_parity"] = (
        type(_exc(lambda: sdk_d.complete(_MSGS, backend="byok"))).__name__ == "Fx1HonestyError"
        and type(_exc(lambda: remote_d.complete(_MSGS, backend="byok"))).__name__
        == "Fx1HonestyError"
    )
    # pinned divergence: revoke-on-revoked is a state conflict — the SDK
    # folds it to ValueError while the wire's 409 lands on the
    # else-branch HarnessTransportError. Both are honest; the classes
    # differ and this probe documents that.
    minted = sdk.key_create(name="rev-twice")
    sdk.key_revoke(minted["id"])
    sdk_cls = type(_exc(lambda: sdk.key_revoke(minted["id"]))).__name__
    remote_minted = remote.key_create(name="rev-twice-2")
    remote.key_revoke(remote_minted["id"])
    remote_cls = type(_exc(lambda: remote.key_revoke(remote_minted["id"]))).__name__
    out["revoke_twice_divergence_pinned"] = (
        sdk_cls == "ValueError" and remote_cls == "HarnessTransportError"
    )
    # auth is a wire-only surface: the in-process SDK trusts the process
    # owner and never raises HarnessAuthError.
    from fx1.serve.client import HarnessAuthError  # noqa: PLC0415

    out["sdk_has_no_auth_surface"] = not hasattr(sdk, "authenticate") and issubclass(
        HarnessAuthError, PermissionError
    )
    return out


def _cancel_probes() -> dict[str, bool]:
    """Cancellation + abandonment: a dropped waiter leaves the record
    untouched — the server does not infer disconnect. Job cancel is
    queued-only by design (no mid-run kill handle): a running job 409s
    honestly, a finished one 409s terminal. A background response does
    carry a cooperative cancel — it lands 'cancelled' mid-flight and
    refuses honestly once terminal."""
    out: dict[str, bool] = {}
    # The worker claims the job (status running) and then calls the
    # runner. A wall-clock sleep can finish in the gap between that
    # observation and wait_run's first poll when the suite is under
    # coverage, so the abandoned waiter sees success instead of a
    # timeout. Hold the runner until those observations are recorded.
    release_runner = threading.Event()

    def slow_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:  # NOSONAR(S1172)
        release_runner.wait(30.0)
        return 0, "ok", ""

    try:
        client, _ = _client({"hosted_k3": _StubBackend()}, runner=slow_runner)
        remote = _remote(client)

        job_id = remote.submit_run("doctor")
        # wait for the worker to claim it (bounded), then abandon the poll.
        st = remote.job_status(job_id)
        for _ in range(200):
            if st["status"] != "queued":
                break
            time.sleep(0.01)
            st = remote.job_status(job_id)
        out["job_reaches_running"] = st["status"] == "running"
        exc = _exc(lambda: remote.wait_run(job_id, poll_s=0.05, timeout_s=0.2))
        out["abandoned_poll_times_out"] = type(exc).__name__ == "HarnessTransportError"
        st2 = remote.job_status(job_id)
        out["abandoned_record_still_running"] = (
            st2["status"] == "running" and st2["job_id"] == job_id
        )
        exc = _exc(lambda: remote.cancel_job(job_id))
        out["cancel_running_409_mapped"] = type(
            exc
        ).__name__ == "HarnessTransportError" and "is running" in str(exc)
        release_runner.set()
        out["job_runs_to_end"] = remote.wait_run(job_id, poll_s=0.1, timeout_s=10).ok
        exc = _exc(lambda: remote.cancel_job(job_id))
        out["cancel_terminal_409_mapped"] = type(exc).__name__ == "HarnessTransportError"
        rec = remote.job_receipt(job_id)
        out["finished_job_receipt_sealed"] = isinstance(rec, dict) and isinstance(
            rec.get("receipt_sha256"), str
        )
    finally:
        release_runner.set()

    # a mid-stream abandonment: wait_run_stream on a job whose stream
    # ends pre-terminal errors honestly while the record runs.
    jrec = b'data: {"job_id":"j1","status":"running"}\n\n'
    tr, _ = _scripted((200, {}, jrec))
    exc = _exc(lambda: _mk(tr).wait_run_stream("j1"))
    out["stream_abandon_errors"] = type(exc).__name__ == "HarnessTransportError"

    # background response: create + abandon + cancel; the record reports
    # cancelled and the replay ends on a terminal cancelled frame.
    # Hold the model call until those reads land — a 0.6s sleep can
    # finish first when the worker is scheduled ahead of the client.
    release_bg = threading.Event()
    slow2 = _StubBackend(release=release_bg)
    client2, _ = _client({"byok": slow2})
    remote2 = _remote(client2)
    try:
        env, _cid3 = remote2.responses_create("hi", backend="byok", background=True)
        rid = env["id"]
        cancelled_r = remote2.cancel_response(rid)
        rec2 = remote2.retrieve_response(rid)
        out["bg_response_cancel_record"] = (
            cancelled_r["status"] == "cancelled" and rec2["status"] == "cancelled"
        )
        exc = _exc(lambda: remote2.cancel_response(rid))
        out["cancel_terminal_response_409"] = type(exc).__name__ == "HarnessTransportError"
        evs, _cid = remote2.responses_replay(rid)
        out["cancelled_replay_terminal_frame"] = evs[-1]["type"] in (
            "response.cancelled",
            "response.failed",
            "response.completed",
        )
    finally:
        release_bg.set()
    exc = _exc(lambda: remote2.cancel_response("resp_nope"))
    out["cancel_unknown_response_404"] = type(exc).__name__ == "KeyError"
    return out


def client_audit() -> dict[str, Any]:
    """Run the client-boundary battery; returns literal bools."""
    out: dict[str, Any] = {}
    out.update(_error_map_probes())
    out.update(_retry_probes())
    out.update(_timeout_probes())
    out.update(_transport_fault_probes())
    out.update(_stream_probes())
    out.update(_idem_probes())
    out.update(_auth_probes())
    out.update(_cli_probes())
    out.update(_parity_probes())
    out.update(_cancel_probes())
    return out


def client_audit_bench(results: dict[str, Any] | None = None) -> dict[str, Any]:
    """Seal client-audit results; run the battery when results are omitted."""
    r = client_audit() if results is None else dict(results)
    ok = bool(r) and all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "client_audit",
        "schema": "client_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "The HarnessClient boundary honors its declared taxonomy end "
            "to end: every status class maps to its declared exception "
            "carrying the server code and detail through both error "
            "envelopes; retries fire only on Retry-After-bearing 429/503 "
            "or transport faults on retryable calls, bounded by "
            "max_retries and max_retry_wait_s; timeouts and socket "
            "faults are honest HarnessTransportErrors; every SSE surface "
            "maps in-band error frames (code included) and enforces its "
            "terminal frame; Idempotency-Key retries collapse to one "
            "server-side execution while conflicting bodies 409; auth "
            "refusals (missing, wrong, revoked, scope-limited) land on "
            "HarnessAuthError; a hard quota 429 fails fast while a "
            "window 429 retries; the fx1 harness CLI surfaces refusals "
            "as one clean error line + exit 2 with stdout left "
            "machine-readable; the in-process SDK raises the same "
            "exception classes as the remote leg for every refusal both "
            "implement — the one documented divergence (revoke-on-"
            "revoked: ValueError in-process, 409 HarnessTransportError "
            "on the wire) is pinned, not hidden. An abandoned poll "
            "leaves the server record honest and a late cancel still "
            "lands. Three defects were found and fixed in this lane: "
            "wait_message_batch ignored the injected clock/sleep pair "
            "its sibling waiters honor; stream_complete dropped the "
            "server's machine code from in-band SSE error frames; the "
            "in-process harness run/list CLI legs bypassed _or_exit and "
            "tracebacked exit 1 on registry refusals."
            if ok
            else f"CLIENT AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(client_audit_bench(), indent=2, sort_keys=True))
