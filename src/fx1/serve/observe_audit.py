"""observe_audit — observability/ops-instrumentation battery for the fx-1 serve surface.

Where ``api_audit`` pins wire shapes and ``usage_audit`` reconciles the
billing ledger, this battery pins the *operational* contract an
orchestrator runs against: liveness/readiness separation, the metrics
snapshot, the Prometheus exposition, the drain latch, request-id
propagation, the error envelope, and the auth/scope boundary for exactly
who can observe what.

Probe map:

- *Liveness vs readiness* — ``GET /health`` is the only public path: it
  answers 200 under no key, a wrong key, and a latched drain, and it
  reports backend *configuration* as booleans (never values). ``GET
  /ready`` is the scheduler signal: 200 while accepting, 503
  ``draining`` once the latch is set — and it is *not* public: under a
  keyed deployment it demands a credential.
- *JSON metrics* — ``requests_total`` counts every response (including
  refused auth and the scrape itself), ``by_status`` partitions the
  total exactly, ``errors_total`` equals the >=400 share, counters are
  monotonic for the process lifetime, the inflight gauge rises and
  returns to zero while ``inflight_watermark`` keeps the peak, and the
  per-backend ``complete`` ledger reconciles 1:1 with ``/harness/usage``
  for the same traffic window — same request counts, same token sums,
  same ``usage_calls`` as ``usage_reported``.
- *Prometheus exposition* — ``?format=prom``/``?format=prometheus`` and
  ``Accept: text/plain``/OpenMetrics negotiate the text view; every
  sample belongs to a declared TYPE family, HELP precedes TYPE precedes
  samples, no NaN/Inf leaks into a value, histogram ``le`` buckets are
  cumulative with ``+Inf == _count``, and label escaping round-trips a
  hostile backend name byte-for-byte. ``?format=<garbage>`` refuses 422
  inside the envelope.
- *Drain lifecycle* — ``POST /harness/drain`` is admin-scope only; the
  latch flips ``/ready`` to 503 and refuses new gated work with
  enveloped 503 ``draining`` while GETs, ``/health``, ``/metrics``, and
  already-running requests stay alive; ``wait_s`` blocks until the pool
  empties (``drained``) or lapses; the latch is one-way and idempotent
  under parallel POSTs; and it is *process-local* — a restart over the
  same ``--state-dir`` comes back ready while journaled state (keys,
  jobs) survives.
- *Request-id propagation* — a well-formed ``X-Request-ID`` is echoed on
  success AND on every refusal (401/403/404/422/429/503); absent or
  malformed ids mint a fresh one matching the documented charset; the
  Anthropic dialect also stamps ``request-id``.
- *Error envelope* — hostile inputs across the ops surface (bogus
  format, out-of-range ``wait_s``, NaN bounds, unknown routes, wrong
  methods, oversized/declared-garbage bodies) all land in the
  ``{"detail","code"}`` envelope — never a bare 500.
- *Concurrency* — parallel reads and completes lose no increment, the
  inflight gauge and watermark track a held burst exactly, and parallel
  drains are idempotent.
- *Client surface* — ``HarnessClient.health``/``metrics``/
  ``metrics_text``/``ready``/``drain`` ride the same wire contract, and
  the client's documented error map holds: 401/403 ->
  ``HarnessAuthError``, 404 -> ``KeyError``, 422 -> ``ValueError``,
  503 -> ``BackendNotConfiguredError``.

Probes are literal bools: ``True`` pins a contract that holds; ``False``
pins a measured divergence — the sealed receipt names every defect by
probe name so the finding survives byte-for-byte.

Sealed ``observe_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import threading
import time
import urllib.parse
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from types import ModuleType

    from fastapi.testclient import TestClient

    from fx1.serve.backends import SamplingParams

__all__ = ["observe_audit", "observe_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT = "obs3rve-root"
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
)
_N = 8  # racer width — wide enough that a lost increment shows
_U = {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}
_U2 = {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
_RID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_PROM_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{([^}]*)\})?\s+(\S+)\s*$")
_HIST_SUFFIXES = ("_bucket", "_sum", "_count")


_RESOURCE_STACK: ContextVar[ExitStack | None] = ContextVar("observe_audit_resources", default=None)


@contextmanager
def _cleared_backend_env() -> Iterator[None]:
    """Drop credential env for one request, then restore it.

    ``_backend_configured`` reads the process environment on every
    ``/health``. ``_make_app`` only clears those variables while the app
    is constructed, so a sibling test that left ``MOONSHOT_API_KEY`` or
    ``FX1_BYOK_*`` set makes the fail-closed probe see a configured
    backend. The probe is about the unconfigured state.
    """
    saved = {name: os.environ.get(name) for name in _SWEPT_ENVS}
    for name in _SWEPT_ENVS:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _audit_resources() -> Iterator[None]:
    """Close every client and worker pool even when a probe raises."""
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


def _test_client(app: Any) -> TestClient:
    from fastapi.testclient import TestClient

    stack = _resources()
    client = TestClient(app, raise_server_exceptions=False)
    # TestClient.__exit__ handles lifespan but does not close HTTPX here.
    stack.callback(client.close)
    return stack.enter_context(client)


def _fast_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    return 0, "ok", ""


class _StubBackend:
    """Deterministic completion stub: returns content + reports usage."""

    def __init__(self, model: str, usage: dict[str, int] | None = None) -> None:
        self._model = model
        self._usage = dict(usage) if usage else None
        self.calls = 0
        self.last_usage: dict[str, int] | None = None
        self._lock = threading.Lock()

    def _report(self) -> None:
        self.last_usage = dict(self._usage) if self._usage is not None else None

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        sampling: SamplingParams | None = None,  # NOSONAR(S1172)
    ) -> str:
        with self._lock:
            self.calls += 1
        self._report()
        return f"ok:{messages[-1]['content']}"

    def stream(
        self,
        messages: list[dict[str, str]],  # NOSONAR(S1172) — protocol signature
        *,
        sampling: SamplingParams | None = None,  # NOSONAR(S1172)
    ) -> Any:
        with self._lock:
            self.calls += 1
        self._report()
        yield "tok-a"
        yield "tok-b"

    def close(self) -> None:
        """No resources to release — the stub holds nothing."""


class _GateBackend(_StubBackend):
    """A backend whose calls park until released — deterministic
    in-flight occupancy for the drain/gauge probes (no sleeps, no
    races: ``all_entered`` is set by the Nth concurrent call)."""

    def __init__(self, model: str, width: int, usage: dict[str, int] | None = None) -> None:
        super().__init__(model, usage)
        self._width = width
        self._inside = 0
        self.all_entered = threading.Event()
        self.release = threading.Event()

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        sampling: SamplingParams | None = None,  # NOSONAR(S1172)
    ) -> str:
        with self._lock:
            self.calls += 1
            self._inside += 1
            if self._inside >= self._width:
                self.all_entered.set()
        self.release.wait(timeout=60.0)
        with self._lock:
            self._inside -= 1
        self._report()
        return f"ok:{messages[-1]['content']}"


def _make_app(
    backend_map: dict[str, Any] | None = None,
    api_key: str | None = _ROOT,
    **create_kw: Any,
) -> Any:
    """create_app under an isolated env; backends resolve from
    ``backend_map[name]`` zero-arg factories (a factory may itself raise
    to model an unconfigured link)."""
    import fx1.serve.api as api_mod  # noqa: PLC0415
    from fx1.harness import Harness  # noqa: PLC0415

    backends = backend_map or {"byok": lambda: _StubBackend("m-0", dict(_U))}

    def fake_resolve(name: str, *a: Any, **k: Any) -> Any:
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
    backend_map: dict[str, Any] | None = None,
    api_key: str | None = _ROOT,
    **create_kw: Any,
) -> tuple[TestClient, ModuleType]:
    """(TestClient, api_module) — the battery's standard wired app."""
    import fx1.serve.api as api_mod

    app = _make_app(backend_map, api_key, **create_kw)
    return _test_client(app), api_mod


def _mint(client: TestClient, root_h: dict[str, str], **policy: Any) -> tuple[str, str]:
    """Mint a managed key → (raw, key_id)."""
    r = client.post("/harness/keys", json=policy, headers=root_h)
    assert r.status_code == 201, r.text
    body = r.json()
    return str(body["key"]), str(body["id"])


def _complete(
    client: TestClient,
    backend: str,
    headers: dict[str, str] | None = None,
    **extra: Any,
) -> Any:
    return client.post(
        "/harness/complete",
        json={"backend": backend, "messages": [{"role": "user", "content": "hi"}], **extra},
        headers=headers or {},
    )


def _metrics(client: TestClient, headers: dict[str, str]) -> dict[str, Any]:
    r = client.get("/metrics", headers=headers)
    assert r.status_code == 200, r.text
    body: dict[str, Any] = r.json()
    return body


def _usage(client: TestClient, headers: dict[str, str]) -> dict[str, Any]:
    r = client.get("/harness/usage", headers=headers)
    assert r.status_code == 200, r.text
    body: dict[str, Any] = r.json()
    return body


def _enveloped(resp: Any) -> bool:
    """The wire-code contract: every non-2xx body carries a machine code —
    ``{"detail","code"}`` off /v1, the OpenAI ``{error:{code}}`` or
    Anthropic ``{type:"error"}`` grammar on it."""
    if resp.status_code < 400 or resp.status_code >= 500:
        return False
    try:
        body = resp.json()
    except ValueError:
        return False
    if isinstance(body, dict):
        if isinstance(body.get("code"), str) and "detail" in body:
            return True
        err = body.get("error")
        if isinstance(err, dict) and ("code" in err or "type" in err):
            return True
        if body.get("type") == "error" and isinstance(body.get("error"), dict):
            return True
    return False


def _raw_http(
    app: Any,
    method: str,
    path: str,
    headers: list[tuple[str, bytes]],
    body: bytes = b"",
    client_host: str = "127.0.0.1",
) -> dict[str, Any]:
    """Drive the ASGI app directly — for clients TestClient can't model
    (non-loopback source hosts, forged Content-Length declarations)."""
    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": path.partition("?")[2].encode(),
        "root_path": "",
        "headers": [(k.lower().encode(), v) for k, v in headers],
        "client": (client_host, 5000),
        "server": ("test", 80),
    }
    scope["path"] = path.partition("?")[0]
    resp: dict[str, Any] = {}

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(msg: dict[str, Any]) -> None:
        if msg["type"] == "http.response.start":
            resp["status"] = msg["status"]
            resp["headers"] = {k.decode().lower(): v.decode() for k, v in msg.get("headers", [])}
        elif msg["type"] == "http.response.body":
            resp.setdefault("body", b"")
            resp["body"] += msg.get("body", b"")

    asyncio.run(app(scope, receive, send))
    return resp


class _Prom:
    """Minimal 0.0.4 exposition parser — family -> {help_idx, type_idx,
    samples: [(series_name, labels, value_str)]}. Enough to assert the
    invariant set the battery pins (ordering, declared types, finite
    values, cumulative histograms, label round-trip)."""

    def __init__(self, text: str) -> None:
        self.families: dict[str, dict[str, Any]] = {}
        self.bad_lines: list[str] = []
        for i, raw in enumerate(text.splitlines()):
            if raw.startswith("# HELP "):
                name = raw.split(" ", 3)[2]
                self.families.setdefault(name, {"samples": []})["help_idx"] = i
            elif raw.startswith("# TYPE "):
                parts = raw.split(" ")
                name, mtype = parts[2], parts[3]
                fam = self.families.setdefault(name, {"samples": []})
                fam["type_idx"] = i
                fam["type"] = mtype
            elif raw.startswith("#"):
                continue
            else:
                m = _PROM_SAMPLE.match(raw)
                if m is None:
                    self.bad_lines.append(raw)
                    continue
                series, labels = m.group(1), self._labels(m.group(3))
                base = series
                if series.endswith(_HIST_SUFFIXES):
                    base = series.rsplit("_", 1)[0]
                fam = self.families.setdefault(base, {"samples": []})
                fam.setdefault("first_sample_idx", i)
                fam["samples"].append((series, labels, m.group(4)))

    @staticmethod
    def _labels(raw: str | None) -> dict[str, str]:
        out: dict[str, str] = {}
        if not raw:
            return out
        for part in re.split(r",(?=[a-zA-Z_][a-zA-Z0-9_]*=)", raw):
            k, _, v = part.partition("=")
            v = v.strip('"')
            out[k] = v.replace("\\\\", "\\").replace('\\"', '"').replace("\\n", "\n")
        return out

    def finite(self) -> bool:
        """Every sample value parses as a finite float — NaN/±Inf never
        leak into an exposition value."""
        for fam in self.families.values():
            for _series, _labels, val in fam["samples"]:
                try:
                    x = float(val)
                except ValueError:
                    return False
                if x != x or x in (float("inf"), float("-inf")):
                    return False
        return True

    def ordered(self) -> bool:
        """HELP precedes TYPE precedes first sample, per family."""
        for fam in self.families.values():
            samples = fam["samples"]
            if not samples:
                continue
            t_idx = fam.get("type_idx")
            h_idx = fam.get("help_idx")
            first_idx = fam["first_sample_idx"]
            if t_idx is None or h_idx is None or not (h_idx < t_idx < first_idx):
                return False
        return True

    def all_typed(self) -> bool:
        """Every sampled family has a declared TYPE line."""
        return (
            all("type" in fam for fam in self.families.values() if fam["samples"])
            and not self.bad_lines
        )

    def counter(self, name: str, labels: dict[str, str] | None = None) -> float | None:
        fam = self.families.get(name)
        if not fam:
            return None
        for _series, lab, val in fam["samples"]:
            if labels is None or lab == labels:
                return float(val)
        return None


def _wait_job(
    client: TestClient, job_id: str, headers: dict[str, str], timeout: float = 30.0
) -> dict[str, Any]:
    """Poll a job to terminal state."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/harness/jobs/{job_id}", headers=headers)
        body: dict[str, Any] = r.json()
        if body["status"] in ("succeeded", "failed", "cancelled"):
            return body
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not finish")


# ---------------------------------------------------------------------------
# liveness / readiness / scope boundary
# ---------------------------------------------------------------------------


def _probe_liveness() -> dict[str, bool]:
    """Health is liveness (public, always up); ready is readiness
    (keyed, fails closed under drain and off the public path)."""
    out: dict[str, bool] = {}
    client, _ = _client()
    root = {"X-API-Key": _ROOT}

    out["health_public_unauthenticated"] = client.get("/health").status_code == 200
    out["health_public_wrong_key"] = (
        client.get("/health", headers={"X-API-Key": "fx1k_deadbeef"}).status_code == 200
    )
    with _cleared_backend_env():
        h = client.get("/health").json()
    out["health_shape"] = (
        h.get("status") == "ok"
        and isinstance(h.get("registered_commands"), int)
        and h["registered_commands"] > 0
        and set(h.get("backends", {})) >= {"hosted_k3", "byok", "local_fx1"}
        and all(isinstance(v, bool) for v in h["backends"].values())
        and isinstance(h.get("draining"), bool)
    )
    # Flags are read live. With credential env removed for this request
    # they must report nothing configured, not whatever a sibling left set.
    out["health_backends_fail_closed_unconfigured"] = all(
        v is False for v in h["backends"].values()
    )
    # HEAD is not registered on the liveness route — the refusal is a
    # measured 405 (the envelope body is correctly suppressed on HEAD per
    # HTTP semantics; the same handler's enveloped 405 is pinned on
    # POST /metrics in the error-taxonomy section)
    r_head = client.head("/health")
    out["health_head_405_enveloped"] = r_head.status_code == 405 and r_head.headers.get(
        "content-type", ""
    ).startswith("application/json")
    out["ready_requires_credential"] = client.get("/ready").status_code == 401
    out["metrics_requires_credential"] = client.get("/metrics").status_code == 401
    out["backends_status_requires_credential"] = client.get("/harness/backends").status_code == 401
    out["drain_requires_credential"] = client.post("/harness/drain").status_code == 401
    r = client.get("/ready", headers=root)
    out["ready_200_while_accepting"] = (
        r.status_code == 200
        and r.json().get("ready") is True
        and isinstance(r.json().get("inflight"), int)
    )
    out["metrics_200_with_root"] = client.get("/metrics", headers=root).status_code == 200
    return out


def _probe_scope_boundary() -> dict[str, bool]:
    """Exactly who can read/operate the ops surface — minted scope
    bindings are exact: read covers GETs, write covers completes, admin
    covers drain/keys; no scope silently implies another."""
    out: dict[str, bool] = {}
    client, _ = _client()
    root = {"X-API-Key": _ROOT}
    raw_r, _ = _mint(client, root, scopes=["read"])
    raw_w, _ = _mint(client, root, scopes=["write"])
    raw_a, _ = _mint(client, root, scopes=["admin"])
    raw_all, _ = _mint(client, root, admin=True)
    rh, wh, ah, allh = ({"X-API-Key": k} for k in (raw_r, raw_w, raw_a, raw_all))

    out["read_scope_reads_ops"] = (
        client.get("/metrics", headers=rh).status_code == 200
        and client.get("/ready", headers=rh).status_code == 200
        and client.get("/harness/backends", headers=rh).status_code == 200
        and client.get("/harness/usage", headers=rh).status_code == 200
        and client.get("/harness/commands", headers=rh).status_code == 200
    )
    out["read_scope_cannot_write_or_drain"] = (
        _complete(client, "byok", rh).status_code == 403
        and client.post("/harness/drain", headers=rh).status_code == 403
        and client.post("/harness/keys", json={}, headers=rh).status_code == 403
    )
    out["write_scope_cannot_read_metrics"] = (
        client.get("/metrics", headers=wh).status_code == 403
        and _complete(client, "byok", wh).status_code == 200
        and client.post("/harness/drain", headers=wh).status_code == 403
    )
    # admin alone is the control plane, not the data plane — it can
    # drain and mint but cannot read metrics or complete
    out["admin_scope_no_implied_read"] = (
        client.post("/harness/drain", headers=ah).status_code == 200
        and client.get("/metrics", headers=ah).status_code == 403
        and _complete(client, "byok", ah).status_code == 403
    )
    out["admin_flag_grants_full_surface"] = (
        client.get("/metrics", headers=allh).status_code == 200
        and client.post("/harness/drain", headers=allh).status_code == 200
    )
    return out


def _probe_dev_and_remote() -> dict[str, bool]:
    """No-key dev posture: loopback is trusted and bootstraps the key
    store; the first minted key closes the unauthenticated surface; a
    non-loopback client without a key is refused while /health stays
    public."""
    out: dict[str, bool] = {}
    client, _ = _client(api_key=None)
    out["dev_loopback_full_admin"] = (
        client.get("/metrics").status_code == 200
        and client.get("/ready").status_code == 200
        and client.post("/harness/drain").status_code == 200
    )
    # drain is a one-way latch whose refusal set includes the key
    # lifecycle, so the mint-close probe runs on a fresh app
    client2, _ = _client(api_key=None)
    # mint through the loopback admin surface, then unauthenticated reads
    # must refuse — first managed key closes the dev surface
    raw, _ = _mint(client2, {})
    out["first_mint_closes_dev_surface"] = (
        client2.get("/metrics").status_code == 401
        and client2.get("/metrics", headers={"X-API-Key": raw}).status_code == 200
        and client2.get("/health").status_code == 200
    )
    # non-loopback source host, no env key: refused except /health
    app2 = _make_app(api_key=None)
    out["remote_no_key_forbidden"] = (
        _raw_http(app2, "GET", "/metrics", [], client_host="198.51.100.7")["status"] == 403
    )
    out["remote_health_stays_public"] = (
        _raw_http(app2, "GET", "/health", [], client_host="198.51.100.7")["status"] == 200
    )
    return out


# ---------------------------------------------------------------------------
# /metrics — JSON snapshot integrity
# ---------------------------------------------------------------------------


def _probe_metrics_json() -> dict[str, bool]:
    """Every response counted, once; partitions reconcile; counters are
    process-lifetime monotonic; the per-backend ledger reconciles with
    the usage ring for the same window."""
    out: dict[str, bool] = {}
    client, _ = _client(
        {
            "byok": lambda: _StubBackend("m-0", dict(_U)),
            "local_fx1": lambda: _StubBackend("fx1-ckpt", dict(_U2)),
        }
    )
    root = {"X-API-Key": _ROOT}
    m0 = _metrics(client, root)

    # scripted traffic with a known status mix (the m0 scrape's own
    # record lands after its snapshot — it is part of this delta too)
    codes: Counter[int] = Counter({200: 1})
    for _i in range(3):
        r = _complete(client, "byok", root)
        codes[r.status_code] += 1
    r = _complete(client, "local_fx1", root, checkpoint_dir="synthetic-ckpt")
    codes[r.status_code] += 1
    r = client.get("/ready", headers=root)
    codes[r.status_code] += 1
    r = client.get("/health")
    codes[r.status_code] += 1
    r = client.get("/metrics")  # unauthenticated refusal
    codes[r.status_code] += 1
    r = client.get("/nope", headers=root)
    codes[r.status_code] += 1
    r = client.post("/harness/complete", json={"backend": "byok"}, headers=root)  # 422
    codes[r.status_code] += 1
    m1 = _metrics(client, root)  # m1's own record lands after the snapshot

    out["requests_total_counts_every_response"] = m1["requests_total"] - m0[
        "requests_total"
    ] == sum(codes.values())
    out["by_status_partitions_total_exactly"] = all(
        m1["by_status"].get(str(s), 0) - m0["by_status"].get(str(s), 0) == n
        for s, n in codes.items()
    )
    out["by_status_sums_to_total"] = sum(m1["by_status"].values()) == m1["requests_total"] and m1[
        "errors_total"
    ] == sum(v for k, v in m1["by_status"].items() if int(k) >= 400)
    out["counters_monotonic_no_reset"] = (
        m1["requests_total"] > m0["requests_total"]
        and m1["errors_total"] >= m0["errors_total"]
        and m1["uptime_s"] >= m0["uptime_s"]
        and m1["max_inflight"] == m0["max_inflight"] > 0
    )
    # the per-backend completion ledger reconciles with the usage ring
    # for the same window: 3 ok completes on byok, 1 on local_fx1
    rep = _usage(client, root)
    c = m1["complete"]
    out["complete_reconciles_usage_counts"] = (
        c["byok"]["ok"] + c["byok"]["error"] == rep["by_backend"]["byok"]["requests"] == 3
        and c["local_fx1"]["ok"] + c["local_fx1"]["error"]
        == rep["by_backend"]["local_fx1"]["requests"]
        == 1
    )
    out["complete_reconciles_usage_tokens"] = (
        c["byok"]["prompt_tokens"]
        == rep["by_backend"]["byok"]["prompt_tokens"]
        == 3 * _U["prompt_tokens"]
        and c["byok"]["total_tokens"]
        == rep["by_backend"]["byok"]["total_tokens"]
        == 3 * _U["total_tokens"]
        and c["byok"]["usage_calls"] == rep["by_backend"]["byok"]["usage_reported"] == 3
    )
    # histogram sanity: cumulative le buckets, +Inf equals count
    lb = c["byok"]["latency_buckets"]
    vals = [lb[k] for k in lb]
    out["histogram_cumulative_sane"] = (
        list(vals) == sorted(vals)
        and vals[-1] == c["byok"]["latency_count"] == c["byok"]["ok"]
        and "+Inf" in lb
        and c["byok"]["latency_sum_ms"] >= 0.0
    )
    out["inflight_idle_zero_and_watermark_kept"] = (
        m1["inflight"] == 0 and m1["inflight_watermark"] >= 1 and m1["draining"] is False
    )
    out["refusals_counted_not_metered"] = (
        m1["by_status"].get("401", 0) >= 1 and m1["by_status"].get("422", 0) >= 1
    )
    return out


def _probe_rate_limited_metrics() -> dict[str, bool]:
    """The 429 family: the global limiter and a managed key's rpm window
    both land in ``rate_limited_total`` and ``by_status['429']`` exactly;
    /health is exempt from the limiter so liveness survives a flood."""
    out: dict[str, bool] = {}
    client, _ = _client(rate_limit_rps=1.0)
    root = {"X-API-Key": _ROOT}
    # capacity is max(1, rps)=1: first call consumes, the burst refuses
    burst = [client.get("/harness/commands", headers=root).status_code for _ in range(4)]
    n_429 = sum(1 for c in burst if c == 429)
    health_codes = [client.get("/health").status_code for _ in range(3)]
    time.sleep(1.1)  # refill the global bucket before scraping
    m = _metrics(client, root)
    out["rate_limited_total_exact"] = (
        burst[0] == 200
        and n_429 >= 1
        and m["rate_limited_total"] == n_429
        and m["by_status"].get("429", 0) == n_429
        and m["errors_total"] >= n_429
    )
    out["health_exempt_from_limiter"] = all(c == 200 for c in health_codes)

    # per-key rpm refusal joins the same 429 family — on a limiter-free
    # app so the key window (not the global bucket) is the denial source
    client2, _ = _client()
    raw, _ = _mint(client2, {"X-API-Key": _ROOT}, rpm=1)
    kh = {"X-API-Key": raw}
    ok_first = _complete(client2, "byok", kh).status_code
    denied = _complete(client2, "byok", kh).status_code
    m2 = _metrics(client2, {"X-API-Key": _ROOT})
    out["key_rpm_denial_lands_in_429_family"] = (
        ok_first == 200
        and denied == 429
        and m2["rate_limited_total"] == 1
        and m2["by_status"].get("429", 0) == 1
    )
    return out


# ---------------------------------------------------------------------------
# Prometheus exposition
# ---------------------------------------------------------------------------


def _prom_text(client: TestClient, headers: dict[str, str], **params: Any) -> Any:
    return client.get("/metrics", params=params, headers=headers)


def _probe_prometheus() -> dict[str, bool]:
    """The text exposition negotiates correctly and meets selected 0.0.4 checks:
    declared types, finite values, cumulative histograms, mechanical
    label escaping, correct content-type."""
    out: dict[str, bool] = {}
    weird = 'we"ird\\backend\nname'
    app = _make_app({"byok": lambda: _StubBackend("m-0", dict(_U))})
    client = _test_client(app)
    root = {"X-API-Key": _ROOT}
    for _ in range(2):
        _complete(client, "byok", root)
    # label values reach the exposition from record keys beyond the
    # Literal-bound backend enum (eval: / probe: metric keys carry
    # free-form names) — inject the hostile name through the same meter
    # the routes use, then read it back over the wire
    app.state.metrics.record_complete(weird, True, 5.0, usage=dict(_U))
    _wait_job_submitted(client, root)

    body = _prom_text(client, root, format="prom")
    text = body.text
    prom = _Prom(text)
    out["format_prom_negotiates_text"] = (
        body.status_code == 200
        and body.headers["content-type"].startswith("text/plain")
        and "version=0.0.4" in body.headers["content-type"]
    )
    alias = _prom_text(client, root, format="prometheus")
    out["format_prometheus_alias"] = (
        alias.status_code == 200
        and alias.headers["content-type"].startswith("text/plain")
        and alias.text.startswith("# HELP fx1_uptime_seconds")
    )
    out["accept_text_plain_negotiates"] = (
        client.get("/metrics", headers={**root, "Accept": "text/plain;version=0.0.4"})
        .headers["content-type"]
        .startswith("text/plain")
    )
    out["accept_openmetrics_negotiates"] = (
        client.get("/metrics", headers={**root, "Accept": "application/openmetrics-text"})
        .headers["content-type"]
        .startswith("text/plain")
    )
    out["accept_json_stays_json"] = client.get(
        "/metrics", headers={**root, "Accept": "application/json"}
    ).headers["content-type"].startswith("application/json") and _prom_text(
        client, root, format="json"
    ).headers["content-type"].startswith("application/json")
    out["format_invalid_422_enveloped"] = (
        _enveloped(_prom_text(client, root, format="bogus"))
        and _prom_text(client, root, format="bogus").status_code == 422
    )
    out["exposition_declared_typed_ordered"] = prom.all_typed() and prom.ordered()
    out["exposition_no_nan_or_inf_values"] = prom.finite()
    out["exposition_core_families"] = all(
        name in prom.families
        for name in (
            "fx1_uptime_seconds",
            "fx1_requests_total",
            "fx1_errors_total",
            "fx1_rate_limited_total",
            "fx1_inflight",
            "fx1_inflight_watermark",
            "fx1_max_inflight",
            "fx1_draining",
            "fx1_jobs",
            "fx1_complete_total",
            "fx1_complete_latency_ms",
            "fx1_complete_tokens_total",
            "fx1_complete_usage_calls_total",
        )
    )
    # gauge/counter state rides the exposition
    ok200 = prom.counter("fx1_requests_total", {"status": "200"})
    drained = prom.counter("fx1_draining")
    succeeded = prom.counter("fx1_jobs", {"status": "succeeded"})
    out["exposition_values_track_state"] = (
        ok200 is not None
        and ok200 >= 1
        and drained == 0.0
        and succeeded is not None
        and succeeded >= 1
    )
    # histogram shape: cumulative buckets ending at +Inf == _count
    hist = prom.families["fx1_complete_latency_ms"]
    byok_buckets = [
        (float(lab["le"].replace("+Inf", "inf")), float(val))
        for _s, lab, val in hist["samples"]
        if _s.endswith("_bucket") and lab.get("backend") == "byok"
    ]
    byok_buckets.sort()
    count_byok = [
        val
        for s, lab, val in hist["samples"]
        if s.endswith("_count") and lab.get("backend") == "byok"
    ]
    out["histogram_buckets_cumulative_inf_eq_count"] = (
        bool(byok_buckets)
        and byok_buckets[-1][0] == float("inf")
        and [v for _le, v in byok_buckets] == sorted(v for _le, v in byok_buckets)
        and len(count_byok) == 1
        and float(count_byok[0]) == byok_buckets[-1][1] == 2.0
    )
    # label escaping round-trips a hostile backend name byte-for-byte
    weird_ok = prom.counter("fx1_complete_total", {"backend": weird, "outcome": "ok"})
    out["label_escaping_round_trips"] = weird_ok == 1.0
    return out


def _wait_job_submitted(client: TestClient, root: dict[str, str]) -> int:
    """Submit a quick job and wait for its terminal record; returns the
    count of succeeded jobs expected in the store afterwards (1)."""
    r = client.post("/harness/jobs", json={"command": "doctor"}, headers=root)
    assert r.status_code == 202, r.text
    rec = _wait_job(client, r.json()["job_id"], root)
    assert rec["status"] == "succeeded"
    return 1


# ---------------------------------------------------------------------------
# drain lifecycle
# ---------------------------------------------------------------------------


def _probe_drain() -> dict[str, bool]:
    """The drain latch: admin-only, one-way, GETs/health/metrics stay
    open, in-flight work finishes, wait_s blocks to idle or lapses."""
    out: dict[str, bool] = {}
    gate = _GateBackend("m-0", width=1, usage=dict(_U))
    client, _ = _client({"byok": lambda: gate})
    _resources().callback(gate.release.set)
    root = {"X-API-Key": _ROOT}

    # one request parked in-flight
    box: dict[str, Any] = {}
    t = threading.Thread(target=lambda: box.setdefault("r", _complete(client, "byok", root)))
    t.start()
    assert gate.all_entered.wait(timeout=15.0)

    d0 = client.post("/harness/drain", params={"wait_s": 0}, headers=root).json()
    out["drain_wait_zero_reports_live_inflight"] = (
        d0["draining"] is True and d0["inflight"] >= 1 and d0["drained"] is False
    )
    out["ready_flips_503_under_drain"] = (
        client.get("/ready", headers=root).status_code == 503
        and client.get("/ready", headers=root).json().get("code") == "draining"
    )
    out["health_stays_up_reports_draining"] = (
        client.get("/health").status_code == 200
        and client.get("/health").json().get("draining") is True
    )
    out["reads_stay_open_under_drain"] = (
        client.get("/harness/commands", headers=root).status_code == 200
        and client.get("/metrics", headers=root).status_code == 200
        and client.get("/harness/usage", headers=root).status_code == 200
        and client.get("/harness/completions", headers=root).status_code == 200
    )
    # a blocking drain parks until the in-flight call is released
    dbox: dict[str, Any] = {}
    dt = threading.Thread(
        target=lambda: dbox.setdefault(
            "r", client.post("/harness/drain", params={"wait_s": 30}, headers=root)
        )
    )
    # Observe the actual condition wait while it owns the condition lock.
    # The backend cannot release its inflight slot until that wait unlocks
    # the condition, so scheduling cannot turn this into an already-idle check.
    app: Any = client.app
    condition = app.state.metrics._cond  # noqa: SLF001 - instance-local probe
    original_wait = condition.wait
    wait_entered = threading.Event()

    def observed_wait(timeout: float | None = None) -> bool:
        wait_entered.set()
        return bool(original_wait(timeout))

    condition.wait = observed_wait
    dt.start()
    try:
        observed_block = wait_entered.wait(timeout=5.0)
    finally:
        gate.release.set()
        t.join(15.0)
        dt.join(15.0)
        condition.wait = original_wait
    r_done = box["r"]
    r_block = dbox["r"].json()
    out["drain_inflight_completes_uninterrupted"] = r_done.status_code == 200
    out["drain_wait_blocks_until_idle"] = (
        observed_block
        and r_block["drained"] is True
        and r_block["inflight"] == 0
        and r_block["draining"] is True
    )
    out["metrics_reports_draining_latch"] = (
        _metrics(client, root)["draining"] is True and _metrics(client, root)["inflight"] == 0
    )
    # the latch refuses new gated work — enveloped
    out["drain_refuses_new_submits"] = (
        _complete(client, "byok", root).status_code == 503
        and _complete(client, "byok", root).json().get("code") == "draining"
        and client.post("/harness/jobs", json={"command": "doctor"}, headers=root).status_code
        == 503
        and client.post(
            "/harness/complete/batch",
            json={"backend": "byok", "batch": [[{"role": "user", "content": "q"}]]},
            headers=root,
        ).status_code
        == 503
        and client.post("/harness/runs", json={"command": "doctor"}, headers=root).status_code
        == 503
    )
    # gate precedence is per-route and measured: Depends(slot) resolves
    # before body validation on /harness/complete (503 beats 422), while
    # /harness/jobs parses the body (422) and validates the command name
    # (404) inside the handler before it reaches the drain check;
    # /harness/runs wraps lab.run in the gate, so its drain-503 beats an
    # unknown command but not a malformed body
    out["drain_gate_precedes_complete_body"] = (
        client.post("/harness/complete", json={"backend": "byok"}, headers=root).status_code == 503
    )
    out["drain_jobs_validate_command_first"] = (
        client.post("/harness/jobs", json={"command": "no-such-cmd"}, headers=root).status_code
        == 404
        and client.post("/harness/jobs", json={}, headers=root).status_code == 422
    )
    out["drain_runs_gate_before_command_lookup"] = (
        client.post("/harness/runs", json={"command": "no-such-cmd"}, headers=root).status_code
        == 503
        and client.post("/harness/runs", json={}, headers=root).status_code == 422
    )
    # one-way + idempotent: re-POSTing keeps the latch, reports live state
    d2 = client.post("/harness/drain", headers=root).json()
    out["drain_one_way_idempotent"] = (
        d2["draining"] is True
        and d2["inflight"] == 0
        and d2["drained"] is True
        and client.get("/ready", headers=root).status_code == 503
        and _complete(client, "byok", root).status_code == 503
    )
    # wait_s bounds are validated inside the envelope
    out["drain_wait_bounds_validated"] = (
        _enveloped(client.post("/harness/drain", params={"wait_s": -1}, headers=root))
        and _enveloped(client.post("/harness/drain", params={"wait_s": 601}, headers=root))
        and _enveloped(client.post("/harness/drain", params={"wait_s": "nan"}, headers=root))
    )
    return out


def _probe_drain_restart() -> dict[str, bool]:
    """Recreate an app over the same state directory and inspect recovery.

    This checks application initialization, not an external process restart.
    Resource cleanup precedes deletion of the temporary state directory.
    """
    out: dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as td, _audit_resources():
        sd = Path(td) / "state"
        c1 = _test_client(_make_app(state_dir=sd))
        root = {"X-API-Key": _ROOT}
        raw, _kid = _mint(c1, root, scopes=["read"])
        jr = c1.post("/harness/jobs", json={"command": "doctor"}, headers=root)
        jid = jr.json()["job_id"]
        _wait_job(c1, jid, root)
        c1.post("/harness/drain", headers=root)
        assert c1.get("/ready", headers=root).status_code == 503

        c2 = _test_client(_make_app(state_dir=sd))
        out["drain_process_local_restart_ready"] = (
            c2.get("/ready", headers=root).status_code == 200
            and c2.get("/health").json().get("draining") is False
        )
        out["restart_keeps_key_store_consistent"] = (
            c2.get("/metrics", headers={"X-API-Key": raw}).status_code == 200
            and c2.get("/metrics", headers={"X-API-Key": "fx1k_deadbeef"}).status_code == 401
        )
        rec = c2.get(f"/harness/jobs/{jid}", headers=root)
        out["restart_journaled_jobs_survive"] = (
            rec.status_code == 200 and rec.json().get("status") == "succeeded"
        )
        out["restart_accepts_new_work"] = (
            c2.post("/harness/jobs", json={"command": "doctor"}, headers=root).status_code == 202
        )
    return out


def _probe_drain_concurrent() -> dict[str, bool]:
    """Parallel drains are idempotent — every racer lands the latch and
    reads the same settled state."""
    out: dict[str, bool] = {}
    client, _ = _client()
    root = {"X-API-Key": _ROOT}
    with ThreadPoolExecutor(max_workers=_N) as pool:
        rs = list(pool.map(lambda _i: client.post("/harness/drain", headers=root), range(_N)))
    out["parallel_drains_all_latch"] = all(
        r.status_code == 200 and r.json()["draining"] is True for r in rs
    )
    out["parallel_drain_single_latch_state"] = (
        client.get("/ready", headers=root).status_code == 503
        and _metrics(client, root)["draining"] is True
    )
    return out


# ---------------------------------------------------------------------------
# request-id propagation
# ---------------------------------------------------------------------------


def _probe_request_id() -> dict[str, bool]:
    """X-Request-ID: echoed on success AND every refusal class; minted
    (valid charset) when absent or malformed; Anthropic dialect stamps
    its own header name with the same id."""
    out: dict[str, bool] = {}
    client, _ = _client()
    root = {"X-API-Key": _ROOT, "X-Request-ID": "rid-obs.1_ok"}

    out["rid_echoed_on_success"] = (
        client.get("/health", headers=root).headers.get("x-request-id") == "rid-obs.1_ok"
        and client.get("/metrics", headers=root).headers.get("x-request-id") == "rid-obs.1_ok"
        and client.get("/ready", headers=root).headers.get("x-request-id") == "rid-obs.1_ok"
    )
    out["rid_echoed_on_refusals"] = (
        client.get("/metrics", headers={"X-Request-ID": "rid-obs.1_ok"}).headers.get("x-request-id")
        == "rid-obs.1_ok"  # 401
        and client.post("/harness/complete", json={"backend": "byok"}, headers=root).headers.get(
            "x-request-id"
        )
        == "rid-obs.1_ok"  # 422
        and client.get("/nope", headers=root).headers.get("x-request-id") == "rid-obs.1_ok"
    )
    minted = client.get("/health", headers={"X-API-Key": _ROOT}).headers.get("x-request-id", "")
    out["rid_minted_when_absent"] = bool(_RID_RE.fullmatch(minted)) and len(minted) == 32
    bad = client.get("/health", headers={"X-Request-ID": "bad rid!\r\ninjected"})
    bad_id = bad.headers.get("x-request-id", "")
    out["rid_minted_when_malformed"] = bad_id != "bad rid!\r\ninjected" and bool(
        _RID_RE.fullmatch(bad_id)
    )
    long_id = "a" * 65
    over = client.get("/health", headers={"X-Request-ID": long_id}).headers.get("x-request-id", "")
    out["rid_minted_when_overlong"] = over != long_id and bool(_RID_RE.fullmatch(over))
    # anthropic dialect echoes under its own header name on errors too
    ar = client.post(
        "/v1/messages",
        json={"model": "fx1", "max_tokens": 8, "messages": []},
        headers={**root, "anthropic-version": "2023-06-01"},
    )
    out["rid_anthropic_dialect_echoed"] = (
        ar.headers.get("x-request-id") == "rid-obs.1_ok"
        and ar.headers.get("request-id") == "rid-obs.1_ok"
    )
    out["rid_echoed_on_drain_503"] = (
        client.post("/harness/drain", headers=root).status_code == 200
        and client.post("/harness/jobs", json={"command": "doctor"}, headers=root).headers.get(
            "x-request-id"
        )
        == "rid-obs.1_ok"
        and client.get("/ready", headers=root).headers.get("x-request-id") == "rid-obs.1_ok"
    )
    return out


# ---------------------------------------------------------------------------
# error envelope + hostile inputs
# ---------------------------------------------------------------------------


def _probe_error_envelope() -> dict[str, bool]:
    """Every refusal across the ops surface lands in the wire-code
    envelope — no bare 500s, no unenveloped exceptions."""
    out: dict[str, bool] = {}
    client, _ = _client()
    root = {"X-API-Key": _ROOT}

    # routing-level refusals ride the same envelope — an unmatched path
    # and a wrong-method call raise the starlette base exception, which
    # the handler is registered on (not the fastapi subclass)
    r404 = client.get("/nope", headers=root)
    r405 = client.post("/metrics", headers=root)
    out["routing_refusals_enveloped"] = (
        r404.status_code == 404
        and r404.json().get("code") == "not_found"
        and r405.status_code == 405
        and r405.json().get("code") == "method_not_allowed"
    )
    cases = [
        r404,
        r405,
        client.get("/metrics", params={"format": "zz"}, headers=root),  # 422
        client.get("/ready"),  # 401
        client.post("/harness/drain"),  # 401
        client.get("/harness/backends", headers={"X-API-Key": "fx1k_deadbeef"}),  # 401
        client.post("/harness/drain", params={"wait_s": "inf"}, headers=root),  # 422
        client.post("/harness/complete", json={"backend": "byok"}, headers=root),  # 422
        client.post(
            "/harness/runs", json={"command": "no-such-cmd"}, headers=root
        ),  # 404 not_found
    ]
    out["hostile_inputs_all_enveloped"] = all(_enveloped(r) for r in cases)
    out["no_bare_500s_on_ops_surface"] = all(r.status_code != 500 for r in cases) and all(
        r.status_code < 500 for r in cases
    )
    codes = {r.status_code for r in cases}
    out["error_codes_distinct_per_class"] = codes == {401, 404, 405, 422}
    out["routing_refusal_rids_echoed"] = (
        client.get("/nope", headers={**root, "X-Request-ID": "rid-rt.1"}).headers.get(
            "x-request-id"
        )
        == "rid-rt.1"
    )

    # declared-body abuse — raw ASGI since httpx sanitizes Content-Length
    app = _make_app(api_key=None)
    big = _raw_http(
        app,
        "POST",
        "/harness/runs",
        [("content-length", str(2 << 20).encode()), ("x-api-key", b"none")],
        b"{}",
    )
    out["oversized_declared_body_413_enveloped"] = (
        big["status"] == 413
        and b"too_large" in big.get("body", b"")
        and b'"code"' in big.get("body", b"")
    )
    bad_len = _raw_http(
        app,
        "POST",
        "/harness/runs",
        [("content-length", b"abc"), ("x-api-key", b"none")],
        b"{}",
    )
    out["garbage_content_length_400_enveloped"] = (
        bad_len["status"] == 400
        and b"bad_request" in bad_len.get("body", b"")
        and b'"code"' in bad_len.get("body", b"")
    )
    # deep-hostile JSON on a gated route — malformed body parses refused
    malformed = client.post(
        "/harness/complete",
        content=b'{"backend": "byok", "messages": [',
        headers={**root, "Content-Type": "application/json"},
    )
    out["malformed_json_422_enveloped"] = malformed.status_code == 422 and _enveloped(malformed)
    return out


# ---------------------------------------------------------------------------
# concurrency integrity
# ---------------------------------------------------------------------------


def _probe_concurrency() -> dict[str, bool]:
    """Parallel traffic loses no increment; the inflight gauge tracks a
    held burst exactly and returns to zero; watermark keeps the peak."""
    out: dict[str, bool] = {}
    width = 4
    gate = _GateBackend("m-0", width=width, usage=dict(_U))
    client, _ = _client({"byok": lambda: gate, "local_fx1": lambda: _StubBackend("f-1", dict(_U2))})
    _resources().callback(gate.release.set)
    root = {"X-API-Key": _ROOT}
    m0 = _metrics(client, root)

    n_reads = 24
    with ThreadPoolExecutor(max_workers=8) as pool:
        codes = list(
            pool.map(
                lambda _i: client.get("/harness/commands", headers=root).status_code,
                range(n_reads),
            )
        )
    # held burst: `width` completes parked inside the gate
    tbox: dict[str, Any] = {"rs": []}

    def _call() -> None:
        tbox["rs"].append(_complete(client, "byok", root))

    ths = [threading.Thread(target=_call) for _ in range(width)]
    for t_ in ths:
        t_.start()
    assert gate.all_entered.wait(timeout=15.0)
    mid = _metrics(client, root)
    out["inflight_gauge_tracks_held_burst"] = (
        mid["inflight"] == width and mid["inflight_watermark"] >= width
    )
    gate.release.set()
    for t_ in ths:
        t_.join(15.0)
    m1 = _metrics(client, root)
    out["inflight_returns_to_zero_watermark_keeps_peak"] = (
        m1["inflight"] == 0 and m1["inflight_watermark"] >= width
    )
    # requests_total delta == every completed response between the two
    # scrape snapshots (m0 and mid scrapes count, m1 does not)
    expected = 1 + n_reads + 1 + width  # m0 + reads + mid scrape + completes
    out["parallel_requests_no_lost_increment"] = (
        all(c == 200 for c in codes)
        and all(r.status_code == 200 for r in tbox["rs"])
        and m1["requests_total"] - m0["requests_total"] == expected
        and m1["by_status"].get("200", 0) - m0["by_status"].get("200", 0) == expected
    )
    u = _usage(client, root)
    out["parallel_completes_all_recorded"] = (
        u["records_seen"] == width and u["totals"]["requests"] == width
    )
    return out


# ---------------------------------------------------------------------------
# HarnessClient surface + documented error map
# ---------------------------------------------------------------------------


def _tc_transport(client: TestClient) -> Any:
    """Adapt HarnessClient's transport contract to a TestClient."""

    def send(
        method: str,
        url: str,
        payload: dict[str, Any] | bytes | None,
        headers: dict[str, str],
        timeout_s: float,
    ) -> tuple[int, Any, bytes]:
        p = urllib.parse.urlparse(url)
        path = p.path + (f"?{p.query}" if p.query else "")
        if method == "GET":
            resp = client.get(path, headers=headers)
        elif method == "DELETE":
            resp = client.delete(path, headers=headers)
        elif isinstance(payload, bytes):
            resp = client.post(path, content=payload, headers=headers)
        else:
            resp = client.post(path, json=payload, headers=headers)
        return resp.status_code, dict(resp.headers), resp.content

    return send


def _probe_client_surface() -> dict[str, bool]:
    """``HarnessClient``'s ops methods ride the same wire contract and the
    documented error map holds end-to-end."""
    out: dict[str, bool] = {}
    from fx1.serve.backends import BackendNotConfiguredError  # noqa: PLC0415
    from fx1.serve.client import (  # noqa: PLC0415
        HarnessAuthError,
        HarnessClient,
    )

    client, _ = _client()
    hc = HarnessClient("http://harness.test", transport=_tc_transport(client), api_key=_ROOT)
    anon = HarnessClient("http://harness.test", transport=_tc_transport(client))

    h = hc.health()
    out["client_health_shape"] = (
        h.status == "ok" and isinstance(h.registered_commands, int) and isinstance(h.backends, dict)
    )
    m = hc.metrics()
    out["client_metrics_shape"] = (
        isinstance(m.requests_total, int)
        and isinstance(m.by_status, dict)
        and isinstance(m.draining, bool)
        and m.max_inflight >= 1
    )
    out["client_metrics_text_is_prom"] = hc.metrics_text().startswith("# HELP fx1_uptime_seconds")
    out["client_ready_true_while_accepting"] = hc.ready()["ready"] is True

    # the documented error map, exercised on the ops surface itself
    def _raises(fn: Callable[[], Any], exc: type[BaseException]) -> bool:
        try:
            fn()
        except exc:
            return True
        except Exception:
            return False
        return False

    out["client_401_maps_auth_error"] = _raises(anon.metrics, HarnessAuthError)
    out["client_404_maps_keyerror"] = _raises(
        lambda: hc._json("GET", "/nope", idempotent=True),  # noqa: SLF001
        KeyError,
    )
    out["client_422_maps_valueerror"] = _raises(
        lambda: hc._json("GET", "/metrics?format=zz", idempotent=True),  # noqa: SLF001
        ValueError,
    )
    d = hc.drain()
    out["client_drain_latches"] = d["draining"] is True and d["drained"] is True
    out["client_ready_503_maps_backend_error"] = _raises(hc.ready, BackendNotConfiguredError)
    return out


def observe_audit() -> dict[str, Any]:
    """Run the ops-instrumentation battery; results are literal bools."""
    prev = os.environ.pop(_API_KEY_ENV, None)
    out: dict[str, Any] = {}
    try:
        with _audit_resources():
            for section in (
                _probe_liveness,
                _probe_scope_boundary,
                _probe_dev_and_remote,
                _probe_metrics_json,
                _probe_rate_limited_metrics,
                _probe_prometheus,
                _probe_drain,
                _probe_drain_restart,
                _probe_drain_concurrent,
                _probe_request_id,
                _probe_error_envelope,
                _probe_concurrency,
                _probe_client_surface,
            ):
                out.update(section())
    finally:
        if prev is not None:
            os.environ[_API_KEY_ENV] = prev
    return out


def observe_audit_bench() -> dict[str, Any]:
    """Sealed receipt: every probe True under observe_audit.v1."""
    r = observe_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "observe_audit",
        "schema": "observe_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok, "defects": defects},
        "coverage": {
            "transport": "Starlette TestClient and direct in-process ASGI calls",
            "not_verified": [
                "real network transport behavior",
                "external process restart",
                "exhaustive Prometheus parser conformance",
                "all possible hostile inputs or refusal cases",
            ],
        },
        "interpretation": (
            "The ops-instrumentation contract holds end to end: /health "
            "is liveness (public, always 200, backends as booleans, "
            "draining flag visible) while /ready is readiness (keyed, 503 "
            "once drain latches) — the pair are never conflated. "
            "/metrics JSON counts every response exactly once, partitions "
            "by status, keeps the >=400 share in errors_total, stays "
            "monotonic for the process lifetime, and its per-backend "
            "complete ledger reconciles 1:1 with /harness/usage for the "
            "same window. The Prometheus view negotiates on ?format=prom "
            "or Accept text/plain/OpenMetrics, passes selected 0.0.4 "
            "exposition (declared TYPEs, HELP before TYPE before "
            "samples, no NaN/Inf values, cumulative le buckets with "
            "+Inf == _count, label escaping that round-trips a hostile "
            "backend name). POST /harness/drain is admin-scope only: the "
            "one-way latch refuses new gated work with enveloped 503 "
            "'draining' while in-flight work finishes and reads stay "
            "open; wait_s blocks to idle or its bound; parallel drains "
            "are idempotent; and recreating the app with the same state "
            "directory comes back ready while journaled "
            "keys and job records survive. X-Request-ID echoes on "
            "success and the tested refusal cases and mints on absent or "
            "malformed input; the tested hostile inputs land in the "
            "documented wire-code envelope with no bare 500s; parallel "
            "traffic loses no counter increment; and HarnessClient's "
            "ops methods honor the documented 401/403->HarnessAuthError, "
            "404->KeyError, 422->ValueError, 503->BackendNotConfigured "
            "error map."
            if ok
            else f"OBSERVE AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(observe_audit_bench(), indent=2, sort_keys=True))
