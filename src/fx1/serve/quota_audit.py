"""quota_audit — adversarial probes on managed-key quota/rate-limit boundaries.

The claim under test: a managed key's declared limits are *exact*
boundaries — the window admits precisely ``rpm`` calls, the request
budget admits precisely ``max_requests`` calls, the token budget admits
calls while ``tokens_used < max_tokens`` and refuses the next — and a
refusal is terminal when it should be (``quota_exceeded`` carries no
``Retry-After`` and is never retried by the clients) or recoverable
when it should be (``rate_limited`` carries the hint and the window
re-admits on expiry). Every refusal consumes nothing.

Coverage map:

- *rpm window* — exactly N calls admitted inside the window; call N+1
  refuses 429 ``rate_limited`` with ``Retry-After`` and the
  ``X-RateLimit-{Limit,Remaining,Reset}-Requests`` trio — on the
  refusal and on every admitted answer, decrementing N-1..0. Window
  expiry re-admits; a refusal consumes no slot and no use; N parallel
  calls at the boundary admit exactly N (the claim is atomic) with no
  overshoot; windows are per-key — one key's full window never blocks
  the next key's.
- *request quota* — ``max_requests`` decrements exactly once per
  admitted call; the boundary is inclusive (uses==N admits, N+1
  refuses); ``quota_exceeded`` is terminal: no ``Retry-After``, no
  window slot burned; the exhausted check precedes scope and window
  checks; a PATCH raising the budget admits immediately, a PATCH
  tightening to the current count refuses immediately.
- *token quota* — ``max_tokens`` meters off provider-reported usage on
  the sync, SSE-stream, and batch surfaces alike; the call that
  crosses the budget is the last admitted (the spend is honest, the
  refusal lands on the next call); ``tokens_remaining`` clamps at
  zero, never negative, while ``tokens_used`` reports the real
  overshoot; a backend that reports no usage bills nothing.
- *scope refusals* — a 403 ``insufficient_scope`` never decrements
  ``uses`` nor consumes a window slot (post-#2796 contract): a
  read-scoped key refused on a write path admits immediately after on
  a read path.
- *persistence* — spent quota survives a fresh process on the same
  ``--state-dir``: lifetime uses and tokens replay from the journal; RPM windows remain
  process-local and reset on restart. A revoked key stays
  dead across restart; a rotated key's successor starts at zero with
  the predecessor's declared policy and lineage — quota never leaks
  across the rotation boundary.
- *uniform metering* — every authenticated path claims exactly one
  use and one window slot: chat, responses, messages, legacy
  completions, sync/batch/stream completes, job and eval submits, and
  GET surfaces alike; ``/health`` is public and unmetered even when it
  carries a credential; the env root key is unmetered (no budget
  headers); an authenticated 404 on an unknown path still counts —
  uses measures authenticated calls, not successes.
- *error paths* — pre-auth refusals bill nothing: the global rate
  limiter and the 413 body cap answer before auth runs. Post-auth
  failures count a use but never bill tokens: a 422 validation error,
  a 503 backend fault, and an over-capacity 503 each spent a
  credential the caller presented — the request budget is a count of
  authenticated calls, declared verbatim on the usage card.
- *client retry policy* — the wire contract both clients key on: a
  ``quota_exceeded`` 429 without ``Retry-After`` is terminal, so
  ``HarnessClient`` issues it exactly once even under
  ``retry_writes``/``max_retries``; a ``rate_limited`` 429 carries
  ``Retry-After``, is retried within ``max_retry_wait_s``, and is
  abandoned when the hint exceeds that budget. The TypeScript client is not executed by this battery.
- *anthropic surface* — on ``/v1/messages`` the same refusals speak
  the Anthropic grammar: ``x-should-retry: false`` on
  ``quota_exceeded``, ``true`` on ``rate_limited``, plus the
  ``anthropic-ratelimit-requests-*`` trio; the OpenAI legs answer the
  ``{error: {...}}`` envelope.

The lifetime counter and recovery behavior is provided by the existing
key store. These probes pin that behavior without changing its journal
or process-local RPM-window contract. This is a stub-backed TestClient
battery, not network timing, crash-durability, or live-provider evidence.

Sealed ``quota_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import functools
import json
import os
import tempfile
import threading
import time
import urllib.parse
from collections.abc import Callable, Iterator, Mapping
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

__all__ = ["quota_audit", "quota_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT = "k3y-material"
_AUDIT_LOCK = threading.Lock()
_RESOURCES: ContextVar[ExitStack] = ContextVar("quota_audit_resources")


@contextmanager
def _audit_context() -> Iterator[None]:
    """Restore ambient configuration and close all synthetic resources.

    Run this diagnostic in a dedicated process: its environment and
    rate-window overrides are process-wide, not application configuration.
    The lock serializes calls made through this module.
    """
    with _AUDIT_LOCK:
        saved = {
            name: value
            for name, value in os.environ.items()
            if name.startswith("FX1_") or name == "MOONSHOT_API_KEY"
        }
        for name in saved:
            os.environ.pop(name, None)
        try:
            with ExitStack() as resources:
                token = _RESOURCES.set(resources)
                try:
                    yield
                finally:
                    _RESOURCES.reset(token)
        finally:
            for name in list(os.environ):
                if name.startswith("FX1_") or name == "MOONSHOT_API_KEY":
                    os.environ.pop(name, None)
            os.environ.update(saved)


def _temporary_directory() -> Path:
    return Path(_RESOURCES.get().enter_context(tempfile.TemporaryDirectory(prefix="quota_audit_")))


_COMPLETE_PATH = "/harness/complete"
_KEYS_PATH = "/harness/keys"
_SELF_PATH = "/harness/self"
_JOBS_PATH = "/harness/jobs"
_HEALTH_PATH = "/health"
_CHAT_PATH = "/v1/chat/completions"
_RESPONSES_PATH = "/v1/responses"
_MESSAGES_PATH = "/v1/messages"
_LEGACY_PATH = "/v1/completions"
_STREAM_PATH = "/harness/complete/stream"
_BATCH_PATH = "/harness/complete/batch"
_EVALS_PATH = "/harness/evals"

_H_KEY = "X-API-Key"
_H_RETRY_AFTER = "retry-after"
_H_RL_LIMIT = "x-ratelimit-limit-requests"
_H_RL_REMAINING = "x-ratelimit-remaining-requests"
_H_RL_RESET = "x-ratelimit-reset-requests"
_H_SHOULD_RETRY = "x-should-retry"
_H_ANTH_LIMIT = "anthropic-ratelimit-requests-limit"
_H_ANTH_REMAINING = "anthropic-ratelimit-requests-remaining"
_H_ANTH_RESET = "anthropic-ratelimit-requests-reset"
_RATE_WINDOW_CONST = "_RATE_WINDOW_S"
_TINY_WINDOW_S = 0.05
_WINDOW_RECOVER_S = 0.08
_KEY_STORE_CLASS = "ApiKeyStore"

_U6 = {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6}
_U5 = {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}


class _MeterBackend:
    """Usage-reporting stub — per-call ``last_usage`` plus the
    cumulative ``total_usage`` the batch delta reads; the stream
    surface may report its own scripted payload."""

    def __init__(
        self,
        model: str,
        usage: dict[str, int] | None,
        stream_usage: dict[str, int] | None = None,
    ) -> None:
        self._model = model
        self._usage = usage
        self._stream_usage = usage if stream_usage is None else stream_usage
        self.calls = 0
        self.last_usage: dict[str, int] | None = None
        self.total_usage: dict[str, int] = {}
        self._lock = threading.Lock()

    def _report(self, usage: dict[str, int] | None) -> None:
        self.last_usage = usage
        if not usage:
            return
        with self._lock:
            for k, v in usage.items():
                self.total_usage[k] = self.total_usage.get(k, 0) + v

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        sampling: SamplingParams | None = None,  # NOSONAR(S1172)
    ) -> str:
        self.calls += 1
        self._report(dict(self._usage) if self._usage is not None else None)
        return f"ok:{messages[-1]['content']}"

    def stream(
        self,
        messages: list[dict[str, str]],  # NOSONAR(S1172) — protocol signature
        *,
        sampling: SamplingParams | None = None,  # NOSONAR(S1172)
    ) -> Any:
        self.calls += 1
        use = self._stream_usage
        self._report(dict(use) if use is not None else None)
        yield "tok-a"
        yield "tok-b"

    def close(self) -> None:
        """No resources to release — the stub holds nothing."""


class _FailBackend:
    """Configured-but-dead stub — raises the wire's
    ``BackendNotConfiguredError`` so resolution answers 503."""

    def __init__(self, model: str) -> None:
        self._model = model

    def complete(
        self,
        messages: list[dict[str, str]],  # NOSONAR(S1172) — protocol signature
        *,
        sampling: SamplingParams | None = None,  # NOSONAR(S1172)
    ) -> str:
        from fx1.serve.backends import BackendNotConfiguredError  # noqa: PLC0415

        raise BackendNotConfiguredError("no credentials configured")

    def close(self) -> None:
        """No resources to release — the stub holds nothing."""


class _HoldBackend:
    """Blocks inside ``complete`` until released — holds the inflight
    slot deterministically so a racing call over-caps (503)."""

    def __init__(self, model: str) -> None:
        self._model = model
        self._gate = threading.Event()
        self.entered = threading.Event()
        self.last_usage: dict[str, int] | None = None

    def complete(
        self,
        messages: list[dict[str, str]],  # NOSONAR(S1172) — protocol signature
        *,
        sampling: SamplingParams | None = None,  # NOSONAR(S1172)
    ) -> str:
        self.entered.set()
        self._gate.wait(timeout=10)
        return "ok"

    def release(self) -> None:
        self._gate.set()

    def close(self) -> None:
        """No resources to release — the stub holds nothing."""


class _BareBackend:
    """No ``_model`` and no ``last_usage`` — provider silence bills
    nothing on the token meter."""

    def complete(
        self,
        messages: list[dict[str, str]],  # NOSONAR(S1172) — protocol signature
        *,
        sampling: SamplingParams | None = None,  # NOSONAR(S1172)
    ) -> str:
        return "ok"

    def close(self) -> None:
        """No resources to release — the stub holds nothing."""


class _SteppedClock:
    """Key-store clock for the rate-window probes: frozen at construction
    and moved only by ``advance``. Window edges are then decided by the
    probe's scripted steps, not by how long a request took — a >50 ms
    stall between two calls on a loaded runner used to expire the patched
    window and admit the call the probe expects refused."""

    def __init__(self) -> None:
        self._now = time.time()

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _client(
    backend_map: dict[str, Any],
    api_key: str | None = None,
    *,
    state_dir: Path | None = None,
    rate_limit_rps: float = 0.0,
    max_inflight: int = 16,
    key_clock: Callable[[], float] | None = None,
) -> tuple[TestClient, ModuleType]:
    """(TestClient, api_module) — isolated env per construction; backends
    resolve from ``backend_map[name]`` zero-arg factories. ``key_clock``
    binds the app's key store to an injected clock (``_SteppedClock``)."""
    from fastapi.testclient import TestClient

    import fx1.serve.api as api_mod
    from fx1.harness import Harness

    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, "ok", ""

    resources = _RESOURCES.get()
    isolated = _temporary_directory()
    receipts = isolated / "receipts"
    receipts.mkdir()
    saved_key = os.environ.get(_API_KEY_ENV)
    try:
        if api_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = api_key
        real_store = getattr(api_mod, _KEY_STORE_CLASS)
        if key_clock is not None:
            setattr(api_mod, _KEY_STORE_CLASS, functools.partial(real_store, clock=key_clock))
        try:
            app = api_mod.create_app(
                harness=Harness(runner=fake_runner),
                backend_resolver=lambda name, *a, **k: backend_map[name](),
                state_dir=state_dir if state_dir is not None else isolated / "state",
                receipts_dir=receipts,
                ft_dir=isolated / "fine_tuning",
                rate_limit_rps=rate_limit_rps,
                max_inflight=max_inflight,
            )
        finally:
            setattr(api_mod, _KEY_STORE_CLASS, real_store)
        resources.callback(app.state.jobs_executor.shutdown, wait=True, cancel_futures=True)
        client = TestClient(app, raise_server_exceptions=False)
        resources.callback(client.close)
        resources.enter_context(client)
        return client, api_mod
    finally:
        if saved_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = saved_key


def _mint(client: TestClient, root_h: dict[str, str], **policy: Any) -> tuple[str, str]:
    """Mint a managed key → (raw, key_id)."""
    r = client.post(_KEYS_PATH, json=policy, headers=root_h)
    assert r.status_code == 201, r.text
    body = r.json()
    return str(body["key"]), str(body["id"])


def _key_card(client: TestClient, root_h: dict[str, str], key_id: str) -> dict[str, Any]:
    r = client.get(f"{_KEYS_PATH}/{key_id}/usage", headers=root_h)
    assert r.status_code == 200, r.text
    body: dict[str, Any] = r.json()
    return body


def _patch_key(key_id: str, root_h: dict[str, str], client: TestClient, **fields: Any) -> None:
    r = client.patch(f"{_KEYS_PATH}/{key_id}", json=fields, headers=root_h)
    assert r.status_code == 200, r.text


def _complete(
    client: TestClient,
    headers: dict[str, str],
    backend: str = "byok",
    **extra: Any,
) -> Any:
    return client.post(
        _COMPLETE_PATH,
        json={
            "backend": backend,
            "messages": [{"role": "user", "content": "hi"}],
            **extra,
        },
        headers=headers,
    )


def _chat(client: TestClient, headers: dict[str, str]) -> Any:
    return client.post(
        _CHAT_PATH,
        json={"model": "byok", "messages": [{"role": "user", "content": "hi"}]},
        headers=headers,
    )


def _responses(client: TestClient, headers: dict[str, str]) -> Any:
    return client.post(_RESPONSES_PATH, json={"model": "byok", "input": "hi"}, headers=headers)


def _messages(client: TestClient, headers: dict[str, str]) -> Any:
    return client.post(
        _MESSAGES_PATH,
        json={
            "model": "fx1",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={**headers, "X-Fx1-Backend": "byok"},
    )


def _legacy(client: TestClient, headers: dict[str, str]) -> Any:
    return client.post(_LEGACY_PATH, json={"model": "byok", "prompt": "hi"}, headers=headers)


def _stream(client: TestClient, headers: dict[str, str]) -> Any:
    return client.post(
        _STREAM_PATH,
        json={"backend": "byok", "messages": [{"role": "user", "content": "hi"}]},
        headers=headers,
    )


def _batch(client: TestClient, headers: dict[str, str]) -> Any:
    return client.post(
        _BATCH_PATH,
        json={
            "backend": "byok",
            "batch": [[{"role": "user", "content": "hi"}]],
        },
        headers=headers,
    )


def _job(client: TestClient, headers: dict[str, str]) -> Any:
    return client.post(_JOBS_PATH, json={"command": "doctor"}, headers=headers)


def _eval(client: TestClient, headers: dict[str, str]) -> Any:
    return client.post(
        _EVALS_PATH,
        json={"suite": "tooluse", "backend": "byok", "seed": 0},
        headers=headers,
    )


def _keys_mod() -> ModuleType:
    """The key-store module — the patched-window probes rewrite its
    ``_RATE_WINDOW_S`` and every ``ApiKeyStore`` reads it at call time,
    so no per-app wiring is needed."""
    import fx1.serve.keys as keys_mod  # noqa: PLC0415

    return keys_mod


def _rl_headers(resp: Any) -> tuple[bool, bool, bool]:
    h = {k.lower(): v for k, v in resp.headers.items()}
    return (_H_RL_LIMIT in h, _H_RL_REMAINING in h, _H_RL_RESET in h)


def _retry_after(resp: Any) -> int | None:
    raw = resp.headers.get(_H_RETRY_AFTER)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _parallel(client: TestClient, calls: list[Callable[[], Any]], workers: int) -> list[Any]:
    """Fire ``calls`` concurrently — the boundary race's overshoot check."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda c: c(), calls))


# ---------------------------------------------------------------------------
# rpm window boundary
# ---------------------------------------------------------------------------


def _rpm_window_probes() -> dict[str, Any]:  # NOSONAR(S3776) — scripted traffic fans out per edge
    out: dict[str, Any] = {}
    client, api_mod = _client(
        {"byok": lambda: _MeterBackend("byok-model", dict(_U6))}, api_key=_ROOT
    )
    root_h = {_H_KEY: _ROOT}

    # --- exact boundary + refusal wire shape ---------------------------------
    k_raw, k_id = _mint(client, root_h, rpm=3)
    k_h = {_H_KEY: k_raw}
    statuses = [_complete(client, k_h).status_code for _ in range(4)]
    out["rpm_exact_boundary_admits_n_then_429"] = statuses == [200, 200, 200, 429]
    refused = _complete(client, k_h)
    out["rpm_refusal_code_rate_limited"] = (
        refused.status_code == 429 and refused.json().get("code") == "rate_limited"
    )
    out["rpm_refusal_detail_declares_limit"] = "rate limit" in str(refused.json().get("detail", ""))
    out["rpm_refusal_retry_after_present"] = (_retry_after(refused) or 0) >= 1
    lim, rem, rst = _rl_headers(refused)
    out["rpm_refusal_rate_headers"] = lim and rem and rst
    out["rpm_refusal_limit_is_rpm"] = refused.headers.get(_H_RL_LIMIT) == "3"
    out["rpm_refusal_remaining_zero"] = refused.headers.get(_H_RL_REMAINING) == "0"

    # --- admitted answers decrement the announced headroom --------------------
    k2_raw, _k2_id = _mint(client, root_h, rpm=3)
    k2_h = {_H_KEY: k2_raw}
    remaining_seen = [
        int(_complete(client, k2_h).headers.get(_H_RL_REMAINING, "-1")) for _ in range(3)
    ]
    out["rpm_admitted_headers_decrement"] = remaining_seen == [2, 1, 0]

    # --- a refusal consumes no use and no slot --------------------------------
    card = _key_card(client, root_h, k_id)
    out["rpm_refusal_bills_no_use"] = card["uses"] == 3
    out["rpm_refusal_no_slot_backfill"] = card["window_remaining"] == 0

    # --- window expiry re-admits ----------------------------------------------
    clock2 = _SteppedClock()
    client2, api_mod2 = _client(
        {"byok": lambda: _MeterBackend("byok-model", dict(_U6))},
        api_key=_ROOT,
        key_clock=clock2,
    )
    keys_mod2 = _keys_mod()
    saved_window = float(getattr(keys_mod2, _RATE_WINDOW_CONST))
    try:
        setattr(keys_mod2, _RATE_WINDOW_CONST, _TINY_WINDOW_S)
        w_raw, _w_id = _mint(client2, root_h, rpm=1)
        w_h = {_H_KEY: w_raw}
        first = _complete(client2, w_h)
        denied = _complete(client2, w_h)
        clock2.advance(_WINDOW_RECOVER_S)
        third = _complete(client2, w_h)
        out["rpm_window_expiry_readmits"] = (
            first.status_code == 200 and denied.status_code == 429 and third.status_code == 200
        )
        out["rpm_window_restart_fresh_count"] = third.headers.get(_H_RL_REMAINING) == "0"
    finally:
        setattr(keys_mod2, _RATE_WINDOW_CONST, saved_window)

    # --- boundary race: parallel calls at the boundary admit exactly N --------
    client3, _ = _client(
        {"byok": lambda: _MeterBackend("byok-model", dict(_U6))},
        api_key=_ROOT,
        max_inflight=32,
    )
    r_raw, r_id = _mint(client3, root_h, rpm=6)
    r_h = {_H_KEY: r_raw}
    race = _parallel(client3, [lambda: _complete(client3, r_h)] * 12, 12)
    codes = [r.status_code for r in race]
    out["rpm_race_exact_admission"] = codes.count(200) == 6 and codes.count(429) == 6
    refused_race = next(r for r in race if r.status_code == 429)
    out["rpm_race_refusals_declare_retry"] = (_retry_after(refused_race) or 0) >= 1
    out["rpm_race_no_overshoot"] = _key_card(client3, root_h, r_id)["uses"] == 6

    # --- per-key isolation ------------------------------------------------------
    i1_raw, _ = _mint(client, root_h, rpm=1)
    i2_raw, _ = _mint(client, root_h, rpm=1)
    out["rpm_window_per_key"] = (
        _complete(client, {_H_KEY: i1_raw}).status_code == 200
        and _complete(client, {_H_KEY: i1_raw}).status_code == 429
        and _complete(client, {_H_KEY: i2_raw}).status_code == 200
    )

    # --- env root key is unmetered ----------------------------------------------
    root_resp = _complete(client, root_h)
    rl_on_root = _rl_headers(root_resp)
    out["env_key_unmetered_no_budget_headers"] = not any(rl_on_root)

    # --- unknown credential carries no budget headers ---------------------------
    bad = _complete(client, {_H_KEY: "fx1k_forged"})
    out["bad_credential_no_budget_headers"] = not any(_rl_headers(bad))
    return out


# ---------------------------------------------------------------------------
# max_requests quota boundary
# ---------------------------------------------------------------------------


def _quota_boundary_probes() -> dict[str, Any]:  # NOSONAR(S3776)
    out: dict[str, Any] = {}
    client, _ = _client({"byok": lambda: _MeterBackend("byok-model", dict(_U6))}, api_key=_ROOT)
    root_h = {_H_KEY: _ROOT}

    q_raw, q_id = _mint(client, root_h, max_requests=3)
    q_h = {_H_KEY: q_raw}
    statuses = [_complete(client, q_h).status_code for _ in range(4)]
    out["quota_exact_boundary_admits_n"] = statuses == [200, 200, 200, 429]
    refused = _complete(client, q_h)
    out["quota_refusal_code"] = (
        refused.status_code == 429 and refused.json().get("code") == "quota_exceeded"
    )
    out["quota_refusal_no_retry_after"] = _H_RETRY_AFTER not in {k.lower() for k in refused.headers}
    out["quota_refusal_status_is_429"] = refused.status_code == 429
    card = _key_card(client, root_h, q_id)
    out["quota_refusal_bills_no_use"] = card["uses"] == 3
    out["quota_requests_remaining_zero"] = card["requests_remaining"] == 0

    # quota check precedes the window claim: an exhausted key never
    # consumes a window slot
    o_raw, o_id = _mint(client, root_h, rpm=1, max_requests=1)
    o_h = {_H_KEY: o_raw}
    assert _complete(client, o_h).status_code == 200
    o_refused = _complete(client, o_h)
    out["quota_precedes_window"] = (
        o_refused.status_code == 429
        and o_refused.json().get("code") == "quota_exceeded"
        and _key_card(client, root_h, o_id)["window_remaining"] == 0
    )

    # quota check precedes the scope check: a spent read-key refuses a
    # write call 429, not 403
    s_raw, _ = _mint(client, root_h, max_requests=1, scopes=["read"])
    s_h = {_H_KEY: s_raw}
    assert client.get(_SELF_PATH, headers=s_h).status_code == 200
    s_refused = _complete(client, s_h)
    out["quota_precedes_scope"] = (
        s_refused.status_code == 429 and s_refused.json().get("code") == "quota_exceeded"
    )

    # PATCH moves the boundary live — relief and tightening both apply
    p_raw, p_id = _mint(client, root_h, max_requests=1)
    p_h = {_H_KEY: p_raw}
    assert _complete(client, p_h).status_code == 200
    assert _complete(client, p_h).status_code == 429
    _patch_key(p_id, root_h, client, max_requests=3)
    out["quota_patch_relief_admits"] = _complete(client, p_h).status_code == 200
    _patch_key(p_id, root_h, client, max_requests=2)
    out["quota_patch_tighten_refuses"] = _complete(client, p_h).status_code == 429

    # boundary race on the request budget
    client2, _ = _client(
        {"byok": lambda: _MeterBackend("byok-model", dict(_U6))},
        api_key=_ROOT,
        max_inflight=32,
    )
    b_raw, b_id = _mint(client2, root_h, max_requests=5)
    b_h = {_H_KEY: b_raw}
    race = _parallel(client2, [lambda: _complete(client2, b_h)] * 12, 12)
    codes = [r.status_code for r in race]
    out["quota_race_exact_admission"] = codes.count(200) == 5 and codes.count(429) == 7
    out["quota_race_refusals_terminal"] = all(
        _H_RETRY_AFTER not in {k.lower() for k in r.headers} for r in race if r.status_code == 429
    )
    out["quota_race_no_overshoot"] = _key_card(client2, root_h, b_id)["uses"] == 5

    # revoked and rotated credentials cannot spend
    v_raw, v_id = _mint(client, root_h, max_requests=4)
    v_h = {_H_KEY: v_raw}
    _complete(client, v_h)
    d = client.delete(f"{_KEYS_PATH}/{v_id}", headers=root_h)
    assert d.status_code == 200, d.text
    out["revoked_key_cannot_spend"] = _complete(client, v_h).status_code == 401

    t_raw, t_id = _mint(client, root_h, max_requests=4)
    t_h = {_H_KEY: t_raw}
    _complete(client, t_h)
    _complete(client, t_h)
    rot = client.post(f"{_KEYS_PATH}/{t_id}/rotate", json={}, headers=root_h)
    assert rot.status_code == 201, rot.text
    succ_raw = str(rot.json()["key"]["key"])
    succ_id = str(rot.json()["key"]["id"])
    out["rotate_old_key_dies"] = _complete(client, t_h).status_code == 401
    succ_card = _key_card(client, root_h, succ_id)
    out["rotate_successor_fresh_quota"] = (
        succ_card["uses"] == 0
        and succ_card["tokens_used"] == 0
        and succ_card["max_requests"] == 4
        and succ_card["rotated_from"] == t_id
    )
    out["rotate_successor_can_spend"] = _complete(client, {_H_KEY: succ_raw}).status_code == 200
    return out


# ---------------------------------------------------------------------------
# max_tokens quota boundary
# ---------------------------------------------------------------------------


def _token_budget_probes() -> dict[str, Any]:
    out: dict[str, Any] = {}
    client, _ = _client(
        {
            "byok": lambda: _MeterBackend("byok-model", dict(_U6), dict(_U5)),
            "hosted_k3": lambda: _MeterBackend("k3-model", dict(_U6)),
            "local_fx1": lambda: _BareBackend(),
        },
        api_key=_ROOT,
    )
    root_h = {_H_KEY: _ROOT}

    # the crossing call is the last admitted; the refusal lands after
    t_raw, t_id = _mint(client, root_h, max_tokens=10)
    t_h = {_H_KEY: t_raw}
    statuses = [_complete(client, t_h).status_code for _ in range(3)]
    out["tokens_crossing_call_last_admitted"] = statuses == [200, 200, 429]
    card = _key_card(client, root_h, t_id)
    out["tokens_used_reports_overshoot"] = card["tokens_used"] == 12
    out["tokens_remaining_clamps_zero"] = card["tokens_remaining"] == 0
    refused = _complete(client, t_h)
    out["tokens_refusal_terminal_no_retry_after"] = (
        refused.status_code == 429
        and refused.json().get("code") == "quota_exceeded"
        and _H_RETRY_AFTER not in {k.lower() for k in refused.headers}
    )
    out["tokens_refusal_bills_no_use"] = _key_card(client, root_h, t_id)["uses"] == 2

    # the stream surface meters its own reported usage — 5/call against
    # a budget of 8: admitted, admitted (crossing), refused
    st_raw, st_id = _mint(client, root_h, max_tokens=8)
    st_h = {_H_KEY: st_raw}
    stream_statuses = [_stream(client, st_h).status_code for _ in range(3)]
    out["tokens_stream_metered"] = _key_card(client, root_h, st_id)["tokens_used"] == 10
    out["tokens_stream_boundary_refuses"] = stream_statuses == [200, 200, 429]

    # the batch surface bills the aggregate usage_total delta once
    bt_raw, bt_id = _mint(client, root_h, max_tokens=11)
    bt_h = {_H_KEY: bt_raw}
    r_batch = _batch(client, bt_h)
    bt_card = _key_card(client, root_h, bt_id)
    out["tokens_batch_metered_delta"] = r_batch.status_code == 200 and bt_card["tokens_used"] == 6
    r_batch2 = _batch(client, bt_h)
    out["tokens_batch_crossing_admits_then_refuses"] = (
        r_batch2.status_code == 200
        and _key_card(client, root_h, bt_id)["tokens_used"] == 12
        and _batch(client, bt_h).status_code == 429
    )

    # provider silence bills nothing — the call admits without a token debit
    n_raw, n_id = _mint(client, root_h, max_tokens=1)
    n_h = {_H_KEY: n_raw}
    r_none = _complete(client, n_h, backend="local_fx1", checkpoint_dir="synthetic")
    out["tokens_provider_silence_bills_nothing"] = (
        r_none.status_code == 200 and _key_card(client, root_h, n_id)["tokens_used"] == 0
    )
    return out


# ---------------------------------------------------------------------------
# scope refusals
# ---------------------------------------------------------------------------


def _scope_refusal_probes() -> dict[str, Any]:
    out: dict[str, Any] = {}
    client, _ = _client({"byok": lambda: _MeterBackend("byok-model", dict(_U6))}, api_key=_ROOT)
    root_h = {_H_KEY: _ROOT}

    r_raw, r_id = _mint(client, root_h, scopes=["read"], rpm=2)
    r_h = {_H_KEY: r_raw}
    denied = _complete(client, r_h)
    out["scope_refusal_is_403"] = (
        denied.status_code == 403 and denied.json().get("code") == "insufficient_scope"
    )
    card = _key_card(client, root_h, r_id)
    out["scope_refusal_bills_no_use"] = card["uses"] == 0
    out["scope_refusal_no_tokens"] = card["tokens_used"] == 0
    out["scope_refusal_no_window_slot"] = card["window_remaining"] == 2
    # a denied write call cannot spend the key's read allowance
    out["scope_read_path_still_admits"] = (
        client.get(_SELF_PATH, headers=r_h).status_code == 200
        and _key_card(client, root_h, r_id)["uses"] == 1
    )

    # write-scoped key on the admin surface refuses identically
    w_raw, _w_id = _mint(client, root_h, scopes=["write"])
    w_h = {_H_KEY: w_raw}
    denied_admin = client.post(_KEYS_PATH, json={}, headers=w_h)
    out["scope_admin_refusal_is_403"] = denied_admin.status_code == 403
    return out


# ---------------------------------------------------------------------------
# persistence across restart (journaled counters)
# ---------------------------------------------------------------------------


def _persistence_probes() -> dict[str, Any]:  # NOSONAR(S3776)
    """Restart semantics on a journaled ``--state-dir``: two ``create_app``
    constructions over one directory stand in for two processes — the
    second sees only what the journal carried."""
    out: dict[str, Any] = {}
    state_dir = _temporary_directory()
    backends: dict[str, Any] = {"byok": lambda: _MeterBackend("byok-model", dict(_U6))}
    root_h = {_H_KEY: _ROOT}

    # process A — mint and spend
    client_a, _ = _client(backends, api_key=_ROOT, state_dir=state_dir)
    p_raw, p_id = _mint(client_a, root_h, max_requests=5, max_tokens=20, rpm=2)
    p_h = {_H_KEY: p_raw}
    # two spends: uses 1 (tokens 6, slot 1) then uses 2 (tokens 12, window full)
    for _ in range(2):
        assert _complete(client_a, p_h).status_code == 200
    v_raw, v_id = _mint(client_a, root_h)
    assert client_a.delete(f"{_KEYS_PATH}/{v_id}", headers=root_h).status_code == 200
    dead_h = {_H_KEY: v_raw}
    rot_src_raw, rot_src_id = _mint(client_a, root_h, max_requests=4)
    _complete(client_a, {_H_KEY: rot_src_raw})
    rot = client_a.post(f"{_KEYS_PATH}/{rot_src_id}/rotate", json={}, headers=root_h)
    assert rot.status_code == 201, rot.text
    succ_raw = str(rot.json()["key"]["key"])
    succ_id = str(rot.json()["key"]["id"])
    rot_dead_h = {_H_KEY: rot_src_raw}
    # an unwindowed budget key — the restart spend-down probe: two uses
    # burned on process A, three left on its declared budget
    b_raw, b_id = _mint(client_a, root_h, max_requests=5)
    b_h = {_H_KEY: b_raw}
    for _ in range(2):
        assert _complete(client_a, b_h).status_code == 200

    # process B — same --state-dir; nothing but the journal survives
    client_b, _ = _client(backends, api_key=_ROOT, state_dir=state_dir)
    card_b = _key_card(client_b, root_h, p_id)
    out["persisted_uses_after_restart"] = card_b["uses"] == 2
    out["persisted_tokens_after_restart"] = card_b["tokens_used"] == 12
    out["process_local_window_resets_after_restart"] = card_b["window_remaining"] == 2
    out["restarted_window_admits_with_durable_spend"] = (
        _complete(client_b, p_h).status_code == 200
        and _key_card(client_b, root_h, p_id)["tokens_used"] == 18
    )
    out["persisted_remaining_budget_enforced"] = card_b["requests_remaining"] == 3
    # the restarted process spends the declared remainder down to the
    # quota_exceeded refusal instead of laundering the budget to full
    post = [_complete(client_b, b_h).status_code for _ in range(4)]
    out["persisted_spenddown_to_refusal"] = post == [200, 200, 200, 429]
    out["persisted_refusal_is_quota"] = _key_card(client_b, root_h, b_id)["uses"] == 5
    out["persisted_revocation_holds"] = _complete(client_b, dead_h).status_code == 401
    out["persisted_rotation_holds"] = (
        _complete(client_b, rot_dead_h).status_code == 401
        and _complete(client_b, {_H_KEY: succ_raw}).status_code == 200
    )
    succ_card_b = _key_card(client_b, root_h, succ_id)
    out["persisted_rotation_fresh_quota"] = (
        succ_card_b["uses"] == 1
        and succ_card_b["max_requests"] == 4
        and succ_card_b["rotated_from"] == rot_src_id
    )
    return out


# ---------------------------------------------------------------------------
# uniform metering across surfaces
# ---------------------------------------------------------------------------


def _surface_probes() -> dict[str, Any]:  # NOSONAR(S3776)
    """One window, every path: the rpm claim and the use counter are the
    middleware's, so a call is a call no matter which surface it lands
    on — a route that forgot to meter is a defect."""
    out: dict[str, Any] = {}
    client, _ = _client({"byok": lambda: _MeterBackend("byok-model", dict(_U6))}, api_key=_ROOT)
    root_h = {_H_KEY: _ROOT}

    # rotate one rpm=6 key across six different surfaces: all admit,
    # the seventh refuses on whichever surface it lands on
    u_raw, u_id = _mint(client, root_h, rpm=6)
    u_h = {_H_KEY: u_raw}

    def complete_call() -> Any:
        return _complete(client, u_h)

    def chat_call() -> Any:
        return _chat(client, u_h)

    def responses_call() -> Any:
        return _responses(client, u_h)

    def messages_call() -> Any:
        return _messages(client, u_h)

    def legacy_call() -> Any:
        return _legacy(client, u_h)

    def job_call() -> Any:
        return _job(client, u_h)

    surface_calls = [
        complete_call,
        chat_call,
        responses_call,
        messages_call,
        legacy_call,
        job_call,
    ]
    statuses = [c().status_code for c in surface_calls]
    out["surface_all_paths_admit"] = all(s in (200, 202) for s in statuses)
    seventh = _complete(client, u_h)
    out["surface_shared_window_refuses_seventh"] = (
        seventh.status_code == 429 and seventh.json().get("code") == "rate_limited"
    )
    out["surface_window_shared_not_per_route"] = _key_card(client, root_h, u_id)["uses"] == 6

    # GET surfaces meter too (read scope work still spends the budget)
    g_raw, g_id = _mint(client, root_h, rpm=2)
    g_h = {_H_KEY: g_raw}
    get_statuses = [
        client.get(_SELF_PATH, headers=g_h).status_code,
        client.get(_JOBS_PATH, headers=g_h).status_code,
        client.get(_SELF_PATH, headers=g_h).status_code,
    ]
    out["surface_gets_metered"] = get_statuses == [200, 200, 429]

    # /health is public: a probe carrying a credential never meters
    h_raw, h_id = _mint(client, root_h, rpm=1)
    h_h = {_H_KEY: h_raw}
    health = [client.get(_HEALTH_PATH, headers=h_h).status_code for _ in range(3)]
    admitted = _complete(client, h_h)
    out["health_never_meters"] = health == [200, 200, 200] and admitted.status_code == 200
    out["health_key_window_untouched"] = _key_card(client, root_h, h_id)["uses"] == 1

    # an authenticated 404 still counts — uses measures authenticated
    # calls, not successes
    nf_raw, nf_id = _mint(client, root_h)
    nf_h = {_H_KEY: nf_raw}
    missing = client.get("/harness/nonexistent", headers=nf_h)
    out["authenticated_404_bills_use"] = (
        missing.status_code == 404 and _key_card(client, root_h, nf_id)["uses"] == 1
    )

    # eval submissions meter like jobs (async submit, one claim)
    e_raw, e_id = _mint(client, root_h, rpm=1)
    e_h = {_H_KEY: e_raw}
    first_eval = _eval(client, e_h)
    out["eval_submit_metered"] = first_eval.status_code in (202, 200)
    out["eval_second_submit_rate_limited"] = _eval(client, e_h).status_code == 429
    return out


# ---------------------------------------------------------------------------
# error paths — what bills and what doesn't
# ---------------------------------------------------------------------------


def _error_path_probes() -> dict[str, Any]:  # NOSONAR(S3776)
    """The exact contract: ``uses`` counts *authenticated* requests.
    Pre-auth refusals (global limiter, body cap, bad credential) never
    reach the meter; post-auth failures (422 validation, backend 503,
    over-capacity 503, route 404) each spent a presented credential and
    so count — while tokens stay unbilled because no usage existed."""
    out: dict[str, Any] = {}
    client, _ = _client(
        {
            "byok": lambda: _MeterBackend("byok-model", dict(_U6)),
            "hosted_k3": lambda: _FailBackend("dead-0"),
        },
        api_key=_ROOT,
    )
    root_h = {_H_KEY: _ROOT}

    # post-auth failures count a use, never tokens
    f_raw, f_id = _mint(client, root_h)
    f_h = {_H_KEY: f_raw}
    dead = _complete(client, f_h, backend="hosted_k3")
    card = _key_card(client, root_h, f_id)
    out["backend_503_counts_use"] = dead.status_code == 503 and card["uses"] == 1
    out["backend_503_no_tokens"] = card["tokens_used"] == 0

    bad_req = client.post(
        _COMPLETE_PATH, json={"backend": "byok"}, headers=f_h
    )  # missing messages → 422
    card = _key_card(client, root_h, f_id)
    out["postauth_422_counts_use"] = bad_req.status_code == 422 and card["uses"] == 2
    out["postauth_422_no_tokens"] = card["tokens_used"] == 0

    # pre-auth refusals bill nothing
    big = client.post(
        _COMPLETE_PATH,
        content=b"x" * ((1 << 20) + 1),
        headers={**f_h, "Content-Type": "application/json"},
    )
    card = _key_card(client, root_h, f_id)
    out["preauth_413_never_meters"] = big.status_code == 413 and card["uses"] == 2

    # over-capacity refuses after auth — the claim counted. A blocking
    # backend holds the single inflight slot while the second call lands.
    held = _HoldBackend("byok-model")
    client2, _ = _client(
        {"byok": lambda: held},
        api_key=_ROOT,
        max_inflight=1,
    )
    oc_raw, oc_id = _mint(client2, root_h)
    oc_h = {_H_KEY: oc_raw}

    def _hold() -> int:
        response = client2.post(
            _COMPLETE_PATH,
            json={"backend": "byok", "messages": [{"role": "user", "content": "hi"}]},
            headers=oc_h,
        )
        return int(response.status_code)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut = pool.submit(_hold)
        assert held.entered.wait(timeout=5), "blocking backend never entered"
        try:
            second = client2.post(
                _COMPLETE_PATH,
                json={
                    "backend": "byok",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers=oc_h,
            )
        finally:
            held.release()
        fut.result()
    card2 = _key_card(client2, root_h, oc_id)
    out["over_capacity_still_bills_use"] = second.status_code == 503 and card2["uses"] == 2
    out["over_capacity_declares_retry_after"] = (_retry_after(second) or 0) >= 1

    # the global limiter answers before auth — its refusal bills nothing
    client3, _ = _client(
        {"byok": lambda: _MeterBackend("byok-model", dict(_U6))},
        api_key=_ROOT,
        rate_limit_rps=1.0,
    )
    g_raw, g_id = _mint(client3, root_h)
    denied_g = _complete(client3, {_H_KEY: g_raw})
    time.sleep(1.05)  # let the 1 rps bucket refill so the card read admits
    card3 = _key_card(client3, root_h, g_id)
    out["global_limiter_precedes_auth"] = (
        denied_g.status_code == 429
        and denied_g.json().get("code") == "too_many_requests"
        and card3["uses"] == 0
    )
    out["global_limiter_declares_retry_after"] = (_retry_after(denied_g) or 0) >= 1
    return out


# ---------------------------------------------------------------------------
# client retry policy — the wire contract both clients key on
# ---------------------------------------------------------------------------


class _TransportSpy:
    """Counting transport: mirrors ``_urllib_transport``'s serialization
    verbatim and forwards into the TestClient, recording every wire
    attempt so retry behavior is measured, not assumed."""

    def __init__(self, tc: TestClient) -> None:
        self._tc = tc
        self.calls: list[tuple[str, str]] = []

    def __call__(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | bytes | None,
        headers: dict[str, str],
        timeout_s: float,  # NOSONAR(S1172) — transport signature
    ) -> tuple[int, Mapping[str, str], bytes]:
        path = urllib.parse.urlparse(url).path
        if urllib.parse.urlparse(url).query:
            path += "?" + urllib.parse.urlparse(url).query
        self.calls.append((method, path))
        req_headers = {"Accept": "application/json", **headers}
        data: bytes | None
        if isinstance(payload, bytes):
            data = payload
        elif payload is not None:
            data = json.dumps(payload).encode()
            req_headers.setdefault("Content-Type", "application/json")
        else:
            data = None
        r = self._tc.request(method, path, content=data, headers=req_headers)
        return r.status_code, dict(r.headers), r.content


def _client_probes() -> dict[str, Any]:
    from fx1.serve.client import HarnessClient, HarnessTransportError  # noqa: PLC0415

    out: dict[str, Any] = {}
    clock = _SteppedClock()
    client, _api_mod = _client(
        {"byok": lambda: _MeterBackend("byok-model", dict(_U6))},
        api_key=_ROOT,
        key_clock=clock,
    )
    root_h = {_H_KEY: _ROOT}
    keys_mod = _keys_mod()

    # quota_exceeded is terminal: one wire attempt, even under
    # retry_writes + a deep retry budget
    q_raw, _q_id = _mint(client, root_h, max_requests=1)
    spy = _TransportSpy(client)
    hc = HarnessClient(
        "http://quota-audit.dev",
        api_key=q_raw,
        transport=spy,
        max_retries=6,
        retry_writes=True,
        sleep=lambda s: None,
    )
    hc.complete([{"role": "user", "content": "a"}], backend="byok")
    refused_exc: HarnessTransportError | None = None
    try:
        hc.complete([{"role": "user", "content": "b"}], backend="byok")
    except HarnessTransportError as exc:
        refused_exc = exc
    posts = [c for c in spy.calls if c[0] == "POST" and c[1] == _COMPLETE_PATH]
    out["client_never_retries_quota_exceeded"] = refused_exc is not None and len(posts) == 2

    # an unkeyed write retries nothing by default — rate_limited is
    # refused once, the Retry-After hint notwithstanding
    spy2 = _TransportSpy(client)
    hc2 = HarnessClient(
        "http://quota-audit.dev",
        api_key=_mint(client, root_h, rpm=1)[0],
        transport=spy2,
        max_retries=6,
        sleep=lambda s: None,
    )
    hc2.complete([{"role": "user", "content": "a"}], backend="byok")
    refused_exc2: HarnessTransportError | None = None
    try:
        hc2.complete([{"role": "user", "content": "b"}], backend="byok")
    except HarnessTransportError as exc:
        refused_exc2 = exc
    posts2 = [c for c in spy2.calls if c[0] == "POST" and c[1] == _COMPLETE_PATH]
    out["client_unkeyed_write_never_retries"] = refused_exc2 is not None and len(posts2) == 2

    # retry_writes + a Retry-After hint inside the wait budget: the
    # client retries and lands once the (patched) window re-opens
    saved_window = float(getattr(keys_mod, _RATE_WINDOW_CONST))
    try:
        setattr(keys_mod, _RATE_WINDOW_CONST, _TINY_WINDOW_S)
        spy3 = _TransportSpy(client)
        hc3 = HarnessClient(
            "http://quota-audit.dev",
            api_key=_mint(client, root_h, rpm=1)[0],
            transport=spy3,
            max_retries=4,
            retry_writes=True,
            sleep=lambda s: clock.advance(min(s, _WINDOW_RECOVER_S)),
        )
        hc3.complete([{"role": "user", "content": "a"}], backend="byok")
        ok_retry = hc3.complete([{"role": "user", "content": "b"}], backend="byok")
        posts3 = [c for c in spy3.calls if c[0] == "POST" and c[1] == _COMPLETE_PATH]
        out["client_retries_rate_limited_to_admit"] = (
            ok_retry.content == "ok:b" and len(posts3) == 3
        )
    finally:
        setattr(keys_mod, _RATE_WINDOW_CONST, saved_window)

    # a hint beyond max_retry_wait_s abandons immediately — no busy wait
    spy4 = _TransportSpy(client)
    hc4 = HarnessClient(
        "http://quota-audit.dev",
        api_key=_mint(client, root_h, rpm=1)[0],
        transport=spy4,
        max_retries=6,
        retry_writes=True,
        max_retry_wait_s=1.0,
        sleep=lambda s: None,
    )
    hc4.complete([{"role": "user", "content": "a"}], backend="byok")
    refused_exc4: HarnessTransportError | None = None
    try:
        hc4.complete([{"role": "user", "content": "b"}], backend="byok")
    except HarnessTransportError as exc:
        refused_exc4 = exc
    posts4 = [c for c in spy4.calls if c[0] == "POST" and c[1] == _COMPLETE_PATH]
    out["client_abandons_beyond_wait_budget"] = refused_exc4 is not None and len(posts4) == 2
    return out


# ---------------------------------------------------------------------------
# anthropic surface — same refusals, anthropic grammar
# ---------------------------------------------------------------------------


def _anthropic_probes() -> dict[str, Any]:
    out: dict[str, Any] = {}
    client, _ = _client({"byok": lambda: _MeterBackend("byok-model", dict(_U6))}, api_key=_ROOT)
    root_h = {_H_KEY: _ROOT}

    a_raw, _a_id = _mint(client, root_h, rpm=1)
    a_h = {_H_KEY: a_raw}
    ok = _messages(client, a_h)
    out["anthropic_admitted_budget_headers"] = (
        ok.status_code == 200
        and _H_ANTH_LIMIT in {k.lower() for k in ok.headers}
        and _H_ANTH_REMAINING in {k.lower() for k in ok.headers}
        and _H_ANTH_RESET in {k.lower() for k in ok.headers}
    )
    denied = _messages(client, a_h)
    out["anthropic_rate_limited_should_retry"] = (
        denied.status_code == 429
        and denied.headers.get(_H_SHOULD_RETRY) == "true"
        and (_retry_after(denied) or 0) >= 1
    )
    out["anthropic_rate_limited_headers"] = _H_ANTH_LIMIT in {
        k.lower() for k in denied.headers
    } and _H_ANTH_REMAINING in {k.lower() for k in denied.headers}
    out["anthropic_error_envelope"] = denied.json().get("type") == "error"

    q_raw, _q_id = _mint(client, root_h, max_requests=1)
    q_h = {_H_KEY: q_raw}
    assert _messages(client, q_h).status_code == 200
    spent = _messages(client, q_h)
    out["anthropic_quota_exceeded_terminal"] = (
        spent.status_code == 429
        and spent.headers.get(_H_SHOULD_RETRY) == "false"
        and _H_RETRY_AFTER not in {k.lower() for k in spent.headers}
    )

    # the OpenAI leg keeps its own envelope on the same refusal
    o_raw, _o_id = _mint(client, root_h, rpm=1)
    o_h = {_H_KEY: o_raw}
    assert _chat(client, o_h).status_code == 200
    denied_o = _chat(client, o_h)
    o_body = denied_o.json()
    out["openai_refusal_error_envelope"] = (
        denied_o.status_code == 429
        and isinstance(o_body.get("error"), dict)
        and o_body["error"].get("code") == "rate_limited"
    )
    return out


# ---------------------------------------------------------------------------
# store unit probes — the journal contract beneath the wire
# ---------------------------------------------------------------------------


def _store_unit_probes() -> dict[str, Any]:
    out: dict[str, Any] = {}
    from fx1.serve.journal import JobJournal  # noqa: PLC0415
    from fx1.serve.keys import ApiKeyStore, KeyStoreError  # noqa: PLC0415

    # a refusal never moves any counter — the atomic claim under one lock
    t = [1000.0]
    store = ApiKeyStore(journal=None, clock=lambda: t[0])
    raw, _rec = store.mint(rpm=2, max_requests=3)
    first = store.authenticate(raw)
    assert first is not None
    uses_after_first = int(first["uses"])
    try:
        store.authenticate(raw, required_scope="admin")
        out["store_scope_refusal_raises"] = False
    except KeyStoreError as exc:
        out["store_scope_refusal_raises"] = exc.code == "insufficient_scope"
    out["store_refusal_uses_untouched"] = (store.get(store.list()[0]["key_id"] or "") or {}).get(
        "uses"
    ) == uses_after_first

    # replay: the journal's fold reconstructs counters — unit-level pin
    # for the wire probes above (the same contract the restart battery
    # exercises end to end)
    journal_path = _temporary_directory() / "keys.jsonl"
    store_a = ApiKeyStore(journal=JobJournal(journal_path))
    raw_a, rec_a = store_a.mint(max_requests=9, rpm=5)
    store_a.authenticate(raw_a)
    store_a.authenticate(raw_a)
    store_a.charge_tokens(rec_a["key_id"], 7)
    store_b = ApiKeyStore(journal=JobJournal(journal_path))
    reborn = store_b.get(rec_a["key_id"])
    out["store_replay_reconstructs_counters"] = (
        reborn is not None and reborn["uses"] == 2 and reborn["tokens_used"] == 7
    )
    ws = store_b.window_state(rec_a["key_id"])
    out["store_replay_window_is_process_local"] = ws is not None and ws[:2] == (5, 5)
    return out


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def quota_audit() -> dict[str, Any]:
    """Run the boundary battery; returns literal bools."""
    with _audit_context():
        out: dict[str, Any] = {}
        out.update(_rpm_window_probes())
        out.update(_quota_boundary_probes())
        out.update(_token_budget_probes())
        out.update(_scope_refusal_probes())
        out.update(_persistence_probes())
        out.update(_surface_probes())
        out.update(_error_path_probes())
        out.update(_client_probes())
        out.update(_anthropic_probes())
        out.update(_store_unit_probes())
        return out


def quota_audit_bench() -> dict[str, Any]:
    """Sealed receipt: every probe True under quota_audit.v1."""
    r = quota_audit()
    ok = bool(r) and all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "quota_audit",
        "schema": "quota_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "in-process buffered TestClient; stub backends",
            "not_executed": [
                "TypeScript client runtime",
                "network delivery or disconnect timing",
                "process-crash/power-loss durability",
                "live provider billing",
            ],
        },
        "interpretation": (
            "Managed-key limits hold as exact boundaries end to end: the "
            "rpm window admits precisely N calls and refuses N+1 with "
            "Retry-After plus the X-RateLimit-{Limit,Remaining,Reset}"
            "-Requests trio — on refusals and admitted answers alike, "
            "under parallel races with no overshoot, per-key isolated, "
            "and re-admitting on expiry. The request budget decrements "
            "once per admitted call and refuses terminally (quota_"
            "exceeded carries no Retry-After — the Python client does not "
            "retry it); the token budget meters provider usage on sync, "
            "stream, and batch surfaces with the crossing call last "
            "admitted. Refusals consume nothing: no use, no slot, no "
            "tokens — including the 403 scope denial. Spent quota "
            "survives a clean journaled restart (uses and tokens replay; "
            "RPM windows reset as process-local state); a revoked key stays dead and "
            "a rotated successor starts fresh under the predecessor's "
            "policy. Metering is uniform across every authenticated "
            "path — an authenticated 404 counts because uses measures "
            "authenticated calls, while pre-auth refusals (global "
            "limiter, 413 body cap) never reach the meter. Post-auth "
            "failures count a use but never bill tokens. The Anthropic "
            "surface speaks its own retry grammar (x-should-retry, "
            "anthropic-ratelimit-*) over the same refusals."
            if ok
            else f"QUOTA AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(quota_audit_bench(), indent=2, sort_keys=True))
