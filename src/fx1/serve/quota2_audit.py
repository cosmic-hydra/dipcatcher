"""quota2_audit — truthfulness audit of the quota/rpm introspection surfaces.

``quota_audit`` (lane 148) pins the enforcement boundaries themselves:
windows admit exactly ``rpm`` calls, budgets refuse exactly at the
declared count, refusals bill nothing. This battery audits the *other
half* of the contract — what the harness tells a client about its own
budget: the usage cards, the per-request ``X-RateLimit-*`` headers, the
``Retry-After`` hint, and the clock fields. The claim under test: every
number the introspection surfaces report is the truth at read time —
the remaining budget, the window edge, when capacity returns — and the
two refusal modes stay distinguishable (retryable ``rate_limited`` vs
terminal ``quota_exceeded``) at every layer a client can look at.

Coverage map:

- *self surface* — ``GET /harness/self`` reports the calling
  credential's own card: a fresh key's declared budgets, the read
  billing itself (``uses`` counts the introspection call too, so the
  card is post-consume truth), a depleted key's self-read refusing
  ``quota_exceeded`` — the 429 *is* the honest introspection — and a
  penultimate self-read consuming the final budget slot. The env root
  reports ``credential: env`` / ``metered: false`` / ``key: null``;
  loopback dev mode reports ``none``.
- *admin card* — ``GET /harness/keys/{id}/usage`` returns the target's
  spend, never the reader's: the admin's own read bills the admin, not
  the target; the card's derived headroom (``requests_remaining`` /
  ``tokens_remaining`` / ``window_remaining``) is exact at read time;
  the ``served`` split folds the completion ring by credential; a
  non-admin key 403s and an unknown id 404s, both enveloped.
- *reset introspection* — when capacity returns is reported on every
  leg: ``window_reset_s`` on the card (seconds until the declared
  60 s window re-opens, 0 for a fresh/never-started window),
  ``X-RateLimit-Reset-Requests`` on admitted answers and refusals,
  ``Retry-After`` on the retryable 429, and
  ``anthropic-ratelimit-requests-reset`` as an absolute RFC 3339
  instant on the Anthropic leg. The card legs carry no absolute
  ``window_reset_at`` — the relative countdown is the declared form,
  pinned here so a future field can't drift in silently.
- *header truthfulness* — the ``X-RateLimit-{Limit,Remaining,Reset}-
  Requests`` trio reflects the caller's own rpm window at emit time:
  ``Limit`` is the declared ``rpm``, ``Remaining`` decrements once per
  admitted call to exactly 0, a ``rate_limited`` refusal reports
  ``Remaining: 0`` with a bounded ``Retry-After``, and a
  ``quota_exceeded`` refusal deliberately carries no rate-limit
  headers at all (the window is not the binding constraint — faking
  ``remaining=0`` would lie about it). Unwindowed keys and unmetered
  credentials get no headers (no false scarcity).
- *Retry-After semantics* — present only where the refusal is
  retryable: ``rate_limited`` (bounded 1..window seconds) and
  over-capacity/global-limit refusals; absent on ``quota_exceeded``
  (terminal — a hard budget never clears in-call), on 401/403/404
  refusals, and on validation 422s.
- *window-edge policy* — the rpm window is a *fixed* window anchored
  at the consume that opened it, not a per-request sliding window and
  not aligned to calendar minutes: ``reset_s ≈ 60`` immediately after
  the first use; at the elapsed boundary the whole occupancy drops at
  once — a call two-slots deep on the far side of the edge comes back
  with ``Remaining = rpm - 1``, the full recovery a sliding window
  would not give.
- *quota distinction* — clients must never confuse the two 429s:
  ``rate_limited`` is retryable (``Retry-After`` + window headers),
  ``quota_exceeded`` is terminal (no ``Retry-After``, no window
  headers, ``x-should-retry: false`` on the Anthropic leg).
- *per-key independence* — one key's exhaustion never shows on a
  peer's card: a spent key's ``quota_exceeded`` and a full rpm window
  leave the peer's ``/harness/self`` untouched, its own window
  remaining intact.
- *clock honesty* — ``created_at`` is stamped at mint and identical on
  the mint response, the record read, and the usage card;
  ``last_used_at`` is ``None`` until the first authenticated call,
  then the server's own clock — client-supplied headers and body
  fields can never stamp it.
- *persistence* — ``uses``/``tokens_used``/``last_used_at`` are
  journaled and replay into an independently constructed app verbatim;
  ``requests_remaining`` stays honest about the prior spend. The rpm
  window occupancy is the declared process-local policy (pinned by
  ``quota_audit``): ``window_remaining`` re-opens to ``rpm`` in the new
  app while the durable budget keeps its spend — and the ``served``
  ring is this process's evidence, empty in the new app even
  though ``tokens_used`` persists (``log_cap``/``log_dropped`` bound
  it, so the horizon split is declared, not silent).
- *metering precision* — ``tokens_used`` equals the summed
  provider-reported usage exactly; a refused request bills no tokens
  and no use; a post-auth 422 bills the use but never tokens; a
  retried sequence bills only the admitted calls.
- *concurrency* — a parallel burst reports honest window state: every
  admitted answer's ``Remaining`` is a bounded int ``0 <= r < rpm``
  (advisory — the emit-time read can reflect later consumes under
  race), every refusal reports ``Remaining: 0``, and the settled card
  is exact (``window_remaining: 0``).
- *client leg* — ``HarnessClient.key_usage`` / ``self_usage`` deliver
  the same card fields; the SDK in-process twin's card has the
  identical field shape, honest zero-spend on a fresh mint.
- *envelope* — every refusal in this surface family is enveloped on
  the harness leg (``{detail, code}``) and the OpenAI leg
  (``{error: {code}}``) alike.

This is a stub-backed TestClient battery, not network timing,
crash-durability, or live-provider evidence. Sealed ``quota2_audit.v1``
(fx1-side receipt).
"""

from __future__ import annotations

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
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from types import ModuleType

    from fastapi.testclient import TestClient

    from fx1.serve.backends import SamplingParams

__all__ = ["quota2_audit", "quota2_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT = "k3y-material"
_AUDIT_LOCK = threading.Lock()
_RESOURCES: ContextVar[ExitStack] = ContextVar("quota2_audit_resources")


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
    return Path(_RESOURCES.get().enter_context(tempfile.TemporaryDirectory(prefix="quota2_audit_")))


_COMPLETE_PATH = "/harness/complete"
_KEYS_PATH = "/harness/keys"
_SELF_PATH = "/harness/self"
_CHAT_PATH = "/v1/chat/completions"
_MESSAGES_PATH = "/v1/messages"

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
_REAL_WINDOW_S = 60.0
_TINY_WINDOW_S = 0.05
_WINDOW_RECOVER_S = 0.08

_U6 = {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6}


class _MeterBackend:
    """Usage-reporting stub — per-call ``last_usage`` plus the
    cumulative ``total_usage`` the batch delta reads."""

    def __init__(self, model: str, usage: dict[str, int] | None) -> None:
        self._model = model
        self._usage = usage
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

    def close(self) -> None:
        """No resources to release — the stub holds nothing."""


def _client(
    backend_map: dict[str, Any],
    api_key: str | None = None,
    *,
    state_dir: Path | None = None,
    rate_limit_rps: float = 0.0,
    max_inflight: int = 16,
) -> tuple[TestClient, ModuleType]:
    """(TestClient, api_module) — isolated env per construction; backends
    resolve from ``backend_map[name]`` zero-arg factories."""
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
        app = api_mod.create_app(
            harness=Harness(runner=fake_runner),
            backend_resolver=lambda name, *a, **k: backend_map[name](),
            state_dir=state_dir if state_dir is not None else isolated / "state",
            receipts_dir=receipts,
            ft_dir=isolated / "fine_tuning",
            rate_limit_rps=rate_limit_rps,
            max_inflight=max_inflight,
        )
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


def _usage_card(client: TestClient, admin_h: dict[str, str], key_id: str) -> dict[str, Any]:
    r = client.get(f"{_KEYS_PATH}/{key_id}/usage", headers=admin_h)
    assert r.status_code == 200, r.text
    body: dict[str, Any] = r.json()
    return body


def _self_card(client: TestClient, headers: dict[str, str]) -> dict[str, Any]:
    r = client.get(_SELF_PATH, headers=headers)
    assert r.status_code == 200, r.text
    body: dict[str, Any] = r.json()
    return body


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


def _keys_mod() -> ModuleType:
    """The key-store module — the patched-window probes rewrite its
    ``_RATE_WINDOW_S`` and every ``ApiKeyStore`` reads it at call time,
    so no per-app wiring is needed."""
    import fx1.serve.keys as keys_mod  # noqa: PLC0415

    return keys_mod


class _SteppedClock:
    """Key-store clock that moves only when the probe says so.

    A 50 ms wall-clock window expires inside one slow ``complete()`` on a
    loaded runner, so the refusal reads as a fresh admission. Stepping this
    clock keeps the window edge independent of request latency.
    """

    def __init__(self) -> None:
        self.now = time.time()

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@contextmanager
def _pinned_key_clock(clock: _SteppedClock) -> Iterator[None]:
    """Stores constructed inside the block read *clock* instead of ``time.time``."""
    keys_mod = _keys_mod()
    original = keys_mod.ApiKeyStore.__init__

    def _init(self: Any, *args: Any, **kwargs: Any) -> None:
        kwargs["clock"] = clock
        original(self, *args, **kwargs)

    keys_mod.ApiKeyStore.__init__ = _init  # type: ignore[method-assign]
    try:
        yield
    finally:
        keys_mod.ApiKeyStore.__init__ = original  # type: ignore[method-assign]


def _rl_headers(resp: Any) -> dict[str, str]:
    return {k.lower(): v for k, v in resp.headers.items()}


def _retry_after(resp: Any) -> int | None:
    raw = resp.headers.get(_H_RETRY_AFTER)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _rfc3339_instant(value: object) -> float | None:
    """Parse an RFC 3339 instant → epoch seconds, else None."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.utcoffset() is None:
        return None
    return parsed.timestamp()


def _parallel(client: TestClient, calls: list[Callable[[], Any]], workers: int) -> list[Any]:
    """Fire ``calls`` concurrently — the boundary race's overshoot check."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda c: c(), calls))


# ---------------------------------------------------------------------------
# self surface — the calling credential's own card
# ---------------------------------------------------------------------------


def _self_card_probes() -> dict[str, Any]:  # NOSONAR(S3776) — scripted traffic fans out per edge
    out: dict[str, Any] = {}
    client, _ = _client({"byok": lambda: _MeterBackend("byok-model", dict(_U6))}, api_key=_ROOT)
    root_h = {_H_KEY: _ROOT}

    # --- fresh key: the card reports the declared budget, and the read
    # bills itself (uses counts the introspection call too)
    f_raw, _f_id = _mint(client, root_h, max_requests=5, max_tokens=10)
    f_h = {_H_KEY: f_raw}
    first = client.get(_SELF_PATH, headers=f_h)
    body = first.json()
    key = body["key"]
    out["self_reports_managed_metered"] = (
        first.status_code == 200 and body["credential"] == "managed" and body["metered"] is True
    )
    out["self_fresh_key_declared_budget"] = (
        key["max_requests"] == 5 and key["max_tokens"] == 10 and isinstance(key["uses"], int)
    )
    out["self_read_bills_own_use"] = key["uses"] == 1
    out["self_requests_remaining_post_read"] = key["requests_remaining"] == 4
    out["self_fresh_tokens_zero"] = key["tokens_used"] == 0 and key["tokens_remaining"] == 10
    out["self_last_used_at_server_stamped"] = (
        isinstance(key["last_used_at"], (int, float))
        and not isinstance(key["last_used_at"], bool)
        and float(key["created_at"]) <= float(key["last_used_at"]) <= time.time()
    )
    second = client.get(_SELF_PATH, headers=f_h)
    out["self_each_read_counts"] = second.json()["key"]["uses"] == 2

    # --- the env root and loopback dev caller are unmetered
    env = client.get(_SELF_PATH, headers=root_h)
    eb = env.json()
    out["self_env_unmetered"] = (
        eb["credential"] == "env" and eb["metered"] is False and eb["key"] is None
    )
    client_dev, _ = _client({"byok": lambda: _MeterBackend("byok-model", dict(_U6))})
    dev = client_dev.get(_SELF_PATH)
    db = dev.json()
    out["self_dev_none_unmetered"] = (
        db["credential"] == "none" and db["metered"] is False and db["key"] is None
    )

    # --- a depleted key's self-read IS the refusal: quota_exceeded is
    # terminal (no Retry-After) and enveloped — the 429 is the honest
    # introspection; no card exists for a credential that can't auth
    d_raw, _d_id = _mint(client, root_h, max_requests=1)
    d_h = {_H_KEY: d_raw}
    assert _complete(client, d_h).status_code == 200
    depleted = client.get(_SELF_PATH, headers=d_h)
    out["self_depleted_is_quota_refusal"] = (
        depleted.status_code == 429 and depleted.json().get("code") == "quota_exceeded"
    )
    out["self_depleted_terminal_no_retry"] = _retry_after(depleted) is None
    out["self_depleted_enveloped"] = isinstance(depleted.json().get("detail"), str)

    # --- the penultimate read consumes the last budget slot: the card
    # reports exhausted truth (uses==max, remaining==0) and the next
    # call of any kind refuses
    p_raw, _p_id = _mint(client, root_h, max_requests=2)
    p_h = {_H_KEY: p_raw}
    assert _complete(client, p_h).status_code == 200
    penult = client.get(_SELF_PATH, headers=p_h)
    penult_key = penult.json()["key"]
    out["self_penultimate_read_exhausts"] = (
        penult.status_code == 200
        and penult_key["uses"] == 2
        and penult_key["requests_remaining"] == 0
        and penult_key["enabled"] is True
    )
    out["self_penultimate_next_call_refuses"] = _complete(client, p_h).status_code == 429

    # --- a write-only key cannot introspect: read scope is required —
    # and the refusal bills nothing
    w_raw, w_id = _mint(client, root_h, scopes=["write"])
    w_h = {_H_KEY: w_raw}
    denied = client.get(_SELF_PATH, headers=w_h)
    out["self_write_only_refused_403"] = (
        denied.status_code == 403 and denied.json().get("code") == "insufficient_scope"
    )
    out["self_scope_refusal_bills_nothing"] = _usage_card(client, root_h, w_id)["uses"] == 0
    return out


# ---------------------------------------------------------------------------
# admin card — the target's spend, never the reader's
# ---------------------------------------------------------------------------


def _admin_card_probes() -> dict[str, Any]:  # NOSONAR(S3776)
    out: dict[str, Any] = {}
    client, _ = _client({"byok": lambda: _MeterBackend("byok-model", dict(_U6))}, api_key=_ROOT)
    root_h = {_H_KEY: _ROOT}

    admin_raw, _admin_id = _mint(client, root_h, admin=True)
    admin_h = {_H_KEY: admin_raw}
    t_raw, t_id = _mint(client, root_h, rpm=3, max_requests=8, max_tokens=20)
    t_h = {_H_KEY: t_raw}
    _complete(client, t_h)
    _complete(client, t_h)

    card = _usage_card(client, admin_h, t_id)
    out["admin_reads_target_card"] = card["id"] == t_id
    out["admin_card_shows_target_spend"] = card["uses"] == 2 and card["tokens_used"] == 12
    card2 = _usage_card(client, admin_h, t_id)
    out["admin_read_never_bills_target"] = card2["uses"] == 2 and card2["tokens_used"] == 12
    # the two card reads billed the admin's own use counter instead
    admin_self = _self_card(client, admin_h)["key"]
    out["admin_reads_bill_admin_only"] = admin_self["uses"] == 3
    # and the admin's read never touched the target's rpm window
    out["admin_read_untouched_target_window"] = card2["window_remaining"] == 1

    # derived headroom is exact at read time
    out["derived_headroom_exact"] = (
        card["requests_remaining"] == 8 - 2
        and card["tokens_remaining"] == 20 - 12
        and card["window_remaining"] == 3 - 2
    )

    # the served split folds the completion ring by credential
    served = card["served"]
    out["served_matches_metered_calls"] = (
        served["calls"] == 2
        and served["total_tokens"] == 12
        and served["prompt_tokens"] == 4
        and served["completion_tokens"] == 8
        and served["by_backend"].get("byok", {}).get("calls") == 2
    )
    out["served_ring_bounds_declared"] = card["log_cap"] > 0 and card["log_dropped"] == 0

    # uses counts every authenticated call, not just served ones — two
    # self-reads push uses past served.calls honestly
    s_raw, s_id = _mint(client, root_h)
    s_h = {_H_KEY: s_raw}
    client.get(_SELF_PATH, headers=s_h)
    client.get(_SELF_PATH, headers=s_h)
    _complete(client, s_h)
    s_card = _usage_card(client, admin_h, s_id)
    out["uses_counts_reads_not_served"] = s_card["uses"] == 3 and s_card["served"]["calls"] == 1

    # a non-admin managed key can't read the admin surface — 403, enveloped
    na_raw, _na_id = _mint(client, root_h)
    denied = client.get(f"{_KEYS_PATH}/{t_id}/usage", headers={_H_KEY: na_raw})
    out["usage_non_admin_refused_403"] = (
        denied.status_code == 403
        and denied.json().get("code") == "insufficient_scope"
        and _retry_after(denied) is None
    )
    out["usage_non_admin_enveloped"] = isinstance(denied.json().get("detail"), str)

    # an unknown id 404s enveloped — never a fabricated empty card
    missing = client.get(f"{_KEYS_PATH}/fx1k_deadbeefdead/usage", headers=admin_h)
    out["usage_unknown_key_404"] = (
        missing.status_code == 404
        and missing.json().get("code") == "key_not_found"
        and _retry_after(missing) is None
    )

    # a depleted card stays honest: still enabled, budget exhausted, and
    # the unbounded leg reports None rather than faking a bound
    e_raw, e_id = _mint(client, root_h, max_requests=1)
    e_h = {_H_KEY: e_raw}
    assert _complete(client, e_h).status_code == 200
    e_card = _usage_card(client, admin_h, e_id)
    out["depleted_card_still_enabled"] = e_card["enabled"] is True
    out["depleted_card_exact_zero"] = e_card["uses"] == 1 and e_card["requests_remaining"] == 0
    out["unbounded_legs_report_null"] = (
        e_card["tokens_remaining"] is None
        and e_card["window_remaining"] is None
        and e_card["window_reset_s"] is None
    )
    return out


# ---------------------------------------------------------------------------
# reset introspection — when capacity returns, on every leg
# ---------------------------------------------------------------------------


def _reset_surface_probes() -> dict[str, Any]:  # NOSONAR(S3776)
    out: dict[str, Any] = {}
    client, _ = _client({"byok": lambda: _MeterBackend("byok-model", dict(_U6))}, api_key=_ROOT)
    root_h = {_H_KEY: _ROOT}

    # --- the card legs report the relative countdown
    r_raw, r_id = _mint(client, root_h, rpm=3)
    r_h = {_H_KEY: r_raw}
    _complete(client, r_h)
    card = _usage_card(client, root_h, r_id)
    out["card_reports_window_reset_s"] = (
        isinstance(card["window_reset_s"], int) and 0 < card["window_reset_s"] <= 60
    )
    # anchored at the first consume: immediately after the first use the
    # reset sits near the full window length — an epoch-aligned policy
    # would land on a calendar-minute edge instead
    out["reset_anchored_near_window_len"] = card["window_reset_s"] >= 58
    out["card_window_remaining_exact"] = card["window_remaining"] == 2

    # a never-started window declares reset 0 (no live window pending)
    _f_raw, f_id = _mint(client, root_h, rpm=4)
    f_card = _usage_card(client, root_h, f_id)
    out["fresh_window_reset_zero"] = (
        f_card["window_reset_s"] == 0 and f_card["window_remaining"] == 4
    )
    # an unwindowed key reports no reset surface at all
    _u_raw, u_id = _mint(client, root_h)
    u_card = _usage_card(client, root_h, u_id)
    out["unwindowed_no_reset_surface"] = (
        u_card["window_reset_s"] is None and u_card["window_remaining"] is None
    )
    # the card legs carry only the relative countdown — no absolute
    # ``window_reset_at`` exists to pin; pinned so the declared shape
    # can't drift silently (the Anthropic leg is the absolute-instant one)
    out["card_reset_is_relative_only"] = "window_reset_at" not in card and "resets_at" not in card

    # --- the header leg agrees with the card at read time
    k_raw, k_id = _mint(client, root_h, rpm=4)
    k_h = {_H_KEY: k_raw}
    resp = _complete(client, k_h)
    h = _rl_headers(resp)
    k_card = _usage_card(client, root_h, k_id)
    out["header_reset_bounded_window"] = (
        h.get(_H_RL_RESET) is not None and 0 <= int(h[_H_RL_RESET]) <= 60
    )
    out["header_reset_agrees_with_card"] = (
        k_card["window_reset_s"] is not None
        and abs(int(h[_H_RL_RESET]) - int(k_card["window_reset_s"])) <= 1
    )

    # --- the refusal legs: Retry-After and the window header agree within
    # one second (ceil vs round on two reads)
    rl_raw, rl_id = _mint(client, root_h, rpm=1)
    rl_h = {_H_KEY: rl_raw}
    assert _complete(client, rl_h).status_code == 200
    refused = _complete(client, rl_h)
    rh = _rl_headers(refused)
    retry = _retry_after(refused)
    out["refusal_retry_after_bounded"] = (
        refused.status_code == 429 and retry is not None and 1 <= retry <= int(_REAL_WINDOW_S)
    )
    out["refusal_reset_consistent_with_retry"] = (
        rh.get(_H_RL_RESET) is not None and abs(int(rh[_H_RL_RESET]) - int(retry or 0)) <= 1
    )
    out["refusal_headers_match_live_window"] = (
        rh.get(_H_RL_REMAINING) == "0" and rh.get(_H_RL_LIMIT) == "1"
    )
    # the card agrees with the refusal: the refused key's window is full
    rl_card = _usage_card(client, root_h, rl_id)
    out["refusal_card_window_full"] = rl_card["window_remaining"] == 0

    # --- the Anthropic leg reports the reset as an absolute instant
    a_raw, _a_id = _mint(client, root_h, rpm=2)
    a_h = {_H_KEY: a_raw}
    admitted = _messages(client, a_h)
    ah = _rl_headers(admitted)
    instant = _rfc3339_instant(ah.get(_H_ANTH_RESET))
    out["anthropic_reset_absolute_instant"] = (
        admitted.status_code == 200
        and instant is not None
        and time.time() - 2 <= instant <= time.time() + 61
    )
    out["anthropic_admitted_trio_present"] = (
        ah.get(_H_ANTH_LIMIT) == "2" and ah.get(_H_ANTH_REMAINING) == "1"
    )

    # --- quota budgets report no reset surface at all: a hard budget
    # never returns inside a call — honest absence, not a countdown
    q_raw, q_id = _mint(client, root_h, max_requests=1)
    q_h = {_H_KEY: q_raw}
    _complete(client, q_h)
    q_card = _usage_card(client, root_h, q_id)
    out["quota_budget_reports_no_reset"] = (
        q_card["window_reset_s"] is None and q_card["requests_remaining"] == 0
    )
    return out


# ---------------------------------------------------------------------------
# X-RateLimit header truthfulness
# ---------------------------------------------------------------------------


def _header_surface_probes() -> dict[str, Any]:  # NOSONAR(S3776)
    out: dict[str, Any] = {}
    client, _ = _client({"byok": lambda: _MeterBackend("byok-model", dict(_U6))}, api_key=_ROOT)
    root_h = {_H_KEY: _ROOT}

    # --- the trio reflects the caller's own window, decrementing exactly
    r_raw, _r_id = _mint(client, root_h, rpm=3)
    r_h = {_H_KEY: r_raw}
    seq = [_complete(client, r_h) for _ in range(3)]
    remaining_seen = [int(r.headers.get(_H_RL_REMAINING, "-1")) for r in seq]
    out["headers_remaining_decrements_exact"] = remaining_seen == [2, 1, 0]
    out["headers_limit_is_declared_rpm"] = all(r.headers.get(_H_RL_LIMIT) == "3" for r in seq)
    out["headers_int_valued"] = all(
        (r.headers.get(_H_RL_LIMIT) or "").isdigit()
        and (r.headers.get(_H_RL_REMAINING) or "").isdigit()
        and (r.headers.get(_H_RL_RESET) or "").isdigit()
        for r in seq
    )
    refused = _complete(client, r_h)
    out["refusal_headers_remaining_zero"] = (
        refused.status_code == 429 and refused.headers.get(_H_RL_REMAINING) == "0"
    )
    out["refusal_headers_limit_honest"] = refused.headers.get(_H_RL_LIMIT) == "3"

    # --- unwindowed keys and the env root get no budget headers — no
    # false scarcity on credentials that declare no window
    u_raw, _u_id = _mint(client, root_h)
    u_resp = _complete(client, {_H_KEY: u_raw})
    uh = _rl_headers(u_resp)
    out["unwindowed_key_no_budget_headers"] = (
        _H_RL_LIMIT not in uh and _H_RL_REMAINING not in uh and _H_RL_RESET not in uh
    )
    root_resp = _complete(client, root_h)
    rooth = _rl_headers(root_resp)
    out["env_key_no_budget_headers"] = _H_RL_LIMIT not in rooth and _H_RL_REMAINING not in rooth

    # --- the terminal quota refusal carries no rate-limit headers: the
    # window is not the binding constraint, so the refusal refuses to
    # fake a window state — clients distinguish terminal from retryable
    q_raw, _q_id = _mint(client, root_h, max_requests=1, rpm=5)
    q_h = {_H_KEY: q_raw}
    assert _complete(client, q_h).status_code == 200
    q_refused = _complete(client, q_h)
    qh = _rl_headers(q_refused)
    out["quota_refusal_no_rate_headers"] = (
        q_refused.status_code == 429
        and _H_RL_LIMIT not in qh
        and _H_RL_REMAINING not in qh
        and _H_RL_RESET not in qh
    )
    out["quota_refusal_distinct_from_rpm"] = (
        q_refused.json().get("code") == "quota_exceeded" and _retry_after(q_refused) is None
    )

    # --- every authenticated read reports the caller's window too —
    # /harness/self consumes the key's own slot and the headers agree
    # with the card it carries
    s_raw, _s_id = _mint(client, root_h, rpm=2)
    s_h = {_H_KEY: s_raw}
    self_resp = client.get(_SELF_PATH, headers=s_h)
    sh = _rl_headers(self_resp)
    self_key = self_resp.json()["key"]
    out["self_headers_report_own_window"] = (
        sh.get(_H_RL_REMAINING) == "1" and sh.get(_H_RL_LIMIT) == "2"
    )
    out["self_card_matches_headers"] = (
        self_key["window_remaining"] == 1
        and abs(int(sh.get(_H_RL_RESET, "-1")) - int(self_key["window_reset_s"])) <= 1
    )
    return out


# ---------------------------------------------------------------------------
# window-edge policy — fixed anchor, declared and measured
# ---------------------------------------------------------------------------


def _window_edge_probes() -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys_mod = _keys_mod()
    root_h = {_H_KEY: _ROOT}

    # --- fixed-window anchor: a two-deep window drops whole at the
    # elapsed boundary — a sliding window would still hold the second
    # call's slot. Window 1.2s; call1 at t≈0, call2 at t≈0.7 (full),
    # call3 at t≈1.4 — past the anchor, inside call2's sliding expiry.
    saved_window = float(getattr(keys_mod, _RATE_WINDOW_CONST))
    clock = _SteppedClock()
    try:
        with _pinned_key_clock(clock):
            setattr(keys_mod, _RATE_WINDOW_CONST, 1.2)
            client, _ = _client(
                {"byok": lambda: _MeterBackend("byok-model", dict(_U6))}, api_key=_ROOT
            )
            w_raw, w_id = _mint(client, root_h, rpm=2)
            w_h = {_H_KEY: w_raw}
            assert _complete(client, w_h).status_code == 200
            clock.advance(0.7)
            assert _complete(client, w_h).status_code == 200
            clock.advance(0.7)
            third = _complete(client, w_h)
            # sliding would report remaining 0 (call2's slot still live);
            # the declared fixed anchor drops the whole occupancy
            out["window_fixed_anchor_drops_whole"] = (
                third.status_code == 200 and third.headers.get(_H_RL_REMAINING) == "1"
            )
            w_card = _usage_card(client, root_h, w_id)
            out["window_recovery_card_fresh"] = w_card["window_remaining"] == 1

            # --- expiry re-admits and the surfaces report the fresh count.
            # The refusal is issued at the same instant as the admission, then
            # the clock steps past the window. Wall time between the calls
            # must not open a new window on its own.
            setattr(keys_mod, _RATE_WINDOW_CONST, _TINY_WINDOW_S)
            e_raw, e_id = _mint(client, root_h, rpm=1)
            e_h = {_H_KEY: e_raw}
            first = _complete(client, e_h)
            denied = _complete(client, e_h)
            clock.advance(_WINDOW_RECOVER_S)
            third2 = _complete(client, e_h)
            out["window_expiry_readmits"] = (
                first.status_code == 200 and denied.status_code == 429 and third2.status_code == 200
            )
            out["window_recovery_remaining_fresh"] = third2.headers.get(_H_RL_REMAINING) == "0"
            e_card = _usage_card(client, root_h, e_id)
            out["window_recovery_card_honest"] = e_card["window_remaining"] == 0
    finally:
        setattr(keys_mod, _RATE_WINDOW_CONST, saved_window)

    # --- PATCH relief: removing rpm clears the window surface honestly —
    # the very next call admits, no rate headers, the card's window
    # fields go null instead of freezing at the old occupancy
    client2, _ = _client({"byok": lambda: _MeterBackend("byok-model", dict(_U6))}, api_key=_ROOT)
    p_raw, p_id = _mint(client2, root_h, rpm=1)
    p_h = {_H_KEY: p_raw}
    assert _complete(client2, p_h).status_code == 200
    assert _complete(client2, p_h).status_code == 429
    patched = client2.patch(f"{_KEYS_PATH}/{p_id}", json={"rpm": None}, headers=root_h)
    assert patched.status_code == 200, patched.text
    relief = _complete(client2, p_h)
    out["patch_rpm_relief_admits"] = relief.status_code == 200
    rh = _rl_headers(relief)
    out["patch_rpm_relief_headers_gone"] = _H_RL_REMAINING not in rh
    p_card = _usage_card(client2, root_h, p_id)
    out["patch_rpm_card_window_null"] = (
        p_card["window_remaining"] is None
        and p_card["window_reset_s"] is None
        and p_card["rpm"] is None
    )
    return out


# ---------------------------------------------------------------------------
# per-key independence — one key's exhaustion never leaks
# ---------------------------------------------------------------------------


def _independence_probes() -> dict[str, Any]:
    out: dict[str, Any] = {}
    client, _ = _client({"byok": lambda: _MeterBackend("byok-model", dict(_U6))}, api_key=_ROOT)
    root_h = {_H_KEY: _ROOT}

    a_raw, _a_id = _mint(client, root_h, max_requests=1, rpm=1, max_tokens=6)
    a_h = {_H_KEY: a_raw}
    _complete(client, a_h)
    assert _complete(client, a_h).status_code == 429  # quota dead
    b_raw, b_id = _mint(client, root_h, max_requests=5, rpm=3, max_tokens=30)
    b_h = {_H_KEY: b_raw}
    b_card = _self_card(client, b_h)["key"]
    # the peer's card is untouched: its own declared budget minus its own
    # reads only — the dead key's exhaustion and spent tokens never leak
    out["peer_quota_isolated"] = (
        b_card["uses"] == 1 and b_card["requests_remaining"] == 4 and b_card["max_requests"] == 5
    )
    out["peer_window_isolated"] = b_card["window_remaining"] == 2 and b_card["rpm"] == 3
    out["peer_tokens_isolated"] = b_card["tokens_used"] == 0 and b_card["tokens_remaining"] == 30
    # and the peer still admits — its window and budget are its own
    admitted = _complete(client, b_h)
    out["peer_still_admits"] = admitted.status_code == 200
    b_card2 = _usage_card(client, root_h, b_id)
    out["peer_spend_is_own"] = b_card2["uses"] == 2 and b_card2["tokens_used"] == 6
    return out


# ---------------------------------------------------------------------------
# clock honesty — server-stamped fields only
# ---------------------------------------------------------------------------


def _clock_probes() -> dict[str, Any]:
    out: dict[str, Any] = {}
    client, _ = _client({"byok": lambda: _MeterBackend("byok-model", dict(_U6))}, api_key=_ROOT)
    root_h = {_H_KEY: _ROOT}

    # --- created_at: one server stamp, identical on every leg
    r = client.post(_KEYS_PATH, json={}, headers=root_h)
    mint = r.json()
    raw, kid = str(mint["key"]), str(mint["id"])
    get_rec = client.get(f"{_KEYS_PATH}/{kid}", headers=root_h).json()
    card = _usage_card(client, root_h, kid)
    out["created_at_consistent_across_legs"] = (
        mint["created_at"] == get_rec["created_at"] == card["created_at"]
    )
    out["created_at_server_time"] = (
        isinstance(mint["created_at"], (int, float))
        and abs(float(mint["created_at"]) - time.time()) < 60
    )

    # --- last_used_at: None until the first authenticated call, then
    # the server's own clock — never client-supplied
    out["last_used_at_none_until_use"] = card["last_used_at"] is None
    kh = {_H_KEY: raw}
    before = time.time()
    forged = client.get(
        _SELF_PATH,
        headers={
            **kh,
            "X-Last-Used-At": "0",
            "X-Created-At": "0",
        },
    )
    after = time.time()
    used_card = forged.json()["key"]
    lua = used_card["last_used_at"]
    out["client_headers_cannot_stamp"] = (
        isinstance(lua, (int, float)) and before <= float(lua) <= after and float(lua) > 0
    )
    out["last_used_at_monotone_per_use"] = float(
        _self_card(client, kh)["key"]["last_used_at"]
    ) >= float(lua)
    return out


# ---------------------------------------------------------------------------
# persistence — the journaled card across restart
# ---------------------------------------------------------------------------


def _persistence_probes() -> dict[str, Any]:  # NOSONAR(S3776)
    """Two ``create_app`` constructions over one ``--state-dir`` stand in
    for two processes: only the journal carries over. ``uses``,
    ``tokens_used`` and ``last_used_at`` are journaled truth; the rpm
    window and the served ring are the declared process-local horizon —
    the card must say so honestly, not fake a reset budget."""
    out: dict[str, Any] = {}
    state_dir = _temporary_directory()
    backends: dict[str, Any] = {"byok": lambda: _MeterBackend("byok-model", dict(_U6))}
    root_h = {_H_KEY: _ROOT}

    # process A — mint and spend
    client_a, _ = _client(backends, api_key=_ROOT, state_dir=state_dir)
    p_raw, p_id = _mint(client_a, root_h, max_requests=5, max_tokens=20, rpm=2)
    p_h = {_H_KEY: p_raw}
    for _ in range(2):
        assert _complete(client_a, p_h).status_code == 200
    card_a = _usage_card(client_a, root_h, p_id)
    lua_a = card_a["last_used_at"]

    # process B — same --state-dir
    client_b, _ = _client(backends, api_key=_ROOT, state_dir=state_dir)
    card_b = _usage_card(client_b, root_h, p_id)
    out["persist_uses_across_restart"] = card_b["uses"] == 2
    out["persist_tokens_across_restart"] = card_b["tokens_used"] == 12
    out["persist_last_used_at_exact"] = card_b["last_used_at"] == lua_a
    out["persist_requests_remaining_honest"] = card_b["requests_remaining"] == 3
    # the window is declared process-local: the restarted card reports
    # the fresh window (remaining=rpm), NOT a laundered lifetime budget
    out["persist_window_declared_process_local"] = card_b["window_remaining"] == 2
    # the served ring is this process's evidence — empty after restart
    # while the durable tokens persist; log_dropped bounds it honestly
    out["persist_served_process_local"] = (
        card_b["served"]["calls"] == 0 and card_b["tokens_used"] == 12
    )
    out["persist_ring_bounds_declared"] = card_b["log_cap"] > 0 and card_b["log_dropped"] == 0
    # the restarted key's next admitted call reports its fresh window
    # truthfully (Remaining=rpm-1) and the durable counters keep counting
    nxt = _complete(client_b, p_h)
    card_b2 = _usage_card(client_b, root_h, p_id)
    out["persist_next_admit_remaining_honest"] = (
        nxt.status_code == 200 and nxt.headers.get(_H_RL_REMAINING) == "1"
    )
    out["persist_next_use_counts_durable"] = card_b2["uses"] == 3 and card_b2["tokens_used"] == 18
    return out


# ---------------------------------------------------------------------------
# metering precision — refused vs admitted billing
# ---------------------------------------------------------------------------


def _metering_probes() -> dict[str, Any]:
    out: dict[str, Any] = {}
    meter = _MeterBackend("byok-model", dict(_U6))
    client, _ = _client({"byok": lambda: meter}, api_key=_ROOT)
    root_h = {_H_KEY: _ROOT}

    # --- a rate-limit refusal bills nothing: no use, no tokens, and the
    # request never reaches the backend (the refusal is pre-dispatch)
    r_raw, r_id = _mint(client, root_h, rpm=1)
    r_h = {_H_KEY: r_raw}
    _complete(client, r_h)
    assert meter.calls == 1
    refused = _complete(client, r_h)
    r_card = _usage_card(client, root_h, r_id)
    out["rate_refusal_bills_nothing"] = (
        refused.status_code == 429 and r_card["uses"] == 1 and r_card["tokens_used"] == 6
    )
    out["rate_refusal_no_backend_call"] = meter.calls == 1

    # --- a post-auth failure bills the use, never tokens
    f_raw, f_id = _mint(client, root_h)
    f_h = {_H_KEY: f_raw}
    bad = client.post(_COMPLETE_PATH, json={"backend": "byok"}, headers=f_h)
    f_card = _usage_card(client, root_h, f_id)
    out["postauth_422_bills_use_not_tokens"] = (
        bad.status_code == 422
        and _retry_after(bad) is None
        and f_card["uses"] == 1
        and f_card["tokens_used"] == 0
    )

    # --- a retried sequence bills only the admitted calls: refuse →
    # admit → refuse nets exactly one use and one charge
    keys_mod = _keys_mod()
    saved = float(getattr(keys_mod, _RATE_WINDOW_CONST))
    clock = _SteppedClock()
    try:
        with _pinned_key_clock(clock):
            setattr(keys_mod, _RATE_WINDOW_CONST, _TINY_WINDOW_S)
            timed, _ = _client(
                {"byok": lambda: _MeterBackend("byok-model", dict(_U6))}, api_key=_ROOT
            )
            t_raw, t_id = _mint(timed, root_h, rpm=1)
            t_h = {_H_KEY: t_raw}
            first = _complete(timed, t_h)
            denied = _complete(timed, t_h)
            clock.advance(_WINDOW_RECOVER_S)
            second = _complete(timed, t_h)
            t_card = _usage_card(timed, root_h, t_id)
        out["retry_sequence_bills_admitted_only"] = (
            first.status_code == 200
            and denied.status_code == 429
            and second.status_code == 200
            and t_card["uses"] == 2
            and t_card["tokens_used"] == 12
        )
    finally:
        setattr(keys_mod, _RATE_WINDOW_CONST, saved)

    # --- tokens_used is the exact provider-reported sum, no more
    m_raw, m_id = _mint(client, root_h)
    m_h = {_H_KEY: m_raw}
    for _ in range(3):
        _complete(client, m_h)
    m_card = _usage_card(client, root_h, m_id)
    out["tokens_equal_provider_sum"] = m_card["tokens_used"] == 18
    out["tokens_served_agree"] = m_card["served"]["total_tokens"] == 18
    return out


# ---------------------------------------------------------------------------
# concurrency — honest window reporting under race
# ---------------------------------------------------------------------------


def _concurrency_probes() -> dict[str, Any]:
    out: dict[str, Any] = {}
    client, _ = _client(
        {"byok": lambda: _MeterBackend("byok-model", dict(_U6))},
        api_key=_ROOT,
        max_inflight=32,
    )
    root_h = {_H_KEY: _ROOT}
    r_raw, r_id = _mint(client, root_h, rpm=8)
    r_h = {_H_KEY: r_raw}
    race = _parallel(client, [lambda: _complete(client, r_h)] * 16, 16)
    admitted = [r for r in race if r.status_code == 200]
    refusals = [r for r in race if r.status_code == 429]
    out["race_exact_admission"] = len(admitted) == 8 and len(refusals) == 8
    # advisory truthfulness: every admitted header is a bounded, honest
    # read of the window at emit time — never negative, never over-limit
    out["race_remaining_bounded"] = all(
        (r.headers.get(_H_RL_REMAINING) or "").isdigit()
        and 0 <= int(r.headers[_H_RL_REMAINING]) < 8
        for r in admitted
    )
    out["race_refusals_remaining_zero"] = all(
        r.headers.get(_H_RL_REMAINING) == "0" for r in refusals
    )
    out["race_refusals_retry_bounded"] = all((_retry_after(r) or 0) >= 1 for r in refusals)
    card = _usage_card(client, root_h, r_id)
    out["race_settled_card_exact"] = card["window_remaining"] == 0 and card["uses"] == 8
    out["race_next_call_refuses_honestly"] = _complete(client, r_h).status_code == 429
    return out


# ---------------------------------------------------------------------------
# client + SDK legs — the same card through the libraries
# ---------------------------------------------------------------------------


class _TransportSpy:
    """Counting transport: mirrors ``_urllib_transport``'s serialization
    verbatim and forwards into the TestClient."""

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


def _client_leg_probes() -> dict[str, Any]:
    from fx1.sdk import Fx1Harness  # noqa: PLC0415
    from fx1.serve.client import HarnessClient  # noqa: PLC0415

    out: dict[str, Any] = {}
    client, _ = _client({"byok": lambda: _MeterBackend("byok-model", dict(_U6))}, api_key=_ROOT)
    root_h = {_H_KEY: _ROOT}

    # --- HarnessClient sees the same card the wire serves
    t_raw, t_id = _mint(client, root_h, max_requests=6)
    t_h = {_H_KEY: t_raw}
    _complete(client, t_h)
    hc = HarnessClient(
        "http://quota2-audit.dev",
        api_key=_ROOT,
        transport=_TransportSpy(client),
        sleep=lambda s: None,
    )
    card = hc.key_usage(t_id)
    out["client_key_usage_matches_wire"] = (
        card["id"] == t_id and card["uses"] == 1 and card["requests_remaining"] == 5
    )
    self_card = HarnessClient(
        "http://quota2-audit.dev",
        api_key=t_raw,
        transport=_TransportSpy(client),
        sleep=lambda s: None,
    ).self_usage()
    out["client_self_usage_managed"] = (
        self_card["credential"] == "managed" and self_card["key"]["uses"] == 2
    )

    # --- the SDK in-process twin carries the identical card shape and
    # honest zero-spend on a fresh mint
    sdk = Fx1Harness()
    mint = sdk.key_create("audit", rpm=3, max_requests=4)
    sdk_card = sdk.key_usage(mint["id"])
    wire_card = _usage_card(client, root_h, t_id)
    out["sdk_card_shape_matches_wire"] = set(sdk_card.keys()) == set(wire_card.keys())
    out["sdk_card_honest_zero_spend"] = (
        sdk_card["uses"] == 0
        and sdk_card["tokens_used"] == 0
        and sdk_card["last_used_at"] is None
        and sdk_card["requests_remaining"] == 4
        and sdk_card["window_remaining"] == 3
        and sdk_card["window_reset_s"] == 0
    )
    return out


# ---------------------------------------------------------------------------
# envelope — every refusal in this family, both grammars
# ---------------------------------------------------------------------------


def _envelope_probes() -> dict[str, Any]:
    out: dict[str, Any] = {}
    client, _ = _client({"byok": lambda: _MeterBackend("byok-model", dict(_U6))}, api_key=_ROOT)
    root_h = {_H_KEY: _ROOT}

    # harness leg — flat {detail, code}
    r_raw, _r_id = _mint(client, root_h, rpm=1)
    r_h = {_H_KEY: r_raw}
    _complete(client, r_h)
    rl = _complete(client, r_h)
    rl_body = rl.json()
    out["rate_limited_harness_enveloped"] = (
        rl.status_code == 429
        and isinstance(rl_body.get("detail"), str)
        and rl_body.get("code") == "rate_limited"
    )
    q_raw, _q_id = _mint(client, root_h, max_requests=1)
    q_h = {_H_KEY: q_raw}
    _complete(client, q_h)
    qe = _complete(client, q_h)
    qe_body = qe.json()
    out["quota_exceeded_harness_enveloped"] = (
        qe.status_code == 429
        and isinstance(qe_body.get("detail"), str)
        and qe_body.get("code") == "quota_exceeded"
        and _retry_after(qe) is None
        and not any(name.startswith("x-ratelimit-") for name in _rl_headers(qe))
    )
    unauth = _complete(client, {_H_KEY: "fx1k_forged"})
    out["unauthorized_401_enveloped"] = (
        unauth.status_code == 401
        and isinstance(unauth.json().get("detail"), str)
        and _retry_after(unauth) is None
    )
    missing = client.get("/harness/keys/fx1k_deadbeef/usage", headers=root_h)
    out["not_found_404_enveloped"] = (
        missing.status_code == 404
        and isinstance(missing.json().get("detail"), str)
        and _retry_after(missing) is None
    )

    # openai leg — {error: {code, message, type}}
    oraw, _o_id = _mint(client, root_h, rpm=1)
    o_h = {_H_KEY: oraw}
    _chat(client, o_h)
    orl = _chat(client, o_h)
    orl_err = orl.json().get("error")
    out["rate_limited_openai_enveloped"] = (
        orl.status_code == 429
        and isinstance(orl_err, dict)
        and orl_err.get("code") == "rate_limited"
    )
    oq_raw, _oq_id = _mint(client, root_h, max_requests=1)
    oq_h = {_H_KEY: oq_raw}
    _chat(client, oq_h)
    oqe = _chat(client, oq_h)
    oqe_err = oqe.json().get("error")
    out["quota_exceeded_openai_enveloped"] = (
        oqe.status_code == 429
        and isinstance(oqe_err, dict)
        and oqe_err.get("code") == "quota_exceeded"
        and _retry_after(oqe) is None
        and not any(name.startswith("x-ratelimit-") for name in _rl_headers(oqe))
    )
    # the anthropic leg keeps its own retry grammar distinct
    am_raw, _am_id = _mint(client, root_h, max_requests=1)
    am_h = {_H_KEY: am_raw}
    _messages(client, am_h)
    ame = _messages(client, am_h)
    out["quota_exceeded_anthropic_terminal"] = (
        ame.status_code == 429
        and ame.headers.get(_H_SHOULD_RETRY) == "false"
        and _retry_after(ame) is None
        and isinstance(ame.json().get("error"), dict)
        and not any(name.startswith("anthropic-ratelimit-") for name in _rl_headers(ame))
    )
    return out


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def quota2_audit() -> dict[str, Any]:
    """Run the introspection-truthfulness battery; returns literal bools."""
    with _audit_context():
        out: dict[str, Any] = {}
        out.update(_self_card_probes())
        out.update(_admin_card_probes())
        out.update(_reset_surface_probes())
        out.update(_header_surface_probes())
        out.update(_window_edge_probes())
        out.update(_independence_probes())
        out.update(_clock_probes())
        out.update(_persistence_probes())
        out.update(_metering_probes())
        out.update(_concurrency_probes())
        out.update(_client_leg_probes())
        out.update(_envelope_probes())
        return out


def quota2_audit_bench() -> dict[str, Any]:
    """Sealed receipt: every probe True under quota2_audit.v1."""
    r = quota2_audit()
    ok = len(r) == 107 and all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "quota2_audit",
        "schema": "quota2_audit.v1",
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
            "The quota/rpm introspection surface reports the truth at read "
            "time on every leg: /harness/self shows the calling key's own "
            "card post-consume (the read bills itself; a depleted key's "
            "self-read is itself the terminal quota_exceeded refusal), and "
            "/harness/keys/{id}/usage returns the target's spend while the "
            "admin's read bills only the admin. When capacity returns is "
            "declared on all four legs — window_reset_s on the cards, "
            "X-RateLimit-Reset-Requests on admitted answers and refusals, "
            "bounded Retry-After on the retryable 429, and an absolute "
            "RFC 3339 instant on the Anthropic leg — while the card legs "
            "carry only the relative form (no absolute window_reset_at). "
            "The rpm window is a fixed anchor at first consume (not "
            "sliding, not calendar-aligned): reset_s ≈ 60 after first "
            "use and the whole occupancy drops at the elapsed edge. "
            "Headers decrement once per admitted call to an exact zero; "
            "quota_exceeded carries no rate headers at all (the window "
            "is not the binding constraint), keeping the terminal "
            "refusal distinguishable from the retryable one. One key's "
            "exhaustion never leaks onto a peer's card; created_at and "
            "last_used_at are server-stamped (None until first use, "
            "client headers can't forge them); uses, tokens and "
            "last_used_at replay into a new app from --state-dir while the "
            "window and the served ring stay honestly process-local. Refusals "
            "bill nothing, post-auth failures bill the use but never "
            "tokens, and every refusal lands enveloped on both grammars."
            if ok
            else f"QUOTA2 AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(quota2_audit_bench(), indent=2, sort_keys=True))
