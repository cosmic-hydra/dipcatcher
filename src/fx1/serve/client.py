"""HarnessClient — the HTTP twin of ``Fx1Harness`` for remote deployments.

``fx1.sdk`` is the in-process surface; this module is the same contract
over the wire for callers that point at a ``fx1 harness serve`` instance —
fx-1 pods, other services, notebooks. It deliberately returns the SDK's own
result types (``CompletionResult``, ``ReceiptVerdict``, ``HarnessHealth``,
``HarnessResult``) so switching between in-process and remote is a
one-line construction change.

Error mapping mirrors the SDK taxonomy (the wire's status codes map back
to the same exception classes the SDK raises):

- 404                        ``KeyError``
- 422                        ``ValueError``
- 503                        ``BackendNotConfiguredError``
- 501                        ``NotImplementedError``
- 502 ``honesty gate``       ``Fx1HonestyError``
- 502 other                  ``RuntimeError``
- 401/403                    ``HarnessAuthError``
- transport / other          ``HarnessTransportError``

``transport`` is injectable: ``(method, url, json_payload, headers,
timeout_s) -> (status, headers, body_bytes)``. Production uses urllib;
tests route it at a ``fastapi.testclient.TestClient``.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from fx1 import __version__
from fx1.harness import HarnessResult
from fx1.honesty import Fx1HonestyError
from fx1.sdk import (
    CompletionRecord,
    CompletionResult,
    GateCheckResult,
    HarnessHealth,
    OpsMetrics,
    ProbeResult,
    ReceiptRef,
    ReceiptVerdict,
    StoredReceipt,
)
from fx1.serve.backends import BackendNotConfiguredError as BackendNotConfiguredError
from fx1.serve.contract import API_VERSION as EXPECTED_API_VERSION
from fx1.serve.usage_report import UsageReport

__all__ = [
    "BackendNotConfiguredError",
    "EXPECTED_API_VERSION",
    "HarnessAuthError",
    "HarnessClient",
    "HarnessCompatError",
    "HarnessJobError",
    "HarnessTransportError",
]

Transport = Callable[
    [str, str, dict[str, Any] | bytes | None, dict[str, str], float],
    "tuple[int, Mapping[str, str], bytes]",
]

# Patch three-state: an omitted kwarg keeps the declared policy while
# an explicit ``None`` sends the JSON-null that clears the bound.
_UNSET: Any = object()


class HarnessTransportError(RuntimeError):
    """Network-level failure talking to the harness API."""

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class HarnessAuthError(PermissionError):
    """401/403 from the harness API — key missing or wrong."""

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class HarnessJobError(RuntimeError):
    """An async run job reached its terminal state without a result —
    the worker captured an exception (``status == 'failed'``)."""

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class HarnessCompatError(RuntimeError):
    """The remote harness speaks a wire contract this client can't parse —
    raised by ``check_compat`` when the server's ``api_version`` differs
    from ``EXPECTED_API_VERSION`` (or the peer predates versioning)."""

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


def _urllib_transport(
    method: str,
    url: str,
    payload: dict[str, Any] | bytes | None,
    headers: dict[str, str],
    timeout_s: float,
) -> tuple[int, Mapping[str, str], bytes]:
    req_headers = {"Accept": "application/json", **headers}
    data = None
    if isinstance(payload, bytes):
        # raw upload bytes (multipart file posts) — Content-Type rides
        # the caller's headers, never JSON-encoded.
        data = payload
    elif payload is not None:
        data = json.dumps(payload).encode()
        req_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 — URL is validated at construction  # nosec B310
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        # HTTPError owns the rejected response/socket.  The normal ``with``
        # path cannot close it because urlopen raises before it is entered.
        try:
            body = exc.read() if exc.fp is not None else b""
            return exc.code, dict(exc.headers or {}), body
        finally:
            with contextlib.suppress(Exception):
                exc.close()
    except urllib.error.URLError as exc:
        raise HarnessTransportError(f"harness unreachable at {url}: {exc.reason}") from exc
    except (http.client.HTTPException, OSError) as exc:
        # socket faults before a status line exists — e.g. a mid-upload
        # reset when the server rejects early on headers alone (413
        # body-cap) — are transport errors under the same contract as
        # URLError; don't leak raw errno classes to callers.
        raise HarnessTransportError(f"harness transport fault at {url}: {exc}") from exc


def _retry_after_s(headers: Mapping[str, str]) -> float | None:
    """Parse a Retry-After seconds hint; absent/malformed -> None."""
    for key, value in headers.items():
        if key.lower() == "retry-after":
            try:
                return max(0.0, float(value))
            except ValueError:
                return None
    return None


def _fx1_opts(
    *,
    backend: str | None,
    byok: dict[str, str] | None,
    checkpoint_dir: str | Path | None,
    fallbacks: list[str] | None,
    receipt_hashes: list[str] | None,
    timeout_s: float | None,
) -> dict[str, Any]:
    """Assemble the ``fx1`` extension object every completion-surface
    method merges into its payload: backend link name, BYOK endpoint
    override, checkpoint dir, ordered fallbacks, evidence seals, and the
    backend deadline — only the knobs the caller actually set."""
    fx1: dict[str, Any] = {}
    if backend is not None:
        fx1["backend"] = backend
    if byok is not None:
        fx1["byok"] = byok
    if checkpoint_dir is not None:
        fx1["checkpoint_dir"] = str(checkpoint_dir)
    if fallbacks:
        fx1["fallbacks"] = list(fallbacks)
    if receipt_hashes:
        fx1["receipt_hashes"] = list(receipt_hashes)
    if timeout_s is not None:
        fx1["timeout_s"] = timeout_s
    return fx1


def _openai_sse_chunks(body: bytes) -> list[dict[str, Any]]:
    """Parse the OpenAI SSE grammar — ``data: <json>`` frames up to the
    terminal ``[DONE]``. Fails closed when the stream ends unterminated."""
    chunks: list[dict[str, Any]] = []
    saw_done = False
    for line in body.decode().splitlines():
        if not line.startswith("data: "):
            continue
        frame = line[len("data: ") :].strip()
        if frame == "[DONE]":
            saw_done = True
            break
        chunks.append(json.loads(frame))
    if not saw_done:
        raise HarnessTransportError("stream ended without [DONE]")
    return chunks


def _hget(headers: Mapping[str, str], name: str) -> str | None:
    """Case-insensitive header lookup — transports differ in casing
    (urllib preserves the wire casing; test clients lowercase)."""
    low = name.lower()
    for k, v in headers.items():
        if k.lower() == low:
            return v
    return None


def _responses_sse_events(body: bytes) -> list[dict[str, Any]]:
    """Collect Responses SSE frames until the terminal event.

    ``response.incomplete`` is the terminal event on a truncated turn
    (e.g. ``max_tool_calls``) — it ends the stream like ``completed``.
    """
    events: list[dict[str, Any]] = []
    saw_terminal = False
    for line in body.decode().splitlines():
        if not line.startswith("data: "):
            continue
        frame = json.loads(line[len("data: ") :])
        events.append(frame)
        if frame.get("type") in ("response.completed", "response.incomplete"):
            saw_terminal = True
            break
    if not saw_terminal:
        raise HarnessTransportError("stream ended without response.completed/response.incomplete")
    return events


_REPLAY_TERMINAL_EVENTS = frozenset(
    {
        "response.completed",
        "response.incomplete",
        "response.failed",
        "response.cancelled",
    }
)


def _responses_replay_events(body: bytes) -> list[dict[str, Any]]:
    """Collect Responses *replay* frames until the terminal event.

    Wider than :func:`_responses_sse_events`: a replayed record can end
    in any terminal status — ``response.failed`` and ``response.cancelled``
    are terminals here (a create-time stream never emits them).
    """
    events: list[dict[str, Any]] = []
    saw_terminal = False
    for line in body.decode().splitlines():
        if not line.startswith("data: "):
            continue
        frame = json.loads(line[len("data: ") :])
        events.append(frame)
        if frame.get("type") in _REPLAY_TERMINAL_EVENTS:
            saw_terminal = True
            break
    if not saw_terminal:
        raise HarnessTransportError(
            "replay stream ended without a terminal response.* event "
            "(response.completed/incomplete/failed/cancelled)"
        )
    return events


class HarnessClient:
    """Remote harness client — the SDK contract over HTTP."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        transport: Transport | None = None,
        max_retries: int = 0,
        retry_backoff_s: float = 0.1,
        retry_writes: bool = False,
        max_retry_wait_s: float = 5.0,
        sleep: Callable[[float], None] = time.sleep,
        circuit_breaker_threshold: int = 0,
        circuit_reset_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"base_url must be http(s)://host[:port], got {base_url!r}")
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        if retry_backoff_s <= 0:
            raise ValueError(f"retry_backoff_s must be > 0, got {retry_backoff_s}")
        if max_retry_wait_s <= 0:
            raise ValueError(f"max_retry_wait_s must be > 0, got {max_retry_wait_s}")
        if circuit_breaker_threshold < 0:
            raise ValueError(
                f"circuit_breaker_threshold must be >= 0, got {circuit_breaker_threshold}"
            )
        if circuit_reset_s <= 0:
            raise ValueError(f"circuit_reset_s must be > 0, got {circuit_reset_s}")
        self._base = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
        self._headers = {"X-API-Key": api_key} if api_key else {}
        self._timeout_s = timeout_s
        self._transport = transport or _urllib_transport
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_s
        self._retry_writes = retry_writes
        self._max_retry_wait_s = max_retry_wait_s
        self._sleep = sleep
        self._cb_threshold = circuit_breaker_threshold
        self._cb_reset_s = circuit_reset_s
        self._clock = clock
        self._last_api_version: str | None = None
        self._last_response_headers: dict[str, str] = {}
        self._cb_failures = 0
        self._cb_open_until = 0.0

    # ---- transport ----------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | bytes | None = None,
        *,
        idempotent: bool = False,
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[int, Mapping[str, str], bytes]:
        if self._cb_threshold and self._clock() < self._cb_open_until:
            raise HarnessTransportError(
                f"circuit open for {self._base} — fail fast until "
                f"{self._cb_open_until:.1f} (monotonic)"
            )
        retries = self._max_retries if (idempotent or self._retry_writes) else 0
        backoff = self._retry_backoff_s
        try:
            for attempt in range(retries + 1):
                try:
                    status, headers, body = self._transport(
                        method,
                        self._base + path,
                        payload,
                        {**self._headers, **(extra_headers or {})},
                        self._timeout_s,
                    )
                    self._last_response_headers = {
                        str(k).lower(): str(v) for k, v in headers.items()
                    }
                    for hk, hv in headers.items():
                        if hk.lower() == "x-fx1-api-version":
                            self._last_api_version = hv
                except HarnessTransportError:
                    if attempt >= retries:
                        raise
                    self._sleep(backoff)
                    backoff *= 2
                    continue
                if self._retryable_status(status, headers) and attempt < retries:
                    wait = _retry_after_s(headers)
                    if wait is not None and wait > self._max_retry_wait_s:
                        break  # server asked for a wait longer than the budget allows
                    self._sleep(min(wait if wait is not None else backoff, self._max_retry_wait_s))
                    backoff *= 2
                    continue
                if status < 200 or status >= 300:
                    raise self._map_error(status, body)
                self._cb_reset()
                return status, headers, body
            raise HarnessTransportError(f"harness {method} {path} exhausted {retries} retries")
        except HarnessTransportError:
            self._last_response_headers = {}
            self._cb_trip()
            raise

    def _cb_reset(self) -> None:
        self._cb_failures = 0
        self._cb_open_until = 0.0

    def _cb_trip(self) -> None:
        """Count a transport fault; past the threshold the circuit opens for
        ``circuit_reset_s`` (the first call after that is the half-open probe —
        a success closes it, a fault re-opens the window)."""
        if not self._cb_threshold:
            return
        self._cb_failures += 1
        if self._cb_failures >= self._cb_threshold:
            self._cb_open_until = self._clock() + self._cb_reset_s
            self._cb_failures = 0

    @staticmethod
    def _map_error(status: int, body: bytes) -> Exception:
        code: str | None = None
        try:
            parsed = json.loads(body)
            detail = parsed.get("detail", body.decode(errors="replace"))
            raw_code = parsed.get("code")
            if isinstance(raw_code, str):
                code = raw_code
            if isinstance(detail, list):  # pydantic validation errors
                detail = "; ".join(
                    d.get("msg", str(d)) for d in detail if isinstance(d, dict)
                ) or str(detail)
            openai_err = parsed.get("error")
            if isinstance(openai_err, dict):
                # /v1 routes shape errors as {error: {message, type, code}} —
                # the machine code and the human message live there.
                msg = openai_err.get("message")
                if isinstance(msg, str):
                    detail = msg
                raw_code = openai_err.get("code")
                if isinstance(raw_code, str):
                    code = raw_code
        except (json.JSONDecodeError, AttributeError, UnicodeDecodeError):
            detail = body.decode(errors="replace")[:500]
        if status in (401, 403):
            return HarnessAuthError(f"harness auth refused ({status}): {detail}", code=code)
        if status == 404:
            return KeyError(str(detail))
        if status == 422:
            return ValueError(str(detail))
        if status == 501:
            return NotImplementedError(str(detail))
        if status == 503:
            return BackendNotConfiguredError(str(detail), code=code)
        if status == 502:
            text = str(detail)
            if "honesty gate" in text:
                return Fx1HonestyError(
                    text.split("honesty gate refused model output: ")[-1],
                    code=code,
                )
            return RuntimeError(text)
        return HarnessTransportError(f"harness API returned {status}: {detail}", code=code)

    @staticmethod
    def _retryable_status(status: int, headers: Mapping[str, str]) -> bool:
        """429/503 retry only when they carry Retry-After — every
        retryable refusal on this wire declares one (rate windows,
        in-flight cap); a hard ``quota_exceeded`` 429 carries none and
        must not be retried — the budget never clears inside a call."""
        return status in (429, 503) and _retry_after_s(headers) is not None

    def _json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        idempotent: bool = False,
        extra_headers: dict[str, str] | None = None,
    ) -> Any:
        _, _, body = self._request(
            method,
            path,
            payload,
            idempotent=idempotent,
            extra_headers=extra_headers,
        )
        return json.loads(body)

    # ---- registry -----------------------------------------------------

    def commands(self, role: str | None = None) -> list[str]:
        """Registered command names, like ``Fx1Harness.commands``."""
        path = "/harness/commands"
        if role is not None:
            path += f"?role={urllib.parse.quote(role)}"
        out = self._json("GET", path, idempotent=True)
        return [item["name"] for item in out["items"]]

    # ---- execution ------------------------------------------------------

    def run(
        self,
        command: str,
        extra_args: list[str] | None = None,
        *,
        config: str | Path | None = None,
        idempotency_key: str | None = None,
    ) -> HarnessResult:
        """Remote counterpart of ``Fx1Harness.run``.

        Carries an ``Idempotency-Key`` (auto-minted unless the caller
        supplies one for cross-process dedup), so a transport-level
        retry returns the stored result instead of re-executing the
        command — ``max_retries``/``retry_writes`` is safe here.
        """
        key = idempotency_key or uuid.uuid4().hex
        out = self._json(
            "POST",
            "/harness/runs",
            {
                "command": command,
                "extra_args": extra_args or [],
                "config": str(config) if config is not None else None,
            },
            idempotent=True,
            extra_headers={"Idempotency-Key": key},
        )
        return HarnessResult(
            command=out["command"],
            exit_code=out["exit_code"],
            stdout=out["stdout"],
            stderr=out["stderr"],
        )

    def submit_run(
        self,
        command: str,
        extra_args: list[str] | None = None,
        *,
        config: str | Path | None = None,
        idempotency_key: str | None = None,
        callback_url: str | None = None,
        callback_secret: str | None = None,
    ) -> str:
        """Submit a run as a background job; returns the job id.

        Long-running commands shouldn't hold a request open — submit,
        keep the id, and poll with ``job_status``/``wait_run``. Same
        ``Idempotency-Key`` dedup as ``run``: a retried submit returns
        the original job id instead of spawning a second execution.
        ``callback_url`` (http(s)) gets the full job record POSTed on
        every terminal transition — succeeded, failed, or cancelled.
        ``callback_secret`` signs the delivery: the POST carries
        ``X-Fx1-Webhook-Timestamp`` + ``X-Fx1-Webhook-Signature``
        (``sha256=<hmac-sha256>`` over ``<ts>.<body>``); verify with
        ``fx1.serve.webhooks.verify_webhook``.
        """
        key = idempotency_key or uuid.uuid4().hex
        out = self._json(
            "POST",
            "/harness/jobs",
            {
                "command": command,
                "extra_args": extra_args or [],
                "config": str(config) if config is not None else None,
                "callback_url": callback_url,
                "callback_secret": callback_secret,
            },
            idempotent=True,
            extra_headers={"Idempotency-Key": key},
        )
        return str(out["job_id"])

    def submit_batch(
        self,
        jobs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """POST /harness/jobs/batch — fan-out submit, one request.

        Each item is a run-request dict (``command``, ``extra_args``,
        ``config``, ``callback_url``, ``callback_secret``); a per-item
        ``idempotency_key`` dedups retries (headers carry no per-item
        keys). Per-item failures land in ``jobs[i].error``/``code`` —
        the response carries ``submitted``/``failed`` counts."""
        out = self._json("POST", "/harness/jobs/batch", {"jobs": jobs})
        return dict(out)

    def job_status(self, job_id: str) -> dict[str, Any]:
        """Live job record: ``status`` in queued/running/succeeded/
        failed/cancelled; ``result`` (the HarnessRunResponse fields)
        appears once terminal."""
        out = self._json(
            "GET",
            f"/harness/jobs/{urllib.parse.quote(job_id)}",
            idempotent=True,
        )
        return dict(out)

    def job_receipt(self, job_id: str) -> dict[str, Any]:
        """Export the job's ledger record as a sealed
        ``fx1_job_record.v1`` document — ``GET /harness/jobs/{id}/receipt``;
        404 maps to KeyError. Feed it to :meth:`verify_receipt`."""
        out = self._json(
            "GET",
            f"/harness/jobs/{urllib.parse.quote(job_id)}/receipt",
            idempotent=True,
        )
        return dict(out)

    def list_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """GET /harness/jobs — inventory page, newest first; ``total`` is
        the filtered count so callers can page until offset >= total."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status is not None:
            params["status"] = status
        out = self._json(
            "GET",
            "/harness/jobs?" + urllib.parse.urlencode(params),
            idempotent=True,
        )
        return dict(out)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """DELETE /harness/jobs/{job_id} — cooperative cancel: a queued
        job lands 'cancelled' and its slot frees on dequeue; a running
        or terminal job maps the 409 through the error table."""
        out = self._json(
            "DELETE",
            f"/harness/jobs/{urllib.parse.quote(job_id)}",
        )
        return dict(out)

    # ---- evals ----------------------------------------------------------

    def submit_eval(
        self,
        suite: str,
        *,
        backend: str = "hosted_k3",
        seed: int = 0,
        checkpoint_dir: str | None = None,
        byok: dict[str, str] | None = None,
        timeout_s: float | None = None,
        fallbacks: list[str] | None = None,
        judge_backend: str | None = None,
        judge_byok: dict[str, str] | None = None,
        callback_url: str | None = None,
        callback_secret: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """POST /harness/evals — submit a seeded eval suite against a
        backend chain (202). Returns ``{eval_id, status, replayed}``;
        poll :meth:`eval_status` or :meth:`wait_eval` for the terminal
        record, then export it sealed via :meth:`eval_receipt`."""
        body: dict[str, Any] = {"suite": suite, "backend": backend, "seed": seed}
        if checkpoint_dir is not None:
            body["checkpoint_dir"] = checkpoint_dir
        if byok is not None:
            body["byok"] = byok
        if timeout_s is not None:
            body["timeout_s"] = timeout_s
        if fallbacks is not None:
            body["fallbacks"] = fallbacks
        if judge_backend is not None:
            body["judge_backend"] = judge_backend
        if judge_byok is not None:
            body["judge_byok"] = judge_byok
        if callback_url is not None:
            body["callback_url"] = callback_url
        if callback_secret is not None:
            body["callback_secret"] = callback_secret
        out = self._json(
            "POST",
            "/harness/evals",
            body,
            extra_headers={"Idempotency-Key": idempotency_key} if idempotency_key else None,
        )
        return dict(out)

    def eval_status(self, eval_id: str) -> dict[str, Any]:
        """GET /harness/evals/{eval_id} — the live EvalRecord:
        status/report/attempts/sampling pin."""
        out = self._json(
            "GET",
            f"/harness/evals/{urllib.parse.quote(eval_id)}",
            idempotent=True,
        )
        return dict(out)

    def list_evals(
        self,
        *,
        status: str | None = None,
        suite: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """GET /harness/evals — inventory page, newest first; ``total`` is
        the filtered count before paging."""
        params: dict[str, Any] = {"limit": limit}
        if status is not None:
            params["status"] = status
        if suite is not None:
            params["suite"] = suite
        out = self._json(
            "GET",
            "/harness/evals?" + urllib.parse.urlencode(params),
            idempotent=True,
        )
        return dict(out)

    def eval_receipt(self, eval_id: str) -> dict[str, Any]:
        """GET /harness/evals/{eval_id}/receipt — the sealed
        ``fx1_eval_record.v1`` doc (409 while the eval is non-terminal;
        feed it to :meth:`verify_receipt`)."""
        out = self._json(
            "GET",
            f"/harness/evals/{urllib.parse.quote(eval_id)}/receipt",
            idempotent=True,
        )
        return dict(out)

    def cancel_eval(self, eval_id: str) -> dict[str, Any]:
        """DELETE /harness/evals/{eval_id} — cooperative cancel of a
        queued eval; running/terminal map the 409 through the error
        table."""
        out = self._json(
            "DELETE",
            f"/harness/evals/{urllib.parse.quote(eval_id)}",
        )
        return dict(out)

    def diff_evals(self, base_id: str, candidate_id: str) -> dict[str, Any]:
        """GET /harness/evals/{base}/diff/{candidate} — the promotion-gate
        diff: task transitions, gate move, by_kind deltas, verdict.
        404 unknown id, 409 non-terminal/missing report."""
        out = self._json(
            "GET",
            f"/harness/evals/{urllib.parse.quote(base_id)}/diff/{urllib.parse.quote(candidate_id)}",
            idempotent=True,
        )
        return dict(out)

    # ---- /v1/evals — the OpenAI Evals-shaped spec/run surface ----------

    def eval_spec_create(
        self,
        name: str,
        *,
        suite: str,
        seed: int = 0,
        backend: str | None = None,
        fallbacks: list[str] | None = None,
        checkpoint_dir: str | None = None,
        judge_backend: str | None = None,
        timeout_s: float | None = None,
        testing_criteria: list[dict[str, Any]] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """POST /v1/evals — declare the named eval container. The
        item_schema pins suite knobs; credentials never live on a spec."""
        item_schema: dict[str, Any] = {"suite": suite, "seed": seed}
        if backend is not None:
            item_schema["backend"] = backend
        if fallbacks is not None:
            item_schema["fallbacks"] = fallbacks
        if checkpoint_dir is not None:
            item_schema["checkpoint_dir"] = checkpoint_dir
        if judge_backend is not None:
            item_schema["judge_backend"] = judge_backend
        if timeout_s is not None:
            item_schema["timeout_s"] = timeout_s
        body: dict[str, Any] = {
            "name": name,
            "data_source_config": {"type": "custom", "item_schema": item_schema},
        }
        if testing_criteria is not None:
            body["testing_criteria"] = testing_criteria
        if metadata is not None:
            body["metadata"] = metadata
        return dict(self._json("POST", "/v1/evals", body))

    def eval_spec_get(self, eval_id: str) -> dict[str, Any]:
        """GET /v1/evals/{eval_id}."""
        return dict(self._json("GET", f"/v1/evals/{urllib.parse.quote(eval_id)}", idempotent=True))

    def eval_specs(self, *, limit: int = 20, after: str | None = None) -> dict[str, Any]:
        """GET /v1/evals — newest-first spec page."""
        q = f"?limit={limit}" + (f"&after={urllib.parse.quote(after)}" if after else "")
        return dict(self._json("GET", f"/v1/evals{q}", idempotent=True))

    def eval_spec_update(
        self,
        eval_id: str,
        *,
        name: str | None = None,
        metadata: dict[str, str] | None = None,
        data_source_config: dict[str, Any] | None = None,
        testing_criteria: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """POST /v1/evals/{eval_id} — name/metadata edits plus the
        hot-reload fields: ``data_source_config``/``testing_criteria``
        replace the spec's declared shape only while no run binds it
        (the server answers 409 ``eval_spec_frozen`` afterwards)."""
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if metadata is not None:
            body["metadata"] = metadata
        if data_source_config is not None:
            body["data_source_config"] = data_source_config
        if testing_criteria is not None:
            body["testing_criteria"] = testing_criteria
        return dict(self._json("POST", f"/v1/evals/{urllib.parse.quote(eval_id)}", body))

    def eval_spec_delete(self, eval_id: str) -> dict[str, Any]:
        """DELETE /v1/evals/{eval_id} — journaled tombstone."""
        return dict(self._json("DELETE", f"/v1/evals/{urllib.parse.quote(eval_id)}"))

    def eval_run_create(
        self,
        eval_id: str,
        *,
        model: str,
        data_source: dict[str, Any] | None = None,
        byok: dict[str, str] | None = None,
        judge_byok: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
        callback_url: str | None = None,
        callback_secret: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """POST /v1/evals/{eval_id}/runs — ``model`` is a link name,
        ``fx1``, or a registered ``ft:`` name (resolves to its
        checkpoint). Same capacity/drain gates as ``submit_eval``."""
        body: dict[str, Any] = {"model": model}
        if data_source is not None:
            body["data_source"] = data_source
        if byok is not None:
            body["byok"] = byok
        if judge_byok is not None:
            body["judge_byok"] = judge_byok
        if metadata is not None:
            body["metadata"] = metadata
        if callback_url is not None:
            body["callback_url"] = callback_url
        if callback_secret is not None:
            body["callback_secret"] = callback_secret
        return dict(
            self._json(
                "POST",
                f"/v1/evals/{urllib.parse.quote(eval_id)}/runs",
                body,
                extra_headers={"Idempotency-Key": idempotency_key} if idempotency_key else None,
            )
        )

    def eval_runs(self, eval_id: str, *, limit: int = 20) -> dict[str, Any]:
        """GET /v1/evals/{eval_id}/runs — newest-first run page."""
        return dict(
            self._json(
                "GET",
                f"/v1/evals/{urllib.parse.quote(eval_id)}/runs?limit={limit}",
                idempotent=True,
            )
        )

    def eval_run_get(self, eval_id: str, run_id: str) -> dict[str, Any]:
        """GET /v1/evals/{eval_id}/runs/{run_id}."""
        return dict(
            self._json(
                "GET",
                f"/v1/evals/{urllib.parse.quote(eval_id)}/runs/{urllib.parse.quote(run_id)}",
                idempotent=True,
            )
        )

    def eval_run_cancel(self, eval_id: str, run_id: str) -> dict[str, Any]:
        """POST .../runs/{run_id}/cancel — queued runs cancel; running or
        terminal map the 409 through."""
        return dict(
            self._json(
                "POST",
                f"/v1/evals/{urllib.parse.quote(eval_id)}/runs/{urllib.parse.quote(run_id)}/cancel",
                {},
            )
        )

    def eval_run_delete(self, eval_id: str, run_id: str) -> dict[str, Any]:
        """DELETE .../runs/{run_id} — terminal records only (409 live)."""
        return dict(
            self._json(
                "DELETE",
                f"/v1/evals/{urllib.parse.quote(eval_id)}/runs/{urllib.parse.quote(run_id)}",
            )
        )

    def eval_run_output_items(
        self, eval_id: str, run_id: str, *, limit: int = 20, after: str | None = None
    ) -> dict[str, Any]:
        """GET .../output_items — per-task verdict rows verbatim."""
        q = f"?limit={limit}" + (f"&after={urllib.parse.quote(after)}" if after else "")
        return dict(
            self._json(
                "GET",
                f"/v1/evals/{urllib.parse.quote(eval_id)}/runs/{urllib.parse.quote(run_id)}/output_items{q}",
                idempotent=True,
            )
        )

    # ---- fine-tuning (/v1/fine_tuning/jobs) ----------------------------------

    def create_finetune_job(
        self,
        *,
        model: str,
        training_file: str,
        hyperparameters: dict[str, Any] | None = None,
        suffix: str | None = None,
        validation_file: str | None = None,
        seed: int | None = None,
        metadata: dict[str, str] | None = None,
        idempotency_key: str | None = None,
        callback_url: str | None = None,
        callback_secret: str | None = None,
    ) -> dict[str, Any]:
        """POST /v1/fine_tuning/jobs — queue a gated fine-tune over an
        uploaded chat-format JSONL (upload with ``purpose='fine-tune'``).
        Validation is synchronous: a malformed corpus 400s at submit.
        ``callback_url``/``callback_secret`` are the fx1 terminal-webhook
        extension (the finished job record POSTs to the URL, signed)."""
        payload: dict[str, Any] = {"model": model, "training_file": training_file}
        if hyperparameters is not None:
            payload["hyperparameters"] = hyperparameters
        if suffix is not None:
            payload["suffix"] = suffix
        if validation_file is not None:
            payload["validation_file"] = validation_file
        if seed is not None:
            payload["seed"] = seed
        if metadata is not None:
            payload["metadata"] = metadata
        if callback_url is not None:
            payload["callback_url"] = callback_url
        if callback_secret is not None:
            payload["callback_secret"] = callback_secret
        out = self._json(
            "POST",
            "/v1/fine_tuning/jobs",
            payload,
            extra_headers=({"Idempotency-Key": idempotency_key} if idempotency_key else None),
        )
        return dict(out)

    def finetune_jobs(self, *, limit: int = 20, after: str | None = None) -> dict[str, Any]:
        """GET /v1/fine_tuning/jobs — newest-first page + has_more."""
        q = f"limit={limit}" + (f"&after={urllib.parse.quote(after)}" if after else "")
        return dict(self._json("GET", f"/v1/fine_tuning/jobs?{q}", idempotent=True))

    def finetune_job(self, job_id: str) -> dict[str, Any]:
        """GET /v1/fine_tuning/jobs/{id} — the job record."""
        out = self._json(
            "GET",
            f"/v1/fine_tuning/jobs/{urllib.parse.quote(job_id)}",
            idempotent=True,
        )
        return dict(out)

    def finetune_job_events(
        self, job_id: str, *, limit: int = 20, after: str | None = None
    ) -> dict[str, Any]:
        """GET /v1/fine_tuning/jobs/{id}/events — oldest-first feed."""
        q = f"limit={limit}" + (f"&after={urllib.parse.quote(after)}" if after else "")
        jid = urllib.parse.quote(job_id)
        out = self._json("GET", f"/v1/fine_tuning/jobs/{jid}/events?{q}", idempotent=True)
        return dict(out)

    def finetune_job_checkpoints(
        self, job_id: str, *, limit: int = 10, after: str | None = None
    ) -> dict[str, Any]:
        """GET /v1/fine_tuning/jobs/{id}/checkpoints — the model
        artifacts the job registered, oldest-first (OpenAI's
        ``fine_tuning.jobs.list_checkpoints``)."""
        q = f"limit={limit}" + (f"&after={urllib.parse.quote(after)}" if after else "")
        jid = urllib.parse.quote(job_id)
        out = self._json("GET", f"/v1/fine_tuning/jobs/{jid}/checkpoints?{q}", idempotent=True)
        return dict(out)

    def cancel_finetune_job(self, job_id: str) -> dict[str, Any]:
        """POST /v1/fine_tuning/jobs/{id}/cancel — cooperative: queued
        cancels at once, running stops at the next stage boundary."""
        out = self._json(
            "POST",
            f"/v1/fine_tuning/jobs/{urllib.parse.quote(job_id)}/cancel",
        )
        return dict(out)

    def pause_finetune_job(self, job_id: str) -> dict[str, Any]:
        """POST /v1/fine_tuning/jobs/{id}/pause — cooperative: queued
        parks before starting, running parks at the next stage boundary.
        Pausing a paused job is idempotent."""
        out = self._json(
            "POST",
            f"/v1/fine_tuning/jobs/{urllib.parse.quote(job_id)}/pause",
        )
        return dict(out)

    def resume_finetune_job(self, job_id: str) -> dict[str, Any]:
        """POST /v1/fine_tuning/jobs/{id}/resume — restores the status
        pause captured; resuming a non-paused job is a 409."""
        out = self._json(
            "POST",
            f"/v1/fine_tuning/jobs/{urllib.parse.quote(job_id)}/resume",
        )
        return dict(out)

    def wait_finetune_job(
        self,
        job_id: str,
        *,
        poll_s: float = 0.5,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Poll ``finetune_job`` until terminal; returns the record —
        ``result_files`` carries the registered artifacts. Raises
        ``HarnessJobError`` on 'failed'/'cancelled' and
        ``HarnessTransportError`` on ``timeout_s``."""
        deadline = None if timeout_s is None else self._clock() + timeout_s
        while True:
            st = self.finetune_job(job_id)
            if st["status"] == "succeeded":
                return st
            if st["status"] == "failed":
                raise HarnessJobError(
                    f"fine-tuning job {job_id} failed: {(st.get('error') or {}).get('message')}"
                )
            if st["status"] == "cancelled":
                raise HarnessJobError(f"fine-tuning job {job_id} cancelled")
            remaining = None if deadline is None else deadline - self._clock()
            if remaining is not None and remaining <= 0:
                raise HarnessTransportError(
                    f"fine-tuning job {job_id} did not finish within {timeout_s}s"
                )
            self._sleep(poll_s if remaining is None else min(poll_s, remaining))

    def wait_eval(
        self,
        eval_id: str,
        *,
        poll_s: float = 0.5,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Poll ``eval_status`` until terminal; returns the record —
        ``report`` carries the suite output. Raises ``HarnessJobError``
        on 'failed'/'cancelled' and ``HarnessTransportError`` on
        ``timeout_s`` (the eval keeps running server-side)."""
        deadline = None if timeout_s is None else self._clock() + timeout_s
        while True:
            st = self.eval_status(eval_id)
            if st["status"] == "succeeded":
                return st
            if st["status"] == "failed":
                raise HarnessJobError(f"eval {eval_id} failed: {st.get('error')}")
            if st["status"] == "cancelled":
                raise HarnessJobError(f"eval {eval_id} cancelled")
            remaining = None if deadline is None else deadline - self._clock()
            if remaining is not None and remaining <= 0:
                raise HarnessTransportError(f"eval {eval_id} did not finish within {timeout_s}s")
            self._sleep(poll_s if remaining is None else min(poll_s, remaining))

    def wait_run(
        self,
        job_id: str,
        *,
        poll_s: float = 0.5,
        timeout_s: float | None = None,
    ) -> HarnessResult:
        """Poll ``job_status`` until the job finishes; returns its result.

        Raises ``HarnessJobError`` on status 'failed' and
        ``HarnessTransportError`` when ``timeout_s`` elapses — a timed-out
        waiter leaves the job running server-side; keep polling or leave
        it for the record.
        """
        deadline = None if timeout_s is None else self._clock() + timeout_s
        while True:
            st = self.job_status(job_id)
            if st["status"] == "succeeded":
                r = st["result"]
                return HarnessResult(
                    command=r["command"],
                    exit_code=r["exit_code"],
                    stdout=r["stdout"],
                    stderr=r["stderr"],
                )
            if st["status"] == "failed":
                raise HarnessJobError(f"job {job_id} failed: {st.get('error')}")
            if st["status"] == "cancelled":
                raise HarnessJobError(f"job {job_id} cancelled")
            remaining = None if deadline is None else deadline - self._clock()
            if remaining is not None and remaining <= 0:
                raise HarnessTransportError(f"job {job_id} did not finish within {timeout_s}s")
            self._sleep(poll_s if remaining is None else min(poll_s, remaining))

    def stream_job(
        self,
        job_id: str,
        *,
        timeout_s: float = 600.0,
    ) -> list[dict[str, Any]]:
        """Follow a job over SSE — ``GET /harness/jobs/{job_id}/events``
        returns the job record as an ``event: job`` frame on every status
        change until the job goes terminal, then closes. Returns every
        distinct record snapshot in order; the last frame is terminal
        unless the server-side ``timeout_s`` elapses first (reconnect to
        resume — frames carry the full record, not diffs)."""
        _, _, body = self._request(
            "GET",
            f"/harness/jobs/{urllib.parse.quote(job_id)}/events"
            f"?timeout_s={urllib.parse.quote(str(timeout_s))}",
            idempotent=True,
        )
        frames: list[dict[str, Any]] = []
        for line in body.decode().splitlines():
            if not line.startswith("data: "):
                continue
            frames.append(json.loads(line[len("data: ") :]))
        if not frames:
            raise HarnessTransportError("job stream ended without a status frame")
        return frames

    def wait_run_stream(
        self,
        job_id: str,
        *,
        timeout_s: float = 600.0,
    ) -> HarnessResult:
        """``wait_run`` without polling: follows the SSE job stream, then
        maps the terminal record through the same outcome table. A stream
        that ends before a terminal status raises HarnessTransportError
        (the job may still be running server-side — reconnect or poll)."""
        st = self.stream_job(job_id, timeout_s=timeout_s)[-1]
        if st["status"] == "succeeded":
            r = st["result"]
            return HarnessResult(
                command=r["command"],
                exit_code=r["exit_code"],
                stdout=r["stdout"],
                stderr=r["stderr"],
            )
        if st["status"] == "failed":
            raise HarnessJobError(f"job {job_id} failed: {st.get('error')}")
        if st["status"] == "cancelled":
            raise HarnessJobError(f"job {job_id} cancelled")
        raise HarnessTransportError(
            f"job {job_id} stream ended at {st['status']!r} before terminal"
        )

    # ---- gated completion ------------------------------------------------

    def complete(  # NOSONAR(S107)
        self,
        messages: list[dict[str, str]],
        *,
        backend: str = "local_fx1",
        checkpoint_dir: str | Path | None = None,
        receipt_hashes: list[str] | None = None,
        idempotency_key: str | None = None,
        timeout_s: float | None = None,
        byok: dict[str, str] | None = None,
        fallbacks: list[str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        service_tier: str | None = None,
        verbosity: str | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
    ) -> CompletionResult:
        """Remote counterpart of ``Fx1Harness.complete``."""
        out = self._json(
            "POST",
            "/harness/complete",
            {
                "backend": backend,
                "messages": messages,
                "checkpoint_dir": (str(checkpoint_dir) if checkpoint_dir is not None else None),
                "receipt_hashes": receipt_hashes,
                "timeout_s": timeout_s,
                "byok": byok,
                "fallbacks": fallbacks or [],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "seed": seed,
                "reasoning_effort": reasoning_effort,
                "service_tier": service_tier,
                "verbosity": verbosity,
                "prompt_cache_key": prompt_cache_key,
                "prompt_cache_retention": prompt_cache_retention,
            },
            # A keyed complete dedupes server-side — safe to retry by
            # construction, so it marks idempotent for the retry policy.
            idempotent=idempotency_key is not None,
            extra_headers={"Idempotency-Key": idempotency_key} if idempotency_key else None,
        )
        return CompletionResult(
            backend=out["backend"],
            model=out["model"],
            content=out["content"],
            receipt_hashes=tuple(out["receipt_hashes"]),
            replayed=out.get("replayed", False),
            usage=out.get("usage") if isinstance(out.get("usage"), dict) else None,
            completion_id=out.get("completion_id"),
            attempts=tuple(dict(a) for a in out["attempts"] if isinstance(a, dict))
            if isinstance(out.get("attempts"), list)
            else (),
            sampling=out.get("sampling") if isinstance(out.get("sampling"), dict) else None,
        )

    def completion(self, completion_id: str) -> CompletionRecord:
        """Fetch one recorded call from the server's completion log —
        ``GET /harness/completions/{id}``; 404 maps to KeyError."""
        out = self._json("GET", f"/harness/completions/{completion_id}", idempotent=True)
        return CompletionRecord(
            completion_id=out["completion_id"],
            backend=out["backend"],
            model=out.get("model"),
            ok=out["ok"],
            latency_ms=out["latency_ms"],
            at=out["at"],
            usage=out.get("usage") if isinstance(out.get("usage"), dict) else None,
            error=out.get("error"),
            error_class=out.get("error_class"),
            prompt_sha256=out["prompt_sha256"],
            output_sha256=out.get("output_sha256"),
            attempts=tuple(dict(a) for a in out["attempts"] if isinstance(a, dict))
            if isinstance(out.get("attempts"), list)
            else None,
            sampling=out.get("sampling") if isinstance(out.get("sampling"), dict) else None,
        )

    def completions(self, *, limit: int = 50, backend: str | None = None) -> list[CompletionRecord]:
        """Newest-first window on the server's completion log —
        ``GET /harness/completions``."""
        params: dict[str, Any] = {"limit": limit}
        if backend is not None:
            params["backend"] = backend
        out = self._json(
            "GET",
            f"/harness/completions?{urllib.parse.urlencode(params)}",
            idempotent=True,
        )
        return [
            CompletionRecord(
                completion_id=r["completion_id"],
                backend=r["backend"],
                model=r.get("model"),
                ok=r["ok"],
                latency_ms=r["latency_ms"],
                at=r["at"],
                usage=r.get("usage") if isinstance(r.get("usage"), dict) else None,
                error=r.get("error"),
                error_class=r.get("error_class"),
                prompt_sha256=r["prompt_sha256"],
                output_sha256=r.get("output_sha256"),
                attempts=tuple(dict(a) for a in r["attempts"] if isinstance(a, dict))
                if isinstance(r.get("attempts"), list)
                else None,
                sampling=r.get("sampling") if isinstance(r.get("sampling"), dict) else None,
            )
            for r in out["items"]
        ]

    def completion_receipt(self, completion_id: str) -> dict[str, Any]:
        """Export one logged call's sealed ``fx1_completion_record.v1``
        document — ``GET /harness/completions/{id}/receipt``; 404 maps to
        KeyError. Feed it to :meth:`verify_receipt` to check the seal."""
        out = self._json(
            "GET",
            f"/harness/completions/{completion_id}/receipt",
            idempotent=True,
        )
        return dict(out)

    def usage(
        self,
        *,
        backend: str | None = None,
        model: str | None = None,
        since: float | None = None,
        until: float | None = None,
        key_id: str | None = None,
    ) -> UsageReport:
        """Token/request accounting over the server's completion log —
        ``GET /harness/usage``. `since`/`until` are unix-second bounds;
        since>until is a fail-closed 400 on the wire. ``key_id`` filters
        to one credential fingerprint; the report's ``by_key`` splits
        the window per key."""
        params: dict[str, Any] = {}
        if backend is not None:
            params["backend"] = backend
        if model is not None:
            params["model"] = model
        if key_id is not None:
            params["key_id"] = key_id
        if since is not None:
            params["since"] = since
        if until is not None:
            params["until"] = until
        query = f"?{urllib.parse.urlencode(params)}" if params else ""
        out = self._json("GET", f"/harness/usage{query}", idempotent=True)
        return UsageReport.model_validate(out)

    def key_create(
        self,
        name: str | None = None,
        admin: bool = False,
        *,
        rpm: int | None = None,
        ttl_s: float | None = None,
        scopes: list[str] | tuple[str, ...] | None = None,
        max_requests: int | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """``POST /harness/keys`` — mint a managed API key. The raw
        ``key`` appears once in the response; it is never stored
        server-side. ``admin=True`` keys may manage keys on the wire;
        ``rpm`` bounds the key's request rate (over-limit answers 429),
        ``ttl_s`` bakes an expiry. ``scopes`` bounds the key to
        ``read``/``write``/``admin`` surface classes (out-of-scope calls
        answer 403 ``insufficient_scope``). ``max_requests``/``max_tokens``
        declare hard budgets — an exhausted key answers 429
        ``quota_exceeded``. Requires the bootstrap credential on the wire."""
        body: dict[str, Any] = {"admin": admin}
        if name is not None:
            body["name"] = name
        if rpm is not None:
            body["rpm"] = rpm
        if ttl_s is not None:
            body["ttl_s"] = ttl_s
        if scopes is not None:
            body["scopes"] = list(scopes)
        if max_requests is not None:
            body["max_requests"] = max_requests
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        return dict(self._json("POST", "/harness/keys", body))

    def keys(self) -> list[dict[str, Any]]:
        """``GET /harness/keys`` — every minted key's fingerprint +
        metadata (never secrets)."""
        out = self._json("GET", "/harness/keys", idempotent=True)
        return list(out.get("data", []))

    def key_get(self, key_id: str) -> dict[str, Any]:
        """``GET /harness/keys/{id}`` — one key's record."""
        return dict(self._json("GET", f"/harness/keys/{key_id}", idempotent=True))

    def key_revoke(self, key_id: str) -> dict[str, Any]:
        """``DELETE /harness/keys/{id}`` — tombstone the key; auth with
        it fails closed immediately after."""
        return dict(self._json("DELETE", f"/harness/keys/{key_id}"))

    def key_usage(self, key_id: str) -> dict[str, Any]:
        """``GET /harness/keys/{id}/usage`` — the key's usage card:
        live counters, declared budgets with headroom, rate-window
        state, and the completion-ring spend split. Needs admin."""
        return dict(self._json("GET", f"/harness/keys/{key_id}/usage", idempotent=True))

    def key_rotate(
        self,
        key_id: str,
        *,
        revoke_old: bool = True,
        name: str | None = None,
        ttl_s: float | None = None,
    ) -> dict[str, Any]:
        """``POST /harness/keys/{id}/rotate`` — atomic rotation: the
        successor mints under the predecessor's declared policy, the old
        secret tombstones in the same transaction unless ``revoke_old``
        is false, and the response's ``key.key`` is the only place the
        new secret appears. Without ``ttl_s`` the successor inherits the
        predecessor's absolute expiry — rotation never extends a
        credential's lifetime."""
        body: dict[str, Any] = {"revoke_old": revoke_old}
        if name is not None:
            body["name"] = name
        if ttl_s is not None:
            body["ttl_s"] = ttl_s
        return dict(self._json("POST", f"/harness/keys/{key_id}/rotate", body))

    def key_update(
        self,
        key_id: str,
        *,
        name: str | None = _UNSET,
        rpm: int | None = _UNSET,
        scopes: list[str] | tuple[str, ...] | None = _UNSET,
        admin: bool | None = _UNSET,
        max_requests: int | None = _UNSET,
        max_tokens: int | None = _UNSET,
        expires_at: float | None = _UNSET,
        clear: Iterable[str] = (),
    ) -> dict[str, Any]:
        """``PATCH /harness/keys/{id}`` — mutable policy update on a
        live managed key. An omitted kwarg keeps the declared policy;
        an explicit ``None`` sends the JSON-null that clears the bound
        (``name``/``rpm``/``max_requests``/``max_tokens``/
        ``expires_at`` — ``clear=`` names the same fields);
        ``scopes``/``admin`` take concrete values when given —
        ``admin:true`` unions the admin scope while ``admin:false``
        never strips one. Returns the updated record; ``KeyError`` on
        404, ``ValueError`` on 422, a refused revoked key maps the
        wire's 409."""
        body: dict[str, Any] = {}
        for field, value in (
            ("name", name),
            ("rpm", rpm),
            ("scopes", scopes),
            ("admin", admin),
            ("max_requests", max_requests),
            ("max_tokens", max_tokens),
            ("expires_at", expires_at),
        ):
            if value is _UNSET:
                continue
            body[field] = (
                list(value) if field == "scopes" and isinstance(value, (list, tuple)) else value
            )
        for field in clear:
            body[field] = None
        return dict(self._json("PATCH", f"/harness/keys/{key_id}", body))

    def self_usage(self) -> dict[str, Any]:
        """``GET /harness/self`` — the calling credential's own card:
        which class it is (``managed`` / ``env`` / ``none``) plus, for
        managed keys, live budget headroom. Only needs ``read`` scope."""
        return dict(self._json("GET", "/harness/self", idempotent=True))

    def check_text(self, text: str) -> GateCheckResult:
        """Pre-flight text through the remote honesty gate — POSTs
        ``/harness/gate/check``; a refusal rides ``ok=False``, it never
        raises ``Fx1HonestyError``."""
        out = self._json("POST", "/harness/gate/check", {"text": text})
        return GateCheckResult(ok=out["ok"], error=out.get("error"))

    def score(self, input: str | list[str]) -> list[dict[str, Any]]:  # noqa: A002
        """Score text through the remote reward contract — POSTs
        ``/harness/score``; returns the per-item breakdowns
        (``total``/``components``/``violations``)."""
        out = self._json("POST", "/harness/score", {"input": input})
        return list(out["data"])

    def moderate(self, input: str | list[str]) -> dict[str, Any]:  # noqa: A002
        """Classify text through the remote honesty gate — POSTs
        ``/v1/moderations``; returns the full wire payload (``id``,
        ``model``, per-input ``results``)."""
        out = self._json("POST", "/v1/moderations", {"input": input})
        return dict(out)

    def probe_backend(
        self,
        backend: str,
        *,
        checkpoint_dir: str | Path | None = None,
        byok: dict[str, str] | None = None,
        timeout_s: float | None = None,
        prompt: str = "ping",
    ) -> ProbeResult:
        """Remote deep-health probe — POSTs ``/harness/backends/{name}/probe``
        so a BYOK probe tests the caller's own endpoint through the same
        credential plumbing as a completion."""
        out = self._json(
            "POST",
            f"/harness/backends/{backend}/probe",
            {
                "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
                "byok": byok,
                "timeout_s": timeout_s,
                "prompt": prompt,
            },
        )
        return ProbeResult(
            backend=out["backend"],
            ok=out["ok"],
            model=out.get("model"),
            latency_ms=out["latency_ms"],
            error=out.get("error"),
            error_class=out.get("error_class"),
        )

    def complete_many(  # NOSONAR(S107)
        self,
        batch: list[list[dict[str, str]]],
        *,
        backend: str = "local_fx1",
        checkpoint_dir: str | Path | None = None,
        receipt_hashes: list[str] | None = None,
        max_workers: int = 4,
        idempotency_key: str | None = None,
        timeout_s: float | None = None,
        byok: dict[str, str] | None = None,
        fallbacks: list[str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        service_tier: str | None = None,
        verbosity: str | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
    ) -> list[CompletionResult]:
        """Remote counterpart of ``Fx1Harness.complete_many``.

        The server parallelizes over one shared backend; the client mirrors
        the SDK's failure contract: the lowest-index failed item raises —
        ``Fx1HonestyError`` for ``honesty_refusal``, ``RuntimeError``
        carrying the error class otherwise.
        """
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        if not batch:
            return []
        out = self._json(
            "POST",
            "/harness/complete/batch",
            {
                "backend": backend,
                "batch": batch,
                "checkpoint_dir": (str(checkpoint_dir) if checkpoint_dir is not None else None),
                "receipt_hashes": receipt_hashes,
                "timeout_s": timeout_s,
                "byok": byok,
                "max_workers": max_workers,
                "fallbacks": fallbacks or [],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "seed": seed,
                "reasoning_effort": reasoning_effort,
                "service_tier": service_tier,
                "verbosity": verbosity,
                "prompt_cache_key": prompt_cache_key,
                "prompt_cache_retention": prompt_cache_retention,
            },
            idempotent=idempotency_key is not None,
            extra_headers={"Idempotency-Key": idempotency_key} if idempotency_key else None,
        )
        results: list[CompletionResult] = []
        for item in out["results"]:
            if not item["ok"]:
                msg = item["error"] or "unknown backend failure"
                if item.get("error_class") == "honesty_refusal":
                    raise Fx1HonestyError(msg)
                raise RuntimeError(f"{item.get('error_class')}: {msg}")
            results.append(
                CompletionResult(
                    backend=out["backend"],
                    model=out["model"],
                    content=item["content"],
                    receipt_hashes=tuple(out["receipt_hashes"]),
                    completion_id=item.get("completion_id"),
                    sampling=out.get("sampling") if isinstance(out.get("sampling"), dict) else None,
                )
            )
        return results

    def stream_complete(  # NOSONAR(S107)
        self,
        messages: list[dict[str, str]],
        *,
        backend: str = "local_fx1",
        checkpoint_dir: str | Path | None = None,
        receipt_hashes: list[str] | None = None,
        timeout_s: float | None = None,
        byok: dict[str, str] | None = None,
        fallbacks: list[str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        service_tier: str | None = None,
        verbosity: str | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
    ) -> list[str]:
        """Remote counterpart of ``Fx1Harness.stream_complete``.

        Parses the SSE token frames; the server already gated the joined
        text, so a refusal surfaces as a mapped exception, never a
        truncated stream.
        """
        _, _, body = self._request(
            "POST",
            "/harness/complete/stream",
            {
                "backend": backend,
                "messages": messages,
                "checkpoint_dir": (str(checkpoint_dir) if checkpoint_dir is not None else None),
                "receipt_hashes": receipt_hashes,
                "timeout_s": timeout_s,
                "byok": byok,
                "fallbacks": fallbacks or [],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "seed": seed,
                "reasoning_effort": reasoning_effort,
                "service_tier": service_tier,
                "verbosity": verbosity,
                "prompt_cache_key": prompt_cache_key,
                "prompt_cache_retention": prompt_cache_retention,
            },
        )
        chunks: list[str] = []
        saw_done = False
        for line in body.decode().splitlines():
            if not line.startswith("data: "):
                continue
            frame = line[len("data: ") :].strip()
            if frame == "[DONE]":
                saw_done = True
                break
            payload = json.loads(frame)
            if payload.get("type") == "token":
                chunks.append(payload["content"])
            elif payload.get("type") == "error":
                # Terminal in-band error (keepalive mode committed the 200
                # before the gate/backend resolved) — map it through the
                # same table as HTTP error responses; the frame's machine
                # ``code`` survives so callers see e.g. "backend_unavailable".
                raise self._map_error(
                    int(payload.get("status", 502)),
                    json.dumps(
                        {
                            "detail": payload.get("detail", ""),
                            "code": payload.get("code"),
                        }
                    ).encode(),
                )
        if not saw_done:
            raise HarnessTransportError("stream ended without [DONE]")
        return chunks

    # ---- OpenAI-compatible ingress (/v1) ------------------------------------

    def list_models(self) -> dict[str, Any]:
        """``GET /v1/models`` — the OpenAI ``list`` envelope: ``fx1``
        (the default link) plus the backend names a request ``model`` may
        carry (``hosted_k3``/``local_fx1``/``byok``)."""
        return dict(self._json("GET", "/v1/models", idempotent=True))

    def retrieve_model(self, model: str) -> dict[str, Any]:
        """``GET /v1/models/{model}`` — OpenAI's ``models.retrieve``:
        one card for a listed id; unknown ids raise the 404-class
        error (``model_not_found``), never a fabricated card."""
        return dict(
            self._json("GET", f"/v1/models/{urllib.parse.quote(model, safe='')}", idempotent=True)
        )

    def delete_model(self, model: str) -> dict[str, Any]:
        """``DELETE /v1/models/{model}`` — unregister a fine-tuned
        ``ft:`` model (OpenAI's ``models.delete``). Built-in link ids
        refuse 400; unregistered names fail closed 404 — the verdict is
        real, never fabricated."""
        return dict(self._json("DELETE", f"/v1/models/{urllib.parse.quote(model, safe='')}"))

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str = "fx1",
        backend: str | None = None,
        byok: dict[str, str] | None = None,
        checkpoint_dir: str | Path | None = None,
        fallbacks: list[str] | None = None,
        receipt_hashes: list[str] | None = None,
        timeout_s: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        n: int = 1,
        stop: str | list[str] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, int] | None = None,
        user: str | None = None,
        metadata: dict[str, str] | None = None,
        service_tier: str | None = None,
        reasoning_effort: str | None = None,
        verbosity: str | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        max_completion_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        idempotency_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        """``POST /v1/chat/completions`` — the OpenAI surface over the
        gated pipeline. ``model`` selects a backend when it names one;
        ``backend`` maps to ``X-Fx1-Backend`` and wins over ``model``, and
        the remaining knobs ride the ``fx1`` extension object (BYOK callers
        may instead pass ``X-Fx1-Byok-*`` via ``extra_headers``).

        ``idempotency_key`` rides the ``Idempotency-Key`` header — a
        retried call (same key + same body) replays the stored response
        byte-identically instead of re-spending the model, and marks the
        call retryable for the transport policy.

        ``n>1`` runs n gated calls server-side — each ``choices[i]`` got
        its own honesty-gate pass; ``stop`` cuts at the earliest match
        (harness-enforced, so stub/local backends honor it too);
        ``user``/``metadata`` stamp the audit record.
        ``tools``/``tool_choice``/``parallel_tool_calls`` carry the
        function-calling surface verbatim — agent ``tool_calls`` history
        and ``role: 'tool'`` results ride ``messages`` itself; a link
        without the tool channel answers 501, never a dropped tool spec.

        Returns ``(chat_completion_envelope, completion_id)`` — the id
        links the call to ``completion()``/``completion_receipt()``. Call
        :meth:`chat_completion_stream` for SSE deltas."""
        fx1 = _fx1_opts(
            backend=backend,
            byok=byok,
            checkpoint_dir=checkpoint_dir,
            fallbacks=fallbacks,
            receipt_hashes=receipt_hashes,
            timeout_s=timeout_s,
        )
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "max_completion_tokens": max_completion_tokens,
            "seed": seed,
            "n": n,
            "stop": stop,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "user": user,
            "metadata": metadata,
            "service_tier": service_tier,
            "reasoning_effort": reasoning_effort,
            "verbosity": verbosity,
            "prompt_cache_key": prompt_cache_key,
            "prompt_cache_retention": prompt_cache_retention,
            "response_format": response_format,
            "tools": tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "stream": False,
        }
        if fx1:
            payload["fx1"] = fx1
        if idempotency_key is not None:
            extra_headers = {**(extra_headers or {}), "Idempotency-Key": idempotency_key}
        _status, headers, body = self._request(
            "POST",
            "/v1/chat/completions",
            payload,
            idempotent=idempotency_key is not None,
            extra_headers=extra_headers,
        )
        envelope = json.loads(body)
        return envelope, _hget(headers, "X-Fx1-Completion-Id")

    def chat_completion_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str = "fx1",
        backend: str | None = None,
        byok: dict[str, str] | None = None,
        checkpoint_dir: str | Path | None = None,
        fallbacks: list[str] | None = None,
        receipt_hashes: list[str] | None = None,
        timeout_s: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        n: int = 1,
        stop: str | list[str] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, int] | None = None,
        user: str | None = None,
        metadata: dict[str, str] | None = None,
        service_tier: str | None = None,
        reasoning_effort: str | None = None,
        verbosity: str | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        max_completion_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        include_usage: bool = False,
        idempotency_key: str | None = None,
        last_event_id: int | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Streaming counterpart of :meth:`chat_completion` — returns
        ``(chunks, completion_id)`` where chunks are the parsed
        ``chat.completion.chunk`` frames (the terminal ``include_usage``
        chunk carries ``choices: []`` + ``usage``). The text is already
        past the honesty gate before the first delta ships.

        ``last_event_id`` resumes a dropped keyed stream: the wire's SSE
        frames carry ``id:`` equal to their chunk index, so a caller that
        received k chunks resends the call with the same
        ``idempotency_key`` + ``last_event_id=k - 1`` and gets the
        byte-identical suffix. Resume without a key fails closed 400; a
        key with no pinned stream 409s."""
        fx1 = _fx1_opts(
            backend=backend,
            byok=byok,
            checkpoint_dir=checkpoint_dir,
            fallbacks=fallbacks,
            receipt_hashes=receipt_hashes,
            timeout_s=timeout_s,
        )
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "max_completion_tokens": max_completion_tokens,
            "seed": seed,
            "n": n,
            "stop": stop,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "user": user,
            "metadata": metadata,
            "service_tier": service_tier,
            "reasoning_effort": reasoning_effort,
            "verbosity": verbosity,
            "prompt_cache_key": prompt_cache_key,
            "prompt_cache_retention": prompt_cache_retention,
            "response_format": response_format,
            "tools": tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "stream": True,
            "stream_options": {"include_usage": True} if include_usage else None,
        }
        if fx1:
            payload["fx1"] = fx1
        if idempotency_key is not None:
            extra_headers = {**(extra_headers or {}), "Idempotency-Key": idempotency_key}
        if last_event_id is not None:
            extra_headers = {
                **(extra_headers or {}),
                "Last-Event-ID": str(last_event_id),
            }
        _status, headers, body = self._request(
            "POST",
            "/v1/chat/completions",
            payload,
            idempotent=idempotency_key is not None,
            extra_headers=extra_headers,
        )
        chunks = _openai_sse_chunks(body)
        return chunks, _hget(headers, "X-Fx1-Completion-Id")

    def create_completion(  # NOSONAR(S107) — mirrors the /v1/completions param surface
        self,  # NOSONAR(S107)
        prompt: str | list[str],
        *,
        model: str = "fx1",
        backend: str | None = None,
        byok: dict[str, str] | None = None,
        checkpoint_dir: str | Path | None = None,
        fallbacks: list[str] | None = None,
        receipt_hashes: list[str] | None = None,
        timeout_s: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        n: int = 1,
        stop: str | list[str] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, int] | None = None,
        user: str | None = None,
        metadata: dict[str, str] | None = None,
        echo: bool = False,
        idempotency_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        """``POST /v1/completions`` — the legacy ``text_completion``
        surface (what ``client.completions.create`` and pre-chat tooling
        call). ``prompt`` accepts a string or a list — each element runs
        the gated pipeline independently and ``n`` repeats within an
        element; ``echo`` prepends the prompt to each choice's text.
        ``suffix``/``best_of``/``logprobs`` refuse server-side 422.

        Returns ``(text_completion_envelope, completion_id)`` — the id
        links the call to ``completion()``/``completion_receipt()``;
        ``idempotency_key`` rides ``Idempotency-Key``."""
        fx1 = _fx1_opts(
            backend=backend,
            byok=byok,
            checkpoint_dir=checkpoint_dir,
            fallbacks=fallbacks,
            receipt_hashes=receipt_hashes,
            timeout_s=timeout_s,
        )
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
            "n": n,
            "stop": stop,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "user": user,
            "metadata": metadata,
            "echo": echo,
            "stream": False,
        }
        if fx1:
            payload["fx1"] = fx1
        if idempotency_key is not None:
            extra_headers = {**(extra_headers or {}), "Idempotency-Key": idempotency_key}
        _status, headers, body = self._request(
            "POST",
            "/v1/completions",
            payload,
            idempotent=idempotency_key is not None,
            extra_headers=extra_headers,
        )
        envelope = json.loads(body)
        return envelope, _hget(headers, "X-Fx1-Completion-Id")

    def create_completion_stream(  # NOSONAR(S107)
        self,  # NOSONAR(S107)
        prompt: str | list[str],
        *,
        model: str = "fx1",
        backend: str | None = None,
        byok: dict[str, str] | None = None,
        checkpoint_dir: str | Path | None = None,
        fallbacks: list[str] | None = None,
        receipt_hashes: list[str] | None = None,
        timeout_s: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        n: int = 1,
        stop: str | list[str] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, int] | None = None,
        user: str | None = None,
        metadata: dict[str, str] | None = None,
        echo: bool = False,
        include_usage: bool = False,
        idempotency_key: str | None = None,
        last_event_id: int | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Streaming counterpart of :meth:`create_completion` — returns
        ``(chunks, completion_id)`` where chunks are the parsed
        ``text_completion`` frames (per-choice ``text`` deltas + the
        ``finish_reason`` terminal frame; ``include_usage`` adds the
        ``choices: []`` usage chunk). ``last_event_id`` resumes a dropped
        keyed stream exactly like the chat surface."""
        fx1 = _fx1_opts(
            backend=backend,
            byok=byok,
            checkpoint_dir=checkpoint_dir,
            fallbacks=fallbacks,
            receipt_hashes=receipt_hashes,
            timeout_s=timeout_s,
        )
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
            "n": n,
            "stop": stop,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "user": user,
            "metadata": metadata,
            "echo": echo,
            "stream": True,
            "stream_options": {"include_usage": True} if include_usage else None,
        }
        if fx1:
            payload["fx1"] = fx1
        if idempotency_key is not None:
            extra_headers = {**(extra_headers or {}), "Idempotency-Key": idempotency_key}
        if last_event_id is not None:
            extra_headers = {
                **(extra_headers or {}),
                "Last-Event-ID": str(last_event_id),
            }
        _status, headers, body = self._request(
            "POST",
            "/v1/completions",
            payload,
            idempotent=idempotency_key is not None,
            extra_headers=extra_headers,
        )
        chunks = _openai_sse_chunks(body)
        return chunks, _hget(headers, "X-Fx1-Completion-Id")

    def create_message(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str = "fx1",
        max_tokens: int = 1024,
        system: str | list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop_sequences: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        user_id: str | None = None,
        backend: str | None = None,
        byok: dict[str, str] | None = None,
        checkpoint_dir: str | Path | None = None,
        fallbacks: list[str] | None = None,
        receipt_hashes: list[str] | None = None,
        timeout_s: float | None = None,
        idempotency_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        """``POST /v1/messages`` — the Anthropic Messages surface over the
        gated pipeline.

        ``messages`` are Anthropic turns (``{"role": "user"|"assistant",
        "content": str | [blocks]}``); ``system`` takes a string or
        text-block list; ``tools`` use Anthropic's ``{"name",
        "description", "input_schema"}`` shape and ``tool_choice``
        ``{"type": "auto"|"any"|"tool"|"none"}``. ``max_tokens`` is
        required by the contract. Anthropic-only knobs the pipeline
        cannot honor (``top_k``, ``thinking``, ``cache_control``,
        image/document blocks) fail closed — the refusal is the
        Anthropic error envelope ``{type: "error", error: {...}}``.

        Returns ``(message_object, completion_id)`` — the cid links the
        call to ``completion()``/``completion_receipt()``;
        ``idempotency_key`` rides ``Idempotency-Key`` (same-key+body
        replays byte-identically)."""
        fx1 = _fx1_opts(
            backend=backend,
            byok=byok,
            checkpoint_dir=checkpoint_dir,
            fallbacks=fallbacks,
            receipt_hashes=receipt_hashes,
            timeout_s=timeout_s,
        )
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "system": system,
            "temperature": temperature,
            "top_p": top_p,
            "stop_sequences": stop_sequences,
            "tools": tools,
            "tool_choice": tool_choice,
            "metadata": {"user_id": user_id} if user_id is not None else None,
            "stream": False,
        }
        if fx1:
            payload["fx1"] = fx1
        if idempotency_key is not None:
            extra_headers = {**(extra_headers or {}), "Idempotency-Key": idempotency_key}
        _status, headers, body = self._request(
            "POST",
            "/v1/messages",
            payload,
            idempotent=idempotency_key is not None,
            extra_headers=extra_headers,
        )
        return json.loads(body), _hget(headers, "X-Fx1-Completion-Id")

    def create_message_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str = "fx1",
        max_tokens: int = 1024,
        system: str | list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop_sequences: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        user_id: str | None = None,
        backend: str | None = None,
        byok: dict[str, str] | None = None,
        checkpoint_dir: str | Path | None = None,
        fallbacks: list[str] | None = None,
        receipt_hashes: list[str] | None = None,
        timeout_s: float | None = None,
        idempotency_key: str | None = None,
        last_event_id: int | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Streaming counterpart of :meth:`create_message` — returns
        ``(events, completion_id)`` where events are the parsed Anthropic
        SSE payloads (``message_start``/``content_block_*``/
        ``message_delta``/``message_stop`` frames). ``last_event_id``
        resumes a dropped keyed stream — frames carry ``id: <index>``
        like the OpenAI surface."""
        fx1 = _fx1_opts(
            backend=backend,
            byok=byok,
            checkpoint_dir=checkpoint_dir,
            fallbacks=fallbacks,
            receipt_hashes=receipt_hashes,
            timeout_s=timeout_s,
        )
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "system": system,
            "temperature": temperature,
            "top_p": top_p,
            "stop_sequences": stop_sequences,
            "tools": tools,
            "tool_choice": tool_choice,
            "metadata": {"user_id": user_id} if user_id is not None else None,
            "stream": True,
        }
        if fx1:
            payload["fx1"] = fx1
        if idempotency_key is not None:
            extra_headers = {**(extra_headers or {}), "Idempotency-Key": idempotency_key}
        if last_event_id is not None:
            extra_headers = {
                **(extra_headers or {}),
                "Last-Event-ID": str(last_event_id),
            }
        _status, headers, body = self._request(
            "POST",
            "/v1/messages",
            payload,
            idempotent=idempotency_key is not None,
            extra_headers=extra_headers,
        )
        events: list[dict[str, Any]] = []
        saw_stop = False
        for line in body.decode().splitlines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: ") :])
            events.append(event)
            if event.get("type") == "message_stop":
                saw_stop = True
                break
        if not saw_stop:
            raise HarnessTransportError("stream ended without message_stop")
        return events, _hget(headers, "X-Fx1-Completion-Id")

    # ---- anthropic message batches --------------------------------------------

    def create_message_batch(
        self,
        requests: list[dict[str, Any]],
        *,
        callback_url: str | None = None,
        callback_secret: str | None = None,
        idempotency_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """``POST /v1/messages/batches`` — submit an Anthropic message
        batch. ``requests`` ride inline: each ``{"custom_id", "params"}``
        carries a full ``/v1/messages`` body (``stream`` inside a batch
        refuses at validation). The batch runs under the caller's
        ``X-Fx1-*`` headers — pass ``X-Fx1-Backend``/``X-Fx1-Byok-*``/
        ``X-Fx1-Fallbacks`` via ``extra_headers`` exactly like
        :meth:`create_batch`. ``Idempotency-Key`` replays the submit
        envelope; ``callback_url``/``callback_secret`` are the fx1
        terminal-webhook extension (fire-once on ``ended``, HMAC-signed
        when the secret is set)."""
        payload: dict[str, Any] = {"requests": requests}
        if callback_url is not None:
            payload["callback_url"] = callback_url
        if callback_secret is not None:
            payload["callback_secret"] = callback_secret
        if idempotency_key is not None:
            extra_headers = {**(extra_headers or {}), "Idempotency-Key": idempotency_key}
        _status, _headers, body = self._request(
            "POST",
            "/v1/messages/batches",
            payload,
            idempotent=idempotency_key is not None,
            extra_headers=extra_headers,
        )
        return dict(json.loads(body))

    def message_batch(self, batch_id: str) -> dict[str, Any]:
        """``GET /v1/messages/batches/{id}`` — processing status +
        request counts. Read-time expiry applies: a batch past
        ``expires_at`` ends ``expired`` with unfinished items expired."""
        return dict(self._json("GET", f"/v1/messages/batches/{batch_id}", idempotent=True))

    def message_batches(
        self,
        *,
        limit: int = 20,
        after_id: str | None = None,
        before_id: str | None = None,
    ) -> dict[str, Any]:
        """``GET /v1/messages/batches`` — newest-first page. ``after_id``
        pages to entries newer than the cursor id, ``before_id`` to
        entries older than it (Anthropic's cursor convention)."""
        path = f"/v1/messages/batches?limit={limit}"
        if after_id is not None:
            path += f"&after_id={urllib.parse.quote(after_id)}"
        if before_id is not None:
            path += f"&before_id={urllib.parse.quote(before_id)}"
        return dict(self._json("GET", path, idempotent=True))

    def cancel_message_batch(self, batch_id: str) -> dict[str, Any]:
        """``POST /v1/messages/batches/{id}/cancel`` — cooperative cancel;
        the batch flips to ``canceling`` and in-flight items complete
        before it ends."""
        return dict(self._json("POST", f"/v1/messages/batches/{batch_id}/cancel"))

    def delete_message_batch(self, batch_id: str) -> dict[str, Any]:
        """``DELETE /v1/messages/batches/{id}`` — tombstone an ended
        batch; returns ``{id, type: "message_batch_deleted"}``. A batch
        that isn't ended refuses (Anthropic's contract)."""
        return dict(self._json("DELETE", f"/v1/messages/batches/{batch_id}"))

    def message_batch_results(self, batch_id: str) -> list[dict[str, Any]]:
        """``GET /v1/messages/batches/{id}/results`` — the results JSONL,
        parsed into ``{custom_id, result}`` row dicts. Only served once
        the batch has ended; before that the wire 400s."""
        _status, _headers, body = self._request(
            "GET", f"/v1/messages/batches/{batch_id}/results", idempotent=True
        )
        return [json.loads(line) for line in body.decode().splitlines() if line.strip()]

    def wait_message_batch(
        self,
        batch_id: str,
        *,
        poll_s: float = 0.5,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Poll ``message_batch`` until ``processing_status`` is
        ``ended``; returns the batch object. ``timeout_s`` None waits
        forever (the server expires the batch at ``expires_at``)."""
        deadline = None if timeout_s is None else self._clock() + timeout_s
        while True:
            batch = self.message_batch(batch_id)
            if batch.get("processing_status") == "ended":
                return batch
            remaining = None if deadline is None else deadline - self._clock()
            if remaining is not None and remaining <= 0:
                raise HarnessTransportError(
                    f"message batch {batch_id} did not end within {timeout_s}s"
                )
            self._sleep(poll_s if remaining is None else min(poll_s, remaining))

    def count_message_tokens(
        self, body: dict[str, Any], *, extra_headers: dict[str, str] | None = None
    ) -> int:
        """``POST /v1/messages/count_tokens`` — the provider's own input
        count for a ``/v1/messages``-shaped body, returned as the
        ``input_tokens`` int. ``tools``/``tool_choice`` refuse 400 (the
        tokenize channel sees only messages); a backend without a
        tokenize route fails closed 501 — the server never estimates.
        ``extra_headers`` carries ``X-Fx1-*`` link picks exactly like
        :meth:`create_message_batch`."""
        out = self._json("POST", "/v1/messages/count_tokens", body, extra_headers=extra_headers)
        return int(out["input_tokens"])

    def anthropic_models(
        self,
        *,
        limit: int | None = None,
        after_id: str | None = None,
        before_id: str | None = None,
    ) -> dict[str, Any]:
        """``GET /v1/models`` with ``anthropic-version`` — Anthropic's
        ``{data: [{type: "model", id, display_name, created_at}],
        first_id, last_id, has_more}`` envelope over the same inventory
        :meth:`list_models` serves. ``after_id``/``before_id`` are the
        positional id cursors; unknown cursors page to empty."""
        path = "/v1/models"
        params: list[str] = []
        if limit is not None:
            params.append(f"limit={limit}")
        if after_id is not None:
            params.append(f"after_id={urllib.parse.quote(after_id)}")
        if before_id is not None:
            params.append(f"before_id={urllib.parse.quote(before_id)}")
        if params:
            path += "?" + "&".join(params)
        return dict(
            self._json(
                "GET", path, idempotent=True, extra_headers={"anthropic-version": "2023-06-01"}
            )
        )

    def anthropic_model(self, model: str) -> dict[str, Any]:
        """``GET /v1/models/{id}`` with ``anthropic-version`` — the
        ``{type: "model"}`` card; unknown ids raise the 404-class
        ``not_found_error``, never a fabricated card."""
        return dict(
            self._json(
                "GET",
                f"/v1/models/{urllib.parse.quote(model, safe='')}",
                idempotent=True,
                extra_headers={"anthropic-version": "2023-06-01"},
            )
        )

    def responses_create(
        self,
        input: str | list[dict[str, Any]],
        *,
        model: str = "fx1",
        instructions: str | None = None,
        backend: str | None = None,
        byok: dict[str, str] | None = None,
        checkpoint_dir: str | Path | None = None,
        fallbacks: list[str] | None = None,
        receipt_hashes: list[str] | None = None,
        timeout_s: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
        metadata: dict[str, str] | None = None,
        service_tier: str | None = None,
        user: str | None = None,
        safety_identifier: str | None = None,
        reasoning_effort: str | None = None,
        verbosity: str | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        text_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        include: list[str] | None = None,
        top_logprobs: int | None = None,
        previous_response_id: str | None = None,
        conversation: str | dict[str, Any] | None = None,
        max_tool_calls: int | None = None,
        background: bool = False,
        idempotency_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        """POST /v1/responses — the Responses API over the gated pipeline.

        ``input`` is a string or a list of items
        (``{"type": "message", "role": ..., "content": [{"type": "input_text",
        "text": ...}]}`` or the shorthand ``{"role": ..., "content": "..."}``;
        ``function_call``/``function_call_output`` items carry a tool
        history into the next turn); ``instructions`` prepends a system
        turn. ``text_format`` is the ``text.format`` object
        (``{"type": "json_object"}`` / ``{"type": "json_schema",
        "schema": {...}}``) — post-validated, a violation is a
        provider-side 502. ``tools`` takes the flattened Responses spec
        (``{"type": "function", "name", "description", "parameters"}``);
        ``tool_choice`` is ``"none"``/``"auto"``/``"required"`` or
        ``{"type": "function", "name": ...}``. Calls land in ``output`` as
        ``{"type": "function_call", "call_id", "name", "arguments"}`` items.

        Returns ``(response_object, completion_id)`` — the response's
        ``output`` holds a ``message`` item whose ``content[0].text`` is
        the gated text when the model answers in prose; the cid links to
        the completion log. ``Idempotency-Key`` replays byte-identically.
        """
        payload = self._responses_payload(
            input,
            model=model,
            instructions=instructions,
            previous_response_id=previous_response_id,
            conversation=conversation,
            backend=backend,
            byok=byok,
            checkpoint_dir=checkpoint_dir,
            fallbacks=fallbacks,
            receipt_hashes=receipt_hashes,
            timeout_s=timeout_s,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            metadata=metadata,
            service_tier=service_tier,
            user=user,
            safety_identifier=safety_identifier,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
            text_format=text_format,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            include=include,
            top_logprobs=top_logprobs,
            max_tool_calls=max_tool_calls,
            background=background,
            stream=False,
        )
        if idempotency_key is not None:
            extra_headers = {**(extra_headers or {}), "Idempotency-Key": idempotency_key}
        _status, headers, body = self._request(
            "POST",
            "/v1/responses",
            payload,
            idempotent=idempotency_key is not None,
            extra_headers=extra_headers,
        )
        return json.loads(body), _hget(headers, "X-Fx1-Completion-Id")

    def responses_create_stream(  # NOSONAR(S3776)
        self,
        input: str | list[dict[str, Any]],
        *,
        idempotency_key: str | None = None,
        last_event_id: int | None = None,
        extra_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Streaming counterpart of :meth:`responses_create` — returns
        ``(events, completion_id)`` where events are the parsed Responses
        event payloads (``response.created`` … ``response.completed``,
        each carrying ``type``). ``last_event_id`` resumes a dropped
        keyed stream exactly like the chat surface — frames carry ``id:``
        equal to their event index."""
        payload = self._responses_payload(input, stream=True, **kwargs)
        if idempotency_key is not None:
            extra_headers = {**(extra_headers or {}), "Idempotency-Key": idempotency_key}
        if last_event_id is not None:
            extra_headers = {
                **(extra_headers or {}),
                "Last-Event-ID": str(last_event_id),
            }
        _status, headers, body = self._request(
            "POST",
            "/v1/responses",
            payload,
            idempotent=idempotency_key is not None,
            extra_headers=extra_headers,
        )
        return (
            _responses_sse_events(body),
            _hget(headers, "X-Fx1-Completion-Id"),
        )

    def _responses_payload(
        self,
        input: str | list[dict[str, Any]],
        *,
        model: str = "fx1",
        instructions: str | None = None,
        backend: str | None = None,
        byok: dict[str, str] | None = None,
        checkpoint_dir: str | Path | None = None,
        fallbacks: list[str] | None = None,
        receipt_hashes: list[str] | None = None,
        timeout_s: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
        metadata: dict[str, str] | None = None,
        service_tier: str | None = None,
        user: str | None = None,
        safety_identifier: str | None = None,
        reasoning_effort: str | None = None,
        verbosity: str | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        text_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        include: list[str] | None = None,
        top_logprobs: int | None = None,
        previous_response_id: str | None = None,
        conversation: str | dict[str, Any] | None = None,
        max_tool_calls: int | None = None,
        background: bool = False,
        stream: bool = False,
    ) -> dict[str, Any]:
        fx1 = _fx1_opts(
            backend=backend,
            byok=byok,
            checkpoint_dir=checkpoint_dir,
            fallbacks=fallbacks,
            receipt_hashes=receipt_hashes,
            timeout_s=timeout_s,
        )
        payload: dict[str, Any] = {
            "model": model,
            "input": input,
            "instructions": instructions,
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
            "metadata": metadata,
            "service_tier": service_tier,
            "user": user,
            "safety_identifier": safety_identifier,
            "previous_response_id": previous_response_id,
            "conversation": conversation,
            "max_tool_calls": max_tool_calls,
            "prompt_cache_key": prompt_cache_key,
            "prompt_cache_retention": prompt_cache_retention,
            "background": background,
            "stream": stream,
        }
        if reasoning_effort is not None:
            payload["reasoning"] = {"effort": reasoning_effort}
        if text_format is not None or verbosity is not None:
            text: dict[str, Any] = {}
            if text_format is not None:
                text["format"] = text_format
            if verbosity is not None:
                text["verbosity"] = verbosity
            payload["text"] = text
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = parallel_tool_calls
        if include is not None:
            payload["include"] = include
        if top_logprobs is not None:
            payload["top_logprobs"] = top_logprobs
        if fx1:
            payload["fx1"] = fx1
        return payload

    # ---- embeddings -----------------------------------------------------------

    def embeddings_create(
        self,
        input: str | list[str] | list[int] | list[list[int]],
        *,
        model: str = "fx1",
        backend: str | None = None,
        byok: dict[str, str] | None = None,
        checkpoint_dir: str | Path | None = None,
        fallbacks: list[str] | None = None,
        timeout_s: float | None = None,
        encoding_format: str | None = None,
        dimensions: int | None = None,
        user: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        """``POST /v1/embeddings`` — the embedding surface over the gated
        pipeline. ``input`` is a string, a string list, a token array, or
        a list of token arrays — forwarded verbatim. ``model`` reaches
        the provider verbatim (embedding models name themselves — the
        link's chat pin does not apply); ``encoding_format``
        (``"float"``/``"base64"``) and ``dimensions`` pass through. A
        link without the channel answers 501, never fabricated vectors.

        Returns ``(list envelope, completion_id)`` — the envelope's
        ``data[]`` is the provider's verbatim answer; the id links the
        completion-log record."""
        fx1 = _fx1_opts(
            backend=backend,
            byok=byok,
            checkpoint_dir=checkpoint_dir,
            fallbacks=fallbacks,
            receipt_hashes=None,
            timeout_s=timeout_s,
        )
        payload: dict[str, Any] = {
            "model": model,
            "input": input,
            "encoding_format": encoding_format,
            "dimensions": dimensions,
            "user": user,
        }
        if fx1:
            payload["fx1"] = fx1
        _status, headers, body = self._request(
            "POST",
            "/v1/embeddings",
            payload,
            extra_headers=extra_headers,
        )
        envelope = json.loads(body)
        return envelope, _hget(headers, "X-Fx1-Completion-Id")

    # ---- files + batches -----------------------------------------------------

    def upload_file(
        self,
        content: bytes,
        *,
        filename: str = "input.jsonl",
        purpose: str = "batch",
    ) -> dict[str, Any]:
        """``POST /v1/files`` — multipart upload of a batch-input JSONL.

        The multipart body is assembled here (stdlib only — no extra dep
        on the client); the server accepts only ``purpose='batch'`` and
        ``.jsonl`` names."""
        boundary = f"fx1{uuid.uuid4().hex}"
        head = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="purpose"\r\n\r\n'
            f"{purpose}\r\n"
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            "Content-Type: application/jsonl\r\n\r\n"
        ).encode()
        body = head + content + f"\r\n--{boundary}--\r\n".encode()
        _status, _headers, raw = self._request(
            "POST",
            "/v1/files",
            body,
            extra_headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        return dict(json.loads(raw))

    def files(self) -> list[dict[str, Any]]:
        """``GET /v1/files`` — newest-first listing."""
        out = self._json("GET", "/v1/files", idempotent=True)
        return list(out["data"])

    def file(self, file_id: str) -> dict[str, Any]:
        """``GET /v1/files/{id}`` — one file's card."""
        return dict(self._json("GET", f"/v1/files/{file_id}", idempotent=True))

    def file_content(self, file_id: str) -> bytes:
        """``GET /v1/files/{id}/content`` — raw bytes (JSONL in, JSONL out)."""
        _status, _headers, body = self._request(
            "GET", f"/v1/files/{file_id}/content", idempotent=True
        )
        return body

    def delete_file(self, file_id: str) -> dict[str, Any]:
        """``DELETE /v1/files/{id}``."""
        return dict(self._json("DELETE", f"/v1/files/{file_id}"))

    # ---- uploads (chunked files) ----------------------------------------------

    def upload_create(
        self,
        *,
        purpose: str = "batch",
        filename: str = "input.jsonl",
        bytes: int,
        mime_type: str = "application/jsonl",
    ) -> dict[str, Any]:
        """``POST /v1/uploads`` — open an upload intent for a payload
        larger than the request cap. Parts land via ``upload_part``;
        ``upload_complete`` mints the file.``bytes`` is the DECLARED total
        the parts must sum to — fail-closed both ways."""
        return dict(
            self._json(
                "POST",
                "/v1/uploads",
                {
                    "purpose": purpose,
                    "filename": filename,
                    "bytes": bytes,
                    "mime_type": mime_type,
                },
            )
        )

    def upload_part(self, upload_id: str, data: bytes) -> dict[str, Any]:
        """``POST /v1/uploads/{id}/parts`` — one chunk (multipart ``data``
        field, same hand-rolled assembly as ``upload_file``)."""
        boundary = f"fx1{uuid.uuid4().hex}"
        body = (
            (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="data"; filename="part"\r\n'
                "Content-Type: application/octet-stream\r\n\r\n"
            ).encode()
            + data
            + f"\r\n--{boundary}--\r\n".encode()
        )
        _status, _headers, raw = self._request(
            "POST",
            f"/v1/uploads/{upload_id}/parts",
            body,
            extra_headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        return dict(json.loads(raw))

    def upload_complete(
        self,
        upload_id: str,
        part_ids: list[str],
        *,
        md5: str | None = None,
    ) -> dict[str, Any]:
        """``POST /v1/uploads/{id}/complete`` — concatenate the parts in
        the given order into the file record. ``md5`` is checked before
        the file mints (a mismatch leaves no orphan)."""
        payload: dict[str, Any] = {"part_ids": part_ids}
        if md5 is not None:
            payload["md5"] = md5
        return dict(self._json("POST", f"/v1/uploads/{upload_id}/complete", payload))

    def upload_cancel(self, upload_id: str) -> dict[str, Any]:
        """``POST /v1/uploads/{id}/cancel`` — terminal cancel; replays
        200 on an already-cancelled record."""
        return dict(self._json("POST", f"/v1/uploads/{upload_id}/cancel"))

    def create_batch(
        self,
        input_file_id: str,
        *,
        endpoint: str = "/v1/chat/completions",
        metadata: dict[str, str] | None = None,
        idempotency_key: str | None = None,
        callback_url: str | None = None,
        callback_secret: str | None = None,
    ) -> dict[str, Any]:
        """``POST /v1/batches`` — submit an uploaded file as one batch.

        The batch runs under the caller's X-Fx1-* headers (backend/Byok
        routing applies to every line). ``Idempotency-Key`` replays the
        submit envelope — the shared /v1 idempotency space.
        ``callback_url``/``callback_secret`` are the fx1 terminal-webhook
        extension: the finished batch envelope POSTs to the URL once
        (completed/failed/expired/cancelled), signed when the secret is
        set."""
        payload: dict[str, Any] = {
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "completion_window": "24h",
        }
        if metadata is not None:
            payload["metadata"] = metadata
        if callback_url is not None:
            payload["callback_url"] = callback_url
        if callback_secret is not None:
            payload["callback_secret"] = callback_secret
        if idempotency_key is not None:
            hdrs = {"Idempotency-Key": idempotency_key}
            out = self._json("POST", "/v1/batches", payload, idempotent=True, extra_headers=hdrs)
        else:
            out = self._json("POST", "/v1/batches", payload)
        return dict(out)

    def batch(self, batch_id: str) -> dict[str, Any]:
        """``GET /v1/batches/{id}`` — status + request counts."""
        return dict(self._json("GET", f"/v1/batches/{batch_id}", idempotent=True))

    def batches(self, *, limit: int = 20, after: str | None = None) -> dict[str, Any]:
        """``GET /v1/batches`` — newest-first page (``after`` = last id of
        the previous page)."""
        path = f"/v1/batches?limit={limit}"
        if after is not None:
            path += f"&after={urllib.parse.quote(after)}"
        return dict(self._json("GET", path, idempotent=True))

    def cancel_batch(self, batch_id: str) -> dict[str, Any]:
        """``POST /v1/batches/{id}/cancel`` — cooperative cancel; the
        worker checks between lines and lands 'cancelled' with partial
        output written."""
        return dict(self._json("POST", f"/v1/batches/{batch_id}/cancel"))

    def wait_batch(
        self,
        batch_id: str,
        *,
        poll_s: float = 0.5,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Poll ``batch`` until a terminal status; returns the batch object.

        Raises ``HarnessJobError`` on failed/expired/cancelled and
        ``HarnessTransportError`` on timeout — same contract as
        ``wait_run``/``wait_eval``."""
        deadline = None if timeout_s is None else self._clock() + timeout_s
        while True:
            b = self.batch(batch_id)
            if b["status"] == "completed":
                return b
            if b["status"] in ("failed", "expired", "cancelled"):
                raise HarnessJobError(f"batch {batch_id} {b['status']}")
            remaining = None if deadline is None else deadline - self._clock()
            if remaining is not None and remaining <= 0:
                raise HarnessTransportError(
                    f"batch {batch_id} still {b['status']} after {timeout_s}s"
                )
            self._sleep(min(poll_s, remaining) if remaining is not None else poll_s)

    # ---- /v1 retrieval ------------------------------------------------------

    def retrieve_chat_completion(self, completion_id: str) -> dict[str, Any]:
        """``GET /v1/chat/completions/{id}`` — the stored ``chat.completion``
        envelope (``KeyError`` on 404: evicted, deleted, or ``store=false``)."""
        return dict(
            self._json(
                "GET",
                f"/v1/chat/completions/{urllib.parse.quote(completion_id)}",
                idempotent=True,
            )
        )

    def update_chat_completion(
        self, completion_id: str, *, metadata: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """``POST /v1/chat/completions/{id}`` — replace the stored
        completion's ``metadata`` wholesale (the only mutable field)."""
        body: dict[str, Any] = {"metadata": dict(metadata) if metadata is not None else {}}
        return dict(
            self._json(
                "POST",
                f"/v1/chat/completions/{urllib.parse.quote(completion_id)}",
                body,
            )
        )

    def delete_chat_completion(self, completion_id: str) -> dict[str, Any]:
        """``DELETE /v1/chat/completions/{id}`` — drop the stored envelope."""
        return dict(
            self._json(
                "DELETE",
                f"/v1/chat/completions/{urllib.parse.quote(completion_id)}",
            )
        )

    def retrieve_response(self, response_id: str) -> dict[str, Any]:
        """``GET /v1/responses/{id}`` — the stored ``response`` object."""
        return dict(
            self._json("GET", f"/v1/responses/{urllib.parse.quote(response_id)}", idempotent=True)
        )

    def delete_response(self, response_id: str) -> dict[str, Any]:
        """``DELETE /v1/responses/{id}`` — drop the stored envelope."""
        return dict(self._json("DELETE", f"/v1/responses/{urllib.parse.quote(response_id)}"))

    def cancel_response(self, response_id: str) -> dict[str, Any]:
        """``POST /v1/responses/{id}/cancel`` — cancel a queued or
        in-progress background response. Terminal responses are a 409;
        unknown ids a 404 (both surface as ``HarnessTransportError``)."""
        return dict(
            self._json("POST", f"/v1/responses/{urllib.parse.quote(response_id)}/cancel", {})
        )

    def responses_replay(
        self,
        response_id: str,
        *,
        starting_after: int | None = None,
        timeout_s: float | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """``GET /v1/responses/{id}?stream=true`` — the replay stream:
        returns ``(events, completion_id)`` where events are the parsed
        Responses event payloads replayed from the stored envelope — the
        full grammar for a terminal response, or the prelude plus a live
        follow for a still-``queued``/``in_progress`` background call.
        ``starting_after`` resumes past sequence N (the frames' ``id:``
        cursor); ``timeout_s`` bounds a live follow server-side. Unknown,
        deleted, or ``store=false`` ids raise ``KeyError`` off the 404
        envelope like :meth:`retrieve_response`; a stream that closes
        without a terminal ``response.*`` frame raises
        ``HarnessTransportError`` (record vanished mid-follow)."""
        q = "stream=true"
        if starting_after is not None:
            q += f"&starting_after={starting_after}"
        if timeout_s is not None:
            q += f"&timeout_s={timeout_s}"
        _status, headers, body = self._request(
            "GET",
            f"/v1/responses/{urllib.parse.quote(response_id)}?{q}",
            idempotent=True,
            extra_headers=extra_headers,
        )
        return (
            _responses_replay_events(body),
            _hget(headers, "X-Fx1-Completion-Id"),
        )

    def list_chat_completions(
        self,
        *,
        model: str | None = None,
        metadata: Mapping[str, str] | None = None,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
        order: str = "asc",
    ) -> dict[str, Any]:
        """``GET /v1/chat/completions`` — stored completions, filtered by
        ``model`` and/or an exact ``metadata`` subset, paged by id."""
        q = f"limit={limit}&order={order}"
        if model:
            q += f"&model={urllib.parse.quote(model)}"
        for k, v in (metadata or {}).items():
            q += f"&metadata[{urllib.parse.quote(k)}]={urllib.parse.quote(v)}"
        if after:
            q += f"&after={urllib.parse.quote(after)}"
        if before:
            q += f"&before={urllib.parse.quote(before)}"
        return dict(self._json("GET", f"/v1/chat/completions?{q}", idempotent=True))

    def chat_completion_messages(
        self,
        completion_id: str,
        *,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
        order: str = "asc",
    ) -> dict[str, Any]:
        """``GET /v1/chat/completions/{id}/messages`` — the stored
        request messages, paged by item id."""
        cid = urllib.parse.quote(completion_id)
        q = f"limit={limit}&order={order}"
        if after:
            q += f"&after={urllib.parse.quote(after)}"
        if before:
            q += f"&before={urllib.parse.quote(before)}"
        return dict(self._json("GET", f"/v1/chat/completions/{cid}/messages?{q}", idempotent=True))

    def response_input_items(
        self,
        response_id: str,
        *,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
        order: str = "asc",
    ) -> dict[str, Any]:
        """``GET /v1/responses/{id}/input_items`` — the stored ``input``
        items, paged by item id."""
        rid = urllib.parse.quote(response_id)
        q = f"limit={limit}&order={order}"
        if after:
            q += f"&after={urllib.parse.quote(after)}"
        if before:
            q += f"&before={urllib.parse.quote(before)}"
        return dict(self._json("GET", f"/v1/responses/{rid}/input_items?{q}", idempotent=True))

    # ---- conversations -------------------------------------------------------

    def conversation_create(
        self,
        *,
        items: list[dict[str, Any]] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """``POST /v1/conversations`` — mint a ``conv_*`` container.
        ``items`` seeds the list; ``metadata`` stamps the object."""
        payload: dict[str, Any] = {}
        if items is not None:
            payload["items"] = items
        if metadata is not None:
            payload["metadata"] = metadata
        return dict(self._json("POST", "/v1/conversations", payload))

    def conversation_get(self, conversation_id: str) -> dict[str, Any]:
        """``GET /v1/conversations/{id}`` — the conversation object."""
        return dict(
            self._json(
                "GET",
                f"/v1/conversations/{urllib.parse.quote(conversation_id)}",
                idempotent=True,
            )
        )

    def conversation_update(
        self,
        conversation_id: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """``POST /v1/conversations/{id}`` — ``metadata`` replaces the
        object's metadata wholesale."""
        return dict(
            self._json(
                "POST",
                f"/v1/conversations/{urllib.parse.quote(conversation_id)}",
                {"metadata": metadata},
            )
        )

    def conversation_delete(self, conversation_id: str) -> dict[str, Any]:
        """``DELETE /v1/conversations/{id}`` — drop the container and its
        items."""
        return dict(
            self._json("DELETE", f"/v1/conversations/{urllib.parse.quote(conversation_id)}")
        )

    def conversation_items(
        self,
        conversation_id: str,
        *,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
        order: str = "asc",
    ) -> dict[str, Any]:
        """``GET /v1/conversations/{id}/items`` — the accumulated items,
        paged by item id."""
        cid = urllib.parse.quote(conversation_id)
        q = f"limit={limit}&order={order}"
        if after:
            q += f"&after={urllib.parse.quote(after)}"
        if before:
            q += f"&before={urllib.parse.quote(before)}"
        return dict(self._json("GET", f"/v1/conversations/{cid}/items?{q}", idempotent=True))

    def conversation_items_add(
        self,
        conversation_id: str,
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """``POST /v1/conversations/{id}/items`` — append item dicts,
        returns the minted list."""
        return dict(
            self._json(
                "POST",
                f"/v1/conversations/{urllib.parse.quote(conversation_id)}/items",
                {"items": items},
            )
        )

    def conversation_item(
        self,
        conversation_id: str,
        item_id: str,
    ) -> dict[str, Any]:
        """``GET /v1/conversations/{id}/items/{item_id}`` — one item by
        id."""
        return dict(
            self._json(
                "GET",
                f"/v1/conversations/{urllib.parse.quote(conversation_id)}"
                f"/items/{urllib.parse.quote(item_id)}",
                idempotent=True,
            )
        )

    def conversation_item_delete(
        self,
        conversation_id: str,
        item_id: str,
    ) -> dict[str, Any]:
        """``DELETE /v1/conversations/{id}/items/{item_id}`` — drop one
        item; returns the conversation object."""
        return dict(
            self._json(
                "DELETE",
                f"/v1/conversations/{urllib.parse.quote(conversation_id)}"
                f"/items/{urllib.parse.quote(item_id)}",
            )
        )

    # ---- vector stores -------------------------------------------------------

    def vector_store_create(
        self,
        *,
        name: str | None = None,
        metadata: dict[str, str] | None = None,
        file_ids: list[str] | None = None,
        expires_after: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """``POST /v1/vector_stores`` — mint a ``vs_*`` retrieval store;
        ``file_ids`` attach at create (a bogus id fails the call);
        ``expires_after`` is the OpenAI anchor policy
        ``{"anchor": "last_active_at", "days": 1..365}``."""
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if metadata is not None:
            payload["metadata"] = metadata
        if file_ids is not None:
            payload["file_ids"] = file_ids
        if expires_after is not None:
            payload["expires_after"] = expires_after
        return dict(self._json("POST", "/v1/vector_stores", payload))

    def vector_store_get(self, vector_store_id: str) -> dict[str, Any]:
        """``GET /v1/vector_stores/{id}`` — the store object."""
        return dict(
            self._json(
                "GET",
                f"/v1/vector_stores/{urllib.parse.quote(vector_store_id)}",
                idempotent=True,
            )
        )

    def vector_store_update(
        self,
        vector_store_id: str,
        *,
        name: str | None = None,
        metadata: dict[str, str] | None = None,
        expires_after: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """``POST /v1/vector_stores/{id}`` — name/metadata replace
        wholesale when given; ``expires_after`` re-anchors the expiry
        window (revives an expired store)."""
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if metadata is not None:
            payload["metadata"] = metadata
        if expires_after is not None:
            payload["expires_after"] = expires_after
        return dict(
            self._json(
                "POST",
                f"/v1/vector_stores/{urllib.parse.quote(vector_store_id)}",
                payload,
            )
        )

    def vector_store_delete(self, vector_store_id: str) -> dict[str, Any]:
        """``DELETE /v1/vector_stores/{id}`` — drop the store + index;
        member ``file-*`` records survive."""
        return dict(
            self._json(
                "DELETE",
                f"/v1/vector_stores/{urllib.parse.quote(vector_store_id)}",
            )
        )

    def vector_store_list(
        self,
        *,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
        order: str = "desc",
    ) -> dict[str, Any]:
        """``GET /v1/vector_stores`` — cursor-paged store list."""
        q = f"limit={limit}&order={order}"
        if after:
            q += f"&after={urllib.parse.quote(after)}"
        if before:
            q += f"&before={urllib.parse.quote(before)}"
        return dict(self._json("GET", f"/v1/vector_stores?{q}", idempotent=True))

    def vector_store_file_create(
        self,
        vector_store_id: str,
        file_id: str,
        *,
        attributes: dict[str, Any] | None = None,
        chunking_strategy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """``POST /v1/vector_stores/{id}/files`` — index a ``file-*``
        record into the store."""
        payload: dict[str, Any] = {"file_id": file_id}
        if attributes is not None:
            payload["attributes"] = attributes
        if chunking_strategy is not None:
            payload["chunking_strategy"] = chunking_strategy
        return dict(
            self._json(
                "POST",
                f"/v1/vector_stores/{urllib.parse.quote(vector_store_id)}/files",
                payload,
            )
        )

    def vector_store_file_list(
        self,
        vector_store_id: str,
        *,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
        order: str = "asc",
        filter: str | None = None,
    ) -> dict[str, Any]:
        """``GET /v1/vector_stores/{id}/files`` — attachments, paged;
        ``filter`` is an OpenAI status word."""
        q = f"limit={limit}&order={order}"
        if after:
            q += f"&after={urllib.parse.quote(after)}"
        if before:
            q += f"&before={urllib.parse.quote(before)}"
        if filter:
            q += f"&filter={urllib.parse.quote(filter)}"
        return dict(
            self._json(
                "GET",
                f"/v1/vector_stores/{urllib.parse.quote(vector_store_id)}/files?{q}",
                idempotent=True,
            )
        )

    def vector_store_file_get(self, vector_store_id: str, file_id: str) -> dict[str, Any]:
        """``GET /v1/vector_stores/{id}/files/{file_id}`` — one
        attachment's status/chunks/attributes."""
        return dict(
            self._json(
                "GET",
                f"/v1/vector_stores/{urllib.parse.quote(vector_store_id)}"
                f"/files/{urllib.parse.quote(file_id)}",
                idempotent=True,
            )
        )

    def vector_store_file_delete(self, vector_store_id: str, file_id: str) -> dict[str, Any]:
        """``DELETE /v1/vector_stores/{id}/files/{file_id}`` — detach;
        the file record survives."""
        return dict(
            self._json(
                "DELETE",
                f"/v1/vector_stores/{urllib.parse.quote(vector_store_id)}"
                f"/files/{urllib.parse.quote(file_id)}",
            )
        )

    def vector_store_file_content(self, vector_store_id: str, file_id: str) -> dict[str, Any]:
        """``GET /v1/vector_stores/{id}/files/{file_id}/content`` — the
        stored decoded text as a page of ``{type: 'text'}`` parts."""
        return dict(
            self._json(
                "GET",
                f"/v1/vector_stores/{urllib.parse.quote(vector_store_id)}"
                f"/files/{urllib.parse.quote(file_id)}/content",
                idempotent=True,
            )
        )

    def vector_store_search(
        self,
        vector_store_id: str,
        query: str | list[str],
        *,
        max_num_results: int | None = None,
        filters: dict[str, Any] | None = None,
        ranking_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """``POST /v1/vector_stores/{id}/search`` — ranked hits without
        spending a response turn; a ``vector_store.search_results.page``
        dict."""
        payload: dict[str, Any] = {"query": query}
        if max_num_results is not None:
            payload["max_num_results"] = max_num_results
        if filters is not None:
            payload["filters"] = filters
        if ranking_options is not None:
            payload["ranking_options"] = ranking_options
        return dict(
            self._json(
                "POST",
                f"/v1/vector_stores/{urllib.parse.quote(vector_store_id)}/search",
                payload,
            )
        )

    def vector_store_file_batch_create(
        self,
        vector_store_id: str,
        file_ids: list[str],
        *,
        attributes: dict[str, Any] | None = None,
        chunking_strategy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """``POST /v1/vector_stores/{id}/file_batches`` — attach many
        ``file-*`` records; per-file refusals count, never abort.
        Status is terminal at return (sync attach)."""
        payload: dict[str, Any] = {"file_ids": file_ids}
        if attributes is not None:
            payload["attributes"] = attributes
        if chunking_strategy is not None:
            payload["chunking_strategy"] = chunking_strategy
        return dict(
            self._json(
                "POST",
                f"/v1/vector_stores/{urllib.parse.quote(vector_store_id)}/file_batches",
                payload,
            )
        )

    def vector_store_file_batch_get(self, vector_store_id: str, batch_id: str) -> dict[str, Any]:
        """``GET .../file_batches/{batch_id}`` — standing status +
        file_counts."""
        return dict(
            self._json(
                "GET",
                f"/v1/vector_stores/{urllib.parse.quote(vector_store_id)}"
                f"/file_batches/{urllib.parse.quote(batch_id)}",
                idempotent=True,
            )
        )

    def vector_store_file_batch_cancel(self, vector_store_id: str, batch_id: str) -> dict[str, Any]:
        """``POST .../file_batches/{batch_id}/cancel`` — batches are
        terminal at create; the server answers 409 ``file_batch_terminal``."""
        return dict(
            self._json(
                "POST",
                f"/v1/vector_stores/{urllib.parse.quote(vector_store_id)}"
                f"/file_batches/{urllib.parse.quote(batch_id)}/cancel",
                {},
            )
        )

    def vector_store_file_batch_files(
        self,
        vector_store_id: str,
        batch_id: str,
        *,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
        order: str = "asc",
        filter: str | None = None,
    ) -> dict[str, Any]:
        """``GET .../file_batches/{batch_id}/files`` — the frozen
        per-file verdicts, paged; ``filter`` is an OpenAI status word."""
        q = f"limit={limit}&order={order}"
        if after:
            q += f"&after={urllib.parse.quote(after)}"
        if before:
            q += f"&before={urllib.parse.quote(before)}"
        if filter:
            q += f"&filter={urllib.parse.quote(filter)}"
        return dict(
            self._json(
                "GET",
                f"/v1/vector_stores/{urllib.parse.quote(vector_store_id)}"
                f"/file_batches/{urllib.parse.quote(batch_id)}/files?{q}",
                idempotent=True,
            )
        )

    # ---- receipt store -------------------------------------------------------

    def receipts(self) -> tuple[ReceiptRef, ...]:
        """GET /receipts — index the server's sealed-receipt store
        (content hash → filename, sorted)."""
        out = self._json("GET", "/receipts", idempotent=True)
        assert isinstance(out, dict)
        return tuple(ReceiptRef(sha256=item["sha256"], name=item["name"]) for item in out["items"])

    def receipt(self, sha256: str) -> StoredReceipt:
        """GET /receipts/{sha256} — the sealed document verbatim plus the
        server's live re-verify verdict (``X-Fx1-Receipt-Valid``). Unknown
        hashes raise KeyError; malformed digests raise ValueError."""
        _status, headers, body = self._request("GET", f"/receipts/{sha256}", idempotent=True)
        doc = json.loads(body)
        assert isinstance(doc, dict)
        return StoredReceipt(
            sha256=sha256,
            document=doc,
            valid=headers.get("x-fx1-receipt-valid") == "true",
        )

    # ---- receipts --------------------------------------------------------

    def verify_receipt(self, receipt: dict[str, Any]) -> ReceiptVerdict:
        """Remote counterpart of ``Fx1Harness.verify_receipt``."""
        out = self._json("POST", "/receipts/verify", {"receipt": receipt}, idempotent=True)
        return ReceiptVerdict(
            valid=out["valid"],
            path=out["path"],
            schema_tag=out["schema_tag"],
            kind=out["kind"],
            verdict=out["verdict"],
            digest_convention=out["digest_convention"],
            errors=tuple(out["errors"]),
            warnings=tuple(out["warnings"]),
        )

    def verify_receipts(self, receipts: list[dict[str, Any]]) -> tuple[ReceiptVerdict, ...]:
        """One-round-trip batch verify — the remote counterpart of looping
        ``verify_receipt``; per-item verdicts, no partial-failure abort."""
        out = self._json("POST", "/receipts/verify/batch", {"receipts": receipts}, idempotent=True)
        verdicts = []
        for item in out["results"]:
            verdicts.append(
                ReceiptVerdict(
                    valid=item["valid"],
                    path=item["path"],
                    schema_tag=item["schema_tag"],
                    kind=item["kind"],
                    verdict=item["verdict"],
                    digest_convention=item["digest_convention"],
                    errors=tuple(item["errors"]),
                    warnings=tuple(item["warnings"]),
                )
            )
        return tuple(verdicts)

    # ---- health ------------------------------------------------------------

    def health(self) -> HarnessHealth:
        """Remote counterpart of ``Fx1Harness.health``."""
        out = self._json("GET", "/health", idempotent=True)
        return HarnessHealth(
            status=out["status"],
            version=out["version"],
            registered_commands=out["registered_commands"],
            backends=dict(out["backends"]),
        )

    # ---- ops --------------------------------------------------------------

    def metrics(self) -> OpsMetrics:
        """Ops counters from the remote harness — wire-only (the in-process
        SDK has no HTTP layer to meter)."""
        out = self._json("GET", "/metrics", idempotent=True)
        return OpsMetrics(
            uptime_s=out["uptime_s"],
            requests_total=out["requests_total"],
            errors_total=out["errors_total"],
            by_status=dict(out["by_status"]),
            inflight=out["inflight"],
            inflight_watermark=out["inflight_watermark"],
            max_inflight=out["max_inflight"],
            draining=bool(out.get("draining", False)),
            rate_limited_total=int(out.get("rate_limited_total", 0)),
            complete={k: dict(v) for k, v in dict(out.get("complete", {})).items()},
        )

    def metrics_text(self) -> str:
        """Prometheus text exposition of the remote harness — same
        ``/metrics`` route, content-negotiated to ``text/plain``."""
        _, _, body = self._request(
            "GET",
            "/metrics",
            idempotent=True,
            extra_headers={"Accept": "text/plain"},
        )
        return body.decode("utf-8")

    def drain(self, wait_s: float = 0.0) -> dict[str, Any]:
        """Latch the remote harness into drain mode — one-way: gated routes
        refuse new work (503), in-flight requests finish, ``/metrics`` keeps
        reporting ``inflight`` so a deploy can wait for it to hit zero before
        stopping the process. Idempotent; marks the latch idempotent=True so
        transport blips retry. ``wait_s>0`` lets the server block until the
        in-flight pool empties — ``drained`` in the payload reports whether
        it did within the window."""
        path = "/harness/drain"
        if wait_s > 0:
            path += "?" + urllib.parse.urlencode({"wait_s": wait_s})
        out = self._json("POST", path, {}, idempotent=True)
        return {
            "draining": bool(out["draining"]),
            "inflight": int(out["inflight"]),
            "drained": bool(out.get("drained", out["inflight"] == 0)),
        }

    @property
    def last_api_version(self) -> str | None:
        """Wire-contract version stamped on the last response
        (``X-Fx1-Api-Version``); ``None`` before the first call or when
        talking to a pre-versioning server."""
        return self._last_api_version

    @property
    def last_response_headers(self) -> dict[str, str]:
        """Lowercased name → value map of the last transport response —
        the drop-in header surface (``x-request-id``,
        ``openai-processing-ms``, ``openai-version``, ``x-ratelimit-*``,
        plus ``request-id`` / ``anthropic-ratelimit-*`` /
        ``x-should-retry`` on the Anthropic grammar). ``{}`` before the
        first call or after a transport fault (no response arrived)."""
        return dict(self._last_response_headers)

    def server_version(self) -> dict[str, Any]:
        """GET /harness/version — the server's ``{"api_version",
        "fx1_version"}`` for version negotiation before sending work."""
        out = self._json("GET", "/harness/version", idempotent=True)
        return dict(out)

    def check_compat(self, *, strict: bool = True) -> dict[str, Any]:
        """Wire-contract negotiation: fetch ``/harness/version`` and compare
        the server's ``api_version`` against ``EXPECTED_API_VERSION`` — the
        contract this client was built to speak. Returns the report dict
        (``compatible``, both api/fx1 versions); ``strict`` (the default)
        raises ``HarnessCompatError`` on any mismatch, including a peer old
        enough to have no version route at all."""
        try:
            out = self.server_version()
            raw = out.get("api_version")
            remote: str | None = str(raw) if raw is not None else None
            fx1_version = out.get("fx1_version")
        except KeyError:
            remote = None  # peer predates the version route entirely
            fx1_version = None
        compatible = remote == EXPECTED_API_VERSION
        report = {
            "compatible": compatible,
            "client_api_version": EXPECTED_API_VERSION,
            "server_api_version": remote,
            "server_fx1_version": fx1_version,
            "client_fx1_version": __version__,
        }
        if strict and not compatible:
            raise HarnessCompatError(
                f"harness wire contract mismatch: server api_version={remote!r}, "
                f"client expects {EXPECTED_API_VERSION!r} — upgrade the server "
                f"or pin the client",
                code="incompatible_contract",
            )
        return report

    def capabilities(self) -> dict[str, Any]:
        """GET /harness/capabilities — feature flags and effective limits.
        Clients self-configure from this instead of hardcoding limits."""
        out = self._json("GET", "/harness/capabilities", idempotent=True)
        return dict(out)

    def ready(self) -> dict[str, Any]:
        """Readiness probe: returns the payload while the server accepts new
        work; raises ``BackendNotConfiguredError`` (503) once drain is
        latched — a deploy loop polls this before cutting traffic."""
        out = self._json("GET", "/ready", idempotent=True)
        return {"ready": bool(out["ready"]), "inflight": int(out["inflight"])}
