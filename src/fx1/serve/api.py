"""The fx-1 harness HTTP surface.

``fx1.harness`` registers every lab command the model may touch; this app
exposes that registry — plus receipt verification and backend completion —
over HTTP so an fx-1 instance drives the harness through one authenticated
socket instead of shell access.

Routes (all POST bodies are ``extra="forbid"`` — no silent arguments):

- ``GET  /health`` — liveness + which backends are configured (booleans
  only; credential values never leave this process).
- ``GET  /harness/commands`` — the registry; ``?role=`` filters by
  HarnessRole. Anything not listed is unreachable — fail-closed by
  construction, same as the in-process surface.
- ``POST /harness/runs`` — execute a registered command through
  :class:`~fx1.harness.Harness`; unknown names 404, ``configs/`` escapes
  422, and the command's own timeout bounds the call.
- ``POST /harness/complete`` and ``/complete/batch`` — gated chat through
  ``cited_complete``. Config failures 503, unknown backends 404, honesty
  violations 502. Batch items return per-item verdicts.
- ``POST /receipts/verify`` — ``verify_receipt_payload`` with no filesystem.

Auth posture mirrors ``quant_fund.api.research_api``: ``/health`` is the
only unauthenticated route; when ``FX1_API_KEY`` is set every other route
requires ``X-API-Key``; unset, only loopback clients are served.
"""

from __future__ import annotations

import builtins
import contextvars
import hashlib
import hmac
import inspect
import io
import json
import logging
import math
import os
import queue
import re
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    asynccontextmanager,
    contextmanager,
    suppress,
)
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, cast

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.gzip import GZipMiddleware

from fx1 import __version__
from fx1.harness import Harness, HarnessRole
from fx1.honesty import Fx1HonestyError, honesty_categories, validate_fx1_output
from fx1.reward import score_response
from fx1.serve.anthropic_compat import (
    AnthropicBatchCounts,
    AnthropicBatchCreate,
    AnthropicBatchItem,
    AnthropicCountTokensRequest,
    AnthropicMessageObject,
    AnthropicMessagesRequest,
    _rfc3339,
    anthropic_batch_object,
    anthropic_batch_result,
    anthropic_count_messages,
    anthropic_envelope,
    anthropic_error_body,
    anthropic_model_object,
    anthropic_sse,
    anthropic_to_openai,
)
from fx1.serve.backends import (
    BYOK_API_KEY_ENV,
    BYOK_BASE_URL_ENV,
    BYOK_MODEL_ENV,
    LOCAL_SERVE_CMD_ENV,
    LOCAL_SERVE_URL_ENV,
    BackendNotConfiguredError,
    EmbeddingBackend,
    SamplingParams,
    StreamingBackend,
    TokenCountingBackend,
    TokenCountUnavailableError,
    get_backend,
    truncate_chunks,
)
from fx1.serve.chat import cited_complete, cited_complete_tools
from fx1.serve.contract import API_VERSION as API_VERSION
from fx1.serve.evals import (
    EVAL_SAMPLING,
    EVAL_SUITES,
    EvalDiff,
    EvalRecord,
    EvalSpec,
    EvalSpecItemSchema,
    EvalSpecStore,
    EvalStore,
    EvalSuiteName,
    diff_eval_records,
    eval_record_receipt,
    metered_model,
    report_task_items,
    run_eval_record,
    run_wire,
    spec_wire,
    suite_accepts_judge,
)
from fx1.serve.finetune import (
    TRAINABLE_MODELS,
    FTEventList,
    FTHyperparameters,
    FTJob,
    FTJobCheckpointList,
    FTJobEntry,
    FTJobError,
    FTJobList,
    FTJobRequest,
    FTJobRunner,
    FTJobSpec,
    FTJobStore,
    default_ft_runner,
    validate_chat_jsonl,
)
from fx1.serve.journal import JobJournal, _ClaimLocks
from fx1.serve.keys import CLEARABLE_KEY_FIELDS, SCOPES, ApiKeyStore, KeyStoreError
from fx1.serve.openai_compat import (
    OPENAI_FILE_PURPOSE_ACCEPT,
    OPENAI_MODEL_IDS,
    OPENAI_RESPONSE_TERMINAL,
    ByokOverride,
    OpenAIBatchRequest,
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIChatUpdate,
    OpenAICompatError,
    OpenAICompletionRequest,
    OpenAIConversationCreate,
    OpenAIConversationItemsAdd,
    OpenAIConversationUpdate,
    OpenAIEmbeddingRequest,
    OpenAIEmbeddingResponse,
    OpenAIEnvelopeStore,
    OpenAIModel,
    OpenAIModelDelete,
    OpenAIModelList,
    OpenAIResponseRequest,
    OpenAIUploadCompleteRequest,
    OpenAIUploadCreateRequest,
    OpenAIVectorStoreCreate,
    OpenAIVectorStoreFileBatchCreate,
    OpenAIVectorStoreFileCreate,
    OpenAIVectorStoreSearch,
    OpenAIVectorStoreUpdate,
    _resolve_openai_link,
    _resolve_timeout,
    batch_line_body,
    batch_line_shape,
    batch_object,
    batch_output_line,
    chained_response_input,
    chat_messages_for_store,
    completion_events,
    conversation_id_of,
    embeddings_to_kwargs,
    file_object,
    file_search_call_item,
    is_openai_path,
    legacy_to_chat,
    openai_chunks,
    openai_completion_envelope,
    openai_conversation_object,
    openai_embedding_envelope,
    openai_envelope,
    openai_error_body,
    openai_model,
    openai_response_events,
    openai_response_object,
    openai_response_replay_events,
    openai_to_kwargs,
    paged_item_list,
    response_cap_call_items,
    response_input_item_dicts,
    response_input_items_for_store,
    response_output_pieces,
    response_query_text,
    response_text_format,
    response_to_kwargs,
    validate_openai_output,
    validate_response_format,
)
from fx1.serve.receipt_store import SHA256_HEX as _SHA256_HEX
from fx1.serve.receipt_store import ReceiptIndex as _ReceiptIndex
from fx1.serve.uploads import (
    UploadMeta,
    UploadStore,
    UploadStoreError,
    upload_object,
    validate_upload_intent,
)
from fx1.serve.usage_report import UsageReport, aggregate_usage
from fx1.serve.vectorstores import (
    VS_MAX_RESULTS,
    VectorStoreError,
    VectorStoreStore,
)
from fx1.serve.webhooks import check_callback_url, deliver_signed
from quant_fund.schemas.receipt import verify_receipt_bytes, verify_receipt_payload

_OBJ_CHAT_COMPLETION = "chat.completion"
_EV_JOB_CANCELLED = "job cancelled"

_API_KEY_ENV = "FX1_API_KEY"
# The authenticated credential fingerprint for the in-flight request.
# Set by the auth middleware; read where completion records are stamped
# so a completion is attributable to the key that made it. ``None`` on
# the unauthenticated loopback-dev surface.
_REQUEST_KEY_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "fx1_request_key_id", default=None
)
_MAX_INFLIGHT_ENV = "FX1_API_MAX_INFLIGHT"
_SSE_KEEPALIVE_ENV = "FX1_API_SSE_KEEPALIVE_S"
_IDEM_MAX_ENV = "FX1_API_IDEM_MAX"
_IDEM_KEY_MAX = 256
# One POST verifies a whole receipt bundle — fx-1's gating path handles
# dozens per task; still bounded so a hostile bundle can't pin a worker.
_VERIFY_BATCH_MAX = 64
_JOB_MAX_ENV = "FX1_API_JOB_MAX"
_RATE_LIMIT_ENV = "FX1_API_RATE_LIMIT_RPS"
_GZIP_MIN_ENV = "FX1_API_GZIP_MIN_BYTES"
_CORS_ORIGINS_ENV = "FX1_API_CORS_ORIGINS"
_BREAKER_THRESHOLD_ENV = "FX1_API_BREAKER_THRESHOLD"
_BREAKER_COOLDOWN_ENV = "FX1_API_BREAKER_COOLDOWN_S"
_RECEIPTS_DIR_ENV = "FX1_API_RECEIPTS_DIR"
_BYOK_OVERRIDE_ENV = "FX1_API_BYOK_OVERRIDE"
_FILE_MAX_ENV = "FX1_API_FILE_MAX"
_FILE_BYTES_ENV = "FX1_API_FILE_BYTES"
_BATCH_MAX_ENV = "FX1_API_BATCH_MAX"
_BATCH_LINES_ENV = "FX1_API_BATCH_LINES"
_STORE_MAX_ENV = "FX1_API_STORE_MAX"
_STATE_DIR_ENV = "FX1_API_STATE_DIR"

# Headers browser clients can read off responses when CORS is enabled.
_CORS_EXPOSE_HEADERS = [
    "ETag",
    "Location",
    "Openai-Processing-Ms",
    "Retry-After",
    "X-Fx1-Api-Version",
    "X-Fx1-Completion-Id",
    "X-Fx1-Receipt-Sha256",
    "X-Fx1-Receipt-Valid",
    "X-RateLimit-Limit",
    "X-RateLimit-Remaining",
    "X-RateLimit-Reset",
    "X-RateLimit-Limit-Requests",
    "X-RateLimit-Remaining-Requests",
    "X-RateLimit-Reset-Requests",
    "X-Request-ID",
    # Anthropic-grammar responses (/v1/messages*) — the stock anthropic
    # SDK's names for the same request-id / standing-budget surfaces
    "Anthropic-RateLimit-Requests-Limit",
    "Anthropic-RateLimit-Requests-Remaining",
    "Anthropic-RateLimit-Requests-Reset",
    "Request-ID",
    "X-Should-Retry",
]
_CORS_ALLOW_HEADERS = [
    "Content-Type",
    "Idempotency-Key",
    "Last-Event-ID",
    "X-API-Key",
    # the X-Fx1-* request knobs — browser clients must be able to send
    # backend/byok/fallbacks/checkpoint/timeout/citation headers too
    "X-Fx1-Backend",
    "X-Fx1-Byok-Api-Key",
    "X-Fx1-Byok-Base-Url",
    "X-Fx1-Byok-Model",
    "X-Fx1-Checkpoint-Dir",
    "X-Fx1-Fallbacks",
    "X-Fx1-Receipt-Hashes",
    "X-Fx1-Timeout",
    "X-Request-ID",
    # the Anthropic-grammar request headers — the stock anthropic SDK
    # sends these unconditionally on /v1/messages*
    "Anthropic-Beta",
    "Anthropic-Dangerous-Direct-Browser-Access",
    "Anthropic-Version",
    "X-Api-Key",
]
_RATE_LIMIT_KEYS_MAX = 4096


def _version_info() -> VersionResponse:
    """Version negotiation: the wire contract + package release."""
    return VersionResponse(api_version=API_VERSION, fx1_version=__version__)


_PUBLIC_PATHS = frozenset({"/health"})

# Response headers the middleware stamps on every response — declared on the
# OpenAPI spec so generated clients see them typed instead of having to know.
_DECLARED_COMMON_HEADERS: dict[str, dict[str, Any]] = {
    "X-Request-ID": {
        "schema": {"type": "string"},
        "description": "Request id — echoed from the inbound X-Request-ID or minted.",
    },
    "X-Fx1-Api-Version": {
        "schema": {"type": "string"},
        "description": "Wire-contract version; clients gate on it via /harness/version.",
    },
    "X-Content-Type-Options": {
        "schema": {"type": "string"},
        "description": "Always `nosniff`.",
    },
    "Cache-Control": {"schema": {"type": "string"}, "description": "Always `no-store`."},
    "Referrer-Policy": {
        "schema": {"type": "string"},
        "description": "Always `no-referrer`.",
    },
    "Openai-Processing-Ms": {
        "schema": {"type": "integer"},
        "description": "Server-side wall-clock milliseconds for the request — the "
        "OpenAI-convention tracing header, present on every response.",
    },
    "X-RateLimit-Limit-Requests": {
        "schema": {"type": "integer"},
        "description": "Managed-key rpm window size — present only on responses "
        "authenticated by an `fx1k_` key minted with `rpm` (and its 429s).",
    },
    "X-RateLimit-Remaining-Requests": {
        "schema": {"type": "integer"},
        "description": "Requests left in the key's fixed 60 s window after this response.",
    },
    "X-RateLimit-Reset-Requests": {
        "schema": {"type": "integer"},
        "description": "Seconds until the key's rpm window reopens.",
    },
}
_DECLARED_RATELIMIT_HEADERS: dict[str, dict[str, Any]] = {
    "X-RateLimit-Limit": {
        "schema": {"type": "integer"},
        "description": "Configured request budget per second (present only when the "
        "rate limiter is enabled).",
    },
    "X-RateLimit-Remaining": {
        "schema": {"type": "integer"},
        "description": "Tokens left in this client's bucket after this response.",
    },
    "X-RateLimit-Reset": {
        "schema": {"type": "integer"},
        "description": "Seconds until the bucket refills.",
    },
}
# Anthropic-dialect responses (/v1/messages*, or /v1/* under an
# `anthropic-version` header) carry the same request-id / standing-budget
# surfaces under the names the stock anthropic SDK reads.
_DECLARED_ANTHROPIC_HEADERS: dict[str, dict[str, Any]] = {
    "request-id": {
        "schema": {"type": "string"},
        "description": "Anthropic's request-id header — the same id as X-Request-ID.",
    },
    "anthropic-ratelimit-requests-limit": {
        "schema": {"type": "integer"},
        "description": "Managed-key rpm window size — present only when the "
        "credential carries a declared rpm window.",
    },
    "anthropic-ratelimit-requests-remaining": {
        "schema": {"type": "integer"},
        "description": "Requests left in the key's fixed 60 s window after this response.",
    },
    "anthropic-ratelimit-requests-reset": {
        "schema": {"type": "string", "format": "date-time"},
        "description": "RFC 3339 instant when the key's rpm window reopens.",
    },
    "x-should-retry": {
        "schema": {"type": "string", "enum": ["true", "false"]},
        "description": "Retry guidance for the stock anthropic SDK on statuses its "
        "default policy would get wrong.",
    },
}
_DECLARED_RETRY_AFTER: dict[str, Any] = {
    "schema": {"type": "integer"},
    "description": "Seconds to wait before retrying (429 rate-limit and 503 capacity responses).",
}
_DECLARED_LOCATION: dict[str, Any] = {
    "schema": {"type": "string"},
    "description": "URL of the created job's status endpoint.",
}


def _declare_response_headers(app: FastAPI, rate_limited: bool) -> None:
    """Materialize the cached spec once and stamp the headers the middleware
    actually sets — generated clients inherit the contract instead of guessing."""
    spec = app.openapi()
    for path, item in spec.get("paths", {}).items():
        for op in item.values():
            if not isinstance(op, dict):
                continue
            for code, resp in op.get("responses", {}).items():
                if not isinstance(resp, dict):
                    continue
                hdrs = resp.setdefault("headers", {})
                hdrs.update(_DECLARED_COMMON_HEADERS)
                if _is_anthropic_path(path):
                    hdrs.update(_DECLARED_ANTHROPIC_HEADERS)
                if rate_limited:
                    hdrs.update(_DECLARED_RATELIMIT_HEADERS)
                if code in ("429", "503"):
                    hdrs.setdefault("Retry-After", _DECLARED_RETRY_AFTER)
            if op.get("operationId") == "submit_job":
                op["responses"]["202"].setdefault("headers", {})["Location"] = _DECLARED_LOCATION


# Canonical machine code for unambiguous statuses; ambiguous statuses
# (three different 503s, two 502s) carry an explicit ApiError code.
_STATUS_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "too_large",
    422: "validation",
    429: "too_many_requests",
    500: "internal",
    501: "not_implemented",
    503: "unavailable",
}


_MSG_CALLBACK_NEEDS_URL = "callback_secret requires callback_url"
_MSG_LIMIT_RANGE = "limit must be 1..100"


class ApiError(HTTPException):
    """HTTPException carrying a stable machine ``code``, surfaced in the
    ``{"detail", "code"}`` error envelope so clients switch on it instead
    of matching detail text."""

    def __init__(
        self,
        status_code: int,
        detail: Any,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(status_code, detail, headers=headers)
        self.code = code or _STATUS_CODES.get(status_code, "internal")


def _err_code(exc: StarletteHTTPException) -> str:
    if isinstance(exc, ApiError):
        return exc.code
    return _STATUS_CODES.get(exc.status_code, "internal")


def _v1_error_body(path: str, message: str, status: int, code: str) -> dict[str, Any]:
    """Error envelope for `/v1` paths — Anthropic's `{type: "error",
    error}` shape on the `/v1/messages` surface, OpenAI's `{error}` shape
    everywhere else (stock SDKs of either family read their own grammar).
    """
    if path == "/v1/messages" or path.startswith("/v1/messages/"):
        return anthropic_error_body(message, status)
    return openai_error_body(message, status, code)


def _bad_content_length_body(path: str) -> dict[str, Any]:
    """The unparseable ``Content-Length`` refusal body in the path's own
    grammar — ``{error}`` under ``/v1``, flat ``{detail, code}`` on the
    ``/harness`` dialect."""
    if is_openai_path(path):
        return _v1_error_body(path, "invalid content-length", 400, "bad_request")
    return {"detail": "invalid content-length", "code": "bad_request"}


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})
_MAX_BODY_BYTES = 1 << 20
_SINGLETON_REQUEST_HEADERS = frozenset(
    {
        b"authorization",
        b"content-length",
        b"content-type",
        b"host",
        b"idempotency-key",
        b"last-event-id",
        b"transfer-encoding",
        b"x-api-key",
        b"x-request-id",
        b"anthropic-version",
    }
)


def _ambiguous_request_headers(request: Request) -> str | None:
    """Reject security-sensitive duplicates before any API collapses them.

    Starlette's ``Headers.get`` selects the first duplicate while the
    OpenAI translator's dict comprehension selects the last.  Accepting
    both creates a proxy/application interpretation gap.  These headers
    are singletons in this API; list-valued standard headers remain
    untouched.
    """
    seen: set[bytes] = set()
    for raw_name, _raw_value in request.scope.get("headers", []):
        name = bytes(raw_name).lower()
        singleton = name in _SINGLETON_REQUEST_HEADERS or name.startswith(b"x-fx1-")
        if singleton and name in seen:
            return f"duplicate {name.decode('latin-1')} header"
        seen.add(name)
    if b"content-length" in seen and b"transfer-encoding" in seen:
        return "content-length and transfer-encoding must not be combined"
    if b"authorization" in seen and b"x-api-key" in seen:
        return "send exactly one authentication header"
    return None


def _request_header_error_body(path: str, message: str) -> dict[str, Any]:
    if is_openai_path(path):
        return _v1_error_body(path, message, 400, "bad_request")
    return {"detail": message, "code": "bad_request"}


async def _buffer_request_body(request: Request) -> tuple[int, bytes]:
    """Read the ASGI entity once without buffering beyond the body cap.

    ``Request.stream()`` can yield arbitrarily small chunks.  Keeping each
    chunk in a list therefore lets a one-megabyte request consume tens of
    megabytes of Python object overhead.  A bounded in-memory stream keeps the
    allocation proportional to the advertised cap, and an over-cap entity
    is refused as soon as the first excess bytes arrive instead of draining
    an attacker-controlled stream before answering.
    """
    size = 0
    body = io.BytesIO()
    async for chunk in request.stream():
        size += len(chunk)
        if size > _MAX_BODY_BYTES:
            request._body = b""  # noqa: SLF001 — no downstream consumer on refusal
            return size, b""
        body.write(chunk)
    buffered = body.getvalue()
    request._body = buffered  # noqa: SLF001 — preserve entity for downstream parsing
    return size, buffered


def _too_large_body(path: str) -> dict[str, Any]:
    detail = f"body exceeds {_MAX_BODY_BYTES}-byte cap"
    if is_openai_path(path):
        return _v1_error_body(path, detail, 413, "too_large")
    return {"detail": detail, "code": "too_large"}


async def _request_ingress_refusal(request: Request) -> JSONResponse | None:
    """Validate singleton/framing headers and cap the actual ASGI entity."""
    ambiguous = _ambiguous_request_headers(request)
    if ambiguous is not None:
        return JSONResponse(
            status_code=400,
            content=_request_header_error_body(request.url.path, ambiguous),
        )
    if request.method not in ("POST", "PUT", "PATCH", "DELETE"):
        return None
    declared = request.headers.get("content-length")
    length: int | None = None
    if declared is not None:
        if re.fullmatch(r"[0-9]+", declared) is None:
            return JSONResponse(
                status_code=400,
                content=_bad_content_length_body(request.url.path),
            )
        length = int(declared)
        if length > _MAX_BODY_BYTES:
            return JSONResponse(status_code=413, content=_too_large_body(request.url.path))
    actual_length, _body = await _buffer_request_body(request)
    if actual_length > _MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content=_too_large_body(request.url.path))
    if length is not None and actual_length != length:
        return JSONResponse(
            status_code=400,
            content=_request_header_error_body(
                request.url.path, "content-length does not match request body"
            ),
        )
    return None


# Async job results persist in the store until eviction — stdout/stderr
# are capped per field so one chatty command can't pin unbounded memory.
_JOB_RESULT_MAX_BYTES = 1 << 20


def _cap_job_text(text: str) -> tuple[str, bool]:
    """Bound a stored job field; truncate at the byte cap, flag honestly."""
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= _JOB_RESULT_MAX_BYTES:
        return text, False
    return raw[:_JOB_RESULT_MAX_BYTES].decode("utf-8", errors="ignore"), True


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(_Model):
    status: Literal["ok"] = "ok"
    service: str = "fx1-harness-api"
    version: str = __version__
    registered_commands: int
    backends: dict[str, bool]
    draining: bool = False


class HarnessCommandItem(_Model):
    name: str
    role: str
    argv: list[str]
    timeout_s: int
    description: str


class HarnessCommandListResponse(_Model):
    items: list[HarnessCommandItem]
    total: int


class HarnessRunRequest(_Model):
    command: str
    extra_args: list[str] = Field(default_factory=list, max_length=64)
    config: str | None = None
    callback_url: str | None = None
    # HMAC signing secret for the callback delivery — never echoed on the
    # job record (stored as a PrivateAttr, excluded from serialization).
    callback_secret: str | None = None
    # In-body dedup key: on the single-submit route the Idempotency-Key
    # header wins when both are present; on batch submit this is the only
    # channel. Excluded from the idempotency fingerprint (it's transport,
    # not payload semantics).
    idempotency_key: str | None = Field(default=None, max_length=_IDEM_KEY_MAX)

    @field_validator("callback_url")
    @classmethod
    def _callback_url_http(cls, v: str | None) -> str | None:
        """Webhook target must be a real http(s) URL — the job record is
        POSTed to it on every terminal transition."""
        return check_callback_url(v)

    @model_validator(mode="after")
    def _callback_secret_needs_url(self) -> HarnessRunRequest:
        if self.callback_secret is not None and not self.callback_url:
            raise ValueError(_MSG_CALLBACK_NEEDS_URL)
        return self


class HarnessRunResponse(_Model):
    command: str
    exit_code: int
    stdout: str
    stderr: str
    ok: bool
    timeout_s: int
    replayed: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False


class ChatMessage(_Model):
    """One harness message — plain turns carry role+content; agent turns
    may carry ``tool_calls`` (assistant) or answer a call as ``role: tool``
    with ``tool_call_id``. The shapes are fail-closed: a tool_call_id on
    a non-tool role, or tool_calls on a non-assistant turn, is a 422."""

    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    @model_validator(mode="after")
    def _tool_context_valid(self) -> ChatMessage:
        if self.content is None and self.tool_calls is None and self.role != "tool":
            raise ValueError("content is required")
        if self.role == "tool":
            if not self.tool_call_id or self.content is None:
                raise ValueError("role 'tool' requires content + tool_call_id")
        elif self.tool_call_id is not None:
            raise ValueError("tool_call_id belongs on role 'tool'")
        if self.tool_calls is not None:
            if self.role != "assistant":
                raise ValueError("tool_calls belongs on role 'assistant'")
            for j, call in enumerate(self.tool_calls):
                fn = call.get("function") if isinstance(call, dict) else None
                args = fn.get("arguments") if isinstance(fn, dict) else None
                if (
                    not isinstance(call, dict)
                    or not isinstance(call.get("id"), str)
                    or call.get("type") != "function"
                    or not isinstance(fn, dict)
                    or not isinstance(fn.get("name"), str)
                    or not isinstance(args, str)
                ):
                    raise ValueError(
                        f"messages[].tool_calls[{j}]: needs "
                        "{id, type: 'function', function: {name, arguments}}"
                    )
        return self


class BackendAttempt(_Model):
    """One link of a backend fallback chain: which name was tried and how
    it ended (``error_class`` carries the verdict on a failed link)."""

    backend: str
    ok: bool
    error_class: str | None = None
    latency_ms: float | None = None


def _fallback_chain_valid(
    backend: str,
    fallbacks: Sequence[str],
    checkpoint_dir: str | None,
    byok: ByokOverride | None,
) -> None:
    """Chain validation shared by the sync and batch request models."""
    if len(set(fallbacks)) != len(fallbacks) or backend in fallbacks:
        raise ValueError("fallbacks must be distinct and not repeat the primary backend")
    chain = {backend, *fallbacks}
    if byok is not None and "byok" not in chain:
        raise ValueError("a byok override applies only to a 'byok' link in the chain")
    if checkpoint_dir is not None and "local_fx1" not in chain:
        raise ValueError("checkpoint_dir applies only to a 'local_fx1' link in the chain")


def _sampling_extras_valid(
    stop: list[str] | None,
    logit_bias: dict[str, int] | None,
    metadata: dict[str, str] | None,
) -> None:
    """Caps shared by the sync/batch request models — pydantic can't
    range-check dict values, so stop/logit_bias/metadata limits live here."""
    if stop is not None and (len(stop) > 4 or any(not 1 <= len(s) <= 512 for s in stop)):
        raise ValueError("stop accepts ≤4 sequences of 1–512 chars")
    if logit_bias is not None:
        for k, v in logit_bias.items():
            try:
                int(k)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"logit_bias keys must be token ids, got {k!r}") from exc
            if not -100 <= v <= 100:
                raise ValueError(f"logit_bias[{k!r}]={v} outside [-100, 100]")
    if metadata is not None and (
        len(metadata) > 16 or any(len(k) > 64 or len(v) > 512 for k, v in metadata.items())
    ):
        raise ValueError("metadata accepts ≤16 pairs, keys ≤64 chars, values ≤512")


def _sampling_of(body: CompleteRequest | CompleteBatchRequest) -> SamplingParams:
    """Decode params declared on the request → the dataclass the backends
    take. The wire-facing flat fields are validated by pydantic; the
    resolved set is what the completion record seals."""
    return SamplingParams(
        temperature=body.temperature,
        top_p=body.top_p,
        max_tokens=body.max_tokens,
        seed=body.seed,
        stop=tuple(body.stop) if body.stop else None,
        presence_penalty=body.presence_penalty,
        frequency_penalty=body.frequency_penalty,
        logit_bias=body.logit_bias,
        reasoning_effort=body.reasoning_effort,
        service_tier=body.service_tier,
        prompt_cache_key=body.prompt_cache_key,
        prompt_cache_retention=body.prompt_cache_retention,
        verbosity=body.verbosity,
        user=body.user,
    )


class CompleteRequest(_Model):
    backend: Literal["hosted_k3", "local_fx1", "byok"]
    messages: list[ChatMessage] = Field(min_length=1, max_length=512)
    checkpoint_dir: str | None = Field(
        default=None, min_length=1, max_length=4096, pattern=r"^[^\x00]+$"
    )
    receipt_hashes: list[str] | None = None
    # Per-call credentials; only meaningful with backend="byok".
    byok: ByokOverride | None = None
    # Per-call backend deadline; beats each backend's env/config default.
    timeout_s: float | None = Field(default=None, gt=0, le=3600)
    # Ordered alternates tried after the primary — only on availability
    # faults (unconfigured / transport / circuit open). A gate refusal or
    # a client error aborts the request; a refusal is a verdict, not a
    # reason to spend another backend's capacity.
    fallbacks: list[Literal["hosted_k3", "local_fx1", "byok"]] = Field(
        default_factory=list, max_length=2
    )
    # Declared decode params — only fields set here reach the wire beyond
    # temperature (a provider that doesn't know ``seed`` never sees it);
    # the resolved set lands in the completion record as evidence of what
    # was sampled. Unset temperature defaults to 0 (eval runs stay
    # deterministic unless the caller opts out).
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0, le=262144)
    seed: int | None = Field(default=None, ge=0)
    # OpenAI-parity decode contract — ``stop`` truncates the completion at
    # the earliest match (harness-enforced, so stub/local backends honor it
    # too); penalties/logit_bias and the provider hints pass through
    # verbatim. ``user``/``metadata`` stamp the completion record for
    # caller-side attribution in the audit ledger.
    stop: list[str] | None = None
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    logit_bias: dict[str, int] | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None
    service_tier: Literal["auto", "default", "flex", "priority", "scale"] | None = None
    prompt_cache_key: str | None = Field(default=None, max_length=128)
    prompt_cache_retention: Literal["in-memory", "24h"] | None = None
    verbosity: Literal["low", "medium", "high"] | None = None
    user: str | None = Field(default=None, max_length=512)
    metadata: dict[str, str] | None = None
    # Internal-only registry attribution.  A private attribute keeps this
    # out of the public /harness/complete schema: callers must not be able
    # to forge the model name sealed into the completion record.
    _served_model: str | None = PrivateAttr(default=None)
    # Agent-loop tool context — the OpenAI tool-calling surface on the
    # harness route: function specs the model may call (verbatim
    # OpenAI-shaped dicts), the provider's call policy, and the
    # parallel-call flag. ``tool_choice`` accepts the three OpenAI
    # literals or {"type": "function", "function": {"name": ...}}; any
    # other shape is a 422. All three pass through to tool-capable
    # links; a link without the channel answers 501.
    tools: list[dict[str, Any]] | None = Field(default=None, max_length=128)
    tool_choice: Literal["none", "auto", "required"] | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    # Provider token-level scores — ``logprobs: true`` asks the backend
    # for its ``choices[].logprobs`` payload (echoed verbatim); the
    # ``top_logprobs`` cap rides the same wire field. Both need the
    # structured channel — a link without it answers 501.
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)

    @model_validator(mode="after")
    def _chain_valid(self) -> CompleteRequest:
        _fallback_chain_valid(self.backend, self.fallbacks, self.checkpoint_dir, self.byok)
        _sampling_extras_valid(self.stop, self.logit_bias, self.metadata)
        if self.tools is not None:
            if not self.tools:
                raise ValueError("tools must be a non-empty list when present")
            for k, tool in enumerate(self.tools):
                fn = tool.get("function") if isinstance(tool, dict) else None
                name = fn.get("name") if isinstance(fn, dict) else None
                if (
                    not isinstance(tool, dict)
                    or tool.get("type") != "function"
                    or not isinstance(fn, dict)
                    or not isinstance(name, str)
                    or not name
                    or len(name) > 64
                ):
                    raise ValueError(f"tools[{k}]: needs {{type: 'function', function: {{name}}}}")
        if isinstance(self.tool_choice, dict):
            fn = self.tool_choice.get("function")
            name = fn.get("name") if isinstance(fn, dict) else None
            if (
                self.tool_choice.get("type") != "function"
                or not isinstance(fn, dict)
                or not isinstance(name, str)
                or not name
            ):
                raise ValueError(
                    "tool_choice must be 'none'|'auto'|'required' or "
                    "{type: 'function', function: {name}}"
                )
        if not self.tools and (
            self.tool_choice is not None or self.parallel_tool_calls is not None
        ):
            raise ValueError("tool_choice/parallel_tool_calls require a non-empty tools list")
        if self.top_logprobs is not None and not self.logprobs:
            raise ValueError("top_logprobs requires logprobs: true")
        return self


class CompleteResponse(_Model):
    backend: str
    model: str | None
    content: str
    receipt_hashes: list[str]
    # wall-clock ms inside the backend call — fx-1 sees per-call cost without
    # timing middleware overhead; idempotent replays report the original call's
    # latency alongside ``replayed``.
    latency_ms: float
    # Token counts reported by the endpoint for THIS call (prompt_tokens /
    # completion_tokens / total_tokens where the provider supplies them);
    # None when the backend has no usage channel — never fabricated.
    usage: dict[str, int] | None = None
    # True when the response came from the Idempotency-Key cache — lets
    # fx-1 audit retried calls without paying for them twice.
    replayed: bool = False
    # Upstream-reported tool calls (None when the model answered with
    # text), the provider's own finish_reason, and the verbatim
    # ``choices[].logprobs`` payload when the request asked for it —
    # verbatim fields on structured-channel links; None on plain-text
    # turns.
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    logprobs: dict[str, Any] | None = None
    # Server-minted handle into the completion log
    # (``GET /harness/completions/{id}``) — replays keep the original id.
    completion_id: str | None = None
    # Ordered chain trace — every link tried (last is the serving link),
    # empty when the primary served unchallenged.
    attempts: list[BackendAttempt] = []
    # The resolved decode params actually sent to the provider.
    sampling: dict[str, Any] | None = None


class CompleteBatchRequest(_Model):
    backend: Literal["hosted_k3", "local_fx1", "byok"]
    batch: list[Annotated[list[ChatMessage], Field(min_length=1)]] = Field(
        min_length=1, max_length=64
    )
    checkpoint_dir: str | None = Field(
        default=None, min_length=1, max_length=4096, pattern=r"^[^\x00]+$"
    )
    receipt_hashes: list[str] | None = None
    byok: ByokOverride | None = None
    timeout_s: float | None = Field(default=None, gt=0, le=3600)
    max_workers: int = Field(default=4, ge=1, le=16)
    # Resolve-level chain: first resolvable link serves the whole batch
    # (one backend per batch — per-item failover can't attribute usage).
    fallbacks: list[Literal["hosted_k3", "local_fx1", "byok"]] = Field(
        default_factory=list, max_length=2
    )

    # Declared decode params applied to every item — same semantics as
    # ``CompleteRequest``; the resolved set is echoed on the response and
    # on each item's completion record.
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0, le=262144)
    seed: int | None = Field(default=None, ge=0)
    stop: list[str] | None = None
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    logit_bias: dict[str, int] | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None
    service_tier: Literal["auto", "default", "flex", "priority", "scale"] | None = None
    prompt_cache_key: str | None = Field(default=None, max_length=128)
    prompt_cache_retention: Literal["in-memory", "24h"] | None = None
    verbosity: Literal["low", "medium", "high"] | None = None
    user: str | None = Field(default=None, max_length=512)
    metadata: dict[str, str] | None = None

    @model_validator(mode="after")
    def _chain_valid(self) -> CompleteBatchRequest:
        _fallback_chain_valid(self.backend, self.fallbacks, self.checkpoint_dir, self.byok)
        _sampling_extras_valid(self.stop, self.logit_bias, self.metadata)
        return self


class EmbedRequest(_Model):
    """``/v1/embeddings`` resolved-call record — the link fields plus the
    provider-forwarded embedding params (``embeddings_to_kwargs`` output
    lands here so the chain contract stays fail-closed)."""

    backend: Literal["hosted_k3", "local_fx1", "byok"]
    fallbacks: list[Literal["hosted_k3", "local_fx1", "byok"]] = Field(
        default_factory=list, max_length=2
    )
    checkpoint_dir: str | None = Field(
        default=None, min_length=1, max_length=4096, pattern=r"^[^\x00]+$"
    )
    byok: ByokOverride | None = None
    timeout_s: float | None = Field(default=None, gt=0, le=3600)
    model: str = Field(min_length=1, max_length=256)
    input: str | list[str] | list[int] | list[list[int]]
    encoding_format: Literal["float", "base64"] | None = None
    dimensions: int | None = Field(default=None, ge=1)
    user: str | None = Field(default=None, max_length=512)

    @model_validator(mode="after")
    def _chain_valid(self) -> EmbedRequest:
        _fallback_chain_valid(self.backend, self.fallbacks, self.checkpoint_dir, self.byok)
        return self


class BackendProbeRequest(_Model):
    """Optional controls for a backend liveness probe — same credential
    plumbing as a completion, so a BYOK probe tests the caller's real
    endpoint. ``prompt`` defaults to a one-token ping."""

    checkpoint_dir: str | None = None
    byok: ByokOverride | None = None
    timeout_s: float | None = Field(default=None, gt=0, le=3600)
    prompt: str = Field(default="ping", min_length=1, max_length=256)


class BackendProbeResponse(_Model):
    """Deep-health verdict for one backend — whether a real minimal
    completion succeeded, how long it took, and why not when it didn't."""

    backend: str
    ok: bool
    model: str | None = None
    latency_ms: float
    error: str | None = None
    error_class: str | None = None


class GateCheckRequest(_Model):
    """Text to run through the honesty gate — pre-flight for writers
    before they spend model tokens (or for validators on the way out)."""

    text: str = Field(min_length=0, max_length=262144)


class GateCheckResponse(_Model):
    """Gate verdict — ``ok`` mirrors whether the text would pass the gate;
    ``error`` carries the refusal reason when it wouldn't."""

    ok: bool
    error: str | None = None


class ScoreRequest(_Model):
    """Text to run through the reward contract — the full deterministic
    breakdown (components, violations, total) for writers preflighting a
    response or validators auditing one. A str is one input; a list scores
    each element independently (cap 128)."""

    input: str | list[str]

    @model_validator(mode="after")
    def _input_shape(self) -> ScoreRequest:
        raw = self.input
        if isinstance(raw, str):
            if len(raw) > 262144:
                raise ValueError("input strings must be at most 262144 characters")
            return self
        if not isinstance(raw, list) or len(raw) == 0 or len(raw) > 128:
            raise ValueError("input must be a non-empty list of at most 128 items")
        if not all(isinstance(v, str) for v in raw):
            raise ValueError("input must be a string or a list of strings")
        if any(len(v) > 262144 for v in raw):
            raise ValueError("input strings must be at most 262144 characters")
        return self


class ScoreItem(_Model):
    """One scored input — the reward contract's verdict verbatim."""

    object: Literal["score"] = "score"
    index: int
    total: float
    components: dict[str, float]
    violations: list[str]


class ScoreResponse(_Model):
    object: Literal["list"]
    data: list[ScoreItem]


class ModerationRequest(ScoreRequest):
    """OpenAI-compatible moderation request — ``input`` is one string or a
    list (same caps as ``/harness/score``); ``model`` is accepted for wire
    compatibility and reported back as the gate's canonical name."""

    model: str | None = None


class ModerationResult(_Model):
    """One input's moderation verdict. ``category_scores`` are deterministic
    0.0/1.0 — the gate is a lexical contract, not a learned classifier, so
    scores carry the verdict, not a confidence."""

    flagged: bool
    categories: dict[str, bool]
    category_scores: dict[str, float]
    category_applied_input_types: dict[str, list[str]]


class ModerationResponse(_Model):
    id: str
    model: str
    results: list[ModerationResult]


class EvalSubmitRequest(_Model):
    """Async eval submission: one seeded suite against a backend chain.

    The suite runs on the shared jobs executor under the same drain/cap
    contract as run jobs; poll ``GET /harness/evals/{eval_id}`` and export
    the sealed ``fx1_eval_record.v1`` doc at ``/receipt`` once terminal.
    Evals always run under the decode pin ``{"temperature": 0.0}`` — eval
    evidence is deterministic evidence; the pin is recorded on the
    record.
    """

    suite: EvalSuiteName
    backend: Literal["hosted_k3", "local_fx1", "byok"]
    seed: int = Field(default=0, ge=0)
    checkpoint_dir: str | None = None
    byok: ByokOverride | None = None
    timeout_s: float | None = Field(default=None, gt=0, le=3600)
    fallbacks: list[Literal["hosted_k3", "local_fx1", "byok"]] = Field(
        default_factory=list, max_length=2
    )
    # Optional grader link — only the judge suites (capability, ext_bench)
    # consume it. Resolved once, unchained; ``judge_byok`` binds only a
    # 'byok' judge.
    judge_backend: Literal["hosted_k3", "local_fx1", "byok"] | None = None
    judge_byok: ByokOverride | None = None
    # Terminal-state webhook (the job contract): the finished record is
    # POSTed to ``callback_url`` on every terminal transition;
    # ``callback_secret`` HMAC-signs the delivery and is never echoed.
    callback_url: str | None = None
    callback_secret: str | None = None

    @field_validator("callback_url")
    @classmethod
    def _eval_callback_url_http(cls, v: str | None) -> str | None:
        return check_callback_url(v)

    @model_validator(mode="after")
    def _eval_valid(self) -> EvalSubmitRequest:
        _fallback_chain_valid(self.backend, self.fallbacks, self.checkpoint_dir, self.byok)
        if self.judge_byok is not None and self.judge_backend != "byok":
            raise ValueError("judge_byok applies only to judge_backend='byok'")
        if self.judge_backend is not None and not suite_accepts_judge(self.suite):
            raise ValueError(f"suite '{self.suite}' takes no judge")
        if self.callback_secret is not None and not self.callback_url:
            raise ValueError(_MSG_CALLBACK_NEEDS_URL)
        return self


class EvalSubmitResponse(_Model):
    """Submission ack — ``replayed`` marks an Idempotency-Key hit (the
    eval ran once already; no second execution)."""

    eval_id: str
    status: str
    replayed: bool = False


class EvalListResponse(_Model):
    """Eval inventory page: ``total`` is the filtered count before paging."""

    records: list[EvalRecord]
    total: int


# ---- /v1/evals — the OpenAI Evals-shaped spec/run surface -------------------
#
# OpenAI's evals API separates the eval (a named container declaring the
# datasource shape + grading criteria) from its runs (executions against
# a model). ``data_source_config.item_schema`` is the harness twin of the
# submission knobs — validated fail-closed at spec create so a bad spec
# can never exist; credentials never live on a spec (BYOK blocks attach
# to the run body only — the spec is stored state and must stay
# export-safe).


class EvalSpecCriterion(_Model):
    """One declared testing criterion — grader kwargs ride ``extra`` (the
    suite's own graders are the measurements; criteria are declarative)."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1, max_length=255)
    type: str | None = None


class EvalSpecDataSource(_Model):
    """``data_source_config`` — custom type + the validated item_schema."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["custom"]
    item_schema: EvalSpecItemSchema
    include_sample_schema: bool | None = None


class EvalSpecCreate(_Model):
    """``POST /v1/evals`` body — name + config + criteria + metadata."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    data_source_config: EvalSpecDataSource
    testing_criteria: list[EvalSpecCriterion] = Field(default_factory=list, max_length=64)
    metadata: dict[str, str] | None = None


class EvalSpecUpdate(_Model):
    """``POST /v1/evals/{id}`` — metadata/name edits plus the hot-reload
    fields: ``data_source_config``/``testing_criteria`` replace the spec's
    declared shape ONLY while no run binds it — once a run exists the
    shape is evidence and the update answers 409 ``eval_spec_frozen``."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=255)
    metadata: dict[str, str] | None = None
    data_source_config: EvalSpecDataSource | None = None
    testing_criteria: list[EvalSpecCriterion] | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def _some_field(self) -> EvalSpecUpdate:
        if (
            self.name is None
            and self.metadata is None
            and self.data_source_config is None
            and self.testing_criteria is None
        ):
            raise ValueError("update must carry a field")
        return self


class EvalSpecWire(_Model):
    """The OpenAI ``eval`` object."""

    model_config = ConfigDict(extra="forbid")

    id: str
    object: Literal["eval"] = "eval"
    name: str
    data_source_config: dict[str, Any]
    testing_criteria: list[dict[str, Any]]
    metadata: dict[str, str]
    created_at: int


class EvalSpecPage(_Model):
    """``{object:'list', data:[eval], has_more}`` — the OpenAI list shape."""

    model_config = ConfigDict(extra="forbid")

    object: Literal["list"] = "list"
    data: list[EvalSpecWire]
    has_more: bool


class EvalSpecDeleted(_Model):
    model_config = ConfigDict(extra="forbid")

    id: str
    object: Literal["eval.deleted"] = "eval.deleted"
    deleted: Literal[True] = True


class EvalRunDataSource(_Model):
    """Optional run-time overrides on the spec's item_schema."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["custom"] = "custom"
    source: dict[str, Any] | None = None


class EvalRunCreate(_Model):
    """``POST /v1/evals/{id}/runs`` — ``model`` is the eval target: a
    backend link name (``hosted_k3``/``local_fx1``/``byok``), the base
    ``fx1``, or a registered ``ft:`` name (resolves to local_fx1 at the
    card's checkpoint — unregistered names fail closed). ``byok``
    carries the credentials for a byok link — never stored on the spec
    or record."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=512)
    data_source: EvalRunDataSource | None = None
    byok: ByokOverride | None = None
    judge_byok: ByokOverride | None = None
    metadata: dict[str, str] | None = None
    callback_url: str | None = None
    callback_secret: str | None = None

    @field_validator("callback_url")
    @classmethod
    def _run_callback_url_http(cls, v: str | None) -> str | None:
        return check_callback_url(v)

    @model_validator(mode="after")
    def _run_valid(self) -> EvalRunCreate:
        if self.callback_secret is not None and not self.callback_url:
            raise ValueError(_MSG_CALLBACK_NEEDS_URL)
        return self


class EvalRunCounts(_Model):
    model_config = ConfigDict(extra="forbid")

    total: int
    passed: int
    failed: int
    errored: int


class EvalRunError(_Model):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str


class EvalRunObject(_Model):
    """The OpenAI ``eval.run`` object over the record."""

    model_config = ConfigDict(extra="forbid")

    id: str
    object: Literal["eval.run"] = "eval.run"
    eval_id: str | None
    model: str
    status: str
    created_at: int
    suite: str
    seed: int
    backend: str
    result_counts: EvalRunCounts | None = None
    per_testing_criteria_results: list[dict[str, Any]] = []
    error: EvalRunError | None = None
    receipt_url: str


class EvalRunPage(_Model):
    model_config = ConfigDict(extra="forbid")

    object: Literal["list"] = "list"
    data: list[EvalRunObject]
    has_more: bool


class EvalOutputItem(_Model):
    """One per-task verdict row — ``datasource_item`` is the suite's raw
    report row, served verbatim."""

    model_config = ConfigDict(extra="forbid")

    id: str
    object: Literal["eval.run.output_item"] = "eval.run.output_item"
    run_id: str
    eval_id: str | None
    created_at: int
    status: Literal["pass", "fail"]
    datasource_item_id: str
    datasource_item: dict[str, Any]
    results: list[dict[str, Any]]


class EvalOutputItemPage(_Model):
    model_config = ConfigDict(extra="forbid")

    object: Literal["list"] = "list"
    data: list[EvalOutputItem]
    has_more: bool
    first_id: str | None = None
    last_id: str | None = None


class EvalRunDeleted(_Model):
    model_config = ConfigDict(extra="forbid")

    id: str
    object: Literal["eval.run.deleted"] = "eval.run.deleted"
    deleted: Literal[True] = True


class CompleteBatchItem(_Model):
    ok: bool
    latency_ms: float
    content: str | None = None
    error: str | None = None
    error_class: str | None = None
    completion_id: str | None = None


class CompleteBatchResponse(_Model):
    backend: str
    model: str | None
    receipt_hashes: list[str]
    results: list[CompleteBatchItem]
    # The resolved decode params sent for every item in the batch.
    sampling: dict[str, Any] | None = None
    # Sum of the backend's reported usage across this batch (a shared
    # endpoint can't attribute counts per item under worker threads —
    # only the batch-level delta is honest). None when the backend is
    # silent on usage.
    usage_total: dict[str, int] | None = None
    replayed: bool = False
    # Resolve-level chain trace — the link that served the batch.
    attempts: list[BackendAttempt] = []


class CompletionRecord(_Model):
    """One recorded model call: hashes of what went in and came out,
    latency, reported usage, and the verdict — the harness's own calls
    are auditable evidence. Content itself is never stored (hashes only),
    and the log is a bounded in-process ring."""

    completion_id: str
    backend: str
    model: str | None = None
    ok: bool
    latency_ms: float
    at: float
    usage: dict[str, int] | None = None
    error: str | None = None
    error_class: str | None = None
    prompt_sha256: str
    output_sha256: str | None = None
    # Chain trace when a fallback chain ran — sealed evidence of failover.
    attempts: list[BackendAttempt] | None = None
    # The resolved decode params sent to the provider.
    sampling: dict[str, Any] | None = None
    # Caller-side attribution — the request's ``user`` tag / ``metadata``
    # pairs, when declared. Evidence fields, never returned to the model.
    user: str | None = None
    metadata: dict[str, str] | None = None
    # Which credential made the call: ``"env"`` for the bootstrap
    # FX1_API_KEY, the managed key's sha256 fingerprint, or ``None`` on
    # the unauthenticated loopback-dev surface. A fingerprint is not a
    # secret — it cannot authenticate.
    key_id: str | None = None


class CompletionListResponse(_Model):
    items: list[CompletionRecord]
    count: int


class ApiKeyCreateRequest(_Model):
    name: str | None = Field(default=None, max_length=128)
    # admin keys may themselves manage keys — the bootstrap credential
    # mints the first admin key so an env-key-less deployment keeps a
    # control plane after provisioning turns auth on.
    admin: bool = False
    # Least-privilege scopes: "read" (safe methods), "write" (data-plane
    # mutations), "admin" (key management + drain). Unset keeps the
    # pre-scope contract [read, write] (+admin for admin keys); the
    # admin flag unions its scope onto an explicit list.
    scopes: list[str] | None = None
    # Declared per-key policy, journaled at mint: rpm bounds the key to
    # a fixed 60 s request window (over-limit answers 429 + Retry-After);
    # ttl_s bakes an expiry — a dead credential fails closed like a
    # revoked one.
    rpm: int | None = Field(default=None, ge=1, le=1_000_000)
    ttl_s: float | None = Field(default=None, gt=0, le=315_576_000)
    # Hard budgets, declared at mint: max_requests counts authenticated
    # calls, max_tokens counts provider-reported usage charged after each
    # served response. An exhausted key answers 429 quota_exceeded.
    max_requests: int | None = Field(default=None, ge=1, le=2_147_483_647)
    max_tokens: int | None = Field(default=None, ge=1, le=9_223_372_036_854_775_807)


class ApiKeyMintResponse(_Model):
    """Mint response — the only place the raw key ever appears."""

    id: str
    object: Literal["key"] = "key"
    name: str | None
    prefix: str
    admin: bool
    scopes: list[str]
    rpm: int | None
    max_requests: int | None
    max_tokens: int | None
    expires_at: float | None
    created_at: float
    tokens_used: int
    rotated_from: str | None
    key: str


class ApiKeyRecordModel(_Model):
    """The wire view of a managed key — fingerprint + metadata only;
    the sha256 and raw secret never leave the store."""

    id: str
    object: Literal["key"] = "key"
    name: str | None
    prefix: str
    admin: bool
    scopes: list[str]
    rpm: int | None
    max_requests: int | None
    max_tokens: int | None
    expires_at: float | None
    created_at: float
    enabled: bool
    revoked_at: float | None
    uses: int
    tokens_used: int
    last_used_at: float | None
    rotated_from: str | None


class ApiKeyListResponse(_Model):
    object: Literal["list"] = "list"
    data: list[ApiKeyRecordModel]


class ApiKeyRotateRequest(_Model):
    """Rotate body — every field optional; omitted fields inherit the
    predecessor's declared policy verbatim."""

    name: str | None = None
    ttl_s: float | None = Field(default=None, gt=0)
    revoke_old: bool = True


class ApiKeyRotateResponse(_Model):
    """Rotation response — ``key`` is the minted successor (raw secret
    shown once); ``revoked_previous`` reports whether the predecessor
    was tombstoned atomically with the mint."""

    object: Literal["key_rotation"] = "key_rotation"
    key: ApiKeyMintResponse
    rotated_from: str
    revoked_previous: bool


class ApiKeyPatchRequest(_Model):
    """Patch body — every field optional; the three states are
    distinct: omitted keeps the declared policy, explicit ``null``
    clears a nullable bound (``name``/``rpm``/``max_requests``/
    ``max_tokens``/``expires_at`` — the unbounded default), and a
    concrete value replaces it. ``scopes``/``admin`` take concrete
    values when sent (``null`` clears nothing there — an explicit
    list or flag instead). ``enabled`` and the live counters are
    never patchable — revocation is permanent."""

    name: str | None = Field(default=None, max_length=128)
    rpm: int | None = Field(default=None, ge=1, le=1_000_000)
    scopes: list[str] | None = None
    admin: bool | None = None
    max_requests: int | None = Field(default=None, ge=1, le=2_147_483_647)
    max_tokens: int | None = Field(default=None, ge=1, le=9_223_372_036_854_775_807)
    expires_at: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _concrete_when_sent(self) -> ApiKeyPatchRequest:
        for field in ("scopes", "admin"):
            if field in self.model_fields_set and getattr(self, field) is None:
                raise ValueError(f"{field} must take a concrete value when sent")
        return self


class KeyUsageBackendSplit(_Model):
    calls: int = 0
    total_tokens: int = 0


class KeyServedUsage(_Model):
    """Served-call aggregation over the completion ring for one
    credential fingerprint — a bounded window: ``log_dropped`` on the
    parent card marks when these totals are a lower bound on lifetime
    spend, not the full record."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    by_backend: dict[str, KeyUsageBackendSplit] = {}


class ApiKeyUsageResponse(_Model):
    """Usage card for one managed key: lifetime ``uses`` and ``tokens_used``
    survive a clean ``--state-dir`` restart; RPM windows remain process-local.
    Includes declared budgets, derived headroom, and completion-ring spend."""

    id: str
    object: Literal["key_usage"] = "key_usage"
    name: str | None
    admin: bool
    enabled: bool
    created_at: float
    expires_at: float | None
    revoked_at: float | None
    rotated_from: str | None
    uses: int
    tokens_used: int
    last_used_at: float | None
    max_requests: int | None
    requests_remaining: int | None
    max_tokens: int | None
    tokens_remaining: int | None
    rpm: int | None
    window_remaining: int | None
    window_reset_s: int | None
    served: KeyServedUsage
    log_cap: int
    log_dropped: int


class SelfUsageResponse(_Model):
    """The calling credential's own card — read-scope self-introspection
    so a key holder watches its own budgets without admin. ``env`` (the
    bootstrap credential) and ``none`` (loopback dev) are unmetered
    roots; ``managed`` embeds the full usage card."""

    object: Literal["self_usage"] = "self_usage"
    credential: Literal["managed", "env", "none"]
    scopes: list[str]
    metered: bool
    key: ApiKeyUsageResponse | None


def _key_wire(rec: dict[str, Any]) -> ApiKeyRecordModel:
    return ApiKeyRecordModel(
        id=rec["key_id"],
        name=rec["name"],
        prefix=rec["prefix"],
        admin=bool(rec.get("admin")),
        scopes=list(
            rec.get("scopes")
            or (["read", "write", "admin"] if rec.get("admin") else ["read", "write"])
        ),
        rpm=rec.get("rpm"),
        max_requests=rec.get("max_requests"),
        max_tokens=rec.get("max_tokens"),
        expires_at=rec.get("expires_at"),
        created_at=rec["created_at"],
        enabled=rec["enabled"],
        revoked_at=rec["revoked_at"],
        uses=rec["uses"],
        tokens_used=int(rec.get("tokens_used") or 0),
        last_used_at=rec["last_used_at"],
        rotated_from=rec.get("rotated_from"),
    )


def _key_mint_wire(rec: dict[str, Any], raw: str) -> ApiKeyMintResponse:
    """Mint/rotate response — the record plus the raw secret, shown once."""
    return ApiKeyMintResponse(
        id=rec["key_id"],
        name=rec["name"],
        prefix=rec["prefix"],
        admin=bool(rec.get("admin")),
        scopes=list(rec["scopes"]),
        rpm=rec.get("rpm"),
        max_requests=rec.get("max_requests"),
        max_tokens=rec.get("max_tokens"),
        expires_at=rec.get("expires_at"),
        created_at=rec["created_at"],
        tokens_used=int(rec.get("tokens_used") or 0),
        rotated_from=rec.get("rotated_from"),
        key=raw,
    )


class ReceiptIndexItem(_Model):
    sha256: str
    name: str


class ReceiptIndexResponse(_Model):
    items: list[ReceiptIndexItem]
    count: int


class ReceiptVerifyRequest(_Model):
    receipt: dict[str, Any]


class ReceiptVerifyResponse(_Model):
    valid: bool
    path: str
    schema_tag: str
    kind: str | None
    verdict: str | None
    digest_convention: str | None
    errors: list[str]
    warnings: list[str]


class ReceiptVerifyBatchRequest(_Model):
    receipts: list[dict[str, Any]] = Field(min_length=1, max_length=_VERIFY_BATCH_MAX)


class ReceiptVerifyBatchItem(ReceiptVerifyResponse):
    """One per-item verdict: the same payload the single-verify route
    returns, plus the index so callers can map results back."""

    index: int


class ReceiptVerifyBatchResponse(_Model):
    verified: int
    failed: int
    results: list[ReceiptVerifyBatchItem]


class BackendCompletionStats(_Model):
    """Per-backend completion accounting: outcome counts, a cumulative
    latency histogram (edges are ``_LAT_BUCKETS_MS`` plus ``+Inf``), and
    backend-reported token sums. ``usage_calls`` counts calls that carried
    a usage dict — a silent provider shows zero tokens AND zero calls."""

    ok: int = 0
    error: int = 0
    latency_count: int = 0
    latency_sum_ms: float = 0.0
    latency_buckets: dict[str, int] = {}
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    usage_calls: int = 0


class MetricsResponse(_Model):
    """Point-in-time ops snapshot: totals since process start."""

    uptime_s: float
    requests_total: int
    errors_total: int
    by_status: dict[str, int]
    inflight: int
    inflight_watermark: int
    max_inflight: int
    draining: bool
    rate_limited_total: int = 0
    # completion calls only — per backend name as requested on the wire
    complete: dict[str, BackendCompletionStats] = {}


class DrainResponse(_Model):
    """Result of latching drain mode: state + live in-flight count."""

    draining: bool
    inflight: int
    drained: bool = False


class ReadyResponse(_Model):
    """Readiness probe payload — only emitted while accepting work."""

    ready: Literal[True] = True
    inflight: int


class VersionResponse(_Model):
    """Wire-contract + package versions — the client's negotiation payload."""

    api_version: str
    fx1_version: str


class CapabilitiesResponse(_Model):
    """Self-describing discovery payload: which wire features this build
    serves and the operational limits in effect — clients self-configure
    (batch sizes, retry budgets, stream use) from one call instead of
    hardcoding server internals."""

    api_version: str
    fx1_version: str
    features: dict[str, bool]
    eval_suites: list[str]
    limits: dict[str, float]
    backends: dict[str, bool]
    roles: list[str]


class BackendProbeVerdict(_Model):
    """The most recent ``/harness/backends/{name}/probe`` outcome — kept
    process-local so a monitoring scrape can read the last deep-health
    verdict without spending another live call."""

    ok: bool
    latency_ms: float
    checked_at: float
    error_class: str | None = None


class BackendStatusEntry(_Model):
    """One backend's liveness surface: whether it is configured and, when
    the circuit breaker is enabled, whether it is currently fast-failing."""

    configured: bool
    circuit_open: bool
    cooldown_remaining_s: float
    consecutive_failures: int
    last_probe: BackendProbeVerdict | None = None


_JOB_STATUSES = ("queued", "running", "succeeded", "failed", "cancelled")
_TERMINAL_JOB_STATUS = frozenset({"succeeded", "failed", "cancelled"})
_JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class JobSubmitResponse(_Model):
    job_id: str
    status: _JobStatus
    replayed: bool = False


class JobStatusResponse(_Model):
    """Live job record; ``result`` appears only once status is terminal."""

    job_id: str
    status: _JobStatus
    created_at: float
    finished_at: float | None
    result: HarnessRunResponse | None
    error: str | None
    callback_url: str | None = None
    callback_status: Literal["delivered", "failed"] | None = None
    _callback_secret: str | None = PrivateAttr(default=None)
    _callback_fired: bool = PrivateAttr(default=False)
    _callback_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    callback_error: str | None = None
    callback_attempts: int = 0


_LAT_BUCKETS_MS = (100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 30000.0, 60000.0)


class _Metrics:
    """Request counters + inflight gauge + per-backend completion
    histograms, shared via app.state."""

    def __init__(self, max_inflight: int) -> None:
        self.started = time.monotonic()
        self.max_inflight = max_inflight
        self._cond = threading.Condition()
        self._requests_total = 0
        self._errors_total = 0
        self._by_status: dict[int, int] = {}
        self._inflight = 0
        self._watermark = 0
        self._rate_limited_total = 0
        # backend -> {"ok","error","count","sum","buckets" (cumulative),
        #              "*_tokens","usage_calls"}
        self._complete: dict[str, dict[str, Any]] = {}
        self.draining = threading.Event()

    def record_complete(
        self,
        backend: str,
        ok: bool,
        latency_ms: float,
        usage: dict[str, int] | None = None,
    ) -> None:
        """One attempted model call: outcome + latency into the backend's
        cumulative histogram, plus backend-reported token sums when the
        endpoint supplies them. Called only when the backend was invoked —
        breaker rejections and pre-call validation aren't model work.
        ``usage_calls`` counts calls that reported usage at all, so a
        silent provider is distinguishable from a zero bill."""
        with self._cond:
            st = self._complete.setdefault(
                backend,
                {
                    "ok": 0,
                    "error": 0,
                    "count": 0,
                    "sum": 0.0,
                    # one entry per edge + a final +Inf slot
                    "buckets": [0] * (len(_LAT_BUCKETS_MS) + 1),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "usage_calls": 0,
                },
            )
            st["ok" if ok else "error"] += 1
            st["count"] += 1
            st["sum"] += latency_ms
            for i, edge in enumerate(_LAT_BUCKETS_MS):
                if latency_ms <= edge:
                    st["buckets"][i] += 1
            st["buckets"][-1] += 1
            if usage:
                st["usage_calls"] += 1
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    v = usage.get(k)
                    if isinstance(v, int) and not isinstance(v, bool):
                        st[k] += v

    def record_rate_limited(self) -> None:
        with self._cond:
            self._rate_limited_total += 1

    def record(self, status: int) -> None:
        with self._cond:
            self._requests_total += 1
            self._by_status[status] = self._by_status.get(status, 0) + 1
            if status >= 400:
                self._errors_total += 1

    def acquire(self) -> None:
        with self._cond:
            self._inflight += 1
            self._watermark = max(self._watermark, self._inflight)

    def release(self) -> None:
        with self._cond:
            self._inflight -= 1
            self._cond.notify_all()

    def wait_idle(self, timeout_s: float) -> bool:
        """Block until inflight reaches zero or the timeout lapses — the
        drain-lifecycle wait channel. True when the pool actually emptied."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while self._inflight > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return True

    def snapshot(self) -> MetricsResponse:
        with self._cond:
            return MetricsResponse(
                uptime_s=round(time.monotonic() - self.started, 3),
                requests_total=self._requests_total,
                errors_total=self._errors_total,
                by_status={str(k): v for k, v in sorted(self._by_status.items())},
                inflight=self._inflight,
                inflight_watermark=self._watermark,
                max_inflight=self.max_inflight,
                draining=self.draining.is_set(),
                rate_limited_total=self._rate_limited_total,
                complete={
                    b: BackendCompletionStats(
                        ok=st["ok"],
                        error=st["error"],
                        latency_count=st["count"],
                        latency_sum_ms=round(st["sum"], 3),
                        latency_buckets={
                            **_lat_bucket_labels(st["buckets"]),
                        },
                        prompt_tokens=st["prompt_tokens"],
                        completion_tokens=st["completion_tokens"],
                        total_tokens=st["total_tokens"],
                        usage_calls=st["usage_calls"],
                    )
                    for b, st in sorted(self._complete.items())
                },
            )


def _lat_bucket_labels(counts: list[int]) -> dict[str, int]:
    """Cumulative bucket counts keyed by le-label, ``+Inf`` last — the
    Prometheus histogram convention, kept identical on the JSON snapshot."""
    labels = [f"{edge:g}" for edge in _LAT_BUCKETS_MS] + ["+Inf"]
    return dict(zip(labels, counts, strict=True))


def _env_int_bound(name: str, default: int, given: int | None) -> int:
    """Positive-int tunable from arg or env, fail-closed below 1."""
    v = int(os.environ.get(name, str(default))) if given is None else given
    if v < 1:
        raise ValueError(f"{name} bound must be >= 1, got {v}")
    return v


def _env_int_floor(name: str, default: int, given: int | None) -> int:
    """Non-negative-int tunable from arg or env; 0 disables the feature."""
    v = int(os.environ.get(name, str(default))) if given is None else given
    if v < 0:
        raise ValueError(f"{name} bound must be >= 0, got {v}")
    return v


def _env_float_floor(name: str, default: float, given: float | None) -> float:
    """Non-negative float tunable from arg or env, fail-closed below 0."""
    v = float(os.environ.get(name, str(default))) if given is None else given
    if v < 0:
        raise ValueError(f"{name} must be >= 0, got {v}")
    return v


_IDEM_SEMANTIC_HEADERS = (
    "x-fx1-backend",
    "x-fx1-fallbacks",
    "x-fx1-checkpoint-dir",
    "x-fx1-byok-base-url",
    "x-fx1-byok-api-key",
    "x-fx1-byok-model",
    "x-fx1-timeout",
    "x-fx1-receipt-hashes",
)


def _body_fp(
    body: BaseModel,
    *,
    exclude: set[str] | None = None,
    headers: Mapping[str, str] | None = None,
) -> str:
    """Fingerprint for the idempotency contract — hashed so a stored
    dedupe record can never carry a secret (per-request BYOK keys).

    Backend-selection headers affect the executed request just as their
    body-level ``fx1`` twins do. Bind their values into the digest so a
    caller cannot change a BYOK destination or credential and silently
    receive the first upstream's replay. The raw values are never stored.
    Preserve the historical body-only digest when none are present.
    """
    serialized = body.model_dump_json(exclude=exclude).encode()
    if headers is not None:
        selected = {
            name: value
            for name in _IDEM_SEMANTIC_HEADERS
            if (value := headers.get(name)) is not None
        }
        if selected:
            serialized += (
                b"\0" + json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
            )
    return hashlib.sha256(serialized).hexdigest()


def _responses_sse(
    body: OpenAIResponseRequest,
    *,
    content: str,
    rid: str,
    item_id: str,
    model: str | None,
    usage: dict[str, int] | None,
    created: int | None = None,
    skip: int = 0,
    call_items: list[dict[str, Any]] | None = None,
    search_items: list[dict[str, Any]] | None = None,
    logprobs: list[dict[str, Any]] | None = None,
    final_status: str = "completed",
    incomplete_details: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Serialize ``openai_response_events`` into SSE frames —
    ``event:`` + ``id:`` + ``data:`` per frame, ``id`` equal to the
    frame index so ``Last-Event-ID`` resume works identically to the
    chat-completions stream. No ``[DONE]`` marker — ``response.completed``
    (or ``response.incomplete`` on a truncated turn) is the terminal
    event."""

    def _frame(event: str, payload: dict[str, Any], seq: int) -> str:
        return f"event: {event}\nid: {seq}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"

    for seq, (event, payload) in enumerate(
        openai_response_events(
            text=content,
            rid=rid,
            item_id=item_id,
            body=body,
            model=model,
            usage=usage,
            created=created,
            call_items=call_items,
            search_items=search_items,
            logprobs=logprobs,
            final_status=final_status,
            incomplete_details=incomplete_details,
        )
    ):
        if seq >= skip:
            yield _frame(event, payload, seq)


# ``timeout_s``'s ceiling — the route clamps it (<=3600) and the generator
# clamps again locally so the follow loop's trip count is statically
# bounded regardless of how it is called
_REPLAY_FOLLOW_TIMEOUT_MAX_S = 3600.0
_REPLAY_MIN_POLL_S = 0.01
_REPLAY_FOLLOW_MAX_ITERS = int(_REPLAY_FOLLOW_TIMEOUT_MAX_S / _REPLAY_MIN_POLL_S) + 1


def _responses_replay_frames(
    envelope_store: OpenAIEnvelopeStore,
    response_id: str,
    *,
    skip: int = 0,
    timeout_s: float = 600.0,
    keepalive_s: float = 15.0,
) -> Iterator[str]:
    """The ``GET /v1/responses/{id}?stream=true`` frame generator — the
    replay grammar (:func:`openai_response_replay_events`) serialized with
    the same ``event:``/``id:``/``data:`` shape as the create stream, so a
    re-attaching client's SSE parser rebuilds the same typed ``Response``.

    A terminal envelope emits the full recorded sequence (``skip`` =
    ``starting_after + 1`` resumes past a sequence cursor — the cursor is
    the frame's ``id:``, monotonically increasing by absolute index).
    A still-``queued``/``in_progress`` background response emits its
    current prelude then live-follows: each pass re-reads the envelope and
    emits any newly-derivable events until the terminal frame
    (``response.completed``/``incomplete``/``failed``/``cancelled``) or the
    ``timeout_s`` deadline (clamped to ``_REPLAY_FOLLOW_TIMEOUT_MAX_S``
    so the follow can't outrun the literal cap) — a ``: keepalive``
    comment rides each idle interval like ``/harness/jobs/{id}/events``.
    A record deleted or evicted mid-follow ends the stream with no
    terminal frame (the record is gone — there is nothing honest left
    to say).
    """

    def _frame(event: str, payload: dict[str, Any], seq: int) -> str:
        return f"event: {event}\nid: {seq}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"

    cursor = skip
    deadline = time.monotonic() + min(timeout_s, _REPLAY_FOLLOW_TIMEOUT_MAX_S)
    next_keep = time.monotonic() + keepalive_s if keepalive_s > 0 else math.inf
    poll_s = 0.25 if keepalive_s <= 0 else max(_REPLAY_MIN_POLL_S, min(0.25, keepalive_s))
    # static trip bound: the follow window is request input, so the loop
    # iterates a fixed worst case and the deadline still ends it early
    for _ in range(_REPLAY_FOLLOW_MAX_ITERS):
        env = envelope_store.get(response_id)
        if env is None:
            return
        events = list(openai_response_replay_events(env))
        for i, (event, payload) in enumerate(events):
            # the cursor is request input — compare, never bound the loop
            # on it (a huge starting_after just skips, it can't extend)
            if i < cursor:
                continue
            yield _frame(event, payload, i)
        if env.get("status") in OPENAI_RESPONSE_TERMINAL:
            return
        cursor = max(cursor, len(events))
        now = time.monotonic()
        if now >= deadline:
            return
        if now >= next_keep:
            yield ": keepalive\n\n"
            next_keep = now + keepalive_s
        time.sleep(poll_s)


def _openai_sse(
    body: OpenAIChatRequest,
    *,
    content: str | list[str],
    backend: str,
    model: str | None,
    usage: dict[str, int] | None,
    cid: str,
    tool_calls: Sequence[Sequence[dict[str, Any]] | None] | None = None,
    finish_reasons: Sequence[str] | None = None,
    created: int | None = None,
    logprobs: Sequence[dict[str, Any] | None] | None = None,
    skip: int = 0,
) -> Iterator[str]:
    """Serialize ``openai_chunks`` payloads into SSE frames +
    the terminal ``[DONE]`` marker. The chunk payloads themselves are
    generated by the shared compat layer (``openai_chunks``), so the wire
    and the SDK emit the same sequence. ``created`` pins the chunk
    timestamp — idempotent replays pass the stored envelope's value so a
    resumed stream is byte-identical.

    Every frame carries an ``id:`` equal to its index in the sequence
    (``[DONE]`` takes the index past the last chunk), so a client that
    records ``Last-Event-ID`` can resume a dropped keyed stream — the
    route replays the pinned response and ``skip`` drops frames at or
    below the last delivered index."""

    def _frame(payload: dict[str, Any], seq: int) -> str:
        return f"id: {seq}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"

    seq = 0
    for payload in openai_chunks(
        text=content,
        backend=backend,
        model=model,
        cid=cid,
        include_usage=bool((body.stream_options or {}).get("include_usage")),
        usage=usage,
        tool_calls=tool_calls,
        finish_reasons=finish_reasons,
        created=created,
        logprobs=logprobs,
    ):
        if seq >= skip:
            yield _frame(payload, seq)
        seq += 1
    yield f"id: {seq}\ndata: [DONE]\n\n"


def _legacy_sse(
    env: dict[str, Any],
    *,
    body: OpenAICompletionRequest,
    skip: int = 0,
) -> Iterator[str]:
    """Serialize ``completion_events`` payloads into SSE frames + the
    terminal ``[DONE]`` marker — the legacy ``/v1/completions`` stream
    grammar. Frame ids/indexing and ``Last-Event-ID`` resume semantics
    match ``_openai_sse`` (a replayed keyed call regenerates the frames
    byte-identically and ``skip`` drops the prefix)."""

    def _frame(payload: dict[str, Any], seq: int) -> str:
        return f"id: {seq}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"

    seq = 0
    for payload in completion_events(
        env, include_usage=bool((body.stream_options or {}).get("include_usage"))
    ):
        if seq >= skip:
            yield _frame(payload, seq)
        seq += 1
    yield f"id: {seq}\ndata: [DONE]\n\n"


def _breaker_key_name(name: str, byok: ByokOverride | None) -> str:
    """Backend key for one chain link — a BYOK endpoint gets its own
    circuit keyed by endpoint hash wherever it sits in the chain."""
    if byok is None or name != "byok":
        return name
    return f"{name}:{hashlib.sha256(byok.base_url.encode()).hexdigest()[:16]}"


def _breaker_key(body: CompleteRequest | CompleteBatchRequest) -> str:
    """Backend key for the circuit breaker. A per-request BYOK override gets
    its own circuit keyed by endpoint hash — one caller's dead endpoint must
    never fast-fail another caller's BYOK or the env-configured default."""
    return _breaker_key_name(body.backend, body.byok)


def _idem_scope(key: str | None, *, namespace: str | None = None) -> str | None:
    """Credential-namespace an Idempotency-Key: the record stored under
    ``key`` by credential A must never replay under credential B — the
    recorded answer may carry tenant state B has no right to (a minted
    secret, a private job, a queue slot). The env root credential is
    ``env``, managed keys namespace under their ``key_id`` fingerprint,
    and loopback dev sessions (no credential at all) share one
    ``loopback`` scope. The caller-provided key is represented only by
    its SHA-256 digest so journals never persist the raw header value.
    ``namespace`` further partitions a key per
    target/verb (``{scope}:{key}:{namespace}``) so one key can pin the
    same logical request against different path objects — rotate vs
    patch vs complete on different ``{id}`` path params never collide."""
    if key is None:
        return None
    key_id = _REQUEST_KEY_ID.get()
    key_digest = hashlib.sha256(key.encode()).hexdigest()
    scoped = f"{key_id if key_id is not None else 'loopback'}:{key_digest}"
    return f"{scoped}:{namespace}" if namespace is not None else scoped


def _idem_lookup[IdemT: BaseModel](
    idempotency_key: str | None,
    store: _IdemStore[IdemT],
    body_fp: str,
    *,
    namespace: str | None = None,
) -> tuple[str | None, IdemT | None]:
    """Shared Idempotency-Key preamble: normalize + bound the key,
    credential-scope it, then look up a stored replay. Returns
    ``(scoped_key, cached)`` — a cached hit is the response to return
    verbatim plus ``replayed: True``; a key reused with a different body
    fails closed ``409 idempotency_conflict``."""
    key = (idempotency_key or "").strip() or None
    if key is None:
        return None, None
    if len(key) > _IDEM_KEY_MAX:
        raise ApiError(400, "Idempotency-Key must be <= 256 chars")
    skey = _idem_scope(key, namespace=namespace)
    assert skey is not None  # noqa: S101 — key is not None here
    entry = store.get(skey)
    if entry is None:
        return skey, None
    fp, cached = entry
    if fp != body_fp:
        raise ApiError(
            409,
            "Idempotency-Key reuse with a different request body",
            code="idempotency_conflict",
        )
    return skey, cached.model_copy(update={"replayed": True})


_RESUME_MISS_MSG = (
    "Last-Event-ID resume needs a pinned stream under this Idempotency-Key — nothing stored"
)


def _resume_skip(last_event_id: str | None, *, stream: bool, idempotency_key: str | None) -> int:
    """``Last-Event-ID`` → chunk-skip count, fail-closed.

    The header is SSE-only (400 on non-stream), must parse to an int >= 0,
    and only means anything under an ``Idempotency-Key`` — resuming needs
    the pinned stream the key names. Returns the number of already
    delivered frames the caller's SSE generator should skip."""
    if last_event_id is None:
        return 0
    if not stream:
        raise ApiError(400, "Last-Event-ID applies to stream requests only", code="bad_resume")
    try:
        seen = int(last_event_id)
    except ValueError as exc:
        raise ApiError(
            400,
            f"Last-Event-ID must be a frame index, got {last_event_id!r}",
            code="bad_resume",
        ) from exc
    if seen < 0:
        raise ApiError(400, "Last-Event-ID must be >= 0", code="bad_resume")
    if not (idempotency_key or "").strip():
        raise ApiError(
            400,
            "resuming a stream needs the original call's Idempotency-Key",
            code="resume_needs_key",
        )
    return seen + 1


# Headers every SSE leg on the create surfaces sends — the jobs-status
# stream set the precedent: proxies must not buffer event frames.
_SSE_STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
# OpenAI-dialect SSE wire literals shared by every create-stream leg.
_SSE_DATA_PREFIX = "data: "
_SSE_DONE = "data: [DONE]\n\n"
# The chat-surface gated call's resolved payload: response envelope plus
# the completion fingerprint that rides headers or in-band ids.
_ChatEnv = tuple[dict[str, Any], str]


def _grace_pipe(work: Callable[[], Any]) -> queue.Queue[tuple[str, Any]]:
    """Run ``work`` on a daemon thread, publishing ``("ok", result)`` or
    ``("error", HTTPException)`` — the transport for the grace-window
    streams on the ``/v1`` create surfaces (the same pattern
    ``/harness/complete/stream`` uses).

    ``contextvars.copy_context`` carries request-scoped attribution into
    the worker — auth key id and request logging live on contextvars
    that a fresh thread does not inherit, so without this a keepalived
    call would record completions unattributed and skip credential
    metering. A generic failure wraps to a 502 HTTPException so a
    backend fault lands as an honest in-band frame, never a hang."""
    pipe: queue.Queue[tuple[str, Any]] = queue.Queue()
    ctx = contextvars.copy_context()

    def _produce() -> None:
        try:
            pipe.put(("ok", ctx.run(work)))
        except HTTPException as exc:
            pipe.put(("error", exc))
        except Exception as exc:  # noqa: BLE001
            # a dead pipe is an honest in-band frame, never a hang
            pipe.put(("error", HTTPException(502, f"backend failed: {exc}")))

    threading.Thread(target=_produce, daemon=True).start()
    return pipe


def _grace_await(pipe: queue.Queue[tuple[str, Any]], keepalive_s: float) -> tuple[str, Any] | None:
    """First published outcome inside the grace window, or ``None`` when
    the window lapses — the caller then commits to the keepalived
    stream."""
    try:
        return pipe.get(timeout=keepalive_s)
    except queue.Empty:
        return None


def _grace_stage(
    generate: Callable[[], Any], *, stream: bool, keepalive_s: float
) -> tuple[tuple[str, Any] | None, queue.Queue[tuple[str, Any]] | None]:
    """Run ``generate`` through the grace-window pipe when ``stream`` and
    ``keepalive_s`` are on; resolve inline otherwise. Returns the outcome
    (``None`` once the window lapsed — the caller then commits to the
    keepalived stream) and the pipe for that leg to drain."""
    pipe: queue.Queue[tuple[str, Any]] | None = None
    outcome: tuple[str, Any] | None = None
    if stream and keepalive_s > 0:
        pipe = _grace_pipe(generate)
        outcome = _grace_await(pipe, keepalive_s)
    if outcome is None and (not stream or keepalive_s <= 0):
        outcome = ("ok", generate())
    return outcome, pipe


def _deliver_callback(
    rec: JobStatusResponse | EvalRecord | FTJob | _BatchRecord | _AnthropicBatchRecord,
    *,
    body: bytes | None = None,
) -> None:
    """Terminal-state webhook: POST the record to the caller's
    ``callback_url`` — the shared contract for jobs, evals, fine-tuning
    jobs, and batches. ``body`` overrides the serialized payload (batches
    deliver the projected OpenAI envelope, not the internal record).
    Best-effort — a dead or slow endpoint records
    ``callback_status='failed'`` on the record, never raises into the
    worker and never changes the record's own status. Transient faults
    (network errors, 5xx) retry ``WEBHOOK_MAX_ATTEMPTS`` times with
    capped backoff; a 4xx is a definitive rejection and is never
    retried. Fire-once: the first call for a record wins the
    ``_callback_fired`` flag under ``_callback_lock`` — duplicate terminal
    transitions (a repeated DELETE on a cancelled record, a cancel+
    complete race, expiry-on-read re-projects) never re-deliver. Recovery
    also claims this flag because callback secrets are never journaled."""
    url = rec.callback_url
    if not url:
        return
    with rec._callback_lock:
        if rec._callback_fired:
            return
        rec._callback_fired = True
    payload = body if body is not None else rec.model_dump_json().encode()
    ok, err, attempts = deliver_signed(url, rec._callback_secret, payload)
    rec.callback_status = "delivered" if ok else "failed"
    rec.callback_error = None if ok else err
    rec.callback_attempts = attempts


_CREDENTIAL_LOC_SUFFIXES = ("_key", "_secret", "_token", "_password")


def _redact_credential_inputs(errs: Sequence[Any]) -> None:
    """Blank the echoed ``input`` on errors inside credential-bearing
    fields. A rejection's ``loc``+``msg`` stay (the caller needs them to
    fix the request); the rejected *value* of a ``byok``/``api_key``/
    ``*_secret``-style field never round-trips into the wire body."""
    credential_locs = {"byok", "judge_byok", "api_key"}
    for err in errs:
        loc = err.get("loc")
        if not isinstance(loc, (tuple, list)):
            continue
        if any(
            str(part) in credential_locs or str(part).endswith(_CREDENTIAL_LOC_SUFFIXES)
            for part in loc
        ):
            if "input" in err:
                err["input"] = "[redacted]"
            err.pop("ctx", None)
            err.pop("url", None)


def _validation_msgs(exc: ValidationError) -> str:
    return "; ".join(str(e.get("msg", "invalid request")) for e in exc.errors())


def _request_or_422[ReqT: BaseModel](model_cls: type[ReqT], kwargs: dict[str, Any]) -> ReqT:
    """Bind a translated request through the harness model — the
    cross-field checks (chain membership, sampling shape) only exist
    there, so a rejection must land the same 422 the body-level
    validator would produce, never escape as a bare 500."""
    try:
        return model_cls(**kwargs)
    except ValidationError as exc:
        raise ApiError(422, _validation_msgs(exc), code="invalid_request") from exc


def _validation_response(request: Request, exc: RequestValidationError) -> JSONResponse:
    errs = exc.errors()
    _redact_credential_inputs(errs)
    if is_openai_path(request.url.path):
        msg = "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', []))}: {e.get('msg', '')}" for e in errs[:4]
        )
        return JSONResponse(
            status_code=422,
            content=_v1_error_body(request.url.path, msg or "invalid request", 422, "validation"),
        )
    try:
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder({"detail": errs, "code": "validation"}),
        )
    except RecursionError:
        # A rejected input can be too deeply nested to echo. Preserve
        # the validation failure without recursively encoding that input.
        details = [{key: err[key] for key in ("type", "loc", "msg") if key in err} for err in errs]
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder({"detail": details, "code": "validation"}),
        )


def _idem_key(key: str | None) -> str | None:
    """Normalize a claim key exactly as the existing replay lookup does."""
    key = (key or "").strip() or None
    if key is not None and len(key) > _IDEM_KEY_MAX:
        raise ApiError(400, "Idempotency-Key must be <= 256 chars")
    if key is not None and any(ord(char) < 0x20 or ord(char) == 0x7F for char in key):
        raise ApiError(400, "Idempotency-Key must not contain control characters")
    return key


class _IdemClaimStore(Protocol):
    """Any store carrying per-key async claims — ``_IdemStore``,
    ``_JobStore``, ``EvalStore``, ``FTJobStore`` all satisfy it."""

    def async_claim_lock(self, key: str | None) -> AbstractAsyncContextManager[None]: ...


def _idem_claim_dep(
    store: _IdemClaimStore,
) -> Callable[..., AsyncIterator[None]]:
    """``Depends`` factory — hold the store's claim on the credential-
    scoped key for the whole handler span, so a retry's lookup cannot
    slide between a twin's lookup and replay-record publication
    (async-acquire so waiters yield the event loop instead of stranding
    sync-pool workers, same contract as the batch claim deps)."""

    async def dep(
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> AsyncIterator[None]:
        key = _idem_key(idempotency_key)
        async with store.async_claim_lock(_idem_scope(key)):
            yield

    return dep


async def _run_claimed[ResultT](
    claim: AbstractAsyncContextManager[None], action: Callable[[], ResultT]
) -> ResultT:
    """Keep a key claimed until its synchronous worker has really finished.

    Waiting retries yield without occupying request workers. A shielded
    child keeps executing after its caller is cancelled; the task group
    drains it before releasing the claim, so a retry cannot overlap work
    that is still publishing its replay record. Worker errors are raised
    outside the group to preserve their original HTTP error type.
    """
    import anyio

    results: list[ResultT] = []
    failures: list[Exception] = []

    async def execute() -> None:
        with anyio.CancelScope(shield=True):
            try:
                results.append(await anyio.to_thread.run_sync(action))
            except Exception as exc:  # noqa: BLE001 — preserve the worker's exact error
                failures.append(exc)

    async with claim:
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(execute)
        if failures:
            raise failures[0]
        return results[0]


def _drain_refusal(metrics: _Metrics) -> None:
    """The drain half of the work gate, standalone: mutating routes that
    take no inflight slot (stored-resource writes, the key lifecycle)
    still refuse once the latch is set — drain admits no new work at
    all. Reads, replays, cancels, deletes, and the documented advisory
    preflights stay open under it."""
    if metrics.draining.is_set():
        raise ApiError(503, "harness is draining — no new work accepted", code="draining")


def _submit_job(
    body: HarnessRunRequest,
    idempotency_key: str | None,
    lab: Harness,
    job_store: _JobStore,
    metrics: _Metrics,
    inflight: threading.BoundedSemaphore,
    jobs_executor: ThreadPoolExecutor,
) -> JobSubmitResponse:
    """Job submission core: idempotency lookup -> command validation ->
    slot admission -> background execution. The slot is held for the
    job's lifetime and released by the worker, so the queue can never
    grow past ``max_inflight`` (no unbounded buffering)."""
    key = (idempotency_key or "").strip() or None
    if key is not None and len(key) > _IDEM_KEY_MAX:
        raise ApiError(400, "Idempotency-Key must be <= 256 chars")
    skey = _idem_scope(key)
    body_fp = _body_fp(body, exclude={"idempotency_key"})
    if skey is not None:
        entry = job_store.get_key(skey)
        if entry is not None:
            fp, job_id = entry
            if fp != body_fp:
                raise ApiError(
                    409,
                    "Idempotency-Key reuse with a different request body",
                    code="idempotency_conflict",
                )
            job = job_store.get(job_id)
            if job is not None:
                return JobSubmitResponse(job_id=job_id, status=job.status, replayed=True)
    try:
        lab.get(body.command)  # fail closed at submit, not in the worker
    except KeyError as exc:
        raise ApiError(404, str(exc)) from exc
    _drain_refusal(metrics)
    if not inflight.acquire(blocking=False):
        raise ApiError(
            503,
            f"harness at max_inflight={metrics.max_inflight} — retry later",
            code="over_capacity",
            headers={"Retry-After": "1"},
        )
    metrics.acquire()
    job = JobStatusResponse(
        job_id=uuid.uuid4().hex,
        status="queued",
        created_at=time.time(),
        finished_at=None,
        result=None,
        error=None,
        callback_url=body.callback_url,
    )
    job._callback_secret = body.callback_secret

    def _exec() -> None:
        # The queued→running hop goes through the store's atomic claim:
        # a cancel that landed (or lands) while this future sat pending
        # wins under the lock — a cancelled job is never resurrected.
        try:
            started = job_store.start(job.job_id)
        except Exception:
            metrics.release()
            inflight.release()
            raise
        if started is None:
            metrics.release()
            inflight.release()
            return
        try:
            result = lab.run(
                body.command,
                body.extra_args or None,
                config=Path(body.config) if body.config else None,
            )
            command = lab.get(body.command)
            stdout, stdout_truncated = _cap_job_text(result.stdout)
            stderr, stderr_truncated = _cap_job_text(result.stderr)
            job.result = HarnessRunResponse(
                command=result.command,
                exit_code=result.exit_code,
                stdout=stdout,
                stderr=stderr,
                ok=result.ok,
                timeout_s=command.timeout_s,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
            )
            job.status = "succeeded"
        except Exception as exc:  # noqa: BLE001 — worker faults land in the record
            job.error = f"{type(exc).__name__}: {exc}"
            job.status = "failed"
        finally:
            # stamp finished_at before the webhook fires — the delivered
            # record is the final record, never a pre-terminal snapshot
            job.finished_at = time.time()
            _deliver_callback(job)
        try:
            job_store.mark(job)
        finally:
            metrics.release()
            inflight.release()

    # The record registers with the store *before* hand-off: the worker
    # claims by id through start(), so a transition can never journal
    # ahead of the job existing. A refused hand-off leaves a tombstone
    # rather than a ghost record.
    try:
        job_store.put(job, skey, body_fp)
    except Exception:
        metrics.release()
        inflight.release()
        raise
    try:
        jobs_executor.submit(_exec)
    except RuntimeError as exc:  # executor gone (shutdown race)
        try:
            job_store.delete(job.job_id)
        finally:
            metrics.release()
            inflight.release()
        raise ApiError(503, "job executor unavailable", code="over_capacity") from exc
    return JobSubmitResponse(job_id=job.job_id, status=job.status, replayed=False)


def _make_lifespan(
    metrics: _Metrics,
    job_store: _JobStore,
    eval_store: EvalStore,
    jobs_executor: ThreadPoolExecutor,
    ft_store: FTJobStore | None = None,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Graceful-exit contract: on shutdown the gate drains (new work gets
    503), every still-queued job flips to 'cancelled' and fires its
    signed webhook, queued evals flip to 'cancelled', and the executor
    releases pending futures. Running work isn't interrupted — it
    finishes bounded by its command timeout or dies with the process."""

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        metrics.draining.set()
        for pending in job_store.cancel_pending():
            _deliver_callback(pending)
        for pending_eval in eval_store.cancel_pending():
            _deliver_callback(pending_eval)
        if ft_store is not None:
            for pending_ft in ft_store.cancel_pending():
                _deliver_callback(pending_ft)
        jobs_executor.shutdown(wait=False, cancel_futures=True)

    return _lifespan


def _stored_receipt_matches(body: bytes, sha256: str) -> bool:
    """True when the sealed JSON's ``receipt_sha256`` equals *sha256*."""
    try:
        stored_sha = json.loads(body).get("receipt_sha256")
    except (AttributeError, TypeError, ValueError):
        stored_sha = None
    return isinstance(stored_sha, str) and hmac.compare_digest(stored_sha, sha256)


def _mount_receipt_routes(app: FastAPI, receipt_index: _ReceiptIndex) -> None:
    """Receipt-verify + content-addressed fetch routes, kept out of
    ``create_app`` to keep its branch complexity under the repo's ruff cap."""

    @app.post(
        "/receipts/verify",
        tags=["receipts"],
        operation_id="verify_receipt",
    )
    def verify_receipt(body: ReceiptVerifyRequest) -> ReceiptVerifyResponse:
        try:
            result = verify_receipt_payload(body.receipt, path=Path("<api>"))
        except Exception as exc:  # verifier must fail closed, never 500
            return ReceiptVerifyResponse(
                valid=False,
                path="<api>",
                schema_tag="",
                kind=None,
                verdict=None,
                digest_convention=None,
                errors=[str(exc)],
                warnings=[],
            )
        return ReceiptVerifyResponse(
            valid=result["valid"],
            path=result["path"],
            schema_tag=result["schema"],
            kind=result["kind"] if isinstance(result["kind"], str) else None,
            verdict=result["verdict"] if isinstance(result["verdict"], str) else None,
            digest_convention=result["digest_convention"],
            errors=list(result["errors"]),
            warnings=list(result["warnings"]),
        )

    @app.post(
        "/receipts/verify/batch",
        tags=["receipts"],
        operation_id="verify_receipts_batch",
    )
    def verify_receipts_batch(body: ReceiptVerifyBatchRequest) -> ReceiptVerifyBatchResponse:
        items: list[ReceiptVerifyBatchItem] = []
        for i, receipt in enumerate(body.receipts):
            try:
                result = verify_receipt_payload(receipt, path=Path("<api>"))
            except Exception as exc:  # verifier must fail item-local, never 500
                items.append(
                    ReceiptVerifyBatchItem(
                        index=i,
                        valid=False,
                        path="<api>",
                        schema_tag="",
                        kind=None,
                        verdict=None,
                        digest_convention=None,
                        errors=[str(exc)],
                        warnings=[],
                    )
                )
                continue
            items.append(
                ReceiptVerifyBatchItem(
                    index=i,
                    valid=result["valid"],
                    path=result["path"],
                    schema_tag=result["schema"],
                    kind=result["kind"] if isinstance(result["kind"], str) else None,
                    verdict=result["verdict"] if isinstance(result["verdict"], str) else None,
                    digest_convention=result["digest_convention"],
                    errors=list(result["errors"]),
                    warnings=list(result["warnings"]),
                )
            )
        verified = sum(1 for it in items if it.valid)
        return ReceiptVerifyBatchResponse(
            verified=verified,
            failed=len(items) - verified,
            results=items,
        )

    @app.get(
        "/receipts",
        tags=["receipts"],
        operation_id="receipts_index",
    )
    def receipts_index() -> ReceiptIndexResponse:
        """List the store's sealed receipts: sha256 → filename, sorted."""
        if not receipt_index.available():
            raise ApiError(503, "receipts store unavailable", code="receipts_unavailable")
        items = receipt_index.items()
        return ReceiptIndexResponse(
            items=[ReceiptIndexItem(sha256=sha, name=path.name) for sha, path in items],
            count=len(items),
        )

    @app.get(
        "/receipts/{sha256}",
        tags=["receipts"],
        operation_id="receipt_fetch",
        response_class=Response,
    )
    def receipt_fetch(sha256: str, request: Request) -> Response:
        """Serve the sealed receipt addressed by its content hash.

        The body is the receipt's committed bytes verbatim, and receipts are
        content-addressed by their seal — so ``ETag`` is the hash itself and
        the response is ``Cache-Control: immutable``. Validity under the live
        verifier is reported on ``X-Fx1-Receipt-Valid`` so a caller learns
        the receipt it fetched still verifies without a second round trip.
        """
        if not _SHA256_HEX.fullmatch(sha256):
            raise ApiError(422, "sha256 must be 64 lowercase hex", code="invalid_sha256")
        if not receipt_index.available():
            raise ApiError(503, "receipts store unavailable", code="receipts_unavailable")
        path = receipt_index.lookup(sha256)
        if path is None:
            raise ApiError(404, "receipt not found", code="receipt_not_found")
        try:
            body = path.read_bytes()
        except OSError as exc:
            raise ApiError(503, "receipts store read failed", code="receipts_unavailable") from exc
        verdict = verify_receipt_bytes(body, path)
        if not verdict["valid"]:
            raise ApiError(
                409,
                "stored receipt failed integrity verification",
                code="receipt_integrity_failed",
            )
        if not _stored_receipt_matches(body, sha256):
            raise ApiError(
                409,
                "stored receipt does not match requested digest",
                code="receipt_integrity_failed",
            )
        inm = request.headers.get("if-none-match", "")
        if inm.strip() == "*" or f'"{sha256}"' in inm:
            return Response(
                status_code=304,
                headers={
                    "ETag": f'"{sha256}"',
                    "Cache-Control": "public, immutable",
                    "X-Fx1-Receipt-Valid": "true",
                },
            )
        return Response(
            content=body,
            media_type="application/json",
            headers={
                "ETag": f'"{sha256}"',
                "Cache-Control": "public, immutable",
                "X-Fx1-Receipt-Valid": "true",
            },
        )


def _mount_v1_catch_all(app: FastAPI) -> None:
    """Catch-all for unmapped ``/v1`` paths — a stock SDK hitting a route
    the surface doesn't implement gets the provider's own error grammar
    (OpenAI ``Invalid URL (METHOD /path)`` / Anthropic ``not_found_error``
    under ``/v1/messages``), not FastAPI's ``{"detail": "Not Found"}``.
    Registered last so method mismatches land here too — matching OpenAI,
    which 404s unknown method+path pairs."""

    @app.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
        include_in_schema=False,
    )
    def _v1_catch_all(request: Request) -> JSONResponse:
        path = request.url.path
        if path == "/v1/messages" or path.startswith("/v1/messages/"):
            body: dict[str, Any] = anthropic_error_body("Not Found", 404)
        else:
            body = openai_error_body(f"Invalid URL ({request.method} {path})", 404, "not_found")
        return JSONResponse(status_code=404, content=body)


def _mount_job_routes(
    app: FastAPI,
    lab: Harness,
    job_store: _JobStore,
    metrics: _Metrics,
    inflight: threading.BoundedSemaphore,
    jobs_executor: ThreadPoolExecutor,
) -> None:
    """Async job lifecycle routes: submit (202) / status / list / cancel.
    Submit is gated by drain + max_inflight; reads and cancel are
    control-plane and stay open under drain."""

    @app.post(
        "/harness/jobs",
        response_model=JobSubmitResponse,
        status_code=202,
        tags=["jobs"],
        operation_id="submit_job",
    )
    async def submit_job(
        body: HarnessRunRequest,
        response: Response,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> JobSubmitResponse:
        """Async run submission: work starts in the background, the caller
        polls ``GET /harness/jobs/{job_id}`` for the terminal record (also
        echoed as the ``Location`` header). Same drain/cap/idempotency
        contract as the sync route."""
        key = _idem_key(idempotency_key or body.idempotency_key)
        out = await _run_claimed(
            job_store.async_claim_lock(_idem_scope(key)),
            lambda: _submit_job(body, key, lab, job_store, metrics, inflight, jobs_executor),
        )
        response.headers["Location"] = f"/harness/jobs/{out.job_id}"
        return out

    @app.post(
        "/harness/jobs/batch",
        response_model=JobBatchResponse,
        status_code=202,
        tags=["jobs"],
        operation_id="submit_jobs_batch",
    )
    async def submit_jobs_batch(body: JobBatchRequest) -> JobBatchResponse:
        """Fan-out submit: each item takes the same path as the single
        route — command validation, drain latch, inflight cap, and
        per-item ``idempotency_key`` dedup (headers carry no per-item
        keys, so the body field is the channel here). An item that fails
        lands in its own ``jobs[i]`` slot as ``{error, code}`` — the
        batch reports honest ``submitted``/``failed`` counts rather than
        turning one bad item into a whole-batch failure."""
        items: list[JobBatchItemResponse] = []
        submitted = 0
        for i, req in enumerate(body.jobs):
            try:
                key = _idem_key(req.idempotency_key)

                def _submit_one(
                    req: HarnessRunRequest = req,
                    key: str | None = key,
                ) -> JobSubmitResponse:
                    return _submit_job(req, key, lab, job_store, metrics, inflight, jobs_executor)

                resp = await _run_claimed(job_store.async_claim_lock(_idem_scope(key)), _submit_one)
                items.append(
                    JobBatchItemResponse(
                        index=i,
                        job_id=resp.job_id,
                        status=resp.status,
                        replayed=resp.replayed,
                    )
                )
                submitted += 1
            except ApiError as exc:
                items.append(JobBatchItemResponse(index=i, error=str(exc.detail), code=exc.code))
        return JobBatchResponse(jobs=items, submitted=submitted, failed=len(body.jobs) - submitted)

    @app.get(
        "/harness/jobs", response_model=JobListResponse, tags=["jobs"], operation_id="list_jobs"
    )
    def list_jobs(
        status: _JobStatus | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> JobListResponse:
        """Job inventory, newest first: ``status`` filter + offset/limit
        paging; ``total`` is the filtered count before the page is cut."""
        jobs = job_store.list(status=status)
        return JobListResponse(jobs=jobs[offset : offset + limit], total=len(jobs))

    @app.get(
        "/harness/jobs/{job_id}",
        response_model=JobStatusResponse,
        tags=["jobs"],
        operation_id="get_job",
    )
    def job_status(job_id: str) -> JobStatusResponse:
        job = job_store.get(job_id)
        if job is None:
            raise ApiError(404, f"unknown job_id {job_id!r}")
        return job

    @app.get("/harness/jobs/{job_id}/receipt", tags=["jobs"], operation_id="job_receipt")
    def job_receipt(job_id: str) -> dict[str, Any]:
        """Export the job's ledger record as a sealed
        ``fx1_job_record.v1`` document — the terminal ``result`` embeds
        with its streams digested (stdout/stderr sha256, never content).
        Verify with ``POST /receipts/verify`` or the SDK."""
        from fx1.serve.ops_receipt import job_record_receipt  # noqa: PLC0415

        job = job_store.get(job_id)
        if job is None:
            raise ApiError(404, f"unknown job_id {job_id!r}")
        return job_record_receipt(job.model_dump(mode="json"))

    @app.delete(
        "/harness/jobs/{job_id}",
        tags=["jobs"],
        operation_id="cancel_job",
    )
    def cancel_job(job_id: str) -> JobStatusResponse:
        """Cooperative cancel: a queued job flips to 'cancelled' and its
        executor slot frees on dequeue. Running and terminal jobs 409 —
        the subprocess runner has no mid-run kill handle."""
        job, outcome = job_store.cancel(job_id)
        if job is None:
            raise ApiError(404, f"unknown job_id {job_id!r}")
        if outcome != "cancelled":
            raise ApiError(409, f"job {job_id!r} is {outcome}")
        _deliver_callback(job)  # cancelled is terminal — fire the webhook
        return job

    @app.get(
        "/harness/jobs/{job_id}/events",
        tags=["jobs"],
        operation_id="stream_job_events",
    )
    def job_events(
        job_id: str,
        timeout_s: float = Query(default=600.0, ge=1.0, le=3600.0),
    ) -> StreamingResponse:
        """SSE job-status stream: the job record arrives as an
        ``event: job`` frame on every change until a terminal status,
        then the stream closes. ``: keepalive`` comment frames fire every
        ``sse_keepalive_s`` while the job is quiet; ``timeout_s`` bounds
        the stream — reconnect (and resume from the last frame) to keep
        watching. Same read path as ``GET /harness/jobs/{job_id}`` — a
        snapshot before the stream opens, so an unknown id 404s rather
        than hanging."""
        job = job_store.get(job_id)
        if job is None:
            raise ApiError(404, f"unknown job_id {job_id!r}")
        keepalive_s = float(getattr(app.state, "sse_keepalive_s", 15.0))

        def _frames() -> Iterator[str]:
            deadline = time.monotonic() + timeout_s
            last = ""
            next_keep = time.monotonic() + keepalive_s if keepalive_s > 0 else math.inf
            poll_s = 0.25 if keepalive_s <= 0 else min(0.25, keepalive_s)
            while True:
                cur = job_store.get(job_id)
                if cur is None:
                    return  # evicted from the bounded store mid-stream
                snap = cur.model_dump_json()
                if snap != last:
                    last = snap
                    yield f"event: job\ndata: {snap}\n\n"
                now = time.monotonic()
                if cur.status in _TERMINAL_JOB_STATUS or now >= deadline:
                    return
                if now >= next_keep:
                    yield ": keepalive\n\n"
                    next_keep = now + keepalive_s
                time.sleep(poll_s)

        return StreamingResponse(
            _frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


class _IdemStore[IdemT: BaseModel]:
    """Bounded LRU of ``Idempotency-Key`` -> stored response.

    Lets a client (or the HarnessClient, which mints a key per ``run``)
    retry a submission after a transport blip without double-executing
    the work. Read-only replays bypass the drain latch and the
    concurrency cap: the work already happened.

    With a ``JobJournal`` bound, each ``put`` is journaled (key,
    fingerprint, and the stored response itself — the replay *is* the
    response, so the payload must ride along) and boot restores the map:
    a retried submission post-restart replays the recorded answer
    instead of re-running it. ``model`` is the response class to decode
    with; required when a journal is bound.
    """

    def __init__(
        self,
        max_entries: int,
        journal: JobJournal | None = None,
        model: type[IdemT] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._max = max_entries
        self._claims = _ClaimLocks(max_entries)
        self._map: OrderedDict[str, tuple[str, IdemT]] = OrderedDict()
        self._journal = journal
        self.recover_warnings: list[str] = []
        if journal is not None:
            if model is None:
                raise ValueError("_IdemStore with a journal requires the response model class")
            res = journal.replay()
            self.recover_warnings = list(res.warnings)
            for payload in res.payloads:
                for evict in payload.get("evicted") or ():
                    self._map.pop(str(evict), None)
                rec = payload.get("idem")
                if rec is None:
                    continue
                self._map[str(rec["key"])] = (
                    str(rec["fp"]),
                    model.model_validate(rec["resp"]),
                )
                self._map.move_to_end(str(rec["key"]))
            self._compact_locked()

    def async_claim_lock(self, key: str | None) -> AbstractAsyncContextManager[None]:
        return self._claims.ahold(key)

    def _compact_locked(self) -> None:
        if self._journal is not None:
            self._journal.compact(
                [
                    {"idem": {"key": k, "fp": fp, "resp": resp.model_dump(mode="json")}}
                    for k, (fp, resp) in self._map.items()
                ]
            )

    def get(self, key: str) -> tuple[str, IdemT] | None:
        with self._lock:
            hit = self._map.get(key)
            if hit is not None:
                self._map.move_to_end(key)
            return hit

    def put(self, key: str, fingerprint: str, resp: IdemT) -> None:
        with self._lock:
            # Build the next state without publishing it.  The replay
            # journal is the durability boundary: if its fsync fails, the
            # process-visible map must remain identical to what a restart
            # will recover, including every entry that would have been
            # evicted by this insertion.
            stored = resp.model_copy(deep=True)
            next_map = self._map.copy()
            next_map[key] = (fingerprint, stored)
            next_map.move_to_end(key)
            evicted: list[str] = []
            while len(next_map) > self._max:
                old_key, _ = next_map.popitem(last=False)
                evicted.append(old_key)
            if self._journal is not None:
                payload: dict[str, Any] = {
                    "idem": {
                        "key": key,
                        "fp": fingerprint,
                        "resp": stored.model_dump(mode="json"),
                    }
                }
                if evicted:
                    payload["evicted"] = evicted
                self._journal.append(payload)
            self._map = next_map


def _has_stored_replay(request: Request, stores: Mapping[str, _IdemStore[Any]]) -> bool:
    """Whether this keyed completion route can answer without work."""
    store = stores.get(request.url.path)
    if store is None:
        return False
    key = _idem_key(request.headers.get("Idempotency-Key"))
    skey = _idem_scope(key)
    return skey is not None and store.get(skey) is not None


def _replay_aware_slot(
    request: Request,
    stores: Mapping[str, _IdemStore[Any]],
    work_gate: Callable[[], AbstractContextManager[None]],
) -> Iterator[None]:
    """Skip admission only when the claimed key already has a record."""
    if _has_stored_replay(request, stores):
        yield
        return
    with work_gate():
        yield


class _OpenAIIdemRecord(_Model):
    """Stored ``/v1/chat/completions`` outcome for ``Idempotency-Key``
    replay. The minted envelope is the source of truth: JSON responses
    replay it verbatim and SSE chunk payloads regenerate
    deterministically from it (same ``id``/``created``/content — a
    replayed stream is byte-identical to the first send). Only
    successful completions are stored: a gate refusal or backend fault
    is never pinned, so retrying a failed call re-executes."""

    envelope: dict[str, Any]
    # ``_idem_lookup`` stamps this on a cache hit, matching the other
    # idempotent routes' ``replayed`` convention (surfaced on the wire
    # as the ``X-Fx1-Idempotent-Replay`` header).
    replayed: bool = False


class _JsonIdemRecord(_Model):
    """Stored outcome for the mutating surfaces whose keyed replay is a
    JSON body verbatim — file/upload mints, key mint/rotate/patch/
    revoke. The wire envelope is served back byte-identical with the
    ``X-Fx1-Idempotent-Replay`` header; only successful outcomes are
    stored, so a refused request always re-executes on retry."""

    envelope: dict[str, Any]
    replayed: bool = False


class JobListResponse(_Model):
    """Job inventory page: ``total`` is the filtered count before paging."""

    jobs: list[JobStatusResponse]
    total: int


_JOB_BATCH_MAX = 64


class JobBatchRequest(_Model):
    """Batch submit envelope — ``min_length``/``max_length`` fail closed:
    an empty batch and an over-cap batch are both 422s."""

    jobs: list[HarnessRunRequest] = Field(min_length=1, max_length=_JOB_BATCH_MAX)


class JobBatchItemResponse(_Model):
    """Per-item outcome — either the submit tuple or the ApiError the
    item failed with (``code`` carries the machine-stable class)."""

    index: int
    job_id: str | None = None
    status: _JobStatus | None = None
    replayed: bool | None = None
    error: str | None = None
    code: str | None = None


class JobBatchResponse(_Model):
    jobs: list[JobBatchItemResponse]
    submitted: int
    failed: int


class _JobStore:
    """Bounded store of async run jobs + their Idempotency-Key index.

    Job submissions take a worker slot at submit time (same drain/cap
    contract as the sync route) and release it when the job finishes, so
    the queue can never grow past the declared concurrency bound. The
    store is an LRU; evicting a job also drops its idempotency mapping.

    With a ``JobJournal`` bound (``--state-dir`` on serve), every
    transition is journaled before the store mutates, and boot replays
    the chain: terminal records come back as-was; jobs still queued or
    running at the crash are restored as ``failed`` with an honest
    restart error (never re-run — their payload isn't journaled), and
    their idempotency keys still resolve so a retried submission returns
    the lost record instead of duplicating work.
    """

    def __init__(self, max_entries: int, journal: JobJournal | None = None) -> None:
        self._lock = threading.Lock()
        self._max = max_entries
        self._claims = _ClaimLocks(max_entries)
        self._jobs: OrderedDict[str, JobStatusResponse] = OrderedDict()
        self._keys: OrderedDict[str, tuple[str, str]] = OrderedDict()
        self._job_key: dict[str, str] = {}
        self._job_fp: dict[str, str] = {}
        self._journal = journal
        self.recover_warnings: list[str] = []
        if journal is not None:
            res = journal.replay()
            self.recover_warnings = list(res.warnings)
            now = time.time()
            for payload in res.payloads:
                for evict in payload.get("evicted") or ():
                    self._drop(str(evict))
                deleted = payload.get("deleted")
                if deleted is not None:
                    self._drop(str(deleted))
                    continue
                if "job" not in payload:
                    continue
                job = JobStatusResponse.model_validate(payload["job"])
                # Signing secrets are not journaled; recovered records never re-deliver.
                job._callback_fired = True
                self._jobs[job.job_id] = job
                self._jobs.move_to_end(job.job_id)
                key, fp = payload.get("key"), payload.get("fp")
                if key is not None and fp is not None:
                    self._keys[key] = (fp, job.job_id)
                    self._job_key[job.job_id] = key
                    self._job_fp[job.job_id] = fp
            for job in self._jobs.values():
                if job.status in ("queued", "running"):
                    job.status = "failed"
                    job.error = "process restarted before the job reached a terminal state"
                    job.finished_at = now
            self._compact_locked()

    def async_claim_lock(self, key: str | None) -> AbstractAsyncContextManager[None]:
        return self._claims.ahold(key)

    def _drop(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)
        key = self._job_key.pop(job_id, None)
        self._job_fp.pop(job_id, None)
        if key is not None:
            self._keys.pop(key, None)

    def _record(self, job: JobStatusResponse) -> dict[str, Any]:
        return {
            "job": job.model_dump(mode="json"),
            "key": self._job_key.get(job.job_id),
            "fp": self._job_fp.get(job.job_id),
        }

    def _compact_locked(self) -> None:
        """Rewrite the journal with only the live state — called on boot
        post-replay so dead history and torn tails don't accumulate."""
        if self._journal is not None:
            self._journal.compact([self._record(j) for j in self._jobs.values()])

    def mark(self, job: JobStatusResponse) -> None:
        """Journal a status transition made outside the store (the worker
        mutates ``job`` in place; this makes each hop durable)."""
        if self._journal is not None:
            with self._lock:
                self._journal.append(self._record(job))

    def start(self, job_id: str) -> JobStatusResponse | None:
        """Atomically claim a queued job for running — the dequeue half
        of the cancel contract. Under the store lock a racing ``cancel``
        either flips the record first (start loses, the cancelled job
        never runs) or reports 'running' (cancel 409s) — there is no
        check-then-set window for a cancelled job to resurrect through.
        The transition is journaled inside the lock so crash replay can
        never resurrect either."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != "queued":
                return None
            running = job.model_copy()
            running.status = "running"
            if self._journal is not None:
                self._journal.append(self._record(running))
            job.status = "running"
            return job

    def delete(self, job_id: str) -> None:
        """Drop a record + a durable tombstone — the submit path's
        compensation when executor hand-off fails after ``put``: the
        refused submission leaves no ghost job behind."""
        with self._lock:
            if job_id not in self._jobs:
                return
            if self._journal is not None:
                self._journal.append({"deleted": job_id})
            self._drop(job_id)

    def get(self, job_id: str) -> JobStatusResponse | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, status: str | None = None) -> list[JobStatusResponse]:
        """Newest-first snapshot, optionally filtered by status."""
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.reverse()
        if status is not None:
            jobs = [j for j in jobs if j.status == status]
        return jobs

    def cancel(self, job_id: str) -> tuple[JobStatusResponse | None, str]:
        """Cooperative cancel: a 'queued' job flips to 'cancelled' (the
        worker frees its slot on dequeue without running). Running and
        terminal jobs report their status so the route can 409."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None, "missing"
            if job.status == "queued":
                cancelled = job.model_copy()
                cancelled.status = "cancelled"
                cancelled.finished_at = time.time()
                if self._journal is not None:
                    self._journal.append(self._record(cancelled))
                job.status = cancelled.status
                job.finished_at = cancelled.finished_at
                return job, "cancelled"
            return job, job.status

    def cancel_pending(self) -> builtins.list[JobStatusResponse]:
        """Shutdown path: every queued job flips to 'cancelled' — its
        future never starts, so it holds no slot; running jobs finish or
        die with the process. Returns the cancelled records so the caller
        can fire their webhooks."""
        with self._lock:
            out = [j for j in self._jobs.values() if j.status == "queued"]
            for job in out:
                cancelled = job.model_copy()
                cancelled.status = "cancelled"
                cancelled.finished_at = time.time()
                if self._journal is not None:
                    self._journal.append(self._record(cancelled))
                job.status = cancelled.status
                job.finished_at = cancelled.finished_at
            return out

    def get_key(self, key: str) -> tuple[str, str] | None:
        with self._lock:
            hit = self._keys.get(key)
            if hit is not None:
                self._keys.move_to_end(key)
            return hit

    def put(
        self,
        job: JobStatusResponse,
        key: str | None,
        fingerprint: str | None,
    ) -> None:
        with self._lock:
            future_ids = [job_id for job_id in self._jobs if job_id != job.job_id]
            future_ids.append(job.job_id)
            evicted = future_ids[: max(0, len(future_ids) - self._max)]
            if self._journal is not None:
                payload: dict[str, Any] = {
                    "job": job.model_dump(mode="json"),
                    "key": key,
                    "fp": fingerprint,
                }
                if evicted:
                    payload["evicted"] = evicted
                self._journal.append(payload)

            self._drop(job.job_id)
            self._jobs[job.job_id] = job
            self._jobs.move_to_end(job.job_id)
            if key is not None and fingerprint is not None:
                self._keys[key] = (fingerprint, job.job_id)
                self._keys.move_to_end(key)
                self._job_key[job.job_id] = key
                self._job_fp[job.job_id] = fingerprint
            for old_id in evicted:
                self._drop(old_id)


class _FileRecord(_Model):
    """A stored upload — LRU-bounded in memory; with ``--state-dir`` the
    content rides along as a blob under ``files/`` (journaled metadata +
    blob bytes, so a restart returns the same bytes under the same id)."""

    file_id: str
    filename: str
    purpose: str
    size: int
    created_at: int
    content: bytes
    # New records bind metadata to exact bytes. ``None`` keeps journals
    # written before this field backward-compatible; a clean replay upgrades
    # them before compaction.
    content_sha256: str | None = None


_FILE_STORE_ID = re.compile(r"file-[0-9a-f]{32}")


class _FileStore:
    """Bounded LRU store of uploaded files (the /v1/files surface).

    Files are the batch input channel — one upload, then batches reference
    it by id. Bounded by entry count AND per-file bytes so an upload flood
    can't pin the process; eviction is silent LRU like the job store.

    With ``--state-dir`` the store is durable: content lands in
    ``files/<file_id>.bin`` (atomic tmp+rename) *before* the journaled
    metadata line, so replay only ever restores a record whose bytes are
    already on disk; eviction and delete journal tombstones and unlink the
    blob. Orphan blobs (journaled metadata lost to a torn tail, or a crash
    between blob write and journal) are GC'd on boot — a file id can never
    resurrect pointing at content that isn't there."""

    def __init__(
        self,
        max_entries: int,
        max_bytes: int,
        state_dir: Path | None = None,
    ) -> None:
        if max_entries < 1:
            raise ValueError(f"file store max_entries must be >= 1, got {max_entries}")
        if max_bytes < 1:
            raise ValueError(f"file store max_bytes must be >= 1, got {max_bytes}")
        self._lock = threading.Lock()
        self._max = max_entries
        self._max_bytes = max_bytes
        self._files: OrderedDict[str, _FileRecord] = OrderedDict()
        self._dir = Path(state_dir) / "files" if state_dir is not None else None
        self._journal = (
            JobJournal(Path(state_dir) / "files.jsonl") if state_dir is not None else None
        )
        self.recover_warnings: list[str] = []
        if self._journal is not None:
            res = self._journal.replay()
            self.recover_warnings = list(res.warnings)
            # Never rewrite a damaged journal around the last verified prefix:
            # doing so would destroy the corrupt suffix and can resurrect a
            # record whose delete lived there. Operator repair is required.
            if res.truncated_at is not None or res.dropped:
                raise RuntimeError("file journal is damaged; recovery requires operator repair")
            try:
                for payload in res.payloads:
                    self._apply_replay_op(payload)
                for file_id, rec in list(self._files.items()):
                    content = self._blob(file_id).read_bytes()
                    digest = hashlib.sha256(content).hexdigest()
                    if rec.size != len(content) or rec.size > self._max_bytes:
                        raise ValueError(
                            f"file {file_id}: metadata size does not match its content blob "
                            "or exceeds the configured byte cap"
                        )
                    if rec.content_sha256 is not None and not hmac.compare_digest(
                        rec.content_sha256, digest
                    ):
                        raise ValueError(f"file {file_id}: content sha256 does not match metadata")
                    self._files[file_id] = rec.model_copy(
                        update={"content": content, "content_sha256": digest}
                    )
                if len(self._files) > self._max:
                    raise ValueError("file journal live set exceeds the configured entry cap")
            except (OSError, ValueError) as exc:
                raise RuntimeError("file journal contains an invalid operation") from exc
            # GC only after the complete chain and every referenced blob have
            # validated. A corrupt suffix may still be the sole owner of a blob.
            blobs = (
                self._dir.glob("file-*.bin") if self._dir is not None and self._dir.is_dir() else ()
            )
            for blob in blobs:
                if blob.stem not in self._files:
                    blob.unlink(missing_ok=True)
            temporary = (
                self._dir.glob(".file-*.tmp")
                if self._dir is not None and self._dir.is_dir()
                else ()
            )
            for tmp in temporary:
                tmp.unlink(missing_ok=True)
            self._compact_locked()

    @staticmethod
    def _valid_id(file_id: str) -> bool:
        return _FILE_STORE_ID.fullmatch(file_id) is not None

    def _blob(self, file_id: str) -> Path:
        if self._dir is None:  # only reachable in the journal-less mode
            raise RuntimeError("file store has no state_dir — nothing durable to address")
        if not self._valid_id(file_id):
            raise ValueError(f"invalid file id {file_id!r}")
        return self._dir / f"{file_id}.bin"

    @staticmethod
    def _meta(rec: _FileRecord) -> dict[str, Any]:
        return rec.model_dump(mode="json", exclude={"content"})

    def _apply_replay_op(self, payload: dict[str, Any]) -> None:
        """Validate and apply one legacy-compatible journal operation."""
        if not isinstance(payload, dict):
            raise ValueError("file journal operation must be an object")
        evicted = payload.get("evicted", [])
        if not isinstance(evicted, list) or any(
            not isinstance(file_id, str) or not self._valid_id(file_id) for file_id in evicted
        ):
            raise ValueError("file journal evicted must be a list of valid ids")
        for file_id in evicted:
            self._files.pop(file_id, None)

        if "file" in payload:
            meta = payload["file"]
            if not isinstance(meta, dict):
                raise ValueError("file journal metadata must be an object")
            file_id = meta.get("file_id")
            if not isinstance(file_id, str) or not self._valid_id(file_id):
                raise ValueError("file journal metadata requires a valid file_id")
            # Content is hydrated after every operation has replayed: a blob
            # legitimately may be absent when a later tombstone removes this
            # record from the final live set.
            rec = _FileRecord.model_validate({**meta, "content": b""})
            self._files[file_id] = rec
            self._files.move_to_end(file_id)
            allowed = {"file", "evicted"}
        elif "file_deleted" in payload:
            file_id = payload["file_deleted"]
            if not isinstance(file_id, str) or not self._valid_id(file_id):
                raise ValueError("file journal delete requires a valid file id")
            self._files.pop(file_id, None)
            allowed = {"file_deleted"}
        elif "file_touched" in payload:
            file_id = payload["file_touched"]
            if (
                not isinstance(file_id, str)
                or not self._valid_id(file_id)
                or file_id not in self._files
            ):
                raise ValueError("file journal touch requires a live file id")
            self._files.move_to_end(file_id)
            allowed = {"file_touched"}
        else:
            raise ValueError("unknown file journal operation")
        if set(payload) - allowed:
            raise ValueError("file journal operation contains unexpected fields")

    def _compact_locked(self) -> None:
        if self._journal is not None:
            self._journal.compact([{"file": self._meta(r)} for r in self._files.values()])

    def put(self, *, filename: str, purpose: str, content: bytes) -> _FileRecord:
        if len(content) > self._max_bytes:
            raise ValueError(f"file exceeds the {self._max_bytes}-byte cap")
        rec = _FileRecord(
            file_id=f"file-{uuid.uuid4().hex}",
            filename=filename,
            purpose=purpose,
            size=len(content),
            created_at=int(time.time()),
            content=content,
            content_sha256=hashlib.sha256(content).hexdigest(),
        )
        if self._dir is not None:
            # Blob first, fsync'd: the journaled metadata line may only ever
            # name content that is already durable.
            self._dir.mkdir(parents=True, exist_ok=True)
            blob = self._blob(rec.file_id)
            tmp = self._dir / f".{rec.file_id}.tmp"
            try:
                with tmp.open("wb") as fh:
                    fh.write(content)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, blob)
            except Exception:
                tmp.unlink(missing_ok=True)
                blob.unlink(missing_ok=True)
                raise
        evicted: list[str] = []
        journal_size = 0
        if self._journal is not None:
            with suppress(FileNotFoundError):
                journal_size = self._journal.path.stat().st_size
        try:
            with self._lock:
                evicted = list(self._files)[: max(0, len(self._files) + 1 - self._max)]
                if self._journal is not None:
                    payload: dict[str, Any] = {"file": self._meta(rec)}
                    if evicted:
                        payload["evicted"] = evicted
                    self._journal.append(payload)
                for old_id in evicted:
                    self._files.pop(old_id, None)
                self._files[rec.file_id] = rec
                self._files.move_to_end(rec.file_id)
        except Exception:
            if self._dir is not None:
                # Before-write failures leave an orphan blob and are safe to
                # clean. If the journal length changed, retain it: the append
                # may be durable or torn and repair needs the referenced bytes.
                unchanged = self._journal is None
                if self._journal is not None:
                    try:
                        unchanged = self._journal.path.stat().st_size == journal_size
                    except OSError:
                        unchanged = False
                if unchanged:
                    self._blob(rec.file_id).unlink(missing_ok=True)
            raise
        if self._dir is not None:
            for fid in evicted:
                self._blob(fid).unlink(missing_ok=True)
        return rec

    def get(self, file_id: str) -> _FileRecord | None:
        with self._lock:
            rec = self._files.get(file_id)
            if rec is not None:
                # Avoid an fsync on repeated reads of the current MRU entry.
                # Only an order-changing touch needs durable representation.
                if next(reversed(self._files)) != file_id:
                    if self._journal is not None:
                        self._journal.append({"file_touched": file_id})
                    self._files.move_to_end(file_id)
                return rec.model_copy(deep=True)
            return None

    def list(self) -> builtins.list[_FileRecord]:
        """Newest-first snapshot."""
        with self._lock:
            out = [rec.model_copy(deep=True) for rec in self._files.values()]
        out.reverse()
        return out

    def delete(self, file_id: str) -> _FileRecord | None:
        with self._lock:
            rec = self._files.get(file_id)
            if rec is not None and self._journal is not None:
                self._journal.append({"file_deleted": file_id})
            if rec is not None:
                self._files.pop(file_id)
        if rec is not None and self._dir is not None:
            self._blob(file_id).unlink(missing_ok=True)
        return rec.model_copy(deep=True) if rec is not None else None

    @property
    def max_bytes(self) -> int:
        return self._max_bytes


class _BatchCounts(_Model):
    """OpenAI ``request_counts`` shape."""

    total: int = 0
    completed: int = 0
    failed: int = 0


class _BatchRecord(_Model):
    """A running/finished batch — the fields ``batch_object`` projects."""

    batch_id: str
    input_file_id: str
    endpoint: str
    completion_window: str
    status: str  # validating | in_progress | finalizing | completed | failed | expired | cancelling | cancelled
    created_at: int
    expires_at: int
    metadata: dict[str, str] | None = None
    output_file_id: str | None = None
    error_file_id: str | None = None
    errors: dict[str, Any] | None = None
    in_progress_at: int | None = None
    finalizing_at: int | None = None
    completed_at: int | None = None
    failed_at: int | None = None
    expired_at: int | None = None
    cancelling_at: int | None = None
    cancelled_at: int | None = None
    request_counts: _BatchCounts = Field(default_factory=_BatchCounts)
    # fx1 extension — terminal webhook bookkeeping (the same fields the
    # /harness/* jobs surface); projected onto the batch envelope.
    callback_url: str | None = None
    callback_status: Literal["delivered", "failed"] | None = None
    callback_attempts: int = 0
    callback_error: str | None = None
    _cancel: threading.Event = PrivateAttr(default_factory=threading.Event)
    _lines: builtins.list[dict[str, Any]] = PrivateAttr(default_factory=builtins.list)
    _headers: dict[str, str] = PrivateAttr(default_factory=dict)
    _key_id: str | None = PrivateAttr(default=None)
    # Serializes every lifecycle transition and snapshot.  A plain
    # check-then-write is not sufficient here: expiry/cancel routes and the
    # worker run on different threads and can otherwise overwrite a terminal
    # verdict after observing an older status.
    _state_lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)
    _callback_secret: str | None = PrivateAttr(default=None)
    _callback_fired: bool = PrivateAttr(default=False)
    _callback_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)


class _BatchStore:
    """Bounded LRU store of batches (newest-first listing).

    With a ``JobJournal`` bound (``--state-dir``) every status transition
    is journaled (``put``/``mark``/evict) and boot replays the chain:
    terminal batches return as-was; a batch still mid-flight at the crash
    recovers as ``failed`` with a restart-explaining error — its input
    lines aren't journaled, so nothing is silently re-run.
    ``_callback_secret`` never touches disk, so a recovered batch keeps
    ``callback_url`` for audit but cannot deliver post-restart."""

    def __init__(self, max_entries: int, journal: JobJournal | None = None) -> None:
        self._lock = threading.Lock()
        self._max = max_entries
        self._batches: OrderedDict[str, _BatchRecord] = OrderedDict()
        self._journal = journal
        self.recover_warnings: list[str] = []
        if journal is not None:
            res = journal.replay()
            self.recover_warnings = list(res.warnings)
            now = int(time.time())
            for payload in res.payloads:
                for evict in payload.get("evicted") or ():
                    self._batches.pop(str(evict), None)
                if "batch" not in payload:
                    continue
                batch = _BatchRecord.model_validate(payload["batch"])
                # Signing secrets are not journaled; recovered records never re-deliver.
                batch._callback_fired = True
                self._batches[batch.batch_id] = batch
                self._batches.move_to_end(batch.batch_id)
            for batch in self._batches.values():
                if batch.status not in _BATCH_TERMINAL:
                    batch.status = "failed"
                    batch.failed_at = now
                    batch.errors = {
                        "object": "list",
                        "data": [
                            {
                                "code": "internal_error",
                                "message": "process restarted before the batch "
                                "reached a terminal state",
                            }
                        ],
                    }
            self._compact_locked()

    def _record(self, batch: _BatchRecord) -> dict[str, Any]:
        return {"batch": batch.model_dump(mode="json")}

    def _compact_locked(self) -> None:
        if self._journal is not None:
            self._journal.compact([self._record(b) for b in self._batches.values()])

    def mark(self, batch: _BatchRecord) -> None:
        """Journal a status transition made outside the store (the worker
        mutates ``batch`` in place; this makes each hop durable)."""
        if self._journal is not None:
            with self._lock:
                # A worker may start just before put(), or finish after the
                # bounded store evicts its record.  put() snapshots the first
                # case; ignoring the second prevents a late journal row from
                # resurrecting an evicted batch on restart.
                if self._batches.get(batch.batch_id) is batch:
                    self._journal.append(self._record(batch))

    def put(self, batch: _BatchRecord) -> None:
        with self._lock:
            future_ids = [batch_id for batch_id in self._batches if batch_id != batch.batch_id]
            future_ids.append(batch.batch_id)
            evicted = future_ids[: max(0, len(future_ids) - self._max)]
            if self._journal is not None:
                payload = self._record(batch)
                if evicted:
                    payload["evicted"] = evicted
                self._journal.append(payload)
            # Publish only after the journal accepts the complete transition.
            # Otherwise an fsync failure can expose an unrecoverable batch or
            # evict a healthy record that replay correctly retains.
            self._batches[batch.batch_id] = batch
            self._batches.move_to_end(batch.batch_id)
            for old_id in evicted:
                self._batches.pop(old_id, None)

    def get(self, batch_id: str) -> _BatchRecord | None:
        with self._lock:
            return self._batches.get(batch_id)

    def list(self) -> builtins.list[_BatchRecord]:
        """Newest-first snapshot."""
        with self._lock:
            out = list(self._batches.values())
        out.reverse()
        return out


_BATCH_TERMINAL = frozenset({"completed", "failed", "expired", "cancelled"})


class _AnthropicBatchRecord(_Model):
    """A running/finished Anthropic message batch (``msgbatch_*``).

    ``item_ids`` — every ``custom_id`` in submit order — and
    ``result_lines`` (the JSONL result rows) are journaled fields, not
    private attrs: a recovered batch can still serve ``results`` (errored
    restart rows for whatever never ran) instead of going silent."""

    batch_id: str
    status: Literal["in_progress", "canceling", "ended"] = "in_progress"
    created_at: int
    expires_at: int
    ended_at: int | None = None
    cancel_initiated_at: int | None = None
    request_counts: AnthropicBatchCounts = Field(default_factory=AnthropicBatchCounts)
    item_ids: builtins.list[str] = Field(default_factory=builtins.list)
    result_lines: builtins.list[str] = Field(default_factory=builtins.list)
    callback_url: str | None = None
    callback_status: Literal["delivered", "failed"] | None = None
    callback_attempts: int = 0
    callback_error: str | None = None
    _cancel: threading.Event = PrivateAttr(default_factory=threading.Event)
    _items: builtins.list[AnthropicBatchItem] = PrivateAttr(default_factory=builtins.list)
    _headers: dict[str, str] = PrivateAttr(default_factory=dict)
    _key_id: str | None = PrivateAttr(default=None)
    _callback_secret: str | None = PrivateAttr(default=None)
    _callback_fired: bool = PrivateAttr(default=False)
    _callback_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    # Covers result rows and every lifecycle transition.  Reentrancy lets
    # the shared finish path be called by a route that already stabilized
    # the record.
    _row_lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)


_ABATCH_TERMINAL = frozenset({"ended"})


class _AnthropicBatchStore:
    """Bounded LRU store of Anthropic message batches (newest-first).

    Same journal contract as :class:`_BatchStore` (``abatches.jsonl`` under
    ``--state-dir``): terminal batches return as-was; a batch still
    mid-flight at the crash recovers ``ended`` with every unfinished
    ``custom_id`` surfaced as an ``errored`` restart row — result rows are
    journaled, so what did finish still serves."""

    def __init__(self, max_entries: int, journal: JobJournal | None = None) -> None:
        self._lock = threading.Lock()
        self._max = max_entries
        self._batches: OrderedDict[str, _AnthropicBatchRecord] = OrderedDict()
        self._journal = journal
        self.recover_warnings: list[str] = []
        if journal is not None:
            res = journal.replay()
            self.recover_warnings = list(res.warnings)
            now = int(time.time())
            for payload in res.payloads:
                for evict in payload.get("evicted") or ():
                    self._batches.pop(str(evict), None)
                if "batch" not in payload:
                    continue
                batch = _AnthropicBatchRecord.model_validate(payload["batch"])
                # Signing secrets are not journaled; recovered records never re-deliver.
                batch._callback_fired = True
                self._batches[batch.batch_id] = batch
                self._batches.move_to_end(batch.batch_id)
            for batch in self._batches.values():
                if batch.status != "ended":
                    self._recover_unfinished(batch, now)
            self._compact_locked()

    @staticmethod
    def _recover_unfinished(batch: _AnthropicBatchRecord, now: int) -> None:
        """End a batch interrupted by a restart — every item without a
        journaled result row lands an ``errored`` row explaining the
        restart (nothing is silently re-run or silently dropped)."""
        done_ids = {
            json.loads(line)["custom_id"] for line in batch.result_lines if isinstance(line, str)
        }
        for cid in batch.item_ids:
            if cid in done_ids:
                continue
            batch.result_lines.append(
                json.dumps(
                    anthropic_batch_result(
                        cid,
                        {
                            "type": "errored",
                            "error": anthropic_error_body(
                                "process restarted before this request ran",
                                500,
                            )["error"],
                        },
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        batch.request_counts = AnthropicBatchCounts(
            succeeded=sum(1 for ln in batch.result_lines if '"succeeded"' in ln),
            errored=sum(1 for ln in batch.result_lines if '"errored"' in ln),
        )
        batch.status = "ended"
        batch.ended_at = now

    def _record(self, batch: _AnthropicBatchRecord) -> dict[str, Any]:
        return {"batch": batch.model_dump(mode="json")}

    def _compact_locked(self) -> None:
        if self._journal is not None:
            self._journal.compact([self._record(b) for b in self._batches.values()])

    def mark(self, batch: _AnthropicBatchRecord) -> None:
        """Journal a status transition made outside the store (the worker
        mutates ``batch`` in place; this makes each hop durable)."""
        if self._journal is not None:
            with self._lock:
                # A terminal webhook may finish while DELETE removes the
                # batch.  Whichever operation obtains the store lock first
                # wins; a late mark after delete is ignored so replay cannot
                # resurrect the tombstone.
                if self._batches.get(batch.batch_id) is batch:
                    self._journal.append(self._record(batch))

    def put(self, batch: _AnthropicBatchRecord) -> None:
        with self._lock:
            future_ids = [batch_id for batch_id in self._batches if batch_id != batch.batch_id]
            future_ids.append(batch.batch_id)
            evicted = future_ids[: max(0, len(future_ids) - self._max)]
            if self._journal is not None:
                payload = self._record(batch)
                if evicted:
                    payload["evicted"] = evicted
                self._journal.append(payload)
            self._batches[batch.batch_id] = batch
            self._batches.move_to_end(batch.batch_id)
            for old_id in evicted:
                self._batches.pop(old_id, None)

    def get(self, batch_id: str) -> _AnthropicBatchRecord | None:
        with self._lock:
            return self._batches.get(batch_id)

    def delete(self, batch_id: str) -> _AnthropicBatchRecord | None:
        with self._lock:
            batch = self._batches.pop(batch_id, None)
            if batch is not None:
                self._compact_locked()
            return batch

    def list(self) -> builtins.list[_AnthropicBatchRecord]:
        """Newest-first snapshot."""
        with self._lock:
            out = list(self._batches.values())
        out.reverse()
        return out


def _backend_configured() -> dict[str, bool]:
    """Presence-of-credentials flags only — values never leave the process."""
    checkpoint_env = os.environ.get("FX1_CHECKPOINT_DIR", "")
    return {
        "hosted_k3": bool(os.environ.get("MOONSHOT_API_KEY")),
        "byok": all(
            os.environ.get(name) for name in (BYOK_BASE_URL_ENV, BYOK_API_KEY_ENV, BYOK_MODEL_ENV)
        ),
        "local_fx1": bool(checkpoint_env)
        and (Path(checkpoint_env) / "modelcard.json").is_file()
        and bool(os.environ.get(LOCAL_SERVE_URL_ENV) or os.environ.get(LOCAL_SERVE_CMD_ENV)),
    }


_RID_OK = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _request_id(raw: str | None) -> str:
    """Echo a well-formed client request id; anything else mints a fresh one
    (untrusted headers never reach the response unparsed)."""
    if raw is not None and _RID_OK.fullmatch(raw):
        return raw
    return uuid.uuid4().hex


def _is_anthropic_path(path: str) -> bool:
    """The Anthropic-grammar surface — ``/v1/messages`` and everything
    beneath it (count_tokens, message batches)."""
    return path == "/v1/messages" or path.startswith("/v1/messages/")


def _is_anthropic_surface(request: Request) -> bool:
    """Requests answered in Anthropic's dialect — the /v1/messages tree,
    plus any /v1/* path addressed with an ``anthropic-version`` header
    (the dual-grammar routes, e.g. model listing). The header is only
    meaningful on the OpenAI surface: an ``anthropic-version`` header on a
    non-/v1 path (ops endpoints, harness control) must not upgrade the
    answer to Anthropic's dialect."""
    return _is_anthropic_path(request.url.path) or (
        is_openai_path(request.url.path) and "anthropic-version" in request.headers
    )


# Statuses the stock anthropic SDK retries by default — x-should-retry
# confirms them — and the ones its default gets wrong here: 409
# (idempotency-key conflict) and 501 (unimplemented knob) are terminal,
# never retried. Every other status omits the header.
_ANTHROPIC_RETRY_TRUE = frozenset({408, 429, 500, 502, 503, 504, 529})
_ANTHROPIC_RETRY_FALSE = frozenset({409, 501})


def _anthropic_budget_headers(key_store: ApiKeyStore, key_id: str | None) -> dict[str, str]:
    """Anthropic's standing rate-limit headers for a managed key with a
    declared rpm window — the requests family only (there is no token
    window to report); empty for env/loopback auth or unwindowed keys
    (no false scarcity, same honesty rule as the X-RateLimit-* family)."""
    if key_id in (None, "env"):
        return {}
    ws = key_store.window_state(key_id)
    if ws is None:
        return {}
    out = {
        "anthropic-ratelimit-requests-limit": str(ws[0]),
        "anthropic-ratelimit-requests-remaining": str(ws[1]),
    }
    reset = _rfc3339(time.time() + ws[2])
    if reset is not None:
        # Anthropic's convention: an instant, not a countdown
        out["anthropic-ratelimit-requests-reset"] = reset
    return out


logger = logging.getLogger("fx1.serve.api")
if not logger.handlers:  # embedders may still attach their own handlers
    _log_handler = logging.StreamHandler()
    _log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s fx1-api %(message)s"))
    logger.addHandler(_log_handler)
    logger.setLevel(logging.INFO)


def _finish(request: Request, request_id: str, response: Any, started: float) -> Any:
    """Single post-processing tail for every response — security headers,
    request-id echo, and one structured access line."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    # no-store is the default posture; a route that deliberately declares a
    # caching policy (immutable content-addressed bytes) wins over it.
    if "cache-control" not in response.headers:
        response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Fx1-Api-Version"] = API_VERSION
    if _is_anthropic_surface(request):
        # the stock anthropic SDK reads `request-id` (same id, its name)
        # and `x-should-retry` where its defaults disagree with our
        # terminal shapes
        response.headers["request-id"] = request_id
        status = response.status_code
        if status in _ANTHROPIC_RETRY_TRUE:
            response.headers.setdefault("x-should-retry", "true")
        elif status in _ANTHROPIC_RETRY_FALSE:
            response.headers.setdefault("x-should-retry", "false")
    if is_openai_path(request.url.path):
        # OpenAI's api-version response header — the stock SDK + proxies
        # log it for compat debugging on every /v1 call
        response.headers["openai-version"] = API_VERSION
    elapsed_ms = (time.monotonic() - started) * 1000
    # OpenAI's server-side timing header — every response carries it so
    # clients can split transport vs processing without trusting the log
    response.headers["Openai-Processing-Ms"] = str(int(elapsed_ms))
    request.app.state.metrics.record(response.status_code)
    logger.info(
        "request method=%s path=%s status=%d elapsed_ms=%.1f rid=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        request_id,
    )
    return response


class _RateLimiter:
    """Per-client-host token bucket: ``rps`` refill, one-second burst, LRU cap."""

    _clock_box: dict[str, float] | None = None

    def __init__(self, rps: float, max_keys: int = _RATE_LIMIT_KEYS_MAX) -> None:
        self.rps = rps
        self.capacity = max(1.0, rps)
        self.max_keys = max_keys
        self._lock = threading.Lock()
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        box = type(self)._clock_box
        self._clock = time.monotonic if box is None else (lambda: float(box["t"]))

    def allow(self, identity: str) -> tuple[float, float]:
        """Return ``(wait, remaining)`` after consuming one token."""
        now = self._clock()
        with self._lock:
            entry = self._buckets.pop(identity, None)
            tokens, ts = entry if entry is not None else (self.capacity, now)
            tokens = min(self.capacity, tokens + self.rps * (now - ts))
            if tokens >= 1.0:
                self._buckets[identity] = (tokens - 1.0, now)
                return 0.0, max(0.0, tokens - 1.0)
            wait = (1.0 - tokens) / self.rps
            self._buckets[identity] = (tokens, now)
            while len(self._buckets) > self.max_keys:
                self._buckets.popitem(last=False)
            return wait, 0.0


class _BackendBreaker:
    """Consecutive-fault circuit breaker keyed on backend name.

    `threshold` consecutive call faults open the circuit for
    `cooldown_s`: calls fail fast (503 + Retry-After) without touching the
    backend or holding an inflight slot. When the window lapses a single
    half-open probe is admitted; success closes the circuit, failure
    re-opens it for a fresh window. A probe in flight fast-fails other
    callers — only the probe touches the backend. Configuration faults
    (501, honesty-gate refusals) never count: the breaker measures
    backend health, not model output.
    """

    def __init__(self, threshold: int, cooldown_s: float) -> None:
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._lock = threading.Lock()
        self._fails: dict[str, int] = {}
        self._opened: dict[str, float] = {}
        self._probing: set[str] = set()

    def check(self, backend: str) -> float:
        """Admit or refuse a call: 0.0 proceeds, >0 fast-fails for that
        many seconds of remaining cooldown."""
        with self._lock:
            opened = self._opened.get(backend)
            if opened is None:
                return 0.0
            remaining = self.cooldown_s - (time.monotonic() - opened)
            if remaining <= 0.0:
                if backend in self._probing:
                    return 0.01
                self._probing.add(backend)
                return 0.0
            return remaining

    def report(self, backend: str, ok: bool) -> None:
        """Record the admitted call's outcome."""
        with self._lock:
            self._probing.discard(backend)
            if ok:
                self._fails.pop(backend, None)
                self._opened.pop(backend, None)
                return
            fails = self._fails.get(backend, 0) + 1
            self._fails[backend] = fails
            if fails >= self.threshold:
                self._opened[backend] = time.monotonic()

    def state(self, backend: str) -> tuple[bool, float, int]:
        """``(circuit_open, cooldown_remaining_s, consecutive_failures)``."""
        with self._lock:
            opened = self._opened.get(backend)
            remaining = (
                max(0.0, self.cooldown_s - (time.monotonic() - opened))
                if opened is not None
                else 0.0
            )
            return opened is not None and remaining > 0.0, remaining, self._fails.get(backend, 0)


_COMPLETION_LOG_MAX = 256


class _CompletionLog:
    """Bounded in-process ring of per-call completion records. Hashes of
    prompt/output are stored, never the content — the log is evidence,
    not a transcript. Probes stay out of it: they already carry their own
    surface (``probe:<name>`` metrics + the status-cache verdicts)."""

    def __init__(
        self,
        cap: int = _COMPLETION_LOG_MAX,
        on_record: Callable[[CompletionRecord], None] | None = None,
    ) -> None:
        self._cap = cap
        self._lock = threading.Lock()
        self._items: dict[str, CompletionRecord] = {}
        self._dropped = 0
        self._on_record = on_record

    def append(self, rec: CompletionRecord) -> None:
        # Attribute to the authenticated credential: the key fingerprint
        # rides the request contextvar — record-stamping code never sees
        # the Request. Records that set ``key_id`` explicitly (batch
        # workers run in a thread pool where contextvars don't propagate)
        # keep their own.
        if rec.key_id is None:
            ctx_key = _REQUEST_KEY_ID.get()
            if ctx_key is not None:
                rec = rec.model_copy(update={"key_id": ctx_key})
        with self._lock:
            self._items[rec.completion_id] = rec
            while len(self._items) > self._cap:
                self._items.pop(next(iter(self._items)))
                self._dropped += 1
        # Post-commit hook: the wire folds provider-reported usage into
        # the credential's token-budget meter here, so every surface that
        # records a call charges identically.
        if self._on_record is not None:
            self._on_record(rec)

    @property
    def cap(self) -> int:
        return self._cap

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def get(self, completion_id: str) -> CompletionRecord | None:
        with self._lock:
            return self._items.get(completion_id)

    def latest(self, limit: int, backend: str | None) -> list[CompletionRecord]:
        with self._lock:
            items = sorted(self._items.values(), key=lambda r: r.at, reverse=True)
        if backend is not None:
            items = [r for r in items if r.backend == backend]
        return items[:limit]

    def all(self, backend: str | None = None) -> list[CompletionRecord]:
        """Every retained record — the aggregation view (no limit)."""
        return self.latest(self._cap, backend)


def _render_prometheus(snap: MetricsResponse, job_counts: dict[str, int]) -> str:
    """Render the ops snapshot as Prometheus text exposition (format
    0.0.4) — ``GET /metrics`` stays JSON by default; ``?format=prom``
    or an ``Accept: text/plain``/OpenMetrics header selects this view.
    Label values come from integer status codes and the fixed
    ``_JOB_STATUSES`` set, so quoting is mechanical, not user data."""

    def _label(v: str) -> str:
        return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    lines = [
        "# HELP fx1_uptime_seconds Seconds since the harness API process started.",
        "# TYPE fx1_uptime_seconds gauge",
        f"fx1_uptime_seconds {snap.uptime_s:.6f}",
        "# HELP fx1_requests_total Harness API requests by response status code.",
        "# TYPE fx1_requests_total counter",
    ]
    for status in sorted(snap.by_status):
        lines.append(f'fx1_requests_total{{status="{_label(status)}"}} {snap.by_status[status]}')
    lines += [
        "# HELP fx1_errors_total Harness API responses with status >= 400.",
        "# TYPE fx1_errors_total counter",
        f"fx1_errors_total {snap.errors_total}",
        "# HELP fx1_rate_limited_total Requests denied by the rate limiter.",
        "# TYPE fx1_rate_limited_total counter",
        f"fx1_rate_limited_total {snap.rate_limited_total}",
        "# HELP fx1_inflight Requests currently executing.",
        "# TYPE fx1_inflight gauge",
        f"fx1_inflight {snap.inflight}",
        "# HELP fx1_inflight_watermark High-water mark of concurrent requests.",
        "# TYPE fx1_inflight_watermark gauge",
        f"fx1_inflight_watermark {snap.inflight_watermark}",
        "# HELP fx1_max_inflight Configured concurrency bound.",
        "# TYPE fx1_max_inflight gauge",
        f"fx1_max_inflight {snap.max_inflight}",
        "# HELP fx1_draining 1 while the drain latch is set.",
        "# TYPE fx1_draining gauge",
        f"fx1_draining {1 if snap.draining else 0}",
        "# HELP fx1_jobs Harness jobs by lifecycle status.",
        "# TYPE fx1_jobs gauge",
    ]
    for status in _JOB_STATUSES:
        lines.append(f'fx1_jobs{{status="{status}"}} {job_counts.get(status, 0)}')
    lines += [
        "# HELP fx1_complete_total Model calls attempted, by backend and outcome.",
        "# TYPE fx1_complete_total counter",
        "# HELP fx1_complete_latency_ms Model-call latency histogram (ms), per backend.",
        "# TYPE fx1_complete_latency_ms histogram",
    ]
    for b in sorted(snap.complete):
        c = snap.complete[b]
        lb = _label(b)
        lines.append(f'fx1_complete_total{{backend="{lb}",outcome="ok"}} {c.ok}')
        lines.append(f'fx1_complete_total{{backend="{lb}",outcome="error"}} {c.error}')
        for le, n in c.latency_buckets.items():
            lines.append(f'fx1_complete_latency_ms_bucket{{backend="{lb}",le="{le}"}} {n}')
        lines.append(f'fx1_complete_latency_ms_sum{{backend="{lb}"}} {c.latency_sum_ms}')
        lines.append(f'fx1_complete_latency_ms_count{{backend="{lb}"}} {c.latency_count}')
    lines += [
        "# HELP fx1_complete_tokens_total Backend-reported token sums, per backend and kind.",
        "# TYPE fx1_complete_tokens_total counter",
        "# HELP fx1_complete_usage_calls_total Model calls that carried a usage report.",
        "# TYPE fx1_complete_usage_calls_total counter",
    ]
    for b in sorted(snap.complete):
        c = snap.complete[b]
        lb = _label(b)
        lines.append(f'fx1_complete_tokens_total{{backend="{lb}",kind="prompt"}} {c.prompt_tokens}')
        lines.append(
            f'fx1_complete_tokens_total{{backend="{lb}",kind="completion"}} {c.completion_tokens}'
        )
        lines.append(f'fx1_complete_tokens_total{{backend="{lb}",kind="total"}} {c.total_tokens}')
        lines.append(f'fx1_complete_usage_calls_total{{backend="{lb}"}} {c.usage_calls}')
    return "\n".join(lines) + "\n"


def _close_backend(backend: Any) -> None:
    closer = getattr(backend, "close", None)
    if callable(closer):
        closer()


def _mount_complete_routes(  # noqa: C901 — eval submission shares the chain/job surface
    app: FastAPI,
    *,
    slot: Callable[..., Iterator[None]],
    resolve_backend: Callable[[str, str | None, dict[str, str] | None, float | None], Any],
    sse_keepalive_s: float,
    complete_idem_store: _IdemStore[CompleteResponse],
    complete_batch_idem_store: _IdemStore[CompleteBatchResponse],
    openai_idem_store: _IdemStore[_OpenAIIdemRecord],
    anthropic_idem_store: _IdemStore[_OpenAIIdemRecord],
    legacy_idem_store: _IdemStore[_OpenAIIdemRecord],
    breaker: _BackendBreaker | None,
    receipt_index: _ReceiptIndex,
    metrics: _Metrics,
    probe_cache: dict[str, BackendProbeVerdict],
    probe_lock: threading.Lock,
    completion_log: _CompletionLog,
    eval_store: EvalStore,
    inflight: threading.BoundedSemaphore,
    jobs_executor: ThreadPoolExecutor,
    file_store: _FileStore,
    batch_store: _BatchStore,
    abatch_store: _AnthropicBatchStore,
    upload_store: UploadStore,
    upload_idem_store: _IdemStore[_JsonIdemRecord],
    envelope_store: OpenAIEnvelopeStore,
    batch_line_max: int,
    file_bytes_max: int,
    ft_store: FTJobStore,
    ft_runner: FTJobRunner,
    ft_dir: Path,
    bg_cancel: dict[str, threading.Event],
    conv_store: OpenAIEnvelopeStore,
    eval_spec_store: EvalSpecStore,
    vs_store: VectorStoreStore,
    key_store: ApiKeyStore,
) -> None:
    """Complete routes (sync / SSE stream / batch) + eval submissions —
    extracted from ``create_app`` to keep its branch complexity under the
    ruff cap. Evals share the complete chain resolution and the jobs
    executor's slot contract."""

    # Idempotency-Key claims: the dep holds the store's per-key mutex for
    # the whole handler span — lookup, model spend, and replay-record
    # publication stay atomic against a retry racing the same key. The
    # claim dep is declared before ``slot`` so waiters never queue on a
    # worker slot while holding one.
    _complete_idem_claim = _idem_claim_dep(complete_idem_store)
    _complete_batch_idem_claim = _idem_claim_dep(complete_batch_idem_store)
    _openai_idem_claim = _idem_claim_dep(openai_idem_store)
    _anthropic_idem_claim = _idem_claim_dep(anthropic_idem_store)
    _legacy_idem_claim = _idem_claim_dep(legacy_idem_store)
    _file_idem_claim = _idem_claim_dep(upload_idem_store)
    _upload_idem_claim = _idem_claim_dep(upload_idem_store)
    _vs_idem_claim = _idem_claim_dep(vs_store)
    _ft_idem_claim = _idem_claim_dep(ft_store)

    def _breaker_admit(name: str) -> None:
        """Fast-fail while the backend's circuit is open — the call never
        burns an inflight slot waiting on a dead endpoint."""
        if breaker is None:
            return
        wait = breaker.check(name)
        if wait > 0:
            raise ApiError(
                503,
                f"backend {name!r} circuit open — retry in {wait:.1f}s",
                code="backend_unavailable",
                headers={"Retry-After": str(max(1, math.ceil(wait)))},
            )

    def _check_citations(hashes: list[str] | None) -> None:
        """Cited evidence must resolve against the mounted receipt store:
        a completion may not footnote a receipt the server cannot produce.
        Runs before the breaker and the backend call — a bogus citation is a
        422 client fault and never burns a slot or trips the circuit. When no
        store is mounted, citations stay advisory (nothing to check against)."""
        if not hashes or not receipt_index.available():
            return
        missing = [
            h for h in hashes if not _SHA256_HEX.fullmatch(h) or receipt_index.lookup(h) is None
        ]
        if missing:
            raise ApiError(
                422,
                f"unresolvable receipt citations: {missing}",
                code="receipt_not_found",
            )

    def _completion_receipt_sha(completion_id: str | None) -> str | None:
        """Digest of the logged call's sealed ``fx1_completion_record.v1``
        document — the response-side twin of ``X-Fx1-Completion-Id``, so a
        client can pin the evidence without a second fetch. None when the
        record is gone (bounded log evicts)."""
        if completion_id is None:
            return None
        rec = completion_log.get(completion_id)
        if rec is None:
            return None
        from fx1.serve.ops_receipt import (  # noqa: PLC0415
            completion_record_receipt,
        )

        return str(completion_record_receipt(rec.model_dump(mode="json"))["receipt_sha256"])

    def _completion_headers(cid: str, *, replay: bool = False) -> dict[str, str]:
        """The completion-surface response headers — the call's log id
        plus the sealed receipt digest when the record still lives in the
        bounded log. ``replay`` marks a byte-identical idempotent hit."""
        out = {"X-Fx1-Completion-Id": cid}
        if replay:
            out["X-Fx1-Idempotent-Replay"] = "true"
        rsha = _completion_receipt_sha(cid)
        if rsha is not None:
            out["X-Fx1-Receipt-Sha256"] = rsha
        return out

    def _resolve_candidate(
        name: str,
        body: CompleteRequest | CompleteBatchRequest | EvalSubmitRequest | EmbedRequest,
    ) -> Any:
        """Resolve one chain link — per-link kwargs: the byok override binds
        only a 'byok' link, checkpoint_dir only a 'local_fx1' link.
        ``resolve_backend`` is create_app's normalized request resolver
        (``_resolve_request_backend``) — resolver faults already arrive as
        the wire map's errors."""
        return resolve_backend(
            name,
            body.checkpoint_dir if name == "local_fx1" else None,
            body.byok.model_dump() if name == "byok" and body.byok is not None else None,
            body.timeout_s,
        )

    def _resolve_chain(
        body: CompleteRequest | CompleteBatchRequest | EvalSubmitRequest | EmbedRequest,
        *,
        out: list[BackendAttempt] | None = None,
    ) -> tuple[str, Any, list[BackendAttempt]]:
        """First chain link that admits + resolves serves; a 503
        (unconfigured / unavailable / circuit open) records the attempt and
        moves on. Any other error is a request fault and aborts. ``out``
        shares the attempts list with the caller so a dead chain still seals
        which links were tried on the failed record."""
        attempts: list[BackendAttempt] = out if out is not None else []
        last: ApiError | None = None
        for cand in [body.backend, *body.fallbacks]:
            key = _breaker_key_name(cand, body.byok)
            try:
                _breaker_admit(key)
                backend = _resolve_candidate(cand, body)
            except ApiError as exc:
                if exc.status_code == 503:
                    attempts.append(
                        BackendAttempt(backend=cand, ok=False, error_class="backend_unavailable")
                    )
                    if breaker is not None:
                        breaker.report(key, False)
                    last = exc
                    continue
                raise
            attempts.append(BackendAttempt(backend=cand, ok=True))
            return cand, backend, attempts
        assert last is not None  # noqa: S101 — every link failed retriably
        raise last

    def _submit_eval(
        body: EvalSubmitRequest,
        idempotency_key: str | None,
        *,
        eval_spec: str | None = None,
        eval_model: str | None = None,
    ) -> EvalSubmitResponse:
        """Serialize lookup -> create -> record insert under the key's
        claim (the job contract's claim_lock, applied to the eval
        suites): a retried submit can never slide between an in-flight
        twin's lookup and its replay record."""
        key = _idem_key(idempotency_key)
        skey = _idem_scope(key)
        with eval_store.claim_lock(skey):
            return _submit_eval_claimed(body, skey, eval_spec=eval_spec, eval_model=eval_model)

    def _submit_eval_claimed(
        body: EvalSubmitRequest,
        key: str | None,
        *,
        eval_spec: str | None = None,
        eval_model: str | None = None,
    ) -> EvalSubmitResponse:
        """Eval submission core — the job contract (idempotency lookup ->
        drain check -> slot admission -> background execution) applied to
        the eval suites under an already-held claim. The slot is held for
        the eval's lifetime and released by the worker, so evals queue no
        deeper than ``max_inflight``. ``key`` is the final credential-
        scoped store key (already normalized + bounded by the caller).
        ``eval_spec``/``eval_model`` bind a /v1/evals run at construction
        so every journal entry and the terminal callback carry the
        binding — a fast eval can never fire its webhook before the
        binding lands."""
        body_fp = _body_fp(body)
        if key is not None:
            entry = eval_store.get_key(key)
            if entry is not None:
                fp, eval_id = entry
                if fp != body_fp:
                    raise ApiError(
                        409,
                        "Idempotency-Key reuse with a different request body",
                        code="idempotency_conflict",
                    )
                rec = eval_store.get(eval_id)
                if rec is not None:
                    return EvalSubmitResponse(eval_id=eval_id, status=rec.status, replayed=True)
        _drain_refusal(metrics)
        if not inflight.acquire(blocking=False):
            raise ApiError(
                503,
                f"harness at max_inflight={metrics.max_inflight} — retry later",
                code="over_capacity",
                headers={"Retry-After": "1"},
            )
        metrics.acquire()
        record = EvalRecord(
            eval_id=uuid.uuid4().hex,
            suite=body.suite,
            backend=body.backend,
            seed=body.seed,
            status="queued",
            created_at=time.time(),
            sampling=EVAL_SAMPLING.body_fields(),
            callback_url=body.callback_url,
            eval_spec=eval_spec,
            eval_model=eval_model,
        )
        record._callback_secret = body.callback_secret

        def _exec() -> None:
            chain_att: list[BackendAttempt] = []
            backend: Any = None
            judge_obj: Any = None
            finalize = False
            try:
                # The claim itself journals and can fail. It belongs to
                # the same exception and capacity boundary as model work.
                if eval_store.start(record.eval_id) is None:
                    return
                finalize = True
                name, backend, _ = _resolve_chain(body, out=chain_att)
                record.backend = name
                record.attempts = [a.model_dump(mode="json") for a in chain_att]
                model = metered_model(
                    backend,
                    metric_key=f"eval:{body.suite}:{name}",
                    record=metrics.record_complete,
                )
                judge = None
                if body.judge_backend is not None:
                    judge_key = _breaker_key_name(body.judge_backend, body.judge_byok)
                    _breaker_admit(judge_key)
                    judge_obj = resolve_backend(
                        body.judge_backend,
                        None,
                        (
                            body.judge_byok.model_dump()
                            if body.judge_backend == "byok" and body.judge_byok is not None
                            else None
                        ),
                        body.timeout_s,
                    )
                    judge = metered_model(
                        judge_obj,
                        metric_key=f"eval:{body.suite}:judge:{body.judge_backend}",
                        record=metrics.record_complete,
                    )
                run_eval_record(record, model=model, judge=judge)
            except ApiError as exc:
                finalize = True
                record.error = f"{exc.status_code}: {exc.detail}"
                record.attempts = [a.model_dump(mode="json") for a in chain_att]
                record.status = "failed"
                record.finished_at = time.time()
            except Exception as exc:  # noqa: BLE001 — worker faults land in the record
                finalize = True
                record.error = f"{type(exc).__name__}: {exc}"
                record.attempts = [a.model_dump(mode="json") for a in chain_att]
                record.status = "failed"
                record.finished_at = time.time()
            finally:
                try:
                    try:
                        if judge_obj is not backend:
                            _close_backend(judge_obj)
                    finally:
                        _close_backend(backend)
                finally:
                    try:
                        if finalize:
                            try:
                                _deliver_callback(record)
                            finally:
                                eval_store.mark(record)
                    finally:
                        # Cleanup faults must not skip the remaining
                        # resources, durable transition, or capacity release.
                        metrics.release()
                        inflight.release()

        # Put before the executor hand-off: the worker's atomic start() claim
        # can only ever lose to a cancel that already landed — never to a
        # store that doesn't know the record yet.
        handed_off = False
        try:
            eval_store.put(record, key, body_fp)
            try:
                jobs_executor.submit(_exec)
            except RuntimeError as exc:  # executor gone (shutdown race)
                raise ApiError(503, "job executor unavailable", code="over_capacity") from exc
            handed_off = True
        finally:
            if not handed_off:
                try:
                    eval_store.delete(record.eval_id)
                finally:
                    metrics.release()
                    inflight.release()
        return EvalSubmitResponse(eval_id=record.eval_id, status=record.status, replayed=False)

    @app.post(
        "/harness/evals",
        response_model=EvalSubmitResponse,
        status_code=202,
        tags=["evals"],
        operation_id="submit_eval",
    )
    def submit_eval(
        body: EvalSubmitRequest,
        response: Response,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> EvalSubmitResponse:
        """Submit an eval suite against a backend chain — same
        drain/cap/Idempotency-Key contract as job submission. The record
        carries the suite, seed, serving backend, chain attempts, decode
        pin, and the serialized report once terminal; export it sealed at
        ``GET /harness/evals/{eval_id}/receipt``."""
        out = _submit_eval(body, idempotency_key)
        response.headers["Location"] = f"/harness/evals/{out.eval_id}"
        return out

    @app.get(
        "/harness/evals",
        response_model=EvalListResponse,
        tags=["evals"],
        operation_id="list_evals",
    )
    def list_evals(
        status: Literal["queued", "running", "succeeded", "failed", "cancelled"] | None = None,
        suite: EvalSuiteName | None = None,
        limit: int = Query(default=100, ge=1, le=256),
    ) -> EvalListResponse:
        """Newest-first eval inventory, filterable by status and suite."""
        records, total = eval_store.list_records(status=status, suite=suite, limit=limit)
        return EvalListResponse(records=records, total=total)

    @app.get(
        "/harness/evals/{eval_id}",
        response_model=EvalRecord,
        tags=["evals"],
        operation_id="get_eval",
    )
    def get_eval(eval_id: str) -> EvalRecord:
        rec = eval_store.get(eval_id)
        if rec is None:
            raise ApiError(404, f"unknown eval_id {eval_id!r}")
        return rec

    @app.get("/harness/evals/{eval_id}/receipt", tags=["evals"], operation_id="eval_receipt")
    def eval_receipt(eval_id: str) -> dict[str, Any]:
        """Export the eval record as a sealed ``fx1_eval_record.v1``
        document — terminal records only: a still-running eval's receipt
        would seal a mutable report. Verify with ``POST /receipts/verify``
        or the SDK."""
        rec = eval_store.get(eval_id)
        if rec is None:
            raise ApiError(404, f"unknown eval_id {eval_id!r}")
        if rec.status not in _TERMINAL_JOB_STATUS:
            raise ApiError(
                409,
                f"eval {eval_id!r} is {rec.status} — receipts export on terminal records only",
                code="eval_not_terminal",
            )
        return eval_record_receipt(rec.model_dump(mode="json"))

    @app.get(
        "/harness/evals/{eval_id}/diff/{candidate_id}",
        tags=["evals"],
        operation_id="diff_evals",
    )
    def diff_evals(eval_id: str, candidate_id: str) -> EvalDiff:
        """Promotion-gate primitive: diff two terminal eval records —
        task-level pass/fail transitions, the honesty-gate move, and
        ``by_kind`` counter deltas. ``comparable`` requires the same
        suite over the same eval bank (``eval_bank_sha256``); a
        cross-bank diff is served but reads ``verdict='unknown'``."""
        base = eval_store.get(eval_id)
        if base is None:
            raise ApiError(404, f"unknown eval_id {eval_id!r}")
        cand = eval_store.get(candidate_id)
        if cand is None:
            raise ApiError(404, f"unknown eval_id {candidate_id!r}")
        for rec in (base, cand):
            if rec.status not in _TERMINAL_JOB_STATUS or rec.report is None:
                raise ApiError(
                    409,
                    f"eval {rec.eval_id!r} is {rec.status} — diffs need terminal records with reports",
                    code="eval_not_terminal",
                )
        return diff_eval_records(base, cand)

    @app.delete(
        "/harness/evals/{eval_id}",
        tags=["evals"],
        operation_id="cancel_eval",
    )
    def cancel_eval(eval_id: str) -> EvalRecord:
        """Cooperative cancel: a queued eval flips to 'cancelled' and its
        executor slot frees on dequeue. Running and terminal evals 409 —
        suite runners have no mid-run kill handle."""
        rec, outcome = eval_store.cancel(eval_id)
        if rec is None:
            raise ApiError(404, f"unknown eval_id {eval_id!r}")
        if outcome != "cancelled":
            raise ApiError(409, f"eval {eval_id!r} is {outcome}")
        _deliver_callback(rec)  # cancelled is terminal — fire the webhook
        return rec

    # ---- /v1/evals — the OpenAI Evals-shaped spec/run surface ----------

    def _spec_or_404(eval_id: str) -> EvalSpec:
        spec = eval_spec_store.get(eval_id)
        if spec is None:
            raise ApiError(404, f"eval '{eval_id}' not found", code="eval_not_found")
        return spec

    def _run_or_404(eval_id: str, run_id: str) -> EvalRecord:
        bare = run_id.removeprefix("evalrun_")
        rec = eval_store.get(bare)
        if rec is None or rec.eval_spec != eval_id:
            raise ApiError(404, f"run '{run_id}' not found", code="run_not_found")
        return rec

    def _run_model_backend(
        model: str,
    ) -> tuple[Literal["hosted_k3", "local_fx1", "byok"], str | None]:
        """``model`` → (backend, checkpoint_dir): link names pass through;
        ``fx1`` is the base checkpoint; ``ft:<name>`` resolves through the
        model-card registry to its checkpoint (unregistered fails closed)."""
        if model in ("hosted_k3", "local_fx1", "byok"):
            return cast(Literal["hosted_k3", "local_fx1", "byok"], model), None
        if model == "fx1":
            return "local_fx1", None
        if model.startswith("ft:"):
            ckpt = ft_store.checkpoint_for(model)
            if ckpt is None:
                raise ApiError(404, f"model '{model}' not found", code="model_not_found")
            return "local_fx1", ckpt
        raise ApiError(400, f"unknown eval model {model!r}", code="invalid_request")

    @app.post(
        "/v1/evals",
        status_code=201,
        operation_id="createEval",
        tags=["evals"],
    )
    def eval_spec_create(body: EvalSpecCreate) -> EvalSpecWire:
        _drain_refusal(metrics)
        spec = EvalSpec(
            spec_id=f"eval_{uuid.uuid4().hex[:24]}",
            name=body.name,
            data_source_config=body.data_source_config.model_dump(exclude_none=True),
            testing_criteria=[c.model_dump(exclude_none=True) for c in body.testing_criteria],
            metadata=body.metadata or {},
            created_at=time.time(),
        )
        eval_spec_store.put(spec)
        return EvalSpecWire.model_validate(spec_wire(spec))

    @app.get("/v1/evals", operation_id="listEvals", tags=["evals"])
    def eval_spec_list(limit: int = 20, after: str | None = None) -> EvalSpecPage:
        if not 1 <= limit <= 100:
            raise ApiError(400, _MSG_LIMIT_RANGE, code="invalid_request")
        try:
            page, more = eval_spec_store.list_specs(limit=limit, after=after)
        except ValueError as exc:
            raise ApiError(400, str(exc), code="invalid_cursor") from exc
        return EvalSpecPage(
            data=[EvalSpecWire.model_validate(spec_wire(s)) for s in page], has_more=more
        )

    @app.get("/v1/evals/{eval_id}", operation_id="getEval", tags=["evals"])
    def eval_spec_get(eval_id: str) -> EvalSpecWire:
        return EvalSpecWire.model_validate(spec_wire(_spec_or_404(eval_id)))

    @app.post(
        "/v1/evals/{eval_id}",
        operation_id="updateEval",
        tags=["evals"],
    )
    def eval_spec_update(eval_id: str, body: EvalSpecUpdate) -> EvalSpecWire:
        _drain_refusal(metrics)
        spec = _spec_or_404(eval_id)
        if body.data_source_config is not None or body.testing_criteria is not None:
            # Hot-reload window: the declared shape may only change while
            # no run binds the spec — a bound run's evidence must not have
            # its criteria shifted under it. Total counts every record
            # bound to the spec (queued/running/terminal all freeze it).
            _page, bound = eval_store.list_records(spec=eval_id, limit=1)
            if bound:
                raise ApiError(
                    409,
                    f"eval '{eval_id}' is bound to {bound} run(s) — "
                    "datasource/criteria are frozen evidence",
                    code="eval_spec_frozen",
                )
            if body.data_source_config is not None:
                spec.data_source_config = body.data_source_config.model_dump(exclude_none=True)
            if body.testing_criteria is not None:
                spec.testing_criteria = [
                    c.model_dump(exclude_none=True) for c in body.testing_criteria
                ]
        if body.name is not None:
            spec.name = body.name
        if body.metadata is not None:
            spec.metadata = body.metadata
        eval_spec_store.update(spec)
        return EvalSpecWire.model_validate(spec_wire(spec))

    @app.delete(
        "/v1/evals/{eval_id}",
        operation_id="deleteEval",
        tags=["evals"],
    )
    def eval_spec_delete(eval_id: str) -> EvalSpecDeleted:
        spec = eval_spec_store.delete(eval_id)
        if spec is None:
            raise ApiError(404, f"eval '{eval_id}' not found", code="eval_not_found")
        return EvalSpecDeleted(id=eval_id)

    @app.post(
        "/v1/evals/{eval_id}/runs",
        response_model=EvalRunObject,
        status_code=201,
        operation_id="createEvalRun",
        tags=["evals"],
    )
    def eval_run_create(
        eval_id: str,
        body: EvalRunCreate,
        response: Response,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> EvalRunObject:
        """Submit a run under the spec: ``model`` resolves to the backend
        chain head, the spec's item_schema supplies the suite knobs, and
        ``data_source.source`` may override them per run. Same capacity
        gates as ``/harness/evals`` (503 over_capacity / draining)."""
        spec = _spec_or_404(eval_id)
        schema = EvalSpecItemSchema.model_validate(spec.data_source_config["item_schema"])
        if body.data_source is not None and body.data_source.source:
            merged = {**schema.model_dump(), **body.data_source.source}
            try:
                schema = EvalSpecItemSchema.model_validate(merged)
            except ValidationError as exc:
                raise ApiError(
                    422, f"data_source.source failed validation: {exc.errors()[0].get('msg')}"
                ) from exc
        backend, ckpt = _run_model_backend(body.model)
        if body.judge_byok is not None and schema.judge_backend != "byok":
            raise ApiError(422, "judge_byok applies only when judge_backend='byok'")
        sub = _request_or_422(
            EvalSubmitRequest,
            {
                "suite": schema.suite,
                "backend": backend,
                "seed": schema.seed,
                "checkpoint_dir": ckpt or schema.checkpoint_dir,
                "byok": body.byok,
                "timeout_s": schema.timeout_s,
                "fallbacks": schema.fallbacks,
                "judge_backend": schema.judge_backend,
                "judge_byok": body.judge_byok,
                "callback_url": body.callback_url,
                "callback_secret": body.callback_secret,
            },
        )
        # The run's dedupe namespace is scoped to the spec — the same
        # Idempotency-Key under a different eval is a different run.
        scoped_key = f"{idempotency_key}:{spec.spec_id}" if idempotency_key else None
        submitted = _submit_eval(sub, scoped_key, eval_spec=spec.spec_id, eval_model=body.model)
        rec = eval_store.get(submitted.eval_id)
        if rec is None:
            raise ApiError(404, f"run '{submitted.eval_id}' evicted")
        response.headers["Location"] = f"/v1/evals/{eval_id}/runs/{rec.eval_id}"
        return EvalRunObject.model_validate(run_wire(rec))

    @app.get(
        "/v1/evals/{eval_id}/runs",
        response_model=EvalRunPage,
        operation_id="listEvalRuns",
        tags=["evals"],
    )
    def eval_run_list(eval_id: str, limit: int = 20, after: str | None = None) -> EvalRunPage:
        _spec_or_404(eval_id)
        if not 1 <= limit <= 100:
            raise ApiError(400, _MSG_LIMIT_RANGE, code="invalid_request")
        records, _total = eval_store.list_records(spec=eval_id)
        if after is not None:
            # The wire hands out run ids as ``evalrun_<id>`` — the cursor
            # round-trips in that shape or bare.
            bare_after = after.removeprefix("evalrun_")
            idx = next((i for i, r in enumerate(records) if r.eval_id == bare_after), None)
            if idx is None:
                raise ApiError(
                    400,
                    f"cursor {after!r} is not a run id under {eval_id!r}",
                    code="invalid_cursor",
                )
            records = records[idx + 1 :]
        page = records[:limit]
        more = len(records) > limit
        return EvalRunPage(
            data=[EvalRunObject.model_validate(run_wire(r)) for r in page[:limit]],
            has_more=more,
        )

    @app.get(
        "/v1/evals/{eval_id}/runs/{run_id}",
        operation_id="getEvalRun",
        tags=["evals"],
    )
    def eval_run_get(eval_id: str, run_id: str) -> EvalRunObject:
        _spec_or_404(eval_id)
        return EvalRunObject.model_validate(run_wire(_run_or_404(eval_id, run_id)))

    @app.post(
        "/v1/evals/{eval_id}/runs/{run_id}/cancel",
        operation_id="cancelEvalRun",
        tags=["evals"],
    )
    def eval_run_cancel(eval_id: str, run_id: str) -> EvalRunObject:
        _spec_or_404(eval_id)
        rec = _run_or_404(eval_id, run_id)
        cancelled, outcome = eval_store.cancel(rec.eval_id)
        if cancelled is None or outcome != "cancelled":
            raise ApiError(409, f"run '{run_id}' is {outcome}")
        _deliver_callback(cancelled)
        return EvalRunObject.model_validate(run_wire(cancelled))

    @app.delete(
        "/v1/evals/{eval_id}/runs/{run_id}",
        operation_id="deleteEvalRun",
        tags=["evals"],
    )
    def eval_run_delete(eval_id: str, run_id: str) -> EvalRunDeleted:
        _spec_or_404(eval_id)
        rec = _run_or_404(eval_id, run_id)
        if rec.status not in _TERMINAL_JOB_STATUS:
            raise ApiError(409, f"run '{run_id}' is {rec.status} — only terminal runs delete")
        eval_store.delete(rec.eval_id)
        return EvalRunDeleted(id=run_id)

    @app.get(
        "/v1/evals/{eval_id}/runs/{run_id}/output_items",
        operation_id="listEvalRunOutputItems",
        tags=["evals"],
    )
    def eval_run_items(
        eval_id: str, run_id: str, limit: int = 20, after: str | None = None
    ) -> EvalOutputItemPage:
        """Per-task verdict rows from the completed run's report — the
        suite's raw rows verbatim, paged by cursor (index-encoded ids)."""
        _spec_or_404(eval_id)
        rec = _run_or_404(eval_id, run_id)
        if not 1 <= limit <= 100:
            raise ApiError(400, _MSG_LIMIT_RANGE, code="invalid_request")
        tasks = (
            report_task_items(rec.report)
            if rec.status == "succeeded" and isinstance(rec.report, dict)
            else []
        )
        start = 0
        if after is not None:
            m = re.fullmatch(r"evalrun_(.+)-(\d+)", after)
            if m is None or m.group(1) != rec.eval_id:
                raise ApiError(
                    400,
                    f"cursor {after!r} is not an item id under run {run_id!r}",
                    code="invalid_cursor",
                )
            start = int(m.group(2)) + 1
        page = tasks[start : start + limit]
        items = [
            EvalOutputItem(
                id=f"evalrun_{rec.eval_id}-{start + i}",
                run_id=f"evalrun_{rec.eval_id}",
                eval_id=eval_id,
                created_at=int(rec.finished_at or rec.created_at),
                status="pass" if t["passed"] else "fail",
                datasource_item_id=t["name"],
                datasource_item=t["row"],
                results=[{"name": rec.suite, "passed": t["passed"]}],
            )
            for i, t in enumerate(page)
        ]
        return EvalOutputItemPage(
            data=items,
            has_more=start + len(page) < len(tasks),
            first_id=items[0].id if items else None,
            last_id=items[-1].id if items else None,
        )

    @app.post(
        "/harness/complete",
        response_model=CompleteResponse,
        tags=["complete"],
        operation_id="complete",
    )
    def complete(
        body: CompleteRequest,
        response: Response,
        _idem_claim_held: None = Depends(_complete_idem_claim),
        _slot_held: None = Depends(slot),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> CompleteResponse:
        # Same Idempotency-Key contract as the runs route: retried
        # completions replay from the cache instead of re-billing the model.
        body_fp = _body_fp(body)
        key, replay = _idem_lookup(idempotency_key, complete_idem_store, body_fp)
        if replay is not None:
            if replay.completion_id is not None:
                response.headers["X-Fx1-Completion-Id"] = replay.completion_id
                _rsha = _completion_receipt_sha(replay.completion_id)
                if _rsha is not None:
                    response.headers["X-Fx1-Receipt-Sha256"] = _rsha
            return replay
        _check_citations(body.receipt_hashes)
        messages = [m.model_dump(exclude_none=True) for m in body.messages]
        # the structured channel carries tool context AND provider
        # logprob scores — a logprobs request on a plain link is the
        # same capability gap (501) as a tool request on one
        wants_tools = (
            body.tools is not None
            or body.logprobs
            or body.top_logprobs is not None
            or any(m.role == "tool" or m.tool_calls for m in body.messages)
        )
        # plain-path messages are str-typed — the tool-context branch is
        # the only place a dict carries tool_calls/tool_call_id values
        plain_messages: list[dict[str, str]] = [
            {"role": m.role, "content": cast(str, m.content)} for m in body.messages
        ]
        cid = uuid.uuid4().hex
        prompt_sha256 = hashlib.sha256(
            json.dumps(messages, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        rec_err: str | None = None
        rec_cls: str | None = None
        content = ""
        tool_calls_out: list[dict[str, Any]] | None = None
        logprobs_out: dict[str, Any] | None = None
        finish_out: str | None = None
        serving: str | None = None
        usage_snap: Any = None
        model_snap: Any = None
        call_latency_ms = 0.0
        attempts: list[BackendAttempt] = []
        last_exc: ApiError | None = None
        sampling_params = _sampling_of(body)
        sampling_fields = sampling_params.body_fields()
        # Ordered fallback chain: each link gets its own admit + resolve +
        # call. Only availability faults advance the chain — a gate
        # refusal, a capability gap (501), or a client error aborts.
        try:
            for cand in [body.backend, *body.fallbacks]:
                cand_key = _breaker_key_name(cand, body.byok)
                backend: Any = None
                try:
                    _breaker_admit(cand_key)
                    backend = _resolve_candidate(cand, body)
                except ApiError as exc:
                    # only backend-unavailable counts — client errors (404
                    # unknown backend, 422 bad args) must never trip the
                    # circuit, or a caller could deny the backend for
                    # everyone by spamming bad requests.
                    if exc.status_code == 503:
                        rec_cls = "backend_unavailable"
                        rec_err = str(exc)
                        attempts.append(
                            BackendAttempt(
                                backend=cand, ok=False, error_class="backend_unavailable"
                            )
                        )
                        if breaker is not None:
                            breaker.report(cand_key, False)
                        last_exc = exc
                        continue
                    raise
                t0 = time.monotonic()
                try:
                    if wants_tools:
                        tool_result = cited_complete_tools(
                            backend,
                            messages,
                            receipt_hashes=body.receipt_hashes,
                            sampling=sampling_params,
                            tools=body.tools,
                            tool_choice=body.tool_choice,
                            parallel_tool_calls=body.parallel_tool_calls,
                            logprobs=body.logprobs,
                            top_logprobs=body.top_logprobs,
                        )
                        content = tool_result.content or ""
                        tool_calls_out = (
                            list(tool_result.tool_calls)
                            if tool_result.tool_calls is not None
                            else None
                        )
                        logprobs_out = tool_result.logprobs
                        finish_out = tool_result.finish_reason
                    else:
                        content = cited_complete(
                            backend,
                            plain_messages,
                            receipt_hashes=body.receipt_hashes,
                            sampling=sampling_params,
                        )
                except NotImplementedError as exc:
                    rec_cls = "not_supported"
                    rec_err = str(exc)
                    attempts.append(
                        BackendAttempt(backend=cand, ok=False, error_class="not_supported")
                    )
                    raise ApiError(501, str(exc)) from exc
                except Fx1HonestyError as exc:
                    rec_cls = "honesty_refusal"
                    rec_err = str(exc)
                    attempts.append(
                        BackendAttempt(backend=cand, ok=False, error_class="honesty_refusal")
                    )
                    # The model produced a contract-violating headline; the
                    # gate caught it before the bytes left — surface as 502,
                    # not success. Never falls back: a refusal is a verdict.
                    raise ApiError(
                        502,
                        f"honesty gate refused model output: {exc}",
                        code="honesty_gate",
                    ) from exc
                except (BackendNotConfiguredError, RuntimeError) as exc:
                    rec_cls = type(exc).__name__
                    rec_err = str(exc)
                    call_latency_ms = (time.monotonic() - t0) * 1000.0
                    attempts.append(
                        BackendAttempt(
                            backend=cand,
                            ok=False,
                            error_class=rec_cls,
                            latency_ms=call_latency_ms,
                        )
                    )
                    if breaker is not None:
                        breaker.report(cand_key, False)
                    last_exc = (
                        ApiError(503, str(exc), code="backend_unavailable")
                        if isinstance(exc, BackendNotConfiguredError)
                        else ApiError(502, str(exc), code="backend_failure")
                    )
                    continue
                finally:
                    _close_backend(backend)
                call_latency_ms = (time.monotonic() - t0) * 1000.0
                if breaker is not None:
                    breaker.report(cand_key, True)
                usage_snap = _billable_usage(getattr(backend, "last_usage", None))
                model_snap = (
                    body._served_model
                    if body._served_model is not None and cand == body.backend
                    else getattr(backend, "_model", None)
                )
                serving = cand
                attempts.append(BackendAttempt(backend=cand, ok=True, latency_ms=call_latency_ms))
                break
            if serving is None:
                raise (
                    last_exc
                    if last_exc is not None
                    else ApiError(503, "no backend in the chain served")
                )
        finally:
            # (content was already stop-truncated inside cited_complete —
            # the record's output_sha256 covers the shipped bytes.)
            metrics.record_complete(
                serving or body.backend,
                serving is not None,
                call_latency_ms,
                usage=usage_snap if isinstance(usage_snap, dict) else None,
            )
            completion_log.append(
                CompletionRecord(
                    completion_id=cid,
                    backend=serving or body.backend,
                    model=model_snap if isinstance(model_snap, str) else None,
                    ok=serving is not None,
                    latency_ms=call_latency_ms,
                    at=time.time(),
                    usage=usage_snap if isinstance(usage_snap, dict) else None,
                    error=rec_err if serving is None else None,
                    error_class=rec_cls if serving is None else None,
                    prompt_sha256=prompt_sha256,
                    # the digest binds what shipped — text alone for a
                    # plain turn; text + the verbatim call list when the
                    # answer is a tool call (content "" would digest an
                    # empty answer and let the payload slip the seal),
                    # plus the logprob payload when the caller asked for
                    # scores (they're part of what shipped).
                    output_sha256=(
                        hashlib.sha256(
                            (
                                content
                                + (
                                    "\n" + json.dumps(tool_calls_out, sort_keys=True)
                                    if tool_calls_out
                                    else ""
                                )
                                + (
                                    "\n" + json.dumps(logprobs_out, sort_keys=True)
                                    if logprobs_out
                                    else ""
                                )
                            ).encode("utf-8")
                        ).hexdigest()
                        if serving is not None
                        else None
                    ),
                    attempts=attempts if len(attempts) > 1 else None,
                    sampling=sampling_fields,
                    user=body.user,
                    metadata=body.metadata,
                )
            )
        assert serving is not None  # noqa: S101 — None already raised above
        resp = CompleteResponse(
            backend=serving,
            model=model_snap if isinstance(model_snap, str) else None,
            content=content,
            receipt_hashes=body.receipt_hashes or [],
            latency_ms=call_latency_ms,
            usage=usage_snap if isinstance(usage_snap, dict) else None,
            completion_id=cid,
            attempts=attempts if len(attempts) > 1 else [],
            sampling=sampling_fields,
            tool_calls=tool_calls_out,
            finish_reason=finish_out,
            logprobs=logprobs_out,
        )
        response.headers["X-Fx1-Completion-Id"] = cid
        _rsha = _completion_receipt_sha(cid)
        if _rsha is not None:
            response.headers["X-Fx1-Receipt-Sha256"] = _rsha
        if key is not None:
            complete_idem_store.put(key, body_fp, resp)
        return resp

    @app.post(
        "/harness/complete/stream",
        tags=["complete"],
        operation_id="complete_stream",
    )
    def complete_stream(
        body: CompleteRequest, _slot_held: None = Depends(slot)
    ) -> StreamingResponse:
        """Server-sent-event stream of one gated completion.

        The backend's token deltas are buffered, the joined text passes the
        honesty gate, and only then are chunks emitted as ``token`` events
        plus a ``final`` envelope — no ungated model bytes ever reach a
        ``data:`` frame. A backend without ``stream`` fails 501 rather than
        faking chunking.

        With ``sse_keepalive_s`` > 0 (default 15) the buffer+gate phase runs
        on a worker thread under a one-interval grace period: outcomes that
        resolve within the interval keep today's wire contract exactly —
        completions stream as usual, failures stay ordinary JSON error
        responses. Only a generation still running past the interval
        commits to SSE: ``: keepalive`` comment frames (no payload —
        parsers ignore them) hold the connection open against proxy/LB
        idle timeouts, and a failure after that point arrives as a terminal
        ``{"type": "error", "status", "detail"}`` data frame followed by
        ``[DONE]`` (the HTTP status is committed once a byte is on the
        wire). ``sse_keepalive_s`` = 0 disables the worker entirely — the
        fully synchronous path.
        """
        _check_citations(body.receipt_hashes)
        if body.tools is not None or any(m.role == "tool" or m.tool_calls for m in body.messages):
            # native-SSE tool streaming is not the wire — tool calls ride
            # /v1/chat/completions' delta.tool_calls frames instead
            raise ApiError(
                501,
                "tool calls are not streamable on /harness/complete/stream; "
                "use /v1/chat/completions?stream=true",
                code="not_supported",
            )
        if body.logprobs or body.top_logprobs is not None:
            # same channel boundary — provider scores ride the /v1 wire's
            # aggregated delta.logprobs frame
            raise ApiError(
                501,
                "logprobs are not streamable on /harness/complete/stream; "
                "use /v1/chat/completions?stream=true",
                code="not_supported",
            )
        messages = [{"role": m.role, "content": cast(str, m.content)} for m in body.messages]
        sampling_params = _sampling_of(body)
        sampling_fields = sampling_params.body_fields()
        # The record is appended from the streaming generator / worker
        # thread, where this request's contextvars are long gone — the
        # credential fingerprint is captured while it is still bound and
        # stamped on the record explicitly (same play as the batch path).
        req_key_id = _REQUEST_KEY_ID.get()

        def _gather() -> tuple[list[str], str | None, float, dict[str, int] | None, str]:
            """Buffer + gate the backend stream; raises the mapped errors.

            The fallback chain applies at resolve level only — once a link
            is streaming there is no honest restart point."""
            serving, backend, chain_attempts = _resolve_chain(body)
            t0 = time.monotonic()
            call_ok = False
            cid = uuid.uuid4().hex
            prompt_sha256 = hashlib.sha256(
                json.dumps(messages, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            rec_err: str | None = None
            rec_cls: str | None = None
            try:
                if not isinstance(backend, StreamingBackend):
                    raise NotImplementedError(
                        f"backend {body.backend!r} does not support streaming"
                    )
                chunks = list(backend.stream(messages, sampling=sampling_params))
                joined = "".join(chunks)
                try:
                    validate_fx1_output(joined)
                    # stop-sequence cut after the gate: a gated prefix is
                    # still gated — shipped deltas keep provider chunk
                    # boundaries up to the cut.
                    chunks = truncate_chunks(chunks, sampling_params.stop)
                except Fx1HonestyError as exc:
                    rec_cls = "honesty_refusal"
                    rec_err = str(exc)
                    raise ApiError(
                        502, f"honesty gate refused model output: {exc}", code="honesty_gate"
                    ) from exc
            except NotImplementedError as exc:
                rec_cls = "not_supported"
                rec_err = str(exc)
                raise ApiError(501, str(exc)) from exc
            except (BackendNotConfiguredError, RuntimeError) as exc:
                rec_cls = type(exc).__name__
                rec_err = str(exc)
                if breaker is not None:
                    breaker.report(_breaker_key_name(serving, body.byok), False)
                if isinstance(exc, BackendNotConfiguredError):
                    raise ApiError(503, str(exc), code="backend_unavailable") from exc
                raise ApiError(502, str(exc), code="backend_failure") from exc
            else:
                call_ok = True
                if breaker is not None:
                    breaker.report(_breaker_key_name(serving, body.byok), True)
            finally:
                usage_snap = _billable_usage(getattr(backend, "last_usage", None))
                model_snap = (
                    body._served_model
                    if body._served_model is not None and serving == body.backend
                    else getattr(backend, "_model", None)
                )
                joined_snap = "".join(chunks) if call_ok else ""
                metrics.record_complete(
                    serving,
                    call_ok,
                    (time.monotonic() - t0) * 1000.0,
                    usage=usage_snap if isinstance(usage_snap, dict) else None,
                )
                completion_log.append(
                    CompletionRecord(
                        completion_id=cid,
                        backend=serving,
                        attempts=chain_attempts if len(chain_attempts) > 1 else None,
                        model=model_snap if isinstance(model_snap, str) else None,
                        ok=call_ok,
                        latency_ms=(time.monotonic() - t0) * 1000.0,
                        at=time.time(),
                        usage=usage_snap if isinstance(usage_snap, dict) else None,
                        error=rec_err,
                        error_class=rec_cls,
                        prompt_sha256=prompt_sha256,
                        sampling=sampling_fields,
                        user=body.user,
                        metadata=body.metadata,
                        key_id=req_key_id,
                        output_sha256=(
                            hashlib.sha256(joined_snap.encode("utf-8")).hexdigest()
                            if call_ok
                            else None
                        ),
                    )
                )
                _close_backend(backend)
            if body.receipt_hashes:
                chunks.append(
                    "\n\nEvidence: "
                    + ", ".join(f"`{h[:16]}…`" for h in body.receipt_hashes)
                    + " — verify with `dipcatcher verify-research`."
                )
            return (
                chunks,
                model_snap if isinstance(model_snap, str) else None,
                (time.monotonic() - t0) * 1000.0,
                cast("dict[str, int]", dict(usage_snap)) if isinstance(usage_snap, dict) else None,
                cid,
            )

        def _events(
            chunks: list[str],
            model_name: str | None,
            latency_ms: float,
            usage: dict[str, int] | None = None,
            completion_id: str | None = None,
        ) -> Iterator[str]:
            for chunk in chunks:
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            yield (
                _SSE_DATA_PREFIX
                + json.dumps(
                    {
                        "type": "final",
                        "model": model_name,
                        "receipt_hashes": body.receipt_hashes or [],
                        "latency_ms": latency_ms,
                        "usage": usage,
                        "completion_id": completion_id,
                        "receipt_sha256": _completion_receipt_sha(completion_id),
                        "sampling": sampling_fields,
                    }
                )
                + "\n\n"
            )
            yield _SSE_DONE

        if sse_keepalive_s <= 0:
            chunks, model_name, latency_ms, usage, cid = _gather()
            return StreamingResponse(
                _events(chunks, model_name, latency_ms, usage, cid),
                media_type="text/event-stream",
            )

        pipe: queue.Queue[tuple[str, Any]] = queue.Queue()

        def _produce() -> None:
            try:
                pipe.put(("ok", _gather()))
            except HTTPException as exc:
                pipe.put(("error", exc))
            except Exception as exc:  # noqa: BLE001
                # a dead pipe is an honest in-band frame, never a hang
                pipe.put(("error", HTTPException(502, f"backend failed: {exc}")))

        threading.Thread(target=_produce, daemon=True).start()

        # Grace window: an outcome inside one keepalive interval keeps the
        # synchronous contract (SSE stream / JSON error); only a generation
        # still running past it commits to the keepalived stream.
        grace: tuple[str, Any] | None
        try:
            grace = pipe.get(timeout=sse_keepalive_s)
        except queue.Empty:
            grace = None
        if grace is not None:
            tag, payload = grace
            if tag == "error":
                raise payload
            chunks, model_name, latency_ms, usage, cid = payload
            return StreamingResponse(
                _events(chunks, model_name, latency_ms, usage, cid),
                media_type="text/event-stream",
            )

        def _events_keepalived() -> Iterator[str]:
            while True:
                try:
                    tag, payload = pipe.get(timeout=sse_keepalive_s)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                if tag == "error":
                    yield (
                        _SSE_DATA_PREFIX
                        + json.dumps(
                            {
                                "type": "error",
                                "status": payload.status_code,
                                "detail": payload.detail,
                                "code": _err_code(payload),
                            }
                        )
                        + "\n\n"
                    )
                    yield _SSE_DONE
                    return
                chunks, model_name, latency_ms, usage, cid = payload
                yield from _events(chunks, model_name, latency_ms, usage, cid)
                return

        return StreamingResponse(_events_keepalived(), media_type="text/event-stream")

    # --- OpenAI-compatible ingress ----------------------------------------
    # Drop-in surface for stock OpenAI clients: translate the request onto
    # the gated `complete` chain (same resolve/fallback/gate/meter/log path
    # — no second pipeline), then emit the OpenAI envelope. Errors under
    # /v1 get the OpenAI error shape via the app's exception handlers and
    # auth middleware (path-prefix check). Streaming emits gated text as
    # deltas after the honesty gate — first byte arrives post-gate, so
    # first-token latency equals full latency; no ungated bytes ever ship.

    _openai_created = int(time.time())

    def _complete_request_from_kwargs(kwargs: dict[str, Any]) -> CompleteRequest:
        """Validate translated wire kwargs while binding registry attribution
        privately.  ``_served_model`` is intentionally not a public request
        field: only the registry translator may stamp an alias on evidence."""
        translated = dict(kwargs)
        served_model = translated.pop("_served_model", None)
        try:
            request = CompleteRequest(**translated)
        except ValidationError as exc:
            raise OpenAICompatError(_validation_msgs(exc), status=422) from exc
        request._served_model = served_model
        return request

    def _openai_chat_core(
        body: OpenAIChatRequest,
        headers: Mapping[str, str],
    ) -> _ChatEnv:
        """The non-streaming chat-completions completion core, shared by
        the ``/v1/chat/completions`` route and the ``/v1/batches`` worker —
        one gated path, one envelope. Raises ``OpenAICompatError`` on
        translation or post-validation failures (callers map it to the
        wire shape)."""
        creq = _complete_request_from_kwargs(
            openai_to_kwargs(body, headers, ft_resolver=ft_store.checkpoint_for)
        )
        # n>1 fans out into n gated calls — each completion gets its own
        # honesty-gate pass, format check, and completion-log record; usage
        # sums what was actually spent (n calls × provider-reported counts).
        outs = [
            complete(body=creq, response=Response(), _slot_held=None, idempotency_key=None)
            for _ in range(body.n)
        ]
        contents: list[str] = []
        choice_calls: list[list[dict[str, Any]] | None] = []
        choice_reasons: list[str] = []
        choice_lps: list[dict[str, Any] | None] = []
        usage_sum: dict[str, int] = {}
        usage_seen = False
        for out in outs:
            # response_format post-validation: the provider can't be
            # constrain-decoded, so a format violation is its failure (502),
            # never shipped, never pinned into the idempotency record.
            validate_openai_output(body, out.content)
            contents.append(out.content)
            choice_calls.append(out.tool_calls)
            choice_reasons.append(out.finish_reason or "stop")
            choice_lps.append(out.logprobs)
            if isinstance(out.usage, dict):
                usage_seen = True
                for uk, uv in out.usage.items():
                    if isinstance(uv, int):
                        usage_sum[uk] = usage_sum.get(uk, 0) + uv
        cid = outs[0].completion_id or uuid.uuid4().hex
        served_by = dict.fromkeys(o.backend for o in outs)
        envelope = openai_envelope(
            cid=cid,
            content=contents if body.n > 1 else contents[0],
            backend="+".join(served_by),
            model=outs[0].model,
            usage=usage_sum if usage_seen else None,
            tool_calls=(choice_calls if any(c is not None for c in choice_calls) else None),
            finish_reasons=choice_reasons,
            logprobs=choice_lps,
            metadata=body.metadata,
        )
        if body.store is not False:
            envelope_store.put(
                envelope,
                items={
                    "messages": chat_messages_for_store(
                        body.messages, envelope_id=str(envelope["id"])
                    )
                },
            )
        return envelope, cid

    # Server-side ``file_search`` — the tool's retrieval runs in the
    # harness before the backend call. Hits inject as a developer-role
    # context item prepended to the effective input (so the stored
    # input_items transcript carries exactly what the model saw), and
    # one ``file_search_call`` output item per tool spec records the
    # run. Retrieval failures fail closed — a bogus store id is a
    # client-visible 404, not a silent miss.
    _RETRIEVAL_INJECT_BUDGET = 32768

    def _file_search_turn(
        body: OpenAIResponseRequest,
        eff_body: OpenAIResponseRequest,
    ) -> tuple[list[dict[str, Any]], OpenAIResponseRequest]:
        fs_specs = [t for t in (body.tools or []) if t.type == "file_search"]
        if not fs_specs:
            return [], eff_body
        query = response_query_text(eff_body.input)
        if not query.strip():
            raise OpenAICompatError(
                "file_search needs a non-empty user message to query",
                status=400,
                code="empty_query",
            )
        include_results = bool(body.include and "file_search_call.results" in body.include)
        search_items: list[dict[str, Any]] = []
        inject_parts: list[str] = []
        budget = _RETRIEVAL_INJECT_BUDGET
        for spec in fs_specs:
            if spec.type != "file_search":
                continue
            ro = spec.ranking_options or {}
            threshold = ro.get("score_threshold")
            try:
                hits = vs_store.search(
                    list(spec.vector_store_ids),
                    query,
                    max_results=spec.max_num_results or 10,
                    filters=spec.filters,
                    score_threshold=float(threshold) if threshold is not None else None,
                )
            except VectorStoreError as exc:
                raise OpenAICompatError(str(exc), status=exc.status, code=exc.code) from exc
            search_items.append(
                file_search_call_item(
                    queries=[query],
                    results=[
                        {
                            "file_id": h["file_id"],
                            "filename": h["filename"],
                            "vector_store_id": h["vector_store_id"],
                            "score": h["score"],
                            "text": h["text"],
                            "attributes": h["attributes"],
                        }
                        for h in hits
                    ]
                    if include_results
                    else None,
                )
            )
            for h in hits:
                piece = f"[{h['file_id']} {h['filename']} score {h['score']:.3f}] {h['text']}"
                if len(piece) <= budget:
                    inject_parts.append(piece)
                    budget -= len(piece)
        if inject_parts:
            ctx = "[file_search results — retrieved context]\n" + "\n\n".join(inject_parts)
            prior: list[Any] = (
                list(eff_body.input)
                if isinstance(eff_body.input, list)
                else [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": str(eff_body.input)}],
                    }
                ]
            )
            eff_body = eff_body.model_copy(
                update={
                    "input": [
                        {
                            "type": "message",
                            "role": "developer",
                            "content": [{"type": "input_text", "text": ctx}],
                        },
                        *prior,
                    ]
                }
            )
        return search_items, eff_body

    def _openai_response_core(
        body: OpenAIResponseRequest,
        headers: Mapping[str, str],
        *,
        rid: str | None = None,
        created: int | None = None,
    ) -> tuple[dict[str, Any], str, dict[str, int] | None]:
        """The non-streaming ``/v1/responses`` completion core — shared by
        the route and the ``/v1/batches`` worker. Returns the envelope
        plus the completion-log id and raw usage (the caller decides what
        rides the idempotency record).``previous_response_id`` chains the
        turn onto a stored response — the parent must sit in the retrieval
        index (a ``store=false`` or evicted parent fails closed).
        ``conversation`` anchors to a named container instead: the conv's
        accumulated items become the context, and after the turn completes
        the request input + output items append onto the conv (the conv
        is its own store — it keeps the turn even under
        ``store=false``). The two anchors are mutually exclusive
        (422 at validation); an unknown conv fails closed
        ``400 conversation_not_found``."""
        eff_body = body
        conv_cid: str | None = None
        if body.previous_response_id is not None:
            prev = envelope_store.get(body.previous_response_id)
            if prev is None or prev.get("object") != "response":
                raise OpenAICompatError(
                    f"previous_response_id {body.previous_response_id!r} not found — "
                    "the chain parent must be a stored response (store=true)",
                    status=400,
                    code="previous_response_not_found",
                )
            eff_body = body.model_copy(
                update={
                    "input": chained_response_input(
                        prev,
                        envelope_store.get_items(body.previous_response_id, "input_items") or [],
                        body.input,
                    )
                }
            )
        else:
            conv_cid = conversation_id_of(body.conversation)
            if conv_cid is not None:
                conv = conv_store.get(conv_cid)
                if conv is None or conv.get("object") != "conversation":
                    raise OpenAICompatError(
                        f"conversation {conv_cid!r} not found — create it with "
                        "POST /v1/conversations first",
                        status=400,
                        code="conversation_not_found",
                    )
                eff_body = body.model_copy(
                    update={
                        "input": chained_response_input(
                            {"output": []},
                            conv_store.get_items(conv_cid, "items") or [],
                            body.input,
                        )
                    }
                )
        search_items, eff_body = _file_search_turn(body, eff_body)
        creq = _complete_request_from_kwargs(
            response_to_kwargs(eff_body, headers, ft_resolver=ft_store.checkpoint_for)
        )
        out = complete(body=creq, response=Response(), _slot_held=None, idempotency_key=None)
        # a tool-call turn carries no text — there is nothing to
        # post-validate against text.format on an empty content
        if out.content or not out.tool_calls:
            validate_response_format(response_text_format(body), out.content)
        cid = out.completion_id or uuid.uuid4().hex
        # ``max_tool_calls`` truncates a turn that emitted over the cap —
        # the response lands 'incomplete' with the bounded call list, never
        # a silent drop
        call_items, inc_details = response_cap_call_items(body, out.tool_calls or [])
        lp_arr = out.logprobs.get("content") if isinstance(out.logprobs, dict) else None
        envelope = openai_response_object(
            rid=rid or f"resp_{uuid.uuid4().hex}",
            item_id=f"msg_{uuid.uuid4().hex}",
            content=out.content,
            body=body,
            model=out.model,
            usage=out.usage,
            status=("incomplete" if inc_details else "completed"),
            created=created,
            call_items=call_items,
            search_items=search_items or None,
            logprobs=(lp_arr if isinstance(lp_arr, list) else None),
            incomplete_details=inc_details,
        )
        if body.store is not False:
            stored_env = {
                **envelope,
                # the completion-log link the GET ?stream replay surface
                # needs for its X-Fx1-* headers — stripped before any
                # wire read, matching the idem-record convention
                "_fx1_completion_id": cid,
            }
            store_items = {
                "input_items": response_input_items_for_store(
                    eff_body.input, rid=str(envelope["id"])
                )
            }
            if rid is None:
                envelope_store.put(stored_env, items=store_items)
            else:
                # Commit only while this background response is active.
                # A cancelled/deleted result remains in the completion log
                # but must not overwrite its verdict or its conversation.
                stored = envelope_store.put_unless_status(
                    stored_env,
                    forbidden=OPENAI_RESPONSE_TERMINAL,
                    require_existing=True,
                    items=store_items,
                )
                if not stored:
                    return envelope, cid, out.usage
        if conv_cid is not None:
            # the conv accumulates each turn's own items (request input +
            # response output) — the conv IS the store, so this happens
            # even under ``store=false`` on the response itself.
            # ``mutate_items`` merges under the store lock: two turns
            # completing together can't lose each other's append, and a
            # delete landing mid-turn resolves as a skip, never a
            # get-then-put resurrection of the container.
            appended = [
                *response_input_items_for_store(body.input, rid=str(envelope["id"])),
                *[it for it in envelope["output"] if isinstance(it, dict)],
            ]
            conv_store.mutate_items(conv_cid, "items", lambda current: [*current, *appended])
        return envelope, cid, out.usage

    def _openai_embeddings_core(
        body: OpenAIEmbeddingRequest,
        headers: Mapping[str, str],
    ) -> _ChatEnv:
        """The embeddings core, shared by the ``/v1/embeddings`` route and
        the ``/v1/batches`` worker — same chain contract as chat:
        availability faults advance the fallback chain, capability gaps
        answer 501, provider errors surface as their own class. Vectors
        aren't claims — there's no honesty gate — but the call lands in
        the completion log and metrics exactly like a completion (the
        digest binds the sent input and the verbatim ``data[]``).

        Returns ``(envelope, completion_id)``."""
        ereq = _request_or_422(
            EmbedRequest,
            embeddings_to_kwargs(body, headers, ft_resolver=ft_store.checkpoint_for),
        )
        cid = uuid.uuid4().hex
        prompt_sha256 = hashlib.sha256(
            json.dumps(
                {"model": ereq.model, "input": ereq.input},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        rec_err: str | None = None
        rec_cls: str | None = None
        serving: str | None = None
        usage_snap: Any = None
        model_out: str = ereq.model
        call_latency_ms = 0.0
        attempts: list[BackendAttempt] = []
        last_exc: ApiError | None = None
        data_out: tuple[dict[str, Any], ...] = ()
        try:
            for cand in [ereq.backend, *ereq.fallbacks]:
                cand_key = _breaker_key_name(cand, ereq.byok)
                backend: Any = None
                try:
                    _breaker_admit(cand_key)
                    backend = _resolve_candidate(cand, ereq)
                except ApiError as exc:
                    if exc.status_code == 503:
                        rec_cls = "backend_unavailable"
                        rec_err = str(exc)
                        attempts.append(
                            BackendAttempt(
                                backend=cand, ok=False, error_class="backend_unavailable"
                            )
                        )
                        if breaker is not None:
                            breaker.report(cand_key, False)
                        last_exc = exc
                        continue
                    raise
                t0 = time.monotonic()
                try:
                    if not isinstance(backend, EmbeddingBackend):
                        raise NotImplementedError(f"backend {cand!r} has no embeddings channel")
                    result = backend.embeddings(
                        ereq.input,
                        model=ereq.model,
                        encoding_format=ereq.encoding_format,
                        dimensions=ereq.dimensions,
                        user=ereq.user,
                    )
                except NotImplementedError as exc:
                    rec_cls = "not_supported"
                    rec_err = str(exc)
                    attempts.append(
                        BackendAttempt(backend=cand, ok=False, error_class="not_supported")
                    )
                    raise ApiError(501, str(exc), code="not_supported") from exc
                except (BackendNotConfiguredError, RuntimeError) as exc:
                    rec_cls = type(exc).__name__
                    rec_err = str(exc)
                    call_latency_ms = (time.monotonic() - t0) * 1000.0
                    attempts.append(
                        BackendAttempt(
                            backend=cand,
                            ok=False,
                            error_class=rec_cls,
                            latency_ms=call_latency_ms,
                        )
                    )
                    if breaker is not None:
                        breaker.report(cand_key, False)
                    last_exc = (
                        ApiError(503, str(exc), code="backend_unavailable")
                        if isinstance(exc, BackendNotConfiguredError)
                        else ApiError(502, str(exc), code="backend_failure")
                    )
                    continue
                finally:
                    _close_backend(backend)
                call_latency_ms = (time.monotonic() - t0) * 1000.0
                if breaker is not None:
                    breaker.report(cand_key, True)
                data_out = result.data
                usage_snap = _billable_usage(result.usage)
                model_out = result.model or ereq.model
                serving = cand
                attempts.append(BackendAttempt(backend=cand, ok=True, latency_ms=call_latency_ms))
                break
            if serving is None:
                raise (
                    last_exc
                    if last_exc is not None
                    else ApiError(503, "no backend in the chain served")
                )
        finally:
            metrics.record_complete(
                serving or ereq.backend,
                serving is not None,
                call_latency_ms,
                usage=usage_snap if isinstance(usage_snap, dict) else None,
            )
            completion_log.append(
                CompletionRecord(
                    completion_id=cid,
                    backend=serving or ereq.backend,
                    model=model_out if serving is not None else None,
                    ok=serving is not None,
                    latency_ms=call_latency_ms,
                    at=time.time(),
                    usage=usage_snap if isinstance(usage_snap, dict) else None,
                    error=rec_err if serving is None else None,
                    error_class=rec_cls if serving is None else None,
                    prompt_sha256=prompt_sha256,
                    output_sha256=(
                        hashlib.sha256(
                            json.dumps(list(data_out), sort_keys=True).encode("utf-8")
                        ).hexdigest()
                        if serving is not None
                        else None
                    ),
                    attempts=attempts if len(attempts) > 1 else None,
                    sampling={
                        k: v
                        for k, v in (
                            ("encoding_format", ereq.encoding_format),
                            ("dimensions", ereq.dimensions),
                        )
                        if v is not None
                    }
                    or None,
                    user=ereq.user,
                    metadata=None,
                )
            )
        assert serving is not None  # noqa: S101 — None already raised above
        envelope = openai_embedding_envelope(
            data=data_out,
            model=model_out,
            usage=usage_snap if isinstance(usage_snap, dict) else None,
        )
        return envelope, cid

    @app.get(
        "/v1/models",
        tags=["openai"],
        operation_id="openai_list_models",
        response_model=OpenAIModelList,
    )
    def openai_models(
        request: Request,
        limit: int = Query(default=20, ge=1, le=1000),
        before_id: str | None = Query(default=None),
        after_id: str | None = Query(default=None),
    ) -> Response:
        """Model inventory — the backend names a `model` field may carry,
        plus the `fx1` alias for the default link (hosted_k3).

        One route, two envelopes: an ``anthropic-version`` header (the
        stock anthropic SDK sends it on every call) switches the payload
        to Anthropic's ``{data: [{type: \"model\", id, display_name,
        created_at}], first_id, last_id, has_more}`` grammar with its
        ``limit``/``after_id``/``before_id`` cursors —
        ``client.models.list()`` works unmodified. Without the header the
        OpenAI ``{object: \"list\"}`` shape answers; the cursor params
        are Anthropic's and ignored on the OpenAI branch."""
        refs = ft_store.models()
        stamps = {str(r["id"]): int(r["created"]) for r in refs}
        ids = [m for m in (*OPENAI_MODEL_IDS, *(r["id"] for r in refs))]
        if "anthropic-version" not in request.headers:
            return JSONResponse(
                OpenAIModelList(
                    data=[OpenAIModel(id=m, created=stamps.get(m, _openai_created)) for m in ids]
                ).model_dump(mode="json")
            )
        if after_id is not None:
            idx = next((i for i, x in enumerate(ids) if x == after_id), None)
            ids = ids[idx + 1 :] if idx is not None else []
        if before_id is not None:
            idx = next((i for i, x in enumerate(ids) if x == before_id), None)
            ids = ids[:idx] if idx is not None else []
        # ``before_id`` pages *backward* — the tail of the remaining
        # window, so first_id chains through the list the way after_id
        # chains forward through last_id.
        page = ids[-limit:] if before_id is not None else ids[:limit]
        return JSONResponse(
            {
                "data": [
                    anthropic_model_object(mid, created=stamps.get(mid, _openai_created))
                    for mid in page
                ],
                "first_id": page[0] if page else None,
                "last_id": page[-1] if page else None,
                "has_more": len(ids) > limit,
            }
        )

    @app.get(
        "/v1/models/{model}",
        tags=["openai"],
        operation_id="openai_retrieve_model",
        response_model=OpenAIModel,
    )
    def openai_retrieve_model(model: str, request: Request) -> Response:
        """OpenAI's models.retrieve — one card for a listed id; unknown
        ids fail closed 404 in the OpenAI error shape, never a
        fabricated card. Registered ``ft:`` fine-tunes resolve too.

        Under ``anthropic-version`` the same route answers Anthropic's
        ``{type: \"model\", id, display_name, created_at}`` card (the
        stock SDK's ``client.models.retrieve``), with unknown ids in the
        ``not_found_error`` grammar."""
        anthropic = "anthropic-version" in request.headers
        refs = ft_store.models()
        try:
            card = openai_model(
                model,
                created=_openai_created,
                extra_ids=(r["id"] for r in refs),
                created_by_id={str(r["id"]): int(r["created"]) for r in refs},
            )
        except OpenAICompatError as exc:
            if anthropic:
                return JSONResponse(
                    anthropic_error_body(f"model {model!r} not found", exc.status),
                    status_code=exc.status,
                )
            raise ApiError(exc.status, str(exc), code=exc.code) from exc
        if anthropic:
            return JSONResponse(anthropic_model_object(model, created=card.created))
        return JSONResponse(card.model_dump(mode="json"))

    @app.delete(
        "/v1/models/{model}",
        tags=["openai"],
        operation_id="openai_delete_model",
    )
    def openai_delete_model(model: str) -> OpenAIModelDelete:
        """OpenAI's ``models.delete`` — unregister a fine-tuned model.
        Only registered ``ft:`` names are deletable: the built-in link
        ids are permanent (400) and unknown names fail closed 404 — a
        delete verdict is never fabricated for a model that isn't real.
        The tombstone journals, so a restart can't resurrect it."""
        if model in OPENAI_MODEL_IDS:
            raise ApiError(
                400,
                f"the built-in link '{model}' is not deletable",
                code="invalid_request",
            )
        if ft_store.unregister_model(model) is None:
            raise ApiError(404, f"The model '{model}' does not exist", code="model_not_found")
        return OpenAIModelDelete(id=model, deleted=True)

    @app.post(
        "/v1/chat/completions",
        # the JSON path returns OpenAIChatResponse; stream=true returns SSE
        # chunks (response_model stays None — the route returns a Response)
        responses={200: {"model": OpenAIChatResponse}},
        tags=["openai"],
        operation_id="openai_chat_completions",
    )
    def openai_chat_completions(
        body: OpenAIChatRequest,
        request: Request,
        _idem_claim_held: None = Depends(_openai_idem_claim),
        _slot_held: None = Depends(slot),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> Response:
        """OpenAI-compatible chat completion over the gated pipeline.

        The `X-Fx1-Completion-Id` header links the response to the
        completion-log record (GET /harness/completions/{id}) and its
        sealed receipt. `model` selects a backend when it names one
        (hosted_k3/local_fx1/byok); anything else is the default link.
        BYOK binds via `X-Fx1-Byok-{Base-Url,Api-Key,Model}` headers or the
        `fx1` extension object; `X-Fx1-Backend`/`X-Fx1-Fallbacks`/
        `X-Fx1-Checkpoint-Dir` select the chain without body extensions.
        `Idempotency-Key` makes the call retry-safe: a retry with the
        same key and body replays the stored response byte-identically
        (JSON or SSE) instead of re-spending the model — flagged via
        `X-Fx1-Idempotent-Replay`; a key reused with a different body
        fails closed 409.

        Streams are resumable: every SSE frame carries `id: <index>`
        and a client that records `Last-Event-ID` can replay the keyed
        call with that header — the stored response regenerates
        byte-identically and frames at or below the delivered index are
        dropped. Resume fails closed: it needs the original
        `Idempotency-Key` (an unpinned stream has nothing to resume)
        and a stored record for that key (otherwise 409 — executing
        fresh and skipping would graft a different completion onto the
        client's earlier frames).

        Slow generations keepalived under `sse_keepalive_s` emit
        `: keepalive` comment frames — spec-valid and invisible to both
        SSE parsers and the `id:` sequence resume counts — until the
        gated call lands, then the normal chunk sequence; a backend
        fault mid-window answers an in-band `{error}` + `[DONE]` instead
        of a hang. The `X-Fx1-Completion-Id` header only exists once the
        completion lands, so keepalived legs carry the fingerprint
        in-band as `chatcmpl-<cid>` on every chunk.
        """
        # resume parsing first — a malformed Last-Event-ID fails before
        # any idempotency store work or model spend
        skip = _resume_skip(last_event_id, stream=body.stream, idempotency_key=idempotency_key)
        body_fp = _body_fp(body, headers=request.headers)
        key, replay = _idem_lookup(idempotency_key, openai_idem_store, body_fp)
        if skip and replay is None:
            raise ApiError(409, _RESUME_MISS_MSG, code="resume_miss")
        if replay is not None:
            env = replay.envelope
            cid_replay = str(env["id"]).removeprefix("chatcmpl-")
            # re-pin in the retrieval index — a replay refreshes the entry
            if body.store is not False:
                envelope_store.put(env)
            headers = _completion_headers(cid_replay, replay=True)
            if body.stream:
                return StreamingResponse(
                    _openai_sse(
                        body,
                        content=[c["message"]["content"] or "" for c in env["choices"]],
                        tool_calls=[c["message"].get("tool_calls") for c in env["choices"]],
                        finish_reasons=[c["finish_reason"] for c in env["choices"]],
                        logprobs=[c.get("logprobs") for c in env["choices"]],
                        backend=env["system_fingerprint"],
                        model=env["model"],
                        usage=env["usage"],
                        cid=cid_replay,
                        created=env["created"],
                        skip=skip,
                    ),
                    media_type="text/event-stream",
                    headers=headers,
                )
            return JSONResponse(env, headers=headers)

        def _generate() -> _ChatEnv:
            """The gated call packaged for the grace pipe — translation
            faults surface as ApiError so the keepalived leg answers them
            in grammar, and the keyed replay pins only on a real
            completion."""
            try:
                env, cid = _openai_chat_core(body, request.headers)
            except OpenAICompatError as exc:
                raise ApiError(exc.status, str(exc), code=exc.code) from exc
            if key is not None:
                openai_idem_store.put(key, body_fp, _OpenAIIdemRecord(envelope=env))
            return env, cid

        def _sse_frames(env_chat: dict[str, Any], cid: str) -> Iterator[str]:
            return _openai_sse(
                body,
                content=[c["message"]["content"] or "" for c in env_chat["choices"]],
                tool_calls=[c["message"].get("tool_calls") for c in env_chat["choices"]],
                finish_reasons=[c["finish_reason"] for c in env_chat["choices"]],
                logprobs=[c.get("logprobs") for c in env_chat["choices"]],
                backend=env_chat["system_fingerprint"],
                model=env_chat["model"],
                usage=env_chat["usage"],
                cid=cid,
            )

        keepalive_s = float(getattr(app.state, "sse_keepalive_s", 15.0))
        outcome, pipe = _grace_stage(_generate, stream=body.stream, keepalive_s=keepalive_s)
        if outcome is not None:
            tag, payload = outcome
            if tag == "error":
                raise cast("HTTPException", payload)
            env_chat, cid = cast("_ChatEnv", payload)
            headers = _completion_headers(cid)
            if body.stream:
                return StreamingResponse(
                    _sse_frames(env_chat, cid),
                    media_type="text/event-stream",
                    headers={**headers, **_SSE_STREAM_HEADERS},
                )
            return JSONResponse(env_chat, headers=headers)

        def _keepalived() -> Iterator[str]:
            """Past the grace window: ``: keepalive`` comment frames hold
            the connection (spec-valid, invisible to SSE parsers and to
            the ``id:`` sequence ``Last-Event-ID`` counts), the
            ``chatcmpl-<cid>`` id rides in-band on every chunk, and a
            backend fault lands as an in-band ``{error}`` + ``[DONE]``
            rather than a hung stream."""
            assert pipe is not None
            while True:
                outcome = _grace_await(pipe, keepalive_s)
                if outcome is None:
                    yield ": keepalive\n\n"
                    continue
                tag, payload = outcome
                if tag == "error":
                    fault = cast("HTTPException", payload)
                    yield (
                        _SSE_DATA_PREFIX
                        + json.dumps(
                            openai_error_body(
                                str(fault.detail), fault.status_code, _err_code(fault)
                            ),
                            separators=(",", ":"),
                        )
                        + "\n\n"
                    )
                    yield _SSE_DONE
                    return
                env_chat, cid = cast("_ChatEnv", payload)
                yield from _sse_frames(env_chat, cid)
                return

        return StreamingResponse(
            _keepalived(),
            media_type="text/event-stream",
            headers=_SSE_STREAM_HEADERS,
        )

    @app.post(
        "/v1/messages",
        # the JSON path returns the Anthropic `message` object;
        # stream=true returns the Anthropic SSE event grammar
        responses={200: {"model": AnthropicMessageObject}},
        tags=["anthropic"],
        operation_id="anthropic_messages",
    )
    def anthropic_messages(
        body: AnthropicMessagesRequest,
        request: Request,
        _idem_claim_held: None = Depends(_anthropic_idem_claim),
        _slot_held: None = Depends(slot),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> Response:
        """Anthropic Messages-compatible completion over the gated pipeline.

        The request translates into the shared OpenAI core — same honesty
        gate, same fail-closed validation, same metering — and the
        completion translates out to Anthropic's `message` object (or the
        `message_start`/`content_block_*`/`message_delta`/`message_stop`
        SSE grammar when `stream` is set). Auth is the same surface:
        `X-API-Key`, `Authorization: Bearer`, or a managed `/harness/keys`
        key; the stock anthropic SDK's `x-api-key` header works
        unmodified. Anthropic-only knobs the pipeline cannot honor
        (`top_k`, `thinking`, `service_tier`, `cache_control`, image or
        document blocks, …) refuse `400 invalid_request_error` instead
        of silently dropping.

        `Idempotency-Key` makes the call retry-safe — a same-key+body
        retry replays the pinned completion byte-identically (JSON or
        SSE, `X-Fx1-Idempotent-Replay: true`); a key reused with a
        different body fails closed 409. Keyed streams resume on
        `Last-Event-ID` — frames at or below the delivered index are
        dropped, and a resume with no pinned record fails closed 409.
        `X-Fx1-Completion-Id` links the response to the completion-log
        record (`GET /harness/completions/{id}`) and its sealed receipt.
        `X-Fx1-*` backend headers and the `fx1` extension object carry
        over from the OpenAI surface (backend selection, BYOK, deadline).

        Slow generations keepalived under `sse_keepalive_s` emit
        unnumbered `ping` frames until the gated call lands — Anthropic's
        own keepalive grammar — then the normal event sequence; a backend
        fault mid-window answers `event: error` in grammar instead of a
        hang. The `X-Fx1-Completion-Id` header only exists once the
        completion lands, so keepalived legs carry the same fingerprint
        in-band as `msg_<cid>` on `message_start`.
        """

        def _refusal(status: int, message: str) -> JSONResponse:
            return JSONResponse(anthropic_error_body(message, status), status_code=status)

        # resume parsing first — a malformed Last-Event-ID refuses before
        # any idempotency store work or model spend
        try:
            skip = _resume_skip(last_event_id, stream=body.stream, idempotency_key=idempotency_key)
        except ApiError as exc:
            return _refusal(exc.status_code, str(exc.detail))
        body_fp = _body_fp(body, headers=request.headers)
        key, replay = _idem_lookup(idempotency_key, anthropic_idem_store, body_fp)
        if skip and replay is None:
            return _refusal(409, _RESUME_MISS_MSG)
        if replay is not None:
            env = replay.envelope
            cid_replay = str(env["id"]).removeprefix("chatcmpl-")
            headers = _completion_headers(cid_replay, replay=True)
            if body.stream:
                return StreamingResponse(
                    anthropic_sse(env, model=body.model, skip=skip),
                    media_type="text/event-stream",
                    headers=headers,
                )
            return JSONResponse(anthropic_envelope(env, model=body.model), headers=headers)
        try:
            oai_body = OpenAIChatRequest.model_validate(anthropic_to_openai(body))
        except OpenAICompatError as exc:
            return _refusal(exc.status, str(exc))
        except ValidationError as exc:
            return _refusal(400, str(exc))

        def _generate() -> _ChatEnv:
            """The gated call packaged for the grace pipe — translation
            faults surface as HTTPException so the keepalived leg answers
            them in grammar, and the keyed replay pins only on a real
            completion."""
            try:
                env, cid = _openai_chat_core(oai_body, request.headers)
            except OpenAICompatError as exc:
                raise ApiError(exc.status, str(exc), code=exc.code) from exc
            if key is not None:
                anthropic_idem_store.put(key, body_fp, _OpenAIIdemRecord(envelope=env))
            return env, cid

        keepalive_s = float(getattr(app.state, "sse_keepalive_s", 15.0))
        try:
            outcome, pipe = _grace_stage(_generate, stream=body.stream, keepalive_s=keepalive_s)
        except ApiError as exc:
            return _refusal(exc.status_code, str(exc.detail))
        if outcome is not None:
            tag, payload = outcome
            if tag == "error":
                fault = cast("HTTPException", payload)
                return _refusal(fault.status_code, str(fault.detail))
            env_chat, cid = cast("_ChatEnv", payload)
            headers = _completion_headers(cid)
            if body.stream:
                return StreamingResponse(
                    anthropic_sse(env_chat, model=body.model),
                    media_type="text/event-stream",
                    headers={**headers, **_SSE_STREAM_HEADERS},
                )
            return JSONResponse(anthropic_envelope(env_chat, model=body.model), headers=headers)

        def _keepalived() -> Iterator[str]:
            """Past the grace window: unnumbered ``ping`` frames keep the
            connection live (they consume no ``id:`` slot, so a
            ``Last-Event-ID`` resume still counts only real events), the
            completion id rides in-band as ``msg_<cid>`` on
            ``message_start``, and a backend fault lands as an honest
            ``event: error`` in Anthropic grammar — never a hung stream."""
            assert pipe is not None
            while True:
                outcome = _grace_await(pipe, keepalive_s)
                if outcome is None:
                    yield 'event: ping\ndata: {"type":"ping"}\n\n'
                    continue
                tag, payload = outcome
                if tag == "error":
                    fault = cast("HTTPException", payload)
                    yield (
                        "event: error\ndata: "
                        + json.dumps(
                            anthropic_error_body(str(fault.detail), fault.status_code),
                            separators=(",", ":"),
                        )
                        + "\n\n"
                    )
                    return
                env_chat, _cid = cast("_ChatEnv", payload)
                yield from anthropic_sse(env_chat, model=body.model)
                return

        return StreamingResponse(
            _keepalived(),
            media_type="text/event-stream",
            headers=_SSE_STREAM_HEADERS,
        )

    # ------------------------------------------------------------------
    # Anthropic Message Batches — POST /v1/messages/batches
    # ------------------------------------------------------------------
    #
    # Anthropic's async surface: requests ride inline (no input file), the
    # batch runs on one jobs-executor slot, ``request_counts`` stays
    # all-``processing`` until the batch ends (the Anthropic contract —
    # tallies move only at the terminal transition), results are the
    # unordered JSONL of ``{custom_id, result}`` rows.
    #
    # Errors on these routes keep the Anthropic envelope — every path
    # under ``/v1/messages*`` projects through ``_v1_error_body``.

    def _abatch_row_fault(item: AnthropicBatchItem, exc: Exception) -> dict[str, Any]:
        """One item's fault → an ``errored`` result row — data, never a
        crash into the batch worker."""
        if isinstance(exc, OpenAICompatError):
            body = anthropic_error_body(str(exc), exc.status)
        elif isinstance(exc, ApiError):
            body = anthropic_error_body(str(exc.detail), exc.status_code)
        elif isinstance(exc, ValidationError):
            body = anthropic_error_body(str(exc), 400)
        else:
            body = anthropic_error_body(f"{type(exc).__name__}: {exc}", 500)
        # the row's `error` is the inner {type, message} object, not the
        # whole error envelope — Anthropic's results line shape
        return anthropic_batch_result(item.custom_id, {"type": "errored", "error": body["error"]})

    def _run_abatch_item(item: AnthropicBatchItem, batch: _AnthropicBatchRecord) -> dict[str, Any]:
        """One batch request through the same translate → gated chat core →
        anthropic envelope path the live ``/v1/messages`` route runs. The
        batch's ``_headers`` carry the submitter's X-Fx1-* backend choice
        into every line — never ambient env."""
        try:
            oai_body = OpenAIChatRequest.model_validate(anthropic_to_openai(item.params))
            env, _cid = _openai_chat_core(oai_body, batch._headers)
            return anthropic_batch_result(
                item.custom_id,
                {
                    "type": "succeeded",
                    "message": anthropic_envelope(env, model=item.params.model),
                },
            )
        except Exception as exc:  # noqa: BLE001 — per-item faults are rows
            return _abatch_row_fault(item, exc)

    def _abatch_unfinished(batch: _AnthropicBatchRecord) -> builtins.list[str]:
        """``custom_id``s with no result row yet (survives restart — the
        ids are journaled even though the item bodies are not)."""
        done: set[str] = set()
        for ln in batch.result_lines:
            try:
                done.add(str(json.loads(ln)["custom_id"]))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        return [cid for cid in batch.item_ids if cid not in done]

    def _abatch_tally(batch: _AnthropicBatchRecord) -> AnthropicBatchCounts:
        """Terminal counts, re-derived from the result rows — the numbers
        on the record can never disagree with what ``results`` serves."""
        counts = AnthropicBatchCounts()
        for ln in batch.result_lines:
            try:
                typ = str(json.loads(ln)["result"]["type"])
            except (json.JSONDecodeError, KeyError, TypeError):
                counts.errored += 1
                continue
            if typ == "succeeded":
                counts.succeeded += 1
            elif typ == "errored":
                counts.errored += 1
            elif typ == "canceled":
                counts.canceled += 1
            else:
                counts.expired += 1
        return counts

    def _abatch_finish(
        batch: _AnthropicBatchRecord, kind: Literal["canceled", "expired"] | None = None
    ) -> None:
        """End the batch now: every unfinished ``custom_id`` lands a
        ``kind`` row (cancel mid-flight / expiry-on-read), counts project
        to the terminal split, and the terminal webhook fires once."""
        # the row lock makes the unfinished-scan + ended flip atomic vs a
        # worker append — a row the worker lands after ``ended`` is dropped
        # there (Anthropic semantics: results are final once ended).
        with batch._row_lock:
            if batch.status == "ended":
                return
            if kind is not None:
                for cid in _abatch_unfinished(batch):
                    batch.result_lines.append(
                        json.dumps(
                            anthropic_batch_result(cid, {"type": kind}),
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
            batch.request_counts = _abatch_tally(batch)
            batch.status = "ended"
            batch.ended_at = int(time.time())
        abatch_store.mark(batch)
        _abatch_webhook(batch)
        # Delivery outcome is part of the public record.  Persist it after
        # the callback attempt as well as the terminal state before it.
        abatch_store.mark(batch)

    def _abatch_webhook(batch: _AnthropicBatchRecord) -> None:
        """Fire-once terminal webhook: the projected ``message_batch``
        envelope (the same shape GET returns) POSTs to ``callback_url``
        once — expiry-on-read is a terminal transition too."""
        if not batch.callback_url or batch.status != "ended":
            return
        _deliver_callback(
            batch,
            body=json.dumps(anthropic_batch_object(batch.model_dump(mode="json"))).encode(),
        )

    def _exec_abatch(batch: _AnthropicBatchRecord) -> None:
        """Worker: run every request item through the gated chat core in
        submit order; a cancel flag between items ends the batch with
        ``canceled`` rows for the tail. Holds ONE inflight slot for the
        batch's lifetime (the jobs-channel contract) — released in
        ``finally`` so a cancel or worker fault never leaks it."""
        ctx_key = _REQUEST_KEY_ID.set(batch._key_id)
        try:
            for item in batch._items:
                if batch._cancel.is_set():
                    _abatch_finish(batch, "canceled")
                    return
                row = _run_abatch_item(item, batch)
                with batch._row_lock:
                    if batch.status == "ended":
                        return  # expiry ended the batch mid-flight — the
                        # computed row drops; the expired row already stands
                    batch.result_lines.append(
                        json.dumps(row, sort_keys=True, separators=(",", ":"))
                    )
                # Result rows are declared durable.  Persist each committed
                # row rather than waiting for the terminal transition: a
                # crash after item N must not turn already-served work into
                # restart-error rows.
                abatch_store.mark(batch)
            _abatch_finish(batch)
        except Exception as exc:  # noqa: BLE001 — a worker fault ends the batch, not the process
            with batch._row_lock:
                if batch.status != "ended":
                    for cid in _abatch_unfinished(batch):
                        batch.result_lines.append(
                            json.dumps(
                                anthropic_batch_result(
                                    cid,
                                    {
                                        "type": "errored",
                                        "error": anthropic_error_body(
                                            f"{type(exc).__name__}: {exc}", 500
                                        )["error"],
                                    },
                                ),
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                        )
                    batch.request_counts = _abatch_tally(batch)
                    batch.status = "ended"
                    batch.ended_at = int(time.time())
                    should_finish = True
                else:
                    should_finish = False
            if should_finish:
                abatch_store.mark(batch)
                _abatch_webhook(batch)
                abatch_store.mark(batch)
        finally:
            _REQUEST_KEY_ID.reset(ctx_key)
            metrics.release()
            inflight.release()

    def _abatch_project(batch: _AnthropicBatchRecord) -> dict[str, Any]:
        """Expiry check + envelope projection — a batch past its
        ``expires_at`` ends on read, unfinished rows ``expired``."""
        if batch.status != "ended" and time.time() > batch.expires_at:
            batch._cancel.set()  # the worker exits its loop at the next item
            _abatch_finish(batch, "expired")
        with batch._row_lock:
            return anthropic_batch_object(batch.model_dump(mode="json"))

    def _submit_after_persist(
        *,
        persist: Callable[[], None],
        execute: Callable[[], None],
    ) -> None:
        """Queue a batch worker, but do not run it before durable publication.

        ``ThreadPoolExecutor.submit`` can start the callable immediately. A
        submit-then-persist sequence therefore lets provider work escape even
        when the journal append fails and the create request returns an error.
        The launch barrier keeps the executor's bounded admission semantics
        while ensuring that work becomes observable and recoverable first.
        """
        published = threading.Event()
        abandoned = threading.Event()

        def run_when_published() -> None:
            published.wait()
            if not abandoned.is_set():
                execute()

        try:
            jobs_executor.submit(run_when_published)
        except RuntimeError as exc:
            metrics.release()
            inflight.release()
            raise ApiError(503, "job executor unavailable", code="over_capacity") from exc
        try:
            persist()
        except Exception:
            # Wake the already-admitted wrapper, but forbid it from touching
            # the provider or mutating an unjournaled record.
            abandoned.set()
            published.set()
            metrics.release()
            inflight.release()
            raise
        published.set()

    async def _anthropic_batch_idem_claim(
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> AsyncIterator[None]:
        """Serialize first use of one batch idempotency key.

        The claim is an async dependency so waiters yield the event loop
        instead of consuming every sync-handler worker while the winner
        creates and stores the batch response.
        """
        key = _idem_key(idempotency_key)
        async with anthropic_idem_store.async_claim_lock(_idem_scope(key)):
            yield

    @app.post(
        "/v1/messages/batches",
        tags=["anthropic"],
        operation_id="anthropic_batches_create",
    )
    def anthropic_batches_create(
        body: AnthropicBatchCreate,
        request: Request,
        _idem_claim_held: None = Depends(_anthropic_batch_idem_claim),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> JSONResponse:
        """Submit an Anthropic message batch — ``requests`` ride inline
        (each ``{custom_id, params}`` carrying a full ``/v1/messages``
        body; ``stream`` inside a batch refuses at validation). One
        worker slot runs every request through the gated chat core;
        ``expires_at`` is +24h. ``request_counts`` stays all-processing
        until the batch ends; results then land as one JSONL row per
        ``custom_id`` (``succeeded``/``errored``/``canceled``/``expired``).
        ``callback_url``/``callback_secret`` are the fx1 webhook
        extension — the terminal ``message_batch`` envelope POSTs once,
        HMAC-signed when the secret is set."""
        body_fp = _body_fp(body, headers=request.headers)
        key, replay = _idem_lookup(idempotency_key, anthropic_idem_store, body_fp)
        if replay is not None:
            return JSONResponse(replay.envelope, headers={"X-Fx1-Idempotent-Replay": "true"})
        if len(body.requests) > batch_line_max:
            raise ApiError(
                400,
                f"{len(body.requests)} requests exceeds the {batch_line_max} cap",
                code="batch_input_limit",
            )
        if metrics.draining.is_set():
            raise ApiError(503, "harness is draining — no new work accepted", code="draining")
        if not inflight.acquire(blocking=False):
            raise ApiError(
                503,
                "harness at max_inflight — retry later",
                code="over_capacity",
                headers={"Retry-After": "1"},
            )
        metrics.acquire()
        now = int(time.time())
        batch = _AnthropicBatchRecord(
            batch_id=f"msgbatch_{uuid.uuid4().hex}",
            created_at=now,
            expires_at=now + 86400,
            request_counts=AnthropicBatchCounts(processing=len(body.requests)),
            item_ids=[it.custom_id for it in body.requests],
            callback_url=body.callback_url,
        )
        batch._callback_secret = body.callback_secret
        batch._items = list(body.requests)
        # the caller's X-Fx1-* routing headers apply to every item — the
        # batch inherits the submitter's backend choice, never ambient env.
        batch._headers = {
            k: v for k, v in request.headers.items() if k.lower().startswith("x-fx1-")
        }
        batch._key_id = _REQUEST_KEY_ID.get()
        _submit_after_persist(
            persist=lambda: abatch_store.put(batch),
            execute=lambda: _exec_abatch(batch),
        )
        env = _abatch_project(batch)
        if key is not None:
            anthropic_idem_store.put(key, body_fp, _OpenAIIdemRecord(envelope=env))
        return JSONResponse(env)

    @app.get(
        "/v1/messages/batches",
        tags=["anthropic"],
        operation_id="anthropic_batches_list",
    )
    def anthropic_batches_list(
        limit: int = Query(default=20, ge=1, le=100),
        before_id: str | None = Query(default=None),
        after_id: str | None = Query(default=None),
    ) -> JSONResponse:
        """Newest-first batch listing; ``after_id`` pages to entries
        older than the cursor id (the page that follows it in list
        order), ``before_id`` to entries newer than it — the same
        cursor grammar ``/v1/models`` speaks, so the stock SDK's
        auto-pagination walks the full list."""
        items = abatch_store.list()
        if after_id is not None:
            idx = next((i for i, b in enumerate(items) if b.batch_id == after_id), None)
            items = items[idx + 1 :] if idx is not None else []
        if before_id is not None:
            idx = next((i for i, b in enumerate(items) if b.batch_id == before_id), None)
            items = items[:idx] if idx is not None else []
        # ``before_id`` pages *backward* — the tail of the remaining
        # window, so first_id chains through the list the way after_id
        # chains forward through last_id.
        page = items[-limit:] if before_id is not None else items[:limit]
        return JSONResponse(
            {
                "data": [_abatch_project(b) for b in page],
                "first_id": page[0].batch_id if page else None,
                "last_id": page[-1].batch_id if page else None,
                "has_more": len(items) > limit,
            }
        )

    @app.get(
        "/v1/messages/batches/{batch_id}",
        tags=["anthropic"],
        operation_id="anthropic_batches_get",
    )
    def anthropic_batches_get(batch_id: str) -> JSONResponse:
        batch = abatch_store.get(batch_id)
        if batch is None:
            raise ApiError(404, f"message batch {batch_id!r} not found", code="not_found")
        return JSONResponse(_abatch_project(batch))

    @app.post(
        "/v1/messages/batches/{batch_id}/cancel",
        tags=["anthropic"],
        operation_id="anthropic_batches_cancel",
    )
    def anthropic_batches_cancel(batch_id: str) -> JSONResponse:
        """Cooperative cancel — the worker checks the flag between items;
        an in-flight item finishes, then the batch ends with ``canceled``
        rows for the unprocessed tail."""
        batch = abatch_store.get(batch_id)
        if batch is None:
            raise ApiError(404, f"message batch {batch_id!r} not found", code="not_found")
        with batch._row_lock:
            if batch.status == "ended":
                raise ApiError(
                    400,
                    f"message batch {batch_id!r} has already ended",
                    code="invalid_request",
                )
            if batch.status == "canceling":
                already_canceling = True
            else:
                already_canceling = False
                batch._cancel.set()
                batch.status = "canceling"
                batch.cancel_initiated_at = int(time.time())
        if already_canceling:
            return JSONResponse(_abatch_project(batch))
        abatch_store.mark(batch)
        return JSONResponse(_abatch_project(batch))

    @app.delete(
        "/v1/messages/batches/{batch_id}",
        tags=["anthropic"],
        operation_id="anthropic_batches_delete",
    )
    def anthropic_batches_delete(batch_id: str) -> JSONResponse:
        """Delete an ended batch (Anthropic refuses delete mid-flight —
        cancel first)."""
        batch = abatch_store.get(batch_id)
        if batch is None:
            raise ApiError(404, f"message batch {batch_id!r} not found", code="not_found")
        _abatch_project(batch)  # expiry is a terminal transition too
        if batch.status != "ended":
            raise ApiError(
                400,
                f"message batch {batch_id!r} is still {batch.status} — cancel it first",
                code="invalid_request",
            )
        abatch_store.delete(batch_id)
        return JSONResponse({"id": batch_id, "type": "message_batch_deleted"})

    @app.get(
        "/v1/messages/batches/{batch_id}/results",
        tags=["anthropic"],
        operation_id="anthropic_batches_results",
    )
    def anthropic_batches_results(batch_id: str) -> Response:
        """Stream the batch's results as ``.jsonl`` — one
        ``{custom_id, result}`` object per request, unordered. Available
        only once the batch has ``ended`` (``results_url`` on the batch
        object points here)."""
        batch = abatch_store.get(batch_id)
        if batch is None:
            raise ApiError(404, f"message batch {batch_id!r} not found", code="not_found")
        _abatch_project(batch)
        if batch.status != "ended":
            raise ApiError(
                400,
                f"message batch {batch_id!r} results are available once the batch has ended",
                code="invalid_request",
            )
        content = ("\n".join(batch.result_lines) + "\n") if batch.result_lines else ""
        return Response(content=content, media_type="application/jsonl")

    # ------------------------------------------------------------------
    # Anthropic count_tokens — POST /v1/messages/count_tokens
    # ------------------------------------------------------------------

    @app.post(
        "/v1/messages/count_tokens",
        tags=["anthropic"],
        operation_id="anthropic_count_tokens",
    )
    def anthropic_count_tokens_route(
        body: AnthropicCountTokensRequest,
        request: Request,
    ) -> JSONResponse:
        """Anthropic's ``POST /v1/messages/count_tokens`` — the provider's
        own tokenizer count over the message channel, ``{input_tokens: N}``.

        The request validates the same contract as ``/v1/messages``
        (user-first alternation, system shape, unsupported knobs refuse),
        then the resolved backend answers through its own tokenize route
        — vLLM/SGLang-style ``/tokenize`` on BYOK and local engines,
        Moonshot's ``tokenizers/estimate-token-count`` on the hosted link.
        A backend or endpoint without the channel fails closed 501
        ``api_error`` — the harness never estimates. ``tools`` /
        ``tool_choice`` refuse 400: provider tokenize routes see only the
        message channel, so counting a toolful request would undercount.
        ``X-Fx1-*`` headers and the ``fx1`` extension pick the link
        exactly like the create path; ``X-Fx1-Timeout`` caps the call."""
        if body.tools is not None or body.tool_choice is not None:
            raise ApiError(
                400,
                "count_tokens covers the message channel — tools/tool_choice "
                "have no tokenize route to honor them",
                code="invalid_request",
            )
        hdrs = {str(k).lower(): str(v) for k, v in request.headers.items()}
        try:
            backend_name, _fb, checkpoint_dir, byok, _sv = _resolve_openai_link(
                body.model,
                body.fx1,
                hdrs,
                ft_resolver=ft_store.checkpoint_for,
                require_known_model=True,
            )
            timeout_s = _resolve_timeout(body.fx1.timeout_s if body.fx1 is not None else None, hdrs)
        except OpenAICompatError as exc:
            raise ApiError(exc.status, str(exc), code=exc.code) from exc
        backend = resolve_backend(
            backend_name,
            checkpoint_dir,
            byok.model_dump() if byok is not None else None,
            timeout_s,
        )
        if not isinstance(backend, TokenCountingBackend):
            raise ApiError(
                501,
                f"backend {backend_name!r} has no tokenize channel",
                code="not_implemented",
            )
        try:
            n = backend.count_tokens(anthropic_count_messages(body))
        except TokenCountUnavailableError as exc:
            raise ApiError(501, str(exc), code="not_implemented") from exc
        except RuntimeError as exc:
            raise ApiError(502, str(exc), code="backend_failure") from exc
        return JSONResponse({"input_tokens": n})

    @app.post(
        "/v1/completions",
        # the JSON path returns the legacy ``text_completion`` object;
        # stream=true returns the legacy SSE chunk grammar
        responses={200: {"model": None}},
        tags=["openai"],
        operation_id="openai_completions",
    )
    def openai_completions(
        body: OpenAICompletionRequest,
        request: Request,
        _idem_claim_held: None = Depends(_legacy_idem_claim),
        _slot_held: None = Depends(slot),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> Response:
        """Legacy ``/v1/completions`` — the ``text_completion`` surface
        ``client.completions.create`` and pre-chat agents still target.

        Each ``prompt`` element becomes one user turn through the shared
        gated pipeline (same honesty gate, fail-closed validation, and
        completion-log metering as ``/v1/chat/completions``); ``n`` repeats
        within an element, so ``prompt=[a,b]`` with ``n=2`` lands four
        flat choices. ``echo`` prepends the prompt to each choice's text.
        ``suffix``, ``best_of`` and ``logprobs`` refuse 422 — the
        pipeline has no FIM head and no token-logprob scorer.

        ``store`` is tolerated and ignored (legacy completions have no
        retrieval twin); ``Idempotency-Key`` replay and ``Last-Event-ID``
        stream resume work exactly like the chat surface — a pinned call
        replays byte-identically (JSON or SSE) and a resumed keyed stream
        drops frames at or below the delivered index. Slow generations
        keepalived under ``sse_keepalive_s`` emit ``: keepalive`` comment
        frames, and a mid-window backend fault answers an in-band
        ``{error}`` + ``[DONE]`` — same contract as the chat surface.
        """
        # resume parsing first — a malformed Last-Event-ID fails before
        # any idempotency store work or model spend
        skip = _resume_skip(last_event_id, stream=body.stream, idempotency_key=idempotency_key)
        body_fp = _body_fp(body, headers=request.headers)
        key, replay = _idem_lookup(idempotency_key, legacy_idem_store, body_fp)
        if skip and replay is None:
            raise ApiError(409, _RESUME_MISS_MSG, code="resume_miss")
        if replay is not None:
            env_legacy = replay.envelope
            cid_replay = str(env_legacy["id"]).removeprefix("cmpl-")
            headers = _completion_headers(cid_replay, replay=True)
            if body.stream:
                return StreamingResponse(
                    _legacy_sse(env_legacy, body=body, skip=skip),
                    media_type="text/event-stream",
                    headers=headers,
                )
            return JSONResponse(env_legacy, headers=headers)
        prompts = [body.prompt] if isinstance(body.prompt, str) else list(body.prompt)

        def _generate() -> _ChatEnv:
            """The gated call packaged for the grace pipe — one chat call
            per prompt element, translation faults surfacing as ApiError,
            the keyed replay pinning only on a real completion."""
            try:
                envs: list[dict[str, Any]] = []
                cid = ""
                for prompt_text in prompts:
                    env_chat, cid = _openai_chat_core(
                        legacy_to_chat(body, prompt_text), request.headers
                    )
                    envs.append(env_chat)
                env_legacy = openai_completion_envelope(
                    cid=cid, envs=envs, prompts=prompts, echo=body.echo
                )
            except OpenAICompatError as exc:
                raise ApiError(exc.status, str(exc), code=exc.code) from exc
            except ValidationError as exc:
                raise ApiError(400, str(exc)) from exc
            if key is not None:
                legacy_idem_store.put(key, body_fp, _OpenAIIdemRecord(envelope=env_legacy))
            return env_legacy, cid

        keepalive_s = float(getattr(app.state, "sse_keepalive_s", 15.0))
        outcome, pipe = _grace_stage(_generate, stream=body.stream, keepalive_s=keepalive_s)
        if outcome is not None:
            tag, payload = outcome
            if tag == "error":
                raise cast("HTTPException", payload)
            env_legacy, cid = cast("_ChatEnv", payload)
            headers = _completion_headers(cid)
            if body.stream:
                return StreamingResponse(
                    _legacy_sse(env_legacy, body=body),
                    media_type="text/event-stream",
                    headers={**headers, **_SSE_STREAM_HEADERS},
                )
            return JSONResponse(env_legacy, headers=headers)

        def _keepalived() -> Iterator[str]:
            """Past the grace window: ``: keepalive`` comment frames hold
            the connection without consuming ``id:`` slots, and a backend
            fault lands as an in-band ``{error}`` + ``[DONE]`` — never a
            hang."""
            assert pipe is not None
            while True:
                outcome = _grace_await(pipe, keepalive_s)
                if outcome is None:
                    yield ": keepalive\n\n"
                    continue
                tag, payload = outcome
                if tag == "error":
                    fault = cast("HTTPException", payload)
                    yield (
                        _SSE_DATA_PREFIX
                        + json.dumps(
                            openai_error_body(
                                str(fault.detail), fault.status_code, _err_code(fault)
                            ),
                            separators=(",", ":"),
                        )
                        + "\n\n"
                    )
                    yield _SSE_DONE
                    return
                env_legacy, _cid = cast("_ChatEnv", payload)
                yield from _legacy_sse(env_legacy, body=body)
                return

        return StreamingResponse(
            _keepalived(),
            media_type="text/event-stream",
            headers=_SSE_STREAM_HEADERS,
        )

    @app.post(
        "/v1/responses",
        # the JSON path returns the ``response`` object; stream=true returns
        # the Responses SSE event grammar (response_model stays None)
        responses={200: {"model": None}},
        tags=["openai"],
        operation_id="openai_responses",
    )
    def openai_responses(
        body: OpenAIResponseRequest,
        request: Request,
        _idem_claim_held: None = Depends(_openai_idem_claim),
        _slot_held: None = Depends(slot),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> Response:
        """OpenAI Responses API over the same gated pipeline.

        ``input`` is a string or a list of message items
        (``{type: "message", role, content: [{type: "input_text", text}]}``
        or the shorthand ``{role, content: "..."}``); ``instructions``
        prepends a system turn; ``developer`` roles map to ``system``.
        Slow generations keepalived under ``sse_keepalive_s`` emit
        ``: keepalive`` comment frames (no ``id:`` slot consumed — resume
        still counts only real events), then the normal event sequence;
        a mid-window backend fault answers ``event: error`` in grammar.
        ``max_output_tokens`` lands on the decode cap, ``reasoning.effort``
        on the reasoning hint, ``text.format`` on the post-validated
        ``response_format`` channel (a violation is a provider-side 502),
        ``user``/``safety_identifier``/``metadata`` stamp the audit record.

        ``tools`` (flattened Responses specs), ``tool_choice``
        (``none``/``auto``/``required`` or ``{type: "function", name}``),
        ``parallel_tool_calls``, ``function_call``/``function_call_output``
        input items, and ``function_call`` output items are first-class —
        ``max_tool_calls`` bounds the calls one response may carry; a turn
        over the cap truncates and lands ``status: 'incomplete'`` with
        ``incomplete_details.reason == 'max_tool_calls'`` (the stream's
        terminal frame is ``response.incomplete``) —
        the same tool channel as ``/v1/chat/completions`` under its own
        grammar (a link without the channel answers 501). Same fail-closed
        rule as chat completions for the rest: ``truncation``/``include``
        refuse at validation (422). ``background=true`` (with the default
        ``store=true`` and no ``stream``) queues the work on the jobs
        executor and returns the ``queued`` response object — poll
        ``GET /v1/responses/{id}`` or ``POST .../cancel`` — while
        ``background``+``store=false`` is a 400 and ``background``+
        ``stream`` runs the normal stream (a stream is already async).
        ``previous_response_id`` chains the turn onto a stored ``response``
        — the model runs on the parent's stored items + its output + this
        request's input, and the child's ``input_items`` carry the whole
        history. An unknown, deleted, or ``store=false`` parent fails
        closed ``400 previous_response_not_found``.
        ``conversation`` (a ``conv_*`` id or ``{id: …}``) anchors to a
        named container instead — the conv's accumulated items are the
        context and the turn appends onto it when it completes; the two
        anchors are mutually exclusive (422) and an unknown conv fails
        closed ``400 conversation_not_found``.
        ``store`` governs the retrieval index — ``store=false`` keeps the
        call out of ``GET /v1/responses/{id}`` (the audit ledger still
        records it).
        ``Idempotency-Key`` and ``Last-Event-ID`` resume behave exactly as
        on ``/v1/chat/completions`` (the stream's terminal frame is
        ``response.completed``, not ``[DONE]``).
        """
        skip = _resume_skip(last_event_id, stream=body.stream, idempotency_key=idempotency_key)
        body_fp = _body_fp(body, headers=request.headers)
        key, replay = _idem_lookup(idempotency_key, openai_idem_store, body_fp)
        if skip and replay is None:
            raise ApiError(409, _RESUME_MISS_MSG, code="resume_miss")

        def _resp_sse_from(env: dict[str, Any], drop: int) -> Iterator[str]:
            text, item_id, call_list, env_search, env_lp = response_output_pieces(env)
            return _responses_sse(
                body,
                content=text,
                rid=env["id"],
                item_id=item_id,
                model=env.get("model"),
                usage=env.get("_fx1_usage"),
                created=env.get("created_at"),
                skip=drop,
                call_items=call_list,
                search_items=env_search,
                logprobs=env_lp,
                # a truncated turn replays its terminal event too —
                # response.incomplete, not response.completed
                final_status=("incomplete" if env.get("status") == "incomplete" else "completed"),
                incomplete_details=env.get("incomplete_details"),
            )

        if replay is not None:
            env = replay.envelope
            if body.store is not False:
                # Refresh retrieval order while retaining a newer live
                # verdict. Only an absent record rehydrates from cache.
                # Cache-only raw usage remains available for stream replay;
                # every field present in the live envelope takes precedence.
                env = {**env, **envelope_store.repin(env)}
            headers = {"X-Fx1-Idempotent-Replay": "true"}
            if env.get("_fx1_completion_id"):
                headers["X-Fx1-Completion-Id"] = str(env["_fx1_completion_id"])
                _rsha = _completion_receipt_sha(headers["X-Fx1-Completion-Id"])
                if _rsha is not None:
                    headers["X-Fx1-Receipt-Sha256"] = _rsha
            if body.stream:
                return StreamingResponse(
                    _resp_sse_from(env, skip),
                    media_type="text/event-stream",
                    headers=headers,
                )
            out_env = {
                k: v for k, v in env.items() if k not in ("_fx1_completion_id", "_fx1_usage")
            }
            return JSONResponse(out_env, headers=headers)
        if body.background and not body.stream:
            if body.store is False:
                raise ApiError(
                    400,
                    "background=true needs store=true — a background response is "
                    "only reachable through the retrieval index",
                    code="background_requires_store",
                )
            if metrics.draining.is_set():
                raise ApiError(
                    503,
                    "harness is draining — no new work accepted",
                    code="draining",
                )
            # validate the chain anchor at submit so a bad parent fails now,
            # not in the worker; the same check re-runs inside the core so a
            # parent deleted mid-flight still fails the work honestly
            if body.previous_response_id is not None:
                prev = envelope_store.get(body.previous_response_id)
                if prev is None or prev.get("object") != "response":
                    raise ApiError(
                        400,
                        f"previous_response_id {body.previous_response_id!r} not "
                        "found — the chain parent must be a stored response "
                        "(store=true)",
                        code="previous_response_not_found",
                    )
                submit_items = chained_response_input(
                    prev,
                    envelope_store.get_items(body.previous_response_id, "input_items") or [],
                    body.input,
                )
            elif (conv_cid_sub := conversation_id_of(body.conversation)) is not None:
                # same contract for the conv anchor: a missing conv fails
                # now at submit, not inside the worker — the core re-checks
                # so a delete mid-flight still fails the work honestly
                conv = conv_store.get(conv_cid_sub)
                if conv is None or conv.get("object") != "conversation":
                    raise ApiError(
                        400,
                        f"conversation {conv_cid_sub!r} not found — create it "
                        "with POST /v1/conversations first",
                        code="conversation_not_found",
                    )
                submit_items = chained_response_input(
                    {"output": []},
                    conv_store.get_items(conv_cid_sub, "items") or [],
                    body.input,
                )
            else:
                submit_items = response_input_item_dicts(body.input)
            rid = f"resp_{uuid.uuid4().hex}"
            queued = openai_response_object(
                rid=rid,
                item_id="",
                content="",
                body=body,
                model=body.model,
                usage=None,
                status="queued",
            )
            envelope_store.put(
                queued,
                items={"input_items": response_input_items_for_store(submit_items, rid=rid)},
            )
            cancel_ev = threading.Event()
            bg_cancel[rid] = cancel_ev
            # Key attribution must be captured here — worker threads in the
            # pool do not inherit this request's contextvars.
            bg_key_id = _REQUEST_KEY_ID.get()

            def _bg_run() -> None:
                inflight.acquire()
                key_token = _REQUEST_KEY_ID.set(bg_key_id)
                try:
                    if cancel_ev.is_set():
                        return
                    # A cancellation or deletion after the event check
                    # must prevent the queued backend call from starting.
                    if not envelope_store.transition_status(
                        rid, expect={"queued"}, status="in_progress"
                    ):
                        return
                    try:
                        env_done, cid_done, usage_done = _openai_response_core(
                            body, request.headers, rid=rid, created=int(queued["created_at"])
                        )
                        if key is not None:
                            openai_idem_store.put(
                                key,
                                body_fp,
                                _OpenAIIdemRecord(
                                    envelope={
                                        **env_done,
                                        "_fx1_completion_id": cid_done,
                                        "_fx1_usage": usage_done,
                                    }
                                ),
                            )
                    except OpenAICompatError as exc:
                        _bg_fail(
                            rid,
                            error={"message": str(exc), "code": exc.code},
                        )
                    except ApiError as exc:
                        _bg_fail(
                            rid,
                            error={"message": str(exc.detail), "code": _err_code(exc)},
                        )
                    except Exception as exc:  # noqa: BLE001 — worker faults land on the record
                        _bg_fail(
                            rid,
                            error={
                                "message": f"{type(exc).__name__}: {exc}",
                                "code": "internal_error",
                            },
                        )
                finally:
                    _REQUEST_KEY_ID.reset(key_token)
                    inflight.release()
                    bg_cancel.pop(rid, None)

            def _bg_fail(response_id: str, *, error: dict[str, Any]) -> None:
                if cancel_ev.is_set():
                    return  # the cancel verdict stands
                cur = envelope_store.get(response_id)
                if cur is None:
                    return
                cur["status"] = "failed"
                cur["error"] = error
                envelope_store.put_unless_status(
                    cur, forbidden=OPENAI_RESPONSE_TERMINAL, require_existing=True
                )

            if key is not None:
                openai_idem_store.put(
                    key,
                    body_fp,
                    _OpenAIIdemRecord(
                        envelope={
                            **queued,
                            "_fx1_completion_id": None,
                            "_fx1_usage": None,
                        }
                    ),
                )
            try:
                jobs_executor.submit(_bg_run)
            except RuntimeError as exc:  # executor gone (shutdown race)
                envelope_store.delete(rid)
                bg_cancel.pop(rid, None)
                raise ApiError(503, "job executor unavailable", code="over_capacity") from exc
            return JSONResponse(queued, status_code=200)

        def _generate() -> tuple[dict[str, Any], str, dict[str, int] | None]:
            """The gated call packaged for the grace pipe — translation
            faults surface as ApiError, and the keyed replay pins the
            completion id + raw usage into the stored envelope (both
            stripped before the JSON leaves)."""
            try:
                env, cid, usage = _openai_response_core(body, request.headers)
            except OpenAICompatError as exc:
                raise ApiError(exc.status, str(exc), code=exc.code) from exc
            if key is not None:
                openai_idem_store.put(
                    key,
                    body_fp,
                    _OpenAIIdemRecord(
                        envelope={
                            **env,
                            "_fx1_completion_id": cid,
                            "_fx1_usage": usage,
                        }
                    ),
                )
            return env, cid, usage

        def _sse_frames(env: dict[str, Any], usage: dict[str, int] | None) -> Iterator[str]:
            (
                msg_text,
                msg_item_id,
                call_list,
                env_search_items,
                env_logprobs,
            ) = response_output_pieces(env)
            return _responses_sse(
                body,
                content=msg_text,
                rid=str(env["id"]),
                item_id=msg_item_id,
                model=env.get("model"),
                usage=usage,
                created=int(env["created_at"]),
                call_items=call_list,
                search_items=env_search_items,
                logprobs=env_logprobs,
                final_status=str(env.get("status") or "completed"),
                incomplete_details=env.get("incomplete_details"),
            )

        keepalive_s = float(getattr(app.state, "sse_keepalive_s", 15.0))
        outcome, pipe = _grace_stage(_generate, stream=body.stream, keepalive_s=keepalive_s)
        if outcome is not None:
            tag, payload = outcome
            if tag == "error":
                raise cast("HTTPException", payload)
            envelope, cid, usage = cast(
                "tuple[dict[str, Any], str, dict[str, int] | None]", payload
            )
            headers = {"X-Fx1-Completion-Id": cid}
            _rsha = _completion_receipt_sha(cid)
            if _rsha is not None:
                headers["X-Fx1-Receipt-Sha256"] = _rsha
            if body.stream:
                return StreamingResponse(
                    _sse_frames(envelope, usage),
                    media_type="text/event-stream",
                    headers={**headers, **_SSE_STREAM_HEADERS},
                )
            return JSONResponse(envelope, headers=headers)

        def _keepalived() -> Iterator[str]:
            """Past the grace window: ``: keepalive`` comment frames hold
            the connection without consuming ``id:`` slots (resume counts
            only real events), the ``resp_<id>`` of the response rides
            in-band on every frame, and a backend fault lands as an
            ``event: error`` in grammar — never a hung stream."""
            assert pipe is not None
            while True:
                outcome = _grace_await(pipe, keepalive_s)
                if outcome is None:
                    yield ": keepalive\n\n"
                    continue
                tag, payload = outcome
                if tag == "error":
                    fault = cast("HTTPException", payload)
                    yield (
                        "event: error\ndata: "
                        + json.dumps(
                            openai_error_body(
                                str(fault.detail), fault.status_code, _err_code(fault)
                            ),
                            separators=(",", ":"),
                        )
                        + "\n\n"
                    )
                    return
                env, _cid, usage = cast(
                    "tuple[dict[str, Any], str, dict[str, int] | None]", payload
                )
                yield from _sse_frames(env, usage)
                return

        return StreamingResponse(
            _keepalived(),
            media_type="text/event-stream",
            headers=_SSE_STREAM_HEADERS,
        )

    # --- /v1 retrieval tier --------------------------------------------------
    # The OpenAI retrieval surface: GET by the issued id returns the stored
    # envelope; DELETE drops it. ``store=false`` on the original call keeps
    # it out of the index (the completion log still records the call — the
    # flag governs retrieval, never evidence).

    def _stored_envelope(envelope_id: str, *, object_: str) -> dict[str, Any]:
        env = envelope_store.get(envelope_id)
        if env is None or env.get("object") != object_:
            raise ApiError(
                404,
                f"{envelope_id!r} not found — evicted, deleted, or sent with store=false",
                code="not_found",
            )
        return {k: v for k, v in env.items() if not k.startswith("_fx1_")}

    def _drop_envelope(envelope_id: str, *, object_: str) -> dict[str, Any]:
        env = envelope_store.get(envelope_id)
        if env is None or env.get("object") != object_:
            raise ApiError(404, f"{envelope_id!r} not found", code="not_found")
        envelope_store.delete(envelope_id)
        return {"id": envelope_id, "object": f"{object_}.deleted", "deleted": True}

    @app.get(
        "/v1/chat/completions/{completion_id}", tags=["openai"], operation_id="openai_chat_retrieve"
    )
    def openai_chat_retrieve(completion_id: str) -> dict[str, Any]:
        """Retrieve a stored chat completion (``chatcmpl-…``)."""
        return _stored_envelope(completion_id, object_=_OBJ_CHAT_COMPLETION)

    @app.post(
        "/v1/chat/completions/{completion_id}",
        response_model=None,
        tags=["openai"],
        operation_id="openai_chat_update",
    )
    def openai_chat_update(completion_id: str, body: OpenAIChatUpdate) -> dict[str, Any]:
        """Update a stored chat completion's ``metadata`` (the only
        mutable field — choices/usage are sealed at creation)."""
        _drain_refusal(metrics)
        env = envelope_store.get(completion_id)
        if env is None or env.get("object") != _OBJ_CHAT_COMPLETION:
            raise ApiError(
                404,
                f"{completion_id!r} not found — evicted, deleted, or sent with store=false",
                code="not_found",
            )
        updated = envelope_store.update_metadata(
            completion_id, dict(body.metadata) if body.metadata is not None else {}
        )
        if updated is None:
            raise ApiError(
                404,
                f"{completion_id!r} not found — evicted, deleted, or sent with store=false",
                code="not_found",
            )
        return {k: v for k, v in updated.items() if not k.startswith("_fx1_")}

    @app.delete(
        "/v1/chat/completions/{completion_id}",
        response_model=None,
        tags=["openai"],
        operation_id="openai_chat_delete",
    )
    def openai_chat_delete(completion_id: str) -> dict[str, Any]:
        """Drop a stored chat completion from the retrieval index."""
        return _drop_envelope(completion_id, object_=_OBJ_CHAT_COMPLETION)

    @app.get(
        "/v1/responses/{response_id}",
        response_model=None,
        tags=["openai"],
        operation_id="openai_responses_retrieve",
    )
    def openai_response_retrieve(
        response_id: str,
        stream: bool = Query(default=False),
        starting_after: int | None = Query(default=None, ge=0),
        timeout_s: float = Query(default=600.0, ge=1.0, le=3600.0),
    ) -> Response:
        """Retrieve a stored response object (``resp_…``).

        ``stream=true`` replays the response as the Responses SSE event
        grammar — how a client re-attaches to a ``background:true`` call it
        disconnected from, or re-streams a completed one: a terminal
        envelope emits the full recorded sequence, a still
        ``queued``/``in_progress`` envelope emits its prelude then
        live-follows until the terminal frame
        (``response.completed``/``incomplete``/``failed``/``cancelled``) or
        the ``timeout_s`` deadline. ``starting_after`` resumes past
        sequence number N — the cursor is the frame's ``id:``. Unknown,
        deleted, or ``store=false`` ids answer the same 404 ``not_found``
        envelope as the JSON read."""
        env = envelope_store.get(response_id)
        if env is None or env.get("object") != "response":
            raise ApiError(
                404,
                f"{response_id!r} not found — evicted, deleted, or sent with store=false",
                code="not_found",
            )
        if not stream:
            return JSONResponse({k: v for k, v in env.items() if not k.startswith("_fx1_")})
        headers: dict[str, str] = {}
        cid = env.get("_fx1_completion_id")
        if isinstance(cid, str) and cid:
            headers["X-Fx1-Completion-Id"] = cid
            _rsha = _completion_receipt_sha(cid)
            if _rsha is not None:
                headers["X-Fx1-Receipt-Sha256"] = _rsha
        skip = (starting_after + 1) if starting_after is not None else 0
        return StreamingResponse(
            _responses_replay_frames(
                envelope_store,
                response_id,
                skip=skip,
                timeout_s=timeout_s,
                keepalive_s=float(getattr(app.state, "sse_keepalive_s", 15.0)),
            ),
            media_type="text/event-stream",
            headers=headers,
        )

    @app.delete(
        "/v1/responses/{response_id}",
        response_model=None,
        tags=["openai"],
        operation_id="openai_responses_delete",
    )
    def openai_response_delete(response_id: str) -> dict[str, Any]:
        """Drop a stored response object from the retrieval index."""
        return _drop_envelope(response_id, object_="response")

    @app.post(
        "/v1/responses/{response_id}/cancel",
        response_model=None,
        tags=["openai"],
        operation_id="openai_responses_cancel",
    )
    def openai_response_cancel(response_id: str) -> dict[str, Any]:
        """Cancel a queued or in-progress background response (OpenAI's
        ``responses.cancel``). The stored envelope flips to ``cancelled``
        and a running worker is told to discard its result — the model
        call still lands in the completion log when it had already
        started. Terminal responses refuse 409; unknown ids 404."""
        env = _stored_envelope(response_id, object_="response")
        if env["status"] in OPENAI_RESPONSE_TERMINAL:
            raise ApiError(
                409,
                f"{response_id!r} is already {env['status']} — only queued or "
                "in_progress responses cancel",
                code="cancel_terminal",
            )
        # Commit before signalling: a terminal completion that already
        # won cannot be undone by a late cancellation request. The atomic
        # status update preserves internal completion/evidence headers.
        if not envelope_store.transition_status(
            response_id, expect={"queued", "in_progress"}, status="cancelled"
        ):
            current = _stored_envelope(response_id, object_="response")
            raise ApiError(
                409,
                f"{response_id!r} is already {current['status']} — only queued or "
                "in_progress responses cancel",
                code="cancel_terminal",
            )
        ev = bg_cancel.get(response_id)
        if ev is not None:
            ev.set()
        env["status"] = "cancelled"
        return env

    # --- /v1/conversations ---------------------------------------------------
    # The named-container twin of ``previous_response_id``: a conv id a
    # turn joins via ``conversation``, whose accumulated items (each turn's
    # input + output) become the next turn's context. Items page through
    # the same cursor contract as the stored-request subresources; the
    # conv and its items live in one bounded LRU keyed on ``conv_*`` ids.

    def _stored_conversation(conversation_id: str) -> dict[str, Any]:
        conv = conv_store.get(conversation_id)
        if conv is None or conv.get("object") != "conversation":
            raise ApiError(
                404,
                f"{conversation_id!r} not found — no conversation under this id",
                code="not_found",
            )
        return conv

    @app.post(
        "/v1/conversations",
        response_model=None,
        tags=["openai"],
        operation_id="openai_conversation_create",
    )
    def openai_conversation_create(body: OpenAIConversationCreate) -> dict[str, Any]:
        """Create a conversation container (``conv_…``). ``items`` seeds
        the item list with message items; a response joins it with
        ``conversation`` and appends its turn when it completes."""
        _drain_refusal(metrics)
        cid = f"conv_{uuid.uuid4().hex}"
        env = openai_conversation_object(cid=cid, metadata=body.metadata)
        conv_store.put(
            env,
            items={
                "items": (response_input_items_for_store(body.items, rid=cid) if body.items else [])
            },
        )
        return env

    @app.get(
        "/v1/conversations/{conversation_id}",
        response_model=None,
        tags=["openai"],
        operation_id="openai_conversation_retrieve",
    )
    def openai_conversation_retrieve(conversation_id: str) -> dict[str, Any]:
        """Retrieve a conversation object."""
        return _stored_conversation(conversation_id)

    @app.post(
        "/v1/conversations/{conversation_id}",
        response_model=None,
        tags=["openai"],
        operation_id="openai_conversation_update",
    )
    def openai_conversation_update(
        body: OpenAIConversationUpdate, conversation_id: str
    ) -> dict[str, Any]:
        """Update a conversation — ``metadata`` replaces wholesale."""
        _drain_refusal(metrics)
        conv = _stored_conversation(conversation_id)
        new_conv = dict(conv)
        new_conv["metadata"] = dict(body.metadata) if body.metadata is not None else {}
        # put-if-present: a delete racing between fetch and write must
        # win — an unconditional re-put would resurrect the tombstone.
        if not conv_store.put_if_present(new_conv):
            raise ApiError(
                404,
                f"{conversation_id!r} not found — no conversation under this id",
                code="not_found",
            )
        return new_conv

    @app.delete(
        "/v1/conversations/{conversation_id}",
        response_model=None,
        tags=["openai"],
        operation_id="openai_conversation_delete",
    )
    def openai_conversation_delete(conversation_id: str) -> dict[str, Any]:
        """Delete a conversation — its items drop with it; responses that
        joined it stay in the retrieval index on their own ids."""
        _stored_conversation(conversation_id)
        conv_store.delete(conversation_id)
        return {"id": conversation_id, "object": "conversation.deleted", "deleted": True}

    @app.get(
        "/v1/conversations/{conversation_id}/items",
        response_model=None,
        tags=["openai"],
        operation_id="openai_conversation_items_list",
    )
    def openai_conversation_items(
        conversation_id: str,
        limit: int = Query(default=20, ge=1, le=100),
        after: str | None = Query(default=None),
        before: str | None = Query(default=None),
        order: Literal["asc", "desc"] = Query(default="asc"),
    ) -> dict[str, Any]:
        """A conversation's items, oldest first — the same ``after``/
        ``before``/``order`` cursor contract as the other item lists."""
        _stored_conversation(conversation_id)
        items = conv_store.get_items(conversation_id, "items")
        try:
            return paged_item_list(
                items or [], limit=limit, after=after, before=before, order=order
            )
        except OpenAICompatError as exc:
            raise ApiError(exc.status, str(exc), code=exc.code) from exc

    @app.post(
        "/v1/conversations/{conversation_id}/items",
        response_model=None,
        tags=["openai"],
        operation_id="openai_conversation_items_add",
    )
    def openai_conversation_items_add(
        body: OpenAIConversationItemsAdd, conversation_id: str
    ) -> dict[str, Any]:
        """Append items to a conversation — returns the minted items as a
        list object. ``item_ids`` (alias-by-reference) is refused: items
        are minted per append, never aliased."""
        _drain_refusal(metrics)
        _stored_conversation(conversation_id)
        minted: list[dict[str, Any]] = []

        def _extend(current: list[dict[str, Any]]) -> list[dict[str, Any]]:
            # A fresh append namespace avoids reusing ids after deletion;
            # the list merge remains atomic under the store lock.
            minted.extend(
                response_input_items_for_store(
                    body.items or [], rid=f"{conversation_id}:{uuid.uuid4().hex}"
                )
            )
            return [*current, *minted]

        merged = conv_store.mutate_items(conversation_id, "items", _extend)
        if merged is None:
            raise ApiError(
                404,
                f"{conversation_id!r} not found — no conversation under this id",
                code="not_found",
            )
        return {
            "object": "list",
            "data": minted,
            "first_id": minted[0].get("id") if minted else None,
            "last_id": minted[-1].get("id") if minted else None,
            "has_more": False,
        }

    @app.get(
        "/v1/conversations/{conversation_id}/items/{item_id}",
        response_model=None,
        tags=["openai"],
        operation_id="openai_conversation_item_retrieve",
    )
    def openai_conversation_item_retrieve(conversation_id: str, item_id: str) -> dict[str, Any]:
        """One item by id — the same store the list/delete routes read.
        A missing item (or wrong conversation) is a 404, never a lookup
        into another conversation's namespace."""
        _stored_conversation(conversation_id)
        items = conv_store.get_items(conversation_id, "items") or []
        for it in items:
            if it.get("id") == item_id:
                return it
        raise ApiError(404, f"item {item_id!r} not found in {conversation_id!r}", code="not_found")

    @app.delete(
        "/v1/conversations/{conversation_id}/items/{item_id}",
        response_model=None,
        tags=["openai"],
        operation_id="openai_conversation_item_delete",
    )
    def openai_conversation_item_delete(conversation_id: str, item_id: str) -> dict[str, Any]:
        """Delete one item from a conversation — the conv object returns;
        a missing item id is a 404."""
        conv = _stored_conversation(conversation_id)

        def _drop(current: list[dict[str, Any]]) -> list[dict[str, Any]]:
            kept = [it for it in current if it.get("id") != item_id]
            if len(kept) == len(current):
                raise ApiError(
                    404, f"item {item_id!r} not found in {conversation_id!r}", code="not_found"
                )
            return kept

        if conv_store.mutate_items(conversation_id, "items", _drop) is None:
            raise ApiError(
                404,
                f"{conversation_id!r} not found — no conversation under this id",
                code="not_found",
            )
        return conv

    # --- /v1/vector_stores ---------------------------------------------------
    # The OpenAI vector-store surface — a journaled lexical retrieval
    # corpus over /v1/files. Searching a store is not a wire route; it
    # runs inside ``file_search`` on /v1/responses.

    def _vs_err(exc: VectorStoreError) -> ApiError:
        return ApiError(exc.status, str(exc), code=exc.code)

    def _vs_idem_lookup(
        idempotency_key: str | None,
        body_fp: str,
        *,
        namespace: str,
    ) -> tuple[str | None, dict[str, Any] | None]:
        """Lookup the replay record co-journaled with a vector mutation."""
        key = _idem_key(idempotency_key)
        if key is None:
            return None, None
        scoped = _idem_scope(key, namespace=namespace)
        assert scoped is not None  # noqa: S101 — key is not None here
        hit = vs_store.idempotency_get(scoped)
        if hit is None:
            return scoped, None
        fingerprint, response = hit
        if fingerprint != body_fp:
            raise ApiError(
                409,
                "Idempotency-Key reuse with a different request body",
                code="idempotency_conflict",
            )
        return scoped, response

    @app.post(
        "/v1/vector_stores",
        response_model=None,
        tags=["openai"],
        operation_id="openai_vectorstore_create",
    )
    def openai_vectorstore_create(
        body: OpenAIVectorStoreCreate,
        _idem_claim_held: None = Depends(_vs_idem_claim),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any] | JSONResponse:
        """Create a vector store — ``file_ids`` attach existing
        ``file-*`` records; an unresolvable id fails the whole create
        fail-closed (no partial store). ``Idempotency-Key`` pins the
        mint: a keyed retry replays the recorded object (same ``vs_``)
        instead of minting a duplicate store."""
        body_fp = _body_fp(body)
        key, replay = _vs_idem_lookup(idempotency_key, body_fp, namespace="vs")
        if replay is not None:
            return JSONResponse(replay, headers={"X-Fx1-Idempotent-Replay": "true"})
        _drain_refusal(metrics)
        try:
            out = vs_store.create(
                name=body.name,
                metadata=body.metadata,
                file_ids=tuple(body.file_ids or ()),
                expires_after=body.expires_after,
                idempotency_key=key,
                body_fingerprint=body_fp,
            )
        except VectorStoreError as exc:
            raise _vs_err(exc) from exc
        return out

    @app.get(
        "/v1/vector_stores/{vector_store_id}",
        response_model=None,
        tags=["openai"],
        operation_id="openai_vectorstore_retrieve",
    )
    def openai_vectorstore_retrieve(vector_store_id: str) -> dict[str, Any]:
        """Retrieve a vector store by id — ``vs_*``."""
        try:
            return vs_store.get(vector_store_id)
        except VectorStoreError as exc:
            raise _vs_err(exc) from exc

    @app.post(
        "/v1/vector_stores/{vector_store_id}",
        response_model=None,
        tags=["openai"],
        operation_id="openai_vectorstore_update",
    )
    def openai_vectorstore_update(
        vector_store_id: str, body: OpenAIVectorStoreUpdate
    ) -> dict[str, Any]:
        """Update a vector store — ``name``/``metadata`` replace
        wholesale when present. A mutation, so drain-gated like the
        other writes; deletes/detaches stay open under drain."""
        _drain_refusal(metrics)
        try:
            return vs_store.update(
                vector_store_id,
                name=body.name,
                metadata=body.metadata,
                expires_after=body.expires_after,
            )
        except VectorStoreError as exc:
            raise _vs_err(exc) from exc

    @app.delete(
        "/v1/vector_stores/{vector_store_id}",
        response_model=None,
        tags=["openai"],
        operation_id="openai_vectorstore_delete",
    )
    def openai_vectorstore_delete(vector_store_id: str) -> dict[str, Any]:
        """Delete a vector store — member files' index entries and
        chunks drop with it; the ``file-*`` records survive (the store
        borrows, never owns)."""
        try:
            return vs_store.delete(vector_store_id)
        except VectorStoreError as exc:
            raise _vs_err(exc) from exc

    @app.get(
        "/v1/vector_stores",
        response_model=None,
        tags=["openai"],
        operation_id="openai_vectorstore_list",
    )
    def openai_vectorstore_list(
        limit: int = Query(default=20, ge=1, le=100),
        after: str | None = Query(default=None),
        before: str | None = Query(default=None),
        order: str = Query(default="desc"),
    ) -> dict[str, Any]:
        """List vector stores — the same ``after``/``before``/``order``
        cursor contract as the other list surfaces."""
        try:
            return vs_store.list_stores(limit=limit, order=order, after=after, before=before)
        except VectorStoreError as exc:
            raise _vs_err(exc) from exc

    @app.post(
        "/v1/vector_stores/{vector_store_id}/files",
        response_model=None,
        tags=["openai"],
        operation_id="openai_vectorstore_file_create",
    )
    def openai_vectorstore_file_create(
        vector_store_id: str,
        body: OpenAIVectorStoreFileCreate,
        _idem_claim_held: None = Depends(_vs_idem_claim),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any] | JSONResponse:
        """Attach a ``file-*`` record — the file is decoded, chunked,
        and indexed in-place; a file whose text is empty lands
        ``status: failed`` with ``last_error``, never silently.
        ``Idempotency-Key`` pins the attachment (per-store namespace):
        a keyed retry replays the recorded object instead of answering
        ``file_already_attached``."""
        body_fp = _body_fp(body)
        key, replay = _vs_idem_lookup(
            idempotency_key,
            body_fp,
            namespace=f"vsfile:{vector_store_id}",
        )
        if replay is not None:
            return JSONResponse(replay, headers={"X-Fx1-Idempotent-Replay": "true"})
        _drain_refusal(metrics)
        try:
            out = vs_store.attach(
                vector_store_id,
                body.file_id,
                attributes=body.attributes,
                chunking_strategy=body.chunking_strategy,
                idempotency_key=key,
                body_fingerprint=body_fp,
            )
        except VectorStoreError as exc:
            raise _vs_err(exc) from exc
        return out

    @app.get(
        "/v1/vector_stores/{vector_store_id}/files",
        response_model=None,
        tags=["openai"],
        operation_id="openai_vectorstore_file_list",
    )
    def openai_vectorstore_file_list(
        vector_store_id: str,
        limit: int = Query(default=20, ge=1, le=100),
        after: str | None = Query(default=None),
        before: str | None = Query(default=None),
        order: str = Query(default="asc"),
        filter: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """List a store's attached files — ``filter`` takes an OpenAI
        status word (``in_progress|completed|cancelled|failed``)."""
        try:
            return vs_store.list_files(
                vector_store_id,
                limit=limit,
                order=order,
                after=after,
                before=before,
                filter=filter,
            )
        except VectorStoreError as exc:
            raise _vs_err(exc) from exc

    @app.get(
        "/v1/vector_stores/{vector_store_id}/files/{file_id}",
        response_model=None,
        tags=["openai"],
        operation_id="openai_vectorstore_file_retrieve",
    )
    def openai_vectorstore_file_retrieve(vector_store_id: str, file_id: str) -> dict[str, Any]:
        """Retrieve one attachment — status, chunk count, attributes."""
        try:
            return vs_store.get_file(vector_store_id, file_id)
        except VectorStoreError as exc:
            raise _vs_err(exc) from exc

    @app.delete(
        "/v1/vector_stores/{vector_store_id}/files/{file_id}",
        response_model=None,
        tags=["openai"],
        operation_id="openai_vectorstore_file_delete",
    )
    def openai_vectorstore_file_delete(vector_store_id: str, file_id: str) -> dict[str, Any]:
        """Detach a file — its chunks leave the index; the underlying
        ``file-*`` record survives."""
        try:
            return vs_store.detach(vector_store_id, file_id)
        except VectorStoreError as exc:
            raise _vs_err(exc) from exc

    @app.get(
        "/v1/vector_stores/{vector_store_id}/files/{file_id}/content",
        response_model=None,
        tags=["openai"],
        operation_id="openai_vectorstore_file_content",
    )
    def openai_vectorstore_file_content(vector_store_id: str, file_id: str) -> dict[str, Any]:
        """The stored text of one attachment — a
        ``vector_store.file_content.page`` list of ``{type: 'text'}``
        parts (the index holds the decoded text, not raw bytes)."""
        try:
            return vs_store.file_content(vector_store_id, file_id)
        except VectorStoreError as exc:
            raise _vs_err(exc) from exc

    @app.post(
        "/v1/vector_stores/{vector_store_id}/search",
        response_model=None,
        tags=["openai"],
        operation_id="openai_vectorstore_search",
    )
    def openai_vectorstore_search(
        vector_store_id: str, body: OpenAIVectorStoreSearch
    ) -> dict[str, Any]:
        """Direct store search — OpenAI's ``vector_stores.search``: the
        ranked hits without spending a response turn. ``query`` may be a
        string or list of strings (joined); ``filters`` evaluate against
        file attributes; ``ranking_options.score_threshold`` bounds the
        cosine floor."""
        query = body.query if isinstance(body.query, str) else " ".join(str(q) for q in body.query)
        ro = body.ranking_options or {}
        try:
            hits = vs_store.search(
                [vector_store_id],
                query,
                max_results=body.max_num_results or 10,
                filters=body.filters,
                score_threshold=ro.get("score_threshold"),
            )
        except VectorStoreError as exc:
            raise _vs_err(exc) from exc
        return {
            "object": "vector_store.search_results.page",
            "search_query": query,
            "data": [
                {
                    "file_id": h["file_id"],
                    "filename": h["filename"],
                    "score": h["score"],
                    "attributes": h["attributes"],
                    "content": [{"type": "text", "text": h["text"]}],
                }
                for h in hits
            ],
            "has_more": False,
            "next_page": None,
        }

    @app.post(
        "/v1/vector_stores/{vector_store_id}/file_batches",
        response_model=None,
        tags=["openai"],
        operation_id="openai_vectorstore_file_batch_create",
    )
    def openai_vectorstore_file_batch_create(
        vector_store_id: str,
        body: OpenAIVectorStoreFileBatchCreate,
        _idem_claim_held: None = Depends(_vs_idem_claim),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any] | JSONResponse:
        """Attach many ``file-*`` records in one call — the OpenAI
        ``vector_store.files_batch`` surface. Members attach
        synchronously; per-file refusals (missing, already attached,
        oversized, store full) count ``failed`` with ``last_error``,
        never abort the batch. Status is terminal at return.
        ``Idempotency-Key`` pins the minted batch (per-store
        namespace): a keyed retry replays the recorded verdicts
        instead of turning every member ``file_already_attached``."""
        body_fp = _body_fp(body)
        key, replay = _vs_idem_lookup(
            idempotency_key,
            body_fp,
            namespace=f"vsbatch:{vector_store_id}",
        )
        if replay is not None:
            return JSONResponse(replay, headers={"X-Fx1-Idempotent-Replay": "true"})
        _drain_refusal(metrics)
        try:
            out = vs_store.file_batch_create(
                vector_store_id,
                body.file_ids,
                attributes=body.attributes,
                chunking_strategy=body.chunking_strategy,
                idempotency_key=key,
                body_fingerprint=body_fp,
            )
        except VectorStoreError as exc:
            raise _vs_err(exc) from exc
        return out

    @app.get(
        "/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}",
        response_model=None,
        tags=["openai"],
        operation_id="openai_vectorstore_file_batch_retrieve",
    )
    def openai_vectorstore_file_batch_retrieve(
        vector_store_id: str, batch_id: str
    ) -> dict[str, Any]:
        """Retrieve one ``vsfb_*`` batch — standing status + counts."""
        try:
            return vs_store.file_batch_get(vector_store_id, batch_id)
        except VectorStoreError as exc:
            raise _vs_err(exc) from exc

    @app.post(
        "/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel",
        response_model=None,
        tags=["openai"],
        operation_id="openai_vectorstore_file_batch_cancel",
    )
    def openai_vectorstore_file_batch_cancel(vector_store_id: str, batch_id: str) -> dict[str, Any]:
        """Cancel a batch — members attach synchronously at create so a
        batch is always terminal; the 409 reports the standing status
        rather than faking a mid-flight window."""
        try:
            return vs_store.file_batch_cancel(vector_store_id, batch_id)
        except VectorStoreError as exc:
            raise _vs_err(exc) from exc

    @app.get(
        "/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/files",
        response_model=None,
        tags=["openai"],
        operation_id="openai_vectorstore_file_batch_files",
    )
    def openai_vectorstore_file_batch_files(
        vector_store_id: str,
        batch_id: str,
        limit: int = Query(default=20, ge=1, le=100),
        after: str | None = Query(default=None),
        before: str | None = Query(default=None),
        order: str = Query(default="asc"),
        filter: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """List a batch's per-file verdicts — frozen at processing time,
        ``filter`` takes an OpenAI status word."""
        try:
            return vs_store.file_batch_files(
                vector_store_id,
                batch_id,
                limit=limit,
                order=order,
                after=after,
                before=before,
                filter=filter,
            )
        except VectorStoreError as exc:
            raise _vs_err(exc) from exc

    def _request_items(
        envelope_id: str,
        *,
        object_: str,
        key: str,
        limit: int,
        after: str | None,
        before: str | None,
        order: Literal["asc", "desc"],
    ) -> dict[str, Any]:
        env = envelope_store.get(envelope_id)
        if env is None or env.get("object") != object_:
            raise ApiError(404, f"{envelope_id!r} not found", code="not_found")
        items = envelope_store.get_items(envelope_id, key)
        if items is None:
            raise ApiError(404, f"{envelope_id!r} not found", code="not_found")
        try:
            return paged_item_list(items, limit=limit, after=after, before=before, order=order)
        except OpenAICompatError as exc:
            raise ApiError(exc.status, str(exc), code=exc.code) from exc

    @app.get(
        "/v1/chat/completions",
        response_model=None,
        tags=["openai"],
        operation_id="openai_chat_list",
    )
    def openai_chat_list(
        request: Request,
        limit: int = Query(default=20, ge=1, le=100),
        after: str | None = Query(default=None),
        before: str | None = Query(default=None),
        order: Literal["asc", "desc"] = Query(default="asc"),
        model: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """Stored chat completions, oldest first — OpenAI's
        ``chat.completions.list``. ``metadata[key]=value`` query pairs
        filter to envelopes carrying that exact subset."""
        meta_filter = {
            k[9:-1]: v
            for k, v in request.query_params.multi_items()
            if k.startswith("metadata[") and k.endswith("]") and len(k) > 10
        }
        envs = envelope_store.list_envelopes(_OBJ_CHAT_COMPLETION)
        if model is not None:
            envs = [e for e in envs if e.get("model") == model]
        if meta_filter:
            envs = [
                e
                for e in envs
                if isinstance(e.get("metadata"), dict)
                and all(e["metadata"].get(k) == v for k, v in meta_filter.items())
            ]
        try:
            return paged_item_list(envs, limit=limit, after=after, before=before, order=order)
        except OpenAICompatError as exc:
            raise ApiError(exc.status, str(exc), code=exc.code) from exc

    @app.get(
        "/v1/chat/completions/{completion_id}/messages",
        response_model=None,
        tags=["openai"],
        operation_id="openai_chat_messages",
    )
    def openai_chat_messages(
        completion_id: str,
        limit: int = Query(default=20, ge=1, le=100),
        after: str | None = Query(default=None),
        before: str | None = Query(default=None),
        order: Literal["asc", "desc"] = Query(default="asc"),
    ) -> dict[str, Any]:
        """The request messages a stored chat completion ran on
        (OpenAI's ``chat.completions.messages.list``)."""
        return _request_items(
            completion_id,
            object_=_OBJ_CHAT_COMPLETION,
            key="messages",
            limit=limit,
            after=after,
            before=before,
            order=order,
        )

    @app.get(
        "/v1/responses/{response_id}/input_items",
        response_model=None,
        tags=["openai"],
        operation_id="openai_responses_input_items",
    )
    def openai_response_input_items(
        response_id: str,
        limit: int = Query(default=20, ge=1, le=100),
        after: str | None = Query(default=None),
        before: str | None = Query(default=None),
        order: Literal["asc", "desc"] = Query(default="asc"),
    ) -> dict[str, Any]:
        """The ``input`` items a stored response ran on (OpenAI's
        ``responses.input_items.list``)."""
        return _request_items(
            response_id,
            object_="response",
            key="input_items",
            limit=limit,
            after=after,
            before=before,
            order=order,
        )

    @app.post(
        "/v1/embeddings",
        response_model=OpenAIEmbeddingResponse,
        tags=["openai"],
        operation_id="openai_create_embedding",
    )
    def openai_create_embedding(
        body: OpenAIEmbeddingRequest,
        request: Request,
        response: Response,
        _slot_held: None = Depends(slot),
    ) -> dict[str, Any]:
        """OpenAI's embeddings.create — vectors for retrieval/eval lanes.

        ``model`` forwards verbatim (embedding models name themselves on
        the provider); the link chain is chat's — ``fx1.backend`` >
        ``X-Fx1-Backend`` > a ``model`` naming a backend > ``hosted_k3``,
        BYOK via ``fx1.byok`` or the ``X-Fx1-Byok-*`` headers. A link
        without the embeddings channel answers 501 — never fabricated
        vectors. ``encoding_format``/``dimensions``/``user`` pass through;
        the provider's ``data[]``/``model``/``usage`` echo verbatim (null
        usage under provider silence). The call lands in the completion
        log — ``X-Fx1-Completion-Id`` links it.
        """
        try:
            env, cid = _openai_embeddings_core(body, request.headers)
        except OpenAICompatError as exc:
            raise ApiError(exc.status, str(exc), code=exc.code) from exc
        response.headers["X-Fx1-Completion-Id"] = cid
        _rsha = _completion_receipt_sha(cid)
        if _rsha is not None:
            response.headers["X-Fx1-Receipt-Sha256"] = _rsha
        return env

    # --- /v1/files + /v1/batches ------------------------------------------
    # The async-batch surface: files carry request JSONL (multipart upload,
    # purpose="batch"), a batch runs its lines through the SAME gated route
    # cores above (literal parity, not a second pipeline), and the output
    # file holds one OpenAI batch-result line per input line — per-line
    # request errors land as status_code-carrying output lines, never as a
    # failed batch.

    def _run_batch_line(line: dict[str, Any], batch: _BatchRecord) -> dict[str, Any]:
        """One batch line through the endpoint's own validation + core —
        the same verdicts the live route returns, packed into the output
        line shape. Per-line faults never abort the batch."""
        rid = uuid.uuid4().hex
        custom_id = str(line["custom_id"])
        try:
            obj = batch_line_body(line, batch.endpoint)
            if getattr(obj, "stream", False):
                raise OpenAICompatError(
                    "stream requests are not valid inside a batch", code="invalid_request"
                )
            if getattr(obj, "background", False):
                raise OpenAICompatError(
                    "background requests are not valid inside a batch — the "
                    "batch itself is the async surface",
                    code="invalid_request",
                )
            if getattr(obj, "conversation", None) is not None:
                raise OpenAICompatError(
                    "conversation requests are not valid inside a batch — "
                    "a shared conv container would race across lines",
                    code="invalid_request",
                )
            if isinstance(obj, OpenAIChatRequest):
                env, _cid = _openai_chat_core(obj, batch._headers)
            elif isinstance(obj, OpenAIEmbeddingRequest):
                env, _cid = _openai_embeddings_core(obj, batch._headers)
            else:
                env, _cid, _usage = _openai_response_core(obj, batch._headers)
            return batch_output_line(custom_id=custom_id, status_code=200, body=env, rid=rid)
        except OpenAICompatError as exc:
            return batch_output_line(
                custom_id=custom_id,
                status_code=exc.status,
                body=openai_error_body(str(exc), exc.status, exc.code),
                rid=rid,
            )
        except ApiError as exc:
            return batch_output_line(
                custom_id=custom_id,
                status_code=exc.status_code,
                body=openai_error_body(str(exc.detail), exc.status_code, _err_code(exc)),
                rid=rid,
            )
        except Exception as exc:  # noqa: BLE001 — a line fault is data, not a crash
            return batch_output_line(
                custom_id=custom_id,
                status_code=500,
                body=openai_error_body(f"{type(exc).__name__}: {exc}", 500, "server_error"),
                rid=rid,
            )

    def _exec_batch(batch: _BatchRecord) -> None:
        """Worker: validate → in_progress → per-line through the gated cores
        → finalizing → write the output file → terminal status. Holds ONE
        inflight slot for the whole batch (the jobs-channel contract).

        Every status write respects an already-terminal record: expiry-on-
        read can flip the batch while it queues or runs, and a terminal
        state is never overwritten. The submitter's ``key_id`` is re-
        installed per line — worker threads do not inherit request
        contextvars, and every line must still attribute + meter under
        the credential that submitted the batch."""
        ctx_key: contextvars.Token[str | None] | None = None
        out_lines: builtins.list[str] = []
        try:
            with batch._state_lock:
                if batch.status in _BATCH_TERMINAL:
                    return  # expiry-on-read already terminalized it while queued
                if batch._cancel.is_set():
                    # Cancellation won before the worker started.  Preserve
                    # the monotone lifecycle instead of regressing
                    # cancelling -> in_progress for an empty batch.
                    batch.status = "cancelled"
                    batch.cancelled_at = int(time.time())
                    return
                ctx_key = _REQUEST_KEY_ID.set(batch._key_id)
                batch.status = "in_progress"
                batch.in_progress_at = int(time.time())
            batch_store.mark(batch)
            cancelled = False
            for line in batch._lines:
                if batch._cancel.is_set():
                    cancelled = True
                    break
                out = _run_batch_line(line, batch)
                with batch._state_lock:
                    if batch.status in _BATCH_TERMINAL:
                        # Expiry won while the provider call was in flight.
                        # Terminal records are immutable: usage remains in
                        # the completion ledger, but no late result is
                        # attached to an already-delivered terminal payload.
                        return
                    if out["response"]["status_code"] == 200:
                        batch.request_counts.completed += 1
                    else:
                        batch.request_counts.failed += 1
                    out_lines.append(json.dumps(out, sort_keys=True, separators=(",", ":")))
            with batch._state_lock:
                if batch.status in _BATCH_TERMINAL:
                    return
                batch.status = "finalizing"
                batch.finalizing_at = int(time.time())
            batch_store.mark(batch)
            rec: _FileRecord | None = None
            if out_lines:
                rec = file_store.put(
                    filename=f"{batch.batch_id}_output.jsonl",
                    purpose="batch_output",
                    content=("\n".join(out_lines) + "\n").encode(),
                )
            with batch._state_lock:
                if batch.status in _BATCH_TERMINAL:
                    terminal_won = True
                else:
                    terminal_won = False
                    if rec is not None:
                        batch.output_file_id = rec.file_id
                    if cancelled or batch._cancel.is_set():
                        batch.status = "cancelled"
                        batch.cancelled_at = int(time.time())
                    else:
                        batch.status = "completed"
                        batch.completed_at = int(time.time())
            if terminal_won and rec is not None:
                # Expiry won while the output blob was being published.
                # Roll back the unreferenced file; terminal state and webhook
                # payload remain immutable.
                file_store.delete(rec.file_id)
        except Exception as exc:  # noqa: BLE001 — worker fault fails the batch, not the process
            with batch._state_lock:
                can_fail = batch.status not in _BATCH_TERMINAL
            err_rec: _FileRecord | None = None
            if can_fail:
                with suppress(Exception):
                    err_rec = file_store.put(
                        filename=f"{batch.batch_id}_errors.jsonl",
                        purpose="batch_output",
                        content=(json.dumps(out_lines) + "\n").encode() if out_lines else b"\n",
                    )
                with batch._state_lock:
                    if batch.status not in _BATCH_TERMINAL:
                        batch.status = "failed"
                        batch.failed_at = int(time.time())
                        batch.errors = {
                            "object": "list",
                            "data": [
                                {
                                    "code": "internal_error",
                                    "message": f"{type(exc).__name__}: {exc}",
                                }
                            ],
                        }
                        if err_rec is not None:
                            batch.error_file_id = err_rec.file_id
                        err_rec = None
                if err_rec is not None:
                    file_store.delete(err_rec.file_id)
        finally:
            if ctx_key is not None:
                _REQUEST_KEY_ID.reset(ctx_key)
            metrics.release()
            inflight.release()
            _batch_webhook(batch)
            batch_store.mark(batch)

    def _batch_webhook(batch: _BatchRecord) -> None:
        """Fire-once terminal webhook: the projected OpenAI envelope is the
        payload (the same shape GET returns — never the internal record).
        The first terminal transition fires; expiry-on-read is one."""
        with batch._state_lock:
            if not batch.callback_url or batch.status not in _BATCH_TERMINAL:
                return
            payload = json.dumps(batch_object(batch.model_dump(mode="json"))).encode()
        _deliver_callback(
            batch,
            body=payload,
        )

    def _batch_project(batch: _BatchRecord) -> dict[str, Any]:
        """Expiry check + envelope projection.

        Expiry sets ``_cancel`` (mirroring the Anthropic dialect) so a
        queued worker returns early and a mid-flight worker stops after
        its current line.  A result that returns after expiry is still
        metered in the completion ledger but is not attached to the already
        terminal batch or its fire-once webhook payload."""
        expired_now = False
        with batch._state_lock:
            if batch.status not in _BATCH_TERMINAL and time.time() > batch.expires_at:
                batch._cancel.set()  # the worker exits its loop at the next line
                batch.status = "expired"
                batch.expired_at = int(time.time())
                expired_now = True
        if expired_now:
            batch_store.mark(batch)
            _batch_webhook(batch)  # expiry is a terminal transition too
            batch_store.mark(batch)  # persist callback outcome as well
        with batch._state_lock:
            return batch_object(batch.model_dump())

    @app.post(
        "/v1/files",
        tags=["openai"],
        operation_id="openai_file_upload",
    )
    async def openai_file_upload(
        file: UploadFile | None = File(default=None),
        purpose: str = Form(default=""),
        _idem_claim_held: None = Depends(_file_idem_claim),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> JSONResponse:
        """Upload a batch-input or fine-tuning JSONL (multipart/form-data).
        Purpose is fail-closed — ``batch`` and ``fine-tune`` are the only
        purposes served; the file is validated into the store as-is
        (shape checks happen at batch submit / fine-tune submit).
        ``Idempotency-Key`` pins the minted file: a retried multipart
        upload replays the recorded object (same ``id``) instead of
        storing a duplicate."""
        if file is None:
            raise ApiError(400, "multipart field 'file' is required", code="invalid_request")
        if purpose not in OPENAI_FILE_PURPOSE_ACCEPT:
            raise ApiError(
                400,
                f"unsupported purpose {purpose!r} — only "
                f"{sorted(OPENAI_FILE_PURPOSE_ACCEPT)} are served",
                code="invalid_request",
            )
        data = await file.read()
        if not data:
            raise ApiError(400, "file is empty", code="invalid_request")
        if len(data) > file_bytes_max:
            raise ApiError(
                413,
                f"file exceeds the {file_bytes_max}-byte cap",
                code="file_too_large",
            )
        filename = file.filename or "upload.jsonl"
        if not filename.endswith(".jsonl"):
            raise ApiError(
                400,
                f"input must be a .jsonl file, got {filename!r}",
                code="invalid_request",
            )
        # Content fingerprint: same key + same purpose+name+bytes replays;
        # same key + different bytes conflicts.
        body_fp = hashlib.sha256(purpose.encode() + b"\0" + filename.encode() + b"\0" + data)
        key, replay = _idem_lookup(
            idempotency_key, upload_idem_store, body_fp.hexdigest(), namespace="file"
        )
        if replay is not None:
            return JSONResponse(replay.envelope, headers={"X-Fx1-Idempotent-Replay": "true"})
        _drain_refusal(metrics)
        rec = file_store.put(filename=filename, purpose=purpose, content=data)
        out = file_object(rec.model_dump())
        if key is not None:
            try:
                upload_idem_store.put(key, body_fp.hexdigest(), _JsonIdemRecord(envelope=out))
            except Exception:
                # The file store committed before its minted id was known.
                # If replay-record publication fails, remove that exact new
                # file so a client retry cannot create a second durable copy.
                # This is an in-process rollback; a crash between the two
                # journals still needs a unified transaction to close.
                with suppress(Exception):
                    file_store.delete(rec.file_id)
                raise
        return JSONResponse(out)

    @app.get(
        "/v1/files",
        tags=["openai"],
        operation_id="openai_file_list",
    )
    def openai_file_list(
        limit: int = Query(default=20, ge=1, le=100),
        after: str | None = Query(default=None),
        before: str | None = Query(default=None),
        order: str = Query(default="desc"),
        purpose: str | None = Query(default=None),
    ) -> JSONResponse:
        """Newest-first file listing — the shared cursor page shape
        (``has_more`` + ``first_id``/``last_id``) so stock-SDK
        auto-pagination terminates; ``purpose`` filters by the upload's
        declared intent."""
        try:
            if order not in ("asc", "desc"):
                raise OpenAICompatError(
                    f"order must be 'asc' or 'desc', got {order!r}",
                    status=400,
                    code="invalid_cursor",
                )
            records = file_store.list()
            if purpose is not None:
                records = [r for r in records if r.purpose == purpose]
            items = [file_object(r.model_dump()) for r in records]
            # the store is newest-first — that IS desc; "asc" flips to
            # oldest-first before the shared pager walks it
            if order == "asc":
                items.reverse()
            return JSONResponse(
                paged_item_list(items, limit=limit, after=after, before=before, order="asc")
            )
        except OpenAICompatError as exc:
            raise ApiError(exc.status, str(exc), code=exc.code) from exc

    @app.get(
        "/v1/files/{file_id}",
        tags=["openai"],
        operation_id="openai_file_get",
    )
    def openai_file_get(file_id: str) -> JSONResponse:
        rec = file_store.get(file_id)
        if rec is None:
            raise ApiError(404, f"file {file_id!r} not found", code="file_not_found")
        return JSONResponse(file_object(rec.model_dump()))

    @app.get(
        "/v1/files/{file_id}/content",
        tags=["openai"],
        operation_id="openai_file_content",
    )
    def openai_file_content(file_id: str) -> Response:
        """Raw bytes — JSONL in, JSONL out (batch results land here too)."""
        rec = file_store.get(file_id)
        if rec is None:
            raise ApiError(404, f"file {file_id!r} not found", code="file_not_found")
        return Response(
            content=rec.content,
            media_type="application/jsonl",
            headers={"Cache-Control": "no-store"},
        )

    @app.delete(
        "/v1/files/{file_id}",
        tags=["openai"],
        operation_id="openai_file_delete",
    )
    def openai_file_delete(file_id: str) -> JSONResponse:
        rec = file_store.delete(file_id)
        if rec is None:
            raise ApiError(404, f"file {file_id!r} not found", code="file_not_found")
        return JSONResponse({"id": file_id, "object": "file", "deleted": True})

    # --- /v1/uploads ------------------------------------------------------
    # Chunked upload surface: create an intent, add parts, complete into a
    # /v1/files record. Same durability contract as files (journaled meta
    # + blob parts under --state-dir). Parts are multipart `data` fields,
    # bounded by the declared byte count — the file store's cap applies at
    # create, so a completed upload can never exceed one file's budget.

    def _upload_err(exc: UploadStoreError) -> ApiError:
        return ApiError(exc.status, str(exc), code=exc.code)

    async def _upload_part_idem_claim(
        upload_id: str,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> AsyncIterator[None]:
        skey = _idem_scope(_idem_key(idempotency_key), namespace=f"part:{upload_id}")
        async with upload_idem_store.async_claim_lock(skey):
            yield

    def _upload_verb_skey(upload_id: str, verb: str, idempotency_key: str | None) -> str | None:
        """Shared scoped-key compute for the per-upload verb routes — kept
        as a helper so the deps below compose the exact same key the
        handler's ``_idem_lookup`` does."""
        return _idem_scope(_idem_key(idempotency_key), namespace=f"{verb}:{upload_id}")

    async def _upload_complete_idem_claim(
        upload_id: str,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> AsyncIterator[None]:
        skey = _upload_verb_skey(upload_id, "complete", idempotency_key)
        async with upload_idem_store.async_claim_lock(skey):
            yield

    async def _upload_cancel_idem_claim(
        upload_id: str,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> AsyncIterator[None]:
        skey = _upload_verb_skey(upload_id, "cancel", idempotency_key)
        async with upload_idem_store.async_claim_lock(skey):
            yield

    @app.post(
        "/v1/uploads",
        tags=["openai"],
        operation_id="openai_upload_create",
    )
    def openai_upload_create(
        body: OpenAIUploadCreateRequest,
        _idem_claim_held: None = Depends(_upload_idem_claim),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> JSONResponse:
        """Open an upload intent — parts land under ``.../parts`` until
        ``complete`` assembles them into a file record.``Idempotency-Key``
        pins the intent: a retried create replays the same ``upload_id``."""
        body_fp = _body_fp(body)
        key, replay = _idem_lookup(idempotency_key, upload_idem_store, body_fp, namespace="create")
        if replay is not None:
            return JSONResponse(replay.envelope, headers={"X-Fx1-Idempotent-Replay": "true"})
        _drain_refusal(metrics)
        try:
            validate_upload_intent(body.purpose, body.filename, OPENAI_FILE_PURPOSE_ACCEPT)
            meta = upload_store.create(
                purpose=body.purpose,
                filename=body.filename,
                nbytes=body.bytes,
                mime_type=body.mime_type,
            )
        except UploadStoreError as exc:
            raise _upload_err(exc) from exc
        out = upload_object(meta)
        if key is not None:
            upload_idem_store.put(key, body_fp, _JsonIdemRecord(envelope=out))
        return JSONResponse(out, status_code=200)

    @app.post(
        "/v1/uploads/{upload_id}/parts",
        tags=["openai"],
        operation_id="openai_upload_part",
    )
    async def openai_upload_part(
        upload_id: str,
        data: UploadFile | None = File(default=None),
        _idem_claim_held: None = Depends(_upload_part_idem_claim),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> JSONResponse:
        """Add one part (multipart ``data`` field). Blob lands durable
        before the journal names it. A keyed retry replays the recorded
        part instead of appending the bytes twice."""
        if data is None:
            raise ApiError(400, "multipart field 'data' is required", code="invalid_request")
        blob = await data.read()
        part_fp = hashlib.sha256(blob).hexdigest()
        key, replay = _idem_lookup(
            idempotency_key, upload_idem_store, part_fp, namespace=f"part:{upload_id}"
        )
        if replay is not None:
            return JSONResponse(replay.envelope, headers={"X-Fx1-Idempotent-Replay": "true"})
        _drain_refusal(metrics)
        try:
            part = upload_store.add_part(upload_id, blob)
        except UploadStoreError as exc:
            raise _upload_err(exc) from exc
        if key is not None:
            upload_idem_store.put(key, part_fp, _JsonIdemRecord(envelope=part))
        return JSONResponse(part)

    @app.post(
        "/v1/uploads/{upload_id}/complete",
        tags=["openai"],
        operation_id="openai_upload_complete",
    )
    def openai_upload_complete(
        upload_id: str,
        body: OpenAIUploadCompleteRequest,
        _idem_claim_held: None = Depends(_upload_complete_idem_claim),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> JSONResponse:
        """Validate and mint one file under the upload lifecycle lock. A
        keyed retry replays the minted file object instead of failing on
        the already-completed upload."""
        body_fp = _body_fp(body)
        key, replay = _idem_lookup(
            idempotency_key, upload_idem_store, body_fp, namespace=f"complete:{upload_id}"
        )
        if replay is not None:
            return JSONResponse(replay.envelope, headers={"X-Fx1-Idempotent-Replay": "true"})
        _drain_refusal(metrics)

        def publish(meta: UploadMeta, content: bytes) -> tuple[str, _FileRecord]:
            rec = file_store.put(filename=meta.filename, purpose=meta.purpose, content=content)
            return rec.file_id, rec

        def rollback(file_id: str) -> None:
            file_store.delete(file_id)

        try:
            done, rec = upload_store.complete_with(
                upload_id, body.part_ids, publish=publish, rollback=rollback, md5=body.md5
            )
        except UploadStoreError as exc:
            raise _upload_err(exc) from exc
        out = upload_object(done, file_obj=file_object(rec.model_dump()))
        if key is not None:
            upload_idem_store.put(key, body_fp, _JsonIdemRecord(envelope=out))
        return JSONResponse(out)

    @app.post(
        "/v1/uploads/{upload_id}/cancel",
        tags=["openai"],
        operation_id="openai_upload_cancel",
    )
    def openai_upload_cancel(
        upload_id: str,
        _idem_claim_held: None = Depends(_upload_cancel_idem_claim),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> JSONResponse:
        """Cancel a pending upload — replays 200 on an already-cancelled
        record, 409 once completed; a keyed retry replays the recorded
        cancel answer verbatim."""
        key, replay = _idem_lookup(
            idempotency_key, upload_idem_store, "cancel", namespace=f"cancel:{upload_id}"
        )
        if replay is not None:
            return JSONResponse(replay.envelope, headers={"X-Fx1-Idempotent-Replay": "true"})
        try:
            meta = upload_store.cancel(upload_id)
        except UploadStoreError as exc:
            raise _upload_err(exc) from exc
        out = upload_object(meta)
        if key is not None:
            upload_idem_store.put(key, "cancel", _JsonIdemRecord(envelope=out))
        return JSONResponse(out)

    async def _openai_batch_idem_claim(
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> AsyncIterator[None]:
        """Serialize lookup + creation + insertion for one batch key."""
        key = _idem_key(idempotency_key)
        async with openai_idem_store.async_claim_lock(_idem_scope(key)):
            yield

    @app.post(
        "/v1/batches",
        tags=["openai"],
        operation_id="openai_batches_create",
    )
    def openai_batches_create(
        body: OpenAIBatchRequest,
        request: Request,
        _idem_claim_held: None = Depends(_openai_batch_idem_claim),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> JSONResponse:
        """Submit a batch over an uploaded input file. One worker slot
        runs the whole batch through the gated route cores; ``expires_at``
        is +24h (the completion_window the surface declares).
        ``callback_url``/``callback_secret`` are the fx1 webhook
        extension: the terminal batch envelope
        (completed/failed/expired/cancelled — including expiry observed
        on read) is POSTed to the URL once, HMAC-signed when the secret
        is set; ``callback_status``/``callback_attempts``/
        ``callback_error`` ride the projected record."""
        body_fp = _body_fp(body, headers=request.headers)
        key, replay = _idem_lookup(idempotency_key, openai_idem_store, body_fp)
        if replay is not None:
            return JSONResponse(replay.envelope, headers={"X-Fx1-Idempotent-Replay": "true"})
        frec = file_store.get(body.input_file_id)
        if frec is None:
            raise ApiError(404, f"file {body.input_file_id!r} not found", code="file_not_found")
        if frec.purpose != "batch":
            raise ApiError(
                400,
                f"file {body.input_file_id!r} purpose is {frec.purpose!r}, not 'batch'",
                code="invalid_request",
            )
        try:
            text = frec.content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ApiError(400, "input file is not valid UTF-8", code="invalid_request") from exc
        raw_lines = [ln for ln in text.splitlines() if ln.strip()]
        if not raw_lines:
            raise ApiError(400, "input file has no request lines", code="invalid_request")
        if len(raw_lines) > batch_line_max:
            raise ApiError(
                400,
                f"input file has {len(raw_lines)} lines > cap {batch_line_max}",
                code="batch_input_limit",
            )
        parsed: builtins.list[dict[str, Any]] = []
        for i, raw in enumerate(raw_lines, 1):
            try:
                parsed.append(batch_line_shape(json.loads(raw), endpoint=body.endpoint, lineno=i))
            except json.JSONDecodeError as exc:
                raise ApiError(
                    400, f"line {i}: invalid JSON — {exc}", code="invalid_request"
                ) from exc
            except OpenAICompatError as exc:
                raise ApiError(exc.status, str(exc), code=exc.code) from exc
        if metrics.draining.is_set():
            raise ApiError(503, "harness is draining — no new work accepted", code="draining")
        if not inflight.acquire(blocking=False):
            raise ApiError(
                503,
                "harness at max_inflight — retry later",
                code="over_capacity",
                headers={"Retry-After": "1"},
            )
        metrics.acquire()
        now = int(time.time())
        batch = _BatchRecord(
            batch_id=f"batch_{uuid.uuid4().hex}",
            input_file_id=frec.file_id,
            endpoint=body.endpoint,
            completion_window=body.completion_window,
            metadata=body.metadata,
            status="validating",
            created_at=now,
            expires_at=now + 86400,
            request_counts=_BatchCounts(total=len(parsed)),
            callback_url=body.callback_url,
        )
        batch._callback_secret = body.callback_secret
        batch._lines = parsed
        # the caller's X-Fx1-* routing headers apply to every line — the
        # batch inherits the submitter's backend choice, never ambient env.
        batch._headers = {
            k: v for k, v in request.headers.items() if k.lower().startswith("x-fx1-")
        }
        # every line runs under the submitting credential's fingerprint —
        # worker threads do not inherit request contextvars, so the key_id
        # rides the record and the worker re-installs it per line.
        batch._key_id = _REQUEST_KEY_ID.get()
        _submit_after_persist(
            persist=lambda: batch_store.put(batch),
            execute=lambda: _exec_batch(batch),
        )
        env = _batch_project(batch)
        if key is not None:
            openai_idem_store.put(key, body_fp, _OpenAIIdemRecord(envelope=env))
        return JSONResponse(env)

    @app.get(
        "/v1/batches",
        tags=["openai"],
        operation_id="openai_batches_list",
    )
    def openai_batches_list(
        limit: int = Query(default=20, ge=1, le=100),
        after: str | None = Query(default=None),
    ) -> JSONResponse:
        """Newest-first batch listing; ``after`` pages by batch id."""
        items = batch_store.list()
        if after is not None:
            idx = next((i for i, b in enumerate(items) if b.batch_id == after), None)
            if idx is None:
                raise ApiError(400, f"cursor {after!r} is not a batch id", code="invalid_cursor")
            items = items[idx + 1 :]
        page = items[:limit]
        return JSONResponse(
            {
                "object": "list",
                "data": [_batch_project(b) for b in page],
                "first_id": page[0].batch_id if page else None,
                "last_id": page[-1].batch_id if page else None,
                "has_more": len(items) > limit,
            }
        )

    @app.get(
        "/v1/batches/{batch_id}",
        tags=["openai"],
        operation_id="openai_batches_get",
    )
    def openai_batches_get(batch_id: str) -> JSONResponse:
        batch = batch_store.get(batch_id)
        if batch is None:
            raise ApiError(404, f"batch {batch_id!r} not found", code="batch_not_found")
        return JSONResponse(_batch_project(batch))

    @app.post(
        "/v1/batches/{batch_id}/cancel",
        tags=["openai"],
        operation_id="openai_batches_cancel",
    )
    def openai_batches_cancel(batch_id: str) -> JSONResponse:
        """Cooperative cancel: the worker checks the flag between lines —
        an in-flight line finishes, then the batch lands 'cancelled' with
        whatever output lines exist written to output_file_id."""
        batch = batch_store.get(batch_id)
        if batch is None:
            raise ApiError(404, f"batch {batch_id!r} not found", code="batch_not_found")
        with batch._state_lock:
            if batch.status in _BATCH_TERMINAL:
                raise ApiError(
                    409,
                    f"batch {batch_id!r} is already {batch.status}",
                    code="batch_terminal",
                )
            if batch.status == "cancelling":
                already_cancelling = True
            else:
                already_cancelling = False
                batch._cancel.set()
                batch.status = "cancelling"
                batch.cancelling_at = int(time.time())
        if already_cancelling:
            return JSONResponse(_batch_project(batch))
        batch_store.mark(batch)
        return JSONResponse(_batch_project(batch))

    @app.post(
        "/harness/complete/batch",
        response_model=CompleteBatchResponse,
        tags=["complete"],
        operation_id="complete_batch",
    )
    def complete_batch(
        body: CompleteBatchRequest,
        _idem_claim_held: None = Depends(_complete_batch_idem_claim),
        _slot_held: None = Depends(slot),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> CompleteBatchResponse:
        body_fp = _body_fp(body)
        key, replay = _idem_lookup(idempotency_key, complete_batch_idem_store, body_fp)
        if replay is not None:
            return replay
        _check_citations(body.receipt_hashes)
        if any(
            m.role == "tool" or m.tool_calls is not None or m.tool_call_id is not None
            for msgs in body.batch
            for m in msgs
        ):
            # text-only surface — tool turns take the richer /v1 wire
            raise ApiError(
                422,
                "tool-call context isn't a batch surface — run agent turns "
                "through /v1/chat/completions or /harness/complete",
            )
        # Resolve-level fallback: first resolvable link serves the whole
        # batch — a shared backend can't attribute per-item usage, so
        # per-item failover is intentionally not offered.
        serving, backend, batch_attempts = _resolve_chain(body)
        sampling_params = _sampling_of(body)
        sampling_fields = sampling_params.body_fields()
        usage_pre = getattr(backend, "total_usage", None)
        usage_pre = dict(usage_pre) if isinstance(usage_pre, dict) else None
        # Key attribution must be captured here — worker threads in the
        # pool do not inherit this request's contextvars.
        req_key_id = _REQUEST_KEY_ID.get()
        # One backend serves the whole batch — a spawned local engine is
        # shared across workers (spawn path is lock-guarded). Item failures
        # are per-slot verdicts: a gate refusal on one prompt does not lose
        # the rest of the batch. An open circuit short-circuits items
        # individually so a batch over a dead backend ends fast.
        try:
            with ThreadPoolExecutor(
                max_workers=min(body.max_workers, len(body.batch)),
                thread_name_prefix="fx1-complete",
            ) as pool:

                def _one(messages: list[dict[str, str]]) -> CompleteBatchItem:
                    t0 = time.monotonic()
                    cid = uuid.uuid4().hex
                    prompt_sha256 = hashlib.sha256(
                        json.dumps(messages, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    ).hexdigest()
                    model_snap = getattr(backend, "_model", None)

                    def _log_item(
                        ok: bool, content: str | None, err: str | None, cls: str | None
                    ) -> None:
                        completion_log.append(
                            CompletionRecord(
                                completion_id=cid,
                                backend=serving,
                                model=model_snap if isinstance(model_snap, str) else None,
                                ok=ok,
                                latency_ms=(time.monotonic() - t0) * 1000.0,
                                at=time.time(),
                                # usage is only honest at batch level (shared
                                # endpoint attribution) — per-item stays None.
                                key_id=req_key_id,
                                error=err,
                                error_class=cls,
                                prompt_sha256=prompt_sha256,
                                sampling=sampling_fields,
                                user=body.user,
                                metadata=body.metadata,
                                output_sha256=(
                                    hashlib.sha256(content.encode("utf-8")).hexdigest()
                                    if content is not None
                                    else None
                                ),
                            )
                        )

                    serving_key = _breaker_key_name(serving, body.byok)
                    if breaker is not None and breaker.check(serving_key) > 0:
                        _log_item(False, None, "backend circuit open", "backend_unavailable")
                        return CompleteBatchItem(
                            ok=False,
                            latency_ms=0.0,
                            error="backend circuit open",
                            error_class="backend_unavailable",
                            completion_id=cid,
                        )
                    item_ok = False
                    try:
                        content = cited_complete(
                            backend,
                            messages,
                            receipt_hashes=body.receipt_hashes,
                            sampling=sampling_params,
                        )
                        item_ok = True
                    except Fx1HonestyError as exc:
                        _log_item(False, None, str(exc), "honesty_refusal")
                        return CompleteBatchItem(
                            ok=False,
                            latency_ms=(time.monotonic() - t0) * 1000.0,
                            error=str(exc),
                            error_class="honesty_refusal",
                            completion_id=cid,
                        )
                    except (
                        BackendNotConfiguredError,
                        RuntimeError,
                        ValueError,
                    ) as exc:
                        if breaker is not None and not isinstance(exc, NotImplementedError):
                            breaker.report(serving_key, False)
                        _log_item(False, None, str(exc), type(exc).__name__)
                        return CompleteBatchItem(
                            ok=False,
                            latency_ms=(time.monotonic() - t0) * 1000.0,
                            error=str(exc),
                            error_class=type(exc).__name__,
                            completion_id=cid,
                        )
                    finally:
                        metrics.record_complete(
                            serving,
                            item_ok,
                            (time.monotonic() - t0) * 1000.0,
                            usage=_billable_usage(getattr(backend, "last_usage", None)),
                        )
                    _log_item(True, content, None, None)
                    if breaker is not None:
                        breaker.report(serving_key, True)
                    return CompleteBatchItem(
                        ok=True,
                        latency_ms=(time.monotonic() - t0) * 1000.0,
                        content=content,
                        completion_id=cid,
                    )

                results = list(
                    pool.map(
                        _one,
                        [
                            [{"role": m.role, "content": cast(str, m.content)} for m in msgs]
                            for msgs in body.batch
                        ],
                    )
                )
        finally:
            _close_backend(backend)
        model_name = getattr(backend, "_model", None)
        usage_post = getattr(backend, "total_usage", None)
        if isinstance(usage_post, dict) and usage_post:
            usage_total = {
                k: v - (usage_pre.get(k, 0) if usage_pre else 0) for k, v in usage_post.items()
            }
        else:
            usage_total = None
        # Batch items carry no per-call usage (shared endpoint can't
        # attribute it) — charge the key the batch's aggregate delta so
        # token budgets still meter batch traffic.
        if req_key_id is not None and usage_total:
            _tt = usage_total.get("total_tokens")
            if not isinstance(_tt, int):
                _pt = usage_total.get("prompt_tokens")
                _ct = usage_total.get("completion_tokens")
                _tt = (_pt if isinstance(_pt, int) else 0) + (_ct if isinstance(_ct, int) else 0)
            key_store.charge_tokens(req_key_id, _tt)
        resp = CompleteBatchResponse(
            backend=serving,
            model=model_name if isinstance(model_name, str) else None,
            receipt_hashes=body.receipt_hashes or [],
            results=results,
            sampling=sampling_fields,
            usage_total=usage_total,
            attempts=batch_attempts if len(batch_attempts) > 1 else [],
        )
        if key is not None:
            complete_batch_idem_store.put(key, body_fp, resp)
        return resp

    @app.post("/harness/backends/{name}/probe", tags=["ops"], operation_id="backend_probe")
    def backend_probe(
        name: Literal["hosted_k3", "local_fx1", "byok"],
        body: BackendProbeRequest | None = None,
        _slot_held: None = Depends(slot),
    ) -> BackendProbeResponse:
        """Deep health: run one minimal gated completion through the real
        resolver. Unlike ``GET /harness/backends`` (config + circuit state),
        this answers "can this backend serve right now" — including a BYOK
        endpoint supplied inline. Deliberately bypasses the breaker admit
        and never reports to it, so a monitoring scrape can't trip or heal
        the circuit; the verdict series lands under ``probe:<name>``."""
        req = body or BackendProbeRequest()
        t0 = time.monotonic()

        def _verdict(resp: BackendProbeResponse) -> BackendProbeResponse:
            with probe_lock:
                probe_cache[name] = BackendProbeVerdict(
                    ok=resp.ok,
                    latency_ms=resp.latency_ms,
                    checked_at=time.time(),
                    error_class=resp.error_class,
                )
            return resp

        try:
            backend = resolve_backend(
                name,
                req.checkpoint_dir,
                req.byok.model_dump() if req.byok is not None else None,
                req.timeout_s if req.timeout_s is not None else 30.0,
            )
        except ApiError as exc:
            # An unconfigured/unreachable backend is a verdict, not an
            # HTTP fault — report it as ok:false. Client-side arg errors
            # (404/422) still propagate as request errors. The verdict
            # still meters under probe:<name> — a resolver failure is a
            # probe outcome, and a monitoring scrape that only watches the
            # series must see it.
            if exc.status_code == 503:
                metrics.record_complete(
                    f"probe:{name}",
                    False,
                    (time.monotonic() - t0) * 1000.0,
                    usage=None,
                )
                return _verdict(
                    BackendProbeResponse(
                        backend=name,
                        ok=False,
                        model=None,
                        latency_ms=(time.monotonic() - t0) * 1000.0,
                        error=str(exc.detail),
                        error_class="backend_unavailable",
                    )
                )
            raise
        ok = False
        error: str | None = None
        error_class: str | None = None
        try:
            content = backend.complete([{"role": "user", "content": req.prompt}])
            try:
                validate_fx1_output(content)
            except Fx1HonestyError as exc:
                error, error_class = str(exc), "honesty_refusal"
            else:
                ok = True
        except NotImplementedError as exc:
            error, error_class = str(exc), "NotImplementedError"
        except (BackendNotConfiguredError, RuntimeError, ValueError) as exc:
            error, error_class = str(exc), type(exc).__name__
        finally:
            metrics.record_complete(
                f"probe:{name}",
                ok,
                (time.monotonic() - t0) * 1000.0,
                usage=_billable_usage(getattr(backend, "last_usage", None)),
            )
            _close_backend(backend)
        model_name = getattr(backend, "_model", None)
        return _verdict(
            BackendProbeResponse(
                backend=name,
                ok=ok,
                model=model_name if isinstance(model_name, str) else None,
                latency_ms=(time.monotonic() - t0) * 1000.0,
                error=error,
                error_class=error_class,
            )
        )

    @app.post(
        "/harness/gate/check",
        response_model=GateCheckResponse,
        tags=["ops"],
        operation_id="gate_check",
    )
    async def gate_check(body: GateCheckRequest) -> GateCheckResponse:
        """Pre-flight the honesty gate without spending model tokens —
        writers (fx-1 or BYOK callers) can validate text before or after
        generation. Advisory: not slot-gated, stays up during drain, and
        never touches a backend or the metrics series."""
        try:
            validate_fx1_output(body.text)
        except Fx1HonestyError as exc:
            return GateCheckResponse(ok=False, error=str(exc))
        return GateCheckResponse(ok=True)

    @app.post(
        "/harness/score",
        response_model=ScoreResponse,
        tags=["harness"],
        operation_id="harness_score",
    )
    async def score(body: ScoreRequest) -> ScoreResponse:
        """Score text through the deterministic reward contract — the same
        breakdown the corpus and reward lanes use (honesty violation caps
        the total at -10; empty text scores 0). Advisory like the gate
        pre-flight: no backend, no slot, stays up during drain."""
        texts = [body.input] if isinstance(body.input, str) else body.input
        return ScoreResponse(
            object="list",
            data=[
                ScoreItem(
                    index=i,
                    total=bd.total,
                    components=bd.components,
                    violations=bd.violations,
                )
                for i, bd in enumerate(score_response(t) for t in texts)
            ],
        )

    @app.post(
        "/v1/moderations",
        response_model=ModerationResponse,
        tags=["openai"],
        operation_id="openai_create_moderation",
    )
    async def openai_create_moderation(body: ModerationRequest) -> ModerationResponse:
        """OpenAI-compatible moderation surface over the honesty gate: each
        input is classified against the three gate categories (forbidden
        headline metric, live/synthetic-as-live claim, unlabeled synthetic
        evidence) and flagged when any fires. The ``id`` is content-derived
        (``modr-<sha256>``) so identical inputs get identical receipts.
        Advisory like the other preflight surfaces: no backend, no slot,
        stays up during drain."""
        texts = [body.input] if isinstance(body.input, str) else body.input
        results = []
        for text in texts:
            categories = honesty_categories(text)
            results.append(
                ModerationResult(
                    flagged=any(categories.values()),
                    categories=categories,
                    category_scores={name: 1.0 if hit else 0.0 for name, hit in categories.items()},
                    category_applied_input_types={name: ["text"] for name in categories},
                )
            )
        digest = hashlib.sha256("\x1e".join(texts).encode("utf-8")).hexdigest()
        return ModerationResponse(
            id=f"modr-{digest[:24]}",
            model="fx1-honesty-gate",
            results=results,
        )

    # --- /v1/fine_tuning/jobs -------------------------------------------
    # OpenAI's fine-tuning surface over the staged fx-1 Pipeline: quality
    # gate -> frozen split -> base eval -> receipted train -> candidate
    # eval -> ship-gate comparison. The training file is validated
    # synchronously at submit (chat-format JSONL, every line), so a bad
    # upload never reaches the queue. The runner is injectable; the
    # shipped default wires the real stages and fails honestly at the
    # trainer when no GPU backend is configured.

    def _ft_worker(entry: FTJobEntry, spec: FTJobSpec) -> None:
        """One inflight slot drives the staged pipeline; the runner checks
        the entry's cancel event between stages — a job cancelled
        mid-pipeline ends ``cancelled``, not ``failed``."""
        job: FTJob = entry.job
        try:
            # a job paused while queued parks here — resume (or cancel,
            # which also opens the gate) releases it
            entry.resume.wait()
            if entry.cancel.is_set():
                job.status = "cancelled"
                job.finished_at = int(time.time())
                ft_store.mark(entry)
                return
            job.status = "running"
            ft_store.add_event(job.id, "info", "job started", None)
            ft_store.mark(entry)

            def _pause_gate() -> bool:
                """Block while paused; True iff a cancel landed parked."""
                entry.resume.wait()
                return entry.cancel.is_set()

            # pause_gate is opt-in on the runner contract: runners that
            # accept it park at stage boundaries; older runners simply
            # can't pause mid-pipeline (queued pause still holds — the
            # pre-start gate above is worker-side)
            _runner_extra: dict[str, Any] = {}
            try:
                _sig = inspect.signature(ft_runner)
                if "pause_gate" in _sig.parameters or any(
                    p.kind is inspect.Parameter.VAR_KEYWORD for p in _sig.parameters.values()
                ):
                    _runner_extra["pause_gate"] = _pause_gate
            except (TypeError, ValueError):  # pragma: no cover - C callables
                _runner_extra["pause_gate"] = _pause_gate
            outcome = ft_runner(
                spec,
                emit=lambda level, msg, data=None: ft_store.add_event(
                    job.id,
                    cast(Literal["info", "warn", "error"], level),
                    msg,
                    data,
                ),
                should_cancel=entry.cancel.is_set,
                **_runner_extra,
            )
            if entry.cancel.is_set():
                # the cancel sweep can stamp the terminal status before the
                # runner returns — read wide so the check isn't narrowed to
                # the last in-scope assignment
                status_now: str = job.status
                already = status_now == "cancelled"
                job.status = "cancelled"
                if not already:
                    ft_store.add_event(job.id, "info", _EV_JOB_CANCELLED, None)
            else:
                for name, path in outcome.artifacts.items():
                    try:
                        content = Path(path).read_bytes()
                    except OSError:
                        continue
                    rec = file_store.put(
                        filename=f"{job.id}-{Path(path).name}",
                        purpose="fine-tune-result",
                        content=content,
                    )
                    job.result_files.append(rec.file_id)
                    ft_store.add_event(
                        job.id,
                        "info",
                        f"result artifact registered: {name}",
                        {"file_id": rec.file_id, "path": str(path)},
                    )
                job.fine_tuned_model = outcome.fine_tuned_model
                job.trained_tokens = outcome.trained_tokens
                if outcome.fine_tuned_model is not None and outcome.checkpoint:
                    ft_store.register_model(
                        outcome.fine_tuned_model,
                        job_id=job.id,
                        checkpoint=outcome.checkpoint,
                        created=job.finished_at or int(time.time()),
                    )
                    ft_store.add_event(
                        job.id,
                        "info",
                        f"model registered: {outcome.fine_tuned_model}",
                        {"checkpoint": outcome.checkpoint},
                    )
                ft_store.add_event(
                    job.id,
                    "info",
                    "job succeeded",
                    {"checkpoint": outcome.checkpoint},
                )
                # Publish the terminal status only after the complete event
                # feed is durable. Otherwise a reader can observe
                # ``succeeded`` and immediately retrieve a truncated feed.
                job.status = "succeeded"
        except Exception as exc:  # noqa: BLE001 — a runner fault is job data
            if entry.cancel.is_set():
                already = job.status == "cancelled"
                job.status = "cancelled"
                if not already:
                    ft_store.add_event(job.id, "info", _EV_JOB_CANCELLED, None)
            else:
                job.error = FTJobError(
                    code="job_failed",
                    message=f"{type(exc).__name__}: {exc}",
                    param=None,
                )
                ft_store.add_event(
                    job.id, "error", f"job failed: {type(exc).__name__}: {exc}", None
                )
                # As on success, terminal status means terminal retrieval
                # surfaces are already complete.
                job.status = "failed"
        finally:
            job.finished_at = job.finished_at or int(time.time())
            _deliver_callback(job)
            ft_store.mark(entry)
            metrics.release()
            inflight.release()

    @app.post("/v1/fine_tuning/jobs", tags=["openai"], operation_id="create_finetune_job")
    def create_finetune_job(
        body: FTJobRequest,
        _idem_claim_held: None = Depends(_ft_idem_claim),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> FTJob:
        """Queue a gated fine-tuning run against an uploaded chat-format
        JSONL training file. File validation is synchronous — malformed
        corpora 400 at submit, never limbo in ``validating_files``.
        ``callback_url``/``callback_secret`` are the fx1 webhook
        extension: the terminal job record (succeeded/failed/cancelled)
        is POSTed to the URL, HMAC-signed when the secret is set — the
        same delivery contract as ``/harness/jobs`` webhooks."""
        body_fp = _body_fp(body)
        skey = _idem_scope(_idem_key(idempotency_key))
        if skey is not None:
            entry = ft_store.lookup_idem(skey)
            if entry is not None:
                if entry.body_fp != body_fp:
                    raise ApiError(
                        409,
                        "Idempotency-Key reuse with a different request body",
                        code="idempotency_conflict",
                    )
                return entry.job
        if body.model not in TRAINABLE_MODELS:
            raise ApiError(
                400,
                f"model {body.model!r} is not trainable through the harness "
                f"(trainable: {list(TRAINABLE_MODELS)})",
                code="model_not_trainable",
            )
        frec = file_store.get(body.training_file)
        if frec is None:
            raise ApiError(
                404, f"training file {body.training_file!r} not found", code="file_not_found"
            )
        if frec.purpose != "fine-tune":
            raise ApiError(
                400,
                f"file {body.training_file!r} was uploaded with purpose "
                f"{frec.purpose!r} — training corpora upload as 'fine-tune'",
                code="invalid_training_file",
            )
        try:
            n_examples = validate_chat_jsonl(frec.content, file_id=body.training_file)
        except ValueError as exc:
            raise ApiError(400, str(exc), code="invalid_training_file") from exc
        vrec = None
        if body.validation_file is not None:
            vrec = file_store.get(body.validation_file)
            if vrec is None:
                raise ApiError(
                    404,
                    f"validation file {body.validation_file!r} not found",
                    code="file_not_found",
                )
            if vrec.purpose != "fine-tune":
                raise ApiError(
                    400,
                    f"file {body.validation_file!r} was uploaded with purpose "
                    f"{vrec.purpose!r} — training corpora upload as 'fine-tune'",
                    code="invalid_training_file",
                )
            try:
                validate_chat_jsonl(vrec.content, file_id=body.validation_file)
            except ValueError as exc:
                raise ApiError(400, str(exc), code="invalid_training_file") from exc
        if metrics.draining.is_set():
            raise ApiError(503, "harness is draining — no new work accepted", code="draining")
        if not inflight.acquire(blocking=False):
            raise ApiError(
                503,
                f"harness at max_inflight={metrics.max_inflight} — retry later",
                code="over_capacity",
                headers={"Retry-After": "1"},
            )
        metrics.acquire()
        job = FTJob(
            id=f"ftjob-{uuid.uuid4().hex}",
            model=body.model,
            created_at=int(time.time()),
            status="queued",
            training_file=body.training_file,
            validation_file=body.validation_file,
            hyperparameters=body.hyperparameters or FTHyperparameters(),
            seed=body.seed,
            metadata=body.metadata,
            user_provided_suffix=body.suffix,
            callback_url=body.callback_url,
        )
        job._callback_secret = body.callback_secret
        ft_name = f"ft:{body.model}:{body.suffix or 'job'}:{job.id.split('-', 1)[1][:12]}"
        work_dir = ft_dir / job.id
        work_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = work_dir / "corpus.jsonl"
        corpus_path.write_bytes(frec.content)
        val_path = None
        if vrec is not None:
            val_path = work_dir / "validation.jsonl"
            val_path.write_bytes(vrec.content)
        spec = FTJobSpec(
            job_id=job.id,
            model=body.model,
            corpus_path=corpus_path,
            val_path=val_path,
            hyperparameters=(
                body.hyperparameters.model_dump(mode="json")
                if body.hyperparameters is not None
                else {}
            ),
            seed=body.seed if body.seed is not None else 17,
            work_dir=work_dir,
            ft_model_name=ft_name,
        )
        entry = ft_store.put(job, skey, body_fp)
        hp = body.hyperparameters.model_dump() if body.hyperparameters else {}
        if hp.get("batch_size"):
            ft_store.add_event(
                job.id,
                "info",
                "batch_size is advisory — the staged pipeline's trainer "
                "decides batching; the value is recorded on the job",
                {"batch_size": hp["batch_size"]},
            )
        ft_store.add_event(
            job.id,
            "info",
            f"training file validated: {n_examples} examples",
            {"training_file": body.training_file, "examples": n_examples},
        )
        try:
            jobs_executor.submit(_ft_worker, entry, spec)
        except RuntimeError as exc:  # executor gone (shutdown race)
            metrics.release()
            inflight.release()
            raise ApiError(503, "job executor unavailable", code="over_capacity") from exc
        return job

    @app.get(
        "/v1/fine_tuning/jobs",
        response_model=FTJobList,
        tags=["openai"],
        operation_id="list_finetune_jobs",
    )
    def list_finetune_jobs(
        limit: int = Query(default=20, ge=1, le=100), after: str | None = Query(default=None)
    ) -> FTJobList:
        """Newest-first page; ``after`` is the exclusive id cursor."""
        jobs, has_more = ft_store.list_jobs(limit=limit, after=after)
        return FTJobList(data=jobs, has_more=has_more)

    @app.get(
        "/v1/fine_tuning/jobs/{job_id}",
        response_model=FTJob,
        tags=["openai"],
        operation_id="get_finetune_job",
    )
    def get_finetune_job(job_id: str) -> FTJob:
        entry = ft_store.get(job_id)
        if entry is None:
            raise ApiError(404, f"fine-tuning job {job_id!r} not found", code="job_not_found")
        return entry.job

    @app.post(
        "/v1/fine_tuning/jobs/{job_id}/cancel",
        tags=["openai"],
        operation_id="cancel_finetune_job",
    )
    def cancel_finetune_job(job_id: str) -> FTJob:
        """Cooperative cancel — a queued job ends immediately; a running
        one is marked and the pipeline stops at the next stage boundary."""
        outcome = ft_store.request_cancel(job_id)
        if outcome == "missing":
            raise ApiError(404, f"fine-tuning job {job_id!r} not found", code="job_not_found")
        entry = ft_store.get(job_id)
        assert entry is not None  # noqa: S101 — request_cancel found it
        if outcome == "terminal":
            raise ApiError(
                409,
                f"job {job_id!r} is already {entry.job.status} — only "
                "queued or running jobs can be cancelled",
                code="job_terminal",
            )
        if outcome in ("queued", "paused"):
            ft_store.add_event(job_id, "info", _EV_JOB_CANCELLED, None)
        else:
            ft_store.add_event(
                job_id,
                "info",
                "cancellation requested — takes effect at the next stage boundary",
                None,
            )
        return entry.job

    @app.post(
        "/v1/fine_tuning/jobs/{job_id}/pause",
        tags=["openai"],
        operation_id="pause_finetune_job",
    )
    def pause_finetune_job(job_id: str) -> FTJob:
        """Cooperative pause — a queued job parks before starting; a
        running one parks at the next pipeline-stage boundary (the gate
        blocks inside the runner, so an in-flight trainer call is never
        interrupted mid-write). ``paused`` is non-terminal: resume
        restores, cancel still wins, drain still drains. Pausing a paused
        job replays its record — idempotent."""
        outcome = ft_store.request_pause(job_id)
        if outcome == "missing":
            raise ApiError(404, f"fine-tuning job {job_id!r} not found", code="job_not_found")
        entry = ft_store.get(job_id)
        assert entry is not None  # noqa: S101 — request_pause found it
        if outcome == "terminal":
            raise ApiError(
                409,
                f"job {job_id!r} is already {entry.job.status} — only "
                "queued or running jobs can be paused",
                code="job_terminal",
            )
        if outcome == "running":
            ft_store.add_event(
                job_id,
                "info",
                "pause requested — takes effect at the next stage boundary",
                None,
            )
        elif outcome == "queued":
            ft_store.add_event(job_id, "info", "job paused — will not start until resumed", None)
        return entry.job

    @app.post(
        "/v1/fine_tuning/jobs/{job_id}/resume",
        tags=["openai"],
        operation_id="resume_finetune_job",
    )
    def resume_finetune_job(job_id: str) -> FTJob:
        """Resume a paused job — restores the status pause captured
        (queued jobs re-queue, running jobs proceed from the boundary the
        worker parked at). Resuming a non-paused job is a 409."""
        _drain_refusal(metrics)
        outcome = ft_store.request_resume(job_id)
        if outcome == "missing":
            raise ApiError(404, f"fine-tuning job {job_id!r} not found", code="job_not_found")
        entry = ft_store.get(job_id)
        assert entry is not None  # noqa: S101 — request_resume found it
        if outcome == "terminal":
            raise ApiError(
                409,
                f"job {job_id!r} is already {entry.job.status} — a terminal job cannot be resumed",
                code="job_terminal",
            )
        if outcome == "not_paused":
            raise ApiError(
                409,
                f"job {job_id!r} is {entry.job.status}, not paused — "
                "only paused jobs can be resumed",
                code="job_not_paused",
            )
        ft_store.add_event(job_id, "info", "job resumed", None)
        return entry.job

    @app.get(
        "/v1/fine_tuning/jobs/{job_id}/events",
        response_model=FTEventList,
        tags=["openai"],
        operation_id="list_finetune_job_events",
    )
    def list_finetune_job_events(
        job_id: str,
        limit: int = Query(default=20, ge=1, le=100),
        after: str | None = Query(default=None),
    ) -> FTEventList:
        """Oldest-first event feed for one job (OpenAI's order)."""
        entry = ft_store.get(job_id)
        if entry is None:
            raise ApiError(404, f"fine-tuning job {job_id!r} not found", code="job_not_found")
        events, has_more = ft_store.list_events(job_id, limit=limit, after=after)
        return FTEventList(data=events, has_more=has_more)

    @app.get(
        "/v1/fine_tuning/jobs/{job_id}/checkpoints",
        response_model=FTJobCheckpointList,
        tags=["openai"],
        operation_id="list_finetune_job_checkpoints",
    )
    def list_finetune_job_checkpoints(
        job_id: str,
        limit: int = Query(default=10, ge=1, le=100),
        after: str | None = Query(default=None),
    ) -> FTJobCheckpointList:
        """OpenAI's ``fine_tuning.jobs.list_checkpoints`` — the model
        artifacts a job registered, oldest-first. A job that produced no
        model lists empty (never a fabricated checkpoint); a deleted
        ``ft:`` name drops off — tombstones don't fabricate history."""
        if ft_store.get(job_id) is None:
            raise ApiError(404, f"fine-tuning job {job_id!r} not found", code="job_not_found")
        items, has_more = ft_store.checkpoints_for(job_id, limit=limit, after=after)
        return FTJobCheckpointList(
            data=items,
            first_id=items[0].id if items else None,
            last_id=items[-1].id if items else None,
            has_more=has_more,
        )


def _billable_usage(snap: Any) -> dict[str, int] | None:
    """The billable-value sieve on a provider usage claim: only genuine
    ints count — a bool is not a token count, and strings/floats can
    never reach the ledger. ``None`` for an absent channel, a non-dict
    payload, or a claim with nothing billable in it. Negative ints are
    kept verbatim (the provider's claim, reported faithfully — the
    charge path clamps them at zero)."""
    if not isinstance(snap, dict):
        return None
    clean = {str(k): v for k, v in snap.items() if isinstance(v, int) and not isinstance(v, bool)}
    return clean or None


def _as_tokens(value: Any) -> int:
    """One usage claim → its billable int, or 0 — bools, strings, and
    floats are unbillable and never reach the ledger."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _charge_key_tokens(rec: CompletionRecord, key_store: ApiKeyStore) -> None:
    """Fold one served call's provider-reported usage into the key's
    live token-budget meter. ``usage`` is None when the backend has no
    usage channel — a silent provider never fabricates spend. Env-key
    and loopback calls are unmetered."""
    usage = rec.usage
    if rec.key_id in (None, "env") or not usage:
        return
    total = usage.get("total_tokens")
    if not isinstance(total, int) or isinstance(total, bool):
        total = _as_tokens(usage.get("prompt_tokens")) + _as_tokens(usage.get("completion_tokens"))
    key_store.charge_tokens(rec.key_id or "", total)


def _key_refusal_response(
    exc: KeyStoreError, request: Request, key_store: ApiKeyStore
) -> JSONResponse:
    """429 shape for a managed-key refusal. ``quota_exceeded`` is a hard
    budget — no ``Retry-After`` (it never clears inside a call, so
    clients must not retry it); ``rate_limited`` is a window refusal —
    an honest ``Retry-After`` plus the key's standing budget headers.
    ``insufficient_scope`` is the authorization refusal — a 403 in the
    path's own error grammar."""
    path = request.url.path
    if exc.code == "insufficient_scope":
        msg = str(exc)
        scope_body: dict[str, Any] = {"detail": msg, "code": exc.code}
        if is_openai_path(path):
            scope_body = _v1_error_body(path, msg, 403, exc.code)
        return JSONResponse(status_code=403, content=scope_body)
    if exc.code == "quota_exceeded":
        msg = str(exc)
        body: dict[str, Any] = {"detail": msg, "code": exc.code}
        if is_openai_path(path):
            body = _v1_error_body(path, msg, 429, exc.code)
        # A hard budget cannot recover through retry. Keep this explicit
        # route hint when _finish applies generic transient-status hints.
        headers = {"x-should-retry": "false"} if _is_anthropic_surface(request) else None
        return JSONResponse(status_code=429, content=body, headers=headers)
    wait_s = max(1, math.ceil(exc.retry_after or 1.0))
    rl_msg = f"key rate limit exceeded; retry in {wait_s}s"
    rl_body: dict[str, Any] = {"detail": rl_msg, "code": exc.code}
    if is_openai_path(path):
        rl_body = _v1_error_body(path, rl_msg, 429, exc.code)
    headers = {"Retry-After": str(wait_s)}
    headers.update(_key_budget_headers(key_store, exc.key_id))
    if _is_anthropic_surface(request):
        headers.update(_anthropic_budget_headers(key_store, exc.key_id))
    return JSONResponse(status_code=429, content=rl_body, headers=headers)


def _require_admin(request: Request) -> None:
    if not getattr(request.state, "admin", False):
        raise ApiError(
            403,
            "key management requires the bootstrap credential",
            code="admin_required",
        )


def _key_budget_headers(key_store: ApiKeyStore, key_id: str | None) -> dict[str, str]:
    """OpenAI's standing rate-limit headers for a managed key with a
    declared rpm window — empty for env/loopback auth or unwindowed
    keys (no false scarcity)."""
    if key_id in (None, "env"):
        return {}
    ws = key_store.window_state(key_id)
    if ws is None:
        return {}
    return {
        "X-RateLimit-Limit-Requests": str(ws[0]),
        "X-RateLimit-Remaining-Requests": str(ws[1]),
        "X-RateLimit-Reset-Requests": str(ws[2]),
    }


def _key_served_usage(records: list[CompletionRecord], key_id: str) -> KeyServedUsage:
    """Aggregate the completion ring for one credential fingerprint."""
    calls = 0
    prompt = 0
    completion = 0
    total_all = 0
    by_backend: dict[str, list[int]] = {}
    for rec in records:
        if rec.key_id != key_id:
            continue
        calls += 1
        usage = rec.usage or {}
        p = _as_tokens(usage.get("prompt_tokens"))
        c = _as_tokens(usage.get("completion_tokens"))
        t = usage.get("total_tokens")
        tt = t if isinstance(t, int) and not isinstance(t, bool) else p + c
        prompt += p
        completion += c
        total_all += tt
        bb = by_backend.setdefault(rec.backend, [0, 0])
        bb[0] += 1
        bb[1] += tt
    return KeyServedUsage(
        calls=calls,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total_all,
        by_backend={
            k: KeyUsageBackendSplit(calls=v[0], total_tokens=v[1])
            for k, v in sorted(by_backend.items())
        },
    )


def _key_usage_response(
    rec: dict[str, Any],
    key_store: ApiKeyStore,
    completion_log: _CompletionLog,
) -> ApiKeyUsageResponse:
    """Build one key's usage card — counters + budgets + window + the
    completion-ring spend split. Shared by the admin route and the
    managed branch of ``/harness/self``."""
    uses = int(rec.get("uses") or 0)
    tokens_used = int(rec.get("tokens_used") or 0)
    max_req = rec.get("max_requests")
    max_tok = rec.get("max_tokens")
    ws = key_store.window_state(rec["key_id"])
    return ApiKeyUsageResponse(
        id=rec["key_id"],
        name=rec.get("name"),
        admin=bool(rec.get("admin")),
        enabled=bool(rec.get("enabled", True)),
        created_at=rec["created_at"],
        expires_at=rec.get("expires_at"),
        revoked_at=rec.get("revoked_at"),
        rotated_from=rec.get("rotated_from"),
        uses=uses,
        tokens_used=tokens_used,
        last_used_at=rec.get("last_used_at"),
        max_requests=max_req,
        requests_remaining=(max(0, int(max_req) - uses) if max_req is not None else None),
        max_tokens=max_tok,
        tokens_remaining=(max(0, int(max_tok) - tokens_used) if max_tok is not None else None),
        rpm=rec.get("rpm"),
        window_remaining=(ws[1] if ws is not None else None),
        window_reset_s=(ws[2] if ws is not None else None),
        served=_key_served_usage(completion_log.all(), rec["key_id"]),
        log_cap=completion_log.cap,
        log_dropped=completion_log.dropped,
    )


def _mount_key_lifecycle(
    app: FastAPI,
    *,
    key_store: ApiKeyStore,
    key_idem_store: _IdemStore[_JsonIdemRecord],
    completion_log: _CompletionLog,
    metrics: _Metrics,
) -> None:
    """Key lifecycle routes beyond mint/get/revoke: the usage cards and
    rotation. Lifted out of ``create_app`` for the ruff complexity
    ceiling."""

    # The key-mutation dedupe namespace is per (credential, key, verb,
    # target): the same header key can pin a rotate on K1 and a patch on
    # K2 without colliding, while a keyed retry of the same verb+target
    # replays the recorded answer.
    async def _key_rotate_idem_claim(
        key_id: str,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> AsyncIterator[None]:
        skey = _idem_scope(_idem_key(idempotency_key), namespace=f"rotate:{key_id}")
        async with key_idem_store.async_claim_lock(skey):
            yield

    async def _key_patch_idem_claim(
        key_id: str,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> AsyncIterator[None]:
        skey = _idem_scope(_idem_key(idempotency_key), namespace=f"patch:{key_id}")
        async with key_idem_store.async_claim_lock(skey):
            yield

    async def _key_revoke_idem_claim(
        key_id: str,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> AsyncIterator[None]:
        skey = _idem_scope(_idem_key(idempotency_key), namespace=f"revoke:{key_id}")
        async with key_idem_store.async_claim_lock(skey):
            yield

    @app.post(
        "/harness/keys/{key_id}/rotate",
        response_model=ApiKeyRotateResponse,
        status_code=201,
        tags=["ops"],
        operation_id="key_rotate",
    )
    def key_rotate(
        key_id: str,
        body: ApiKeyRotateRequest,
        request: Request,
        _idem_claim_held: None = Depends(_key_rotate_idem_claim),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> ApiKeyRotateResponse | JSONResponse:
        """Atomic rotation: mint a successor inheriting the predecessor's
        declared policy (name/scopes/admin/rpm/budgets) and, by default,
        tombstone the predecessor in the same store transaction. The new
        raw secret is returned once; lineage (``rotated_from``) is
        journaled with the successor record. Without ``ttl_s`` the
        successor inherits the predecessor's absolute ``expires_at`` —
        rotation never extends a credential's lifetime. A keyed retry
        replays the recorded successor (same id, same raw secret)
        instead of minting a third credential."""
        _require_admin(request)
        body_fp = _body_fp(body)
        key, replay = _idem_lookup(
            idempotency_key, key_idem_store, body_fp, namespace=f"rotate:{key_id}"
        )
        if replay is not None:
            return JSONResponse(
                replay.envelope,
                status_code=201,
                headers={"X-Fx1-Idempotent-Replay": "true"},
            )
        _drain_refusal(metrics)
        try:
            raw, rec = key_store.rotate(
                key_id,
                revoke_old=body.revoke_old,
                name=body.name,
                ttl_s=body.ttl_s,
            )
        except KeyStoreError as exc:
            raise ApiError(
                404 if exc.code == "key_not_found" else 409,
                str(exc),
                code=exc.code,
            ) from exc
        except ValueError as exc:
            raise ApiError(422, str(exc), code="invalid_rotation") from exc
        resp = ApiKeyRotateResponse(
            key=_key_mint_wire(rec, raw),
            rotated_from=key_id,
            revoked_previous=body.revoke_old,
        )
        if key is not None:
            key_idem_store.put(key, body_fp, _JsonIdemRecord(envelope=resp.model_dump(mode="json")))
        return resp

    @app.patch(
        "/harness/keys/{key_id}",
        response_model=ApiKeyRecordModel,
        tags=["ops"],
        operation_id="key_patch",
    )
    def key_patch(
        key_id: str,
        body: ApiKeyPatchRequest,
        request: Request,
        _idem_claim_held: None = Depends(_key_patch_idem_claim),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> ApiKeyRecordModel | JSONResponse:
        """Mutable policy update on a live managed key — the patched
        record returns, shaped like ``key_get``. Omitted fields keep
        the declared policy; explicit ``null`` clears a nullable
        bound (``name``/``rpm``/``max_requests``/``max_tokens``/
        ``expires_at``); ``admin:true`` unions the admin scope the
        mint way while ``admin:false`` never strips a declared scope.
        Patching is in place — no new secret, no slot consumed — and
        the updated record journals so a ``--state-dir`` restart
        restores it. ``enabled``/live counters stay unpatchable:
        revocation is permanent (rotate covers re-keying). A keyed
        retry replays the recorded patch outcome."""
        _require_admin(request)
        body_fp = _body_fp(body)
        key, replay = _idem_lookup(
            idempotency_key, key_idem_store, body_fp, namespace=f"patch:{key_id}"
        )
        if replay is not None:
            return JSONResponse(replay.envelope, headers={"X-Fx1-Idempotent-Replay": "true"})
        _drain_refusal(metrics)
        sent = body.model_fields_set
        clear = {f for f in CLEARABLE_KEY_FIELDS if f in sent and getattr(body, f) is None}
        try:
            rec = key_store.update(
                key_id,
                name=body.name,
                rpm=body.rpm,
                scopes=body.scopes,
                admin=body.admin,
                max_requests=body.max_requests,
                max_tokens=body.max_tokens,
                expires_at=body.expires_at,
                clear=clear,
            )
        except KeyStoreError as exc:
            status = {"key_not_found": 404, "key_revoked": 409}.get(exc.code, 422)
            raise ApiError(status, str(exc), code=exc.code) from exc
        except ValueError as exc:
            raise ApiError(422, str(exc), code="invalid_patch") from exc
        resp = _key_wire(rec)
        if key is not None:
            key_idem_store.put(key, body_fp, _JsonIdemRecord(envelope=resp.model_dump(mode="json")))
        return resp

    @app.delete(
        "/harness/keys/{key_id}",
        response_model=ApiKeyRecordModel,
        tags=["ops"],
        operation_id="key_revoke",
    )
    def key_revoke(
        key_id: str,
        request: Request,
        _idem_claim_held: None = Depends(_key_revoke_idem_claim),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> ApiKeyRecordModel | JSONResponse:
        """Tombstone a key — ``enabled=false`` + ``revoked_at``. The record
        stays so the audit trail of which keys existed survives; auth
        with it fails closed immediately after. A keyed retry replays
        the recorded tombstone instead of 409ing on the second revoke."""
        _require_admin(request)
        key, replay = _idem_lookup(
            idempotency_key, key_idem_store, "revoke", namespace=f"revoke:{key_id}"
        )
        if replay is not None:
            return JSONResponse(replay.envelope, headers={"X-Fx1-Idempotent-Replay": "true"})
        # Revocation removes authority and must remain available during drain.
        try:
            rec = key_store.revoke(key_id)
        except KeyStoreError as exc:
            status = 404 if exc.code == "key_not_found" else 409
            raise ApiError(status, str(exc), code=exc.code) from exc
        resp = _key_wire(rec)
        if key is not None:
            key_idem_store.put(
                key, "revoke", _JsonIdemRecord(envelope=resp.model_dump(mode="json"))
            )
        return resp

    @app.get(
        "/harness/keys/{key_id}/usage",
        response_model=ApiKeyUsageResponse,
        tags=["ops"],
        operation_id="key_usage",
    )
    def key_usage(key_id: str, request: Request) -> ApiKeyUsageResponse:
        """One key's usage card — lifetime counters, declared budgets with
        derived headroom, RPM window state, and completion-ring spend.
        Journaled ``uses`` and ``tokens_used`` survive a clean ``--state-dir``
        restart; the RPM window remains process-local and resets."""
        _require_admin(request)
        rec = key_store.get(key_id)
        if rec is None:
            raise ApiError(404, f"unknown key {key_id!r}", code="key_not_found")
        return _key_usage_response(rec, key_store, completion_log)

    @app.get(
        "/harness/self",
        response_model=SelfUsageResponse,
        tags=["ops"],
        operation_id="self_usage",
    )
    def self_usage(request: Request) -> SelfUsageResponse:
        """The calling credential's own card — ``read`` scope, so any
        managed key watches its own budgets without admin. The env key
        and loopback dev callers are the unmetered roots."""
        key_id = getattr(request.state, "key_id", None)
        if key_id in (None, "env"):
            return SelfUsageResponse(
                credential="env" if key_id == "env" else "none",
                scopes=list(SCOPES),
                metered=False,
                key=None,
            )
        rec = key_store.get(key_id)
        if rec is None:
            raise ApiError(404, f"unknown key {key_id!r}", code="key_not_found")
        return SelfUsageResponse(
            credential="managed",
            scopes=list(rec.get("scopes") or []),
            metered=True,
            key=_key_usage_response(rec, key_store, completion_log),
        )


def _required_scope(method: str, path: str) -> str:
    """The scope a request needs: the control plane (key management,
    drain) is ``admin`` on any method, safe methods are ``read``,
    everything else is ``write``."""
    if path == "/harness/keys" or path.startswith("/harness/keys/") or path == "/harness/drain":
        return "admin"
    if method in ("GET", "HEAD", "OPTIONS"):
        return "read"
    return "write"


def _insufficient_scope(request: Request, scopes: frozenset[str] | None) -> JSONResponse | None:
    """403 in the path's own error grammar when the resolved credential's
    declared scopes don't cover this request — the 401 got the caller
    authenticated; this is the authorization refusal. ``None`` scopes
    (env key, loopback dev, public paths) are unrestricted."""
    if scopes is None:
        return None
    required = _required_scope(request.method, request.url.path)
    if required in scopes:
        return None
    msg = f"key lacks required scope {required!r}"
    body: dict[str, Any] = {"detail": msg, "code": "insufficient_scope"}
    if is_openai_path(request.url.path):
        body = _v1_error_body(request.url.path, msg, 403, "insufficient_scope")
    return JSONResponse(status_code=403, content=body)


def _resolve_auth(
    request: Request,
    api_key: str | None,
    key_store: ApiKeyStore,
) -> tuple[str | None, bool, frozenset[str] | None] | JSONResponse:
    """Resolve the request's credential → ``(key_id, admin, scopes)``, or
    the refusal response. ``scopes`` is the minted key's declared set;
    ``None`` marks unrestricted credentials (env key, loopback dev,
    public paths).

    - Public paths: ``(None, False, None)`` — no auth consumed.
    - Auth enabled (env key set, or any managed key exists): the env key
      resolves as ``("env", True, None)``; a managed key resolves to
      ``(key_id, admin_flag, frozenset(scopes))`` — minted ``admin`` keys
      can manage keys themselves; anything else 401s.
    - Auth disabled: loopback resolves ``(None, True, None)`` — the dev
      surface is trusted and bootstraps key provisioning; non-loopback
      403s.
    """
    if request.url.path in _PUBLIC_PATHS:
        return (None, False, None)
    if api_key or key_store.has_keys:
        provided = request.headers.get("X-API-Key")
        # OpenAI-shape clients authenticate with Authorization: Bearer
        # — accept it on /v1 so stock SDKs work unmodified.
        if not provided and is_openai_path(request.url.path):
            auth_hdr = request.headers.get("Authorization", "")
            if auth_hdr.startswith("Bearer "):
                provided = auth_hdr[len("Bearer ") :]
        # compare_digest refuses non-ASCII str; the utf-8 encodings keep
        # ordinary header values byte-exact. ``os.environ`` can contain
        # surrogate-escaped bytes on POSIX, so surrogatepass is required
        # on both sides as well: a malformed configured key must not turn
        # an otherwise ordinary bad credential into a server fault.
        if (
            provided
            and api_key
            and hmac.compare_digest(
                provided.encode("utf-8", errors="surrogatepass"),
                api_key.encode("utf-8", errors="surrogatepass"),
            )
        ):
            return ("env", True, None)
        if provided:
            key_rec = key_store.authenticate(
                provided,
                required_scope=_required_scope(request.method, request.url.path),
            )
            if key_rec is not None:
                scopes = key_rec.get("scopes")
                scope_set = frozenset(scopes) if isinstance(scopes, list) else None
                scope_refusal = _insufficient_scope(request, scope_set)
                if scope_refusal is not None:
                    return scope_refusal
                return (
                    key_rec["key_id"],
                    bool(key_rec.get("admin")),
                    scope_set,
                )
        content: dict[str, Any] = {
            "detail": "invalid or missing X-API-Key",
            "code": "unauthorized",
        }
        if is_openai_path(request.url.path):
            content = _v1_error_body(
                request.url.path, "invalid or missing API key", 401, "unauthorized"
            )
        return JSONResponse(status_code=401, content=content)
    host = (request.client.host if request.client else "") or ""
    if host not in _LOOPBACK_HOSTS:
        forbidden_body: dict[str, Any] = {
            "detail": (
                "FX1_API_KEY is unset; non-localhost clients are "
                "refused. Set FX1_API_KEY and send X-API-Key, or "
                "bind to 127.0.0.1 only."
            ),
            "code": "forbidden",
        }
        if is_openai_path(request.url.path):
            forbidden_body = _v1_error_body(
                request.url.path, str(forbidden_body["detail"]), 403, "forbidden"
            )
        return JSONResponse(status_code=403, content=forbidden_body)
    return (None, True, None)


def create_app(
    harness: Harness | None = None,
    backend_resolver: Any | None = None,
    max_inflight: int | None = None,
    sse_keepalive_s: float | None = None,
    idem_max: int | None = None,
    job_max: int | None = None,
    rate_limit_rps: float | None = None,
    gzip_min_bytes: int | None = None,
    cors_origins: str | None = None,
    breaker_threshold: int | None = None,
    breaker_cooldown_s: float | None = None,
    receipts_dir: str | os.PathLike[str] | None = None,
    byok_override: bool | None = None,
    file_max: int | None = None,
    file_bytes_max: int | None = None,
    batch_max: int | None = None,
    batch_line_max: int | None = None,
    store_max: int | None = None,
    ft_runner: FTJobRunner | None = None,
    ft_dir: str | os.PathLike[str] | None = None,
    state_dir: str | os.PathLike[str] | None = None,
) -> FastAPI:
    api_key = os.environ.get(_API_KEY_ENV) or None
    lab = harness or Harness()
    resolve_backend = backend_resolver or get_backend
    max_inflight = _env_int_bound(_MAX_INFLIGHT_ENV, 16, max_inflight)
    idem_max = _env_int_bound(_IDEM_MAX_ENV, 1024, idem_max)
    job_max = _env_int_bound(_JOB_MAX_ENV, 1024, job_max)
    sse_keepalive_s = _env_float_floor(_SSE_KEEPALIVE_ENV, 15.0, sse_keepalive_s)
    rate_limit_rps = _env_float_floor(_RATE_LIMIT_ENV, 0.0, rate_limit_rps)
    byok_override_enabled = (
        os.environ.get(_BYOK_OVERRIDE_ENV, "1").strip().lower() in ("1", "true", "yes", "on")
        if byok_override is None
        else byok_override
    )
    gzip_min_bytes = _env_int_floor(_GZIP_MIN_ENV, 1024, gzip_min_bytes)
    breaker_threshold = _env_int_floor(_BREAKER_THRESHOLD_ENV, 5, breaker_threshold)
    breaker_cooldown_s = _env_float_floor(_BREAKER_COOLDOWN_ENV, 30.0, breaker_cooldown_s)
    cors_raw = cors_origins if cors_origins is not None else os.environ.get(_CORS_ORIGINS_ENV, "")
    cors_list = [o.strip() for o in cors_raw.split(",") if o.strip()]
    for origin in cors_list:
        parsed = urllib.parse.urlparse(origin)
        if origin == "*" or parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                f"invalid CORS origin {origin!r} — expected explicit scheme://host, "
                "never the wildcard"
            )
    limiter = _RateLimiter(rate_limit_rps) if rate_limit_rps > 0 else None
    breaker = (
        _BackendBreaker(breaker_threshold, breaker_cooldown_s) if breaker_threshold > 0 else None
    )
    receipts_root = Path(
        receipts_dir if receipts_dir is not None else os.environ.get(_RECEIPTS_DIR_ENV, "receipts")
    )
    receipt_index = _ReceiptIndex(receipts_root)
    # Bounded in-flight work: the harness executes lab commands and model
    # calls on shared resources (a spawned local engine, GPU memory, the
    # box itself) — saturation must fail honestly as 503, never queue
    # unboundedly or crash mid-request. Cheap routes (commands, verify,
    # health) stay uncapped so liveness answers under load.
    inflight = threading.BoundedSemaphore(max_inflight)
    metrics = _Metrics(max_inflight)
    # Last deep-health verdict per backend — process-local so monitoring
    # scrapes read it off ``GET /harness/backends`` without re-probing.
    probe_cache: dict[str, BackendProbeVerdict] = {}
    probe_lock = threading.Lock()
    state_dir = state_dir or os.environ.get(_STATE_DIR_ENV) or None
    state_path = Path(state_dir) if state_dir is not None else None

    def _journal(name: str) -> JobJournal | None:
        return JobJournal(state_path / name) if state_path is not None else None

    idem_store: _IdemStore[HarnessRunResponse] = _IdemStore(
        idem_max, journal=_journal("idem_runs.jsonl"), model=HarnessRunResponse
    )
    complete_idem_store: _IdemStore[CompleteResponse] = _IdemStore(
        idem_max, journal=_journal("idem_complete.jsonl"), model=CompleteResponse
    )
    complete_batch_idem_store: _IdemStore[CompleteBatchResponse] = _IdemStore(
        idem_max,
        journal=_journal("idem_complete_batch.jsonl"),
        model=CompleteBatchResponse,
    )
    openai_idem_store: _IdemStore[_OpenAIIdemRecord] = _IdemStore(
        idem_max, journal=_journal("idem_openai.jsonl"), model=_OpenAIIdemRecord
    )
    # /v1/messages pins the shared chat.completion envelope — replay
    # regenerates the Anthropic surface deterministically from it.
    anthropic_idem_store: _IdemStore[_OpenAIIdemRecord] = _IdemStore(
        idem_max, journal=_journal("idem_anthropic.jsonl"), model=_OpenAIIdemRecord
    )
    # /v1/completions pins the already-mapped text_completion envelope —
    # a legacy replay regenerates the legacy SSE grammar from it.
    legacy_idem_store: _IdemStore[_OpenAIIdemRecord] = _IdemStore(
        idem_max, journal=_journal("idem_legacy.jsonl"), model=_OpenAIIdemRecord
    )
    # /v1/files + /v1/uploads pin the minted object envelope — journaled
    # like the sibling replay ledgers (no secrets ride along).
    upload_idem_store: _IdemStore[_JsonIdemRecord] = _IdemStore(
        idem_max, journal=_journal("idem_uploads.jsonl"), model=_JsonIdemRecord
    )
    # Key-mint/rotate/patch/revoke replays stay process-local: the
    # recorded mint answer carries the raw credential, and the journal
    # contract refuses persisted secrets (same rule as callback
    # secrets). A post-restart retry re-executes honestly instead of
    # replaying an unjournaled record.
    key_idem_store: _IdemStore[_JsonIdemRecord] = _IdemStore(idem_max)
    file_max = _env_int_bound(_FILE_MAX_ENV, 128, file_max)
    file_bytes_max = _env_int_bound(_FILE_BYTES_ENV, 8 << 20, file_bytes_max)
    batch_max = _env_int_bound(_BATCH_MAX_ENV, 256, batch_max)
    batch_line_max = _env_int_bound(_BATCH_LINES_ENV, 1024, batch_line_max)
    store_max = _env_int_bound(_STORE_MAX_ENV, 256, store_max)
    job_store = _JobStore(job_max, journal=_journal("jobs.jsonl"))
    eval_store = EvalStore(job_max, journal=_journal("evals.jsonl"))
    eval_spec_store = EvalSpecStore(job_max, journal=_journal("eval_specs.jsonl"))
    file_store = _FileStore(file_max, file_bytes_max, state_dir=state_path)
    upload_store = UploadStore(file_max, file_bytes_max, state_dir=state_path)
    key_store = ApiKeyStore(
        journal=JobJournal(state_path / "keys.jsonl") if state_path is not None else None
    )
    completion_log = _CompletionLog(on_record=lambda rec: _charge_key_tokens(rec, key_store))
    batch_store = _BatchStore(batch_max, journal=_journal("batches.jsonl"))
    abatch_store = _AnthropicBatchStore(batch_max, journal=_journal("abatches.jsonl"))
    # The OpenAI-shaped fine-tuning surface: bounded like the other job
    # stores; the runner defaults to the real staged Pipeline over the
    # in-repo tiny-LM trainer. Job work dirs — and the checkpoints they
    # mint — default under --state-dir so a completed job's ft: model
    # still resolves and serves after a restart.
    ft_store = FTJobStore(job_max, journal=_journal("ft_jobs.jsonl"))
    ft_work_root = Path(
        ft_dir
        if ft_dir is not None
        else os.environ.get(
            "FX1_FT_DIR",
            str(
                state_path / "ft"
                if state_path is not None
                else Path(tempfile.gettempdir()) / "fx1_ft"
            ),
        )
    )
    ft_runner_eff = ft_runner or default_ft_runner(resolve_backend)
    # The /v1 retrieval index behind GET/DELETE /v1/chat/completions/{id}
    # and /v1/responses/{id} — `store=false` keeps a call out of it.
    envelope_store = OpenAIEnvelopeStore(store_max)
    # Named conversation containers — ``POST /v1/conversations`` mints
    # them, ``conversation`` on a response joins one. Same bounded LRU
    # contract as the envelope store; items live in subitems. Journaled
    # under state_dir like the other stateful surfaces — a named
    # container whose contents die on restart is a stateful surface that
    # isn't; the retrieval index above stays a documented in-memory
    # fetch cache.
    conv_store = OpenAIEnvelopeStore(store_max, journal=_journal("conversations.jsonl"))
    # Vector stores borrow file content through the reader closure — a
    # deleted/oversized file fails the attach honestly rather than
    # silently indexing nothing; journaled under state_dir like the
    # other stores so a restart restores the corpus.

    def _vs_file_reader(file_id: str) -> tuple[bytes, str] | None:
        rec = file_store.get(file_id)
        if rec is None:
            return None
        return bytes(rec.content), rec.filename

    vs_store = VectorStoreStore(
        store_max,
        state_dir=state_path,
        file_reader=_vs_file_reader,
        idem_max=idem_max,
    )
    # Cancel flags for background responses — a set event means the stored
    # envelope was flipped to ``cancelled`` and the worker must not
    # overwrite it with a terminal result.
    _bg_cancel: dict[str, threading.Event] = {}
    jobs_executor = ThreadPoolExecutor(max_workers=max_inflight, thread_name_prefix="fx1-job")

    @contextmanager
    def _work_gate() -> Iterator[None]:
        _drain_refusal(metrics)
        if not inflight.acquire(blocking=False):
            raise ApiError(
                503,
                f"harness at max_inflight={max_inflight} — retry later",
                code="over_capacity",
                headers={"Retry-After": "1"},
            )
        metrics.acquire()
        try:
            yield
        finally:
            metrics.release()
            inflight.release()

    def _slot(request: Request) -> Iterator[None]:
        # A stored replay/conflict performs no work and must stay readable
        # while draining or saturated.  The route's claim dependency runs
        # first, so an entry found here cannot disappear before the handler
        # validates its fingerprint.  A conflicting body also bypasses the
        # work gate only to fail closed 409 in the handler.
        replay_stores: dict[str, _IdemStore[Any]] = {
            "/harness/complete": complete_idem_store,
            "/harness/complete/batch": complete_batch_idem_store,
            "/v1/chat/completions": openai_idem_store,
            "/v1/responses": openai_idem_store,
            "/v1/messages": anthropic_idem_store,
            "/v1/completions": legacy_idem_store,
        }
        yield from _replay_aware_slot(request, replay_stores, _work_gate)

    app = FastAPI(
        title="fx-1 harness API",
        version=__version__,
        description=(
            "Execution surface for the fx-1 harness: the registered lab "
            "commands, sealed-receipt verification, and gated model "
            "completion over hosted_k3 / local_fx1 / BYOK backends."
        ),
        lifespan=_make_lifespan(metrics, job_store, eval_store, jobs_executor, ft_store),
        openapi_tags=[
            {"name": "runs", "description": "Synchronous lab-command execution."},
            {"name": "jobs", "description": "Async run jobs: submit, poll, SSE, cancel, batch."},
            {
                "name": "complete",
                "description": "Gated model completion (sync, batch, SSE stream).",
            },
            {
                "name": "evals",
                "description": "Async eval-suite submissions against any backend.",
            },
            {"name": "receipts", "description": "Sealed-receipt verification."},
            {"name": "ops", "description": "Liveness, readiness, metrics, drain, version."},
        ],
    )
    # Large responses (job listings, receipt payloads, openapi) compress well;
    # urllib-based clients send no Accept-Encoding so SSE stays uncompressed.
    if gzip_min_bytes > 0:
        app.add_middleware(GZipMiddleware, minimum_size=gzip_min_bytes)
    app.state.cors_origins = cors_list
    app.state.gzip_min_bytes = gzip_min_bytes
    app.state.inflight_slots = inflight
    app.state.metrics = metrics
    app.state.idem_store = idem_store
    app.state.job_store = job_store
    app.state.eval_store = eval_store
    app.state.eval_spec_store = eval_spec_store
    app.state.file_store = file_store
    app.state.batch_store = batch_store
    app.state.abatch_store = abatch_store
    app.state.ft_store = ft_store
    app.state.vs_store = vs_store
    app.state.jobs_executor = jobs_executor
    app.state.sse_keepalive_s = sse_keepalive_s
    app.state.rate_limiter = limiter
    app.state.breaker = breaker

    # Routing errors raise the Starlette base; FastAPI and ApiError subclasses
    # still resolve to this handler through their exception hierarchy.
    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if is_openai_path(request.url.path):
            return JSONResponse(
                status_code=exc.status_code,
                content=_v1_error_body(
                    request.url.path, str(exc.detail), exc.status_code, _err_code(exc)
                ),
                headers=exc.headers,
            )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "code": _err_code(exc)},
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _validation_response(request, exc)

    @app.middleware("http")
    async def harness_api_auth(request: Request, call_next: Any) -> Any:
        request_id = _request_id(request.headers.get("x-request-id"))
        request.state.request_id = request_id
        started = time.monotonic()
        rl_headers: dict[str, str] | None = None
        # Public paths are exempt — a load balancer's /health probe cadence
        # must never consume the client's own request budget (or 429 liveness).
        if limiter is not None and request.url.path not in _PUBLIC_PATHS:
            # keyed on client host so a rotating fake API key can't evade it
            host = (request.client.host if request.client else "") or "unknown"
            wait, remaining = limiter.allow(host)
            reset = max(0, math.ceil((limiter.capacity - remaining) / limiter.rps))
            rl_headers = {
                "X-RateLimit-Limit": str(int(limiter.rps)),
                "X-RateLimit-Remaining": str(int(remaining)),
                "X-RateLimit-Reset": str(reset),
            }
            if wait > 0:
                metrics.record_rate_limited()
                rl_msg = f"rate limit exceeded; retry in {wait:.1f}s"
                rl_content: dict[str, Any] = {
                    "detail": rl_msg,
                    "code": "too_many_requests",
                }
                if is_openai_path(request.url.path):
                    rl_content = _v1_error_body(request.url.path, rl_msg, 429, "too_many_requests")
                response = JSONResponse(
                    status_code=429,
                    content=rl_content,
                    headers={"Retry-After": str(max(1, math.ceil(wait))), **rl_headers},
                )
                return _finish(request, request_id, response, started)
        ingress_refusal = await _request_ingress_refusal(request)
        if ingress_refusal is not None:
            return _finish(request, request_id, ingress_refusal, started)
        # Auth surface: ``FX1_API_KEY`` is the root credential (admin);
        # managed keys from ``key_store`` additionally authenticate.
        # ``request.state.admin`` gates the key-management routes — env
        # key, or loopback dev mode (no env key and an empty store).
        request.state.key_id = None
        request.state.admin = False
        try:
            auth = _resolve_auth(request, api_key, key_store)
        except KeyStoreError as exc:
            # a managed key past its declared rpm/budget refuses 429 —
            # same fail-closed shape as the global limiter, keyed to the
            # credential's own window/budget; a scope denial is 403, not
            # rate limiting
            if exc.code != "insufficient_scope":
                metrics.record_rate_limited()
            refused = _key_refusal_response(exc, request, key_store)
            if rl_headers is not None:
                # the refusal consumed a global-bucket slot — report it;
                # the key's own *-Requests family is set by the refusal
                refused.headers.update(rl_headers)
            return _finish(request, request_id, refused, started)
        if isinstance(auth, JSONResponse):
            response = auth
        else:
            key_id, admin, _scopes = auth
            request.state.key_id = key_id
            request.state.admin = admin
            if key_id is not None:
                ctx_token = _REQUEST_KEY_ID.set(key_id)
                try:
                    response = await call_next(request)
                finally:
                    _REQUEST_KEY_ID.reset(ctx_token)
            else:
                response = await call_next(request)
            # a managed key with a declared rpm window reports its
            # standing budget on every answer (OpenAI header names) —
            # env-key and loopback auth declare no window and get none
            response.headers.update(_key_budget_headers(key_store, key_id))
            if _is_anthropic_surface(request):
                response.headers.update(_anthropic_budget_headers(key_store, key_id))
        if rl_headers is not None:
            response.headers.update(rl_headers)
        return _finish(request, request_id, response, started)

    @app.get("/metrics", response_model=MetricsResponse, tags=["ops"], operation_id="get_metrics")
    def metrics_route(
        request: Request,
        format: Literal["json", "prom", "prometheus"] | None = None,
    ) -> MetricsResponse | PlainTextResponse:
        """Ops snapshot — JSON by default; Prometheus text exposition via
        ``?format=prom`` or an ``Accept: text/plain``/OpenMetrics header."""
        accept = request.headers.get("accept", "")
        wants_text = format in ("prom", "prometheus") or (
            format is None and ("text/plain" in accept or "openmetrics" in accept)
        )
        if wants_text:
            snap = metrics.snapshot()
            counts: dict[str, int] = {}
            for job in job_store.list():
                counts[job.status] = counts.get(job.status, 0) + 1
            return PlainTextResponse(
                _render_prometheus(snap, counts),
                media_type="text/plain; version=0.0.4",
            )
        return metrics.snapshot()

    @app.get("/health", tags=["ops"], operation_id="health")
    def health() -> HealthResponse:
        return HealthResponse(
            registered_commands=len(lab.list_commands()),
            backends=_backend_configured(),
            draining=metrics.draining.is_set(),
        )

    @app.get("/ready", tags=["ops"], operation_id="ready")
    def ready() -> ReadyResponse:
        """Kubernetes-style readiness: 200 while accepting work, 503 once
        drain is latched — the load balancer's signal to deregister the
        pod before gated routes start refusing."""
        if metrics.draining.is_set():
            raise ApiError(503, "harness is draining", code="draining")
        return ReadyResponse(inflight=metrics.snapshot().inflight)

    app.add_api_route(
        "/harness/version",
        _version_info,
        methods=["GET"],
        response_model=VersionResponse,
        tags=["ops"],
        operation_id="get_version",
    )

    @app.get(
        "/harness/capabilities",
        tags=["ops"],
        operation_id="get_capabilities",
    )
    def capabilities() -> CapabilitiesResponse:
        """Discovery: the wire features this build serves and the
        operational limits actually in effect. A client learns batch
        caps, store bounds, rate limits, and feature flags (idempotency,
        SSE, webhooks, drain) from one call — nothing is hardcoded."""
        return CapabilitiesResponse(
            api_version=API_VERSION,
            fx1_version=__version__,
            features={
                "idempotency": True,
                "sse": True,
                "webhooks": True,
                "batch": True,
                "jobs": True,
                "drain": True,
                "streaming": True,
                "cors": bool(cors_list),
                "breaker": breaker is not None,
                "receipts_store": receipt_index.available(),
                "byok_override": byok_override_enabled,
                "openai_compat": True,
                "openai_retrieval": True,
                "openai_responses_replay": True,
                "openai_tools": True,
                "openai_responses_tools": True,
                "openai_logprobs": True,
                "openai_embeddings": True,
                "openai_moderations": True,
                "openai_vector_stores": True,
                "openai_file_search": True,
                "anthropic_messages": True,
                "legacy_completions": True,
                "anthropic_message_batches": True,
                "anthropic_count_tokens": True,
                "anthropic_models": True,
                "key_scopes": True,
                "key_quotas": True,
                "key_usage": True,
                "key_rotation": True,
                "key_patch": True,
                "score": True,
                "evals": True,
                "eval_diff": True,
                "fine_tuning": True,
            },
            eval_suites=list(EVAL_SUITES),
            limits={
                "max_inflight": float(metrics.max_inflight),
                "job_max": float(job_store._max),
                "eval_max": float(eval_store.capacity),
                "idem_max": float(idem_store._max),
                "file_max": float(file_max),
                "file_bytes_max": float(file_bytes_max),
                "batch_max": float(batch_max),
                "batch_line_max": float(batch_line_max),
                "store_max": float(store_max),
                "vs_store_max": float(vs_store.max_stores),
                "vs_file_max": float(vs_store.max_files),
                "vs_max_results": float(VS_MAX_RESULTS),
                "job_batch_max": float(_JOB_BATCH_MAX),
                "verify_batch_max": float(_VERIFY_BATCH_MAX),
                "complete_batch_max": 64.0,
                "body_max_bytes": float(_MAX_BODY_BYTES),
                "job_result_max_bytes": float(_JOB_RESULT_MAX_BYTES),
                "rate_limit_rps": limiter.rps if limiter is not None else 0.0,
                "breaker_threshold": float(breaker_threshold),
                "breaker_cooldown_s": float(breaker_cooldown_s),
                "sse_keepalive_s": float(sse_keepalive_s),
                "gzip_min_bytes": float(gzip_min_bytes),
            },
            backends=_backend_configured(),
            roles=sorted({str(c.role) for c in lab.list_commands()}),
        )

    @app.get(
        "/harness/backends",
        response_model=dict[str, BackendStatusEntry],
        tags=["ops"],
        operation_id="backends_status",
    )
    def backends_status() -> dict[str, BackendStatusEntry]:
        """Per-backend liveness: configured flag plus circuit state —
        whether the breaker is fast-failing this backend, how much
        cooldown remains, and the consecutive-fault streak."""
        configured = _backend_configured()
        out: dict[str, BackendStatusEntry] = {}
        for name, cfg in configured.items():
            circuit_open, remaining, fails = (
                breaker.state(name) if breaker is not None else (False, 0.0, 0)
            )
            with probe_lock:
                last_probe = probe_cache.get(name)
            out[name] = BackendStatusEntry(
                configured=cfg,
                circuit_open=circuit_open,
                cooldown_remaining_s=round(remaining, 3),
                consecutive_failures=fails,
                last_probe=last_probe,
            )
        return out

    @app.get(
        "/harness/completions",
        response_model=CompletionListResponse,
        tags=["ops"],
        operation_id="completions_list",
    )
    def completions_list(
        limit: int = Query(default=50, ge=1, le=_COMPLETION_LOG_MAX),
        backend: Literal["hosted_k3", "local_fx1", "byok"] | None = None,
    ) -> CompletionListResponse:
        """Newest-first window on the completion log — per-call evidence
        (hashes, usage, verdict) for every gated model call the process
        has served, bounded by the ring."""
        items = completion_log.latest(limit, backend)
        return CompletionListResponse(items=items, count=len(items))

    @app.get(
        "/harness/completions/{completion_id}",
        response_model=CompletionRecord,
        tags=["ops"],
        operation_id="completion_get",
    )
    def completion_get(completion_id: str) -> CompletionRecord:
        rec = completion_log.get(completion_id)
        if rec is None:
            raise ApiError(404, f"completion {completion_id!r} not in the log", code="not_found")
        return rec

    @app.get(
        "/harness/completions/{completion_id}/receipt",
        tags=["ops"],
        operation_id="completion_receipt",
    )
    def completion_receipt(completion_id: str) -> dict[str, Any]:
        """Export one logged call as a sealed ``fx1_completion_record.v1``
        document — verify with ``POST /receipts/verify`` or the SDK."""
        from fx1.serve.ops_receipt import (  # noqa: PLC0415
            completion_record_receipt,
        )

        rec = completion_log.get(completion_id)
        if rec is None:
            raise ApiError(404, f"completion {completion_id!r} not in the log", code="not_found")
        return completion_record_receipt(rec.model_dump(mode="json"))

    @app.get(
        "/harness/usage",
        response_model=UsageReport,
        tags=["ops"],
        operation_id="usage_report",
    )
    def usage_report(
        backend: Literal["hosted_k3", "local_fx1", "byok"] | None = None,
        model: str | None = Query(default=None, max_length=256),
        since: float | None = Query(default=None, ge=0.0),
        until: float | None = Query(default=None, ge=0.0),
        key_id: str | None = Query(default=None, max_length=64),
    ) -> UsageReport:
        """Token/request accounting over the retained completion records —
        the billing/ops view. Totals plus per-backend/per-model splits;
        `records_dropped`/`ring_cap` declare a truncated window, and the
        model split keys absent models as ``(none)``. `since`/`until`
        are unix-second bounds on the record timestamps; since>until is
        a fail-closed 400."""
        if since is not None and until is not None and since > until:
            raise ApiError(400, "since must be <= until", code="bad_window")
        return aggregate_usage(
            completion_log.all(backend),
            cap=completion_log.cap,
            dropped=completion_log.dropped,
            backend=backend,
            model=model,
            key_id=key_id,
            since=since,
            until=until,
        )

    async def _key_mint_idem_claim(
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> AsyncIterator[None]:
        """Serialize first use of one mint key — a keyed retry replays the
        recorded credential (same id, same raw secret) instead of minting
        a second key."""
        skey = _idem_scope(_idem_key(idempotency_key), namespace="mint")
        async with key_idem_store.async_claim_lock(skey):
            yield

    @app.post(
        "/harness/keys",
        response_model=ApiKeyMintResponse,
        status_code=201,
        tags=["ops"],
        operation_id="key_create",
    )
    def key_create(
        body: ApiKeyCreateRequest,
        request: Request,
        _idem_claim_held: None = Depends(_key_mint_idem_claim),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> ApiKeyMintResponse | JSONResponse:
        """Mint a managed API key. The raw ``key`` is returned once here
        and never stored — the store keeps only its sha256. Requires the
        bootstrap credential (``FX1_API_KEY``) or loopback dev mode.
        ``Idempotency-Key`` pins the mint: a keyed retry replays the
        recorded credential verbatim instead of minting a duplicate."""
        _require_admin(request)
        body_fp = _body_fp(body)
        key, replay = _idem_lookup(idempotency_key, key_idem_store, body_fp, namespace="mint")
        if replay is not None:
            return JSONResponse(
                replay.envelope,
                status_code=201,
                headers={"X-Fx1-Idempotent-Replay": "true"},
            )
        _drain_refusal(metrics)
        try:
            raw, rec = key_store.mint(
                body.name,
                admin=body.admin,
                rpm=body.rpm,
                ttl_s=body.ttl_s,
                scopes=body.scopes,
                max_requests=body.max_requests,
                max_tokens=body.max_tokens,
            )
        except KeyStoreError as exc:
            raise ApiError(400, str(exc), code=exc.code) from exc
        resp = _key_mint_wire(rec, raw)
        if key is not None:
            key_idem_store.put(key, body_fp, _JsonIdemRecord(envelope=resp.model_dump(mode="json")))
        return resp

    @app.get(
        "/harness/keys",
        response_model=ApiKeyListResponse,
        tags=["ops"],
        operation_id="key_list",
    )
    def key_list(request: Request) -> ApiKeyListResponse:
        """Every minted key's public fingerprint + metadata — never the
        secret or its hash."""
        _require_admin(request)
        return ApiKeyListResponse(data=[_key_wire(r) for r in key_store.list()])

    @app.get(
        "/harness/keys/{key_id}",
        response_model=ApiKeyRecordModel,
        tags=["ops"],
        operation_id="key_get",
    )
    def key_get(key_id: str, request: Request) -> ApiKeyRecordModel:
        """One key's record by its fingerprint id."""
        _require_admin(request)
        rec = key_store.get(key_id)
        if rec is None:
            raise ApiError(404, f"unknown key {key_id!r}", code="key_not_found")
        return _key_wire(rec)

    @app.post(
        "/harness/drain",
        response_model=DrainResponse,
        tags=["ops"],
        operation_id="drain",
    )
    def drain(
        wait_s: float = Query(default=0.0, ge=0.0, le=600.0),
    ) -> DrainResponse:
        """Latch drain mode (one-way): gated routes refuse new work with
        503 while in-flight requests finish; ``/health``, ``/metrics``
        and ``/ready`` keep answering so orchestrators can watch
        ``inflight`` bleed to zero before stopping the process.
        Idempotent — re-POSTing just re-reads the latch. ``wait_s>0``
        blocks (server-side) until in-flight work empties or the window
        lapses; ``drained`` reports which happened."""
        metrics.draining.set()
        if wait_s > 0:
            metrics.wait_idle(wait_s)
        snap = metrics.snapshot()
        return DrainResponse(draining=True, inflight=snap.inflight, drained=snap.inflight == 0)

    @app.get(
        "/harness/commands",
        response_model=HarnessCommandListResponse,
        tags=["runs"],
        operation_id="list_commands",
    )
    def list_commands(
        role: HarnessRole | None = Query(default=None),
    ) -> HarnessCommandListResponse:
        items = [
            HarnessCommandItem(
                name=c.name,
                role=str(c.role),
                argv=list(c.argv),
                timeout_s=c.timeout_s,
                description=c.description,
            )
            for c in lab.list_commands(role)
        ]
        return HarnessCommandListResponse(items=items, total=len(items))

    @app.post(
        "/harness/runs",
        response_model=HarnessRunResponse,
        tags=["runs"],
        operation_id="run_command",
    )
    async def run_command(
        body: HarnessRunRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> HarnessRunResponse:
        def execute() -> HarnessRunResponse:
            body_fp = _body_fp(body, exclude={"idempotency_key"})
            key, replay = _idem_lookup(idempotency_key or body.idempotency_key, idem_store, body_fp)
            if replay is not None:
                return replay
            with _work_gate():
                try:
                    result = lab.run(
                        body.command,
                        body.extra_args or None,
                        config=Path(body.config) if body.config else None,
                    )
                except KeyError as exc:
                    raise ApiError(404, str(exc)) from exc
                except ValueError as exc:
                    raise ApiError(422, str(exc)) from exc
                except Exception as exc:  # noqa: BLE001 — executor faults envelope like the jobs twin
                    raise ApiError(500, f"{type(exc).__name__}: {exc}") from exc
                command = lab.get(body.command)
                resp = HarnessRunResponse(
                    command=result.command,
                    exit_code=result.exit_code,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    ok=result.ok,
                    timeout_s=command.timeout_s,
                )
            if key is not None:
                idem_store.put(key, body_fp, resp)
            return resp

        key = _idem_key(idempotency_key or body.idempotency_key)
        return await _run_claimed(idem_store.async_claim_lock(_idem_scope(key)), execute)

    _mount_job_routes(app, lab, job_store, metrics, inflight, jobs_executor)
    _mount_key_lifecycle(
        app,
        key_store=key_store,
        key_idem_store=key_idem_store,
        completion_log=completion_log,
        metrics=metrics,
    )

    def _resolve_request_backend(
        backend_name: str,
        checkpoint_dir: str | None,
        byok: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> Any:
        """Checkpoint validation + backend resolution → HTTP error map."""
        kwargs: dict[str, Any] = {}
        if timeout_s is not None:
            kwargs["timeout_s"] = timeout_s
        if byok is not None:
            if backend_name != "byok":
                raise ApiError(422, "a byok override applies only to backend='byok'")
            if not byok_override_enabled:
                raise ApiError(
                    422,
                    "per-request BYOK credentials are disabled on this server "
                    f"({_BYOK_OVERRIDE_ENV}=0 / byok_override=False)",
                    code="byok_override_disabled",
                )
            kwargs.update(byok)
        if backend_name == "local_fx1":
            checkpoint = checkpoint_dir or os.environ.get("FX1_CHECKPOINT_DIR")
            if not checkpoint:
                raise ApiError(
                    422,
                    "local_fx1 needs a checkpoint_dir in the request or "
                    "FX1_CHECKPOINT_DIR on the server",
                )
            kwargs["checkpoint_dir"] = checkpoint
        elif checkpoint_dir is not None:
            raise ApiError(422, "checkpoint_dir applies only to the local_fx1 backend")
        try:
            return resolve_backend(backend_name, **kwargs)
        except KeyError as exc:
            raise ApiError(404, str(exc)) from exc
        except FileNotFoundError as exc:
            raise ApiError(422, str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            # Missing credentials / unsigned release / failed ship gate are
            # server-side configuration faults, not client input errors.
            raise ApiError(503, str(exc), code="backend_unavailable") from exc

    _mount_complete_routes(
        app,
        slot=_slot,
        resolve_backend=_resolve_request_backend,
        sse_keepalive_s=sse_keepalive_s,
        complete_idem_store=complete_idem_store,
        complete_batch_idem_store=complete_batch_idem_store,
        openai_idem_store=openai_idem_store,
        anthropic_idem_store=anthropic_idem_store,
        legacy_idem_store=legacy_idem_store,
        breaker=breaker,
        receipt_index=receipt_index,
        metrics=metrics,
        probe_cache=probe_cache,
        probe_lock=probe_lock,
        completion_log=completion_log,
        eval_store=eval_store,
        inflight=inflight,
        jobs_executor=jobs_executor,
        file_store=file_store,
        batch_store=batch_store,
        abatch_store=abatch_store,
        upload_store=upload_store,
        upload_idem_store=upload_idem_store,
        ft_store=ft_store,
        ft_runner=ft_runner_eff,
        ft_dir=ft_work_root,
        envelope_store=envelope_store,
        batch_line_max=batch_line_max,
        file_bytes_max=file_bytes_max,
        bg_cancel=_bg_cancel,
        conv_store=conv_store,
        eval_spec_store=eval_spec_store,
        vs_store=vs_store,
        key_store=key_store,
    )

    _mount_receipt_routes(app, receipt_index)
    _mount_v1_catch_all(app)

    # Opt-in CORS for browser consumers: off by default (closed), explicit
    # origins only — the wildcard and credentials are refused. Registered
    # last so it wraps the auth middleware — preflight OPTIONS reach CORS
    # before the API-key check (preflights carry no credentials).
    if cors_list:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_list,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=_CORS_ALLOW_HEADERS,
            expose_headers=_CORS_EXPOSE_HEADERS,
            allow_credentials=False,
            max_age=600,
        )

    _declare_response_headers(app, rate_limited=limiter is not None)

    return app


app = create_app()


if __name__ == "__main__":
    # `python -m fx1.serve.api` — loopback guard matches the research API.
    import uvicorn

    host = os.environ.get("FX1_API_HOST", "127.0.0.1")
    port = int(os.environ.get("FX1_API_PORT", "8011"))
    if host not in {"127.0.0.1", "::1", "localhost"} and not os.environ.get(_API_KEY_ENV):
        raise SystemExit(
            "non-loopback binding requires FX1_API_KEY; refusing unauthenticated exposure"
        )
    uvicorn.run(app, host=host, port=port, reload=False, server_header=False)
