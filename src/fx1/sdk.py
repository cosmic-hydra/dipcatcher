"""Fx1Harness — the typed in-process client for the dipcatcher harness.

``fx1.serve.api`` exposes the harness over HTTP for remote fx-1 instances;
this module is the same surface for in-process callers — the SDK twin.
Same registry, same backend set, same honesty gate, same receipt verifier;
the only thing dropped is the socket (auth/body-cap are transport concerns,
not contract).

Call-shape parity with the API is the contract:

- ``commands(role=None)``   ↔ ``GET  /harness/commands``
- ``run(...)``              ↔ ``POST /harness/runs``
- ``complete(...)``         ↔ ``POST /harness/complete``
- ``verify_receipt(...)``   ↔ ``POST /receipts/verify``
- ``health()``              ↔ ``GET  /health``

Error taxonomy (the SDK raises; the API maps to status codes):

- ``KeyError``                    unknown command / backend        (404)
- ``ValueError``                  contract violation, escapes       (422)
- ``FileNotFoundError``           missing checkpoint                (422)
- ``BackendNotConfiguredError``   missing credentials/engine        (503)
- ``NotImplementedError``         backend lacks the operation       (501)
- ``Fx1HonestyError``             model output refused by the gate  (502)
- ``RuntimeError``                transport/other backend fault     (502)

Everything is injectable: pass a ``harness=`` or ``backend_resolver=``
double and no subprocess or network is touched.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fx1 import __version__
from fx1.harness import Harness, HarnessCommand, HarnessResult, HarnessRole
from fx1.honesty import Fx1HonestyError, honesty_categories, validate_fx1_output
from fx1.serve.anthropic_compat import (
    AnthropicBatchCounts,
    AnthropicBatchCreate,
    AnthropicCountTokensRequest,
    AnthropicMessageObject,
    AnthropicMessagesRequest,
    anthropic_batch_object,
    anthropic_batch_result,
    anthropic_count_messages,
    anthropic_envelope,
    anthropic_error_body,
    anthropic_events,
    anthropic_model_object,
    anthropic_to_openai,
)
from fx1.serve.backends import (
    BackendNotConfiguredError,
    EmbeddingBackend,
    InferenceBackend,
    SamplingParams,
    StreamingBackend,
    TokenCountingBackend,
    byok_base_url_problem,
    get_backend,
    truncate_chunks,
)
from fx1.serve.chat import cited_complete, cited_complete_tools
from fx1.serve.contract import API_VERSION
from fx1.serve.evals import (
    EvalDiff,
    EvalRecord,
    EvalSpecStore,
    EvalStore,
    diff_eval_records,
)
from fx1.serve.finetune import (
    TRAINABLE_MODELS,
    FTHyperparameters,
    FTJob,
    FTJobCheckpointList,
    FTJobError,
    FTJobRunner,
    FTJobSpec,
    FTJobStore,
    default_ft_runner,
    validate_chat_jsonl,
)
from fx1.serve.journal import JobJournal
from fx1.serve.openai_compat import (
    OPENAI_BATCH_ENDPOINTS,
    OPENAI_FILE_BYTES_MAX,
    OPENAI_FILE_PURPOSE_ACCEPT,
    OPENAI_MODEL_IDS,
    OPENAI_RESPONSE_TERMINAL,
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAICompatError,
    OpenAICompletionRequest,
    OpenAIEmbeddingRequest,
    OpenAIEnvelopeStore,
    OpenAIModel,
    OpenAIModelDelete,
    OpenAIModelList,
    OpenAIResponseRequest,
    OpenAIVectorStoreFileBatchCreate,
    OpenAIVectorStoreSearch,
    _resolve_openai_link,
    _resolve_timeout,
    batch_line_body,
    batch_line_shape,
    batch_output_line,
    chained_response_input,
    chat_messages_for_store,
    completion_events,
    conversation_id_of,
    embeddings_to_kwargs,
    file_search_call_item,
    legacy_to_chat,
    openai_chunks,
    openai_completion_envelope,
    openai_conversation_object,
    openai_embedding_envelope,
    openai_envelope,
    openai_error_body,
    openai_model,
    openai_models,
    openai_response_events,
    openai_response_object,
    openai_response_replay_events,
    openai_to_kwargs,
    paged_item_list,
    response_cap_call_items,
    response_input_items_for_store,
    response_query_text,
    response_text_format,
    response_to_kwargs,
    validate_openai_output,
    validate_response_format,
)
from fx1.serve.receipt_store import SHA256_HEX, ReceiptIndex
from fx1.serve.uploads import (
    UploadMeta,
    UploadStore,
    UploadStoreError,
    upload_object,
    validate_upload_intent,
)
from fx1.serve.usage_report import UsageReport
from fx1.serve.vectorstores import VectorStoreError, VectorStoreStore
from quant_fund.schemas.receipt import verify_receipt_file, verify_receipt_payload

__all__ = [
    "BackendNotConfiguredError",
    "CompletionRecord",
    "CompletionResult",
    "GateCheckResult",
    "ProbeResult",
    "Fx1Harness",
    "OpenAIChatRequest",
    "OpenAIChatResponse",
    "OpenAIModelList",
    "OpenAIResponseRequest",
    "OpsMetrics",
    "ReceiptRef",
    "ReceiptVerdict",
    "StoredReceipt",
]

BackendResolver = Callable[..., InferenceBackend]


@dataclass(frozen=True)
class CompletionResult:
    """One gated completion — mirrors ``CompleteResponse`` on the API."""

    backend: str
    model: str | None
    content: str
    receipt_hashes: tuple[str, ...] = ()
    # Wire-only flag: True when the API replayed an idempotency-cached
    # response instead of re-running the model.
    replayed: bool = False
    # Endpoint-reported token counts for this call (None when the backend
    # has no usage channel). Never populated on ``complete_many`` items —
    # a shared backend can't attribute counts per prompt.
    usage: dict[str, int] | None = None
    # Handle into the surface's completion log — minted server-side on the
    # wire, by the SDK in-process. Every gated call is fetchable evidence.
    completion_id: str | None = None
    # Ordered fallback-chain trace — one entry per link tried (the last
    # is the serving link); empty when the primary served unchallenged.
    attempts: tuple[dict[str, Any], ...] = ()
    # The resolved decode params sent to the provider.
    sampling: dict[str, Any] | None = None
    # Upstream-reported tool calls, the provider's own finish_reason, and
    # the verbatim ``choices[].logprobs`` payload when the request asked
    # for it — verbatim on structured-channel links; None on plain-text
    # turns. Mirrors the wire's response fields.
    tool_calls: tuple[dict[str, Any], ...] | None = None
    finish_reason: str | None = None
    logprobs: dict[str, Any] | None = None


@dataclass(frozen=True)
class CompletionRecord:
    """One recorded gated call — mirrors ``CompletionRecord`` on the API.

    Hashes of the prompt and output, never the content: the log is
    evidence, not a transcript.
    """

    completion_id: str
    backend: str
    ok: bool
    latency_ms: float
    at: float
    prompt_sha256: str
    model: str | None = None
    usage: dict[str, int] | None = None
    error: str | None = None
    error_class: str | None = None
    output_sha256: str | None = None
    attempts: tuple[dict[str, Any], ...] | None = None
    sampling: dict[str, Any] | None = None
    # Caller-side attribution — mirrors the wire record's ``user`` /
    # ``metadata`` evidence fields.
    user: str | None = None
    metadata: dict[str, str] | None = None
    # Credential fingerprint on the wire (``env`` / managed-key id);
    # in-process calls are unauthenticated so this stays None.
    key_id: str | None = None


class _CompletionLog:
    """Bounded in-process ring of completion records, newest-first on
    read; the in-process twin of the API's log."""

    def __init__(self, cap: int = 256) -> None:
        self._cap = cap
        self._lock = threading.Lock()
        self._items: dict[str, CompletionRecord] = {}
        self._dropped = 0

    def append(self, rec: CompletionRecord) -> None:
        with self._lock:
            self._items[rec.completion_id] = rec
            while len(self._items) > self._cap:
                self._items.pop(next(iter(self._items)))
                self._dropped += 1

    @property
    def cap(self) -> int:
        return self._cap

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def all(self, backend: str | None = None) -> list[CompletionRecord]:
        """Every retained record — the aggregation view (no limit)."""
        return self.latest(self._cap, backend)

    def get(self, completion_id: str) -> CompletionRecord | None:
        with self._lock:
            return self._items.get(completion_id)

    def latest(self, limit: int, backend: str | None) -> list[CompletionRecord]:
        with self._lock:
            items = sorted(self._items.values(), key=lambda r: r.at, reverse=True)
        if backend is not None:
            items = [r for r in items if r.backend == backend]
        return items[:limit]


_BACKEND_NAMES = ("hosted_k3", "local_fx1", "byok")

# Patch three-state: an omitted kwarg stays the declared policy while
# an explicit ``None`` clears the bound — the wire's JSON-null.
_UNSET: Any = object()


def _fallback_chain(backend: str, fallbacks: list[str] | None) -> list[str]:
    """Validated ordered chain: primary first, then distinct alternates —
    the same contract the API enforces on ``fallbacks``."""
    links = [backend, *(fallbacks or ())]
    if len(set(links)) != len(links) or len(links) > 3:
        raise ValueError("fallbacks must be distinct, at most two, and not repeat the primary")
    bad = [n for n in fallbacks or () if n not in _BACKEND_NAMES]
    if bad:
        raise ValueError(f"unknown fallback backends: {bad}")
    return links


def _check_link_kwargs(
    chain: list[str],
    checkpoint_dir: str | Path | None,
    byok: dict[str, str] | None,
) -> None:
    """Per-link kwargs must bind a chain link — the request-level twin of
    the API's ``_chain_valid`` (wire 422 ↔ sdk ValueError)."""
    if byok is not None and "byok" not in chain:
        raise ValueError("a byok override applies only to a 'byok' chain link")
    if checkpoint_dir is not None and "local_fx1" not in chain:
        raise ValueError("checkpoint_dir applies only to the local_fx1 backend")


def _messages_sha256(messages: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(messages, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sampling_params(  # NOSONAR(S107)
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    stop: list[str] | str | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    logit_bias: dict[str, int] | None = None,
    reasoning_effort: str | None = None,
    service_tier: str | None = None,
    prompt_cache_key: str | None = None,
    prompt_cache_retention: str | None = None,
    verbosity: str | None = None,
    user: str | None = None,
) -> SamplingParams:
    """Build the declared-params dataclass — the SDK twin of the wire's
    request-model validation (same caps, fail-closed ``ValueError``)."""
    stops = [stop] if isinstance(stop, str) else (list(stop) if stop is not None else None)
    if stops is not None and (len(stops) > 4 or any(not 1 <= len(s) <= 512 for s in stops)):
        raise ValueError("stop accepts ≤4 sequences of 1–512 chars")
    if logit_bias is not None:
        for k, v in logit_bias.items():
            try:
                int(k)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"logit_bias keys must be token ids, got {k!r}") from exc
            if not -100 <= v <= 100:
                raise ValueError(f"logit_bias[{k!r}]={v} outside [-100, 100]")
    for name, val in (
        ("presence_penalty", presence_penalty),
        ("frequency_penalty", frequency_penalty),
    ):
        if val is not None and not -2.0 <= val <= 2.0:
            raise ValueError(f"{name} must be in [-2, 2], got {val}")
    if temperature is not None and not 0.0 <= temperature <= 2.0:
        raise ValueError(f"temperature must be in [0, 2], got {temperature}")
    if top_p is not None and not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if max_tokens is not None and not 0 < max_tokens <= 262144:
        raise ValueError(f"max_tokens must be in (0, 262144], got {max_tokens}")
    if seed is not None and seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    if reasoning_effort is not None and reasoning_effort not in (
        "none",
        "minimal",
        "low",
        "medium",
        "high",
    ):
        raise ValueError("reasoning_effort must be none|minimal|low|medium|high")
    if service_tier is not None and service_tier not in (
        "auto",
        "default",
        "flex",
        "priority",
        "scale",
    ):
        raise ValueError("service_tier must be auto|default|flex|priority|scale")
    if prompt_cache_retention is not None and prompt_cache_retention not in (
        "in-memory",
        "24h",
    ):
        raise ValueError("prompt_cache_retention must be in-memory|24h")
    if verbosity is not None and verbosity not in ("low", "medium", "high"):
        raise ValueError("verbosity must be low|medium|high")
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        stop=tuple(stops) if stops else None,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
        reasoning_effort=reasoning_effort,
        service_tier=service_tier,
        prompt_cache_key=prompt_cache_key,
        prompt_cache_retention=prompt_cache_retention,
        verbosity=verbosity,
        user=user,
    )


@dataclass(frozen=True)
class GateCheckResult:
    """Honesty-gate verdict — mirrors ``GateCheckResponse`` on the API."""

    ok: bool
    error: str | None = None


@dataclass(frozen=True)
class ProbeResult:
    """Deep-health verdict for one backend — mirrors
    ``BackendProbeResponse`` on the API."""

    backend: str
    ok: bool
    model: str | None
    latency_ms: float
    error: str | None = None
    error_class: str | None = None


@dataclass(frozen=True)
class ReceiptVerdict:
    """Verifier output — mirrors ``ReceiptVerifyResponse`` on the API."""

    valid: bool
    path: str
    schema_tag: str
    kind: str | None
    verdict: str | None
    digest_convention: str | None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReceiptRef:
    """One entry in the sealed-receipt store index — mirrors ReceiptIndexItem."""

    sha256: str
    name: str


@dataclass(frozen=True)
class StoredReceipt:
    """A fetched receipt: the verbatim sealed document plus the live
    verifier's verdict (``X-Fx1-Receipt-Valid`` on the wire, re-verify
    in-process here)."""

    sha256: str
    document: dict[str, Any]
    valid: bool


@dataclass(frozen=True)
class HarnessHealth:
    """Liveness + configured-backend presence booleans — no secret values."""

    status: str
    version: str
    registered_commands: int
    backends: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class OpsMetrics:
    """Remote ops snapshot — mirrors ``MetricsResponse`` on the API.

    Wire-only: the in-process surface has no HTTP layer to meter, so
    ``Fx1Harness`` deliberately does not expose this shape.
    """

    uptime_s: float
    requests_total: int
    errors_total: int
    by_status: dict[str, int] = field(default_factory=dict)
    inflight: int = 0
    inflight_watermark: int = 0
    max_inflight: int = 0
    draining: bool = False
    rate_limited_total: int = 0
    # per-backend outcome counters + cumulative latency buckets, keyed by
    # backend name (probe verdicts live under ``probe:<name>`` entries)
    complete: dict[str, dict[str, Any]] = field(default_factory=dict)


class Fx1Harness:
    """In-process harness client: registry, runs, gated complete, verify."""

    def __init__(
        self,
        harness: Harness | None = None,
        backend_resolver: BackendResolver | None = None,
        receipts_dir: str | Path = "receipts",
        ft_runner: FTJobRunner | None = None,
        ft_dir: str | Path | None = None,
        state_dir: str | Path | None = None,
    ) -> None:
        self._harness = harness or Harness()
        self._resolve_backend = backend_resolver or get_backend
        self._receipts = ReceiptIndex(Path(receipts_dir))
        self._log = _CompletionLog()
        # state_dir (or FX1_SDK_STATE_DIR) binds the eval/ft stores to
        # the same hash-chained journals the server uses — an SDK
        # process restart recovers records, idem keys, events, and the
        # ft: model registry exactly like the wire.
        state_path = Path(state_dir) if state_dir is not None else None
        if state_path is None:
            env_dir = os.environ.get("FX1_SDK_STATE_DIR")
            state_path = Path(env_dir) if env_dir else None
        self._eval_store = EvalStore(
            256,
            journal=JobJournal(state_path / "evals.jsonl") if state_path is not None else None,
        )
        # The /v1/evals spec containers — journaled alongside the records.
        self._eval_spec_store = EvalSpecStore(
            256,
            journal=JobJournal(state_path / "eval_specs.jsonl") if state_path is not None else None,
        )
        # The /v1 retrieval index, in-process — store=false keeps a call
        # out of it, matching the wire's OpenAIEnvelopeStore semantics.
        self._openai_store = OpenAIEnvelopeStore(256)
        # Named conversation containers — the wire's conv_store twin;
        # journaled under state_dir like every other store so a fresh
        # harness on the same dir sees the same conversations.
        self._conv_store = OpenAIEnvelopeStore(
            256,
            journal=(
                JobJournal(state_path / "conversations.jsonl") if state_path is not None else None
            ),
        )
        # Cancel flags for background responses — a set event means the
        # stored envelope was flipped to ``cancelled`` and the worker must
        # not overwrite it with a terminal result.
        self._bg_cancel: dict[str, threading.Event] = {}
        # The /v1/fine_tuning twin — synchronous in process (no queue),
        # same store/runner contract as the wire.
        self._ft_store = FTJobStore(
            256,
            journal=JobJournal(state_path / "ft_jobs.jsonl") if state_path is not None else None,
        )
        # ft job work dirs — and the checkpoints they mint — default under
        # state_dir so a completed job's ft: model keeps resolving to real
        # weights after a process restart.
        if ft_dir is not None:
            self._ft_dir = Path(ft_dir) / "fx1_ft_sdk"
        elif state_path is not None:
            self._ft_dir = state_path / "ft"
        else:
            self._ft_dir = Path(tempfile.gettempdir()) / "fx1_ft_sdk"
        self._ft_runner = ft_runner or default_ft_runner(self._resolve_backend)
        # The /v1/uploads twin — chunked assembly into process-local file
        # records; journaled under state_dir like every other store.
        self._upload_store = UploadStore(64, OPENAI_FILE_BYTES_MAX, state_dir=state_path)
        # File records minted by upload_complete — the wire's _FileStore
        # twin in miniature (process-local; the upload journal is the
        # durable half).
        self._files: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._files_lock = threading.Lock()
        # The wire-correlate headers of the last gated call — stamped by
        # ``_record_call`` so in-process callers read the same tracing
        # surface the HTTP middleware emits for that call.
        self._last_response_headers: dict[str, str] = {}
        # Managed API keys — the wire's ApiKeyStore twin. Bound to
        # ``state_dir/keys.jsonl`` when durable, so provisioning keys
        # from the SDK writes the same journal the server replays.
        from fx1.serve.keys import ApiKeyStore  # noqa: PLC0415

        self._key_store = ApiKeyStore(
            journal=JobJournal(state_path / "keys.jsonl") if state_path is not None else None
        )
        # The /v1/vector_stores twin — borrows file content through a
        # reader over ``self._files`` (the upload_complete records);
        # journaled under state_dir like every other store. A file that
        # isn't on disk fails the attach honestly.

        def _sdk_file_reader(file_id: str) -> tuple[bytes, str] | None:
            rec = self._files.get(file_id)
            if rec is None:
                return None
            content = rec.get("_content")
            if isinstance(content, (bytes, bytearray)):
                data = bytes(content)
            elif isinstance(content, str):
                data = content.encode()
            else:
                return None
            return data, str(rec.get("filename") or file_id)

        self._vs_store = VectorStoreStore(512, state_dir=state_path, file_reader=_sdk_file_reader)

    def _record_call(
        self,
        backend: str,
        model: str | None,
        ok: bool,
        latency_ms: float,
        usage: dict[str, int] | None,
        error: str | None,
        error_class: str | None,
        prompt_sha256: str,
        output_sha256: str | None,
        attempts: tuple[dict[str, Any], ...] | None = None,
        sampling: dict[str, Any] | None = None,
        user: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Append one call to the completion log; returns its id."""
        cid = uuid.uuid4().hex
        self._log.append(
            CompletionRecord(
                completion_id=cid,
                backend=backend,
                model=model,
                ok=ok,
                latency_ms=latency_ms,
                at=time.time(),
                usage=usage,
                error=error,
                error_class=error_class,
                prompt_sha256=prompt_sha256,
                output_sha256=output_sha256,
                attempts=attempts,
                sampling=sampling,
                user=user,
                metadata=metadata,
            )
        )
        # No transport means no socket headers — but every quantity the
        # middleware stamps is owned here: a per-call trace id, the wire
        # contract, the call's own wall-clock, and the completion id.
        # Build locally and publish once so concurrent readers cannot observe
        # a partially populated response-header dictionary.
        headers = {
            "x-request-id": uuid.uuid4().hex,
            "x-fx1-api-version": API_VERSION,
            "openai-processing-ms": str(int(latency_ms)),
        }
        if ok:
            # the wire's X-Fx1-Completion-Id appears on served responses;
            # a refused/failed call's error envelope carries none
            headers["x-fx1-completion-id"] = cid
        self._last_response_headers = headers
        return cid

    @property
    def last_response_headers(self) -> dict[str, str]:
        """The headers the HTTP surface would stamp for the same gated
        call — ``x-request-id`` (minted per call), ``x-fx1-api-version``,
        ``openai-processing-ms`` (the call's measured latency), and
        ``x-fx1-completion-id`` — so SDK and remote callers read the same
        tracing surface. ``{}`` before the first gated call."""
        return dict(self._last_response_headers)

    def completions(self, *, limit: int = 50, backend: str | None = None) -> list[CompletionRecord]:
        """Newest-first window on the in-process completion log — the
        same audit evidence the API exposes at ``GET /harness/completions``."""
        return self._log.latest(limit, backend)

    def completion(self, completion_id: str) -> CompletionRecord:
        """Fetch one recorded call — KeyError when the id is unknown or
        already evicted (mirrors the wire's 404)."""
        rec = self._log.get(completion_id)
        if rec is None:
            raise KeyError(completion_id)
        return rec

    def usage(
        self,
        *,
        backend: str | None = None,
        model: str | None = None,
        key_id: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> UsageReport:
        """Token/request accounting over the in-process completion log —
        the in-process twin of ``GET /harness/usage``. `since`/`until`
        are unix-second bounds; since>until is a fail-closed ValueError."""
        from fx1.serve.usage_report import aggregate_usage  # noqa: PLC0415

        if since is not None and until is not None and since > until:
            raise ValueError("since must be <= until")
        if backend is not None and backend not in _BACKEND_NAMES:
            raise ValueError(f"unknown backend {backend!r}")
        return aggregate_usage(
            self._log.all(backend),
            cap=self._log.cap,
            dropped=self._log.dropped,
            backend=backend,
            model=model,
            key_id=key_id,
            since=since,
            until=until,
        )

    # Managed API keys — the in-process twins of ``/harness/keys``.
    # Bound to ``state_dir``'s keys.jsonl when durable, so an SDK
    # process can provision the keys a server deployment replays.

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
        """Mint a managed key — returns the wire mint shape including the
        raw ``key`` (shown once, never stored). ``admin=True`` keys may
        manage keys on the wire surface; ``rpm`` bounds the key to a
        fixed-window request rate and ``ttl_s`` bakes an expiry into the
        journaled record. ``scopes`` bounds the key to ``read``/``write``/
        ``admin`` surface classes — journaled with the record so the
        wire enforces the declared policy across restarts.
        ``max_requests``/``max_tokens`` declare hard budgets — an
        exhausted key answers 429 ``quota_exceeded`` on the wire."""
        from fx1.serve.keys import KeyStoreError  # noqa: PLC0415

        try:
            raw, rec = self._key_store.mint(
                name,
                admin=admin,
                rpm=rpm,
                ttl_s=ttl_s,
                scopes=scopes,
                max_requests=max_requests,
                max_tokens=max_tokens,
            )
        except KeyStoreError as exc:
            raise ValueError(str(exc)) from exc
        return {
            "id": rec["key_id"],
            "object": "key",
            "name": rec["name"],
            "prefix": rec["prefix"],
            "admin": bool(rec.get("admin")),
            "scopes": list(rec["scopes"]),
            "rpm": rec.get("rpm"),
            "max_requests": rec.get("max_requests"),
            "max_tokens": rec.get("max_tokens"),
            "expires_at": rec.get("expires_at"),
            "created_at": rec["created_at"],
            "tokens_used": int(rec.get("tokens_used") or 0),
            "rotated_from": rec.get("rotated_from"),
            "key": raw,
        }

    def key_rotate(
        self,
        key_id: str,
        *,
        revoke_old: bool = True,
        name: str | None = None,
        ttl_s: float | None = None,
    ) -> dict[str, Any]:
        """Atomic rotation — the in-process twin of
        ``POST /harness/keys/{id}/rotate``: mints a successor under the
        predecessor's declared policy (name/scopes/admin/rpm/budgets),
        ``revoke_old`` (default) tombstones the predecessor in the same
        store transaction, and without ``ttl_s`` the successor inherits
        the predecessor's absolute expiry — rotation never extends a
        credential's lifetime. Returns the ``key_rotation`` envelope with
        the minted ``key`` (raw secret shown once). ``KeyError`` when
        unknown; ``ValueError`` when the predecessor is already revoked
        or ``ttl_s`` is invalid."""
        from fx1.serve.keys import KeyStoreError  # noqa: PLC0415

        try:
            raw, rec = self._key_store.rotate(key_id, revoke_old=revoke_old, name=name, ttl_s=ttl_s)
        except KeyStoreError as exc:
            if exc.code == "key_not_found":
                raise KeyError(key_id) from exc
            raise ValueError(str(exc)) from exc
        return {
            "object": "key_rotation",
            "key": {
                "id": rec["key_id"],
                "object": "key",
                "name": rec["name"],
                "prefix": rec["prefix"],
                "admin": bool(rec.get("admin")),
                "scopes": list(rec["scopes"]),
                "rpm": rec.get("rpm"),
                "max_requests": rec.get("max_requests"),
                "max_tokens": rec.get("max_tokens"),
                "expires_at": rec.get("expires_at"),
                "created_at": rec["created_at"],
                "tokens_used": int(rec.get("tokens_used") or 0),
                "rotated_from": rec.get("rotated_from"),
                "key": raw,
            },
            "rotated_from": key_id,
            "revoked_previous": revoke_old,
        }

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
        """Mutable policy update on a live key — the in-process twin of
        ``PATCH /harness/keys/{id}``: an omitted kwarg keeps the
        declared policy, an explicit ``None`` clears the bound back to
        unbounded (the wire's JSON-null; ``clear=`` names the same
        fields), and a concrete value replaces it. ``admin=True``
        unions the admin scope the mint way; ``admin=False`` never
        strips a declared scope — ``scopes``/``admin`` take concrete
        values when given (an explicit ``None`` is a ``ValueError``,
        like the wire's 422). ``enabled`` and the live counters are
        not patchable — revocation is permanent. Returns the updated
        record. ``KeyError`` when unknown; ``ValueError`` when the key
        is revoked, a constraint is violated, or ``clear`` names a
        non-nullable field."""
        from fx1.serve.keys import CLEARABLE_KEY_FIELDS, KeyStoreError  # noqa: PLC0415

        clears = {f for f in clear}
        patch: dict[str, Any] = {}
        for fname, value in (
            ("name", name),
            ("rpm", rpm),
            ("max_requests", max_requests),
            ("max_tokens", max_tokens),
            ("expires_at", expires_at),
        ):
            if value is _UNSET:
                continue
            if value is None:
                clears.add(fname)
            else:
                patch[fname] = value
        for fname, cval in (("scopes", scopes), ("admin", admin)):
            if cval is _UNSET:
                continue
            if cval is None:
                raise ValueError(f"{fname} must take a concrete value when given")
            patch[fname] = cval
        bad = sorted(clears - CLEARABLE_KEY_FIELDS)
        if bad:
            raise ValueError(f"cannot clear non-nullable field {bad[0]!r}")
        try:
            rec = self._key_store.update(key_id, clear=clears, **patch)
        except KeyStoreError as exc:
            if exc.code == "key_not_found":
                raise KeyError(key_id) from exc
            raise ValueError(str(exc)) from exc
        return self._key_wire(rec)

    def keys(self) -> list[dict[str, Any]]:
        """Every minted key's fingerprint + metadata — never secrets."""
        return [self._key_wire(rec) for rec in self._key_store.list()]

    def key_get(self, key_id: str) -> dict[str, Any]:
        """One key's record by fingerprint id; KeyError when unknown."""
        rec = self._key_store.get(key_id)
        if rec is None:
            raise KeyError(key_id)
        return self._key_wire(rec)

    def key_revoke(self, key_id: str) -> dict[str, Any]:
        """Tombstone a key; KeyError unknown, ValueError already revoked."""
        from fx1.serve.keys import KeyStoreError  # noqa: PLC0415

        try:
            rec = self._key_store.revoke(key_id)
        except KeyStoreError as exc:
            if exc.code == "key_not_found":
                raise KeyError(key_id) from exc
            raise ValueError(str(exc)) from exc
        return self._key_wire(rec)

    def key_usage(self, key_id: str) -> dict[str, Any]:
        """One managed key's usage card — the in-process twin of
        ``GET /harness/keys/{id}/usage``: live counters, declared budgets
        with derived headroom, the rpm window state, and this process's
        completion-ring spend split (in-process calls are unauthenticated,
        so ``served`` counts only records already attributed to the key).
        ``KeyError`` when the key is unknown."""
        rec = self._key_store.get(key_id)
        if rec is None:
            raise KeyError(key_id)
        return self._usage_card(rec)

    def self_usage(self) -> dict[str, Any]:
        """``GET /harness/self`` in-process twin — the SDK IS the root
        credential, so the card reports the unmetered ``env`` class."""
        return {
            "object": "self_usage",
            "credential": "env",
            "scopes": ["read", "write", "admin"],
            "metered": False,
            "key": None,
        }

    def _usage_card(self, rec: dict[str, Any]) -> dict[str, Any]:
        uses = int(rec.get("uses") or 0)
        tokens_used = int(rec.get("tokens_used") or 0)
        max_req = rec.get("max_requests")
        max_tok = rec.get("max_tokens")
        ws = self._key_store.window_state(rec["key_id"])
        calls = 0
        prompt = 0
        completion = 0
        total_all = 0
        by_backend: dict[str, list[int]] = {}
        for r in self._log.all():
            if r.key_id != rec["key_id"]:
                continue
            calls += 1
            usage = r.usage or {}
            p = int(usage.get("prompt_tokens") or 0)
            c = int(usage.get("completion_tokens") or 0)
            t = usage.get("total_tokens")
            tt = int(t) if t is not None else p + c
            prompt += p
            completion += c
            total_all += tt
            bb = by_backend.setdefault(r.backend, [0, 0])
            bb[0] += 1
            bb[1] += tt
        return {
            "id": rec["key_id"],
            "object": "key_usage",
            "name": rec.get("name"),
            "admin": bool(rec.get("admin")),
            "enabled": bool(rec.get("enabled", True)),
            "created_at": rec["created_at"],
            "expires_at": rec.get("expires_at"),
            "revoked_at": rec.get("revoked_at"),
            "rotated_from": rec.get("rotated_from"),
            "uses": uses,
            "tokens_used": tokens_used,
            "last_used_at": rec.get("last_used_at"),
            "max_requests": max_req,
            "requests_remaining": (max(0, int(max_req) - uses) if max_req is not None else None),
            "max_tokens": max_tok,
            "tokens_remaining": (
                max(0, int(max_tok) - tokens_used) if max_tok is not None else None
            ),
            "rpm": rec.get("rpm"),
            "window_remaining": (ws[1] if ws is not None else None),
            "window_reset_s": (ws[2] if ws is not None else None),
            "served": {
                "calls": calls,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total_all,
                "by_backend": {
                    k: {"calls": v[0], "total_tokens": v[1]} for k, v in sorted(by_backend.items())
                },
            },
            "log_cap": self._log.cap,
            "log_dropped": self._log.dropped,
        }

    @staticmethod
    def _key_wire(rec: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": rec["key_id"],
            "object": "key",
            "name": rec["name"],
            "prefix": rec["prefix"],
            "admin": bool(rec.get("admin")),
            "scopes": list(
                rec.get("scopes")
                or (["read", "write", "admin"] if rec.get("admin") else ["read", "write"])
            ),
            "rpm": rec.get("rpm"),
            "max_requests": rec.get("max_requests"),
            "max_tokens": rec.get("max_tokens"),
            "expires_at": rec.get("expires_at"),
            "created_at": rec["created_at"],
            "enabled": rec["enabled"],
            "revoked_at": rec["revoked_at"],
            "uses": rec["uses"],
            "tokens_used": int(rec.get("tokens_used") or 0),
            "last_used_at": rec["last_used_at"],
            "rotated_from": rec.get("rotated_from"),
        }

    def completion_receipt(self, completion_id: str) -> dict[str, Any]:
        """Export one logged call as a sealed ``fx1_completion_record.v1``
        document — pass it to :meth:`verify_receipt` to check the seal."""
        from dataclasses import asdict  # noqa: PLC0415

        from fx1.serve.ops_receipt import (  # noqa: PLC0415
            completion_record_receipt,
        )

        rec = self._log.get(completion_id)
        if rec is None:
            raise KeyError(completion_id)
        return completion_record_receipt(asdict(rec))

    # ---- evals -----------------------------------------------------------

    def run_eval(
        self,
        suite: str,
        *,
        model_fn: Callable[[list[dict[str, str]]], str] | None = None,
        backend: str = "local_fx1",
        checkpoint_dir: str | Path | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        byok: dict[str, str] | None = None,
        timeout_s: float | None = None,
        fallbacks: list[str] | None = None,
        judge_backend: str | None = None,
        judge_byok: dict[str, str] | None = None,
        seed: int = 0,
    ) -> EvalRecord:
        """Run one seeded eval suite in-process — the API's
        ``POST /harness/evals`` twin. ``suite`` is one of the registry
        names (``capability``/``calibration``/``tooluse``/``retrieval``/
        ``ts_reasoning``/``ext_bench``/``options_reasoning``).

        The model is either ``model_fn`` (a raw callable — the
        weights-direct lane: any in-process model, no backend machinery)
        or a resolved backend chain (``backend`` + ``fallbacks`` +
        ``byok``/``checkpoint_dir`` — same per-link kwarg binding as
        ``complete``). ``judge_backend``/``judge_byok`` feed the suites
        that grade with a second model (capability, ext_bench).

        Every model call lands on the completion log under
        ``eval:{suite}:{backend}`` — eval work is metered evidence, never
        silent. Evals run under the decode pin ``{"temperature": 0.0}``;
        the record carries it so the sealed receipt states the decode
        config. The terminal record is stored — :meth:`evals`,
        :meth:`eval_record`, :meth:`eval_receipt` read it back."""
        from fx1.serve.evals import (  # noqa: PLC0415
            EVAL_SAMPLING,
            EVAL_SUITES,
            EvalRecord,
            run_eval_record,
            suite_accepts_judge,
        )

        if suite not in EVAL_SUITES:
            raise KeyError(f"unknown eval suite {suite!r} — registered: {sorted(EVAL_SUITES)}")
        if judge_backend is not None and not suite_accepts_judge(suite):
            raise ValueError(f"suite {suite!r} takes no judge")
        if judge_byok is not None and judge_backend != "byok":
            raise ValueError("judge_byok applies only to judge_backend='byok'")
        record = EvalRecord(
            eval_id=uuid.uuid4().hex,
            suite=suite,
            backend=backend if model_fn is None else "model_fn",
            seed=seed,
            status="running",
            created_at=time.time(),
            sampling=EVAL_SAMPLING.body_fields(),
        )
        # Journal the in-flight record before the suite runs — a process
        # crash mid-eval recovers it as failed, matching the wire.
        self._eval_store.put(record, None, None)
        try:
            if model_fn is not None:
                metric_key = f"eval:{suite}:model_fn"
                name = "model_fn"

                def _fn(messages: list[dict[str, str]]) -> str:
                    p_sha = _messages_sha256(messages)
                    t0 = time.monotonic()
                    try:
                        out = model_fn(messages)
                    except Exception as exc:
                        self._record_call(
                            metric_key,
                            None,
                            False,
                            (time.monotonic() - t0) * 1000.0,
                            None,
                            str(exc),
                            type(exc).__name__,
                            p_sha,
                            None,
                            None,
                            EVAL_SAMPLING.body_fields(),
                        )
                        raise
                    o_sha = hashlib.sha256(out.encode("utf-8")).hexdigest()
                    self._record_call(
                        metric_key,
                        None,
                        True,
                        (time.monotonic() - t0) * 1000.0,
                        None,
                        None,
                        None,
                        p_sha,
                        o_sha,
                        None,
                        EVAL_SAMPLING.body_fields(),
                    )
                    return out

            else:
                chain = _fallback_chain(backend, fallbacks)
                _check_link_kwargs(chain, checkpoint_dir, byok)
                attempts: list[dict[str, Any]] = []
                last_exc: Exception | None = None
                name = ""
                backend_obj: Any = None
                for cand in chain:
                    t0 = time.monotonic()
                    try:
                        backend_obj = self._resolve_link(
                            cand, checkpoint_dir, backend_kwargs, byok, timeout_s
                        )
                    except (BackendNotConfiguredError, RuntimeError, ValueError) as exc:
                        attempts.append(
                            {
                                "backend": cand,
                                "ok": False,
                                "error_class": type(exc).__name__,
                                "latency_ms": (time.monotonic() - t0) * 1000.0,
                            }
                        )
                        last_exc = exc
                        continue
                    attempts.append({"backend": cand, "ok": True})
                    name = cand
                    break
                if backend_obj is None:
                    if last_exc is None:
                        raise RuntimeError("backend chain exhausted without exception")
                    raise last_exc
                record.backend = name
                record.attempts = attempts
                metric_key = f"eval:{suite}:{name}"

                def _fn(messages: list[dict[str, str]]) -> str:
                    p_sha = _messages_sha256(messages)
                    t0 = time.monotonic()
                    try:
                        out: str = backend_obj.complete(messages, sampling=EVAL_SAMPLING)
                    except Exception as exc:
                        self._record_call(
                            metric_key,
                            getattr(backend_obj, "_model", None)
                            if isinstance(getattr(backend_obj, "_model", None), str)
                            else None,
                            False,
                            (time.monotonic() - t0) * 1000.0,
                            None,
                            str(exc),
                            type(exc).__name__,
                            p_sha,
                            None,
                            None,
                            EVAL_SAMPLING.body_fields(),
                        )
                        raise
                    o_sha = hashlib.sha256(out.encode("utf-8")).hexdigest()
                    self._record_call(
                        metric_key,
                        getattr(backend_obj, "_model", None)
                        if isinstance(getattr(backend_obj, "_model", None), str)
                        else None,
                        True,
                        (time.monotonic() - t0) * 1000.0,
                        getattr(backend_obj, "last_usage", None),
                        None,
                        None,
                        p_sha,
                        o_sha,
                        None,
                        EVAL_SAMPLING.body_fields(),
                    )
                    return out

            judge_fn: Callable[[list[dict[str, str]]], str] | None = None
            if judge_backend is not None:
                judge_obj = self._resolve_completion_backend(
                    judge_backend,
                    None,
                    backend_kwargs,
                    judge_byok if judge_backend == "byok" else None,
                    timeout_s,
                )
                judge_key = f"eval:{suite}:judge:{judge_backend}"

                def judge_fn(messages: list[dict[str, str]]) -> str:
                    p_sha = _messages_sha256(messages)
                    t0 = time.monotonic()
                    try:
                        out: str = judge_obj.complete(messages, sampling=EVAL_SAMPLING)
                    except Exception as exc:
                        self._record_call(
                            judge_key,
                            None,
                            False,
                            (time.monotonic() - t0) * 1000.0,
                            None,
                            str(exc),
                            type(exc).__name__,
                            p_sha,
                            None,
                            None,
                            EVAL_SAMPLING.body_fields(),
                        )
                        raise
                    o_sha = hashlib.sha256(out.encode("utf-8")).hexdigest()
                    self._record_call(
                        judge_key,
                        None,
                        True,
                        (time.monotonic() - t0) * 1000.0,
                        getattr(judge_obj, "last_usage", None),
                        None,
                        None,
                        p_sha,
                        o_sha,
                        None,
                        EVAL_SAMPLING.body_fields(),
                    )
                    return out

            run_eval_record(record, model=_fn, judge=judge_fn)
        except Exception as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            record.status = "failed"
            record.finished_at = time.time()
        self._eval_store.mark(record)
        return record

    def evals(
        self,
        *,
        status: str | None = None,
        suite: str | None = None,
        limit: int | None = None,
    ) -> list[EvalRecord]:
        """Newest-first in-process eval records — the wire twin is
        ``GET /harness/evals``."""
        records, _total = self._eval_store.list_records(status=status, suite=suite, limit=limit)
        return records

    def eval_record(self, eval_id: str) -> EvalRecord:
        """One stored eval record — KeyError on unknown/evicted ids (the
        wire's 404)."""
        rec = self._eval_store.get(eval_id)
        if rec is None:
            raise KeyError(eval_id)
        return rec

    def eval_receipt(self, eval_id: str) -> dict[str, Any]:
        """Export a terminal eval record as sealed ``fx1_eval_record.v1``
        — mirrors ``GET /harness/evals/{id}/receipt``; non-terminal
        records refuse (a sealed receipt must be immutable evidence)."""
        from fx1.serve.evals import eval_record_receipt  # noqa: PLC0415

        rec = self._eval_store.get(eval_id)
        if rec is None:
            raise KeyError(eval_id)
        if rec.status in ("queued", "running"):
            raise RuntimeError(
                f"eval {eval_id} is {rec.status} — receipts export on terminal records only"
            )
        return eval_record_receipt(rec.model_dump(mode="json"))

    def eval_diff(self, base_id: str, candidate_id: str) -> EvalDiff:
        """Diff two stored eval records — the wire twin is
        ``GET /harness/evals/{base}/diff/{candidate}``. Unknown ids raise
        ``KeyError``; non-terminal records or missing reports raise
        ``RuntimeError`` (the wire's 404/409)."""
        base = self._eval_store.get(base_id)
        if base is None:
            raise KeyError(base_id)
        cand = self._eval_store.get(candidate_id)
        if cand is None:
            raise KeyError(candidate_id)
        for rec in (base, cand):
            if rec.status in ("queued", "running") or rec.report is None:
                raise RuntimeError(
                    f"eval {rec.eval_id} is {rec.status} — diffs need terminal records with reports"
                )
        return diff_eval_records(base, cand)

    # ---- /v1/evals — named spec containers + bound runs ----------------

    def eval_spec_create(
        self,
        name: str,
        *,
        suite: str,
        seed: int = 0,
        backend: str | None = None,
        fallbacks: list[str] | None = None,
        checkpoint_dir: str | Path | None = None,
        judge_backend: str | None = None,
        timeout_s: float | None = None,
        testing_criteria: list[dict[str, Any]] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """The ``POST /v1/evals`` twin — declare a named eval container.
        The item_schema pins the suite knobs (validated fail-closed);
        credentials never live on a spec."""
        from fx1.serve.evals import EvalSpec, EvalSpecItemSchema, spec_wire  # noqa: PLC0415

        item_schema = EvalSpecItemSchema.model_validate(
            {
                "suite": suite,
                "seed": seed,
                "backend": backend,
                "fallbacks": fallbacks or [],
                "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
                "judge_backend": judge_backend,
                "timeout_s": timeout_s,
            }
        )
        spec = EvalSpec(
            spec_id=f"eval_{uuid.uuid4().hex[:24]}",
            name=name,
            data_source_config={
                "type": "custom",
                "item_schema": item_schema.model_dump(exclude_none=True),
            },
            testing_criteria=list(testing_criteria or []),
            metadata=dict(metadata or {}),
            created_at=time.time(),
        )
        self._eval_spec_store.put(spec)
        return spec_wire(spec)

    def eval_spec_get(self, spec_id: str) -> dict[str, Any]:
        """``GET /v1/evals/{id}`` twin — KeyError on unknown ids."""
        from fx1.serve.evals import spec_wire  # noqa: PLC0415

        spec = self._eval_spec_store.get(spec_id)
        if spec is None:
            raise KeyError(spec_id)
        return spec_wire(spec)

    def eval_specs(self, *, limit: int = 20, after: str | None = None) -> list[dict[str, Any]]:
        """``GET /v1/evals`` twin — newest-first page."""
        from fx1.serve.evals import spec_wire  # noqa: PLC0415

        page, _more = self._eval_spec_store.list_specs(limit=limit, after=after)
        return [spec_wire(s) for s in page]

    def eval_spec_update(
        self,
        spec_id: str,
        *,
        name: str | None = None,
        metadata: dict[str, str] | None = None,
        data_source_config: dict[str, Any] | None = None,
        testing_criteria: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """``POST /v1/evals/{id}`` twin — name/metadata edits always;
        ``data_source_config``/``testing_criteria`` hot-reload the spec's
        declared shape ONLY while no run binds it — the wire's 409
        ``eval_spec_frozen`` arrives in-process as ``RuntimeError``."""
        from fx1.serve.api import EvalSpecCriterion, EvalSpecDataSource  # noqa: PLC0415
        from fx1.serve.evals import spec_wire  # noqa: PLC0415

        spec = self._eval_spec_store.get(spec_id)
        if spec is None:
            raise KeyError(spec_id)
        if (
            name is None
            and metadata is None
            and data_source_config is None
            and testing_criteria is None
        ):
            raise ValueError("update must carry a field")
        if data_source_config is not None or testing_criteria is not None:
            _page, bound = self._eval_store.list_records(spec=spec_id, limit=1)
            if bound:
                raise RuntimeError(
                    f"eval '{spec_id}' is bound to {bound} run(s) — "
                    "datasource/criteria are frozen evidence"
                )
            if data_source_config is not None:
                spec.data_source_config = EvalSpecDataSource.model_validate(
                    data_source_config
                ).model_dump(exclude_none=True)
            if testing_criteria is not None:
                spec.testing_criteria = [
                    EvalSpecCriterion.model_validate(c).model_dump(exclude_none=True)
                    for c in testing_criteria
                ]
        if name is not None:
            spec.name = name
        if metadata is not None:
            spec.metadata = dict(metadata)
        self._eval_spec_store.update(spec)
        return spec_wire(spec)

    def eval_spec_delete(self, spec_id: str) -> None:
        """``DELETE /v1/evals/{id}`` twin — journaled tombstone; runs
        under it stay readable (evidence is never spec-cascaded)."""
        if self._eval_spec_store.delete(spec_id) is None:
            raise KeyError(spec_id)

    def eval_run_create(
        self,
        spec_id: str,
        *,
        model: str,
        model_fn: Callable[[list[dict[str, str]]], str] | None = None,
        byok: dict[str, str] | None = None,
        judge_byok: dict[str, str] | None = None,
        data_source_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The ``POST /v1/evals/{id}/runs`` twin — synchronous in process
        (returns the terminal run object). ``model`` maps to the backend
        chain head: a link name, ``fx1``, or a registered ``ft:`` name;
        ``model_fn`` is the weights-direct leg."""
        from fx1.serve.evals import EvalSpecItemSchema, run_wire  # noqa: PLC0415

        spec = self._eval_spec_store.get(spec_id)
        if spec is None:
            raise KeyError(spec_id)
        schema = EvalSpecItemSchema.model_validate(spec.data_source_config["item_schema"])
        if data_source_overrides:
            schema = EvalSpecItemSchema.model_validate(
                {**schema.model_dump(), **data_source_overrides}
            )
        backend, ckpt = self._eval_model_backend(model)
        record = self.run_eval(
            schema.suite,
            model_fn=model_fn,
            backend=backend,
            checkpoint_dir=ckpt or schema.checkpoint_dir,
            byok=byok,
            timeout_s=schema.timeout_s,
            fallbacks=list(schema.fallbacks),
            judge_backend=schema.judge_backend,
            judge_byok=judge_byok,
            seed=schema.seed,
        )
        record.eval_spec = spec.spec_id
        record.eval_model = model
        self._eval_store.mark(record)
        return run_wire(record)

    def _eval_model_backend(self, model: str) -> tuple[str, str | None]:
        """The wire's ``model``→backend mapping: link names pass through,
        ``fx1`` is the base checkpoint, ``ft:`` resolves via the card
        registry (unregistered → KeyError, the wire's 404)."""
        if model in ("hosted_k3", "local_fx1", "byok"):
            return model, None
        if model == "fx1":
            return "local_fx1", None
        if model.startswith("ft:"):
            ckpt = self._ft_store.checkpoint_for(model)
            if ckpt is None:
                raise KeyError(model)
            return "local_fx1", ckpt
        raise ValueError(f"unknown eval model {model!r}")

    def eval_runs(self, spec_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """``GET /v1/evals/{id}/runs`` twin — newest-first."""
        from fx1.serve.evals import run_wire  # noqa: PLC0415

        if self._eval_spec_store.get(spec_id) is None:
            raise KeyError(spec_id)
        page, _total = self._eval_store.list_records(spec=spec_id, limit=limit)
        return [run_wire(r) for r in page]

    def eval_run_get(self, spec_id: str, run_id: str) -> dict[str, Any]:
        """``GET /v1/evals/{id}/runs/{run_id}`` twin."""
        from fx1.serve.evals import run_wire  # noqa: PLC0415

        rec = self._eval_store.get(run_id.removeprefix("evalrun_"))
        if rec is None or rec.eval_spec != spec_id:
            raise KeyError(run_id)
        return run_wire(rec)

    def eval_run_items(
        self, spec_id: str, run_id: str, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        """``.../output_items`` twin — per-task verdict rows verbatim."""
        from fx1.serve.evals import report_task_items  # noqa: PLC0415

        rec = self._eval_store.get(run_id.removeprefix("evalrun_"))
        if rec is None or rec.eval_spec != spec_id:
            raise KeyError(run_id)
        tasks = (
            report_task_items(rec.report)
            if rec.status == "succeeded" and isinstance(rec.report, dict)
            else []
        )
        return [
            {
                "id": f"evalrun_{rec.eval_id}-{i}",
                "object": "eval.run.output_item",
                "run_id": f"evalrun_{rec.eval_id}",
                "eval_id": spec_id,
                "created_at": int(rec.finished_at or rec.created_at),
                "status": "pass" if t["passed"] else "fail",
                "datasource_item_id": t["name"],
                "datasource_item": t["row"],
                "results": [{"name": rec.suite, "passed": t["passed"]}],
            }
            for i, t in enumerate(tasks[:limit])
        ]

    def eval_run_delete(self, spec_id: str, run_id: str) -> None:
        """``DELETE .../runs/{run_id}`` twin — terminal records only."""
        rec = self._eval_store.get(run_id.removeprefix("evalrun_"))
        if rec is None or rec.eval_spec != spec_id:
            raise KeyError(run_id)
        if rec.status in ("queued", "running"):
            raise RuntimeError(f"eval {rec.eval_id} is {rec.status} — only terminal runs delete")
        self._eval_store.delete(rec.eval_id)

    # ---- fine-tuning (the /v1/fine_tuning twin) ------------------------

    def create_finetune_job(
        self,
        *,
        model: str,
        training_jsonl: bytes | str,
        validation_jsonl: bytes | str | None = None,
        hyperparameters: FTHyperparameters | None = None,
        suffix: str | None = None,
        seed: int | None = None,
        metadata: dict[str, str] | None = None,
        callback_url: str | None = None,
        callback_secret: str | None = None,
    ) -> FTJob:
        """The ``POST /v1/fine_tuning/jobs`` surface, weights-direct —
        synchronous in process (there is no queue to observe): the corpus
        is raw JSONL content here (the wire's file-upload step is an HTTP
        artifact), the runner runs inline, and the returned job is
        terminal. The record lands in the in-process store — the same
        FTJobStore the wire serves — for ``finetune_job``/events reads.

        ``callback_url``/``callback_secret`` are honored the same as the
        wire: the terminal record POSTs to the URL, HMAC-signed via the
        X-Fx1-Webhook-* headers when the secret is set (still real HTTP —
        in-process doesn't mean silent).

        ``ValueError`` is the wire's 400; a runner exception yields a
        ``failed`` job record (never a raise — the job's own verdict is
        the honest outcome).
        """
        if callback_secret is not None and not callback_url:
            raise ValueError("callback_secret requires callback_url")
        if callback_url is not None:
            from fx1.serve.webhooks import check_callback_url  # noqa: PLC0415

            check_callback_url(callback_url)
        if model not in TRAINABLE_MODELS:
            raise ValueError(
                f"model {model!r} is not trainable through the harness "
                f"(trainable: {list(TRAINABLE_MODELS)})"
            )
        corpus = (
            training_jsonl.encode() if isinstance(training_jsonl, str) else bytes(training_jsonl)
        )
        n = validate_chat_jsonl(corpus, file_id="<in-process>")
        hp = hyperparameters or FTHyperparameters()
        job = FTJob(
            id=f"ftjob-{uuid.uuid4().hex}",
            model=model,
            created_at=int(time.time()),
            status="running",
            hyperparameters=hp,
            seed=seed,
            metadata=metadata,
            user_provided_suffix=suffix,
            callback_url=callback_url,
            # The wire echoes training_file=the file id; in-process carries
            # the corpus content-length as the honest handle.
            training_file=f"inline:{len(corpus)}B",
        )
        job._callback_secret = callback_secret
        ft_name = f"ft:{model}:{suffix or 'job'}:{job.id.split('-', 1)[1][:12]}"
        work_dir = self._ft_dir / job.id
        work_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = work_dir / "corpus.jsonl"
        corpus_path.write_bytes(corpus)
        val_path = None
        if validation_jsonl is not None:
            vbytes = (
                validation_jsonl.encode()
                if isinstance(validation_jsonl, str)
                else bytes(validation_jsonl)
            )
            validate_chat_jsonl(vbytes, file_id="<validation>")
            val_path = work_dir / "validation.jsonl"
            val_path.write_bytes(vbytes)
        entry = self._ft_store.put(job, None, "")
        self._ft_store.add_event(job.id, "info", f"training corpus validated: {n} examples")
        spec = FTJobSpec(
            job_id=job.id,
            model=model,
            corpus_path=corpus_path,
            val_path=val_path,
            hyperparameters=hp.model_dump(mode="json"),
            seed=seed if seed is not None else 17,
            work_dir=work_dir,
            ft_model_name=ft_name,
        )
        self._ft_store.add_event(job.id, "info", "job started")
        try:
            # pause_gate is opt-in on the runner contract — the in-process
            # twin is synchronous so there is no pause window, but the
            # gate still reports a pre-dispatch cancel to gate-aware
            # runners; older runners without the kwarg keep working
            _extra: dict[str, Any] = {}
            try:
                if "pause_gate" in inspect.signature(self._ft_runner).parameters:
                    _extra["pause_gate"] = entry.cancel.is_set
            except (TypeError, ValueError):
                _extra["pause_gate"] = entry.cancel.is_set
            outcome = self._ft_runner(
                spec,
                emit=lambda level, message, data=None: self._ft_store.add_event(
                    job.id, level, message, data
                ),
                should_cancel=entry.cancel.is_set,
                **_extra,
            )
            if entry.cancel.is_set():
                job.status = "cancelled"
            else:
                # The wire worker registers runner artifacts back into
                # the files store so GET /v1/files/{id}/content downloads
                # them; the in-process twin lands them in the same
                # {job.id}-{name} shape under purpose fine-tune-result.
                for name, path in outcome.artifacts.items():
                    try:
                        content = Path(path).read_bytes()
                    except OSError:
                        continue
                    file_id = f"file-{uuid.uuid4().hex}"
                    fobj = {
                        "id": file_id,
                        "object": "file",
                        "purpose": "fine-tune-result",
                        "filename": f"{job.id}-{Path(path).name}",
                        "bytes": len(content),
                        "created_at": int(time.time()),
                        "status": "processed",
                    }
                    with self._files_lock:
                        self._files[file_id] = {**fobj, "_content": content}
                    job.result_files.append(file_id)
                    self._ft_store.add_event(
                        job.id,
                        "info",
                        f"result artifact registered: {name}",
                        {"file_id": file_id, "path": str(path)},
                    )
                job.fine_tuned_model = outcome.fine_tuned_model
                job.trained_tokens = outcome.trained_tokens
                job.status = "succeeded"
                if outcome.fine_tuned_model is not None and outcome.checkpoint:
                    self._ft_store.register_model(
                        outcome.fine_tuned_model,
                        job_id=job.id,
                        checkpoint=outcome.checkpoint,
                        created=int(time.time()),
                    )
        except Exception as exc:
            if entry.cancel.is_set():
                job.status = "cancelled"
            else:
                job.status = "failed"
                job.error = FTJobError(code="job_failed", message=f"{type(exc).__name__}: {exc}")
        job.finished_at = int(time.time())
        self._ft_store.add_event(
            job.id, "info" if job.status == "succeeded" else "error", f"job {job.status}"
        )
        if job.callback_url:
            from fx1.serve.webhooks import deliver_signed  # noqa: PLC0415

            ok, err, attempts = deliver_signed(
                job.callback_url, callback_secret, job.model_dump_json().encode()
            )
            job.callback_status = "delivered" if ok else "failed"
            job.callback_error = None if ok else err
            job.callback_attempts = attempts
        # Terminal state + callback verdicts ride the journal — same
        # ordering as the wire worker (deliver, then mark).
        self._ft_store.mark(entry)
        return job

    def finetune_jobs(self, *, limit: int | None = None) -> list[FTJob]:
        """In-process fine-tune records, newest-first — the wire twin is
        ``GET /v1/fine_tuning/jobs``."""
        jobs, _has_more = self._ft_store.list_jobs(limit=limit or 100, after=None)
        return jobs

    def finetune_job(self, job_id: str) -> FTJob:
        """One job record — ``KeyError`` on unknown ids (the wire's 404)."""
        entry = self._ft_store.get(job_id)
        if entry is None:
            raise KeyError(job_id)
        return entry.job

    def finetune_job_events(self, job_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        """The job's event feed, oldest-first — the wire twin is
        ``GET /v1/fine_tuning/jobs/{id}/events``."""
        events, _has_more = self._ft_store.list_events(job_id, limit=limit or 100, after=None)
        return [e.model_dump(mode="json") for e in events]

    def finetune_job_checkpoints(
        self, job_id: str, *, limit: int | None = None
    ) -> FTJobCheckpointList:
        """The job's registered checkpoint artifacts, oldest-first — the
        wire twin is ``GET /v1/fine_tuning/jobs/{id}/checkpoints``.
        ``KeyError`` on unknown ids; a deleted ``ft:`` name drops off
        (the tombstone is real)."""
        if self._ft_store.get(job_id) is None:
            raise KeyError(job_id)
        items, has_more = self._ft_store.checkpoints_for(job_id, limit=limit or 100, after=None)
        return FTJobCheckpointList(
            data=items,
            first_id=items[0].id if items else None,
            last_id=items[-1].id if items else None,
            has_more=has_more,
        )

    # ---- registry ------------------------------------------------------

    def commands(self, role: HarnessRole | None = None) -> list[str]:
        """Registered command names; anything unlisted is unreachable."""
        return [c.name for c in self._harness.list_commands(role)]

    def command(self, name: str) -> HarnessCommand:
        """The registered ``HarnessCommand`` (KeyError on unknown names)."""
        return self._harness.get(name)

    # ---- execution -----------------------------------------------------

    def run(
        self,
        command: str,
        extra_args: list[str] | None = None,
        *,
        config: Path | str | None = None,
    ) -> HarnessResult:
        """Execute a registered lab command (fail-closed on unknown names)."""
        return self._harness.run(
            command,
            extra_args,
            config=Path(config) if config is not None else None,
        )

    def run_receipt(self, result: HarnessResult) -> dict[str, Any]:
        """Seal an in-process run's outcome as ``fx1_run_result.v1`` —
        stdout/stderr digested, never content. The wire twin lives under
        ``GET /harness/jobs/{id}/receipt`` where the digested run record
        embeds in the job record."""
        from fx1.serve.ops_receipt import run_result_receipt  # noqa: PLC0415

        return run_result_receipt(result.model_dump())

    # ---- gated completion ----------------------------------------------

    def check_text(self, text: str) -> GateCheckResult:
        """Pre-flight text through the honesty gate in-process — never
        raises ``Fx1HonestyError``; the refusal rides ``ok=False, error``.
        Argument faults (non-str) propagate."""
        try:
            validate_fx1_output(text)
        except Fx1HonestyError as exc:
            return GateCheckResult(ok=False, error=str(exc))
        return GateCheckResult(ok=True)

    def score(self, input: str | list[str]) -> list[dict[str, Any]]:  # noqa: A002
        """Score text through the deterministic reward contract in-process
        — same breakdown shape as ``POST /harness/score`` on the wire:
        ``{object:"score", index, total, components, violations}`` per
        input. Argument faults raise ``ValueError``."""
        from fx1.reward import score_response  # noqa: PLC0415
        from fx1.serve.api import ScoreRequest  # noqa: PLC0415

        validated = ScoreRequest(input=input).input
        items = [validated] if isinstance(validated, str) else list(validated)
        return [
            {
                "object": "score",
                "index": i,
                "total": bd.total,
                "components": dict(bd.components),
                "violations": list(bd.violations),
            }
            for i, bd in enumerate(score_response(t) for t in items)
        ]

    def moderate(self, input: str | list[str]) -> dict[str, Any]:  # noqa: A002
        """Classify text through the honesty gate in-process — the
        OpenAI-moderations wire shape (``POST /v1/moderations``): one
        ``{flagged, categories, category_scores, category_applied_input_types}``
        result per input, plus the content-derived ``modr-<sha256>`` id the
        wire returns for identical inputs. Argument faults raise
        ``ValidationError``."""
        from fx1.serve.api import ModerationRequest  # noqa: PLC0415

        validated = ModerationRequest(input=input).input
        items = [validated] if isinstance(validated, str) else list(validated)
        results = []
        for text in items:
            categories = honesty_categories(text)
            results.append(
                {
                    "flagged": any(categories.values()),
                    "categories": categories,
                    "category_scores": {
                        name: 1.0 if hit else 0.0 for name, hit in categories.items()
                    },
                    "category_applied_input_types": {name: ["text"] for name in categories},
                }
            )
        digest = hashlib.sha256("\x1e".join(items).encode("utf-8")).hexdigest()
        return {
            "id": f"modr-{digest[:24]}",
            "model": "fx1-honesty-gate",
            "results": results,
        }

    def probe_backend(
        self,
        backend: str,
        *,
        checkpoint_dir: str | Path | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        byok: dict[str, str] | None = None,
        timeout_s: float | None = None,
        prompt: str = "ping",
    ) -> ProbeResult:
        """Deep health: run one minimal gated completion through the real
        resolver — same checkpoint/BYOK plumbing as ``complete``, so a
        BYOK probe tests the caller's own endpoint. Never raises for
        backend-side faults (that IS the verdict); argument faults raise
        as usual."""
        backend_obj = self._resolve_completion_backend(
            backend, checkpoint_dir, backend_kwargs, byok, timeout_s
        )
        t0 = time.monotonic()
        ok = False
        error: str | None = None
        error_class: str | None = None
        try:
            content = backend_obj.complete([{"role": "user", "content": prompt}])
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
            closer = getattr(backend_obj, "close", None)
            if callable(closer):
                closer()
        model_name = getattr(backend_obj, "_model", None)
        return ProbeResult(
            backend=backend,
            ok=ok,
            model=model_name if isinstance(model_name, str) else None,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            error=error,
            error_class=error_class,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        backend: str = "local_fx1",
        checkpoint_dir: str | Path | None = None,
        receipt_hashes: list[str] | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        byok: dict[str, str] | None = None,
        timeout_s: float | None = None,
        fallbacks: list[str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        stop: list[str] | str | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, int] | None = None,
        reasoning_effort: str | None = None,
        service_tier: str | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        verbosity: str | None = None,
        user: str | None = None,
        metadata: dict[str, str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        _served_model: str | None = None,
    ) -> CompletionResult:
        """One chat completion through the honesty gate.

        ``backend`` is one of ``hosted_k3`` / ``local_fx1`` / ``byok``.
        ``local_fx1`` requires ``checkpoint_dir`` here or
        ``FX1_CHECKPOINT_DIR`` in the environment; it may only be passed for
        ``local_fx1``. The backend is always closed afterwards — engines
        spawned by ``LocalFx1Backend`` never leak.

        ``fallbacks`` is an ordered chain of alternate backends tried after
        the primary — only on availability faults (unconfigured /
        transport). A gate refusal or a client error aborts the request;
        a refusal is a verdict, not a reason to spend another backend's
        capacity. Per-link kwargs: ``byok`` binds only a ``'byok'`` link,
        ``checkpoint_dir`` only a ``'local_fx1'`` link.

        ``temperature``/``top_p``/``max_tokens``/``seed`` are declared
        decode params — only set fields reach the wire beyond temperature
        (default 0.0 keeps eval runs deterministic); the resolved set is
        recorded on the completion record. ``stop`` truncates the output
        at the earliest match (harness-enforced — providers that ignore
        it still ship the cut text); the penalty pair, ``logit_bias``,
        and the provider hints pass through verbatim. ``user`` and
        ``metadata`` stamp the completion record for attribution.

        ``tools``/``tool_choice``/``parallel_tool_calls`` carry the
        OpenAI function-calling surface — function specs, the call
        policy, the parallel flag — verbatim to tool-capable links.
        ``logprobs``/``top_logprobs`` ask the provider for its
        ``choices[].logprobs`` payload — echoed verbatim on the result
        (``None`` under provider silence). A link without
        ``complete_with_tools`` answers ``NotImplementedError``, never a
        silently dropped tool or logprobs intent. Tool-call *history*
        (assistant ``tool_calls`` entries, ``role: 'tool'`` results)
        passes through the message list itself. The honesty gate reads
        the assistant's text only — tool arguments are machine-bound
        JSON and logprobs are provider scores, not claims.

        ``_served_model`` is the internal registry alias that resolved the request
        — an ``ft:`` name set by the OpenAI/Anthropic translators when
        the ft registry routed the call — so the result and the
        completion record name what the caller addressed rather than
        the checkpoint's internal version stamp.
        """
        if top_logprobs is not None and not logprobs:
            raise ValueError("top_logprobs requires logprobs=True")
        if top_logprobs is not None and not 0 <= top_logprobs <= 20:
            raise ValueError(f"top_logprobs must be 0..20, got {top_logprobs}")
        chain = _fallback_chain(backend, fallbacks)
        _check_link_kwargs(chain, checkpoint_dir, byok)
        sampling = _sampling_params(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            reasoning_effort=reasoning_effort,
            service_tier=service_tier,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
            verbosity=verbosity,
            user=user,
        )
        sampling_fields = sampling.body_fields()
        prompt_sha256 = _messages_sha256(messages)
        attempts: list[dict[str, Any]] = []
        last_exc: Exception | None = None
        for cand in chain:
            t0 = time.monotonic()
            try:
                backend_obj = self._resolve_link(
                    cand, checkpoint_dir, backend_kwargs, byok, timeout_s
                )
            except (BackendNotConfiguredError, RuntimeError, ValueError) as exc:
                # resolve-level availability fault — same retriable class
                # the wire treats as backend_unavailable (503).
                attempts.append(
                    {
                        "backend": cand,
                        "ok": False,
                        "error_class": type(exc).__name__,
                        "latency_ms": (time.monotonic() - t0) * 1000.0,
                    }
                )
                last_exc = exc
                continue
            except Exception as exc:
                # client errors abort — an unknown name, a bad receipt is
                # the request's fault, not the backend's.
                attempts.append(
                    {
                        "backend": cand,
                        "ok": False,
                        "error_class": type(exc).__name__,
                        "latency_ms": (time.monotonic() - t0) * 1000.0,
                    }
                )
                self._record_call(
                    cand,
                    None,
                    False,
                    (time.monotonic() - t0) * 1000.0,
                    None,
                    str(exc),
                    type(exc).__name__,
                    prompt_sha256,
                    None,
                    tuple(attempts),
                    sampling_fields,
                    user,
                    metadata,
                )
                raise
            t0 = time.monotonic()
            tool_calls_out: tuple[dict[str, Any], ...] | None = None
            logprobs_out: dict[str, Any] | None = None
            finish_out: str | None = None
            try:
                wants_tools = (
                    tools is not None
                    or logprobs
                    or top_logprobs is not None
                    or any(m.get("role") == "tool" or m.get("tool_calls") for m in messages)
                )
                if wants_tools:
                    tool_result = cited_complete_tools(
                        backend_obj,
                        messages,
                        receipt_hashes=receipt_hashes,
                        sampling=sampling,
                        tools=tools,
                        tool_choice=tool_choice,
                        parallel_tool_calls=parallel_tool_calls,
                        logprobs=logprobs,
                        top_logprobs=top_logprobs,
                    )
                    content = tool_result.content or ""
                    tool_calls_out = tool_result.tool_calls
                    logprobs_out = tool_result.logprobs
                    finish_out = tool_result.finish_reason
                else:
                    content = cited_complete(
                        backend_obj,
                        messages,
                        receipt_hashes=receipt_hashes,
                        sampling=sampling,
                    )
            except (BackendNotConfiguredError, RuntimeError) as exc:
                # availability fault — record the link, try the next.
                attempts.append(
                    {
                        "backend": cand,
                        "ok": False,
                        "error_class": type(exc).__name__,
                        "latency_ms": (time.monotonic() - t0) * 1000.0,
                    }
                )
                last_exc = exc
                closer = getattr(backend_obj, "close", None)
                if callable(closer):
                    closer()
                continue
            except Exception as exc:
                # refusals / capability gaps / client errors abort the chain.
                attempts.append(
                    {
                        "backend": cand,
                        "ok": False,
                        "error_class": type(exc).__name__,
                        "latency_ms": (time.monotonic() - t0) * 1000.0,
                    }
                )
                self._record_call(
                    cand,
                    getattr(backend_obj, "_model", None)
                    if isinstance(getattr(backend_obj, "_model", None), str)
                    else None,
                    False,
                    (time.monotonic() - t0) * 1000.0,
                    getattr(backend_obj, "last_usage", None)
                    if isinstance(getattr(backend_obj, "last_usage", None), dict)
                    else None,
                    str(exc),
                    type(exc).__name__,
                    prompt_sha256,
                    None,
                    tuple(attempts),
                    sampling_fields,
                    user,
                    metadata,
                )
                closer = getattr(backend_obj, "close", None)
                if callable(closer):
                    closer()
                raise
            model_name = (
                _served_model
                if _served_model is not None and cand == backend
                else getattr(backend_obj, "_model", None)
            )
            usage = getattr(backend_obj, "last_usage", None)
            attempts.append(
                {
                    "backend": cand,
                    "ok": True,
                    "latency_ms": (time.monotonic() - t0) * 1000.0,
                }
            )
            cid = self._record_call(
                cand,
                model_name if isinstance(model_name, str) else None,
                True,
                (time.monotonic() - t0) * 1000.0,
                usage if isinstance(usage, dict) else None,
                None,
                None,
                prompt_sha256,
                hashlib.sha256(
                    (
                        content
                        + (
                            "\n" + json.dumps(list(tool_calls_out), sort_keys=True)
                            if tool_calls_out
                            else ""
                        )
                        + ("\n" + json.dumps(logprobs_out, sort_keys=True) if logprobs_out else "")
                    ).encode("utf-8")
                ).hexdigest(),
                tuple(attempts) if len(attempts) > 1 else None,
                sampling_fields,
                user,
                metadata,
            )
            closer = getattr(backend_obj, "close", None)
            if callable(closer):
                closer()
            return CompletionResult(
                backend=cand,
                model=model_name if isinstance(model_name, str) else None,
                content=content,
                usage=usage if isinstance(usage, dict) else None,
                receipt_hashes=tuple(receipt_hashes or ()),
                completion_id=cid,
                attempts=tuple(attempts) if len(attempts) > 1 else (),
                sampling=sampling_fields,
                tool_calls=tool_calls_out,
                finish_reason=finish_out,
                logprobs=logprobs_out,
            )
        assert last_exc is not None  # every link failed retriably
        self._record_call(
            backend,
            None,
            False,
            0.0,
            None,
            str(last_exc),
            type(last_exc).__name__,
            prompt_sha256,
            None,
            tuple(attempts) if len(attempts) > 1 else None,
            sampling_fields,
            user,
            metadata,
        )
        raise last_exc

    def complete_many(
        self,
        batch: list[list[dict[str, str]]],
        *,
        backend: str = "local_fx1",
        checkpoint_dir: str | Path | None = None,
        receipt_hashes: list[str] | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        byok: dict[str, str] | None = None,
        timeout_s: float | None = None,
        max_workers: int = 4,
        fallbacks: list[str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        stop: list[str] | str | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, int] | None = None,
        reasoning_effort: str | None = None,
        service_tier: str | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        verbosity: str | None = None,
        user: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> list[CompletionResult]:
        """Many gated completions over ONE shared backend instance.

        The backend resolves once and is shared across worker threads — a
        spawned local engine serves the whole batch, not one spawn per
        item (``LocalFx1Backend``'s spawn path is lock-guarded). Results
        come back in submission order; the lowest-index failure propagates
        after the pool drains. The backend is always closed afterwards.
        """
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        if not batch:
            return []
        # the batch surface is text-only like the wire's /v1/batches —
        # a tool-context line fails the request, never silently degrades
        for i, msgs in enumerate(batch):
            for j, msg in enumerate(msgs):
                if msg.get("role") == "tool" or "tool_calls" in msg:
                    raise ValueError(
                        f"batch[{i}][{j}] carries tool context — the batch "
                        "surface is text-only; call complete() per turn instead"
                    )
        serving, backend_obj = self._resolve_chain(
            backend, fallbacks, checkpoint_dir, backend_kwargs, byok, timeout_s
        )
        sampling = _sampling_params(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            reasoning_effort=reasoning_effort,
            service_tier=service_tier,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
            verbosity=verbosity,
            user=user,
        )
        sampling_fields = sampling.body_fields()
        model_name = getattr(backend_obj, "_model", None)
        model_str = model_name if isinstance(model_name, str) else None

        def _one(msgs: list[dict[str, str]]) -> tuple[str, str]:
            t0 = time.monotonic()
            p_sha = _messages_sha256(msgs)
            try:
                content = cited_complete(
                    backend_obj, msgs, receipt_hashes=receipt_hashes, sampling=sampling
                )
            except Exception as exc:
                self._record_call(
                    serving,
                    model_str,
                    False,
                    (time.monotonic() - t0) * 1000.0,
                    None,
                    str(exc),
                    type(exc).__name__,
                    p_sha,
                    None,
                    None,
                    sampling_fields,
                    user,
                    metadata,
                )
                raise
            cid = self._record_call(
                serving,
                model_str,
                True,
                (time.monotonic() - t0) * 1000.0,
                None,
                None,
                None,
                p_sha,
                hashlib.sha256(content.encode("utf-8")).hexdigest(),
                None,
                sampling_fields,
                user,
                metadata,
            )
            return content, cid

        try:
            with ThreadPoolExecutor(
                max_workers=min(max_workers, len(batch)), thread_name_prefix="fx1-complete"
            ) as pool:
                done = list(pool.map(_one, batch))
        finally:
            closer = getattr(backend_obj, "close", None)
            if callable(closer):
                closer()
        return [
            CompletionResult(
                backend=serving,
                model=model_str,
                content=content,
                receipt_hashes=tuple(receipt_hashes or ()),
                completion_id=cid,
                sampling=sampling_fields,
            )
            for content, cid in done
        ]

    # ---- receipts --------------------------------------------------------

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        *,
        backend: str = "local_fx1",
        checkpoint_dir: str | Path | None = None,
        receipt_hashes: list[str] | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        byok: dict[str, str] | None = None,
        timeout_s: float | None = None,
        fallbacks: list[str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        stop: list[str] | str | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, int] | None = None,
        reasoning_effort: str | None = None,
        service_tier: str | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        verbosity: str | None = None,
        user: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> list[str]:
        """Token-delta chunks of one gated completion.

        Mirrors ``POST /harness/complete/stream``: the backend's SSE deltas
        are buffered, the joined text passes the honesty gate, and the
        chunk list is returned (with the evidence footer appended as a
        final chunk when ``receipt_hashes`` is given). Buffer-then-gate is
        the contract — an in-process consumer never holds ungated bytes
        either. A backend without ``stream`` raises ``NotImplementedError``
        (501-class); the backend is always closed afterwards.
        ``fallbacks`` applies at resolve level only — once a link is
        streaming there is no honest restart point.
        """
        serving, backend_obj = self._resolve_chain(
            backend, fallbacks, checkpoint_dir, backend_kwargs, byok, timeout_s
        )
        sampling = _sampling_params(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            reasoning_effort=reasoning_effort,
            service_tier=service_tier,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
            verbosity=verbosity,
            user=user,
        )
        sampling_fields = sampling.body_fields()
        t0 = time.monotonic()
        prompt_sha256 = _messages_sha256(messages)
        try:
            if not isinstance(backend_obj, StreamingBackend):
                raise NotImplementedError(f"backend {serving!r} does not support streaming")
            chunks = list(backend_obj.stream(messages, sampling=sampling))
            joined = "".join(chunks)
            validate_fx1_output(joined)
            # stop-sequence cut after the gate — a gated prefix stays gated
            chunks = truncate_chunks(chunks, sampling.stop)
            joined = "".join(chunks)
            if receipt_hashes:
                chunks.append(
                    "\n\nEvidence: "
                    + ", ".join(f"`{h[:16]}…`" for h in receipt_hashes)
                    + " — verify with `dipcatcher verify-research`."
                )
        except Exception as exc:
            self._record_call(
                serving,
                getattr(backend_obj, "_model", None)
                if isinstance(getattr(backend_obj, "_model", None), str)
                else None,
                False,
                (time.monotonic() - t0) * 1000.0,
                getattr(backend_obj, "last_usage", None)
                if isinstance(getattr(backend_obj, "last_usage", None), dict)
                else None,
                str(exc),
                type(exc).__name__,
                prompt_sha256,
                None,
                None,
                sampling_fields,
                user,
                metadata,
            )
            raise
        finally:
            closer = getattr(backend_obj, "close", None)
            if callable(closer):
                closer()
        model_name = getattr(backend_obj, "_model", None)
        usage = getattr(backend_obj, "last_usage", None)
        self._record_call(
            serving,
            model_name if isinstance(model_name, str) else None,
            True,
            (time.monotonic() - t0) * 1000.0,
            usage if isinstance(usage, dict) else None,
            None,
            None,
            prompt_sha256,
            hashlib.sha256(joined.encode("utf-8")).hexdigest(),
            None,
            sampling_fields,
            user,
            metadata,
        )
        return chunks

    def openai_models(self) -> OpenAIModelList:
        """The ``GET /v1/models`` inventory in-process — `fx1` plus the
        backend names an OpenAI ``model`` field may carry."""
        refs = self._ft_store.models()
        return openai_models(
            extra_ids=[m["id"] for m in refs],
            created_by_id={str(m["id"]): int(m["created"]) for m in refs},
        )

    def openai_model(self, model_id: str) -> OpenAIModel:
        """``GET /v1/models/{id}`` in-process — one card for a listed id;
        unknown ids raise :class:`OpenAICompatError` (a ``ValueError``),
        the SDK's request-error class."""
        refs = self._ft_store.models()
        return openai_model(
            model_id,
            extra_ids=[m["id"] for m in refs],
            created_by_id={str(m["id"]): int(m["created"]) for m in refs},
        )

    def openai_delete_model(self, model_id: str) -> OpenAIModelDelete:
        """``DELETE /v1/models/{id}`` in-process — unregister an ``ft:``
        name. Built-in link ids and unregistered names raise
        :class:`OpenAICompatError` carrying the wire's status/code; a
        deleted registration never resolves again."""
        if model_id in OPENAI_MODEL_IDS:
            raise OpenAICompatError(
                f"the built-in link '{model_id}' is not deletable",
                code="invalid_request",
            )
        if self._ft_store.unregister_model(model_id) is None:
            raise OpenAICompatError(
                f"The model '{model_id}' does not exist",
                status=404,
                code="model_not_found",
            )
        return OpenAIModelDelete(id=model_id)

    def openai_chat(
        self,
        request: OpenAIChatRequest | dict[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[OpenAIChatResponse, str | None]:
        """One OpenAI-surface chat completion, weights-direct.

        ``request`` is the same body ``POST /v1/chat/completions`` takes —
        a dict or a parsed :class:`OpenAIChatRequest`. ``headers`` accepts
        the wire's ``X-Fx1-*`` knobs (backend/byok/checkpoint/fallbacks)
        for callers porting a header-based integration; the ``fx1``
        extension object covers the same ground in the body. Validation
        and backend precedence come from ``fx1.serve.openai_compat`` —
        the wire's own translation layer — so this path cannot drift from
        ``/v1``: same fail-closed surface, same gate, same metering and
        completion log.

        Returns the ``chat.completion`` envelope (``id`` mints the
        ``chatcmpl-`` handle; ``system_fingerprint`` is the serving
        backend) plus the completion-log id for receipt lookup — ``None``
        when the log dropped it.
        """
        body = (
            request
            if isinstance(request, OpenAIChatRequest)
            else OpenAIChatRequest.model_validate(request)
        )
        kwargs = openai_to_kwargs(
            body, dict(headers or {}), ft_resolver=self._ft_store.checkpoint_for
        )
        # n>1 fans out into n gated calls — each choice gets its own
        # honesty-gate pass, format check, and completion-log record.
        results = [self.complete(**kwargs) for _ in range(body.n)]
        contents: list[str] = []
        choice_calls: list[list[dict[str, Any]] | None] = []
        choice_reasons: list[str] = []
        choice_lps: list[dict[str, Any] | None] = []
        usage_sum: dict[str, int] = {}
        usage_seen = False
        for result in results:
            validate_openai_output(body, result.content)
            contents.append(result.content)
            choice_calls.append(list(result.tool_calls) if result.tool_calls is not None else None)
            choice_reasons.append(result.finish_reason or "stop")
            choice_lps.append(result.logprobs)
            if isinstance(result.usage, dict):
                usage_seen = True
                for uk, uv in result.usage.items():
                    if isinstance(uv, int):
                        usage_sum[uk] = usage_sum.get(uk, 0) + uv
        first = results[0]
        cid = first.completion_id or uuid.uuid4().hex
        envelope = openai_envelope(
            cid=cid,
            content=contents if body.n > 1 else contents[0],
            backend="+".join(dict.fromkeys(r.backend for r in results)),
            model=first.model,
            usage=usage_sum if usage_seen else None,
            tool_calls=(choice_calls if any(c is not None for c in choice_calls) else None),
            finish_reasons=choice_reasons,
            logprobs=choice_lps,
            metadata=body.metadata,
        )
        if body.store is not False:
            self._openai_store.put(
                envelope,
                items={
                    "messages": chat_messages_for_store(
                        body.messages, envelope_id=str(envelope["id"])
                    )
                },
            )
        return OpenAIChatResponse.model_validate(envelope), first.completion_id

    def openai_chat_stream(
        self,
        request: OpenAIChatRequest | dict[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
        last_event_id: int | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """The ``stream: true`` surface in-process — chunk payloads.

        Buffer-then-gate like ``stream_complete``: the completion runs
        the gated pipeline, then the joined text chunks through the same
        ``openai_chunks`` generator the wire serializes — a caller gets
        the identical ``chat.completion.chunk`` sequence (role delta,
        ~64-char whitespace deltas, ``finish_reason`` ``stop``, optional
        ``choices: []``+``usage`` when ``stream_options.include_usage``),
        minus the ``data:``/``[DONE]`` framing.

        ``last_event_id`` applies the same sequence filter the wire's
        ``Last-Event-ID`` resume uses — only chunks above that index are
        returned (the wire adds SSE ``id:`` fields equal to the chunk
        index; in-process there is no transport to resume, this is the
        replay/filter parity surface).

        Returns ``(chunks, completion_id)`` — the id links to the
        completion log and its sealed receipt.
        """
        if last_event_id is not None and last_event_id < 0:
            raise ValueError(f"last_event_id must be >= 0, got {last_event_id}")
        body = (
            request
            if isinstance(request, OpenAIChatRequest)
            else OpenAIChatRequest.model_validate(request)
        )
        kwargs = openai_to_kwargs(
            body, dict(headers or {}), ft_resolver=self._ft_store.checkpoint_for
        )
        results = [self.complete(**kwargs) for _ in range(body.n)]
        contents: list[str] = []
        choice_calls: list[list[dict[str, Any]] | None] = []
        choice_reasons: list[str] = []
        choice_lps: list[dict[str, Any] | None] = []
        usage_sum: dict[str, int] = {}
        usage_seen = False
        for result in results:
            validate_openai_output(body, result.content)
            contents.append(result.content)
            choice_calls.append(list(result.tool_calls) if result.tool_calls is not None else None)
            choice_reasons.append(result.finish_reason or "stop")
            choice_lps.append(result.logprobs)
            if isinstance(result.usage, dict):
                usage_seen = True
                for uk, uv in result.usage.items():
                    if isinstance(uv, int):
                        usage_sum[uk] = usage_sum.get(uk, 0) + uv
        first = results[0]
        cid = first.completion_id or uuid.uuid4().hex
        if body.store is not False:
            env = openai_envelope(
                cid=cid,
                content=contents if body.n > 1 else contents[0],
                backend="+".join(dict.fromkeys(r.backend for r in results)),
                model=first.model,
                usage=usage_sum if usage_seen else None,
                tool_calls=(choice_calls if any(c is not None for c in choice_calls) else None),
                finish_reasons=choice_reasons,
                logprobs=choice_lps,
                metadata=body.metadata,
            )
            self._openai_store.put(
                env,
                items={
                    "messages": chat_messages_for_store(body.messages, envelope_id=str(env["id"]))
                },
            )
        chunks = list(
            openai_chunks(
                text=contents,
                backend="+".join(dict.fromkeys(r.backend for r in results)),
                model=first.model,
                cid=cid,
                include_usage=bool((body.stream_options or {}).get("include_usage")),
                usage=usage_sum if usage_seen else None,
                tool_calls=(choice_calls if any(c is not None for c in choice_calls) else None),
                finish_reasons=choice_reasons,
                logprobs=choice_lps,
            )
        )
        if last_event_id is not None:
            chunks = chunks[last_event_id + 1 :]
        return chunks, first.completion_id

    def anthropic_message(
        self,
        request: AnthropicMessagesRequest | dict[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[AnthropicMessageObject, str | None]:
        """One Anthropic Messages-surface call, weights-direct.

        ``request`` is the same body ``POST /v1/messages`` takes — a dict
        or a parsed :class:`AnthropicMessagesRequest`. The Anthropic
        contract translates into the shared OpenAI gated path
        (``anthropic_to_openai`` → :meth:`openai_chat`), then the
        completion envelope translates out to the ``message`` object —
        the wire's own translation layer, so this leg cannot drift from
        ``/v1/messages``: same fail-closed validation (``max_tokens``
        required, strict role alternation, unsupported knobs refuse),
        same gate, same metering, same completion log.

        Returns the ``message`` object plus the completion-log id —
        ``None`` when the log dropped it.
        """
        body = (
            request
            if isinstance(request, AnthropicMessagesRequest)
            else AnthropicMessagesRequest.model_validate(request)
        )
        env, cid = self.openai_chat(
            OpenAIChatRequest.model_validate(anthropic_to_openai(body)),
            headers=headers,
        )
        return (
            AnthropicMessageObject.model_validate(
                anthropic_envelope(env.model_dump(mode="json"), model=body.model)
            ),
            cid,
        )

    def anthropic_message_stream(
        self,
        request: AnthropicMessagesRequest | dict[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
        last_event_id: int | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """The ``stream: true`` surface in-process — Anthropic SSE event
        payloads (``{"event", "data"}`` pairs), minus the wire framing.

        Runs the same gated completion as :meth:`anthropic_message`,
        then emits the ``message_start`` → ``content_block_*`` →
        ``message_delta`` → ``message_stop`` grammar the wire serializes.
        ``last_event_id`` applies the same sequence filter the wire's
        resume uses — only events above that index are returned.

        Returns ``(events, completion_id)`` — the id links to the
        completion log and its sealed receipt.
        """
        if last_event_id is not None and last_event_id < 0:
            raise ValueError(f"last_event_id must be >= 0, got {last_event_id}")
        body = (
            request
            if isinstance(request, AnthropicMessagesRequest)
            else AnthropicMessagesRequest.model_validate(request)
        )
        env, cid = self.openai_chat(
            OpenAIChatRequest.model_validate(anthropic_to_openai(body)),
            headers=headers,
        )
        events = list(anthropic_events(env.model_dump(mode="json"), model=body.model))
        if last_event_id is not None:
            events = events[last_event_id + 1 :]
        return events, cid

    def anthropic_count_tokens(
        self,
        request: AnthropicCountTokensRequest | dict[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> int:
        """``POST /v1/messages/count_tokens`` in-process — the provider's
        own tokenizer count over the message channel, returned as the
        ``input_tokens`` int.

        Same contract as the wire: the body validates like
        ``/v1/messages`` (strict alternation, unsupported knobs refuse),
        ``tools``/``tool_choice`` raise ``ValueError`` (a provider
        ``/tokenize`` route sees only messages — counting them would
        undercount), and the resolved backend answers through its own
        tokenize route — never an estimate. A backend without the
        channel raises ``NotImplementedError`` (the 501 class); the
        endpoint refusing raises :class:`TokenCountUnavailableError`'s
        wire twin — a ``RuntimeError`` chain surfaces as
        ``OpenAICompatError``/``ValueError`` like every other SDK fault.
        ``headers`` accepts the ``X-Fx1-*`` link knobs exactly like
        :meth:`anthropic_message`.
        """
        body = (
            request
            if isinstance(request, AnthropicCountTokensRequest)
            else AnthropicCountTokensRequest.model_validate(request)
        )
        if body.tools is not None or body.tool_choice is not None:
            raise ValueError(
                "count_tokens covers the message channel — tools/tool_choice "
                "have no tokenize route to honor them"
            )
        hdrs = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
        backend_name, _fallbacks, checkpoint_dir, byok, _served = _resolve_openai_link(
            body.model,
            body.fx1,
            hdrs,
            ft_resolver=self._ft_store.checkpoint_for,
            require_known_model=True,
        )
        timeout_s = _resolve_timeout(body.fx1.timeout_s if body.fx1 is not None else None, hdrs)
        backend = self._resolve_completion_backend(
            backend_name,
            checkpoint_dir,
            None,
            byok.model_dump() if byok is not None else None,
            timeout_s,
        )
        if not isinstance(backend, TokenCountingBackend):
            raise NotImplementedError(f"backend {backend_name!r} has no tokenize channel")
        return backend.count_tokens(anthropic_count_messages(body))

    def anthropic_models(
        self,
        *,
        limit: int | None = None,
        after_id: str | None = None,
        before_id: str | None = None,
    ) -> dict[str, Any]:
        """``GET /v1/models`` under ``anthropic-version``, in-process —
        the ``{data, first_id, last_id, has_more}`` envelope over the
        same inventory :meth:`openai_models` serves, with Anthropic's
        ``limit``/cursor contract (unknown cursors page to empty)."""
        if limit is not None and not 1 <= limit <= 1000:
            raise ValueError(f"limit must be 1–1000, got {limit}")
        cards = list(self.openai_models().data)
        if after_id is not None:
            idx = next((i for i, m in enumerate(cards) if m.id == after_id), None)
            cards = cards[idx + 1 :] if idx is not None else []
        if before_id is not None:
            idx = next((i for i, m in enumerate(cards) if m.id == before_id), None)
            cards = cards[:idx] if idx is not None else []
        lim = limit or 20
        # back-pagination returns the window's tail — mirroring the
        # wire's before_id contract
        page = cards[-lim:] if before_id is not None else cards[:lim]
        return {
            "data": [anthropic_model_object(m.id, created=m.created) for m in page],
            "first_id": page[0].id if page else None,
            "last_id": page[-1].id if page else None,
            "has_more": len(cards) > len(page),
        }

    def anthropic_model(self, model_id: str) -> dict[str, Any]:
        """``GET /v1/models/{id}`` under ``anthropic-version``, in-process
        — the ``{type: \"model\"}`` card for a listed id; unknown ids
        raise :class:`OpenAICompatError` with the wire's 404 code."""
        card = self.openai_model(model_id)
        return anthropic_model_object(card.id, created=card.created)

    def openai_completion(
        self,
        request: OpenAICompletionRequest | dict[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        """One legacy ``/v1/completions`` call, weights-direct.

        ``request`` is the same body ``POST /v1/completions`` takes — a
        dict or a parsed :class:`OpenAICompletionRequest`. Each ``prompt``
        element translates into one gated user turn
        (``legacy_to_chat`` → :meth:`openai_chat`) and the envelopes fold
        into the flat ``text_completion`` choices array — the wire's own
        translation layer, so this leg cannot drift: same fail-closed
        validation (``suffix``/``best_of``/``logprobs`` refuse), same
        gate, same metering.

        Returns the ``text_completion`` envelope plus the last
        completion-log id for receipt lookup.
        """
        body = (
            request
            if isinstance(request, OpenAICompletionRequest)
            else OpenAICompletionRequest.model_validate(request)
        )
        prompts = [body.prompt] if isinstance(body.prompt, str) else list(body.prompt)
        envs: list[dict[str, Any]] = []
        cid: str | None = None
        for prompt in prompts:
            env, cid = self.openai_chat(legacy_to_chat(body, prompt), headers=headers)
            envs.append(env.model_dump(mode="json"))
        return (
            openai_completion_envelope(
                cid=cid or uuid.uuid4().hex,
                envs=envs,
                prompts=prompts,
                echo=body.echo,
            ),
            cid,
        )

    def openai_completion_stream(
        self,
        request: OpenAICompletionRequest | dict[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
        last_event_id: int | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """The legacy ``stream: true`` surface in-process — the
        ``text_completion`` chunk payloads, minus the wire framing.

        Runs the same gated completion as :meth:`openai_completion`,
        then emits the per-choice content deltas + ``finish_reason``
        terminal frames (plus the ``choices: []`` usage frame when
        ``stream_options.include_usage``) — identical to what
        ``POST /v1/completions`` serializes. ``last_event_id`` applies
        the same sequence filter the wire's resume uses.

        Returns ``(chunks, completion_id)``.
        """
        if last_event_id is not None and last_event_id < 0:
            raise ValueError(f"last_event_id must be >= 0, got {last_event_id}")
        body = (
            request
            if isinstance(request, OpenAICompletionRequest)
            else OpenAICompletionRequest.model_validate(request)
        )
        env, cid = self.openai_completion(body, headers=headers)
        chunks = list(
            completion_events(
                env,
                include_usage=bool((body.stream_options or {}).get("include_usage")),
            )
        )
        if last_event_id is not None:
            chunks = chunks[last_event_id + 1 :]
        return chunks, cid

    def openai_response(
        self,
        request: OpenAIResponseRequest | dict[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        """One Responses-API surface call, weights-direct.

        ``request`` is the same body ``POST /v1/responses`` takes — a dict
        or a parsed :class:`OpenAIResponseRequest`. Same fail-closed
        translation as the wire (``response_to_kwargs``), same honesty
        gate, same metering and completion log. ``previous_response_id``
        chains the turn onto a stored response — the parent must sit in
        the retrieval index (a ``store=false`` or evicted parent fails
        closed). Returns the ``response``
        object (``id`` mints the ``resp_`` handle; ``output[0]`` is the
        message item) plus the completion-log id for receipt lookup.
        """
        body = (
            request
            if isinstance(request, OpenAIResponseRequest)
            else OpenAIResponseRequest.model_validate(request)
        )
        # chaining resolves (and fails closed) at call time, on both the
        # sync and the background path — a bad parent never queues
        eff_body, conv_cid = self._chain_response_input(body)
        if body.background and not body.stream:
            if body.store is False:
                raise OpenAICompatError(
                    "background=true needs store=true — a background response is "
                    "only reachable through the retrieval index",
                    status=400,
                    code="background_requires_store",
                )
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
            self._openai_store.put(
                queued,
                items={"input_items": response_input_items_for_store(eff_body.input, rid=rid)},
            )
            cancel_ev = threading.Event()
            self._bg_cancel[rid] = cancel_ev

            def _bg() -> None:
                try:
                    if cancel_ev.is_set():
                        return
                    try:
                        self._openai_response_finish(
                            body,
                            eff_body,
                            headers,
                            rid=rid,
                            created=int(queued["created_at"]),
                            conv_cid=conv_cid,
                        )
                    except Exception as exc:  # noqa: BLE001 — worker faults land on the record
                        if cancel_ev.is_set():
                            return
                        cur = self._openai_store.get(rid)
                        if cur is not None:
                            cur["status"] = "failed"
                            cur["error"] = {
                                "message": (
                                    str(exc)
                                    if isinstance(exc, OpenAICompatError)
                                    else f"{type(exc).__name__}: {exc}"
                                ),
                                "code": (
                                    exc.code
                                    if isinstance(exc, OpenAICompatError)
                                    else "internal_error"
                                ),
                            }
                            self._openai_store.put(cur)
                    else:
                        if cancel_ev.is_set():
                            done = self._openai_store.get(rid)
                            if done is not None:
                                done["status"] = "cancelled"
                                self._openai_store.put(done)
                finally:
                    self._bg_cancel.pop(rid, None)

            threading.Thread(target=_bg, daemon=True).start()
            return queued, None
        return self._openai_response_finish(body, eff_body, headers, conv_cid=conv_cid)

    _RETRIEVAL_INJECT_BUDGET = 32768

    def _file_search_turn(
        self,
        body: OpenAIResponseRequest,
        eff_body: OpenAIResponseRequest,
    ) -> tuple[list[dict[str, Any]], OpenAIResponseRequest]:
        """The wire's ``_file_search_turn`` twin — runs the advertised
        ``file_search`` tool specs against ``self._vs_store`` before the
        backend call; hits inject as a developer context item on the
        effective input (stored in input_items = the honest transcript)
        and one ``file_search_call`` output item records the run."""
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
        budget = self._RETRIEVAL_INJECT_BUDGET
        for spec in fs_specs:
            # fs_specs is already filtered to type == "file_search".
            if spec.type != "file_search":  # pragma: no cover
                continue  # pragma: no cover
            ro = spec.ranking_options or {}
            threshold = ro.get("score_threshold")
            try:
                hits = self._vs_store.search(
                    list(spec.vector_store_ids),
                    query,
                    max_results=spec.max_num_results or 10,
                    filters=spec.filters,
                    score_threshold=(float(threshold) if threshold is not None else None),
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

    def _openai_response_finish(
        self,
        body: OpenAIResponseRequest,
        eff_body: OpenAIResponseRequest,
        headers: Mapping[str, str] | None,
        *,
        rid: str | None = None,
        created: int | None = None,
        conv_cid: str | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        """The synchronous tail of ``openai_response`` — shared by the
        direct call and the background worker (which pins ``rid``/
        ``created`` so the completed record lands on the queued
        envelope's id).``conv_cid`` carries the resolved conversation
        anchor — the turn's items append onto the conv when it
        completes."""
        search_items, eff_body = self._file_search_turn(body, eff_body)
        kwargs = response_to_kwargs(
            eff_body, dict(headers or {}), ft_resolver=self._ft_store.checkpoint_for
        )
        result = self.complete(**kwargs)
        # a tool-call turn carries no text — nothing to post-validate
        if result.content or not result.tool_calls:
            validate_response_format(response_text_format(body), result.content)
        call_items, inc_details = response_cap_call_items(body, result.tool_calls or [])
        lp_arr = result.logprobs.get("content") if isinstance(result.logprobs, dict) else None
        envelope = openai_response_object(
            rid=rid or f"resp_{uuid.uuid4().hex}",
            item_id=f"msg_{uuid.uuid4().hex}",
            content=result.content,
            body=body,
            model=result.model,
            usage=result.usage,
            status=("incomplete" if inc_details else "completed"),
            created=created,
            call_items=call_items,
            search_items=search_items or None,
            logprobs=(lp_arr if isinstance(lp_arr, list) else None),
            incomplete_details=inc_details,
        )
        if body.store is not False:
            self._openai_store.put(
                {
                    **envelope,
                    # the completion-log link the replay surface returns
                    # with its events — stripped before any wire read
                    "_fx1_completion_id": result.completion_id,
                },
                items={
                    "input_items": response_input_items_for_store(
                        eff_body.input, rid=str(envelope["id"])
                    )
                },
            )
        self._conv_append(conv_cid, body, envelope)
        return envelope, result.completion_id

    def _conv_append(
        self,
        conv_cid: str | None,
        body: OpenAIResponseRequest,
        envelope: dict[str, Any],
    ) -> None:
        """Append the turn's items onto the conversation — the request's
        input items + the response's output items, ids minted off the
        response rid. The conv container is its own store: the turn lands
        on it even when the response itself carried ``store=false``."""
        if conv_cid is None:
            return
        if self._conv_store.get(conv_cid) is None:
            return  # deleted mid-flight — the response record stands alone
        appended = [
            *response_input_items_for_store(body.input, rid=str(envelope["id"])),
            *[it for it in envelope["output"] if isinstance(it, dict)],
        ]
        # merged under the store lock — parallel turns can't lose each
        # other's append; a delete racing in resolves as a skip.
        self._conv_store.mutate_items(conv_cid, "items", lambda current: [*current, *appended])

    def _chain_response_input(
        self, body: OpenAIResponseRequest
    ) -> tuple[OpenAIResponseRequest, str | None]:
        """``previous_response_id``/``conversation`` → the effective input
        item list plus the resolved conv id (``None`` when unanchored or
        chain-anchored). The chain parent must be a stored ``response`` —
        a ``store=false`` or evicted parent fails closed — and the conv
        anchor must be a live ``conv_*`` container in the conv store.
        The two anchors are mutually exclusive at validation."""
        if body.previous_response_id is not None:
            prev = self._openai_store.get(body.previous_response_id)
            if prev is None or prev.get("object") != "response":
                raise OpenAICompatError(
                    f"previous_response_id {body.previous_response_id!r} not found — "
                    "the chain parent must be a stored response (store=true)",
                    status=400,
                    code="previous_response_not_found",
                )
            return (
                body.model_copy(
                    update={
                        "input": chained_response_input(
                            prev,
                            self._openai_store.get_items(body.previous_response_id, "input_items")
                            or [],
                            body.input,
                        )
                    }
                ),
                None,
            )
        conv_cid = conversation_id_of(body.conversation)
        if conv_cid is None:
            return body, None
        conv = self._conv_store.get(conv_cid)
        if conv is None or conv.get("object") != "conversation":
            raise OpenAICompatError(
                f"conversation {conv_cid!r} not found — create it with "
                "openai_conversation_create first",
                status=400,
                code="conversation_not_found",
            )
        return (
            body.model_copy(
                update={
                    "input": chained_response_input(
                        {"output": []},
                        self._conv_store.get_items(conv_cid, "items") or [],
                        body.input,
                    )
                }
            ),
            conv_cid,
        )

    def openai_response_stream(
        self,
        request: OpenAIResponseRequest | dict[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
        last_event_id: int | None = None,
    ) -> tuple[list[tuple[str, dict[str, Any]]], str | None]:
        """The Responses ``stream: true`` surface in-process —
        ``(event, payload)`` pairs, identical to what the wire serializes
        into ``event:``/``data:`` frames (minus framing). ``last_event_id``
        applies the same sequence filter as the wire resume."""
        if last_event_id is not None and last_event_id < 0:
            raise ValueError(f"last_event_id must be >= 0, got {last_event_id}")
        body = (
            request
            if isinstance(request, OpenAIResponseRequest)
            else OpenAIResponseRequest.model_validate(request)
        )
        eff_body, conv_cid = self._chain_response_input(body)
        search_items_s, eff_body = self._file_search_turn(body, eff_body)
        kwargs = response_to_kwargs(
            eff_body, dict(headers or {}), ft_resolver=self._ft_store.checkpoint_for
        )
        result = self.complete(**kwargs)
        if result.content or not result.tool_calls:
            validate_response_format(response_text_format(body), result.content)
        rid = f"resp_{uuid.uuid4().hex}"
        item_id = f"msg_{uuid.uuid4().hex}"
        call_items, inc_details_s = response_cap_call_items(body, result.tool_calls or [])
        lp_arr_s = result.logprobs.get("content") if isinstance(result.logprobs, dict) else None
        env_s = openai_response_object(
            rid=rid,
            item_id=item_id,
            content=result.content,
            body=body,
            model=result.model,
            usage=result.usage,
            status=("incomplete" if inc_details_s else "completed"),
            call_items=call_items,
            search_items=search_items_s or None,
            logprobs=(lp_arr_s if isinstance(lp_arr_s, list) else None),
            incomplete_details=inc_details_s,
        )
        if body.store is not False:
            self._openai_store.put(
                {
                    **env_s,
                    # same completion-log link the replay surface returns
                    "_fx1_completion_id": result.completion_id,
                },
                items={"input_items": response_input_items_for_store(eff_body.input, rid=rid)},
            )
        self._conv_append(conv_cid, body, env_s)
        events = list(
            openai_response_events(
                text=result.content,
                rid=rid,
                item_id=item_id,
                body=body,
                model=result.model,
                usage=result.usage,
                call_items=call_items,
                search_items=search_items_s or None,
                logprobs=(lp_arr_s if isinstance(lp_arr_s, list) else None),
                final_status=("incomplete" if inc_details_s else "completed"),
                incomplete_details=inc_details_s,
            )
        )
        if last_event_id is not None:
            events = events[last_event_id + 1 :]
        return events, result.completion_id

    def openai_embeddings(
        self,
        request: OpenAIEmbeddingRequest | dict[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        """One OpenAI ``embeddings.create``, weights-direct.

        ``request`` is the same body ``POST /v1/embeddings`` takes — a
        dict or a parsed :class:`OpenAIEmbeddingRequest`; ``headers``
        accepts the wire's ``X-Fx1-*`` knobs. Validation and link
        resolution are the wire's own (``embeddings_to_kwargs``), so this
        path cannot drift from ``/v1``. ``model`` reaches the provider
        verbatim — embedding models name themselves; a link without the
        ``embeddings`` channel answers ``NotImplementedError``, never
        fabricated vectors. Returns ``(list envelope, completion_id)`` —
        the id links the completion-log record.
        """
        body = (
            request
            if isinstance(request, OpenAIEmbeddingRequest)
            else OpenAIEmbeddingRequest.model_validate(request)
        )
        kwargs = embeddings_to_kwargs(
            body, dict(headers or {}), ft_resolver=self._ft_store.checkpoint_for
        )
        prompt_sha256 = hashlib.sha256(
            json.dumps(
                {"model": kwargs["model"], "input": kwargs["input"]},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        params: dict[str, Any] = {
            k: v
            for k, v in (
                ("encoding_format", kwargs["encoding_format"]),
                ("dimensions", kwargs["dimensions"]),
            )
            if v is not None
        }
        attempts: list[dict[str, Any]] = []
        chain = _fallback_chain(kwargs["backend"], kwargs["fallbacks"])
        _check_link_kwargs(chain, kwargs["checkpoint_dir"], kwargs["byok"])
        last_exc: Exception | None = None
        for cand in chain:
            t0 = time.monotonic()
            try:
                backend_obj = self._resolve_link(
                    cand,
                    kwargs["checkpoint_dir"],
                    None,
                    kwargs["byok"],
                    kwargs["timeout_s"],
                )
            except (BackendNotConfiguredError, RuntimeError, ValueError) as exc:
                # resolve-level availability fault — the wire's 503 class
                attempts.append(
                    {
                        "backend": cand,
                        "ok": False,
                        "error_class": type(exc).__name__,
                        "latency_ms": (time.monotonic() - t0) * 1000.0,
                    }
                )
                last_exc = exc
                continue
            except Exception as exc:
                attempts.append(
                    {
                        "backend": cand,
                        "ok": False,
                        "error_class": type(exc).__name__,
                        "latency_ms": (time.monotonic() - t0) * 1000.0,
                    }
                )
                self._record_call(
                    cand,
                    None,
                    False,
                    (time.monotonic() - t0) * 1000.0,
                    None,
                    str(exc),
                    type(exc).__name__,
                    prompt_sha256,
                    None,
                    tuple(attempts),
                    params or None,
                    kwargs["user"],
                    None,
                )
                raise
            t0 = time.monotonic()
            try:
                if not isinstance(backend_obj, EmbeddingBackend):
                    raise NotImplementedError(f"backend {cand!r} has no embeddings channel")
                result = backend_obj.embeddings(
                    kwargs["input"],
                    model=kwargs["model"],
                    encoding_format=kwargs["encoding_format"],
                    dimensions=kwargs["dimensions"],
                    user=kwargs["user"],
                )
            except Exception as exc:
                attempts.append(
                    {
                        "backend": cand,
                        "ok": False,
                        "error_class": type(exc).__name__,
                        "latency_ms": (time.monotonic() - t0) * 1000.0,
                    }
                )
                closer = getattr(backend_obj, "close", None)
                if callable(closer):
                    closer()
                if isinstance(exc, (BackendNotConfiguredError, RuntimeError)) and not isinstance(
                    exc, NotImplementedError
                ):
                    # call-time availability fault — advance the chain
                    last_exc = exc
                    continue
                self._record_call(
                    cand,
                    None,
                    False,
                    (time.monotonic() - t0) * 1000.0,
                    getattr(backend_obj, "last_usage", None)
                    if isinstance(getattr(backend_obj, "last_usage", None), dict)
                    else None,
                    str(exc),
                    type(exc).__name__,
                    prompt_sha256,
                    None,
                    tuple(attempts),
                    params or None,
                    kwargs["user"],
                    None,
                )
                raise
            attempts.append(
                {
                    "backend": cand,
                    "ok": True,
                    "latency_ms": (time.monotonic() - t0) * 1000.0,
                }
            )
            cid = self._record_call(
                cand,
                result.model or kwargs["model"],
                True,
                (time.monotonic() - t0) * 1000.0,
                result.usage,
                None,
                None,
                prompt_sha256,
                hashlib.sha256(
                    json.dumps(list(result.data), sort_keys=True).encode("utf-8")
                ).hexdigest(),
                tuple(attempts) if len(attempts) > 1 else None,
                params or None,
                kwargs["user"],
                None,
            )
            closer = getattr(backend_obj, "close", None)
            if callable(closer):
                closer()
            return (
                openai_embedding_envelope(
                    data=result.data,
                    model=result.model or kwargs["model"],
                    usage=result.usage,
                ),
                cid,
            )
        assert last_exc is not None  # noqa: S101 — every link failed retriably
        raise last_exc

    def openai_chat_get(self, completion_id: str) -> dict[str, Any]:
        """``GET /v1/chat/completions/{id}`` in-process — the stored
        ``chat.completion`` envelope, or ``KeyError`` (404 on the wire:
        evicted, deleted, or sent with ``store=false``)."""
        env = self._openai_store.get(completion_id)
        if env is None or env.get("object") != "chat.completion":
            raise KeyError(f"completion {completion_id!r} not in the retrieval index")
        return {k: v for k, v in env.items() if not k.startswith("_fx1_")}

    def openai_chat_list(
        self,
        *,
        model: str | None = None,
        metadata: Mapping[str, str] | None = None,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
        order: str = "asc",
    ) -> dict[str, Any]:
        """``GET /v1/chat/completions`` in-process — stored completions,
        optionally filtered by ``model`` and an exact ``metadata`` subset."""
        envs = self._openai_store.list_envelopes("chat.completion")
        if model is not None:
            envs = [e for e in envs if e.get("model") == model]
        if metadata:
            envs = [
                e
                for e in envs
                if isinstance(e.get("metadata"), dict)
                and all(e["metadata"].get(k) == v for k, v in metadata.items())
            ]
        return paged_item_list(envs, limit=limit, after=after, before=before, order=order)

    def openai_chat_delete(self, completion_id: str) -> dict[str, Any]:
        """``DELETE /v1/chat/completions/{id}`` in-process."""
        if not self._openai_store.delete(completion_id):
            raise KeyError(f"completion {completion_id!r} not in the retrieval index")
        return {"id": completion_id, "object": "chat.completion.deleted", "deleted": True}

    def openai_chat_update(
        self, completion_id: str, *, metadata: Mapping[str, str] | None = None
    ) -> dict[str, Any]:
        """``POST /v1/chat/completions/{id}`` in-process — ``metadata``
        replaces the stored completion's wholesale (the only mutable
        field); ``KeyError`` when the id isn't a live stored completion."""
        env = self._openai_store.get(completion_id)
        if env is None or env.get("object") != "chat.completion":
            raise KeyError(f"completion {completion_id!r} not in the retrieval index")
        md = dict(metadata) if metadata is not None else {}
        if len(md) > 16 or any(len(k) > 64 or len(v) > 512 for k, v in md.items()):
            raise ValueError("metadata accepts ≤16 pairs, keys ≤64 chars, values ≤512")
        updated = self._openai_store.update_metadata(completion_id, md)
        if updated is None:
            raise KeyError(f"completion {completion_id!r} not in the retrieval index")
        return {k: v for k, v in updated.items() if not k.startswith("_fx1_")}

    def openai_response_get(self, response_id: str) -> dict[str, Any]:
        """``GET /v1/responses/{id}`` in-process — the stored ``response``
        envelope, or ``KeyError``."""
        env = self._openai_store.get(response_id)
        if env is None or env.get("object") != "response":
            raise KeyError(f"response {response_id!r} not in the retrieval index")
        return {k: v for k, v in env.items() if not k.startswith("_fx1_")}

    def openai_response_delete(self, response_id: str) -> dict[str, Any]:
        """``DELETE /v1/responses/{id}`` in-process."""
        if not self._openai_store.delete(response_id):
            raise KeyError(f"response {response_id!r} not in the retrieval index")
        return {"id": response_id, "object": "response.deleted", "deleted": True}

    def openai_response_cancel(self, response_id: str) -> dict[str, Any]:
        """``POST /v1/responses/{id}/cancel`` in-process — flips a queued
        or in-progress background response to ``cancelled`` and tells the
        worker to discard its result. Terminal responses raise
        ``OpenAICompatError(cancel_terminal)``; unknown ids ``KeyError``."""
        env = self._openai_store.get(response_id)
        if env is None or env.get("object") != "response":
            raise KeyError(f"response {response_id!r} not in the retrieval index")
        if env["status"] in OPENAI_RESPONSE_TERMINAL:
            raise OpenAICompatError(
                f"{response_id!r} is already {env['status']} — only queued or "
                "in_progress responses cancel",
                status=409,
                code="cancel_terminal",
            )
        ev = self._bg_cancel.get(response_id)
        if ev is not None:
            ev.set()
        env["status"] = "cancelled"
        self._openai_store.put(env)
        return {k: v for k, v in env.items() if not k.startswith("_fx1_")}

    def openai_response_replay(
        self,
        response_id: str,
        *,
        starting_after: int | None = None,
        timeout_s: float = 600.0,
        poll_s: float = 0.25,
    ) -> tuple[list[tuple[str, dict[str, Any]]], str | None]:
        """``GET /v1/responses/{id}?stream=true`` in-process — the stored
        envelope's replay grammar as ``(event, payload)`` pairs, identical
        to what the wire serializes into ``event:``/``id:``/``data:``
        frames. A terminal envelope returns the full recorded sequence;
        a still-``queued``/``in_progress`` background response emits its
        prelude then live-follows (polling ``poll_s``) until the terminal
        frame or the ``timeout_s`` deadline — the same contract the wire
        generator runs. ``starting_after`` resumes past sequence N (the
        frame's ``id:`` cursor). Returns ``(events, completion_id)`` — the
        completion-log link the wire answers via ``X-Fx1-Completion-Id``;
        a record deleted mid-follow returns the events seen so far
        (no terminal frame), matching the wire's silent close."""
        if starting_after is not None and starting_after < 0:
            raise ValueError(f"starting_after must be >= 0, got {starting_after}")
        env = self._openai_store.get(response_id)
        if env is None or env.get("object") != "response":
            raise KeyError(f"response {response_id!r} not in the retrieval index")
        skip = (starting_after + 1) if starting_after is not None else 0
        deadline = time.monotonic() + timeout_s
        cursor = 0
        events: list[tuple[str, dict[str, Any]]] = []
        while env is not None and env.get("object") == "response":
            evs = list(openai_response_replay_events(env))
            events.extend(evs[cursor:])
            cursor = len(evs)
            if env.get("status") in OPENAI_RESPONSE_TERMINAL:
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(poll_s)
            env = self._openai_store.get(response_id)
        cid = env.get("_fx1_completion_id") if isinstance(env, dict) else None
        return events[skip:], (cid if isinstance(cid, str) else None)

    def openai_chat_messages(
        self,
        completion_id: str,
        *,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
        order: str = "asc",
    ) -> dict[str, Any]:
        """``GET /v1/chat/completions/{id}/messages`` in-process — the
        stored request messages, paged."""
        env = self._openai_store.get(completion_id)
        if env is None or env.get("object") != "chat.completion":
            raise KeyError(f"completion {completion_id!r} not in the retrieval index")
        return paged_item_list(
            self._openai_store.get_items(completion_id, "messages") or [],
            limit=limit,
            after=after,
            before=before,
            order=order,
        )

    def openai_response_input_items(
        self,
        response_id: str,
        *,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
        order: str = "asc",
    ) -> dict[str, Any]:
        """``GET /v1/responses/{id}/input_items`` in-process — the stored
        ``input`` items, paged."""
        env = self._openai_store.get(response_id)
        if env is None or env.get("object") != "response":
            raise KeyError(f"response {response_id!r} not in the retrieval index")
        return paged_item_list(
            self._openai_store.get_items(response_id, "input_items") or [],
            limit=limit,
            after=after,
            before=before,
            order=order,
        )

    # --- conversations (the /v1/conversations twin) --------------------------

    def _conversation_get(self, conversation_id: str) -> dict[str, Any]:
        conv = self._conv_store.get(conversation_id)
        if conv is None or conv.get("object") != "conversation":
            raise KeyError(f"conversation {conversation_id!r} not found")
        return conv

    def openai_conversation_create(
        self,
        *,
        items: list[dict[str, Any]] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """``POST /v1/conversations`` in-process — mints a ``conv_*``
        container; ``items`` seeds it, ``metadata`` stamps it."""
        cid = f"conv_{uuid.uuid4().hex}"
        env = openai_conversation_object(cid=cid, metadata=metadata)
        self._conv_store.put(
            env,
            items={"items": (response_input_items_for_store(items, rid=cid) if items else [])},
        )
        return env

    def openai_conversation_get(self, conversation_id: str) -> dict[str, Any]:
        """``GET /v1/conversations/{id}`` in-process."""
        return self._conversation_get(conversation_id)

    def openai_conversation_update(
        self,
        conversation_id: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """``POST /v1/conversations/{id}`` in-process — metadata replaces
        wholesale."""
        conv = self._conversation_get(conversation_id)
        new_conv = dict(conv)
        new_conv["metadata"] = dict(metadata) if metadata is not None else {}
        if not self._conv_store.put_if_present(new_conv):
            raise KeyError(f"conversation {conversation_id!r} not found")
        return new_conv

    def openai_conversation_delete(self, conversation_id: str) -> dict[str, Any]:
        """``DELETE /v1/conversations/{id}`` in-process — the conv and its
        items drop; member responses stay in the index."""
        self._conversation_get(conversation_id)
        self._conv_store.delete(conversation_id)
        return {"id": conversation_id, "object": "conversation.deleted", "deleted": True}

    def openai_conversation_items(
        self,
        conversation_id: str,
        *,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
        order: str = "asc",
    ) -> dict[str, Any]:
        """``GET /v1/conversations/{id}/items`` in-process — the conv's
        accumulated items, paged."""
        self._conversation_get(conversation_id)
        return paged_item_list(
            self._conv_store.get_items(conversation_id, "items") or [],
            limit=limit,
            after=after,
            before=before,
            order=order,
        )

    def openai_conversation_items_add(
        self,
        conversation_id: str,
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """``POST /v1/conversations/{id}/items`` in-process — appends the
        dicts (minted per append), returns the minted list."""
        self._conversation_get(conversation_id)
        minted: list[dict[str, Any]] = []

        def _extend(current: list[dict[str, Any]]) -> list[dict[str, Any]]:
            # A fresh namespace avoids id reuse after deleting earlier items;
            # the merge remains inside the store lock.
            minted.extend(
                response_input_items_for_store(items, rid=f"{conversation_id}:{uuid.uuid4().hex}")
            )
            return [*current, *minted]

        if self._conv_store.mutate_items(conversation_id, "items", _extend) is None:
            raise KeyError(f"conversation {conversation_id!r} not found")
        return {
            "object": "list",
            "data": minted,
            "first_id": minted[0].get("id") if minted else None,
            "last_id": minted[-1].get("id") if minted else None,
            "has_more": False,
        }

    def openai_conversation_item(
        self,
        conversation_id: str,
        item_id: str,
    ) -> dict[str, Any]:
        """``GET /v1/conversations/{id}/items/{item_id}`` in-process —
        one item by id; a miss raises ``KeyError``."""
        self._conversation_get(conversation_id)
        items = self._conv_store.get_items(conversation_id, "items") or []
        for it in items:
            if it.get("id") == item_id:
                return it
        raise KeyError(f"item {item_id!r} not found in {conversation_id!r}")

    def openai_conversation_item_delete(
        self,
        conversation_id: str,
        item_id: str,
    ) -> dict[str, Any]:
        """``DELETE /v1/conversations/{id}/items/{item_id}`` in-process —
        drops one item; a missing id raises ``KeyError``."""
        conv = self._conversation_get(conversation_id)

        def _drop(current: list[dict[str, Any]]) -> list[dict[str, Any]]:
            kept = [it for it in current if it.get("id") != item_id]
            if len(kept) == len(current):
                raise KeyError(f"item {item_id!r} not found in {conversation_id!r}")
            return kept

        if self._conv_store.mutate_items(conversation_id, "items", _drop) is None:
            raise KeyError(f"conversation {conversation_id!r} not found")
        return conv

    # --- vector stores (the /v1/vector_stores twin) --------------------------
    # A journaled lexical retrieval corpus over the SDK's upload records —
    # the file_search tool on openai_response searches it in-process.
    # VectorStoreError surfaces as ValueError-class request faults with
    # the same status/code the wire maps.

    def vector_store_create(
        self,
        *,
        name: str | None = None,
        metadata: dict[str, str] | None = None,
        file_ids: list[str] | None = None,
        expires_after: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """``POST /v1/vector_stores`` in-process — mints a ``vs_*`` store;
        ``file_ids`` attach at create (an unresolvable id fails closed);
        ``expires_after`` sets the OpenAI anchor policy
        ``{"anchor": "last_active_at", "days": 1..365}``."""
        return self._vs_store.create(
            name=name,
            metadata=metadata,
            file_ids=tuple(file_ids or ()),
            expires_after=expires_after,
        )

    def vector_store_get(self, vector_store_id: str) -> dict[str, Any]:
        """``GET /v1/vector_stores/{id}`` in-process."""
        return self._vs_store.get(vector_store_id)

    def vector_store_update(
        self,
        vector_store_id: str,
        *,
        name: str | None = None,
        metadata: dict[str, str] | None = None,
        expires_after: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """``POST /v1/vector_stores/{id}`` in-process — name/metadata
        replace wholesale when given; ``expires_after`` re-anchors the
        expiry window (revives an expired store)."""
        return self._vs_store.update(
            vector_store_id,
            name=name,
            metadata=metadata,
            expires_after=expires_after,
        )

    def vector_store_delete(self, vector_store_id: str) -> dict[str, Any]:
        """``DELETE /v1/vector_stores/{id}`` in-process — the store and
        its index drop; the file records survive."""
        return self._vs_store.delete(vector_store_id)

    def vector_store_list(
        self,
        *,
        limit: int = 20,
        order: str = "desc",
        after: str | None = None,
        before: str | None = None,
    ) -> dict[str, Any]:
        """``GET /v1/vector_stores`` in-process — cursor-paged like the
        wire."""
        return self._vs_store.list_stores(limit=limit, order=order, after=after, before=before)

    def vector_store_file_create(
        self,
        vector_store_id: str,
        file_id: str,
        *,
        attributes: dict[str, Any] | None = None,
        chunking_strategy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """``POST /v1/vector_stores/{id}/files`` in-process — indexes the
        upload record's bytes; empty text lands ``status: failed``."""
        return self._vs_store.attach(
            vector_store_id,
            file_id,
            attributes=attributes,
            chunking_strategy=chunking_strategy,
        )

    def vector_store_file_list(
        self,
        vector_store_id: str,
        *,
        limit: int = 20,
        order: str = "asc",
        after: str | None = None,
        before: str | None = None,
        filter: str | None = None,
    ) -> dict[str, Any]:
        """``GET /v1/vector_stores/{id}/files`` in-process — ``filter``
        is an OpenAI status word (completed|failed|…)."""
        return self._vs_store.list_files(
            vector_store_id,
            limit=limit,
            order=order,
            after=after,
            before=before,
            filter=filter,
        )

    def vector_store_file_get(self, vector_store_id: str, file_id: str) -> dict[str, Any]:
        """``GET /v1/vector_stores/{id}/files/{file_id}`` in-process."""
        return self._vs_store.get_file(vector_store_id, file_id)

    def vector_store_file_delete(self, vector_store_id: str, file_id: str) -> dict[str, Any]:
        """``DELETE /v1/vector_stores/{id}/files/{file_id}`` in-process —
        the attachment's chunks leave the index; the file record
        survives."""
        return self._vs_store.detach(vector_store_id, file_id)

    def vector_store_file_content(self, vector_store_id: str, file_id: str) -> dict[str, Any]:
        """``GET /v1/vector_stores/{id}/files/{file_id}/content``
        in-process — the stored decoded text as text parts."""
        return self._vs_store.file_content(vector_store_id, file_id)

    def vector_store_search(
        self,
        vector_store_id: str,
        query: str | list[str],
        *,
        max_num_results: int | None = None,
        filters: dict[str, Any] | None = None,
        ranking_options: dict[str, Any] | None = None,
        rewrite_query: bool | None = None,
    ) -> dict[str, Any]:
        """``POST /v1/vector_stores/{id}/search`` in-process — the
        ranked hits without a model call, as a
        ``vector_store.search_results.page`` dict. Same fail-closed
        contract as the wire: ``rewrite_query`` refused, ``ranker``
        accepts only ``"auto"``, ``score_threshold`` bounds the cosine
        floor."""
        body = OpenAIVectorStoreSearch.model_validate(
            {
                "query": query,
                "max_num_results": max_num_results,
                "filters": filters,
                "ranking_options": ranking_options,
                "rewrite_query": rewrite_query,
            }
        )
        q = body.query if isinstance(body.query, str) else " ".join(str(x) for x in body.query)
        ro = body.ranking_options or {}
        try:
            hits = self._vs_store.search(
                [vector_store_id],
                q,
                max_results=body.max_num_results or 10,
                filters=body.filters,
                score_threshold=ro.get("score_threshold"),
            )
        except VectorStoreError as exc:
            raise OpenAICompatError(str(exc), status=exc.status, code=exc.code) from exc
        return {
            "object": "vector_store.search_results.page",
            "search_query": q,
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

    def vector_store_file_batch_create(
        self,
        vector_store_id: str,
        file_ids: list[str],
        *,
        attributes: dict[str, Any] | None = None,
        chunking_strategy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """``POST /v1/vector_stores/{id}/file_batches`` in-process —
        attach many ``file-*`` records in one call. Members attach
        synchronously through the same code path the wire calls;
        per-file refusals count ``failed`` with ``last_error``, never
        abort. Status is terminal at return."""
        body = OpenAIVectorStoreFileBatchCreate.model_validate(
            {
                "file_ids": file_ids,
                "attributes": attributes,
                "chunking_strategy": chunking_strategy,
            }
        )
        try:
            return self._vs_store.file_batch_create(
                vector_store_id,
                body.file_ids,
                attributes=body.attributes,
                chunking_strategy=body.chunking_strategy,
            )
        except VectorStoreError as exc:
            raise OpenAICompatError(str(exc), status=exc.status, code=exc.code) from exc

    def vector_store_file_batch_get(self, vector_store_id: str, batch_id: str) -> dict[str, Any]:
        """``GET /v1/vector_stores/{id}/file_batches/{batch_id}`` —
        standing status + file_counts."""
        try:
            return self._vs_store.file_batch_get(vector_store_id, batch_id)
        except VectorStoreError as exc:
            raise OpenAICompatError(str(exc), status=exc.status, code=exc.code) from exc

    def vector_store_file_batch_cancel(self, vector_store_id: str, batch_id: str) -> dict[str, Any]:
        """``POST .../file_batches/{id}/cancel`` — batches are terminal
        at create; raises ``OpenAICompatError`` 409 ``file_batch_terminal``."""
        try:
            return self._vs_store.file_batch_cancel(vector_store_id, batch_id)
        except VectorStoreError as exc:
            raise OpenAICompatError(str(exc), status=exc.status, code=exc.code) from exc

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
        """``GET .../file_batches/{id}/files`` — the frozen per-file
        verdicts in request order."""
        try:
            return self._vs_store.file_batch_files(
                vector_store_id,
                batch_id,
                limit=limit,
                order=order,
                after=after,
                before=before,
                filter=filter,
            )
        except VectorStoreError as exc:
            raise OpenAICompatError(str(exc), status=exc.status, code=exc.code) from exc

    def openai_batch(
        self,
        lines: list[dict[str, Any]],
        *,
        endpoint: str = "/v1/chat/completions",
        headers: Mapping[str, str] | None = None,
        callback_url: str | None = None,
        callback_secret: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """The ``/v1/batches`` surface, weights-direct — synchronous in
        process (no upload/poll machinery: lines in, the batch object +
        output lines out). Each line goes through the endpoint's own
        request model and the same ``openai_chat``/``openai_response``
        calls the wire batch worker makes — one pipeline, not a second
        codepath. Per-line faults land as ``status_code``-carrying output
        lines, never as a raised batch — same rule as the wire.

        Returns ``(batch, output_lines)``: the batch envelope (status
        ``completed`` — in-process has no queue to observe) and the
        OpenAI batch-result lines the wire would write into the output
        file.

        ``callback_url``/``callback_secret`` behave as on the wire: the
        finished batch envelope POSTs to the URL once, HMAC-signed when
        the secret is set, and ``callback_status``/``callback_attempts``/
        ``callback_error`` land on the returned envelope.
        """
        if callback_secret is not None and not callback_url:
            raise ValueError("callback_secret requires callback_url")
        if callback_url is not None:
            from fx1.serve.webhooks import check_callback_url  # noqa: PLC0415

            check_callback_url(callback_url)
        if endpoint not in OPENAI_BATCH_ENDPOINTS:
            raise OpenAICompatError(
                f"endpoint must be one of {sorted(OPENAI_BATCH_ENDPOINTS)}, got {endpoint!r}"
            )
        parsed = [
            batch_line_shape(dict(line), endpoint=endpoint, lineno=i)
            for i, line in enumerate(lines, 1)
        ]
        out_lines: list[dict[str, Any]] = []
        completed = failed = 0
        hdrs = dict(headers or {})
        for line in parsed:
            rid = uuid.uuid4().hex
            custom_id = str(line["custom_id"])
            status = 200
            try:
                obj = batch_line_body(line, endpoint)
                if getattr(obj, "stream", False):
                    raise OpenAICompatError(
                        "stream requests are not valid inside a batch",
                        code="invalid_request",
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
                    env, _cid = self.openai_chat(obj, headers=hdrs)
                    body_out: dict[str, Any] = env.model_dump(mode="json")
                elif isinstance(obj, OpenAIEmbeddingRequest):
                    env_e, _cid = self.openai_embeddings(obj, headers=hdrs)
                    body_out = env_e
                else:
                    env_r, _cid = self.openai_response(obj, headers=hdrs)
                    body_out = env_r
            except OpenAICompatError as exc:
                status = exc.status
                body_out = openai_error_body(str(exc), status, exc.code)
            except BackendNotConfiguredError as exc:  # 503 on the wire
                status = 503
                body_out = openai_error_body(str(exc), status, "backend_unavailable")
            except NotImplementedError as exc:  # no channel — 501 on the wire
                status = 501
                body_out = openai_error_body(str(exc), status, "not_supported")
            except Fx1HonestyError as exc:  # gate refusal — 502 on the wire
                status = 502
                body_out = openai_error_body(str(exc), status, "gate_refused")
            except ValueError as exc:  # remaining contract violations — 422
                status = 422
                body_out = openai_error_body(str(exc), status, "invalid_request")
            except Exception as exc:  # noqa: BLE001 — a line fault is data, not a crash
                status = 500
                body_out = openai_error_body(f"{type(exc).__name__}: {exc}", status, "server_error")
            if status == 200:
                completed += 1
            else:
                failed += 1
            out_lines.append(
                batch_output_line(custom_id=custom_id, status_code=status, body=body_out, rid=rid)
            )
        now = int(time.time())
        batch = {
            "id": f"batch_{uuid.uuid4().hex}",
            "object": "batch",
            "endpoint": endpoint,
            "errors": None,
            "input_file_id": None,
            "completion_window": "24h",
            "status": "completed",
            "output_file_id": None,
            "error_file_id": None,
            "created_at": now,
            "in_progress_at": now,
            "expires_at": now + 86400,
            "finalizing_at": now,
            "completed_at": now,
            "failed_at": None,
            "expired_at": None,
            "cancelling_at": None,
            "cancelled_at": None,
            "request_counts": {
                "total": len(parsed),
                "completed": completed,
                "failed": failed,
            },
            "metadata": None,
            "callback_url": callback_url,
            "callback_status": None,
            "callback_attempts": 0,
            "callback_error": None,
        }
        if callback_url:
            from fx1.serve.webhooks import deliver_signed  # noqa: PLC0415

            ok, err, attempts = deliver_signed(
                callback_url, callback_secret, json.dumps(batch).encode()
            )
            batch["callback_status"] = "delivered" if ok else "failed"
            batch["callback_error"] = None if ok else err
            batch["callback_attempts"] = attempts
        return batch, out_lines

    def anthropic_batch(
        self,
        requests: list[dict[str, Any]],
        *,
        headers: Mapping[str, str] | None = None,
        callback_url: str | None = None,
        callback_secret: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """The ``/v1/messages/batches`` surface, weights-direct —
        synchronous in process (no queue: items in, the
        ``message_batch`` object + results lines out). ``requests`` is
        the Anthropic-shaped ``[{custom_id, params}]`` list, validated
        by :class:`AnthropicBatchCreate` exactly like the wire — same
        unique-``custom_id`` rule, same ``stream`` refusal, same
        callback checks. Each item then runs :meth:`anthropic_message`
        — the wire's own translation + gated path, not a second
        codepath. Per-item faults land as ``type: "errored"`` result
        rows, never as a raised batch.

        Returns ``(batch, result_lines)``: the ``message_batch`` object
        (``processing_status: "ended"`` — in-process has no queue to
        observe; ``results_url`` still surfaces because the batch is
        ended) and the ``{custom_id, result}`` rows the wire's
        ``results`` route would serve, in submission order.
        """
        body = AnthropicBatchCreate.model_validate(
            {
                "requests": requests,
                "callback_url": callback_url,
                "callback_secret": callback_secret,
            }
        )
        out_rows: list[dict[str, Any]] = []
        counts = AnthropicBatchCounts(processing=len(body.requests))
        hdrs = dict(headers or {})
        for item in body.requests:
            try:
                msg, _cid = self.anthropic_message(item.params, headers=hdrs)
                row = anthropic_batch_result(
                    item.custom_id,
                    {"type": "succeeded", "message": msg.model_dump(mode="json")},
                )
                counts.succeeded += 1
            except OpenAICompatError as exc:
                row = anthropic_batch_result(
                    item.custom_id,
                    {
                        "type": "errored",
                        "error": anthropic_error_body(str(exc), exc.status)["error"],
                    },
                )
                counts.errored += 1
            except BackendNotConfiguredError as exc:  # 503 on the wire
                row = anthropic_batch_result(
                    item.custom_id,
                    {
                        "type": "errored",
                        "error": anthropic_error_body(str(exc), 503)["error"],
                    },
                )
                counts.errored += 1
            except Fx1HonestyError as exc:  # gate refusal — 502 on the wire
                row = anthropic_batch_result(
                    item.custom_id,
                    {
                        "type": "errored",
                        "error": anthropic_error_body(str(exc), 502)["error"],
                    },
                )
                counts.errored += 1
            except (ValueError, NotImplementedError) as exc:
                row = anthropic_batch_result(
                    item.custom_id,
                    {
                        "type": "errored",
                        "error": anthropic_error_body(str(exc), 400)["error"],
                    },
                )
                counts.errored += 1
            except Exception as exc:  # noqa: BLE001 — an item fault is a row, not a crash
                row = anthropic_batch_result(
                    item.custom_id,
                    {
                        "type": "errored",
                        "error": anthropic_error_body(f"{type(exc).__name__}: {exc}", 500)["error"],
                    },
                )
                counts.errored += 1
            out_rows.append(row)
        counts.processing = 0
        now = int(time.time())
        batch = anthropic_batch_object(
            {
                "batch_id": f"msgbatch_{uuid.uuid4().hex}",
                "status": "ended",
                "request_counts": counts.model_dump(),
                "created_at": now,
                "expires_at": now + 86400,
                "ended_at": now,
                "cancel_initiated_at": None,
                "callback_url": callback_url,
                "callback_status": None,
                "callback_attempts": 0,
                "callback_error": None,
            }
        )
        if callback_url:
            from fx1.serve.webhooks import deliver_signed  # noqa: PLC0415

            ok, err, attempts = deliver_signed(
                callback_url, callback_secret, json.dumps(batch).encode()
            )
            batch["callback_status"] = "delivered" if ok else "failed"
            batch["callback_error"] = None if ok else err
            batch["callback_attempts"] = attempts
        return batch, out_rows

    # ---- uploads (chunked files) ----------------------------------------

    def upload_create(
        self,
        *,
        purpose: str = "batch",
        filename: str = "input.jsonl",
        bytes: int,
        mime_type: str = "application/jsonl",
    ) -> dict[str, Any]:
        """``POST /v1/uploads``, in-process — open a chunked-upload
        intent. ``bytes`` is the DECLARED total the parts must sum to
        (fail-closed both ways, same as the wire)."""
        validate_upload_intent(purpose, filename, OPENAI_FILE_PURPOSE_ACCEPT)
        return upload_object(
            self._upload_store.create(
                purpose=purpose, filename=filename, nbytes=bytes, mime_type=mime_type
            )
        )

    def upload_part(self, upload_id: str, data: bytes) -> dict[str, Any]:
        """``POST /v1/uploads/{id}/parts`` — one chunk, journaled under
        ``state_dir`` exactly like the wire."""
        return self._upload_store.add_part(upload_id, data)

    def upload_complete(
        self,
        upload_id: str,
        part_ids: list[str],
        *,
        md5: str | None = None,
    ) -> dict[str, Any]:
        """``POST /v1/uploads/{id}/complete`` — assemble the parts into a
        process-local ``file-`` record (fetch bytes back with
        :meth:`file_content`). The md5 check runs before the mint, same
        as the wire. Validation and publication share one upload
        transition so concurrent completes cannot mint extra files."""

        def _publish(meta: UploadMeta, content: bytes) -> tuple[str, dict[str, Any]]:
            file_id = f"file-{uuid.uuid4().hex}"
            fobj = {
                "id": file_id,
                "object": "file",
                "purpose": meta.purpose,
                "filename": meta.filename,
                "bytes": len(content),
                "created_at": int(time.time()),
                "status": "processed",
            }
            with self._files_lock:
                self._files[file_id] = {**fobj, "_content": content}
                while len(self._files) > 256:
                    self._files.popitem(last=False)
            return file_id, fobj

        def _rollback(file_id: str) -> None:
            with self._files_lock:
                self._files.pop(file_id, None)

        done, fobj = self._upload_store.complete_with(
            upload_id, part_ids, publish=_publish, rollback=_rollback, md5=md5
        )
        return upload_object(done, file_obj=fobj)

    def upload_cancel(self, upload_id: str) -> dict[str, Any]:
        """``POST /v1/uploads/{id}/cancel``."""
        return upload_object(self._upload_store.cancel(upload_id))

    def openai_file_create(
        self,
        content: bytes,
        *,
        purpose: str = "batch",
        filename: str = "input.jsonl",
    ) -> dict[str, Any]:
        """``POST /v1/files`` in-process twin — a one-shot upload
        landing the same ``file-*`` record chunked uploads mint.
        Same fail-closed rules as the wire: purposes in
        ``OPENAI_FILE_PURPOSE_ACCEPT``, non-empty ``.jsonl`` bytes under
        the 8 MiB cap."""
        validate_upload_intent(purpose, filename, OPENAI_FILE_PURPOSE_ACCEPT)
        if not content:
            raise UploadStoreError(400, "file is empty", "invalid_request")
        if len(content) > 8 << 20:
            raise UploadStoreError(413, "file exceeds the 8 MiB cap", "file_too_large")
        file_id = f"file-{uuid.uuid4().hex}"
        fobj = {
            "id": file_id,
            "object": "file",
            "purpose": purpose,
            "filename": filename,
            "bytes": len(content),
            "created_at": int(time.time()),
            "status": "processed",
        }
        with self._files_lock:
            self._files[file_id] = {**fobj, "_content": bytes(content)}
            while len(self._files) > 256:
                self._files.popitem(last=False)
        return fobj

    def file_content(self, file_id: str) -> bytes:
        """``GET /v1/files/{id}/content`` twin — raw bytes of an
        upload-minted file."""
        rec = self._files.get(file_id)
        if rec is None:
            raise KeyError(file_id)
        return bytes(rec["_content"])

    def file_card(self, file_id: str) -> dict[str, Any]:
        """``GET /v1/files/{id}`` twin — the file object (no content)."""
        rec = self._files.get(file_id)
        if rec is None:
            raise KeyError(file_id)
        return {k: v for k, v in rec.items() if k != "_content"}

    def _resolve_chain(
        self,
        backend: str,
        fallbacks: list[str] | None,
        checkpoint_dir: str | Path | None,
        backend_kwargs: dict[str, Any] | None,
        byok: dict[str, str] | None,
        timeout_s: float | None,
    ) -> tuple[str, Any]:
        """First resolvable link serves — resolve-level fallback for the
        shared-backend calls (batch / stream), same contract as the wire."""
        chain = _fallback_chain(backend, fallbacks)
        _check_link_kwargs(chain, checkpoint_dir, byok)
        last: Exception | None = None
        for cand in chain:
            try:
                return cand, self._resolve_link(
                    cand, checkpoint_dir, backend_kwargs, byok, timeout_s
                )
            except (BackendNotConfiguredError, RuntimeError) as exc:
                last = exc
        assert last is not None  # every link failed retriably
        raise last

    def _resolve_link(
        self,
        link: str,
        checkpoint_dir: str | Path | None,
        backend_kwargs: dict[str, Any] | None,
        byok: dict[str, str] | None,
        timeout_s: float | None,
    ) -> Any:
        """Resolve one fallback-chain link — per-link kwargs: ``byok``
        binds only a ``'byok'`` link, ``checkpoint_dir`` only ``'local_fx1'``."""
        return self._resolve_completion_backend(
            link,
            checkpoint_dir if link == "local_fx1" else None,
            backend_kwargs,
            byok if link == "byok" else None,
            timeout_s,
        )

    def _resolve_completion_backend(
        self,
        backend: str,
        checkpoint_dir: str | Path | None,
        backend_kwargs: dict[str, Any] | None,
        byok: dict[str, str] | None,
        timeout_s: float | None,
    ) -> Any:
        """Checkpoint contract + backend resolution shared by completes."""
        kwargs: dict[str, Any] = dict(backend_kwargs or {})
        if timeout_s is not None:
            kwargs["timeout_s"] = timeout_s
        if byok is not None:
            if backend != "byok":
                raise ValueError("a byok override applies only to backend='byok'")
            required = {"base_url", "api_key", "model"}
            if set(byok) != required:
                raise ValueError(f"byok needs exactly {sorted(required)}, got {sorted(byok)}")
            problem = byok_base_url_problem(byok["base_url"])
            if problem is not None:
                raise ValueError(f"byok.base_url {problem}")
            kwargs.update(byok)
        if backend == "local_fx1":
            checkpoint = checkpoint_dir or os.environ.get("FX1_CHECKPOINT_DIR")
            if not checkpoint:
                raise ValueError(
                    "local_fx1 needs a checkpoint_dir argument or "
                    "FX1_CHECKPOINT_DIR in the environment"
                )
            kwargs["checkpoint_dir"] = str(checkpoint)
        elif checkpoint_dir is not None:
            raise ValueError("checkpoint_dir applies only to the local_fx1 backend")
        return self._resolve_backend(backend, **kwargs)

    def verify_receipt(self, receipt: dict[str, Any]) -> ReceiptVerdict:
        """Deep-verify a receipt object against the v2 contract."""
        result = verify_receipt_payload(receipt, path=Path("<sdk>"))
        return ReceiptVerdict(
            valid=result["valid"],
            path=result["path"],
            schema_tag=result["schema"],
            kind=result["kind"] if isinstance(result["kind"], str) else None,
            verdict=result["verdict"] if isinstance(result["verdict"], str) else None,
            digest_convention=result["digest_convention"],
            errors=tuple(result["errors"]),
            warnings=tuple(result["warnings"]),
        )

    def receipts(self) -> tuple[ReceiptRef, ...]:
        """Index the local sealed-receipt store — the in-process twin of
        ``GET /receipts``. An absent store raises FileNotFoundError."""
        if not self._receipts.available():
            raise FileNotFoundError(f"receipts store unavailable at {self._receipts.root}")
        return tuple(ReceiptRef(sha256=sha, name=path.name) for sha, path in self._receipts.items())

    def receipt(self, sha256: str) -> StoredReceipt:
        """Fetch one sealed receipt by content hash — the in-process twin of
        ``GET /receipts/{sha256}``: verbatim document plus a live re-verify.
        Unknown hashes raise KeyError; a malformed digest raises ValueError."""
        if SHA256_HEX.fullmatch(sha256) is None:
            raise ValueError(f"malformed receipt sha256: {sha256!r}")
        path = self._receipts.lookup(sha256)
        if path is None or not path.is_file():
            raise KeyError(f"receipt not found: {sha256}")
        return StoredReceipt(
            sha256=sha256,
            document=json.loads(path.read_bytes()),
            valid=bool(verify_receipt_file(path)["valid"]),
        )

    def verify_receipts(self, receipts: list[dict[str, Any]]) -> tuple[ReceiptVerdict, ...]:
        """Batch counterpart: in-process it's a loop; the wire client's twin
        is one POST — same return shape either way."""
        return tuple(self.verify_receipt(r) for r in receipts)

    # ---- health ----------------------------------------------------------

    def health(self) -> HarnessHealth:
        """Configured-backend presence flags — booleans only, never values."""
        from fx1.serve.backends import (
            BYOK_API_KEY_ENV,
            BYOK_BASE_URL_ENV,
            BYOK_MODEL_ENV,
            LOCAL_SERVE_CMD_ENV,
            LOCAL_SERVE_URL_ENV,
        )

        checkpoint_env = os.environ.get("FX1_CHECKPOINT_DIR", "")
        return HarnessHealth(
            status="ok",
            version=__version__,
            registered_commands=len(self._harness.list_commands()),
            backends={
                "hosted_k3": bool(os.environ.get("MOONSHOT_API_KEY")),
                "byok": all(
                    os.environ.get(n) for n in (BYOK_BASE_URL_ENV, BYOK_API_KEY_ENV, BYOK_MODEL_ENV)
                ),
                "local_fx1": bool(checkpoint_env)
                and (Path(checkpoint_env) / "modelcard.json").is_file()
                and bool(
                    os.environ.get(LOCAL_SERVE_URL_ENV) or os.environ.get(LOCAL_SERVE_CMD_ENV)
                ),
            },
        )
