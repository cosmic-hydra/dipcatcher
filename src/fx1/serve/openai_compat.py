"""OpenAI-compatible ingress — the shared translation layer.

The ``/v1`` HTTP routes (``fx1.serve.api``) and the in-process SDK surface
(``fx1.sdk.Fx1Harness.openai_chat``) translate through THIS module — one
pydantic validation object, one backend-resolution order, one error
taxonomy — so the wire and the weights-direct path cannot drift apart.

- *Fail-closed surface* — every OpenAI feature the gated pipeline cannot
  honor (tools, ``logprobs``, non-text ``response_format`` types, content
  parts other than ``text``…) is rejected by
  :class:`OpenAICompatError`, never silently dropped. The
  ``/v1/responses`` surface applies the same rule; ``store`` governs the
  retrieval index (``store=false`` keeps the call out of
  ``GET /v1/responses/{id}`` — the audit ledger still records it).
- *Backend precedence* — ``fx1.backend`` > ``X-Fx1-Backend`` >
  ``model`` naming a backend > ``byok`` when BYOK headers are present >
  ``hosted_k3``. Header-based BYOK treats a non-backend ``model`` as the
  upstream model name (``gpt-4o`` → base_url + key + model).
- *Error mapping* — :func:`openai_to_kwargs` raises
  :class:`OpenAICompatError` (a ``ValueError`` carrying ``status``); the
  API converts it to an :class:`ApiError` with that status, the SDK
  surfaces it as a ``ValueError`` — same verdict, each surface's own
  exception class.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal

import jsonschema
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from fx1.serve.backends import byok_base_url_problem
from fx1.serve.receipt_store import SHA256_HEX
from fx1.serve.webhooks import check_callback_url

if TYPE_CHECKING:
    from fx1.serve.journal import JobJournal

__all__ = [
    "OPENAI_BACKENDS",
    "OPENAI_ERR_TYPES",
    "OPENAI_UNSUPPORTED",
    "OPENAI_MODEL_IDS",
    "OPENAI_RESPONSE_FORMATS",
    "RESPONSE_ITEM_TYPES_REFUSED",
    "RESPONSE_PART_TYPES",
    "RESPONSE_ROLES",
    "RESPONSES_UNSUPPORTED",
    "ByokOverride",
    "OpenAIChatMessage",
    "OpenAIChatResponse",
    "OpenAIChatChoice",
    "OpenAIChatUpdate",
    "OpenAICompatError",
    "OpenAICompletionRequest",
    "OpenAIEmbeddingItem",
    "OpenAIEmbeddingRequest",
    "OpenAIEmbeddingResponse",
    "OpenAIFx1",
    "OpenAIModel",
    "OpenAIModelDelete",
    "OpenAIModelList",
    "OpenAIConversationCreate",
    "OpenAIConversationItemsAdd",
    "OpenAIConversationUpdate",
    "OpenAIFileSearchTool",
    "OpenAIResponseRequest",
    "OpenAIResponseTool",
    "OpenAITool",
    "OpenAIToolFunction",
    "OpenAIVectorStoreCreate",
    "OpenAIVectorStoreFileBatchCreate",
    "OpenAIVectorStoreFileCreate",
    "OpenAIVectorStoreSearch",
    "OpenAIVectorStoreUpdate",
    "OPENAI_BATCH_ENDPOINTS",
    "OPENAI_BATCH_LINE_MAX",
    "OPENAI_FILE_BYTES_MAX",
    "OPENAI_FILE_PURPOSE_ACCEPT",
    "OpenAIBatchRequest",
    "OpenAIUploadCompleteRequest",
    "OpenAIUploadCreateRequest",
    "batch_line_body",
    "batch_line_shape",
    "chat_messages_for_store",
    "completion_events",
    "embeddings_to_kwargs",
    "openai_embedding_envelope",
    "batch_object",
    "batch_output_line",
    "file_object",
    "is_openai_path",
    "legacy_to_chat",
    "openai_chunks",
    "openai_completion_envelope",
    "openai_conversation_object",
    "openai_envelope",
    "openai_error_body",
    "openai_messages",
    "openai_model",
    "openai_models",
    "openai_response_call_items",
    "openai_response_events",
    "openai_response_object",
    "openai_response_replay_events",
    "response_output_pieces",
    "openai_to_kwargs",
    "openai_usage",
    "file_search_call_item",
    "paged_item_list",
    "response_query_text",
    "OpenAIEnvelopeStore",
    "OPENAI_RESPONSE_TERMINAL",
    "chained_response_input",
    "conversation_id_of",
    "response_cap_call_items",
    "response_input_item_dicts",
    "response_input_items_for_store",
    "response_input_to_messages",
    "response_text_format",
    "response_to_kwargs",
    "validate_openai_output",
    "validate_response_format",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OpenAICompatError(ValueError):
    """A rejection at the OpenAI translation layer.

    ``status`` is the HTTP status the wire maps the rejection to; on the
    SDK surface the exception itself is the verdict (a ``ValueError``, the
    SDK's request-error class). ``code`` lands in the OpenAI error
    envelope's ``code`` field when the wire emits it.
    """

    def __init__(self, message: str, *, status: int = 400, code: str = "invalid_request") -> None:
        super().__init__(message)
        self.status = status
        self.code = code


# Backend names an OpenAI `model` field or `X-Fx1-Backend` header may carry.
OPENAI_BACKENDS = frozenset({"hosted_k3", "local_fx1", "byok"})

_STOP_SEQ_ERR = "stop sequences must be 1–512 char strings"

# response_format types the gated pipeline honors. `text` is freeform;
# `json_object`/`json_schema` are post-validated against the returned text
# (the harness can't constrain-decode arbitrary backends — validation is
# the honest mechanism, and a violation is a provider-side 502).
OPENAI_RESPONSE_FORMATS = frozenset({"text", "json_object", "json_schema"})

# The GET /v1/models inventory — `fx1` is the alias for the default link.
OPENAI_MODEL_IDS = ("fx1", "hosted_k3", "local_fx1", "byok")

# Fields a caller may send that the gated pipeline cannot honor. Naming the
# field beats silently dropping it — honest compat over fake compat.
# ``store`` is honored, not refused: it governs the retrieval index behind
# ``GET /v1/chat/completions/{id}`` (the audit ledger records every call
# regardless — retrieval is a convenience surface, not the evidence).
OPENAI_UNSUPPORTED = (
    # legacy function-calling fields — superseded by ``tools`` /
    # ``tool_choice`` (which the pipeline honors; see OpenAITool)
    "functions",
    "function_call",
    "modalities",
    "audio",
    "prediction",
    "web_search_options",
    "suffix",
    "echo",
    "best_of",
)

# Legacy-only knobs the completions surface cannot honor. ``logprobs``
# needs a token-level scorer the pipeline does not expose; ``suffix`` is
# fill-in-middle; ``best_of`` is a logprob-ranked rerank — all three
# refuse 422 instead of silently dropping.
LEGACY_UNSUPPORTED = ("suffix", "best_of", "logprobs")

# OpenAI's `type` names per status — the error envelope stays SDK-faithful.
OPENAI_ERR_TYPES = {
    400: "invalid_request_error",
    404: "invalid_request_error",
    409: "invalid_request_error",
    413: "invalid_request_error",
    422: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    429: "rate_limit_error",
    503: "service_unavailable",
}


def openai_error_body(message: str, status: int, code: str) -> dict[str, Any]:
    """The OpenAI error envelope — `{error: {message, type, param, code}}`.

    ``code`` carries the harness's own code (``honesty_gate``,
    ``over_capacity``, ``draining``, …) so clients can branch on it.
    """
    return {
        "error": {
            "message": message,
            "type": OPENAI_ERR_TYPES.get(
                status, "server_error" if status >= 500 else "invalid_request_error"
            ),
            "param": None,
            "code": code,
        }
    }


def is_openai_path(path: str) -> bool:
    """Whether a request path is under the OpenAI surface."""
    return path.startswith("/v1")


class ByokOverride(_Model):
    """Per-request BYOK credentials — the caller's own OpenAI-compatible
    endpoint. ``api_key`` is used for the upstream call only; it is never
    logged and never echoed into error text. The idempotency fingerprint
    is hashed, so the stored dedupe record carries no secret bytes."""

    base_url: str = Field(min_length=1, max_length=2048)
    api_key: str = Field(min_length=1, max_length=4096)
    model: str = Field(min_length=1, max_length=512)

    @field_validator("base_url")
    @classmethod
    def _http_url(cls, v: str) -> str:
        # the shared wire rule — never echoes the URL back (it may
        # carry userinfo secrets a caller pasted by mistake)
        problem = byok_base_url_problem(v)
        if problem is not None:
            raise ValueError(f"byok.base_url {problem}")
        return v


class OpenAIChatMessage(_Model):
    """One chat message — content may be a string or an OpenAI
    content-part list; non-text parts are rejected at translation.
    ``tool_calls`` (assistant) and ``tool_call_id`` (role ``tool``) pass
    through to tool-capable links — agent loops need both halves."""

    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[dict[str, Any]] | None = None


class OpenAIToolFunction(_Model):
    """One ``tools[].function`` — the callable spec an agent advertises."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    description: str | None = Field(default=None, max_length=8192)
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class OpenAITool(_Model):
    """One ``tools[]`` entry — only ``type: \"function\"`` exists on the
    OpenAI surface today; anything else fails closed at validation."""

    model_config = ConfigDict(extra="allow")

    type: Literal["function"] = "function"
    function: OpenAIToolFunction


class OpenAIFx1(_Model):
    """The ``fx1`` extension object: harness knobs that have no OpenAI
    field — backend selection, fallbacks, BYOK credentials, the local
    checkpoint dir, receipt citations, and the per-call deadline."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["hosted_k3", "local_fx1", "byok"] | None = None
    fallbacks: list[Literal["hosted_k3", "local_fx1", "byok"]] = Field(
        default_factory=list, max_length=2
    )
    byok: ByokOverride | None = None
    checkpoint_dir: str | None = Field(
        default=None, min_length=1, max_length=4096, pattern=r"^[^\x00]+$"
    )
    receipt_hashes: list[str] | None = None
    timeout_s: float | None = Field(default=None, gt=0, le=3600)


class OpenAIChatRequest(_Model):
    """POST /v1/chat/completions body — the OpenAI surface, extra fields
    tolerated (SDKs send bookkeeping keys like ``user``)."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(default="fx1", min_length=1, max_length=512)
    messages: list[OpenAIChatMessage] = Field(min_length=1, max_length=512)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0, le=262144)
    max_completion_tokens: int | None = Field(default=None, gt=0, le=262144)
    seed: int | None = Field(default=None, ge=0)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    n: int = Field(default=1, ge=1, le=8)
    stop: str | list[str] | None = None
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    logit_bias: dict[str, int] | None = None
    user: str | None = Field(default=None, max_length=512)
    metadata: dict[str, str] | None = None
    service_tier: Literal["auto", "default", "flex", "priority", "scale"] | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None
    verbosity: Literal["low", "medium", "high"] | None = None
    prompt_cache_key: str | None = Field(default=None, max_length=128)
    prompt_cache_retention: Literal["in-memory", "24h"] | None = None
    response_format: dict[str, Any] | None = None
    store: bool | None = None
    tools: list[OpenAITool] | None = None
    tool_choice: Literal["none", "auto", "required"] | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    fx1: OpenAIFx1 | None = None

    @model_validator(mode="after")
    def _openai_valid(self) -> OpenAIChatRequest:
        if isinstance(self.stop, list):
            if len(self.stop) > 4:
                raise ValueError("stop accepts at most 4 sequences")
            if any(not isinstance(s, str) or not 1 <= len(s) <= 512 for s in self.stop):
                raise ValueError(_STOP_SEQ_ERR)
        elif isinstance(self.stop, str) and not 1 <= len(self.stop) <= 512:
            raise ValueError(_STOP_SEQ_ERR)
        if self.logit_bias is not None:
            for key, bias in self.logit_bias.items():
                try:
                    int(key)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"logit_bias keys must be token ids, got {key!r}") from exc
                if not -100 <= bias <= 100:
                    raise ValueError(f"logit_bias[{key!r}]={bias} outside [-100, 100]")
        if (
            self.max_completion_tokens is not None
            and self.max_tokens is not None
            and self.max_completion_tokens != self.max_tokens
        ):
            raise ValueError(
                "max_tokens and max_completion_tokens disagree "
                f"({self.max_tokens} vs {self.max_completion_tokens})"
            )
        if self.metadata is not None:
            if len(self.metadata) > 16:
                raise ValueError("metadata accepts at most 16 entries")
            for k, v in self.metadata.items():
                if len(k) > 64 or len(v) > 512:
                    raise ValueError("metadata keys are ≤64 chars, values ≤512")
        rf = self.response_format
        if rf is not None:
            rtype = rf.get("type", "text")
            if rtype not in OPENAI_RESPONSE_FORMATS:
                raise ValueError(
                    "response_format.type must be one of "
                    f"{sorted(OPENAI_RESPONSE_FORMATS)}; got {rtype!r}"
                )
            if rtype == "json_schema":
                spec = rf.get("json_schema")
                schema = spec.get("schema") if isinstance(spec, dict) else None
                if not isinstance(schema, dict):
                    raise ValueError(
                        "response_format json_schema needs "
                        "{type: 'json_schema', json_schema: {name?, schema: {...}}}"
                    )
                try:
                    jsonschema.validators.validator_for(schema).check_schema(schema)
                except jsonschema.SchemaError as exc:
                    raise ValueError(
                        f"response_format json_schema is not a valid schema: {exc.message}"
                    ) from exc
        if self.tools is not None and len(self.tools) > 128:
            raise ValueError("tools accepts at most 128 entries")
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
        present = [f for f in OPENAI_UNSUPPORTED if getattr(self, f, None) is not None]
        # extras (extra="allow") that are also unsupported features
        extra_bad = sorted(f for f in OPENAI_UNSUPPORTED if f in (self.__pydantic_extra__ or {}))
        bad = sorted(set(present) | set(extra_bad))
        if bad:
            raise ValueError(f"unsupported for the gated pipeline: {', '.join(bad)}")
        return self


class OpenAIModel(_Model):
    """One entry of GET /v1/models — `id` is the name `model` may carry."""

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "dipcatcher"


class OpenAIModelList(_Model):
    """GET /v1/models body — the OpenAI `list` envelope."""

    object: Literal["list"] = "list"
    data: list[OpenAIModel]


class OpenAIModelDelete(_Model):
    """DELETE /v1/models/{id} — OpenAI's delete verdict: the removed id
    plus the boolean tombstone."""

    id: str
    object: Literal["model"] = "model"
    deleted: bool = True


class OpenAIChatChoice(_Model):
    """One choice of a `chat.completion` — the gated text lands here.
    ``message`` may carry ``tool_calls`` (content then null);
    ``logprobs`` is the verbatim provider payload when the request asked
    for it (null otherwise); ``finish_reason`` is the upstream's own
    verdict."""

    index: int
    message: dict[str, Any]
    finish_reason: str
    logprobs: dict[str, Any] | None = None


class OpenAIChatResponse(_Model):
    """POST /v1/chat/completions 200 body — the `chat.completion`
    envelope. `system_fingerprint` carries the serving backend name;
    `usage` is null when the provider didn't report token counts."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    system_fingerprint: str
    choices: list[OpenAIChatChoice]
    usage: dict[str, int] | None = None
    metadata: dict[str, str] | None = None


class OpenAICompletionRequest(_Model):
    """POST /v1/completions body — the legacy ``text_completion`` surface
    that ``client.completions.create`` and older agents still target.

    ``prompt`` is a string or a list of strings — one completion chain
    per element; ``n`` repeats within each element, so a list of ``k``
    prompts with ``n`` repeats produces ``k * n`` flat choices. ``echo``
    prepends the prompt to each choice's ``text`` (the one legacy flag
    the surface honors verbatim). ``max_tokens`` defaults to 16 at
    translation, matching OpenAI's legacy default; ``suffix``,
    ``best_of`` and ``logprobs`` refuse — the pipeline cannot honor
    them. ``store`` is tolerated and ignored: completions have no
    retrieval twin (there is no ``GET /v1/completions/{id}`` — a pinned
    ``Idempotency-Key`` replay is the retrieval path).
    """

    model_config = ConfigDict(extra="allow")

    model: str = Field(default="fx1", min_length=1, max_length=512)
    prompt: str | list[str]
    max_tokens: int | None = Field(default=None, gt=0, le=262144)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=8)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    stop: str | list[str] | None = None
    seed: int | None = Field(default=None, ge=0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    logit_bias: dict[str, int] | None = None
    user: str | None = Field(default=None, max_length=512)
    metadata: dict[str, str] | None = None
    echo: bool = False
    store: bool | None = None
    fx1: OpenAIFx1 | None = None

    @model_validator(mode="after")
    def _legacy_valid(self) -> OpenAICompletionRequest:  # NOSONAR(S3776)
        prompts = [self.prompt] if isinstance(self.prompt, str) else list(self.prompt)
        if not prompts:
            raise ValueError("prompt must be a non-empty list when a list is given")
        if len(prompts) > 512:
            raise ValueError("prompt list accepts at most 512 entries")
        for p in prompts:
            if not isinstance(p, str) or not p or len(p) > 131072:
                raise ValueError("prompt entries must be 1–131072 char strings")
        if isinstance(self.stop, list):
            if len(self.stop) > 4:
                raise ValueError("stop accepts at most 4 sequences")
            if any(not isinstance(s, str) or not 1 <= len(s) <= 512 for s in self.stop):
                raise ValueError(_STOP_SEQ_ERR)
        elif isinstance(self.stop, str) and not 1 <= len(self.stop) <= 512:
            raise ValueError(_STOP_SEQ_ERR)
        if self.logit_bias is not None:
            for key, bias in self.logit_bias.items():
                try:
                    int(key)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"logit_bias keys must be token ids, got {key!r}") from exc
                if not -100 <= bias <= 100:
                    raise ValueError(f"logit_bias[{key!r}]={bias} outside [-100, 100]")
        if self.metadata is not None:
            if len(self.metadata) > 16:
                raise ValueError("metadata accepts at most 16 entries")
            for k, v in self.metadata.items():
                if len(k) > 64 or len(v) > 512:
                    raise ValueError("metadata keys are ≤64 chars, values ≤512")
        extra = self.__pydantic_extra__ or {}
        bad = sorted(f for f in LEGACY_UNSUPPORTED if extra.get(f) is not None)
        if bad:
            raise ValueError(f"unsupported for the gated pipeline: {', '.join(bad)}")
        return self


def openai_models(
    *,
    created: int | None = None,
    extra_ids: Iterable[str] = (),
    created_by_id: Mapping[str, int] | None = None,
) -> OpenAIModelList:
    """The model inventory — `fx1` plus the backend names `model` may
    carry. ``created`` defaults to call time for the built-ins;
    ``created_by_id`` stamps each extra (registry) card with the id's own
    recorded creation time."""
    ts = int(time.time()) if created is None else created
    stamps = created_by_id or {}
    extra = [m for m in dict.fromkeys(extra_ids) if m not in OPENAI_MODEL_IDS]
    return OpenAIModelList(
        data=[
            OpenAIModel(id=m, created=int(stamps.get(m, ts)) if m in stamps else ts)
            for m in (*OPENAI_MODEL_IDS, *sorted(extra))
        ]
    )


def openai_model(
    model_id: str,
    *,
    created: int | None = None,
    extra_ids: Iterable[str] = (),
    created_by_id: Mapping[str, int] | None = None,
) -> OpenAIModel:
    """One model card — ``GET /v1/models/{id}`` retrieve semantics.

    Unknown ids fail closed 404 (OpenAI's ``invalid_request_error`` /
    ``model_not_found``) — an SDK's ``models.retrieve`` never gets a
    fabricated card. ``extra_ids`` admits registered ``ft:`` models;
    ``created_by_id`` stamps an extra card with its own recorded
    creation time instead of the serve boot stamp.
    """
    if model_id not in OPENAI_MODEL_IDS and model_id not in frozenset(extra_ids):
        raise OpenAICompatError(
            f"The model '{model_id}' does not exist", status=404, code="model_not_found"
        )
    stamps = created_by_id or {}
    ts = int(time.time()) if created is None else created
    if model_id in stamps:
        ts = int(stamps[model_id])
    return OpenAIModel(id=model_id, created=ts)


def _history_tool_call_shape(raw: Any, i: int, j: int) -> dict[str, Any]:
    """One entry of an assistant message's ``tool_calls`` history —
    validated before it rides the wire so a malformed replay is a 400,
    not upstream confusion."""
    fn = raw.get("function") if isinstance(raw, dict) else None
    args = fn.get("arguments") if isinstance(fn, dict) else None
    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("id"), str)
        or raw.get("type") != "function"
        or not isinstance(fn, dict)
        or not isinstance(fn.get("name"), str)
        or not isinstance(args, str)
    ):
        raise OpenAICompatError(
            f"messages[{i}].tool_calls[{j}]: needs "
            "{id, type: 'function', function: {name, arguments}}"
        )
    return raw


def openai_messages(msgs: list[OpenAIChatMessage]) -> list[dict[str, Any]]:
    """Flatten OpenAI messages to harness message dicts.

    Content-part lists keep only ``{"type": "text"}`` entries; any other
    part type fails closed — silently dropping caller content is a lie.
    Tool context passes through verbatim: an assistant turn's
    ``tool_calls`` and a ``tool`` role's ``tool_call_id``/``name`` are
    required for agent loops — the backend-level capability check
    (``ToolBackend``) decides whether the link can honor them.
    """
    out: list[dict[str, Any]] = []
    for i, m in enumerate(msgs):
        tool_calls = getattr(m, "tool_calls", None)
        tool_call_id = getattr(m, "tool_call_id", None)
        name = getattr(m, "name", None)
        if tool_call_id is not None and m.role != "tool":
            raise OpenAICompatError(f"messages[{i}]: tool_call_id belongs on role 'tool'")
        if tool_calls is not None and m.role != "assistant":
            raise OpenAICompatError(f"messages[{i}]: tool_calls belongs on role 'assistant'")
        content = m.content
        if content is None and tool_calls is None and m.role != "tool":
            raise OpenAICompatError(f"messages[{i}]: content is required")
        if isinstance(content, list):
            parts: list[str] = []
            for j, part in enumerate(content):
                if part.get("type") != "text":
                    raise OpenAICompatError(
                        f"messages[{i}].content[{j}]: only 'text' parts are supported",
                    )
                parts.append(str(part.get("text", "")))
            content = "".join(parts)
        flat: dict[str, Any] = {"role": m.role, "content": content}
        if tool_calls is not None:
            if not isinstance(tool_calls, list) or not tool_calls:
                raise OpenAICompatError(f"messages[{i}]: tool_calls must be a non-empty list")
            flat["tool_calls"] = [
                _history_tool_call_shape(raw, i, j) for j, raw in enumerate(tool_calls)
            ]
        if m.role == "tool":
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise OpenAICompatError(
                    f"messages[{i}]: role 'tool' requires a non-empty tool_call_id"
                )
            flat["tool_call_id"] = tool_call_id
            if content is None:
                raise OpenAICompatError(f"messages[{i}]: tool output is required")
        if name is not None:
            if not isinstance(name, str) or not name:
                raise OpenAICompatError(f"messages[{i}]: name must be a non-empty string")
            flat["name"] = name
        out.append(flat)
    return out


def _resolve_openai_link(
    model: str,
    ext: OpenAIFx1 | None,
    hdrs: dict[str, str],
    *,
    ft_resolver: Callable[[str], str | None] | None = None,
    require_known_model: bool = False,
) -> tuple[str, list[str], str | None, ByokOverride | None, str | None]:
    """Backend resolution shared by the chat and responses translators —
    ``fx1.backend`` > ``X-Fx1-Backend`` > a ``model`` naming a backend >
    ``byok`` when BYOK headers are present > ``hosted_k3``. Returns
    ``(backend, fallbacks, checkpoint_dir, byok, served_model)`` where
    ``served_model`` is the ``ft:`` name the registry resolved — the name
    the caller addressed — or ``None`` on every other path.

    A ``model`` of the form ``ft:*`` names a registered fine-tuned model:
    when no explicit backend was chosen (no ext/backend header and no
    BYOK headers), it resolves to the ``local_fx1`` lane pinned at the
    producing job's checkpoint. An unregistered ``ft:`` name is a
    fail-closed 404 ``model_not_found`` — never a silent default link.

    ``require_known_model`` arms the same refusal for every other
    unregistered name: the completion surfaces (chat, responses,
    messages, count_tokens) pass it, so a typo'd ``model`` answers 404
    ``model_not_found`` instead of silently running the default link.
    The embeddings surface leaves it off — there ``model`` is the
    upstream deployment name passed verbatim to the link."""
    byok_headers = hdrs.get("x-fx1-byok-base-url")
    explicit = (ext.backend if ext is not None else None) or hdrs.get("x-fx1-backend")
    if (
        explicit is None
        and model not in OPENAI_BACKENDS
        and not byok_headers
        and model.startswith("ft:")
    ):
        if ext is not None and ext.byok is not None:
            # an ft: name pins the local_fx1 lane — per-request BYOK
            # credentials have no link to bind, so refusing beats
            # silently dropping them (a dropped credential is a
            # silently misrouted call).
            raise OpenAICompatError(
                "a byok override applies only to a 'byok' link in the chain",
                status=422,
            )
        if (ext is not None and ext.checkpoint_dir is not None) or hdrs.get(
            "x-fx1-checkpoint-dir"
        ) is not None:
            # an ft: name resolves through the registry's own checkpoint —
            # a checkpoint_dir override has no link to bind and would
            # otherwise drop silently (a dropped override is a silently
            # misrouted call, same class as the byok refusal above).
            raise OpenAICompatError(
                "an ft: model pins its registered checkpoint — checkpoint_dir cannot override it",
                status=422,
            )
        checkpoint = ft_resolver(model) if ft_resolver is not None else None
        if checkpoint is None:
            raise OpenAICompatError(
                f"The model '{model}' does not exist",
                status=404,
                code="model_not_found",
            )
        fallbacks_ft: list[str] = list(ext.fallbacks) if ext is not None else []
        if not fallbacks_ft and hdrs.get("x-fx1-fallbacks"):
            fallbacks_ft = [f.strip() for f in hdrs["x-fx1-fallbacks"].split(",") if f.strip()]
        return "local_fx1", fallbacks_ft, checkpoint, None, model
    if (
        require_known_model
        and explicit is None
        and model not in OPENAI_MODEL_IDS
        and not byok_headers
        and (ext is None or ext.byok is None)
    ):
        raise OpenAICompatError(
            f"The model '{model}' does not exist",
            status=404,
            code="model_not_found",
        )
    backend = (
        explicit
        or (model if model in OPENAI_BACKENDS else None)
        # BYOK headers present and no explicit backend → the model string is
        # the upstream model (e.g. "gpt-4o"), the link is byok.
        or ("byok" if byok_headers else "hosted_k3")
    )
    if backend not in OPENAI_BACKENDS:
        raise OpenAICompatError(f"unknown backend {backend!r}")
    fallbacks: list[str] = list(ext.fallbacks) if ext is not None else []
    if not fallbacks and hdrs.get("x-fx1-fallbacks"):
        fallbacks = [f.strip() for f in hdrs["x-fx1-fallbacks"].split(",") if f.strip()]
    checkpoint_dir = (ext.checkpoint_dir if ext is not None else None) or hdrs.get(
        "x-fx1-checkpoint-dir"
    )
    byok = ext.byok if ext is not None else None
    byok_header_names = (
        "x-fx1-byok-base-url",
        "x-fx1-byok-api-key",
        "x-fx1-byok-model",
    )
    if byok is None and any(name in hdrs for name in byok_header_names) and not byok_headers:
        raise OpenAICompatError(
            "X-Fx1-Byok-* headers require X-Fx1-Byok-Base-Url",
            status=400,
            code="invalid_byok_headers",
        )
    if byok is None and byok_headers:
        api_key = hdrs.get("x-fx1-byok-api-key")
        byok_model = hdrs.get("x-fx1-byok-model") or (
            model if model not in OPENAI_BACKENDS else None
        )
        if not api_key:
            raise OpenAICompatError("X-Fx1-Byok-Base-Url requires X-Fx1-Byok-Api-Key")
        if not byok_model:
            raise OpenAICompatError("byok needs a model — set X-Fx1-Byok-Model or body.model")
        try:
            byok = ByokOverride(base_url=byok_headers, api_key=api_key, model=byok_model)
        except ValidationError as exc:
            # the body path reports these as request-validation 422s —
            # the header path must land the same verdict, never a bare
            # 500 when the exception escapes the translator.
            raise OpenAICompatError(
                "; ".join(str(e.get("msg", "invalid byok override")) for e in exc.errors()),
                status=422,
                code="invalid_byok_headers",
            ) from exc
    if byok is not None and "byok" not in {backend, *fallbacks}:
        raise OpenAICompatError(
            "a byok override applies only to a 'byok' link in the chain", status=422
        )
    if checkpoint_dir is not None and "local_fx1" not in {backend, *fallbacks}:
        raise OpenAICompatError(
            "checkpoint_dir applies only to a 'local_fx1' link in the chain", status=422
        )
    return backend, fallbacks, checkpoint_dir, byok, None


def _resolve_timeout(ext_timeout: float | None, hdrs: dict[str, str]) -> float | None:
    """Per-request backend timeout: ``fx1.timeout_s`` > ``X-Fx1-Timeout``
    header (seconds). A malformed header is a fail-closed 400 — never a
    silent default."""
    if ext_timeout is not None:
        return ext_timeout
    raw = hdrs.get("x-fx1-timeout")
    if raw is None:
        return None
    try:
        val = float(raw)
    except ValueError as exc:
        raise OpenAICompatError("X-Fx1-Timeout must be seconds as a number") from exc
    if not math.isfinite(val) or val <= 0.0 or val > 3600.0:
        raise OpenAICompatError("X-Fx1-Timeout must be in (0, 3600] seconds")
    return val


def _resolve_receipt_hashes(ext_hashes: list[str] | None, hdrs: dict[str, str]) -> list[str] | None:
    """Evidence citations: ``fx1.receipt_hashes`` > ``X-Fx1-Receipt-Hashes``
    header (comma-separated sha256 digests — the knob for clients that
    can't edit the JSON body, same channel as ``X-Fx1-Fallbacks``). A
    malformed digest is a fail-closed 400; resolvability stays with the
    mounted store's check downstream."""
    if ext_hashes:
        if any(SHA256_HEX.fullmatch(h) is None for h in ext_hashes):
            # the body channel enforces the same digest shape as the
            # header — a garbage citation must never reach the Evidence
            # footer unchallenged (it stays advisory only when it is a
            # well-formed digest the store cannot resolve).
            raise OpenAICompatError("fx1.receipt_hashes entries must be sha256 digests")
        return list(ext_hashes)
    raw = hdrs.get("x-fx1-receipt-hashes")
    if raw is None:
        return None
    out = [h.strip() for h in raw.split(",") if h.strip()]
    if any(SHA256_HEX.fullmatch(h) is None for h in out):
        raise OpenAICompatError("X-Fx1-Receipt-Hashes must be comma-separated sha256 digests")
    return out or None


def openai_to_kwargs(
    body: OpenAIChatRequest,
    headers: Mapping[str, str] | None = None,
    *,
    ft_resolver: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Translate an OpenAI request into ``complete`` kwargs.

    Returns the keyword set both surfaces consume: the API wraps it in a
    ``CompleteRequest`` (model-level validation → 422), the SDK passes it
    to ``Fx1Harness.complete``. Keys: ``messages`` (``{role, content}``
    dicts), ``backend``, ``fallbacks``, ``byok`` (a validated dict),
    ``checkpoint_dir``, ``receipt_hashes``, ``timeout_s``, ``temperature``,
    ``top_p``, ``max_tokens``, ``seed``.

    Backend resolution order: ``fx1.backend`` > ``X-Fx1-Backend`` header >
    a ``model`` naming a backend > ``hosted_k3``. BYOK credentials come
    from ``fx1.byok`` or the ``X-Fx1-Byok-*`` headers — ``X-Fx1-Byok-Model``
    defaults to the request ``model`` when it isn't a backend name.
    """
    hdrs = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    ext = body.fx1
    backend, fallbacks, checkpoint_dir, byok, served_model = _resolve_openai_link(
        body.model, ext, hdrs, ft_resolver=ft_resolver, require_known_model=True
    )
    return {
        "backend": backend,
        "messages": openai_messages(body.messages),
        "_served_model": served_model,
        "checkpoint_dir": checkpoint_dir,
        "receipt_hashes": _resolve_receipt_hashes(
            ext.receipt_hashes if ext is not None else None, hdrs
        ),
        "timeout_s": _resolve_timeout(ext.timeout_s if ext is not None else None, hdrs),
        "fallbacks": fallbacks,
        "byok": byok.model_dump() if byok is not None else None,
        "temperature": body.temperature,
        "top_p": body.top_p,
        "max_tokens": (
            body.max_completion_tokens
            if body.max_completion_tokens is not None
            else body.max_tokens
        ),
        "seed": body.seed,
        "stop": (
            [body.stop]
            if isinstance(body.stop, str)
            else (list(body.stop) or None)
            if body.stop is not None
            else None
        ),
        "presence_penalty": body.presence_penalty,
        "frequency_penalty": body.frequency_penalty,
        "logit_bias": body.logit_bias,
        "user": body.user,
        "metadata": body.metadata,
        "service_tier": body.service_tier,
        "reasoning_effort": body.reasoning_effort,
        "verbosity": body.verbosity,
        "prompt_cache_key": body.prompt_cache_key,
        "prompt_cache_retention": body.prompt_cache_retention,
        "tools": ([t.model_dump(exclude_none=True) for t in body.tools] if body.tools else None),
        "tool_choice": body.tool_choice,
        "parallel_tool_calls": body.parallel_tool_calls,
        "logprobs": body.logprobs,
        "top_logprobs": body.top_logprobs,
    }


def validate_response_format(rf: dict[str, Any] | None, content: str) -> None:
    """Post-validate a gated completion against a response_format dict.

    The harness can't constrain-decode arbitrary providers (BYOK),
    so the gate's second pass verifies the returned text: a
    non-conforming output is a provider-side failure — 502 /
    ``format_violation``, never silently shipped and never pinned
    into an idempotency record.
    """
    if not rf:
        return
    rtype = rf.get("type", "text")
    if rtype == "text":
        return
    try:
        parsed: Any = json.loads(content)
    except json.JSONDecodeError as exc:
        raise OpenAICompatError(
            f"model output is not valid JSON under response_format {rtype!r}: {exc}",
            status=502,
            code="format_violation",
        ) from exc
    if rtype == "json_object" and not isinstance(parsed, dict):
        raise OpenAICompatError(
            f"model output under json_object must be a JSON object, got {type(parsed).__name__}",
            status=502,
            code="format_violation",
        )
    if rtype == "json_schema":
        schema = rf["json_schema"]["schema"]
        try:
            jsonschema.validate(parsed, schema)
        except jsonschema.ValidationError as exc:
            raise OpenAICompatError(
                "model output failed the declared json_schema: "
                f"{exc.message} (at {list(exc.absolute_path)})",
                status=502,
                code="format_violation",
            ) from exc


def validate_openai_output(body: OpenAIChatRequest, content: str) -> None:
    """Post-validate a gated chat completion — thin shim over
    :func:`validate_response_format` on ``body.response_format``."""
    validate_response_format(body.response_format, content)


def openai_usage(usage: dict[str, int] | None) -> dict[str, int] | None:
    """Usage dict in the OpenAI key set — the BYOK/hosted backends already
    report prompt/completion/total; anything else passes through so the
    evidence field stays faithful."""
    if not isinstance(usage, dict):
        return None
    return {k: v for k, v in usage.items() if isinstance(v, int) and not isinstance(v, bool)}


def openai_envelope(
    *,
    cid: str,
    content: str | list[str],
    backend: str,
    model: str | None = None,
    usage: dict[str, int] | None = None,
    tool_calls: Sequence[Sequence[dict[str, Any]] | None] | None = None,
    finish_reasons: Sequence[str] | None = None,
    created: int | None = None,
    logprobs: Sequence[dict[str, Any] | None] | None = None,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """A gated result → the `chat.completion` envelope. `model` reports
    the serving link's own model id (or the backend name); the completion
    id mints the `chatcmpl-` handle. ``content`` accepts the ``n>1``
    choice list — one entry per completion, index-ordered.

    ``tool_calls``/``finish_reasons``/``logprobs`` are per-choice
    upstream-verbatim values (index-ordered like ``content``). A choice
    carrying tool calls with no assistant text emits ``content: null`` —
    OpenAI's own encoding for a pure tool-call turn. ``logprobs`` is the
    provider's ``choices[i].logprobs`` payload verbatim (null when the
    provider stayed silent — the field always serializes, matching
    OpenAI's envelope shape)."""
    contents = [content] if isinstance(content, str) else list(content)
    calls = list(tool_calls or [])
    reasons = list(finish_reasons or [])
    lps = list(logprobs or [])
    choices: list[dict[str, Any]] = []
    for i, text in enumerate(contents):
        tc = calls[i] if i < len(calls) else None
        message: dict[str, Any] = {
            "role": "assistant",
            "content": None if (tc and not text) else text,
        }
        if tc:
            message["tool_calls"] = list(tc)
        choices.append(
            {
                "index": i,
                "message": message,
                "finish_reason": reasons[i] if i < len(reasons) else "stop",
                "logprobs": lps[i] if i < len(lps) else None,
            }
        )
    return {
        "id": f"chatcmpl-{cid}",
        "object": "chat.completion",
        "created": int(time.time()) if created is None else created,
        "model": model or backend,
        "system_fingerprint": backend,
        "choices": choices,
        "usage": openai_usage(usage),
        "metadata": metadata,
    }


def legacy_to_chat(body: OpenAICompletionRequest, prompt: str) -> OpenAIChatRequest:
    """Translate one legacy ``prompt`` element into the chat request the
    gated pipeline runs — the prompt becomes a single user turn; the
    decode knobs pass through unchanged (``max_tokens`` defaults to 16,
    OpenAI's own legacy default). Validation of the synthesized body is
    the chat surface's own, so a field the chat validator refuses fails
    identically here."""
    return OpenAIChatRequest.model_validate(
        {
            "model": body.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": body.max_tokens if body.max_tokens is not None else 16,
            "temperature": body.temperature,
            "top_p": body.top_p,
            "n": body.n,
            "stream": False,
            "stop": body.stop,
            "seed": body.seed,
            "presence_penalty": body.presence_penalty,
            "frequency_penalty": body.frequency_penalty,
            "logit_bias": body.logit_bias,
            "user": body.user,
            "metadata": body.metadata,
            # completions have no retrieval twin — never pin into the
            # stored-completion index
            "store": False,
            **({"fx1": body.fx1.model_dump()} if body.fx1 is not None else {}),
        }
    )


def openai_completion_envelope(  # NOSONAR(S3776) — per-choice mapping is a flat loop by contract
    *,
    cid: str,
    envs: Sequence[dict[str, Any]],
    prompts: Sequence[str],
    echo: bool = False,
) -> dict[str, Any]:
    """Map gated ``chat.completion`` envelopes onto the legacy
    ``text_completion`` object — one flat ``choices`` array,
    index-ordered across prompt elements × ``n`` repeats.

    ``choices[i].logprobs`` always serializes ``null`` (never fabricated
    token scores); ``echo`` prepends the element's own prompt text;
    ``usage`` sums the per-element provider-reported counts — ``null``
    when every element stayed silent. ``id`` mints the ``cmpl-`` handle
    from the first completion's log id."""
    choices: list[dict[str, Any]] = []
    usage_sum: dict[str, int] = {}
    usage_seen = False
    created = int(time.time())
    backends: dict[str, None] = {}
    models: dict[str, None] = {}
    for env, prompt in zip(envs, prompts, strict=True):
        created = int(env["created"])
        models[str(env["model"])] = None
        backends[str(env["system_fingerprint"])] = None
        for ch in env["choices"]:
            msg = ch["message"]
            text = msg.get("content") or ""
            choices.append(
                {
                    "index": len(choices),
                    "text": f"{prompt}{text}" if echo else text,
                    "logprobs": None,
                    "finish_reason": ch.get("finish_reason", "stop"),
                }
            )
        u = env.get("usage")
        if isinstance(u, dict):
            usage_seen = True
            for k, v in u.items():
                if isinstance(v, int) and not isinstance(v, bool):
                    usage_sum[k] = usage_sum.get(k, 0) + v
    return {
        "id": f"cmpl-{cid}",
        "object": "text_completion",
        "created": created,
        "model": "+".join(models) if models else "fx1",
        "system_fingerprint": "+".join(backends),
        "choices": choices,
        "usage": usage_sum if usage_seen else None,
    }


def completion_events(
    env: dict[str, Any],
    *,
    include_usage: bool = False,
) -> Iterator[dict[str, Any]]:
    """`text_completion` chunk payloads over a gated legacy envelope —
    the same whitespace chunking as ``openai_chunks``, one content frame
    per ~64-char piece then a terminal frame carrying the choice's
    ``finish_reason`` (per-index grouped, spec-legal). ``include_usage``
    appends the ``choices: []`` usage frame the wire's ``[DONE]``
    follows."""
    base: dict[str, Any] = {
        "id": env["id"],
        "object": "text_completion",
        "created": env["created"],
        "model": env["model"],
        "system_fingerprint": env["system_fingerprint"],
    }
    for ch in env["choices"]:
        for piece in _text_pieces(str(ch["text"])):
            frame = dict(base)
            frame["choices"] = [
                {
                    "index": ch["index"],
                    "text": piece,
                    "logprobs": None,
                    "finish_reason": None,
                }
            ]
            yield frame
        last = dict(base)
        last["choices"] = [
            {
                "index": ch["index"],
                "text": "",
                "logprobs": None,
                "finish_reason": ch["finish_reason"],
            }
        ]
        yield last
    if include_usage:
        usage_frame = dict(base)
        usage_frame["choices"] = []
        usage_frame["usage"] = env["usage"]
        yield usage_frame


def _text_pieces(text: str) -> Iterator[str]:
    """~64-char delta pieces on whitespace boundaries — the shared
    chunking for ``chat.completion.chunk`` deltas and ``output_text``
    response events."""
    pos = 0
    while pos < len(text):
        end = min(pos + 64, len(text))
        if end < len(text):
            sp = text.rfind(" ", pos, end)
            if sp > pos:
                end = sp + 1
        yield text[pos:end]
        pos = end


def openai_chunks(
    *,
    text: str | list[str],
    backend: str,
    model: str | None = None,
    cid: str,
    include_usage: bool = False,
    usage: dict[str, int] | None = None,
    tool_calls: Sequence[Sequence[dict[str, Any]] | None] | None = None,
    finish_reasons: Sequence[str] | None = None,
    created: int | None = None,
    logprobs: Sequence[dict[str, Any] | None] | None = None,
) -> Iterator[dict[str, Any]]:
    """`chat.completion.chunk` payloads over gated text.

    The content is already past the honesty gate (it is the gated
    response), so delta chunking is a presentation choice — ~64-char
    pieces on whitespace boundaries so streaming clients render
    incrementally. ``include_usage`` adds the OpenAI usage chunk
    (``choices=[]``, usage set) before the wire's ``[DONE]`` marker —
    which is emitted by the serializer, not this generator.

    ``text`` accepts the ``n>1`` choice list: each index emits its own
    role delta, content deltas, and ``finish_reason`` frame — grouped
    per index (spec-legal; ``choices[].index`` disambiguates).
    ``tool_calls`` per choice ride one ``delta.tool_calls`` frame
    carrying the complete call list (spec-legal single-shot deltas);
    ``logprobs`` per choice ride one aggregated ``delta.logprobs``
    frame after the content pieces — the provider's token boundaries
    don't align with the harness's whitespace re-chunking, so the full
    token array ships in one frame rather than faking alignment;
    ``finish_reasons`` override the ``stop`` default.
    """
    texts = [text] if isinstance(text, str) else list(text)
    calls = list(tool_calls or [])
    reasons = list(finish_reasons or [])
    lps = list(logprobs or [])
    base: dict[str, Any] = {
        "id": f"chatcmpl-{cid}",
        "object": "chat.completion.chunk",
        "created": int(time.time()) if created is None else created,
        "model": model or backend,
        "system_fingerprint": backend,
    }
    for i, choice_text in enumerate(texts):
        tc = calls[i] if i < len(calls) else None
        finish = reasons[i] if i < len(reasons) else "stop"
        first = dict(base)
        first["choices"] = [{"index": i, "delta": {"role": "assistant"}, "finish_reason": None}]
        yield first

        if tc:
            tc_frame = dict(base)
            tc_frame["choices"] = [
                {"index": i, "delta": {"tool_calls": list(tc)}, "finish_reason": None}
            ]
            yield tc_frame

        for piece in _text_pieces(choice_text):
            frame = dict(base)
            frame["choices"] = [{"index": i, "delta": {"content": piece}, "finish_reason": None}]
            yield frame

        lp = lps[i] if i < len(lps) else None
        if lp:
            lp_frame = dict(base)
            lp_frame["choices"] = [{"index": i, "delta": {"logprobs": lp}, "finish_reason": None}]
            yield lp_frame

        last = dict(base)
        last["choices"] = [{"index": i, "delta": {}, "finish_reason": finish}]
        yield last

    if include_usage:
        usage_frame = dict(base)
        usage_frame["choices"] = []
        usage_frame["usage"] = openai_usage(usage)
        yield usage_frame


# --- Responses API surface ----------------------------------------------------
# POST /v1/responses — OpenAI's canonical surface. Same gated pipeline, same
# translation layer, same fail-closed rule: a field the pipeline can't honor
# is a 400, never a silent drop. ``store`` is honored: it governs the
# retrieval index behind ``GET|DELETE /v1/responses/{id}`` (the audit ledger
# records every call regardless — retrieval is a convenience surface, not
# the evidence).
RESPONSES_UNSUPPORTED = (
    "truncation",
    # chat-completions fields that don't exist on this surface — refuse
    # rather than drop so a caller's intent never evaporates
    "n",
    "stop",
    "logit_bias",
    "presence_penalty",
    "frequency_penalty",
    "logprobs",
    "messages",
    "stream_options",
    "response_format",
    "max_tokens",
    "seed",
    "echo",
    "suffix",
    "best_of",
    "modalities",
    "audio",
    "prediction",
    "web_search_options",
    "functions",
    "function_call",
)

# Item types inside ``input[]`` that a text-only gated pipeline cannot honor.
# ``file_search_call`` is NOT refused: the server-side retrieval surface
# emits them as output items, and a chained/conv turn may re-feed one —
# it flattens into a context message carrying its prior results.
RESPONSE_ITEM_TYPES_REFUSED = frozenset(
    {
        "item_reference",
        "reasoning",
        "web_search_call",
        "computer_call",
        "computer_call_output",
        "code_interpreter_call",
        "image_generation_call",
        "local_shell_call",
        "mcp_call",
        "mcp_list_tools",
        "mcp_approval_request",
        "mcp_approval_response",
    }
)

# Content-part types inside a message item the pipeline accepts.
RESPONSE_PART_TYPES = frozenset({"input_text", "output_text"})

RESPONSE_ROLES = frozenset({"user", "assistant", "system", "developer"})

# ``status`` values a response stops moving at — cancel refuses these and a
# background worker never overwrites them.
OPENAI_RESPONSE_TERMINAL = frozenset({"completed", "failed", "cancelled", "incomplete"})


class OpenAIResponseTool(OpenAIToolFunction):
    """One ``tools[]`` entry on the Responses surface — the flattened
    function spec (``name``/``parameters`` sit beside ``type`` rather
    than under a ``function`` key). Same bounds as the chat spec."""

    type: Literal["function"] = "function"


class OpenAIFileSearchTool(_Model):
    """``tools[]`` entry — the server-side ``file_search`` tool against
    ``/v1/vector_stores``. ``vector_store_ids`` bounds the corpus (≤8
    stores per OpenAI's own cap); ``max_num_results`` bounds hits (≤50);
    ``filters`` evaluates the file-attributes comparison grammar;
    ``ranking_options`` accepts ``score_threshold`` (cosine, 0..1) —
    ``ranker`` is echoed but the harness ranker is lexical, not
    embedding-based (documented in FX1_HARNESS_API)."""

    model_config = ConfigDict(extra="allow")

    type: Literal["file_search"]
    vector_store_ids: list[str] = Field(min_length=1, max_length=8)
    max_num_results: int | None = Field(default=None, ge=1, le=50)
    filters: dict[str, Any] | None = None
    ranking_options: dict[str, Any] | None = None


class OpenAIResponseRequest(_Model):
    """POST /v1/responses body — the Responses surface over the same
    gated pipeline. ``input`` is one string or a list of message items;
    ``instructions`` prepends a system message. ``reasoning.effort`` maps
    to the decode hint; ``text.format`` maps to the post-validated
    ``response_format`` channel; ``user``/``safety_identifier`` stamp the
    audit record."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(default="fx1", min_length=1, max_length=512)
    input: str | list[dict[str, Any]]
    instructions: str | None = Field(default=None, max_length=32768)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_output_tokens: int | None = Field(default=None, gt=0, le=262144)
    # ``max_tool_calls`` bounds the function calls one response may carry —
    # a turn whose model emits more than the cap truncates at the bound and
    # lands ``status: 'incomplete'`` with ``incomplete_details.reason``
    # ``'max_tool_calls'`` (OpenAI's own semantics); 0 refuses calls outright.
    max_tool_calls: int | None = Field(default=None, ge=0)
    prompt_cache_key: str | None = Field(default=None, max_length=128)
    prompt_cache_retention: Literal["in-memory", "24h"] | None = None
    stream: bool = False
    store: bool | None = None
    metadata: dict[str, str] | None = None
    service_tier: Literal["auto", "default", "flex", "priority", "scale"] | None = None
    user: str | None = Field(default=None, max_length=512)
    safety_identifier: str | None = Field(default=None, max_length=512)
    reasoning: dict[str, Any] | None = None
    text: dict[str, Any] | None = None
    tools: list[OpenAIResponseTool | OpenAIFileSearchTool] | None = None
    tool_choice: Literal["none", "auto", "required"] | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    include: list[str] | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    previous_response_id: str | None = Field(default=None, max_length=512)
    # ``conversation`` is the named-container twin of
    # ``previous_response_id`` — a conv id (or ``{"id": "conv_*"}``
    # object) the turn joins; the conv's accumulated items are the
    # context. The two chain surfaces are mutually exclusive per OpenAI's
    # contract.
    conversation: str | dict[str, Any] | None = None
    background: bool = Field(default=False)
    fx1: OpenAIFx1 | None = None

    @model_validator(mode="after")
    def _response_valid(self) -> OpenAIResponseRequest:
        if isinstance(self.input, str) and not self.input.strip():
            raise ValueError("input must not be empty")
        if isinstance(self.input, list) and not self.input:
            raise ValueError("input must not be empty")
        if self.metadata is not None:
            if len(self.metadata) > 16:
                raise ValueError("metadata accepts at most 16 entries")
            for k, v in self.metadata.items():
                if len(k) > 64 or len(v) > 512:
                    raise ValueError("metadata keys are ≤64 chars, values ≤512")
        if self.reasoning is not None:
            extra = set(self.reasoning) - {"effort"}
            if extra:
                raise ValueError(
                    f"unsupported reasoning keys: {sorted(extra)} "
                    "(only 'effort' maps onto the gated decode hints)"
                )
            effort = self.reasoning.get("effort")
            if effort is not None and effort not in (
                "none",
                "minimal",
                "low",
                "medium",
                "high",
            ):
                raise ValueError(
                    f"reasoning.effort must be none|minimal|low|medium|high, got {effort!r}"
                )
        if self.text is not None:
            extra_t = set(self.text) - {"format", "verbosity"}
            if extra_t:
                raise ValueError(f"unsupported text keys: {sorted(extra_t)}")
            fmt = self.text.get("format")
            if fmt is not None:
                if not isinstance(fmt, dict):
                    raise ValueError("text.format must be an object")
                ftype = fmt.get("type", "text")
                if ftype not in OPENAI_RESPONSE_FORMATS:
                    raise ValueError(
                        "text.format.type must be one of "
                        f"{sorted(OPENAI_RESPONSE_FORMATS)}; got {ftype!r}"
                    )
                if ftype == "json_schema":
                    schema = fmt.get("schema")
                    if not isinstance(schema, dict):
                        raise ValueError(
                            "text.format json_schema needs "
                            "{type: 'json_schema', name?, schema: {...}}"
                        )
                    try:
                        jsonschema.validators.validator_for(schema).check_schema(schema)
                    except jsonschema.SchemaError as exc:
                        raise ValueError(
                            f"text.format json_schema is not a valid schema: {exc.message}"
                        ) from exc
            vb = self.text.get("verbosity")
            if vb is not None and vb not in ("low", "medium", "high"):
                raise ValueError(f"text.verbosity must be low|medium|high, got {vb!r}")
        if self.tools is not None and len(self.tools) > 128:
            raise ValueError("tools accepts at most 128 entries")
        if isinstance(self.tool_choice, dict):
            tc_type = self.tool_choice.get("type")
            if tc_type == "function":
                if not isinstance(self.tool_choice.get("name"), str):
                    raise ValueError("tool_choice {type: 'function'} needs a string 'name'")
            elif tc_type == "file_search":
                # forcing the server-side retrieval tool is meaningful —
                # the tool always runs when advertised; a bare
                # {type: 'file_search'} choice just asserts it exists
                if not any(t.type == "file_search" for t in self.tools or []):
                    raise ValueError(
                        "tool_choice {type: 'file_search'} requires a file_search tool"
                    )
            else:
                raise ValueError(
                    "tool_choice must be 'none'|'auto'|'required', "
                    "{type: 'function', name: 'fn_name'}, or "
                    "{type: 'file_search'}"
                )
        if not self.tools and (
            self.tool_choice is not None or self.parallel_tool_calls is not None
        ):
            raise ValueError("tool_choice/parallel_tool_calls require a non-empty tools list")
        for t in self.tools or []:
            if t.type == "file_search":
                ro = t.ranking_options
                if ro is not None:
                    extra_ro = set(ro) - {"ranker", "score_threshold"}
                    if extra_ro:
                        raise ValueError(
                            f"ranking_options keys must be ranker|score_threshold; "
                            f"got {sorted(extra_ro)}"
                        )
                    st = ro.get("score_threshold")
                    if st is not None and (
                        not isinstance(st, (int, float))
                        or isinstance(st, bool)
                        or not 0.0 <= st <= 1.0
                    ):
                        raise ValueError("score_threshold must be a number in [0, 1]")
        if self.include is not None:
            bad_inc = sorted(
                set(self.include) - {"message.output_text.logprobs", "file_search_call.results"}
            )
            if bad_inc:
                raise ValueError(
                    "include accepts only 'message.output_text.logprobs' and "
                    f"'file_search_call.results' on this surface; got {bad_inc}"
                )
        if self.top_logprobs is not None and (
            not self.include or "message.output_text.logprobs" not in self.include
        ):
            raise ValueError("top_logprobs requires include: ['message.output_text.logprobs']")
        if isinstance(self.conversation, dict):
            cid = self.conversation.get("id")
            if not isinstance(cid, str) or not cid.strip():
                raise ValueError("conversation must be an id string or {id: 'conv_*'}")
        if self.conversation is not None and self.previous_response_id is not None:
            raise ValueError(
                "conversation and previous_response_id are mutually exclusive — "
                "a turn anchors to one context surface"
            )
        if isinstance(self.conversation, str) and not self.conversation.strip():
            raise ValueError("conversation must be a non-empty id")
        present = [f for f in RESPONSES_UNSUPPORTED if getattr(self, f, None) is not None]
        extra_bad = sorted(f for f in RESPONSES_UNSUPPORTED if f in (self.__pydantic_extra__ or {}))
        bad = sorted(set(present) | set(extra_bad))
        if bad:
            raise ValueError(f"unsupported for the gated pipeline: {', '.join(bad)}")
        return self


def response_input_to_messages(
    input_: str | list[dict[str, Any]], instructions: str | None
) -> list[dict[str, Any]]:
    """Flatten a Responses ``input`` + ``instructions`` into harness
    message dicts. ``developer`` items map to ``system``; consecutive
    ``function_call`` items fold into one assistant turn's
    ``tool_calls``; a ``function_call_output`` item maps to a
    ``role: "tool"`` message keyed by its ``call_id``. Every other
    non-message item type and non-text content part fails closed."""
    msgs: list[dict[str, Any]] = []
    if instructions:
        msgs.append({"role": "system", "content": instructions})
    if isinstance(input_, str):
        msgs.append({"role": "user", "content": input_})
        return msgs
    for i, item in enumerate(input_):
        if not isinstance(item, dict):
            raise OpenAICompatError(f"input[{i}]: items must be objects")
        itype = item.get("type")
        if itype == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            args = item.get("arguments")
            if not isinstance(call_id, str) or not call_id:
                raise OpenAICompatError(
                    f"input[{i}]: function_call needs a non-empty string call_id"
                )
            if not isinstance(name, str) or not name:
                raise OpenAICompatError(f"input[{i}]: function_call needs a non-empty string name")
            if not isinstance(args, str):
                raise OpenAICompatError(f"input[{i}]: function_call arguments must be a string")
            call = {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
            prior = msgs[-1] if msgs else {}
            if prior.get("role") == "assistant" and prior.get("tool_calls"):
                prior["tool_calls"].append(call)
            else:
                msgs.append({"role": "assistant", "tool_calls": [call]})
            continue
        if itype == "function_call_output":
            call_id = item.get("call_id")
            output = item.get("output")
            if not isinstance(call_id, str) or not call_id:
                raise OpenAICompatError(
                    f"input[{i}]: function_call_output needs a non-empty string call_id"
                )
            if isinstance(output, list):
                outs: list[str] = []
                for j, part in enumerate(output):
                    if not isinstance(part, dict) or part.get("type") != "output_text":
                        raise OpenAICompatError(
                            f"input[{i}].output[{j}]: only output_text parts are supported"
                        )
                    outs.append(str(part.get("text", "")))
                output = "".join(outs)
            if not isinstance(output, str):
                raise OpenAICompatError(f"input[{i}]: function_call_output needs a string output")
            msgs.append({"role": "tool", "content": output, "tool_call_id": call_id})
            continue
        if itype == "file_search_call":
            # a prior turn's retrieval item re-fed as input flattens to a
            # system context message carrying its recorded results — the
            # same text the model originally saw
            results = item.get("results")
            if isinstance(results, list) and results:
                texts: list[str] = []
                for r in results:
                    if isinstance(r, dict) and isinstance(r.get("text"), str):
                        texts.append(r["text"][:4096])
                if texts:
                    msgs.append(
                        {
                            "role": "system",
                            "content": "[prior file_search results]\n" + "\n\n".join(texts),
                        }
                    )
            continue
        if itype in RESPONSE_ITEM_TYPES_REFUSED:
            raise OpenAICompatError(f"input[{i}]: {itype!r} items are not supported")
        role = item.get("role")
        if itype not in (None, "message") or role is None:
            raise OpenAICompatError(f"input[{i}]: only message items with a role are supported")
        if role not in RESPONSE_ROLES:
            raise OpenAICompatError(f"input[{i}]: unknown role {role!r}")
        content = item.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for j, part in enumerate(content):
                if not isinstance(part, dict):
                    raise OpenAICompatError(f"input[{i}].content[{j}]: parts must be objects")
                ptype = part.get("type")
                if ptype not in RESPONSE_PART_TYPES:
                    raise OpenAICompatError(
                        f"input[{i}].content[{j}]: only text parts are supported, got {ptype!r}"
                    )
                if ptype == "output_text" and role != "assistant":
                    raise OpenAICompatError(
                        f"input[{i}].content[{j}]: output_text belongs to assistant items"
                    )
                parts.append(str(part.get("text", "")))
            text = "".join(parts)
        else:
            raise OpenAICompatError(f"input[{i}]: content is required")
        msgs.append({"role": "system" if role == "developer" else role, "content": text})
    if not any(m["role"] != "system" for m in msgs):
        raise OpenAICompatError("input carried no user/assistant turn")
    return msgs


def _stored_item_id(prefix: str, envelope_id: str, i: int) -> str:
    """Deterministic item id for the stored-request subresources —
    ``<prefix>_<sha256(envelope_id:i)[:24]>`` so ``after``/``before``
    cursors stay stable across retrieval calls without extra state."""
    digest = hashlib.sha256(f"{envelope_id}:{i}".encode()).hexdigest()[:24]
    return f"{prefix}_{digest}"


def chat_messages_for_store(
    messages: Sequence[OpenAIChatMessage], *, envelope_id: str
) -> list[dict[str, Any]]:
    """Request messages in retrieval shape for
    ``GET /v1/chat/completions/{id}/messages`` — verbatim as submitted
    (``extra="allow"`` fields survive), each carrying the digest id."""
    out: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        item = msg.model_dump(mode="json", exclude_none=True)
        item.setdefault("id", _stored_item_id("msg", envelope_id, i))
        out.append(item)
    return out


def response_input_item_dicts(input_: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``input`` normalized to item dicts, no ids — a plain string wraps
    as one user message item with an ``input_text`` part."""
    if isinstance(input_, str):
        return [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": input_}],
            }
        ]
    return [dict(it) for it in input_]


def response_input_items_for_store(
    input_: str | list[dict[str, Any]], *, rid: str, start_at: int = 0
) -> list[dict[str, Any]]:
    """``input`` in retrieval shape for
    ``GET /v1/responses/{id}/input_items`` — the items as submitted
    (a plain string wraps as one user message item with ``input_text``),
    each carrying a caller-supplied or digest ``msg_`` id. ``start_at``
    offsets deterministic positions within one request. An append must
    use a fresh ``rid`` namespace: list length alone cannot prevent id
    reuse after earlier items have been deleted."""
    out: list[dict[str, Any]] = []
    for i, item in enumerate(response_input_item_dicts(input_)):
        item.setdefault("id", _stored_item_id("msg", rid, start_at + i))
        out.append(item)
    return out


def conversation_id_of(
    conversation: str | dict[str, Any] | None,
) -> str | None:
    """``conversation`` request field → the bare conv id (a string passes
    through; ``{id: 'conv_*'}`` unwraps; ``None`` stays ``None``)."""
    if conversation is None:
        return None
    if isinstance(conversation, str):
        return conversation
    cid = conversation.get("id")
    return cid if isinstance(cid, str) else None


def chained_response_input(
    prev_env: dict[str, Any],
    prev_items: Sequence[dict[str, Any]],
    new_input: str | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """``previous_response_id`` chain semantics: the new response's
    effective item list is the previous response's stored input items +
    its ``output`` items + this request's ``input``. Item ``id`` fields
    are dropped here — the new response's stored list re-mints them
    deterministically off (new rid, index) inside
    :func:`response_input_items_for_store`."""
    out: list[dict[str, Any]] = []
    for it in [*prev_items, *list(prev_env.get("output") or [])]:
        if isinstance(it, dict):
            out.append({k: v for k, v in it.items() if k != "id"})
    out.extend(response_input_item_dicts(new_input))
    return out


def paged_item_list(
    items: Sequence[dict[str, Any]],
    *,
    limit: int,
    after: str | None = None,
    before: str | None = None,
    order: str = "asc",
) -> dict[str, Any]:
    """The shared ``{object: list, data, first_id, last_id, has_more}``
    page shape the stored-request subresources return — ``after`` /
    ``before`` are id cursors into the ordered list; an unknown cursor
    fails closed ``400 invalid_cursor`` rather than silently restarting."""
    if not 1 <= limit <= 100:
        raise OpenAICompatError(
            "limit must be 1..100",
            status=400,
            code="invalid_limit",
        )
    if order not in ("asc", "desc"):
        raise OpenAICompatError(
            f"order must be 'asc' or 'desc', got {order!r}",
            status=400,
            code="invalid_cursor",
        )
    ordered = list(items)
    if order == "desc":
        ordered.reverse()
    for cursor, keep_after in ((after, True), (before, False)):
        if cursor is None:
            continue
        idx = next((k for k, it in enumerate(ordered) if it.get("id") == cursor), None)
        if idx is None:
            raise OpenAICompatError(
                f"cursor {cursor!r} is not an item id in this stored request",
                status=400,
                code="invalid_cursor",
            )
        ordered = ordered[idx + 1 :] if keep_after else ordered[:idx]
    # ``before`` means the page *preceding* the cursor — the tail of the
    # remaining window (the same convention the anthropic before_id pagers
    # pin), not its head: a backward walk chains first_id through the list.
    page = ordered[-limit:] if before is not None else ordered[:limit]
    return {
        "object": "list",
        "data": page,
        "first_id": page[0].get("id") if page else None,
        "last_id": page[-1].get("id") if page else None,
        "has_more": len(ordered) > limit,
    }


def response_text_format(body: OpenAIResponseRequest) -> dict[str, Any] | None:
    """``text.format`` → the chat-shape ``response_format`` dict
    :func:`validate_response_format` consumes (or ``None``)."""
    if not body.text:
        return None
    fmt = body.text.get("format")
    if not isinstance(fmt, dict):
        return None
    ftype = fmt.get("type", "text")
    if ftype == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {"name": fmt.get("name"), "schema": fmt["schema"]},
        }
    return {"type": ftype}


def response_to_kwargs(
    body: OpenAIResponseRequest,
    headers: Mapping[str, str] | None = None,
    *,
    ft_resolver: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Translate a Responses request into ``complete`` kwargs — the same
    backend-resolution order and the same extension/headers as the chat
    surface. ``max_output_tokens`` lands on ``max_tokens``;
    ``reasoning.effort`` on ``reasoning_effort``; the audit stamp takes
    ``user`` or ``safety_identifier``."""
    hdrs = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    ext = body.fx1
    backend, fallbacks, checkpoint_dir, byok, served_model = _resolve_openai_link(
        body.model, ext, hdrs, ft_resolver=ft_resolver, require_known_model=True
    )
    effort = (body.reasoning or {}).get("effort")
    return {
        "backend": backend,
        "messages": response_input_to_messages(body.input, body.instructions),
        "_served_model": served_model,
        "checkpoint_dir": checkpoint_dir,
        "receipt_hashes": _resolve_receipt_hashes(
            ext.receipt_hashes if ext is not None else None, hdrs
        ),
        "timeout_s": _resolve_timeout(ext.timeout_s if ext is not None else None, hdrs),
        "fallbacks": fallbacks,
        "byok": byok.model_dump() if byok is not None else None,
        "temperature": body.temperature,
        "top_p": body.top_p,
        "max_tokens": body.max_output_tokens,
        "user": body.user or body.safety_identifier,
        "metadata": body.metadata,
        "service_tier": body.service_tier,
        "reasoning_effort": effort,
        "verbosity": body.text.get("verbosity") if body.text else None,
        "prompt_cache_key": body.prompt_cache_key,
        "prompt_cache_retention": body.prompt_cache_retention,
        # the flattened Responses spec nests under ``function`` for the
        # shared tool channel; ``file_search`` entries are server-side —
        # they never reach the backend's tool list (retrieval ran in the
        # harness and lands as context). A dict tool_choice folds the
        # same way; {type: 'file_search'} carries no function name.
        "tools": (
            [
                {
                    "type": "function",
                    "function": t.model_dump(exclude_none=True, exclude={"type"}),
                }
                for t in body.tools
                if t.type == "function"
            ]
            if body.tools and any(t.type == "function" for t in body.tools)
            else None
        ),
        # tool_choice only carries on the model channel when function
        # tools are advertised — a choice over server-side tools
        # ({type: 'file_search'}) or a string choice with a file-only
        # list has nothing to bind (CompleteRequest refuses a bare
        # tool_choice).
        "tool_choice": (
            {"type": "function", "function": {"name": body.tool_choice["name"]}}
            if isinstance(body.tool_choice, dict) and body.tool_choice.get("type") == "function"
            else (body.tool_choice if not isinstance(body.tool_choice, dict) else None)
            if body.tools and any(t.type == "function" for t in body.tools)
            else None
        ),
        "parallel_tool_calls": (
            body.parallel_tool_calls
            if body.tools and any(t.type == "function" for t in body.tools)
            else None
        ),
        "logprobs": (True if _wants_response_logprobs(body) else None),
        "top_logprobs": body.top_logprobs,
    }


def _wants_response_logprobs(body: OpenAIResponseRequest) -> bool:
    """True when ``include`` requests the logprobs channel — the request
    param that maps onto the shared channel's ``logprobs: true``."""
    return bool(body.include and "message.output_text.logprobs" in body.include)


def _response_echoes(body: OpenAIResponseRequest) -> dict[str, Any]:
    """The request fields a faithful ``response`` object echoes back."""
    return {
        "temperature": body.temperature,
        "top_p": body.top_p,
        "max_output_tokens": body.max_output_tokens,
        "metadata": body.metadata,
        "instructions": body.instructions,
        "service_tier": body.service_tier,
        "reasoning": body.reasoning,
        "text": body.text,
        # fields we refuse on input are echoed as their honest constants;
        # `store` echoes the actual knob — the retrieval index honors it
        "store": body.store is not False,
        "tools": ([t.model_dump(exclude_none=True) for t in body.tools] if body.tools else []),
        # OpenAI echoes "auto" once tools are advertised without a choice
        "tool_choice": (
            body.tool_choice if body.tool_choice is not None else ("auto" if body.tools else "none")
        ),
        "parallel_tool_calls": bool(body.parallel_tool_calls),
        "include": body.include or [],
        "top_logprobs": body.top_logprobs,
        "max_tool_calls": body.max_tool_calls,
        "prompt_cache_key": body.prompt_cache_key,
        "prompt_cache_retention": body.prompt_cache_retention,
        "truncation": "disabled",
        # a ``background:true`` + ``stream:true`` request runs the sync
        # stream path (the route only queues when the caller is not
        # streaming) — the echo reports what actually happened, so the
        # ``?stream=true`` replay grammar emits the lifecycle that ran
        "background": body.background and not body.stream,
        "previous_response_id": body.previous_response_id,
        # OpenAI echoes ``conversation: {id}`` on the response when set
        "conversation": (
            {"id": conversation_id_of(body.conversation)} if body.conversation is not None else None
        ),
    }


def response_query_text(input_: str | list[dict[str, Any]]) -> str:
    """The retrieval query for ``file_search`` — the last user-role
    message's text (a bare string input is itself the query). Falls back
    to the last message item's text when no user item exists; "" when
    the input carries no message at all."""
    if isinstance(input_, str):
        return input_
    user_text: str | None = None
    last_text: str | None = None
    for item in input_:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        role = item.get("role")
        if itype not in (None, "message") or not isinstance(role, str):
            continue
        content = item.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(
                str(p.get("text", ""))
                for p in content
                if isinstance(p, dict) and p.get("type") in ("input_text", "output_text")
            )
        else:
            continue
        last_text = text
        if role == "user":
            user_text = text
    return user_text if user_text is not None else (last_text or "")


def file_search_call_item(
    *,
    queries: list[str],
    results: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """One ``file_search_call`` output item — the transcript record of
    the server-side retrieval run. ``results`` is ``None`` unless the
    request's ``include`` listed ``file_search_call.results`` (OpenAI's
    own field contract)."""
    return {
        "type": "file_search_call",
        "id": f"fs_{uuid.uuid4().hex}",
        "status": "completed",
        "queries": list(queries),
        "results": results,
    }


def openai_response_call_items(
    tool_calls: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Chat-shaped ``tool_calls`` → Responses ``function_call`` output
    items. Ids mint once here — pass the returned items to both
    :func:`openai_response_object` and :func:`openai_response_events` so
    the envelope and the stream frames carry identical item ids."""
    items: list[dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            # a provider-verbatim member that isn't an object can't shape
            # a function_call item — sieve it like the usage claim's
            # non-ints rather than crashing the envelope into a bare 500
            continue
        fn = call.get("function")
        fn = fn if isinstance(fn, dict) else {}
        args = fn.get("arguments")
        items.append(
            {
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex}",
                "call_id": call.get("id"),
                "name": fn.get("name"),
                "arguments": (
                    args if isinstance(args, str) else json.dumps(args or {}, sort_keys=True)
                ),
                "status": "completed",
            }
        )
    return items


def response_cap_call_items(
    body: OpenAIResponseRequest,
    tool_calls: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]] | None, dict[str, str] | None]:
    """Apply ``body.max_tool_calls`` — the cap on the calls one response
    may carry. Over the cap the emitted items truncate at the bound and
    the response lands ``status: 'incomplete'`` with
    ``{'reason': 'max_tool_calls'}`` — OpenAI's own truncation semantics,
    never a silent drop.

    Returns ``(items, incomplete_details)`` where ``items`` is ``None``
    when the model emitted no calls at all (a prose turn) and a list —
    possibly empty when the cap truncated everything — when it did, so a
    calls-only turn capped at zero ships no phantom empty message item."""
    calls = list(tool_calls)
    if not calls:
        return None, None
    items = openai_response_call_items(calls)
    if body.max_tool_calls is not None and len(items) > body.max_tool_calls:
        return items[: body.max_tool_calls], {"reason": "max_tool_calls"}
    return items, None


def openai_response_object(
    *,
    rid: str,
    item_id: str,
    content: str,
    body: OpenAIResponseRequest,
    model: str | None,
    usage: dict[str, int] | None,
    status: str = "completed",
    created: int | None = None,
    call_items: list[dict[str, Any]] | None = None,
    search_items: list[dict[str, Any]] | None = None,
    logprobs: list[dict[str, Any]] | None = None,
    error: dict[str, Any] | None = None,
    incomplete_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A gated result → the ``response`` object. ``output`` carries one
    ``message`` item with one ``output_text`` part — plus one
    ``function_call`` item per tool call when the backend answered with
    calls (a calls-only turn ships no message item, matching OpenAI).
    ``usage`` maps the provider's counts onto input/output/total
    (``None`` when the backend reports nothing — never fabricated).
    ``logprobs`` is the provider's per-token array — it lands verbatim
    on the ``output_text`` part's ``logprobs`` field (the key is emitted
    only when the provider reported scores). ``status`` is ``in_progress``
    inside the pre-completion stream events, and ``queued``/``failed``/
    ``cancelled`` on the background lifecycle (non-``completed`` ships an
    empty ``output``; ``error`` carries the failure record when set).
    ``incomplete`` is the exception — it carries the partial ``output``
    OpenAI ships on a truncated turn (``incomplete_details`` records the
    reason, e.g. ``{'reason': 'max_tool_calls'}``)."""
    resp_usage: dict[str, int] | None = None
    if isinstance(usage, dict):
        # a bool is not a token count — the wire claim gets the same
        # sieve the billable ledger's usage claim does
        it, ot, tt = (
            v if isinstance(v, int) and not isinstance(v, bool) else None
            for v in (
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                usage.get("total_tokens"),
            )
        )
        if it is not None or ot is not None or tt is not None:
            i_v = it if it is not None else 0
            o_v = ot if ot is not None else 0
            resp_usage = {
                "input_tokens": i_v,
                "output_tokens": o_v,
                "total_tokens": tt if tt is not None else i_v + o_v,
            }
    output: list[dict[str, Any]] = []
    if status in ("completed", "incomplete"):
        # ``file_search_call`` items precede the message — retrieval
        # runs before the model answers, so the transcript orders them
        # first (OpenAI's own ordering)
        output.extend(search_items or [])
        # ``call_items is None`` marks a prose turn; a calls turn —
        # including one the cap truncated to zero — is a non-None list
        if content or call_items is None:
            part: dict[str, Any] = {
                "type": "output_text",
                "text": content,
                "annotations": [],
            }
            if logprobs:
                part["logprobs"] = list(logprobs)
            output.append(
                {
                    "type": "message",
                    "id": item_id,
                    "status": status,
                    "role": "assistant",
                    "content": [part],
                }
            )
        output.extend(call_items or [])
    return {
        "id": rid,
        "object": "response",
        "created_at": int(time.time()) if created is None else created,
        "status": status,
        "model": model or body.model,
        "output": output,
        "usage": resp_usage,
        "error": error,
        "incomplete_details": incomplete_details,
        **_response_echoes(body),
    }


def openai_conversation_object(
    *,
    cid: str,
    metadata: dict[str, str] | None = None,
    created: int | None = None,
) -> dict[str, Any]:
    """A ``conversation`` object — the named container a response turn
    can join via ``conversation``. ``items`` never ride the object; they
    live in the store's subitems under ``"items"`` and page through
    ``GET /v1/conversations/{id}/items``."""
    return {
        "id": cid,
        "object": "conversation",
        "created_at": int(time.time()) if created is None else created,
        "metadata": metadata or {},
    }


class OpenAIConversationCreate(_Model):
    """``POST /v1/conversations`` body — ``items`` seeds the conv's item
    list (same item dicts a response's ``input`` accepts); ``metadata``
    follows the same bounds as every other stamped surface."""

    model_config = ConfigDict(extra="allow")

    items: list[dict[str, Any]] | None = None
    metadata: dict[str, str] | None = None

    @model_validator(mode="after")
    def _valid(self) -> OpenAIConversationCreate:
        if self.metadata is not None:
            if len(self.metadata) > 16:
                raise ValueError("metadata accepts at most 16 entries")
            for k, v in self.metadata.items():
                if len(k) > 64 or len(v) > 512:
                    raise ValueError("metadata keys are ≤64 chars, values ≤512")
        if self.items is not None:
            for it in self.items:
                if not isinstance(it, dict):
                    raise ValueError("items must be message-item dicts")
        return self


class OpenAIChatUpdate(_Model):
    """``POST /v1/chat/completions/{id}`` body — ``metadata`` replaces
    the stored completion's metadata wholesale (OpenAI's update
    semantics; the only mutable field on a stored completion)."""

    model_config = ConfigDict(extra="allow")

    metadata: dict[str, str] | None = None

    @model_validator(mode="after")
    def _valid(self) -> OpenAIChatUpdate:
        if self.metadata is not None:
            if len(self.metadata) > 16:
                raise ValueError("metadata accepts at most 16 entries")
            for k, v in self.metadata.items():
                if len(k) > 64 or len(v) > 512:
                    raise ValueError("metadata keys are ≤64 chars, values ≤512")
        return self


class OpenAIConversationUpdate(_Model):
    """``POST /v1/conversations/{id}`` body — ``metadata`` replaces the
    conv's metadata wholesale (OpenAI's update semantics)."""

    model_config = ConfigDict(extra="allow")

    metadata: dict[str, str] | None = None

    @model_validator(mode="after")
    def _valid(self) -> OpenAIConversationUpdate:
        if self.metadata is not None:
            if len(self.metadata) > 16:
                raise ValueError("metadata accepts at most 16 entries")
            for k, v in self.metadata.items():
                if len(k) > 64 or len(v) > 512:
                    raise ValueError("metadata keys are ≤64 chars, values ≤512")
        return self


class OpenAIConversationItemsAdd(_Model):
    """``POST /v1/conversations/{id}/items`` body — input items to
    append. ``item_ids`` (reference existing stored items) is refused:
    the harness's items are minted per turn, never aliased."""

    model_config = ConfigDict(extra="allow")

    items: list[dict[str, Any]] | None = None
    item_ids: list[str] | None = None

    @model_validator(mode="after")
    def _valid(self) -> OpenAIConversationItemsAdd:
        if self.item_ids is not None:
            raise ValueError(
                "item_ids (alias by reference) is not supported — pass full item dicts"
            )
        if not self.items:
            raise ValueError("items must be a non-empty list of item dicts")
        for it in self.items:
            if not isinstance(it, dict):
                raise ValueError("items must be message-item dicts")
        return self


class OpenAIVectorStoreCreate(_Model):
    """``POST /v1/vector_stores`` body — ``name``/``metadata`` are free
    labels; ``file_ids`` attaches existing ``file-*`` records at create
    time (a bogus id fails the attach honestly)."""

    model_config = ConfigDict(extra="allow")

    name: str | None = Field(default=None, max_length=512)
    file_ids: list[str] | None = Field(default=None, max_length=64)
    metadata: dict[str, str] | None = None
    expires_after: dict[str, Any] | None = None


class OpenAIVectorStoreUpdate(_Model):
    """``POST /v1/vector_stores/{id}`` body — ``name``/``metadata``
    replace wholesale when present; ``expires_after`` re-anchors the
    expiry window from ``last_active_at``."""

    model_config = ConfigDict(extra="allow")

    name: str | None = Field(default=None, max_length=512)
    metadata: dict[str, str] | None = None
    expires_after: dict[str, Any] | None = None


class OpenAIVectorStoreFileCreate(_Model):
    """``POST /v1/vector_stores/{id}/files`` body — attach a ``file-*``
    record. ``attributes`` are the keys ``filters`` evaluate against;
    ``chunking_strategy`` is ``{"type": "auto"}`` or ``{"type":
    "static", "static": {max_chunk_size_tokens, chunk_overlap_tokens}}``
    (the word-window index maps token bounds ~0.75×)."""

    model_config = ConfigDict(extra="allow")

    file_id: str = Field(min_length=1, max_length=128)
    attributes: dict[str, Any] | None = None
    chunking_strategy: dict[str, Any] | None = None


class OpenAIVectorStoreSearch(_Model):
    """``POST /v1/vector_stores/{id}/search`` body — query the store
    directly without spending a response turn. ``query`` accepts a
    string or a list of strings (joined with spaces). ``rewrite_query``
    is refused: the store never rewrites the caller's query —
    ``filters`` apply to file attributes (OpenAI's comparison grammar)
    and ``ranking_options.score_threshold`` bounds the cosine floor
    (``ranker`` accepts only ``"auto"`` — no other ranker exists)."""

    model_config = ConfigDict(extra="allow")

    query: str | list[str]
    max_num_results: int | None = Field(default=None, ge=1, le=50)
    filters: dict[str, Any] | None = None
    ranking_options: dict[str, Any] | None = None
    rewrite_query: bool | None = None

    @model_validator(mode="after")
    def _valid(self) -> OpenAIVectorStoreSearch:
        if isinstance(self.query, list) and (
            not self.query or any(not isinstance(q, str) for q in self.query)
        ):
            raise ValueError("query must be a string or a list of strings")
        if self.rewrite_query:
            raise ValueError("rewrite_query is not supported")
        ro = self.ranking_options or {}
        if not isinstance(ro, dict):
            raise ValueError("ranking_options must be an object")
        unknown = set(ro) - {"ranker", "score_threshold"}
        if unknown:
            raise ValueError(f"ranking_options keys unknown: {sorted(unknown)}")
        if ro.get("ranker", "auto") != "auto":
            raise ValueError("ranking_options.ranker accepts only 'auto'")
        st = ro.get("score_threshold")
        # A bool is not a threshold. Numeric range is enforced by
        # VectorStore.search as 400 invalid_filters — rejecting it here
        # turns that contract into a 422 before the core sees the call.
        if st is not None and (not isinstance(st, (int, float)) or isinstance(st, bool)):
            raise ValueError("score_threshold must be a number in [0, 1]")
        return self


class OpenAIVectorStoreFileBatchCreate(_Model):
    """``POST /v1/vector_stores/{id}/file_batches`` body — attach up to
    500 ``file-*`` records in one call (OpenAI's cap). Members attach
    synchronously; per-file failures count, never abort the batch.
    ``attributes``/``chunking_strategy`` apply to every member."""

    model_config = ConfigDict(extra="allow")

    file_ids: list[str] = Field(min_length=1, max_length=500)
    attributes: dict[str, Any] | None = None
    chunking_strategy: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _ids_valid(self) -> OpenAIVectorStoreFileBatchCreate:
        if any(not fid or len(fid) > 128 for fid in self.file_ids):
            raise ValueError("file_ids entries must be non-empty strings ≤128 chars")
        return self


def openai_response_events(
    *,
    text: str,
    rid: str,
    item_id: str,
    body: OpenAIResponseRequest,
    model: str | None,
    usage: dict[str, int] | None,
    created: int | None = None,
    call_items: list[dict[str, Any]] | None = None,
    search_items: list[dict[str, Any]] | None = None,
    logprobs: list[dict[str, Any]] | None = None,
    final_status: str = "completed",
    incomplete_details: dict[str, Any] | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """The Responses SSE event sequence over gated text — the core grammar
    a streaming client needs: ``response.created``/``in_progress``, the
    output-item lifecycle, ``output_text.delta`` frames (the shared
    ~64-char splitter), and the terminal frame carrying the full response
    object with usage. ``final_status`` selects that terminal event —
    ``response.completed`` normally, ``response.incomplete`` when the turn
    truncated (``incomplete_details`` rides the terminal object, e.g.
    ``{'reason': 'max_tool_calls'}``). ``logprobs`` lands on the terminal
    ``content_part.done`` / ``output_item.done`` payloads' ``output_text``
    part and inside ``response.completed`` — provider token boundaries
    don't align with the text deltas, so the array ships whole at
    completion rather than faking per-delta alignment. Each
    ``(event, payload)`` pair serializes as an SSE ``event:`` + ``data:``
    frame."""
    created_obj = openai_response_object(
        rid=rid,
        item_id=item_id,
        content="",
        body=body,
        model=model,
        usage=None,
        status="in_progress",
        created=created,
    )
    yield "response.created", {"type": "response.created", "response": created_obj}
    yield (
        "response.in_progress",
        {
            "type": "response.in_progress",
            "response": created_obj,
        },
    )
    yield from _response_output_item_events(
        text=text,
        item_id=item_id,
        call_items=call_items,
        search_items=search_items,
        logprobs=logprobs,
        final_status=final_status,
    )
    terminal = "response.incomplete" if final_status == "incomplete" else "response.completed"
    yield (
        terminal,
        {
            "type": terminal,
            "response": openai_response_object(
                rid=rid,
                item_id=item_id,
                content=text,
                body=body,
                model=model,
                usage=usage,
                status=final_status,
                created=created,
                call_items=call_items,
                search_items=search_items,
                logprobs=logprobs,
                incomplete_details=incomplete_details,
            ),
        },
    )


def _response_output_item_events(
    *,
    text: str,
    item_id: str,
    call_items: list[dict[str, Any]] | None,
    search_items: list[dict[str, Any]] | None,
    logprobs: list[dict[str, Any]] | None,
    final_status: str,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """The output-item lifecycle section of the Responses event grammar —
    shared verbatim between the create-time stream and the stored-response
    replay."""
    next_index = 0
    # ``file_search_call`` items lead the output — each emits its
    # output_item lifecycle plus the file_search_call-specific events
    # (in_progress → searching → completed); there is no arguments delta
    # channel for a server-side tool
    for item in search_items or []:
        idx = next_index
        next_index += 1
        fs_id = str(item["id"])
        yield (
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": idx,
                "item": {**item, "status": "in_progress"},
            },
        )
        for ev in (
            "response.file_search_call.in_progress",
            "response.file_search_call.searching",
            "response.file_search_call.completed",
        ):
            yield (
                ev,
                {"type": ev, "output_index": idx, "item_id": fs_id},
            )
        yield (
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": idx,
                "item": item,
            },
        )
    # ``call_items is None`` marks a prose turn — a calls turn truncated
    # to zero by ``max_tool_calls`` ships no phantom empty message item
    if text or call_items is None:
        msg_index = next_index
        next_index += 1
        yield (
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": msg_index,
                "item": {
                    "type": "message",
                    "id": item_id,
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            },
        )
        yield (
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "item_id": item_id,
                "output_index": msg_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        )
        for piece in _text_pieces(text):
            yield (
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": item_id,
                    "output_index": msg_index,
                    "content_index": 0,
                    "delta": piece,
                },
            )
        yield (
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": item_id,
                "output_index": msg_index,
                "content_index": 0,
                "text": text,
            },
        )
        done_part: dict[str, Any] = {
            "type": "output_text",
            "text": text,
            "annotations": [],
        }
        if logprobs:
            done_part["logprobs"] = list(logprobs)
        yield (
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": item_id,
                "output_index": msg_index,
                "content_index": 0,
                "part": done_part,
            },
        )
        yield (
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": msg_index,
                "item": {
                    "type": "message",
                    "id": item_id,
                    "status": "incomplete" if final_status == "incomplete" else "completed",
                    "role": "assistant",
                    "content": [done_part],
                },
            },
        )
    # one output_item lifecycle per function call — arguments stream as
    # function_call_arguments.delta chunks inside it
    for k, item in enumerate(call_items or []):
        idx = next_index + k
        iid = str(item["id"])
        yield (
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": idx,
                "item": {**item, "arguments": "", "status": "in_progress"},
            },
        )
        for piece in _text_pieces(str(item.get("arguments") or "")):
            yield (
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": iid,
                    "output_index": idx,
                    "delta": piece,
                },
            )
        yield (
            "response.function_call_arguments.done",
            {
                "type": "response.function_call_arguments.done",
                "item_id": iid,
                "output_index": idx,
                "arguments": item.get("arguments"),
            },
        )
        yield (
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": idx,
                "item": item,
            },
        )


def response_output_pieces(
    env: Mapping[str, Any],
) -> tuple[
    str, str, list[dict[str, Any]] | None, list[dict[str, Any]] | None, list[dict[str, Any]] | None
]:
    """The stream-grammar pieces extracted from a stored ``response``
    envelope: ``(text, item_id, call_items, search_items, logprobs)`` — the
    deterministic reconstruction the replay surface shares with the
    create-time emitter. ``call_items`` is ``None`` for a prose turn and
    ``[]`` for a calls turn truncated to zero by ``max_tool_calls`` (the
    same distinction ``openai_response_events`` makes), so the replayed
    grammar matches the create-time one byte-for-byte."""
    items = [it for it in env.get("output") or [] if isinstance(it, dict)]
    msg = next((it for it in items if it.get("type") == "message"), None)
    calls = [it for it in items if it.get("type") == "function_call"]
    search = [it for it in items if it.get("type") == "file_search_call"]
    parts = msg.get("content") if isinstance(msg, dict) else None
    lp = (
        parts[0].get("logprobs")
        if isinstance(parts, list) and parts and isinstance(parts[0], dict)
        else None
    )
    if calls:
        call_list: list[dict[str, Any]] | None = calls
    elif env.get("status") == "incomplete" and msg is None:
        # an ``incomplete`` response with no message item = a calls turn
        # truncated to zero calls by ``max_tool_calls`` — not a prose turn
        call_list = []
    else:
        call_list = None
    text = ""
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, list) and content and isinstance(content[0], dict):
            text = str(content[0].get("text") or "")
    item_id = str(msg["id"]) if isinstance(msg, dict) and msg.get("id") else ""
    return text, item_id, call_list, (search or None), (lp if isinstance(lp, list) else None)


def openai_response_replay_events(
    env: Mapping[str, Any],
) -> Iterator[tuple[str, dict[str, Any]]]:
    """The ``GET /v1/responses/{id}?stream=true`` replay grammar derived
    deterministically from the stored envelope — OpenAI emits the same
    event sequence the create-time stream produced, so a re-attaching
    client's stream parser rebuilds the same typed ``Response``.

    The envelope carries enough structure to replay without a persisted
    event log: the stored ``id``/``created_at``/output items pin the
    create-time minted values, and the envelope object itself IS the
    terminal-frame payload — so the replayed terminal event is
    byte-consistent with the non-stream retrieve body. ``response.created``
    carries the as-created state (``output: []``, ``usage: null``, status
    ``queued`` for a background response / ``in_progress`` otherwise);
    ``response.queued`` follows for ``background: true`` envelopes (the
    recorded lifecycle); ``response.in_progress`` emits only when the
    record genuinely left ``queued`` — every non-``cancelled`` terminal
    status implies it, and a ``cancelled`` record proves it through the
    ``_fx1_progressed`` marker the ``queued → in_progress`` transition
    stamps (a record cancelled while still queued emits no phantom
    ``in_progress``). Non-terminal envelopes emit only this prelude — the
    route's follow loop emits the rest once the record lands terminal
    (``response.completed`` / ``response.incomplete`` / ``response.failed``
    / ``response.cancelled``). ``_fx1_*`` internals never reach the wire.
    """
    clean = {k: v for k, v in env.items() if not k.startswith("_fx1_")}
    status = str(clean.get("status") or "completed")
    background = clean.get("background") is True
    created_obj = {
        **clean,
        "status": "queued" if background else "in_progress",
        "output": [],
        "usage": None,
        "error": None,
        "incomplete_details": None,
    }
    yield "response.created", {"type": "response.created", "response": created_obj}
    if background:
        yield "response.queued", {"type": "response.queued", "response": created_obj}
    if status != "queued" and (status != "cancelled" or env.get("_fx1_progressed") is True):
        yield (
            "response.in_progress",
            {
                "type": "response.in_progress",
                "response": {**created_obj, "status": "in_progress"},
            },
        )
    if status in ("completed", "incomplete"):
        text, item_id, call_items, search_items, logprobs = response_output_pieces(clean)
        yield from _response_output_item_events(
            text=text,
            item_id=item_id,
            call_items=call_items,
            search_items=search_items,
            logprobs=logprobs,
            final_status=status,
        )
    if status in OPENAI_RESPONSE_TERMINAL:
        terminal = f"response.{status}"
        yield terminal, {"type": terminal, "response": clean}


# ---- /v1/embeddings ---------------------------------------------------------
# The embedding surface: POST /v1/embeddings → provider ``data[]`` verbatim.
# Same fail-closed rule as the rest of /v1 — a link without the channel
# answers 501, a provider's own 4xx surfaces as its own error, and input
# shapes the route can't honor refuse 422 before any spend. Embeddings
# have no stream and no retrieval id — the envelope is the response.


class OpenAIEmbeddingRequest(_Model):
    """``POST /v1/embeddings`` body — the OpenAI surface, extra fields
    tolerated (SDKs send bookkeeping keys)."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(default="fx1", min_length=1, max_length=512)
    input: str | list[str] | list[int] | list[list[int]]
    encoding_format: Literal["float", "base64"] | None = None
    dimensions: int | None = Field(default=None, ge=1)
    user: str | None = Field(default=None, max_length=512)
    fx1: OpenAIFx1 | None = None

    @model_validator(mode="after")
    def _input_shape(self) -> OpenAIEmbeddingRequest:
        raw = self.input
        if isinstance(raw, str):
            if not raw.strip():
                raise ValueError("input must not be empty")
            return self
        if not isinstance(raw, list) or len(raw) == 0 or len(raw) > 2048:
            raise ValueError("input must be a non-empty list of at most 2048 items")
        strings = [v for v in raw if isinstance(v, str)]
        tokens = all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in raw)
        token_lists = all(
            isinstance(v, list)
            and len(v) > 0
            and all(isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in v)
            for v in raw
        )
        if len(strings) == len(raw):
            if any(not v.strip() for v in strings):
                raise ValueError("input strings must not be empty")
        elif not tokens and not token_lists:
            raise ValueError(
                "input must be a string, a list of strings, a token array, "
                "or a list of token arrays — not a mixture"
            )
        return self


class OpenAIEmbeddingItem(_Model):
    """One ``data[]`` entry — vector numbers or a base64 payload.
    Extra keys tolerate provider-specific fields (echoed verbatim)."""

    model_config = ConfigDict(extra="allow")

    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float] | str


class OpenAIEmbeddingResponse(_Model):
    """``POST /v1/embeddings`` answer — ``data`` in request order."""

    object: Literal["list"] = "list"
    data: list[OpenAIEmbeddingItem]
    model: str
    usage: dict[str, int] | None = None


def openai_embedding_envelope(
    *, data: Iterable[dict[str, Any]], model: str, usage: dict[str, int] | None
) -> dict[str, Any]:
    """The OpenAI ``list`` envelope — ``data`` verbatim, ``model``/``usage``
    echo what the provider answered (``None`` under provider silence)."""
    return {
        "object": "list",
        "data": list(data),
        "model": model,
        "usage": usage,
    }


def embeddings_to_kwargs(
    body: OpenAIEmbeddingRequest,
    headers: Mapping[str, str] | None = None,
    *,
    ft_resolver: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Translate an embeddings request into call kwargs — same link
    resolution as chat (``fx1.backend`` > header > model > hosted_k3);
    keys: ``backend``, ``fallbacks``, ``byok``, ``checkpoint_dir``,
    ``timeout_s`` plus the forwarded fields ``model``, ``input``,
    ``encoding_format``, ``dimensions``, ``user``."""
    hdrs = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    ext = body.fx1
    # ``model`` here is the upstream embedding deployment name — free-form
    # and passed verbatim, so the strict known-model gate stays off.
    backend, fallbacks, checkpoint_dir, byok, _served = _resolve_openai_link(
        body.model, ext, hdrs, ft_resolver=ft_resolver
    )
    return {
        "backend": backend,
        "fallbacks": fallbacks,
        "byok": byok.model_dump() if byok is not None else None,
        "checkpoint_dir": checkpoint_dir,
        "timeout_s": _resolve_timeout(ext.timeout_s if ext is not None else None, hdrs),
        "model": body.model,
        "input": body.input,
        "encoding_format": body.encoding_format,
        "dimensions": body.dimensions,
        "user": body.user,
    }


# ---- /v1/files + /v1/batches ------------------------------------------------
# The async-batch surface: ``POST /v1/files`` takes the request JSONL
# (multipart, ``purpose="batch"``), ``POST /v1/batches`` runs it through the
# gated pipeline as one tracked unit, and ``GET /v1/files/{id}/content``
# returns the output JSONL. Same fail-closed rule as the rest of /v1 —
# shapes the pipeline can't honor refuse at submit; per-line request errors
# land in the output file with their OpenAI error body, never silently.

OPENAI_BATCH_ENDPOINTS = frozenset({"/v1/chat/completions", "/v1/responses", "/v1/embeddings"})
"""Endpoints a batch may target — one per batch, declared up front."""

OPENAI_BATCH_LINE_MAX = 1024
"""Max request lines per batch file."""

OPENAI_FILE_BYTES_MAX = 8 << 20
"""Max upload size (8 MiB)."""

OPENAI_FILE_PURPOSE_ACCEPT = frozenset({"batch", "fine-tune"})
"""The upload purposes the harness serves — batch input JSONL and
fine-tuning corpora (consumed by ``POST /v1/fine_tuning/jobs``)."""


class OpenAIBatchRequest(_Model):
    """``POST /v1/batches`` body."""

    model_config = ConfigDict(extra="allow")
    input_file_id: str = Field(min_length=1, max_length=64)
    endpoint: Literal["/v1/chat/completions", "/v1/responses", "/v1/embeddings"]
    # only "24h" exists on the real surface; anything else refuses (422)
    completion_window: Literal["24h"] = "24h"
    metadata: dict[str, str] | None = None
    # fx1 extension: terminal-state webhook — the finished batch envelope
    # is POSTed to ``callback_url`` on completed/failed/expired/cancelled,
    # signed with ``callback_secret`` via the X-Fx1-Webhook-* headers
    # (never echoed on the record).
    callback_url: str | None = None
    callback_secret: str | None = None

    @field_validator("metadata")
    @classmethod
    def _meta_bounds(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        if v is not None:
            if len(v) > 16:
                raise ValueError("metadata must have <= 16 keys")
            # the values persist into the journaled batch record and every
            # GET projection — apply the same ≤64/≤512 bounds every
            # sibling metadata surface enforces.
            for k, val in v.items():
                if len(k) > 64 or len(val) > 512:
                    raise ValueError("metadata keys are ≤64 chars, values ≤512")
        return v

    @field_validator("callback_url")
    @classmethod
    def _callback_url_http(cls, v: str | None) -> str | None:
        return check_callback_url(v)

    @model_validator(mode="after")
    def _callback_secret_needs_url(self) -> OpenAIBatchRequest:
        if self.callback_secret is not None and not self.callback_url:
            raise ValueError("callback_secret requires callback_url")
        return self


class OpenAIUploadCreateRequest(_Model):
    """``POST /v1/uploads`` body — the upload intent record."""

    model_config = ConfigDict(extra="forbid")
    purpose: str = Field(min_length=1, max_length=32)
    filename: str = Field(min_length=1, max_length=256)
    bytes: int = Field(gt=0)
    mime_type: str = Field(min_length=1, max_length=128)


class OpenAIUploadCompleteRequest(_Model):
    """``POST /v1/uploads/{id}/complete`` body — the part order to
    concatenate, plus an optional content md5 the store checks before
    the file is minted."""

    model_config = ConfigDict(extra="forbid")
    part_ids: list[str] = Field(min_length=1, max_length=64)
    md5: str | None = Field(default=None, min_length=32, max_length=32)


def batch_line_shape(line: Any, *, endpoint: str, lineno: int) -> dict[str, Any]:
    """Validate one input-file line into ``{custom_id, body}``.

    Line-shape errors are submit-time failures (400) — a batch whose input
    can't be trusted never starts, rather than half-running a corrupt file.
    Per-line *request* errors (a valid line whose body fails the endpoint's
    own validation) run through the pipeline and land in the output file.
    """
    where = f"line {lineno}"
    if not isinstance(line, dict):
        raise OpenAICompatError(f"{where}: must be a JSON object")
    custom_id = line.get("custom_id")
    if not isinstance(custom_id, str) or not custom_id.strip() or len(custom_id) > 64:
        raise OpenAICompatError(f"{where}: custom_id must be a non-empty string <= 64 chars")
    if line.get("method") != "POST":
        raise OpenAICompatError(f"{where}: method must be 'POST', got {line.get('method')!r}")
    url = line.get("url")
    if url != endpoint:
        raise OpenAICompatError(
            f"{where}: url {url!r} does not match the batch endpoint {endpoint!r}"
        )
    body = line.get("body")
    if not isinstance(body, dict):
        raise OpenAICompatError(f"{where}: body must be a JSON object")
    return {"custom_id": custom_id, "body": body}


def batch_line_body(
    line: dict[str, Any], endpoint: str
) -> OpenAIChatRequest | OpenAIResponseRequest | OpenAIEmbeddingRequest:
    """Parse a line's ``body`` into the endpoint's request model — the same
    validation object the wire route uses, so a batch line can never carry
    a request the live route would refuse differently."""
    try:
        if endpoint == "/v1/responses":
            return OpenAIResponseRequest.model_validate(line["body"])
        if endpoint == "/v1/embeddings":
            return OpenAIEmbeddingRequest.model_validate(line["body"])
        return OpenAIChatRequest.model_validate(line["body"])
    except ValueError as exc:
        raise OpenAICompatError(f"invalid request body: {exc}") from exc


def file_object(rec: Mapping[str, Any]) -> dict[str, Any]:
    """The OpenAI ``file`` envelope for a stored record."""
    return {
        "id": rec["file_id"],
        "object": "file",
        "purpose": rec["purpose"],
        "filename": rec["filename"],
        "bytes": rec["size"],
        "created_at": rec["created_at"],
        "status": "processed",
    }


def batch_object(rec: Mapping[str, Any]) -> dict[str, Any]:
    """The OpenAI ``batch`` envelope for a stored record."""
    return {
        "id": rec["batch_id"],
        "object": "batch",
        "endpoint": rec["endpoint"],
        "errors": rec.get("errors"),
        "input_file_id": rec["input_file_id"],
        "completion_window": rec["completion_window"],
        "status": rec["status"],
        "output_file_id": rec.get("output_file_id"),
        "error_file_id": rec.get("error_file_id"),
        "created_at": rec["created_at"],
        "in_progress_at": rec.get("in_progress_at"),
        "expires_at": rec.get("expires_at"),
        "finalizing_at": rec.get("finalizing_at"),
        "completed_at": rec.get("completed_at"),
        "failed_at": rec.get("failed_at"),
        "expired_at": rec.get("expired_at"),
        "cancelling_at": rec.get("cancelling_at"),
        "cancelled_at": rec.get("cancelled_at"),
        "request_counts": dict(rec["request_counts"]),
        "metadata": rec.get("metadata"),
        # fx1 extension — terminal webhook bookkeeping (the same fields
        # the /harness/* jobs surface); absent keys read as null.
        "callback_url": rec.get("callback_url"),
        "callback_status": rec.get("callback_status"),
        "callback_attempts": rec.get("callback_attempts", 0),
        "callback_error": rec.get("callback_error"),
    }


def batch_output_line(
    *,
    custom_id: str,
    status_code: int,
    body: dict[str, Any],
    rid: str,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One output-file line — the OpenAI batch result shape."""
    return {
        "id": f"batch_req_{rid}",
        "custom_id": custom_id,
        "response": {
            "status_code": status_code,
            "request_id": f"req_{rid}",
            "body": body,
        },
        "error": error,
    }


# ---- retrieval index (GET /v1/chat/completions/{id}, GET|DELETE /v1/responses/{id}) ----


class OpenAIEnvelopeStore:
    """Bounded ``id → envelope`` index backing the ``/v1`` retrieval
    routes. ``store=false`` keeps a call out of this index — the
    completion log still records it; retrieval is a convenience surface,
    not the evidence. Oldest entries evict at ``cap`` (insertion order —
    a re-put refreshes the id's position).

    With a ``JobJournal`` bound, every mutation is journaled inside the
    store lock before the in-memory write lands, and boot replays the
    chain then compacts to the live records. Invalid journal records
    refuse startup without rewriting the file: replaying an older
    verified prefix could otherwise resurrect a deleted conversation.
    Evictions re-derive from insertion order; explicit deletions are
    journaled before their envelopes disappear from memory."""

    def __init__(self, cap: int = 256, journal: JobJournal | None = None) -> None:
        if cap < 1:
            raise ValueError(f"store cap must be >= 1, got {cap}")
        self._cap = cap
        self._lock = threading.Lock()
        self._items: dict[str, dict[str, Any]] = {}
        # Request items backing the stored-request subresources
        # (``/messages``, ``/input_items``). Kept OUT of the envelope dict:
        # the envelope doubles as the POST response body, batch output
        # line, and SSE replay source — a stash key would serialize onto
        # the wire. Items share their envelope's lifetime.
        self._subitems: dict[str, dict[str, list[dict[str, Any]]]] = {}
        self._journal = journal
        self.recover_warnings: list[str] = []
        if journal is not None:
            res = journal.replay()
            self.recover_warnings = list(res.warnings)
            if res.truncated_at is not None or res.dropped:
                raise RuntimeError("envelope journal is damaged; recovery requires operator repair")
            try:
                for payload in res.payloads:
                    self._apply(payload)
            except ValueError as exc:
                raise RuntimeError("envelope journal contains an invalid operation") from exc
            self._compact_locked()

    def _journal_op(self, payload: dict[str, Any]) -> None:
        """Append one mutation record — callers hold ``self._lock``, so
        the journal's line order matches the in-memory mutation order."""
        self._validate_op(payload)
        if self._journal is not None:
            self._journal.append(payload)

    @staticmethod
    def _validate_op(payload: dict[str, Any]) -> None:
        """Reject semantically invalid records, even when their hash verifies."""
        if not isinstance(payload, dict):
            raise ValueError("envelope journal operation must be an object")
        op = payload.get("op")
        if op == "put":
            env = payload.get("envelope")
            if not isinstance(env, dict) or not isinstance(env.get("id"), str) or not env["id"]:
                raise ValueError("envelope journal put requires a nonempty string id")
            if "refresh" in payload and not isinstance(payload["refresh"], bool):
                raise ValueError("envelope journal refresh must be boolean")
            if "items" in payload:
                items = payload["items"]
                if not isinstance(items, dict) or any(
                    not isinstance(key, str)
                    or not isinstance(values, list)
                    or any(not isinstance(item, dict) for item in values)
                    for key, values in items.items()
                ):
                    raise ValueError("envelope journal put has invalid items")
        elif op in ("set_items", "delete"):
            if not isinstance(payload.get("id"), str) or not payload["id"]:
                raise ValueError("envelope journal operation requires a nonempty string id")
            if op == "set_items" and (
                not isinstance(payload.get("key"), str)
                or not isinstance(payload.get("items"), list)
                or any(not isinstance(item, dict) for item in payload["items"])
            ):
                raise ValueError("envelope journal set_items has invalid items")
        else:
            raise ValueError("unknown envelope journal operation")

    def _apply(self, payload: dict[str, Any]) -> None:
        """Replay one journaled op onto the in-memory maps (boot path —
        never re-journals)."""
        self._validate_op(payload)
        op = payload.get("op")
        if op == "put":
            env = deepcopy(payload["envelope"])
            eid = str(env["id"])
            if payload.get("refresh", True):
                self._items.pop(eid, None)
            self._items[eid] = env
            items = payload.get("items")
            if items is not None:
                self._subitems[eid] = deepcopy(items)
            while len(self._items) > self._cap:
                evicted = next(iter(self._items))
                self._items.pop(evicted)
                self._subitems.pop(evicted, None)
        elif op == "set_items":
            sid, key, items = (
                payload.get("id"),
                payload.get("key"),
                payload.get("items"),
            )
            if (
                isinstance(sid, str)
                and sid in self._items
                and isinstance(key, str)
                and isinstance(items, list)
            ):
                self._subitems.setdefault(sid, {})[key] = deepcopy(items)
                self._items[sid] = self._items.pop(sid)
        elif op == "delete":
            did = payload.get("id")
            if isinstance(did, str):
                self._items.pop(did, None)
                self._subitems.pop(did, None)

    def _compact_locked(self) -> None:
        """Rewrite the journal with only the live state — called on boot
        after a fully verified replay so dead history does not accumulate."""
        if self._journal is not None:
            self._journal.compact(
                [
                    {
                        "op": "put",
                        "envelope": env,
                        "items": {
                            k: [dict(it) for it in v]
                            for k, v in self._subitems.get(eid, {}).items()
                        },
                    }
                    for eid, env in self._items.items()
                ]
            )

    def _put_locked(
        self,
        envelope: dict[str, Any],
        items: Mapping[str, Sequence[dict[str, Any]]] | None = None,
        *,
        refresh: bool = True,
    ) -> None:
        """Snapshot, journal, then publish with the same ordering as replay."""
        payload: dict[str, Any] = {"op": "put", "envelope": deepcopy(envelope)}
        if not refresh:
            payload["refresh"] = False
        if items is not None:
            payload["items"] = {key: deepcopy(list(values)) for key, values in items.items()}
        self._journal_op(payload)
        self._apply(payload)

    def put(
        self,
        envelope: dict[str, Any],
        *,
        items: Mapping[str, Sequence[dict[str, Any]]] | None = None,
    ) -> None:
        eid = envelope.get("id")
        if not isinstance(eid, str) or not eid:
            raise ValueError("envelope carries no string 'id'")
        with self._lock:
            self._put_locked(envelope, items)

    def transition_status(
        self,
        envelope_id: str,
        *,
        expect: Iterable[str],
        status: str,
    ) -> bool:
        """Compare-and-set on the stored envelope's status — the bg
        worker's ``queued → in_progress`` claim lands only while no
        terminal verdict has. A status the poller has already seen can
        never regress: the write is refused when the stored status is
        not in ``expect``."""
        allowed = frozenset(expect)
        with self._lock:
            cur = self._items.get(envelope_id)
            if cur is None or cur.get("status") not in allowed:
                return False
            updated = deepcopy(cur)
            updated["status"] = status
            if status == "in_progress":
                # the queued → in_progress claim is the only signal that a
                # record actually left the queue — ``cancelled`` alone is
                # reachable from either side of it. The marker lets the
                # replay grammar emit ``response.in_progress`` only when the
                # phase genuinely ran. ``_fx1_*`` internals never hit the wire.
                updated["_fx1_progressed"] = True
            self._put_locked(updated, refresh=False)
            return True

    def put_unless_status(
        self,
        envelope: dict[str, Any],
        *,
        forbidden: Iterable[str],
        require_existing: bool = False,
        items: Mapping[str, Sequence[dict[str, Any]]] | None = None,
    ) -> bool:
        """``put`` guarded on the *stored* envelope's status: refuses
        when a record under the id already carries a status in
        ``forbidden``. A terminal verdict (``cancelled``) is sticky — a
        worker's late result never overwrites it. Returns whether the
        put landed. ``require_existing`` protects background updates
        from recreating an envelope that was deleted or evicted."""
        eid = envelope.get("id")
        if not isinstance(eid, str) or not eid:
            raise ValueError("envelope carries no string 'id'")
        blocked = frozenset(forbidden)
        with self._lock:
            cur = self._items.get(eid)
            if cur is None and require_existing:
                return False
            if cur is not None and cur.get("status") in blocked:
                return False
            self._put_locked(envelope, items)
            return True

    def repin(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """Refresh a replay's retrieval entry without replacing live state.

        An existing envelope wins over the idempotency cache's older
        snapshot. An absent entry is rehydrated from the cache, preserving
        the existing replay contract after deletion or capacity eviction.
        Selection, insertion and eviction share the store lock.
        """
        eid = envelope.get("id")
        if not isinstance(eid, str) or not eid:
            raise ValueError("envelope carries no string 'id'")
        with self._lock:
            live = self._items.get(eid)
            selected = live if live is not None else envelope
            self._put_locked(selected)
            return deepcopy(selected)

    def put_if_present(
        self,
        envelope: dict[str, Any],
        *,
        items: Mapping[str, Sequence[dict[str, Any]]] | None = None,
    ) -> bool:
        """Atomic check-and-``put``: lands only while the id is still in
        the index. A background turn's late writes (status flips, the
        terminal envelope) must not resurrect a record deleted
        mid-flight. Returns whether the envelope was stored."""
        eid = envelope.get("id")
        if not isinstance(eid, str) or not eid:
            raise ValueError("envelope carries no string 'id'")
        with self._lock:
            if eid not in self._items:
                return False
            self._put_locked(envelope, items)
            return True

    def get(self, envelope_id: str) -> dict[str, Any] | None:
        with self._lock:
            env = self._items.get(envelope_id)
            return deepcopy(env) if env is not None else None

    def get_items(self, envelope_id: str, key: str) -> list[dict[str, Any]] | None:
        """The request items stored under ``key`` for ``envelope_id`` —
        None when the envelope is gone, [] when it never carried them."""
        with self._lock:
            if envelope_id not in self._items:
                return None
            its = self._subitems.get(envelope_id, {}).get(key)
            return deepcopy(its) if its is not None else []

    def list_envelopes(self, object_: str) -> list[dict[str, Any]]:
        """All stored envelopes of one ``object`` type, oldest first."""
        with self._lock:
            return [deepcopy(env) for env in self._items.values() if env.get("object") == object_]

    def mutate_items(
        self,
        envelope_id: str,
        key: str,
        fn: Callable[[list[dict[str, Any]]], Sequence[dict[str, Any]]],
    ) -> list[dict[str, Any]] | None:
        """Read-modify-write one item list under the store lock: ``fn``
        gets a copy of the current list (``[]`` when the envelope never
        carried items under ``key``) and its return becomes the list.
        The whole merge — read, transform, store — is one critical
        section, so parallel mutators can't lose each other's writes and
        a delete racing in resolves as ``None`` instead of a get-then-put
        resurrection. Returns the merged list, or ``None`` when the id is
        gone."""
        with self._lock:
            if envelope_id not in self._items:
                return None
            bucket = self._subitems.get(envelope_id, {})
            merged = deepcopy(list(fn(deepcopy(bucket.get(key, [])))))
            payload = {"op": "set_items", "id": envelope_id, "key": key, "items": merged}
            self._journal_op(payload)
            self._apply(payload)
            return deepcopy(merged)

    def delete(self, envelope_id: str) -> bool:
        with self._lock:
            existed = envelope_id in self._items or envelope_id in self._subitems
            if existed:
                self._journal_op({"op": "delete", "id": envelope_id})
            self._subitems.pop(envelope_id, None)
            return self._items.pop(envelope_id, None) is not None

    def update_metadata(self, envelope_id: str, metadata: dict[str, str]) -> dict[str, Any] | None:
        """Replace a stored envelope's ``metadata`` atomically — the wire
        model already bounds the mapping (≤16 pairs / ≤64-char keys /
        ≤512-char values); a re-put keeps the envelope's slot. None when
        the id is gone (evicted, deleted, never stored)."""
        with self._lock:
            env = self._items.get(envelope_id)
            if env is None:
                return None
            updated = deepcopy(env)
            updated["metadata"] = dict(metadata)
            self._put_locked(updated, refresh=False)
            return deepcopy(updated)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
