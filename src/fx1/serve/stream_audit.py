"""stream_audit — buffered SSE-body checks across streaming surfaces.

``api_audit`` pins the wire contract's happy paths; this battery attacks
the *byte-level grammar* every ``text/event-stream`` response speaks —
``/v1/chat/completions``, ``/v1/completions``, ``/v1/responses``
(create + ``GET`` replay), ``/v1/messages``, and
``/harness/complete/stream`` — against a live in-process app (nothing
leaves the box). Starlette TestClient buffers each ASGI response before
the client can read it; these probes inspect completed response bodies,
not live network delivery or client-disconnect propagation:

- *Frame grammar* — blocks are blank-line delimited; every physical line
  is a legal SSE field (``id:``/``event:``/``data:`` or a ``:`` comment);
  the body ends ``\\n\\n``; no phantom frames; no bytes after the
  terminal marker; embedded newlines and non-ASCII content stay inside a
  single ``data:`` line (JSON escapes, never raw).
- *Chat grammar* — ``chat.completion.chunk`` frames: ``delta.role``
  first, content deltas, exactly one ``finish_reason`` frame, ``[DONE]``
  last; deltas join byte-identical to the non-stream twin's
  ``message.content``; ``n>1`` groups per index (never interleaved);
  ``tool_calls`` ride one verbatim ``delta.tool_calls`` frame; provider
  ``logprobs`` ride one aggregated ``delta.logprobs`` frame;
  ``stream_options.include_usage`` emits the ``choices:[]`` usage chunk
  only when asked.
- *Responses grammar* — ``response.created``/``in_progress`` →
  ``output_item.added``/``content_part.added`` → ``output_text.delta``×n
  → part/item ``done`` → ``response.completed`` last (no ``[DONE]``
  sentinel — the dialect's terminal is an event); accumulated deltas are
  byte-identical to the non-stream twin's ``output_text``; function calls
  ship ``response.function_call_arguments.delta`` frames whose join
  parses to the item's ``arguments``; ``max_tool_calls`` truncation
  terminates in ``response.incomplete``.
- *Anthropic grammar* — ``message_start`` → ``ping`` → the
  ``content_block_*`` lifecycle → ``message_delta`` → ``message_stop``;
  ``event:`` always equals ``data.type``; ``tool_use`` ships its input as
  one ``input_json_delta`` ``partial_json``.
- *Harness grammar* — ``{"type":"token"}`` per gated chunk →
  ``{"type":"final"}`` → ``[DONE]``; tools/logprobs refuse 501 before any
  frame; a backend without ``stream`` refuses 501; ``: keepalive``
  comments hold a slow generation open; a backend fault past the grace
  window arrives as a terminal ``{"type":"error"}`` data frame +
  ``[DONE]`` with HTTP 200 already committed — never a silent close, and
  never an ungated token frame (the buffer+gate phase is all-or-nothing).
- *Replay determinism* — ``GET /v1/responses/{id}?stream=true``
  regenerates the create-time stream byte-for-byte; ``starting_after``
  slices by the absolute ``id:`` index; keyed POSTs replay
  byte-identically; ``Last-Event-ID`` re-emits the strict byte suffix
  (400 malformed/non-stream/no-key, 409 on a miss); the live-follow
  stream on an in-flight background response emits its prelude then
  ``: keepalive`` comments until the terminal event. A separate buffered
  follow fixture contains no terminal frame before record deletion;
  post-deletion checks continue to return 404.
- *Concurrency* — N parallel streams over one app never interleave:
  every body's frames carry only that request's content with dense ids.
- *Failures* — pre-stream errors are ordinary dialect-JSON errors, never
  SSE. Partial reads of already-buffered responses preserve stored
  records and responsiveness; they do not exercise an early disconnect.

Probes are literal bools: ``True`` pins a contract that holds;
``False`` pins a measured divergence — the sealed receipt names every
defect by probe name so the finding survives byte-for-byte.

Sealed ``stream_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fx1.harness import Harness
from fx1.serve.backends import ToolCompletion
from fx1.serve.finetune import FTJobOutcome

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

__all__ = ["stream_audit", "stream_audit_bench"]

# every env that can bend app construction or backend resolution —
# cleared for the audit so ambient settings never leak into a probe
_ENV_KEYS = (
    "FX1_API_KEY",
    "MOONSHOT_API_KEY",
    "FX1_CHECKPOINT_DIR",
    "FX1_BYOK_BASE_URL",
    "FX1_BYOK_API_KEY",
    "FX1_BYOK_MODEL",
    "FX1_LOCAL_SERVE_URL",
    "FX1_LOCAL_SERVE_CMD",
    "FX1_LOCAL_MODEL",
    "FX1_LOCAL_API_KEY",
    "FX1_FT_DIR",
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
    "FX1_API_STATE_DIR",
    "FX1_API_HOST",
    "FX1_API_PORT",
)

_SSE_CT = "text/event-stream"
_ECHO_PROMPT = "audit-tick " + "echo " * 40  # ~200 chars → several delta pieces
_UNI_PROMPT = "first líne\nsecond ✓ end"  # raw newline + non-ASCII in content
_TOOL_CALL_ID = "call_7f2a9b"
_TOOL_NAME = "get_price"
_TOOL_ARGS_TEXT = '{"symbol":"ESZ5","venue":"CME"}'
_TOOL_ARGS_OBJ = {"symbol": "ESZ5", "venue": "CME"}
_TOOL_SPEC_CHAT = {
    "type": "function",
    "function": {
        "name": _TOOL_NAME,
        "parameters": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
}
_TOOL_SPEC_RESP = {
    "type": "function",
    "name": _TOOL_NAME,
    "parameters": {
        "type": "object",
        "properties": {"symbol": {"type": "string"}},
        "required": ["symbol"],
    },
}
_TOOL_SPEC_ANTH = {
    "name": _TOOL_NAME,
    "input_schema": {
        "type": "object",
        "properties": {"symbol": {"type": "string"}},
        "required": ["symbol"],
    },
}
_STUB_USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
_CONC_N = 6
_EV_CREATED = "response.created"
_EV_INCOMPLETE = "response.incomplete"
_KEEPALIVE = ": keepalive"
_TERMINAL_EVENTS = {
    "response.completed",
    _EV_INCOMPLETE,
    "response.failed",
    "response.cancelled",
}


# ---------------------------------------------------------------------------
# SSE grammar parser — every physical line must be a spec field; anything
# else is a malformed frame the probes flag by name.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SseFrame:
    """One blank-line-delimited SSE block, classified line by line."""

    raw: str
    seq: str | None  # the ``id:`` field, verbatim
    event: str | None  # the ``event:`` field
    data: str | None  # joined ``data:`` payload lines (None when absent)
    comment_only: bool  # every line a ``:`` comment (keepalive)
    malformed: bool  # some line carried no legal SSE field


def _sse_frames(body: str) -> list[_SseFrame]:
    """Split a wire body into frames; a block that is not entirely
    ``id:``/``event:``/``data:``/``:`` lines is flagged malformed."""
    frames: list[_SseFrame] = []
    for block in body.split("\n\n"):
        block = block.strip("\r\n")
        if not block:
            continue
        seq: str | None = None
        event: str | None = None
        datas: list[str] = []
        comment = True
        malformed = False
        for ln in block.splitlines():
            if ln.startswith(":"):
                continue
            comment = False
            if ln.startswith("data:"):
                datas.append(ln[5:].removeprefix(" "))
            elif ln.startswith("id:"):
                seq = ln[3:].removeprefix(" ")
            elif ln.startswith("event:"):
                event = ln[6:].removeprefix(" ")
            else:
                malformed = True
        frames.append(
            _SseFrame(
                raw=block,
                seq=seq,
                event=event,
                data="\n".join(datas) if datas else None,
                comment_only=comment,
                malformed=malformed,
            )
        )
    return frames


def _data_frames(frames: list[_SseFrame]) -> list[_SseFrame]:
    return [f for f in frames if f.data is not None]


def _json_datas(frames: list[_SseFrame]) -> list[dict[str, Any]]:
    return [
        json.loads(f.data)
        for f in _data_frames(frames)
        if f.data is not None and f.data != "[DONE]"
    ]


def _event_payloads(frames: list[_SseFrame]) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for f in _data_frames(frames):
        if f.event is not None and f.data is not None:
            out.append((f.event, json.loads(f.data)))
    return out


def _grammar(out: dict[str, bool], tag: str, body: str, frames: list[_SseFrame]) -> None:
    """The shared physical-layer assertions every dialect inherits."""
    datas = _data_frames(frames)
    out[f"{tag}_body_terminated"] = body.endswith("\n\n")
    out[f"{tag}_no_malformed_lines"] = all(not f.malformed for f in frames)
    out[f"{tag}_ids_dense"] = [f.seq for f in datas] == [str(i) for i in range(len(datas))]


# ---------------------------------------------------------------------------
# Stub backends — deterministic gated output the wire must carry verbatim.
# ---------------------------------------------------------------------------


class _StubBackend:
    """Clean gated completion — echoes the last user turn, reports usage."""

    def __init__(self) -> None:
        self._model = "stream-stub-0"
        self.last_usage: dict[str, int] = dict(_STUB_USAGE)

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        del sampling
        return f"stub:{messages[-1]['content']}"

    def close(self) -> None:
        pass


class _ToolStubBackend(_StubBackend):
    """Answers the tools channel — content None, one verbatim call, and a
    provider ``logprobs`` payload whenever the request asked for it."""

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: Any = None,
        tools: Any = None,
        tool_choice: Any = None,
        parallel_tool_calls: Any = None,
        logprobs: Any = None,
        top_logprobs: Any = None,
    ) -> ToolCompletion:
        del messages, sampling, tools, tool_choice, parallel_tool_calls, top_logprobs
        return ToolCompletion(
            content=None,
            tool_calls=(
                {
                    "id": _TOOL_CALL_ID,
                    "type": "function",
                    "index": 0,
                    "function": {"name": _TOOL_NAME, "arguments": _TOOL_ARGS_TEXT},
                },
            ),
            finish_reason="tool_calls",
            logprobs=(
                {"content": [{"token": "px", "logprob": -0.07, "top_logprobs": []}]}
                if logprobs
                else None
            ),
        )


class _StreamStubBackend(_StubBackend):
    """A real ``StreamingBackend`` — prompt-tagged chunks so concurrent
    streams prove their bytes never cross wires."""

    def stream(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> Iterator[str]:
        del sampling
        tag = messages[-1]["content"]
        yield f"{tag}:a"
        yield f"{tag}:b"
        yield f"{tag}:c"


class _SlowStreamBackend(_StubBackend):
    """Streams nothing until a delay passes — drives the keepalive path."""

    def __init__(self, delay_s: float = 0.5) -> None:
        super().__init__()
        self._delay = delay_s

    def stream(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> Iterator[str]:
        del sampling
        time.sleep(self._delay)
        yield "post-sleep-a"
        yield "post-sleep-b"


class _FailStreamBackend(_StubBackend):
    """Yields a chunk then dies past the keepalive grace window — the
    mid-stream failure the wire must report as an SSE error event."""

    def __init__(self, delay_s: float = 0.4) -> None:
        super().__init__()
        self._delay = delay_s

    def stream(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> Iterator[str]:
        del messages, sampling
        yield "never-emitted"  # buffered: the gate dies before any frame ships
        time.sleep(self._delay)
        raise RuntimeError("engine died mid-generation")


class _DirtyStreamBackend(_StubBackend):
    """Streams a forbidden research headline — the honesty gate must refuse
    it before any frame ships (a pre-stream 502, not an SSE error)."""

    def stream(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> Iterator[str]:
        del messages, sampling
        yield "total Sharpe 4.2 on NAV"  # NOSONAR — gate-bait fixture text
        yield " more"


class _FailBackend(_StubBackend):
    """Dies inside ``complete`` — pre-stream failure on /v1 surfaces, and
    the background-worker failure behind ``response.failed``."""

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        del messages, sampling
        raise RuntimeError("backend boom — gate never reached")


class _SlowFailBackend(_StubBackend):
    """Sleeps past the grace window then dies — a mid-window backend
    fault: the wire is already a keepalived SSE stream, so the failure
    lands as an in-band error frame, never a JSON 5xx."""

    def __init__(self, delay_s: float = 0.4) -> None:
        super().__init__()
        self._delay = delay_s

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        del messages, sampling
        time.sleep(self._delay)
        raise RuntimeError("engine died mid-generation")


class _SlowBackend(_StubBackend):
    """``complete`` sleeps — a background response stays in_progress long
    enough for the replay follow loop to emit ``: keepalive`` comments."""

    def __init__(self, delay_s: float = 0.6) -> None:
        super().__init__()
        self._delay = delay_s

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        time.sleep(self._delay)
        return super().complete(messages)


class _GatedBackend(_StubBackend):
    """``complete`` blocks on an event — the deterministic way to hold a
    background response in-flight while a cancel or delete lands."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        self.gate.wait(timeout=30.0)
        return super().complete(messages)


# ---------------------------------------------------------------------------
# App construction
# ---------------------------------------------------------------------------


@dataclass
class _Ctx:
    """One app's test surface."""

    client: TestClient
    app: FastAPI


def _fast_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    del argv, timeout_s
    return 0, "ok", ""


def _ft_runner(spec: Any, *, emit: Any, should_cancel: Any) -> FTJobOutcome:
    emit("info", "bench runner")
    return FTJobOutcome(fine_tuned_model=None)


def _app(
    workdir: Path,
    backend: Any,
    *,
    sse_keepalive_s: float = 15.0,
) -> FastAPI:
    import fx1.serve.api as api_mod  # noqa: PLC0415

    return api_mod.create_app(
        harness=Harness(runner=_fast_runner),
        backend_resolver=lambda *a, **k: backend,
        ft_runner=_ft_runner,
        ft_dir=workdir / "ft",
        state_dir=workdir / "state",
        sse_keepalive_s=sse_keepalive_s,
    )


def _make_ctx(
    workdir: Path,
    backend: Any,
    *,
    sse_keepalive_s: float = 15.0,
) -> _Ctx:
    from fastapi.testclient import TestClient  # noqa: PLC0415

    app = _app(workdir, backend, sse_keepalive_s=sse_keepalive_s)
    return _Ctx(client=TestClient(app, raise_server_exceptions=False), app=app)


def _parallel(n: int, fn: Callable[[int], Any]) -> list[Any]:
    """Run ``fn(i)`` on N barrier-released threads over one client."""
    barrier = threading.Barrier(n)
    out: list[Any] = [None] * n

    def w(i: int) -> None:
        barrier.wait()
        out[i] = fn(i)

    ts = [threading.Thread(target=w, args=(i,), daemon=True) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=60)
    return out


# ---------------------------------------------------------------------------
# Chat completions — the ``chat.completion.chunk`` + ``[DONE]`` dialect
# ---------------------------------------------------------------------------


def _chat_body(prompt: str, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "fx1",
        "messages": [{"role": "user", "content": prompt}],
    }
    body.update(extra)
    return body


def _chat_chunks(frames: list[_SseFrame]) -> list[dict[str, Any]]:
    return _json_datas(frames)


def _chat_deltas(chunks: list[dict[str, Any]], index: int = 0) -> list[dict[str, Any]]:
    return [
        c["choices"][0]["delta"]
        for c in chunks
        if c.get("choices") and c["choices"][0]["index"] == index
    ]


def _probe_chat(ctx: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    body = _chat_body(_ECHO_PROMPT, stream=True)
    r = ctx.client.post("/v1/chat/completions", json=body)
    twin_r = ctx.client.post("/v1/chat/completions", json=_chat_body(_ECHO_PROMPT))
    twin = twin_r.json()
    frames = _sse_frames(r.text)
    datas = _data_frames(frames)
    done = [f for f in datas if f.data == "[DONE]"]
    chunks = _chat_chunks(frames)
    choice_frames = [c for c in chunks if c.get("choices")]
    deltas = _chat_deltas(chunks)
    contents = [d["content"] for d in deltas if "content" in d]
    expected = f"stub:{_ECHO_PROMPT}"
    twin_content = twin["choices"][0]["message"]["content"]

    out["chat_status_200"] = r.status_code == 200
    out["chat_sse_content_type"] = r.headers.get("content-type", "").startswith(_SSE_CT)
    _grammar(out, "chat", r.text, frames)
    out["chat_no_event_lines"] = all(f.event is None for f in datas)
    out["chat_field_order"] = all(f.raw.startswith("id: ") and "\ndata: " in f.raw for f in datas)
    out["chat_done_once_last"] = len(done) == 1 and datas[-1].data == "[DONE]"
    out["chat_no_bytes_after_done"] = r.text.endswith("data: [DONE]\n\n")
    out["chat_chunk_object"] = all(
        c.get("object") == "chat.completion.chunk" and str(c.get("id", "")).startswith("chatcmpl-")
        for c in chunks
    )
    out["chat_chunk_meta_consistent"] = (
        len({c["id"] for c in chunks}) == 1
        and len({c["created"] for c in chunks}) == 1
        and all(c.get("model") == "stream-stub-0" for c in chunks)
    )
    out["chat_delta_role_first"] = bool(deltas) and deltas[0] == {"role": "assistant"}
    out["chat_finish_exactly_once"] = (
        len(choice_frames) >= 2
        and sum(1 for cf in choice_frames if cf["choices"][0].get("finish_reason")) == 1
        and choice_frames[-1]["choices"][0]["finish_reason"] == "stop"
        and choice_frames[-1]["choices"][0]["delta"] == {}
    )
    out["chat_finish_null_mid"] = all(
        cf["choices"][0].get("finish_reason") is None for cf in choice_frames[:-1]
    )
    out["chat_single_choice_index0"] = all(
        len(c["choices"]) == 1 and c["choices"][0]["index"] == 0 for c in choice_frames
    )
    out["chat_multi_deltas"] = len(contents) >= 2 and all(
        isinstance(c, str) and c for c in contents
    )
    out["chat_deltas_join_twin"] = "".join(contents) == twin_content == expected
    out["chat_usage_absent_default"] = all(c.get("choices") for c in chunks)
    out["chat_done_seq_next"] = bool(done) and datas[-1].seq == str(len(datas) - 1)
    out["chat_completion_id_header"] = bool(r.headers.get("x-fx1-completion-id"))
    out["chat_twin_nonstream_json"] = (
        twin_r.status_code == 200
        and twin_r.headers.get("content-type", "").startswith("application/json")
        and twin.get("object") == "chat.completion"
    )
    return out


def _probe_chat_framing(ctx: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    r = ctx.client.post("/v1/chat/completions", json=_chat_body(_UNI_PROMPT, stream=True))
    frames = _sse_frames(r.text)
    chunks = _chat_chunks(frames)
    contents = [d["content"] for d in _chat_deltas(chunks) if "content" in d]
    datas = _data_frames(frames)
    # a raw newline inside the JSON payload would split a data: line in
    # two — the continuation flags malformed; unicode must round-trip
    out["chat_unicode_roundtrip"] = "".join(contents) == f"stub:{_UNI_PROMPT}"
    out["chat_data_lines_singleline"] = all(
        sum(1 for ln in f.raw.splitlines() if ln.startswith("data:")) == 1 for f in datas
    ) and all(not f.malformed for f in frames)
    return out


def _probe_chat_tools(ctx: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    body = _chat_body(
        "price of ESZ5",
        stream=True,
        tools=[_TOOL_SPEC_CHAT],
        tool_choice="auto",
    )
    r = ctx.client.post("/v1/chat/completions", json=body)
    twin = ctx.client.post(
        "/v1/chat/completions",
        json=_chat_body("price of ESZ5", tools=[_TOOL_SPEC_CHAT], tool_choice="auto"),
    ).json()
    frames = _sse_frames(r.text)
    datas = _data_frames(frames)
    chunks = _chat_chunks(frames)
    deltas = _chat_deltas(chunks)
    tc_frames = [d for d in deltas if "tool_calls" in d]
    out["chat_tool_status_200"] = r.status_code == 200
    out["chat_tool_delta_single_frame"] = len(tc_frames) == 1
    tc = (tc_frames[0]["tool_calls"] or [{}])[0]
    out["chat_tool_calls_verbatim"] = (
        tc.get("id") == _TOOL_CALL_ID
        and tc.get("type") == "function"
        and tc.get("index") == 0
        and tc.get("function", {}).get("name") == _TOOL_NAME
        and tc.get("function", {}).get("arguments") == _TOOL_ARGS_TEXT
    )
    out["chat_tool_no_content_deltas"] = all("content" not in d for d in deltas)
    out["chat_tool_finish_reason"] = (
        chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
        and chunks[-1]["choices"][0]["delta"] == {}
    )
    out["chat_tool_done_last"] = datas[-1].data == "[DONE]"
    out["chat_tool_twin_shape"] = (
        twin["choices"][0]["message"]["content"] is None
        and twin["choices"][0]["message"]["tool_calls"][0]["id"] == _TOOL_CALL_ID
        and twin["choices"][0]["finish_reason"] == "tool_calls"
    )
    return out


def _probe_chat_logprobs(ctx: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    r = ctx.client.post(
        "/v1/chat/completions",
        json=_chat_body("lp", stream=True, logprobs=True),
    )
    frames = _sse_frames(r.text)
    chunks = _chat_chunks(frames)
    deltas = _chat_deltas(chunks)
    lp_frames = [d for d in deltas if "logprobs" in d]
    tc_frames = [d for d in deltas if "tool_calls" in d]
    fin = [d for d in deltas if d == {}]
    out["chat_logprobs_single_frame"] = len(lp_frames) == 1 and lp_frames[0]["logprobs"] == {
        "content": [{"token": "px", "logprob": -0.07, "top_logprobs": []}]
    }
    out["chat_logprobs_before_finish"] = bool(lp_frames) and deltas.index(
        lp_frames[0]
    ) < deltas.index(fin[-1])
    out["chat_tool_calls_before_content"] = bool(tc_frames) and deltas.index(
        tc_frames[0]
    ) < deltas.index(lp_frames[0])
    return out


def _probe_chat_usage(ctx: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    r = ctx.client.post(
        "/v1/chat/completions",
        json=_chat_body(_ECHO_PROMPT, stream=True, stream_options={"include_usage": True}),
    )
    frames = _sse_frames(r.text)
    chunks = _chat_chunks(frames)
    usage_frames = [c for c in chunks if c.get("choices") == []]
    datas = _data_frames(frames)
    out["chat_usage_chunk_once"] = len(usage_frames) == 1
    out["chat_usage_chunk_penultimate"] = (
        bool(usage_frames) and chunks[-1] in usage_frames and datas[-1].data == "[DONE]"
    )
    out["chat_usage_values"] = bool(usage_frames) and usage_frames[0].get("usage") == _STUB_USAGE
    return out


def _probe_chat_n2(ctx: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    r = ctx.client.post("/v1/chat/completions", json=_chat_body(_ECHO_PROMPT, stream=True, n=2))
    twin = ctx.client.post("/v1/chat/completions", json=_chat_body(_ECHO_PROMPT, n=2)).json()
    frames = _sse_frames(r.text)
    chunks = _chat_chunks(frames)
    choice_frames = [c for c in chunks if c.get("choices")]
    indexes = [cf["choices"][0]["index"] for cf in choice_frames]
    per_choice_ok = True
    for idx in (0, 1):
        ds = _chat_deltas(chunks, index=idx)
        contents = [d["content"] for d in ds if "content" in d]
        idx_frames = [cf for cf in choice_frames if cf["choices"][0]["index"] == idx]
        per_choice_ok = (
            per_choice_ok
            and ds[0] == {"role": "assistant"}
            and idx_frames[-1]["choices"][0]["finish_reason"] == "stop"
            and "".join(contents) == twin["choices"][idx]["message"]["content"]
        )
    datas = _data_frames(frames)
    out["chat_n2_grouped_not_interleaved"] = (
        indexes == [0] * indexes.count(0) + [1] * indexes.count(1) and len(set(indexes)) == 2
    )
    out["chat_n2_each_complete"] = per_choice_ok
    out["chat_n2_shared_ids"] = [f.seq for f in datas] == [str(i) for i in range(len(datas))]
    out["chat_n2_done_last"] = datas[-1].data == "[DONE]"
    return out


def _probe_completions(ctx: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    r = ctx.client.post(
        "/v1/completions", json={"model": "fx1", "prompt": _ECHO_PROMPT, "stream": True}
    )
    twin_r = ctx.client.post("/v1/completions", json={"model": "fx1", "prompt": _ECHO_PROMPT})
    twin = twin_r.json()
    frames = _sse_frames(r.text)
    datas = _data_frames(frames)
    chunks = _chat_chunks(frames)
    texts = [
        c["choices"][0].get("text", "")
        for c in chunks
        if c.get("choices") and c["choices"][0].get("finish_reason") is None
    ]
    fin_frames = [c for c in chunks if c.get("choices") and c["choices"][0].get("finish_reason")]
    out["legacy_status_200"] = r.status_code == 200
    out["legacy_sse_content_type"] = r.headers.get("content-type", "").startswith(_SSE_CT)
    _grammar(out, "legacy", r.text, frames)
    out["legacy_no_event_lines"] = all(f.event is None for f in datas)
    out["legacy_object_text_completion"] = all(c.get("object") == "text_completion" for c in chunks)
    out["legacy_done_once_last"] = (
        sum(1 for f in datas if f.data == "[DONE]") == 1 and datas[-1].data == "[DONE]"
    )
    out["legacy_join_twin"] = (
        "".join(texts) == twin["choices"][0]["text"] == (f"stub:{_ECHO_PROMPT}")
    )
    out["legacy_finish_last_choice"] = (
        len(fin_frames) == 1 and fin_frames[0]["choices"][0]["finish_reason"] == "stop"
    )
    return out


# ---------------------------------------------------------------------------
# Responses — the ``event:``-typed dialect ending in ``response.completed``
# ---------------------------------------------------------------------------


def _probe_responses(ctx: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    body = {"model": "fx1", "input": _ECHO_PROMPT, "stream": True}
    r = ctx.client.post("/v1/responses", json=body)
    twin_r = ctx.client.post("/v1/responses", json={"model": "fx1", "input": _ECHO_PROMPT})
    twin = twin_r.json()
    frames = _sse_frames(r.text)
    datas = _data_frames(frames)
    events = _event_payloads(frames)
    names = [e for e, _ in events]
    payloads = [p for _, p in events]
    expected = f"stub:{_ECHO_PROMPT}"
    twin_text = twin["output"][0]["content"][0]["text"]

    out["resp_status_200"] = r.status_code == 200
    out["resp_sse_content_type"] = r.headers.get("content-type", "").startswith(_SSE_CT)
    _grammar(out, "resp", r.text, frames)
    out["resp_event_data_pairs"] = all(
        f.event is not None and f.data is not None and json.loads(f.data).get("type") == f.event
        for f in datas
    )
    out["resp_field_order"] = all(
        f.raw.startswith("event: ") and "\nid: " in f.raw and "\ndata: " in f.raw for f in datas
    )
    out["resp_no_done_sentinel"] = all(f.data != "[DONE]" for f in datas)
    out["resp_seq_head"] = names[:4] == [
        _EV_CREATED,
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
    ]
    out["resp_seq_tail"] = names[-2:] == [
        "response.output_item.done",
        "response.completed",
    ]
    out["resp_completed_last"] = (
        names[-1] == "response.completed" and payloads[-1]["response"]["status"] == "completed"
    )
    created = payloads[0]["response"]
    out["resp_created_prelude_shape"] = (
        created["status"] == "in_progress"
        and created["output"] == []
        and created["usage"] is None
        and created["object"] == "response"
        and created["id"].startswith("resp_")
    )
    deltas = [p["delta"] for e, p in events if e == "response.output_text.delta"]
    done_texts = [p["text"] for e, p in events if e == "response.output_text.done"]
    out["resp_delta_count_multi"] = len(deltas) >= 2
    out["resp_deltas_join_done"] = bool(done_texts) and "".join(deltas) == done_texts[0]
    out["resp_deltas_join_twin"] = "".join(deltas) == twin_text == expected
    out["resp_completed_is_retrieve"] = (
        payloads[-1]["response"]["output"][0]["content"][0]["text"] == twin_text
        and payloads[-1]["response"]["usage"] == twin["usage"]
    )
    out["resp_item_lifecycle"] = (
        payloads[2]["item"]["type"] == "message"
        and payloads[2]["item"]["status"] == "in_progress"
        and payloads[-2]["item"]["status"] == "completed"
        and payloads[3]["part"]["type"] == "output_text"
        and payloads[3]["part"]["text"] == ""
    )
    out["resp_done_text_full"] = bool(done_texts) and done_texts[0] == expected
    out["resp_twin_nonstream_json"] = (
        twin_r.status_code == 200
        and twin.get("object") == "response"
        and twin["status"] == "completed"
    )
    return out


def _probe_resp_tools(ctx: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    r = ctx.client.post(
        "/v1/responses",
        json={
            "model": "fx1",
            "input": "price of ESZ5",
            "stream": True,
            "tools": [_TOOL_SPEC_RESP],
        },
    )
    frames = _sse_frames(r.text)
    events = _event_payloads(frames)
    names = [e for e, _ in events]
    payloads = [p for _, p in events]
    arg_deltas = [p["delta"] for e, p in events if e == "response.function_call_arguments.delta"]
    arg_done = [p for e, p in events if e == "response.function_call_arguments.done"]
    added = [p for e, p in events if e == "response.output_item.added"]
    completed = payloads[-1]["response"]
    call_items = [it for it in completed["output"] if it["type"] == "function_call"]

    out["resp_tool_status_200"] = r.status_code == 200
    out["resp_tool_no_message_item"] = all(p["item"]["type"] != "message" for p in added)
    out["resp_tool_item_added_shape"] = (
        len(added) == 1
        and added[0]["item"]["type"] == "function_call"
        and added[0]["item"]["call_id"] == _TOOL_CALL_ID
        and added[0]["item"]["name"] == _TOOL_NAME
        and added[0]["item"]["status"] == "in_progress"
    )
    out["resp_tool_args_deltas_join"] = (
        bool(arg_deltas)
        and bool(arg_done)
        and "".join(arg_deltas) == arg_done[0]["arguments"] == _TOOL_ARGS_TEXT
        and json.loads(arg_done[0]["arguments"]) == _TOOL_ARGS_OBJ
    )
    out["resp_tool_completed_item"] = (
        len(call_items) == 1
        and call_items[0]["arguments"] == _TOOL_ARGS_TEXT
        and call_items[0]["call_id"] == _TOOL_CALL_ID
        and call_items[0]["status"] == "completed"
    )
    out["resp_tool_completed_last"] = names[-1] == "response.completed"
    # ``max_tool_calls: 0`` truncates the calls-only turn — the terminal
    # event flips to response.incomplete carrying the reason
    r2 = ctx.client.post(
        "/v1/responses",
        json={
            "model": "fx1",
            "input": "price of ESZ5",
            "stream": True,
            "tools": [_TOOL_SPEC_RESP],
            "max_tool_calls": 0,
        },
    )
    events2 = _event_payloads(_sse_frames(r2.text))
    names2 = [e for e, _ in events2]
    term = events2[-1][1]["response"]
    out["resp_incomplete_terminal"] = names2[-1] == _EV_INCOMPLETE
    out["resp_incomplete_reason"] = term["status"] == "incomplete" and term[
        "incomplete_details"
    ] == {"reason": "max_tool_calls"}
    return out


# ---------------------------------------------------------------------------
# Anthropic — the ``message_*``/``content_block_*`` dialect
# ---------------------------------------------------------------------------


def _anth_body(prompt: str, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "fx1",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": prompt}],
    }
    body.update(extra)
    return body


def _probe_anthropic(ctx: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    r = ctx.client.post("/v1/messages", json=_anth_body(_ECHO_PROMPT, stream=True))
    twin_r = ctx.client.post("/v1/messages", json=_anth_body(_ECHO_PROMPT))
    twin = twin_r.json()
    frames = _sse_frames(r.text)
    datas = _data_frames(frames)
    events = _event_payloads(frames)
    names = [e for e, _ in events]
    payloads = [p for _, p in events]
    expected = f"stub:{_ECHO_PROMPT}"
    twin_text = twin["content"][0]["text"]

    out["anth_status_200"] = r.status_code == 200
    out["anth_sse_content_type"] = r.headers.get("content-type", "").startswith(_SSE_CT)
    _grammar(out, "anth", r.text, frames)
    out["anth_event_type_matches"] = all(p.get("type") == e for e, p in events)
    out["anth_field_order"] = all(
        f.raw.startswith("id: ") and "\nevent: " in f.raw and "\ndata: " in f.raw for f in datas
    )
    out["anth_no_done_sentinel"] = all(f.data != "[DONE]" for f in datas)
    out["anth_seq_head"] = names[:3] == [
        "message_start",
        "ping",
        "content_block_start",
    ]
    out["anth_seq_tail"] = names[-2:] == ["message_delta", "message_stop"]
    msg_start = payloads[0]["message"]
    out["anth_message_start_shape"] = (
        msg_start["role"] == "assistant"
        and msg_start["content"] == []
        and msg_start["stop_reason"] is None
        and isinstance(msg_start["usage"]["input_tokens"], int)
    )
    out["anth_ping_once"] = names.count("ping") == 1 and names[1] == "ping"
    starts = [p for e, p in events if e == "content_block_start"]
    deltas_p = [p for e, p in events if e == "content_block_delta"]
    stops = [p for e, p in events if e == "content_block_stop"]
    out["anth_block_lifecycle"] = (
        len(starts) == 1
        and starts[0]["index"] == 0
        and starts[0]["content_block"] == {"type": "text", "text": ""}
        and len(stops) == 1
        and stops[0]["index"] == 0
    )
    text_deltas = [p["delta"]["text"] for p in deltas_p if p["delta"]["type"] == "text_delta"]
    out["anth_text_deltas_join"] = "".join(text_deltas) == twin_text == expected
    md = payloads[-2]
    out["anth_message_delta_shape"] = md["delta"]["stop_reason"] == "end_turn" and isinstance(
        md["usage"]["output_tokens"], int
    )
    out["anth_stop_last"] = names[-1] == "message_stop"
    out["anth_twin_nonstream_json"] = (
        twin_r.status_code == 200
        and twin.get("type") == "message"
        and twin["stop_reason"] == "end_turn"
    )
    return out


def _probe_anthropic_tools(ctx: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    r = ctx.client.post(
        "/v1/messages",
        json=_anth_body(
            "price of ESZ5",
            stream=True,
            tools=[_TOOL_SPEC_ANTH],
            tool_choice={"type": "auto"},
        ),
    )
    frames = _sse_frames(r.text)
    events = _event_payloads(frames)
    names = [e for e, _ in events]
    starts = [p for e, p in events if e == "content_block_start"]
    deltas_p = [p for e, p in events if e == "content_block_delta"]
    md = [p for e, p in events if e == "message_delta"]
    out["anth_tool_block_shape"] = (
        len(starts) == 1
        and starts[0]["content_block"]["type"] == "tool_use"
        and starts[0]["content_block"]["id"] == _TOOL_CALL_ID
        and starts[0]["content_block"]["name"] == _TOOL_NAME
        and starts[0]["content_block"]["input"] == {}
    )
    out["anth_tool_input_delta"] = (
        len(deltas_p) == 1
        and deltas_p[0]["delta"]["type"] == "input_json_delta"
        and json.loads(deltas_p[0]["delta"]["partial_json"]) == _TOOL_ARGS_OBJ
    )
    out["anth_tool_stop_reason"] = len(md) == 1 and md[0]["delta"]["stop_reason"] == "tool_use"
    out["anth_tool_stop_last"] = names[-1] == "message_stop"
    return out


# ---------------------------------------------------------------------------
# /harness/complete/stream — the token/final/[DONE] dialect
# ---------------------------------------------------------------------------


def _hstream_body(prompt: str, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "backend": "byok",
        "messages": [{"role": "user", "content": prompt}],
    }
    body.update(extra)
    return body


def _probe_hstream(ctx: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    r = ctx.client.post("/harness/complete/stream", json=_hstream_body("hs-tag"))
    frames = _sse_frames(r.text)
    datas = _data_frames(frames)
    payloads = _json_datas(frames)
    out["hstream_status_200"] = r.status_code == 200
    out["hstream_sse_content_type"] = r.headers.get("content-type", "").startswith(_SSE_CT)
    out["hstream_body_terminated"] = r.text.endswith("\n\n")
    out["hstream_no_malformed_lines"] = all(not f.malformed for f in frames)
    out["hstream_no_id_lines"] = all(f.seq is None for f in datas)
    out["hstream_no_event_lines"] = all(f.event is None for f in datas)
    out["hstream_data_only_frames"] = all(f.raw.startswith("data: ") for f in datas)
    token_payloads = [p for p in payloads if p.get("type") == "token"]
    finals = [p for p in payloads if p.get("type") == "final"]
    out["hstream_token_shape"] = len(token_payloads) == 3 and all(
        isinstance(p.get("content"), str) for p in token_payloads
    )
    out["hstream_chunk_boundaries"] = [p["content"] for p in token_payloads] == [
        "hs-tag:a",
        "hs-tag:b",
        "hs-tag:c",
    ]
    out["hstream_final_shape"] = (
        len(finals) == 1
        and finals[0]["model"] == "stream-stub-0"
        and finals[0]["usage"] == _STUB_USAGE
        and isinstance(finals[0]["completion_id"], str)
        and isinstance(finals[0]["receipt_sha256"], str)
        and isinstance(finals[0]["latency_ms"], (int, float))
    )
    out["hstream_done_last"] = datas[-1].data == "[DONE]" and [p.get("type") for p in payloads] == [
        "token",
        "token",
        "token",
        "final",
    ]
    return out


def _probe_hstream_refusals(ctx: _Ctx, ctx_plain: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    # tools and logprobs are non-streamable channels — loud 501 before any frame
    r_tools = ctx.client.post(
        "/harness/complete/stream",
        json=_hstream_body("hs", tools=[_TOOL_SPEC_CHAT]),
    )
    r_lp = ctx.client.post("/harness/complete/stream", json=_hstream_body("hs", logprobs=True))
    out["hstream_tools_501_json"] = (
        r_tools.status_code == 501
        and not r_tools.headers.get("content-type", "").startswith(_SSE_CT)
        and "streamable" in r_tools.text
    )
    out["hstream_logprobs_501_json"] = r_lp.status_code == 501 and not r_lp.headers.get(
        "content-type", ""
    ).startswith(_SSE_CT)
    # a backend without a stream method refuses 501 rather than faking chunks
    r_no = ctx_plain.client.post("/harness/complete/stream", json=_hstream_body("hs"))
    out["hstream_no_channel_501_json"] = r_no.status_code == 501 and not r_no.headers.get(
        "content-type", ""
    ).startswith(_SSE_CT)
    return out


def _probe_hstream_gate(ctx: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    r = ctx.client.post("/harness/complete/stream", json=_hstream_body("hs"))
    out["hstream_gate_502_json"] = r.status_code == 502 and not r.headers.get(
        "content-type", ""
    ).startswith(_SSE_CT)
    out["hstream_gate_no_sse_bytes"] = "data:" not in r.text
    return out


def _probe_hstream_keepalive(ctx: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    r = ctx.client.post("/harness/complete/stream", json=_hstream_body("hs-slow"))
    frames = _sse_frames(r.text)
    comments = [f for f in frames if f.comment_only]
    datas = _data_frames(frames)
    payloads = _json_datas(frames)
    first_data_idx = next((i for i, f in enumerate(frames) if not f.comment_only), len(frames))
    out["hstream_keepalive_comments"] = (
        len(comments) >= 1
        and all(f.raw.startswith(_KEEPALIVE) for f in comments)
        and all(i < first_data_idx for i, f in enumerate(frames) if f.comment_only)
    )
    out["hstream_keepalive_then_stream"] = datas[-1].data == "[DONE]" and [
        p.get("type") for p in payloads
    ] == ["token", "token", "final"]
    out["hstream_keepalive_status_200"] = r.status_code == 200
    return out


def _probe_hstream_miderr(ctx: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    r = ctx.client.post("/harness/complete/stream", json=_hstream_body("hs-err"))
    frames = _sse_frames(r.text)
    datas = _data_frames(frames)
    payloads = _json_datas(frames)
    errors = [p for p in payloads if p.get("type") == "error"]
    out["hstream_miderr_status_200"] = r.status_code == 200
    out["hstream_miderr_no_token_frames"] = all(p.get("type") != "token" for p in payloads)
    out["hstream_miderr_error_frame"] = (
        len(errors) == 1
        and errors[0]["status"] == 502
        and "mid-generation" in str(errors[0]["detail"])
        and isinstance(errors[0]["code"], str)
    )
    out["hstream_miderr_done_after_error"] = (
        datas[-1].data == "[DONE]" and payloads[-1].get("type") == "error"
    )
    # the app survives a mid-stream fault — next request answers normally
    r2 = ctx.client.post("/harness/complete/stream", json=_hstream_body("hs-err"))
    frames2 = _sse_frames(r2.text)
    payloads2 = _json_datas(frames2)
    out["hstream_miderr_app_responsive"] = (
        r2.status_code == 200
        and bool(payloads2)
        and payloads2[-1].get("type") == "error"
        and _data_frames(frames2)[-1].data == "[DONE]"
    )
    return out


# ---------------------------------------------------------------------------
# /v1 create-stream keepalive — the grace-window dialect on every surface:
# ``: keepalive`` comments on the OpenAI legs, unnumbered ``ping`` events
# on /v1/messages, an in-band error frame on a mid-window fault, and the
# ordinary contract (headers included) when work resolves inside the
# window.
# ---------------------------------------------------------------------------


def _probe_v1_keepalive(wd: Path) -> dict[str, bool]:  # noqa: C901 — probe accumulator
    out: dict[str, bool] = {}
    ka = _make_ctx(wd / "slow", _SlowBackend(0.4), sse_keepalive_s=0.05)

    rc = ka.client.post("/v1/chat/completions", json=_chat_body(_ECHO_PROMPT, stream=True))
    fc = _sse_frames(rc.text)
    comments = [f for f in fc if f.comment_only]
    first_chunk = next((i for i, f in enumerate(fc) if f.raw.startswith("id:")), len(fc))
    out["chat_ka_comments_before_chunks"] = (
        rc.status_code == 200
        and bool(comments)
        and all(f.raw.startswith(_KEEPALIVE) for f in comments)
        and all(i < first_chunk for i, f in enumerate(fc) if f.comment_only)
    )
    # headers committed before the worker resolved — the fingerprint is
    # in-band only on the keepalived leg
    out["chat_ka_cid_inband"] = (
        "x-fx1-completion-id" not in rc.headers and '"id":"chatcmpl-' in rc.text
    )
    out["chat_ka_done_terminates"] = _data_frames(fc)[-1].data == "[DONE]"
    # keepalive comments consume no ``id:`` slot — the numbered sequence
    # stays contiguous for Last-Event-ID resume arithmetic
    numbered = [int(f.seq) for f in fc if f.seq is not None]
    out["chat_ka_ids_contiguous"] = numbered == list(range(len(numbered)))

    rm = ka.client.post("/v1/messages", json=_anth_body(_ECHO_PROMPT, stream=True))
    fm = _sse_frames(rm.text)
    names = [f.event for f in fm]
    ms_idx = names.index("message_start") if "message_start" in names else len(fm)
    ka_pings = [
        f for f in fm[:ms_idx] if f.event == "ping" and f.data is not None and '"ping"' in f.data
    ]
    out["msg_ka_ping_unnumbered"] = (
        rm.status_code == 200
        and bool(ka_pings)
        and all(f.seq is None for f in ka_pings)
        and all(n in ("ping",) for n in names[:ms_idx])
    )
    numbered_m = [int(f.seq) for f in fm if f.seq is not None]
    out["msg_ka_ids_contiguous"] = numbered_m == list(range(len(numbered_m)))
    out["msg_ka_cid_inband"] = "x-fx1-completion-id" not in rm.headers and '"id":"msg_' in rm.text

    rr = ka.client.post(
        "/v1/responses", json={"model": "fx1", "input": _ECHO_PROMPT, "stream": True}
    )
    fr = _sse_frames(rr.text)
    r_comments = [f for f in fr if f.comment_only]
    first_event = next((i for i, f in enumerate(fr) if not f.comment_only), len(fr))
    out["resp_ka_comments_before_events"] = (
        rr.status_code == 200
        and bool(r_comments)
        and all(f.raw.startswith(_KEEPALIVE) for f in r_comments)
        and all(i < first_event for i, f in enumerate(fr) if f.comment_only)
    )
    names_r = [f.event for f in fr]
    out["resp_ka_then_lifecycle"] = _EV_CREATED in names_r and names_r[-1] in (
        "response.completed",
        _EV_INCOMPLETE,
    )

    rl = ka.client.post(
        "/v1/completions", json={"model": "fx1", "prompt": _ECHO_PROMPT, "stream": True}
    )
    fl = _sse_frames(rl.text)
    out["legacy_ka_comments_done"] = (
        rl.status_code == 200
        and any(f.comment_only and f.raw.startswith(_KEEPALIVE) for f in fl)
        and _data_frames(fl)[-1].data == "[DONE]"
    )

    # mid-window fault: headers are already committed, so the failure is
    # an in-band frame in each leg's grammar — never a JSON 5xx.
    fk = _make_ctx(wd / "fail", _SlowFailBackend(0.4), sse_keepalive_s=0.05)
    frc = fk.client.post("/v1/chat/completions", json=_chat_body(_ECHO_PROMPT, stream=True))
    frc_j = _json_datas(_sse_frames(frc.text))
    errs = [p for p in frc_j if p.get("error")]
    out["chat_ka_fault_inband"] = (
        frc.status_code == 200
        and bool(errs)
        and errs[0]["error"].get("type") == "server_error"
        and _data_frames(_sse_frames(frc.text))[-1].data == "[DONE]"
    )
    frm = fk.client.post("/v1/messages", json=_anth_body(_ECHO_PROMPT, stream=True))
    frm_ev = _event_payloads(_sse_frames(frm.text))
    m_errs = [p for e, p in frm_ev if e == "error"]
    out["msg_ka_fault_event_error"] = (
        frm.status_code == 200
        and bool(m_errs)
        and m_errs[0].get("type") == "error"
        and "message_start" not in [e for e, _ in frm_ev]
    )
    frr = fk.client.post(
        "/v1/responses", json={"model": "fx1", "input": _ECHO_PROMPT, "stream": True}
    )
    frr_ev = [e for e, _ in _event_payloads(_sse_frames(frr.text))]
    out["resp_ka_fault_event_error"] = (
        frr.status_code == 200 and "error" in frr_ev and "response.completed" not in frr_ev
    )

    # inside the grace window the ordinary contract is byte-identical —
    # full headers, no keepalive frames, and a fault stays a JSON 5xx.
    fast = _make_ctx(wd / "fast", _StubBackend(), sse_keepalive_s=30.0)
    rf = fast.client.post("/v1/chat/completions", json=_chat_body(_ECHO_PROMPT, stream=True))
    out["v1grace_fast_unchanged"] = (
        rf.status_code == 200
        and "x-fx1-completion-id" in rf.headers
        and _KEEPALIVE not in rf.text
        and _data_frames(_sse_frames(rf.text))[-1].data == "[DONE]"
    )
    ff = _make_ctx(wd / "ff", _SlowFailBackend(0.4), sse_keepalive_s=30.0)
    fj = ff.client.post("/v1/chat/completions", json=_chat_body(_ECHO_PROMPT, stream=True))
    out["v1grace_fault_json"] = (
        fj.status_code == 502
        and not fj.headers.get("content-type", "").startswith(_SSE_CT)
        and fj.json().get("error", {}).get("type") == "server_error"
    )
    return out


# ---------------------------------------------------------------------------
# Resume — Idempotency-Key replays and Last-Event-ID cursors
# ---------------------------------------------------------------------------


def _probe_resume(ctx: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    body = _chat_body(_ECHO_PROMPT, stream=True)
    key = "sk-chat-1"
    r1 = ctx.client.post("/v1/chat/completions", json=body, headers={"Idempotency-Key": key})
    r2 = ctx.client.post("/v1/chat/completions", json=body, headers={"Idempotency-Key": key})
    out["idem_stream_byte_identical"] = r1.status_code == 200 and r1.text == r2.text
    out["idem_replay_header"] = r2.headers.get("x-fx1-idempotent-replay") == "true"
    r3 = ctx.client.post(
        "/v1/chat/completions",
        json=body,
        headers={"Idempotency-Key": key, "Last-Event-ID": "2"},
    )
    out["resume_suffix_bytes"] = (
        r1.text.endswith(r3.text) and r3.text != r1.text and r3.status_code == 200
    )
    r3_frames = _data_frames(_sse_frames(r3.text))
    out["resume_cursor_absolute"] = bool(r3_frames) and r3_frames[0].seq == "3"
    n_frames = len(_data_frames(_sse_frames(r1.text)))
    r4 = ctx.client.post(
        "/v1/chat/completions",
        json=body,
        headers={"Idempotency-Key": key, "Last-Event-ID": str(n_frames - 1)},
    )
    # skip slices the chunk list; [DONE] always terminates — past-end means
    # exactly one [DONE] frame and nothing else
    r4_frames = _data_frames(_sse_frames(r4.text))
    out["resume_past_end_only_done"] = (
        r4.status_code == 200 and len(r4_frames) == 1 and r4_frames[0].data == "[DONE]"
    )
    # keyed replay with a different body is loudly refused, not grafted
    r5 = ctx.client.post(
        "/v1/chat/completions",
        json=_chat_body(_ECHO_PROMPT),
        headers={"Idempotency-Key": key},
    )
    out["resume_key_conflict_409"] = r5.status_code == 409
    # the failure matrix — loud refusals, never silent grafting
    r_bad = ctx.client.post(
        "/v1/chat/completions", json=_chat_body(_ECHO_PROMPT), headers={"Last-Event-ID": "1"}
    )
    out["resume_nonstream_400"] = r_bad.status_code == 400
    r_mal = ctx.client.post(
        "/v1/chat/completions",
        json=body,
        headers={"Idempotency-Key": "fresh-1", "Last-Event-ID": "abc"},
    )
    out["resume_malformed_400"] = r_mal.status_code == 400
    r_neg = ctx.client.post(
        "/v1/chat/completions",
        json=body,
        headers={"Idempotency-Key": "fresh-2", "Last-Event-ID": "-1"},
    )
    out["resume_negative_400"] = r_neg.status_code == 400
    r_nokey = ctx.client.post("/v1/chat/completions", json=body, headers={"Last-Event-ID": "1"})
    out["resume_no_key_400"] = r_nokey.status_code == 400
    r_miss = ctx.client.post(
        "/v1/chat/completions",
        json=body,
        headers={"Idempotency-Key": "fresh-3", "Last-Event-ID": "0"},
    )
    out["resume_miss_409"] = r_miss.status_code == 409 and "resume_miss" in r_miss.text
    # the same contract on the other dialects
    rb = {"model": "fx1", "input": _ECHO_PROMPT, "stream": True}
    rr1 = ctx.client.post("/v1/responses", json=rb, headers={"Idempotency-Key": "sk-r"})
    rr2 = ctx.client.post(
        "/v1/responses",
        json=rb,
        headers={"Idempotency-Key": "sk-r", "Last-Event-ID": "1"},
    )
    out["resp_resume_suffix"] = rr1.text.endswith(rr2.text) and rr2.text != rr1.text
    rr_miss = ctx.client.post(
        "/v1/responses",
        json=rb,
        headers={"Idempotency-Key": "sk-r2", "Last-Event-ID": "0"},
    )
    out["resp_resume_miss_409"] = rr_miss.status_code == 409
    ab = _anth_body(_ECHO_PROMPT, stream=True)
    ra1 = ctx.client.post("/v1/messages", json=ab, headers={"Idempotency-Key": "sk-a"})
    ra2 = ctx.client.post(
        "/v1/messages",
        json=ab,
        headers={"Idempotency-Key": "sk-a", "Last-Event-ID": "1"},
    )
    out["anth_resume_suffix"] = ra1.text.endswith(ra2.text) and ra2.text != ra1.text
    ra_miss = ctx.client.post(
        "/v1/messages",
        json=ab,
        headers={"Idempotency-Key": "sk-a2", "Last-Event-ID": "0"},
    )
    out["anth_resume_miss_409"] = ra_miss.status_code == 409
    lb = {"model": "fx1", "prompt": _ECHO_PROMPT, "stream": True}
    rl1 = ctx.client.post("/v1/completions", json=lb, headers={"Idempotency-Key": "sk-l"})
    rl2 = ctx.client.post("/v1/completions", json=lb, headers={"Idempotency-Key": "sk-l"})
    rl3 = ctx.client.post(
        "/v1/completions",
        json=lb,
        headers={"Idempotency-Key": "sk-l", "Last-Event-ID": "1"},
    )
    out["legacy_idem_identical"] = rl1.text == rl2.text
    out["legacy_resume_suffix"] = rl1.text.endswith(rl3.text) and rl3.text != rl1.text
    return out


# ---------------------------------------------------------------------------
# Replay — GET /v1/responses/{id}?stream=true
# ---------------------------------------------------------------------------


def _probe_replay(ctx: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    body = {"model": "fx1", "input": _ECHO_PROMPT, "stream": True}
    r = ctx.client.post("/v1/responses", json=body)
    frames = _sse_frames(r.text)
    events = _event_payloads(frames)
    rid = events[0][1]["response"]["id"]
    n_frames = len(_data_frames(frames))

    g = ctx.client.get(f"/v1/responses/{rid}", params={"stream": "true"})
    out["replay_byte_identical"] = g.status_code == 200 and g.text == r.text
    out["replay_headers"] = bool(
        g.headers.get("x-fx1-completion-id") and g.headers.get("x-fx1-receipt-sha256")
    )
    gj = ctx.client.get(f"/v1/responses/{rid}")
    out["replay_get_json_twin"] = (
        gj.status_code == 200
        and gj.headers.get("content-type", "").startswith("application/json")
        and gj.json() == events[-1][1]["response"]
    )
    g0 = ctx.client.get(f"/v1/responses/{rid}", params={"stream": "true", "starting_after": 0})
    out["replay_skip_0_drops_first"] = g0.text == r.text.split("\n\n", 1)[1]
    g3 = ctx.client.get(f"/v1/responses/{rid}", params={"stream": "true", "starting_after": 3})
    g3_frames = _data_frames(_sse_frames(g3.text))
    out["replay_skip_cursor_absolute"] = (
        bool(g3_frames)
        and g3_frames[0].seq == "4"
        and r.text.endswith(g3.text)
        and g3.text != r.text
    )
    gb = ctx.client.get(
        f"/v1/responses/{rid}",
        params={"stream": "true", "starting_after": n_frames},
    )
    out["replay_skip_beyond_empty"] = gb.status_code == 200 and gb.text == ""
    g1 = ctx.client.get(f"/v1/responses/{rid}", params={"stream": "1"})
    out["replay_stream_flag_lenient"] = g1.status_code == 200 and g1.text == r.text
    gf = ctx.client.get(f"/v1/responses/{rid}", params={"stream": "false"})
    out["replay_stream_false_json"] = (
        gf.status_code == 200
        and gf.headers.get("content-type", "").startswith("application/json")
        and "data:" not in gf.text
    )
    out["replay_404_unknown"] = (
        ctx.client.get("/v1/responses/resp_nope", params={"stream": "true"}).status_code == 404
    )
    # store=false never enters the retrieval index — same 404 as unknown
    rs = ctx.client.post("/v1/responses", json={**body, "store": False})
    rid_ns = _event_payloads(_sse_frames(rs.text))[0][1]["response"]["id"]
    out["replay_404_store_false"] = (
        ctx.client.get(f"/v1/responses/{rid_ns}", params={"stream": "true"}).status_code == 404
    )
    # delete drops it for both projections
    ctx.client.delete(f"/v1/responses/{rid}")
    out["replay_404_after_delete"] = (
        ctx.client.get(f"/v1/responses/{rid}", params={"stream": "true"}).status_code == 404
        and ctx.client.get(f"/v1/responses/{rid}").status_code == 404
    )
    return out


def _probe_replay_live(workdir: Path) -> dict[str, bool]:
    """A background response's live-follow stream: queued prelude →
    ``: keepalive`` comments → in_progress → completed — then the
    post-hoc replay of the terminal record must be byte-identical."""
    out: dict[str, bool] = {}
    backend = _GatedBackend()
    ctx = _make_ctx(workdir / "live", backend, sse_keepalive_s=0.05)
    executor = ctx.app.state.jobs_executor
    sleeper_rel = threading.Event()
    workers = int(getattr(executor, "_max_workers", 4))
    futs = [executor.submit(lambda: sleeper_rel.wait(timeout=15.0)) for _ in range(workers)]
    time.sleep(0.15)  # every worker slot occupied → the job stays queued
    timers = [
        threading.Timer(0.3, sleeper_rel.set),  # free the executor → job starts
        threading.Timer(0.8, backend.gate.set),  # let complete() return → terminal
    ]
    try:
        env = ctx.client.post(
            "/v1/responses", json={"model": "fx1", "input": "live", "background": True}
        ).json()
        rid = env["id"]
        out["replay_live_submit_queued"] = env.get("status") == "queued"
        for t in timers:
            t.start()
        g = ctx.client.get(f"/v1/responses/{rid}", params={"stream": "true", "timeout_s": 20})
    finally:
        for t in timers:
            t.cancel()
        sleeper_rel.set()
        backend.gate.set()
        for f in futs:
            f.cancel()
    frames = _sse_frames(g.text)
    comments = [f for f in frames if f.comment_only]
    events = _event_payloads(frames)
    names = [e for e, _ in events]
    datas = _data_frames(frames)
    out["replay_live_prelude_queued"] = names[:2] == [
        _EV_CREATED,
        "response.queued",
    ]
    out["replay_live_queued_status"] = events[0][1]["response"]["status"] == "queued"
    out["replay_live_in_progress"] = "response.in_progress" in names
    out["replay_live_keepalives"] = len(comments) >= 1 and all(
        f.raw == _KEEPALIVE for f in comments
    )
    out["replay_live_completed_last"] = (
        names[-1] == "response.completed" and events[-1][1]["response"]["status"] == "completed"
    )
    out["replay_live_ids_dense"] = [f.seq for f in datas] == [str(i) for i in range(len(datas))]
    # the follow streams live state snapshots (the queued record still
    # carries the REQUEST model); post-hoc replays regenerate every frame
    # from the terminal record — the same event sequence and terminal
    # payload must come out, and two post-hoc replays must be byte-exact
    g2 = ctx.client.get(f"/v1/responses/{rid}", params={"stream": "true"})
    events2 = _event_payloads(_sse_frames(g2.text))
    out["replay_live_event_names_match"] = [e for e, _ in events2] == names
    out["replay_live_terminal_identical"] = events2[-1] == events[-1]
    g3 = ctx.client.get(f"/v1/responses/{rid}", params={"stream": "true"})
    out["replay_deterministic"] = g3.text == g2.text
    g4 = ctx.client.get(f"/v1/responses/{rid}")
    out["replay_live_record_completed"] = g4.json().get("status") == "completed"
    return out


def _probe_replay_failed(ctx: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    env = ctx.client.post(
        "/v1/responses", json={"model": "fx1", "input": "die", "background": True}
    ).json()
    rid = env["id"]
    end = time.monotonic() + 15.0
    rec: dict[str, Any] = {}
    while time.monotonic() < end:
        rec = ctx.client.get(f"/v1/responses/{rid}").json()
        if rec.get("status") == "failed":
            break
        time.sleep(0.05)
    out["replay_failed_record"] = rec.get("status") == "failed"
    g = ctx.client.get(f"/v1/responses/{rid}", params={"stream": "true"})
    events = _event_payloads(_sse_frames(g.text))
    names = [e for e, _ in events]
    term = events[-1][1]["response"]
    out["replay_failed_terminal_event"] = (
        names[-1] == "response.failed" and term["status"] == "failed"
    )
    out["replay_failed_error_field"] = isinstance(term.get("error"), dict) and bool(
        term["error"].get("message")
    )
    return out


def _probe_replay_cancel_delete(workdir: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    # --- cancelled background response → response.cancelled terminal ---
    backend = _GatedBackend()
    ctx = _make_ctx(workdir / "cancel", backend, sse_keepalive_s=0.05)
    env = ctx.client.post(
        "/v1/responses", json={"model": "fx1", "input": "c", "background": True}
    ).json()
    rid = env["id"]
    d = ctx.client.post(f"/v1/responses/{rid}/cancel")
    g = ctx.client.get(f"/v1/responses/{rid}", params={"stream": "true"})
    events = _event_payloads(_sse_frames(g.text))
    names = [e for e, _ in events]
    out["replay_cancelled_200"] = d.status_code == 200 and d.json().get("status") == "cancelled"
    out["replay_cancelled_terminal"] = (
        names[-1] == "response.cancelled" and events[-1][1]["response"]["status"] == "cancelled"
    )
    out["replay_cancelled_no_completed"] = "response.completed" not in names
    # a cancelled record replays deterministically afterwards too
    g2 = ctx.client.get(f"/v1/responses/{rid}", params={"stream": "true"})
    out["replay_cancelled_deterministic"] = g2.text == g.text
    backend.gate.set()  # drain the still-blocked worker before teardown
    # --- delete after the follow response has already been buffered ---
    backend2 = _GatedBackend()
    ctx2 = _make_ctx(workdir / "del", backend2, sse_keepalive_s=0.05)
    env2 = ctx2.client.post(
        "/v1/responses", json={"model": "fx1", "input": "d", "background": True}
    ).json()
    rid2 = env2["id"]
    lines: list[str] = []
    with ctx2.client.stream("GET", f"/v1/responses/{rid2}?stream=true&timeout_s=20") as s:
        it = s.iter_lines()
        blanks = 0
        for ln in it:  # read a prelude from the already-buffered response
            lines.append(ln)
            if ln == "":
                blanks += 1
            if blanks >= 2:
                break
        # TestClient has already consumed the ASGI response. This deletion
        # does not test termination of an active client connection.
        dr = ctx2.client.delete(f"/v1/responses/{rid2}")
        for ln in it:  # drain to stream end
            lines.append(ln)
    frames2 = _sse_frames("\n".join(lines))
    events2 = _event_payloads(frames2)
    out["replay_delete_after_buffered_follow_200"] = (
        dr.status_code == 200 and dr.json().get("deleted") is True
    )
    out["replay_buffered_follow_no_terminal"] = len(events2) >= 1 and all(
        e not in _TERMINAL_EVENTS for e, _ in events2
    )
    # releasing the worker must not resurrect the deleted record
    backend2.gate.set()
    time.sleep(0.8)
    out["replay_deleted_stays_404"] = ctx2.client.get(f"/v1/responses/{rid2}").status_code == 404
    out["replay_deleted_replay_404"] = (
        ctx2.client.get(f"/v1/responses/{rid2}", params={"stream": "true"}).status_code == 404
    )
    return out


# ---------------------------------------------------------------------------
# Buffered partial reads — no live client disconnect is exercised
# ---------------------------------------------------------------------------


def _pinned_prefix_lines(lines: Iterator[str], n: int) -> list[str]:
    """First *n* completion lines, skipping leading keepalive comments.

    A slow grace window emits ``: keepalive`` frames before the stored
    envelope. The keyed replay regenerates that envelope and does not
    repeat the comments, so they are not part of the pinned prefix.
    """
    prefix: list[str] = []
    for _ in range(64):
        try:
            line = next(lines)
        except StopIteration:
            break
        if line.startswith(":"):
            try:
                blank = next(lines)
            except StopIteration:
                break
            if blank != "":
                prefix.append(blank)
            continue
        prefix.append(line)
        if len(prefix) >= n:
            break
    return prefix


def _probe_buffered_reads(ctx: _Ctx, workdir: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    # Partial consumer read of a completed, buffered body, then a replay.
    key = "dc-1"
    body = _chat_body(_ECHO_PROMPT, stream=True)
    with ctx.client.stream(
        "POST", "/v1/chat/completions", json=body, headers={"Idempotency-Key": key}
    ) as s:
        it = s.iter_lines()
        prefix = _pinned_prefix_lines(it, 6)
        # Closing here releases a buffered response; its ASGI producer
        # already finished before TestClient exposed the first line.
    r_full = ctx.client.post("/v1/chat/completions", json=body, headers={"Idempotency-Key": key})
    pinned = "\n".join(prefix)
    out["buffered_prefix_matches_replay"] = pinned != "" and r_full.text.startswith(pinned)
    out["buffered_partial_read_record_intact"] = (
        r_full.status_code == 200
        and r_full.headers.get("x-fx1-idempotent-replay") == "true"
        and r_full.text.endswith("data: [DONE]\n\n")
        and _data_frames(_sse_frames(r_full.text))[-1].data == "[DONE]"
    )
    out["buffered_partial_read_app_responsive"] = (
        ctx.client.post("/v1/chat/completions", json=_chat_body("pong")).status_code == 200
    )
    # Read part of a buffered follow response, then inspect its record.
    backend2 = _GatedBackend()
    ctx2 = _make_ctx(workdir / "dc", backend2, sse_keepalive_s=0.05)
    env = ctx2.client.post(
        "/v1/responses", json={"model": "fx1", "input": "x", "background": True}
    ).json()
    rid = env["id"]
    with ctx2.client.stream("GET", f"/v1/responses/{rid}?stream=true&timeout_s=20") as s:
        it = s.iter_lines()
        for _ in range(7):
            try:
                next(it)
            except StopIteration:
                break
        # This closes the completed TestClient buffer, not a live ASGI
        # connection; disconnect handling remains outside this audit.
    backend2.gate.set()
    end = time.monotonic() + 15.0
    rec: dict[str, Any] = {}
    while time.monotonic() < end:
        rec = ctx2.client.get(f"/v1/responses/{rid}").json()
        if rec.get("status") == "completed":
            break
        time.sleep(0.05)
    out["buffered_follow_record_intact"] = rec.get("status") == "completed"
    g = ctx2.client.get(f"/v1/responses/{rid}", params={"stream": "true"})
    events = _event_payloads(_sse_frames(g.text))
    out["buffered_follow_replays"] = bool(events) and events[-1][0] == ("response.completed")
    out["buffered_follow_responsive"] = (
        ctx2.client.post("/v1/responses", json={"model": "fx1", "input": "pong"}).status_code == 200
    )
    return out


# ---------------------------------------------------------------------------
# Concurrency — parallel streams must never interleave
# ---------------------------------------------------------------------------


def _probe_concurrent(ctx: _Ctx, ctx_stream: _Ctx, rid_pool: str) -> dict[str, bool]:
    out: dict[str, bool] = {}
    rs = _parallel(
        _CONC_N,
        lambda i: ctx.client.post(
            "/v1/chat/completions",
            json=_chat_body(f"conc-{i} " + "fill " * 30, stream=True),
        ),
    )
    per_ok = True
    for i, r in enumerate(rs):
        frames = _sse_frames(r.text)
        datas = _data_frames(frames)
        chunks = _chat_chunks(frames)
        contents = [d["content"] for d in _chat_deltas(chunks) if "content" in d]
        per_ok = (
            per_ok
            and r.status_code == 200
            and "".join(contents) == f"stub:conc-{i} " + "fill " * 30
            and datas[-1].data == "[DONE]"
            and [f.seq for f in datas] == [str(j) for j in range(len(datas))]
            and len({c["id"] for c in chunks}) == 1
        )
    out["conc_chat_no_interleave"] = per_ok
    out["conc_chat_all_200"] = all(r.status_code == 200 for r in rs)
    hs = _parallel(
        _CONC_N,
        lambda i: ctx_stream.client.post(
            "/harness/complete/stream", json=_hstream_body(f"tag-{i}")
        ),
    )
    tok_ok = True
    for i, r in enumerate(hs):
        frames = _sse_frames(r.text)
        payloads = _json_datas(frames)
        toks = [p["content"] for p in payloads if p.get("type") == "token"]
        tok_ok = (
            tok_ok
            and toks == [f"tag-{i}:a", f"tag-{i}:b", f"tag-{i}:c"]
            and _data_frames(frames)[-1].data == "[DONE]"
        )
    out["conc_hstream_no_interleave"] = tok_ok
    out["conc_hstream_all_200"] = all(r.status_code == 200 for r in hs)
    reps = _parallel(
        3,
        lambda _i: ctx.client.get(f"/v1/responses/{rid_pool}", params={"stream": "true"}).text,
    )
    out["conc_replay_identical"] = len(set(reps)) == 1 and bool(reps[0])
    return out


# ---------------------------------------------------------------------------
# Pre-stream failures stay dialect-JSON, never SSE
# ---------------------------------------------------------------------------


def _probe_prestream(ctx_fail: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    rc = ctx_fail.client.post("/v1/chat/completions", json=_chat_body(_ECHO_PROMPT, stream=True))
    out["prestream_chat_502_json"] = (
        rc.status_code == 502
        and not rc.headers.get("content-type", "").startswith(_SSE_CT)
        and rc.json().get("error", {}).get("code") is not None
    )
    rm = ctx_fail.client.post("/v1/messages", json=_anth_body(_ECHO_PROMPT, stream=True))
    out["prestream_anth_502_json"] = (
        rm.status_code == 502 and rm.json().get("type") == "error" and "data:" not in rm.text
    )
    rr = ctx_fail.client.post(
        "/v1/responses", json={"model": "fx1", "input": _ECHO_PROMPT, "stream": True}
    )
    out["prestream_resp_502_json"] = (
        rr.status_code == 502
        and not rr.headers.get("content-type", "").startswith(_SSE_CT)
        and "data:" not in rr.text
    )
    return out


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def stream_audit() -> dict[str, Any]:
    """Run every probe against live in-process apps; literal bools out."""
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    for k in _ENV_KEYS:
        os.environ.pop(k, None)
    out: dict[str, Any] = {}
    try:
        with tempfile.TemporaryDirectory() as td:
            wd = Path(td)
            ctx = _make_ctx(wd / "a", _StubBackend())
            out.update(_probe_chat(ctx))
            out.update(_probe_chat_framing(ctx))
            out.update(_probe_chat_usage(ctx))
            out.update(_probe_chat_n2(ctx))
            out.update(_probe_completions(ctx))
            out.update(_probe_responses(ctx))
            out.update(_probe_anthropic(ctx))
            out.update(_probe_resume(ctx))
            rid_pool = _event_payloads(
                _sse_frames(
                    ctx.client.post(
                        "/v1/responses",
                        json={"model": "fx1", "input": "pool", "stream": True},
                    ).text
                )
            )[0][1]["response"]["id"]
            out.update(_probe_replay(ctx))
            out.update(_probe_replay_live(wd / "live"))
            ctx_tools = _make_ctx(wd / "b", _ToolStubBackend())
            out.update(_probe_chat_tools(ctx_tools))
            out.update(_probe_chat_logprobs(ctx_tools))
            out.update(_probe_resp_tools(ctx_tools))
            out.update(_probe_anthropic_tools(ctx_tools))
            ctx_stream = _make_ctx(wd / "c", _StreamStubBackend())
            out.update(_probe_hstream(ctx_stream))
            out.update(_probe_hstream_refusals(ctx_stream, ctx))
            ctx_gate = _make_ctx(wd / "e", _DirtyStreamBackend())
            out.update(_probe_hstream_gate(ctx_gate))
            ctx_ka = _make_ctx(wd / "f", _SlowStreamBackend(0.5), sse_keepalive_s=0.05)
            out.update(_probe_hstream_keepalive(ctx_ka))
            ctx_ka2 = _make_ctx(wd / "g", _FailStreamBackend(), sse_keepalive_s=0.05)
            out.update(_probe_hstream_miderr(ctx_ka2))
            out.update(_probe_v1_keepalive(wd / "v1ka"))
            ctx_fail = _make_ctx(wd / "h", _FailBackend())
            out.update(_probe_replay_failed(ctx_fail))
            out.update(_probe_prestream(ctx_fail))
            out.update(_probe_replay_cancel_delete(wd / "j"))
            out.update(_probe_buffered_reads(ctx, wd / "k"))
            out.update(_probe_concurrent(ctx, ctx_stream, rid_pool))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out


def stream_audit_bench() -> dict[str, Any]:
    """Sealed receipt: contract probes True, divergences named."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = stream_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "stream_audit",
        "schema": "stream_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "Starlette TestClient buffered ASGI response bodies",
            "not_verified": [
                "early client disconnect propagation",
                "deletion during an active network follow response",
                "network streaming delivery timing",
            ],
        },
        "interpretation": (
            "Buffered SSE-body fixture checks pass across the five surfaces: "
            "frames are well-formed blank-line-delimited blocks of legal SSE "
            "fields, with dense ``id:`` indices where the dialect uses them; chat/completions "
            "emit role→content→finish chunks under one completion id "
            "ending ``[DONE]``; the Responses dialect emits the full "
            "event lifecycle ending ``response.completed`` with deltas "
            "byte-identical to the non-stream twin and stored-object "
            "replay; Anthropic emits ``message_start``/``ping``/block "
            "lifecycle/``message_delta``/``message_stop`` with ``event:`` "
            "always matching ``data.type``; the harness stream emits "
            "gated token frames plus a ``final`` envelope and reports "
            "mid-stream faults as a terminal error event + ``[DONE]`` "
            "under HTTP 200; keyed replays are byte-identical, "
            "``Last-Event-ID``/``starting_after`` slice by absolute frame "
            "index with loud 400/409 refusals, background responses "
            "follow fixtures include ``: keepalive`` comments; buffered follow "
            "bodies have no terminal frame before deletion and post-deletion "
            "checks return 404. Parallel fixture bodies show no interleaving; "
            "pre-stream failures stay "
            "dialect-JSON, never SSE. This buffered transport does not verify "
            "early disconnects, deletion during an active network response, "
            "or network delivery timing."
            if ok
            else f"STREAM AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(stream_audit_bench(), indent=2, sort_keys=True))
