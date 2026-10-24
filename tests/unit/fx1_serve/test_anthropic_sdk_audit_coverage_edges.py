"""Edge arms in anthropic_sdk_audit: stub faults and probes that failed to raise."""

from __future__ import annotations

import asyncio
import os
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import pytest

from fx1.serve import anthropic_sdk_audit as audit


def _exec_main(mod: Any, bench: str) -> None:
    path = mod.__file__
    assert isinstance(path, str)
    source = Path(path).read_text(encoding="utf-8")

    def _stub_bench() -> dict[str, bool]:
        return {"ok": True}

    class _NS(dict[str, Any]):
        def __setitem__(self, key: str, value: object) -> None:
            if key == bench and callable(value):
                value = _stub_bench
            dict.__setitem__(self, key, value)

    ns = _NS()
    ns.update({"__name__": "__main__", "__file__": path, "__package__": mod.__package__})
    with redirect_stdout(StringIO()):
        exec(compile(source, path, "exec"), ns, ns)  # noqa: S102 — module __main__ line


def _status_error() -> Any:
    import anthropic

    request = httpx2.Request("GET", "http://sdk-audit/v1/messages/batches/x/results")
    response = httpx2.Response(409, request=request)
    return anthropic.APIStatusError("replay refused", response=response, body=None)


def _sanitize_create(kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(kwargs)
    messages = kwargs.get("messages")
    if messages == []:
        kwargs["messages"] = [{"role": "user", "content": "hi"}]
    elif isinstance(messages, list) and len(messages) == 1:
        content = messages[0].get("content")
        if (
            messages[0].get("role") == "assistant"
            or isinstance(content, list)
            or content in {audit._FAIL, audit._BLOCK}
        ):
            kwargs["messages"] = [{"role": "user", "content": "hi"}]
        elif content == "different":
            headers = dict(kwargs.get("extra_headers") or {})
            headers.pop("Idempotency-Key", None)
            kwargs["extra_headers"] = headers
    elif (
        isinstance(messages, list)
        and len(messages) >= 2
        and all(item.get("role") == "user" for item in messages)
    ):
        kwargs["messages"] = [{"role": "user", "content": "hi"}]
    extra = kwargs.get("extra_body")
    if isinstance(extra, dict) and (extra.get("temperature") == 1.5 or "cache_control" in extra):
        kwargs.pop("extra_body", None)
    for key in ("thinking", "service_tier", "output_config", "diagnostics"):
        kwargs.pop(key, None)
    if "tool_choice" in kwargs and not kwargs.get("tools"):
        kwargs.pop("tool_choice", None)
    stops = kwargs.get("stop_sequences")
    if isinstance(stops, list) and len(stops) > 4:
        kwargs["stop_sequences"] = ["a"]
    meta = kwargs.get("metadata")
    if (
        isinstance(meta, dict)
        and isinstance(meta.get("user_id"), str)
        and len(meta["user_id"]) > 256
    ):
        kwargs["metadata"] = {"user_id": "u"}
    system = kwargs.get("system")
    if isinstance(system, str) and len(system) > 10_000:
        kwargs["system"] = "short"
    headers = dict(kwargs.get("extra_headers") or {})
    if "Last-Event-ID" in headers and not kwargs.get("stream"):
        headers.pop("Last-Event-ID", None)
    if str(headers.get("Idempotency-Key", "")).startswith("sdk-miss"):
        headers.pop("Last-Event-ID", None)
    if headers != dict(kwargs.get("extra_headers") or {}):
        kwargs["extra_headers"] = headers
    return kwargs


def _rewrite_requests(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rewritten: list[dict[str, Any]] = []
    for index, row in enumerate(requests):
        params = dict(row.get("params") or {})
        params.pop("stream", None)
        messages = []
        for message in params.get("messages") or []:
            if message.get("content") == audit._BLOCK:
                message = {**message, "content": "hi"}
            messages.append(message)
        if messages:
            params["messages"] = messages
        custom_id = str(row.get("custom_id"))
        if custom_id in seen:
            custom_id = f"{custom_id}-{index}"
        seen.add(custom_id)
        rewritten.append({**row, "custom_id": custom_id, "params": params})
    return rewritten


def _wrap_client(cl: Any) -> None:
    import anthropic

    real_create = cl.messages.create
    real_tokens = cl.messages.count_tokens
    real_retrieve_model = cl.models.retrieve
    real_batch_create = cl.messages.batches.create
    real_batch_retrieve = cl.messages.batches.retrieve
    real_batch_results = cl.messages.batches.results
    real_post = cl.post
    real_get = cl.get
    results_seen: dict[str, int] = {}

    async def _create(**kwargs: Any) -> Any:
        try:
            return await real_create(**_sanitize_create(kwargs))
        except (
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.RateLimitError,
        ):
            return SimpleNamespace()

    async def _tokens(**kwargs: Any) -> Any:
        kwargs.pop("tools", None)
        kwargs.pop("thinking", None)
        try:
            return await real_tokens(**kwargs)
        except anthropic.InternalServerError:
            return SimpleNamespace(input_tokens=1)

    async def _model(model_id: str, **kwargs: Any) -> Any:
        if model_id == "definitely-not-a-model":
            model_id = "fx1"
        return await real_retrieve_model(model_id, **kwargs)

    async def _bcreate(**kwargs: Any) -> Any:
        rows = kwargs.get("requests")
        if isinstance(rows, list):
            kwargs = {**kwargs, "requests": _rewrite_requests(rows)}
        return await real_batch_create(**kwargs)

    def _batch_id(*args: Any, **kwargs: Any) -> str:
        if args:
            return str(args[0])
        return str(kwargs.get("message_batch_id") or kwargs.get("batch_id") or "")

    async def _bretrieve(*args: Any, **kwargs: Any) -> Any:
        batch_id = _batch_id(*args, **kwargs)
        if batch_id == "msgbatch_nonexistent":
            return SimpleNamespace(id=batch_id, type="message_batch", processing_status="ended")
        try:
            return await real_batch_retrieve(*args, **kwargs)
        except anthropic.NotFoundError:
            return SimpleNamespace(id=batch_id, type="message_batch", processing_status="ended")

    async def _bdelete(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(id=_batch_id(*args, **kwargs), type="message_batch_deleted")

    async def _bcancel(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            id=_batch_id(*args, **kwargs),
            processing_status="canceling",
            cancel_initiated_at="t",
        )

    async def _bresults(*args: Any, **kwargs: Any) -> Any:
        batch_id = _batch_id(*args, **kwargs)
        results_seen[batch_id] = results_seen.get(batch_id, 0) + 1
        if results_seen[batch_id] >= 2:
            raise _status_error()
        return await real_batch_results(*args, **kwargs)

    async def _post(path: str, **kwargs: Any) -> Any:
        body = dict(kwargs.get("body") or {})
        if "max_tokens" not in body:
            body["max_tokens"] = 8
        if body.get("temperature") == "hot":
            body.pop("temperature")
        kwargs["body"] = body
        return await real_post(path, **kwargs)

    async def _get(path: str, **kwargs: Any) -> Any:
        if isinstance(path, str) and path.endswith("/results"):
            return SimpleNamespace()
        return await real_get(path, **kwargs)

    cl.messages.create = _create
    cl.messages.count_tokens = _tokens
    cl.models.retrieve = _model
    cl.messages.batches.create = _bcreate
    cl.messages.batches.retrieve = _bretrieve
    cl.messages.batches.delete = _bdelete
    cl.messages.batches.cancel = _bcancel
    cl.messages.batches.results = _bresults
    cl.post = _post
    cl.get = _get


def test_stub_methods_fail_block_and_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit.time, "sleep", lambda _seconds: None)
    backend = audit._SdkBackend()
    with pytest.raises(RuntimeError, match="exploded"):
        backend.complete_with_tools([{"role": "user", "content": audit._FAIL}])
    done = backend.complete_with_tools(
        [{"role": "user", "content": audit._BLOCK}],
        tools=[{"function": {"name": "lookup"}}],
    )
    assert done.finish_reason == "tool_calls"
    assert list(backend.stream([{"role": "user", "content": "hi"}])) == ["stub:", "alpha"]
    assert audit._NoTokensBackend().complete([{"role": "user", "content": "z"}]) == "stub:z"


def test_wait_ended_falls_through_when_the_batch_stays_open() -> None:
    class _Batches:
        async def retrieve(self, batch_id: str) -> SimpleNamespace:
            del batch_id
            return SimpleNamespace(processing_status="in_progress")

    client = SimpleNamespace(messages=SimpleNamespace(batches=_Batches()))
    ended = asyncio.run(audit._wait_ended(client, "batch", tries=1))
    assert ended.processing_status == "in_progress"


def test_probe_false_arms_and_env_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "moon-prev")
    monkeypatch.setenv("FX1_API_RATE_LIMIT_RPS", "50")
    monkeypatch.setattr(audit.time, "sleep", lambda _seconds: None)
    real = audit._sdk_client

    @contextmanager
    def _wrapped(**kwargs: Any) -> Any:
        with real(**kwargs) as triple:
            _wrap_client(triple[0])
            yield triple

    monkeypatch.setattr(audit, "_sdk_client", _wrapped)
    out = audit.anthropic_sdk_audit()
    assert out["sdk_msg_temperature_422"] is False
    assert out["sdk_batch_results_replay"] is False
    assert out["sdk_err_401"] is False
    assert out["sdk_tokens_no_channel_501"] is False
    assert os.environ["MOONSHOT_API_KEY"] == "moon-prev"
    assert os.environ["FX1_API_RATE_LIMIT_RPS"] == "50"
    assert os.environ.get("FX1_API_KEY") is None


def test_main_prints_stubbed_bench() -> None:
    _exec_main(audit, "anthropic_sdk_audit_bench")
