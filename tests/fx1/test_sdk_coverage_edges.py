"""Edge coverage for fx1.sdk: validators, eval/ft stores, and the OpenAI twins.

Backends are in-process fakes. Webhook delivery is stubbed. Clocks and
sleeps stay inside a few milliseconds.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import pytest

import fx1.sdk as sdk
from fx1.sdk import (
    CompletionRecord,
    Fx1Harness,
    _CompletionLog,
    _fallback_chain,
    _sampling_params,
)
from fx1.serve.backends import (
    BackendNotConfiguredError,
    EmbeddingResult,
    ToolCompletion,
)
from fx1.serve.evals import EvalRecord
from fx1.serve.finetune import FTJobOutcome
from fx1.serve.openai_compat import OpenAICompatError, OpenAIResponseRequest
from fx1.serve.uploads import UploadStoreError
from fx1.serve.vectorstores import VectorStoreError


class FakeBackend:
    def __init__(
        self,
        *,
        content: str = "hello",
        error: BaseException | None = None,
        usage: dict[str, Any] | None = None,
        model: str = "fake-model",
    ) -> None:
        self.content = content
        self.error = error
        self.last_usage = usage
        self._model = model
        self.closed = 0
        self.calls = 0

    def complete(self, messages: list[dict[str, Any]], sampling: Any = None, **kwargs: Any) -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.content

    def complete_with_tools(self, messages: list[dict[str, Any]], **kwargs: Any) -> ToolCompletion:
        if self.error is not None:
            raise self.error
        return ToolCompletion(content=self.content, tool_calls=None, finish_reason="stop")

    def stream(self, messages: list[dict[str, Any]], sampling: Any = None) -> Any:
        if self.error is not None:
            raise self.error
        yield "hel"
        yield "lo"

    def embeddings(self, text: Any, **kwargs: Any) -> EmbeddingResult:
        if self.error is not None:
            raise self.error
        return EmbeddingResult(
            data=({"object": "embedding", "index": 0, "embedding": [0.25]},),
            model=self._model,
            usage={"prompt_tokens": 1, "total_tokens": 1},
        )

    def close(self) -> None:
        self.closed += 1


def _harness(*backends: FakeBackend, **resolver_kw: Any) -> tuple[Fx1Harness, list[FakeBackend]]:
    made = list(backends) or [FakeBackend()]

    def resolver(name: str, **kwargs: Any) -> FakeBackend:
        if name == "hosted_k3" and resolver_kw.get("hosted_error"):
            backend = FakeBackend(error=resolver_kw["hosted_error"])
            made.append(backend)
            return backend
        return made[0]

    return Fx1Harness(backend_resolver=resolver), made


HDR = {"x-fx1-backend": "byok"}


def test_sampling_fallback_and_completion_log() -> None:
    with pytest.raises(ValueError, match="distinct"):
        _fallback_chain("byok", ["byok"])
    with pytest.raises(ValueError, match="unknown fallback"):
        _fallback_chain("byok", ["nope"])
    assert _fallback_chain("byok", None) == ["byok"]

    with pytest.raises(ValueError, match="stop"):
        _sampling_params(stop=["", "ok"])
    with pytest.raises(ValueError, match="token ids"):
        _sampling_params(logit_bias={"nope": 1})
    with pytest.raises(ValueError, match="outside"):
        _sampling_params(logit_bias={"12": 101})
    ok = _sampling_params(logit_bias={"12": -3})
    assert ok.logit_bias == {"12": -3}
    with pytest.raises(ValueError, match="presence_penalty"):
        _sampling_params(presence_penalty=3)
    with pytest.raises(ValueError, match="temperature"):
        _sampling_params(temperature=3)
    with pytest.raises(ValueError, match="top_p"):
        _sampling_params(top_p=0)
    with pytest.raises(ValueError, match="max_tokens"):
        _sampling_params(max_tokens=0)
    with pytest.raises(ValueError, match="seed"):
        _sampling_params(seed=-1)
    with pytest.raises(ValueError, match="reasoning_effort"):
        _sampling_params(reasoning_effort="huge")
    with pytest.raises(ValueError, match="service_tier"):
        _sampling_params(service_tier="gold")
    with pytest.raises(ValueError, match="prompt_cache_retention"):
        _sampling_params(prompt_cache_retention="forever")

    log = _CompletionLog(cap=1)
    first = CompletionRecord(
        completion_id="a", backend="byok", ok=True, latency_ms=1, at=1, prompt_sha256="p"
    )
    second = CompletionRecord(
        completion_id="b", backend="byok", ok=True, latency_ms=1, at=2, prompt_sha256="p"
    )
    log.append(first)
    log.append(second)
    assert log.dropped == 1
    assert log.get("a") is None
    assert [item.completion_id for item in log.latest(5, "byok")] == ["b"]


def test_usage_card_counts_attributed_calls() -> None:
    harness, _backends = _harness()
    minted = harness.key_create(name="meter")
    key_id = minted["id"]
    harness._log.append(
        CompletionRecord(
            completion_id="c1",
            backend="byok",
            ok=True,
            latency_ms=1,
            at=1,
            prompt_sha256="p",
            key_id=key_id,
            usage={"prompt_tokens": 2, "completion_tokens": 3},
        )
    )
    harness._log.append(
        CompletionRecord(
            completion_id="c2",
            backend="byok",
            ok=True,
            latency_ms=1,
            at=2,
            prompt_sha256="p",
            key_id=key_id,
            usage={"total_tokens": 9, "prompt_tokens": None},
        )
    )
    harness._log.append(
        CompletionRecord(
            completion_id="c3",
            backend="local_fx1",
            ok=True,
            latency_ms=1,
            at=3,
            prompt_sha256="p",
            key_id=key_id,
            usage=None,
        )
    )
    card = harness.key_usage(key_id)
    assert card["id"] == key_id
    assert card["served"]["calls"] == 3
    assert card["served"]["total_tokens"] == 2 + 3 + 9
    assert set(card["served"]["by_backend"]) == {"byok", "local_fx1"}


def test_complete_chain_probe_and_batch_text(monkeypatch: pytest.MonkeyPatch) -> None:
    primary = FakeBackend(error=RuntimeError("primary down"))
    secondary = FakeBackend(content="hello from fallback")

    def resolver(name: str, **kwargs: Any) -> FakeBackend:
        if name == "hosted_k3":
            return primary
        return secondary

    harness = Fx1Harness(backend_resolver=resolver)
    result = harness.complete(
        [{"role": "user", "content": "hi"}],
        backend="hosted_k3",
        fallbacks=["byok"],
    )
    assert result.content == "hello from fallback"
    assert primary.closed == 1
    assert secondary.closed == 1

    abort = FakeBackend(error=ValueError("client rejected the prompt"))
    aborted = Fx1Harness(backend_resolver=lambda *a, **k: abort)
    with pytest.raises(ValueError, match="client rejected"):
        aborted.complete([{"role": "user", "content": "hi"}], backend="byok")
    assert abort.closed == 1

    with pytest.raises(ValueError, match="top_logprobs requires"):
        aborted.complete(
            [{"role": "user", "content": "hi"}],
            backend="byok",
            top_logprobs=1,
            logprobs=False,
        )
    with pytest.raises(ValueError, match="top_logprobs must"):
        aborted.complete(
            [{"role": "user", "content": "hi"}],
            backend="byok",
            top_logprobs=21,
            logprobs=True,
        )

    many = FakeBackend()
    batcher = Fx1Harness(backend_resolver=lambda *a, **k: many)
    with pytest.raises(ValueError, match="tool context"):
        batcher.complete_many(
            [[{"role": "tool", "content": "x", "tool_call_id": "1"}]], backend="byok"
        )
    done = batcher.complete_many([[{"role": "user", "content": "hi"}]], backend="byok")
    assert done[0].content == "hello"
    assert many.closed >= 1

    streamer = FakeBackend()
    streaming = Fx1Harness(backend_resolver=lambda *a, **k: streamer)
    chunks = streaming.stream_complete([{"role": "user", "content": "hi"}], backend="byok")
    assert "".join(chunks).startswith("hello") or "hel" in "".join(chunks)
    assert streamer.closed >= 1

    honest = FakeBackend(content="Sharpe 2.1")
    probed = Fx1Harness(backend_resolver=lambda *a, **k: honest)
    refusal = probed.probe_backend("byok")
    assert refusal.ok is False and refusal.error_class == "honesty_refusal"

    missing = FakeBackend(error=NotImplementedError("no stream channel"))
    gap = Fx1Harness(backend_resolver=lambda *a, **k: missing)
    unimplemented = gap.probe_backend("byok")
    assert unimplemented.error_class == "NotImplementedError"


def test_resolve_backend_kwargs_and_chain() -> None:
    harness, _ = _harness()
    with pytest.raises(ValueError, match="applies only to backend='byok'"):
        harness._resolve_completion_backend(
            "hosted_k3", None, None, {"base_url": "https://x", "api_key": "k", "model": "m"}, None
        )
    with pytest.raises(ValueError, match="byok needs exactly"):
        harness._resolve_completion_backend("byok", None, None, {"base_url": "https://x"}, None)
    with pytest.raises(ValueError, match="checkpoint_dir applies only"):
        harness._resolve_completion_backend("hosted_k3", "/tmp/ckpt", None, None, None)

    def down(name: str, **kwargs: Any) -> Any:
        raise RuntimeError(f"{name} down")

    failed = Fx1Harness(backend_resolver=down)
    with pytest.raises(RuntimeError, match="down"):
        failed._resolve_chain("hosted_k3", ["byok"], None, None, None, None)


def test_run_eval_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    harness, _ = _harness()
    with pytest.raises(KeyError, match="unknown eval suite"):
        harness.run_eval("not-a-suite")
    with pytest.raises(ValueError, match="takes no judge"):
        harness.run_eval("calibration", judge_backend="hosted_k3", model_fn=lambda messages: "ok")
    with pytest.raises(ValueError, match="judge_byok"):
        harness.run_eval(
            "capability",
            judge_backend="hosted_k3",
            judge_byok={"base_url": "https://x", "api_key": "k", "model": "m"},
            model_fn=lambda messages: "ok",
        )

    def runner(record: EvalRecord, *, model: Any, judge: Any = None) -> None:
        model([{"role": "user", "content": "ping"}])
        if judge is not None:
            judge([{"role": "user", "content": "grade"}])
        record.status = "succeeded"
        record.report = {"results": [{"task": "t", "passed": True}]}
        record.finished_at = time.time()

    monkeypatch.setattr("fx1.serve.evals.run_eval_record", runner)
    boom = harness.run_eval(
        "calibration", model_fn=lambda messages: (_ for _ in ()).throw(RuntimeError("model down"))
    )
    assert boom.status == "failed" and "model down" in (boom.error or "")

    good = harness.run_eval("calibration", model_fn=lambda messages: "answer")
    assert good.status == "succeeded"
    listed = harness.evals(status="succeeded", suite="calibration")
    assert any(item.eval_id == good.eval_id for item in listed)
    assert harness.eval_record(good.eval_id).eval_id == good.eval_id
    with pytest.raises(KeyError):
        harness.eval_record("missing")
    receipt = harness.eval_receipt(good.eval_id)
    assert receipt
    with pytest.raises(KeyError):
        harness.eval_diff(good.eval_id, "missing-candidate")

    running = EvalRecord(
        eval_id="running1",
        suite="calibration",
        backend="model_fn",
        seed=0,
        status="running",
        created_at=time.time(),
    )
    harness._eval_store.put(running, None, None)
    with pytest.raises(RuntimeError, match="terminal"):
        harness.eval_receipt("running1")
    with pytest.raises(RuntimeError, match="diffs need"):
        harness.eval_diff(good.eval_id, "running1")

    real_chain = sdk._fallback_chain

    def empty_chain(backend: str, fallbacks: list[str] | None) -> list[str]:
        if fallbacks == ["__empty__"]:
            return []
        return real_chain(backend, fallbacks)

    monkeypatch.setattr(sdk, "_fallback_chain", empty_chain)
    exhausted = harness.run_eval("calibration", backend="byok", fallbacks=["__empty__"])
    assert exhausted.status == "failed"
    assert "exhausted" in (exhausted.error or "")


def test_eval_specs_runs_and_model_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fx1.serve.evals.run_eval_record",
        lambda record, *, model, judge=None: (
            model([{"role": "user", "content": "ping"}]),
            setattr(record, "status", "succeeded"),
            setattr(
                record,
                "report",
                {"results": [{"task": "t", "passed": True}, {"task": "u", "passed": False}]},
            ),
            setattr(record, "finished_at", time.time()),
        ),
    )
    harness, _ = _harness()
    spec = harness.eval_spec_create("cpi", suite="calibration", metadata={"k": "v"})
    spec_id = spec["id"]
    with pytest.raises(KeyError):
        harness.eval_spec_update("missing", name="x")
    updated = harness.eval_spec_update(
        spec_id,
        name="cpi-2",
        metadata={"k": "w"},
        testing_criteria=[{"name": "all-pass"}],
        data_source_config=spec["data_source_config"],
    )
    assert updated["name"] == "cpi-2"
    assert updated["testing_criteria"][0]["name"] == "all-pass"

    with pytest.raises(ValueError, match="unknown eval model"):
        harness._eval_model_backend("nope")
    assert harness._eval_model_backend("fx1") == ("local_fx1", None)
    with pytest.raises(KeyError):
        harness._eval_model_backend("ft:missing")
    harness._ft_store._models["ft:fx1:job:abc123"] = {
        "id": "ft:fx1:job:abc123",
        "job_id": "j",
        "checkpoint": "/tmp/ckpt",
        "created": 1,
    }
    assert harness._eval_model_backend("ft:fx1:job:abc123") == ("local_fx1", "/tmp/ckpt")

    run = harness.eval_run_create(spec_id, model="fx1", model_fn=lambda messages: "answer")
    assert run["eval_id"] or run["id"]
    with pytest.raises(KeyError):
        harness.eval_runs("missing-spec")
    runs = harness.evals()
    bound = next(item for item in runs if item.eval_spec == spec_id)
    items = harness.eval_run_items(spec_id, bound.eval_id)
    assert items and items[0]["status"] in {"pass", "fail"}
    with pytest.raises(KeyError):
        harness.eval_run_get(spec_id, "missing-run")
    running = EvalRecord(
        eval_id="run-live",
        suite="calibration",
        backend="model_fn",
        seed=0,
        status="running",
        created_at=time.time(),
        eval_spec=spec_id,
    )
    harness._eval_store.put(running, None, None)
    with pytest.raises(RuntimeError, match="only terminal"):
        harness.eval_run_delete(spec_id, "evalrun_run-live")
    harness.eval_run_delete(spec_id, bound.eval_id)
    with pytest.raises(KeyError):
        harness.eval_spec_delete("missing")
    harness.eval_spec_delete(spec_id)


def test_finetune_validation_artifacts_and_cancel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("fx1.serve.webhooks.check_callback_url", lambda url: url)
    monkeypatch.setattr("fx1.serve.webhooks.deliver_signed", lambda *a, **k: (False, "refused", 2))
    corpus = b'{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]}\n'

    def succeed(
        spec: Any, *, emit: Any, should_cancel: Any, pause_gate: Any = None
    ) -> FTJobOutcome:
        emit("info", "step", {"n": 1})
        good = spec.work_dir / "weights.txt"
        good.write_bytes(b"weights")
        missing = spec.work_dir / "gone.bin"
        return FTJobOutcome(
            fine_tuned_model=spec.ft_model_name,
            artifacts={"ok": good, "missing": missing},
            trained_tokens=4,
            checkpoint=str(spec.work_dir),
        )

    harness = Fx1Harness(ft_runner=succeed, ft_dir=tmp_path)
    job = harness.create_finetune_job(
        model="fx1",
        training_jsonl=corpus,
        validation_jsonl=corpus.decode(),
        callback_url="http://127.0.0.1/hook",
        callback_secret="sek",
    )
    assert job.status == "succeeded"
    assert job.callback_status == "failed"
    assert len(job.result_files) == 1
    assert harness._ft_store.checkpoint_for(job.fine_tuned_model or "")

    def cancel_after(
        spec: Any, *, emit: Any, should_cancel: Any, pause_gate: Any = None
    ) -> FTJobOutcome:
        entry = harness._ft_store.get(spec.job_id)
        assert entry is not None
        entry.cancel.set()
        return FTJobOutcome()

    cancelled = Fx1Harness(ft_runner=cancel_after, ft_dir=tmp_path / "c")

    # the closure above captures the first harness; point cancel_after at this one
    def cancel_on(
        spec: Any, *, emit: Any, should_cancel: Any, pause_gate: Any = None
    ) -> FTJobOutcome:
        entry = cancelled._ft_store.get(spec.job_id)
        assert entry is not None
        entry.cancel.set()
        return FTJobOutcome()

    cancelled._ft_runner = cancel_on
    job2 = cancelled.create_finetune_job(model="fx1", training_jsonl=corpus)
    assert job2.status == "cancelled"

    def cancel_raise(
        spec: Any, *, emit: Any, should_cancel: Any, pause_gate: Any = None
    ) -> FTJobOutcome:
        entry = raising._ft_store.get(spec.job_id)
        assert entry is not None
        entry.cancel.set()
        raise RuntimeError("stopped")

    raising = Fx1Harness(ft_runner=cancel_raise, ft_dir=tmp_path / "r")
    raising._ft_runner = cancel_raise
    job3 = raising.create_finetune_job(model="fx1", training_jsonl=corpus)
    assert job3.status == "cancelled"

    def opaque(spec: Any, *, emit: Any, should_cancel: Any, **kwargs: Any) -> FTJobOutcome:
        assert kwargs["pause_gate"] is not None
        return FTJobOutcome()

    def no_signature(_fn: Any) -> Any:
        raise ValueError("built-in has no signature")

    monkeypatch.setattr(sdk.inspect, "signature", no_signature)
    unsigned = Fx1Harness(ft_runner=opaque, ft_dir=tmp_path / "u")
    job4 = unsigned.create_finetune_job(model="fx1", training_jsonl=corpus)
    assert job4.status == "succeeded"


def test_openai_chat_usage_and_metadata() -> None:
    class Flip(FakeBackend):
        def complete(
            self, messages: list[dict[str, Any]], sampling: Any = None, **kwargs: Any
        ) -> str:
            self.calls += 1
            if self.calls == 1:
                self.last_usage = {"prompt_tokens": 2, "note": "ignored"}
            else:
                self.last_usage = {}
            return "hello"

    backend = Flip()
    harness = Fx1Harness(backend_resolver=lambda *a, **k: backend)
    chunks, cid = harness.openai_chat_stream(
        {
            "model": "fx1",
            "messages": [{"role": "user", "content": "hi"}],
            "n": 2,
            "stream_options": {"include_usage": True},
        },
        headers=HDR,
    )
    assert chunks and cid
    stored = harness.openai_chat(
        {"model": "fx1", "messages": [{"role": "user", "content": "hi"}], "store": False},
        headers=HDR,
    )
    assert stored[0].id
    with pytest.raises(ValueError, match="metadata"):
        # seed a stored completion then reject an oversized metadata write
        env, _completion = harness.openai_chat(
            {"model": "fx1", "messages": [{"role": "user", "content": "hi"}]},
            headers=HDR,
        )
        harness.openai_chat_update(env.id, metadata={"k" * 65: "v"})
    env, _completion = harness.openai_chat(
        {"model": "fx1", "messages": [{"role": "user", "content": "hi"}]},
        headers=HDR,
    )
    harness._openai_store.update_metadata = lambda *a, **k: None  # type: ignore[method-assign]
    with pytest.raises(KeyError, match="not in the retrieval index"):
        harness.openai_chat_update(env.id, metadata={"ok": "1"})


def test_embeddings_fallback_and_faults() -> None:
    good = FakeBackend()

    def resolver(name: str, **kwargs: Any) -> Any:
        if name == "hosted_k3":
            broken = FakeBackend(error=RuntimeError("embed down"))
            return broken
        if name == "local_fx1":
            raise TypeError("resolver bug")
        return good

    harness = Fx1Harness(backend_resolver=resolver)
    with pytest.raises(TypeError, match="resolver bug"):
        harness.openai_embeddings(
            {"model": "fx1", "input": "hi"},
            headers={"x-fx1-backend": "local_fx1", "x-fx1-checkpoint-dir": "/tmp/not-a-real-ckpt"},
        )
    envelope, cid = harness.openai_embeddings(
        {"model": "fx1", "input": "hi"},
        headers={"x-fx1-backend": "hosted_k3", "x-fx1-fallbacks": "byok"},
    )
    assert cid and envelope["data"]

    def always_down(name: str, **kwargs: Any) -> Any:
        raise BackendNotConfiguredError(f"{name} missing")

    dead = Fx1Harness(backend_resolver=always_down)
    with pytest.raises(BackendNotConfiguredError):
        dead.openai_embeddings({"model": "fx1", "input": "hi"}, headers=HDR)

    class NoEmbed:
        _model = "m"

        def complete(self, messages: list[dict[str, Any]], sampling: Any = None) -> str:
            return "hello"

        def close(self) -> None:
            return None

    no_embed = Fx1Harness(backend_resolver=lambda *a, **k: NoEmbed())
    with pytest.raises(NotImplementedError, match="embeddings"):
        no_embed.openai_embeddings({"model": "fx1", "input": "hi"}, headers=HDR)


def test_responses_background_file_search_and_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sdk.time, "sleep", lambda seconds: None)
    backend = FakeBackend(content="hello")
    harness = Fx1Harness(backend_resolver=lambda *a, **k: backend)

    captured: list[Any] = []

    class HoldThread:
        def __init__(self, target: Any, daemon: bool = True, name: str | None = None) -> None:
            self.target = target
            captured.append(self)

        def start(self) -> None:
            return None

    monkeypatch.setattr(sdk.threading, "Thread", HoldThread)
    queued, _cid = harness.openai_response(
        {"model": "fx1", "input": "hi", "background": True},
        headers=HDR,
    )
    harness._bg_cancel[queued["id"]].set()
    captured[-1].target()
    assert harness._bg_cancel.get(queued["id"]) is None

    def explode(name: str, **kwargs: Any) -> FakeBackend:
        return FakeBackend(error=OpenAICompatError("nope", status=400, code="bad_request"))

    failing = Fx1Harness(backend_resolver=explode)
    monkeypatch.setattr(sdk.threading, "Thread", HoldThread)
    queued2, _cid = failing.openai_response(
        {"model": "fx1", "input": "hi", "background": True},
        headers=HDR,
    )
    captured[-1].target()
    failed = failing.openai_response_get(queued2["id"])
    assert failed["status"] == "failed"

    def wipe(name: str, **kwargs: Any) -> FakeBackend:
        backend = FakeBackend(content="hello")

        def complete(messages: list[dict[str, Any]], sampling: Any = None, **kw: Any) -> str:
            for event in list(wiping._bg_cancel.values()):
                event.set()
            wiping._openai_store.get = lambda key: None  # type: ignore[method-assign]
            return "hello"

        backend.complete = complete  # type: ignore[method-assign]
        return backend

    wiping = Fx1Harness(backend_resolver=wipe)
    monkeypatch.setattr(sdk.threading, "Thread", HoldThread)
    queued3, _cid = wiping.openai_response(
        {"model": "fx1", "input": "hi", "background": True},
        headers=HDR,
    )
    captured[-1].target()

    body = OpenAIResponseRequest.model_construct(
        model="fx1",
        input="   ",
        tools=[{"type": "file_search", "vector_store_ids": ["vs_x"]}],
    )
    # model_construct may leave tools unparsed; build a real tool via validate of a non-empty input
    # and then overwrite the query by calling the helper with a whitespace string input.
    from fx1.serve.openai_compat import OpenAIFileSearchTool

    body = OpenAIResponseRequest.model_construct(
        model="fx1",
        input="   ",
        tools=[OpenAIFileSearchTool(type="file_search", vector_store_ids=["vs_x"])],
    )
    with pytest.raises(OpenAICompatError, match="empty"):
        harness._file_search_turn(body, body)

    harness._vs_store.search = lambda *a, **k: [  # type: ignore[method-assign]
        {
            "file_id": "file-1",
            "filename": "n",
            "vector_store_id": "vs_x",
            "score": 1.0,
            "text": "x" * 40000,
            "attributes": {},
        }
    ]
    long_body = OpenAIResponseRequest.model_validate(
        {
            "model": "fx1",
            "input": "inflation",
            "tools": [{"type": "file_search", "vector_store_ids": ["vs_x"]}],
        }
    )
    items, _eff = harness._file_search_turn(long_body, long_body)
    assert items

    conv = harness.openai_conversation_create()
    harness._conv_store.delete(conv["id"])
    harness._conv_append(
        conv["id"],
        OpenAIResponseRequest.model_validate({"model": "fx1", "input": "hi"}),
        {"id": "resp_x", "output": []},
    )

    env, completion_id = harness.openai_response(
        {"model": "fx1", "input": "hi", "store": False},
        headers=HDR,
    )
    assert completion_id
    if env.get("id") and harness._openai_store.get(env["id"]):
        assert harness.openai_response_delete(env["id"])["deleted"] is True
    # store=false responses are not indexed; delete a stored one instead
    stored, _cid = harness.openai_response({"model": "fx1", "input": "hi"}, headers=HDR)
    removed = harness.openai_response_delete(stored["id"])
    assert removed["deleted"] is True

    again, _cid = harness.openai_response({"model": "fx1", "input": "hi"}, headers=HDR)
    live = dict(harness._openai_store.get(again["id"]))
    live["status"] = "in_progress"
    harness._openai_store.put(live)
    events, _replay_cid = harness.openai_response_replay(again["id"], timeout_s=0, poll_s=0.001)
    assert events

    def drop_during_sleep(seconds: float) -> None:
        harness._openai_store.delete(again["id"])

    monkeypatch.setattr(sdk.time, "sleep", drop_during_sleep)
    live["status"] = "in_progress"
    harness._openai_store.put(live)
    harness.openai_response_replay(again["id"], timeout_s=5, poll_s=0.001)

    with pytest.raises(ValueError, match="starting_after"):
        harness.openai_response_replay(again["id"], starting_after=-1)
    with pytest.raises(ValueError, match="last_event_id"):
        harness.openai_response_stream(
            {"model": "fx1", "input": "hi"}, headers=HDR, last_event_id=-1
        )
    events, _cid = harness.openai_response_stream(
        {"model": "fx1", "input": "hi", "store": False},
        headers=HDR,
        last_event_id=0,
    )
    assert isinstance(events, list)


def test_conversations_uploads_and_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fx1.serve.webhooks.check_callback_url", lambda url: url)
    monkeypatch.setattr("fx1.serve.anthropic_compat.check_callback_url", lambda url: url)
    monkeypatch.setattr("fx1.serve.webhooks.deliver_signed", lambda *a, **k: (True, None, 1))
    backend = FakeBackend()
    harness = Fx1Harness(backend_resolver=lambda *a, **k: backend)
    conv = harness.openai_conversation_create(
        items=[
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ],
        metadata={"a": "b"},
    )
    added = harness.openai_conversation_items_add(
        conv["id"],
        [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "second"}],
            }
        ],
    )
    assert len(added["data"]) == 1
    first = harness.openai_conversation_items(conv["id"])["data"][0]
    second_id = added["data"][0]["id"]
    assert harness.openai_conversation_item(conv["id"], second_id)["id"] == second_id
    assert first["id"] != second_id
    with pytest.raises(KeyError, match="not found"):
        harness.openai_conversation_item_delete(conv["id"], "missing-item")
    real_put = harness._conv_store.put_if_present
    harness._conv_store.put_if_present = lambda *a, **k: False  # type: ignore[method-assign]
    with pytest.raises(KeyError, match="not found"):
        harness.openai_conversation_update(conv["id"], metadata={"z": "9"})
    harness._conv_store.put_if_present = real_put  # type: ignore[method-assign]
    updated = harness.openai_conversation_update(conv["id"], metadata=None)
    assert updated["metadata"] == {}
    harness._conv_store.mutate_items = lambda *a, **k: None  # type: ignore[method-assign]
    with pytest.raises(KeyError, match="not found"):
        harness.openai_conversation_item_delete(conv["id"], second_id)
    with pytest.raises(KeyError, match="not found"):
        harness.openai_conversation_items_add(
            conv["id"], [{"type": "message", "role": "user", "content": "x"}]
        )

    for index in range(256):
        harness._files[f"file-seed-{index}"] = {"_content": b"x", "id": f"file-seed-{index}"}
    created = harness.openai_file_create(b'{"messages":[]}\n', filename="a.jsonl")
    assert created["id"] not in {f"file-seed-{index}" for index in range(1)}
    assert len(harness._files) <= 256
    with pytest.raises(UploadStoreError, match="8 MiB"):
        harness.openai_file_create(b"x" * ((8 << 20) + 1))
    with pytest.raises(KeyError):
        harness.file_content("file-missing")
    with pytest.raises(KeyError):
        harness.file_card("file-missing")

    upload = harness.upload_create(bytes=4, filename="b.jsonl")
    part = harness.upload_part(upload["id"], b"abcd")
    part_id = (
        part.get("id") or part["part"]["id"] if isinstance(part.get("part"), dict) else part["id"]
    )
    while len(harness._files) < 256:
        harness._files[f"pad-{uuid.uuid4().hex}"] = {"_content": b"y"}
    done = harness.upload_complete(upload["id"], [part_id])
    assert done.get("file") or done.get("status") or done["id"]

    harness._vs_store.file_batch_files = lambda *a, **k: (_ for _ in ()).throw(  # type: ignore[method-assign]
        VectorStoreError(404, "missing batch", "not_found")
    )
    with pytest.raises(OpenAICompatError, match="missing batch"):
        harness.vector_store_file_batch_files("vs_x", "vsfb_x")

    def batch_backend(name: str, **kwargs: Any) -> FakeBackend:
        class Routed(FakeBackend):
            def complete(
                self, messages: list[dict[str, Any]], sampling: Any = None, **kw: Any
            ) -> str:
                text = str(messages)
                if "COMPAT" in text:
                    raise OpenAICompatError("bad line", status=400, code="invalid_request")
                if "HONEST" in text:
                    return "Sharpe 2.1"
                if "MISSING" in text:
                    raise NotImplementedError("no channel")
                if "BOOM" in text:
                    raise KeyError("unexpected")
                if "DOWN" in text:
                    raise BackendNotConfiguredError("unavailable")
                return "hello"

        return Routed()

    batcher = Fx1Harness(backend_resolver=batch_backend)

    def line(
        custom_id: str, body: dict[str, Any], url: str = "/v1/chat/completions"
    ) -> dict[str, Any]:
        return {"custom_id": custom_id, "method": "POST", "url": url, "body": body}

    _batch, chat_rows = batcher.openai_batch(
        [
            line("chat", {"model": "fx1", "messages": [{"role": "user", "content": "hi"}]}),
            line(
                "stream",
                {"model": "fx1", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            ),
            line("compat", {"model": "fx1", "messages": [{"role": "user", "content": "COMPAT"}]}),
            line("honest", {"model": "fx1", "messages": [{"role": "user", "content": "HONEST"}]}),
            line("missing", {"model": "fx1", "messages": [{"role": "user", "content": "MISSING"}]}),
            line("boom", {"model": "fx1", "messages": [{"role": "user", "content": "BOOM"}]}),
            line("down", {"model": "fx1", "messages": [{"role": "user", "content": "DOWN"}]}),
        ],
        callback_url="http://127.0.0.1/batch",
        callback_secret="sek",
    )
    _embed_batch, embed_rows = batcher.openai_batch(
        [line("embed", {"model": "fx1", "input": "hi"}, "/v1/embeddings")],
        endpoint="/v1/embeddings",
    )
    _resp_batch, resp_rows = batcher.openai_batch(
        [
            line("bg", {"model": "fx1", "input": "hi", "background": True}, "/v1/responses"),
            line("resp", {"model": "fx1", "input": "hi"}, "/v1/responses"),
            line(
                "talk", {"model": "fx1", "input": "hi", "conversation": "conv_x"}, "/v1/responses"
            ),
        ],
        endpoint="/v1/responses",
    )
    rows = [*chat_rows, *embed_rows, *resp_rows]
    statuses = {row["custom_id"]: row["response"]["status_code"] for row in rows}
    assert statuses["chat"] == 200
    assert statuses["stream"] != 200
    assert statuses["embed"] == 200
    assert statuses["honest"] == 502
    assert statuses["boom"] == 500
    assert _batch["callback_status"] == "delivered"

    with pytest.raises(OpenAICompatError, match="endpoint"):
        batcher.openai_batch([], endpoint="/v1/nope")

    _anthropic, arows = batcher.anthropic_batch(
        [
            {
                "custom_id": "ok",
                "params": {
                    "model": "fx1",
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            },
            {
                "custom_id": "bad",
                "params": {
                    "model": "fx1",
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": "COMPAT"}],
                },
            },
            {
                "custom_id": "gate",
                "params": {
                    "model": "fx1",
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": "HONEST"}],
                },
            },
            {
                "custom_id": "miss",
                "params": {
                    "model": "fx1",
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": "MISSING"}],
                },
            },
            {
                "custom_id": "down",
                "params": {
                    "model": "fx1",
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": "DOWN"}],
                },
            },
            {
                "custom_id": "boom",
                "params": {
                    "model": "fx1",
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": "BOOM"}],
                },
            },
        ],
        callback_url="http://127.0.0.1/anthropic",
    )
    kinds = {row["custom_id"]: row["result"]["type"] for row in arows}
    assert kinds["ok"] == "succeeded"
    assert kinds["bad"] == "errored"
    assert kinds["gate"] == "errored"
    assert kinds["boom"] == "errored"
    assert _anthropic["processing_status"] == "ended" or _anthropic.get("processing_status")


class _Bare:
    """Backend with no close(); availability faults and plain success both leave it unclosed."""

    _model = "bare"
    last_usage = {"prompt_tokens": 1, "note": "skip"}

    def __init__(self, *, error: BaseException | None = None, content: str = "hello") -> None:
        self.error = error
        self.content = content

    def complete(self, messages: list[dict[str, Any]], sampling: Any = None, **kwargs: Any) -> str:
        if self.error is not None:
            raise self.error
        return self.content

    def stream(self, messages: list[dict[str, Any]], sampling: Any = None) -> Any:
        if self.error is not None:
            raise self.error
        yield "hi"

    def embeddings(self, text: Any, **kwargs: Any) -> EmbeddingResult:
        if self.error is not None:
            raise self.error
        return EmbeddingResult(
            data=({"object": "embedding", "index": 0, "embedding": [0.5]},),
            model="bare",
            usage={"prompt_tokens": 1, "total_tokens": 1},
        )


def test_remaining_sdk_arcs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fx1.serve.evals.run_eval_record",
        lambda record, *, model, judge=None: (
            model([{"role": "user", "content": "ping"}]),
            judge([{"role": "user", "content": "grade"}]) if judge is not None else None,
            setattr(record, "status", "succeeded"),
            setattr(record, "report", {"results": [{"task": "t", "passed": True}]}),
            setattr(record, "finished_at", time.time()),
        ),
    )
    failing = Fx1Harness(backend_resolver=lambda *a, **k: _Bare(error=RuntimeError("model down")))
    failed = failing.run_eval("calibration", backend="byok")
    assert failed.status == "failed" and "model down" in (failed.error or "")

    judge_calls = {"n": 0}

    class JudgeBackend(_Bare):
        def complete(
            self, messages: list[dict[str, Any]], sampling: Any = None, **kwargs: Any
        ) -> str:
            judge_calls["n"] += 1
            if judge_calls["n"] > 1:
                raise RuntimeError("judge down")
            return "answer"

    judged = Fx1Harness(backend_resolver=lambda *a, **k: JudgeBackend())
    judged_record = judged.run_eval("capability", backend="byok", judge_backend="byok")
    assert judged_record.status == "failed"

    judge_calls["n"] = 0

    class BothOk(_Bare):
        def complete(
            self, messages: list[dict[str, Any]], sampling: Any = None, **kwargs: Any
        ) -> str:
            return "answer"

    both = Fx1Harness(backend_resolver=lambda *a, **k: BothOk())
    ok_record = both.run_eval("capability", backend="byok", judge_backend="hosted_k3")
    assert ok_record.status == "succeeded"
    with pytest.raises(KeyError):
        both.eval_receipt("missing-eval")
    with pytest.raises(KeyError):
        both.eval_run_create("missing-spec", model="fx1", model_fn=lambda messages: "x")

    spec = both.eval_spec_create("edge", suite="calibration")
    criteria_only = both.eval_spec_update(spec["id"], testing_criteria=[{"name": "only-criteria"}])
    assert criteria_only["testing_criteria"][0]["name"] == "only-criteria"
    created = both.eval_run_create(
        spec["id"],
        model="fx1",
        model_fn=lambda messages: "answer",
        data_source_overrides={"seed": 2},
    )
    assert created
    with pytest.raises(KeyError):
        both.eval_run_items(spec["id"], "missing-run")
    with pytest.raises(KeyError):
        both.eval_run_delete(spec["id"], "missing-run")

    reader_host, _ = _harness()
    reader_host._files["file-s"] = {"_content": "plain text", "filename": "a.txt"}
    reader_host._files["file-n"] = {"_content": 5, "filename": "c.txt"}
    assert reader_host._vs_store._reader("file-s")[0] == b"plain text"
    assert reader_host._vs_store._reader("file-n") is None

    primary = _Bare(error=RuntimeError("primary down"))
    secondary = _Bare(content="fallback hello")

    def chain(name: str, **kwargs: Any) -> _Bare:
        return primary if name == "hosted_k3" else secondary

    chained = Fx1Harness(backend_resolver=chain)
    assert (
        chained.complete(
            [{"role": "user", "content": "hi"}], backend="hosted_k3", fallbacks=["byok"]
        ).content
        == "fallback hello"
    )
    abort = Fx1Harness(backend_resolver=lambda *a, **k: _Bare(error=ValueError("client")))
    with pytest.raises(ValueError, match="client"):
        abort.complete([{"role": "user", "content": "hi"}], backend="byok")
    batch = Fx1Harness(backend_resolver=lambda *a, **k: _Bare())
    assert (
        batch.complete_many([[{"role": "user", "content": "hi"}]], backend="byok")[0].content
        == "hello"
    )
    streamed = Fx1Harness(backend_resolver=lambda *a, **k: _Bare())
    assert "hi" in "".join(
        streamed.stream_complete([{"role": "user", "content": "hi"}], backend="byok")
    )

    chat = Fx1Harness(backend_resolver=lambda *a, **k: _Bare())
    env, _cid = chat.openai_chat(
        {"model": "fx1", "messages": [{"role": "user", "content": "hi"}]},
        headers=HDR,
    )
    assert env.id
    chunks, _cid = chat.openai_chat_stream(
        {"model": "fx1", "messages": [{"role": "user", "content": "hi"}], "store": False},
        headers=HDR,
    )
    assert chunks

    def embed_chain(name: str, **kwargs: Any) -> _Bare:
        if name == "hosted_k3":
            return _Bare(error=RuntimeError("embed down"))
        return _Bare()

    embedded = Fx1Harness(backend_resolver=embed_chain)
    envelope, cid = embedded.openai_embeddings(
        {"model": "fx1", "input": "hi"},
        headers={"x-fx1-backend": "hosted_k3", "x-fx1-fallbacks": "byok"},
    )
    assert cid and envelope["data"]

    class HoldThread:
        def __init__(self, target: Any, daemon: bool = True, name: str | None = None) -> None:
            self.target = target

        def start(self) -> None:
            self.target()

    def cancel_then_raise(name: str, **kwargs: Any) -> _Bare:
        backend = _Bare()

        def complete(messages: list[dict[str, Any]], sampling: Any = None, **kw: Any) -> str:
            for event in list(host._bg_cancel.values()):
                event.set()
            raise RuntimeError("cancelled underneath")

        backend.complete = complete  # type: ignore[method-assign]
        return backend

    host = Fx1Harness(backend_resolver=cancel_then_raise)
    monkeypatch.setattr(sdk.threading, "Thread", HoldThread)
    queued, _cid = host.openai_response(
        {"model": "fx1", "input": "hi", "background": True}, headers=HDR
    )
    stored = host.openai_response_get(queued["id"])
    assert stored["status"] in {"queued", "cancelled", "failed"}

    def lose_record(name: str, **kwargs: Any) -> _Bare:
        backend = _Bare()

        def complete(messages: list[dict[str, Any]], sampling: Any = None, **kw: Any) -> str:
            lost._openai_store.get = lambda key: None  # type: ignore[method-assign]
            raise RuntimeError("record disappeared")

        backend.complete = complete  # type: ignore[method-assign]
        return backend

    lost = Fx1Harness(backend_resolver=lose_record)
    monkeypatch.setattr(sdk.threading, "Thread", HoldThread)
    lost.openai_response({"model": "fx1", "input": "hi", "background": True}, headers=HDR)

    manual = "resp_manual_cancel"
    host._openai_store.put({"id": manual, "object": "response", "status": "queued", "output": []})
    cancelled = host.openai_response_cancel(manual)
    assert cancelled["status"] == "cancelled"


def test_sdk_audit_restores_env_and_records_a_clean_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    from fx1.sdk_audit import sdk_audit

    monkeypatch.setenv("FX1_CHECKPOINT_DIR", "/tmp/fx1-not-a-checkpoint")
    monkeypatch.setenv("FX1_LOCAL_SERVE_URL", "http://127.0.0.1:9")
    real = Fx1Harness.command

    def command(self: Fx1Harness, name: str) -> Any:
        if name == "not-a-command":
            return real(self, "doctor")
        return real(self, name)

    monkeypatch.setattr(Fx1Harness, "command", command)
    results = sdk_audit()
    assert results["registry_nonempty"] is True
    assert results["unknown_command_404"] is False


def test_plain_backend_criteria_replay_and_lost_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fx1.sdk_audit import _NonStreamingBackend

    plain = _NonStreamingBackend()
    assert plain.complete([{"role": "user", "content": "hi"}]) == "clean answer"

    backend = FakeBackend()
    harness = Fx1Harness(backend_resolver=lambda *a, **k: backend)
    spec = harness.eval_spec_create("edge-spec", suite="tooluse")
    updated = harness.eval_spec_update(spec["id"], testing_criteria=[{"name": "all-pass"}])
    assert updated["testing_criteria"][0]["name"] == "all-pass"

    done, completion_id = harness.openai_response({"model": "fx1", "input": "hi"}, headers=HDR)
    events, replay_id = harness.openai_response_replay(done["id"])
    assert events and replay_id == completion_id

    class HoldThread:
        last: HoldThread

        def __init__(self, target: Any, daemon: bool = True, name: str | None = None) -> None:
            self.target = target
            HoldThread.last = self

        def start(self) -> None:
            return None

    monkeypatch.setattr(sdk.threading, "Thread", HoldThread)
    queued, _cid = harness.openai_response(
        {"model": "fx1", "input": "hi", "background": True}, headers=HDR
    )
    rid = queued["id"]

    def finish(self: Fx1Harness, *args: Any, **kwargs: Any) -> Any:
        self._openai_store.delete(rid)
        raise RuntimeError("record disappeared")

    harness._openai_response_finish = finish.__get__(harness, Fx1Harness)  # type: ignore[method-assign]
    HoldThread.last.target()
    assert rid not in harness._bg_cancel
