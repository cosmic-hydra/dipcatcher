"""SYNTHETIC independent checks of the OpenAI-compat translation leaf.

Deterministic, no network: link resolution, request validators, envelope
sieves, stream grammars, the batch/conversation/vector-store surfaces, and
the bounded envelope store are exercised directly against hostile shapes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from fx1.serve.journal import JobJournal
from fx1.serve.openai_compat import (
    OpenAIBatchRequest,
    OpenAIChatRequest,
    OpenAIChatUpdate,
    OpenAICompatError,
    OpenAICompletionRequest,
    OpenAIConversationCreate,
    OpenAIConversationItemsAdd,
    OpenAIConversationUpdate,
    OpenAIEmbeddingRequest,
    OpenAIEnvelopeStore,
    OpenAIResponseRequest,
    OpenAIUploadCompleteRequest,
    OpenAIUploadCreateRequest,
    OpenAIVectorStoreFileBatchCreate,
    OpenAIVectorStoreFileCreate,
    OpenAIVectorStoreSearch,
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
    is_openai_path,
    legacy_to_chat,
    openai_chunks,
    openai_completion_envelope,
    openai_embedding_envelope,
    openai_envelope,
    openai_error_body,
    openai_messages,
    openai_model,
    openai_models,
    openai_response_call_items,
    openai_response_events,
    openai_response_object,
    openai_response_replay_events,
    openai_to_kwargs,
    openai_usage,
    paged_item_list,
    response_cap_call_items,
    response_input_item_dicts,
    response_input_items_for_store,
    response_input_to_messages,
    response_output_pieces,
    response_query_text,
    response_text_format,
    response_to_kwargs,
    validate_openai_output,
    validate_response_format,
)

_GOOD_SHA = "a" * 64
_BYOK_KEY_HDR = "X-Fx1-Byok-Api-Key"  # gitleaks:allow — header name, not a secret


def _chat(**over: Any) -> OpenAIChatRequest:
    body: dict[str, Any] = {
        "model": "fx1",
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(over)
    return OpenAIChatRequest.model_validate(body)


def _resp(**over: Any) -> OpenAIResponseRequest:
    body: dict[str, Any] = {"model": "fx1", "input": "hi"}
    body.update(over)
    return OpenAIResponseRequest.model_validate(body)


# --- link resolution: precedence + fail-closed overrides ----------------


def test_ext_backend_beats_header_beats_model_name() -> None:
    kw = openai_to_kwargs(
        _chat(fx1={"backend": "local_fx1", "checkpoint_dir": "/ck"}),
        {"X-Fx1-Backend": "hosted_k3"},
    )
    assert kw["backend"] == "local_fx1" and kw["checkpoint_dir"] == "/ck"
    kw = openai_to_kwargs(_chat(model="byok"), {"X-Fx1-Backend": "hosted_k3"})
    assert kw["backend"] == "hosted_k3"
    assert openai_to_kwargs(_chat(model="local_fx1"))["backend"] == "local_fx1"


def test_unknown_model_fails_closed_404_on_completion_surfaces() -> None:
    for body in (_chat(model="gpt-4o"), _resp(model="gpt-4o")):
        with pytest.raises(OpenAICompatError) as exc:
            if isinstance(body, OpenAIChatRequest):
                openai_to_kwargs(body, {})
            else:
                response_to_kwargs(body, {})
        assert exc.value.status == 404 and exc.value.code == "model_not_found"


def test_embeddings_model_is_upstream_name_not_link_gate() -> None:
    body = OpenAIEmbeddingRequest.model_validate({"model": "text-embedding-3", "input": "x"})
    kw = embeddings_to_kwargs(body, {})
    assert kw["backend"] == "hosted_k3" and kw["model"] == "text-embedding-3"


def test_ft_name_pins_registry_checkpoint() -> None:
    kw = openai_to_kwargs(
        _chat(model="ft:fx1:job:abc"),
        {},
        ft_resolver=lambda name: "/srv/ckpt-1",
    )
    assert kw["backend"] == "local_fx1"
    assert kw["checkpoint_dir"] == "/srv/ckpt-1"
    assert kw["_served_model"] == "ft:fx1:job:abc"


def test_ft_unregistered_fails_closed_404() -> None:
    with pytest.raises(OpenAICompatError) as exc:
        openai_to_kwargs(_chat(model="ft:fx1:ghost:000"), {}, ft_resolver=lambda n: None)
    assert exc.value.status == 404


def test_ft_with_byok_refuses_422() -> None:
    body = _chat(
        model="ft:fx1:job:abc",
        fx1={"byok": {"base_url": "https://api.example.com/v1", "api_key": "k", "model": "m"}},
    )
    with pytest.raises(OpenAICompatError) as exc:
        openai_to_kwargs(body, {}, ft_resolver=lambda n: "/ck")
    assert exc.value.status == 422


@pytest.mark.parametrize("channel", ["body", "header"])
def test_ft_with_checkpoint_dir_refuses_422(channel: str) -> None:
    # an ft: name resolves through the registry's own checkpoint — a
    # checkpoint_dir override has no link to bind and must refuse, never
    # drop silently (same class as the byok refusal).
    body = (
        _chat(model="ft:fx1:job:abc", fx1={"checkpoint_dir": "/other"})
        if channel == "body"
        else _chat(model="ft:fx1:job:abc")
    )
    hdrs = {} if channel == "body" else {"X-Fx1-Checkpoint-Dir": "/other"}
    with pytest.raises(OpenAICompatError) as exc:
        openai_to_kwargs(body, hdrs, ft_resolver=lambda n: "/registered")
    assert exc.value.status == 422


def test_byok_header_pairing_and_model_defaults() -> None:
    with pytest.raises(OpenAICompatError) as exc:
        openai_to_kwargs(_chat(), {_BYOK_KEY_HDR: "k"})
    assert exc.value.status == 400 and exc.value.code == "invalid_byok_headers"
    with pytest.raises(OpenAICompatError):
        openai_to_kwargs(_chat(), {"X-Fx1-Byok-Base-Url": "https://api.example.com"})
    with pytest.raises(OpenAICompatError) as exc2:
        # base+key but no upstream model anywhere
        openai_to_kwargs(
            _chat(model="byok"),
            {
                "X-Fx1-Byok-Base-Url": "https://api.example.com",
                _BYOK_KEY_HDR: "k",  # gitleaks:allow — probe literal
            },
        )
    assert "byok needs a model" in str(exc2.value)
    kw = openai_to_kwargs(
        _chat(model="gpt-4o"),
        {
            "X-Fx1-Byok-Base-Url": "https://api.example.com",
            _BYOK_KEY_HDR: "k",  # gitleaks:allow — probe literal
        },
    )
    assert kw["backend"] == "byok" and kw["byok"]["model"] == "gpt-4o"


def test_byok_and_checkpoint_dir_refuse_when_link_absent_from_chain() -> None:
    body = _chat(
        model="hosted_k3",
        fx1={"byok": {"base_url": "https://api.example.com", "api_key": "k", "model": "m"}},
    )
    with pytest.raises(OpenAICompatError) as exc:
        openai_to_kwargs(body, {})
    assert exc.value.status == 422
    # the override binds when byok is anywhere in the fallback chain
    ok = _chat(
        model="hosted_k3",
        fx1={
            "fallbacks": ["byok"],
            "byok": {"base_url": "https://api.example.com", "api_key": "k", "model": "m"},
        },
    )
    assert openai_to_kwargs(ok, {})["fallbacks"] == ["byok"]
    with pytest.raises(OpenAICompatError) as exc2:
        openai_to_kwargs(_chat(model="hosted_k3"), {"X-Fx1-Checkpoint-Dir": "/x"})
    assert exc2.value.status == 422
    with pytest.raises(OpenAICompatError):
        openai_to_kwargs(_chat(), {"X-Fx1-Backend": "nope"})


# --- receipt hashes + timeout: both channels enforce the same shape -----


def test_receipt_hash_shape_enforced_on_both_channels() -> None:
    with pytest.raises(OpenAICompatError):
        openai_to_kwargs(_chat(), {"X-Fx1-Receipt-Hashes": "not-a-digest"})
    with pytest.raises(OpenAICompatError):
        openai_to_kwargs(_chat(fx1={"receipt_hashes": ["not-a-digest"]}), {})
    kw = openai_to_kwargs(_chat(fx1={"receipt_hashes": [_GOOD_SHA]}), {})
    assert kw["receipt_hashes"] == [_GOOD_SHA]
    kw = openai_to_kwargs(_chat(), {"X-Fx1-Receipt-Hashes": f"{_GOOD_SHA}, {'b' * 64}"})
    assert kw["receipt_hashes"] == [_GOOD_SHA, "b" * 64]
    assert openai_to_kwargs(_chat(), {"X-Fx1-Receipt-Hashes": " , ,"})["receipt_hashes"] is None
    # body channel wins over the header on conflict
    kw = openai_to_kwargs(
        _chat(fx1={"receipt_hashes": [_GOOD_SHA]}), {"X-Fx1-Receipt-Hashes": "b" * 64}
    )
    assert kw["receipt_hashes"] == [_GOOD_SHA]


@pytest.mark.parametrize("raw", ["abc", "nan", "inf", "0", "-1", "3601"])
def test_timeout_header_fail_closed_400(raw: str) -> None:
    with pytest.raises(OpenAICompatError):
        openai_to_kwargs(_chat(), {"X-Fx1-Timeout": raw})


def test_timeout_resolution_precedence() -> None:
    kw = openai_to_kwargs(_chat(fx1={"timeout_s": 2.5}), {"X-Fx1-Timeout": "9"})
    assert kw["timeout_s"] == 2.5
    assert openai_to_kwargs(_chat(), {"X-Fx1-Timeout": "0.5"})["timeout_s"] == 0.5
    assert openai_to_kwargs(_chat(), {})["timeout_s"] is None


# --- openai_messages: the wire shape sieve -------------------------------


def test_tool_context_shape_refusals() -> None:
    with pytest.raises(OpenAICompatError):
        openai_messages(
            _chat(messages=[{"role": "user", "content": "x", "tool_call_id": "c"}]).messages
        )
    with pytest.raises(OpenAICompatError):
        openai_messages(
            _chat(
                messages=[
                    {
                        "role": "user",
                        "content": "x",
                        "tool_calls": [
                            {
                                "id": "c",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    }
                ]
            ).messages
        )
    with pytest.raises(OpenAICompatError):
        openai_messages(_chat(messages=[{"role": "tool", "content": "r"}]).messages)
    with pytest.raises(OpenAICompatError):
        # an assistant turn may carry tool_calls, but every member must be
        # a well-formed function call — a bare name is not enough
        openai_messages(
            _chat(messages=[{"role": "assistant", "tool_calls": [{"id": "c"}]}]).messages
        )
    flat = openai_messages(
        _chat(
            messages=[
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "42"},
            ]
        ).messages
    )
    assert flat[0]["tool_calls"][0]["id"] == "c1"
    assert flat[1]["tool_call_id"] == "c1"


def test_content_part_and_name_refusals() -> None:
    with pytest.raises(OpenAICompatError):
        openai_messages(
            _chat(messages=[{"role": "user", "content": [{"type": "image", "x": 1}]}]).messages
        )
    with pytest.raises(OpenAICompatError):
        openai_messages(_chat(messages=[{"role": "user", "content": [{"no_type": 1}]}]).messages)
    with pytest.raises(OpenAICompatError):
        openai_messages(_chat(messages=[{"role": "user", "content": "x", "name": 5}]).messages)
    flat = openai_messages(
        _chat(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
                }
            ]
        ).messages
    )
    assert flat[0]["content"] == "ab"  # parts concat verbatim — no injected separator


# --- request validators ---------------------------------------------------


def test_unsupported_fields_refuse_named_extras() -> None:
    for field in ("functions", "best_of", "suffix"):
        with pytest.raises(ValidationError):
            OpenAIChatRequest.model_validate(
                {"model": "fx1", "messages": [{"role": "user", "content": "x"}], field: 1}
            )
    # an explicit-null unsupported extra is still a named lie — refused
    with pytest.raises(ValidationError):
        OpenAIChatRequest.model_validate(
            {"model": "fx1", "messages": [{"role": "user", "content": "x"}], "functions": None}
        )


def test_chat_metadata_and_stop_and_token_bounds() -> None:
    with pytest.raises(ValidationError):
        _chat(metadata={f"k{i}": "v" for i in range(17)})
    with pytest.raises(ValidationError):
        _chat(metadata={"k" * 65: "v"})
    with pytest.raises(ValidationError):
        _chat(metadata={"k": "v" * 513})
    with pytest.raises(ValidationError):
        _chat(max_tokens=5, max_completion_tokens=6)
    assert _chat(max_tokens=5, max_completion_tokens=5).max_tokens == 5
    with pytest.raises(ValidationError):
        _chat(top_logprobs=1)  # top_logprobs without logprobs
    kw = openai_to_kwargs(_chat(stop="END"), {})
    assert kw["stop"] == ["END"]
    assert openai_to_kwargs(_chat(stop=[]), {})["stop"] is None
    with pytest.raises(ValidationError):
        _chat(tools=[{"type": "function", "function": {"name": f"f{i}"}} for i in range(129)])


def test_response_format_and_logit_bias_bounds() -> None:
    with pytest.raises(ValidationError):
        _chat(response_format={"type": "xml"})
    with pytest.raises(ValidationError):
        _chat(response_format={"type": "json_schema", "json_schema": {"schema": {"type": "bogus"}}})
    ok = _chat(
        response_format={"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}
    )
    assert ok.response_format["type"] == "json_schema"
    with pytest.raises(ValidationError):
        _chat(logit_bias={"not-an-int": 1})
    with pytest.raises(ValidationError):
        _chat(logit_bias={"42": 101})


# --- usage + envelope sieves ------------------------------------------------


def test_openai_usage_sieves_non_ints_and_bools() -> None:
    assert openai_usage(None) is None
    assert openai_usage("x") is None  # type: ignore[arg-type]
    # a bool is not a token count — same sieve as the billable ledger
    assert openai_usage({"prompt_tokens": 5, "b": True, "c": "x", "d": 1.5}) == {"prompt_tokens": 5}


def test_openai_envelope_tool_call_choice_shape() -> None:
    env = openai_envelope(
        cid="deadbeef",
        content="",
        backend="hosted_k3",
        tool_calls=[
            [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        ],
        finish_reasons=["tool_calls"],
        created=1,
    )
    ch = env["choices"][0]
    assert ch["message"]["content"] is None  # calls-only turn ships null content
    assert ch["finish_reason"] == "tool_calls"
    env2 = openai_envelope(cid="x", content=["a", "b"], backend="hosted_k3", created=1)
    assert [c["index"] for c in env2["choices"]] == [0, 1]
    assert all(c["finish_reason"] == "stop" for c in env2["choices"])
    # bools never reach the wire usage claim
    env3 = openai_envelope(
        cid="x", content="a", backend="b", created=1, usage={"prompt_tokens": True, "t": 2}
    )
    assert env3["usage"] == {"t": 2}


def test_legacy_envelope_flat_choices_echo_and_model_join() -> None:
    envs = [
        openai_envelope(cid="c1", content="alpha", backend="b1", model="m1", created=10),
        openai_envelope(
            cid="c2",
            content="beta",
            backend="b2",
            model="m2",
            created=11,
            usage={"prompt_tokens": True, "completion_tokens": 3},
        ),
    ]
    env = openai_completion_envelope(cid="cc", envs=envs, prompts=["p1", "p2"], echo=True)
    assert env["object"] == "text_completion"
    assert [c["text"] for c in env["choices"]] == ["p1alpha", "p2beta"]
    assert all(c["logprobs"] is None for c in env["choices"])
    assert env["system_fingerprint"] == "b1+b2"
    # mixed-model elements join like the fingerprint — never last-wins
    assert env["model"] == "m1+m2"
    # the bool claim is sieved, the real int still sums
    assert env["usage"] == {"completion_tokens": 3}
    assert env["created"] == 11


def test_legacy_to_chat_forces_store_off_and_defaults() -> None:
    body = OpenAICompletionRequest.model_validate({"model": "fx1", "prompt": "x"})
    req = legacy_to_chat(body, "x")
    assert req.store is False
    assert req.max_tokens == 16
    assert req.messages[0].content == "x"


def test_completion_events_and_chunks_grammar() -> None:
    env = openai_completion_envelope(
        cid="c",
        envs=[
            openai_envelope(cid="c1", content="one two " * 20, backend="b", model="m", created=1)
        ],
        prompts=["p"],
    )
    events = list(completion_events(env, include_usage=True))
    assert events[0]["choices"][0]["finish_reason"] is None
    assert events[-2]["choices"][0]["finish_reason"] == "stop"
    assert events[-1]["choices"] == [] and events[-1]["usage"] is None
    chunks = list(
        openai_chunks(
            text="hi", backend="b", model="m", cid="c", include_usage=True, usage={"t": 1}
        )
    )
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["usage"] == {"t": 1}


def test_validate_response_format_502s_on_violations() -> None:
    validate_response_format(None, "anything")
    validate_response_format({"type": "text"}, "anything")
    validate_response_format({"type": "json_object"}, '{"a": 1}')
    with pytest.raises(OpenAICompatError) as exc:
        validate_response_format({"type": "json_object"}, "not json")
    assert exc.value.status == 502 and exc.value.code == "format_violation"
    with pytest.raises(OpenAICompatError):
        validate_response_format({"type": "json_object"}, "[1]")
    with pytest.raises(OpenAICompatError):
        validate_response_format(
            {
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object", "required": ["a"]}},
            },
            "{}",
        )
    with pytest.raises(OpenAICompatError):
        validate_openai_output(_chat(response_format={"type": "json_object"}), "x")


# --- responses surface ------------------------------------------------------


def test_response_input_flattening_and_refusals() -> None:
    msgs = response_input_to_messages("hello", "be terse")
    assert msgs == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hello"},
    ]
    msgs = response_input_to_messages(
        [
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "sys"}],
            },
            {"type": "function_call", "name": "f", "arguments": "{}", "call_id": "c1"},
            {"type": "function_call_output", "call_id": "c1", "output": "42"},
        ],
        None,
    )
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == "sys"
    assert msgs[1]["tool_calls"][0]["id"] == "c1"
    assert msgs[2] == {"role": "tool", "tool_call_id": "c1", "content": "42"}
    with pytest.raises(OpenAICompatError):
        response_input_to_messages(
            [{"type": "file_search_call", "queries": [], "results": []}], None
        )
    with pytest.raises(OpenAICompatError):
        response_input_to_messages([{"type": "web_search_call"}], None)
    with pytest.raises(OpenAICompatError):
        response_input_to_messages([42], None)


def test_response_to_kwargs_gates_and_echoes() -> None:
    body = _resp(
        max_output_tokens=7,
        reasoning={"effort": "high"},
        safety_identifier="u-1",
        tools=[{"type": "file_search", "vector_store_ids": ["vs_1"]}],
        tool_choice="auto",
    )
    kw = response_to_kwargs(body, {})
    assert kw["max_tokens"] == 7
    assert kw["reasoning_effort"] == "high"
    assert kw["user"] == "u-1"
    # a file-only tool list has nothing to bind on the model channel —
    # all three tool kwargs stay None (never a bare tool_choice)
    assert kw["tools"] is None and kw["tool_choice"] is None and kw["parallel_tool_calls"] is None
    body2 = _resp(
        tools=[{"type": "function", "name": "f"}],
        tool_choice={"type": "function", "name": "f"},
        parallel_tool_calls=False,
        include=["message.output_text.logprobs"],
        top_logprobs=3,
    )
    kw2 = response_to_kwargs(body2, {})
    assert kw2["tools"] == [{"type": "function", "function": {"name": "f"}}]
    assert kw2["tool_choice"] == {"type": "function", "function": {"name": "f"}}
    assert kw2["parallel_tool_calls"] is False
    assert kw2["logprobs"] is True and kw2["top_logprobs"] == 3
    tf = response_text_format(_resp(text={"format": {"type": "json_object"}}))
    assert tf == {"type": "json_object"}
    assert (
        response_query_text(
            [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "q"}]}]
        )
        == "q"
    )


def test_response_request_mutual_exclusion_and_bounds() -> None:
    with pytest.raises(ValidationError):
        _resp(conversation="conv_1", previous_response_id="resp_1")
    with pytest.raises(ValidationError):
        _resp(reasoning={"effort": "bogus"})
    with pytest.raises(ValidationError):
        _resp(reasoning={"effort": "low", "extra": 1})
    with pytest.raises(ValidationError):
        _resp(
            text={"format": {"type": "json_schema", "json_schema": {"schema": {"type": "bogus"}}}}
        )
    with pytest.raises(ValidationError):
        _resp(top_logprobs=1)  # without the include knob
    with pytest.raises(ValidationError):
        _resp(include=["bogus.channel"])
    with pytest.raises(ValidationError):
        _resp(
            n=2
        )  # a named unsupported chat-completions field  # unsupported request field refuses


def test_response_call_items_sieve_and_cap() -> None:
    calls = [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}},
        {"id": "c2", "type": "function", "function": {"name": "g", "arguments": {"b": 2}}},
        "garbage",  # provider-verbatim member that isn't an object
        {"id": "c3", "function": 42},  # dict member, unshapeable function
    ]
    items = openai_response_call_items(calls)
    assert len(items) == 3  # the non-dict member is sieved, never a crash
    assert items[0]["arguments"] == '{"a": 1}'
    assert items[1]["arguments"] == '{"b": 2}'
    assert items[2]["call_id"] == "c3" and items[2]["name"] is None
    assert all(i["id"].startswith("fc_") for i in items)
    got, inc = response_cap_call_items(_resp(max_tool_calls=1), calls[:2])
    assert got is not None and len(got) == 1 and inc == {"reason": "max_tool_calls"}
    got0, inc0 = response_cap_call_items(_resp(max_tool_calls=0), calls[:1])
    assert got0 == [] and inc0 == {"reason": "max_tool_calls"}
    none_items, none_inc = response_cap_call_items(_resp(), [])
    assert none_items is None and none_inc is None


def test_response_object_usage_sieve_and_status_output() -> None:
    env = openai_response_object(
        rid="r1",
        item_id="m1",
        content="hi",
        body=_resp(),
        model="m",
        usage={"prompt_tokens": True, "completion_tokens": 4, "total_tokens": 9},
        created=1,
    )
    # bools drop out of the wire claim; ints pass through
    assert env["usage"] == {"input_tokens": 0, "output_tokens": 4, "total_tokens": 9}
    env2 = openai_response_object(
        rid="r2",
        item_id="m2",
        content="",
        body=_resp(),
        model=None,
        usage={"prompt_tokens": True},
        created=1,
    )
    assert env2["usage"] is None  # nothing billable stays absent
    inc = openai_response_object(
        rid="r3",
        item_id="m3",
        content="part",
        body=_resp(),
        model=None,
        usage=None,
        status="incomplete",
        incomplete_details={"reason": "max_tool_calls"},
        created=1,
    )
    assert inc["status"] == "incomplete" and inc["output"]
    dead = openai_response_object(
        rid="r4",
        item_id="m4",
        content="hidden",
        body=_resp(),
        model=None,
        usage=None,
        status="failed",
        error={"code": "x", "message": "y"},
        created=1,
    )
    assert dead["output"] == [] and dead["error"]["code"] == "x"


def test_response_events_terminal_and_incomplete() -> None:
    events = dict(
        openai_response_events(
            text="hi",
            rid="r1",
            item_id="m1",
            body=_resp(),
            model="m",
            usage={"prompt_tokens": 1, "completion_tokens": 2},
            created=1,
        )
    )
    assert "response.created" in events and "response.completed" in events
    inc_events = dict(
        openai_response_events(
            text="part",
            rid="r2",
            item_id="m2",
            body=_resp(max_tool_calls=0),
            model=None,
            usage=None,
            final_status="incomplete",
            incomplete_details={"reason": "max_tool_calls"},
        )
    )
    assert "response.incomplete" in inc_events
    assert inc_events["response.incomplete"]["response"]["incomplete_details"] == {
        "reason": "max_tool_calls"
    }


def test_response_output_pieces_and_replay_grammar() -> None:
    text, item_id, calls, search, lp = response_output_pieces({"output": [], "status": "completed"})
    assert text == "" and item_id == "" and calls is None and search is None and lp is None
    # an incomplete envelope with no message item = a calls turn truncated
    # to zero — replays the create-time grammar, not a prose turn
    _t, _i, call_list, _s, _l = response_output_pieces({"output": [], "status": "incomplete"})
    assert call_list == []
    env = {
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "created_at": 1,
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi", "annotations": []}],
            }
        ],
        "_fx1_completion_id": "secret-stash",
    }
    frames = list(openai_response_replay_events(env))
    assert frames[-1][0] == "response.completed"
    terminal = frames[-1][1]["response"]
    assert "_fx1_completion_id" not in terminal
    assert frames[0][0] == "response.created"
    assert frames[0][1]["response"]["status"] == "in_progress"
    bg_frames = list(openai_response_replay_events({**env, "background": True, "status": "queued"}))
    assert [e for e, _p in bg_frames] == ["response.created", "response.queued"]
    # a non-terminal background record emits only the lifecycle prelude
    assert bg_frames[0][1]["response"]["status"] == "queued"
    assert bg_frames[1][1]["response"]["status"] == "queued"


def test_paged_item_list_cursors_and_bounds() -> None:
    items = [{"id": f"i{n}"} for n in range(5)]
    with pytest.raises(OpenAICompatError):
        paged_item_list(items, limit=0)
    with pytest.raises(OpenAICompatError):
        paged_item_list(items, limit=101)
    with pytest.raises(OpenAICompatError):
        paged_item_list(items, limit=2, order="random")
    page = paged_item_list(items, limit=2)
    assert [i["id"] for i in page["data"]] == ["i0", "i1"]
    assert page["has_more"] and page["first_id"] == "i0" and page["last_id"] == "i1"
    page = paged_item_list(items, limit=2, after="i1")
    assert [i["id"] for i in page["data"]] == ["i2", "i3"]
    # before takes the tail window before the cursor (same pin as Lane 172)
    page = paged_item_list(items, limit=2, before="i4")
    assert [i["id"] for i in page["data"]] == ["i2", "i3"]
    page = paged_item_list(items, limit=2, order="desc")
    assert [i["id"] for i in page["data"]] == ["i4", "i3"]
    with pytest.raises(OpenAICompatError) as exc:
        paged_item_list(items, limit=2, after="ghost")
    assert exc.value.code == "invalid_cursor"


def test_chained_input_and_conversation_id_of() -> None:
    chained = chained_response_input(
        {"output": [{"type": "message", "id": "x", "content": []}]},
        [{"type": "message", "id": "y", "role": "user", "content": []}],
        "new",
    )
    assert all("id" not in it for it in chained)
    assert chained[-1]["content"] == [{"type": "input_text", "text": "new"}]
    assert conversation_id_of("conv_1") == "conv_1"
    assert conversation_id_of({"id": "conv_2"}) == "conv_2"
    assert conversation_id_of({"id": 7}) is None and conversation_id_of(None) is None


def test_store_item_id_determinism_and_copies() -> None:
    items = response_input_items_for_store("hello", rid="resp_x")
    assert items[0]["id"].startswith("msg_")
    again = response_input_items_for_store("hello", rid="resp_x")
    assert [i["id"] for i in items] == [i["id"] for i in again]  # deterministic re-mint
    items[0]["content"] = "mutated"
    fresh = response_input_item_dicts("hello")
    assert fresh[0]["content"] == [{"type": "input_text", "text": "hello"}]  # fresh dicts
    stored = chat_messages_for_store(_chat().messages, envelope_id="c")
    assert stored[0]["id"].startswith("msg_")


# --- embeddings -------------------------------------------------------------


def test_embedding_input_sieve() -> None:
    for good in ("x", ["a", "b"], [1, 2, 3], [[1, 2], [3]]):
        OpenAIEmbeddingRequest.model_validate({"model": "m", "input": good})
    for bad in (
        [],
        ["a", 1],
        [1, "a"],
        [[1], "a"],
        [i for i in range(2049)],
        ["x"] * 2049,
    ):
        with pytest.raises(ValidationError):
            OpenAIEmbeddingRequest.model_validate({"model": "m", "input": bad})
    # pydantic's lax union coerces bool→int before the field validator sees
    # it — pin the documented contract rather than pretend it is refused
    assert OpenAIEmbeddingRequest.model_validate({"model": "m", "input": [True]}).input == [1]


def test_embedding_envelope_verbatim() -> None:
    env = openai_embedding_envelope(
        data=[{"object": "embedding", "embedding": [0.1]}], model="m", usage=None
    )
    assert env["object"] == "list" and env["usage"] is None and env["model"] == "m"


# --- batch / file / upload surfaces -----------------------------------------


def test_batch_line_shape_refusals_and_body_parity() -> None:
    good = {
        "custom_id": "a",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": "fx1", "messages": [{"role": "user", "content": "x"}]},
    }
    line = batch_line_shape(good, endpoint="/v1/chat/completions", lineno=1)
    assert line["custom_id"] == "a"
    obj = batch_line_body(line, "/v1/chat/completions")
    assert isinstance(obj, OpenAIChatRequest)
    for bad, _why in [
        ("x", "object"),
        ({**good, "custom_id": ""}, "custom_id"),
        ({**good, "custom_id": "x" * 65}, "custom_id"),
        ({**good, "method": "GET"}, "method"),
        ({**good, "url": "/v1/embeddings"}, "url"),
        ({**good, "body": "x"}, "body"),
    ]:
        with pytest.raises(OpenAICompatError):
            batch_line_shape(bad, endpoint="/v1/chat/completions", lineno=1)
    with pytest.raises(OpenAICompatError):
        batch_line_body(
            {"custom_id": "a", "body": {"model": "fx1"}}, "/v1/chat/completions"
        )  # messages missing → endpoint validation refuses


def test_batch_object_never_projects_the_callback_secret() -> None:
    rec = {
        "batch_id": "batch_1",
        "endpoint": "/v1/chat/completions",
        "input_file_id": "file_1",
        "completion_window": "24h",
        "status": "completed",
        "created_at": 1,
        "expires_at": 2,
        "request_counts": {"total": 1, "completed": 1, "failed": 0},
        "callback_url": "https://example.com/cb",
        "_callback_secret": "fx1k_bad",
        "metadata": None,
        "errors": None,
        "output_file_id": None,
        "error_file_id": None,
        "in_progress_at": None,
        "finalizing_at": None,
        "completed_at": 2,
        "failed_at": None,
        "expired_at": None,
        "cancelling_at": None,
        "cancelled_at": None,
        "callback_status": None,
        "callback_attempts": 0,
        "callback_error": None,
    }
    env = batch_object(rec)
    assert "fx1k_bad" not in str(env) and "_callback_secret" not in env
    line = batch_output_line(custom_id="c", status_code=200, body={"x": 1}, rid="r")
    assert line["response"]["request_id"] == "req_r" and line["id"] == "batch_req_r"
    f = file_object(
        {"file_id": "file_1", "purpose": "batch", "filename": "a.jsonl", "size": 3, "created_at": 1}
    )
    assert f["object"] == "file" and f["bytes"] == 3


def test_batch_request_metadata_and_callback_bounds() -> None:
    with pytest.raises(ValidationError):
        OpenAIBatchRequest.model_validate(
            {
                "input_file_id": "f",
                "endpoint": "/v1/chat/completions",
                "metadata": {f"k{i}": "v" for i in range(17)},
            }
        )
    # per-key/value bounds — same contract every sibling surface enforces
    for meta in ({"k" * 65: "v"}, {"k": "v" * 513}):
        with pytest.raises(ValidationError):
            OpenAIBatchRequest.model_validate(
                {
                    "input_file_id": "f",
                    "endpoint": "/v1/chat/completions",
                    "metadata": meta,
                }
            )
    with pytest.raises(ValidationError):
        OpenAIBatchRequest.model_validate(
            {
                "input_file_id": "f",
                "endpoint": "/v1/chat/completions",
                "callback_secret": "s",
            }
        )
    for bad_url in ("ftp://h", "http://user:pw@h/", "http://h:0/", "noscheme"):
        with pytest.raises(ValidationError):
            OpenAIBatchRequest.model_validate(
                {
                    "input_file_id": "f",
                    "endpoint": "/v1/chat/completions",
                    "callback_url": bad_url,
                }
            )


def test_callback_url_private_literal_ssrf_gate() -> None:
    # a private IP literal refuses without DNS at submit time
    with pytest.raises(ValidationError):
        OpenAIBatchRequest.model_validate(
            {
                "input_file_id": "f",
                "endpoint": "/v1/chat/completions",
                "callback_url": "http://127.0.0.1:8080/cb",
            }
        )
    prev = os.environ.get("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS")
    os.environ["FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"] = "1"
    try:
        req = OpenAIBatchRequest.model_validate(
            {
                "input_file_id": "f",
                "endpoint": "/v1/chat/completions",
                "callback_url": "http://127.0.0.1:8080/cb",
            }
        )
        assert req.callback_url == "http://127.0.0.1:8080/cb"
    finally:
        if prev is None:
            os.environ.pop("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS", None)
        else:
            os.environ["FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"] = prev


def test_upload_models_forbid_extras_and_bound() -> None:
    OpenAIUploadCreateRequest.model_validate(
        {"purpose": "batch", "filename": "a.jsonl", "bytes": 5, "mime_type": "application/jsonl"}
    )
    with pytest.raises(ValidationError):
        OpenAIUploadCreateRequest.model_validate(
            {
                "purpose": "batch",
                "filename": "a.jsonl",
                "bytes": 5,
                "mime_type": "x",
                "extra": 1,
            }
        )
    with pytest.raises(ValidationError):
        OpenAIUploadCompleteRequest.model_validate({"part_ids": [], "md5": "x" * 33})


# --- conversation + vector-store surfaces -----------------------------------


def test_conversation_models_bound_metadata_and_refuse_item_ids() -> None:
    OpenAIConversationCreate.model_validate(
        {"metadata": {"a": "b"}, "items": [{"type": "message"}]}
    )
    for bad in ({"a" * 65: "v"}, {"a": "v" * 513}):
        with pytest.raises(ValidationError):
            OpenAIConversationCreate.model_validate({"metadata": bad})
        with pytest.raises(ValidationError):
            OpenAIConversationUpdate.model_validate({"metadata": bad})
        with pytest.raises(ValidationError):
            OpenAIChatUpdate.model_validate({"metadata": bad})
    with pytest.raises(ValidationError):
        OpenAIConversationItemsAdd.model_validate({"items": []})
    with pytest.raises(ValidationError):
        OpenAIConversationItemsAdd.model_validate({"item_ids": ["itm_x"]})
    OpenAIConversationItemsAdd.model_validate({"items": [{"type": "message"}]})


def test_vector_store_search_bounds_and_threshold_sieve() -> None:
    OpenAIVectorStoreSearch.model_validate({"query": "x"})
    OpenAIVectorStoreSearch.model_validate({"query": ["a", "b"], "max_num_results": 50})
    for bad_ro in (
        {"ranker": "bm25"},
        {"bogus": 1},
        {"score_threshold": "x"},
        {"score_threshold": True},
    ):
        with pytest.raises(ValidationError):
            OpenAIVectorStoreSearch.model_validate({"query": "x", "ranking_options": bad_ro})
    # Numeric out-of-range thresholds pass the body model so VectorStore.search
    # can answer 400 invalid_filters. Rejecting them here becomes a 422 first.
    from fx1.serve.vectorstores import VectorStoreError, VectorStoreStore

    store = VectorStoreStore.__new__(VectorStoreStore)
    for threshold in (1.5, -0.1):
        body = OpenAIVectorStoreSearch.model_validate(
            {"query": "x", "ranking_options": {"score_threshold": threshold}}
        )
        assert body.ranking_options is not None
        with pytest.raises(VectorStoreError) as exc:
            store.search(["vs_1"], "x", score_threshold=float(threshold))
        assert exc.value.status == 400 and exc.value.code == "invalid_filters"
    with pytest.raises(ValidationError):
        OpenAIVectorStoreSearch.model_validate({"query": "x", "rewrite_query": True})
    with pytest.raises(ValidationError):
        OpenAIVectorStoreSearch.model_validate({"query": ["a", 1]})
    OpenAIVectorStoreFileCreate.model_validate({"file_id": "file_1"})
    with pytest.raises(ValidationError):
        OpenAIVectorStoreFileBatchCreate.model_validate({"file_ids": []})


# --- models + error envelopes ------------------------------------------------


def test_models_inventory_and_error_shapes() -> None:
    ml = openai_models(created_by_id={"ft:x": 0}, extra_ids=["ft:x"])
    ids = [m.id for m in ml.data]
    assert ids == ["fx1", "hosted_k3", "local_fx1", "byok", "ft:x"]
    with pytest.raises(OpenAICompatError) as exc:
        openai_model("ghost")
    assert exc.value.status == 404 and exc.value.code == "model_not_found"
    assert is_openai_path("/v1/chat/completions") and not is_openai_path("/harness/complete")
    for status, typ in (
        (400, "invalid_request_error"),
        (404, "invalid_request_error"),
        (500, "server_error"),
    ):
        body = openai_error_body("m", status, "c")
        assert body["error"]["type"] == typ and body["error"]["param"] is None


# --- OpenAIEnvelopeStore ------------------------------------------------------


def _env_store(cap: int = 3, journal: JobJournal | None = None) -> OpenAIEnvelopeStore:
    return OpenAIEnvelopeStore(cap=cap, journal=journal)


def test_envelope_store_cap_eviction_and_isolation() -> None:
    with pytest.raises(ValueError):
        _env_store(cap=0)
    store = _env_store(cap=2)
    store.put({"id": "e1", "object": "response"})
    store.put({"id": "e2", "object": "response"}, items={"input_items": [{"a": 1}]})
    got = store.get("e1")
    assert got is not None
    got["mutated"] = True
    assert "mutated" not in store.get("e1")  # deepcopy isolation
    assert store.get_items("e2", "input_items") == [{"a": 1}]
    assert store.get_items("e2", "ghost") == []  # live envelope, missing key
    store.put({"id": "e3", "object": "response"})
    assert store.get("e1") is None  # oldest evicted with its items
    assert store.get_items("e1", "input_items") is None
    assert store.delete("e2") is True
    assert store.get_items("e2", "input_items") is None  # gone with the envelope
    assert store.delete("ghost") is False


def test_envelope_store_cas_and_repins() -> None:
    store = _env_store()
    store.put({"id": "e1", "status": "queued", "object": "response"})
    assert store.transition_status("e1", expect={"queued"}, status="in_progress") is True
    assert store.transition_status("e1", expect={"queued"}, status="in_progress") is False
    assert (
        store.put_unless_status({"id": "e1", "status": "completed"}, forbidden={"cancelled"})
        is True
    )
    store.put({"id": "e2", "status": "cancelled"})
    assert (
        store.put_unless_status({"id": "e2", "status": "completed"}, forbidden={"cancelled"})
        is False
    )  # a terminal verdict is sticky
    assert store.get("e2")["status"] == "cancelled"
    store.delete("e1")
    assert store.put_if_present({"id": "e1", "status": "completed"}) is False
    assert store.put_unless_status({"id": "e1"}, forbidden=set(), require_existing=True) is False
    store.put({"id": "e3", "status": "live", "object": "response"})
    rehydrated = store.repin({"id": "e3", "status": "stale"})
    assert rehydrated["status"] == "live"  # live wins over the cached replay
    rehydrated2 = store.repin({"id": "e4", "status": "cached"})
    assert rehydrated2["status"] == "cached" and store.get("e4")["status"] == "cached"


def test_envelope_store_items_mutation_and_listing() -> None:
    store = _env_store()
    store.put({"id": "e1", "object": "response"})
    merged = store.mutate_items("e1", "items", lambda cur: [*cur, {"a": 1}])
    assert merged == [{"a": 1}]
    merged2 = store.mutate_items("e1", "items", lambda cur: [*cur, {"b": 2}])
    assert merged2 == [{"a": 1}, {"b": 2}]
    assert store.mutate_items("ghost", "items", lambda c: c) is None
    upd = store.update_metadata("e1", {"k": "v"})
    assert upd is not None and upd["metadata"] == {"k": "v"}
    assert store.update_metadata("ghost", {}) is None
    store.put({"id": "e2", "object": "chat.completion"})
    store.put({"id": "e3", "object": "response"})
    assert [e["id"] for e in store.list_envelopes("response")] == ["e1", "e3"]


def test_envelope_store_journal_replay_and_fail_closed_boot(tmp_path: Path) -> None:
    path = tmp_path / "env.jsonl"
    store = _env_store(journal=JobJournal(path))
    store.put({"id": "e1", "object": "response"}, items={"input_items": [{"a": 1}]})
    store.mutate_items("e1", "input_items", lambda cur: [*cur, {"b": 2}])
    store.put({"id": "e2", "object": "response"})
    store.delete("e2")
    for _ in range(2):
        recovered = _env_store(journal=JobJournal(path))
        assert recovered.get("e1")["object"] == "response"
        assert recovered.get_items("e1", "input_items") == [{"a": 1}, {"b": 2}]
        assert recovered.get("e2") is None
        store = recovered  # compacted state keeps reloading cleanly
    with path.open("ab") as stream:
        stream.write(b"{SYNTHETIC partial tail")
    with pytest.raises(RuntimeError):
        _env_store(journal=JobJournal(path))
