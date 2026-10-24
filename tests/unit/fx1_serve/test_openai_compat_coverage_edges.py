"""Line and branch edges for fx1.serve.openai_compat left open by the fx1 run."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from fx1.serve.openai_compat import (
    OpenAIChatMessage,
    OpenAIChatRequest,
    OpenAICompletionRequest,
    OpenAIConversationCreate,
    OpenAIConversationItemsAdd,
    OpenAIConversationUpdate,
    OpenAIEmbeddingRequest,
    OpenAIEnvelopeStore,
    OpenAIFileSearchTool,
    OpenAIResponseRequest,
    OpenAITool,
    OpenAIToolFunction,
    OpenAIVectorStoreSearch,
    _resolve_openai_link,
    _response_output_item_events,
    openai_messages,
    response_input_to_messages,
    response_query_text,
)


def _invalid(model: type[Any], **kwargs: Any) -> str:
    with pytest.raises(ValidationError) as exc:
        model(**kwargs)
    return str(exc.value)


def _tool() -> OpenAITool:
    return OpenAITool(function=OpenAIToolFunction(name="lookup"))


def _search_tool(**ranking: Any) -> OpenAIFileSearchTool:
    return OpenAIFileSearchTool(
        type="file_search",
        vector_store_ids=["vs_1"],
        ranking_options=ranking or None,
    )


def test_chat_and_legacy_completion_shapes_fail_closed() -> None:
    assert "1–512" in _invalid(
        OpenAIChatRequest, messages=[{"role": "user", "content": "hi"}], stop=[""]
    )
    assert "function" in _invalid(
        OpenAIChatRequest,
        messages=[{"role": "user", "content": "hi"}],
        tools=[_tool()],
        tool_choice={"type": "nope"},
    )
    bad_schema = _invalid(
        OpenAIChatRequest,
        messages=[{"role": "user", "content": "hi"}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "out", "schema": {"type": "object", "required": "nope"}},
        },
    )
    assert "not a valid schema" in bad_schema

    assert "non-empty list" in _invalid(OpenAICompletionRequest, prompt=[])
    assert "512" in _invalid(OpenAICompletionRequest, prompt=["x"] * 513)
    assert "131072" in _invalid(OpenAICompletionRequest, prompt="")
    assert "at most 4" in _invalid(
        OpenAICompletionRequest, prompt="hi", stop=["a", "b", "c", "d", "e"]
    )
    assert "1–512" in _invalid(OpenAICompletionRequest, prompt="hi", stop=[""])
    assert "1–512" in _invalid(OpenAICompletionRequest, prompt="hi", stop="")
    assert "token ids" in _invalid(OpenAICompletionRequest, prompt="hi", logit_bias={"nope": 1})
    assert "outside" in _invalid(OpenAICompletionRequest, prompt="hi", logit_bias={"1": 101})
    wide = {f"k{i}": "v" for i in range(17)}
    assert "16" in _invalid(OpenAICompletionRequest, prompt="hi", metadata=wide)
    assert "64" in _invalid(OpenAICompletionRequest, prompt="hi", metadata={"k" * 65: "v"})

    legacy = OpenAICompletionRequest(prompt="hi")
    legacy.logit_bias = {"nope": 1}  # type: ignore[assignment]
    with pytest.raises(ValueError, match="token ids"):
        legacy._legacy_valid()
    legacy.logit_bias = {"1": 1}
    legacy.metadata = {"ok": "v" * 513}
    with pytest.raises(ValueError, match="512"):
        legacy._legacy_valid()


def test_openai_messages_and_ft_header_fallbacks() -> None:
    bare = OpenAIChatMessage(role="user", content=None)
    with pytest.raises(Exception, match="content is required"):
        openai_messages([bare])
    empty_calls = OpenAIChatMessage(role="assistant", content="hi", tool_calls=[])
    with pytest.raises(Exception, match="non-empty list"):
        openai_messages([empty_calls])
    tool = OpenAIChatMessage(role="tool", content=None, tool_call_id="call_1")
    with pytest.raises(Exception, match="tool output"):
        openai_messages([tool])
    named = OpenAIChatMessage(role="user", content="hi", name="ada")
    assert openai_messages([named]) == [{"role": "user", "content": "hi", "name": "ada"}]

    backend, fallbacks, checkpoint, byok, served = _resolve_openai_link(
        "ft:tiny",
        None,
        {"x-fx1-fallbacks": " hosted_k3 , , local_fx1 "},
        ft_resolver=lambda _name: "/tmp/ckpt",
    )
    assert backend == "local_fx1"
    assert fallbacks == ["hosted_k3", "local_fx1"]
    assert checkpoint == "/tmp/ckpt" and byok is None and served == "ft:tiny"


def test_response_request_and_input_translation() -> None:
    assert "empty" in _invalid(OpenAIResponseRequest, input="   ")
    wide = {f"k{i}": "v" for i in range(17)}
    assert "16" in _invalid(OpenAIResponseRequest, input="hi", metadata=wide)
    assert "512" in _invalid(OpenAIResponseRequest, input="hi", metadata={"k" * 65: "v"})
    assert "unsupported text keys" in _invalid(
        OpenAIResponseRequest, input="hi", text={"format": {}, "extra": 1}
    )
    assert "must be an object" in _invalid(
        OpenAIResponseRequest, input="hi", text={"format": "text"}
    )
    assert "not a valid schema" in _invalid(
        OpenAIResponseRequest,
        input="hi",
        text={"format": {"type": "json_schema", "schema": {"type": "object", "required": "nope"}}},
    )
    assert "ranking_options" in _invalid(
        OpenAIResponseRequest,
        input="hi",
        tools=[_search_tool(ranker="auto", extra="no")],
    )
    query = response_query_text(
        [  # type: ignore[list-item]
            "nope",
            {"type": "other"},
            {"role": 1},
            {"content": 1},
            {"role": "assistant", "content": "last"},
            {"role": "user", "content": 1},
        ]
    )
    assert query == "last"
    user_query = response_query_text([{"role": "user", "content": "ask"}])  # type: ignore[list-item]
    assert user_query == "ask"

    with pytest.raises(Exception, match="output_text"):
        response_input_to_messages(
            [{"type": "function_call_output", "call_id": "c1", "output": ["nope"]}],
            None,
        )
    with pytest.raises(Exception, match="unknown role"):
        response_input_to_messages(
            [{"type": "message", "role": "narrator", "content": "hi"}],
            None,
        )
    with pytest.raises(Exception, match="must be objects"):
        response_input_to_messages(
            [{"type": "message", "role": "user", "content": ["nope"]}],
            None,
        )
    with pytest.raises(Exception, match="only text parts"):
        response_input_to_messages(
            [{"type": "message", "role": "user", "content": [{"type": "image"}]}],
            None,
        )
    with pytest.raises(Exception, match="output_text"):
        response_input_to_messages(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "output_text", "text": "x"}],
                }
            ],
            None,
        )


def test_conversation_search_embedding_and_envelope_edges() -> None:
    created = OpenAIConversationCreate()
    created.metadata = {f"k{i}": "v" for i in range(17)}
    with pytest.raises(ValueError, match="16"):
        created._valid()
    created.metadata = None
    created.items = ["nope"]  # type: ignore[assignment]
    with pytest.raises(ValueError, match="message-item"):
        created._valid()
    updated = OpenAIConversationUpdate()
    updated.metadata = {f"k{i}": "v" for i in range(17)}
    with pytest.raises(ValueError, match="16"):
        updated._valid()
    added = OpenAIConversationItemsAdd(items=[{"type": "message"}])
    added.items = ["nope"]  # type: ignore[assignment]
    with pytest.raises(ValueError, match="message-item"):
        added._valid()
    search = OpenAIVectorStoreSearch(query="alpha")
    search.ranking_options = "nope"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="object"):
        search._valid()
    mixed = OpenAIEmbeddingRequest(input=["hello"])
    mixed.input = ["hello", 1]  # type: ignore[assignment]
    with pytest.raises(ValueError, match="mixture"):
        mixed._input_shape()

    events = list(
        _response_output_item_events(
            text="hi",
            item_id="item_1",
            call_items=None,
            search_items=None,
            logprobs=[{"token": "hi"}],
            final_status="completed",
        )
    )
    assert any(
        name == "response.content_part.done" and "logprobs" in event["part"]
        for name, event in events
    )

    store = OpenAIEnvelopeStore(2)
    with pytest.raises(ValueError, match="string 'id'"):
        store.put({})
    with pytest.raises(ValueError, match="string 'id'"):
        store.put_unless_status({}, forbidden=())
    with pytest.raises(ValueError, match="string 'id'"):
        store.repin({})
    with pytest.raises(ValueError, match="string 'id'"):
        store.put_if_present({})
    with pytest.raises(ValueError, match="must be an object"):
        OpenAIEnvelopeStore._validate_op("nope")  # type: ignore[arg-type]
    store.put({"id": "resp_1", "status": "queued"})
    assert store.put_unless_status(
        {"id": "resp_1", "status": "completed"}, forbidden=("cancelled",), require_existing=True
    )
    assert store.repin({"id": "resp_1", "status": "older"})["status"] == "completed"
    assert store.put_if_present({"id": "resp_missing", "status": "completed"}) is False
