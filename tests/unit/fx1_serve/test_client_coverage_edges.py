"""Line and branch edges for fx1.serve.client left open by the fx1 run."""

from __future__ import annotations

import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from fx1.honesty import Fx1HonestyError
from fx1.serve.client import (
    HarnessClient,
    HarnessJobError,
    HarnessTransportError,
    _fx1_opts,
    _urllib_transport,
)

_FX = {
    "backend": "byok",
    "byok": {"base_url": "https://example.com/v1", "api_key": "k", "model": "m"},
    "checkpoint_dir": "/tmp/ckpt",
    "fallbacks": ["local_fx1"],
    "receipt_hashes": ["abc"],
    "timeout_s": 1.5,
}
_BUCKET = {
    "requests": 1,
    "ok": 1,
    "errors": 0,
    "usage_reported": 1,
    "prompt_tokens": 2,
    "completion_tokens": 3,
    "total_tokens": 5,
    "other_usage": {},
    "mean_latency_ms": 1.0,
}
_USAGE = {
    "generated_at": 1.0,
    "since": None,
    "until": None,
    "backend": None,
    "model": None,
    "key_id": None,
    "records_seen": 1,
    "records_dropped": 0,
    "ring_cap": 10,
    "totals": _BUCKET,
    "by_backend": {},
    "by_model": {},
    "by_key": {},
}
_CALLS: list[tuple[str, str, Any, dict[str, str]]] = []


def _transport(
    method: str,
    url: str,
    payload: dict[str, Any] | bytes | None,
    headers: dict[str, str],
    timeout_s: float,
) -> tuple[int, dict[str, str], bytes]:
    _CALLS.append((method, url, payload, dict(headers)))
    path = urllib.parse.urlparse(url).path
    hdrs = {"X-Fx1-Api-Version": "9", "X-Fx1-Completion-Id": "cid-1"}
    if isinstance(payload, dict) and payload.get("stream"):
        if path == "/v1/messages":
            raw = b'data: {"type": "message_stop"}\n\n'
        elif path == "/v1/responses":
            raw = b'data: {"type": "response.completed"}\n\n'
        else:
            raw = b'data: {"id": "chunk"}\n\ndata: [DONE]\n\n'
        return 200, hdrs, raw
    if path.endswith("/events") and "/harness/jobs/" in path:
        return 200, hdrs, b'data: {"status": "cancelled", "error": "stopped"}\n'
    if path.endswith("/results"):
        return 200, hdrs, b'{"custom_id": "a", "result": {"type": "succeeded"}}\n'
    if path == "/harness/completions":
        return (
            200,
            hdrs,
            json.dumps(
                {
                    "items": [
                        {
                            "completion_id": "c1",
                            "backend": "byok",
                            "ok": True,
                            "latency_ms": 1.0,
                            "at": 1.0,
                            "prompt_sha256": "p" * 32,
                            "attempts": [],
                        }
                    ]
                }
            ).encode(),
        )
    if "/harness/usage" in path:
        return 200, hdrs, json.dumps(_USAGE).encode()
    if path.endswith("/complete/batch"):
        return (
            200,
            hdrs,
            json.dumps(
                {
                    "backend": "local_fx1",
                    "model": "fx1",
                    "receipt_hashes": [],
                    "results": [{"ok": False, "error": "backend down", "error_class": "transport"}],
                }
            ).encode(),
        )
    body: dict[str, Any] = {"status": "succeeded", "id": "obj-1", "object": "list", "data": []}
    if "ftjob-failed" in url:
        body = {"status": "failed", "error": {"message": "disk full"}}
    elif "ftjob-cancelled" in url:
        body = {"status": "cancelled"}
    elif "batch-expired" in url:
        body = {"status": "expired", "id": "batch-expired"}
    return 200, hdrs, json.dumps(body).encode()


def _client() -> HarnessClient:
    _CALLS.clear()
    return HarnessClient(
        "http://harness.test",
        transport=_transport,
        sleep=lambda _seconds: None,
        clock=lambda: 0.0,
    )


def _last() -> tuple[str, str, Any, dict[str, str]]:
    return _CALLS[-1]


def test_urllib_transport_posts_raw_bytes() -> None:
    seen: dict[str, Any] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            count = int(self.headers.get("Content-Length", "0"))
            seen["type"] = self.headers.get("Content-Type")
            seen["body"] = self.rfile.read(count)
            self.send_response(201)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(seen["body"])))
            self.end_headers()
            self.wfile.write(seen["body"])

        def log_message(self, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = int(server.server_address[1])
        status, _headers, body = _urllib_transport(
            "POST",
            f"http://127.0.0.1:{port}/up",
            b"raw-bytes",
            {"Content-Type": "application/octet-stream"},
            2.0,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert status == 201
    assert body == b"raw-bytes"
    assert seen["type"] == "application/octet-stream"


def test_map_error_ignores_a_non_string_openai_message() -> None:
    body = json.dumps({"error": {"message": ["nope"], "code": "bad_code"}}).encode()
    err = HarnessClient._map_error(400, body)
    assert isinstance(err, HarnessTransportError)
    assert err.code == "bad_code"
    assert "nope" in str(err)


def test_fx1_opts_keeps_only_set_knobs() -> None:
    opts = _fx1_opts(
        backend="byok",
        byok={"base_url": "https://example.com"},
        checkpoint_dir="/tmp/ckpt",
        fallbacks=["local_fx1"],
        receipt_hashes=["abc"],
        timeout_s=1.5,
    )
    assert opts == {
        "backend": "byok",
        "byok": {"base_url": "https://example.com"},
        "checkpoint_dir": "/tmp/ckpt",
        "fallbacks": ["local_fx1"],
        "receipt_hashes": ["abc"],
        "timeout_s": 1.5,
    }
    assert (
        _fx1_opts(
            backend=None,
            byok=None,
            checkpoint_dir=None,
            fallbacks=None,
            receipt_hashes=None,
            timeout_s=None,
        )
        == {}
    )


def test_eval_and_finetune_optional_fields_reach_the_wire() -> None:
    client = _client()
    submitted = client.submit_eval(
        "suite",
        checkpoint_dir="/tmp/ckpt",
        byok={"base_url": "https://example.com"},
        timeout_s=2.0,
        fallbacks=["local_fx1"],
        judge_backend="hosted_k3",
        judge_byok={"api_key": "j"},
        callback_url="https://example.com/hook",
        callback_secret="sek",
        idempotency_key="eval-key",
    )
    method, url, payload, headers = _last()
    assert method == "POST" and url.endswith("/harness/evals")
    assert payload["checkpoint_dir"] == "/tmp/ckpt"
    assert payload["judge_backend"] == "hosted_k3"
    assert payload["callback_secret"] == "sek"
    assert headers["Idempotency-Key"] == "eval-key"
    assert submitted["status"] == "succeeded"

    listed = client.list_evals(status="succeeded", suite="suite")
    assert "status=succeeded" in _last()[1] and "suite=suite" in _last()[1]
    assert listed["object"] == "list"

    assert client.cancel_eval("ev-1")["id"] == "obj-1"
    assert _last()[0] == "DELETE"

    meta_only = client.eval_spec_create("named-meta", suite="suite", metadata={"k": "v"})
    assert meta_only["id"] == "obj-1"
    assert "testing_criteria" not in _last()[2]
    assert _last()[2]["metadata"] == {"k": "v"}
    spec = client.eval_spec_create(
        "named",
        suite="suite",
        backend="local_fx1",
        fallbacks=["byok"],
        checkpoint_dir="/tmp/ckpt",
        judge_backend="hosted_k3",
        timeout_s=3.0,
        testing_criteria=[{"type": "score"}],
        metadata={"k": "v"},
    )
    assert spec["id"] == "obj-1"
    body = _last()[2]
    assert body["data_source_config"]["item_schema"]["backend"] == "local_fx1"
    assert body["metadata"] == {"k": "v"}
    page = client.eval_specs(after="ev_prev")
    assert "after=ev_prev" in _last()[1]
    assert page["data"] == []
    bare = client.eval_specs()
    assert "after=" not in _last()[1]
    assert bare["object"] == "list"
    got = client.eval_spec_get("ev_1")
    assert got["id"] == "obj-1" and _last()[1].endswith("/v1/evals/ev_1")
    updated = client.eval_spec_update("ev_1", metadata={"k": "v"})
    assert updated["id"] == "obj-1" and _last()[2]["metadata"] == {"k": "v"}
    run = client.eval_run_create(
        "ev_1",
        model="fx1",
        data_source={"type": "custom"},
        byok={"api_key": "k"},
        judge_byok={"api_key": "j"},
        metadata={"k": "v"},
        callback_url="https://example.com/hook",
        callback_secret="sek",
        idempotency_key="run-key",
    )
    assert run["id"] == "obj-1"
    assert _last()[2]["callback_secret"] == "sek"
    assert _last()[3]["Idempotency-Key"] == "run-key"
    assert client.eval_run_cancel("ev_1", "run_1")["status"] == "succeeded"

    job = client.create_finetune_job(
        model="fx1",
        training_file="file-1",
        hyperparameters={"n_epochs": 1},
        suffix="sfx",
        validation_file="file-2",
        seed=7,
        metadata={"k": "v"},
        callback_url="https://example.com/ft",
        callback_secret="sek",
    )
    assert job["id"] == "obj-1"
    assert _last()[2]["suffix"] == "sfx" and _last()[2]["seed"] == 7
    assert client.cancel_finetune_job("ftjob-1")["status"] == "succeeded"
    with pytest.raises(HarnessJobError, match="disk full"):
        client.wait_finetune_job("ftjob-failed", poll_s=0.0, timeout_s=1.0)
    with pytest.raises(HarnessJobError, match="cancelled"):
        client.wait_finetune_job("ftjob-cancelled", poll_s=0.0, timeout_s=1.0)
    with pytest.raises(HarnessJobError, match="cancelled"):
        client.wait_run_stream("job-1")


def test_usage_keys_and_batch_failures() -> None:
    client = _client()
    rows = client.completions(backend="byok")
    assert [row.completion_id for row in rows] == ["c1"]
    assert rows[0].backend == "byok"
    assert "backend=byok" in _last()[1]
    unfiltered = client.usage()
    assert unfiltered.records_seen == 1
    assert urllib.parse.urlparse(_last()[1]).query == ""
    report = client.usage(backend="byok", model="m", key_id="key_1", since=1.0, until=2.0)
    assert report.records_seen == 1
    assert report.totals.total_tokens == 5
    query = urllib.parse.parse_qs(urllib.parse.urlparse(_last()[1]).query)
    assert query["backend"] == ["byok"] and query["key_id"] == ["key_1"]
    rotated = client.key_rotate("key_1", name="next", ttl_s=60.0)
    assert rotated["id"] == "obj-1"
    assert _last()[2]["name"] == "next" and _last()[2]["ttl_s"] == 60.0

    assert client.complete_many([]) == []
    with pytest.raises(ValueError, match="max_workers"):
        client.complete_many([[{"role": "user", "content": "hi"}]], max_workers=0)
    with pytest.raises(RuntimeError, match="transport: backend down"):
        client.complete_many([[{"role": "user", "content": "hi"}]])

    with pytest.raises(HarnessJobError, match="expired"):
        client.wait_batch("batch-expired", poll_s=0.0, timeout_s=1.0)


def test_openai_and_anthropic_optional_headers() -> None:
    client = _client()
    messages = [{"role": "user", "content": "hi"}]
    envelope, completion_id = client.chat_completion(
        messages,
        **_FX,
        idempotency_key="chat-key",
        extra_headers={"X-Trace": "1"},
    )
    assert completion_id == "cid-1"
    assert envelope["id"] == "obj-1"
    assert _last()[2]["fx1"]["byok"]["model"] == "m"
    assert _last()[3]["Idempotency-Key"] == "chat-key"
    assert _last()[3]["X-Trace"] == "1"
    bare, bare_id = client.chat_completion(messages)
    assert bare_id == "cid-1" and "fx1" not in _last()[2]

    chunks, chunk_id = client.chat_completion_stream(
        messages,
        **_FX,
        idempotency_key="chat-stream",
        last_event_id=3,
        extra_headers={"X-Trace": "1"},
    )
    assert chunks and chunks[0]["id"] == "chunk" and chunk_id == "cid-1"
    assert _last()[2]["fx1"]["receipt_hashes"] == ["abc"]
    assert _last()[3]["Last-Event-ID"] == "3"
    bare_chunks, _bare = client.chat_completion_stream(messages)
    assert bare_chunks[0]["id"] == "chunk" and "fx1" not in _last()[2]

    text, text_id = client.create_completion(
        "hi", **_FX, idempotency_key="text-key", extra_headers={"X-Trace": "1"}
    )
    assert text_id == "cid-1" and text["object"] == "list"
    assert _last()[2]["fx1"]["checkpoint_dir"] == "/tmp/ckpt"
    plain, _plain_id = client.create_completion("hi")
    assert plain["id"] == "obj-1" and "fx1" not in _last()[2]
    streamed, stream_id = client.create_completion_stream(
        "hi",
        **_FX,
        idempotency_key="text-stream",
        last_event_id=1,
        extra_headers={"X-Trace": "1"},
    )
    assert stream_id == "cid-1" and streamed[0]["id"] == "chunk"
    assert _last()[3]["Last-Event-ID"] == "1"
    client.create_completion_stream("hi")
    assert "fx1" not in _last()[2] and "Last-Event-ID" not in _last()[3]

    message, message_id = client.create_message(
        messages, **_FX, idempotency_key="msg-key", extra_headers={"X-Trace": "1"}
    )
    assert message_id == "cid-1" and message["id"] == "obj-1"
    assert _last()[3]["Idempotency-Key"] == "msg-key"
    client.create_message(messages)
    assert "fx1" not in _last()[2]
    events, event_id = client.create_message_stream(
        messages,
        **_FX,
        idempotency_key="msg-stream",
        last_event_id=4,
        extra_headers={"X-Trace": "1"},
    )
    assert event_id == "cid-1" and events[0]["type"] == "message_stop"
    assert _last()[3]["Last-Event-ID"] == "4"

    def empty_stream(
        method: str,
        url: str,
        payload: dict[str, Any] | bytes | None,
        headers: dict[str, str],
        timeout_s: float,
    ) -> tuple[int, dict[str, str], bytes]:
        return 200, {"X-Fx1-Completion-Id": "cid-1"}, b'data: {"type": "ping"}\n\n'

    empty = HarnessClient("http://harness.test", transport=empty_stream)
    with pytest.raises(HarnessTransportError, match="without message_stop"):
        empty.create_message_stream(messages)

    bare_batch = client.create_message_batch([{"custom_id": "a", "params": {}}])
    assert bare_batch["id"] == "obj-1"
    assert _last()[2] == {"requests": [{"custom_id": "a", "params": {}}]}
    assert "Idempotency-Key" not in _last()[3]
    batch = client.create_message_batch(
        [{"custom_id": "a", "params": {}}],
        callback_url="https://example.com/hook",
        callback_secret="sek",
        idempotency_key="batch-key",
        extra_headers={"X-Trace": "1"},
    )
    assert batch["id"] == "obj-1"
    assert _last()[2]["callback_secret"] == "sek"
    assert _last()[3]["Idempotency-Key"] == "batch-key"
    bare_page = client.message_batches()
    assert bare_page["object"] == "list"
    assert "after_id=" not in _last()[1] and "before_id=" not in _last()[1]
    page = client.message_batches(after_id="mb_a", before_id="mb_b")
    assert "after_id=mb_a" in _last()[1] and "before_id=mb_b" in _last()[1]
    assert page["object"] == "list"
    assert client.cancel_message_batch("mb_1")["id"] == "obj-1"
    assert client.delete_message_batch("mb_1")["id"] == "obj-1"
    rows = client.message_batch_results("mb_1")
    assert rows == [{"custom_id": "a", "result": {"type": "succeeded"}}]
    models = client.anthropic_models(limit=5, after_id="m1", before_id="m2")
    assert models["data"] == []
    assert "limit=5" in _last()[1] and "after_id=m1" in _last()[1]

    response, response_id = client.responses_create(
        "hi",
        **_FX,
        reasoning_effort="low",
        text_format={"type": "text"},
        verbosity="low",
        tools=[{"type": "function"}],
        tool_choice="auto",
        parallel_tool_calls=False,
        include=["message.output_text.logprobs"],
        top_logprobs=2,
        idempotency_key="resp-key",
        extra_headers={"X-Trace": "1"},
    )
    assert response_id == "cid-1" and response["id"] == "obj-1"
    sent = _last()[2]
    assert sent["reasoning"] == {"effort": "low"}
    assert sent["text"]["format"]["type"] == "text"
    assert sent["text"]["verbosity"] == "low"
    assert sent["top_logprobs"] == 2
    assert _last()[3]["Idempotency-Key"] == "resp-key"
    format_only, _fid = client.responses_create("hi", text_format={"type": "text"})
    assert format_only["id"] == "obj-1"
    assert "format" in _last()[2]["text"] and "verbosity" not in _last()[2]["text"]
    verbose, _vid = client.responses_create("hi", verbosity="high")
    assert verbose["id"] == "obj-1"
    assert _last()[2]["text"] == {"verbosity": "high"}
    streamed_resp, streamed_id = client.responses_create_stream(
        "hi",
        idempotency_key="resp-stream",
        last_event_id=2,
        extra_headers={"X-Trace": "1"},
        **_FX,
    )
    assert streamed_id == "cid-1"
    assert streamed_resp[0]["type"] == "response.completed"
    assert _last()[3]["Last-Event-ID"] == "2"
    client.responses_create_stream("hi")
    assert "Idempotency-Key" not in _last()[3] and "Last-Event-ID" not in _last()[3]

    vectors, vector_id = client.embeddings_create(
        "hi",
        **{k: v for k, v in _FX.items() if k != "receipt_hashes"},
        extra_headers={"X-Trace": "1"},
    )
    assert vector_id == "cid-1" and vectors["object"] == "list"
    assert _last()[2]["fx1"]["backend"] == "byok"
    plain_vectors, _plain_vector = client.embeddings_create("hi")
    assert plain_vectors["id"] == "obj-1" and "fx1" not in _last()[2]


def test_listing_cursors_and_vector_store_fields() -> None:
    client = _client()
    made = client.create_batch(
        "file-1",
        metadata={"k": "v"},
        idempotency_key="batch-key",
        callback_url="https://example.com/hook",
        callback_secret="sek",
    )
    assert made["id"] == "obj-1"
    assert _last()[2]["metadata"] == {"k": "v"}
    assert _last()[3]["Idempotency-Key"] == "batch-key"
    page = client.batches(after="batch_prev")
    assert page["object"] == "list"
    assert "after=batch_prev" in _last()[1]

    updated = client.update_chat_completion("chatcmpl-1", metadata={"k": "v"})
    assert updated["metadata"] == {"k": "v"} if "metadata" in updated else updated["id"] == "obj-1"
    assert _last()[2]["metadata"] == {"k": "v"}
    assert client.delete_chat_completion("chatcmpl-1")["id"] == "obj-1"
    listed = client.list_chat_completions(after="c1", before="c2", metadata={"k": "v"})
    assert "after=c1" in _last()[1] and "before=c2" in _last()[1]
    assert listed["object"] == "list"
    messages = client.chat_completion_messages("c1", after="m1", before="m2")
    assert "after=m1" in _last()[1] and "before=m2" in _last()[1]
    assert messages["data"] == []
    items = client.response_input_items("resp_1", after="i1", before="i2")
    assert "after=i1" in _last()[1] and "before=i2" in _last()[1]
    assert items["object"] == "list"

    items_only = client.conversation_create(items=[{"type": "message"}])
    assert items_only["id"] == "obj-1" and "metadata" not in _last()[2]
    created = client.conversation_create(items=[{"type": "message"}], metadata={"k": "v"})
    assert created["id"] == "obj-1" and _last()[2]["items"][0]["type"] == "message"
    patched = client.conversation_update("conv_1", metadata={"k": "v"})
    assert patched["id"] == "obj-1"
    page = client.conversation_items("conv_1", after="it1", before="it2")
    assert "after=it1" in _last()[1] and "before=it2" in _last()[1]
    assert page["data"] == []
    added = client.conversation_items_add("conv_1", [{"type": "message"}])
    assert added["id"] == "obj-1"
    deleted = client.conversation_item_delete("conv_1", "item_1")
    assert deleted["id"] == "obj-1"

    store = client.vector_store_create(
        name="docs",
        metadata={"k": "v"},
        expires_after={"anchor": "last_active_at", "days": 7},
    )
    assert store["id"] == "obj-1"
    assert _last()[2]["expires_after"]["days"] == 7
    renamed = client.vector_store_update(
        "vs_1",
        metadata={"k": "v"},
        expires_after={"anchor": "last_active_at", "days": 3},
    )
    assert renamed["id"] == "obj-1"
    stores = client.vector_store_list(after="vs_a", before="vs_b")
    assert "after=vs_a" in _last()[1] and "before=vs_b" in _last()[1]
    assert stores["object"] == "list"
    attached = client.vector_store_file_create("vs_1", "file-1", chunking_strategy={"type": "auto"})
    assert attached["id"] == "obj-1"
    assert _last()[2]["chunking_strategy"] == {"type": "auto"}
    files = client.vector_store_file_list("vs_1", after="f1", before="f2")
    assert "after=f1" in _last()[1] and "before=f2" in _last()[1]
    assert files["data"] == []
    found = client.vector_store_search(
        "vs_1",
        "alpha",
        filters={"type": "eq", "key": "k", "value": "v"},
        ranking_options={"ranker": "auto"},
    )
    assert found["id"] == "obj-1"
    assert _last()[2]["ranking_options"]["ranker"] == "auto"
    batch = client.vector_store_file_batch_create(
        "vs_1",
        ["file-1"],
        attributes={"k": "v"},
        chunking_strategy={"type": "auto"},
    )
    assert batch["id"] == "obj-1"
    assert _last()[2]["attributes"] == {"k": "v"}
    members = client.vector_store_file_batch_files("vs_1", "vsfb_1", after="f1", before="f2")
    assert "after=f1" in _last()[1] and "before=f2" in _last()[1]
    assert members["object"] == "list"
    named = client.vector_store_update("vs_1", name="docs")
    assert named["id"] == "obj-1"
    assert _last()[2]["name"] == "docs"
    attributed = client.vector_store_file_create("vs_1", "file-1", attributes={"k": "v"})
    assert attributed["id"] == "obj-1"
    assert _last()[2]["attributes"] == {"k": "v"}
    client.eval_spec_update("ev_1", name="renamed")
    assert _last()[2]["name"] == "renamed"
    done = client.wait_finetune_job("ftjob-ok", poll_s=0.0, timeout_s=1.0)
    assert done["status"] == "succeeded"


def test_batch_and_stream_terminal_failures() -> None:
    def transport(
        method: str,
        url: str,
        payload: dict[str, Any] | bytes | None,
        headers: dict[str, str],
        timeout_s: float,
    ) -> tuple[int, dict[str, str], bytes]:
        del method, payload, headers, timeout_s
        if url.endswith("/complete/batch"):
            body = {
                "backend": "local_fx1",
                "model": "fx1",
                "results": [
                    {
                        "ok": False,
                        "error": "refused",
                        "error_class": "honesty_refusal",
                    }
                ],
            }
            return 200, {}, json.dumps(body).encode()
        if "/events" in url:
            raw = b'data: {"status": "failed", "error": "disk"}\n'
            return 200, {}, raw
        if "/v1/batches/" in url:
            return 200, {}, b'{"status": "completed", "id": "batch_1"}'
        return 200, {}, b'{"status": "succeeded", "id": "obj-1"}'

    client = HarnessClient("http://harness.test", transport=transport, sleep=lambda _s: None)
    with pytest.raises(Fx1HonestyError, match="refused"):
        client.complete_many([[{"role": "user", "content": "hi"}]])
    with pytest.raises(HarnessJobError, match="failed"):
        client.wait_run_stream("job-1")
    finished = client.wait_batch("batch_1", poll_s=0.0, timeout_s=1.0)
    assert finished["status"] == "completed"
