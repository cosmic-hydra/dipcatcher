"""Line and branch edges for the remaining fx1.serve modules."""

from __future__ import annotations

import hashlib
import ipaddress
import os
import urllib.parse
from pathlib import Path
from typing import Any

import pytest

from fx1.serve import journal as journal_mod
from fx1.serve.anthropic_compat import (
    AnthropicCountTokensRequest,
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicTool,
    AnthropicToolChoice,
    _anthropic_message_contract,
    _messages_to_openai,
    _tool_result_content,
    anthropic_count_messages,
    anthropic_to_openai,
)
from fx1.serve.attestation import (
    TEEQuote,
    attestation_ladder_status,
    verify_quote,
)
from fx1.serve.journal import JobJournal, _fsync_parent
from fx1.serve.keys import ApiKeyStore, KeyStoreError
from fx1.serve.ops_receipt import job_record_receipt, run_result_receipt
from fx1.serve.uploads import UploadMeta, UploadStore, UploadStoreError, upload_status
from fx1.serve.usage_report import aggregate_usage
from fx1.serve.webhooks import _is_public_unicast, _post_once, deliver_signed


class _Usage:
    def __init__(self, usage: dict[str, Any] | None) -> None:
        self.ok = True
        self.usage = usage
        self.latency_ms = 1.0
        self.backend = "hosted_k3"
        self.model = "m"
        self.key_id = "k"
        self.at = 1.0


def _tool(name: str = "lookup") -> AnthropicTool:
    return AnthropicTool(name=name, input_schema={"type": "object"})


def _quote(*, report: str, sha: str) -> TEEQuote:
    return TEEQuote(
        platform="simulated",
        checkpoint_sha256=sha,
        measurement="m" * 8,
        report_data=report,
        signature="sig",
    )


def test_anthropic_shapes_fail_closed() -> None:
    with pytest.raises(ValueError, match="input_schema"):
        AnthropicTool(name="lookup", input_schema={"type": "string"})
    with pytest.raises(ValueError, match="only applies"):
        AnthropicToolChoice(type="auto", name="lookup")
    with pytest.raises(ValueError, match="non-empty"):
        AnthropicMessage(role="user", content=[])
    with pytest.raises(ValueError, match="string 'text'"):
        AnthropicMessage(role="user", content=[{"type": "text", "text": 1}])
    with pytest.raises(ValueError, match="need an 'id'"):
        AnthropicMessage(
            role="assistant",
            content=[{"type": "tool_use", "id": "", "name": "lookup", "input": {}}],
        )
    with pytest.raises(ValueError, match="need a 'name'"):
        AnthropicMessage(
            role="assistant",
            content=[{"type": "tool_use", "id": "t1", "name": "", "input": {}}],
        )
    with pytest.raises(ValueError, match="need an 'input'"):
        AnthropicMessage(
            role="assistant", content=[{"type": "tool_use", "id": "t1", "name": "lookup"}]
        )
    with pytest.raises(ValueError, match="tool_use_id"):
        AnthropicMessage(role="user", content=[{"type": "tool_result", "tool_use_id": ""}])

    class _Flip:
        def __init__(self) -> None:
            self.reads = 0
            self.content = [{"type": "tool_use", "id": "t1", "name": "lookup", "input": {}}]

        @property
        def role(self) -> str:
            self.reads += 1
            return "assistant" if self.reads == 1 else "user"

    with pytest.raises(ValueError, match="belong on assistant"):
        AnthropicMessage._blocks_valid(_Flip())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-empty text-block"):
        _anthropic_message_contract(
            [AnthropicMessage(role="user", content="hi")],
            [],
            None,
            None,
        )
    with pytest.raises(ValueError, match="cache_control"):
        _anthropic_message_contract(
            [AnthropicMessage(role="user", content="hi")],
            [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
            None,
            None,
        )
    tools = [_tool(f"t{i}") for i in range(129)]
    with pytest.raises(ValueError, match="128"):
        _anthropic_message_contract(
            [AnthropicMessage(role="user", content="hi")], None, tools, None
        )

    assert _tool_result_content({}) == ""
    calls_only = AnthropicMessage(role="assistant", content="x")
    calls_only.content = []
    with pytest.raises(Exception, match="no text or tool_use"):
        _messages_to_openai([calls_only])
    mixed = AnthropicMessage(
        role="user",
        content=[
            {"type": "text", "text": "before"},
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
            {"type": "text", "text": "after"},
        ],
    )
    translated = _messages_to_openai([mixed])
    assert translated[0] == {"role": "user", "content": "before"}
    assert translated[1]["role"] == "tool"
    counted = anthropic_count_messages(
        AnthropicCountTokensRequest(
            messages=[AnthropicMessage(role="user", content="hi")],
            system="be brief",
        )
    )
    assert counted[0]["role"] == "system"
    translated_req = anthropic_to_openai(
        AnthropicMessagesRequest(
            messages=[AnthropicMessage(role="user", content="hi")],
            max_tokens=8,
            system="be brief",
        )
    )
    assert translated_req["messages"][0]["role"] == "system"
    blank = anthropic_count_messages(
        AnthropicCountTokensRequest(
            messages=[AnthropicMessage(role="user", content="hi")],
            system="",
        )
    )
    assert blank[0]["role"] == "user"
    blank_req = anthropic_to_openai(
        AnthropicMessagesRequest(
            messages=[AnthropicMessage(role="user", content="hi")],
            max_tokens=8,
            system="",
        )
    )
    assert blank_req["messages"][0]["role"] == "user"
    empty_user = AnthropicMessage(role="user", content="x")
    empty_user.content = []
    with pytest.raises(Exception, match="no text or tool_result"):
        _messages_to_openai([empty_user])


def test_keys_reject_bad_bounds_and_a_full_store(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_keys"):
        ApiKeyStore(0)
    bare = ApiKeyStore(1)
    bare._compact_locked()
    assert bare.window_state("missing") is None
    raw, rec = bare.mint(name="only")
    assert raw.startswith("fx1k_")
    with pytest.raises(KeyStoreError, match="full"):
        bare.rotate(rec["key_id"])
    with pytest.raises(ValueError, match="max_requests"):
        bare.update(rec["key_id"], max_requests=0)
    with pytest.raises(ValueError, match="max_tokens"):
        bare.update(rec["key_id"], max_tokens=0)
    with pytest.raises(ValueError, match="cannot clear"):
        bare.update(rec["key_id"], clear=["scopes"])
    journaled = ApiKeyStore(2, JobJournal(tmp_path / "keys.jsonl"))
    _raw, kept = journaled.mint(name="kept")
    assert journaled.get(kept["key_id"])["name"] == "kept"
    bare._restore_payload({"record": {"sha256": "ab" * 32, "key_id": None, "admin": False}})
    bare.charge_tokens("missing", 4)
    assert bare.get(rec["key_id"])["tokens_used"] == 0


def test_upload_replay_expiry_and_part_faults(tmp_path: Path) -> None:
    expired = UploadMeta(
        upload_id="upload_old",
        filename="a.txt",
        purpose="batch",
        nbytes=1,
        mime_type="text/plain",
        created_at=1,
        expires_at=1,
        status="pending",
    )
    assert upload_status(expired, now=2) == "expired"

    state = tmp_path / "state"
    uid = "upload_live"
    udir = state / "uploads" / uid
    udir.mkdir(parents=True)
    (udir / "part_real.bin").write_bytes(b"hi")
    (udir / "orphan.bin").write_bytes(b"x")
    (state / "uploads" / "note.txt").write_text("x", encoding="utf-8")
    meta = UploadMeta(
        upload_id=uid,
        filename="a.txt",
        purpose="batch",
        nbytes=2,
        mime_type="text/plain",
        created_at=1,
        expires_at=10**12,
        status="pending",
    )
    journal = JobJournal(state / "uploads.jsonl")
    journal.append({"upload": meta.model_dump()})
    journal.append({"evicted": ["upload_gone"]})
    journal.append({"upload_part": {"upload_id": "upload_missing", "part_id": "p", "bytes": 1}})
    journal.append({"upload_part": {"upload_id": uid, "part_id": "part_real", "bytes": 2}})
    UploadStore(4, 100, state)
    assert "orphan.bin" not in {path.name for path in udir.iterdir()}
    memory = UploadStore(2, 100)
    memory._compact_locked()
    with pytest.raises(RuntimeError, match="no state_dir"):
        memory._blob("upload_x", "part_x")
    intent = memory.create(purpose="batch", filename="z.txt", nbytes=2, mime_type="text/plain")
    memory._inmem.pop(intent.upload_id)
    with pytest.raises(KeyError):
        memory.add_part(intent.upload_id, b"ab")

    durable = UploadStore(2, 100, tmp_path / "fresh")
    created = durable.create(purpose="batch", filename="b.txt", nbytes=2, mime_type="text/plain")
    assert durable._journal is not None
    durable._journal.path.unlink()
    added = durable.add_part(created.upload_id, b"ab")
    assert added["object"] == "upload.part"

    stat_store = UploadStore(2, 100, tmp_path / "stat")
    pending = stat_store.create(purpose="batch", filename="c.txt", nbytes=2, mime_type="text/plain")

    class _Broken:
        def __init__(self, path: Path) -> None:
            self._path = path

        def stat(self) -> Any:
            raise OSError("stat refused")

        def __getattr__(self, name: str) -> Any:
            return getattr(self._path, name)

    assert stat_store._journal is not None
    stat_store._journal.path = _Broken(stat_store._journal.path)  # type: ignore[assignment]
    part = stat_store.add_part(pending.upload_id, b"ab")
    assert part["upload_id"] == pending.upload_id

    cleanup = UploadStore(2, 100, tmp_path / "cleanup")
    target = cleanup.create(purpose="batch", filename="d.txt", nbytes=2, mime_type="text/plain")
    assert cleanup._journal is not None
    calls = {"n": 0}
    real = cleanup._journal.path.stat

    class _Later:
        def stat(self) -> Any:
            calls["n"] += 1
            if calls["n"] > 1:
                raise OSError("later")
            return real()

    cleanup._journal.path = _Later()  # type: ignore[assignment]

    def refuse(_payload: dict[str, Any]) -> None:
        raise RuntimeError("append refused")

    cleanup._journal.append = refuse  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="append refused"):
        cleanup.add_part(target.upload_id, b"ab")
    assert calls["n"] >= 2

    with pytest.raises(UploadStoreError, match="empty") as empty:
        memory._assemble_locked(created, [])
    assert empty.value.code == "invalid_request"
    created.parts = {"part_missing": 2}
    durable_missing = UploadStore(2, 100, tmp_path / "missing-blob")
    live = durable_missing.create(
        purpose="batch", filename="e.txt", nbytes=2, mime_type="text/plain"
    )
    durable_missing.add_part(live.upload_id, b"ab")
    live.parts["part_ghost"] = 2
    with pytest.raises(UploadStoreError, match="content is missing") as missing:
        durable_missing._assemble_locked(live, ["part_ghost"])
    assert missing.value.code == "part_not_found"
    with pytest.raises(UploadStoreError, match="md5") as checksum:
        UploadStore._validate_complete(live, ["part_ghost"], b"ab", "00")
    assert checksum.value.code == "checksum_mismatch"


def test_webhooks_classify_mapped_addresses_and_hostless_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapped = ipaddress.ip_address("::ffff:1.1.1.1")
    loop = ipaddress.ip_address("::ffff:127.0.0.1")
    assert _is_public_unicast(mapped) is True  # type: ignore[arg-type]
    assert _is_public_unicast(loop) is False  # type: ignore[arg-type]
    hostless = urllib.parse.urlparse("http:///hook")
    with pytest.raises(ValueError, match="no host"):
        _post_once(hostless, "1.1.1.1", b"{}", {}, 0.1)
    monkeypatch.setattr("fx1.serve.webhooks.check_callback_url", lambda url: url)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    refused = deliver_signed("http:///hook", None, b"{}")
    assert refused[0] is False and "no host" in str(refused[1])
    monkeypatch.setattr("fx1.serve.webhooks._resolved_addresses", lambda host, port: ())
    empty = deliver_signed("http://1.1.1.1/hook", None, b"{}", max_attempts=2, backoff_s=0.001)
    assert empty == (False, None, 2)


def test_journal_skips_directory_fsync_off_posix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(journal_mod.os, "name", "nt")
    _fsync_parent(tmp_path)
    monkeypatch.undo()
    journal = JobJournal(tmp_path / "jobs.jsonl")
    journal.append({"n": 1})
    assert journal.replay().payloads == [{"n": 1}]


def test_attestation_fails_closed_on_unbound_quotes_and_bad_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sha = "ab" * 32
    nonce = "nonce-12"
    quote = _quote(report=nonce + "zzzz", sha=sha)
    assert verify_quote(quote, expected_checkpoint_sha256=sha, nonce=nonce) is False
    root = tmp_path / "ckpt"
    root.mkdir()

    def explode(_path: Path) -> bool:
        raise RuntimeError("signing key missing")

    monkeypatch.setattr("fx1.serve.signing.verify_release", explode)
    (root / "attestation.quote.json").write_text("{", encoding="utf-8")
    (root / "zkml.manifest.json").write_text("{", encoding="utf-8")
    status = attestation_ladder_status(root)
    assert status == {"signed_release": False, "tee": False, "selective_zkml": False}


def test_ops_and_usage_ignore_unbillable_claims() -> None:
    sealed = job_record_receipt(
        {"job_id": "j", "status": "succeeded", "callback_url": "https://example.com/h"}
    )
    digest = hashlib.sha256(b"https://example.com/h").hexdigest()
    assert sealed["record"]["callback_url_sha256"] == digest
    quiet = run_result_receipt({"exit_code": 1})
    assert quiet["record"]["ok"] is False
    assert "stdout_sha256" not in quiet["record"]
    report = aggregate_usage(
        [_Usage({"prompt_tokens": True, "completion_tokens": "no"})], cap=4, dropped=0
    )
    assert report.totals.prompt_tokens == 0
    assert report.totals.usage_reported == 1


def test_posix_name_is_restored() -> None:
    assert os.name == "posix"
