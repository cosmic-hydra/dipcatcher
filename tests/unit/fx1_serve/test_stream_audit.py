"""Tests for fx1.serve.stream_audit."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from fx1.serve.stream_audit import (
    _ECHO_PROMPT,
    _ENV_KEYS,
    _chat_body,
    _make_ctx,
    _probe_buffered_reads,
    _StubBackend,
    stream_audit,
    stream_audit_bench,
)
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_KEYS:
        monkeypatch.delenv(name, raising=False)


def test_all_probes_hold() -> None:
    results = stream_audit()
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies() -> None:
    blob = stream_audit_bench()
    assert blob["claim"]["ok"] is True
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    a = stream_audit_bench()
    b = stream_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]


def test_buffered_prefix_skips_leading_keepalive() -> None:
    """A slow grace window emits keepalive comments the replay does not repeat."""

    class _SlowStub(_StubBackend):
        def complete(self, messages: list[dict[str, object]], *, sampling: object = None) -> str:
            time.sleep(0.2)
            return super().complete(messages, sampling=sampling)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ctx = _make_ctx(root / "a", _SlowStub(), sse_keepalive_s=0.05)
        out = _probe_buffered_reads(ctx, root / "k")
    assert out["buffered_prefix_matches_replay"] is True


def test_idempotent_chat_stream_pins_envelope_created(monkeypatch: pytest.MonkeyPatch) -> None:
    """A keyed replay stays byte-identical when the clock moves.

    The live SSE path used to call ``time.time()`` again after the envelope
    was stored. Crossing a second boundary changed ``created`` on the
    original frames, so the replay (which pins the envelope timestamp) no
    longer started with the bytes a partial reader had already seen.
    """
    calls = {"n": 0}

    def skewed() -> float:
        calls["n"] += 1
        return 1_700_000_000 + calls["n"] * 5

    monkeypatch.setattr(time, "time", skewed)
    with tempfile.TemporaryDirectory() as td:
        ctx = _make_ctx(Path(td) / "a", _StubBackend())
        body = _chat_body(_ECHO_PROMPT, stream=True)
        headers = {"Idempotency-Key": "created-pin"}
        first = ctx.client.post("/v1/chat/completions", json=body, headers=headers)
        second = ctx.client.post("/v1/chat/completions", json=body, headers=headers)
    assert first.status_code == 200
    assert second.headers.get("x-fx1-idempotent-replay") == "true"
    assert first.text == second.text
    assert calls["n"] > 1


def test_empty_audit_result_is_not_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fx1.serve.stream_audit.stream_audit", lambda: {})
    blob = stream_audit_bench()
    assert blob["claim"]["ok"] is False
    assert "no_probes" in blob["interpretation"]
