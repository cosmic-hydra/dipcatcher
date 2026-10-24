"""Line and branch edges for fx1.serve.local_engine left open by the fx1 run."""

from __future__ import annotations

import http.client
import json
import os
import runpy
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from fx1.serve import local_engine as le
from fx1.serve.local_engine import (
    EngineCompletion,
    LocalWeightsEngine,
    _EngineHTTPServer,
    _Handler,
    _load_safetensors,
    _stop_list,
    _stream_frames,
    _TinyLM,
    _watch_orphan,
    main,
    serve,
)

_BASE = Path(__file__).resolve().parents[3] / "artifacts" / "fx1_tiny_lm"
_MSGS = [{"role": "user", "content": "hi"}]


def _f32(*shape: int, fill: float = 0.01) -> np.ndarray:
    return np.full(shape, fill, dtype=np.float32)


def _embeddings(*, vocab: int = 8, dim: int = 4, ctx: int = 4) -> dict[str, np.ndarray]:
    return {
        "tok_emb.weight": _f32(vocab, dim),
        "pos_emb.weight": _f32(ctx, dim),
        "ln_f.weight": _f32(dim),
        "ln_f.bias": _f32(dim),
    }


@pytest.fixture(scope="module")
def engine() -> LocalWeightsEngine:
    return LocalWeightsEngine(_BASE)


def test_safetensors_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real = builtins.__import__

    def blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "safetensors.numpy":
            raise ImportError("safetensors blocked for the test")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="safetensors is required"):
        _load_safetensors(Path("unused.safetensors"))


def test_from_tensors_rejects_incomplete_or_malformed_weights() -> None:
    with pytest.raises(RuntimeError, match="missing tensor 'tok_emb.weight'"):
        _TinyLM.from_tensors({"pos_emb.weight": _f32(2, 2)})

    bad_rank = _embeddings()
    bad_rank["tok_emb.weight"] = _f32(8)
    with pytest.raises(RuntimeError, match="malformed embedding"):
        _TinyLM.from_tensors(bad_rank)

    mismatch = _embeddings(vocab=8, dim=4)
    mismatch["lm_head.weight"] = _f32(3, 4)
    with pytest.raises(RuntimeError, match="lm_head vocab"):
        _TinyLM.from_tensors(mismatch)

    with pytest.raises(RuntimeError, match="no transformer blocks"):
        _TinyLM.from_tensors(_embeddings())

    qkv = _embeddings()
    qkv["blocks.0.attn.qkv.weight"] = _f32(2, 2)
    with pytest.raises(RuntimeError, match="qkv projection"):
        _TinyLM.from_tensors(qkv)

    heads = _embeddings(vocab=4, dim=6, ctx=2)
    heads["blocks.0.attn.qkv.weight"] = _f32(18, 6)
    with pytest.raises(RuntimeError, match="does not divide"):
        _TinyLM.from_tensors(heads, n_heads=4)


def test_checkpoint_manifest_and_vocab_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing"
    shutil.copytree(_BASE, missing)
    (missing / "weights.manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="weights.manifest.json"):
        LocalWeightsEngine(missing)

    listed = tmp_path / "listed"
    shutil.copytree(_BASE, listed)
    (listed / "weights.manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not a JSON object"):
        LocalWeightsEngine(listed)

    empty_sha = tmp_path / "empty-sha"
    shutil.copytree(_BASE, empty_sha)
    manifest = json.loads((empty_sha / "weights.manifest.json").read_text())
    manifest["weights_sha256"] = ""
    (empty_sha / "weights.manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="non-empty 'weights_sha256'"):
        LocalWeightsEngine(empty_sha)

    escaped = tmp_path / "escaped"
    shutil.copytree(_BASE, escaped)
    manifest = json.loads((escaped / "weights.manifest.json").read_text())
    manifest["weights_file"] = "../weights.safetensors"
    (escaped / "weights.manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="not a checkpoint member"):
        LocalWeightsEngine(escaped)

    default_heads = tmp_path / "default-heads"
    shutil.copytree(_BASE, default_heads)
    manifest = json.loads((default_heads / "weights.manifest.json").read_text())
    manifest["arch"]["n_heads"] = "4"
    (default_heads / "weights.manifest.json").write_text(json.dumps(manifest))
    loaded = LocalWeightsEngine(default_heads)
    assert loaded.model.n_heads == 4

    real_load = le._load_safetensors

    def shrink(path: Path) -> dict[str, np.ndarray]:
        tensors = dict(real_load(path))
        tensors["tok_emb.weight"] = tensors["tok_emb.weight"][:10]
        tensors["lm_head.weight"] = tensors["lm_head.weight"][:10]
        return tensors

    monkeypatch.setattr(le, "_load_safetensors", shrink)
    with pytest.raises(RuntimeError, match="byte tokenizer"):
        LocalWeightsEngine(_BASE)


def test_prompt_sampling_and_stop(engine: LocalWeightsEngine) -> None:
    with pytest.raises(ValueError, match="string content"):
        engine._prompt_ids([{"role": "user", "content": 1}])
    ids = engine._prompt_ids([{"content": "hi"}])
    assert ids[0] == le.BOS_ID
    assert engine.model.detokenize(ids).startswith("user: hi")

    seeded = engine.complete_messages(_MSGS, max_tokens=2, temperature=0.8, seed=7)
    again = engine.complete_messages(_MSGS, max_tokens=2, temperature=0.8, seed=7)
    assert seeded.token_ids == again.token_ids
    assert seeded.finish_reason in {"stop", "length"}
    unseeded = engine.complete_messages(_MSGS, max_tokens=2, temperature=0.8)
    assert isinstance(unseeded.text, str)
    assert unseeded.completion_tokens == len(unseeded.token_ids)

    cut = engine.complete_messages(_MSGS, max_tokens=2, stop=["\x00NEVER", ""])
    assert cut.text == ""
    assert cut.finish_reason == "stop"


def test_stream_frames_partial_utf8_and_empty(engine: LocalWeightsEngine) -> None:
    split = EngineCompletion(
        text="",
        prompt_tokens=3,
        completion_tokens=3,
        finish_reason="stop",
        token_ids=[0xC3, 0xA9, 65],
    )
    body = _stream_frames(engine, split)
    assert b"data: [DONE]" in body
    assert "\\u00e9" in body.decode()
    assert body.count(b"data: ") >= 3

    empty = EngineCompletion(
        text="",
        prompt_tokens=1,
        completion_tokens=0,
        finish_reason="length",
        token_ids=[],
    )
    tail = _stream_frames(engine, empty)
    assert b"[DONE]" in tail
    assert b'"finish_reason": "length"' in tail

    assert _stop_list("END") == ["END"]
    assert _stop_list(["a", 1, "b"]) == ["a", "b"]
    assert _stop_list(None) == []


def _request(port: int, method: str, path: str, payload: bytes | None) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        headers = {}
        if payload is not None:
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(payload))
        conn.request(method, path, body=payload, headers=headers)
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def test_http_fail_closed_and_stream(engine: LocalWeightsEngine) -> None:
    srv = _EngineHTTPServer(("127.0.0.1", 0), _Handler)
    srv.engine = engine
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    port = int(srv.server_address[1])
    try:
        status, body = _request(port, "GET", "/nope", None)
        assert status == 404
        assert b"unknown route" in body

        status, models = _request(port, "GET", "/v1/models", None)
        assert status == 200
        assert json.loads(models)["data"][0]["id"] == engine.served_model

        status, body = _request(port, "POST", "/v1/tokenize", b"{")
        assert status == 400 and b"malformed JSON" in body

        status, body = _request(port, "POST", "/v1/tokenize", b"[1]")
        assert status == 400 and b"JSON object" in body

        status, body = _request(
            port,
            "POST",
            "/v1/tokenize",
            json.dumps({"messages": [{"content": 1}]}).encode(),
        )
        assert status == 400 and b"string content" in body

        status, counted = _request(
            port,
            "POST",
            "/v1/tokenize",
            json.dumps({"messages": _MSGS}).encode(),
        )
        assert status == 200
        assert json.loads(counted)["count"] > 0

        status, body = _request(port, "POST", "/v1/other", b"{}")
        assert status == 404 and b"unknown route" in body

        status, body = _request(port, "POST", "/v1/chat/completions", b"{}")
        assert status == 400 and b"non-empty list" in body

        status, body = _request(
            port,
            "POST",
            "/v1/chat/completions",
            json.dumps({"messages": [{"content": None}]}).encode(),
        )
        assert status == 400 and b"string content" in body

        status, streamed = _request(
            port,
            "POST",
            "/v1/chat/completions",
            json.dumps({"messages": _MSGS, "max_tokens": 1, "stream": True, "stop": "\n"}).encode(),
        )
        assert status == 200
        assert b"data: [DONE]" in streamed

        status, done = _request(
            port,
            "POST",
            "/v1/chat/completions",
            json.dumps(
                {
                    "messages": _MSGS,
                    "max_tokens": 1,
                    "temperature": 0.5,
                    "seed": 1,
                    "stop": ["nope", 3],
                }
            ).encode(),
        )
        assert status == 200
        payload = json.loads(done)
        assert payload["choices"][0]["finish_reason"] in {"stop", "length"}
        assert "content" in payload["choices"][0]["message"]
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=2)


def test_watch_orphan_sleeps_then_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    ppids = iter((4242, 1))
    monkeypatch.setattr(os, "getppid", lambda: next(ppids))
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: slept.append(seconds))
    monkeypatch.setattr(os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    with pytest.raises(SystemExit) as exc:
        _watch_orphan(4242)
    assert exc.value.code == 0
    assert slept == [1.0]


def test_serve_interrupt_and_clean_return(
    engine: LocalWeightsEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    watched: list[int] = []
    monkeypatch.setattr(le, "_watch_orphan", lambda ppid: watched.append(ppid))

    def interrupt(self: object, poll_interval: float = 0.5) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(_EngineHTTPServer, "serve_forever", interrupt)
    assert serve(engine, "127.0.0.1", 0, watch_orphan=True) == 0
    assert watched and watched[0] > 0

    monkeypatch.setattr(_EngineHTTPServer, "serve_forever", lambda self, poll_interval=0.5: None)
    assert serve(engine, "127.0.0.1", 0, watch_orphan=False) == 0


def test_main_checkpoint_required_and_serves(
    engine: LocalWeightsEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FX1_CHECKPOINT_DIR", raising=False)
    with pytest.raises(SystemExit) as missing:
        main(["--port", "9"])
    assert missing.value.code == 2

    seen: list[tuple[str, int, bool]] = []

    def fake_serve(
        got: LocalWeightsEngine, host: str, port: int, *, watch_orphan: bool = True
    ) -> int:
        seen.append((host, port, watch_orphan))
        assert got.weights_sha256 == engine.weights_sha256
        return 0

    monkeypatch.setattr(le, "serve", fake_serve)
    assert main(["--checkpoint-dir", str(_BASE), "--port", "11", "--host", "127.0.0.1"]) == 0
    assert seen[-1] == ("127.0.0.1", 11, True)

    monkeypatch.setenv("FX1_CHECKPOINT_DIR", str(_BASE))
    assert main(["--port", "12", "--no-orphan-watch"]) == 0
    assert seen[-1] == ("127.0.0.1", 12, False)


def test_module_main_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FX1_CHECKPOINT_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", ["fx1.serve.local_engine", "--port", "1"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("fx1.serve.local_engine", run_name="__main__")
    assert exc.value.code == 2
