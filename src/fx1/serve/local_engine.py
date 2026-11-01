"""Local fx-1 weights engine — a real checkpoint served over OpenAI-compatible HTTP.

``LocalFx1Backend`` requires a serving engine: ``FX1_LOCAL_SERVE_URL``
attaches to one already running, ``FX1_LOCAL_SERVE_CMD`` spawns one from a
template. This module is that engine for the checkpoint this repository
actually ships — it loads ``weights.safetensors`` out of a verified
checkpoint dir and answers ``/v1/chat/completions`` by running a real
forward pass over the loaded tensors (a one-block byte-level transformer,
greedy decode by default).

Honesty: every completion is computed from the deserialized weight values.
There is no canned text, no echo path, and no response that survives a
tampered or missing artifact — ``weights.manifest.json`` byte-pins the
weights file and a checksum mismatch refuses to serve. The tokenizer is a
fixed byte-level scheme (256 byte ids + BOS/EOS/PAD), so a checkpoint is
fully self-describing: safetensors + model card, nothing else.

References: OpenAI chat-completions wire shape (``_openai_chat_complete``
in ``fx1.serve.backends``), the safetensors format spec, and GPT-2's
pre-norm decoder block. Composition: sibling of ``backends`` (the client
of this server) and ``signing`` (the stronger release-attestation tier);
the fixture weights are produced by ``scripts/fx1_tiny_lm_train.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from fx1.modelcard import ModelCard

_F32 = npt.NDArray[np.float32]

WEIGHTS_FILENAME = "weights.safetensors"
MANIFEST_FILENAME = "weights.manifest.json"
CARD_FILENAME = "modelcard.json"

_TOK_EMB = "tok_emb.weight"
_POS_EMB = "pos_emb.weight"
_LM_HEAD = "lm_head.weight"

VOCAB_SIZE = 259  # 256 byte-value tokens + BOS + EOS + PAD
BOS_ID = 256
EOS_ID = 257
PAD_ID = 258

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_MAX_NEW = 48
_ENGINE_TAG = "fx1-local-weights/1"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_safetensors(path: Path) -> dict[str, _F32]:
    """Deserialize real tensor bytes — the ``nn`` extra's safetensors build."""
    try:
        from safetensors.numpy import load_file  # noqa: PLC0415 — optional extra
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required to load fx-1 weights "
            "(install the `nn` extra — `uv sync --all-extras`)"
        ) from exc
    raw = load_file(str(path))
    return {name: np.asarray(arr, dtype=np.float32) for name, arr in raw.items()}


def _layernorm(x: _F32, weight: _F32, bias: _F32, eps: float = 1e-5) -> _F32:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)  # biased — matches torch layer_norm
    return cast(_F32, (x - mu) / np.sqrt(var + eps) * weight + bias)


def _gelu_tanh(x: _F32) -> _F32:
    return cast(
        _F32,
        0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3))),
    )


def _softmax(x: _F32, axis: int = -1) -> _F32:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return cast(_F32, e / e.sum(axis=axis, keepdims=True))


@dataclass(frozen=True)
class _TinyLM:
    """A byte-level decoder-only transformer; every dim is read off the
    tensors themselves, so the weight file is the architecture spec."""

    tensors: dict[str, _F32]
    vocab: int
    dim: int
    ctx: int
    n_layers: int
    n_heads: int
    n_params: int

    @classmethod
    def from_tensors(cls, tensors: dict[str, _F32], *, n_heads: int = 4) -> _TinyLM:
        required = (_TOK_EMB, _POS_EMB, "ln_f.weight", "ln_f.bias")
        for key in required:
            if key not in tensors:
                raise RuntimeError(f"weights file is missing tensor {key!r}")
        tok = tensors[_TOK_EMB]
        pos = tensors[_POS_EMB]
        if tok.ndim != 2 or pos.ndim != 2 or pos.shape[1] != tok.shape[1]:
            raise RuntimeError("weights file has malformed embedding tensors")
        if _LM_HEAD in tensors and tensors[_LM_HEAD].shape[0] != tok.shape[0]:
            raise RuntimeError("lm_head vocab does not match tok_emb vocab")
        n_layers = len({k.split(".")[1] for k in tensors if k.startswith("blocks.")})
        if n_layers < 1:
            raise RuntimeError("weights file carries no transformer blocks")
        head_weight = tensors["blocks.0.attn.qkv.weight"]
        if head_weight.ndim != 2 or head_weight.shape[0] != 3 * tok.shape[1]:
            raise RuntimeError("qkv projection shape does not match model dim")
        if n_heads < 1 or tok.shape[1] % n_heads != 0:
            raise RuntimeError(f"n_heads {n_heads} does not divide model dim {tok.shape[1]}")
        n_params = int(sum(int(np.prod(t.shape)) for t in tensors.values()))
        return cls(
            tensors=dict(tensors),
            vocab=int(tok.shape[0]),
            dim=int(tok.shape[1]),
            ctx=int(pos.shape[0]),
            n_layers=n_layers,
            n_heads=n_heads,
            n_params=n_params,
        )

    def tokenize(self, text: str) -> list[int]:
        """Byte-level ids with a BOS prefix — no external tokenizer file."""
        return [BOS_ID, *text.encode("utf-8")]

    def detokenize(self, ids: list[int]) -> str:
        return bytes(t for t in ids if t < 256).decode("utf-8", "replace")

    def forward(self, ids: list[int]) -> _F32:
        """One real forward pass: returns logits ``(len(ids), vocab)``."""
        w = self.tensors
        idx = np.asarray(ids[-self.ctx :], dtype=np.int64)
        x = w[_TOK_EMB][idx] + w[_POS_EMB][: len(idx)]
        n = len(idx)
        dh = self.dim // self.n_heads
        causal = np.triu(np.ones((n, n), dtype=bool), k=1)
        for i in range(self.n_layers):
            p = f"blocks.{i}."
            a = _layernorm(x, w[p + "ln1.weight"], w[p + "ln1.bias"])
            qkv = a @ w[p + "attn.qkv.weight"].T + w[p + "attn.qkv.bias"]
            q, k, v = np.split(qkv, 3, axis=-1)
            qh = q.reshape(n, self.n_heads, dh).transpose(1, 0, 2)
            kh = k.reshape(n, self.n_heads, dh).transpose(1, 0, 2)
            vh = v.reshape(n, self.n_heads, dh).transpose(1, 0, 2)
            scores = qh @ kh.transpose(0, 2, 1) / math.sqrt(dh)
            scores = np.where(causal, -np.inf, scores)
            attended = (_softmax(scores) @ vh).transpose(1, 0, 2).reshape(n, self.dim)
            x = x + (attended @ w[p + "attn.proj.weight"].T + w[p + "attn.proj.bias"])
            m = _layernorm(x, w[p + "ln2.weight"], w[p + "ln2.bias"])
            hidden = _gelu_tanh(m @ w[p + "mlp.fc1.weight"].T + w[p + "mlp.fc1.bias"])
            x = x + (hidden @ w[p + "mlp.fc2.weight"].T + w[p + "mlp.fc2.bias"])
        x = _layernorm(x, w["ln_f.weight"], w["ln_f.bias"])
        head = w[_LM_HEAD] if _LM_HEAD in w else w[_TOK_EMB]
        return cast(_F32, x @ head.T)

    def generate(
        self,
        prompt_ids: list[int],
        *,
        max_new: int,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> list[int]:
        """Real decode loop: argmax at temperature 0, seeded multinomial above.

        Unseeded positive-temperature sampling is still deterministic —
        the rng seeds off the prompt bytes so a replayed request replays
        its draw."""
        ids = list(prompt_ids[-self.ctx :])
        rng: np.random.Generator | None = None
        if temperature is not None and temperature > 0:
            draw_seed = seed
            if draw_seed is None:
                digest = hashlib.blake2b(
                    bytes(t % 256 for t in ids) + str(temperature).encode(),
                    digest_size=8,
                ).digest()
                draw_seed = int.from_bytes(digest)
            rng = np.random.default_rng(draw_seed)
        out: list[int] = []
        for _ in range(max_new):
            logits = self.forward(ids)[-1]
            if rng is None:
                nxt = int(np.argmax(logits))
            else:
                probs = _softmax(logits / temperature)
                nxt = int(rng.choice(len(logits), p=probs))
            if nxt in (EOS_ID, PAD_ID):
                break
            ids.append(nxt)
            out.append(nxt)
        return out


@dataclass(frozen=True)
class EngineCompletion:
    """One real inference result, plus the provider-style token counters."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    token_ids: list[int]


class LocalWeightsEngine:
    """A verified fx-1 checkpoint dir, loaded and queryable.

    ``LocalFx1Backend`` gates the card (ship eligibility, optional release
    signature); this engine adds the artifact layer — the manifest's sha256
    pin must match the bytes on disk, and the safetensors header must name
    a complete tiny-LM tensor set. The same object powers both the HTTP
    surface (``serve``) and in-process calls (``complete_messages``), so a
    wire answer and a direct-weights answer are the same computation.

    ``enforce_ship_gate=False`` is the measurement path, not a serving
    path: an eval must be able to load the candidate checkpoint *before*
    the real eval_delta it is about to measure exists (and after a card
    has measured an honest gate failure). The artifact layer — manifest
    sha pin, tensor completeness, vocab — always verifies; only the card
    verdict is bypassed, and callers must never serve through an
    engine built this way.
    """

    def __init__(self, checkpoint_dir: str | Path, *, enforce_ship_gate: bool = True) -> None:
        root = Path(checkpoint_dir)
        card_path = root / CARD_FILENAME
        if not card_path.is_file():
            raise FileNotFoundError(f"no model card at {card_path}")
        card = ModelCard.load(card_path)
        if enforce_ship_gate and not card.eval_delta.ship_eligible:
            raise RuntimeError(
                f"{card.version} failed the ship gate; refusing to serve its weights"
            )
        manifest = self._load_manifest(root)
        weights_rel = manifest["weights_file"]
        weights_path = (root / weights_rel).resolve()
        if weights_path.parent != root.resolve() or not weights_path.is_file():
            raise RuntimeError(f"manifest weights_file {weights_rel!r} is not a checkpoint member")
        sha = _sha256_file(weights_path)
        if sha != manifest["weights_sha256"]:
            raise RuntimeError(
                f"weights integrity pin failed: {weights_rel} hashes {sha[:16]}…, "
                f"manifest pins {str(manifest['weights_sha256'])[:16]}…"
            )
        arch = manifest.get("arch")
        n_heads = 4
        if isinstance(arch, dict) and isinstance(arch.get("n_heads"), int):
            n_heads = int(arch["n_heads"])
        self.root = root
        self.card = card
        self.manifest = manifest
        self.model = _TinyLM.from_tensors(_load_safetensors(weights_path), n_heads=n_heads)
        if self.model.vocab != VOCAB_SIZE:
            raise RuntimeError(
                f"weights vocab {self.model.vocab} does not match the byte tokenizer "
                f"({VOCAB_SIZE}); this engine serves byte-level fx-1 checkpoints only"
            )
        self.served_model = card.version
        self.weights_sha256 = sha

    @staticmethod
    def _load_manifest(root: Path) -> dict[str, Any]:
        manifest_path = root / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"no weights manifest at {manifest_path}; an fx-1 checkpoint dir "
                "carries weights.manifest.json pinning its weight file's sha256"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise RuntimeError(f"{manifest_path} is not a JSON object")
        for key in ("weights_file", "weights_sha256"):
            if not isinstance(manifest.get(key), str) or not manifest[key]:
                raise RuntimeError(f"weights manifest needs a non-empty {key!r}")
        return manifest

    def _prompt_ids(self, messages: list[dict[str, Any]]) -> list[int]:
        """Render chat messages into the engine's deterministic prompt text."""
        parts: list[str] = []
        for m in messages:
            if not isinstance(m, dict) or not isinstance(m.get("content"), str):
                raise ValueError("each message must be an object with string content")
            role = m.get("role")
            parts.append(f"{role if isinstance(role, str) else 'user'}: {m['content']}")
        text = "\n".join(parts) + "\nassistant:"
        return self.model.tokenize(text)

    def complete_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
    ) -> EngineCompletion:
        """Run real inference over the loaded weights for one chat request."""
        prompt_ids = self._prompt_ids(messages)
        cap = max_tokens if isinstance(max_tokens, int) and max_tokens > 0 else _DEFAULT_MAX_NEW
        out_ids = self.model.generate(prompt_ids, max_new=cap, temperature=temperature, seed=seed)
        text = self.model.detokenize(out_ids)
        finish = "stop"
        if out_ids and len(out_ids) >= cap:
            finish = "length"
        for s in stop or []:
            cut = text.find(s)
            if cut >= 0:
                text = text[:cut]
                finish = "stop"
                break
        return EngineCompletion(
            text=text,
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(out_ids),
            finish_reason=finish,
            token_ids=out_ids,
        )

    def describe(self) -> dict[str, Any]:
        """The /v1/models entry — includes the weight pin it actually loaded."""
        return {
            "id": self.served_model,
            "object": "model",
            "created": int(self.manifest.get("created_unix", 0)),
            "owned_by": "fx1-local",
            "fx1": {
                "engine": _ENGINE_TAG,
                "weights_file": str(self.manifest["weights_file"]),
                "weights_sha256": self.weights_sha256,
                "n_params": self.model.n_params,
                "tokenizer": "byte-level-259",
            },
        }


def _models_payload(engine: LocalWeightsEngine) -> dict[str, Any]:
    return {"object": "list", "data": [engine.describe()]}


def _chat_payload(engine: LocalWeightsEngine, comp: EngineCompletion) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-fx1local-{int(time.time() * 1000):x}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": engine.served_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": comp.text},
                "finish_reason": comp.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": comp.prompt_tokens,
            "completion_tokens": comp.completion_tokens,
            "total_tokens": comp.prompt_tokens + comp.completion_tokens,
        },
    }


def _stream_frames(engine: LocalWeightsEngine, comp: EngineCompletion) -> bytes:
    """SSE bytes: one delta per generated token (incremental UTF-8 decode),
    a terminal finish_reason + usage frame, then [DONE]."""
    import codecs

    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    head = {
        "id": f"chatcmpl-fx1local-{int(time.time() * 1000):x}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": engine.served_model,
    }
    frames: list[bytes] = []
    for tok in comp.token_ids:
        piece = decoder.decode(bytes([tok]))
        if not piece:
            continue
        chunk = dict(head)
        chunk["choices"] = [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]
        frames.append(b"data: " + json.dumps(chunk).encode() + b"\n\n")
    tail = dict(head)
    tail["choices"] = [{"index": 0, "delta": {}, "finish_reason": comp.finish_reason}]
    tail["usage"] = {
        "prompt_tokens": comp.prompt_tokens,
        "completion_tokens": comp.completion_tokens,
        "total_tokens": comp.prompt_tokens + comp.completion_tokens,
    }
    frames.append(b"data: " + json.dumps(tail).encode() + b"\n\n")
    frames.append(b"data: [DONE]\n\n")
    return b"".join(frames)


def _stop_list(raw: Any) -> list[str]:
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [s for s in raw if isinstance(s, str)]
    return []


class _EngineHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    engine: LocalWeightsEngine


class _Handler(BaseHTTPRequestHandler):
    """Thin HTTP layer — every POST body drives ``complete_messages`` for real."""

    protocol_version = "HTTP/1.1"
    server_version = "fx1-local-weights"

    @property
    def _engine(self) -> LocalWeightsEngine:
        return cast(_EngineHTTPServer, self.server).engine

    def _respond_json(self, status: int, obj: dict[str, Any]) -> None:
        payload = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _respond_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_body(self) -> dict[str, Any] | None:
        n = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._respond_json(400, {"error": {"message": "malformed JSON body"}})
            return None
        if not isinstance(body, dict):
            self._respond_json(400, {"error": {"message": "body must be a JSON object"}})
            return None
        return body

    def do_GET(self) -> None:  # noqa: N802 — stdlib hook name
        path = urllib.parse.urlparse(self.path).path
        if path == "/v1/models":
            self._respond_json(200, _models_payload(self._engine))
            return
        self._respond_json(404, {"error": {"message": f"unknown route {path}"}})

    def do_POST(self) -> None:  # noqa: N802 — stdlib hook name
        path = urllib.parse.urlparse(self.path).path
        body = self._read_body()
        if body is None:
            return
        engine = self._engine
        if path == "/v1/tokenize":
            try:
                ids = engine._prompt_ids(body.get("messages") or [])
            except ValueError as exc:
                self._respond_json(400, {"error": {"message": str(exc)}})
                return
            self._respond_json(200, {"count": len(ids), "model": engine.served_model})
            return
        if path != "/v1/chat/completions":
            self._respond_json(404, {"error": {"message": f"unknown route {path}"}})
            return
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            self._respond_json(400, {"error": {"message": "messages must be a non-empty list"}})
            return
        try:
            comp = engine.complete_messages(
                messages,
                max_tokens=body.get("max_tokens"),
                temperature=body.get("temperature"),
                seed=body.get("seed"),
                stop=_stop_list(body.get("stop")),
            )
        except ValueError as exc:
            self._respond_json(400, {"error": {"message": str(exc)}})
            return
        if body.get("stream"):
            self._respond_bytes(_stream_frames(engine, comp), "text/event-stream")
            return
        self._respond_json(200, _chat_payload(engine, comp))

    def log_message(self, *args: Any) -> None:  # keep serve output quiet
        return None


def _watch_orphan(spawner_ppid: int) -> None:
    """Exit once the spawning process is gone (reparented to init).

    A ``LocalFx1Backend``-spawned engine is scoped to its server: when the
    harness dies (including SIGKILL), the socket must not outlive it as an
    orphan holding the port. ``ppid == 1`` means orphaned regardless of
    the recorded value — a parent that died before capture is still dead."""
    while True:
        # ``ppid == 1`` means orphaned even when it equals the recorded
        # value — a parent that died during the slow checkpoint load was
        # recorded posthumously; reparenting to init is always the end.
        ppid = os.getppid()
        if ppid != spawner_ppid or ppid == 1:
            os._exit(0)
        time.sleep(1.0)


def serve(engine: LocalWeightsEngine, host: str, port: int, *, watch_orphan: bool = True) -> int:
    """Serve forever on a real socket; returns 0 on a clean shutdown."""
    srv = _EngineHTTPServer((host, port), _Handler)
    srv.engine = engine
    if watch_orphan:
        threading.Thread(target=_watch_orphan, args=(os.getppid(),), daemon=True).start()
    try:
        srv.serve_forever()  # NOSONAR — loopback fixture engine; harness-local, no TLS needed
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    """``python -m fx1.serve.local_engine`` — the FX1_LOCAL_SERVE_CMD target."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--host", default=_DEFAULT_HOST)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--no-orphan-watch",
        action="store_true",
        help="do not exit when the spawning process dies (standalone daemon mode)",
    )
    args = parser.parse_args(argv)
    env_dir = os.environ.get("FX1_CHECKPOINT_DIR", "")
    root = args.checkpoint_dir if args.checkpoint_dir is not None else Path(env_dir)
    if args.checkpoint_dir is None and not env_dir:
        parser.error("a checkpoint dir is required (--checkpoint-dir or FX1_CHECKPOINT_DIR)")
    engine = LocalWeightsEngine(root)
    print(
        f"fx1 local weights engine: serving {engine.served_model} "
        f"({engine.model.n_params} params, sha256 {engine.weights_sha256[:16]}…) "
        f"on {args.host}:{args.port}",
        file=sys.stderr,
        flush=True,
    )
    return serve(engine, args.host, args.port, watch_orphan=not args.no_orphan_watch)


if __name__ == "__main__":
    raise SystemExit(main())
