"""api_audit — adversarial probes on the fx-1 harness HTTP surface.

The harness API (``fx1.serve.api``) is the network half of the harness
contract: anything reachable over the socket must be exactly what the
in-process ``Harness`` allows — no more. These probes pin that parity:

- *Registry parity* — every command in ``HARNESS_REGISTRY`` is listed and
  executable; unregistered names 404 over the wire exactly as ``Harness.get``
  raises in-process.
- *Containment parity* — ``config`` / ``extra_args=--config`` escapes hit
  the same ``configs/`` allowlist and 422; a config keyword on a non-local
  backend is refused.
- *Fail-closed shape* — wrong field types, unknown keys, empty message
  lists, oversized bodies, and bad methods all 4xx uniformly; nothing
  reaches a route handler malformed.
- *Completion gate* — completions run through ``cited_complete``, so a
  backend that emits a forbidden headline is cut off at the gate (502)
  rather than served; unconfigured BYOK/hosted creds surface as 503, a
  backend that doesn't exist as 422. ``local_fx1`` completes through an
  attached/spawned OpenAI-compatible engine: no ``FX1_LOCAL_SERVE_URL`` /
  ``FX1_LOCAL_SERVE_CMD`` → 503, a dead engine → 502, a live one → 200
  with the card's version as the model id; ``/health`` reports
  ``local_fx1`` ready only when a card'd checkpoint *and* engine config
  are both present.
- *Verification surface* — ``POST /receipts/verify`` verifies arbitrary
  payloads through the same ``verify_receipt_payload`` the CLI uses;
  malformed receipts fail closed, never crash.
- *Auth posture* — ``/health`` is the only public route; ``FX1_API_KEY``
  set → every other route requires ``X-API-Key`` (compare_digest); unset →
  non-loopback clients 403. Security headers on every response.

Sealed ``api_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING, Any, ClassVar

_MESSAGES_PATH = "/v1/messages"
_LEGACY_PATH = "/v1/completions"
_PING_MSG = "clean:ping"
_SSE_EVENT_PREFIX = "event: "


if TYPE_CHECKING:
    from types import ModuleType

    from fastapi.testclient import TestClient

    from fx1.serve.backends import SamplingParams

__all__ = ["api_audit", "api_audit_bench"]

_PATH_UPLOADS = "/v1/uploads"
_PATH_EVALS = "/v1/evals"
_PATH_FT_JOBS = "/v1/fine_tuning/jobs"
_CORPUS_FILE = "c.jsonl"

_API_KEY_ENV = "FX1_API_KEY"
_CONVERSATIONS_URL = "/v1/conversations"
_BYOK_ENVS = ("FX1_BYOK_BASE_URL", "FX1_BYOK_API_KEY", "FX1_BYOK_MODEL")
_LOCAL_ENVS = (
    "FX1_LOCAL_SERVE_URL",
    "FX1_LOCAL_SERVE_CMD",
    "FX1_LOCAL_MODEL",
    "FX1_LOCAL_API_KEY",
    "FX1_CHECKPOINT_DIR",
)


def _client(
    api_key: str | None = None,
) -> tuple[TestClient, ModuleType]:
    """(TestClient, api_module) — isolated env per construction."""
    from fastapi.testclient import TestClient

    import fx1.serve.api as api_mod
    from fx1.harness import Harness

    # An injected runner keeps the audit hermetic — probes never spawn real
    # lab processes, exactly like harness_audit's in-process probes.
    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, f"ran:{' '.join(argv)}", ""

    saved = {
        k: os.environ.get(k) for k in (_API_KEY_ENV, *_BYOK_ENVS, *_LOCAL_ENVS, "MOONSHOT_API_KEY")
    }
    try:
        if api_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = api_key
        app = api_mod.create_app(harness=Harness(runner=fake_runner))
        return TestClient(app, raise_server_exceptions=False), api_mod
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def api_audit() -> dict[str, Any]:
    _clk = {"t": 1_000_000.0}
    from fx1.serve.api import _RateLimiter

    _RateLimiter._clock_box = _clk
    try:
        return _api_audit_impl(_clk)
    finally:
        _RateLimiter._clock_box = None


def _api_audit_impl(_clk: dict[str, float]) -> dict[str, Any]:  # noqa: C901
    out: dict[str, Any] = {}

    client, api_mod = _client()
    from fx1.harness import HARNESS_REGISTRY

    resp = client.get("/harness/commands")
    names = {c["name"] for c in resp.json()["items"]}
    out["commands_all_listed"] = resp.status_code == 200 and names == {
        c.name for c in HARNESS_REGISTRY
    }
    resp = client.get("/harness/commands", params={"role": "verification"})
    out["commands_role_filter"] = resp.status_code == 200 and all(
        c["role"] == "verification" for c in resp.json()["items"]
    )
    out["commands_role_bad_422"] = (
        client.get("/harness/commands", params={"role": "bogus"}).status_code == 422
    )

    resp = client.post("/harness/runs", json={"command": "doctor"})
    body = resp.json()
    out["run_executes"] = (
        resp.status_code == 200
        and body["ok"] is True
        and body["exit_code"] == 0
        and "doctor" in body["stdout"]
    )
    out["run_unknown_404"] = (
        client.post("/harness/runs", json={"command": "pwn"}).status_code == 404
    )
    out["run_extra_key_422"] = (
        client.post("/harness/runs", json={"command": "doctor", "evil": True}).status_code == 422
    )
    out["run_args_wrongtype_422"] = (
        client.post(
            "/harness/runs", json={"command": "doctor", "extra_args": "rm -rf /"}
        ).status_code
        == 422
    )
    out["run_method_shape"] = client.get("/harness/runs").status_code == 405
    out["run_timeout_pinned"] = resp.json()["timeout_s"] > 0 if resp.status_code == 200 else False

    out["config_escape_422"] = (
        client.post(
            "/harness/runs",
            json={"command": "doctor", "config": "/etc/passwd"},
        ).status_code
        == 422
    )
    out["extra_args_config_escape_422"] = (
        client.post(
            "/harness/runs",
            json={"command": "doctor", "extra_args": ["--config", "/etc/passwd"]},
        ).status_code
        == 422
    )
    out["extra_args_config_eq_escape_422"] = (
        client.post(
            "/harness/runs",
            json={"command": "doctor", "extra_args": ["--config=/etc/passwd"]},
        ).status_code
        == 422
    )

    for name in _BYOK_ENVS:
        os.environ.pop(name, None)
    os.environ.pop("MOONSHOT_API_KEY", None)
    out["complete_byok_unconfigured_503"] = (
        client.post(
            "/harness/complete",
            json={
                "backend": "byok",
                "messages": [{"role": "user", "content": "ping"}],
            },
        ).status_code
        == 503
    )
    out["complete_hosted_unconfigured_503"] = (
        client.post(
            "/harness/complete",
            json={
                "backend": "hosted_k3",
                "messages": [{"role": "user", "content": "ping"}],
            },
        ).status_code
        == 503
    )
    out["complete_unknown_backend_422"] = (
        client.post(
            "/harness/complete",
            json={
                "backend": "bogus",
                "messages": [{"role": "user", "content": "ping"}],
            },
        ).status_code
        == 422
    )
    out["complete_empty_messages_422"] = (
        client.post("/harness/complete", json={"backend": "byok", "messages": []}).status_code
        == 422
    )
    out["complete_checkpoint_misplaced_422"] = (
        client.post(
            "/harness/complete",
            json={
                "backend": "byok",
                "messages": [{"role": "user", "content": "ping"}],
                "checkpoint_dir": "./no-such-dir",
            },
        ).status_code
        == 422
    )
    out["complete_local_no_checkpoint_422"] = (
        client.post(
            "/harness/complete",
            json={
                "backend": "local_fx1",
                "messages": [{"role": "user", "content": "ping"}],
            },
        ).status_code
        == 422
    )

    from fastapi.testclient import TestClient as _IdemTC  # noqa: PLC0415

    class _IdemBackend:
        def __init__(self) -> None:
            self.calls = 0

        def complete(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> str:
            self.calls += 1
            return f"clean:{messages[-1]['content']}"

    _idem_backend = _IdemBackend()
    idem_resolves = {"n": 0}

    def _counting_resolver(*a: Any, **k: Any) -> Any:
        idem_resolves["n"] += 1
        return _idem_backend

    idem_api = api_mod.create_app(backend_resolver=_counting_resolver)
    ic = _IdemTC(idem_api)
    cbody = {"backend": "byok", "messages": [{"role": "user", "content": "ping"}]}
    c1 = ic.post("/harness/complete", json=cbody, headers={"Idempotency-Key": "ck1"})
    c2 = ic.post("/harness/complete", json=cbody, headers={"Idempotency-Key": "ck1"})
    out["complete_idem_replay"] = (
        idem_resolves["n"] == 1
        and _idem_backend.calls == 1
        and c2.json()["replayed"] is True
        and c2.json()["content"] == c1.json()["content"]
    )
    out["complete_idem_conflict_409"] = (
        ic.post(
            "/harness/complete",
            json={"backend": "byok", "messages": [{"role": "user", "content": "other"}]},
            headers={"Idempotency-Key": "ck1"},
        ).status_code
        == 409
    )
    before = idem_resolves["n"]
    ic.post("/harness/complete", json=cbody)
    ic.post("/harness/complete", json=cbody)
    out["complete_no_key_reexecutes"] = idem_resolves["n"] == before + 2
    bbody = {"backend": "byok", "batch": [[{"role": "user", "content": "p"}]]}
    ic.post("/harness/complete/batch", json=bbody, headers={"Idempotency-Key": "bk1"})
    cb2 = ic.post("/harness/complete/batch", json=bbody, headers={"Idempotency-Key": "bk1"}).json()
    out["complete_batch_idem_replay"] = cb2["replayed"] is True
    out["complete_reports_latency_ms"] = (
        isinstance(c1.json()["latency_ms"], (int, float)) and c1.json()["latency_ms"] >= 0
    )
    out["complete_replay_reuses_latency_ms"] = c2.json()["latency_ms"] == c1.json()["latency_ms"]

    import json as _json
    import tempfile
    from pathlib import Path
    from unittest.mock import patch

    from fx1.modelcard import EvalDelta, ModelCard

    card = ModelCard(
        version="fx-1.v0.1",
        corpus_sha256="a" * 64,
        corpus_receipt_range="aa..bb",
        training_manifest_sha256="b" * 64,
        eval_delta=EvalDelta(
            domain_pass_rate_base=0.5,
            domain_pass_rate_candidate=0.6,
            general_pass_rate_base=0.5,
            general_pass_rate_candidate=0.5,
            honesty_gate_candidate=True,
        ),
    )
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "ckpt"
        ckpt.mkdir()
        card.save(ckpt / "modelcard.json")

        out["local_unconfigured_503"] = (
            client.post(
                "/harness/complete",
                json={
                    "backend": "local_fx1",
                    "messages": [{"role": "user", "content": "ping"}],
                    "checkpoint_dir": str(ckpt),
                },
            ).status_code
            == 503
        )
        os.environ["FX1_CHECKPOINT_DIR"] = str(ckpt)
        try:
            out["health_local_needs_engine"] = (
                client.get("/health").json()["backends"]["local_fx1"] is False
            )
        finally:
            os.environ.pop("FX1_CHECKPOINT_DIR", None)

        class _Resp:
            def read(self) -> bytes:
                return _json.dumps({"choices": [{"message": {"content": "answer"}}]}).encode()

            def __enter__(self) -> _Resp:
                return self

            def __exit__(self, *a: Any) -> None:
                return None

        def fake_urlopen(req: Any, **kw: Any) -> _Resp:
            return _Resp()

        import fx1.serve.backends as _be_mod  # noqa: PLC0415

        os.environ["FX1_CHECKPOINT_DIR"] = str(ckpt)
        os.environ["FX1_LOCAL_SERVE_URL"] = "http://127.0.0.1:8011/v1"
        try:
            # wire calls route through _openai_urlopen — patching
            # urllib.request.urlopen is a dead seam (a private
            # OpenerDirector owns the dial).
            with patch.object(_be_mod, "_openai_urlopen", fake_urlopen):
                resp = client.post(
                    "/harness/complete",
                    json={
                        "backend": "local_fx1",
                        "messages": [{"role": "user", "content": "ping"}],
                        "checkpoint_dir": str(ckpt),
                    },
                )
            body = resp.json()
            out["local_attach_complete_200"] = (
                resp.status_code == 200
                and body["content"].startswith("answer")
                and body["model"] == "fx-1.v0.1"
            )
            out["health_local_configured"] = (
                client.get("/health").json()["backends"]["local_fx1"] is True
            )
        finally:
            os.environ.pop("FX1_CHECKPOINT_DIR", None)
            os.environ.pop("FX1_LOCAL_SERVE_URL", None)
        os.environ["FX1_LOCAL_SERVE_URL"] = "http://127.0.0.1:9/v1"
        try:
            out["local_engine_down_502"] = (
                client.post(
                    "/harness/complete",
                    json={
                        "backend": "local_fx1",
                        "messages": [{"role": "user", "content": "ping"}],
                        "checkpoint_dir": str(ckpt),
                    },
                ).status_code
                == 502
            )
        finally:
            os.environ.pop("FX1_LOCAL_SERVE_URL", None)

    # honesty gate fires over the wire: a backend emitting a forbidden
    # headline must not serve it.
    class _DirtyBackend:
        def complete(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> str:
            return "The strategy achieved a sharpe of 2.1 on the tape."

    # Injected resolver — the honesty gate must hold even when the model
    # behind the socket is adversarial.
    from fastapi.testclient import TestClient as _TC2

    dirty = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _DirtyBackend()))
    out["complete_honesty_gate_502"] = (
        dirty.post(
            "/harness/complete",
            json={
                "backend": "byok",
                "messages": [{"role": "user", "content": "report"}],
            },
        ).status_code
        == 502
    )

    # One backend instance serves the whole batch; per-item verdicts.
    class _CleanBackend:
        def __init__(self) -> None:
            self._model = "fake-0"
            self.calls = 0
            self.closed = 0

        def complete(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> str:
            self.calls += 1
            return f"clean:{messages[-1]['content']}"

        def close(self) -> None:
            self.closed += 1

    clean = _CleanBackend()
    resolves = [0]

    def _resolve_once(*a: Any, **k: Any) -> _CleanBackend:
        resolves[0] += 1
        return clean

    batch_client = _TC2(api_mod.create_app(backend_resolver=_resolve_once))
    r = batch_client.post(
        "/harness/complete/batch",
        json={
            "backend": "byok",
            "batch": [[{"role": "user", "content": f"q{i}"}] for i in range(3)],
        },
    )
    bj = r.json()
    out["batch_complete_200"] = (
        r.status_code == 200 and len(bj["results"]) == 3 and all(i["ok"] for i in bj["results"])
    )
    out["batch_in_order"] = [i["content"] for i in bj["results"]] == [
        "clean:q0",
        "clean:q1",
        "clean:q2",
    ]
    out["batch_one_backend"] = resolves[0] == 1 and clean.calls == 3
    out["batch_closed_once"] = clean.closed == 1
    out["batch_item_latency_ms"] = all(
        isinstance(i["latency_ms"], (int, float)) and i["latency_ms"] >= 0 for i in bj["results"]
    )
    out["batch_empty_422"] = (
        batch_client.post(
            "/harness/complete/batch", json={"backend": "byok", "batch": []}
        ).status_code
        == 422
    )
    out["batch_local_no_checkpoint_422"] = (
        batch_client.post(
            "/harness/complete/batch",
            json={
                "backend": "local_fx1",
                "batch": [[{"role": "user", "content": "x"}]],
            },
        ).status_code
        == 422
    )
    out["batch_unknown_backend_422"] = (
        batch_client.post(
            "/harness/complete/batch",
            json={"backend": "bogus", "batch": [[{"role": "u", "content": "x"}]]},
        ).status_code
        == 422
    )

    # Per-slot verdicts: a refusal on one item doesn't lose the batch.
    class _PartialBackend(_CleanBackend):
        def complete(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> str:
            if messages[-1]["content"] == "bad":
                return "total Sharpe 4.2 on NAV"  # forbidden headline
            return super().complete(messages)

    partial_client = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _PartialBackend()))
    r2 = partial_client.post(
        "/harness/complete/batch",
        json={
            "backend": "byok",
            "batch": [
                [{"role": "user", "content": "good"}],
                [{"role": "user", "content": "bad"}],
                [{"role": "user", "content": "good2"}],
            ],
        },
    )
    rj = r2.json()["results"]
    out["batch_per_item_verdicts"] = (
        r2.status_code == 200
        and rj[0]["ok"]
        and rj[2]["ok"]
        and not rj[1]["ok"]
        and rj[1]["error_class"] == "honesty_refusal"
    )

    class _StreamBackend(_CleanBackend):
        def stream(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> Any:
            yield "tok-a"
            yield "tok-b"

    # no receipt store -> the synthetic citation below is advisory, not 422.
    stream_client = _TC2(
        api_mod.create_app(
            backend_resolver=lambda *a, **k: _StreamBackend(),
            receipts_dir="/nonexistent-stream-store",
        )
    )
    rs = stream_client.post(
        "/harness/complete/stream",
        json={
            "backend": "byok",
            "messages": [{"role": "user", "content": "hi"}],
            "receipt_hashes": ["a" * 64],
        },
    )
    body_text = rs.text
    frames = [ln for ln in body_text.split("\n\n") if ln.strip()]
    out["stream_200_sse"] = (
        rs.status_code == 200
        and rs.headers.get("content-type", "").startswith("text/event-stream")
        and len(frames) == 5  # 2 tokens + footer + final + [DONE]
    )
    payloads = [
        _json.loads(ln[len("data: ") :])
        for ln in frames
        if ln.startswith("data: ") and ln[len("data: ") :].strip() != "[DONE]"
    ]
    out["stream_token_order"] = [p["content"] for p in payloads if p.get("type") == "token"][
        :2
    ] == ["tok-a", "tok-b"]
    out["stream_final_envelope"] = payloads[-1].get("type") == "final" and payloads[-1].get(
        "receipt_hashes"
    ) == ["a" * 64]
    out["stream_done_terminates"] = frames[-1].strip() == "data: [DONE]"
    out["stream_footer_cited"] = any("Evidence:" in p.get("content", "") for p in payloads)
    out["stream_final_reports_latency_ms"] = (
        isinstance(payloads[-1].get("latency_ms"), (int, float)) and payloads[-1]["latency_ms"] >= 0
    )
    # the final frame self-describes its evidence: receipt_sha256 equals the
    # sealed export of the logged record — a stream client pins the record
    # without a second call
    sf_cid = payloads[-1].get("completion_id")
    sf_doc = stream_client.get(f"/harness/completions/{sf_cid}/receipt")
    out["stream_final_receipt_sha"] = (
        sf_cid is not None and payloads[-1].get("receipt_sha256") == sf_doc.json()["receipt_sha256"]
    )

    class _DirtyStreamBackend(_DirtyBackend):
        def stream(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> Any:
            yield "total Sharpe 4.2 on NAV"

    dirty_stream = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _DirtyStreamBackend()))
    rd = dirty_stream.post(
        "/harness/complete/stream",
        json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
    )
    # Gate fires before any SSE frame is emitted — the refusal is a plain
    # JSON error, never a truncated event stream.
    out["stream_gate_502_json"] = (
        rd.status_code == 502
        and not rd.headers.get("content-type", "").startswith("text/event-stream")
        and "total Sharpe" not in rd.text
    )
    out["stream_local_no_checkpoint_422"] = (
        stream_client.post(
            "/harness/complete/stream",
            json={"backend": "local_fx1", "messages": [{"role": "u", "content": "x"}]},
        ).status_code
        == 422
    )
    out["stream_unsupported_501"] = (
        batch_client.post(  # _CleanBackend has no stream()
            "/harness/complete/stream",
            json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
        ).status_code
        == 501
    )

    class _SlowStreamBackend(_CleanBackend):
        def stream(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> Any:
            time.sleep(0.3)
            yield "slow-tok"

    ka_client = _TC2(
        api_mod.create_app(
            backend_resolver=lambda *a, **k: _SlowStreamBackend(),
            sse_keepalive_s=0.05,
        )
    )
    rk = ka_client.post(
        "/harness/complete/stream",
        json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
    )
    ka_frames = [ln for ln in rk.text.split("\n\n") if ln.strip()]
    out["stream_keepalive_comments"] = (
        rk.status_code == 200
        and rk.headers.get("content-type", "").startswith("text/event-stream")
        and any(ln.strip() == ": keepalive" for ln in ka_frames)
        and '"type": "final"' in rk.text
        and ka_frames[-1].strip() == "data: [DONE]"
    )

    class _SlowFailBackend(_CleanBackend):
        def stream(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> Any:
            time.sleep(0.3)
            raise RuntimeError("engine died mid-generation")

    ka_fail = _TC2(
        api_mod.create_app(
            backend_resolver=lambda *a, **k: _SlowFailBackend(),
            sse_keepalive_s=0.05,
        )
    )
    rf = ka_fail.post(
        "/harness/complete/stream",
        json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
    )
    err_frames = [
        _json.loads(ln[len("data: ") :])
        for ln in rf.text.split("\n\n")
        if ln.startswith("data: ") and '"type": "error"' in ln
    ]
    out["stream_keepalive_error_inband"] = (
        rf.status_code == 200
        and len(err_frames) == 1
        and err_frames[0]["status"] == 502
        and "engine died" in err_frames[0]["detail"]
        and rf.text.rstrip().endswith("data: [DONE]")
    )

    # Resolved inside the grace window → the synchronous contract holds:
    # refusals are still JSON errors, completions still plain SSE.
    ka_dirty = _TC2(
        api_mod.create_app(
            backend_resolver=lambda *a, **k: _DirtyStreamBackend(),
            sse_keepalive_s=5.0,
        )
    )
    rg = ka_dirty.post(
        "/harness/complete/stream",
        json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
    )
    out["stream_keepalive_grace_json_error"] = (
        rg.status_code == 502
        and not rg.headers.get("content-type", "").startswith("text/event-stream")
        and "total Sharpe" not in rg.text
    )
    ka_fast = _TC2(
        api_mod.create_app(
            backend_resolver=lambda *a, **k: _StreamBackend(),
            sse_keepalive_s=5.0,
        )
    )
    rfast = ka_fast.post(
        "/harness/complete/stream",
        json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
    )
    out["stream_keepalive_grace_sse"] = (
        rfast.status_code == 200
        and ": keepalive" not in rfast.text
        and rfast.text.rstrip().endswith("data: [DONE]")
    )

    # Disabled → fully synchronous; misconfig fails closed.
    off_client = _TC2(
        api_mod.create_app(
            backend_resolver=lambda *a, **k: _DirtyStreamBackend(),
            sse_keepalive_s=0,
        )
    )
    out["stream_keepalive_off_json_error"] = (
        off_client.post(
            "/harness/complete/stream",
            json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
        ).status_code
        == 502
    )
    try:
        api_mod.create_app(sse_keepalive_s=-1.0)
        out["stream_keepalive_negative_rejected"] = False
    except ValueError:
        out["stream_keepalive_negative_rejected"] = True
    keep_env = os.environ.get("FX1_API_SSE_KEEPALIVE_S")
    os.environ["FX1_API_SSE_KEEPALIVE_S"] = "7.5"
    try:
        out["stream_keepalive_env_config"] = api_mod.create_app().state.sse_keepalive_s == 7.5
    finally:
        if keep_env is None:
            os.environ.pop("FX1_API_SSE_KEEPALIVE_S", None)
        else:
            os.environ["FX1_API_SSE_KEEPALIVE_S"] = keep_env

    from fx1.harness import Harness as _Harness  # noqa: PLC0415

    drain_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
    )
    dclient = _TC2(drain_app)
    drain_app.state.metrics.acquire()  # pretend one request is in-flight
    # a stored key must still serve its replay while draining
    dclient.post(
        "/harness/runs",
        json={"command": "doctor"},
        headers={"Idempotency-Key": "pre-drain-key"},
    )
    try:
        ready0 = dclient.get("/ready")
        out["ready_200"] = (
            ready0.status_code == 200
            and ready0.json()["ready"] is True
            and ready0.json()["inflight"] >= 1
        )
        d0 = dclient.post("/harness/drain")
        out["drain_response_shape"] = (
            d0.status_code == 200
            and d0.json()["draining"] is True
            and isinstance(d0.json()["inflight"], int)
        )
        out["drain_reports_inflight"] = d0.json()["inflight"] >= 1
        out["drain_complete_refused_503"] = (
            dclient.post(
                "/harness/complete",
                json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
            ).status_code
            == 503
        )
        out["drain_runs_refused_503"] = (
            dclient.post("/harness/runs", json={"command": "x"}).status_code == 503
        )
        out["drain_gate_check_up"] = (
            dclient.post("/harness/gate/check", json={"text": "ok"}).status_code == 200
        )
        out["drain_score_up"] = (
            dclient.post("/harness/score", json={"input": "ok"}).status_code == 200
        )
        out["drain_moderations_up"] = (
            dclient.post("/v1/moderations", json={"input": "ok"}).status_code == 200
        )
        out["drain_uncapped_routes_up"] = (
            dclient.get("/harness/commands").status_code == 200
            and dclient.post("/receipts/verify", json={"receipt": {"x": 1}}).status_code == 200
        )
        out["drain_eval_diff_up"] = dclient.get("/harness/evals/a/diff/b").status_code == 404
        out["drain_health_reports"] = (
            dclient.get("/health").status_code == 200
            and dclient.get("/health").json()["draining"] is True
        )
        out["drain_metrics_reports"] = dclient.get("/metrics").json()["draining"] is True
        out["drain_idempotent"] = dclient.post("/harness/drain").json()["draining"] is True
        out["ready_under_drain_503"] = dclient.get("/ready").status_code == 503
        out["error_code_draining"] = (
            dclient.post("/harness/runs", json={"command": "x"}).json()["code"] == "draining"
        )
        replayed = dclient.post(
            "/harness/runs",
            json={"command": "doctor"},
            headers={"Idempotency-Key": "pre-drain-key"},
        )
        out["drain_replay_served"] = (
            replayed.status_code == 200 and replayed.json()["replayed"] is True
        )
    finally:
        drain_app.state.metrics.release()

    import time as _time  # noqa: PLC0415

    def _slow_runner(a: list[str], t: float) -> tuple[int, str, str]:
        _time.sleep(0.4)
        return (0, "ran:" + " ".join(a), "")

    wait_app = api_mod.create_app(
        harness=_Harness(runner=_slow_runner),
        backend_resolver=lambda *a, **k: _CleanBackend(),
    )
    wclient = _TC2(wait_app)
    wjob = wclient.post("/harness/jobs", json={"command": "doctor"}).json()["job_id"]
    w0 = wclient.post("/harness/drain?wait_s=0.01")
    out["drain_wait_s_timeout_reports_not_drained"] = (
        w0.status_code == 200 and w0.json()["drained"] is False
    )
    w1 = wclient.post("/harness/drain?wait_s=5")
    out["drain_wait_s_blocks_until_empty"] = (
        w1.status_code == 200 and w1.json()["drained"] is True and w1.json()["inflight"] == 0
    )
    out["drain_wait_s_job_completes"] = (
        wclient.get(f"/harness/jobs/{wjob}").json()["status"] == "succeeded"
    )
    out["drain_wait_s_validated"] = wclient.post("/harness/drain?wait_s=-1").status_code == 422

    # --- idempotency keys: dedup retries of a submitted run ------------------
    r1 = client.post(
        "/harness/runs",
        json={"command": "doctor"},
        headers={"Idempotency-Key": "audit-key-1"},
    )
    r2 = client.post(
        "/harness/runs",
        json={"command": "doctor"},
        headers={"Idempotency-Key": "audit-key-1"},
    )
    out["idem_same_key_replays"] = (
        r1.status_code == 200
        and r2.status_code == 200
        and r1.json()["replayed"] is False
        and r2.json()["replayed"] is True
        and r2.json()["stdout"] == r1.json()["stdout"]
    )
    r3 = client.post(
        "/harness/runs",
        json={"command": "doctor"},
        headers={"Idempotency-Key": "audit-key-2"},
    )
    out["idem_distinct_keys_fresh"] = r3.json()["replayed"] is False
    out["idem_absent_ok"] = (
        client.post("/harness/runs", json={"command": "doctor"}).json()["replayed"] is False
    )
    out["idem_oversized_400"] = (
        client.post(
            "/harness/runs",
            json={"command": "doctor"},
            headers={"Idempotency-Key": "k" * 300},
        ).status_code
        == 400
    )
    tiny_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        idem_max=2,
    )
    tc = _TC2(tiny_app)
    for k_ in ("tk1", "tk2", "tk3"):
        tc.post(
            "/harness/runs",
            json={"command": "doctor"},
            headers={"Idempotency-Key": k_},
        )
    out["idem_bound_evicts_oldest"] = (
        tc.post(
            "/harness/runs",
            json={"command": "doctor"},
            headers={"Idempotency-Key": "tk1"},
        ).json()["replayed"]
        is False
    )
    try:
        api_mod.create_app(idem_max=0)
        out["idem_max_validated"] = False
    except ValueError:
        out["idem_max_validated"] = True

    # --- async jobs ------------------------------------------------------------
    jobs_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
    )
    jclient = _TC2(jobs_app)
    submit = jclient.post("/harness/jobs", json={"command": "doctor"})
    out["job_submit_202"] = submit.status_code == 202
    job_id = submit.json()["job_id"]
    out["job_id_shape"] = bool(job_id) and len(job_id) == 32
    job = None
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        job = jclient.get(f"/harness/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.02)
    out["job_completes_succeeded"] = job is not None and job["status"] == "succeeded"
    out["job_result_fields"] = (
        job is not None
        and job["result"] is not None
        and job["result"]["command"] == "doctor"
        and job["result"]["ok"] is True
        and job["finished_at"] is not None
    )
    out["job_unknown_submit_404"] = (
        jclient.post("/harness/jobs", json={"command": "nope-nope"}).status_code == 404
    )
    out["job_status_unknown_404"] = jclient.get("/harness/jobs/does-not-exist").status_code == 404
    replay = jclient.post(
        "/harness/jobs",
        json={"command": "doctor"},
        headers={"Idempotency-Key": "job-key-1"},
    )
    replay2 = jclient.post(
        "/harness/jobs",
        json={"command": "doctor"},
        headers={"Idempotency-Key": "job-key-1"},
    )
    out["job_idem_replay"] = (
        replay.status_code == 202
        and replay2.json()["job_id"] == replay.json()["job_id"]
        and replay2.json()["replayed"] is True
    )
    out["job_idem_conflict_409"] = (
        jclient.post(
            "/harness/jobs",
            json={"command": "doctor", "extra_args": ["--x"]},
            headers={"Idempotency-Key": "job-key-1"},
        ).status_code
        == 409
    )
    # inventory: newest-first listing with status filter + offset/limit paging
    lst = jclient.get("/harness/jobs").json()
    out["job_list_200"] = "jobs" in lst and isinstance(lst["total"], int) and lst["total"] >= 1
    out["job_list_status_filter"] = all(
        j["status"] == "succeeded"
        for j in jclient.get("/harness/jobs", params={"status": "succeeded"}).json()["jobs"]
    )
    page1 = jclient.get("/harness/jobs", params={"limit": 1}).json()
    page2 = jclient.get("/harness/jobs", params={"limit": 1, "offset": 1}).json()
    out["job_list_paging"] = (
        len(page1["jobs"]) == 1
        and page2["jobs"]
        and page1["jobs"][0]["job_id"] != page2["jobs"][0]["job_id"]
    )
    out["job_list_bad_status_422"] = (
        jclient.get("/harness/jobs", params={"status": "bogus"}).status_code == 422
    )
    # drain: new submissions refused; existing records still readable
    jclient.post("/harness/drain")
    out["job_drain_refuses_submit"] = (
        jclient.post("/harness/jobs", json={"command": "doctor"}).status_code == 503
    )
    out["job_status_under_drain"] = jclient.get(f"/harness/jobs/{job_id}").status_code == 200
    out["job_replay_under_drain"] = (
        jclient.post(
            "/harness/jobs",
            json={"command": "doctor"},
            headers={"Idempotency-Key": "job-key-1"},
        ).json()["replayed"]
        is True
    )
    # store bound: evicting the oldest job drops its idem mapping
    tiny_jobs = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        job_max=2,
    )
    tc3 = _TC2(tiny_jobs)
    ids = [tc3.post("/harness/jobs", json={"command": "doctor"}).json()["job_id"] for _ in range(3)]
    out["job_bound_evicts"] = (
        tc3.get(f"/harness/jobs/{ids[0]}").status_code == 404
        and tc3.get(f"/harness/jobs/{ids[2]}").status_code == 200
    )
    try:
        api_mod.create_app(job_max=0)
        out["job_max_validated"] = False
    except ValueError:
        out["job_max_validated"] = True

    # cooperative cancel: occupy both executor workers without slots
    # (slots == workers, so jobs only stay queued when a worker is busy
    # without holding one) -> both jobs queue deterministically.
    qapp = api_mod.create_app(
        harness=_Harness(runner=_slow_runner),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        max_inflight=2,
    )
    qc = _TC2(qapp)
    qapp.state.jobs_executor.submit(lambda: time.sleep(2.5))
    qapp.state.jobs_executor.submit(lambda: time.sleep(2.5))
    j1 = qc.post("/harness/jobs", json={"command": "doctor"}).json()["job_id"]
    j2 = qc.post("/harness/jobs", json={"command": "doctor"}).json()["job_id"]
    out["job_queues_when_workers_busy"] = (
        qc.get(f"/harness/jobs/{j2}").json()["status"] == "queued"
        and qc.get(f"/harness/jobs/{j1}").json()["status"] == "queued"
    )
    cancelled = qc.delete(f"/harness/jobs/{j2}")
    out["job_cancel_queued_200"] = (
        cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
    )
    # DELETE is idempotent: re-cancelling the cancelled job re-reads 200
    out["job_cancel_idempotent_200"] = qc.delete(f"/harness/jobs/{j2}").status_code == 200
    out["job_cancel_unknown_404"] = qc.delete("/harness/jobs/nope").status_code == 404
    out["job_cancelled_never_runs"] = (
        qc.get(f"/harness/jobs/{j2}").json()["result"] is None
        and qc.get(f"/harness/jobs/{j2}").json()["finished_at"] is not None
    )
    # once a worker frees, j1 dequeues; deleting a live or finished job is 409
    dl = time.monotonic() + 10.0
    while qc.get(f"/harness/jobs/{j1}").json()["status"] == "queued" and time.monotonic() < dl:
        time.sleep(0.05)
    out["job_cancel_active_409"] = qc.delete(f"/harness/jobs/{j1}").status_code == 409
    dl = time.monotonic() + 10.0
    while qc.get(f"/harness/jobs/{j1}").json()["status"] not in ("succeeded", "failed") and (
        time.monotonic() < dl
    ):
        time.sleep(0.05)
    out["job_cancel_succeeded_409"] = qc.delete(f"/harness/jobs/{j1}").status_code == 409
    # the cancelled job's slot frees once its executor task dequeues
    dl = time.monotonic() + 10.0
    while qc.get("/metrics").json()["inflight"] != 0 and time.monotonic() < dl:
        time.sleep(0.1)
    out["job_cancel_slot_recovered"] = qc.get("/metrics").json()["inflight"] == 0

    # --- webhooks: terminal-state job callbacks --------------------------------
    # A submitted job's record is POSTed to its callback_url on every
    # terminal transition (succeeded/failed/cancelled); delivery is
    # best-effort — a dead or erroring endpoint is recorded on the job,
    # never raised into the worker.
    import json as _json  # noqa: PLC0415
    import socket as _socket  # noqa: PLC0415
    import threading as _threading  # noqa: PLC0415
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: PLC0415

    _cb_hits: list[dict[str, Any]] = []
    _cb_raw: list[bytes] = []
    _cb_hdrs: list[dict[str, str]] = []
    _cb_path_n: dict[str, int] = {}

    class _JobHook(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 — http.server handler name
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n)
            _cb_raw.append(raw)
            _cb_hdrs.append(dict(self.headers.items()))
            _cb_hits.append(_json.loads(raw))
            _cb_path_n[self.path] = _cb_path_n.get(self.path, 0) + 1
            if self.path == "/fail":
                self.send_response(500)
            elif self.path == "/flaky":
                self.send_response(500 if _cb_path_n[self.path] < 3 else 200)
            elif self.path == "/reject":
                self.send_response(404)
            else:
                self.send_response(200)
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            pass

    cb_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        max_inflight=2,
    )
    cbc = _TC2(cb_app)
    cb_srv = ThreadingHTTPServer(("127.0.0.1", 0), _JobHook)
    cb_thread = _threading.Thread(target=cb_srv.serve_forever, daemon=True)
    cb_thread.start()
    cb_url = f"http://127.0.0.1:{cb_srv.server_address[1]}/hook"
    _dead_sock = _socket.socket()
    _dead_sock.bind(("127.0.0.1", 0))
    _dead_port = _dead_sock.getsockname()[1]
    _dead_sock.close()
    # The probes below submit loopback callback_urls — the SSRF guard
    # refuses them unless the private-networks opt-in is set for this leg,
    # read per request (saved/restored verbatim).
    saved_webhook_env = os.environ.get("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS")
    os.environ["FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"] = "1"
    try:
        ok_job = cbc.post("/harness/jobs", json={"command": "doctor", "callback_url": cb_url})
        jid_cb = ok_job.json()["job_id"]
        deadline = time.monotonic() + 10.0
        st_cb: dict[str, Any] = {}
        while time.monotonic() < deadline:
            st_cb = cbc.get(f"/harness/jobs/{jid_cb}").json()
            if st_cb["status"] == "succeeded" and st_cb.get("callback_status"):
                break
            time.sleep(0.05)
        out["callback_delivered_on_success"] = (
            ok_job.status_code == 202
            and ok_job.headers.get("location") == f"/harness/jobs/{jid_cb}"
            and st_cb.get("callback_status") == "delivered"
            and st_cb.get("callback_url") == cb_url
            and len(_cb_hits) == 1
            and _cb_hits[0]["job_id"] == jid_cb
            and _cb_hits[0]["status"] == "succeeded"
            and _cb_hits[0]["result"]["ok"] is True
        )
        # endpoint returning 5xx -> recorded as failed, job unaffected
        err_job = cbc.post(
            "/harness/jobs",
            json={"command": "doctor", "callback_url": cb_url.replace("/hook", "/fail")},
        )
        jid_err = err_job.json()["job_id"]
        deadline = time.monotonic() + 10.0
        st_err: dict[str, Any] = {}
        while time.monotonic() < deadline:
            st_err = cbc.get(f"/harness/jobs/{jid_err}").json()
            if st_err.get("callback_attempts") == 3 and _cb_path_n.get("/fail") == 3:
                break
            time.sleep(0.05)
        out["callback_http_error_recorded"] = (
            st_err["status"] == "succeeded"
            and st_err.get("callback_status") == "failed"
            and "500" in (st_err.get("callback_error") or "")
        )
        out["callback_5xx_retried_3x"] = (
            _cb_path_n.get("/fail") == 3 and st_err.get("callback_attempts") == 3
        )

        # transient 5xx sequence eventually delivers; 4xx never retries.
        # Delivery retries hold a worker briefly — a submit can race a
        # saturated inflight cap, so re-submit until a slot frees.
        def _submit_cb(url: str) -> str:
            end = time.monotonic() + 10.0
            while True:
                r = cbc.post(
                    "/harness/jobs",
                    json={"command": "doctor", "callback_url": url},
                )
                if r.status_code == 202:
                    return str(r.json()["job_id"])
                if r.status_code != 503 or time.monotonic() > end:
                    raise AssertionError(f"callback submit failed: {r.status_code} {r.text}")
                time.sleep(0.1)

        jid_fk = _submit_cb(cb_url.replace("/hook", "/flaky"))
        deadline = time.monotonic() + 10.0
        st_fk: dict[str, Any] = {}
        while time.monotonic() < deadline:
            st_fk = cbc.get(f"/harness/jobs/{jid_fk}").json()
            if st_fk.get("callback_status") == "delivered" and _cb_path_n.get("/flaky") == 3:
                break
            time.sleep(0.05)
        out["callback_flaky_delivers"] = (
            st_fk.get("callback_status") == "delivered"
            and st_fk.get("callback_attempts") == 3
            and _cb_path_n.get("/flaky") == 3
        )
        jid_rj = _submit_cb(cb_url.replace("/hook", "/reject"))
        deadline = time.monotonic() + 10.0
        st_rj: dict[str, Any] = {}
        while time.monotonic() < deadline:
            st_rj = cbc.get(f"/harness/jobs/{jid_rj}").json()
            if st_rj["status"] == "succeeded" and st_rj.get("callback_status"):
                break
            time.sleep(0.05)
        out["callback_4xx_never_retried"] = (
            st_rj.get("callback_status") == "failed"
            and st_rj.get("callback_attempts") == 1
            and _cb_path_n.get("/reject") == 1
        )
        # dead endpoint -> connection error recorded, job unaffected
        jid_dead = _submit_cb(f"http://127.0.0.1:{_dead_port}/hook")
        deadline = time.monotonic() + 15.0
        st_dead: dict[str, Any] = {}
        while time.monotonic() < deadline:
            st_dead = cbc.get(f"/harness/jobs/{jid_dead}").json()
            if (
                st_dead["status"] == "succeeded"
                and st_dead.get("callback_status")
                and st_dead.get("callback_attempts") == 3
            ):
                break
            time.sleep(0.05)
        out["callback_dead_endpoint_recorded"] = (
            st_dead["status"] == "succeeded"
            and st_dead.get("callback_status") == "failed"
            and bool(st_dead.get("callback_error"))
        )
        # cancelling a queued job is a terminal transition — it fires too
        cb_app.state.jobs_executor.submit(lambda: time.sleep(3.0))
        cb_app.state.jobs_executor.submit(lambda: time.sleep(3.0))
        qid = _submit_cb(cb_url)
        cxl = cbc.delete(f"/harness/jobs/{qid}")
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and (
            len(_cb_hits) < 3 or _cb_hits[-1].get("job_id") != qid
        ):
            time.sleep(0.05)
        out["callback_fires_on_cancel"] = (
            cxl.status_code == 200
            and cxl.json()["status"] == "cancelled"
            and cxl.json().get("callback_status") == "delivered"
            and _cb_hits[-1]["job_id"] == qid
            and _cb_hits[-1]["status"] == "cancelled"
        )
        dl = time.monotonic() + 10.0
        while cbc.get("/metrics").json()["inflight"] != 0 and time.monotonic() < dl:
            time.sleep(0.1)
        bad = cbc.post(
            "/harness/jobs",
            json={"command": "doctor", "callback_url": "ftp://x/hook"},
        )
        nohost = cbc.post(
            "/harness/jobs",
            json={"command": "doctor", "callback_url": "http:///hook"},
        )
        out["callback_bad_url_422"] = bad.status_code == 422 and nohost.status_code == 422
        # unsigned deliveries carry no signature headers
        out["callback_unsigned_no_sig"] = (
            "X-Fx1-Webhook-Signature" not in _cb_hdrs[0]
            and "X-Fx1-Webhook-Timestamp" not in _cb_hdrs[0]
        )
        # callback_secret HMAC-signs the delivery; receivers verify the raw
        # body — the job record itself never echoes the secret.
        from fx1.serve.webhooks import verify_webhook  # noqa: PLC0415

        sig_job = cbc.post(
            "/harness/jobs",
            json={
                "command": "doctor",
                "callback_url": cb_url,
                "callback_secret": "whsec-test",
            },
        )
        jid_sig = sig_job.json()["job_id"]
        deadline = time.monotonic() + 10.0
        st_sig: dict[str, Any] = {}
        while time.monotonic() < deadline:
            st_sig = cbc.get(f"/harness/jobs/{jid_sig}").json()
            if st_sig["status"] == "succeeded" and st_sig.get("callback_status"):
                break
            time.sleep(0.05)
        sig_hdrs = _cb_hdrs[-1]
        out["callback_signed_verifies"] = (
            st_sig.get("callback_status") == "delivered"
            and sig_hdrs.get("X-Fx1-Webhook-Signature", "").startswith("sha256=")
            and verify_webhook(
                "whsec-test",
                sig_hdrs.get("X-Fx1-Webhook-Timestamp"),
                sig_hdrs.get("X-Fx1-Webhook-Signature"),
                _cb_raw[-1],
            )
        )
        out["callback_secret_not_echoed"] = (
            "callback_secret" not in st_sig
            and "whsec-test" not in cbc.get(f"/harness/jobs/{jid_sig}").text
            and b"whsec-test" not in _cb_raw[-1]
        )
        good_sig = sig_hdrs.get("X-Fx1-Webhook-Signature")
        good_ts = sig_hdrs.get("X-Fx1-Webhook-Timestamp")
        out["webhook_verify_tampered_body"] = not verify_webhook(
            "whsec-test", good_ts, good_sig, _cb_raw[-1] + b" "
        )
        out["webhook_verify_wrong_secret"] = not verify_webhook(
            "whsec-other", good_ts, good_sig, _cb_raw[-1]
        )
        out["webhook_verify_stale_ts"] = not verify_webhook(
            "whsec-test", "1", good_sig, _cb_raw[-1]
        )
        out["webhook_verify_malformed"] = not verify_webhook(
            "whsec-test", good_ts, "nonsense", _cb_raw[-1]
        ) and not verify_webhook("whsec-test", good_ts, None, _cb_raw[-1])
        sec_no_url = cbc.post(
            "/harness/jobs",
            json={"command": "doctor", "callback_secret": "whsec-test"},
        )
        out["callback_secret_requires_url_422"] = sec_no_url.status_code == 422
    finally:
        cb_srv.shutdown()
        cb_srv.server_close()
        if saved_webhook_env is None:
            os.environ.pop("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS", None)
        else:
            os.environ["FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"] = saved_webhook_env

    out["rate_limit_default_off"] = all(client.get("/health").status_code == 200 for _ in range(8))
    try:
        api_mod.create_app(rate_limit_rps=-1)
        out["rate_limit_negative_rejected"] = False
    except ValueError:
        out["rate_limit_negative_rejected"] = True
    limited = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        rate_limit_rps=5.0,
    )
    lc = _TC2(limited)
    hits = [lc.get("/harness/version").status_code for _ in range(7)]
    denied = lc.get("/harness/version")
    out["rate_limit_429"] = hits[:5] == [200] * 5 and 429 in hits[5:] + [denied.status_code]
    out["rate_limit_envelope"] = (
        denied.status_code == 429
        and denied.json()["code"] == "too_many_requests"
        and denied.headers.get("retry-after") is not None
        and int(denied.headers["retry-after"]) >= 1
    )
    _clk["t"] += 1.0
    out["rate_limit_recovers"] = lc.get("/harness/version").status_code == 200
    # X-RateLimit-* headers on every response while the limiter is active
    limited3 = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        rate_limit_rps=10.0,
    )
    lc3 = _TC2(limited3)
    spec_limited = lc3.get("/openapi.json").json()
    first = lc3.get("/harness/version")
    second = lc3.get("/harness/version")
    out["rate_limit_headers_on_success"] = (
        first.status_code == 200
        and first.headers.get("x-ratelimit-limit") == "10"
        and first.headers.get("x-ratelimit-remaining") == "8"
        and int(first.headers["x-ratelimit-reset"]) >= 0
        and int(second.headers["x-ratelimit-remaining"]) == 7
    )
    for _ in range(16):
        if lc3.get("/harness/version").status_code != 200:
            break
    denied3 = lc3.get("/harness/version")
    out["rate_limit_headers_on_429"] = (
        denied3.status_code == 429
        and denied3.headers.get("x-ratelimit-limit") == "10"
        and denied3.headers.get("x-ratelimit-remaining") == "0"
        and int(denied3.headers["x-ratelimit-reset"]) >= 1
    )
    out["rate_limit_headers_absent_when_off"] = (
        "x-ratelimit-limit" not in client.get("/health").headers
    )
    # public paths are exempt: a health probe must not consume the budget —
    # with a 1 rps bucket, /version takes the only token, /health still 200s
    # (no bucket touch, no limiter headers), and the next /version 429s.
    lim_pub = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        rate_limit_rps=1.0,
    )
    lp = _TC2(lim_pub)
    lp.get("/harness/version")
    h_exempt = lp.get("/health")
    v_after = lp.get("/harness/version")
    out["rate_limit_public_path_exempt"] = (
        h_exempt.status_code == 200
        and "x-ratelimit-limit" not in h_exempt.headers
        and v_after.status_code == 429
    )
    # the spec declares the headers the middleware sets — generated clients
    # see them typed instead of having to know
    spec_main = client.get("/openapi.json").json()
    common = {
        "X-Request-ID",
        "X-Fx1-Api-Version",
        "X-Content-Type-Options",
        "Cache-Control",
        "Referrer-Policy",
        "Openai-Processing-Ms",
        "X-RateLimit-Limit-Requests",
        "X-RateLimit-Remaining-Requests",
        "X-RateLimit-Reset-Requests",
    }
    spec_ops = [
        op
        for item in spec_main.get("paths", {}).values()
        for op in item.values()
        if isinstance(op, dict)
    ]
    out["openapi_declares_common_headers"] = bool(spec_ops) and all(
        common <= set(resp.get("headers", {}))
        for op in spec_ops
        for resp in op.get("responses", {}).values()
    )
    # OpenAI's processing-ms tracing header rides every response — 2xx,
    # 4xx and 5xx alike — and parses as a non-negative int
    _pm_ok = client.get("/harness/version")
    _pm_4xx = client.get("/harness/jobs/does-not-exist")
    _pm_5xx = client.post("/harness/complete", json={"prompt": "x", "seed": 1})
    out["processing_ms_header"] = all(
        int(r.headers["openai-processing-ms"]) >= 0 for r in (_pm_ok, _pm_4xx, _pm_5xx)
    )
    submit_op = spec_main["paths"]["/harness/jobs"]["post"]
    out["openapi_declares_location_202"] = "Location" in submit_op["responses"]["202"].get(
        "headers", {}
    )
    out["openapi_declares_retry_after_503"] = all(
        "Retry-After" in resp.get("headers", {})
        for op in spec_ops
        for code, resp in op.get("responses", {}).items()
        if code in ("429", "503")
    )
    out["openapi_ratelimit_absent_when_off"] = all(
        "X-RateLimit-Limit" not in resp.get("headers", {})
        for op in spec_ops
        for resp in op.get("responses", {}).values()
    )
    out["openapi_ratelimit_present_when_on"] = all(
        {"X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"}
        <= set(resp.get("headers", {}))
        for item in spec_limited.get("paths", {}).values()
        for op in item.values()
        if isinstance(op, dict)
        for resp in op.get("responses", {}).values()
    )
    # CORS: closed by default; explicit origins only; preflight handled
    # before the auth layer (preflights carry no credentials)
    pre_off = client.options(
        "/health",
        headers={
            "Origin": "https://fx1.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    out["cors_off_by_default"] = "access-control-allow-origin" not in pre_off.headers
    cors_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        cors_origins="https://fx1.example.com, https://ops.internal:8443",
    )
    cc = _TC2(cors_app)
    pre_on = cc.options(
        "/health",
        headers={
            "Origin": "https://fx1.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    out["cors_preflight_ok"] = (
        pre_on.status_code == 200
        and pre_on.headers.get("access-control-allow-origin") == "https://fx1.example.com"
        and "POST" in pre_on.headers.get("access-control-allow-methods", "")
    )
    actual_resp = cc.get("/health", headers={"Origin": "https://fx1.example.com"})
    out["cors_exposes_stamped_headers"] = (
        "x-fx1-api-version" in actual_resp.headers.get("access-control-expose-headers", "").lower()
    )
    out["cors_wrong_origin_refused"] = (
        "access-control-allow-origin"
        not in cc.options(
            "/health",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "GET",
            },
        ).headers
    )
    out["cors_actual_response_marked"] = (
        actual_resp.headers.get("access-control-allow-origin") == "https://fx1.example.com"
    )
    try:
        api_mod.create_app(cors_origins="*")
        out["cors_wildcard_refused"] = False
    except ValueError:
        out["cors_wildcard_refused"] = True
    out["capabilities_reports_cors"] = (
        client.get("/harness/capabilities").json()["features"]["cors"] is False
        and cc.get("/harness/capabilities").json()["features"]["cors"] is True
    )

    # Backend circuit breaker: consecutive call faults open the circuit —
    # later calls fast-fail 503 + Retry-After without touching the backend;
    # a half-open probe admits after cooldown and closes on success.
    class _FlakyBackend:
        def __init__(self) -> None:
            self.calls = 0

        def complete(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> str:
            self.calls += 1
            if self.calls <= 2:
                raise RuntimeError("backend exploded")
            return "ok"

    _flaky = _FlakyBackend()
    brk_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda name, *a, **k: _flaky if name == "byok" else _CleanBackend(),
        breaker_threshold=2,
        breaker_cooldown_s=60.0,
    )
    bc = _TC2(brk_app)
    cp_body = {"backend": "byok", "messages": [{"role": "user", "content": "x"}]}
    r_fail1 = bc.post("/harness/complete", json=cp_body)
    r_fail2 = bc.post("/harness/complete", json=cp_body)
    r_open = bc.post("/harness/complete", json=cp_body)
    out["breaker_opens_at_threshold"] = (
        r_fail1.status_code == 502
        and r_fail2.status_code == 502
        and r_open.status_code == 503
        and r_open.json()["code"] == "backend_unavailable"
        and "circuit open" in r_open.json()["detail"]
        and "retry-after" in {k.lower() for k in r_open.headers}
        and _flaky.calls == 2  # the open-circuit call never reached the backend
    )
    st = bc.get("/harness/backends").json()
    out["backends_route_reports_state"] = (
        set(st) == {"hosted_k3", "byok", "local_fx1"}
        and st["byok"]["circuit_open"] is True
        and st["byok"]["consecutive_failures"] >= 2
        and st["hosted_k3"]["circuit_open"] is False
    )
    r_other = bc.post(
        "/harness/complete",
        json={"backend": "hosted_k3", "messages": [{"role": "user", "content": "x"}]},
    )
    out["breaker_isolated_per_backend"] = r_other.status_code == 200
    caps_brk = bc.get("/harness/capabilities").json()
    out["capabilities_reports_breaker"] = (
        caps_brk["features"]["breaker"] is True
        and caps_brk["limits"]["breaker_threshold"] == 2.0
        and caps_brk["limits"]["breaker_cooldown_s"] == 60.0
    )
    brk = api_mod._BackendBreaker(2, 0.05)
    brk.report("x", False)
    brk.report("x", False)
    opened = brk.check("x") > 0
    time.sleep(0.06)
    probe1 = brk.check("x")  # admits the single half-open probe
    probe2 = brk.check("x")  # a second concurrent caller fast-fails
    brk.report("x", True)  # probe succeeded → closed
    closed = brk.check("x") == 0 and brk.state("x")[0] is False
    out["breaker_half_open_recovers"] = opened and probe1 == 0.0 and probe2 > 0 and closed
    off_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        breaker_threshold=0,
    )
    oc = _TC2(off_app)
    out["breaker_disabled_by_zero"] = (
        oc.get("/harness/capabilities").json()["features"]["breaker"] is False
        and oc.get("/harness/backends").json()["byok"]["circuit_open"] is False
    )
    # honesty-gate refusals are model output, not backend health — they
    # must never trip the circuit.
    dirty_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _DirtyBackend(),
        breaker_threshold=2,
        breaker_cooldown_s=60.0,
    )
    dc = _TC2(dirty_app)
    gate_fails = [
        dc.post(
            "/harness/complete",
            json={"backend": "byok", "messages": [{"role": "user", "content": "x"}]},
        ).status_code
        for _ in range(3)
    ]
    out["breaker_ignores_honesty_gate"] = (
        gate_fails == [502, 502, 502]
        and dc.get("/harness/backends").json()["byok"]["circuit_open"] is False
    )

    # Backend fallback chain: `fallbacks` advances only on availability
    # faults — unconfigured (503 resolve), call faults (502/503), or an
    # open circuit. A gate refusal or client error is a verdict, not a
    # retry signal, and never reaches the next link. Every tried link is
    # sealed in `attempts` on the response AND the completion record.
    class _BoomBackend:
        def __init__(self) -> None:
            self.calls = 0

        def complete(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> str:
            self.calls += 1
            raise RuntimeError("backend exploded")

    _boom = _BoomBackend()
    chain_app = _TC2(
        api_mod.create_app(
            backend_resolver=lambda name, *a, **k: _boom if name == "hosted_k3" else _CleanBackend()
        )
    )
    r_chain = chain_app.post(
        "/harness/complete",
        json={
            "backend": "hosted_k3",
            "fallbacks": ["byok"],
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    chain_body = r_chain.json()
    out["fallback_serves_next_link"] = (
        r_chain.status_code == 200
        and chain_body["backend"] == "byok"
        and chain_body["content"] == "clean:hi"
        and [(a["backend"], a["ok"]) for a in chain_body["attempts"]]
        == [("hosted_k3", False), ("byok", True)]
        and chain_body["attempts"][0]["error_class"] == "RuntimeError"
    )
    # the sealed completion record carries the same chain evidence and is
    # filed under the serving backend.
    rec_chain = chain_app.get("/harness/completions?limit=1").json()["items"][0]
    out["fallback_record_carries_attempts"] = (
        rec_chain["backend"] == "byok"
        and [a["backend"] for a in rec_chain["attempts"]] == ["hosted_k3", "byok"]
        and chain_body["completion_id"] == rec_chain["completion_id"]
    )

    # resolve-level fallback: a 503 from resolution (unconfigured primary)
    # advances the chain before any model call.
    def _resolve_dies(name: str, *a: Any, **k: Any) -> Any:
        if name == "byok":
            return _CleanBackend()
        raise RuntimeError("primary unconfigured")

    res_app = _TC2(api_mod.create_app(backend_resolver=_resolve_dies))
    r_res = res_app.post(
        "/harness/complete",
        json={
            "backend": "hosted_k3",
            "fallbacks": ["byok"],
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    res_body = r_res.json()
    out["fallback_resolve_503_advances"] = (
        r_res.status_code == 200
        and res_body["backend"] == "byok"
        and res_body["attempts"][0]["backend"] == "hosted_k3"
        and res_body["attempts"][0]["error_class"] == "backend_unavailable"
    )
    # a gate refusal is a verdict — the request aborts, the fallback is
    # never spent.
    _clean_spy = _CleanBackend()
    ref_app = _TC2(
        api_mod.create_app(
            backend_resolver=lambda name, *a, **k: (
                _DirtyBackend() if name == "hosted_k3" else _clean_spy
            )
        )
    )
    r_ref = ref_app.post(
        "/harness/complete",
        json={
            "backend": "hosted_k3",
            "fallbacks": ["byok"],
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    out["fallback_no_retry_on_honesty_gate"] = (
        r_ref.status_code == 502
        and r_ref.json()["code"] == "honesty_gate"
        and _clean_spy.calls == 0
    )
    # a client error (bad citation) likewise aborts before any backend.
    r_bad = chain_app.post(
        "/harness/complete",
        json={
            "backend": "hosted_k3",
            "fallbacks": ["byok"],
            "messages": [{"role": "user", "content": "hi"}],
            "receipt_hashes": ["0" * 64],
        },
    )
    out["fallback_client_error_aborts"] = r_bad.status_code == 422
    # every link dead → the last retriable verdict surfaces, with the full
    # chain sealed on the record.
    dead_app = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _BoomBackend()))
    r_dead = dead_app.post(
        "/harness/complete",
        json={
            "backend": "hosted_k3",
            "fallbacks": ["local_fx1", "byok"],
            # lets the local_fx1 link resolve (resolver ignores kwargs —
            # every link faults at the call itself)
            "checkpoint_dir": "/nonexistent/fx1-cp",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    dead_rec = dead_app.get("/harness/completions?limit=1").json()["items"][0]
    out["fallback_exhausted_502"] = (
        r_dead.status_code == 502
        and r_dead.json()["code"] == "backend_failure"
        and [a["backend"] for a in dead_rec["attempts"]] == ["hosted_k3", "local_fx1", "byok"]
        and all(a["ok"] is False for a in dead_rec["attempts"])
    )
    # an open primary circuit advances the chain — a downed backend's
    # breaker never blocks a healthy fallback.
    open_app = _TC2(
        api_mod.create_app(
            backend_resolver=lambda name, *a, **k: (
                _boom if name == "hosted_k3" else _CleanBackend()
            ),
            breaker_threshold=1,
            breaker_cooldown_s=60.0,
        )
    )
    open_app.post(
        "/harness/complete",
        json={"backend": "hosted_k3", "messages": [{"role": "user", "content": "x"}]},
    )
    r_open_fb = open_app.post(
        "/harness/complete",
        json={
            "backend": "hosted_k3",
            "fallbacks": ["byok"],
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    fb_body = r_open_fb.json()
    out["fallback_skips_open_circuit"] = (
        r_open_fb.status_code == 200
        and fb_body["backend"] == "byok"
        and fb_body["attempts"][0]["error_class"] == "backend_unavailable"
    )
    # chain validation: repeats, self-reference, over-cap, and kwargs
    # bound to a link absent from the chain all fail closed at 422.
    _msgs = [{"role": "user", "content": "x"}]
    bad_chains: list[dict[str, Any]] = [
        {"backend": "byok", "fallbacks": ["byok"], "messages": _msgs},
        {"backend": "byok", "fallbacks": ["hosted_k3", "hosted_k3"], "messages": _msgs},
        {
            "backend": "byok",
            "fallbacks": ["hosted_k3", "local_fx1", "byok"],
            "messages": _msgs,
        },
        {
            "backend": "hosted_k3",
            "messages": _msgs,
            "byok": {"base_url": "https://x.example.com", "api_key": "k", "model": "m"},
        },
        {"backend": "hosted_k3", "messages": _msgs, "checkpoint_dir": "/x"},
        {
            "backend": "hosted_k3",
            "fallbacks": ["hosted_k3"],
            "messages": _msgs,
        },
    ]
    out["fallback_validation_422"] = all(
        chain_app.post("/harness/complete", json=b).status_code == 422 for b in bad_chains
    ) and all(
        res_app.post(
            "/harness/complete/batch",
            json={k: v for k, v in b.items() if k != "messages"} | {"batch": [_msgs]},
        ).status_code
        == 422
        for b in bad_chains
    )
    # batch applies the chain at resolve level — one link serves the whole
    # batch, per-item usage attribution stays honest.
    r_batch_fb = res_app.post(
        "/harness/complete/batch",
        json={
            "backend": "hosted_k3",
            "fallbacks": ["byok"],
            "batch": [[{"role": "user", "content": "a"}]],
        },
    )
    batch_fb = r_batch_fb.json()
    out["fallback_batch_resolve_level"] = (
        r_batch_fb.status_code == 200
        and batch_fb["backend"] == "byok"
        and batch_fb["results"][0]["ok"] is True
        and [a["backend"] for a in batch_fb["attempts"]] == ["hosted_k3", "byok"]
    )

    # streaming resolves through the same chain before any byte commits.
    def _resolve_dies_stream(name: str, *a: Any, **k: Any) -> Any:
        if name == "byok":
            return _StreamBackend()
        raise RuntimeError("primary unconfigured")

    stream_app = _TC2(api_mod.create_app(backend_resolver=_resolve_dies_stream))
    r_stream_fb = stream_app.post(
        "/harness/complete/stream",
        json={
            "backend": "hosted_k3",
            "fallbacks": ["byok"],
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    stream_lines = [ln for ln in r_stream_fb.text.splitlines() if ln.startswith("data: ")]
    out["fallback_stream_resolves"] = r_stream_fb.status_code == 200 and any(
        '"type": "final"' in ln for ln in stream_lines
    )

    # Per-request BYOK: the caller's {base_url, api_key, model} rides the
    # request body and reaches the backend factory; overrides are validated
    # (422, never breaker-counted), gated by a server flag, and breaker-
    # isolated per endpoint so one tenant's dead endpoint can't fast-fail
    # another's.
    _cap: list[dict[str, Any]] = []
    _flaky2 = _FlakyBackend()
    _flaky2.calls = 0

    def _cap_resolver(name: str, cp: str | None = None, **kw: Any) -> Any:
        _cap.append(kw)
        if name == "byok" and "dead" in kw.get("base_url", ""):
            return _flaky2
        return _CleanBackend()

    bapp = _TC2(api_mod.create_app(backend_resolver=_cap_resolver))
    ovr = {"base_url": "https://llm.example.com/v1", "api_key": "sk-live-x", "model": "m1"}
    ok = bapp.post(
        "/harness/complete",
        json={
            "backend": "byok",
            "messages": [{"role": "user", "content": "x"}],
            "byok": ovr,
        },
    )
    out["byok_override_reaches_backend"] = ok.status_code == 200 and _cap[-1] == ovr
    wrong_be = bapp.post(
        "/harness/complete",
        json={
            "backend": "hosted_k3",
            "messages": [{"role": "user", "content": "x"}],
            "byok": ovr,
        },
    )
    out["byok_override_non_byok_422"] = wrong_be.status_code == 422
    out["byok_override_bad_url_422"] = (
        bapp.post(
            "/harness/complete",
            json={
                "backend": "byok",
                "messages": [{"role": "user", "content": "x"}],
                "byok": {**ovr, "base_url": "ftp://x"},
            },
        ).status_code
        == 422
    )
    out["byok_override_partial_422"] = (
        bapp.post(
            "/harness/complete",
            json={
                "backend": "byok",
                "messages": [{"role": "user", "content": "x"}],
                "byok": {"base_url": "https://e.com", "api_key": "k"},
            },
        ).status_code
        == 422
    )
    disabled = _TC2(api_mod.create_app(backend_resolver=_cap_resolver, byok_override=False))
    dresp = disabled.post(
        "/harness/complete",
        json={
            "backend": "byok",
            "messages": [{"role": "user", "content": "x"}],
            "byok": ovr,
        },
    )
    out["byok_override_disabled_422"] = (
        dresp.status_code == 422 and dresp.json().get("code") == "byok_override_disabled"
    )
    caps_byok = bapp.get("/harness/capabilities").json()["features"]
    out["capabilities_reports_byok_override"] = (
        caps_byok["byok_override"] is True
        and _TC2(api_mod.create_app(backend_resolver=_cap_resolver, byok_override=False))
        .get("/harness/capabilities")
        .json()["features"]["byok_override"]
        is False
    )
    # Per-request backend deadline — flows into resolver kwargs on every
    # surface; out-of-range values are model-level 422s.
    _cap.clear()
    t_ok = bapp.post(
        "/harness/complete",
        json={
            "backend": "byok",
            "messages": [{"role": "user", "content": "x"}],
            "byok": ovr,
            "timeout_s": 2.5,
        },
    )
    out["timeout_s_reaches_backend"] = t_ok.status_code == 200 and _cap[-1].get("timeout_s") == 2.5
    out["timeout_s_nonpositive_422"] = (
        bapp.post(
            "/harness/complete",
            json={
                "backend": "byok",
                "messages": [{"role": "user", "content": "x"}],
                "timeout_s": 0,
            },
        ).status_code
        == 422
    )
    out["timeout_s_over_cap_422"] = (
        bapp.post(
            "/harness/complete",
            json={
                "backend": "byok",
                "messages": [{"role": "user", "content": "x"}],
                "timeout_s": 99999,
            },
        ).status_code
        == 422
    )
    _cap.clear()
    bapp.post(
        "/harness/complete/batch",
        json={
            "backend": "byok",
            "batch": [[{"role": "user", "content": "x"}]],
            "byok": ovr,
            "timeout_s": 7.0,
        },
    )
    out["timeout_s_batch_reaches_backend"] = _cap[-1].get("timeout_s") == 7.0
    _cap.clear()
    bapp.post(
        "/harness/complete/stream",
        json={
            "backend": "byok",
            "messages": [{"role": "user", "content": "x"}],
            "byok": ovr,
            "timeout_s": 3.0,
        },
    )
    # _CleanBackend lacks stream() — the resolver still saw the deadline
    # before the 501.
    out["timeout_s_stream_reaches_backend"] = _cap[-1].get("timeout_s") == 3.0
    # The hosted backend honors the knob it previously hardcoded.
    import fx1.serve.backends as be_mod  # noqa: PLC0415

    out["timeout_s_hosted_backend"] = (
        be_mod.HostedK3Backend(api_key="k", timeout_s=9.0)._timeout_s == 9.0
    )
    # X-Fx1-Timeout header feeds the same per-request deadline on the
    # OpenAI surface (chat + responses + embeddings share the resolver);
    # the body's fx1.timeout_s extension wins; a malformed or out-of-range
    # header is a fail-closed 400. Batches inherit it — the submitter's
    # X-Fx1-* set replays per line.
    _cap.clear()
    hh = bapp.post(
        "/v1/chat/completions",
        headers={"X-Fx1-Timeout": "7"},
        json={"model": "fx1", "messages": [{"role": "user", "content": "x"}]},
    )
    out["xfx_timeout_header_reaches_backend"] = (
        hh.status_code == 200 and _cap[-1].get("timeout_s") == 7.0
    )
    _cap.clear()
    hh2 = bapp.post(
        "/v1/chat/completions",
        headers={"X-Fx1-Timeout": "7"},
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "x"}],
            "fx1": {"timeout_s": 3},
        },
    )
    out["xfx_timeout_ext_wins"] = hh2.status_code == 200 and _cap[-1].get("timeout_s") == 3.0
    out["xfx_timeout_bad_400"] = (
        bapp.post(
            "/v1/chat/completions",
            headers={"X-Fx1-Timeout": "bogus"},
            json={"model": "fx1", "messages": [{"role": "user", "content": "x"}]},
        ).status_code
        == 400
    )
    out["xfx_timeout_range_400"] = (
        bapp.post(
            "/v1/chat/completions",
            headers={"X-Fx1-Timeout": "99999"},
            json={"model": "fx1", "messages": [{"role": "user", "content": "x"}]},
        ).status_code
        == 400
    )
    _cap.clear()
    hr = bapp.post(
        "/v1/responses",
        headers={"X-Fx1-Timeout": "4"},
        json={"model": "fx1", "input": "x"},
    )
    out["xfx_timeout_responses_reaches_backend"] = (
        hr.status_code == 200 and _cap[-1].get("timeout_s") == 4.0
    )

    # X-Fx1-Receipt-Sha256 — every gated response self-describes the seal of
    # its completion-log record: the header equals the receipt_sha256 of the
    # document GET /harness/completions/{id}/receipt exports, so the wire is
    # evidence-pinned without a second fetch. Idempotent replays echo the
    # original seal; SSE responses carry it as a header.
    rc = bapp.post(
        "/v1/chat/completions",
        json={"model": "fx1", "messages": [{"role": "user", "content": "x"}]},
    )
    rc_doc = bapp.get(f"/harness/completions/{rc.headers.get('x-fx1-completion-id')}/receipt")
    out["receipt_sha_header_chat"] = (
        rc.status_code == 200
        and rc.headers.get("x-fx1-receipt-sha256") == rc_doc.json()["receipt_sha256"]
    )
    rr = bapp.post("/v1/responses", json={"model": "fx1", "input": "x"})
    rr_doc = bapp.get(f"/harness/completions/{rr.headers.get('x-fx1-completion-id')}/receipt")
    out["receipt_sha_header_responses"] = (
        rr.status_code == 200
        and rr.headers.get("x-fx1-receipt-sha256") == rr_doc.json()["receipt_sha256"]
    )
    rkey = {"Idempotency-Key": "rsha-probe-1"}
    r1 = ic.post("/harness/complete", json=cbody, headers=rkey)
    r2 = ic.post("/harness/complete", json=cbody, headers=rkey)
    r1_doc = ic.get(f"/harness/completions/{r1.headers.get('x-fx1-completion-id')}/receipt")
    out["receipt_sha_header_complete_replay"] = (
        r1.status_code == 200
        and r1.headers.get("x-fx1-receipt-sha256") == r1_doc.json()["receipt_sha256"]
        and r2.json()["replayed"] is True
        and r2.headers.get("x-fx1-receipt-sha256") == r1.headers["x-fx1-receipt-sha256"]
    )
    rss = bapp.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
        },
    )
    out["receipt_sha_header_sse"] = (
        rss.status_code == 200 and rss.headers.get("x-fx1-receipt-sha256") is not None
    )
    # Isolated circuits per endpoint: the dead override opens its own
    # breaker key while the healthy override (and the env default) pass.
    bapp_brk = _TC2(
        api_mod.create_app(
            backend_resolver=_cap_resolver,
            breaker_threshold=2,
            breaker_cooldown_s=60.0,
        )
    )
    dead = {"base_url": "https://dead.example.com/v1", "api_key": "k", "model": "m"}
    dead_body = {"backend": "byok", "messages": [{"role": "user", "content": "x"}], "byok": dead}
    bapp_brk.post("/harness/complete", json=dead_body)
    bapp_brk.post("/harness/complete", json=dead_body)
    open_dead = bapp_brk.post("/harness/complete", json=dead_body)
    ok_live = bapp_brk.post(
        "/harness/complete",
        json={
            "backend": "byok",
            "messages": [{"role": "user", "content": "x"}],
            "byok": ovr,
        },
    )
    out["byok_override_breaker_isolated"] = (
        open_dead.status_code == 503 and ok_live.status_code == 200
    )
    # Idempotency still works with an override on the body — same body
    # replays, a different key inside byok gets the 409.
    idem_byok = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _CleanBackend()))
    k_body = {
        "backend": "byok",
        "messages": [{"role": "user", "content": "x"}],
        "byok": ovr,
    }
    i1 = idem_byok.post("/harness/complete", json=k_body, headers={"Idempotency-Key": "bk1"})
    i2 = idem_byok.post("/harness/complete", json=k_body, headers={"Idempotency-Key": "bk1"})
    i3 = idem_byok.post(
        "/harness/complete",
        json={**k_body, "byok": {**ovr, "api_key": "sk-different"}},
        headers={"Idempotency-Key": "bk1"},
    )
    out["byok_override_idem_contract"] = (
        i1.status_code == 200 and i2.json().get("replayed") is True and i3.status_code == 409
    )
    # limiter counts denials; a public-path request also draws a token
    limited2 = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        rate_limit_rps=20.0,
    )
    lc2 = _TC2(limited2)
    statuses = [lc2.get("/harness/version").status_code for _ in range(23)]
    _clk["t"] += 1.0
    m = lc2.get("/metrics")
    out["rate_limit_metrics_counts"] = (
        statuses.count(429) >= 2
        and m.status_code == 200
        and m.json()["rate_limited_total"] == statuses.count(429)
    )
    os.environ["FX1_API_RATE_LIMIT_RPS"] = "3"
    try:
        env_app = api_mod.create_app(
            harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
            backend_resolver=lambda *a, **k: _CleanBackend(),
        )
        ec = _TC2(env_app)
        out["rate_limit_env_config"] = [
            ec.get("/harness/version").status_code for _ in range(5)
        ].count(429) >= 1
    finally:
        os.environ.pop("FX1_API_RATE_LIMIT_RPS", None)

    # --- job event stream: SSE status feed until terminal -------------------
    ev_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        max_inflight=1,
        sse_keepalive_s=0.05,
    )
    evc = _TC2(ev_app)
    ev_jid = evc.post("/harness/jobs", json={"command": "doctor"}).json()["job_id"]
    ev_resp = evc.get(f"/harness/jobs/{ev_jid}/events")
    ev_frames = [
        _json.loads(ln[len("data: ") :])
        for ln in ev_resp.text.splitlines()
        if ln.startswith("data: ")
    ]
    out["job_events_stream_terminal"] = (
        ev_resp.status_code == 200
        and "text/event-stream" in ev_resp.headers.get("content-type", "")
        and ev_resp.headers.get("cache-control", "") in ("no-cache", "no-store")
        and len(ev_frames) >= 1
        and ev_frames[-1]["status"] == "succeeded"
        and ev_frames[-1]["job_id"] == ev_jid
    )
    out["job_events_unknown_404"] = evc.get("/harness/jobs/nope/events").status_code == 404
    out["job_events_timeout_422"] = (
        evc.get(f"/harness/jobs/{ev_jid}/events?timeout_s=0.5").status_code == 422
    )
    # a blocked worker keeps the next job queued: the stream emits the
    # queued frame, keeps alive on the quiet job, then closes at timeout_s
    ev_app.state.jobs_executor.submit(lambda: time.sleep(3))
    blk_jid = evc.post("/harness/jobs", json={"command": "doctor"}).json()["job_id"]
    blk_resp = evc.get(f"/harness/jobs/{blk_jid}/events?timeout_s=1")
    blk_statuses = [
        _json.loads(ln[len("data: ") :])["status"]
        for ln in blk_resp.text.splitlines()
        if ln.startswith("data: ")
    ]
    out["job_events_queued_frame_then_close"] = blk_resp.status_code == 200 and blk_statuses == [
        "queued"
    ]
    out["job_events_keepalive_frames"] = ": keepalive" in blk_resp.text

    # --- batch submit: per-item outcomes, per-item idempotency ---------------
    b_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        max_inflight=4,
    )
    bc = _TC2(b_app)
    b_resp = bc.post(
        "/harness/jobs/batch",
        json={
            "jobs": [
                {"command": "doctor", "idempotency_key": "bk1"},
                {"command": "doctor"},
                {"command": "no-such-command"},
            ]
        },
    )
    b_out = b_resp.json()
    out["jobs_batch_per_item"] = (
        b_resp.status_code == 202
        and b_out["submitted"] == 2
        and b_out["failed"] == 1
        and bool(b_out["jobs"][0].get("job_id"))
        and b_out["jobs"][2]["code"] == "not_found"
    )
    b_replay = bc.post(
        "/harness/jobs/batch",
        json={"jobs": [{"command": "doctor", "idempotency_key": "bk1"}]},
    ).json()
    out["jobs_batch_idem_replay"] = (
        b_replay["submitted"] == 1
        and b_replay["jobs"][0]["replayed"] is True
        and b_replay["jobs"][0]["job_id"] == b_out["jobs"][0]["job_id"]
    )
    b_conflict = bc.post(
        "/harness/jobs/batch",
        json={"jobs": [{"command": "operations", "idempotency_key": "bk1"}]},
    ).json()
    out["jobs_batch_idem_conflict"] = (
        b_conflict["failed"] == 1
        and b_conflict["jobs"][0]["code"] == "idempotency_conflict"
        and b_conflict["jobs"][0]["job_id"] is None
    )
    out["jobs_batch_empty_422"] = (
        bc.post("/harness/jobs/batch", json={"jobs": []}).status_code == 422
    )
    out["jobs_batch_over_cap_422"] = (
        bc.post("/harness/jobs/batch", json={"jobs": [{"command": "doctor"}] * 65}).status_code
        == 422
    )
    # single-submit idempotency via the body field (header still wins)
    body_key = bc.post(
        "/harness/jobs", json={"command": "doctor", "idempotency_key": "hdr-body"}
    ).json()
    body_key2 = bc.post(
        "/harness/jobs",
        json={"command": "doctor", "idempotency_key": "hdr-body"},
        headers={"Idempotency-Key": "hdr-body"},
    ).json()
    out["jobs_body_idem_replays"] = (
        body_key2["replayed"] is True and body_key2["job_id"] == body_key["job_id"]
    )
    cap_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        max_inflight=1,
    )
    cap_app.state.jobs_executor.submit(lambda: time.sleep(3))
    cc2 = _TC2(cap_app)
    cap_out = cc2.post(
        "/harness/jobs/batch",
        json={"jobs": [{"command": "doctor"}, {"command": "doctor"}]},
    ).json()
    out["jobs_batch_capacity_partial"] = (
        cap_out["submitted"] == 1
        and cap_out["failed"] == 1
        and cap_out["jobs"][1]["code"] == "over_capacity"
    )

    # --- batch receipt verification ---------------------------------------------
    vb = bc.post(
        "/receipts/verify/batch",
        json={"receipts": [{"not": "a receipt"}, {"x": 1}]},
    ).json()
    out["verify_batch_per_item"] = (
        vb["verified"] + vb["failed"] == 2
        and len(vb["results"]) == 2
        and vb["results"][0]["index"] == 0
        and vb["results"][0]["valid"] is False
    )
    out["verify_batch_empty_422"] = (
        bc.post("/receipts/verify/batch", json={"receipts": []}).status_code == 422
    )
    out["verify_batch_over_cap_422"] = (
        bc.post("/receipts/verify/batch", json={"receipts": [{}] * 65}).status_code == 422
    )

    # --- gzip response compression --------------------------------------------
    g_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran", "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
    )
    gc = _TC2(g_app)
    g_resp = gc.get("/openapi.json", headers={"Accept-Encoding": "gzip"})
    out["gzip_compresses_large_response"] = (
        g_resp.status_code == 200
        and g_resp.headers.get("content-encoding") == "gzip"
        and len(g_resp.content) > 4096
    )
    g_small = gc.get("/health", headers={"Accept-Encoding": "gzip"})
    out["gzip_small_response_untouched"] = (
        g_small.status_code == 200 and "content-encoding" not in g_small.headers
    )
    g_plain = gc.get("/openapi.json", headers={"Accept-Encoding": "identity"})
    out["gzip_requires_accept_encoding"] = (
        g_plain.status_code == 200
        and "content-encoding" not in g_plain.headers
        and len(g_plain.content) > 4096
    )
    g_off_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran", "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        gzip_min_bytes=0,
    )
    g_off = _TC2(g_off_app).get("/openapi.json", headers={"Accept-Encoding": "gzip"})
    out["gzip_zero_disables"] = "content-encoding" not in g_off.headers
    try:
        api_mod.create_app(
            harness=_Harness(runner=lambda argv, t: (0, "ran", "")),
            backend_resolver=lambda *a, **k: _CleanBackend(),
            gzip_min_bytes=-1,
        )
        out["gzip_negative_rejected"] = False
    except ValueError:
        out["gzip_negative_rejected"] = True

    # --- job lifecycle hygiene --------------------------------------------------
    big = "x" * (1 << 21)
    chatty_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, big, "e" * (1 << 21))),
        backend_resolver=lambda *a, **k: _CleanBackend(),
    )
    cc3 = _TC2(chatty_app)
    jid_big = cc3.post("/harness/jobs", json={"command": "doctor"}).json()["job_id"]
    deadline = time.time() + 10.0
    while time.time() < deadline:
        rec = cc3.get(f"/harness/jobs/{jid_big}").json()
        if rec["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.05)
    out["job_result_capped"] = (
        rec["status"] == "succeeded"
        and rec["result"]["stdout_truncated"] is True
        and rec["result"]["stderr_truncated"] is True
        and len(rec["result"]["stdout"].encode()) <= 1 << 20
        and len(rec["result"]["stderr"].encode()) <= 1 << 20
    )
    small_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "tiny", "warn")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
    )
    sc = _TC2(small_app)
    jid_small = sc.post("/harness/jobs", json={"command": "doctor"}).json()["job_id"]
    deadline = time.time() + 10.0
    while time.time() < deadline:
        rec2 = sc.get(f"/harness/jobs/{jid_small}").json()
        if rec2["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.05)
    out["job_result_small_unflagged"] = (
        rec2["status"] == "succeeded"
        and rec2["result"]["stdout_truncated"] is False
        and rec2["result"]["stdout"] == "tiny"
    )

    # shutdown: TestClient __exit__ runs the lifespan teardown — queued jobs
    # must flip to cancelled (never silently die) and the drain latch set.
    drain_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran", "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        max_inflight=1,
    )
    with _TC2(drain_app) as dc:
        drain_app.state.jobs_executor.submit(lambda: time.sleep(3))
        jid_q = dc.post("/harness/jobs", json={"command": "doctor"}).json()["job_id"]
    dead = drain_app.state.job_store.get(jid_q)
    out["shutdown_cancels_queued"] = (
        dead is not None and dead.status == "cancelled" and dead.finished_at is not None
    )
    out["shutdown_sets_drain"] = drain_app.state.metrics.draining.is_set()

    # --- receipt verification -------------------------------------------------
    from fx1.serve.byok_audit import byok_audit_bench

    good = byok_audit_bench()
    resp = client.post("/receipts/verify", json={"receipt": good})
    out["receipt_valid_payload"] = resp.status_code == 200 and resp.json()["valid"] is True
    tampered = dict(good)
    tampered["receipt_sha256"] = "0" * 64
    resp = client.post("/receipts/verify", json={"receipt": tampered})
    out["receipt_tamper_detected"] = resp.status_code == 200 and resp.json()["valid"] is False
    resp = client.post("/receipts/verify", json={"receipt": {"not": "a receipt"}})
    out["receipt_malformed_fails_closed"] = (
        resp.status_code == 200 and resp.json()["valid"] is False
    )
    out["receipt_non_object_422"] = (
        client.post("/receipts/verify", json={"receipt": [1, 2]}).status_code == 422
    )
    out["receipt_verify_never_5xx"] = client.post(
        "/receipts/verify",
        json={"receipt": {"nested": {"deep": [[{"x": None}]]}}},
    ).status_code in (200, 422)

    # --- content-addressed receipt fetch --------------------------------------
    # A caller cites sha256 in a completion; these routes serve the sealed
    # bytes back by that hash — ETag is the hash itself.
    import tempfile as _tf  # noqa: PLC0415
    from pathlib import Path as _Pt  # noqa: PLC0415

    with _tf.TemporaryDirectory() as td:
        rdir = _Pt(td)
        sha = good["receipt_sha256"]
        (rdir / "probe_receipt.json").write_text(_json.dumps(good))
        rclient = _TC2(api_mod.create_app(receipts_dir=rdir))
        lst = rclient.get("/receipts")
        out["receipts_index_lists"] = (
            lst.status_code == 200
            and lst.json()["count"] == 1
            and lst.json()["items"][0]["sha256"] == sha
            and lst.json()["items"][0]["name"] == "probe_receipt.json"
        )
        got = rclient.get(f"/receipts/{sha}")
        raw = (rdir / "probe_receipt.json").read_bytes()
        out["receipt_fetch_verbatim"] = (
            got.status_code == 200
            and got.content == raw
            and got.json()["receipt_sha256"] == sha
            and got.headers.get("etag") == f'"{sha}"'
            and "immutable" in (got.headers.get("cache-control") or "")
            and got.headers.get("x-fx1-receipt-valid") == "true"
        )
        miss = rclient.get("/receipts/" + "f" * 64)
        out["receipt_fetch_404"] = (
            miss.status_code == 404 and miss.json().get("code") == "receipt_not_found"
        )
        out["receipt_fetch_422_malformed"] = rclient.get("/receipts/zzz").status_code == 422
        nm = rclient.get(f"/receipts/{sha}", headers={"if-none-match": f'"{sha}"'})
        out["receipt_fetch_conditional_304"] = (
            nm.status_code == 304
            and nm.content == b""
            and nm.headers.get("etag") == f'"{sha}"'
            and rclient.get(
                f"/receipts/{sha}", headers={"if-none-match": '"' + "b" * 64 + '"'}
            ).status_code
            == 200
        )
        # staleness key: a receipt written after the first scan is indexed
        (rdir / "second.json").write_text(_json.dumps({**good, "receipt_sha256": "b" * 64}))
        out["receipts_index_refreshes"] = rclient.get("/receipts").json()["count"] == 2

        # --- cited-receipt verification --------------------------------------
        # With a store mounted, a completion may not footnote evidence the
        # server cannot produce: unknown or malformed hashes are 422 across
        # sync, batch, and stream — checked before the breaker/backend call.
        cite = _TC2(
            api_mod.create_app(receipts_dir=rdir, backend_resolver=lambda *a, **k: _CleanBackend())
        )
        out["cite_resolves_200"] = (
            cite.post(
                "/harness/complete",
                json={
                    "backend": "hosted_k3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "receipt_hashes": [sha],
                },
            ).status_code
            == 200
        )
        bad = cite.post(
            "/harness/complete",
            json={
                "backend": "hosted_k3",
                "messages": [{"role": "user", "content": "hi"}],
                "receipt_hashes": ["e" * 64],
            },
        )
        out["cite_unknown_422"] = (
            bad.status_code == 422 and bad.json().get("code") == "receipt_not_found"
        )
        out["cite_malformed_422"] = (
            cite.post(
                "/harness/complete",
                json={
                    "backend": "hosted_k3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "receipt_hashes": ["zzz"],
                },
            ).status_code
            == 422
        )
        out["cite_batch_checked"] = (
            cite.post(
                "/harness/complete/batch",
                json={
                    "backend": "hosted_k3",
                    "batch": [[{"role": "user", "content": "hi"}]],
                    "receipt_hashes": ["e" * 64],
                },
            ).status_code
            == 422
        )
        out["cite_stream_checked"] = (
            cite.post(
                "/harness/complete/stream",
                json={
                    "backend": "hosted_k3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "receipt_hashes": ["e" * 64],
                },
            ).status_code
            == 422
        )
        # X-Fx1-Receipt-Hashes — the header twin of fx1.receipt_hashes for
        # clients that can't edit the JSON body (same channel as
        # X-Fx1-Fallbacks): comma-separated sha256 digests go through the
        # same store check, the body extension wins, and a malformed
        # digest is a fail-closed 400 at the translation layer.
        out["xfx_receipt_hashes_header_cites"] = (
            cite.post(
                "/v1/chat/completions",
                headers={"X-Fx1-Receipt-Hashes": sha},
                json={"model": "fx1", "messages": [{"role": "user", "content": "x"}]},
            ).status_code
            == 200
        )
        out["xfx_receipt_hashes_unknown_422"] = (
            cite.post(
                "/v1/chat/completions",
                headers={"X-Fx1-Receipt-Hashes": "e" * 64},
                json={"model": "fx1", "messages": [{"role": "user", "content": "x"}]},
            ).status_code
            == 422
        )
        out["xfx_receipt_hashes_bad_400"] = (
            cite.post(
                "/v1/chat/completions",
                headers={"X-Fx1-Receipt-Hashes": "zzz"},
                json={"model": "fx1", "messages": [{"role": "user", "content": "x"}]},
            ).status_code
            == 400
        )
        out["xfx_receipt_hashes_ext_wins"] = (
            cite.post(
                "/v1/chat/completions",
                headers={"X-Fx1-Receipt-Hashes": "e" * 64},
                json={
                    "model": "fx1",
                    "messages": [{"role": "user", "content": "x"}],
                    "fx1": {"receipt_hashes": [sha]},
                },
            ).status_code
            == 200
        )
        out["xfx_receipt_hashes_responses"] = (
            cite.post(
                "/v1/responses",
                headers={"X-Fx1-Receipt-Hashes": "e" * 64},
                json={"model": "fx1", "input": "x"},
            ).status_code
            == 422
        )
    # No store mounted -> citations stay advisory (nothing to check against).
    nostore = _TC2(
        api_mod.create_app(
            receipts_dir="/nonexistent-receipts-dir-zzz",
            backend_resolver=lambda *a, **k: _CleanBackend(),
        )
    )
    out["cite_advisory_no_store"] = (
        nostore.post(
            "/harness/complete",
            json={
                "backend": "hosted_k3",
                "messages": [{"role": "user", "content": "hi"}],
                "receipt_hashes": ["e" * 64],
            },
        ).status_code
        == 200
    )
    gone = _TC2(api_mod.create_app(receipts_dir="/nonexistent-receipts-dir-zzz"))
    out["receipt_fetch_store_unavailable"] = (
        gone.get("/receipts/" + "a" * 64).status_code == 503
        and gone.get("/receipts").status_code == 503
        and gone.get("/receipts/" + "a" * 64).json().get("code") == "receipts_unavailable"
    )

    # --- auth + headers ---------------------------------------------------------
    health_resp = client.get("/health")
    out["health_public"] = health_resp.status_code == 200
    health = health_resp.json()
    out["health_no_secrets"] = all(isinstance(v, bool) for v in health["backends"].values()) and (
        os.environ.get("MOONSHOT_API_KEY", "\x00") not in health_resp.text
    )

    secured, _ = _client(api_key="k3y-material")
    out["auth_required_401"] = secured.get("/harness/commands").status_code == 401
    out["auth_wrong_key_401"] = (
        secured.get("/harness/commands", headers={"X-API-Key": "wrong"}).status_code == 401
    )
    out["auth_accepts_key"] = (
        secured.get("/harness/commands", headers={"X-API-Key": "k3y-material"}).status_code == 200
    )
    out["auth_health_still_public"] = secured.get("/health").status_code == 200

    # --- managed API keys --------------------------------------------------
    # env key = bootstrap admin; minted keys serve gated routes but cannot
    # manage keys; revocation is a tombstone; requests attribute key_id.
    saved_api_key = os.environ.get(_API_KEY_ENV)
    os.environ[_API_KEY_ENV] = "k3y-material"
    try:
        keys_client = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _CleanBackend()))
    finally:
        if saved_api_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = saved_api_key
    root_h = {"X-API-Key": "k3y-material"}
    mint = keys_client.post("/harness/keys", json={"name": "svc"}, headers=root_h)
    mint_body = mint.json() if mint.status_code == 201 else {}
    mkey = str(mint_body.get("key", ""))
    kid = str(mint_body.get("id", ""))
    out["key_mint_201_raw_once"] = mint.status_code == 201 and mkey.startswith("fx1k_")
    out["key_mint_needs_admin"] = keys_client.post("/harness/keys", json={}).status_code == 401
    out["key_authenticates"] = (
        keys_client.get("/harness/commands", headers={"X-API-Key": mkey}).status_code == 200
    )
    denied = keys_client.get("/harness/keys", headers={"X-API-Key": mkey})
    # a default key's scopes are [read, write] — the control plane
    # needs admin, so the scope layer refuses before the route's own
    # admin check ever sees the request
    out["key_not_admin_403"] = (
        denied.status_code == 403 and denied.json().get("code") == "insufficient_scope"
    )
    listed = keys_client.get("/harness/keys", headers=root_h)
    listed_rows = listed.json()["data"] if listed.status_code == 200 else []
    out["key_list_prefix_no_secret"] = (
        listed.status_code == 200
        and len(listed_rows) == 1
        and listed_rows[0]["prefix"] == mkey[:13]
        and mkey not in listed.text
        and "sha256" not in listed.text
    )
    got = keys_client.get(f"/harness/keys/{kid}", headers=root_h)
    out["key_get_by_id"] = got.status_code == 200 and got.json()["id"] == kid
    out["key_get_unknown_404"] = (
        keys_client.get("/harness/keys/" + "0" * 16, headers=root_h).status_code == 404
    )
    # an admin managed key minted by the bootstrap credential keeps the
    # control plane — it may mint/list/revoke keys itself; a plain
    # managed key cannot.
    admin_mint = keys_client.post(
        "/harness/keys", json={"name": "ops", "admin": True}, headers=root_h
    )
    admin_raw = str(admin_mint.json().get("key", ""))
    out["key_admin_mint_flagged"] = (
        admin_mint.status_code == 201 and admin_mint.json()["admin"] is True
    )
    out["key_admin_manages_keys"] = (
        keys_client.post("/harness/keys", json={}, headers={"X-API-Key": admin_raw}).status_code
        == 201
        and keys_client.get("/harness/keys", headers={"X-API-Key": admin_raw}).status_code == 200
    )
    out["key_managed_cannot_mint"] = (
        keys_client.post("/harness/keys", json={}, headers={"X-API-Key": mkey}).status_code == 403
    )
    # attribution: a managed-key call lands under the key's fingerprint;
    # an env-key call lands under "env".
    keys_client.post(
        "/harness/complete",
        json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
        headers={"X-API-Key": mkey},
    )
    keys_client.post(
        "/harness/complete",
        json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
        headers=root_h,
    )
    usage_kid = keys_client.get(f"/harness/usage?key_id={kid}", headers=root_h).json()
    out["key_usage_filtered"] = usage_kid.get("records_seen") == 1
    usage_all = keys_client.get("/harness/usage", headers=root_h).json()
    out["key_usage_by_key_buckets"] = (
        usage_all.get("by_key", {}).get(kid, {}).get("requests") == 1
        and usage_all.get("by_key", {}).get("env", {}).get("requests") == 1
    )
    revoke = keys_client.delete(f"/harness/keys/{kid}", headers=root_h)
    out["key_revoke_tombstone"] = revoke.status_code == 200 and revoke.json()["enabled"] is False
    out["key_revoked_auth_401"] = (
        keys_client.get("/harness/commands", headers={"X-API-Key": mkey}).status_code == 401
    )
    out["key_revoke_again_409"] = (
        keys_client.delete(f"/harness/keys/{kid}", headers=root_h).status_code == 409
    )
    out["key_revoke_unknown_404"] = (
        keys_client.delete("/harness/keys/" + "0" * 16, headers=root_h).status_code == 404
    )
    # declared per-key policy: rpm bounds the key to a fixed 60 s
    # request window — the over-limit refusal is 429 + Retry-After and
    # never counts as a use; ttl_s bakes an expires_at into the record.
    rpm_mint = keys_client.post("/harness/keys", json={"rpm": 1}, headers=root_h)
    rpm_raw = str(rpm_mint.json().get("key", ""))
    rpm_id = str(rpm_mint.json().get("id", ""))
    first = keys_client.get("/harness/commands", headers={"X-API-Key": rpm_raw})
    limited = keys_client.get("/harness/commands", headers={"X-API-Key": rpm_raw})
    out["key_rpm_429"] = (
        rpm_mint.status_code == 201
        and rpm_mint.json()["rpm"] == 1
        and first.status_code == 200
        and limited.status_code == 429
        and limited.json().get("code") == "rate_limited"
        and int(limited.headers.get("Retry-After", "0")) >= 1
    )
    # a rpm-declared key answers its standing budget on every response
    # (OpenAI's header names); the 429 still carries the declared window
    out["key_rpm_headers"] = (
        first.headers.get("x-ratelimit-limit-requests") == "1"
        and first.headers.get("x-ratelimit-remaining-requests") == "0"
        and int(first.headers.get("x-ratelimit-reset-requests", "-1")) >= 0
        and limited.headers.get("x-ratelimit-limit-requests") == "1"
        and limited.headers.get("x-ratelimit-remaining-requests") == "0"
        and int(limited.headers.get("x-ratelimit-reset-requests", "-1")) >= 1
    )
    # keys without a declared window, the env credential, and loopback
    # auth emit no budget headers — no false scarcity
    plain_mint = keys_client.post("/harness/keys", json={}, headers=root_h)
    plain_raw = str(plain_mint.json().get("key", ""))
    plain_hit = keys_client.get("/harness/commands", headers={"X-API-Key": plain_raw})
    env_hit = keys_client.get("/harness/commands", headers=root_h)
    out["key_rpm_headers_absent"] = (
        plain_mint.status_code == 201
        and "x-ratelimit-limit-requests" not in plain_hit.headers
        and "x-ratelimit-limit-requests" not in env_hit.headers
    )
    rpm_rec = keys_client.get(f"/harness/keys/{rpm_id}", headers=root_h)
    out["key_rpm_refusal_no_burn"] = rpm_rec.status_code == 200 and rpm_rec.json()["uses"] == 1
    out["key_policy_bad_422"] = (
        keys_client.post("/harness/keys", json={"rpm": 0}, headers=root_h).status_code == 422
        and keys_client.post("/harness/keys", json={"ttl_s": -1}, headers=root_h).status_code == 422
    )
    # an expired key fails closed — same 401 shape as revoked
    ttl_mint = keys_client.post("/harness/keys", json={"ttl_s": 0.05}, headers=root_h)
    ttl_raw = str(ttl_mint.json().get("key", ""))
    ttl_ok = keys_client.get("/harness/commands", headers={"X-API-Key": ttl_raw})
    time.sleep(0.06)
    ttl_dead = keys_client.get("/harness/commands", headers={"X-API-Key": ttl_raw})
    out["key_ttl_expires_401"] = (
        ttl_mint.status_code == 201
        and ttl_mint.json()["expires_at"] is not None
        and ttl_ok.status_code == 200
        and ttl_dead.status_code == 401
    )
    # --- key scopes: least-privilege read/write/admin on minted keys ----
    # a read-only key serves safe methods and is refused on mutations and
    # the control plane; the 403 carries ``insufficient_scope`` in the
    # path's own error grammar.
    ro_mint = keys_client.post("/harness/keys", json={"scopes": ["read"]}, headers=root_h)
    ro_raw = str(ro_mint.json().get("key", ""))
    out["key_scope_mint_201"] = (
        ro_mint.status_code == 201
        and ro_mint.json()["scopes"] == ["read"]
        and ro_mint.json()["admin"] is False
    )
    out["key_scope_read_allows"] = (
        keys_client.get("/harness/commands", headers={"X-API-Key": ro_raw}).status_code == 200
    )
    ro_write = keys_client.post(
        "/harness/complete",
        json={"messages": [{"role": "user", "content": "x"}]},
        headers={"X-API-Key": ro_raw},
    )
    ro_admin = keys_client.get("/harness/keys", headers={"X-API-Key": ro_raw})
    out["key_scope_read_denies_write"] = (
        ro_write.status_code == 403 and ro_write.json()["code"] == "insufficient_scope"
    )
    out["key_scope_read_denies_admin"] = (
        ro_admin.status_code == 403 and ro_admin.json()["code"] == "insufficient_scope"
    )
    # ...in OpenAI/Anthropic wire grammar on the /v1 surface
    ro_v1 = keys_client.post(
        "/v1/chat/completions",
        json={"model": "fx1", "messages": [{"role": "user", "content": "x"}]},
        headers={"X-API-Key": ro_raw},
    )
    ro_am = keys_client.post(
        "/v1/messages",
        json={"model": "fx1", "messages": [{"role": "user", "content": "x"}], "max_tokens": 8},
        headers={"X-API-Key": ro_raw},
    )
    out["key_scope_v1_shape"] = (
        ro_v1.status_code == 403
        and ro_v1.json()["error"]["code"] == "insufficient_scope"
        and ro_am.status_code == 403
        and ro_am.json()["error"]["type"] == "permission_error"
    )
    # write-only is the mirror: mutations reach the gated pipeline (the
    # clean backend 200s), reads refuse
    wr_mint = keys_client.post("/harness/keys", json={"scopes": ["write"]}, headers=root_h)
    wr_raw = str(wr_mint.json().get("key", ""))
    wr_post = keys_client.post(
        "/harness/gate/check", json={"text": "hi"}, headers={"X-API-Key": wr_raw}
    )
    out["key_scope_write_allows_write"] = wr_post.status_code == 200
    out["key_scope_write_denies_read"] = (
        keys_client.get("/harness/commands", headers={"X-API-Key": wr_raw}).status_code == 403
    )
    # an admin-scoped key (no flag) manages the control plane but cannot
    # touch data-plane calls — scope is authoritative, not the flag
    adm_mint = keys_client.post("/harness/keys", json={"scopes": ["admin"]}, headers=root_h)
    adm_raw = str(adm_mint.json().get("key", ""))
    out["key_scope_admin_scoped"] = (
        adm_mint.status_code == 201
        and adm_mint.json()["admin"] is True
        and keys_client.get("/harness/keys", headers={"X-API-Key": adm_raw}).status_code == 200
        and keys_client.post("/harness/keys", json={}, headers={"X-API-Key": adm_raw}).status_code
        == 201
        and keys_client.get("/harness/commands", headers={"X-API-Key": adm_raw}).status_code == 403
    )
    # admin=True unions its scope onto an explicit list — the flag is
    # additive, never silently dropped by a narrower declaration
    un_mint = keys_client.post(
        "/harness/keys", json={"admin": True, "scopes": ["read"]}, headers=root_h
    )
    un_raw = str(un_mint.json().get("key", ""))
    out["key_scope_admin_union"] = (
        un_mint.status_code == 201
        and un_mint.json()["scopes"] == ["read", "admin"]
        and keys_client.get("/harness/keys", headers={"X-API-Key": un_raw}).status_code == 200
        and keys_client.post(
            "/harness/gate/check", json={"text": "hi"}, headers={"X-API-Key": un_raw}
        ).status_code
        == 403
    )
    out["key_scope_bad_400"] = (
        keys_client.post("/harness/keys", json={"scopes": ["bogus"]}, headers=root_h).status_code
        == 400
        and keys_client.post("/harness/keys", json={"scopes": []}, headers=root_h).status_code
        == 400
    )
    # a scoped refusal on drain is admin — the control plane is one scope
    out["key_scope_drain_admin"] = (
        keys_client.post("/harness/drain", headers={"X-API-Key": ro_raw}).status_code == 403
    )
    # --- per-key quota budgets ------------------------------------------
    # max_requests counts authenticated calls; max_tokens counts
    # provider-reported usage charged after each served response. An
    # exhausted key refuses 429 quota_exceeded with NO Retry-After — a
    # hard budget does not clear inside a window.
    rq_mint = keys_client.post(
        "/harness/keys", json={"name": "q-req", "max_requests": 2}, headers=root_h
    )
    rq_raw = str(rq_mint.json().get("key", ""))
    rq_h = {"X-API-Key": rq_raw}
    out["key_quota_mint_201"] = (
        rq_mint.status_code == 201
        and rq_mint.json()["max_requests"] == 2
        and rq_mint.json()["max_tokens"] is None
    )
    out["key_quota_request_429"] = (
        keys_client.get("/harness/commands", headers=rq_h).status_code == 200
        and keys_client.get("/harness/commands", headers=rq_h).status_code == 200
        and keys_client.get("/harness/commands", headers=rq_h).status_code == 429
    )
    rq_over = keys_client.get("/harness/commands", headers=rq_h)
    out["key_quota_refusal_shape"] = (
        rq_over.status_code == 429
        and rq_over.json().get("code") == "quota_exceeded"
        and "retry-after" not in {k.lower() for k in rq_over.headers}
    )
    # the refusal fires on the /v1 surface in OpenAI error grammar too
    rq_v1 = keys_client.post(
        "/v1/chat/completions",
        json={"model": "fx1", "messages": [{"role": "user", "content": "x"}]},
        headers=rq_h,
    )
    out["key_quota_v1_shape"] = (
        rq_v1.status_code == 429 and rq_v1.json()["error"]["code"] == "quota_exceeded"
    )
    rq_rec = keys_client.get(f"/harness/keys/{rq_mint.json()['id']}", headers=root_h)
    out["key_quota_record_fields"] = (
        rq_rec.status_code == 200
        and rq_rec.json()["max_requests"] == 2
        and rq_rec.json()["uses"] == 2
        and rq_rec.json()["tokens_used"] == 0
    )
    out["key_quota_bad_422"] = (
        keys_client.post("/harness/keys", json={"max_requests": 0}, headers=root_h).status_code
        == 422
        and keys_client.post("/harness/keys", json={"max_tokens": 0}, headers=root_h).status_code
        == 422
    )

    # token budgets meter provider-reported usage — needs a backend that
    # reports it; spin a second app under the same env-key pattern
    class _MeterBackend:
        def __init__(self) -> None:
            self.last_usage: dict[str, int] | None = None
            self._model = "meter-0"

        def complete(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> str:
            self.last_usage = {
                "prompt_tokens": 4,
                "completion_tokens": 6,
                "total_tokens": 10,
            }
            return f"metered:{messages[-1]['content']}"

    saved_api_key2 = os.environ.get(_API_KEY_ENV)
    os.environ[_API_KEY_ENV] = "k3y-material"
    try:
        tok_app = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _MeterBackend()))
    finally:
        if saved_api_key2 is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = saved_api_key2
    tok_mint = tok_app.post(
        "/harness/keys", json={"name": "q-tok", "max_tokens": 10}, headers=root_h
    )
    tok_raw = str(tok_mint.json().get("key", ""))
    tok_h = {"X-API-Key": tok_raw}
    t_first = tok_app.post(
        "/harness/complete",
        json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
        headers=tok_h,
    )
    t_second = tok_app.post(
        "/harness/complete",
        json={"backend": "byok", "messages": [{"role": "u", "content": "y"}]},
        headers=tok_h,
    )
    out["key_quota_token_429"] = t_first.status_code == 200 and t_second.status_code == 429
    tok_rec = tok_app.get(f"/harness/keys/{tok_mint.json()['id']}", headers=root_h)
    out["key_quota_token_metered"] = (
        tok_rec.status_code == 200 and tok_rec.json()["tokens_used"] == 10
    )
    # the env key is unmetered — budgets bind managed keys only
    out["key_quota_env_unmetered"] = (
        tok_app.get("/harness/commands", headers=root_h).status_code == 200
        and keys_client.get("/harness/commands", headers=root_h).status_code == 200
    )
    # --- per-key usage card + caller self-introspection -----------------
    # GET /harness/keys/{id}/usage is the admin view; GET /harness/self
    # is the read-scope twin any credential calls on itself.
    uq_mint = keys_client.post(
        "/harness/keys",
        json={"name": "u-card", "max_requests": 5, "max_tokens": 100, "rpm": 60},
        headers=root_h,
    )
    uq_id = uq_mint.json()["id"]
    uq_raw = str(uq_mint.json().get("key", ""))
    uq_h = {"X-API-Key": uq_raw}
    keys_client.get("/harness/commands", headers=uq_h)
    keys_client.get("/harness/commands", headers=uq_h)
    uq_card = keys_client.get(f"/harness/keys/{uq_id}/usage", headers=root_h)
    uq = uq_card.json() if uq_card.status_code == 200 else {}
    out["key_usage_200"] = (
        uq_card.status_code == 200
        and uq.get("object") == "key_usage"
        and uq.get("uses") == 2
        and uq.get("max_requests") == 5
        and uq.get("requests_remaining") == 3
        and uq.get("max_tokens") == 100
        and uq.get("tokens_used") == 0
        and uq.get("tokens_remaining") == 100
        and uq.get("rpm") == 60
        and uq.get("window_remaining") == 58
        and uq.get("enabled") is True
        and uq.get("served", {}).get("calls") == 0
        and uq.get("log_cap", 0) > 0
        and "log_dropped" in uq
    )
    out["key_usage_404"] = (
        keys_client.get("/harness/keys/0000000000000000/usage", headers=root_h).status_code == 404
    )
    out["key_usage_admin_scope"] = (
        keys_client.get(f"/harness/keys/{uq_id}/usage", headers={"X-API-Key": ro_raw}).status_code
        == 403
    )
    # served split attributes metered calls to the credential
    tq_mint = tok_app.post("/harness/keys", json={"name": "u-svc"}, headers=root_h)
    tq_raw = str(tq_mint.json().get("key", ""))
    tq_id = tq_mint.json()["id"]
    tok_app.post(
        "/harness/complete",
        json={"backend": "byok", "messages": [{"role": "u", "content": "z"}]},
        headers={"X-API-Key": tq_raw},
    )
    tq_card = tok_app.get(f"/harness/keys/{tq_id}/usage", headers=root_h)
    tq = tq_card.json() if tq_card.status_code == 200 else {}
    out["key_usage_served_split"] = (
        tq_card.status_code == 200
        and tq.get("uses") == 1
        and tq.get("tokens_used") == 10
        and tq.get("served", {}).get("calls") == 1
        and tq.get("served", {}).get("prompt_tokens") == 4
        and tq.get("served", {}).get("total_tokens") == 10
        and tq.get("served", {}).get("by_backend", {}).get("byok", {}).get("calls") == 1
    )
    # /harness/self — the caller's own card; the introspection call
    # itself is an authenticated use, so uses counts it
    sl = tok_app.get("/harness/self", headers={"X-API-Key": tq_raw})
    slj = sl.json() if sl.status_code == 200 else {}
    out["self_managed_200"] = (
        sl.status_code == 200
        and slj.get("object") == "self_usage"
        and slj.get("credential") == "managed"
        and slj.get("metered") is True
        and slj.get("scopes") == ["read", "write"]
        and slj.get("key", {}).get("id") == tq_id
        and slj.get("key", {}).get("uses") == 2
        and slj.get("key", {}).get("tokens_used") == 10
    )
    out["self_read_scope"] = (
        keys_client.get("/harness/self", headers={"X-API-Key": ro_raw}).status_code == 200
    )
    out["self_write_denied"] = (
        keys_client.get("/harness/self", headers={"X-API-Key": wr_raw}).status_code == 403
    )
    se = tok_app.get("/harness/self", headers=root_h)
    sej = se.json() if se.status_code == 200 else {}
    out["self_env_unmetered"] = (
        se.status_code == 200
        and sej.get("credential") == "env"
        and sej.get("metered") is False
        and sej.get("key") is None
        and sej.get("scopes") == ["read", "write", "admin"]
    )
    noenv_self = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _CleanBackend()))
    sn = noenv_self.get("/harness/self")
    out["self_loopback_none"] = (
        sn.status_code == 200
        and sn.json().get("credential") == "none"
        and sn.json().get("metered") is False
    )
    # no env key + empty store → loopback dev (admin); minting turns auth on
    noenv_client = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _CleanBackend()))
    minted = noenv_client.post("/harness/keys", json={})
    m_raw = minted.json()["key"] if minted.status_code == 201 else ""
    out["key_bootstrap_loopback_201"] = minted.status_code == 201
    out["key_store_enables_auth"] = noenv_client.get("/harness/commands").status_code == 401
    out["key_bootstrapped_works"] = (
        noenv_client.get("/harness/commands", headers={"X-API-Key": m_raw}).status_code == 200
    )
    # --- key rotation ---------------------------------------------------
    # POST /harness/keys/{id}/rotate mints a successor under the
    # predecessor's declared policy and tombstones the old secret in the
    # same transaction by default.
    rt_mint = keys_client.post(
        "/harness/keys",
        json={
            "name": "rot-src",
            "rpm": 45,
            "scopes": ["read"],
            "max_requests": 50,
            "max_tokens": 500,
        },
        headers=root_h,
    )
    rt_old_id = rt_mint.json()["id"]
    rt_old_raw = str(rt_mint.json().get("key", ""))
    rot = keys_client.post(f"/harness/keys/{rt_old_id}/rotate", json={}, headers=root_h)
    rotj = rot.json() if rot.status_code == 201 else {}
    rtk = rotj.get("key", {})
    out["key_rotate_201"] = (
        rot.status_code == 201
        and rotj.get("object") == "key_rotation"
        and rotj.get("rotated_from") == rt_old_id
        and rotj.get("revoked_previous") is True
        and str(rtk.get("key", "")).startswith("fx1k_")
    )
    out["key_rotate_inherits_policy"] = (
        rtk.get("name") == "rot-src"
        and rtk.get("rpm") == 45
        and rtk.get("scopes") == ["read"]
        and rtk.get("admin") is False
        and rtk.get("max_requests") == 50
        and rtk.get("max_tokens") == 500
        and rtk.get("rotated_from") == rt_old_id
        and rtk.get("id") != rt_old_id
    )
    # the swap is atomic: the old secret fails closed immediately
    out["key_rotate_old_fails_closed"] = (
        keys_client.get("/harness/self", headers={"X-API-Key": rt_old_raw}).status_code == 401
        and keys_client.get(
            "/harness/self", headers={"X-API-Key": str(rtk.get("key", ""))}
        ).status_code
        == 200
    )
    # the record carries lineage on key_get
    rot_get = keys_client.get(f"/harness/keys/{rtk.get('id', '')}", headers=root_h)
    out["key_rotate_lineage_on_record"] = (
        rot_get.status_code == 200 and rot_get.json().get("rotated_from") == rt_old_id
    )
    # keep-old: both secrets authenticate until the old key is revoked
    rt2_mint = keys_client.post("/harness/keys", json={"name": "rot-keep"}, headers=root_h)
    rt2_id = rt2_mint.json()["id"]
    rt2_raw = str(rt2_mint.json().get("key", ""))
    keep = keys_client.post(
        f"/harness/keys/{rt2_id}/rotate", json={"revoke_old": False}, headers=root_h
    )
    keep_j = keep.json() if keep.status_code == 201 else {}
    keep_rec = keys_client.get(f"/harness/keys/{rt2_id}", headers=root_h)
    out["key_rotate_keep_old"] = (
        keep.status_code == 201
        and keep_j.get("revoked_previous") is False
        and keys_client.get("/harness/self", headers={"X-API-Key": rt2_raw}).status_code == 200
        and keep_rec.status_code == 200
        and keep_rec.json().get("enabled") is True
    )
    # expiry: omitted ttl_s inherits the predecessor's absolute deadline;
    # a declared ttl_s mints the successor a fresh lifetime
    rt3_mint = keys_client.post(
        "/harness/keys", json={"name": "rot-ttl", "ttl_s": 300}, headers=root_h
    )
    rt3_id = rt3_mint.json()["id"]
    rt3_exp = rt3_mint.json()["expires_at"]
    inher = keys_client.post(f"/harness/keys/{rt3_id}/rotate", json={}, headers=root_h)
    inher_k = inher.json().get("key", {})
    fresh = keys_client.post(
        f"/harness/keys/{inher_k.get('id')}/rotate", json={"ttl_s": 7200}, headers=root_h
    )
    out["key_rotate_expiry"] = (
        inher.status_code == 201
        and abs(float(inher_k.get("expires_at") or 0.0) - float(rt3_exp)) < 1e-6
        and fresh.status_code == 201
        and float(fresh.json()["key"]["expires_at"]) > float(rt3_exp) + 3000
    )
    # scope binding: rotate is admin-plane — a read-scope key is refused
    out["key_rotate_admin_scope"] = (
        keys_client.post(
            f"/harness/keys/{rt2_id}/rotate", json={}, headers={"X-API-Key": ro_raw}
        ).status_code
        == 403
    )
    out["key_rotate_404"] = (
        keys_client.post(
            "/harness/keys/0000000000000000/rotate", json={}, headers=root_h
        ).status_code
        == 404
    )
    # a revoked credential cannot mint a live successor
    rev_rot = keys_client.post(f"/harness/keys/{rt_old_id}/rotate", json={}, headers=root_h)
    out["key_rotate_revoked_409"] = (
        rev_rot.status_code == 409 and rev_rot.json().get("code") == "key_revoked"
    )
    # --- mutable key policy: PATCH /harness/keys/{id} ------------------
    # three states: omitted keeps the declared policy, a concrete value
    # replaces it, explicit JSON null clears a nullable bound.
    pt_mint = keys_client.post(
        "/harness/keys",
        json={"name": "patch-me", "rpm": 30, "scopes": ["read"], "max_requests": 20},
        headers=root_h,
    )
    pt_id = pt_mint.json()["id"]
    pt_raw = str(pt_mint.json().get("key", ""))
    pt_exp = time.time() + 3600.0
    pt = keys_client.patch(
        f"/harness/keys/{pt_id}",
        json={
            "name": "patched",
            "rpm": 7,
            "max_requests": 9,
            "max_tokens": 77,
            "expires_at": pt_exp,
        },
        headers=root_h,
    )
    ptj = pt.json() if pt.status_code == 200 else {}
    out["key_patch_200"] = (
        pt.status_code == 200
        and ptj.get("object") == "key"
        and ptj.get("id") == pt_id
        and ptj.get("name") == "patched"
        and ptj.get("rpm") == 7
        and ptj.get("max_requests") == 9
        and ptj.get("max_tokens") == 77
        and abs(float(ptj.get("expires_at") or 0) - pt_exp) < 1.0
        and ptj.get("scopes") == ["read"]
        and ptj.get("enabled") is True
    )
    # the patch is in place and journaled: key_get reads the same record
    pt_get = keys_client.get(f"/harness/keys/{pt_id}", headers=root_h)
    out["key_patch_persists"] = (
        pt_get.status_code == 200
        and pt_get.json().get("name") == "patched"
        and pt_get.json().get("rpm") == 7
        and pt_get.json().get("max_requests") == 9
        and pt_get.json().get("uses") == ptj.get("uses") == 0
    )
    # explicit null unbounds: cleared rpm emits no rate-limit headers on
    # the key's next authenticated call — no false scarcity
    pc = keys_client.patch(f"/harness/keys/{pt_id}", json={"rpm": None}, headers=root_h)
    cleared = keys_client.get("/harness/commands", headers={"X-API-Key": pt_raw})
    out["key_patch_clear_unbounds"] = (
        pc.status_code == 200
        and pc.json().get("rpm") is None
        and cleared.status_code == 200
        and "x-ratelimit-limit-requests" not in cleared.headers
    )
    # admin is purely additive like mint: ``admin:true`` unions the
    # scope onto the surviving list; an explicit scopes list is literal
    # — ``admin:false`` never strips a declared scope
    adm = keys_client.patch(f"/harness/keys/{pt_id}", json={"admin": True}, headers=root_h)
    lit = keys_client.patch(
        f"/harness/keys/{pt_id}", json={"scopes": ["write"], "admin": False}, headers=root_h
    )
    out["key_patch_admin_union"] = (
        adm.status_code == 200
        and adm.json().get("scopes") == ["read", "admin"]
        and adm.json().get("admin") is True
        and lit.status_code == 200
        and lit.json().get("scopes") == ["write"]
        and lit.json().get("admin") is False
    )
    # fail closed: unknown id, a non-admin scope, and an unpatchable
    # field (enabled — revocation is permanent) all refuse
    out["key_patch_404"] = (
        keys_client.patch(
            "/harness/keys/0000000000000000", json={"name": "x"}, headers=root_h
        ).status_code
        == 404
    )
    out["key_patch_admin_scope"] = (
        keys_client.patch(
            f"/harness/keys/{pt_id}", json={"name": "x"}, headers={"X-API-Key": ro_raw}
        ).status_code
        == 403
    )
    out["key_patch_enabled_422"] = (
        keys_client.patch(
            f"/harness/keys/{pt_id}", json={"enabled": False}, headers=root_h
        ).status_code
        == 422
    )
    # a tombstoned credential stays dead — patch cannot resurrect it
    dead_mint = keys_client.post("/harness/keys", json={"name": "dead"}, headers=root_h)
    dead_id = dead_mint.json()["id"]
    keys_client.delete(f"/harness/keys/{dead_id}", headers=root_h)
    dead_patch = keys_client.patch(f"/harness/keys/{dead_id}", json={"name": "x"}, headers=root_h)
    out["key_patch_revoked_409"] = (
        dead_patch.status_code == 409 and dead_patch.json().get("code") == "key_revoked"
    )
    # the FIRST mint on a no-env deployment carries admin so the operator
    # keeps a control plane after provisioning turns auth on
    noenv2 = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _CleanBackend()))
    admin_minted = noenv2.post("/harness/keys", json={"admin": True})
    out["key_admin_loopback_manages"] = (
        admin_minted.status_code == 201
        and admin_minted.json()["admin"] is True
        and noenv2.get(
            "/harness/keys", headers={"X-API-Key": admin_minted.json()["key"]}
        ).status_code
        == 200
    )
    out["key_remote_unauthed_401"] = (
        _TC2(noenv_client.app, client=("198.51.100.9", 7)).get("/harness/commands").status_code
        == 401
    )

    out["loopback_served"] = client.get("/harness/commands").status_code == 200
    from fastapi.testclient import TestClient as _TC

    # A non-loopback client with no key configured is refused outright.
    remote_client = _TC(client.app, client=("203.0.113.7", 9))
    out["remote_refused_403"] = remote_client.get("/harness/commands").status_code == 403

    out["security_headers"] = all(
        h in client.get("/health").headers and h in client.get("/harness/commands").headers
        for h in ("x-content-type-options", "cache-control", "referrer-policy")
    )
    big_resp = client.post(
        "/receipts/verify",
        content=b" " * ((1 << 20) + 1),
        headers={"content-type": "application/json"},
    )
    out["body_cap_413"] = big_resp.status_code == 413

    # request tracing: X-Request-ID echoes when well-formed, mints otherwise
    echoed = client.get("/health", headers={"X-Request-ID": "trace-abc.123"})
    minted = client.get("/health")
    forged = client.get("/health", headers={"X-Request-ID": "bad\nid\x00inj"})
    out["request_id_echoed"] = echoed.headers.get("x-request-id") == "trace-abc.123"
    out["request_id_minted_when_absent"] = (
        minted.headers.get("x-request-id") is not None and len(minted.headers["x-request-id"]) == 32
    )
    out["request_id_malformed_replaced"] = (
        forged.headers.get("x-request-id") is not None
        and "\n" not in forged.headers["x-request-id"]
    )
    out["request_id_on_error_too"] = (
        secured.get("/harness/commands").headers.get("x-request-id") is not None
    )
    out["request_id_distinct"] = (
        client.get("/health").headers["x-request-id"]
        != client.get("/health").headers["x-request-id"]
    )

    # the response tail is single: body-cap / bad-length errors carry the
    # same security headers + request id as every other path
    bad_len = client.post(
        "/receipts/verify",
        content=b"{}",
        headers={"content-length": "abc"},
    )
    out["bad_content_length_400"] = bad_len.status_code == 400
    for label, err in (("cap_413", big_resp), ("bad_len", bad_len)):
        hdrs = {k.lower(): v for k, v in err.headers.items()}
        out[f"error_tail_{label}"] = all(
            hdrs.get(k) is not None
            for k in (
                "x-request-id",
                "x-content-type-options",
                "cache-control",
                "referrer-policy",
            )
        )

    # --- error envelope: stable machine codes + API version negotiation ----
    v = client.get("/harness/version")
    out["version_route_200"] = (
        v.status_code == 200
        and v.json()["api_version"] == api_mod.API_VERSION
        and bool(v.json()["fx1_version"])
    )
    from fx1.serve.client import EXPECTED_API_VERSION  # noqa: PLC0415
    from fx1.serve.contract import API_VERSION as _WIRE_VERSION  # noqa: PLC0415

    out["contract_single_source"] = (
        api_mod.API_VERSION == _WIRE_VERSION == EXPECTED_API_VERSION == "1"
    )
    from fastapi.testclient import TestClient as _CapTC  # noqa: PLC0415

    from fx1.harness import HarnessRole as _HarnessRole  # noqa: PLC0415

    # /harness/capabilities — self-describing feature/limit discovery.
    cap = client.get("/harness/capabilities")
    out["capabilities_route_200"] = cap.status_code == 200
    if cap.status_code == 200:
        capj = cap.json()
        out["capabilities_shape"] = all(
            k in capj for k in ("features", "limits", "backends", "roles", "api_version")
        )
        out["capabilities_features"] = all(
            capj["features"].get(f) is True
            for f in (
                "idempotency",
                "sse",
                "webhooks",
                "batch",
                "jobs",
                "drain",
                "eval_diff",
                "fine_tuning",
            )
        )
        out["capabilities_roles_cover_registry"] = set(capj["roles"]) == {
            str(r) for r in _HarnessRole
        }
    else:
        out["capabilities_shape"] = False
        out["capabilities_features"] = False
        out["capabilities_roles_cover_registry"] = False
    cap_app = api_mod.create_app(max_inflight=7, job_max=33, idem_max=17, rate_limit_rps=50.0)
    cap2 = _CapTC(cap_app).get("/harness/capabilities").json()
    out["capabilities_limits_reflect_config"] = (
        cap2["limits"]["max_inflight"] == 7.0
        and cap2["limits"]["job_max"] == 33.0
        and cap2["limits"]["idem_max"] == 17.0
        and cap2["limits"]["rate_limit_rps"] == 50.0
    )
    cap3 = _CapTC(api_mod.create_app(rate_limit_rps=0.0)).get("/harness/capabilities").json()
    out["capabilities_limiter_disabled_reports_zero"] = cap3["limits"]["rate_limit_rps"] == 0.0
    out["api_version_header_on_every_response"] = (
        client.get("/health").headers.get("x-fx1-api-version") == api_mod.API_VERSION
        and big_resp.headers.get("x-fx1-api-version") == api_mod.API_VERSION
        and bad_len.headers.get("x-fx1-api-version") == api_mod.API_VERSION
    )
    out["error_code_not_found"] = (
        client.post("/harness/runs", json={"command": "pwn"}).json()["code"] == "not_found"
    )
    out["error_code_validation"] = (
        client.post("/harness/runs", json={"command": 1}).json()["code"] == "validation"
    )
    out["error_code_unauthorized"] = (
        secured.get("/harness/commands").json()["code"] == "unauthorized"
    )
    out["error_code_forbidden"] = (
        remote_client.get("/harness/commands").json()["code"] == "forbidden"
    )
    out["error_code_too_large"] = big_resp.json()["code"] == "too_large"
    out["error_code_bad_request"] = bad_len.json()["code"] == "bad_request"
    client.post(
        "/harness/runs",
        json={"command": "doctor"},
        headers={"Idempotency-Key": "ec-conflict"},
    )
    out["error_code_conflict"] = (
        client.post(
            "/harness/runs",
            json={"command": "doctor", "extra_args": ["--x"]},
            headers={"Idempotency-Key": "ec-conflict"},
        ).json()["code"]
        == "idempotency_conflict"
    )
    out["error_code_honesty_gate"] = rd.json()["code"] == "honesty_gate"
    out["stream_error_code_inband"] = err_frames[0]["code"] == "backend_failure"
    cap_app = api_mod.create_app(max_inflight=1)
    cap_client = _TC2(cap_app)
    cap_app.state.inflight_slots.acquire()
    try:
        capped = cap_client.post("/harness/runs", json={"command": "doctor"})
        out["error_code_over_capacity"] = (
            capped.status_code == 503 and capped.json()["code"] == "over_capacity"
        )
    finally:
        cap_app.state.inflight_slots.release()

    # one structured access line per request, keyed by the request id
    import logging  # noqa: PLC0415

    from fx1.serve.api import logger as api_logger  # noqa: PLC0415

    class _Capture(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.lines: list[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.lines.append(record.getMessage())

    cap_log = _Capture()
    api_logger.addHandler(cap_log)
    try:
        client.get("/health", headers={"X-Request-ID": "rid-probe-1"})
    finally:
        api_logger.removeHandler(cap_log)
    line = next((ln for ln in cap_log.lines if "rid-probe-1" in ln), "")
    out["access_log_emitted"] = (
        "method=GET" in line
        and "path=/health" in line
        and "status=200" in line
        and "rid=rid-probe-1" in line
        and "elapsed_ms=" in line
    )

    # /metrics: ops snapshot — counters from the requests above (this very
    # request is recorded too, so totals are strictly increasing)
    m_before = client.get("/metrics")
    out["metrics_200"] = m_before.status_code == 200
    m = m_before.json()
    out["metrics_shape"] = all(
        k in m
        for k in (
            "uptime_s",
            "requests_total",
            "errors_total",
            "by_status",
            "inflight",
            "inflight_watermark",
            "max_inflight",
        )
    )
    out["metrics_counts_requests"] = m["requests_total"] >= 10 and "200" in m["by_status"]
    out["metrics_counts_errors"] = (
        m["errors_total"] >= 2 and "413" in m["by_status"] and "400" in m["by_status"]
    )
    out["metrics_inflight_idle"] = m["inflight"] == 0 and m["inflight_watermark"] >= 1
    out["metrics_config"] = m["max_inflight"] == 16 and m["uptime_s"] >= 0.0
    m2 = client.get("/metrics").json()
    out["metrics_monotone"] = m2["requests_total"] > m["requests_total"]
    out["metrics_secured_401"] = secured.get("/metrics").status_code == 401

    # Prometheus text exposition: content-negotiated /metrics for a
    # scrape endpoint — Accept: text/plain (or ?format=prom) renders the
    # 0.0.4 text format; unknown formats are a clean 422; the secured
    # surface keeps requiring auth.
    prom = client.get("/metrics", headers={"Accept": "text/plain"})
    body_text = prom.text
    out["prom_accept_negotiated"] = prom.status_code == 200 and prom.headers[
        "content-type"
    ].startswith("text/plain")
    out["prom_requests_series"] = (
        "# TYPE fx1_requests_total counter" in body_text
        and 'fx1_requests_total{status="200"}' in body_text
    )
    out["prom_gauges"] = (
        "# TYPE fx1_inflight gauge" in body_text
        and "fx1_draining 0" in body_text
        and "# TYPE fx1_rate_limited_total counter" in body_text
        and 'fx1_jobs{status="succeeded"}' in body_text
    )
    prom_q = client.get("/metrics?format=prometheus")
    out["prom_query_param"] = prom_q.status_code == 200 and "fx1_uptime_seconds" in prom_q.text
    out["prom_format_422"] = client.get("/metrics?format=xml").status_code == 422
    out["prom_secured_401"] = (
        secured.get("/metrics", headers={"Accept": "text/plain"}).status_code == 401
    )
    prom_json = client.get("/metrics", headers={"Accept": "application/json"})
    out["prom_json_default"] = prom_json.status_code == 200 and prom_json.headers[
        "content-type"
    ].startswith("application/json")

    # --- usage accounting -------------------------------------------------
    # Token counts the endpoint reports ride the response — per-call on
    # sync complete, batch-level delta on batch, absent (not fabricated)
    # when the backend is silent.
    class _UsageBackend(_CleanBackend):
        def __init__(self) -> None:
            super().__init__()
            self.last_usage: dict[str, int] | None = None
            self.total_usage: dict[str, int] = {}

        def complete(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> str:
            usage = {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}
            self.last_usage = usage
            for k, v in usage.items():
                self.total_usage[k] = self.total_usage.get(k, 0) + v
            return super().complete(messages)

    ube = _UsageBackend()
    uapp = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: ube))
    u_ok = uapp.post(
        "/harness/complete", json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]}
    )
    out["usage_reported_on_sync"] = u_ok.status_code == 200 and u_ok.json()["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
    }
    u_batch = uapp.post(
        "/harness/complete/batch",
        json={
            "backend": "byok",
            "batch": [
                [{"role": "u", "content": "a"}],
                [{"role": "u", "content": "b"}],
            ],
        },
    )
    out["usage_batch_total_is_delta"] = u_batch.status_code == 200 and u_batch.json()[
        "usage_total"
    ] == {"prompt_tokens": 6, "completion_tokens": 10, "total_tokens": 16}
    silent = batch_client.post(
        "/harness/complete",
        json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
    )
    out["usage_absent_when_backend_silent"] = (
        silent.status_code == 200 and silent.json()["usage"] is None
    )
    # Keep genuine integer claims; malformed counters never become charges.
    import fx1.serve.backends as _be_mod  # noqa: PLC0415

    out["usage_extract_filters"] = (
        _be_mod._extract_usage(
            {"usage": {"prompt_tokens": 3, "completion_tokens": 0, "weird": "no", "neg": -1}}
        )
        == {"prompt_tokens": 3, "completion_tokens": 0, "neg": -1}
        and all(
            _be_mod._extract_usage({"usage": {"prompt_tokens": value}}) is None
            for value in (True, False, "3", 3.0, 3.5, float("nan"), float("inf"), float("-inf"))
        )
        and _be_mod._extract_usage({}) is None
        and _be_mod._extract_usage({"usage": "broken"}) is None
    )

    # --- completion observability ------------------------------------------
    # Per-backend outcome counters + a latency histogram over attempted
    # model calls — the uapp backend served 1 sync + 2 batch items above.
    um = uapp.get("/metrics").json()
    out["metrics_complete_counters"] = um["complete"]["byok"]["ok"] == 3
    up = uapp.get("/metrics", headers={"Accept": "text/plain"}).text
    out["metrics_latency_histogram"] = (
        'fx1_complete_total{backend="byok",outcome="ok"} 3' in up
        and 'fx1_complete_latency_ms_count{backend="byok"} 3' in up
        and 'fx1_complete_latency_ms_bucket{backend="byok",le="+Inf"} 3' in up
    )
    # gate refusals land in the error outcome, not silently dropped
    dm = dirty.get("/metrics").json()
    out["metrics_complete_error_outcome"] = dm["complete"]["byok"]["error"] >= 1

    _probe_backend_probes(client, uapp, dirty, api_mod, out)
    _probe_finetune(api_mod, out)
    return out


def _probe_finetune(api_mod: Any, out: dict[str, Any]) -> None:
    """/v1/fine_tuning/jobs — the OpenAI fine-tune lifecycle over the
    staged pipeline. The stub runner is hermetic; probes pin submit-time
    validation, the event feed, cooperative cancel, idempotent replay,
    and artifact registration into the files store."""
    from fastapi.testclient import TestClient as _TC3

    from fx1.serve.finetune import FTJobOutcome

    _CORPUS = b'{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n'

    def _runner(spec: Any, *, emit: Any, should_cancel: Any) -> FTJobOutcome:
        emit("info", "stub train step", {"epoch": 1})
        art = spec.work_dir / "train_receipt.json"
        art.write_text("{}")
        return FTJobOutcome(
            fine_tuned_model=spec.ft_model_name,
            artifacts={"train_receipt": art},
            trained_tokens=None,
            checkpoint=str(spec.work_dir / "ckpt"),
        )

    ft = _TC3(api_mod.create_app(ft_runner=_runner))

    def _upload(content: bytes, purpose: str = "fine-tune") -> Any:
        return ft.post(
            "/v1/files",
            files={"file": ("corpus.jsonl", content)},
            data={"purpose": purpose},
        )

    def _wait_ft(job_id: str) -> dict[str, Any]:
        for _i in range(400):
            j = ft.get(f"/v1/fine_tuning/jobs/{job_id}").json()
            if j["status"] in ("succeeded", "failed", "cancelled"):
                return dict(j)
            time.sleep(0.02)
        return dict(ft.get(f"/v1/fine_tuning/jobs/{job_id}").json())

    # Happy path: upload purpose=fine-tune → submit → poll → succeeded,
    # artifacts registered into the files store.
    up = _upload(_CORPUS)
    out["ft_upload_finetune_purpose"] = (
        up.status_code == 200 and up.json()["purpose"] == "fine-tune"
    )
    fid = up.json()["id"]
    sub = ft.post(_PATH_FT_JOBS, json={"model": "fx1", "training_file": fid, "suffix": "audit"})
    out["ft_create_200"] = (
        sub.status_code == 200
        and sub.json()["id"].startswith("ftjob-")
        and sub.json()["object"] == "fine_tuning.job"
        and sub.json()["training_file"] == fid
    )
    jid = sub.json()["id"]
    fin = _wait_ft(jid)
    out["ft_succeeds"] = (
        fin["status"] == "succeeded"
        and fin["fine_tuned_model"] == f"ft:fx1:audit:{jid.split('-', 1)[1][:12]}"
        and fin["finished_at"] is not None
        and len(fin["result_files"]) == 1
        and fin["trained_tokens"] is None  # no tokenizer — never fabricated
    )
    out["ft_artifact_downloadable"] = (
        bool(fin["result_files"])
        and ft.get(f"/v1/files/{fin['result_files'][0]}/content").status_code == 200
        and ft.get(f"/v1/files/{fin['result_files'][0]}").json()["purpose"] == "fine-tune-result"
    )
    evs = ft.get(f"/v1/fine_tuning/jobs/{jid}/events")
    ev_msgs = [e["message"] for e in evs.json()["data"]] if evs.status_code == 200 else []
    out["ft_events_feed"] = (
        evs.status_code == 200
        and evs.json()["object"] == "list"
        and any("validated" in m for m in ev_msgs)
        and any("job started" in m for m in ev_msgs)
        and any("stub train step" in m for m in ev_msgs)
        and ev_msgs[-1].startswith("job succeeded")
    )
    out["ft_list_shape"] = (
        ft.get(_PATH_FT_JOBS).status_code == 200
        and ft.get(_PATH_FT_JOBS).json()["object"] == "list"
        and ft.get(_PATH_FT_JOBS).json()["has_more"] is False
        and ft.get(_PATH_FT_JOBS).json()["data"][0]["id"] == jid
    )

    # Fail-closed submit surface.
    out["ft_model_not_trainable_400"] = (
        ft.post(_PATH_FT_JOBS, json={"model": "byok", "training_file": fid})
        .json()
        .get("error", {})
        .get("code")
        == "model_not_trainable"
    )
    out["ft_unknown_file_404"] = (
        ft.post(_PATH_FT_JOBS, json={"model": "fx1", "training_file": "file-nope"})
        .json()
        .get("error", {})
        .get("code")
        == "file_not_found"
    )
    bfid = _upload(_CORPUS, purpose="batch").json()["id"]
    out["ft_wrong_purpose_400"] = (
        ft.post(_PATH_FT_JOBS, json={"model": "fx1", "training_file": bfid})
        .json()
        .get("error", {})
        .get("code")
        == "invalid_training_file"
    )
    mfid = _upload(b"not jsonl\n").json()["id"]
    out["ft_malformed_corpus_400"] = (
        ft.post(_PATH_FT_JOBS, json={"model": "fx1", "training_file": mfid})
        .json()
        .get("error", {})
        .get("code")
        == "invalid_training_file"
    )
    out["ft_bad_purpose_upload_400"] = (
        ft.post(
            "/v1/files",
            files={"file": (_CORPUS_FILE, _CORPUS)},
            data={"purpose": "user_data"},
        ).status_code
        == 400
    )
    out["ft_missing_routes_404"] = (
        ft.get("/v1/fine_tuning/jobs/ftjob-nope").status_code == 404
        and ft.post("/v1/fine_tuning/jobs/ftjob-nope/cancel").status_code == 404
        and ft.get("/v1/fine_tuning/jobs/ftjob-nope/events").status_code == 404
    )
    out["ft_cancel_terminal_409"] = (
        ft.post(f"/v1/fine_tuning/jobs/{jid}/cancel").status_code == 409
        and ft.post(f"/v1/fine_tuning/jobs/{jid}/cancel").json().get("error", {}).get("code")
        == "job_terminal"
    )

    # Idempotent replay: same key+body returns the same job; a different
    # body under the same key is a 409 conflict.
    ik = "ft-audit-key-1"
    r1 = ft.post(
        _PATH_FT_JOBS,
        json={"model": "fx1", "training_file": fid},
        headers={"Idempotency-Key": ik},
    )
    r2 = ft.post(
        _PATH_FT_JOBS,
        json={"model": "fx1", "training_file": fid},
        headers={"Idempotency-Key": ik},
    )
    r3 = ft.post(
        _PATH_FT_JOBS,
        json={"model": "fx1", "training_file": fid, "suffix": "other"},
        headers={"Idempotency-Key": ik},
    )
    out["ft_idem_replay"] = (
        r1.status_code == 200
        and r2.status_code == 200
        and r2.json()["id"] == r1.json()["id"]
        and r3.status_code == 409
        and r3.json().get("error", {}).get("code") == "idempotency_conflict"
    )

    # Runner failure → failed job with the typed error + error event;
    # never a 5xx on the submit itself.
    def _boom(spec: Any, *, emit: Any, should_cancel: Any) -> FTJobOutcome:
        raise RuntimeError("no gpu")

    ftf = _TC3(api_mod.create_app(ft_runner=_boom))
    bf = ftf.post(
        _PATH_FT_JOBS,
        json={
            "model": "fx1",
            "training_file": ftf.post(
                "/v1/files",
                files={"file": (_CORPUS_FILE, _CORPUS)},
                data={"purpose": "fine-tune"},
            ).json()["id"],
        },
    )
    out["ft_runner_failure_submit_200"] = bf.status_code == 200
    for _i in range(400):
        jj = ftf.get(f"/v1/fine_tuning/jobs/{bf.json()['id']}").json()
        if jj["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.02)
    out["ft_runner_failure_failed"] = (
        jj["status"] == "failed"
        and jj["error"]["code"] == "job_failed"
        and "no gpu" in jj["error"]["message"]
    )

    # Cooperative cancel: a running job honors the flag at the runner's
    # boundary and lands 'cancelled' (the runner returns early — never
    # killed mid-write).
    def _gate_runner(spec: Any, *, emit: Any, should_cancel: Any) -> FTJobOutcome:
        for _i in range(500):
            if should_cancel():
                return FTJobOutcome()
            time.sleep(0.02)
        return FTJobOutcome(fine_tuned_model=spec.ft_model_name, artifacts={})

    ftc = _TC3(api_mod.create_app(ft_runner=_gate_runner))
    cf = ftc.post(
        "/v1/files", files={"file": (_CORPUS_FILE, _CORPUS)}, data={"purpose": "fine-tune"}
    )
    cj = ftc.post(_PATH_FT_JOBS, json={"model": "fx1", "training_file": cf.json()["id"]})
    cjid = cj.json()["id"]
    for _i in range(400):
        if ftc.get(f"/v1/fine_tuning/jobs/{cjid}").json()["status"] == "running":
            break
        time.sleep(0.02)
    cc = ftc.post(f"/v1/fine_tuning/jobs/{cjid}/cancel")
    for _i in range(500):
        cjj = ftc.get(f"/v1/fine_tuning/jobs/{cjid}").json()
        if cjj["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.02)
    out["ft_cancel_running_cooperative"] = cc.status_code == 200 and cjj["status"] == "cancelled"

    # Cooperative pause/resume: a gate-aware runner parks inside
    # pause_gate once paused — status reads 'paused' while parked and
    # resume releases it to completion. 'paused' is non-terminal: a
    # parked job still honors cancel, exactly once; pause on a paused
    # job replays 200; pause or resume on a terminal job is 409; resume
    # on a non-paused job is 409.
    #
    # The queued branch is probed through a store-fabricated entry:
    # inflight == worker count, so 'queued' only exists in the scheduler
    # gap between admission and worker dispatch — unreachable
    # deterministically over the wire. The endpoint + store contract for
    # it is identical (the parked worker waits on resume.wait).
    p_hold = threading.Event()
    p_gate = threading.Event()
    p_hold2 = threading.Event()
    p_gate2 = threading.Event()

    def _pause_runner(spec: Any, *, emit: Any, should_cancel: Any, pause_gate: Any) -> FTJobOutcome:
        emit("info", "stage A")
        p_hold.wait(timeout=20)
        p_gate.set()
        if pause_gate():
            return FTJobOutcome()
        emit("info", "stage B")
        # hold the worker in-flight across the resume read — the
        # response body is the record at read time, so the probe can
        # only observe the restored 'running' while the runner is live
        p_hold2.wait(timeout=20)
        p_gate2.set()
        return FTJobOutcome(fine_tuned_model=spec.ft_model_name, artifacts={})

    ftp_app = api_mod.create_app(ft_runner=_pause_runner, max_inflight=1)
    ftp = _TC3(ftp_app)
    pfid = ftp.post(
        "/v1/files", files={"file": (_CORPUS_FILE, _CORPUS)}, data={"purpose": "fine-tune"}
    ).json()["id"]
    pa = ftp.post(
        _PATH_FT_JOBS,
        json={"model": "fx1", "training_file": pfid, "suffix": "pa"},
    ).json()
    for _i in range(400):  # pa holds the single slot, running
        if ftp.get(f"/v1/fine_tuning/jobs/{pa['id']}").json()["status"] == "running":
            break
        time.sleep(0.02)
    pqa = ftp.post(f"/v1/fine_tuning/jobs/{pa['id']}/pause")
    pqa2 = ftp.post(f"/v1/fine_tuning/jobs/{pa['id']}/pause")

    from fx1.serve.finetune import FTJob as _FTJob

    qentry = ftp_app.state.ft_store.put(
        _FTJob(
            id="ftjob-qprobe",
            model="fx1",
            created_at=int(time.time()),
            status="queued",
            training_file=pfid,
        ),
        None,
        "qprobe",
    )
    pqj = ftp.post("/v1/fine_tuning/jobs/ftjob-qprobe/pause")
    rqj = ftp.post("/v1/fine_tuning/jobs/ftjob-qprobe/resume")
    out["ft_pause_queued_running_200"] = (
        pqa.status_code == 200
        and pqa.json()["status"] == "paused"
        and pqa2.status_code == 200
        and pqa2.json()["status"] == "paused"  # idempotent replay
        and pqj.status_code == 200
        and pqj.json()["status"] == "paused"
        and rqj.status_code == 200
        and rqj.json()["status"] == "queued"  # resume restores paused_from
        and qentry.paused_from is None
    )
    # queued resume still leaves the fabricated job queued (no worker
    # ever ran) — pause it again and cancel from paused.
    ftp.post("/v1/fine_tuning/jobs/ftjob-qprobe/pause")
    cq = ftp.post("/v1/fine_tuning/jobs/ftjob-qprobe/cancel")
    out["ft_queued_cancel_terminal"] = (
        cq.status_code == 200
        and cq.json()["status"] == "cancelled"
        and ftp.post("/v1/fine_tuning/jobs/ftjob-qprobe/resume").status_code == 409
    )

    p_hold.set()  # pa's runner reaches the gate and parks
    p_gate.wait(timeout=15)
    time.sleep(0.05)
    out["ft_paused_holds_at_gate"] = ftp.get(f"/v1/fine_tuning/jobs/{pa['id']}").json()[
        "status"
    ] == "paused" and not any(
        "job succeeded" in e["message"]
        for e in ftp.get(f"/v1/fine_tuning/jobs/{pa['id']}/events").json()["data"]
    )
    rpa = ftp.post(f"/v1/fine_tuning/jobs/{pa['id']}/resume")
    p_hold2.set()  # release the resumed runner to terminal
    for _i in range(400):
        paf = ftp.get(f"/v1/fine_tuning/jobs/{pa['id']}").json()
        if paf["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.02)
    out["ft_resume_releases"] = (
        rpa.status_code == 200
        and rpa.json()["status"] == "running"  # restores paused_from
        and paf["status"] == "succeeded"
    )

    # cancel against a parked worker: one 'job cancelled' event total —
    # the store owns the terminal write, the parked worker dedupes.
    p_hold.clear()
    p_gate.clear()
    p_hold2.clear()
    p_gate2.clear()
    pc = ftp.post(
        _PATH_FT_JOBS,
        json={"model": "fx1", "training_file": pfid, "suffix": "pc"},
    ).json()
    for _i in range(400):
        if ftp.get(f"/v1/fine_tuning/jobs/{pc['id']}").json()["status"] == "running":
            break
        time.sleep(0.02)
    ftp.post(f"/v1/fine_tuning/jobs/{pc['id']}/pause")
    p_hold.set()
    p_gate.wait(timeout=15)
    ccancel = ftp.post(f"/v1/fine_tuning/jobs/{pc['id']}/cancel")
    for _i in range(400):
        pcj = ftp.get(f"/v1/fine_tuning/jobs/{pc['id']}").json()
        if pcj["status"] == "cancelled":
            break
        time.sleep(0.02)
    pc_events = ftp.get(f"/v1/fine_tuning/jobs/{pc['id']}/events").json()["data"]
    out["ft_cancel_paused_terminal"] = (
        ccancel.status_code == 200
        and ccancel.json()["status"] == "cancelled"
        and pcj["status"] == "cancelled"
        and sum("job cancelled" in e["message"] for e in pc_events) == 1
        and ftp.post(f"/v1/fine_tuning/jobs/{pc['id']}/resume").status_code == 409
    )
    out["ft_pause_guards"] = (
        ftp.post(f"/v1/fine_tuning/jobs/{pa['id']}/pause").json().get("error", {}).get("code")
        == "job_terminal"
        and ftp.post(f"/v1/fine_tuning/jobs/{pa['id']}/resume").json().get("error", {}).get("code")
        == "job_terminal"
        and ftp.post("/v1/fine_tuning/jobs/ftjob-nope/pause").status_code == 404
        and ftp.post("/v1/fine_tuning/jobs/ftjob-nope/resume").status_code == 404
    )

    # Gate-free runners (no pause_gate kwarg) accept the pause verb but
    # run to completion — the hook is opt-in, never a crash.
    def _nogate_runner(spec: Any, *, emit: Any, should_cancel: Any) -> FTJobOutcome:
        deadline = time.time() + 1.5
        while time.time() < deadline:
            if should_cancel():
                return FTJobOutcome()
            time.sleep(0.02)
        return FTJobOutcome(fine_tuned_model=spec.ft_model_name, artifacts={})

    ftn = _TC3(api_mod.create_app(ft_runner=_nogate_runner))
    nfid = ftn.post(
        "/v1/files", files={"file": (_CORPUS_FILE, _CORPUS)}, data={"purpose": "fine-tune"}
    ).json()["id"]
    nj = ftn.post(_PATH_FT_JOBS, json={"model": "fx1", "training_file": nfid}).json()
    for _i in range(400):
        if ftn.get(f"/v1/fine_tuning/jobs/{nj['id']}").json()["status"] == "running":
            break
        time.sleep(0.02)
    npa = ftn.post(f"/v1/fine_tuning/jobs/{nj['id']}/pause")
    for _i in range(400):
        njf = ftn.get(f"/v1/fine_tuning/jobs/{nj['id']}").json()
        if njf["status"] in ("succeeded", "failed", "cancelled", "paused"):
            break
        time.sleep(0.02)
    if njf["status"] == "paused":  # worker parked pre-start — release it
        ftn.post(f"/v1/fine_tuning/jobs/{nj['id']}/resume")
        for _i in range(400):
            njf = ftn.get(f"/v1/fine_tuning/jobs/{nj['id']}").json()
            if njf["status"] in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(0.02)
    out["ft_pause_gatefree_completes"] = npa.status_code == 200 and njf["status"] == "succeeded"

    # Model registry: a succeeded job with a checkpoint registers its
    # ft: name into the model inventory, and a request naming it resolves
    # to the local_fx1 lane pinned at the job's checkpoint — never the
    # default link. Unregistered ft: names fail closed 404.
    mname = fin["fine_tuned_model"]
    models = ft.get("/v1/models").json()
    out["ft_model_listed"] = mname in {m["id"] for m in models["data"]}
    card = ft.get(f"/v1/models/{mname}")
    out["ft_model_card_200"] = card.status_code == 200 and card.json()["id"] == mname
    out["ft_model_ghost_404"] = ft.get("/v1/models/ft:fx1:ghost:000000000000").status_code == 404

    resolved: list[tuple[str, Any]] = []

    def _spy(name: str, *a: Any, **k: Any) -> Any:
        resolved.append((name, k.get("checkpoint_dir") or (a[0] if a else None)))
        raise RuntimeError("no engine — resolution reached")

    ft2 = _TC3(api_mod.create_app(backend_resolver=_spy, ft_runner=_runner))
    fid2 = ft2.post(
        "/v1/files", files={"file": (_CORPUS_FILE, _CORPUS)}, data={"purpose": "fine-tune"}
    ).json()["id"]
    j2 = ft2.post(_PATH_FT_JOBS, json={"model": "fx1", "training_file": fid2, "suffix": "r"}).json()
    fin2 = j2
    for _i in range(400):
        fin2 = ft2.get(f"/v1/fine_tuning/jobs/{j2['id']}").json()
        if fin2["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.02)
    ftname = fin2["fine_tuned_model"]
    chat = ft2.post(
        "/v1/chat/completions",
        json={"model": ftname, "messages": [{"role": "user", "content": "hi"}]},
    )
    out["ft_model_routes_local_fx1"] = (
        chat.status_code == 503
        and resolved
        and resolved[-1][0] == "local_fx1"
        and str(resolved[-1][1]).endswith("ckpt")
    )
    ghost = ft2.post(
        "/v1/chat/completions",
        json={
            "model": "ft:fx1:ghost:000000000000",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    out["ft_model_unknown_404"] = (
        ghost.status_code == 404 and ghost.json().get("error", {}).get("code") == "model_not_found"
    )
    # An explicit backend + checkpoint header still wins over an ft:
    # model name — the registry never overrides a caller's stated link.
    resolved.clear()
    ft2.post(
        "/v1/chat/completions",
        json={"model": ftname, "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Fx1-Backend": "local_fx1", "X-Fx1-Checkpoint-Dir": "/srv/fx1/explicit-ckpt"},
    )
    out["ft_model_explicit_backend_wins"] = bool(
        resolved and resolved[-1] == ("local_fx1", "/srv/fx1/explicit-ckpt")
    )

    # GET /v1/fine_tuning/jobs/{id}/checkpoints — OpenAI's
    # list_checkpoints: the registered model artifacts a job produced,
    # oldest-first; unknown jobs fail closed 404.
    ck = ft2.get(f"/v1/fine_tuning/jobs/{j2['id']}/checkpoints")
    out["ft_checkpoints_list"] = (
        ck.status_code == 200
        and ck.json()["object"] == "list"
        and ck.json()["has_more"] is False
        and ck.json()["first_id"] == ck.json()["last_id"]
        and [c["fine_tuned_model_checkpoint"] for c in ck.json()["data"]] == [ftname]
        and ck.json()["data"][0]["id"].startswith("ftckpt-")
        and ck.json()["data"][0]["object"] == "fine_tuning.job.checkpoint"
    )
    out["ft_checkpoints_404"] = (
        ft2.get("/v1/fine_tuning/jobs/ftjob-nope/checkpoints").status_code == 404
    )

    # DELETE /v1/models/{id} — OpenAI's models.delete for ft: names: the
    # tombstone is real (list/retrieve/chat all go 404 after), a built-in
    # link id refuses 400, and a ghost name fails closed 404 — a delete
    # verdict is never fabricated.
    dele = ft2.delete(f"/v1/models/{ftname}")
    out["ft_model_delete_200"] = (
        dele.status_code == 200
        and dele.json()["id"] == ftname
        and dele.json()["object"] == "model"
        and dele.json()["deleted"] is True
    )
    out["ft_model_delete_gone"] = (
        ft2.get(f"/v1/models/{ftname}").status_code == 404
        and ftname not in {m["id"] for m in ft2.get("/v1/models").json()["data"]}
        and ft2.post(
            "/v1/chat/completions",
            json={"model": ftname, "messages": [{"role": "user", "content": "hi"}]},
        ).status_code
        == 404
    )
    out["ft_model_delete_ghost_404"] = (
        ft2.delete("/v1/models/ft:fx1:ghost:000000000000").status_code == 404
    )
    out["ft_model_delete_builtin_400"] = ft2.delete("/v1/models/fx1").status_code == 400
    # a deleted ft: name drops off the job's checkpoint listing — the
    # tombstone is real, no fabricated history.
    out["ft_checkpoints_delete_drops"] = (
        ft2.get(f"/v1/fine_tuning/jobs/{j2['id']}/checkpoints").json()["data"] == []
    )

    # Terminal webhooks on the /v1 surface — the fx1 extension mirrors
    # the /harness/jobs contract: fire once at the terminal transition,
    # HMAC-signed X-Fx1-Webhook-* headers when callback_secret is set,
    # and the delivery verdict (status/attempts/error) rides the record.
    import json as _json4  # noqa: PLC0415
    import threading as _threading  # noqa: PLC0415
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: PLC0415

    _ft_hits: list[tuple[dict[str, str], bytes]] = []

    class _FTHook(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 — http.server handler name
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            _ft_hits.append((dict(self.headers.items()), raw))
            self.send_response(404 if self.path == "/reject" else 200)
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            pass

    _ft_srv = ThreadingHTTPServer(("127.0.0.1", 0), _FTHook)
    _threading.Thread(target=_ft_srv.serve_forever, daemon=True).start()
    _ft_cb = f"http://127.0.0.1:{_ft_srv.server_address[1]}/ft"
    # loopback callback_url is refused by the SSRF guard without the
    # private-networks opt-in, read per request (saved/restored verbatim).
    saved_wh_env = os.environ.get("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS")
    os.environ["FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"] = "1"

    def _wait_cb(job_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + 10.0
        j: dict[str, Any] = {}
        while time.monotonic() < deadline:
            j = _wait_ft(job_id)
            if j.get("callback_status") is not None:
                return j
            time.sleep(0.02)
        return j

    cb_fid = _upload(_CORPUS).json()["id"]
    cb_job = ft.post(
        _PATH_FT_JOBS,
        json={
            "model": "fx1",
            "training_file": cb_fid,
            "callback_url": _ft_cb,
            "callback_secret": "whsec-audit",
        },
    ).json()
    cb_fin = _wait_cb(cb_job["id"])
    signed_ok = False
    if _ft_hits:
        from fx1.serve.webhooks import verify_webhook  # noqa: PLC0415

        _h, _b = _ft_hits[0]
        signed_ok = verify_webhook(
            "whsec-audit",
            _h.get("X-Fx1-Webhook-Timestamp"),
            _h.get("X-Fx1-Webhook-Signature"),
            _b,
        )
    out["ft_webhook_fires_signed"] = (
        len(_ft_hits) == 1
        and cb_fin.get("callback_status") == "delivered"
        and cb_fin.get("callback_attempts") == 1
        and _json4.loads(_ft_hits[0][1])["status"] == "succeeded"
        and signed_ok
    )
    out["ft_webhook_secret_never_serializes"] = (
        "callback_secret" not in cb_fin and "callback_secret" not in _json4.loads(_ft_hits[0][1])
    )
    # 4xx is definitive — one attempt, no retry storm.
    rj_fid = _upload(_CORPUS).json()["id"]
    rj_job = ft.post(
        _PATH_FT_JOBS,
        json={
            "model": "fx1",
            "training_file": rj_fid,
            "callback_url": _ft_cb.replace("/ft", "/reject"),
        },
    ).json()
    rj_fin = _wait_cb(rj_job["id"])
    out["ft_webhook_4xx_never_retried"] = (
        rj_fin.get("callback_status") == "failed"
        and rj_fin.get("callback_attempts") == 1
        and len(_ft_hits) == 2
        and "404" in (rj_fin.get("callback_error") or "")
    )
    # Submit-time guards: secret requires url; url must be http(s) with a
    # host — both as the /v1 envelope's 422, never a queued zombie.
    sec_only = ft.post(
        _PATH_FT_JOBS,
        json={"model": "fx1", "training_file": cb_fid, "callback_secret": "x"},
    )
    bad_url = ft.post(
        _PATH_FT_JOBS,
        json={"model": "fx1", "training_file": cb_fid, "callback_url": "ftp://x"},
    )
    out["ft_webhook_guards_422"] = (
        sec_only.status_code == 422
        and bad_url.status_code == 422
        and sec_only.json().get("error", {}).get("code") == "validation"
    )
    _ft_srv.shutdown()
    _ft_srv.server_close()
    if saved_wh_env is None:
        os.environ.pop("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS", None)
    else:
        os.environ["FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"] = saved_wh_env


def _probe_backend_probes(  # NOSONAR
    client: Any, uapp: Any, dirty: Any, api_mod: Any, out: dict[str, Any]
) -> None:
    import json as _json  # noqa: PLC0415

    from fastapi.testclient import TestClient as _TC2  # noqa: PLC0415

    # Deep health through the real resolver: a live call, not config flags.
    # Bypasses the breaker, never feeds it; verdicts land under
    # ``probe:<name>`` so monitoring never pollutes completion SLOs.
    p_ok = uapp.post("/harness/backends/byok/probe", json={})
    pj = p_ok.json()
    out["probe_ok"] = (
        p_ok.status_code == 200
        and pj["ok"] is True
        and pj["model"] == "fake-0"
        and pj["latency_ms"] >= 0
    )
    p_dirty = dirty.post("/harness/backends/byok/probe", json={})
    out["probe_honesty_refusal"] = (
        p_dirty.status_code == 200
        and p_dirty.json()["ok"] is False
        and p_dirty.json()["error_class"] == "honesty_refusal"
    )

    # unconfigured backend is a verdict, not a wire fault
    def _unconfigured(*a: Any, **k: Any) -> Any:
        raise RuntimeError("BYOK backend is not configured")

    unconf = _TC2(api_mod.create_app(backend_resolver=_unconfigured))
    p_none = unconf.post("/harness/backends/byok/probe", json={})
    out["probe_unconfigured_verdict"] = (
        p_none.status_code == 200
        and p_none.json()["ok"] is False
        and p_none.json()["error_class"] == "backend_unavailable"
    )
    out["probe_unknown_422"] = (
        client.post("/harness/backends/bogus/probe", json={}).status_code == 422
    )
    pm = uapp.get("/metrics").json()
    out["probe_own_series"] = pm["complete"]["probe:byok"]["ok"] >= 1 and (
        "byok" in pm["complete"] and pm["complete"]["byok"]["ok"] == 3
    )

    # gate pre-flight: the honesty gate callable over the wire — a verdict,
    # not a model call. Stays up during drain, never metered.
    g_ok = client.post("/harness/gate/check", json={"text": "the result used bootstrap intervals"})
    g_bad = client.post("/harness/gate/check", json={"text": "we report Sharpe 2.1 out of sample"})
    out["gate_check_clean"] = g_ok.status_code == 200 and g_ok.json()["ok"] is True
    out["gate_check_refusal"] = (
        g_bad.status_code == 200
        and g_bad.json()["ok"] is False
        and "forbidden" in (g_bad.json()["error"] or "")
    )
    out["gate_check_oversize_422"] = (
        client.post("/harness/gate/check", json={"text": "x" * 262145}).status_code == 422
    )

    # score surface: the deterministic reward contract over the wire —
    # components + violations verbatim, not a model call.
    s_one = client.post(
        "/harness/score",
        json={
            "input": "verify-research pins receipt 0123456789abcdef — "
            "simulated evidence, CRPS 0.4, uncertainty calibrated"
        },
    )
    s_many = client.post(
        "/harness/score",
        json={"input": ["", "Sharpe 3.2 live trading NAV up", "x"]},
    )
    sd_many = s_many.json()["data"]
    out["score_single_200"] = (
        s_one.status_code == 200
        and s_one.json()["object"] == "list"
        and len(s_one.json()["data"]) == 1
        and s_one.json()["data"][0]["object"] == "score"
        and s_one.json()["data"][0]["index"] == 0
        and s_one.json()["data"][0]["total"] == 8.5
    )
    out["score_components"] = (
        s_one.json()["data"][0]["components"].get("honesty_clean") == 4.0
        and s_one.json()["data"][0]["components"].get("cites_receipt") == 2.0
        and s_one.json()["data"][0]["components"].get("proper_score_vocabulary") == 1.0
    )
    out["score_empty_zero"] = (
        s_many.status_code == 200
        and len(sd_many) == 3
        and sd_many[0]["index"] == 0
        and sd_many[0]["total"] == 0.0
        and sd_many[0]["components"] == {}
    )
    out["score_honesty_violation"] = (
        sd_many[1]["total"] == -10.0
        and len(sd_many[1]["violations"]) == 1
        and "forbidden" in sd_many[1]["violations"][0]
    )
    out["score_input_422"] = (
        client.post("/harness/score", json={"input": []}).status_code == 422
        and client.post("/harness/score", json={"input": [1]}).status_code == 422
        and client.post("/harness/score", json={"input": "x" * 262145}).status_code == 422
        and client.post("/harness/score", json={"input": ["x"] * 129}).status_code == 422
    )

    # /v1/moderations — the honesty gate in the OpenAI moderation wire
    # shape: per-input {flagged, categories, category_scores} and a
    # content-derived modr- id.
    m_flag = client.post("/v1/moderations", json={"input": "Sharpe 3.2 live trading NAV up"})
    mr_flag = m_flag.json()["results"][0]
    out["moderations_flagged_200"] = (
        m_flag.status_code == 200
        and m_flag.json()["model"] == "fx1-honesty-gate"
        and m_flag.json()["id"].startswith("modr-")
        and len(m_flag.json()["results"]) == 1
        and mr_flag["flagged"] is True
        and mr_flag["categories"]["forbidden_headline_metric"] is True
        and mr_flag["category_scores"]["forbidden_headline_metric"] == 1.0
        and mr_flag["category_applied_input_types"]["forbidden_headline_metric"] == ["text"]
    )
    m_clean = client.post(
        "/v1/moderations",
        json={"input": "verify-research pins receipt 0123456789abcdef — CRPS 0.4"},
    )
    mr_clean = m_clean.json()["results"][0]
    out["moderations_clean"] = (
        m_clean.status_code == 200
        and mr_clean["flagged"] is False
        and all(v is False for v in mr_clean["categories"].values())
        and all(v == 0.0 for v in mr_clean["category_scores"].values())
    )
    m_many = client.post(
        "/v1/moderations",
        json={"input": ["clean text", "Sharpe 3.2 live trading NAV up", "also clean"]},
    )
    mr_many = m_many.json()["results"]
    out["moderations_list"] = (
        m_many.status_code == 200
        and len(mr_many) == 3
        and [r["flagged"] for r in mr_many] == [False, True, False]
    )
    m_same = client.post(
        "/v1/moderations",
        json={"input": "verify-research pins receipt 0123456789abcdef — CRPS 0.4"},
    )
    m_diff = client.post("/v1/moderations", json={"input": "other text"})
    out["moderations_deterministic_id"] = (
        m_clean.json()["id"] == m_same.json()["id"] and m_same.json()["id"] != m_diff.json()["id"]
    )
    m_synth = client.post(
        "/v1/moderations",
        json={"input": "the synthetic results show accuracy 0.99"},
    )
    m_labeled = client.post(
        "/v1/moderations",
        json={"input": "the SYNTHETIC results show accuracy 0.99"},
    )
    out["moderations_unlabeled_synthetic"] = (
        m_synth.json()["results"][0]["categories"]["unlabeled_synthetic"] is True
        and m_synth.json()["results"][0]["flagged"] is True
        and m_labeled.json()["results"][0]["flagged"] is False
    )
    out["moderations_input_422"] = (
        client.post("/v1/moderations", json={"input": []}).status_code == 422
        and client.post("/v1/moderations", json={"input": [1]}).status_code == 422
        and client.post("/v1/moderations", json={"input": "x" * 262145}).status_code == 422
        and client.post("/v1/moderations", json={"input": ["x"] * 129}).status_code == 422
    )

    # usage ledger: backend-reported tokens accumulate per series; the
    # usage_calls counter separates "silent provider" from "zero bill".
    m2 = uapp.get("/metrics").json()["complete"]
    out["usage_metrics_tokens"] = (
        m2["byok"]["total_tokens"] >= 24
        and m2["byok"]["usage_calls"] >= 3
        and m2["byok"]["prompt_tokens"] >= 9
        and m2["probe:byok"]["total_tokens"] >= 8
        and m2["probe:byok"]["usage_calls"] == 1
    )
    prom = uapp.get("/metrics", params={"format": "prom"}).text
    out["usage_prom_lines"] = (
        'fx1_complete_tokens_total{backend="byok",kind="total"}' in prom
        and 'fx1_complete_usage_calls_total{backend="byok"}' in prom
    )

    # streaming usage: a backend that reports usage at stream end lands it
    # on the final SSE frame AND the metrics ledger.
    class _StreamUsage:
        def __init__(self) -> None:
            self.last_usage: dict[str, int] | None = None
            self._model = "stream-usage-0"

        def complete(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> str:
            return "hello"

        def stream(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> Any:
            yield "he"
            yield "llo"
            self.last_usage = {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}

    sapp = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _StreamUsage()))
    s_ok = sapp.post(
        "/harness/complete/stream",
        json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
    )
    frames = [
        _json.loads(ln[len("data: ") :])
        for ln in s_ok.text.splitlines()
        if ln.startswith("data: ") and ln[len("data: ") :] != "[DONE]"
    ]
    s_final = next(f for f in frames if f.get("type") == "final")
    s_m = sapp.get("/metrics").json()["complete"]["byok"]
    out["stream_usage_final_frame"] = s_final.get("usage") == {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
    }
    out["stream_usage_metrics"] = s_m["total_tokens"] == 8 and s_m["usage_calls"] == 1

    # wire parser: a provider chunk carrying ``usage`` (incl. one with no
    # ``choices`` at all) lands in the out-box; malformed stays fatal.
    class _FakeResp:
        def __init__(self, lines: list[bytes]) -> None:
            self._lines = lines

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

        def __iter__(self) -> Any:
            return iter(self._lines)

    import fx1.serve.backends as _be_mod  # noqa: PLC0415

    wire_frames = [
        b'data: {"choices":[{"delta":{"content":"he"}}]}\n',
        b'data: {"usage":{"prompt_tokens":7,"completion_tokens":1,"total_tokens":8}}\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}\n',
        b"data: [DONE]\n",
    ]
    # patch the live seam: all wire calls route through
    # ``backends._openai_urlopen``, not urllib.request.urlopen.
    orig_urlopen = _be_mod._openai_urlopen
    _be_mod._openai_urlopen = lambda req, timeout_s=None, **kw: _FakeResp(wire_frames)  # type: ignore[assignment]
    try:
        box: list[dict[str, int]] = []
        toks = list(
            _be_mod._openai_chat_stream(
                "http://wire.test",
                model="m",
                messages=[],
                timeout_s=1.0,
                api_key=None,
                label="t",
                usage_out=box,
            )
        )
    finally:
        _be_mod._openai_urlopen = orig_urlopen
    out["stream_usage_parser"] = toks == ["he"] and box == [
        {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
    ]

    # last-probe cache: GET /harness/backends surfaces the most recent
    # verdict so a scrape reads deep health without spending a live call.
    b_u = uapp.get("/harness/backends").json()["byok"]["last_probe"]
    b_d = dirty.get("/harness/backends").json()["byok"]["last_probe"]
    fresh = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: None))
    out["backends_last_probe"] = (
        b_u is not None
        and b_u["ok"] is True
        and b_u["checked_at"] > 0
        and b_d is not None
        and b_d["ok"] is False
        and b_d["error_class"] == "honesty_refusal"
        and fresh.get("/harness/backends").json()["byok"]["last_probe"] is None
    )

    # completion log: every gated call leaves a fetchable record (hashes,
    # usage, verdict) — the harness's own calls are auditable evidence.
    import hashlib as _hl  # noqa: PLC0415

    c_res = uapp.post(
        "/harness/complete",
        json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
    )
    c_id = c_res.json()["completion_id"]
    c_rec = uapp.get(f"/harness/completions/{c_id}")
    want_p = _hl.sha256(
        _json.dumps([{"role": "u", "content": "x"}], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    want_o = _hl.sha256(b"clean:x").hexdigest()
    out["completions_logged_sync"] = (
        c_res.status_code == 200
        and c_res.headers.get("x-fx1-completion-id") == c_id
        and c_rec.status_code == 200
        and c_rec.json()["ok"] is True
        and c_rec.json()["prompt_sha256"] == want_p
        and c_rec.json()["output_sha256"] == want_o
        and c_rec.json()["usage"] == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}
    )
    out["completions_get_404"] = (
        uapp.get("/harness/completions/00000000000000000000000000000000").status_code == 404
    )
    d_fail = [
        r
        for r in dirty.get("/harness/completions", params={"backend": "byok"}).json()["items"]
        if r["ok"] is False
    ]
    out["completions_logged_failure"] = (
        len(d_fail) >= 1
        and d_fail[0]["error_class"] == "honesty_refusal"
        and d_fail[0]["output_sha256"] is None
    )
    out["completions_list_filter"] = (
        uapp.get("/harness/completions", params={"backend": "byok"}).json()["count"] >= 4
        and uapp.get("/harness/completions", params={"backend": "local_fx1"}).json()["count"] == 0
        and uapp.get("/harness/completions", params={"limit": 0}).status_code == 422
    )

    s_final_id = s_final.get("completion_id")
    s_rec = sapp.get(f"/harness/completions/{s_final_id}")
    out["completions_logged_stream"] = (
        isinstance(s_final_id, str)
        and s_rec.status_code == 200
        and s_rec.json()["ok"] is True
        and s_rec.json()["output_sha256"] == _hl.sha256(b"hello").hexdigest()
    )

    b_res = uapp.post(
        "/harness/complete/batch",
        json={"backend": "byok", "batch": [[{"role": "u", "content": "q"}]]},
    ).json()
    b_cid = b_res["results"][0]["completion_id"]
    b_rec = uapp.get(f"/harness/completions/{b_cid}")
    out["completions_logged_batch"] = (
        isinstance(b_cid, str)
        and b_rec.status_code == 200
        and b_rec.json()["ok"] is True
        and b_rec.json()["usage"] is None
    )

    # the ring is bounded and newest-first — oldest records evict at cap
    cap3 = api_mod._CompletionLog(cap=3)
    for i in range(5):
        cap3.append(
            api_mod.CompletionRecord(
                completion_id=f"c{i}",
                backend="byok",
                ok=True,
                latency_ms=0.0,
                at=float(i),
                prompt_sha256="p",
            )
        )
    out["completions_ring_cap"] = (
        cap3.get("c0") is None
        and len(cap3.latest(10, None)) == 3
        and [r.completion_id for r in cap3.latest(10, None)] == ["c4", "c3", "c2"]
        and cap3.dropped == 2
        and cap3.cap == 3
    )

    # --- /harness/usage — token/request accounting over the ring ----------
    # two ok calls carrying usage + one 502 — totals, splits, filters, and
    # the truncation honesty fields all assert.
    class _UsageBackend:
        def __init__(self) -> None:
            self._model = "fake-0"

        def complete(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> str:
            self.last_usage = {
                "prompt_tokens": 5,
                "completion_tokens": 7,
                "total_tokens": 12,
                "cached_tokens": 2,
            }
            return "clean"

        def close(self) -> None:
            pass

    class _DeadBackend:
        def complete(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> str:
            raise RuntimeError("dead")

        def close(self) -> None:
            pass

    u_ok = _UsageBackend()
    u_app = _TC2(
        api_mod.create_app(
            backend_resolver=lambda name, *a, **k: u_ok if name == "byok" else _DeadBackend()
        )
    )
    for _i in range(2):
        u_app.post(
            "/harness/complete",
            json={"backend": "byok", "messages": [{"role": "user", "content": "hi"}]},
        )
    u_app.post(
        "/harness/complete",
        json={"backend": "hosted_k3", "messages": [{"role": "user", "content": "hi"}]},
    )
    u = u_app.get("/harness/usage").json()
    out["usage_totals"] = (
        u["totals"]["requests"] == 3
        and u["totals"]["ok"] == 2
        and u["totals"]["errors"] == 1
        and u["totals"]["prompt_tokens"] == 10
        and u["totals"]["completion_tokens"] == 14
        and u["totals"]["total_tokens"] == 24
        and u["totals"]["other_usage"] == {"cached_tokens": 4}
        and u["totals"]["usage_reported"] == 2
        and u["totals"]["mean_latency_ms"] is not None
        and u["records_seen"] == 3
        and u["records_dropped"] == 0
        and u["ring_cap"] == api_mod._COMPLETION_LOG_MAX
    )
    out["usage_splits"] = (
        u["by_backend"]["byok"]["requests"] == 2
        and u["by_backend"]["hosted_k3"]["errors"] == 1
        and u["by_model"]["fake-0"]["requests"] == 2
    )
    out["usage_filters"] = (
        u_app.get("/harness/usage?backend=byok").json()["totals"]["requests"] == 2
        and u_app.get("/harness/usage?model=fake-0").json()["totals"]["requests"] == 2
        and u_app.get("/harness/usage?model=nope").json()["totals"]["requests"] == 0
        and u_app.get(f"/harness/usage?since={time.time() + 60}").json()["records_seen"] == 0
        and u_app.get("/harness/usage?until=1").json()["records_seen"] == 0
    )
    out["usage_bad_window_400"] = (
        u_app.get("/harness/usage?since=2&until=1").status_code == 400
        and u_app.get("/harness/usage?since=-1").status_code == 422
    )

    # --- sealed per-call receipt export -------------------------------------
    # one logged call exports as a sealed fx1_completion_record.v1 doc:
    # seal re-derives, verify_receipt accepts it, tampering the record's
    # output hash breaks the seal, and exports are byte-deterministic.
    from quant_fund.schemas.receipt import (  # noqa: PLC0415
        verify_receipt_payload as _vrp,
    )
    from quant_fund.utils.hashing import (  # noqa: PLC0415
        canonical_json_bytes as _cjb,
    )
    from quant_fund.utils.hashing import (
        hash_bytes as _hb,
    )

    rc = uapp.get(f"/harness/completions/{c_id}/receipt")
    rc_doc = rc.json()
    out["completion_receipt_export"] = (
        rc.status_code == 200
        and rc_doc["kind"] == "fx1_completion_record"
        and rc_doc["schema"] == "fx1_completion_record.v1"
        and rc_doc["record"]["completion_id"] == c_id
        and rc_doc["receipt_sha256"]
        == _hb(_cjb({k: v for k, v in rc_doc.items() if k != "receipt_sha256"}))
    )
    out["completion_receipt_verifies"] = _vrp(rc_doc)["valid"] is True
    out["completion_receipt_verify_route"] = (
        uapp.post("/receipts/verify", json={"receipt": rc_doc}).json().get("valid") is True
    )
    out["completion_receipt_deterministic"] = (
        uapp.get(f"/harness/completions/{c_id}/receipt").json() == rc_doc
    )
    tampered_rec = _json.loads(_json.dumps(rc_doc))
    tampered_rec["record"]["output_sha256"] = "0" * 64
    out["completion_receipt_tamper"] = _vrp(tampered_rec)["valid"] is False
    out["completion_receipt_404"] = (
        uapp.get("/harness/completions/00000000000000000000000000000000/receipt").status_code == 404
    )

    # --- sealed job receipt -------------------------------------------------
    # a terminal job exports as fx1_job_record.v1: streams digested (never
    # content), seal re-derives, verify route accepts, tampering breaks.
    from fx1.harness import Harness as _Harness  # noqa: PLC0415

    jr = _TC2(
        api_mod.create_app(
            harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), ""))
        )
    )
    j_id = jr.post("/harness/jobs", json={"command": "doctor"}).json()["job_id"]
    for _ in range(500):
        if jr.get(f"/harness/jobs/{j_id}").json()["status"] in (
            "succeeded",
            "failed",
            "cancelled",
        ):
            break
        time.sleep(0.01)
    jr_res = jr.get(f"/harness/jobs/{j_id}/receipt")
    jr_doc = jr_res.json()
    want_stdout = _hl.sha256(b"ran:doctor").hexdigest()
    out["job_receipt_export"] = (
        jr_res.status_code == 200
        and jr_doc["kind"] == "fx1_job_record"
        and jr_doc["schema"] == "fx1_job_record.v1"
        and jr_doc["record"]["job_id"] == j_id
        and jr_doc["record"]["status"] == "succeeded"
        and jr_doc["record"]["result"]["stdout_sha256"] == want_stdout
        and "stdout" not in jr_doc["record"]["result"]
        and jr_doc["receipt_sha256"]
        == _hb(_cjb({k: v for k, v in jr_doc.items() if k != "receipt_sha256"}))
    )
    out["job_receipt_verifies"] = (
        _vrp(jr_doc)["valid"] is True
        and jr.post("/receipts/verify", json={"receipt": jr_doc}).json().get("valid") is True
    )
    out["job_receipt_deterministic"] = jr.get(f"/harness/jobs/{j_id}/receipt").json() == jr_doc
    tampered_job = _json.loads(_json.dumps(jr_doc))
    tampered_job["record"]["status"] = "failed"
    out["job_receipt_tamper"] = _vrp(tampered_job)["valid"] is False
    out["job_receipt_404"] = jr.get("/harness/jobs/nope/receipt").status_code == 404

    # --- sampling controls --------------------------------------------------
    # declared decode fields resolve server-side into one wire dict: the
    # backend receives a SamplingParams carrying exactly what was declared,
    # the response/record echo the resolved set, and the unpinned default
    # stays temperature-pinned at 0.0 (eval determinism).
    from fx1.serve.backends import SamplingParams as _SP  # noqa: PLC0415

    class _SamplingSpy:
        def __init__(self) -> None:
            self.seen: _SP | None = None
            self._model = "sampling-spy-0"

        def complete(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> str:
            self.seen = sampling
            return "clean:x"

        def stream(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> Any:
            self.seen = sampling
            yield "clean:x"

    spy = _SamplingSpy()
    sp_app = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: spy))
    sp_res = sp_app.post(
        "/harness/complete",
        json={
            "backend": "byok",
            "messages": [{"role": "u", "content": "x"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 64,
            "seed": 17,
        },
    )
    want_fields = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 64, "seed": 17}
    sp_rec = sp_app.get(f"/harness/completions/{sp_res.json()['completion_id']}")
    out["sampling_sync_resolved"] = (
        sp_res.status_code == 200
        and sp_res.json()["sampling"] == want_fields
        and isinstance(spy.seen, _SP)
        and spy.seen.body_fields() == want_fields
        and sp_rec.json()["sampling"] == want_fields
    )
    spy.seen = None
    d_res = sp_app.post(
        "/harness/complete",
        json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
    )
    out["sampling_default_pin"] = (
        d_res.status_code == 200
        and d_res.json()["sampling"] == {"temperature": 0.0}
        and spy.seen is not None
        and spy.seen.body_fields() == {"temperature": 0.0}
    )
    out["sampling_field_422"] = all(
        sp_app.post(
            "/harness/complete",
            json={
                "backend": "byok",
                "messages": [{"role": "u", "content": "x"}],
                k: v,
            },
        ).status_code
        == 422
        for k, v in (
            ("temperature", 2.5),
            ("top_p", 0.0),
            ("max_tokens", 0),
            ("seed", -1),
        )
    )
    spy.seen = None
    sb_res = sp_app.post(
        "/harness/complete/batch",
        json={
            "backend": "byok",
            "batch": [[{"role": "u", "content": "q"}]],
            "temperature": 0.3,
            "seed": 7,
        },
    )
    out["sampling_batch_resolved"] = (
        sb_res.status_code == 200
        and sb_res.json()["sampling"] == {"temperature": 0.3, "seed": 7}
        and sb_res.json()["results"][0]["ok"] is True
        and spy.seen is not None
        and spy.seen.body_fields() == {"temperature": 0.3, "seed": 7}
    )
    spy.seen = None
    ss_res = sp_app.post(
        "/harness/complete/stream",
        json={
            "backend": "byok",
            "messages": [{"role": "u", "content": "x"}],
            "temperature": 0.2,
            "max_tokens": 8,
        },
    )
    ss_final = next(
        _json.loads(ln[len("data: ") :])
        for ln in ss_res.text.splitlines()
        if ln.startswith("data: ") and _json.loads(ln[len("data: ") :]).get("type") == "final"
    )
    out["sampling_stream_final_frame"] = (
        ss_res.status_code == 200
        and ss_final.get("sampling") == {"temperature": 0.2, "max_tokens": 8}
        and spy.seen is not None
        and spy.seen.body_fields() == {"temperature": 0.2, "max_tokens": 8}
    )

    # wire level: declared fields reach the provider body; undeclared knobs
    # (seed/top_p/max_tokens) are never sent to a backend that got no request
    # for them — and the temperature pin still ships at 0.0.
    captured_wire: dict[str, Any] = {}

    class _WireResp:
        def read(self) -> bytes:
            return _json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

    def _wire_urlopen(req: Any, **kw: Any) -> Any:
        captured_wire["body"] = _json.loads(req.data.decode())
        return _WireResp()

    _be_mod._openai_urlopen = _wire_urlopen  # type: ignore[assignment]
    try:
        _be_mod._openai_chat_complete(
            "http://wire.test",
            model="m",
            messages=[{"role": "user", "content": "q"}],
            timeout_s=1.0,
            api_key=None,
            label="t",
            sampling=_SP(temperature=0.5, top_p=0.95, max_tokens=32, seed=42),
        )
        full_body = dict(captured_wire["body"])
        _be_mod._openai_chat_complete(
            "http://wire.test",
            model="m",
            messages=[{"role": "user", "content": "q"}],
            timeout_s=1.0,
            api_key=None,
            label="t",
        )
        default_body = dict(captured_wire["body"])
    finally:
        _be_mod._openai_urlopen = orig_urlopen
    out["sampling_wire_declared"] = (
        full_body.get("temperature") == 0.5
        and full_body.get("top_p") == 0.95
        and full_body.get("max_tokens") == 32
        and full_body.get("seed") == 42
        and default_body
        == {
            "model": "m",
            "messages": [{"role": "user", "content": "q"}],
            "temperature": 0.0,
        }
    )

    # --- eval submissions ---------------------------------------------------
    # Evals are jobs: submit → poll → terminal record → sealed receipt.
    # The model under test is the resolved backend chain, metered under
    # ``eval:{suite}:{backend}``; the decode pin ({"temperature": 0.0}) is
    # stamped on the record.
    import threading as _threading  # noqa: PLC0415

    class _EvalBackend:
        def __init__(self) -> None:
            self._model = "fake-0"
            self.calls = 0

        def complete(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> str:
            self.calls += 1
            return "clean:yes"

    eval_backend = _EvalBackend()
    eval_app = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: eval_backend))
    ev_sub = eval_app.post(
        "/harness/evals", json={"suite": "tooluse", "backend": "byok", "seed": 0}
    )
    ev_j = ev_sub.json()
    ev_id = ev_j.get("eval_id", "")
    # the eval worker can finish before the response snapshot is read on a
    # warm process — a 202 with a terminal status is the same acceptance.
    out["eval_submit_202"] = (
        ev_sub.status_code == 202
        and ev_j.get("status") in ("queued", "running", "succeeded")
        and ev_j.get("replayed") is False
        and bool(ev_id)
        and ev_sub.headers.get("location") == f"/harness/evals/{ev_id}"
    )

    def _wait_eval(c: Any, eid: str, tries: int = 200) -> dict[str, Any]:
        st: dict[str, Any] = {}
        for _i in range(tries):
            st = c.get(f"/harness/evals/{eid}").json()
            if st.get("status") in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(0.02)
        return st

    ev_rec = _wait_eval(eval_app, ev_id)
    out["eval_terminal_succeeded"] = (
        ev_rec.get("status") == "succeeded" and ev_rec.get("error") is None
    )
    out["eval_report_serialized"] = isinstance(ev_rec.get("report"), dict) and bool(
        ev_rec["report"]
    )
    out["eval_sampling_pin"] = ev_rec.get("sampling") == {"temperature": 0.0}
    out["eval_backend_recorded"] = ev_rec.get("backend") == "byok" and isinstance(
        ev_rec.get("attempts"), list
    )
    out["eval_model_metered"] = eval_backend.calls > 0
    pm_eval = eval_app.get("/metrics").json()
    out["eval_metric_key"] = (
        pm_eval["complete"].get("eval:tooluse:byok", {}).get("latency_count", 0) > 0
    )
    ev_receipt = eval_app.get(f"/harness/evals/{ev_id}/receipt")
    ev_rcpt = ev_receipt.json() if ev_receipt.status_code == 200 else {}
    out["eval_receipt_200"] = (
        ev_receipt.status_code == 200
        and ev_rcpt.get("kind") == "fx1_eval_record"
        and ev_rcpt.get("schema") == "fx1_eval_record.v1"
        and ev_rcpt.get("record", {}).get("eval_id") == ev_id
        and ev_rcpt["record"].get("status") == "succeeded"
        and ev_rcpt.get("receipt_sha256")
        == _hb(_cjb({k: v for k, v in ev_rcpt.items() if k != "receipt_sha256"}))
    )
    out["eval_receipt_verifies"] = (
        bool(ev_rcpt)
        and _vrp(ev_rcpt)["valid"] is True
        and eval_app.post("/receipts/verify", json={"receipt": ev_rcpt}).json().get("valid") is True
    )
    tampered_ev = _json.loads(_json.dumps(ev_rcpt))
    tampered_ev["record"]["status"] = "cancelled"
    out["eval_receipt_tamper_breaks"] = _vrp(tampered_ev)["valid"] is False
    ev_list = eval_app.get("/harness/evals?suite=tooluse&status=succeeded")
    ev_list_j = ev_list.json()
    out["eval_list_filters"] = (
        ev_list.status_code == 200
        and ev_list_j["total"] >= 1
        and all(
            r["suite"] == "tooluse" and r["status"] == "succeeded" for r in ev_list_j["records"]
        )
    )
    ev_list_all = eval_app.get("/harness/evals")
    out["eval_list_all"] = ev_list_all.json()["total"] >= 1

    # Idempotency: same key+body replays, same key+different body 409s.
    key = "eval-idem-probe"
    hdrs = {"Idempotency-Key": key}
    body = {"suite": "retrieval", "backend": "byok", "seed": 1}
    idem1 = eval_app.post("/harness/evals", json=body, headers=hdrs)
    idem2 = eval_app.post("/harness/evals", json=body, headers=hdrs)
    idem3 = eval_app.post(
        "/harness/evals",
        json={**body, "seed": 2},
        headers=hdrs,
    )
    out["eval_idem_replay"] = (
        idem1.status_code == 202
        and idem2.status_code == 202
        and idem2.json()["eval_id"] == idem1.json()["eval_id"]
        and idem2.json()["replayed"] is True
        and idem3.status_code == 409
    )
    _wait_eval(eval_app, idem1.json()["eval_id"])

    # Contract guards: unknown suite 422, judge on a non-judge suite 422,
    # judge_byok without judge_backend='byok' 422, unknown eval id 404,
    # receipt on a non-terminal record 409, cancel on queued 200.
    out["eval_unknown_suite_422"] = (
        eval_app.post("/harness/evals", json={"suite": "bogus", "backend": "byok"}).status_code
        == 422
    )
    out["eval_judge_on_nonjudge_422"] = (
        eval_app.post(
            "/harness/evals",
            json={"suite": "tooluse", "backend": "byok", "judge_backend": "hosted_k3"},
        ).status_code
        == 422
    )
    out["eval_judge_byok_misbind_422"] = (
        eval_app.post(
            "/harness/evals",
            json={
                "suite": "capability",
                "backend": "byok",
                "judge_backend": "hosted_k3",
                "judge_byok": {"base_url": "http://x", "api_key": "k", "model": "m"},
            },
        ).status_code
        == 422
    )
    out["eval_unknown_404"] = eval_app.get("/harness/evals/nope").status_code == 404

    # Cooperative cancel of a queued eval + non-terminal receipt 409:
    # occupy the single executor worker without holding an inflight slot
    # (slots == workers) — a slot-free sleeper leaves the submission's
    # semaphore acquire free while its _exec sits queued behind the sleeper.
    hold_ev = _threading.Event()
    qapp_ev = api_mod.create_app(backend_resolver=lambda *a, **k: eval_backend, max_inflight=1)
    qc2 = _TC2(qapp_ev)
    qapp_ev.state.jobs_executor.submit(lambda: hold_ev.wait(timeout=20))
    be_id = qc2.post("/harness/evals", json={"suite": "tooluse", "backend": "byok"}).json()[
        "eval_id"
    ]
    out["eval_queues_when_workers_busy"] = (
        qc2.get(f"/harness/evals/{be_id}").json()["status"] == "queued"
    )
    out["eval_receipt_nonterminal_409"] = (
        qc2.get(f"/harness/evals/{be_id}/receipt").status_code == 409
    )
    canc = qc2.delete(f"/harness/evals/{be_id}")
    out["eval_cancel_queued_200"] = canc.status_code == 200 and canc.json()["status"] == "cancelled"
    hold_ev.set()
    out["eval_cancelled_never_runs"] = qc2.get(f"/harness/evals/{be_id}").json()["report"] is None

    # Eval-diff — the promotion-gate primitive. Two succeeded evals on
    # the same bank → comparable + verdict; a cancelled record 409s; a
    # cross-suite diff is served but incomparable (verdict 'unknown').
    ev2_id = eval_app.post(
        "/harness/evals", json={"suite": "tooluse", "backend": "byok", "seed": 0}
    ).json()["eval_id"]
    ev2_rec = _wait_eval(eval_app, ev2_id)
    diff = eval_app.get(f"/harness/evals/{ev_id}/diff/{ev2_id}")
    dj = diff.json() if diff.status_code == 200 else {}
    out["eval_diff_200"] = (
        ev2_rec.get("status") == "succeeded"
        and diff.status_code == 200
        and dj.get("object") == "eval_diff"
        and dj.get("same_suite") is True
        and dj.get("same_seed") is True
        # tooluse reports don't stamp eval_bank_sha256 — same_bank is the
        # stamp evidence, comparable is the suite+seed-pinning contract
        and dj.get("same_bank") is False
        and dj.get("comparable") is True
        and dj.get("verdict") == "unchanged"
        and dj.get("tasks_fixed") == []
        and dj.get("tasks_regressed") == []
        # identical records: zero discordant pairs, exact sign test p = 1
        and (dj.get("significance") or {}).get("p_value") == 1.0
        and (dj.get("significance") or {}).get("significant_p05") is False
    )
    out["eval_diff_unknown_404"] = (
        eval_app.get(f"/harness/evals/nope/diff/{ev2_id}").status_code == 404
        and eval_app.get(f"/harness/evals/{ev_id}/diff/nope").status_code == 404
    )
    cap_id = eval_app.post(
        "/harness/evals", json={"suite": "capability", "backend": "byok", "seed": 0}
    ).json()["eval_id"]
    _wait_eval(eval_app, cap_id)
    xdiff = eval_app.get(f"/harness/evals/{ev_id}/diff/{cap_id}")
    out["eval_diff_cross_suite_incomparable"] = (
        xdiff.status_code == 200
        and xdiff.json().get("same_suite") is False
        and xdiff.json().get("comparable") is False
        and xdiff.json().get("verdict") == "unknown"
    )
    qc2b_id = qc2.post("/harness/evals", json={"suite": "tooluse", "backend": "byok"}).json()[
        "eval_id"
    ]
    _wait_eval(qc2, qc2b_id)
    out["eval_diff_nonterminal_409"] = (
        qc2.get(f"/harness/evals/{be_id}/diff/{qc2b_id}").status_code == 409
        and qc2.get(f"/harness/evals/{be_id}/diff/{qc2b_id}").json().get("code")
        == "eval_not_terminal"
    )

    # /v1/evals — the OpenAI Evals-shaped spec/run surface over the same
    # store. Specs declare suite knobs in item_schema; runs bind a model
    # (backend link or ft: name); everything cross-links to /harness/evals.
    spec1 = eval_app.post(
        _PATH_EVALS,
        json={
            "name": "tooluse-baseline",
            "data_source_config": {
                "type": "custom",
                "item_schema": {"suite": "tooluse", "seed": 0},
            },
            "testing_criteria": [{"name": "all-pass"}],
            "metadata": {"lane": "audit"},
        },
    )
    spec1_j = spec1.json() if spec1.status_code == 201 else {}
    spec1_id = spec1_j.get("id", "")
    out["evalspec_create_201"] = (
        spec1.status_code == 201
        and spec1_id.startswith("eval_")
        and spec1_j.get("object") == "eval"
        and spec1_j.get("name") == "tooluse-baseline"
        and spec1_j.get("metadata") == {"lane": "audit"}
        and (spec1_j.get("data_source_config") or {}).get("item_schema", {}).get("suite")
        == "tooluse"
    )
    out["evalspec_bad_suite_422"] = (
        eval_app.post(
            _PATH_EVALS,
            json={
                "name": "x",
                "data_source_config": {"type": "custom", "item_schema": {"suite": "nope"}},
            },
        ).status_code
        == 422
    )
    spec_list = eval_app.get("/v1/evals?limit=10").json()
    out["evalspec_list"] = spec_list.get("object") == "list" and any(
        s.get("id") == spec1_id for s in spec_list.get("data", [])
    )
    out["evalspec_get"] = (
        eval_app.get(f"/v1/evals/{spec1_id}").json().get("id") == spec1_id
        and eval_app.get("/v1/evals/eval_nope").status_code == 404
    )
    spec_upd = eval_app.post(
        f"/v1/evals/{spec1_id}",
        json={"name": "tooluse-baseline-v2", "metadata": {"lane": "audit", "v": "2"}},
    )
    out["evalspec_update"] = (
        spec_upd.status_code == 200
        and spec_upd.json().get("name") == "tooluse-baseline-v2"
        and spec_upd.json().get("metadata", {}).get("v") == "2"
        and eval_app.post(f"/v1/evals/{spec1_id}", json={}).status_code == 422
    )
    # A spec whose schema validator rejects a combination fail-closes 422.
    out["evalspec_schema_validated"] = (
        eval_app.post(
            _PATH_EVALS,
            json={
                "name": "bad-chain",
                "data_source_config": {
                    "type": "custom",
                    "item_schema": {
                        "suite": "tooluse",
                        "backend": "byok",
                        "fallbacks": ["local_fx1", "local_fx1"],
                    },
                },
            },
        ).status_code
        == 422
    )

    run1 = eval_app.post(
        f"/v1/evals/{spec1_id}/runs",
        json={"model": "byok"},
        headers={"Idempotency-Key": "spec-run-1"},
    )
    run1_j = run1.json() if run1.status_code == 201 else {}
    run1_id = run1_j.get("id", "")
    out["evalrun_create_201"] = (
        run1.status_code == 201
        and run1_id.startswith("evalrun_")
        and run1_j.get("object") == "eval.run"
        and run1_j.get("eval_id") == spec1_id
        and run1_j.get("model") == "byok"
        and run1.headers.get("location") == f"/v1/evals/{spec1_id}/runs/{run1_id[8:]}"
    )
    # Same key, same body → replay; same key, different body → 409.
    run1_replay = eval_app.post(
        f"/v1/evals/{spec1_id}/runs",
        json={"model": "byok"},
        headers={"Idempotency-Key": "spec-run-1"},
    )
    out["evalrun_idem_replay"] = (
        run1_replay.status_code == 201
        and run1_replay.json().get("id") == run1_id
        and eval_app.post(
            f"/v1/evals/{spec1_id}/runs",
            json={"model": "local_fx1"},
            headers={"Idempotency-Key": "spec-run-1"},
        ).status_code
        == 409
    )
    ev_run_rec = _wait_eval(eval_app, run1_j.get("eval_run_id") or run1_id[8:] or "")
    run1_term = eval_app.get(f"/v1/evals/{spec1_id}/runs/{run1_id}").json()
    out["evalrun_terminal_completed"] = (
        ev_run_rec.get("status") == "succeeded"
        and run1_term.get("status") == "completed"
        and (run1_term.get("result_counts") or {}).get("total", 0) > 0
        and run1_term.get("receipt_url") == f"/harness/evals/{run1_id[8:]}/receipt"
    )
    out["evalrun_list"] = any(
        r.get("id") == run1_id
        for r in eval_app.get(f"/v1/evals/{spec1_id}/runs").json().get("data", [])
    )
    items = eval_app.get(f"/v1/evals/{spec1_id}/runs/{run1_id}/output_items").json()
    out["evalrun_output_items"] = (
        items.get("object") == "list"
        and len(items.get("data", [])) > 0
        and all(it.get("object") == "eval.run.output_item" for it in items.get("data", []))
        and all(
            any("name" in r and "passed" in r for r in it.get("results", []))
            for it in items.get("data", [])
        )
    )
    out["evalrun_404s"] = (
        eval_app.get("/v1/evals/eval_nope/runs").status_code == 404
        and eval_app.get(f"/v1/evals/{spec1_id}/runs/evalrun_nope").status_code == 404
        and eval_app.post(f"/v1/evals/{spec1_id}/runs/evalrun_nope/cancel").status_code == 404
    )

    # A second spec namespaces idempotency and list/get — cross-spec run
    # ids must not leak.
    spec2_id = (
        eval_app.post(
            _PATH_EVALS,
            json={
                "name": "retrieval",
                "data_source_config": {
                    "type": "custom",
                    "item_schema": {"suite": "retrieval", "seed": 1},
                },
            },
        )
        .json()
        .get("id", "")
    )
    out["evalrun_cross_spec_404"] = (
        eval_app.get(f"/v1/evals/{spec2_id}/runs/{run1_id}").status_code == 404
        and eval_app.get(f"/v1/evals/{spec2_id}/runs/{run1_id}/output_items").status_code == 404
    )
    spec2_reuse = eval_app.post(
        f"/v1/evals/{spec2_id}/runs",
        json={"model": "byok"},
        headers={"Idempotency-Key": "spec-run-1"},
    )
    out["evalrun_idem_per_spec"] = (
        spec2_reuse.status_code == 201 and spec2_reuse.json().get("id") != run1_id
    )
    eval_run_delete = eval_app.delete(f"/v1/evals/{spec1_id}/runs/{run1_id}")
    out["evalrun_delete_terminal_200"] = (
        eval_run_delete.status_code == 200
        and eval_run_delete.json().get("object") == "eval.run.deleted"
        and eval_app.get(f"/v1/evals/{spec1_id}/runs/{run1_id}").status_code == 404
    )
    spec_del = eval_app.delete(f"/v1/evals/{spec1_id}")
    out["evalspec_delete"] = (
        spec_del.status_code == 200
        and spec_del.json().get("object") == "eval.deleted"
        and spec_del.json().get("deleted") is True
        and eval_app.get(f"/v1/evals/{spec1_id}").status_code == 404
        # The surviving spec's runs still resolve through the run surface.
        and eval_app.get(f"/v1/evals/{spec2_id}/runs/{spec2_reuse.json().get('id')}").status_code
        == 200
    )

    # Cancel on a running eval is 409 — suite runners have no kill handle.
    # The gate event is set before wait expires so the suite completes.
    gate_ev = _threading.Event()

    class _BlockEvalBackend:
        def __init__(self) -> None:
            self._model = "fake-0"

        def complete(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> str:
            gate_ev.wait(timeout=30)
            return "clean:yes"

    gate_app = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _BlockEvalBackend()))
    g1 = gate_app.post("/harness/evals", json={"suite": "tooluse", "backend": "byok"}).json()[
        "eval_id"
    ]
    for _i in range(200):
        if gate_app.get(f"/harness/evals/{g1}").json()["status"] == "running":
            break
        time.sleep(0.02)
    out["eval_cancel_running_409"] = gate_app.delete(f"/harness/evals/{g1}").status_code == 409
    gate_ev.set()
    _wait_eval(gate_app, g1)

    # Judge suites meter the grader under its own key.
    class _JudgeEvalBackend:
        def __init__(self) -> None:
            self._model = "judge-0"

        def complete(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> str:
            return "A"

    def _judge_resolver(name: str, *a: Any, **k: Any) -> Any:
        return _JudgeEvalBackend() if name == "hosted_k3" else eval_backend

    judge_app = _TC2(api_mod.create_app(backend_resolver=_judge_resolver))
    jsub = judge_app.post(
        "/harness/evals",
        json={
            "suite": "capability",
            "backend": "byok",
            "seed": 0,
            "judge_backend": "hosted_k3",
        },
    )
    jrec = _wait_eval(judge_app, jsub.json()["eval_id"])
    pm_judge = judge_app.get("/metrics").json()
    out["eval_judge_metered"] = (
        jrec.get("status") == "succeeded"
        and pm_judge["complete"].get("eval:capability:judge:hosted_k3", {}).get("latency_count", 0)
        > 0
    )
    out["eval_capabilities_lists_suites"] = (
        "tooluse" in eval_app.get("/harness/capabilities").json()["eval_suites"]
    )

    # --- eval callbacks: the job webhook contract on the eval surface -----
    # A terminal eval POSTs its record to callback_url (HMAC-signed when a
    # secret is given); delivery is best-effort and lands on the record.
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: PLC0415

    _ev_hits: list[dict[str, Any]] = []
    _ev_raw: list[bytes] = []
    _ev_hdrs: list[dict[str, str]] = []

    class _EvalHook(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 — http.server handler name
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n)
            _ev_raw.append(raw)
            _ev_hdrs.append(dict(self.headers.items()))
            _ev_hits.append(_json.loads(raw))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            pass

    ev_srv = ThreadingHTTPServer(("127.0.0.1", 0), _EvalHook)
    ev_thread = _threading.Thread(target=ev_srv.serve_forever, daemon=True)
    ev_thread.start()
    ev_cb_url = f"http://127.0.0.1:{ev_srv.server_address[1]}/evalhook"
    import socket as _socket2  # noqa: PLC0415

    _dsock = _socket2.socket()
    _dsock.bind(("127.0.0.1", 0))
    _ev_dead_port = _dsock.getsockname()[1]
    _dsock.close()
    # loopback callback_url needs the private-networks opt-in, read per
    # request (saved/restored verbatim).
    saved_ev_wh = os.environ.get("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS")
    os.environ["FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"] = "1"
    try:
        sub_cb = eval_app.post(
            "/harness/evals",
            json={
                "suite": "tooluse",
                "backend": "byok",
                "callback_url": ev_cb_url,
                "callback_secret": "ev-whsec",
            },
        )
        ev_cb_id = sub_cb.json()["eval_id"]
        deadline = time.monotonic() + 10.0
        st_ev: dict[str, Any] = {}
        while time.monotonic() < deadline:
            st_ev = eval_app.get(f"/harness/evals/{ev_cb_id}").json()
            if st_ev["status"] == "succeeded" and st_ev.get("callback_status"):
                break
            time.sleep(0.05)
        out["eval_callback_delivered"] = (
            sub_cb.status_code == 202
            and st_ev.get("callback_status") == "delivered"
            and st_ev.get("callback_url") == ev_cb_url
            and len(_ev_hits) >= 1
            and _ev_hits[-1]["eval_id"] == ev_cb_id
            and _ev_hits[-1]["status"] == "succeeded"
            and _ev_hits[-1]["suite"] == "tooluse"
        )
        from fx1.serve.webhooks import verify_webhook  # noqa: PLC0415

        sig_h = _ev_hdrs[-1]
        out["eval_callback_signed_verifies"] = verify_webhook(
            "ev-whsec",
            sig_h.get("X-Fx1-Webhook-Timestamp"),
            sig_h.get("X-Fx1-Webhook-Signature"),
            _ev_raw[-1],
        )
        out["eval_callback_secret_not_echoed"] = (
            "callback_secret" not in st_ev and b"ev-whsec" not in _ev_raw[-1]
        )
        # dead endpoint -> recorded failure on the record, eval unaffected
        sub_dead = eval_app.post(
            "/harness/evals",
            json={
                "suite": "tooluse",
                "backend": "byok",
                "callback_url": f"http://127.0.0.1:{_ev_dead_port}/hook",
            },
        )
        ev_dead_id = sub_dead.json()["eval_id"]
        deadline = time.monotonic() + 20.0
        st_dead: dict[str, Any] = {}
        while time.monotonic() < deadline:
            st_dead = eval_app.get(f"/harness/evals/{ev_dead_id}").json()
            if (
                st_dead["status"] == "succeeded"
                and st_dead.get("callback_status")
                and st_dead.get("callback_attempts") == 3
            ):
                break
            time.sleep(0.05)
        out["eval_callback_dead_recorded"] = (
            st_dead["status"] == "succeeded"
            and st_dead.get("callback_status") == "failed"
            and st_dead.get("callback_attempts") == 3
            and bool(st_dead.get("callback_error"))
        )
        # cancelling a queued eval is a terminal transition — it fires too
        hold_ev2 = _threading.Event()
        capp = api_mod.create_app(backend_resolver=lambda *a, **k: eval_backend, max_inflight=1)
        qc3 = _TC2(capp)
        capp.state.jobs_executor.submit(lambda: hold_ev2.wait(timeout=20))
        q_cb = qc3.post(
            "/harness/evals",
            json={"suite": "tooluse", "backend": "byok", "callback_url": ev_cb_url},
        ).json()["eval_id"]
        cx = qc3.delete(f"/harness/evals/{q_cb}")
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and (not _ev_hits or _ev_hits[-1].get("eval_id") != q_cb):
            time.sleep(0.05)
        out["eval_callback_fires_on_cancel"] = (
            cx.status_code == 200
            and cx.json()["status"] == "cancelled"
            and cx.json().get("callback_status") == "delivered"
            and _ev_hits[-1]["eval_id"] == q_cb
            and _ev_hits[-1]["status"] == "cancelled"
        )
        hold_ev2.set()
        out["eval_callback_bad_url_422"] = (
            eval_app.post(
                "/harness/evals",
                json={
                    "suite": "tooluse",
                    "backend": "byok",
                    "callback_url": "ftp://x/h",
                },
            ).status_code
            == 422
        )
        out["eval_callback_secret_no_url_422"] = (
            eval_app.post(
                "/harness/evals",
                json={
                    "suite": "tooluse",
                    "backend": "byok",
                    "callback_secret": "s",
                },
            ).status_code
            == 422
        )
    finally:
        ev_srv.shutdown()
        ev_srv.server_close()
        if saved_ev_wh is None:
            os.environ.pop("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS", None)
        else:
            os.environ["FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"] = saved_ev_wh

    # --- OpenAI-compatible ingress ------------------------------------------
    # POST /v1/chat/completions is a drop-in OpenAI surface over the gated
    # complete chain — probes pin the envelope, plumbing-through (metering,
    # completion log, gate), header-based backend/BYOK selection, and the
    # OpenAI error envelope on every failure class.
    import json as _json3  # noqa: PLC0415

    class _OiBackend:
        def __init__(self) -> None:
            self._model = "fake-0"

        def complete(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> str:
            return f"clean:{messages[-1]['content']}"

    class _OiDirty:
        def complete(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> str:
            return "The strategy achieved a sharpe of 2.1 on the tape."

    class _OiUsage(_OiBackend):
        """Same echo backend with a usage channel + param capture — the
        wire probe for declared params reaching the provider verbatim."""

        last_usage = {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
        seen: SamplingParams | None = None

        def complete(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> str:
            self.seen = sampling
            return super().complete(messages, sampling=sampling)

    oi_clean = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _OiBackend()))
    r = oi_clean.get("/v1/models")
    out["openai_models_200"] = (
        r.status_code == 200
        and r.json().get("object") == "list"
        and {m["id"] for m in r.json()["data"]} == {"fx1", "hosted_k3", "local_fx1", "byok"}
    )

    r = oi_clean.post(
        "/v1/chat/completions",
        json={"model": "fx1", "messages": [{"role": "user", "content": "ping"}]},
    )
    oi = r.json()
    out["openai_chat_200_envelope"] = (
        r.status_code == 200
        and oi.get("object") == "chat.completion"
        and oi.get("id", "").startswith("chatcmpl-")
        and oi["choices"][0]["message"]["role"] == "assistant"
        and oi["choices"][0]["message"]["content"] == _PING_MSG
        and oi["choices"][0]["finish_reason"] == "stop"
        and oi.get("system_fingerprint") == "hosted_k3"
        and oi.get("model") == "fake-0"
    )
    oc_cid = r.headers.get("X-Fx1-Completion-Id", "")
    out["openai_completion_id_header"] = bool(oc_cid)
    if oc_cid:
        rl = oi_clean.get(f"/harness/completions/{oc_cid}")
        out["openai_completion_logged"] = (
            rl.status_code == 200 and rl.json().get("backend") == "hosted_k3"
        )
        rr = oi_clean.get(f"/harness/completions/{oc_cid}/receipt")
        out["openai_completion_receipt_verifies"] = (
            rr.status_code == 200 and _vrp(rr.json())["valid"] is True
        )

    # metering: an OpenAI call is a complete call
    pre = oi_clean.get("/metrics").json()["complete"].get("hosted_k3", {})
    r = oi_clean.post(
        "/v1/chat/completions",
        json={"model": "fx1", "messages": [{"role": "user", "content": "m2"}]},
    )
    post = oi_clean.get("/metrics").json()["complete"].get("hosted_k3", {})
    out["openai_metered"] = r.status_code == 200 and post.get("ok", 0) - pre.get("ok", 0) == 1

    # SSE: stream:true → chat.completion.chunk frames, gated text, [DONE]
    r = oi_clean.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "stream me"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    sframes = [ln for ln in r.text.split("\n\n") if ln.strip()]
    schunks = [
        _json3.loads(dln[len("data: ") :])
        for ln in sframes
        for dln in ln.splitlines()
        if dln.startswith("data: ") and dln[len("data: ") :].strip() != "[DONE]"
    ]
    deltas = [
        c["choices"][0]["delta"].get("content", "")
        for c in schunks
        if c.get("choices") and c["choices"][0]["delta"].get("content")
    ]
    out["openai_stream_frames"] = (
        r.status_code == 200
        and r.headers.get("content-type", "").startswith("text/event-stream")
        and all(c.get("object") == "chat.completion.chunk" for c in schunks)
        and schunks[0]["choices"][0]["delta"].get("role") == "assistant"
        and "".join(deltas) == "clean:stream me"
        and schunks[-1]["choices"] == []
        and "usage" in schunks[-1]
        and schunks[-2]["choices"][0]["finish_reason"] == "stop"
        and sframes[-1].strip().endswith("data: [DONE]")
    )

    # backend selection: X-Fx1-Backend header names the link
    r = oi_clean.post(
        "/v1/chat/completions",
        headers={"X-Fx1-Backend": "byok"},
        json={"model": "fx1", "messages": [{"role": "user", "content": "h"}]},
    )
    out["openai_backend_header"] = (
        r.status_code == 200 and r.json().get("system_fingerprint") == "byok"
    )
    # model naming a backend routes there (BYOK headers supply the creds)
    r = oi_clean.post(
        "/v1/chat/completions",
        headers={
            "X-Fx1-Byok-Base-Url": "https://provider.example/v1",
            "X-Fx1-Byok-Api-Key": "sk-fake",
            "X-Fx1-Byok-Model": "gpt-fake",
        },
        json={
            "model": "byok",
            "messages": [{"role": "user", "content": "h"}],
        },
    )
    out["openai_model_selects_backend"] = (
        r.status_code == 200 and r.json().get("system_fingerprint") == "byok"
    )
    # BYOK headers → byok link; body model is the upstream model
    r = oi_clean.post(
        "/v1/chat/completions",
        headers={
            "X-Fx1-Byok-Base-Url": "https://provider.example/v1",
            "X-Fx1-Byok-Api-Key": "sk-fake",
            "X-Fx1-Byok-Model": "gpt-fake",
        },
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "h"}]},
    )
    out["openai_byok_headers_route_byok"] = (
        r.status_code == 200 and r.json().get("system_fingerprint") == "byok"
    )
    r = oi_clean.post(
        "/v1/chat/completions",
        headers={"X-Fx1-Byok-Base-Url": "https://provider.example/v1"},
        json={"model": "x", "messages": [{"role": "user", "content": "h"}]},
    )
    out["openai_byok_missing_key_400"] = (
        r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
    )

    # fail-closed surface — every rejection in the OpenAI envelope
    r = oi_clean.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "h"}],
            "tools": [{"type": "function"}],
        },
    )
    out["openai_tool_shape_422"] = (
        r.status_code == 422 and r.json()["error"]["type"] == "invalid_request_error"
    )

    # ---- lane 81: the tools channel ----
    # A tool-capable link receives the spec verbatim and the envelope
    # carries the machine call — finish_reason 'tool_calls', content
    # null (OpenAI's encoding for a call-only turn). A link without
    # complete_with_tools answers 501, never a silently dropped spec.
    from fx1.serve.backends import ToolCompletion as _ToolCompletion  # noqa: PLC0415

    class _OiToolBackend:
        """Tool-capable stub: records the forwarded spec, answers calls.
        Answers a canned ``logprobs`` payload when the request asks for
        one — the wire's verbatim echo is what gets probed. ``n_calls``
        sets how many ``function_call`` entries one turn emits (one by
        default — the multi-call variant probes ``max_tool_calls``)."""

        def __init__(self, n_calls: int = 1) -> None:
            self._model = "tool-0"
            self.n_calls = n_calls
            self.calls = 0
            self.seen_tools: list[dict[str, Any]] | None = None
            self.seen_choice: Any = None
            self.seen_parallel: bool | None = None
            self.seen_messages: list[dict[str, Any]] | None = None
            self.seen_logprobs: bool | None = None
            self.seen_top_logprobs: int | None = None

        def complete(
            self,
            messages: list[dict[str, Any]],
            *,
            sampling: SamplingParams | None = None,
        ) -> str:
            return "clean:text"

        def complete_with_tools(
            self,
            messages: list[dict[str, Any]],
            *,
            sampling: SamplingParams | None = None,
            tools: list[dict[str, Any]] | None = None,
            tool_choice: Any = None,
            parallel_tool_calls: bool | None = None,
            logprobs: bool | None = None,
            top_logprobs: int | None = None,
        ) -> _ToolCompletion:
            self.calls += 1
            self.seen_tools = tools
            self.seen_choice = tool_choice
            self.seen_parallel = parallel_tool_calls
            self.seen_messages = [dict(m) for m in messages]
            self.seen_logprobs = logprobs
            self.seen_top_logprobs = top_logprobs
            lp: dict[str, Any] | None = None
            if logprobs:
                lp = {
                    "content": [
                        {
                            "token": "x",
                            "logprob": -0.5,
                            "bytes": [120],
                            "top_logprobs": (
                                [{"token": "x", "logprob": -0.5, "bytes": [120]}]
                                if top_logprobs is not None
                                else []
                            ),
                        }
                    ]
                }
            return _ToolCompletion(
                content=None,
                tool_calls=tuple(
                    {
                        "id": f"call_{k}",
                        "type": "function",
                        "function": {"name": "calc", "arguments": '{"x": 1}'},
                    }
                    for k in range(self.n_calls)
                ),
                finish_reason="tool_calls",
                logprobs=lp,
            )

    class _OiLpBackend:
        """Structured-channel stub answering a text turn with scores —
        no tool_calls, so the message part exists to carry logprobs."""

        def __init__(self) -> None:
            self._model = "lp-0"
            self.seen_logprobs: bool | None = None
            self.seen_top_logprobs: int | None = None

        def complete(
            self,
            messages: list[dict[str, Any]],
            *,
            sampling: SamplingParams | None = None,
        ) -> str:
            return f"clean:{messages[-1]['content']}"

        def complete_with_tools(
            self,
            messages: list[dict[str, Any]],
            *,
            sampling: SamplingParams | None = None,
            tools: list[dict[str, Any]] | None = None,
            tool_choice: Any = None,
            parallel_tool_calls: bool | None = None,
            logprobs: bool | None = None,
            top_logprobs: int | None = None,
        ) -> _ToolCompletion:
            self.seen_logprobs = logprobs
            self.seen_top_logprobs = top_logprobs
            lp: dict[str, Any] | None = None
            if logprobs:
                lp = {
                    "content": [
                        {
                            "token": "x",
                            "logprob": -0.5,
                            "bytes": [120],
                            "top_logprobs": (
                                [{"token": "x", "logprob": -0.5, "bytes": [120]}]
                                if top_logprobs is not None
                                else []
                            ),
                        }
                    ]
                }
            return _ToolCompletion(
                content=self.complete(messages),
                tool_calls=None,
                finish_reason="stop",
                logprobs=lp,
            )

    from fx1.serve.backends import EmbeddingResult as _EmbeddingResult  # noqa: PLC0415

    class _OiEmbedBackend:
        """Embedding-capable stub: records the forwarded fields verbatim
        and answers a canned ``data[]`` — one item per string in a list
        input, one item for a bare string or token array. The ``model``
        echo is the request model suffixed (the provider's own model id),
        which is what the envelope reports."""

        def __init__(self) -> None:
            self._model = "emb-chat-pin"  # the link's chat pin — not the wire model
            self.seen_model: str | None = None
            self.seen_input: Any = None
            self.seen_format: str | None = None
            self.seen_dimensions: int | None = None
            self.seen_user: str | None = None

        def complete(
            self,
            messages: list[dict[str, Any]],
            *,
            sampling: SamplingParams | None = None,
        ) -> str:
            return f"clean:{messages[-1]['content']}"

        def embeddings(
            self,
            input: Any,  # noqa: A002 — the wire field's own name
            *,
            model: str,
            encoding_format: str | None = None,
            dimensions: int | None = None,
            user: str | None = None,
        ) -> _EmbeddingResult:
            self.seen_model = model
            self.seen_input = input
            self.seen_format = encoding_format
            self.seen_dimensions = dimensions
            self.seen_user = user
            # OpenAI arity: a list of strings or a list of token arrays is
            # N inputs; a bare string or a flat token array is ONE.
            n = (
                len(input)
                if isinstance(input, list) and input and isinstance(input[0], (str, list))
                else 1
            )
            if encoding_format == "base64":
                data = tuple(
                    {"object": "embedding", "index": i, "embedding": "AAE="} for i in range(n)
                )
            else:
                data = tuple(
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": [0.1 * (i + 1), 0.2],
                    }
                    for i in range(n)
                )
            return _EmbeddingResult(
                data=data,
                model=f"{model}-v1",
                usage={"prompt_tokens": 4, "total_tokens": 4},
            )

    class _OiEmbedBadBackend:
        """Embedding stub whose provider frame was malformed — the
        helper's fail-closed raise arrives as RuntimeError → 502."""

        def __init__(self) -> None:
            self._model = "emb-bad"

        def embeddings(
            self,
            input: Any,  # noqa: A002
            *,
            model: str,
            encoding_format: str | None = None,
            dimensions: int | None = None,
            user: str | None = None,
        ) -> _EmbeddingResult:
            raise RuntimeError("malformed embeddings payload: data is dict, not list")

    oi_tool = _OiToolBackend()
    oi_tools = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: oi_tool))
    # three calls a turn — probes ``max_tool_calls`` truncation
    oi_tool3 = _OiToolBackend(n_calls=3)
    oi_tool3_app = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: oi_tool3))
    oi_lp_b = _OiLpBackend()
    oi_lp = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: oi_lp_b))
    oi_emb_b = _OiEmbedBackend()
    oi_emb = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: oi_emb_b))
    oi_emb_bad = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _OiEmbedBadBackend()))
    tool_spec = {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "arithmetic",
            "parameters": {"type": "object", "properties": {"x": {"type": "number"}}},
        },
    }
    r = oi_tools.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "calc one"}],
            "tools": [tool_spec],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        },
    )
    tmsg = (r.json().get("choices") or [{}])[0].get("message", {})
    out["openai_tools_call_envelope"] = (
        r.status_code == 200
        and tmsg.get("content") is None
        and (tmsg.get("tool_calls") or [{}])[0].get("function", {}).get("name") == "calc"
        and r.json()["choices"][0].get("finish_reason") == "tool_calls"
    )
    out["openai_tools_forwarded_verbatim"] = (
        oi_tool.seen_tools == [tool_spec]
        and oi_tool.seen_choice == "auto"
        and oi_tool.seen_parallel is False
    )
    # agent history passes through verbatim — assistant tool_calls and a
    # role:'tool' output reach the backend's message list unedited
    tool_hist = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_9",
                    "type": "function",
                    "function": {"name": "calc", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "2", "tool_call_id": "call_9"},
        {"role": "user", "content": "and?"},
    ]
    r = oi_tools.post("/v1/chat/completions", json={"model": "fx1", "messages": tool_hist})
    out["openai_tool_history_passed_verbatim"] = (
        r.status_code == 200 and oi_tool.seen_messages == tool_hist
    )
    out["openai_tool_needs_channel_501"] = (
        oi_clean.post(
            "/v1/chat/completions",
            json={
                "model": "fx1",
                "messages": [{"role": "user", "content": "h"}],
                "tools": [tool_spec],
            },
        ).status_code
        == 501
    )
    out["openai_tool_role_needs_call_id_400"] = (
        oi_tools.post(
            "/v1/chat/completions",
            json={"model": "fx1", "messages": [{"role": "tool", "content": "2"}]},
        ).status_code
        == 400
    )
    out["openai_tool_calls_wrong_role_400"] = (
        oi_tools.post(
            "/v1/chat/completions",
            json={
                "model": "fx1",
                "messages": [
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
                ],
            },
        ).status_code
        == 400
    )
    out["openai_tool_choice_needs_tools_422"] = (
        oi_tools.post(
            "/v1/chat/completions",
            json={
                "model": "fx1",
                "messages": [{"role": "user", "content": "h"}],
                "tool_choice": "auto",
            },
        ).status_code
        == 422
    )
    out["openai_tools_over_128_422"] = (
        oi_tools.post(
            "/v1/chat/completions",
            json={
                "model": "fx1",
                "messages": [{"role": "user", "content": "h"}],
                "tools": [tool_spec] * 129,
            },
        ).status_code
        == 422
    )
    # legacy function_calling fields stay refused — tools is the only
    # function-calling grammar the surface honors
    out["openai_functions_still_refused_422"] = all(
        oi_tools.post(
            "/v1/chat/completions",
            json={
                "model": "fx1",
                "messages": [{"role": "user", "content": "h"}],
                field: value,
            },
        ).status_code
        == 422
        for field, value in (
            ("functions", [{"name": "f"}]),
            ("function_call", {"name": "f"}),
        )
    )
    # keyed replay re-emits the same tool_calls envelope byte-identically
    tk = {"Idempotency-Key": "tool-idem-81"}
    tbody = {
        "model": "fx1",
        "messages": [{"role": "user", "content": "calc"}],
        "tools": [tool_spec],
    }
    tr1 = oi_tools.post("/v1/chat/completions", json=tbody, headers=tk)
    tr2 = oi_tools.post("/v1/chat/completions", json=tbody, headers=tk)
    out["openai_tool_idem_replay"] = (
        tr1.status_code == 200
        and tr2.status_code == 200
        and tr1.json() == tr2.json()
        and tr2.headers.get("X-Fx1-Idempotent-Replay") == "true"
        and (tr2.json()["choices"][0]["message"].get("tool_calls") or [{}])[0].get("id") == "call_0"
    )
    # the seal binds what shipped — the record's output digest covers
    # text + the verbatim call list, so a stripped tool_calls can't
    # pass under the same receipt
    import hashlib as _hl_t  # noqa: PLC0415

    tc_cid = tr1.headers.get("X-Fx1-Completion-Id", "")
    if tc_cid:
        trec = oi_tools.get(f"/harness/completions/{tc_cid}")
        shipped_calls = [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "calc", "arguments": '{"x": 1}'},
            }
        ]
        expect = _hl_t.sha256(
            ("" + "\n" + _json3.dumps(shipped_calls, sort_keys=True)).encode("utf-8")
        ).hexdigest()
        out["openai_tool_digest_binds_calls"] = (
            trec.status_code == 200 and trec.json().get("output_sha256") == expect
        )
    else:
        out["openai_tool_digest_binds_calls"] = False
    # the retrieval index carries tool_calls too — a stored call
    # round-trips the full envelope
    tstore_id = tr1.json().get("id", "")
    rget = oi_tools.get(f"/v1/chat/completions/{tstore_id}")
    out["openai_tool_store_retrieve"] = (
        rget.status_code == 200
        and (rget.json()["choices"][0]["message"].get("tool_calls") or [{}])[0]
        .get("function", {})
        .get("name")
        == "calc"
    )
    rdel = oi_tools.delete(f"/v1/chat/completions/{tstore_id}")
    out["openai_tool_store_delete"] = (
        rdel.status_code == 200
        and oi_tools.get(f"/v1/chat/completions/{tstore_id}").status_code == 404
    )
    # SSE emits the delta.tool_calls frame + the real finish_reason
    r = oi_tools.post(
        "/v1/chat/completions",
        json={**tbody, "stream": True},
    )
    tchunks = [
        _json3.loads(dln[len("data: ") :])
        for dln in r.text.splitlines()
        if dln.startswith("data: ") and dln[len("data: ") :].strip() != "[DONE]"
    ]
    out["openai_tool_stream_delta"] = (
        r.status_code == 200
        and any(
            (c["choices"][0]["delta"].get("tool_calls") or [{}])[0].get("id") == "call_0"
            for c in tchunks
            if c.get("choices")
        )
        and any(
            c["choices"][0].get("finish_reason") == "tool_calls"
            for c in tchunks
            if c.get("choices")
        )
    )
    # the native /harness surface carries the channel too
    r = oi_tools.post(
        "/harness/complete",
        json={
            "backend": "byok",
            "messages": [{"role": "user", "content": "calc"}],
            "tools": [tool_spec],
            "tool_choice": "required",
        },
    )
    out["complete_tools_200"] = (
        r.status_code == 200
        and (r.json().get("tool_calls") or [{}])[0].get("function", {}).get("name") == "calc"
        and r.json().get("finish_reason") == "tool_calls"
    )
    out["complete_tools_no_channel_501"] = (
        oi_clean.post(
            "/harness/complete",
            json={
                "backend": "byok",
                "messages": [{"role": "user", "content": "calc"}],
                "tools": [tool_spec],
            },
        ).status_code
        == 501
    )
    out["complete_stream_tools_501"] = (
        oi_tools.post(
            "/harness/complete/stream",
            json={
                "backend": "byok",
                "messages": [{"role": "user", "content": "calc"}],
                "tools": [tool_spec],
            },
        ).status_code
        == 501
    )
    out["batch_tool_context_422"] = (
        oi_tools.post(
            "/harness/complete/batch",
            json={
                "backend": "byok",
                "batch": [
                    [{"role": "user", "content": "ok"}],
                    [{"role": "tool", "content": "2", "tool_call_id": "c"}],
                ],
            },
        ).status_code
        == 422
    )
    out["capabilities_reports_openai_tools"] = (
        oi_clean.get("/harness/capabilities").json()["features"].get("openai_tools") is True
    )

    # — logprobs channel: request fields reach the provider verbatim and
    # the provider's payload lands verbatim on the envelope —
    lp_req = {
        "model": "fx1",
        "messages": [{"role": "user", "content": "calc"}],
        "logprobs": True,
        "top_logprobs": 3,
    }
    r = oi_tools.post("/v1/chat/completions", json=lp_req)
    out["openai_logprobs_forwarded_verbatim"] = (
        oi_tool.seen_logprobs is True and oi_tool.seen_top_logprobs == 3
    )
    out["openai_logprobs_envelope"] = r.status_code == 200 and (
        r.json()["choices"][0].get("logprobs") or {}
    ).get("content") == [
        {
            "token": "x",
            "logprob": -0.5,
            "bytes": [120],
            "top_logprobs": [{"token": "x", "logprob": -0.5, "bytes": [120]}],
        }
    ]
    # no request → null slot (OpenAI's own null shape); a plain-link
    # request fails closed — scores need the structured channel
    r = oi_tools.post(
        "/v1/chat/completions",
        json={"model": "fx1", "messages": [{"role": "user", "content": "h"}]},
    )
    out["openai_logprobs_absent_null"] = (
        r.status_code == 200 and r.json()["choices"][0].get("logprobs") is None
    )
    out["openai_logprobs_no_channel_501"] = (
        oi_clean.post("/v1/chat/completions", json=lp_req).status_code == 501
    )
    out["openai_toplogprobs_needs_logprobs_422"] = (
        oi_tools.post(
            "/v1/chat/completions",
            json={
                "model": "fx1",
                "messages": [{"role": "user", "content": "h"}],
                "top_logprobs": 2,
            },
        ).status_code
        == 422
    )
    out["openai_toplogprobs_bounds_422"] = all(
        oi_tools.post(
            "/v1/chat/completions",
            json={
                "model": "fx1",
                "messages": [{"role": "user", "content": "h"}],
                "logprobs": True,
                "top_logprobs": bad,
            },
        ).status_code
        == 422
        for bad in (-1, 21)
    )
    # SSE emits one aggregated delta.logprobs frame — the provider's
    # token boundaries don't align with the text re-chunking, so the
    # array ships whole before the finish frame rather than fake-aligned
    r = oi_tools.post("/v1/chat/completions", json={**lp_req, "stream": True})
    lp_chunks = [
        _json3.loads(dln[len("data: ") :])
        for dln in r.text.splitlines()
        if dln.startswith("data: ") and dln[len("data: ") :].strip() != "[DONE]"
    ]
    lp_frames = [
        c for c in lp_chunks if c.get("choices") and c["choices"][0]["delta"].get("logprobs")
    ]
    out["openai_logprobs_stream_frame"] = (
        r.status_code == 200
        and len(lp_frames) == 1
        and lp_frames[0]["choices"][0]["delta"]["logprobs"].get("content")
        == [
            {
                "token": "x",
                "logprob": -0.5,
                "bytes": [120],
                "top_logprobs": [{"token": "x", "logprob": -0.5, "bytes": [120]}],
            }
        ]
    )
    # the native surface carries the channel too — and the digest binds
    # the score payload alongside text
    r = oi_tools.post(
        "/harness/complete",
        json={
            "backend": "byok",
            "messages": [{"role": "user", "content": "calc"}],
            "logprobs": True,
        },
    )
    lp_body = r.json().get("logprobs") or {}
    out["complete_logprobs_200"] = r.status_code == 200 and lp_body.get("content") == [
        {
            "token": "x",
            "logprob": -0.5,
            "bytes": [120],
            "top_logprobs": [],
        }
    ]
    out["complete_logprobs_no_channel_501"] = (
        oi_clean.post(
            "/harness/complete",
            json={
                "backend": "byok",
                "messages": [{"role": "user", "content": "calc"}],
                "logprobs": True,
            },
        ).status_code
        == 501
    )
    lp_cid = r.headers.get("X-Fx1-Completion-Id", "")
    if lp_cid:
        lp_rec = oi_tools.get(f"/harness/completions/{lp_cid}")
        lp_calls = [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "calc", "arguments": '{"x": 1}'},
            }
        ]
        expect_lp = _hl_t.sha256(
            (
                ""
                + "\n"
                + _json3.dumps(lp_calls, sort_keys=True)
                + "\n"
                + _json3.dumps(lp_body, sort_keys=True)
            ).encode("utf-8")
        ).hexdigest()
        out["complete_logprobs_digest_binds"] = (
            lp_rec.status_code == 200 and lp_rec.json().get("output_sha256") == expect_lp
        )
    else:
        out["complete_logprobs_digest_binds"] = False
    # the Responses surface: include/['message.output_text.logprobs'] +
    # top_logprobs gate the channel; output lands on the text part
    # (message stub — a call-only turn has no output_text to carry it)
    r = oi_lp.post(
        "/v1/responses",
        json={
            "model": "fx1",
            "input": "calc",
            "include": ["message.output_text.logprobs"],
            "top_logprobs": 2,
        },
    )
    resp_msg: dict[str, Any] = next(
        (it for it in r.json().get("output", []) if it.get("type") == "message"),
        {},
    )
    resp_lp = (
        resp_msg.get("content", [{}])[0].get("logprobs")
        if isinstance(resp_msg.get("content"), list)
        else None
    )
    out["responses_logprobs_include"] = (
        r.status_code == 200
        and oi_lp_b.seen_logprobs is True
        and oi_lp_b.seen_top_logprobs == 2
        and isinstance(resp_lp, list)
        and resp_lp[0].get("token") == "x"
    )
    out["responses_toplogprobs_needs_include_422"] = (
        oi_tools.post(
            "/v1/responses",
            json={"model": "fx1", "input": "x", "top_logprobs": 1},
        ).status_code
        == 422
    )
    out["responses_include_bad_member_422"] = (
        oi_tools.post(
            "/v1/responses",
            json={"model": "fx1", "input": "x", "include": ["bogus.member"]},
        ).status_code
        == 422
    )
    out["responses_logprobs_no_channel_501"] = (
        oi_clean.post(
            "/v1/responses",
            json={
                "model": "fx1",
                "input": "x",
                "include": ["message.output_text.logprobs"],
            },
        ).status_code
        == 501
    )
    out["capabilities_reports_openai_logprobs"] = (
        oi_clean.get("/harness/capabilities").json()["features"].get("openai_logprobs") is True
    )

    # ---- lane 84: the embeddings channel ----
    # ``POST /v1/embeddings`` forwards model/input/encoding_format/
    # dimensions/user verbatim to an embedding-capable link and echoes the
    # provider's ``data[]``/``model``/``usage`` untouched — a link without
    # the channel answers 501, never fabricated vectors.
    r = oi_emb.post(
        "/v1/embeddings",
        json={"model": "emb-m", "input": "hello"},
    )
    out["openai_embeddings_200"] = (
        r.status_code == 200
        and r.json()["object"] == "list"
        and r.json()["data"] == [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}]
        and r.json()["model"] == "emb-m-v1"
        and r.json()["usage"] == {"prompt_tokens": 4, "total_tokens": 4}
        and r.headers.get("X-Fx1-Completion-Id") is not None
    )
    out["openai_embeddings_forwarded"] = (
        oi_emb_b.seen_model == "emb-m"
        and oi_emb_b.seen_input == "hello"
        and oi_emb_b.seen_format is None
        and oi_emb_b.seen_dimensions is None
        and oi_emb_b.seen_user is None
    )
    oi_emb.post(
        "/v1/embeddings",
        json={
            "model": "emb-x",
            "input": ["a", "b", "c"],
            "dimensions": 2,
            "encoding_format": "float",
            "user": "u-1",
        },
    )
    out["openai_embeddings_list_input"] = (
        oi_emb_b.seen_input == ["a", "b", "c"]
        and oi_emb_b.seen_dimensions == 2
        and oi_emb_b.seen_format == "float"
        and oi_emb_b.seen_user == "u-1"
    )
    em_doc = oi_emb.get(f"/harness/completions/{r.headers.get('x-fx1-completion-id')}/receipt")
    out["receipt_sha_header_embeddings"] = (
        r.headers.get("x-fx1-receipt-sha256") == em_doc.json()["receipt_sha256"]
    )
    r = oi_emb.post(
        "/v1/embeddings",
        json={"model": "emb-tok", "input": [1, 2, 3]},
    )
    out["openai_embeddings_token_input"] = (
        r.status_code == 200 and len(r.json()["data"]) == 1 and oi_emb_b.seen_input == [1, 2, 3]
    )
    r = oi_emb.post(
        "/v1/embeddings",
        json={"model": "emb-b", "input": "x", "encoding_format": "base64"},
    )
    out["openai_embeddings_base64_passthrough"] = (
        r.status_code == 200
        and oi_emb_b.seen_format == "base64"
        and r.json()["data"][0]["embedding"] == "AAE="
    )
    out["openai_embeddings_no_channel_501"] = (
        oi_clean.post(
            "/v1/embeddings",
            json={"model": "emb-m", "input": "x"},
        ).status_code
        == 501
    )
    out["openai_embeddings_empty_422"] = (
        oi_emb.post("/v1/embeddings", json={"model": "emb-m", "input": ""}).status_code == 422
        and oi_emb.post("/v1/embeddings", json={"model": "emb-m", "input": []}).status_code == 422
        and oi_emb.post(
            "/v1/embeddings", json={"model": "emb-m", "input": ["ok", "  "]}
        ).status_code
        == 422
    )
    out["openai_embeddings_mixed_422"] = (
        oi_emb.post("/v1/embeddings", json={"model": "emb-m", "input": ["a", 1]}).status_code == 422
    )
    out["openai_embeddings_bad_encoding_422"] = (
        oi_emb.post(
            "/v1/embeddings",
            json={"model": "emb-m", "input": "x", "encoding_format": "utf8"},
        ).status_code
        == 422
    )
    out["openai_embeddings_provider_failure_502"] = (
        oi_emb_bad.post("/v1/embeddings", json={"model": "emb-m", "input": "x"}).status_code == 502
    )
    # the call lands in the completion log — digests bind the sent input
    # and the verbatim data[] exactly like a completion
    r = oi_emb.post(
        "/v1/embeddings",
        json={"model": "emb-log", "input": "audit"},
    )
    _emb_cid = r.headers.get("X-Fx1-Completion-Id")
    _emb_rec = oi_emb.get(f"/harness/completions/{_emb_cid}") if _emb_cid is not None else None
    _emb_expect_in = _hl_t.sha256(
        _json3.dumps(
            {"model": "emb-log", "input": "audit"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    _emb_expect_out = _hl_t.sha256(
        _json3.dumps(
            [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    out["openai_embeddings_completion_log"] = (
        _emb_rec is not None
        and _emb_rec.status_code == 200
        and _emb_rec.json().get("ok") is True
        and _emb_rec.json().get("prompt_sha256") == _emb_expect_in
        and _emb_rec.json().get("output_sha256") == _emb_expect_out
        and _emb_rec.json().get("model") == "emb-log-v1"
    )
    out["capabilities_reports_openai_embeddings"] = (
        oi_clean.get("/harness/capabilities").json()["features"].get("openai_embeddings") is True
    )
    out["capabilities_reports_score"] = (
        oi_clean.get("/harness/capabilities").json()["features"].get("score") is True
    )
    out["capabilities_reports_openai_moderations"] = (
        oi_clean.get("/harness/capabilities").json()["features"].get("openai_moderations") is True
    )

    # — decode contract: n / stop / penalties / bias / hints / attribution —
    # n fans out into n independent gated calls (each its own gate pass).
    r = oi_clean.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "h"}],
            "n": 2,
        },
    )
    out["openai_n_choices"] = (
        r.status_code == 200
        and [c["index"] for c in r.json()["choices"]] == [0, 1]
        and all(
            c["message"]["content"] == "clean:h" and c["finish_reason"] == "stop"
            for c in r.json()["choices"]
        )
    )
    r = oi_clean.post(
        "/v1/chat/completions",
        json={"model": "fx1", "messages": [{"role": "user", "content": "h"}], "n": 9},
    )
    out["openai_n_over_max_422"] = r.status_code == 422
    r = oi_clean.post(
        "/v1/chat/completions",
        json={"model": "fx1", "messages": [{"role": "user", "content": "h"}], "n": 0},
    )
    out["openai_n_zero_422"] = r.status_code == 422

    # stop: earliest match wins; harness-enforced (stub honors it too)
    r = oi_clean.post(
        "/v1/chat/completions",
        json={"model": "fx1", "messages": [{"role": "user", "content": "x"}], "stop": ":x"},
    )
    out["openai_stop_truncates"] = (
        r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "clean"
    )
    r = oi_clean.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "ab cd"}],
            "stop": ["zz", "b"],
        },
    )
    out["openai_stop_list_earliest"] = (
        r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "clean:a"
    )
    r = oi_clean.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "h"}],
            "stop": ["a", "b", "c", "d", "e"],
        },
    )
    out["openai_stop_over4_422"] = r.status_code == 422
    r = oi_clean.post(
        "/v1/chat/completions",
        json={"model": "fx1", "messages": [{"role": "user", "content": "h"}], "stop": ""},
    )
    out["openai_stop_empty_422"] = r.status_code == 422

    # stop on the stream — deltas end at the cut, [DONE] still ships
    r = oi_clean.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "ab cd"}],
            "stop": "ab",
            "stream": True,
        },
    )
    _stop_frames = [
        _json3.loads(dln[len("data: ") :])
        for dln in r.text.splitlines()
        if dln.startswith("data: ") and dln[len("data: ") :].strip() != "[DONE]"
    ]
    out["openai_stream_stop_cut"] = (
        r.status_code == 200
        and "".join(
            f["choices"][0]["delta"].get("content", "") for f in _stop_frames if f.get("choices")
        )
        == "clean:"
        and r.text.endswith("data: [DONE]\n\n")
    )

    # declared params reach the provider + stamp the audit record;
    # usage on n>1 is the honest sum of n actual calls.
    usage_be = _OiUsage()
    oi_usage = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: usage_be))
    r = oi_usage.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "h"}],
            "n": 3,
            "presence_penalty": 0.5,
            "frequency_penalty": -0.5,
            "logit_bias": {"42": -10},
            "reasoning_effort": "low",
            "service_tier": "flex",
            "verbosity": "high",
            "prompt_cache_key": "pck",
            "prompt_cache_retention": "24h",
            "user": "u-1",
            "metadata": {"team": "risk"},
        },
    )
    seen = usage_be.seen
    out["openai_decode_params_forwarded"] = (
        r.status_code == 200
        and len(r.json()["choices"]) == 3
        and r.json()["usage"]["total_tokens"] == 27
        and seen is not None
        and seen.presence_penalty == 0.5
        and seen.frequency_penalty == -0.5
        and seen.logit_bias == {"42": -10}
        and seen.reasoning_effort == "low"
        and seen.service_tier == "flex"
        and seen.verbosity == "high"
        and seen.prompt_cache_key == "pck"
        and seen.prompt_cache_retention == "24h"
        and seen.user == "u-1"
    )
    cid_dec = r.headers.get("X-Fx1-Completion-Id", "")
    if cid_dec:
        rl = oi_usage.get(f"/harness/completions/{cid_dec}")
        out["openai_user_metadata_recorded"] = (
            rl.status_code == 200
            and rl.json().get("user") == "u-1"
            and rl.json().get("metadata") == {"team": "risk"}
        )
        out["openai_sampling_record_seals_declared"] = (
            rl.status_code == 200
            and rl.json().get("sampling", {}).get("logit_bias") == {"42": -10}
            and rl.json().get("sampling", {}).get("presence_penalty") == 0.5
            and rl.json().get("sampling", {}).get("verbosity") == "high"
            and rl.json().get("sampling", {}).get("prompt_cache_retention") == "24h"
        )
    # enum-valued provider hints fail closed at the model — a value
    # outside the Literal set is a 422, never silently dropped
    out["openai_hint_enum_422"] = all(
        oi_clean.post(
            "/v1/chat/completions",
            json={
                "model": "fx1",
                "messages": [{"role": "user", "content": "h"}],
                field: value,
            },
        ).status_code
        == 422
        for field, value in (
            ("verbosity", "extreme"),
            ("prompt_cache_retention", "forever"),
        )
    )

    # max_completion_tokens alias + disagreeing pair fails closed
    r = oi_usage.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "h"}],
            "max_completion_tokens": 33,
        },
    )
    out["openai_mct_alias"] = (
        r.status_code == 200 and usage_be.seen is not None and usage_be.seen.max_tokens == 33
    )
    r = oi_clean.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "h"}],
            "max_tokens": 5,
            "max_completion_tokens": 9,
        },
    )
    out["openai_mct_conflict_422"] = r.status_code == 422

    # bias/penalty/metadata bounds are range-checked at the model
    r = oi_clean.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "h"}],
            "logit_bias": {"not-a-token": 1},
        },
    )
    out["openai_logit_bias_badkey_422"] = r.status_code == 422
    r = oi_clean.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "h"}],
            "logit_bias": {"1": 200},
        },
    )
    out["openai_logit_bias_range_422"] = r.status_code == 422
    r = oi_clean.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "h"}],
            "presence_penalty": 2.5,
        },
    )
    out["openai_penalty_range_422"] = r.status_code == 422
    r = oi_clean.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "h"}],
            "metadata": {f"k{i}": "v" for i in range(17)},
        },
    )
    out["openai_metadata_over16_422"] = r.status_code == 422

    # newly fail-closed fields that extra="allow" used to drop silently
    out["openai_newly_closed_422"] = all(
        oi_clean.post(
            "/v1/chat/completions",
            json={
                "model": "fx1",
                "messages": [{"role": "user", "content": "h"}],
                field: value,
            },
        ).status_code
        == 422
        for field, value in (
            ("web_search_options", {}),
            ("suffix", "x"),
            ("echo", True),
            ("best_of", 2),
        )
    )
    # `store` is honored, not refused — it admits the call and governs the
    # retrieval index (probed below under retrieve_*)
    out["openai_store_accepted"] = (
        oi_clean.post(
            "/v1/chat/completions",
            json={
                "model": "fx1",
                "messages": [{"role": "user", "content": "h"}],
                "store": False,
            },
        ).status_code
        == 200
    )

    # n>1 streams emit per-index frame groups; keyed replay reproduces them
    r = oi_clean.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "h"}],
            "n": 2,
            "stream": True,
        },
    )
    _nframes = [
        _json3.loads(dln[len("data: ") :])
        for dln in r.text.splitlines()
        if dln.startswith("data: ") and dln[len("data: ") :].strip() != "[DONE]"
    ]
    out["openai_n_stream_indexes"] = (
        r.status_code == 200
        and {f["choices"][0]["index"] for f in _nframes if f.get("choices")} == {0, 1}
        and all(
            f["choices"][0].get("finish_reason") is not None or f["choices"][0]["delta"]
            for f in _nframes
            if f.get("choices")
        )
    )
    _nkey = {"Idempotency-Key": "n-idem-77"}
    r1 = oi_clean.post(
        "/v1/chat/completions",
        json={"model": "fx1", "messages": [{"role": "user", "content": "h"}], "n": 2},
        headers=_nkey,
    )
    r2 = oi_clean.post(
        "/v1/chat/completions",
        json={"model": "fx1", "messages": [{"role": "user", "content": "h"}], "n": 2},
        headers=_nkey,
    )
    out["openai_n_idem_replay"] = (
        r1.status_code == 200
        and r2.status_code == 200
        and r1.json() == r2.json()
        and r2.headers.get("X-Fx1-Idempotent-Replay") == "true"
        and len(r2.json()["choices"]) == 2
    )
    # resume a dropped keyed n=2 stream — the suffix still carries both indexes
    r3 = oi_clean.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "h"}],
            "n": 2,
            "stream": True,
        },
        headers={"Idempotency-Key": "n-resume-77"},
    )
    r4 = oi_clean.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "h"}],
            "n": 2,
            "stream": True,
        },
        headers={"Idempotency-Key": "n-resume-77", "Last-Event-ID": "2"},
    )
    _resumed = [
        _json3.loads(dln[len("data: ") :])
        for dln in r4.text.splitlines()
        if dln.startswith("data: ") and dln[len("data: ") :].strip() != "[DONE]"
    ]
    out["openai_n_resume_suffix"] = (
        r3.status_code == 200
        and r4.status_code == 200
        and len(_resumed) == len(_nframes) - 3
        and {f["choices"][0]["index"] for f in _resumed if f.get("choices")} <= {0, 1}
        and r4.text.endswith("data: [DONE]\n\n")
    )

    # native surface honors the same stop contract (CompleteRequest twin)
    r = oi_clean.post(
        "/harness/complete",
        json={
            "backend": "hosted_k3",
            "messages": [{"role": "user", "content": "x"}],
            "stop": [":x"],
        },
    )
    out["native_stop_truncates"] = (
        r.status_code == 200
        and r.json()["content"] == "clean"
        and r.json()["sampling"]["stop"] == [":x"]
    )
    r = oi_clean.post("/v1/chat/completions", json={"model": "fx1"})
    out["openai_missing_messages_422_openai_shape"] = (
        r.status_code == 422
        and set(r.json()) == {"error"}
        and r.json()["error"]["code"] == "validation"
    )
    r = oi_clean.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": [{"type": "image_url", "url": "x"}]}],
        },
    )
    out["openai_nontext_part_400"] = (
        r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
    )
    r = oi_clean.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [
                {"role": "user", "content": "part "},
                {"role": "user", "content": [{"type": "text", "text": "two"}]},
            ],
        },
    )
    out["openai_content_parts_flattened"] = (
        r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "clean:two"
    )

    # the gate fires over the OpenAI surface
    oi_dirty = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _OiDirty()))
    r = oi_dirty.post(
        "/v1/chat/completions",
        json={"model": "fx1", "messages": [{"role": "user", "content": "h"}]},
    )
    out["openai_gate_502_openai_shape"] = (
        r.status_code == 502
        and set(r.json()) == {"error"}
        and r.json()["error"]["code"] == "honesty_gate"
        and r.json()["error"]["type"] == "server_error"
    )

    # auth: Authorization Bearer works for stock clients; X-API-Key still works
    _saved_key = os.environ.get(_API_KEY_ENV)
    os.environ[_API_KEY_ENV] = "probe-key"
    try:
        keyed = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _OiBackend()))
    finally:
        if _saved_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = _saved_key
    r = keyed.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer probe-key"},
        json={"model": "fx1", "messages": [{"role": "user", "content": "h"}]},
    )
    out["openai_bearer_auth_ok"] = r.status_code == 200
    r = keyed.post(
        "/v1/chat/completions",
        headers={"X-API-Key": "probe-key"},
        json={"model": "fx1", "messages": [{"role": "user", "content": "h"}]},
    )
    out["openai_xapikey_still_ok"] = r.status_code == 200
    r = keyed.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer wrong"},
        json={"model": "fx1", "messages": [{"role": "user", "content": "h"}]},
    )
    out["openai_bad_bearer_401_openai_shape"] = (
        r.status_code == 401 and r.json()["error"]["type"] == "authentication_error"
    )

    # the fx1 extension object carries the chain knobs (incl. body byok)
    r = oi_clean.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "h"}],
            "fx1": {
                "backend": "byok",
                "byok": {
                    "base_url": "https://provider.example/v1",
                    "api_key": "sk-fake",
                    "model": "gpt-fake",
                },
                "fallbacks": ["hosted_k3"],
            },
        },
    )
    out["openai_fx1_extension_backend"] = (
        r.status_code == 200 and r.json().get("system_fingerprint") == "byok"
    )

    # capacity admission applies to the OpenAI surface too
    oi_drained = api_mod.create_app(backend_resolver=lambda *a, **k: _OiBackend(), max_inflight=1)
    oi_drained.state.inflight_slots.acquire()
    try:
        r = _TC2(oi_drained).post(
            "/v1/chat/completions",
            json={"model": "fx1", "messages": [{"role": "user", "content": "h"}]},
        )
    finally:
        oi_drained.state.inflight_slots.release()
    out["openai_over_capacity_openai_shape"] = (
        r.status_code == 503 and r.json()["error"]["code"] == "over_capacity"
    )

    # Idempotency-Key on the OpenAI surface: a keyed retry replays the
    # stored response byte-identically (no re-spend), flagged via
    # X-Fx1-Idempotent-Replay with the original completion id. A key
    # reused under a different body fails closed 409; an over-long key
    # 400s; a refusal is never pinned (the key stays unbound).
    _oi_idem = {"model": "fx1", "messages": [{"role": "user", "content": "idem"}]}
    _oi_key = {"Idempotency-Key": "oi-k1"}
    r1 = oi_clean.post("/v1/chat/completions", json=_oi_idem, headers=_oi_key)
    r2 = oi_clean.post("/v1/chat/completions", json=_oi_idem, headers=_oi_key)
    out["openai_idem_replay_byte_identical"] = (
        r1.status_code == 200
        and r2.status_code == 200
        and r1.content == r2.content
        and r2.headers.get("X-Fx1-Idempotent-Replay") == "true"
        and r1.headers.get("X-Fx1-Completion-Id") == r2.headers.get("X-Fx1-Completion-Id")
    )
    out["openai_idem_first_not_flagged"] = "X-Fx1-Idempotent-Replay" not in r1.headers
    # a replay is a cache hit, not a fresh call — the backend counter
    # and completion log each gain exactly one record for the two posts
    _log_page = oi_clean.get("/harness/completions").json()
    out["openai_idem_replay_no_respend"] = (
        sum(
            1
            for c in _log_page.get("items", [])
            if c.get("completion_id") == r1.headers.get("X-Fx1-Completion-Id")
        )
        == 1
    )
    r = oi_clean.post(
        "/v1/chat/completions",
        json={"model": "fx1", "messages": [{"role": "user", "content": "different"}]},
        headers=_oi_key,
    )
    out["openai_idem_conflict_409"] = (
        r.status_code == 409 and r.json()["error"]["type"] == "invalid_request_error"
    )
    r = oi_clean.post(
        "/v1/chat/completions",
        json=_oi_idem,
        headers={"Idempotency-Key": "x" * 257},
    )
    out["openai_idem_key_bound_400"] = (
        r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
    )
    # keyed stores are per-route — the same key on /harness/complete is
    # an independent dedup slot, not a cross-surface collision
    r = oi_clean.post(
        "/harness/complete",
        json={"backend": "hosted_k3", "messages": [{"role": "user", "content": "idem"}]},
        headers=_oi_key,
    )
    out["openai_idem_scope_isolated"] = r.status_code == 200
    # stream replay regenerates the identical SSE byte sequence
    _oi_idem_s = {**_oi_idem, "stream": True}
    s1 = oi_clean.post(
        "/v1/chat/completions", json=_oi_idem_s, headers={"Idempotency-Key": "oi-k2"}
    )
    s2 = oi_clean.post(
        "/v1/chat/completions", json=_oi_idem_s, headers={"Idempotency-Key": "oi-k2"}
    )
    out["openai_idem_stream_replay"] = (
        s1.status_code == 200
        and s2.status_code == 200
        and s1.content == s2.content
        and s1.content.endswith(b"data: [DONE]\n\n")
        and s2.headers.get("X-Fx1-Idempotent-Replay") == "true"
    )
    # stream=true vs stream=false under one key is a different request
    s3 = oi_clean.post("/v1/chat/completions", json=_oi_idem_s, headers=_oi_key)
    out["openai_idem_stream_mismatch_409"] = s3.status_code == 409
    # a gate refusal never pins the key — a retry re-executes (another
    # 502, not a replayed error, and the key stays free for other bodies)
    d1 = oi_dirty.post(
        "/v1/chat/completions",
        json=_oi_idem,
        headers={"Idempotency-Key": "oi-k3"},
    )
    d2 = oi_dirty.post(
        "/v1/chat/completions",
        json={"model": "fx1", "messages": [{"role": "user", "content": "other"}]},
        headers={"Idempotency-Key": "oi-k3"},
    )
    out["openai_idem_refusal_not_pinned"] = d1.status_code == 502 and d2.status_code == 502

    # Last-Event-ID resume: every SSE frame carries `id:` equal to its
    # sequence index ([DONE] takes the index past the last chunk); a
    # keyed replay with Last-Event-ID=k replays the pinned response
    # minus frames <= k — byte-identical suffix, no re-spend. Fail
    # closed: resume needs stream:true (400), an integer >= 0 (400),
    # the original Idempotency-Key (400 resume_needs_key), and a stored
    # record under it (409 resume_miss — executing fresh and skipping
    # would graft a different completion onto the client's earlier
    # frames).
    _oi_rkey = {"Idempotency-Key": "oi-rs1"}
    v1s = oi_clean.post("/v1/chat/completions", json=_oi_idem_s, headers=_oi_rkey)
    _events = v1s.text.split("\n\n")
    _ids = [ln for ln in v1s.text.splitlines() if ln.startswith("id: ")]
    out["openai_sse_frame_ids"] = (
        v1s.status_code == 200
        and _ids == [f"id: {i}" for i in range(len(_ids))]
        and _events[-2].startswith(f"id: {len(_ids) - 1}\ndata: [DONE]")
    )
    resumed = oi_clean.post(
        "/v1/chat/completions",
        json=_oi_idem_s,
        headers={**_oi_rkey, "Last-Event-ID": "1"},
    )
    out["openai_resume_suffix"] = (
        resumed.status_code == 200
        and resumed.text == "\n\n".join(_events[2:])
        and resumed.headers.get("X-Fx1-Idempotent-Replay") == "true"
    )
    tail = oi_clean.post(
        "/v1/chat/completions",
        json=_oi_idem_s,
        headers={**_oi_rkey, "Last-Event-ID": str(len(_ids) - 1)},
    )
    out["openai_resume_done_only"] = (
        tail.status_code == 200 and tail.text == f"id: {len(_ids) - 1}\ndata: [DONE]\n\n"
    )
    past = oi_clean.post(
        "/v1/chat/completions",
        json=_oi_idem_s,
        headers={**_oi_rkey, "Last-Event-ID": "9999"},
    )
    out["openai_resume_past_end_done"] = past.text.endswith("data: [DONE]\n\n")
    out["openai_resume_needs_key_400"] = (
        oi_clean.post(
            "/v1/chat/completions", json=_oi_idem_s, headers={"Last-Event-ID": "0"}
        ).status_code
        == 400
    )
    miss = oi_clean.post(
        "/v1/chat/completions",
        json=_oi_idem_s,
        headers={"Idempotency-Key": "oi-rs-fresh", "Last-Event-ID": "0"},
    )
    out["openai_resume_unknown_key_409"] = (
        miss.status_code == 409 and miss.json()["error"]["code"] == "resume_miss"
    )
    out["openai_resume_nonstream_400"] = (
        oi_clean.post(
            "/v1/chat/completions",
            json=_oi_idem,
            headers={**_oi_rkey, "Last-Event-ID": "0"},
        ).status_code
        == 400
    )
    out["openai_resume_bad_id_400"] = (
        oi_clean.post(
            "/v1/chat/completions",
            json=_oi_idem_s,
            headers={**_oi_rkey, "Last-Event-ID": "notanint"},
        ).status_code
        == 400
        and oi_clean.post(
            "/v1/chat/completions",
            json=_oi_idem_s,
            headers={**_oi_rkey, "Last-Event-ID": "-1"},
        ).status_code
        == 400
    )

    # GET /v1/models/{id} — OpenAI's models.retrieve: every listed id
    # returns its card; unknown ids fail closed 404 in the OpenAI error
    # shape (code model_not_found), never a fabricated card. Retrieve and
    # list agree — the same `created` stamp on both.
    rm = oi_clean.get("/v1/models/fx1")
    out["openai_retrieve_model_200"] = (
        rm.status_code == 200
        and rm.json().get("object") == "model"
        and rm.json().get("id") == "fx1"
        and isinstance(rm.json().get("created"), int)
    )
    rm_all = [oi_clean.get(f"/v1/models/{m}") for m in ("hosted_k3", "local_fx1", "byok")]
    out["openai_retrieve_all_listed_ids"] = all(
        r.status_code == 200 and r.json().get("id") == m
        for r, m in zip(rm_all, ("hosted_k3", "local_fx1", "byok"), strict=True)
    )
    rn = oi_clean.get("/v1/models/not-a-model")
    out["openai_retrieve_unknown_404_shape"] = (
        rn.status_code == 404
        and rn.json()["error"]["code"] == "model_not_found"
        and rn.json()["error"]["type"] == "invalid_request_error"
    )
    _list = oi_clean.get("/v1/models").json()
    out["openai_retrieve_list_consistent"] = {m["id"] for m in _list["data"]} == {
        "fx1",
        "hosted_k3",
        "local_fx1",
        "byok",
    } and _list["data"][0]["created"] == rm.json()["created"]

    # response_format post-validation — the gate's second pass. The
    # provider can't be constrain-decoded, so the harness validates the
    # returned text: conforming output ships (200); a violation is a
    # provider-side 502 (code format_violation); a malformed schema spec
    # fails closed at request time, before any spend.
    class _OiJson:
        def __init__(self, content: str) -> None:
            self._content = content

        def complete(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> str:
            return self._content

    oi_json = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _OiJson('{"score": 0.9}')))
    _rf_obj = {"type": "json_object"}
    r = oi_json.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "x"}],
            "response_format": _rf_obj,
        },
    )
    out["openai_json_object_pass"] = (
        r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == '{"score": 0.9}'
    )
    oi_broken = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _OiJson("oops")))
    r = oi_broken.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "x"}],
            "response_format": _rf_obj,
        },
    )
    out["openai_json_object_violation_502"] = (
        r.status_code == 502 and r.json()["error"]["code"] == "format_violation"
    )
    # json_object means a JSON object — a bare array is still a violation
    oi_arr = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _OiJson("[1, 2]")))
    r = oi_arr.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "x"}],
            "response_format": _rf_obj,
        },
    )
    out["openai_json_object_array_502"] = r.status_code == 502
    # json_schema: schema-conforming ships, violating output 502s, a
    # malformed schema spec is rejected pre-spend
    _schema = {
        "type": "object",
        "properties": {"score": {"type": "number"}},
        "required": ["score"],
    }
    _rf_schema = {"type": "json_schema", "json_schema": {"name": "s", "schema": _schema}}
    r = oi_json.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "x"}],
            "response_format": _rf_schema,
        },
    )
    out["openai_json_schema_pass"] = r.status_code == 200
    oi_bad_schema_out = _TC2(
        api_mod.create_app(backend_resolver=lambda *a, **k: _OiJson('{"score": "hi"}'))
    )
    r = oi_bad_schema_out.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "x"}],
            "response_format": _rf_schema,
        },
    )
    out["openai_json_schema_violation_502"] = (
        r.status_code == 502 and r.json()["error"]["code"] == "format_violation"
    )
    r = oi_json.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "x"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "s"}},
        },
    )
    out["openai_json_schema_malformed_rejected"] = r.status_code == 422
    r = oi_json.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "x"}],
            "response_format": {"type": "xml"},
        },
    )
    out["openai_response_format_unknown_rejected"] = r.status_code == 422
    # a format violation never occupies the idempotency key — retry
    # re-executes (still 502), the key stays unbound for other bodies
    v1 = oi_broken.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "x"}],
            "response_format": _rf_obj,
        },
        headers={"Idempotency-Key": "oi-fmt"},
    )
    v2 = oi_broken.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "other"}],
            "response_format": _rf_obj,
        },
        headers={"Idempotency-Key": "oi-fmt"},
    )
    out["openai_format_violation_not_pinned"] = v1.status_code == 502 and v2.status_code == 502

    # POST /v1/responses — the Responses API surface over the same gated
    # pipeline. `input` is a string or a message-item list, `instructions`
    # prepends a system turn, `developer` roles map to system,
    # `reasoning.effort` lands on the decode hint, `text.format` lands on
    # the post-validated structured-output channel, and the wire carries
    # the same idempotency + resume + fail-closed contracts as
    # /v1/chat/completions. The `store` flag is refused at false — the
    # audit ledger records every call; there is no retrieval tier for it
    # to govern.
    r = oi_clean.post("/v1/responses", json={"model": "fx1", "input": "hello"})
    out["responses_string_input_200"] = (
        r.status_code == 200
        and r.json()["object"] == "response"
        and r.json()["id"].startswith("resp_")
        and r.json()["status"] == "completed"
        # the model slot echoes the backend's reported model id (same as
        # the chat surface); the requested id is the fallback
        and r.json()["model"] == "fake-0"
        and r.json()["created_at"] > 0
        and isinstance(r.headers.get("X-Fx1-Completion-Id"), str)
    )
    _item = r.json()["output"][0]
    out["responses_output_shape"] = (
        _item["type"] == "message"
        and _item["id"].startswith("msg_")
        and _item["status"] == "completed"
        and _item["role"] == "assistant"
        and _item["content"] == [{"type": "output_text", "text": "clean:hello", "annotations": []}]
    )
    out["responses_defaults_echo"] = (
        r.json()["tools"] == []
        and r.json()["tool_choice"] == "none"
        and r.json()["parallel_tool_calls"] is False
        and r.json()["truncation"] == "disabled"
        and r.json()["store"] is True
        and r.json()["error"] is None
        and r.json()["incomplete_details"] is None
    )
    # usage maps the provider's prompt/completion/total onto input/output/
    # total — never fabricated (None when the backend reports nothing)
    r = oi_clean.post("/v1/responses", json={"model": "fx1", "input": "x"})
    out["responses_no_usage_channel_null"] = r.json()["usage"] is None
    r = oi_usage.post("/v1/responses", json={"model": "fx1", "input": "x"})
    out["responses_usage_shape"] = r.json()["usage"] == {
        "input_tokens": 5,
        "output_tokens": 4,
        "total_tokens": 9,
    }
    # instructions prepends a system turn; a `developer` role maps to
    # system; the last user turn reaches the backend verbatim
    usage_be.seen = None
    r = oi_usage.post(
        "/v1/responses",
        json={
            "model": "fx1",
            "instructions": "be terse",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "dev rules"}],
                },
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": [{"type": "output_text", "text": "ack"}]},
                {"role": "user", "content": "ping"},
            ],
            "reasoning": {"effort": "high"},
            "max_output_tokens": 77,
        },
    )
    out["responses_items_and_instructions"] = (
        r.status_code == 200 and r.json()["output"][0]["content"][0]["text"] == _PING_MSG
    )
    out["responses_reasoning_effort_forwarded"] = (
        usage_be.seen is not None
        and usage_be.seen.reasoning_effort == "high"
        and usage_be.seen.max_tokens == 77
    )
    # request fidelity: the provider-hint knobs that extra=allow used to
    # drop silently — prompt_cache_key/prompt_cache_retention ride the
    # top level, text.verbosity nests under text; all three reach the
    # backend's SamplingParams and echo on the response object
    usage_be.seen = None
    r = oi_usage.post(
        "/v1/responses",
        json={
            "model": "fx1",
            "input": "hints",
            "prompt_cache_key": "pck2",
            "prompt_cache_retention": "24h",
            "text": {"verbosity": "low"},
        },
    )
    out["responses_provider_hints_forwarded"] = (
        r.status_code == 200
        and usage_be.seen is not None
        and usage_be.seen.prompt_cache_key == "pck2"
        and usage_be.seen.prompt_cache_retention == "24h"
        and usage_be.seen.verbosity == "low"
        and r.json().get("prompt_cache_key") == "pck2"
        and r.json().get("prompt_cache_retention") == "24h"
        and r.json().get("text", {}).get("verbosity") == "low"
    )
    out["responses_hint_enum_422"] = all(
        oi_clean.post(
            "/v1/responses",
            json={"model": "fx1", "input": "x", **bad},
        ).status_code
        == 422
        for bad in (
            {"prompt_cache_retention": "forever"},
            {"text": {"verbosity": "extreme"}},
            {"text": {"verbosity": 3}},
        )
    )
    # the shorthand `{role, content: "..."}` item and multi-part input_text
    # lists join before reaching the model
    r = oi_clean.post(
        "/v1/responses",
        json={
            "model": "fx1",
            "input": [
                {"role": "user", "content": "a"},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "b"},
                        {"type": "input_text", "text": "c"},
                    ],
                },
            ],
        },
    )
    out["responses_shorthand_and_parts_join"] = (
        r.status_code == 200 and r.json()["output"][0]["content"][0]["text"] == "clean:bc"
    )
    # fail closed: the fields the pipeline can't honor never reach the
    # model — truncation/include, a
    # refused item type, an unknown item type, an empty input.
    # tools/tool_choice/parallel_tool_calls are honored (the lane-82 tool
    # channel probes below); `store` is honored too (retrieval below),
    # ``previous_response_id`` is honored — the stateful chain surface
    # probed below — and ``background`` is honored (the async lifecycle
    # probes below).
    out["responses_unsupported_refused"] = all(
        oi_clean.post("/v1/responses", json={"model": "fx1", "input": "x", k: v}).status_code == 422
        for k, v in (
            ("truncation", "auto"),
            ("include", ["output_text"]),
        )
    )
    # refused item types fail at translation — a 400 invalid_request_error
    # in the OpenAI shape, not a pydantic 422 (function_call /
    # function_call_output are honored — they carry a tool history)
    out["responses_item_types_refused"] = all(
        oi_clean.post(
            "/v1/responses",
            json={"model": "fx1", "input": [{"type": t, "role": "user", "content": "x"}]},
        ).status_code
        == 400
        for t in ("item_reference", "reasoning", "bogus")
    )
    out["responses_empty_input_422"] = (
        oi_clean.post("/v1/responses", json={"model": "fx1", "input": []}).status_code == 422
        and oi_clean.post(
            "/v1/responses", json={"model": "fx1", "input": [{"role": "user"}]}
        ).status_code
        == 400
    )
    # reasoning.effort outside the pinned set and a non-object reasoning
    # block both fail validation
    out["responses_reasoning_bounds"] = all(
        oi_clean.post(
            "/v1/responses", json={"model": "fx1", "input": "x", "reasoning": r_}
        ).status_code
        == 422
        for r_ in ({"effort": "extreme"}, {"effort": "high", "extra": 1}, "high")
    )
    # text.format is the post-validated structured-output channel:
    # json_object/schema ship when the output conforms, violation 502s in
    # the OpenAI error shape, a malformed spec is refused pre-spend, and
    # the validated bytes are what lands on the response object
    r = oi_json.post(
        "/v1/responses",
        json={
            "model": "fx1",
            "input": "x",
            "text": {"format": {"type": "json_object"}},
        },
    )
    out["responses_text_format_json_pass"] = (
        r.status_code == 200 and r.json()["output"][0]["content"][0]["text"] == '{"score": 0.9}'
    )
    r = oi_broken.post(
        "/v1/responses",
        json={
            "model": "fx1",
            "input": "x",
            "text": {"format": {"type": "json_object"}},
        },
    )
    out["responses_text_format_violation_502"] = (
        r.status_code == 502 and r.json()["error"]["code"] == "format_violation"
    )
    r = oi_json.post(
        "/v1/responses",
        json={
            "model": "fx1",
            "input": "x",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "s",
                    "schema": _schema,
                }
            },
        },
    )
    out["responses_text_format_schema_pass"] = r.status_code == 200
    out["responses_text_format_bad_spec_422"] = (
        oi_json.post(
            "/v1/responses",
            json={"model": "fx1", "input": "x", "text": {"format": {"type": "weird"}}},
        ).status_code
        == 422
    )
    # user/safety_identifier/metadata stamp the completion record
    r = oi_clean.post(
        "/v1/responses",
        json={
            "model": "fx1",
            "input": "stamp",
            "user": "u-9",
            "metadata": {"team": "risk"},
        },
    )
    _resp_rec = oi_clean.get(f"/harness/completions/{r.headers['X-Fx1-Completion-Id']}").json()
    out["responses_stamps_record"] = (
        r.status_code == 200
        and _resp_rec.get("user") == "u-9"
        and _resp_rec.get("metadata") == {"team": "risk"}
    )
    # the gate fires over the Responses surface too — honesty refusal in
    # the OpenAI error shape
    r = oi_dirty.post("/v1/responses", json={"model": "fx1", "input": "h"})
    out["responses_gate_502_openai_shape"] = (
        r.status_code == 502
        and set(r.json()) == {"error"}
        and r.json()["error"]["code"] == "honesty_gate"
    )
    # Idempotency-Key: keyed retry replays byte-identically (no re-spend),
    # a different body under the same key 409s, a refusal never pins
    _r_idem = {"model": "fx1", "input": "idem"}
    _r_key = {"Idempotency-Key": "resp-k1"}
    i1 = oi_clean.post("/v1/responses", json=_r_idem, headers=_r_key)
    i2 = oi_clean.post("/v1/responses", json=_r_idem, headers=_r_key)
    out["responses_idem_replay_byte_identical"] = (
        i1.status_code == 200
        and i2.status_code == 200
        and i1.content == i2.content
        and i2.headers.get("X-Fx1-Idempotent-Replay") == "true"
        and "_fx1_completion_id" not in i2.text
        and "_fx1_usage" not in i2.text
    )
    i3 = oi_clean.post("/v1/responses", json={"model": "fx1", "input": "different"}, headers=_r_key)
    out["responses_idem_conflict_409"] = i3.status_code == 409
    # stream: the Responses event grammar — event:/id:/data: per frame,
    # sequential ids, terminal frame is response.completed (no [DONE]),
    # and the completed event embeds usage + the same response object the
    # JSON path returns
    rs1 = oi_usage.post(
        "/v1/responses",
        json={**_r_idem, "stream": True},
        headers={"Idempotency-Key": "resp-rs1"},
    )
    _rlines = rs1.text.splitlines()
    _rev = [ln[7:] for ln in _rlines if ln.startswith(_SSE_EVENT_PREFIX)]
    _rid_lines = [ln[4:] for ln in _rlines if ln.startswith("id: ")]
    _rdata = [_json3.loads(ln[6:]) for ln in _rlines if ln.startswith("data: ")]
    out["responses_stream_event_grammar"] = (
        rs1.status_code == 200
        and rs1.headers["content-type"].startswith("text/event-stream")
        and len(_rev) == len(_rdata) == len(_rid_lines)
        and _rid_lines == [str(i) for i in range(len(_rid_lines))]
        and _rev[0] == "response.created"
        and _rev[1] == "response.in_progress"
        and _rev[-1] == "response.completed"
        and "response.output_text.delta" in _rev
        and "[DONE]" not in rs1.text
        and all(d.get("type") == e for d, e in zip(_rdata, _rev, strict=True))
        and _rdata[-1]["response"]["usage"]
        == {
            "input_tokens": 5,
            "output_tokens": 4,
            "total_tokens": 9,
        }
        and "".join(d["delta"] for d in _rdata if d["type"] == "response.output_text.delta")
        == "clean:idem"
    )
    # keyed stream replay regenerates the byte-identical byte sequence;
    # Last-Event-ID resume replays the pinned stream minus the seen prefix
    rs2 = oi_usage.post(
        "/v1/responses",
        json={**_r_idem, "stream": True},
        headers={"Idempotency-Key": "resp-rs1"},
    )
    out["responses_stream_replay_byte_identical"] = (
        rs2.status_code == 200
        and rs1.content == rs2.content
        and rs2.headers.get("X-Fx1-Idempotent-Replay") == "true"
    )
    _rframes = rs1.text.split("\n\n")[:-1]
    rs_res = oi_usage.post(
        "/v1/responses",
        json={**_r_idem, "stream": True},
        headers={"Idempotency-Key": "resp-rs1", "Last-Event-ID": "3"},
    )
    out["responses_resume_suffix"] = rs_res.status_code == 200 and rs_res.text == "".join(
        f + "\n\n" for f in _rframes[4:]
    )
    out["responses_resume_miss_409"] = (
        oi_usage.post(
            "/v1/responses",
            json={**_r_idem, "stream": True},
            headers={"Idempotency-Key": "resp-fresh", "Last-Event-ID": "0"},
        ).status_code
        == 409
    )
    out["responses_resume_needs_stream_400"] = (
        oi_clean.post(
            "/v1/responses",
            json=_r_idem,
            headers={**_r_key, "Last-Event-ID": "0"},
        ).status_code
        == 400
    )
    # capacity admission applies to the Responses surface too
    rd_app = api_mod.create_app(backend_resolver=lambda *a, **k: _OiBackend(), max_inflight=1)
    rd_app.state.inflight_slots.acquire()
    try:
        r = _TC2(rd_app).post("/v1/responses", json={"model": "fx1", "input": "h"})
    finally:
        rd_app.state.inflight_slots.release()
    out["responses_over_capacity_503_shape"] = (
        r.status_code == 503 and r.json()["error"]["code"] == "over_capacity"
    )

    # ---- lane 82: the tool channel on /v1/responses ----
    # The flattened Responses spec and the function_call output items run
    # the same gated pipeline as chat — the request translates onto the
    # shared tool channel (specs nest under `function`), calls land in
    # `output` as function_call items, and a link without the channel
    # answers 501.
    rt_spec = {
        "type": "function",
        "name": "calc",
        "description": "arithmetic",
        "parameters": {"type": "object", "properties": {"x": {"type": "number"}}},
    }
    r = oi_tools.post(
        "/v1/responses",
        json={
            "model": "fx1",
            "input": "calc one",
            "tools": [rt_spec],
            "tool_choice": "required",
            "parallel_tool_calls": True,
        },
    )
    ritems = r.json().get("output") or []
    rfc: dict[str, Any] = next((it for it in ritems if it.get("type") == "function_call"), {})
    out["responses_tools_call_items"] = (
        r.status_code == 200
        and rfc.get("call_id") == "call_0"
        and rfc.get("name") == "calc"
        and rfc.get("arguments") == '{"x": 1}'
        and rfc.get("status") == "completed"
        and str(rfc.get("id", "")).startswith("fc_")
        # a calls-only turn ships no message item
        and not any(it.get("type") == "message" for it in ritems)
    )
    out["responses_tools_forwarded_verbatim"] = (
        r.status_code == 200
        and oi_tool.seen_tools
        == [
            {
                "type": "function",
                "function": {k: v for k, v in rt_spec.items() if k != "type"},
            }
        ]
        and oi_tool.seen_choice == "required"
        and oi_tool.seen_parallel is True
        and r.json().get("tools") == [rt_spec]
        and r.json().get("tool_choice") == "required"
        and r.json().get("parallel_tool_calls") is True
    )
    # dict tool_choice folds onto the shared channel's {type, function:{name}}
    r = oi_tools.post(
        "/v1/responses",
        json={
            "model": "fx1",
            "input": "x",
            "tools": [rt_spec],
            "tool_choice": {"type": "function", "name": "calc"},
        },
    )
    out["responses_tool_choice_dict_folds"] = (
        r.status_code == 200
        and oi_tool.seen_choice == {"type": "function", "function": {"name": "calc"}}
        and r.json().get("tool_choice") == {"type": "function", "name": "calc"}
    )
    r = oi_tools.post(
        "/v1/responses",
        json={"model": "fx1", "input": "x", "tools": [rt_spec]},
    )
    out["responses_tool_choice_default_auto"] = r.json().get("tool_choice") == "auto"
    # function_call/function_call_output items fold into the shared chat
    # history — one assistant turn per call-run, a role:tool message per
    # output, verbatim
    r = oi_tools.post(
        "/v1/responses",
        json={
            "model": "fx1",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "q"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_a",
                    "name": "calc",
                    "arguments": '{"a": 1}',
                },
                {
                    "type": "function_call",
                    "call_id": "call_b",
                    "name": "calc",
                    "arguments": '{"b": 2}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_a",
                    "output": "2",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_b",
                    "output": [{"type": "output_text", "text": "3"}],
                },
                {"role": "user", "content": "and?"},
            ],
        },
    )
    out["responses_tool_items_fold_history"] = r.status_code == 200 and oi_tool.seen_messages == [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "calc", "arguments": '{"a": 1}'},
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {"name": "calc", "arguments": '{"b": 2}'},
                },
            ],
        },
        {"role": "tool", "content": "2", "tool_call_id": "call_a"},
        {"role": "tool", "content": "3", "tool_call_id": "call_b"},
        {"role": "user", "content": "and?"},
    ]
    # a link without the channel answers 501 — the spec never drops
    out["responses_tools_no_channel_501"] = (
        oi_clean.post(
            "/v1/responses",
            json={"model": "fx1", "input": "x", "tools": [rt_spec]},
        ).status_code
        == 501
    )
    # fail-closed bounds: >128 tools, tool_choice/parallel without tools,
    # a bad dict choice, malformed items — all refuse before model spend
    out["responses_tools_bounds_refused"] = all(
        oi_tools.post("/v1/responses", json={"model": "fx1", "input": "x", **kw}).status_code == 422
        for kw in (
            {"tools": [rt_spec] * 129},
            {"tool_choice": "auto"},
            {"parallel_tool_calls": True},
            {"tools": [rt_spec], "tool_choice": {"type": "function"}},
            {"tools": [rt_spec], "tool_choice": {"type": "bogus", "name": "f"}},
            {"tools": [{"type": "bogus", "name": "f"}]},
        )
    )
    out["responses_tool_items_bad_shape_400"] = all(
        oi_tools.post("/v1/responses", json={"model": "fx1", "input": [it]}).status_code == 400
        for it in (
            {"type": "function_call", "name": "calc", "arguments": "{}"},
            {"type": "function_call", "call_id": "c", "arguments": "{}"},
            {"type": "function_call", "call_id": "c", "name": "calc"},
            {"type": "function_call_output", "output": "2"},
            {"type": "function_call_output", "call_id": "c"},
        )
    )
    # the stream emits the fc-item events — output_item.added, per-part
    # arguments deltas, arguments.done, output_item.done — and the
    # completed frame carries the same object the JSON path returns
    rts1 = oi_tools.post(
        "/v1/responses",
        json={
            "model": "fx1",
            "input": "calc",
            "tools": [rt_spec],
            "stream": True,
        },
        headers={"Idempotency-Key": "resp-tools-82"},
    )
    _tlines = rts1.text.splitlines()
    _tev = [ln[7:] for ln in _tlines if ln.startswith(_SSE_EVENT_PREFIX)]
    _tdata = [_json3.loads(ln[6:]) for ln in _tlines if ln.startswith("data: ")]
    _tfc_done: dict[str, Any] = next(
        (
            d["item"]
            for d in _tdata
            if d["type"] == "response.output_item.done" and d["item"].get("type") == "function_call"
        ),
        {},
    )
    out["responses_tool_stream_events"] = (
        rts1.status_code == 200
        and "response.output_item.added" in _tev
        and "response.function_call_arguments.delta" in _tev
        and "response.function_call_arguments.done" in _tev
        and any(
            d.get("type") == "response.output_item.added"
            and d.get("item", {}).get("type") == "function_call"
            and d["item"].get("status") == "in_progress"
            for d in _tdata
        )
        and "".join(
            d["delta"] for d in _tdata if d["type"] == "response.function_call_arguments.delta"
        )
        == '{"x": 1}'
        and _tfc_done.get("call_id") == "call_0"
        and _tfc_done.get("status") == "completed"
        and _tdata[-1]["response"]["output"]
        == [
            {
                "type": "function_call",
                "id": _tfc_done["id"],
                "call_id": "call_0",
                "name": "calc",
                "arguments": '{"x": 1}',
                "status": "completed",
            }
        ]
    )
    # fc item ids mint once — a keyed stream replays byte-identically
    rts2 = oi_tools.post(
        "/v1/responses",
        json={
            "model": "fx1",
            "input": "calc",
            "tools": [rt_spec],
            "stream": True,
        },
        headers={"Idempotency-Key": "resp-tools-82"},
    )
    out["responses_tool_stream_replay"] = (
        rts2.status_code == 200
        and rts1.content == rts2.content
        and rts2.headers.get("X-Fx1-Idempotent-Replay") == "true"
    )
    # the retrieval index carries the fc items too — a stored calls-only
    # response round-trips the full output
    rt_store = oi_tools.post(
        "/v1/responses",
        json={"model": "fx1", "input": "calc", "tools": [rt_spec]},
    )
    rt_id = rt_store.json().get("id", "")
    rget = oi_tools.get(f"/v1/responses/{rt_id}")
    out["responses_tool_store_retrieve"] = (
        rget.status_code == 200
        and (rget.json().get("output") or [{}])[0].get("type") == "function_call"
        and (rget.json().get("output") or [{}])[0].get("call_id") == "call_0"
    )
    out["responses_tool_store_delete"] = (
        oi_tools.delete(f"/v1/responses/{rt_id}").status_code == 200
        and oi_tools.get(f"/v1/responses/{rt_id}").status_code == 404
    )
    out["capabilities_reports_responses_tools"] = (
        oi_clean.get("/harness/capabilities").json()["features"].get("openai_responses_tools")
        is True
    )

    # ---- /v1/files + /v1/batches: the OpenAI async channel over the jobs
    # executor — store caps, submit-time line validation, per-line gated
    # execution through the live route cores, output files, cooperative
    # cancel, expiry projection, and the shared /v1 idempotency space.
    fb = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _OiBackend()))
    _bf_lines = [
        {
            "custom_id": "a",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "fx1", "messages": [{"role": "user", "content": "x"}]},
        },
        {
            "custom_id": "b",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "fx1", "messages": [{"role": "user", "content": "y"}]},
        },
        {
            "custom_id": "bad-temp",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "fx1",
                "temperature": 5.0,
                "messages": [{"role": "user", "content": "z"}],
            },
        },
    ]
    _bf_bytes = ("\n".join(_json3.dumps(line) for line in _bf_lines) + "\n").encode()

    def _upload(client: Any, content: bytes = _bf_bytes) -> dict[str, Any]:
        r = client.post(
            "/v1/files",
            files={"file": ("in.jsonl", content, "application/jsonl")},
            data={"purpose": "batch"},
        )
        return dict(r.json())

    def _wait_batch(client: Any, batch_id: str) -> dict[str, Any]:
        for _ in range(500):
            b = client.get(f"/v1/batches/{batch_id}").json()
            if b["status"] in ("completed", "failed", "expired", "cancelled"):
                return dict(b)
            time.sleep(0.01)
        return dict(b)

    up = fb.post(
        "/v1/files",
        files={"file": ("in.jsonl", _bf_bytes, "application/jsonl")},
        data={"purpose": "batch"},
    )
    fobj = up.json()
    out["file_upload_200_shape"] = (
        up.status_code == 200
        and fobj["object"] == "file"
        and fobj["purpose"] == "batch"
        and fobj["filename"] == "in.jsonl"
        and fobj["bytes"] == len(_bf_bytes)
        and fobj["status"] == "processed"
        and fobj["id"].startswith("file-")
    )
    # fail-closed upload surface: wrong purpose, missing file part,
    # non-jsonl name, empty content, oversized — all OpenAI-shaped 4xx
    up_purpose = fb.post(
        "/v1/files",
        files={"file": ("in.jsonl", _bf_bytes, "application/jsonl")},
        data={"purpose": "user_data"},
    )
    out["file_upload_purpose_400"] = (
        up_purpose.status_code == 400
        and up_purpose.json()["error"]["code"] == "invalid_request"
        and "purpose" in up_purpose.json()["error"]["message"]
    )
    out["file_upload_missing_400"] = (
        fb.post("/v1/files", data={"purpose": "batch"}).status_code == 400
    )
    out["file_upload_ext_400"] = (
        fb.post(
            "/v1/files",
            files={"file": ("in.txt", _bf_bytes, "text/plain")},
            data={"purpose": "batch"},
        ).status_code
        == 400
    )
    out["file_upload_empty_400"] = (
        fb.post(
            "/v1/files",
            files={"file": ("in.jsonl", b"", "application/jsonl")},
            data={"purpose": "batch"},
        ).status_code
        == 400
    )
    tiny = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _OiBackend(), file_bytes_max=4))
    out["file_upload_oversize_413"] = (
        tiny.post(
            "/v1/files",
            files={"file": ("in.jsonl", _bf_bytes, "application/jsonl")},
            data={"purpose": "batch"},
        ).status_code
        == 413
    )
    # listing newest-first + retrieve + content round-trip + delete
    fid = fobj["id"]
    flist = fb.get("/v1/files").json()
    out["files_list_newest_first"] = (
        flist["object"] == "list"
        and flist["data"][0]["id"] == fid
        and all(f["object"] == "file" for f in flist["data"])
    )
    out["file_retrieve_200"] = fb.get(f"/v1/files/{fid}").json()["id"] == fid
    out["file_retrieve_404_shape"] = (
        fb.get("/v1/files/file-nope").status_code == 404
        and fb.get("/v1/files/file-nope").json()["error"]["code"] == "file_not_found"
    )
    fcont = fb.get(f"/v1/files/{fid}/content")
    out["file_content_roundtrip"] = (
        fcont.status_code == 200
        and fcont.content == _bf_bytes
        and fcont.headers["content-type"].startswith("application/jsonl")
    )
    out["file_delete_then_404"] = (
        fb.delete(f"/v1/files/{fid}").json() == {"id": fid, "object": "file", "deleted": True}
        and fb.get(f"/v1/files/{fid}").status_code == 404
    )
    # LRU bound: file_max=1 evicts the first upload
    lru = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _OiBackend(), file_max=1))
    fid1 = _upload(lru)["id"]
    fid2 = _upload(lru)["id"]
    out["file_lru_evicts_oldest"] = (
        lru.get(f"/v1/files/{fid1}").status_code == 404
        and lru.get(f"/v1/files/{fid2}").status_code == 200
    )

    # batch lifecycle over the uploaded file
    upb = _upload(fb)
    bc = fb.post(
        "/v1/batches",
        json={
            "input_file_id": upb["id"],
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "metadata": {"k": "v"},
        },
    )
    bobj = bc.json()
    out["batch_create_200_shape"] = (
        bc.status_code == 200
        and bobj["object"] == "batch"
        and bobj["id"].startswith("batch_")
        and bobj["input_file_id"] == upb["id"]
        and bobj["endpoint"] == "/v1/chat/completions"
        and bobj["completion_window"] == "24h"
        and bobj["metadata"] == {"k": "v"}
        # counts race with the worker — pin only the total at create time
        and bobj["request_counts"]["total"] == 3
        and bobj["request_counts"]["completed"] + bobj["request_counts"]["failed"] <= 3
        and bobj["status"] in ("validating", "in_progress", "completed")
        and bobj["expires_at"] == bobj["created_at"] + 86400
    )
    bterm = _wait_batch(fb, bobj["id"])
    out["batch_completes_counts"] = (
        bterm["status"] == "completed"
        and bterm["request_counts"] == {"total": 3, "completed": 2, "failed": 1}
        and bterm["completed_at"] is not None
        and bterm["finalizing_at"] is not None
        and bterm["in_progress_at"] is not None
        and bobj["id"] == bterm["id"]
    )
    # output file: one OpenAI batch-result line per input; the bad line
    # carries the same 400 error body the live route would have returned
    outf = fb.get(f"/v1/files/{bterm['output_file_id']}/content")
    olines = [_json3.loads(line) for line in outf.text.splitlines() if line.strip()]
    out["batch_output_file_shape"] = (
        outf.status_code == 200
        and outf.headers["content-type"].startswith("application/jsonl")
        and len(olines) == 3
        and all(
            o["id"].startswith("batch_req_") and o["response"]["request_id"].startswith("req_")
            for o in olines
        )
    )
    out["batch_output_line_bodies"] = (
        olines[0]["custom_id"] == "a"
        and olines[0]["response"]["status_code"] == 200
        and olines[0]["response"]["body"]["object"] == "chat.completion"
        and olines[0]["response"]["body"]["choices"][0]["message"]["content"] == "clean:x"
        and olines[1]["response"]["body"]["choices"][0]["message"]["content"] == "clean:y"
        and olines[2]["custom_id"] == "bad-temp"
        and olines[2]["response"]["status_code"] == 400
        and olines[2]["response"]["body"]["error"]["type"] == "invalid_request_error"
    )
    # output files are listed with purpose=batch_output
    out["batch_output_file_listed"] = any(
        f["id"] == bterm["output_file_id"] and f["purpose"] == "batch_output"
        for f in fb.get("/v1/files").json()["data"]
    )
    # submit-time validation: corrupt lines fail the whole create 400
    out["batch_submit_bad_json_400"] = (
        fb.post(
            "/v1/batches",
            json={
                "input_file_id": _upload(fb, b"not json\n")["id"],
                "endpoint": "/v1/chat/completions",
            },
        ).status_code
        == 400
    )
    _bad_shape = (
        _json3.dumps({"custom_id": "x", "method": "GET", "url": "/v1/chat/completions", "body": {}})
        + "\n"
    ).encode()
    r_bad = fb.post(
        "/v1/batches",
        json={
            "input_file_id": _upload(fb, _bad_shape)["id"],
            "endpoint": "/v1/chat/completions",
        },
    )
    out["batch_submit_line_shape_400"] = (
        r_bad.status_code == 400 and "line 1" in r_bad.json()["error"]["message"]
    )
    _url_mm = (
        _json3.dumps(
            {
                "custom_id": "x",
                "method": "POST",
                "url": "/v1/responses",
                "body": {"model": "fx1", "input": "h"},
            }
        )
        + "\n"
    ).encode()
    out["batch_submit_url_mismatch_400"] = (
        fb.post(
            "/v1/batches",
            json={
                "input_file_id": _upload(fb, _url_mm)["id"],
                "endpoint": "/v1/chat/completions",
            },
        ).status_code
        == 400
    )
    out["batch_bad_endpoint_422"] = (
        fb.post(
            "/v1/batches",
            json={"input_file_id": upb["id"], "endpoint": "/v1/completions"},
        ).status_code
        == 422
    )
    out["batch_bad_file_404"] = (
        fb.post(
            "/v1/batches",
            json={"input_file_id": "file-nope", "endpoint": "/v1/chat/completions"},
        ).status_code
        == 404
    )
    # embeddings lines ride the same channel — the endpoint's own request
    # model validates the body and the embed core answers the envelope
    _emb_lines = (
        _json3.dumps(
            {
                "custom_id": "e1",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {"model": "emb-m", "input": ["a", "b"]},
            }
        )
        + "\n"
        + _json3.dumps(
            {
                "custom_id": "e2",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {"model": "emb-m", "input": ""},
            }
        )
        + "\n"
    ).encode()
    _eb_fid = _upload(oi_emb, _emb_lines)["id"]
    _eb = oi_emb.post(
        "/v1/batches",
        json={"input_file_id": _eb_fid, "endpoint": "/v1/embeddings"},
    )
    _eb_done = _wait_batch(oi_emb, _eb.json()["id"])
    _eb_lines = [
        _json3.loads(ol)
        for ol in oi_emb.get(f"/v1/files/{_eb_done['output_file_id']}/content").text.splitlines()
        if ol.strip()
    ]
    out["openai_embeddings_batch_lines"] = (
        _eb.status_code == 200
        and _eb_done["status"] == "completed"
        and _eb_lines[0]["response"]["status_code"] == 200
        and _eb_lines[0]["response"]["body"]["object"] == "list"
        and len(_eb_lines[0]["response"]["body"]["data"]) == 2
        and _eb_lines[1]["response"]["status_code"] == 400
    )
    # output files can't be resubmitted as batch input
    out["batch_output_as_input_400"] = (
        fb.post(
            "/v1/batches",
            json={
                "input_file_id": bterm["output_file_id"],
                "endpoint": "/v1/chat/completions",
            },
        ).status_code
        == 400
    )
    # idempotency: keyed create replays the submit envelope; conflict 409
    bidem = {"input_file_id": upb["id"], "endpoint": "/v1/chat/completions"}
    bi1 = fb.post("/v1/batches", json=bidem, headers={"Idempotency-Key": "bk-1"})
    bi2 = fb.post("/v1/batches", json=bidem, headers={"Idempotency-Key": "bk-1"})
    out["batch_idem_replay"] = (
        bi1.json()["id"] == bi2.json()["id"]
        and bi2.headers.get("X-Fx1-Idempotent-Replay") == "true"
    )
    out["batch_idem_conflict_409"] = (
        fb.post(
            "/v1/batches",
            json={"input_file_id": upb["id"], "endpoint": "/v1/responses"},
            headers={"Idempotency-Key": "bk-1"},
        ).status_code
        == 409
    )
    # listing + cursor pagination
    blist = fb.get("/v1/batches?limit=1").json()
    out["batches_list_shape"] = (
        blist["object"] == "list"
        and len(blist["data"]) == 1
        and blist["has_more"] is True
        and blist["first_id"] == blist["data"][0]["id"]
        and blist["last_id"] == blist["data"][0]["id"]
    )
    bpage2 = fb.get(f"/v1/batches?limit=50&after={blist['last_id']}").json()
    out["batches_list_after_cursor"] = (
        bpage2["has_more"] is False
        and all(b["id"] != blist["last_id"] for b in bpage2["data"])
        and len(bpage2["data"]) >= 1
    )
    out["batch_retrieve_404"] = fb.get("/v1/batches/batch_nope").status_code == 404
    out["batch_cancel_terminal_409"] = (
        fb.post(f"/v1/batches/{bterm['id']}/cancel").status_code == 409
    )
    # cooperative cancel: a gated backend holds the worker mid-batch;
    # cancel lands 'cancelling', the batch finishes 'cancelled' with the
    # lines completed so far written to the output file
    gate_ev = _threading.Event()

    class _GateBackend(_OiBackend):
        def complete(
            self,
            messages: list[dict[str, str]],
            *,
            sampling: SamplingParams | None = None,
        ) -> str:
            gate_ev.wait(10)
            return super().complete(messages, sampling=sampling)

    gapp = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _GateBackend()))
    gid = _upload(gapp)["id"]
    gcb = gapp.post(
        "/v1/batches",
        json={"input_file_id": gid, "endpoint": "/v1/chat/completions"},
    )
    gcc = gapp.post(f"/v1/batches/{gcb.json()['id']}/cancel")
    gate_ev.set()
    gterm = _wait_batch(gapp, gcb.json()["id"])
    out["batch_cancel_cooperative"] = (
        gcc.status_code == 200
        and gcc.json()["status"] == "cancelling"
        and gterm["status"] == "cancelled"
        and gterm["cancelled_at"] is not None
        and gterm["request_counts"]["completed"] <= 1
        and gterm["output_file_id"] is not None
    )
    # expiry projection: a record past expires_at reports 'expired'
    exp_app = api_mod.create_app(backend_resolver=lambda *a, **k: _OiBackend())
    past = api_mod._BatchRecord(  # noqa: SLF001
        batch_id="batch_past",
        input_file_id="file-x",
        endpoint="/v1/chat/completions",
        completion_window="24h",
        status="in_progress",
        created_at=1,
        expires_at=2,
    )
    exp_app.state.batch_store.put(past)
    r_exp = _TC2(exp_app).get("/v1/batches/batch_past").json()
    out["batch_expiry_projection"] = (
        r_exp["status"] == "expired" and r_exp["expired_at"] is not None
    )
    # Terminal webhooks on /v1/batches — the same fx1 extension as
    # /harness/jobs and /v1/fine_tuning/jobs: fire once at terminal,
    # signed when callback_secret is set, verdict rides the record.
    _bwh_hits: list[dict[str, Any]] = []
    _bwh_raw: list[bytes] = []
    _bwh_hdrs: list[dict[str, str]] = []
    _bwh_path_n: dict[str, int] = {}

    class _BatchHook(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 — http.server handler name
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            _bwh_raw.append(raw)
            _bwh_hdrs.append(dict(self.headers.items()))
            _bwh_hits.append(_json.loads(raw))
            _bwh_path_n[self.path] = _bwh_path_n.get(self.path, 0) + 1
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            pass

    _bwh_srv = ThreadingHTTPServer(("127.0.0.1", 0), _BatchHook)
    _threading.Thread(target=_bwh_srv.serve_forever, daemon=True).start()
    _bwh_url = f"http://127.0.0.1:{_bwh_srv.server_address[1]}"
    # loopback callback_url needs the private-networks opt-in, read per
    # request (saved/restored verbatim).
    saved_bwh_env = os.environ.get("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS")
    os.environ["FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"] = "1"
    bwh_fid = _upload(fb)["id"]
    bwh = fb.post(
        "/v1/batches",
        json={
            "input_file_id": bwh_fid,
            "endpoint": "/v1/chat/completions",
            "callback_url": f"{_bwh_url}/batch-hook",
            "callback_secret": "whsec-batch",
        },
    )
    bwh_id = bwh.json()["id"]
    bwh_fin = _wait_batch(fb, bwh_id)
    deadline = time.monotonic() + 10.0
    while bwh_fin.get("callback_status") is None and time.monotonic() < deadline:
        time.sleep(0.02)
        bwh_fin = fb.get(f"/v1/batches/{bwh_id}").json()
    _bh = _bwh_hits[-1] if _bwh_hits else {}
    _bh_ok = False
    if _bwh_path_n.get("/batch-hook") == 1:
        from fx1.serve.webhooks import verify_webhook  # noqa: PLC0415

        _bh_ok = verify_webhook(
            "whsec-batch",
            _bwh_hdrs[-1].get("X-Fx1-Webhook-Timestamp"),
            _bwh_hdrs[-1].get("X-Fx1-Webhook-Signature"),
            _bwh_raw[-1],
        )
    out["batch_webhook_fires_signed"] = (
        bwh_fin["status"] == "completed"
        and bwh_fin.get("callback_status") == "delivered"
        and bwh_fin.get("callback_attempts") == 1
        and _bwh_path_n.get("/batch-hook") == 1
        and _bh.get("id") == bwh_id
        and _bh.get("status") == "completed"
        and _bh_ok
    )
    out["batch_webhook_secret_never_serializes"] = "callback_secret" not in _bh
    # Lazy expiry also fires — exactly once across reads.
    exp_cb_app = api_mod.create_app(backend_resolver=lambda *a, **k: _OiBackend())
    past_cb = api_mod._BatchRecord(  # noqa: SLF001
        batch_id="batch_past_cb",
        input_file_id="file-x",
        endpoint="/v1/chat/completions",
        completion_window="24h",
        status="in_progress",
        created_at=1,
        expires_at=2,
        callback_url=f"{_bwh_url}/batch-expiry",
    )
    exp_cb_app.state.batch_store.put(past_cb)
    exp_cbc = _TC2(exp_cb_app)
    r_expc = exp_cbc.get("/v1/batches/batch_past_cb").json()
    deadline = time.monotonic() + 10.0
    while _bwh_path_n.get("/batch-expiry", 0) < 1 and time.monotonic() < deadline:
        time.sleep(0.02)
    exp_cbc.get("/v1/batches/batch_past_cb")
    exp_cbc.get("/v1/batches/batch_past_cb")
    time.sleep(0.1)
    out["batch_webhook_expiry_fires_once"] = (
        r_expc["status"] == "expired"
        and r_expc.get("callback_status") == "delivered"
        and _bwh_path_n.get("/batch-expiry") == 1
    )
    _bwh_srv.shutdown()
    _bwh_srv.server_close()
    if saved_bwh_env is None:
        os.environ.pop("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS", None)
    else:
        os.environ["FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"] = saved_bwh_env
    # Submit-time guards: a secret without a url is a 422, never a zombie.
    bad_cb = fb.post(
        "/v1/batches",
        json={
            "input_file_id": bwh_fid,
            "endpoint": "/v1/chat/completions",
            "callback_secret": "x",
        },
    )
    out["batch_webhook_guards_422"] = (
        bad_cb.status_code == 422 and bad_cb.json().get("error", {}).get("code") == "validation"
    )
    # over-capacity admission: a batch submit under a held inflight slot
    # is the same 503 over_capacity as the sync surface
    cap_app = api_mod.create_app(backend_resolver=lambda *a, **k: _OiBackend(), max_inflight=1)
    cap = _TC2(cap_app)
    cap_fid = _upload(cap)["id"]
    cap_app.state.inflight_slots.acquire()
    try:
        cap_r = cap.post(
            "/v1/batches",
            json={"input_file_id": cap_fid, "endpoint": "/v1/chat/completions"},
        )
    finally:
        cap_app.state.inflight_slots.release()
    out["batch_over_capacity_503"] = (
        cap_r.status_code == 503
        and cap_r.json()["error"]["code"] == "over_capacity"
        and cap_r.headers.get("retry-after") == "1"
    )

    # header routing: the submitter's X-Fx1-Backend applies to every line
    # (unknown names are rejected per-line 400, like the live route)
    def _hdr_resolver(name: str, **kw: Any) -> _OiBackend:
        be = _OiBackend()
        be._model = f"routed-{name}"
        return be

    happ = _TC2(api_mod.create_app(backend_resolver=_hdr_resolver))
    hid = _upload(happ)["id"]
    hcb = happ.post(
        "/v1/batches",
        json={"input_file_id": hid, "endpoint": "/v1/chat/completions"},
        headers={"X-Fx1-Backend": "byok"},
    )
    hterm = _wait_batch(happ, hcb.json()["id"])
    hout = happ.get(f"/v1/files/{hterm['output_file_id']}/content")
    hlines = [_json3.loads(line) for line in hout.text.splitlines() if line.strip()]
    out["batch_header_backend_routes_lines"] = (
        hterm["status"] == "completed"
        and hlines[0]["response"]["status_code"] == 200
        and hlines[0]["response"]["body"]["model"] == "routed-byok"
        and hlines[1]["response"]["body"]["model"] == "routed-byok"
        and hlines[2]["response"]["status_code"] == 400
    )
    out["batch_header_unknown_backend_line_400"] = (
        lambda hb: _wait_batch(happ, hb["id"])["request_counts"]["failed"] == 3
    )(
        happ.post(
            "/v1/batches",
            json={"input_file_id": hid, "endpoint": "/v1/chat/completions"},
            headers={"X-Fx1-Backend": "not-a-backend"},
        ).json()
    )
    # the Responses endpoint runs through the same machinery
    rid2 = _upload(
        fb,
        (
            _json3.dumps(
                {
                    "custom_id": "r1",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {"model": "fx1", "input": "hi"},
                }
            )
            + "\n"
        ).encode(),
    )["id"]
    rcb = fb.post(
        "/v1/batches",
        json={"input_file_id": rid2, "endpoint": "/v1/responses"},
    )
    rterm = _wait_batch(fb, rcb.json()["id"])
    rout = fb.get(f"/v1/files/{rterm['output_file_id']}/content")
    rlines = [_json3.loads(line) for line in rout.text.splitlines() if line.strip()]
    out["batch_responses_endpoint"] = (
        rterm["status"] == "completed"
        and rlines[0]["response"]["status_code"] == 200
        and rlines[0]["response"]["body"]["object"] == "response"
        and rlines[0]["response"]["body"]["output"][0]["content"][0]["text"] == "clean:hi"
    )

    # ---- /v1/uploads: chunked file assembly — intent → parts →
    # complete mints a /v1/files record; md5 checked pre-mint; terminal
    # states and declared-byte bounds fail closed.
    import hashlib as _hashlib  # noqa: PLC0415
    import tempfile as _tempfile  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    _ul_payload = b'{"l":1}\n{"l":2}\n{"l":3}\n'
    ulp1, ulp2, ulp3 = _ul_payload[:8], _ul_payload[8:18], _ul_payload[18:]
    uc = fb.post(
        _PATH_UPLOADS,
        json={
            "purpose": "batch",
            "filename": "big.jsonl",
            "bytes": len(_ul_payload),
            "mime_type": "application/jsonl",
        },
    )
    uobj = uc.json()
    out["upload_create_200_shape"] = (
        uc.status_code == 200
        and uobj["object"] == "upload"
        and uobj["id"].startswith("upload_")
        and uobj["status"] == "pending"
        and uobj["bytes"] == len(_ul_payload)
        and uobj["expires_at"] > uobj["created_at"]
        and uobj["file"] is None
    )
    out["upload_create_purpose_400"] = (
        fb.post(
            _PATH_UPLOADS,
            json={
                "purpose": "user_data",
                "filename": "x.jsonl",
                "bytes": 1,
                "mime_type": "t",
            },
        ).status_code
        == 400
    )
    out["upload_create_ext_400"] = (
        fb.post(
            _PATH_UPLOADS,
            json={
                "purpose": "batch",
                "filename": "x.txt",
                "bytes": 1,
                "mime_type": "t",
            },
        ).status_code
        == 400
    )
    uid = uobj["id"]
    upart = fb.post(f"/v1/uploads/{uid}/parts", files={"data": ("p", ulp1)})
    out["upload_part_200_shape"] = (
        upart.status_code == 200
        and upart.json()["object"] == "upload.part"
        and upart.json()["id"].startswith("part_")
        and upart.json()["upload_id"] == uid
    )
    out["upload_part_missing_404"] = (
        fb.post("/v1/uploads/upload_nope/parts", files={"data": ("p", b"x")}).status_code == 404
        and fb.post("/v1/uploads/upload_nope/parts", files={"data": ("p", b"x")}).json()["error"][
            "code"
        ]
        == "upload_not_found"
    )
    out["upload_part_no_field_400"] = fb.post(f"/v1/uploads/{uid}/parts").status_code == 400
    # cumulative bytes may never exceed the declared total
    out["upload_part_over_declared_400"] = (
        fb.post(f"/v1/uploads/{uid}/parts", files={"data": ("p", _ul_payload)}).json()["error"][
            "code"
        ]
        == "part_exceeds_declared_bytes"
    )
    pid2 = fb.post(f"/v1/uploads/{uid}/parts", files={"data": ("p", ulp2)}).json()["id"]
    pid3 = fb.post(f"/v1/uploads/{uid}/parts", files={"data": ("p", ulp3)}).json()["id"]
    pid1 = upart.json()["id"]
    # complete honors the caller's part order
    udone = fb.post(f"/v1/uploads/{uid}/complete", json={"part_ids": [pid3, pid1, pid2]}).json()
    out["upload_complete_caller_order"] = (
        udone["status"] == "completed"
        and udone["file"]["object"] == "file"
        and udone["file"]["id"].startswith("file-")
        and fb.get(f"/v1/files/{udone['file']['id']}/content").content == ulp3 + ulp1 + ulp2
    )
    # terminal: parts and re-complete refuse on a completed record
    out["upload_terminal_409"] = (
        fb.post(f"/v1/uploads/{uid}/parts", files={"data": ("p", b"x")}).json()["error"]["code"]
        == "upload_terminal"
        and fb.post(f"/v1/uploads/{uid}/complete", json={"part_ids": [pid1]}).json()["error"][
            "code"
        ]
        == "upload_terminal"
    )
    # unknown part id fails closed before the mint
    uc2 = fb.post(
        _PATH_UPLOADS,
        json={
            "purpose": "batch",
            "filename": "m.jsonl",
            "bytes": 2,
            "mime_type": "t",
        },
    ).json()["id"]
    pab = fb.post(f"/v1/uploads/{uc2}/parts", files={"data": ("p", b"ab")}).json()["id"]
    out["upload_complete_part_not_found_400"] = (
        fb.post(f"/v1/uploads/{uc2}/complete", json={"part_ids": ["part_nope"]}).json()["error"][
            "code"
        ]
        == "part_not_found"
    )
    # md5 mismatch refuses before the file mints; correct digest completes
    out["upload_md5_mismatch_400"] = (
        fb.post(
            f"/v1/uploads/{uc2}/complete",
            json={"part_ids": [pab], "md5": "0" * 32},
        ).json()["error"]["code"]
        == "checksum_mismatch"
    )
    uok = fb.post(
        f"/v1/uploads/{uc2}/complete",
        json={
            "part_ids": [pab],
            "md5": _hashlib.md5(b"ab", usedforsecurity=False).hexdigest(),
        },
    )
    out["upload_md5_ok_completes"] = (
        uok.status_code == 200
        and uok.json()["status"] == "completed"
        and uok.json()["file"]["bytes"] == 2
    )
    # declared-but-under-parted refuses (assembled != declared)
    uc3 = fb.post(
        _PATH_UPLOADS,
        json={
            "purpose": "batch",
            "filename": "d.jsonl",
            "bytes": 64,
            "mime_type": "t",
        },
    ).json()["id"]
    pu3 = fb.post(f"/v1/uploads/{uc3}/parts", files={"data": ("p", b"short")}).json()["id"]
    out["upload_under_declared_400"] = (
        fb.post(f"/v1/uploads/{uc3}/complete", json={"part_ids": [pu3]}).json()["error"]["code"]
        == "upload_incomplete"
    )
    # cancel is terminal and replays 200; completing a cancelled intent 409s
    uc4 = fb.post(
        _PATH_UPLOADS,
        json={
            "purpose": "batch",
            "filename": _CORPUS_FILE,
            "bytes": 2,
            "mime_type": "t",
        },
    ).json()["id"]
    cnl = fb.post(f"/v1/uploads/{uc4}/cancel").json()
    out["upload_cancel_then_409"] = (
        cnl["status"] == "cancelled"
        and fb.post(f"/v1/uploads/{uc4}/cancel").json()["status"] == "cancelled"
        and fb.post(f"/v1/uploads/{uc4}/parts", files={"data": ("p", b"ab")}).json()["error"][
            "code"
        ]
        == "upload_terminal"
    )
    out["upload_cancel_missing_404"] = fb.post("/v1/uploads/upload_nope/cancel").status_code == 404

    # durability: a fresh app over the same state_dir recovers pending
    # uploads AND their parts (blob before journal), and cancelled
    # records stay cancelled
    _sd = _Path(_tempfile.mkdtemp(prefix="fx1-ul-"))
    du1 = _TC2(
        api_mod.create_app(backend_resolver=lambda *a, **k: _OiBackend(), state_dir=str(_sd))
    )
    uup = du1.post(
        _PATH_UPLOADS,
        json={
            "purpose": "batch",
            "filename": "d.jsonl",
            "bytes": 4,
            "mime_type": "t",
        },
    ).json()["id"]
    dpart = du1.post(f"/v1/uploads/{uup}/parts", files={"data": ("p", b"ab")}).json()["id"]
    du1.post(
        _PATH_UPLOADS,
        json={
            "purpose": "batch",
            "filename": "gone.jsonl",
            "bytes": 1,
            "mime_type": "t",
        },
    ).json()
    du2 = _TC2(
        api_mod.create_app(backend_resolver=lambda *a, **k: _OiBackend(), state_dir=str(_sd))
    )
    ddone = du2.post(
        f"/v1/uploads/{uup}/complete",
        json={
            "part_ids": [
                dpart,
                du2.post(f"/v1/uploads/{uup}/parts", files={"data": ("p", b"cd")}).json()["id"],
            ]
        },
    )
    out["upload_state_dir_recovers"] = (
        ddone.status_code == 200
        and ddone.json()["status"] == "completed"
        and du2.get(f"/v1/files/{ddone.json()['file']['id']}/content").content == b"abcd"
    )

    # ---- /v1 retrieval: the `store` flag honored end-to-end — stored
    # envelopes fetch verbatim by id (sync, stream, n-fan-out, batch
    # lines, idem replays all index identically), store=false and
    # deletes 404, wrong-surface ids 404, the LRU bound evicts.

    # sync chat call → GET returns the identical envelope
    s1 = fb.post(
        "/v1/chat/completions",
        json={"model": "fx1", "messages": [{"role": "user", "content": "keep-me"}]},
    )
    sid = s1.json()["id"]
    out["retrieve_chat_stored"] = (
        s1.status_code == 200 and fb.get(f"/v1/chat/completions/{sid}").json() == s1.json()
    )
    # POST /v1/chat/completions/{id} — metadata replaces wholesale; choices/
    # usage sealed; the items subresource survives the update
    u1 = fb.post(
        f"/v1/chat/completions/{sid}",
        json={"metadata": {"tenant": "t1", "trace": "abc"}},
    )
    u2 = fb.post(f"/v1/chat/completions/{sid}", json={"metadata": {"trace": "xyz"}})
    u_after = fb.get(f"/v1/chat/completions/{sid}")
    u_items = fb.get(f"/v1/chat/completions/{sid}/messages")
    out["chat_update_metadata"] = (
        u1.status_code == 200
        and u1.json()["id"] == sid
        and u1.json()["metadata"] == {"tenant": "t1", "trace": "abc"}
        and u1.json()["choices"] == s1.json()["choices"]
        and u1.json()["usage"] == s1.json()["usage"]
        # wholesale replace — `tenant` is gone, not merged
        and u2.json()["metadata"] == {"trace": "xyz"}
        and u_after.json()["metadata"] == {"trace": "xyz"}
        and u_items.status_code == 200
        and len(u_items.json()["data"]) >= 1
    )
    # bounded + fail-closed: >16 pairs 422, a response id isn't a completion,
    # deleted/gone ids 404
    out["chat_update_failclosed"] = (
        fb.post(
            f"/v1/chat/completions/{sid}",
            json={"metadata": {f"k{i}": "v" for i in range(17)}},
        ).status_code
        == 422
        and fb.post("/v1/chat/completions/resp_deadbeef", json={}).status_code == 404
        and fb.post("/v1/chat/completions/chatcmpl-gone", json={}).status_code == 404
    )
    # store=false keeps the call out of the index (still logged)
    s2 = fb.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "skip-me"}],
            "store": False,
        },
    )
    r_get_miss = fb.get(f"/v1/chat/completions/{s2.json()['id']}")
    out["retrieve_store_false_404"] = (
        s2.status_code == 200
        and r_get_miss.status_code == 404
        and r_get_miss.json()["error"]["code"] == "not_found"
        # still evidence-logged — the flag gates retrieval, not the ledger
        and fb.get(f"/harness/completions/{s2.headers['x-fx1-completion-id']}").status_code == 200
    )
    # delete drops the envelope; a second delete 404s
    d1 = fb.delete(f"/v1/chat/completions/{sid}")
    out["retrieve_chat_delete"] = (
        d1.status_code == 200
        and d1.json()["object"] == "chat.completion.deleted"
        and d1.json()["deleted"] is True
        and fb.get(f"/v1/chat/completions/{sid}").status_code == 404
        and fb.delete(f"/v1/chat/completions/{sid}").status_code == 404
    )
    # responses surface stores + deletes symmetrically
    r1 = fb.post("/v1/responses", json={"model": "fx1", "input": "keep-r"})
    rid = r1.json()["id"]
    out["retrieve_response_stored"] = (
        r1.status_code == 200
        and fb.get(f"/v1/responses/{rid}").json() == r1.json()
        and fb.delete(f"/v1/responses/{rid}").json()["object"] == "response.deleted"
        and fb.get(f"/v1/responses/{rid}").status_code == 404
    )
    r2 = fb.post("/v1/responses", json={"model": "fx1", "input": "skip-r", "store": False})
    out["retrieve_response_store_false_404"] = (
        r2.status_code == 200 and fb.get(f"/v1/responses/{r2.json()['id']}").status_code == 404
    )
    # wrong-surface id is a miss, not a cross-read
    out["retrieve_wrong_surface_404"] = (
        fb.get(f"/v1/responses/{sid}").status_code == 404
        and fb.get("/v1/chat/completions/resp_deadbeef").status_code == 404
    )
    # a streamed call lands the assembled envelope under the same id
    st = fb.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "stream-me"}],
            "stream": True,
        },
    )
    st_id = next(
        _json3.loads(ln[6:])["id"]
        for ln in st.text.splitlines()
        if ln.startswith("data: ") and ln != "data: [DONE]"
    )
    st_env = fb.get(f"/v1/chat/completions/{st_id}")
    out["retrieve_stream_stored"] = (
        st.status_code == 200
        and st_env.status_code == 200
        and st_env.json()["choices"][0]["message"]["content"] == "clean:stream-me"
    )
    # n>1: one envelope (n choices) under one id
    n2 = fb.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "fan"}],
            "n": 2,
        },
    )
    n2_env = fb.get(f"/v1/chat/completions/{n2.json()['id']}")
    out["retrieve_n_fanout_single_envelope"] = (
        n2_env.status_code == 200 and len(n2_env.json()["choices"]) == 2
    )
    # batch lines store too — the output body's id fetches the same envelope
    bfid = _upload(fb)["id"]
    bcb = fb.post(
        "/v1/batches",
        json={"input_file_id": bfid, "endpoint": "/v1/chat/completions"},
    )
    bterm = _wait_batch(fb, bcb.json()["id"])
    bline0 = _json3.loads(
        fb.get(f"/v1/files/{bterm['output_file_id']}/content").text.splitlines()[0]
    )
    b_id = bline0["response"]["body"]["id"]
    out["retrieve_batch_line_stored"] = (
        bterm["status"] == "completed"
        and fb.get(f"/v1/chat/completions/{b_id}").json() == bline0["response"]["body"]
    )
    # an idempotent replay re-pins the envelope — still retrievable
    idem = fb.post(
        "/v1/chat/completions",
        json={"model": "fx1", "messages": [{"role": "user", "content": "idem"}]},
        headers={"Idempotency-Key": "lane80-idem"},
    )
    fb.post(
        "/v1/chat/completions",
        json={"model": "fx1", "messages": [{"role": "user", "content": "idem"}]},
        headers={"Idempotency-Key": "lane80-idem"},
    )
    out["retrieve_idem_replay_stored"] = (
        fb.get(f"/v1/chat/completions/{idem.json()['id']}").status_code == 200
    )
    # GET /v1/chat/completions/{id}/messages + /v1/responses/{id}/input_items
    # — OpenAI's stored-request subresources: the items the model ran on,
    # paged by deterministic item ids.
    ic = fb.post(
        "/v1/chat/completions",
        json={
            "model": "fx1",
            "messages": [
                {"role": "user", "content": "m-one"},
                {"role": "user", "content": "m-two"},
            ],
        },
    )
    ic_id = ic.json()["id"]
    imsgs = fb.get(f"/v1/chat/completions/{ic_id}/messages").json()
    out["items_chat_messages"] = (
        ic.status_code == 200
        and imsgs["object"] == "list"
        and [m["content"] for m in imsgs["data"]] == ["m-one", "m-two"]
        and all(m["id"].startswith("msg_") for m in imsgs["data"])
        and imsgs["first_id"] == imsgs["data"][0]["id"]
        and imsgs["has_more"] is False
    )
    page1 = fb.get(f"/v1/chat/completions/{ic_id}/messages?limit=1").json()
    page2 = fb.get(f"/v1/chat/completions/{ic_id}/messages?limit=1&after={page1['last_id']}").json()
    out["items_chat_paged"] = (
        page1["has_more"] is True
        and page2["data"][0]["content"] == "m-two"
        and page2["has_more"] is False
        and page2["first_id"] != page1["first_id"]
    )
    out["items_chat_order_desc"] = (
        fb.get(f"/v1/chat/completions/{ic_id}/messages?order=desc").json()["data"][0]["content"]
        == "m-two"
    )
    out["items_chat_cursor_400"] = (
        fb.get(f"/v1/chat/completions/{ic_id}/messages?after=msg_bogus").status_code == 400
        and fb.get(f"/v1/chat/completions/{ic_id}/messages?order=sideways").status_code == 422
    )
    # wrong-surface and unknown ids are misses; store=false never lists
    out["items_404s"] = (
        fb.get("/v1/chat/completions/chatcmpl-ghost/messages").status_code == 404
        and fb.get(f"/v1/responses/{ic_id}/input_items").status_code == 404
        and fb.get(f"/v1/responses/{r2.json()['id']}/input_items").status_code == 404
    )
    ri = fb.post("/v1/responses", json={"model": "fx1", "input": "itemize-me"})
    ritems = fb.get(f"/v1/responses/{ri.json()['id']}/input_items").json()
    out["items_response_input_items"] = (
        ritems["object"] == "list"
        and len(ritems["data"]) == 1
        and ritems["data"][0]["role"] == "user"
        and ritems["data"][0]["content"][0]["text"] == "itemize-me"
        and ritems["data"][0]["id"].startswith("msg_")
    )
    rlist = fb.post(
        "/v1/responses",
        json={
            "model": "fx1",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "l1"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "l2"}],
                },
            ],
        },
    )
    rit2 = fb.get(f"/v1/responses/{rlist.json()['id']}/input_items").json()
    out["items_response_list_input"] = (
        len(rit2["data"]) == 2
        and rit2["data"][1]["role"] == "assistant"
        and all(it["id"].startswith("msg_") for it in rit2["data"])
    )
    # the subresource dies with its envelope — no orphaned request history
    fb.delete(f"/v1/chat/completions/{ic_id}")
    out["items_die_with_envelope"] = (
        fb.get(f"/v1/chat/completions/{ic_id}/messages").status_code == 404
    )
    # LRU bound: store_max=2 evicts the oldest entry
    ev_app = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _OiBackend(), store_max=2))
    ev_ids = [
        ev_app.post(
            "/v1/chat/completions",
            json={"model": "fx1", "messages": [{"role": "user", "content": f"ev{i}"}]},
        ).json()["id"]
        for i in range(3)
    ]
    out["retrieve_store_max_evicts"] = (
        ev_app.get(f"/v1/chat/completions/{ev_ids[0]}").status_code == 404
        and ev_app.get(f"/v1/chat/completions/{ev_ids[1]}").status_code == 200
        and ev_app.get(f"/v1/chat/completions/{ev_ids[2]}").status_code == 200
    )
    out["items_evict_with_envelope"] = (
        ev_app.get(f"/v1/chat/completions/{ev_ids[0]}/messages").status_code == 404
    )
    # GET /v1/chat/completions — OpenAI's stored-completion list surface:
    # paged by completion id, filtered by model and metadata subset.
    l_ids = [
        fb.post(
            "/v1/chat/completions",
            json={
                "model": "fx1-listprobe",
                # an arbitrary model name now refuses without a stated link —
                # pin the byok lane explicitly so the list filter keeps a
                # distinct model stamp.
                "fx1": {"backend": "byok"},
                "messages": [{"role": "user", "content": f"lp{i}"}],
                "metadata": {"lane": "list-probe", "kind": f"k{i % 2}"},
            },
        ).json()["id"]
        for i in range(3)
    ]
    l_all = fb.get("/v1/chat/completions?metadata[lane]=list-probe&limit=50").json()
    l_model = l_all["data"][0]["model"]
    out["list_chat_basic"] = (
        l_all["object"] == "list"
        and [d["id"] for d in l_all["data"]] == l_ids
        and all(d["object"] == "chat.completion" for d in l_all["data"])
        and all(
            d["metadata"] == {"lane": "list-probe", "kind": f"k{i % 2}"}
            for i, d in enumerate(l_all["data"])
        )
        and l_all["first_id"] == l_ids[0]
        and l_all["has_more"] is False
    )
    lp1 = fb.get("/v1/chat/completions?metadata[lane]=list-probe&limit=2").json()
    lp2 = fb.get(
        f"/v1/chat/completions?metadata[lane]=list-probe&limit=2&after={lp1['last_id']}"
    ).json()
    out["list_chat_paged"] = (
        lp1["has_more"] is True and lp2["data"][0]["id"] == l_ids[2] and lp2["has_more"] is False
    )
    out["list_chat_desc"] = (
        fb.get("/v1/chat/completions?metadata[lane]=list-probe&order=desc").json()["data"][0]["id"]
        == l_ids[2]
    )
    l_meta = fb.get("/v1/chat/completions?metadata[lane]=list-probe&metadata[kind]=k1").json()
    out["list_chat_metadata"] = [d["id"] for d in l_meta["data"]] == [l_ids[1]]
    out["list_chat_model"] = [
        d["id"]
        for d in fb.get(f"/v1/chat/completions?model={l_model}&metadata[lane]=list-probe").json()[
            "data"
        ]
    ] == l_ids and fb.get(
        "/v1/chat/completions?model=fx1-none-such&metadata[lane]=list-probe"
    ).json()["data"] == []
    out["list_chat_edges"] = (
        fb.get("/v1/chat/completions?after=chatcmpl-ghost").status_code == 400
        and fb.get("/v1/chat/completions?metadata[lane]=none-such").json()["data"] == []
        and fb.get("/v1/chat/completions?order=sideways").status_code == 422
    )

    # previous_response_id — OpenAI's stateful-agent primitive: the
    # child's effective input is the parent's stored items + the parent's
    # output + this request's input, and the whole history lands on the
    # child's stored item list.
    class _ChainBackend(_OiBackend):
        seen: list[list[dict[str, str]]] = []

        def complete(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> str:
            _ChainBackend.seen.append(list(messages))
            return super().complete(messages, sampling=sampling)

    ch = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _ChainBackend()))
    ch1 = ch.post("/v1/responses", json={"model": "fx1", "input": "chain-one"})
    ch2 = ch.post(
        "/v1/responses",
        json={
            "model": "fx1",
            "input": "chain-two",
            "previous_response_id": ch1.json()["id"],
        },
    )
    ch2_items = ch.get(f"/v1/responses/{ch2.json()['id']}/input_items").json()
    out["resp_chain_ok"] = (
        ch1.status_code == 200
        and ch2.status_code == 200
        and ch1.json()["previous_response_id"] is None
        and ch2.json()["previous_response_id"] == ch1.json()["id"]
        and ch2.json()["output"][0]["content"][0]["text"] == "clean:chain-two"
        # the model actually ran on the history, not just the new turn
        and [m["role"] for m in _ChainBackend.seen[-1]] == ["user", "assistant", "user"]
        and _ChainBackend.seen[-1][1]["content"] == "clean:chain-one"
        # the stored item list is the full chain, deterministic ids
        and [it.get("role") for it in ch2_items["data"]] == ["user", "assistant", "user"]
        and [it["content"][0]["type"] for it in ch2_items["data"]]
        == ["input_text", "output_text", "input_text"]
        and all(it["id"].startswith("msg_") for it in ch2_items["data"])
    )
    # a three-hop chain keeps growing the stored list; deleting the
    # parent can't orphan the child's self-contained items
    ch3 = ch.post(
        "/v1/responses",
        json={
            "model": "fx1",
            "input": "chain-three",
            "previous_response_id": ch2.json()["id"],
        },
    )
    ch3_items = ch.get(f"/v1/responses/{ch3.json()['id']}/input_items").json()
    ch.delete(f"/v1/responses/{ch1.json()['id']}")
    out["resp_chain_multihop_selfcontained"] = (
        ch3.status_code == 200
        and len(ch3_items["data"]) == 5
        and ch.get(f"/v1/responses/{ch3.json()['id']}/input_items").status_code == 200
        and ch.get(f"/v1/responses/{ch1.json()['id']}").status_code == 404
    )
    # fail closed: unknown parent, a non-response envelope, and a
    # store=false parent all refuse before the model runs
    ch_ns = ch.post(
        "/v1/responses", json={"model": "fx1", "input": "nostore", "store": False}
    ).json()
    ch_cc = ch.post(
        "/v1/chat/completions",
        json={"model": "fx1", "messages": [{"role": "user", "content": "cmpl"}]},
    ).json()
    ch_seen_pre = len(_ChainBackend.seen)
    out["resp_chain_fail_closed"] = (
        ch.post(
            "/v1/responses",
            json={"model": "fx1", "input": "x", "previous_response_id": "resp_ghost"},
        ).json()["error"]["code"]
        == "previous_response_not_found"
        and ch.post(
            "/v1/responses",
            json={
                "model": "fx1",
                "input": "x",
                "previous_response_id": ch_ns["id"],
            },
        ).json()["error"]["code"]
        == "previous_response_not_found"
        and ch.post(
            "/v1/responses",
            json={
                "model": "fx1",
                "input": "x",
                "previous_response_id": ch_cc["id"],
            },
        ).json()["error"]["code"]
        == "previous_response_not_found"
        and len(_ChainBackend.seen) == ch_seen_pre
    )
    # the chained stream replays identically (terminal response.completed)
    ch_s = ch.post(
        "/v1/responses",
        json={
            "model": "fx1",
            "input": "chain-stream",
            "previous_response_id": ch2.json()["id"],
            "stream": True,
        },
    )
    out["resp_chain_stream"] = (
        ch_s.status_code == 200
        and "event: response.completed" in ch_s.text
        and _ChainBackend.seen[-1][-1]["content"] == "chain-stream"
    )
    # background:true — OpenAI's long-running-call primitive: the POST
    # returns a queued response object at once; the job executor runs the
    # model under the same id and a stored GET flips to terminal.
    bg = ch.post("/v1/responses", json={"model": "fx1", "input": "bg-run", "background": True})
    bg_env = bg.json()
    bg_fin: dict[str, Any] = {}
    for _ in range(500):
        bg_fin = ch.get(f"/v1/responses/{bg_env['id']}").json()
        if bg_fin["status"] in ("completed", "failed", "cancelled", "incomplete"):
            break
        time.sleep(0.01)
    bg_items = ch.get(f"/v1/responses/{bg_env['id']}/input_items").json()
    out["resp_background_lifecycle"] = (
        bg.status_code == 200
        and bg_env["status"] == "queued"
        and bg_env["output"] == []
        and bg_fin["status"] == "completed"
        and bg_fin["output"][0]["content"][0]["text"] == "clean:bg-run"
        and bg_fin["id"] == bg_env["id"]
        and bg_fin["created_at"] == bg_env["created_at"]
        and bg_items["data"][0]["content"][0]["text"] == "bg-run"
        and _ChainBackend.seen[-1][-1]["content"] == "bg-run"
    )

    # cancel: a still-running background job flips to cancelled — the
    # cancel verdict wins over the late model result; terminal responses
    # refuse 409; unknown ids 404.
    class _SlowBackend(_OiBackend):
        def complete(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> str:
            time.sleep(0.3)
            return super().complete(messages, sampling=sampling)

    ch_slow = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _SlowBackend()))
    sbg = ch_slow.post(
        "/v1/responses", json={"model": "fx1", "input": "bg-slow", "background": True}
    ).json()
    cx = ch_slow.post(f"/v1/responses/{sbg['id']}/cancel")
    cx_again = ch_slow.post(f"/v1/responses/{sbg['id']}/cancel")
    sx_fin: dict[str, Any] = {}
    for _ in range(500):
        sx_fin = ch_slow.get(f"/v1/responses/{sbg['id']}").json()
        if sx_fin["status"] == "cancelled" and sx_fin.get("output") is not None:
            # let the worker settle — the cancel verdict must survive it
            time.sleep(0.4)
            sx_fin = ch_slow.get(f"/v1/responses/{sbg['id']}").json()
            break
        time.sleep(0.01)
    out["resp_background_cancel"] = (
        sbg["status"] == "queued"
        and cx.status_code == 200
        and cx.json()["status"] == "cancelled"
        and cx_again.status_code == 409
        and cx_again.json()["error"]["code"] == "cancel_terminal"
        and sx_fin["status"] == "cancelled"
        and ch_slow.post("/v1/responses/resp_ghost/cancel").status_code == 404
    )
    # fail closed: background needs store (it IS the retrieval surface);
    # a ghost chain parent fails at submit, not in the worker; a batch
    # line carrying background is a per-line error, not a nested async.
    bg_ns = ch.post(
        "/v1/responses",
        json={"model": "fx1", "input": "x", "background": True, "store": False},
    )
    bg_chain = ch.post(
        "/v1/responses",
        json={
            "model": "fx1",
            "input": "x",
            "background": True,
            "previous_response_id": "resp_ghost",
        },
    )
    bgb_lines = (
        _json3.dumps(
            {
                "custom_id": "bg-line",
                "method": "POST",
                "url": "/v1/responses",
                "body": {"model": "fx1", "input": "x", "background": True},
            }
        )
        + "\n"
    ).encode()
    bgb_up = fb.post(
        "/v1/files",
        files={"file": ("bg.jsonl", bgb_lines, "application/jsonl")},
        data={"purpose": "batch"},
    ).json()
    bgb = fb.post(
        "/v1/batches",
        json={
            "input_file_id": bgb_up["id"],
            "endpoint": "/v1/responses",
            "completion_window": "24h",
        },
    ).json()
    bgb_term = _wait_batch(fb, bgb["id"])
    bgb_out = fb.get(f"/v1/files/{bgb_term['output_file_id']}/content").text
    out["resp_background_fail_closed"] = (
        bg_ns.status_code == 400
        and bg_ns.json()["error"]["code"] == "background_requires_store"
        and bg_chain.status_code == 400
        and bg_chain.json()["error"]["code"] == "previous_response_not_found"
        and bgb_term["status"] == "completed"
        and _json3.loads(bgb_out.strip())["response"]["status_code"] == 400
        and _json3.loads(bgb_out.strip())["response"]["body"]["error"]["code"] == "invalid_request"
    )
    # an idempotent replay of a background submit returns the LIVE
    # envelope — the queued snapshot in the idem record is refreshed by
    # the worker's completion re-put, so a replay after completion lands
    # the terminal object. The replay may beat the worker's idem re-put
    # by a tick — the record first reports the live envelope's status
    # (freshness merge), then carries the completion id once the worker
    # re-pins it; poll the replay itself until the terminal record lands.
    bg_idem = ch.post(
        "/v1/responses",
        json={"model": "fx1", "input": "bg-idem", "background": True},
        headers={"Idempotency-Key": "bg-idem-1"},
    )
    for _ in range(500):
        if ch.get(f"/v1/responses/{bg_idem.json()['id']}").json()["status"] == "completed":
            break
        time.sleep(0.01)
    bg_rep = None
    for _ in range(500):
        cand = ch.post(
            "/v1/responses",
            json={"model": "fx1", "input": "bg-idem", "background": True},
            headers={"Idempotency-Key": "bg-idem-1"},
        )
        if cand.json().get("status") == "completed" and cand.headers.get("X-Fx1-Completion-Id"):
            bg_rep = cand
            break
        time.sleep(0.01)
    out["resp_background_idem_replay"] = (
        bg_rep is not None
        and bg_rep.headers.get("X-Fx1-Idempotent-Replay") == "true"
        and bg_rep.json()["id"] == bg_idem.json()["id"]
        and bg_rep.json()["status"] == "completed"
        and bg_rep.headers.get("X-Fx1-Completion-Id") is not None
    )

    # GET /v1/responses/{id}?stream=true — OpenAI's response replay: the
    # stored envelope re-emits the recorded event grammar so a client
    # that dropped the create stream (or submitted background:true,
    # which answers JSON) rebuilds the same typed Response. Frames carry
    # the create-time seq on the id: line — monotonic from 0 — and each
    # event's data payload keeps the typed {"type", "response"} shape a
    # stock SDK stream parser reads.
    def _replay_frames(rp: Any) -> tuple[list[str], list[int], list[dict[str, Any]]]:
        lines = rp.text.splitlines()
        return (
            [ln[len("event: ") :] for ln in lines if ln.startswith("event: ")],
            [int(ln[len("id: ") :]) for ln in lines if ln.startswith("id: ")],
            [_json3.loads(ln[len("data: ") :]) for ln in lines if ln.startswith("data: ")],
        )

    rp_create = ch.post(
        "/v1/responses", json={"model": "fx1", "input": "replay-me", "stream": True}
    )
    rp_rid = _json3.loads(
        next(ln for ln in rp_create.text.splitlines() if ln.startswith("data: "))[len("data: ") :]
    )["response"]["id"]
    rp_get = ch.get(f"/v1/responses/{rp_rid}?stream=true")
    rp_events, rp_ids, rp_data = _replay_frames(rp_get)
    out["resp_replay_completed_grammar"] = (
        rp_get.status_code == 200
        and rp_get.headers["content-type"].startswith("text/event-stream")
        and rp_events[:2] == ["response.created", "response.in_progress"]
        and rp_events[-1] == "response.completed"
        and "response.output_item.added" in rp_events
        and "response.output_text.delta" in rp_events
        and rp_ids == list(range(len(rp_ids)))
        and rp_data[-1]["type"] == "response.completed"
        and rp_data[-1]["response"]["id"] == rp_rid
        and rp_data[-1]["response"]["status"] == "completed"
    )
    # a replay of a stream-created response is byte-identical to the
    # stream it replays — the grammar derives from the stored object, not
    # a second code path; the terminal payload IS the non-stream body.
    out["resp_replay_byte_identical"] = rp_get.text == rp_create.text
    out["resp_replay_terminal_is_retrieve"] = (
        rp_data[-1]["response"] == ch.get(f"/v1/responses/{rp_rid}").json()
    )
    # evidence + trace headers ride the replay like every gated call
    out["resp_replay_headers"] = (
        rp_get.headers.get("x-request-id") is not None
        and int(rp_get.headers.get("openai-processing-ms", "-1")) >= 0
        and rp_get.headers.get("X-Fx1-Completion-Id") is not None
        and rp_get.headers.get("X-Fx1-Receipt-Sha256") is not None
    )
    # starting_after=N resumes past sequence N — the cursor is the frame's
    # id: line; past the end the stream closes empty.
    rp_slice = ch.get(f"/v1/responses/{rp_rid}?stream=true&starting_after=2")
    _rp_sev, _rp_sid, _rp_sdata = _replay_frames(rp_slice)
    rp_past = ch.get(f"/v1/responses/{rp_rid}?stream=true&starting_after=9999")
    out["resp_replay_starting_after"] = (
        rp_slice.status_code == 200
        and _rp_sid == list(range(3, len(rp_ids)))
        and _rp_sev[0] != "response.created"
        and _rp_sev[-1] == "response.completed"
        and len(_rp_sdata) == len(rp_data) - 3
    )
    out["resp_replay_starting_after_beyond"] = (
        rp_past.status_code == 200 and not _replay_frames(rp_past)[0]
    )
    # unknown / deleted / store=false ids answer the same 404 not_found
    # envelope as the JSON read — a replay never invents a record.
    rp_ns = ch.post("/v1/responses", json={"model": "fx1", "input": "ns", "store": False})
    rp_del = ch.post("/v1/responses", json={"model": "fx1", "input": "del"})
    ch.delete(f"/v1/responses/{rp_del.json()['id']}")
    out["resp_replay_not_found"] = (
        ch.get("/v1/responses/resp_ghost?stream=true").status_code == 404
        and ch.get("/v1/responses/resp_ghost?stream=true").json()["error"]["code"] == "not_found"
        and ch.get(f"/v1/responses/{rp_del.json()['id']}?stream=true").status_code == 404
        and ch.get(f"/v1/responses/{rp_ns.json()['id']}?stream=true").status_code == 404
    )
    # flag fallthrough: stream=false (or absent) is the JSON retrieve;
    # the truthy spellings pydantic accepts (1/yes/on) all stream.
    rp_false = ch.get(f"/v1/responses/{rp_rid}?stream=false")
    out["resp_replay_flag_fallthrough"] = (
        rp_false.status_code == 200
        and rp_false.headers["content-type"].startswith("application/json")
        and rp_false.json()["id"] == rp_rid
        and ch.get(f"/v1/responses/{rp_rid}?stream=1")
        .headers["content-type"]
        .startswith("text/event-stream")
        and ch.get(f"/v1/responses/{rp_rid}?stream=yes").status_code == 200
    )
    # a completed background envelope replays the recorded lifecycle —
    # response.queued is a real event on this surface.
    bg_rp = ch.get(f"/v1/responses/{bg_env['id']}?stream=true")
    _bg_ev, _bg_i, _bg_d = _replay_frames(bg_rp)
    out["resp_replay_background_grammar"] = (
        bg_rp.status_code == 200
        and _bg_ev[:3] == ["response.created", "response.queued", "response.in_progress"]
        and _bg_ev[-1] == "response.completed"
        and _bg_d[-1]["response"]["status"] == "completed"
        and _bg_d[-1]["response"]["output"][0]["content"][0]["text"] == "clean:bg-run"
    )

    # in-flight attach: a still-running background response emits its
    # prelude then live-follows — the : keepalive comments prove the
    # stream observed a non-terminal record before the terminal frame.
    class _ReplaySlowBackend(_OiBackend):
        def complete(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> str:
            time.sleep(0.8)
            return super().complete(messages, sampling=sampling)

    rp_app = api_mod.create_app(backend_resolver=lambda *a, **k: _ReplaySlowBackend())
    rp_app.state.sse_keepalive_s = 0.05
    rp_follow = _TC2(rp_app)
    rfa = rp_follow.post(
        "/v1/responses", json={"model": "fx1", "input": "rfa", "background": True}
    ).json()
    rfa_rep = rp_follow.get(f"/v1/responses/{rfa['id']}?stream=true&timeout_s=10")
    _rfa_ev, _rfa_i, _rfa_d = _replay_frames(rfa_rep)
    out["resp_replay_inflight_attach"] = (
        rfa_rep.status_code == 200
        and _rfa_ev[:2] == ["response.created", "response.queued"]
        and _rfa_ev[-1] == "response.completed"
        and _rfa_i == list(range(len(_rfa_i)))
        and any(ln.startswith(":") for ln in rfa_rep.text.splitlines())
        and _rfa_d[-1]["response"]["status"] == "completed"
    )
    # drain/cancel interplay: replaying a cancelled record ends with the
    # response.cancelled frame; the cancelled object is the terminal
    # payload verbatim. (sbg was cancelled in resp_background_cancel.)
    cx_rep = ch_slow.get(f"/v1/responses/{sbg['id']}?stream=true")
    _cx_ev, _cx_i, _cx_d = _replay_frames(cx_rep)
    out["resp_replay_cancelled_terminal"] = (
        cx_rep.status_code == 200
        and _cx_ev[-1] == "response.cancelled"
        and _cx_d[-1]["response"]["status"] == "cancelled"
        and _cx_d[-1]["response"]["id"] == sbg["id"]
        and _cx_i == list(range(len(_cx_i)))
    )

    # a failed background job replays to response.failed — the recorded
    # error object is the terminal payload's error field.
    class _ReplayBoomBackend:
        def complete(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> str:
            raise RuntimeError("replay boom")

    rp_boom = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _ReplayBoomBackend()))
    rfb = rp_boom.post(
        "/v1/responses", json={"model": "fx1", "input": "rfb", "background": True}
    ).json()
    rfb_fin: dict[str, Any] = {}
    for _ in range(500):
        rfb_fin = rp_boom.get(f"/v1/responses/{rfb['id']}").json()
        if rfb_fin["status"] == "failed":
            break
        time.sleep(0.01)
    rfb_rep = rp_boom.get(f"/v1/responses/{rfb['id']}?stream=true")
    _rfb_ev, _rfb_i, _rfb_d = _replay_frames(rfb_rep)
    out["resp_replay_failed_terminal"] = (
        rfb_fin["status"] == "failed"
        and _rfb_ev[-1] == "response.failed"
        and _rfb_d[-1]["response"]["status"] == "failed"
        and _rfb_d[-1]["response"]["error"]["code"] == "backend_failure"
    )

    # /v1/conversations — the named-container twin of
    # previous_response_id: a conv_* carries an accumulated item stream;
    # a response anchored to it runs on the conv context and appends its
    # own turn back.
    cv = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _ChainBackend()))
    cv0 = cv.post(
        _CONVERSATIONS_URL,
        json={
            "items": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "seed-q"}],
                }
            ],
            "metadata": {"lane": "conv"},
        },
    )
    cid0 = cv0.json()["id"]
    cv_seen_pre = len(_ChainBackend.seen)
    cv_r1 = cv.post(
        "/v1/responses", json={"model": "fx1", "input": "turn-one", "conversation": cid0}
    )
    cv_items = cv.get(f"/v1/conversations/{cid0}/items").json()
    out["conv_lifecycle"] = (
        cv0.status_code == 200
        and cv0.json()["object"] == "conversation"
        and cid0.startswith("conv_")
        and cv0.json()["metadata"] == {"lane": "conv"}
        and cv.get(f"/v1/conversations/{cid0}").json()["id"] == cid0
        # the seeded item + the turn's input/output all landed
        and [it["content"][0]["text"] for it in cv_items["data"]][:3]
        == ["seed-q", "turn-one", "clean:turn-one"]
        and cv_r1.status_code == 200
        and cv_r1.json()["conversation"] == {"id": cid0}
        # the model ran on the conv's context, not the bare input
        and [m["role"] for m in _ChainBackend.seen[cv_seen_pre]] == ["user", "user"]
        and _ChainBackend.seen[cv_seen_pre][0]["content"] == "seed-q"
    )
    # the conv accumulates across turns; a second turn sees turn-one's
    # output as assistant history; item delete drops exactly one
    cv_r2 = cv.post(
        "/v1/responses", json={"model": "fx1", "input": "turn-two", "conversation": {"id": cid0}}
    )
    cv_items2 = cv.get(f"/v1/conversations/{cid0}/items").json()["data"]
    drop = cv.delete(f"/v1/conversations/{cid0}/items/{cv_items2[0]['id']}")
    cv_items3 = cv.get(f"/v1/conversations/{cid0}/items").json()["data"]
    out["conv_turn_accumulates"] = (
        cv_r2.status_code == 200
        and len(cv_items2) == 5
        and _ChainBackend.seen[-1][-1]["content"] == "turn-two"
        and [m["role"] for m in _ChainBackend.seen[-1]][-2] == "assistant"
        and drop.status_code == 200
        and drop.json()["id"] == cid0
        and len(cv_items3) == 4
        and cv_items3[0]["content"][0]["text"] == "turn-one"
    )
    # single-item retrieve — OpenAI's conversations.items.retrieve: hit
    # returns the item, a miss inside a live conv or a ghost conv 404s
    got_item = cv.get(f"/v1/conversations/{cid0}/items/{cv_items3[0]['id']}")
    out["conv_item_get"] = (
        got_item.status_code == 200
        and got_item.json()["id"] == cv_items3[0]["id"]
        and cv.get(f"/v1/conversations/{cid0}/items/msg_ghost").status_code == 404
        and cv.get("/v1/conversations/conv_ghost/items/msg_x").status_code == 404
    )
    # conv is its own store: a store:false response still appends its
    # turn to the conv even though the envelope itself never indexes
    cv_ns = cv.post(
        "/v1/responses",
        json={"model": "fx1", "input": "ghost-turn", "conversation": cid0, "store": False},
    ).json()
    out["conv_is_own_store"] = (
        cv_ns["id"].startswith("resp_")
        and cv.get(f"/v1/responses/{cv_ns['id']}").status_code == 404
        and cv.get(f"/v1/conversations/{cid0}/items?limit=100").json()["data"][-1]["content"][0][
            "text"
        ]
        == "clean:ghost-turn"
    )
    # fail closed: unknown conv id 400s before the model runs;
    # conversation + previous_response_id is a 422 validation pair; a
    # batch line can't anchor to a shared container
    cv2_seen_pre = len(_ChainBackend.seen)
    out["conv_fail_closed"] = (
        cv.post(
            "/v1/responses",
            json={"model": "fx1", "input": "x", "conversation": "conv_ghost"},
        ).json()["error"]["code"]
        == "conversation_not_found"
        and cv.post(
            "/v1/responses",
            json={
                "model": "fx1",
                "input": "x",
                "conversation": cid0,
                "previous_response_id": cv_r1.json()["id"],
            },
        ).status_code
        == 422
        and cv.post(
            "/v1/responses", json={"model": "fx1", "input": "x", "conversation": {"id": 7}}
        ).status_code
        == 422
        and len(_ChainBackend.seen) == cv2_seen_pre
        and cv.delete(f"/v1/conversations/{cid0}/items/msg_ghost").status_code == 404
        and cv.get(f"/v1/conversations/{cid0}/items?after=msg_ghost").status_code == 400
        and cv.get("/v1/conversations/conv_ghost").status_code == 404
        and cv.get("/v1/conversations/conv_ghost/items").status_code == 404
    )
    # a deleted conv orphans nothing — its member responses still GET,
    # and the deleted conv itself refuses a turn join
    cv_del = cv.delete(f"/v1/conversations/{cid0}")
    out["conv_delete"] = (
        cv_del.status_code == 200
        and cv_del.json()["object"] == "conversation.deleted"
        and cv_del.json()["deleted"] is True
        and cv.get(f"/v1/conversations/{cid0}").status_code == 404
        and cv.get(f"/v1/responses/{cv_r1.json()['id']}").status_code == 200
        and cv.post(
            "/v1/responses", json={"model": "fx1", "input": "x", "conversation": cid0}
        ).json()["error"]["code"]
        == "conversation_not_found"
    )
    # background + conv: submit validates the conv, the worker appends
    cvb = cv.post(_CONVERSATIONS_URL, json={})
    cvb_id = cvb.json()["id"]
    cvb_r = cv.post(
        "/v1/responses",
        json={"model": "fx1", "input": "bg-conv", "conversation": cvb_id, "background": True},
    )
    for _ in range(500):
        if cv.get(f"/v1/responses/{cvb_r.json()['id']}").json()["status"] == "completed":
            break
        time.sleep(0.01)
    cvb_items = cv.get(f"/v1/conversations/{cvb_id}/items").json()["data"]
    out["conv_background_turn"] = (
        cvb_r.status_code == 200
        and cvb_r.json()["status"] == "queued"
        and len(cvb_items) == 2
        and cvb_items[-1]["content"][0]["text"] == "clean:bg-conv"
        and cv.post(
            "/v1/responses",
            json={
                "model": "fx1",
                "input": "x",
                "conversation": "conv_ghost",
                "background": True,
            },
        ).json()["error"]["code"]
        == "conversation_not_found"
    )

    # ---- lane 112: max_tool_calls ----
    # A caller's safety bound must not evaporate into ``extra="allow"``:
    # over the cap the emitted call list truncates at the bound and the
    # response lands ``status: 'incomplete'`` with
    # ``incomplete_details.reason == 'max_tool_calls'`` — OpenAI's own
    # truncation semantics, never a silent drop. The stream's terminal
    # frame is ``response.incomplete``; the stored object and its conv
    # append keep the truncation; at/under the cap the turn completes.
    # /v1/responses takes the flattened Responses tool spec (the nested
    # chat shape is a wire 422)
    mt_req = {
        "model": "fx1",
        "input": "calls",
        "tools": [
            {
                "type": "function",
                "name": "calc",
                "description": "arithmetic",
                "parameters": {"type": "object"},
            }
        ],
        "max_tool_calls": 2,
    }
    mt = oi_tool3_app.post("/v1/responses", json=mt_req)
    mt_b = mt.json()
    mt_out = mt_b.get("output") or []
    mt_fc = [it for it in mt_out if it.get("type") == "function_call"]
    mt_st = oi_tool3_app.get(f"/v1/responses/{mt_b['id']}").json()
    out["resp_max_tool_calls_incomplete"] = (
        mt.status_code == 200
        and mt_b["status"] == "incomplete"
        and mt_b["incomplete_details"] == {"reason": "max_tool_calls"}
        and mt_b["max_tool_calls"] == 2
        and len(mt_out) == 2
        and [it["call_id"] for it in mt_fc] == ["call_0", "call_1"]
        # the truncated turn has no prose — no phantom empty message item
        and not any(it.get("type") == "message" for it in mt_out)
        and mt_st["status"] == "incomplete"
        and len(mt_st["output"]) == 2
    )
    mt0 = oi_tool3_app.post("/v1/responses", json={**mt_req, "max_tool_calls": 0}).json()
    out["resp_max_tool_calls_zero"] = (
        mt0["status"] == "incomplete"
        and mt0["incomplete_details"]["reason"] == "max_tool_calls"
        and mt0["output"] == []
    )
    mt_at = oi_tool3_app.post("/v1/responses", json={**mt_req, "max_tool_calls": 3}).json()
    mt_off = oi_tool3_app.post(
        "/v1/responses",
        json={k: v for k, v in mt_req.items() if k != "max_tool_calls"},
    ).json()
    out["resp_max_tool_calls_at_or_off"] = (
        mt_at["status"] == "completed"
        and len(mt_at["output"]) == 3
        and mt_at["incomplete_details"] is None
        and mt_off["status"] == "completed"
        and len(mt_off["output"]) == 3
        and mt_off["max_tool_calls"] is None
    )
    out["resp_max_tool_calls_422"] = (
        oi_tool3_app.post("/v1/responses", json={**mt_req, "max_tool_calls": -1}).status_code == 422
    )
    mt_s = oi_tool3_app.post("/v1/responses", json={**mt_req, "stream": True})
    out["resp_max_tool_calls_stream_incomplete"] = (
        mt_s.status_code == 200
        and "event: response.incomplete" in mt_s.text
        and "event: response.completed" not in mt_s.text
    )
    # a batch line honours the per-line cap — the output-file body is the
    # same truncated envelope
    mt_batch = (
        _json3.dumps(
            {
                "custom_id": "cap2",
                "method": "POST",
                "url": "/v1/responses",
                "body": {**mt_req, "model": "fx1"},
            }
        )
        + "\n"
    ).encode()
    mt_up = oi_tool3_app.post(
        "/v1/files",
        files={"file": ("cap.jsonl", mt_batch, "application/jsonl")},
        data={"purpose": "batch"},
    ).json()
    mt_bc = oi_tool3_app.post(
        "/v1/batches",
        json={"input_file_id": mt_up["id"], "endpoint": "/v1/responses"},
    ).json()
    mt_bterm = _wait_batch(oi_tool3_app, mt_bc["id"])
    mt_line = _json3.loads(
        oi_tool3_app.get(f"/v1/files/{mt_bterm['output_file_id']}/content").text.strip()
    )
    out["resp_max_tool_calls_batch_line"] = (
        mt_bterm["status"] == "completed"
        and mt_line["response"]["status_code"] == 200
        and mt_line["response"]["body"]["status"] == "incomplete"
        and mt_line["response"]["body"]["incomplete_details"]["reason"] == "max_tool_calls"
        and len(mt_line["response"]["body"]["output"]) == 2
    )
    # the conv trail records the truncation, not a fake completion
    mt_conv = oi_tool3_app.post(_CONVERSATIONS_URL, json={}).json()["id"]
    mt_cv = oi_tool3_app.post("/v1/responses", json={**mt_req, "conversation": mt_conv}).json()
    mt_cv_items = oi_tool3_app.get(f"/v1/conversations/{mt_conv}/items").json()["data"]
    out["resp_max_tool_calls_conv_append"] = (
        mt_cv["status"] == "incomplete"
        and len([it for it in mt_cv_items if it.get("type") == "function_call"]) == 2
        # the only message item is the request's user input — no phantom
        # assistant message fabricated by the truncation
        and not any(
            it.get("type") == "message" and it.get("role") == "assistant" for it in mt_cv_items
        )
    )

    # capabilities advertises the index bound + flag
    caps = fb.get("/harness/capabilities").json()
    out["capabilities_retrieval"] = (
        caps["features"]["openai_retrieval"] is True and int(caps["limits"]["store_max"]) == 256
    )
    out["capabilities_vector_stores"] = (
        caps["features"]["openai_vector_stores"] is True
        and caps["features"]["openai_file_search"] is True
        and int(caps["limits"]["vs_store_max"]) == 256
        and int(caps["limits"]["vs_file_max"]) == 32
        and int(caps["limits"]["vs_max_results"]) == 50
    )

    # --- /v1/vector_stores + server-side file_search -----------------------
    vs_up = fb.post(
        "/v1/files",
        files={"file": ("vs.jsonl", b"alpha beta gamma delta", "application/jsonl")},
        data={"purpose": "batch"},
    ).json()
    vs = fb.post("/v1/vector_stores", json={"name": "kb"}).json()
    out["vs_create"] = vs["object"] == "vector_store" and vs["id"].startswith("vs_")
    vs_id = str(vs["id"])
    out["vs_retrieve_404"] = fb.get("/v1/vector_stores/vs_nope").status_code == 404
    out["vs_update"] = (
        fb.post(f"/v1/vector_stores/{vs_id}", json={"name": "kb2"}).json()["name"] == "kb2"
    )
    vs_list = fb.get("/v1/vector_stores", params={"limit": 1}).json()
    out["vs_list_page"] = (
        vs_list["object"] == "list"
        and vs_list["data"][0]["id"] == vs_id
        and vs_list["has_more"] is False
    )
    vf = fb.post(f"/v1/vector_stores/{vs_id}/files", json={"file_id": vs_up["id"]})
    out["vs_file_attach"] = vf.status_code == 200 and vf.json()["status"] == "completed"
    out["vs_file_attach_409"] = (
        fb.post(f"/v1/vector_stores/{vs_id}/files", json={"file_id": vs_up["id"]}).status_code
        == 409
    )
    out["vs_file_unknown_vs_404"] = (
        fb.post("/v1/vector_stores/vs_nope/files", json={"file_id": vs_up["id"]}).status_code == 404
    )
    out["vs_file_unknown_file_404"] = (
        fb.post(f"/v1/vector_stores/{vs_id}/files", json={"file_id": "file-nope"}).status_code
        == 404
    )
    vfiles = fb.get(f"/v1/vector_stores/{vs_id}/files").json()
    out["vs_files_list"] = vfiles["object"] == "list" and vfiles["data"][0]["id"] == vs_up["id"]
    out["vs_files_filter_400"] = (
        fb.get(f"/v1/vector_stores/{vs_id}/files", params={"filter": "bogus"}).status_code == 400
    )
    out["vs_files_filter_ok"] = (
        fb.get(f"/v1/vector_stores/{vs_id}/files", params={"filter": "completed"}).json()["data"][
            0
        ]["id"]
        == vs_up["id"]
        and fb.get(f"/v1/vector_stores/{vs_id}/files", params={"filter": "failed"}).json()["data"]
        == []
    )
    vcontent = fb.get(f"/v1/vector_stores/{vs_id}/files/{vs_up['id']}/content")
    out["vs_file_content"] = (
        vcontent.status_code == 200
        and vcontent.json()["object"] == "vector_store.file_content.page"
        and vcontent.json()["data"][0]["type"] == "text"
        and "alpha" in vcontent.json()["data"][0]["text"]
    )
    out["vs_file_get"] = (
        fb.get(f"/v1/vector_stores/{vs_id}/files/{vs_up['id']}").json()["status"] == "completed"
    )
    # the file_search tool turn: retrieval precedes the message, include
    # gates the results block, the injected context lands in input_items
    fs_req = {
        "model": "fx1",
        "input": "what is alpha",
        "tools": [{"type": "file_search", "vector_store_ids": [vs_id]}],
        "include": ["file_search_call.results"],
    }
    fs_r = fb.post("/v1/responses", json=fs_req)
    fs_env = fs_r.json()
    fs_items = [o for o in fs_env["output"] if o["type"] == "file_search_call"]
    out["resp_file_search"] = (
        fs_r.status_code == 200
        and fs_items
        and fs_items[0]["status"] == "completed"
        and fs_items[0]["queries"] == ["what is alpha"]
        and fs_items[0]["results"][0]["file_id"] == vs_up["id"]
        and fs_items[0]["results"][0]["filename"] == "vs.jsonl"
        and fs_env["output"][-1]["type"] == "message"
        and fs_env["output"].index(fs_items[0]) < len(fs_env["output"]) - 1
    )
    fs_no_inc = fb.post(
        "/v1/responses", json={k: v for k, v in fs_req.items() if k != "include"}
    ).json()
    out["resp_file_search_include_gate"] = all(
        o.get("results") is None for o in fs_no_inc["output"] if o["type"] == "file_search_call"
    )
    fs_items_in = fb.get(f"/v1/responses/{fs_env['id']}/input_items").json()["data"]
    out["resp_file_search_inject"] = (
        fs_items_in[0]["role"] == "developer"
        and "[file_search results" in str(fs_items_in[0]["content"])
        and "alpha" in str(fs_items_in[0]["content"])
    )
    out["resp_file_search_unknown_vs_404"] = (
        fb.post(
            "/v1/responses",
            json={
                "model": "fx1",
                "input": "x",
                "tools": [{"type": "file_search", "vector_store_ids": ["vs_nope"]}],
            },
        ).status_code
        == 404
    )
    out["resp_tool_choice_file_search"] = (
        fb.post(
            "/v1/responses", json={**fs_req, "tool_choice": {"type": "file_search"}}
        ).status_code
        == 200
        # drop the include so the dict choice only binds the tool spec
    )
    out["resp_tool_choice_file_search_422"] = (
        fb.post(
            "/v1/responses",
            json={"model": "fx1", "input": "x", "tool_choice": {"type": "file_search"}},
        ).status_code
        == 422
    )
    out["resp_file_search_vsids_422"] = (
        fb.post(
            "/v1/responses",
            json={
                "model": "fx1",
                "input": "x",
                "tools": [{"type": "file_search", "vector_store_ids": []}],
            },
        ).status_code
        == 422
    )
    out["resp_file_search_results_cap_422"] = (
        fb.post(
            "/v1/responses",
            json={
                "model": "fx1",
                "input": "x",
                "tools": [
                    {
                        "type": "file_search",
                        "vector_store_ids": [vs_id],
                        "max_num_results": 51,
                    }
                ],
            },
        ).status_code
        == 422
    )
    out["resp_file_search_threshold_400"] = (
        fb.post(
            "/v1/responses",
            json={
                "model": "fx1",
                "input": "x",
                "tools": [
                    {
                        "type": "file_search",
                        "vector_store_ids": [vs_id],
                        "ranking_options": {"score_threshold": 1.5},
                    }
                ],
            },
        ).status_code
        == 422
    )
    # a file_search_call item re-fed as input carries its results as
    # system context — not a refusal
    refeed = fb.post(
        "/v1/responses",
        json={
            "model": "fx1",
            "input": [
                {
                    "type": "file_search_call",
                    "id": "fs_past",
                    "queries": ["alpha"],
                    "results": [
                        {
                            "file_id": "file-a",
                            "filename": "a.jsonl",
                            "score": 0.9,
                            "text": "alpha context",
                            "attributes": {},
                        }
                    ],
                },
                {"type": "message", "role": "user", "content": "next"},
            ],
        },
    )
    out["resp_file_search_refeed"] = refeed.status_code == 200
    # SSE: the file_search item lifecycle precedes the message item
    fs_sse = fb.post("/v1/responses", json={**fs_req, "stream": True})
    fs_body = fs_sse.text
    out["resp_file_search_sse"] = (
        fs_sse.status_code == 200
        and "response.file_search_call.in_progress" in fs_body
        and "response.file_search_call.searching" in fs_body
        and "response.file_search_call.completed" in fs_body
        and fs_body.index("file_search_call") < fs_body.index("response.output_text.delta")
    )
    # batch line honours the tool
    fs_batch_body = (
        _json3.dumps(
            {
                "custom_id": "fs1",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": "fx1",
                    "input": "alpha",
                    "tools": [{"type": "file_search", "vector_store_ids": [vs_id]}],
                },
            }
        )
        + "\n"
    ).encode()
    fs_up2 = fb.post(
        "/v1/files",
        files={"file": ("fsb.jsonl", fs_batch_body, "application/jsonl")},
        data={"purpose": "batch"},
    ).json()
    fs_bc = fb.post(
        "/v1/batches", json={"input_file_id": fs_up2["id"], "endpoint": "/v1/responses"}
    ).json()
    fs_bterm = _wait_batch(fb, fs_bc["id"])
    fs_line = _json3.loads(fb.get(f"/v1/files/{fs_bterm['output_file_id']}/content").text.strip())
    out["resp_file_search_batch"] = (
        fs_bterm["status"] == "completed"
        and fs_line["response"]["status_code"] == 200
        and any(o["type"] == "file_search_call" for o in fs_line["response"]["body"]["output"])
    )
    # --- POST /v1/vector_stores/{id}/search — ranked hits, no turn ----
    vs_search = fb.post(f"/v1/vector_stores/{vs_id}/search", json={"query": "alpha"})
    vs_search_page = vs_search.json()
    out["vs_search"] = (
        vs_search.status_code == 200
        and vs_search_page["object"] == "vector_store.search_results.page"
        and vs_search_page["search_query"] == "alpha"
        and vs_search_page["data"][0]["file_id"] == vs_up["id"]
        and vs_search_page["data"][0]["filename"] == "vs.jsonl"
        and vs_search_page["data"][0]["content"][0]["type"] == "text"
        and "alpha" in vs_search_page["data"][0]["content"][0]["text"]
        and vs_search_page["has_more"] is False
        and vs_search_page["next_page"] is None
    )
    vs_search_list = fb.post(
        f"/v1/vector_stores/{vs_id}/search",
        json={"query": ["alpha", "gamma"], "max_num_results": 5},
    ).json()
    out["vs_search_query_list"] = vs_search_list["search_query"] == "alpha gamma"
    out["vs_search_threshold"] = (
        fb.post(
            f"/v1/vector_stores/{vs_id}/search",
            json={"query": "alpha", "ranking_options": {"score_threshold": 0.999}},
        ).json()["data"]
        == []
    )
    out["vs_search_filters"] = (
        fb.post(
            f"/v1/vector_stores/{vs_id}/search",
            json={
                "query": "alpha",
                "filters": {"type": "eq", "key": "team", "value": "nope"},
            },
        ).json()["data"]
        == []
        and fb.post(
            f"/v1/vector_stores/{vs_id}/search",
            json={"query": "alpha", "filters": {"bad": "shape"}},
        ).status_code
        == 400
    )
    out["vs_search_404"] = (
        fb.post("/v1/vector_stores/vs_nope/search", json={"query": "x"}).status_code == 404
    )
    out["vs_search_empty_400"] = (
        fb.post(f"/v1/vector_stores/{vs_id}/search", json={"query": "  "}).status_code == 400
    )
    out["vs_search_rewrite_422"] = (
        fb.post(
            f"/v1/vector_stores/{vs_id}/search",
            json={"query": "x", "rewrite_query": True},
        ).status_code
        == 422
    )
    out["vs_search_ranker_422"] = (
        fb.post(
            f"/v1/vector_stores/{vs_id}/search",
            json={"query": "x", "ranking_options": {"ranker": "bm25"}},
        ).status_code
        == 422
    )
    out["vs_search_max_results_422"] = (
        fb.post(
            f"/v1/vector_stores/{vs_id}/search",
            json={"query": "x", "max_num_results": 51},
        ).status_code
        == 422
    )
    # --- /v1/vector_stores/{id}/file_batches — bulk attach, per-file verdicts
    vs_up2 = fb.post(
        "/v1/files",
        files={"file": ("vs2.jsonl", b"delta docs here", "application/jsonl")},
        data={"purpose": "batch"},
    ).json()
    vs_batch = fb.post(
        f"/v1/vector_stores/{vs_id}/file_batches",
        json={"file_ids": [vs_up2["id"], "file-ghost"]},
    )
    vs_batch_body = vs_batch.json()
    out["vs_batch_create"] = (
        vs_batch.status_code == 200
        and vs_batch_body["object"] == "vector_store.files_batch"
        and vs_batch_body["id"].startswith("vsfb_")
        and vs_batch_body["vector_store_id"] == vs_id
        and vs_batch_body["status"] == "completed"
        and vs_batch_body["file_counts"]
        == {
            "in_progress": 0,
            "completed": 1,
            "failed": 1,
            "cancelled": 0,
            "total": 2,
        }
    )
    vs_batch_get = fb.get(f"/v1/vector_stores/{vs_id}/file_batches/{vs_batch_body['id']}")
    out["vs_batch_get"] = (
        vs_batch_get.status_code == 200
        and vs_batch_get.json()["id"] == vs_batch_body["id"]
        and vs_batch_get.json()["status"] == "completed"
    )
    vs_batch_files = fb.get(
        f"/v1/vector_stores/{vs_id}/file_batches/{vs_batch_body['id']}/files",
        params={"filter": "failed"},
    )
    out["vs_batch_files"] = (
        vs_batch_files.status_code == 200
        and vs_batch_files.json()["object"] == "list"
        and [r["id"] for r in vs_batch_files.json()["data"]] == ["file-ghost"]
        and vs_batch_files.json()["data"][0]["last_error"]["code"] == "file_not_found"
    )
    vs_batch_files_all = fb.get(
        f"/v1/vector_stores/{vs_id}/file_batches/{vs_batch_body['id']}/files"
    ).json()
    out["vs_batch_files_all"] = {r["id"] for r in vs_batch_files_all["data"]} == {
        vs_up2["id"],
        "file-ghost",
    } and all(r["object"] == "vector_store.file" for r in vs_batch_files_all["data"])
    out["vs_batch_cancel_409"] = (
        fb.post(f"/v1/vector_stores/{vs_id}/file_batches/{vs_batch_body['id']}/cancel").status_code
        == 409
    )
    out["vs_batch_404"] = (
        fb.get(f"/v1/vector_stores/{vs_id}/file_batches/vsfb_nope").status_code == 404
        and fb.post(
            "/v1/vector_stores/vs_nope/file_batches", json={"file_ids": ["file-a"]}
        ).status_code
        == 404
    )
    out["vs_batch_empty_422"] = (
        fb.post(f"/v1/vector_stores/{vs_id}/file_batches", json={"file_ids": []}).status_code == 422
    )
    out["vs_batch_filter_400"] = (
        fb.get(
            f"/v1/vector_stores/{vs_id}/file_batches/{vs_batch_body['id']}/files",
            params={"filter": "bogus"},
        ).status_code
        == 400
    )
    # --- expires_after / last_active_at / standing expiry -------------
    exp_app = api_mod.create_app()
    fbx = _TC2(exp_app)
    vs_exp = fbx.post(
        "/v1/vector_stores",
        json={
            "name": "ephemeral",
            "expires_after": {"anchor": "last_active_at", "days": 1},
        },
    ).json()
    vs_exp_id = str(vs_exp["id"])
    out["vs_expires_after_create"] = (
        vs_exp["expires_after"] == {"anchor": "last_active_at", "days": 1}
        and vs_exp["expires_at"] == vs_exp["last_active_at"] + 86400
        and vs_exp["last_active_at"] >= vs_exp["created_at"]
        and vs_exp["status"] == "completed"
    )
    out["vs_expires_after_400"] = (
        fbx.post(
            "/v1/vector_stores",
            json={"expires_after": {"anchor": "created_at", "days": 1}},
        ).status_code
        == 400
        and fbx.post(
            "/v1/vector_stores",
            json={"expires_after": {"anchor": "last_active_at", "days": 0}},
        ).status_code
        == 400
        and fbx.post(
            "/v1/vector_stores",
            json={"expires_after": {"anchor": "last_active_at", "days": 366}},
        ).status_code
        == 400
    )
    # deterministic expiry: stamp expires_at in the past on the record —
    # the store flips read-only (writes/search refuse) while reads stay
    exp_app.state.vs_store._stores[vs_exp_id].expires_at = 1
    out["vs_expired_status"] = (
        fbx.get(f"/v1/vector_stores/{vs_exp_id}").json()["status"] == "expired"
    )
    out["vs_expired_writes_410"] = (
        fbx.post(f"/v1/vector_stores/{vs_exp_id}/files", json={"file_id": "file-x"}).status_code
        == 410
        and fbx.post(
            f"/v1/vector_stores/{vs_exp_id}/file_batches",
            json={"file_ids": ["file-x"]},
        ).status_code
        == 410
        and fbx.post(f"/v1/vector_stores/{vs_exp_id}/search", json={"query": "x"}).status_code
        == 410
    )
    # reads on an expired store still resolve
    out["vs_expired_reads_ok"] = (
        fbx.get(f"/v1/vector_stores/{vs_exp_id}/files").status_code == 200
        and fbx.get(f"/v1/vector_stores/{vs_exp_id}").status_code == 200
    )
    vs_exp_rev = fbx.post(
        f"/v1/vector_stores/{vs_exp_id}",
        json={"expires_after": {"anchor": "last_active_at", "days": 7}},
    ).json()
    out["vs_expiry_revive_update"] = (
        vs_exp_rev["status"] == "completed"
        and vs_exp_rev["expires_at"] == vs_exp_rev["last_active_at"] + 7 * 86400
        and fbx.post(f"/v1/vector_stores/{vs_exp_id}/search", json={"query": "x"}).status_code
        == 200
    )
    fbx.delete(f"/v1/vector_stores/{vs_exp_id}")
    # delete tombstone + detach shape + 404 after
    out["vs_file_detach"] = fb.delete(f"/v1/vector_stores/{vs_id}/files/{vs_up['id']}").json() == {
        "id": vs_up["id"],
        "object": "vector_store.file.deleted",
        "deleted": True,
    }
    out["vs_file_gone"] = (
        fb.get(f"/v1/vector_stores/{vs_id}/files/{vs_up['id']}").status_code == 404
    )
    vs_del = fb.delete(f"/v1/vector_stores/{vs_id}")
    out["vs_delete"] = vs_del.json() == {
        "id": vs_id,
        "object": "vector_store.deleted",
        "deleted": True,
    }
    out["vs_gone_404"] = fb.get(f"/v1/vector_stores/{vs_id}").status_code == 404

    # ---- Anthropic /v1/messages drop-in -----------------------------------
    # The Anthropic surface is a translation layer over the gated
    # _openai_chat_core — same auth (the SDK's x-api-key is the
    # case-insensitive X-API-Key), metering, backend chain, completion
    # log, and /v1 idempotency space; the response is re-minted in
    # Anthropic's message grammar and every refusal carries the
    # {type:"error",error:{...}} envelope (stock anthropic SDK parses it).

    am = fb.post(
        _MESSAGES_PATH,
        json={
            "model": "fx1",
            "max_tokens": 64,
            "system": "be terse",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )
    amb = am.json()
    am_cid = am.headers.get("X-Fx1-Completion-Id", "")
    out["anthropic_message_200"] = (
        am.status_code == 200
        and amb.get("type") == "message"
        and amb.get("role") == "assistant"
        and str(amb.get("id", "")).startswith("msg_")
        and amb.get("content") == [{"type": "text", "text": _PING_MSG}]
        and amb.get("stop_reason") == "end_turn"
        and amb.get("model") == "fake-0"
        and isinstance(amb.get("usage", {}).get("input_tokens"), int)
        and isinstance(amb.get("usage", {}).get("output_tokens"), int)
        and bool(am_cid)
    )
    # the answer is logged/receipted like any gated call, but `store` is
    # forced off — the chat-completions retrieval twin 404s
    if am_cid:
        out["anthropic_logged_not_stored"] = (
            fb.get(f"/harness/completions/{am_cid}").status_code == 200
            and fb.get(f"/v1/chat/completions/chatcmpl-{am_cid}").status_code == 404
        )

    # stream:true → Anthropic SSE grammar; frames carry id:<idx>; the
    # text deltas re-assemble the gated answer; message_stop is terminal
    ams = fb.post(
        _MESSAGES_PATH,
        json={
            "model": "fx1",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "stream me"}],
            "stream": True,
        },
    )
    aframes = [ln for ln in ams.text.split("\n\n") if ln.strip()]
    aparsed: list[dict[str, Any]] = []
    for ln in aframes:
        ev_name = ""
        data = ""
        for sub in ln.splitlines():
            if sub.startswith(_SSE_EVENT_PREFIX):
                ev_name = sub[7:]
            elif sub.startswith("data: "):
                data = sub[6:]
        if data:
            aparsed.append({"event": ev_name, "data": _json3.loads(data)})
    aseq = [f["event"] for f in aparsed]
    atext = "".join(
        f["data"]["delta"]["text"]
        for f in aparsed
        if f["data"].get("type") == "content_block_delta"
        and f["data"].get("delta", {}).get("type") == "text_delta"
    )
    out["anthropic_stream_grammar"] = (
        ams.status_code == 200
        and ams.headers.get("content-type", "").startswith("text/event-stream")
        and aseq[0] == "message_start"
        and aseq[1] == "ping"
        and "content_block_start" in aseq
        and "content_block_stop" in aseq
        and aseq[-2] == "message_delta"
        and aseq[-1] == "message_stop"
        and atext == "clean:stream me"
        and aparsed[0]["data"]["message"]["role"] == "assistant"
        and aparsed[-2]["data"]["delta"]["stop_reason"] == "end_turn"
        and "output_tokens" in aparsed[-2]["data"]["usage"]
    )
    # idempotency: same key+body replays the identical answer without a
    # second backend call; a mismatched body under the same key 409s
    ikey = {"Idempotency-Key": "am-1"}
    ai1 = fb.post(
        _MESSAGES_PATH,
        headers=ikey,
        json={
            "model": "fx1",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "idem"}],
        },
    )
    ai2 = fb.post(
        _MESSAGES_PATH,
        headers=ikey,
        json={
            "model": "fx1",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "idem"}],
        },
    )
    ai3 = fb.post(
        _MESSAGES_PATH,
        headers=ikey,
        json={
            "model": "fx1",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "different"}],
        },
    )
    out["anthropic_idem_replay"] = (
        ai1.status_code == 200
        and ai2.status_code == 200
        and ai1.json() == ai2.json()
        and ai2.headers.get("X-Fx1-Idempotent-Replay") == "true"
        and ai3.status_code == 409
        and ai3.json().get("type") == "error"
        and ai3.json()["error"]["type"] == "invalid_request_error"
    )
    # keyed stream replays resume: Last-Event-ID skips already-sent frames
    skey = {"Idempotency-Key": "am-s1"}
    asi1 = fb.post(
        _MESSAGES_PATH,
        headers=skey,
        json={
            "model": "fx1",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "resume"}],
            "stream": True,
        },
    )
    n_frames_1 = sum(1 for ln in asi1.text.split("\n\n") if "data:" in ln)
    asi2 = fb.post(
        _MESSAGES_PATH,
        headers={**skey, "Last-Event-ID": "1"},
        json={
            "model": "fx1",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "resume"}],
            "stream": True,
        },
    )
    n_frames_2 = sum(1 for ln in asi2.text.split("\n\n") if "data:" in ln)
    out["anthropic_stream_resume"] = (
        asi1.status_code == 200
        and asi2.status_code == 200
        and n_frames_2 == n_frames_1 - 2
        and "message_stop" in asi2.text
        # resume without the key / for an unkeyed original fails closed
        and fb.post(
            _MESSAGES_PATH,
            headers={"Last-Event-ID": "1"},
            json={
                "model": "fx1",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "resume"}],
                "stream": True,
            },
        ).status_code
        == 400
        and fb.post(
            _MESSAGES_PATH,
            headers={"Idempotency-Key": "no-pin", "Last-Event-ID": "0"},
            json={
                "model": "fx1",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "x"}],
                "stream": True,
            },
        ).status_code
        == 409
    )

    # tools: Anthropic input_schema → OpenAI function spec verbatim; a
    # tool_calls turn re-mints as a tool_use block with parsed input;
    # tool_result history translates to OpenAI tool turns
    at_backend = _OiToolBackend()
    tool_app = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: at_backend))
    at = tool_app.post(
        _MESSAGES_PATH,
        json={
            "model": "fx1",
            "max_tokens": 64,
            "tools": [
                {
                    "name": "calc",
                    "description": "adds",
                    "input_schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
                }
            ],
            "tool_choice": {"type": "any"},
            "messages": [{"role": "user", "content": "use the tool"}],
        },
    )
    atb = at.json()
    out["anthropic_tool_use_envelope"] = (
        at.status_code == 200
        and atb.get("stop_reason") == "tool_use"
        and atb["content"][0]["type"] == "tool_use"
        and atb["content"][0]["id"] == "call_0"
        and atb["content"][0]["name"] == "calc"
        and atb["content"][0]["input"] == {"x": 1}
    )
    # the forwarded spec: input_schema lands as function.parameters
    # verbatim and {type:"any"} maps to OpenAI's "required"
    out["anthropic_tools_forwarded"] = (
        at_backend.seen_tools
        == [
            {
                "type": "function",
                "function": {
                    "name": "calc",
                    "description": "adds",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}},
                    },
                },
            }
        ]
        and at_backend.seen_choice == "required"
    )
    at2 = tool_app.post(
        _MESSAGES_PATH,
        json={
            "model": "fx1",
            "max_tokens": 64,
            "tools": [
                {
                    "name": "calc",
                    "description": "adds",
                    "input_schema": {"type": "object"},
                }
            ],
            "tool_choice": {"type": "tool", "name": "calc"},
            "messages": [{"role": "user", "content": "go"}],
        },
    )
    out["anthropic_tool_choice_named"] = at2.status_code == 200 and (
        at_backend.seen_choice == {"type": "function", "function": {"name": "calc"}}
    )
    # tool_use/tool_result history → assistant tool_calls + role:"tool" turn
    at3 = tool_app.post(
        _MESSAGES_PATH,
        json={
            "model": "fx1",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "add"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_0",
                            "name": "calc",
                            "input": {"x": 1},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_0",
                            "content": "2",
                        }
                    ],
                },
            ],
        },
    )
    aseen = at_backend.seen_messages or []
    out["anthropic_tool_result_turns"] = (
        at3.status_code == 200
        and aseen[0] == {"role": "user", "content": "add"}
        and aseen[1]["role"] == "assistant"
        and aseen[1]["tool_calls"]
        == [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "calc", "arguments": '{"x": 1}'},
            }
        ]
        and aseen[2] == {"role": "tool", "tool_call_id": "call_0", "content": "2"}
    )

    # fail-closed surface — every refusal in the Anthropic envelope
    bad1 = fb.post(
        _MESSAGES_PATH,
        json={"model": "fx1", "messages": [{"role": "user", "content": "x"}]},
    )
    bad2 = fb.post(
        _MESSAGES_PATH,
        json={
            "model": "fx1",
            "max_tokens": 64,
            "messages": [{"role": "assistant", "content": "x"}],
        },
    )
    bad3 = fb.post(
        _MESSAGES_PATH,
        json={
            "model": "fx1",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "x"}],
            "top_k": 40,
        },
    )
    bad4 = fb.post(
        _MESSAGES_PATH,
        json={
            "model": "fx1",
            "max_tokens": 64,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "x"},
                        }
                    ],
                }
            ],
        },
    )
    bad5 = fb.post(
        _MESSAGES_PATH,
        json={
            "model": "fx1",
            "max_tokens": 64,
            "tools": [{"name": "f", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "missing"},
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    out["anthropic_failclosed"] = (
        # max_tokens required; non-user-first turn; top_k unsupported;
        # image blocks unsupported; tool_choice naming an absent tool —
        # each refusal is the Anthropic envelope, never OpenAI-shaped
        all(
            b.status_code in (400, 422)
            and b.json().get("type") == "error"
            and b.json()["error"]["type"] == "invalid_request_error"
            for b in (bad1, bad2, bad3, bad4, bad5)
        )
    )

    # ---- /v1/messages/batches: the Anthropic async channel over the
    # same jobs machinery — inline {custom_id, params} items, the
    # submitter's X-Fx1-* headers route every item, per-item faults land
    # as errored rows (data, not a crashed batch), cancel → canceling,
    # delete + results only once ended.
    class _AbatchBackend(_OiBackend):
        def complete(
            self,
            messages: list[dict[str, str]],
            *,
            sampling: SamplingParams | None = None,
        ) -> str:
            if "boom" in messages[-1]["content"]:
                raise RuntimeError("item boom")
            return super().complete(messages, sampling=sampling)

    ab_app = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _AbatchBackend()))

    def _wait_abatch(client: Any, batch_id: str) -> dict[str, Any]:
        b: dict[str, Any] = {}
        for _ in range(500):
            b = client.get(f"{_MESSAGES_PATH}/batches/{batch_id}").json()
            if b.get("processing_status") == "ended":
                return b
            time.sleep(0.01)
        return b

    ab_reqs = [
        {
            "custom_id": "ok-1",
            "params": {
                "model": "fx1",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "ping"}],
            },
        },
        {
            "custom_id": "boom-1",
            "params": {
                "model": "fx1",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "boom"}],
            },
        },
    ]
    ab1 = ab_app.post(f"{_MESSAGES_PATH}/batches", json={"requests": ab_reqs})
    ab1b = ab1.json()
    abterm = _wait_abatch(ab_app, ab1b["id"])
    abres = ab_app.get(f"{_MESSAGES_PATH}/batches/{ab1b['id']}/results")
    abrows = {
        row["custom_id"]: row["result"]
        for row in (_json3.loads(ln) for ln in abres.text.splitlines() if ln.strip())
    }
    out["anthropic_batch_roundtrip"] = (
        ab1.status_code == 200
        and ab1b.get("type") == "message_batch"
        and str(ab1b.get("id", "")).startswith("msgbatch_")
        and ab1b.get("request_counts", {}).get("processing") == 2
        and abterm["processing_status"] == "ended"
        and str(abterm.get("created_at", "")).endswith("Z")
        and str(abterm.get("expires_at", "")).endswith("Z")
        and str(abterm.get("ended_at", "")).endswith("Z")
        and abterm["request_counts"]
        == {
            "processing": 0,
            "succeeded": 1,
            "errored": 1,
            "canceled": 0,
            "expired": 0,
        }
        and str(abterm.get("results_url", "")).endswith(
            f"/v1/messages/batches/{ab1b['id']}/results"
        )
        and abres.status_code == 200
        and abres.headers.get("content-type", "").startswith("application/jsonl")
        and abrows["ok-1"]["type"] == "succeeded"
        and abrows["ok-1"]["message"]["type"] == "message"
        and abrows["ok-1"]["message"]["content"] == [{"type": "text", "text": _PING_MSG}]
        and abrows["boom-1"]["type"] == "errored"
        and abrows["boom-1"]["error"]["type"] == "api_error"
    )
    # submit-time validation: duplicate custom_id, stream inside a batch,
    # and an empty requests list all refuse in the Anthropic envelope
    ab_dup = ab_app.post(
        f"{_MESSAGES_PATH}/batches",
        json={"requests": [ab_reqs[0], ab_reqs[0]]},
    )
    ab_stream = ab_app.post(
        f"{_MESSAGES_PATH}/batches",
        json={
            "requests": [
                {
                    "custom_id": "s1",
                    "params": {
                        "model": "fx1",
                        "max_tokens": 64,
                        "stream": True,
                        "messages": [{"role": "user", "content": "x"}],
                    },
                }
            ]
        },
    )
    ab_empty = ab_app.post(f"{_MESSAGES_PATH}/batches", json={"requests": []})
    out["anthropic_batch_submit_failclosed"] = all(
        r.status_code in (400, 422)
        and r.json().get("type") == "error"
        and r.json()["error"]["type"] == "invalid_request_error"
        for r in (ab_dup, ab_stream, ab_empty)
    )
    # shared /v1 idempotency space: same key+body replays the submit
    # envelope; a different body under the same key 409s
    abk = {"Idempotency-Key": "ab-1"}
    abi1 = ab_app.post(f"{_MESSAGES_PATH}/batches", headers=abk, json={"requests": ab_reqs})
    abi2 = ab_app.post(f"{_MESSAGES_PATH}/batches", headers=abk, json={"requests": ab_reqs})
    abi3 = ab_app.post(
        f"{_MESSAGES_PATH}/batches",
        headers=abk,
        json={"requests": ab_reqs[:1]},
    )
    out["anthropic_batch_idem"] = (
        abi1.status_code == 200
        and abi2.status_code == 200
        and abi1.json() == abi2.json()
        and abi2.headers.get("X-Fx1-Idempotent-Replay") == "true"
        and abi3.status_code == 409
        and abi3.json()["error"]["type"] == "invalid_request_error"
    )
    _wait_abatch(ab_app, abi1.json()["id"])
    # listing: newest-first page + before_id/after_id cursors
    ablist = ab_app.get(f"{_MESSAGES_PATH}/batches?limit=1").json()
    out["anthropic_batch_list_shape"] = (
        ablist.get("data")
        and ablist["data"][0]["type"] == "message_batch"
        and "first_id" in ablist
        and "last_id" in ablist
        and "has_more" in ablist
    )
    ab_after = ab_app.get(f"{_MESSAGES_PATH}/batches?limit=50&after_id={ablist['last_id']}").json()
    ab_before = ab_app.get(f"{_MESSAGES_PATH}/batches?limit=50&before_id={ab1b['id']}").json()
    out["anthropic_batch_list_cursor"] = (
        # after_id pages forward past the cursor — the next page of
        # entries after it in list order, the way /v1/models (and the
        # stock SDK's auto-pagination) expects
        [b["id"] for b in ab_after["data"]] == [ab1b["id"]]
        and ab_after["has_more"] is False
        # before_id returns the tail of the window before the cursor —
        # the entries listed ahead of it, the cursor itself excluded
        and [b["id"] for b in ab_before["data"]] == [abi1.json()["id"]]
        and ab_before["has_more"] is False
    )
    # error grammar + terminal-state rules: unknown id 404s in the
    # Anthropic envelope; ended batches refuse cancel (400) and delete
    # cleanly; results and the record are gone after delete
    out["anthropic_batch_grammar"] = (
        ab_app.get(f"{_MESSAGES_PATH}/batches/msgbatch_nope").status_code == 404
        and ab_app.get(f"{_MESSAGES_PATH}/batches/msgbatch_nope").json()["error"]["type"]
        == "not_found_error"
        and ab_app.post(f"{_MESSAGES_PATH}/batches/{ab1b['id']}/cancel").status_code == 400
        and ab_app.get(f"{_MESSAGES_PATH}/batches/{ab1b['id']}/results").status_code == 200
    )
    abdel = ab_app.delete(f"{_MESSAGES_PATH}/batches/{ab1b['id']}")
    out["anthropic_batch_delete"] = (
        abdel.status_code == 200
        and abdel.json() == {"id": ab1b["id"], "type": "message_batch_deleted"}
        and ab_app.get(f"{_MESSAGES_PATH}/batches/{ab1b['id']}").status_code == 404
        and ab_app.get(f"{_MESSAGES_PATH}/batches/{ab1b['id']}/results").status_code == 404
    )
    # in-flight rules under a gated backend: counts stay all-processing,
    # results/delete refuse mid-flight, cancel lands 'canceling' and the
    # batch ends with canceled rows for the unprocessed tail
    ab_gate_ev = threading.Event()

    class _AbatchGateBackend(_OiBackend):
        def complete(
            self,
            messages: list[dict[str, str]],
            *,
            sampling: SamplingParams | None = None,
        ) -> str:
            ab_gate_ev.wait(10)
            return super().complete(messages, sampling=sampling)

    ab_gapp = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _AbatchGateBackend()))
    abg = ab_gapp.post(f"{_MESSAGES_PATH}/batches", json={"requests": ab_reqs})
    abg_id = abg.json()["id"]
    abg_mid = ab_gapp.get(f"{_MESSAGES_PATH}/batches/{abg_id}").json()
    abg_results_early = ab_gapp.get(f"{_MESSAGES_PATH}/batches/{abg_id}/results")
    abg_del_early = ab_gapp.delete(f"{_MESSAGES_PATH}/batches/{abg_id}")
    abg_cancel = ab_gapp.post(f"{_MESSAGES_PATH}/batches/{abg_id}/cancel")
    ab_gate_ev.set()
    abg_term = _wait_abatch(ab_gapp, abg_id)
    abg_rows = {
        row["custom_id"]: row["result"]
        for row in (
            _json3.loads(ln)
            for ln in ab_gapp.get(f"{_MESSAGES_PATH}/batches/{abg_id}/results").text.splitlines()
            if ln.strip()
        )
    }
    out["anthropic_batch_inflight_rules"] = (
        abg.status_code == 200
        and abg_mid["processing_status"] in ("in_progress", "canceling")
        and abg_mid["request_counts"]["processing"] == 2
        and abg_mid["request_counts"]["succeeded"] == 0
        and abg_mid["results_url"] is None
        and abg_results_early.status_code == 400
        and abg_del_early.status_code == 400
        and abg_cancel.status_code == 200
        and abg_cancel.json()["processing_status"] == "canceling"
        and abg_term["processing_status"] == "ended"
        and abg_term["request_counts"]["canceled"] >= 1
        and any(row["type"] == "canceled" for row in abg_rows.values())
    )
    # expiry projection: a record past expires_at ends 'expired' with
    # unfinished items landing expired rows on read
    ab_exp_app = api_mod.create_app(backend_resolver=lambda *a, **k: _OiBackend())
    ab_past = api_mod._AnthropicBatchRecord(  # noqa: SLF001
        batch_id="msgbatch_past",
        created_at=1,
        expires_at=2,
        request_counts={"processing": 1, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0},
        item_ids=["never"],
    )
    ab_exp_app.state.abatch_store.put(ab_past)
    ab_rexp = _TC2(ab_exp_app).get(f"{_MESSAGES_PATH}/batches/msgbatch_past").json()
    out["anthropic_batch_expiry_projection"] = (
        ab_rexp["processing_status"] == "ended"
        and ab_rexp["request_counts"]["expired"] == 1
        and ab_rexp["ended_at"] is not None
    )

    # ---- /v1/messages/count_tokens — the provider's own tokenizer count
    # over the message channel; honest 501 when the backend has no
    # tokenize route, 400 on tools (counting them would undercount).
    class _CountBackend(_OiBackend):
        """A backend WITH the tokenize channel — the route resolves
        through TokenCountingBackend, so this answers a real count."""

        last_msgs: ClassVar[list[dict[str, Any]] | None] = None

        def count_tokens(self, messages: list[dict[str, Any]]) -> int:
            _CountBackend.last_msgs = messages
            return 42

    ct_app = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _CountBackend()))
    ct_ok = ct_app.post(
        f"{_MESSAGES_PATH}/count_tokens",
        json={
            "model": "fx1",
            "system": "You are terse.",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )
    ct_msgs = _CountBackend.last_msgs or []
    out["anthropic_count_200"] = (
        ct_ok.status_code == 200
        and ct_ok.json() == {"input_tokens": 42}
        and [m.get("role") for m in ct_msgs] == ["system", "user"]
        and ct_msgs[0].get("content") == "You are terse."
    )
    ct_tools = ct_app.post(
        f"{_MESSAGES_PATH}/count_tokens",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "ping"}],
            "tools": [{"name": "t", "input_schema": {"type": "object"}}],
        },
    )
    ct_tools_b = ct_tools.json()
    out["anthropic_count_tools_refused"] = (
        ct_tools.status_code == 400
        and ct_tools_b.get("type") == "error"
        and ct_tools_b.get("error", {}).get("type") == "invalid_request_error"
    )
    # the shared message contract applies: assistant-first rejects at
    # validation (422 invalid_request_error in the Anthropic envelope)
    ct_bad = ct_app.post(
        f"{_MESSAGES_PATH}/count_tokens",
        json={
            "model": "fx1",
            "messages": [{"role": "assistant", "content": "hi"}],
        },
    )
    out["anthropic_count_contract"] = (
        ct_bad.status_code == 422
        and ct_bad.json().get("error", {}).get("type") == "invalid_request_error"
    )
    # a backend without the channel: honest 501, Anthropic envelope
    ct_501 = oi_clean.post(
        f"{_MESSAGES_PATH}/count_tokens",
        json={
            "model": "fx1",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )
    ct_501_b = ct_501.json()
    out["anthropic_count_501"] = (
        ct_501.status_code == 501
        and ct_501_b.get("type") == "error"
        and ct_501_b.get("error", {}).get("type") == "api_error"
    )

    # ---- anthropic-version on /v1/models{,/{id}} — the stock anthropic
    # SDK's models.list/retrieve grammar over the same inventory.
    a_models = oi_clean.get("/v1/models", headers={"anthropic-version": "2023-06-01"})
    a_models_b = a_models.json()
    out["anthropic_models_list"] = (
        a_models.status_code == 200
        and [m["id"] for m in a_models_b["data"]] == ["fx1", "hosted_k3", "local_fx1", "byok"]
        and all(
            m.get("type") == "model" and isinstance(m.get("display_name"), str)
            for m in a_models_b["data"]
        )
        and all(str(m.get("created_at", "")).endswith("Z") for m in a_models_b["data"])
        and a_models_b["first_id"] == "fx1"
        and a_models_b["last_id"] == "byok"
        and a_models_b["has_more"] is False
    )
    a_p1 = oi_clean.get("/v1/models?limit=1", headers={"anthropic-version": "2023-06-01"}).json()
    a_p2 = oi_clean.get(
        "/v1/models?limit=2&after_id=hosted_k3",
        headers={"anthropic-version": "2023-06-01"},
    ).json()
    a_none = oi_clean.get(
        "/v1/models?after_id=nope", headers={"anthropic-version": "2023-06-01"}
    ).json()
    # before_id pages backward — the *tail* of the window before the
    # cursor, so first_id chains backward the way last_id chains forward
    a_back = oi_clean.get(
        "/v1/models?limit=2&before_id=byok",
        headers={"anthropic-version": "2023-06-01"},
    ).json()
    out["anthropic_models_pagination"] = (
        [m["id"] for m in a_p1["data"]] == ["fx1"]
        and a_p1["has_more"] is True
        and a_p1["first_id"] == a_p1["last_id"] == "fx1"
        and [m["id"] for m in a_p2["data"]] == ["local_fx1", "byok"]
        and [m["id"] for m in a_back["data"]] == ["hosted_k3", "local_fx1"]
        and a_back["has_more"] is True
        and a_none["data"] == []
        and a_none["first_id"] is None
        and a_none["has_more"] is False
    )
    a_card = oi_clean.get("/v1/models/fx1", headers={"anthropic-version": "2023-06-01"}).json()
    a_404 = oi_clean.get("/v1/models/nope", headers={"anthropic-version": "2023-06-01"})
    out["anthropic_model_get"] = (
        a_card.get("type") == "model"
        and a_card.get("id") == "fx1"
        and a_card.get("display_name") == "fx1"
        and str(a_card.get("created_at", "")).endswith("Z")
    )
    out["anthropic_model_404"] = (
        a_404.status_code == 404
        and a_404.json().get("type") == "error"
        and a_404.json().get("error", {}).get("type") == "not_found_error"
    )

    # ---- drop-in response headers ------------------------------------------
    # `request-id` is the Anthropic grammar's name for the same id —
    # dialect responses carry both names with one value; an inbound
    # X-Request-ID echoes on both.
    a_echo = oi_clean.get(
        "/v1/models",
        headers={"anthropic-version": "2023-06-01", "X-Request-ID": "am-trace.1"},
    )
    out["anthropic_request_id_echo"] = (
        a_echo.status_code == 200
        and a_echo.headers.get("request-id") == "am-trace.1"
        and a_echo.headers.get("x-request-id") == "am-trace.1"
    )
    out["anthropic_request_id_200"] = am.headers.get("request-id") is not None and am.headers.get(
        "request-id"
    ) == am.headers.get("x-request-id")
    out["anthropic_request_id_error"] = (
        ai3.headers.get("request-id") is not None
        and ai3.headers.get("request-id") == ai3.headers.get("x-request-id")
        and a_404.headers.get("request-id") == a_404.headers.get("x-request-id")
    )
    # SSE: terminal frames can't carry metadata — request-id and
    # processing-ms ride the stream's opening headers, like OpenAI's
    out["anthropic_stream_headers"] = (
        ams.headers.get("request-id") is not None
        and ams.headers.get("request-id") == ams.headers.get("x-request-id")
        and int(ams.headers.get("openai-processing-ms", "-1")) >= 0
    )
    # x-should-retry: the stock SDK retries 429/5xx by default and treats
    # the rest as terminal — a 409 idempotency conflict carries false,
    # a clean 2xx omits the header (the SDK's default is already right)
    out["anthropic_retry_hint_terminal"] = ai3.headers.get("x-should-retry") == "false"
    out["anthropic_retry_hint_absent_2xx"] = (
        "x-should-retry" not in am.headers and "x-should-retry" not in a_404.headers
    )
    # openai surface never speaks Anthropic's names — and vice versa the
    # /harness routes carry neither grammar's headers
    o_models = oi_clean.get("/v1/models")
    out["openai_surface_no_anthropic_headers"] = (
        "request-id" not in o_models.headers
        and "anthropic-ratelimit-requests-limit" not in o_models.headers
        and "x-should-retry" not in o_models.headers
        and o_models.headers.get("openai-version") is not None
    )
    out["openai_version_openai_only"] = (
        o_models.headers.get("openai-version") == api_mod.API_VERSION
        and "openai-version" not in client.get("/harness/version").headers
    )
    # managed-key budget: the anthropic-ratelimit-requests-* family reports
    # the same declared window the X-RateLimit-* family does — remaining
    # decrements per call, reset is Anthropic's RFC 3339 instant, and the
    # over-limit 429 carries both the window and x-should-retry:true.
    # Env/loopback credentials declare no window and emit none — same
    # no-false-scarcity rule as the OpenAI family.
    saved_api_key = os.environ.get(_API_KEY_ENV)
    os.environ[_API_KEY_ENV] = "k3y-material"
    try:
        a_sec = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _OiBackend()))
    finally:
        if saved_api_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = saved_api_key
    a_rpm = a_sec.post("/harness/keys", json={"rpm": 2}, headers={"X-API-Key": "k3y-material"})
    a_rkey = str(a_rpm.json().get("key", ""))
    a_kh = {"X-API-Key": a_rkey}
    a_r1 = a_sec.post(
        _MESSAGES_PATH,
        json={"model": "fx1", "max_tokens": 8, "messages": [{"role": "user", "content": "a"}]},
        headers=a_kh,
    )
    a_r2 = a_sec.get("/v1/models", headers={**a_kh, "anthropic-version": "2023-06-01"})
    a_r3 = a_sec.get("/v1/models", headers={**a_kh, "anthropic-version": "2023-06-01"})
    out["anthropic_ratelimit_managed"] = (
        a_r1.status_code == 200
        and a_r1.headers.get("anthropic-ratelimit-requests-limit") == "2"
        and a_r1.headers.get("anthropic-ratelimit-requests-remaining") == "1"
        and a_r1.headers.get("anthropic-ratelimit-requests-reset", "").endswith("Z")
        and a_r1.headers.get("x-ratelimit-limit-requests") == "2"
        and a_r2.headers.get("anthropic-ratelimit-requests-remaining") == "0"
        and a_r2.headers.get("x-ratelimit-remaining-requests") == "0"
    )
    out["anthropic_ratelimit_429"] = (
        a_r3.status_code == 429
        and a_r3.headers.get("anthropic-ratelimit-requests-limit") == "2"
        and a_r3.headers.get("anthropic-ratelimit-requests-remaining") == "0"
        and a_r3.headers.get("x-should-retry") == "true"
        and int(a_r3.headers.get("retry-after", "0")) >= 1
    )
    out["anthropic_ratelimit_absent_env_loopback"] = (
        "anthropic-ratelimit-requests-limit" not in am.headers
        and "anthropic-ratelimit-requests-limit"
        not in a_sec.get(
            "/harness/version",
            headers={"X-API-Key": "k3y-material", "anthropic-version": "2023-06-01"},
        ).headers
    )
    # headers we deliberately DON'T emit: no org/proxy/edge provenance —
    # there is no organization layer or CDN in front of this process
    out["no_proxy_header_leak"] = all(
        h not in am.headers and h not in a_r1.headers
        for h in ("openai-organization", "cf-ray", "cf-cache-status", "cf-request-id")
    )
    # the spec itself declares the Anthropic family on /v1/messages* ops
    # only — generated clients see them typed on the right surface
    a_spec = oi_clean.get("/openapi.json").json()
    a_msg_ops = [
        op
        for p, item in a_spec["paths"].items()
        if p == _MESSAGES_PATH or p.startswith(_MESSAGES_PATH + "/")
        for op in item.values()
        if isinstance(op, dict)
    ]
    a_other_ops = [
        op
        for p, item in a_spec["paths"].items()
        if not (p == _MESSAGES_PATH or p.startswith(_MESSAGES_PATH + "/"))
        for op in item.values()
        if isinstance(op, dict)
    ]
    out["openapi_declares_anthropic_headers"] = (
        bool(a_msg_ops)
        and all(
            "request-id" in resp.get("headers", {})
            for op in a_msg_ops
            for resp in op.get("responses", {}).values()
        )
        and all(
            "request-id" not in resp.get("headers", {})
            for op in a_other_ops
            for resp in op.get("responses", {}).values()
        )
    )

    # ---- legacy /v1/completions drop-in ------------------------------------
    # the pre-chat text surface: each prompt element is one user turn
    # through the same gated pipeline (honesty gate, fail-closed
    # validation, completion-log metering); choices flatten to
    # prompt×n, echo prepends the prompt to each text
    lc = fb.post(
        _LEGACY_PATH,
        json={"model": "fx1", "prompt": "ping", "max_tokens": 32},
    )
    lcb = lc.json()
    lc_cid = lc.headers.get("X-Fx1-Completion-Id", "")
    out["legacy_completion_200"] = (
        lc.status_code == 200
        and lcb.get("object") == "text_completion"
        and str(lcb.get("id", "")).startswith("cmpl-")
        and lcb.get("model") == "fake-0"
        and lcb.get("choices")
        == [
            {
                "index": 0,
                "text": _PING_MSG,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ]
        and "usage" in lcb
        and bool(lc_cid)
        and lc.headers.get("openai-version") == "1"
        and bool(lc.headers.get("X-Fx1-Receipt-Sha256"))
    )
    # the gated call is logged/receipted like chat; legacy completions
    # have no retrieval twin — GET /v1/completions/{id} is the 404
    if lc_cid:
        out["legacy_logged_not_stored"] = (
            fb.get(f"/harness/completions/{lc_cid}").status_code == 200
            and fb.get(f"/v1/completions/cmpl-{lc_cid}").status_code == 404
        )
    # prompt list + n flatten to prompt×n choices in order; echo
    # prepends the prompt text
    lm = fb.post(
        _LEGACY_PATH,
        json={
            "model": "fx1",
            "prompt": ["a", "b"],
            "n": 2,
            "echo": True,
            "max_tokens": 32,
        },
    )
    lmb = lm.json()
    out["legacy_multi_prompt_flat"] = (
        lm.status_code == 200
        and len(lmb.get("choices", [])) == 4
        and [c.get("index") for c in lmb["choices"]] == [0, 1, 2, 3]
        and all(str(c.get("text", "")).startswith(("a", "b")) for c in lmb["choices"])
    )
    # unsupported legacy fields refuse 422 — no FIM head, no
    # logprob scorer, no best-of picker in this pipeline
    for field in ("suffix", "best_of", "logprobs"):
        out[f"legacy_refuses_{field}_422"] = (
            fb.post(
                _LEGACY_PATH,
                json={"model": "fx1", "prompt": "x", field: ("s" if field == "suffix" else 1)},
            ).status_code
            == 422
        )
    # idempotency: same key+body replays byte-identically without a
    # second backend call; mismatched body under the same key 409s
    lkey = {"Idempotency-Key": "lc-1"}
    li1 = fb.post(
        _LEGACY_PATH,
        headers=lkey,
        json={"model": "fx1", "prompt": "idem", "max_tokens": 16},
    )
    li2 = fb.post(
        _LEGACY_PATH,
        headers=lkey,
        json={"model": "fx1", "prompt": "idem", "max_tokens": 16},
    )
    li3 = fb.post(
        _LEGACY_PATH,
        headers=lkey,
        json={"model": "fx1", "prompt": "different body", "max_tokens": 16},
    )
    out["legacy_idempotent_replay"] = (
        li1.status_code == 200
        and li2.status_code == 200
        and li1.content == li2.content
        and li2.headers.get("X-Fx1-Idempotent-Replay") == "true"
        and li3.status_code == 409
    )
    # stream:true → legacy chunk grammar (text_completion chunks, usage
    # when stream_options asks, [DONE] terminal); a keyed stream replays
    # the identical frames; Last-Event-ID resumes below the floor
    ls = fb.post(
        _LEGACY_PATH,
        json={"model": "fx1", "prompt": "stream", "stream": True},
    )
    lframes = [ln for ln in ls.text.split("\n\n") if ln.strip()]
    ldata_lines = [sub[6:] for ln in lframes for sub in ln.splitlines() if sub.startswith("data: ")]
    ldatas = [_json3.loads(d) for d in ldata_lines if d != "[DONE]"]
    out["legacy_stream_grammar"] = (
        ls.status_code == 200
        and ls.headers.get("content-type", "").startswith("text/event-stream")
        and ldata_lines[-1] == "[DONE]"
        and all(d.get("object") == "text_completion" for d in ldatas)
        and ldatas
        and "".join(d["choices"][0]["text"] for d in ldatas) == "clean:stream"
        and ldatas[-1]["choices"][0]["finish_reason"] == "stop"
    )
    # the unknown /v1 catch-all still answers provider-shaped 404s under
    # the new surface
    out["legacy_catchall_404"] = (
        fb.get("/v1/no-such-route").json().get("error", {}).get("type") == "invalid_request_error"
    )


def api_audit_bench() -> dict[str, Any]:
    """Sealed receipt: every probe True under api_audit.v1."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = api_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "api_audit",
        "schema": "api_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Harness API holds: registry parity with the in-process Harness, "
            "configs/ containment over the wire, uniform 4xx fail-closed shape, "
            "the honesty gate fires on model output (502), credential-less "
            "backends 503, arbitrary receipts verify through the posted "
            "payload, auth is X-API-Key or loopback-only, and every gated "
            "call lands in the bounded completion log (X-Fx1-Completion-Id "
            "handle; hashes, usage, verdict — never content) fetchable via "
            "GET /harness/completions[/{id}], and each logged call exports "
            "as a sealed fx1_completion_record.v1 document "
            "(GET …/{id}/receipt) — deterministic, verifiable through "
            "verify_receipt / POST /receipts/verify, and broken by any "
            "byte of record tampering. Terminal jobs export the same way "
            "as fx1_job_record.v1 (GET /harness/jobs/{id}/receipt): "
            "stdout/stderr digested inside record.result, callback URL "
            "hashed, seal re-derives and verifies, tampering breaks it. "
            "The /v1/fine_tuning surface holds the OpenAI job grammar: "
            "synchronous corpus validation (bad corpus/model/file "
            "fail closed 4xx before any queue), cooperative cancel, "
            "idempotent submit, events feed, artifacts re-registered as "
            "fine-tune-result files. A succeeded job's ft: name registers "
            "into the model inventory (listed + retrievable), completions "
            "naming it resolve to the local_fx1 lane pinned at the job's "
            "checkpoint, explicit backend headers still override, and "
            "unregistered ft: names fail closed 404 model_not_found. "
            "X-Fx1-Timeout sets the per-request backend deadline on the "
            "OpenAI surface (fx1.timeout_s extension wins; malformed or "
            "out-of-range values fail closed 400). Every gated response "
            "self-describes its evidence: X-Fx1-Receipt-Sha256 carries the "
            "seal of the logged record (identical to the document "
            "GET /harness/completions/{id}/receipt exports), idempotent "
            "replays echo the original seal, and SSE streams carry the "
            "digest in the final frame. Evidence citations ride "
            "X-Fx1-Receipt-Hashes for clients that can't edit the body — "
            "comma-separated digests, the same store check, a malformed "
            "digest a fail-closed 400, and fx1.receipt_hashes wins. "
            "DELETE /v1/models/{id} unregisters an ft: name with a real "
            "tombstone (list/retrieve/chat all 404 after; built-ins "
            "refuse 400), and every X-Fx1-* request knob is in the CORS "
            "allow-headers list so browser clients can send them. "
            "Drop-in response headers hold: every Anthropic-dialect "
            "response carries `request-id` (the same id as X-Request-ID, "
            "echoed when supplied) on 2xx, error, and SSE-open alike; "
            "x-should-retry pins the statuses the stock SDK's defaults "
            "would get wrong (true on transient 429/5xx, false on "
            "terminal 409/501); managed-key rpm windows report as "
            "anthropic-ratelimit-requests-* alongside X-RateLimit-* "
            "(remaining decrements, reset an RFC 3339 instant, absent "
            "for env/loopback — no false scarcity); the OpenAI surface "
            "never speaks Anthropic's names; and no org/proxy/edge "
            "headers leak (no openai-organization, no cf-*). "
            "GET /v1/responses/{id}?stream=true replays a stored "
            "response as the Responses SSE grammar — the emitted "
            "sequence derives deterministically from the stored "
            "envelope (byte-identical to the create-time stream for a "
            "stream-created record), frames carry the monotonic id: "
            "cursor starting_after slices on, an in-flight "
            "background:true attach emits the prelude then live-follows "
            "keepalives to terminal, and cancelled/failed records end "
            "response.cancelled/response.failed with the stored object "
            "verbatim."
            if ok
            else f"HARNESS API AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
