"""``fx1`` command-line interface — the model project's front door.

dipcatcher remains available as the harness CLI (``dipcatcher``/``quant``
aliases); this CLI drives the fx-1 lifecycle: corpus, eval, training
manifests, and harness inspection.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast

import typer

from fx1.cli_errors import Fx1TyperGroup
from fx1.cli_sources import register_source_commands

if TYPE_CHECKING:
    from fx1.eval.suite import ModelFn
    from fx1.harness import HarnessRole
    from fx1.sdk import Fx1Harness
    from fx1.serve.client import HarnessClient

_REMOTE_HELP = (
    "Drive a remote harness API at this base URL (HarnessClient) instead of the in-process SDK."
)
_API_KEY_HELP = "X-API-Key for the remote harness; falls back to FX1_API_KEY."
_TIMEOUT_HELP = "Remote request timeout in seconds."
_BYOK_URL_HELP = "Per-request BYOK endpoint (OpenAI-compatible base URL)."
_BYOK_KEY_HELP = "Per-request BYOK API key."
_BYOK_MODEL_HELP = "Per-request BYOK model name."
_LIMIT_HELP = "Page size (max 100)."
_EVAL_ID_HELP = "eval_ id."
_EVAL_ID_CREATE_HELP = "eval_ id from eval-spec-create."
_EVALRUN_ID_HELP = "evalrun_ id."
_FTJOB_ID_HELP = "ftjob- id from ft-create."
_BACKEND_OVERRIDE_HELP = "Backend link override."
_FALLBACK_HELP = "Alternate backend on availability faults (repeatable)."
_CHAT_ID_HELP = "Stored chat.completion id (chatcmpl-*)."
_RESPONSE_ID_HELP = "Stored response id (resp_*)."
_CONV_ID_HELP = "Conversation id (conv_*)."
_VS_ID_HELP = "Vector store id (vs_*)."
_FILE_ID_HELP = "File record id (file-*)."
_VSFB_ID_HELP = "Batch id (vsfb_*)."
_ORDER_HELP = "asc | desc"
_ITEM_CURSOR_HELP = "Page cursor — an item id."
_METADATA_PAIRS_HELP = "JSON object of string pairs."
_ITEMS_JSON_ERR = "--items must be a JSON array of item dicts"
_METADATA_PAIRS_ERR = "--metadata must be a JSON object of string pairs"
_EVAL_ID_REMOTE_HELP = "Eval id returned by harness eval --remote."
_CALLBACK_SECRET_HELP = "HMAC key signing the webhook delivery."

_POLL_HELP = "Status poll interval, seconds."
_BATCH_ID_HELP = "batch_ id from batch-submit."
_MODEL_ID_OPT_HELP = "Model id — backend name, fx1, or ft:name."
_TOP_P_OPT_HELP = "Nucleus sampling mass."


def _byok_opts(
    base_url: str | None, api_key: str | None, model: str | None
) -> dict[str, str] | None:
    """Pack the three --byok-* flags into the wire override; all-or-none."""
    parts = {"base_url": base_url, "api_key": api_key, "model": model}
    if all(v is None for v in parts.values()):
        return None
    if any(v is None for v in parts.values()):
        typer.echo("--byok-base-url, --byok-api-key and --byok-model go together", err=True)
        raise typer.Exit(2)
    return {k: str(v) for k, v in parts.items()}


def _fx1_headers(
    backend: str | None,
    checkpoint_dir: Path | None,
    byok: dict[str, str] | None,
    fallbacks: list[str],
) -> dict[str, str]:
    """Pack the link override flags into the X-Fx1-* header set the
    in-process SDK path sends — the remote client packs them into the
    ``fx1`` extension object instead, same semantics."""
    headers: dict[str, str] = {}
    if backend is not None:
        headers["x-fx1-backend"] = backend
    if checkpoint_dir is not None:
        headers["x-fx1-checkpoint-dir"] = str(checkpoint_dir)
    if byok is not None:
        headers["x-fx1-byok-base-url"] = byok["base_url"]
        headers["x-fx1-byok-api-key"] = byok["api_key"]
        headers["x-fx1-byok-model"] = byok["model"]
    if fallbacks:
        headers["x-fx1-fallbacks"] = ",".join(fallbacks)
    return headers


def _surface(
    remote: str | None, api_key: str | None, timeout_s: float
) -> Fx1Harness | HarnessClient:
    """In-process SDK by default; HarnessClient when ``--remote`` is given."""
    if remote is None:
        from fx1.sdk import Fx1Harness

        return Fx1Harness()
    from fx1.serve.client import HarnessClient

    return HarnessClient(
        remote,
        api_key=api_key or os.environ.get("FX1_API_KEY") or None,
        timeout_s=timeout_s,
    )


def _last_request_id(client: Any) -> str | None:
    """``x-request-id`` off the client's last wire response, when it
    reports one — the trace id a bug report would quote. ``None`` for
    clients without a header channel or a transport that saw no
    response."""
    hmap = getattr(client, "last_response_headers", None)
    if not isinstance(hmap, Mapping):
        return None
    rid = hmap.get("x-request-id")
    return rid if isinstance(rid, str) and rid else None


def _or_exit[T](fn: Callable[[], T]) -> T:
    """Run a surface call; faults print one clean line and exit 2."""
    try:
        return fn()
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001 — CLI reports the class+message, not a traceback
        typer.echo(f"error: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=2) from exc


def _bad_arg(msg: str) -> NoReturn:
    """A flag/value fault — one clean line, exit 2, no traceback."""
    typer.echo(f"error: {msg}", err=True)
    raise typer.Exit(code=2)


def _json_meta(raw: str | None) -> dict[str, str] | None:
    """``--metadata`` JSON — must be an object of string pairs."""
    if raw is None:
        return None
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError:
        _bad_arg(_METADATA_PAIRS_ERR)
        raise AssertionError("unreachable") from None
    if not (
        isinstance(meta, dict)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items())
    ):
        _bad_arg(_METADATA_PAIRS_ERR)
    return meta


def _json_obj_opt(raw: str | None, flag: str) -> dict[str, Any] | None:
    """A JSON-object flag — object or None, clean exit on anything else."""
    if raw is None:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        _bad_arg(f"{flag} must be a JSON object")
    if not isinstance(obj, dict):
        _bad_arg(f"{flag} must be a JSON object")
    return obj


def _json_list_opt(raw: str | None, flag: str) -> list[dict[str, Any]] | None:
    """A JSON list-of-objects flag (testing_criteria entries)."""
    if raw is None:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        _bad_arg(f"{flag} must be a JSON list of objects")
    if not (isinstance(obj, list) and all(isinstance(it, dict) for it in obj)):
        _bad_arg(f"{flag} must be a JSON list of objects")
    return obj


def _response_frame_of(event: Any) -> tuple[str | None, str | None]:
    """One stream frame → ``(delta text, incomplete reason)``."""
    payload = event[1] if isinstance(event, tuple) else event
    if not isinstance(payload, dict):
        return None, None
    reason: str | None = None
    if payload.get("type") == "response.incomplete":
        resp = payload.get("response")
        details = resp.get("incomplete_details") if isinstance(resp, dict) else None
        if isinstance(details, dict) and isinstance(details.get("reason"), str):
            reason = details["reason"]
    if payload.get("type") == "response.failed":
        resp = payload.get("response")
        err = resp.get("error") if isinstance(resp, dict) else None
        reason = (
            f"failed: {err['message']}"
            if isinstance(err, dict) and isinstance(err.get("message"), str)
            else "failed"
        )
    if payload.get("type") == "response.cancelled":
        reason = "cancelled"
    delta = payload.get("delta")
    return delta if isinstance(delta, str) else None, reason


def _emit_response_deltas(events: Iterable[Any]) -> None:
    """Print a Responses event stream's text — every ``*.delta`` frame's
    ``delta`` string (output text and function-call arguments alike),
    nothing else. Accepts bare payload dicts (remote) or ``(event,
    payload)`` pairs (in-process SDK). A ``response.incomplete``/``failed``/
    ``cancelled`` terminal reports its reason on stderr — a quiet text
    stream would look like a full answer."""
    terminal_reason: str | None = None
    for event in events:
        delta, reason = _response_frame_of(event)
        if delta is not None:
            typer.echo(delta, nl=False)
        if reason is not None:
            terminal_reason = reason
    typer.echo()
    if terminal_reason is not None:
        typer.echo(f"[stream ended: {terminal_reason}]", err=True)


def _emit_anthropic_deltas(events: Iterable[Any]) -> None:
    """Print an Anthropic event stream — ``text_delta`` pieces inline,
    a ``tool_use`` block's ``input_json_delta`` into the same stream
    (delimited), nothing else. ``message_stop`` is the terminal frame;
    its absence means the stream was cut (the callers fail on that)."""
    for event in events:
        data = event.get("data", event) if isinstance(event, dict) else event
        payload = data if isinstance(data, dict) else {}
        ptype = payload.get("type")
        if ptype == "content_block_start":
            if (payload.get("content_block") or {}).get("type") == "tool_use":
                typer.echo("[tool_use]", nl=False)
        elif ptype == "content_block_delta":
            delta = payload.get("delta") or {}
            text = (
                delta.get("text")
                if delta.get("type") == "text_delta"
                else delta.get("partial_json")
            )
            if text:
                typer.echo(text, nl=False)
    typer.echo()


app = typer.Typer(
    cls=Fx1TyperGroup,
    name="fx1",
    help=(
        "fx-1 — the quant LLM. dipcatcher is the harness that builds, evaluates, and verifies it. "
        "--version on any command prints the version and exits."
    ),
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the fx-1 package version and exit.",
        is_eager=True,
    ),
) -> None:
    if version:
        from fx1 import __version__

        typer.echo(f"fx1 {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


corpus_app = typer.Typer(help="Training-corpus construction.")
train_app = typer.Typer(help="Training-run manifests (LoRA/QLoRA on K3).")
harness_app = typer.Typer(help="Inspect/run the dipcatcher harness.")
sources_app = typer.Typer(help="Professional datasource registry, routing, fetch.")
app.add_typer(corpus_app, name="corpus")
app.add_typer(train_app, name="train")
app.add_typer(harness_app, name="harness")
app.add_typer(sources_app, name="sources")

register_source_commands(sources_app, corpus_app)


@harness_app.command("operations")
def harness_operations(
    query: str = typer.Argument("", help="Substring filter over operation ids."),
    kind: str | None = typer.Option(None, help="Restrict to one kind: feature, skill, or plugin."),
    offset: int = typer.Option(0, min=0, help="Zero-based page offset."),
    limit: int = typer.Option(20, min=1, max=100, help="Page size (1-100)."),
) -> None:
    """List registered dipcatcher capabilities through the explicit registry."""
    from fx1.operations.base import OperationKind
    from fx1.operations.registry import list_operations

    if kind is not None and kind not in ("feature", "skill", "plugin"):
        raise typer.BadParameter("must be feature, skill, or plugin", param_hint="--kind")
    kind_arg = cast(OperationKind, kind) if kind is not None else None
    typer.echo(
        json.dumps(
            list_operations(query, kind=kind_arg, offset=offset, limit=limit),
            indent=2,
        )
    )


@harness_app.command("describe-operation")
def harness_describe_operation(
    operation_id: str = typer.Argument(
        ..., help="Registered operation id, e.g. features.simple_returns."
    ),
) -> None:
    """Print the exact input/output schema of one registered operation."""
    from fx1.operations.registry import get_operation

    typer.echo(json.dumps(get_operation(operation_id).describe(), indent=2))


_OPERATION_ARGUMENT_MAX_BYTES = 2_000_000


@harness_app.command("execute-operation")
def harness_execute_operation(
    operation_id: str = typer.Argument(..., help="Registered operation id."),
    arguments: str | None = typer.Option(
        None, "--arguments", help="JSON object of inputs (at most 2,000,000 UTF-8 bytes)."
    ),
    arguments_file: Path | None = typer.Option(
        None,
        "--arguments-file",
        help="Path to a UTF-8 JSON file of inputs (at most 2,000,000 bytes).",
    ),
    workspace_root: Path | None = typer.Option(
        None, "--workspace-root", help="Workspace filesystem boundary."
    ),
) -> None:
    """Execute one registered operation with validated, bounded inputs."""
    if (arguments is None) == (arguments_file is None):
        raise typer.BadParameter("Provide exactly one of --arguments or --arguments-file.")
    try:
        if arguments is not None:
            # Check characters first so even encoding a large argv value is bounded.
            if len(arguments) > _OPERATION_ARGUMENT_MAX_BYTES:
                raise ValueError(
                    f"arguments exceed the {_OPERATION_ARGUMENT_MAX_BYTES}-byte budget"
                )
            raw = arguments.encode("utf-8")
        else:
            assert arguments_file is not None
            # Read one sentinel byte beyond the budget; never materialize the whole file.
            with arguments_file.open("rb") as stream:
                raw = stream.read(_OPERATION_ARGUMENT_MAX_BYTES + 1)
        if len(raw) > _OPERATION_ARGUMENT_MAX_BYTES:
            raise ValueError(f"arguments exceed the {_OPERATION_ARGUMENT_MAX_BYTES}-byte budget")
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, OSError, RecursionError) as exc:
        raise typer.BadParameter(f"cannot read arguments: {exc}") from exc
    try:
        resolved_root = workspace_root.resolve() if workspace_root is not None else None
    except (OSError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(
            f"cannot resolve workspace root: {exc}", param_hint="--workspace-root"
        ) from exc
    from fx1.operations.registry import execute_operation

    result = execute_operation(operation_id, payload, workspace_root=resolved_root)
    typer.echo(json.dumps(result, indent=2))


@corpus_app.command("build")
def corpus_build(
    receipts_dir: list[Path] = typer.Option(
        [Path("receipts")],
        help="Receipt directory (repeatable — the flywheel spans several).",
    ),
    out: Path = typer.Option(Path("data/fx1/corpus.jsonl"), help="Output JSONL."),
) -> None:
    """Build the fx-1 SFT corpus from gate-passed dipcatcher receipts."""
    from fx1.data import build_corpus

    stats = build_corpus(list(receipts_dir), out)
    typer.echo(json.dumps(stats, indent=2))


@corpus_app.command("build-full")
def corpus_build_full(
    receipts_dir: list[Path] = typer.Option([Path("receipts")]),
    out: Path = typer.Option(Path("data/fx1/corpus.jsonl")),
    notebooks: list[Path] = typer.Option([], help="Research notebooks/docs to include."),
    artifacts_dir: Path | None = typer.Option(None, help="Ledger artifacts dir."),
) -> None:
    """Full corpus: receipts + notebooks + ledgers, all provenance-hashed."""
    from fx1.data import build_full_corpus

    stats = build_full_corpus(
        list(receipts_dir), out, notebooks=list(notebooks), artifacts_dir=artifacts_dir
    )
    typer.echo(json.dumps(stats, indent=2))


@train_app.command("manifest")
def train_manifest(
    config: Path = typer.Option(..., help="TrainConfig JSON file."),
    out: Path = typer.Option(Path("data/fx1/manifest.json")),
) -> None:
    """Validate the run contract (eval-before-train, provenance, cost) and
    write an immutable training manifest."""
    from fx1.train import TrainConfig, build_training_manifest

    cfg = TrainConfig.model_validate_json(config.read_text(encoding="utf-8"))
    manifest = build_training_manifest(cfg, out)
    typer.echo(json.dumps({"run_name": manifest["run_name"], "out": str(out)}))


@harness_app.command("list")
def harness_list(
    role: str | None = typer.Option(
        None, help="Filter: data_engine | evaluation | verification | model_training"
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """List the lab commands fx-1 may invoke through the harness."""
    if remote is not None:
        from fx1.serve.client import HarnessClient

        surface = HarnessClient(
            remote,
            api_key=api_key or os.environ.get("FX1_API_KEY") or None,
            timeout_s=timeout_s,
        )
        for name in _or_exit(lambda: surface.commands(role=role)):
            typer.echo(name)
        return
    from fx1.harness import Harness, HarnessRole

    # a bogus --role is an arg fault on the same clean line the remote leg
    # gets from _or_exit — never a raw traceback.
    role_filter = _or_exit(lambda: HarnessRole(role)) if role else None
    for cmd in Harness().list_commands(role=role_filter):
        typer.echo(f"{cmd.name:<18} [{cmd.role}] {cmd.description}")


@harness_app.command("run")
def harness_run(
    name: str = typer.Argument(..., help="Registered harness command name."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    idempotency_key: str | None = typer.Option(
        None,
        "--idempotency-key",
        help="Dedup key for the remote submission; a retried call returns the stored result. Auto-minted per invocation.",
    ),
) -> None:
    """Run a registered dipcatcher harness command (fail-closed registry)."""
    if remote is not None:
        from fx1.serve.client import HarnessClient

        client = HarnessClient(
            remote,
            api_key=api_key or os.environ.get("FX1_API_KEY") or None,
            timeout_s=timeout_s,
        )
        result = _or_exit(lambda: client.run(name, idempotency_key=idempotency_key))
    else:
        from fx1.harness import Harness

        # same refusal shape as the remote leg: registry misses print one
        # clean error line + exit 2, never a traceback exit 1.
        result = _or_exit(lambda: Harness().run(name))
    typer.echo(result.stdout)
    if result.stderr:
        typer.echo(result.stderr, err=True)
    raise typer.Exit(code=result.exit_code)


@harness_app.command("doctor")
def harness_doctor(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    receipts_dir: Path = typer.Option(
        Path("receipts"),
        "--receipts-dir",
        help="Local leg: sealed-receipt store for the round-trip check.",
    ),
    state_dir: Path | None = typer.Option(
        None,
        "--state-dir",
        help="Local leg: also probe durable-state replay on this dir.",
    ),
) -> None:
    """Run the harness diagnostic battery; exit 1 when any check fails.

    With ``--remote`` the battery diagnoses the deployment over the wire
    (readiness, health, wire-contract negotiation, auth enforcement, the
    managed-key lifecycle, metrics, a sealed-receipt round-trip); without
    it the same report is produced in-process against ``Fx1Harness`` —
    honesty gate, a gated completion through a deterministic injected
    backend, the managed-key lifecycle, and a sealed-receipt round-trip.
    Checks a target legitimately cannot answer report ``skipped`` and do
    not fail the run. Secrets never appear in the output.
    """
    from fx1.serve.harness_doctor import run_local_doctor, run_remote_doctor

    if remote is not None:
        from fx1.serve.client import HarnessClient

        def make_client(key: str | None) -> HarnessClient:
            return HarnessClient(remote, api_key=key, timeout_s=timeout_s)

        report = run_remote_doctor(
            make_client(api_key or os.environ.get("FX1_API_KEY") or None),
            make_client,
        )
    else:
        report = run_local_doctor(
            receipts_dir=receipts_dir,
            state_dir=state_dir,
        )
    typer.echo(json.dumps(report, indent=2))
    raise typer.Exit(code=0 if report["ok"] else 1)


@harness_app.command("serve")
def harness_serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8011, help="Bind port."),
    max_inflight: int | None = typer.Option(
        None, help="Concurrent heavy requests (env FX1_API_MAX_INFLIGHT, default 16)."
    ),
    job_max: int | None = typer.Option(
        None, help="Job-store capacity (env FX1_API_JOB_MAX, default 1024)."
    ),
    idem_max: int | None = typer.Option(
        None, help="Idempotency-store capacity (env FX1_API_IDEM_MAX, default 1024)."
    ),
    sse_keepalive_s: float | None = typer.Option(
        None,
        help="SSE keepalive interval seconds (env FX1_API_SSE_KEEPALIVE_S, default 15).",
    ),
    rate_limit_rps: float | None = typer.Option(
        None, help="Per-client req/s cap (env FX1_API_RATE_LIMIT_RPS, 0 = off)."
    ),
    gzip_min_bytes: int | None = typer.Option(
        None,
        help="Response compression floor bytes (env FX1_API_GZIP_MIN_BYTES, 0 = off).",
    ),
    cors_origins: str | None = typer.Option(
        None,
        help="Comma-separated allowed browser origins (env FX1_API_CORS_ORIGINS, empty = off).",
    ),
    breaker_threshold: int | None = typer.Option(
        None,
        help="Consecutive backend faults that open the circuit (env FX1_API_BREAKER_THRESHOLD, 0 = off).",
    ),
    breaker_cooldown_s: float | None = typer.Option(
        None,
        help="Seconds an open circuit fast-fails before a probe (env FX1_API_BREAKER_COOLDOWN_S).",
    ),
    receipts_dir: str | None = typer.Option(
        None,
        help="Sealed-receipts directory for GET /receipts fetch (env FX1_API_RECEIPTS_DIR).",
    ),
    store_max: int | None = typer.Option(
        None,
        help="/v1 retrieval-index capacity (env FX1_API_STORE_MAX, default 256).",
    ),
    state_dir: str | None = typer.Option(
        None,
        help="Durable state dir — journals async-job transitions across restarts "
        "(env FX1_API_STATE_DIR; unset = in-memory only).",
    ),
) -> None:
    """Serve the harness API (POST /harness/runs, /harness/complete, /receipts/verify)."""
    import uvicorn

    from fx1.serve.api import create_app

    if host not in {"127.0.0.1", "::1", "localhost"} and not os.environ.get("FX1_API_KEY"):
        typer.echo(
            "non-loopback binding requires FX1_API_KEY; refusing unauthenticated exposure",
            err=True,
        )
        raise typer.Exit(code=2)
    try:
        harness_api = create_app(
            max_inflight=max_inflight,
            sse_keepalive_s=sse_keepalive_s,
            idem_max=idem_max,
            job_max=job_max,
            rate_limit_rps=rate_limit_rps,
            gzip_min_bytes=gzip_min_bytes,
            cors_origins=cors_origins,
            breaker_threshold=breaker_threshold,
            breaker_cooldown_s=breaker_cooldown_s,
            receipts_dir=receipts_dir,
            store_max=store_max,
            state_dir=state_dir,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    uvicorn.run(harness_api, host=host, port=port, reload=False, server_header=False)


_BACKEND_HELP = (
    "hosted_k3 | local_fx1 | byok (BYOK reads FX1_BYOK_BASE_URL / "
    "FX1_BYOK_API_KEY / FX1_BYOK_MODEL)"
)


@harness_app.command("complete")
def harness_complete(
    prompt: str = typer.Argument(..., help="User message to complete."),
    backend: str = typer.Option("local_fx1", help=_BACKEND_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    receipt: list[str] = typer.Option([], "--receipt", help="Receipt sha256 to cite (repeatable)."),
    stream: bool = typer.Option(
        False, "--stream", help="Emit gated token deltas instead of one block."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(120.0, "--timeout", help=_TIMEOUT_HELP),
    backend_timeout: float | None = typer.Option(
        None, "--backend-timeout", help="Per-call backend deadline in seconds."
    ),
    byok_base_url: str | None = typer.Option(None, "--byok-base-url", help=_BYOK_URL_HELP),
    byok_api_key: str | None = typer.Option(None, "--byok-api-key", help=_BYOK_KEY_HELP),
    byok_model: str | None = typer.Option(None, "--byok-model", help=_BYOK_MODEL_HELP),
    fallbacks: list[str] = typer.Option(
        [],
        "--fallback",
        help="Alternate backend to try on availability faults (repeatable, max 2).",
    ),
    temperature: float | None = typer.Option(
        None, "--temperature", help="Decode temperature (default 0.0 — deterministic)."
    ),
    top_p: float | None = typer.Option(None, "--top-p", help="Nucleus sampling mass (0,1]."),
    max_tokens: int | None = typer.Option(
        None, "--max-tokens", help="Completion token cap sent to the provider."
    ),
    seed: int | None = typer.Option(
        None, "--seed", help="Decode seed passed to providers that support it."
    ),
) -> None:
    """One gated completion — the honesty gate runs before output."""
    surface = _surface(remote, api_key, timeout_s)
    byok = _byok_opts(byok_base_url, byok_api_key, byok_model)
    if stream:
        chunks = _or_exit(
            lambda: surface.stream_complete(
                [{"role": "user", "content": prompt}],
                backend=backend,
                checkpoint_dir=checkpoint_dir,
                receipt_hashes=receipt or None,
                byok=byok,
                timeout_s=backend_timeout,
                fallbacks=fallbacks or None,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
            )
        )
        for chunk in chunks:
            typer.echo(chunk, nl=False)
        typer.echo()
        return
    out = _or_exit(
        lambda: surface.complete(
            [{"role": "user", "content": prompt}],
            backend=backend,
            checkpoint_dir=checkpoint_dir,
            receipt_hashes=receipt or None,
            byok=byok,
            timeout_s=backend_timeout,
            fallbacks=fallbacks or None,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
    )
    typer.echo(out.content)


@harness_app.command("batch")
def harness_batch(
    prompts_file: Path = typer.Argument(
        ..., help="JSON array or JSONL of prompt strings to complete."
    ),
    backend: str = typer.Option("local_fx1", help=_BACKEND_HELP),
    workers: int = typer.Option(4, "--workers", help="Concurrent slots (1-16)."),
    receipt: list[str] = typer.Option([], "--receipt", help="Receipt sha256 to cite (repeatable)."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(120.0, "--timeout", help=_TIMEOUT_HELP),
    out_file: Path | None = typer.Option(
        None, "--out", help="Write the JSON results here (default: stdout)."
    ),
    byok_base_url: str | None = typer.Option(None, "--byok-base-url", help=_BYOK_URL_HELP),
    byok_api_key: str | None = typer.Option(None, "--byok-api-key", help=_BYOK_KEY_HELP),
    byok_model: str | None = typer.Option(None, "--byok-model", help=_BYOK_MODEL_HELP),
    backend_timeout: float | None = typer.Option(
        None, "--backend-timeout", help="Per-call backend deadline in seconds."
    ),
    fallbacks: list[str] = typer.Option(
        [],
        "--fallback",
        help="Alternate backend to try on availability faults (repeatable, max 2).",
    ),
    temperature: float | None = typer.Option(
        None, "--temperature", help="Decode temperature (default 0.0 — deterministic)."
    ),
    top_p: float | None = typer.Option(None, "--top-p", help="Nucleus sampling mass (0,1]."),
    max_tokens: int | None = typer.Option(
        None, "--max-tokens", help="Completion token cap sent to the provider."
    ),
    seed: int | None = typer.Option(
        None, "--seed", help="Decode seed passed to providers that support it."
    ),
) -> None:
    """Gated batch completion — per-item failures surface as exit 2."""
    surface = _surface(remote, api_key, timeout_s)
    byok = _byok_opts(byok_base_url, byok_api_key, byok_model)
    prompts = _or_exit(lambda: _load_prompts(prompts_file))
    results = _or_exit(
        lambda: surface.complete_many(
            [[{"role": "user", "content": p}] for p in prompts],
            backend=backend,
            receipt_hashes=receipt or None,
            byok=byok,
            timeout_s=backend_timeout,
            max_workers=workers,
            fallbacks=fallbacks or None,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
    )
    payload = json.dumps(
        [{"prompt": p, "content": r.content} for p, r in zip(prompts, results, strict=True)],
        indent=2,
    )
    if out_file is not None:
        out_file.write_text(payload + "\n", encoding="utf-8")
    else:
        typer.echo(payload)


def _load_prompts(path: Path) -> list[str]:
    """JSON array of strings or JSONL of str / ``{\"prompt\": ...}`` lines."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path}: empty prompts file")
    if text.startswith("["):
        items = json.loads(text)
    else:
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    prompts = [it["prompt"] if isinstance(it, dict) else it for it in items]
    if not all(isinstance(p, str) and p for p in prompts):
        raise ValueError(f"{path}: every prompt must be a non-empty string")
    return prompts


@harness_app.command("verify")
def harness_verify(
    receipt_path: Path = typer.Argument(
        ..., help="Receipt JSON file — or a directory of them — to verify."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Deep-verify receipt file(s) through the verifier surface."""
    surface = _surface(remote, api_key, timeout_s)
    if receipt_path.is_dir():
        files = sorted(receipt_path.glob("*.json"))
        if not files:
            typer.echo(f"error: no *.json receipts under {receipt_path}", err=True)
            raise typer.Exit(code=2)
        # one round-trip on the wire (receipts/verify/batch); the SDK twin
        # loops in-process — same per-item verdict shape either way.
        verdicts = _or_exit(
            lambda: surface.verify_receipts([json.loads(f.read_text()) for f in files])
        )
        results = [{"file": f.name, "valid": v.valid} for f, v in zip(files, verdicts, strict=True)]
        typer.echo(
            json.dumps(
                {
                    "files": len(results),
                    "valid": sum(1 for r in results if r["valid"]),
                    "results": results,
                },
                indent=2,
            )
        )
        raise typer.Exit(code=0 if all(r["valid"] for r in results) else 1)
    verdict = _or_exit(lambda: surface.verify_receipt(json.loads(receipt_path.read_text())))
    typer.echo(
        json.dumps(
            {
                "valid": verdict.valid,
                "errors": list(verdict.errors),
                "warnings": list(verdict.warnings),
                "schema": verdict.schema_tag,
            },
            indent=2,
        )
    )
    raise typer.Exit(code=0 if verdict.valid else 1)


@harness_app.command("score")
def harness_score(
    inputs: list[str] = typer.Argument(..., help="Text to score (repeatable)."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /harness/score`` — the deterministic reward contract as a
    preflight surface: per-input gate categories + reward signal, no model
    spend. In-process by default."""
    inp: str | list[str] = inputs[0] if len(inputs) == 1 else list(inputs)
    surface = _surface(remote, api_key, timeout_s)
    out = _or_exit(lambda: surface.score(inp))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("commands")
def harness_commands(
    role: str | None = typer.Option(
        None, "--role", help="Filter to one role's reachable commands."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /harness/commands`` — registered command names; anything
    unlisted is unreachable on that role."""
    surface = _surface(remote, api_key, timeout_s)
    # lazy: keeps the fx1.harness import out of CLI startup
    from fx1.harness import HarnessRole

    role_val: HarnessRole | None = None
    if role is not None:
        # bogus role names are an arg fault, same as the wire's 422.
        role_val = _or_exit(lambda: HarnessRole(role))
    names = _or_exit(lambda: surface.commands(role=role_val))
    typer.echo(json.dumps({"count": len(names), "commands": names}, indent=2))


@harness_app.command("receipts")
def harness_receipts(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    receipts_dir: Path = typer.Option(
        Path("receipts"),
        "--receipts-dir",
        help="Local sealed-receipts store (ignored with --remote).",
    ),
) -> None:
    """Index the sealed-receipt store — a local dir or the remote's over HTTP."""
    if remote is None:
        from fx1.sdk import Fx1Harness

        surface: Fx1Harness | HarnessClient = Fx1Harness(receipts_dir=receipts_dir)
    else:
        surface = _remote_client(remote, api_key, timeout_s)
    items = _or_exit(lambda: surface.receipts())
    typer.echo(
        json.dumps(
            {
                "count": len(items),
                "items": [{"sha256": r.sha256, "name": r.name} for r in items],
            },
            indent=2,
        )
    )


@harness_app.command("receipt")
def harness_receipt(
    sha256: str = typer.Argument(..., help="Receipt content hash (64 lowercase hex)."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    receipts_dir: Path = typer.Option(
        Path("receipts"),
        "--receipts-dir",
        help="Local sealed-receipts store (ignored with --remote).",
    ),
) -> None:
    """Fetch one sealed receipt by content hash: verbatim document + validity."""
    if remote is None:
        from fx1.sdk import Fx1Harness

        surface: Fx1Harness | HarnessClient = Fx1Harness(receipts_dir=receipts_dir)
    else:
        surface = _remote_client(remote, api_key, timeout_s)
    r = _or_exit(lambda: surface.receipt(sha256))
    typer.echo(json.dumps({"sha256": r.sha256, "valid": r.valid, "receipt": r.document}, indent=2))


@harness_app.command("probe")
def harness_probe(
    backend: str = typer.Option("byok", help=_BACKEND_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    prompt: str = typer.Option("ping", "--prompt", help="Probe prompt."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    backend_timeout: float | None = typer.Option(
        None, "--backend-timeout", help="Per-call backend deadline in seconds (route default: 30)."
    ),
    byok_base_url: str | None = typer.Option(None, "--byok-base-url", help=_BYOK_URL_HELP),
    byok_api_key: str | None = typer.Option(None, "--byok-api-key", help=_BYOK_KEY_HELP),
    byok_model: str | None = typer.Option(None, "--byok-model", help=_BYOK_MODEL_HELP),
) -> None:
    """Deep health: one live gated completion per call — prints the
    verdict JSON and exits 0 when ok, 1 when the backend is unhealthy.
    ``--remote`` probes via the wire route (probe verdicts never feed
    the remote's circuit breaker); in-process probes run the same
    resolver the SDK uses, so a BYOK probe tests your own endpoint."""
    surface = _surface(remote, api_key, timeout_s)
    byok = _byok_opts(byok_base_url, byok_api_key, byok_model)
    out = _or_exit(
        lambda: surface.probe_backend(
            backend,
            checkpoint_dir=checkpoint_dir,
            byok=byok,
            timeout_s=backend_timeout,
            prompt=prompt,
        )
    )
    typer.echo(
        json.dumps(
            {
                "backend": out.backend,
                "ok": out.ok,
                "model": out.model,
                "latency_ms": round(out.latency_ms, 1),
                "error": out.error,
                "error_class": out.error_class,
            },
            indent=2,
        )
    )
    if not out.ok:
        raise typer.Exit(code=1)


@harness_app.command("check-text")
def harness_check_text(
    text: str = typer.Argument(..., help="Text to run through the honesty gate."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Pre-flight text through the honesty gate — prints the verdict JSON
    and exits 0 when clean / 1 on refusal. In-process (no backend needed);
    ``--remote`` checks via the wire route."""
    surface = _surface(remote, api_key, timeout_s)
    out = _or_exit(lambda: surface.check_text(text))
    typer.echo(json.dumps({"ok": out.ok, "error": out.error}, indent=2))
    if not out.ok:
        raise typer.Exit(code=1)


def _record_json(rec: Any) -> dict[str, Any]:
    return {
        "completion_id": rec.completion_id,
        "backend": rec.backend,
        "model": rec.model,
        "ok": rec.ok,
        "latency_ms": rec.latency_ms,
        "at": rec.at,
        "usage": rec.usage,
        "error": rec.error,
        "error_class": rec.error_class,
        "prompt_sha256": rec.prompt_sha256,
        "output_sha256": rec.output_sha256,
    }


@harness_app.command("completions")
def harness_completions(
    limit: int = typer.Option(50, "--limit", help="Newest N records (log is ring-bounded)."),
    backend: str | None = typer.Option(None, "--backend", help=_BACKEND_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Newest-first window on the completion log — per-call evidence
    (hashes, usage, verdict) for every gated call this surface served."""
    surface = _surface(remote, api_key, timeout_s)
    items = _or_exit(lambda: surface.completions(limit=limit, backend=backend))
    typer.echo(
        json.dumps({"count": len(items), "items": [_record_json(r) for r in items]}, indent=2)
    )


@harness_app.command("completion")
def harness_completion(
    completion_id: str = typer.Argument(..., help="Completion record id (hex)."),
    receipt: bool = typer.Option(
        False, "--receipt", help="Print the sealed fx1_completion_record.v1 doc instead."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Fetch one recorded call by id — the audit handle minted per call.
    ``--receipt`` prints the sealed export instead (verify with
    ``verify-research`` or ``POST /receipts/verify``)."""
    surface = _surface(remote, api_key, timeout_s)
    if receipt:
        doc = _or_exit(lambda: surface.completion_receipt(completion_id))
        typer.echo(json.dumps(doc, indent=2, sort_keys=True))
        return
    rec = _or_exit(lambda: surface.completion(completion_id))
    typer.echo(json.dumps(_record_json(rec), indent=2))


@harness_app.command("health")
def harness_health(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Configured-backend presence booleans — never secret values."""
    surface = _surface(remote, api_key, timeout_s)
    h = _or_exit(surface.health)
    typer.echo(
        json.dumps(
            {
                "status": h.status,
                "version": h.version,
                "registered_commands": h.registered_commands,
                "backends": h.backends,
            },
            indent=2,
        )
    )


@harness_app.command("metrics")
def harness_metrics(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    format: str = typer.Option(
        "json", "--format", help="Output format: 'json' or 'prom' (Prometheus text)"
    ),
) -> None:
    """Remote ops counters — requires --remote (nothing meters in-process)."""
    if remote is None:
        typer.echo(
            "error: harness metrics is a wire-ops surface; pass --remote",
            err=True,
        )
        raise typer.Exit(code=2)
    if format not in ("json", "prom"):
        typer.echo(f"error: --format must be 'json' or 'prom', got {format!r}", err=True)
        raise typer.Exit(code=2)
    from fx1.serve.client import HarnessClient

    client = HarnessClient(
        remote,
        api_key=api_key or os.environ.get("FX1_API_KEY") or None,
        timeout_s=timeout_s,
    )
    if format == "prom":
        typer.echo(_or_exit(client.metrics_text))
        return
    m = _or_exit(client.metrics)
    typer.echo(
        json.dumps(
            {
                "uptime_s": m.uptime_s,
                "requests_total": m.requests_total,
                "errors_total": m.errors_total,
                "by_status": m.by_status,
                "inflight": m.inflight,
                "inflight_watermark": m.inflight_watermark,
                "max_inflight": m.max_inflight,
                "draining": m.draining,
                "rate_limited_total": m.rate_limited_total,
                "complete": m.complete,
            },
            indent=2,
        )
    )


@harness_app.command("drain")
def harness_drain(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    wait_s: float = typer.Option(
        0.0,
        "--wait-s",
        help="Block server-side up to N seconds for in-flight work to drain.",
    ),
) -> None:
    """Latch the remote harness into drain mode — one-way, idempotent.

    Gated routes start refusing new work with 503 while in-flight
    requests finish; ``--wait-s N`` blocks until the pool empties and
    reports ``drained``; otherwise watch ``harness metrics --remote``
    until ``inflight`` reaches zero, then stop the process.
    """
    if remote is None:
        typer.echo(
            "error: harness drain is a wire-ops surface; pass --remote",
            err=True,
        )
        raise typer.Exit(code=2)
    from fx1.serve.client import HarnessClient

    client = HarnessClient(
        remote,
        api_key=api_key or os.environ.get("FX1_API_KEY") or None,
        timeout_s=timeout_s,
    )
    typer.echo(json.dumps(_or_exit(lambda: client.drain(wait_s=wait_s)), indent=2))


@harness_app.command("version")
def harness_version(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Wire-contract + package versions — remote mode asks the server's
    /harness/version, local mode prints this install's own pair."""
    if remote is not None:
        client = _remote_client(remote, api_key, timeout_s)
        out = _or_exit(lambda: client.server_version())
        # the wire's request-id for this very call — the trace surface a
        # deploy loop would paste into a bug report
        rid = _last_request_id(client)
        if rid is not None:
            out = {**out, "request_id": rid}
        typer.echo(json.dumps(out))
        return
    from fx1 import __version__
    from fx1.serve.contract import API_VERSION

    typer.echo(json.dumps({"api_version": API_VERSION, "fx1_version": __version__, "local": True}))


@harness_app.command("selftest")
def harness_selftest(
    remote: str | None = typer.Option(
        None,
        "--remote",
        help="Smoke a live deployment (read-only checks); unset boots a "
        "stub engine + the production app on loopback and runs the full "
        "golden path.",
    ),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    state_dir: str | None = typer.Option(
        None,
        "--state-dir",
        help="Also prove durability: restart the local app on this dir and "
        "recover the job record (local mode only).",
    ),
) -> None:
    """Golden-path smoke: zero-config proof the harness actually works.

    Local mode spins a stub OpenAI engine and the production app on
    loopback, then walks auth, commands, a BYOK completion, SSE stream,
    idempotent replay, the async job lifecycle, sealed-receipt verify,
    drain, and in-process parity. Remote mode runs read-only checks only.
    Exits 0 when every check passes, 2 otherwise — usable as a deploy gate."""
    from fx1.selftest import run_selftest

    report = run_selftest(
        remote=remote,
        api_key=api_key or os.environ.get("FX1_API_KEY") or None,
        state_dir=state_dir,
        timeout_s=timeout_s,
    )
    typer.echo(json.dumps(report.as_dict(), indent=2))
    if not report.ok:
        raise typer.Exit(code=2)


@harness_app.command("bench")
def harness_bench(
    remote: str | None = typer.Option(
        None,
        "--remote",
        help="Bench a live deployment; unset times the in-process SDK.",
    ),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    n: int = typer.Option(32, "--n", help="Measured requests (1..4096)."),
    concurrency: int = typer.Option(4, "--concurrency", help="Worker pool size (1..256)."),
    warmup: int = typer.Option(2, "--warmup", help="Unmeasured warmup requests (0..256)."),
    prompt: str = typer.Option(
        "", "--prompt", help="Probe prompt (default: a fixed summary prompt)."
    ),
    max_tokens: int = typer.Option(32, "--max-tokens", help="Per-request token cap."),
    backend: str = typer.Option("local_fx1", "--backend", help=_BACKEND_HELP),
    seed: int | None = typer.Option(
        None, "--seed", help="Sampling seed forwarded to each request."
    ),
    byok_base_url: str | None = typer.Option(None, "--byok-base-url", help=_BYOK_URL_HELP),
    byok_api_key: str | None = typer.Option(None, "--byok-api-key", help=_BYOK_KEY_HELP),
    byok_model: str | None = typer.Option(None, "--byok-model", help=_BYOK_MODEL_HELP),
    receipt: bool = typer.Option(
        False,
        "--receipt",
        help="Print the sealed fx1_bench_result.v1 doc instead of the raw record.",
    ),
) -> None:
    """Perf gate: time ``n`` gated completions at ``--concurrency`` workers.

    Prints the latency card (p50/p90/p95/p99/max/mean), throughput, token
    rates, and an error histogram as JSON; ``--receipt`` prints the sealed
    fx1_bench_result.v1 document (verify with ``dipcatcher verify-receipt``).
    Exits 0 when every measured request succeeded, 1 when any failed —
    usable as a deploy gate alongside ``harness selftest``."""
    from fx1.harness_bench import DEFAULT_BENCH_PROMPT, run_bench
    from fx1.serve.ops_receipt import bench_receipt

    surface = _surface(remote, api_key or os.environ.get("FX1_API_KEY"), timeout_s)
    record = _or_exit(
        lambda: run_bench(
            surface,
            n=n,
            concurrency=concurrency,
            warmup=warmup,
            prompt=prompt or DEFAULT_BENCH_PROMPT,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            backend=backend,
            byok=_byok_opts(byok_base_url, byok_api_key, byok_model),
            seed=seed,
            mode="remote" if remote else "in_process",
        )
    )
    typer.echo(json.dumps(bench_receipt(record) if receipt else record, indent=2, sort_keys=True))
    if record["metrics"]["error_count"]:
        raise typer.Exit(code=1)


@harness_app.command("usage")
def harness_usage(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    backend: str | None = typer.Option(None, "--backend", help=_BACKEND_HELP),
    model: str | None = typer.Option(None, "--model", help="Only count this reported model."),
    since: float | None = typer.Option(
        None, "--since", help="Unix-second lower bound on record timestamps."
    ),
    until: float | None = typer.Option(
        None, "--until", help="Unix-second upper bound on record timestamps."
    ),
    key_id: str | None = typer.Option(
        None, "--key-id", help="Only count calls made under this key fingerprint."
    ),
) -> None:
    """Usage accounting: token/request aggregates over the completion log.

    Totals plus per-backend/per-model splits as JSON — the billing/ops
    view of every gated call the surface served. ``records_dropped`` in
    the report declares a truncated ring window. ``--remote`` reads the
    server's log (``GET /harness/usage``); default reads the in-process
    SDK's own log."""
    surface = _surface(remote, api_key or os.environ.get("FX1_API_KEY"), timeout_s)
    report = _or_exit(
        lambda: surface.usage(backend=backend, model=model, key_id=key_id, since=since, until=until)
    )
    typer.echo(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))


@harness_app.command("key-create")
def harness_key_create(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    name: str | None = typer.Option(None, "--name", help="Label for the key (≤128 chars)."),
    admin: bool = typer.Option(
        False,
        "--admin",
        help="Mint an admin key — it may itself mint/list/revoke keys.",
    ),
    rpm: int | None = typer.Option(
        None,
        "--rpm",
        min=1,
        help="Bound the key to this many requests per 60 s window (over-limit → 429).",
    ),
    ttl_s: float | None = typer.Option(
        None,
        "--ttl-s",
        help="Key expiry in seconds from mint — the credential dies after.",
    ),
    scopes: list[str] = typer.Option(
        [],
        "--scope",
        help="Bound the key to a surface class: read | write | admin (repeatable; unset = read+write).",
    ),
    max_requests: int | None = typer.Option(
        None,
        "--max-requests",
        min=1,
        help="Hard budget: the key refuses 429 quota_exceeded after this many authenticated calls.",
    ),
    max_tokens: int | None = typer.Option(
        None,
        "--max-tokens",
        min=1,
        help="Hard budget: refuses once the key's provider-reported token spend reaches this.",
    ),
) -> None:
    """Mint a managed API key — prints the mint record including the raw
    ``key``, which is shown once and never stored server-side. On
    ``--remote`` this needs the bootstrap credential (FX1_API_KEY)."""
    surface = _surface(remote, api_key or os.environ.get("FX1_API_KEY"), timeout_s)
    out = _or_exit(
        lambda: surface.key_create(
            name,
            admin=admin,
            rpm=rpm,
            ttl_s=ttl_s,
            scopes=scopes or None,
            max_requests=max_requests,
            max_tokens=max_tokens,
        )
    )
    typer.echo(json.dumps(out, indent=2, sort_keys=True))


@harness_app.command("keys")
def harness_keys(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """List managed API keys — fingerprint ids + metadata, never secrets."""
    surface = _surface(remote, api_key or os.environ.get("FX1_API_KEY"), timeout_s)
    out = _or_exit(lambda: surface.keys())
    typer.echo(json.dumps(out, indent=2, sort_keys=True))


@harness_app.command("key-get")
def harness_key_get(
    key_id: str,
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """One managed key's record by its fingerprint id."""
    surface = _surface(remote, api_key or os.environ.get("FX1_API_KEY"), timeout_s)
    out = _or_exit(lambda: surface.key_get(key_id))
    typer.echo(json.dumps(out, indent=2, sort_keys=True))


@harness_app.command("key-revoke")
def harness_key_revoke(
    key_id: str,
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Tombstone a managed key — auth with it fails closed immediately
    after; the record stays for audit."""
    surface = _surface(remote, api_key or os.environ.get("FX1_API_KEY"), timeout_s)
    out = _or_exit(lambda: surface.key_revoke(key_id))
    typer.echo(json.dumps(out, indent=2, sort_keys=True))


@harness_app.command("key-usage")
def harness_key_usage(
    key_id: str,
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """One managed key's usage card — live counters, budget headroom,
    rate-window state, and the completion-ring spend split. Admin on
    the wire."""
    surface = _surface(remote, api_key or os.environ.get("FX1_API_KEY"), timeout_s)
    out = _or_exit(lambda: surface.key_usage(key_id))
    typer.echo(json.dumps(out, indent=2, sort_keys=True))


@harness_app.command("key-rotate")
def harness_key_rotate(
    key_id: str,
    revoke_old: bool = typer.Option(
        True,
        "--revoke-old/--keep-old",
        help="Tombstone the predecessor atomically with the mint (default). "
        "--keep-old leaves both secrets live until the old key is revoked or expires.",
    ),
    name: str | None = typer.Option(
        None, "--name", help="Successor display name — defaults to the predecessor's."
    ),
    ttl_s: float | None = typer.Option(
        None,
        "--ttl-s",
        min=1e-9,
        help="Fresh lifetime for the successor (seconds). Omitted: inherits the "
        "predecessor's absolute expires_at — rotation never extends a credential.",
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Rotate a managed key — mints a successor under the predecessor's
    declared policy and, by default, tombstones the old secret in the
    same transaction. The response's ``key.key`` is the only place the
    new secret appears."""
    surface = _surface(remote, api_key or os.environ.get("FX1_API_KEY"), timeout_s)
    out = _or_exit(
        lambda: surface.key_rotate(key_id, revoke_old=revoke_old, name=name, ttl_s=ttl_s)
    )
    typer.echo(json.dumps(out, indent=2, sort_keys=True))


@harness_app.command("key-patch")
def harness_key_patch(
    key_id: str,
    name: str | None = typer.Option(
        None, "--name", help="New display name — unset keeps the declared name."
    ),
    rpm: int | None = typer.Option(
        None,
        "--rpm",
        min=1,
        help="New per-60s request bound — unset keeps the declared one.",
    ),
    scopes: list[str] = typer.Option(
        [],
        "--scope",
        help="Replace the surface classes: read | write | admin (repeatable).",
    ),
    admin: bool | None = typer.Option(
        None,
        "--admin/--no-admin",
        help="--admin unions the admin scope onto the surviving list; "
        "--no-admin never strips a declared scope (purely additive, like mint).",
    ),
    max_requests: int | None = typer.Option(
        None, "--max-requests", min=1, help="New authenticated-call budget."
    ),
    max_tokens: int | None = typer.Option(
        None, "--max-tokens", min=1, help="New provider-reported token budget."
    ),
    expires_at: float | None = typer.Option(
        None,
        "--expires-at",
        min=1e-9,
        help="New absolute expiry (unix seconds).",
    ),
    clear: list[str] = typer.Option(
        [],
        "--clear",
        help="Clear a nullable bound back to unbounded: name | rpm | "
        "max_requests | max_tokens | expires_at (repeatable).",
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Patch a live managed key's policy in place — prints the updated
    record. Unset flags keep the declared policy; ``--clear FIELD``
    sends the JSON-null that unbounds it; ``enabled`` and the live
    counters are not patchable — revocation is permanent."""
    kw: dict[str, Any] = {}
    if name is not None:
        kw["name"] = name
    if rpm is not None:
        kw["rpm"] = rpm
    if scopes:
        kw["scopes"] = scopes
    if admin is not None:
        kw["admin"] = admin
    if max_requests is not None:
        kw["max_requests"] = max_requests
    if max_tokens is not None:
        kw["max_tokens"] = max_tokens
    if expires_at is not None:
        kw["expires_at"] = expires_at
    if clear:
        kw["clear"] = clear
    surface = _surface(remote, api_key or os.environ.get("FX1_API_KEY"), timeout_s)
    out = _or_exit(lambda: surface.key_update(key_id, **kw))
    typer.echo(json.dumps(out, indent=2, sort_keys=True))


@harness_app.command("self")
def harness_self(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """The calling credential's own card — its class (managed / env /
    loopback) plus, for managed keys, live budget headroom. Read scope."""
    surface = _surface(remote, api_key or os.environ.get("FX1_API_KEY"), timeout_s)
    out = _or_exit(lambda: surface.self_usage())
    typer.echo(json.dumps(out, indent=2, sort_keys=True))


@harness_app.command("compat")
def harness_compat(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Wire-contract negotiation: prints the compat report and exits 0 when
    the remote speaks this client's api_version, 1 on any mismatch (incl.
    a peer too old to have a version route)."""
    if remote is None:
        typer.echo(
            "error: harness compat is a wire-ops surface; pass --remote",
            err=True,
        )
        raise typer.Exit(code=2)
    client = _remote_client(remote, api_key, timeout_s)
    report = _or_exit(lambda: client.check_compat(strict=False))
    rid = _last_request_id(client)
    if rid is not None:
        report = {**report, "request_id": rid}
    typer.echo(json.dumps(report))
    if not report["compatible"]:
        raise typer.Exit(code=1)


@harness_app.command("capabilities")
def harness_capabilities(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Server self-description: wire features + effective limits, so a
    deploy can sanity-check the remote's config before routing work."""
    if remote is None:
        typer.echo(
            "error: harness capabilities is a wire-ops surface; pass --remote",
            err=True,
        )
        raise typer.Exit(code=2)
    client = _remote_client(remote, api_key, timeout_s)
    typer.echo(json.dumps(_or_exit(lambda: client.capabilities())))


@harness_app.command("ready")
def harness_ready(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Readiness probe: exits 0 while the remote accepts work, 1 once
    draining — for deploy scripts and Kubernetes readiness gates."""
    if remote is None:
        typer.echo(
            "error: harness ready is a wire-ops surface; pass --remote",
            err=True,
        )
        raise typer.Exit(code=2)
    from fx1.serve.backends import BackendNotConfiguredError
    from fx1.serve.client import HarnessClient

    client = HarnessClient(
        remote,
        api_key=api_key or os.environ.get("FX1_API_KEY") or None,
        timeout_s=timeout_s,
    )
    try:
        out = client.ready()
    except BackendNotConfiguredError:
        typer.echo(json.dumps({"ready": False}))
        raise typer.Exit(code=1) from None
    typer.echo(json.dumps(out))


def _remote_client(remote: str, api_key: str | None, timeout_s: float) -> HarnessClient:
    from fx1.serve.client import HarnessClient

    return HarnessClient(
        remote,
        api_key=api_key or os.environ.get("FX1_API_KEY") or None,
        timeout_s=timeout_s,
    )


def _need_remote(remote: str | None) -> None:
    if remote is None:
        typer.echo(
            "error: async jobs are a wire surface; pass --remote",
            err=True,
        )
        raise typer.Exit(code=2)


@harness_app.command("submit")
def harness_submit(
    name: str = typer.Argument(..., help="Registered harness command name."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    idempotency_key: str | None = typer.Option(
        None,
        "--idempotency-key",
        help="Dedup key for the submission; a retried submit returns the same job id.",
    ),
    callback_url: str | None = typer.Option(
        None,
        "--callback-url",
        help="http(s) webhook; the full job record is POSTed on terminal status.",
    ),
    callback_secret: str | None = typer.Option(
        None,
        "--callback-secret",
        help="HMAC secret signing the webhook (X-Fx1-Webhook-Signature).",
    ),
) -> None:
    """Submit a run as a background job; prints the job id."""
    _need_remote(remote)
    job_id = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).submit_run(
            name,
            idempotency_key=idempotency_key,
            callback_url=callback_url,
            callback_secret=callback_secret,
        )
    )
    typer.echo(job_id)


@harness_app.command("submit-batch")
def harness_submit_batch(
    spec: Path = typer.Argument(
        ..., help="JSON file: a list of run-request objects (command, extra_args, …)."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Submit a batch of jobs in one request; prints the per-item outcome JSON."""
    _need_remote(remote)
    try:
        jobs = json.loads(spec.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        typer.echo(f"cannot read batch spec {spec}: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if not isinstance(jobs, list):
        typer.echo("batch spec must be a JSON list of run-request objects", err=True)
        raise typer.Exit(code=2)
    out = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).submit_batch(jobs))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("job")
def harness_job(
    job_id: str = typer.Argument(..., help="Job id returned by harness submit."),
    receipt: bool = typer.Option(
        False, "--receipt", help="Print the sealed fx1_job_record.v1 doc instead."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Print a job's live status record. ``--receipt`` prints the sealed
    export instead (verify with ``verify-research`` /
    ``POST /receipts/verify``)."""
    _need_remote(remote)
    if receipt:
        doc = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).job_receipt(job_id))
        typer.echo(json.dumps(doc, indent=2, sort_keys=True))
        return
    st = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).job_status(job_id))
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("jobs")
def harness_jobs(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    status: str | None = typer.Option(
        None, "--status", help="Filter: queued|running|succeeded|failed|cancelled."
    ),
    limit: int = typer.Option(100, "--limit", help="Page size (max 500)."),
    offset: int = typer.Option(0, "--offset", help="Skip the newest N jobs."),
) -> None:
    """List the job inventory (newest first) with total for paging."""
    _need_remote(remote)
    page = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).list_jobs(
            status=status, limit=limit, offset=offset
        )
    )
    typer.echo(json.dumps(page, indent=2))


@harness_app.command("cancel")
def harness_cancel(
    job_id: str = typer.Argument(..., help="Job id returned by harness submit."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Cancel a queued job; running/terminal jobs report a 409 conflict."""
    _need_remote(remote)
    st = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).cancel_job(job_id))
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("wait")
def harness_wait(
    job_id: str = typer.Argument(..., help="Job id returned by harness submit."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    poll_s: float = typer.Option(0.5, "--poll", help=_POLL_HELP),
    wait_timeout_s: float | None = typer.Option(
        None, "--wait-timeout", help="Give up waiting after N seconds (job keeps running)."
    ),
) -> None:
    """Wait for a job to finish, then print its stdout like harness run."""
    _need_remote(remote)
    result = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).wait_run(
            job_id, poll_s=poll_s, timeout_s=wait_timeout_s
        )
    )
    typer.echo(result.stdout)
    if result.stderr:
        typer.echo(result.stderr, err=True)
    raise typer.Exit(code=result.exit_code)


@harness_app.command("watch")
def harness_watch(
    job_id: str = typer.Argument(..., help="Job id returned by harness submit."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    stream_timeout_s: float = typer.Option(
        600.0, "--stream-timeout", help="Server-side SSE stream bound, seconds."
    ),
) -> None:
    """Follow a job over the SSE event stream — one connection, no polling."""
    _need_remote(remote)
    frames = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).stream_job(
            job_id, timeout_s=stream_timeout_s
        )
    )
    for st in frames:
        typer.echo(f"{st['status']}\t{job_id}")
    last = frames[-1]
    if last["status"] == "succeeded":
        result = last["result"]
        typer.echo(result["stdout"])
        if result["stderr"]:
            typer.echo(result["stderr"], err=True)
        raise typer.Exit(code=result["exit_code"])
    if last["status"] == "failed":
        typer.echo(f"job failed: {last.get('error')}", err=True)
        raise typer.Exit(code=1)
    if last["status"] == "cancelled":
        typer.echo("job cancelled", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"stream ended before terminal state (last status {last['status']!r})", err=True)
    raise typer.Exit(code=1)


@harness_app.command("eval")
def harness_eval(
    suite: str = typer.Argument(
        ...,
        help="Eval suite: capability|calibration|tooluse|retrieval|ts_reasoning|ext_bench|options_reasoning.",
    ),
    backend: str = typer.Option("local_fx1", help=_BACKEND_HELP),
    seed: int = typer.Option(0, "--seed", help="Eval seed (the banks are seeded)."),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(120.0, "--timeout", help=_TIMEOUT_HELP),
    backend_timeout: float | None = typer.Option(
        None, "--backend-timeout", help="Per-call backend deadline in seconds."
    ),
    byok_base_url: str | None = typer.Option(None, "--byok-base-url", help=_BYOK_URL_HELP),
    byok_api_key: str | None = typer.Option(None, "--byok-api-key", help=_BYOK_KEY_HELP),
    byok_model: str | None = typer.Option(None, "--byok-model", help=_BYOK_MODEL_HELP),
    fallbacks: list[str] = typer.Option(
        [],
        "--fallback",
        help="Alternate backend on availability faults (repeatable, max 2).",
    ),
    judge_backend: str | None = typer.Option(
        None, "--judge-backend", help="Grader link for judge suites (capability, ext_bench)."
    ),
    receipt: bool = typer.Option(
        False, "--receipt", help="Print the sealed fx1_eval_record.v1 doc after the run."
    ),
    no_wait: bool = typer.Option(
        False, "--no-wait", help="Remote only: submit and return immediately."
    ),
    callback_url: str | None = typer.Option(
        None,
        "--callback-url",
        help="Remote only: http(s) URL the finished eval record is POSTed to.",
    ),
    callback_secret: str | None = typer.Option(
        None,
        "--callback-secret",
        help="Remote only: HMAC secret signing the callback delivery.",
    ),
) -> None:
    """Run a seeded eval suite against a backend — in-process by default
    (SDK twin), or ``--remote`` submits to POST /harness/evals and waits
    for the terminal record. Every eval runs under the temperature=0
    pin and lands as metered evidence on the eval/completion logs."""
    byok = _byok_opts(byok_base_url, byok_api_key, byok_model)
    if remote is not None:
        from fx1.serve.client import HarnessClient

        client = HarnessClient(
            remote,
            api_key=api_key or os.environ.get("FX1_API_KEY") or None,
            timeout_s=timeout_s,
        )
        sub = _or_exit(
            lambda: client.submit_eval(
                suite,
                backend=backend,
                seed=seed,
                checkpoint_dir=str(checkpoint_dir) if checkpoint_dir else None,
                byok=byok,
                timeout_s=backend_timeout,
                fallbacks=fallbacks or None,
                judge_backend=judge_backend,
                callback_url=callback_url,
                callback_secret=callback_secret,
            )
        )
        if no_wait:
            typer.echo(json.dumps(sub, indent=2))
            return
        rec = _or_exit(lambda: client.wait_eval(sub["eval_id"], timeout_s=None))
        typer.echo(json.dumps(rec, indent=2))
        if receipt:
            doc = _or_exit(lambda: client.eval_receipt(sub["eval_id"]))
            typer.echo(json.dumps(doc, indent=2, sort_keys=True))
        return
    from fx1.sdk import Fx1Harness

    if callback_url is not None or callback_secret is not None:
        typer.echo(
            "--callback-url/--callback-secret are remote-only (webhooks need the server)",
            err=True,
        )
        raise typer.Exit(code=2)
    harness = Fx1Harness()
    ev_rec = _or_exit(
        lambda: harness.run_eval(
            suite,
            backend=backend,
            checkpoint_dir=checkpoint_dir,
            byok=byok,
            timeout_s=backend_timeout,
            fallbacks=fallbacks or None,
            judge_backend=judge_backend,
            seed=seed,
        )
    )
    typer.echo(ev_rec.model_dump_json(indent=2))
    if receipt:
        doc = _or_exit(lambda: harness.eval_receipt(ev_rec.eval_id))
        typer.echo(json.dumps(doc, indent=2, sort_keys=True))


@harness_app.command("evals")
def harness_evals(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    status: str | None = typer.Option(
        None, "--status", help="Filter: queued|running|succeeded|failed|cancelled."
    ),
    suite: str | None = typer.Option(None, "--suite", help="Filter by suite name."),
    limit: int = typer.Option(100, "--limit", help="Page size (max 256)."),
) -> None:
    """List the remote eval inventory (newest first)."""
    _need_remote(remote)
    page = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).list_evals(
            status=status, suite=suite, limit=limit
        )
    )
    typer.echo(json.dumps(page, indent=2))


@harness_app.command("eval-status")
def harness_eval_status(
    eval_id: str = typer.Argument(..., help=_EVAL_ID_REMOTE_HELP),
    receipt: bool = typer.Option(
        False, "--receipt", help="Print the sealed fx1_eval_record.v1 doc instead."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Print an eval's live record; ``--receipt`` prints the sealed
    export (409 while the eval is non-terminal)."""
    _need_remote(remote)
    if receipt:
        doc = _or_exit(
            lambda: _remote_client(remote or "", api_key, timeout_s).eval_receipt(eval_id)
        )
        typer.echo(json.dumps(doc, indent=2, sort_keys=True))
        return
    st = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).eval_status(eval_id))
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("eval-cancel")
def harness_eval_cancel(
    eval_id: str = typer.Argument(..., help=_EVAL_ID_REMOTE_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Cancel a queued eval; running/terminal evals report a 409 conflict."""
    _need_remote(remote)
    st = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).cancel_eval(eval_id))
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("eval-wait")
def harness_eval_wait(
    eval_id: str = typer.Argument(..., help=_EVAL_ID_REMOTE_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    poll_s: float = typer.Option(0.5, "--poll", help=_POLL_HELP),
    wait_timeout_s: float | None = typer.Option(
        None, "--wait-timeout", help="Give up waiting after N seconds (eval keeps running)."
    ),
    receipt: bool = typer.Option(
        False, "--receipt", help="Print the sealed fx1_eval_record.v1 doc after the record."
    ),
) -> None:
    """Re-attach to an eval submitted ``--no-wait`` and poll to terminal;
    prints the terminal record (report carries the suite output)."""
    _need_remote(remote)
    client = _remote_client(remote or "", api_key, timeout_s)
    rec = _or_exit(lambda: client.wait_eval(eval_id, poll_s=poll_s, timeout_s=wait_timeout_s))
    typer.echo(json.dumps(rec, indent=2))
    if receipt:
        doc = _or_exit(lambda: client.eval_receipt(eval_id))
        typer.echo(json.dumps(doc, indent=2, sort_keys=True))


@harness_app.command("eval-diff")
def harness_eval_diff(
    base_id: str = typer.Argument(..., help="Baseline eval id (terminal record)."),
    candidate_id: str = typer.Argument(..., help="Candidate eval id (terminal record)."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Promotion-gate diff over two terminal evals: per-task transitions,
    gate move, by_kind deltas, and a verdict. ``comparable`` needs the
    same suite+seed and no stamped bank mismatch; non-terminal records
    report a 409 conflict."""
    _need_remote(remote)
    st = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).diff_evals(base_id, candidate_id)
    )
    typer.echo(json.dumps(st, indent=2))


# ---- /v1/evals — the OpenAI Evals-shaped spec/run surface ------------------


@harness_app.command("eval-spec-create")
def harness_eval_spec_create(
    name: str = typer.Argument(..., help="Human label for the eval container."),
    suite: str = typer.Option(
        ...,
        "--suite",
        help="Eval suite: capability|calibration|tooluse|retrieval|ts_reasoning|ext_bench|options_reasoning.",
    ),
    seed: int = typer.Option(0, "--seed", help="Eval seed (the banks are seeded)."),
    backend: str | None = typer.Option(
        None, "--backend", help="Pin the chain head (runs may still choose another model)."
    ),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    judge_backend: str | None = typer.Option(
        None, "--judge-backend", help="Grader link for judge suites."
    ),
    backend_timeout: float | None = typer.Option(
        None, "--backend-timeout", help="Per-call backend deadline in seconds."
    ),
    criteria_json: str | None = typer.Option(
        None, "--criteria", help='testing_criteria as a JSON list, e.g. \'[{"name": "all-pass"}]\'.'
    ),
    metadata_json: str | None = typer.Option(
        None, "--metadata", help="Spec metadata as a JSON object of string pairs."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Declare a named eval container — ``POST /v1/evals`` (remote) or the
    in-process twin. The item_schema pins suite knobs; credentials never
    live on a spec."""
    criteria = _json_list_opt(criteria_json, "--criteria")
    metadata = _json_meta(metadata_json)
    if remote is not None:
        spec = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).eval_spec_create(
                name,
                suite=suite,
                seed=seed,
                backend=backend,
                checkpoint_dir=str(checkpoint_dir) if checkpoint_dir else None,
                judge_backend=judge_backend,
                timeout_s=backend_timeout,
                testing_criteria=criteria,
                metadata=metadata,
            )
        )
    else:
        from fx1.sdk import Fx1Harness

        spec = _or_exit(
            lambda: Fx1Harness().eval_spec_create(
                name,
                suite=suite,
                seed=seed,
                backend=backend,
                checkpoint_dir=checkpoint_dir,
                judge_backend=judge_backend,
                timeout_s=backend_timeout,
                testing_criteria=criteria,
                metadata=metadata,
            )
        )
    typer.echo(json.dumps(spec, indent=2))


@harness_app.command("eval-spec-list")
def harness_eval_spec_list(
    limit: int = typer.Option(20, "--limit", help=_LIMIT_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """List declared eval specs (newest first) — ``GET /v1/evals``."""
    if remote is not None:
        page = _or_exit(lambda: _remote_client(remote, api_key, timeout_s).eval_specs(limit=limit))
    else:
        from fx1.sdk import Fx1Harness

        page = {"object": "list", "data": _or_exit(lambda: Fx1Harness().eval_specs(limit=limit))}
    typer.echo(json.dumps(page, indent=2))


@harness_app.command("eval-spec-get")
def harness_eval_spec_get(
    eval_id: str = typer.Argument(..., help=_EVAL_ID_CREATE_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Print one eval spec — ``GET /v1/evals/{id}``."""
    if remote is not None:
        spec = _or_exit(lambda: _remote_client(remote, api_key, timeout_s).eval_spec_get(eval_id))
    else:
        from fx1.sdk import Fx1Harness

        spec = _or_exit(lambda: Fx1Harness().eval_spec_get(eval_id))
    typer.echo(json.dumps(spec, indent=2))


@harness_app.command("eval-spec-update")
def harness_eval_spec_update(
    eval_id: str = typer.Argument(..., help=_EVAL_ID_HELP),
    name: str | None = typer.Option(None, "--name", help="New display name."),
    metadata_json: str | None = typer.Option(
        None, "--metadata", help="Replacement metadata as a JSON object."
    ),
    datasource_json: str | None = typer.Option(
        None,
        "--datasource",
        help="Replacement data_source_config as a JSON object — hot-reloads "
        "the declared shape only while no run binds the spec (409 once bound).",
    ),
    criteria_json: str | None = typer.Option(
        None,
        "--criteria",
        help="Replacement testing_criteria as a JSON list — same bound-freeze rule.",
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Edit a spec's name/metadata, or hot-reload its declared shape while
    it has no runs — datasource/criteria freeze once a run binds."""
    metadata = _json_meta(metadata_json)
    datasource = _json_meta(datasource_json)
    criteria = _json_list_opt(criteria_json, "--criteria")
    if remote is not None:
        spec = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).eval_spec_update(
                eval_id,
                name=name,
                metadata=metadata,
                data_source_config=datasource,
                testing_criteria=criteria,
            )
        )
    else:
        from fx1.sdk import Fx1Harness

        spec = _or_exit(
            lambda: Fx1Harness().eval_spec_update(
                eval_id,
                name=name,
                metadata=metadata,
                data_source_config=datasource,
                testing_criteria=criteria,
            )
        )
    typer.echo(json.dumps(spec, indent=2))


@harness_app.command("eval-spec-delete")
def harness_eval_spec_delete(
    eval_id: str = typer.Argument(..., help=_EVAL_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Delete an eval spec — journaled tombstone; bound runs stay readable."""
    if remote is not None:
        out = _or_exit(lambda: _remote_client(remote, api_key, timeout_s).eval_spec_delete(eval_id))
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    _or_exit(lambda: Fx1Harness().eval_spec_delete(eval_id))
    typer.echo(json.dumps({"id": eval_id, "object": "eval.deleted", "deleted": True}))


@harness_app.command("eval-run")
def harness_eval_run(
    eval_id: str = typer.Argument(..., help=_EVAL_ID_CREATE_HELP),
    model: str = typer.Option(
        ...,
        "--model",
        help="Eval target: a link name (hosted_k3|local_fx1|byok), fx1, or a registered ft: name.",
    ),
    data_source_json: str | None = typer.Option(
        None,
        "--data-source",
        help='Item-schema overrides as JSON, e.g. \'{"type":"custom","source":{"seed":1}}\'.',
    ),
    byok_base_url: str | None = typer.Option(None, "--byok-base-url", help=_BYOK_URL_HELP),
    byok_api_key: str | None = typer.Option(None, "--byok-api-key", help=_BYOK_KEY_HELP),
    byok_model: str | None = typer.Option(None, "--byok-model", help=_BYOK_MODEL_HELP),
    poll_s: float = typer.Option(0.5, "--poll", help="Remote: status poll interval, seconds."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(120.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Submit a run under an eval spec — ``POST /v1/evals/{id}/runs``.
    Remote polls to the terminal run object; in-process runs are
    synchronous (the terminal object prints directly)."""
    byok = _byok_opts(byok_base_url, byok_api_key, byok_model)
    data_source = _json_obj_opt(data_source_json, "--data-source")
    if remote is not None:
        client = _remote_client(remote, api_key, timeout_s)
        run = _or_exit(
            lambda: client.eval_run_create(eval_id, model=model, data_source=data_source, byok=byok)
        )
        bare = run["id"].removeprefix("evalrun_")
        _or_exit(lambda: client.wait_eval(bare, poll_s=poll_s, timeout_s=None))
        typer.echo(json.dumps(_or_exit(lambda: client.eval_run_get(eval_id, bare)), indent=2))
        return
    from fx1.sdk import Fx1Harness

    run = _or_exit(
        lambda: Fx1Harness().eval_run_create(
            eval_id,
            model=model,
            byok=byok,
            data_source_overrides=(data_source or {}).get("source"),
        )
    )
    typer.echo(json.dumps(run, indent=2))


@harness_app.command("eval-run-list")
def harness_eval_run_list(
    eval_id: str = typer.Argument(..., help=_EVAL_ID_HELP),
    limit: int = typer.Option(20, "--limit", help=_LIMIT_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """List a spec's runs (newest first) — ``GET /v1/evals/{id}/runs``."""
    if remote is not None:
        page = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).eval_runs(eval_id, limit=limit)
        )
    else:
        from fx1.sdk import Fx1Harness

        page = {
            "object": "list",
            "data": _or_exit(lambda: Fx1Harness().eval_runs(eval_id, limit=limit)),
        }
    typer.echo(json.dumps(page, indent=2))


@harness_app.command("eval-run-get")
def harness_eval_run_get(
    eval_id: str = typer.Argument(..., help=_EVAL_ID_HELP),
    run_id: str = typer.Argument(..., help=_EVALRUN_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Print one run — ``GET /v1/evals/{id}/runs/{run_id}``."""
    if remote is not None:
        run = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).eval_run_get(eval_id, run_id)
        )
    else:
        from fx1.sdk import Fx1Harness

        run = _or_exit(lambda: Fx1Harness().eval_run_get(eval_id, run_id))
    typer.echo(json.dumps(run, indent=2))


@harness_app.command("eval-run-cancel")
def harness_eval_run_cancel(
    eval_id: str = typer.Argument(..., help=_EVAL_ID_HELP),
    run_id: str = typer.Argument(..., help=_EVALRUN_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Cancel a queued run — ``POST .../runs/{run_id}/cancel`` (remote only:
    in-process runs are synchronous, already terminal)."""
    _need_remote(remote)
    run = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).eval_run_cancel(eval_id, run_id)
    )
    typer.echo(json.dumps(run, indent=2))


@harness_app.command("eval-run-delete")
def harness_eval_run_delete(
    eval_id: str = typer.Argument(..., help=_EVAL_ID_HELP),
    run_id: str = typer.Argument(..., help=_EVALRUN_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Delete a terminal run record — ``DELETE .../runs/{run_id}``."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).eval_run_delete(eval_id, run_id)
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    _or_exit(lambda: Fx1Harness().eval_run_delete(eval_id, run_id))
    typer.echo(json.dumps({"id": run_id, "object": "eval.run.deleted", "deleted": True}))


@harness_app.command("eval-run-items")
def harness_eval_run_items(
    eval_id: str = typer.Argument(..., help=_EVAL_ID_HELP),
    run_id: str = typer.Argument(..., help=_EVALRUN_ID_HELP),
    limit: int = typer.Option(20, "--limit", help=_LIMIT_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Per-task verdict rows from a completed run — ``.../output_items``."""
    if remote is not None:
        page = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).eval_run_output_items(
                eval_id, run_id, limit=limit
            )
        )
    else:
        from fx1.sdk import Fx1Harness

        page = {
            "object": "list",
            "data": _or_exit(lambda: Fx1Harness().eval_run_items(eval_id, run_id, limit=limit)),
        }
    typer.echo(json.dumps(page, indent=2))


@harness_app.command("ft-create")
def harness_ft_create(
    training_file: str = typer.Argument(
        ...,
        help="Chat-format .jsonl corpus ({messages:[...]} per line), or a "
        "file-* id already uploaded with purpose=fine-tune (remote only).",
    ),
    model: str = typer.Option("fx1", "--model", help="Trainable model: fx1|local_fx1."),
    suffix: str | None = typer.Option(
        None, "--suffix", help="Fine-tuned model name suffix (a-z0-9_-)."
    ),
    validation_file: str | None = typer.Option(
        None, "--validation-file", help="Optional held-out .jsonl corpus, or a file-* id."
    ),
    seed: int | None = typer.Option(None, "--seed", help="Pipeline seed."),
    epochs: int | None = typer.Option(None, "--epochs", help="n_epochs hyperparameter (1-50)."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(120.0, "--timeout", help=_TIMEOUT_HELP),
    no_wait: bool = typer.Option(
        False, "--no-wait", help="Remote only: submit and return immediately."
    ),
    callback_url: str | None = typer.Option(
        None, "--callback-url", help="Terminal webhook URL (POSTs the job record once)."
    ),
    callback_secret: str | None = typer.Option(
        None, "--callback-secret", help=_CALLBACK_SECRET_HELP
    ),
) -> None:
    """Create a gated fine-tuning job on the /v1/fine_tuning/jobs surface:
    in-process by default (SDK twin — synchronous, returns the terminal
    job), or ``--remote`` uploads the corpus (purpose=fine-tune) then
    submits and waits. Corpus validation is synchronous either way — a
    malformed file fails before the job exists."""

    def _resolve_input(raw: str, remote_mode: bool) -> tuple[str | None, bytes | None]:
        """A ``file-*`` string is an already-uploaded id (remote leg only —
        the in-process twin needs the corpus bytes); otherwise a path."""
        if raw.startswith("file-"):
            if not remote_mode:
                _bad_arg(f"{raw} is an uploaded id — the in-process leg needs a path")
            return raw, None
        p = Path(raw)
        content = p.read_bytes() if p.exists() else None
        if content is None:
            _bad_arg(f"{raw} does not exist")
        return None, content

    train_id, corpus = _resolve_input(training_file, remote is not None)
    val_id_opt: str | None = None
    val_bytes: bytes | None = None
    if validation_file is not None:
        val_id_opt, val_bytes = _resolve_input(validation_file, remote is not None)
    hp = {"n_epochs": epochs} if epochs is not None else None
    if remote is not None:
        client = _remote_client(remote, api_key, timeout_s)
        up_id = (
            train_id
            or _or_exit(
                lambda: client.upload_file(
                    corpus or b"", filename=Path(training_file).name, purpose="fine-tune"
                )
            )["id"]
        )
        val_id = val_id_opt
        if validation_file is not None and val_id is None:
            val_id = _or_exit(
                lambda: client.upload_file(
                    val_bytes or b"",
                    filename=Path(validation_file).name,
                    purpose="fine-tune",
                )
            )["id"]
        job = _or_exit(
            lambda: client.create_finetune_job(
                model=model,
                training_file=up_id,
                hyperparameters=hp,
                suffix=suffix,
                validation_file=val_id,
                seed=seed,
                callback_url=callback_url,
                callback_secret=callback_secret,
            )
        )
        if no_wait:
            typer.echo(json.dumps(job, indent=2))
            return
        rec = _or_exit(lambda: client.wait_finetune_job(job["id"], timeout_s=None))
        typer.echo(json.dumps(rec, indent=2))
        return
    from fx1.sdk import Fx1Harness

    harness = Fx1Harness()
    from fx1.serve.finetune import FTHyperparameters

    assert corpus is not None  # _resolve_input returns bytes or exits in-process
    ftjob = _or_exit(
        lambda: harness.create_finetune_job(
            model=model,
            training_jsonl=corpus,
            validation_jsonl=val_bytes,
            hyperparameters=FTHyperparameters(**hp) if hp is not None else None,
            suffix=suffix,
            seed=seed,
            callback_url=callback_url,
            callback_secret=callback_secret,
        )
    )
    typer.echo(ftjob.model_dump_json(indent=2))


@harness_app.command("ft-jobs")
def harness_ft_jobs(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    limit: int = typer.Option(20, "--limit", help=_LIMIT_HELP),
    after: str | None = typer.Option(None, "--after", help="Pagination cursor (job id)."),
) -> None:
    """List the remote fine-tuning jobs, newest first (in-process runs
    are synchronous — they return their terminal record at once)."""
    _need_remote(remote)
    page = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).finetune_jobs(
            limit=limit, after=after
        )
    )
    typer.echo(json.dumps(page, indent=2))


@harness_app.command("ft-status")
def harness_ft_status(
    job_id: str = typer.Argument(..., help=_FTJOB_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Print a remote fine-tuning job's live record."""
    _need_remote(remote)
    st = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).finetune_job(job_id))
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("ft-events")
def harness_ft_events(
    job_id: str = typer.Argument(..., help=_FTJOB_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    limit: int = typer.Option(20, "--limit", help=_LIMIT_HELP),
) -> None:
    """Print a remote fine-tuning job's event feed (oldest first)."""
    _need_remote(remote)
    st = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).finetune_job_events(
            job_id, limit=limit
        )
    )
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("ft-checkpoints")
def harness_ft_checkpoints(
    job_id: str = typer.Argument(..., help=_FTJOB_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    limit: int = typer.Option(10, "--limit", help=_LIMIT_HELP),
    after: str | None = typer.Option(None, "--after", help="Pagination cursor (ftckpt- id)."),
) -> None:
    """``GET /v1/fine_tuning/jobs/{id}/checkpoints`` — the model artifacts
    the job registered, oldest first; empty for a job that produced none."""
    _need_remote(remote)
    st = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).finetune_job_checkpoints(
            job_id, limit=limit, after=after
        )
    )
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("ft-wait")
def harness_ft_wait(
    job_id: str = typer.Argument(..., help=_FTJOB_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    poll_s: float = typer.Option(0.5, "--poll", help=_POLL_HELP),
    wait_timeout_s: float | None = typer.Option(
        None, "--wait-timeout", help="Give up waiting after N seconds (job keeps running)."
    ),
) -> None:
    """Re-attach to a job submitted ``--no-wait`` and poll to terminal;
    prints the terminal record (``result_files`` carries the artifacts)."""
    _need_remote(remote)
    rec = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).wait_finetune_job(
            job_id, poll_s=poll_s, timeout_s=wait_timeout_s
        )
    )
    typer.echo(json.dumps(rec, indent=2))


@harness_app.command("ft-cancel")
def harness_ft_cancel(
    job_id: str = typer.Argument(..., help=_FTJOB_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Cancel a remote fine-tuning job — queued cancels at once; running
    stops cooperatively at the next pipeline-stage boundary."""
    _need_remote(remote)
    st = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).cancel_finetune_job(job_id)
    )
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("ft-pause")
def harness_ft_pause(
    job_id: str = typer.Argument(..., help=_FTJOB_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Pause a remote fine-tuning job — queued parks before starting;
    running parks at the next stage boundary. Idempotent on paused."""
    _need_remote(remote)
    st = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).pause_finetune_job(job_id)
    )
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("ft-resume")
def harness_ft_resume(
    job_id: str = typer.Argument(..., help=_FTJOB_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Resume a paused remote fine-tuning job — restores the status
    pause captured (queued re-queues, running proceeds at its boundary)."""
    _need_remote(remote)
    st = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).resume_finetune_job(job_id)
    )
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("files")
def harness_files(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """List the remote /v1/files store, newest first."""
    _need_remote(remote)
    typer.echo(
        json.dumps(
            _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).files()), indent=2
        )
    )


@harness_app.command("file-upload")
def harness_file_upload(
    file: Path = typer.Argument(..., help="Local .jsonl to upload."),
    purpose: str = typer.Option("batch", "--purpose", help="File purpose: batch|fine-tune."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(60.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Upload a local JSONL to /v1/files — prints the file object."""
    _need_remote(remote)
    body = file.read_bytes() if file.exists() else None
    if body is None:
        typer.echo(f"error: {file} does not exist", err=True)
        raise typer.Exit(code=2)
    up = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).upload_file(
            body, filename=file.name, purpose=purpose
        )
    )
    typer.echo(json.dumps(up, indent=2))


@harness_app.command("file-content")
def harness_file_content(
    file_id: str = typer.Argument(..., help="file- id."),
    out: Path | None = typer.Option(
        None, "--out", help="Write bytes to this path instead of stdout."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(60.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Fetch a remote file's raw bytes (JSONL in, JSONL out)."""
    _need_remote(remote)
    blob = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).file_content(file_id))
    if out is not None:
        out.write_bytes(blob)
        typer.echo(json.dumps({"file_id": file_id, "bytes": len(blob), "out": str(out)}))
        return
    typer.echo(blob.decode("utf-8", "replace"), nl=False)


@harness_app.command("file-delete")
def harness_file_delete(
    file_id: str = typer.Argument(..., help="file- id."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Delete a remote file."""
    _need_remote(remote)
    st = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).delete_file(file_id))
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("upload")
def harness_upload(
    file: Path = typer.Argument(..., help="Local .jsonl to ship in chunks."),
    purpose: str = typer.Option("batch", "--purpose", help="File purpose: batch|fine-tune."),
    chunk_bytes: int = typer.Option(
        4 * 1024 * 1024, "--chunk-bytes", help="Part size for /v1/uploads parts."
    ),
    md5: bool = typer.Option(
        True, "--md5/--no-md5", help="Send the content md5 on complete (fail-closed check)."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(120.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Chunked upload via /v1/uploads — for files too big for one request.

    Opens the upload intent, posts the file in --chunk-bytes parts, and
    completes it into a file- record (printed on stdout; its id feeds
    batch-submit/ft-create like any /v1/files id).
    """
    _need_remote(remote)
    if chunk_bytes <= 0:
        typer.echo("error: --chunk-bytes must be positive", err=True)
        raise typer.Exit(code=2)
    body = file.read_bytes() if file.exists() else None
    if body is None:
        typer.echo(f"error: {file} does not exist", err=True)
        raise typer.Exit(code=2)
    client = _remote_client(remote or "", api_key, timeout_s)
    up = _or_exit(
        lambda: client.upload_create(
            purpose=purpose,
            filename=file.name,
            bytes=len(body),
            mime_type="application/jsonl",
        )
    )
    uid = str(up["id"])
    part_ids: list[str] = []
    for off in range(0, len(body), chunk_bytes):
        chunk = body[off : off + chunk_bytes]
        part = _or_exit(partial(client.upload_part, uid, chunk))
        part_ids.append(str(part["id"]))
    digest = hashlib.md5(body, usedforsecurity=False).hexdigest() if md5 else None
    done = _or_exit(lambda: client.upload_complete(uid, part_ids, md5=digest))
    typer.echo(json.dumps(done, indent=2))


@harness_app.command("upload-cancel")
def harness_upload_cancel(
    upload_id: str = typer.Argument(..., help="upload_ id."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Cancel a pending chunked upload (replays 200 when already cancelled)."""
    _need_remote(remote)
    st = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).upload_cancel(upload_id))
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("batch-submit")
def harness_batch_submit(
    input_file: Path = typer.Argument(
        ..., help="Local batch-input .jsonl ({custom_id,method,url,body} per line)."
    ),
    endpoint: str = typer.Option(
        "/v1/chat/completions",
        "--endpoint",
        help="/v1/chat/completions | /v1/responses | /v1/embeddings.",
    ),
    metadata: str | None = typer.Option(
        None, "--metadata", help="JSON object of str->str batch metadata."
    ),
    idem_key: str | None = typer.Option(
        None, "--idem-key", help="Idempotency-Key — replays the submit envelope."
    ),
    callback_url: str | None = typer.Option(
        None, "--callback-url", help="Terminal webhook URL (POSTs the batch once)."
    ),
    callback_secret: str | None = typer.Option(
        None, "--callback-secret", help=_CALLBACK_SECRET_HELP
    ),
    no_wait: bool = typer.Option(
        False, "--no-wait", help="Submit and return immediately (don't poll)."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(120.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Upload + submit a batch on /v1/batches, then poll to terminal.
    Prints the terminal batch object (or the submitted one with
    --no-wait). The submitter's X-Fx1-* env routes every line."""
    _need_remote(remote)
    body = input_file.read_bytes() if input_file.exists() else None
    if body is None:
        typer.echo(f"error: {input_file} does not exist", err=True)
        raise typer.Exit(code=2)
    meta = None
    if metadata is not None:
        try:
            meta = json.loads(metadata)
        except ValueError as exc:
            typer.echo(f"error: --metadata is not JSON: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        if not isinstance(meta, dict):
            typer.echo("error: --metadata must be a JSON object", err=True)
            raise typer.Exit(code=2)
    client = _remote_client(remote or "", api_key, timeout_s)
    up = _or_exit(lambda: client.upload_file(body, filename=input_file.name, purpose="batch"))
    sub = _or_exit(
        lambda: client.create_batch(
            up["id"],
            endpoint=endpoint,
            metadata=meta,
            idempotency_key=idem_key,
            callback_url=callback_url,
            callback_secret=callback_secret,
        )
    )
    if no_wait:
        typer.echo(json.dumps(sub, indent=2))
        return
    fin = _or_exit(lambda: client.wait_batch(sub["id"]))
    typer.echo(json.dumps(fin, indent=2))


@harness_app.command("batches")
def harness_batches(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    limit: int = typer.Option(20, "--limit", help=_LIMIT_HELP),
    after: str | None = typer.Option(None, "--after", help="Pagination cursor (batch id)."),
) -> None:
    """List remote batches, newest first."""
    _need_remote(remote)
    page = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).batches(limit=limit, after=after)
    )
    typer.echo(json.dumps(page, indent=2))


@harness_app.command("batch-status")
def harness_batch_status(
    batch_id: str = typer.Argument(..., help=_BATCH_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Print a remote batch's live record (status + request_counts)."""
    _need_remote(remote)
    st = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).batch(batch_id))
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("batch-wait")
def harness_batch_wait(
    batch_id: str = typer.Argument(..., help=_BATCH_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    poll_s: float = typer.Option(0.5, "--poll", help=_POLL_HELP),
    wait_timeout_s: float | None = typer.Option(
        None, "--wait-timeout", help="Give up waiting after N seconds (batch keeps running)."
    ),
) -> None:
    """Re-attach to a batch submitted ``--no-wait`` and poll to terminal;
    prints the terminal record (``output_file_id`` feeds batch-output)."""
    _need_remote(remote)
    rec = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).wait_batch(
            batch_id, poll_s=poll_s, timeout_s=wait_timeout_s
        )
    )
    typer.echo(json.dumps(rec, indent=2))


@harness_app.command("batch-cancel")
def harness_batch_cancel(
    batch_id: str = typer.Argument(..., help=_BATCH_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Cooperative cancel — the worker checks between lines; the batch
    lands 'cancelled' with partial output in output_file_id."""
    _need_remote(remote)
    st = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).cancel_batch(batch_id))
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("batch-output")
def harness_batch_output(
    batch_id: str = typer.Argument(..., help=_BATCH_ID_HELP),
    out: Path | None = typer.Option(
        None, "--out", help="Write output JSONL bytes to this path instead of stdout."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(60.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Fetch a finished batch's output file (output_file_id → bytes)."""
    _need_remote(remote)
    client = _remote_client(remote or "", api_key, timeout_s)
    rec = _or_exit(lambda: client.batch(batch_id))
    out_fid = rec.get("output_file_id")
    if out_fid is None:
        typer.echo(
            f"error: batch {batch_id} has no output_file_id (status {rec.get('status')})",
            err=True,
        )
        raise typer.Exit(code=2)
    blob = _or_exit(lambda: client.file_content(out_fid))
    if out is not None:
        out.write_bytes(blob)
        typer.echo(
            json.dumps(
                {
                    "batch_id": batch_id,
                    "output_file_id": out_fid,
                    "bytes": len(blob),
                    "out": str(out),
                }
            )
        )
        return
    typer.echo(blob.decode("utf-8", "replace"), nl=False)


@harness_app.command("batch-run")
def harness_batch_run(
    input_file: Path = typer.Argument(
        ..., help="Local batch-input .jsonl ({custom_id,method,url,body} per line)."
    ),
    endpoint: str = typer.Option(
        "/v1/chat/completions",
        "--endpoint",
        help="/v1/chat/completions | /v1/responses | /v1/embeddings.",
    ),
    backend: str | None = typer.Option(
        None,
        help=_BACKEND_HELP
        + " — unset lets each line's model field resolve the link (wire parity).",
    ),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    byok_base_url: str | None = typer.Option(None, "--byok-base-url", help=_BYOK_URL_HELP),
    byok_api_key: str | None = typer.Option(None, "--byok-api-key", help=_BYOK_KEY_HELP),
    byok_model: str | None = typer.Option(None, "--byok-model", help=_BYOK_MODEL_HELP),
    fallbacks: list[str] = typer.Option(
        [], "--fallback", help="Alternate backend on availability faults (repeatable, max 2)."
    ),
    callback_url: str | None = typer.Option(
        None, "--callback-url", help="Terminal webhook URL (POSTs the batch once)."
    ),
    callback_secret: str | None = typer.Option(
        None, "--callback-secret", help=_CALLBACK_SECRET_HELP
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Write output JSONL lines to this path (default: batch object only)."
    ),
) -> None:
    """The ``batch-submit`` twin, weights-direct: run the batch input
    synchronously in-process through the same per-endpoint request
    models and gated completion core the wire worker uses — no server,
    no upload, no poll. Prints the terminal batch envelope; ``--out``
    writes the OpenAI batch-result JSONL (one line per input, ``id,
    custom_id, response:{status_code, request_id, body}, error``).
    ``--backend``/``--checkpoint-dir``/``--byok-*``/``--fallback`` map
    to the wire's X-Fx1-* headers — every line resolves the same link."""
    body = input_file.read_bytes() if input_file.exists() else None
    if body is None:
        typer.echo(f"error: {input_file} does not exist", err=True)
        raise typer.Exit(code=2)
    try:
        lines = [json.loads(ln) for ln in body.decode().splitlines() if ln.strip()]
    except ValueError as exc:
        typer.echo(f"error: {input_file} is not JSONL: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if not all(isinstance(ln, dict) for ln in lines):
        typer.echo("error: every line must be a JSON object", err=True)
        raise typer.Exit(code=2)
    byok = _byok_opts(byok_base_url, byok_api_key, byok_model)
    headers = _fx1_headers(backend, checkpoint_dir, byok, fallbacks)

    from fx1.sdk import Fx1Harness

    harness = Fx1Harness()
    batch, out_lines = _or_exit(
        lambda: harness.openai_batch(
            lines,
            endpoint=endpoint,
            headers=headers,
            callback_url=callback_url,
            callback_secret=callback_secret,
        )
    )
    if out is not None:
        out.write_text("".join(json.dumps(ln) + "\n" for ln in out_lines))
    typer.echo(json.dumps(batch, indent=2))


@harness_app.command("models")
def harness_models(
    anthropic: bool = typer.Option(
        False,
        "--anthropic",
        help="Answer in Anthropic's model-list envelope (the anthropic-version projection).",
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Anthropic page size (1–1000, default 20)."
    ),
    after_id: str | None = typer.Option(
        None, "--after-id", help="Anthropic cursor: page after this model id."
    ),
    before_id: str | None = typer.Option(
        None, "--before-id", help="Anthropic cursor: page before this model id."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/models`` — the OpenAI list envelope: backend names a
    ``model`` field may carry, the ``fx1`` alias, and every registered
    ``ft:`` fine-tune. ``--anthropic`` answers the anthropic-version
    projection instead (``{data, first_id, last_id, has_more}``) with
    its ``--limit``/cursor contract. In-process by default (SDK twin)."""
    if remote is not None:
        client = _remote_client(remote, api_key, timeout_s)
        if anthropic:
            out = _or_exit(
                lambda: client.anthropic_models(limit=limit, after_id=after_id, before_id=before_id)
            )
        else:
            out = _or_exit(client.list_models)
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    if anthropic:
        out = _or_exit(
            lambda: Fx1Harness().anthropic_models(
                limit=limit, after_id=after_id, before_id=before_id
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    typer.echo(Fx1Harness().openai_models().model_dump_json(indent=2))


@harness_app.command("model")
def harness_model(
    model_id: str = typer.Argument(..., help="Model id — backend name, 'fx1', or ft:name."),
    anthropic: bool = typer.Option(
        False,
        "--anthropic",
        help="Answer in Anthropic's model-card shape (the anthropic-version projection).",
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/models/{id}`` — one card for a listed id; unknown ids
    fail closed (exit 2, the wire's 404). ``--anthropic`` answers the
    anthropic-version projection — ``{type: \"model\", id,
    display_name, created_at}`` — with the wire's ``not_found_error``.
    In-process by default."""
    if remote is not None:
        client = _remote_client(remote, api_key, timeout_s)
        if anthropic:
            out = _or_exit(lambda: client.anthropic_model(model_id))
            typer.echo(json.dumps(out, indent=2))
            return
        out = _or_exit(lambda: client.retrieve_model(model_id))
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    if anthropic:
        out = _or_exit(lambda: Fx1Harness().anthropic_model(model_id))
        typer.echo(json.dumps(out, indent=2))
        return
    card = _or_exit(lambda: Fx1Harness().openai_model(model_id))
    typer.echo(card.model_dump_json(indent=2))


@harness_app.command("model-delete")
def harness_model_delete(
    model_id: str = typer.Argument(..., help="Registered ft: name to unregister."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``DELETE /v1/models/{id}`` — unregister an ``ft:`` fine-tune. The
    tombstone is real (the name stops resolving everywhere); built-in
    link ids refuse (exit 2, the wire's 400) and unregistered names fail
    closed (exit 2, the wire's 404). In-process by default."""
    if remote is not None:
        out = _or_exit(lambda: _remote_client(remote, api_key, timeout_s).delete_model(model_id))
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    deleted = _or_exit(lambda: Fx1Harness().openai_delete_model(model_id))
    typer.echo(deleted.model_dump_json(indent=2))


@harness_app.command("respond")
def harness_respond(  # NOSONAR
    input_: str = typer.Argument(..., help="Input string, or a JSON array of Responses items."),
    model: str = typer.Option("fx1", "--model", help=_MODEL_ID_OPT_HELP),
    instructions: str | None = typer.Option(
        None, "--instructions", help="Prepended system-level instructions."
    ),
    backend: str | None = typer.Option(None, "--backend", help=_BACKEND_OVERRIDE_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    byok_base_url: str | None = typer.Option(None, "--byok-base-url", help=_BYOK_URL_HELP),
    byok_api_key: str | None = typer.Option(None, "--byok-api-key", help=_BYOK_KEY_HELP),
    byok_model: str | None = typer.Option(None, "--byok-model", help=_BYOK_MODEL_HELP),
    fallbacks: list[str] = typer.Option([], "--fallback", help=_FALLBACK_HELP),
    temperature: float | None = typer.Option(None, "--temperature", help="Decode temperature."),
    top_p: float | None = typer.Option(None, "--top-p", help=_TOP_P_OPT_HELP),
    max_output_tokens: int | None = typer.Option(
        None, "--max-output-tokens", help="Output token cap."
    ),
    metadata: str | None = typer.Option(None, "--metadata", help=_METADATA_PAIRS_HELP),
    text_format: str | None = typer.Option(
        None, "--format", help='text.format JSON, e.g. \'{"type":"json_object"}\'.'
    ),
    tools: str | None = typer.Option(None, "--tools", help="JSON array of Responses tools."),
    tool_choice: str | None = typer.Option(
        None, "--tool-choice", help='"none"/"auto"/"required" or a JSON choice object.'
    ),
    max_tool_calls: int | None = typer.Option(
        None,
        "--max-tool-calls",
        help="Cap the function calls one response may carry — over the cap the "
        "turn truncates to status='incomplete'.",
    ),
    verbosity: str | None = typer.Option(
        None, "--verbosity", help="Output verbosity hint: low|medium|high."
    ),
    prompt_cache_key: str | None = typer.Option(
        None, "--prompt-cache-key", help="Provider prompt-cache key hint."
    ),
    prompt_cache_retention: str | None = typer.Option(
        None, "--prompt-cache-retention", help="in-memory|24h."
    ),
    stream: bool = typer.Option(
        False, "--stream", help="Emit Responses event deltas instead of one JSON block."
    ),
    previous_response_id: str | None = typer.Option(
        None,
        "--previous-response-id",
        help="Chain onto a stored response (resp_…) — the turn runs with the parent history.",
    ),
    conversation: str | None = typer.Option(
        None,
        "--conversation",
        help="Join a conversation container (conv_…) — its items are the turn's context.",
    ),
    background: bool = typer.Option(
        False,
        "--background",
        help="Queue the response and return a status='queued' object — poll "
        "with `response-get`, cancel with `response-cancel`.",
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(120.0, "--timeout", help=_TIMEOUT_HELP),
    backend_timeout: float | None = typer.Option(
        None, "--backend-timeout", help="Per-call backend deadline in seconds."
    ),
) -> None:
    """``POST /v1/responses`` — the Responses API over the gated pipeline.

    The response object prints verbatim (``output[0]`` is the message
    item); a completion-log id rides ``X-Fx1-Completion-Id`` for receipt
    lookup. In-process by default — the SDK twin runs the same request
    model, link resolution, and gate.
    """
    byok = _byok_opts(byok_base_url, byok_api_key, byok_model)

    def _json_opt(raw: str | None, what: str) -> Any:
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            _bad_arg(f"invalid {what} JSON")
            raise AssertionError("unreachable") from None

    try:
        parsed = json.loads(input_)
        items: str | list[Any] = parsed if isinstance(parsed, list) else input_
    except json.JSONDecodeError:
        items = input_
    meta = _json_opt(metadata, "metadata")
    if meta is not None and not (
        isinstance(meta, dict)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items())
    ):
        _bad_arg(_METADATA_PAIRS_ERR)
    tfmt = _json_opt(text_format, "format")
    tool_list = _json_opt(tools, "tools")
    if tool_list is not None and not isinstance(tool_list, list):
        _bad_arg("--tools must be a JSON array")
    tchoice: str | dict[str, Any] | None = None
    if tool_choice is not None:
        if tool_choice in {"none", "auto", "required"}:
            tchoice = tool_choice
        else:
            tc = _json_opt(tool_choice, "tool-choice")
            if not isinstance(tc, dict):
                _bad_arg("--tool-choice must be none/auto/required or a JSON object")
            tchoice = tc

    if remote is not None:
        client = _remote_client(remote, api_key, timeout_s)
        if stream:
            events, _cid = _or_exit(
                lambda: client.responses_create_stream(
                    items,
                    model=model,
                    instructions=instructions,
                    backend=backend,
                    byok=byok,
                    checkpoint_dir=checkpoint_dir,
                    fallbacks=fallbacks or None,
                    timeout_s=backend_timeout,
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_output_tokens,
                    metadata=meta,
                    text_format=tfmt,
                    tools=tool_list,
                    tool_choice=tchoice,
                    previous_response_id=previous_response_id,
                    conversation=conversation,
                    max_tool_calls=max_tool_calls,
                    verbosity=verbosity,
                    prompt_cache_key=prompt_cache_key,
                    prompt_cache_retention=prompt_cache_retention,
                )
            )
            _emit_response_deltas(events)
            return
        resp, _cid = _or_exit(
            lambda: client.responses_create(
                items,
                model=model,
                instructions=instructions,
                backend=backend,
                byok=byok,
                checkpoint_dir=checkpoint_dir,
                fallbacks=fallbacks or None,
                timeout_s=backend_timeout,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
                metadata=meta,
                text_format=tfmt,
                tools=tool_list,
                tool_choice=tchoice,
                previous_response_id=previous_response_id,
                conversation=conversation,
                max_tool_calls=max_tool_calls,
                verbosity=verbosity,
                prompt_cache_key=prompt_cache_key,
                prompt_cache_retention=prompt_cache_retention,
                background=background,
            )
        )
        typer.echo(json.dumps(resp, indent=2))
        return
    from fx1.sdk import Fx1Harness

    headers = _fx1_headers(backend, checkpoint_dir, byok, fallbacks)
    fx1: dict[str, Any] = {}
    if backend_timeout is not None:
        fx1["timeout_s"] = backend_timeout
    body: dict[str, Any] = {
        "model": model,
        "input": items,
        "instructions": instructions,
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_output_tokens,
        "metadata": meta,
        "tools": tool_list,
        "tool_choice": tchoice,
        "previous_response_id": previous_response_id,
        "conversation": conversation,
        "max_tool_calls": max_tool_calls,
        "prompt_cache_key": prompt_cache_key,
        "prompt_cache_retention": prompt_cache_retention,
        "background": background,
        "fx1": fx1 or None,
    }
    if tfmt is not None or verbosity is not None:
        text: dict[str, Any] = {}
        if tfmt is not None:
            text["format"] = tfmt
        if verbosity is not None:
            text["verbosity"] = verbosity
        body["text"] = text
    if stream:
        sevents, _cid = _or_exit(lambda: Fx1Harness().openai_response_stream(body, headers=headers))
        _emit_response_deltas(sevents)
        return
    resp, _cid = _or_exit(lambda: Fx1Harness().openai_response(body, headers=headers))
    typer.echo(json.dumps(resp, indent=2))


@harness_app.command("message")
def harness_message(  # NOSONAR
    input_: str = typer.Argument(
        ...,
        help="User-turn text, or a JSON array of Anthropic message objects.",
    ),
    model: str = typer.Option("fx1", "--model", help=_MODEL_ID_OPT_HELP),
    max_tokens: int = typer.Option(
        1024, "--max-tokens", help="Output token cap (required by the contract)."
    ),
    system: str | None = typer.Option(
        None,
        "--system",
        help="System prompt text, or a JSON array of {type:'text',text} blocks.",
    ),
    temperature: float | None = typer.Option(None, "--temperature", help="Decode temperature 0–1."),
    top_p: float | None = typer.Option(None, "--top-p", help=_TOP_P_OPT_HELP),
    stop: list[str] = typer.Option([], "--stop", help="Stop sequence (repeatable, at most 4)."),
    tools: str | None = typer.Option(
        None, "--tools", help="JSON array of Anthropic tools ({name, description, input_schema})."
    ),
    tool_choice: str | None = typer.Option(
        None,
        "--tool-choice",
        help='JSON choice object, e.g. \'{"type":"auto"}\' or \'{"type":"tool","name":"x"}\'.',
    ),
    user_id: str | None = typer.Option(None, "--user-id", help="metadata.user_id audit stamp."),
    stream: bool = typer.Option(
        False, "--stream", help="Emit Anthropic SSE event deltas instead of one JSON block."
    ),
    backend: str | None = typer.Option(None, "--backend", help=_BACKEND_OVERRIDE_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    byok_base_url: str | None = typer.Option(None, "--byok-base-url", help=_BYOK_URL_HELP),
    byok_api_key: str | None = typer.Option(None, "--byok-api-key", help=_BYOK_KEY_HELP),
    byok_model: str | None = typer.Option(None, "--byok-model", help=_BYOK_MODEL_HELP),
    fallbacks: list[str] = typer.Option([], "--fallback", help=_FALLBACK_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(120.0, "--timeout", help=_TIMEOUT_HELP),
    backend_timeout: float | None = typer.Option(
        None, "--backend-timeout", help="Per-call backend deadline in seconds."
    ),
) -> None:
    """``POST /v1/messages`` — the Anthropic Messages surface over the gated
    pipeline.

    Anthropic turns (``user``/``assistant``, string or block-list content),
    ``--system``, tools in the Anthropic ``input_schema`` shape. The answer
    prints as the ``message`` object; ``--stream`` prints text deltas. The
    completion-log id rides ``X-Fx1-Completion-Id`` for receipt lookup.
    Anthropic-only knobs the pipeline cannot honor (top_k, thinking,
    cache_control, image blocks) fail closed — never silently dropped.
    """
    byok = _byok_opts(byok_base_url, byok_api_key, byok_model)

    def _json_opt(raw: str | None, what: str) -> Any:
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            _bad_arg(f"invalid {what} JSON")
            raise AssertionError("unreachable") from None

    try:
        parsed = json.loads(input_)
        messages: list[dict[str, Any]] = (
            cast(list[dict[str, Any]], parsed)
            if isinstance(parsed, list)
            else [{"role": "user", "content": input_}]
        )
    except json.JSONDecodeError:
        messages = [{"role": "user", "content": input_}]
    sys_val: str | list[Any] | None = None
    if system is not None:
        try:
            parsed_sys = json.loads(system)
        except json.JSONDecodeError:
            sys_val = system
        else:
            sys_val = parsed_sys if isinstance(parsed_sys, list) else system
    tool_list = _json_opt(tools, "tools")
    if tool_list is not None and not isinstance(tool_list, list):
        _bad_arg("--tools must be a JSON array")
    tchoice = _json_opt(tool_choice, "tool-choice")
    if tchoice is not None and not isinstance(tchoice, dict):
        _bad_arg('--tool-choice must be a JSON object, e.g. {"type":"auto"}')

    if remote is not None:
        client = _remote_client(remote, api_key, timeout_s)
        if stream:
            events, _cid = _or_exit(
                lambda: client.create_message_stream(
                    messages,
                    model=model,
                    max_tokens=max_tokens,
                    system=sys_val,
                    temperature=temperature,
                    top_p=top_p,
                    stop_sequences=stop or None,
                    tools=tool_list,
                    tool_choice=tchoice,
                    user_id=user_id,
                    backend=backend,
                    byok=byok,
                    checkpoint_dir=checkpoint_dir,
                    fallbacks=fallbacks or None,
                    timeout_s=backend_timeout,
                )
            )
            _emit_anthropic_deltas(events)
            return
        msg_remote, _cid = _or_exit(
            lambda: client.create_message(
                messages,
                model=model,
                max_tokens=max_tokens,
                system=sys_val,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop or None,
                tools=tool_list,
                tool_choice=tchoice,
                user_id=user_id,
                backend=backend,
                byok=byok,
                checkpoint_dir=checkpoint_dir,
                fallbacks=fallbacks or None,
                timeout_s=backend_timeout,
            )
        )
        typer.echo(json.dumps(msg_remote, indent=2))
        return
    from fx1.sdk import Fx1Harness

    headers = _fx1_headers(backend, checkpoint_dir, byok, fallbacks)
    fx1: dict[str, Any] = {}
    if backend_timeout is not None:
        fx1["timeout_s"] = backend_timeout
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "system": sys_val,
        "temperature": temperature,
        "top_p": top_p,
        "stop_sequences": stop or None,
        "tools": tool_list,
        "tool_choice": tchoice,
        "metadata": {"user_id": user_id} if user_id is not None else None,
        "fx1": fx1 or None,
    }
    if stream:
        sevents, _cid = _or_exit(
            lambda: Fx1Harness().anthropic_message_stream(body, headers=headers)
        )
        _emit_anthropic_deltas(sevents)
        return
    amsg, _cid = _or_exit(lambda: Fx1Harness().anthropic_message(body, headers=headers))
    typer.echo(json.dumps(amsg.model_dump(mode="json"), indent=2))


@harness_app.command("message-batch")
def harness_message_batch(
    input_file: Path = typer.Argument(
        ..., help="Batch input .jsonl — {custom_id, params} per line (params = /v1/messages body)."
    ),
    callback_url: str | None = typer.Option(
        None, "--callback-url", help="Terminal webhook URL (POSTs the batch once)."
    ),
    callback_secret: str | None = typer.Option(
        None, "--callback-secret", help=_CALLBACK_SECRET_HELP
    ),
    idem_key: str | None = typer.Option(
        None, "--idem-key", help="Idempotency-Key — replays the submit envelope."
    ),
    no_wait: bool = typer.Option(
        False, "--no-wait", help="Submit and return immediately (remote leg only)."
    ),
    backend: str | None = typer.Option(None, "--backend", help=_BACKEND_OVERRIDE_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    byok_base_url: str | None = typer.Option(None, "--byok-base-url", help=_BYOK_URL_HELP),
    byok_api_key: str | None = typer.Option(None, "--byok-api-key", help=_BYOK_KEY_HELP),
    byok_model: str | None = typer.Option(None, "--byok-model", help=_BYOK_MODEL_HELP),
    fallbacks: list[str] = typer.Option([], "--fallback", help=_FALLBACK_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(120.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/messages/batches`` — submit an Anthropic message batch.

    Each line of ``input_file`` is ``{custom_id, params}`` — ``params`` a
    full ``/v1/messages`` body (``stream`` inside a batch refuses at
    validation). With ``--remote`` the submit runs async server-side and
    polls to ``ended``; in-process it runs synchronously and prints the
    ended batch plus its result rows. The submitter's backend choice
    (``--backend``/BYOK/``--fallback``) applies to every item."""
    body = input_file.read_text(encoding="utf-8") if input_file.exists() else None
    if body is None:
        typer.echo(f"error: {input_file} does not exist", err=True)
        raise typer.Exit(code=2)
    try:
        requests = [json.loads(line) for line in body.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        typer.echo(f"error: {input_file} is not valid JSONL: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if not requests:
        typer.echo("error: batch input is empty", err=True)
        raise typer.Exit(code=2)
    if remote is not None:
        client = _remote_client(remote, api_key, timeout_s)
        headers = _fx1_headers(
            backend, checkpoint_dir, _byok_opts(byok_base_url, byok_api_key, byok_model), fallbacks
        )
        sub = _or_exit(
            lambda: client.create_message_batch(
                requests,
                callback_url=callback_url,
                callback_secret=callback_secret,
                idempotency_key=idem_key,
                extra_headers=headers or None,
            )
        )
        if no_wait:
            typer.echo(json.dumps(sub, indent=2))
            return
        fin = _or_exit(lambda: client.wait_message_batch(sub["id"]))
        typer.echo(json.dumps(fin, indent=2))
        return
    from fx1.sdk import Fx1Harness

    headers = _fx1_headers(
        backend, checkpoint_dir, _byok_opts(byok_base_url, byok_api_key, byok_model), fallbacks
    )
    batch, rows = _or_exit(
        lambda: Fx1Harness().anthropic_batch(
            requests,
            headers=headers,
            callback_url=callback_url,
            callback_secret=callback_secret,
        )
    )
    typer.echo(json.dumps({"batch": batch, "results": rows}, indent=2))


@harness_app.command("message-batches")
def harness_message_batches(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    limit: int = typer.Option(20, "--limit", help=_LIMIT_HELP),
    after_id: str | None = typer.Option(
        None, "--after-id", help="Cursor — entries newer than this batch id."
    ),
    before_id: str | None = typer.Option(
        None, "--before-id", help="Cursor — entries older than this batch id."
    ),
) -> None:
    """List remote Anthropic message batches, newest first."""
    _need_remote(remote)
    page = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).message_batches(
            limit=limit, after_id=after_id, before_id=before_id
        )
    )
    typer.echo(json.dumps(page, indent=2))


@harness_app.command("message-batch-status")
def harness_message_batch_status(
    batch_id: str = typer.Argument(..., help=_BATCH_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Print a remote message batch's record (``processing_status`` +
    ``request_counts`` — counts stay all-processing until the batch ends)."""
    _need_remote(remote)
    st = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).message_batch(batch_id))
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("message-batch-wait")
def harness_message_batch_wait(
    batch_id: str = typer.Argument(..., help=_BATCH_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    poll_s: float = typer.Option(0.5, "--poll", help=_POLL_HELP),
    wait_timeout_s: float | None = typer.Option(
        None, "--wait-timeout", help="Give up waiting after N seconds (batch keeps running)."
    ),
) -> None:
    """Re-attach to a ``--no-wait`` message batch and poll to ``ended``."""
    _need_remote(remote)
    rec = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).wait_message_batch(
            batch_id, poll_s=poll_s, timeout_s=wait_timeout_s
        )
    )
    typer.echo(json.dumps(rec, indent=2))


@harness_app.command("message-batch-cancel")
def harness_message_batch_cancel(
    batch_id: str = typer.Argument(..., help=_BATCH_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Cooperative cancel — flips to ``canceling``; in-flight items
    complete and the rest land ``canceled`` result rows."""
    _need_remote(remote)
    st = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).cancel_message_batch(batch_id)
    )
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("message-batch-results")
def harness_message_batch_results(
    batch_id: str = typer.Argument(..., help=_BATCH_ID_HELP),
    out: Path | None = typer.Option(
        None, "--out", help="Write the results JSONL to this path instead of stdout."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(60.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Fetch an ended message batch's results — one ``{custom_id,
    result}`` row per request (``succeeded``/``errored``/``canceled``/
    ``expired``). Before the batch ends the wire 400s."""
    _need_remote(remote)
    rows = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).message_batch_results(batch_id)
    )
    text = "".join(json.dumps(row) + "\n" for row in rows)
    if out is not None:
        out.write_text(text, encoding="utf-8")
        typer.echo(json.dumps({"batch_id": batch_id, "rows": len(rows), "out": str(out)}))
        return
    typer.echo(text, nl=False)


@harness_app.command("message-batch-delete")
def harness_message_batch_delete(
    batch_id: str = typer.Argument(..., help=_BATCH_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Tombstone an ended message batch — deletes its stored rows.
    A batch that isn't ended refuses (Anthropic's contract)."""
    _need_remote(remote)
    out = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).delete_message_batch(batch_id)
    )
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("message-tokens")
def harness_message_tokens(
    request_file: Path = typer.Argument(
        ..., help="JSON file: a /v1/messages-count_tokens body — {model?, messages, system?}."
    ),
    backend: str | None = typer.Option(None, "--backend", help=_BACKEND_OVERRIDE_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    byok_base_url: str | None = typer.Option(None, "--byok-base-url", help=_BYOK_URL_HELP),
    byok_api_key: str | None = typer.Option(None, "--byok-api-key", help=_BYOK_KEY_HELP),
    byok_model: str | None = typer.Option(None, "--byok-model", help=_BYOK_MODEL_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(60.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/messages/count_tokens`` — the provider's own input
    count, ``{"input_tokens": N}``.

    The request body is the ``/v1/messages`` shape minus ``max_tokens``
    (and minus ``stream``); ``tools``/``tool_choice`` refuse — the
    tokenize channel sees only messages, so counting a toolful request
    would undercount. A backend without a tokenize route fails closed
    (exit 2, the wire's 501) — the harness never estimates."""
    body_text = request_file.read_text(encoding="utf-8") if request_file.exists() else None
    if body_text is None:
        typer.echo(f"error: {request_file} does not exist", err=True)
        raise typer.Exit(code=2)
    try:
        body = json.loads(body_text)
    except json.JSONDecodeError as exc:
        typer.echo(f"error: {request_file} is not valid JSON: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if not isinstance(body, dict):
        typer.echo(f"error: {request_file} must hold one JSON object", err=True)
        raise typer.Exit(code=2)
    headers = _fx1_headers(
        backend, checkpoint_dir, _byok_opts(byok_base_url, byok_api_key, byok_model), []
    )
    if remote is not None:
        n = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).count_message_tokens(
                body, extra_headers=headers or None
            )
        )
        typer.echo(json.dumps({"input_tokens": n}, indent=2))
        return
    from fx1.sdk import Fx1Harness

    n = _or_exit(lambda: Fx1Harness().anthropic_count_tokens(body, headers=headers or None))
    typer.echo(json.dumps({"input_tokens": n}, indent=2))


def _emit_completion_deltas(chunks: list[dict[str, Any]]) -> None:
    """Print the legacy text deltas — the terminal frame carries no text,
    so ordering is by frame emission, not ``choices[i].index``."""
    for chunk in chunks:
        for choice in cast(list[dict[str, Any]], chunk.get("choices") or []):
            text = choice.get("text")
            if isinstance(text, str) and text:
                typer.echo(text, nl=False)
    typer.echo()


@harness_app.command("text-completion")
def harness_text_completion(  # NOSONAR
    prompt: str = typer.Argument(  # NOSONAR(S107) — sonar anchors S107 at the first param line
        ...,
        help="Prompt text, or a JSON array of prompt strings.",
    ),
    model: str = typer.Option("fx1", "--model", help=_MODEL_ID_OPT_HELP),
    max_tokens: int | None = typer.Option(
        None, "--max-tokens", help="Output token cap (server default 16)."
    ),
    temperature: float | None = typer.Option(None, "--temperature", help="Decode temperature 0–1."),
    top_p: float | None = typer.Option(None, "--top-p", help=_TOP_P_OPT_HELP),
    n: int = typer.Option(1, "--n", help="Choice count per prompt element (≤8)."),
    stop: list[str] = typer.Option([], "--stop", help="Stop sequence (repeatable, at most 4)."),
    seed: int | None = typer.Option(None, "--seed", help="Deterministic decode seed."),
    echo: bool = typer.Option(
        False, "--echo", help="Prepend the prompt text to each choice (legacy FIM-style echo)."
    ),
    stream: bool = typer.Option(
        False, "--stream", help="Emit legacy text-completion SSE deltas instead of one JSON block."
    ),
    backend: str | None = typer.Option(None, "--backend", help=_BACKEND_OVERRIDE_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    byok_base_url: str | None = typer.Option(None, "--byok-base-url", help=_BYOK_URL_HELP),
    byok_api_key: str | None = typer.Option(None, "--byok-api-key", help=_BYOK_KEY_HELP),
    byok_model: str | None = typer.Option(None, "--byok-model", help=_BYOK_MODEL_HELP),
    fallbacks: list[str] = typer.Option([], "--fallback", help=_FALLBACK_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(120.0, "--timeout", help=_TIMEOUT_HELP),
    backend_timeout: float | None = typer.Option(
        None, "--backend-timeout", help="Per-call backend deadline in seconds."
    ),
) -> None:
    """``POST /v1/completions`` — the legacy ``text_completion`` surface
    (what ``client.completions.create`` and pre-chat agents target).

    ``prompt`` takes a string or a JSON array of prompt strings; each
    element runs the gated pipeline independently and ``--n`` repeats
    within an element — a two-prompt ``--n 2`` call lands four flat
    choices. ``--stream`` prints the legacy chunk grammar's text deltas.
    ``suffix``/``best_of``/``logprobs`` are unsupported — the server
    fails them closed 422 rather than silently dropping them.
    """
    byok = _byok_opts(byok_base_url, byok_api_key, byok_model)
    try:
        parsed = json.loads(prompt)
        prompt_val: str | list[str] = (
            cast(list[str], parsed)
            if isinstance(parsed, list) and all(isinstance(p, str) for p in parsed)
            else prompt
        )
    except json.JSONDecodeError:
        prompt_val = prompt

    if remote is not None:
        client = _remote_client(remote, api_key, timeout_s)
        if stream:
            chunks, _cid = _or_exit(
                lambda: client.create_completion_stream(
                    prompt_val,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    stop=stop or None,
                    seed=seed,
                    echo=echo,
                    backend=backend,
                    byok=byok,
                    checkpoint_dir=checkpoint_dir,
                    fallbacks=fallbacks or None,
                    timeout_s=backend_timeout,
                )
            )
            _emit_completion_deltas(chunks)
            return
        env, _cid = _or_exit(
            lambda: client.create_completion(
                prompt_val,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stop=stop or None,
                seed=seed,
                echo=echo,
                backend=backend,
                byok=byok,
                checkpoint_dir=checkpoint_dir,
                fallbacks=fallbacks or None,
                timeout_s=backend_timeout,
            )
        )
        typer.echo(json.dumps(env, indent=2))
        return
    from fx1.sdk import Fx1Harness

    headers = _fx1_headers(backend, checkpoint_dir, byok, fallbacks)
    fx1: dict[str, Any] = {}
    if backend_timeout is not None:
        fx1["timeout_s"] = backend_timeout
    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt_val,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "stop": stop or None,
        "seed": seed,
        "echo": echo,
        "fx1": fx1 or None,
    }
    if stream:
        schunks, _cid = _or_exit(
            lambda: Fx1Harness().openai_completion_stream({**body, "stream": True}, headers=headers)
        )
        _emit_completion_deltas(schunks)
        return
    cenv, _cid = _or_exit(lambda: Fx1Harness().openai_completion(body, headers=headers))
    typer.echo(json.dumps(cenv, indent=2))


@harness_app.command("embed")
def harness_embed(
    inputs: list[str] = typer.Argument(..., help="Text to embed (repeatable)."),
    model: str = typer.Option("fx1", "--model", help="Embedding model id."),
    backend: str | None = typer.Option(None, "--backend", help=_BACKEND_OVERRIDE_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    byok_base_url: str | None = typer.Option(None, "--byok-base-url", help=_BYOK_URL_HELP),
    byok_api_key: str | None = typer.Option(None, "--byok-api-key", help=_BYOK_KEY_HELP),
    byok_model: str | None = typer.Option(None, "--byok-model", help=_BYOK_MODEL_HELP),
    fallbacks: list[str] = typer.Option([], "--fallback", help=_FALLBACK_HELP),
    encoding_format: str | None = typer.Option(None, "--encoding", help='"float" | "base64".'),
    dimensions: int | None = typer.Option(None, "--dimensions", help="Output dimensions."),
    user: str | None = typer.Option(None, "--user", help="End-user tag."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(120.0, "--timeout", help=_TIMEOUT_HELP),
    backend_timeout: float | None = typer.Option(
        None, "--backend-timeout", help="Per-call backend deadline in seconds."
    ),
) -> None:
    """``POST /v1/embeddings`` — vectors through the gated pipeline; a link
    without the channel answers a wire error, never fabricated floats.
    In-process by default."""
    byok = _byok_opts(byok_base_url, byok_api_key, byok_model)
    inp: str | list[str] = inputs[0] if len(inputs) == 1 else list(inputs)
    if remote is not None:
        env, _cid = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).embeddings_create(
                inp,
                model=model,
                backend=backend,
                byok=byok,
                checkpoint_dir=checkpoint_dir,
                fallbacks=fallbacks or None,
                timeout_s=backend_timeout,
                encoding_format=encoding_format,
                dimensions=dimensions,
                user=user,
            )
        )
        typer.echo(json.dumps(env, indent=2))
        return
    from fx1.sdk import Fx1Harness

    headers = _fx1_headers(backend, checkpoint_dir, byok, fallbacks)
    fx1: dict[str, Any] = {}
    if backend_timeout is not None:
        fx1["timeout_s"] = backend_timeout
    body: dict[str, Any] = {
        "model": model,
        "input": inp,
        "encoding_format": encoding_format,
        "dimensions": dimensions,
        "user": user,
        "fx1": fx1 or None,
    }
    env, _cid = _or_exit(lambda: Fx1Harness().openai_embeddings(body, headers=headers))
    typer.echo(json.dumps(env, indent=2))


@harness_app.command("moderate")
def harness_moderate(
    inputs: list[str] = typer.Argument(..., help="Text to classify (repeatable)."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/moderations`` — the honesty gate as an OpenAI-moderations
    verdict; flagged inputs carry per-category scores. Advisory — no
    backend needed. In-process by default."""
    inp: str | list[str] = inputs[0] if len(inputs) == 1 else list(inputs)
    if remote is not None:
        out = _or_exit(lambda: _remote_client(remote, api_key, timeout_s).moderate(inp))
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().moderate(inp))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("chat-get")
def harness_chat_get(
    completion_id: str = typer.Argument(..., help=_CHAT_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/chat/completions/{id}`` — the stored envelope; missing ids
    (evicted, deleted, ``store: false``) exit 2, never a fabricated object."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).retrieve_chat_completion(
                completion_id
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_chat_get(completion_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("chat-update")
def harness_chat_update(
    completion_id: str = typer.Argument(..., help=_CHAT_ID_HELP),
    metadata_json: str | None = typer.Option(
        None, "--metadata", help="Replacement metadata as a JSON object of string pairs."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/chat/completions/{id}`` — replace the stored completion's
    metadata wholesale (the only mutable field); unknown ids exit 2."""
    metadata = _json_meta(metadata_json) if metadata_json is not None else {}
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).update_chat_completion(
                completion_id, metadata=metadata
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_chat_update(completion_id, metadata=metadata))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("chat-delete")
def harness_chat_delete(
    completion_id: str = typer.Argument(..., help="Stored chat.completion id."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``DELETE /v1/chat/completions/{id}`` — drop the stored envelope."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).delete_chat_completion(completion_id)
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_chat_delete(completion_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("response-get")
def harness_response_get(
    response_id: str = typer.Argument(..., help=_RESPONSE_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/responses/{id}`` — the stored response object; missing ids
    exit 2, never a fabricated object."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).retrieve_response(response_id)
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_response_get(response_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("response-delete")
def harness_response_delete(
    response_id: str = typer.Argument(..., help=_RESPONSE_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``DELETE /v1/responses/{id}`` — drop the stored envelope."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).delete_response(response_id)
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_response_delete(response_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("response-cancel")
def harness_response_cancel(
    response_id: str = typer.Argument(..., help="Background response id (resp_*)."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/responses/{id}/cancel`` — cancel a queued or in-progress
    background response. Terminal responses exit 2 (``cancel_terminal``),
    unknown ids exit 2 (``not_found``)."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).cancel_response(response_id)
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_response_cancel(response_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("response-replay")
def harness_response_replay(
    response_id: str = typer.Argument(..., help=_RESPONSE_ID_HELP),
    starting_after: int | None = typer.Option(
        None,
        "--starting-after",
        help="Resume past sequence N — only events whose sequence number "
        "exceeds N (the frames' id: cursor).",
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/responses/{id}?stream=true`` — replay the stored response
    as the Responses SSE event stream: the full grammar for a terminal
    response, or the prelude plus a live follow for a still-running
    ``background:true`` call. Prints the text-delta channel like
    ``respond --stream``; unknown/deleted ids exit 2 (``not_found``)."""
    if remote is not None:
        revents, _cid = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).responses_replay(
                response_id, starting_after=starting_after, timeout_s=timeout_s
            )
        )
        _emit_response_deltas(revents)
        return
    from fx1.sdk import Fx1Harness

    sevents, _cid = _or_exit(
        lambda: Fx1Harness().openai_response_replay(
            response_id, starting_after=starting_after, timeout_s=timeout_s
        )
    )
    _emit_response_deltas(sevents)


@harness_app.command("chat-list")
def harness_chat_list(
    model: str | None = typer.Option(None, "--model", help="Filter to this model id."),
    metadata: list[str] = typer.Option(
        [], "--metadata", help="Exact-match filter, repeatable: --metadata key=value"
    ),
    limit: int = typer.Option(20, "--limit", min=1, max=100),
    after: str | None = typer.Option(None, "--after", help="Page cursor — a completion id."),
    before: str | None = typer.Option(None, "--before", help="Page cursor — a completion id."),
    order: str = typer.Option("asc", "--order", help=_ORDER_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/chat/completions`` — stored completions, filtered by
    model and/or a ``--metadata key=value`` subset, paged by id."""
    meta: dict[str, str] = {}
    for pair in metadata:
        if "=" not in pair:
            typer.echo(f"error: --metadata expects key=value, got {pair!r}", err=True)
            raise typer.Exit(code=2)
        k, v = pair.split("=", 1)
        meta[k] = v
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).list_chat_completions(
                model=model,
                metadata=meta or None,
                limit=limit,
                after=after,
                before=before,
                order=order,
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(
        lambda: Fx1Harness().openai_chat_list(
            model=model,
            metadata=meta or None,
            limit=limit,
            after=after,
            before=before,
            order=order,
        )
    )
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("chat-messages")
def harness_chat_messages(
    completion_id: str = typer.Argument(..., help=_CHAT_ID_HELP),
    limit: int = typer.Option(20, "--limit", min=1, max=100),
    after: str | None = typer.Option(None, "--after", help=_ITEM_CURSOR_HELP),
    before: str | None = typer.Option(None, "--before", help=_ITEM_CURSOR_HELP),
    order: str = typer.Option("asc", "--order", help=_ORDER_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/chat/completions/{id}/messages`` — the request messages a
    stored completion ran on; missing ids exit 2."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).chat_completion_messages(
                completion_id, limit=limit, after=after, before=before, order=order
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(
        lambda: Fx1Harness().openai_chat_messages(
            completion_id, limit=limit, after=after, before=before, order=order
        )
    )
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("response-input-items")
def harness_response_input_items(
    response_id: str = typer.Argument(..., help=_RESPONSE_ID_HELP),
    limit: int = typer.Option(20, "--limit", min=1, max=100),
    after: str | None = typer.Option(None, "--after", help=_ITEM_CURSOR_HELP),
    before: str | None = typer.Option(None, "--before", help=_ITEM_CURSOR_HELP),
    order: str = typer.Option("asc", "--order", help=_ORDER_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/responses/{id}/input_items`` — the ``input`` items a
    stored response ran on; missing ids exit 2."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).response_input_items(
                response_id, limit=limit, after=after, before=before, order=order
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(
        lambda: Fx1Harness().openai_response_input_items(
            response_id, limit=limit, after=after, before=before, order=order
        )
    )
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("conv-create")
def harness_conv_create(
    items: str | None = typer.Option(
        None, "--items", help="JSON array of seed item dicts (message items)."
    ),
    metadata: str | None = typer.Option(None, "--metadata", help=_METADATA_PAIRS_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/conversations`` — mint a ``conv_*`` container a response
    joins via ``--conversation``. Prints the object (``id`` is the join
    handle)."""
    seed: list[Any] | None = None
    if items is not None:
        try:
            parsed = json.loads(items)
        except json.JSONDecodeError:
            _bad_arg(_ITEMS_JSON_ERR)
            raise AssertionError("unreachable") from None
        if not isinstance(parsed, list):
            _bad_arg(_ITEMS_JSON_ERR)
        seed = parsed
    meta = _json_meta(metadata)
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).conversation_create(
                items=seed, metadata=meta
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_conversation_create(items=seed, metadata=meta))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("conv-get")
def harness_conv_get(
    conversation_id: str = typer.Argument(..., help=_CONV_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/conversations/{id}`` — the conversation object; missing
    ids exit 2."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).conversation_get(conversation_id)
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_conversation_get(conversation_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("conv-update")
def harness_conv_update(
    conversation_id: str = typer.Argument(..., help=_CONV_ID_HELP),
    metadata: str | None = typer.Option(
        None, "--metadata", help="JSON object of string pairs — replaces wholesale."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/conversations/{id}`` — set the conv's metadata (an omitted
    flag clears it)."""
    meta = _json_meta(metadata)
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).conversation_update(
                conversation_id, metadata=meta
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_conversation_update(conversation_id, metadata=meta))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("conv-delete")
def harness_conv_delete(
    conversation_id: str = typer.Argument(..., help=_CONV_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``DELETE /v1/conversations/{id}`` — drop the container and its
    items; member responses stay retrievable on their own ids."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).conversation_delete(conversation_id)
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_conversation_delete(conversation_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("conv-items")
def harness_conv_items(
    conversation_id: str = typer.Argument(..., help=_CONV_ID_HELP),
    limit: int = typer.Option(20, "--limit", min=1, max=100),
    after: str | None = typer.Option(None, "--after", help=_ITEM_CURSOR_HELP),
    before: str | None = typer.Option(None, "--before", help=_ITEM_CURSOR_HELP),
    order: str = typer.Option("asc", "--order", help=_ORDER_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/conversations/{id}/items`` — the conv's accumulated
    items, paged by item id."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).conversation_items(
                conversation_id, limit=limit, after=after, before=before, order=order
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(
        lambda: Fx1Harness().openai_conversation_items(
            conversation_id, limit=limit, after=after, before=before, order=order
        )
    )
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("conv-items-add")
def harness_conv_items_add(
    conversation_id: str = typer.Argument(..., help=_CONV_ID_HELP),
    items: str = typer.Option(..., "--items", help="JSON array of item dicts to append."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/conversations/{id}/items`` — append item dicts; prints
    the minted items list."""
    try:
        parsed = json.loads(items)
    except json.JSONDecodeError:
        _bad_arg(_ITEMS_JSON_ERR)
        raise AssertionError("unreachable") from None
    if not isinstance(parsed, list):
        _bad_arg(_ITEMS_JSON_ERR)
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).conversation_items_add(
                conversation_id, parsed
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_conversation_items_add(conversation_id, parsed))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("conv-item")
def harness_conv_item(
    conversation_id: str = typer.Argument(..., help=_CONV_ID_HELP),
    item_id: str = typer.Argument(..., help="Item id inside the conv."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/conversations/{id}/items/{item_id}`` — one item by id;
    prints the item."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).conversation_item(
                conversation_id, item_id
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_conversation_item(conversation_id, item_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("conv-items-delete")
def harness_conv_item_delete(
    conversation_id: str = typer.Argument(..., help=_CONV_ID_HELP),
    item_id: str = typer.Argument(..., help="Item id inside the conv."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``DELETE /v1/conversations/{id}/items/{item_id}`` — drop one item;
    prints the conv object."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).conversation_item_delete(
                conversation_id, item_id
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_conversation_item_delete(conversation_id, item_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-create")
def harness_vs_create(
    name: str | None = typer.Option(None, "--name", help="Store name (free text)."),
    file_ids: str | None = typer.Option(
        None, "--file-ids", help="JSON array of file-* ids to attach at create."
    ),
    metadata: str | None = typer.Option(None, "--metadata", help=_METADATA_PAIRS_HELP),
    expires_after: str | None = typer.Option(
        None,
        "--expires-after",
        help='JSON expiry policy — e.g. {"anchor":"last_active_at","days":7}',
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/vector_stores`` — mint a ``vs_*`` retrieval store the
    ``file_search`` tool searches on ``respond --tools``."""
    ids: list[str] | None = None
    if file_ids is not None:
        try:
            parsed_ids = json.loads(file_ids)
        except json.JSONDecodeError:
            _bad_arg("--file-ids must be a JSON array of file-* ids")
            raise AssertionError("unreachable") from None
        if not (isinstance(parsed_ids, list) and all(isinstance(i, str) for i in parsed_ids)):
            _bad_arg("--file-ids must be a JSON array of file-* ids")
        ids = parsed_ids
    meta = _json_meta(metadata)
    expiry = _json_obj_opt(expires_after, "--expires-after")
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_create(
                name=name, metadata=meta, file_ids=ids, expires_after=expiry
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(
        lambda: Fx1Harness().vector_store_create(
            name=name, metadata=meta, file_ids=ids, expires_after=expiry
        )
    )
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-get")
def harness_vs_get(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/vector_stores/{id}`` — the store object; missing ids
    exit 2."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_get(vector_store_id)
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().vector_store_get(vector_store_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-update")
def harness_vs_update(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    name: str | None = typer.Option(None, "--name", help="New name (omitted keeps current)."),
    metadata: str | None = typer.Option(
        None, "--metadata", help="JSON object of string pairs — replaces wholesale."
    ),
    expires_after: str | None = typer.Option(
        None,
        "--expires-after",
        help='JSON expiry policy — e.g. {"anchor":"last_active_at","days":7}',
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/vector_stores/{id}`` — set name/metadata (omitted
    fields keep their current values); ``--expires-after`` re-anchors
    the expiry window from the store's last activity."""
    meta = _json_meta(metadata)
    expiry = _json_obj_opt(expires_after, "--expires-after")
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_update(
                vector_store_id, name=name, metadata=meta, expires_after=expiry
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(
        lambda: Fx1Harness().vector_store_update(
            vector_store_id, name=name, metadata=meta, expires_after=expiry
        )
    )
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-delete")
def harness_vs_delete(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``DELETE /v1/vector_stores/{id}`` — drop the store and its index;
    member file-* records survive."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_delete(vector_store_id)
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().vector_store_delete(vector_store_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-list")
def harness_vs_list(
    limit: int = typer.Option(20, "--limit", min=1, max=100),
    after: str | None = typer.Option(None, "--after", help="Page cursor — a vs_* id."),
    before: str | None = typer.Option(None, "--before", help="Page cursor — a vs_* id."),
    order: str = typer.Option("desc", "--order", help=_ORDER_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/vector_stores`` — stores, cursor-paged."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_list(
                limit=limit, after=after, before=before, order=order
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(
        lambda: Fx1Harness().vector_store_list(limit=limit, order=order, after=after, before=before)
    )
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-file-add")
def harness_vs_file_add(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    file_id: str = typer.Argument(..., help=_FILE_ID_HELP),
    attributes: str | None = typer.Option(
        None, "--attributes", help="JSON object — filter keys for file_search."
    ),
    chunking_strategy: str | None = typer.Option(
        None,
        "--chunking-strategy",
        help='JSON: {"type":"auto"} or {"type":"static","static":{...}}.',
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/vector_stores/{id}/files`` — index a file record into
    the store; empty text lands status=failed."""
    attrs = _json_obj_opt(attributes, "--attributes")
    strat = _json_obj_opt(chunking_strategy, "--chunking-strategy")
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_file_create(
                vector_store_id, file_id, attributes=attrs, chunking_strategy=strat
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(
        lambda: Fx1Harness().vector_store_file_create(
            vector_store_id, file_id, attributes=attrs, chunking_strategy=strat
        )
    )
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-files")
def harness_vs_files(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    limit: int = typer.Option(20, "--limit", min=1, max=100),
    after: str | None = typer.Option(None, "--after", help="Page cursor — a file id."),
    before: str | None = typer.Option(None, "--before", help="Page cursor — a file id."),
    order: str = typer.Option("asc", "--order", help=_ORDER_HELP),
    filter: str | None = typer.Option(
        None, "--filter", help="in_progress | completed | cancelled | failed"
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/vector_stores/{id}/files`` — attachments, paged;
    ``--filter`` takes an OpenAI status word."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_file_list(
                vector_store_id,
                limit=limit,
                after=after,
                before=before,
                order=order,
                filter=filter,
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(
        lambda: Fx1Harness().vector_store_file_list(
            vector_store_id, limit=limit, order=order, after=after, before=before, filter=filter
        )
    )
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-file-get")
def harness_vs_file_get(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    file_id: str = typer.Argument(..., help=_FILE_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/vector_stores/{id}/files/{file_id}`` — one attachment's
    status/chunks/attributes."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_file_get(
                vector_store_id, file_id
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().vector_store_file_get(vector_store_id, file_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-file-delete")
def harness_vs_file_delete(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    file_id: str = typer.Argument(..., help=_FILE_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``DELETE /v1/vector_stores/{id}/files/{file_id}`` — detach; the
    file record survives."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_file_delete(
                vector_store_id, file_id
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().vector_store_file_delete(vector_store_id, file_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-file-content")
def harness_vs_file_content(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    file_id: str = typer.Argument(..., help=_FILE_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/vector_stores/{id}/files/{file_id}/content`` — the
    stored decoded text as a page of text parts."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_file_content(
                vector_store_id, file_id
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().vector_store_file_content(vector_store_id, file_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-search")
def harness_vs_search(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    query: str = typer.Option(..., "--query", "-q", help="Search query text."),
    max_num_results: int | None = typer.Option(
        None, "--max-results", help="Cap on returned hits (≤50)."
    ),
    filters_json: str | None = typer.Option(
        None, "--filters", help="JSON attribute-filter object."
    ),
    score_threshold: float | None = typer.Option(
        None, "--score-threshold", help="Cosine floor in [0,1]."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/vector_stores/{id}/search`` — ranked hits without
    spending a response turn."""
    filters = _json_obj_opt(filters_json, "--filters")
    ranking_options = {"score_threshold": score_threshold} if score_threshold is not None else None
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_search(
                vector_store_id,
                query,
                max_num_results=max_num_results,
                filters=filters,
                ranking_options=ranking_options,
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(
        lambda: Fx1Harness().vector_store_search(
            vector_store_id,
            query,
            max_num_results=max_num_results,
            filters=filters,
            ranking_options=ranking_options,
        )
    )
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-batch-create")
def harness_vs_batch_create(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    file_ids: list[str] = typer.Argument(..., help="file-* ids to attach (1..500)."),
    attributes: str | None = typer.Option(
        None, "--attributes", help="JSON object applied to every member."
    ),
    chunking_strategy: str | None = typer.Option(
        None, "--chunking-strategy", help="JSON object (auto|static)."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/vector_stores/{id}/file_batches`` — attach many files
    in one call; per-file refusals count ``failed``, never abort."""
    attrs = _json_obj_opt(attributes, "--attributes")
    strategy = _json_obj_opt(chunking_strategy, "--chunking-strategy")
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_file_batch_create(
                vector_store_id,
                file_ids,
                attributes=attrs,
                chunking_strategy=strategy,
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(
        lambda: Fx1Harness().vector_store_file_batch_create(
            vector_store_id,
            file_ids,
            attributes=attrs,
            chunking_strategy=strategy,
        )
    )
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-batch-get")
def harness_vs_batch_get(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    batch_id: str = typer.Argument(..., help=_VSFB_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/vector_stores/{id}/file_batches/{batch_id}`` — status +
    file_counts."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_file_batch_get(
                vector_store_id, batch_id
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().vector_store_file_batch_get(vector_store_id, batch_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-batch-cancel")
def harness_vs_batch_cancel(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    batch_id: str = typer.Argument(..., help=_VSFB_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST .../file_batches/{batch_id}/cancel`` — members attach
    synchronously at create, so a batch is always terminal: this
    answers 409 ``file_batch_terminal`` (exit 2)."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_file_batch_cancel(
                vector_store_id, batch_id
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().vector_store_file_batch_cancel(vector_store_id, batch_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("vs-batch-files")
def harness_vs_batch_files(
    vector_store_id: str = typer.Argument(..., help=_VS_ID_HELP),
    batch_id: str = typer.Argument(..., help=_VSFB_ID_HELP),
    limit: int = typer.Option(20, "--limit", min=1, max=100),
    after: str | None = typer.Option(None, "--after", help="Page cursor — a file-* id."),
    before: str | None = typer.Option(None, "--before", help="Page cursor — a file-* id."),
    order: str = typer.Option("asc", "--order", help=_ORDER_HELP),
    filter: str | None = typer.Option(
        None, "--filter", help="in_progress | completed | cancelled | failed"
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET .../file_batches/{batch_id}/files`` — the frozen per-file
    verdicts in request order."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).vector_store_file_batch_files(
                vector_store_id,
                batch_id,
                limit=limit,
                after=after,
                before=before,
                order=order,
                filter=filter,
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(
        lambda: Fx1Harness().vector_store_file_batch_files(
            vector_store_id,
            batch_id,
            limit=limit,
            after=after,
            before=before,
            order=order,
            filter=filter,
        )
    )
    typer.echo(json.dumps(out, indent=2))


@app.command("eval")
def eval_bank(
    backend: str = typer.Option("hosted_k3", help=_BACKEND_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    out: Path = typer.Option(Path("data/fx1/eval.json")),
    model: str | None = typer.Option(
        None,
        help="Model id sent to a hosted backend (default: the backend's pin).",
    ),
) -> None:
    """Run the built-in eval task bank against an fx-1 backend."""
    from fx1.eval import DEFAULT_BANK, run_suite

    model_backend = _resolve_model_backend(backend, checkpoint_dir, model)
    summary = run_suite(model_backend, list(DEFAULT_BANK))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {"by_kind": summary["by_kind"], "honesty_gate_passed": summary["honesty_gate_passed"]},
            indent=2,
        )
    )


def _resolve_model_backend(
    backend: str, checkpoint_dir: Path | None, model: str | None = None
) -> ModelFn:
    """Resolve the model under test. Fail-closed on unknown names.

    ``local_fx1`` needs ``--checkpoint-dir``; ``byok`` reads the
    FX1_BYOK_* env contract (never argv); ``hosted_k3`` needs
    MOONSHOT_API_KEY. An unknown backend string exits 2 — silently
    running a different backend than the flag names would corrupt the
    receipt's meaning.
    """
    from fx1.serve import get_backend

    if backend == "local_fx1":
        return get_backend("local_fx1", checkpoint_dir=checkpoint_dir).complete
    if backend in ("hosted_k3", "byok"):
        kwargs: dict[str, object] = {}
        if model is not None:
            kwargs["model"] = model
        return get_backend(backend, **kwargs).complete
    typer.echo(f"unknown --backend {backend!r}; choose hosted_k3 | local_fx1 | byok", err=True)
    raise typer.Exit(code=2)


_JUDGE_BACKEND_HELP = (
    "Optional 'hosted_k3' LLM judge for MT-Bench-style grading (needs "
    "MOONSHOT_API_KEY); omit for the deterministic rule-based judge."
)


def _resolve_judge(judge_backend: str | None) -> ModelFn | None:
    """Build the optional ext-bench judge ModelFn. Fail-closed on usage.

    ``None`` selects the deterministic rule-based judge by configuration;
    ``hosted_k3`` reuses the same backend construction as the model under
    test (a missing MOONSHOT_API_KEY raises, never fabricates a judge).
    """
    if judge_backend is None:
        return None
    if judge_backend not in ("hosted_k3", "byok"):
        typer.echo(
            f"unknown --judge-backend {judge_backend!r}; 'hosted_k3' and "
            "'byok' are supported (omit the flag for the deterministic "
            "rule-based judge)",
            err=True,
        )
        raise typer.Exit(code=2)
    from fx1.serve import get_backend

    return get_backend(judge_backend).complete


@app.command("capability-eval")
def capability_eval(
    backend: str = typer.Option("hosted_k3", help=_BACKEND_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    seed: int = typer.Option(0, help="Seeded SYNTHETIC bank seed."),
    judge_backend: str | None = typer.Option(None, "--judge-backend", help=_JUDGE_BACKEND_HELP),
    out: Path = typer.Option(Path("data/fx1/capability_eval.json")),
) -> None:
    """Run the capability battery: time-series reasoning, probability
    calibration, harness tool-use, retrieval-with-citation, external-
    benchmark-format adapters (MT-Bench / FinanceBench / FinToolBench-style),
    and options reasoning — all on seeded SYNTHETIC banks. Exit 1 when any
    honesty sub-gate, the calibration gate, or an ext-bench score gate
    fails. Real ext-bench JSONL sources plug in via `fx1 ext-bench-eval`;
    the options bank is sealed (no external loader)."""
    from fx1.eval import run_capability_eval

    judge = _resolve_judge(judge_backend)
    model = _resolve_model_backend(backend, checkpoint_dir)
    report = run_capability_eval(model, seed=seed, judge=judge)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {
                "ts_reasoning_overall": report.ts_reasoning.overall,
                "calibration_ece": report.calibration.ece,
                "calibration_passed": report.calibration.passed,
                "tooluse_pass_rate": report.tooluse.pass_rate,
                "retrieval_accuracy": report.retrieval.accuracy,
                "ext_bench_honesty_gate_passed": (
                    report.ext_bench.honesty_gate_passed if report.ext_bench is not None else None
                ),
                "ext_bench_score_gate_passed": (
                    report.ext_bench.score_gate_passed if report.ext_bench is not None else None
                ),
                "options_reasoning_passed": (
                    report.options_reasoning.passed
                    if report.options_reasoning is not None
                    else None
                ),
                "honesty_gate_passed": report.honesty_gate_passed,
                "passed": report.passed,
            },
            indent=2,
        )
    )
    raise typer.Exit(code=0 if report.passed else 1)


@app.command("ext-bench-eval")
def ext_bench_eval(
    backend: str = typer.Option("hosted_k3", help=_BACKEND_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    seed: int = typer.Option(0, help="Seeded SYNTHETIC bank seed."),
    judge_backend: str | None = typer.Option(None, "--judge-backend", help=_JUDGE_BACKEND_HELP),
    mtbench_jsonl: Path | None = typer.Option(
        None,
        "--mtbench-jsonl",
        help="Genuine MT-Bench-style JSONL replacing the sealed synthetic bank.",
    ),
    financebench_jsonl: Path | None = typer.Option(
        None,
        "--financebench-jsonl",
        help="Genuine FinanceBench-style JSONL replacing the sealed synthetic bank.",
    ),
    fintoolbench_jsonl: Path | None = typer.Option(
        None,
        "--fintoolbench-jsonl",
        help="Genuine FinToolBench-style JSONL replacing the sealed synthetic bank.",
    ),
    out: Path = typer.Option(Path("data/fx1/ext_bench_eval.json")),
) -> None:
    """Run the external-benchmark-format adapters (MT-Bench / FinanceBench /
    FinToolBench-style). Default banks are sealed SYNTHETIC correctness
    gates — NOT market evidence and NOT real benchmark scores; genuine JSONL
    exports plug in per benchmark (schema-validated, fail-closed). Exit 1
    when any refusal/honesty gate or score gate fails."""
    from fx1.eval.ext_bench import run_ext_bench_eval

    judge = _resolve_judge(judge_backend)
    model = _resolve_model_backend(backend, checkpoint_dir)
    sources: dict[str, Path] = {}
    if mtbench_jsonl is not None:
        sources["mtbench"] = mtbench_jsonl
    if financebench_jsonl is not None:
        sources["financebench"] = financebench_jsonl
    if fintoolbench_jsonl is not None:
        sources["fintoolbench"] = fintoolbench_jsonl
    report = run_ext_bench_eval(model, seed=seed, judge=judge, sources=sources or None)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {
                "synthetic": report.synthetic,
                "label": report.label,
                "benchmarks": {
                    name: {
                        "n_instances": score.n_instances,
                        "metrics": score.metrics,
                        "refusal_gate_passed": score.refusal_gate_passed,
                        "honesty_violations": score.honesty_violations,
                        "gate_passed": score.gate_passed,
                    }
                    for name, score in report.benchmarks.items()
                },
                "honesty_gate_passed": report.honesty_gate_passed,
                "passed": report.passed,
            },
            indent=2,
        )
    )
    raise typer.Exit(code=0 if report.passed else 1)


@app.command("options-reasoning-eval")
def options_reasoning_eval(
    backend: str = typer.Option("hosted_k3", help=_BACKEND_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    seed: int = typer.Option(0, help="Seeded SYNTHETIC bank seed."),
    out: Path = typer.Option(Path("data/fx1/options_reasoning_eval.json")),
) -> None:
    """Run the sealed SYNTHETIC options-reasoning battery (LiveOption-
    inspired levels: action validity, decision quality, risk characteristics,
    outcome, bait refusals). Gold answers come from the repo's own pricing
    modules — correctness gates, NOT market evidence. The bank is sealed
    (no external JSONL loader exists), so there is no --options-jsonl
    pass-through. Exit 1 when the bait/honesty gate fails."""
    from fx1.eval.options_reasoning_eval import run_options_reasoning_eval

    model = _resolve_model_backend(backend, checkpoint_dir)
    report = run_options_reasoning_eval(model, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {
                "label": report.label,
                "n_items": report.n_items,
                "overall": report.overall,
                "by_level": report.by_level,
                "bait_accuracy": report.bait_accuracy,
                "bait_gate_passed": report.bait_gate_passed,
                "honesty_violations": report.honesty_violations,
                "passed": report.passed,
            },
            indent=2,
        )
    )
    raise typer.Exit(code=0 if report.passed else 1)


@app.command("modelcard")
def modelcard_validate(path: Path = typer.Argument(...)) -> None:
    """Validate an fx-1 model card and report ship-gate status."""
    from fx1.modelcard import ModelCard

    card = ModelCard.load(path)
    typer.echo(
        json.dumps(
            {"version": card.version, "ship_eligible": card.eval_delta.ship_eligible}, indent=2
        )
    )


@app.command("redteam")
def redteam(
    backend: str = typer.Option("hosted_k3"),
    checkpoint_dir: Path | None = typer.Option(None),
) -> None:
    """Run the adversarial red-team suite against an fx-1 backend."""
    from fx1.eval.redteam import REDTEAM_TASKS
    from fx1.eval.suite import run_suite

    model = _resolve_model_backend(backend, checkpoint_dir)
    summary = run_suite(model, list(REDTEAM_TASKS))
    failed = [r["task"] for r in summary.results if not r["passed"]]
    typer.echo(
        json.dumps(
            {"honesty_gate_passed": summary["honesty_gate_passed"], "results": failed}, indent=2
        )
    )
    raise typer.Exit(code=0 if summary["honesty_gate_passed"] else 1)


@app.command("dpo")
def dpo_build(
    out: Path = typer.Option(Path("data/fx1/dpo.jsonl")),
) -> None:
    """Build contract-derived DPO preference pairs."""
    from fx1.train.dpo import build_preference_pairs

    pairs = build_preference_pairs(out)
    typer.echo(json.dumps({"pairs": len(pairs), "out": str(out)}))


@app.command("curriculum")
def curriculum_build(
    corpus: Path = typer.Option(Path("data/fx1/corpus.jsonl")),
    out: Path = typer.Option(Path("data/fx1/corpus_curriculum.jsonl")),
) -> None:
    """Order the corpus contracts -> interpretation -> loops -> refusal."""
    from fx1.train.curriculum import build_curriculum

    counts = build_curriculum(corpus, out)
    typer.echo(json.dumps(counts, indent=2))


@app.command("maskedaEval")
def masked_eval(
    budget: float = typer.Option(0.25, help="Memory-gap ship budget."),
) -> None:
    """Masked/unmasked twin evaluation + memory-gap ship metric (offline
    structural check; live model runs inject a backend via fx1.eval)."""
    from fx1.eval import DEFAULT_BANK, masked_twins

    twins = masked_twins(list(DEFAULT_BANK))
    typer.echo(
        json.dumps(
            {
                "twin_tasks": len(twins),
                "memory_gap_budget": budget,
                "note": "inject a backend via fx1.eval.run_suite for live scoring",
            },
            indent=2,
        )
    )


@app.command("contamination-audit")
def contamination_audit(
    corpus: Path = typer.Option(Path("data/fx1/corpus.jsonl")),
    out: Path = typer.Option(Path("data/fx1/contamination_report.json")),
    with_rephrased_gap: bool = typer.Option(
        False,
        "--with-rephrased-gap",
        help="Also run the canonical-vs-rephrased gap probe against a live backend.",
    ),
    backend: str = typer.Option("hosted_k3", help="Backend for the gap probe."),
) -> None:
    """Run the publishable contamination audit over the corpus vs eval bank."""
    from fx1.eval import eval_prompt_surface, run_contamination_audit

    # Fail-closed: an absent or empty corpus certifies nothing — auditing
    # zero texts would vacuously report "not contaminated" and exit 0.
    if not corpus.exists():
        typer.echo(f"corpus not found: {corpus}; refusing to certify an empty audit", err=True)
        raise typer.Exit(code=2)
    texts = []
    for line in corpus.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            texts.append(" ".join(m.get("content", "") for m in record.get("messages", [])))
    if not texts:
        typer.echo(f"corpus {corpus} contains no examples; audit cannot certify", err=True)
        raise typer.Exit(code=2)
    prompts = eval_prompt_surface()
    report = run_contamination_audit(texts, prompts)
    if with_rephrased_gap:
        from fx1.eval import run_rephrased_gap
        from fx1.serve import get_backend

        model = get_backend(backend)
        report.probes.append(run_rephrased_gap(model.complete))
        report.overall_flagged = report.overall_flagged or any(p.flagged for p in report.probes)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {
                "overall_flagged": report.overall_flagged,
                "ngram_hits": len(report.ngram_hits),
                "probes": [p.method for p in report.probes],
            },
            indent=2,
        )
    )
    raise typer.Exit(code=1 if report.overall_flagged else 0)


@app.command("sign")
def sign_checkpoint(
    checkpoint_dir: Path = typer.Argument(...),
) -> None:
    """Sign an fx-1 release (attestation ladder tier 1)."""
    from fx1.serve import sign_release

    sig = sign_release(checkpoint_dir)
    typer.echo(f"signed: {sig}")


@app.command("attestation")
def attestation_status(
    checkpoint_dir: Path = typer.Argument(...),
) -> None:
    """Report which attestation tiers a checkpoint satisfies."""
    from fx1.serve import attestation_ladder_status

    typer.echo(json.dumps(attestation_ladder_status(checkpoint_dir), indent=2))


@app.command("sbom")
def sbom_generate(
    lockfile: Path = typer.Option(Path("uv.lock")),
    out: Path = typer.Option(Path("data/fx1/sbom.json")),
) -> None:
    """Generate a hash-pinned SBOM from the locked dependency set."""
    from fx1.sbom import generate_sbom

    sbom = generate_sbom(lockfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sbom.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {"entries": len(sbom.entries), "lockfile_sha256": sbom.lockfile_sha256[:16] + "…"},
            indent=2,
        )
    )


@app.command("mrm")
def mrm_dossier(
    modelcard: Path = typer.Option(..., help="Model card JSON."),
    validation_artifact: Path = typer.Option(..., help="Contamination report JSON."),
    artifact: list[str] | None = typer.Option(
        None,
        "--artifact",
        help="Extra activity evidence as activity=path (repeatable; "
        "development/implementation/monitoring/governance).",
    ),
    out: Path = typer.Option(Path("data/fx1/mrm_dossier.json")),
) -> None:
    """Compile the five-activity model-risk dossier (SR 26-2 era)."""
    from fx1.mrm import compile_dossier

    artifacts: dict[str, str | Path] = {"validation": validation_artifact}
    for spec in artifact or []:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise typer.BadParameter("--artifact expects activity=path")
        artifacts[name] = path

    dossier = compile_dossier(
        modelcard_path=modelcard,
        artifacts=artifacts,
        out_path=out,
    )
    typer.echo(
        json.dumps(
            {
                "version": dossier.model_version,
                "complete": dossier.complete,
                "ship_eligible": dossier.ship_eligible,
                "contamination_flagged": dossier.contamination_flagged,
            },
            indent=2,
        )
    )


@app.command("dipbench")
def dipbench_demo(
    data_dir: Path | None = typer.Option(
        None, help="Directory of *_1d.parquet series for a real-data run."
    ),
    out: Path | None = typer.Option(None, help="Receipt JSON path (real-data run)."),
    threshold: float = typer.Option(0.10, help="Drawdown depth that defines a dip."),
) -> None:
    """Dip Quality Score bench. Without --data-dir: smoke the bench on a
    built-in synthetic series (SYNTHETIC — correctness only, not market
    evidence). With --data-dir: full receipt over real historical bars."""
    if data_dir is not None:
        from fx1.bench.run import run_dip_bench

        receipt = run_dip_bench(data_dir, threshold=threshold)
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        typer.echo(
            json.dumps(
                {
                    "label": "research/backtest evidence, not live performance",
                    "events": receipt["n_events"],
                    "baseline_recovery": receipt["baseline_recovery"],
                    "receipt": str(out) if out else None,
                },
                indent=2,
            )
        )
        return
    from fx1.bench.dip import (
        DipForecast,
        detect_dip_events,
        evaluate_forecasts,
        unconditional_baseline,
    )

    closes = [100.0, 102.0, 88.0, 90.0, 103.0, 104.0, 92.0, 95.0, 106.0]
    dates = [f"2026-01-{i + 1:02d}" for i in range(len(closes))]
    events = detect_dip_events(closes, dates, "SYNTHETIC", threshold=0.10, horizons_bars={"1m": 3})
    baseline = unconditional_baseline(events)
    forecasts = [DipForecast(e.asset, e.trough_date, {"1m": 0.8}) for e in events]
    metrics = evaluate_forecasts(events, forecasts)
    typer.echo(
        json.dumps(
            {"label": "SYNTHETIC", "events": len(events), "baseline": baseline, "metrics": metrics},
            indent=2,
        )
    )


@app.command("infer")
def infer_cmd(
    config: Path = typer.Option(..., "--config", help="Harness YAML or JSON config."),
) -> None:
    """Run batch or walk-forward inference and write a forecast parquet.

    The fx-1 forecaster is external: ``model.name=fx-1`` requires
    ``model.entrypoint``. Reference names ``dummy-zero`` and ``dummy-momentum``
    are not fx-1. This command does not train and does not place orders.
    """
    from fx1.forecast.config import load_harness_config
    from fx1.forecast.runner import run_inference

    result = run_inference(load_harness_config(config))
    typer.echo(
        json.dumps(
            {
                "forecasts": str(result.parquet_path),
                "metadata": str(result.meta_path),
                "n_rows": result.n_rows,
                "model_name": result.metadata["model_name"],
                "model_role": result.metadata["model_role"],
                "data_label": result.metadata["data_label"],
                "research_only": True,
                "live_pnl_claim": False,
            },
            indent=2,
        )
    )


@app.command("backtest")
def backtest_cmd(
    config: Path = typer.Option(..., "--config", help="Harness YAML or JSON config."),
    forecasts: Path | None = typer.Option(
        None,
        "--forecasts",
        help="Forecast parquet. Defaults to inference.output_parquet in the config.",
    ),
) -> None:
    """Score forecasts and a placeholder signal map.

    Reports forecast scores (IC, rank IC, hit rate, MAE, RMSE) and research
    diagnostics of the placeholder mapping. Does not place orders. Run
    ``fx1 infer`` first when the forecast parquet is not already on disk.
    """
    from fx1.forecast.config import load_harness_config
    from fx1.forecast.runner import run_signal_evaluation

    cfg = load_harness_config(config)
    frame = None
    if forecasts is not None:
        import polars as pl

        frame = pl.read_parquet(forecasts)
    report = run_signal_evaluation(cfg, forecasts=frame)
    typer.echo(json.dumps(report, indent=2))


@app.command("doctor")
def doctor(root: Path = typer.Option(Path("."), help="Repo root to inspect.")) -> None:
    """fx-1 readiness status (presence flags only — never secret values)."""
    from fx1.doctor import collect_status

    typer.echo(json.dumps(collect_status(root), indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
