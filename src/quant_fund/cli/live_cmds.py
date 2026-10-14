"""Operator live desk commands.

Shadow transmits nothing. Loopback fills are SYNTHETIC. Drop writes order
tickets only after ``--confirm SEND-LIVE-ORDERS`` and ``DIP_LIVE_ARM=1``.
Reports keep live_pnl_claim false. Evidence conditions stay blocked.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from collections.abc import Callable
from pathlib import Path

import typer

from quant_fund.cli._app import app
from quant_fund.live.arming import ENV_ROOT, ArmingError
from quant_fund.live.desk import LiveDesk, LiveDeskError

live_app = typer.Typer(
    help=(
        "Live desk. Shadow by default. Loopback is a SYNTHETIC rehearsal. "
        "Drop writes tickets only when armed and DIP_LIVE_ARM=1. "
        "No live-performance claim."
    )
)
app.add_typer(live_app, name="live")

_CONFIG = Path("configs/pretrade_risk.yaml")


def _default_root() -> Path:
    override = os.environ.get(ENV_ROOT)
    if override:
        return Path(override)
    return Path.home() / ".dipcatcher" / "live"


def _resolve(root: Path | None) -> Path:
    return _default_root() if root is None else root


def _key(root: Path) -> bytes:
    path = root / "hmac.key"
    if path.exists():
        key = path.read_bytes()
        if len(key) < 16:
            raise LiveDeskError("hmac.key must be at least 16 bytes")
        return key
    root.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, key)
    finally:
        os.close(fd)
    return key


def _open(
    root: Path,
    config_path: Path,
    *,
    cash: float | None,
    account: float | None,
    ts_ns: int | None,
) -> LiveDesk:
    from quant_fund.pretrade.config import load_pretrade_config

    key = _key(root)
    config, _digest, _sig = load_pretrade_config(config_path, hmac_key=key)
    return LiveDesk.open(
        root,
        hmac_key=key,
        config=config,
        cash=cash,
        account_value=account,
        ts_ns=ts_ns,
    )


def _emit(payload: dict[str, object]) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _fail(exc: Exception) -> None:
    _emit(
        {
            "schema": "live_desk_report.v1",
            "ok": False,
            "error": str(exc),
            "live_pnl_claim": False,
            "would_promote_live": False,
            "evidence_conditions": "blocked",
        }
    )
    raise typer.Exit(1) from exc


def _session(
    root: Path | None,
    config: Path,
    fn: Callable[[LiveDesk], dict[str, object]],
    *,
    cash: float | None = None,
    account: float | None = None,
    ts_ns: int | None = None,
) -> None:
    try:
        desk = _open(_resolve(root), config, cash=cash, account=account, ts_ns=ts_ns)
    except (LiveDeskError, ArmingError, OSError, ValueError) as exc:
        _fail(exc)
        return
    try:
        try:
            _emit(fn(desk))
        except (LiveDeskError, ArmingError, OSError, ValueError) as exc:
            _fail(exc)
    finally:
        desk.close()


def _stamp(ts_ns: int | None) -> int:
    return time.time_ns() if ts_ns is None else ts_ns


@live_app.callback(invoke_without_command=True)
def live_callback(ctx: typer.Context) -> None:
    """Show the default desk when no subcommand is given.

    ``dipcatcher live status --root DIR`` selects a directory. The bare
    command reads ``~/.dipcatcher/live``.
    """
    if ctx.invoked_subcommand is not None:
        return
    _session(None, _CONFIG, lambda desk: desk.status())


@live_app.command("status")
def live_status(
    root: Path | None = typer.Option(None, "--root", help="Desk directory."),
    config: Path = typer.Option(_CONFIG),
) -> None:
    """Print venue, halt, positions, and whether this process may transmit."""
    _session(root, config, lambda desk: desk.status())


@live_app.command("arm")
def live_arm(
    venue: str = typer.Option(..., help="shadow, loopback, or drop."),
    confirm: str = typer.Option("", help="Drop requires SEND-LIVE-ORDERS."),
    root: Path | None = typer.Option(None, "--root"),
    config: Path = typer.Option(_CONFIG),
    ts_ns: int | None = typer.Option(None, help="Event time in nanoseconds. Default: now."),
    cash: float | None = typer.Option(
        None, help="Starting settled cash. Required to fund a new desk."
    ),
    account: float | None = typer.Option(
        None, help="Starting account value. Required to fund a new desk."
    ),
) -> None:
    """Select the venue. Drop still does not transmit until DIP_LIVE_ARM=1."""
    stamp = _stamp(ts_ns)
    _session(
        root,
        config,
        lambda desk: desk.arm(venue, confirm, stamp),
        cash=cash,
        account=account,
        ts_ns=stamp,
    )


@live_app.command("disarm")
def live_disarm(
    root: Path | None = typer.Option(None, "--root"),
    config: Path = typer.Option(_CONFIG),
    ts_ns: int | None = typer.Option(None),
) -> None:
    """Return the desk to shadow. Tickets already written are not recalled."""
    stamp = _stamp(ts_ns)
    _session(root, config, lambda desk: desk.disarm(stamp), ts_ns=stamp)


@live_app.command("submit")
def live_submit(
    symbol: str = typer.Option(..., help="Ticker. Stored uppercase."),
    side: str = typer.Option(..., help="BUY or SELL."),
    qty: float = typer.Option(..., help="Limit quantity."),
    mark: float = typer.Option(..., help="Limit price and reference mark."),
    root: Path | None = typer.Option(None, "--root"),
    config: Path = typer.Option(_CONFIG),
    ts_ns: int | None = typer.Option(None, help="Decision time in nanoseconds."),
    client_id: str | None = typer.Option(None, help="Idempotency key."),
    cash: float | None = typer.Option(None, help="Starting settled cash. Only on a new desk."),
    account: float | None = typer.Option(None, help="Starting account value. Only on a new desk."),
) -> None:
    """Submit one limit. Shadow and an unarmed drop record it and send nothing."""
    stamp = _stamp(ts_ns)

    def run(desk: LiveDesk) -> dict[str, object]:
        return desk.submit(symbol, side, qty, mark, stamp, client_order_id=client_id)

    _session(root, config, run, cash=cash, account=account, ts_ns=stamp)


@live_app.command("cancel")
def live_cancel(
    order_id: str = typer.Option(...),
    root: Path | None = typer.Option(None, "--root"),
    config: Path = typer.Option(_CONFIG),
    ts_ns: int | None = typer.Option(None),
) -> None:
    """Request a cancel. A ticket already sent needs DIP_LIVE_ARM=1 to cancel."""
    stamp = _stamp(ts_ns)
    _session(root, config, lambda desk: desk.cancel(order_id, stamp), ts_ns=stamp)


@live_app.command("poll")
def live_poll(
    root: Path | None = typer.Option(None, "--root"),
    config: Path = typer.Option(_CONFIG),
    ts_ns: int | None = typer.Option(None),
) -> None:
    """Read inbound JSON events and apply acks, fills, and cancels."""
    stamp = _stamp(ts_ns)
    _session(root, config, lambda desk: desk.poll(stamp), ts_ns=stamp)


@live_app.command("kill")
def live_kill(
    reason: str = typer.Option(..., help="Why new orders are stopping."),
    actor: str = typer.Option("operator"),
    root: Path | None = typer.Option(None, "--root"),
    config: Path = typer.Option(_CONFIG),
    ts_ns: int | None = typer.Option(None),
) -> None:
    """Latch the kill switch. Working tickets are not flattened by themselves."""
    stamp = _stamp(ts_ns)
    _session(root, config, lambda desk: desk.kill(actor, reason, stamp), ts_ns=stamp)


@live_app.command("reset")
def live_reset(
    reason: str = typer.Option(..., help="Why the halt is clearing."),
    actor: str = typer.Option("operator"),
    root: Path | None = typer.Option(None, "--root"),
    config: Path = typer.Option(_CONFIG),
    ts_ns: int | None = typer.Option(None),
) -> None:
    """Clear a halt. Drop transmission resumes if DIP_LIVE_ARM=1 is still set."""
    stamp = _stamp(ts_ns)
    _session(root, config, lambda desk: desk.reset(actor, reason, stamp), ts_ns=stamp)


@live_app.command("flatten")
def live_flatten(
    root: Path | None = typer.Option(None, "--root"),
    config: Path = typer.Option(_CONFIG),
    ts_ns: int | None = typer.Option(None),
) -> None:
    """Submit a reduce-only limit at the last mark for each position."""
    stamp = _stamp(ts_ns)
    _session(root, config, lambda desk: desk.flatten(stamp), ts_ns=stamp)
