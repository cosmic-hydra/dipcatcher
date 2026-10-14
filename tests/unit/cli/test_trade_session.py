"""CLI for the paper trading session. Live orders stay refused."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from quant_fund.cli.main import app

runner = CliRunner()


def test_trade_readiness_reports_the_block() -> None:
    result = runner.invoke(app, ["trade", "readiness"])
    assert result.exit_code == 0, result.output
    assert "DATA_LABEL=SIMULATED" in result.output
    assert "Live orders: disabled" in result.output
    assert '"eligible": false' in result.output
    assert '"live_orders_enabled": false' in result.output
    assert "Traceback" not in result.output


def test_trade_session_runs_a_simulated_book(tmp_path: Path) -> None:
    commands = tmp_path / "orders.jsonl"
    commands.write_text(
        "\n".join(
            [
                json.dumps({"op": "mark", "security_id": "AAPL", "price": 20}),
                json.dumps(
                    {
                        "op": "submit",
                        "order_id": "o1",
                        "security_id": "AAPL",
                        "side": "buy",
                        "quantity": 2,
                    }
                ),
                json.dumps(
                    {
                        "op": "fill",
                        "fill_id": "f1",
                        "order_id": "o1",
                        "security_id": "AAPL",
                        "quantity": 2,
                        "price": 20,
                        "fill_time": "2026-10-13T15:00:00+00:00",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    state = tmp_path / "book.json"
    result = runner.invoke(
        app,
        [
            "trade",
            "session",
            "--state",
            str(state),
            "--commands",
            str(commands),
            "--starting-cash",
            "1000",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "PAPER_SESSION_ONLY" in result.output
    assert state.is_file()
    payload = json.loads(result.output.split("PAPER_SESSION_ONLY", 1)[1].split("\n", 1)[1])
    assert payload["consistent"] is True
    assert payload["reconciliation"]["computed_positions"]["AAPL"] == 2.0
    assert payload["live_pnl_claim"] is False


def test_trade_session_exits_on_a_refused_order(tmp_path: Path) -> None:
    commands = tmp_path / "orders.jsonl"
    commands.write_text(
        json.dumps(
            {"op": "submit", "order_id": "o1", "security_id": "AAPL", "side": "BUY", "quantity": 1}
        )
        + "\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["trade", "session", "--state", str(tmp_path / "book.json"), "--commands", str(commands)],
    )
    assert result.exit_code == 1
    assert "missing_mark" in result.output
    assert "Traceback" not in result.output


def test_trade_rejects_unknown_op() -> None:
    result = runner.invoke(app, ["trade", "live"])
    assert result.exit_code != 0
    assert "Traceback" not in (result.output + str(result.exception))
