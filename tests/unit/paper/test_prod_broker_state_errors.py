from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.execution.order_recon import OrderLedgerError
from quant_fund.paper.broker_adapter import PaperBrokerAdapter


def _state() -> dict[str, object]:
    return {
        "schema": "paper_broker_state.v1",
        "endpoint_kind": "paper",
        "starting_cash": 0.0,
        "nav_tolerance": 1e-6,
        "orders": [],
        "fills": [],
        "cancelled_order_ids": [],
        "marks": {},
        "kill_state": "CLEAR",
        "nav": 0.0,
    }


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_seam_report_requires_an_existing_state_file(tmp_path: Path) -> None:
    adapter = PaperBrokerAdapter(state_path=tmp_path / "missing.json")
    with pytest.raises(OrderLedgerError, match="no broker state"):
        adapter.seam_report()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "is not an object"),
        ({"schema": "other"}, "schema must be paper_broker_state.v1"),
    ],
)
def test_seam_report_rejects_invalid_state_envelopes(
    tmp_path: Path, payload: object, message: str
) -> None:
    path = tmp_path / "state.json"
    _write(path, payload)
    adapter = PaperBrokerAdapter(state_path=path)
    with pytest.raises(OrderLedgerError, match=message):
        adapter.seam_report()


def test_resume_rejects_invalid_cancelled_order_ids(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = _state()
    state["cancelled_order_ids"] = ["order-1", ""]
    _write(path, state)

    with pytest.raises(OrderLedgerError, match="list of non-empty strings"):
        PaperBrokerAdapter.resume(path)


def test_seam_report_rejects_a_cancel_for_an_unknown_order(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = _state()
    state["cancelled_order_ids"] = ["missing-order"]
    _write(path, state)
    adapter = PaperBrokerAdapter(state_path=path)

    with pytest.raises(OrderLedgerError, match="cannot replay cancel of missing-order"):
        adapter.seam_report()


def test_seam_report_logs_endpoint_and_nav_mismatches(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = _state()
    state["nav"] = 5.0
    _write(path, state)
    adapter = PaperBrokerAdapter(state_path=path, endpoint_kind="simulated")

    report = adapter.seam_report()
    assert report["ok"] is False
    assert report["nav_seam_parity"] is False
    assert report["nav_delta"] == pytest.approx(5.0)
    assert report["errors"] == ["endpoint_kind_mismatch", "nav_seam_mismatch"]
    assert report["live_pnl_claim"] is False
