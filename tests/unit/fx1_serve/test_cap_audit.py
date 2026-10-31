"""Tests for fx1.serve.cap_audit — the config × surface capability matrix."""

from __future__ import annotations

from fx1.serve.cap_audit import cap_audit, cap_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_all_probes_hold() -> None:
    results = cap_audit()
    assert len(results) >= 30
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies() -> None:
    blob = cap_audit_bench()
    assert blob["claim"]["ok"] is True
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_probe_results_repeat() -> None:
    # Each run seals its observed wire verdicts. Independent measurements
    # can differ under load while still satisfying every capability probe.
    first = cap_audit_bench()
    second = cap_audit_bench()
    assert first["claim"]["results"] == second["claim"]["results"]
    assert first["claim"]["ok"] is True
    assert second["claim"]["ok"] is True
    assert verify_receipt_payload(first)["valid"] is True
    assert verify_receipt_payload(second)["valid"] is True


def test_receipt_matrix_honest() -> None:
    """The coverage matrix must carry every config and declare real gaps —
    an all-200 matrix would mean the battery measured nothing."""
    blob = cap_audit_bench()
    matrix = blob["coverage"]["matrix"]
    assert set(matrix) == {
        "stub",
        "plain",
        "unconf",
        "byok",
        "byok_bad",
        "byok_notok",
        "local",
        "hosted",
        "managed",
    }
    assert matrix["unconf"]["chat"].startswith("503")
    assert matrix["plain"]["tools"].startswith("501")
    assert matrix["byok_notok"]["tokens"].startswith("501")


def test_committed_receipt_still_verifies() -> None:
    import json
    from pathlib import Path

    path = Path("receipts/fx1_cap_audit.json")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["claim"]["ok"] is True
    verdict = verify_receipt_payload(payload)
    assert verdict["valid"] is True
    assert verdict["errors"] == []
