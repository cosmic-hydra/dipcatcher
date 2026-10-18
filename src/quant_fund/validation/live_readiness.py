"""Fail-closed live-trading readiness. Always ineligible.

Code cannot purchase a vendor feed, authorize a broker, or sign a
live-capital approval. This module imports nothing else in the lab so the
chat shell can print the block without reaching the order path.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def live_readiness_report() -> dict[str, Any]:
    """The five live-trading conditions. There is no argument that flips this."""
    return {
        "schema": "live_readiness.v1",
        "eligible": False,
        "live_orders_enabled": False,
        "would_promote_live": False,
        "live_pnl_claim": False,
        "research_only": True,
        "executable_mode": "paper",
        "conditions": [
            {
                "id": 1,
                "name": "licensed_point_in_time_data",
                "status": "BLOCKED",
                "reason": "No entitled vendor feed is configured. Synthetic data is not a substitute.",
            },
            {
                "id": 2,
                "name": "authenticated_broker_reconciliation",
                "status": "BLOCKED",
                "reason": (
                    "No broker credentials. The paper ledger is the reference shape; "
                    "a live endpoint is refused."
                ),
            },
            {
                "id": 3,
                "name": "non_synthetic_forward_record",
                "status": "NOT_YET_COLLECTED",
                "reason": (
                    "forward_2026H2 is not yet collected. The 2025 holdout is spent. "
                    "The freeze is 2026-10-07; collection has not begun."
                ),
            },
            {
                "id": 4,
                "name": "venue_cost_liquidity_borrow_financing",
                "status": "BLOCKED",
                "reason": "No dated venue measurements. Modeled costs are not venue evidence.",
            },
            {
                "id": 5,
                "name": "signed_promotion_and_live_authorization",
                "status": "ABSENT",
                "reason": "No written live authorization. Promotion stays fail-closed.",
            },
        ],
    }


def format_readiness(report: Mapping[str, Any] | None = None) -> str:
    """Plain-text readiness for the chat shell. States the block, not a claim."""
    body = live_readiness_report() if report is None else report
    lines = [
        "Live orders: disabled",
        "Eligible for live capital: no",
        "Executable mode: paper session only (simulated fills, no broker connection)",
        "",
        "Conditions:",
    ]
    for condition in body["conditions"]:
        lines.append(
            f"{condition['id']}. {condition['name']} — {condition['status']}: {condition['reason']}"
        )
    lines.extend(
        [
            "",
            "would_promote_live=false",
            "live_pnl_claim=false",
            "research_only=true",
            "",
            "Paper session (shell):",
            "  dipcatcher trade session --state <path> --commands <jsonl>",
            "Ops: mark, submit, fill, cancel, kill, clear_kill, checkpoint",
            "Prices are caller-supplied. Short sales are refused.",
        ]
    )
    return "\n".join(lines)
