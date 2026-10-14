"""Operator live desk. Shadow by default. No live-performance claim.

The paper broker still refuses live endpoints. This package is the separate
operator surface: pre-trade risk, an order lifecycle, and a bridge. Shadow
transmits nothing. Loopback fills are SYNTHETIC. The drop folder writes order
tickets only when the desk is armed and ``DIP_LIVE_ARM=1``. Evidence
conditions stay blocked; ``live_pnl_claim`` is false.
"""

from quant_fund.live.arming import CONFIRM_DROP, ENV_ARM, Arming
from quant_fund.live.desk import LiveDesk, LiveDeskError

__all__ = [
    "CONFIRM_DROP",
    "ENV_ARM",
    "Arming",
    "LiveDesk",
    "LiveDeskError",
]
