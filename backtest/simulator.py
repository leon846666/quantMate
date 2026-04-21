"""Trade execution simulator: applies slippage and commission."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExecutionConfig:
    commission_rate: float = 3e-4    # 万三 (both sides)
    slippage_bps: float = 5.0        # 0.05%
    min_commission: float = 5.0      # RMB minimum per trade (A-share common rule)


def apply_slippage(price: float, side: str, bps: float) -> float:
    """`side` in {'buy','sell'}. Buy pays higher, sell receives lower."""
    delta = price * bps / 1e4
    return price + delta if side == "buy" else price - delta


def apply_commission(
    notional: float,
    side: str,
    cfg: ExecutionConfig,
) -> float:
    comm = max(notional * cfg.commission_rate, cfg.min_commission)
    # Stamp duty on sell side: 0.1% (simplified — A-share reality is 1‰ on sell only)
    if side == "sell":
        comm += notional * 1e-3
    return comm
