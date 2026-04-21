"""Core data contracts — dataclasses only. No behaviour here."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Optional

import pandas as pd


@dataclass
class StockData:
    """One stock's metadata."""
    ts_code: str               # e.g. '600519.SH'
    symbol: str                # e.g. '600519'
    name: str
    industry: Optional[str] = None
    market: Optional[str] = None   # SH / SZ / BJ
    list_date: Optional[str] = None


@dataclass
class DailyBar:
    """Adjusted daily OHLCV bar."""
    ts_code: str
    trade_date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    vol: float                 # 成交量 (手)
    amount: float              # 成交额 (千元)
    adj_factor: float = 1.0    # 复权因子


@dataclass
class Signal:
    """A strategy's output for one (date, ts_code) pair."""
    trade_date: pd.Timestamp
    ts_code: str
    side: Literal["buy", "sell", "hold"]
    strength: float = 1.0      # 0..1
    strategy: str = ""         # 策略名
    reason: str = ""


@dataclass
class Order:
    trade_date: pd.Timestamp
    ts_code: str
    side: Literal["buy", "sell"]
    qty: int                   # 股数 (A 股 100 股一手)
    price: float
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Trade(Order):
    """Filled order."""
    filled_qty: int = 0
    filled_price: float = 0.0


@dataclass
class PortfolioSnapshot:
    trade_date: pd.Timestamp
    cash: float
    positions: dict = field(default_factory=dict)   # ts_code -> qty
    nav: float = 0.0
