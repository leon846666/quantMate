"""Portfolio state container used during a backtest run."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Portfolio:
    cash: float
    positions: dict = field(default_factory=dict)   # ts_code -> qty (shares)
    trade_log: list = field(default_factory=list)
    nav_history: list = field(default_factory=list)

    def buy(self, ts_code: str, qty: int, price: float,
            commission: float, trade_date: pd.Timestamp) -> None:
        cost = qty * price + commission
        self.cash -= cost
        self.positions[ts_code] = self.positions.get(ts_code, 0) + qty
        self.trade_log.append({
            "trade_date": trade_date, "ts_code": ts_code, "side": "buy",
            "qty": qty, "price": price, "commission": commission,
        })

    def sell(self, ts_code: str, qty: int, price: float,
             commission: float, trade_date: pd.Timestamp) -> None:
        proceeds = qty * price - commission
        self.cash += proceeds
        self.positions[ts_code] = self.positions.get(ts_code, 0) - qty
        if self.positions[ts_code] <= 0:
            self.positions.pop(ts_code, None)
        self.trade_log.append({
            "trade_date": trade_date, "ts_code": ts_code, "side": "sell",
            "qty": qty, "price": price, "commission": commission,
        })

    def mark_to_market(self, prices: dict[str, float]) -> float:
        mv = sum(qty * prices.get(code, 0.0) for code, qty in self.positions.items())
        return self.cash + mv

    def snapshot(self, trade_date: pd.Timestamp, prices: dict[str, float]) -> None:
        self.nav_history.append({
            "trade_date": trade_date,
            "cash": self.cash,
            "position_value": sum(
                qty * prices.get(code, 0.0) for code, qty in self.positions.items()
            ),
            "nav": self.mark_to_market(prices),
            "n_positions": len(self.positions),
        })

    def nav_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.nav_history).set_index("trade_date").sort_index()

    def trade_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.trade_log)
