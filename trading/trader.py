"""Live trading interface — broker-agnostic stub.

Wire this up to a real broker SDK (易盛 / QMT / 同花顺 / 雪球量化 / IB) later.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass
class LiveOrder:
    ts_code: str
    side: Literal["buy", "sell"]
    qty: int
    price: float | None = None   # None = market order
    order_type: Literal["limit", "market"] = "limit"


class BrokerAdapter(ABC):
    """Wrap this around whatever real broker SDK you end up using."""

    @abstractmethod
    def submit(self, order: LiveOrder) -> str: ...

    @abstractmethod
    def cancel(self, order_id: str) -> None: ...

    @abstractmethod
    def positions(self) -> dict[str, int]: ...

    @abstractmethod
    def cash(self) -> float: ...


class DummyBroker(BrokerAdapter):
    """In-memory broker for smoke testing the plumbing."""
    def __init__(self, cash: float = 1_000_000.0):
        self._cash = cash
        self._positions: dict[str, int] = {}
        self._orders: dict[str, LiveOrder] = {}
        self._next = 0

    def submit(self, order: LiveOrder) -> str:
        self._next += 1
        oid = f"DUMMY-{self._next}"
        self._orders[oid] = order
        return oid

    def cancel(self, order_id: str) -> None:
        self._orders.pop(order_id, None)

    def positions(self) -> dict[str, int]:
        return dict(self._positions)

    def cash(self) -> float:
        return self._cash


class RealTrader:
    """High-level orchestrator: translates target weights into broker orders."""

    def __init__(self, broker: BrokerAdapter):
        self.broker = broker

    def rebalance(self, target_weights: dict[str, float],
                  price_lookup: dict[str, float]) -> list[str]:
        """Translate weights to orders — this is a stub; real implementation
        should reuse the diff logic from backtest.engine._rebalance."""
        raise NotImplementedError("Hook up to backtest.engine._rebalance when going live.")
