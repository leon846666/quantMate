"""Strategy base class + registry.

A strategy consumes a panel of (trade_date, ts_code, features) and emits
a per-day target portfolio — a Series of target weights indexed by ts_code.
The backtest engine takes care of turning weights into trades.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Type, cast

import pandas as pd


# ------------------ registry ------------------

_STRATEGIES: Dict[str, Type["BaseStrategy"]] = {}


def register_strategy(name: str) -> Callable[[Type["BaseStrategy"]], Type["BaseStrategy"]]:
    def deco(cls: Type["BaseStrategy"]) -> Type["BaseStrategy"]:
        if name in _STRATEGIES:
            raise ValueError(f"Duplicate strategy name: {name}")
        _STRATEGIES[name] = cls
        cls.strategy_name = name  # type: ignore[attr-defined]
        return cls

    return deco


def get_strategy(name: str) -> Type["BaseStrategy"]:
    if name not in _STRATEGIES:
        raise KeyError(f"Unknown strategy {name}. Registered: {list(_STRATEGIES)}")
    return _STRATEGIES[name]


def list_strategies() -> list[str]:
    return sorted(_STRATEGIES.keys())


# ------------------ config / base class ------------------


@dataclass
class StrategyConfig:
    top_n: int = 30
    long_only: bool = True
    weight: str = "equal"  # equal | score
    extras: dict = field(default_factory=dict)


class BaseStrategy(ABC):
    """All strategies inherit from this."""

    strategy_name: str = "base"  # overwritten by @register_strategy

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()

    @abstractmethod
    def generate_targets(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return a long-form DataFrame with columns:

            ['trade_date', 'ts_code', 'weight']

        One row per (rebalance date, stock the strategy wants to hold).
        Weights across one date should sum to 1 (or be 0 -> flat).
        """


# ------------------ helpers shared by subclasses ------------------


def equal_weight_top_n(scores: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """`scores` has columns ['trade_date','ts_code','score'].
    Pick top N per date, give each equal weight."""
    out = []
    for day, g in scores.groupby("trade_date"):
        g = g.dropna(subset=["score"]).sort_values("score", ascending=False).head(top_n)
        if g.empty:
            continue
        g = g.assign(weight=1.0 / len(g))
        out.append(g[["trade_date", "ts_code", "weight"]])
    if out:
        return cast(pd.DataFrame, pd.concat(out, ignore_index=True))
    return pd.DataFrame(columns=["trade_date", "ts_code", "weight"])


def score_weight_top_n(scores: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Weight proportional to score (shifted to be non-negative)."""
    out = []
    for day, g in scores.groupby("trade_date"):
        g = g.dropna(subset=["score"]).sort_values("score", ascending=False).head(top_n)
        if g.empty:
            continue
        s = g["score"].values
        s = s - s.min() + 1e-6
        w = s / s.sum()
        g = g.assign(weight=w)
        out.append(g[["trade_date", "ts_code", "weight"]])
    if out:
        return cast(pd.DataFrame, pd.concat(out, ignore_index=True))
    return pd.DataFrame(columns=["trade_date", "ts_code", "weight"])
