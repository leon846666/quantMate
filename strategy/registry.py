"""Strategy registry front-end + ensemble voter.

Example:
    from strategy.registry import make_strategy
    strat = make_strategy('momentum', lookback=20, top_n=30)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# Trigger registration side-effects
from strategy import momentum, mean_reversion, lightgbm_strategy  # noqa: F401
from strategy.base import BaseStrategy, StrategyConfig, get_strategy, list_strategies


def make_strategy(name: str, **kwargs: Any) -> BaseStrategy:
    """Instantiate a registered strategy by name.

    Pulls `top_n`, `long_only`, `weight` out of kwargs into StrategyConfig
    and passes the rest straight to the strategy constructor.
    """
    cls = get_strategy(name)
    cfg_keys = {"top_n", "long_only", "weight"}
    cfg_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in cfg_keys}
    cfg = StrategyConfig(**cfg_kwargs) if cfg_kwargs else StrategyConfig()
    return cls(config=cfg, **kwargs) if kwargs else cls(config=cfg)


def ensemble_vote(targets: list[pd.DataFrame], top_n: int = 30) -> pd.DataFrame:
    """Take several strategies' target frames and combine by counting votes,
    then equal-weight the top-voted names per day.
    """
    if not targets:
        return pd.DataFrame(columns=["trade_date", "ts_code", "weight"])
    combined = pd.concat(targets, ignore_index=True)
    votes = (
        combined.groupby(["trade_date", "ts_code"])
                .size()
                .reset_index(name="votes")
    )
    out = []
    for day, g in votes.groupby("trade_date"):
        g = g.sort_values("votes", ascending=False).head(top_n)
        g = g.assign(weight=1.0 / len(g))
        out.append(g[["trade_date", "ts_code", "weight"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["trade_date", "ts_code", "weight"]
    )


__all__ = ["make_strategy", "ensemble_vote", "list_strategies"]
