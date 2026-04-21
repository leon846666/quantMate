"""Cross-sectional momentum: sort by trailing N-day return, long the top."""
from __future__ import annotations

import pandas as pd

from strategy.base import (
    BaseStrategy,
    StrategyConfig,
    register_strategy,
    equal_weight_top_n,
    score_weight_top_n,
)


@register_strategy("momentum")
class MomentumStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig | None = None, lookback: int = 20):
        super().__init__(config)
        self.lookback = lookback

    def generate_targets(self, panel: pd.DataFrame) -> pd.DataFrame:
        df = panel.sort_values(["ts_code", "trade_date"]).copy()
        df["score"] = (
            df.groupby("ts_code", group_keys=False)["close"]
              .apply(lambda s: s.pct_change(self.lookback))
        )
        df = df.dropna(subset=["score"])
        scores = df[["trade_date", "ts_code", "score"]]
        picker = equal_weight_top_n if self.config.weight == "equal" else score_weight_top_n
        return picker(scores, self.config.top_n)
