"""Mean reversion on short-term z-score: long the most oversold."""
from __future__ import annotations

import pandas as pd

from strategy.base import (
    BaseStrategy,
    StrategyConfig,
    register_strategy,
    equal_weight_top_n,
)


@register_strategy("mean_reversion")
class MeanReversionStrategy(BaseStrategy):
    def __init__(
        self,
        config: StrategyConfig | None = None,
        lookback: int = 10,
        zscore_threshold: float = -1.5,
    ):
        super().__init__(config)
        self.lookback = lookback
        self.zscore_threshold = zscore_threshold

    def generate_targets(self, panel: pd.DataFrame) -> pd.DataFrame:
        df = panel.sort_values(["ts_code", "trade_date"]).copy()

        def _zscore(s: pd.Series) -> pd.Series:
            mu = s.rolling(self.lookback).mean()
            sd = s.rolling(self.lookback).std()
            return (s - mu) / sd

        df["zs"] = (
            df.groupby("ts_code", group_keys=False)["close"].apply(_zscore)
        )
        df["score"] = -df["zs"]          # more negative z → higher score
        df = df[df["zs"] <= self.zscore_threshold]
        if df.empty:
            return pd.DataFrame(columns=["trade_date", "ts_code", "weight"])

        scores = df[["trade_date", "ts_code", "score"]]
        return equal_weight_top_n(scores, self.config.top_n)
