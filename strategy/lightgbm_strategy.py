"""LightGBM multi-factor strategy.

Pipeline (matches the "先 LightGBM, 再 GROUP9 -> TFT" plan):
    1. Train a LightGBM regressor on factor residuals → forward N-day return.
    2. At each rebalance date, predict for every stock, sort into N equal
       groups (default 9).  Long the top-group top-K, or use all groups for
       group-spread analysis.

The actual training code lives in this file so the strategy is self-contained.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from strategy.base import (
    BaseStrategy,
    StrategyConfig,
    register_strategy,
    equal_weight_top_n,
    score_weight_top_n,
)
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class LGBParams:
    objective: str = "regression"
    metric: str = "rmse"
    num_leaves: int = 63
    learning_rate: float = 0.05
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    min_data_in_leaf: int = 200
    num_boost_round: int = 500
    early_stopping_rounds: int = 50
    verbose: int = -1
    extras: dict = field(default_factory=dict)


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: Optional[pd.DataFrame] = None,
    y_valid: Optional[pd.Series] = None,
    params: Optional[LGBParams] = None,
):
    """Train a LightGBM regressor and return the booster."""
    import lightgbm as lgb

    params = params or LGBParams()
    p = {
        "objective": params.objective,
        "metric": params.metric,
        "num_leaves": params.num_leaves,
        "learning_rate": params.learning_rate,
        "feature_fraction": params.feature_fraction,
        "bagging_fraction": params.bagging_fraction,
        "bagging_freq": params.bagging_freq,
        "min_data_in_leaf": params.min_data_in_leaf,
        "verbose": params.verbose,
        **params.extras,
    }
    train_set = lgb.Dataset(X_train, y_train)
    valid_sets = [train_set]
    valid_names = ["train"]
    callbacks: list[Any] = [lgb.log_evaluation(period=50)]
    if X_valid is not None and y_valid is not None:
        valid_sets.append(lgb.Dataset(X_valid, y_valid, reference=train_set))
        valid_names.append("valid")
        callbacks.append(
            lgb.early_stopping(params.early_stopping_rounds, verbose=False)
        )

    booster = lgb.train(
        p,
        train_set,
        num_boost_round=params.num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return booster


def save_booster(booster, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(booster, f)


def load_booster(path: str | Path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def assign_groups(scores: pd.Series, n_groups: int = 9) -> pd.Series:
    """Within each cross-section, assign group 1..n_groups (1 = worst)."""
    return pd.qcut(scores.rank(method="first"), n_groups, labels=False) + 1


@register_strategy("lightgbm")
class LightGBMStrategy(BaseStrategy):
    """Ranks stocks by predicted forward return; longs the top bucket."""

    def __init__(
        self,
        booster,
        feature_cols: list[str],
        config: StrategyConfig | None = None,
        group_count: int = 9,
        top_group_only: bool = True,
    ):
        super().__init__(config)
        self.booster = booster
        self.feature_cols = feature_cols
        self.group_count = group_count
        self.top_group_only = top_group_only

    def predict(self, panel: pd.DataFrame) -> pd.DataFrame:
        X = panel[self.feature_cols].astype(float).fillna(0)
        panel = panel.copy()
        panel["score"] = self.booster.predict(X)
        panel["group"] = panel.groupby("trade_date", group_keys=False)[
            "score"
        ].transform(lambda s: assign_groups(s, self.group_count))
        return panel

    def generate_targets(self, panel: pd.DataFrame) -> pd.DataFrame:
        scored = self.predict(panel)
        if self.top_group_only:
            scored = scored[scored["group"] == self.group_count]
        scores = scored[["trade_date", "ts_code", "score"]]
        picker = (
            equal_weight_top_n if self.config.weight == "equal" else score_weight_top_n
        )
        return picker(scores, self.config.top_n)
