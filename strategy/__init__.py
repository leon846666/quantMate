"""Strategy module: BaseStrategy + built-in strategies + registry."""
from strategy.base import BaseStrategy, StrategyConfig, register_strategy
from strategy.registry import make_strategy, ensemble_vote, list_strategies
from strategy.lightgbm_strategy import (
    LightGBMStrategy,
    LGBParams,
    train_lightgbm,
    save_booster,
    load_booster,
    assign_groups,
)
from strategy.composite_score_strategy import CompositeScoreStrategy
from strategy.volume_ranking_strategy import VolumeRankingStrategy

__all__ = [
    "BaseStrategy",
    "StrategyConfig",
    "register_strategy",
    "make_strategy",
    "ensemble_vote",
    "list_strategies",
    "LightGBMStrategy",
    "LGBParams",
    "train_lightgbm",
    "save_booster",
    "load_booster",
    "assign_groups",
    "CompositeScoreStrategy",
    "VolumeRankingStrategy",
]
