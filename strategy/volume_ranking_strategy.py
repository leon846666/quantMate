"""strategy/volume_ranking_strategy.py — 成交量排名策略

策略逻辑:
    1. 每5天调仓一次
    2. 按昨日收盘成交量排名，选择前20只股票
    3. 等权重买入
    4. 股票池：沪深300成分股

调仓时间: 每5个交易日的9:30
持仓数量: 20只
权重分配: 等权重 (每只5%)
"""

from __future__ import annotations

from typing import Optional
import pandas as pd
import numpy as np

from strategy.base import BaseStrategy, StrategyConfig, register_strategy
from utils.logger import get_logger

log = get_logger(__name__)


@register_strategy("volume_ranking")
class VolumeRankingStrategy(BaseStrategy):
    """成交量排名策略"""

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        top_n: int = 20,
        rebalance_days: int = 5,
    ):
        super().__init__(config)
        self.top_n = top_n
        self.rebalance_days = rebalance_days
        self._last_rebalance_date = None

    def _should_rebalance(self, current_date: pd.Timestamp) -> bool:
        """判断是否需要调仓（每5个交易日）"""
        if self._last_rebalance_date is None:
            return True

        # 简单的交易日计数（实际应该用交易日历，这里用近似方法）
        days_diff = (current_date - self._last_rebalance_date).days
        # 假设5个自然日约等于3-4个交易日，这里设置为4天触发调仓
        return days_diff >= 4

    def generate_targets(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易目标

        Parameters
        ----------
        panel : pd.DataFrame
            面板数据，必须包含 ['trade_date', 'ts_code', 'close', 'vol']

        Returns
        -------
        pd.DataFrame
            目标权重，列: ['trade_date', 'ts_code', 'weight']
        """
        panel = panel.sort_values(["trade_date", "ts_code"]).copy()

        # 确保必要的列存在
        required_cols = ["trade_date", "ts_code", "close", "vol"]
        missing_cols = [col for col in required_cols if col not in panel.columns]
        if missing_cols:
            raise ValueError(f"缺少必要列: {missing_cols}")

        targets = []
        trade_dates = sorted(panel["trade_date"].unique())

        for current_date in trade_dates:
            # 判断是否需要调仓
            if not self._should_rebalance(current_date):
                continue

            # 获取当日数据
            daily_data = panel[panel["trade_date"] == current_date].copy()

            if daily_data.empty:
                continue

            # 过滤掉成交量为0或NaN的股票
            daily_data = daily_data[
                (daily_data["vol"] > 0) &
                (daily_data["vol"].notna()) &
                (daily_data["close"] > 0) &
                (daily_data["close"].notna())
            ].copy()

            if len(daily_data) < self.top_n:
                log.warning(
                    "日期 %s 可用股票数 %d 少于目标数量 %d",
                    current_date.date(), len(daily_data), self.top_n
                )

            # 按成交量降序排序，选择前top_n只
            daily_data = daily_data.sort_values("vol", ascending=False)
            selected = daily_data.head(self.top_n).copy()

            # 等权重分配
            weight_per_stock = 1.0 / len(selected)
            selected["weight"] = weight_per_stock

            # 添加到目标列表
            targets.append(selected[["trade_date", "ts_code", "weight"]])

            # 更新最后调仓日期
            self._last_rebalance_date = current_date

        if not targets:
            log.warning("volume_ranking: 没有生成任何交易信号")
            return pd.DataFrame(columns=["trade_date", "ts_code", "weight"])

        result = pd.concat(targets, ignore_index=True)

        log.info(
            "volume_ranking: %d 调仓日, %d 持仓记录, 平均 %.1f 只/日",
            result["trade_date"].nunique(),
            len(result),
            len(result) / result["trade_date"].nunique(),
        )

        return result