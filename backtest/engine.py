"""Backtest engine — driven by a target-weight DataFrame from a strategy.

Workflow:
    1. Build a wide price matrix (trade_date x ts_code) from the panel.
    2. Walk forward through dates.
    3. On rebalance dates, read target weights; compute diff against current
       NAV allocation; execute the diff with slippage + commission.
    4. Mark-to-market every day, record NAV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from backtest.portfolio import Portfolio
from backtest.simulator import ExecutionConfig, apply_slippage, apply_commission
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class BacktestConfig:
    initial_cash: float = 1_000_000.0
    start_date: str = "2020-01-01"
    end_date: str = "2022-12-31"
    rebalance_freq: str = "W-FRI"        # pandas offset alias
    execution: ExecutionConfig = None    # type: ignore[assignment]

    def __post_init__(self):
        if self.execution is None:
            self.execution = ExecutionConfig()


def _wide_prices(panel: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    return panel.pivot(index="trade_date", columns="ts_code", values=price_col).sort_index()


def _rebalance_dates(all_dates: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """For each bucket in `freq`, pick the last trading date that actually exists."""
    grouper = pd.Series(all_dates, index=all_dates).resample(freq).last().dropna()
    return pd.DatetimeIndex(grouper.values).sort_values()


def run_backtest(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the backtest.

    Parameters
    ----------
    panel : long-form OHLCV panel (must include 'close').
    targets : DataFrame from a strategy — columns ['trade_date','ts_code','weight'].
    config : BacktestConfig.

    Returns
    -------
    (nav_df, trades_df)
    """
    price_matrix = _wide_prices(panel, "close")

    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)
    price_matrix = price_matrix.loc[(price_matrix.index >= start) & (price_matrix.index <= end)]
    if price_matrix.empty:
        raise ValueError("No prices in the backtest window.")

    all_dates = price_matrix.index
    rebalance_dates = _rebalance_dates(all_dates, config.rebalance_freq)
    rebalance_set = set(rebalance_dates)
    log.info("Backtest: %d days, %d rebalances", len(all_dates), len(rebalance_dates))

    targets = targets.copy()
    targets["trade_date"] = pd.to_datetime(targets["trade_date"])
    # Snap each target row to the nearest on-or-after trading date
    snap = {}
    for d in pd.DatetimeIndex(targets["trade_date"].unique()):
        loc = all_dates.searchsorted(d, side="left")
        if loc >= len(all_dates):
            continue
        snap[d] = all_dates[loc]
    targets["trade_date"] = targets["trade_date"].map(snap)
    targets = targets.dropna(subset=["trade_date"])

    targets_by_date = {d: g for d, g in targets.groupby("trade_date")}

    pf = Portfolio(cash=config.initial_cash)
    exec_cfg = config.execution
    last_prices: dict[str, float] = {}   # 持续维护最后已知价格，避免休市日归零

    for d in all_dates:
        prices_today = price_matrix.loc[d].dropna().to_dict()
        last_prices.update(prices_today)   # 只更新有报价的股票
        # 用最后已知价格做市值估算（持仓中无今日报价的股票用昨日价）
        prices_for_nav = {**last_prices, **prices_today}

        if d in rebalance_set:
            # 调仓只用今日有报价的股票
            _rebalance(pf, d, prices_today, targets_by_date.get(d), exec_cfg)

        pf.snapshot(d, prices_for_nav)

    nav_df = pf.nav_frame()
    trades_df = pf.trade_frame()
    return nav_df, trades_df


def _rebalance(pf: Portfolio, date: pd.Timestamp, prices: dict,
               target_frame: Optional[pd.DataFrame],
               exec_cfg: ExecutionConfig) -> None:
    """Execute trades to move portfolio to the target weights."""
    if target_frame is None or target_frame.empty:
        target_map: dict[str, float] = {}
    else:
        target_map = dict(zip(target_frame["ts_code"], target_frame["weight"]))

    nav = pf.mark_to_market(prices)

    # 1. Sell anything not in target (or whose target weight is 0)
    to_sell = [code for code in list(pf.positions) if target_map.get(code, 0.0) <= 0]
    for code in to_sell:
        if code not in prices:
            continue
        qty = pf.positions[code]
        px = apply_slippage(prices[code], "sell", exec_cfg.slippage_bps)
        comm = apply_commission(qty * px, "sell", exec_cfg)
        pf.sell(code, qty, px, comm, date)

    nav = pf.mark_to_market(prices)

    # 2. Compute target notional per stock, buy the diff (A-shares trade in lots of 100)
    for code, w in target_map.items():
        if code not in prices or w <= 0:
            continue
        target_notional = nav * w
        current_qty = pf.positions.get(code, 0)
        px = apply_slippage(prices[code], "buy", exec_cfg.slippage_bps)
        target_qty = int(target_notional // (px * 100)) * 100
        diff_qty = target_qty - current_qty
        if diff_qty > 0:
            cost = diff_qty * px
            if cost + apply_commission(cost, "buy", exec_cfg) > pf.cash:
                # Scale down if not enough cash
                diff_qty = int(pf.cash // (px * 100)) * 100
                if diff_qty <= 0:
                    continue
            comm = apply_commission(diff_qty * px, "buy", exec_cfg)
            pf.buy(code, diff_qty, px, comm, date)
        elif diff_qty < 0:
            sell_qty = -diff_qty
            comm = apply_commission(sell_qty * px, "sell", exec_cfg)
            pf.sell(code, sell_qty, px, comm, date)


def run_group_backtest(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    config: BacktestConfig,
    n_groups: int = 9,
) -> dict[int, pd.DataFrame]:
    """Run a separate equal-weighted backtest for each of N prediction groups.

    `predictions` must have columns ['trade_date','ts_code','score'].

    Returns {group_id: nav_df}.
    """
    from strategy.lightgbm_strategy import assign_groups
    preds = predictions.copy()
    preds["group"] = (
        preds.groupby("trade_date", group_keys=False)["score"]
             .transform(lambda s: assign_groups(s, n_groups))
    )

    out: dict[int, pd.DataFrame] = {}
    for g_id in range(1, n_groups + 1):
        members = preds[preds["group"] == g_id]
        # Equal weight within group
        targets = []
        for day, grp in members.groupby("trade_date"):
            if len(grp) == 0:
                continue
            w = 1.0 / len(grp)
            t = grp[["trade_date", "ts_code"]].copy()
            t["weight"] = w
            targets.append(t)
        if not targets:
            log.warning("Group %d has no members", g_id)
            continue
        tgt = pd.concat(targets, ignore_index=True)
        nav, _trades = run_backtest(panel, tgt, config)
        out[g_id] = nav
        log.info("Group %d: final NAV = %.2f", g_id, nav["nav"].iloc[-1] if not nav.empty else float('nan'))
    return out
