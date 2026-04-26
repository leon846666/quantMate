"""strategy/composite_score_strategy.py — 多因子综合评分策略

所有技术指标通过 TA-Lib 计算（工业级，与 TradingView / Bloomberg 结果一致）。

信号权重表:
    RSI(14)      15%   <30→+0.8 <40→+0.3 | >60→-0.3 >70→-0.8
    5日动量      15%   ROC>10%→+0.5 >5%→+0.2 | <-5%→-0.2 <-10%→-0.5
    主力资金     15%   净流入>2亿→+0.4     | 净流出>2亿→-0.4
    MA均线       10%   多头排列→+0.3       | 空头排列→-0.3
    支撑阻力      5%   近20日低点→+0.6     | 近20日高点→-0.4
    成交量        5%   OBV上升→+0.2        | 放量下跌→-0.3

总评分 = Σ(信号评分 × 权重) / Σ有效权重
置信度 = |评分| × (1 - 子信号标准差)

买入: 评分 ≥ +0.15 → weight 按置信度加权归一化
卖出: 评分 ≤ -0.10 → weight = 0
"""

from __future__ import annotations

from typing import Any, Optional, cast

import numpy as np
import pandas as pd
import talib

from strategy.base import BaseStrategy, StrategyConfig, register_strategy
from utils.logger import get_logger

log = get_logger(__name__)

# ── moneyflow 全局缓存 ───────────────────────────────────────────────
_mf_cache: pd.DataFrame | None = None


def _load_moneyflow() -> pd.DataFrame:
    """从 PostgreSQL 加载 moneyflow，全局缓存一次。"""
    global _mf_cache
    if _mf_cache is not None:
        return _mf_cache
    try:
        from data.database import query

        df = query("SELECT ts_code, trade_date, net_mf_amount FROM moneyflow")
        if not df.empty:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
        _mf_cache = df
        return df
    except Exception as e:
        log.warning("moneyflow 读取失败: %s", e)
        _mf_cache = pd.DataFrame()
        return _mf_cache


# ── 信号权重 ─────────────────────────────────────────────────────────
SIGNAL_WEIGHTS: dict[str, float] = {
    "rsi": 0.15,
    "momentum5": 0.15,
    "moneyflow": 0.15,  # 缺失时自动跳过，权重归零
    "ma_align": 0.10,
    "support": 0.05,
    "volume": 0.05,
}

BUY_THRESHOLD = 0.15
SELL_THRESHOLD = -0.10


# ── 信号函数（全部调用 TA-Lib）───────────────────────────────────────


def _score_rsi(close: np.ndarray) -> np.ndarray:
    rsi = talib.RSI(close, timeperiod=14)
    s = np.zeros(len(close))
    s[rsi < 30] = 0.8
    s[(rsi >= 30) & (rsi < 40)] = 0.3
    s[(rsi > 60) & (rsi <= 70)] = -0.3
    s[rsi > 70] = -0.8
    return s


def _score_momentum5(close: np.ndarray) -> np.ndarray:
    # ROC = (close - close[n]) / close[n] * 100  (百分比)
    roc = talib.ROC(close, timeperiod=5)
    s = np.zeros(len(close))
    s[roc > 10] = 0.5
    s[(roc > 5) & (roc <= 10)] = 0.2
    s[(roc < -5) & (roc >= -10)] = -0.2
    s[roc < -10] = -0.5
    return s


def _score_ma(close: np.ndarray) -> np.ndarray:
    ma5 = talib.SMA(close, timeperiod=5)
    ma20 = talib.SMA(close, timeperiod=20)
    s = np.zeros(len(close))
    bull = (close > ma5) & (ma5 > ma20)
    bear = (close < ma5) & (ma5 < ma20)
    s[bull] = 0.3
    s[bear] = -0.3
    return s


def _score_support(close: np.ndarray, window: int = 20) -> np.ndarray:
    low20 = talib.MIN(close, timeperiod=window)
    high20 = talib.MAX(close, timeperiod=window)
    rng = high20 - low20
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = np.where(rng > 0, (close - low20) / rng, np.nan)
    s = np.zeros(len(close))
    s[pos < 0.15] = 0.6
    s[pos > 0.85] = -0.4
    return s


def _score_volume(close: np.ndarray, vol: np.ndarray) -> np.ndarray:
    # OBV 趋势：OBV 5日均线方向
    obv = talib.OBV(close, vol)
    obv_ma = talib.SMA(obv, timeperiod=5)
    # 放量下跌：成交量 > 20日均量 × 1.5 且价格下跌
    vol_ma20 = talib.SMA(vol, timeperiod=20)
    with np.errstate(invalid="ignore", divide="ignore"):
        vol_ratio = np.where(vol_ma20 > 0, vol / vol_ma20, np.nan)
    price_down = np.diff(close, prepend=close[0]) < 0
    s = np.zeros(len(close))
    # OBV 上升趋势 → 正向
    obv_rising = np.diff(obv_ma, prepend=obv_ma[0]) > 0
    s[obv_rising] = 0.2
    # 放量下跌 → 负向
    heavy_sell = (vol_ratio > 1.5) & price_down
    s[heavy_sell] = -0.3
    return s


def _score_moneyflow(mf: np.ndarray) -> np.ndarray:
    """主力净流入（元） → 信号分"""
    s = np.zeros(len(mf))
    s[mf > 2e8] = 0.4
    s[mf < -2e8] = -0.4
    return s


# ── 策略主体 ─────────────────────────────────────────────────────────


@register_strategy("composite_score")
class CompositeScoreStrategy(BaseStrategy):
    """多因子综合评分策略（TA-Lib 指标）"""

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        buy_threshold: float = BUY_THRESHOLD,
        sell_threshold: float = SELL_THRESHOLD,
    ):
        super().__init__(config)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def _score_one(self, grp: pd.DataFrame) -> pd.DataFrame:
        """单只股票逐日评分，返回带 composite / confidence 列的 DataFrame。"""
        close = grp["close"].to_numpy(dtype=float)
        vol = grp["vol"].to_numpy(dtype=float)
        n = len(close)

        raw: dict[str, np.ndarray] = {
            "rsi": _score_rsi(close),
            "momentum5": _score_momentum5(close),
            "ma_align": _score_ma(close),
            "support": _score_support(close),
            "volume": _score_volume(close, vol),
        }

        # 资金流向（有数据才加入）
        if "net_mf_amount" in grp.columns:
            mf_arr = grp["net_mf_amount"].to_numpy(dtype=float)
            if not np.all(np.isnan(mf_arr)):
                raw["moneyflow"] = _score_moneyflow(np.nan_to_num(mf_arr, nan=0.0))

        active_w = {k: SIGNAL_WEIGHTS[k] for k in raw}
        total_w = sum(active_w.values())

        score_mat = np.column_stack(list(raw.values()))  # (n, signals)
        weights = np.array([active_w[k] for k in raw])
        composite = score_mat @ weights / total_w  # weighted mean
        std_score = score_mat.std(axis=1)
        confidence = np.abs(composite) * (1 - np.clip(std_score, 0, 1))

        out = grp.copy()
        out["composite"] = composite
        out["confidence"] = confidence
        return out

    def generate_targets(self, panel: pd.DataFrame) -> pd.DataFrame:
        panel = panel.sort_values(["ts_code", "trade_date"]).copy()

        # merge 资金流向
        mf = _load_moneyflow()
        if not mf.empty:
            panel = panel.merge(
                mf[["ts_code", "trade_date", "net_mf_amount"]],
                on=["ts_code", "trade_date"],
                how="left",
            )
            covered = panel["net_mf_amount"].notna().mean()
            log.info("moneyflow 覆盖率: %.1f%%", covered * 100)
        else:
            panel["net_mf_amount"] = float("nan")

        scored = cast(
            pd.DataFrame,
            panel.groupby("ts_code", group_keys=False)
            .apply(self._score_one, include_groups=False)
            .reset_index(drop=True),
        )
        if "ts_code" not in scored.columns:
            scored["ts_code"] = panel["ts_code"].values

        scored = scored.dropna(subset=["composite"])

        targets: list[pd.DataFrame] = []
        for _day, day_df in scored.groupby("trade_date"):
            buys = day_df[day_df["composite"] >= self.buy_threshold].copy()
            if buys.empty:
                continue
            raw_w = (buys["composite"] * buys["confidence"].clip(0.1, 1.0)).clip(
                lower=0
            )
            if raw_w.sum() == 0:
                continue
            buys = buys.copy()
            buys["weight"] = raw_w / raw_w.sum()
            targets.append(
                buys[["trade_date", "ts_code", "weight"]].reset_index(drop=True)
            )

        if not targets:
            log.warning("composite_score: 无满足买入阈值(%.2f)的信号", self.buy_threshold)
            return pd.DataFrame(columns=["trade_date", "ts_code", "weight"])

        result = cast(pd.DataFrame, pd.concat(targets, ignore_index=True))
        log.info(
            "composite_score: %d 调仓日  %d 持仓记录  均 %.1f 只/日",
            result["trade_date"].nunique(),
            len(result),
            len(result) / result["trade_date"].nunique(),
        )
        return result
